"""Trainer: Growing-NCA recipe — grow the target image from a seed,
sample pool for long-term persistence, damage for regeneration."""
import gc
import os
import random
import time
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ca_model import CAModel, MLPCAModel, quantize_visible, wgsl_gate_mask
from paths import checkpoints_root
from preview_utils import make_grid_axes
from progress import TrainingMonitor
from targets import load_target_content, make_seed, pad_to_canvas


def _mse(diff):
    return diff ** 2


def _l1(diff):
    return diff.abs()


def _huber(diff, delta=0.1):
    """Quadratic near zero, linear past delta - splits the difference between
    mse's smoothness and l1's robustness to the occasional bad rollout."""
    abs_diff = diff.abs()
    quadratic = 0.5 * diff ** 2 / delta
    linear = abs_diff - 0.5 * delta
    return torch.where(abs_diff <= delta, quadratic, linear)


# distance functions for the rgb/edge reconstruction terms, keyed by the
# loss_fn= name train() accepts - swap to experiment without touching the
# rest of the loss recipe (weighting, edge term, overflow/leak penalties)
LOSS_FUNCTIONS = {
    "mse": _mse,
    "l1": _l1,
    "huber": _huber,
}


def robustness_score(eval_results):
    """Collapse evaluate_robustness() results into one comparable number:
    mean mse tracks how faithfully the pattern grows back, max leak is the
    canary for worm eruptions on canvases bigger than the training grid.
    Shared by grid_search's leaderboard and train()'s best-weight selection
    so both rank kernels by the same deployment-condition metric.
    Returns (score, mean_mse, max_leak); lower score is better."""
    mean_mse = sum(r["mse"] for r in eval_results) / len(eval_results)
    max_leak = max(r["leak"] for r in eval_results)
    return mean_mse + max_leak, mean_mse, max_leak


class CATrainer:
    def __init__(self, img_path, img_size=48, channels=11, pool_size=None,
                 batch_size=8, lr=1e-3, delta=0.25, rule="tanh", margin=None,
                 fire_rate=0.5, fg_weight=3.0, mlp_hidden=128,
                 mlp_state_input=True, mlp_hidden2=None, device=None, checkpoint_dir=None,
                 amp=None, compile_step=False, cuda_graph=None, perception_init="structured",
                 output_init_std=1e-3):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if 'cuda' in str(self.device):
            # shapes are fixed for a whole run, so letting cudnn benchmark
            # conv algorithms is free speed; 'high' allows TF32 matmuls
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        # amp=None auto-enables bf16 autocast where the hardware supports it;
        # the rollout state itself stays f32 (see _rollout_chunk)
        self.amp = amp if amp is not None else (
            'cuda' in str(self.device)
            and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        # Small NCA grids are launch-bound: Python spends much longer issuing
        # their thousands of tiny kernels than the GPU spends executing them.
        # None selects the fast path automatically once the rollout dimensions
        # are known in train(). True forces a capture attempt and False is the
        # escape hatch for debugging or memory-constrained runs.
        self.cuda_graph = cuda_graph
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            checkpoints_root, f"{image_name}_{time.strftime('%Y%m%d-%H%M%S')}")
        self.size = img_size
        self.channels = channels
        # default pool scales with batch so each pool entry gets refreshed
        # every ~32 epochs regardless of batch size — a pool much larger than
        # the batch fills up with states rolled out by long-outdated weights
        self.pool_size = pool_size if pool_size is not None else 32 * batch_size
        self.batch_size = batch_size
        # Default margin scales with grid size so the target never touches
        # the training canvas border (~15% dead space per side).
        self.margin = margin if margin is not None else max(4, round(img_size * 0.15))

        # mlp_hidden > 0 adds a per-cell MLP between the 5x5 conv outputs and
        # the update rule; 0 falls back to the legacy conv-only kernel
        if mlp_hidden:
            self.model = MLPCAModel(channels=channels, hidden_dim=mlp_hidden,
                                    delta=delta, rule=rule, fire_rate=fire_rate,
                                    state_input=mlp_state_input,
                                    hidden_dim2=mlp_hidden2,
                                    perception_init=perception_init,
                                    output_init_std=output_init_std).to(self.device)
        else:
            self.model = CAModel(channels=channels, delta=delta, rule=rule,
                                 fire_rate=fire_rate,
                                 perception_init=perception_init).to(self.device)
        # NHWC/channels-last lets cuDNN use substantially faster tensor-core
        # convolution paths on large grids (about 1.8x at 150px on RTX 3070).
        # Tensor dimensions remain [N,C,H,W]; only their memory layout changes.
        self.memory_format = (torch.channels_last if 'cuda' in str(self.device)
                              else torch.contiguous_format)
        if self.memory_format == torch.channels_last:
            self.model.to(memory_format=self.memory_format)
        self.target_content = load_target_content(img_path, img_size - 2 * self.margin).to(self.device)
        self.target = pad_to_canvas(self.target_content, img_size).unsqueeze(0).to(self.device)
        self.seed = make_seed(channels, img_size).to(self.device).contiguous(
            memory_format=self.memory_format)
        self.pool = make_seed(channels, img_size, batch=self.pool_size).to(self.device).contiguous(
            memory_format=self.memory_format)
        self.dead_mask = self._make_dead_mask().to(self.device)

        if compile_step:
            # opt-in: inductor/Triton is flaky on Windows, so compile eagerly
            # against a throwaway seed and fall back rather than crash the run
            compiled = torch.compile(self.model.step)
            try:
                with torch.no_grad():
                    compiled(self.seed.clone())
                self.model.step = compiled
                print("torch.compile: model step compiled")
            except Exception as e:
                print(f"torch.compile failed ({e!r}); continuing uncompiled")

        # Plain MSE lets the trivially-easy black margin (about half the
        # canvas at the default margin) dilute the gradient on the actual
        # image, which reads as blur. Content pixels get (1 + fg_weight)
        # times the weight of margin pixels.
        content = torch.zeros(1, 1, img_size, img_size)
        content[:, :, self.margin:img_size - self.margin, self.margin:img_size - self.margin] = 1.0
        self.pixel_weight = (1.0 + fg_weight * content).to(self.device)

        # Depthwise Sobel filters over RGB: matching image gradients, not
        # just pixel values, is the cheap counter to MSE's tendency to
        # average toward blur.
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.sobel = torch.stack([sx, sx.t()]).unsqueeze(1).repeat(3, 1, 1, 1).to(self.device)
        self.target_edges = self._edges(self.target)
        self.target_fft = torch.fft.fft2(self.target, norm="ortho")

        # coordinate grid for cutting circular damage holes
        ys, xs = torch.meshgrid(torch.arange(img_size, dtype=torch.float32),
                                torch.arange(img_size, dtype=torch.float32), indexing="ij")
        self.grid_y = ys.to(self.device)
        self.grid_x = xs.to(self.device)

        self.base_lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.95))
        self.scheduler = None  # created in train() once epochs is known
        self.last_eval = None  # latest robustness-probe score (see _eval_and_save_best)
        self._gate_bases = {}

    def _make_dead_mask(self, halo=6):
        """1.0 in the margin far from the target content, 0.0 over the content
        plus a small halo. Hidden channels get penalized in this dead zone:
        the RGB loss can't see hidden activity leaking into the margin, and on
        a canvas larger than the training grid that leak keeps propagating and
        erupts into visible wormy structures. The halo leaves room for the
        hidden boundary signals the pattern legitimately needs at its edge."""
        mask = torch.ones(1, 1, self.size, self.size)
        lo = max(0, self.margin - halo)
        hi = min(self.size, self.size - self.margin + halo)
        mask[:, :, lo:hi, lo:hi] = 0.0
        return mask

    def _edges(self, rgb):
        return F.conv2d(rgb, self.sobel, padding=1, groups=3)

    def _fft_loss(self, rgb):
        """Focal frequency loss (Jiang et al. 2021): squared spectrum error,
        re-weighted by its own detached, per-image-normalized magnitude so the
        most-wrong frequencies dominate. The weighting is the whole point — by
        Parseval an unweighted spectrum MSE IS pixel MSE. Where MSE spends its
        gradient on broad low-frequency agreement, this term concentrates on
        whichever frequencies remain worst, which late in training is the
        high-frequency detail that reads as sharpness."""
        diff = torch.fft.fft2(rgb, norm="ortho") - self.target_fft
        err = diff.real ** 2 + diff.imag ** 2
        w = err.sqrt().detach()
        w = w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return (w * err).mean()

    def _damage_sample(self, sample):
        """Zero all channels in a random circle over the content area. The
        classic regeneration trick: the best-grown pool samples get holes cut
        so the kernel learns to regrow — which is exactly what a user
        scribbling over the pattern in the browser demands of it."""
        r = random.uniform(0.08, 0.25) * self.size
        cx = random.uniform(self.margin, self.size - self.margin)
        cy = random.uniform(self.margin, self.size - self.margin)
        hole = (self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2 <= r * r
        sample[:, hole] = 0.0

    def _gate_base(self, width, height):
        """Cache the spatial half of the browser gate hash."""
        key = (width, height, str(self.device))
        base = self._gate_bases.get(key)
        if base is None:
            xs = torch.arange(width, dtype=torch.int64, device=self.device).view(1, -1)
            ys = torch.arange(height, dtype=torch.int64, device=self.device).view(-1, 1)
            base = (xs * 374761393 + ys * 668265263) & 0xFFFFFFFF
            self._gate_bases[key] = base
        return base

    def _gate_mask(self, width, height, timestep):
        """wgsl_gate_mask(), using a cached spatial hash base."""
        if self.model.fire_rate >= 1.0:
            return None
        h = (self._gate_base(width, height) + int(timestep) * 2246822519) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        gate = h.to(torch.float32) / 4294967295.0
        return (gate < self.model.fire_rate).view(1, 1, height, width).to(torch.float32)

    def _rollout_chunk(self, x, steps, gate_t0=None, step_base=0):
        """One backprop segment of a rollout: step, then round the visible
        channels like the browser's rgba8unorm texture does. Kept small so it
        can be gradient-checkpointed — long rollouts recompute activations on
        the backward pass instead of storing every step.

        gate_t0 (one timestep offset per batch sample) switches the update
        gate from Bernoulli to the browser's deterministic hash, advancing by
        step_base + i like the shader's frame counter; per-sample offsets keep
        the mask diversity Bernoulli gave within a batch, and determinism
        makes the checkpointed backward recompute exact. None keeps the
        model's Bernoulli fallback.

        With amp on, only the model step runs in bf16 — quantize_visible
        needs f32 (round(x*255) overruns bf16's mantissa), so the carried
        state is cast back every step and the loss stays f32 too."""
        overflow = x.new_zeros(())
        size = x.shape[-1]
        for i in range(steps):
            mask = None
            if gate_t0 is not None:
                mask = torch.cat([
                    self._gate_mask(size, size, int(t) + step_base + i)
                    for t in gate_t0])
            if self.amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    x, ov = self.model.step(x, update_mask=mask)
            else:
                x, ov = self.model.step(x, update_mask=mask)
            x = quantize_visible(x.float())
            overflow = overflow + ov.float()
        return x, overflow

    def _rollout_graph(self, x, requested_steps, max_steps):
        """Fixed-shape rollout for CUDA Graph capture.

        The graph contains ``max_steps`` calls. A device scalar makes calls
        after ``requested_steps`` no-ops for state, overflow, and gradients,
        retaining random rollout lengths without maintaining many graphs.
        """
        overflow = x.new_zeros(())
        for i in range(max_steps):
            if self.amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    candidate, ov = self.model.step(x)
            else:
                candidate, ov = self.model.step(x)
            candidate = quantize_visible(candidate.float())
            active = requested_steps > i
            x = torch.where(active, candidate, x)
            overflow = overflow + ov.float() * active.to(ov.dtype)
        return x, overflow

    def _make_train_graph(self, max_steps, target_batch, target_edges,
                          dead_cells, has_dead_zone, edge_weight, fft_weight,
                          overflow_weight, leak_weight, distance):
        """Capture rollout, loss, backward, and gradient normalization.

        Adam remains outside the graph, preserving optimizer checkpoint and
        LR-scheduler behavior while removing the dominant launch overhead.
        """
        # Valid initial values keep warmup/capture losses finite even though
        # callers overwrite this buffer before every replay.
        static_x = self.seed.expand(self.batch_size, -1, -1, -1).clone()
        static_steps = torch.full((), max_steps, dtype=torch.int64, device=self.device)
        outputs = None

        def body():
            nonlocal outputs
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            x, overflow_total = self._rollout_graph(static_x, static_steps, max_steps)
            loss_parts = self._compute_loss(
                x, target_batch, target_edges, dead_cells, has_dead_zone,
                overflow_total, static_steps, edge_weight, fft_weight,
                overflow_weight, leak_weight, distance)
            loss_parts[0].backward()
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.div_(p.grad.norm() + 1e-8)
            outputs = (x, overflow_total, *loss_parts)

        # Warm up on the current stream. In particular, this creates each
        # parameter's AccumulateGrad node on the same stream lineage used by
        # capture; warming on a separate stream can make the legacy stream
        # depend on a blocking capture stream during backward.
        for _ in range(2):
            body()
        torch.cuda.synchronize()
        # ``outputs`` contains tensors with grad_fn links into the last warmup
        # graph. Keeping that graph alive while capture starts causes its old
        # AccumulateGrad nodes to introduce an illegal cross-stream dependency.
        outputs = None
        gc.collect()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        assert outputs is not None
        return graph, static_x, static_steps, cast(tuple, outputs)

    def _cuda_graph_fits(self, max_steps, target_batch, target_edges,
                         dead_cells, has_dead_zone, edge_weight, fft_weight,
                         overflow_weight, leak_weight, distance):
        """Measure this exact workload's activation-memory slope and decide
        whether a full max-length CUDA Graph fits the currently free VRAM.

        A formula based only on H/W/channels misses cuDNN workspaces, AMP,
        architecture details, allocator behavior, and other GPU users. Two
        short real forward/backward samples include all of those. Activation
        memory is approximately linear in rollout length, so extrapolating
        their slope is both cheap and much more portable across systems.
        """
        sample_steps = (2, 6)
        measurements = []
        x = self.seed.expand(self.batch_size, -1, -1, -1).clone()
        try:
            for steps in sample_steps:
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                baseline = torch.cuda.memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                y, overflow = self._rollout_chunk(x, steps)
                losses = self._compute_loss(
                    y, target_batch, target_edges, dead_cells, has_dead_zone,
                    overflow, steps, edge_weight, fft_weight, overflow_weight,
                    leak_weight, distance)
                losses[0].backward()
                torch.cuda.synchronize(self.device)
                extra = torch.cuda.max_memory_allocated(self.device) - baseline
                measurements.append((steps, extra))
                del y, overflow, losses

            (s0, m0), (s1, m1) = measurements
            per_step = max(0.0, (m1 - m0) / (s1 - s0))
            fixed = max(0.0, m0 - per_step * s0)
            # Capture has private-pool bookkeeping and can choose slightly
            # different cuDNN workspaces, so pad the observed eager estimate.
            estimated = int((fixed + per_step * max_steps) * 1.25)
        except (torch.OutOfMemoryError, RuntimeError) as e:
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False, f"probe failed: {type(e).__name__}"
        finally:
            del x
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()

        free, total = torch.cuda.mem_get_info(self.device)
        # Leave room for the persistent pool, Adam updates, robustness probes,
        # display use, and allocator fragmentation after capture.
        reserve = max(1 << 30, int(total * 0.15))
        available = max(0, free - reserve)
        fits = estimated <= available
        gib = 1 << 30
        detail = (f"estimated {estimated / gib:.2f}GB, "
                  f"available {available / gib:.2f}GB after reserve")
        return fits, detail

    def _compute_loss(self, x, target_batch, target_edges, dead_cells, has_dead_zone,
                      overflow_total, steps, edge_weight, fft_weight, overflow_weight,
                      leak_weight, distance):
        """Reconstruction (rgb + Sobel edges, via `distance`; focal frequency
        term when fft_weight > 0) plus the overflow and hidden-leak penalties.
        Returns (loss, rgb_loss, edge_loss, fft_loss, hidden_leak) so the
        caller can log the components alongside the combined total."""
        if has_dead_zone:
            hidden_leak = (x[:, 3:] ** 2 * self.dead_mask).sum() \
                / (dead_cells * (self.channels - 3) * x.shape[0])
        else:
            hidden_leak = x.new_zeros(())

        weight_norm = self.pixel_weight.sum() * x.shape[0]
        rgb_loss = (distance(x[:, :3] - target_batch) * self.pixel_weight).sum() / (weight_norm * 3)
        edge_loss = (distance(self._edges(x[:, :3]) - target_edges) * self.pixel_weight).sum() / (weight_norm * 6)
        fft_loss = self._fft_loss(x[:, :3]) if fft_weight else x.new_zeros(())

        loss = rgb_loss \
            + edge_weight * edge_loss \
            + fft_weight * fft_loss \
            + overflow_weight * overflow_total / steps \
            + leak_weight * hidden_leak

        return loss, rgb_loss, edge_loss, fft_loss, hidden_leak

    def _eval_and_save_best(self, monitor=None):
        """Probe growth under deployment conditions (bigger canvas, 1-3px
        seed dabs) and keep the weights whenever the score improves. The
        training loss is a noisy proxy — rollout length is random and it
        never sees a canvas bigger than the training grid — so best-weight
        selection keys off this instead."""
        results = self.evaluate_robustness(steps=3 * self.size, extra_sizes=(0, 64),
                                           radii=(0, 1), verbose=False)
        score, mean_mse, max_leak = robustness_score(results)
        self.model.saveBest(score)
        self.last_eval = score
        message = (f"eval: score={score:.4f} (mse={mean_mse:.4f}, leak={max_leak:.4f}), "
                   f"best={self.model.bestEval:.4f}")
        if monitor is not None:
            monitor.note(message)
        else:
            print(message)
        return score

    def save_checkpoint(self, epoch, loss, resume_state=True):
        """loss is recorded as the checkpoint's headline score — the latest
        robustness-probe score when one has run, else the smoothed training
        loss. resume_state also writes/overwrites resume_state.pt (pool +
        best-eval bookkeeping) beside the .pt files; it's a rolling side file
        because the pool is far bigger than the weights (grid_search skips it)."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        stem = f"epoch_{epoch:06d}"
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            # lets checkpoint_preview.py regrow this pattern without the
            # caller having to know how it was trained
            "config": {"size": self.size, "channels": self.channels,
                       "rule": self.model.rule, "fire_rate": self.model.fire_rate,
                       "mlp_hidden": getattr(self.model, "hidden_dim", 0),
                       "mlp_hidden2": getattr(self.model, "hidden_dim2", 0),
                       "mlp_state_input": getattr(self.model, "state_input", False),
                       "perception_init": getattr(self.model, "perception_init", None),
                       "output_init_std": getattr(self.model, "output_init_std", None)},
        }, os.path.join(self.checkpoint_dir, f"{stem}.pt"))

        if resume_state:
            torch.save({
                "epoch": epoch,
                "pool": self.pool,
                "bestEval": self.model.bestEval,
                "best_weights": getattr(self.model, "best_weights", None),
            }, os.path.join(self.checkpoint_dir, "resume_state.pt"))

        self.model.exportToPlaygroundFormat(self.checkpoint_dir, filepath=f"{stem}.json")

    def load_checkpoint(self, path):
        """Restore model + optimizer state from a saved checkpoint so training
        can continue from where it left off. If a resume_state.pt sits beside
        the checkpoint (written by save_checkpoint) and matches this config,
        the matured sample pool and best-eval bookkeeping are restored too —
        otherwise they restart fresh (best-eval seeded with the checkpoint's
        own loss/weights), which costs the resumed run the epochs it takes to
        re-mature the pool. Returns the epoch it was saved at, so the caller
        can resume epoch numbering from there."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.model.bestEval = ckpt["loss"]
        self.model.best_weights = {k: v.clone() for k, v in ckpt["model_state"].items()}

        state_path = os.path.join(os.path.dirname(path), "resume_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            if state["pool"].shape == self.pool.shape:
                self.pool = state["pool"].to(self.device).contiguous(
                    memory_format=self.memory_format)
                if state.get("bestEval") is not None:
                    self.model.bestEval = state["bestEval"]
                    if state.get("best_weights") is not None:
                        self.model.best_weights = {k: v.clone() for k, v in state["best_weights"].items()}
                print(f"Resumed pool + best-eval bookkeeping from {state_path}")
            else:
                print(f"resume_state.pt pool shape {tuple(state['pool'].shape)} doesn't match "
                      f"this config's {tuple(self.pool.shape)}; starting with a fresh pool")
        return ckpt["epoch"]

    def train(self, epochs=4000, min_steps=None, max_steps=None,
              overflow_weight=1, leak_weight=1, edge_weight=1.0, fft_weight=2.0,
              damage_n=1, grad_ckpt_steps=64, print_every=5, checkpoint_every=1000,
              catch_interrupt=True, label=None, loss_fn="mse", start_epoch=0,
              eval_every=500, gate="bernoulli", compile_rollout=False,
              bptt_steps=None):
        """catch_interrupt=True swallows Ctrl+C and keeps the best weights
        (interactive use); grid_search passes False so one Ctrl+C can abort
        the whole sweep instead of silently skipping to the next config.
        loss_fn selects the rgb/edge reconstruction distance - one of
        LOSS_FUNCTIONS ("mse", "l1", "huber"). fft_weight scales the focal
        frequency term (see _fft_loss; 0 disables it). start_epoch resumes numbering
        after load_checkpoint(); the LR schedule always decays fresh, from
        whatever rate the optimizer currently has, over the epochs remaining
        in this call (epochs - start_epoch) — so a new stage with a new lr
        gets its own clean cosine decay rather than continuing the previous
        stage's curve. eval_every runs the robustness probe that selects the
        best weights (0 defers it to a single probe at the end). gate picks
        the stochastic-update mask (moot at fire_rate 1.0): "bernoulli"
        (default) samples a fresh random mask each step — statistically what
        the browser's hash gate does, without rebuilding the WGSL mask every
        step; "hash" replays the browser's exact deterministic gate for
        parity-sensitive A/Bs. The robustness eval always uses the hash gate,
        so deployment behavior is still what gets scored."""
        distance = LOSS_FUNCTIONS[loss_fn]
        use_hash_gate = gate == "hash" and self.model.fire_rate < 1.0
        # Information crosses ~2px per step (5x5 kernel), scaled by the fire
        # rate, so the step budget must grow with the canvas and stretch as
        # the fire rate drops: too few steps and the far side of the pattern
        # can never even be reached, let alone refined. (2 - fire_rate) gives
        # 1x the canvas at fire_rate 1 and 1.5x at 0.5 — arrival time doubles
        # there, but the refinement share of the rollout doesn't need to.
        if min_steps is None:
            min_steps = round(self.size * (2 - self.model.fire_rate))
        if max_steps is None:
            max_steps = 2 * min_steps
        damage_n = min(damage_n, self.batch_size - 1)
        if bptt_steps is not None and bptt_steps <= 0:
            raise ValueError("bptt_steps must be positive or None")

        # Compile one fixed-size rollout segment instead of model.step. This
        # lets Inductor/Triton fuse elementwise work across recurrent steps.
        # Internal cudagraphs are disabled because checkpointing invokes the
        # same compiled callable repeatedly while earlier outputs remain live.
        compiled_chunk = None
        compile_detail = "off"
        compile_segment = grad_ckpt_steps or (bptt_steps or 16)
        if compile_rollout and 'cuda' in str(self.device):
            try:
                compiled_chunk = torch.compile(
                    self._rollout_chunk, fullgraph=True,
                    options={"triton.cudagraphs": False})
                compile_detail = f"on ({compile_segment}-step segments)"
            except Exception as e:
                compile_detail = f"failed: {type(e).__name__}"
                print(f"Rollout compilation failed ({e!r}); continuing eagerly")

        # Adam's step size follows lr, not gradient magnitude, so once the
        # cosine bottoms out the weights keep taking eta_min-sized steps
        # forever — a 0.1x floor left the whole second half of a run jittering
        # around the attractor instead of settling into it, which reads as
        # blur. 0.001x is small enough to actually converge.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs - start_epoch), eta_min=self.base_lr * 0.001)
        target_batch = self.target.expand(self.batch_size, -1, -1, -1)
        target_edges = self.target_edges.expand(self.batch_size, -1, -1, -1)
        epoch = start_epoch
        metric_names = ("loss", "rgb", "edge", "fft", "overflow", "leak")
        # Keep metrics device-side between reports. The former six .item()
        # calls every epoch synchronized graph replay with the CPU and erased
        # much of the launch-overhead win.
        metrics_acc = torch.zeros(len(metric_names), device=self.device)
        metrics_count = 0
        running_values = torch.zeros(50, device=self.device)
        running_count = 0
        running_pos = 0


        # the dead zone is empty when the margin is smaller than the halo
        # (tiny grids) or there are no hidden channels — dividing by its
        # zero cell count would poison the loss with NaN
        dead_cells = self.dead_mask.sum().item()
        has_dead_zone = dead_cells > 0 and self.channels > 3

        title = label or os.path.basename(self.checkpoint_dir)
        monitor = TrainingMonitor(epochs, title=title)

        # A full graph retains the maximum rollout's activation storage. In
        # auto mode, measure this exact model/GPU workload and capture whenever
        # it fits rather than relying on a card- or architecture-specific limit.
        graph_detail = "disabled"
        if (self.cuda_graph and 'cuda' in str(self.device)
                and not use_hash_gate and bptt_steps is None and compiled_chunk is None):
            use_cuda_graph, graph_detail = self._cuda_graph_fits(
                max_steps, target_batch, target_edges, dead_cells, has_dead_zone,
                edge_weight, fft_weight, overflow_weight, leak_weight, distance)
        else:
            use_cuda_graph = bool(self.cuda_graph) and compiled_chunk is None
            if use_cuda_graph:
                graph_detail = "forced"
        graph_state = None
        if use_cuda_graph and not use_hash_gate:
            try:
                graph_state = self._make_train_graph(
                    max_steps, target_batch, target_edges, dead_cells, has_dead_zone,
                    edge_weight, fft_weight, overflow_weight, leak_weight, distance)
            except Exception as e:
                graph_state = None
                torch.cuda.empty_cache()
                graph_detail = f"capture failed: {type(e).__name__}"
                print(f"CUDA Graph capture failed ({e!r}); continuing eagerly")

        interrupted = False
        with monitor:
            monitor.note(
                f"Rollout {min_steps}-{max_steps} steps/epoch, fire_rate={self.model.fire_rate}, "
                f"gate={'browser hash' if use_hash_gate else 'bernoulli' if self.model.fire_rate < 1.0 else 'off'}, "
                f"amp={'bf16' if self.amp else 'off'}, "
                f"cuda_graph={'on' if graph_state is not None else 'off'} ({graph_detail}), "
                f"compile={compile_detail}, "
                f"bptt={'full' if bptt_steps is None else f'last {bptt_steps} steps'}, "
                f"layout={'channels_last' if self.memory_format == torch.channels_last else 'contiguous'}, "
                f"edge_weight={edge_weight}, fft_weight={fft_weight}, damage_n={damage_n}, "
                f"grad checkpoint segment={grad_ckpt_steps if grad_ckpt_steps else 'off'}, "
                f"leak zone={'active' if has_dead_zone else 'EMPTY (margin too small, leak penalty off)'}")
            try:
                for epoch in range(start_epoch, epochs):
                    idx = torch.randint(0, self.pool_size, (self.batch_size,), device=self.device)
                    x = self.pool[idx]

                    # reseed the worst-looking sample so growth from seed keeps training;
                    # randomize the seed's radius and add a touch of value noise so the
                    # model learns to grow correctly from an imprecise brush dab, not
                    # just a perfect pixel
                    with torch.no_grad():
                        per_sample = ((x[:, :3] - target_batch) ** 2).mean(dim=(1, 2, 3))
                    # cut holes in the best-grown samples so regrowth keeps training
                    if damage_n:
                        damage_idx = per_sample.topk(damage_n, largest=False).indices
                        radii = (0.08 + 0.17 * torch.rand(damage_n, 1, 1, device=self.device)) * self.size
                        cxs = self.margin + (self.size - 2 * self.margin) * torch.rand(
                            damage_n, 1, 1, device=self.device)
                        cys = self.margin + (self.size - 2 * self.margin) * torch.rand(
                            damage_n, 1, 1, device=self.device)
                        holes = ((self.grid_x - cxs) ** 2 + (self.grid_y - cys) ** 2
                                 <= radii ** 2)
                        x[damage_idx] *= (~holes).unsqueeze(1)

                    radius = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
                    noise = random.uniform(0.0, 0.05)
                    reseed = torch.zeros_like(x[:1])
                    c = self.size // 2
                    reseed[:, :, c - radius:c + radius + 1,
                           c - radius:c + radius + 1] = 1.0
                    if noise:
                        reseed.add_(noise * torch.randn_like(reseed)).clamp_(0.0, 1.0)
                    worst = per_sample.argmax()
                    x[worst] = reseed[0]

                    # random per-sample offsets into the hash gate's timestep
                    # sequence, so every epoch sees a different stretch of the
                    # exact update pattern the browser will replay
                    gate_t0 = torch.randint(0, 2 ** 30, (self.batch_size,)) if use_hash_gate else None
                    steps = random.randint(min_steps, max_steps)
                    if graph_state is not None:
                        graph, static_x, static_steps, graph_outputs = graph_state
                        static_x.copy_(x)
                        static_steps.fill_(steps)
                        graph.replay()
                        (x, overflow_total, loss, rgb_loss, edge_loss,
                         fft_loss, hidden_leak) = graph_outputs
                    else:
                        overflow_total = x.new_zeros(())

                        def run_segments(state, count, base, track_grad):
                            segment_overflow = state.new_zeros(())
                            remaining = count
                            while remaining > 0:
                                # Only call the compiled function at its fixed
                                # specialization. A short final remainder stays
                                # eager, avoiding up to N recompilations for
                                # random rollout lengths.
                                if compiled_chunk is not None and remaining >= compile_segment:
                                    seg = compile_segment
                                    rollout_fn = compiled_chunk
                                else:
                                    seg = (min(grad_ckpt_steps, remaining)
                                           if grad_ckpt_steps else remaining)
                                    rollout_fn = self._rollout_chunk
                                if track_grad and grad_ckpt_steps:
                                    state, ov = grad_checkpoint(
                                        rollout_fn, state, seg, gate_t0, base,
                                        use_reentrant=False)
                                else:
                                    state, ov = rollout_fn(state, seg, gate_t0, base)
                                segment_overflow = segment_overflow + ov
                                remaining -= seg
                                base += seg
                            return state, segment_overflow

                        if bptt_steps is not None and steps > bptt_steps:
                            prefix = steps - bptt_steps
                            with torch.no_grad():
                                x, prefix_overflow = run_segments(x, prefix, 0, False)
                            x, grad_overflow = run_segments(x, bptt_steps, prefix, True)
                            overflow_total = prefix_overflow + grad_overflow
                        else:
                            x, overflow_total = run_segments(x, steps, 0, True)

                        loss, rgb_loss, edge_loss, fft_loss, hidden_leak = self._compute_loss(
                            x, target_batch, target_edges, dead_cells, has_dead_zone,
                            overflow_total, steps, edge_weight, fft_weight, overflow_weight,
                            leak_weight, distance)

                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.div_(p.grad.norm() + 1e-8)
                    self.optimizer.step()
                    self.scheduler.step()

                    self.pool[idx] = x.detach()

                    detached_metrics = torch.stack((
                        loss.detach(), rgb_loss.detach(), edge_loss.detach(),
                        fft_loss.detach(), (overflow_total / steps).detach(),
                        hidden_leak.detach()))
                    metrics_acc.add_(detached_metrics)
                    metrics_count += 1
                    running_values[running_pos].copy_(loss.detach())
                    running_pos = (running_pos + 1) % len(running_values)
                    running_count = min(running_count + 1, len(running_values))

                    if epoch % print_every == 0:
                        lr_now = self.optimizer.param_groups[0]['lr']
                        avg_values = (metrics_acc / metrics_count).cpu().tolist()
                        avg = dict(zip(metric_names, avg_values))
                        smooth = running_values[:running_count].mean().item()
                        monitor.update(epoch, loss=avg["loss"], smooth=smooth,
                                      rgb=avg["rgb"], edge=avg["edge"],
                                      fft=avg["fft"],
                                      overflow=avg["overflow"],
                                      leak=avg["leak"],
                                      lr=lr_now, best=self.model.bestEval)
                        metrics_acc.zero_()
                        metrics_count = 0


                    if eval_every and epoch > start_epoch and epoch % eval_every == 0:
                        self._eval_and_save_best(monitor)

                    if checkpoint_every and epoch > 0 and epoch % checkpoint_every == 0:
                        checkpoint_score = (self.last_eval if self.last_eval is not None
                                            else running_values[:running_count].mean().item())
                        self.save_checkpoint(epoch, checkpoint_score)
                        monitor.note(f"Checkpoint saved: {self.checkpoint_dir}/epoch_{epoch:06d}.pt")

            except KeyboardInterrupt:
                interrupted = True
                monitor.note("Training interrupted. Keeping best weights.")

        # final probe so the last weights get a fair shot at best before the
        # closing checkpoint records the score
        self._eval_and_save_best()
        self.save_checkpoint(epoch, self.last_eval, resume_state=bool(checkpoint_every))
        print(f"Final checkpoint saved: {self.checkpoint_dir}/epoch_{epoch:06d}.pt")
        if interrupted and not catch_interrupt:
            raise KeyboardInterrupt

    def evaluate_robustness(self, steps=600, extra_sizes=(0, 64, 192), radii=(0, 1, 2),
                            verbose=True):
        """Grow the pattern under conditions the browser will actually produce
        — canvases bigger than training, and seed dabs from a 1px pixel up to
        a few px wide — quantizing visible channels every step like the
        rgba8unorm texture does and gating updates with the same hash the
        exported shader uses (timestep = step index, i.e. a fresh canvas).
        Catches a fragile kernel before you export and deploy it, instead of
        finding out on the website. Returns a list of
        {canvas, seed_radius, mse, leak} dicts (grid_search scores with it)."""
        content = self.target_content.detach().cpu()
        content_size = content.shape[-1]
        results = []
        if verbose:
            print("Robustness check (mse to target; ~0.01 or under reproduces cleanly;\n"
                  "leak is max hidden activity outside the pattern - near 0 or the\n"
                  "margin sprouts worm structures on a big canvas):")
        for extra in extra_sizes:
            canvas_size = self.size + extra
            tgt = pad_to_canvas(content, canvas_size)
            m = (canvas_size - content_size) // 2
            halo = 6
            dead = torch.ones(canvas_size, canvas_size)
            dead[max(0, m - halo):m + content_size + halo,
                 max(0, m - halo):m + content_size + halo] = 0.0
            dead = dead.to(self.device)
            seed = torch.cat([make_seed(self.channels, canvas_size, radius=radius)
                              for radius in radii]).to(self.device).contiguous(
                                  memory_format=self.memory_format)
            with torch.no_grad():
                x = seed
                for t in range(steps):
                    mask = self._gate_mask(canvas_size, canvas_size, t)
                    x, _ = self.model.step(x, update_mask=mask)
                    x = quantize_visible(x)
                leaks = (x[:, 3:] * dead).abs().flatten(1).amax(1).cpu().tolist()
                # Preserve the legacy CPU reduction's exact reported values;
                # only the expensive CA rollout is batched on the GPU.
                rgb_cpu = x[:, :3].cpu()
                mses = [F.mse_loss(rgb, tgt).item() for rgb in rgb_cpu]
            for radius, mse, leak in zip(radii, mses, leaks):
                results.append({"canvas": canvas_size, "seed_radius": radius,
                                "mse": mse, "leak": leak})
                if verbose:
                    print(f"  canvas={canvas_size:4d}px seed_radius={radius}: mse={mse:.4f} leak={leak:.4f}")
        return results

    def animate(self, steps=400, from_seed=True, quantize=False,
                show_hidden=False, interval=30):
        """Animate CA evolution. from_seed grows from the center-pixel seed;
        quantize emulates the browser's 8-bit visible channels;
        show_hidden also displays the hidden memory channels. Updates are
        gated with the browser's hash (timestep = step index)."""
        with torch.no_grad():
            x = self.seed.clone() if from_seed else self.pool[random.randrange(self.pool_size)].unsqueeze(0)
            frames = [x.clone()]
            for t in range(steps):
                mask = wgsl_gate_mask(x.shape[-1], x.shape[-2], t,
                                      self.model.fire_rate, device=x.device)
                x, _ = self.model.step(x, update_mask=mask)
                if quantize:
                    x = quantize_visible(x)
                frames.append(x.clone())

        hidden = self.channels - 3
        if show_hidden and hidden > 0:
            # grid scales with the channel count (RGB pane + one per hidden)
            fig, axes = make_grid_axes(1 + hidden)
            ims = [axes[0].imshow(frames[0][0, :3].permute(1, 2, 0).cpu())]
            axes[0].set_title("RGB")
            for h in range(hidden):
                ims.append(axes[h + 1].imshow(frames[0][0, 3 + h].cpu(), cmap='gray', vmin=0, vmax=1))
                axes[h + 1].set_title(f"hidden {h}")
            for ax in axes:
                ax.axis("off")

            def update_grid(i):
                ims[0].set_data(frames[i][0, :3].permute(1, 2, 0).cpu())
                for h in range(hidden):
                    ims[h + 1].set_data(frames[i][0, 3 + h].cpu())
                fig.suptitle(f"step {i}")
                return ims

            update = update_grid
        else:
            fig, ax = plt.subplots()
            im = ax.imshow(frames[0][0, :3].permute(1, 2, 0).cpu())
            ax.axis("off")

            def update_rgb(i):
                im.set_data(frames[i][0, :3].permute(1, 2, 0).cpu())
                ax.set_title(f"step {i}")
                return [im]

            update = update_rgb

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
        plt.show()
