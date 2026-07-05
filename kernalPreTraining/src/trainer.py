"""Trainer: Growing-NCA recipe — grow the target image from a seed,
sample pool for long-term persistence, damage for regeneration."""
import os
import random
import time
from collections import deque

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


class CATrainer:
    def __init__(self, img_path, img_size=48, channels=11, pool_size=256,
                 batch_size=8, lr=1e-3, delta=0.25, rule="tanh", margin=None,
                 fire_rate=0.5, fg_weight=3.0, mlp_hidden=128,
                 mlp_state_input=True, device=None, checkpoint_dir=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            checkpoints_root, f"{image_name}_{time.strftime('%Y%m%d-%H%M%S')}")
        self.size = img_size
        self.channels = channels
        self.pool_size = pool_size
        self.batch_size = batch_size
        # Default margin scales with grid size so the target never touches
        # the training canvas border (~15% dead space per side).
        self.margin = margin if margin is not None else max(4, round(img_size * 0.15))

        # mlp_hidden > 0 adds a per-cell MLP between the 5x5 conv outputs and
        # the update rule; 0 falls back to the legacy conv-only kernel
        if mlp_hidden:
            self.model = MLPCAModel(channels=channels, hidden_dim=mlp_hidden,
                                    delta=delta, rule=rule, fire_rate=fire_rate,
                                    state_input=mlp_state_input).to(self.device)
        else:
            self.model = CAModel(channels=channels, delta=delta, rule=rule,
                                 fire_rate=fire_rate).to(self.device)
        self.target_content = load_target_content(img_path, img_size - 2 * self.margin).to(self.device)
        self.target = pad_to_canvas(self.target_content, img_size).unsqueeze(0).to(self.device)
        self.seed = make_seed(channels, img_size).to(self.device)
        self.pool = make_seed(channels, img_size, batch=pool_size).to(self.device)
        self.dead_mask = self._make_dead_mask().to(self.device)

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

        # coordinate grid for cutting circular damage holes
        ys, xs = torch.meshgrid(torch.arange(img_size, dtype=torch.float32),
                                torch.arange(img_size, dtype=torch.float32), indexing="ij")
        self.grid_y = ys.to(self.device)
        self.grid_x = xs.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.95))
        self.scheduler = None  # created in train() once epochs is known

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

    def _rollout_chunk(self, x, steps):
        """One backprop segment of a rollout: step, then round the visible
        channels like the browser's rgba8unorm texture does. Kept small so it
        can be gradient-checkpointed — long rollouts recompute activations on
        the backward pass instead of storing every step."""
        overflow = x.new_zeros(())
        for _ in range(steps):
            x, ov = self.model.step(x)
            x = quantize_visible(x)
            overflow = overflow + ov
        return x, overflow

    def _compute_loss(self, x, target_batch, target_edges, dead_cells, has_dead_zone,
                      overflow_total, steps, edge_weight, overflow_weight, leak_weight, distance):
        """Reconstruction (rgb + Sobel edges, via `distance`) plus the overflow
        and hidden-leak penalties. Returns (loss, rgb_loss, edge_loss, hidden_leak)
        so the caller can log the components alongside the combined total."""
        if has_dead_zone:
            hidden_leak = (x[:, 3:] ** 2 * self.dead_mask).sum() \
                / (dead_cells * (self.channels - 3) * x.shape[0])
        else:
            hidden_leak = x.new_zeros(())

        weight_norm = self.pixel_weight.sum() * x.shape[0]
        rgb_loss = (distance(x[:, :3] - target_batch) * self.pixel_weight).sum() / (weight_norm * 3)
        edge_loss = (distance(self._edges(x[:, :3]) - target_edges) * self.pixel_weight).sum() / (weight_norm * 6)

        loss = rgb_loss \
            + edge_weight * edge_loss \
            + overflow_weight * overflow_total / steps \
            + leak_weight * hidden_leak

        return loss, rgb_loss, edge_loss, hidden_leak

    def save_checkpoint(self, epoch, smooth_loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        stem = f"epoch_{epoch:06d}"
        torch.save({
            "epoch": epoch,
            "loss": smooth_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            # lets checkpoint_preview.py regrow this pattern without the
            # caller having to know how it was trained
            "config": {"size": self.size, "channels": self.channels,
                       "rule": self.model.rule, "fire_rate": self.model.fire_rate,
                       "mlp_hidden": getattr(self.model, "hidden_dim", 0),
                       "mlp_state_input": getattr(self.model, "state_input", False)},
        }, os.path.join(self.checkpoint_dir, f"{stem}.pt"))

        self.model.exportToPlaygroundFormat(self.checkpoint_dir, filepath=f"{stem}.json")

    def train(self, epochs=4000, min_steps=None, max_steps=None,
              overflow_weight=0.1, leak_weight=0.5, edge_weight=2.0,
              damage_n=2, grad_ckpt_steps=64, print_every=10, checkpoint_every=1000,
              catch_interrupt=True, label=None, loss_fn="huber"):
        """catch_interrupt=True swallows Ctrl+C and keeps the best weights
        (interactive use); grid_search passes False so one Ctrl+C can abort
        the whole sweep instead of silently skipping to the next config.
        loss_fn selects the rgb/edge reconstruction distance - one of
        LOSS_FUNCTIONS ("mse", "l1", "huber")."""
        distance = LOSS_FUNCTIONS[loss_fn]
        # Information crosses ~2px per step (5x5 kernel), halved by the fire
        # rate, so the step budget must grow with the canvas: too few steps
        # and the far side of the pattern can never even be reached, let
        # alone refined.
        if min_steps is None:
            min_steps = self.size
        if max_steps is None:
            max_steps = 3 * self.size
        damage_n = min(damage_n, self.batch_size - 1)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=1e-4)
        target_batch = self.target.expand(self.batch_size, -1, -1, -1)
        target_edges = self.target_edges.expand(self.batch_size, -1, -1, -1)
        running = deque(maxlen=50)
        epoch = 0

        # the dead zone is empty when the margin is smaller than the halo
        # (tiny grids) or there are no hidden channels — dividing by its
        # zero cell count would poison the loss with NaN
        dead_cells = self.dead_mask.sum().item()
        has_dead_zone = dead_cells > 0 and self.channels > 3

        title = label or os.path.basename(self.checkpoint_dir)
        monitor = TrainingMonitor(epochs, title=title)

        interrupted = False
        with monitor:
            monitor.note(
                f"Rollout {min_steps}-{max_steps} steps/epoch, fire_rate={self.model.fire_rate}, "
                f"edge_weight={edge_weight}, damage_n={damage_n}, "
                f"grad checkpoint segment={grad_ckpt_steps if grad_ckpt_steps else 'off'}, "
                f"leak zone={'active' if has_dead_zone else 'EMPTY (margin too small, leak penalty off)'}")
            try:
                for epoch in range(epochs):
                    idx = torch.randint(0, self.pool_size, (self.batch_size,))
                    x = self.pool[idx]

                    # reseed the worst-looking sample so growth from seed keeps training;
                    # randomize the seed's radius/jitter so the model learns to grow
                    # correctly from an imprecise brush dab, not just a perfect pixel
                    with torch.no_grad():
                        per_sample = ((x[:, :3] - target_batch) ** 2).mean(dim=(1, 2, 3))
                    radius = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
                    jitter = random.randint(0, 3)
                    reseed = make_seed(self.channels, self.size, radius=radius, jitter=jitter).to(self.device)
                    worst = per_sample.argmax().item()
                    x[worst] = reseed[0]

                    # cut holes in the best-grown samples so regrowth keeps training
                    for i in per_sample.argsort()[:damage_n].tolist():
                        if i != worst:
                            self._damage_sample(x[i])

                    steps = random.randint(min_steps, max_steps)
                    overflow_total = x.new_zeros(())
                    remaining = steps
                    while remaining > 0:
                        seg = min(grad_ckpt_steps, remaining) if grad_ckpt_steps else remaining
                        if grad_ckpt_steps:
                            x, ov = grad_checkpoint(self._rollout_chunk, x, seg, use_reentrant=False)
                        else:
                            x, ov = self._rollout_chunk(x, seg)
                        overflow_total = overflow_total + ov
                        remaining -= seg

                    loss, rgb_loss, edge_loss, hidden_leak = self._compute_loss(
                        x, target_batch, target_edges, dead_cells, has_dead_zone,
                        overflow_total, steps, edge_weight, overflow_weight, leak_weight, distance)

                    self.optimizer.zero_grad()
                    loss.backward()
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad /= (p.grad.norm() + 1e-8)
                    self.optimizer.step()
                    self.scheduler.step()

                    self.pool[idx] = x.detach()

                    running.append(loss.item())
                    smooth = sum(running) / len(running)
                    self.model.saveBest(smooth)

                    if epoch % print_every == 0:
                        lr_now = self.optimizer.param_groups[0]['lr']
                        monitor.update(epoch, loss=loss.item(), smooth=smooth,
                                      rgb=rgb_loss.item(), edge=edge_loss.item(),
                                      overflow=(overflow_total / steps).item(),
                                      leak=hidden_leak.item(),
                                      lr=lr_now, best=self.model.bestEval)

                    if checkpoint_every and epoch > 0 and epoch % checkpoint_every == 0:
                        self.save_checkpoint(epoch, smooth)
                        monitor.note(f"Checkpoint saved: {self.checkpoint_dir}/epoch_{epoch:06d}.pt")

            except KeyboardInterrupt:
                interrupted = True
                monitor.note("Training interrupted. Keeping best weights.")

        self.save_checkpoint(epoch, running[-1] if running else float('nan'))
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
            for radius in radii:
                seed = make_seed(self.channels, canvas_size, radius=radius).to(self.device)
                with torch.no_grad():
                    x = seed
                    for t in range(steps):
                        mask = wgsl_gate_mask(canvas_size, canvas_size, t,
                                              self.model.fire_rate, device=self.device)
                        x, _ = self.model.step(x, update_mask=mask)
                        x = quantize_visible(x)
                    leak = (x[0, 3:] * dead).abs().max().item()
                mse = F.mse_loss(x[0, :3].cpu(), tgt).item()
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
