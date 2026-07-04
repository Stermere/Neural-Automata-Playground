import argparse
import json
import os
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from matplotlib.animation import FuncAnimation
from PIL import Image

default_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainingImages", "Emoji.png")
checkpoints_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


# ======================================================
# Update rules: each entry pairs the Python step math with the WGSL
# activation exported to the playground, so they cannot diverge.
# The shader wraps every channel's activation in clamp(..., 0, 1).
# ======================================================
UPDATE_RULES = {
    "linear": {
        "step": lambda x, conv, delta: x + delta * conv,
        # margin: only penalize runaway pre-activations, allow healthy saturation
        "margin": (-0.5, 1.5),
        "wgsl": """fn activation(convX: f32) -> f32 {{
    let lastX: f32 = activationContext.cellState[activationContext.channel];
    return clamp(lastX + {delta} * convX, 0.0, 1.0);
}}""",
    },
    "tanh": {
        # tanh bounds updates to +/-delta, so pre never exceeds the linear margin
        # and saturation would be gradient-dead: penalize anything outside [0, 1]
        "step": lambda x, conv, delta: x + delta * torch.tanh(conv),
        "margin": (0.0, 1.0),
        "wgsl": """fn activation(convX: f32) -> f32 {{
    let lastX: f32 = activationContext.cellState[activationContext.channel];
    return clamp(lastX + {delta} * tanh(convX), 0.0, 1.0);
}}""",
    },
}


# ======================================================
# Model: single 5x5 conv over all channels, mirroring the compute shader
# (wrap-around sampling, no bias, every cell updated every step)
# ======================================================
class CAModel(nn.Module):
    def __init__(self, channels=11, kernel_size=5, delta=0.25, rule="tanh"):
        super().__init__()
        self.channels = channels
        self.delta = delta
        self.rule = rule
        self.bestEval = None
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=kernel_size // 2, padding_mode='circular', bias=False)
        nn.init.zeros_(self.conv.weight)

    def step(self, x):
        rule = UPDATE_RULES[self.rule]
        pre = rule["step"](x, self.conv(x), self.delta)
        lo, hi = rule["margin"]
        overflow = (pre - pre.clamp(lo, hi)).abs().mean()
        return pre.clamp(0.0, 1.0), overflow

    def forward(self, x, steps=1):
        for _ in range(steps):
            x, _ = self.step(x)
        return x

    def exportToPlaygroundFormat(self, save_dir, filepath="TrainedWeights.json"):
        weights = self.conv.weight.detach().cpu().tolist()  # [out][in][5][5] — the shader's layout
        assert len(weights) == self.channels and len(weights[0]) == self.channels

        activation_code = UPDATE_RULES[self.rule]["wgsl"].format(delta=float(self.delta))

        export_dict = {
            "weights": weights,
            "activationCode": activation_code,
            "normalize": False
        }

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filepath)
        with open(save_path, "w") as f:
            json.dump(export_dict, f, indent=2)

        print(f"CA kernel exported to {save_path}")

    def saveBest(self, score):
        if self.bestEval is None or score < self.bestEval:
            self.bestEval = score
            self.best_weights = {k: v.clone() for k, v in self.state_dict().items()}

    def loadBest(self):
        if self.bestEval is not None:
            self.load_state_dict(self.best_weights)
            print(f"Best model with score {self.bestEval:.6f} loaded.")
        else:
            print("No best model to load.")


# ======================================================
# Seed / target helpers
# ======================================================
def make_seed(channels, size, batch=1, radius=0, jitter=0):
    """All channels 1.0 in a small blob near the center.
    radius=0, jitter=0 (the default) is a single pixel, matching the browser's
    smallest brush. Training reseeds with randomized radius/jitter so the
    model tolerates a real brush's imprecision instead of requiring a
    mathematically perfect dab."""
    seed = torch.zeros(batch, channels, size, size)
    cy = size // 2 + (random.randint(-jitter, jitter) if jitter else 0)
    cx = size // 2 + (random.randint(-jitter, jitter) if jitter else 0)
    y0, y1 = max(0, cy - radius), min(size, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(size, cx + radius + 1)
    seed[:, :, y0:y1, x0:x1] = 1.0
    return seed


def load_target_content(path, content_size):
    img = Image.open(path).convert("RGBA")
    black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(black_bg, img).convert("RGB")  # transparency -> black, the CA's zero fixed point
    transform = T.Compose([T.Resize((content_size, content_size)), T.ToTensor()])
    return transform(img)  # [3, content_size, content_size]


def pad_to_canvas(content, canvas_size):
    """Centers `content` in a black canvas of canvas_size, adding a dead-cell
    margin. Training against a padded canvas keeps growth away from the
    training border, so the kernel doesn't learn to rely on wraparound at one
    exact canvas size — that's what lets it generalize to a bigger canvas."""
    content_size = content.shape[-1]
    if content_size == canvas_size:
        return content
    margin = (canvas_size - content_size) // 2
    canvas = torch.zeros(3, canvas_size, canvas_size)
    canvas[:, margin:margin + content_size, margin:margin + content_size] = content
    return canvas


def load_target(path, size, margin=0):
    content = load_target_content(path, size - 2 * margin)
    return pad_to_canvas(content, size).unsqueeze(0)  # [1, 3, H, W]


# ======================================================
# Parity check: reimplement the compute shader's math with plain loops
# and compare against CAModel.step. Catches any weight-layout mistake
# before hours of training.
# ======================================================
def verify_shader_parity(channels=11, size=12, rule="tanh", delta=0.25):
    model = CAModel(channels=channels, delta=delta, rule=rule)
    with torch.no_grad():
        model.conv.weight.uniform_(-0.5, 0.5)
    x = torch.rand(1, channels, size, size)

    with torch.no_grad():
        got, _ = model.step(x)

    w = model.conv.weight
    k = w.shape[-1]
    half = k // 2
    expected = torch.empty_like(x)
    for out_ch in range(channels):
        for y in range(size):
            for px in range(size):
                conv_x = 0.0
                for in_ch in range(channels):
                    for ky in range(-half, half + 1):
                        for kx in range(-half, half + 1):
                            weight = w[out_ch, in_ch, ky + half, kx + half].item()
                            state = x[0, in_ch, (y + ky) % size, (px + kx) % size].item()
                            conv_x += weight * state
                last = x[0, out_ch, y, px].item()
                if rule == "linear":
                    pre = last + delta * conv_x
                else:
                    pre = last + delta * torch.tanh(torch.tensor(conv_x)).item()
                expected[0, out_ch, y, px] = min(max(pre, 0.0), 1.0)

    diff = (got - expected).abs().max().item()
    if diff > 1e-4:
        raise AssertionError(f"Shader parity check FAILED: max diff {diff:.6f}")
    print(f"Shader parity check passed (max diff {diff:.2e}).")


# ======================================================
# Trainer: Growing-NCA recipe — grow the target image from a seed,
# sample pool for long-term persistence.
# ======================================================
class CATrainer:
    def __init__(self, img_path, img_size=48, channels=11, pool_size=256,
                 batch_size=8, lr=1e-3, delta=0.25, rule="tanh", margin=None, device=None,
                 checkpoint_dir=None):
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

        self.model = CAModel(channels=channels, delta=delta, rule=rule).to(self.device)
        self.target_content = load_target_content(img_path, img_size - 2 * self.margin).to(self.device)
        self.target = pad_to_canvas(self.target_content, img_size).unsqueeze(0).to(self.device)
        self.seed = make_seed(channels, img_size).to(self.device)
        self.pool = make_seed(channels, img_size, batch=pool_size).to(self.device)
        self.dead_mask = self._make_dead_mask().to(self.device)

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

    def save_checkpoint(self, epoch, smooth_loss):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        stem = f"epoch_{epoch:06d}"
        torch.save({
            "epoch": epoch,
            "loss": smooth_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, f"{stem}.pt"))

        self.model.exportToPlaygroundFormat(self.checkpoint_dir, filepath=f"{stem}.json")

    def train(self, epochs=4000, min_steps=48, max_steps=96,
              overflow_weight=0.1, leak_weight=0.5, print_every=50, checkpoint_every=500):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=1e-4)
        target_batch = self.target.expand(self.batch_size, -1, -1, -1)
        running = deque(maxlen=50)
        epoch = 0

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
                x[per_sample.argmax()] = reseed[0]

                steps = random.randint(min_steps, max_steps)
                overflow_total = x.new_zeros(())
                for _ in range(steps):
                    x, ov = self.model.step(x)
                    overflow_total = overflow_total + ov

                # hidden activity must be dead in the margin, not merely
                # invisible — leaked hidden state grows without bound once the
                # canvas is bigger than the training grid
                hidden_leak = (x[:, 3:] ** 2 * self.dead_mask).sum() \
                    / (self.dead_mask.sum() * (self.channels - 3) * x.shape[0])

                loss = F.mse_loss(x[:, :3], target_batch) \
                    + overflow_weight * overflow_total / steps \
                    + leak_weight * hidden_leak

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
                    print(f"[{epoch}] loss={loss.item():.5f} smooth={smooth:.5f} "
                          f"leak={hidden_leak.item():.5f} lr={lr_now:.2e}")

                if checkpoint_every and epoch > 0 and epoch % checkpoint_every == 0:
                    self.save_checkpoint(epoch, smooth)
                    print(f"Checkpoint saved: {self.checkpoint_dir}/epoch_{epoch:06d}.pt")

        except KeyboardInterrupt:
            print("Training interrupted. Keeping best weights.")

        self.save_checkpoint(epoch, running[-1] if running else float('nan'))
        print(f"Final checkpoint saved: {self.checkpoint_dir}/epoch_{epoch:06d}.pt")

    def evaluate_robustness(self, steps=600, extra_sizes=(0, 64, 192), radii=(0, 1, 2)):
        """Grow the pattern under conditions the browser will actually produce
        — canvases bigger than training, and seed dabs from a 1px pixel up to
        a few px wide — quantizing visible channels every step like the
        rgba8unorm texture does. Catches a fragile kernel before you export
        and deploy it, instead of finding out on the website."""
        content = self.target_content.detach().cpu()
        content_size = content.shape[-1]
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
                    for _ in range(steps):
                        x, _ = self.model.step(x)
                        x = torch.cat([torch.round(x[:, :3] * 255) / 255, x[:, 3:]], dim=1)
                    leak = (x[0, 3:] * dead).abs().max().item()
                mse = F.mse_loss(x[0, :3].cpu(), tgt).item()
                print(f"  canvas={canvas_size:4d}px seed_radius={radius}: mse={mse:.4f} leak={leak:.4f}")

    def animate(self, steps=400, from_seed=True, quantize=False,
                show_hidden=False, interval=30):
        """Animate CA evolution. from_seed grows from the center-pixel seed;
        quantize emulates the browser's 8-bit visible channels;
        show_hidden also displays the hidden memory channels."""
        with torch.no_grad():
            x = self.seed.clone() if from_seed else self.pool[random.randrange(self.pool_size)].unsqueeze(0)
            frames = [x.clone()]
            for _ in range(steps):
                x, _ = self.model.step(x)
                if quantize:
                    x[:, :3] = torch.round(x[:, :3] * 255) / 255
                frames.append(x.clone())

        hidden = self.channels - 3
        if show_hidden and hidden > 0:
            rows, cols = 3, 4
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
            axes = axes.ravel()
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


# ======================================================
# Usage
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train an NCA kernel for the WebGPU playground")
    parser.add_argument("--image", default=default_image)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rule", choices=list(UPDATE_RULES), default="tanh")
    parser.add_argument("--delta", type=float, default=0.25, help="update strength per step")
    parser.add_argument("--steps", type=int, default=400, help="animation steps")
    parser.add_argument("--quantize", action="store_true", help="emulate 8-bit visible channels in animation")
    parser.add_argument("--show-hidden", action="store_true", help="also animate hidden channels")
    parser.add_argument("--margin", type=int, default=None,
                         help="blank margin (px) around the target inside the training canvas, "
                              "so growth doesn't rely on wraparound at one exact canvas size; "
                              "default scales with --size (~15%% per side), use 0 for the old edge-to-edge behavior")
    parser.add_argument("--leak-weight", type=float, default=0.5,
                        help="penalty on hidden-channel activity in the dead margin; "
                             "keeps the pattern from sprouting worm structures into empty "
                             "space on canvases larger than the training grid")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="save a checkpoint every N epochs (0 to disable periodic saves)")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="directory for this run's checkpoints; "
                              "defaults to a new timestamped folder under kernalPreTraining/checkpoints")
    parser.add_argument("-y", "--export", action="store_true", help="export without prompting")
    args = parser.parse_args()

    verify_shader_parity(channels=args.channels, rule=args.rule, delta=args.delta)

    trainer = CATrainer(args.image, img_size=args.size, channels=args.channels,
                        lr=args.lr, rule=args.rule, delta=args.delta, margin=args.margin,
                        checkpoint_dir=args.checkpoint_dir)
    print(f"Training on {args.image} ({args.size}px, margin={trainer.margin}px, "
          f"{args.channels} channels, device={trainer.device})")
    print(f"Checkpoints: {trainer.checkpoint_dir}")

    trainer.train(epochs=args.epochs, leak_weight=args.leak_weight,
                  checkpoint_every=args.checkpoint_every)
    trainer.model.loadBest()
    trainer.evaluate_robustness()

    if args.export or input("Save pattern? ").strip().lower() == 'y':
        trainer.model.exportToPlaygroundFormat(trainer.checkpoint_dir)

    trainer.animate(steps=args.steps, quantize=args.quantize, show_hidden=args.show_hidden)
