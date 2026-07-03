import argparse
import json
import os
import random
from collections import deque

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from matplotlib.animation import FuncAnimation
from PIL import Image

downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
default_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trainingImages", "Emoji.png")


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

    def exportToPlaygroundFormat(self, filepath="TrainedWeights.json"):
        weights = self.conv.weight.detach().cpu().tolist()  # [out][in][5][5] — the shader's layout
        assert len(weights) == self.channels and len(weights[0]) == self.channels

        activation_code = UPDATE_RULES[self.rule]["wgsl"].format(delta=float(self.delta))

        export_dict = {
            "weights": weights,
            "activationCode": activation_code,
            "normalize": False
        }

        save_path = os.path.join(downloads_path, filepath)
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
def make_seed(channels, size, batch=1):
    """All channels 1.0 at the center pixel — matches the browser brush."""
    seed = torch.zeros(batch, channels, size, size)
    seed[:, :, size // 2, size // 2] = 1.0
    return seed


def load_target(path, size):
    img = Image.open(path).convert("RGBA")
    black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(black_bg, img).convert("RGB")  # transparency -> black, the CA's zero fixed point
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


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
                 batch_size=8, lr=1e-3, delta=0.25, rule="tanh", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = img_size
        self.channels = channels
        self.pool_size = pool_size
        self.batch_size = batch_size

        self.model = CAModel(channels=channels, delta=delta, rule=rule).to(self.device)
        self.target = load_target(img_path, img_size).to(self.device)
        self.seed = make_seed(channels, img_size).to(self.device)
        self.pool = make_seed(channels, img_size, batch=pool_size).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.95))
        self.scheduler = None  # created in train() once epochs is known

    def train(self, epochs=4000, min_steps=48, max_steps=96,
              overflow_weight=0.1, print_every=50):
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=1e-4)
        target_batch = self.target.expand(self.batch_size, -1, -1, -1)
        running = deque(maxlen=50)

        try:
            for epoch in range(epochs):
                idx = torch.randint(0, self.pool_size, (self.batch_size,))
                x = self.pool[idx]

                # reseed the worst-looking sample so growth from seed keeps training
                with torch.no_grad():
                    per_sample = ((x[:, :3] - target_batch) ** 2).mean(dim=(1, 2, 3))
                x[per_sample.argmax()] = self.seed[0]

                steps = random.randint(min_steps, max_steps)
                overflow_total = x.new_zeros(())
                for _ in range(steps):
                    x, ov = self.model.step(x)
                    overflow_total = overflow_total + ov

                loss = F.mse_loss(x[:, :3], target_batch) \
                    + overflow_weight * overflow_total / steps

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
                    print(f"[{epoch}] loss={loss.item():.5f} smooth={smooth:.5f} lr={lr_now:.2e}")

        except KeyboardInterrupt:
            print("Training interrupted. Keeping best weights.")

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
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rule", choices=list(UPDATE_RULES), default="tanh")
    parser.add_argument("--delta", type=float, default=0.25, help="update strength per step")
    parser.add_argument("--steps", type=int, default=400, help="animation steps")
    parser.add_argument("--quantize", action="store_true", help="emulate 8-bit visible channels in animation")
    parser.add_argument("--show-hidden", action="store_true", help="also animate hidden channels")
    parser.add_argument("-y", "--export", action="store_true", help="export without prompting")
    args = parser.parse_args()

    verify_shader_parity(channels=args.channels, rule=args.rule, delta=args.delta)

    trainer = CATrainer(args.image, img_size=args.size, channels=args.channels,
                        lr=args.lr, rule=args.rule, delta=args.delta)
    print(f"Training on {args.image} ({args.size}px, {args.channels} channels, device={trainer.device})")

    trainer.train(epochs=args.epochs)
    trainer.model.loadBest()

    if args.export or input("Save pattern? ").strip().lower() == 'y':
        trainer.model.exportToPlaygroundFormat()

    trainer.animate(steps=args.steps, quantize=args.quantize, show_hidden=args.show_hidden)
