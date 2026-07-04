"""Shared helpers for the checkpoint/grid-search animation previews."""
import glob
import math
import os

import matplotlib.pyplot as plt
import torch

from ca_model import quantize_visible, wgsl_gate_mask
from targets import make_seed


def latest_checkpoint(path):
    """path is either a checkpoint .pt file, or a directory to search for one."""
    if os.path.isfile(path):
        return path
    checkpoints = sorted(glob.glob(os.path.join(path, "epoch_*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    return checkpoints[-1]


def rollout_frames(model, channels, size, seed_radius, steps, device, stride):
    """Grow from a seed the same way evaluate_robustness does (quantized visible
    channels, browser-exact hash gate). Returns a list of [channels, H, W]
    numpy arrays — the full cell state, not just RGB — captured every `stride`
    steps, starting with the seed itself."""
    x = make_seed(channels, size, radius=seed_radius).to(device)
    frames = [x[0].cpu().numpy()]
    with torch.no_grad():
        for t in range(steps):
            mask = wgsl_gate_mask(size, size, t, model.fire_rate, device=device)
            x, _ = model.step(x, update_mask=mask)
            x = quantize_visible(x)
            if (t + 1) % stride == 0:
                frames.append(x[0].cpu().numpy())
    return frames


def rgb_of(frame):
    return frame[:3].transpose(1, 2, 0).clip(0, 1)


def make_grid_axes(n, cols=None):
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def save_animation(anim, out, fps):
    try:
        anim.save(out, writer="pillow", fps=fps)
        print(f"Saved {out}")
    except Exception as e:
        print(f"Could not save {out} ({e}); showing animation only.")
