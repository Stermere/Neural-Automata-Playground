"""Seed and target-image helpers."""
import random

import torch
import torchvision.transforms as T
from PIL import Image


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
