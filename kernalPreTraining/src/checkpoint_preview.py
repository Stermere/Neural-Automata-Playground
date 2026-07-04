"""Animate a single checkpoint's growth including its hidden channels — the
same view Train.py shows live after training (trainer.animate(show_hidden=True)),
but for a checkpoint saved earlier from Train.py or a grid_search run.

Reads size/channels/rule/fire_rate straight from the checkpoint's saved
config (trainer.py writes one into every checkpoint), so a bare checkpoint
path is normally all you need. --size/--rule/--fire-rate below only exist to
override that, or to fill in for checkpoints saved before this field
existed (channels alone still gets inferred from the conv weight shape).
Regrows the pattern from a seed the same way evaluate_robustness does
(quantized visible channels, browser-exact hash gate), capturing all 500
steps by default and animating RGB alongside every hidden channel.

See grid_preview.py to compare RGB across many runs instead of inspecting
one run's hidden state.

Usage:

  python checkpoint_preview.py path/to/run_dir            # picks the latest epoch_*.pt
  python checkpoint_preview.py path/to/epoch_004000.pt
  python checkpoint_preview.py run_dir --rule linear --fire-rate 1.0  # override a stale/missing config
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from ca_model import UPDATE_RULES, model_from_checkpoint
from preview_utils import latest_checkpoint, make_grid_axes, rgb_of, rollout_frames, save_animation


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a checkpoint .pt file, or a directory to pick the latest one from")
    parser.add_argument("--rule", choices=list(UPDATE_RULES), default=None,
                        help="override; default: read from the checkpoint, else 'tanh'")
    parser.add_argument("--fire-rate", type=float, default=None,
                        help="override; default: read from the checkpoint, else 0.5")
    parser.add_argument("--size", type=int, default=None,
                        help="canvas size to grow on; default: read from the checkpoint, else 48")
    parser.add_argument("--canvas-extra", type=int, default=0,
                        help="grow on a canvas this many px bigger than training, to preview generalization")
    parser.add_argument("--seed-radius", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500, help="rollout steps for the preview render")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="capture every Nth step; default 1 captures and shows every frame")
    parser.add_argument("--cols", type=int, default=None, help="grid columns; default ~sqrt(panes)")
    parser.add_argument("--fps", type=int, default=24, help="playback speed, for both the interactive window and the saved gif")
    parser.add_argument("--out", default=None, help="also save the animation here (default: alongside the checkpoint)")
    parser.add_argument("--no-show", action="store_true", help="skip the interactive window, just save")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = latest_checkpoint(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    if not config:
        print(f"No saved config in {ckpt_path} (older checkpoint) — "
              f"falling back to defaults/overrides for size/rule/fire-rate.")
    base_size = args.size if args.size is not None else config.get("size", 48)

    # picks CAModel or MLPCAModel from the state-dict shape automatically
    model = model_from_checkpoint(checkpoint, rule=args.rule, fire_rate=args.fire_rate).to(device)
    model.eval()
    channels = model.channels

    size = base_size + args.canvas_extra
    hidden = channels - 3
    print(f"Rendering {ckpt_path} ({channels} channels, {size}px, "
          f"{args.steps} steps, every {args.frame_stride})...")
    frames = rollout_frames(model, channels, size, args.seed_radius, args.steps, device, args.frame_stride)

    fig, axes = make_grid_axes(1 + hidden, args.cols)
    ims = [axes[0].imshow(rgb_of(frames[0]))]
    axes[0].set_title("RGB", fontsize=9)
    for h in range(hidden):
        ims.append(axes[h + 1].imshow(frames[0][3 + h], cmap='gray', vmin=0, vmax=1))
        axes[h + 1].set_title(f"hidden {h}", fontsize=9)
    for ax in axes[:1 + hidden]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        ims[0].set_data(rgb_of(frames[i]))
        for h in range(hidden):
            ims[h + 1].set_data(frames[i][3 + h])
        fig.suptitle(f"{os.path.basename(ckpt_path)} — step {i * args.frame_stride}")
        return ims

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)

    out = args.out or os.path.join(os.path.dirname(ckpt_path) or ".", "preview.gif")
    save_animation(anim, out, args.fps)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
