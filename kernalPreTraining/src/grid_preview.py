"""Animate every run of a grid_search sweep side by side, so you can watch
all configs grow at once instead of digging through run subfolders.

Loads each run's final checkpoint and regrows its pattern from a seed the
same way evaluate_robustness does (quantized visible channels, browser-exact
hash gate), capturing all 500 steps by default and playing every one of them
back in sync across one matplotlib grid, sorted best-to-worst by score,
titled with the swept params and score/mse/leak. Works on an in-progress
sweep too, since grid_search rewrites results.json after every run. Pass
--static for a single final-frame snapshot instead (faster, useful for a
quick glance at a large sweep).

See checkpoint_preview.py to inspect one run's hidden channels instead of
comparing RGB across many runs.

Usage:

  python grid_preview.py                       # animate the most recent gridsearch_* sweep
  python grid_preview.py path/to/gridsearch_dir
  python grid_preview.py --canvas-extra 64      # preview growth on a bigger-than-trained canvas
  python grid_preview.py --static               # single final-frame grid instead of an animation
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

from ca_model import CAModel
from paths import checkpoints_root
from preview_utils import latest_checkpoint, make_grid_axes, rgb_of, rollout_frames, save_animation


def find_latest_sweep():
    candidates = sorted(glob.glob(os.path.join(checkpoints_root, "gridsearch_*")))
    if not candidates:
        raise SystemExit(f"No gridsearch_* folders found under {checkpoints_root}")
    return candidates[-1]


def build_model(run, meta, canvas_extra, device):
    params = run["params"]
    channels = params.get("channels", meta["channels"])
    rule = params.get("rule", "tanh")
    fire_rate = params.get("fire_rate", 0.5)
    delta = params.get("delta", 0.25)  # overwritten by the loaded checkpoint anyway
    size = params.get("size", meta["size"]) + canvas_extra

    model = CAModel(channels=channels, delta=delta, rule=rule, fire_rate=fire_rate).to(device)
    checkpoint = torch.load(latest_checkpoint(run["dir"]), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, channels, size


def run_title(run):
    return f"#{run['run']} {run['label']}\n" \
           f"score={run['score']:.4f} mse={run['mean_mse']:.4f} leak={run['max_leak']:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", nargs="?", default=None,
                        help="grid_search output folder; defaults to the most recent "
                             "gridsearch_* under kernalPreTraining/checkpoints")
    parser.add_argument("--static", action="store_true",
                        help="render one final-frame snapshot per run instead of animating growth")
    parser.add_argument("--steps", type=int, default=500,
                        help="rollout steps for the preview render")
    parser.add_argument("--seed-radius", type=int, default=1)
    parser.add_argument("--canvas-extra", type=int, default=12,
                        help="grow on a canvas this many px bigger than training, to preview generalization")
    parser.add_argument("--cols", type=int, default=None, help="grid columns; default ~sqrt(n)")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="capture every Nth step; default 1 captures and shows every frame")
    parser.add_argument("--fps", type=int, default=24, help="playback speed, for both the interactive window and the saved gif")
    parser.add_argument("--out", default=None,
                        help="also save the grid here (default: <sweep_dir>/grid.gif or grid.png for --static)")
    parser.add_argument("--no-show", action="store_true", help="skip the interactive window, just save")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir or find_latest_sweep()
    results_path = os.path.join(sweep_dir, "results.json")
    with open(results_path) as f:
        meta = json.load(f)
    runs = sorted(meta["runs"], key=lambda r: r["score"])
    if not runs:
        raise SystemExit(f"No completed runs in {results_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = args.steps
    stride = steps if args.static else args.frame_stride

    print(f"Rendering {len(runs)} runs from {sweep_dir} ({steps} steps each, every {stride})...")
    all_frames = []
    for run in runs:
        model, channels, size = build_model(run, meta, args.canvas_extra, device)
        all_frames.append(rollout_frames(model, channels, size, args.seed_radius, steps, device, stride))

    fig, axes = make_grid_axes(len(runs), args.cols)

    if args.static:
        for ax, run, frames in zip(axes, runs, all_frames):
            ax.imshow(rgb_of(frames[-1]))
            ax.set_title(run_title(run), fontsize=7)
            ax.axis("off")

        fig.suptitle(os.path.basename(sweep_dir))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = args.out or os.path.join(sweep_dir, "grid.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
        if not args.no_show:
            plt.show()
        return

    n_frames = min(len(f) for f in all_frames)
    ims = []
    for ax, run, frames in zip(axes, runs, all_frames):
        ims.append(ax.imshow(rgb_of(frames[0])))
        ax.set_title(run_title(run), fontsize=7)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(i):
        for im, frames in zip(ims, all_frames):
            im.set_data(rgb_of(frames[i]))
        fig.suptitle(f"{os.path.basename(sweep_dir)} — step {i * stride}")
        return ims

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)

    out = args.out or os.path.join(sweep_dir, "grid.gif")
    save_animation(anim, out, args.fps)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
