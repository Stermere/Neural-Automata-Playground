"""Preview the exact RGB tensor used as an NCA training target."""
import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image

from paths import default_image
from targets import load_target_content, pad_to_canvas


def main():
    parser = argparse.ArgumentParser(
        description="Show the target after the same resize, square padding, and "
                    "canvas padding used by CATrainer.")
    parser.add_argument("--image", default=default_image, help="source image")
    parser.add_argument("--size", type=int, default=32,
                        help="training canvas size in pixels")
    parser.add_argument("--content-size", type=int, default=None,
                        help="call load_target_content(image, content_size) directly "
                             "and preview its square output without outer training margin")
    parser.add_argument("--margin", type=int, default=None,
                        help="target margin in pixels; default matches training: "
                             "max(4, round(size * 0.15))")
    parser.add_argument("--save", default=None,
                        help="optional path for an exact-size RGB PNG")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open the matplotlib preview window")
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.content_size is not None and args.content_size <= 0:
        parser.error("--content-size must be positive")

    # These are the exact helpers CATrainer uses. This includes RGBA compositing
    # onto black, aspect-preserving bilinear resize, square black padding, and
    # centering inside the final training canvas.
    if args.content_size is not None:
        content_size = args.content_size
        margin = None
        target = load_target_content(args.image, content_size)
        canvas_size = content_size
    else:
        margin = args.margin if args.margin is not None else max(4, round(args.size * 0.15))
        if margin < 0:
            parser.error("--margin cannot be negative")
        content_size = args.size - 2 * margin
        if content_size <= 0:
            parser.error(f"margin {margin}px leaves no target area on a {args.size}px canvas")
        content = load_target_content(args.image, content_size)
        target = pad_to_canvas(content, args.size)
        canvas_size = args.size

    print(f"source:  {os.path.abspath(args.image)}")
    print(f"canvas:  {canvas_size}x{canvas_size}px")
    print(f"margin:  {f'{margin}px per side' if margin is not None else 'none (direct content mode)'}")
    print(f"content: {content_size}x{content_size}px")

    if args.save:
        # Values originate from an 8-bit PIL resize, so this round-trip writes
        # the exact visible target pixels without matplotlib scaling or axes.
        rgb8 = target.mul(255).round().byte().permute(1, 2, 0).numpy()
        Image.fromarray(rgb8, mode="RGB").save(args.save)
        print(f"saved:   {os.path.abspath(args.save)}")

    if not args.no_show:
        fig, ax = plt.subplots()
        ax.imshow(target.permute(1, 2, 0).numpy(), interpolation="nearest")
        title = (f"load_target_content: {content_size}px" if margin is None
                 else f"Training target: {canvas_size}px, margin {margin}px")
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()