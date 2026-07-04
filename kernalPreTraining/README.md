# Kernel Pre-Training

Trains an 11-channel Neural Cellular Automata kernel (3 visible RGB + 8 hidden
memory channels) using the Growing Neural CA recipe — grow a target image
from a single seed pixel, sample-pool training for long-term persistence —
then exports it as a config the WebGPU playground can import directly.

The exported kernel is mathematically matched to `src/shaders/compute.wgsl`
(same 5x5 circular convolution, same per-channel clamp), verified automatically
every run by a parity check before training starts.

## Setup

```
python -m venv kernalPreTraining/.venv
kernalPreTraining/.venv/Scripts/pip install -r kernalPreTraining/requirements.txt
```

(On Linux/Mac use `kernalPreTraining/.venv/bin/pip` instead.)

## Usage

Train with defaults (Emoji.png, 48px, 11 channels, 4000 epochs, tanh rule) and
export + animate at the end:

```
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py -y
```

Without `-y` you'll be prompted `Save pattern? ` after training finishes.

### Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--image` | `trainingImages/Emoji.png` | Target image to grow (also try `Fall.png`, `Wave.png`) |
| `--size` | `48` | Training grid size in pixels — smaller trains faster, good for smoke tests |
| `--epochs` | `4000` | Training iterations |
| `--channels` | `11` | Total channels (3 visible + hidden); max the playground supports is 11 |
| `--lr` | `1e-3` | Adam learning rate |
| `--rule` | `tanh` | Update rule: `tanh` (more expressive) or `linear` (weaker, converges to flatter patterns) |
| `--delta` | `0.25` | Update strength per step |
| `--steps` | `400` | Number of steps to animate after training |
| `--quantize` | off | Round visible channels to 8-bit each animation step, emulating the browser's texture format |
| `--show-hidden` | off | Also animate the 8 hidden memory channels as grayscale panels |
| `--margin` | ~15% of `--size` | Blank border (px) around the target inside the training canvas, so growth learns to stay clear of the training edge instead of relying on wraparound at that one exact size — this is what lets the kernel reproduce correctly on a browser canvas of a different size. Use `0` for the old edge-to-edge behavior |
| `-y`, `--export` | off | Export without the interactive prompt |

Examples:

```
# quick smoke test, small grid, few epochs
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py --epochs 300 --size 40 -y

# just check shader math matches, no training
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py --epochs 0

# train on a different image, inspect hidden channels + quantization
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py --image kernalPreTraining/trainingImages/Wave.png --show-hidden --quantize -y
```

## Output

Exports to `~/Downloads/TrainedWeights.json`:

```jsonc
{
  "weights": [ /* [outChannel][inChannel][5][5] nested array */ ],
  "activationCode": "fn activation(convX: f32) -> f32 { ... }",  // WGSL
  "normalize": false
}
```

Import this file in the playground's config widget — channel count is
inferred automatically from `weights.length`.

## Notes

- Training reseeds with a randomized dab (a 1-5px blob, jittered a few
  pixels off-center) so the kernel tolerates the imprecision of a real
  brush click, not just a mathematically perfect single pixel. The default
  `--margin` keeps the target away from the training canvas edge so growth
  doesn't depend on wraparound at one exact canvas size. Together these are
  what let a trained kernel reproduce correctly on a browser canvas that's a
  different size than `--size`, painted with a normal (not pixel-precise)
  brush click.
- After training, a robustness check grows the pattern at several canvas
  sizes and seed radii and prints the MSE to the target for each — check
  this before exporting. If a config still looks fragile there, increase
  `--margin` or lower `--delta` (large deltas make each step's update bigger,
  which amplifies any mismatch between training and deployment conditions).
- If `linear` plateaus at a flat blob, `tanh` is more expressive and is the
  default for that reason.
- Interrupting training with Ctrl+C skips straight to loading the best
  saved weights so far.
