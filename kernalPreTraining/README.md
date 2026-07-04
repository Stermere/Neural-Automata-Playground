# Kernel Pre-Training

Trains an 11-channel Neural Cellular Automata kernel (3 visible RGB + 8 hidden
memory channels) using the Growing Neural CA recipe — grow a target image
from a single seed pixel, sample-pool training for long-term persistence,
random damage for regeneration — then exports it as a config the WebGPU
playground can import directly.

Beyond the 5x5 conv weights, the kernel carries a per-channel bias, a
per-channel learned update rate, and stochastic updates (each cell applies its
update with probability `--fire-rate` per step). All three ship inside the
exported activation code — `NCA_BIAS` / `NCA_DELTA` arrays and a deterministic
`ncaGate()` hash of (x, y, timestep) — so no shader changes are needed.
Training also rounds the visible channels to 8 bits every step, exactly like
the browser's rgba8unorm texture, so long deployed rollouts don't drift from
what the kernel saw in training.

The exported kernel is mathematically matched to `src/shaders/compute.wgsl`
(same 5x5 circular convolution, same per-channel clamp), verified automatically
every run by a parity check before training starts.

## Layout

| File | What's in it |
|---|---|
| `src/Train.py` | CLI entry point — train one config, evaluate, export, animate |
| `src/grid_search.py` | CLI entry point — sweep hyperparameters, score and rank configs |
| `src/ca_model.py` | `CAModel`, update rules, exported WGSL templates, browser-exact quantization and stochastic-gate helpers |
| `src/trainer.py` | `CATrainer` — training loop, robustness evaluation, animation |
| `src/targets.py` | Seed and target-image helpers |
| `src/parity.py` | Shader parity check |
| `src/paths.py` | Shared path constants |

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
| `--pool-size` | `256` | Number of persistent samples kept in the training pool |
| `--batch-size` | `8` | Samples drawn from the pool per training step |
| `--lr` | `1e-3` | Adam learning rate |
| `--rule` | `tanh` | Update rule: `tanh` (more expressive) or `linear` (weaker, converges to flatter patterns) |
| `--delta` | `0.25` | Initial per-channel update strength (learned per channel during training) |
| `--fire-rate` | `0.5` | Probability a cell applies its update each step; exported as a deterministic hash gate. `1.0` disables stochastic updates |
| `--min-steps` / `--max-steps` | `--size` / `2x --size` | Rollout length range per training epoch — scales with the canvas so growth can reach and refine the whole pattern |
| `--edge-weight` | `2.0` | Weight of the Sobel edge loss; pushes for sharp boundaries instead of MSE's blurry average |
| `--fg-weight` | `3.0` | Extra loss weight on the content region vs the black margin, so the easy margin doesn't dilute the image gradient |
| `--damage-n` | `2` | Pool samples that get a random hole cut each epoch, training regeneration (what a user scribbling over the pattern demands). `0` disables |
| `--overflow-weight` | `0.1` | Penalty on pre-clamp cell values leaving the rule's valid range; discourages relying on hard clamping |
| `--grad-ckpt-steps` | `16` | Rollout steps per gradient-checkpoint segment; cuts backprop memory ~10x for long rollouts at ~1/3 extra compute. `0` disables |
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

## Grid search

`grid_search.py` trains every combination of the swept parameters and ranks
them by the same robustness evaluation used before export (grow from 1px and
3px seeds on the training canvas and a 64px-bigger one):

```
score = mean(mse) + max(leak)    # lower is better
```

Sweep values are comma-separated `name=v1,v2` pairs; anything not swept uses
the normal defaults:

```
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/grid_search.py --size 48 --epochs 1200 lr=1e-3,3e-3 delta=0.15,0.25 fire-rate=0.5,1.0
```

Searchable: `size, channels, pool-size, batch-size, lr, delta, rule, margin,
fire-rate, fg-weight, epochs, min-steps, max-steps, overflow-weight,
leak-weight, edge-weight, damage-n, grad-ckpt-steps`.

Each run gets its own subfolder (final checkpoint + `TrainedWeights.json`)
under a timestamped `gridsearch_*` folder; `results.json` is rewritten after
every run so an interrupted sweep keeps what it finished, and Ctrl+C prints
the leaderboard of completed runs. Pass `--seed N` to compare configs on
identical random draws. The default `--epochs 1200` is a screening budget —
re-train the winner at full length with Train.py.

## Output

Exports to `~/Downloads/TrainedWeights.json`:

```jsonc
{
  "weights": [ /* [outChannel][inChannel][5][5] nested array */ ],
  // WGSL: NCA_DELTA / NCA_BIAS per-channel arrays, the ncaGate() stochastic
  // update hash, and the activation function that ties them together
  "activationCode": "var<private> NCA_DELTA ... fn activation(convX: f32) -> f32 { ... }",
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
- Training matches deployment as closely as possible: visible channels are
  rounded to 8 bits every step (the browser's rgba8unorm texture does the
  same), and updates are stochastically gated at `--fire-rate`. The exported
  gate is a deterministic hash of (x, y, timestep), mirrored bit-for-bit by
  `wgsl_gate_mask()` in ca_model.py, so the robustness check and animation
  reproduce the browser's exact update pattern.
- Each epoch the worst pool sample is reseeded and the `--damage-n`
  best-grown samples get a random circular hole cut, so the kernel keeps
  training both growth-from-seed and regrowth-after-damage.
- Rollout length defaults to `--size`..`2x --size` steps because information
  crosses at most ~2px per step (halved by the fire rate) — bigger canvases
  need proportionally longer rollouts. Gradient checkpointing
  (`--grad-ckpt-steps`) keeps those long rollouts inside GPU memory.
- After training, a robustness check grows the pattern at several canvas
  sizes and seed radii and prints the MSE to the target for each — check
  this before exporting. If a config still looks fragile there, increase
  `--margin` or lower `--delta` (large deltas make each step's update bigger,
  which amplifies any mismatch between training and deployment conditions).
- If `linear` plateaus at a flat blob, `tanh` is more expressive and is the
  default for that reason.
- Interrupting training with Ctrl+C skips straight to loading the best
  saved weights so far.
