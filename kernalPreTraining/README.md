# Kernel Pre-Training

Trains an 11-channel Neural Cellular Automata kernel (3 visible RGB + 8 hidden
memory channels) using the Growing Neural CA recipe — grow a target image
from a single seed pixel, sample-pool training for long-term persistence,
random damage for regeneration — then exports it as a config the WebGPU
playground can import directly.

By default the update rule is a 5x5 conv followed by a per-cell two-layer MLP
(`--mlp-hidden` units: conv outputs -> ReLU layer -> output layer). The MLP is
the genuinely nonlinear step a single convolution can't express — it's what
lets detailed images converge instead of plateauing into blur — and it ships
in the export as an `mlp` block the shader applies between the conv results
and the activation. The conv weights themselves stay a normal part of the
config, so trained patterns remain editable in the playground's weight editor.
`--mlp-hidden 0` trains the legacy conv-only kernel.

The MLP also sees the cell's own raw state, concatenated after the conv
results (the Growing-NCA identity-filter idea: without it the conv must spend
capacity approximating identity kernels just to pass the cell's state
through). `--mlp-no-state-input` trains the older conv-outputs-only MLP for
A/B comparison; both variants load in the playground, which infers the input
width from `w1`.

Beyond that, the kernel carries a per-channel bias (conv-only; the MLP's
biases live in the `mlp` block), a per-channel learned update rate, and
stochastic updates (each cell applies its update with probability
`--fire-rate` per step). These ship inside the exported activation code —
`NCA_BIAS` / `NCA_DELTA` arrays and a deterministic `ncaGate()` hash of
(x, y, timestep). Training also rounds the visible channels to 8 bits every
step, exactly like the browser's rgba8unorm texture, so long deployed
rollouts don't drift from what the kernel saw in training.

The exported kernel is mathematically matched to `src/shaders/compute.wgsl`
(same 5x5 circular convolution, same per-cell MLP, same per-channel clamp),
verified automatically every run by a parity check before training starts.

## Layout

| File | What's in it |
|---|---|
| `src/Train.py` | CLI entry point — train one config, evaluate, export, animate |
| `src/grid_search.py` | CLI entry point — sweep hyperparameters, score and rank configs |
| `src/ca_model.py` | `CAModel` (conv-only) and `MLPCAModel` (conv + per-cell MLP), update rules, exported WGSL templates, browser-exact quantization and stochastic-gate helpers |
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

Train with defaults (Emoji.png, 48px, 11 channels, 128-unit MLP, 4000 epochs,
tanh rule) and export + animate at the end:

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
| `--channels` | `11` | Total channels (3 visible + hidden); max the playground supports is 16 |
| `--mlp-hidden` | `128` | Hidden units of the per-cell MLP between the conv outputs and the update rule; `0` trains the legacy conv-only kernel |
| `--mlp-no-state-input` | off | Drop the cell's own raw state from the MLP inputs (conv outputs only — the pre-state-input architecture, kept for A/B comparison) |
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
| `--no-cuda-graph` | off | Disable automatic CUDA Graph acceleration (mainly useful for debugging or very low-VRAM runs) |
| `--compile` | off | Compile fixed rollout segments with Inductor/Triton; measured ~2.3x at 150px with full BPTT |
| `--bptt-steps` | full rollout | Backpropagate only through the final N simulated steps; `16` combined with compilation measured >5x at 150px, with a shorter gradient horizon |
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

# 150px, semantics-preserving fusion (full gradient history; ~2.3x measured)
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py --size 150 --compile --grad-ckpt-steps 16 -y

# 150px maximum-throughput mode (~5.34x measured; gradients cover final 16 steps)
kernalPreTraining/.venv/Scripts/python.exe kernalPreTraining/src/Train.py --size 150 --compile --grad-ckpt-steps 16 --bptt-steps 16 -y
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
fire-rate, fg-weight, mlp-hidden, mlp-state-input, epochs, min-steps,
max-steps, overflow-weight, leak-weight, edge-weight, damage-n,
grad-ckpt-steps`.

Each run gets its own subfolder (final checkpoint + `TrainedWeights.json`)
under a timestamped `gridsearch_*` folder; `results.json` is rewritten after
every run so an interrupted sweep keeps what it finished, and Ctrl+C prints
the leaderboard of completed runs. Pass `--seed N` to compare configs on
identical random draws. The default `--epochs 1200` is a screening budget —
re-train the winner at full length with Train.py.

## Output

Exports `TrainedWeights.json` into the run's checkpoint folder:

```jsonc
{
  "weights": [ /* [outChannel][inChannel][5][5] nested array */ ],
  // the per-cell MLP the shader applies to the conv results, flattened
  // into a storage buffer as [w1][b1][w2][b2]; absent for --mlp-hidden 0.
  // With stateInput, w1's rows are 2*channels wide: conv results first,
  // then the cell's raw state (the playground keys off the row width)
  "mlp": {
    "hiddenDim": 128,
    "stateInput": true,
    "w1": [ /* [hiddenDim][channels or 2*channels] */ ],
    "b1": [ /* [hiddenDim] */ ],
    "w2": [ /* [channels][hiddenDim] */ ],
    "b2": [ /* [channels] */ ]
  },
  // WGSL: NCA_DELTA / NCA_BIAS per-channel arrays, the ncaGate() stochastic
  // update hash, and the activation function that ties them together
  "activationCode": "var<private> NCA_DELTA ... fn activation(convX: f32) -> f32 { ... }",
  "normalize": false
}
```

Import this file in the playground's config widget — channel count is
inferred automatically from `weights.length`. Configs without an `mlp` block
(everything exported before it existed, and anything hand-built in the
playground) load exactly as before.

## Notes

- CUDA training uses channels-last tensor storage, which improved a measured
  150px/16-channel rollout on an RTX 3070 from **68 to 123 steps/s (1.81x)**.
  It also captures rollout, loss, backward, and gradient normalization in one
  reusable CUDA Graph whenever that graph fits the current card. At startup the
  trainer measures two short real backward passes for the exact canvas, batch,
  architecture, AMP mode, and GPU; it extrapolates activation memory to the
  maximum rollout, adds capture overhead, and compares it with live free VRAM
  after reserving headroom. Small launch-bound grids benefit most: the default
  32px/16-channel CLI architecture improved from **5.35 to 40.64 training
  epochs/s (7.60x)** in a 48-step steady-state benchmark. A 150px/450-step
  rollout was estimated at about 31GB on the 8GB test card, so it automatically
  keeps gradient checkpointing instead. Different cards/configurations make
  their own measured decision. Capture failures also fall back safely, and
  `--no-cuda-graph` forces the eager/checkpointed path. Exact browser-hash
  training stays eager; the default statistically-equivalent Bernoulli gate is
  graph accelerated when memory permits.
- Large-canvas Inductor/Triton compilation works at the **rollout-segment**
  level, not one CA step at a time. That exposes recurrent elementwise work to
  fusion while fixed 16-step specializations avoid recompiling every random
  rollout length. On the 150px/16-channel RTX 3070 benchmark, compiled full
  BPTT reduced a 96-step training operation by about 2.3x without changing its
  math. The opt-in `--bptt-steps 16` mode still simulates and quantizes all 96
  forward steps but detaches the earlier trajectory, so only the final 16 steps
  contribute gradients. Combined with `--compile --grad-ckpt-steps 16`, this
  reduced the same operation from **1.095s to 0.205s (5.34x)**, including loss,
  backward, gradient normalization, and Adam. This changes optimization—not
  the exported architecture or forward CA behavior—and may weaken learning of
  very long-term credit assignment. Compare robustness scores against a full-
  BPTT run; a good compromise is fast TBPTT pretraining followed by a shorter
  full-BPTT (`--compile` without `--bptt-steps`) fine-tuning stage.
- Training reseeds with a randomized dab (a 1-5px blob with a touch of
  value noise) so the kernel tolerates the imprecision of a real
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
