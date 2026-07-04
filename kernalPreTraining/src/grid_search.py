"""Hyperparameter grid search over the NCA trainer.

Trains every combination of the swept parameters, then scores each config
with the same robustness evaluation used before export: grow the pattern
from 1px and 3px seeds on the training canvas and a 64px-bigger one, and
measure MSE-to-target plus hidden leak.

  score = mean(mse) + max(leak)          (lower is better)

MSE tracks how faithfully the pattern grows back; max leak is the canary for
worm eruptions on canvases bigger than the training grid. Each run gets its
own subfolder (final checkpoint + playground export) under a sweep folder,
results.json is rewritten after every run (crash/interrupt safe), and a
leaderboard prints at the end. Ctrl+C aborts the sweep and prints the
leaderboard of the runs that finished.

Usage (sweep values comma-separated; dashes or underscores both fine):

  python grid_search.py --size 48 --epochs 1200 lr=1e-3,3e-3 delta=0.15,0.25 fire-rate=0.5,1.0

The per-config epoch budget is for screening, not final quality — re-train
the winning config at full length with Train.py.
"""
import argparse
import itertools
import json
import os
import random
import time

import torch

from parity import verify_shader_parity, verify_shader_parity_mlp
from paths import checkpoints_root, default_image
from trainer import CATrainer

# sweepable name -> CATrainer.__init__ keyword
TRAINER_KEYS = {"size": "img_size", "channels": "channels", "pool_size": "pool_size",
                "batch_size": "batch_size", "lr": "lr", "delta": "delta", "rule": "rule",
                "margin": "margin", "fire_rate": "fire_rate", "fg_weight": "fg_weight",
                "mlp_hidden": "mlp_hidden"}
# sweepable names passed straight to CATrainer.train
TRAIN_KEYS = ("epochs", "min_steps", "max_steps", "overflow_weight", "leak_weight",
              "edge_weight", "damage_n", "grad_ckpt_steps")


def parse_value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text  # e.g. rule=tanh


def parse_grid(specs):
    grid = {}
    for spec in specs:
        name, sep, values = spec.partition("=")
        key = name.strip().replace("-", "_")
        if not sep or not values:
            raise SystemExit(f"Bad sweep spec '{spec}' — expected name=value1,value2,...")
        if key not in TRAINER_KEYS and key not in TRAIN_KEYS:
            valid = ", ".join(sorted(list(TRAINER_KEYS) + list(TRAIN_KEYS)))
            raise SystemExit(f"Unknown parameter '{name}'. Searchable: {valid}")
        grid[key] = [parse_value(v) for v in values.split(",")]
    return grid


def score_run(eval_results):
    mean_mse = sum(r["mse"] for r in eval_results) / len(eval_results)
    max_leak = max(r["leak"] for r in eval_results)
    return mean_mse + max_leak, mean_mse, max_leak


def print_leaderboard(runs):
    print("\n=== Leaderboard (lower score is better) ===")
    for rank, run in enumerate(sorted(runs, key=lambda r: r["score"]), 1):
        print(f"{rank:3d}. score={run['score']:.4f} (mse={run['mean_mse']:.4f}, "
              f"leak={run['max_leak']:.4f})  {run['label']}")
    best = min(runs, key=lambda r: r["score"])
    print(f"\nBest config: {best['label']}")
    print(f"Export: {os.path.join(best['dir'], 'TrainedWeights.json')}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", nargs="+",
                        help="parameters to sweep, e.g. lr=1e-3,3e-3 delta=0.15,0.25")
    parser.add_argument("--image", default=default_image)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--mlp-hidden", type=int, default=128,
                        help="hidden units of the per-cell MLP for runs that don't sweep "
                             "mlp-hidden; 0 trains the legacy conv-only kernel")
    parser.add_argument("--epochs", type=int, default=1200,
                        help="epochs per config — a screening budget; re-train the winner "
                             "with Train.py at full length")
    parser.add_argument("--eval-steps", type=int, default=400,
                        help="rollout length of the scoring evaluation")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None,
                        help="seed python/torch RNG identically before each run, so configs "
                             "are compared on the same random draws")
    parser.add_argument("--sweep-dir", default=None,
                        help="output folder; defaults to a timestamped gridsearch folder "
                             "under kernalPreTraining/checkpoints")
    args = parser.parse_args()

    from ca_model import PLAYGROUND_MAX_CHANNELS
    grid = parse_grid(args.sweep)
    for ch in grid.get("channels", [args.channels]):
        if ch > PLAYGROUND_MAX_CHANNELS:
            raise SystemExit(f"channels={ch} exceeds the playground's maximum of "
                             f"{PLAYGROUND_MAX_CHANNELS} — the export would load as garbage")
    names = list(grid)
    combos = list(itertools.product(*(grid[n] for n in names)))
    sweep_dir = args.sweep_dir or os.path.join(
        checkpoints_root, f"gridsearch_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(sweep_dir, exist_ok=True)
    results_path = os.path.join(sweep_dir, "results.json")

    # one parity check per shader-affecting combo before burning GPU-hours
    for rule in set(grid.get("rule", ["tanh"])):
        for fire_rate in set(grid.get("fire_rate", [0.5])):
            for mlp_hidden in set(grid.get("mlp_hidden", [args.mlp_hidden])):
                if mlp_hidden:
                    verify_shader_parity_mlp(channels=args.channels, rule=rule,
                                             fire_rate=fire_rate)
                else:
                    verify_shader_parity(channels=args.channels, rule=rule,
                                         fire_rate=fire_rate)

    print(f"Sweeping {len(combos)} configurations -> {sweep_dir}")
    runs = []
    try:
        for i, combo in enumerate(combos, start=1):
            overrides = dict(zip(names, combo))
            label = " ".join(f"{k}={v}" for k, v in overrides.items())
            run_dir = os.path.join(sweep_dir, f"run{i:03d}")

            trainer_kwargs = {"img_size": args.size, "channels": args.channels,
                              "mlp_hidden": args.mlp_hidden}
            train_kwargs = {"epochs": args.epochs, "print_every": args.print_every,
                            "checkpoint_every": 0, "catch_interrupt": False,
                            "label": f"run {i}/{len(combos)}: {label}"}
            for key, value in overrides.items():
                if key in TRAINER_KEYS:
                    trainer_kwargs[TRAINER_KEYS[key]] = value
                else:
                    train_kwargs[key] = value

            print(f"\n=== run {i}/{len(combos)}: {label} ===")
            if args.seed is not None:
                random.seed(args.seed)
                torch.manual_seed(args.seed)
            trainer = CATrainer(args.image, checkpoint_dir=run_dir, **trainer_kwargs)
            trainer.train(**train_kwargs)
            trainer.model.loadBest()
            eval_results = trainer.evaluate_robustness(
                steps=args.eval_steps, extra_sizes=(0, 64), radii=(0, 1), verbose=False)
            score, mean_mse, max_leak = score_run(eval_results)
            trainer.model.exportToPlaygroundFormat(run_dir)

            runs.append({"run": i, "label": label, "params": overrides, "dir": run_dir,
                         "score": score, "mean_mse": mean_mse, "max_leak": max_leak,
                         "eval": eval_results})
            with open(results_path, "w") as f:
                json.dump({"image": args.image, "size": args.size, "channels": args.channels,
                           "epochs": args.epochs, "eval_steps": args.eval_steps,
                           "runs": runs}, f, indent=2)
            print(f"run {i}: score={score:.4f} (mse={mean_mse:.4f}, leak={max_leak:.4f})")

            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print(f"\nSweep interrupted after {len(runs)}/{len(combos)} completed runs.")

    if runs:
        print_leaderboard(runs)
        print(f"Full results: {results_path}")
    else:
        print("No completed runs.")


if __name__ == "__main__":
    main()
