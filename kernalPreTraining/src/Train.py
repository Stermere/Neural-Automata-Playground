"""CLI entry point: pre-train an NCA kernel and export it for the playground.

The implementation lives in the sibling modules:
  ca_model.py  — CAModel, update rules, exported WGSL, quantization, hash gate
  targets.py   — seed and target-image helpers
  parity.py    — shader parity check
  trainer.py   — CATrainer (training loop, robustness eval, animation)
  grid_search.py — hyperparameter sweeps over CATrainer
"""
import argparse

from ca_model import PLAYGROUND_MAX_CHANNELS, UPDATE_RULES
from parity import verify_shader_parity, verify_shader_parity_mlp
from paths import default_image
from trainer import CATrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train an NCA kernel for the WebGPU playground")
    parser.add_argument("--image", default=default_image)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--channels", type=int, default=11)
    parser.add_argument("--pool-size", type=int, default=256,
                        help="number of persistent samples kept in the training pool")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="samples drawn from the pool per training step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rule", choices=list(UPDATE_RULES), default="tanh")
    parser.add_argument("--delta", type=float, default=0.25,
                        help="initial per-channel update strength (learned during training)")
    parser.add_argument("--fire-rate", type=float, default=0.5,
                        help="probability a cell applies its update each step (stochastic "
                             "updates, exported as a hash gate); 1.0 disables the gate")
    parser.add_argument("--mlp-hidden", type=int, default=128,
                        help="hidden units of the per-cell MLP inserted between the 5x5 "
                             "conv outputs and the update rule (conv -> ReLU layer -> "
                             "output layer); 0 trains the legacy conv-only kernel")
    parser.add_argument("--steps", type=int, default=400, help="animation steps")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="min rollout steps per training epoch; default = --size")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="max rollout steps per training epoch; default = 2x --size")
    parser.add_argument("--quantize", action="store_true", help="emulate 8-bit visible channels in animation")
    parser.add_argument("--show-hidden", action="store_true", help="also animate hidden channels")
    parser.add_argument("--margin", type=int, default=None,
                         help="blank margin (px) around the target inside the training canvas, "
                              "so growth doesn't rely on wraparound at one exact canvas size; "
                              "default scales with --size (~15%% per side), use 0 for the old edge-to-edge behavior")
    parser.add_argument("--leak-weight", type=float, default=0.5,
                        help="penalty on hidden-channel activity in the dead margin; "
                             "keeps the pattern from sprouting worm structures into empty "
                             "space on canvases larger than the training grid")
    parser.add_argument("--edge-weight", type=float, default=2.0,
                        help="weight of the Sobel edge loss; pushes for sharp boundaries "
                             "instead of MSE's blurry average")
    parser.add_argument("--fg-weight", type=float, default=3.0,
                        help="extra loss weight on the content region relative to the black "
                             "margin, so the easy margin doesn't dilute the image gradient")
    parser.add_argument("--damage-n", type=int, default=2,
                        help="pool samples that get a random hole cut each epoch "
                             "(regeneration training); 0 disables")
    parser.add_argument("--overflow-weight", type=float, default=0.1,
                        help="penalty on pre-clamp cell values leaving the rule's valid "
                             "range; discourages the model from relying on hard clamping")
    parser.add_argument("--grad-ckpt-steps", type=int, default=64,
                        help="rollout steps per gradient-checkpoint segment; cuts backprop "
                             "memory so long rollouts / big canvases fit, at ~1/3 extra "
                             "compute; 0 disables checkpointing")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="save a checkpoint every N epochs (0 to disable periodic saves)")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="directory for this run's checkpoints; "
                              "defaults to a new timestamped folder under kernalPreTraining/checkpoints")
    parser.add_argument("-y", "--export", action="store_true", help="export without prompting")
    args = parser.parse_args()

    if args.channels > PLAYGROUND_MAX_CHANNELS:
        parser.error(f"--channels {args.channels} exceeds the playground's maximum of "
                     f"{PLAYGROUND_MAX_CHANNELS} — the export would load as garbage in the browser")

    if args.mlp_hidden:
        verify_shader_parity_mlp(channels=args.channels, rule=args.rule, delta=args.delta,
                                 fire_rate=args.fire_rate)
    else:
        verify_shader_parity(channels=args.channels, rule=args.rule, delta=args.delta,
                             fire_rate=args.fire_rate)

    trainer = CATrainer(args.image, img_size=args.size, channels=args.channels,
                        pool_size=args.pool_size, batch_size=args.batch_size,
                        lr=args.lr, rule=args.rule, delta=args.delta, margin=args.margin,
                        fire_rate=args.fire_rate, fg_weight=args.fg_weight,
                        mlp_hidden=args.mlp_hidden, checkpoint_dir=args.checkpoint_dir)
    arch = f"mlp_hidden={args.mlp_hidden}" if args.mlp_hidden else "legacy 5x5 conv"
    print(f"Training on {args.image} ({args.size}px, margin={trainer.margin}px, "
          f"{args.channels} channels, {arch}, pool={args.pool_size}, "
          f"batch={args.batch_size}, device={trainer.device})")
    print(f"Checkpoints: {trainer.checkpoint_dir}")

    trainer.train(epochs=args.epochs, min_steps=args.min_steps, max_steps=args.max_steps,
                  overflow_weight=args.overflow_weight, leak_weight=args.leak_weight,
                  edge_weight=args.edge_weight, damage_n=args.damage_n,
                  grad_ckpt_steps=args.grad_ckpt_steps, checkpoint_every=args.checkpoint_every)
    trainer.model.loadBest()
    trainer.evaluate_robustness()

    if args.export or input("Save pattern? ").strip().lower() == 'y':
        trainer.model.exportToPlaygroundFormat(trainer.checkpoint_dir)

    trainer.animate(steps=args.steps, quantize=args.quantize, show_hidden=args.show_hidden)
