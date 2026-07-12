"""CLI entry point: pre-train an NCA kernel and export it for the playground.

The implementation lives in the sibling modules:
  ca_model.py  — CAModel, update rules, exported WGSL, quantization, hash gate
  targets.py   — seed and target-image helpers
  parity.py    — shader parity check
  trainer.py   — CATrainer (training loop, robustness eval, animation)
  grid_search.py — hyperparameter sweeps over CATrainer
"""
import argparse
import os

from ca_model import PERCEPTION_INITIALIZATIONS, PLAYGROUND_MAX_CHANNELS, UPDATE_RULES
from parity import verify_shader_parity, verify_shader_parity_mlp
from paths import default_image
from trainer import CATrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train an NCA kernel for the WebGPU playground")
    parser.add_argument("--image", default=default_image)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=None,
                        help="number of persistent samples kept in the training pool; "
                             "default 32x --batch-size, so every entry gets refreshed "
                             "every ~32 epochs instead of going stale")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="samples drawn from the pool per training step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--rule", choices=list(UPDATE_RULES), default="tanh")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="initial per-channel update strength (learned during training)")
    parser.add_argument("--fire-rate", type=float, default=0.5,
                        help="probability a cell applies its update each step (stochastic "
                             "updates, exported as a hash gate); 1.0 disables the gate")
    parser.add_argument("--mlp-hidden", type=int, default=32,
                        help="hidden units of the per-cell MLP inserted between the 5x5 "
                             "conv outputs and the update rule (conv -> ReLU layer -> "
                             "output layer); 0 trains the legacy conv-only kernel")
    parser.add_argument("--mlp-hidden2", type=int, default=16,
                        help="hidden units of a second per-cell MLP layer (conv -> ReLU "
                             "-> ReLU layer -> output layer); defaults to --mlp-hidden's "
                             "width, 0 trains the single-hidden-layer MLP")
    parser.add_argument("--mlp-no-state-input", action="store_true",
                        help="drop the cell's own raw state from the MLP inputs, leaving "
                             "only the conv outputs (the pre-state-input architecture, "
                             "kept for A/B comparison)")
    parser.add_argument("--perception-init", choices=PERCEPTION_INITIALIZATIONS,
                        default="structured",
                        help="initialization of the trainable 5x5 perception conv: "
                             "structured uses balanced randomized identity/Sobel-X/"
                             "Sobel-Y/Laplacian diagonal filters; random and zeros "
                             "are A/B baselines")
    parser.add_argument("--output-init-std", type=float, default=1e-3,
                        help="standard deviation of the MLP's final-layer random "
                             "weights; small nonzero values preserve near-identity "
                             "updates while allowing immediate upstream gradients "
                             "(0 restores exact zero initialization)")
    parser.add_argument("--steps", type=int, default=400, help="animation steps")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="min rollout steps per training epoch; default scales with "
                             "--size and stretches as --fire-rate drops (size x (2 - fire_rate), "
                             "e.g. 1.5x size at fire-rate 0.5)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="max rollout steps per training epoch; default = 2x the min")
    parser.add_argument("--quantize", action="store_true", help="emulate 8-bit visible channels in animation")
    parser.add_argument("--show-hidden", action="store_true", help="also animate hidden channels")
    parser.add_argument("--margin", type=int, default=None,
                         help="blank margin (px) around the target inside the training canvas, "
                              "so growth doesn't rely on wraparound at one exact canvas size; "
                              "default scales with --size (~15%% per side), use 0 for the old edge-to-edge behavior")
    parser.add_argument("--leak-weight", type=float, default=3,
                        help="penalty on hidden-channel activity in the dead margin; "
                             "keeps the pattern from sprouting worm structures into empty "
                             "space on canvases larger than the training grid")
    parser.add_argument("--edge-weight", type=float, default=3,
                        help="weight of the Sobel edge loss; pushes for sharp boundaries "
                             "instead of MSE's blurry average")
    parser.add_argument("--fft-weight", type=float, default=3,
                        help="weight of the focal frequency loss: spectrum error re-weighted "
                             "toward the most-wrong frequencies, recovering the high-frequency "
                             "detail MSE averages away; 0 disables")
    parser.add_argument("--fg-weight", type=float, default=1.0,
                        help="extra loss weight on the content region relative to the black "
                             "margin, so the easy margin doesn't dilute the image gradient")
    parser.add_argument("--damage-n", type=int, default=1,
                        help="pool samples that get a random hole cut each epoch "
                             "(regeneration training — browser users scribble over "
                             "the pattern, so it's on by default); 0 disables")
    parser.add_argument("--train-gate", choices=("hash", "bernoulli"), default="bernoulli",
                        help="update gate used during training rollouts when --fire-rate < 1: "
                             "'bernoulli' samples a fresh random mask each step (statistically "
                             "matches the browser's gate, and skips rebuilding the WGSL mask "
                             "every step), 'hash' replays the browser's exact deterministic gate; "
                             "the robustness eval always scores with the hash gate either way")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="every N epochs, grow from scratch under deployment conditions "
                             "(bigger canvas, 1-3px seeds) and keep the best-scoring weights; "
                             "0 defers to a single probe at the end of training")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable bf16 mixed precision for the rollout (on by default "
                             "when the GPU supports it; loss math always stays f32)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile fixed rollout segments with Inductor/Triton; "
                             "fuses recurrent elementwise work (about 2.3x measured at "
                             "150px), with an eager fallback if compilation is unavailable")
    parser.add_argument("--bptt-steps", type=int, default=None,
                        help="truncate gradients to the final N rollout steps while still "
                             "simulating the full rollout; --compile --grad-ckpt-steps 16 "
                             "--bptt-steps 16 measured about 5.6x at 150px, but changes "
                             "the gradient horizon (default: full BPTT)")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="disable automatic full-training-step CUDA Graph replay; "
                             "useful for debugging or unusually memory-constrained runs")
    parser.add_argument("--overflow-weight", type=float, default=1,
                        help="penalty on pre-clamp cell values leaving the rule's valid "
                             "range; discourages the model from relying on hard clamping")
    parser.add_argument("--grad-ckpt-steps", type=int, default=64,
                        help="rollout steps per gradient-checkpoint segment; cuts backprop "
                             "memory so long rollouts / big canvases fit, at ~1/3 extra "
                             "compute; 0 disables checkpointing")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="save a checkpoint every N epochs (0 to disable periodic saves)")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="directory for this run's checkpoints; "
                              "defaults to a new timestamped folder under kernalPreTraining/checkpoints "
                              "(or, with --resume, the resumed checkpoint's own folder)")
    parser.add_argument("--resume", default=None,
                        help="path to a .pt checkpoint to resume from (model + optimizer "
                             "state and epoch count); the sample pool and best-eval "
                             "bookkeeping resume too when a resume_state.pt sits beside the "
                             "checkpoint, otherwise they restart fresh. The LR schedule "
                             "restarts a clean cosine decay over the epochs remaining (see "
                             "--lr to also change the rate). --epochs is the new total "
                             "epoch target, not an additional count")
    parser.add_argument("-y", "--export", action="store_true", help="export without prompting")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if args.resume and checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(args.resume)

    if args.channels > PLAYGROUND_MAX_CHANNELS:
        parser.error(f"--channels {args.channels} exceeds the playground's maximum of "
                     f"{PLAYGROUND_MAX_CHANNELS} — the export would load as garbage in the browser")

    if args.mlp_hidden:
        verify_shader_parity_mlp(channels=args.channels, rule=args.rule, delta=args.delta,
                                 fire_rate=args.fire_rate, hidden_dim2=args.mlp_hidden2,
                                 state_input=not args.mlp_no_state_input)
    else:
        verify_shader_parity(channels=args.channels, rule=args.rule, delta=args.delta,
                             fire_rate=args.fire_rate)

    trainer = CATrainer(args.image, img_size=args.size, channels=args.channels,
                        pool_size=args.pool_size, batch_size=args.batch_size,
                        lr=args.lr if args.lr is not None else 1e-3,
                        rule=args.rule, delta=args.delta, margin=args.margin,
                        fire_rate=args.fire_rate, fg_weight=args.fg_weight,
                        mlp_hidden=args.mlp_hidden, mlp_hidden2=args.mlp_hidden2,
                        mlp_state_input=not args.mlp_no_state_input,
                        perception_init=args.perception_init,
                        output_init_std=args.output_init_std,
                        checkpoint_dir=checkpoint_dir,
                         amp=False if args.no_amp else None, compile_step=False,
                         cuda_graph=False if args.no_cuda_graph else True)
    arch = (f"mlp_hidden={args.mlp_hidden}"
            f"{f'+{args.mlp_hidden2}' if args.mlp_hidden2 else ''}"
            f"{'' if args.mlp_no_state_input else '+state'}"
            if args.mlp_hidden else "legacy 5x5 conv")
    print(f"Training on {args.image} ({args.size}px, margin={trainer.margin}px, "
          f"{args.channels} channels, {arch}, pool={trainer.pool_size}, "
          f"batch={args.batch_size}, device={trainer.device})")
    print(f"Checkpoints: {trainer.checkpoint_dir}")

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        # load_checkpoint restores the optimizer's saved (already-decayed)
        # rate; an explicit --lr overrides it for this stage
        if args.lr is not None:
            for group in trainer.optimizer.param_groups:
                group['lr'] = args.lr
        lr_now = trainer.optimizer.param_groups[0]['lr']
        print(f"Resumed from {args.resume} at epoch {start_epoch}, lr={lr_now:g}")

    trainer.train(epochs=args.epochs, min_steps=args.min_steps, max_steps=args.max_steps,
                  overflow_weight=args.overflow_weight, leak_weight=args.leak_weight,
                  edge_weight=args.edge_weight, fft_weight=args.fft_weight,
                  damage_n=args.damage_n,
                  grad_ckpt_steps=args.grad_ckpt_steps, checkpoint_every=args.checkpoint_every,
                  start_epoch=start_epoch, eval_every=args.eval_every, gate=args.train_gate,
                  compile_rollout=args.compile, bptt_steps=args.bptt_steps)
    trainer.model.loadBest()
    trainer.evaluate_robustness()

    if args.export or input("Save pattern? ").strip().lower() == 'y':
        trainer.model.exportToPlaygroundFormat(trainer.checkpoint_dir)

    trainer.animate(steps=args.steps, quantize=args.quantize, show_hidden=args.show_hidden)
