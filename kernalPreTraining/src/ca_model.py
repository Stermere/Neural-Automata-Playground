"""The NCA models and everything that must stay in lockstep with the shader:
the update-rule math, the exported WGSL activation code, browser-exact
quantization, and the stochastic-update gate."""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Update rules: each entry pairs the Python step math with the WGSL
# update expression exported to the playground, so they cannot diverge.
# In Python `conv` is the model's raw update signal and already contains
# the bias; the shader's convX is the raw weighted sum, so NCA_BIAS is
# added explicitly in the WGSL. The exported activation wraps the
# expression in the per-channel delta/bias arrays, the stochastic-update
# gate (when fire_rate < 1), and clamp(..., 0, 1).
# ======================================================
# The playground's channel ceiling (MAX_TOTAL_CHANNELS in
# src/client/constants/channelConstants.ts — keep in sync). Training past it
# produces an export the browser can only load as garbage: the shader clamps
# to 16 channels while the weight/MLP buffers stay laid out for more.
PLAYGROUND_MAX_CHANNELS = 16

# Initialization modes for the learned spatial perception convolution. The
# structured mode gives the optimizer useful local features immediately while
# keeping every coefficient trainable; random/zeros retain useful A/B
# baselines for the two architectures' former defaults.
PERCEPTION_INITIALIZATIONS = ("structured", "random", "zeros")


def _init_perception(conv, mode="structured"):
    """Initialize a square C->C perception conv.

    ``structured`` places one identity, Sobel-X, Sobel-Y, or Laplacian filter
    on each channel's diagonal connection. Assignments are randomly shuffled
    but balanced (counts differ by at most one), and off-diagonal connections
    start at zero. This is the Growing-NCA perception prior adapted to the
    playground's fixed C-output convolution: unlike fixed paper filters, these
    weights remain ordinary trainable parameters.
    """
    if mode not in PERCEPTION_INITIALIZATIONS:
        raise ValueError(f"unknown perception_init={mode!r}; expected one of "
                         f"{PERCEPTION_INITIALIZATIONS}")

    with torch.no_grad():
        if mode == "zeros":
            nn.init.zeros_(conv.weight)
            return
        if mode == "random":
            # Match nn.Conv2d.reset_parameters rather than depending on the
            # random values consumed during module construction.
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            return

        out_channels, in_channels, kh, kw = conv.weight.shape
        if out_channels != in_channels or kh != kw or kh < 3 or kh % 2 == 0:
            raise ValueError("structured perception requires a square C->C conv "
                             "with an odd kernel size >= 3")

        identity = torch.zeros(3, 3, dtype=conv.weight.dtype,
                               device=conv.weight.device)
        identity[1, 1] = 1.0
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=conv.weight.dtype,
                               device=conv.weight.device) / 8.0
        sobel_y = sobel_x.t().contiguous()
        # L1-normalized like Sobel so its initial response scale is comparable.
        laplacian = torch.tensor([[0., 1., 0.],
                                  [1., -4., 1.],
                                  [0., 1., 0.]], dtype=conv.weight.dtype,
                                 device=conv.weight.device) / 8.0
        filters = torch.stack((identity, sobel_x, sobel_y, laplacian))

        conv.weight.zero_()
        assignments = torch.arange(out_channels, device=conv.weight.device) % len(filters)
        assignments = assignments[torch.randperm(out_channels, device=conv.weight.device)]
        offset = kh // 2 - 1
        for channel, filter_idx in enumerate(assignments.tolist()):
            conv.weight[channel, channel,
                        offset:offset + 3, offset:offset + 3] = filters[filter_idx]


def _init_output_layer(layer, std):
    """Near-zero random output init: stable updates without blocking gradients."""
    if std < 0:
        raise ValueError("output_init_std must be non-negative")
    if std:
        nn.init.normal_(layer.weight, mean=0.0, std=std)
    else:
        nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)

UPDATE_RULES = {
    "linear": {
        "step": lambda x, conv, delta: x + delta * conv,
        # margin: only penalize runaway pre-activations, allow healthy saturation
        "margin": (-0.5, 1.5),
        "wgsl_update": "lastX + NCA_DELTA[ch] * (convX + NCA_BIAS[ch])",
    },
    "tanh": {
        # tanh bounds updates to +/-delta, so pre never exceeds the linear margin
        # and saturation would be gradient-dead: penalize anything outside [0, 1]
        "step": lambda x, conv, delta: x + delta * torch.tanh(conv),
        "margin": (0.0, 1.0),
        "wgsl_update": "lastX + NCA_DELTA[ch] * tanh(convX + NCA_BIAS[ch])",
    },
    "relu": {
        # ReLU floored at -0.5 instead of 0: max(conv, 0) - 0.5
        "step": lambda x, conv, delta: x + delta * (F.relu(conv) - 0.5),
        "margin": (-0.5, 2.5),
        "wgsl_update": "lastX + NCA_DELTA[ch] * (max(convX + NCA_BIAS[ch], 0.0) - 0.5)",
    },
}


WGSL_GATE = """const NCA_FIRE_RATE: f32 = {fire_rate};

// Deterministic per-cell coin flip: a cell only applies its update when the
// hash of (x, y, timestep) lands under the fire rate, reproducing the
// stochastic updates the kernel was trained with. Mirrored bit-for-bit by
// wgsl_gate_mask() in ca_model.py so Python rollouts can match the browser.
fn ncaGate() -> bool {{
  var h: u32 = activationContext.gid.x * 374761393u
             + activationContext.gid.y * 668265263u
             + u32(activationContext.timestep) * 2246822519u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0 < NCA_FIRE_RATE;
}}

"""

WGSL_ACTIVATION = """var<private> NCA_DELTA: array<f32, {channels}> = array<f32, {channels}>({deltas});
var<private> NCA_BIAS: array<f32, {channels}> = array<f32, {channels}>({biases});

{gate_decl}fn activation(convX: f32) -> f32 {{
  let ch: u32 = activationContext.channel;
  let lastX: f32 = activationContext.cellState[ch];
{gate_check}  return clamp({update}, 0.0, 1.0);
}}"""

def _fmt_f32_list(values):
    return ", ".join(repr(float(v)) for v in values)


def quantize_visible(x):
    """Round the visible channels to 8 bits, straight-through for gradients.
    The browser stores RGB in an rgba8unorm texture, so deployment quantizes
    every single step; training through the same rounding keeps long rollouts
    from drifting away from what the deployed kernel actually experiences.
    Hidden channels live in f32 buffers and are left untouched."""
    q = torch.round(x[:, :3] * 255.0) / 255.0
    vis = x[:, :3] + (q - x[:, :3]).detach()
    return torch.cat([vis, x[:, 3:]], dim=1)


def wgsl_gate_mask(width, height, timestep, fire_rate, device=None):
    """[1,1,H,W] update mask mirroring the exported ncaGate() WGSL bit-for-bit
    (u32 wraparound hash, f32 division), so a Python rollout that uses the
    step index as the timestep reproduces the browser's exact update pattern.
    Returns None when fire_rate >= 1 (every cell updates every step)."""
    if fire_rate >= 1.0:
        return None
    m = 0xFFFFFFFF
    xs = torch.arange(width, dtype=torch.int64, device=device).view(1, -1)
    ys = torch.arange(height, dtype=torch.int64, device=device).view(-1, 1)
    h = (xs * 374761393 + ys * 668265263 + int(timestep) * 2246822519) & m
    h = ((h ^ (h >> 13)) * 1274126177) & m
    h = h ^ (h >> 16)
    gate = h.to(torch.float32) / 4294967295.0
    return (gate < fire_rate).view(1, 1, height, width).to(torch.float32)


class _CAModelBase(nn.Module):
    """Step math, best-weights bookkeeping, and activation-code export shared
    by both architectures. Subclasses provide _conv_out(x) — the raw
    per-channel update signal that UPDATE_RULES wraps — plus their own
    exportToPlaygroundFormat."""

    def __init__(self, channels, delta, rule, fire_rate):
        super().__init__()
        self.channels = channels
        self.rule = rule
        self.fire_rate = fire_rate
        self.bestEval = None
        # per-channel update rate: lets slow memory channels and fast visible
        # channels coexist; exported as the NCA_DELTA array
        self.delta = nn.Parameter(torch.full((channels,), float(delta)))

    def _conv_out(self, x):
        raise NotImplementedError

    def step(self, x, update_mask=None):
        """One CA step. update_mask ([B or 1, 1, H, W] of 0/1) selects which
        cells apply their update; None samples a Bernoulli(fire_rate) mask,
        matching the browser's hash gate statistically. Pass
        wgsl_gate_mask(...) instead to match the browser exactly."""
        rule = UPDATE_RULES[self.rule]
        delta = self.delta.view(1, -1, 1, 1)
        pre = rule["step"](x, self._conv_out(x), delta)
        lo, hi = rule["margin"]
        overflow = (pre - pre.clamp(lo, hi)).abs().mean()
        out = pre.clamp(0.0, 1.0)
        if update_mask is None and self.fire_rate < 1.0:
            update_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                                      device=x.device) < self.fire_rate).to(x.dtype)
        if update_mask is not None:
            out = update_mask * out + (1.0 - update_mask) * x
        return out, overflow

    def forward(self, x, steps=1):
        for _ in range(steps):
            x, _ = self.step(x)
        return x

    def _export_activation_code(self, biases):
        if self.fire_rate < 1.0:
            gate_decl = WGSL_GATE.format(fire_rate=repr(float(self.fire_rate)))
            gate_check = "  if (!ncaGate()) {\n    return lastX;\n  }\n"
        else:
            gate_decl = ""
            gate_check = ""
        return WGSL_ACTIVATION.format(
            channels=self.channels,
            deltas=_fmt_f32_list(self.delta.detach().cpu().tolist()),
            biases=_fmt_f32_list(biases),
            gate_decl=gate_decl,
            gate_check=gate_check,
            update=UPDATE_RULES[self.rule]["wgsl_update"],
        )

    def _write_export(self, export_dict, save_dir, filepath):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filepath)
        with open(save_path, "w") as f:
            json.dump(export_dict, f, indent=2)
        print(f"CA kernel exported to {save_path}")

    def saveBest(self, score):
        if self.bestEval is None or score < self.bestEval:
            self.bestEval = score
            self.best_weights = {k: v.clone() for k, v in self.state_dict().items()}

    def loadBest(self):
        if self.bestEval is not None:
            self.load_state_dict(self.best_weights)
            print(f"Best model with score {self.bestEval:.6f} loaded.")
        else:
            print("No best model to load.")


# ======================================================
# Legacy model: single 5x5 conv over all channels, mirroring the compute
# shader's conv path (wrap-around sampling). Beyond the conv it carries a
# per-channel bias, a per-channel learnable update rate (delta), and a
# stochastic update gate — all reproduced in the exported activation code,
# none of them need shader changes.
# ======================================================
class CAModel(_CAModelBase):
    def __init__(self, channels=11, kernel_size=5, delta=0.25, rule="tanh", fire_rate=0.5,
                 perception_init="structured"):
        super().__init__(channels, delta, rule, fire_rate)
        self.perception_init = perception_init
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=kernel_size // 2, padding_mode='circular', bias=True)
        _init_perception(self.conv, perception_init)
        nn.init.zeros_(self.conv.bias)

    def _conv_out(self, x):
        return self.conv(x)

    def exportToPlaygroundFormat(self, save_dir, filepath="TrainedWeights.json"):
        weights = self.conv.weight.detach().cpu().tolist()  # [out][in][5][5] — the shader's layout
        assert len(weights) == self.channels and len(weights[0]) == self.channels

        export_dict = {
            "weights": weights,
            "activationCode": self._export_activation_code(self.conv.bias.detach().cpu().tolist()),
            "normalize": False
        }
        self._write_export(export_dict, save_dir, filepath)


# ======================================================
# MLP model: the same learned 5x5 conv as the legacy model — still exported
# as `weights`, still editable in the playground — with a per-cell MLP
# (1x1 conv -> ReLU -> 1x1 conv, optionally -> ReLU -> 1x1 conv again)
# inserted between the conv outputs and the update rule. That MLP is the
# genuinely nonlinear update a single conv can't express; it ships as an
# "mlp" block the shader applies to the conv results, while delta/gate ride
# through the same exported activation code as the legacy model.
# ======================================================
class MLPCAModel(_CAModelBase):
    def __init__(self, channels=11, hidden_dim=128, kernel_size=5, delta=0.25,
                 rule="tanh", fire_rate=0.5, state_input=True, hidden_dim2=None,
                 perception_init="structured", output_init_std=1e-3):
        super().__init__(channels, delta, rule, fire_rate)
        self.hidden_dim = hidden_dim
        self.perception_init = perception_init
        self.output_init_std = output_init_std
        # state_input feeds the cell's own raw state to the MLP alongside the
        # conv outputs (the Growing-NCA identity-filter idea): without it the
        # conv must spend capacity approximating identity kernels just so the
        # MLP can see the state it is updating. The mode is inferred back from
        # w1's input width everywhere (checkpoints, playground), so old
        # conv-only-input kernels keep loading.
        self.state_input = state_input
        # hidden_dim2 adds a second hidden layer (w1 -> ReLU -> w2 -> ReLU ->
        # w3) before the output layer; falsy collapses to the original
        # single-hidden-layer shape (w1 -> ReLU -> w2) so checkpoints/exports
        # from before this existed keep loading unchanged.
        self.hidden_dim2 = hidden_dim2 or 0
        # No conv bias: the shader's conv path is a raw weighted sum, and a
        # per-channel bias before w1 is absorbed exactly by b1 anyway
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=kernel_size // 2, padding_mode='circular', bias=False)
        _init_perception(self.conv, perception_init)
        self.w1 = nn.Conv2d(channels * (2 if state_input else 1), hidden_dim, 1)
        # A tiny random final layer keeps the CA very close to the identity
        # map while allowing gradients into perception and hidden layers from
        # the first optimizer step. Exactly zero final weights block those
        # gradients until the final layer itself has moved away from zero.
        if self.hidden_dim2:
            self.w2 = nn.Conv2d(hidden_dim, self.hidden_dim2, 1)
            self.w3 = nn.Conv2d(self.hidden_dim2, channels, 1)
            _init_output_layer(self.w3, output_init_std)
        else:
            self.w2 = nn.Conv2d(hidden_dim, channels, 1)
            _init_output_layer(self.w2, output_init_std)

    def _conv_out(self, x):
        h = self.conv(x)
        if self.state_input:
            # conv results first, then the raw state — the shader's
            # mlpWeights layout and the export both assume this order
            h = torch.cat([h, x], dim=1)
        h = F.relu(self.w1(h))
        if self.hidden_dim2:
            h = F.relu(self.w2(h))
            return self.w3(h)
        return self.w2(h)

    def exportToPlaygroundFormat(self, save_dir, filepath="TrainedWeights.json"):
        weights = self.conv.weight.detach().cpu().tolist()  # [out][in][5][5] — the shader's layout
        assert len(weights) == self.channels and len(weights[0]) == self.channels

        # Flattened by the playground into the shader's mlpWeights buffer as
        # [w1][b1][w2][b2] (plus [w3][b3] when hidden_dim2 is set); the final
        # layer's bias ships here, so NCA_BIAS is zero. stateInput is
        # informational — the playground infers the MLP input width from
        # w1's row length.
        mlp = {
            "hiddenDim": self.hidden_dim,
            "stateInput": self.state_input,
            "w1": self.w1.weight.detach().cpu().squeeze(-1).squeeze(-1).tolist(),
            "b1": self.w1.bias.detach().cpu().tolist(),
            "w2": self.w2.weight.detach().cpu().squeeze(-1).squeeze(-1).tolist(),
            "b2": self.w2.bias.detach().cpu().tolist(),
        }
        if self.hidden_dim2:
            mlp["hiddenDim2"] = self.hidden_dim2
            mlp["w3"] = self.w3.weight.detach().cpu().squeeze(-1).squeeze(-1).tolist()
            mlp["b3"] = self.w3.bias.detach().cpu().tolist()

        export_dict = {
            "weights": weights,
            "mlp": mlp,
            "activationCode": self._export_activation_code([0.0] * self.channels),
            "normalize": False
        }
        self._write_export(export_dict, save_dir, filepath)


def model_from_checkpoint(checkpoint, rule=None, fire_rate=None):
    """Build the right architecture from a saved checkpoint dict — MLPCAModel
    when the state dict has MLP layers, the legacy conv CAModel otherwise —
    reading rule/fire_rate from the checkpoint's config (explicit args
    override, and cover checkpoints from before the config field existed)."""
    config = checkpoint.get("config", {})
    state = checkpoint["model_state"]
    rule = rule if rule is not None else config.get("rule", "tanh")
    fire_rate = fire_rate if fire_rate is not None else config.get("fire_rate", 0.5)
    if "w2.weight" in state:
        # w3 present means w2 is an interior hidden layer (its output width
        # is hidden_dim2, not the channel count) and w3 is the real output
        # layer; its absence is the original single-hidden-layer shape.
        has_layer2 = "w3.weight" in state
        channels = state["w3.weight"].shape[0] if has_layer2 else state["w2.weight"].shape[0]
        hidden_dim2 = state["w2.weight"].shape[0] if has_layer2 else 0
        model = MLPCAModel(channels=channels,
                           hidden_dim=state["w1.weight"].shape[0],
                           hidden_dim2=hidden_dim2,
                           # w1 twice as wide as the channel count means the
                           # cell state was concatenated to the MLP input
                           state_input=state["w1.weight"].shape[1] == 2 * channels,
                           rule=rule, fire_rate=fire_rate)
    else:
        model = CAModel(channels=state["conv.weight"].shape[0],
                        rule=rule, fire_rate=fire_rate)
    model.load_state_dict(state)
    return model
