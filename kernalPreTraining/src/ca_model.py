"""The NCA model and everything that must stay in lockstep with the shader:
the update-rule math, the exported WGSL activation code, browser-exact
quantization, and the stochastic-update gate."""
import json
import os

import torch
import torch.nn as nn

# ======================================================
# Update rules: each entry pairs the Python step math with the WGSL
# update expression exported to the playground, so they cannot diverge.
# In Python `conv` is the Conv2d output and already contains the bias;
# the shader's convX is the raw weighted sum, so NCA_BIAS is added
# explicitly in the WGSL. The exported activation wraps the expression
# in the per-channel delta/bias arrays, the stochastic-update gate
# (when fire_rate < 1), and clamp(..., 0, 1).
# ======================================================
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


# ======================================================
# Model: single 5x5 conv over all channels, mirroring the compute shader
# (wrap-around sampling, every cell sampled every step). Beyond the conv
# it carries a per-channel bias, a per-channel learnable update rate
# (delta), and a stochastic update gate — all reproduced in the exported
# activation code, none of them need shader changes.
# ======================================================
class CAModel(nn.Module):
    def __init__(self, channels=11, kernel_size=5, delta=0.25, rule="tanh", fire_rate=0.5):
        super().__init__()
        self.channels = channels
        self.rule = rule
        self.fire_rate = fire_rate
        self.bestEval = None
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              padding=kernel_size // 2, padding_mode='circular', bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # per-channel update rate: lets slow memory channels and fast visible
        # channels coexist; exported as the NCA_DELTA array
        self.delta = nn.Parameter(torch.full((channels,), float(delta)))

    def step(self, x, update_mask=None):
        """One CA step. update_mask ([B or 1, 1, H, W] of 0/1) selects which
        cells apply their update; None samples a Bernoulli(fire_rate) mask,
        matching the browser's hash gate statistically. Pass
        wgsl_gate_mask(...) instead to match the browser exactly."""
        rule = UPDATE_RULES[self.rule]
        delta = self.delta.view(1, -1, 1, 1)
        pre = rule["step"](x, self.conv(x), delta)
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

    def exportToPlaygroundFormat(self, save_dir, filepath="TrainedWeights.json"):
        weights = self.conv.weight.detach().cpu().tolist()  # [out][in][5][5] — the shader's layout
        assert len(weights) == self.channels and len(weights[0]) == self.channels

        if self.fire_rate < 1.0:
            gate_decl = WGSL_GATE.format(fire_rate=repr(float(self.fire_rate)))
            gate_check = "  if (!ncaGate()) {\n    return lastX;\n  }\n"
        else:
            gate_decl = ""
            gate_check = ""
        activation_code = WGSL_ACTIVATION.format(
            channels=self.channels,
            deltas=_fmt_f32_list(self.delta.detach().cpu().tolist()),
            biases=_fmt_f32_list(self.conv.bias.detach().cpu().tolist()),
            gate_decl=gate_decl,
            gate_check=gate_check,
            update=UPDATE_RULES[self.rule]["wgsl_update"],
        )

        export_dict = {
            "weights": weights,
            "activationCode": activation_code,
            "normalize": False
        }

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
