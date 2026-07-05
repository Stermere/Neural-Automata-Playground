"""Parity check: reimplement the compute shader's math with plain loops
and compare against CAModel.step. Catches any weight-layout mistake
before hours of training."""
import random

import torch

from ca_model import CAModel, MLPCAModel, wgsl_gate_mask


def _scalar_gate(x, y, t, fire_rate):
    """Scalar mirror of the exported ncaGate() WGSL; cross-checks the
    vectorized wgsl_gate_mask before rollouts trust it."""
    m = 0xFFFFFFFF
    h = (x * 374761393 + y * 668265263 + t * 2246822519) & m
    h = ((h ^ (h >> 13)) * 1274126177) & m
    h = h ^ (h >> 16)
    gate = torch.tensor(float(h), dtype=torch.float32) / torch.tensor(4294967295.0, dtype=torch.float32)
    return bool((gate < fire_rate).item())


def verify_shader_parity(channels=11, size=12, rule="tanh", delta=0.25, fire_rate=0.5):
    model = CAModel(channels=channels, delta=delta, rule=rule, fire_rate=fire_rate)
    with torch.no_grad():
        model.conv.weight.uniform_(-0.5, 0.5)
        model.conv.bias.uniform_(-0.25, 0.25)
        model.delta.uniform_(0.05, 0.5)
    x = torch.rand(1, channels, size, size)
    timestep = random.randint(0, 100_000)
    mask = wgsl_gate_mask(size, size, timestep, fire_rate)

    if mask is not None:
        scalar_mask = torch.tensor([[1.0 if _scalar_gate(px, y, timestep, fire_rate) else 0.0
                                     for px in range(size)] for y in range(size)])
        if not torch.equal(mask.view(size, size), scalar_mask):
            raise AssertionError("Shader parity check FAILED: vectorized gate mask != scalar WGSL mirror")

    with torch.no_grad():
        got, _ = model.step(x, update_mask=mask)

    w = model.conv.weight
    k = w.shape[-1]
    half = k // 2
    expected = torch.empty_like(x)
    for out_ch in range(channels):
        delta_c = model.delta[out_ch].item()
        bias_c = model.conv.bias[out_ch].item()
        for y in range(size):
            for px in range(size):
                last = x[0, out_ch, y, px].item()
                if mask is not None and mask[0, 0, y, px].item() == 0.0:
                    expected[0, out_ch, y, px] = last
                    continue
                conv_x = 0.0
                for in_ch in range(channels):
                    for ky in range(-half, half + 1):
                        for kx in range(-half, half + 1):
                            weight = w[out_ch, in_ch, ky + half, kx + half].item()
                            state = x[0, in_ch, (y + ky) % size, (px + kx) % size].item()
                            conv_x += weight * state
                if rule == "linear":
                    pre = last + delta_c * (conv_x + bias_c)
                else:
                    pre = last + delta_c * torch.tanh(torch.tensor(conv_x + bias_c)).item()
                expected[0, out_ch, y, px] = min(max(pre, 0.0), 1.0)

    diff = (got - expected).abs().max().item()
    if diff > 1e-4:
        raise AssertionError(f"Shader parity check FAILED: max diff {diff:.6f}")
    print(f"Shader parity check passed (max diff {diff:.2e}).")


def verify_shader_parity_mlp(channels=11, hidden_dim=8, size=8, rule="tanh",
                             delta=0.25, fire_rate=0.5, state_input=True):
    """Plain-loop mirror of the compute shader's MLP path (the 5x5 wrapped
    conv, then the per-cell hidden ReLU layer and output layer applied to the
    conv results — plus, with state_input, the cell's own raw state as a
    second block of MLP inputs) against MLPCAModel.step. Catches any
    weight-layout mistake before training."""
    model = MLPCAModel(channels=channels, hidden_dim=hidden_dim, delta=delta,
                       rule=rule, fire_rate=fire_rate, state_input=state_input)
    with torch.no_grad():
        model.conv.weight.uniform_(-0.5, 0.5)
        model.w1.weight.uniform_(-0.5, 0.5)
        model.w1.bias.uniform_(-0.25, 0.25)
        model.w2.weight.uniform_(-0.5, 0.5)
        model.w2.bias.uniform_(-0.25, 0.25)
        model.delta.uniform_(0.05, 0.5)
    x = torch.rand(1, channels, size, size)
    timestep = random.randint(0, 100_000)
    mask = wgsl_gate_mask(size, size, timestep, fire_rate)

    with torch.no_grad():
        got, _ = model.step(x, update_mask=mask)

    w = model.conv.weight
    k = w.shape[-1]
    half = k // 2
    w1 = model.w1.weight.squeeze(-1).squeeze(-1)  # [hidden][channels or 2*channels]
    b1 = model.w1.bias
    w2 = model.w2.weight.squeeze(-1).squeeze(-1)  # [channels][hidden]
    b2 = model.w2.bias
    expected = torch.empty_like(x)
    for y in range(size):
        for px in range(size):
            if mask is not None and mask[0, 0, y, px].item() == 0.0:
                expected[0, :, y, px] = x[0, :, y, px]
                continue
            # the shader's conv loop: raw wrapped weighted sums, no bias
            sums = [0.0] * channels
            for out_ch in range(channels):
                for in_ch in range(channels):
                    for ky in range(-half, half + 1):
                        for kx in range(-half, half + 1):
                            weight = w[out_ch, in_ch, ky + half, kx + half].item()
                            state = x[0, in_ch, (y + ky) % size, (px + kx) % size].item()
                            sums[out_ch] += weight * state
            # the per-cell MLP inputs: conv outputs, then (with state_input)
            # the cell's raw state — the shader's mlpWeights column order
            inputs = list(sums)
            if state_input:
                inputs += [x[0, ch, y, px].item() for ch in range(channels)]
            hidden = [max(b1[j].item() + sum(w1[j, i].item() * inputs[i]
                                             for i in range(len(inputs))), 0.0)
                      for j in range(hidden_dim)]
            for out_ch in range(channels):
                conv_x = b2[out_ch].item() + sum(w2[out_ch, j].item() * hidden[j]
                                                 for j in range(hidden_dim))
                last = x[0, out_ch, y, px].item()
                delta_c = model.delta[out_ch].item()
                if rule == "linear":
                    pre = last + delta_c * conv_x
                else:
                    pre = last + delta_c * torch.tanh(torch.tensor(conv_x)).item()
                expected[0, out_ch, y, px] = min(max(pre, 0.0), 1.0)

    diff = (got - expected).abs().max().item()
    if diff > 1e-4:
        raise AssertionError(f"MLP shader parity check FAILED: max diff {diff:.6f}")
    print(f"MLP shader parity check passed (max diff {diff:.2e}).")
