export const BASE_ACTIVATIONS: Record<string, string> = {
  "Linear": `// f(x) = x
fn activation(x: f32, weightSum: f32) -> f32 {
  return x;
}`,

  "Exponential Linear Unit": `// f(x) = x ≥ 0 ? x : e^x - 1
fn activation(x: f32, weightSum: f32) -> f32 {
  return select(exp(x) - 1.0, x, x >= 0.0);
}`,

  "Binary Spike Detector": `// f(x) = (x = 3 or x = 11 or x = 12) ? 1 : 0
fn activation(x: f32, weightSum: f32) -> f32 {
  if (x == 3.0 || x == 11.0 || x == 12.0) {
    return 1.0;
  }
  return 0.0;
}`,

  "Softsign": `// f(x) = x / (1 + |x|)
fn activation(x: f32, weightSum: f32) -> f32 {
  return x / (1.0 + abs(x));
}`,

  "Inverse Gaussian": `// f(x) = -1 / (2^(0.6·x²)) + 1
fn activation(x: f32, weightSum: f32) -> f32 {
  return -1.0 / pow(2.0, 0.6 * x * x) + 1.0;
}`,

  "Tanh": `// f(x) = tanh(x)
fn activation(x: f32, weightSum: f32) -> f32 {
  let e1 = exp(x);
  let e2 = exp(-x);
  return (e1 - e2) / (e1 + e2);
}`,

  "Sigmoid": `// f(x) = 1 / (1 + e^(-x))
fn activation(x: f32, weightSum: f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}`,

  "Step Function": `// f(x) = x > 0 ? 1 : 0
fn activation(x: f32, weightSum: f32) -> f32 {
  return select(0.0, 1.0, x > 0.0);
}`,

  "Sigmoid-Gated Linear Unit (Swish)": `// f(x) = x / (1 + e^(-x))
fn activation(x: f32, weightSum: f32) -> f32 {
  return x / (1.0 + exp(-x));
}`,

  "Smooth Self-Gated Activation (Mish)": `// f(x) = x * tanh(ln(1 + e^x))
fn activation(x: f32, weightSum: f32) -> f32 {
  let softplus = log(1.0 + exp(x));
  return x * tanh(softplus);
}`,

  "Absolute Value Spike": `// f(x) = 1 - |x|
fn activation(x: f32, weightSum: f32) -> f32 {
  return 1.0 - abs(x);
}`,
};
