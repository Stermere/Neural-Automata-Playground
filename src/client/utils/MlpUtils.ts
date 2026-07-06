// Per-cell MLP of a trained kernel: w1 -> ReLU -> w2, optionally -> ReLU ->
// w3, applied by the compute shader to the 5x5 conv's per-channel results
// before the activation. The conv weights stay live (and editable) — the
// MLP is an extra nonlinear stage between them and the update rule. Kernels
// trained with stateInput also feed the cell's own raw state to w1, as a
// second block of columns after the conv results; w1's row width is the
// authoritative signal (channels = conv-only, 2*channels = stateInput).
// hiddenDim2/w3/b3 are absent for kernels trained with the original
// single-hidden-layer MLP (w2 is then the output layer); when present, w2
// is an interior hidden layer of width hiddenDim2 and w3 is the output layer.
export type MlpConfig = {
  hiddenDim: number;
  hiddenDim2?: number;
  w1: number[][]; // [hiddenDim][channels or 2*channels]
  b1: number[];   // [hiddenDim]
  w2: number[][]; // [channels or hiddenDim2][hiddenDim]
  b2: number[];   // [channels or hiddenDim2]
  w3?: number[][]; // [channels][hiddenDim2], present iff hiddenDim2
  b3?: number[];   // [channels], present iff hiddenDim2
  stateInput?: boolean; // informational; w1's row width decides
};

export class MlpUtils {
  // Total channel count implied by the MLP's output layer
  static channelCount(mlp: MlpConfig): number {
    return mlp.w3 ? mlp.w3.length : mlp.w2.length;
  }

  // Width of w1's input — what the shader templates as MLP_IN
  static inputDim(mlp: MlpConfig): number {
    return mlp.w1[0].length;
  }

  static isValid(mlp: unknown): mlp is MlpConfig {
    const m = mlp as MlpConfig;
    if (!m || !Number.isInteger(m.hiddenDim) || m.hiddenDim <= 0 ||
        !Array.isArray(m.w1) || m.w1.length !== m.hiddenDim ||
        !Array.isArray(m.b1) || m.b1.length !== m.hiddenDim ||
        !Array.isArray(m.w2) || m.w2.length === 0 ||
        !Array.isArray(m.b2) || m.b2.length !== m.w2.length ||
        !Array.isArray(m.w1[0])) {
      return false;
    }
    const channels = this.channelCount(m);
    const inputMatchesChannels = m.w1[0].length === channels || m.w1[0].length === 2 * channels;
    if (!m.hiddenDim2) {
      return inputMatchesChannels;
    }
    return Number.isInteger(m.hiddenDim2) && m.hiddenDim2 > 0 &&
      m.w2.length === m.hiddenDim2 &&
      inputMatchesChannels &&
      Array.isArray(m.w3) && m.w3.length > 0 &&
      Array.isArray(m.w3[0]) && m.w3[0].length === m.hiddenDim2 &&
      Array.isArray(m.b3) && m.b3.length === m.w3.length;
  }

  // Flat buffer layout must match compute.wgsl's MLP offsets:
  // [w1][b1][w2][b2] plus [w3][b3] when hiddenDim2 is set
  static flatten(mlp: MlpConfig): Float32Array {
    const tail = mlp.hiddenDim2 ? [...(mlp.w3 as number[][]).flat(), ...(mlp.b3 as number[])] : [];
    return new Float32Array([...mlp.w1.flat(), ...mlp.b1, ...mlp.w2.flat(), ...mlp.b2, ...tail]);
  }
}
