// Per-cell MLP of a trained kernel: w1 -> ReLU -> w2, applied by the compute
// shader to the 5x5 conv's per-channel results before the activation. The
// conv weights stay live (and editable) — the MLP is an extra nonlinear
// stage between them and the update rule. Kernels trained with stateInput
// also feed the cell's own raw state to w1, as a second block of columns
// after the conv results; w1's row width is the authoritative signal
// (channels = conv-only, 2*channels = stateInput).
export type MlpConfig = {
  hiddenDim: number;
  w1: number[][]; // [hiddenDim][channels or 2*channels]
  b1: number[];   // [hiddenDim]
  w2: number[][]; // [channels][hiddenDim]
  b2: number[];   // [channels]
  stateInput?: boolean; // informational; w1's row width decides
};

export class MlpUtils {
  // Total channel count implied by the MLP's output layer
  static channelCount(mlp: MlpConfig): number {
    return mlp.w2.length;
  }

  // Width of w1's input — what the shader templates as MLP_IN
  static inputDim(mlp: MlpConfig): number {
    return mlp.w1[0].length;
  }

  static isValid(mlp: unknown): mlp is MlpConfig {
    const m = mlp as MlpConfig;
    return !!m && Number.isInteger(m.hiddenDim) && m.hiddenDim > 0 &&
      Array.isArray(m.w1) && m.w1.length === m.hiddenDim &&
      Array.isArray(m.b1) && m.b1.length === m.hiddenDim &&
      Array.isArray(m.w2) && m.w2.length > 0 &&
      Array.isArray(m.b2) && m.b2.length === m.w2.length &&
      Array.isArray(m.w1[0]) &&
      (m.w1[0].length === m.w2.length || m.w1[0].length === 2 * m.w2.length);
  }

  // Flat buffer layout must match compute.wgsl's MLP offsets: [w1][b1][w2][b2]
  static flatten(mlp: MlpConfig): Float32Array {
    return new Float32Array([...mlp.w1.flat(), ...mlp.b1, ...mlp.w2.flat(), ...mlp.b2]);
  }
}
