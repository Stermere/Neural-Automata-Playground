import { cloneDeep } from 'lodash';

export type Weights3D = number[][][][]; // [out][in][row][col]

export type WaveConfig = {
  amplitudeMax: number; // max absolute amplitude for any weight
  freqMin: number; // in Hz
  freqMax: number; // in Hz
  globalSpeed: number; // multiplier for time
};

export class WeightExplorerController {
  public init = false;
  private amplitudes: Weights3D | null = null;
  private phases: Weights3D | null = null;
  private freqs: Weights3D | null = null;
  private weights: Weights3D | null = null;
  private config: WaveConfig = {
    amplitudeMax: 0.5,
    freqMin: 0.1,
    freqMax: 1.0,
    globalSpeed: 1.0,
  };

  public WeightExplorerController(weights: Weights3D) {
    this.weights = weights;
  }

  public updateWeights(weights: Weights3D) {
    this.weights = weights;
  }

  // Initializes random wave parameters per-weight
  initRandom(config?: Partial<WaveConfig>, dims: { out: number; input: number; size: number } = { out: 3, input: 3, size: 5 }): void {
    this.config = { ...this.config, ...(config ?? {}) };
    const { out, input, size } = dims;
    const mk = (): Weights3D =>
      Array.from({ length: out }, () =>
        Array.from({ length: input }, () =>
          Array.from({ length: size }, () => Array.from({ length: size }, () => 0))
        )
      );

    this.amplitudes = mk();
    this.phases = mk();
    this.freqs = mk();

    for (let o = 0; o < out; o++) {
      for (let i = 0; i < input; i++) {
        for (let r = 0; r < size; r++) {
          for (let c = 0; c < size; c++) {
            // amplitude in [0, amplitudeMax]
            this.amplitudes[o][i][r][c] = Math.random() * this.config.amplitudeMax;
            // phase in [0, 2π)
            this.phases[o][i][r][c] = Math.random() * Math.PI * 2;
            // frequency in [freqMin, freqMax]
            const fRange = Math.max(0, this.config.freqMax - this.config.freqMin);
            this.freqs[o][i][r][c] = this.config.freqMin + Math.random() * fRange;
          }
        }
      }
    }
  }

  setConfig(config: Partial<WaveConfig>): void {
    this.config = { ...this.config, ...config };
  }

  getConfig(): WaveConfig {
    return { ...this.config };
  }

  setSpeed(speed: number): void {
    this.config.globalSpeed = speed;
  }

  // Computes weights at time t (in seconds)
  weightsAtTime(tSeconds: number, dims: { out: number; input: number; size: number } = { out: 3, input: 3, size: 5 }): Weights3D {
    const { out, input, size } = dims;
    const w: Weights3D = cloneDeep(this.weights);

    const t = tSeconds * this.config.globalSpeed;
    for (let o = 0; o < out; o++) {
      for (let i = 0; i < input; i++) {
        for (let r = 0; r < size; r++) {
          for (let c = 0; c < size; c++) {
            const A = this.amplitudes![o][i][r][c];
            const ph = this.phases![o][i][r][c];
            const f = this.freqs![o][i][r][c];
            w[o][i][r][c] += (A * Math.sin(2 * Math.PI * f * t + ph));
          }
        }
      }
    }

    return w;
  }
}
