import { cloneDeep } from 'lodash';
import { VISIBLE_CHANNELS, KERNEL_SIZE } from '../constants/channelConstants';

export type Weights3D = number[][][][]; // [out][in][row][col]

export type WaveConfig = {
  amplitudeMax: number;
  freqMin: number;
  freqMax: number;
  globalSpeed: number;
  continuityStrength: number;
  continuityDrift: number;
  continuityCenterPull: number;
};

export class WeightExplorerController {
  private amplitudes: Weights3D | null = null;
  private freqs: Weights3D | null = null;
  private currentPhases: Weights3D | null = null;
  private continuity: Weights3D | null = null;
  private weights: Weights3D | null = null;
  private lastTime = 0;

  private config: WaveConfig = {
    amplitudeMax: 0.5,
    freqMin: 0.1,
    freqMax: 1.0,
    globalSpeed: 1.0,
    continuityStrength: 0.6,
    continuityDrift: 0.1,
    continuityCenterPull: 0.01,
  };

  init = false;
  
  public updateWeights(weights: Weights3D) {
    this.weights = weights;
  }

  private mkZero(out: number, input: number, size: number): Weights3D {
    return Array.from({ length: out }, () =>
      Array.from({ length: input }, () =>
        Array.from({ length: size }, () => Array.from({ length: size }, () => 0))
      )
    );
  }

  // Dimensions follow the shape of the current weights (channel count varies)
  private currentDims(): { out: number; input: number; size: number } {
    const out = this.weights?.length ?? VISIBLE_CHANNELS;
    const input = this.weights?.[0]?.length ?? out;
    const size = this.weights?.[0]?.[0]?.length ?? KERNEL_SIZE;
    return { out, input, size };
  }

  initRandom(config?: Partial<WaveConfig>, dims?: { out: number; input: number; size: number }): void {
    this.config = { ...this.config, ...(config ?? {}) };
    const { out, input, size } = dims ?? this.currentDims();

    this.amplitudes = this.mkZero(out, input, size);
    this.freqs = this.mkZero(out, input, size);
    this.currentPhases = this.mkZero(out, input, size);
    this.continuity = this.mkZero(out, input, size);

    for (let o = 0; o < out; o++) {
      for (let i = 0; i < input; i++) {
        for (let r = 0; r < size; r++) {
          for (let c = 0; c < size; c++) {
            this.amplitudes[o][i][r][c] = Math.random() * this.config.amplitudeMax;
            const fRange = Math.max(0, this.config.freqMax - this.config.freqMin);
            this.freqs[o][i][r][c] = this.config.freqMin + Math.random() * fRange;
            this.currentPhases[o][i][r][c] = Math.random() * Math.PI * 2;
            this.continuity[o][i][r][c] = Math.random();
          }
        }
      }
    }

    this.lastTime = 0;
    this.init = true;
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

  private clamp(x: number, min = 0, max = 1): number {
    return Math.max(min, Math.min(max, x));
  }

  weightsAtTime(tSeconds: number): Weights3D {
    if (!this.weights || !this.currentPhases || !this.continuity)
      throw new Error("Controller not initialized");

    const { out, input, size } = this.currentDims();

    // Re-roll the wave arrays if the weight shape changed since init
    if (this.amplitudes!.length !== out || this.amplitudes![0].length !== input) {
      this.initRandom();
      this.lastTime = tSeconds; // avoid a discontinuity jump
    }

    const w: Weights3D = cloneDeep(this.weights);

    const dt = (tSeconds - this.lastTime) * this.config.globalSpeed;
    this.lastTime = tSeconds;

    const drift = this.config.continuityDrift * dt;
    const strength = this.config.continuityStrength;
    const pull = this.config.continuityCenterPull;

    for (let o = 0; o < out; o++) {
      for (let i = 0; i < input; i++) {
        for (let r = 0; r < size; r++) {
          for (let c = 0; c < size; c++) {
            const A = this.amplitudes![o][i][r][c];
            const f = this.freqs![o][i][r][c];

            let cont = this.continuity[o][i][r][c]!;

            // random drift + mean reversion
            const noise = (Math.random() * 2 - 1) * drift;
            const restore = (0.5 - cont) * pull * dt;
            cont += (noise + restore) * strength;
            cont = this.clamp(cont);

            this.continuity[o][i][r][c] = cont;

            // advance phase proportional to continuity
            this.currentPhases[o][i][r][c]! += 2 * Math.PI * f * cont * dt;
            if (this.currentPhases[o][i][r][c]! > Math.PI * 1e6)
              this.currentPhases[o][i][r][c]! %= 2 * Math.PI;

            w[o][i][r][c] += A * Math.sin(this.currentPhases[o][i][r][c]!);
          }
        }
      }
    }

    return w;
  }
}