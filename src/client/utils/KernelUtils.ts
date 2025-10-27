const KERNEL_VARIATIONS = [
  [[1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
  ],
  [[0, 1, 0, 0, 0],
   [0, 1, 0, 0, 0],
   [0, 1, 0, 0, 0],
   [0, 1, 0, 0, 0],
   [0, 1, 0, 0, 0],
  ],
  [[0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
  ],
  [[1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 1, 1, 0, 0],
   [1, 1, 1, 0, 0],
   [1, 1, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 1, 1, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [0, 0, 0, 1, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 10, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 0, 0, 0, 0],
   [0, 1, 0, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 0, 0, 0, 0],
   [1, 1, 0, 0, 0],
   [1, 1, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 1, 1, 0, 0],
   [0, 1, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 1, 1, 0, 0],
   [0, 1, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 1, 0, 0, 0],
   [1, 1, 0, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 0, 1, 0],
   [0, 0, 0, 0, 1],
  ],
  [[0, 1, 1, 0, 0],
   [0, 1, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [1, 1, 1, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 5, 0, 0],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 0, 0, 0, 0],
   [1, 0, 0, 0, 0],
   [1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0],
  ],
  [[0, 0, 0, 0, 0],
   [0, 1, 1, 1, 0],
   [0, 1, -10, 1, 0],
   [0, 1, 1, 1, 0],
   [0, 0, 0, 0, 0],
  ],
  [[1, 1, 1, 1, 1],
   [1, 0, 0, 0, 1],
   [1, 0, -10, 0, 1],
   [1, 0, 0, 0, 1],
   [1, 1, 1, 1, 1],
  ],
  [[0, 0, 1, 0, 0],
   [0, 1, 0, 0, 0],
   [1, 0, 0, 0, 1],
   [1, 0, 0, 1, 0],
   [1, 1, 1, 0, 0],
  ],
  [[0, 0, 0, 0, 1],
   [0, 0, 1, 1, 0],
   [0, 1, 0, 1, 0],
   [1, 0, 1, 0, 0],
   [0, 1, 0, 0, 0],
  ]
];


export class KernelUtils {
  static readonly EXPECTED_COUNT = 2;
  static readonly MIN_WEIGHT = 0.2;
  static readonly MAX_WEIGHT = 1.0;

  // Returns a semi random kernal using the above kernel variants
  static getPartialKernelVariation(): number[][] {
      const totalVariations = this.kernelVariations.length;
      const probability = this.EXPECTED_COUNT / totalVariations;
      const sample = this.kernelVariations[0];
      const rows = sample.length;
      const cols = sample[0].length;
      const result: number[][] = Array(rows).fill(0).map(() => Array(cols).fill(0));

      for (const variation of this.kernelVariations) {
        if (Math.random() < probability) {
          const weight = Math.random() * (this.MAX_WEIGHT - this.MIN_WEIGHT) + this.MIN_WEIGHT;
          for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
              result[i][j] += weight * variation[i][j];
            }
          }
        }
      }
      return result;
  }

  static mirrorHorizontal(kernel: number[][]): number[][] {
    return kernel.map(row => [...row].reverse());
  }

  static mirrorVertical(kernel: number[][]): number[][] {
    return [...kernel].reverse();
  }

  static mirrorBoth(kernel: number[][]): number[][] {
    return this.mirrorHorizontal(this.mirrorVertical(kernel));
  }

  static negate(kernel: number[][]): number[][] {
    return kernel.map(row => row.map(val => -val));
  }

  static kernelVariations: number[][][] = (() => {
    const variations: number[][][] = [];
    for (const kernel of KERNEL_VARIATIONS) {
      const original = kernel;
      const hMirror = KernelUtils.mirrorHorizontal(original);
      const vMirror = KernelUtils.mirrorVertical(original);
      const bothMirror = KernelUtils.mirrorBoth(original);
      const mirrors = [original, hMirror, vMirror, bothMirror];
      const negates = mirrors.map(k => KernelUtils.negate(k));
      variations.push(...mirrors, ...negates);
    }
    return variations;
  })();
}
