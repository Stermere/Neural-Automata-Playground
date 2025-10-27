import { useState } from 'react';
import { KernelUtils } from '../utils/KernelUtils';
import styles from './styles/weightEditor.module.css';

type Weights3D = number[][][][];

interface WeightEditorProps {
  weights: Weights3D;
  onWeightUpdate: (weights: Weights3D) => void;
}

const channelLabels = ['Red', 'Green', 'Blue'];

export default function WeightEditor({ weights, onWeightUpdate }: WeightEditorProps) {
  const [copySrc, setCopySrc] = useState<string>('0-0');

  const [symmetrySettings, setSymmetrySettings] = useState<Record<string, {
    horizontal: boolean;
    vertical: boolean;
  }>>({});

  const handleChange = (
    outIdx: number,
    inIdx: number,
    row: number,
    col: number,
    value: number
  ) => {
    const key = `${outIdx}-${inIdx}`;
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx][row][col] = value;

    const symmetry = symmetrySettings[key];

    if (symmetry?.horizontal) {
      const mirrorRow = newWeights[outIdx][inIdx].length - 1 - row;
      newWeights[outIdx][inIdx][mirrorRow][col] = value;
    }

    if (symmetry?.vertical) {
      const mirrorCol = newWeights[outIdx][inIdx][row].length - 1 - col;
      newWeights[outIdx][inIdx][row][mirrorCol] = value;
    }

    if (symmetry?.vertical && symmetry?.horizontal) {
      const mirrorCol = newWeights[outIdx][inIdx][row].length - 1 - col;
      const mirrorRow = newWeights[outIdx][inIdx].length - 1 - row;
      newWeights[outIdx][inIdx][mirrorRow][mirrorCol] = value;
    }

    onWeightUpdate(newWeights);
  };

  const handleClear = (outIdx: number, inIdx: number) => {
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx] = newWeights[outIdx][inIdx].map(r => r.map(_ => 0));
    onWeightUpdate(newWeights);
  };

  const handleFullRandomize = () => {
    const newWeights = structuredClone(weights);
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
          newWeights[i][j] = KernelUtils.getPartialKernelVariation();
      }
    }
    onWeightUpdate(newWeights);
  }

  const handlePartialRandomize = (outIdx: number, inIdx: number) => {
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx] = KernelUtils.getPartialKernelVariation();
    onWeightUpdate(newWeights);
  };

  const handleCopy = (destKey: string) => {
    const [sOut, sIn] = copySrc.split('-').map(Number);
    const [dOut, dIn] = destKey.split('-').map(Number);
    if (copySrc === destKey) return;
    const newWeights = structuredClone(weights);
    newWeights[dOut][dIn] = structuredClone(newWeights[sOut][sIn]);
    onWeightUpdate(newWeights);
  };

  // prepare copy sources list
  const copyOptions = [] as { key: string; label: string }[];
  weights.forEach((grp, o) => grp.forEach((_, i) => {
    copyOptions.push({ key: `${o}-${i}`, label: `Kernel ${channelLabels[o]} Channel ${channelLabels[i]}` });
  }));

  return (
    <div>
      <div className={styles.editorContainer}>
        {weights.map((outputGroup, outIdx) => (
          <div key={outIdx} className={styles.outputGroup}>
            <h2 className={styles.label}>{channelLabels[outIdx]} Kernel</h2>
            <div className={styles.inputGroup}>
              {outputGroup.map((kernel, inIdx) => {
                const destKey = `${outIdx}-${inIdx}`;
                return (
                  <div key={inIdx} className={styles.kernelWrapper}>
                    <div className={styles.kernelHeader}>
                      <h3 className={styles.label}>{channelLabels[inIdx]} Channel</h3>
                      <div className={styles.kernelActions}>
                        <button
                          className={styles.btn}
                          onClick={() => handleClear(outIdx, inIdx)}
                        >
                          Clear
                        </button>
                        <button
                          className={styles.btn}
                          onClick={() => handlePartialRandomize(outIdx, inIdx)}
                        >
                          Rand
                        </button>
                        <button
                          className={styles.btn}
                          onClick={() => handleCopy(destKey)}
                        >
                          Copy Here
                        </button>
                        <label className={styles.checkboxLabel}>
                          <input
                            type="checkbox"
                            className={styles.checkboxInput}
                            checked={symmetrySettings[destKey]?.horizontal || false}
                            onChange={e =>
                              setSymmetrySettings(prev => ({
                                ...prev,
                                [destKey]: {
                                  ...prev[destKey],
                                  horizontal: e.target.checked,
                                },
                              }))
                            }
                          />
                          ↕︎
                        </label>
                        <label className={styles.checkboxLabel}>
                          <input
                            type="checkbox"
                            className={styles.checkboxInput}
                            checked={symmetrySettings[destKey]?.vertical || false}
                            onChange={e =>
                              setSymmetrySettings(prev => ({
                                ...prev,
                                [destKey]: {
                                  ...prev[destKey],
                                  vertical: e.target.checked,
                                },
                              }))
                            }
                          />
                          ↔︎
                        </label>
                      </div>
                    </div>
                    <div className={styles.kernelGrid}>
                      {kernel.map((row, rowIdx) => (
                        <div key={rowIdx} className={styles.kernelRow}>
                          {row.map((val, colIdx) => (
                            <input
                              key={colIdx}
                              type="number"
                              inputMode="decimal"
                              value={val.toFixed(2)}
                              className={styles.weightInput}
                              step={0.1}
                              onChange={e => {
                                const newVal = e.target.value;
                                if (newVal === '' || newVal === '-' || newVal === '.' || newVal === '-.') return;
                                const parsed = parseFloat(newVal);
                                if (!isNaN(parsed)) {
                                  handleChange(outIdx, inIdx, rowIdx, colIdx, parsed);
                                }}
                              }
                            />
                          ))}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
        <div className={styles.copyContainer}>
          <label className={styles.label} htmlFor="copySrc">Copy From:</label>
          <select
            id="copySrc"
            className={styles.select}
            value={copySrc}
            onChange={e => setCopySrc(e.target.value)}
          >
            {copyOptions.map(opt => (
              <option key={opt.key} value={opt.key}>{opt.label}</option>
            ))}
          </select>
          <button
            className={styles.btn}
            onClick={() => handleFullRandomize()}
          >
            Randomize All
          </button>
        </div>
      </div>
    </div>
  );
}
