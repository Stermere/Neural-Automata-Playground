import React, { useState } from 'react';
import styles from './weightEditor.module.css';
import WeightConfig from './WeightConfig';

type Weights3D = number[][][][];

interface WeightEditorProps {
  initialWeights: Weights3D;
  onChange?: (weights: Weights3D) => void;
}

const channelLabels = ['R', 'G', 'B'];

export default function WeightEditor({ initialWeights, onChange }: WeightEditorProps) {
  const [weights, setWeights] = useState<Weights3D>(initialWeights);
  const [copySrc, setCopySrc] = useState<string>('0-0');

  const notifyChange = (newW: Weights3D) => {
    setWeights(newW);
    onChange?.(newW);
  };

  const handleChange = (
    outIdx: number,
    inIdx: number,
    row: number,
    col: number,
    value: number
  ) => {
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx][row][col] = value;
    notifyChange(newWeights);
  };

  const handleClear = (outIdx: number, inIdx: number) => {
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx] = newWeights[outIdx][inIdx].map(r => r.map(_ => 0));
    notifyChange(newWeights);
  };

  const handleNoise = (outIdx: number, inIdx: number) => {
    const newWeights = structuredClone(weights);
    newWeights[outIdx][inIdx] = newWeights[outIdx][inIdx].map(r =>
      r.map(v => v + (Math.random() * 2.0 - 1.0))
    );
    notifyChange(newWeights);
  };

  const handleCopy = (destKey: string) => {
    const [sOut, sIn] = copySrc.split('-').map(Number);
    const [dOut, dIn] = destKey.split('-').map(Number);
    if (copySrc === destKey) return;
    const newWeights = structuredClone(weights);
    newWeights[dOut][dIn] = structuredClone(newWeights[sOut][sIn]);
    notifyChange(newWeights);
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
            <h2 className={styles.label}>Kernel {channelLabels[outIdx]}</h2>
            <div className={styles.inputGroup}>
              {outputGroup.map((kernel, inIdx) => {
                const destKey = `${outIdx}-${inIdx}`;
                return (
                  <div key={inIdx} className={styles.kernelWrapper}>
                    <div className={styles.kernelHeader}>
                      <h3 className={styles.label}>Channel {channelLabels[inIdx]}</h3>
                      <div className={styles.kernelActions}>
                        <button
                          className={styles.btn}
                          onClick={() => handleClear(outIdx, inIdx)}
                        >
                          Clear
                        </button>
                        <button
                          className={styles.btn}
                          onClick={() => handleNoise(outIdx, inIdx)}
                        >
                          Noise
                        </button>
                        <button
                          className={styles.btn}
                          onClick={() => handleCopy(destKey)}
                        >
                          Copy Here
                        </button>
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
                              value={val}
                              className={styles.weightInput}
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
        <div className={styles.controlContainer}>
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
          </div>
          <WeightConfig weights={weights} onLoad={notifyChange} />
        </div>
      </div>
    </div>
  );
}
