import { useState, useEffect, useMemo } from 'react';
import styles from './styles/activationEditor.module.css';

interface ActivationEditorProps {
  code: string;
  normalize: boolean;
  onCodeChange: (params: { code: string; normalize: boolean }) => void;
  presets?: Record<string, string>;
}

export default function ActivationEditor({ code, normalize, onCodeChange, presets = {} }: ActivationEditorProps) {
  const [editedCode, setEditedCode] = useState(code);
  const [normalizeState, setNormalizeState] = useState(normalize);
  const [selectedPreset, setSelectedPreset] = useState('');

  const presetEntries = useMemo(() => {
    const entries = Object.entries(presets);
    const matched = entries.find(([_, presetCode]) => presetCode.trim() === code.trim());
    if (!matched) {
      return [['Currently Active', code], ...entries];
    }
    return entries;
  }, [presets, code]);

  useEffect(() => {
    setEditedCode(code);
  }, [code]);

  useEffect(() => {
    setNormalizeState(normalize);
  }, [normalize]);

  const onApply = () => {
    onCodeChange({
      code: editedCode,
      normalize: normalizeState,
    });
  };

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value;
    setSelectedPreset(selected);
    const presetCode = presetEntries.find(([name]) => name === selected)?.[1];
    if (presetCode !== undefined) {
      setEditedCode(presetCode);
    }
    setSelectedPreset('')
  };

  return (
    <div className={styles.container}>
      <label className={styles.label} htmlFor="presetSelector">
        Load Preset:
      </label>
      <select
        id="presetSelector"
        className={styles.select}
        value={selectedPreset}
        onChange={handlePresetChange}
      >
        <option value="" disabled>Select preset...</option>
        {presetEntries.map(([name]) => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>

      <label className={styles.label} htmlFor="activationCode">
        Activation Function (WGSL):
      </label>
      <textarea
        id="activationCode"
        className={styles.textarea}
        value={editedCode}
        onChange={e => setEditedCode(e.target.value)}
        spellCheck={false}
      />
      <div className={styles.buttonRow}>
        <button className={styles.button} onClick={onApply}>
          Apply Changes
        </button>
        <label className={styles.checkboxLabel}>
          <input
            type="checkbox"
            className={styles.checkboxInput}
            checked={normalizeState}
            onChange={e => setNormalizeState(e.target.checked)}
          />
          Pre-normalize input by sum of weights
        </label>
      </div>
    </div>
  );
}