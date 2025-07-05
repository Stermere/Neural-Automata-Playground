import { useState, useEffect, useMemo } from 'react';
import styles from './styles/activationEditor.module.css';

interface ActivationEditorProps {
  code: string;
  onCodeChange: (newCode: string) => void;
  presets?: Record<string, string>;
}

export default function ActivationEditor({ code, onCodeChange, presets = {} }: ActivationEditorProps) {
  const [editedCode, setEditedCode] = useState(code);
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

  const onApply = () => {
    onCodeChange(editedCode);
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
          Apply Code
        </button>
      </div>
    </div>
  );
}