import React, { useState, useRef, useEffect } from 'react';
import styles from './styles/neuralAutomataConfig.module.css';

interface NeuralAutomataConfig {
  weights: number[][][][];
  activationCode: string
  onLoad: (weights: number[][][][], activationCode: string) => void;
}

export default function NeuralAutomataConfig({ weights, activationCode, onLoad }: NeuralAutomataConfig) {
  const [filename, setFilename] = useState('');
  const [selected, setSelected] = useState("");
  const [savedFiles, setSavedFiles] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const existing = Object.keys(localStorage).filter(key => key.startsWith('weights:')).map(k => k.slice(8));
    setSavedFiles(existing);
  }, []);

  const handleSave = () => {
    const writenFilename = (filename.length === 0) ? 'NoName' : filename;
    const data = {
      weights,
      activationCode,
    };
    localStorage.setItem(`weights:${writenFilename}`, JSON.stringify(data));
    if (!savedFiles.includes(writenFilename)) {
      setSavedFiles(prev => [...prev, writenFilename]);
    }
  };

  const handleLoad = (name: string) => {
    setSelected("");
    const data = localStorage.getItem(`weights:${name}`);
    if (!data) return;

    try {
      const parsed = JSON.parse(data);
      if (Array.isArray(parsed)) {
        // legacy support: just weights TODO migrate defaults
        onLoad(parsed, activationCode);
      } else {
        onLoad(parsed.weights, parsed.activationCode || '');
      }
    } catch (err) {
      console.error('Failed to parse weights:', err);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        if (Array.isArray(data)) {
          onLoad(data, activationCode);
        } else {
          onLoad(data.weights, data.activationCode || '');
        }
      } catch (err) {
        console.error('Failed to load weights:', err);
      }
    };
    reader.readAsText(file);
  };

  const handleExport = () => {
    const blob = new Blob(
      [JSON.stringify({ weights, activationCode }, null, 2)],
      { type: 'application/json' }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename || 'config'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={styles.container}>
      <input
        className={styles.input}
        value={filename}
        onChange={e => setFilename(e.target.value)}
        placeholder="Enter filename"
      />
      <div className={styles.actionsRow}>
        <button className={styles.btn} onClick={handleSave}>Save</button>
        <button className={styles.btn} onClick={handleExport}>Export</button>
        <button className={styles.btn} onClick={() => fileInputRef.current?.click()}>import</button>
      </div>
      <input
        type="file"
        accept="application/json"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileUpload}
      />
      <select className={styles.select} onClick={e => handleLoad((e.target as HTMLSelectElement).value)} onChange={e => handleLoad(e.target.value)} value={selected}>
        <option value="" disabled>Select file to load</option>
        {savedFiles.map(name => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>
    </div>
  );
}