import React, { useState, useRef, useEffect } from 'react';
import styles from './styles/neuralAutomataConfig.module.css';
import { DEFAULT_EXPORT_NAME, LOCAL_STORAGE_CONFIG_NAME } from '../constants/filenameConstants';

interface NeuralAutomataConfig {
  weights: number[][][][];
  activationCode: string;
  normalize: boolean;
  onLoad: (weights: number[][][][], activationCode: { code: string; normalize: boolean }) => void;
}

export default function NeuralAutomataConfig({ weights, activationCode, normalize, onLoad }: NeuralAutomataConfig) {
  const [filename, setFilename] = useState('');
  const [selected, setSelected] = useState('');
  const [savedFiles, setSavedFiles] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const existing = Object.keys(localStorage).filter(key => key.startsWith(LOCAL_STORAGE_CONFIG_NAME)).map(k => k.slice(LOCAL_STORAGE_CONFIG_NAME.length));
    setSavedFiles(existing);
  }, []);

  const handleSave = () => {
    let baseFilename = (filename.length === 0) ? DEFAULT_EXPORT_NAME : filename;
    let writenFilename = baseFilename;
    let counter = 1;

    // Check for existing filename and append a number if needed
    while (savedFiles.includes(writenFilename)) {
      writenFilename = `${baseFilename} (${counter})`;
      counter++;
    }

    const data = {
      weights,
      activationCode,
      normalize,
    };

    localStorage.setItem(`${LOCAL_STORAGE_CONFIG_NAME}${writenFilename}`, JSON.stringify(data));
    setSavedFiles(prev => [...prev, writenFilename]);
  };

  const handleLoad = (name: string) => {
    setSelected("")
    const data = localStorage.getItem(`${LOCAL_STORAGE_CONFIG_NAME}${name}`);
    if (!data) return;

    try {
      const parsed = JSON.parse(data);
      onLoad(parsed.weights, { code: parsed.activationCode, normalize: parsed.normalize });
      setFilename(name);
    } catch (err) {
      console.error('Failed to parse weights:', err);
    }
  };

  const handleDelete = () => {
    if (!filename || !savedFiles.includes(filename)) return;
    localStorage.removeItem(`${LOCAL_STORAGE_CONFIG_NAME}${filename}`);
    setSavedFiles(prev => prev.filter(name => name !== filename));
    setFilename('');
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        onLoad(data.weights, { code: data.activationCode, normalize: data.normalize });
      } catch (err) {
        console.error('Failed to load weights:', err);
      }
    };
    reader.readAsText(file);
  };

  const handleExport = () => {
    const blob = new Blob(
      [JSON.stringify({ weights, activationCode, normalize }, null, 2)],
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
      <h2 className={styles.label}>Profile Managment</h2>
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
        <button
          className={styles.btn}
          onClick={handleDelete}
        >
          Delete
        </button>
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