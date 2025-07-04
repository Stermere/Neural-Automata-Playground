import { useState, useEffect } from 'react';
import styles from './styles/activationEditor.module.css';

interface ActivationEditorProps {
  code: string;
  onCodeChange: (newCode: string) => void;
}

export default function ActivationEditor({ code, onCodeChange }: ActivationEditorProps) {
  const [editedCode, setEditedCode] = useState(code);

  useEffect(() => {
    setEditedCode(code);
  }, [code]);

  const onApply = () => {
    onCodeChange(editedCode);
  };

  return (
    <div className={styles.container}>
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
