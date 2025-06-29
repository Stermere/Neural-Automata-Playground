import React from 'react';
import styles from './canvasControl.module.css';

interface CanvasControlProps {
  onClear: (() => void) | undefined;
  onRandomize: (() => void) | undefined;
  disabled?: boolean;
}

export default function CanvasControl({
  onClear,
  onRandomize,
  disabled = false,
}: CanvasControlProps) {
  return (
    <div className={styles.controlContainer}>
      <button
        className={styles.btn}
        onClick={onClear}
        disabled={disabled}
      >
        Clear Screen
      </button>
      <button
        className={styles.btn}
        onClick={onRandomize}
        disabled={disabled}
      >
        Randomize
      </button>
    </div>
  );
}
