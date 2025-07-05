import styles from './styles/canvasControl.module.css';

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
      <h2 className={styles.label}>Canvas Controls</h2>
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
