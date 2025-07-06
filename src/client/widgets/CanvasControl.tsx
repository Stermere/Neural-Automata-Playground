import { useState } from 'react';
import styles from './styles/canvasControl.module.css';

interface CanvasControlProps {
  onClear: (() => void) | undefined;
  onRandomize: (() => void) | undefined;
  onFpsChange: ((fps: number) => void) | undefined;
  onPause: ((paused: boolean) => void) | undefined;
  disabled?: boolean;
  initialFps?: number;
  initialPaused?: boolean;
}

export default function CanvasControl({
  onClear,
  onRandomize,
  onFpsChange,
  onPause,
  disabled = false,
  initialFps = 120,
  initialPaused = false,
}: CanvasControlProps) {
  const [fps, setFps] = useState(initialFps);
  const [isPaused, setIsPaused] = useState(initialPaused);

  const handleFpsChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newFps = parseInt(event.target.value);
    setFps(newFps);
    onFpsChange?.(newFps);
  };

  const handlePauseToggle = () => {
    const newPaused = !isPaused;
    setIsPaused(newPaused);
    onPause?.(newPaused);
  };

  return (
    <div className={styles.controlContainer}>
      <h2 className={styles.label}>Canvas Controls</h2>
      
      <div className={styles.controlGroup}>
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
        
        <button
          className={`${styles.btn} ${isPaused ? styles.paused : styles.playing}`}
          onClick={handlePauseToggle}
          disabled={disabled}
        >
          {isPaused ? 'Resume' : 'Pause'}
        </button>
        <label className={styles.label} htmlFor="fps-slider">
          Fps Target: {fps}
        </label>
        <input
          id="fps-slider"
          type="range"
          min="1"
          max="240"
          value={fps}
          onChange={handleFpsChange}
          disabled={disabled}
          className={styles.fpsSlider}
        />
      </div>
    </div>
  );
}