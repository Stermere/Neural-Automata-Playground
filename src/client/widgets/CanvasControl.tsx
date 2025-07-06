import { useState } from 'react';
import styles from './styles/canvasControl.module.css';

interface CanvasControlProps {
  onClear: (() => void) | undefined;
  onRandomize: (() => void) | undefined;
  onFpsChange: ((fps: number) => void) | undefined;
  onPause: ((paused: boolean) => void) | undefined;
  onZoomChange: ((zoom: number) => void) | undefined;
  disabled?: boolean;
  initialFps?: number;
  initialPaused?: boolean;
  initialZoom?: number;
}

export default function CanvasControl({
  onClear,
  onRandomize,
  onFpsChange,
  onPause,
  onZoomChange,
  disabled = false,
  initialFps = 120,
  initialPaused = false,
  initialZoom = 1.0,
}: CanvasControlProps) {
  const [fps, setFps] = useState(initialFps);
  const [isPaused, setIsPaused] = useState(initialPaused);
  const [zoom, setZoom] = useState(initialZoom);

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

  const handleZoomChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newZoom = parseFloat(event.target.value);
    setZoom(newZoom);
    onZoomChange?.(newZoom);
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
          className={styles.slider}
        />
        <label className={styles.label} htmlFor="zoom-slider">
          Zoom: {zoom.toFixed(2)}x
        </label>
        <input
          id="zoom-slider"
          type="range"
          min="1.0"
          max="10"
          step="0.25"
          value={zoom}
          onChange={handleZoomChange}
          disabled={disabled}
          className={styles.slider}
        />
      </div>
    </div>
  );
}