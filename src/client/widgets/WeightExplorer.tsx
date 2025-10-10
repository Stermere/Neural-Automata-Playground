import { useEffect, useRef, useState } from 'react';
import styles from './styles/weightExplorer.module.css';
import { WeightExplorerController } from '../controllers/WeightExplorerController';

interface WeightExplorerProps {
  controller: WeightExplorerController | null;
  updateWeights: (updatedWeights: number[][][][]) => void;
}

export default function WeightExplorer({ controller, updateWeights }: WeightExplorerProps) {
  const [running, setRunning] = useState(false);
  const [amplitudeMax, setAmplitudeMax] = useState(1);
  const [freqMin, setFreqMin] = useState(0.0);
  const [freqMax, setFreqMax] = useState(2.0);
  const [speed, setSpeed] = useState(0.5);
  const [reverse, setReverse] = useState(false);
  const lastTimeMsRef = useRef<number | null>(null);
  const timeAccumRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!controller) return;
    if (!running) {
      controller.initRandom({ amplitudeMax, freqMin, freqMax, globalSpeed: speed / 5 });
    }
  }, [amplitudeMax, freqMin, freqMax]);

  useEffect(() => {
    if (!controller) return;
    controller.setConfig({ amplitudeMax, freqMin, freqMax, globalSpeed: speed / 5 });
    if (running) {
      controller.initRandom({ amplitudeMax, freqMin, freqMax, globalSpeed: speed / 5 });
      lastTimeMsRef.current = null;
    }
  }, [amplitudeMax, freqMin, freqMax]);

  useEffect(() => {
    if (!controller) return;
    controller.setSpeed(speed);
  }, [speed]);

  useEffect(() => {
    if (!controller) return;
    if (!running) {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTimeMsRef.current = null;
      return;
    }

    // Animation loop: compute time in seconds, request new weights, push to parent
    const tick = (nowMs: number) => {
      if (lastTimeMsRef.current == null) lastTimeMsRef.current = nowMs;
      const deltaSec = Math.max(0, (nowMs - lastTimeMsRef.current) / 1000);
      lastTimeMsRef.current = nowMs;
      const dir = reverse ? -1 : 1;
      timeAccumRef.current += dir * deltaSec;
      const weights = controller.weightsAtTime(timeAccumRef.current);
      updateWeights(weights);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [controller, running, reverse, updateWeights]);

  const handleStart = () => {
    if (!controller?.init) {
      controller?.initRandom({ amplitudeMax, freqMin, freqMax, globalSpeed: speed / 5 });
    }

    setRunning(true);
  };

  const handleStop = () => setRunning(false);

  return (
    <div className={styles.container}>
      <div className={styles.note}>Search through latent space</div>

      <div className={styles.controlsHeader}>
        <button className={styles.btn} onClick={running ? handleStop : handleStart} disabled={!controller}>
          {running ? 'Stop' : 'Start'}
        </button>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={reverse}
            onChange={(e) => setReverse(e.target.checked)}
          />
          Reverse Time
        </label>
      </div>

      <div className={styles.outputs}>
        <div className={styles.output}>
          <label className={styles.label} htmlFor="slider-amplitude">Amplitude Max: {amplitudeMax.toFixed(2)}</label>
          <input
            id="slider-amplitude"
            className={styles.slider}
            type="range"
            min={0}
            max={10}
            step={0.01}
            value={amplitudeMax}
            onChange={(e) => setAmplitudeMax(parseFloat(e.target.value))}
          />
        </div>

        <div className={styles.output}>
          <label className={styles.label} htmlFor="slider-freqmin">Freq Min (Hz): {freqMin.toFixed(2)}</label>
          <input
            id="slider-freqmin"
            className={styles.slider}
            type="range"
            min={0}
            max={5}
            step={0.01}
            value={Math.min(freqMin, freqMax)}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setFreqMin(Math.min(v, freqMax));
            }}
          />
        </div>

        <div className={styles.output}>
          <label className={styles.label} htmlFor="slider-freqmax">Freq Max (Hz): {freqMax.toFixed(2)}</label>
          <input
            id="slider-freqmax"
            className={styles.slider}
            type="range"
            min={0}
            max={5}
            step={0.01}
            value={Math.max(freqMax, freqMin)}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setFreqMax(Math.max(v, freqMin));
            }}
          />
        </div>

        <div className={styles.output}>
          <label className={styles.label} htmlFor="slider-speed">Speed: {speed.toFixed(2)}x</label>
          <input
            id="slider-speed"
            className={styles.slider}
            type="range"
            min={0.01}
            max={1}
            step={0.01}
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  );
}
