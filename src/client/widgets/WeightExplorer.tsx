import { useEffect, useRef, useState } from 'react';
import styles from './styles/weightExplorer.module.css';
import { WeightExplorerController } from '../controllers/WeightExplorerController';

interface WeightExplorerProps {
  controller: WeightExplorerController | null;
  updateWeights: (updatedWeights: number[][][][]) => void;
  setWeights: (updatedWeights: number[][][][]) => void
}

export default function WeightExplorer({ controller, updateWeights, setWeights }: WeightExplorerProps) {
  const [running, setRunning] = useState(false);
  const [amplitudeMax, setAmplitudeMax] = useState(1);
  const [freqMin, setFreqMin] = useState(0.0);
  const [freqMax, setFreqMax] = useState(2.0);
  const [speed, setSpeed] = useState(0.5);

  const [direction, setDirection] = useState<1 | -1>(1);

  const WINDOW_SECONDS = 20;
  const [scrubTime, setScrubTime] = useState<number>(0);
  const scrubTimeRef = useRef<number>(scrubTime);
  useEffect(() => { scrubTimeRef.current = scrubTime; }, [scrubTime]);

  const viewCenterRef = useRef<number>(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const startWindowMinRef = useRef<number | null>(null);
  const lastTimeMsRef = useRef<number | null>(null);
  const timeAccumRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

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
  }, [speed, controller]);

  useEffect(() => {
    if (!controller) return;

    if (!running) {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTimeMsRef.current = null;
      return;
    }

    const tick = (nowMs: number) => {
      if (lastTimeMsRef.current == null) lastTimeMsRef.current = nowMs;
      const deltaSec = Math.max(0, (nowMs - lastTimeMsRef.current) / 1000);
      lastTimeMsRef.current = nowMs;

      const dir = direction;
      timeAccumRef.current += dir * deltaSec * (speed || 1);

      if (!isScrubbing) {
        viewCenterRef.current = timeAccumRef.current;
        setScrubTime(timeAccumRef.current);
      }

      const weights = controller.weightsAtTime(timeAccumRef.current);
      updateWeights(weights);

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [controller, running, direction, speed, isScrubbing, updateWeights]);

  useEffect(() => {
    if (!controller) return;
    const weights = controller.weightsAtTime(scrubTime);
    updateWeights(weights);
  }, [scrubTime, controller, updateWeights]);

  const handleStart = () => {
    if (!controller?.init) {
      controller?.initRandom({ amplitudeMax, freqMin, freqMax, globalSpeed: speed / 5 });
    }
    timeAccumRef.current = scrubTime;
    viewCenterRef.current = scrubTime;
    lastTimeMsRef.current = null;
    setRunning(true);
  };

  const handleStop = () => {
    setRunning(false);
    const weights = controller?.weightsAtTime(timeAccumRef.current)
    if (weights) setWeights(weights)
  }

  useEffect(() => {
    return () => {
      setRunning(false);
      const weights = controller?.weightsAtTime(timeAccumRef.current);
      if (weights) setWeights(weights);
    };
  }, [controller, setWeights]);

  const toggleDirection = () => setDirection((d) => (d === 1 ? -1 : 1));

  const computeWindowBounds = () => {
    const center = viewCenterRef.current;
    const min = center - WINDOW_SECONDS / 2;
    const max = center + WINDOW_SECONDS / 2;
    return { min, max };
  };
  const { min: sliderMin, max: sliderMax } = computeWindowBounds();

  const ticks: number[] = [];
  const startTick = Math.floor(sliderMin);
  const endTick = Math.ceil(sliderMax);
  for (let t = startTick; t <= endTick; t++) ticks.push(t);

  const thumbPercent = ((scrubTime - sliderMin) / (sliderMax - sliderMin)) * 100;

  const pointerEventToTime = (clientX: number) => {
    const el = timelineRef.current;
    if (!el) return scrubTimeRef.current;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedX = Math.min(rect.width, Math.max(0, x));
    const ratio = rect.width > 0 ? clampedX / rect.width : 0;

    if (isScrubbing && startWindowMinRef.current != null) {
      const min = startWindowMinRef.current;
      const max = min + WINDOW_SECONDS;
      return min + ratio * (max - min);
    }

    const min = viewCenterRef.current - WINDOW_SECONDS / 2;
    const max = viewCenterRef.current + WINDOW_SECONDS / 2;
    return min + ratio * (max - min);
  };

  const onTimelinePointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    e.preventDefault();

    viewCenterRef.current = scrubTimeRef.current;
    startWindowMinRef.current = viewCenterRef.current - WINDOW_SECONDS / 2;

    setIsScrubbing(true);
    const newT = pointerEventToTime(e.clientX);
    setScrubTime(newT);
  };

  useEffect(() => {
    if (!isScrubbing) {
      startWindowMinRef.current = null;
      return;
    }

    const onMove = (ev: PointerEvent) => {
      const newT = pointerEventToTime(ev.clientX);
      setScrubTime(newT);
    };

    const onUp = (ev: PointerEvent) => {
      const newT = pointerEventToTime(ev.clientX);
      setScrubTime(newT);
      setIsScrubbing(false);
      timeAccumRef.current = newT;
      viewCenterRef.current = newT;
      startWindowMinRef.current = null;
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);

    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      startWindowMinRef.current = null;
    };
  }, [isScrubbing]);

  const onTickClick = (t: number) => {
    setScrubTime(t);
    timeAccumRef.current = t;
    viewCenterRef.current = t;
  };

  return (
    <div className={styles.container}>
      <div className={styles.note}>Search through latent space</div>

      <div className={styles.controlsHeader}>
        <button className={styles.btn} onClick={running ? handleStop : handleStart} disabled={!controller}>
          {running ? 'Stop' : 'Start'}
        </button>

        <div className={styles.timelineWrapper}>
          <button
            className={styles.dirBtn}
            onClick={toggleDirection}
            aria-label="Toggle playback direction"
            title={direction === 1 ? 'Play forward' : 'Play backward'}
          >
            {direction === 1 ? '⟳' : '⟲'}
          </button>

          <div className={styles.timeline}>
            <div
              ref={timelineRef}
              className={styles.timelineTrack}
              onPointerDown={onTimelinePointerDown}
              role="slider"
              aria-valuemin={sliderMin}
              aria-valuemax={sliderMax}
              aria-valuenow={scrubTime}
              tabIndex={0}
            >
              {/* ticks */}
              <div className={styles.ticksContainer}>
                {ticks.map((t) => {
                  const percent = ((t - sliderMin) / (sliderMax - sliderMin)) * 100;
                  return (
                    <div
                      key={t}
                      className={styles.tick}
                      style={{ left: `${percent}%` }}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        const newT = t;
                        setScrubTime(newT);
                        timeAccumRef.current = newT;
                        viewCenterRef.current = newT;
                      }}
                    >
                      <div className={styles.tickMark} />
                    </div>
                  );
                })}
              </div>

              <div
                className={styles.thumb}
                style={{ left: `${thumbPercent}%` }}
                aria-hidden
              />

              <div className={styles.timeReadout}>{scrubTime.toFixed(2)}s</div>
            </div>
          </div>
        </div>
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
