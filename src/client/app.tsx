import { JSX, useEffect, useRef, useState } from 'react';
import styles from './app.module.css';
import { WebGPUNeuralAutomataController, AutomataConfig } from './widgets/WebGPUAutomataController.ts';
import WeightEditor from './widgets/WeightEditor.tsx';
import CanvasControl from './widgets/CanvasControl.tsx';
import { wave, gameOfLife, worms, matrix, organicMatrix, fire, neonWave, galacticGoo } from './constants/defaultWeights.ts';

const SIZE: [number, number] = [1024, 1024];

export default function WebGPUNeuralAutomata(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<WebGPUNeuralAutomataController | null>(null);
  const initialized = useRef(false);

  const [weights, setWeights] = useState(wave);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const config: AutomataConfig = {
      canvas,
      gridSize: SIZE,
      brushRadius: 20,
    };

    const controller = new WebGPUNeuralAutomataController(config);
    controllerRef.current = controller;

    controller.init().then(() => {
      const flatWeights = wave.flat(3);
      controller.updateWeights(flatWeights);
    }).catch(console.error);

    return () => {
      controllerRef.current?.destroy();
    };
  }, []);

  useEffect(() => {
    localStorage.setItem('weights:wave', JSON.stringify(wave));
    localStorage.setItem('weights:worms', JSON.stringify(worms));
    localStorage.setItem('weights:matrix', JSON.stringify(matrix));
    localStorage.setItem('weights:organicMatrix', JSON.stringify(organicMatrix));
    localStorage.setItem('weights:fire', JSON.stringify(fire));
    localStorage.setItem('weights:neonWave', JSON.stringify(neonWave));
    localStorage.setItem('weights:galacticGoo', JSON.stringify(galacticGoo));
    localStorage.setItem('weights:gameOfLife', JSON.stringify(gameOfLife));
  }, []);

  const handleWeightChange = (updatedWeights: number[][][][]) => {
    setWeights(updatedWeights);
    const flatWeights = updatedWeights.flat(3);
    controllerRef.current?.updateWeights(flatWeights);
  };

  return (
    <div className={styles.canvasContainer}>
      <canvas
        ref={canvasRef}
        width={SIZE[0]}
        height={SIZE[1]}
        className={styles.canvas}
      />
      <div className={styles.controlContainer}>
        <CanvasControl
          onClear={() => controllerRef.current?.clearCanvas()}
          onRandomize={() => controllerRef.current?.randomizeCanvas()}
        />
        <WeightEditor
          initialWeights={weights}
          onChange={handleWeightChange}
        />
        {/* TODO create an activation function editor + config */}
      </div>
    </div>
  );
}
