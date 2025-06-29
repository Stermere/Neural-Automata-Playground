import { JSX, useEffect, useRef, useState } from 'react';
import styles from './app.module.css';
import { WebGPUNeuralAutomataController, AutomataConfig } from './widgets/WebGPUAutomataWidget.ts';
import WeightEditor from './widgets/WeightEditor.tsx';
import { defaultWeights } from './constants/defaultWeights.ts';

const SHADER_PATH = '/src/shaders/';
const SIZE: [number, number] = [1024, 1024];

export default function WebGPUNeuralAutomata(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<WebGPUNeuralAutomataController | null>(null);
  const initialized = useRef(false);

  const [weights, setWeights] = useState(defaultWeights);

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const config: AutomataConfig = {
      canvas,
      shaderPath: SHADER_PATH,
      gridSize: SIZE,
      brushRadius: 20,
    };

    const controller = new WebGPUNeuralAutomataController(config);
    controllerRef.current = controller;

    controller.init().then(() => {
      const flatWeights = defaultWeights.flat(3);
      controller.updateWeights(flatWeights);
    }).catch(console.error);

    return () => {
      controllerRef.current?.destroy();
    };
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
      <WeightEditor
        initialWeights={weights}
        onChange={handleWeightChange}
      />
    </div>
  );
}
