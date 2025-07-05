import { JSX, useEffect, useRef, useState } from 'react';
import styles from './app.module.css';
import { WebGPUNeuralAutomataController, AutomataConfig } from './controllers/WebGPUAutomataController.ts';
import WeightEditor from './widgets/WeightEditor.tsx';
import CanvasControl from './widgets/CanvasControl.tsx';
import NeuralAutomataConfig from './widgets/NeuralAutomataConfig.tsx';
import ContentSwitcher from './widgets/ContentSwitcher.tsx';
import ActivationEditor from './widgets/ActivationEditor.tsx';
import { DefaultConfigController } from './controllers/DefaultConfigController.ts';
import { LOCAL_STORAGE_CONFIG_NAME } from './constants/filenameConstants.ts';
import { BASE_ACTIVATIONS } from './constants/baseActivations.ts';
import NeuralAutomataIntroduction from './widgets/NeuralAutomataIntroduction.tsx';

const SIZE: [number, number] = [1024, 1024];

export default function WebGPUNeuralAutomata(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<WebGPUNeuralAutomataController | null>(null);
  const initialized = useRef(false);
  const initialConfig = DefaultConfigController.getDefault()

  const [weights, setWeights] = useState(initialConfig.weights);
  const [activationCode, setActivationCode] = useState(initialConfig.activationCode);

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

    controller.init().then(async () => {
      const flatWeights = initialConfig.weights.flat(3);
      const activation = initialConfig.activationCode
      controller.updateWeights(flatWeights);
      controller.setActivationFunction(activation)
      controller.randomizeCanvas();
    }).catch(console.error);

    return () => {
      controllerRef.current?.destroy();
    };
  }, []);

  useEffect(() => {
    for (const [name, config] of Object.entries(DefaultConfigController.configMap)) {
      localStorage.setItem(`${LOCAL_STORAGE_CONFIG_NAME}${name}`, JSON.stringify(config));
    }
  }, []);

  const handleConfigLoad = (updatedWeights: number[][][][], updatedActivation: string) => {
    handleWeightChange(updatedWeights);
    handleActivationChange(updatedActivation);
  };

  const handleWeightChange = (updatedWeights: number[][][][]) => {
    setWeights(updatedWeights);
    controllerRef.current?.updateWeights(updatedWeights.flat(3));
  };

  const handleActivationChange = (updatedActivationCode: string) => {
    setActivationCode(updatedActivationCode)
    controllerRef.current?.setActivationFunction(updatedActivationCode)
  }

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
        <NeuralAutomataConfig weights={weights} activationCode={activationCode} onLoad={handleConfigLoad} />

        <ContentSwitcher labels={['Explanation', 'Weight Editor', 'Activation Editor']}>
          <NeuralAutomataIntroduction />
          <WeightEditor
            weights={weights}
            onWeightUpdate={handleWeightChange}
          />
          <ActivationEditor
            code={activationCode}
            onCodeChange={handleActivationChange}
            presets={BASE_ACTIVATIONS}
          />
        </ContentSwitcher>
      </div>
    </div>
  );
}
