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
  const [normalizeInputToActivation, setNormalize] = useState(initialConfig.normalize);
  const [zoom, setZoom] = useState(1);

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
      controller.updateWeights(initialConfig.weights.flat(3));
      controller.setActivationFunction({
        code: initialConfig.activationCode,
        normalize: initialConfig.normalize
      });
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

  useEffect(() => {
    const handleResize = () => {
      updateCanvasZoom(zoom);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [zoom]);

  const handleConfigLoad = (updatedWeights: number[][][][], updatedActivation: { code: string; normalize: boolean }) => {
    handleWeightChange(updatedWeights);
    handleActivationChange(updatedActivation);
  };

  const handleWeightChange = (updatedWeights: number[][][][]) => {
    setWeights(updatedWeights);
    controllerRef.current?.updateWeights(updatedWeights.flat(3));
  };

  const handleActivationChange = (updatedActivation: { code: string; normalize: boolean }) => {
    setActivationCode(updatedActivation.code);
    setNormalize(updatedActivation.normalize);
    controllerRef.current?.setActivationFunction(updatedActivation);
  }

  const handleZoomChange = (newZoom: number) => {
  setZoom(newZoom);
  updateCanvasZoom(newZoom);
};

  const updateCanvasZoom = (zoomLevel: number) => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.style.transform = `scale(${zoomLevel})`;
      const isMobile = window.innerWidth <= 1910;
      if (isMobile) {
        canvas.style.transformOrigin = 'top center';
      } else {
        canvas.style.transformOrigin = 'top left';
      }
    }
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
          onFpsChange={(fps: number) => controllerRef.current?.setMaxFps(fps)}
          onPause={(pause: boolean) => controllerRef.current?.togglePaused(pause)}
          onZoomChange={(zoom: number) => handleZoomChange(zoom)}
        />
        <NeuralAutomataConfig weights={weights} activationCode={activationCode} normalize={normalizeInputToActivation} onLoad={handleConfigLoad} />

        <ContentSwitcher labels={['Explanation', 'Weight Editor', 'Activation Editor']}>
          <NeuralAutomataIntroduction />
          <WeightEditor
            weights={weights}
            onWeightUpdate={handleWeightChange}
          />
          <ActivationEditor
            code={activationCode}
            normalize={normalizeInputToActivation}
            onCodeChange={handleActivationChange}
            presets={BASE_ACTIVATIONS}
          />
        </ContentSwitcher>
      </div>
    </div>
  );
}
