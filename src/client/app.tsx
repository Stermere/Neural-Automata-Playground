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
import { DEFAULT_CONFIG } from "./constants/filenameConstants";
import ActivationVariableEditor from './widgets/ActivationVariableEditor.tsx';
import { ActivationVariableUtils } from './utils/ActivationVariableUtils.ts';

const SIZE: [number, number] = [1024, 1024];

export default function WebGPUNeuralAutomata(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<WebGPUNeuralAutomataController | null>(null);
  const initialized = useRef(false);
  const initialConfig = DefaultConfigController.getDefault()

  const [weights, setWeights] = useState(initialConfig.weights);
  const [activationCode, setActivationCode] = useState(initialConfig.activationCode);
  const [activationVariables, setActivationVariables] = useState(ActivationVariableUtils.getDefaultVariableValues(ActivationVariableUtils.getVariables(initialConfig.activationCode)));
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
      controller.setActivationParameters(initialConfig.normalize);
      controller.setActivationFunctionCode(ActivationVariableUtils.transformActivationCodeDefault(initialConfig.activationCode));
      controller.randomizeCanvas();
    }).catch(console.error);

    return () => {
      controllerRef.current?.destroy();
    };
  }, []);

  useEffect(() => {
    const defaultKey = `${LOCAL_STORAGE_CONFIG_NAME}${DEFAULT_CONFIG}`;
    const firstRun = !localStorage.getItem(defaultKey);
    for (const [name, config] of Object.entries(DefaultConfigController.configMap)) {
      localStorage.setItem(`${LOCAL_STORAGE_CONFIG_NAME}${name}`, JSON.stringify(config));
    }
    
    if (firstRun) {
      window.location.reload();
    }
  }, []);

  useEffect(() => {
    const handleResize = () => {
      updateCanvasZoom(zoom);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [zoom]);

  // Allow zooming via scroll wheel
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let isZooming = false;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (isZooming) return;
      isZooming = true;
      
      const zoomSpeed = 0.2;
      const zoomFactor = e.deltaY > 0 ? (1 - zoomSpeed) : (1 + zoomSpeed);
      const newZoom = Math.max(0.1, Math.min(15, zoom * zoomFactor));
            
      if (newZoom !== zoom) {
        // Get the mouse position relative to the canvas
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Parse current translation from canvas transform style
        const getCurrentTranslation = () => {
          const transform = canvas.style.transform;
          const translateMatch = transform.match(/translate\(([^,]+)px,\s*([^)]+)px\)/);
          if (translateMatch) {
            return {
              x: parseFloat(translateMatch[1]),
              y: parseFloat(translateMatch[2])
            };
          }
          return { x: 0, y: 0 };
        };

        const currentTranslation = getCurrentTranslation();
        const mousePercentX = mouseX / rect.width;
        const mousePercentY = mouseY / rect.height;
        const deltaCanvasSizeX =  (newZoom * canvas.width) - (zoom * canvas.width);
        const deltaCanvasSizeY = (newZoom * canvas.height) - (zoom * canvas.height);
        const newTranslateX = currentTranslation.x - (deltaCanvasSizeX * mousePercentX);
        const newTranslateY = currentTranslation.y - (deltaCanvasSizeY * mousePercentY);
        
        setZoom(newZoom);
        updateCanvasZoom(newZoom, `translate(${newTranslateX}px, ${newTranslateY}px)`);

      }
      setTimeout(() => {
        isZooming = false;
      }, 16);
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [zoom]);

  const handleConfigLoad = (updatedWeights: number[][][][], updatedActivation: { code: string; normalize: boolean, computeKernel: boolean }) => {
    handleWeightChange(updatedWeights);
    handleActivationChange(updatedActivation);
  };

  const handleWeightChange = (updatedWeights: number[][][][]) => {
    setWeights(updatedWeights);
    controllerRef.current?.updateWeights(updatedWeights.flat(3));
  };

  const handleActivationChange = (updatedActivation: { code: string; normalize: boolean, computeKernel?: boolean }) => {
    setActivationCode(updatedActivation.code);
    setNormalize(updatedActivation.normalize);
    controllerRef.current?.setActivationParameters(updatedActivation.normalize, updatedActivation.computeKernel);
    controllerRef.current?.setActivationFunctionCode(ActivationVariableUtils.transformActivationCodeDefault(updatedActivation.code));
  }

  const handleActivationVariableChange = (code: string) => {
    controllerRef.current?.setActivationFunctionCode(code);
  }

  const handleZoomChange = (newZoom: number) => {
    setZoom(newZoom);
    updateCanvasZoom(newZoom);
  };

  const updateCanvasZoom = (zoomLevel: number, transformTranslation?: string) => {
    const canvas = canvasRef.current;
    if (canvas) {
      if (transformTranslation) {
        const currentTransform = canvas.style.transform || '';
        const transformWithoutTranslation = currentTransform.replace(/translate\([^)]*\)/g, '');
        canvas.style.transform  = `${transformWithoutTranslation} ${transformTranslation}`.trim();
      }

      const currentTransform = canvas.style.transform || '';
      const transformWithoutScale = currentTransform.replace(/scale\([^)]*\)/g, '');
      canvas.style.transform  = `${transformWithoutScale} scale(${zoomLevel})`.trim();
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
          onBrushSizeChange={(size: number) => controllerRef.current?.setBrushSize(size)}
          initialZoom={zoom}
        />
        <NeuralAutomataConfig 
          weights={weights}
          activationCode={activationCode} 
          normalize={normalizeInputToActivation}
          onLoad={handleConfigLoad} 
        />

        <ContentSwitcher labels={['Explanation', 'Weight Editor', 'Activation Editor', 'Variable Editor', 'None']}>
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
          <ActivationVariableEditor
            code={activationCode}
            values={activationVariables}
            setValues={setActivationVariables}
            onVariableChange={handleActivationVariableChange}
          />
        </ContentSwitcher>
      </div>
    </div>
  );
}
