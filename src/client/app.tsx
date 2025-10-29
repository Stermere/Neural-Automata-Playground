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
import { ActivationVariableUtils, VariableValue } from './utils/ActivationVariableUtils.ts';
import WeightExplorer from './widgets/WeightExplorer.tsx';
import { WeightExplorerController, Weights3D } from './controllers/WeightExplorerController.ts';


const params = new URLSearchParams(window.location.search);
const width = parseInt(params.get("width") ?? "") || 1024;
const height = parseInt(params.get("height") ?? "") || 1024;

const SIZE: [number, number] = [width, height];

export default function WebGPUNeuralAutomata(): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<WebGPUNeuralAutomataController | null>(null);
  const weightExplorerRef = useRef<WeightExplorerController | null>(null)
  const initialized = useRef(false);
  const initialConfig = DefaultConfigController.getConfig(DEFAULT_CONFIG)
  const [weights, setWeights] = useState(initialConfig.weights);
  const [activationCode, setActivationCode] = useState(initialConfig.activationCode);
  const [activationVariables, setActivationVariables] = useState(ActivationVariableUtils.getDefaultVariableValues(ActivationVariableUtils.getVariables(initialConfig.activationCode)));
  const [normalizeInputToActivation, setNormalize] = useState(initialConfig.normalize);
  const [zoom, setZoom] = useState(1);
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const [webgpuError, setWebgpuError] = useState<string | null>(null);
  const [controlsVisible, setControlsVisible] = useState(true);

  useEffect(() => {
    if (initialized.current || webgpuError) return;
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
    
    const weightExplorer = new WeightExplorerController()
    weightExplorerRef.current = weightExplorer;
    weightExplorer.initRandom();
    weightExplorer.updateWeights(initialConfig.weights);
    
    controller.init().then(async () => {
      handleWeightChange(initialConfig.weights)
      controllerRef.current?.setActivationParameters(initialConfig.normalize);
      controllerRef.current?.setActivationFunctionCode(ActivationVariableUtils.transformActivationCodeDefault(initialConfig.activationCode));
      controllerRef.current?.randomizeCanvas();
    }).catch((err) => {
      console.error(err);
      setWebgpuError('Failed to initialize WebGPU: ' + (err?.message || String(err)));
    });
  }, [webgpuError]);

  useEffect(() => {
    const configKeys = Object.keys(DefaultConfigController.configMap)
      .map(name => `${LOCAL_STORAGE_CONFIG_NAME}${name}`);
      
    const allExist = configKeys.every(key => localStorage.getItem(key) !== null);

    if (!allExist) {
      for (const [name, config] of Object.entries(DefaultConfigController.configMap)) {
        localStorage.setItem(`${LOCAL_STORAGE_CONFIG_NAME}${name}`, JSON.stringify(config));
      }

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
      const newZoom = (temp => Math.abs(temp - 1) <= 0.1 ? 1 : temp)(Math.max(0.5, Math.min(15, zoom * zoomFactor)));
            
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

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'F4') {
        e.preventDefault();
        setControlsVisible(!controlsVisible);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [controlsVisible]);

  const handleConfigLoad = (updatedWeights: number[][][][], updatedActivation: { code: string; normalize: boolean, computeKernel: boolean }) => {
    const activationVariables = ActivationVariableUtils.getDefaultVariableValues(ActivationVariableUtils.getVariables(updatedActivation.code))
    setActivationVariables(activationVariables);
    handleWeightChange(updatedWeights);
    handleActivationChange(updatedActivation, activationVariables);
  };

  const handleWeightChange = (updatedWeights: number[][][][]) => {
    setWeights(updatedWeights);
    controllerRef.current?.updateWeights(updatedWeights.flat(3));
    weightExplorerRef.current?.updateWeights(updatedWeights);
  };

  const handleActivationChange = (updatedActivation: { code: string; normalize: boolean, computeKernel?: boolean }, activationVariablesConfig?: VariableValue[]) => {
    setActivationCode(updatedActivation.code);
    setNormalize(updatedActivation.normalize);
    controllerRef.current?.setActivationParameters(updatedActivation.normalize, updatedActivation.computeKernel);
    controllerRef.current?.setActivationFunctionCode(ActivationVariableUtils.transformActivationCode(updatedActivation.code, activationVariablesConfig ?? activationVariables));
  }

  const handleApplyActivationVariableChange = (code: string) => {
    controllerRef.current?.setActivationFunctionCode(code);
  }

  const handleSetActivationVariables = (values: VariableValue[]) => {
    setActivationVariables(values);
  }

  const handleZoomChange = (newZoom: number) => {
    setZoom(newZoom);
    updateCanvasZoom(newZoom);
  };

  const updateCanvasZoom = (zoomLevel: number, transformTranslation?: string) => {
    controllerRef.current?.setZoom(zoomLevel, transformTranslation);
  };

  return (
    <div className={styles.canvasContainer}>
      {webgpuError ? (
        <div
          className={styles.canvas}
          style={{ width: SIZE[0], height: SIZE[1], display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#111', color: '#fff', padding: '16px', boxSizing: 'border-box' }}
        >
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ margin: '0 0 8px' }}>WebGPU Unavailable have you enabled it?</h3>
            <p style={{ margin: 0 }}>{webgpuError}</p>
          </div>
        </div>
      ) : (
        <canvas
          ref={canvasRef}
          width={SIZE[0]}
          height={SIZE[1]}
          className={styles.canvas}
          style={{ width: SIZE[0], height: SIZE[1] }}
        />
      )}
      <div className={styles.controlContainer} style={{ display: controlsVisible ? 'block' : 'none' }}>
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
          activationVariables={activationVariables}
          onLoad={handleConfigLoad} 
        />

        <ContentSwitcher labels={['Explanation', 'Weight Editor', 'Activation Editor', 'Variable Editor', 'Weight Explorer', 'None']} setSelectedIndex={setSelectedTabIndex}>
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
            setValues={handleSetActivationVariables}
            onVariableChange={handleApplyActivationVariableChange}
          />
          <WeightExplorer
            controller={weightExplorerRef.current}
            updateWeights={(modifiedWeights: Weights3D) => controllerRef.current?.updateWeights(modifiedWeights.flat(3))}
          />
        </ContentSwitcher>
      </div>
    </div>
  );
}
