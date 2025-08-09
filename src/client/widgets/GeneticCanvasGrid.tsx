import { useEffect, useRef, forwardRef, useImperativeHandle, useState } from 'react';
import { WebGPUNeuralAutomataController, AutomataConfig } from '../controllers/WebGPUAutomataController';
import { ActivationVariableUtils, VariableValue } from '../utils/ActivationVariableUtils';
import styles from './styles/geneticCanvasGrid.module.css';
import { GeneticCandidate, GeneticEvolutionController } from '../controllers/GeneticEvolutionController';

export interface GeneticCanvasGridRef {
  updateAllPanels: (candidates: GeneticCandidate[], code?: string, normalize?: boolean, computeKernel?: boolean) => void;
}

interface GeneticCanvasGridProps {
  activationCode: string;
  normalize: boolean;
  geneticEvolutionControllerRef: React.RefObject<GeneticEvolutionController | null>
}

export const GeneticCanvasGrid = forwardRef<GeneticCanvasGridRef, GeneticCanvasGridProps>(
  ({ activationCode, normalize, geneticEvolutionControllerRef }, ref) => {
    const canvasRefs = useRef<(HTMLCanvasElement | null)[]>([null, null, null, null]);
    const controllerRefs = useRef<(WebGPUNeuralAutomataController | null)[]>([null, null, null, null]);
    const [translation, setTranslation] = useState({ x: 0, y: 0 });
    const isDragging = useRef(false);
    const dragStart = useRef({ x: 0, y: 0 });

    useImperativeHandle(ref, () => ({
      updateAllPanels: (candidates: GeneticCandidate[], code?: string, normalize?: boolean, computeKernel?: boolean) => {
        if (candidates.length !== 4) {
          console.error('Expected exactly 4 candidates');
          return;
        }

        candidates.forEach((candidate, index) => {
          const controller = controllerRefs.current[index];
          if (controller) {
            controller.updateWeights(candidate.weights.flat(3));
            (normalize !== undefined) ? controller.setActivationParameters(normalize, computeKernel) : null;
            controller.setActivationFunctionCode(ActivationVariableUtils.transformActivationCode(code ?? activationCode, candidate.activationVariables));
            controller.randomizeCanvas();
          }
        });
      }
    }));

    useEffect(() => {
      let isMounted = true;
      
      const initializeControllers = async () => {
        const canidates = geneticEvolutionControllerRef.current?.presentNext();
        if (!canidates) return;

        for (let i = 0; i < 4; i++) {
          if (!isMounted) {
            return;
          }
          
          const canvas = canvasRefs.current[i];
          if (!canvas) {
            continue;
          }

          const config: AutomataConfig = {
            canvas,
            gridSize: [512, 512],
            brushRadius: 20,
            isDraggable: false,
          };

          const controller = new WebGPUNeuralAutomataController(config);
          const canidate = canidates[i];
          
          try {
            await controller.init();
            
            if (!isMounted) {
              controller.destroy();
              return;
            }
            
            controllerRefs.current[i] = controller;
            controller.setActivationParameters(normalize);
            controller.updateWeights(canidate.weights.flat(3));
            controller.setActivationFunctionCode(ActivationVariableUtils.transformActivationCode(activationCode, canidate.activationVariables));
            controller.randomizeCanvas();
          } catch (error) {
            console.error(`Failed to initialize controller ${i}:`, error);
            controller?.destroy();
          }
        }
      };

      initializeControllers();

      return () => {
        isMounted = false;
        for (let i = 0; i < controllerRefs.current.length; i++) {
          const controller = controllerRefs.current[i];
          if (controller) {
            try {
              controller.destroy();
            } catch (error) {
              console.error(`Error destroying controller ${i}:`, error);
            }
            controllerRefs.current[i] = null;
          }
        }
      };
    }, []);

    const handleMouseDown = (e: React.MouseEvent) => {
      if (e.button !== 2) return;
      isDragging.current = true;
      dragStart.current = { x: e.clientX - translation.x, y: e.clientY - translation.y };
    };

    const handleMouseMove = (e: React.MouseEvent) => {
      if (isDragging.current) {
        setTranslation({
          x: e.clientX - dragStart.current.x,
          y: e.clientY - dragStart.current.y,
        });
      }
    };

    const handleMouseUp = () => {
      isDragging.current = false;
    };

    return (
      <div
        className={styles.canvas}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          transform: `translate(${translation.x}px, ${translation.y}px) scale(${1})`,
        }}
      >
        <div className={styles.gridContainer}>
          {[0, 1, 2, 3].map((index) => (
            <canvas
              key={index}
              ref={(el) => {
                canvasRefs.current[index] = el;
              }}
              width={512}
              height={512}
              className={styles.gridCanvas}
            />
          ))}
        </div>
      </div>
    );
  }
);