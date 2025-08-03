import { useState } from 'react';
import styles from './styles/geneticEditor.module.css';
import { GeneticCandidate, GeneticEvolutionController } from '../controllers/GeneticEvolutionController';
import { VariableValue } from '../utils/ActivationVariableUtils';


interface GeneticEditorProps {
  controller: GeneticEvolutionController | null;
  showMutation: (updatedWeights: number[][][][], activationVariables: VariableValue[]) => void;
  showMutations: (candidates: GeneticCandidate[]) => void;
}

export default function GeneticEditor({ controller, showMutation, showMutations }: GeneticEditorProps) {
  const [sliderValue, setSliderValue] = useState(-1.0);
  const mutationRate = Math.pow(10, sliderValue);
  const [candidates, setCandidates] = useState<GeneticCandidate[]>([]);

  const submitChoice = async (choiceIndex: number) => {
    if (!controller) return;
    await controller.submitChoice(choiceIndex - 1);
    const newCandidates = controller.presentNext();
    if (candidates.length > 0) showMutation(candidates[choiceIndex - 1].weights, candidates[choiceIndex - 1].activationVariables);
    setCandidates(newCandidates);
    showMutations(newCandidates);
  };

  const startFromRandom = () => {
    if (!controller) return;
    controller.init();
    const candidates = controller.generateRandomGeneration();
    setCandidates(candidates);
    showMutations(candidates);
  };

  const ratingButtons = Array.from({ length: 4 }, (_, i) => i + 1);

  return (
    <div className={styles.container}>
      <h3>Chose your favorite</h3>
      <p>{controller?.getStatus()}</p>
      <div className={styles.ratingButtons}>
        {ratingButtons.map((value) => (
          <button
            key={value}
            className={styles.ratingButton}
            onClick={() => submitChoice(value)}
          >
            {value}
          </button>
        ))}
      </div>

      <div className={styles.controls}>
        <label>
          Mutation Rate:
          <span className={styles.mutationRateDisplay}>
            {mutationRate.toPrecision(2)}
          </span>
          <input
            className={styles.slider}
            type="range"
            min={-1.5}
            max={1.5}
            step={0.01}
            value={sliderValue}
            onChange={(e) => {
              const logRate = parseFloat(e.target.value);
              setSliderValue(logRate);
              controller?.setMutationRate(Math.pow(10, logRate));
            }}
          />
        </label>

        <button className={styles.controlButton} onClick={startFromRandom}>
          Start From Random
        </button>
      </div>
    </div>
  );
}