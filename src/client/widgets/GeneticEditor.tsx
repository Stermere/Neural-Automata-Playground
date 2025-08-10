import { useEffect, useState } from 'react';
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

  const [mutationProb, setMutationProb] = useState(0.05);
  const [topN1, setTopN1] = useState(1);
  const [topN2, setTopN2] = useState(2); 
  const [crossoverRatio, setCrossoverRatio] = useState(0.9);

  const [mutateWeights, setMutateWeights] = useState(true);
  const [mutateVars, setMutateVars] = useState(false);
  const [crossoverWeights, setCrossoverWeights] = useState(true);
  const [crossoverVars, setCrossoverVars] = useState(false);

  const [candidates, setCandidates] = useState<GeneticCandidate[]>([]);

  useEffect(() => {
    if (!controller) return;
    controller.setMutationRate(Math.pow(10, sliderValue));
    controller.setMutationProbability(mutationProb);
    controller.setTopNParent1(topN1);
    controller.setTopNParent2(topN2);
    controller.setCrossoverRatio(crossoverRatio);

    controller.setMutateWeightsEnabled(mutateWeights);
    controller.setMutateVarsEnabled(mutateVars);
    controller.setCrossoverWeightsEnabled(crossoverWeights);
    controller.setCrossoverVarsEnabled(crossoverVars);
  }, [controller]);

  const submitChoice = async (choiceIndex: number) => {
    if (!controller) return;
    await controller.submitChoice(choiceIndex - 1);
    const newCandidates = controller.presentNext();
    if (candidates.length > 0)
      showMutation(
        candidates[choiceIndex - 1].weights,
        candidates[choiceIndex - 1].activationVariables
      );
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
      <div className={styles.editorGrid}>
        <div className={styles.leftColumn}>
          <h3>Choose your favorite to slowly evolve the pattern:</h3>
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
          <label>{controller?.getStatus()}</label>
          <div/>
          <button className={styles.controlButton} onClick={startFromRandom}>
            Start From Random
          </button>
        </div>

        <div className={styles.rightColumn}>
          <h3>Evolution Settings:</h3>
          <div className={styles.sliderGroup}>
            <label className={styles.label}>
              Mutation Rate:
              <span className={styles.mutationRateDisplay}>
                {mutationRate.toPrecision(2)}
              </span>
            </label>
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

            <label className={styles.label}>
              Mutation Probability:
              <span>{mutationProb.toFixed(2)}</span>
            </label>
            <input
              className={styles.slider}
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={mutationProb}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setMutationProb(val);
                controller?.setMutationProbability(val);
              }}
            />

            <label className={styles.label}>
              Top N Parent 1:
              <span>{topN1}</span>
            </label>
            <input
              className={styles.slider}
              type="range"
              min={1}
              max={controller ? controller["populationSize"] : 32}
              step={1}
              value={topN1}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setTopN1(val);
                controller?.setTopNParent1(val);
              }}
            />

            <label className={styles.label}>
              Top N Parent 2:
              <span>{topN2}</span>
            </label>
            <input
              className={styles.slider}
              type="range"
              min={1}
              max={controller ? controller["populationSize"] : 32}
              step={1}
              value={topN2}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setTopN2(val);
                controller?.setTopNParent2(val);
              }}
            />

            <label className={styles.label}>
              Crossover Ratio:
              <span>{crossoverRatio.toFixed(2)}</span>
            </label>
            <input
              className={styles.slider}
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={crossoverRatio}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setCrossoverRatio(val);
                controller?.setCrossoverRatio(val);
              }}
            />
          </div>

          <div className={styles.checkboxGroup}>
            <label className={styles.label}>
              <input
                type="checkbox"
                checked={mutateWeights}
                onChange={(e) => {
                  setMutateWeights(e.target.checked);
                  controller?.setMutateWeightsEnabled(e.target.checked);
                }}
              />
              Enable Weight Mutation
            </label>
            <label className={styles.label}>
              <input
                type="checkbox"
                checked={crossoverWeights}
                onChange={(e) => {
                  setCrossoverWeights(e.target.checked);
                  controller?.setCrossoverWeightsEnabled(e.target.checked);
                }}
              />
              Enable Weight Crossover
            </label>
            <label className={styles.label}>
              <input
                type="checkbox"
                checked={mutateVars}
                onChange={(e) => {
                  setMutateVars(e.target.checked);
                  controller?.setMutateVarsEnabled(e.target.checked);
                }}
              />
              Enable Activation Variable Mutation
            </label>
            <label className={styles.label}>
              <input
                type="checkbox"
                checked={crossoverVars}
                onChange={(e) => {
                  setCrossoverVars(e.target.checked);
                  controller?.setCrossoverVarsEnabled(e.target.checked);
                }}
              />
              Enable Activation Variable Crossover
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
