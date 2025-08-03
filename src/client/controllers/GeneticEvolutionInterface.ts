import { VariableValue } from "../utils/ActivationVariableUtils";

export interface GeneticEvolutionConfig {
  activationCode: string;
  weights: number[][][][]; 
  activationVariables: VariableValue[];
}

export interface ScoredCandidate {
  weights: number[][][][]; 
  activationVariables: VariableValue[];
  score: number;
  filler?: boolean;
}

export interface GeneticCandidate {
  weights: number[][][][]; 
  activationVariables: VariableValue[];
  filler?: boolean;
}

export interface IGeneticEvolutionController {
  init(): Promise<void>;
  getGenerationCount(): number;
  getBestOfGeneration(idx: number): ScoredCandidate | null;
  getStatus(): string;
  presentNext(): GeneticCandidate[];
  setMutationRate(rate: number): void;
  setActivationFunctionCode(newCode: string): void;
  updateWeights(newWeights: number[][][][]): void;
  setActivationVariables(newVars: VariableValue[]): void;
  generateRandomGeneration(): GeneticCandidate[];
  generateGeneration(): void;
  submitChoice(chosenIndex: number): Promise<void>;
}