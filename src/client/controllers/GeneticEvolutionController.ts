import { ActivationVariableUtils, VariableValue } from "../utils/ActivationVariableUtils";
import { IGeneticEvolutionController } from "./GeneticEvolutionInterface";

export interface GeneticEvolutionConfig {
  activationCode: string;
  weights: number[][][][];
  activationVariables: VariableValue[];
}

interface ScoredCandidate {
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

export class GeneticEvolutionController implements IGeneticEvolutionController  {
  private populationSize = 32;
  private mutationIntensity = 1.0;
  private mutationProbability = 0.05;
  private topNParent1 = 1;
  private topNParent2 = 2;
  private crossoverRatio = 0.9;

  private currentPopulation: ScoredCandidate[] = [];
  private generations: ScoredCandidate[][] = [];
  private history: ScoredCandidate[] = [];

  private currentRoundPool: ScoredCandidate[] = [];
  private currentBatch = 0;
  private currentCandidates: GeneticCandidate[] = [];

  private generationCount = 0;
  private bestCandidate: ScoredCandidate | null = null;

  private activationCode: string;
  private baseWeights: number[][][][];
  private baseActivationVariables: VariableValue[];

  constructor(private config: GeneticEvolutionConfig) {
    this.activationCode = config.activationCode;
    this.baseWeights = config.weights;
    this.baseActivationVariables = config.activationVariables;
  }

  async init(): Promise<void> {
    this.reset();
    this.generateGeneration();
    this.startTournament(this.currentPopulation);
  }

  private reset(): void {
    this.history = [];
    this.generations = [];
    this.generationCount = 0;
    this.bestCandidate = null;
    this.currentBatch = 0;
  }

  getGenerationCount(): number {
    return this.generationCount;
  }

  getBestOfGeneration(idx: number): ScoredCandidate | null {
    const gen = this.generations[idx];
    return gen ? gen.reduce((best, c) => (c.score > best.score ? c : best), gen[0]) : null;
  }

  getStatus(): string {
    const totalBatches = Math.ceil(this.currentRoundPool.length / 4);
    return `Generation: ${this.generationCount}, ` +
           `Round Size: ${this.currentRoundPool.length}, ` +
           `Batch Progress: ${this.currentBatch + 1}/${totalBatches}, `
  }

  presentNext(): GeneticCandidate[] {
    while (this.currentCandidates.length < 4) {
      this.currentCandidates.push({ ...this.makeEmptyCandidate(), filler: true });
    }
    return [...this.currentCandidates];
  }

  setMutationRate(rate: number) {
    this.mutationIntensity = rate;
  }

  setActivationFunctionCode(newCode: string) {
    this.activationCode = newCode;
  }

  updateWeights(newWeights: number[][][][]) {
    this.baseWeights = newWeights;
  }

  setActivationVariables(newVars: VariableValue[]) {
    this.baseActivationVariables = newVars;
  }

  generateRandomGeneration(): GeneticCandidate[] {
    this.reset();
    this.currentPopulation = Array.from({ length: this.populationSize }, () => this.makeRandomCandidate());
    this.startTournament(this.currentPopulation);
    return this.currentCandidates;
  }

  generateGeneration(): void {
    this.currentPopulation = Array.from({ length: this.populationSize }, () => ({
      weights: this.mutateWeights(this.baseWeights),
      activationVariables: this.mutateVariables(this.baseActivationVariables),
      score: 0,
    }));
  }

  setMutationProbability(prob: number) {
    this.mutationProbability = Math.max(0, Math.min(1, prob));
  }

  setTopNParent1(count: number) {
    this.topNParent1 = Math.max(1, Math.floor(count));
  }

  setTopNParent2(count: number) {
    this.topNParent2 = Math.max(1, Math.floor(count));
  }

  setCrossoverRatio(ratio: number) {
    this.crossoverRatio = Math.max(0, Math.min(1, ratio));
  }

  async submitChoice(chosenIndex: number): Promise<void> {
    if (chosenIndex < 0 || chosenIndex >= this.currentCandidates.length) throw new Error("Invalid choice index");

    const chosen = this.currentCandidates[chosenIndex];

    if (chosen.filler) {
      this.loadNextBatch();
      return;
    }

    const batch = this.currentRoundPool.slice(this.currentBatch * 4, this.currentBatch * 4 + 4);
    const selected = batch[chosenIndex];

    selected.score++;
    this.history.push(selected);
    if (!this.bestCandidate || selected.score > this.bestCandidate.score) this.bestCandidate = selected;

    this.currentBatch++;
    const totalBatches = Math.ceil(this.currentRoundPool.length / 4);

    if (this.currentBatch >= totalBatches) {
      const winners = this.history.slice(-totalBatches);
      if (winners.length > 1) this.startTournament(winners);
      else {
        this.evolve();
        this.startTournament(this.currentPopulation);
      }
    } else this.loadNextBatch();
  }

  private startTournament(pool: ScoredCandidate[]): void {
    this.currentRoundPool = [...pool].sort(() => Math.random() - 0.5);
    this.currentBatch = 0;
    this.loadNextBatch();
  }

  private loadNextBatch(): void {
    const batch = this.currentRoundPool.slice(this.currentBatch * 4, this.currentBatch * 4 + 4);
    this.currentCandidates = batch.map(c => ({ weights: c.weights, activationVariables: c.activationVariables }));
    while (this.currentCandidates.length < 4) {
      this.currentCandidates.push({ ...this.makeEmptyCandidate(), filler: true });
    }
  }

  private evolve(): void {
    this.generationCount++;
    this.generations.push([...this.currentPopulation]);

    const recent = this.history
      .slice(-this.populationSize)
      .sort((a, b) => b.score - a.score);

    const top1 = recent.slice(0, this.topNParent1);
    const top2 = recent.slice(0, this.topNParent2);

    const newPop: ScoredCandidate[] = [];
    newPop.push(...top1);

    while (newPop.length < this.populationSize) {
      if (top1.length > 0 && top2.length > 0) {
        const p1 = this.selectParent(top1);
        const p2 = this.selectParent(top2);
        const child = this.crossover(p1, p2);
        newPop.push(this.mutate(child));
      } else {
        newPop.push(this.makeRandomCandidate());
      }
    }

    this.currentPopulation = newPop;
  }

  private selectParent(candidates: ScoredCandidate[]): ScoredCandidate {
    const tourneySize = Math.min(5, candidates.length);
    const tourney = Array.from({ length: tourneySize }, () => candidates[Math.floor(Math.random() * candidates.length)]);
    return tourney.reduce((best, c) => (c.score > best.score ? c : best), tourney[0]);
  }

  private crossover(p1: ScoredCandidate, p2: ScoredCandidate): ScoredCandidate {
    const weights = p1.weights.map((c, i) =>
      c.map((k, j) =>
        k.map((r, m) =>
          r.map((_, n) =>
            Math.random() < this.crossoverRatio
              ? p1.weights[i][j][m][n]
              : p2.weights[i][j][m][n]
          )
        )
      )
    );

    const activationVariables = p1.activationVariables.map((v, i) => ({
      name: v.name,
      value:
        Math.random() < this.crossoverRatio
          ? v.value
          : p2.activationVariables[i].value,
    }));

    return { weights, activationVariables, score: 0 };
  }

  private mutate(candidate: Omit<ScoredCandidate, "score">): ScoredCandidate {
    return { weights: this.mutateWeights(candidate.weights), activationVariables: this.mutateVariables(candidate.activationVariables), score: 0 };
  }

  private makeRandomCandidate(): ScoredCandidate {
    return { weights: this.generateRandomWeights(), activationVariables: this.generateRandomVariables(), score: 0 };
  }

  private makeEmptyCandidate(): ScoredCandidate {
    return { weights: this.generateEmptyWeights(), activationVariables: this.generateEmptyVariables(), score: 0 };
  }

  private mutateWeights(weights: number[][][][]): number[][][][] {
    return weights.map(channel => channel.map(kernel => kernel.map(row => row.map(w => (Math.random() < this.mutationProbability ? this.roundToSigFigs(w + this.gaussianRandom(0, this.mutationIntensity), 2) : w)))));
  }
  
  private gaussianRandom(mean = 0, stdev = 1): number {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * stdev + mean;
  }

  private mutateVariables(vars: VariableValue[]): VariableValue[] {
    const metas = ActivationVariableUtils.getVariables(this.activationCode);
    return vars.map(v => {
      const meta = metas.find(x => x.name === v.name)!;
      if (Math.random() < this.mutationProbability) {
        const delta = (Math.random() * 2 - 1) * this.mutationIntensity;
        const newValue = Math.max(meta.min, Math.min(meta.max, v.value + delta));
        return { name: v.name, value: this.roundToSigFigs(newValue, 2) };
      }
      return v;
    });
  }

  private roundToSigFigs(num: number, sig: number): number {
    return Number(num.toFixed(sig));
  }

  private generateRandomWeights(): number[][][][] {
    return Array.from({ length: 3 }, () =>
      Array.from({ length: 3 }, () =>
        Array.from({ length: 5 }, () =>
          Array.from({ length: 5 }, () => this.roundToSigFigs(Math.random() * 2 - 1, 2))
        )
      )
    );
  }

  private generateRandomVariables(): VariableValue[] {
    return ActivationVariableUtils.getVariables(this.activationCode).map(m => ({
      name: m.name,
      value: this.roundToSigFigs(m.min + Math.random() * (m.max - m.min), 2),
    }));
  }

  private generateEmptyWeights(): number[][][][] {
    return Array.from({ length: 3 }, () =>
      Array.from({ length: 3 }, () =>
        Array.from({ length: 5 }, () =>
          Array.from({ length: 5 }, () => 0)
        )
      )
    );
  }

  private generateEmptyVariables(): VariableValue[] {
    return ActivationVariableUtils.getVariables(this.activationCode).map(m => ({
      name: m.name,
      value: 0,
    }));
  }
}
