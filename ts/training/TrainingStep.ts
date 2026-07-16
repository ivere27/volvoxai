import type { RuntimeTypedArray } from '../types.js';
import type { CrossEntropyTrainingOptions } from './TrainingLosses.js';
import type { TrainingOptimizerOptions } from './TrainingOptimizer.js';

/** Backend-neutral options shared by the CPU, WebGPU, and WASM trainers. */
export interface TrainingStepOptions extends CrossEntropyTrainingOptions {
  inputs?: Record<string, RuntimeTypedArray>;
  trainableTensors?: string[];
  optimizer?: TrainingOptimizerOptions;
  updateMode?: unknown;
  dropout?: { seed?: number; counter?: number };
  gradientAccumulationSteps?: number;
  flushGradientAccumulation?: boolean;
  resetGradientAccumulation?: boolean;
}

/** Backward-compatible name retained for callers of CPUAutograd. */
export type CPUTrainStepOptions = TrainingStepOptions;
