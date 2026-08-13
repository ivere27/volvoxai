import type { ShapedRuntimeTensorView } from '../ops/shapeSystem.js';
import type { CrossEntropyTrainingOptions } from './TrainingLosses.js';
import type { TrainingOptimizerOptions } from './TrainingOptimizer.js';

/** Backend-neutral options shared by the CPU, WebGPU, and WASM trainers. */
export interface TrainingStepOptions extends CrossEntropyTrainingOptions {
  /**
   * Complete public-input binding for one concrete training step. Dynamic v1
   * never infers a shape from storage length, including for static graphs.
   */
  inputs?: Readonly<Record<string, ShapedRuntimeTensorView>>;
  trainableTensors?: string[];
  optimizer?: TrainingOptimizerOptions;
  updateMode?: unknown;
  dropout?: { seed?: number; counter?: number };
  gradientAccumulationSteps?: number;
  flushGradientAccumulation?: boolean;
  resetGradientAccumulation?: boolean;
}

/** @internal Concrete binding metadata installed by the Trainer. */
export interface TrainingKernelStepOptions extends Omit<TrainingStepOptions, 'inputs'> {
  inputs?: Record<string, import('../types.js').RuntimeTypedArray>;
  readonly shapeSignature?: string;
  readonly tacticSignature?: string;
}

/** @internal Fully bound variant passed by the public Trainer to a driver. */
export interface ResolvedTrainingStepOptions extends TrainingKernelStepOptions {
  inputs: Record<string, import('../types.js').RuntimeTypedArray>;
  readonly shapeSignature: string;
  readonly tacticSignature: string;
}
