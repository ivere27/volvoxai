export * from './index.js';
export * from './training/index.js';
export { TrainingGraph as Graph } from './training/TrainingGraph.js';
export { TrainingModelBuilder as ModelBuilder } from './training/TrainingModelBuilder.js';

import {
  VolvoxAI as BaseProfile,
  createRuntime as createBaseRuntime,
  type CreateRuntimeOptions,
} from './VolvoxAI.js';
import { createTrainer } from './training/Trainer.js';

/** Create an inference runtime whose default WASM provider uses the full sidecar. */
export function createRuntime(options: CreateRuntimeOptions = {}) {
  return createBaseRuntime({
    ...options,
    wasmUrl: options.wasmUrl ?? new URL('./volvoxai.full.wasm', import.meta.url),
  });
}

/** Stateless full-profile namespace. Mutable state starts at Runtime or Trainer. */
export const VolvoxAI = Object.freeze({
  ...BaseProfile,
  createRuntime,
  createTrainer,
});
