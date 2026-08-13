export * from './index.js';
export * from './training/index.js';

import {
  VolvoxAI as BaseProfile,
  createRuntime as createBaseRuntime,
  type CreateRuntimeOptions,
} from './VolvoxAI.js';
import type { Runtime } from './core/ContextRuntime.js';
import { createTrainer } from './training/Trainer.js';

export type FullRuntime = Runtime;

/** Create an inference runtime whose default WASM provider uses the full sidecar. */
export async function createRuntime(options: CreateRuntimeOptions = {}): Promise<FullRuntime> {
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
