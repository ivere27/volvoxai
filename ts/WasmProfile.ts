import { WasmEngine } from './backends/WasmEngine.js';
import { WasmBackendProvider } from './backends/WasmBackendProvider.js';
import { Runtime, type RuntimeOptions } from './core/ContextRuntime.js';
import type { Model } from './core/Model.js';
import { VolvoxAIError, runtimeError } from './core/RuntimeErrors.js';
import {
  createTrainer as createStrictTrainer,
  type TrainerOptions,
} from './training/WasmTrainer.js';

export interface WasmRuntimeOptions extends RuntimeOptions {
  readonly wasmUrl?: string | URL;
}

export type WasmRuntime = Runtime;

/** Create the strict browser-only WASM runtime. */
export async function createRuntime(options: WasmRuntimeOptions = {}): Promise<WasmRuntime> {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] WASM runtime options must be an object.', {
        phase: 'initialization', backend: 'wasm',
      });
  }
  const {
    wasmUrl = new URL('./volvoxai.full.wasm', import.meta.url),
    ...runtimeOptions
  } = options;
  if (!((typeof wasmUrl === 'string' && wasmUrl.length > 0) || wasmUrl instanceof URL)) {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] WASM runtime wasmUrl must be a non-empty string or URL.', {
        phase: 'initialization', backend: 'wasm',
      });
  }
  if (runtimeOptions.onDiagnostic != null &&
      typeof runtimeOptions.onDiagnostic !== 'function') {
    throw new VolvoxAIError('INVALID_ARGUMENT',
      '[VolvoxAI] WASM runtime onDiagnostic must be a function or null.', {
        phase: 'initialization', backend: 'wasm',
      });
  }

  let engine: WasmEngine | null;
  try {
    engine = await WasmEngine.init(wasmUrl);
  } catch (error) {
    throw runtimeError(error, 'BACKEND_UNAVAILABLE',
      `WASM runtime could not load '${String(wasmUrl)}'.`, {
        phase: 'initialization', backend: 'wasm',
      });
  }
  if (!engine) {
    throw new VolvoxAIError('BACKEND_UNAVAILABLE',
      `WASM runtime could not load '${String(wasmUrl)}'.`, {
        phase: 'initialization', backend: 'wasm',
      });
  }

  let runtime: Runtime | null = null;
  let provider: WasmBackendProvider | null = null;
  try {
    // Keep this as the complete profile-independent option object. Adding a
    // RuntimeOption must not require a second allowlist in the strict entry.
    runtime = new Runtime(runtimeOptions);
    provider = new WasmBackendProvider(engine);
    runtime._addProvider('wasm', provider);
    return runtime;
  } catch (error) {
    // Close the Runtime first: _addProvider may have transferred ownership
    // before reporting a later construction failure. If it did not, the
    // provider remains open and is reclaimed explicitly below.
    try {
      if (runtime) await runtime.close();
    } catch {
      // Preserve the construction failure; cleanup failure is secondary.
    }
    try {
      if (provider) {
        if (!provider._isClosed()) provider.close();
      } else {
        engine.dispose();
      }
    } catch {
      // Preserve the construction failure; cleanup failure is secondary.
    }
    throw error;
  }
}

export function createTrainer(
  snapshot: Model,
  options: TrainerOptions = {},
) {
  return createStrictTrainer(snapshot, {
    ...options,
    wasmUrl: options.wasmUrl ?? new URL('./volvoxai.full.wasm', import.meta.url),
  });
}

/** Stateless namespace for the strict WASM build. */
export const VolvoxAI = Object.freeze({ createRuntime, createTrainer });
