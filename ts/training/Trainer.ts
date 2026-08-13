import { Model } from '../core/Model.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import { WasmEngine } from '../backends/WasmEngine.js';
import { CPUAutograd } from './CPUAutograd.js';
import {
  assertTrainerRequest,
  TrainerCore,
  prepareTrainer,
  type TrainerCoreOptions,
  type TrainingDriver,
  type TrainingDriverFactory,
  type TrainerStepOptions,
  type TrainerStepResult,
} from './TrainerCore.js';
import { WasmAutograd } from './WasmAutograd.js';
import { WebGPUAutograd } from './WebGPUAutograd.js';

export type TrainingBackend = 'cpu-js' | 'wasm' | 'webgpu';

export interface TrainerOptions extends TrainerCoreOptions {
  readonly backend?: TrainingBackend;
  readonly wasmUrl?: string | URL;
  readonly device?: GPUDevice | null;
}

export type { TrainerStepOptions, TrainerStepResult } from './TrainerCore.js';

interface BrowserGPUFactory {
  requestAdapter(): Promise<{ requestDevice(): Promise<GPUDevice> } | null>;
}

async function createWebGPUDevice(): Promise<GPUDevice> {
  const gpu = typeof navigator === 'undefined'
    ? null
    : (navigator as Navigator & { gpu?: BrowserGPUFactory }).gpu;
  if (!gpu) {
    throw new VolvoxAIError('BACKEND_UNAVAILABLE', 'WebGPU training is unavailable.', {
      phase: 'initialization', backend: 'webgpu',
    });
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    throw new VolvoxAIError('BACKEND_UNAVAILABLE', 'WebGPU training adapter is unavailable.', {
      phase: 'initialization', backend: 'webgpu',
    });
  }
  return adapter.requestDevice();
}

/** Full-profile logical Trainer with an explicit CPU, WASM, or WebGPU driver. */
export class Trainer extends TrainerCore<TrainingBackend> {
  private constructor(
    snapshot: Model,
    state: ReturnType<typeof prepareTrainer>['state'],
    backend: TrainingBackend,
    driverFactory: TrainingDriverFactory,
    shapeOptions: ReturnType<typeof prepareTrainer>['shapeOptions'],
    disposeBackend: (() => void) | null,
  ) {
    super(snapshot, state, backend, driverFactory, shapeOptions, disposeBackend);
  }

  static async create(
    snapshot: Model,
    options: TrainerOptions = {},
  ): Promise<Trainer> {
    assertTrainerRequest(snapshot, options);
    const {
      backend = 'cpu-js',
      wasmUrl = new URL('./volvoxai.full.wasm', import.meta.url),
      device = null,
    } = options;
    if (backend !== 'cpu-js' && backend !== 'wasm' && backend !== 'webgpu') {
      throw new VolvoxAIError(
        'INVALID_ARGUMENT',
        `Unsupported training backend '${String(backend)}'.`,
        { phase: 'selection', backend: typeof backend === 'string' ? backend : null },
      );
    }
    const prepared = prepareTrainer(snapshot, options, backend);
    let ownedDevice: GPUDevice | null = null;
    try {
      if (backend === 'cpu-js') {
        const factory: TrainingDriverFactory = (graph) => ({
          graph,
          trainStep: (step) => CPUAutograd.trainStep(graph, step),
        });
        return new Trainer(
          snapshot,
          prepared.state,
          backend,
          factory,
          prepared.shapeOptions,
          null,
        );
      }
      if (backend === 'wasm') {
        let engine: WasmEngine | null;
        try {
          engine = await WasmEngine.init(wasmUrl);
        } catch (error) {
          throw runtimeError(
            error,
            'BACKEND_UNAVAILABLE',
            `WASM training could not load '${String(wasmUrl)}'.`,
            { phase: 'initialization', backend },
          );
        }
        if (!engine) {
          throw new VolvoxAIError(
            'BACKEND_UNAVAILABLE',
            `WASM training could not load '${String(wasmUrl)}'.`,
            { phase: 'initialization', backend },
          );
        }
        const factory: TrainingDriverFactory = (graph) =>
          WasmAutograd.create(engine, graph, prepared.shapeOptions) as unknown as
            Promise<TrainingDriver>;
        return new Trainer(
          snapshot,
          prepared.state,
          backend,
          factory,
          prepared.shapeOptions,
          () => engine.dispose(),
        );
      }

      const selectedDevice = device || await createWebGPUDevice();
      if (!device) ownedDevice = selectedDevice;
      const factory: TrainingDriverFactory = (graph) =>
        new WebGPUAutograd(selectedDevice, graph, null, prepared.shapeOptions) as unknown as
          TrainingDriver;
      return new Trainer(
        snapshot,
        prepared.state,
        backend,
        factory,
        prepared.shapeOptions,
        ownedDevice ? () => ownedDevice?.destroy?.() : null,
      );
    } catch (error) {
      try { ownedDevice?.destroy?.(); } catch { /* Preserve initialization failure. */ }
      throw error;
    }
  }
}

export function createTrainer(
  snapshot: Model,
  options: TrainerOptions = {},
): Promise<Trainer> {
  return Trainer.create(snapshot, options);
}
