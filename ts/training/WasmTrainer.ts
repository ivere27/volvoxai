import { WasmEngine } from '../backends/WasmEngine.js';
import { Model } from '../core/Model.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import {
  TrainerCore,
  prepareTrainer,
  type TrainerCoreOptions,
  type TrainingDriver,
  type TrainingDriverFactory,
} from './TrainerCore.js';
import { WasmAutograd } from './WasmAutograd.js';

export interface TrainerOptions extends TrainerCoreOptions {
  readonly wasmUrl?: string | URL;
}

export type { TrainerStepOptions, TrainerStepResult } from './TrainerCore.js';

/** Strict-WASM logical Trainer; this module has no CPU, WebGPU, or WGSL edge. */
export class Trainer extends TrainerCore<'wasm'> {
  private constructor(
    snapshot: Model,
    state: ReturnType<typeof prepareTrainer>['state'],
    driverFactory: TrainingDriverFactory,
    shapeOptions: ReturnType<typeof prepareTrainer>['shapeOptions'],
    disposeBackend: () => void,
  ) {
    super(snapshot, state, 'wasm', driverFactory, shapeOptions, disposeBackend);
  }

  static async create(
    snapshot: Model,
    options: TrainerOptions = {},
  ): Promise<Trainer> {
    const prepared = prepareTrainer(snapshot, options, 'wasm');
    const wasmUrl = options.wasmUrl ?? new URL('./volvoxai.full.wasm', import.meta.url);
    let engine: WasmEngine | null;
    try {
      engine = await WasmEngine.init(wasmUrl);
    } catch (error) {
      throw runtimeError(
        error,
        'BACKEND_UNAVAILABLE',
        `WASM training could not load '${String(wasmUrl)}'.`,
        { phase: 'initialization', backend: 'wasm' },
      );
    }
    if (!engine) {
      throw new VolvoxAIError(
        'BACKEND_UNAVAILABLE',
        `WASM training could not load '${String(wasmUrl)}'.`,
        { phase: 'initialization', backend: 'wasm' },
      );
    }
    const factory: TrainingDriverFactory = (graph) =>
      WasmAutograd.create(engine, graph, prepared.shapeOptions) as unknown as
        Promise<TrainingDriver>;
    return new Trainer(
      snapshot,
      prepared.state,
      factory,
      prepared.shapeOptions,
      () => engine.dispose(),
    );
  }
}

export function createTrainer(
  snapshot: Model,
  options: TrainerOptions = {},
): Promise<Trainer> {
  return Trainer.create(snapshot, options);
}
