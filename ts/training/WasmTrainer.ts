import { Model } from '../core/ContextRuntime.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import { GraphOperatorNormalizer } from '../ops/graphOperatorNormalization.js';
import { WasmEngine } from '../backends/WasmEngine.js';
import { WasmAutograd } from './WasmAutograd.js';
import type { TrainingStepOptions } from './TrainingStep.js';
import type { TrainerStepResult } from './Trainer.js';
import type { TrainerCommitResult } from './Trainer.js';
import {
  getGradientAccumulationState,
  resetGradientAccumulation,
} from './GradientAccumulation.js';
import { resolveTrainingOptimizer } from './TrainingOptimizer.js';
import {
  ensureTrainingGraphState,
  type StatefulTrainingGraph,
} from './TrainingGraph.js';
import {
  exportModelCheckpoint,
  importModelCheckpoint,
  type ModelCheckpoint,
  type ModelCheckpointExportOptions,
} from './ModelCheckpoint.js';

export interface TrainerOptions {
  readonly wasmUrl?: string | URL;
  readonly checkpoint?: ModelCheckpoint | null;
}

/** Training owner used by the strict browser-only WASM profile. */
export class Trainer {
  readonly model: Model;
  #graph: StatefulTrainingGraph;
  #driver: WasmAutograd;
  readonly #releaseModel: () => void;
  readonly #wasmUrl: string | URL;
  #baselineCheckpoint: ModelCheckpoint;
  #baseWeightRevisionId: string;
  #hasUncommittedUpdates: boolean;
  #tail: Promise<void> = Promise.resolve();
  #state: 'open' | 'closing' | 'closed' = 'open';
  #closePromise: Promise<void> | null = null;

  private constructor(
    model: Model,
    graph: StatefulTrainingGraph,
    driver: WasmAutograd,
    releaseModel: () => void,
    wasmUrl: string | URL,
    baselineCheckpoint: ModelCheckpoint,
    baseWeightRevisionId: string,
    hasUncommittedUpdates: boolean,
  ) {
    this.model = model;
    this.#graph = graph;
    this.#driver = driver;
    this.#releaseModel = releaseModel;
    this.#wasmUrl = wasmUrl;
    this.#baselineCheckpoint = baselineCheckpoint;
    this.#baseWeightRevisionId = baseWeightRevisionId;
    this.#hasUncommittedUpdates = hasUncommittedUpdates;
  }

  static async create(
    model: Model,
    options: TrainerOptions = {},
  ): Promise<Trainer> {
    if (!(model instanceof Model)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Trainer.create requires a Model.', {
        phase: 'initialization', backend: 'wasm',
      });
    }
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Trainer options must be an object.', {
        phase: 'initialization', backend: 'wasm',
      });
    }
    const {
      wasmUrl = new URL('./volvoxai.full.wasm', import.meta.url),
      checkpoint = null,
    } = options;
    const releaseModel = model._retainChild();
    const baseWeightRevisionId = model.weightRevisionId;
    let driver: WasmAutograd | null = null;
    try {
      const baseGraph = ensureTrainingGraphState(model._createRetainedWorkingGraph());
      baseGraph.assertValid();
      GraphOperatorNormalizer.ensureMatMulLayouts(baseGraph);
      const baselineCheckpoint = exportModelCheckpoint(baseGraph);
      const graph = checkpoint == null
        ? baseGraph
        : importModelCheckpoint(checkpoint).graph;
      graph.assertValid();
      GraphOperatorNormalizer.ensureMatMulLayouts(graph);
      if (checkpoint != null) model._assertRetainedDefinition(graph);
      let engine;
      try {
        engine = await WasmEngine.init(wasmUrl);
      } catch (error) {
        throw runtimeError(error, 'BACKEND_UNAVAILABLE',
          `WASM training could not load '${String(wasmUrl)}'.`, {
            phase: 'initialization', backend: 'wasm',
          });
      }
      if (!engine) {
        throw new VolvoxAIError('BACKEND_UNAVAILABLE',
          `WASM training could not load '${String(wasmUrl)}'.`, {
            phase: 'initialization', backend: 'wasm',
          });
      }
      driver = await WasmAutograd.create(engine, graph);
      const trainer = new Trainer(
        model,
        graph,
        driver,
        releaseModel,
        wasmUrl,
        baselineCheckpoint,
        baseWeightRevisionId,
        checkpoint != null,
      );
      return trainer;
    } catch (error) {
      try { driver?.dispose(); } catch { /* Preserve the construction failure. */ }
      releaseModel();
      throw error;
    }
  }

  trainStep(options: TrainingStepOptions = {}): Promise<TrainerStepResult> {
    return this.#enqueue(async () => {
      const resolved = resolveTrainingOptimizer(
        this.#graph,
        options.updateMode,
        options.optimizer,
      );
      const result = await this.#driver.trainStep({
        ...options,
        updateMode: resolved.updateMode,
        optimizer: resolved.optimizer,
      });
      if (result.updatedTensors.length > 0) this.#hasUncommittedUpdates = true;
      const { updatedTensors, gradients, ...metadata } = result;
      return {
        ...metadata,
        updatedTensorNames: Object.freeze(updatedTensors.map((tensor) => tensor.name)),
        gradients: new Map(Array.from(gradients, ([name, values]) => [name, new Float32Array(values)])),
      };
    });
  }

  get hasUncommittedUpdates(): boolean {
    return this.#hasUncommittedUpdates;
  }

  commit(): Promise<TrainerCommitResult> {
    return this.#enqueue(async () => {
      if (getGradientAccumulationState(this.#graph).pending) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Trainer commit requires gradient accumulation to be completed or reset.', {
            phase: 'execution',
          });
      }
      if (!this.#hasUncommittedUpdates) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          'Trainer has no private update to commit.', { phase: 'execution' });
      }
      const nextBaseline = exportModelCheckpoint(this.#graph);
      const releasePublication = await this.model._acquireRetainedRevision(
        this.#baseWeightRevisionId,
      );
      try {
        const published = this.model._publishRetainedRevision(
          this.#graph,
          this.#baseWeightRevisionId,
        );
        this.#baseWeightRevisionId = published.weightRevisionId;
        this.#baselineCheckpoint = nextBaseline;
        this.#hasUncommittedUpdates = false;
        return Object.freeze({
          definitionId: published.definitionId,
          topologyRevision: published.topologyRevision,
          weightRevision: published.weightRevision,
          weightRevisionId: published.weightRevisionId,
          adapterRevisionId: published.adapterRevisionId,
          adapterRevisionIds: Object.freeze([...published.adapterRevisionIds]),
        });
      } finally {
        releasePublication();
      }
    });
  }

  rollback(): Promise<void> {
    return this.#enqueue(async () => {
      const restored = importModelCheckpoint(this.#baselineCheckpoint).graph;
      restored.assertValid();
      GraphOperatorNormalizer.ensureMatMulLayouts(restored);
      let engine;
      try {
        engine = await WasmEngine.init(this.#wasmUrl);
      } catch (error) {
        throw runtimeError(error, 'BACKEND_UNAVAILABLE',
          `WASM training could not load '${String(this.#wasmUrl)}'.`, {
            phase: 'initialization', backend: 'wasm',
          });
      }
      if (!engine) {
        throw new VolvoxAIError('BACKEND_UNAVAILABLE',
          `WASM training could not load '${String(this.#wasmUrl)}'.`, {
            phase: 'initialization', backend: 'wasm',
          });
      }
      const replacement = await WasmAutograd.create(engine, restored);
      const previous = this.#driver;
      this.#graph = restored;
      this.#driver = replacement;
      this.#hasUncommittedUpdates = false;
      previous.dispose();
    });
  }

  resetGradientAccumulation(): Promise<void> {
    return this.#enqueue(() => resetGradientAccumulation(this.#graph));
  }

  exportCheckpoint(options: ModelCheckpointExportOptions = {}): Promise<ModelCheckpoint> {
    return this.#enqueue(() => exportModelCheckpoint(this.#graph, options));
  }

  getGradientAccumulationState() {
    if (this.#state !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Trainer is closing or closed.', {
        phase: 'lifecycle', backend: 'wasm',
      });
    }
    return getGradientAccumulationState(this.#graph);
  }

  get trainingStep(): number {
    return this.#graph.trainingStep;
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = this.#tail.then(
      () => this.#dispose(),
      () => this.#dispose(),
    );
    this.#tail = this.#closePromise.then(() => undefined, () => undefined);
    return this.#closePromise;
  }

  dispose(): Promise<void> {
    return this.close();
  }

  #enqueue<TResult>(operation: () => Promise<TResult> | TResult): Promise<TResult> {
    if (this.#state !== 'open') {
      return Promise.reject(new VolvoxAIError('HANDLE_DISPOSED',
        'Trainer is closing or closed.', {
          phase: 'lifecycle', backend: 'wasm',
        }));
    }
    const pending = this.#tail.then(operation, operation);
    this.#tail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async #dispose(): Promise<void> {
    try {
      this.#driver.dispose();
    } finally {
      this.#state = 'closed';
      this.#releaseModel();
    }
  }
}

export function createTrainer(
  model: Model,
  options: TrainerOptions = {},
): Promise<Trainer> {
  return Trainer.create(model, options);
}
