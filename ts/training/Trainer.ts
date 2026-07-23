import { Model } from '../core/ContextRuntime.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import { GraphOperatorNormalizer } from '../ops/graphOperatorNormalization.js';
import { WasmEngine } from '../backends/WasmEngine.js';
import { CPUAutograd } from './CPUAutograd.js';
import type { TrainingStepOptions } from './TrainingStep.js';
import { WasmAutograd } from './WasmAutograd.js';
import { WebGPUAutograd } from './WebGPUAutograd.js';
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

export type TrainingBackend = 'cpu' | 'wasm' | 'webgpu';

export interface TrainerOptions {
  readonly backend?: TrainingBackend;
  readonly wasmUrl?: string | URL;
  readonly device?: GPUDevice | null;
  readonly checkpoint?: ModelCheckpoint | null;
}

export interface TrainerStepOptions extends TrainingStepOptions {}

export interface TrainerCommitResult {
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly string[];
}

type TrainingDriverResult = Awaited<ReturnType<typeof CPUAutograd.trainStep>>;

export type TrainerStepResult = Omit<TrainingDriverResult, 'updatedTensors' | 'gradients'> & {
  readonly updatedTensorNames: readonly string[];
  readonly gradients: ReadonlyMap<string, Float32Array>;
};

interface TrainingDriver {
  trainStep(options: TrainingStepOptions): Promise<TrainingDriverResult>;
  dispose?(): void;
}

async function createTrainingDriver(
  backend: TrainingBackend,
  graph: StatefulTrainingGraph,
  wasmUrl: string | URL,
  device: GPUDevice | null,
): Promise<TrainingDriver> {
  if (backend === 'cpu') {
    return { trainStep: (options) => CPUAutograd.trainStep(graph, options) };
  }
  if (backend === 'wasm') {
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
    return WasmAutograd.create(engine, graph);
  }
  if (!device) {
    throw new VolvoxAIError('BACKEND_UNAVAILABLE', 'WebGPU training requires a device.', {
      phase: 'initialization', backend: 'webgpu',
    });
  }
  return new WebGPUAutograd(device, graph);
}

function commitResult(snapshot: {
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
  readonly adapterRevisionId: string | null;
  readonly adapterRevisionIds: readonly string[];
}): TrainerCommitResult {
  return Object.freeze({
    definitionId: snapshot.definitionId,
    topologyRevision: snapshot.topologyRevision,
    weightRevision: snapshot.weightRevision,
    weightRevisionId: snapshot.weightRevisionId,
    adapterRevisionId: snapshot.adapterRevisionId,
    adapterRevisionIds: Object.freeze([...snapshot.adapterRevisionIds]),
  });
}

function stableStepResult(result: TrainingDriverResult): TrainerStepResult {
  const { updatedTensors, gradients, ...metadata } = result;
  return {
    ...metadata,
    updatedTensorNames: Object.freeze(updatedTensors.map((tensor) => tensor.name)),
    gradients: new Map(Array.from(gradients, ([name, values]) => [name, new Float32Array(values)])),
  };
}

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
  let adapter;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error) {
    throw runtimeError(error, 'BACKEND_UNAVAILABLE',
      'WebGPU training adapter request failed.', {
        phase: 'initialization', backend: 'webgpu',
      });
  }
  if (!adapter) {
    throw new VolvoxAIError('BACKEND_UNAVAILABLE',
      'WebGPU training adapter is unavailable.', {
        phase: 'initialization', backend: 'webgpu',
      });
  }
  try {
    return await adapter.requestDevice();
  } catch (error) {
    throw runtimeError(error, 'BACKEND_UNAVAILABLE',
      'WebGPU training device request failed.', {
        phase: 'initialization', backend: 'webgpu',
      });
  }
}

/** Full-profile mutable training owner for one Model and private working revision. */
export class Trainer {
  readonly model: Model;
  readonly backend: TrainingBackend;
  #graph: StatefulTrainingGraph;
  #driver: TrainingDriver;
  readonly #releaseModel: () => void;
  readonly #ownedDevice: GPUDevice | null;
  readonly #device: GPUDevice | null;
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
    backend: TrainingBackend,
    driver: TrainingDriver,
    releaseModel: () => void,
    ownedDevice: GPUDevice | null,
    device: GPUDevice | null,
    wasmUrl: string | URL,
    baselineCheckpoint: ModelCheckpoint,
    baseWeightRevisionId: string,
    hasUncommittedUpdates: boolean,
  ) {
    this.model = model;
    this.#graph = graph;
    this.backend = backend;
    this.#driver = driver;
    this.#releaseModel = releaseModel;
    this.#ownedDevice = ownedDevice;
    this.#device = device;
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
        phase: 'initialization',
      });
    }
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'Trainer options must be an object.', {
        phase: 'initialization',
      });
    }
    const {
      backend = 'cpu',
      wasmUrl = new URL('./volvoxai.full.wasm', import.meta.url),
      device = null,
      checkpoint = null,
    } = options;
    if (backend !== 'cpu' && backend !== 'wasm' && backend !== 'webgpu') {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        `Unsupported training backend '${String(backend)}'.`, {
          phase: 'selection', backend: typeof backend === 'string' ? backend : null,
        });
    }
    const releaseModel = model._retainChild();
    const baseWeightRevisionId = model.weightRevisionId;
    let ownedDevice: GPUDevice | null = null;
    let driver: TrainingDriver | null = null;
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
      const selectedDevice = backend === 'webgpu'
        ? device || await createWebGPUDevice()
        : null;
      if (backend === 'webgpu' && !device) ownedDevice = selectedDevice;
      driver = await createTrainingDriver(backend, graph, wasmUrl, selectedDevice);
      return new Trainer(
        model,
        graph,
        backend,
        driver,
        releaseModel,
        ownedDevice,
        selectedDevice,
        wasmUrl,
        baselineCheckpoint,
        baseWeightRevisionId,
        checkpoint != null,
      );
    } catch (error) {
      try { driver?.dispose?.(); } catch { /* Preserve the construction failure. */ }
      try { ownedDevice?.destroy?.(); } catch { /* Preserve the construction failure. */ }
      releaseModel();
      throw error;
    }
  }

  get closed(): boolean {
    return this.#state === 'closed';
  }

  get trainingStep(): number {
    return this.#graph.trainingStep;
  }

  get hasUncommittedUpdates(): boolean {
    return this.#hasUncommittedUpdates;
  }

  trainStep(options: TrainerStepOptions = {}) {
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
      return stableStepResult(result);
    });
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
        return commitResult(published);
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
      const replacement = await createTrainingDriver(
        this.backend,
        restored,
        this.#wasmUrl,
        this.#device,
      );
      const previous = this.#driver;
      this.#graph = restored;
      this.#driver = replacement;
      this.#hasUncommittedUpdates = false;
      previous.dispose?.();
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
        phase: 'lifecycle', backend: this.backend,
      });
    }
    return getGradientAccumulationState(this.#graph);
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
          phase: 'lifecycle', backend: this.backend,
        }));
    }
    const pending = this.#tail.then(operation, operation);
    this.#tail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async #dispose(): Promise<void> {
    let failure: unknown = null;
    try {
      try {
        this.#driver.dispose?.();
      } catch (error) {
        failure = error;
      }
      try {
        this.#ownedDevice?.destroy?.();
      } catch (error) {
        failure ??= error;
      }
    } finally {
      this.#state = 'closed';
      this.#releaseModel();
    }
    if (failure) throw failure;
  }
}

export function createTrainer(
  model: Model,
  options: TrainerOptions = {},
): Promise<Trainer> {
  return Trainer.create(model, options);
}
