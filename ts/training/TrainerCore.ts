import { Model } from '../core/Model.js';
import { VolvoxAIError, runtimeError } from '../core/RuntimeErrors.js';
import type { CPUAutograd } from './CPUAutograd.js';
import {
  getGradientAccumulationState,
  resetGradientAccumulation,
} from './GradientAccumulation.js';
import {
  exportTrainingCheckpoint,
  restoreTrainingCheckpoint,
  type ModelCheckpoint,
  type ModelCheckpointExportOptions,
} from './ModelCheckpoint.js';
import { resolveTrainingOptimizer } from './TrainingOptimizer.js';
import type { StatefulTrainingGraph } from './TrainingGraph.js';
import {
  cloneTrainingMutableState,
  createTrainingMutableState,
  TrainingShapeContext,
  type TrainingConcreteBinding,
  type TrainingMutableState,
  type TrainingShapeContextOptions,
  type TrainingShapeInspection,
} from './TrainingShapeContext.js';
import type {
  ResolvedTrainingStepOptions,
  TrainingStepOptions,
} from './TrainingStep.js';
import type { BackendPlanCacheInspection } from './BackendPlanCache.js';

export interface TrainerCoreOptions extends TrainingShapeContextOptions {
  readonly checkpoint?: ModelCheckpoint | null;
}

export interface TrainerStepOptions extends TrainingStepOptions {}

type TrainingDriverResult = Awaited<ReturnType<typeof CPUAutograd.trainStep>>;

export type TrainerStepResult = Omit<TrainingDriverResult, 'updatedTensors' | 'gradients'> & {
  readonly updatedTensorNames: readonly string[];
  readonly gradients: ReadonlyMap<string, Float32Array>;
  readonly shapeSignature: string;
  readonly tacticSignature: string;
  readonly activationShapes: Readonly<Record<string, readonly number[]>>;
  readonly activationGradientShapes: Readonly<Record<string, readonly number[]>>;
};

export interface TrainingDriver {
  readonly graph: StatefulTrainingGraph;
  trainStep(options: ResolvedTrainingStepOptions): Promise<TrainingDriverResult>;
  rebind?(graph: StatefulTrainingGraph, binding: TrainingConcreteBinding): Promise<void> | void;
  inspectPlanCache?(): Readonly<BackendPlanCacheInspection>;
  dispose?(): void;
}

export type TrainingDriverFactory = (
  graph: StatefulTrainingGraph,
) => Promise<TrainingDriver> | TrainingDriver;

export interface PreparedTrainer {
  readonly state: TrainingMutableState;
  readonly shapeOptions: TrainingShapeContextOptions;
}

function snapshotSource(snapshot: Model, state: TrainingMutableState) {
  return {
    graph: snapshot.graph,
    weights: Object.fromEntries(snapshot.weightDescriptors.map((descriptor) => [
      descriptor.name,
      {
        name: descriptor.name,
        dtype: descriptor.dtype,
        shape: descriptor.shape,
        data: state.parameters.get(descriptor.name)!,
        role: descriptor.role,
      },
    ])),
    quantizationByTensor: snapshot.quantizationByTensor,
  };
}

function stableStepResult(
  result: TrainingDriverResult,
  binding: TrainingConcreteBinding,
): TrainerStepResult {
  const { updatedTensors, gradients, ...metadata } = result;
  return Object.freeze({
    ...metadata,
    updatedTensorNames: Object.freeze(updatedTensors.map((tensor) => tensor.name)),
    gradients: new Map(Array.from(gradients, ([name, values]) => [
      name,
      new Float32Array(values),
    ])),
    shapeSignature: binding.shapeSignature,
    tacticSignature: binding.tacticSignature,
    activationShapes: binding.activationShapes,
    activationGradientShapes: binding.activationGradientShapes,
  });
}

export function assertTrainerRequest(
  snapshot: unknown,
  options: unknown,
): asserts snapshot is Model {
  if (!(snapshot instanceof Model)) {
    throw new VolvoxAIError(
      'INVALID_ARGUMENT',
      'Trainer.create requires an immutable Model.',
      { phase: 'initialization' },
    );
  }
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new VolvoxAIError('INVALID_ARGUMENT', 'Trainer options must be an object.', {
      phase: 'initialization',
    });
  }
}

export function prepareTrainer(
  snapshot: Model,
  options: TrainerCoreOptions,
  backend: string,
): PreparedTrainer {
  assertTrainerRequest(snapshot, options);
  const {
    checkpoint = null,
    planCacheEntries,
    planCacheMetadataBytes,
    maxCapacityBytes,
    capacityGrowthFactor,
  } = options;
  const shapeOptions = {
    ...(planCacheEntries === undefined ? {} : { planCacheEntries }),
    ...(planCacheMetadataBytes === undefined ? {} : { planCacheMetadataBytes }),
    ...(maxCapacityBytes === undefined ? {} : { maxCapacityBytes }),
    ...(capacityGrowthFactor === undefined ? {} : { capacityGrowthFactor }),
  };
  let state = createTrainingMutableState(snapshot);
  if (checkpoint !== null) {
    try {
      state = restoreTrainingCheckpoint(checkpoint, snapshot).state;
    } catch (error) {
      throw runtimeError(error, 'INVALID_ARGUMENT', 'Training checkpoint restore failed.', {
        phase: 'initialization', backend,
      });
    }
  }
  return Object.freeze({ state, shapeOptions });
}

/**
 * Backend-neutral owner for logical training state. Entry-specific subclasses
 * inject one statically imported driver factory, preserving bundle boundaries.
 */
export class TrainerCore<TBackend extends string = string> {
  readonly backend: TBackend;
  #driverFactory: TrainingDriverFactory | null;
  #disposeBackend: (() => void) | null;
  readonly #shapeOptions: TrainingShapeContextOptions;
  #snapshot: Model;
  #stateData: TrainingMutableState;
  #baselineState: TrainingMutableState;
  #shapeContext: TrainingShapeContext;
  #driver: TrainingDriver | null = null;
  #hasUncommittedUpdates = false;
  #tail: Promise<void> = Promise.resolve();
  #lifecycle: 'open' | 'closing' | 'closed' = 'open';
  #closePromise: Promise<void> | null = null;

  protected constructor(
    snapshot: Model,
    state: TrainingMutableState,
    backend: TBackend,
    driverFactory: TrainingDriverFactory,
    shapeOptions: TrainingShapeContextOptions,
    disposeBackend: (() => void) | null = null,
  ) {
    this.#snapshot = snapshot;
    this.#stateData = state;
    this.#baselineState = cloneTrainingMutableState(state);
    this.backend = backend;
    this.#driverFactory = driverFactory;
    this.#disposeBackend = disposeBackend;
    this.#shapeOptions = shapeOptions;
    this.#shapeContext = new TrainingShapeContext(snapshot, state, shapeOptions);
  }

  get snapshot(): Model {
    return this.#snapshot;
  }

  get closed(): boolean {
    return this.#lifecycle === 'closed';
  }

  get trainingStep(): number {
    return this.#stateData.trainingStep;
  }

  get hasUncommittedUpdates(): boolean {
    return this.#hasUncommittedUpdates;
  }

  trainStep(options: TrainerStepOptions = {}): Promise<TrainerStepResult> {
    return this.#enqueue(async () => {
      const currentGraph = this.#shapeContext.currentGraph();
      if (options.resetGradientAccumulation === true && currentGraph) {
        resetGradientAccumulation(currentGraph);
      }
      let requestedSignature: string;
      try {
        requestedSignature = this.#snapshot.bindShapes(options.inputs ?? {}).signature;
      } catch (error) {
        throw runtimeError(
          error,
          'INVALID_ARGUMENT',
          'Training inputs do not satisfy the complete logical shape contract.',
          { phase: 'execution', backend: this.backend },
        );
      }
      if (currentGraph && getGradientAccumulationState(currentGraph).pending &&
          this.#shapeContext.inspect().currentShapeSignature !== requestedSignature) {
        throw new VolvoxAIError(
          'INVALID_ARGUMENT',
          'A gradient accumulation window requires one exact activation shape signature; ' +
            'flush or reset it before changing shape.',
          { phase: 'execution', backend: this.backend },
        );
      }
      let binding: TrainingConcreteBinding;
      try {
        binding = this.#shapeContext.bind(options.inputs ?? {});
      } catch (error) {
        throw runtimeError(
          error,
          'INVALID_ARGUMENT',
          'Training inputs do not satisfy the complete logical shape contract.',
          { phase: 'execution', backend: this.backend },
        );
      }
      if (this.#driver?.graph !== binding.graph) {
        if (this.#driver?.rebind) {
          await this.#driver.rebind(binding.graph, binding);
          if (this.#driver.graph !== binding.graph) {
            throw new Error('Training backend rebind did not publish the requested concrete graph.');
          }
        } else {
          const factory = this.#driverFactory;
          if (!factory) throw new Error('Training backend factory has been released.');
          const replacement = await factory(binding.graph);
          const previous = this.#driver;
          this.#driver = replacement;
          previous?.dispose?.();
        }
      }
      const resolved = resolveTrainingOptimizer(
        binding.graph,
        options.updateMode,
        options.optimizer,
      );
      const result = await this.#driver!.trainStep({
        ...options,
        inputs: binding.rawInputs,
        shapeSignature: binding.shapeSignature,
        tacticSignature: binding.tacticSignature,
        updateMode: resolved.updateMode,
        optimizer: resolved.optimizer,
      });
      this.#shapeContext.captureState(binding.graph);
      for (const name of result.updatedTensors.map((tensor) => tensor.name)) {
        const parameter = this.#stateData.parameters.get(name);
        const gradient = result.gradients.get(name);
        if (!(parameter instanceof Float32Array) || !gradient ||
            gradient.length !== parameter.length) {
          throw new Error(`Training gradient for '${name}' changed its fixed parameter shape.`);
        }
      }
      if (result.updatedTensors.length > 0) this.#hasUncommittedUpdates = true;
      return stableStepResult(result, binding);
    });
  }

  commit(): Promise<Model> {
    return this.#enqueue(() => {
      this.#assertNoPendingAccumulation('commit');
      if (!this.#hasUncommittedUpdates) {
        throw new VolvoxAIError(
          'INVALID_ARGUMENT',
          'Trainer has no private parameter update to commit.',
          { phase: 'execution' },
        );
      }
      const successor = Model.derive(
        snapshotSource(this.#snapshot, this.#stateData),
        this.#snapshot,
      );
      this.#snapshot = successor;
      this.#baselineState = cloneTrainingMutableState(this.#stateData);
      this.#hasUncommittedUpdates = false;
      this.#replaceShapeContext(cloneTrainingMutableState(this.#stateData));
      return successor;
    });
  }

  rollback(): Promise<void> {
    return this.#enqueue(() => {
      this.#replaceShapeContext(cloneTrainingMutableState(this.#baselineState));
      this.#hasUncommittedUpdates = false;
    });
  }

  resetGradientAccumulation(): Promise<void> {
    return this.#enqueue(() => {
      const graph = this.#shapeContext.currentGraph();
      if (graph) resetGradientAccumulation(graph);
    });
  }

  exportCheckpoint(options: ModelCheckpointExportOptions = {}): Promise<ModelCheckpoint> {
    return this.#enqueue(() => {
      this.#assertNoPendingAccumulation('checkpoint export');
      return exportTrainingCheckpoint(this.#snapshot, this.#stateData, options);
    });
  }

  getGradientAccumulationState() {
    this.#assertOpen();
    const graph = this.#shapeContext.currentGraph();
    return graph
      ? getGradientAccumulationState(graph)
      : Object.freeze({ pending: false, microbatches: 0, accumulationSteps: 0, examples: 0 });
  }

  inspectShapeState(): Readonly<TrainingShapeInspection> {
    this.#assertOpen();
    return Object.freeze({
      ...this.#shapeContext.inspect(),
      backendPlanCache: this.#driver?.inspectPlanCache?.() ?? null,
    });
  }

  close(): Promise<void> {
    if (this.#closePromise) return this.#closePromise;
    this.#lifecycle = 'closing';
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

  #replaceShapeContext(state: TrainingMutableState): void {
    this.#driver?.dispose?.();
    this.#driver = null;
    this.#stateData = state;
    this.#shapeContext = new TrainingShapeContext(
      this.#snapshot,
      this.#stateData,
      this.#shapeOptions,
    );
  }

  #assertNoPendingAccumulation(operation: string): void {
    const graph = this.#shapeContext.currentGraph();
    if (graph && getGradientAccumulationState(graph).pending) {
      throw new VolvoxAIError(
        'INVALID_ARGUMENT',
        `Trainer ${operation} requires gradient accumulation to be completed or reset.`,
        { phase: 'execution', backend: this.backend },
      );
    }
  }

  #assertOpen(): void {
    if (this.#lifecycle !== 'open') {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Trainer is closing or closed.', {
        phase: 'lifecycle', backend: this.backend,
      });
    }
  }

  #enqueue<TResult>(operation: () => Promise<TResult> | TResult): Promise<TResult> {
    if (this.#lifecycle !== 'open') {
      return Promise.reject(new VolvoxAIError(
        'HANDLE_DISPOSED',
        'Trainer is closing or closed.',
        { phase: 'lifecycle', backend: this.backend },
      ));
    }
    const pending = this.#tail.then(operation, operation);
    this.#tail = pending.then(() => undefined, () => undefined);
    return pending;
  }

  async #dispose(): Promise<void> {
    let failure: unknown = null;
    try {
      try { this.#driver?.dispose?.(); } catch (error) { failure = error; }
      this.#driver = null;
      this.#driverFactory = null;
      const disposeBackend = this.#disposeBackend;
      this.#disposeBackend = null;
      try { disposeBackend?.(); } catch (error) { failure ??= error; }
    } finally {
      this.#lifecycle = 'closed';
    }
    if (failure) throw failure;
  }
}
