import { createBoundExecutionGraph, type BoundExecutionGraph } from '../core/BoundExecutionGraph.js';
import { Model } from '../core/Model.js';
import type { ResolvedShapePlan, ResolvedTensorDescriptor } from '../core/ResolvedShapePlan.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import {
  runtimeDTypeBytes,
  type ShapedRuntimeTensorView,
} from '../ops/shapeSystem.js';
import {
  ensureTrainingGraphState,
  type OptimizerMoments,
  type StatefulTrainingGraph,
} from './TrainingGraph.js';
import type { TrainingOptimizerDescriptor } from './TrainingOptimizer.js';
import type { BackendPlanCacheInspection } from './BackendPlanCache.js';

const DEFAULT_PLAN_CACHE_ENTRIES = 8;
const DEFAULT_PLAN_CACHE_METADATA_BYTES = 1024 * 1024;
const DEFAULT_MAX_CAPACITY_BYTES = 512 * 1024 * 1024;
const DEFAULT_CAPACITY_GROWTH_FACTOR = 2;
const UTF8_ENCODER = new TextEncoder();

interface CapacitySlot {
  readonly dtype: RuntimeDType;
  readonly storage: RuntimeTypedArray;
  readonly capacityBytes: number;
}

interface PlanCacheEntry {
  readonly plan: ResolvedShapePlan;
  readonly metadataBytes: number;
  readonly tacticSignature: string;
}

export interface TrainingMutableState {
  readonly parameters: Map<string, RuntimeTypedArray>;
  optimizerState: Map<string, OptimizerMoments> | null;
  optimizerDescriptor: TrainingOptimizerDescriptor | null;
  trainingStep: number;
  trainingMetadata: unknown;
}

export interface TrainingShapeContextOptions {
  readonly planCacheEntries?: number;
  readonly planCacheMetadataBytes?: number;
  readonly maxCapacityBytes?: number;
  readonly capacityGrowthFactor?: number;
}

export interface TrainingConcreteBinding {
  readonly plan: ResolvedShapePlan;
  readonly shapeSignature: string;
  readonly tacticSignature: string;
  readonly graph: StatefulTrainingGraph;
  readonly rawInputs: Record<string, RuntimeTypedArray>;
  readonly activationShapes: Readonly<Record<string, readonly number[]>>;
  readonly activationGradientShapes: Readonly<Record<string, readonly number[]>>;
  readonly parameterGradientShapes: Readonly<Record<string, readonly number[]>>;
}

export interface TrainingShapeInspection {
  readonly currentShapeSignature: string | null;
  readonly currentTacticSignature: string | null;
  readonly planCacheEntries: number;
  readonly planCacheMetadataBytes: number;
  readonly planCacheHits: number;
  readonly planCacheMisses: number;
  readonly planCacheEvictions: number;
  readonly activationCapacityBytes: number;
  readonly activationCapacityHighWaterBytes: number;
  readonly activationGrowCount: number;
  readonly parameterBytes: number;
  readonly parameterGradientBytes: number;
  readonly optimizerBytes: number;
  readonly backendPlanCache: Readonly<BackendPlanCacheInspection> | null;
}

function positiveSafeInteger(value: unknown, fallback: number, label: string): number {
  const selected = value === undefined ? fallback : value;
  if (!Number.isSafeInteger(selected) || (selected as number) <= 0) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return selected as number;
}

function runtimeDType(value: unknown): RuntimeDType | null {
  if (value instanceof Float32Array) return 'float32';
  if (value instanceof Int32Array) return 'int32';
  if (value instanceof Int8Array) return 'int8';
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function cloneRuntimeData(value: RuntimeTypedArray): RuntimeTypedArray {
  if (value instanceof Float32Array) return new Float32Array(value);
  if (value instanceof Int32Array) return new Int32Array(value);
  if (value instanceof Int8Array) return new Int8Array(value);
  if (value instanceof Uint8ClampedArray) return new Uint8ClampedArray(value);
  return new Uint8Array(value);
}

function allocate(dtype: RuntimeDType, bytes: number): RuntimeTypedArray {
  const elements = bytes / runtimeDTypeBytes(dtype);
  if (!Number.isSafeInteger(elements) || elements < 0) {
    throw new Error(`Training activation capacity for ${dtype} has invalid byte size ${bytes}.`);
  }
  if (dtype === 'float32') return new Float32Array(elements);
  if (dtype === 'int32') return new Int32Array(elements);
  if (dtype === 'int8') return new Int8Array(elements);
  return new Uint8Array(elements);
}

function exactView(storage: RuntimeTypedArray, elements: number): RuntimeTypedArray {
  if (storage instanceof Float32Array) {
    return new Float32Array(storage.buffer, storage.byteOffset, elements);
  }
  if (storage instanceof Int32Array) {
    return new Int32Array(storage.buffer, storage.byteOffset, elements);
  }
  if (storage instanceof Int8Array) {
    return new Int8Array(storage.buffer, storage.byteOffset, elements);
  }
  if (storage instanceof Uint8ClampedArray) {
    return new Uint8ClampedArray(storage.buffer, storage.byteOffset, elements);
  }
  return new Uint8Array(storage.buffer, storage.byteOffset, elements);
}

function metadataBytes(plan: ResolvedShapePlan): number {
  return UTF8_ENCODER.encode(JSON.stringify(plan)).byteLength;
}

function tacticSignature(plan: ResolvedShapePlan): string {
  const nodes = plan.nodes.map((node) => ({
    id: node.id,
    opType: node.opType,
    shapeFunctionId: node.shapeFunctionId,
    inputs: Object.fromEntries(Object.entries(node.inputs).map(([name, value]) => [name, value.shape])),
    outputs: Object.fromEntries(Object.entries(node.outputs).map(([name, value]) => [name, value.shape])),
  }));
  return `volvox-training-tactic/v1|${JSON.stringify(nodes)}`;
}

function descriptorShapeRecord(
  plan: ResolvedShapePlan,
  predicate: (descriptor: ResolvedTensorDescriptor) => boolean,
): Readonly<Record<string, readonly number[]>> {
  return Object.freeze(Object.fromEntries(Object.entries(plan.tensors)
    .filter(([, descriptor]) => predicate(descriptor))
    .map(([name, descriptor]) => [name, descriptor.shape])));
}

function stateFromSnapshot(snapshot: Model): TrainingMutableState {
  return {
    parameters: new Map(snapshot.weightNames.map((name) => [name, snapshot.copyWeightData(name)])),
    optimizerState: null,
    optimizerDescriptor: null,
    trainingStep: 0,
    trainingMetadata: null,
  };
}

function assertParameterState(snapshot: Model, state: TrainingMutableState): void {
  if (!(state.parameters instanceof Map) || state.parameters.size !== snapshot.weightNames.length) {
    throw new Error('Training parameter state must contain exactly the logical fixed weights.');
  }
  for (const descriptor of snapshot.weightDescriptors) {
    const storage = state.parameters.get(descriptor.name);
    if (runtimeDType(storage) !== descriptor.dtype || storage!.byteLength !== descriptor.sizeBytes) {
      throw new Error(
        `Training parameter '${descriptor.name}' changed dtype, shape, or storage size.`,
      );
    }
  }
  for (const name of state.parameters.keys()) {
    if (!snapshot.hasWeight(name)) throw new Error(`Training parameter '${name}' is not logical weight.`);
  }
  if (!Number.isSafeInteger(state.trainingStep) || state.trainingStep < 0) {
    throw new Error('Training step must be a non-negative safe integer.');
  }
}

/** Create mutable fixed-size parameter state from an immutable logical revision. */
export function createTrainingMutableState(
  snapshot: Model,
): TrainingMutableState {
  if (!(snapshot instanceof Model)) {
    throw new Error('Training state requires a Model.');
  }
  return stateFromSnapshot(snapshot);
}

/**
 * Full-profile context owner. Plan entries contain immutable metadata only;
 * one growable activation pool and one concrete graph are retained regardless
 * of the number of observed signatures.
 */
export class TrainingShapeContext {
  readonly snapshot: Model;
  readonly state: TrainingMutableState;
  readonly #planCacheLimit: number;
  readonly #planMetadataLimit: number;
  readonly #capacityLimit: number;
  readonly #growthFactor: number;
  #planCache = new Map<string, PlanCacheEntry>();
  #planCacheBytes = 0;
  #planCacheHits = 0;
  #planCacheMisses = 0;
  #planCacheEvictions = 0;
  #capacities = new Map<string, CapacitySlot>();
  #capacityBytes = 0;
  #capacityHighWaterBytes = 0;
  #capacityGrowCount = 0;
  #current: BoundExecutionGraph | null = null;
  #currentTacticSignature: string | null = null;

  constructor(
    snapshot: Model,
    state: TrainingMutableState = createTrainingMutableState(snapshot),
    options: TrainingShapeContextOptions = {},
  ) {
    if (!(snapshot instanceof Model)) {
      throw new Error('TrainingShapeContext requires a Model.');
    }
    if (!options || typeof options !== 'object' || Array.isArray(options)) {
      throw new Error('TrainingShapeContext options must be an object.');
    }
    this.snapshot = snapshot;
    this.state = state;
    assertParameterState(snapshot, state);
    this.#planCacheLimit = positiveSafeInteger(
      options.planCacheEntries, DEFAULT_PLAN_CACHE_ENTRIES, 'Training planCacheEntries',
    );
    this.#planMetadataLimit = positiveSafeInteger(
      options.planCacheMetadataBytes,
      DEFAULT_PLAN_CACHE_METADATA_BYTES,
      'Training planCacheMetadataBytes',
    );
    this.#capacityLimit = positiveSafeInteger(
      options.maxCapacityBytes, DEFAULT_MAX_CAPACITY_BYTES, 'Training maxCapacityBytes',
    );
    const growth = options.capacityGrowthFactor ?? DEFAULT_CAPACITY_GROWTH_FACTOR;
    if (typeof growth !== 'number' || !Number.isFinite(growth) || growth <= 1) {
      throw new Error('Training capacityGrowthFactor must be finite and greater than one.');
    }
    this.#growthFactor = growth;
  }

  bind(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
  ): TrainingConcreteBinding {
    assertParameterState(this.snapshot, this.state);
    const plan = this.snapshot.bindShapes(inputs);
    const cached = this.#rememberPlan(plan);
    const rawInputs = Object.fromEntries(this.snapshot.inputNames.map((name) => [name, inputs[name].data]));
    if (this.#current?.signature !== plan.signature) {
      const candidateCapacities = this.#prepareCapacities(plan);
      const candidate = createBoundExecutionGraph(this.snapshot, plan, {
        invariantWeightStorageFactory: ({ name }) => this.state.parameters.get(name),
        activationStorageFactory: ({ name, dtype, elementCount }) => {
          const slot = candidateCapacities.slots.get(name);
          if (!slot || slot.dtype !== dtype) {
            throw new Error(`Training activation capacity '${name}' is missing or has the wrong dtype.`);
          }
          return exactView(slot.storage, elementCount);
        },
      });
      const graph = ensureTrainingGraphState(candidate.graph);
      graph.optimizerState = this.state.optimizerState;
      graph.optimizerDescriptor = this.state.optimizerDescriptor;
      graph.trainingStep = this.state.trainingStep;
      graph.trainingMetadata = this.state.trainingMetadata;
      graph.trainingShapeSignature = plan.signature;
      graph.trainingTacticSignature = cached.tacticSignature;
      this.#capacities = candidateCapacities.slots;
      this.#capacityBytes = candidateCapacities.bytes;
      this.#capacityHighWaterBytes = Math.max(this.#capacityHighWaterBytes, this.#capacityBytes);
      this.#capacityGrowCount += candidateCapacities.growCount;
      this.#current = candidate;
      this.#currentTacticSignature = cached.tacticSignature;
    }
    const graph = ensureTrainingGraphState(this.#current.graph);
    graph.trainingShapeSignature = plan.signature;
    graph.trainingTacticSignature = cached.tacticSignature;
    return Object.freeze({
      plan,
      shapeSignature: plan.signature,
      tacticSignature: cached.tacticSignature,
      graph,
      rawInputs,
      activationShapes: descriptorShapeRecord(plan, (descriptor) => descriptor.kind !== 'weight'),
      activationGradientShapes: descriptorShapeRecord(
        plan,
        (descriptor) => descriptor.kind !== 'weight' && descriptor.dtype === 'float32',
      ),
      parameterGradientShapes: descriptorShapeRecord(
        plan,
        (descriptor) => descriptor.kind === 'weight' && descriptor.dtype === 'float32',
      ),
    });
  }

  /** Persist shape-invariant state after the active concrete graph completes a step. */
  captureState(graph: StatefulTrainingGraph): void {
    if (this.#current?.graph !== graph) {
      throw new Error('Training state can only be captured from the active concrete binding.');
    }
    assertParameterState(this.snapshot, this.state);
    this.state.optimizerState = graph.optimizerState;
    this.state.optimizerDescriptor = graph.optimizerDescriptor;
    this.state.trainingStep = graph.trainingStep;
    this.state.trainingMetadata = graph.trainingMetadata;
    if (this.state.optimizerState) {
      for (const [name, moments] of this.state.optimizerState) {
        const parameter = this.state.parameters.get(name);
        if (!(parameter instanceof Float32Array) ||
            !(moments.m instanceof Float32Array) || !(moments.v instanceof Float32Array) ||
            moments.m.length !== parameter.length || moments.v.length !== parameter.length) {
          throw new Error(`Optimizer state for '${name}' changed its fixed parameter shape.`);
        }
      }
    }
  }

  currentGraph(): StatefulTrainingGraph | null {
    return this.#current ? ensureTrainingGraphState(this.#current.graph) : null;
  }

  inspect(): Readonly<TrainingShapeInspection> {
    assertParameterState(this.snapshot, this.state);
    const parameterBytes = [...this.state.parameters.values()]
      .reduce((total, value) => total + value.byteLength, 0);
    const parameterGradientBytes = this.snapshot.weightDescriptors
      .filter((descriptor) => descriptor.dtype === 'float32')
      .reduce((total, descriptor) => total + descriptor.sizeBytes, 0);
    const optimizerBytes = this.state.optimizerState
      ? [...this.state.optimizerState.values()]
        .reduce((total, value) => total + value.m.byteLength + value.v.byteLength, 0)
      : 0;
    return Object.freeze({
      currentShapeSignature: this.#current?.signature ?? null,
      currentTacticSignature: this.#currentTacticSignature,
      planCacheEntries: this.#planCache.size,
      planCacheMetadataBytes: this.#planCacheBytes,
      planCacheHits: this.#planCacheHits,
      planCacheMisses: this.#planCacheMisses,
      planCacheEvictions: this.#planCacheEvictions,
      activationCapacityBytes: this.#capacityBytes,
      activationCapacityHighWaterBytes: this.#capacityHighWaterBytes,
      activationGrowCount: this.#capacityGrowCount,
      parameterBytes,
      parameterGradientBytes,
      optimizerBytes,
      backendPlanCache: null,
    });
  }

  #rememberPlan(plan: ResolvedShapePlan): PlanCacheEntry {
    const prior = this.#planCache.get(plan.signature);
    if (prior) {
      this.#planCache.delete(plan.signature);
      this.#planCache.set(plan.signature, prior);
      this.#planCacheHits++;
      return prior;
    }
    this.#planCacheMisses++;
    const entry = Object.freeze({
      plan,
      metadataBytes: metadataBytes(plan),
      tacticSignature: tacticSignature(plan),
    });
    if (entry.metadataBytes > this.#planMetadataLimit) return entry;
    while (this.#planCache.size >= this.#planCacheLimit ||
           this.#planCacheBytes + entry.metadataBytes > this.#planMetadataLimit) {
      const oldest = this.#planCache.entries().next().value as [string, PlanCacheEntry] | undefined;
      if (!oldest) break;
      this.#planCache.delete(oldest[0]);
      this.#planCacheBytes -= oldest[1].metadataBytes;
      this.#planCacheEvictions++;
    }
    this.#planCache.set(plan.signature, entry);
    this.#planCacheBytes += entry.metadataBytes;
    return entry;
  }

  #prepareCapacities(plan: ResolvedShapePlan): {
    slots: Map<string, CapacitySlot>;
    bytes: number;
    growCount: number;
  } {
    const slots = new Map(this.#capacities);
    let bytes = this.#capacityBytes;
    let growCount = 0;
    for (const [name, descriptor] of Object.entries(plan.tensors)) {
      if (descriptor.kind === 'weight') continue;
      const previous = slots.get(name);
      if (previous?.dtype === descriptor.dtype && previous.capacityBytes >= descriptor.sizeBytes) continue;
      const minimum = descriptor.sizeBytes;
      const proposed = previous?.dtype === descriptor.dtype
        ? Math.max(minimum, Math.ceil(previous.capacityBytes * this.#growthFactor))
        : minimum;
      const alignment = runtimeDTypeBytes(descriptor.dtype);
      const capacityBytes = Math.ceil(proposed / alignment) * alignment;
      const nextBytes = bytes - (previous?.capacityBytes ?? 0) + capacityBytes;
      if (!Number.isSafeInteger(nextBytes) || nextBytes > this.#capacityLimit) {
        throw new Error(
          `Training activation capacity exceeds the ${this.#capacityLimit}-byte context limit.`,
        );
      }
      slots.set(name, Object.freeze({
        dtype: descriptor.dtype,
        storage: allocate(descriptor.dtype, capacityBytes),
        capacityBytes,
      }));
      bytes = nextBytes;
      growCount++;
    }
    return { slots, bytes, growCount };
  }
}

/** Defensive fixed-storage clone used by checkpoint rollback/restore. */
export function cloneTrainingMutableState(state: TrainingMutableState): TrainingMutableState {
  return {
    parameters: new Map([...state.parameters].map(([name, value]) => [name, cloneRuntimeData(value)])),
    optimizerState: state.optimizerState === null ? null : new Map(
      [...state.optimizerState].map(([name, value]) => [name, {
        m: new Float32Array(value.m),
        v: new Float32Array(value.v),
        step: value.step,
      }]),
    ),
    optimizerDescriptor: state.optimizerDescriptor === null
      ? null
      : structuredClone(state.optimizerDescriptor),
    trainingStep: state.trainingStep,
    trainingMetadata: state.trainingMetadata == null
      ? state.trainingMetadata
      : structuredClone(state.trainingMetadata),
  };
}
