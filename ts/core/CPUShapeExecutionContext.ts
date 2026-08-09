import { CPUEngine } from '../backends/CPUEngine.js';
import type { BackendExecutionOptions } from '../backends/BackendEngine.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import {
  PublicInputShapeContract,
  ShapeContractError,
  bindPublicInputShapes,
  canonicalShapeSignature,
  runtimeDTypeBytes,
  type ShapedRuntimeTensorView,
} from '../ops/shapeSystem.js';
import {
  createBoundExecutionGraph,
  type BoundActivationLivenessLayout,
  type BoundExecutionGraph,
} from './BoundExecutionGraph.js';
import {
  concreteCPUActivationArenaTensors,
  maximumCPUActivationArenaTensors,
  planCPUActivationArena,
  planCPUActivationArenaWithinMaximum,
} from './CPUActivationArenaPlan.js';
import { Model } from './Model.js';
import {
  canonicalBankResidencySuffix,
  type ResolvedShapePlan,
} from './ResolvedShapePlan.js';

export type CPUShapeExecutionContextErrorCode =
  | 'INVALID_OPTIONS'
  | 'CONTEXT_CLOSED'
  | 'CAPACITY_LIMIT_EXCEEDED'
  | 'CAPACITY_ALLOCATION_FAILED'
  | 'CAPACITY_STORAGE_MISMATCH'
  | 'MATERIALIZATION_FAILED'
  | 'EXECUTION_FAILED';

/** Stable lifecycle/resource error for the bounded-shape CPU vertical slice. */
export class CPUShapeExecutionContextError extends Error {
  readonly code: CPUShapeExecutionContextErrorCode;
  readonly path: string;

  constructor(
    code: CPUShapeExecutionContextErrorCode,
    path: string,
    message: string,
    cause?: unknown,
  ) {
    super(
      `[CPUShapeExecutionContext] ${path}: ${message}`,
      cause === undefined ? undefined : { cause },
    );
    this.name = 'CPUShapeExecutionContextError';
    this.code = code;
    this.path = path;
  }
}

export interface CPUShapeCapacityAllocationRequest {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly requiredBytes: number;
  readonly targetCapacityBytes: number;
  readonly previousCapacityBytes: number;
}

export type CPUShapeCapacityAllocator = (
  request: CPUShapeCapacityAllocationRequest,
) => RuntimeTypedArray;

export interface CPUShapeExecutionContextOptions {
  /** Maximum metadata-only dynamic plan entries. Default: 8. */
  readonly planCacheEntries?: number;
  /** Maximum canonical metadata bytes across dynamic plans. Default: 16 MiB. */
  readonly planCacheMetadataBytes?: number;
  /** Maximum sum of committed activation capacities. Default: 512 MiB. */
  readonly maxCapacityBytes?: number;
  /** Geometric multiplier used only when an existing tensor capacity grows. Default: 2. */
  readonly capacityGrowthFactor?: number;
  /**
   * Dynamic ordinary contexts use topology-liveness arenas. Incremental decode
   * contexts must retain every intermediate and use persistent tensor slots.
   * Static contexts always preserve the constant-only persistent fast path.
   */
  readonly activationStorage?: 'liveness' | 'persistent';
  /** Optional context-private allocator, primarily for embedding and deterministic tests. */
  readonly capacityAllocator?: CPUShapeCapacityAllocator;
}

export interface CPUShapeOutputSnapshot {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  /** Fresh exact logical storage owned by this result. */
  readonly data: RuntimeTypedArray;
}

export interface CPUShapeLogicalRevision {
  readonly definitionFingerprint: string;
  readonly definitionId: string;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly weightRevisionId: string;
}

/** Stable result whose storage does not depend on context or capacity lifetime. */
export class CPUShapeExecutionResult {
  readonly signature: string;
  readonly revision: CPUShapeLogicalRevision;
  readonly outputNames: readonly string[];
  readonly outputs: Readonly<Record<string, CPUShapeOutputSnapshot>>;

  constructor(
    signature: string,
    revision: CPUShapeLogicalRevision,
    outputs: readonly CPUShapeOutputSnapshot[],
  ) {
    this.signature = signature;
    this.revision = revision;
    const outputNames: string[] = [];
    const outputRecord: Record<string, CPUShapeOutputSnapshot> = {};
    for (const output of outputs) {
      outputNames.push(output.name);
      outputRecord[output.name] = Object.freeze({
        name: output.name,
        dtype: output.dtype,
        shape: Object.isFrozen(output.shape) ? output.shape : Object.freeze([...output.shape]),
        data: output.data,
      });
    }
    this.outputNames = Object.freeze(outputNames);
    this.outputs = Object.freeze(outputRecord);
    Object.freeze(this);
  }

  get(name: string): CPUShapeOutputSnapshot {
    const output = this.outputs[name];
    if (output === undefined) throw new Error(`CPU result output '${name}' does not exist.`);
    return output;
  }
}

export type CPUShapeExecutionContextState = 'open' | 'closing' | 'closed';

export interface CPUShapePlanCacheInspection {
  readonly entryCount: number;
  readonly metadataBytes: number;
  /** Exact signatures in deterministic least- to most-recently-used order. */
  readonly signatures: readonly string[];
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
  readonly oversizeSkips: number;
}

export interface CPUShapeTensorCapacityInspection {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly capacityBytes: number;
}

export interface CPUShapeCapacityInspection {
  readonly currentBytes: number;
  readonly highWaterBytes: number;
  readonly growEvents: number;
  readonly grownTensorCount: number;
  readonly tensors: readonly CPUShapeTensorCapacityInspection[];
}

export interface CPUShapeExecutionInspection {
  readonly accepted: number;
  readonly completed: number;
  readonly failed: number;
}

export interface CPUShapeExecutionContextInspection {
  readonly state: CPUShapeExecutionContextState;
  readonly staticFastPath: boolean;
  readonly currentSignature: string | null;
  readonly revision: CPUShapeLogicalRevision;
  readonly limits: Readonly<{
    planCacheEntries: number;
    planCacheMetadataBytes: number;
    maxCapacityBytes: number;
    capacityGrowthFactor: number;
    activationStorage: 'liveness' | 'persistent';
  }>;
  readonly planCache: CPUShapePlanCacheInspection;
  readonly invariantWeights: Readonly<{
    readonly tensorCount: number;
    readonly sizeBytes: number;
  }>;
  readonly capacities: CPUShapeCapacityInspection;
  readonly executions: CPUShapeExecutionInspection;
}

/** Allocation-light counters sampled by the provider on every dynamic run. */
export interface CPUShapeExecutionTelemetry {
  readonly staticFastPath: boolean;
  readonly currentSignature: string | null;
  readonly planCacheEntries: number;
  readonly planCacheMetadataBytes: number;
  readonly planCacheHits: number;
  readonly planCacheMisses: number;
  readonly planCacheEvictions: number;
  readonly planCacheOversizeSkips: number;
  readonly invariantWeightBytes: number;
  readonly activationCapacityBytes: number;
  readonly activationCapacityHighWaterBytes: number;
  readonly activationGrowCount: number;
  readonly grownTensorCount: number;
}

interface PlanCacheEntry {
  readonly plan: ResolvedShapePlan;
  readonly layout: BoundActivationLivenessLayout | null;
  readonly metadataBytes: number;
}

interface CapacitySlot {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly storage: RuntimeTypedArray;
  readonly capacityBytes: number;
}

interface CapacityCandidate {
  readonly slots: Map<string, CapacitySlot>;
  readonly currentBytes: number;
  readonly grew: boolean;
  readonly grownTensorCount: number;
}

interface PlanCandidate {
  readonly plan: ResolvedShapePlan;
  readonly layout: BoundActivationLivenessLayout | null;
  readonly cache: Map<string, PlanCacheEntry>;
  readonly cacheBytes: number;
  readonly hitDelta: number;
  readonly missDelta: number;
  readonly evictionDelta: number;
  readonly oversizeSkipDelta: number;
}

const DEFAULT_PLAN_CACHE_ENTRIES = 8;
// A resolved plan for a production graph can exceed 1 MiB even though it owns
// metadata only. Keep the byte bound explicit, but size it consistently with
// the eight-entry default so retained decode does not record one oversize miss
// per token for otherwise identical signatures.
const DEFAULT_PLAN_CACHE_METADATA_BYTES = 16 * 1024 * 1024;
const DEFAULT_MAX_CAPACITY_BYTES = 512 * 1024 * 1024;
const DEFAULT_CAPACITY_GROWTH_FACTOR = 2;
const UTF8_ENCODER = new TextEncoder();

function fail(
  code: CPUShapeExecutionContextErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): never {
  throw new CPUShapeExecutionContextError(code, path, message, cause);
}

function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length;
}

function sortedNames(value: object): string[] {
  return Object.getOwnPropertyNames(value).sort(compareCanonicalNames);
}

function checkedNonNegativeSafeInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    fail('INVALID_OPTIONS', path, 'must be a non-negative safe integer.');
  }
  return value as number;
}

function checkedAdd(left: number, right: number, path: string): number {
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0 ||
      left > Number.MAX_SAFE_INTEGER - right) {
    fail('CAPACITY_LIMIT_EXCEEDED', path, 'exceeds the safe integer range.');
  }
  return left + right;
}

function runtimeStorageMatches(dtype: RuntimeDType, value: unknown): value is RuntimeTypedArray {
  return runtimeStorageDType(value) === dtype;
}

function runtimeStorageDType(value: unknown): RuntimeDType | null {
  if (value instanceof Float32Array) return 'float32';
  if (value instanceof Int32Array) return 'int32';
  if (value instanceof Int8Array) return 'int8';
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function shapeFail(
  code: ConstructorParameters<typeof ShapeContractError>[0],
  path: string,
  message: string,
): never {
  throw new ShapeContractError(code, path, message);
}

function defaultAllocateCapacity(request: CPUShapeCapacityAllocationRequest): RuntimeTypedArray {
  const bytes = runtimeDTypeBytes(request.dtype);
  const elements = request.targetCapacityBytes / bytes;
  if (!Number.isSafeInteger(elements)) {
    fail(
      'CAPACITY_ALLOCATION_FAILED',
      `capacity '${request.name}'`,
      'target bytes do not describe an integral safe element count.',
    );
  }
  if (request.dtype === 'float32') return new Float32Array(elements);
  if (request.dtype === 'int32') return new Int32Array(elements);
  if (request.dtype === 'int8') return new Int8Array(elements);
  return new Uint8Array(elements);
}

function exactLogicalView(slot: CapacitySlot, elementCount: number): RuntimeTypedArray {
  const { storage } = slot;
  if (storage instanceof Float32Array) {
    return new Float32Array(storage.buffer, storage.byteOffset, elementCount);
  }
  if (storage instanceof Int32Array) {
    return new Int32Array(storage.buffer, storage.byteOffset, elementCount);
  }
  if (storage instanceof Int8Array) {
    return new Int8Array(storage.buffer, storage.byteOffset, elementCount);
  }
  if (storage instanceof Uint8ClampedArray) {
    return new Uint8ClampedArray(storage.buffer, storage.byteOffset, elementCount);
  }
  return new Uint8Array(storage.buffer, storage.byteOffset, elementCount);
}

function exactArenaLogicalView(
  slot: CapacitySlot,
  offsetBytes: number,
  elementCount: number,
): RuntimeTypedArray {
  const { storage } = slot;
  const byteOffset = checkedAdd(storage.byteOffset, offsetBytes, 'activation arena view offset');
  const logicalBytes = elementCount * runtimeDTypeBytes(slot.dtype);
  if (!Number.isSafeInteger(logicalBytes) || logicalBytes <= 0 ||
      offsetBytes > slot.capacityBytes - logicalBytes) {
    fail(
      'CAPACITY_STORAGE_MISMATCH',
      `capacity '${slot.name}'`,
      'cannot expose the requested exact logical arena view.',
    );
  }
  if (storage instanceof Float32Array) {
    return new Float32Array(storage.buffer, byteOffset, elementCount);
  }
  if (storage instanceof Int32Array) {
    return new Int32Array(storage.buffer, byteOffset, elementCount);
  }
  if (storage instanceof Int8Array) {
    return new Int8Array(storage.buffer, byteOffset, elementCount);
  }
  if (storage instanceof Uint8ClampedArray) {
    return new Uint8ClampedArray(storage.buffer, byteOffset, elementCount);
  }
  return new Uint8Array(storage.buffer, byteOffset, elementCount);
}

function cloneRuntimeData(data: RuntimeTypedArray): RuntimeTypedArray {
  if (data instanceof Float32Array) return new Float32Array(data);
  if (data instanceof Int32Array) return new Int32Array(data);
  if (data instanceof Int8Array) return new Int8Array(data);
  if (data instanceof Uint8ClampedArray) return new Uint8ClampedArray(data);
  return new Uint8Array(data);
}

function canonicalMetadataJson(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('plan metadata contains a non-finite number.');
    return JSON.stringify(value);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return `[${value.map(canonicalMetadataJson).join(',')}]`;
  if (value !== null && typeof value === 'object' && !ArrayBuffer.isView(value) &&
      !(value instanceof ArrayBuffer)) {
    const record = value as Readonly<Record<string, unknown>>;
    return `{${sortedNames(record).map((name) =>
      `${JSON.stringify(name)}:${canonicalMetadataJson(record[name])}`).join(',')}}`;
  }
  throw new Error('plan metadata contains unsupported storage or a non-JSON value.');
}

/** Deterministic byte charge for one immutable plan-cache entry. */
export function cpuShapePlanMetadataBytes(plan: ResolvedShapePlan): number {
  return UTF8_ENCODER.encode(canonicalMetadataJson(plan)).byteLength;
}

/** Deterministic charge for one cached topology-liveness offset layout. */
export function cpuActivationArenaMetadataBytes(
  layout: BoundActivationLivenessLayout | null,
): number {
  return layout === null ? 0 : UTF8_ENCODER.encode(canonicalMetadataJson(layout)).byteLength;
}

function overlaps(
  left: RuntimeTypedArray,
  right: RuntimeTypedArray,
): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftEnd = left.byteOffset + left.byteLength;
  const rightEnd = right.byteOffset + right.byteLength;
  return left.byteOffset < rightEnd && right.byteOffset < leftEnd;
}

function applyStaticIdentityAliases(bound: BoundExecutionGraph): void {
  // Constant-only graphs have already passed complete plan and kernel-contract
  // validation. Identity is a true storage view, so its output may share the
  // exact input view without changing any observable tensor shape or result
  // ownership. The context still owns its conservative per-tensor capacities;
  // this only removes a redundant full-tensor copy from the static fast path.
  for (const node of bound.graph.nodes) {
    if (node.opType !== 'Identity') continue;
    const input = node.inputs?.input;
    const output = node.outputs?.out;
    if (!input?.buffer || !output?.buffer || input.dtype !== output.dtype ||
        input.sizeBytes !== output.sizeBytes || input.shape.length !== output.shape.length ||
        input.shape.some((dimension, axis) => dimension !== output.shape[axis])) {
      fail(
        'MATERIALIZATION_FAILED',
        `static Identity '${node.id}'`,
        'cannot alias descriptors that are not exactly shape- and dtype-equivalent.',
      );
    }
    output.buffer = input.buffer;
  }
}

/**
 * CPU implementation of the dynamic-v1 context lifecycle. The provider SPI uses
 * executeResolved() after the public runtime has committed one exact logical
 * shape plan; direct tests and embeddings may use execute() to bind locally.
 */
export class CPUShapeExecutionContext {
  readonly snapshot: Model;
  readonly revision: CPUShapeLogicalRevision;
  readonly limits: CPUShapeExecutionContextInspection['limits'];

  readonly #inputContract: PublicInputShapeContract;
  readonly #allocator: CPUShapeCapacityAllocator;
  readonly #maximumTensorBytes: ReadonlyMap<string, number>;
  readonly #maximumArenaBytes: ReadonlyMap<string, number>;
  readonly #maximumArenaLayout: BoundActivationLivenessLayout | null;
  #state: CPUShapeExecutionContextState = 'open';
  #tail: Promise<void> = Promise.resolve();
  #closePromise: Promise<void> | null = null;
  #planCache = new Map<string, PlanCacheEntry>();
  #planCacheBytes = 0;
  #capacities = new Map<string, CapacitySlot>();
  #invariantWeights = new Map<string, RuntimeTypedArray>();
  #invariantWeightBytes = 0;
  #capacityBytes = 0;
  #capacityHighWaterBytes = 0;
  #capacityGrowEvents = 0;
  #grownTensorCount = 0;
  #cacheHits = 0;
  #cacheMisses = 0;
  #cacheEvictions = 0;
  #cacheOversizeSkips = 0;
  #acceptedExecutions = 0;
  #completedExecutions = 0;
  #failedExecutions = 0;
  #currentPlan: ResolvedShapePlan | null = null;
  #currentBound: BoundExecutionGraph | null = null;
  #engine: CPUEngine | null = null;

  constructor(
    snapshot: Model,
    options: CPUShapeExecutionContextOptions = {},
  ) {
    if (!(snapshot instanceof Model)) {
      fail('INVALID_OPTIONS', 'snapshot', 'must be a Model.');
    }
    if (options === null || typeof options !== 'object' || Array.isArray(options)) {
      fail('INVALID_OPTIONS', 'options', 'must be an object.');
    }
    const planCacheEntries = options.planCacheEntries === undefined
      ? DEFAULT_PLAN_CACHE_ENTRIES
      : checkedNonNegativeSafeInteger(options.planCacheEntries, 'options.planCacheEntries');
    const planCacheMetadataBytes = options.planCacheMetadataBytes === undefined
      ? DEFAULT_PLAN_CACHE_METADATA_BYTES
      : checkedNonNegativeSafeInteger(
        options.planCacheMetadataBytes,
        'options.planCacheMetadataBytes',
      );
    const maxCapacityBytes = options.maxCapacityBytes === undefined
      ? DEFAULT_MAX_CAPACITY_BYTES
      : checkedNonNegativeSafeInteger(options.maxCapacityBytes, 'options.maxCapacityBytes');
    const capacityGrowthFactor = options.capacityGrowthFactor === undefined
      ? DEFAULT_CAPACITY_GROWTH_FACTOR
      : options.capacityGrowthFactor;
    if (typeof capacityGrowthFactor !== 'number' || !Number.isFinite(capacityGrowthFactor) ||
        capacityGrowthFactor <= 1) {
      fail('INVALID_OPTIONS', 'options.capacityGrowthFactor', 'must be finite and greater than one.');
    }
    if (options.capacityAllocator !== undefined &&
        typeof options.capacityAllocator !== 'function') {
      fail('INVALID_OPTIONS', 'options.capacityAllocator', 'must be a function.');
    }
    const requestedActivationStorage = options.activationStorage ?? 'liveness';
    if (requestedActivationStorage !== 'liveness' && requestedActivationStorage !== 'persistent') {
      fail(
        'INVALID_OPTIONS',
        'options.activationStorage',
        "must be 'liveness' or 'persistent'.",
      );
    }
    const activationStorage = snapshot.isStatic ? 'persistent' : requestedActivationStorage;

    let maximumTensors: ReturnType<typeof maximumCPUActivationArenaTensors>;
    try {
      maximumTensors = maximumCPUActivationArenaTensors(snapshot.graph);
    } catch (error) {
      fail(
        'INVALID_OPTIONS',
        'snapshot maximum activation domain',
        'could not be represented safely.',
        error,
      );
    }
    this.#maximumTensorBytes = new Map(Object.entries(maximumTensors)
      .filter(([, descriptor]) => descriptor.kind !== 'weight')
      .map(([name, descriptor]) => [name, descriptor.sizeBytes]));
    if (activationStorage === 'liveness') {
      let maximumLayout: BoundActivationLivenessLayout;
      try {
        maximumLayout = planCPUActivationArena(
          snapshot.graph,
          `${snapshot.definitionFingerprint}|maximum-domain`,
          maximumTensors,
        );
      } catch (error) {
        fail(
          'INVALID_OPTIONS',
          'snapshot maximum activation arena',
          'could not prove a bounded topology-liveness layout.',
          error,
        );
      }
      this.#maximumArenaLayout = maximumLayout;
      this.#maximumArenaBytes = new Map(Object.entries(maximumLayout.capacityByArena));
    } else {
      this.#maximumArenaLayout = null;
      this.#maximumArenaBytes = new Map();
    }

    this.snapshot = snapshot;
    this.revision = Object.freeze({
      definitionFingerprint: snapshot.definitionFingerprint,
      definitionId: snapshot.definitionId,
      topologyRevision: snapshot.topologyRevision,
      weightRevision: snapshot.weightRevision,
      weightRevisionId: snapshot.weightRevisionId,
    });
    this.limits = Object.freeze({
      planCacheEntries,
      planCacheMetadataBytes,
      maxCapacityBytes,
      capacityGrowthFactor,
      activationStorage,
    });
    this.#allocator = options.capacityAllocator ?? defaultAllocateCapacity;
    this.#inputContract = new PublicInputShapeContract(
      snapshot.graph.environment,
      snapshot.inputDescriptors.map((input) => ({
        name: input.name,
        dtype: input.dtype,
        shape: input.shape,
      })),
    );
  }

  get state(): CPUShapeExecutionContextState {
    return this.#state;
  }

  execute(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
  ): Promise<CPUShapeExecutionResult> {
    return this.#enqueueExecution(() => this.#executeOne(inputs));
  }

  /**
   * @internal Provider entry. The public lifecycle has already performed
   * graph-wide resolution; this path revalidates the concrete input ownership
   * contract but deliberately never invokes logical shape inference again.
   */
  executeResolved(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
    plan: ResolvedShapePlan,
    options: BackendExecutionOptions = {},
    beforeDispatch?: (() => void),
  ): Promise<CPUShapeExecutionResult> {
    return this.#enqueueExecution(() => this.#executeOne(inputs, plan, options, beforeDispatch));
  }

  /** @internal Provider decode invalidation; public FIFO owns serialization. */
  resetDecodeCache(): void {
    this.#engine?.resetDecodeCache();
  }

  #enqueueExecution(
    execute: () => Promise<CPUShapeExecutionResult>,
  ): Promise<CPUShapeExecutionResult> {
    if (this.#state !== 'open') {
      return Promise.reject(new CPUShapeExecutionContextError(
        'CONTEXT_CLOSED',
        'execute',
        `cannot accept work while context state is '${this.#state}'.`,
      ));
    }
    this.#acceptedExecutions++;
    const operation = this.#tail.then(
      execute,
      execute,
    );
    this.#tail = operation.then(
      () => { this.#completedExecutions++; },
      () => { this.#failedExecutions++; },
    );
    return operation;
  }

  /** Starting close rejects new work and drains every already accepted execute. */
  close(): Promise<void> {
    if (this.#closePromise !== null) return this.#closePromise;
    this.#state = 'closing';
    this.#closePromise = this.#tail.then(() => {
      this.#engine?.dispose();
      this.#engine = null;
      this.#currentBound = null;
      this.#currentPlan = null;
      this.#capacities.clear();
      this.#invariantWeights.clear();
      this.#invariantWeightBytes = 0;
      this.#capacityBytes = 0;
      this.#planCache.clear();
      this.#planCacheBytes = 0;
      this.#state = 'closed';
    });
    return this.#closePromise;
  }

  inspect(): CPUShapeExecutionContextInspection {
    const capacityTensors = Object.freeze(
      [...this.#capacities.values()]
        .sort((left, right) => compareCanonicalNames(left.name, right.name))
        .map((slot) => Object.freeze({
          name: slot.name,
          dtype: slot.dtype,
          capacityBytes: slot.capacityBytes,
        })),
    );
    return Object.freeze({
      state: this.#state,
      staticFastPath: this.snapshot.isStatic,
      currentSignature: this.#currentPlan?.signature ?? null,
      revision: this.revision,
      limits: this.limits,
      planCache: Object.freeze({
        entryCount: this.#planCache.size,
        metadataBytes: this.#planCacheBytes,
        signatures: Object.freeze([...this.#planCache.keys()]),
        hits: this.#cacheHits,
        misses: this.#cacheMisses,
        evictions: this.#cacheEvictions,
        oversizeSkips: this.#cacheOversizeSkips,
      }),
      invariantWeights: Object.freeze({
        tensorCount: this.#invariantWeights.size,
        sizeBytes: this.#invariantWeightBytes,
      }),
      capacities: Object.freeze({
        currentBytes: this.#capacityBytes,
        highWaterBytes: this.#capacityHighWaterBytes,
        growEvents: this.#capacityGrowEvents,
        grownTensorCount: this.#grownTensorCount,
        tensors: capacityTensors,
      }),
      executions: Object.freeze({
        accepted: this.#acceptedExecutions,
        completed: this.#completedExecutions,
        failed: this.#failedExecutions,
      }),
    });
  }

  /** Provider hot-path telemetry without sorting or materializing capacity lists. */
  telemetry(): Readonly<CPUShapeExecutionTelemetry> {
    return Object.freeze({
      staticFastPath: this.snapshot.isStatic,
      currentSignature: this.#currentPlan?.signature ?? null,
      planCacheEntries: this.#planCache.size,
      planCacheMetadataBytes: this.#planCacheBytes,
      planCacheHits: this.#cacheHits,
      planCacheMisses: this.#cacheMisses,
      planCacheEvictions: this.#cacheEvictions,
      planCacheOversizeSkips: this.#cacheOversizeSkips,
      invariantWeightBytes: this.#invariantWeightBytes,
      activationCapacityBytes: this.#capacityBytes,
      activationCapacityHighWaterBytes: this.#capacityHighWaterBytes,
      activationGrowCount: this.#capacityGrowEvents,
      grownTensorCount: this.#grownTensorCount,
    });
  }

  #executeOne(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
    resolvedPlan: ResolvedShapePlan | null = null,
    executionOptions: BackendExecutionOptions = {},
    beforeDispatch?: (() => void),
  ): Promise<CPUShapeExecutionResult> {
      let signature: string;
      let normalizedInputs: Readonly<Record<string, ShapedRuntimeTensorView>>;
      let rawInputs: Record<string, RuntimeTypedArray>;
      if (resolvedPlan !== null) {
        rawInputs = this.snapshot.isStatic && resolvedPlan === this.snapshot.staticShapePlan
          ? this.#validateStaticResolvedInputs(inputs, resolvedPlan)
          : this.#validateResolvedInputs(inputs, resolvedPlan);
        normalizedInputs = inputs;
        signature = resolvedPlan.signature;
        if (this.snapshot.isStatic && this.#currentPlan === resolvedPlan &&
            this.#currentBound !== null && this.#engine !== null) {
          return this.#dispatch(
            resolvedPlan, this.#engine, rawInputs, executionOptions, beforeDispatch,
          );
        }
      } else if (this.snapshot.isStatic) {
        // Validate against the pre-resolved concrete descriptors without
        // rebuilding shapes, symbols, or a signature on every constant run.
        rawInputs = this.#validateStaticInputs(inputs);
        signature = this.snapshot.staticShapePlan!.signature;
        normalizedInputs = Object.freeze({});
        if (this.#currentPlan !== null && this.#currentBound !== null && this.#engine !== null) {
          return this.#dispatch(
            this.#currentPlan, this.#engine, rawInputs, executionOptions, beforeDispatch,
          );
        }
      } else {
        // This complete public bind validates names, dtype, rank, shape, bounds,
        // symbol equality, and exact bytes without copying caller data.
        const binding = bindPublicInputShapes(this.#inputContract, inputs);
        normalizedInputs = Object.freeze(Object.fromEntries(
          this.snapshot.inputNames.map((name) => [name, Object.freeze({
            data: binding.inputs[name].data,
            shape: binding.inputs[name].shape,
          })]),
        ));
        rawInputs = Object.fromEntries(
          this.snapshot.inputNames.map((name) => [name, binding.inputs[name].data]),
        );
        signature = binding.signature;
      }
      const planCandidate = this.#preparePlan(signature, normalizedInputs, resolvedPlan);
      if (this.#currentPlan?.signature === planCandidate.plan.signature &&
          this.#currentBound !== null && this.#engine !== null) {
        // An exact-signature hit has identical concrete tensor requirements.
        // Capacity and graph materialization are private and immutable between
        // queued executions, so skip the transactional capacity scan entirely.
        this.#publishPlanCandidate(planCandidate);
        return this.#dispatch(
          this.#currentPlan, this.#engine, rawInputs, executionOptions, beforeDispatch,
        );
      }
      const capacityCandidate = this.#prepareCapacities(
        planCandidate.plan,
        planCandidate.layout,
      );
      const canReuseMaterialization = this.#currentPlan?.signature === planCandidate.plan.signature &&
        !capacityCandidate.grew && this.#currentBound !== null && this.#engine !== null;

      let candidateBound = this.#currentBound;
      let candidateEngine = this.#engine;
      let candidateInvariantWeights: Map<string, RuntimeTypedArray> =
        this.#invariantWeights;
      if (!canReuseMaterialization) {
        candidateBound = null;
        candidateEngine = null;
        try {
          candidateBound = createBoundExecutionGraph(this.snapshot, planCandidate.plan, {
            invariantWeightStorageFactory: (request) =>
              this.#invariantWeights.get(request.name),
            activationStorageFactory: (request) => {
              const region = planCandidate.layout?.regions[request.name];
              const slotName = region === undefined
                ? request.name
                : `@arena/${region.arena}`;
              const slot = capacityCandidate.slots.get(slotName);
              if (slot === undefined) {
                fail(
                  'CAPACITY_STORAGE_MISMATCH',
                  `capacity '${slotName}'`,
                  'is absent from the complete capacity candidate.',
                );
              }
              return region === undefined
                ? exactLogicalView(slot, request.elementCount)
                : exactArenaLogicalView(slot, region.offsetBytes, request.elementCount);
            },
            ...(planCandidate.layout === null
              ? {}
              : { activationStorageLayout: planCandidate.layout }),
          });
          if (this.snapshot.isStatic) applyStaticIdentityAliases(candidateBound);
          candidateEngine = new CPUEngine();
          candidateEngine.allocateGraph(candidateBound.graph);
          const stagedInvariantWeights = new Map(this.#invariantWeights);
          for (const name of this.snapshot.weightNames) {
            const storage = candidateBound!.graph.getTensor(name)?.buffer;
            const descriptor = planCandidate.plan.tensors[name];
            if (!runtimeStorageMatches(descriptor.dtype, storage) ||
                storage.byteLength !== descriptor.sizeBytes) {
              fail(
                'MATERIALIZATION_FAILED',
                `invariant weight '${name}'`,
                'was not materialized as exact typed storage.',
              );
            }
            stagedInvariantWeights.set(name, storage);
          }
          candidateInvariantWeights = stagedInvariantWeights;
        } catch (error) {
          candidateEngine?.dispose();
          if (error instanceof CPUShapeExecutionContextError) throw error;
          fail(
            'MATERIALIZATION_FAILED',
            'candidate graph',
            'could not materialize complete context-private CPU resources.',
            error,
          );
        }
      }
      if (candidateBound === null || candidateEngine === null) {
        fail('MATERIALIZATION_FAILED', 'candidate graph', 'was not completely prepared.');
      }

      // Atomic synchronous publication: no input copy or dispatch has happened.
      const previousEngine = this.#engine;
      this.#publishPlanCandidate(planCandidate);
      this.#capacities = capacityCandidate.slots;
      this.#invariantWeights = candidateInvariantWeights;
      if (this.#invariantWeightBytes === 0 && candidateInvariantWeights.size !== 0) {
        this.#invariantWeightBytes = [...candidateInvariantWeights.values()].reduce(
          (total, storage) => total + storage.byteLength,
          0,
        );
      }
      this.#capacityBytes = capacityCandidate.currentBytes;
      this.#capacityHighWaterBytes = Math.max(
        this.#capacityHighWaterBytes,
        capacityCandidate.currentBytes,
      );
      if (capacityCandidate.grew) {
        this.#capacityGrowEvents++;
        this.#grownTensorCount += capacityCandidate.grownTensorCount;
      }
      this.#currentPlan = planCandidate.plan;
      this.#currentBound = candidateBound;
      this.#engine = candidateEngine;
      if (previousEngine !== null && previousEngine !== candidateEngine) previousEngine.dispose();

      return this.#dispatch(
        planCandidate.plan, candidateEngine, rawInputs, executionOptions, beforeDispatch,
      );
  }

  #validateResolvedInputs(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
    plan: ResolvedShapePlan,
  ): Record<string, RuntimeTypedArray> {
    if (!plan || typeof plan !== 'object' || plan.graphFingerprint !== this.snapshot.definitionFingerprint) {
      fail(
        'MATERIALIZATION_FAILED',
        'resolved shape plan',
        'does not belong to this logical model definition.',
      );
    }
    if (inputs == null || typeof inputs !== 'object' || Array.isArray(inputs) ||
        ArrayBuffer.isView(inputs) || Object.getOwnPropertySymbols(inputs).length !== 0) {
      shapeFail('INVALID_INPUT_SET', 'execution inputs', 'must be a named tensor-view record.');
    }
    const actualNames = Object.getOwnPropertyNames(inputs).sort(compareCanonicalNames);
    if (actualNames.length !== this.snapshot.inputNames.length ||
        this.snapshot.inputNames.some((name) => !Object.prototype.hasOwnProperty.call(inputs, name))) {
      shapeFail(
        'INVALID_INPUT_SET',
        'execution inputs',
        'must contain exactly the declared public inputs.',
      );
    }
    const rawInputs: Record<string, RuntimeTypedArray> = {};
    const signatureInputs: Array<{ readonly name: string; readonly shape: readonly number[] }> = [];
    for (const name of this.snapshot.inputNames) {
      const descriptor = plan.tensors[name];
      const view = inputs[name];
      const path = `execution input '${name}'`;
      if (!descriptor || descriptor.kind !== 'input') {
        fail('MATERIALIZATION_FAILED', `resolved input '${name}'`, 'is absent from the plan.');
      }
      if (!view || typeof view !== 'object' || Array.isArray(view)) {
        shapeFail('INVALID_TENSOR_VIEW', path, 'must be an object containing data and shape.');
      }
      const dtype = runtimeStorageDType(view.data);
      if (dtype !== descriptor.dtype) {
        shapeFail(
          dtype === null ? 'INVALID_TENSOR_VIEW' : 'DTYPE_MISMATCH',
          `${path}.data`,
          `must expose exact ${descriptor.dtype} storage.`,
        );
      }
      if (!Array.isArray(view.shape) || view.shape.length !== descriptor.shape.length) {
        shapeFail('RANK_MISMATCH', `${path}.shape`,
          `must have resolved rank ${descriptor.shape.length}.`);
      }
      for (let axis = 0; axis < descriptor.shape.length; axis++) {
        if (view.shape[axis] !== descriptor.shape[axis]) {
          shapeFail('DIMENSION_MISMATCH', `${path}.shape[${axis}]`,
            `must equal resolved dimension ${descriptor.shape[axis]}.`);
        }
      }
      if (view.data.byteLength !== descriptor.sizeBytes) {
        shapeFail('BYTE_LENGTH_MISMATCH', `${path}.data`,
          `must contain exactly ${descriptor.sizeBytes} logical bytes.`);
      }
      rawInputs[name] = view.data;
      signatureInputs.push({ name, shape: descriptor.shape });
    }
    // Bank residency is part of plan identity, so recompute the same suffix
    // the resolver appended rather than comparing shapes alone.
    const expectedSignature = `${canonicalShapeSignature(signatureInputs)}` +
      canonicalBankResidencySuffix(plan.bankResidency);
    if (expectedSignature !== plan.signature) {
      fail(
        'MATERIALIZATION_FAILED',
        'resolved shape plan signature',
        'does not match its complete concrete public-input descriptors.',
      );
    }
    return rawInputs;
  }

  /**
   * The provider runtime has already validated a static input set and
   * canonicalized every view to the exact descriptor shape object. Recheck the
   * ownership-sensitive storage fields and exact input set without repeating
   * rank/dimension arithmetic.
   */
  #validateStaticResolvedInputs(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
    plan: ResolvedShapePlan,
  ): Record<string, RuntimeTypedArray> {
    if (inputs == null || typeof inputs !== 'object' || Array.isArray(inputs) ||
        ArrayBuffer.isView(inputs) || Object.getOwnPropertySymbols(inputs).length !== 0 ||
        Object.getOwnPropertyNames(inputs).length !== this.snapshot.inputNames.length) {
      shapeFail('INVALID_INPUT_SET', 'execution inputs', 'must be the complete resolved input set.');
    }
    const rawInputs: Record<string, RuntimeTypedArray> = {};
    for (const name of this.snapshot.inputNames) {
      const descriptor = plan.tensors[name];
      const view = inputs[name];
      const path = `execution input '${name}'`;
      if (!Object.prototype.hasOwnProperty.call(inputs, name) || !view ||
          typeof view !== 'object' || Array.isArray(view) || view.shape !== descriptor.shape) {
        shapeFail('INVALID_TENSOR_VIEW', path, 'must use the canonical resolved descriptor.');
      }
      if (!runtimeStorageMatches(descriptor.dtype, view.data)) {
        shapeFail('DTYPE_MISMATCH', `${path}.data`,
          `must expose exact ${descriptor.dtype} storage.`);
      }
      if (view.data.byteLength !== descriptor.sizeBytes) {
        shapeFail('BYTE_LENGTH_MISMATCH', `${path}.data`,
          `must contain exactly ${descriptor.sizeBytes} logical bytes.`);
      }
      rawInputs[name] = view.data;
    }
    return rawInputs;
  }

  #validateStaticInputs(
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
  ): Record<string, RuntimeTypedArray> {
    if (inputs == null || typeof inputs !== 'object' || Array.isArray(inputs) ||
        ArrayBuffer.isView(inputs)) {
      shapeFail('INVALID_INPUT_SET', 'execution inputs', 'must be a named tensor-view record.');
    }
    if (Object.getOwnPropertySymbols(inputs).length !== 0) {
      shapeFail('INVALID_INPUT_SET', 'execution inputs', 'must not contain symbol-keyed inputs.');
    }
    const actualNames = Object.getOwnPropertyNames(inputs);
    const exactSet = actualNames.length === this.snapshot.inputNames.length &&
      this.snapshot.inputNames.every((name) => Object.prototype.hasOwnProperty.call(inputs, name));
    if (!exactSet) {
      const actual = new Set(actualNames);
      const expected = new Set(this.snapshot.inputNames);
      const missing = this.snapshot.inputNames.filter((name) => !actual.has(name));
      const unexpected = actualNames.filter((name) => !expected.has(name)).sort(compareCanonicalNames);
      const details: string[] = [];
      if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
      if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
      shapeFail(
        'INVALID_INPUT_SET',
        'execution inputs',
        `must contain exactly the declared public inputs; ${details.join('; ')}.`,
      );
    }

    const plan = this.snapshot.staticShapePlan!;
    const rawInputs: Record<string, RuntimeTypedArray> = {};
    for (const name of this.snapshot.inputNames) {
      const path = `execution input '${name}'`;
      const view = inputs[name];
      if (view == null || typeof view !== 'object' || Array.isArray(view)) {
        shapeFail('INVALID_TENSOR_VIEW', path, 'must be an object containing data and shape.');
      }
      const descriptor = plan.tensors[name];
      const actualDType = runtimeStorageDType(view.data);
      if (actualDType === null) {
        shapeFail('INVALID_TENSOR_VIEW', `${path}.data`, 'must be a supported runtime typed array.');
      }
      if (actualDType !== descriptor.dtype) {
        shapeFail(
          'DTYPE_MISMATCH',
          `${path}.data`,
          `has dtype '${actualDType}', but the logical descriptor requires '${descriptor.dtype}'.`,
        );
      }
      if (!Array.isArray(view.shape)) {
        shapeFail('INVALID_TENSOR_VIEW', `${path}.shape`, 'must be an explicit number array.');
      }
      if (view.shape.length !== descriptor.shape.length) {
        shapeFail(
          'RANK_MISMATCH',
          `${path}.shape`,
          `has rank ${view.shape.length}, but the logical descriptor requires rank ${descriptor.shape.length}.`,
        );
      }
      for (let axis = 0; axis < descriptor.shape.length; axis++) {
        const dimension = view.shape[axis];
        const dimensionPath = `${path}.shape[${axis}]`;
        if (!Number.isSafeInteger(dimension) || dimension <= 0) {
          shapeFail('INVALID_TENSOR_VIEW', dimensionPath, 'must be a positive safe integer.');
        }
        if (dimension !== descriptor.shape[axis]) {
          shapeFail(
            'DIMENSION_MISMATCH',
            dimensionPath,
            `is ${dimension}, but the resolved static descriptor requires ${descriptor.shape[axis]}.`,
          );
        }
      }
      if (view.data.byteLength !== descriptor.sizeBytes) {
        shapeFail(
          'BYTE_LENGTH_MISMATCH',
          `${path}.data`,
          `has ${view.data.byteLength} bytes, but shape [${descriptor.shape.join(', ')}] ` +
          `and dtype '${descriptor.dtype}' require ${descriptor.sizeBytes}.`,
        );
      }
      rawInputs[name] = view.data;
    }
    return rawInputs;
  }

  async #dispatch(
    plan: ResolvedShapePlan,
    engine: CPUEngine,
    rawInputs: Record<string, RuntimeTypedArray>,
    executionOptions: BackendExecutionOptions,
    beforeDispatch?: (() => void),
  ): Promise<CPUShapeExecutionResult> {
    let rawOutputs: Record<string, unknown>;
    try {
      beforeDispatch?.();
      rawOutputs = await engine.execute(rawInputs, executionOptions) as Record<string, unknown>;
    } catch (error) {
      fail('EXECUTION_FAILED', 'CPU dispatch', 'failed after binding was committed.', error);
    }
    const outputs = plan.outputs.map((descriptor) => {
      const storage = rawOutputs[descriptor.name];
      if (!runtimeStorageMatches(descriptor.dtype, storage) ||
          storage.byteLength !== descriptor.sizeBytes) {
        fail(
          'EXECUTION_FAILED',
          `output '${descriptor.name}'`,
          `must expose exact ${descriptor.dtype} storage with ${descriptor.sizeBytes} bytes.`,
        );
      }
      return {
        name: descriptor.name,
        dtype: descriptor.dtype,
        shape: descriptor.shape,
        data: cloneRuntimeData(storage),
      };
    });
    return new CPUShapeExecutionResult(plan.signature, this.revision, outputs);
  }

  #preparePlan(
    signature: string,
    inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
    resolvedPlan: ResolvedShapePlan | null = null,
  ): PlanCandidate {
    if (this.snapshot.isStatic) {
      const plan = resolvedPlan ?? this.snapshot.staticShapePlan!;
      if (plan.signature !== signature) {
        fail(
          'MATERIALIZATION_FAILED',
          'static shape plan',
          'does not match the completely validated public-input signature.',
        );
      }
      return {
        plan,
        layout: null,
        cache: this.#planCache,
        cacheBytes: this.#planCacheBytes,
        hitDelta: 0,
        missDelta: 0,
        evictionDelta: 0,
        oversizeSkipDelta: 0,
      };
    }

    const cache = new Map(this.#planCache);
    const cached = cache.get(signature);
    if (cached !== undefined) {
      cache.delete(signature);
      // Within one immutable graph definition the canonical public-input
      // signature uniquely determines every resolved tensor. Keep the already
      // proved immutable entry instead of serializing an equivalent fresh plan
      // supplied by the public bind on every execution.
      cache.set(signature, cached);
      return {
        plan: cached.plan,
        layout: cached.layout,
        cache,
        cacheBytes: this.#planCacheBytes,
        hitDelta: 1,
        missDelta: 0,
        evictionDelta: 0,
        oversizeSkipDelta: 0,
      };
    }

    const plan = resolvedPlan ?? this.snapshot.bindShapes(inputs);
    if (plan.signature !== signature) {
      fail(
        'MATERIALIZATION_FAILED',
        'resolved shape plan',
        'does not match the completely validated public-input signature.',
      );
    }
    let layout: BoundActivationLivenessLayout | null = null;
    if (this.limits.activationStorage === 'liveness') {
      try {
        layout = planCPUActivationArenaWithinMaximum(
          this.snapshot.graph,
          plan.signature,
          concreteCPUActivationArenaTensors(plan),
          this.#maximumArenaLayout!,
        );
      } catch (error) {
        fail(
          'MATERIALIZATION_FAILED',
          'activation liveness layout',
          'could not prepare an exact topology-bound arena.',
          error,
        );
      }
    }
    const metadataBytes = checkedAdd(
      cpuShapePlanMetadataBytes(plan),
      cpuActivationArenaMetadataBytes(layout),
      'prepared plan metadata bytes',
    );
    if (this.limits.planCacheEntries === 0 ||
        metadataBytes > this.limits.planCacheMetadataBytes) {
      return {
        plan,
        layout,
        cache,
        cacheBytes: this.#planCacheBytes,
        hitDelta: 0,
        missDelta: 1,
        evictionDelta: 0,
        oversizeSkipDelta: 1,
      };
    }

    cache.set(signature, Object.freeze({ plan, layout, metadataBytes }));
    let cacheBytes = checkedAdd(
      this.#planCacheBytes,
      metadataBytes,
      'plan cache metadata bytes',
    );
    let evictions = 0;
    while (cache.size > this.limits.planCacheEntries ||
           cacheBytes > this.limits.planCacheMetadataBytes) {
      const oldestSignature = cache.keys().next().value as string | undefined;
      if (oldestSignature === undefined) break;
      const oldest = cache.get(oldestSignature)!;
      cache.delete(oldestSignature);
      cacheBytes -= oldest.metadataBytes;
      evictions++;
    }
    return {
      plan,
      layout,
      cache,
      cacheBytes,
      hitDelta: 0,
      missDelta: 1,
      evictionDelta: evictions,
      oversizeSkipDelta: 0,
    };
  }

  #publishPlanCandidate(candidate: PlanCandidate): void {
    this.#planCache = candidate.cache;
    this.#planCacheBytes = candidate.cacheBytes;
    this.#cacheHits += candidate.hitDelta;
    this.#cacheMisses += candidate.missDelta;
    this.#cacheEvictions += candidate.evictionDelta;
    this.#cacheOversizeSkips += candidate.oversizeSkipDelta;
  }

  #prepareCapacities(
    plan: ResolvedShapePlan,
    layout: BoundActivationLivenessLayout | null,
  ): CapacityCandidate {
    const requirements: Array<{
      readonly name: string;
      readonly dtype: RuntimeDType;
      readonly requiredBytes: number;
      readonly maximumBytes: number;
    }> = layout === null
      ? sortedNames(plan.tensors)
        .map((name) => plan.tensors[name])
        .filter((descriptor) => descriptor.kind !== 'weight')
        .map((descriptor) => Object.freeze({
          name: descriptor.name,
          dtype: descriptor.dtype,
          requiredBytes: descriptor.sizeBytes,
          maximumBytes: this.#maximumTensorBytes.get(descriptor.name)!,
        }))
      : sortedNames(layout.capacityByArena).map((arena) => Object.freeze({
        name: `@arena/${arena}`,
        dtype: arena as RuntimeDType,
        requiredBytes: layout.capacityByArena[arena],
        maximumBytes: this.#maximumArenaBytes.get(arena)!,
      }));
    const targets = new Map<string, number>();
    let totalBytes = 0;
    let grownTensorCount = 0;
    for (const descriptor of requirements) {
      const current = this.#capacities.get(descriptor.name);
      if (current !== undefined && current.dtype !== descriptor.dtype) {
        fail(
          'CAPACITY_STORAGE_MISMATCH',
          `capacity '${descriptor.name}'`,
          `committed dtype '${current.dtype}' disagrees with '${descriptor.dtype}'.`,
        );
      }
      if (!Number.isSafeInteger(descriptor.maximumBytes) ||
          descriptor.maximumBytes < descriptor.requiredBytes) {
        fail(
          'CAPACITY_LIMIT_EXCEEDED',
          `capacity '${descriptor.name}'`,
          'has no safe whole-domain maximum.',
        );
      }
      let targetBytes = current?.capacityBytes ?? descriptor.requiredBytes;
      if (targetBytes < descriptor.requiredBytes) {
        while (targetBytes < descriptor.requiredBytes) {
          const next = Math.ceil(targetBytes * this.limits.capacityGrowthFactor);
          if (!Number.isSafeInteger(next) || next <= targetBytes) {
            targetBytes = descriptor.requiredBytes;
            break;
          }
          targetBytes = Math.min(next, descriptor.maximumBytes);
        }
      }
      const dtypeBytes = runtimeDTypeBytes(descriptor.dtype);
      const remainder = targetBytes % dtypeBytes;
      if (remainder !== 0) targetBytes = checkedAdd(
        targetBytes,
        dtypeBytes - remainder,
        `capacity '${descriptor.name}' alignment`,
      );
      targetBytes = Math.min(targetBytes, descriptor.maximumBytes);
      if (targetBytes < descriptor.requiredBytes || targetBytes % dtypeBytes !== 0) {
        fail(
          'CAPACITY_LIMIT_EXCEEDED',
          `capacity '${descriptor.name}'`,
          'cannot satisfy the exact request within its whole-domain maximum.',
        );
      }
      targets.set(descriptor.name, targetBytes);
      totalBytes = checkedAdd(totalBytes, targetBytes, 'total activation capacity');
      if (current === undefined || targetBytes !== current.capacityBytes) grownTensorCount++;
    }
    if (totalBytes > this.limits.maxCapacityBytes) {
      fail(
        'CAPACITY_LIMIT_EXCEEDED',
        'capacity candidate',
        `${totalBytes} bytes exceed the context limit ${this.limits.maxCapacityBytes}.`,
      );
    }

    const slots = new Map<string, CapacitySlot>();
    const assertCandidateNonOverlap = (name: string, storage: RuntimeTypedArray): void => {
      for (const other of slots.values()) {
        if (overlaps(storage, other.storage)) {
          fail(
            'CAPACITY_STORAGE_MISMATCH',
            `capacity '${name}'`,
            `overlaps context storage for '${other.name}'.`,
          );
        }
      }
    };
    for (const descriptor of requirements) {
      const targetCapacityBytes = targets.get(descriptor.name)!;
      const current = this.#capacities.get(descriptor.name);
      if (current !== undefined && current.capacityBytes === targetCapacityBytes) {
        // A newly allocated earlier slot may alias a later reused slot. Check
        // both reused and new storage as it enters the complete candidate.
        assertCandidateNonOverlap(descriptor.name, current.storage);
        slots.set(descriptor.name, current);
        continue;
      }
      const request = Object.freeze({
        name: descriptor.name,
        dtype: descriptor.dtype,
        requiredBytes: descriptor.requiredBytes,
        targetCapacityBytes,
        previousCapacityBytes: current?.capacityBytes ?? 0,
      });
      let storage: unknown;
      try {
        storage = this.#allocator(request);
      } catch (error) {
        if (error instanceof CPUShapeExecutionContextError) throw error;
        fail(
          'CAPACITY_ALLOCATION_FAILED',
          `capacity '${descriptor.name}'`,
          `allocator failed for ${targetCapacityBytes} bytes.`,
          error,
        );
      }
      if (!runtimeStorageMatches(descriptor.dtype, storage) ||
          storage.byteLength !== targetCapacityBytes) {
        fail(
          'CAPACITY_STORAGE_MISMATCH',
          `capacity '${descriptor.name}'`,
          `allocator must return exact ${descriptor.dtype} storage with ` +
          `${targetCapacityBytes} bytes.`,
        );
      }
      assertCandidateNonOverlap(descriptor.name, storage);
      slots.set(descriptor.name, Object.freeze({
        name: descriptor.name,
        dtype: descriptor.dtype,
        storage,
        capacityBytes: targetCapacityBytes,
      }));
    }
    return {
      slots,
      currentBytes: totalBytes,
      grew: grownTensorCount !== 0,
      grownTensorCount,
    };
  }
}
