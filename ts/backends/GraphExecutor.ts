import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import { WebGPUDeviceState } from './WebGPUDeviceState.js';
import {
  WebGPUGraphCompiler,
  compileWebGPUGraphPlan,
  type WebGPUCompiledGraphPlan,
} from './WebGPUGraphCompiler.js';
import {
  createWebGPUBufferOrOOM,
  releaseWebGPUBufferLeases,
  WebGPUResources,
  type WebGPUBufferLease,
} from './WebGPUResources.js';
import { WebGPUDispatch } from './WebGPUDispatch.js';
import { WebGPUDecodeState } from './WebGPUDecodeState.js';
import {
  captureWebGPUErrorScopesSync,
  runWebGPUErrorScopedSync,
} from './WebGPUErrorScopes.js';
import type { KVPagePlan } from './kvPageAddressing.js';
import type { DecodeRowSet } from './decodeRowSet.js';
import { WebGPUResults, type WebGPUOutputSnapshot } from './WebGPUResults.js';
import type {
  AdapterExecutionPlan,
  AdapterTarget,
  CompiledWebGPUPipeline,
  DeviceFeedbackDecodeOptions,
  DeviceFeedbackDescriptor,
  DeviceFeedbackState,
  ExecutorGraph,
  ExecutorNode,
  ExecutorTensor,
  GraphExecutorOptions,
  IncrementalRowByteCopy,
  IncrementalRowCandidate,
  IncrementalRowPlan,
  WebGPURebindOptions,
  WebGPUResourceInspection,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
} from './WebGPUContracts.js';

interface WebGPUSpecializationBufferSlot {
  readonly buffer: GPUBuffer;
  readonly size: number;
  readonly usage: GPUBufferUsageFlags;
}

interface DeferredWebGPUWrite {
  readonly buffer: GPUBuffer;
  readonly bufferOffset: number;
  /** Byte-normalized view; GPUQueue typed-view offsets are element based. */
  readonly data: Uint8Array;
}

interface ActiveWebGPUSpecializationStage {
  scope: string;
  ordinal: number;
  readonly previousSlots: ReadonlyMap<string, WebGPUSpecializationBufferSlot>;
  readonly slots: Map<string, WebGPUSpecializationBufferSlot>;
  readonly previousBindGroups: ReadonlyMap<string, GPUBindGroup>;
  readonly bindGroups: Map<string, GPUBindGroup>;
  readonly previousContents: ReadonlyMap<GPUBuffer, Uint8Array>;
  readonly contentEligible: Set<GPUBuffer>;
  readonly contentInvariant: Set<GPUBuffer>;
  readonly resources: Set<GPUBuffer>;
  readonly created: Set<GPUBuffer>;
  readonly writes: DeferredWebGPUWrite[];
  readonly errorChecks: Promise<void>[];
}

interface PreparedWebGPUSpecializationWrites {
  readonly writes: readonly DeferredWebGPUWrite[];
  readonly skippedCount: number;
  readonly skippedBytes: number;
  readonly contents: Map<GPUBuffer, Uint8Array>;
}

function exactAlignedBytesEqual(
  previous: Uint8Array,
  previousOffset: number,
  candidate: Uint8Array,
): boolean {
  if (previousOffset > previous.byteLength ||
      candidate.byteLength > previous.byteLength - previousOffset) {
    return false;
  }
  // Specialization writes are four-byte aligned by contract. DataView keeps
  // this exact even when a caller's source view begins at an unaligned host
  // offset, while comparing one native word per iteration instead of one byte.
  const left = new DataView(
    previous.buffer,
    previous.byteOffset + previousOffset,
    candidate.byteLength,
  );
  const right = new DataView(
    candidate.buffer,
    candidate.byteOffset,
    candidate.byteLength,
  );
  for (let offset = 0; offset < candidate.byteLength; offset += 4) {
    if (left.getUint32(offset, true) !== right.getUint32(offset, true)) return false;
  }
  return true;
}

interface WebGPUForwardNodeRecipe {
  readonly nodeIndex: number;
  readonly nodeId: string | number;
  readonly opType: string;
}

type WebGPUPlannedRebindOptions = WebGPURebindOptions & {
  /** @internal Detached full-profile plan; resource bindings are always rebuilt. */
  readonly precompiledGraphPlan?: WebGPUCompiledGraphPlan;
  /** @internal Stable node traversal authored by the full-profile Trainer. */
  readonly forwardNodeRecipe?: readonly WebGPUForwardNodeRecipe[];
  /** @internal Full-profile forward owns replaceable training Dropout dispatches. */
  readonly includeDropoutNodes?: boolean;
};

const EMPTY_BANK_RESIDENCY: Readonly<Record<string, readonly number[]>> = Object.freeze({});

function normalizeBankResidency(
  residency: Readonly<Record<string, readonly number[]>> | undefined,
): Readonly<Record<string, readonly number[]>> {
  if (residency === undefined) return EMPTY_BANK_RESIDENCY;
  if (residency === null || typeof residency !== 'object' || Array.isArray(residency)) {
    throw new Error('WebGPU bankResidency must be an object of slot-id arrays.');
  }
  const normalized: Record<string, readonly number[]> = {};
  for (const name of Object.keys(residency).sort()) {
    const slots = residency[name];
    if (!name || !Array.isArray(slots) || slots.length === 0) {
      throw new Error(`WebGPU bankResidency '${name}' must be a non-empty slot-id array.`);
    }
    let previous = -1;
    for (let index = 0; index < slots.length; index++) {
      const slot = slots[index];
      if (!Number.isSafeInteger(slot) || slot < 0 || slot <= previous) {
        throw new Error(
          `WebGPU bankResidency '${name}' must contain strictly ascending non-negative slot ids.`,
        );
      }
      previous = slot;
    }
    normalized[name] = Object.freeze([...slots]);
  }
  return Object.freeze(normalized);
}

function changedBankWeightNames(
  previous: Readonly<Record<string, readonly number[]>>,
  candidate: Readonly<Record<string, readonly number[]>>,
): ReadonlySet<string> {
  const changed = new Set<string>();
  for (const name of new Set([...Object.keys(previous), ...Object.keys(candidate)])) {
    const left = previous[name];
    const right = candidate[name];
    if (left === undefined || right === undefined || left.length !== right.length ||
        left.some((slot, index) => slot !== right[index])) {
      changed.add(name);
    }
  }
  return changed;
}

export type {
  CompiledWebGPUPipeline,
  DeviceFeedbackDecodeOptions,
  GraphExecutorOptions,
  WebGPUAdapterSelector,
  WebGPUExecutionInputs,
  WebGPUExecutionOptions,
  WebGPURebindOptions,
  WebGPUResourceInspection,
} from './WebGPUContracts.js';

/**
 * RuntimeGraph-bound WebGPU orchestration facade. Operator compilation, allocation,
 * dispatch, decode mutation, and result snapshots are owned by dedicated
 * collaborators; this class only composes their lifecycle and preserves the
 * internal engine hooks used by training and backend integration.
 */
export class GraphExecutor extends BackendEngine {
  declare readonly allocateGraph: never;

  declare device: GPUDevice;
  declare graph: ExecutorGraph;
  declare hasPackedDot4: boolean;
  declare deviceState: WebGPUDeviceState;
  declare _ownsDeviceState: boolean;
  declare pipelines: CompiledWebGPUPipeline[];
  declare computePipelineCache: Map<string, Map<string, Promise<GPUComputePipeline>>>;
  declare rejectedSpecializedShaders: Set<string>;
  declare gpuBuffers: Map<string, GPUBuffer>;
  declare tensorBufferUsages: Map<string, GPUBufferUsageFlags>;
  declare tensorCapacityBytes: Map<string, number>;
  declare adapterTargetBuffers: Map<AdapterTarget, { a: GPUBuffer; b: GPUBuffer }>;
  declare auxiliaryBuffers: Set<GPUBuffer>;
  declare incrementalRowPlans: Map<number, IncrementalRowPlan>;
  declare incrementalRowCandidates: Map<number, IncrementalRowCandidate>;
  declare incrementalRowCopyTensorNames: Set<string>;
  /** Graph inputs a batched keep mask reads host-side; see WebGPUDecodeState. */
  declare incrementalRowMirrorNames: Set<string>;
  declare decodeState: WebGPUDecodeState;
  declare adapterPipeline: GPUComputePipeline | null;
  declare compiledWeightRevision: number | undefined;
  declare compiledTopologyRevision: number | undefined;
  declare _activePipelineTarget: CompiledWebGPUPipeline[] | undefined;
  declare _activePipelineBuffers: Map<string, GPUBuffer> | undefined;
  declare _activeSpecializationResources: Set<GPUBuffer> | undefined;
  declare _activeSpecializationStage: ActiveWebGPUSpecializationStage | undefined;
  declare specializationBufferSlots: Map<string, WebGPUSpecializationBufferSlot>;
  declare specializationBindGroups: Map<string, GPUBindGroup>;
  declare specializationBufferContents: Map<GPUBuffer, Uint8Array>;
  declare pendingRetiredBuffers: Set<GPUBuffer>;
  declare pendingRetiredTensorLeases: Set<WebGPUBufferLease>;
  declare pendingRetiredPrivateBuffers: Set<GPUBuffer>;
  declare pendingRetirementFence: Promise<void> | null;
  declare retirementFenceUnavailable: boolean;
  declare _specializationObjectIds: WeakMap<object, number>;
  declare _nextSpecializationObjectId: number;
  declare compiledGraphPlan: WebGPUCompiledGraphPlan | null;
  declare currentShapeSignature: string | null;
  declare currentBankResidency: Readonly<Record<string, readonly number[]>>;
  declare logicalActivationBytes: number;
  declare activationCapacityBytes: number;
  declare activationCapacityHighWaterBytes: number;
  declare activationGrowCount: number;
  declare specializationRebindCount: number;
  declare specializationBufferCreateCount: number;
  declare specializationBufferReuseCount: number;
  declare bindGroupCreateCount: number;
  declare bindGroupReuseCount: number;
  declare specializationWriteCount: number;
  declare specializationWriteBytes: number;
  declare specializationWriteSkipCount: number;
  declare specializationWriteSkipBytes: number;
  declare pendingDeviceErrorChecks: Set<Promise<void>>;

  readonly graphCompiler: WebGPUGraphCompiler;
  readonly resources: WebGPUResources;
  readonly dispatch: WebGPUDispatch;
  readonly results: WebGPUResults;

  constructor(device: GPUDevice, graph: RuntimeGraph, {
    shaderLibrary = null,
    wgslLanguageFeatures = globalThis.navigator?.gpu?.wgslLanguageFeatures,
    deviceState = null,
    invariantWeightBorrow = null,
  }: GraphExecutorOptions = {}) {
    super('webgpu', {
      incrementalExecution: true,
      incrementalRows: true,
      outputLocation: 'device',
    });
    this.device = device;
    if (deviceState && deviceState.device !== device) {
      throw new Error('GraphExecutor deviceState belongs to a different GPUDevice.');
    }
    this.deviceState = deviceState || new WebGPUDeviceState(device);
    this._ownsDeviceState = deviceState == null;
    this.graph = graph as ExecutorGraph;
    this.hasPackedDot4 = wgslLanguageFeatures?.has?.('packed_4x8_integer_dot_product') === true;
    this._incrementalCacheValid = false;
    this.pipelines = [];
    this.computePipelineCache = this.deviceState.computePipelineCache;
    this.rejectedSpecializedShaders = this.deviceState.rejectedSpecializedShaders;
    this.gpuBuffers = new Map();
    this.tensorBufferUsages = new Map();
    this.tensorCapacityBytes = new Map();
    this.adapterTargetBuffers = new Map();
    this.auxiliaryBuffers = new Set();
    this.specializationBufferSlots = new Map();
    this.specializationBindGroups = new Map();
    this.specializationBufferContents = new Map();
    this.pendingRetiredBuffers = new Set();
    this.pendingRetiredTensorLeases = new Set();
    this.pendingRetiredPrivateBuffers = new Set();
    this.pendingRetirementFence = null;
    this.retirementFenceUnavailable = false;
    this._specializationObjectIds = new WeakMap();
    this._nextSpecializationObjectId = 1;
    this.adapterPipeline = null;
    this.compiledGraphPlan = null;
    this.currentShapeSignature = null;
    this.currentBankResidency = EMPTY_BANK_RESIDENCY;
    this.logicalActivationBytes = 0;
    this.activationCapacityBytes = 0;
    this.activationCapacityHighWaterBytes = 0;
    this.activationGrowCount = 0;
    this.specializationRebindCount = 0;
    this.specializationBufferCreateCount = 0;
    this.specializationBufferReuseCount = 0;
    this.bindGroupCreateCount = 0;
    this.bindGroupReuseCount = 0;
    this.specializationWriteCount = 0;
    this.specializationWriteBytes = 0;
    this.specializationWriteSkipCount = 0;
    this.specializationWriteSkipBytes = 0;
    this.pendingDeviceErrorChecks = new Set();

    this.graphCompiler = new WebGPUGraphCompiler(this);
    this.resources = new WebGPUResources(this, invariantWeightBorrow);
    this.dispatch = new WebGPUDispatch(this);
    this.decodeState = new WebGPUDecodeState(this);
    this.results = new WebGPUResults(this);
    this.incrementalRowPlans = this.decodeState.rowPlans;
    this.incrementalRowCandidates = this.decodeState.rowCandidates;
    this.incrementalRowCopyTensorNames = this.decodeState.rowCopyTensorNames;
    this.incrementalRowMirrorNames = this.decodeState.rowMirrorNames;
    this.graphCompiler.installShaderLibrary(shaderLibrary);
    console.log('[VolvoxAI WebGPU] Starting RuntimeGraph Compilation...');
  }

  resetDecodeCache(): void {
    super.resetDecodeCache();
    this.decodeState.resetExecution();
  }

  /** @internal Incompatible provider seeds discard row/replay specialization. */
  resetDecodeBindingState(): void {
    this.decodeState.resetBinding();
  }

  get _webGPUIncrementalCacheValid(): boolean {
    return this._incrementalCacheValid;
  }

  set _webGPUIncrementalCacheValid(value: boolean) {
    this._incrementalCacheValid = value;
  }

  _beginWebGPUDecodeExecution(options: BackendExecutionOptions = {}): boolean {
    return this._beginDecodeExecution(options);
  }

  get deviceFeedbackControlBuffer(): GPUBuffer | null {
    return this.decodeState.controlBuffer;
  }

  set deviceFeedbackControlBuffer(value: GPUBuffer | null) {
    this.decodeState.controlBuffer = value;
  }

  get deviceFeedbackSequenceLength(): number {
    return this.decodeState.sequenceLength;
  }

  set deviceFeedbackSequenceLength(value: number) {
    this.decodeState.sequenceLength = value;
  }

  get deviceFeedbackState(): Readonly<DeviceFeedbackState> | null {
    return this.decodeState.feedback;
  }

  set deviceFeedbackState(value: Readonly<DeviceFeedbackState> | null) {
    this.decodeState.feedback = value;
  }

  _buffer(tensor: ExecutorTensor | null | undefined): GPUBuffer {
    return (tensor
      ? (this._activePipelineBuffers || this.gpuBuffers).get(tensor.name)
      : undefined) as GPUBuffer;
  }

  /** Reuse one context-owned uniform/scratch slot when its identity contract is stable. */
  _createSpecializationBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const size = Number(descriptor.size);
    const usage = descriptor.usage;
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new Error('WebGPU specialization buffers require a positive safe byte size.');
    }
    const stage = this._activeSpecializationStage;
    if (stage) {
      const key = `${stage.scope}:${stage.ordinal++}:${descriptor.label || ''}`;
      const previous = stage.previousSlots.get(key);
      if (previous && previous.size === size && previous.usage === usage) {
        stage.slots.set(key, previous);
        stage.resources.add(previous.buffer);
        if ((usage & GPUBufferUsage.UNIFORM) !== 0) {
          stage.contentEligible.add(previous.buffer);
        }
        this.specializationBufferReuseCount++;
        return previous.buffer;
      }
      const allocation = `WebGPU specialization '${descriptor.label || stage.scope}'`;
      const captured = captureWebGPUErrorScopesSync(this.device, { label: allocation }, () =>
        createWebGPUBufferOrOOM(this.device, descriptor, allocation));
      stage.errorChecks.push(captured.check);
      // Avoid an unhandled rejection if pipeline compilation itself fails
      // before the transactional rebind reaches its explicit check barrier.
      void captured.check.catch(() => {});
      const buffer = captured.value;
      const slot = Object.freeze({ buffer, size, usage });
      stage.slots.set(key, slot);
      stage.resources.add(buffer);
      stage.created.add(buffer);
      if ((usage & GPUBufferUsage.UNIFORM) !== 0) stage.contentEligible.add(buffer);
      this.specializationBufferCreateCount++;
      return buffer;
    }
    const allocation = `WebGPU specialization '${descriptor.label || 'buffer'}'`;
    const captured = captureWebGPUErrorScopesSync(this.device, { label: allocation }, () =>
      createWebGPUBufferOrOOM(this.device, descriptor, allocation));
    this._trackPendingDeviceErrorCheck(captured.check);
    const buffer = captured.value;
    (this._activeSpecializationResources || this.auxiliaryBuffers).add(buffer);
    this.specializationBufferCreateCount++;
    return buffer;
  }

  /** Mark a model-revision-invariant shader input for exact write elision. */
  _markSpecializationBufferInvariant(buffer: GPUBuffer): void {
    const stage = this._activeSpecializationStage;
    if (stage) {
      if (!stage.resources.has(buffer)) {
        throw new Error('WebGPU cannot mark an unstaged specialization buffer read-only.');
      }
      stage.contentEligible.add(buffer);
      stage.contentInvariant.add(buffer);
      return;
    }
  }

  _setSpecializationScope(scope: string): void {
    const stage = this._activeSpecializationStage;
    if (!stage) return;
    stage.scope = scope;
    stage.ordinal = 0;
  }

  _writeSpecializationBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: AllowSharedBufferSource,
  ): void {
    const bytes = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    const stage = this._activeSpecializationStage;
    if (stage) {
      stage.writes.push({ buffer, bufferOffset, data: bytes });
      return;
    }
    // Out-of-stage writes are not part of a transactional rebind. If an
    // internal caller targets a previously mirrored buffer, force its next
    // specialization to establish exact contents again.
    this.specializationBufferContents.delete(buffer);
    const captured = captureWebGPUErrorScopesSync(
      this.device,
      { label: 'WebGPU specialization buffer write' },
      () => this.device.queue.writeBuffer(buffer, bufferOffset, bytes),
    );
    this._trackPendingDeviceErrorCheck(captured.check);
  }

  private _specializationObjectId(value: object): number {
    let identity = this._specializationObjectIds.get(value);
    if (identity === undefined) {
      identity = this._nextSpecializationObjectId++;
      this._specializationObjectIds.set(value, identity);
    }
    return identity;
  }

  _createSpecializationBindGroup(
    pipeline: GPUComputePipeline,
    descriptor: GPUBindGroupDescriptor,
  ): GPUBindGroup {
    const stage = this._activeSpecializationStage;
    if (!stage) {
      this.bindGroupCreateCount++;
      const captured = captureWebGPUErrorScopesSync(
        this.device,
        { label: 'WebGPU specialization bind group' },
        () => this.device.createBindGroup(descriptor),
      );
      this._trackPendingDeviceErrorCheck(captured.check);
      return captured.value;
    }
    const entries = [...descriptor.entries]
      .sort((left, right) => left.binding - right.binding)
      .map((entry) => {
        const resource = entry.resource;
        if ('buffer' in resource) {
          return [
            entry.binding,
            'buffer',
            this._specializationObjectId(resource.buffer),
            Number(resource.offset || 0),
            resource.size === undefined ? null : Number(resource.size),
          ];
        }
        return [entry.binding, 'object', this._specializationObjectId(resource as object)];
      });
    const key = JSON.stringify([this._specializationObjectId(pipeline), entries]);
    const previous = stage.previousBindGroups.get(key);
    if (previous) {
      stage.bindGroups.set(key, previous);
      this.bindGroupReuseCount++;
      return previous;
    }
    const captured = captureWebGPUErrorScopesSync(
      this.device,
      { label: `WebGPU specialization '${stage.scope}' bind group` },
      () => this.device.createBindGroup(descriptor),
    );
    stage.errorChecks.push(captured.check);
    void captured.check.catch(() => {});
    const bindGroup = captured.value;
    stage.bindGroups.set(key, bindGroup);
    this.bindGroupCreateCount++;
    return bindGroup;
  }

  private _trackPendingDeviceErrorCheck(check: Promise<void>): void {
    this.pendingDeviceErrorChecks.add(check);
    // A lazy decode specialization can finish its error check before control
    // returns to the executor's drain barrier. Mark it handled without losing
    // the original rejecting promise that the barrier will inspect.
    void check.catch(() => {});
  }

  private async _drainPendingDeviceErrorChecks(): Promise<void> {
    const pending = [...this.pendingDeviceErrorChecks];
    this.pendingDeviceErrorChecks.clear();
    if (pending.length !== 0) await Promise.all(pending);
  }

  private _prepareSpecializationWrites(
    stage: ActiveWebGPUSpecializationStage,
  ): PreparedWebGPUSpecializationWrites {
    const contents = new Map<GPUBuffer, Uint8Array>();
    for (const buffer of stage.contentEligible) {
      const previous = stage.previousContents.get(buffer);
      if (previous) contents.set(buffer, previous);
    }
    const writes: DeferredWebGPUWrite[] = [];
    let skippedCount = 0;
    let skippedBytes = 0;
    for (const [index, write] of stage.writes.entries()) {
      const size = write.data.byteLength;
      if (!stage.resources.has(write.buffer)) {
        throw new Error(`WebGPU specialization write ${index} targets an unstaged buffer.`);
      }
      if (!Number.isSafeInteger(write.bufferOffset) || write.bufferOffset < 0 ||
          write.bufferOffset % 4 !== 0 || !Number.isSafeInteger(size) ||
          size <= 0 || size % 4 !== 0 ||
          !Number.isSafeInteger(write.buffer.size) ||
          write.bufferOffset > write.buffer.size || size > write.buffer.size - write.bufferOffset) {
        throw new Error(
          `WebGPU specialization write ${index} has an invalid aligned source or destination range.`,
        );
      }
      const eligible = stage.contentEligible.has(write.buffer);
      let previous = eligible ? contents.get(write.buffer) : undefined;
      // A zero mirror synthesized for a newly allocated buffer belongs only
      // to this candidate. It must not gain the stronger model-invariant
      // shortcut if the same buffer receives more than one staged write.
      const hadCommittedMirror = stage.previousContents.has(write.buffer);
      if (eligible && !previous && stage.created.has(write.buffer)) {
        // WebGPU buffers are zero-initialized. Materialize that exact state
        // only for newly created buffers that receive a write, so partial
        // uniform writes can participate in later collision-free comparisons.
        previous = new Uint8Array(write.buffer.size);
        contents.set(write.buffer, previous);
      }
      const equal = eligible && previous !== undefined &&
        ((hadCommittedMirror && stage.contentInvariant.has(write.buffer)) ||
          exactAlignedBytesEqual(previous, write.bufferOffset, write.data));
      if (eligible && equal) {
        skippedCount++;
        skippedBytes += size;
        continue;
      }
      writes.push(write);
      if (!eligible) {
        contents.delete(write.buffer);
      } else if (write.bufferOffset === 0 && size === write.buffer.size) {
        contents.set(write.buffer, write.data.slice());
      } else if (previous?.byteLength === write.buffer.size) {
        const updated = stage.created.has(write.buffer) ? previous : previous.slice();
        updated.set(write.data, write.bufferOffset);
        contents.set(write.buffer, updated);
      } else {
        // Only exact known contents may suppress a later write. A partial
        // update to a buffer without a complete mirror stays intentionally
        // uncacheable.
        contents.delete(write.buffer);
      }
    }
    return Object.freeze({
      writes: Object.freeze(writes),
      skippedCount,
      skippedBytes,
      contents,
    });
  }

  private _commitSpecializationWrites(
    writes: readonly DeferredWebGPUWrite[],
  ): void {
    for (const write of writes) {
      this.device.queue.writeBuffer(
        write.buffer,
        write.bufferOffset,
        write.data,
      );
    }
  }

  private _retireAfterSubmittedWork(
    tensorLeases: Iterable<WebGPUBufferLease>,
    privateBuffers: Iterable<GPUBuffer>,
  ): void {
    const leases = [...new Set(tensorLeases)];
    const buffers = [...new Set(privateBuffers)];
    if (leases.length === 0 && buffers.length === 0) return;
    for (const lease of leases) {
      this.pendingRetiredTensorLeases.add(lease);
      this.pendingRetiredBuffers.add(lease.buffer);
    }
    for (const buffer of buffers) {
      this.pendingRetiredPrivateBuffers.add(buffer);
      this.pendingRetiredBuffers.add(buffer);
    }
    const release = () => {
      releaseWebGPUBufferLeases(leases);
      for (const lease of leases) this.pendingRetiredTensorLeases.delete(lease);
      for (const buffer of buffers) {
        this.pendingRetiredPrivateBuffers.delete(buffer);
        buffer.destroy?.();
      }
      for (const lease of leases) this.pendingRetiredBuffers.delete(lease.buffer);
      for (const buffer of buffers) this.pendingRetiredBuffers.delete(buffer);
    };
    let settled: Promise<void> | undefined;
    try {
      settled = this.device.queue.onSubmittedWorkDone?.();
    } catch {
      // Keep the buffers context-owned until dispose if the queue cannot issue
      // a completion fence (for example, while device loss is being reported),
      // and fail closed before another generation can allocate.
      this.retirementFenceUnavailable = true;
      return;
    }
    if (!settled || typeof settled.then !== 'function') {
      // A conforming WebGPU queue always supplies a completion promise. Treat
      // a missing/non-promise fence like a failed fence: destroying submitted
      // buffers here would make the prior generation observably unsafe.
      this.retirementFenceUnavailable = true;
      return;
    }
    let retirement: Promise<void>;
    retirement = settled.then(release, release).then(() => {
      if (this.pendingRetirementFence === retirement) {
        this.pendingRetirementFence = null;
      }
    });
    this.pendingRetirementFence = retirement;
  }

  private _releasePendingRetirement(): void {
    /* A live fence owns these references until it settles. `dispose()` is
     * synchronous, so leave them with its closure rather than reclaiming work
     * the device may still be reading. A missing fence deliberately falls
     * through: that path retained resources until explicit disposal. */
    if (this.pendingRetirementFence) return;
    releaseWebGPUBufferLeases(this.pendingRetiredTensorLeases);
    for (const buffer of this.pendingRetiredPrivateBuffers) buffer.destroy?.();
    this.pendingRetiredTensorLeases.clear();
    this.pendingRetiredPrivateBuffers.clear();
    this.pendingRetiredBuffers.clear();
  }

  private async _awaitPendingRetirement(): Promise<void> {
    if (this.retirementFenceUnavailable) {
      throw new Error(
        'WebGPU cannot safely specialize another shape without a queue retirement fence.',
      );
    }
    const pending = this.pendingRetirementFence;
    if (pending) await pending;
  }

  private _assertPipelineWorkgroupLimits(
    pipelines: readonly CompiledWebGPUPipeline[],
  ): void {
    const candidate = (this.device.limits as GPUSupportedLimits | undefined)
      ?.maxComputeWorkgroupsPerDimension;
    const limit = Number.isSafeInteger(candidate) && (candidate as number) > 0
      ? candidate as number
      : 0xffffffff;
    for (const [pipelineIndex, pipeline] of pipelines.entries()) {
      if (!Array.isArray(pipeline.workgroupCount) || pipeline.workgroupCount.length !== 3) {
        throw new Error(`WebGPU pipeline ${pipelineIndex} has no three-dimensional dispatch.`);
      }
      for (let axis = 0; axis < 3; axis++) {
        const count = pipeline.workgroupCount[axis];
        if (!Number.isSafeInteger(count) || count <= 0 || count > 0xffffffff || count > limit) {
          throw new Error(
            `WebGPU pipeline ${String(pipeline.nodeName ?? pipelineIndex)} workgroup axis ${axis} ` +
            `count ${String(count)} exceeds the device limit ${limit}.`,
          );
        }
      }
    }
  }

  /**
   * Transactionally publish a concrete shape generation. Shader modules and
   * compute pipelines remain device-cached; only concrete tensor capacities,
   * uniforms, bind groups, and workgroup counts are rebuilt. Failed staging
   * leaves the previous graph generation executable.
   */
  async rebindGraph(graph: RuntimeGraph, {
    shapeSignature,
    bankResidency,
    beforeCommit,
    tensorMaximumBytes,
    capacityGrowthFactor = 2,
    precompiledGraphPlan,
    forwardNodeRecipe,
    includeDropoutNodes = false,
  }: WebGPUPlannedRebindOptions): Promise<void> {
    if (!graph || !(graph.tensors instanceof Map) || !Array.isArray(graph.nodes)) {
      throw new Error('GraphExecutor.rebindGraph requires a concrete RuntimeGraph.');
    }
    if (typeof shapeSignature !== 'string' || shapeSignature.length === 0) {
      throw new Error('GraphExecutor.rebindGraph requires a canonical shape signature.');
    }
    // At most one submitted tensor/specialization generation may await
    // retirement. Drain it before staging another candidate so the bounded
    // resident proof cannot be defeated by rapid shape alternation.
    await this._awaitPendingRetirement();
    this._assertPortableQuantizedGraph(graph);
    await this.graphCompiler.ensureShaderLibrary();
    const candidatePlan = precompiledGraphPlan ?? compileWebGPUGraphPlan(graph);
    const candidateBankResidency = normalizeBankResidency(bankResidency);
    for (const name of Object.keys(candidateBankResidency)) {
      if (!graph.tensors.get(name)?.isWeight) {
        throw new Error(`WebGPU bankResidency '${name}' does not name a graph weight.`);
      }
    }
    const previousGraph = this.graph;
    const previousPlan = this.compiledGraphPlan;
    const previousPipelines = this.pipelines;
    const previousBuffers = this.gpuBuffers;
    const previousUsages = this.tensorBufferUsages;
    const previousCapacities = this.tensorCapacityBytes;
    const previousAuxiliary = this.auxiliaryBuffers;
    const previousSpecializationSlots = this.specializationBufferSlots;
    const previousBindGroups = this.specializationBindGroups;
    const previousSpecializationContents = this.specializationBufferContents;
    const previousBufferCreateCount = this.specializationBufferCreateCount;
    const previousBufferReuseCount = this.specializationBufferReuseCount;
    const previousBindGroupCreateCount = this.bindGroupCreateCount;
    const previousBindGroupReuseCount = this.bindGroupReuseCount;
    const previousSpecializationWriteCount = this.specializationWriteCount;
    const previousSpecializationWriteBytes = this.specializationWriteBytes;
    const previousSpecializationWriteSkipCount = this.specializationWriteSkipCount;
    const previousSpecializationWriteSkipBytes = this.specializationWriteSkipBytes;
    const previousCompiledWeightRevision = this.compiledWeightRevision;
    const previousCompiledTopologyRevision = this.compiledTopologyRevision;
    const previousShapeSignature = this.currentShapeSignature;
    const previousBankResidency = this.currentBankResidency;
    const previousLogicalActivationBytes = this.logicalActivationBytes;
    const previousActivationCapacityBytes = this.activationCapacityBytes;
    const previousActivationCapacityHighWaterBytes = this.activationCapacityHighWaterBytes;
    const previousActivationGrowCount = this.activationGrowCount;
    const previousSpecializationRebindCount = this.specializationRebindCount;
    const candidateAuxiliary = new Set<GPUBuffer>();
    const candidateSpecializationCreated = new Set<GPUBuffer>();
    const specializationStage: ActiveWebGPUSpecializationStage = {
      scope: 'graph',
      ordinal: 0,
      previousSlots: previousSpecializationSlots,
      slots: new Map(),
      previousBindGroups,
      bindGroups: new Map(),
      previousContents: previousSpecializationContents,
      contentEligible: new Set(),
      contentInvariant: new Set(),
      resources: candidateAuxiliary,
      created: candidateSpecializationCreated,
      writes: [],
      errorChecks: [],
    };
    let staged: ReturnType<WebGPUResources['stageTensorBuffers']> | null = null;
    let incrementalRows: ReturnType<WebGPUDecodeState['_collectIncrementalRows']> | null = null;
    let commitStarted = false;
    let tensorResourcesCommitted = false;
    try {
      // Descriptor construction consults these fields while candidate buffers
      // and uniforms are staged. They are restored on every failure path.
      this.graph = graph as ExecutorGraph;
      this.compiledGraphPlan = candidatePlan;
      // Row analysis is pure and must precede tensor staging: every full tensor
      // copied to/from a row scratch buffer needs COPY_SRC|COPY_DST on the
      // candidate generation. Publish the analysis only after all allocation
      // and pipeline preparation succeeds, preserving transactional rebinds.
      incrementalRows = this.decodeState._collectIncrementalRows();
      const capturedTensorGeneration = captureWebGPUErrorScopesSync(
        this.device,
        { label: `WebGPU tensor generation '${shapeSignature}'` },
        () => this.resources.stageTensorBuffers(
          graph,
          tensorMaximumBytes,
          capacityGrowthFactor,
          !includeDropoutNodes,
          candidatePlan.resultCopyTensorNames,
          incrementalRows!.copyTensorNames,
          changedBankWeightNames(previousBankResidency, candidateBankResidency),
        ),
      );
      staged = capturedTensorGeneration.value;
      await capturedTensorGeneration.check;
      this._activeSpecializationStage = specializationStage;
      const candidatePipelines: CompiledWebGPUPipeline[] = [];
      const nodeRecipe = forwardNodeRecipe ?? graph.nodes.map((node, nodeIndex) => ({
        nodeIndex,
        nodeId: node.id,
        opType: node.opType,
      }));
      if (nodeRecipe.length !== graph.nodes.length) {
        throw new Error('WebGPU forward plan node count does not match its concrete graph.');
      }
      for (const descriptor of nodeRecipe) {
        const nodeIndex = descriptor.nodeIndex;
        const node = graph.nodes[nodeIndex] as ExecutorNode;
        if (!node || node.id !== descriptor.nodeId || node.opType !== descriptor.opType) {
          throw new Error(`WebGPU forward plan node ${nodeIndex} changed identity or operator.`);
        }
        if (node.opType === 'Dropout' && !includeDropoutNodes) continue;
        const start = candidatePipelines.length;
        await this._buildNodePipeline(node, {
          pipelines: candidatePipelines,
          buffers: staged.buffers,
          resources: candidateAuxiliary,
        });
        for (let index = start; index < candidatePipelines.length; index++) {
          candidatePipelines[index].graphNodeIndex = nodeIndex;
        }
      }
      // createBuffer/createBindGroup return objects before WebGPU resolves
      // their error scopes. Candidate resources remain rollback-owned until
      // every already-popped scope confirms that those objects are valid.
      await Promise.all(specializationStage.errorChecks);
      this._assertPipelineWorkgroupLimits(candidatePipelines);
      this._activeSpecializationStage = undefined;
      const specializationWrites = this._prepareSpecializationWrites(specializationStage);
      beforeCommit?.();
      commitStarted = true;
      await runWebGPUErrorScopedSync(
        this.device,
        { label: `WebGPU specialization writes for '${shapeSignature}'` },
        () => this._commitSpecializationWrites(specializationWrites.writes),
      );

      this.pipelines = candidatePipelines;
      this.gpuBuffers = staged.buffers;
      this.tensorBufferUsages = staged.usages;
      this.tensorCapacityBytes = staged.capacities;
      this.auxiliaryBuffers = candidateAuxiliary;
      this.specializationBufferSlots = specializationStage.slots;
      this.specializationBindGroups = specializationStage.bindGroups;
      this.specializationBufferContents = specializationWrites.contents;
      this.specializationWriteCount += specializationWrites.writes.length;
      this.specializationWriteBytes += specializationWrites.writes.reduce(
        (bytes, write) => bytes + write.data.byteLength,
        0,
      );
      this.specializationWriteSkipCount += specializationWrites.skippedCount;
      this.specializationWriteSkipBytes += specializationWrites.skippedBytes;
      this.compiledWeightRevision = graph.weightRevision || 0;
      this.compiledTopologyRevision = graph.topologyRevision || 0;
      this.currentShapeSignature = shapeSignature;
      this.currentBankResidency = candidateBankResidency;
      this.logicalActivationBytes = staged.logicalActivationBytes;
      this.activationCapacityBytes = staged.activationCapacityBytes;
      this.activationCapacityHighWaterBytes = Math.max(
        this.activationCapacityHighWaterBytes,
        staged.activationCapacityBytes,
      );
      if (staged.grew) this.activationGrowCount++;
      this.specializationRebindCount++;
      // No fallible graph-composition hook remains: publish the candidate row
      // contract only now, so a rejected rebind preserves the prior maps.
      this.resetDecodeCache();
      this.decodeState._publishIncrementalRows(incrementalRows);

      const retiredTensorLeases = this.resources.commitTensorBuffers(staged);
      tensorResourcesCommitted = true;
      const retiredAuxiliary = [...previousAuxiliary]
        .filter((buffer) => !this.auxiliaryBuffers.has(buffer));
      this._retireAfterSubmittedWork(retiredTensorLeases, retiredAuxiliary);
    } catch (error) {
      this._activeSpecializationStage = undefined;
      if (staged && !tensorResourcesCommitted) this.resources.rollbackTensorBuffers(staged);
      for (const buffer of candidateSpecializationCreated) buffer.destroy?.();
      this.graph = previousGraph;
      this.compiledGraphPlan = previousPlan;
      this.pipelines = previousPipelines;
      this.gpuBuffers = previousBuffers;
      this.tensorBufferUsages = previousUsages;
      this.tensorCapacityBytes = previousCapacities;
      this.auxiliaryBuffers = previousAuxiliary;
      this.specializationBufferSlots = previousSpecializationSlots;
      this.specializationBindGroups = previousBindGroups;
      // A write failure may have changed an unknown prefix of reused device
      // buffers. Keep the old exact mirrors only when no queue write started;
      // otherwise the forced recovery rebind must rewrite every candidate.
      this.specializationBufferContents = commitStarted
        ? new Map()
        : previousSpecializationContents;
      this.specializationBufferCreateCount = previousBufferCreateCount;
      this.specializationBufferReuseCount = previousBufferReuseCount;
      this.bindGroupCreateCount = previousBindGroupCreateCount;
      this.bindGroupReuseCount = previousBindGroupReuseCount;
      this.specializationWriteCount = previousSpecializationWriteCount;
      this.specializationWriteBytes = previousSpecializationWriteBytes;
      this.specializationWriteSkipCount = previousSpecializationWriteSkipCount;
      this.specializationWriteSkipBytes = previousSpecializationWriteSkipBytes;
      this.compiledWeightRevision = previousCompiledWeightRevision;
      this.compiledTopologyRevision = previousCompiledTopologyRevision;
      // Queue writes may target specialization buffers reused by the prior
      // generation. A synchronous queue failure can therefore leave their
      // contents indeterminate even though graph objects roll back. Mark that
      // binding unexecutable so the provider must fully rewrite it before the
      // next dispatch.
      this.currentShapeSignature = commitStarted ? null : previousShapeSignature;
      this.currentBankResidency = previousBankResidency;
      this.logicalActivationBytes = previousLogicalActivationBytes;
      this.activationCapacityBytes = previousActivationCapacityBytes;
      this.activationCapacityHighWaterBytes = previousActivationCapacityHighWaterBytes;
      this.activationGrowCount = previousActivationGrowCount;
      this.specializationRebindCount = previousSpecializationRebindCount;
      throw error;
    }
  }

  inspectDynamicResources(): Readonly<WebGPUResourceInspection> {
    const tensors = [...this.graph.tensors.values()]
      .filter((tensor) => !tensor.isWeight)
      .sort((left, right) => left.name.localeCompare(right.name))
      .map((tensor) => Object.freeze({
        name: tensor.name,
        logicalBytes: tensor.sizeBytes,
        capacityBytes: this.tensorCapacityBytes.get(tensor.name) || 0,
      }));
    return Object.freeze({
      shapeSignature: this.currentShapeSignature,
      logicalActivationBytes: this.logicalActivationBytes,
      activationCapacityBytes: this.activationCapacityBytes,
      activationCapacityHighWaterBytes: this.activationCapacityHighWaterBytes,
      activationGrowCount: this.activationGrowCount,
      specializationRebindCount: this.specializationRebindCount,
      specializationBufferCreateCount: this.specializationBufferCreateCount,
      specializationBufferReuseCount: this.specializationBufferReuseCount,
      liveSpecializationBufferCount: this.specializationBufferSlots.size,
      bindGroupCreateCount: this.bindGroupCreateCount,
      bindGroupReuseCount: this.bindGroupReuseCount,
      liveBindGroupCount: this.specializationBindGroups.size,
      specializationWriteCount: this.specializationWriteCount,
      specializationWriteBytes: this.specializationWriteBytes,
      specializationWriteSkipCount: this.specializationWriteSkipCount,
      specializationWriteSkipBytes: this.specializationWriteSkipBytes,
      specializationContentBytes: [...this.specializationBufferContents.values()]
        .reduce((bytes, contents) => bytes + contents.byteLength, 0),
      pendingRetiredBufferCount: this.pendingRetiredBuffers.size,
      selectedTactics: Object.freeze(this.pipelines.map((pipeline) =>
        pipeline.tacticId || `webgpu.${String(pipeline.nodeName ?? 'node')}.generic`)),
      tensorCapacities: Object.freeze(tensors),
    });
  }

  async compile(): Promise<void> {
    this._assertPortableQuantizedGraph(this.graph);
    this.resetDecodeCache();
    await this.graphCompiler.ensureShaderLibrary();
    await this._awaitPendingRetirement();
    this.resources.resetCompilationResources();
    this.currentBankResidency = EMPTY_BANK_RESIDENCY;
    this.pipelines = [];
    this.specializationBufferSlots.clear();
    this.specializationBindGroups.clear();
    this.specializationBufferContents.clear();
    this.pendingDeviceErrorChecks.clear();
    this._releasePendingRetirement();
    this.pendingRetirementFence = null;
    this.retirementFenceUnavailable = false;
    this.decodeState.resetCompilation();
    this.compiledGraphPlan = compileWebGPUGraphPlan(this.graph as RuntimeGraph);
    this._analyzeIncrementalRows();
    await runWebGPUErrorScopedSync(
      this.device,
      { label: 'WebGPU graph tensor allocation', phase: 'compilation' },
      () => this._allocateBuffers(),
    );
    this._aliasInferenceDropoutBuffers();
    for (let nodeIndex = 0; nodeIndex < this.graph.nodes.length; nodeIndex++) {
      const node = this.graph.nodes[nodeIndex];
      if (node.opType === 'Dropout') continue;
      const start = this.pipelines.length;
      await this._buildNodePipeline(node);
      for (let i = start; i < this.pipelines.length; i++) {
        this.pipelines[i].graphNodeIndex = nodeIndex;
      }
    }
    await this._drainPendingDeviceErrorChecks();
    this._assertPipelineWorkgroupLimits(this.pipelines);
    this.compiledWeightRevision = this.graph.weightRevision || 0;
    this.compiledTopologyRevision = this.graph.topologyRevision || 0;
    console.log(`[VolvoxAI WebGPU] Compilation complete. Allocated ${this.gpuBuffers.size} VRAM buffers.`);
  }

  dispose(): void {
    this.resetDecodeCache();
    this.resources.resetCompilationResources();
    this._releasePendingRetirement();
    this.decodeState.resetCompilation();
    this.pipelines = [];
    this.specializationBufferSlots.clear();
    this.specializationBindGroups.clear();
    this.specializationBufferContents.clear();
    if (this._ownsDeviceState) this.deviceState.clear();
    this.adapterPipeline = null;
    this.compiledGraphPlan = null;
    this.currentBankResidency = EMPTY_BANK_RESIDENCY;
  }

  _rowTypedStorage(tensor: ExecutorTensor, sizeBytes = tensor.sizeBytes): RuntimeTypedArray {
    return this.decodeState._rowTypedStorage(tensor, sizeBytes);
  }

  _prepareIncrementalRowNode(node: ExecutorNode, rowSet: DecodeRowSet): ExecutorNode {
    return this.decodeState._prepareIncrementalRowNode(node, rowSet);
  }

  _incrementalRowCandidate(
    node: ExecutorNode, nodeIndex: number, lanes = 1,
  ): IncrementalRowCandidate | null {
    return this.decodeState._incrementalRowCandidate(node, nodeIndex, lanes);
  }

  _ensureIncrementalRowLanes(lanes: number): void {
    return this.decodeState._ensureIncrementalRowLanes(lanes);
  }

  _assertIncrementalRowInvariants(
    selectedNodes: Iterable<number>,
    changedInputs: readonly string[],
  ): void {
    return this.decodeState._assertIncrementalRowInvariants(selectedNodes, changedInputs);
  }

  _analyzeIncrementalRows(): void {
    return this.decodeState._analyzeIncrementalRows();
  }

  async _compileIncrementalRowPipelines(
    nodeIndices: Iterable<number> = this.incrementalRowCandidates.keys(),
  ): Promise<void> {
    const previousPlans = new Map(this.incrementalRowPlans);
    const previousResources = new Set(this.auxiliaryBuffers);
    const previousBufferCreateCount = this.specializationBufferCreateCount;
    const previousBindGroupCreateCount = this.bindGroupCreateCount;
    try {
      await this.graphCompiler._compileIncrementalRowPipelines(nodeIndices);
      // Lazy plans publish inside the compiler, but dispatch is still waiting
      // on this method. Resolve every per-create scope here, before any invalid
      // buffer or bind group can be encoded into a command stream.
      await this._drainPendingDeviceErrorChecks();
    } catch (error) {
      for (const buffer of [...this.auxiliaryBuffers]) {
        if (previousResources.has(buffer)) continue;
        this.auxiliaryBuffers.delete(buffer);
        buffer.destroy?.();
      }
      this.incrementalRowPlans.clear();
      for (const [nodeIndex, plan] of previousPlans) {
        this.incrementalRowPlans.set(nodeIndex, plan);
      }
      this.specializationBufferCreateCount = previousBufferCreateCount;
      this.bindGroupCreateCount = previousBindGroupCreateCount;
      throw error;
    }
  }

  _ensureAdapterPipeline(): Promise<GPUComputePipeline> {
    return this.graphCompiler._ensureAdapterPipeline();
  }

  _cachedComputePipeline(
    code: string,
    options: { entryPoint?: string; constants?: Record<string, number | boolean> | null } = {},
  ): Promise<GPUComputePipeline> {
    return this.graphCompiler._cachedComputePipeline(code, options);
  }

  _adapterBuffers(target: AdapterTarget): { a: GPUBuffer; b: GPUBuffer } {
    return this.resources._adapterBuffers(target);
  }

  _createAuxiliaryStorageBuffer(label: string, values: ArrayBufferView): GPUBuffer {
    return this.resources._createAuxiliaryStorageBuffer(label, values);
  }

  releaseAdapterTargets(targets: readonly AdapterTarget[] | null | undefined): void {
    return this.resources.releaseAdapterTargets(targets);
  }

  _allocateBuffers(): void {
    return this.resources._allocateBuffers();
  }

  _aliasInferenceDropoutBuffers(): void {
    return this.resources._aliasInferenceDropoutBuffers();
  }

  _buildNodePipeline(
    node: ExecutorNode,
    options: {
      pipelines?: CompiledWebGPUPipeline[];
      buffers?: Map<string, GPUBuffer>;
      resources?: Set<GPUBuffer>;
    } = {},
  ): Promise<void> {
    return this.graphCompiler._buildNodePipeline(node, options);
  }

  _buildNodePipelineImpl(node: ExecutorNode): Promise<void> {
    return this.graphCompiler._buildNodePipelineImpl(node);
  }

  _preflightQGroupNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQGroupNormDynamicAffines(inputs);
  }

  _preflightQLayerNormDynamicAffines(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQLayerNormDynamicAffines(inputs);
  }

  _preflightQSDPAMasks(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQSDPAMasks(inputs);
  }

  _preflightQMaskedMeanMasks(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQMaskedMeanMasks(inputs);
  }

  _preflightQEmbeddingIds(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightQEmbeddingIds(inputs);
  }

  _preflightCanonicalValueDomains(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightCanonicalValueDomains(inputs);
  }

  _preflightExecutionInputs(inputs: WebGPUExecutionInputs): void {
    return this.dispatch._preflightExecutionInputs(inputs);
  }


  _uploadExecutionInputs(
    inputs: WebGPUExecutionInputs,
    rowPosition: number | null,
    changedInputs?: readonly string[],
    reuseRetainedInputs = false,
  ): void {
    return this.dispatch._uploadExecutionInputs(
      inputs,
      rowPosition,
      changedInputs,
      reuseRetainedInputs,
    );
  }

  _encodeIncrementalRowByteCopy(
    commandEncoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline | null,
    copy: IncrementalRowByteCopy | undefined,
    index: number,
    sourceOffset: number,
    destinationOffset: number,
    size: number,
    label: string,
  ): void {
    return this.decodeState._encodeIncrementalRowByteCopy(
      commandEncoder, pipeline, copy, index, sourceOffset, destinationOffset, size, label,
    );
  }

  _rowTensorByName(
    tensors: Record<string, ExecutorTensor>,
    name: string,
  ): ExecutorTensor | null {
    return this.decodeState._rowTensorByName(tensors, name);
  }

  _encodeIncrementalRowsInto(
    commandEncoder: GPUCommandEncoder,
    selectedNodes: Iterable<number>,
    rowSet: DecodeRowSet,
    options: { qsdpaControlBuffer?: GPUBuffer | null } = {},
  ): void {
    return this.decodeState._encodeIncrementalRowsInto(
      commandEncoder, selectedNodes, rowSet, options,
    );
  }

  _attentionRowBytes(tensor: ExecutorTensor | undefined, label: string): number {
    return this.decodeState._attentionRowBytes(tensor, label);
  }

  async _compilePagedRowVariants(
    nodeIndices: Iterable<number>,
    pagedTensors: ReadonlySet<string>,
  ): Promise<void> {
    const previousResources = new Set(this.auxiliaryBuffers);
    const previousBufferCreateCount = this.specializationBufferCreateCount;
    const previousBindGroupCreateCount = this.bindGroupCreateCount;
    const previousPlans = new Map([...this.incrementalRowPlans].map(([nodeIndex, plan]) => [
      nodeIndex,
      {
        pagedStaging: plan.pagedStaging,
        pagedPipelines: plan.pagedPipelines,
        pagedQsdpaParamsBuffer: plan.pagedQsdpaParamsBuffer,
      },
    ] as const));
    try {
      await this.graphCompiler._compilePagedRowVariants(nodeIndices, pagedTensors);
      await this._drainPendingDeviceErrorChecks();
    } catch (error) {
      for (const buffer of [...this.auxiliaryBuffers]) {
        if (previousResources.has(buffer)) continue;
        this.auxiliaryBuffers.delete(buffer);
        buffer.destroy?.();
      }
      for (const [nodeIndex, previous] of previousPlans) {
        const plan = this.incrementalRowPlans.get(nodeIndex);
        if (!plan) continue;
        plan.pagedStaging = previous.pagedStaging;
        plan.pagedPipelines = previous.pagedPipelines;
        plan.pagedQsdpaParamsBuffer = previous.pagedQsdpaParamsBuffer;
      }
      this.specializationBufferCreateCount = previousBufferCreateCount;
      this.bindGroupCreateCount = previousBindGroupCreateCount;
      throw error;
    }
  }

  _encodeIncrementalRows(
    selectedNodes: Iterable<number>,
    rowSet: DecodeRowSet,
  ): void {
    return this.decodeState._encodeIncrementalRows(selectedNodes, rowSet);
  }

  _deviceFeedbackDescriptor(
    inputs: WebGPUExecutionInputs,
    options: DeviceFeedbackDecodeOptions = {},
  ): DeviceFeedbackDescriptor {
    return this.decodeState._deviceFeedbackDescriptor(inputs, options);
  }

  _deviceFeedbackControl(sequenceLength: number): GPUBuffer {
    return this.decodeState._deviceFeedbackControl(sequenceLength);
  }

  executeDeviceFeedbackDecode(
    inputs: WebGPUExecutionInputs,
    options: DeviceFeedbackDecodeOptions = {},
  ): Promise<GPUBuffer> {
    return this.decodeState.executeDeviceFeedbackDecode(inputs, options);
  }

  execute(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions = {},
  ): Promise<GPUBuffer | undefined> {
    assertInferenceExecutionOptions(options, 'WebGPU inference');
    return this._executeFailClosed(inputs, options, false);
  }

  /** @internal The caller completed pure value preflight for this graph. */
  _executePreflighted(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions = {},
  ): Promise<GPUBuffer | undefined> {
    assertInferenceExecutionOptions(options, 'WebGPU inference');
    return this._executeFailClosed(inputs, options, true);
  }

  private async _executeFailClosed(
    inputs: WebGPUExecutionInputs,
    options: WebGPUExecutionOptions,
    preflightComplete: boolean,
  ): Promise<GPUBuffer | undefined> {
    try {
      const output = await this.dispatch.execute(inputs, options, null, preflightComplete);
      await this._drainPendingDeviceErrorChecks();
      return output;
    } catch (error) {
      // Input writes or a rejected command stream may have partially touched
      // mutable execution storage. Force the provider to specialize again and
      // never reuse decode state from the rejected submission.
      this.currentShapeSignature = null;
      this._webGPUIncrementalCacheValid = false;
      this.deviceFeedbackState = null;
      // Drain any lazy row-specialization checks so they cannot leak an
      // unhandled rejection after the public operation has already failed.
      try {
        await this._drainPendingDeviceErrorChecks();
      } catch {
        // Preserve the first execution failure; the physical signature above
        // already makes the complete binding ineligible for reuse.
      }
      throw error;
    }
  }

  async snapshotOutputs(): Promise<ReadonlyMap<string, WebGPUOutputSnapshot>> {
    return this.results.snapshotOutputs();
  }

  readBuffer(
    gpuBuffer: GPUBuffer,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return this.results.readBuffer(gpuBuffer, sizeBytes, dtype);
  }

  readBufferRange(
    gpuBuffer: GPUBuffer,
    byteOffset: number,
    sizeBytes: number,
    dtype: RuntimeDType = 'float32',
  ): Promise<RuntimeTypedArray> {
    return this.results.readBufferRange(gpuBuffer, byteOffset, sizeBytes, dtype);
  }
}
