import { createBoundExecutionGraph } from '../core/BoundExecutionGraph.js';
import type { BackendExecutionSnapshot } from '../core/ExecutionResult.js';
import { Model } from '../core/Model.js';
import type { ResolvedShapePlan } from '../core/ResolvedShapePlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import {
  runtimeDTypeBytes,
} from '../ops/shapeSystem.js';
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendDeviceIdentity,
  createBackendProviderCapabilities,
  type BackendDeviceIdentity,
  type BackendLogicalCompileInput,
  type BackendProvider,
  type BackendProviderCapabilities,
  type BackendProviderCompilationEvidence,
  type BackendProviderCompiledModel,
  type BackendProviderCompileOptions,
  type BackendProviderContextOptions,
  type BackendProviderExecutionContext,
  type BackendResolvedExecutionRequest,
} from './BackendProvider.js';
import {
  ProviderDecodeLifecycle,
  type PreparedProviderDecode,
  type ProviderDecodeTelemetry,
} from './ProviderDecodeLifecycle.js';
import { WebGPUEngine } from './WebGPUEngine.js';
import {
  checkedWebGPUExtentBounds,
  checkedWebGPUPhysicalDomain,
} from './WebGPUPhysicalDomain.js';
import { readWebGPUBufferRange } from './WebGPUResults.js';
import type { RuntimeTypedArray } from '../types.js';
import {
  deferDeviceTensorInputLeaseReleaseUntil,
  resolveDeviceTensorInputResource,
  type DeviceTensorInputLease,
} from '../ops/deviceTensorReference.js';

const DEFAULT_PLAN_CACHE_ENTRIES = 8;
const DEFAULT_PLAN_CACHE_METADATA_BYTES = 1024 * 1024;
const UTF8_ENCODER = new TextEncoder();

function createSubmittedBufferRetirement(queue: GPUQueue): (buffer: GPUBuffer) => void {
  const pending = new Set<GPUBuffer>();
  let retired = false;
  const finish = () => {
    retired = true;
    for (const buffer of pending) buffer.destroy?.();
    pending.clear();
  };
  try {
    const completion = queue.onSubmittedWorkDone?.();
    if (completion) void Promise.resolve(completion).then(finish, finish);
  } catch {
    // Without a usable fence, retain result storage rather than destroy a
    // buffer that the already-submitted snapshot copy may still reference.
  }
  return (buffer) => {
    if (retired) buffer.destroy?.();
    else pending.add(buffer);
  };
}

interface WebGPUResourceDomainProof {
  readonly maximumTensorBytes: number;
  /** Complete maximum-domain public-input snapshot, without GPU alignment. */
  readonly maximumInputBytes: number;
  readonly maximumResidentBytes: number;
  readonly resourceLimitBytes: null;
  readonly tensorMaximumBytes: ReadonlyMap<string, number>;
}

interface WebGPULimitsLike {
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
  readonly maxComputeWorkgroupsPerDimension?: number;
}

interface CachedWebGPUVariant {
  readonly plan: ResolvedShapePlan;
  readonly metadataBytes: number;
}

function alignedBytes(bytes: bigint, label: string): bigint {
  if (bytes <= 0n) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED', `${label} has no positive storage extent.`, {
      phase: 'compilation', backend: 'webgpu',
    });
  }
  return bytes < 4n ? 4n : ((bytes + 3n) / 4n) * 4n;
}

function requiredDeviceLimit(value: unknown, name: string): bigint {
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WebGPU device does not expose a valid ${name} limit.`, {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  return BigInt(value as number);
}

function descriptorMaximumShape(
  input: BackendLogicalCompileInput,
  name: string,
): readonly bigint[] {
  const descriptor = input.graph.tensors[name];
  return descriptor.shape.map((dimension, axis) => {
    return checkedWebGPUExtentBounds(
      input,
      dimension,
      `tensor '${name}' axis ${axis}`,
    ).maximum;
  });
}

function checkedWebGPUResourceDomain(
  input: BackendLogicalCompileInput,
  device: GPUDevice,
  maximumAuxiliaryBytes: number,
): WebGPUResourceDomainProof {
  const limits = (device as GPUDevice & { limits?: WebGPULimitsLike }).limits || {};
  const maxBufferSize = requiredDeviceLimit(limits.maxBufferSize, 'maxBufferSize');
  const maxStorageBinding = requiredDeviceLimit(
    limits.maxStorageBufferBindingSize,
    'maxStorageBufferBindingSize',
  );
  const perBufferLimit = maxBufferSize < maxStorageBinding ? maxBufferSize : maxStorageBinding;
  const tensorMaximumBytes = new Map<string, number>();
  const outputNames = new Set(input.outputNames);
  let maximumTensorBytes = 0n;
  let rawWeightBytes = 0n;
  let rawBankWeightBytes = 0n;
  let maximumInputBytes = 0n;
  let maximumGPUTensorResidentBytes = 0n;
  let maximumResultSnapshotBytes = 0n;
  for (const name of Object.keys(input.graph.tensors)) {
    const descriptor = input.graph.tensors[name];
    const shape = descriptorMaximumShape(input, name);
    let elements = 1n;
    for (let axis = 0; axis < shape.length; axis++) {
      const dimension = shape[axis];
      if (dimension > 0xffffffffn) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `WebGPU tensor '${name}' axis ${axis} exceeds u32 metadata.`, {
            phase: 'compilation', backend: 'webgpu',
          });
      }
      elements *= dimension;
    }
    if (elements > 0xffffffffn) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU tensor '${name}' maximum element count exceeds u32 metadata.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
    const logicalBytes = elements * BigInt(runtimeDTypeBytes(descriptor.dtype));
    const capacityBytes = alignedBytes(logicalBytes, `WebGPU tensor '${name}'`);
    if (capacityBytes > perBufferLimit) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU tensor '${name}' can exceed maxBufferSize or maxStorageBufferBindingSize.`, {
          phase: 'compilation', backend: 'webgpu',
        });
    }
    maximumTensorBytes = logicalBytes > maximumTensorBytes ? logicalBytes : maximumTensorBytes;
    const isWeight = Object.prototype.hasOwnProperty.call(input.graph.weights, name);
    const isBankedWeight = isWeight &&
      Object.prototype.hasOwnProperty.call(input.graph.banks, name);
    const isDynamic = descriptor.shape.some((dimension) => typeof dimension !== 'number');
    if (isWeight) rawWeightBytes += logicalBytes;
    if (isBankedWeight) rawBankWeightBytes += logicalBytes;
    if (descriptor.kind === 'input') maximumInputBytes += logicalBytes;
    if (outputNames.has(name)) maximumResultSnapshotBytes += capacityBytes;
    maximumGPUTensorResidentBytes += isBankedWeight
      ? capacityBytes * 2n
      : isWeight || !isDynamic
        ? capacityBytes
        : capacityBytes * 3n;
    tensorMaximumBytes.set(name, Number(capacityBytes));
  }
  /* Incremental row plans are prepared lazily after a successful seed. Bound
   * every non-weight tensor that a node could conservatively slice into a row
   * scratch allocation, once per node, plus one packed-byte copy uniform for
   * each input/output direction. The concrete row planner admits only a subset
   * of these tensors, so this remains an upper bound without coupling the
   * provider proof to one application graph. */
  let maximumIncrementalRowScratchBytes = 0n;
  for (const node of input.graph.nodes) {
    const inputNames = new Set(Object.values(node.inputs));
    const outputNamesForNode = new Set(Object.values(node.outputs).map((output) => output.tensor));
    const scratchNames = new Set([...inputNames, ...outputNamesForNode]);
    for (const name of scratchNames) {
      if (Object.prototype.hasOwnProperty.call(input.graph.weights, name)) continue;
      const capacity = tensorMaximumBytes.get(name);
      if (capacity === undefined) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `WebGPU row resource proof cannot resolve tensor '${name}'.`, {
            phase: 'compilation', backend: 'webgpu',
          });
      }
      maximumIncrementalRowScratchBytes += BigInt(capacity);
    }
    for (const name of inputNames) {
      if (!Object.prototype.hasOwnProperty.call(input.graph.weights, name)) {
        maximumIncrementalRowScratchBytes += 16n;
      }
    }
    maximumIncrementalRowScratchBytes += BigInt(outputNamesForNode.size) * 16n;
  }
  /* The immutable Model payload and the context's private invariant host clone
   * coexist. A bank additionally keeps its committed and candidate slices.
   * Binding that candidate happens before GraphExecutor can drain an older GPU
   * retirement fence, and stageBankSlots transiently holds two complete copies
   * of the selected bank while the committed slice is still live. Charge one
   * further complete bank payload for that construction peak. GPU bank
   * accounting keeps exactly the committed and candidate buffers; dynamic
   * activation storage retains its conservative three-capacity charge. Rebind
   * backpressure drains an older retirement fence before device staging, so
   * this dominates both the current/candidate and submitted/current GPU phases.
   * One exact device result
   * snapshot remains result-owned after dispatch. The submitted forward
   * generation, its lazily prepared row specialization, and a new forward
   * candidate can coexist until queue completion; the conservative row charge
   * above also covers its scratch and packed-copy uniforms. The context keeps
   * one exact host mirror of committed specialization bytes so byte-identical
   * dynamic rebind writes can be elided without hashes or collision risk.
   * Finally, the context owns a byte-bounded plan LRU independently of its GPU
   * allocations.
   * Core decode replacement/reseed also stages a complete candidate public-
   * input clone before provider entry while retaining the committed clone. */
  const residentHostWeights = rawWeightBytes * 2n + rawBankWeightBytes * 3n;
  const retainedDecodeInputs = maximumInputBytes * 2n;
  const maximumAuxiliaryResidentBytes = BigInt(maximumAuxiliaryBytes) * 4n +
    maximumIncrementalRowScratchBytes;
  const maximumPlanCacheMetadataBytes = BigInt(DEFAULT_PLAN_CACHE_METADATA_BYTES);
  const maximumResidentBytes = residentHostWeights + maximumGPUTensorResidentBytes +
    maximumResultSnapshotBytes + maximumAuxiliaryResidentBytes +
    maximumPlanCacheMetadataBytes + retainedDecodeInputs;
  if (maximumTensorBytes > BigInt(Number.MAX_SAFE_INTEGER) ||
      maximumResidentBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      'WebGPU bounded-domain resource totals exceed JavaScript safe integers.', {
        phase: 'compilation', backend: 'webgpu',
      });
  }
  return Object.freeze({
    maximumTensorBytes: Number(maximumTensorBytes),
    maximumInputBytes: Number(maximumInputBytes),
    maximumResidentBytes: Number(maximumResidentBytes),
    resourceLimitBytes: null,
    tensorMaximumBytes,
  });
}

function planMetadataBytes(plan: ResolvedShapePlan): number {
  let bytes = UTF8_ENCODER.encode(plan.signature).byteLength + 128;
  for (const tensor of Object.values(plan.tensors)) {
    bytes += UTF8_ENCODER.encode(tensor.name).byteLength + 64 + tensor.shape.length * 8;
  }
  return Number.isSafeInteger(bytes) ? bytes : Number.MAX_SAFE_INTEGER;
}

class WebGPUProviderExecutionContext implements BackendProviderExecutionContext {
  readonly backendName = 'webgpu';
  readonly #snapshot: Model;
  readonly #tensorMaximumBytes: ReadonlyMap<string, number>;
  readonly #invariantWeights = new Map<string, RuntimeTypedArray>();
  readonly #deviceLostPromise: Promise<GPUDeviceLostInfo> | null;
  readonly #decode: ProviderDecodeLifecycle;
  #engine: WebGPUEngine | null;
  #cache = new Map<string, CachedWebGPUVariant>();
  #cacheBytes = 0;
  #cacheHits = 0;
  #cacheMisses = 0;
  #cacheEvictions = 0;
  #currentSignature: string | null = null;
  #deviceLost: GPUDeviceLostInfo | null = null;
  #closed = false;

  constructor(
    snapshot: Model,
    engine: WebGPUEngine,
    tensorMaximumBytes: ReadonlyMap<string, number>,
    options?: BackendProviderContextOptions,
  ) {
    this.#snapshot = snapshot;
    this.#engine = engine;
    this.#tensorMaximumBytes = tensorMaximumBytes;
    this.#decode = new ProviderDecodeLifecycle(snapshot, this.backendName, {
      incrementalExecution: true,
      incrementalRows: true,
    }, options?.decode);
    for (const name of snapshot.weightNames) {
      this.#invariantWeights.set(name, snapshot.copyWeightData(name));
    }
    const lost = (engine.device as GPUDevice & { lost?: Promise<GPUDeviceLostInfo> }).lost;
    if (lost && typeof lost.then === 'function') {
      this.#deviceLostPromise = lost;
      void this.#deviceLostPromise.then((info) => {
        this.#deviceLost = info || ({ reason: 'unknown', message: 'WebGPU device was lost.' } as GPUDeviceLostInfo);
        this.#cache.clear();
        this.#cacheBytes = 0;
        this.#currentSignature = null;
      }, (error) => {
        this.#deviceLost = ({
          reason: 'unknown',
          message: error instanceof Error ? error.message : String(error),
        } as GPUDeviceLostInfo);
        this.#cache.clear();
        this.#cacheBytes = 0;
        this.#currentSignature = null;
      });
    } else {
      this.#deviceLostPromise = null;
    }
  }

  #assertDeviceAvailable(): void {
    if (!this.#deviceLost) return;
    throw new VolvoxAIError('DEVICE_LOST',
      `WebGPU device was lost (${String(this.#deviceLost.reason)}): ${this.#deviceLost.message}`, {
        phase: 'execution', backend: this.backendName,
      });
  }

  async execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    if (this.#closed || !this.#engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebGPU execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#assertDeviceAvailable();
    const engine = this.#engine;
    if (request.operation !== 'execute') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WebGPU ordinary execute does not implement '${request.operation}'.`, {
          phase: 'execution', backend: this.backendName,
        });
    }
    const invalidateSeed = this.#decode.seeded;
    const beforeDispatch = () => {
      request.commitExecution();
      if (invalidateSeed) this.#decode.invalidate('ordinary-execution');
      if (invalidateSeed) engine.resetDecodeBinding();
    };
    const deviceInputs = this.#validateDeviceInputs(request, engine);
    if (deviceInputs.length === 0) {
      return this.#executeResolved(request, {}, beforeDispatch);
    }
    const queue = engine.device.queue;
    if (typeof queue.onSubmittedWorkDone !== 'function') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'WebGPU device inputs require a queue completion fence.', {
          phase: 'execution', backend: this.backendName,
        });
    }
    let settleRetirement!: () => void;
    const retirement = new Promise<void>((resolve) => {
      settleRetirement = resolve;
    });
    for (const lease of deviceInputs) {
      deferDeviceTensorInputLeaseReleaseUntil(lease, retirement);
    }
    try {
      return await this.#executeResolved(request, {}, beforeDispatch);
    } finally {
      // Called after every queue submission attempted by this execution. The
      // fence covers D2D inputs, compute, and result snapshots. If acquiring
      // the fence itself fails, keep the retirement unresolved rather than
      // destroying storage that submitted work may still reference.
      try {
        void queue.onSubmittedWorkDone().then(settleRetirement, settleRetirement);
      } catch {
        // Safety takes precedence over reclaiming one result after a broken queue.
      }
    }
  }

  #validateDeviceInputs(
    request: BackendResolvedExecutionRequest,
    engine: WebGPUEngine,
  ): readonly DeviceTensorInputLease[] {
    const leases = Object.entries(request.deviceInputs);
    for (const [name, lease] of leases) {
      const descriptor = request.plan.tensors[name];
      let source: GPUBuffer;
      try {
        source = resolveDeviceTensorInputResource(
          lease, 'webgpu', engine.device,
        ) as GPUBuffer;
      } catch (error) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `WebGPU device input '${name}' must originate from the same physical GPUDevice.`, {
            phase: 'execution', backend: this.backendName, cause: error,
          });
      }
      const paddedBytes = descriptor
        ? Math.max(4, Math.ceil(descriptor.sizeBytes / 4) * 4)
        : 0;
      if (!descriptor || descriptor.kind !== 'input' ||
          lease.dtype !== descriptor.dtype ||
          lease.logicalSizeBytes !== descriptor.sizeBytes ||
          source.size !== paddedBytes ||
          (source as { destroyed?: boolean }).destroyed === true) {
        throw new VolvoxAIError('INVALID_ARGUMENT',
          `WebGPU device input '${name}' disagrees with its resolved descriptor or live storage.`, {
            phase: 'execution', backend: this.backendName,
          });
      }
    }
    return Object.freeze(leases.map(([, lease]) => lease));
  }

  async decodeSeed(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    return this.#executeDecode(request, this.#decode.prepareSeed(request));
  }

  async decodeStep(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    return this.#executeDecode(request, this.#decode.prepareStep(request));
  }

  async decodeReset(): Promise<void> {
    if (this.#closed || !this.#engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebGPU execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#engine.resetDecodeBinding();
    this.#decode.reset();
  }

  async #executeDecode(
    request: BackendResolvedExecutionRequest,
    prepared: PreparedProviderDecode,
  ): Promise<BackendExecutionSnapshot> {
    const engine = this.#engine;
    if (this.#closed || !engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebGPU execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    let mutationStarted = false;
    try {
      const snapshot = await this.#executeResolved(
        request,
        prepared.engineOptions,
        () => {
          request.commitExecution();
          mutationStarted = true;
          if (prepared.operation === 'seed') {
            if (prepared.automaticResetReason !== null &&
                prepared.automaticResetReason !== 'reseed') {
              engine.resetDecodeBinding();
            } else {
              engine.resetDecodeCache();
            }
          }
        },
      );
      return this.#withDecodeTelemetry(snapshot, this.#decode.commit(prepared));
    } catch (error) {
      if (mutationStarted) {
        engine.resetDecodeBinding();
        this.#decode.failExecution();
      }
      throw error;
    }
  }

  #withDecodeTelemetry(
    snapshot: BackendExecutionSnapshot,
    decode: Readonly<ProviderDecodeTelemetry>,
  ): BackendExecutionSnapshot {
    return Object.freeze({
      outputs: snapshot.outputs,
      backendReport: Object.freeze({
        ...(snapshot.backendReport ?? {}),
        decode,
      }),
    });
  }

  async #executeResolved(
    request: BackendResolvedExecutionRequest,
    executionOptions: PreparedProviderDecode['engineOptions'] = {},
    beforeDispatch?: (() => void),
  ): Promise<BackendExecutionSnapshot> {
    const engine = this.#engine;
    if (this.#closed || !engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WebGPU execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#assertDeviceAvailable();
    if (request.adapters.selectors.some((selector) => selector !== null)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'Logical bounded-shape WebGPU models do not contain adapter revisions.', {
          phase: 'execution', backend: this.backendName,
        });
    }
    let variant = this.#cache.get(request.signature);
    const cacheHit = variant !== undefined;
    let cacheEvictions = 0;
    const candidateCache = new Map(this.#cache);
    let candidateBytes = this.#cacheBytes;
    if (variant) {
      candidateCache.delete(request.signature);
      candidateCache.set(request.signature, variant);
    } else {
      variant = Object.freeze({
        plan: request.plan,
        metadataBytes: planMetadataBytes(request.plan),
      });
      if (variant.metadataBytes <= DEFAULT_PLAN_CACHE_METADATA_BYTES) {
        candidateCache.set(request.signature, variant);
        candidateBytes += variant.metadataBytes;
        while (candidateCache.size > DEFAULT_PLAN_CACHE_ENTRIES ||
               candidateBytes > DEFAULT_PLAN_CACHE_METADATA_BYTES) {
          const oldest = candidateCache.keys().next().value;
          if (oldest === undefined) break;
          const removed = candidateCache.get(oldest);
          candidateCache.delete(oldest);
          candidateBytes -= removed?.metadataBytes || 0;
          cacheEvictions++;
        }
      }
    }

    const rawInputs = Object.fromEntries(this.#snapshot.inputNames.map((name) => [
      name,
      request.deviceInputs[name] ?? request.inputs[name].data,
    ]));
    let executionCommitted = false;
    const commitExecution = () => {
      if (executionCommitted) return;
      executionCommitted = true;
      beforeDispatch?.();
    };
    if (this.#currentSignature !== request.signature) {
      if (variant.plan.graphFingerprint !== request.plan.graphFingerprint ||
          variant.plan.signature !== request.signature) {
        throw new VolvoxAIError('EXECUTION_FAILED',
          'WebGPU cached specialization metadata disagrees with the resolved request.', {
            phase: 'execution', backend: this.backendName,
          });
      }
      const bound = createBoundExecutionGraph(this.#snapshot, variant.plan, {
        invariantWeightStorageFactory: ({ name }) => this.#invariantWeights.get(name),
      });
      // Reject request values against the candidate graph before rebind can
      // publish capacities, specializations, or a new current signature.
      engine.preflightExecutionInputs(rawInputs, bound.graph);
      try {
        await engine.rebindGraph(bound.graph, {
          shapeSignature: request.signature,
          bankResidency: variant.plan.bankResidency,
          beforeCommit: commitExecution,
          tensorMaximumBytes: this.#tensorMaximumBytes,
          capacityGrowthFactor: 2,
        });
      } catch (error) {
        // A queue write can fail after it has partially updated reused
        // specialization buffers. GraphExecutor marks that physical binding
        // unexecutable; mirror the invalidation here so an old-shape request
        // cannot take the provider's same-signature fast path.
        let physicalSignature: string | null | undefined;
        try {
          physicalSignature = engine.inspectDynamicResources().shapeSignature;
        } catch {
          // An initial candidate has no published executor to inspect.
          physicalSignature = undefined;
        }
        if (physicalSignature === null) {
          this.#currentSignature = null;
        }
        throw error;
      }
      this.#assertDeviceAvailable();
    } else {
      // Cache hit accounting and LRU order are committed below, so they are
      // covered by the same no-state-change-on-preflight-failure boundary.
      engine.preflightExecutionInputs(rawInputs);
    }
    // Publish metadata only after complete specialization; no input bytes have
    // reached the queue before this point.
    this.#cache = candidateCache;
    this.#cacheBytes = candidateBytes;
    this.#currentSignature = request.signature;
    if (cacheHit) this.#cacheHits++;
    else this.#cacheMisses++;
    this.#cacheEvictions += cacheEvictions;

    commitExecution();
    await engine._executePreflighted(rawInputs, executionOptions);
    this.#assertDeviceAvailable();
    const snapshots = engine.snapshotOutputs();
    const device = engine.device;
    // snapshotOutputs submits D2D copies. Result.close() and every construction
    // failure path may request release immediately, but physical destruction
    // must remain behind completion of those submitted copies.
    const retireSnapshotBuffer = createSubmittedBufferRetirement(device.queue);
    try {
      const outputs = request.outputDescriptors.map((descriptor) => {
        const output = snapshots.get(descriptor.name);
        if (!output || output.dtype !== descriptor.dtype ||
            output.sizeBytes !== descriptor.sizeBytes ||
            output.shape.length !== descriptor.shape.length ||
            output.shape.some((dimension, axis) => dimension !== descriptor.shape[axis])) {
          throw new VolvoxAIError('EXECUTION_FAILED',
            `WebGPU output '${descriptor.name}' disagrees with its resolved descriptor.`, {
              phase: 'execution', backend: this.backendName,
            });
        }
        let released = false;
        return Object.freeze({
          name: output.name,
          shape: output.shape,
          dtype: output.dtype,
          location: 'device' as const,
          logicalSizeBytes: output.sizeBytes,
          deviceBuffer: output.deviceBuffer,
          deviceType: 'webgpu' as const,
          device,
          read: () => readWebGPUBufferRange(
            device,
            output.deviceBuffer,
            0,
            output.sizeBytes,
            output.dtype,
          ),
          release: () => {
            if (released) return;
            released = true;
            retireSnapshotBuffer(output.deviceBuffer);
          },
        });
      });
      const resources = engine.inspectDynamicResources();
      return Object.freeze({
        outputs: Object.freeze(outputs),
        backendReport: Object.freeze({
          shapeSignature: request.signature,
          specializationCacheHit: cacheHit,
          specializationCacheEntries: this.#cache.size,
          specializationCacheMetadataBytes: this.#cacheBytes,
          specializationCacheHits: this.#cacheHits,
          specializationCacheMisses: this.#cacheMisses,
          specializationCacheEvictions: this.#cacheEvictions,
          logicalActivationBytes: resources.logicalActivationBytes,
          activationCapacityBytes: resources.activationCapacityBytes,
          activationCapacityHighWaterBytes: resources.activationCapacityHighWaterBytes,
          activationGrowCount: resources.activationGrowCount,
          specializationRebindCount: resources.specializationRebindCount,
          specializationBufferCreateCount: resources.specializationBufferCreateCount,
          specializationBufferReuseCount: resources.specializationBufferReuseCount,
          liveSpecializationBufferCount: resources.liveSpecializationBufferCount,
          bindGroupCreateCount: resources.bindGroupCreateCount,
          bindGroupReuseCount: resources.bindGroupReuseCount,
          liveBindGroupCount: resources.liveBindGroupCount,
          specializationWriteCount: resources.specializationWriteCount,
          specializationWriteBytes: resources.specializationWriteBytes,
          specializationWriteSkipCount: resources.specializationWriteSkipCount,
          specializationWriteSkipBytes: resources.specializationWriteSkipBytes,
          specializationContentBytes: resources.specializationContentBytes,
          pendingRetiredBufferCount: resources.pendingRetiredBufferCount,
          selectedTactics: resources.selectedTactics,
        }),
      });
    } catch (error) {
      for (const output of snapshots.values()) retireSnapshotBuffer(output.deviceBuffer);
      throw error;
    }
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#decode.close();
    const engine = this.#engine;
    this.#engine = null;
    this.#cache.clear();
    this.#cacheBytes = 0;
    this.#invariantWeights.clear();
    if (!engine) return;
    try {
      await engine.device.queue.onSubmittedWorkDone?.();
    } catch (error) {
      if (!this.#deviceLost && this.#deviceLostPromise) {
        try {
          const info = await this.#deviceLostPromise;
          this.#deviceLost = info || ({
            reason: 'unknown', message: 'WebGPU device was lost.',
          } as GPUDeviceLostInfo);
        } catch (lostError) {
          this.#deviceLost = ({
            reason: 'unknown',
            message: lostError instanceof Error ? lostError.message : String(lostError),
          } as GPUDeviceLostInfo);
        }
      }
      if (!this.#deviceLost) throw error;
    } finally {
      engine.dispose();
    }
  }
}

class WebGPUProviderCompiledModel implements BackendProviderCompiledModel {
  readonly backendName = 'webgpu';
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly #snapshot: Model;
  readonly #resources: WebGPUResourceDomainProof;
  #source: WebGPUEngine | null;
  #closed = false;

  constructor(
    input: BackendLogicalCompileInput,
    source: WebGPUEngine,
    deviceIdentity: BackendDeviceIdentity | null,
    resources: WebGPUResourceDomainProof,
  ) {
    this.#snapshot = input.snapshot;
    this.#source = source;
    this.#resources = resources;
    this.compilationEvidence = Object.freeze({
      device: deviceIdentity,
      allocationBytes: 0,
      operatorFallbackUsed: false,
      offendingNode: null,
      shapeDomain: Object.freeze({
        proofProtocol: 'canonical-symbolic-domain-proof/v1' as const,
        resourceProtocol: 'bounded-resource-maxima/v1' as const,
        support: 'full' as const,
        graphFingerprint: input.graphFingerprint,
        proof: input.shapeDomainProof,
        maximumTensorBytes: resources.maximumTensorBytes,
        maximumInputBytes: resources.maximumInputBytes,
        maximumResidentBytes: resources.maximumResidentBytes,
        resourceLimitBytes: resources.resourceLimitBytes,
      }),
    });
  }

  createContext(options?: BackendProviderContextOptions): BackendProviderExecutionContext {
    if (this.#closed || !this.#source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled WebGPU model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    return new WebGPUProviderExecutionContext(
      this.#snapshot,
      this.#source.fork(),
      this.#resources.tensorMaximumBytes,
      options,
    );
  }

  close(): void {
    this.#closed = true;
    this.#source = null;
  }
}

/** Bounded-shape WebGPU provider with one growable resource generation per context. */
export class WebGPUBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName = 'webgpu';
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  #source: WebGPUEngine | null;
  #closed = false;

  constructor(source: WebGPUEngine) {
    if (!(source instanceof WebGPUEngine)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'WebGPU provider requires a WebGPUEngine.', {
        phase: 'initialization', backend: this.backendName,
      });
    }
    this.#source = source;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'device',
      dynamicShapeDomain: 'full',
    });
    this.deviceIdentity = createBackendDeviceIdentity(source.adapterInfo) || Object.freeze({
      backend: 'webgpu',
      device: 'gpu',
    });
  }

  async compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed || !this.#source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', "Backend provider 'webgpu' is closed.", {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!(input?.snapshot instanceof Model) ||
        input.shapeDomainProof !== input.snapshot.shapeDomainProof ||
        input.graphFingerprint !== input.snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'WebGPU provider compile requires the exact immutable logical compile view.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    if (options.operatorFallback === 'forbid' && this.capabilities.operatorFallback !== 'none') {
      throw new VolvoxAIError('OPERATOR_FALLBACK_FORBIDDEN',
        "Backend 'webgpu' cannot attest strict operator routing.", {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const physical = checkedWebGPUPhysicalDomain(input, this.#source.device);
    const resources = checkedWebGPUResourceDomain(
      input,
      this.#source.device,
      physical.maximumAuxiliaryBytes,
    );
    try {
      await this.#source.precompileDynamicPipelines(
        physical.requiredShaderMethods,
        [
          ...physical.optionalShaderMethods,
          ...(this.#source.packedDot4Available ? physical.packedDot4OptionalShaderMethods : []),
        ],
      );
    } catch (error) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'WebGPU failed to compile a required bounded-domain shader route.', {
          phase: 'compilation', backend: this.backendName, cause: error,
        });
    }
    return new WebGPUProviderCompiledModel(input, this.#source, this.deviceIdentity, resources);
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#source?.dispose();
    this.#source = null;
  }
}
