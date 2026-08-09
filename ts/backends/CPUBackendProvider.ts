import { assertBuiltInEngine } from './BackendEngine.js';
import { CPUEngine } from './CPUEngine.js';
import { CPUShapeExecutionContext } from '../core/CPUShapeExecutionContext.js';
import type { BackendExecutionSnapshot } from '../core/ExecutionResult.js';
import { Model } from '../core/Model.js';
import {
  maximumCPUActivationArenaTensors,
  planCPUActivationArena,
} from '../core/CPUActivationArenaPlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import {
  runtimeDTypeBytes,
} from '../ops/shapeSystem.js';
import { runtimeSupportsOperator } from '../generated/kernelRegistry.js';
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendDeviceIdentity,
  createBackendProviderCapabilities,
  requireHostExecutionInputs,
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

const CPU_CONTEXT_CAPACITY_LIMIT_BYTES = 512 * 1024 * 1024;
const CPU_CONTEXT_RESIDENT_LIMIT_BYTES = 1024 * 1024 * 1024;

interface CPUResourceDomainProof {
  readonly maximumTensorBytes: number;
  readonly weightBytes: number;
  readonly maximumActivationArenaBytes: number;
  readonly maximumPersistentActivationBytes: number;
  /** Complete maximum-domain public-input snapshot, without arena alignment. */
  readonly maximumInputBytes: number;
  readonly maximumResultBytes: number;
  readonly maximumTypedScratchBytes: number;
  /** Ordinary liveness context: weight staging, grow transaction, or execution peak. */
  readonly maximumResidentBytes: number;
  /** Incremental context: every activation is retained between seed and steps. */
  readonly maximumDecodeResidentBytes: number;
  readonly decodeContextSupported: boolean;
  readonly capacityLimitBytes: number;
  readonly residentLimitBytes: number;
  /** Generic provider attestation ceiling; for CPU this is the resident limit. */
  readonly resourceLimitBytes: number;
}

const CPU_ZERO_TYPED_SCRATCH_OPERATORS = new Set([
  'MatMul', 'Linear', 'QLinear', 'BatchMatMul', 'QBatchMatMul', 'QGemm', 'Gemm',
  'QMatMul', 'MoELinear', 'LayerNorm', 'QLayerNorm', 'GroupNorm', 'QGroupNorm',
  'RMSNorm', 'BatchNorm2D', 'Conv2D', 'QConv2D', 'Conv1D', 'ConvTranspose2D',
  'MaxPool2D', 'AveragePool2D', 'GlobalAveragePool', 'Resize', 'Interpolate1D',
  'ResizeNearest2D', 'UpsampleNearest2D', 'ReLU', 'LeakyReLU', 'PReLU', 'GELU',
  'QGELU', 'SiLU', 'QSiLU', 'Sigmoid', 'HardSwish', 'HardSigmoid', 'Tanh', 'Sin',
  'Cos', 'Clip', 'Add', 'QAdd', 'Sub', 'Mul', 'Div', 'Softmax', 'LogSoftmax',
  'ReduceSum', 'ReduceMean', 'ArgMax', 'QArgMax', 'Equal', 'GreaterOrEqual', 'Not',
  'Where', 'Mask', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose',
  'Concat', 'Split', 'Slice', 'Pad', 'Expand', 'Gather', 'GatherElements', 'Identity',
  'Broadcast', 'Concat2', 'QuantizeLinear', 'DequantizeLinear', 'RequantizeLinear',
  'Cast', 'Embedding', 'RoPE', 'NonMaxSuppression', 'SpatialSoftargmaxY',
  'MeanHeight', 'ProfileX', 'ProfileY', 'Dropout',
]);

function checkedResourceAdd(...values: bigint[]): bigint {
  let total = 0n;
  for (const value of values) {
    if (value < 0n) throw new Error('CPU resource term must be non-negative.');
    total += value;
  }
  if (total > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error('CPU resource proof exceeds the safe integer range.');
  }
  return total;
}

function admittedMaximum(
  input: BackendLogicalCompileInput,
  dimension: number | string,
): number {
  if (typeof dimension === 'number') return dimension;
  const constraint = input.graph.dimensions[dimension];
  if (!constraint) throw new Error(`CPU resource proof cannot resolve '${dimension}'.`);
  const maximum = constraint.max - (constraint.max % constraint.multiple_of);
  if (!Number.isSafeInteger(maximum) || maximum < constraint.min) {
    throw new Error(`CPU resource proof has no admitted maximum for '${dimension}'.`);
  }
  return maximum;
}

function maximumShapeExtent(
  input: BackendLogicalCompileInput,
  tensorName: string,
  axis: number,
): number {
  const tensor = input.graph.tensors[tensorName];
  if (!tensor) throw new Error(`CPU scratch proof references unknown tensor '${tensorName}'.`);
  const normalizedAxis = axis < 0 ? tensor.shape.length + axis : axis;
  const dimension = tensor.shape[normalizedAxis];
  if (dimension === undefined) {
    throw new Error(`CPU scratch proof axis ${axis} is invalid for '${tensorName}'.`);
  }
  return admittedMaximum(input, dimension);
}

function checkedScratchProduct(label: string, ...values: number[]): bigint {
  let result = 1n;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${label} contains an invalid extent.`);
    }
    result *= BigInt(value);
  }
  if (result > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds the safe integer range.`);
  }
  return result;
}

function maximumCPUNodeTypedScratch(
  input: BackendLogicalCompileInput,
  node: BackendLogicalCompileInput['graph']['nodes'][number],
): bigint {
  const name = (port: string): string => {
    const value = node.inputs[port];
    if (typeof value !== 'string') {
      throw new Error(`CPU scratch proof for '${node.id}' requires input '${port}'.`);
    }
    return value;
  };
  const oneOf = (...ports: string[]): string => {
    for (const port of ports) {
      const value = node.inputs[port];
      if (typeof value === 'string') return value;
    }
    throw new Error(
      `CPU scratch proof for '${node.id}' requires one of inputs ${ports.join(', ')}.`,
    );
  };
  if (CPU_ZERO_TYPED_SCRATCH_OPERATORS.has(node.opType)) return 0n;
  switch (node.opType) {
    case 'QEmbedding':
      return checkedScratchProduct('QEmbedding row-scale scratch',
        maximumShapeExtent(input, name('weight'), 0), 4);
    case 'QSDPA': {
      const q = name('q');
      const heads = Number(node.params.heads);
      const dModel = maximumShapeExtent(input, q, -1);
      if (!Number.isSafeInteger(heads) || heads <= 0 || dModel % heads !== 0) {
        throw new Error(`CPU QSDPA scratch proof cannot divide ${dModel} by heads.`);
      }
      return checkedScratchProduct('QSDPA accumulator scratch', dModel / heads, 4);
    }
    case 'SDPA':
      return checkedScratchProduct('SDPA probability scratch',
        maximumShapeExtent(input, name('qkv'), -2), 8);
    case 'CrossSDPA':
      return checkedScratchProduct('CrossSDPA probability scratch',
        maximumShapeExtent(input, name('k'), -2), 8);
    case 'CrossAttention': {
      const q = name('q');
      const kv = name('kv');
      const heads = node.params.heads === undefined ? 8 : Number(node.params.heads);
      const dModel = maximumShapeExtent(input, q, -1);
      if (!Number.isSafeInteger(heads) || heads <= 0 || dModel % heads !== 0) {
        throw new Error('CPU CrossAttention scratch proof has invalid head geometry.');
      }
      return checkedScratchProduct('CrossAttention private vectors',
        dModel / heads + maximumShapeExtent(input, kv, -2), 4);
    }
    case 'MoERouter': {
      const weight = oneOf('weight', 'router_weight');
      const experts = maximumShapeExtent(input, weight, -1);
      const topK = node.params.top_k === undefined ? 2 : Number(node.params.top_k);
      return checkedScratchProduct('MoERouter private vectors', experts + topK, 4);
    }
    case 'QMaskedMean':
      return checkedScratchProduct('QMaskedMean row counts',
        maximumShapeExtent(input, name('input'), 0), 4);
    case 'SSMScan':
    case 'SelectiveScan': {
      const activation = oneOf('input', 'u');
      const rank = input.graph.tensors[activation].shape.length;
      const batch = rank === 2 ? 1 : maximumShapeExtent(input, activation, 0);
      const channels = maximumShapeExtent(input, activation, -1);
      const stateWidth = maximumShapeExtent(input, oneOf('A', 'a'), 1);
      return checkedScratchProduct('SSMScan retained state copy', batch, channels, stateWidth, 4);
    }
    default:
      throw new Error(
        `CPU operator '${node.opType}' lacks an audited typed-scratch resource proof.`,
      );
  }
}

function cpuRegistryPreflight(input: BackendLogicalCompileInput): void {
  for (const node of input.graph.nodes) {
    if (!runtimeSupportsOperator('cpu-js', node.opType)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend 'cpu' does not support operator '${node.opType}' at node '${node.id}'.`, {
          phase: 'compilation', backend: 'cpu', node: node.id,
        });
    }
  }
}

function checkedCPUResourceDomain(input: BackendLogicalCompileInput): CPUResourceDomainProof {
  let maximumTensorBytes = 0n;
  let weightBytes = 0n;
  let persistentActivationBytes = 0n;
  let maximumInputBytes = 0n;
  let resultBytes = 0n;
  let typedScratchBytes = 0n;
  let maximumTensors: ReturnType<typeof maximumCPUActivationArenaTensors>;
  try {
    maximumTensors = maximumCPUActivationArenaTensors(input.graph);
    for (const [name, descriptor] of Object.entries(maximumTensors)) {
      const logicalBytes = BigInt(descriptor.sizeBytes);
      maximumTensorBytes = logicalBytes > maximumTensorBytes
        ? logicalBytes
        : maximumTensorBytes;
      if (descriptor.kind === 'weight') weightBytes += logicalBytes;
      else {
        persistentActivationBytes += logicalBytes;
        if (descriptor.kind === 'input') maximumInputBytes += logicalBytes;
      }
      if (input.graph.outputs.includes(name)) resultBytes += logicalBytes;
    }
    for (const node of input.graph.nodes) {
      const scratch = maximumCPUNodeTypedScratch(input, node);
      if (scratch > typedScratchBytes) typedScratchBytes = scratch;
    }
  } catch (error) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `CPU resource proof failed: ${error instanceof Error ? error.message : String(error)}`, {
        phase: 'compilation', backend: 'cpu', cause: error,
      });
  }
  let activationArenaBytes: bigint;
  try {
    activationArenaBytes = BigInt(planCPUActivationArena(
      input.graph,
      `${input.graphFingerprint}|maximum-domain`,
      maximumTensors,
    ).capacityBytes);
  } catch (error) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `CPU topology-liveness proof failed: ${error instanceof Error ? error.message : String(error)}`, {
        phase: 'compilation', backend: 'cpu', cause: error,
      });
  }

  // The compiled Model retains its immutable payload while a
  // context owns a separate invariant-weight materialization, so every phase
  // carries two complete raw weight revisions.
  const residentWeights = weightBytes * 2n;
  const ordinaryInitial = checkedResourceAdd(residentWeights, activationArenaBytes);
  const ordinaryTransaction = input.snapshot.isStatic
    ? checkedResourceAdd(residentWeights, persistentActivationBytes)
    : checkedResourceAdd(residentWeights, activationArenaBytes * 2n);
  const ordinaryExecution = input.snapshot.isStatic
    ? checkedResourceAdd(residentWeights, persistentActivationBytes, resultBytes, typedScratchBytes)
    : checkedResourceAdd(residentWeights, activationArenaBytes, resultBytes, typedScratchBytes);
  const maximumResidentBytes = [ordinaryInitial, ordinaryTransaction, ordinaryExecution]
    .reduce((maximum, value) => value > maximum ? value : maximum, 0n);
  const decodeInitial = checkedResourceAdd(residentWeights, persistentActivationBytes);
  /* Core decode replacement/reseed stages a complete candidate public-input
   * clone before entering the provider while the previous committed clone is
   * still retained. Both host snapshots therefore coexist with provider
   * specialization growth and execution resources. */
  const retainedDecodeInputs = maximumInputBytes * 2n;
  const decodeTransaction = checkedResourceAdd(
    residentWeights,
    persistentActivationBytes * 2n,
    retainedDecodeInputs,
  );
  const decodeExecution = checkedResourceAdd(
    residentWeights,
    persistentActivationBytes,
    resultBytes,
    typedScratchBytes,
    retainedDecodeInputs,
  );
  const maximumDecodeResidentBytes = [decodeInitial, decodeTransaction, decodeExecution]
    .reduce((maximum, value) => value > maximum ? value : maximum, 0n);
  if (activationArenaBytes > BigInt(CPU_CONTEXT_CAPACITY_LIMIT_BYTES)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `CPU bounded-domain activation arena requires ${activationArenaBytes} bytes, exceeding ` +
      `the committed-capacity limit ${CPU_CONTEXT_CAPACITY_LIMIT_BYTES}.`, {
        phase: 'compilation', backend: 'cpu',
      });
  }
  if (maximumResidentBytes > BigInt(CPU_CONTEXT_RESIDENT_LIMIT_BYTES)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `CPU bounded-domain resident resources require ${maximumResidentBytes} bytes, exceeding ` +
      `the resident/transaction limit ${CPU_CONTEXT_RESIDENT_LIMIT_BYTES}.`, {
        phase: 'compilation', backend: 'cpu',
      });
  }
  const decodeContextSupported = persistentActivationBytes <=
      BigInt(CPU_CONTEXT_CAPACITY_LIMIT_BYTES) &&
    maximumDecodeResidentBytes <= BigInt(CPU_CONTEXT_RESIDENT_LIMIT_BYTES);
  return Object.freeze({
    maximumTensorBytes: Number(maximumTensorBytes),
    weightBytes: Number(weightBytes),
    maximumActivationArenaBytes: Number(activationArenaBytes),
    maximumPersistentActivationBytes: Number(persistentActivationBytes),
    maximumInputBytes: Number(maximumInputBytes),
    maximumResultBytes: Number(resultBytes),
    maximumTypedScratchBytes: Number(typedScratchBytes),
    maximumResidentBytes: Number(maximumResidentBytes),
    maximumDecodeResidentBytes: Number(maximumDecodeResidentBytes),
    decodeContextSupported,
    capacityLimitBytes: CPU_CONTEXT_CAPACITY_LIMIT_BYTES,
    residentLimitBytes: CPU_CONTEXT_RESIDENT_LIMIT_BYTES,
    resourceLimitBytes: CPU_CONTEXT_RESIDENT_LIMIT_BYTES,
  });
}

/** Context-local concrete CPU bridge. Shape inference has already happened. */
class CPUProviderExecutionContext implements BackendProviderExecutionContext {
  readonly backendName = 'cpu';
  readonly #context: CPUShapeExecutionContext;
  readonly #decode: ProviderDecodeLifecycle;
  readonly #decodeEnabled: boolean;
  #closed = false;

  constructor(
    snapshot: Model,
    options: BackendProviderContextOptions | undefined,
    decodeEnabled: boolean,
  ) {
    this.#context = new CPUShapeExecutionContext(snapshot, {
      activationStorage: decodeEnabled ? 'persistent' : 'liveness',
    });
    this.#decodeEnabled = decodeEnabled;
    this.#decode = new ProviderDecodeLifecycle(snapshot, this.backendName, {
      incrementalExecution: true,
      incrementalRows: true,
    }, options?.decode);
  }

  async execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Backend execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (request.operation !== 'execute') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Built-in CPU context does not implement '${request.operation}' through execute().`, {
          phase: 'execution', backend: this.backendName,
        });
    }
    const invalidateSeed = this.#decode.seeded;
    return this.#executeResolved(request, {}, () => {
      request.commitExecution();
      if (invalidateSeed) this.#decode.invalidate('ordinary-execution');
      if (invalidateSeed) this.#context.resetDecodeCache();
    });
  }

  async decodeSeed(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    this.#assertDecodeEnabled();
    return this.#executeDecode(request, this.#decode.prepareSeed(request));
  }

  async decodeStep(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    this.#assertDecodeEnabled();
    return this.#executeDecode(request, this.#decode.prepareStep(request));
  }

  async decodeReset(): Promise<void> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Backend execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#assertDecodeEnabled();
    this.#context.resetDecodeCache();
    this.#decode.reset();
  }

  #assertDecodeEnabled(): void {
    if (!this.#decodeEnabled) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        "CPU decode requires createContext({ decode: ... }) so retained activation storage is explicit.", {
          phase: 'execution', backend: this.backendName,
        });
    }
  }

  async #executeDecode(
    request: BackendResolvedExecutionRequest,
    prepared: PreparedProviderDecode,
  ): Promise<BackendExecutionSnapshot> {
    // A full seed always replaces retained intermediates. Compatibility was
    // proved before this mutation, so malformed/incompatible steps leave the
    // previous seed usable.
    let mutationStarted = false;
    try {
      const snapshot = await this.#executeResolved(
        request,
        prepared.engineOptions,
        prepared.operation === 'seed'
          ? () => {
              request.commitExecution();
              mutationStarted = true;
              this.#context.resetDecodeCache();
            }
          : () => {
              request.commitExecution();
              mutationStarted = true;
            },
      );
      return this.#withDecodeTelemetry(snapshot, this.#decode.commit(prepared));
    } catch (error) {
      if (mutationStarted) {
        this.#context.resetDecodeCache();
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
    const hostInputs = requireHostExecutionInputs(request, this.backendName);
    const staticFastPath = this.#context.snapshot.isStatic;
    const before = staticFastPath ? null : this.#context.telemetry();
    const result = await this.#context.executeResolved(
      hostInputs,
      request.plan,
      executionOptions,
      beforeDispatch,
    );
    const outputs = Object.freeze(request.outputDescriptors.map((descriptor) => {
      const output = result.get(descriptor.name);
      return Object.freeze({
        name: output.name,
        shape: output.shape,
        dtype: output.dtype,
        location: 'host' as const,
        ownership: 'transfer' as const,
        data: output.data,
      });
    }));
    // Keep the constant-only performance path allocation-equivalent to the
    // pre-redesign baseline. Dynamic contexts publish the detailed cache and
    // capacity counters needed by the performance protocol below.
    if (staticFastPath) return Object.freeze({ outputs, backendReport: null });
    const after = this.#context.telemetry();
    const cacheHit = after.planCacheHits > before!.planCacheHits ||
      (after.staticFastPath && before!.currentSignature === request.signature);
    return Object.freeze({
      outputs,
      backendReport: Object.freeze({
        shapeSignature: request.signature,
        specializationCacheHit: cacheHit,
        specializationCacheEntries: after.planCacheEntries,
        specializationCacheMetadataBytes: after.planCacheMetadataBytes,
        specializationCacheHits: after.planCacheHits,
        specializationCacheMisses: after.planCacheMisses,
        specializationCacheEvictions: after.planCacheEvictions,
        specializationCacheOversizeSkips: after.planCacheOversizeSkips,
        persistentWeightBytes: after.invariantWeightBytes,
        logicalActivationBytes: request.plan.logicalActivationBytes,
        activationCapacityBytes: after.activationCapacityBytes,
        activationCapacityHighWaterBytes: after.activationCapacityHighWaterBytes,
        activationGrowCount: after.activationGrowCount,
        grownTensorCount: after.grownTensorCount,
        staticFastPath: after.staticFastPath,
      }),
    });
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#decode.close();
    await this.#context.close();
  }
}

class CPUProviderCompiledModel implements BackendProviderCompiledModel {
  readonly backendName = 'cpu';
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly #snapshot: Model;
  readonly #resources: CPUResourceDomainProof;
  #closed = false;

  constructor(
    input: BackendLogicalCompileInput,
    deviceIdentity: BackendDeviceIdentity | null,
    resources: CPUResourceDomainProof,
  ) {
    this.#snapshot = input.snapshot;
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
        maximumResidentBytes: resources.maximumResidentBytes,
        resourceLimitBytes: resources.resourceLimitBytes,
        weightBytes: resources.weightBytes,
        maximumActivationArenaBytes: resources.maximumActivationArenaBytes,
        maximumPersistentActivationBytes: resources.maximumPersistentActivationBytes,
        maximumInputBytes: resources.maximumInputBytes,
        maximumResultBytes: resources.maximumResultBytes,
        maximumTypedScratchBytes: resources.maximumTypedScratchBytes,
        maximumDecodeResidentBytes: resources.maximumDecodeResidentBytes,
        decodeContextSupported: resources.decodeContextSupported,
        capacityLimitBytes: resources.capacityLimitBytes,
        residentLimitBytes: resources.residentLimitBytes,
      }),
    });
  }

  createContext(options?: BackendProviderContextOptions): BackendProviderExecutionContext {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled backend model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    const decodeEnabled = options?.decode !== undefined;
    if (decodeEnabled && !this.#resources.decodeContextSupported) {
      const capacityExceeded = this.#resources.maximumPersistentActivationBytes >
        this.#resources.capacityLimitBytes;
      const required = capacityExceeded
        ? this.#resources.maximumPersistentActivationBytes
        : this.#resources.maximumDecodeResidentBytes;
      const limit = capacityExceeded
        ? this.#resources.capacityLimitBytes
        : this.#resources.residentLimitBytes;
      const ceiling = capacityExceeded ? 'committed-capacity' : 'resident/transaction';
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `CPU retained decode resources require ${required} bytes, exceeding the ` +
        `${ceiling} limit ${limit}.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    return new CPUProviderExecutionContext(this.#snapshot, options, decodeEnabled);
  }

  async close(): Promise<void> {
    this.#closed = true;
  }
}

/** Main-profile CPU provider, split out so WASM-only composition stays CPU-free. */
export class CPUBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName = 'cpu';
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  readonly #source: CPUEngine;
  #closed = false;

  constructor(source: CPUEngine) {
    this.#source = assertBuiltInEngine(source, 'CPU backend provider source') as CPUEngine;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
      dynamicShapeDomain: 'full',
    });
    this.deviceIdentity = createBackendDeviceIdentity(
      (source as CPUEngine & { adapterInfo?: Readonly<Record<string, string>> | null }).adapterInfo,
    ) || Object.freeze({
      backend: 'cpu',
      device: 'host',
    });
  }

  async compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', "Backend provider 'cpu' is closed.", {
        phase: 'lifecycle', backend: 'cpu',
      });
    }
    if (!(input?.snapshot instanceof Model) ||
        input.shapeDomainProof !== input.snapshot.shapeDomainProof ||
        input.graphFingerprint !== input.snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'CPU provider compile requires the exact immutable logical compile view.', {
          phase: 'compilation', backend: 'cpu',
        });
    }
    if (options.operatorFallback === 'forbid' && this.capabilities.operatorFallback !== 'none') {
      throw new VolvoxAIError('OPERATOR_FALLBACK_FORBIDDEN',
        "Backend 'cpu' cannot attest strict operator routing.", {
          phase: 'compilation', backend: 'cpu',
        });
    }
    cpuRegistryPreflight(input);
    const resources = checkedCPUResourceDomain(input);
    return new CPUProviderCompiledModel(input, this.deviceIdentity, resources);
  }

  async close(): Promise<void> {
    if (this.#closed) return;
    this.#closed = true;
    this.#source.dispose();
  }
}
