import {
  createBoundExecutionGraphMaterializer,
  type BoundExecutionGraph,
  type BoundExecutionGraphMaterializer,
} from '../core/BoundExecutionGraph.js';
import type { BackendExecutionSnapshot } from '../core/ExecutionResult.js';
import {
  BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
  type BackendMemoryCaptureRequest,
  type BackendMemorySnapshot,
  type RuntimeMemoryOwnerRef,
} from '../core/MemoryCapture.js';
import { Model } from '../core/Model.js';
import type { ResolvedShapePlan } from '../core/ResolvedShapePlan.js';
import { resolveMinimumGraphShapes } from '../core/ResolvedShapePlan.js';
import { VolvoxAIError } from '../core/RuntimeErrors.js';
import { kernelRoute, operatorShapeContract } from '../generated/kernelRegistry.js';
import {
  MemoryBackingRelation,
  MemoryEvidenceSource,
  MemoryInventoryKind,
  MemoryMetric,
  MemoryOwnerKind,
  MemoryResourceRole,
  MemorySpace,
  MemoryTemporalCoverage,
  MemoryValueRelation,
} from '../generated/volvoxaiEnums.js';
import {
  runtimeDTypeBytes,
} from '../ops/shapeSystem.js';
import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import {
  VOLVOXAI_BACKEND_PROVIDER_VERSION,
  createBackendProviderBatchContract,
  createBackendProviderPreparedBatchRoute,
  createBackendDeviceIdentity,
  createBackendProviderCapabilities,
  requireHostExecutionInputs,
  type BackendDeviceIdentity,
  type BackendLogicalCompileInput,
  type BackendProvider,
  type BackendProviderBatchContract,
  type BackendProviderCapabilities,
  type BackendProviderCompilationEvidence,
  type BackendProviderCompiledModel,
  type BackendProviderCompileOptions,
  type BackendProviderContextOptions,
  type BackendProviderExecutionContext,
  type BackendResolvedExecutionRequest,
} from './BackendProvider.js';
import {
  InvariantResourceStore,
  type InvariantResourceLease,
} from './InvariantResources.js';
import {
  ProviderDecodeLifecycle,
  type PreparedProviderDecode,
  type ProviderDecodeTelemetry,
} from './ProviderDecodeLifecycle.js';
import {
  WasmEngine,
  WasmCompiledInvariantPrefix,
  type WasmArenaInspection,
} from './WasmEngine.js';


/**
 * Host-side invariant weight storage, shared by every context over one model
 * revision.
 *
 * `docs/adr-dynamic-shape-v1.md` assigns invariant packed weights to the
 * compiled model rather than to a context; `snapshot.copyWeightData(name)` per
 * context is the copy that assignment forbids. Sharing is safe because nothing
 * writes to these buffers during execution — a single context already reuses
 * one buffer across every execution and replan.
 *
 * The exact `WasmProviderCompiledModel` owns both this host map and one
 * immutable raw/packed prefix in the provider's linear-memory root. It is
 * never keyed by a logical Model shared across independent compilations.
 */

const WASM_SIGNED_ALLOCATOR_MAX_BYTES = 0x7ffffff0;
const WASM_RUNTIME_ABI_VERSION = 1;
const WASM_CONTEXT_RESOURCE_LIMIT_BYTES = WASM_SIGNED_ALLOCATOR_MAX_BYTES;
const WASM_MAX_RANK = 8;
const WASM_MAX_KERNEL_EXTENT = 0x7fffffff;
const DEFAULT_PLAN_CACHE_ENTRIES = 8;
const DEFAULT_PLAN_CACHE_METADATA_BYTES = 1024 * 1024;
const WASM_MAX_SCRATCH_BYTES = 64 * 1024 * 1024;
/* Keep this in lockstep with the runtime guard in WasmEngine. */
const WASM_MOE_SLOT_TABLE_MAX_BYTES = 64 * 1024 * 1024;
const WASM_MAX_GROWTH_HEADROOM_BYTES = 64 * 1024 * 1024;
const WASM_PAGE_BYTES = 64 * 1024;
const WASM32_MAX_MEMORY_BYTES = 0x100000000;
const UTF8_ENCODER = new TextEncoder();
const WASM_HOST_INVARIANT_WEIGHTS_KEY = 'wasm-host-invariant-weights/v1';
const WASM_LINEAR_INVARIANT_PREFIX_KEY = 'wasm-linear-invariant-prefix/v1';

class WasmCompiledHostWeights {
  readonly #weights = new Map<string, RuntimeTypedArray>();
  readonly ownedBytes: number;
  #closed = false;

  constructor(snapshot: Model) {
    let ownedBytes = 0;
    for (const name of snapshot.weightNames) {
      const value = snapshot.copyWeightData(name);
      this.#weights.set(name, value);
      ownedBytes += value.byteLength;
    }
    this.ownedBytes = ownedBytes;
  }

  get(name: string): RuntimeTypedArray {
    if (this.#closed) throw new Error('WASM compiled host weights are closed.');
    const value = this.#weights.get(name);
    if (value === undefined) {
      throw new Error(`WASM compiled host weight '${name}' is not owned by this revision.`);
    }
    return value;
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#weights.clear();
  }
}

type WasmCompiledInvariantResource =
  | WasmCompiledHostWeights
  | WasmCompiledInvariantPrefix;

interface WasmResourceDomainProof {
  readonly maximumTensorBytes: number;
  readonly staticPrefixBytes: number;
  readonly initialMemoryBytes: number;
  readonly maximumLinearResidentBytes: number;
  readonly maximumInvariantPrefixBytes: number;
  readonly maximumMutableArenaBytes: number;
  readonly maximumBankedMutableArenaBytes: number;
  readonly maximumPersistentMetadataBytes: number;
  readonly maximumSharedScratchBytes: number;
  readonly weightBytes: number;
  readonly maximumBankResidentBytes: number;
  readonly maximumBankStagingTransactionBytes: number;
  readonly maximumInputBytes: number;
  readonly maximumResultBytes: number;
  readonly maximumResidentBytes: number;
  readonly maximumDecodeResidentBytes: number;
  readonly decodeContextSupported: boolean;
  readonly residentLimitBytes: number;
  readonly resourceLimitBytes: number;
  readonly tensorCapacityBytes: Readonly<Record<string, number>>;
}

interface WasmRuntimeAbiProof {
  readonly staticPrefixBytes: bigint;
  readonly initialMemoryBytes: bigint;
}

const F32_PACKABLE_OPERATORS = new Set(['MatMul', 'Linear', 'Gemm']);
const W8A32_PACKABLE_OPERATORS = new Set(['MatMul', 'Linear', 'Gemm']);
const Q8_PACKABLE_OPERATORS = new Set([
  ...W8A32_PACKABLE_OPERATORS,
  'QLinear', 'QMatMul', 'QGemm', 'QConv2D',
]);
/* Operators audited as making no per-variant metadata/scratch allocation in
 * WasmEngine._compilePortableMetadata. Keep this explicit so a newly routed
 * operator fails closed until its resource behavior is classified. */
const WASM_VARIANT_ALLOCATION_FREE_OPERATORS = new Set([
  'MatMul', 'Linear', 'BatchMatMul', 'QBatchMatMul', 'Gemm', 'MoERouter',
  'SDPA', 'QSDPA', 'CrossSDPA', 'CrossAttention', 'LayerNorm', 'QLayerNorm',
  'GroupNorm', 'QGroupNorm', 'RMSNorm', 'BatchNorm2D', 'Conv2D', 'Conv1D',
  'ConvTranspose2D', 'MaxPool2D', 'AveragePool2D', 'GlobalAveragePool',
  'Resize', 'Interpolate1D', 'ResizeNearest2D', 'UpsampleNearest2D', 'ReLU',
  'LeakyReLU', 'PReLU', 'GELU', 'QGELU', 'SiLU', 'QSiLU', 'Sigmoid',
  'HardSwish', 'HardSigmoid', 'Tanh', 'Sin', 'Cos', 'Clip', 'QAdd',
  'Softmax', 'LogSoftmax', 'ReduceSum', 'ReduceMean', 'QMaskedMean', 'ArgMax',
  'QArgMax', 'Not', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Concat',
  'Split', 'Pad', 'Gather', 'Identity', 'Concat2', 'QuantizeLinear',
  'DequantizeLinear', 'RequantizeLinear', 'Cast', 'Embedding', 'RoPE',
  'NonMaxSuppression', 'SpatialSoftargmaxY', 'MeanHeight', 'ProfileX',
  'ProfileY', 'Dropout',
]);

interface CachedWasmVariant {
  readonly plan: ResolvedShapePlan;
  readonly bound: BoundExecutionGraph;
  readonly metadataBytes: number;
}

function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length || (left < right ? -1 : 1);
}

function sortedNames(value: object): string[] {
  return Object.getOwnPropertyNames(value).sort(compareCanonicalNames);
}

function checkedBigIntNumber(value: bigint, label: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED', `WASM ${label} exceeds exact JavaScript arithmetic.`, {
      phase: 'compilation', backend: 'wasm',
    });
  }
  return Number(value);
}

function align16(value: bigint): bigint {
  return ((value + 15n) / 16n) * 16n;
}

function wasmExportedGlobalValue(value: unknown): unknown {
  if (value === null || (typeof value !== 'object' && typeof value !== 'function')) {
    return undefined;
  }
  /* instanceof is realm-sensitive. The WebAssembly toStringTag and its value
   * getter are stable across browser realms and Node vm contexts. */
  try {
    if (Object.prototype.toString.call(value) !== '[object WebAssembly.Global]') {
      return undefined;
    }
    return Reflect.get(value, 'value');
  } catch {
    return undefined;
  }
}

function checkedWasmRuntimeAbi(source: WasmEngine): WasmRuntimeAbiProof {
  const api = source.api;
  let abiVersion: unknown;
  try {
    abiVersion = typeof api.wasm_runtime_abi_version === 'function'
      ? api.wasm_runtime_abi_version() : undefined;
  } catch {
    abiVersion = undefined;
  }
  if (abiVersion !== WASM_RUNTIME_ABI_VERSION ||
      typeof api.heap_mark !== 'function' || typeof api.heap_rewind !== 'function') {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      `WASM sidecar does not expose runtime ABI ${WASM_RUNTIME_ABI_VERSION}; ` +
      'rebuild the sidecar.', {
        phase: 'compilation', backend: 'wasm',
      });
  }

  const heapBase = wasmExportedGlobalValue(api.__heap_base);
  if (typeof heapBase !== 'number' || !Number.isSafeInteger(heapBase) || heapBase <= 0 ||
      heapBase > WASM_SIGNED_ALLOCATOR_MAX_BYTES || heapBase % 16 !== 0) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      'WASM sidecar exposes an invalid positive aligned __heap_base static-prefix address.', {
        phase: 'compilation', backend: 'wasm',
      });
  }

  let mark: unknown;
  try {
    mark = api.heap_mark();
  } catch {
    mark = undefined;
  }
  if (typeof mark !== 'number' || !Number.isSafeInteger(mark) || mark !== heapBase) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      'WASM sidecar allocator does not start at its exported __heap_base address.', {
        phase: 'compilation', backend: 'wasm',
      });
  }

  let initialMemoryBytes: unknown;
  try {
    initialMemoryBytes = Reflect.get(Reflect.get(api.memory, 'buffer'), 'byteLength');
  } catch {
    initialMemoryBytes = undefined;
  }
  if (typeof initialMemoryBytes !== 'number' || !Number.isSafeInteger(initialMemoryBytes) ||
      initialMemoryBytes <= 0 || initialMemoryBytes > WASM32_MAX_MEMORY_BYTES ||
      initialMemoryBytes % WASM_PAGE_BYTES !== 0 || initialMemoryBytes < heapBase) {
    throw new VolvoxAIError('ABI_UNSUPPORTED',
      'WASM sidecar exposes an invalid initial linear-memory extent.', {
        phase: 'compilation', backend: 'wasm',
      });
  }
  return Object.freeze({
    staticPrefixBytes: BigInt(heapBase),
    initialMemoryBytes: BigInt(initialMemoryBytes),
  });
}

function wasmRegistryPreflight(input: BackendLogicalCompileInput): void {
  if (input.shapeDomainProof.nodes.length !== input.graph.nodes.length) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      "Backend 'wasm' received incomplete canonical domain evidence.", {
        phase: 'compilation', backend: 'wasm',
      });
  }
  for (let index = 0; index < input.graph.nodes.length; index++) {
    const node = input.graph.nodes[index];
    const nodeProof = input.shapeDomainProof.nodes[index];
    const contract = operatorShapeContract(node.opType);
    const route = kernelRoute('wasm', node.opType);
    if (!contract || contract.classification !== 'canonical' || !route ||
        nodeProof?.id !== node.id || nodeProof.opType !== node.opType ||
        nodeProof.shapeFunctionId !== contract.shapeFunctionId) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `Backend 'wasm' cannot bind canonical route evidence for operator ` +
        `'${node.opType}' at node '${node.id}'.`, {
          phase: 'compilation', backend: 'wasm', node: node.id,
        });
    }
  }
}

function possiblePackedPartitions(shape: readonly number[]): readonly (readonly [number, number])[] {
  if (shape.length === 2) {
    return Object.freeze([
      Object.freeze([shape[0], shape[1]] as const),
      Object.freeze([shape[1], shape[0]] as const),
    ]);
  }
  if (shape.length === 4) {
    return Object.freeze(shape.map((dOut, axis) => Object.freeze([
      shape.reduce((product, dimension, index) => index === axis ? product : product * dimension, 1),
      dOut,
    ] as const)));
  }
  return Object.freeze([]);
}

function checkedInvariantPackUpperBytes(
  input: BackendLogicalCompileInput,
  source: WasmEngine,
): Readonly<{ total: bigint; banked: bigint }> {
  let packedBytes = 0n;
  let bankedPackedBytes = 0n;
  for (const node of input.graph.nodes) {
    const weightName = node.inputs.weight;
    const weight = weightName === undefined ? undefined : input.graph.weights[weightName];
    if (!weight) continue;
    const partitions = possiblePackedPartitions(weight.shape);
    let maximum = 0;
    let unrepresentable = false;
    if (weight.dtype === 'float32' && F32_PACKABLE_OPERATORS.has(node.opType) &&
        typeof source.api.gemm_f32_packed_elements === 'function') {
      for (const [dIn, dOut] of partitions) {
        const elements = Number(source.api.gemm_f32_packed_elements(dIn, dOut));
        const bytes = elements * Float32Array.BYTES_PER_ELEMENT;
        if (Number.isSafeInteger(elements) && elements > 0 &&
            Number.isSafeInteger(bytes) && bytes <= WASM_SIGNED_ALLOCATOR_MAX_BYTES) {
          maximum = Math.max(maximum, bytes);
        } else {
          unrepresentable = true;
        }
      }
    } else if ((weight.dtype === 'int8' || weight.dtype === 'uint8') &&
        Q8_PACKABLE_OPERATORS.has(node.opType)) {
      for (const [dIn, dOut] of partitions) {
        const canonicalOnly = W8A32_PACKABLE_OPERATORS.has(node.opType) ||
          (node.opType !== 'QConv2D' && dOut < 8);
        const sizeKernel = canonicalOnly
          ? source.api.packed_q8_weight_canonical_size
          : source.api.packed_q8_weight_size;
        const bytes = typeof sizeKernel === 'function'
          ? Number(sizeKernel(dIn, dOut)) : 0;
        if (Number.isSafeInteger(bytes) && bytes > 0 &&
            bytes <= WASM_SIGNED_ALLOCATOR_MAX_BYTES) {
          maximum = Math.max(maximum, bytes);
        } else {
          unrepresentable = true;
        }
      }
    }
    if (unrepresentable) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM cannot bound invariant packing for weight '${weight.name}' at node '${node.id}'.`, {
          phase: 'compilation', backend: 'wasm', node: node.id,
        });
    }
    /* Charge every potentially packed node independently. The engine may
     * deduplicate identical pack keys, but relying on that would under-prove a
     * graph that legally references one weight through multiple layouts. */
    const charged = align16(BigInt(maximum));
    packedBytes += charged;
    if (weight.bank !== null) bankedPackedBytes += charged;
  }
  return Object.freeze({ total: packedBytes, banked: bankedPackedBytes });
}

interface WasmExtentBounds {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

type WasmLogicalNode = BackendLogicalCompileInput['graph']['nodes'][number];

function failWasmNodeResource(node: WasmLogicalNode, reason: string): never {
  throw new VolvoxAIError('BACKEND_UNSUPPORTED',
    `WASM cannot bound resources for ${node.opType} node '${node.id}': ${reason}.`, {
      phase: 'compilation', backend: 'wasm', node: node.id,
    });
}

function checkedWasmExtentBounds(
  input: BackendLogicalCompileInput,
  dimension: number | string,
  label: string,
  visiting = new Set<string>(),
): WasmExtentBounds {
  if (typeof dimension === 'number') {
    if (!Number.isSafeInteger(dimension) || dimension <= 0 ||
        dimension > WASM_MAX_KERNEL_EXTENT) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM cannot represent the bounded extent of ${label}.`, {
          phase: 'compilation', backend: 'wasm',
        });
    }
    const value = BigInt(dimension);
    return Object.freeze({ minimum: value, maximum: value });
  }
  if (visiting.has(dimension)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WASM found a cyclic symbolic extent while proving ${label}.`, {
        phase: 'compilation', backend: 'wasm',
      });
  }
  visiting.add(dimension);
  let bounds: WasmExtentBounds | null = null;
  const affine = input.shapeDomainProof.affineSymbolRelations[dimension];
  if (affine !== undefined) {
    const source = checkedWasmExtentBounds(input, affine.source, label, visiting);
    const offset = BigInt(affine.offset);
    bounds = Object.freeze({
      minimum: source.minimum + offset,
      maximum: source.maximum + offset,
    });
  } else {
    const relation = input.shapeDomainProof.symbolRelations[dimension];
    if (relation !== undefined && relation !== dimension) {
      bounds = checkedWasmExtentBounds(input, relation, label, visiting);
    } else {
      const constraint = input.graph.dimensions[dimension];
      if (constraint !== undefined && Number.isSafeInteger(constraint.min) &&
          Number.isSafeInteger(constraint.max) &&
          Number.isSafeInteger(constraint.multiple_of) && constraint.min > 0 &&
          constraint.max >= constraint.min && constraint.multiple_of > 0) {
        const multiple = BigInt(constraint.multiple_of);
        const rawMinimum = BigInt(constraint.min);
        const rawMaximum = BigInt(constraint.max);
        const minimum = ((rawMinimum + multiple - 1n) / multiple) * multiple;
        const maximum = (rawMaximum / multiple) * multiple;
        bounds = Object.freeze({ minimum, maximum });
      }
    }
  }
  visiting.delete(dimension);
  if (!bounds || bounds.minimum <= 0n || bounds.maximum < bounds.minimum ||
      bounds.maximum > BigInt(WASM_MAX_KERNEL_EXTENT)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WASM cannot represent the bounded extent of ${label}.`, {
        phase: 'compilation', backend: 'wasm',
      });
  }
  return bounds;
}

function requiredNodeInputName(
  node: WasmLogicalNode,
  ports: readonly string[],
  role: string,
): string {
  for (const port of ports) {
    const name = node.inputs[port];
    if (name !== undefined) return name;
  }
  return failWasmNodeResource(node, `missing ${role} tensor`);
}

function requiredNodeOutputName(node: WasmLogicalNode): string {
  const preferred = node.outputs.out;
  if (preferred !== undefined) return preferred.tensor;
  const first = sortedNames(node.outputs)[0];
  if (first !== undefined) return node.outputs[first].tensor;
  return failWasmNodeResource(node, 'missing output tensor');
}

function requiredTensorShape(
  input: BackendLogicalCompileInput,
  node: WasmLogicalNode,
  name: string,
  role: string,
): readonly (number | string)[] {
  const tensor = input.graph.tensors[name];
  if (!tensor) return failWasmNodeResource(node, `missing ${role} descriptor '${name}'`);
  return tensor.shape;
}

function checkedVariantAllocationBytes(
  node: WasmLogicalNode,
  bytes: bigint,
  label: string,
): bigint {
  if (bytes <= 0n || bytes > BigInt(WASM_SIGNED_ALLOCATOR_MAX_BYTES)) {
    return failWasmNodeResource(node, `${label} allocation exceeds the signed allocator limit`);
  }
  const aligned = align16(bytes);
  if (aligned > BigInt(WASM_SIGNED_ALLOCATOR_MAX_BYTES)) {
    return failWasmNodeResource(node, `${label} alignment exceeds the signed allocator limit`);
  }
  return aligned;
}

function checkedMetadataWords(
  node: WasmLogicalNode,
  words: number | bigint,
  label: string,
): bigint {
  return checkedVariantAllocationBytes(node, BigInt(words) * 4n, label);
}

function checkedShapeProductBounds(
  input: BackendLogicalCompileInput,
  node: WasmLogicalNode,
  dimensions: readonly (readonly [number | string, string])[],
): WasmExtentBounds {
  let minimum = 1n;
  let maximum = 1n;
  for (const [dimension, label] of dimensions) {
    const bounds = checkedWasmExtentBounds(input, dimension, `${node.opType} ${label}`);
    minimum *= bounds.minimum;
    maximum *= bounds.maximum;
  }
  return Object.freeze({ minimum, maximum });
}

function checkedWasmPersistentMetadataBytes(input: BackendLogicalCompileInput): bigint {
  let total = 0n;
  for (const node of input.graph.nodes) {
    let nodeBytes = 0n;
    if (node.opType === 'MoELinear') {
      let maximumDomain = 0;
      for (const name of Object.values(node.inputs)) {
        const weight = input.graph.weights[name];
        if (weight?.bank !== null && weight?.bank !== undefined) {
          maximumDomain = Math.max(maximumDomain, weight.shape[0]);
        }
      }
      if (maximumDomain > 0) {
        const rawBytes = BigInt(maximumDomain) * 4n;
        if (rawBytes > BigInt(WASM_MOE_SLOT_TABLE_MAX_BYTES)) {
          failWasmNodeResource(node,
            `resident slot-table domain exceeds ${WASM_MOE_SLOT_TABLE_MAX_BYTES} bytes`);
        }
        nodeBytes += checkedVariantAllocationBytes(node, rawBytes, 'resident slot table');
      }
    } else if (node.opType === 'QLinear' || node.opType === 'QMatMul' ||
               node.opType === 'QGemm') {
      const weightName = requiredNodeInputName(node, ['weight'], 'weight');
      const weightShape = requiredTensorShape(input, node, weightName, 'weight');
      if (weightShape.length !== 2 || typeof weightShape[0] !== 'number') {
        failWasmNodeResource(node, 'QLinear weight shape is not fixed rank two');
      }
      const allocation = checkedMetadataWords(node, weightShape[0], 'weight scales');
      nodeBytes += allocation * 2n;
    } else if (node.opType === 'QEmbedding') {
      const weightName = requiredNodeInputName(node, ['weight'], 'weight');
      const weightShape = requiredTensorShape(input, node, weightName, 'weight');
      if (weightShape.length !== 2 || typeof weightShape[0] !== 'number') {
        failWasmNodeResource(node, 'QEmbedding weight shape is not fixed rank two');
      }
      const allocation = checkedMetadataWords(node, weightShape[0], 'weight scales');
      nodeBytes += allocation * 2n;
    } else if (node.opType === 'QConv2D') {
      const weightName = requiredNodeInputName(node, ['weight'], 'weight');
      const weightShape = requiredTensorShape(input, node, weightName, 'weight');
      if (weightShape.length !== 4 || typeof weightShape[0] !== 'number') {
        failWasmNodeResource(node, 'QConv2D weight shape is not fixed rank four');
      }
      const allocation = checkedMetadataWords(node, weightShape[0], 'weight scales');
      nodeBytes += allocation * 2n;
    } else if (node.opType === 'Equal' || node.opType === 'GreaterOrEqual') {
      const a = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['a'], 'a'), 'a');
      const b = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['b'], 'b'), 'b');
      const output = requiredTensorShape(input, node, requiredNodeOutputName(node), 'output');
      nodeBytes += checkedMetadataWords(node,
        Math.max(1, a.length) + Math.max(1, b.length) + Math.max(1, output.length),
        'comparison shapes');
    } else if (node.opType === 'Add' || node.opType === 'Mul' ||
               node.opType === 'Sub' || node.opType === 'Div') {
      const a = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['a'], 'a'), 'a');
      const b = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['b'], 'b'), 'b');
      const output = requiredTensorShape(input, node, requiredNodeOutputName(node), 'output');
      const allocation = checkedMetadataWords(node, a.length + b.length + output.length,
        'broadcast shapes');
      /* Add/Mul may retain a second descriptor for incremental row execution. */
      nodeBytes += allocation * ((node.opType === 'Add' || node.opType === 'Mul') ? 2n : 1n);
    } else if (node.opType === 'Expand' || node.opType === 'Broadcast') {
      const source = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['input', 'x', 'data'], 'input'), 'input');
      const output = requiredTensorShape(input, node, requiredNodeOutputName(node), 'output');
      const allocation = checkedMetadataWords(node, source.length + output.length,
        'expand shapes');
      /* Expand may retain a second descriptor for incremental row execution. */
      nodeBytes += allocation * (node.opType === 'Expand' ? 2n : 1n);
    } else if (node.opType === 'Transpose') {
      const source = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['input', 'x', 'data'], 'input'), 'input');
      nodeBytes += checkedMetadataWords(node, source.length * 2, 'transpose shape and permutation');
    } else if (node.opType === 'Where' || node.opType === 'Mask') {
      const condition = requiredTensorShape(input, node,
        requiredNodeInputName(node, node.opType === 'Where' ? ['condition'] : ['mask'],
          'condition'), 'condition');
      const a = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['a'], 'a'), 'a');
      const b = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['b'], 'b'), 'b');
      const output = requiredTensorShape(input, node, requiredNodeOutputName(node), 'output');
      nodeBytes += checkedMetadataWords(node,
        Math.max(1, condition.length) + Math.max(1, a.length) +
          Math.max(1, b.length) + Math.max(1, output.length),
        'where shapes');
    } else if (node.opType === 'GatherElements') {
      const source = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['input', 'data'], 'input'), 'input');
      nodeBytes += checkedMetadataWords(node, source.length * 2, 'gather-element shapes');
    } else if (node.opType === 'Slice') {
      const source = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['input', 'x', 'data'], 'input'), 'input');
      nodeBytes += checkedMetadataWords(node, source.length * 4, 'slice shapes and bounds');
    } else if (node.opType === 'SSMScan' || node.opType === 'SelectiveScan') {
      const source = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['input', 'u'], 'input'), 'input');
      const a = requiredTensorShape(input, node,
        requiredNodeInputName(node, ['A', 'a'], 'A'), 'A');
      if ((source.length !== 2 && source.length !== 3) || a.length !== 2) {
        failWasmNodeResource(node, 'selective-scan tensors do not have canonical ranks');
      }
      const batch = source.length === 2 ? 1 : source[0];
      const state = checkedShapeProductBounds(input, node, [
        [batch, 'batch'],
        [source[source.length - 1], 'channels'],
        [a[1], 'state width'],
      ]);
      nodeBytes += checkedVariantAllocationBytes(node, state.maximum * 4n,
        'selective-scan state');
    } else if (!WASM_VARIANT_ALLOCATION_FREE_OPERATORS.has(node.opType)) {
      failWasmNodeResource(node, 'operator has no audited variant-allocation proof');
    }
    total += nodeBytes;
  }
  return total;
}

function checkedWasmSharedScratchBytes(
  input: BackendLogicalCompileInput,
  source: WasmEngine,
): bigint {
  if (typeof source.api.qconv2d_im2col_i8u8 !== 'function' ||
      typeof source.api.qlinear_i8u8_packed !== 'function' ||
      typeof source.api.packed_q8_weight_size !== 'function' ||
      typeof source.api.pack_q8_weight !== 'function') return 0n;
  let maximum = 0n;
  let maximumNode: WasmLogicalNode | null = null;
  for (const node of input.graph.nodes) {
    if (node.opType !== 'QConv2D' || (node.params.groups ?? 1) !== 1 ||
        (node.params.relu ?? 0) !== 0 || node.inputs.bias === undefined) continue;
    const inputName = requiredNodeInputName(node, ['input', 'x'], 'input');
    const weightName = requiredNodeInputName(node, ['weight'], 'weight');
    const outputName = requiredNodeOutputName(node);
    const inputShape = requiredTensorShape(input, node, inputName, 'input');
    const weight = input.graph.weights[weightName];
    const outputShape = requiredTensorShape(input, node, outputName, 'output');
    if (!weight || inputShape.length !== 4 || weight.shape.length !== 4 ||
        outputShape.length !== 4 || weight.shape[0] < 8) continue;
    const bounds = checkedShapeProductBounds(input, node, [
      [inputShape[0], 'batch'],
      [outputShape[1], 'output height'],
      [outputShape[2], 'output width'],
      [weight.shape[1], 'kernel height'],
      [weight.shape[2], 'kernel width'],
      [inputShape[3], 'input channels'],
    ]);
    const ceiling = BigInt(WASM_MAX_SCRATCH_BYTES);
    const candidate = bounds.maximum > ceiling ? ceiling : bounds.maximum;
    if (bounds.minimum <= ceiling && candidate > maximum) {
      maximum = candidate;
      maximumNode = node;
    }
  }
  return maximumNode === null ? 0n : checkedVariantAllocationBytes(
    maximumNode, maximum, 'shared QConv2D im2col scratch');
}

function checkedWasmResourceDomain(
  input: BackendLogicalCompileInput,
  source: WasmEngine,
  abi: WasmRuntimeAbiProof,
): WasmResourceDomainProof {
  const { staticPrefixBytes, initialMemoryBytes } = abi;
  let maximumTensorBytes = 0n;
  let activationCapacityBytes = 0n;
  let linearWeightBytes = 0n;
  let linearBankWeightBytes = 0n;
  let weightBytes = 0n;
  let maximumBankResidentBytes = 0n;
  let maximumBankStagingTransactionBytes = 0n;
  let maximumInputBytes = 0n;
  let maximumResultBytes = 0n;
  const outputNames = new Set(input.outputNames);
  const capacities: Record<string, number> = {};
  for (const name of sortedNames(input.graph.tensors)) {
    const descriptor = input.graph.tensors[name];
    if (descriptor.shape.length > WASM_MAX_RANK) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM tensor '${name}' rank ${descriptor.shape.length} exceeds ${WASM_MAX_RANK}.`, {
          phase: 'compilation', backend: 'wasm',
        });
    }
    let elements = 1n;
    for (let axis = 0; axis < descriptor.shape.length; axis++) {
      const bounds = checkedWasmExtentBounds(
        input, descriptor.shape[axis], `tensor '${name}' axis ${axis}`,
      );
      elements *= bounds.maximum;
      if (elements > BigInt(WASM_MAX_KERNEL_EXTENT)) {
        throw new VolvoxAIError('BACKEND_UNSUPPORTED',
          `WASM tensor '${name}' can exceed signed kernel element limits.`, {
            phase: 'compilation', backend: 'wasm',
          });
      }
    }
    const bytes = elements * BigInt(runtimeDTypeBytes(descriptor.dtype));
    if (bytes <= 0n || bytes > BigInt(WASM_SIGNED_ALLOCATOR_MAX_BYTES)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM tensor '${name}' exceeds the signed allocator byte limit.`, {
          phase: 'compilation', backend: 'wasm',
        });
    }
    capacities[name] = Number(bytes);
    maximumTensorBytes = bytes > maximumTensorBytes ? bytes : maximumTensorBytes;
    if (descriptor.kind === 'weight') {
      weightBytes += bytes;
      linearWeightBytes += align16(bytes);
      if (descriptor.bank !== null) {
        linearBankWeightBytes += align16(bytes);
        /* stageWeights visits weights in this same canonical name order. At
         * the transient peak for this bank, every earlier bank's resident
         * slice and both full copies made by stageWeight are simultaneously
         * live. Selecting every slot of every bank is a legal context. */
        const transaction = maximumBankResidentBytes + bytes * 2n;
        if (transaction > maximumBankStagingTransactionBytes) {
          maximumBankStagingTransactionBytes = transaction;
        }
        maximumBankResidentBytes += bytes;
      }
    } else {
      activationCapacityBytes += align16(bytes);
      if (descriptor.kind === 'input') maximumInputBytes += bytes;
    }
    if (outputNames.has(name)) maximumResultBytes += bytes;
  }
  if (maximumBankResidentBytes > maximumBankStagingTransactionBytes) {
    maximumBankStagingTransactionBytes = maximumBankResidentBytes;
  }

  /* Charge the linker-defined prefix once, then every persistent arena class.
   * Shape variants replace one another after a rewind, so their node metadata
   * is summed once at its per-node domain maximum and QConv's shared im2col is
   * charged once at the maximum requested by any eligible node. The initial
   * linear memory already contains the prefix and any arena bytes beneath its
   * current extent; max(), rather than addition, prevents double-counting it. */
  const invariantPackBytes = checkedInvariantPackUpperBytes(input, source);
  const maximumPersistentMetadataBytes = checkedWasmPersistentMetadataBytes(input);
  const maximumSharedScratchBytes = checkedWasmSharedScratchBytes(input, source);
  const maximumInvariantPrefixBytes = linearWeightBytes + invariantPackBytes.total;
  const maximumMutableArenaBytes = activationCapacityBytes +
    maximumPersistentMetadataBytes + maximumSharedScratchBytes;
  /* A partially resident bank is a context selection rather than a compiled
   * invariant. Bound it as a private mutable suffix; full residency borrows
   * the already materialized full-bank bytes/panels from the prefix. */
  const maximumBankedMutableArenaBytes = maximumMutableArenaBytes +
    linearBankWeightBytes + invariantPackBytes.banked;
  const allocatorResident = staticPrefixBytes + maximumInvariantPrefixBytes +
    maximumBankedMutableArenaBytes + BigInt(WASM_MAX_GROWTH_HEADROOM_BYTES);
  const wasmResident = initialMemoryBytes > allocatorResident
    ? initialMemoryBytes : allocatorResident;
  /* The Model and compiled provider owner each retain one complete raw weight
   * revision. A context with explicit bank residency additionally retains its
   * selected slices, whose domain maximum is every slot of every bank. WASM
   * output views are cloned once into result-owned host storage. Compare that
   * steady execution phase with construction's double-copy bank transaction. */
  const residentHostWeights = weightBytes * 2n;
  const hostPlanCache = BigInt(DEFAULT_PLAN_CACHE_METADATA_BYTES);
  const steadyExecution = wasmResident + maximumBankResidentBytes + maximumResultBytes;
  const bankStagingTransaction = initialMemoryBytes + maximumBankStagingTransactionBytes;
  const ordinaryPhase = steadyExecution > bankStagingTransaction
    ? steadyExecution : bankStagingTransaction;
  const maximumResident = hostPlanCache + residentHostWeights + ordinaryPhase;
  if (maximumResident > BigInt(WASM_CONTEXT_RESOURCE_LIMIT_BYTES)) {
    throw new VolvoxAIError('BACKEND_UNSUPPORTED',
      `WASM bounded-domain resources require ${maximumResident} bytes, exceeding ` +
      `${WASM_CONTEXT_RESOURCE_LIMIT_BYTES}.`, {
        phase: 'compilation', backend: 'wasm',
      });
  }
  /* Decode replacement/reseed retains the previous complete public-input
   * snapshot while cloning its replacement after result staging. Caller-owned
   * input buffers and older caller-retained results remain outside this
   * per-context proof, as they do for ordinary execution. */
  const decodeExecution = steadyExecution + maximumInputBytes * 2n;
  const decodePhase = decodeExecution > bankStagingTransaction
    ? decodeExecution : bankStagingTransaction;
  const maximumDecodeResident = hostPlanCache + residentHostWeights + decodePhase;
  return Object.freeze({
    maximumTensorBytes: checkedBigIntNumber(maximumTensorBytes, 'maximum tensor bytes'),
    staticPrefixBytes: checkedBigIntNumber(staticPrefixBytes, 'static prefix bytes'),
    initialMemoryBytes: checkedBigIntNumber(initialMemoryBytes, 'initial memory bytes'),
    maximumLinearResidentBytes: checkedBigIntNumber(
      wasmResident, 'maximum linear resident bytes'),
    maximumInvariantPrefixBytes: checkedBigIntNumber(
      maximumInvariantPrefixBytes, 'maximum compiled invariant prefix bytes'),
    maximumMutableArenaBytes: checkedBigIntNumber(
      maximumMutableArenaBytes, 'maximum context mutable arena bytes'),
    maximumBankedMutableArenaBytes: checkedBigIntNumber(
      maximumBankedMutableArenaBytes, 'maximum banked context mutable arena bytes'),
    maximumPersistentMetadataBytes: checkedBigIntNumber(
      maximumPersistentMetadataBytes, 'maximum persistent metadata bytes'),
    maximumSharedScratchBytes: checkedBigIntNumber(
      maximumSharedScratchBytes, 'maximum shared scratch bytes'),
    weightBytes: checkedBigIntNumber(weightBytes, 'raw weight bytes'),
    maximumBankResidentBytes: checkedBigIntNumber(
      maximumBankResidentBytes, 'maximum bank resident bytes'),
    maximumBankStagingTransactionBytes: checkedBigIntNumber(
      maximumBankStagingTransactionBytes, 'maximum bank staging transaction bytes'),
    maximumInputBytes: checkedBigIntNumber(maximumInputBytes, 'maximum input bytes'),
    maximumResultBytes: checkedBigIntNumber(maximumResultBytes, 'maximum result bytes'),
    maximumResidentBytes: checkedBigIntNumber(maximumResident, 'maximum resident bytes'),
    maximumDecodeResidentBytes: checkedBigIntNumber(
      maximumDecodeResident, 'maximum decode resident bytes'),
    decodeContextSupported: maximumDecodeResident <= BigInt(WASM_CONTEXT_RESOURCE_LIMIT_BYTES),
    residentLimitBytes: WASM_CONTEXT_RESOURCE_LIMIT_BYTES,
    resourceLimitBytes: WASM_CONTEXT_RESOURCE_LIMIT_BYTES,
    tensorCapacityBytes: Object.freeze(capacities),
  });
}

function runtimeStorageDType(value: unknown): RuntimeDType | null {
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

function planMetadataBytes(plan: ResolvedShapePlan): number {
  let bytes = 128 + UTF8_ENCODER.encode(plan.signature).byteLength;
  for (const name of sortedNames(plan.tensors)) {
    const descriptor = plan.tensors[name];
    bytes += 64 + UTF8_ENCODER.encode(name).byteLength + descriptor.shape.length * 8;
  }
  for (const node of plan.nodes) {
    bytes += 64 + UTF8_ENCODER.encode(String(node.id)).byteLength +
      UTF8_ENCODER.encode(node.opType).byteLength;
  }
  if (!Number.isSafeInteger(bytes)) return Number.MAX_SAFE_INTEGER;
  return bytes;
}

class WasmProviderExecutionContext implements BackendProviderExecutionContext {
  readonly backendName = 'wasm';
  readonly #snapshot: Model;
  #engine: WasmEngine | null;
  readonly #tensorCapacityBytes: Readonly<Record<string, number>>;
  readonly #invariantSchedule: Readonly<object>;
  /** Borrowed from the compiled invariant owner; this context cannot mutate its index. */
  readonly #invariantWeights: WasmCompiledHostWeights;
  readonly #boundMaterializer: BoundExecutionGraphMaterializer;
  readonly #decode: ProviderDecodeLifecycle;
  readonly #decodeEnabled: boolean;
  #cache = new Map<string, CachedWasmVariant>();
  #cacheBytes = 0;
  #cacheHits = 0;
  #cacheMisses = 0;
  #cacheEvictions = 0;
  #preparedSchedule: Readonly<object> | null = null;
  #preparedScheduleBuildCount = 0;
  #closed = false;

  constructor(
    snapshot: Model,
    engine: WasmEngine,
    tensorCapacityBytes: Readonly<Record<string, number>>,
    invariantSchedule: Readonly<object>,
    invariantWeights: WasmCompiledHostWeights,
    options?: BackendProviderContextOptions,
  ) {
    this.#snapshot = snapshot;
    this.#engine = engine;
    this.#tensorCapacityBytes = tensorCapacityBytes;
    this.#invariantSchedule = invariantSchedule;
    this.#decodeEnabled = options?.decode !== undefined;
    this.#decode = new ProviderDecodeLifecycle(snapshot, this.backendName, {
      incrementalExecution: true,
      incrementalRows: true,
      /* The WASM row executor addresses a row by offset and width, so it reads
       * the sequence-major spellings as well as the batch-major one. WebGPU
       * has no row candidate for them, which is why this is per provider. */
      sequenceMajorRows: true,
      /* Shares `quantizedRowExecution` with the CPU provider, and its staging
       * buffers come from the WASM heap so the kernels can take pointers into
       * them. Sequence-major spellings carry one lane only, which the row
       * attestation refuses to combine with a declared capacity above one. */
      batchedRows: true,
    }, options?.decode);
    this.#invariantWeights = invariantWeights;
    this.#boundMaterializer = createBoundExecutionGraphMaterializer(snapshot, {
      invariantWeightStorageFactory: ({ name }) => this.#invariantWeights.get(name),
    });
  }

  async execute(request: BackendResolvedExecutionRequest): Promise<BackendExecutionSnapshot> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    const engine = this.#engine;
    if (!engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution arena is released.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (request.operation !== 'execute') {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM ordinary execute does not implement '${request.operation}'.`, {
          phase: 'execution', backend: this.backendName,
        });
    }
    const invalidateSeed = this.#decode.seeded;
    return this.#executeResolved(request, {}, () => {
      request.commitExecution();
      if (invalidateSeed) this.#decode.invalidate('ordinary-execution');
      if (invalidateSeed) engine.resetDecodeCache();
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
    if (this.#closed || !this.#engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    this.#assertDecodeEnabled();
    this.#engine.resetDecodeCache();
    this.#decode.reset();
  }

  captureMemorySnapshot(request: BackendMemoryCaptureRequest): BackendMemorySnapshot {
    const engine = this.#engine;
    if (this.#closed || !engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (request.protocol !== BACKEND_MEMORY_SNAPSHOT_PROTOCOL) {
      throw new VolvoxAIError('ABI_UNSUPPORTED', 'Unsupported backend memory snapshot protocol.', {
        phase: 'execution', backend: this.backendName,
      });
    }
    const root = engine.inspectMemoryRoot();
    const arena = engine.inspectArena();
    const invariant = arena.sharedInvariant;
    if (invariant === null || invariant.backingRootId !== root.engineId) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        'WASM context lost its compiled invariant linear-memory ownership.', {
          phase: 'execution', backend: this.backendName,
        });
    }
    const measuredBytes = Object.freeze({ bytes: root.addressableBytes });
    const rootResourceId = `wasm-linear-${root.engineId}-${root.generation}`;
    const compiledOwner = Object.freeze({
      kind: MemoryOwnerKind.CompiledModel,
      ownerId: invariant.prefixId,
    });
    const exactCapacity = (bytes: number) => Object.freeze([Object.freeze({
      metric: MemoryMetric.Capacity,
      bytes: Object.freeze({ bytes }),
      source: MemoryEvidenceSource.RuntimeCounter,
      valueRelation: MemoryValueRelation.Exact,
      temporalCoverage: MemoryTemporalCoverage.Instant,
    })]);
    const resources: BackendMemorySnapshot['resources'][number][] = [Object.freeze({
      resourceId: rootResourceId,
      backingResourceId: '',
      backingRelation: MemoryBackingRelation.Independent,
      owner: Object.freeze({
        kind: MemoryOwnerKind.BackendShared,
        ownerId: root.engineId,
      }),
      role: MemoryResourceRole.Arena,
      space: MemorySpace.WasmLinear,
      allocator: 'WebAssembly.Memory',
      consumers: Object.freeze([compiledOwner, Object.freeze({ ...request.subject })]),
      measurements: exactCapacity(root.addressableBytes),
      addressableBytes: measuredBytes,
    })];
    const addSuballocation = (
      resourceId: string,
      owner: RuntimeMemoryOwnerRef,
      role: MemoryResourceRole,
      offset: number,
      bytes: number,
      consumers: readonly RuntimeMemoryOwnerRef[],
    ) => {
      if (bytes <= 0) return;
      resources.push(Object.freeze({
        resourceId,
        backingResourceId: rootResourceId,
        backingRelation: MemoryBackingRelation.Suballocation,
        backingOffsetBytes: Object.freeze({ bytes: offset }),
        backingLengthBytes: Object.freeze({ bytes }),
        owner,
        role,
        space: MemorySpace.WasmLinear,
        allocator: 'WasmSharedLinearPool',
        consumers: Object.freeze([...consumers]),
        measurements: exactCapacity(bytes),
        addressableBytes: Object.freeze({ bytes }),
      }));
    };
    addSuballocation(
      `wasm-prefix-raw-${invariant.prefixId}`,
      compiledOwner,
      MemoryResourceRole.Weights,
      invariant.backingOffsetBytes,
      invariant.rawWeightPhysicalBytes,
      [Object.freeze({ ...request.subject })],
    );
    addSuballocation(
      `wasm-prefix-packed-${invariant.prefixId}`,
      compiledOwner,
      MemoryResourceRole.PackedWeights,
      invariant.backingOffsetBytes + invariant.rawWeightPhysicalBytes,
      invariant.packedWeightPhysicalBytes,
      [Object.freeze({ ...request.subject })],
    );
    if (arena.mutableArenaOffsetBytes !== null) {
      addSuballocation(
        `wasm-mutable-${request.subject.ownerId}`,
        Object.freeze({ ...request.subject }),
        MemoryResourceRole.Arena,
        arena.mutableArenaOffsetBytes,
        arena.mutableArenaBytes,
        [],
      );
    }
    return Object.freeze({
      protocol: BACKEND_MEMORY_SNAPSHOT_PROTOCOL,
      resources: Object.freeze(resources),
      // JS-side invariant weights and exact result clones live outside the
      // linear-memory root, so this backend-wide inventory remains partial.
      resourceInventory: MemoryInventoryKind.Partial,
    });
  }

  #assertDecodeEnabled(): void {
    if (!this.#decodeEnabled) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        'WASM decode requires createContext({ decode: ... }) so retained activation storage is explicit.', {
          phase: 'execution', backend: this.backendName,
        });
    }
  }

  async #executeDecode(
    request: BackendResolvedExecutionRequest,
    prepared: PreparedProviderDecode,
  ): Promise<BackendExecutionSnapshot> {
    const engine = this.#engine;
    if (!engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution arena is released.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    let mutationStarted = false;
    try {
      const snapshot = await this.#executeResolved(
        request,
        prepared.engineOptions,
        prepared.operation === 'seed'
          ? () => {
              request.commitExecution();
              mutationStarted = true;
              engine.resetDecodeCache();
            }
          : () => {
              request.commitExecution();
              mutationStarted = true;
            },
      );
      return this.#withDecodeTelemetry(snapshot, this.#decode.commit(prepared));
    } catch (error) {
      if (mutationStarted) {
        engine.resetDecodeCache();
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

  #materializeVariant(plan: ResolvedShapePlan): CachedWasmVariant {
    const bound = this.#boundMaterializer.materialize(plan);
    return Object.freeze({
      plan,
      bound,
      metadataBytes: planMetadataBytes(plan),
    });
  }

  #compileTensorMaximumBytes(plan: ResolvedShapePlan): Readonly<Record<string, number>> {
    return Object.freeze(Object.fromEntries(sortedNames(this.#tensorCapacityBytes).map((name) => {
      const descriptor = plan.tensors[name];
      if (!descriptor) {
        throw new VolvoxAIError('EXECUTION_FAILED',
          `WASM specialization plan omits tensor '${name}'.`, {
            phase: 'execution', backend: this.backendName,
          });
      }
      /* A context's bank residency is fixed, so a sliced invariant weight is
       * exact for every specialization in that context. Activations retain
       * the full logical-domain ceiling proved at provider compilation. */
      return [
        name,
        descriptor.kind === 'weight'
          ? descriptor.sizeBytes
          : this.#tensorCapacityBytes[name],
      ] as const;
    })));
  }

  #specializeVariant(variant: CachedWasmVariant): void {
    const engine = this.#engine;
    if (this.#closed || !engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!engine.graph) {
      const prepared = engine.prepareGraph(variant.bound.graph, this.#invariantSchedule);
      engine.compile(variant.bound.graph, prepared, {
        tensorMaximumBytes: this.#compileTensorMaximumBytes(variant.plan),
        capacityGrowthFactor: 2,
        activationStorage: this.#decodeEnabled ? 'persistent' : 'liveness',
        shapeSignature: variant.plan.signature,
      });
      this.#preparedSchedule = prepared;
      this.#preparedScheduleBuildCount++;
    } else if (engine.inspectArena().currentSignature !== variant.plan.signature) {
      const prepared = this.#preparedSchedule;
      if (!prepared) {
        throw new VolvoxAIError('EXECUTION_FAILED',
          'WASM context lost its invariant prepared schedule.', {
            phase: 'execution', backend: this.backendName,
          });
      }
      engine.rebindGraph(variant.bound.graph, prepared, {
        shapeSignature: variant.plan.signature,
      });
    }
  }

  /** Prepare one legal concrete shape without uploading inputs or dispatching. */
  preload(initialPlan: ResolvedShapePlan): void {
    if (this.#closed || !this.#engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (this.#engine.graph || this.#cache.size !== 0) {
      throw new VolvoxAIError('EXECUTION_FAILED',
        'WASM execution context can preload only before its first specialization.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const variant = this.#materializeVariant(initialPlan);
    this.#specializeVariant(variant);
    /* Publish only after the complete physical compile succeeds. Preloading is
     * neither an execution miss nor an eviction; a first request at this exact
     * shape is a real hit on the already prepared variant. */
    if (variant.metadataBytes <= DEFAULT_PLAN_CACHE_METADATA_BYTES) {
      this.#cache = new Map([[initialPlan.signature, variant]]);
      this.#cacheBytes = variant.metadataBytes;
    }
  }

  async #executeResolved(
    request: BackendResolvedExecutionRequest,
    executionOptions: PreparedProviderDecode['engineOptions'] = {},
    beforeDispatch?: (() => void),
  ): Promise<BackendExecutionSnapshot> {
    const engine = this.#engine;
    if (this.#closed || !engine) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'WASM execution context is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (request.adapters.selectors.some((selector) => selector !== null)) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED', 'WASM adapter execution is unsupported.', {
        phase: 'execution', backend: this.backendName,
      });
    }
    const hostInputs = requireHostExecutionInputs(request, this.backendName);

    const existing = this.#cache.get(request.signature);
    let candidateCache = new Map(this.#cache);
    let candidateCacheBytes = this.#cacheBytes;
    let variant: CachedWasmVariant;
    let hit = false;
    let evictions = 0;
    if (existing) {
      if (existing.plan.graphFingerprint !== request.plan.graphFingerprint) {
        throw new VolvoxAIError('EXECUTION_FAILED', 'WASM cached plan fingerprint changed.', {
          phase: 'execution', backend: this.backendName,
        });
      }
      candidateCache.delete(request.signature);
      candidateCache.set(request.signature, existing);
      variant = existing;
      hit = true;
    } else {
      variant = this.#materializeVariant(request.plan);
      if (variant.metadataBytes <= DEFAULT_PLAN_CACHE_METADATA_BYTES) {
        candidateCache.set(request.signature, variant);
        candidateCacheBytes += variant.metadataBytes;
        while (candidateCache.size > DEFAULT_PLAN_CACHE_ENTRIES ||
               candidateCacheBytes > DEFAULT_PLAN_CACHE_METADATA_BYTES) {
          const oldest = candidateCache.keys().next().value;
          if (oldest === undefined) break;
          const removed = candidateCache.get(oldest);
          candidateCache.delete(oldest);
          candidateCacheBytes -= removed?.metadataBytes ?? 0;
          evictions++;
        }
      }
    }

    this.#specializeVariant(variant);

    /* Cache publication follows complete specialization. No request bytes have
     * entered WASM memory before this point. */
    this.#cache = candidateCache;
    this.#cacheBytes = candidateCacheBytes;
    if (hit) this.#cacheHits++;
    else this.#cacheMisses++;
    this.#cacheEvictions += evictions;

    beforeDispatch?.();
    const rawInputs = Object.fromEntries(
      this.#snapshot.inputNames.map((name) => [name, hostInputs[name].data]),
    );
    /* Kernels and the result clone are one synchronous pinning interval. A
     * sibling context may grow the shared memory only after these host-owned
     * result bytes have been captured. */
    const rawOutputs = engine.executePinned(rawInputs, executionOptions);
    const outputs = request.outputDescriptors.map((descriptor) => {
      const data = rawOutputs[descriptor.name];
      if (!data || runtimeStorageDType(data) !== descriptor.dtype ||
          data.byteLength !== descriptor.sizeBytes) {
        throw new VolvoxAIError('EXECUTION_FAILED',
          `WASM output '${descriptor.name}' disagrees with its resolved descriptor.`, {
            phase: 'execution', backend: this.backendName,
          });
      }
      return Object.freeze({
        name: descriptor.name,
        shape: descriptor.shape,
        dtype: descriptor.dtype,
        location: 'host' as const,
        ownership: 'transfer' as const,
        data: cloneRuntimeData(data),
      });
    });
    const arena: WasmArenaInspection = engine.inspectArena();
    return Object.freeze({
      outputs: Object.freeze(outputs),
      backendReport: Object.freeze({
        shapeSignature: request.signature,
        specializationCacheHit: hit,
        specializationCacheEntries: this.#cache.size,
        specializationCacheMetadataBytes: this.#cacheBytes,
        specializationCacheHits: this.#cacheHits,
        specializationCacheMisses: this.#cacheMisses,
        specializationCacheEvictions: this.#cacheEvictions,
        invariantSchedulePrepareCount: this.#preparedScheduleBuildCount,
        persistentWeightBytes: arena.persistentWeightBytes,
        packedWeightBytes: arena.packedWeightBytes,
        logicalActivationBytes: arena.logicalActivationBytes,
        activationStorage: arena.activationStorage,
        activationRequiredBytes: arena.activationRequiredBytes,
        activationReuseBytes: arena.activationReuseBytes,
        activationArenaCount: arena.activationArenaCount,
        activationCapacityBytes: arena.activationCapacityBytes,
        activationCapacityHighWaterBytes: arena.activationCapacityHighWaterBytes,
        activationGrowCount: arena.activationGrowCount,
        variantBytes: arena.variantBytes,
        variantHighWaterBytes: arena.variantHighWaterBytes,
        wasmHeapBytes: arena.heapBytes,
        wasmHeapHighWaterBytes: arena.heapHighWaterBytes,
        wasmMemoryGrowCount: arena.memoryGrowCount,
        specializationRebindCount: arena.variantRebindCount,
        invariantF32PackCount: arena.f32PackCount,
        invariantQ8PackCount: arena.q8PackCount,
        invariantRawWeightCopyCount: arena.rawWeightCopyCount,
        invariantRawWeightCopyBytes: arena.rawWeightCopyBytes,
        sharedInvariantPrefixId: arena.sharedInvariant?.prefixId ?? null,
        sharedInvariantPhysicalAllocationCount:
          arena.sharedInvariant?.physicalAllocationCount ?? 0,
        sharedInvariantRawWeightBytes: arena.sharedInvariant?.rawWeightBytes ?? 0,
        sharedInvariantRawWeightPhysicalBytes:
          arena.sharedInvariant?.rawWeightPhysicalBytes ?? 0,
        sharedInvariantPackedWeightBytes: arena.sharedInvariant?.packedWeightBytes ?? 0,
        sharedInvariantPackedWeightPhysicalBytes:
          arena.sharedInvariant?.packedWeightPhysicalBytes ?? 0,
        sharedInvariantRawWeightCopyCount:
          arena.sharedInvariant?.rawWeightCopyCount ?? 0,
        sharedInvariantRawWeightCopyBytes:
          arena.sharedInvariant?.rawWeightCopyBytes ?? 0,
        sharedInvariantF32PackCount: arena.sharedInvariant?.f32PackCount ?? 0,
        sharedInvariantQ8PackCount: arena.sharedInvariant?.q8PackCount ?? 0,
        sharedInvariantBorrowerCount: arena.sharedInvariant?.borrowerCount ?? 0,
        mutableArenaOffsetBytes: arena.mutableArenaOffsetBytes,
        mutableArenaBytes: arena.mutableArenaBytes,
        mutableArenaMaximumBytes: arena.mutableArenaMaximumBytes,
      }),
    });
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    const engine = this.#engine;
    this.#engine = null;
    this.#decode.close();
    this.#cache.clear();
    this.#cacheBytes = 0;
    this.#boundMaterializer.clearInvariantStorage();
    /* The compiled invariant-resource owner, not a context, owns host weights
     * and the raw/packed linear prefix borrowed by this engine. */
    this.#preparedSchedule = null;
    engine?.dispose();
  }
}

class WasmProviderCompiledModel implements BackendProviderCompiledModel {
  readonly backendName = 'wasm';
  readonly batchContract: Readonly<BackendProviderBatchContract>;
  readonly compilationEvidence: Readonly<BackendProviderCompilationEvidence>;
  readonly invariantResources: InvariantResourceStore<WasmCompiledInvariantResource>;
  readonly #snapshot: Model;
  #source: WasmEngine | null;
  readonly #resources: WasmResourceDomainProof;
  readonly #resourceDomain: object;
  readonly #invariantSchedule: Readonly<object>;
  #closed = false;

  constructor(
    input: BackendLogicalCompileInput,
    source: WasmEngine,
    deviceIdentity: BackendDeviceIdentity | null,
    resources: WasmResourceDomainProof,
    resourceDomain: object,
  ) {
    const batchSemantics = input.batchSemantics;
    this.batchContract = createBackendProviderBatchContract('single-invocation', {
      independentBatch: batchSemantics.supported
        ? 'compiler-proved/v1'
        : 'unsupported',
    });
    this.#snapshot = input.snapshot;
    this.#source = source;
    this.#resources = resources;
    this.#resourceDomain = resourceDomain;
    this.#invariantSchedule = source.prepareInvariantSchedule({
      nodes: input.graph.nodes,
      tensorCount: input.tensorCount,
      outputNames: input.outputNames,
    });
    this.invariantResources = new InvariantResourceStore(
      (resource) => resource instanceof WasmCompiledInvariantPrefix
        ? resource.inspect().backingLengthBytes
        : resource.ownedBytes,
      (resource) => resource.close(),
    );
    const hostWeights = new WasmCompiledHostWeights(this.#snapshot);
    this.invariantResources.define(WASM_HOST_INVARIANT_WEIGHTS_KEY, hostWeights);
    try {
      const prefixMaterializer = createBoundExecutionGraphMaterializer(this.#snapshot, {
        invariantWeightStorageFactory: ({ name }) => hostWeights.get(name),
      });
      try {
        const prefixPlan = this.#snapshot.staticShapePlan ?? resolveMinimumGraphShapes(
          this.#snapshot.graph,
          this.#snapshot.quantizationByTensor,
        );
        const bound = prefixMaterializer.materialize(prefixPlan);
        const prepared = source.prepareGraph(bound.graph, this.#invariantSchedule);
        const bankedWeightNames = new Set(Object.entries(input.graph.weights)
          .filter(([, descriptor]) => descriptor.bank !== null)
          .map(([name]) => name));
        const prefix = source.prepareCompiledInvariantPrefix(
          bound.graph,
          prepared,
          Math.max(16, resources.maximumInvariantPrefixBytes),
          bankedWeightNames,
        );
        this.invariantResources.define(WASM_LINEAR_INVARIANT_PREFIX_KEY, prefix);
      } finally {
        prefixMaterializer.clearInvariantStorage();
      }
    } catch (error) {
      this.invariantResources.close();
      throw error;
    }
    this.compilationEvidence = Object.freeze({
      device: deviceIdentity,
      allocationBytes: this.invariantResources.ownedBytes,
      operatorFallbackUsed: false,
      offendingNode: null,
      batchSemantics,
      shapeDomain: Object.freeze({
        proofProtocol: 'canonical-symbolic-domain-proof/v1' as const,
        resourceProtocol: 'bounded-resource-maxima/v1' as const,
        support: 'full' as const,
        graphFingerprint: input.graphFingerprint,
        proof: input.shapeDomainProof,
        maximumTensorBytes: resources.maximumTensorBytes,
        staticPrefixBytes: resources.staticPrefixBytes,
        initialMemoryBytes: resources.initialMemoryBytes,
        maximumLinearResidentBytes: resources.maximumLinearResidentBytes,
        maximumInvariantPrefixBytes: resources.maximumInvariantPrefixBytes,
        maximumMutableArenaBytes: resources.maximumMutableArenaBytes,
        maximumBankedMutableArenaBytes: resources.maximumBankedMutableArenaBytes,
        maximumPersistentMetadataBytes: resources.maximumPersistentMetadataBytes,
        maximumSharedScratchBytes: resources.maximumSharedScratchBytes,
        weightBytes: resources.weightBytes,
        maximumBankResidentBytes: resources.maximumBankResidentBytes,
        maximumBankStagingTransactionBytes: resources.maximumBankStagingTransactionBytes,
        maximumInputBytes: resources.maximumInputBytes,
        maximumResultBytes: resources.maximumResultBytes,
        maximumResidentBytes: resources.maximumResidentBytes,
        maximumDecodeResidentBytes: resources.maximumDecodeResidentBytes,
        decodeContextSupported: resources.decodeContextSupported,
        residentLimitBytes: resources.residentLimitBytes,
        resourceLimitBytes: resources.resourceLimitBytes,
      }),
    });
  }

  prepareBatchRoute(plan: ResolvedShapePlan) {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled WASM model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (plan.graphFingerprint !== this.#snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'WASM batch route requires a plan from its compiled logical model.', {
          phase: 'execution', backend: this.backendName,
        });
    }
    return createBackendProviderPreparedBatchRoute(
      this.#resourceDomain, plan.signature, this.invariantResources.deviceEpoch,
    );
  }

  async createContext(
    options: BackendProviderContextOptions,
  ): Promise<BackendProviderExecutionContext> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled WASM model is closed.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    const source = this.#source;
    if (!source) {
      throw new VolvoxAIError('HANDLE_DISPOSED', 'Compiled WASM module is released.', {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (options.decode !== undefined && !this.#resources.decodeContextSupported) {
      throw new VolvoxAIError('BACKEND_UNSUPPORTED',
        `WASM retained decode resources require ` +
        `${this.#resources.maximumDecodeResidentBytes} bytes, exceeding the ` +
        `resident/transaction limit ${this.#resources.residentLimitBytes}.`, {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const lease = options.invariantResources as InvariantResourceLease<WasmCompiledInvariantResource>;
    if (typeof lease.borrow !== 'function' ||
        lease.ownerIdentity !== this.invariantResources.ownerIdentity ||
        lease.deviceEpoch !== this.invariantResources.deviceEpoch) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        'WASM context requires the exact compiled invariant-resource lease.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const hostWeights = lease.borrow(WASM_HOST_INVARIANT_WEIGHTS_KEY);
    const prefix = lease.borrow(WASM_LINEAR_INVARIANT_PREFIX_KEY);
    if (!(hostWeights instanceof WasmCompiledHostWeights) ||
        !(prefix instanceof WasmCompiledInvariantPrefix)) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        'WASM compiled invariant-resource kinds are invalid.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const selectedBanks = options.initialPlan === undefined
      ? true
      : Object.keys(options.initialPlan.bankResidency).length !== 0;
    const maximumMutableBytes = selectedBanks
      ? this.#resources.maximumBankedMutableArenaBytes
      : this.#resources.maximumMutableArenaBytes;
    const engine = source.forkWithCompiledInvariants(
      prefix,
      Math.max(16, maximumMutableBytes),
    );
    let context: WasmProviderExecutionContext | null = null;
    try {
      context = new WasmProviderExecutionContext(
        this.#snapshot,
        engine,
        this.#resources.tensorCapacityBytes,
        this.#invariantSchedule,
        hostWeights,
        options,
      );
      if (options?.initialPlan) context.preload(options.initialPlan);
      return context;
    } catch (error) {
      if (context) context.close();
      else engine.dispose();
      throw error;
    }
  }

  close(): void {
    if (this.#closed) return;
    this.invariantResources.close();
    this.#closed = true;
    this.#source = null;
  }
}

/** Dynamic-v1 WASM provider with one reusable arena per execution context. */
export class WasmBackendProvider implements BackendProvider {
  readonly providerVersion = VOLVOXAI_BACKEND_PROVIDER_VERSION;
  readonly backendName = 'wasm';
  readonly capabilities: Readonly<BackendProviderCapabilities>;
  readonly deviceIdentity: BackendDeviceIdentity | null;
  readonly #resourceDomain = Object.freeze({});
  #source: WasmEngine | null;
  #closed = false;

  constructor(source: WasmEngine) {
    if (!(source instanceof WasmEngine)) {
      throw new VolvoxAIError('INVALID_ARGUMENT', 'WASM provider requires a WasmEngine.', {
        phase: 'initialization', backend: this.backendName,
      });
    }
    this.#source = source;
    this.capabilities = createBackendProviderCapabilities({
      operatorFallback: 'none',
      outputLocation: 'host',
      dynamicShapeDomain: 'full',
    });
    this.deviceIdentity = createBackendDeviceIdentity({
      backend: 'wasm',
      device: 'webassembly',
    });
  }

  async compile(
    input: BackendLogicalCompileInput,
    options: BackendProviderCompileOptions,
  ): Promise<BackendProviderCompiledModel> {
    if (this.#closed) {
      throw new VolvoxAIError('HANDLE_DISPOSED', "Backend provider 'wasm' is closed.", {
        phase: 'lifecycle', backend: this.backendName,
      });
    }
    if (!(input?.snapshot instanceof Model) ||
        input.shapeDomainProof !== input.snapshot.shapeDomainProof ||
        input.graphFingerprint !== input.snapshot.definitionFingerprint) {
      throw new VolvoxAIError('INVALID_ARGUMENT',
        'WASM provider compile requires the exact immutable logical compile view.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    if (options.operatorFallback === 'forbid' && this.capabilities.operatorFallback !== 'none') {
      throw new VolvoxAIError('OPERATOR_FALLBACK_FORBIDDEN',
        "Backend 'wasm' cannot attest strict operator routing.", {
          phase: 'compilation', backend: this.backendName,
      });
    }
    const source = this.#source;
    if (!source) {
      throw new VolvoxAIError('ABI_UNSUPPORTED',
        `WASM sidecar does not expose runtime ABI ${WASM_RUNTIME_ABI_VERSION}; ` +
        'rebuild the sidecar.', {
          phase: 'compilation', backend: this.backendName,
        });
    }
    const abi = checkedWasmRuntimeAbi(source);
    wasmRegistryPreflight(input);
    const resources = checkedWasmResourceDomain(input, source, abi);
    return new WasmProviderCompiledModel(
      input,
      source,
      this.deviceIdentity,
      resources,
      this.#resourceDomain,
    );
  }

  close(): void {
    if (this.#closed) return;
    const source = this.#source;
    this.#closed = true;
    this.#source = null;
    source?.dispose();
  }

  /** @internal Construction cleanup uses this to preserve one close owner. */
  _isClosed(): boolean {
    return this.#closed;
  }
}
