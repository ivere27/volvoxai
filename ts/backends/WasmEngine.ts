import { Tensor } from '../core/Tensor.js';
import { DataType } from '../generated/volvoxaiEnums.js';
import { kernelRoute } from '../generated/kernelRegistry.js';
import { normalizeSpatialPair } from '../ops/spatialParameters.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import { assertInferenceExecutionOptions, BackendEngine } from './BackendEngine.js';
import { geluApproximation } from '../ops/gELU.js';
import { ropeDescriptor } from '../ops/roPE.js';
import { ssmScanDescriptor } from '../ops/ssmScan.js';
import { batchMatMulDescriptor } from '../ops/batchMatMul.js';
import { qBatchMatMulDescriptor } from '../ops/qBatchMatMul.js';
import { comparisonDescriptor, logicalNotDescriptor } from '../ops/comparison.js';
import { SHAPE_SYMBOL_PATTERN, runtimeDTypeBytes } from '../ops/shapeSystem.js';
import {
  checkedWindowOutput,
  fullSpatialPads,
  spatialPair,
} from '../ops/spatialKernelValidation.js';
import { incrementalExecutionEnabled, incrementalNodeSelection } from './incrementalExecution.js';
import {
  incrementalRowPosition,
  prepareQuantizedRows,
  quantizedRowNode,
} from './quantizedRowExecution.js';
import type { BackendExecutionOptions } from './BackendEngine.js';
import type { GraphNode, RuntimeDType, RuntimeTypedArray } from '../types.js';

declare const __VOLVOXAI_BROWSER_ONLY__: boolean | undefined;

type WasmAbiArgument = number | undefined;
type WasmNumericExport = (...args: WasmAbiArgument[]) => number;

interface WasmKernelExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  wasm_runtime_abi_version: WasmNumericExport;
  alloc_bytes: WasmNumericExport;
  argmax_axis_typed: WasmNumericExport;
  averagepool2d_f32: WasmNumericExport;
  batch_norm2d_f32: WasmNumericExport;
  binary_broadcast_f32: WasmNumericExport;
  compare_broadcast_i32: WasmNumericExport;
  not_i32: WasmNumericExport;
  cast_typed: WasmNumericExport;
  clip_f32: WasmNumericExport;
  clip_i32: WasmNumericExport;
  concat_slice_f32: WasmNumericExport;
  concat_slice_u32: WasmNumericExport;
  concat_slice_i8u8: WasmNumericExport;
  conv1d_f32: WasmNumericExport;
  conv2d_f32: WasmNumericExport;
  conv_transpose2d_f32: WasmNumericExport;
  copy_f32: WasmNumericExport;
  copy_i8u8: WasmNumericExport;
  cos_f32: WasmNumericExport;
  cross_attention_f32: WasmNumericExport;
  cross_sdpa_f32: WasmNumericExport;
  dequantize_linear_typed: WasmNumericExport;
  embedding_f32: WasmNumericExport;
  expand_nd_f32: WasmNumericExport;
  expand_nd_i8u8: WasmNumericExport;
  expand_nd_u32: WasmNumericExport;
  gather_elements_i32_f32: WasmNumericExport;
  gather_i32_f32: WasmNumericExport;
  gelu_f32: WasmNumericExport;
  gelu_tanh_f32: WasmNumericExport;
  gemm_f32_pack_b: WasmNumericExport;
  gemm_f32_packed: WasmNumericExport;
  gemm_f32_packed_elements: WasmNumericExport;
  global_average_pool_f32: WasmNumericExport;
  groupnorm_f32: WasmNumericExport;
  hardsigmoid_f32: WasmNumericExport;
  hardswish_f32: WasmNumericExport;
  interp1d_f32: WasmNumericExport;
  layernorm_f32: WasmNumericExport;
  leakyrelu_f32: WasmNumericExport;
  linear_f32: WasmNumericExport;
  logsoftmax_f32: WasmNumericExport;
  matmul_f32: WasmNumericExport;
  matmul_quantized_f32: WasmNumericExport;
  matmul_quantized_f32_packed: WasmNumericExport;
  maxpool2d_f32: WasmNumericExport;
  maxpool2d_i8u8: WasmNumericExport;
  mean_height_f32: WasmNumericExport;
  moe_linear_f32: WasmNumericExport;
  vx_moe_linear_banked_f32: WasmNumericExport;
  moe_router_f32: WasmNumericExport;
  non_max_suppression_typed: WasmNumericExport;
  pack_q8_weight: WasmNumericExport;
  pack_q8_weight_canonical: WasmNumericExport;
  packed_q8_weight_size: WasmNumericExport;
  packed_q8_weight_canonical_size: WasmNumericExport;
  pad_2d_f32: WasmNumericExport;
  prelu_generic_f32: WasmNumericExport;
  profile_x_f32: WasmNumericExport;
  profile_y_f32: WasmNumericExport;
  qadd_i8u8: WasmNumericExport;
  qargmax_i8u8: WasmNumericExport;
  qconv2d_i8u8: WasmNumericExport;
  qconv2d_im2col_i8u8: WasmNumericExport;
  qembedding_i8u8: WasmNumericExport;
  qgelu_i8u8: WasmNumericExport;
  qgroupnorm_i8u8: WasmNumericExport;
  qlayernorm_i8u8: WasmNumericExport;
  qlinear_i8u8: WasmNumericExport;
  qbatch_matmul_i8u8: WasmNumericExport;
  qbatch_matmul_i8u8_simd128: WasmNumericExport;
  qlinear_i8u8_packed: WasmNumericExport;
  qmaskedmean_i8u8: WasmNumericExport;
  qsdpa_i8u8: WasmNumericExport;
  qsilu_i8u8: WasmNumericExport;
  quantize_linear_typed: WasmNumericExport;
  reduce_mean_f32: WasmNumericExport;
  reduce_sum_f32: WasmNumericExport;
  relu_f32: WasmNumericExport;
  requantize_linear_i8u8: WasmNumericExport;
  reset_heap: WasmNumericExport;
  heap_mark: WasmNumericExport;
  heap_rewind: WasmNumericExport;
  resize_bilinear_f32: WasmNumericExport;
  resize_nearest2d_f32: WasmNumericExport;
  resize_nearest2d_i8u8: WasmNumericExport;
  rmsnorm_f32: WasmNumericExport;
  rope_f32: WasmNumericExport;
  sdpa_f32: WasmNumericExport;
  sigmoid_f32: WasmNumericExport;
  silu_f32: WasmNumericExport;
  sin_f32: WasmNumericExport;
  slice_nd_f32: WasmNumericExport;
  slice_nd_u32: WasmNumericExport;
  softmax_f32: WasmNumericExport;
  spatial_softargmax_y_f32: WasmNumericExport;
  split_slice_f32: WasmNumericExport;
  split_slice_u32: WasmNumericExport;
  ssm_scan_f32: WasmNumericExport;
  tanh_f32: WasmNumericExport;
  transpose_nd_f32: WasmNumericExport;
  transpose_nd_i8u8: WasmNumericExport;
  transpose_nd_u32: WasmNumericExport;
  upsample_nearest2x_f32: WasmNumericExport;
  where_typed_f32: WasmNumericExport;
  where_typed_32: WasmNumericExport;
  where_broadcast_32: WasmNumericExport;
}

interface RelaxedSimdExports extends WebAssembly.Exports {
  qlinear_i8u8_relaxed: WasmNumericExport;
}

type WasmParentInstance = WebAssembly.Instance & { readonly exports: WasmKernelExports };
type RelaxedSimdInstance = WebAssembly.Instance & { readonly exports: RelaxedSimdExports };

interface WasmInstantiation {
  module: WebAssembly.Module;
  instance: WasmParentInstance;
}

interface RelaxedSimdInstantiation {
  module: WebAssembly.Module;
  instance: RelaxedSimdInstance;
}

type WasmNode = GraphNode<Tensor> & { params: Record<string, any> };
type WasmGraph = RuntimeGraph & { nodes: WasmNode[] };

const WASM_PREPARED_GRAPH = Symbol('WasmPreparedGraph');
const WASM_INVARIANT_SCHEDULE = Symbol('WasmInvariantSchedule');

interface WasmPreparedStep {
  readonly nodeIndex: number;
  readonly nodeId: string | number;
  readonly opType: string;
  readonly kernelRoute: string;
}

interface WasmPreparedInputPreflights {
  readonly qGroupNorm: readonly number[];
  readonly qLayerNorm: readonly number[];
  readonly qSDPA: readonly number[];
  readonly qMaskedMean: readonly number[];
  readonly rope: readonly number[];
}

interface WasmInvariantSchedule {
  readonly [WASM_INVARIANT_SCHEDULE]: true;
  readonly tensorCount: number;
  readonly outputNames: readonly string[];
  readonly schedule: readonly WasmPreparedStep[];
  readonly inputPreflights: WasmPreparedInputPreflights;
}

interface WasmInvariantScheduleInput {
  readonly nodes: readonly Readonly<{ readonly id: string | number; readonly opType: string }>[];
  readonly tensorCount: number;
  readonly outputNames: readonly string[];
}

interface WasmPreparedGraph extends Readonly<object> {
  readonly [WASM_PREPARED_GRAPH]: true;
  readonly topologyRevision: number;
  readonly weightRevision: number;
  readonly tensorCount: number;
  readonly outputNames: readonly string[];
  readonly schedule: readonly WasmPreparedStep[];
  readonly inputPreflights: WasmPreparedInputPreflights;
}

interface PackedF32WeightDescriptor {
  readonly pointer: number;
  readonly bytes: number;
  readonly dIn: number;
  readonly dOut: number;
  readonly doutFirst: boolean;
  readonly cacheKey: string;
}

interface PackedQ8WeightDescriptor {
  readonly pointer: number;
  readonly bytes: number;
  readonly cacheKey: string;
}

interface WasmTensorCapacity {
  readonly dtype: RuntimeDType;
  readonly capacityBytes: number;
  readonly isWeight: boolean;
  readonly owner: string;
  /** Present only for topology-liveness activation storage. */
  readonly arena?: RuntimeDType;
  /** Byte offset from the dtype arena base. */
  readonly offsetBytes?: number;
}

type WasmActivationStorage = 'persistent' | 'liveness';

interface WasmActivationLivenessRegion {
  readonly owner: string;
  readonly dtype: RuntimeDType;
  readonly offsetBytes: number;
  readonly sizeBytes: number;
  readonly birth: number;
  readonly lastUse: number;
}

interface WasmActivationLivenessLayout {
  readonly capacityByArena: ReadonlyMap<RuntimeDType, number>;
  readonly capacityBytes: number;
  readonly uniqueBytes: number;
  readonly regions: ReadonlyMap<string, WasmActivationLivenessRegion>;
}

interface WasmAllocationMeasurement {
  cursor: number;
}

interface WasmStagedMetadataWrite {
  readonly pointer: number;
  readonly bytes: Uint8Array;
}

interface WasmCapacityCandidate {
  readonly capacities: Map<string, WasmTensorCapacity>;
  readonly currentBytes: number;
  readonly requiredBytes: number;
  readonly grew: boolean;
  readonly arenaCapacities: ReadonlyMap<RuntimeDType, number> | null;
}

export interface WasmGraphAllocationOptions {
  /** Optional initial capacity targets. Omitted names start at logical bytes. */
  readonly tensorCapacityBytes?: Readonly<Record<string, number>>;
  /** Hard per-tensor capacity ceilings proved for the complete shape domain. */
  readonly tensorMaximumBytes?: Readonly<Record<string, number>>;
  /** Geometric activation growth factor. Default: 2. */
  readonly capacityGrowthFactor?: number;
  /**
   * Ordinary inference may reuse storage across topology-disjoint activation
   * lifetimes. Incremental decode must retain every intermediate instead.
   * The direct engine defaults to the conservative persistent mode.
   */
  readonly activationStorage?: WasmActivationStorage;
  readonly shapeSignature?: string;
}

export interface WasmArenaInspection {
  readonly reusable: boolean;
  readonly currentSignature: string | null;
  readonly activationStorage: WasmActivationStorage;
  readonly persistentWeightBytes: number;
  readonly packedWeightBytes: number;
  readonly logicalActivationBytes: number;
  readonly activationRequiredBytes: number;
  readonly activationReuseBytes: number;
  readonly activationArenaCount: number;
  readonly activationCapacityBytes: number;
  readonly activationCapacityHighWaterBytes: number;
  readonly activationGrowCount: number;
  readonly variantBytes: number;
  readonly variantHighWaterBytes: number;
  readonly heapBytes: number;
  readonly heapHighWaterBytes: number;
  readonly memoryGrowCount: number;
  readonly variantRebindCount: number;
  readonly f32PackCount: number;
  readonly q8PackCount: number;
}

type WasmNodeMetadata = { kind: string } & Record<string, any>;
type WasmQ8PackMode = 'w8a8-wide' | 'canonical';

/**
 * compile() phase breakdown, opt-in via `globalThis.__VOLVOX_WASM_COMPILE_PROFILE`.
 *
 * The phase fields partition compile(); `unattributedMs` is whatever the phases
 * did not claim. `growMs`/`growCount`/`growPages` are *nested* inside the
 * allocator phases (tensorAllocMs, packWeightMs, metadataAllocMs) rather than
 * being a phase of their own, so they must not be added to the phase sum.
 */
interface WasmCompileProfile {
  preflightMs: number;
  tensorAllocMs: number;
  descriptorMs: number;
  packWeightMs: number;
  viewRefreshMs: number;
  metadataAllocMs: number;
  totalMs: number;
  unattributedMs: number;
  /* Nested inside the allocator phases above. */
  growMs: number;
  growCount: number;
  growPages: number;
  allocBytesCount: number;
  viewRefreshCount: number;
  descriptorCount: number;
  tensorCopyBytes: number;
  packBytes: number;
  finalHeapBytes: number;
}

interface WasmAdapterSelector {
  name: string;
  version?: number;
  scale?: number;
}

interface WasmExecutionOptions extends BackendExecutionOptions {
  adapter?: WasmAdapterSelector | null;
  adapters?: Array<WasmAdapterSelector | null>;
}

const WASM_RELAXED_SIMD_SECTION = 'volvoxai.relaxed_simd.v1';
const WASM_RELAXED_QLINEAR_EXPORT = 'qlinear_i8u8_relaxed';
const WASM_QCONV_IM2COL_MAX_BYTES = 64 * 1024 * 1024;
const wasmW8A8PackMode = (dOut: number): WasmQ8PackMode =>
  dOut >= 8 ? 'w8a8-wide' : 'canonical';
/* Route ids are F32 today, so a dense lookup beyond the contiguous F32 integer
 * domain is neither useful nor safe. Keep a malformed sparse global slot from
 * materializing a near-wasm32-sized JavaScript typed array before the arena
 * allocator gets a chance to reject it. */
const WASM_MOE_SLOT_TABLE_MAX_BYTES = 64 * 1024 * 1024;

async function instantiateRelaxedSimdChild(
  parentModule: unknown,
  memory: unknown,
): Promise<Readonly<RelaxedSimdInstantiation> | null> {
  if (!(parentModule instanceof WebAssembly.Module) ||
      !(memory instanceof WebAssembly.Memory)) return null;
  try {
    const sections = WebAssembly.Module.customSections(
      parentModule, WASM_RELAXED_SIMD_SECTION,
    );
    // A duplicated section is ambiguous and must not select an arbitrary ABI.
    if (sections.length !== 1 || sections[0].byteLength === 0) return null;
    const module = await WebAssembly.compile(sections[0]);
    const imports = WebAssembly.Module.imports(module);
    if (imports.length !== 1 || imports[0].module !== 'env' ||
        imports[0].name !== 'memory' || imports[0].kind !== 'memory') return null;
    const exports = WebAssembly.Module.exports(module);
    if (exports.length !== 1 || exports[0].name !== WASM_RELAXED_QLINEAR_EXPORT ||
        exports[0].kind !== 'function') return null;
    const instance = await WebAssembly.instantiate(module, { env: { memory } }) as RelaxedSimdInstance;
    const kernel = instance.exports[WASM_RELAXED_QLINEAR_EXPORT];
    // Exported WebAssembly functions expose their parameter count as `length`.
    // A zero descriptor is the v1 ABI's side-effect-free rejection sentinel.
    // Together these checks keep an accidentally embedded older/newer child
    // from returning success while accepting a different call shape.
    if (typeof kernel !== 'function' || kernel.length !== 17 ||
        kernel(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) !== 0) return null;
    return Object.freeze({ module, instance });
  } catch {
    // Relaxed SIMD is optional.  A browser without the proposal, a malformed
    // embedded child, or an incompatible memory import retains the baseline
    // SIMD/raw dispatch below.
    return null;
  }
}

function wasmTensorView(tensor, buffer, pointer) {
  if (tensor.dtype === 'int32') return new Int32Array(buffer, pointer, tensor.sizeBytes / 4);
  if (tensor.dtype === 'int8') return new Int8Array(buffer, pointer, tensor.sizeBytes);
  if (tensor.dtype === 'uint8') return new Uint8Array(buffer, pointer, tensor.sizeBytes);
  return new Float32Array(buffer, pointer, tensor.sizeBytes / 4);
}

function wasmTensorPointer(tensor, memory, label) {
  const storage = tensor?.buffer;
  if (!ArrayBuffer.isView(storage) || storage instanceof DataView || storage.buffer !== memory.buffer) {
    throw new Error(`WASM ${label} does not reference the compiled linear memory.`);
  }
  return storage.byteOffset;
}

function wasmDtypeCode(dtype) {
  if (dtype === 'float32') return DataType.F32;
  if (dtype === 'int32') return DataType.I32;
  if (dtype === 'int8') return DataType.I8;
  if (dtype === 'uint8') return DataType.U8;
  throw new Error(`Unsupported WASM kernel dtype '${dtype}'.`);
}

function wasmKernelDtype(tensor, label) {
  switch (tensor?.dtype) {
    case 'float32': return DataType.F32;
    case 'int32': return DataType.I32;
    case 'int8': return DataType.I8;
    case 'uint8': return DataType.U8;
    default: throw new Error(`WASM ${label} has unsupported dtype '${tensor?.dtype}'.`);
  }
}

function wasmScalar(tensor, buffer, pointer, label) {
  if (!tensor) return null;
  const values = wasmTensorView(tensor, buffer, pointer);
  if (values.length < 1) throw new Error(`WASM ${label} must contain at least one value.`);
  return values[0];
}

function attentionMaskMode(mask, batch, seqQ, seqKV, nodeId) {
  if (!mask) return 0;
  if (mask.dtype !== 'int32') throw new Error(`WASM attention mask at node ${nodeId} must be int32.`);
  if (mask.shape.length === 1 && mask.shape[0] === seqKV) return 1;
  if (mask.shape.length === 2 && mask.shape[1] === seqKV) {
    if (mask.shape[0] === batch) return 2;
    if (mask.shape[0] === seqQ) return 3;
  }
  if (mask.shape.length === 3 && mask.shape[0] === batch &&
      mask.shape[1] === seqQ && mask.shape[2] === seqKV) return 4;
  throw new Error(`WASM attention mask at node ${nodeId} has an incompatible shape.`);
}

function localMaskBinding(
  pointers, mask, mode, batchIndex, seqQ, seqKV, baseOverride: number | null = null,
) {
  if (!mask || mode === 0) return [0, 0];
  const base = baseOverride ?? pointers.get(mask.name);
  if (!Number.isSafeInteger(base) || base < 0) {
    throw new Error(`WASM attention mask '${String(mask.name)}' has no valid heap pointer.`);
  }
  if (mode === 1) return [base, 1];
  if (mode === 2) return [base + batchIndex * seqKV * 4, 1];
  if (mode === 3) return [base, 2];
  return [base + batchIndex * seqQ * seqKV * 4, 2];
}

const WASM_PORTABLE_MAX_RANK = 8;
const WASM_PORTABLE_MAX_U32 = 0xffffffff;
/* wasm32 addresses 4 GiB in 64 KiB pages, so the heap can never exceed this. */
const WASM_MAX_MEMORY_PAGES = 65536;
/* Reshape-family operators that only reinterpret a tensor's shape. Their output
 * is the input's bytes, so compile() aliases the heap pointer instead of
 * copying. Transpose is deliberately absent: it permutes storage. */
const WASM_LAYOUT_INVARIANT_OPS = new Set([
  'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Identity',
]);
/* Largest single geometric step, in pages (64 MiB). Doubling below this keeps
 * the grow count logarithmic; capping above it bounds the leftover headroom to
 * one chunk instead of one whole doubling. */
const WASM_GROW_MAX_STEP_PAGES = 1024;
const WASM_NAME_ENCODER = new TextEncoder();

function compareWasmNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = WASM_NAME_ENCODER.encode(left);
  const rightBytes = WASM_NAME_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length || (left < right ? -1 : 1);
}

function checkedWasmActivationAdd(left: number, right: number, label: string): number {
  if (!Number.isSafeInteger(left) || left < 0 || !Number.isSafeInteger(right) || right < 0 ||
      left > 0x7ffffff0 - right) {
    throw new Error(`WASM ${label} exceeds the signed allocator address range.`);
  }
  return left + right;
}

function sameWasmAliasMap(
  left: ReadonlyMap<string, string>,
  right: ReadonlyMap<string, string>,
): boolean {
  return left.size === right.size && [...left.entries()].every(
    ([name, source]) => right.get(name) === source,
  );
}

/**
 * Storage-view operators are represented as one physical lifetime. Extending
 * the source lifetime through every aliased consumer is essential: merely
 * assigning the same pointer after packing could let an unrelated tensor
 * overwrite a still-live view.
 */
function wasmActivationAliases(
  graph: WasmGraph,
  prepared: WasmPreparedGraph,
): ReadonlyMap<string, string> {
  const aliases = new Map<string, string>();
  for (const step of prepared.schedule) {
    const node = graph.nodes[step.nodeIndex];
    const viewOnly = WASM_LAYOUT_INVARIANT_OPS.has(node.opType);
    if (node.opType !== 'Dropout' && !viewOnly) continue;
    const input = node.inputs.input || node.inputs.x || node.inputs.data;
    const output = node.outputs.out || Object.values(node.outputs || {})[0];
    if (!input || !output || input.dtype !== output.dtype ||
        input.sizeBytes !== output.sizeBytes) {
      if (viewOnly) continue;
      throw new Error(`Dropout node ${node.id} requires equal-size input/output tensors.`);
    }
    aliases.set(output.name, input.name);
  }
  return aliases;
}

function wasmActivationAliasRoots(
  graph: WasmGraph,
  aliases: ReadonlyMap<string, string>,
): ReadonlyMap<string, string> {
  const roots = new Map<string, string>();
  const resolve = (name: string, visiting = new Set<string>()): string => {
    const cached = roots.get(name);
    if (cached !== undefined) return cached;
    const source = aliases.get(name);
    if (source === undefined) {
      roots.set(name, name);
      return name;
    }
    if (!graph.tensors.has(source)) {
      throw new Error(`WASM storage alias '${name}' references unknown tensor '${source}'.`);
    }
    if (visiting.has(name)) {
      throw new Error(`WASM storage alias cycle contains tensor '${name}'.`);
    }
    visiting.add(name);
    const root = resolve(source, visiting);
    visiting.delete(name);
    roots.set(name, root);
    return root;
  };
  for (const [name, tensor] of graph.tensors) {
    if (tensor.isWeight !== true) resolve(name);
  }
  return roots;
}

interface MutableWasmLivenessRegion {
  readonly owner: string;
  readonly dtype: RuntimeDType;
  readonly birth: number;
  readonly lastUse: number;
  readonly sizeBytes: number;
  offsetBytes: number;
}

interface MutableWasmFreeRegion {
  offsetBytes: number;
  sizeBytes: number;
}

function coalesceWasmFreeRegions(regions: MutableWasmFreeRegion[]): void {
  regions.sort((left, right) => left.offsetBytes - right.offsetBytes);
  for (let index = 1; index < regions.length;) {
    const previous = regions[index - 1];
    const current = regions[index];
    if (previous.offsetBytes + previous.sizeBytes === current.offsetBytes) {
      previous.sizeBytes += current.sizeBytes;
      regions.splice(index, 1);
    } else {
      index++;
    }
  }
}

/** Deterministic best-fit packing of exact or bounded activation sizes. */
function planWasmActivationLiveness(
  graph: WasmGraph,
  prepared: WasmPreparedGraph,
  aliases: ReadonlyMap<string, string>,
  sizes: ReadonlyMap<string, number>,
): WasmActivationLivenessLayout {
  const roots = wasmActivationAliasRoots(graph, aliases);
  const birth = new Map<string, number>();
  const lastUse = new Map<string, number>();
  const dtypeByRoot = new Map<string, RuntimeDType>();
  const bytesByRoot = new Map<string, number>();
  const invariantRoots = new Set<string>();
  let uniqueBytes = 0;

  for (const [name, tensor] of graph.tensors) {
    if (tensor.isWeight === true) continue;
    const sizeBytes = sizes.get(name);
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes! < tensor.sizeBytes ||
        sizeBytes! > 0x7ffffff0 || sizeBytes! % runtimeDTypeBytes(tensor.dtype) !== 0) {
      throw new Error(`WASM activation '${name}' has an invalid planned byte size.`);
    }
    const root = roots.get(name)!;
    const rootTensor = graph.tensors.get(root);
    if (!rootTensor) {
      throw new Error(`WASM activation alias '${name}' has unknown root '${root}'.`);
    }
    if (rootTensor.isWeight === true) {
      if (rootTensor.dtype !== tensor.dtype || sizeBytes !== rootTensor.sizeBytes) {
        throw new Error(
          `WASM activation alias '${name}' exceeds or disagrees with invariant root '${root}'.`,
        );
      }
      invariantRoots.add(root);
      continue;
    }
    const dtype = dtypeByRoot.get(root);
    if (dtype !== undefined && dtype !== tensor.dtype) {
      throw new Error(`WASM activation alias '${name}' changes storage dtype.`);
    }
    dtypeByRoot.set(root, tensor.dtype);
    bytesByRoot.set(root, Math.max(bytesByRoot.get(root) ?? 0, sizeBytes!));
    if (tensor.isInput) {
      birth.set(root, -1);
      lastUse.set(root, -1);
    }
  }

  const producedInvariantAliases = new Set<string>();
  for (let scheduleIndex = 0; scheduleIndex < prepared.schedule.length; scheduleIndex++) {
    const step = prepared.schedule[scheduleIndex];
    const node = graph.nodes[step.nodeIndex];
    for (const tensor of Object.values(node.inputs)) {
      if (tensor.isWeight === true) continue;
      const root = roots.get(tensor.name);
      if (root !== undefined && invariantRoots.has(root)) {
        if (!aliases.has(tensor.name) || !producedInvariantAliases.has(tensor.name)) {
          throw new Error(`WASM topology consumes '${tensor.name}' before it is produced.`);
        }
        continue;
      }
      if (root === undefined || !birth.has(root)) {
        throw new Error(`WASM topology consumes '${tensor.name}' before it is produced.`);
      }
      lastUse.set(root, Math.max(lastUse.get(root)!, scheduleIndex));
    }
    for (const tensor of Object.values(node.outputs)) {
      if (tensor.isWeight === true) {
        throw new Error(`WASM topology node '${node.id}' produces weight '${tensor.name}'.`);
      }
      const root = roots.get(tensor.name);
      if (root === undefined) {
        throw new Error(`WASM topology node '${node.id}' has unknown output '${tensor.name}'.`);
      }
      if (invariantRoots.has(root)) {
        const source = aliases.get(tensor.name);
        if (source === undefined) {
          throw new Error(
            `WASM topology node '${node.id}' produces invariant-root activation ` +
            `'${tensor.name}' without a storage alias.`,
          );
        }
        const sourceTensor = graph.tensors.get(source);
        if (!sourceTensor || (sourceTensor.isWeight !== true &&
            !producedInvariantAliases.has(source))) {
          throw new Error(`WASM topology aliases '${tensor.name}' before its source is live.`);
        }
        producedInvariantAliases.add(tensor.name);
        continue;
      }
      if (aliases.has(tensor.name)) {
        if (!birth.has(root)) {
          throw new Error(`WASM topology aliases '${tensor.name}' before its source is live.`);
        }
        lastUse.set(root, Math.max(lastUse.get(root)!, scheduleIndex));
      } else {
        if (birth.has(root)) {
          throw new Error(`WASM topology produces activation '${tensor.name}' more than once.`);
        }
        birth.set(root, scheduleIndex);
        lastUse.set(root, scheduleIndex);
      }
    }
  }
  for (const name of graph.outputNames) {
    const tensor = graph.tensors.get(name);
    if (!tensor) throw new Error(`WASM topology declares unknown output '${name}'.`);
    if (tensor.isWeight === true) continue;
    const root = roots.get(name)!;
    if (invariantRoots.has(root)) {
      if (!producedInvariantAliases.has(name)) {
        throw new Error(`WASM topology output '${name}' is never produced.`);
      }
      continue;
    }
    if (!birth.has(root)) throw new Error(`WASM topology output '${name}' is never produced.`);
    lastUse.set(root, prepared.schedule.length);
  }
  const valuesByDtype = new Map<RuntimeDType, MutableWasmLivenessRegion[]>();
  for (const owner of [...bytesByRoot.keys()].sort(compareWasmNames)) {
    const ownerBirth = birth.get(owner);
    const ownerLastUse = lastUse.get(owner);
    if (ownerBirth === undefined || ownerLastUse === undefined) {
      throw new Error(`WASM activation '${owner}' has no complete topology lifetime.`);
    }
    const sizeBytes = bytesByRoot.get(owner)!;
    uniqueBytes = checkedWasmActivationAdd(uniqueBytes, sizeBytes,
      'unique logical activation bytes');
    const dtype = dtypeByRoot.get(owner)!;
    const values = valuesByDtype.get(dtype) ?? [];
    values.push({
      owner,
      dtype,
      birth: ownerBirth,
      lastUse: ownerLastUse,
      sizeBytes,
      offsetBytes: 0,
    });
    valuesByDtype.set(dtype, values);
  }

  const capacityByArena = new Map<RuntimeDType, number>();
  const regions = new Map<string, WasmActivationLivenessRegion>();
  let capacityBytes = 0;
  for (const [dtype, values] of [...valuesByDtype.entries()]
    .sort(([left], [right]) => compareWasmNames(left, right))) {
    values.sort((left, right) => left.birth - right.birth ||
      right.sizeBytes - left.sizeBytes || compareWasmNames(left.owner, right.owner));
    const live: MutableWasmLivenessRegion[] = [];
    const free: MutableWasmFreeRegion[] = [];
    let highWater = 0;
    for (const value of values) {
      for (let index = live.length - 1; index >= 0; index--) {
        if (live[index].lastUse < value.birth) {
          free.push({
            offsetBytes: live[index].offsetBytes,
            sizeBytes: live[index].sizeBytes,
          });
          live.splice(index, 1);
        }
      }
      coalesceWasmFreeRegions(free);
      let best = -1;
      for (let index = 0; index < free.length; index++) {
        if (free[index].sizeBytes < value.sizeBytes) continue;
        if (best === -1 || free[index].sizeBytes < free[best].sizeBytes ||
            (free[index].sizeBytes === free[best].sizeBytes &&
             free[index].offsetBytes < free[best].offsetBytes)) {
          best = index;
        }
      }
      if (best === -1) {
        value.offsetBytes = highWater;
        highWater = checkedWasmActivationAdd(highWater, value.sizeBytes,
          `${dtype} liveness arena capacity`);
      } else {
        const selected = free[best];
        value.offsetBytes = selected.offsetBytes;
        selected.offsetBytes += value.sizeBytes;
        selected.sizeBytes -= value.sizeBytes;
        if (selected.sizeBytes === 0) free.splice(best, 1);
      }
      live.push(value);
      regions.set(value.owner, Object.freeze({ ...value }));
    }
    capacityByArena.set(dtype, highWater);
    capacityBytes = checkedWasmActivationAdd(capacityBytes, highWater,
      'liveness arena capacity');
  }
  return Object.freeze({
    capacityByArena,
    capacityBytes,
    uniqueBytes,
    regions,
  });
}

/**
 * Concrete best-fit fragmentation is not monotone. If it exceeds the
 * independently packed maximum-domain arena, project the exact sizes onto the
 * maximum offsets instead of violating the compiled allocation proof.
 */
function planWasmActivationWithinMaximum(
  graph: WasmGraph,
  prepared: WasmPreparedGraph,
  aliases: ReadonlyMap<string, string>,
  sizes: ReadonlyMap<string, number>,
  maximum: WasmActivationLivenessLayout,
): WasmActivationLivenessLayout {
  const concrete = planWasmActivationLiveness(graph, prepared, aliases, sizes);
  const fits = [...concrete.capacityByArena].every(([arena, bytes]) =>
    bytes <= (maximum.capacityByArena.get(arena) ?? -1));
  if (fits) return concrete;
  const regions = new Map<string, WasmActivationLivenessRegion>();
  const capacityByArena = new Map<RuntimeDType, number>();
  let capacityBytes = 0;
  for (const [owner, region] of concrete.regions) {
    const maximumRegion = maximum.regions.get(owner);
    if (!maximumRegion || maximumRegion.dtype !== region.dtype ||
        maximumRegion.birth !== region.birth || maximumRegion.lastUse !== region.lastUse ||
        maximumRegion.sizeBytes < region.sizeBytes) {
      throw new Error(`WASM maximum liveness layout cannot project activation '${owner}'.`);
    }
    const projected = Object.freeze({ ...region, offsetBytes: maximumRegion.offsetBytes });
    regions.set(owner, projected);
    const end = checkedWasmActivationAdd(projected.offsetBytes, projected.sizeBytes,
      `projected activation '${owner}' end`);
    capacityByArena.set(region.dtype, Math.max(capacityByArena.get(region.dtype) ?? 0, end));
  }
  for (const [arena, bytes] of capacityByArena) {
    const maximumBytes = maximum.capacityByArena.get(arena);
    if (maximumBytes === undefined || bytes > maximumBytes) {
      throw new Error(`WASM projected '${arena}' arena exceeds its maximum layout.`);
    }
    capacityBytes = checkedWasmActivationAdd(capacityBytes, bytes,
      'projected liveness arena capacity');
  }
  return Object.freeze({
    capacityByArena,
    capacityBytes,
    uniqueBytes: concrete.uniqueBytes,
    regions,
  });
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function normalizedTargetMatchesConcreteShape(target, shape) {
  if (target === undefined) return true;
  if (!Array.isArray(target) || !Array.isArray(shape) || target.length !== shape.length) {
    return false;
  }
  const symbols = new Map();
  for (let axis = 0; axis < target.length; axis++) {
    const dimension = target[axis];
    if (typeof dimension === 'number') {
      if (!Number.isSafeInteger(dimension) || dimension <= 0 || dimension !== shape[axis]) {
        return false;
      }
      continue;
    }
    if (typeof dimension !== 'string' || !SHAPE_SYMBOL_PATTERN.test(dimension)) return false;
    const previous = symbols.get(dimension);
    if (previous !== undefined && previous !== shape[axis]) return false;
    symbols.set(dimension, shape[axis]);
  }
  return true;
}

function portableElementCount(shape) {
  if (!Array.isArray(shape)) return null;
  let elements = 1;
  for (const dimension of shape) {
    if (!Number.isInteger(dimension) || dimension <= 0 || dimension > WASM_PORTABLE_MAX_U32 ||
        elements > Math.floor(WASM_PORTABLE_MAX_U32 / dimension)) return null;
    elements *= dimension;
  }
  return elements;
}

function portableTensorElements(tensor) {
  if (!tensor) return null;
  const elements = portableElementCount(tensor.shape);
  if (elements == null) return null;
  let bytes;
  try {
    bytes = Tensor.dtypeBytes(tensor.dtype);
  } catch {
    return null;
  }
  return tensor.sizeBytes === elements * bytes ? elements : null;
}

function portable32BitTensor(tensor) {
  return !!tensor && (tensor.dtype === 'float32' || tensor.dtype === 'int32') &&
    portableTensorElements(tensor) != null;
}

function portablePositiveSliceDescriptor(node, input, output) {
  const rank = input?.shape?.length;
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const startsInput = node.params?.starts ?? [];
  const stepsInput = node.params?.steps ??
    (Array.isArray(startsInput) ? startsInput.map(() => 1) : null);
  const axesInput = node.params?.axes ??
    (Array.isArray(startsInput) ? startsInput.map((_, index) => index) : null);
  if (!input || !output || !portable32BitTensor(input) ||
      output.dtype !== input.dtype || !portable32BitTensor(output) ||
      inputElements == null || outputElements == null || !Number.isInteger(rank) ||
      rank < 1 || rank > WASM_PORTABLE_MAX_RANK || output.shape.length !== rank ||
      !Array.isArray(startsInput) || !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
      startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
    throw new Error(`WASM Slice node ${node.id} requires rank-1..8 same-dtype F32/I32 input/output tensors and matching parameter arrays.`);
  }
  const starts = new Array(rank).fill(0);
  const steps = new Array(rank).fill(1);
  const seen = new Set();
  for (let index = 0; index < axesInput.length; index++) {
    let axis = axesInput[index];
    let start = startsInput[index];
    const step = stepsInput[index];
    if (axis < 0) axis += rank;
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank || seen.has(axis) ||
        !Number.isSafeInteger(start) || !Number.isSafeInteger(step) || step <= 0 ||
        step > WASM_PORTABLE_MAX_U32) {
      throw new Error(`WASM Slice node ${node.id} requires unique axes and positive integer steps.`);
    }
    if (start < 0) start += input.shape[axis];
    if (start < 0 || start >= input.shape[axis]) {
      throw new Error(`WASM Slice node ${node.id} start is outside its input axis.`);
    }
    starts[axis] = start;
    steps[axis] = step;
    seen.add(axis);
  }
  for (let axis = 0; axis < rank; axis++) {
    const outputLength = output.shape[axis];
    if (outputLength - 1 > Math.floor((input.shape[axis] - 1 - starts[axis]) / steps[axis])) {
      throw new Error(`WASM Slice node ${node.id} output shape exceeds its input selection.`);
    }
  }
  return { rank, elements: outputElements, starts, steps, dtype: input.dtype };
}

function portableGatherDescriptor(node) {
  const input = node.inputs.input || node.inputs.data;
  const indices = node.inputs.indices;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const indicesElements = portableTensorElements(indices);
  const outputElements = portableTensorElements(output);
  const rank = input?.shape?.length;
  let axis = node.params?.axis ?? 0;
  if (axis < 0) axis += rank;
  if (!input || !indices || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
      indices.dtype !== 'int32' || inputElements == null || indicesElements == null ||
      outputElements == null || !Number.isInteger(rank) || rank < 1 ||
      rank > WASM_PORTABLE_MAX_RANK || !Array.isArray(indices.shape) ||
      indices.shape.length > WASM_PORTABLE_MAX_RANK || !Number.isInteger(axis) ||
      axis < 0 || axis >= rank) {
    throw new Error(`WASM Gather node ${node.id} requires F32 data/output, I32 indices, rank-1..8 data, and a valid axis.`);
  }
  const expectedShape = [
    ...input.shape.slice(0, axis), ...indices.shape, ...input.shape.slice(axis + 1),
  ];
  if (expectedShape.length > WASM_PORTABLE_MAX_RANK || !sameShape(output.shape, expectedShape)) {
    throw new Error(`WASM Gather node ${node.id} output shape must be input[:axis] + indices.shape + input[axis + 1:].`);
  }
  const outer = portableElementCount(input.shape.slice(0, axis));
  const inner = portableElementCount(input.shape.slice(axis + 1));
  const axisSize = input.shape[axis];
  if (outer == null || inner == null ||
      inputElements !== outer * axisSize * inner ||
      outputElements !== outer * indicesElements * inner) {
    throw new Error(`WASM Gather node ${node.id} has invalid tensor element counts.`);
  }
  return { input, indices, output, outer, inner, axisSize, indicesElements, outputElements };
}

function portableGatherElementsDescriptor(node) {
  const input = node.inputs.input || node.inputs.data;
  const indices = node.inputs.indices;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const indicesElements = portableTensorElements(indices);
  const outputElements = portableTensorElements(output);
  const rank = input?.shape?.length;
  let axis = node.params?.axis ?? 0;
  if (axis < 0) axis += rank;
  if (!input || !indices || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
      indices.dtype !== 'int32' || inputElements == null || indicesElements == null ||
      outputElements == null || !Number.isInteger(rank) || rank < 1 ||
      rank > WASM_PORTABLE_MAX_RANK || indices.shape.length !== rank ||
      !sameShape(output.shape, indices.shape) || !Number.isInteger(axis) || axis < 0 || axis >= rank ||
      indices.shape.some((dimension, index) => index !== axis && dimension > input.shape[index]) ||
      indicesElements !== outputElements) {
    throw new Error(`WASM GatherElements node ${node.id} requires matching rank-1..8 I32 index/output shapes and a valid axis.`);
  }
  return { input, indices, output, rank, axis, elements: outputElements };
}

function portableWhereDescriptor(node) {
  const condition = node.opType === 'Where' ? node.inputs.condition : node.inputs.mask;
  const a = node.inputs.a;
  const b = node.inputs.b;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const elements = portableTensorElements(output);
  const conditionRank = condition?.shape?.length;
  const aRank = a?.shape?.length;
  const bRank = b?.shape?.length;
  const outputRank = output?.shape?.length;
  if (!condition || !a || !b || !output || !portable32BitTensor(condition) ||
      !portable32BitTensor(a) || b.dtype !== a.dtype || output.dtype !== a.dtype ||
      !portable32BitTensor(b) || !portable32BitTensor(output) ||
      elements == null || !Number.isInteger(conditionRank) || !Number.isInteger(aRank) ||
      !Number.isInteger(bRank) || !Number.isInteger(outputRank) ||
      conditionRank > WASM_PORTABLE_MAX_RANK || aRank > WASM_PORTABLE_MAX_RANK ||
      bRank > WASM_PORTABLE_MAX_RANK || outputRank > WASM_PORTABLE_MAX_RANK ||
      outputRank !== Math.max(conditionRank, aRank, bRank)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires rank-0..8 same-dtype F32/I32 branches/output and an F32 or I32 condition.`);
  }
  const inputs = [condition, a, b];
  for (let axis = 0; axis < outputRank; axis++) {
    const expected = Math.max(...inputs.map((tensor) => {
      const offset = outputRank - tensor.shape.length;
      return axis < offset ? 1 : tensor.shape[axis - offset];
    }));
    if (output.shape[axis] !== expected || inputs.some((tensor) => {
      const offset = outputRank - tensor.shape.length;
      const dimension = axis < offset ? 1 : tensor.shape[axis - offset];
      return dimension !== 1 && dimension !== expected;
    })) {
      throw new Error(`WASM ${node.opType} node ${node.id} has incompatible broadcast shapes.`);
    }
  }
  if (node.opType === 'Mask' && inputs.some((tensor) => !sameShape(tensor.shape, output.shape))) {
    throw new Error(`WASM Mask node ${node.id} requires exact-shape inputs and output.`);
  }
  return {
    condition, a, b, output, elements, conditionRank, aRank, bRank, outputRank,
    conditionType: wasmDtypeCode(condition.dtype),
    dataType: wasmDtypeCode(a.dtype),
  };
}

function portableOutput(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function portableF32Tensor(tensor) {
  return !!tensor && tensor.dtype === 'float32' && portableTensorElements(tensor) != null;
}

function portableDenseBias(tensor, dOut) {
  if (!tensor) return true;
  return portableF32Tensor(tensor) && tensor.shape.length >= 1 &&
    tensor.shape.at(-1) === dOut &&
    tensor.shape.slice(0, -1).every((dimension) => dimension === 1) &&
    portableTensorElements(tensor) === dOut;
}

function portableUnaryF32Descriptor(node) {
  const input = node.inputs.input || node.inputs.x;
  const output = portableOutput(node);
  const elements = portableTensorElements(input);
  if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
      portableTensorElements(output) !== elements || !sameShape(input.shape, output.shape)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires same-shape F32 input and output.`);
  }
  return { kind: 'unaryF32', input, output, elements };
}

function portableByteQuantizedTensor(tensor) {
  return !!tensor && (tensor.dtype === 'int8' || tensor.dtype === 'uint8') &&
    portableTensorElements(tensor) != null;
}

function immutableQuantizationDescriptor(tensor) {
  if (!portableByteQuantizedTensor(tensor)) return null;
  const descriptor = tensor.quantization;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  const validScale = (value) => typeof value === 'number' && Number.isFinite(value) && value > 0;
  const validZeroPoint = (value) => Number.isInteger(value) && value >= minimum && value <= maximum;
  if (!descriptor || typeof descriptor !== 'object' || !Object.isFrozen(descriptor)) return null;
  if (descriptor.scheme === 'per_tensor') {
    return validScale(descriptor.scale) && validZeroPoint(descriptor.zero_point) ? descriptor : null;
  }
  if (descriptor.scheme === 'per_axis') {
    const axis = descriptor.axis;
    if (!Number.isInteger(axis) || axis < 0 || axis >= tensor.shape.length ||
        !Array.isArray(descriptor.scales) || !Array.isArray(descriptor.zero_points) ||
        !Object.isFrozen(descriptor.scales) || !Object.isFrozen(descriptor.zero_points) ||
        descriptor.scales.length !== tensor.shape[axis] ||
        descriptor.zero_points.length !== tensor.shape[axis] ||
        !descriptor.scales.every(validScale) || !descriptor.zero_points.every(validZeroPoint)) return null;
    return descriptor;
  }
  return null;
}

function portableQLinearDescriptor(node) {
  if (!['QLinear', 'QMatMul', 'QGemm'].includes(node.opType)) return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.a;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const biasElements = portableTensorElements(bias);
  const outputElements = portableTensorElements(output);
  if (!portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(weight) ||
      !portableByteQuantizedTensor(output) || !bias || bias.dtype !== 'int32' ||
      inputElements == null || weightElements == null || biasElements == null ||
      outputElements == null || input.shape.length < 1 ||
      output.shape.length !== input.shape.length ||
      input.shape.slice(0, -1).some((dimension, index) => dimension !== output.shape[index]) ||
      weight.shape.length !== 2 || bias.shape.length !== 1) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires typed [...,d_in] input/output, [d_out,d_in] I8/U8 weights, and I32 bias.`);
  }
  const dIn = input.shape.at(-1);
  const dOut = output.shape.at(-1);
  const rows = inputElements / dIn;
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(dIn) || dIn <= 0 ||
      !Number.isInteger(dOut) || dOut <= 0 || inputElements !== rows * dIn ||
      outputElements !== rows * dOut || !sameShape(weight.shape, [dOut, dIn]) ||
      weightElements !== dOut * dIn || !sameShape(bias.shape, [dOut]) || biasElements !== dOut) {
    throw new Error(`WASM ${node.opType} node ${node.id} has incompatible [d_out,d_in] dimensions or I32 bias.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!inputQuantization || !outputQuantization || !weightQuantization) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires immutable I8/U8 quantization metadata.`);
  }
  if (inputQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
      weightQuantization.scales.length !== dOut || weightQuantization.zero_points.length !== dOut) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires per-tensor input/output and axis-0 per-channel weight quantization.`);
  }
  const f32Scale = (value) => Math.fround(value);
  const inputScale = f32Scale(inputQuantization.scale);
  const outputScale = f32Scale(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, f32Scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires scales representable as positive F32.`);
  }
  if (!weightScales.every((scale) => {
    const multiplier = Math.fround(Math.fround(inputScale * scale) / outputScale);
    return Number.isFinite(multiplier) && multiplier > 0;
  })) {
    throw new Error(
      `WASM ${node.opType} node ${node.id} requantization multiplier is not representable as positive F32.`,
    );
  }
  return {
    kind: 'qlinear', input, weight, bias, output, rows, dIn, dOut,
    inputDtype: wasmDtypeCode(input.dtype), weightDtype: wasmDtypeCode(weight.dtype),
    outputDtype: wasmDtypeCode(output.dtype), inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}

function portableQEmbeddingDescriptor(node) {
  if (node.opType !== 'QEmbedding') return null;
  const input = node.inputs.input;
  const weight = node.inputs.weight;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const outputElements = portableTensorElements(output);
  const outputCount = Object.values(node.outputs || {}).filter(Boolean).length;
  if (Object.keys(node.inputs || {}).length !== 2 || !input || !weight || outputCount !== 1 ||
      input.dtype !== 'int32' || inputElements == null || input.shape.length < 1 ||
      !portableByteQuantizedTensor(weight) || !portableByteQuantizedTensor(output) ||
      weightElements == null || outputElements == null || weight.shape.length !== 2) {
    throw new Error(`WASM QEmbedding node ${node.id} requires preflight-complete I32 [...token] IDs, rank-2 I8/U8 [vocab,hidden] weights, and an I8/U8 output.`);
  }
  const [vocab, hidden] = weight.shape;
  const expectedOutputShape = [...input.shape, hidden];
  if (!Number.isInteger(vocab) || vocab <= 0 || !Number.isInteger(hidden) || hidden <= 0 ||
      !sameShape(output.shape, expectedOutputShape) || weightElements !== vocab * hidden ||
      outputElements !== inputElements * hidden) {
    throw new Error(`WASM QEmbedding node ${node.id} has incompatible ID, [vocab,hidden] weight, or output dimensions.`);
  }
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!outputQuantization || !weightQuantization || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
      weightQuantization.scales.length !== vocab || weightQuantization.zero_points.length !== vocab) {
    throw new Error(`WASM QEmbedding node ${node.id} requires immutable per-tensor output and axis-0 per-row weight quantization metadata.`);
  }
  const outputScale = Math.fround(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, Math.fround);
  if (!Number.isFinite(outputScale) || outputScale <= 0 ||
      !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QEmbedding node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qembedding', input, weight, output, tokens: inputElements, vocab, hidden,
    weightDtype: wasmDtypeCode(weight.dtype), outputDtype: wasmDtypeCode(output.dtype),
    outputScale, outputZeroPoint: outputQuantization.zero_point,
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}

function portableQAddDescriptor(node) {
  if (node.opType !== 'QAdd') return null;
  const a = node.inputs.a || node.inputs.input || node.inputs.x;
  const b = node.inputs.b || node.inputs.y;
  const output = portableOutput(node);
  const elements = portableTensorElements(output);
  const relu = node.params?.relu ?? 0;
  if (!portableByteQuantizedTensor(a) || !portableByteQuantizedTensor(b) ||
      !portableByteQuantizedTensor(output) || elements == null ||
      portableTensorElements(a) !== elements || portableTensorElements(b) !== elements ||
      !sameShape(a.shape, b.shape) || !sameShape(a.shape, output.shape) ||
      !Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error(`WASM QAdd node ${node.id} requires exact-shape I8/U8 input/output tensors.`);
  }
  const aQuantization = immutableQuantizationDescriptor(a);
  const bQuantization = immutableQuantizationDescriptor(b);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!aQuantization || !bQuantization || !outputQuantization ||
      aQuantization.scheme !== 'per_tensor' || bQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QAdd node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const aScale = Math.fround(aQuantization.scale);
  const bScale = Math.fround(bQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (![aScale, bScale, outputScale].every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QAdd node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qadd', a, b, output, elements,
    aDtype: wasmDtypeCode(a.dtype), bDtype: wasmDtypeCode(b.dtype),
    outputDtype: wasmDtypeCode(output.dtype),
    aScale, bScale, outputScale,
    aZeroPoint: aQuantization.zero_point, bZeroPoint: bQuantization.zero_point,
    outputZeroPoint: outputQuantization.zero_point, relu,
  };
}

function portableQSiLUDescriptor(node) {
  if (node.opType !== 'QSiLU') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  if (!input || !output || inputEntries.length !== 1 || inputEntries[0][1] !== input ||
      outputEntries.length !== 1 || input === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null ||
      outputElements !== inputElements || !sameShape(input.shape, output.shape) ||
      Object.keys(node.params || {}).length !== 0) {
    throw new Error(`WASM QSiLU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and no parameters.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QSiLU node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QSiLU node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qsilu', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}

function portableQGELUDescriptor(node) {
  if (node.opType !== 'QGELU') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const parameterNames = Object.keys(node.params || {});
  const canonicalParameters = parameterNames.length === 0 ||
    (parameterNames.length === 1 && parameterNames[0] === 'approximate' &&
      node.params.approximate === 'none');
  if (!input || !output || inputEntries.length !== 1 || inputEntries[0][1] !== input ||
      outputEntries.length !== 1 || input === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null ||
      outputElements !== inputElements || !sameShape(input.shape, output.shape) ||
      !canonicalParameters) {
    throw new Error(`WASM QGELU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and only omitted parameters or approximate='none'.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QGELU node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QGELU node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qgelu', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}

function qGroupNormByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function qGroupNormAffineStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  return storage instanceof Float32Array && storage.byteLength === tensor.sizeBytes &&
    storage.length === tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT;
}

function qGroupNormStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Do this before `_alloc` converts every tensor to a fresh WASM typed view.
// It preserves the portable operator's typed-storage and no-overlap contract
// for graph inputs supplied as ArrayBuffer views, while still allowing normal
// unallocated graph values to receive their expected typed storage.
function preflightPortableQGroupNormStorage(node) {
  if (node.opType !== 'QGroupNorm') return;
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qGroupNormByteStorageIsCanonical(input) || !qGroupNormByteStorageIsCanonical(output) ||
      !qGroupNormAffineStorageIsCanonical(weight) || !qGroupNormAffineStorageIsCanonical(bias) ||
      qGroupNormStorageRangesOverlap(input, output) ||
      qGroupNormStorageRangesOverlap(weight, output) ||
      qGroupNormStorageRangesOverlap(bias, output)) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires canonical typed input/output storage, finite F32 [C] affine storage, and output storage distinct from every input.`);
  }
}

function portableQGroupNormDescriptor(node, affineValues = (tensor) => tensor?.buffer) {
  if (node.opType !== 'QGroupNorm') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
  const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['num_groups', 'eps', 'data_layout'].includes(name));
  const groups = params?.num_groups;
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const [batch, height, width, channels] = input?.shape || [];
  const validAffine = (tensor) => {
    const values = affineValues(tensor);
    return portableF32Tensor(tensor) && sameShape(tensor.shape, [channels]) &&
      tensor.sizeBytes === channels * Float32Array.BYTES_PER_ELEMENT &&
      values instanceof Float32Array && values.length === channels &&
      (tensor.isWeight !== true || values.every(Number.isFinite));
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputNames.length !== 1 || !input || !weight || !bias || !output ||
      input === output || weight === output || bias === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null || outputElements !== inputElements ||
      input.shape.length !== 4 || !sameShape(input.shape, output.shape) || !allowedParameters ||
      !Number.isInteger(groups) || groups <= 0 || !Number.isInteger(channels) || channels <= 0 ||
      channels % groups !== 0 || (params?.data_layout != null && params.data_layout !== 'NHWC') ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0 || !validAffine(weight) || !validAffine(bias)) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires exact input/weight/bias inputs, matching rank-4 NHWC I8/U8 activation tensors, finite F32 [C] affine tensors, and positive num_groups/eps parameters.`);
  }

  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QGroupNorm node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qgroupnorm', input, weight, bias, output,
    batch, height, width, channels, groups, inputScale, outputScale, epsilon,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}

function qLayerNormByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function qLayerNormAffineStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  return storage instanceof Float32Array && storage.byteLength === tensor.sizeBytes &&
    storage.length === tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT;
}

function qLayerNormStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the source-storage contract before `_alloc` gives every tensor a
// separate WASM heap view. This keeps direct graph aliases and raw ArrayBuffer
// affine storage from being hidden by the copy into linear WASM memory.
function preflightPortableQLayerNormStorage(node) {
  if (node.opType !== 'QLayerNorm') return;
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qLayerNormByteStorageIsCanonical(input) || !qLayerNormByteStorageIsCanonical(output) ||
      !qLayerNormAffineStorageIsCanonical(weight) || !qLayerNormAffineStorageIsCanonical(bias) ||
      qLayerNormStorageRangesOverlap(input, output) ||
      qLayerNormStorageRangesOverlap(weight, output) ||
      qLayerNormStorageRangesOverlap(bias, output)) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires canonical typed input/output storage, finite F32 [D] affine storage, and output storage distinct from every input.`);
  }
}

function portableQLayerNormDescriptor(node, affineValues = (tensor) => tensor?.buffer) {
  if (node.opType !== 'QLayerNorm') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
  const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['eps', 'd_model'].includes(name));
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const dModel = input?.shape?.at(-1);
  const validAffine = (tensor) => {
    const values = affineValues(tensor);
    return portableF32Tensor(tensor) && sameShape(tensor.shape, [dModel]) &&
      tensor.sizeBytes === dModel * Float32Array.BYTES_PER_ELEMENT &&
      values instanceof Float32Array && values.length === dModel &&
      (tensor.isWeight !== true || values.every(Number.isFinite));
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputNames.length !== 1 || !input || !weight || !bias || !output ||
      input === output || weight === output || bias === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null || outputElements !== inputElements ||
      input.shape.length < 1 || !sameShape(input.shape, output.shape) || !allowedParameters ||
      !Number.isInteger(dModel) || dModel <= 0 ||
      (params?.d_model != null && (!Number.isInteger(params.d_model) || params.d_model !== dModel)) ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0 || !validAffine(weight) || !validAffine(bias)) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires exact input/weight/bias inputs, matching rank-at-least-1 I8/U8 activation tensors, finite F32 [D] affine tensors, and positive eps with optional d_model matching D.`);
  }

  const rows = inputElements / dModel;
  if (!Number.isInteger(rows) || rows <= 0 || rows > WASM_PORTABLE_MAX_U32 ||
      inputElements !== rows * dModel) {
    throw new Error(`WASM QLayerNorm node ${node.id} has an invalid contiguous [...,D] row layout.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QLayerNorm node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qlayernorm', input, weight, bias, output, rows, dModel, inputScale, outputScale, epsilon,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}

function qSDPAByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function qSDPAInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}

function qSDPAStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Keep source alias checks before `_alloc` gives every tensor a fresh linear
// WASM view. Without this, overlapping graph views would be hidden by copies
// before qsdpa_i8u8 can preserve its no-partial-write boundary.
function preflightPortableQSDPAStorage(node) {
  if (node.opType !== 'QSDPA') return;
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const mask = node.inputs?.mask || null;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qSDPAByteStorageIsCanonical(q) || !qSDPAByteStorageIsCanonical(k) ||
      !qSDPAByteStorageIsCanonical(v) || !qSDPAByteStorageIsCanonical(output) ||
      (mask && !qSDPAInt32StorageIsCanonical(mask)) || qSDPAStorageRangesOverlap(q, output) ||
      qSDPAStorageRangesOverlap(k, output) || qSDPAStorageRangesOverlap(v, output) ||
      (mask && qSDPAStorageRangesOverlap(mask, output))) {
    throw new Error(`WASM QSDPA node ${node.id} requires canonical typed q/k/v/output storage, optional I32 mask storage, and output storage distinct from every input.`);
  }
}

function portableQSDPADescriptor(node) {
  if (node.opType !== 'QSDPA') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
  const mask = node.inputs?.mask;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const qElements = portableTensorElements(q);
  const kElements = portableTensorElements(k);
  const vElements = portableTensorElements(v);
  const outputElements = portableTensorElements(output);
  const maskElements = hasMask ? portableTensorElements(mask) : 0;
  const params = node.params;
  const paramsAreObject = params != null && typeof params === 'object' && !Array.isArray(params);
  const parameterNames = paramsAreObject ? Object.keys(params) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['heads', 'causal', 'scale'].includes(name));
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const seqQ = q?.shape?.[rank - 2];
  const seqKV = k?.shape?.[rank - 2];
  const dModel = q?.shape?.[rank - 1];
  const heads = params?.heads;
  const headDim = dModel / heads;
  const scaleSource = params?.scale ?? 1 / Math.sqrt(headDim);
  const attentionScale = Math.fround(scaleSource);
  const exactInputs = (inputNames.length === 3 && inputNames[0] === 'k' && inputNames[1] === 'q' &&
    inputNames[2] === 'v') || (inputNames.length === 4 && inputNames[0] === 'k' &&
    inputNames[1] === 'mask' && inputNames[2] === 'q' && inputNames[3] === 'v');
  if (!exactInputs || outputNames.length !== 1 || !q || !k || !v || !output ||
      output === q || output === k || output === v || output === mask ||
      !portableByteQuantizedTensor(q) || !portableByteQuantizedTensor(k) ||
      !portableByteQuantizedTensor(v) || !portableByteQuantizedTensor(output) ||
      qElements == null || kElements == null || vElements == null || outputElements == null ||
      (hasMask && (mask?.dtype !== 'int32' || maskElements == null)) ||
      (rank !== 2 && rank !== 3) || k.shape.length !== rank || v.shape.length !== rank ||
      output.shape.length !== rank || !sameShape(q.shape, output.shape) || !allowedParameters ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'heads') ||
      !Number.isInteger(heads) || heads <= 0 ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'causal') || typeof params?.causal !== 'boolean' ||
      !Number.isInteger(headDim) || headDim <= 0 || dModel % 4 !== 0 || headDim % 4 !== 0 ||
      headDim > 64 || typeof scaleSource !== 'number' || !Number.isFinite(scaleSource) ||
      scaleSource <= 0 || !Number.isFinite(attentionScale) || attentionScale <= 0) {
    throw new Error(`WASM QSDPA node ${node.id} requires exact q/k/v and optional I32 mask inputs, rank-2/3 I8/U8 tensors, D/head dimensions divisible by 4 with head_dim <= 64, and explicit heads/causal.`);
  }

  const kBatch = rank === 2 ? 1 : k.shape[0];
  const vBatch = rank === 2 ? 1 : v.shape[0];
  if (kBatch !== batch || vBatch !== batch || k.shape[rank - 1] !== dModel ||
      v.shape[rank - 1] !== dModel || v.shape[rank - 2] !== seqKV ||
      qElements !== batch * seqQ * dModel || kElements !== batch * seqKV * dModel ||
      vElements !== batch * seqKV * dModel || outputElements !== qElements) {
    throw new Error(`WASM QSDPA node ${node.id} has incompatible Q/K/V/output dimensions.`);
  }

  let maskMode = 0;
  if (hasMask) {
    if (mask.shape.length === 1 && mask.shape[0] === seqKV) maskMode = 1;
    // Preserve established B,K precedence if B and Q have the same value.
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === batch) maskMode = 2;
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === seqQ) maskMode = 3;
    else if (mask.shape.length === 3 && mask.shape[0] === batch && mask.shape[1] === seqQ &&
        mask.shape[2] === seqKV) maskMode = 4;
    else throw new Error(`WASM QSDPA node ${node.id} mask requires I32 [K], [B,K], [Q,K], or [B,Q,K] storage.`);
  }

  const qQuantization = immutableQuantizationDescriptor(q);
  const kQuantization = immutableQuantizationDescriptor(k);
  const vQuantization = immutableQuantizationDescriptor(v);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!qQuantization || !kQuantization || !vQuantization || !outputQuantization ||
      qQuantization.scheme !== 'per_tensor' || kQuantization.scheme !== 'per_tensor' ||
      vQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QSDPA node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const qScale = Math.fround(qQuantization.scale);
  const kScale = Math.fround(kQuantization.scale);
  const vScale = Math.fround(vQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const scoreMultiplier = Math.fround(Math.fround(qScale * kScale) * attentionScale);
  const maximumCenteredMagnitude = (tensor, descriptor) => {
    const minimum = tensor.dtype === 'int8' ? -128 : 0;
    const maximum = tensor.dtype === 'int8' ? 127 : 255;
    return Math.max(Math.abs(minimum - descriptor.zero_point), Math.abs(maximum - descriptor.zero_point));
  };
  const maximumRawDot = headDim * maximumCenteredMagnitude(q, qQuantization) *
    maximumCenteredMagnitude(k, kQuantization);
  const maximumScore = Math.fround(Math.fround(maximumRawDot) * scoreMultiplier);
  if (![qScale, kScale, vScale, outputScale, scoreMultiplier].every((value) =>
    Number.isFinite(value) && value > 0) || !Number.isFinite(maximumScore)) {
    throw new Error(`WASM QSDPA node ${node.id} requires scales and score range representable as finite positive F32.`);
  }
  return {
    kind: 'qsdpa', q, k, v, mask: hasMask ? mask : null, output, batch, seqQ, seqKV, dModel, heads,
    headDim, maskMode, causal: params.causal ? 1 : 0, qScale, kScale, vScale, outputScale,
    attentionScale, qZeroPoint: qQuantization.zero_point, kZeroPoint: kQuantization.zero_point,
    vZeroPoint: vQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    qDtype: wasmDtypeCode(q.dtype), kDtype: wasmDtypeCode(k.dtype),
    vDtype: wasmDtypeCode(v.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}

function qArgMaxByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function qArgMaxInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}

function qArgMaxStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the native kernel's no-partial-write contract before per-tensor
// WASM allocation turns overlapping source views into independent copies.
function preflightPortableQArgMaxStorage(node) {
  if (node.opType !== 'QArgMax') return;
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  if (!qArgMaxByteStorageIsCanonical(input) || !qArgMaxInt32StorageIsCanonical(output) ||
      qArgMaxStorageRangesOverlap(input, output)) {
    throw new Error(`WASM QArgMax node ${node.id} requires canonical I8/U8 input and I32 output storage distinct from input storage.`);
  }
}

function portableQArgMaxDescriptor(node) {
  if (node.opType !== 'QArgMax') return null;
  const inputNames = Object.keys(node.inputs || {});
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 1 && Object.prototype.hasOwnProperty.call(params, 'axis') &&
    Number.isInteger(params.axis);
  const rank = input?.shape?.length;
  let axis = params?.axis;
  if (axis < 0) axis += rank;
  const outer = Number.isInteger(axis) ? portableElementCount(input?.shape?.slice(0, axis)) : null;
  const inner = Number.isInteger(axis) ? portableElementCount(input?.shape?.slice(axis + 1)) : null;
  const axisSize = input?.shape?.[axis];
  const expectedOutputShape = Number.isInteger(axis) && Array.isArray(input?.shape)
    ? [...input.shape.slice(0, axis), ...input.shape.slice(axis + 1)] : null;
  const inputQuantization = immutableQuantizationDescriptor(input);
  const inputScale = Math.fround(inputQuantization?.scale);
  if (inputNames.length !== 1 || inputNames[0] !== 'input' || outputNames.length !== 1 ||
      outputNames[0] !== 'out' || !input || !output || input === output ||
      !portableByteQuantizedTensor(input) || inputQuantization?.scheme !== 'per_tensor' ||
      !Number.isFinite(inputScale) || inputScale <= 0 || output.dtype !== 'int32' ||
      output.quantization != null || inputElements == null || outputElements == null ||
      !Number.isInteger(rank) || rank < 2 || rank > WASM_PORTABLE_MAX_RANK || !paramsAreExact ||
      !Number.isInteger(axis) || axis < 0 || axis >= rank || !Number.isInteger(axisSize) ||
      axisSize <= 0 || axisSize > 0x7fffffff || outer == null || inner == null || outer <= 0 || inner <= 0 ||
      inputElements !== outer * axisSize * inner || outputElements !== outer * inner ||
      !sameShape(output.shape, expectedOutputShape) || !qArgMaxByteStorageIsCanonical(input) ||
      !qArgMaxInt32StorageIsCanonical(output) || qArgMaxStorageRangesOverlap(input, output)) {
    throw new Error(`WASM QArgMax node ${node.id} requires exactly { input } -> { out }, immutable per-tensor rank-2..8 I8/U8 input, exact { axis: integer } parameters, and an unquantized I32 output with the axis removed.`);
  }
  return {
    kind: 'qargmax', input, output, axis, outer, axisSize, inner, outputElements,
    inputDtype: wasmDtypeCode(input.dtype),
  };
}

function qMaskedMeanByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function qMaskedMeanInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}

function qMaskedMeanStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the native kernel's typed-storage and no-overlap contract before
// `_alloc` turns graph tensors into disjoint WASM heap views.
function preflightPortableQMaskedMeanStorage(node) {
  if (node.opType !== 'QMaskedMean') return;
  const input = node.inputs?.input;
  const mask = node.inputs?.mask;
  const output = node.outputs?.out;
  if (!qMaskedMeanByteStorageIsCanonical(input) || !qMaskedMeanInt32StorageIsCanonical(mask) ||
      !qMaskedMeanByteStorageIsCanonical(output) || qMaskedMeanStorageRangesOverlap(input, output) ||
      qMaskedMeanStorageRangesOverlap(mask, output)) {
    throw new Error(`WASM QMaskedMean node ${node.id} requires canonical I8/U8 input/output and I32 mask storage, with output storage distinct from input and mask.`);
  }
}

function qMaskedMeanMaximumCenteredMagnitude(dtype, zeroPoint) {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}

function portableQMaskedMeanDescriptor(node) {
  if (node.opType !== 'QMaskedMean') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const mask = node.inputs?.mask;
  const output = node.outputs?.out;
  const inputElements = portableTensorElements(input);
  const maskElements = portableTensorElements(mask);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 0;
  const [batch, sequence, width] = input?.shape || [];
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const inputScale = Math.fround(inputQuantization?.scale);
  const outputScale = Math.fround(outputQuantization?.scale);
  const multiplier = Math.fround(inputScale / outputScale);

  if (inputNames.length !== 2 || inputNames[0] !== 'input' || inputNames[1] !== 'mask' ||
      outputNames.length !== 1 || outputNames[0] !== 'out' || !input || !mask || !output ||
      input === output || mask === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || mask.dtype !== 'int32' || mask.quantization != null ||
      inputElements == null || maskElements == null || outputElements == null ||
      input.shape.length !== 3 || mask.shape.length !== 2 || output.shape.length !== 2 ||
      !paramsAreExact || !Number.isInteger(batch) || !Number.isInteger(sequence) ||
      !Number.isInteger(width) || batch <= 0 || sequence <= 0 || width <= 0 ||
      batch > WASM_PORTABLE_MAX_U32 || sequence > WASM_PORTABLE_MAX_U32 || width > WASM_PORTABLE_MAX_U32 ||
      mask.shape[0] !== batch || mask.shape[1] !== sequence ||
      output.shape[0] !== batch || output.shape[1] !== width ||
      inputElements !== batch * sequence * width || maskElements !== batch * sequence ||
      outputElements !== batch * width || inputQuantization?.scheme !== 'per_tensor' ||
      outputQuantization?.scheme !== 'per_tensor' || !Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0 || !Number.isFinite(multiplier) ||
      multiplier <= 0 || !qMaskedMeanByteStorageIsCanonical(input) ||
      !qMaskedMeanInt32StorageIsCanonical(mask) || !qMaskedMeanByteStorageIsCanonical(output) ||
      qMaskedMeanStorageRangesOverlap(input, output) || qMaskedMeanStorageRangesOverlap(mask, output)) {
    throw new Error(`WASM QMaskedMean node ${node.id} requires exactly { input, mask } -> { out }, immutable per-tensor I8/U8 [B,S,D] input/output [B,D], unquantized I32 [B,S] mask, and no parameters.`);
  }

  const maximumCenteredSum = sequence * qMaskedMeanMaximumCenteredMagnitude(
    input.dtype, inputQuantization.zero_point,
  );
  if (!Number.isSafeInteger(maximumCenteredSum) || maximumCenteredSum > 0x7fffffff) {
    throw new Error(`WASM QMaskedMean node ${node.id} sequence is too large for an exact I32 centered sum.`);
  }
  return {
    kind: 'qmaskedmean', input, mask, output, batch, sequence, width,
    inputScale, inputZeroPoint: inputQuantization.zero_point,
    outputScale, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}

function portableRequantizeLinearDescriptor(node) {
  if (node.opType !== 'RequantizeLinear') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {});
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  if (inputEntries.length !== 1 || !portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(output) ||
      inputElements == null || outputElements !== inputElements) {
    throw new Error(`WASM RequantizeLinear node ${node.id} requires one metadata-only I8/U8 input and an equal-size output.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM RequantizeLinear node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const multiplier = Math.fround(inputScale / outputScale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(`WASM RequantizeLinear node ${node.id} has a non-representable positive scale ratio.`);
  }
  return {
    kind: 'requantizeLinear', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}

function portableQConv2DDescriptor(node) {
  if (node.opType !== 'QConv2D') return null;
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias || null;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const outputElements = portableTensorElements(output);
  const biasElements = bias ? portableTensorElements(bias) : 0;
  if (!portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(weight) ||
      !portableByteQuantizedTensor(output) || inputElements == null || weightElements == null ||
      outputElements == null || input.shape.length !== 4 || weight.shape.length !== 4 ||
      output.shape.length !== 4 || (bias && (bias.dtype !== 'int32' || biasElements == null)) ||
      (node.params?.data_layout && node.params.data_layout !== 'NHWC') ||
      (node.params?.weight_layout && node.params.weight_layout !== 'OHWI')) {
    throw new Error(`WASM QConv2D node ${node.id} requires canonical NHWC I8/U8 activations and OHWI I8/U8 weights.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!inputQuantization || !outputQuantization || !weightQuantization ||
      inputQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis') {
    throw new Error(`WASM QConv2D node ${node.id} requires immutable per-tensor activation and per-axis weight metadata.`);
  }
  const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
  const [outputChannels, kernelHeight, kernelWidth, inputPerGroup] = weight.shape;
  const [outputBatch, outputHeight, outputWidth, outputChannelsFromOutput] = output.shape;
  const groups = node.params?.groups ?? 1;
  const [strideY, strideX] = portablePairParameter(node, 'stride', 1);
  const [dilationY, dilationX] = portablePairParameter(node, 'dilation', 1);
  const [paddingY, paddingX] = portablePairParameter(node, 'padding', 0, true);
  const pads = node.params?.pads || [paddingY, paddingX, paddingY, paddingX];
  const relu = node.params?.relu ?? 0;
  if (!Number.isInteger(groups) || groups <= 0 || groups > WASM_PORTABLE_MAX_U32 ||
      inputChannels !== inputPerGroup * groups || outputChannels !== outputChannelsFromOutput ||
      outputBatch !== batch || outputChannels % groups !== 0 || !Array.isArray(pads) ||
      pads.length !== 4 || pads.some((value) => !Number.isInteger(value) || value < 0 ||
        value > WASM_PORTABLE_MAX_U32) || !Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error(`WASM QConv2D node ${node.id} has incompatible grouped-convolution parameters.`);
  }
  const effectiveHeight = dilationY * (kernelHeight - 1) + 1;
  const effectiveWidth = dilationX * (kernelWidth - 1) + 1;
  const paddedHeight = inputHeight + pads[0] + pads[2];
  const paddedWidth = inputWidth + pads[1] + pads[3];
  if (![effectiveHeight, effectiveWidth, paddedHeight, paddedWidth].every(Number.isSafeInteger) ||
      paddedHeight < effectiveHeight || paddedWidth < effectiveWidth) {
    throw new Error(`WASM QConv2D node ${node.id} has non-representable stride, padding, or dilation geometry.`);
  }
  const expectedHeight = Math.floor((paddedHeight - effectiveHeight) / strideY) + 1;
  const expectedWidth = Math.floor((paddedWidth - effectiveWidth) / strideX) + 1;
  if (expectedHeight !== outputHeight || expectedWidth !== outputWidth ||
      weightElements !== outputChannels * kernelHeight * kernelWidth * inputPerGroup ||
      inputElements !== batch * inputHeight * inputWidth * inputChannels ||
      outputElements !== batch * outputHeight * outputWidth * outputChannels) {
    throw new Error(`WASM QConv2D node ${node.id} has an output shape incompatible with its canonical descriptor.`);
  }
  if (bias && (!sameShape(bias.shape, [outputChannels]) || biasElements !== outputChannels)) {
    throw new Error(`WASM QConv2D node ${node.id} bias must be an I32 vector with one value per output channel.`);
  }
  if (weightQuantization.axis !== 0 || weightQuantization.scales.length !== outputChannels ||
      weightQuantization.zero_points.length !== outputChannels) {
    throw new Error(`WASM QConv2D node ${node.id} requires per_axis weight quantization along output-channel axis 0.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, Math.fround);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QConv2D node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qconv2d', input, weight, bias, output,
    batch, inputHeight, inputWidth, inputChannels,
    outputHeight, outputWidth, outputChannels, kernelHeight, kernelWidth, inputPerGroup,
    strideY, strideX, dilationY, dilationX,
    paddingTop: pads[0], paddingLeft: pads[1], paddingBottom: pads[2], paddingRight: pads[3],
    groups, relu, inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), weightDtype: wasmDtypeCode(weight.dtype),
    outputDtype: wasmDtypeCode(output.dtype),
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}

function sameQuantizationDescriptor(left, right) {
  if (!left || !right || left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor') {
    return left.scale === right.scale && left.zero_point === right.zero_point;
  }
  if (left.scheme === 'per_axis') {
    return left.axis === right.axis && left.scales.length === right.scales.length &&
      left.zero_points.length === right.zero_points.length &&
      left.scales.every((value, index) => value === right.scales[index]) &&
      left.zero_points.every((value, index) => value === right.zero_points[index]);
  }
  return false;
}

function portableQuantizedCommonDescriptor(node, tensors) {
  const requested = tensors.some((tensor) => tensor &&
    (tensor.dtype === 'int8' || tensor.dtype === 'uint8'));
  if (!requested) return null;
  if (tensors.some((tensor) => !portableByteQuantizedTensor(tensor))) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires same-dtype I8/U8 tensors.`);
  }
  const dtype = tensors[0].dtype;
  if (tensors.some((tensor) => tensor.dtype !== dtype)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires one I8/U8 storage dtype.`);
  }
  const descriptors = tensors.map(immutableQuantizationDescriptor);
  if (descriptors.some((descriptor) => !descriptor)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires immutable quantization descriptors.`);
  }
  if (descriptors.some((descriptor) => !sameQuantizationDescriptor(descriptors[0], descriptor))) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires identical quantization descriptors.`);
  }
  return { dtype: wasmDtypeCode(dtype), quantization: descriptors[0] };
}

function portableQuantizedShapeDescriptor(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const common = portableQuantizedCommonDescriptor(node, [input, output]);
  if (!common) return null;
  const elements = portableTensorElements(input);
  if (elements == null || portableTensorElements(output) !== elements) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires equal-size I8/U8 tensors.`);
  }
  return { kind: 'quantizedShapeCopy', input, output, elements, dtype: common.dtype };
}

function portablePairParameter(node, name, fallback, allowZero = false) {
  const value = node.params?.[name];
  let pair;
  if (value == null) pair = [fallback, fallback];
  else if (Array.isArray(value)) {
    if (value.length < 1 || value.length > 2) {
      throw new Error(`WASM ${node.opType} node ${node.id} ${name} must be a scalar or one/two-element array.`);
    }
    pair = [value[0], value[1] ?? value[0]];
  } else pair = [value, value];
  const minimum = allowZero ? 0 : 1;
  if (pair.some((dimension) => !Number.isInteger(dimension) || dimension < minimum ||
      dimension > WASM_PORTABLE_MAX_U32)) {
    throw new Error(`WASM ${node.opType} node ${node.id} ${name} must contain ${allowZero ? 'non-negative' : 'positive'} U32 values.`);
  }
  return pair;
}

function portableFalseOrAbsent(value) {
  return value == null || value === false || value === 0;
}

function portableF32Pool2DDescriptor(node) {
  if (node.opType !== 'MaxPool2D' && node.opType !== 'AveragePool2D') return null;
  const operation = `WASM ${node.opType} node ${node.id}`;
  const input = node.inputs?.input || node.inputs?.x || node.inputs?.data;
  const output = portableOutput(node);
  const params = node.params ?? {};
  if (!portableF32Tensor(input) || !portableF32Tensor(output) ||
      input.shape.length !== 4 || output.shape.length !== 4 ||
      input.shape[0] !== output.shape[0] || input.shape[3] !== output.shape[3]) {
    throw new Error(`${operation} requires compatible F32 NHWC input/output tensors.`);
  }
  if (params == null || typeof params !== 'object' || Array.isArray(params) ||
      (params.data_layout != null && params.data_layout !== 'NHWC') ||
      !portableFalseOrAbsent(params.ceil_mode)) {
    throw new Error(`${operation} requires canonical NHWC floor-window parameters.`);
  }
  if (node.opType === 'AveragePool2D' &&
      (!portableFalseOrAbsent(params.count_include_pad) ||
       (params.auto_pad != null && params.auto_pad !== '' && params.auto_pad !== 'NOTSET'))) {
    throw new Error(`${operation} does not support count_include_pad or automatic padding.`);
  }
  const [kernelY, kernelX] = spatialPair(
    params.kernel, 1, operation, 'kernel', false, true,
  );
  const [strideY, strideX] = spatialPair(
    params.stride, 1, operation, 'stride', false,
  );
  const [dilationY, dilationX] = spatialPair(
    params.dilation, 1, operation, 'dilation', false,
  );
  if (dilationY !== 1 || dilationX !== 1) {
    throw new Error(`${operation} supports only unit dilation.`);
  }
  const pads = fullSpatialPads(params, operation, node.opType === 'AveragePool2D');
  const [batch, inputHeight, inputWidth, channels] = input.shape;
  const outputHeight = checkedWindowOutput(
    inputHeight, kernelY, strideY, pads[0], pads[2], 1, `${operation} output height`,
  );
  const outputWidth = checkedWindowOutput(
    inputWidth, kernelX, strideX, pads[1], pads[3], 1, `${operation} output width`,
  );
  if (!sameShape(output.shape, [batch, outputHeight, outputWidth, channels])) {
    throw new Error(`${operation} output shape is incompatible with its canonical descriptor.`);
  }
  return {
    kind: 'f32Pool2D', opType: node.opType, input, output,
    batch, inputHeight, inputWidth, channels, outputHeight, outputWidth,
    kernelY, kernelX, strideY, strideX, paddingY: pads[0], paddingX: pads[1],
  };
}

function portableQuantizedNearestParameters(node) {
  const params = node.params || {};
  if (node.opType === 'Resize') {
    if (params.mode !== 'nearest') {
      throw new Error(`WASM Resize node ${node.id} only supports nearest I8/U8 resize.`);
    }
  } else if (params.mode != null && params.mode !== 'nearest') {
    throw new Error(`WASM ResizeNearest2D node ${node.id} only supports mode "nearest" for I8/U8 resize.`);
  }
  if (params.coordinate_transform_mode != null) {
    throw new Error(`WASM ${node.opType} node ${node.id} does not define coordinate_transform_mode; use coordinate_transformation_mode.`);
  }
  if (params.coordinate_transformation_mode != null &&
      params.coordinate_transformation_mode !== 'asymmetric') {
    throw new Error(`WASM ${node.opType} node ${node.id} only supports coordinate_transformation_mode "asymmetric" for I8/U8 resize.`);
  }
  if (params.nearest_mode != null && params.nearest_mode !== 'floor') {
    throw new Error(`WASM ${node.opType} node ${node.id} only supports nearest_mode "floor" for I8/U8 resize.`);
  }
  if (!portableFalseOrAbsent(params.align_corners) || !portableFalseOrAbsent(params.antialias)) {
    throw new Error(`WASM ${node.opType} node ${node.id} does not support align_corners or antialias for I8/U8 resize.`);
  }
}

function portableQuantizedSpatialDescriptor(node, kind) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const common = portableQuantizedCommonDescriptor(node, [input, output]);
  if (!common) return null;
  if (input.shape.length !== 4 || output.shape.length !== 4 ||
      input.shape[0] !== output.shape[0] || input.shape[3] !== output.shape[3]) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires I8/U8 NHWC input/output tensors.`);
  }
  if (common.quantization.scheme === 'per_axis' &&
      common.quantization.axis !== 0 && common.quantization.axis !== 3) {
    throw new Error(`WASM ${node.opType} node ${node.id} only supports per-axis I8/U8 metadata on batch or channel axes.`);
  }
  const [batch, inputHeight, inputWidth, channels] = input.shape;
  const outputHeight = output.shape[1];
  const outputWidth = output.shape[2];
  if (kind === 'nearestResize') {
    portableQuantizedNearestParameters(node);
    return {
      kind: 'quantizedNearestResize', input, output, dtype: common.dtype,
      batch, inputHeight, inputWidth, channels, outputHeight, outputWidth,
    };
  }
  if (!portableFalseOrAbsent(node.params?.ceil_mode)) {
    throw new Error(`WASM MaxPool2D node ${node.id} does not support ceil_mode for I8/U8 tensors.`);
  }
  const [kernelY, kernelX] = portablePairParameter(node, 'kernel', null);
  const [strideY, strideX] = portablePairParameter(node, 'stride', 1);
  const [paddingY, paddingX] = portablePairParameter(node, 'padding', 0, true);
  const pads = node.params?.pads ?? [paddingY, paddingX, paddingY, paddingX];
  if (!Array.isArray(pads) || pads.length !== 4 || pads.some((value) =>
    !Number.isInteger(value) || value < 0 || value > WASM_PORTABLE_MAX_U32)) {
    throw new Error(`WASM MaxPool2D node ${node.id} requires four non-negative top/left/bottom/right pads.`);
  }
  if (node.params?.dilation != null) {
    const [dilationY, dilationX] = portablePairParameter(node, 'dilation', 1);
    if (dilationY !== 1 || dilationX !== 1) {
      throw new Error(`WASM MaxPool2D node ${node.id} only supports unit I8/U8 dilation.`);
    }
  }
  const availableHeight = inputHeight + pads[0] + pads[2] - kernelY;
  const availableWidth = inputWidth + pads[1] + pads[3] - kernelX;
  const expectedHeight = availableHeight < 0 ? 0 : Math.floor(availableHeight / strideY) + 1;
  const expectedWidth = availableWidth < 0 ? 0 : Math.floor(availableWidth / strideX) + 1;
  if (outputHeight !== expectedHeight || outputWidth !== expectedWidth) {
    throw new Error(`WASM MaxPool2D node ${node.id} output shape is incompatible with its I8/U8 descriptor.`);
  }
  return {
    kind: 'quantizedMaxPool', input, output, dtype: common.dtype,
    batch, inputHeight, inputWidth, channels, outputHeight, outputWidth,
    kernelY, kernelX, strideY, strideX, paddingY: pads[0], paddingX: pads[1],
  };
}

function portableQuantizedConcatDescriptor(node) {
  const output = portableOutput(node);
  const rank = output?.shape?.length;
  let axis = node.params?.axis ?? 0;
  const entries = portableConcatEntries(node);
  const common = portableQuantizedCommonDescriptor(node, [...entries.map(([, tensor]) => tensor), output]);
  if (!common) return null;
  if (node.params?.sigmoid) {
    throw new Error(`WASM Concat node ${node.id} cannot fuse sigmoid with raw I8/U8 storage.`);
  }
  if (axis < 0) axis += rank;
  if (!Number.isInteger(rank) || rank < 1 || rank > WASM_PORTABLE_MAX_RANK ||
      !Number.isInteger(axis) || axis < 0 || axis >= rank || entries.length === 0) {
    throw new Error(`WASM Concat node ${node.id} requires rank-1..8 I8/U8 tensors and a valid axis.`);
  }
  const inner = portableElementCount(output.shape.slice(axis + 1));
  const outer = portableElementCount(output.shape.slice(0, axis));
  const outputAxis = output.shape[axis];
  let summedAxis = 0;
  for (const [, tensor] of entries) {
    if (tensor.shape.length !== rank || tensor.shape.some((dimension, index) =>
      index !== axis && dimension !== output.shape[index])) {
      throw new Error(`WASM Concat node ${node.id} has incompatible I8/U8 input shapes.`);
    }
    summedAxis += tensor.shape[axis];
  }
  if (summedAxis !== outputAxis || inner == null || outer == null) {
    throw new Error(`WASM Concat node ${node.id} input axes do not match its I8/U8 output.`);
  }
  return { kind: 'quantizedConcat', output, entries, axis, inner, outer, outputAxis, dtype: common.dtype };
}

function portableQuantizedNodeDescriptor(node) {
  if (['Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Identity'].includes(node.opType)) {
    return portableQuantizedShapeDescriptor(node);
  }
  if (node.opType === 'MaxPool2D') return portableQuantizedSpatialDescriptor(node, 'maxPool');
  if (node.opType === 'ResizeNearest2D' || node.opType === 'Resize') {
    return portableQuantizedSpatialDescriptor(node, 'nearestResize');
  }
  if (node.opType === 'Concat' || node.opType === 'Concat2') return portableQuantizedConcatDescriptor(node);
  return null;
}

function portableF32ElementwiseDescriptor(node) {
  const a = node.inputs.a;
  const b = node.inputs.b;
  const output = portableOutput(node);
  const aElements = portableTensorElements(a);
  const bElements = portableTensorElements(b);
  const elements = portableTensorElements(output);
  const aRank = a?.shape?.length;
  const bRank = b?.shape?.length;
  const outputRank = output?.shape?.length;
  const kinds = { Add: 0, Mul: 1, Sub: 2, Div: 3 };
  if (!Object.prototype.hasOwnProperty.call(kinds, node.opType) || !portableF32Tensor(a) || !portableF32Tensor(b) ||
      !portableF32Tensor(output) || !Number.isInteger(aRank) || !Number.isInteger(bRank) ||
      !Number.isInteger(outputRank) || aRank < 1 || bRank < 1 || outputRank < 1 ||
      aRank > WASM_PORTABLE_MAX_RANK || bRank > WASM_PORTABLE_MAX_RANK ||
      outputRank > WASM_PORTABLE_MAX_RANK || outputRank !== Math.max(aRank, bRank)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires rank-1..8 F32 tensors.`);
  }
  const aOffset = outputRank - aRank;
  const bOffset = outputRank - bRank;
  for (let dimension = 0; dimension < outputRank; dimension++) {
    const aDimension = dimension < aOffset ? 1 : a.shape[dimension - aOffset];
    const bDimension = dimension < bOffset ? 1 : b.shape[dimension - bOffset];
    const expected = Math.max(aDimension, bDimension);
    if ((aDimension !== 1 && bDimension !== 1 && aDimension !== bDimension) ||
        output.shape[dimension] !== expected) {
      throw new Error(`WASM ${node.opType} node ${node.id} has incompatible broadcast shapes.`);
    }
  }
  if (aElements == null || bElements == null || elements == null) {
    throw new Error(`WASM ${node.opType} node ${node.id} has invalid tensor storage.`);
  }
  return { a, b, output, aRank, bRank, outputRank, elements, operation: kinds[node.opType] };
}

function portableExpandDescriptor(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputRank = input?.shape?.length;
  const outputRank = output?.shape?.length;
  const elements = portableTensorElements(output);
  const byteDescriptor = portableQuantizedCommonDescriptor(node, [input, output]);
  const byteStorage = byteDescriptor?.quantization?.scheme === 'per_tensor';
  const plainStorage = portable32BitTensor(input) && output?.dtype === input?.dtype &&
    portable32BitTensor(output);
  const params = node.params ?? {};
  const paramsAreRecord = params !== null && typeof params === 'object' && !Array.isArray(params);
  const paramNames = paramsAreRecord ? Reflect.ownKeys(params) : [];
  const validParams = paramsAreRecord &&
    paramNames.every((name) => name === 'shape') &&
    normalizedTargetMatchesConcreteShape(params.shape, output?.shape);
  if ((!plainStorage && !byteStorage) || !Number.isInteger(inputRank) ||
      !Number.isInteger(outputRank) || inputRank < 1 || outputRank < inputRank ||
      outputRank > WASM_PORTABLE_MAX_RANK || elements == null || input === output ||
      !validParams) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires distinct rank-1..8 same-dtype F32/I32 or descriptor-preserving per-tensor I8/U8 tensors.`);
  }
  const offset = outputRank - inputRank;
  for (let dimension = 0; dimension < outputRank; dimension++) {
    const source = dimension < offset ? 1 : input.shape[dimension - offset];
    if (source !== 1 && source !== output.shape[dimension]) {
      throw new Error(`WASM ${node.opType} node ${node.id} has incompatible broadcast shapes.`);
    }
  }
  return {
    input, output, inputRank, outputRank, elements, dtype: input.dtype,
    byteStorage,
  };
}

function portableTransposeDescriptor(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const rank = input?.shape?.length;
  const elements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const byteDescriptor = portableQuantizedCommonDescriptor(node, [input, output]);
  const permInput = node.params?.perm || (Array.isArray(input?.shape)
    ? Array.from({ length: input.shape.length }, (_, index) => input.shape.length - 1 - index)
    : null);
  const validStorage = byteDescriptor
    ? byteDescriptor.quantization.scheme === 'per_tensor'
    : portable32BitTensor(input) && output?.dtype === input?.dtype &&
      portable32BitTensor(output);
  if (!validStorage || !Number.isInteger(rank) ||
      rank < 1 || rank > WASM_PORTABLE_MAX_RANK || !Array.isArray(permInput) ||
      permInput.length !== rank || elements == null || outputElements !== elements) {
    throw new Error(`WASM Transpose node ${node.id} requires equal-size rank-1..8 same-dtype F32/I32/I8/U8 tensors.`);
  }
  const seen = new Set();
  const perm = permInput.map((axis) => {
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank || seen.has(axis)) {
      throw new Error(`WASM Transpose node ${node.id} has an invalid permutation.`);
    }
    seen.add(axis);
    return axis;
  });
  if (!sameShape(output.shape, perm.map((axis) => input.shape[axis]))) {
    throw new Error(`WASM Transpose node ${node.id} output shape does not match its permutation.`);
  }
  return {
    input, output, rank, elements, perm, dtype: input.dtype,
    byteDtype: byteDescriptor?.dtype ?? null,
  };
}

function portableConcatEntries(node): Array<[string, Tensor]> {
  const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const seen = new Set();
  const entries: Array<[string, Tensor]> = [];
  for (const key of preferred) {
    if (node.inputs[key]) {
      entries.push([key, node.inputs[key]]);
      seen.add(key);
    }
  }
  entries.push(...(Object.entries(node.inputs || {}) as Array<[string, Tensor]>)
    .filter(([key, tensor]) => tensor && !seen.has(key))
    .sort(([left], [right]) => {
      const leftIndex = /^input(\d+)$/.exec(left);
      const rightIndex = /^input(\d+)$/.exec(right);
      if (leftIndex && rightIndex) return Number(leftIndex[1]) - Number(rightIndex[1]);
      return left.localeCompare(right);
    }));
  return entries;
}

function portableConcatDescriptor(node) {
  const output = portableOutput(node);
  const rank = output?.shape?.length;
  let axis = node.params?.axis ?? 0;
  const entries = portableConcatEntries(node);
  if (axis < 0) axis += rank;
  if (!portable32BitTensor(output) || !Number.isInteger(rank) || rank < 1 ||
      rank > WASM_PORTABLE_MAX_RANK || !Number.isInteger(axis) || axis < 0 || axis >= rank ||
      entries.length === 0 || entries.some(([, tensor]) =>
        !portable32BitTensor(tensor) || tensor.dtype !== output.dtype) ||
      (output.dtype === 'int32' && node.params?.sigmoid)) {
    throw new Error(`WASM Concat node ${node.id} requires rank-1..8 same-dtype F32/I32 tensors and a valid axis.`);
  }
  const inner = portableElementCount(output.shape.slice(axis + 1));
  const outer = portableElementCount(output.shape.slice(0, axis));
  const outputAxis = output.shape[axis];
  let summedAxis = 0;
  for (const [, tensor] of entries) {
    if (tensor.shape.length !== rank || tensor.shape.some((dimension, index) =>
      index !== axis && dimension !== output.shape[index])) {
      throw new Error(`WASM Concat node ${node.id} has incompatible input shapes.`);
    }
    summedAxis += tensor.shape[axis];
  }
  if (summedAxis !== outputAxis || inner == null || outer == null) {
    throw new Error(`WASM Concat node ${node.id} input axes do not match its output.`);
  }
  return {
    output, entries, axis, inner, outer, outputAxis,
    sigmoid: !!node.params?.sigmoid, dtype: output.dtype,
  };
}

function portableSplitDescriptor(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  // Object insertion order is the persisted Split output order (out10 must not
  // be moved ahead of out2).
  const outputEntries = Object.entries(node.outputs || {}) as Array<[string, Tensor]>;
  const rank = input?.shape?.length;
  let axis = node.params?.axis ?? 0;
  if (axis < 0) axis += rank;
  if (!portable32BitTensor(input) || !Number.isInteger(rank) || rank < 1 || rank > WASM_PORTABLE_MAX_RANK ||
      !Number.isInteger(axis) || axis < 0 || axis >= rank || outputEntries.length === 0 ||
      outputEntries.some(([, tensor]) =>
        !portable32BitTensor(tensor) || tensor.dtype !== input.dtype)) {
    throw new Error(`WASM Split node ${node.id} requires rank-1..8 same-dtype F32/I32 tensors and a valid axis.`);
  }
  const inputAxis = input.shape[axis];
  if (inputAxis % outputEntries.length !== 0) {
    throw new Error(`WASM Split node ${node.id} requires equal-sized output slices.`);
  }
  const outputAxis = inputAxis / outputEntries.length;
  const outer = portableElementCount(input.shape.slice(0, axis));
  const inner = portableElementCount(input.shape.slice(axis + 1));
  for (const [, output] of outputEntries) {
    const expected = [...input.shape];
    expected[axis] = outputAxis;
    if (!sameShape(output.shape, expected)) {
      throw new Error(`WASM Split node ${node.id} output shape is incompatible with equal-sized slices.`);
    }
  }
  return { input, outputEntries, inputAxis, outputAxis, outer, inner, dtype: input.dtype };
}

function portableGroupNormDescriptor(node) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight || node.inputs.scale;
  const bias = node.inputs.bias;
  const output = portableOutput(node);
  const groups = node.params?.num_groups;
  const eps = node.params?.eps ?? 1e-5;
  if (!portableF32Tensor(input) || !portableF32Tensor(weight) || !portableF32Tensor(bias) ||
      !portableF32Tensor(output) || input.shape.length !== 4 || !sameShape(input.shape, output.shape) ||
      !Number.isInteger(groups) || groups <= 0 || !Number.isFinite(eps) || eps <= 0) {
    throw new Error(`WASM GroupNorm node ${node.id} requires F32 rank-4 NHWC input, affine tensors, and output.`);
  }
  const [batch, height, width, channels] = input.shape;
  if (channels % groups !== 0 || !sameShape(weight.shape, [channels]) || !sameShape(bias.shape, [channels])) {
    throw new Error(`WASM GroupNorm node ${node.id} has incompatible groups or affine tensor shapes.`);
  }
  return { input, weight, bias, output, batch, height, width, channels, groups, eps };
}

function portableMoERouterDescriptor(node) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight || node.inputs.router_weight;
  const bias = node.inputs.bias || null;
  const indices = node.outputs.indices || node.outputs.expert_indices;
  const routeWeights = node.outputs.weights || node.outputs.expert_weights;
  const rows = portableElementCount(input?.shape?.slice(0, -1));
  const dModel = input?.shape?.at(-1);
  const experts = node.params?.num_experts ?? weight?.shape?.at(-1);
  const topK = node.params?.top_k ?? indices?.shape?.at(-1) ?? 2;
  const temperature = node.params?.temperature ?? 1;
  if (!portableF32Tensor(input) || !portableF32Tensor(weight) || !portableF32Tensor(indices) ||
      !portableF32Tensor(routeWeights) || (bias && !portableF32Tensor(bias)) || rows == null || !Number.isInteger(rows) ||
      !Number.isInteger(dModel) || !Number.isInteger(experts) || !Number.isInteger(topK) || rows <= 0 ||
      dModel <= 0 || experts <= 0 || topK <= 0 || topK > experts || !Number.isFinite(temperature) ||
      temperature <= 0 || !sameShape(weight.shape, [dModel, experts]) ||
      portableTensorElements(indices) !== rows * topK || portableTensorElements(routeWeights) !== rows * topK ||
      (bias && !sameShape(bias.shape, [experts]))) {
    throw new Error(`WASM MoERouter node ${node.id} requires canonical F32 router tensors.`);
  }
  return { input, weight, bias, indices, routeWeights, rows, dModel, experts, topK,
    temperature, normalize: node.params?.normalize !== false ? 1 : 0 };
}

function portableMoELinearDescriptor(node) {
  const input = node.inputs.input || node.inputs.x;
  const expertWeight = node.inputs.expert_weight || node.inputs.weight;
  const expertBias = node.inputs.expert_bias || node.inputs.bias || null;
  const routeIndices = node.inputs.route_indices || node.inputs.indices;
  const routeWeights = node.inputs.route_weights || node.inputs.weights;
  const output = portableOutput(node);
  const rows = portableElementCount(input?.shape?.slice(0, -1));
  const dIn = input?.shape?.at(-1);
  const dOut = output?.shape?.at(-1);
  const experts = expertWeight?.shape?.[0];
  const topK = routeIndices?.shape?.at(-1);
  if (!portableF32Tensor(input) || !portableF32Tensor(expertWeight) || !portableF32Tensor(routeIndices) ||
      !portableF32Tensor(routeWeights) || !portableF32Tensor(output) ||
      (expertBias && !portableF32Tensor(expertBias)) || rows == null || !Number.isInteger(rows) ||
      !Number.isInteger(dIn) || !Number.isInteger(dOut) || !Number.isInteger(experts) || !Number.isInteger(topK) ||
      rows <= 0 || dIn <= 0 || dOut <= 0 || experts <= 0 || topK <= 0 || topK > experts ||
      !sameShape(expertWeight.shape, [experts, dIn, dOut]) ||
      portableTensorElements(routeIndices) !== rows * topK || portableTensorElements(routeWeights) !== rows * topK ||
      portableTensorElements(output) !== rows * dOut || (expertBias && !sameShape(expertBias.shape, [experts, dOut]))) {
    throw new Error(`WASM MoELinear node ${node.id} requires canonical F32 expert and route tensors.`);
  }
  const residentSlots = node.residentSlots || null;
  if (residentSlots && (residentSlots.length !== experts ||
      !residentSlots.every((slot, index) =>
        Number.isInteger(slot) && slot >= 0 &&
        (index === 0 || slot > residentSlots[index - 1])))) {
    throw new Error(
      `WASM MoELinear node ${node.id} requires ascending resident slot ids ` +
      `matching its ${experts} staged experts.`);
  }
  return { input, expertWeight, expertBias, routeIndices, routeWeights, output,
    rows, dIn, dOut, experts, topK, residentSlots };
}


export class WasmEngine extends BackendEngine {
  declare graph: WasmGraph;
  preparedGraph!: WasmPreparedGraph;
  compiledTopologyRevision = 0;
  static _newCompileProfile(): WasmCompileProfile {
    return {
      preflightMs: 0, tensorAllocMs: 0, descriptorMs: 0, packWeightMs: 0,
      viewRefreshMs: 0, metadataAllocMs: 0, totalMs: 0, unattributedMs: 0,
      growMs: 0, growCount: 0, growPages: 0,
      allocBytesCount: 0, viewRefreshCount: 0, descriptorCount: 0,
      tensorCopyBytes: 0, packBytes: 0, finalHeapBytes: 0,
    };
  }

  readonly wasmModule: WasmInstantiation;
  readonly api: WasmKernelExports;
  readonly mem: WebAssembly.Memory;
  readonly relaxedApi: RelaxedSimdExports | null;
  relaxedSimdEnabled: boolean;
  qbatchMatMulSimdEnabled: boolean;
  readonly pointers: Map<string, number>;
  readonly f32PackedWeights: Map<string, PackedF32WeightDescriptor>;
  readonly q8PackedWeights: Map<string, PackedQ8WeightDescriptor>;
  readonly f32PackedNodes: Map<WasmNode, PackedF32WeightDescriptor>;
  readonly tensorCapacities: Map<string, WasmTensorCapacity>;
  readonly tensorMaximumBytes: Map<string, number>;
  nodeMetadata!: Map<WasmNode, WasmNodeMetadata>;
  qconvIm2ColPointer: number;
  qconvIm2ColBytes: number;
  compiledWeightRevision = 0;
  _variantArenaMark: number | null = null;
  _activationArenaMark: number | null = null;
  _metadataArenaMark: number | null = null;
  _allocationMeasurement: WasmAllocationMeasurement | null = null;
  _variantMetadataStaging: WasmStagedMetadataWrite[] | null = null;
  _currentShapeSignature: string | null = null;
  _persistentWeightBytes = 0;
  _activationStorage: WasmActivationStorage = 'persistent';
  _activationAliases: ReadonlyMap<string, string> = new Map();
  _maximumLivenessLayout: WasmActivationLivenessLayout | null = null;
  _activationArenaCapacities: ReadonlyMap<RuntimeDType, number> = new Map();
  _activationCapacityBytes = 0;
  _activationCapacityHighWaterBytes = 0;
  _activationRequiredBytes = 0;
  _activationGrowCount = 0;
  _capacityGrowthFactor = 2;
  _variantBytes = 0;
  _variantHighWaterBytes = 0;
  _heapHighWaterBytes = 0;
  _memoryGrowCount = 0;
  _variantRebindCount = 0;
  _f32PackCount = 0;
  _q8PackCount = 0;
  _disposed = false;
  /* Non-null only while an opted-in compile() is running, so every
   * instrumentation site costs one field read when profiling is off. */
  _compileProfile: WasmCompileProfile | null = null;

  constructor(
    wasmModule: WasmInstantiation,
    relaxedSimdModule: Readonly<RelaxedSimdInstantiation> | null = null,
  ) {
    super('wasm', {
      incrementalExecution: true,
      incrementalRows: true,
      /* The WASM row executor takes its rows from a contiguous offset and a
       * width, so it reads [S,D] and [S,1,D] exactly as it reads [1,S,D]. A
       * decoder whose optimizer left it sequence-major would otherwise fall
       * back to recomputing its whole prefix each token. */
      sequenceMajorRows: true,
      outputLocation: 'host',
    });
    this.wasmModule = wasmModule;
    this.api = wasmModule.instance.exports;
    this.mem = this.api.memory;
    this.relaxedApi = relaxedSimdModule?.instance?.exports || null;
    this.relaxedSimdEnabled =
      typeof this.relaxedApi?.[WASM_RELAXED_QLINEAR_EXPORT] === 'function';
    this.qbatchMatMulSimdEnabled =
      typeof this.api.qbatch_matmul_i8u8_simd128 === 'function';
    this.pointers = new Map();
    this.f32PackedWeights = new Map();
    this.q8PackedWeights = new Map();
    this.f32PackedNodes = new Map();
    this.tensorCapacities = new Map();
    this.tensorMaximumBytes = new Map();
    this.qconvIm2ColPointer = 0;
    this.qconvIm2ColBytes = 0;
    this._heapHighWaterBytes = this.mem.buffer.byteLength;
    console.log("[VolvoxAI] WASM Engine ready (Tier 2 C kernels).");
  }
  static async init(wasmUrl: string | URL): Promise<WasmEngine | null> {
    try {
      let buffer;
      const browserOnly = typeof __VOLVOXAI_BROWSER_ONLY__ !== 'undefined' &&
        __VOLVOXAI_BROWSER_ONLY__ === true;
      const nodeProcess = (globalThis as typeof globalThis & {
        process?: { versions?: { node?: string } };
      }).process;
      if (!browserOnly && nodeProcess?.versions?.node) {
          const isUrl = typeof URL !== "undefined" && wasmUrl instanceof URL;
          const href = isUrl ? wasmUrl.href : String(wasmUrl);
          if ((isUrl && wasmUrl.protocol !== "file:") || /^(?:https?|data):/i.test(href)) {
            const response = await fetch(wasmUrl);
            if (!response.ok) throw new Error("WASM file not found.");
            buffer = await response.arrayBuffer();
          } else {
            const fsModuleName = 'fs';
            const fs = await import(fsModuleName);
            const file = href.startsWith("file:") && !isUrl ? new URL(href) : wasmUrl;
            buffer = await fs.promises.readFile(file);
          }
      } else {
          const response = await fetch(wasmUrl);
          if (!response.ok) throw new Error("WASM file not found.");
          buffer = await response.arrayBuffer();
      }
      const env = {
        expf: Math.exp,
        logf: Math.log,
        powf: Math.pow,
        sqrtf: Math.sqrt,
        tanhf: Math.tanh,
        sinf: Math.sin,
        cosf: Math.cos,
      };
      const module = await WebAssembly.instantiate(buffer, { env, math: env }) as WasmInstantiation;
      const relaxedSimdModule = await instantiateRelaxedSimdChild(
        module.module, module.instance.exports.memory,
      );
      return new WasmEngine(module, relaxedSimdModule);
    } catch (e) {
      console.warn(`[VolvoxAI] Failed to load WASM from ${wasmUrl}:`, e);
      return null;
    }
  }

  /**
   * Instantiate a fresh linear memory from the already-compiled parent module.
   * Router and family graphs can then remain allocated at the same time without
   * fetching or recompiling the WASM artifact. Scratch-only callers may skip
   * the optional Relaxed-SIMD child when they cannot dispatch its kernels.
   */
  async fork(
    { relaxedSimd = true }: { relaxedSimd?: boolean } = {},
  ) {
    const compiledModule = this.wasmModule?.module;
    if (!(compiledModule instanceof WebAssembly.Module)) {
      throw new Error('WasmEngine.fork requires the compiled parent module.');
    }
    const env = {
      expf: Math.exp,
      logf: Math.log,
      powf: Math.pow,
      sqrtf: Math.sqrt,
      tanhf: Math.tanh,
      sinf: Math.sin,
      cosf: Math.cos,
    };
    const instance = await WebAssembly.instantiate(compiledModule, { env, math: env }) as WasmParentInstance;
    const wasmModule: WasmInstantiation = { module: compiledModule, instance };
    const relaxedSimdModule = relaxedSimd
      ? await instantiateRelaxedSimdChild(compiledModule, instance.exports.memory)
      : null;
    return new WasmEngine(wasmModule, relaxedSimdModule);
  }

  /** Release all host-side graph ownership and reset the instance allocator. */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._setDecodeCacheGenerationListener(null);
    this._incrementalCacheValid = false;
    this.api.reset_heap?.();
    this.pointers.clear();
    this.f32PackedWeights.clear();
    this.q8PackedWeights.clear();
    this.f32PackedNodes.clear();
    this.tensorCapacities.clear();
    this.tensorMaximumBytes.clear();
    this.nodeMetadata?.clear();
    this.graph = null as unknown as WasmGraph;
    this.preparedGraph = undefined as unknown as WasmPreparedGraph;
  }
  /**
   * The single place this backend grows the WASM heap.
   *
   * Growing a non-shared WebAssembly.Memory may copy the whole heap, so
   * extending by just the shortfall makes a compile that allocates n times cost
   * O(n²) in bytes copied — measured at 725 grows and 41% of compile() for the
   * a large bounded encoder+decoder pair.
   *
   * The policy is therefore geometric: take at least the current size, so the
   * heap doubles and the grow count is logarithmic in the final size rather
   * than linear in the allocation count.
   *
   * Doubling is capped, because unbounded doubling trades the time back for
   * memory that WASM never returns to the OS: on this workload it reached
   * 639.6 MB against a 388.2 MB requirement (+251 MB) for no extra speed.
   * Past the cap the heap grows by fixed chunks instead, which keeps the grow
   * count in the same range while bounding the waste to one chunk.
   */
  _growFor(needed: number): void {
    const currentBytes = this.mem.buffer.byteLength;
    if (needed <= currentBytes) {
      this._heapHighWaterBytes = Math.max(this._heapHighWaterBytes, currentBytes);
      return;
    }
    const shortfallPages = Math.ceil((needed - currentBytes) / 65536);
    const currentPages = currentBytes / 65536;
    const stepPages = Math.min(currentPages, WASM_GROW_MAX_STEP_PAGES);
    /* wasm32 addresses 4 GiB, so the heap can never exceed 65536 pages. */
    const headroomPages = Math.max(0, WASM_MAX_MEMORY_PAGES - currentPages);
    const pages = Math.min(
      Math.max(shortfallPages, stepPages), Math.max(shortfallPages, headroomPages),
    );
    const profile = this._compileProfile;
    const started = profile ? performance.now() : 0;
    try {
      this.mem.grow(pages);
    } catch (error) {
      /* The geometric request can be refused where the exact shortfall still
       * fits (an engine-imposed cap, or memory pressure). grow() leaves the
       * memory untouched when it throws, so retrying smaller is safe. */
      if (pages === shortfallPages) throw error;
      this.mem.grow(shortfallPages);
      this._memoryGrowCount++;
      this._heapHighWaterBytes = Math.max(this._heapHighWaterBytes, this.mem.buffer.byteLength);
      if (profile) {
        profile.growMs += performance.now() - started;
        profile.growCount++;
        profile.growPages += shortfallPages;
      }
      return;
    }
    this._memoryGrowCount++;
    this._heapHighWaterBytes = Math.max(this._heapHighWaterBytes, this.mem.buffer.byteLength);
    if (profile) {
      profile.growMs += performance.now() - started;
      profile.growCount++;
      profile.growPages += pages;
    }
  }

  _allocateBytes(bytes: number, label: string): number {
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0x7ffffff0) {
      throw new Error(`WASM ${label} size is not representable by the signed allocator ABI.`);
    }
    const alignedBytes = Math.ceil(bytes / 16) * 16;
    const measurement = this._allocationMeasurement;
    if (measurement) {
      const pointer = measurement.cursor;
      const end = pointer + alignedBytes;
      if (!Number.isSafeInteger(pointer) || pointer < 0 ||
          !Number.isSafeInteger(end) || end > WASM_PORTABLE_MAX_U32) {
        throw new Error(`WASM ${label} allocation exceeds the wasm32 address space.`);
      }
      measurement.cursor = end;
      return pointer;
    }
    const pointer = Number(this.api.alloc_bytes(bytes));
    if (!Number.isSafeInteger(pointer) || pointer < 0) {
      throw new Error(`WASM allocator returned an invalid ${label} pointer.`);
    }
    const needed = pointer + bytes;
    if (!Number.isSafeInteger(needed) || needed > WASM_PORTABLE_MAX_U32) {
      throw new Error(`WASM ${label} allocation exceeds the wasm32 address space.`);
    }
    this._growFor(needed);
    return pointer;
  }

  _alloc(tensor, capacityBytes = tensor.sizeBytes) {
    if (!this.pointers.has(tensor.name)) {
      if (!Number.isSafeInteger(capacityBytes) || capacityBytes < tensor.sizeBytes) {
        throw new Error(`WASM tensor '${tensor.name}' capacity is smaller than its logical bytes.`);
      }
      const ptr = this._allocateBytes(capacityBytes, `tensor '${tensor.name}'`);
      if (this._compileProfile) this._compileProfile.allocBytesCount++;
      this.pointers.set(tensor.name, ptr);
      this.tensorCapacities.set(tensor.name, Object.freeze({
        dtype: tensor.dtype,
        capacityBytes,
        isWeight: tensor.isWeight === true,
        owner: tensor.name,
      }));
      if (this._allocationMeasurement) return ptr;
      // Keep integer token/mask tensors integer-typed while sharing the same
      // WASM heap with JavaScript fallbacks.
      const wasmView = wasmTensorView(tensor, this.mem.buffer, ptr);
      if (tensor.isWeight && tensor.buffer) {
         const src = new Uint8Array(tensor.buffer.buffer, tensor.buffer.byteOffset, tensor.buffer.byteLength);
         const bytesToCopy = Math.min(tensor.buffer.byteLength, tensor.sizeBytes);
         if (this._compileProfile) this._compileProfile.tensorCopyBytes += bytesToCopy;
         new Uint8Array(this.mem.buffer, ptr, bytesToCopy).set(src.subarray(0, bytesToCopy));
      }
      tensor.buffer = wasmView;
    }
    return this.pointers.get(tensor.name)!;
  }
  _allocMetadata(values) {
    const profileStart = this._compileProfile ? performance.now() : 0;
    if (!(values instanceof Uint32Array) || values.length === 0) {
      throw new Error('WASM metadata must be a non-empty Uint32Array.');
    }
    const ptr = this._allocateBytes(values.byteLength, 'metadata');
    if (this._allocationMeasurement) {
      this._variantMetadataStaging?.push(Object.freeze({
        pointer: ptr,
        bytes: new Uint8Array(
          new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        ),
      }));
    } else {
      new Uint32Array(this.mem.buffer, ptr, values.length).set(values);
    }
    if (this._compileProfile) {
      this._compileProfile.metadataAllocMs += performance.now() - profileStart;
      this._compileProfile.allocBytesCount++;
    }
    return ptr;
  }
  _allocTypedMetadata(values) {
    const profileStart = this._compileProfile ? performance.now() : 0;
    if (!ArrayBuffer.isView(values) || values instanceof DataView || values.byteLength === 0) {
      throw new Error('WASM typed metadata must be a non-empty typed array.');
    }
    const ptr = this._allocateBytes(values.byteLength, 'typed metadata');
    if (this._allocationMeasurement) {
      this._variantMetadataStaging?.push(Object.freeze({
        pointer: ptr,
        bytes: new Uint8Array(
          new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
        ),
      }));
    } else {
      new Uint8Array(this.mem.buffer, ptr, values.byteLength).set(
        new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
      );
    }
    if (this._compileProfile) {
      this._compileProfile.metadataAllocMs += performance.now() - profileStart;
      this._compileProfile.allocBytesCount++;
    }
    return ptr;
  }
  _allocScratchBytes(bytes, label) {
    const ptr = this._allocateBytes(bytes, label);
    if (this._compileProfile) this._compileProfile.allocBytesCount++;
    return ptr;
  }
  _allocPackedQ8Weight(
    weight,
    dIn,
    dOut,
    outIn = true,
    mode: WasmQ8PackMode = 'w8a8-wide',
  ) {
    const profileStart = this._compileProfile ? performance.now() : 0;
    const sizeKernel = mode === 'canonical'
      ? this.api.packed_q8_weight_canonical_size
      : this.api.packed_q8_weight_size;
    const packKernel = mode === 'canonical'
      ? this.api.pack_q8_weight_canonical
      : this.api.pack_q8_weight;
    if (!weight || !Number.isInteger(dIn) || dIn <= 0 ||
        !Number.isInteger(dOut) || dOut <= 0 ||
        typeof sizeKernel !== 'function' || typeof packKernel !== 'function') return 0;
    const cacheKey = `${weight.name}:${weight.dtype}:${dIn}:${dOut}:${outIn ? 1 : 0}:${mode}`;
    const cached = this.q8PackedWeights.get(cacheKey);
    if (cached) return cached.pointer;
    if (this._allocationMeasurement) {
      throw new Error(`WASM invariant Q8 weight '${weight.name}' was not packed during initial compilation.`);
    }
    const bytes = Number(sizeKernel(dIn, dOut));
    /* alloc_bytes has a signed-i32 ABI and rounds up by 15 internally. */
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0x7ffffff0) {
      throw new Error(
        `WASM ${mode} packed Q8 weight size overflow for [${dOut},${dIn}].`,
      );
    }
    const ptr = this._allocateBytes(bytes, 'packed Q8 weight');
    const weightPointer = this.pointers.get(weight.name);
    if (weightPointer == null || !Number.isSafeInteger(weightPointer) || weightPointer < 0 ||
        packKernel(ptr, bytes, weightPointer, dIn, dOut,
          wasmDtypeCode(weight.dtype), outIn ? 1 : 0) !== 1) {
      throw new Error(`WASM failed to create ${mode} Q8 pack for weight '${weight.name}'.`);
    }
    this.q8PackedWeights.set(cacheKey, Object.freeze({ pointer: ptr, bytes, cacheKey }));
    this._q8PackCount++;
    if (this._compileProfile) {
      this._compileProfile.packWeightMs += performance.now() - profileStart;
      this._compileProfile.packBytes += bytes;
      this._compileProfile.allocBytesCount++;
    }
    return ptr;
  }
  _compilePackedF32Linear(node) {
    const profileStart = this._compileProfile ? performance.now() : 0;
    if (!['MatMul', 'Linear', 'Gemm'].includes(node.opType) ||
        node.inputs.scale || node.inputs.weight_scale ||
        typeof this.api.gemm_f32_packed_elements !== 'function' ||
        typeof this.api.gemm_f32_pack_b !== 'function' ||
        typeof this.api.gemm_f32_packed !== 'function') return null;
    const input = node.inputs.input || node.inputs.x || node.inputs.a;
    const weight = node.inputs.weight;
    const output = portableOutput(node);
    if (!portableF32Tensor(input) || !portableF32Tensor(weight) ||
        !portableF32Tensor(output) || !weight.isWeight ||
        input.shape.length < 1 || output.shape.length < 1) return null;
    const dIn = input.shape.at(-1);
    const dOut = output.shape.at(-1);
    const rows = input.shape.slice(0, -1)
      .reduce((product, dimension) => product * dimension, 1);
    const doutFirst = node.wLayout ? node.wLayout === 'dout' :
      (weight.shape[0] === dOut && weight.shape[1] === dIn);
    const expectedWeightShape = doutFirst ? [dOut, dIn] : [dIn, dOut];
    if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isInteger(dIn) || dIn <= 0 ||
        !Number.isInteger(dOut) || dOut <= 0 || !sameShape(weight.shape, expectedWeightShape) ||
        portableTensorElements(input) !== rows * dIn ||
        portableTensorElements(output) !== rows * dOut) return null;
    const cacheKey = `${weight.name}:${dIn}:${dOut}:${doutFirst ? 1 : 0}`;
    const cached = this.f32PackedWeights.get(cacheKey);
    if (cached) return cached;
    if (this._allocationMeasurement) {
      throw new Error(`WASM invariant F32 weight '${weight.name}' was not packed during initial compilation.`);
    }
    const packedElements = Number(this.api.gemm_f32_packed_elements(dIn, dOut));
    const packedBytes = packedElements * Float32Array.BYTES_PER_ELEMENT;
    // alloc_bytes takes a signed i32 byte count in the freestanding kernel ABI.
    if (!Number.isSafeInteger(packedElements) || packedElements <= 0 ||
        // alloc_bytes rounds its signed-i32 argument up by 15.
        !Number.isSafeInteger(packedBytes) || packedBytes > 0x7ffffff0) {
      throw new Error(`WASM ${node.opType} node ${node.id} has an unrepresentable packed F32 weight.`);
    }
    const pointer = this._allocateBytes(packedBytes, `packed F32 weight at node ${node.id}`);
    const weightPointer = this.pointers.get(weight.name);
    if (this.api.gemm_f32_pack_b(weightPointer, pointer, dIn, dOut, doutFirst ? 1 : 0) !== 1) {
      throw new Error(`WASM ${node.opType} node ${node.id} could not pack its immutable F32 weight.`);
    }
    const descriptor = Object.freeze({
      pointer, bytes: packedBytes, dIn, dOut, doutFirst, cacheKey,
    });
    this.f32PackedWeights.set(cacheKey, descriptor);
    this._f32PackCount++;
    if (this._compileProfile) {
      this._compileProfile.packWeightMs += performance.now() - profileStart;
      this._compileProfile.packBytes += packedBytes;
      this._compileProfile.allocBytesCount++;
    }
    return descriptor;
  }
  _compilePortableMetadata(node) {
    if (node.opType === 'MoELinear') {
      const descriptor = portableMoELinearDescriptor(node);
      const slots = descriptor.residentSlots;
      if (!slots) return { kind: 'moeLinear', ...descriptor };
      const domain = slots[slots.length - 1] + 1;
      const slotTableBytes = domain * Uint32Array.BYTES_PER_ELEMENT;
      if (!Number.isSafeInteger(domain) || domain <= 0 ||
          !Number.isSafeInteger(slotTableBytes) ||
          slotTableBytes > WASM_MOE_SLOT_TABLE_MAX_BYTES) {
        throw new Error(`WASM MoELinear node ${node.id} has an unrepresentable resident slot domain.`);
      }
      const rows = new Uint32Array(domain).fill(0xffffffff);
      for (let row = 0; row < slots.length; row++) rows[slots[row]] = row;
      return {
        kind: 'moeLinear',
        ...descriptor,
        slotTablePointer: this._allocMetadata(rows),
        slotTableDomain: domain,
      };
    }
    const qlinearDescriptor = portableQLinearDescriptor(node);
    if (qlinearDescriptor) {
      return {
        ...qlinearDescriptor,
        weightScalesPointer: this._allocTypedMetadata(qlinearDescriptor.weightScales),
        weightZeroPointsPointer: this._allocTypedMetadata(qlinearDescriptor.weightZeroPoints),
        packedWeightPointer: qlinearDescriptor.weight.isWeight === true
          ? this._allocPackedQ8Weight(qlinearDescriptor.weight,
            qlinearDescriptor.dIn, qlinearDescriptor.dOut, true,
            wasmW8A8PackMode(qlinearDescriptor.dOut))
          : 0,
      };
    }
    if (['MatMul', 'Linear', 'Gemm'].includes(node.opType)) {
      const input = node.inputs.input || node.inputs.x || node.inputs.a;
      const weight = node.inputs.weight;
      const output = portableOutput(node);
      const scale = node.inputs.scale || node.inputs.weight_scale;
      if (portableF32Tensor(input) && portableF32Tensor(output) && scale &&
          weight?.isWeight === true && ['int8', 'uint8'].includes(weight.dtype) &&
          input.shape.length >= 1 && output.shape.length >= 1) {
        const dIn = input.shape.at(-1);
        const dOut = output.shape.at(-1);
        if (sameShape(weight.shape, [dOut, dIn]) &&
            portableTensorElements(weight) === dIn * dOut) {
          return {
            kind: 'w8a32',
            packedWeightPointer: this._allocPackedQ8Weight(
              weight, dIn, dOut, true, 'canonical',
            ),
          };
        }
      }
    }
    const qembeddingDescriptor = portableQEmbeddingDescriptor(node);
    if (qembeddingDescriptor) {
      return {
        ...qembeddingDescriptor,
        weightScalesPointer: this._allocTypedMetadata(qembeddingDescriptor.weightScales),
        weightZeroPointsPointer: this._allocTypedMetadata(qembeddingDescriptor.weightZeroPoints),
      };
    }
    const qaddDescriptor = portableQAddDescriptor(node);
    if (qaddDescriptor) return qaddDescriptor;
    const qgeluDescriptor = portableQGELUDescriptor(node);
    if (qgeluDescriptor) return qgeluDescriptor;
    const qgroupnormDescriptor = portableQGroupNormDescriptor(node, (tensor) => {
      const pointer = this.pointers.get(tensor?.name);
      if (!tensor || pointer == null || !Number.isSafeInteger(pointer) || pointer < 0) return null;
      try {
        return new Float32Array(this.mem.buffer, pointer, tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT);
      } catch {
        return null;
      }
    });
    if (qgroupnormDescriptor) return qgroupnormDescriptor;
    const qlayernormDescriptor = portableQLayerNormDescriptor(node, (tensor) => {
      const pointer = this.pointers.get(tensor?.name);
      if (!tensor || pointer == null || !Number.isSafeInteger(pointer) || pointer < 0) return null;
      try {
        return new Float32Array(this.mem.buffer, pointer, tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT);
      } catch {
        return null;
      }
    });
    if (qlayernormDescriptor) return qlayernormDescriptor;
    const qsdpaDescriptor = portableQSDPADescriptor(node);
    if (qsdpaDescriptor) return qsdpaDescriptor;
    const qargmaxDescriptor = portableQArgMaxDescriptor(node);
    if (qargmaxDescriptor) return qargmaxDescriptor;
    const qmaskedmeanDescriptor = portableQMaskedMeanDescriptor(node);
    if (qmaskedmeanDescriptor) return qmaskedmeanDescriptor;
    const qsiluDescriptor = portableQSiLUDescriptor(node);
    if (qsiluDescriptor) return qsiluDescriptor;
    const requantizeLinearDescriptor = portableRequantizeLinearDescriptor(node);
    if (requantizeLinearDescriptor) return requantizeLinearDescriptor;
    const qconv2dDescriptor = portableQConv2DDescriptor(node);
    if (qconv2dDescriptor) {
      const packedRows = qconv2dDescriptor.batch * qconv2dDescriptor.outputHeight *
        qconv2dDescriptor.outputWidth;
      const packedDIn = qconv2dDescriptor.kernelHeight *
        qconv2dDescriptor.kernelWidth * qconv2dDescriptor.inputChannels;
      const im2colBytes = packedRows * packedDIn;
      const packedEligible = qconv2dDescriptor.groups === 1 &&
        qconv2dDescriptor.relu === 0 && qconv2dDescriptor.bias != null &&
        qconv2dDescriptor.weight.isWeight === true &&
        qconv2dDescriptor.outputChannels >= 8 &&
        Number.isSafeInteger(packedRows) && packedRows > 0 &&
        Number.isSafeInteger(packedDIn) && packedDIn > 0 &&
        Number.isSafeInteger(im2colBytes) &&
        im2colBytes <= WASM_QCONV_IM2COL_MAX_BYTES &&
        typeof this.api.qconv2d_im2col_i8u8 === 'function' &&
        typeof this.api.qlinear_i8u8_packed === 'function' &&
        typeof this.api.packed_q8_weight_size === 'function' &&
        typeof this.api.pack_q8_weight === 'function';
      return {
        ...qconv2dDescriptor,
        weightScalesPointer: this._allocTypedMetadata(qconv2dDescriptor.weightScales),
        weightZeroPointsPointer: this._allocTypedMetadata(qconv2dDescriptor.weightZeroPoints),
        packedRows,
        packedDIn,
        im2colBytes: packedEligible ? im2colBytes : 0,
        packedWeightPointer: packedEligible
          ? this._allocPackedQ8Weight(qconv2dDescriptor.weight, packedDIn,
            qconv2dDescriptor.outputChannels, true, 'w8a8-wide')
          : 0,
      };
    }
    const quantizedDescriptor = portableQuantizedNodeDescriptor(node);
    if (quantizedDescriptor) return quantizedDescriptor;
    const f32Pool2DDescriptor = portableF32Pool2DDescriptor(node);
    if (f32Pool2DDescriptor) return f32Pool2DDescriptor;
    if (node.opType === 'BatchMatMul') {
      const descriptor = batchMatMulDescriptor(node);
      let incrementalRow: any = null;
      try {
        incrementalRow = batchMatMulDescriptor(quantizedRowNode(
          node, 1, { allowUnprovenInvariantInputs: true },
        ));
      } catch {
        // The ordinary descriptor remains valid. Dependency-aware row
        // preflight reports a precise error only when row mode is requested.
      }
      return { kind: 'batchMatMul', ...descriptor, incrementalRow };
    }
    if (node.opType === 'QBatchMatMul') {
      const descriptor = qBatchMatMulDescriptor(node);
      let incrementalRow: any = null;
      try {
        incrementalRow = qBatchMatMulDescriptor(quantizedRowNode(
          node, 1, { allowUnprovenInvariantInputs: true },
        ));
      } catch {
        // Preserve full-sequence compilation for descriptors outside the
        // strict logical-row layouts.
      }
      return { kind: 'qBatchMatMul', ...descriptor, incrementalRow };
    }
    if (node.opType === 'Equal' || node.opType === 'GreaterOrEqual') {
      const descriptor = comparisonDescriptor(node);
      // The C broadcast ABI represents a scalar as rank-one [1], while the
      // graph retains the ONNX rank-zero shape.
      const aShape = descriptor.a.shape.length ? descriptor.a.shape : [1];
      const bShape = descriptor.b.shape.length ? descriptor.b.shape : [1];
      const outputShape = descriptor.outputShape.length ? descriptor.outputShape : [1];
      const words = new Uint32Array(aShape.length + bShape.length + outputShape.length);
      words.set(aShape, 0);
      words.set(bShape, aShape.length);
      words.set(outputShape, aShape.length + bShape.length);
      const pointer = this._allocMetadata(words);
      return {
        kind: 'comparison', ...descriptor,
        aRank: aShape.length,
        bRank: bShape.length,
        kernelRank: outputShape.length,
        aShapePointer: pointer,
        bShapePointer: pointer + aShape.length * Uint32Array.BYTES_PER_ELEMENT,
        outputShapePointer: pointer +
          (aShape.length + bShape.length) * Uint32Array.BYTES_PER_ELEMENT,
      };
    }
    if (node.opType === 'Not') {
      return { kind: 'not', ...logicalNotDescriptor(node) };
    }
    if (['Add', 'Mul', 'Sub', 'Div'].includes(node.opType)) {
      const compileBinaryDescriptor = (binaryNode) => {
        const descriptor = portableF32ElementwiseDescriptor(binaryNode);
        const words = new Uint32Array(descriptor.aRank + descriptor.bRank + descriptor.outputRank);
        words.set(descriptor.a.shape, 0);
        words.set(descriptor.b.shape, descriptor.aRank);
        words.set(descriptor.output.shape, descriptor.aRank + descriptor.bRank);
        const pointer = this._allocMetadata(words);
        return {
          ...descriptor,
          aShapePointer: pointer,
          bShapePointer: pointer + descriptor.aRank * Uint32Array.BYTES_PER_ELEMENT,
          outputShapePointer: pointer +
            (descriptor.aRank + descriptor.bRank) * Uint32Array.BYTES_PER_ELEMENT,
        };
      };
      const descriptor = compileBinaryDescriptor(node);
      let incrementalRow: any = null;
      if (node.opType === 'Add' || node.opType === 'Mul') {
        try {
          incrementalRow = compileBinaryDescriptor(quantizedRowNode(
            node, 1, { allowUnprovenInvariantInputs: true },
          ));
        } catch {
          // Row execution is opt-in. The full graph remains valid even when an
          // operand cannot be represented by the strict logical-row contract;
          // prepareQuantizedRows will report that exact incompatibility if the
          // caller later requests row mode.
        }
      }
      return { kind: 'binary', ...descriptor, incrementalRow };
    }
    if (node.opType === 'Expand' || node.opType === 'Broadcast') {
      const compileExpandDescriptor = (expandNode) => {
        const descriptor = portableExpandDescriptor(expandNode);
        const words = new Uint32Array(descriptor.inputRank + descriptor.outputRank);
        words.set(descriptor.input.shape, 0);
        words.set(descriptor.output.shape, descriptor.inputRank);
        const pointer = this._allocMetadata(words);
        return {
          ...descriptor,
          inputShapePointer: pointer,
          outputShapePointer: pointer + descriptor.inputRank * Uint32Array.BYTES_PER_ELEMENT,
        };
      };
      const descriptor = compileExpandDescriptor(node);
      let incrementalRow: any = null;
      if (node.opType === 'Expand') {
        try {
          incrementalRow = compileExpandDescriptor(quantizedRowNode(
            node, 1, { allowUnprovenInvariantInputs: true },
          ));
        } catch {
          // Fixed-shape row execution is optional; its preflight reports the
          // exact incompatible descriptor when requested.
        }
      }
      return { kind: 'expand', ...descriptor, incrementalRow };
    }
    if (node.opType === 'Transpose') {
      const descriptor = portableTransposeDescriptor(node);
      const words = new Uint32Array(descriptor.rank * 2);
      words.set(descriptor.input.shape, 0);
      words.set(descriptor.perm, descriptor.rank);
      const pointer = this._allocMetadata(words);
      return {
        kind: 'transpose', ...descriptor,
        shapePointer: pointer,
        permPointer: pointer + descriptor.rank * Uint32Array.BYTES_PER_ELEMENT,
      };
    }
    if (node.opType === 'Where' || node.opType === 'Mask') {
      const descriptor = portableWhereDescriptor(node);
      // The C ABI represents a rank-zero scalar as rank-one [1].
      const conditionShape = descriptor.conditionRank ? descriptor.condition.shape : [1];
      const aShape = descriptor.aRank ? descriptor.a.shape : [1];
      const bShape = descriptor.bRank ? descriptor.b.shape : [1];
      const outputShape = descriptor.outputRank ? descriptor.output.shape : [1];
      const words = new Uint32Array(
        conditionShape.length + aShape.length + bShape.length + outputShape.length,
      );
      words.set(conditionShape, 0);
      words.set(aShape, conditionShape.length);
      words.set(bShape, conditionShape.length + aShape.length);
      words.set(outputShape, conditionShape.length + aShape.length + bShape.length);
      const pointer = this._allocMetadata(words);
      return {
        kind: 'where', ...descriptor,
        conditionRank: conditionShape.length,
        aRank: aShape.length,
        bRank: bShape.length,
        outputRank: outputShape.length,
        conditionShapePointer: pointer,
        aShapePointer: pointer + conditionShape.length * Uint32Array.BYTES_PER_ELEMENT,
        bShapePointer: pointer +
          (conditionShape.length + aShape.length) * Uint32Array.BYTES_PER_ELEMENT,
        outputShapePointer: pointer +
          (conditionShape.length + aShape.length + bShape.length) * Uint32Array.BYTES_PER_ELEMENT,
      };
    }
    if (node.opType === 'Gather') {
      return { kind: 'gather', ...portableGatherDescriptor(node) };
    }
    if (node.opType === 'GatherElements') {
      const descriptor = portableGatherElementsDescriptor(node);
      const words = new Uint32Array(descriptor.rank * 2);
      words.set(descriptor.input.shape, 0);
      words.set(descriptor.indices.shape, descriptor.rank);
      const pointer = this._allocMetadata(words);
      return {
        kind: 'gatherElements', ...descriptor,
        inputShapePointer: pointer,
        indicesShapePointer: pointer + descriptor.rank * Uint32Array.BYTES_PER_ELEMENT,
      };
    }
    if (node.opType === 'Slice') {
      const input = node.inputs.input || node.inputs.x || node.inputs.data;
      const output = node.outputs.out || Object.values(node.outputs || {})[0];
      const descriptor = portablePositiveSliceDescriptor(node, input, output);
      const words = new Uint32Array(descriptor.rank * 4);
      words.set(input.shape, 0);
      words.set(output.shape, descriptor.rank);
      words.set(descriptor.starts, descriptor.rank * 2);
      words.set(descriptor.steps, descriptor.rank * 3);
      const pointer = this._allocMetadata(words);
      const stride = descriptor.rank * Uint32Array.BYTES_PER_ELEMENT;
      return {
        kind: 'slice', input, output, ...descriptor,
        inputShapePointer: pointer,
        outputShapePointer: pointer + stride,
        startsPointer: pointer + stride * 2,
        stepsPointer: pointer + stride * 3,
      };
    }
    if (node.opType === 'Sin' || node.opType === 'Cos') {
      return portableUnaryF32Descriptor(node);
    }
    if (node.opType === 'RoPE') {
      return {
        kind: 'rope',
        ...ropeDescriptor(node, {
          // RuntimeGraph-input values arrive only after specialization. They are
          // validated against caller storage in execute() before upload;
          // reading the reusable arena here would inspect stale capacity bytes.
          validatePositionValues: node.inputs.position_ids?.isWeight === true,
        }),
      };
    }
    if (node.opType === 'SSMScan' || node.opType === 'SelectiveScan') {
      const descriptor = ssmScanDescriptor(node);
      const stateElements = descriptor.batch * descriptor.channels * descriptor.stateWidth;
      return {
        kind: 'ssmScan', ...descriptor,
        stateScratchPointer: this._allocTypedMetadata(new Float32Array(stateElements)),
      };
    }
    return null;
  }
  prepareInvariantSchedule(input: WasmInvariantScheduleInput): WasmInvariantSchedule {
    if (!input || typeof input !== 'object' || !Array.isArray(input.nodes) ||
        !Number.isSafeInteger(input.tensorCount) || input.tensorCount < 0 ||
        !Array.isArray(input.outputNames)) {
      throw new Error('WASM invariant schedule input is invalid.');
    }
    const qGroupNorm: number[] = [];
    const qLayerNorm: number[] = [];
    const qSDPA: number[] = [];
    const qMaskedMean: number[] = [];
    const rope: number[] = [];
    const schedule = input.nodes.map((node, nodeIndex): WasmPreparedStep => {
      const route = kernelRoute('wasm', node.opType);
      if (!route) throw new Error(`WASM operator '${node.opType}' is unsupported.`);
      if (node.opType === 'QGroupNorm') qGroupNorm.push(nodeIndex);
      if (node.opType === 'QLayerNorm') qLayerNorm.push(nodeIndex);
      if (node.opType === 'QSDPA') qSDPA.push(nodeIndex);
      if (node.opType === 'QMaskedMean') qMaskedMean.push(nodeIndex);
      if (node.opType === 'RoPE') rope.push(nodeIndex);
      return Object.freeze({ nodeIndex, nodeId: node.id, opType: node.opType, kernelRoute: route });
    });
    return Object.freeze({
      [WASM_INVARIANT_SCHEDULE]: true as const,
      tensorCount: input.tensorCount,
      outputNames: Object.freeze([...input.outputNames]),
      schedule: Object.freeze(schedule),
      inputPreflights: Object.freeze({
        qGroupNorm: Object.freeze(qGroupNorm),
        qLayerNorm: Object.freeze(qLayerNorm),
        qSDPA: Object.freeze(qSDPA),
        qMaskedMean: Object.freeze(qMaskedMean),
        rope: Object.freeze(rope),
      }),
    });
  }
  _assertInvariantSchedule(graph: WasmGraph, value: Readonly<object>): WasmInvariantSchedule {
    const invariant = value as Partial<WasmInvariantSchedule>;
    const preflights = invariant?.inputPreflights as
      Partial<WasmPreparedInputPreflights> | undefined;
    const invalid = !invariant || invariant[WASM_INVARIANT_SCHEDULE] !== true ||
      !Object.isFrozen(invariant) || !Array.isArray(invariant.schedule) ||
      !Object.isFrozen(invariant.schedule) || !Array.isArray(invariant.outputNames) ||
      !Object.isFrozen(invariant.outputNames) || !preflights || !Object.isFrozen(preflights) ||
      !Array.isArray(preflights.qGroupNorm) || !Object.isFrozen(preflights.qGroupNorm) ||
      !Array.isArray(preflights.qLayerNorm) || !Object.isFrozen(preflights.qLayerNorm) ||
      !Array.isArray(preflights.qSDPA) || !Object.isFrozen(preflights.qSDPA) ||
      !Array.isArray(preflights.qMaskedMean) || !Object.isFrozen(preflights.qMaskedMean) ||
      !Array.isArray(preflights.rope) || !Object.isFrozen(preflights.rope) ||
      invariant.tensorCount !== graph.tensors.size ||
      invariant.schedule.length !== graph.nodes.length ||
      invariant.outputNames.length !== graph.outputNames.length ||
      invariant.outputNames.some((name, index) => name !== graph.outputNames[index]) ||
      invariant.schedule.some((step, index) => {
        const node = graph.nodes[index];
        return !Object.isFrozen(step) || step.nodeIndex !== index || step.nodeId !== node?.id ||
          step.opType !== node?.opType || typeof step.kernelRoute !== 'string' || !step.kernelRoute;
      });
    if (invalid) {
      throw new Error('WASM invariant prepared schedule does not match this graph topology.');
    }
    return invariant as WasmInvariantSchedule;
  }
  prepareGraph(
    graph: WasmGraph,
    invariantSchedule?: Readonly<object>,
  ): WasmPreparedGraph {
    this._assertPortableQuantizedGraph(graph);
    const invariant = invariantSchedule
      ? this._assertInvariantSchedule(graph, invariantSchedule)
      : this.prepareInvariantSchedule({
          nodes: graph.nodes,
          tensorCount: graph.tensors.size,
          outputNames: graph.outputNames,
        });
    return Object.freeze({
      [WASM_PREPARED_GRAPH]: true as const,
      topologyRevision: graph.topologyRevision || 0,
      weightRevision: graph.weightRevision || 0,
      tensorCount: invariant.tensorCount,
      outputNames: invariant.outputNames,
      schedule: invariant.schedule,
      inputPreflights: invariant.inputPreflights,
    });
  }
  _assertPreparedGraph(graph: WasmGraph, value: Readonly<object>): WasmPreparedGraph {
    const prepared = value as Partial<WasmPreparedGraph>;
    const preflights = prepared?.inputPreflights as Partial<WasmPreparedInputPreflights> | undefined;
    const invalid = !prepared || prepared[WASM_PREPARED_GRAPH] !== true ||
      !Object.isFrozen(prepared) || !Array.isArray(prepared.schedule) ||
      !Object.isFrozen(prepared.schedule) || !Array.isArray(prepared.outputNames) ||
      !Object.isFrozen(prepared.outputNames) || !preflights || !Object.isFrozen(preflights) ||
      !Array.isArray(preflights.qGroupNorm) || !Object.isFrozen(preflights.qGroupNorm) ||
      !Array.isArray(preflights.qLayerNorm) || !Object.isFrozen(preflights.qLayerNorm) ||
      !Array.isArray(preflights.qSDPA) || !Object.isFrozen(preflights.qSDPA) ||
      !Array.isArray(preflights.qMaskedMean) || !Object.isFrozen(preflights.qMaskedMean) ||
      !Array.isArray(preflights.rope) || !Object.isFrozen(preflights.rope) ||
      prepared.topologyRevision !== (graph.topologyRevision || 0) ||
      prepared.weightRevision !== (graph.weightRevision || 0) ||
      prepared.tensorCount !== graph.tensors.size || prepared.schedule.length !== graph.nodes.length ||
      prepared.outputNames.length !== graph.outputNames.length ||
      prepared.outputNames.some((name, index) => name !== graph.outputNames[index]) ||
      prepared.schedule.some((step, index) => {
        const node = graph.nodes[index];
        return !Object.isFrozen(step) || step.nodeIndex !== index || step.nodeId !== node?.id ||
          step.opType !== node?.opType || typeof step.kernelRoute !== 'string' || !step.kernelRoute;
      });
    if (invalid) {
      throw new Error('WASM prepared graph does not match this graph revision; recompile the model.');
    }
    return prepared as WasmPreparedGraph;
  }
  _preparedGraphForExecution(): WasmPreparedGraph {
    if (this.preparedGraph) return this.preparedGraph;
    // Direct WasmEngine users historically may install an already-allocated
    // graph and pointer table without calling compile(). Preserve that
    // low-level contract while keeping provider/CompiledModel execution on the
    // eagerly prepared, shared plan path.
    const prepared = this.prepareGraph(this.graph);
    this.preparedGraph = prepared;
    return prepared;
  }
  allocateGraph(graph: WasmGraph, preparedGraph?: Readonly<object>) {
    return preparedGraph ? this.compile(graph, preparedGraph) : this.compile(graph);
  }
  _preflightQGroupNormDynamicAffines(inputs: Record<string, RuntimeTypedArray>) {
    for (const nodeIndex of this.preparedGraph.inputPreflights.qGroupNorm) {
      const node = this.graph.nodes[nodeIndex];
      for (const [label, tensor] of [
        ['weight', node.inputs.weight],
        ['bias', node.inputs.bias],
      ] as Array<[string, Tensor | undefined]>) {
        if (!tensor || tensor.isWeight === true || tensor.isInput !== true) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, tensor.name)
          ? inputs[tensor.name]
          : null;
        if (!supplied) {
          throw new Error(`WASM QGroupNorm node ${node.id} requires graph-input F32 ${label} supplied on every execution.`);
        }
        Tensor.assertCompatibleInput('float32', supplied, tensor.sizeBytes,
          `QGroupNorm node ${node.id} ${label}`);
        if (!supplied.every(Number.isFinite)) {
          throw new Error(`WASM QGroupNorm node ${node.id} ${label} must contain finite F32 values before execution.`);
        }
      }
    }
  }
  _preflightQLayerNormDynamicAffines(inputs: Record<string, RuntimeTypedArray>) {
    for (const nodeIndex of this.preparedGraph.inputPreflights.qLayerNorm) {
      const node = this.graph.nodes[nodeIndex];
      for (const [label, tensor] of [
        ['weight', node.inputs.weight],
        ['bias', node.inputs.bias],
      ] as Array<[string, Tensor | undefined]>) {
        if (!tensor || tensor.isWeight === true || tensor.isInput !== true) continue;
        const supplied = Object.prototype.hasOwnProperty.call(inputs, tensor.name)
          ? inputs[tensor.name]
          : null;
        if (!supplied) {
          throw new Error(`WASM QLayerNorm node ${node.id} requires graph-input F32 ${label} supplied on every execution.`);
        }
        Tensor.assertCompatibleInput('float32', supplied, tensor.sizeBytes,
          `QLayerNorm node ${node.id} ${label}`);
        if (!supplied.every(Number.isFinite)) {
          throw new Error(`WASM QLayerNorm node ${node.id} ${label} must contain finite F32 values before execution.`);
        }
      }
    }
  }
  _preflightQSDPAMasks(inputs: Record<string, RuntimeTypedArray>) {
    for (const nodeIndex of this.preparedGraph.inputPreflights.qSDPA) {
      const node = this.graph.nodes[nodeIndex];
      const mask = node.inputs.mask;
      if (!mask?.isInput) continue;
      const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
        ? inputs[mask.name]
        : null;
      if (!supplied) {
        throw new Error(`WASM QSDPA node ${node.id} requires graph-input I32 mask supplied on every execution.`);
      }
      Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
        `QSDPA node ${node.id} mask`);
    }
  }
  _preflightQMaskedMeanMasks(inputs: Record<string, RuntimeTypedArray>) {
    for (const nodeIndex of this.preparedGraph.inputPreflights.qMaskedMean) {
      const node = this.graph.nodes[nodeIndex];
      const mask = node.inputs.mask;
      if (!mask?.isInput) continue;
      const supplied = Object.prototype.hasOwnProperty.call(inputs, mask.name)
        ? inputs[mask.name]
        : null;
      if (!supplied) {
        throw new Error(`WASM QMaskedMean node ${node.id} requires graph-input I32 mask supplied on every execution.`);
      }
      Tensor.assertCompatibleInput('int32', supplied, mask.sizeBytes,
        `QMaskedMean node ${node.id} mask`);
    }
  }
  _preflightRoPEPositions(inputs: Record<string, RuntimeTypedArray>) {
    for (const nodeIndex of this.preparedGraph.inputPreflights.rope) {
      const node = this.graph.nodes[nodeIndex];
      const positions = node.inputs.position_ids;
      if (!positions?.isInput) continue;
      const supplied = Object.prototype.hasOwnProperty.call(inputs, positions.name)
        ? inputs[positions.name]
        : null;
      if (!supplied) {
        throw new Error(`WASM RoPE node ${node.id} requires graph-input I32 position_ids supplied on every execution.`);
      }
      Tensor.assertCompatibleInput('int32', supplied, positions.sizeBytes,
        `RoPE node ${node.id} position_ids`);
      if (supplied.some((position) => position < 0)) {
        throw new Error(`WASM RoPE node ${node.id} position_ids must be non-negative before execution.`);
      }
    }
  }

  _refreshGraphTensorViews(graph: WasmGraph): void {
    for (const [name, tensor] of graph.tensors.entries()) {
      const pointer = this.pointers.get(name);
      const capacity = this.tensorCapacities.get(name);
      if (pointer == null || !capacity || capacity.dtype !== tensor.dtype ||
          tensor.sizeBytes > capacity.capacityBytes) {
        throw new Error(`WASM tensor '${name}' exceeds or disagrees with its committed capacity.`);
      }
      tensor.buffer = wasmTensorView(tensor, this.mem.buffer, pointer);
    }
  }

  _assertReusableGraph(graph: WasmGraph): void {
    if (!this.graph || graph.tensors.size !== this.graph.tensors.size) {
      throw new Error('WASM shape variant does not match the compiled tensor topology.');
    }
    if ((graph.topologyRevision || 0) !== this.compiledTopologyRevision ||
        (graph.weightRevision || 0) !== this.compiledWeightRevision) {
      throw new Error('WASM shape variant does not match the compiled model revision.');
    }
    for (const [name, tensor] of graph.tensors.entries()) {
      const previous = this.graph.tensors.get(name);
      const capacity = this.tensorCapacities.get(name);
      const maximum = this.tensorMaximumBytes.get(name);
      if (!previous || !capacity || previous.dtype !== tensor.dtype ||
          maximum == null || capacity.dtype !== tensor.dtype || tensor.sizeBytes > maximum ||
          (tensor.isWeight === true) !== capacity.isWeight) {
        throw new Error(`WASM shape variant tensor '${name}' is incompatible with the compiled arena.`);
      }
      if (tensor.isWeight === true &&
          (!sameShape(previous.shape, tensor.shape) || previous.sizeBytes !== tensor.sizeBytes)) {
        throw new Error(`WASM invariant weight '${name}' changed across shape variants.`);
      }
    }
  }

  _preflightVariantGraph(graph: WasmGraph, prepared: WasmPreparedGraph): void {
    for (const step of prepared.schedule) {
      const node = graph.nodes[step.nodeIndex];
      preflightPortableQGroupNormStorage(node);
      preflightPortableQLayerNormStorage(node);
      preflightPortableQSDPAStorage(node);
      preflightPortableQArgMaxStorage(node);
      preflightPortableQMaskedMeanStorage(node);
    }
  }

  _buildVariantResources(graph: WasmGraph, prepared: WasmPreparedGraph) {
    this._refreshGraphTensorViews(graph);
    const nodeMetadata = new Map<WasmNode, WasmNodeMetadata>();
    const f32PackedNodes = new Map<WasmNode, PackedF32WeightDescriptor>();
    for (const step of prepared.schedule) {
      const node = graph.nodes[step.nodeIndex];
      const metadata = this._compilePortableMetadata(node);
      if (metadata) nodeMetadata.set(node, metadata);
      const packedF32 = this._compilePackedF32Linear(node);
      if (packedF32) f32PackedNodes.set(node, packedF32);
    }
    let qconvIm2ColBytes = 0;
    for (const metadata of nodeMetadata.values()) {
      if (metadata?.kind === 'qconv2d' && metadata.packedWeightPointer &&
          metadata.im2colBytes > qconvIm2ColBytes) {
        qconvIm2ColBytes = metadata.im2colBytes;
      }
    }
    const qconvIm2ColPointer = qconvIm2ColBytes > 0
      ? this._allocScratchBytes(qconvIm2ColBytes, 'QConv2D im2col scratch')
      : 0;
    return { nodeMetadata, f32PackedNodes, qconvIm2ColPointer, qconvIm2ColBytes };
  }

  _commitStagedVariantResources(
    writes: readonly WasmStagedMetadataWrite[],
    metadataMark: number,
    measuredEnd: number,
  ): void {
    if (!Number.isSafeInteger(metadataMark) || !Number.isSafeInteger(measuredEnd) ||
        metadataMark < 0 || measuredEnd < metadataMark) {
      throw new Error('WASM staged variant has invalid arena bounds.');
    }
    const bytes = measuredEnd - metadataMark;
    if (bytes > 0) {
      const pointer = this._allocateBytes(bytes, 'shape-variant metadata and scratch');
      if (pointer !== metadataMark) {
        throw new Error('WASM staged variant allocation did not begin at its measured mark.');
      }
    }
    for (const write of writes) {
      const end = write.pointer + write.bytes.byteLength;
      if (!Number.isSafeInteger(write.pointer) || write.pointer < metadataMark ||
          !Number.isSafeInteger(end) || end > measuredEnd) {
        throw new Error('WASM staged metadata write exceeds its measured variant arena.');
      }
      new Uint8Array(this.mem.buffer, write.pointer, write.bytes.byteLength).set(write.bytes);
    }
    const actualEnd = Number(this.api.heap_mark?.());
    if (!Number.isSafeInteger(actualEnd) || actualEnd !== measuredEnd) {
      throw new Error('WASM staged variant allocation disagreed with its preflight measurement.');
    }
  }

  _prepareInvariantWeightPacks(graph: WasmGraph, prepared: WasmPreparedGraph): void {
    for (const step of prepared.schedule) {
      const node = graph.nodes[step.nodeIndex];
      this._compilePackedF32Linear(node);
      const qlinear = portableQLinearDescriptor(node);
      if (qlinear?.weight.isWeight === true) {
        this._allocPackedQ8Weight(
          qlinear.weight,
          qlinear.dIn,
          qlinear.dOut,
          true,
          wasmW8A8PackMode(qlinear.dOut),
        );
      }
      if (['MatMul', 'Linear', 'Gemm'].includes(node.opType)) {
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const weight = node.inputs.weight;
        const output = portableOutput(node);
        const scale = node.inputs.scale || node.inputs.weight_scale;
        if (portableF32Tensor(input) && portableF32Tensor(output) && scale &&
            weight?.isWeight === true && ['int8', 'uint8'].includes(weight.dtype) &&
            input.shape.length >= 1 && output.shape.length >= 1) {
          const dIn = input.shape[input.shape.length - 1];
          const dOut = output.shape[output.shape.length - 1];
          if (sameShape(weight.shape, [dOut, dIn]) &&
              portableTensorElements(weight) === dIn * dOut) {
            this._allocPackedQ8Weight(weight, dIn, dOut, true, 'canonical');
          }
        }
      }
      const descriptor = portableQConv2DDescriptor(node);
      if (!descriptor) continue;
      const packedDIn = descriptor.kernelHeight * descriptor.kernelWidth *
        descriptor.inputChannels;
      const invariantlyEligible = descriptor.groups === 1 && descriptor.relu === 0 &&
        descriptor.bias != null && descriptor.weight.isWeight === true &&
        descriptor.outputChannels >= 8 && Number.isSafeInteger(packedDIn) &&
        packedDIn > 0 && typeof this.api.qconv2d_im2col_i8u8 === 'function' &&
        typeof this.api.qlinear_i8u8_packed === 'function' &&
        typeof this.api.packed_q8_weight_size === 'function' &&
        typeof this.api.pack_q8_weight === 'function';
      if (invariantlyEligible) {
        this._allocPackedQ8Weight(
          descriptor.weight,
          packedDIn,
          descriptor.outputChannels,
          true,
          'w8a8-wide',
        );
      }
    }
  }

  _prepareLivenessCapacityCandidate(
    graph: WasmGraph,
    prepared: WasmPreparedGraph,
    requested: Readonly<Record<string, number>> | undefined = undefined,
  ): WasmCapacityCandidate {
    const maximumLayout = this._maximumLivenessLayout;
    if (maximumLayout === null) {
      throw new Error('WASM liveness arena has no maximum-domain layout.');
    }
    const aliases = wasmActivationAliases(graph, prepared);
    if (!sameWasmAliasMap(aliases, this._activationAliases)) {
      throw new Error('WASM shape variant changes its activation storage-view topology.');
    }
    const sizes = new Map<string, number>();
    for (const [name, tensor] of graph.tensors) {
      if (tensor.isWeight === true) continue;
      const bytes = requested?.[name] ?? tensor.sizeBytes;
      const maximum = this.tensorMaximumBytes.get(name);
      if (!Number.isSafeInteger(bytes) || bytes < tensor.sizeBytes ||
          !Number.isSafeInteger(maximum) || maximum! < bytes) {
        throw new Error(`WASM activation '${name}' exceeds its proved domain maximum.`);
      }
      sizes.set(name, bytes);
    }
    const layout = planWasmActivationWithinMaximum(
      graph, prepared, aliases, sizes, maximumLayout,
    );
    const arenaCapacities = new Map<RuntimeDType, number>();
    let currentBytes = 0;
    let grew = false;
    for (const [arena, maximum] of maximumLayout.capacityByArena) {
      const required = layout.capacityByArena.get(arena) ?? 0;
      let target = this._activationArenaCapacities.get(arena) ?? required;
      if (required > target) {
        while (target < required) {
          const next = Math.ceil(target * this._capacityGrowthFactor);
          if (!Number.isSafeInteger(next) || next <= target) {
            target = required;
            break;
          }
          target = Math.min(maximum, next);
          if (target === maximum && target < required) break;
        }
      }
      if (target < required || target > maximum || target > 0x7ffffff0) {
        throw new Error(`WASM '${arena}' liveness arena cannot grow within its proved maximum.`);
      }
      if (target > 0) arenaCapacities.set(arena, target);
      currentBytes = checkedWasmActivationAdd(currentBytes, target,
        'activation arena capacity');
      if ((this._activationArenaCapacities.get(arena) ?? target) !== target) grew = true;
    }

    const roots = wasmActivationAliasRoots(graph, aliases);
    const capacities = new Map<string, WasmTensorCapacity>();
    for (const [name, current] of this.tensorCapacities) {
      if (current.isWeight) capacities.set(name, current);
    }
    for (const [name, tensor] of graph.tensors) {
      if (tensor.isWeight === true) continue;
      const owner = roots.get(name)!;
      const invariantCapacity = capacities.get(owner);
      if (invariantCapacity?.isWeight === true) {
        if (invariantCapacity.owner !== owner || invariantCapacity.dtype !== tensor.dtype ||
            tensor.sizeBytes > invariantCapacity.capacityBytes) {
          throw new Error(
            `WASM invariant-root alias '${name}' has no compatible weight capacity.`,
          );
        }
        capacities.set(name, Object.freeze({
          dtype: tensor.dtype,
          capacityBytes: invariantCapacity.capacityBytes,
          isWeight: false,
          owner,
        }));
        continue;
      }
      const region = layout.regions.get(owner);
      if (!region || region.dtype !== tensor.dtype || tensor.sizeBytes > region.sizeBytes) {
        throw new Error(`WASM liveness layout omits activation '${name}'.`);
      }
      capacities.set(name, Object.freeze({
        dtype: tensor.dtype,
        capacityBytes: region.sizeBytes,
        isWeight: false,
        owner,
        arena: region.dtype,
        offsetBytes: region.offsetBytes,
      }));
    }
    return {
      capacities,
      currentBytes,
      requiredBytes: layout.capacityBytes,
      grew,
      arenaCapacities,
    };
  }

  _prepareCapacityCandidate(
    graph: WasmGraph,
    prepared: WasmPreparedGraph,
  ): WasmCapacityCandidate {
    if (this._activationStorage === 'liveness') {
      return this._prepareLivenessCapacityCandidate(graph, prepared);
    }
    const requiredByOwner = new Map<string, number>();
    for (const [name, tensor] of graph.tensors.entries()) {
      const current = this.tensorCapacities.get(name);
      if (!current || current.dtype !== tensor.dtype) {
        throw new Error(`WASM tensor '${name}' has no compatible committed capacity.`);
      }
      if (!current.isWeight) {
        requiredByOwner.set(
          current.owner,
          Math.max(requiredByOwner.get(current.owner) ?? 0, tensor.sizeBytes),
        );
      }
    }

    const targetByOwner = new Map<string, number>();
    let currentBytes = 0;
    let grew = false;
    for (const [name, current] of this.tensorCapacities.entries()) {
      if (current.isWeight || current.owner !== name) continue;
      const required = requiredByOwner.get(name) ?? 0;
      const maximum = this.tensorMaximumBytes.get(name);
      if (!Number.isSafeInteger(maximum) || maximum! < required) {
        throw new Error(`WASM tensor capacity '${name}' exceeds its proved domain maximum.`);
      }
      let target = current.capacityBytes;
      if (required > target) {
        while (target < required) {
          const next = Math.ceil(target * this._capacityGrowthFactor);
          if (!Number.isSafeInteger(next) || next <= target) {
            target = required;
            break;
          }
          target = Math.min(maximum!, next);
          if (target === maximum! && target < required) break;
        }
      }
      const dtypeBytes = runtimeDTypeBytes(current.dtype);
      const remainder = target % dtypeBytes;
      if (remainder !== 0) {
        const rounded = target + dtypeBytes - remainder;
        target = rounded <= maximum!
          ? rounded
          : maximum! - (maximum! % dtypeBytes);
      }
      if (target < required || target > maximum! || target > 0x7ffffff0) {
        throw new Error(`WASM tensor capacity '${name}' cannot grow within its proved maximum.`);
      }
      targetByOwner.set(name, target);
      currentBytes += target;
      if (!Number.isSafeInteger(currentBytes) || currentBytes > 0x7ffffff0) {
        throw new Error('WASM activation arena exceeds the signed allocator address range.');
      }
      if (target !== current.capacityBytes) grew = true;
    }

    const capacities = new Map<string, WasmTensorCapacity>();
    for (const [name, current] of this.tensorCapacities.entries()) {
      if (current.isWeight) {
        capacities.set(name, current);
        continue;
      }
      const target = targetByOwner.get(current.owner);
      if (target == null) throw new Error(`WASM capacity owner '${current.owner}' is missing.`);
      capacities.set(name, Object.freeze({
        dtype: current.dtype,
        capacityBytes: target,
        isWeight: false,
        owner: current.owner,
      }));
    }
    const requiredBytes = [...requiredByOwner.values()].reduce(
      (total, bytes) => checkedWasmActivationAdd(total, bytes,
        'required persistent activation bytes'),
      0,
    );
    return {
      capacities,
      currentBytes,
      requiredBytes,
      grew,
      arenaCapacities: null,
    };
  }

  _installActivationLayout(
    graph: WasmGraph,
    capacities: ReadonlyMap<string, WasmTensorCapacity>,
    arenaCapacities: ReadonlyMap<RuntimeDType, number> | null = null,
  ): void {
    const persistentPointers = new Map<string, number>();
    const persistentCapacities = new Map<string, WasmTensorCapacity>();
    for (const [name, capacity] of this.tensorCapacities.entries()) {
      if (!capacity.isWeight) continue;
      const pointer = this.pointers.get(name);
      if (pointer == null) throw new Error(`WASM invariant weight '${name}' lost its pointer.`);
      persistentPointers.set(name, pointer);
      persistentCapacities.set(name, capacity);
    }
    this.pointers.clear();
    this.tensorCapacities.clear();
    for (const [name, pointer] of persistentPointers) this.pointers.set(name, pointer);
    for (const [name, capacity] of persistentCapacities) {
      this.tensorCapacities.set(name, capacity);
    }

    if (this._activationStorage === 'liveness') {
      if (arenaCapacities === null) {
        throw new Error('WASM liveness activation layout omits its dtype arenas.');
      }
      const arenaPointers = new Map<RuntimeDType, number>();
      for (const [arena, bytes] of [...arenaCapacities.entries()]
        .sort(([left], [right]) => compareWasmNames(left, right))) {
        if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0x7ffffff0) {
          throw new Error(`WASM '${arena}' liveness arena has invalid capacity.`);
        }
        arenaPointers.set(arena,
          this._allocateBytes(bytes, `${arena} activation liveness arena`));
      }
      for (const [name, tensor] of graph.tensors) {
        const capacity = capacities.get(name);
        if (!capacity) throw new Error(`WASM activation layout omits tensor '${name}'.`);
        if (capacity.isWeight) continue;
        const invariantCapacity = capacities.get(capacity.owner);
        if (invariantCapacity?.isWeight === true) {
          const invariantPointer = this.pointers.get(capacity.owner);
          if (invariantPointer == null || invariantCapacity.owner !== capacity.owner ||
              invariantCapacity.dtype !== tensor.dtype ||
              tensor.sizeBytes > invariantCapacity.capacityBytes ||
              capacity.capacityBytes !== invariantCapacity.capacityBytes) {
            throw new Error(
              `WASM invariant-root alias '${name}' has no compatible weight storage.`,
            );
          }
          this.pointers.set(name, invariantPointer);
          this.tensorCapacities.set(name, capacity);
          continue;
        }
        const arena = capacity.arena;
        const offset = capacity.offsetBytes;
        const arenaPointer = arena === undefined ? undefined : arenaPointers.get(arena);
        const arenaBytes = arena === undefined ? undefined : arenaCapacities.get(arena);
        if (arenaPointer === undefined || arenaBytes === undefined ||
            !Number.isSafeInteger(offset) || offset! < 0 ||
            tensor.sizeBytes > capacity.capacityBytes ||
            offset! > arenaBytes - capacity.capacityBytes) {
          throw new Error(`WASM liveness activation '${name}' exceeds its dtype arena.`);
        }
        this.pointers.set(name, arenaPointer + offset!);
        this.tensorCapacities.set(name, capacity);
      }
      return;
    }

    for (const [name, tensor] of graph.tensors.entries()) {
      const capacity = capacities.get(name);
      if (!capacity) throw new Error(`WASM activation layout omits tensor '${name}'.`);
      if (!capacity.isWeight && capacity.owner === name) {
        this._alloc(tensor, capacity.capacityBytes);
      }
    }
    for (const [name, tensor] of graph.tensors.entries()) {
      const capacity = capacities.get(name)!;
      if (capacity.isWeight || capacity.owner === name) continue;
      const ownerPointer = this.pointers.get(capacity.owner);
      const ownerCapacity = this.tensorCapacities.get(capacity.owner);
      if (ownerPointer == null || !ownerCapacity || ownerCapacity.dtype !== tensor.dtype ||
          tensor.sizeBytes > ownerCapacity.capacityBytes) {
        throw new Error(`WASM alias '${name}' has no compatible activation owner.`);
      }
      this.pointers.set(name, ownerPointer);
      this.tensorCapacities.set(name, Object.freeze({
        dtype: tensor.dtype,
        capacityBytes: ownerCapacity.capacityBytes,
        isWeight: false,
        owner: capacity.owner,
      }));
    }
  }

  _restoreVariantResources(
    graph: WasmGraph,
    prepared: WasmPreparedGraph,
    capacities: ReadonlyMap<string, WasmTensorCapacity>,
    arenaCapacities: ReadonlyMap<RuntimeDType, number> | null,
  ): void {
    const mark = this._activationArenaMark;
    if (mark == null || typeof this.api.heap_rewind !== 'function' ||
        this.api.heap_rewind(mark) !== 1) {
      throw new Error('WASM reusable arena could not rewind while restoring its previous variant.');
    }
    this._installActivationLayout(graph, capacities, arenaCapacities);
    const metadataMark = Number(this.api.heap_mark?.());
    if (!Number.isSafeInteger(metadataMark) || metadataMark < mark) {
      throw new Error('WASM reusable arena returned an invalid restored metadata mark.');
    }
    this._metadataArenaMark = metadataMark;
    const restored = this._buildVariantResources(graph, prepared);
    this.nodeMetadata = restored.nodeMetadata;
    this.f32PackedNodes.clear();
    for (const [node, descriptor] of restored.f32PackedNodes) {
      this.f32PackedNodes.set(node, descriptor);
    }
    this.qconvIm2ColPointer = restored.qconvIm2ColPointer;
    this.qconvIm2ColBytes = restored.qconvIm2ColBytes;
    this._refreshGraphTensorViews(graph);
  }

  rebindGraph(
    graph: WasmGraph,
    preparedGraph: Readonly<object> = this.prepareGraph(graph),
    options: Pick<WasmGraphAllocationOptions, 'shapeSignature'> = {},
  ) {
    const mark = this._activationArenaMark;
    if (mark == null || typeof this.api.heap_mark !== 'function' ||
        typeof this.api.heap_rewind !== 'function') {
      throw new Error('WASM module does not expose the reusable dynamic-shape arena ABI.');
    }
    if (!options || typeof options !== 'object' || Array.isArray(options) ||
        (options.shapeSignature !== undefined &&
          (typeof options.shapeSignature !== 'string' || options.shapeSignature.length === 0))) {
      throw new Error('WASM shape-variant options are invalid.');
    }
    const prepared = this._assertPreparedGraph(graph, preparedGraph);
    this._assertReusableGraph(graph);
    this._preflightVariantGraph(graph, prepared);
    const capacityCandidate = this._prepareCapacityCandidate(graph, prepared);
    const previousGraph = this.graph;
    const previousPrepared = this.preparedGraph;
    const previousSignature = this._currentShapeSignature;
    const previousCapacities = new Map(this.tensorCapacities);
    const previousPointers = new Map(this.pointers);
    const previousCapacityBytes = this._activationCapacityBytes;
    const previousRequiredBytes = this._activationRequiredBytes;
    const previousArenaCapacities = this._activationArenaCapacities;
    const previousMetadataMark = this._metadataArenaMark;
    const previousVariantBytes = this._variantBytes;

    /* Dry-run the complete activation layout, descriptors, and scratch against
     * the immutable pack cache. The first grow covers candidate tensor views;
     * the second covers metadata/scratch. Neither changes the allocator mark
     * or committed graph, so an allocation failure leaves the old variant. */
    this._allocationMeasurement = { cursor: mark };
    const stagedMetadataWrites: WasmStagedMetadataWrite[] = [];
    this._variantMetadataStaging = stagedMetadataWrites;
    let measuredEnd: number;
    let measuredMetadataMark: number;
    let candidate: ReturnType<WasmEngine['_buildVariantResources']>;
    try {
      this._installActivationLayout(
        graph,
        capacityCandidate.capacities,
        capacityCandidate.arenaCapacities,
      );
      measuredMetadataMark = this._allocationMeasurement.cursor;
      this._growFor(measuredMetadataMark);
      candidate = this._buildVariantResources(graph, prepared);
      measuredEnd = this._allocationMeasurement.cursor;
      this._growFor(measuredEnd);
    } finally {
      this._variantMetadataStaging = null;
      this._allocationMeasurement = null;
      this.pointers.clear();
      this.tensorCapacities.clear();
      for (const [name, pointer] of previousPointers) this.pointers.set(name, pointer);
      for (const [name, capacity] of previousCapacities) {
        this.tensorCapacities.set(name, capacity);
      }
      this._refreshGraphTensorViews(previousGraph);
    }
    try {
      if (this.api.heap_rewind(mark) !== 1) {
        throw new Error('WASM reusable arena rejected its committed mark.');
      }
      this._installActivationLayout(
        graph,
        capacityCandidate.capacities,
        capacityCandidate.arenaCapacities,
      );
      const actualMetadataMark = Number(this.api.heap_mark());
      if (!Number.isSafeInteger(actualMetadataMark) ||
          actualMetadataMark !== measuredMetadataMark) {
        throw new Error('WASM activation layout disagreed with its preflight measurement.');
      }
      this._commitStagedVariantResources(
        stagedMetadataWrites,
        actualMetadataMark,
        measuredEnd,
      );
      this._refreshGraphTensorViews(graph);
      this.graph = graph;
      this.preparedGraph = prepared;
      this.nodeMetadata = candidate.nodeMetadata;
      this.f32PackedNodes.clear();
      for (const [node, descriptor] of candidate.f32PackedNodes) {
        this.f32PackedNodes.set(node, descriptor);
      }
      this.qconvIm2ColPointer = candidate.qconvIm2ColPointer;
      this.qconvIm2ColBytes = candidate.qconvIm2ColBytes;
      this._currentShapeSignature = options.shapeSignature ?? null;
      this._metadataArenaMark = actualMetadataMark;
      this._activationCapacityBytes = capacityCandidate.currentBytes;
      this._activationRequiredBytes = capacityCandidate.requiredBytes;
      this._activationArenaCapacities = capacityCandidate.arenaCapacities ?? new Map();
      this._activationCapacityHighWaterBytes = Math.max(
        this._activationCapacityHighWaterBytes,
        capacityCandidate.currentBytes,
      );
      if (capacityCandidate.grew) this._activationGrowCount++;
      this._variantBytes = measuredEnd - actualMetadataMark;
      this._variantHighWaterBytes = Math.max(this._variantHighWaterBytes, this._variantBytes);
      this._variantRebindCount++;
      this.resetDecodeCache();
      return this;
    } catch (error) {
      try {
        this._restoreVariantResources(
          previousGraph,
          previousPrepared,
          previousCapacities,
          this._activationStorage === 'liveness' ? previousArenaCapacities : null,
        );
        this.graph = previousGraph;
        this.preparedGraph = previousPrepared;
        this._currentShapeSignature = previousSignature;
        this._activationCapacityBytes = previousCapacityBytes;
        this._activationRequiredBytes = previousRequiredBytes;
        this._activationArenaCapacities = previousArenaCapacities;
        this._metadataArenaMark = previousMetadataMark;
        this._variantBytes = previousVariantBytes;
      } catch (restoreError) {
        throw new AggregateError(
          [error, restoreError],
          'WASM shape specialization failed and its previous variant could not be restored.',
        );
      }
      throw error;
    }
  }

  compile(
    graph: WasmGraph,
    preparedGraph: Readonly<object> = this.prepareGraph(graph),
    allocationOptions: WasmGraphAllocationOptions = {},
  ) {
    /* Opt-in phase breakdown. Reading the global once here is the only cost
     * when it is unset; every site below then tests one null field. */
    const profile = (globalThis as any).__VOLVOX_WASM_COMPILE_PROFILE
      ? WasmEngine._newCompileProfile() : null;
    this._compileProfile = profile;
    const compileStart = profile ? performance.now() : 0;
    const prepared = this._assertPreparedGraph(graph, preparedGraph);
    if (!allocationOptions || typeof allocationOptions !== 'object' ||
        Array.isArray(allocationOptions)) {
      throw new Error('WASM graph allocation options must be an object.');
    }
    const requestedCapacities = allocationOptions.tensorCapacityBytes;
    const maximumCapacities = allocationOptions.tensorMaximumBytes;
    if (requestedCapacities !== undefined &&
        (!requestedCapacities || typeof requestedCapacities !== 'object' ||
          Array.isArray(requestedCapacities))) {
      throw new Error('WASM tensor capacities must be a name-to-byte-count record.');
    }
    if (maximumCapacities !== undefined &&
        (!maximumCapacities || typeof maximumCapacities !== 'object' ||
          Array.isArray(maximumCapacities))) {
      throw new Error('WASM tensor maximums must be a name-to-byte-count record.');
    }
    const capacityGrowthFactor = allocationOptions.capacityGrowthFactor ?? 2;
    if (typeof capacityGrowthFactor !== 'number' || !Number.isFinite(capacityGrowthFactor) ||
        capacityGrowthFactor <= 1) {
      throw new Error('WASM capacity growth factor must be finite and greater than one.');
    }
    const activationStorage = allocationOptions.activationStorage ?? 'persistent';
    if (activationStorage !== 'persistent' && activationStorage !== 'liveness') {
      throw new Error("WASM activation storage must be 'persistent' or 'liveness'.");
    }
    if (allocationOptions.shapeSignature !== undefined &&
        (typeof allocationOptions.shapeSignature !== 'string' ||
          allocationOptions.shapeSignature.length === 0)) {
      throw new Error('WASM shape signature must be a non-empty string.');
    }
    for (const name of new Set([
      ...Object.keys(requestedCapacities ?? {}),
      ...Object.keys(maximumCapacities ?? {}),
    ])) {
      if (!graph.tensors.has(name)) {
        throw new Error(`WASM tensor capacity references unknown tensor '${name}'.`);
      }
      for (const [label, bytes] of [
        ['capacity', requestedCapacities?.[name]],
        ['maximum', maximumCapacities?.[name]],
      ] as const) {
        if (bytes !== undefined &&
            (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > 0x7ffffff0)) {
          throw new Error(`WASM tensor '${name}' ${label} is not representable.`);
        }
      }
    }
    this.resetDecodeCache();
    console.log("[VolvoxAI WASM] Allocating graph tensors on WASM heap...");
    if (this.api.reset_heap) this.api.reset_heap();
    this.pointers.clear();
    this.tensorCapacities.clear();
    this.tensorMaximumBytes.clear();
    this.nodeMetadata = new Map();
    this.f32PackedWeights.clear();
    this.q8PackedWeights.clear();
    this.f32PackedNodes.clear();
    this.qconvIm2ColPointer = 0;
    this.qconvIm2ColBytes = 0;
    this._variantArenaMark = null;
    this._activationArenaMark = null;
    this._metadataArenaMark = null;
    this._allocationMeasurement = null;
    this._variantMetadataStaging = null;
    this._currentShapeSignature = null;
    this._persistentWeightBytes = 0;
    this._activationStorage = activationStorage;
    this._activationAliases = new Map();
    this._maximumLivenessLayout = null;
    this._activationArenaCapacities = new Map();
    this._activationCapacityBytes = 0;
    this._activationCapacityHighWaterBytes = 0;
    this._activationRequiredBytes = 0;
    this._activationGrowCount = 0;
    this._capacityGrowthFactor = capacityGrowthFactor;
    this._variantBytes = 0;
    this._variantHighWaterBytes = 0;
    this._variantRebindCount = 0;
    this._f32PackCount = 0;
    this._q8PackCount = 0;
    // WASM is an inference backend. Eliminate standalone Dropout before heap
    // allocation so it consumes neither a separate activation nor a runtime
    // call. Chained Dropout aliases are resolved after ordinary tensors.
    const dropoutAliases = wasmActivationAliases(graph, prepared);
    this._activationAliases = dropoutAliases;
    const preflightStart = profile ? performance.now() : 0;
    for (const step of prepared.schedule) {
      const node = graph.nodes[step.nodeIndex];
      preflightPortableQGroupNormStorage(node);
      preflightPortableQLayerNormStorage(node);
      preflightPortableQSDPAStorage(node);
      preflightPortableQArgMaxStorage(node);
      preflightPortableQMaskedMeanStorage(node);
    }
    if (profile) profile.preflightMs = performance.now() - preflightStart;
    const tensorAllocStart = profile ? performance.now() : 0;
    const aliasRoots = new Map<string, string>();
    const resolveAliasRootName = (name: string, visiting = new Set<string>()): string => {
      const cached = aliasRoots.get(name);
      if (cached) return cached;
      const sourceName = dropoutAliases.get(name);
      if (!sourceName) {
        aliasRoots.set(name, name);
        return name;
      }
      if (visiting.has(name)) throw new Error(`WASM storage alias cycle contains tensor '${name}'.`);
      visiting.add(name);
      const root = resolveAliasRootName(sourceName, visiting);
      visiting.delete(name);
      aliasRoots.set(name, root);
      return root;
    };
    const capacityByRoot = new Map<string, number>();
    const maximumByRoot = new Map<string, number>();
    for (const [name, tensor] of graph.tensors.entries()) {
      const requested = requestedCapacities?.[name] ?? tensor.sizeBytes;
      const maximum = maximumCapacities?.[name] ?? requested;
      if (!Number.isSafeInteger(requested) || requested < tensor.sizeBytes ||
          requested > 0x7ffffff0) {
        throw new Error(`WASM tensor '${name}' capacity must cover its logical bytes within the signed allocator ABI.`);
      }
      if (!Number.isSafeInteger(maximum) || maximum < requested ||
          maximum > 0x7ffffff0) {
        throw new Error(`WASM tensor '${name}' maximum must cover its initial capacity.`);
      }
      if (tensor.isWeight === true &&
          (requested !== tensor.sizeBytes || maximum !== tensor.sizeBytes)) {
        throw new Error(`WASM invariant weight '${name}' must use its exact logical byte size.`);
      }
      const root = resolveAliasRootName(name);
      capacityByRoot.set(root, Math.max(capacityByRoot.get(root) ?? 0, requested));
      maximumByRoot.set(root, Math.max(maximumByRoot.get(root) ?? 0, maximum));
      this.tensorMaximumBytes.set(name, maximum);
    }
    for (const name of graph.tensors.keys()) {
      const root = resolveAliasRootName(name);
      this.tensorMaximumBytes.set(name, maximumByRoot.get(root)!);
    }
    /* Invariant storage is a true prefix. Packed panels are prepared before
     * any activation pointer so the complete activation/metadata/scratch
     * suffix can later be rewound and geometrically replanned without copying
     * or repacking a weight. */
    for (const [name, tensor] of graph.tensors.entries()) {
      if (tensor.isWeight === true && !dropoutAliases.has(name)) {
        this._alloc(tensor, capacityByRoot.get(name));
      }
    }
    this._prepareInvariantWeightPacks(graph, prepared);
    if (typeof this.api.heap_mark === 'function') {
      const activationMark = Number(this.api.heap_mark());
      if (!Number.isSafeInteger(activationMark) || activationMark < 0 ||
          activationMark > WASM_PORTABLE_MAX_U32 || (activationMark & 15) !== 0) {
        throw new Error('WASM allocator returned an invalid activation-arena mark.');
      }
      this._activationArenaMark = activationMark;
    }
    if (activationStorage === 'liveness') {
      const maximumSizes = new Map<string, number>();
      for (const [name, tensor] of graph.tensors) {
        if (tensor.isWeight !== true) maximumSizes.set(name, this.tensorMaximumBytes.get(name)!);
      }
      this._maximumLivenessLayout = planWasmActivationLiveness(
        graph, prepared, dropoutAliases, maximumSizes,
      );
      const initial = this._prepareLivenessCapacityCandidate(
        graph, prepared, requestedCapacities,
      );
      this._installActivationLayout(graph, initial.capacities, initial.arenaCapacities);
      this._activationArenaCapacities = initial.arenaCapacities ?? new Map();
      this._activationCapacityBytes = initial.currentBytes;
      this._activationRequiredBytes = initial.requiredBytes;
      this._activationCapacityHighWaterBytes = initial.currentBytes;
    } else {
      for (const [name, tensor] of graph.tensors.entries()) {
        if (tensor.isWeight !== true && !dropoutAliases.has(name)) {
          this._alloc(tensor, capacityByRoot.get(name));
        }
      }
      const resolveDropoutAlias = (name: string, visiting = new Set<string>()): number => {
        if (this.pointers.has(name)) return this.pointers.get(name)!;
        if (visiting.has(name)) throw new Error(`Dropout alias cycle contains tensor '${name}'.`);
        const sourceName = dropoutAliases.get(name);
        if (!sourceName) throw new Error(`Dropout alias '${name}' has no allocated source.`);
        visiting.add(name);
        const pointer = resolveDropoutAlias(sourceName, visiting);
        visiting.delete(name);
        this.pointers.set(name, pointer);
        return pointer;
      };
      for (const name of dropoutAliases.keys()) {
        const pointer = resolveDropoutAlias(name);
        const tensor = graph.tensors.get(name)!;
        const root = resolveAliasRootName(name);
        const rootCapacity = this.tensorCapacities.get(root);
        if (!rootCapacity || this.pointers.get(root) !== pointer ||
            rootCapacity.dtype !== tensor.dtype) {
          throw new Error(`WASM storage alias '${name}' has no compatible capacity owner.`);
        }
        this.tensorCapacities.set(name, Object.freeze({
          dtype: tensor.dtype,
          capacityBytes: rootCapacity.capacityBytes,
          isWeight: tensor.isWeight === true,
          owner: root,
        }));
      }
    }
    if (typeof this.api.heap_mark === 'function') {
      const metadataMark = Number(this.api.heap_mark());
      if (!Number.isSafeInteger(metadataMark) || metadataMark < 0 ||
          metadataMark > WASM_PORTABLE_MAX_U32 || (metadataMark & 15) !== 0) {
        throw new Error('WASM allocator returned an invalid metadata-arena mark.');
      }
      this._metadataArenaMark = metadataMark;
    }
    if (profile) profile.tensorAllocMs = performance.now() - tensorAllocStart;
    // Metadata validation for canonical byte operators deliberately inspects
    // the physical typed views to reject malformed/overlapping storage.  A
    // WASM memory grow detaches every earlier view; metadata allocations for
    // one node can therefore make a later valid QMaskedMean/QArgMax/QSDPA
    // descriptor look malformed.  Refresh lazily whenever the backing memory
    // changes, including between metadata-owning nodes.
    let viewedMemory: ArrayBuffer | null = null;
    const refreshTensorViews = () => {
      if (viewedMemory === this.mem.buffer) return;
      const started = profile ? performance.now() : 0;
      for (const [name, tensor] of graph.tensors.entries()) {
        const ptr = this.pointers.get(name);
        tensor.buffer = wasmTensorView(tensor, this.mem.buffer, ptr);
      }
      viewedMemory = this.mem.buffer;
      if (profile) {
        profile.viewRefreshMs += performance.now() - started;
        profile.viewRefreshCount++;
      }
    };
    const descriptorStart = profile ? performance.now() : 0;
    const nestedBefore = profile ? profile.viewRefreshMs + profile.packWeightMs +
      profile.metadataAllocMs : 0;
    for (const step of prepared.schedule) {
      const node = graph.nodes[step.nodeIndex];
      refreshTensorViews();
      const metadata = this._compilePortableMetadata(node);
      if (metadata) {
        this.nodeMetadata.set(node, metadata);
        if (profile) profile.descriptorCount++;
      }
      const packedF32 = this._compilePackedF32Linear(node);
      if (packedF32) this.f32PackedNodes.set(node, packedF32);
    }
    if (profile) {
      /* Descriptor construction is what the loop spends outside the phases it
       * nests (view refresh, weight packing, metadata allocation). */
      const nested = profile.viewRefreshMs + profile.packWeightMs +
        profile.metadataAllocMs - nestedBefore;
      profile.descriptorMs = performance.now() - descriptorStart - nested;
    }
    for (const metadata of this.nodeMetadata.values()) {
      if (metadata?.kind === 'qconv2d' && metadata.packedWeightPointer &&
          metadata.im2colBytes > this.qconvIm2ColBytes) {
        this.qconvIm2ColBytes = metadata.im2colBytes;
      }
    }
    if (this.qconvIm2ColBytes > 0) {
      this.qconvIm2ColPointer = this._allocScratchBytes(
        this.qconvIm2ColBytes, 'QConv2D im2col scratch',
      );
    }
    // The final node may have allocated metadata and grown memory after its
    // descriptor was built, so publish one final coherent set of views.
    refreshTensorViews();
    this._persistentWeightBytes = [...this.tensorCapacities.entries()]
      .filter(([name, capacity]) => capacity.isWeight && capacity.owner === name)
      .reduce((total, [, capacity]) => total + capacity.capacityBytes, 0);
    if (activationStorage === 'persistent') {
      this._activationCapacityBytes = [...this.tensorCapacities.entries()]
        .filter(([name, capacity]) => !capacity.isWeight && capacity.owner === name)
        .reduce((total, [, capacity]) => total + capacity.capacityBytes, 0);
      const requiredByOwner = new Map<string, number>();
      for (const [name, tensor] of graph.tensors) {
        const capacity = this.tensorCapacities.get(name);
        if (tensor.isWeight !== true && capacity) {
          requiredByOwner.set(capacity.owner,
            Math.max(requiredByOwner.get(capacity.owner) ?? 0, tensor.sizeBytes));
        }
      }
      this._activationRequiredBytes = [...requiredByOwner.values()].reduce(
        (total, bytes) => checkedWasmActivationAdd(total, bytes,
          'required persistent activation bytes'),
        0,
      );
      this._activationCapacityHighWaterBytes = this._activationCapacityBytes;
    }
    if (typeof this.api.heap_mark === 'function' &&
        typeof this.api.heap_rewind === 'function') {
      const end = Number(this.api.heap_mark());
      if (!Number.isSafeInteger(end) || end < 0 || end > WASM_PORTABLE_MAX_U32 ||
          (end & 15) !== 0 || this._activationArenaMark == null ||
          this._metadataArenaMark == null || this._metadataArenaMark < this._activationArenaMark ||
          end < this._metadataArenaMark) {
        throw new Error('WASM allocator returned an invalid reusable-arena mark.');
      }
      this._variantArenaMark = this._activationArenaMark;
      this._variantBytes = end - this._metadataArenaMark;
      this._variantHighWaterBytes = this._variantBytes;
    }
    this._currentShapeSignature = allocationOptions.shapeSignature ?? null;
    this._heapHighWaterBytes = Math.max(this._heapHighWaterBytes, this.mem.buffer.byteLength);
    this.graph = graph;
    this.preparedGraph = prepared;
    this.compiledWeightRevision = graph.weightRevision || 0;
    this.compiledTopologyRevision = graph.topologyRevision || 0;
    if (profile) {
      profile.totalMs = performance.now() - compileStart;
      profile.finalHeapBytes = this.mem.buffer.byteLength;
      profile.unattributedMs = profile.totalMs -
        (profile.preflightMs + profile.tensorAllocMs + profile.descriptorMs +
         profile.packWeightMs + profile.viewRefreshMs + profile.metadataAllocMs);
      /* A session compiles several graphs (encoder, decoder, ...). Append so a
       * later compile cannot erase an earlier one's breakdown. */
      const sink = (globalThis as any);
      sink.__VOLVOX_WASM_COMPILE_PROFILE_RESULT = profile;
      (sink.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS ||= []).push(profile);
      this._compileProfile = null;
    }
    return this;
  }

  inspectArena(): Readonly<WasmArenaInspection> {
    const logicalActivationBytes = this.graph
      ? [...this.graph.tensors.values()]
        .filter((tensor) => tensor.isWeight !== true)
        .reduce((total, tensor) => total + tensor.sizeBytes, 0)
      : 0;
    const packedWeightBytes = [...this.f32PackedWeights.values()]
      .reduce((total, descriptor) => total + descriptor.bytes, 0) +
      [...this.q8PackedWeights.values()]
        .reduce((total, descriptor) => total + descriptor.bytes, 0);
    return Object.freeze({
      reusable: this._variantArenaMark !== null,
      currentSignature: this._currentShapeSignature,
      activationStorage: this._activationStorage,
      persistentWeightBytes: this._persistentWeightBytes,
      packedWeightBytes,
      logicalActivationBytes,
      activationRequiredBytes: this._activationRequiredBytes,
      activationReuseBytes: Math.max(0, logicalActivationBytes - this._activationRequiredBytes),
      activationArenaCount: this._activationStorage === 'liveness'
        ? this._activationArenaCapacities.size
        : [...this.tensorCapacities.entries()]
          .filter(([name, capacity]) => !capacity.isWeight && capacity.owner === name)
          .length,
      activationCapacityBytes: this._activationCapacityBytes,
      activationCapacityHighWaterBytes: this._activationCapacityHighWaterBytes,
      activationGrowCount: this._activationGrowCount,
      variantBytes: this._variantBytes,
      variantHighWaterBytes: this._variantHighWaterBytes,
      heapBytes: this.mem.buffer.byteLength,
      heapHighWaterBytes: this._heapHighWaterBytes,
      memoryGrowCount: this._memoryGrowCount,
      variantRebindCount: this._variantRebindCount,
      f32PackCount: this._f32PackCount,
      q8PackCount: this._q8PackCount,
    });
  }

  async execute(
    inputs: Record<string, RuntimeTypedArray>,
    options: WasmExecutionOptions = {},
  ): Promise<Record<string, RuntimeTypedArray>> {
    assertInferenceExecutionOptions(options, 'WASM inference');
    const incremental = incrementalExecutionEnabled(options, null);
    if (incremental && this._activationStorage === 'liveness') {
      throw new Error(
        'WASM incremental execution requires persistent activation storage.',
      );
    }
    /* Opt-in execution-phase profiling, the companion to the per-operator map
     * below. Summing operator times answers "which kernel is slow"; it cannot
     * answer "why is the request slower than its kernels", because everything
     * before the schedule loop is outside every operator timer. Set
     * `globalThis.__VOLVOX_WASM_PHASES` to a Map before execute(). */
    const phases = (globalThis as any).__VOLVOX_WASM_PHASES as
      Map<string, number> | undefined;
    const phase = phases
      ? (name: string, since: number) => {
          phases.set(name, (phases.get(name) || 0) + performance.now() - since);
        }
      : null;
    let mark = phase ? performance.now() : 0;
    this.graph.assertTopologyRevision?.(this.compiledTopologyRevision, "WASM");
    this._beginDecodeExecution(options);
    if ((this.graph.weightRevision || 0) !== this.compiledWeightRevision) {
      throw new Error("WASM weights changed after compilation; recompile the graph before execution.");
    }
    const preparedGraph = this._preparedGraphForExecution();
    const cacheWasValid = this._incrementalCacheValid;
    const selectedNodes = incremental
      ? incrementalNodeSelection(this.graph, inputs, options, cacheWasValid)
      : null;
    this._incrementalCacheValid = false;
    const rowPosition = incrementalRowPosition(options, selectedNodes, cacheWasValid);
    const incrementalRows = rowPosition == null
      ? null
      : prepareQuantizedRows(this.graph, selectedNodes, rowPosition, {
          changedInputs: options.changedInputs ?? Object.keys(inputs),
        });
    const copiedInputs = selectedNodes === null
      ? null
      : new Set(options.changedInputs ?? Object.keys(inputs));
    if (phase) { phase('prepare', mark); mark = performance.now(); }
    for (const [name, data] of Object.entries(inputs)) {
      const tensor = this.graph.tensors.get(name);
      if (!tensor?.isInput) throw new Error(`Unknown graph input '${name}'.`);
      Tensor.assertCompatibleInput(tensor.dtype, data, tensor.sizeBytes, `Input '${name}'`);
    }
    if (phase) { phase('validate-inputs', mark); mark = performance.now(); }
    this._preflightQGroupNormDynamicAffines(inputs);
    this._preflightQLayerNormDynamicAffines(inputs);
    this._preflightQSDPAMasks(inputs);
    this._preflightQMaskedMeanMasks(inputs);
    this._preflightRoPEPositions(inputs);
    if (phase) { phase('preflight', mark); mark = performance.now(); }
    for (const [name, data] of Object.entries(inputs)) {
      if (copiedInputs !== null && !copiedInputs.has(name)) continue;
      const tensor = this.graph.tensors.get(name)!;
      const ptr = this.pointers.get(name);
      wasmTensorView(tensor, this.mem.buffer, ptr).set(data);
    }
    if (phase) { phase('upload-inputs', mark); mark = performance.now(); }
    
    for (const step of preparedGraph.schedule) {
      const nodeIndex = step.nodeIndex;
      if (selectedNodes && !selectedNodes.has(nodeIndex)) continue;
      const node = this.graph.nodes[nodeIndex];
      const kernelRoute = step.kernelRoute;
      const decodeNode = incrementalRows?.get(nodeIndex) || null;
      // Opt-in per-operator profiling. Set `globalThis.__VOLVOX_WASM_PROFILE`
      // to a Map before execute() and each node's elapsed time accumulates
      // under its opType; leave it unset and this costs one truthiness check.
      // It exists because operator *choice* and operator *cost* diverge here:
      // a graph can carry strictly fewer, better-fused nodes and still lose if
      // one of them silently misses a fast kernel path, which is invisible to
      // any static inspection of the graph.
      const profile = (globalThis as any).__VOLVOX_WASM_PROFILE as
        Map<string, { ms: number; count: number }> | undefined;
      const profileStart = profile ? performance.now() : 0;
      try {
       const inPtr = this.pointers.get(node.inputs.input?.name)!;
       const outPtr = this.pointers.get(node.outputs.out?.name)!;
       const wPtr = this.pointers.get(node.inputs.weight?.name);
       const bPtr = this.pointers.get(node.inputs.bias?.name);

        if (kernelRoute === "qconv2d") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qconv2d') {
            throw new Error(`WASM QConv2D node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const inputPointer = this.pointers.get(descriptor.input.name);
          const outputPointer = this.pointers.get(descriptor.output.name);
          const biasPointer = descriptor.bias ? this.pointers.get(descriptor.bias.name) : 0;
          let result = 0;
          if (descriptor.packedWeightPointer && descriptor.im2colBytes > 0 &&
              this.qconvIm2ColPointer && descriptor.im2colBytes <= this.qconvIm2ColBytes) {
            result = this.api.qconv2d_im2col_i8u8(
              inputPointer, this.qconvIm2ColPointer,
              biasPointer, descriptor.weightZeroPointsPointer,
              descriptor.batch, descriptor.inputHeight, descriptor.inputWidth,
              descriptor.inputChannels, descriptor.outputHeight,
              descriptor.outputWidth, descriptor.outputChannels,
              descriptor.kernelHeight, descriptor.kernelWidth,
              descriptor.strideY, descriptor.strideX,
              descriptor.dilationY, descriptor.dilationX,
              descriptor.paddingTop, descriptor.paddingLeft,
              descriptor.paddingBottom, descriptor.paddingRight,
              descriptor.inputZeroPoint, descriptor.inputDtype,
              descriptor.weightDtype,
            );
            if (result === 1) {
              result = this.api.qlinear_i8u8_packed(
                this.qconvIm2ColPointer, descriptor.packedWeightPointer,
                biasPointer, descriptor.weightScalesPointer,
                descriptor.weightZeroPointsPointer, outputPointer,
                descriptor.packedRows, descriptor.packedDIn,
                descriptor.outputChannels,
                descriptor.inputScale, descriptor.inputZeroPoint,
                descriptor.outputScale, descriptor.outputZeroPoint,
                descriptor.inputDtype, descriptor.weightDtype,
                descriptor.outputDtype,
              );
            }
          }
          if (result !== 1) result = this.api.qconv2d_i8u8(
            inputPointer, this.pointers.get(descriptor.weight.name), biasPointer,
            descriptor.weightScalesPointer, descriptor.weightZeroPointsPointer,
            outputPointer, descriptor.batch, descriptor.inputHeight,
            descriptor.inputWidth, descriptor.inputChannels,
            descriptor.outputHeight, descriptor.outputWidth,
            descriptor.outputChannels, descriptor.kernelHeight,
            descriptor.kernelWidth, descriptor.inputPerGroup,
            descriptor.strideY, descriptor.strideX,
            descriptor.dilationY, descriptor.dilationX,
            descriptor.paddingTop, descriptor.paddingLeft,
            descriptor.paddingBottom, descriptor.paddingRight,
            descriptor.groups, descriptor.relu,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.weightDtype,
            descriptor.outputDtype,
          );
          if (result !== 1) {
            throw new Error(`WASM QConv2D node ${node.id} rejected its canonical W8A8 descriptor.`);
          }
        } else if (kernelRoute === "conv2d") {
          const wPtr = this.pointers.get(node.inputs.weight.name);
          const bPtr = node.inputs.bias ? this.pointers.get(node.inputs.bias.name) : 0;
          const inShape = node.inputs.input.shape;
          const outShape = node.outputs.out.shape;
          const wShape = node.inputs.weight.shape;
          const [sy, sx] = normalizeSpatialPair(node.params.stride, 1);
          const [dy, dx] = normalizeSpatialPair(node.params.dilation, 1);
          const padPair = normalizeSpatialPair(node.params.padding, 0);
          const pads = node.params.pads || [padPair[0], padPair[1], padPair[0], padPair[1]];
          const groups = node.params.groups || 1;
          const relu = node.params.relu ?? 0;
          this.api.conv2d_f32(
              inPtr, outPtr, wPtr, bPtr,
              inShape[0], inShape[1], inShape[2], inShape[3],
              wShape[0], wShape[1], wShape[2], wShape[3],
              outShape[1], outShape[2], sy, sx, pads[0], pads[1],
              groups, relu, dy, dx
          );
       } else if (kernelRoute === "conv-transpose2d") {
          const wPtr = this.pointers.get(node.inputs.weight.name);
          const bPtr = node.inputs.bias ? this.pointers.get(node.inputs.bias.name) : 0;
          const [b, in_h, in_w, in_c] = node.inputs.input.shape;
          const [out_b, out_h, out_w, out_c] = node.outputs.out.shape;
          const kh = node.params.kernel[0], kw = node.params.kernel[1];
          const sh = node.params.stride ? node.params.stride[0] : 1;
          const sw = node.params.stride ? node.params.stride[1] : 1;
          const ph = node.params.padding ? node.params.padding[0] : 0;
          const pw = node.params.padding ? node.params.padding[1] : 0;
          this.api.conv_transpose2d_f32(
              inPtr, wPtr, bPtr, outPtr,
              b, in_h, in_w, in_c, out_h, out_w, out_c,
              kh, kw, sh, sw, ph, pw
          );
       } else if (kernelRoute === "reduce-sum") {
          const input = node.inputs.input || node.inputs.data;
          const output = portableOutput(node);
          const inputPtr = this.pointers.get(input.name);
          const outputPtr = this.pointers.get(output?.name);
          const inShape = input.shape;
          const elements = portableTensorElements(input);
          const rows = portableElementCount(inShape.slice(0, -1));
          const width = inShape.at(-1);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
              rows == null || typeof width !== 'number' || !Number.isInteger(width) || width <= 0 ||
              elements !== rows * width ||
              portableTensorElements(output) !== rows || inputPtr == null || outputPtr == null) {
            throw new Error(`WASM ReduceSum node ${node.id} must reduce the last axis of an F32 tensor.`);
          }
          this.api.reduce_sum_f32(inputPtr, outputPtr, rows, width);
       } else if (kernelRoute === "reduce-mean") {
          const input = node.inputs.input || node.inputs.data;
          const output = portableOutput(node);
          const inputPtr = this.pointers.get(input.name);
          const outputPtr = this.pointers.get(output?.name);
          const inShape = input.shape;
          const elements = portableTensorElements(input);
          const rows = portableElementCount(inShape.slice(0, -1));
          const width = inShape.at(-1);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
              rows == null || typeof width !== 'number' || !Number.isInteger(width) || width <= 0 ||
              elements !== rows * width ||
              portableTensorElements(output) !== rows || inputPtr == null || outputPtr == null) {
            throw new Error(`WASM ReduceMean node ${node.id} must reduce the last axis of an F32 tensor.`);
          }
          this.api.reduce_mean_f32(inputPtr, outputPtr, rows, width);
       } else if (kernelRoute === "softmax") {
          const input = node.inputs.input || node.inputs.x;
          const output = portableOutput(node);
          const elements = portableTensorElements(input);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
              portableTensorElements(output) !== elements) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires equal-size F32 tensors.`);
          }
          let axis = node.params.axis ?? -1;
          if (axis < 0) axis += input.shape.length;
          if (input.shape.length === 0 || axis !== input.shape.length - 1) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires the last axis.`);
          }
          const width = input.shape[input.shape.length - 1];
          const rows = elements / width;
          if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(width) || width <= 0) {
            throw new Error(`WASM ${node.opType} node ${node.id} has invalid reduction dimensions.`);
          }
          const inputPtr = this.pointers.get(input.name);
          const outputPtr = this.pointers.get(output.name);
          if (node.opType === "Softmax") this.api.softmax_f32(inputPtr, outputPtr, rows, width);
          else this.api.logsoftmax_f32(inputPtr, outputPtr, rows, width);
       } else if (kernelRoute === "argmax") {
          const input = node.inputs.input || node.inputs.data;
          const output = node.outputs.out;
          let axis = node.params.axis !== undefined ? node.params.axis : 0;
          if (!input || !output || !Number.isInteger(axis)) {
            throw new Error(`WASM ArgMax node ${node.id} requires input, output, and an integer axis.`);
          }
          if (axis < 0) axis += input.shape.length;
          if (axis < 0 || axis >= input.shape.length) {
            throw new Error(`WASM ArgMax node ${node.id} axis ${node.params.axis} is outside rank ${input.shape.length}.`);
          }
          const axisSize = input.shape[axis];
          const outer = input.shape.slice(0, axis).reduce((product, dimension) => product * dimension, 1);
          const inner = input.shape.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
          const expectedElements = outer * inner;
          const outputElements = output.sizeBytes / Tensor.dtypeBytes(output.dtype);
          if (![axisSize, outer, inner, expectedElements, outputElements].every(Number.isSafeInteger) ||
              axisSize <= 0 || outer <= 0 || inner <= 0 || expectedElements !== outputElements ||
              axisSize > 0x7fffffff || outer > 0x7fffffff || inner > 0x7fffffff) {
            throw new Error(`WASM ArgMax node ${node.id} has incompatible input/output dimensions.`);
          }
          this.api.argmax_axis_typed(this.pointers.get(input.name), outPtr, outer, axisSize, inner,
              wasmKernelDtype(input, `ArgMax input at node ${node.id}`),
              wasmKernelDtype(output, `ArgMax output at node ${node.id}`));
       } else if (kernelRoute === "qlinear") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qlinear') {
            throw new Error(`WASM ${node.opType} node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeInput = decodeNode &&
            (decodeNode.inputs.input || decodeNode.inputs.x || decodeNode.inputs.a);
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const inputPointer = decodeNode
            ? wasmTensorPointer(decodeInput, this.mem, `${node.opType} row input`)
            : this.pointers.get(descriptor.input.name);
          const outputPointer = decodeNode
            ? wasmTensorPointer(decodeOutput, this.mem, `${node.opType} row output`)
            : this.pointers.get(descriptor.output.name);
          // The baseline packed SIMD128 kernel reuses each activation block
          // across eight outputs and is faster than the one-output Relaxed
          // child on the measured decoder shapes. Prefer it for every packed
          // descriptor; the optional child remains a fail-closed fallback.
          let result = typeof this.api.qlinear_i8u8_packed === 'function' &&
            descriptor.packedWeightPointer ? this.api.qlinear_i8u8_packed(
            inputPointer,
            descriptor.packedWeightPointer,
            this.pointers.get(descriptor.bias.name), descriptor.weightScalesPointer,
            descriptor.weightZeroPointsPointer,
            outputPointer,
            decodeNode ? 1 : descriptor.rows, descriptor.dIn, descriptor.dOut,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.weightDtype, descriptor.outputDtype,
          ) : 0;
          const relaxedApi = this.relaxedApi;
          if (result !== 1 && decodeNode && descriptor.packedWeightPointer &&
              this.relaxedSimdEnabled && relaxedApi) {
            try {
              result = relaxedApi[WASM_RELAXED_QLINEAR_EXPORT](
                inputPointer, this.pointers.get(descriptor.weight.name),
                descriptor.packedWeightPointer,
                this.pointers.get(descriptor.bias.name), descriptor.weightScalesPointer,
                descriptor.weightZeroPointsPointer, outputPointer,
                1, descriptor.dIn, descriptor.dOut,
                descriptor.inputScale, descriptor.inputZeroPoint,
                descriptor.outputScale, descriptor.outputZeroPoint,
                descriptor.inputDtype, descriptor.weightDtype, descriptor.outputDtype,
              );
            } catch {
              // A child trap must not make an otherwise-supported model fail.
              // Disable the broken child so it cannot trap once per layer and
              // token.  The baseline packed kernel overwrites the output row.
              this.relaxedSimdEnabled = false;
              result = 0;
            }
          }
          if (result !== 1) result = this.api.qlinear_i8u8(
            inputPointer,
            this.pointers.get(descriptor.weight.name),
            this.pointers.get(descriptor.bias.name), descriptor.weightScalesPointer,
            descriptor.weightZeroPointsPointer,
            outputPointer,
            decodeNode ? 1 : descriptor.rows, descriptor.dIn, descriptor.dOut,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.weightDtype, descriptor.outputDtype,
          );
          if (result !== 1) {
            throw new Error(`WASM ${node.opType} node ${node.id} rejected its canonical W8A8 descriptor.`);
          }
       } else if (kernelRoute === "batch-matmul") {
          const descriptor = this.nodeMetadata.get(node);
          if (!descriptor || descriptor.kind !== 'batchMatMul') {
            throw new Error(`WASM BatchMatMul node ${node.id} has no compiled portable descriptor.`);
          }
          const executionDescriptor = decodeNode ? descriptor.incrementalRow : descriptor;
          if (!executionDescriptor) {
            throw new Error(
              `WASM BatchMatMul node ${node.id} has no correctness-safe incremental row descriptor.`,
            );
          }
          const executionNode = decodeNode || node;
          const a = executionNode.inputs.a;
          const b = executionNode.inputs.b;
          const output = portableOutput(executionNode);
          const aPointer = decodeNode
            ? wasmTensorPointer(a, this.mem, 'BatchMatMul row input a')
            : this.pointers.get(a.name)!;
          const bPointer = decodeNode
            ? wasmTensorPointer(b, this.mem, 'BatchMatMul invariant input b')
            : this.pointers.get(b.name)!;
          const outputPointer = decodeNode
            ? wasmTensorPointer(output, this.mem, 'BatchMatMul row output')
            : this.pointers.get(output.name)!;
          for (let batch = 0; batch < executionDescriptor.outputBatchCount; batch++) {
            let remaining = batch;
            let aBase = 0;
            let bBase = 0;
            for (let axis = 0; axis < executionDescriptor.batchRank; axis++) {
              const coordinate = Math.floor(
                remaining / executionDescriptor.outputBatchStrides[axis],
              );
              remaining %= executionDescriptor.outputBatchStrides[axis];
              aBase += coordinate * executionDescriptor.aBatchStrides[axis];
              bBase += coordinate * executionDescriptor.bBatchStrides[axis];
            }
            this.api.matmul_f32(
              aPointer + aBase * Float32Array.BYTES_PER_ELEMENT,
              bPointer + bBase * Float32Array.BYTES_PER_ELEMENT,
              0,
              outputPointer + batch * executionDescriptor.m * executionDescriptor.n *
                Float32Array.BYTES_PER_ELEMENT,
              executionDescriptor.m, executionDescriptor.k, executionDescriptor.n,
            );
          }
       } else if (kernelRoute === "qbatch-matmul") {
          const descriptor = this.nodeMetadata.get(node);
          if (!descriptor || descriptor.kind !== 'qBatchMatMul') {
            throw new Error(`WASM QBatchMatMul node ${node.id} has no compiled portable descriptor.`);
          }
          const executionDescriptor = decodeNode ? descriptor.incrementalRow : descriptor;
          if (!executionDescriptor) {
            throw new Error(
              `WASM QBatchMatMul node ${node.id} has no correctness-safe incremental row descriptor.`,
            );
          }
          const executionNode = decodeNode || node;
          const a = executionNode.inputs.a;
          const b = executionNode.inputs.b;
          const output = portableOutput(executionNode);
          const aPointer = decodeNode
            ? wasmTensorPointer(a, this.mem, 'QBatchMatMul row input a')
            : this.pointers.get(a.name)!;
          const bPointer = decodeNode
            ? wasmTensorPointer(b, this.mem, 'QBatchMatMul invariant input b')
            : this.pointers.get(b.name)!;
          const outputPointer = decodeNode
            ? wasmTensorPointer(output, this.mem, 'QBatchMatMul row output')
            : this.pointers.get(output.name)!;
          for (let batch = 0; batch < executionDescriptor.outputBatchCount; batch++) {
            let remaining = batch;
            let aBase = 0;
            let bBase = 0;
            for (let axis = 0; axis < executionDescriptor.batchRank; axis++) {
              const coordinate = Math.floor(
                remaining / executionDescriptor.outputBatchStrides[axis],
              );
              remaining %= executionDescriptor.outputBatchStrides[axis];
              aBase += coordinate * executionDescriptor.aBatchStrides[axis];
              bBase += coordinate * executionDescriptor.bBatchStrides[axis];
            }
            // Standard SIMD128 is part of the shipped parent module, not the
            // optional Relaxed-SIMD child.  Keep the scalar ABI selectable for
            // older/custom parents and tests; a selected kernel rejection is
            // fail-closed and never retries through JS or dequantized F32.
            const kernel = this.qbatchMatMulSimdEnabled &&
              typeof this.api.qbatch_matmul_i8u8_simd128 === 'function'
              ? this.api.qbatch_matmul_i8u8_simd128
              : this.api.qbatch_matmul_i8u8;
            const result = kernel(
              aPointer + aBase,
              bPointer + bBase,
              outputPointer + batch * executionDescriptor.m * executionDescriptor.n,
              executionDescriptor.m,
              executionDescriptor.k,
              executionDescriptor.n,
              executionDescriptor.aScale,
              executionDescriptor.aQuantization.zero_point,
              executionDescriptor.bScale,
              executionDescriptor.bQuantization.zero_point,
              executionDescriptor.outputScale,
              executionDescriptor.outputQuantization.zero_point,
              wasmDtypeCode(a.dtype),
              wasmDtypeCode(b.dtype),
              wasmDtypeCode(output.dtype),
            );
            if (result !== 1) {
              throw new Error(
                `WASM QBatchMatMul node ${node.id} rejected its canonical descriptor.`,
              );
            }
          }
       } else if (kernelRoute === "dense") {
          const executionNode = decodeNode || node;
          const input = executionNode.inputs.input || executionNode.inputs.x || executionNode.inputs.a;
          const weight = executionNode.inputs.weight;
          const output = portableOutput(executionNode);
          const inputPtr = decodeNode
            ? wasmTensorPointer(input, this.mem, `${node.opType} row input`)
            : this.pointers.get(input?.name);
          const weightPtr = this.pointers.get(weight?.name);
          const biasPtr = executionNode.inputs.bias
            ? this.pointers.get(executionNode.inputs.bias.name) : 0;
          if (!portableF32Tensor(input) || !weight || !portableF32Tensor(output) ||
              input.shape.length < 1 || output.shape.length < 1) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires F32 input and output tensors.`);
          }
          const dIn = input.shape.at(-1);
          const dOut = output.shape.at(-1);
          const rows = input.shape.slice(0, -1).reduce((product, dimension) => product * dimension, 1);
          if (!Number.isSafeInteger(rows) || rows <= 0 || dIn == null || dOut == null ||
              !Number.isInteger(dIn) || dIn <= 0 || !Number.isInteger(dOut) || dOut <= 0 ||
              portableTensorElements(input) !== rows * dIn ||
              portableTensorElements(output) !== rows * dOut) {
            throw new Error(`WASM ${node.opType} node ${node.id} has incompatible matrix dimensions.`);
          }
          const scale = executionNode.inputs.scale || executionNode.inputs.weight_scale;
          const zeroPoint = executionNode.inputs.zero_point ||
            executionNode.inputs.weight_zero_point || null;
          const bias = executionNode.inputs.bias || null;
          const outputPtr = decodeNode
            ? wasmTensorPointer(output, this.mem, `${node.opType} row output`)
            : this.pointers.get(output.name);
          if (scale) {
             const scaleElements = portableTensorElements(scale);
             const zeroPointElements = zeroPoint ? portableTensorElements(zeroPoint) : 0;
             if (!['int8', 'uint8'].includes(weight.dtype) || scaleElements == null ||
                 portableTensorElements(weight) !== dIn * dOut ||
                 !sameShape(weight.shape, [dOut, dIn]) ||
                 !portableF32Tensor(scale) || (scaleElements !== 1 && scaleElements !== dOut) ||
                 (zeroPoint && (zeroPointElements == null || (zeroPointElements !== 1 && zeroPointElements !== dOut))) ||
                 (bias && (!portableF32Tensor(bias) || !sameShape(bias.shape, [dOut])))) {
               throw new Error(`WASM quantized ${node.opType} node ${node.id} requires [d_out,d_in] I8/U8 weights, F32 scale, and optional typed zero point/bias.`);
             }
             const descriptor = this.nodeMetadata.get(node);
             let result = descriptor?.kind === 'w8a32' &&
               descriptor.packedWeightPointer &&
               typeof this.api.matmul_quantized_f32_packed === 'function'
               ? this.api.matmul_quantized_f32_packed(
                 inputPtr, descriptor.packedWeightPointer, this.pointers.get(scale.name),
                 zeroPoint ? this.pointers.get(zeroPoint.name) : 0, biasPtr, outputPtr,
                 rows, dIn, dOut, wasmDtypeCode(weight.dtype), scaleElements,
                 zeroPoint ? wasmDtypeCode(zeroPoint.dtype) : 0, zeroPointElements ?? 0,
               ) : 0;
             if (result !== 1) result = this.api.matmul_quantized_f32(
               inputPtr, weightPtr, this.pointers.get(scale.name),
               zeroPoint ? this.pointers.get(zeroPoint.name) : 0, biasPtr, outputPtr,
               rows, dIn, dOut, wasmDtypeCode(weight.dtype), scaleElements,
               zeroPoint ? wasmDtypeCode(zeroPoint.dtype) : 0, zeroPointElements ?? 0,
             );
             if (result !== 1) throw new Error(`WASM quantized ${node.opType} node ${node.id} rejected its canonical descriptor.`);
          } else {
             const doutFirst = node.wLayout ? node.wLayout === 'dout' :
               (weight.shape[0] === dOut && weight.shape[1] === dIn);
             const expectedWeightShape = doutFirst ? [dOut, dIn] : [dIn, dOut];
             if (!portableF32Tensor(weight) || !sameShape(weight.shape, expectedWeightShape) ||
                 !portableDenseBias(bias, dOut)) {
               throw new Error(`WASM ${node.opType} node ${node.id} requires a canonical F32 weight and optional bias.`);
             }
             const packedF32 = this.f32PackedNodes.get(node);
             if (packedF32) {
               if (packedF32.dIn !== dIn || packedF32.dOut !== dOut ||
                   packedF32.doutFirst !== doutFirst ||
                   this.api.gemm_f32_packed(inputPtr, packedF32.pointer, biasPtr,
                     outputPtr, rows, dIn, dOut) !== 1) {
                 throw new Error(`WASM ${node.opType} node ${node.id} rejected its packed F32 descriptor.`);
               }
             } else if (doutFirst) {
               this.api.linear_f32(inputPtr, weightPtr, biasPtr, outputPtr, rows, dIn, dOut);
             } else {
               this.api.matmul_f32(inputPtr, weightPtr, biasPtr, outputPtr, rows, dIn, dOut);
             }
          }
       } else if (kernelRoute === "layernorm") {
          const executionNode = decodeNode || node;
          const input = executionNode.inputs.input;
          const output = portableOutput(executionNode);
          const weight = executionNode.inputs.weight;
          const bias = executionNode.inputs.bias || null;
          const flatSeq = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
          const dModel = executionNode.params?.d_model ?? input.shape.at(-1);
          if (!portableF32Tensor(input) || !portableF32Tensor(weight) ||
              !portableF32Tensor(output) || (bias && !portableF32Tensor(bias)) ||
              !Number.isInteger(dModel) || dModel <= 0 || input.shape.at(-1) !== dModel ||
              !sameShape(weight.shape, [dModel]) || (bias && !sameShape(bias.shape, [dModel])) ||
              !sameShape(input.shape, output.shape)) {
            throw new Error(`WASM LayerNorm node ${node.id} requires matching F32 rows and affine tensors.`);
          }
          this.api.layernorm_f32(
            decodeNode ? wasmTensorPointer(input, this.mem, 'LayerNorm row input') :
              this.pointers.get(input.name),
            this.pointers.get(weight.name), bias ? this.pointers.get(bias.name) : 0,
            decodeNode ? wasmTensorPointer(output, this.mem, 'LayerNorm row output') :
              this.pointers.get(output.name),
            flatSeq, dModel, executionNode.params?.eps ?? 1e-6,
          );
       } else if (kernelRoute === "rmsnorm") {
          const input = node.inputs.input || node.inputs.x;
          const weight = node.inputs.weight;
          const output = portableOutput(node);
          const dModel = node.params?.d_model ?? input?.shape?.at(-1);
          const elements = portableTensorElements(input);
          if (!portableF32Tensor(input) || !portableF32Tensor(weight) || !portableF32Tensor(output) ||
              !Number.isInteger(dModel) || dModel <= 0 || elements == null || elements % dModel !== 0 ||
              !sameShape(weight.shape, [dModel]) || portableTensorElements(output) !== elements) {
            throw new Error(`WASM RMSNorm node ${node.id} requires F32 rows with a matching weight vector.`);
          }
          this.api.rmsnorm_f32(this.pointers.get(input.name), this.pointers.get(weight.name),
            this.pointers.get(output.name), elements / dModel, dModel, node.params?.eps ?? 1e-6);
       } else if (kernelRoute === "groupnorm") {
          const descriptor = portableGroupNormDescriptor(node);
          const result = this.api.groupnorm_f32(
            this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.weight.name),
            this.pointers.get(descriptor.bias.name), this.pointers.get(descriptor.output.name),
            descriptor.batch, descriptor.height, descriptor.width, descriptor.channels,
            descriptor.groups, descriptor.eps,
          );
          if (result !== 1) throw new Error(`WASM GroupNorm node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "moe-router") {
          const descriptor = portableMoERouterDescriptor(node);
          const result = this.api.moe_router_f32(
            this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.weight.name),
            descriptor.bias ? this.pointers.get(descriptor.bias.name) : 0,
            this.pointers.get(descriptor.indices.name), this.pointers.get(descriptor.routeWeights.name),
            descriptor.rows, descriptor.dModel, descriptor.experts, descriptor.topK,
            descriptor.temperature, descriptor.normalize,
          );
          if (result !== 1) throw new Error(`WASM MoERouter node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "moe-linear") {
          let metadata = this.nodeMetadata.get(node);
          if (metadata?.kind !== 'moeLinear') {
            /* Preserve the historical direct-engine contract for callers that
             * install an allocated graph without compile(). Normal compiled
             * and rebound graphs already own this metadata in their variant
             * arena, so this fallback is not on the provider path. */
            const compiled = this._compilePortableMetadata(node);
            if (compiled?.kind === 'moeLinear') {
              this.nodeMetadata.set(node, compiled);
              metadata = compiled;
            }
          }
          const descriptor: any = metadata?.kind === 'moeLinear'
            ? metadata
            : portableMoELinearDescriptor(node);
          const hasSlotTable = descriptor.residentSlots != null;
          if (hasSlotTable &&
              (!Number.isSafeInteger(descriptor.slotTablePointer) ||
               !Number.isSafeInteger(descriptor.slotTableDomain))) {
            throw new Error(`WASM MoELinear node ${node.id} has no compiled resident slot table.`);
          }
          const result = !hasSlotTable
            ? this.api.moe_linear_f32(
              this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.expertWeight.name),
              descriptor.expertBias ? this.pointers.get(descriptor.expertBias.name) : 0,
              this.pointers.get(descriptor.routeIndices.name), this.pointers.get(descriptor.routeWeights.name),
              this.pointers.get(descriptor.output.name), descriptor.rows, descriptor.dIn, descriptor.dOut,
              descriptor.experts, descriptor.topK,
            )
            : this.api.vx_moe_linear_banked_f32(
              this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.expertWeight.name),
              descriptor.expertBias ? this.pointers.get(descriptor.expertBias.name) : 0,
              this.pointers.get(descriptor.routeIndices.name), this.pointers.get(descriptor.routeWeights.name),
              this.pointers.get(descriptor.output.name), descriptor.rows, descriptor.dIn, descriptor.dOut,
              descriptor.experts, descriptor.topK,
              descriptor.slotTablePointer, descriptor.slotTableDomain,
            );
          if (result !== 1) {
            throw new Error(!hasSlotTable
              ? `WASM MoELinear node ${node.id} rejected its canonical descriptor.`
              : `WASM MoELinear node ${node.id} routed to an expert that is not ` +
                `resident in this context.`);
          }
       } else if (kernelRoute === "sdpa") {
          const qkvPtr = this.pointers.get(node.inputs.qkv.name)!;
          const qkvShape = node.inputs.qkv.shape;
          const outputShape = node.outputs.out.shape;
          const rank = qkvShape.length;
          const batch = rank === 2 ? 1 : qkvShape[0];
          const seqLen = qkvShape[rank - 2];
          const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
          const heads = node.params.heads || node.params.num_heads || 8;
          const head_dim = node.params.head_dim || d_model / heads;
          const scale = node.params.scale !== undefined ? node.params.scale : 1.0 / Math.sqrt(head_dim);
          const mask = node.inputs.mask;
          const maskMode = attentionMaskMode(mask, batch, seqLen, seqLen, node.id);
          const causal = node.params.causal === false ? 0 : 1;
          const outputBatch = outputShape.length === 2 ? 1 : outputShape[0];
          if ((rank !== 2 && rank !== 3) || outputShape.length !== rank || batch !== outputBatch ||
              qkvShape[rank - 1] !== 3 * d_model || outputShape[rank - 2] !== seqLen ||
              !Number.isInteger(head_dim) || head_dim <= 0) {
            throw new Error(`WASM SDPA requires compatible [batch?, seq, 3*d_model] input/output tensors.`);
          }
          const qkvBatchBytes = seqLen * 3 * d_model * 4;
          const outputBatchBytes = seqLen * d_model * 4;
          for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
            const [maskPtr, localMaskMode] = localMaskBinding(
              this.pointers, mask, maskMode, batchIndex, seqLen, seqLen);
            this.api.sdpa_f32(qkvPtr + batchIndex * qkvBatchBytes,
                outPtr + batchIndex * outputBatchBytes,
                seqLen, d_model, heads, head_dim, scale, maskPtr, localMaskMode, causal);
          }
       } else if (kernelRoute === "cross-sdpa") {
          const executionNode = decodeNode || node;
          const q = executionNode.inputs.q;
          const k = executionNode.inputs.k;
          const v = executionNode.inputs.v;
          const output = portableOutput(executionNode);
          const pointer = (tensor, label): number => {
            const value = decodeNode
              ? wasmTensorPointer(tensor, this.mem, label)
              : this.pointers.get(tensor.name);
            if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
              throw new Error(`WASM ${label} has no valid heap pointer.`);
            }
            return value;
          };
          const qPtr = pointer(q, 'CrossSDPA row q');
          const kPtr = pointer(k, 'CrossSDPA row k');
          const vPtr = pointer(v, 'CrossSDPA row v');
          const outputPtr = pointer(output, 'CrossSDPA row output');
          const qShape = q.shape;
          const kShape = k.shape;
          const vShape = v.shape;
          const outputShape = output.shape;
          const rank = qShape.length;
          const batch = rank === 2 ? 1 : qShape[0];
          const batchOf = (shape) => shape.length === 2 ? 1 : shape[0];
          const seqQ = qShape[rank - 2];
          const seqKV = kShape[rank - 2];
          const d_model = outputShape[outputShape.length - 1];
          const heads = executionNode.params.heads || executionNode.params.num_heads || 8;
          const head_dim = executionNode.params.head_dim || d_model / heads;
          if ((rank !== 2 && rank !== 3) ||
              [kShape, vShape, outputShape].some((shape) => shape.length !== rank || batchOf(shape) !== batch) ||
              qShape[rank - 1] !== d_model || kShape[rank - 1] !== d_model ||
              vShape[rank - 1] !== d_model || vShape[rank - 2] !== seqKV ||
              outputShape[rank - 2] !== seqQ || !Number.isInteger(head_dim) || head_dim <= 0) {
            throw new Error(`WASM CrossSDPA requires compatible [batch?, seq, d_model] tensors.`);
          }
          const qBatchBytes = seqQ * d_model * 4;
          const kvBatchBytes = seqKV * d_model * 4;
          const scale = executionNode.params.scale ?? 1.0 / Math.sqrt(head_dim);
          const mask = executionNode.inputs.mask;
          const maskMode = attentionMaskMode(mask, batch, seqQ, seqKV, node.id);
          const causal = executionNode.params.causal === true ? 1 : 0;
          for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
            const [maskPtr, localMaskMode] = localMaskBinding(
              this.pointers, mask, maskMode, batchIndex, seqQ, seqKV,
              decodeNode && mask ? wasmTensorPointer(mask, this.mem, 'CrossSDPA row mask') : null,
            );
            this.api.cross_sdpa_f32(qPtr + batchIndex * qBatchBytes,
                kPtr + batchIndex * kvBatchBytes, vPtr + batchIndex * kvBatchBytes,
                outputPtr + batchIndex * qBatchBytes,
                seqQ, seqKV, d_model, heads, head_dim, scale, maskPtr, localMaskMode, causal);
          }
       } else if (kernelRoute === "cross-attention") {
          const q = node.inputs.q;
          const kv = node.inputs.kv;
          const weight = node.inputs.weight;
          const scale = node.inputs.scale || null;
          const bias = node.inputs.bias || null;
          const output = node.outputs.out;
          if (!q || !kv || !weight || !output) {
            throw new Error(`WASM CrossAttention node ${node.id} requires q, kv, weight, and out tensors.`);
          }
          const rank = q?.shape?.length;
          const batch = rank === 3 ? q.shape[0] : 1;
          const kvBatch = kv?.shape?.length === 3 ? kv.shape[0] : 1;
          const outputBatch = output?.shape?.length === 3 ? output.shape[0] : 1;
          const seqQ = q?.shape?.[rank - 2];
          const seqKV = kv?.shape?.[rank - 2];
          const dModel = output?.shape?.[rank - 1];
          const heads = node.params?.heads || 8;
          const headDim = dModel / heads;
          const expectedWeightElements = 3 * dModel * dModel;
          if ((rank !== 2 && rank !== 3) || kv.shape.length !== rank || output.shape.length !== rank ||
              batch <= 0 || batch !== kvBatch || batch !== outputBatch || q.shape[rank - 1] !== dModel ||
              kv.shape[rank - 1] !== dModel || output.shape[rank - 2] !== seqQ ||
              q.dtype !== 'float32' || kv.dtype !== 'float32' || weight?.dtype !== 'float32' ||
              output.dtype !== 'float32' || !Number.isInteger(heads) || heads <= 0 ||
              !Number.isInteger(seqQ) || seqQ <= 0 || !Number.isInteger(seqKV) || seqKV <= 0 ||
              !Number.isInteger(dModel) || dModel <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
              weight.sizeBytes / 4 !== expectedWeightElements ||
              (scale && (scale.dtype !== 'float32' || scale.sizeBytes / 4 !== 3 * dModel)) ||
              (bias && (bias.dtype !== 'float32' || bias.sizeBytes / 4 !== 3 * dModel))) {
            throw new Error(`WASM CrossAttention node ${node.id} requires F32 Q/KV/output and [3*d_model,d_model] projection parameters.`);
          }
          this.api.cross_attention_f32(
            this.pointers.get(q.name), this.pointers.get(kv.name), this.pointers.get(weight.name),
            scale ? this.pointers.get(scale.name) : 0, bias ? this.pointers.get(bias.name) : 0, outPtr,
            batch, seqQ, seqKV, dModel, heads, headDim,
          );
       } else if (kernelRoute === "qembedding") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qembedding') {
            throw new Error(`WASM QEmbedding node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeInput = decodeNode?.inputs?.input;
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const result = this.api.qembedding_i8u8(
            decodeNode ? wasmTensorPointer(decodeInput, this.mem, 'QEmbedding row IDs') :
              this.pointers.get(descriptor.input.name),
            this.pointers.get(descriptor.weight.name),
            descriptor.weightScalesPointer, descriptor.weightZeroPointsPointer,
            decodeNode ? wasmTensorPointer(decodeOutput, this.mem, 'QEmbedding row output') :
              this.pointers.get(descriptor.output.name),
            decodeNode ? 1 : descriptor.tokens, descriptor.vocab, descriptor.hidden,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.weightDtype, descriptor.outputDtype,
          );
          if (result !== 1) {
            throw new Error(`WASM QEmbedding node ${node.id} rejected its canonical W8A8 descriptor or token IDs.`);
          }
       } else if (kernelRoute === "embedding") {
          const executionNode = decodeNode || node;
          const input = executionNode.inputs.input;
          const weight = executionNode.inputs.weight;
          const output = portableOutput(executionNode);
          if (input.dtype !== 'int32' || weight.dtype !== 'float32' ||
              output.dtype !== 'float32' || weight.shape.length !== 2) {
            throw new Error(`WASM Embedding node ${node.id} requires int32 IDs and a rank-2 float32 table.`);
          }
          const seqLen = input.shape.reduce((a, b) => a * b, 1);
          const d_model = output.shape[output.shape.length - 1];
          const vocabularySize = weight.shape[0];
          if (weight.shape[1] !== d_model || output.sizeBytes !== seqLen * d_model * 4) {
            throw new Error(`WASM Embedding node ${node.id} shapes are incompatible.`);
          }
          const inputPtr = decodeNode
            ? wasmTensorPointer(input, this.mem, 'Embedding row IDs')
            : this.pointers.get(input.name);
          const outputPtr = decodeNode
            ? wasmTensorPointer(output, this.mem, 'Embedding row output')
            : this.pointers.get(output.name);
          const ids = new Int32Array(this.mem.buffer, inputPtr, seqLen);
          for (const id of ids) {
            if (id < 0 || id >= vocabularySize) {
              throw new Error(`WASM Embedding node ${node.id} token id ${id} is outside vocabulary size ${vocabularySize}.`);
            }
          }
          this.api.embedding_f32(
            inputPtr, this.pointers.get(weight.name), outputPtr, seqLen, d_model,
          );
       } else if (kernelRoute === "relu") {
          const inShape = node.inputs.input.shape;
          const elements = inShape.reduce((a, b) => a * b, 1);
          this.api.relu_f32(inPtr, outPtr, elements);
       } else if (kernelRoute === "gelu") {
          const executionNode = decodeNode || node;
          const input = executionNode.inputs.input;
          const output = portableOutput(executionNode);
          const elements = portableTensorElements(input);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
              portableTensorElements(output) !== elements || !sameShape(input.shape, output.shape)) {
            throw new Error(`WASM GELU node ${node.id} requires equal-shape F32 tensors.`);
          }
          const fn = geluApproximation(executionNode) === 'tanh'
            ? this.api.gelu_tanh_f32 : this.api.gelu_f32;
          if (typeof fn !== 'function') throw new Error(`WASM GELU ${node.params?.approximate ?? 'none'} kernel is unavailable.`);
          fn(
            decodeNode ? wasmTensorPointer(input, this.mem, 'GELU row input') :
              this.pointers.get(input.name),
            decodeNode ? wasmTensorPointer(output, this.mem, 'GELU row output') :
              this.pointers.get(output.name),
            elements,
          );
       } else if (kernelRoute === "silu") {
          const executionNode = decodeNode || node;
          const input = executionNode.inputs.input || executionNode.inputs.x;
          const output = portableOutput(executionNode);
          const elements = portableTensorElements(input);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) ||
              elements == null || portableTensorElements(output) !== elements) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires equal-size F32 tensors.`);
          }
          this.api.silu_f32(
            decodeNode ? wasmTensorPointer(input, this.mem, `${node.opType} row input`) :
              this.pointers.get(input.name),
            decodeNode ? wasmTensorPointer(output, this.mem, `${node.opType} row output`) :
              this.pointers.get(output.name),
            elements,
          );
       } else if (kernelRoute === "qsilu") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qsilu') {
            throw new Error(`WASM QSiLU node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeInput = decodeNode &&
            (decodeNode.inputs.input || decodeNode.inputs.x || decodeNode.inputs.data);
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const result = this.api.qsilu_i8u8(
            decodeNode ? wasmTensorPointer(decodeInput, this.mem, 'QSiLU row input') :
              this.pointers.get(descriptor.input.name),
            decodeNode ? wasmTensorPointer(decodeOutput, this.mem, 'QSiLU row output') :
              this.pointers.get(descriptor.output.name),
            decodeNode ? decodeOutput.buffer.length : descriptor.elements,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.outputDtype,
          );
          if (result !== 1) throw new Error(`WASM QSiLU node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qgelu") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qgelu') {
            throw new Error(`WASM QGELU node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeInput = decodeNode &&
            (decodeNode.inputs.input || decodeNode.inputs.x || decodeNode.inputs.data);
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const result = this.api.qgelu_i8u8(
            decodeNode ? wasmTensorPointer(decodeInput, this.mem, 'QGELU row input') :
              this.pointers.get(descriptor.input.name),
            decodeNode ? wasmTensorPointer(decodeOutput, this.mem, 'QGELU row output') :
              this.pointers.get(descriptor.output.name),
            decodeNode ? decodeOutput.buffer.length : descriptor.elements,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.outputDtype,
          );
          if (result !== 1) throw new Error(`WASM QGELU node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qgroupnorm") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qgroupnorm') {
            throw new Error(`WASM QGroupNorm node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const result = this.api.qgroupnorm_i8u8(
            this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.weight.name),
            this.pointers.get(descriptor.bias.name), this.pointers.get(descriptor.output.name),
            descriptor.batch, descriptor.height, descriptor.width, descriptor.channels, descriptor.groups,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint, descriptor.epsilon,
            descriptor.inputDtype, descriptor.outputDtype,
          );
          if (result !== 1) throw new Error(`WASM QGroupNorm node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qlayernorm") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qlayernorm') {
            throw new Error(`WASM QLayerNorm node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeInput = decodeNode?.inputs?.input;
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const result = this.api.qlayernorm_i8u8(
            decodeNode ? wasmTensorPointer(decodeInput, this.mem, 'QLayerNorm row input') :
              this.pointers.get(descriptor.input.name),
            this.pointers.get(descriptor.weight.name), this.pointers.get(descriptor.bias.name),
            decodeNode ? wasmTensorPointer(decodeOutput, this.mem, 'QLayerNorm row output') :
              this.pointers.get(descriptor.output.name),
            decodeNode ? 1 : descriptor.rows, descriptor.dModel,
            descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint, descriptor.epsilon,
            descriptor.inputDtype, descriptor.outputDtype,
          );
          if (result !== 1) throw new Error(`WASM QLayerNorm node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qsdpa") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qsdpa' || typeof this.api.qsdpa_i8u8 !== 'function') {
            throw new Error(`WASM QSDPA node ${node.id} has no canonical W8A8 kernel descriptor.`);
          }
          const executionDescriptor = (decodeNode ? portableQSDPADescriptor(decodeNode) : descriptor)!;
          const executionPointer = (tensor, label) => decodeNode
            ? wasmTensorPointer(tensor, this.mem, label)
            : this.pointers.get(tensor.name);
          const result = this.api.qsdpa_i8u8(
            executionPointer(executionDescriptor.q, 'QSDPA row q'),
            executionPointer(executionDescriptor.k, 'QSDPA row k'),
            executionPointer(executionDescriptor.v, 'QSDPA row v'),
            executionDescriptor.mask ? executionPointer(executionDescriptor.mask, 'QSDPA row mask') : 0,
            executionPointer(executionDescriptor.output, 'QSDPA row output'),
            executionDescriptor.batch, executionDescriptor.seqQ, executionDescriptor.seqKV,
            executionDescriptor.dModel, executionDescriptor.heads,
            executionDescriptor.qScale, executionDescriptor.qZeroPoint,
            executionDescriptor.kScale, executionDescriptor.kZeroPoint,
            executionDescriptor.vScale, executionDescriptor.vZeroPoint,
            executionDescriptor.outputScale, executionDescriptor.outputZeroPoint,
            executionDescriptor.attentionScale, executionDescriptor.qDtype,
            executionDescriptor.kDtype, executionDescriptor.vDtype,
            executionDescriptor.outputDtype, executionDescriptor.causal, executionDescriptor.maskMode,
          );
          if (result !== 1) throw new Error(`WASM QSDPA node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qargmax") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qargmax' || typeof this.api.qargmax_i8u8 !== 'function') {
            throw new Error(`WASM QArgMax node ${node.id} has no canonical I8/U8 kernel descriptor.`);
          }
          const executionDescriptor = (decodeNode ? portableQArgMaxDescriptor(decodeNode) : descriptor)!;
          const result = this.api.qargmax_i8u8(
            decodeNode ? wasmTensorPointer(executionDescriptor.input, this.mem, 'QArgMax row input') :
              this.pointers.get(executionDescriptor.input.name),
            decodeNode ? wasmTensorPointer(executionDescriptor.output, this.mem, 'QArgMax row output') :
              this.pointers.get(executionDescriptor.output.name),
            executionDescriptor.outer, executionDescriptor.axisSize, executionDescriptor.inner,
            executionDescriptor.inputDtype,
          );
          if (result !== 1) throw new Error(`WASM QArgMax node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "qmaskedmean") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qmaskedmean' || typeof this.api.qmaskedmean_i8u8 !== 'function') {
            throw new Error(`WASM QMaskedMean node ${node.id} has no canonical W8A8 kernel descriptor.`);
          }
          const result = this.api.qmaskedmean_i8u8(
            this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.mask.name),
            this.pointers.get(descriptor.output.name), descriptor.batch, descriptor.sequence,
            descriptor.width, descriptor.inputScale, descriptor.inputZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.inputDtype, descriptor.outputDtype,
          );
          if (result !== 1) throw new Error(`WASM QMaskedMean node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "tanh") {
          const input = node.inputs.input || node.inputs.x;
          const output = portableOutput(node);
          const elements = portableTensorElements(input);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) ||
              elements == null || portableTensorElements(output) !== elements) {
            throw new Error(`WASM Tanh node ${node.id} requires equal-size F32 tensors.`);
          }
          this.api.tanh_f32(this.pointers.get(input.name), this.pointers.get(output.name), elements);
       } else if (kernelRoute === "trigonometric") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'unaryF32') {
            throw new Error(`WASM ${node.opType} node ${node.id} has no portable descriptor.`);
          }
          const fn = node.opType === 'Sin' ? this.api.sin_f32 : this.api.cos_f32;
          fn(this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.output.name),
            descriptor.elements);
       } else if (kernelRoute === "rope") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'rope' || typeof this.api.rope_f32 !== 'function') {
            throw new Error(`WASM RoPE node ${node.id} has no portable descriptor.`);
          }
          const result = this.api.rope_f32(
            this.pointers.get(descriptor.input.name),
            descriptor.positions ? this.pointers.get(descriptor.positions.name) : 0,
            this.pointers.get(descriptor.output.name), descriptor.batch, descriptor.sequence,
            descriptor.width, descriptor.rotaryWidth, descriptor.theta,
            descriptor.positionOffset, descriptor.interleaved ? 1 : 0, descriptor.positionMode,
          );
          if (result !== 1) throw new Error(`WASM RoPE node ${node.id} rejected its descriptor.`);
       } else if (kernelRoute === "ssm-scan") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'ssmScan' || typeof this.api.ssm_scan_f32 !== 'function') {
            throw new Error(`WASM SSMScan node ${node.id} has no portable descriptor.`);
          }
          const result = this.api.ssm_scan_f32(
            this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.delta.name),
            this.pointers.get(descriptor.a.name), this.pointers.get(descriptor.b.name),
            this.pointers.get(descriptor.c.name),
            descriptor.d ? this.pointers.get(descriptor.d.name) : 0,
            descriptor.z ? this.pointers.get(descriptor.z.name) : 0,
            descriptor.initialState ? this.pointers.get(descriptor.initialState.name) : 0,
            this.pointers.get(descriptor.output.name),
            descriptor.finalState ? this.pointers.get(descriptor.finalState.name) : 0,
            descriptor.stateScratchPointer, descriptor.batch, descriptor.sequence,
            descriptor.channels, descriptor.stateWidth, descriptor.bMode, descriptor.cMode,
            descriptor.deltaSoftplus ? 1 : 0,
          );
          if (result !== 1) throw new Error(`WASM SSMScan node ${node.id} rejected its descriptor.`);
       } else if (kernelRoute === "qadd") {
          const descriptor = this.nodeMetadata.get(node);
          if (descriptor?.kind !== 'qadd') {
            throw new Error(`WASM QAdd node ${node.id} has no canonical W8A8 descriptor.`);
          }
          const decodeA = decodeNode &&
            (decodeNode.inputs.a || decodeNode.inputs.input || decodeNode.inputs.x);
          const decodeB = decodeNode && (decodeNode.inputs.b || decodeNode.inputs.y);
          const decodeOutput = decodeNode && portableOutput(decodeNode);
          const result = this.api.qadd_i8u8(
            decodeNode ? wasmTensorPointer(decodeA, this.mem, 'QAdd row input a') :
              this.pointers.get(descriptor.a.name),
            decodeNode ? wasmTensorPointer(decodeB, this.mem, 'QAdd row input b') :
              this.pointers.get(descriptor.b.name),
            decodeNode ? wasmTensorPointer(decodeOutput, this.mem, 'QAdd row output') :
              this.pointers.get(descriptor.output.name),
            decodeNode ? decodeOutput.buffer.length : descriptor.elements,
            descriptor.aScale, descriptor.aZeroPoint, descriptor.bScale, descriptor.bZeroPoint,
            descriptor.outputScale, descriptor.outputZeroPoint,
            descriptor.aDtype, descriptor.bDtype, descriptor.outputDtype, descriptor.relu,
          );
          if (result !== 1) throw new Error(`WASM QAdd node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "binary") {
          const descriptor = this.nodeMetadata.get(node);
          if (!descriptor || descriptor.kind !== 'binary') {
            throw new Error(`WASM ${node.opType} node ${node.id} has no compiled portable descriptor.`);
          }
          const executionDescriptor = decodeNode ? descriptor.incrementalRow : descriptor;
          if (!executionDescriptor) {
            throw new Error(
              `WASM ${node.opType} node ${node.id} has no correctness-safe incremental row descriptor.`,
            );
          }
          const executionNode = decodeNode || node;
          const a = executionNode.inputs.a;
          const b = executionNode.inputs.b;
          const output = portableOutput(executionNode);
          const pointer = (tensor, label) => decodeNode
            ? wasmTensorPointer(tensor, this.mem, label)
            : this.pointers.get(tensor.name);
          const relu = node.opType === 'Add' ? (node.params.relu ?? 0) : 0;
          if (!Number.isInteger(relu) || relu < 0 || relu > 2) {
            throw new Error(`WASM Add node ${node.id} supports relu values 0 (none), 1 (ReLU), or 2 (ReLU6).`);
          }
          const result = this.api.binary_broadcast_f32(
            pointer(a, `${node.opType} row input a`), pointer(b, `${node.opType} row input b`),
            pointer(output, `${node.opType} row output`), executionDescriptor.aShapePointer,
            executionDescriptor.bShapePointer, executionDescriptor.outputShapePointer,
            executionDescriptor.aRank, executionDescriptor.bRank, executionDescriptor.outputRank,
            executionDescriptor.elements, executionDescriptor.operation,
          );
          if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its canonical descriptor.`);
          if (relu) {
            const outputPointer = pointer(output, `${node.opType} row output`);
            this.api.clip_f32(
              outputPointer, outputPointer, executionDescriptor.elements, 0,
              relu === 2 ? 6 : Number.POSITIVE_INFINITY,
            );
          }
       } else if (kernelRoute === "comparison") {
          const descriptor = this.nodeMetadata.get(node);
          if (!descriptor || descriptor.kind !== 'comparison') {
            throw new Error(`WASM ${node.opType} node ${node.id} has no compiled portable descriptor.`);
          }
          const result = this.api.compare_broadcast_i32(
            this.pointers.get(descriptor.a.name), this.pointers.get(descriptor.b.name),
            this.pointers.get(descriptor.output.name), descriptor.aShapePointer,
            descriptor.bShapePointer, descriptor.outputShapePointer,
            descriptor.aRank, descriptor.bRank, descriptor.kernelRank,
            descriptor.elements, descriptor.operation,
          );
          if (result !== 1) {
            throw new Error(`WASM ${node.opType} node ${node.id} rejected its canonical descriptor.`);
          }
       } else if (kernelRoute === "not") {
          const descriptor = this.nodeMetadata.get(node);
          if (!descriptor || descriptor.kind !== 'not') {
            throw new Error(`WASM Not node ${node.id} has no compiled portable descriptor.`);
          }
          const result = this.api.not_i32(
            this.pointers.get(descriptor.input.name),
            this.pointers.get(descriptor.output.name), descriptor.elements,
          );
          if (result !== 1) throw new Error(`WASM Not node ${node.id} rejected its canonical descriptor.`);
       } else if (kernelRoute === "conv1d") {
          const inShape = node.inputs.input.shape;
          const wShape = node.inputs.weight.shape;
          const [st] = normalizeSpatialPair(node.params.stride, 1);
          const [pd] = normalizeSpatialPair(node.params.padding, 0);
          const groups = node.params.groups || 1;
          const relu = node.params.relu ? 1 : 0;
          const outShape = node.outputs.out.shape;
          const inputBatchBytes = inShape[1] * inShape[2] * 4;
          const outputBatchBytes = outShape[1] * outShape[2] * 4;
          for (let batch = 0; batch < inShape[0]; batch++) {
            // NLC input [batch, l, c]; WIO weight [k, in_per_group, out_c].
            this.api.conv1d_f32(
                inPtr + batch * inputBatchBytes, outPtr + batch * outputBatchBytes, wPtr, bPtr,
                inShape[2], inShape[1], wShape[2], wShape[1], wShape[0], st, pd, groups, relu
            );
          }
       } else if (kernelRoute === "upsample-nearest2d") {
          const input = node.inputs.input || node.inputs.x;
          const output = portableOutput(node);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || input.shape.length !== 4 ||
              output.shape.length !== 4 || output.shape[0] !== input.shape[0] ||
              output.shape[1] !== input.shape[1] * 2 || output.shape[2] !== input.shape[2] * 2 ||
              output.shape[3] !== input.shape[3]) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires F32 NHWC 2x-compatible tensors.`);
          }
          const [batch, height, width, channels] = input.shape;
          const inputBatchBytes = height * width * channels * Float32Array.BYTES_PER_ELEMENT;
          const outputBatchBytes = output.shape[1] * output.shape[2] * channels * Float32Array.BYTES_PER_ELEMENT;
          const inputPtr = this.pointers.get(input.name)!;
          const outputPtr = this.pointers.get(output.name)!;
          for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
            this.api.upsample_nearest2x_f32(inputPtr + batchIndex * inputBatchBytes,
              outputPtr + batchIndex * outputBatchBytes, channels, height, width);
          }
       } else if (kernelRoute === "concat") {
          const quantizedDescriptor = this.nodeMetadata.get(node);
          if (quantizedDescriptor?.kind === 'quantizedConcat') {
            let axisOffset = 0;
            for (const [, input] of quantizedDescriptor.entries) {
              const result = this.api.concat_slice_i8u8(
                this.pointers.get(input.name), this.pointers.get(quantizedDescriptor.output.name),
                quantizedDescriptor.outer, input.shape[quantizedDescriptor.axis],
                quantizedDescriptor.outputAxis, quantizedDescriptor.inner, axisOffset,
                quantizedDescriptor.dtype,
              );
              if (result !== 1) throw new Error(`WASM Concat node ${node.id} rejected its I8/U8 descriptor.`);
              axisOffset += input.shape[quantizedDescriptor.axis];
            }
          } else {
            const descriptor = portableConcatDescriptor(node);
            let axisOffset = 0;
            for (const [, input] of descriptor.entries) {
              const result = descriptor.dtype === 'int32'
                ? this.api.concat_slice_u32(
                  this.pointers.get(input.name), this.pointers.get(descriptor.output.name),
                  descriptor.outer, input.shape[descriptor.axis], descriptor.outputAxis,
                  descriptor.inner, axisOffset,
                )
                : this.api.concat_slice_f32(
                  this.pointers.get(input.name), this.pointers.get(descriptor.output.name),
                  descriptor.outer, input.shape[descriptor.axis], descriptor.outputAxis,
                  descriptor.inner, axisOffset, descriptor.sigmoid ? 1 : 0,
                );
              if (result !== 1) throw new Error(`WASM Concat node ${node.id} rejected its canonical descriptor.`);
              axisOffset += input.shape[descriptor.axis];
            }
          }
       } else if (kernelRoute === "profile-y") {
          const inShape = node.inputs.input.shape;
          const inputBatchBytes = inShape[1] * inShape[2] * inShape[3] * 4;
          const outputBatchBytes = 2 * inShape[3] * inShape[1] * 4;
          for (let batch = 0; batch < inShape[0]; batch++) {
            this.api.profile_y_f32(inPtr + batch * inputBatchBytes, outPtr + batch * outputBatchBytes,
                                   inShape[3], inShape[1], inShape[2]);
          }
       } else if (kernelRoute === "profile-x") {
          const inShape = node.inputs.input.shape;
          const inputBatchBytes = inShape[1] * inShape[2] * inShape[3] * 4;
          const outputBatchBytes = 2 * inShape[3] * inShape[2] * 4;
          for (let batch = 0; batch < inShape[0]; batch++) {
            this.api.profile_x_f32(inPtr + batch * inputBatchBytes, outPtr + batch * outputBatchBytes,
                                   inShape[3], inShape[1], inShape[2]);
          }
       } else if (kernelRoute === "interpolate1d") {
          const input = node.inputs.input || node.inputs.x;
          const output = portableOutput(node);
          if (!portableF32Tensor(input) || !portableF32Tensor(output) || input.shape.length !== 3 ||
              output.shape.length !== 3 || output.shape[0] !== input.shape[0] || output.shape[1] !== input.shape[1]) {
            throw new Error(`WASM ${node.opType} node ${node.id} requires F32 [batch,channels,length] tensors.`);
          }
          const [batch, channels, inputLength] = input.shape;
          const outputLength = output.shape[2];
          const inputBatchBytes = channels * inputLength * Float32Array.BYTES_PER_ELEMENT;
          const outputBatchBytes = channels * outputLength * Float32Array.BYTES_PER_ELEMENT;
          const inputPtr = this.pointers.get(input.name)!;
          const outputPtr = this.pointers.get(output.name)!;
          for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
            this.api.interp1d_f32(inputPtr + batchIndex * inputBatchBytes,
              outputPtr + batchIndex * outputBatchBytes, channels, inputLength, outputLength);
          }
       } else if (kernelRoute === "spatial-softargmax-y") {
           const inShape = node.inputs.input.shape;
           const inputBatchBytes = inShape[1] * inShape[2] * inShape[3] * 4;
           const outputBatchBytes = inShape[3] * inShape[2] * 4;
           for (let batch = 0; batch < inShape[0]; batch++) {
             this.api.spatial_softargmax_y_f32(inPtr + batch * inputBatchBytes,
                                               outPtr + batch * outputBatchBytes,
                                               inShape[3], inShape[1], inShape[2]);
           }
        } else if (kernelRoute === "sigmoid") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.sigmoid_f32(inPtr, outPtr, elements);
        } else if (kernelRoute === "clip") {
           const input = node.inputs.input || node.inputs.x || node.inputs.data;
           const output = portableOutput(node);
           const elements = portableTensorElements(input);
           if (!portable32BitTensor(input) || output?.dtype !== input?.dtype ||
               !portable32BitTensor(output) || portableTensorElements(output) !== elements) {
             throw new Error(`WASM Clip node ${node.id} requires equal-size same-dtype F32/I32 tensors.`);
           }
           if (elements == null) {
             throw new Error(`WASM Clip node ${node.id} has invalid tensor storage.`);
           }
           let minVal = node.params?.min ?? (input.dtype === 'int32' ? -2147483648 : -Infinity);
           let maxVal = node.params?.max ?? (input.dtype === 'int32' ? 2147483647 : Infinity);
           if (node.inputs.min) {
              const p = this.pointers.get(node.inputs.min.name);
              if (node.inputs.min.dtype !== input.dtype || portableTensorElements(node.inputs.min) !== 1) {
                throw new Error(`WASM Clip node ${node.id} min must be a scalar with the input dtype.`);
              }
              minVal = input.dtype === 'int32'
                ? new Int32Array(this.mem.buffer, p, 1)[0]
                : new Float32Array(this.mem.buffer, p, 1)[0];
           }
           if (node.inputs.max) {
              const p = this.pointers.get(node.inputs.max.name);
              if (node.inputs.max.dtype !== input.dtype || portableTensorElements(node.inputs.max) !== 1) {
                throw new Error(`WASM Clip node ${node.id} max must be a scalar with the input dtype.`);
              }
              maxVal = input.dtype === 'int32'
                ? new Int32Array(this.mem.buffer, p, 1)[0]
                : new Float32Array(this.mem.buffer, p, 1)[0];
           }
           if (typeof minVal !== 'number' || typeof maxVal !== 'number' ||
               Number.isNaN(minVal) || Number.isNaN(maxVal) || minVal > maxVal ||
               (input.dtype === 'int32' &&
                (!Number.isInteger(minVal) || !Number.isInteger(maxVal) ||
                 minVal < -2147483648 || maxVal > 2147483647))) {
             throw new Error(`WASM Clip node ${node.id} has invalid bounds.`);
           }
           const result = input.dtype === 'int32'
             ? this.api.clip_i32(
               this.pointers.get(input.name), this.pointers.get(output.name),
               elements, minVal, maxVal,
             )
             : (this.api.clip_f32(
               this.pointers.get(input.name), this.pointers.get(output.name),
               elements, minVal, maxVal,
             ), 1);
           if (result !== 1) throw new Error(`WASM Clip node ${node.id} rejected its typed descriptor.`);
        } else if (kernelRoute === "hard-swish") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.hardswish_f32(inPtr, outPtr, elements);
       } else if (kernelRoute === "leaky-relu") {
           const input = node.inputs.input || node.inputs.x;
           const output = portableOutput(node);
           const elements = portableTensorElements(input);
           if (!portableF32Tensor(input) || !portableF32Tensor(output) || elements == null ||
               portableTensorElements(output) !== elements) {
             throw new Error(`WASM LeakyReLU node ${node.id} requires equal-size F32 tensors.`);
           }
           const alpha = node.params?.alpha ?? 0.01;
           this.api.leakyrelu_f32(this.pointers.get(input.name), this.pointers.get(output.name), elements, alpha);
       } else if (kernelRoute === "prelu") {
           const input = node.inputs.input || node.inputs.x;
           const slope = node.inputs.slope || node.inputs.weight;
           const output = portableOutput(node);
           const elements = portableTensorElements(input);
           const slopeElements = portableTensorElements(slope);
           const channels = input?.shape?.length === 4 ? input.shape[3] : slopeElements;
           if (!portableF32Tensor(input) || !portableF32Tensor(slope) || !portableF32Tensor(output) ||
               elements == null || slopeElements == null || channels == null ||
               !Number.isInteger(channels) || channels <= 0 ||
               portableTensorElements(output) !== elements) {
             throw new Error(`WASM PReLU node ${node.id} requires compatible F32 input, slope, and output tensors.`);
           }
           const result = this.api.prelu_generic_f32(
             this.pointers.get(input.name), this.pointers.get(slope.name), this.pointers.get(output.name),
             elements, slopeElements, channels,
           );
           if (result !== 1) throw new Error(`WASM PReLU node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "hard-sigmoid") {
           const inS = node.inputs.input.shape;
           const elements = inS.reduce((a,b)=>a*b, 1);
           this.api.hardsigmoid_f32(inPtr, outPtr, elements);
        } else if (kernelRoute === "transpose") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'transpose') {
             throw new Error(`WASM Transpose node ${node.id} has no compiled portable descriptor.`);
           }
           const inputPointer = this.pointers.get(descriptor.input.name);
           const outputPointer = this.pointers.get(descriptor.output.name);
           const result = descriptor.byteDtype != null
             ? this.api.transpose_nd_i8u8(
               inputPointer, outputPointer, descriptor.shapePointer,
               descriptor.permPointer, descriptor.rank, descriptor.elements,
               descriptor.byteDtype,
             )
             : (descriptor.dtype === 'int32'
               ? this.api.transpose_nd_u32
               : this.api.transpose_nd_f32)(
               inputPointer, outputPointer, descriptor.shapePointer,
               descriptor.permPointer, descriptor.rank, descriptor.elements,
             );
           if (result !== 1) throw new Error(`WASM Transpose node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "global-average-pool") {
           const inShape = node.inputs.input.shape;
           this.api.global_average_pool_f32(inPtr, outPtr, inShape[0], inShape[1], inShape[2], inShape[3]);
        } else if (kernelRoute === "batchnorm2d") {
           const wPtr = this.pointers.get(node.inputs.weight.name);
           const bPtr = this.pointers.get(node.inputs.bias.name);
           const rmPtr = this.pointers.get(node.inputs.running_mean.name);
           const rvPtr = this.pointers.get(node.inputs.running_var.name);
           const [b, h, w, c] = node.inputs.input.shape;
           const eps = node.params.eps || 1e-5;
           this.api.batch_norm2d_f32(inPtr, wPtr, bPtr, rmPtr, rvPtr, outPtr, b, h, w, c, eps);
        } else if (kernelRoute === "resize") {
           const input = node.inputs.input || node.inputs.x;
           const output = portableOutput(node);
           const quantizedDescriptor = this.nodeMetadata.get(node);
           if (quantizedDescriptor?.kind === 'quantizedNearestResize') {
             const result = this.api.resize_nearest2d_i8u8(
               this.pointers.get(quantizedDescriptor.input.name),
               this.pointers.get(quantizedDescriptor.output.name), quantizedDescriptor.batch,
               quantizedDescriptor.inputHeight, quantizedDescriptor.inputWidth,
               quantizedDescriptor.channels, quantizedDescriptor.outputHeight,
               quantizedDescriptor.outputWidth, quantizedDescriptor.dtype,
             );
             if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its I8/U8 nearest-resize descriptor.`);
           } else {
             if (!portableF32Tensor(input) || !portableF32Tensor(output) || input.shape.length !== 4 ||
                 output.shape.length !== 4 || output.shape[0] !== input.shape[0] ||
                 output.shape[3] !== input.shape[3]) {
               throw new Error(`WASM ${node.opType} node ${node.id} requires F32 NHWC input/output tensors.`);
             }
             const [batch, inputHeight, inputWidth, channels] = input.shape;
             const outputHeight = output.shape[1];
             const outputWidth = output.shape[2];
             const inputPtr = this.pointers.get(input.name);
             const outputPtr = this.pointers.get(output.name);
             if (node.opType === "ResizeNearest2D" || node.params?.mode === "nearest") {
               const result = this.api.resize_nearest2d_f32(inputPtr, outputPtr, batch,
                 inputHeight, inputWidth, channels, outputHeight, outputWidth);
               if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its nearest-resize descriptor.`);
             } else {
               this.api.resize_bilinear_f32(inputPtr, outputPtr, batch,
                 inputHeight, inputWidth, channels, outputHeight, outputWidth);
             }
           }
       } else if (kernelRoute === "cast") {
           const input = node.inputs.input || node.inputs.x || node.inputs.data;
           const output = node.outputs.out || Object.values(node.outputs || {})[0];
           if (!input || !output || input.shape.reduce((a, b) => a * b, 1) !== output.shape.reduce((a, b) => a * b, 1)) {
             throw new Error(`WASM Cast node ${node.id} requires equal input/output element counts.`);
           }
           const inputPtr = this.pointers.get(input.name);
           const outputPtr = this.pointers.get(output.name);
           const result = this.api.cast_typed(inputPtr, wasmDtypeCode(input.dtype), outputPtr,
             wasmDtypeCode(output.dtype), output.shape.reduce((a, b) => a * b, 1));
           if (result !== 1) throw new Error(`WASM Cast node ${node.id} rejected its dtype or shape.`);
        } else if (kernelRoute === "dequantize-linear") {
           const executionNode = decodeNode || node;
           const input = executionNode.inputs.input || executionNode.inputs.x;
           const scale = executionNode.inputs.scale;
           const zeroPoint = executionNode.inputs.zero_point || null;
           const output = portableOutput(executionNode);
           const elements = portableTensorElements(output);
           if (!input || !scale || !output || scale.dtype !== 'float32' || scale.sizeBytes !== 4 ||
               output.dtype !== 'float32' || elements == null ||
               portableTensorElements(input) !== elements ||
               (zeroPoint && zeroPoint.sizeBytes !== Tensor.dtypeBytes(zeroPoint.dtype))) {
             throw new Error(`WASM DequantizeLinear node ${node.id} requires scalar scale/zero-point and matching output.`);
           }
           const result = this.api.dequantize_linear_typed(
             decodeNode ? wasmTensorPointer(input, this.mem, 'DequantizeLinear row input') :
               this.pointers.get(input.name),
             wasmDtypeCode(input.dtype), this.pointers.get(scale.name),
             zeroPoint ? this.pointers.get(zeroPoint.name) : 0, zeroPoint ? wasmDtypeCode(zeroPoint.dtype) : 0,
             decodeNode ? wasmTensorPointer(output, this.mem, 'DequantizeLinear row output') :
               this.pointers.get(output.name),
             elements,
           );
           if (result !== 1) throw new Error(`WASM DequantizeLinear node ${node.id} rejected its dtype or shape.`);
        } else if (kernelRoute === "requantize-linear") {
           const descriptor = this.nodeMetadata.get(node);
           if (descriptor?.kind !== 'requantizeLinear') {
             throw new Error(`WASM RequantizeLinear node ${node.id} has no canonical typed descriptor.`);
           }
           const result = this.api.requantize_linear_i8u8(
             this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.output.name),
             descriptor.elements, descriptor.inputScale, descriptor.inputZeroPoint,
             descriptor.outputScale, descriptor.outputZeroPoint,
             descriptor.inputDtype, descriptor.outputDtype,
           );
           if (result !== 1) throw new Error(`WASM RequantizeLinear node ${node.id} rejected its typed descriptor.`);
        } else if (kernelRoute === "quantize-linear") {
           const executionNode = decodeNode || node;
           const input = executionNode.inputs.input || executionNode.inputs.x || executionNode.inputs.data;
           const scale = executionNode.inputs.scale;
           const zeroPoint = executionNode.inputs.zero_point || null;
           const output = portableOutput(executionNode);
           const inputElements = portableTensorElements(input);
           const outputElements = portableTensorElements(output);
           const scaleElements = portableTensorElements(scale);
           const zeroPointElements = zeroPoint ? portableTensorElements(zeroPoint) : 1;
           if (!input || !scale || !output || input.dtype !== 'float32' ||
               !['int8', 'uint8'].includes(output.dtype) || inputElements == null ||
               outputElements !== inputElements || scale.dtype !== 'float32' || scaleElements !== 1 ||
               output.quantization?.scheme !== 'per_tensor' ||
               (zeroPoint && (zeroPoint.dtype !== output.dtype || zeroPointElements !== 1))) {
             throw new Error(`WASM QuantizeLinear node ${node.id} requires F32 input, scalar F32 scale, optional matching I8/U8 zero point, and matching I8/U8 output with per_tensor quantization metadata.`);
           }
           const scaleValue = wasmScalar(scale, this.mem.buffer, this.pointers.get(scale.name),
             `QuantizeLinear scale at node ${node.id}`);
           const zeroValue = zeroPoint ? wasmScalar(zeroPoint, this.mem.buffer,
             this.pointers.get(zeroPoint.name), `QuantizeLinear zero_point at node ${node.id}`) : 0;
           if (scaleValue == null || zeroValue == null ||
               Math.fround(output.quantization.scale) !== Math.fround(scaleValue) ||
               output.quantization.zero_point !== zeroValue) {
             throw new Error(`WASM QuantizeLinear node ${node.id} scale/zero_point inputs do not match its output quantization metadata.`);
           }
           const result = this.api.quantize_linear_typed(
             decodeNode ? wasmTensorPointer(input, this.mem, 'QuantizeLinear row input') :
               this.pointers.get(input.name),
             this.pointers.get(scale.name),
             zeroPoint ? this.pointers.get(zeroPoint.name) : 0,
             zeroPoint ? wasmDtypeCode(zeroPoint.dtype) : 0,
             decodeNode ? wasmTensorPointer(output, this.mem, 'QuantizeLinear row output') :
               this.pointers.get(output.name),
             wasmDtypeCode(output.dtype), inputElements,
           );
           if (result !== 1) throw new Error(`WASM QuantizeLinear node ${node.id} rejected its scalar quantization descriptor.`);
        } else if (kernelRoute === "slice") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'slice') {
             throw new Error(`WASM Slice node ${node.id} has no compiled portable descriptor.`);
           }
           const kernel = descriptor.dtype === 'int32'
             ? this.api.slice_nd_u32
             : this.api.slice_nd_f32;
           const result = kernel(
             this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.output.name),
             descriptor.inputShapePointer, descriptor.outputShapePointer,
             descriptor.startsPointer, descriptor.stepsPointer,
             descriptor.rank, descriptor.elements,
           );
           if (result !== 1) throw new Error(`WASM Slice node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "expand") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'expand') {
             throw new Error(`WASM ${node.opType} node ${node.id} has no compiled portable descriptor.`);
           }
           const executionDescriptor = decodeNode ? descriptor.incrementalRow : descriptor;
           if (!executionDescriptor) {
             throw new Error(`WASM Expand node ${node.id} has no correctness-safe incremental row descriptor.`);
           }
           const executionNode = decodeNode || node;
           const input = executionNode.inputs.input || executionNode.inputs.x || executionNode.inputs.data;
           const output = portableOutput(executionNode);
           const pointer = (tensor, label) => decodeNode
             ? wasmTensorPointer(tensor, this.mem, label)
             : this.pointers.get(tensor.name);
           const kernel = executionDescriptor.byteStorage
             ? this.api.expand_nd_i8u8
             : executionDescriptor.dtype === 'int32'
               ? this.api.expand_nd_u32
               : this.api.expand_nd_f32;
           const result = kernel(
             pointer(input, 'Expand row input'), pointer(output, 'Expand row output'),
             executionDescriptor.inputShapePointer, executionDescriptor.outputShapePointer,
             executionDescriptor.inputRank, executionDescriptor.outputRank,
             executionDescriptor.elements,
           );
           if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "split") {
           const descriptor = portableSplitDescriptor(node);
           let axisOffset = 0;
           for (const [, output] of descriptor.outputEntries) {
             const result = descriptor.dtype === 'int32'
               ? this.api.split_slice_u32(
                 this.pointers.get(descriptor.input.name), this.pointers.get(output.name),
                 descriptor.outer!, descriptor.inputAxis, descriptor.outputAxis,
                 descriptor.inner!, axisOffset,
               )
               : this.api.split_slice_f32(
                 this.pointers.get(descriptor.input.name), this.pointers.get(output.name),
                 descriptor.outer!, descriptor.inputAxis, descriptor.outputAxis,
                 descriptor.inner!, axisOffset,
               );
             if (result !== 1) throw new Error(`WASM Split node ${node.id} rejected its canonical descriptor.`);
             axisOffset += descriptor.outputAxis;
           }
        } else if (kernelRoute === "gather") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'gather') {
             throw new Error(`WASM Gather node ${node.id} has no compiled portable descriptor.`);
           }
           const result = this.api.gather_i32_f32(
             this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.indices.name),
             this.pointers.get(descriptor.output.name), descriptor.outer, descriptor.axisSize,
             descriptor.inner, descriptor.indicesElements, descriptor.outputElements,
           );
           if (result !== 1) throw new Error(`WASM Gather node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "gather-elements") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'gatherElements') {
             throw new Error(`WASM GatherElements node ${node.id} has no compiled portable descriptor.`);
           }
           const result = this.api.gather_elements_i32_f32(
             this.pointers.get(descriptor.input.name), this.pointers.get(descriptor.indices.name),
             this.pointers.get(descriptor.output.name), descriptor.inputShapePointer,
             descriptor.indicesShapePointer, descriptor.rank, descriptor.axis, descriptor.elements,
           );
           if (result !== 1) throw new Error(`WASM GatherElements node ${node.id} rejected its canonical descriptor.`);
        } else if (kernelRoute === "non-max-suppression") {
           const boxes = node.inputs.boxes;
           const scores = node.inputs.scores;
           const output = node.outputs.out;
           if (!boxes || !scores || !output || boxes.shape.length !== 3 || scores.shape.length !== 3 ||
               boxes.shape[2] !== 4 || scores.shape[0] !== boxes.shape[0] ||
               scores.shape[2] !== boxes.shape[1]) {
             throw new Error(`WASM NonMaxSuppression node ${node.id} requires boxes [B,S,4], scores [B,C,S], and output rows.`);
           }
           const outputElements = output.sizeBytes / Tensor.dtypeBytes(output.dtype);
           if (!Number.isSafeInteger(outputElements) || outputElements < 0 || outputElements % 3 !== 0) {
             throw new Error(`WASM NonMaxSuppression node ${node.id} output must contain complete [batch,class,index] rows.`);
           }
           const scalar = (name, fallback) => node.inputs[name]
             ? wasmScalar(node.inputs[name], this.mem.buffer, this.pointers.get(node.inputs[name].name),
               `NonMaxSuppression ${name} at node ${node.id}`)
             : fallback;
           this.api.non_max_suppression_typed(
             this.pointers.get(boxes.name), this.pointers.get(scores.name), outPtr,
             boxes.shape[0], boxes.shape[1], scores.shape[1], outputElements / 3,
             wasmKernelDtype(boxes, `NonMaxSuppression boxes at node ${node.id}`),
             wasmKernelDtype(scores, `NonMaxSuppression scores at node ${node.id}`),
             wasmKernelDtype(output, `NonMaxSuppression output at node ${node.id}`),
             scalar('max_output_boxes_per_class', 0), scalar('iou_threshold', 0.5),
             scalar('score_threshold', 0),
           );
         } else if (kernelRoute === "where") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'where') {
             throw new Error(`WASM ${node.opType} node ${node.id} has no compiled portable descriptor.`);
           }
           const result = this.api.where_broadcast_32(
             this.pointers.get(descriptor.condition.name), descriptor.conditionType,
             this.pointers.get(descriptor.a.name), this.pointers.get(descriptor.b.name),
             this.pointers.get(descriptor.output.name), descriptor.conditionShapePointer,
             descriptor.aShapePointer, descriptor.bShapePointer, descriptor.outputShapePointer,
             descriptor.conditionRank, descriptor.aRank, descriptor.bRank, descriptor.outputRank,
             descriptor.elements, descriptor.dataType,
           );
           if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its canonical descriptor.`);
         } else if (kernelRoute === "pad") {
           const input = node.inputs.input || node.inputs.data;
           const output = node.outputs.out;
           const inputPtr = this.pointers.get(input.name);
           const outputPtr = this.pointers.get(output?.name);
           if (inputPtr == null || outputPtr == null) {
             throw new Error(`WASM Pad node ${node.id} has an unallocated input or output tensor.`);
           }
           const pads = node.params.pads;
           const pt = pads.length === 8 ? pads[1] : pads[0];
           const pl = pads.length === 8 ? pads[2] : pads[1];
           const pb = pads.length === 8 ? pads[5] : pads[2];
           const pr = pads.length === 8 ? pads[6] : pads[3];
           const val = node.params.value || 0.0;
           const inShape = input.shape;
           const s = inShape.length === 4 ? inShape : [1, inShape[0] || 1, inShape[1] || 1, 1];
           this.api.pad_2d_f32(inputPtr, outputPtr, val, s[0], s[1], s[2], s[3], pt, pb, pl, pr);
         } else if (kernelRoute === "average-pool2d") {
           const descriptor = this.nodeMetadata.get(node);
           if (!descriptor || descriptor.kind !== 'f32Pool2D' ||
               descriptor.opType !== 'AveragePool2D') {
             throw new Error(`WASM AveragePool2D node ${node.id} has no canonical F32 descriptor.`);
           }
           const inputPtr = this.pointers.get(descriptor.input.name);
           const outputPtr = this.pointers.get(descriptor.output.name);
           if (inputPtr == null || outputPtr == null) {
             throw new Error(`WASM AveragePool2D node ${node.id} has an unallocated input or output tensor.`);
           }
           this.api.averagepool2d_f32(
             inputPtr, outputPtr, descriptor.batch, descriptor.inputHeight,
             descriptor.inputWidth, descriptor.channels, descriptor.kernelY,
             descriptor.kernelX, descriptor.strideY, descriptor.strideX,
             descriptor.paddingY, descriptor.paddingX, descriptor.outputHeight,
             descriptor.outputWidth,
           );
        } else if (kernelRoute === "max-pool2d") {
           const descriptor = this.nodeMetadata.get(node);
           if (descriptor?.kind === 'quantizedMaxPool') {
             const result = this.api.maxpool2d_i8u8(
               this.pointers.get(descriptor.input.name),
               this.pointers.get(descriptor.output.name), descriptor.batch,
               descriptor.inputHeight, descriptor.inputWidth,
               descriptor.channels, descriptor.outputHeight,
               descriptor.outputWidth, descriptor.kernelY,
               descriptor.kernelX, descriptor.strideY,
               descriptor.strideX, descriptor.paddingY,
               descriptor.paddingX, descriptor.dtype,
             );
             if (result !== 1) throw new Error(`WASM MaxPool2D node ${node.id} rejected its I8/U8 descriptor.`);
           } else if (descriptor?.kind === 'f32Pool2D' && descriptor.opType === 'MaxPool2D') {
             const inputPointer = this.pointers.get(descriptor.input.name);
             const outputPointer = this.pointers.get(descriptor.output.name);
             if (inputPointer == null || outputPointer == null) {
               throw new Error(`WASM MaxPool2D node ${node.id} has an unallocated input or output tensor.`);
             }
             const inputBatchBytes = descriptor.inputHeight * descriptor.inputWidth *
               descriptor.channels * Float32Array.BYTES_PER_ELEMENT;
             const outputBatchBytes = descriptor.outputHeight * descriptor.outputWidth *
               descriptor.channels * Float32Array.BYTES_PER_ELEMENT;
             for (let batch = 0; batch < descriptor.batch; batch++) {
               this.api.maxpool2d_f32(
                 inputPointer + batch * inputBatchBytes,
                 outputPointer + batch * outputBatchBytes,
                 descriptor.inputHeight, descriptor.inputWidth, descriptor.channels,
                 descriptor.outputHeight, descriptor.outputWidth, descriptor.kernelY,
                 descriptor.kernelX, descriptor.strideY, descriptor.strideX,
                 descriptor.paddingY, descriptor.paddingX,
               );
             }
           } else {
             throw new Error(`WASM MaxPool2D node ${node.id} has no canonical descriptor.`);
           }
        } else if (kernelRoute === "mean-height") {
           const inShape = node.inputs.input.shape;
           const inputBatchBytes = inShape[1] * inShape[2] * inShape[3] * 4;
           const outputBatchBytes = inShape[3] * inShape[2] * 4;
           for (let batch = 0; batch < inShape[0]; batch++) {
             this.api.mean_height_f32(inPtr + batch * inputBatchBytes,
                                      outPtr + batch * outputBatchBytes,
                                      inShape[3], inShape[1], inShape[2]);
           }
        } else if (kernelRoute === "dropout") {
            // compile() aliases the output heap pointer to the input.
        } else if (kernelRoute === "shape-copy") {
            const quantizedDescriptor = this.nodeMetadata.get(node);
            const aliasedInput = node.inputs.input || node.inputs.x || node.inputs.data;
            const aliasedOutput = node.outputs.out || Object.values(node.outputs || {})[0];
            if (aliasedInput && aliasedOutput &&
                this.pointers.get(aliasedInput.name) ===
                this.pointers.get(aliasedOutput.name)) {
              /* compile() aliased this view onto its input, so the bytes are
               * already in place. These reshape-family nodes move 28 MB per
               * inference on this model purely to leave the layout unchanged. */
            } else if (decodeNode) {
              const input = decodeNode.inputs.input || decodeNode.inputs.x ||
                decodeNode.inputs.data;
              const output = portableOutput(decodeNode);
              const elements = portableTensorElements(input);
              if (!input || !output ||
                  !['float32', 'int32', 'int8', 'uint8'].includes(input.dtype) ||
                  output.dtype !== input.dtype || elements == null ||
                  portableTensorElements(output) !== elements) {
                throw new Error(
                  `WASM ${node.opType} node ${node.id} requires an equal-size same-dtype incremental row.`,
                );
              }
              const inputPointer = wasmTensorPointer(
                input, this.mem, `${node.opType} row input`,
              );
              const outputPointer = wasmTensorPointer(
                output, this.mem, `${node.opType} row output`,
              );
              const dtype = wasmDtypeCode(input.dtype);
              const result = input.dtype === 'int8' || input.dtype === 'uint8'
                ? this.api.copy_i8u8(inputPointer, outputPointer, elements, dtype)
                : this.api.cast_typed(
                    inputPointer, dtype, outputPointer, dtype, elements,
                  );
              if (result !== 1) {
                throw new Error(
                  `WASM ${node.opType} node ${node.id} rejected its incremental row copy.`,
                );
              }
            } else if (quantizedDescriptor?.kind === 'quantizedShapeCopy') {
              const result = this.api.copy_i8u8(
                this.pointers.get(quantizedDescriptor.input.name),
                this.pointers.get(quantizedDescriptor.output.name),
                quantizedDescriptor.elements, quantizedDescriptor.dtype,
              );
              if (result !== 1) throw new Error(`WASM ${node.opType} node ${node.id} rejected its I8/U8 descriptor.`);
            } else {
              const input = node.inputs.input || node.inputs.x || node.inputs.data;
              const output = portableOutput(node);
              const elements = portableTensorElements(input);
              if (!input || !output || !['float32', 'int32'].includes(input.dtype) ||
                  output.dtype !== input.dtype || elements == null ||
                  portableTensorElements(output) !== elements) {
                throw new Error(`WASM ${node.opType} node ${node.id} requires equal-size F32 or I32 tensors.`);
              }
              const dtype = wasmDtypeCode(input.dtype);
              const result = this.api.cast_typed(
                this.pointers.get(input.name), dtype,
                this.pointers.get(output.name), dtype, elements,
              );
              if (result !== 1) {
                throw new Error(`WASM ${node.opType} node ${node.id} rejected its typed copy.`);
              }
            }
        } else {
             throw new Error(`WASM operator '${node.opType}' is unsupported.`);
        }
      } catch (e) {
          console.error("[WasmEngine] Execution failed at node:", node, e);
          throw e;
      } finally {
        if (profile) {
          // Setting __VOLVOX_WASM_PROFILE_SHAPES keys by shape as well as
          // opType. Aggregating by opType alone hides the question that
          // usually matters — cost *per node* on identical work — because two
          // packages rarely run the same operator the same number of times.
          const shape = node.inputs?.q?.shape ?? node.outputs?.out?.shape;
          const key = (globalThis as any).__VOLVOX_WASM_PROFILE_SHAPES && shape
            ? `${node.opType}[${shape.join(',')}]` : node.opType;
          const entry = profile.get(key) || { ms: 0, count: 0 };
          entry.ms += performance.now() - profileStart;
          entry.count += 1;
          profile.set(key, entry);
        }
      }
    }

    if (phase) { phase('schedule', mark); mark = performance.now(); }
    if (incremental) this._incrementalCacheValid = true;
    const results = {};
    for (const name of this.graph.outputNames) {
        const tensor = this.graph.tensors.get(name);
        const ptr = this.pointers.get(name);
        // Built-in provider contexts capture host outputs into result-owned
        // storage before another execution can reuse this arena. Publishing the
        // arena view here avoids first making an identical transient copy; the
        // public ExecutionResult remains a stable snapshot.
        results[name] = wasmTensorView(tensor, this.mem.buffer, ptr);
    }
    if (phase) phase('publish-outputs', mark);
    return results;
  }
};
