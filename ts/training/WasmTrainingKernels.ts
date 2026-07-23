import { Tensor } from '../core/Tensor.js';
import type { Graph } from '../core/Graph.js';
import type { RuntimeTypedArray } from '../types.js';
import type { CrossEntropyLossDescriptor } from './TrainingLosses.js';
import type {
  StatefulTrainingGraph,
  TrainingTensorUpdateOptions,
} from './TrainingGraph.js';
import { geluApproximation } from '../ops/gELU.js';
import { dropoutEffectiveSeed, dropoutProbability, dropoutThreshold } from '../ops/dropout.js';
import {
  attentionDropoutEffectiveSeed,
  attentionDropoutProbability,
} from '../ops/attentionDropout.js';
import { DataType, runtimeDTypes } from '../generated/volvoxaiEnums.js';

const ABI_VERSION = 1;
const CAPABILITIES = Object.freeze({
  buffer: 1 << 0,
  crossEntropy: 1 << 1,
  linear: 1 << 2,
  embedding: 1 << 3,
  normalization: 1 << 4,
  activation: 1 << 5,
  elementwise: 1 << 6,
  shape: 1 << 7,
  optimizer: 1 << 8,
});
const REQUIRED_CAPABILITIES = Object.values(CAPABILITIES).reduce((mask, bit) => mask | bit, 0);
const MAX_ALLOCATION_BYTES = 0x7ffffff0;
const MAX_U32 = 0xffffffff;
const RUNTIME_DTYPES = new Set<unknown>(runtimeDTypes);
const TYPED_DTYPE = Object.freeze({
  float32: DataType.F32,
  int32: DataType.I32,
  int8: DataType.I8,
  uint8: DataType.U8,
});

const WEIGHT_SCALE_POLICY = Object.freeze({
  recompute: 0,
  preserve: 1,
});

export type WasmWeightScalePolicy = keyof typeof WEIGHT_SCALE_POLICY;

export interface WasmQuantizedWeight {
  data: Int8Array;
  scales: Float32Array;
  saturationCount: number;
}

type WasmTrainingFunction = (...arguments_: any[]) => any;
export type WasmTrainingApi = Record<string, any> & {
  memory: WebAssembly.Memory;
  alloc_bytes: WasmTrainingFunction;
  reset_heap: WasmTrainingFunction;
};

export interface WasmTrainingSourceEngine {
  wasmModule?: { module?: WebAssembly.Module };
  api: WasmTrainingApi;
  graph?: Graph | null;
  compiledWeightRevision?: number;
}

interface WasmArenaEntry {
  name: string;
  value: ArrayBufferView | null;
  bytes: number;
  ctor: { new(buffer: ArrayBufferLike): RuntimeTypedArray | Uint32Array };
  read: boolean;
}

interface WasmKernelPlan {
  kind: string;
  node: any;
  [name: string]: any;
}

const ACTIVATION_KIND = Object.freeze({
  ReLU: 0,
  GELU: 1,
  // 2 is GELU-tanh in the C ABI. Its existing forward kernel uses a fast
  // approximation that is not paired with the training derivative, so strict
  // training rejects that variant instead of silently mixing semantics.
  SiLU: 3,
  Sigmoid: 4,
  Tanh: 5,
  LeakyReLU: 6,
  HardSigmoid: 7,
  HardSwish: 8,
});

const TRAINING_EXPORTS = Object.freeze([
  'volvoxai_training_abi_version',
  'volvoxai_training_capabilities',
  'volvoxai_training_zero_f32',
  'volvoxai_training_add_f32',
  'volvoxai_training_all_finite_f32',
  'volvoxai_training_sum_squares_f32',
  'volvoxai_training_scale_f32',
  'volvoxai_training_cross_entropy_f32',
  'volvoxai_training_linear_backward_f32',
  'volvoxai_training_linear_backward_packed_f32',
  'volvoxai_training_embedding_backward_f32',
  'volvoxai_training_layernorm_backward_f32',
  'volvoxai_training_rmsnorm_backward_f32',
  'volvoxai_training_groupnorm_f32',
  'volvoxai_training_groupnorm_backward_f32',
  'volvoxai_training_softmax_f32',
  'volvoxai_training_softmax_backward_f32',
  'volvoxai_training_concat_f32',
  'volvoxai_training_concat_backward_f32',
  'volvoxai_training_conv2d_backward_f32',
  'volvoxai_training_conv1d_f32',
  'volvoxai_training_conv1d_backward_f32',
  'volvoxai_training_conv_transpose2d_f32',
  'volvoxai_training_conv_transpose2d_backward_f32',
  'volvoxai_training_pad2d_f32',
  'volvoxai_training_pad2d_backward_f32',
  'volvoxai_training_interp1d_f32',
  'volvoxai_training_interp1d_backward_f32',
  'volvoxai_training_prelu_f32',
  'volvoxai_training_prelu_backward_f32',
  'volvoxai_training_global_average_pool_backward_f32',
  'volvoxai_training_batchnorm2d_backward_f32',
  'volvoxai_training_maxpool2d_f32',
  'volvoxai_training_maxpool2d_backward_f32',
  'volvoxai_training_averagepool2d_backward_f32',
  'volvoxai_training_resize2d_f32',
  'volvoxai_training_resize2d_backward_f32',
  'volvoxai_training_split_backward_f32',
  'volvoxai_training_clip_backward_f32',
  'volvoxai_training_activation_backward_f32',
  'volvoxai_training_add_backward_f32',
  'volvoxai_training_mul_backward_f32',
  'volvoxai_training_dropout_f32',
  'volvoxai_training_dropout_backward_f32',
  'volvoxai_training_sdpa_f32',
  'volvoxai_training_sdpa_backward_f32',
  'volvoxai_training_cross_sdpa_f32',
  'volvoxai_training_cross_sdpa_backward_f32',
  'volvoxai_training_cross_attention_f32',
  'volvoxai_training_cross_attention_backward_f32',
  'volvoxai_training_cast_typed',
  'volvoxai_training_cast_backward_f32',
  'volvoxai_training_dequantize_linear_typed',
  'volvoxai_training_dequantize_linear_backward_f32',
  'volvoxai_training_moe_router_f32',
  'volvoxai_training_moe_router_backward_f32',
  'volvoxai_training_moe_linear_f32',
  'volvoxai_training_moe_linear_backward_f32',
  'volvoxai_training_binary_broadcast_f32',
  'volvoxai_training_binary_broadcast_backward_f32',
  'volvoxai_training_where_f32',
  'volvoxai_training_where_backward_f32',
  'volvoxai_training_slice_f32',
  'volvoxai_training_slice_backward_f32',
  'volvoxai_training_gather_f32',
  'volvoxai_training_gather_backward_f32',
  'volvoxai_training_gather_elements_f32',
  'volvoxai_training_gather_elements_backward_f32',
  'volvoxai_training_mean_height_f32',
  'volvoxai_training_mean_height_backward_f32',
  'volvoxai_training_profile_x_f32',
  'volvoxai_training_profile_x_backward_f32',
  'volvoxai_training_profile_y_f32',
  'volvoxai_training_profile_y_backward_f32',
  'volvoxai_training_spatial_softargmax_y_f32',
  'volvoxai_training_spatial_softargmax_y_backward_f32',
  'volvoxai_training_expand_f32',
  'volvoxai_training_expand_backward_f32',
  'volvoxai_training_reduce_backward_f32',
  'volvoxai_training_copy_backward_f32',
  'volvoxai_training_transpose_backward_f32',
  'volvoxai_training_sgd_update_f32',
  'volvoxai_training_adamw_update_f32',
]);

function elementCount(shape) {
  const count = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(count) || count <= 0 || count > MAX_U32) {
    throw new Error(`WASM training tensor shape [${shape}] is too large.`);
  }
  return count;
}

function firstOutput(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function nodeInput(node) {
  return node.inputs?.input || node.inputs?.x || node.inputs?.data || Object.values(node.inputs || {})[0];
}

function dequantizeInput(node) {
  return node.inputs?.input || node.inputs?.x || node.inputs?.data;
}

function allowsStaticTypedWeight(node, tensor) {
  if (node.opType === 'Cast') return nodeInput(node) === tensor;
  if (node.opType === 'DequantizeLinear') {
    return dequantizeInput(node) === tensor || node.inputs?.zero_point === tensor;
  }
  return false;
}

function sameShape(left, right) {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function pair(value, fallback) {
  return Array.isArray(value) ? [value[0] ?? fallback, value[1] ?? value[0] ?? fallback] : [value ?? fallback, value ?? fallback];
}

function concatInputs(node) {
  const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const seen = new Set();
  const entries: any[] = [];
  for (const key of preferred) if (node.inputs?.[key]) { entries.push(node.inputs[key]); seen.add(key); }
  entries.push(...Object.entries(node.inputs || {}).filter(([key, tensor]) => tensor && !seen.has(key))
    .sort(([left], [right]) => left.localeCompare(right)).map(([, tensor]) => tensor));
  return entries;
}

function typedStorage(dtype, elements) {
  if (dtype === 'float32') return new Float32Array(elements);
  if (dtype === 'int32') return new Int32Array(elements);
  if (dtype === 'int8') return new Int8Array(elements);
  if (dtype === 'uint8') return new Uint8Array(elements);
  throw new Error(`WASM training does not support tensor dtype '${dtype}'.`);
}

function typedStorageConstructor(dtype) {
  if (dtype === 'float32') return Float32Array;
  if (dtype === 'int32') return Int32Array;
  if (dtype === 'int8') return Int8Array;
  if (dtype === 'uint8') return Uint8Array;
  throw new Error(`WASM training does not support tensor dtype '${dtype}'.`);
}

function typedDtypeCode(dtype) {
  if (!Object.prototype.hasOwnProperty.call(TYPED_DTYPE, dtype)) {
    throw new Error(`WASM training does not support tensor dtype '${dtype}'.`);
  }
  return TYPED_DTYPE[dtype];
}

function requireTensor(tensor, dtype, label) {
  if (!tensor || tensor.dtype !== dtype || !Array.isArray(tensor.shape)) {
    throw new Error(`WASM training ${label} must be ${dtype}.`);
  }
  const elements = elementCount(tensor.shape);
  if (tensor.sizeBytes !== elements * Tensor.dtypeBytes(dtype)) {
    throw new Error(`WASM training ${label} storage size does not match its shape.`);
  }
  return tensor;
}

function requireFloatStorage(tensor, label) {
  requireTensor(tensor, 'float32', label);
  if (!(tensor.buffer instanceof Float32Array)) {
    throw new Error(`WASM training ${label} requires initialized Float32 storage.`);
  }
  return tensor.buffer;
}

function requireTypedTensor(tensor, label) {
  typedDtypeCode(tensor?.dtype);
  return requireTensor(tensor, tensor.dtype, label);
}

function requireTypedStorage(tensor, label) {
  requireTypedTensor(tensor, label);
  const value = tensor.buffer;
  if (!ArrayBuffer.isView(value) || value instanceof DataView) {
    throw new Error(`WASM training ${label} requires initialized typed storage.`);
  }
  Tensor.assertCompatibleBuffer(tensor.dtype, value, tensor.sizeBytes,
    `WASM training ${label}`);
  return value;
}

function requireInt32Storage(tensor, label) {
  requireTensor(tensor, 'int32', label);
  if (!(tensor.buffer instanceof Int32Array)) {
    throw new Error(`WASM training ${label} requires initialized Int32 storage.`);
  }
  return tensor.buffer;
}

function validPermutation(perm, rank) {
  return Array.isArray(perm) && perm.length === rank &&
    perm.every((axis) => Number.isInteger(axis) && axis >= 0 && axis < rank) &&
    new Set(perm).size === rank;
}

function positiveSliceDescriptor(node, input, output) {
  const rank = input.shape.length;
  const startsInput = node.params?.starts ?? [];
  const stepsInput = node.params?.steps ?? startsInput.map(() => 1);
  const axesInput = node.params?.axes ?? startsInput.map((_, index) => index);
  if (rank === 0 || rank > 8 || output.shape.length !== rank ||
      !Array.isArray(startsInput) || !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
      startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
    throw new Error(`WASM training Slice node '${node.id}' has invalid rank or parameter arrays.`);
  }
  const starts = new Array(rank).fill(0);
  const steps = new Array(rank).fill(1);
  const seen = new Set();
  for (let index = 0; index < axesInput.length; index++) {
    let axis = axesInput[index];
    if (axis < 0) axis += rank;
    const step = stepsInput[index];
    let start = startsInput[index];
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank || seen.has(axis) ||
        !Number.isInteger(step) || step <= 0 || !Number.isInteger(start)) {
      throw new Error(`WASM training Slice node '${node.id}' requires unique axes and positive integer steps.`);
    }
    if (start < 0) start += input.shape[axis];
    if (start < 0 || start >= input.shape[axis]) {
      throw new Error(`WASM training Slice node '${node.id}' start is outside its input axis.`);
    }
    starts[axis] = start;
    steps[axis] = step;
    seen.add(axis);
  }
  for (let axis = 0; axis < rank; axis++) {
    const outputLength = output.shape[axis];
    if (!Number.isInteger(outputLength) || outputLength <= 0 ||
        starts[axis] + (outputLength - 1) * steps[axis] >= input.shape[axis]) {
      throw new Error(`WASM training Slice node '${node.id}' output shape exceeds its input selection.`);
    }
  }
  return { rank, starts, steps, elements: elementCount(output.shape) };
}

function attentionMaskDescriptor(node, batch, queries, keys) {
  const mask = node.inputs?.mask;
  if (!mask) return { mask: null, maskMode: 0 };
  requireTensor(mask, 'int32', `attention mask at node '${node.id}'`);
  const shape = mask.shape;
  if (shape.length === 1 && shape[0] === keys) return { mask, maskMode: 1 };
  if (shape.length === 2 && shape[1] === keys && (shape[0] === batch || shape[0] === queries)) {
    // Match _cpuAttentionMask: batch semantics intentionally win for [N, K]
    // when N happens to equal both batch and query counts.
    return { mask, maskMode: shape[0] === batch ? 2 : 3 };
  }
  if (shape.length === 3 && shape[0] === batch && shape[1] === queries && shape[2] === keys) {
    return { mask, maskMode: 4 };
  }
  throw new Error(`WASM training attention mask at node '${node.id}' must have shape [K], [B,K], [Q,K], or [B,Q,K].`);
}

function attentionDropoutDescriptor(node, probability, context, nodeIndex) {
  if (!context || probability === 0) {
    return { threshold: 0, seed: 0, counter: 0, scale: 1 };
  }
  return {
    threshold: dropoutThreshold(probability),
    seed: attentionDropoutEffectiveSeed(node, context, nodeIndex),
    counter: context.counter >>> 0,
    scale: 1 / (1 - probability),
  };
}

function activationKind(node) {
  if (node.opType === 'ReLU') return ACTIVATION_KIND.ReLU;
  if (node.opType === 'GELU') {
    if (geluApproximation(node) === 'tanh') {
      throw new Error(`WASM training GELU node '${node.id}' does not support the tanh backward approximation.`);
    }
    return ACTIVATION_KIND.GELU;
  }
  if (node.opType === 'SiLU' || node.opType === 'Swish') return ACTIVATION_KIND.SiLU;
  if (node.opType === 'Sigmoid') return ACTIVATION_KIND.Sigmoid;
  if (node.opType === 'Tanh') return ACTIVATION_KIND.Tanh;
  if (node.opType === 'LeakyReLU') return ACTIVATION_KIND.LeakyReLU;
  if (node.opType === 'HardSigmoid') return ACTIVATION_KIND.HardSigmoid;
  if (node.opType === 'HardSwish') return ACTIVATION_KIND.HardSwish;
  return null;
}

function wasmImports() {
  const env = {
    expf: Math.exp,
    logf: Math.log,
    powf: Math.pow,
    sqrtf: Math.sqrt,
    tanhf: Math.tanh,
    sinf: Math.sin,
    cosf: Math.cos,
  };
  return { env, math: env };
}

function apiResult(value, label) {
  if (Number(value) !== 1) throw new Error(`WASM training ${label} rejected its arguments.`);
}

/**
 * Strict JS orchestration around the freestanding full-profile WASM kernels.
 * The scratch instance never owns canonical graph state; values are copied in
 * and committed only after a complete forward/kernel call succeeds.
 */
export class WasmTrainingKernels {
  declare backend: 'wasm';
  declare sourceEngine: WasmTrainingSourceEngine;
  declare api: WasmTrainingApi;
  declare memory: WebAssembly.Memory;
  declare graph: Graph | null;
  declare topologyRevision: number | null;
  declare plan: WasmKernelPlan[];
  declare planByNode: Map<any, WasmKernelPlan>;

  static async create(wasmEngine: WasmTrainingSourceEngine): Promise<WasmTrainingKernels> {
    const module = wasmEngine?.wasmModule?.module;
    if (!(module instanceof WebAssembly.Module)) {
      throw new Error('WASM training requires an initialized WasmEngine backed by a WebAssembly.Module.');
    }
    this._validateApi(wasmEngine.api, 'initialized');
    const instantiated: any = await WebAssembly.instantiate(module, wasmImports());
    const instance = instantiated?.instance || instantiated;
    this._validateApi(instance?.exports, 'scratch');
    return new WasmTrainingKernels(wasmEngine, instance.exports);
  }

  static _validateApi(api: any, label: string): void {
    if (!api || !(api.memory instanceof WebAssembly.Memory) ||
        typeof api.alloc_bytes !== 'function' || typeof api.reset_heap !== 'function') {
      throw new Error(`The ${label} WASM module is missing memory or allocator exports.`);
    }
    for (const name of TRAINING_EXPORTS) {
      if (typeof api[name] !== 'function') {
        throw new Error(`The ${label} WASM module is missing training export '${name}'.`);
      }
    }
    const version = Number(api.volvoxai_training_abi_version());
    if (version !== ABI_VERSION) {
      throw new Error(`Unsupported WASM training ABI ${version}; expected ${ABI_VERSION}.`);
    }
    const capabilities = Number(api.volvoxai_training_capabilities()) >>> 0;
    if ((capabilities & REQUIRED_CAPABILITIES) !== REQUIRED_CAPABILITIES) {
      const missing = REQUIRED_CAPABILITIES & ~capabilities;
      throw new Error(`WASM training module is missing capability bits 0x${missing.toString(16)}.`);
    }
  }

  constructor(sourceEngine: WasmTrainingSourceEngine, api: WasmTrainingApi) {
    this.backend = 'wasm';
    this.sourceEngine = sourceEngine;
    this.api = api;
    this.memory = api.memory;
    this.graph = null;
    this.topologyRevision = null;
    this.plan = [];
    this.planByNode = new Map();
  }

  _requireExport(name: string): WasmTrainingFunction {
    const fn = this.api[name];
    if (typeof fn !== 'function') throw new Error(`WASM training export '${name}' is unavailable.`);
    return fn;
  }

  /** Fail before training when a sidecar predates F32-master I8 synchronization. */
  requireQuantizedWeightSync(): void {
    this._requireExport('volvoxai_training_quantize_weight_f32_to_i8');
    this._requireExport('volvoxai_training_dequantize_weight_i8_to_f32');
  }

  _ensureMemory(end: number): void {
    if (!Number.isSafeInteger(end) || end < 0 || end > MAX_U32) {
      throw new Error('WASM training allocation exceeds the 32-bit address space.');
    }
    const current = this.memory.buffer.byteLength;
    if (end <= current) return;
    this.memory.grow(Math.ceil((end - current) / 65536));
  }

  _runArena(
    entries: WasmArenaEntry[],
    invoke: (pointers: Record<string, number>) => any,
  ): { value: any; outputs: Record<string, any> } {
    this.api.reset_heap();
    const pointers: Record<string, number> = {};
    for (const entry of entries) {
      if (!Number.isSafeInteger(entry.bytes) || entry.bytes <= 0 || entry.bytes > MAX_ALLOCATION_BYTES) {
        throw new Error(`WASM training allocation '${entry.name}' has invalid size ${entry.bytes}.`);
      }
      const pointer = Number(this.api.alloc_bytes(entry.bytes));
      if (!Number.isSafeInteger(pointer) || pointer < 0) {
        throw new Error(`WASM training allocator returned an invalid pointer for '${entry.name}'.`);
      }
      pointers[entry.name] = pointer;
      this._ensureMemory(pointer + entry.bytes);
    }
    for (const entry of entries) {
      if (!entry.value) continue;
      const source = new Uint8Array(entry.value.buffer, entry.value.byteOffset, entry.value.byteLength);
      new Uint8Array(this.memory.buffer, pointers[entry.name], entry.bytes).set(source);
    }
    const value = invoke(pointers);
    const outputs: Record<string, any> = {};
    for (const entry of entries) {
      if (!entry.read) continue;
      const bytes = this.memory.buffer.slice(pointers[entry.name], pointers[entry.name] + entry.bytes);
      outputs[entry.name] = new entry.ctor(bytes);
    }
    return { value, outputs };
  }

  _floatEntry(name: string, value: unknown, read = false): WasmArenaEntry {
    if (!(value instanceof Float32Array) || value.length <= 0) {
      throw new Error(`WASM training buffer '${name}' must be a non-empty Float32Array.`);
    }
    return { name, value, bytes: value.byteLength, ctor: Float32Array, read };
  }

  _floatScratchEntry(name: string, elements: number): WasmArenaEntry {
    const bytes = elements * Float32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(elements) || elements <= 0 ||
        !Number.isSafeInteger(bytes) || bytes > MAX_ALLOCATION_BYTES) {
      throw new Error(`WASM training scratch '${name}' has invalid element count ${elements}.`);
    }
    return { name, value: null, bytes, ctor: Float32Array, read: false };
  }

  _intEntry(name: string, value: unknown, read = false): WasmArenaEntry {
    if (!(value instanceof Int32Array) || value.length <= 0) {
      throw new Error(`WASM training buffer '${name}' must be a non-empty Int32Array.`);
    }
    return { name, value, bytes: value.byteLength, ctor: Int32Array, read };
  }

  _uintEntry(name: string, value: unknown, read = false): WasmArenaEntry {
    if (!(value instanceof Uint32Array) || value.length <= 0) {
      throw new Error(`WASM training buffer '${name}' must be a non-empty Uint32Array.`);
    }
    return { name, value, bytes: value.byteLength, ctor: Uint32Array, read };
  }

  _int8Entry(name: string, value: unknown, read = false): WasmArenaEntry {
    if (!(value instanceof Int8Array) || value.length <= 0) {
      throw new Error(`WASM training buffer '${name}' must be a non-empty Int8Array.`);
    }
    return { name, value, bytes: value.byteLength, ctor: Int8Array, read };
  }

  _int8ScratchEntry(name: string, elements: number): WasmArenaEntry {
    if (!Number.isSafeInteger(elements) || elements <= 0 || elements > MAX_ALLOCATION_BYTES) {
      throw new Error(`WASM training scratch '${name}' has invalid element count ${elements}.`);
    }
    return { name, value: null, bytes: elements, ctor: Int8Array, read: true };
  }

  /**
   * Quantize one persistent F32 master into canonical symmetric OUT_IN I8
   * storage. This is weight synchronization for an existing quantized graph,
   * not activation calibration or graph PTQ authoring.
   */
  quantizeWeightI8(
    source: Float32Array,
    rows: number,
    columns: number,
    {
      transposeSource = false,
      scalePolicy = 'recompute',
      scales = null,
    }: {
      transposeSource?: boolean;
      scalePolicy?: WasmWeightScalePolicy;
      scales?: Float32Array | null;
    } = {},
  ): WasmQuantizedWeight {
    const quantize = this._requireExport('volvoxai_training_quantize_weight_f32_to_i8');
    if (!(source instanceof Float32Array) || source.length <= 0) {
      throw new Error('WASM weight quantization source must be a non-empty Float32Array.');
    }
    if (!Number.isSafeInteger(rows) || rows <= 0 || rows > MAX_U32 ||
        !Number.isSafeInteger(columns) || columns <= 0 || columns > MAX_U32 ||
        !Number.isSafeInteger(rows * columns) || source.length !== rows * columns) {
      throw new Error('WASM weight quantization dimensions do not match the F32 source.');
    }
    if (!Object.prototype.hasOwnProperty.call(WEIGHT_SCALE_POLICY, scalePolicy)) {
      throw new Error(`Unsupported WASM weight scale policy '${scalePolicy}'.`);
    }
    if (scales != null && (!(scales instanceof Float32Array) || scales.length !== rows)) {
      throw new Error(`WASM weight quantization scales must be Float32Array[${rows}].`);
    }
    if (scalePolicy === 'preserve' && scales == null) {
      throw new Error("WASM weight scale policy 'preserve' requires existing scales.");
    }
    const scaleValues = scales == null ? new Float32Array(rows) : new Float32Array(scales);
    const saturationWords = new Uint32Array(2);
    const { value, outputs } = this._runArena([
      this._floatEntry('source', source),
      this._int8ScratchEntry('output', source.length),
      this._floatEntry('scales', scaleValues, true),
      this._uintEntry('saturation', saturationWords, true),
    ], (pointers) => quantize(
      pointers.source, pointers.output, pointers.scales,
      rows, columns, transposeSource ? 1 : 0,
      WEIGHT_SCALE_POLICY[scalePolicy], pointers.saturation,
    ));
    apiResult(value, `I8 weight quantization (${rows}x${columns})`);
    const saturationCount = outputs.saturation[0] + outputs.saturation[1] * 0x100000000;
    if (!Number.isSafeInteger(saturationCount)) {
      throw new Error('WASM weight quantization saturation count exceeds the safe integer range.');
    }
    return {
      data: outputs.output as Int8Array,
      scales: outputs.scales as Float32Array,
      saturationCount,
    };
  }

  /** Initialize an F32 master from canonical symmetric OUT_IN I8 storage. */
  dequantizeWeightI8(
    source: Int8Array,
    scales: Float32Array,
    rows: number,
    columns: number,
    { transposeDestination = false }: { transposeDestination?: boolean } = {},
  ): Float32Array {
    const dequantize = this._requireExport('volvoxai_training_dequantize_weight_i8_to_f32');
    if (!(source instanceof Int8Array) || source.length <= 0) {
      throw new Error('WASM weight dequantization source must be a non-empty Int8Array.');
    }
    if (!(scales instanceof Float32Array) || scales.length !== rows) {
      throw new Error(`WASM weight dequantization scales must be Float32Array[${rows}].`);
    }
    if (!Number.isSafeInteger(rows) || rows <= 0 || rows > MAX_U32 ||
        !Number.isSafeInteger(columns) || columns <= 0 || columns > MAX_U32 ||
        !Number.isSafeInteger(rows * columns) || source.length !== rows * columns) {
      throw new Error('WASM weight dequantization dimensions do not match the I8 source.');
    }
    const output = new Float32Array(source.length);
    const { value, outputs } = this._runArena([
      this._int8Entry('source', source),
      this._floatEntry('scales', scales),
      this._floatEntry('output', output, true),
    ], (pointers) => dequantize(
      pointers.source, pointers.scales, pointers.output,
      rows, columns, transposeDestination ? 1 : 0,
    ));
    apiResult(value, `I8 weight dequantization (${rows}x${columns})`);
    return outputs.output as Float32Array;
  }

  async preflight(graph: Graph): Promise<void> {
    if (!graph?.tensors || !Array.isArray(graph.nodes)) {
      throw new Error('WASM training requires a valid graph.');
    }
    if (this.graph === graph && this.topologyRevision === graph.topologyRevision) return;
    const plan: WasmKernelPlan[] = [];
    const byNode = new Map<any, WasmKernelPlan>();
    const staticTypedWeights: any[] = [];
    for (const tensor of graph.tensors.values()) {
      if (!RUNTIME_DTYPES.has(tensor.dtype)) {
        throw new Error(`WASM training tensor '${tensor.name}' has unsupported dtype '${tensor.dtype}'.`);
      }
      if (tensor.sizeBytes <= 0 || tensor.sizeBytes > MAX_ALLOCATION_BYTES) {
        throw new Error(`WASM training tensor '${tensor.name}' has an unsupported allocation size.`);
      }
      if (tensor.isWeight) {
        if (tensor.dtype === 'float32') requireFloatStorage(tensor, `weight '${tensor.name}'`);
        else {
          requireTypedStorage(tensor, `static typed weight '${tensor.name}'`);
          staticTypedWeights.push(tensor);
        }
      }
    }
    for (const node of graph.nodes) {
      const item = this._planNode(node);
      plan.push(item);
      byNode.set(node, item);
    }
    for (const tensor of staticTypedWeights) {
      const uses = graph.nodes.filter((node) => Object.values(node.inputs || {}).includes(tensor));
      if (!uses.length || !uses.every((node) => allowsStaticTypedWeight(node, tensor))) {
        throw new Error(`WASM training non-F32 weight '${tensor.name}' is only allowed as a static Cast or DequantizeLinear input/zero point.`);
      }
    }
    this.graph = graph;
    this.topologyRevision = graph.topologyRevision;
    this.plan = plan;
    this.planByNode = byNode;
  }

  _planNode(node: any): WasmKernelPlan {
    const input = nodeInput(node);
    const output = firstOutput(node);
    const op = node.opType;
    if (!output || (op !== 'Split' && op !== 'MoERouter' && Object.keys(node.outputs || {}).length !== 1)) {
      throw new Error(`WASM training node '${node.id}' requires exactly one output.`);
    }

    if (op === 'Cast') {
      const x = requireTypedTensor(input, `Cast input at node '${node.id}'`);
      const out = requireTypedTensor(output, `Cast output at node '${node.id}'`);
      const elements = elementCount(x.shape);
      if (elements !== elementCount(out.shape)) {
        throw new Error(`WASM training Cast node '${node.id}' requires equal input/output element counts.`);
      }
      const inputType = typedDtypeCode(x.dtype);
      const outputType = typedDtypeCode(out.dtype);
      this._requireExport('volvoxai_training_cast_typed');
      this._requireExport('volvoxai_training_cast_backward_f32');
      return {
        kind: 'cast', node, x, output: out, elements, inputType, outputType,
        differentiable: inputType === TYPED_DTYPE.float32 && outputType === TYPED_DTYPE.float32,
      };
    }

    if (op === 'DequantizeLinear') {
      const x = requireTypedTensor(dequantizeInput(node), `DequantizeLinear input at node '${node.id}'`);
      const scale = requireTensor(node.inputs?.scale, 'float32', `DequantizeLinear scale at node '${node.id}'`);
      const zeroPoint = node.inputs?.zero_point
        ? requireTypedTensor(node.inputs.zero_point, `DequantizeLinear zero point at node '${node.id}'`)
        : null;
      const out = requireTensor(output, 'float32', `DequantizeLinear output at node '${node.id}'`);
      const elements = elementCount(x.shape);
      if (elementCount(scale.shape) !== 1 || (zeroPoint && elementCount(zeroPoint.shape) !== 1) ||
          elements !== elementCount(out.shape)) {
        throw new Error(`WASM training DequantizeLinear node '${node.id}' requires scalar F32 scale/zero point and matching input/output elements.`);
      }
      this._requireExport('volvoxai_training_dequantize_linear_typed');
      this._requireExport('volvoxai_training_dequantize_linear_backward_f32');
      return {
        kind: 'dequantizeLinear', node, x, scale, zeroPoint, output: out, elements,
        inputType: typedDtypeCode(x.dtype),
        zeroPointType: zeroPoint ? typedDtypeCode(zeroPoint.dtype) : TYPED_DTYPE.float32,
      };
    }

    if (op === 'Split') {
      const x = requireTensor(input, 'float32', `Split input at node '${node.id}'`);
      const entries = Object.entries(node.outputs || {});
      let axis = node.params?.axis ?? 0;
      if (axis < 0) axis += x.shape.length;
      if (!entries.length || !Number.isInteger(axis) || axis < 0 || axis >= x.shape.length || x.shape[axis] % entries.length) throw new Error(`WASM training Split node '${node.id}' requires equal-sized outputs on a valid axis.`);
      const outputAxis = x.shape[axis] / entries.length;
      const outputs = entries.map(([, tensor]) => requireTensor(tensor, 'float32', `Split output at node '${node.id}'`));
      if (outputs.some((tensor) => tensor.shape.length !== x.shape.length || tensor.shape[axis] !== outputAxis || tensor.shape.some((dimension, index) => index !== axis && dimension !== x.shape[index]))) throw new Error(`WASM training Split node '${node.id}' output shapes are incompatible.`);
      const inner=x.shape.slice(axis+1).reduce((a,b)=>a*b,1), outer=x.shape.slice(0,axis).reduce((a,b)=>a*b,1);
      this._requireExport('volvoxai_training_concat_backward_f32'); this._requireExport('volvoxai_training_split_backward_f32');
      return { kind: 'split', node, x, outputs, axis, inner, outer, inputAxis:x.shape[axis], outputAxis };
    }

    if (op === 'MoERouter') {
      const x = requireTensor(node.inputs?.input || node.inputs?.x, 'float32', `MoERouter input at node '${node.id}'`);
      const weight = requireTensor(node.inputs?.weight || node.inputs?.router_weight, 'float32', `MoERouter weight at node '${node.id}'`);
      const bias = node.inputs?.bias ? requireTensor(node.inputs.bias, 'float32', `MoERouter bias at node '${node.id}'`) : null;
      const indices = requireTensor(node.outputs?.indices || node.outputs?.expert_indices, 'float32', `MoERouter indices at node '${node.id}'`);
      const gates = requireTensor(node.outputs?.weights || node.outputs?.expert_weights, 'float32', `MoERouter weights at node '${node.id}'`);
      const dModel = x.shape.at(-1);
      const rows = elementCount(x.shape) / dModel;
      const experts = node.params?.num_experts ?? weight.shape.at(-1);
      const topK = node.params?.top_k ?? indices.shape.at(-1) ?? 2;
      const temperature = node.params?.temperature ?? 1;
      if (Object.keys(node.outputs || {}).length !== 2 || x.shape.length < 1 || weight.shape.length !== 2 ||
          !Number.isInteger(dModel) || dModel <= 0 || !Number.isInteger(rows) || rows <= 0 ||
          !Number.isInteger(experts) || experts <= 0 || weight.shape[0] !== dModel ||
          weight.shape[1] !== experts || !Number.isInteger(topK) || topK <= 0 || topK > experts ||
          elementCount(indices.shape) !== rows * topK || elementCount(gates.shape) !== rows * topK ||
          (bias && elementCount(bias.shape) !== experts) ||
          typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0) {
        throw new Error(`WASM training MoERouter node '${node.id}' has incompatible F32 routing tensors or parameters.`);
      }
      this._requireExport('volvoxai_training_moe_router_f32');
      this._requireExport('volvoxai_training_moe_router_backward_f32');
      return {
        kind: 'moeRouter', node, x, weight, bias, indices, gates,
        rows, dModel, experts, topK, temperature, normalize: node.params?.normalize !== false,
      };
    }

    if (op === 'MoELinear') {
      const x = requireTensor(node.inputs?.input || node.inputs?.x, 'float32', `MoELinear input at node '${node.id}'`);
      const expertWeight = requireTensor(node.inputs?.expert_weight || node.inputs?.weight, 'float32', `MoELinear expert weight at node '${node.id}'`);
      const expertBias = node.inputs?.expert_bias || node.inputs?.bias;
      const bias = expertBias ? requireTensor(expertBias, 'float32', `MoELinear expert bias at node '${node.id}'`) : null;
      const indices = requireTensor(node.inputs?.route_indices || node.inputs?.indices, 'float32', `MoELinear route indices at node '${node.id}'`);
      const gates = requireTensor(node.inputs?.route_weights || node.inputs?.weights, 'float32', `MoELinear route weights at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `MoELinear output at node '${node.id}'`);
      const dIn = x.shape.at(-1);
      const dOut = out.shape.at(-1);
      const rows = elementCount(x.shape) / dIn;
      const experts = expertWeight.shape[0];
      const topK = indices.shape.at(-1);
      if (x.shape.length < 1 || out.shape.length < 1 || expertWeight.shape.length !== 3 ||
          !Number.isInteger(rows) || rows <= 0 || !Number.isInteger(dIn) || dIn <= 0 ||
          !Number.isInteger(dOut) || dOut <= 0 || !Number.isInteger(experts) || experts <= 0 ||
          !Number.isInteger(topK) || topK <= 0 || topK > experts ||
          expertWeight.shape[1] !== dIn || expertWeight.shape[2] !== dOut ||
          elementCount(indices.shape) !== rows * topK || elementCount(gates.shape) !== rows * topK ||
          elementCount(out.shape) !== rows * dOut ||
          (bias && elementCount(bias.shape) !== experts * dOut)) {
        throw new Error(`WASM training MoELinear node '${node.id}' has incompatible F32 expert or routing tensors.`);
      }
      this._requireExport('volvoxai_training_moe_linear_f32');
      this._requireExport('volvoxai_training_moe_linear_backward_f32');
      return {
        kind: 'moeLinear', node, x, expertWeight, bias, indices, gates, output: out,
        rows, dIn, dOut, experts, topK,
      };
    }

    if (op === 'SDPA') {
      const qkv = requireTensor(node.inputs?.qkv, 'float32', `SDPA QKV input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `SDPA output at node '${node.id}'`);
      const rank = qkv.shape.length;
      const batch = rank === 2 ? 1 : qkv.shape[0];
      const outputBatch = out.shape.length === 2 ? 1 : out.shape[0];
      const seq = qkv.shape[rank - 2];
      const dModel = out.shape.at(-1);
      const heads = node.params?.heads || 8;
      if ((rank !== 2 && rank !== 3) || out.shape.length !== rank || batch !== outputBatch ||
          qkv.shape[rank - 1] !== 3 * dModel || out.shape[rank - 2] !== seq ||
          !Number.isInteger(heads) || heads <= 0 || !Number.isInteger(dModel) ||
          dModel <= 0 || dModel % heads) {
        throw new Error(`WASM training SDPA node '${node.id}' requires compatible [batch?, seq, 3*d_model] F32 tensors and heads.`);
      }
      const rawScale = node.params?.scale !== undefined ? node.params.scale : 1 / Math.sqrt(dModel / heads);
      if (typeof rawScale !== 'number' || !Number.isFinite(rawScale)) {
        throw new Error(`WASM training SDPA node '${node.id}' requires a finite scale.`);
      }
      const { mask, maskMode } = attentionMaskDescriptor(node, batch, seq, seq);
      const probability = attentionDropoutProbability(node);
      this._requireExport('volvoxai_training_sdpa_f32');
      this._requireExport('volvoxai_training_sdpa_backward_f32');
      return {
        kind: 'sdpa', node, qkv, mask, maskMode, output: out,
        batch, seq, dModel, heads, scale: rawScale, causal: node.params?.causal !== false,
        probability,
      };
    }

    if (op === 'CrossSDPA') {
      const q = requireTensor(node.inputs?.q, 'float32', `CrossSDPA Q input at node '${node.id}'`);
      const k = requireTensor(node.inputs?.k, 'float32', `CrossSDPA K input at node '${node.id}'`);
      const v = requireTensor(node.inputs?.v, 'float32', `CrossSDPA V input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `CrossSDPA output at node '${node.id}'`);
      const rank = q.shape.length;
      const batch = rank === 2 ? 1 : q.shape[0];
      const batchOf = (tensor) => tensor.shape.length === 2 ? 1 : tensor.shape[0];
      const seqQ = q.shape[rank - 2];
      const seqKV = k.shape[rank - 2];
      const dModel = out.shape.at(-1);
      const heads = node.params?.heads || 8;
      if ((rank !== 2 && rank !== 3) ||
          [k, v, out].some((tensor) => tensor.shape.length !== rank || batchOf(tensor) !== batch) ||
          q.shape[rank - 1] !== dModel || k.shape[rank - 1] !== dModel ||
          v.shape[rank - 1] !== dModel || v.shape[rank - 2] !== seqKV ||
          out.shape[rank - 2] !== seqQ || !Number.isInteger(heads) || heads <= 0 ||
          !Number.isInteger(dModel) || dModel <= 0 || dModel % heads) {
        throw new Error(`WASM training CrossSDPA node '${node.id}' requires compatible [batch?, sequence, d_model] F32 tensors and heads.`);
      }
      const rawScale = node.params?.scale ?? 1 / Math.sqrt(dModel / heads);
      if (typeof rawScale !== 'number' || !Number.isFinite(rawScale)) {
        throw new Error(`WASM training CrossSDPA node '${node.id}' requires a finite scale.`);
      }
      const { mask, maskMode } = attentionMaskDescriptor(node, batch, seqQ, seqKV);
      const probability = attentionDropoutProbability(node);
      this._requireExport('volvoxai_training_cross_sdpa_f32');
      this._requireExport('volvoxai_training_cross_sdpa_backward_f32');
      return {
        kind: 'crosssdpa', node, q, k, v, mask, maskMode, output: out,
        batch, seqQ, seqKV, dModel, heads, scale: rawScale, causal: node.params?.causal === true,
        probability,
      };
    }

    if (op === 'CrossAttention') {
      const q = requireTensor(node.inputs?.q, 'float32', `CrossAttention Q input at node '${node.id}'`);
      const kv = requireTensor(node.inputs?.kv, 'float32', `CrossAttention KV input at node '${node.id}'`);
      const weight = requireTensor(node.inputs?.weight, 'float32', `CrossAttention projection weight at node '${node.id}'`);
      const scale = node.inputs?.scale ? requireTensor(node.inputs.scale, 'float32', `CrossAttention projection scale at node '${node.id}'`) : null;
      const bias = node.inputs?.bias ? requireTensor(node.inputs.bias, 'float32', `CrossAttention projection bias at node '${node.id}'`) : null;
      const out = requireTensor(output, 'float32', `CrossAttention output at node '${node.id}'`);
      const rank = q.shape.length;
      const batch = rank === 2 ? 1 : q.shape[0];
      const batchOf = (tensor) => tensor.shape.length === 2 ? 1 : tensor.shape[0];
      const seqQ = q.shape[rank - 2];
      const seqKV = kv.shape[rank - 2];
      const dModel = out.shape.at(-1);
      const heads = node.params?.heads || 8;
      const projectionWidth = 3 * dModel;
      if ((rank !== 2 && rank !== 3) || kv.shape.length !== rank || out.shape.length !== rank ||
          batchOf(kv) !== batch || batchOf(out) !== batch || q.shape[rank - 1] !== dModel ||
          kv.shape[rank - 1] !== dModel || out.shape[rank - 2] !== seqQ ||
          weight.shape.length !== 2 || weight.shape[0] !== projectionWidth ||
          weight.shape[1] !== dModel || !Number.isInteger(heads) || heads <= 0 ||
          !Number.isInteger(dModel) || dModel <= 0 || dModel % heads ||
          (scale && elementCount(scale.shape) !== projectionWidth) ||
          (bias && elementCount(bias.shape) !== projectionWidth)) {
        throw new Error(`WASM training CrossAttention node '${node.id}' requires rank-2/3 F32 Q/KV/output and [3*d_model,d_model] projections.`);
      }
      this._requireExport('volvoxai_training_cross_attention_f32');
      this._requireExport('volvoxai_training_cross_attention_backward_f32');
      return {
        kind: 'crossAttention', node, q, kv, weight, scale, bias, output: out,
        batch, seqQ, seqKV, dModel, heads,
      };
    }

    if (op === 'MatMul' || op === 'Linear' || op === 'Gemm') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `${op} weight at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      const k = x.shape.at(-1);
      const n = out.shape.at(-1);
      const doutFirst = node.wLayout === 'dout';
      // A square matrix does not identify its physical layout. Require the
      // graph to declare it instead of silently training a transposed model.
      const ambiguousLayout = node.wLayout == null && weight.shape[0] === n && weight.shape[1] === k;
      if (node.inputs.scale || ambiguousLayout || weight.shape.length !== 2 ||
          (node.wLayout != null && node.wLayout !== 'din' && node.wLayout !== 'dout')) {
        throw new Error(
          `WASM training ${op} node '${node.id}' requires an unquantized weight with explicit din/dout layout.`,
        );
      }
      const m = elementCount(x.shape) / k;
      const expectedWeightShape = doutFirst ? [n, k] : [k, n];
      if (!Number.isInteger(m) || !sameShape(weight.shape, expectedWeightShape) ||
          elementCount(out.shape) !== m * n) {
        throw new Error(`WASM training ${op} node '${node.id}' has incompatible matrix dimensions.`);
      }
      const bias = node.inputs.bias;
      if (bias && (requireTensor(bias, 'float32', `${op} bias at node '${node.id}'`), elementCount(bias.shape) !== n)) {
        throw new Error(`WASM training ${op} node '${node.id}' bias must contain ${n} values.`);
      }
      if (op === 'Gemm' && ((node.params?.alpha ?? 1) !== 1 || (node.params?.beta ?? 1) !== 1 ||
          node.params?.transA === true || node.params?.transB === true)) {
        throw new Error(`WASM training Gemm node '${node.id}' uses unsupported transform parameters.`);
      }
      this._requireExport(doutFirst ? 'linear_f32' : 'matmul_f32');
      this._requireExport('gemm_f32_packed_elements');
      this._requireExport('volvoxai_training_linear_backward_packed_f32');
      const backwardPackedElements = Number(this.api.gemm_f32_packed_elements(n, k));
      if (!Number.isSafeInteger(backwardPackedElements) || backwardPackedElements <= 0 ||
          backwardPackedElements * Float32Array.BYTES_PER_ELEMENT > MAX_ALLOCATION_BYTES) {
        throw new Error(`WASM training ${op} node '${node.id}' has an unrepresentable backward weight pack.`);
      }
      return { kind: 'linear', node, x, weight, bias, output: out, m, k, n,
        layout: doutFirst ? 1 : 0,
        backwardPackedElements };
    }

    if (op === 'Embedding') {
      const ids = requireTensor(input, 'int32', `Embedding ids at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `Embedding weight at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Embedding output at node '${node.id}'`);
      if (weight.shape.length !== 2) throw new Error(`WASM training Embedding node '${node.id}' weight must be rank 2.`);
      const tokens = elementCount(ids.shape);
      const [vocab, dim] = weight.shape;
      if (elementCount(out.shape) !== tokens * dim || out.shape.at(-1) !== dim) {
        throw new Error(`WASM training Embedding node '${node.id}' has incompatible dimensions.`);
      }
      this._requireExport('embedding_f32');
      return { kind: 'embedding', node, ids, weight, output: out, tokens, vocab, dim };
    }

    if (op === 'GroupNorm') {
      const x = requireTensor(input, 'float32', `GroupNorm input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `GroupNorm weight at node '${node.id}'`);
      const bias = requireTensor(node.inputs.bias, 'float32', `GroupNorm bias at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `GroupNorm output at node '${node.id}'`);
      if (x.shape.length !== 4 || !sameShape(x.shape, out.shape)) {
        throw new Error(`WASM training GroupNorm node '${node.id}' requires matching rank-4 NHWC tensors.`);
      }
      const [batch, height, width, channels] = x.shape;
      const groups = node.params?.num_groups;
      const epsilon = node.params?.eps ?? 1e-5;
      if (!Number.isInteger(groups) || groups <= 0 || channels % groups ||
          elementCount(weight.shape) !== channels || elementCount(bias.shape) !== channels ||
          !Number.isFinite(epsilon) || epsilon <= 0) {
        throw new Error(`WASM training GroupNorm node '${node.id}' has incompatible dimensions or parameters.`);
      }
      this._requireExport('volvoxai_training_groupnorm_f32');
      return { kind: 'groupnorm', node, x, weight, bias, output: out, batch, height, width, channels, groups, epsilon };
    }

    if (op === 'Softmax' || op === 'LogSoftmax') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      const width = x.shape.at(-1);
      const rows = elementCount(x.shape) / width;
      const axis = node.params?.axis ?? -1;
      if (!sameShape(x.shape, out.shape) || !Number.isInteger(rows) || (axis !== -1 && axis !== x.shape.length - 1)) {
        throw new Error(`WASM training ${op} node '${node.id}' supports the last axis only.`);
      }
      this._requireExport('volvoxai_training_softmax_f32');
      return { kind: 'softmax', node, x, output: out, rows, width, logOutput: op === 'LogSoftmax' };
    }

    if (op === 'Concat' || op === 'Concat2') {
      const inputs = concatInputs(node);
      const out = requireTensor(output, 'float32', `Concat output at node '${node.id}'`);
      const rank = out.shape.length;
      let axis = node.params?.axis ?? 0;
      if (axis < 0) axis += rank;
      if (!Number.isInteger(axis) || axis < 0 || axis >= rank || node.params?.sigmoid) {
        throw new Error(`WASM training Concat node '${node.id}' has an unsupported axis or sigmoid fusion.`);
      }
      let total = 0;
      for (const value of inputs) {
        requireTensor(value, 'float32', `Concat input at node '${node.id}'`);
        if (value.shape.length !== rank || value.shape.some((dim, index) => index !== axis && dim !== out.shape[index])) {
          throw new Error(`WASM training Concat node '${node.id}' has incompatible input shapes.`);
        }
        total += value.shape[axis];
      }
      if (!inputs.length || total !== out.shape[axis]) throw new Error(`WASM training Concat node '${node.id}' axis lengths are invalid.`);
      const inner = out.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
      const outer = out.shape.slice(0, axis).reduce((a, b) => a * b, 1);
      this._requireExport('volvoxai_training_concat_f32');
      return { kind: 'concat', node, inputs, output: out, axis, inner, outer, outputAxis: out.shape[axis] };
    }

    if (op === 'Conv2D') {
      const x = requireTensor(input, 'float32', `Conv2D input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `Conv2D weight at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Conv2D output at node '${node.id}'`);
      const bias = node.inputs.bias ? requireTensor(node.inputs.bias, 'float32', `Conv2D bias at node '${node.id}'`) : null;
      if (x.shape.length !== 4 || weight.shape.length !== 4 || out.shape.length !== 4) throw new Error(`WASM training Conv2D node '${node.id}' requires NHWC/HWIO rank-4 tensors.`);
      const [batch, inH, inW, inC] = x.shape;
      const [kh, kw, groupIn, weightOut] = weight.shape;
      const [, outH, outW, outputChannels] = out.shape;
      const groups = node.params?.groups ?? 1;
      const depthwise = groups === inC;
      const outC = depthwise ? inC * weightOut : weightOut;
      const [strideY, strideX] = pair(node.params?.stride, 1);
      const [dilationY, dilationX] = pair(node.params?.dilation, 1);
      const [padY, padX] = pair(node.params?.padding, 0);
      const pads = node.params?.pads ?? [padY, padX, padY, padX];
      const relu = node.params?.relu ?? 0;
      if (!Number.isInteger(groups) || groups <= 0 || groups > inC || inC % groups || outC % groups ||
          groupIn !== (depthwise ? inC : inC / groups) || outputChannels !== outC || !Number.isInteger(strideY) || !Number.isInteger(strideX) ||
          !Number.isInteger(dilationY) || !Number.isInteger(dilationX) || strideY <= 0 || strideX <= 0 || dilationY <= 0 || dilationX <= 0 ||
          !Array.isArray(pads) || pads.length !== 4 || pads[0] !== pads[2] || pads[1] !== pads[3] ||
          (bias && elementCount(bias.shape) !== outC)) throw new Error(`WASM training Conv2D node '${node.id}' has an unsupported layout or parameters.`);
      this._requireExport('conv2d_f32');
      return { kind: 'conv2d', node, x, weight, bias, output: out, batch, inH, inW, inC, kh, kw, groupIn, outC, forwardOut: depthwise ? weightOut : outC, outH, outW, strideY, strideX, padTop: pads[0], padLeft: pads[1], groups, relu, dilationY, dilationX };
    }

    if (op === 'Conv1D') {
      const x = requireTensor(input, 'float32', `Conv1D input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `Conv1D weight at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Conv1D output at node '${node.id}'`);
      const bias = node.inputs.bias ? requireTensor(node.inputs.bias, 'float32', `Conv1D bias at node '${node.id}'`) : null;
      if (x.shape.length !== 3 || weight.shape.length !== 3 || out.shape.length !== 3) throw new Error(`WASM training Conv1D node '${node.id}' requires NCL rank-3 tensors.`);
      const [batch, inChannels, inputLength] = x.shape, [outChannels, inputPerGroup, kernel] = weight.shape;
      const [, outputChannels, outputLength] = out.shape;
      const groups = node.params?.groups ?? 1, stride = pair(node.params?.stride, 1)[0], padding = pair(node.params?.padding, 0)[0], relu = node.params?.relu ?? 0;
      if (out.shape[0] !== batch || outputChannels !== outChannels || !Number.isInteger(groups) || groups <= 0 ||
          inChannels % groups || outChannels % groups || inputPerGroup !== inChannels / groups ||
          !Number.isInteger(stride) || !Number.isInteger(padding) || stride <= 0 || padding < 0 ||
          inputLength + 2 * padding < kernel || Math.floor((inputLength + 2 * padding - kernel) / stride) + 1 !== outputLength ||
          (bias && elementCount(bias.shape) !== outChannels)) throw new Error(`WASM training Conv1D node '${node.id}' has incompatible dimensions or parameters.`);
      this._requireExport('volvoxai_training_conv1d_f32');
      return { kind: 'conv1d', node, x, weight, bias, output: out, batch, inChannels, inputLength, outChannels, inputPerGroup, kernel, outputLength, stride, padding, groups, relu };
    }

    if (op === 'ConvTranspose2D') {
      const x=requireTensor(input,'float32',`ConvTranspose2D input at node '${node.id}'`), weight=requireTensor(node.inputs.weight,'float32',`ConvTranspose2D weight at node '${node.id}'`), out=requireTensor(output,'float32',`ConvTranspose2D output at node '${node.id}'`), bias=node.inputs.bias?requireTensor(node.inputs.bias,'float32',`ConvTranspose2D bias at node '${node.id}'`):null;
      if(x.shape.length!==4||weight.shape.length!==4||out.shape.length!==4) throw new Error(`WASM training ConvTranspose2D node '${node.id}' requires rank-4 NHWC tensors.`);
      const [batch,inHeight,inWidth,inChannels]=x.shape,[weightIn,outChannels,kernelY,kernelX]=weight.shape,[,outHeight,outWidth,outputChannels]=out.shape,[strideY,strideX]=pair(node.params?.stride,1),[padY,padX]=pair(node.params?.padding,0);
      if(weightIn!==inChannels||outputChannels!==outChannels||out.shape[0]!==batch||!Number.isInteger(strideY)||!Number.isInteger(strideX)||strideY<=0||strideX<=0||!Number.isInteger(padY)||!Number.isInteger(padX)||padY<0||padX<0||outHeight!==(inHeight-1)*strideY+kernelY-2*padY||outWidth!==(inWidth-1)*strideX+kernelX-2*padX||(bias&&elementCount(bias.shape)!==outChannels)) throw new Error(`WASM training ConvTranspose2D node '${node.id}' has incompatible dimensions or parameters.`);
      this._requireExport('volvoxai_training_conv_transpose2d_f32'); return {kind:'convtranspose2d',node,x,weight,bias,output:out,batch,inHeight,inWidth,inChannels,outHeight,outWidth,outChannels,kernelY,kernelX,strideY,strideX,padY,padX};
    }

    if (op === 'Pad') {
      const x=requireTensor(input,'float32',`Pad input at node '${node.id}'`),out=requireTensor(output,'float32',`Pad output at node '${node.id}'`),pads=node.params?.pads||[];
      if(x.shape.length!==4||out.shape.length!==4||(pads.length!==4&&pads.length!==8)) throw new Error(`WASM training Pad node '${node.id}' requires rank-4 NHWC tensors and 4 or 8 pads.`);
      const [padTop,padLeft,padBottom,padRight]=pads.length===8?[pads[1],pads[2],pads[5],pads[6]]:pads,[batch,height,width,channels]=x.shape,value=node.params?.value??0;
      if(![padTop,padBottom,padLeft,padRight].every(v=>Number.isInteger(v)&&v>=0)||!Number.isFinite(value)||out.shape[0]!==batch||out.shape[1]!==height+padTop+padBottom||out.shape[2]!==width+padLeft+padRight||out.shape[3]!==channels) throw new Error(`WASM training Pad node '${node.id}' has incompatible dimensions or parameters.`);
      this._requireExport('volvoxai_training_pad2d_f32'); return {kind:'pad2d',node,x,output:out,batch,height,width,channels,padTop,padBottom,padLeft,padRight,value};
    }

    if(op==='Interpolate1D'||op==='Interp1D'||op==='InterpLinear1D'){
      const x=requireTensor(input,'float32',`${op} input at node '${node.id}'`),out=requireTensor(output,'float32',`${op} output at node '${node.id}'`);
      if(x.shape.length!==3||out.shape.length!==3||out.shape[0]!==x.shape[0]||out.shape[1]!==x.shape[1]||out.shape[2]!==node.params?.size) throw new Error(`WASM training ${op} node '${node.id}' requires matching rank-3 NCL tensors.`);
      this._requireExport('volvoxai_training_interp1d_f32');return{kind:'interp1d',node,x,output:out,batch:x.shape[0],channels:x.shape[1],inputLength:x.shape[2],outputLength:out.shape[2]};
    }

    if (op === 'PReLU') {
      const x = requireTensor(input, 'float32', `PReLU input at node '${node.id}'`);
      const slope = requireTensor(node.inputs.slope || node.inputs.weight, 'float32', `PReLU slope at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `PReLU output at node '${node.id}'`);
      if (!sameShape(x.shape, out.shape)) throw new Error(`WASM training PReLU node '${node.id}' must preserve shape.`);
      const channels = x.shape.length === 4 ? x.shape[3] : elementCount(slope.shape);
      this._requireExport('volvoxai_training_prelu_f32');
      return { kind: 'prelu', node, x, slope, output: out, elements: elementCount(x.shape), slopeElements: elementCount(slope.shape), channels };
    }

    if (op === 'GlobalAveragePool') {
      const x = requireTensor(input, 'float32', `GlobalAveragePool input at node '${node.id}'`); const out = requireTensor(output, 'float32', `GlobalAveragePool output at node '${node.id}'`);
      if (x.shape.length !== 4 || elementCount(out.shape) !== x.shape[0] * x.shape[3]) throw new Error(`WASM training GlobalAveragePool node '${node.id}' requires NHWC [batch,height,width,channels].`);
      this._requireExport('global_average_pool_f32'); return { kind: 'globalAveragePool', node, x, output: out, batch:x.shape[0], height:x.shape[1], width:x.shape[2], channels:x.shape[3] };
    }

    if (op === 'BatchNorm2D') {
      const x = requireTensor(input, 'float32', `BatchNorm2D input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight || node.inputs.scale, 'float32', `BatchNorm2D weight at node '${node.id}'`);
      const bias = requireTensor(node.inputs.bias || node.inputs.b, 'float32', `BatchNorm2D bias at node '${node.id}'`);
      const runningMean = requireTensor(node.inputs.running_mean || node.inputs.mean, 'float32', `BatchNorm2D running mean at node '${node.id}'`);
      const runningVar = requireTensor(node.inputs.running_var || node.inputs.var, 'float32', `BatchNorm2D running variance at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `BatchNorm2D output at node '${node.id}'`);
      const epsilon = node.params?.eps ?? 1e-5;
      if (x.shape.length !== 4 || !sameShape(x.shape, out.shape) || !Number.isFinite(epsilon) || epsilon <= 0 ||
          [weight, bias, runningMean, runningVar].some((value) => elementCount(value.shape) !== x.shape[3])) {
        throw new Error(`WASM training BatchNorm2D node '${node.id}' requires NHWC tensors and per-channel parameters.`);
      }
      this._requireExport('batch_norm2d_f32');
      this._requireExport('volvoxai_training_batchnorm2d_backward_f32');
      return { kind: 'batchnorm2d', node, x, weight, bias, runningMean, runningVar, output: out,
        batch: x.shape[0], height: x.shape[1], width: x.shape[2], channels: x.shape[3], epsilon };
    }

    if (op === 'MaxPool2D' || op === 'AveragePool' || op === 'AveragePool2D') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      const [kernelY, kernelX] = pair(node.params?.kernel, 0);
      const [strideY, strideX] = pair(node.params?.stride, 1);
      const [padY, padX] = pair(node.params?.padding, 0);
      if (x.shape.length !== 4 || out.shape.length !== 4 || x.shape[0] !== out.shape[0] || x.shape[3] !== out.shape[3] ||
          ![kernelY, kernelX, strideY, strideX, padY, padX].every(Number.isInteger) || kernelY <= 0 || kernelX <= 0 || strideY <= 0 || strideX <= 0 || padY < 0 || padX < 0) {
        throw new Error(`WASM training ${op} node '${node.id}' requires valid NHWC pooling parameters.`);
      }
      if (op === 'MaxPool2D') this._requireExport('volvoxai_training_maxpool2d_f32');
      else this._requireExport('averagepool2d_f32');
      return { kind: op === 'MaxPool2D' ? 'maxpool2d' : 'averagepool2d', node, x, output: out, batch: x.shape[0], height: x.shape[1], width: x.shape[2], channels: x.shape[3], outHeight: out.shape[1], outWidth: out.shape[2], kernelY, kernelX, strideY, strideX, padY, padX };
    }

    if (op === 'Resize' || op === 'ResizeNearest2D' || op === 'UpsampleNearest2D' || op === 'Upsample2x') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (x.shape.length !== 4 || out.shape.length !== 4 || x.shape[0] !== out.shape[0] || x.shape[3] !== out.shape[3]) throw new Error(`WASM training ${op} node '${node.id}' requires matching NHWC batch and channel dimensions.`);
      const mode = node.opType === 'ResizeNearest2D' || node.opType === 'UpsampleNearest2D' || node.opType === 'Upsample2x' || node.params?.mode === 'nearest' ? 1 : 0;
      if ((op === 'UpsampleNearest2D' || op === 'Upsample2x') &&
          (out.shape[1] !== x.shape[1] * 2 || out.shape[2] !== x.shape[2] * 2)) {
        throw new Error(`WASM training ${op} node '${node.id}' requires exactly 2x spatial output.`);
      }
      if (node.params?.mode && node.params.mode !== 'nearest' && node.params.mode !== 'bilinear') throw new Error(`WASM training ${op} node '${node.id}' only supports nearest or bilinear mode.`);
      this._requireExport('volvoxai_training_resize2d_f32');
      return { kind: 'resize2d', node, x, output: out, batch: x.shape[0], inHeight: x.shape[1], inWidth: x.shape[2], channels: x.shape[3], outHeight: out.shape[1], outWidth: out.shape[2], nearest: mode };
    }

    if (op === 'LayerNorm' || op === 'RMSNorm') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const weight = requireTensor(node.inputs.weight, 'float32', `${op} weight at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      const width = x.shape.at(-1);
      const rows = elementCount(x.shape) / width;
      const epsilon = node.params?.eps ?? 1e-6;
      if (elementCount(weight.shape) !== width || !sameShape(x.shape, out.shape) ||
          !Number.isFinite(epsilon) || epsilon <= 0) {
        throw new Error(`WASM training ${op} node '${node.id}' has incompatible dimensions or epsilon.`);
      }
      if (op === 'LayerNorm') {
        const bias = requireTensor(node.inputs.bias, 'float32', `LayerNorm bias at node '${node.id}'`);
        if (elementCount(bias.shape) !== width) {
          throw new Error(`WASM training LayerNorm node '${node.id}' bias must contain ${width} values.`);
        }
        this._requireExport('layernorm_f32');
        return { kind: 'layernorm', node, x, weight, bias, output: out, rows, width, epsilon };
      }
      this._requireExport('rmsnorm_f32');
      return { kind: 'rmsnorm', node, x, weight, output: out, rows, width, epsilon };
    }

    if (op === 'Clip') {
      const x = requireTensor(input, 'float32', `Clip input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Clip output at node '${node.id}'`);
      const minimum = node.params?.min ?? -1e9, maximum = node.params?.max ?? 1e9;
      if (!sameShape(x.shape, out.shape) || !Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum || node.inputs.min || node.inputs.max) throw new Error(`WASM training Clip node '${node.id}' requires finite parameter bounds.`);
      this._requireExport('clip_f32'); this._requireExport('volvoxai_training_clip_backward_f32');
      return { kind: 'clip', node, x, output: out, elements: elementCount(x.shape), minimum, maximum };
    }

    const activation = activationKind(node);
    if (activation != null) {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (!sameShape(x.shape, out.shape)) throw new Error(`WASM training ${op} node '${node.id}' must preserve shape.`);
      const forwardName = activation === ACTIVATION_KIND.ReLU ? 'relu_f32'
        : activation === ACTIVATION_KIND.GELU ? 'gelu_f32'
          : activation === ACTIVATION_KIND.SiLU ? 'silu_f32'
            : activation === ACTIVATION_KIND.Sigmoid ? 'sigmoid_f32'
              : activation === ACTIVATION_KIND.Tanh ? 'tanh_f32'
                : activation === ACTIVATION_KIND.LeakyReLU ? 'leakyrelu_f32'
                  : activation === ACTIVATION_KIND.HardSigmoid ? 'hardsigmoid_f32' : 'hardswish_f32';
      this._requireExport(forwardName);
      return {
        kind: 'activation', node, x, output: out, elements: elementCount(x.shape),
        activation, forwardName, alpha: node.params?.alpha ?? 0.01,
      };
    }

    if (op === 'Dropout') {
      const x = requireTensor(input, 'float32', `Dropout input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Dropout output at node '${node.id}'`);
      if (!sameShape(x.shape, out.shape)) {
        throw new Error(`WASM training Dropout node '${node.id}' must preserve shape.`);
      }
      const probability = dropoutProbability(node);
      this._requireExport('volvoxai_training_dropout_f32');
      return { kind: 'dropout', node, x, output: out, elements: elementCount(x.shape), probability };
    }

    if (['Add', 'Mul', 'Sub', 'Div'].includes(op)) {
      const a = requireTensor(node.inputs.a || input, 'float32', `${op} left input at node '${node.id}'`);
      const b = requireTensor(node.inputs.b, 'float32', `${op} right input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (a.shape.length > 8 || b.shape.length > 8 || out.shape.length > 8) {
        throw new Error(`WASM training ${op} node '${node.id}' supports at most rank 8.`);
      }
      this._requireExport('volvoxai_training_binary_broadcast_f32');
      return { kind: op.toLowerCase(), node, a, b, output: out, elements: elementCount(out.shape) };
    }

    if (op === 'Where' || op === 'Mask') {
      const condition = node.inputs.cond || node.inputs.condition || node.inputs.mask;
      const a = node.inputs.x || node.inputs.a || input;
      const b = node.inputs.y || node.inputs.b;
      if (!condition || !['float32', 'int32'].includes(condition.dtype)) {
        throw new Error(`WASM training ${op} condition at node '${node.id}' must be float32 or int32.`);
      }
      requireTensor(condition, condition.dtype, `${op} condition at node '${node.id}'`);
      const left = requireTensor(a, 'float32', `${op} left input at node '${node.id}'`);
      const right = requireTensor(b, 'float32', `${op} right input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (!sameShape(condition.shape, out.shape) || !sameShape(left.shape, out.shape) ||
          !sameShape(right.shape, out.shape)) {
        throw new Error(`WASM training ${op} node '${node.id}' requires exact-shape condition and operands.`);
      }
      this._requireExport('volvoxai_training_where_f32');
      return {
        kind: 'where', node, condition, a: left, b: right, output: out,
        conditionType: condition.dtype === 'int32' ? DataType.I32 : DataType.F32,
        elements: elementCount(out.shape),
      };
    }

    if (op === 'Slice') {
      const x = requireTensor(input, 'float32', `Slice input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Slice output at node '${node.id}'`);
      const descriptor = positiveSliceDescriptor(node, x, out);
      this._requireExport('volvoxai_training_slice_f32');
      return { kind: 'slice', node, x, output: out, ...descriptor };
    }

    if (op === 'Gather') {
      const x = requireTensor(input, 'float32', `Gather input at node '${node.id}'`);
      const indices = requireTensor(node.inputs.indices, 'int32', `Gather indices at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Gather output at node '${node.id}'`);
      const rank = x.shape.length;
      let axis = node.params?.axis ?? 0;
      if (axis < 0) axis += rank;
      const expectedShape = [...x.shape.slice(0, axis), ...indices.shape, ...x.shape.slice(axis + 1)];
      if (rank === 0 || rank > 8 || !Number.isInteger(axis) || axis < 0 || axis >= rank ||
          !sameShape(out.shape, expectedShape)) {
        throw new Error(`WASM training Gather node '${node.id}' has incompatible axis or output shape.`);
      }
      this._requireExport('volvoxai_training_gather_f32');
      return {
        kind: 'gather', node, x, indices, output: out,
        outer: elementCount(x.shape.slice(0, axis)), axisSize: x.shape[axis],
        inner: elementCount(x.shape.slice(axis + 1)), indicesElements: elementCount(indices.shape),
        elements: elementCount(out.shape),
      };
    }

    if (op === 'GatherElements') {
      const x = requireTensor(input, 'float32', `GatherElements input at node '${node.id}'`);
      const indices = requireTensor(node.inputs.indices, 'int32', `GatherElements indices at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `GatherElements output at node '${node.id}'`);
      const rank = x.shape.length;
      let axis = node.params?.axis ?? 0;
      if (axis < 0) axis += rank;
      if (rank === 0 || rank > 8 || indices.shape.length !== rank || !sameShape(out.shape, indices.shape) ||
          !Number.isInteger(axis) || axis < 0 || axis >= rank ||
          indices.shape.some((dimension, index) => index !== axis && dimension > x.shape[index])) {
        throw new Error(`WASM training GatherElements node '${node.id}' has incompatible axis or output shape.`);
      }
      this._requireExport('volvoxai_training_gather_elements_f32');
      return { kind: 'gatherElements', node, x, indices, output: out, rank, axis, elements: elementCount(out.shape) };
    }

    if (op === 'MeanHeight' || op === 'SpatialSoftargmaxY' || op === 'ProfileX' || op === 'ProfileY') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (x.shape.length !== 4) {
        throw new Error(`WASM training ${op} node '${node.id}' requires rank-4 NHWC input.`);
      }
      const [batch, height, width, channels] = x.shape;
      const expectedShape = op === 'ProfileX' ? [batch, 2 * channels, width]
        : op === 'ProfileY' ? [batch, 2 * channels, height] : [batch, channels, width];
      if (!sameShape(out.shape, expectedShape)) {
        throw new Error(`WASM training ${op} node '${node.id}' has incompatible output shape.`);
      }
      const kind = op === 'MeanHeight' ? 'meanHeight'
        : op === 'SpatialSoftargmaxY' ? 'spatialSoftargmaxY'
          : op === 'ProfileX' ? 'profileX' : 'profileY';
      const exportName = {
        meanHeight: 'volvoxai_training_mean_height_f32',
        spatialSoftargmaxY: 'volvoxai_training_spatial_softargmax_y_f32',
        profileX: 'volvoxai_training_profile_x_f32',
        profileY: 'volvoxai_training_profile_y_f32',
      }[kind];
      this._requireExport(exportName);
      return { kind, node, x, output: out, batch, height, width, channels };
    }

    if (op === 'Expand' || op === 'Broadcast') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (x.shape.length === 0 || x.shape.length > out.shape.length || out.shape.length > 8 ||
          x.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0) ||
          out.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
        throw new Error(`WASM training ${op} node '${node.id}' requires positive rank-1..8 broadcast shapes.`);
      }
      const offset = out.shape.length - x.shape.length;
      if (out.shape.some((dimension, index) => {
        const source = index < offset ? 1 : x.shape[index - offset];
        return source !== 1 && source !== dimension;
      })) throw new Error(`WASM training ${op} node '${node.id}' has incompatible broadcast shapes.`);
      this._requireExport('volvoxai_training_expand_f32');
      return { kind: 'expand', node, x, output: out, elements: elementCount(out.shape) };
    }

    if (op === 'ReduceSum' || op === 'ReduceMean') {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      const width = x.shape.at(-1);
      const rows = elementCount(x.shape) / width;
      if (elementCount(out.shape) !== rows) {
        throw new Error(`WASM training ${op} node '${node.id}' must reduce the last axis.`);
      }
      this._requireExport(op === 'ReduceSum' ? 'reduce_sum_f32' : 'reduce_mean_f32');
      return { kind: 'reduce', node, x, output: out, rows, width, mean: op === 'ReduceMean' };
    }

    if (['Identity', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'].includes(op)) {
      const x = requireTensor(input, 'float32', `${op} input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `${op} output at node '${node.id}'`);
      if (elementCount(x.shape) !== elementCount(out.shape)) {
        throw new Error(`WASM training ${op} node '${node.id}' must preserve element count.`);
      }
      this._requireExport('copy_f32');
      return { kind: 'copy', node, x, output: out, elements: elementCount(x.shape) };
    }

    if (op === 'Transpose') {
      const x = requireTensor(input, 'float32', `Transpose input at node '${node.id}'`);
      const out = requireTensor(output, 'float32', `Transpose output at node '${node.id}'`);
      const rank = x.shape.length;
      const perm = node.params?.perm || [...Array(rank).keys()].reverse();
      if (rank === 0 || rank > 8 || !validPermutation(perm, rank) ||
          !sameShape(out.shape, perm.map((axis) => x.shape[axis]))) {
        throw new Error(`WASM training Transpose node '${node.id}' has an unsupported permutation.`);
      }
      const inverse = new Array(rank);
      for (let axis = 0; axis < rank; axis++) inverse[perm[axis]] = axis;
      return {
        kind: 'transpose', node, x, output: out, rank, perm,
        inverse, elements: elementCount(x.shape),
      };
    }

    throw new Error(`WASM training forward for '${op}' is unsupported at node '${node.id}'.`);
  }

  async forward(
    graph: Graph,
    inputs: Record<string, RuntimeTypedArray> = {},
    training: { dropout?: any } = {},
  ): Promise<void> {
    await this.preflight(graph);
    const unknownInputs = Object.keys(inputs).filter((name) => !graph.tensors.get(name)?.isInput);
    if (unknownInputs.length) throw new Error(`Unknown WASM training input '${unknownInputs[0]}'.`);

    const entries: WasmArenaEntry[] = [];
    const tensorEntry = new Map<any, string>();
    for (const tensor of graph.tensors.values()) {
      const elements = elementCount(tensor.shape);
      let value: any = inputs[tensor.name];
      if (value != null) Tensor.assertCompatibleInput(tensor.dtype, value, tensor.sizeBytes, `Input '${tensor.name}'`);
      else value = tensor.buffer;
      if (tensor.isWeight && value == null) throw new Error(`WASM training weight '${tensor.name}' is uninitialized.`);
      if (tensor.isInput && value == null) throw new Error(`WASM training input '${tensor.name}' has no value.`);
      if (value == null) value = typedStorage(tensor.dtype, elements);
      Tensor.assertCompatibleBuffer(tensor.dtype, value, tensor.sizeBytes, `Tensor '${tensor.name}' buffer`);
      const entry: WasmArenaEntry = {
        name: `tensor:${tensor.name}`,
        value: value as RuntimeTypedArray,
        bytes: tensor.sizeBytes,
        ctor: typedStorageConstructor(tensor.dtype),
        read: !tensor.isWeight,
      };
      entries.push(entry);
      tensorEntry.set(tensor, entry.name);
    }
    for (let index = 0; index < this.plan.length; index++) {
      const item = this.plan[index];
      if (item.kind !== 'transpose') continue;
      entries.push(this._uintEntry(`transpose-shape:${index}`, Uint32Array.from(item.output.shape)));
      entries.push(this._uintEntry(`transpose-inverse:${index}`, Uint32Array.from(item.inverse)));
    }
    for (let index = 0; index < this.plan.length; index++) {
      const item = this.plan[index];
      if (!['add', 'mul', 'sub', 'div', 'where', 'slice', 'gatherElements', 'expand'].includes(item.kind)) continue;
      if (item.kind === 'expand') {
        entries.push(this._uintEntry(`expand-in-shape:${index}`, Uint32Array.from(item.x.shape)));
        entries.push(this._uintEntry(`expand-out-shape:${index}`, Uint32Array.from(item.output.shape)));
        continue;
      }
      if (item.kind === 'slice') {
        entries.push(this._uintEntry(`slice-in-shape:${index}`, Uint32Array.from(item.x.shape)));
        entries.push(this._uintEntry(`slice-out-shape:${index}`, Uint32Array.from(item.output.shape)));
        entries.push(this._uintEntry(`slice-starts:${index}`, Uint32Array.from(item.starts)));
        entries.push(this._uintEntry(`slice-steps:${index}`, Uint32Array.from(item.steps)));
        continue;
      }
      if (item.kind === 'gatherElements') {
        entries.push(this._uintEntry(`gather-elements-input-shape:${index}`, Uint32Array.from(item.x.shape)));
        entries.push(this._uintEntry(`gather-elements-indices-shape:${index}`, Uint32Array.from(item.indices.shape)));
        continue;
      }
      if (item.kind === 'where') continue;
      entries.push(this._uintEntry(`binary-a-shape:${index}`, Uint32Array.from(item.a.shape)));
      entries.push(this._uintEntry(`binary-b-shape:${index}`, Uint32Array.from(item.b.shape)));
      entries.push(this._uintEntry(`binary-out-shape:${index}`, Uint32Array.from(item.output.shape)));
    }

    const { outputs } = this._runArena(entries, (pointers) => {
      for (let index = 0; index < this.plan.length; index++) {
        const item = this.plan[index];
        const pointer = (tensor) => pointers[tensorEntry.get(tensor)!];
        if (item.kind === 'cast') {
          apiResult(this.api.volvoxai_training_cast_typed(
            pointer(item.x), item.inputType, pointer(item.output), item.outputType, item.elements,
          ), `Cast forward at node '${item.node.id}'`);
        } else if (item.kind === 'dequantizeLinear') {
          apiResult(this.api.volvoxai_training_dequantize_linear_typed(
            pointer(item.x), item.inputType, pointer(item.scale),
            item.zeroPoint ? pointer(item.zeroPoint) : 0, item.zeroPointType,
            pointer(item.output), item.elements,
          ), `DequantizeLinear forward at node '${item.node.id}'`);
        } else if (item.kind === 'linear') {
          const forward = item.layout === 1 ? this.api.linear_f32 : this.api.matmul_f32;
          forward(pointer(item.x), pointer(item.weight), item.bias ? pointer(item.bias) : 0,
            pointer(item.output), item.m, item.k, item.n);
        } else if (item.kind === 'moeRouter') {
          apiResult(this.api.volvoxai_training_moe_router_f32(
            pointer(item.x), pointer(item.weight), item.bias ? pointer(item.bias) : 0,
            pointer(item.indices), pointer(item.gates), item.rows, item.dModel,
            item.experts, item.topK, item.temperature, item.normalize ? 1 : 0,
          ), `MoERouter forward at node '${item.node.id}'`);
        } else if (item.kind === 'moeLinear') {
          apiResult(this.api.volvoxai_training_moe_linear_f32(
            pointer(item.x), pointer(item.expertWeight), item.bias ? pointer(item.bias) : 0,
            pointer(item.indices), pointer(item.gates), pointer(item.output), item.rows,
            item.dIn, item.dOut, item.experts, item.topK,
          ), `MoELinear forward at node '${item.node.id}'`);
        } else if (item.kind === 'sdpa') {
          const dropout = attentionDropoutDescriptor(item.node, item.probability, training?.dropout, index);
          apiResult(this.api.volvoxai_training_sdpa_f32(
            pointer(item.qkv), item.mask ? pointer(item.mask) : 0, pointer(item.output),
            item.batch, item.seq, item.dModel, item.heads, item.scale, item.causal ? 1 : 0,
            item.maskMode, dropout.threshold, dropout.seed, dropout.counter, dropout.scale,
          ), `SDPA forward at node '${item.node.id}'`);
        } else if (item.kind === 'crosssdpa') {
          const dropout = attentionDropoutDescriptor(item.node, item.probability, training?.dropout, index);
          apiResult(this.api.volvoxai_training_cross_sdpa_f32(
            pointer(item.q), pointer(item.k), pointer(item.v), item.mask ? pointer(item.mask) : 0,
            pointer(item.output), item.batch, item.seqQ, item.seqKV, item.dModel, item.heads,
            item.scale, item.causal ? 1 : 0, item.maskMode, dropout.threshold, dropout.seed,
            dropout.counter, dropout.scale,
          ), `CrossSDPA forward at node '${item.node.id}'`);
        } else if (item.kind === 'crossAttention') {
          apiResult(this.api.volvoxai_training_cross_attention_f32(
            pointer(item.q), pointer(item.kv), pointer(item.weight), item.scale ? pointer(item.scale) : 0,
            item.bias ? pointer(item.bias) : 0, pointer(item.output), item.batch, item.seqQ,
            item.seqKV, item.dModel, item.heads,
          ), `CrossAttention forward at node '${item.node.id}'`);
        } else if (item.kind === 'embedding') {
          const ids = new Int32Array(this.memory.buffer, pointer(item.ids), item.tokens);
          for (const id of ids) {
            if (id < 0 || id >= item.vocab) {
              throw new Error(`WASM training Embedding node '${item.node.id}' id ${id} is out of range.`);
            }
          }
          this.api.embedding_f32(pointer(item.ids), pointer(item.weight), pointer(item.output), item.tokens, item.dim);
        } else if (item.kind === 'groupnorm') {
          apiResult(this.api.volvoxai_training_groupnorm_f32(
            pointer(item.x), pointer(item.weight), pointer(item.bias), pointer(item.output),
            item.batch, item.height, item.width, item.channels, item.groups, item.epsilon,
          ), `GroupNorm forward at node '${item.node.id}'`);
        } else if (item.kind === 'softmax') {
          apiResult(this.api.volvoxai_training_softmax_f32(
            pointer(item.x), pointer(item.output), item.rows, item.width, item.logOutput ? 1 : 0,
          ), `${item.node.opType} forward at node '${item.node.id}'`);
        } else if (item.kind === 'concat') {
          let axisOffset = 0;
          for (const input of item.inputs) {
            this.api.volvoxai_training_concat_f32(pointer(input), pointer(item.output), item.outer,
              input.shape[item.axis], item.outputAxis, item.inner, axisOffset);
            axisOffset += input.shape[item.axis];
          }
        } else if (item.kind === 'split') {
          for (let outputIndex=0;outputIndex<item.outputs.length;outputIndex++) {
            this.api.volvoxai_training_zero_f32(pointer(item.outputs[outputIndex]), elementCount(item.outputs[outputIndex].shape));
            this.api.volvoxai_training_concat_backward_f32(pointer(item.x), pointer(item.outputs[outputIndex]), item.outer, item.outputAxis, item.inputAxis, item.inner, outputIndex*item.outputAxis);
          }
        } else if (item.kind === 'conv2d') {
          this.api.conv2d_f32(pointer(item.x), pointer(item.output), pointer(item.weight), item.bias ? pointer(item.bias) : 0,
            item.batch, item.inH, item.inW, item.inC, item.kh, item.kw, item.groupIn, item.forwardOut,
            item.outH, item.outW, item.strideY, item.strideX, item.padTop, item.padLeft, item.groups,
            item.relu, item.dilationY, item.dilationX);
        } else if (item.kind === 'conv1d') {
          apiResult(this.api.volvoxai_training_conv1d_f32(pointer(item.x), pointer(item.weight), item.bias ? pointer(item.bias) : 0, pointer(item.output),
            item.batch, item.inChannels, item.inputLength, item.outChannels, item.inputPerGroup, item.kernel,
            item.outputLength, item.stride, item.padding, item.groups, item.relu), `Conv1D forward at node '${item.node.id}'`);
        } else if (item.kind === 'convtranspose2d') {
          apiResult(this.api.volvoxai_training_conv_transpose2d_f32(pointer(item.x),pointer(item.weight),item.bias?pointer(item.bias):0,pointer(item.output),item.batch,item.inHeight,item.inWidth,item.inChannels,item.outHeight,item.outWidth,item.outChannels,item.kernelY,item.kernelX,item.strideY,item.strideX,item.padY,item.padX),`ConvTranspose2D forward at node '${item.node.id}'`);
        } else if (item.kind === 'pad2d') {
          apiResult(this.api.volvoxai_training_pad2d_f32(pointer(item.x),pointer(item.output),item.batch,item.height,item.width,item.channels,item.padTop,item.padBottom,item.padLeft,item.padRight,item.value),`Pad forward at node '${item.node.id}'`);
        } else if(item.kind==='interp1d'){
          apiResult(this.api.volvoxai_training_interp1d_f32(pointer(item.x),pointer(item.output),item.batch,item.channels,item.inputLength,item.outputLength),`Interpolate1D forward at node '${item.node.id}'`);
        } else if (item.kind === 'prelu') {
          apiResult(this.api.volvoxai_training_prelu_f32(pointer(item.x), pointer(item.slope), pointer(item.output), item.elements, item.slopeElements, item.channels), `PReLU forward at node '${item.node.id}'`);
        } else if (item.kind === 'globalAveragePool') {
          this.api.global_average_pool_f32(pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels);
        } else if (item.kind === 'batchnorm2d') {
          this.api.batch_norm2d_f32(pointer(item.x), pointer(item.weight), pointer(item.bias), pointer(item.runningMean), pointer(item.runningVar), pointer(item.output), item.batch, item.height, item.width, item.channels, item.epsilon);
        } else if (item.kind === 'maxpool2d') {
          this.api.volvoxai_training_maxpool2d_f32(pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels, item.outHeight, item.outWidth, item.kernelY, item.kernelX, item.strideY, item.strideX, item.padY, item.padX);
        } else if (item.kind === 'averagepool2d') {
          this.api.averagepool2d_f32(pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels, item.kernelY, item.kernelX, item.strideY, item.strideX, item.padY, item.padX, item.outHeight, item.outWidth);
        } else if (item.kind === 'resize2d') {
          this.api.volvoxai_training_resize2d_f32(pointer(item.x), pointer(item.output), item.batch, item.inHeight, item.inWidth, item.channels, item.outHeight, item.outWidth, item.nearest);
        } else if (item.kind === 'layernorm') {
          this.api.layernorm_f32(pointer(item.x), pointer(item.weight), pointer(item.bias), pointer(item.output),
            item.rows, item.width, item.epsilon);
        } else if (item.kind === 'rmsnorm') {
          this.api.rmsnorm_f32(pointer(item.x), pointer(item.weight), pointer(item.output),
            item.rows, item.width, item.epsilon);
        } else if (item.kind === 'activation') {
          const args = [pointer(item.x), pointer(item.output), item.elements];
          if (item.activation === ACTIVATION_KIND.LeakyReLU) args.push(item.alpha);
          this.api[item.forwardName](...args);
        } else if (item.kind === 'clip') {
          this.api.clip_f32(pointer(item.x), pointer(item.output), item.elements, item.minimum, item.maximum);
        } else if (item.kind === 'dropout') {
          const context = training?.dropout;
          if (!context) throw new Error(`WASM training Dropout node '${item.node.id}' is missing RNG context.`);
          this.api.volvoxai_training_dropout_f32(
            pointer(item.x), pointer(item.output), item.elements,
            dropoutThreshold(item.probability), dropoutEffectiveSeed(item.node, context, index),
            context.counter >>> 0, 1 / (1 - item.probability),
          );
        } else if (['add', 'mul', 'sub', 'div'].includes(item.kind)) {
          apiResult(this.api.volvoxai_training_binary_broadcast_f32(
            pointer(item.a), pointer(item.b), pointer(item.output),
            pointers[`binary-a-shape:${index}`], pointers[`binary-b-shape:${index}`],
            pointers[`binary-out-shape:${index}`], item.a.shape.length, item.b.shape.length,
            item.output.shape.length, item.elements,
            { add: 0, mul: 1, sub: 2, div: 3 }[item.kind],
          ), `${item.node.opType} forward at node '${item.node.id}'`);
        } else if (item.kind === 'where') {
          apiResult(this.api.volvoxai_training_where_f32(
            pointer(item.condition), pointer(item.a), pointer(item.b), pointer(item.output),
            item.conditionType, item.elements,
          ), `${item.node.opType} forward at node '${item.node.id}'`);
        } else if (item.kind === 'slice') {
          apiResult(this.api.volvoxai_training_slice_f32(
            pointer(item.x), pointer(item.output), pointers[`slice-in-shape:${index}`],
            pointers[`slice-out-shape:${index}`], pointers[`slice-starts:${index}`],
            pointers[`slice-steps:${index}`], item.rank, item.elements,
          ), `Slice forward at node '${item.node.id}'`);
        } else if (item.kind === 'gather') {
          apiResult(this.api.volvoxai_training_gather_f32(
            pointer(item.x), pointer(item.indices), pointer(item.output), item.outer, item.axisSize,
            item.inner, item.indicesElements, item.elements,
          ), `Gather forward at node '${item.node.id}'`);
        } else if (item.kind === 'gatherElements') {
          apiResult(this.api.volvoxai_training_gather_elements_f32(
            pointer(item.x), pointer(item.indices), pointer(item.output),
            pointers[`gather-elements-input-shape:${index}`],
            pointers[`gather-elements-indices-shape:${index}`], item.rank, item.axis, item.elements,
          ), `GatherElements forward at node '${item.node.id}'`);
        } else if (item.kind === 'meanHeight') {
          apiResult(this.api.volvoxai_training_mean_height_f32(
            pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels,
          ), `MeanHeight forward at node '${item.node.id}'`);
        } else if (item.kind === 'profileX') {
          apiResult(this.api.volvoxai_training_profile_x_f32(
            pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels,
          ), `ProfileX forward at node '${item.node.id}'`);
        } else if (item.kind === 'profileY') {
          apiResult(this.api.volvoxai_training_profile_y_f32(
            pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels,
          ), `ProfileY forward at node '${item.node.id}'`);
        } else if (item.kind === 'spatialSoftargmaxY') {
          apiResult(this.api.volvoxai_training_spatial_softargmax_y_f32(
            pointer(item.x), pointer(item.output), item.batch, item.height, item.width, item.channels,
          ), `SpatialSoftargmaxY forward at node '${item.node.id}'`);
        } else if (item.kind === 'expand') {
          apiResult(this.api.volvoxai_training_expand_f32(
            pointer(item.x), pointer(item.output), pointers[`expand-in-shape:${index}`],
            pointers[`expand-out-shape:${index}`], item.x.shape.length, item.output.shape.length, item.elements,
          ), `${item.node.opType} forward at node '${item.node.id}'`);
        } else if (item.kind === 'reduce') {
          const fn = item.mean ? this.api.reduce_mean_f32 : this.api.reduce_sum_f32;
          fn(pointer(item.x), pointer(item.output), item.rows, item.width);
        } else if (item.kind === 'copy') {
          this.api.copy_f32(pointer(item.x), pointer(item.output), item.elements);
        } else if (item.kind === 'transpose') {
          this.api.volvoxai_training_zero_f32(pointer(item.output), item.elements);
          apiResult(this.api.volvoxai_training_transpose_backward_f32(
            pointer(item.x), pointer(item.output), pointers[`transpose-shape:${index}`],
            pointers[`transpose-inverse:${index}`], item.rank, item.elements,
          ), `Transpose forward at node '${item.node.id}'`);
        }
      }
    });

    for (const tensor of graph.tensors.values()) {
      if (tensor.isWeight) continue;
      const value = outputs[tensorEntry.get(tensor)!];
      if (!value) continue;
      const currentBuffer: any = tensor.buffer;
      if (currentBuffer?.constructor === value.constructor && currentBuffer.length === value.length) {
        currentBuffer.set(value);
      } else {
        tensor.buffer = value;
      }
      tensor.sizeBytes = value.byteLength;
    }
  }

  crossEntropyGradient(
    logits: Tensor,
    logitsValues: Float32Array,
    descriptor: CrossEntropyLossDescriptor,
  ) {
    requireTensor(logits, 'float32', `logits '${descriptor.logitsTensor}'`);
    if (!(logitsValues instanceof Float32Array) || logitsValues.length === 0) {
      throw new Error(`WASM training logits '${descriptor.logitsTensor}' require Float32 storage.`);
    }
    const classes = logits.shape.at(-1);
    if (!Number.isSafeInteger(classes) || typeof classes !== 'number' || classes <= 0) {
      throw new Error(`WASM training logits '${descriptor.logitsTensor}' have invalid dimensions.`);
    }
    const totalRows = logitsValues.length / classes;
    if (!Number.isSafeInteger(totalRows) || totalRows <= 0) {
      throw new Error(`WASM training logits '${descriptor.logitsTensor}' have invalid dimensions.`);
    }
    const candidateCount = descriptor.targets.length === totalRows ? totalRows : descriptor.targets.length;
    if (candidateCount <= 0 || candidateCount > totalRows) {
      throw new Error(`Target count for loss '${descriptor.name}' does not match logits rows.`);
    }
    const batch = logits.shape.length >= 3 ? logits.shape[0] : 1;
    const rowsPerBatch = Number.isSafeInteger(batch) && batch > 0 && totalRows % batch === 0
      ? totalRows / batch : 0;
    const perBatchTargets = batch > 1 && descriptor.targets.length === batch && rowsPerBatch > 0;
    let row0 = descriptor.targets.length === totalRows ? 0 : totalRows - candidateCount;
    let position = rowsPerBatch - 1;
    if (descriptor.lastToken != null && (descriptor.targets.length === 1 || perBatchTargets)) {
      const limit = perBatchTargets ? rowsPerBatch : totalRows;
      if (descriptor.lastToken >= limit) {
        throw new Error(`lastToken ${descriptor.lastToken} is out of range for ${limit} logits rows.`);
      }
      if (perBatchTargets) position = descriptor.lastToken;
      else row0 = descriptor.lastToken;
    }

    const rows: number[] = [];
    const targets: number[] = [];
    for (let index = 0; index < candidateCount; index++) {
      if (descriptor.lossMask && !descriptor.lossMask[index]) continue;
      const target = descriptor.targets[index];
      if (descriptor.ignoreIndex != null && target === descriptor.ignoreIndex) continue;
      if (!Number.isInteger(target) || target < 0 || target >= classes) {
        throw new Error(`Target id ${target} for loss '${descriptor.name}' is out of range for ${classes} classes.`);
      }
      rows.push(perBatchTargets ? index * rowsPerBatch + position : row0 + index);
      targets.push(target);
    }
    const examples = rows.length;
    const normalizer = descriptor.normalizer ?? examples;
    if (examples > 0 && normalizer <= 0) throw new Error(`Loss '${descriptor.name}' requires a positive normalizer.`);

    const gradient = new Float32Array(logitsValues.length);
    if (examples === 0 || descriptor.weight === 0) {
      const { value, outputs } = this._runArena([
        this._floatEntry('gradient', gradient, true),
      ], (pointers) => this.api.volvoxai_training_zero_f32(pointers.gradient, gradient.length));
      return {
        gradient: outputs.gradient,
        loss: 0,
        unweightedLossSum: 0,
        correct: 0,
        examples,
        normalizer: descriptor.normalizer ?? 0,
      };
    }

    const rowValues = Int32Array.from(rows);
    const targetValues = Int32Array.from(targets);
    const lossSum = new Float32Array(1);
    const correct = new Uint32Array(1);
    const { outputs } = this._runArena([
      this._floatEntry('logits', logitsValues),
      this._intEntry('rows', rowValues),
      this._intEntry('targets', targetValues),
      this._floatEntry('gradient', gradient, true),
      this._floatEntry('loss', lossSum, true),
      this._uintEntry('correct', correct, true),
    ], (pointers) => {
      this.api.volvoxai_training_zero_f32(pointers.gradient, gradient.length);
      apiResult(this.api.volvoxai_training_cross_entropy_f32(
        pointers.logits, pointers.rows, pointers.targets, pointers.gradient,
        examples, classes, descriptor.weight / normalizer, pointers.loss, pointers.correct,
      ), `cross-entropy loss '${descriptor.name}'`);
    });
    return {
      gradient: outputs.gradient,
      loss: descriptor.weight * outputs.loss[0] / normalizer,
      unweightedLossSum: outputs.loss[0],
      correct: outputs.correct[0],
      examples,
      normalizer,
    };
  }

  addGradient(destination: Float32Array, source: Float32Array): Float32Array {
    if (!(destination instanceof Float32Array) || !(source instanceof Float32Array) ||
        destination.length !== source.length || destination.length === 0) {
      throw new Error('WASM training cannot add incompatible gradients.');
    }
    const { outputs } = this._runArena([
      this._floatEntry('destination', destination, true),
      this._floatEntry('source', source),
    ], (pointers) => this.api.volvoxai_training_add_f32(pointers.destination, pointers.source, destination.length));
    destination.set(outputs.destination);
    return destination;
  }

  backwardNode({
    node,
    nodeIndex,
    outputGradient,
    outputGradients,
    gradientFor,
    trainingDropout,
  }: any): boolean {
    const item = this.planByNode.get(node);
    if (!item) return false;
    if (item.kind === 'split') {
      const dx=gradientFor(item.x);
      for (let outputIndex=0;outputIndex<item.outputs.length;outputIndex++) {
        const dy=outputGradients?.get(item.outputs[outputIndex].name);
        if (!dy) continue;
        const { outputs }=this._runArena([this._floatEntry('dy',dy),this._floatEntry('dx',dx,true)],p=>this.api.volvoxai_training_split_backward_f32(p.dy,p.dx,item.outer,item.inputAxis,item.outputAxis,item.inner,outputIndex*item.outputAxis));
        dx.set(outputs.dx);
      }
      return true;
    }

    if (item.kind === 'clip') {
      const dx = gradientFor(item.x);
      const { outputs } = this._runArena([this._floatEntry('x', requireFloatStorage(item.x, `Clip input at node '${node.id}'`)), this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true)], (p) => this.api.volvoxai_training_clip_backward_f32(p.x, p.dy, p.dx, item.elements, item.minimum, item.maximum));
      dx.set(outputs.dx);
      return true;
    }

    if (!(outputGradient instanceof Float32Array)) {
      throw new Error(`WASM training output gradient at node '${node.id}' must be Float32.`);
    }

    if (item.kind === 'cast') {
      if (outputGradient.length !== item.elements) {
        throw new Error(`WASM training Cast output gradient at node '${node.id}' has an incompatible size.`);
      }
      if (!item.differentiable) {
        const { value } = this._runArena([
          this._floatEntry('dy', outputGradient),
        ], (p) => this.api.volvoxai_training_cast_backward_f32(
          p.dy, 0, item.inputType, item.outputType, item.elements,
        ));
        apiResult(value, `Cast backward at node '${node.id}'`);
        return true;
      }
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (p) => this.api.volvoxai_training_cast_backward_f32(
        p.dy, p.dx, item.inputType, item.outputType, item.elements,
      ));
      apiResult(value, `Cast backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'dequantizeLinear') {
      if (outputGradient.length !== item.elements) {
        throw new Error(`WASM training DequantizeLinear output gradient at node '${node.id}' has an incompatible size.`);
      }
      // Input and scalar scale can intentionally alias for a one-element F32
      // dequantization. Keep their destination in one arena allocation so the
      // C kernel accumulates both mathematical roles before committing it.
      const gradientEntries = new Map();
      const gradientEntry = (tensor, name) => {
        let entry = gradientEntries.get(tensor);
        if (!entry) {
          entry = { name, value: gradientFor(tensor) };
          gradientEntries.set(tensor, entry);
        }
        return entry;
      };
      const dx = item.x.dtype === 'float32' ? gradientEntry(item.x, 'dx') : null;
      const dscale = gradientEntry(item.scale, 'dscale');
      const typedEntry = (name, tensor, label) => {
        const value = requireTypedStorage(tensor, label);
        return {
          name, value, bytes: tensor.sizeBytes,
          ctor: typedStorageConstructor(tensor.dtype), read: false,
        };
      };
      const entries = [
        typedEntry('input', item.x, `DequantizeLinear input at node '${node.id}'`),
        this._floatEntry('scale', requireFloatStorage(item.scale,
          `DequantizeLinear scale at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        ...[...gradientEntries.values()].map(({ name, value }) => this._floatEntry(name, value, true)),
      ];
      if (item.zeroPoint) entries.push(typedEntry('zeroPoint', item.zeroPoint,
        `DequantizeLinear zero point at node '${node.id}'`));
      const { value, outputs } = this._runArena(entries, (p) =>
        this.api.volvoxai_training_dequantize_linear_backward_f32(
          p.input, item.inputType, p.scale, item.zeroPoint ? p.zeroPoint : 0,
          item.zeroPointType, p.dy, dx ? p[dx.name] : 0, p[dscale.name], item.elements,
        ));
      apiResult(value, `DequantizeLinear backward at node '${node.id}'`);
      for (const { name, value: gradient } of gradientEntries.values()) gradient.set(outputs[name]);
      return true;
    }

    if (item.kind === 'moeRouter') {
      if (outputGradient.length !== item.rows * item.topK) {
        throw new Error(`WASM training MoERouter gate gradient at node '${node.id}' has an incompatible size.`);
      }
      const dx = gradientFor(item.x);
      const dw = gradientFor(item.weight);
      const db = item.bias ? gradientFor(item.bias) : null;
      const entries = [
        this._floatEntry('input', requireFloatStorage(item.x, `MoERouter input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `MoERouter weight at node '${node.id}'`)),
        this._floatEntry('indices', requireFloatStorage(item.indices, `MoERouter indices at node '${node.id}'`)),
        this._floatEntry('gates', requireFloatStorage(item.gates, `MoERouter weights at node '${node.id}'`)),
        this._floatEntry('gateGradient', outputGradient),
        this._floatEntry('dx', dx, true),
        this._floatEntry('dw', dw, true),
      ];
      if (item.bias) entries.push(
        this._floatEntry('bias', requireFloatStorage(item.bias, `MoERouter bias at node '${node.id}'`)),
        this._floatEntry('db', db, true),
      );
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_moe_router_backward_f32(
        p.input, p.weight, item.bias ? p.bias : 0, p.indices, p.gates, p.gateGradient,
        p.dx, p.dw, db ? p.db : 0, item.rows, item.dModel, item.experts, item.topK,
        item.temperature, item.normalize ? 1 : 0,
      ));
      apiResult(value, `MoERouter backward at node '${node.id}'`);
      dx.set(outputs.dx);
      dw.set(outputs.dw);
      if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'moeLinear') {
      if (outputGradient.length !== item.rows * item.dOut) {
        throw new Error(`WASM training MoELinear output gradient at node '${node.id}' has an incompatible size.`);
      }
      const dx = gradientFor(item.x);
      const dw = gradientFor(item.expertWeight);
      const db = item.bias ? gradientFor(item.bias) : null;
      const dr = gradientFor(item.gates);
      const entries = [
        this._floatEntry('input', requireFloatStorage(item.x, `MoELinear input at node '${node.id}'`)),
        this._floatEntry('expertWeight', requireFloatStorage(item.expertWeight, `MoELinear expert weight at node '${node.id}'`)),
        this._floatEntry('indices', requireFloatStorage(item.indices, `MoELinear route indices at node '${node.id}'`)),
        this._floatEntry('gates', requireFloatStorage(item.gates, `MoELinear route weights at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._floatEntry('dw', dw, true),
        this._floatEntry('dr', dr, true),
      ];
      if (item.bias) entries.push(
        this._floatEntry('bias', requireFloatStorage(item.bias, `MoELinear expert bias at node '${node.id}'`)),
        this._floatEntry('db', db, true),
      );
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_moe_linear_backward_f32(
        p.input, p.expertWeight, item.bias ? p.bias : 0, p.indices, p.gates, p.dy,
        p.dx, p.dw, db ? p.db : 0, p.dr, item.rows, item.dIn, item.dOut,
        item.experts, item.topK,
      ));
      apiResult(value, `MoELinear backward at node '${node.id}'`);
      dx.set(outputs.dx);
      dw.set(outputs.dw);
      dr.set(outputs.dr);
      if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'sdpa') {
      const dqkv = gradientFor(item.qkv);
      const entries = [
        this._floatEntry('qkv', requireFloatStorage(item.qkv, `SDPA QKV input at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dqkv', dqkv, true),
      ];
      if (item.mask) entries.push(this._intEntry(
        'mask', requireInt32Storage(item.mask, `SDPA mask at node '${node.id}'`),
      ));
      const dropout = attentionDropoutDescriptor(item.node, item.probability, trainingDropout, nodeIndex);
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_sdpa_backward_f32(
        p.qkv, item.mask ? p.mask : 0, p.dy, p.dqkv, item.batch, item.seq, item.dModel,
        item.heads, item.scale, item.causal ? 1 : 0, item.maskMode, dropout.threshold,
        dropout.seed, dropout.counter, dropout.scale,
      ));
      apiResult(value, `SDPA backward at node '${node.id}'`);
      dqkv.set(outputs.dqkv);
      return true;
    }

    if (item.kind === 'crosssdpa') {
      // Q/K/V may intentionally share one tensor. Keep those destinations as
      // one arena allocation so the C kernel accumulates every role into the
      // same gradient, rather than committing three competing snapshots.
      const gradientEntries = new Map();
      const gradientEntry = (tensor, name) => {
        let entry = gradientEntries.get(tensor);
        if (!entry) {
          entry = { name, value: gradientFor(tensor) };
          gradientEntries.set(tensor, entry);
        }
        return entry;
      };
      const dq = gradientEntry(item.q, 'dq');
      const dk = gradientEntry(item.k, 'dk');
      const dv = gradientEntry(item.v, 'dv');
      const entries = [
        this._floatEntry('q', requireFloatStorage(item.q, `CrossSDPA Q input at node '${node.id}'`)),
        this._floatEntry('k', requireFloatStorage(item.k, `CrossSDPA K input at node '${node.id}'`)),
        this._floatEntry('v', requireFloatStorage(item.v, `CrossSDPA V input at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        ...[...gradientEntries.values()].map(({ name, value }) => this._floatEntry(name, value, true)),
      ];
      if (item.mask) entries.push(this._intEntry(
        'mask', requireInt32Storage(item.mask, `CrossSDPA mask at node '${node.id}'`),
      ));
      const dropout = attentionDropoutDescriptor(item.node, item.probability, trainingDropout, nodeIndex);
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_cross_sdpa_backward_f32(
        p.q, p.k, p.v, item.mask ? p.mask : 0, p.dy, p[dq.name], p[dk.name], p[dv.name],
        item.batch, item.seqQ, item.seqKV, item.dModel, item.heads, item.scale,
        item.causal ? 1 : 0, item.maskMode, dropout.threshold, dropout.seed,
        dropout.counter, dropout.scale,
      ));
      apiResult(value, `CrossSDPA backward at node '${node.id}'`);
      for (const { name, value: gradient } of gradientEntries.values()) gradient.set(outputs[name]);
      return true;
    }

    if (item.kind === 'crossAttention') {
      if (outputGradient.length !== item.batch * item.seqQ * item.dModel) {
        throw new Error(`WASM training CrossAttention output gradient at node '${node.id}' has an incompatible size.`);
      }
      // Q and KV may deliberately be the same trainable tensor. Group all
      // gradient destinations by tensor so the C kernel can accumulate every
      // role into one arena allocation before it is committed.
      const gradientEntries = new Map();
      const gradientEntry = (tensor, name) => {
        let entry = gradientEntries.get(tensor);
        if (!entry) {
          entry = { name, value: gradientFor(tensor) };
          gradientEntries.set(tensor, entry);
        }
        return entry;
      };
      const dq = gradientEntry(item.q, 'dq');
      const dkv = gradientEntry(item.kv, 'dkv');
      const dw = gradientEntry(item.weight, 'dw');
      const ds = item.scale ? gradientEntry(item.scale, 'ds') : null;
      const db = item.bias ? gradientEntry(item.bias, 'db') : null;
      const entries = [
        this._floatEntry('q', requireFloatStorage(item.q, `CrossAttention Q input at node '${node.id}'`)),
        this._floatEntry('kv', requireFloatStorage(item.kv, `CrossAttention KV input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `CrossAttention projection weight at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        ...[...gradientEntries.values()].map(({ name, value }) => this._floatEntry(name, value, true)),
      ];
      if (item.scale) entries.push(this._floatEntry(
        'scale', requireFloatStorage(item.scale, `CrossAttention projection scale at node '${node.id}'`),
      ));
      if (item.bias) entries.push(this._floatEntry(
        'bias', requireFloatStorage(item.bias, `CrossAttention projection bias at node '${node.id}'`),
      ));
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_cross_attention_backward_f32(
        p.q, p.kv, p.weight, item.scale ? p.scale : 0, item.bias ? p.bias : 0, p.dy,
        p[dq.name], p[dkv.name], p[dw.name], ds ? p[ds.name] : 0, db ? p[db.name] : 0,
        item.batch, item.seqQ, item.seqKV, item.dModel, item.heads,
      ));
      apiResult(value, `CrossAttention backward at node '${node.id}'`);
      for (const { name, value: gradient } of gradientEntries.values()) gradient.set(outputs[name]);
      return true;
    }

    if (item.kind === 'linear') {
      const dx = gradientFor(item.x);
      const dw = gradientFor(item.weight);
      const db = item.bias ? gradientFor(item.bias) : null;
      const entries = [
        this._floatEntry('x', requireFloatStorage(item.x, `linear input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `linear weight at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._floatEntry('dw', dw, true),
        this._floatScratchEntry('packedWeight', item.backwardPackedElements),
      ];
      if (db) entries.push(this._floatEntry('db', db, true));
      const { outputs } = this._runArena(entries, (pointers) => {
        apiResult(this.api.volvoxai_training_linear_backward_packed_f32(
          pointers.x, pointers.weight, pointers.dy, pointers.dx, pointers.dw,
          db ? pointers.db : 0, pointers.packedWeight,
          item.backwardPackedElements, item.m, item.k, item.n, item.layout,
        ), `packed linear backward at node '${node.id}'`);
      });
      dx.set(outputs.dx); dw.set(outputs.dw); if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'embedding') {
      const ids = item.ids.buffer;
      if (!(ids instanceof Int32Array)) throw new Error(`Embedding ids at node '${node.id}' are not initialized.`);
      const dw = gradientFor(item.weight);
      const { value, outputs } = this._runArena([
        this._intEntry('ids', ids),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dw', dw, true),
      ], (pointers) => this.api.volvoxai_training_embedding_backward_f32(
        pointers.ids, pointers.dy, pointers.dw, item.tokens, item.vocab, item.dim,
      ));
      apiResult(value, `Embedding backward at node '${node.id}'`);
      dw.set(outputs.dw);
      return true;
    }

    if (item.kind === 'layernorm' || item.kind === 'rmsnorm') {
      const dx = gradientFor(item.x);
      const dw = gradientFor(item.weight);
      const entries = [
        this._floatEntry('x', requireFloatStorage(item.x, `${item.kind} input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `${item.kind} weight at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._floatEntry('dw', dw, true),
      ];
      let db: Float32Array | null = null;
      if (item.kind === 'layernorm') {
        db = gradientFor(item.bias);
        entries.push(this._floatEntry('db', db, true));
      }
      const { outputs } = this._runArena(entries, (pointers) => {
        if (item.kind === 'layernorm') {
          this.api.volvoxai_training_layernorm_backward_f32(
            pointers.x, pointers.weight, pointers.dy, pointers.dx, pointers.dw, pointers.db,
            item.rows, item.width, item.epsilon,
          );
        } else {
          this.api.volvoxai_training_rmsnorm_backward_f32(
            pointers.x, pointers.weight, pointers.dy, pointers.dx, pointers.dw,
            item.rows, item.width, item.epsilon,
          );
        }
      });
      dx.set(outputs.dx); dw.set(outputs.dw); if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'groupnorm') {
      const dx = gradientFor(item.x);
      const dw = gradientFor(item.weight);
      const db = gradientFor(item.bias);
      const { value, outputs } = this._runArena([
        this._floatEntry('x', requireFloatStorage(item.x, `GroupNorm input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `GroupNorm weight at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._floatEntry('dw', dw, true),
        this._floatEntry('db', db, true),
      ], (pointers) => this.api.volvoxai_training_groupnorm_backward_f32(
        pointers.x, pointers.weight, pointers.dy, pointers.dx, pointers.dw, pointers.db,
        item.batch, item.height, item.width, item.channels, item.groups, item.epsilon,
      ));
      apiResult(value, `GroupNorm backward at node '${node.id}'`);
      dx.set(outputs.dx); dw.set(outputs.dw); db.set(outputs.db);
      return true;
    }

    if (item.kind === 'softmax') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('output', requireFloatStorage(item.output, `${node.opType} output at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_softmax_backward_f32(
        pointers.output, pointers.dy, pointers.dx, item.rows, item.width, item.logOutput ? 1 : 0,
      ));
      apiResult(value, `${node.opType} backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'concat') {
      let axisOffset = 0;
      for (const input of item.inputs) {
        const dx = gradientFor(input);
        const { outputs } = this._runArena([
          this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true),
        ], (pointers) => this.api.volvoxai_training_concat_backward_f32(
          pointers.dy, pointers.dx, item.outer, input.shape[item.axis], item.outputAxis, item.inner, axisOffset,
        ));
        dx.set(outputs.dx);
        axisOffset += input.shape[item.axis];
      }
      return true;
    }

    if (item.kind === 'conv2d') {
      const dx = gradientFor(item.x), dw = gradientFor(item.weight), db = item.bias ? gradientFor(item.bias) : null;
      const entries = [this._floatEntry('x', requireFloatStorage(item.x, `Conv2D input at node '${node.id}'`)), this._floatEntry('w', requireFloatStorage(item.weight, `Conv2D weight at node '${node.id}'`)), this._floatEntry('out', requireFloatStorage(item.output, `Conv2D output at node '${node.id}'`)), this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true), this._floatEntry('dw', dw, true)];
      if (db) entries.push(this._floatEntry('db', db, true));
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_conv2d_backward_f32(p.x, p.w, p.out, p.dy, p.dx, p.dw, db ? p.db : 0, item.batch, item.inH, item.inW, item.inC, item.kh, item.kw, item.groupIn, item.outC, item.outH, item.outW, item.strideY, item.strideX, item.padTop, item.padLeft, item.groups, item.relu, item.dilationY, item.dilationX));
      apiResult(value, `Conv2D backward at node '${node.id}'`); dx.set(outputs.dx); dw.set(outputs.dw); if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'conv1d') {
      const dx = gradientFor(item.x), dw = gradientFor(item.weight), db = item.bias ? gradientFor(item.bias) : null;
      const entries = [
        this._floatEntry('x', requireFloatStorage(item.x, `Conv1D input at node '${node.id}'`)),
        this._floatEntry('w', requireFloatStorage(item.weight, `Conv1D weight at node '${node.id}'`)),
        this._floatEntry('out', requireFloatStorage(item.output, `Conv1D output at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true), this._floatEntry('dw', dw, true),
      ];
      if (db) entries.push(this._floatEntry('db', db, true));
      const { value, outputs } = this._runArena(entries, (p) => this.api.volvoxai_training_conv1d_backward_f32(
        p.x, p.w, p.out, p.dy, p.dx, p.dw, db ? p.db : 0, item.batch, item.inChannels, item.inputLength,
        item.outChannels, item.inputPerGroup, item.kernel, item.outputLength, item.stride, item.padding, item.groups, item.relu,
      ));
      apiResult(value, `Conv1D backward at node '${node.id}'`); dx.set(outputs.dx); dw.set(outputs.dw); if (db) db.set(outputs.db);
      return true;
    }

    if (item.kind === 'convtranspose2d') {
      const dx=gradientFor(item.x),dw=gradientFor(item.weight),db=item.bias?gradientFor(item.bias):null;
      const entries=[this._floatEntry('x',requireFloatStorage(item.x,`ConvTranspose2D input at node '${node.id}'`)),this._floatEntry('w',requireFloatStorage(item.weight,`ConvTranspose2D weight at node '${node.id}'`)),this._floatEntry('dy',outputGradient),this._floatEntry('dx',dx,true),this._floatEntry('dw',dw,true)]; if(db) entries.push(this._floatEntry('db',db,true));
      const {value,outputs}=this._runArena(entries,p=>this.api.volvoxai_training_conv_transpose2d_backward_f32(p.x,p.w,p.dy,p.dx,p.dw,db?p.db:0,item.batch,item.inHeight,item.inWidth,item.inChannels,item.outHeight,item.outWidth,item.outChannels,item.kernelY,item.kernelX,item.strideY,item.strideX,item.padY,item.padX));
      apiResult(value,`ConvTranspose2D backward at node '${node.id}'`); dx.set(outputs.dx);dw.set(outputs.dw);if(db)db.set(outputs.db);return true;
    }

    if(item.kind==='pad2d'){
      const dx=gradientFor(item.x),{value,outputs}=this._runArena([this._floatEntry('dy',outputGradient),this._floatEntry('dx',dx,true)],p=>this.api.volvoxai_training_pad2d_backward_f32(p.dy,p.dx,item.batch,item.height,item.width,item.channels,item.padTop,item.padBottom,item.padLeft,item.padRight));
      apiResult(value,`Pad backward at node '${node.id}'`);dx.set(outputs.dx);return true;
    }
    if(item.kind==='interp1d'){
      const dx=gradientFor(item.x),{value,outputs}=this._runArena([this._floatEntry('dy',outputGradient),this._floatEntry('dx',dx,true)],p=>this.api.volvoxai_training_interp1d_backward_f32(p.dy,p.dx,item.batch,item.channels,item.inputLength,item.outputLength));apiResult(value,`Interpolate1D backward at node '${node.id}'`);dx.set(outputs.dx);return true;
    }

    if (item.kind === 'prelu') {
      const dx = gradientFor(item.x), ds = gradientFor(item.slope);
      const { value, outputs } = this._runArena([this._floatEntry('x', requireFloatStorage(item.x, `PReLU input at node '${node.id}'`)), this._floatEntry('slope', requireFloatStorage(item.slope, `PReLU slope at node '${node.id}'`)), this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true), this._floatEntry('ds', ds, true)], (p) => this.api.volvoxai_training_prelu_backward_f32(p.x, p.slope, p.dy, p.dx, p.ds, item.elements, item.slopeElements, item.channels));
      apiResult(value, `PReLU backward at node '${node.id}'`); dx.set(outputs.dx); ds.set(outputs.ds); return true;
    }

    if (item.kind === 'globalAveragePool') {
      const dx=gradientFor(item.x); const { outputs }=this._runArena([this._floatEntry('dy',outputGradient),this._floatEntry('dx',dx,true)],p=>this.api.volvoxai_training_global_average_pool_backward_f32(p.dy,p.dx,item.batch,item.height,item.width,item.channels)); dx.set(outputs.dx); return true;
    }

    if (item.kind === 'activation') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('x', requireFloatStorage(item.x, `activation input at node '${node.id}'`)),
        this._floatEntry('y', requireFloatStorage(item.output, `activation output at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_activation_backward_f32(
        item.activation, pointers.x, pointers.y, pointers.dy, pointers.dx, item.elements, item.alpha,
      ));
      apiResult(value, `${item.node.opType} backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'batchnorm2d') {
      const dx = gradientFor(item.x), dw = gradientFor(item.weight), db = gradientFor(item.bias);
      const { outputs } = this._runArena([
        this._floatEntry('x', requireFloatStorage(item.x, `BatchNorm2D input at node '${node.id}'`)),
        this._floatEntry('weight', requireFloatStorage(item.weight, `BatchNorm2D weight at node '${node.id}'`)),
        this._floatEntry('mean', requireFloatStorage(item.runningMean, `BatchNorm2D running mean at node '${node.id}'`)),
        this._floatEntry('variance', requireFloatStorage(item.runningVar, `BatchNorm2D running variance at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true), this._floatEntry('dw', dw, true), this._floatEntry('db', db, true),
      ], (p) => this.api.volvoxai_training_batchnorm2d_backward_f32(p.x, p.weight, p.mean, p.variance, p.dy, p.dx, p.dw, p.db, item.batch, item.height, item.width, item.channels, item.epsilon));
      dx.set(outputs.dx); dw.set(outputs.dw); db.set(outputs.db);
      return true;
    }

    if (item.kind === 'maxpool2d' || item.kind === 'averagepool2d') {
      const dx = gradientFor(item.x);
      const entries = [this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true)];
      if (item.kind === 'maxpool2d') entries.unshift(this._floatEntry('x', requireFloatStorage(item.x, `MaxPool2D input at node '${node.id}'`)));
      const { outputs } = this._runArena(entries, (p) => item.kind === 'maxpool2d'
        ? this.api.volvoxai_training_maxpool2d_backward_f32(p.x, p.dy, p.dx, item.batch, item.height, item.width, item.channels, item.outHeight, item.outWidth, item.kernelY, item.kernelX, item.strideY, item.strideX, item.padY, item.padX)
        : this.api.volvoxai_training_averagepool2d_backward_f32(p.dy, p.dx, item.batch, item.height, item.width, item.channels, item.outHeight, item.outWidth, item.kernelY, item.kernelX, item.strideY, item.strideX, item.padY, item.padX));
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'resize2d') {
      const dx = gradientFor(item.x);
      const { outputs } = this._runArena([this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true)], (p) => this.api.volvoxai_training_resize2d_backward_f32(p.dy, p.dx, item.batch, item.inHeight, item.inWidth, item.channels, item.outHeight, item.outWidth, item.nearest));
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'dropout') {
      const dx = gradientFor(item.x);
      if (!trainingDropout) throw new Error(`WASM training Dropout node '${node.id}' is missing RNG context.`);
      const { outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_dropout_backward_f32(
        pointers.dy, pointers.dx, item.elements, dropoutThreshold(item.probability),
        dropoutEffectiveSeed(item.node, trainingDropout, nodeIndex), trainingDropout.counter >>> 0,
        1 / (1 - item.probability),
      ));
      dx.set(outputs.dx);
      return true;
    }

    if (['add', 'mul', 'sub', 'div'].includes(item.kind)) {
      const da = gradientFor(item.a);
      const db = gradientFor(item.b);
      const entries = [
        this._floatEntry('a', requireFloatStorage(item.a, `${node.opType} left input at node '${node.id}'`)),
        this._floatEntry('b', requireFloatStorage(item.b, `${node.opType} right input at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('da', da, true),
        this._floatEntry('db', db, true),
        this._uintEntry('aShape', Uint32Array.from(item.a.shape)),
        this._uintEntry('bShape', Uint32Array.from(item.b.shape)),
        this._uintEntry('outShape', Uint32Array.from(item.output.shape)),
      ];
      const { value, outputs } = this._runArena(entries, (pointers) =>
        this.api.volvoxai_training_binary_broadcast_backward_f32(
          pointers.a, pointers.b, pointers.dy, pointers.da, pointers.db,
          pointers.aShape, pointers.bShape, pointers.outShape, item.a.shape.length,
          item.b.shape.length, item.output.shape.length, item.elements,
          { add: 0, mul: 1, sub: 2, div: 3 }[item.kind],
        ));
      apiResult(value, `${node.opType} backward at node '${node.id}'`);
      da.set(outputs.da); db.set(outputs.db);
      return true;
    }

    if (item.kind === 'where') {
      const condition = item.condition.dtype === 'int32'
        ? this._intEntry('condition', item.condition.buffer)
        : this._floatEntry('condition', item.condition.buffer);
      const da = gradientFor(item.a);
      const db = gradientFor(item.b);
      const { value, outputs } = this._runArena([
        condition,
        this._floatEntry('dy', outputGradient),
        this._floatEntry('da', da, true),
        this._floatEntry('db', db, true),
      ], (pointers) => this.api.volvoxai_training_where_backward_f32(
        pointers.condition, pointers.dy, pointers.da, pointers.db,
        item.conditionType, item.elements,
      ));
      apiResult(value, `${node.opType} backward at node '${node.id}'`);
      da.set(outputs.da);
      db.set(outputs.db);
      return true;
    }

    if (item.kind === 'slice') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._uintEntry('inputShape', Uint32Array.from(item.x.shape)),
        this._uintEntry('outputShape', Uint32Array.from(item.output.shape)),
        this._uintEntry('starts', Uint32Array.from(item.starts)),
        this._uintEntry('steps', Uint32Array.from(item.steps)),
      ], (pointers) => this.api.volvoxai_training_slice_backward_f32(
        pointers.dy, pointers.dx, pointers.inputShape, pointers.outputShape,
        pointers.starts, pointers.steps, item.rank, item.elements,
      ));
      apiResult(value, `Slice backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'gather') {
      const dx = gradientFor(item.x);
      const indices = item.indices.buffer;
      if (!(indices instanceof Int32Array)) {
        throw new Error(`Gather indices at node '${node.id}' are not initialized Int32 storage.`);
      }
      const { value, outputs } = this._runArena([
        this._intEntry('indices', indices),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_gather_backward_f32(
        pointers.indices, pointers.dy, pointers.dx, item.outer, item.axisSize,
        item.inner, item.indicesElements, item.elements,
      ));
      apiResult(value, `Gather backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'gatherElements') {
      const dx = gradientFor(item.x);
      const indices = item.indices.buffer;
      if (!(indices instanceof Int32Array)) {
        throw new Error(`GatherElements indices at node '${node.id}' are not initialized Int32 storage.`);
      }
      const { value, outputs } = this._runArena([
        this._intEntry('indices', indices),
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._uintEntry('inputShape', Uint32Array.from(item.x.shape)),
        this._uintEntry('indicesShape', Uint32Array.from(item.indices.shape)),
      ], (pointers) => this.api.volvoxai_training_gather_elements_backward_f32(
        pointers.indices, pointers.dy, pointers.dx, pointers.inputShape, pointers.indicesShape,
        item.rank, item.axis, item.elements,
      ));
      apiResult(value, `GatherElements backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'meanHeight') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_mean_height_backward_f32(
        pointers.dy, pointers.dx, item.batch, item.height, item.width, item.channels,
      ));
      apiResult(value, `MeanHeight backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'profileX' || item.kind === 'profileY') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('x', requireFloatStorage(item.x, `${node.opType} input at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true),
      ], (pointers) => item.kind === 'profileX'
        ? this.api.volvoxai_training_profile_x_backward_f32(
          pointers.x, pointers.dy, pointers.dx, item.batch, item.height, item.width, item.channels,
        )
        : this.api.volvoxai_training_profile_y_backward_f32(
          pointers.x, pointers.dy, pointers.dx, item.batch, item.height, item.width, item.channels,
        ));
      apiResult(value, `${node.opType} backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'spatialSoftargmaxY') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('x', requireFloatStorage(item.x, `SpatialSoftargmaxY input at node '${node.id}'`)),
        this._floatEntry('output', requireFloatStorage(item.output, `SpatialSoftargmaxY output at node '${node.id}'`)),
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_spatial_softargmax_y_backward_f32(
        pointers.x, pointers.output, pointers.dy, pointers.dx,
        item.batch, item.height, item.width, item.channels,
      ));
      apiResult(value, `SpatialSoftargmaxY backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'expand') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('dy', outputGradient), this._floatEntry('dx', dx, true),
        this._uintEntry('inShape', Uint32Array.from(item.x.shape)),
        this._uintEntry('outShape', Uint32Array.from(item.output.shape)),
      ], (p) => this.api.volvoxai_training_expand_backward_f32(
        p.dy, p.dx, p.inShape, p.outShape, item.x.shape.length, item.output.shape.length, item.elements,
      ));
      apiResult(value, `${node.opType} backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'reduce') {
      const dx = gradientFor(item.x);
      const { outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_reduce_backward_f32(
        pointers.dy, pointers.dx, item.rows, item.width, item.mean ? 1 / item.width : 1,
      ));
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'copy') {
      const dx = gradientFor(item.x);
      const { outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
      ], (pointers) => this.api.volvoxai_training_copy_backward_f32(
        pointers.dy, pointers.dx, item.elements,
      ));
      dx.set(outputs.dx);
      return true;
    }

    if (item.kind === 'transpose') {
      const dx = gradientFor(item.x);
      const { value, outputs } = this._runArena([
        this._floatEntry('dy', outputGradient),
        this._floatEntry('dx', dx, true),
        this._uintEntry('shape', Uint32Array.from(item.x.shape)),
        this._uintEntry('perm', Uint32Array.from(item.perm)),
      ], (pointers) => this.api.volvoxai_training_transpose_backward_f32(
        pointers.dy, pointers.dx, pointers.shape, pointers.perm, item.rank, item.elements,
      ));
      apiResult(value, `Transpose backward at node '${node.id}'`);
      dx.set(outputs.dx);
      return true;
    }

    return false;
  }

  allFinite(values: Float32Array): boolean {
    const { value } = this._runArena([
      this._floatEntry('values', values),
    ], (pointers) => this.api.volvoxai_training_all_finite_f32(pointers.values, values.length));
    return Number(value) === 1;
  }

  clipGradients(gradients: Map<string, Float32Array>, maxGradNorm = 0) {
    if (typeof maxGradNorm !== 'number' || !Number.isFinite(maxGradNorm) || maxGradNorm < 0) {
      throw new Error('maxGradNorm must be finite and non-negative.');
    }
    let sumSquares = 0;
    for (const [name, gradient] of gradients) {
      if (!(gradient instanceof Float32Array)) throw new Error(`Gradient '${name}' must be F32.`);
      const { value } = this._runArena([
        this._floatEntry('gradient', gradient),
      ], (pointers) => this.api.volvoxai_training_sum_squares_f32(pointers.gradient, gradient.length));
      sumSquares += Number(value);
    }
    const norm = Math.sqrt(sumSquares);
    const scale = maxGradNorm > 0 && norm > maxGradNorm ? maxGradNorm / (norm + 1e-12) : 1;
    if (scale !== 1) {
      for (const gradient of gradients.values()) {
        const { outputs } = this._runArena([
          this._floatEntry('gradient', gradient, true),
        ], (pointers) => this.api.volvoxai_training_scale_f32(pointers.gradient, gradient.length, scale));
        gradient.set(outputs.gradient);
      }
    }
    return { norm, scale };
  }

  applyTensorUpdate(
    graph: StatefulTrainingGraph,
    tensor: Tensor,
    gradient: Float32Array,
    options: TrainingTensorUpdateOptions,
  ): Tensor {
    const weights = requireFloatStorage(tensor, `trainable tensor '${tensor.name}'`);
    if (!(gradient instanceof Float32Array) || gradient.length !== weights.length) {
      throw new Error(`WASM training gradient for '${tensor.name}' has an incompatible shape.`);
    }
    const mode = options.mode;
    const learningRate = options.learningRate ?? options.lr ?? 1e-3;
    const weightDecay = options.weightDecay ?? 0;
    const entries: WasmArenaEntry[] = [
      this._floatEntry('weights', weights, true),
      this._floatEntry('gradient', gradient),
    ];
    let firstMoment: Float32Array | null = null;
    let secondMoment: Float32Array | null = null;
    let existingState: { m: Float32Array; v: Float32Array; step: number } | undefined;
    if (mode === 'adamw') {
      existingState = graph.optimizerState?.get(tensor.name);
      firstMoment = existingState && existingState.m.length === weights.length
        ? existingState.m : new Float32Array(weights.length);
      secondMoment = existingState && existingState.v.length === weights.length
        ? existingState.v : new Float32Array(weights.length);
      entries.push(this._floatEntry('firstMoment', firstMoment, true));
      entries.push(this._floatEntry('secondMoment', secondMoment, true));
    }
    const { value, outputs } = this._runArena(entries, (pointers) => {
      if (mode === 'sgd') {
        this.api.volvoxai_training_sgd_update_f32(
          pointers.weights, pointers.gradient, weights.length, learningRate, weightDecay,
        );
        return 1;
      }
      if (mode !== 'adamw') throw new Error(`WASM training optimizer '${mode}' is unsupported.`);
      const step = options.step;
      if (typeof step !== 'number' || !Number.isSafeInteger(step) || step <= 0 || step > MAX_U32) {
        throw new Error('WASM training AdamW step must fit uint32.');
      }
      return this.api.volvoxai_training_adamw_update_f32(
        pointers.weights, pointers.gradient, pointers.firstMoment, pointers.secondMoment,
        weights.length, learningRate, options.beta1 ?? 0.9, options.beta2 ?? 0.999,
        options.epsilon ?? 1e-8, weightDecay, step,
      );
    });
    apiResult(value, `${mode} update for '${tensor.name}'`);
    weights.set(outputs.weights);
    tensor.sizeBytes = weights.byteLength;
    graph._advanceWeightRevision();
    if (mode === 'adamw') {
      graph.optimizerState ||= new Map();
      graph.optimizerState.set(tensor.name, {
        m: outputs.firstMoment,
        v: outputs.secondMoment,
        step: options.step!,
      });
    }
    return tensor;
  }
}
