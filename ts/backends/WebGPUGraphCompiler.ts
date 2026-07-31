import { Tensor } from '../core/Tensor.js';
import { DataType } from '../generated/volvoxaiEnums.js';
import type { Graph } from '../core/Graph.js';
import type { ShaderLibrary as ShaderLibraryClass } from './ShaderLibrary.js';
import { geluApproximation } from '../ops/gELU.js';
import { normalizeSpatialPair } from '../ops/spatialParameters.js';
import { batchMatMulDescriptor } from '../ops/batchMatMul.js';
import { qBatchMatMulMetadataDescriptor } from '../ops/qBatchMatMul.js';
import { comparisonDescriptor, logicalNotDescriptor } from '../ops/comparison.js';
import { qEmbeddingIdsArePreflightComplete } from '../ops/quantizedGraphValidation.js';
import type { GraphExecutor } from './GraphExecutor.js';
import type {
  CompiledWebGPUPipeline,
  ExecutorNode,
  ExecutorTensor,
  IncrementalRowByteCopy,
} from './WebGPUContracts.js';

// ShaderLibrary statically imports WGSL text. Keep it lazy so importing CPU or
// WASM compositions under Node never asks the module loader to resolve WGSL.
type ShaderLibraryConstructor = typeof ShaderLibraryClass;
let ShaderLibrary: ShaderLibraryConstructor;

export interface WebGPUCompiledGraphPlan {
  readonly dinWeights: readonly (readonly [
    string,
    Readonly<{ din: number; dout: number }>,
  ])[];
  readonly resultCopyTensorNames: readonly string[];
}

function nodeOutput(node: Graph['nodes'][number]) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}

function dropoutInput(node: Graph['nodes'][number]) {
  return node.inputs?.input || node.inputs?.x || node.inputs?.data;
}

/** Pure, immutable planning performed before context resources are allocated. */
export function compileWebGPUGraphPlan(
  graph: Graph,
): WebGPUCompiledGraphPlan {
  const dinWeights = new Map<string, { readonly din: number; readonly dout: number }>();
  for (const node of graph.nodes) {
    if (Object.prototype.hasOwnProperty.call(node.params || {}, 'coordinate_transform_mode')) {
      throw new Error(
        `WebGPU node ${String(node.id)} uses unsupported 'coordinate_transform_mode'; ` +
        "use 'coordinate_transformation_mode'.",
      );
    }
    if ((node.opType === 'MatMul' || node.opType === 'Linear' || node.opType === 'Gemm') &&
        node.wLayout === 'din' && node.inputs.weight && !node.inputs.scale &&
        !node.inputs.weight_scale) {
      const input = node.inputs.input || node.inputs.x || node.inputs.a;
      const output = nodeOutput(node);
      const din = input?.shape?.at(-1);
      const dout = output?.shape?.at(-1);
      if (!Number.isInteger(din) || !Number.isInteger(dout) || din! <= 0 || dout! <= 0) {
        throw new Error(`WebGPU linear node ${String(node.id)} has invalid din/dout dimensions.`);
      }
      dinWeights.set(node.inputs.weight.name, Object.freeze({ din: din!, dout: dout! }));
    }
  }

  const resultCopyTensorNames = new Set(graph.outputNames || []);
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of graph.nodes) {
      if (node.opType !== 'Dropout') continue;
      const input = dropoutInput(node);
      const output = nodeOutput(node);
      if (input && output && resultCopyTensorNames.has(output.name) &&
          !resultCopyTensorNames.has(input.name)) {
        resultCopyTensorNames.add(input.name);
        changed = true;
      }
    }
  }

  const immutableDinWeights = Object.freeze([...dinWeights].map(([name, dimensions]) =>
    Object.freeze([name, dimensions] as const)));
  const immutableResultCopyTensorNames = Object.freeze([...resultCopyTensorNames]);
  return Object.freeze({
    dinWeights: immutableDinWeights,
    resultCopyTensorNames: immutableResultCopyTensorNames,
  });
}

function concatInputEntries(node: ExecutorNode): Array<[string, ExecutorTensor]> {
  const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const seen = new Set<string>();
  const ordered: Array<[string, ExecutorTensor]> = [];
  for (const key of preferred) {
    if (node.inputs[key]) {
      ordered.push([key, node.inputs[key]]);
      seen.add(key);
    }
  }
  const remaining = Object.entries(node.inputs)
    .filter(([key, tensor]) => tensor && !seen.has(key))
    .sort(([left], [right]) => {
      const li = /^input(\d+)$/.exec(left);
      const ri = /^input(\d+)$/.exec(right);
      if (li && ri) return Number(li[1]) - Number(ri[1]);
      return left.localeCompare(right);
    });
  ordered.push(...remaining);
  return ordered;
}

function float32Storage(storage: ExecutorTensor['buffer']): Float32Array {
  if (storage instanceof ArrayBuffer) return new Float32Array(storage);
  return new Float32Array(storage as ArrayLike<number>);
}

function attentionMaskMode(mask, batch, seqQ, seqKV, nodeId) {
  if (!mask) return 0;
  if (mask.dtype !== 'int32') {
    throw new Error(`Attention mask at node ${nodeId} must use int32 binary storage.`);
  }
  const shape = mask.shape;
  if (shape.length === 1 && shape[0] === seqKV) return 1;
  if (shape.length === 2 && shape[1] === seqKV) {
    if (shape[0] === batch) return 2;
    if (shape[0] === seqQ) return 3;
  }
  if (shape.length === 3 && shape[0] === batch && shape[1] === seqQ && shape[2] === seqKV) return 4;
  throw new Error(`Attention mask at node ${nodeId} must be [K], [B,K], [Q,K], or [B,Q,K].`);
}

function sameTensorShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function webGpuFalseOrAbsent(value) {
  return value == null || value === false || value === 0;
}

function webGpuTypedNearestParameters(node) {
  const params = node.params || {};
  if (node.opType === 'Resize') {
    if (params.mode !== 'nearest') {
      throw new Error(`WebGPU Resize node ${node.id} supports raw I8/U8 storage only for nearest-neighbor mode.`);
    }
  } else if (params.mode != null && params.mode !== 'nearest') {
    throw new Error(`WebGPU ResizeNearest2D node ${node.id} supports raw I8/U8 storage only with mode "nearest".`);
  }
  if (Object.prototype.hasOwnProperty.call(params, 'coordinate_transform_mode')) {
    throw new Error(
      `WebGPU ${node.opType} node ${node.id} uses unsupported 'coordinate_transform_mode'; ` +
      "use 'coordinate_transformation_mode'.",
    );
  }
  if (params.coordinate_transformation_mode != null &&
      params.coordinate_transformation_mode !== 'asymmetric') {
    throw new Error(`WebGPU ${node.opType} node ${node.id} supports raw I8/U8 nearest resize only with coordinate_transformation_mode "asymmetric".`);
  }
  if (params.nearest_mode != null && params.nearest_mode !== 'floor') {
    throw new Error(`WebGPU ${node.opType} node ${node.id} supports raw I8/U8 nearest resize only with nearest_mode "floor".`);
  }
  if (!webGpuFalseOrAbsent(params.align_corners) || !webGpuFalseOrAbsent(params.antialias)) {
    throw new Error(`WebGPU ${node.opType} node ${node.id} does not support align_corners or antialias for raw I8/U8 nearest resize.`);
  }
}

const WEBGPU_SLICE_MAX_RANK = 8;
function webGpuTypedDtypeCode(dtype) {
  switch (dtype) {
    case 'float32': return DataType.F32;
    case 'int32': return DataType.I32;
    case 'int8': return DataType.I8;
    case 'uint8': return DataType.U8;
    default: return -1;
  }
}

function webGpuTypedElementCount(tensor) {
  if (!tensor || webGpuTypedDtypeCode(tensor.dtype) < 0 || !Array.isArray(tensor.shape) ||
      tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) return null;
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (!Number.isSafeInteger(elements) || elements <= 0 || elements > 0xffffffff ||
      !Number.isSafeInteger(expectedBytes) || tensor.sizeBytes !== expectedBytes) return null;
  return elements;
}

function webGpuBuffer(gpuBuffers, tensor) {
  return tensor ? gpuBuffers.get(tensor.name) : undefined;
}

function webGpuPerTensorQuantization(tensor, label) {
  const descriptor = tensor?.quantization;
  const scale = Math.fround(descriptor?.scale);
  const minimum = tensor?.dtype === 'int8' ? -128 : 0;
  const maximum = tensor?.dtype === 'int8' ? 127 : 255;
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) || descriptor?.scheme !== 'per_tensor' ||
      !Number.isFinite(descriptor.scale) || descriptor.scale <= 0 || !Number.isFinite(scale) || scale <= 0 ||
      !Number.isInteger(descriptor.zero_point) || descriptor.zero_point < minimum ||
      descriptor.zero_point > maximum) {
    throw new Error(`${label} requires I8/U8 storage with finite per_tensor scale and an in-range zero_point.`);
  }
  return descriptor;
}

function webGpuQGroupNormDescriptor(node, gpuBuffers) {
  const inputNames = Object.keys(node.inputs || {}).sort();
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const output = node.outputs?.out || outputEntries[0];
  const inputElements = webGpuTypedElementCount(input);
  const outputElements = webGpuTypedElementCount(output);
  const params = node.params;
  const paramsAreObject = params != null && typeof params === 'object' && !Array.isArray(params);
  const parameterNames = paramsAreObject ? Object.keys(params) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['num_groups', 'eps', 'data_layout'].includes(name));
  const numGroups = params?.num_groups;
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const affineStorageIsCanonical = (tensor) => {
    if (!tensor || tensor.dtype !== 'float32' || !sameTensorShape(tensor.shape, [channels]) ||
        tensor.sizeBytes !== channels * Float32Array.BYTES_PER_ELEMENT) return false;
    // Static parameters are uploaded at compile time, so fail before any GPU
    // work if they contain NaN/Inf. Dynamic graph-input or producer values may
    // intentionally have placeholders here; their supplied execution values
    // are checked immediately before upload/dispatch instead.
    if (tensor.buffer == null) return tensor.isWeight !== true;
    return tensor.buffer instanceof Float32Array && tensor.buffer.length === channels &&
      (tensor.isWeight !== true || tensor.buffer.every(Number.isFinite));
  };
  const storageRangesOverlap = (left, right) => {
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
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputEntries.length !== 1 || !input || !weight || !bias || !output ||
      input === output || output === weight || output === bias || !webGpuBuffer(gpuBuffers, input) ||
      !webGpuBuffer(gpuBuffers, weight) || !webGpuBuffer(gpuBuffers, bias) ||
      !webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, input) === webGpuBuffer(gpuBuffers, output) ||
      inputElements == null || outputElements == null || inputElements !== outputElements ||
      input.shape.length !== 4 || !sameTensorShape(input.shape, output.shape) ||
      !allowedParameters || !Number.isInteger(numGroups) || numGroups <= 0 ||
      (params?.data_layout != null && params.data_layout !== 'NHWC') ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new Error(`WebGPU QGroupNorm node ${node.id} requires exact input/weight/bias inputs, matching NHWC I8/U8 activation tensors, and positive num_groups/eps parameters.`);
  }

  const [batch, height, width, channels] = input.shape;
  if (channels % numGroups !== 0 || !affineStorageIsCanonical(weight) ||
      !affineStorageIsCanonical(bias) ||
      webGpuBuffer(gpuBuffers, weight) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, bias) === webGpuBuffer(gpuBuffers, output) ||
      storageRangesOverlap(input, output) ||
      storageRangesOverlap(weight, output) || storageRangesOverlap(bias, output)) {
    throw new Error(`WebGPU QGroupNorm node ${node.id} requires finite F32 [C] weight/bias tensors and num_groups dividing C.`);
  }

  const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QGroupNorm node ${node.id} input`);
  const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QGroupNorm node ${node.id} output`);
  const groupCount = batch * numGroups;
  const statsBytes = groupCount * 2 * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(groupCount) || groupCount <= 0 || !Number.isSafeInteger(statsBytes) ||
      statsBytes <= 0 || statsBytes > 0xffffffff) {
    throw new Error(`WebGPU QGroupNorm node ${node.id} has an unsupported group-statistics buffer size.`);
  }
  return {
    input, weight, bias, output, inputElements,
    batch, height, width, channels, numGroups, groupCount, statsBytes,
    inputQuantization, outputQuantization, epsilon,
  };
}

function webGpuQLayerNormDescriptor(node, gpuBuffers) {
  const inputNames = Object.keys(node.inputs || {}).sort();
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const output = node.outputs?.out || outputEntries[0];
  const inputElements = webGpuTypedElementCount(input);
  const outputElements = webGpuTypedElementCount(output);
  const params = node.params ?? {};
  const paramsAreObject = typeof params === 'object' && !Array.isArray(params);
  const parameterNames = paramsAreObject ? Object.keys(params) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['eps', 'd_model'].includes(name));
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const dModel = input?.shape?.at(-1);
  const affineStorageIsCanonical = (tensor) => {
    if (!tensor || tensor.dtype !== 'float32' || !sameTensorShape(tensor.shape, [dModel]) ||
        tensor.sizeBytes !== dModel * Float32Array.BYTES_PER_ELEMENT) return false;
    // Static parameters are captured at compile time and must already be
    // finite. Dynamic graph-input or producer placeholders are checked before
    // execution uploads rather than rejected while the graph is compiled.
    if (tensor.buffer == null) return tensor.isWeight !== true;
    return tensor.buffer instanceof Float32Array && tensor.buffer.length === dModel &&
      (tensor.isWeight !== true || tensor.buffer.every(Number.isFinite));
  };
  const storageRangesOverlap = (left, right) => {
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
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputEntries.length !== 1 || !input || !weight || !bias || !output ||
      input === output || output === weight || output === bias || !webGpuBuffer(gpuBuffers, input) ||
      !webGpuBuffer(gpuBuffers, weight) || !webGpuBuffer(gpuBuffers, bias) ||
      !webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, input) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, weight) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, bias) === webGpuBuffer(gpuBuffers, output) ||
      inputElements == null || outputElements == null || inputElements !== outputElements ||
      input.shape.length < 1 || !sameTensorShape(input.shape, output.shape) || !allowedParameters ||
      (params?.d_model != null && (!Number.isInteger(params.d_model) || params.d_model !== dModel)) ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new Error(`WebGPU QLayerNorm node ${node.id} requires exact input/weight/bias inputs, matching rank-at-least-1 I8/U8 activation tensors, and positive eps with optional d_model matching D.`);
  }

  if (!affineStorageIsCanonical(weight) || !affineStorageIsCanonical(bias) ||
      storageRangesOverlap(input, output) || storageRangesOverlap(weight, output) ||
      storageRangesOverlap(bias, output)) {
    throw new Error(`WebGPU QLayerNorm node ${node.id} requires finite F32 [D] weight/bias tensors with output storage distinct from every input.`);
  }

  const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QLayerNorm node ${node.id} input`);
  const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QLayerNorm node ${node.id} output`);
  const rows = inputElements / dModel;
  const statsBytes = rows * 2 * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(rows) || rows <= 0 || rows > 0xffffffff ||
      !Number.isSafeInteger(statsBytes) || statsBytes <= 0 || statsBytes > 0xffffffff) {
    throw new Error(`WebGPU QLayerNorm node ${node.id} has an unsupported row-statistics buffer size.`);
  }
  return {
    input, weight, bias, output, inputElements, rows, dModel, statsBytes,
    inputQuantization, outputQuantization, epsilon,
  };
}

function webGpuQMaskedMeanStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8'
      ? storage instanceof Uint8Array || storage instanceof Uint8ClampedArray
      : tensor.dtype === 'int32' && storage instanceof Int32Array;
  return matches && storage.byteLength === tensor.sizeBytes;
}

function webGpuQMaskedMeanDescriptor(node, gpuBuffers) {
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const mask = node.inputs?.mask;
  const output = node.outputs?.out;
  const inputElements = webGpuTypedElementCount(input);
  const maskElements = webGpuTypedElementCount(mask);
  const outputElements = webGpuTypedElementCount(output);
  const params = node.params || {};
  const paramsAreEmpty = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 0;
  const batch = input?.shape?.[0];
  const sequence = input?.shape?.[1];
  const width = input?.shape?.[2];
  const storageRangesOverlap = (left, right) => {
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
  };
  if (inputNames.length !== 2 || inputNames[0] !== 'input' || inputNames[1] !== 'mask' ||
      outputNames.length !== 1 || outputNames[0] !== 'out' || !input || !mask || !output ||
      input === output || !webGpuBuffer(gpuBuffers, input) || !webGpuBuffer(gpuBuffers, mask) ||
      !webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, input) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, mask) === webGpuBuffer(gpuBuffers, output) ||
      inputElements == null || maskElements == null || outputElements == null || !paramsAreEmpty ||
      !webGpuQMaskedMeanStorageIsCanonical(input) || !webGpuQMaskedMeanStorageIsCanonical(mask) ||
      !webGpuQMaskedMeanStorageIsCanonical(output) || input.shape.length !== 3 ||
      mask.dtype !== 'int32' || mask.quantization != null || mask.shape.length !== 2 ||
      output.shape.length !== 2 || mask.shape[0] !== batch || mask.shape[1] !== sequence ||
      output.shape[0] !== batch || output.shape[1] !== width ||
      inputElements !== batch * sequence * width || maskElements !== batch * sequence ||
      outputElements !== batch * width || storageRangesOverlap(input, output) ||
      storageRangesOverlap(mask, output)) {
    throw new Error(`WebGPU QMaskedMean node ${node.id} requires exactly { input, mask } -> { out }: I8/U8 [B,S,D], unquantized I32 [B,S], I8/U8 [B,D], and no parameters.`);
  }
  if (!Object.isFrozen(input.quantization) || !Object.isFrozen(output.quantization)) {
    throw new Error(`WebGPU QMaskedMean node ${node.id} requires immutable per_tensor input/output quantization metadata.`);
  }
  const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QMaskedMean node ${node.id} input`);
  const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QMaskedMean node ${node.id} output`);
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const multiplier = Math.fround(inputScale / outputScale);
  const minimum = input.dtype === 'int8' ? -128 : 0;
  const maximum = input.dtype === 'int8' ? 127 : 255;
  const maximumCenteredMagnitude = Math.max(
    Math.abs(minimum - inputQuantization.zero_point),
    Math.abs(maximum - inputQuantization.zero_point),
  );
  const maximumSum = sequence * maximumCenteredMagnitude;
  const outputWords = Math.ceil(outputElements / 4);
  if (!Number.isFinite(multiplier) || multiplier <= 0 || !Number.isSafeInteger(maximumSum) ||
      maximumSum > 0x7fffffff || !Number.isSafeInteger(outputWords) || outputWords <= 0 ||
      outputWords > 0xffffffff) {
    throw new Error(`WebGPU QMaskedMean node ${node.id} requires finite input/output scale ratio and an I32-safe centered sum.`);
  }
  return {
    input, mask, output, batch, sequence, width, inputElements, outputElements, outputWords,
    inputQuantization, outputQuantization,
    inputDtype: webGpuTypedDtypeCode(input.dtype), outputDtype: webGpuTypedDtypeCode(output.dtype),
    inputScale, outputScale,
  };
}

function webGpuQSDPAStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8'
      ? storage instanceof Uint8Array || storage instanceof Uint8ClampedArray
      : tensor.dtype === 'int32' && storage instanceof Int32Array;
  return matches && storage.byteLength === tensor.sizeBytes;
}

function webGpuQSDPADescriptor(node, gpuBuffers) {
  const inputNames = Object.keys(node.inputs || {}).sort();
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
  const mask = node.inputs?.mask;
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const output = node.outputs?.out || outputEntries[0];
  const qElements = webGpuTypedElementCount(q);
  const kElements = webGpuTypedElementCount(k);
  const vElements = webGpuTypedElementCount(v);
  const outputElements = webGpuTypedElementCount(output);
  const maskElements = hasMask ? webGpuTypedElementCount(mask) : 0;
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
  const storageRangesOverlap = (left, right) => {
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
  };

  const exactInputs = (inputNames.length === 3 && inputNames[0] === 'k' && inputNames[1] === 'q' &&
    inputNames[2] === 'v') || (inputNames.length === 4 && inputNames[0] === 'k' &&
    inputNames[1] === 'mask' && inputNames[2] === 'q' && inputNames[3] === 'v');
  if (!exactInputs || outputEntries.length !== 1 || !q || !k || !v || !output ||
      output === q || output === k || output === v || output === mask ||
      !webGpuBuffer(gpuBuffers, q) || !webGpuBuffer(gpuBuffers, k) ||
      !webGpuBuffer(gpuBuffers, v) || !webGpuBuffer(gpuBuffers, output) ||
      (hasMask && (!mask || !webGpuBuffer(gpuBuffers, mask))) ||
      webGpuBuffer(gpuBuffers, q) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, k) === webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, v) === webGpuBuffer(gpuBuffers, output) ||
      (hasMask && webGpuBuffer(gpuBuffers, mask) === webGpuBuffer(gpuBuffers, output)) ||
      qElements == null || kElements == null || vElements == null || outputElements == null ||
      (hasMask && maskElements == null) || !webGpuQSDPAStorageIsCanonical(q) ||
      !webGpuQSDPAStorageIsCanonical(k) || !webGpuQSDPAStorageIsCanonical(v) ||
      !webGpuQSDPAStorageIsCanonical(output) || (hasMask && !webGpuQSDPAStorageIsCanonical(mask)) ||
      (rank !== 2 && rank !== 3) || k.shape.length !== rank || v.shape.length !== rank ||
      output.shape.length !== rank || !sameTensorShape(q.shape, output.shape) || !allowedParameters ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'heads') ||
      !Number.isInteger(heads) || heads <= 0 ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'causal') || typeof params?.causal !== 'boolean' ||
      !Number.isInteger(headDim) || headDim <= 0 || dModel % 4 !== 0 || headDim % 4 !== 0 ||
      headDim > 64 || typeof scaleSource !== 'number' || !Number.isFinite(scaleSource) ||
      scaleSource <= 0 || !Number.isFinite(attentionScale) || attentionScale <= 0) {
    throw new Error(`WebGPU QSDPA node ${node.id} requires exact q/k/v and optional I32 mask inputs, rank-2/3 I8/U8 tensors, D/head dimensions divisible by 4 with head_dim <= 64, and explicit heads/causal.`);
  }

  const kBatch = rank === 2 ? 1 : k.shape[0];
  const vBatch = rank === 2 ? 1 : v.shape[0];
  if (kBatch !== batch || vBatch !== batch || k.shape[rank - 1] !== dModel ||
      v.shape[rank - 1] !== dModel || v.shape[rank - 2] !== seqKV ||
      qElements !== batch * seqQ * dModel || kElements !== batch * seqKV * dModel ||
      vElements !== batch * seqKV * dModel || outputElements !== qElements ||
      storageRangesOverlap(q, output) || storageRangesOverlap(k, output) ||
      storageRangesOverlap(v, output) || (hasMask && storageRangesOverlap(mask, output))) {
    throw new Error(`WebGPU QSDPA node ${node.id} has incompatible Q/K/V/output dimensions or output storage overlapping an input.`);
  }

  let maskMode = 0;
  if (hasMask) {
    if (mask.dtype !== 'int32' || !Array.isArray(mask.shape)) {
      throw new Error(`WebGPU QSDPA node ${node.id} mask requires I32 [K], [B,K], [Q,K], or [B,Q,K] storage.`);
    }
    if (mask.shape.length === 1 && mask.shape[0] === seqKV) maskMode = 1;
    // Preserve the established precedence for an ambiguous [N,K] with B==Q:
    // it is [B,K], not [Q,K].
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === batch) maskMode = 2;
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === seqQ) maskMode = 3;
    else if (mask.shape.length === 3 && mask.shape[0] === batch && mask.shape[1] === seqQ &&
        mask.shape[2] === seqKV) maskMode = 4;
    else throw new Error(`WebGPU QSDPA node ${node.id} mask requires I32 [K], [B,K], [Q,K], or [B,Q,K] storage.`);
  }

  const qQuantization = webGpuPerTensorQuantization(q, `WebGPU QSDPA node ${node.id} q`);
  const kQuantization = webGpuPerTensorQuantization(k, `WebGPU QSDPA node ${node.id} k`);
  const vQuantization = webGpuPerTensorQuantization(v, `WebGPU QSDPA node ${node.id} v`);
  const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QSDPA node ${node.id} output`);
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
  if (!Number.isFinite(scoreMultiplier) || scoreMultiplier <= 0 || !Number.isFinite(maximumScore)) {
    throw new Error(`WebGPU QSDPA node ${node.id} requires a finite positive score multiplier and score range.`);
  }
  return {
    q, k, v, mask: hasMask ? mask : null, output, batch, seqQ, seqKV, dModel, heads, headDim,
    maskMode, causal: params.causal ? 1 : 0, attentionScale,
    qQuantization, kQuantization, vQuantization, outputQuantization,
    qScale, kScale, vScale, outputScale,
  };
}

function webGpuQArgMaxByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}

function webGpuQArgMaxI32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}

function webGpuQArgMaxDescriptor(node, gpuBuffers) {
  const inputNames = Object.keys(node.inputs || {});
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  const inputElements = webGpuTypedElementCount(input);
  const outputElements = webGpuTypedElementCount(output);
  const params = node.params;
  const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 1 && Object.prototype.hasOwnProperty.call(params, 'axis') &&
    Number.isInteger(params.axis);
  const rank = input?.shape?.length;
  let axis = params?.axis;
  if (axis < 0) axis += rank;
  const outer = Number.isInteger(axis) && Array.isArray(input?.shape)
    ? input.shape.slice(0, axis).reduce((product, dimension) => product * dimension, 1) : NaN;
  const inner = Number.isInteger(axis) && Array.isArray(input?.shape)
    ? input.shape.slice(axis + 1).reduce((product, dimension) => product * dimension, 1) : NaN;
  const axisSize = input?.shape?.[axis];
  const expectedOutputShape = Number.isInteger(axis) && Array.isArray(input?.shape)
    ? [...input.shape.slice(0, axis), ...input.shape.slice(axis + 1)] : null;
  const storageRangesOverlap = (left, right) => {
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
  };
  if (inputNames.length !== 1 || inputNames[0] !== 'input' || outputNames.length !== 1 ||
      outputNames[0] !== 'out' || !input || !output || input === output ||
      !webGpuBuffer(gpuBuffers, input) || !webGpuBuffer(gpuBuffers, output) ||
      webGpuBuffer(gpuBuffers, input) === webGpuBuffer(gpuBuffers, output) || inputElements == null ||
      outputElements == null || !webGpuQArgMaxByteStorageIsCanonical(input) ||
      !webGpuQArgMaxI32StorageIsCanonical(output) || !paramsAreExact || !Number.isInteger(rank) ||
      rank < 2 || rank > 8 || !Number.isInteger(axis) || axis < 0 || axis >= rank ||
      !Number.isInteger(axisSize) || axisSize <= 0 || axisSize > 0x7fffffff ||
      !Number.isSafeInteger(outer) || outer <= 0 || outer > 0xffffffff ||
      !Number.isSafeInteger(inner) || inner <= 0 || inner > 0xffffffff ||
      inputElements !== outer * axisSize * inner || outputElements !== outer * inner ||
      !sameTensorShape(output.shape, expectedOutputShape) || output.dtype !== 'int32' ||
      output.quantization != null || storageRangesOverlap(input, output)) {
    throw new Error(`WebGPU QArgMax node ${node.id} requires exactly { input } -> { out }, immutable per-tensor rank-2..8 I8/U8 input, exact { axis: integer } parameters, and an unquantized I32 output with the axis removed.`);
  }
  if (!Object.isFrozen(input.quantization)) {
    throw new Error(`WebGPU QArgMax node ${node.id} requires immutable per_tensor input quantization metadata.`);
  }
  const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QArgMax node ${node.id} input`);
  return {
    input, output, axis, outer, axisSize, inner, outputElements,
    inputDtype: webGpuTypedDtypeCode(input.dtype), inputQuantization,
  };
}

// Shape-only quantized operators do not reinterpret the stored bytes. They
// nevertheless need a complete descriptor on both sides so a later W8A8
// kernel knows that the copied/max-pooled/resized values still have the same
// mapping to real values. Per-axis metadata is accepted only when the exact
// descriptor remains valid for the destination shape.
function webGpuTypedShapeQuantization(tensor, label) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) || webGpuTypedElementCount(tensor) == null) {
    throw new Error(`${label} requires physically packed I8/U8 tensor storage.`);
  }
  const descriptor = tensor.quantization;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  const validScale = (value) => {
    const rounded = Math.fround(value);
    return typeof value === 'number' && Number.isFinite(value) && value > 0 &&
      Number.isFinite(rounded) && rounded > 0;
  };
  const validZeroPoint = (value) => Number.isInteger(value) && value >= minimum && value <= maximum;
  if (descriptor?.scheme === 'per_tensor') {
    if (!validScale(descriptor.scale) || !validZeroPoint(descriptor.zero_point)) {
      throw new Error(`${label} has an invalid per_tensor quantization descriptor.`);
    }
    return {
      scheme: 'per_tensor',
      scale: Math.fround(descriptor.scale),
      zero_point: descriptor.zero_point,
    };
  }
  if (descriptor?.scheme === 'per_axis') {
    let axis = descriptor.axis;
    if (axis < 0) axis += tensor.shape.length;
    if (!Number.isInteger(axis) || axis < 0 || axis >= tensor.shape.length ||
        !Array.isArray(descriptor.scales) || !Array.isArray(descriptor.zero_points) ||
        descriptor.scales.length !== tensor.shape[axis] || descriptor.zero_points.length !== tensor.shape[axis] ||
        !descriptor.scales.every(validScale) || !descriptor.zero_points.every(validZeroPoint)) {
      throw new Error(`${label} has an invalid per_axis quantization descriptor.`);
    }
    return {
      scheme: 'per_axis',
      axis,
      scales: descriptor.scales.map((value) => Math.fround(value)),
      zero_points: [...descriptor.zero_points],
    };
  }
  throw new Error(`${label} requires per_tensor or per_axis quantization metadata.`);
}

function sameWebGpuQuantization(left, right) {
  if (!left || !right || left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor') {
    return left.scale === right.scale && left.zero_point === right.zero_point;
  }
  return left.axis === right.axis && left.scales.length === right.scales.length &&
    left.zero_points.length === right.zero_points.length &&
    left.scales.every((value, index) => value === right.scales[index]) &&
    left.zero_points.every((value, index) => value === right.zero_points[index]);
}

function webGpuTypedShapePair(node, input, output, operation, { equalElements = false, sameShapeForPerAxis = false } = {}) {
  const inputElements = webGpuTypedElementCount(input);
  const outputElements = webGpuTypedElementCount(output);
  if (inputElements == null || outputElements == null || input.dtype !== output.dtype ||
      (equalElements && inputElements !== outputElements)) {
    throw new Error(`${operation} node ${node.id} requires matching I8/U8 input/output storage.`);
  }
  const inputQuantization = webGpuTypedShapeQuantization(input, `${operation} node ${node.id} input`);
  const outputQuantization = webGpuTypedShapeQuantization(output, `${operation} node ${node.id} output`);
  if (!sameWebGpuQuantization(inputQuantization, outputQuantization)) {
    throw new Error(`${operation} node ${node.id} requires identical input/output quantization descriptors.`);
  }
  if (sameShapeForPerAxis && inputQuantization.scheme === 'per_axis' && !sameTensorShape(input.shape, output.shape)) {
    throw new Error(`${operation} node ${node.id} cannot reshape per_axis quantization metadata without an explicit requantization.`);
  }
  return { inputElements, outputElements, quantization: inputQuantization };
}

function webGpuSliceDescriptor(node, input, output) {
  const inputElements = webGpuTypedElementCount(input);
  const outputElements = webGpuTypedElementCount(output);
  const rank = input?.shape?.length;
  if (!input || !output || !['float32', 'int32'].includes(input.dtype) ||
      output.dtype !== input.dtype ||
      inputElements == null || outputElements == null || !Number.isInteger(rank) ||
      rank < 1 || rank > WEBGPU_SLICE_MAX_RANK || output.shape.length !== rank) {
    throw new Error(`WebGPU Slice node ${node.id} requires rank-1..8 same-dtype F32/I32 input/output tensors.`);
  }
  const startsInput = node.params?.starts ?? [];
  const stepsInput = node.params?.steps ??
    (Array.isArray(startsInput) ? startsInput.map(() => 1) : null);
  const axesInput = node.params?.axes ??
    (Array.isArray(startsInput) ? startsInput.map((_, index) => index) : null);
  if (!Array.isArray(startsInput) || !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
      startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
    throw new Error(`WebGPU Slice node ${node.id} requires matching starts, steps, and axes arrays.`);
  }
  const starts = new Array(rank).fill(0);
  const steps = new Array(rank).fill(1);
  const seen = new Set();
  for (let index = 0; index < axesInput.length; index++) {
    let axis = axesInput[index];
    if (axis < 0) axis += rank;
    let start = startsInput[index];
    const step = stepsInput[index];
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank || seen.has(axis) ||
        !Number.isSafeInteger(start) || !Number.isSafeInteger(step) || step <= 0 || step > 0xffffffff) {
      throw new Error(`WebGPU Slice node ${node.id} requires unique axes, integer starts, and positive integer steps.`);
    }
    if (start < 0) start += input.shape[axis];
    if (start < 0 || start >= input.shape[axis]) {
      throw new Error(`WebGPU Slice node ${node.id} start is outside its input axis.`);
    }
    starts[axis] = start;
    steps[axis] = step;
    seen.add(axis);
  }
  const inputStrides = new Array(rank);
  let stride = 1;
  for (let axis = rank - 1; axis >= 0; axis--) {
    const outputLength = output.shape[axis];
    const remaining = input.shape[axis] - 1 - starts[axis];
    if (outputLength - 1 > Math.floor(remaining / steps[axis])) {
      throw new Error(`WebGPU Slice node ${node.id} output shape exceeds its input selection.`);
    }
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  return { rank, elements: outputElements, starts, steps, inputStrides };
}

/** Operator descriptor construction and model-independent pipeline compilation. */
export class WebGPUGraphCompiler {
  constructor(readonly host: GraphExecutor) {}

  installShaderLibrary(shaderLibrary: ShaderLibraryConstructor | null): void {
    if (shaderLibrary) ShaderLibrary = shaderLibrary;
  }

  async ensureShaderLibrary(): Promise<void> {
    if (!ShaderLibrary) ({ ShaderLibrary } = await import('./ShaderLibrary.js'));
  }

    async _compileIncrementalRowPipelines(
      nodeIndices: Iterable<number> = this.host.incrementalRowCandidates.keys(),
    ): Promise<void> {
      for (const nodeIndex of nodeIndices) {
        if (this.host.incrementalRowPlans.has(nodeIndex)) continue;
        const candidate = this.host.incrementalRowCandidates.get(nodeIndex);
        if (!candidate) continue;
        const buffers = new Map(this.host.gpuBuffers);
        const scratchByName = new Map<string, GPUBuffer>();
        for (const [name, logicalCapacity] of candidate.scratchCapacities) {
          const scratch = this.host.device.createBuffer({
            label: `IncrementalRow_${nodeIndex}_${name}`,
            size: Math.max(4, Math.ceil(logicalCapacity / 4) * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          });
          this.host.auxiliaryBuffers.add(scratch);
          scratchByName.set(name, scratch);
          buffers.set(name, scratch);
        }
        const pipelines: CompiledWebGPUPipeline[] = [];
        await this.host._buildNodePipeline(candidate.sampleNode, { pipelines, buffers });
        if (pipelines.length === 0) continue;
        for (const pipeline of pipelines) pipeline.graphNodeIndex = nodeIndex;
        const byteCopyPipeline = candidate.byteCopyInputs.size || candidate.byteCopyOutputs.size
          ? await this.host._cachedComputePipeline(ShaderLibrary!.getIncrementalRowByteCopyShader())
          : null;
        const makeByteCopies = (
          names: Set<string>,
          input: boolean,
        ): Map<string, IncrementalRowByteCopy> => {
          const copies = new Map<string, IncrementalRowByteCopy>();
          for (const name of names) {
            const full = this.host.gpuBuffers.get(name);
            const scratch = scratchByName.get(name);
            if (!byteCopyPipeline || !full || !scratch) {
              throw new Error(`WebGPU incremental row byte-copy resources for '${name}' are incomplete.`);
            }
            const paramsBuffer = this.host.device.createBuffer({
              label: `IncrementalRow_${nodeIndex}_${name}_${input ? 'input' : 'output'}_copy_params`,
              size: 16,
              usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            this.host.auxiliaryBuffers.add(paramsBuffer);
            copies.set(name, {
              paramsBuffer,
              bindGroup: this.host.device.createBindGroup({
                layout: byteCopyPipeline.getBindGroupLayout(0),
                entries: [
                  { binding: 0, resource: { buffer: input ? full : scratch } },
                  { binding: 1, resource: { buffer: input ? scratch : full } },
                  { binding: 2, resource: { buffer: paramsBuffer } },
                ],
              }),
            });
          }
          return copies;
        };
        const inputByteCopies = makeByteCopies(candidate.byteCopyInputs, true);
        const outputByteCopies = makeByteCopies(candidate.byteCopyOutputs, false);
        this.host.incrementalRowPlans.set(nodeIndex, {
          ...candidate,
          buffers,
          scratchByName,
          pipelines,
          qsdpaParamsBuffer: candidate.node.opType === 'QSDPA'
            ? pipelines.find((pipeline) => pipeline.paramsBuffer)?.paramsBuffer || null
            : null,
          byteCopyPipeline,
          inputByteCopies,
          outputByteCopies,
        });
      }
    }
    /**
     * Allocate VRAM for all tensors and compile shaders.
     */

    async _ensureAdapterPipeline(): Promise<GPUComputePipeline> {
      if (this.host.adapterPipeline) return this.host.adapterPipeline;
      if (!ShaderLibrary) ({ ShaderLibrary } = await import('./ShaderLibrary.js'));
      const module = this.host.device.createShaderModule({ code: ShaderLibrary.getLoRAApplyShader() });
      this.host.adapterPipeline = await this.host.device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      return this.host.adapterPipeline;
    }

    async _cachedComputePipeline(
      code: string,
      {
        entryPoint = 'main',
        constants = null,
      }: {
        entryPoint?: string;
        constants?: Record<string, number | boolean> | null;
      } = {},
    ): Promise<GPUComputePipeline> {
      const constantEntries = constants
        ? Object.entries(constants).sort(([left], [right]) => left.localeCompare(right))
        : [];
      const constantKeyEntries = constantEntries.map(([name, value]) => {
        let encoded;
        if (typeof value === 'boolean') encoded = value ? 'boolean:true' : 'boolean:false';
        else if (typeof value !== 'number') {
          throw new TypeError(`WebGPU pipeline constant '${name}' must be numeric or boolean.`);
        } else if (Number.isNaN(value)) encoded = 'number:NaN';
        else if (Object.is(value, -0)) encoded = 'number:-0';
        else encoded = `number:${String(value)}`;
        return [name, encoded];
      });
      const variantKey = JSON.stringify([entryPoint, constantKeyEntries]);
      let variants = this.host.computePipelineCache.get(code);
      if (!variants) {
        variants = new Map();
        this.host.computePipelineCache.set(code, variants);
      }
      let pending = variants.get(variantKey);
      if (!pending) {
        const module = this.host.device.createShaderModule({ code });
        const compute: GPUProgrammableStage = { module, entryPoint };
        if (constantEntries.length) {
          compute.constants = Object.fromEntries(constantEntries) as Record<string, number>;
        }
        try {
          pending = Promise.resolve(this.host.device.createComputePipelineAsync({ layout: "auto", compute }));
        } catch (error) {
          if (variants.size === 0) this.host.computePipelineCache.delete(code);
          throw error;
        }
        variants.set(variantKey, pending);
      }
      try {
        return await pending;
      } catch (error) {
        // A rejected optional-feature pipeline must not poison a later retry or
        // prevent the caller from compiling its portable fallback.
        if (variants.get(variantKey) === pending) variants.delete(variantKey);
        if (variants.size === 0) this.host.computePipelineCache.delete(code);
        throw error;
      }
    }

    async _buildNodePipeline(node: ExecutorNode, {
      pipelines = this.host.pipelines,
      buffers = this.host.gpuBuffers,
    }: {
      pipelines?: CompiledWebGPUPipeline[];
      buffers?: Map<string, GPUBuffer>;
    } = {}): Promise<void> {
      const previousPipelines = this.host._activePipelineTarget;
      const previousBuffers = this.host._activePipelineBuffers;
      this.host._activePipelineTarget = pipelines;
      this.host._activePipelineBuffers = buffers;
      try {
        return await this.host._buildNodePipelineImpl(node);
      } finally {
        this.host._activePipelineTarget = previousPipelines;
        this.host._activePipelineBuffers = previousBuffers;
      }
    }

    async _buildNodePipelineImpl(node: ExecutorNode): Promise<void> {
      let wgslCode = "";
      let fallbackWgslCode = "";
      let fallbackWorkgroupCount: number[] | null = null;
      let bindGroupEntries: GPUBindGroupEntry[] = [];
      let workgroupCount = [1, 1, 1];
      if (node.opType === "QConv2D") {
        const input = node.inputs.input || node.inputs.x;
        const weight = node.inputs.weight;
        const bias = node.inputs.bias || null;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const weightElements = webGpuTypedElementCount(weight);
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !weight || !output || inputElements == null || weightElements == null ||
            outputElements == null || !['int8', 'uint8'].includes(input.dtype) ||
            !['int8', 'uint8'].includes(weight.dtype) || !['int8', 'uint8'].includes(output.dtype) ||
            input.shape.length !== 4 || weight.shape.length !== 4 || output.shape.length !== 4 ||
            (node.params?.data_layout && node.params.data_layout !== 'NHWC') ||
            (node.params?.weight_layout && node.params.weight_layout !== 'OHWI')) {
          throw new Error(`WebGPU QConv2D node ${node.id} requires canonical NHWC I8/U8 activations and OHWI I8/U8 weights.`);
        }
        const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QConv2D node ${node.id} input`);
        const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QConv2D node ${node.id} output`);
        const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
        const [outputChannels, kernelHeight, kernelWidth, inputPerGroup] = weight.shape;
        const [, outputHeight, outputWidth, outputChannelsFromOutput] = output.shape;
        const groups = node.params?.groups ?? 1;
        const [strideY, strideX] = normalizeSpatialPair(node.params?.stride, 1);
        const [dilationY, dilationX] = normalizeSpatialPair(node.params?.dilation, 1);
        const [paddingY, paddingX] = normalizeSpatialPair(node.params?.padding, 0);
        const pads = node.params?.pads || [paddingY, paddingX, paddingY, paddingX];
        const relu = node.params?.relu ?? 0;
        if (!Number.isInteger(groups) || groups <= 0 || inputChannels !== inputPerGroup * groups ||
            outputChannels !== outputChannelsFromOutput || outputChannels % groups !== 0 ||
            !Array.isArray(pads) || pads.length !== 4 || ![strideY, strideX, dilationY, dilationX, ...pads]
              .every((value) => Number.isInteger(value) && value >= 0) || strideY === 0 || strideX === 0 ||
            dilationY === 0 || dilationX === 0 || !Number.isInteger(relu) || relu < 0 || relu > 2) {
          throw new Error(`WebGPU QConv2D node ${node.id} has incompatible grouped-convolution parameters.`);
        }
        const expectedHeight = Math.floor((inputHeight + pads[0] + pads[2] - dilationY * (kernelHeight - 1) - 1) / strideY) + 1;
        const expectedWidth = Math.floor((inputWidth + pads[1] + pads[3] - dilationX * (kernelWidth - 1) - 1) / strideX) + 1;
        if (expectedHeight !== outputHeight || expectedWidth !== outputWidth ||
            weightElements !== outputChannels * kernelHeight * kernelWidth * inputPerGroup ||
            outputElements !== batch * outputHeight * outputWidth * outputChannels) {
          throw new Error(`WebGPU QConv2D node ${node.id} has an output shape incompatible with its canonical descriptor.`);
        }
        if (bias && (bias.dtype !== 'int32' || !sameTensorShape(bias.shape, [outputChannels]) ||
            !(bias.buffer instanceof Int32Array) || webGpuTypedElementCount(bias) !== outputChannels)) {
          throw new Error(`WebGPU QConv2D node ${node.id} bias must be an I32 vector with one value per output channel.`);
        }
        const weightQuantization = weight.quantization;
        if (weightQuantization?.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
            !Array.isArray(weightQuantization.scales) || !Array.isArray(weightQuantization.zero_points) ||
            weightQuantization.scales.length !== outputChannels || weightQuantization.zero_points.length !== outputChannels) {
          throw new Error(`WebGPU QConv2D node ${node.id} requires per_axis weight quantization along output-channel axis 0.`);
        }
        const [inputMinimum, inputMaximum] = input.dtype === 'int8' ? [-128, 127] : [0, 255];
        const [weightMinimum, weightMaximum] = weight.dtype === 'int8' ? [-128, 127] : [0, 255];
        const inputMagnitude = Math.max(Math.abs(inputMinimum - inputQuantization.zero_point),
          Math.abs(inputMaximum - inputQuantization.zero_point));
        const terms = kernelHeight * kernelWidth * inputPerGroup;
        for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
          const weightZeroPoint = weightQuantization.zero_points[outputChannel];
          const weightMagnitude = Math.max(Math.abs(weightMinimum - weightZeroPoint),
            Math.abs(weightMaximum - weightZeroPoint));
          const maximumAccumulator = inputMagnitude * weightMagnitude * terms +
            (bias ? Math.abs((bias.buffer as Int32Array)[outputChannel]) : 0);
          if (!Number.isSafeInteger(maximumAccumulator) || maximumAccumulator > 0x7fffffff) {
            throw new Error(`WebGPU QConv2D node ${node.id} may overflow its defined I32 accumulator at output channel ${outputChannel}.`);
          }
        }
        const scales = Float32Array.from(weightQuantization.scales, Math.fround);
        if (scales.some((value) => !Number.isFinite(value) || value <= 0)) {
          throw new Error(`WebGPU QConv2D node ${node.id} has a non-representable per-axis weight scale.`);
        }
        const zeroPoints = Int32Array.from(weightQuantization.zero_points);
        const scaleBuffer = this.host._createAuxiliaryStorageBuffer(`QConv2D_scales_${node.id}`, scales);
        const zeroPointBuffer = this.host._createAuxiliaryStorageBuffer(`QConv2D_zero_points_${node.id}`, zeroPoints);
        const biasBuffer = this.host._buffer(bias) || this.host._createAuxiliaryStorageBuffer(
          `QConv2D_bias_${node.id}`, new Int32Array(outputChannels));
        const scalarWorkgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
        const tiledWorkgroupCount = [
          Math.ceil((outputChannels / 4) / 8),
          Math.ceil((batch * outputHeight * outputWidth) / 4),
          1,
        ];
        const advertisedWorkgroupLimit = this.host.device?.limits?.maxComputeWorkgroupsPerDimension;
        const workgroupLimit = Number.isInteger(advertisedWorkgroupLimit) && advertisedWorkgroupLimit > 0
          ? advertisedWorkgroupLimit
          : 65535;
        const tiledDispatchSupported = groups === 1 && outputChannels % 4 === 0 &&
          terms >= 16 && tiledWorkgroupCount.every((count) => count <= workgroupLimit);
        const tiledDotQConv = this.host.hasPackedDot4 && outputChannels >= 16 &&
          tiledDispatchSupported;
        const tiledPortableQConv = !tiledDotQConv && outputChannels >= 32 &&
          tiledDispatchSupported;
        wgslCode = tiledDotQConv
          ? ShaderLibrary.getQConv2DDotTiledShader()
          : tiledPortableQConv
            ? ShaderLibrary.getQConv2DTiledShader()
            : ShaderLibrary.getQConv2DShader();
        if (tiledDotQConv || tiledPortableQConv) {
          fallbackWgslCode = ShaderLibrary.getQConv2DShader();
          fallbackWorkgroupCount = scalarWorkgroupCount;
        }
        const p = new ArrayBuffer(112);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu.set([
          batch, inputHeight, inputWidth, inputChannels,
          outputHeight, outputWidth, outputChannels, kernelHeight,
          kernelWidth, strideY, strideX, dilationY,
          dilationX, pads[0], pads[1], groups,
          webGpuTypedDtypeCode(input.dtype), webGpuTypedDtypeCode(weight.dtype),
          webGpuTypedDtypeCode(output.dtype), relu,
        ]);
        pi[20] = inputQuantization.zero_point;
        pi[21] = outputQuantization.zero_point;
        pf[24] = inputQuantization.scale;
        pf[25] = outputQuantization.scale;
        const paramsBuf = this.host.device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        this.host.auxiliaryBuffers.add(paramsBuf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: scaleBuffer } },
          { binding: 3, resource: { buffer: zeroPointBuffer } },
          { binding: 4, resource: { buffer: biasBuffer } },
          { binding: 5, resource: { buffer: this.host._buffer(output) } },
          { binding: 6, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = tiledDotQConv || tiledPortableQConv
          ? tiledWorkgroupCount
          : scalarWorkgroupCount;
      } else if (node.opType === "Conv2D") {
        wgslCode = ShaderLibrary.getConv2DShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const weightBuf = this.host._buffer(node.inputs.weight);
        const outputBuf = this.host._buffer(node.outputs.out);
        const [n, h, w, c] = node.inputs.input.shape;
        const [kh, kw] = node.inputs.weight.shape;
        const outC = node.outputs.out.shape[3];
        const outH = node.outputs.out.shape[1];
        const outW = node.outputs.out.shape[2];
        const [sy, sx] = normalizeSpatialPair(node.params.stride, 1);
        const [pt, pl] = normalizeSpatialPair(node.params.padding, 0);
        const [dy, dx] = normalizeSpatialPair(node.params.dilation, 1);
        const groups = node.params.groups || 1;
        const pads = Array.isArray(node.params.pads) ? node.params.pads : [pt, pl, pt, pl];
        const noPad = pads.length >= 4 && pads[0] === 0 && pads[1] === 0 && pads[2] === 0 && pads[3] === 0;
        const weightLayout = node.params.weight_layout || (groups === c ? "HWCM" : "HWIO");
        fallbackWgslCode = wgslCode;
        fallbackWorkgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * outC];
        if (groups === 1 && weightLayout === "HWIO" && c === 3 && (outC & 15) === 0) {
          wgslCode = ShaderLibrary.getConv2DRegularC3Out16Shader();
          workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * (outC / 16)];
        } else if (groups === c && weightLayout === "HWCM" && outC === c && (outC & 7) === 0) {
          wgslCode = ShaderLibrary.getConv2DDepthwise8Shader();
          workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
        } else if (groups === 1 && weightLayout === "HWIO" && kh === 1 && kw === 1 &&
                   sy === 1 && sx === 1 && noPad && dy === 1 && dx === 1 &&
                   outH === h && outW === w) {
          if ((outC & 15) === 0) {
            wgslCode = ShaderLibrary.getConv2DPointwise16TileShader();
            workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * (outC / 16)];
          } else if ((outC & 3) === 0) {
            wgslCode = ShaderLibrary.getConv2DPointwise8Vec4Shader();
            workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
          } else if ((outC & 1) === 0) {
            wgslCode = ShaderLibrary.getConv2DPointwise8Vec2Shader();
            workgroupCount = [Math.ceil(outW / 8), Math.ceil(outH / 8), n * Math.ceil(outC / 8)];
          }
        }
        const biasBuf = this.host._buffer(node.inputs.bias) || this.host.device.createBuffer({
          size: Math.max(4, outC * 4),
          usage: GPUBufferUsage.STORAGE,
        });
        const p = new Uint32Array([
          n,
          h,
          w,
          c,
          outC,
          outH,
          outW,
          kh,
          kw,
          sy,
          sx,
          pads[0],
          pads[1],
          groups,
          node.params.relu || 0,
          dy,
          dx
        ]);
        const paramBuf = this.host.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: weightBuf } },
          { binding: 2, resource: { buffer: biasBuf } },
          { binding: 3, resource: { buffer: outputBuf } },
          { binding: 4, resource: { buffer: paramBuf } }
        ];
        if (wgslCode === fallbackWgslCode) workgroupCount = fallbackWorkgroupCount;
      } else if (node.opType === "Conv1D") {
        wgslCode = ShaderLibrary.getConv1DShader();
        const inputShape = node.inputs.input.shape;
        const outputShape = node.outputs.out.shape;
        if (inputShape.length !== 3 || outputShape.length !== 3 ||
            inputShape[0] <= 0 || outputShape[0] !== inputShape[0]) {
          throw new Error(`Conv1D node ${node.id} has incompatible batched NCL dimensions.`);
        }
        const groups = node.params.groups ?? 1;
        const inPerGroup = node.inputs.weight.shape[1];
        if (!Number.isInteger(groups) || groups <= 0 || inputShape[1] % groups || outputShape[1] % groups ||
            inPerGroup !== inputShape[1] / groups) {
          throw new Error(`Conv1D node ${node.id} has incompatible grouped NCL dimensions.`);
        }
        const inputBuf = this.host._buffer(node.inputs.input);
        const weightBuf = this.host._buffer(node.inputs.weight);
        const outputBuf = this.host._buffer(node.outputs.out);
        const biasBuf = this.host._buffer(node.inputs.bias) || this.host.device.createBuffer({
          size: Math.max(4, node.inputs.weight.shape[0] * 4),
          usage: GPUBufferUsage.STORAGE,
        });
        const p = new Uint32Array([
          node.inputs.input.shape[1],
          node.inputs.input.shape[2],
          node.outputs.out.shape[1],
          node.inputs.weight.shape[2],
          normalizeSpatialPair(node.params.stride, 1)[0],
          normalizeSpatialPair(node.params.padding, 0)[0],
          node.params.relu ? 1 : 0,
          inputShape[0],
          groups,
          inPerGroup,
          outputShape[2],
          0
        ]);
        const paramBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: weightBuf } },
          { binding: 2, resource: { buffer: biasBuf } },
          { binding: 3, resource: { buffer: outputBuf } },
          { binding: 4, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.outputs.out.shape[2] / 64), node.outputs.out.shape[1], inputShape[0]];
      } else if (node.opType === "SpatialSoftargmaxY") {
        wgslCode = ShaderLibrary.getSpatialSoftargmaxYShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const outputBuf = this.host._buffer(node.outputs.out);
        if (node.inputs.input.shape.length !== 4 ||
            node.outputs.out.sizeBytes / 4 !== node.inputs.input.shape[0] * node.inputs.input.shape[2] * node.inputs.input.shape[3]) {
          throw new Error(`SpatialSoftargmaxY node ${node.id} has incompatible batched NHWC dimensions.`);
        }
        const p = new Uint32Array([
          node.inputs.input.shape[1],
          node.inputs.input.shape[2],
          node.inputs.input.shape[3],
          node.inputs.input.shape[0]
        ]);
        const paramBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: outputBuf } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], node.inputs.input.shape[0]];
      } else if (node.opType === "UpsampleNearest2D") {
        wgslCode = ShaderLibrary.getUpsample2xShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const outputBuf = this.host._buffer(node.outputs.out);
        const p = new Uint32Array([
          node.inputs.input.shape[0],
          node.inputs.input.shape[1],
          node.inputs.input.shape[2],
          node.inputs.input.shape[3]
        ]);
        const paramBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: outputBuf } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [
          Math.ceil(node.outputs.out.shape[2] / 8),
          Math.ceil(node.outputs.out.shape[1] / 8),
          node.outputs.out.shape[0] * node.outputs.out.shape[3]
        ];
      } else if (node.opType === "Concat" || node.opType === "Concat2") {
        // N-way concat along the declared axis via one strided-copy pipeline per input.
        const output = node.outputs.out;
        const inputs = concatInputEntries(node);
        const hasTypedStorage = [output, ...inputs.map(([, tensor]) => tensor)]
          .some((tensor) => tensor?.dtype === 'int8' || tensor?.dtype === 'uint8');
        if (hasTypedStorage && ![output, ...inputs.map(([, tensor]) => tensor)]
          .every((tensor) => tensor?.dtype === 'int8' || tensor?.dtype === 'uint8')) {
          throw new Error(`Concat node ${node.id} cannot mix F32/I32 and I8/U8 storage.`);
        }
        if (hasTypedStorage && node.params?.sigmoid) {
          throw new Error(`Concat node ${node.id} cannot fuse sigmoid into raw I8/U8 concatenation; insert a dequantization boundary or quantized sigmoid.`);
        }
        const ordinaryDtype = hasTypedStorage ? null : output?.dtype;
        if (!hasTypedStorage && ((ordinaryDtype !== 'float32' && ordinaryDtype !== 'int32') ||
            inputs.some(([, tensor]) => tensor?.dtype !== ordinaryDtype))) {
          throw new Error(`Concat node ${node.id} requires same-dtype F32/I32 inputs and output.`);
        }
        if (ordinaryDtype === 'int32' && node.params?.sigmoid) {
          throw new Error(`Concat node ${node.id} cannot fuse sigmoid into I32 storage.`);
        }
        const shaderModule = this.host.device.createShaderModule({
          code: hasTypedStorage
            ? ShaderLibrary.getTypedConcatCopyShader()
            : ordinaryDtype === 'int32'
              ? ShaderLibrary.getConcatCopy32Shader()
              : ShaderLibrary.getConcatCopyShader(),
        });
        const pipeline = await this.host.device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: shaderModule, entryPoint: "main" }
        });
        const rank = output.shape.length;
        let axis = node.params?.axis ?? 0;
        if (!Number.isInteger(axis)) throw new Error(`Concat node ${node.id} has a non-integer axis.`);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new Error(`Concat node ${node.id} has an invalid axis.`);
        const inner = output.shape.slice(axis + 1).reduce((count, dim) => count * dim, 1);
        const outputQuantization = hasTypedStorage
          ? webGpuTypedShapeQuantization(output, `Concat node ${node.id} output`)
          : null;
        let axisOffset = 0;
        for (const [, t] of inputs) {
          if (t.shape.length !== rank || t.shape.some((dim, index) => index !== axis && dim !== output.shape[index])) {
            throw new Error(`Concat node ${node.id} has incompatible input shapes.`);
          }
          const size = webGpuTypedElementCount(t);
          if (size == null) throw new Error(`Concat node ${node.id} has invalid typed input storage.`);
          if (hasTypedStorage) {
            const quantization = webGpuTypedShapeQuantization(t, `Concat node ${node.id} input`);
            if (!sameWebGpuQuantization(quantization, outputQuantization)) {
              throw new Error(`Concat node ${node.id} requires identical quantization descriptors for raw I8/U8 concat.`);
            }
          }
          const inputAxis = t.shape[axis];
          const p = new Uint32Array([
            size, axisOffset, inputAxis, output.shape[axis], inner,
            node.params?.sigmoid ? 1 : 0, 0, 0,
          ]);
          const paramsBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.host.device.queue.writeBuffer(paramsBuf, 0, p);
          const bindGroup = this.host.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.host._buffer(t) } },
              { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
              { binding: 2, resource: { buffer: paramsBuf } }
            ]
          });
          (this.host._activePipelineTarget || this.host.pipelines).push({ pipeline, bindGroup, workgroupCount: [Math.ceil(size / 64), 1, 1], nodeName: `${node.id}_concat` });
          axisOffset += inputAxis;
        }
        if (axisOffset !== output.shape[axis]) throw new Error(`Concat node ${node.id} input axes do not match its output.`);
        return;
      } else if (node.opType === "ProfileY") {
        wgslCode = ShaderLibrary.getProfileYShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const outputBuf = this.host._buffer(node.outputs.out);
        if (node.inputs.input.shape.length !== 4 ||
            node.outputs.out.sizeBytes / 4 !== node.inputs.input.shape[0] * 2 * node.inputs.input.shape[1] * node.inputs.input.shape[3]) {
          throw new Error(`ProfileY node ${node.id} has incompatible batched NHWC dimensions.`);
        }
        const p = new Uint32Array([
          node.inputs.input.shape[1], node.inputs.input.shape[2],
          node.inputs.input.shape[3], node.inputs.input.shape[0]
        ]);
        const paramBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: outputBuf } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.inputs.input.shape[1] / 64), node.inputs.input.shape[3], node.inputs.input.shape[0]];
      } else if (node.opType === "ProfileX") {
        wgslCode = ShaderLibrary.getProfileXShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const outputBuf = this.host._buffer(node.outputs.out);
        if (node.inputs.input.shape.length !== 4 ||
            node.outputs.out.sizeBytes / 4 !== node.inputs.input.shape[0] * 2 * node.inputs.input.shape[2] * node.inputs.input.shape[3]) {
          throw new Error(`ProfileX node ${node.id} has incompatible batched NHWC dimensions.`);
        }
        const p = new Uint32Array([
          node.inputs.input.shape[1], node.inputs.input.shape[2],
          node.inputs.input.shape[3], node.inputs.input.shape[0]
        ]);
        const paramBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: outputBuf } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], node.inputs.input.shape[0]];
      } else if (node.opType === "Interpolate1D") {
        wgslCode = ShaderLibrary.getInterp1DShader();
        const inputBuf = this.host._buffer(node.inputs.input);
        const outputBuf = this.host._buffer(node.outputs.out);
        if (node.inputs.input.shape.length !== 3 || node.outputs.out.shape.length !== 3 ||
            node.outputs.out.shape[0] !== node.inputs.input.shape[0] ||
            node.outputs.out.shape[1] !== node.inputs.input.shape[1] ||
            node.outputs.out.shape[2] !== node.params.size) {
          throw new Error(`${node.opType} node ${node.id} requires matching rank-3 NCL tensors.`);
        }
        const p = new Uint32Array([node.inputs.input.shape[1], node.inputs.input.shape[2], node.params.size, node.inputs.input.shape[0]]);
        const paramBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: inputBuf } },
          { binding: 1, resource: { buffer: outputBuf } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.params.size / 64), node.inputs.input.shape[1], node.inputs.input.shape[0]];
      } else if (node.opType === "QLinear" || node.opType === "QMatMul" || node.opType === "QGemm") {
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const weightElements = webGpuTypedElementCount(weight);
        const biasElements = webGpuTypedElementCount(bias);
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !weight || !bias || !output || inputElements == null || weightElements == null ||
            biasElements == null || outputElements == null || !['int8', 'uint8'].includes(input.dtype) ||
            !['int8', 'uint8'].includes(weight.dtype) || !['int8', 'uint8'].includes(output.dtype) ||
            bias.dtype !== 'int32' || weight.shape.length !== 2 || input.shape.length < 1 ||
            output.shape.length !== input.shape.length ||
            input.shape.slice(0, -1).some((dimension, index) => dimension !== output.shape[index])) {
          throw new Error(`WebGPU ${node.opType} node ${node.id} requires typed [...,d_in] input, [d_out,d_in] byte weights, I32 bias, and typed [...,d_out] output.`);
        }
        const dIn = input.shape.at(-1)!;
        const dOut = output.shape.at(-1)!;
        const rows = inputElements / dIn;
        if (!Number.isInteger(rows) || rows <= 0 || weight.shape[0] !== dOut || weight.shape[1] !== dIn ||
            weightElements !== dOut * dIn || outputElements !== rows * dOut || biasElements !== dOut ||
            bias.shape.length !== 1 || bias.shape[0] !== dOut || !(bias.buffer instanceof Int32Array)) {
          throw new Error(`WebGPU ${node.opType} node ${node.id} has incompatible [d_out,d_in] dimensions or I32 bias storage.`);
        }
        const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU ${node.opType} node ${node.id} input`);
        const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU ${node.opType} node ${node.id} output`);
        const weightQuantization = weight.quantization;
        if (weightQuantization?.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
            !Array.isArray(weightQuantization.scales) || !Array.isArray(weightQuantization.zero_points) ||
            weightQuantization.scales.length !== dOut || weightQuantization.zero_points.length !== dOut) {
          throw new Error(`WebGPU ${node.opType} node ${node.id} requires per_axis weight quantization along output-channel axis 0.`);
        }
        const [inputMinimum, inputMaximum] = input.dtype === 'int8' ? [-128, 127] : [0, 255];
        const [weightMinimum, weightMaximum] = weight.dtype === 'int8' ? [-128, 127] : [0, 255];
        const inputMagnitude = Math.max(Math.abs(inputMinimum - inputQuantization.zero_point),
          Math.abs(inputMaximum - inputQuantization.zero_point));
        for (let outputChannel = 0; outputChannel < dOut; outputChannel++) {
          const weightZeroPoint = weightQuantization.zero_points[outputChannel];
          const weightMagnitude = Math.max(Math.abs(weightMinimum - weightZeroPoint),
            Math.abs(weightMaximum - weightZeroPoint));
          const maximumAccumulator = inputMagnitude * weightMagnitude * dIn + Math.abs(bias.buffer[outputChannel]);
          if (!Number.isSafeInteger(maximumAccumulator) || maximumAccumulator > 0x7fffffff) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} may overflow its defined I32 accumulator at output channel ${outputChannel}.`);
          }
        }
        const inputScale = Math.fround(inputQuantization.scale);
        const outputScale = Math.fround(outputQuantization.scale);
        const multipliers = Float32Array.from(weightQuantization.scales, (scale) =>
          Math.fround(
            Math.fround(inputScale * Math.fround(scale)) / outputScale,
          ));
        if (!multipliers.every((multiplier) =>
          Number.isFinite(multiplier) && multiplier > 0)) {
          throw new Error(
            `WebGPU ${node.opType} node ${node.id} requires requantization multipliers representable as positive F32.`,
          );
        }
        const zeroPoints = Int32Array.from(weightQuantization.zero_points);
        const multiplierBuffer = this.host._createAuxiliaryStorageBuffer(
          `QLinear_multipliers_${node.id}`,
          multipliers,
        );
        const zeroPointBuffer = this.host._createAuxiliaryStorageBuffer(`QLinear_zero_points_${node.id}`, zeroPoints);
        const tiledQLinear = rows > 1 && dIn >= 16 && dOut >= 32 && dOut % 4 === 0;
        const portableQLinearShader = tiledQLinear
          ? ShaderLibrary.getQLinearTiledShader()
          : ShaderLibrary.getQLinearShader();
        wgslCode = this.host.hasPackedDot4
          ? (tiledQLinear ? ShaderLibrary.getQLinearDotTiledShader() : ShaderLibrary.getQLinearDotShader())
          : portableQLinearShader;
        const p = new ArrayBuffer(64);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = rows;
        pu[1] = dIn;
        pu[2] = dOut;
        pu[3] = webGpuTypedDtypeCode(input.dtype);
        pu[4] = webGpuTypedDtypeCode(weight.dtype);
        pu[5] = webGpuTypedDtypeCode(output.dtype);
        pi[8] = inputQuantization.zero_point;
        pi[9] = outputQuantization.zero_point;
        pf[12] = inputQuantization.scale;
        pf[13] = outputQuantization.scale;
        const paramsBuf = this.host.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        this.host.auxiliaryBuffers.add(paramsBuf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: multiplierBuffer } },
          { binding: 3, resource: { buffer: zeroPointBuffer } },
          { binding: 4, resource: { buffer: this.host._buffer(bias) } },
          { binding: 5, resource: { buffer: this.host._buffer(output) } },
          { binding: 6, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = tiledQLinear
          ? [Math.ceil((dOut / 4) / 8), Math.ceil(rows / 8), 1]
          : [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
        if (this.host.hasPackedDot4) {
          fallbackWgslCode = portableQLinearShader;
          fallbackWorkgroupCount = [...workgroupCount];
        }
      } else if (node.opType === "QBatchMatMul") {
        const descriptor = qBatchMatMulMetadataDescriptor(node);
        const aBuffer = this.host._buffer(descriptor.a);
        const bBuffer = this.host._buffer(descriptor.b);
        const outputBuffer = this.host._buffer(descriptor.output);
        if (!aBuffer || !bBuffer || !outputBuffer ||
            outputBuffer === aBuffer || outputBuffer === bBuffer) {
          throw new Error(
            `WebGPU QBatchMatMul node ${node.id} requires distinct allocated input/output buffers.`,
          );
        }
        const outputWords = Math.ceil(descriptor.outputElements / 4);
        const workgroups = Math.ceil(outputWords / 64);
        if (workgroups > 65535) {
          throw new Error(
            `WebGPU QBatchMatMul node ${node.id} exceeds the portable dispatch dimension.`,
          );
        }
        const metadata = new Uint32Array(descriptor.batchRank * 3);
        metadata.set(descriptor.outputBatchStrides, 0);
        metadata.set(descriptor.aBatchStrides, descriptor.batchRank);
        metadata.set(descriptor.bBatchStrides, descriptor.batchRank * 2);
        const metadataBuffer = this.host.device.createBuffer({
          size: Math.max(4, metadata.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        if (metadata.byteLength) {
          this.host.device.queue.writeBuffer(metadataBuffer, 0, metadata);
        }
        this.host.auxiliaryBuffers.add(metadataBuffer);
        const params = new ArrayBuffer(64);
        const pu = new Uint32Array(params);
        const pi = new Int32Array(params);
        const pf = new Float32Array(params);
        pu.set([
          descriptor.batchRank,
          descriptor.m,
          descriptor.k,
          descriptor.n,
          descriptor.outputElements,
          webGpuTypedDtypeCode(descriptor.a.dtype),
          webGpuTypedDtypeCode(descriptor.b.dtype),
          webGpuTypedDtypeCode(descriptor.output.dtype),
        ]);
        pi[8] = descriptor.aQuantization.zero_point;
        pi[9] = descriptor.bQuantization.zero_point;
        pi[10] = descriptor.outputQuantization.zero_point;
        pf[12] = descriptor.aScale;
        pf[13] = descriptor.bScale;
        pf[14] = descriptor.outputScale;
        const paramsBuffer = this.host.device.createBuffer({
          size: params.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, params);
        this.host.auxiliaryBuffers.add(paramsBuffer);
        wgslCode = ShaderLibrary.getQBatchMatMulShader();
        bindGroupEntries = [
          { binding: 0, resource: { buffer: aBuffer } },
          { binding: 1, resource: { buffer: bBuffer } },
          { binding: 2, resource: { buffer: outputBuffer } },
          { binding: 3, resource: { buffer: metadataBuffer } },
          { binding: 4, resource: { buffer: paramsBuffer } },
        ];
        workgroupCount = [workgroups, 1, 1];
      } else if (node.opType === "BatchMatMul") {
        const descriptor = batchMatMulDescriptor(node);
        if (descriptor.outputBatchCount > 65535 ||
            Math.ceil(descriptor.m / 8) > 65535 ||
            Math.ceil(descriptor.n / 8) > 65535) {
          throw new Error(`WebGPU BatchMatMul node ${node.id} exceeds portable dispatch dimensions.`);
        }
        const metadata = new Uint32Array(5 + descriptor.batchRank * 3);
        metadata.set([
          descriptor.batchRank, descriptor.m, descriptor.k, descriptor.n,
          descriptor.outputBatchCount,
        ]);
        metadata.set(descriptor.outputBatchStrides, 5);
        metadata.set(descriptor.aBatchStrides, 5 + descriptor.batchRank);
        metadata.set(descriptor.bBatchStrides, 5 + descriptor.batchRank * 2);
        const metadataBuffer = this.host.device.createBuffer({
          size: Math.max(4, metadata.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(metadataBuffer, 0, metadata);
        this.host.auxiliaryBuffers.add(metadataBuffer);
        wgslCode = ShaderLibrary.getBatchMatMulShader();
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(descriptor.a) } },
          { binding: 1, resource: { buffer: this.host._buffer(descriptor.b) } },
          { binding: 2, resource: { buffer: this.host._buffer(descriptor.output) } },
          { binding: 3, resource: { buffer: metadataBuffer } },
        ];
        workgroupCount = [
          Math.ceil(descriptor.n / 8),
          Math.ceil(descriptor.m / 8),
          descriptor.outputBatchCount,
        ];
      } else if (node.opType === "MatMul" || node.opType === "Linear" || node.opType === "Gemm") {
        const input = node.inputs.input || node.inputs.x || node.inputs.a;
        const weight = node.inputs.weight;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const scale = node.inputs.scale || node.inputs.weight_scale;
        const zeroPoint = node.inputs.zero_point || node.inputs.weight_zero_point || null;
        const bias = node.inputs.bias || null;
        if (!input || !weight || !output || input.dtype !== "float32" || output.dtype !== "float32" ||
            input.shape.length < 1 || output.shape.length < 1) {
          throw new Error(`WebGPU ${node.opType} node ${node.id} requires F32 input and output tensors.`);
        }
        const seq_len = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const d_in = input.shape[input.shape.length - 1];
        const d_out = output.shape[output.shape.length - 1];
        if (!Number.isSafeInteger(seq_len) || seq_len <= 0 || !Number.isInteger(d_in) || d_in <= 0 ||
            !Number.isInteger(d_out) || d_out <= 0 || input.sizeBytes !== seq_len * d_in * 4 ||
            output.sizeBytes !== seq_len * d_out * 4) {
          throw new Error(`WebGPU ${node.opType} node ${node.id} has incompatible matrix dimensions.`);
        }
        const dummyBias = this.host.device.createBuffer({
          size: Math.max(4, d_out * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(dummyBias, 0, new Float32Array(Math.max(1, d_out)));
        if (scale) {
          const scaleElements = webGpuTypedElementCount(scale);
          const zeroPointElements = zeroPoint ? webGpuTypedElementCount(zeroPoint) : 0;
          const zeroPointType = zeroPoint ?
            webGpuTypedDtypeCode(zeroPoint.dtype) : DataType.Unspecified;
          if (!['int8', 'uint8'].includes(weight.dtype) ||
              weight.sizeBytes !== d_in * d_out || !sameTensorShape(weight.shape, [d_out, d_in]) ||
              scale.dtype !== 'float32' || (scaleElements !== 1 && scaleElements !== d_out) ||
              (zeroPoint && ((zeroPointType !== DataType.I32 &&
                zeroPointType !== DataType.I8 && zeroPointType !== DataType.U8) ||
                zeroPointElements == null ||
                (zeroPointElements !== 1 && zeroPointElements !== d_out))) ||
              (bias && (bias.dtype !== 'float32' || !sameTensorShape(bias.shape, [d_out])))) {
            throw new Error(`WebGPU quantized ${node.opType} node ${node.id} requires [d_out,d_in] I8/U8 weights, F32 scalar/per-output scale, optional typed zero point, and F32 bias.`);
          }
          const useTiledLinear = seq_len > 1 && d_in >= 16 && d_out >= 16;
          wgslCode = useTiledLinear
            ? ShaderLibrary.getLinearInt8TiledShader()
            : ShaderLibrary.getLinearInt8Shader();
          const dummyZeroPoint = this.host.device.createBuffer({
            size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.host.device.queue.writeBuffer(dummyZeroPoint, 0, new Uint32Array(1));
          const p = new Uint32Array([
            seq_len, d_in, d_out, webGpuTypedDtypeCode(weight.dtype), scaleElements,
            zeroPointType, zeroPointElements, 0,
          ]);
          const paramsBuf = this.host.device.createBuffer({ size: p.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.host.device.queue.writeBuffer(paramsBuf, 0, p);
          bindGroupEntries = [
            { binding: 0, resource: { buffer: this.host._buffer(input) } },
            { binding: 1, resource: { buffer: this.host._buffer(weight) } },
            { binding: 2, resource: { buffer: this.host._buffer(scale) } },
            { binding: 3, resource: { buffer: zeroPoint ? this.host._buffer(zeroPoint) : dummyZeroPoint } },
            { binding: 4, resource: { buffer: bias ? this.host._buffer(bias) : dummyBias } },
            { binding: 5, resource: { buffer: this.host._buffer(output) } },
            { binding: 6, resource: { buffer: paramsBuf } },
          ];
        } else {
          if (weight.dtype !== 'float32' ||
              (bias && (bias.dtype !== 'float32' || !sameTensorShape(bias.shape, [d_out])))) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} requires canonical F32 weight and optional bias.`);
          }
          const useTiledLinear = seq_len > 1 && d_in >= 16 && d_out >= 16;
          wgslCode = useTiledLinear
            ? ShaderLibrary.getLinearF32TiledShader()
            : ShaderLibrary.getLinearF32Shader();
          const p = new Uint32Array([seq_len, d_in, d_out]);
          const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.host.device.queue.writeBuffer(paramsBuf, 0, p);
          bindGroupEntries = [
            { binding: 0, resource: { buffer: this.host._buffer(input) } },
            { binding: 1, resource: { buffer: this.host._buffer(weight) } },
            { binding: 3, resource: { buffer: bias ? this.host._buffer(bias) : dummyBias } },
            { binding: 4, resource: { buffer: this.host._buffer(output) } },
            { binding: 5, resource: { buffer: paramsBuf } },
          ];
        }
        const useTiledLinear = seq_len > 1 && d_in >= 16 && d_out >= 16;
        workgroupCount = useTiledLinear
          ? [Math.ceil(d_out / 16), Math.ceil(seq_len / 16), 1]
          : [Math.ceil(d_out / 64), seq_len, 1];
      } else if (node.opType === "LayerNorm") {
        wgslCode = ShaderLibrary.getLayerNormShader();
        const d_model = node.params.d_model ?? node.inputs.input.shape.at(-1);
        if (d_model !== node.inputs.input.shape.at(-1)) {
          throw new Error(`LayerNorm node ${node.id} d_model must match its last input dimension.`);
        }
        const seq_len = node.inputs.input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const p = new ArrayBuffer(16);
        new Uint32Array(p).set([seq_len, d_model!]);
        new Float32Array(p)[2] = node.params.eps ?? 1e-6;
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        const biasBuf = this.host._buffer(node.inputs.bias) || this.host.device.createBuffer({
          size: Math.max(4, d_model! * 4), usage: GPUBufferUsage.STORAGE,
        });
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.weight) } },
          { binding: 2, resource: { buffer: biasBuf } },
          { binding: 3, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 4, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
      } else if (node.opType === "GroupNorm") {
        wgslCode = ShaderLibrary.getGroupNormShader();
        const input = node.inputs.input;
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        const output = node.outputs.out;
        if (!input || !weight || !bias || !output || input.dtype !== "float32" ||
            weight.dtype !== "float32" || bias.dtype !== "float32" || output.dtype !== "float32") {
          throw new Error(`GroupNorm node ${node.id} requires F32 input, weight, bias, and output tensors.`);
        }
        if (input.shape.length !== 4 || output.shape.length !== 4 ||
            input.shape.some((dimension, index) => output.shape[index] !== dimension)) {
          throw new Error(`GroupNorm node ${node.id} requires matching rank-4 NHWC input and output tensors.`);
        }
        const [batch, height, width, channels] = input.shape;
        const numGroups = node.params?.num_groups;
        const eps = node.params?.eps ?? 1e-5;
        if (!Number.isInteger(numGroups) || numGroups! <= 0 || channels % numGroups! !== 0) {
          throw new Error(`GroupNorm node ${node.id} num_groups must be a positive divisor of ${channels} channels.`);
        }
        if (weight.shape.length !== 1 || weight.shape[0] !== channels ||
            bias.shape.length !== 1 || bias.shape[0] !== channels) {
          throw new Error(`GroupNorm node ${node.id} weight and bias must have shape [${channels}].`);
        }
        if (!Number.isFinite(eps) || eps <= 0) {
          throw new Error(`GroupNorm node ${node.id} eps must be a positive finite number.`);
        }
        const raw = new ArrayBuffer(32);
        const u32 = new Uint32Array(raw);
        const f32 = new Float32Array(raw);
        u32.set([batch, height, width, channels, numGroups!, 1]);
        f32[6] = eps;
        const paramsBuf = this.host.device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuf, 0, raw);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: this.host._buffer(bias) } },
          { binding: 3, resource: { buffer: this.host._buffer(output) } },
          { binding: 4, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(batch * numGroups! / 64), 1, 1];
      } else if (node.opType === "GELU") {
        wgslCode = ShaderLibrary.getGELUShader();
        const num_elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([
          num_elements,
          geluApproximation(node) === "tanh" ? 1 : 0,
          0,
          0,
        ]));
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(num_elements / 64), 1, 1];
      } else if (node.opType === "QEmbedding") {
        const input = node.inputs.input;
        const weight = node.inputs.weight;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const weightElements = webGpuTypedElementCount(weight);
        const outputElements = webGpuTypedElementCount(output);
        const outputCount = Object.values(node.outputs || {}).filter(Boolean).length;
        if (Object.keys(node.inputs || {}).length !== 2 || !input || !weight || !output || outputCount !== 1 ||
            !qEmbeddingIdsArePreflightComplete(this.host.graph, node) || input.shape.length < 1 ||
            inputElements == null || !['int8', 'uint8'].includes(weight.dtype) ||
            !['int8', 'uint8'].includes(output.dtype) || weightElements == null ||
            outputElements == null || weight.shape.length !== 2) {
          throw new Error(`WebGPU QEmbedding node ${node.id} requires preflight-complete I32 IDs, rank-2 I8/U8 weights, and I8/U8 output storage.`);
        }
        const [vocab, hidden] = weight.shape;
        const expectedOutputShape = [...input.shape, hidden];
        if (!Number.isInteger(vocab) || vocab <= 0 || !Number.isInteger(hidden) || hidden <= 0 ||
            !sameTensorShape(output.shape, expectedOutputShape) || weightElements !== vocab * hidden ||
            outputElements !== inputElements * hidden) {
          throw new Error(`WebGPU QEmbedding node ${node.id} has incompatible ID, [vocab,hidden] weight, or output dimensions.`);
        }
        const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QEmbedding node ${node.id} output`);
        const weightQuantization = weight.quantization;
        const [weightMinimum, weightMaximum] = weight.dtype === 'int8' ? [-128, 127] : [0, 255];
        if (weightQuantization?.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
            !Array.isArray(weightQuantization.scales) || !Array.isArray(weightQuantization.zero_points) ||
            weightQuantization.scales.length !== vocab || weightQuantization.zero_points.length !== vocab) {
          throw new Error(`WebGPU QEmbedding node ${node.id} requires per_axis weight quantization along vocabulary axis 0.`);
        }
        const scales = Float32Array.from(weightQuantization.scales, Math.fround);
        if (scales.some((scale) => !Number.isFinite(scale) || scale <= 0) ||
            weightQuantization.zero_points.some((zeroPoint) => !Number.isInteger(zeroPoint) ||
              zeroPoint < weightMinimum || zeroPoint > weightMaximum)) {
          throw new Error(`WebGPU QEmbedding node ${node.id} has a non-representable row quantization descriptor.`);
        }
        const zeroPoints = Int32Array.from(weightQuantization.zero_points);
        const scaleBuffer = this.host._createAuxiliaryStorageBuffer(`QEmbedding_scales_${node.id}`, scales);
        const zeroPointBuffer = this.host._createAuxiliaryStorageBuffer(`QEmbedding_zero_points_${node.id}`, zeroPoints);
        wgslCode = ShaderLibrary.getQEmbeddingShader();
        const p = new ArrayBuffer(32);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = inputElements;
        pu[1] = vocab;
        pu[2] = hidden;
        pu[3] = webGpuTypedDtypeCode(weight.dtype);
        pu[4] = webGpuTypedDtypeCode(output.dtype);
        pi[5] = outputQuantization.zero_point;
        pf[6] = outputQuantization.scale;
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        this.host.auxiliaryBuffers.add(paramsBuf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: scaleBuffer } },
          { binding: 3, resource: { buffer: zeroPointBuffer } },
          { binding: 4, resource: { buffer: this.host._buffer(output) } },
          { binding: 5, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
      } else if (node.opType === "Embedding") {
        wgslCode = ShaderLibrary.getEmbeddingShader();
        if (node.inputs.input.dtype !== "int32" || node.inputs.weight.dtype !== "float32" ||
            node.outputs.out.dtype !== "float32") {
          throw new Error(`Embedding node ${node.id} requires int32 token IDs and F32 weight/output tensors.`);
        }
        const seq_len = node.inputs.input.shape.reduce((a, b) => a * b, 1);
        const d_model = node.outputs.out.shape[node.outputs.out.shape.length - 1];
        const vocabSize = node.inputs.weight.shape[0];
        if (!Number.isInteger(vocabSize) || vocabSize <= 0 ||
            node.inputs.weight.shape.reduce((a, b) => a * b, 1) !== vocabSize * d_model ||
            node.outputs.out.shape.reduce((a, b) => a * b, 1) !== seq_len * d_model) {
          throw new Error(`Embedding node ${node.id} has incompatible token, weight, or output dimensions.`);
        }
        const p = new Uint32Array([seq_len, d_model, vocabSize, 0]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.weight) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
      } else if (node.opType === "MoERouter") {
        wgslCode = ShaderLibrary.getMoERouterShader();
        const input = node.inputs.input || node.inputs.x;
        const weight = node.inputs.weight || node.inputs.router_weight;
        const bias = node.inputs.bias;
        const indices = node.outputs.indices || node.outputs.expert_indices;
        const weights = node.outputs.weights || node.outputs.expert_weights;
        if (!input || !weight || !indices || !weights || input.dtype !== "float32" ||
            weight.dtype !== "float32" || indices.dtype !== "float32" || weights.dtype !== "float32" ||
            input.shape.length < 1 || weight.shape.length !== 2) {
          throw new Error(`MoERouter node ${node.id} requires F32 input/weight/indices/weights and a rank-2 router weight.`);
        }
        const rows = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const dModel = input.shape[input.shape.length - 1];
        const numExperts = node.params.num_experts ?? weight.shape[weight.shape.length - 1];
        const topK = node.params.top_k ?? indices.shape[indices.shape.length - 1];
        const temperature = node.params.temperature ?? 1;
        if (weight.shape[0] !== dModel || !Number.isInteger(numExperts) ||
            numExperts < 1 || numExperts !== weight.shape[1] ||
            (bias && (bias.dtype !== "float32" || bias.sizeBytes / 4 < numExperts))) {
          throw new Error(`MoERouter node ${node.id} has incompatible router dimensions.`);
        }
        if (!Number.isInteger(topK) || topK < 1 || topK > 8 || topK > numExperts ||
            indices.sizeBytes / 4 !== rows * topK || weights.sizeBytes / 4 !== rows * topK) {
          throw new Error(`MoERouter node ${node.id} requires top_k in [1, min(8, num_experts)].`);
        }
        if (!Number.isFinite(temperature) || temperature <= 0) {
          throw new Error(`MoERouter node ${node.id} requires a positive finite temperature.`);
        }
        const p = new ArrayBuffer(32);
        const pu = new Uint32Array(p);
        const pf = new Float32Array(p);
        pu[0] = rows; pu[1] = dModel; pu[2] = numExperts; pu[3] = topK;
        pu[4] = node.params.normalize === false ? 0 : 1;
        pu[5] = bias ? 1 : 0;
        pf[6] = temperature;
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        const dummyBias = this.host.device.createBuffer({ size: Math.max(4, numExperts * 4), usage: GPUBufferUsage.STORAGE });
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: bias ? this.host._buffer(bias) : dummyBias } },
          { binding: 3, resource: { buffer: this.host._buffer(indices) } },
          { binding: 4, resource: { buffer: this.host._buffer(weights) } },
          { binding: 5, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(rows / 64), 1, 1];
      } else if (node.opType === "MoELinear") {
        wgslCode = ShaderLibrary.getMoELinearShader();
        const input = node.inputs.input || node.inputs.x;
        const expertWeight = node.inputs.expert_weight || node.inputs.weight;
        const expertBias = node.inputs.expert_bias || node.inputs.bias;
        const routeIndices = node.inputs.route_indices || node.inputs.indices;
        const routeWeights = node.inputs.route_weights || node.inputs.weights;
        const output = node.outputs.out || Object.values(node.outputs)[0];
        if (!input || !expertWeight || !routeIndices || !routeWeights || !output ||
            input.dtype !== "float32" || expertWeight.dtype !== "float32" ||
            routeIndices.dtype !== "float32" || routeWeights.dtype !== "float32" ||
            output.dtype !== "float32" || expertWeight.shape.length !== 3) {
          throw new Error(`MoELinear node ${node.id} requires F32 routing tensors and rank-3 expert weights.`);
        }
        const rows = input.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const dIn = input.shape[input.shape.length - 1];
        const dOut = output.shape[output.shape.length - 1];
        const numExperts = expertWeight.shape[0];
        const topK = routeIndices.shape[routeIndices.shape.length - 1];
        if (expertWeight.shape[1] !== dIn || expertWeight.shape[2] !== dOut ||
            !Number.isInteger(topK) || topK < 1 || topK > numExperts ||
            routeIndices.sizeBytes / 4 !== rows * topK || routeWeights.sizeBytes / 4 !== rows * topK ||
            output.sizeBytes / 4 !== rows * dOut ||
            (expertBias && (expertBias.dtype !== "float32" || expertBias.sizeBytes / 4 !== numExperts * dOut))) {
          throw new Error(`MoELinear node ${node.id} has incompatible expert or routing dimensions.`);
        }
        const p = new Uint32Array(8);
        p[0] = rows; p[1] = dIn; p[2] = dOut; p[3] = numExperts; p[4] = topK; p[5] = expertBias ? 1 : 0;
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        const dummyBias = this.host.device.createBuffer({ size: Math.max(4, numExperts * dOut * 4), usage: GPUBufferUsage.STORAGE });
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(expertWeight) } },
          { binding: 2, resource: { buffer: expertBias ? this.host._buffer(expertBias) : dummyBias } },
          { binding: 3, resource: { buffer: this.host._buffer(routeIndices) } },
          { binding: 4, resource: { buffer: this.host._buffer(routeWeights) } },
          { binding: 5, resource: { buffer: this.host._buffer(output) } },
          { binding: 6, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(dOut / 64), rows, 1];
      } else if (node.opType === "SDPA") {
        wgslCode = ShaderLibrary.getSDPAShader();
        const qkvShape = node.inputs.qkv.shape;
        const outputShape = node.outputs.out.shape;
        const rank = qkvShape.length;
        const batch = rank === 2 ? 1 : qkvShape[0];
        const outputBatch = outputShape.length === 2 ? 1 : outputShape[0];
        const seq_len = qkvShape[rank - 2];
        const d_model = outputShape[outputShape.length - 1];
        const num_heads = node.params.heads || 8;
        const head_dim = d_model / num_heads;
        const mask = node.inputs.mask;
        const maskMode = attentionMaskMode(mask, batch, seq_len, seq_len, node.id);
        if ((rank !== 2 && rank !== 3) || outputShape.length !== rank || batch !== outputBatch ||
            qkvShape[rank - 1] !== 3 * d_model || outputShape[rank - 2] !== seq_len ||
            !Number.isInteger(num_heads) || num_heads <= 0 || !Number.isInteger(head_dim) ||
            head_dim <= 0 || head_dim > 64) {
          throw new Error(`SDPA node ${node.id} has incompatible batched attention dimensions.`);
        }
        const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(head_dim);
        const p = new ArrayBuffer(32);
        const p_u32 = new Uint32Array(p);
        const p_f32 = new Float32Array(p);
        p_u32[0] = seq_len;
        p_u32[1] = d_model;
        p_u32[2] = num_heads;
        p_u32[3] = head_dim;
        p_u32[4] = batch;
        p_f32[5] = scale;
        p_u32[6] = node.params.causal === false ? 0 : 1;
        p_u32[7] = maskMode;
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.qkv) } },
          { binding: 1, resource: { buffer: this.host._buffer(mask) || this.host._buffer(node.inputs.qkv) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len / 64), num_heads, batch];
      } else if (node.opType === "CrossSDPA") {
        wgslCode = ShaderLibrary.getCrossSDPAShader();
        const qShape = node.inputs.q.shape;
        const kShape = node.inputs.k.shape;
        const vShape = node.inputs.v.shape;
        const outputShape = node.outputs.out.shape;
        const rank = qShape.length;
        const batch = rank === 2 ? 1 : qShape[0];
        const batchOf = (shape) => shape.length === 2 ? 1 : shape[0];
        const seq_len_q = qShape[rank - 2];
        const seq_len_kv = kShape[rank - 2];
        const d_model = qShape[rank - 1];
        const num_heads = node.params.heads || 8;
        const head_dim = d_model / num_heads;
        const mask = node.inputs.mask;
        const maskMode = attentionMaskMode(mask, batch, seq_len_q, seq_len_kv, node.id);
        if ((rank !== 2 && rank !== 3) ||
            [kShape, vShape, outputShape].some((shape) => shape.length !== rank || batchOf(shape) !== batch) ||
            kShape[rank - 1] !== d_model || vShape[rank - 1] !== d_model ||
            outputShape[rank - 1] !== d_model || vShape[rank - 2] !== seq_len_kv ||
            outputShape[rank - 2] !== seq_len_q || !Number.isInteger(num_heads) || num_heads <= 0 ||
            !Number.isInteger(head_dim) || head_dim <= 0 || head_dim > 64) {
          throw new Error(`CrossSDPA node ${node.id} has incompatible batched attention dimensions.`);
        }
        const scale = node.params.scale ?? 1 / Math.sqrt(head_dim);
        const p = new ArrayBuffer(48);
        const p_u32 = new Uint32Array(p);
        const p_f32 = new Float32Array(p);
        p_u32[0] = seq_len_q;
        p_u32[1] = seq_len_kv;
        p_u32[2] = d_model;
        p_u32[3] = num_heads;
        p_u32[4] = head_dim;
        p_u32[5] = batch;
        p_f32[6] = scale;
        p_u32[7] = node.params.causal === true ? 1 : 0;
        p_u32[8] = maskMode;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.q) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.k) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.inputs.v) } },
          { binding: 3, resource: { buffer: this.host._buffer(mask) || this.host._buffer(node.inputs.q) } },
          { binding: 4, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 5, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len_q / 64), num_heads, batch];
      } else if (node.opType === "CrossAttention") {
        const q = node.inputs.q;
        const kv = node.inputs.kv;
        const weight = node.inputs.weight;
        const scale = node.inputs.scale;
        const bias = node.inputs.bias;
        const output = node.outputs.out;
        if (!q || !kv || !weight || !output || q.dtype !== "float32" || kv.dtype !== "float32" ||
            output.dtype !== "float32") {
          throw new Error(`CrossAttention node ${node.id} requires F32 q, kv, and output tensors plus a weight tensor.`);
        }
        const qShape = q.shape;
        const kvShape = kv.shape;
        const outputShape = output.shape;
        const rank = qShape.length;
        const batch = rank === 3 ? qShape[0] : 1;
        const kvBatch = kvShape.length === 3 ? kvShape[0] : 1;
        const outputBatch = outputShape.length === 3 ? outputShape[0] : 1;
        const seq_len_q = qShape[rank - 2];
        const seq_len_kv = kvShape[rank - 2];
        const d_model = outputShape[rank - 1];
        const num_heads = node.params.heads ?? 8;
        const head_dim = d_model / num_heads;
        if ((rank !== 2 && rank !== 3) || kvShape.length !== rank || outputShape.length !== rank ||
            batch <= 0 || batch !== kvBatch || batch !== outputBatch ||
            qShape[rank - 1] !== d_model || kvShape[rank - 1] !== d_model ||
            outputShape[rank - 2] !== seq_len_q || !Number.isInteger(num_heads) || num_heads <= 0 ||
            !Number.isInteger(head_dim) || head_dim <= 0 || head_dim > 64) {
          throw new Error(`CrossAttention node ${node.id} has incompatible batched attention dimensions.`);
        }
        const projectionElements = 3 * d_model * d_model;
        const affineElements = 3 * d_model;
        if (!Number.isSafeInteger(projectionElements) ||
            (weight.dtype !== "float32" && weight.dtype !== "int8") ||
            weight.sizeBytes !== projectionElements * (weight.dtype === "float32" ? 4 : 1)) {
          throw new Error(`CrossAttention node ${node.id} requires a [3*d_model, d_model] F32 or packed I8 projection weight.`);
        }
        for (const [name, tensor] of [["scale", scale], ["bias", bias]] as const) {
          if (tensor && (tensor.dtype !== "float32" || tensor.sizeBytes !== affineElements * 4)) {
            throw new Error(`CrossAttention node ${node.id} ${name} must contain exactly 3*d_model F32 values.`);
          }
        }
        if (weight.dtype === "float32") {
          // This is the same authoritative F32 kernel used by the native GPU
          // backends. Its fixed local arrays deliberately cap d_model at 64.
          if (d_model > 64) {
            throw new Error(`CrossAttention node ${node.id} F32 WebGPU kernel requires d_model <= 64.`);
          }
          wgslCode = ShaderLibrary.getCrossAttentionF32Shader();
        } else {
          // Preserve the existing packed-I8 path. Four signed weights are read
          // from each u32 word, so the input feature width must be word aligned.
          if (d_model % 4 !== 0) {
            throw new Error(`CrossAttention node ${node.id} packed I8 WebGPU kernel requires d_model divisible by 4.`);
          }
          wgslCode = ShaderLibrary.getCrossAttentionShader();
        }
        const scale_factor = 1 / Math.sqrt(head_dim);
        const p = new ArrayBuffer(48);
        const p_u32 = new Uint32Array(p);
        const p_f32 = new Float32Array(p);
        p_u32[0] = seq_len_q;
        p_u32[1] = seq_len_kv;
        p_u32[2] = d_model;
        p_u32[3] = num_heads;
        p_u32[4] = head_dim;
        p_f32[5] = scale_factor;
        p_u32[6] = scale ? 1 : 0;
        p_u32[7] = bias ? 1 : 0;
        p_u32[8] = batch;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        
        const dummyScale = this.host.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
        const dummyBias = this.host.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
        
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(q) } },
          { binding: 1, resource: { buffer: this.host._buffer(kv) } },
          { binding: 2, resource: { buffer: this.host._buffer(weight) } },
          { binding: 3, resource: { buffer: scale ? this.host._buffer(scale) : dummyScale } },
          { binding: 4, resource: { buffer: bias ? this.host._buffer(bias) : dummyBias } },
          { binding: 5, resource: { buffer: this.host._buffer(output) } },
          { binding: 6, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len_q / 64), num_heads, batch];
      } else if (node.opType === "MeanHeight") {
        wgslCode = ShaderLibrary.getMeanHeightShader();
        if (node.inputs.input.shape.length !== 4 ||
            node.outputs.out.sizeBytes / 4 !== node.inputs.input.shape[0] * node.inputs.input.shape[2] * node.inputs.input.shape[3]) {
          throw new Error(`MeanHeight node ${node.id} has incompatible batched NHWC dimensions.`);
        }
        const p = new Uint32Array([
          node.inputs.input.shape[1], node.inputs.input.shape[2],
          node.inputs.input.shape[3], node.inputs.input.shape[0]
        ]);
        const paramBuf = this.host.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = [Math.ceil(node.inputs.input.shape[2] / 64), node.inputs.input.shape[3], node.inputs.input.shape[0]];
      } else if (node.opType === "QGroupNorm") {
        const descriptor = webGpuQGroupNormDescriptor(
          node, this.host._activePipelineBuffers || this.host.gpuBuffers,
        );
        const statsBuffer = this.host.device.createBuffer({
          label: `QGroupNorm_stats_${node.id}`,
          size: descriptor.statsBytes,
          usage: GPUBufferUsage.STORAGE,
        });
        this.host.auxiliaryBuffers.add(statsBuffer);

        const parameterData = new ArrayBuffer(64);
        const parameterU32 = new Uint32Array(parameterData);
        const parameterI32 = new Int32Array(parameterData);
        const parameterF32 = new Float32Array(parameterData);
        parameterU32.set([
          descriptor.batch, descriptor.height, descriptor.width, descriptor.channels,
          descriptor.numGroups, webGpuTypedDtypeCode(descriptor.input.dtype),
          webGpuTypedDtypeCode(descriptor.output.dtype), 0,
        ]);
        parameterI32[8] = descriptor.inputQuantization.zero_point;
        parameterI32[9] = descriptor.outputQuantization.zero_point;
        parameterF32[12] = descriptor.inputQuantization.scale;
        parameterF32[13] = descriptor.outputQuantization.scale;
        parameterF32[14] = descriptor.epsilon;
        const paramsBuffer = this.host.device.createBuffer({
          label: `QGroupNorm_params_${node.id}`,
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, parameterData);
        this.host.auxiliaryBuffers.add(paramsBuffer);

        const statsModule = this.host.device.createShaderModule({ code: ShaderLibrary.getQGroupNormStatsShader() });
        const statsPipeline = await this.host.device.createComputePipelineAsync({
          layout: 'auto', compute: { module: statsModule, entryPoint: 'main' },
        });
        const statsBindGroup = this.host.device.createBindGroup({
          layout: statsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: statsBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });

        const applyModule = this.host.device.createShaderModule({ code: ShaderLibrary.getQGroupNormApplyShader() });
        const applyPipeline = await this.host.device.createComputePipelineAsync({
          layout: 'auto', compute: { module: applyModule, entryPoint: 'main' },
        });
        const applyBindGroup = this.host.device.createBindGroup({
          layout: applyPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: this.host._buffer(descriptor.weight) } },
            { binding: 2, resource: { buffer: this.host._buffer(descriptor.bias) } },
            { binding: 3, resource: { buffer: statsBuffer } },
            { binding: 4, resource: { buffer: this.host._buffer(descriptor.output) } },
            { binding: 5, resource: { buffer: paramsBuffer } },
          ],
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline: statsPipeline,
          bindGroup: statsBindGroup,
          workgroupCount: [descriptor.groupCount, 1, 1],
          nodeName: `${node.id}_qgroupnorm_stats`,
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline: applyPipeline,
          bindGroup: applyBindGroup,
          workgroupCount: [Math.ceil(Math.ceil(descriptor.inputElements / 4) / 64), 1, 1],
          nodeName: `${node.id}_qgroupnorm_apply`,
        });
        return;
      } else if (node.opType === "QLayerNorm") {
        const descriptor = webGpuQLayerNormDescriptor(
          node, this.host._activePipelineBuffers || this.host.gpuBuffers,
        );
        const statsBuffer = this.host.device.createBuffer({
          label: `QLayerNorm_stats_${node.id}`,
          size: descriptor.statsBytes,
          usage: GPUBufferUsage.STORAGE,
        });
        this.host.auxiliaryBuffers.add(statsBuffer);

        const parameterData = new ArrayBuffer(48);
        const parameterU32 = new Uint32Array(parameterData);
        const parameterI32 = new Int32Array(parameterData);
        const parameterF32 = new Float32Array(parameterData);
        parameterU32.set([
          descriptor.rows, descriptor.dModel, webGpuTypedDtypeCode(descriptor.input.dtype),
          webGpuTypedDtypeCode(descriptor.output.dtype),
        ]);
        parameterI32[4] = descriptor.inputQuantization.zero_point;
        parameterI32[5] = descriptor.outputQuantization.zero_point;
        parameterF32[8] = descriptor.inputQuantization.scale;
        parameterF32[9] = descriptor.outputQuantization.scale;
        parameterF32[10] = descriptor.epsilon;
        const paramsBuffer = this.host.device.createBuffer({
          label: `QLayerNorm_params_${node.id}`,
          size: 48,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, parameterData);
        this.host.auxiliaryBuffers.add(paramsBuffer);

        const statsPipeline = await this.host._cachedComputePipeline(
          ShaderLibrary.getQLayerNormStatsShader(),
        );
        const statsBindGroup = this.host.device.createBindGroup({
          layout: statsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: statsBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });

        const applyPipeline = await this.host._cachedComputePipeline(
          ShaderLibrary.getQLayerNormApplyShader(),
        );
        const applyBindGroup = this.host.device.createBindGroup({
          layout: applyPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: this.host._buffer(descriptor.weight) } },
            { binding: 2, resource: { buffer: this.host._buffer(descriptor.bias) } },
            { binding: 3, resource: { buffer: statsBuffer } },
            { binding: 4, resource: { buffer: this.host._buffer(descriptor.output) } },
            { binding: 5, resource: { buffer: paramsBuffer } },
          ],
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline: statsPipeline,
          bindGroup: statsBindGroup,
          workgroupCount: [descriptor.rows, 1, 1],
          nodeName: `${node.id}_qlayernorm_stats`,
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline: applyPipeline,
          bindGroup: applyBindGroup,
          workgroupCount: [Math.ceil(Math.ceil(descriptor.inputElements / 4) / 64), 1, 1],
          nodeName: `${node.id}_qlayernorm_apply`,
        });
        return;
      } else if (node.opType === "QMaskedMean") {
        const descriptor = webGpuQMaskedMeanDescriptor(
          node, this.host._activePipelineBuffers || this.host.gpuBuffers,
        );
        const parameterData = new ArrayBuffer(48);
        const parameterU32 = new Uint32Array(parameterData);
        const parameterI32 = new Int32Array(parameterData);
        const parameterF32 = new Float32Array(parameterData);
        parameterU32.set([
          descriptor.batch, descriptor.sequence, descriptor.width, descriptor.inputDtype,
          descriptor.outputDtype, 0, 0, 0,
        ]);
        parameterI32[8] = descriptor.inputQuantization.zero_point;
        parameterI32[9] = descriptor.outputQuantization.zero_point;
        parameterF32[10] = descriptor.inputScale;
        parameterF32[11] = descriptor.outputScale;
        const paramsBuffer = this.host.device.createBuffer({
          label: `QMaskedMean_params_${node.id}`,
          size: 48,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, parameterData);
        this.host.auxiliaryBuffers.add(paramsBuffer);
        const module = this.host.device.createShaderModule({ code: ShaderLibrary.getQMaskedMeanShader() });
        const pipeline = await this.host.device.createComputePipelineAsync({
          layout: 'auto', compute: { module, entryPoint: 'main' },
        });
        const bindGroup = this.host.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: this.host._buffer(descriptor.mask) } },
            { binding: 2, resource: { buffer: this.host._buffer(descriptor.output) } },
            { binding: 3, resource: { buffer: paramsBuffer } },
          ],
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline,
          bindGroup,
          workgroupCount: [Math.ceil(descriptor.outputWords / 64), 1, 1],
          nodeName: `${node.id}_qmaskedmean`,
        });
        return;
      } else if (node.opType === "QSDPA") {
        const descriptor = webGpuQSDPADescriptor(
          node, this.host._activePipelineBuffers || this.host.gpuBuffers,
        );
        const parameterData = new ArrayBuffer(80);
        const parameterU32 = new Uint32Array(parameterData);
        const parameterI32 = new Int32Array(parameterData);
        const parameterF32 = new Float32Array(parameterData);
        const dtypes = (webGpuTypedDtypeCode(descriptor.q.dtype) |
          (webGpuTypedDtypeCode(descriptor.k.dtype) << 8) |
          (webGpuTypedDtypeCode(descriptor.v.dtype) << 16) |
          (webGpuTypedDtypeCode(descriptor.output.dtype) << 24)) >>> 0;
        parameterU32.set([
          descriptor.seqQ, descriptor.seqKV, descriptor.dModel, descriptor.heads,
          descriptor.batch, descriptor.maskMode, descriptor.causal, dtypes,
        ]);
        parameterI32.set([
          descriptor.qQuantization.zero_point, descriptor.kQuantization.zero_point,
          descriptor.vQuantization.zero_point, descriptor.outputQuantization.zero_point,
        ], 8);
        parameterF32.set([
          descriptor.qScale, descriptor.kScale, descriptor.vScale, descriptor.outputScale,
          descriptor.attentionScale,
        ], 12);
        const paramsBuffer = this.host.device.createBuffer({
          label: `QSDPA_params_${node.id}`,
          size: 80,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, parameterData);
        this.host.auxiliaryBuffers.add(paramsBuffer);
        const dummyMask = descriptor.mask ? null : this.host.device.createBuffer({
          label: `QSDPA_dummy_mask_${node.id}`,
          size: 4,
          usage: GPUBufferUsage.STORAGE,
        });
        if (dummyMask) this.host.auxiliaryBuffers.add(dummyMask);

        const pipeline = await this.host._cachedComputePipeline(ShaderLibrary.getQSDPAShader());
        const bindGroup = this.host.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.q) } },
            { binding: 1, resource: { buffer: this.host._buffer(descriptor.k) } },
            { binding: 2, resource: { buffer: this.host._buffer(descriptor.v) } },
            { binding: 3, resource: { buffer: this.host._buffer(descriptor.mask) || dummyMask } },
            { binding: 4, resource: { buffer: this.host._buffer(descriptor.output) } },
            { binding: 5, resource: { buffer: paramsBuffer } },
          ],
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline,
          bindGroup,
          paramsBuffer,
          workgroupCount: [descriptor.seqQ, descriptor.heads, descriptor.batch],
          nodeName: `${node.id}_qsdpa`,
        });
        return;
      } else if (node.opType === "QArgMax") {
        const descriptor = webGpuQArgMaxDescriptor(
          node, this.host._activePipelineBuffers || this.host.gpuBuffers,
        );
        const paramsBuffer = this.host.device.createBuffer({
          label: `QArgMax_params_${node.id}`,
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([
          descriptor.outer, descriptor.axisSize, descriptor.inner, descriptor.inputDtype,
        ]));
        this.host.auxiliaryBuffers.add(paramsBuffer);
        const pipeline = await this.host._cachedComputePipeline(ShaderLibrary.getQArgMaxShader());
        const bindGroup = this.host.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
            { binding: 1, resource: { buffer: this.host._buffer(descriptor.output) } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });
        (this.host._activePipelineTarget || this.host.pipelines).push({
          pipeline,
          bindGroup,
          workgroupCount: [Math.ceil(descriptor.outputElements / 64), 1, 1],
          nodeName: `${node.id}_qargmax`,
        });
        return;
      } else if (node.opType === "QGELU") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
        const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
        const inputElements = webGpuTypedElementCount(input);
        const outputElements = webGpuTypedElementCount(output);
        const parameterNames = Object.keys(node.params || {});
        const canonicalParameters = parameterNames.length === 0 ||
          (parameterNames.length === 1 && parameterNames[0] === 'approximate' &&
            node.params.approximate === 'none');
        if (!input || !output || input === output ||
            (this.host._buffer(input) && this.host._buffer(input) === this.host._buffer(output)) ||
            inputEntries.length !== 1 || inputEntries[0][1] !== input ||
            outputEntries.length !== 1 || inputElements == null || outputElements == null ||
            inputElements !== outputElements || !sameTensorShape(input.shape, output.shape) ||
            !canonicalParameters) {
          throw new Error(`WebGPU QGELU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and only omitted parameters or approximate='none'.`);
        }
        const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QGELU node ${node.id} input`);
        const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QGELU node ${node.id} output`);
        wgslCode = ShaderLibrary.getQGELUShader();
        const p = new ArrayBuffer(48);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = outputElements;
        pu[1] = webGpuTypedDtypeCode(input.dtype);
        pu[2] = webGpuTypedDtypeCode(output.dtype);
        pi[4] = inputQuantization.zero_point;
        pi[5] = outputQuantization.zero_point;
        pf[8] = inputQuantization.scale;
        pf[9] = outputQuantization.scale;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        this.host.auxiliaryBuffers.add(paramsBuf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
      } else if (node.opType === "QSiLU") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
        const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
        const inputElements = webGpuTypedElementCount(input);
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !output || input === output ||
            (this.host._buffer(input) && this.host._buffer(input) === this.host._buffer(output)) ||
            inputEntries.length !== 1 || inputEntries[0][1] !== input ||
            outputEntries.length !== 1 || inputElements == null || outputElements == null ||
            inputElements !== outputElements || !sameTensorShape(input.shape, output.shape) ||
            Object.keys(node.params || {}).length !== 0) {
          throw new Error(`WebGPU QSiLU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and no parameters.`);
        }
        const inputQuantization = webGpuPerTensorQuantization(input, `WebGPU QSiLU node ${node.id} input`);
        const outputQuantization = webGpuPerTensorQuantization(output, `WebGPU QSiLU node ${node.id} output`);
        wgslCode = ShaderLibrary.getQSiLUShader();
        const p = new ArrayBuffer(48);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = outputElements;
        pu[1] = webGpuTypedDtypeCode(input.dtype);
        pu[2] = webGpuTypedDtypeCode(output.dtype);
        pi[4] = inputQuantization.zero_point;
        pi[5] = outputQuantization.zero_point;
        pf[8] = inputQuantization.scale;
        pf[9] = outputQuantization.scale;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        this.host.auxiliaryBuffers.add(paramsBuf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
      } else if (node.opType === "ReLU" || node.opType === "Sigmoid" || node.opType === "HardSwish"
                 || node.opType === "HardSigmoid" || node.opType === "SiLU"
                 || node.opType === "Tanh" || node.opType === "Reshape" || node.opType === "Squeeze"
                 || node.opType === "Unsqueeze" || node.opType === "Flatten" || node.opType === "Dropout"
                 || node.opType === "Identity") {
        // Elementwise unary ops, plus shape-only ops (Reshape/Squeeze/...) which
        // just copy the data through to the newly-allocated output buffer.
        const input = node.inputs.input;
        const output = node.outputs.out;
        const hasTypedStorage = input?.dtype === 'int8' || input?.dtype === 'uint8' ||
          output?.dtype === 'int8' || output?.dtype === 'uint8';
        const shapeOnly = node.opType === 'Reshape' || node.opType === 'Squeeze' ||
          node.opType === 'Unsqueeze' || node.opType === 'Flatten' || node.opType === 'Dropout' ||
          node.opType === 'Identity';
        let elements;
        if (hasTypedStorage) {
          if (!shapeOnly) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} has no raw I8/U8 implementation; insert an explicit quantized operator or dequantization boundary.`);
          }
          const descriptor = webGpuTypedShapePair(node, input, output, node.opType, {
            equalElements: true,
            sameShapeForPerAxis: true,
          });
          elements = descriptor.outputElements;
          wgslCode = ShaderLibrary.getTypedCopyShader();
        } else if (shapeOnly) {
          const inputElements = webGpuTypedElementCount(input);
          const outputElements = webGpuTypedElementCount(output);
          if (!input || !output || !['float32', 'int32'].includes(input.dtype) ||
              output.dtype !== input.dtype || inputElements == null ||
              outputElements !== inputElements) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} requires equal-size same-dtype F32/I32 tensors.`);
          }
          elements = outputElements;
          wgslCode = ShaderLibrary.getCopy32Shader();
        } else {
          if (!input || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
              webGpuTypedElementCount(input) !== webGpuTypedElementCount(output)) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} requires equal-size F32 tensors.`);
          }
          if (node.opType === "ReLU") wgslCode = ShaderLibrary.getReLUShader();
          else if (node.opType === "Sigmoid") wgslCode = ShaderLibrary.getSigmoidShader();
          else if (node.opType === "HardSwish") wgslCode = ShaderLibrary.getHardSwishShader();
          else if (node.opType === "HardSigmoid") wgslCode = ShaderLibrary.getHardSigmoidShader();
          else if (node.opType === "SiLU") wgslCode = ShaderLibrary.getSiLUShader();
          else if (node.opType === "Tanh") wgslCode = ShaderLibrary.getTanhShader();
          else wgslCode = ShaderLibrary.getCopyShader();
          elements = output.shape.reduce((a, b) => a * b, 1);
        }
        const p = new Uint32Array([elements]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [
          hasTypedStorage ? Math.ceil(Math.ceil(elements / 4) / 64) : Math.ceil(elements / 64),
          1, 1,
        ];
      } else if (node.opType === "QAdd") {
        const a = node.inputs.a || node.inputs.input || node.inputs.x;
        const b = node.inputs.b || node.inputs.y;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const elements = webGpuTypedElementCount(output);
        if (!a || !b || !output || elements == null || webGpuTypedElementCount(a) !== elements ||
            webGpuTypedElementCount(b) !== elements || !sameTensorShape(a.shape, b.shape) ||
            !sameTensorShape(a.shape, output.shape)) {
          throw new Error(`QAdd node ${node.id} requires exact-shape typed input/output tensors.`);
        }
        const qa = webGpuPerTensorQuantization(a, `QAdd node ${node.id} input a`);
        const qb = webGpuPerTensorQuantization(b, `QAdd node ${node.id} input b`);
        const qo = webGpuPerTensorQuantization(output, `QAdd node ${node.id} output`);
        const relu = node.params?.relu ?? 0;
        if (!Number.isInteger(relu) || relu < 0 || relu > 2) {
          throw new Error(`WebGPU QAdd node ${node.id} supports relu values 0 (none), 1 (ReLU), or 2 (ReLU6).`);
        }
        wgslCode = ShaderLibrary.getQAddShader();
        const p = new ArrayBuffer(48);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = elements;
        pu[1] = webGpuTypedDtypeCode(a.dtype);
        pu[2] = webGpuTypedDtypeCode(b.dtype);
        pu[3] = webGpuTypedDtypeCode(output.dtype);
        pi[4] = qa.zero_point;
        pi[5] = qb.zero_point;
        pi[6] = qo.zero_point;
        pf[8] = qa.scale;
        pf[9] = qb.scale;
        pf[10] = qo.scale;
        pu[11] = relu;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(a) } },
          { binding: 1, resource: { buffer: this.host._buffer(b) } },
          { binding: 2, resource: { buffer: this.host._buffer(output) } },
          { binding: 3, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(elements / 4) / 64), 1, 1];
      } else if (node.opType === "Add" || node.opType === "Mul" || node.opType === "Sub" || node.opType === "Div") {
        let binOp = { Add: "out_val = av + bv;", Mul: "out_val = av * bv;",
                      Sub: "out_val = av - bv;", Div: "out_val = av / bv;" }[node.opType];
        if (node.opType === "Add") {
          const relu = node.params?.relu ?? 0;
          if (!Number.isInteger(relu) || relu < 0 || relu > 2) {
            throw new Error(`WebGPU Add node ${node.id} supports relu values 0 (none), 1 (ReLU), or 2 (ReLU6).`);
          }
          if (relu >= 1) binOp += " if (out_val < 0.0) { out_val = 0.0; }";
          if (relu === 2) binOp += " if (out_val > 6.0) { out_val = 6.0; }";
        }
        wgslCode = ShaderLibrary.getBroadcastBinaryShader(binOp);
        const outShape = node.outputs.out.shape;
        const rank = outShape.length;
        const contigStrides = (shape) => {
          const st = new Array(shape.length);
          let s = 1;
          for (let i = shape.length - 1; i >= 0; i--) { st[i] = s; s *= shape[i]; }
          return st;
        };
        const outStrides = contigStrides(outShape);
        // numpy-style broadcast strides: left-pad the operand shape to the output
        // rank, then zero the stride of any dimension that is broadcast (size 1).
        const bcastStrides = (shape) => {
          const padded = new Array(rank).fill(1);
          for (let i = 0; i < shape.length; i++) padded[rank - shape.length + i] = shape[i];
          const st = contigStrides(padded);
          for (let i = 0; i < rank; i++) if (padded[i] === 1 && outShape[i] !== 1) st[i] = 0;
          return st;
        };
        const aStrides = bcastStrides(node.inputs.a.shape);
        const bStrides = bcastStrides(node.inputs.b.shape);
        const total = outShape.reduce((a, b) => a * b, 1);
        const meta = new Uint32Array(2 + rank * 3);
        meta[0] = total; meta[1] = rank;
        for (let d = 0; d < rank; d++) {
          meta[2 + d] = outStrides[d];
          meta[2 + rank + d] = aStrides[d];
          meta[2 + 2 * rank + d] = bStrides[d];
        }
        const metaBuf = this.host.device.createBuffer({ size: Math.ceil(meta.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(metaBuf, 0, meta);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.a) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.b) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 3, resource: { buffer: metaBuf } }
        ];
        workgroupCount = [Math.ceil(total / 64), 1, 1];
      } else if (node.opType === "Equal" || node.opType === "GreaterOrEqual") {
        const descriptor = comparisonDescriptor(node);
        const metadata = new Uint32Array(3 + descriptor.rank * 3);
        metadata[0] = descriptor.elements;
        metadata[1] = descriptor.rank;
        metadata.set(descriptor.outputStrides, 2);
        metadata.set(descriptor.aStrides, 2 + descriptor.rank);
        metadata.set(descriptor.bStrides, 2 + descriptor.rank * 2);
        metadata[2 + descriptor.rank * 3] = descriptor.operation;
        const metadataBuffer = this.host.device.createBuffer({
          size: Math.max(4, metadata.byteLength),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(metadataBuffer, 0, metadata);
        this.host.auxiliaryBuffers.add(metadataBuffer);
        wgslCode = ShaderLibrary.getCompareI32Shader();
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(descriptor.a) } },
          { binding: 1, resource: { buffer: this.host._buffer(descriptor.b) } },
          { binding: 2, resource: { buffer: this.host._buffer(descriptor.output) } },
          { binding: 3, resource: { buffer: metadataBuffer } },
        ];
        workgroupCount = [Math.ceil(descriptor.elements / 64), 1, 1];
      } else if (node.opType === "Not") {
        const descriptor = logicalNotDescriptor(node);
        const paramsBuffer = this.host.device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.host.device.queue.writeBuffer(
          paramsBuffer, 0, new Uint32Array([descriptor.elements, 0, 0, 0]),
        );
        this.host.auxiliaryBuffers.add(paramsBuffer);
        wgslCode = ShaderLibrary.getNotI32Shader();
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(descriptor.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(descriptor.output) } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ];
        workgroupCount = [Math.ceil(descriptor.elements / 64), 1, 1];
      } else if (node.opType === "Transpose") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const outputElements = webGpuTypedElementCount(output);
        const byteTranspose = input && ['int8', 'uint8'].includes(input.dtype);
        if (!input || !output || !['float32', 'int32', 'int8', 'uint8'].includes(input.dtype) ||
            output.dtype !== input.dtype || inputElements == null ||
            outputElements !== inputElements) {
          throw new Error(`WebGPU Transpose node ${node.id} requires equal-size same-dtype F32/I32/I8/U8 tensors.`);
        }
        if (byteTranspose) {
          const inputQuantization = webGpuPerTensorQuantization(
            input, `WebGPU Transpose node ${node.id} input`,
          );
          const outputQuantization = webGpuPerTensorQuantization(
            output, `WebGPU Transpose node ${node.id} output`,
          );
          if (inputQuantization.scale !== outputQuantization.scale ||
              inputQuantization.zero_point !== outputQuantization.zero_point ||
              this.host._buffer(input) === this.host._buffer(output)) {
            throw new Error(`WebGPU Transpose node ${node.id} requires distinct I8/U8 storage with identical per-tensor metadata.`);
          }
        }
        const inShape = input.shape;
        const rank = inShape.length;
        const perm = node.params.perm || [...Array(rank).keys()].reverse();
        if (rank < 1 || rank > 8 || !Array.isArray(perm) || perm.length !== rank ||
            new Set(perm).size !== rank ||
            perm.some((axis) => !Number.isInteger(axis) || axis < 0 || axis >= rank)) {
          throw new Error(`WebGPU Transpose node ${node.id} has an invalid rank or permutation.`);
        }
        const inStrides = new Array(rank);
        { let s = 1; for (let i = rank - 1; i >= 0; i--) { inStrides[i] = s; s *= inShape[i]; } }
        const outShape = perm.map((pp) => inShape[pp]);
        if (!sameTensorShape(output.shape, outShape)) {
          throw new Error(`WebGPU Transpose node ${node.id} output shape does not match its permutation.`);
        }
        const outStrides = new Array(rank);
        { let s = 1; for (let i = rank - 1; i >= 0; i--) { outStrides[i] = s; s *= outShape[i]; } }
        const total = inShape.reduce((a, b) => a * b, 1);
        const meta = new Uint32Array(2 + rank * 2);
        meta[0] = total; meta[1] = rank;
        for (let d = 0; d < rank; d++) { meta[2 + d] = outStrides[d]; meta[2 + rank + d] = inStrides[perm[d]]; }
        const metaBuf = this.host.device.createBuffer({ size: Math.ceil(meta.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(metaBuf, 0, meta);
        this.host.auxiliaryBuffers.add(metaBuf);
        wgslCode = byteTranspose
          ? ShaderLibrary.getTypedTransposeShader()
          : ShaderLibrary.getGeneralTransposeShader();
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: metaBuf } }
        ];
        workgroupCount = [
          Math.ceil((byteTranspose ? Math.ceil(total / 4) : total) / 64), 1, 1,
        ];
      } else if (node.opType === "Softmax" || node.opType === "LogSoftmax") {
        // Normalizes over the last (innermost) axis: rows of `d`, `b` rows total.
        wgslCode = node.opType === "Softmax" ? ShaderLibrary.getSoftmaxShader() : ShaderLibrary.getLogSoftmaxShader();
        const shape = node.inputs.input.shape;
        const d = shape[shape.length - 1];
        const b = shape.reduce((a, x) => a * x, 1) / d;
        if (node.params.axis !== undefined) {
          let ax = node.params.axis; if (ax < 0) ax += shape.length;
          if (ax !== shape.length - 1) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} requires the last axis.`);
          }
        }
        const p = new Uint32Array([b, d]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(b / 64), 1, 1];
      } else if (node.opType === "LeakyReLU") {
        wgslCode = ShaderLibrary.getLeakyReLUShader();
        const elements = node.outputs.out.shape.reduce((a, b) => a * b, 1);
        const alpha = node.params.alpha !== undefined ? node.params.alpha : 0.01;
        const p = new ArrayBuffer(16);
        new Uint32Array(p, 0, 1)[0] = elements;
        new Float32Array(p, 4, 1)[0] = alpha;
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(elements / 64), 1, 1];
      } else if (node.opType === "PReLU") {
        wgslCode = ShaderLibrary.getPReLUShader();
        const input = node.inputs.input || node.inputs.x;
        const weight = node.inputs.slope || node.inputs.weight;
        const output = node.outputs.out || Object.values(node.outputs)[0];
        if (!input || !weight || !output) {
          throw new Error(`PReLU node ${node.id} requires input, slope, and output tensors.`);
        }
        const shape = input.shape;
        const c = shape[shape.length - 1] || 1;
        const elements = shape.reduce((a, b) => a * b, 1);
        const weightLength = weight.sizeBytes / 4;
        if (weight.dtype !== "float32" || (weightLength !== 1 && weightLength !== c)) {
          throw new Error(`PReLU node ${node.id} weight must contain 1 or ${c} F32 values.`);
        }
        const p = new Uint32Array([elements, c, weightLength, 0]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(weight) } },
          { binding: 2, resource: { buffer: this.host._buffer(output) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(elements / 64), 1, 1];
      } else if (node.opType === "RMSNorm") {
        wgslCode = ShaderLibrary.getRMSNormShader();
        const shape = node.inputs.input.shape;
        const d_model = node.params.d_model || shape[shape.length - 1];
        const seq_len = shape.reduce((a, x) => a * x, 1) / d_model;
        const eps = node.params.eps !== undefined ? node.params.eps : 1e-6;
        const p = new ArrayBuffer(16);
        new Uint32Array(p, 0, 2).set([seq_len, d_model]);
        new Float32Array(p, 8, 1)[0] = eps;
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.weight) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(seq_len / 64), 1, 1];
      } else if (node.opType === "GlobalAveragePool") {
        wgslCode = ShaderLibrary.getGlobalAveragePoolShader();
        const inShape = node.inputs.input.shape;
        const B = inShape[0];
        const H = inShape[1] || 1;
        const W = inShape[2] || 1;
        const C = inShape[3] || 1;
        const p = new Uint32Array([B, H, W, C]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(C / 64), B, 1];
      } else if (node.opType === "BatchNorm2D") {
        wgslCode = ShaderLibrary.getBatchNorm2DShader();
        const [bn, hn, wn, cn] = node.inputs.input.shape;
        const eps = node.params.eps !== undefined ? node.params.eps : 1e-5;
        const p = new ArrayBuffer(32);
        new Uint32Array(p, 0, 4).set([bn, cn, hn, wn]);
        new Float32Array(p, 16, 1)[0] = eps;
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(node.inputs.input) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.weight) } },
          { binding: 2, resource: { buffer: this.host._buffer(node.inputs.bias) } },
          { binding: 3, resource: { buffer: this.host._buffer(node.inputs.running_mean) } },
          { binding: 4, resource: { buffer: this.host._buffer(node.inputs.running_var) } },
          { binding: 5, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 6, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil((bn * cn * hn * wn) / 64), 1, 1];
      } else if (node.opType === "Resize" || node.opType === "ResizeNearest2D") {
        const input = node.inputs.input;
        const output = node.outputs.out;
        const hasTypedStorage = input?.dtype === 'int8' || input?.dtype === 'uint8' ||
          output?.dtype === 'int8' || output?.dtype === 'uint8';
        const nearest = node.opType === "ResizeNearest2D" || node.params?.mode === "nearest";
        const [rb, inH, inW, rc] = input.shape;
        const [outB, outH, outW, outC] = output.shape;
        let outputElements = null;
        if (hasTypedStorage) {
          webGpuTypedNearestParameters(node);
          const descriptor = webGpuTypedShapePair(node, input, output, node.opType);
          outputElements = descriptor.outputElements;
          if (input.shape.length !== 4 || output.shape.length !== 4 || rb !== outB || rc !== outC ||
              outputElements !== rb * outH * outW * rc) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} requires compatible rank-4 NHWC I8/U8 tensors.`);
          }
          wgslCode = ShaderLibrary.getTypedResizeNearestShader();
        } else {
          wgslCode = ShaderLibrary.getResizeShader();
        }
        const mode = nearest ? 0 : 1;
        const p = new Uint32Array([rb, inH, inW, rc, outH, outW, mode, 0]);
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = hasTypedStorage
          ? [Math.ceil(Math.ceil(outputElements! / 4) / 64), 1, 1]
          : [Math.ceil(outW / 8), Math.ceil(outH / 8), rb * rc];
      } else if (node.opType === "Split") {
        // Split has multiple outputs; push one strided-copy pipeline per slice.
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        if (!input || !['float32', 'int32'].includes(input.dtype) ||
            webGpuTypedElementCount(input) == null || input.shape.length < 1 ||
            input.shape.length > 8) {
          throw new Error(`WebGPU Split node ${node.id} requires a rank-1..8 F32/I32 input.`);
        }
        const inShape = input.shape;
        let axis = node.params?.axis ?? 0;
        if (axis < 0) axis += inShape.length;
        if (!Number.isInteger(axis) || axis < 0 || axis >= inShape.length) {
          throw new Error(`WebGPU Split node ${node.id} has an invalid axis.`);
        }
        // Preserve the graph declaration order; lexical sorting corrupts
        // numeric-suffixed output sequences once they reach out10.
        const outKeys = Object.keys(node.outputs);
        const numOutputs = outKeys.length;
        if (numOutputs === 0 || inShape[axis] % numOutputs !== 0) {
          throw new Error(`WebGPU Split node ${node.id} requires equal-sized output slices.`);
        }
        const splitSize = inShape[axis] / numOutputs;
        let inner = 1;
        for (let i = axis + 1; i < inShape.length; i++) inner *= inShape[i];
        const shaderModule = this.host.device.createShaderModule({ code: ShaderLibrary.getSplitShader() });
        const pipeline = await this.host.device.createComputePipelineAsync({
          layout: "auto",
          compute: { module: shaderModule, entryPoint: "main" }
        });
        for (let o = 0; o < numOutputs; o++) {
          const outT = node.outputs[outKeys[o]];
          const expectedShape = [...inShape];
          expectedShape[axis] = splitSize;
          const total = webGpuTypedElementCount(outT);
          if (!outT || outT.dtype !== input.dtype || total == null ||
              !sameTensorShape(outT.shape, expectedShape)) {
            throw new Error(`WebGPU Split node ${node.id} has an incompatible ${input.dtype} output.`);
          }
          const p = new Uint32Array([total, inner, splitSize, inShape[axis], o * splitSize]);
          const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.host.device.queue.writeBuffer(paramsBuf, 0, p);
          const bindGroup = this.host.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.host._buffer(input) } },
              { binding: 1, resource: { buffer: this.host._buffer(outT) } },
              { binding: 2, resource: { buffer: paramsBuf } }
            ]
          });
          (this.host._activePipelineTarget || this.host.pipelines).push({
            pipeline, bindGroup,
            workgroupCount: [Math.ceil(total / 64), 1, 1],
            nodeName: `${node.id}_split${o}`
          });
        }
        return;
      } else if (node.opType === "Clip") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const elements = webGpuTypedElementCount(input);
        if (!input || !output || !['float32', 'int32'].includes(input.dtype) ||
            output.dtype !== input.dtype || elements == null ||
            webGpuTypedElementCount(output) !== elements ||
            !sameTensorShape(input.shape, output.shape)) {
          throw new Error(`WebGPU Clip node ${node.id} requires same-shape same-dtype F32/I32 tensors.`);
        }
        const bound = (name, fallback) => {
          const tensor = node.inputs[name];
          if (!tensor) return node.params?.[name] ?? fallback;
          if (tensor.dtype !== input.dtype || webGpuTypedElementCount(tensor) !== 1 ||
              !tensor.buffer) {
            throw new Error(`WebGPU Clip node ${node.id} ${name} must be an immutable scalar with the input dtype.`);
          }
          if (input.dtype === 'int32' && !(tensor.buffer instanceof Int32Array)) {
            throw new Error(`WebGPU Clip node ${node.id} ${name} requires physical I32 storage.`);
          }
          if (input.dtype === 'float32' && !(tensor.buffer instanceof Float32Array)) {
            throw new Error(`WebGPU Clip node ${node.id} ${name} requires physical F32 storage.`);
          }
          return tensor.buffer[0];
        };
        const minVal = bound(
          'min', input.dtype === 'int32' ? -2147483648 : Number.NEGATIVE_INFINITY,
        );
        const maxVal = bound(
          'max', input.dtype === 'int32' ? 2147483647 : Number.POSITIVE_INFINITY,
        );
        if (typeof minVal !== 'number' || typeof maxVal !== 'number' ||
            Number.isNaN(minVal) || Number.isNaN(maxVal) || minVal > maxVal ||
            (input.dtype === 'int32' &&
              (!Number.isInteger(minVal) || !Number.isInteger(maxVal) ||
                minVal < -2147483648 || maxVal > 2147483647))) {
          throw new Error(`WebGPU Clip node ${node.id} has invalid bounds.`);
        }
        wgslCode = ShaderLibrary.getTypedClipShader();
        const p = new ArrayBuffer(48);
        const pu = new Uint32Array(p);
        const pf = new Float32Array(p);
        const pi = new Int32Array(p);
        pu[0] = elements;
        pu[1] = webGpuTypedDtypeCode(input.dtype);
        pf[4] = minVal;
        pf[5] = maxVal;
        pi[8] = input.dtype === 'int32' ? minVal : 0;
        pi[9] = input.dtype === 'int32' ? maxVal : 0;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(elements / 64), 1, 1];
      } else if (node.opType === "MaxPool2D") {
        const input = node.inputs.input;
        const output = node.outputs.out;
        const hasTypedStorage = input?.dtype === 'int8' || input?.dtype === 'uint8' ||
          output?.dtype === 'int8' || output?.dtype === 'uint8';
        const inputShape = input.shape;
        const outputShape = output.shape;
        if (inputShape.length !== 4 || outputShape.length !== 4 || inputShape[0] <= 0 ||
            inputShape[0] !== outputShape[0] || inputShape[3] !== outputShape[3]) {
          throw new Error(`MaxPool2D node ${node.id} requires compatible rank-4 NHWC input/output tensors.`);
        }
        const [ky, kx] = normalizeSpatialPair(node.params?.kernel, 1);
        const [sy, sx] = normalizeSpatialPair(node.params?.stride, 1);
        const [py, px] = normalizeSpatialPair(node.params?.padding, 0);
        const pads = node.params?.pads ?? [py, px, py, px];
        if (![ky, kx, sy, sx].every((value) => Number.isInteger(value) && value > 0) ||
            !Array.isArray(pads) || pads.length !== 4 ||
            !pads.every((value) => Number.isInteger(value) && value >= 0)) {
          throw new Error(`MaxPool2D node ${node.id} has invalid kernel, stride, or padding.`);
        }
        let outputElements = null;
        if (hasTypedStorage) {
          if (!webGpuFalseOrAbsent(node.params?.ceil_mode)) {
            throw new Error(`MaxPool2D node ${node.id} does not support ceil_mode for raw I8/U8 storage.`);
          }
          if (node.params?.kernel == null) {
            throw new Error(`MaxPool2D node ${node.id} requires a kernel for raw I8/U8 storage.`);
          }
          if (node.params?.padding != null &&
              ![py, px].every((value) => Number.isInteger(value) && value >= 0)) {
            throw new Error(`MaxPool2D node ${node.id} requires non-negative padding for raw I8/U8 storage.`);
          }
          const [dilationY, dilationX] = normalizeSpatialPair(node.params?.dilation, 1);
          if (![dilationY, dilationX].every((value) => Number.isInteger(value) && value === 1)) {
            throw new Error(`MaxPool2D node ${node.id} supports raw I8/U8 storage only with unit dilation.`);
          }
          const descriptor = webGpuTypedShapePair(node, input, output, node.opType);
          outputElements = descriptor.outputElements;
          const expectedHeight = Math.floor((inputShape[1] + pads[0] + pads[2] - ky) / sy) + 1;
          const expectedWidth = Math.floor((inputShape[2] + pads[1] + pads[3] - kx) / sx) + 1;
          if (expectedHeight !== outputShape[1] || expectedWidth !== outputShape[2] ||
              outputElements !== outputShape.reduce((product, dimension) => product * dimension, 1)) {
            throw new Error(`MaxPool2D node ${node.id} has invalid I8/U8 output storage.`);
          }
          wgslCode = ShaderLibrary.getTypedMaxPool2DShader();
        } else {
          wgslCode = ShaderLibrary.getMaxPool2DShader();
        }
        const p = new Uint32Array(hasTypedStorage ? 16 : 12);
        p.set([
          inputShape[0], inputShape[1], inputShape[2], inputShape[3],
          outputShape[1], outputShape[2],
          ky, kx, sy, sx, pads[0], pads[1]
        ]);
        if (hasTypedStorage) p[12] = webGpuTypedDtypeCode(input.dtype);
        const paramBuf = this.host.device.createBuffer({ size: Math.ceil(p.byteLength / 16) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramBuf } }
        ];
        workgroupCount = hasTypedStorage
          ? [Math.ceil(Math.ceil(outputElements! / 4) / 64), 1, 1]
          : [Math.ceil(outputShape[2] / 8), Math.ceil(outputShape[1] / 8), outputShape[0] * outputShape[3]];
      } else if (node.opType === "NonMaxSuppression") {
        const boxes = node.inputs.boxes;
        const scores = node.inputs.scores;
        const output = node.outputs.out;
        if (!boxes || !scores || !output || boxes.dtype !== "float32" || scores.dtype !== "float32" ||
            output.dtype !== "float32" || boxes.shape.length !== 3 || boxes.shape[2] !== 4 ||
            scores.shape.length !== 3 || scores.shape[0] !== boxes.shape[0] ||
            scores.shape[2] !== boxes.shape[1] || output.shape.at(-1) !== 3 || output.sizeBytes % 12 !== 0) {
          throw new Error(`NonMaxSuppression node ${node.id} requires boxes [B,S,4], scores [B,C,S], and an F32 [rows,3] output.`);
        }
        const scalar = (name, fallback, integer = false) => {
          const tensor = node.inputs[name];
          if (!tensor) return fallback;
          if ((tensor.dtype !== "float32" && tensor.dtype !== "int32") ||
              !ArrayBuffer.isView(tensor.buffer) || tensor.buffer.length < 1) {
            throw new Error(`NonMaxSuppression node ${node.id} ${name} must be a host-resident scalar F32 or I32 tensor.`);
          }
          const value = Number(tensor.buffer[0]);
          if (!Number.isFinite(value) || (integer && (!Number.isInteger(value) || value < 0))) {
            throw new Error(`NonMaxSuppression node ${node.id} ${name} has an invalid scalar value.`);
          }
          return value;
        };
        const maxOutput = scalar("max_output_boxes_per_class", 0, true);
        const iouThreshold = scalar("iou_threshold", 0.5);
        const scoreThreshold = scalar("score_threshold", 0.0);
        const outputRows = output.sizeBytes / 12;
        const p = new ArrayBuffer(32);
        const p_u32 = new Uint32Array(p);
        const p_f32 = new Float32Array(p);
        p_u32[0] = boxes.shape[0];
        p_u32[1] = boxes.shape[1];
        p_u32[2] = scores.shape[1];
        p_u32[3] = maxOutput;
        p_u32[4] = outputRows;
        p_f32[5] = iouThreshold;
        p_f32[6] = scoreThreshold;
        wgslCode = ShaderLibrary.getNonMaxSuppressionShader();
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(boxes) } },
          { binding: 1, resource: { buffer: this.host._buffer(scores) } },
          { binding: 2, resource: { buffer: this.host._buffer(output) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [1, 1, 1];
      } else if (node.opType === "Cast") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out;
        const inputElements = webGpuTypedElementCount(input);
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !output || inputElements == null || outputElements == null ||
            inputElements !== outputElements) {
          throw new Error(`Cast node ${node.id} requires equal-size F32/I32/I8/U8 input and declared output tensors.`);
        }
        wgslCode = ShaderLibrary.getCastShader();
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([
          outputElements, webGpuTypedDtypeCode(input.dtype), webGpuTypedDtypeCode(output.dtype), 0,
        ]));
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [(output.dtype as string) === "int8" || (output.dtype as string) === "uint8"
          ? Math.ceil(Math.ceil(outputElements / 4) / 64)
          : Math.ceil(outputElements / 64), 1, 1];
      } else if (node.opType === "Where" || node.opType === "Mask") {
        const cond = node.inputs.cond || node.inputs.condition || node.inputs.mask;
        const a = node.inputs.x || node.inputs.a;
        const b = node.inputs.y || node.inputs.b;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const conditionElements = webGpuTypedElementCount(cond);
        const aElements = webGpuTypedElementCount(a);
        const bElements = webGpuTypedElementCount(b);
        const outputElements = webGpuTypedElementCount(output);
        if (!cond || !a || !b || !output ||
            (cond.dtype !== 'float32' && cond.dtype !== 'int32') ||
            (a.dtype !== 'float32' && a.dtype !== 'int32') ||
            b.dtype !== a.dtype || output.dtype !== a.dtype ||
            conditionElements == null || aElements == null || bElements == null || outputElements == null ||
            conditionElements !== outputElements || aElements !== outputElements || bElements !== outputElements ||
            !sameTensorShape(cond.shape, output.shape) || !sameTensorShape(a.shape, output.shape) ||
            !sameTensorShape(b.shape, output.shape)) {
          throw new Error(`${node.opType} node ${node.id} requires exact-shape same-dtype F32/I32 operands/output and an F32 or I32 condition.`);
        }
        wgslCode = ShaderLibrary.getWhereTypedShader();
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([
          outputElements, webGpuTypedDtypeCode(cond.dtype), 0, 0,
        ]));
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(cond) } },
          { binding: 1, resource: { buffer: this.host._buffer(a) } },
          { binding: 2, resource: { buffer: this.host._buffer(b) } },
          { binding: 3, resource: { buffer: this.host._buffer(output) } },
          { binding: 4, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(outputElements / 64), 1, 1];
      } else if (node.opType === "DequantizeLinear") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const scale = node.inputs.scale;
        const zeroPoint = node.inputs.zero_point || null;
        const output = node.outputs.out;
        const inputElements = webGpuTypedElementCount(input);
        const scaleElements = webGpuTypedElementCount(scale);
        const zeroPointElements = zeroPoint ? webGpuTypedElementCount(zeroPoint) : 1;
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !scale || !output || inputElements == null || outputElements == null ||
            scale.dtype !== 'float32' || output.dtype !== 'float32' || scaleElements !== 1 ||
            zeroPointElements !== 1 || inputElements !== outputElements) {
          throw new Error(`DequantizeLinear node ${node.id} requires an F32 output, scalar F32 scale, optional scalar typed zero_point, and matching F32/I32/I8/U8 input elements.`);
        }
        const hasZeroPoint = zeroPoint ? 1 : 0;
        wgslCode = ShaderLibrary.getDequantizeLinearShader();
        const p = new Uint32Array([
          outputElements,
          webGpuTypedDtypeCode(input.dtype),
          webGpuTypedDtypeCode(scale.dtype),
          zeroPoint ? webGpuTypedDtypeCode(zeroPoint.dtype) :
            DataType.Unspecified,
          webGpuTypedDtypeCode(output.dtype),
          hasZeroPoint,
          0,
          0,
        ]);
        const paramsBuf = this.host.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(scale) } },
          // The shader does not read this binding when no zero point is present,
          // so reuse the input buffer instead of allocating an untracked dummy.
          { binding: 2, resource: { buffer: zeroPoint ? this.host._buffer(zeroPoint) : this.host._buffer(input) } },
          { binding: 3, resource: { buffer: this.host._buffer(output) } },
          { binding: 4, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(outputElements / 64), 1, 1];
      } else if (node.opType === "QuantizeLinear") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const scale = node.inputs.scale;
        const zeroPoint = node.inputs.zero_point || null;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const scaleElements = webGpuTypedElementCount(scale);
        const zeroPointElements = zeroPoint ? webGpuTypedElementCount(zeroPoint) : 1;
        const outputElements = webGpuTypedElementCount(output);
        if (!input || !scale || !output || input.dtype !== 'float32' || inputElements == null ||
            !['int8', 'uint8'].includes(output.dtype) || outputElements !== inputElements ||
            scale.dtype !== 'float32' || scaleElements !== 1 ||
            output.quantization?.scheme !== 'per_tensor' ||
            (zeroPoint && (zeroPoint.dtype !== output.dtype || zeroPointElements !== 1))) {
          throw new Error(`QuantizeLinear node ${node.id} requires F32 input, scalar F32 scale, optional matching I8/U8 zero_point, and matching I8/U8 output with per_tensor quantization metadata.`);
        }
        if (scale.buffer && Math.fround(output.quantization.scale) !== Math.fround(scale.buffer[0]) ||
            zeroPoint?.buffer && output.quantization.zero_point !== zeroPoint.buffer[0]) {
          throw new Error(`QuantizeLinear node ${node.id} scale/zero_point inputs do not match its output quantization metadata.`);
        }
        wgslCode = ShaderLibrary.getQuantizeLinearShader();
        const p = new Uint32Array([
          outputElements,
          webGpuTypedDtypeCode(output.dtype),
          zeroPoint ? webGpuTypedDtypeCode(zeroPoint.dtype) :
            DataType.Unspecified,
          zeroPoint ? 1 : 0,
        ]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(scale) } },
          // The shader does not read zero_point when it is omitted, so reuse the
          // input buffer instead of allocating a temporary resource.
          { binding: 2, resource: { buffer: zeroPoint ? this.host._buffer(zeroPoint) : this.host._buffer(input) } },
          { binding: 3, resource: { buffer: this.host._buffer(output) } },
          { binding: 4, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
      } else if (node.opType === "RequantizeLinear") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const inputElements = webGpuTypedElementCount(input);
        const outputElements = webGpuTypedElementCount(output);
        const tensorInputs = Object.values(node.inputs || {}).filter(Boolean);
        if (!input || !output || inputElements == null || outputElements == null ||
            inputElements !== outputElements || tensorInputs.length !== 1 || tensorInputs[0] !== input) {
          throw new Error(`RequantizeLinear node ${node.id} requires exactly one equal-size typed input and one typed output; scales belong only to immutable tensor metadata.`);
        }
        const inputQuantization = webGpuPerTensorQuantization(input, `RequantizeLinear node ${node.id} input`);
        const outputQuantization = webGpuPerTensorQuantization(output, `RequantizeLinear node ${node.id} output`);
        const multiplier = Math.fround(inputQuantization.scale / outputQuantization.scale);
        if (!Number.isFinite(multiplier) || multiplier <= 0) {
          throw new Error(`RequantizeLinear node ${node.id} has a non-representable positive scale ratio.`);
        }
        wgslCode = ShaderLibrary.getRequantizeLinearShader();
        const p = new ArrayBuffer(48);
        const pu = new Uint32Array(p);
        const pi = new Int32Array(p);
        const pf = new Float32Array(p);
        pu[0] = outputElements;
        pu[1] = webGpuTypedDtypeCode(input.dtype);
        pu[2] = webGpuTypedDtypeCode(output.dtype);
        pi[4] = inputQuantization.zero_point;
        pi[5] = outputQuantization.zero_point;
        pf[8] = multiplier;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } },
        ];
        workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
      } else if (node.opType === "Expand" || node.opType === "Broadcast") {
        const inp = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out || Object.values(node.outputs || {})[0];
        const outShape = output?.shape;
        const byteStorage = inp && output && ['int8', 'uint8'].includes(inp.dtype) &&
          output.dtype === inp.dtype;
        const plainStorage = inp && output && ['float32', 'int32'].includes(inp.dtype) &&
          output.dtype === inp.dtype;
        if (!inp || !output || (!plainStorage && !byteStorage) ||
            webGpuTypedElementCount(inp) == null ||
            webGpuTypedElementCount(output) == null ||
            inp.shape.length === 0 || inp.shape.length > outShape.length || outShape.length > 8 ||
            inp.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0) ||
            outShape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0) ||
            inp === output || Object.keys(node.params || {}).length !== 0) {
          throw new Error(`WebGPU ${node.opType} requires positive rank-1..8 input/output shapes.`);
        }
        if (byteStorage) {
          const inputQuantization = webGpuPerTensorQuantization(
            inp, `WebGPU ${node.opType} node ${node.id} input`,
          );
          const outputQuantization = webGpuPerTensorQuantization(
            output, `WebGPU ${node.opType} node ${node.id} output`,
          );
          if (inputQuantization.scale !== outputQuantization.scale ||
              inputQuantization.zero_point !== outputQuantization.zero_point) {
            throw new Error(`WebGPU ${node.opType} node ${node.id} must preserve its I8/U8 affine descriptor.`);
          }
        }
        const offset = outShape.length - inp.shape.length;
        if (outShape.some((dimension, index) => {
          const source = index < offset ? 1 : inp.shape[index - offset];
          return source !== 1 && source !== dimension;
        })) throw new Error(`WebGPU ${node.opType} has incompatible broadcast shapes.`);
        const elements = outShape.reduce((product, dimension) => product * dimension, 1);
        wgslCode = byteStorage
          ? ShaderLibrary.getTypedExpandShader()
          : ShaderLibrary.getExpandShader();
        const p = new Uint32Array(20);
        p.set([inp.shape.length, outShape.length, 0, elements]);
        p.set(inp.shape, 4);
        p.set(outShape, 12);
        const paramsBuf = this.host.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [
          Math.ceil((byteStorage ? Math.ceil(elements / 4) : elements) / 64),
          1,
          1,
        ];
      } else if (node.opType === "Pad") {
        wgslCode = ShaderLibrary.getPadShader();
        const inp = node.inputs.input || node.inputs.data;
        const pads = node.params.pads || [];
        const pt = pads.length === 8 ? pads[1] : (pads[0] || 0);
        const pl = pads.length === 8 ? pads[2] : (pads[1] || 0);
        const val = node.params.value || 0.0;
        const is = [1, 1, 1, 1].slice(0, 4 - inp.shape.length).concat(inp.shape);
        const os = [1, 1, 1, 1].slice(0, 4 - node.outputs.out.shape.length).concat(node.outputs.out.shape);
        const buf = new ArrayBuffer(48);
        const u = new Uint32Array(buf); const f = new Float32Array(buf);
        u[0] = is[0]; u[1] = is[1]; u[2] = is[2]; u[3] = is[3]; u[4] = os[1]; u[5] = os[2]; u[6] = pt; u[7] = pl; f[8] = val;
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, buf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil((os[0] * os[1] * os[2] * os[3]) / 64), 1, 1];
      } else if (node.opType === "Slice") {
        const input = node.inputs.input || node.inputs.x || node.inputs.data;
        const output = node.outputs.out;
        const descriptor = webGpuSliceDescriptor(node, input, output);
        const p = new Uint32Array(36);
        p.set([descriptor.rank, descriptor.elements, 0, 0]);
        p.set([...output.shape, ...new Array(WEBGPU_SLICE_MAX_RANK - descriptor.rank).fill(1)], 4);
        p.set([...descriptor.starts, ...new Array(WEBGPU_SLICE_MAX_RANK - descriptor.rank).fill(0)], 12);
        p.set([...descriptor.steps, ...new Array(WEBGPU_SLICE_MAX_RANK - descriptor.rank).fill(1)], 20);
        p.set([...descriptor.inputStrides, ...new Array(WEBGPU_SLICE_MAX_RANK - descriptor.rank).fill(1)], 28);
        wgslCode = ShaderLibrary.getSliceNdShader();
        const paramsBuf = this.host.device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(input) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(descriptor.elements / 64), 1, 1];
      } else if (node.opType === "Gather") {
        const inp = node.inputs.input || node.inputs.data;
        const indices = node.inputs.indices;
        const output = node.outputs.out;
        if (!inp || !indices || !output || inp.dtype !== "float32" || output.dtype !== "float32") {
          throw new Error(`Gather node ${node.id} requires F32 input/output tensors and an indices tensor.`);
        }
        const dataRank = inp.shape.length;
        const indicesRank = indices.shape.length;
        let axis = node.params.axis ?? 0;
        if (!Number.isInteger(axis)) throw new Error(`Gather node ${node.id} has a non-integer axis.`);
        if (axis < 0) axis += dataRank;
        if (dataRank < 1 || dataRank > 8 || indicesRank > 8 || axis < 0 || axis >= dataRank) {
          throw new Error(`Gather node ${node.id} requires rank-1..8 data, rank-0..8 indices, and a valid axis.`);
        }
        const expectedShape = [
          ...inp.shape.slice(0, axis),
          ...indices.shape,
          ...inp.shape.slice(axis + 1),
        ];
        if (output.shape.length !== expectedShape.length ||
            output.shape.some((dimension, index) => dimension !== expectedShape[index]) ||
            expectedShape.length > 8) {
          throw new Error(`Gather node ${node.id} output shape must be input[:axis] + indices.shape + input[axis + 1:].`);
        }
        const total = output.shape.reduce((product, dimension) => product * dimension, 1);
        if (!Number.isSafeInteger(total) || total <= 0) {
          throw new Error(`Gather node ${node.id} has an invalid output element count.`);
        }
        if (indices.dtype === "int32") {
          // Canonical portable path: I32 indices and arbitrary valid axis.
          wgslCode = ShaderLibrary.getGatherInt32Shader();
          const p = new Uint32Array(20);
          p.set([dataRank, indicesRank, axis, total]);
          p.set(inp.shape, 4);
          p.set(output.shape, 12);
          const paramsBuf = this.host.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          this.host.device.queue.writeBuffer(paramsBuf, 0, p);
          bindGroupEntries = [
            { binding: 0, resource: { buffer: this.host._buffer(inp) } },
            { binding: 1, resource: { buffer: this.host._buffer(indices) } },
            { binding: 2, resource: { buffer: this.host._buffer(output) } },
            { binding: 3, resource: { buffer: paramsBuf } }
          ];
        } else {
          throw new Error(`Gather node ${node.id} requires I32 indices.`);
        }
        workgroupCount = [Math.ceil(total / 64), 1, 1];
      } else if (node.opType === "GatherElements") {
        const inp = node.inputs.input || node.inputs.data;
        const indices = node.inputs.indices;
        const output = node.outputs.out;
        if (!inp || !indices || !output || inp.dtype !== "float32" || output.dtype !== "float32" ||
            indices.dtype !== "int32") {
          throw new Error(`GatherElements node ${node.id} requires F32 input/output tensors and I32 indices.`);
        }
        const rank = inp.shape.length;
        let axis = node.params.axis ?? 0;
        if (!Number.isInteger(axis)) throw new Error(`GatherElements node ${node.id} has a non-integer axis.`);
        if (axis < 0) axis += rank;
        if (rank < 1 || rank > 8 || indices.shape.length !== rank || output.shape.length !== rank ||
            axis < 0 || axis >= rank || output.shape.some((dimension, index) => dimension !== indices.shape[index]) ||
            indices.shape.some((dimension, index) => index !== axis && dimension > inp.shape[index])) {
          throw new Error(`GatherElements node ${node.id} requires matching rank-1..8 index/output shapes and a valid axis.`);
        }
        const total = output.shape.reduce((product, dimension) => product * dimension, 1);
        if (!Number.isSafeInteger(total) || total <= 0) {
          throw new Error(`GatherElements node ${node.id} has an invalid output element count.`);
        }
        wgslCode = ShaderLibrary.getGatherElementsShader();
        const p = new Uint32Array(20);
        p.set([rank, axis, total, 0]);
        p.set(inp.shape, 4);
        p.set(indices.shape, 12);
        const paramsBuf = this.host.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(indices) } },
          { binding: 2, resource: { buffer: this.host._buffer(output) } },
          { binding: 3, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(total / 64), 1, 1];
      } else if (node.opType === "ArgMax") {
        const inp = node.inputs.input || node.inputs.data;
        const output = node.outputs.out;
        if (!inp || !output || inp.dtype !== "float32") {
          throw new Error(`ArgMax node ${node.id} requires an F32 input and a declared output tensor.`);
        }
        const rank = inp.shape.length;
        let axis = node.params.axis ?? 0;
        if (!Number.isInteger(axis)) throw new Error(`ArgMax node ${node.id} has a non-integer axis.`);
        if (axis < 0) axis += rank;
        if (rank < 1 || axis < 0 || axis >= rank) {
          throw new Error(`ArgMax node ${node.id} has an axis outside input rank ${rank}.`);
        }
        const outer = inp.shape.slice(0, axis).reduce((product, dimension) => product * dimension, 1);
        const axisSize = inp.shape[axis];
        const inner = inp.shape.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
        const outputElements = outer * inner;
        const removedAxisShape = [...inp.shape.slice(0, axis), ...inp.shape.slice(axis + 1)];
        const keptAxisShape = inp.shape.map((dimension, index) => index === axis ? 1 : dimension);
        const sameShape = (left, right) => left.length === right.length &&
          left.every((dimension, index) => dimension === right[index]);
        if (!sameShape(output.shape, removedAxisShape) && !sameShape(output.shape, keptAxisShape)) {
          throw new Error(`ArgMax node ${node.id} output must remove its axis or retain it with dimension 1.`);
        }
        const elementBytes = output.dtype === "float32" || output.dtype === "int32" ? 4
          : output.dtype === "int8" || output.dtype === "uint8" ? 1 : 0;
        if (![outer, axisSize, inner, outputElements].every(Number.isSafeInteger) ||
            outer <= 0 || axisSize <= 0 || inner <= 0 || outputElements <= 0 ||
            axisSize > 0x7fffffff || output.sizeBytes !== outputElements * elementBytes) {
          throw new Error(`ArgMax node ${node.id} has incompatible input/output dimensions or dtype '${output.dtype}'.`);
        }
        const p = new Uint32Array([outer, axisSize, inner, outputElements]);
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        if (output.dtype === "float32") {
          wgslCode = ShaderLibrary.getArgMaxF32Shader();
          workgroupCount = [Math.ceil(outputElements / 64), 1, 1];
        } else if (output.dtype === "int32") {
          wgslCode = ShaderLibrary.getArgMaxI32Shader();
          workgroupCount = [Math.ceil(outputElements / 64), 1, 1];
        } else if (output.dtype === "int8" || output.dtype === "uint8") {
          // Byte outputs share one packed-u32 shader. The low byte reproduces
          // both Int8Array and Uint8Array assignment/wrapping semantics.
          wgslCode = ShaderLibrary.getArgMaxI8Shader();
          workgroupCount = [Math.ceil(Math.ceil(outputElements / 4) / 64), 1, 1];
        } else {
          throw new Error(`ArgMax node ${node.id} does not support output dtype '${output.dtype}'.`);
        }
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(output) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
      } else if (node.opType === "ReduceSum" || node.opType === "ReduceMean") {
        wgslCode = ShaderLibrary.getReduceShader();
        const inp = node.inputs.input || node.inputs.data;
        const d = inp.shape.at(-1);
        const b = inp.shape.slice(0, -1).reduce((count, dimension) => count * dimension, 1);
        if (typeof d !== 'number' || !Number.isSafeInteger(d) || d <= 0 ||
            node.outputs.out.sizeBytes / 4 !== b) {
          throw new Error(`${node.opType} node ${node.id} must reduce the last axis into one value per outer row.`);
        }
        const inv = node.opType === "ReduceMean" ? 1.0 / d : 1.0;
        const buf = new ArrayBuffer(16);
        new Uint32Array(buf, 0, 2).set([b, d]); new Float32Array(buf, 8, 1)[0] = inv;
        const paramsBuf = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, buf);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(b / 64), 1, 1];
      } else if (node.opType === "AveragePool2D") {
        wgslCode = ShaderLibrary.getAveragePool2DShader();
        const inp = node.inputs.input || node.inputs.x;
        const [b, in_h, in_w, c] = inp.shape;
        const [, out_h, out_w] = node.outputs.out.shape;
        const [kh, kw] = normalizeSpatialPair(node.params.kernel, 1);
        const [sh, sw] = normalizeSpatialPair(node.params.stride, 1);
        const ph = node.params.padding ? node.params.padding[0] : 0;
        const pw = node.params.padding ? node.params.padding[1] : 0;
        const p = new Uint32Array([b, in_h, in_w, c, out_h, out_w, kh, kw, sh, sw, ph, pw]);
        const paramsBuf = this.host.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 2, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(out_w / 8), Math.ceil(out_h / 8), b * c];
      } else if (node.opType === "ConvTranspose2D") {
        wgslCode = ShaderLibrary.getConvTranspose2DShader();
        const inp = node.inputs.input || node.inputs.x;
        const [b, in_h, in_w, in_c] = inp.shape;
        const [, out_h, out_w, out_c] = node.outputs.out.shape;
        const kh = node.params.kernel![0], kw = node.params.kernel![1];
        const sh = node.params.stride ? node.params.stride[0] : 1;
        const sw = node.params.stride ? node.params.stride[1] : 1;
        const ph = node.params.padding ? node.params.padding[0] : 0;
        const pw = node.params.padding ? node.params.padding[1] : 0;
        const hasBias = node.inputs.bias ? 1 : 0;
        const p = new Uint32Array([b, in_h, in_w, in_c, out_h, out_w, out_c, kh, kw, sh, sw, ph, pw, hasBias]);
        const paramsBuf = this.host.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.host.device.queue.writeBuffer(paramsBuf, 0, p);
        const dummyBias = this.host.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
        bindGroupEntries = [
          { binding: 0, resource: { buffer: this.host._buffer(inp) } },
          { binding: 1, resource: { buffer: this.host._buffer(node.inputs.weight) } },
          { binding: 2, resource: { buffer: hasBias ? this.host._buffer(node.inputs.bias) : dummyBias } },
          { binding: 3, resource: { buffer: this.host._buffer(node.outputs.out) } },
          { binding: 4, resource: { buffer: paramsBuf } }
        ];
        workgroupCount = [Math.ceil(out_w / 8), Math.ceil(out_h / 8), b * out_c];
      } else {
        // No WebGPU shader for this op. Reject rather than silently skip: skipping left
        // the output buffer unwritten -> wrong results downstream (e.g. Sin/Cos returned
        // ~identity instead of erroring). Throwing here (during compile) lets
        // Model.compile fall back to a tier that supports the op (WASM/CPU)
        // instead of producing garbage on WebGPU.
        throw new Error(`[VolvoxAI WebGPU] ${node.opType} (node ${node.id}) has no WebGPU shader; cannot run this graph on WebGPU.`);
      }
      if (!wgslCode) {
        // A branch matched but produced no shader (would leave the output unwritten).
        // Same rationale: reject so compile falls back rather than miscompute silently.
        throw new Error(`[VolvoxAI WebGPU] ${node.opType} (node ${node.id}) has no native GPU shader; cannot run this graph on WebGPU.`);
      }
      if (isNaN(workgroupCount[0]) || isNaN(workgroupCount[1]) || isNaN(workgroupCount[2]) || workgroupCount[0] <= 0 || workgroupCount[1] <= 0 || workgroupCount[2] <= 0) {
        console.error(`Invalid workgroupCount [${workgroupCount}] for node ${node.id} (${node.opType})`);
        workgroupCount = [1, 1, 1];
      }
      if (fallbackWgslCode && fallbackWgslCode !== wgslCode && fallbackWorkgroupCount &&
          this.host.rejectedSpecializedShaders.has(wgslCode)) {
        wgslCode = fallbackWgslCode;
        workgroupCount = fallbackWorkgroupCount;
      }
      let pipeline;
      try {
        pipeline = await this.host._cachedComputePipeline(wgslCode);
      } catch (err) {
        if (!fallbackWgslCode || fallbackWgslCode === wgslCode || !fallbackWorkgroupCount) throw err;
        console.warn(`[VolvoxAI WebGPU] Specialized shader for ${node.id} failed; falling back to its portable kernel.`, err);
        this.host.rejectedSpecializedShaders.add(wgslCode);
        wgslCode = fallbackWgslCode;
        workgroupCount = fallbackWorkgroupCount;
        pipeline = await this.host._cachedComputePipeline(wgslCode);
      }
      const bindGroup = this.host.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindGroupEntries
      });
      (this.host._activePipelineTarget || this.host.pipelines).push({ pipeline, bindGroup, workgroupCount, nodeName: node.id });
    }
}
