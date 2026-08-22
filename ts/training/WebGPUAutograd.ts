import { GraphExecutor } from '../backends/GraphExecutor.js';
import {
  compileWebGPUGraphPlan,
  type WebGPUCompiledGraphPlan,
} from '../backends/WebGPUGraphCompiler.js';
import type { RuntimeGraph } from '../core/RuntimeGraph.js';
import type { Tensor } from '../core/Tensor.js';
import { DataType } from '../generated/volvoxaiEnums.js';
import type { RuntimeTypedArray } from '../types.js';
import { normalizeSpatialPair } from '../ops/spatialParameters.js';
import {
  checkedWindowOutput,
  fullSpatialPads,
  spatialPair,
} from '../ops/spatialKernelValidation.js';
import {
  canonicalOptimizerDescriptor,
  normalizeTrainingUpdateMode,
  validateOptimizerOptions,
} from './TrainingOptimizer.js';
import { ensureTrainingGraphState, synchronizeTrainingGraphState } from './TrainingGraph.js';
import type { TrainingGraph } from './TrainingGraph.js';
import type { TrainingKernelStepOptions } from './TrainingStep.js';
import {
  addGradient,
  crossEntropyGradient,
  normalizeCrossEntropyLosses,
} from './TrainingLosses.js';
import { dropoutContext, dropoutEffectiveSeed, dropoutProbability, dropoutThreshold } from '../ops/dropout.js';
import { geluApproximation } from '../ops/gelu.js';
import {
  attentionDropoutEffectiveSeed,
  attentionDropoutProbability,
} from '../ops/attentionDropout.js';
import {
  accumulateGradients,
  clipGradientsGlobal,
  gradientAccumulationIndex,
  resetGradientAccumulation as resetPendingGradients,
} from './GradientAccumulation.js';
import {
  BackendPlanCache,
  backendPlanCacheKey,
  concreteTrainingSignatures,
  trainingGraphTopologyIdentity,
  type BackendPlanCacheInspection,
  type BackendPlanCacheOptions,
} from './BackendPlanCache.js';

type TrainingShaderMap = typeof import('./TrainingShaderLibrary.js')['TrainingShaderLibrary'];
type TrainingDropoutContext = ReturnType<typeof dropoutContext>;

interface WebGPUTrainingDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroupCount: [number, number, number];
  resources: GPUBuffer[];
}

interface DetachedWebGPUNodeRecipe {
  readonly nodeIndex: number;
  readonly nodeId: string | number;
  readonly opType: string;
  readonly inputs: readonly (readonly [string, string])[];
  readonly outputs: readonly (readonly [string, string])[];
}

interface WebGPUBackendPlanRecipe {
  readonly shapeSignature: string;
  readonly tacticSignature: string;
  readonly topologyIdentity: string;
  readonly concreteDescriptorIdentity: string;
  readonly forwardPlan: WebGPUCompiledGraphPlan;
  readonly forwardNodes: readonly DetachedWebGPUNodeRecipe[];
  readonly backwardNodes: readonly DetachedWebGPUNodeRecipe[];
  readonly metadataBytes: number;
}

type WebGPUTrainingForwardExecutor = GraphExecutor & {
  executeTraining?(
    inputs: Record<string, RuntimeTypedArray>,
    pipelines: ReadonlyMap<number, WebGPUTrainingDispatch>,
  ): Promise<GPUBuffer | undefined>;
};

interface WebGPULossMetric {
  name: string;
  logitsTensor: string;
  loss: number;
  correct: number;
  examples: number;
  normalizer: number | null;
  weight: number;
}

export interface WebGPUTrainStepOptions extends TrainingKernelStepOptions {}

let TrainingShaders: TrainingShaderMap | undefined;

const product = (shape) => shape.reduce((a, b) => a * b, 1);
const firstOutput = (node) => node.outputs.out || Object.values(node.outputs || {})[0];
const nodeInput = (node) => node.inputs.input || node.inputs.x || node.inputs.data || node.inputs.a;
const sameShape = (left, right) => Array.isArray(left) && Array.isArray(right) &&
  left.length === right.length && left.every((dimension, index) => dimension === right[index]);
// Keep the backward descriptor byte-for-byte compatible with the canonical
// rank-1..8 forward Slice metadata in GraphExecutor/sliceNd.wgsl.
const WEBGPU_SLICE_MAX_RANK = 8;
const WEBGPU_GATHER_MAX_RANK = 8;
const MAX_U32 = 0xffffffff;
const WEBGPU_TYPED_DTYPE_CODES = Object.freeze({
  float32: DataType.F32,
  int32: DataType.I32,
  int8: DataType.I8,
  uint8: DataType.U8,
});
const WEBGPU_TYPED_DTYPE_BYTES = Object.freeze({
  float32: 4,
  int32: 4,
  int8: 1,
  uint8: 1,
});
const WEBGPU_PLAN_UTF8_ENCODER = new TextEncoder();

function webgpuConcreteDescriptorIdentity(graph: RuntimeGraph): string {
  return JSON.stringify([...graph.tensors.values()]
    .sort((left, right) => left.name.localeCompare(right.name))
    .map((tensor) => ({
      name: tensor.name,
      dtype: tensor.dtype,
      shape: tensor.shape,
      sizeBytes: tensor.sizeBytes,
      kind: tensor.isWeight ? 'weight' : tensor.isInput ? 'input' : 'activation',
    })));
}

function detachWebGPUNode(graph: RuntimeGraph, nodeIndex: number): DetachedWebGPUNodeRecipe {
  const node = graph.nodes[nodeIndex];
  const ports = (values) => Object.freeze(Object.entries(values || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, tensor]: [string, any]) => Object.freeze([name, tensor.name] as const)));
  return Object.freeze({
    nodeIndex,
    nodeId: node.id,
    opType: node.opType,
    inputs: ports(node.inputs),
    outputs: ports(node.outputs),
  });
}

function buildWebGPUBackendPlanRecipe(
  graph: RuntimeGraph,
  shapeSignature: string,
  tacticSignature: string,
  topologyIdentity: string,
): WebGPUBackendPlanRecipe {
  const forwardNodes = Object.freeze(graph.nodes.map((_, index) => detachWebGPUNode(graph, index)));
  const backwardNodes = Object.freeze([...forwardNodes].reverse());
  const document = {
    shapeSignature,
    tacticSignature,
    topologyIdentity,
    concreteDescriptorIdentity: webgpuConcreteDescriptorIdentity(graph),
    forwardPlan: compileWebGPUGraphPlan(graph),
    forwardNodes,
    backwardNodes,
  };
  const metadataBytes = WEBGPU_PLAN_UTF8_ENCODER.encode(JSON.stringify(document)).byteLength;
  return Object.freeze({ ...document, metadataBytes });
}

function assertWebGPUBackendPlanRecipe(
  graph: RuntimeGraph,
  recipe: WebGPUBackendPlanRecipe,
  shapeSignature: string,
  tacticSignature: string,
): void {
  if (recipe.shapeSignature !== shapeSignature || recipe.tacticSignature !== tacticSignature) {
    throw new Error('WebGPU cached plan key does not match its exact shape/tactic signatures.');
  }
  if (recipe.topologyIdentity !== trainingGraphTopologyIdentity(graph)) {
    throw new Error('WebGPU cached plan belongs to a different logical topology.');
  }
  if (recipe.concreteDescriptorIdentity !== webgpuConcreteDescriptorIdentity(graph)) {
    throw new Error(
      'WebGPU cached plan concrete tensor descriptors do not match the supplied shape/tactic key.',
    );
  }
  if (recipe.forwardNodes.length !== graph.nodes.length ||
      recipe.backwardNodes.length !== graph.nodes.length) {
    throw new Error('WebGPU cached plan node count does not match its concrete graph.');
  }
  for (const descriptor of recipe.forwardNodes) {
    const node = graph.nodes[descriptor.nodeIndex];
    if (!node || node.id !== descriptor.nodeId || node.opType !== descriptor.opType) {
      throw new Error(`WebGPU cached plan node ${descriptor.nodeIndex} changed identity or operator.`);
    }
  }
}

function concatInputs(node) {
  const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const seen = new Set();
  const ordered: any[] = [];
  for (const key of preferred) {
    if (node.inputs[key]) {
      ordered.push(node.inputs[key]);
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
  ordered.push(...remaining.map(([, tensor]) => tensor));
  return ordered;
}

function scalarValue(tensor, fallback) {
  if (!tensor?.buffer) return fallback;
  const view = tensor.buffer;
  if (view instanceof Float32Array) return view[0];
  if (ArrayBuffer.isView(view)) return new Float32Array(view.buffer, view.byteOffset, 1)[0];
  if (view instanceof ArrayBuffer) return new Float32Array(view, 0, 1)[0];
  return fallback;
}

function contiguousStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index--) {
    strides[index] = stride;
    stride *= shape[index];
  }
  return strides;
}

function positiveSliceDescriptor(node, input, output) {
  const rank = input?.shape?.length;
  const inputElements = u32ElementCount(input?.shape);
  const outputElements = u32ElementCount(output?.shape);
  const startsInput = node.params?.starts ?? [];
  const stepsInput = node.params?.steps ??
    (Array.isArray(startsInput) ? startsInput.map(() => 1) : null);
  const axesInput = node.params?.axes ??
    (Array.isArray(startsInput) ? startsInput.map((_, index) => index) : null);
  if (!input || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
      inputElements == null || outputElements == null || !Number.isInteger(rank) ||
      rank < 1 || rank > WEBGPU_SLICE_MAX_RANK || output.shape.length !== rank ||
      !Array.isArray(startsInput) || !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
      startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
    throw new Error(`WebGPU Slice node '${node.id}' requires rank 1..${WEBGPU_SLICE_MAX_RANK} F32 input/output tensors with matching parameter arrays.`);
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
        !Number.isSafeInteger(step) || step <= 0 || step > MAX_U32 || !Number.isSafeInteger(start)) {
      throw new Error(`WebGPU Slice node '${node.id}' requires unique axes and positive integer steps.`);
    }
    if (start < 0) start += input.shape[axis];
    if (start < 0 || start >= input.shape[axis]) {
      throw new Error(`WebGPU Slice node '${node.id}' start is outside its input axis.`);
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
      throw new Error(`WebGPU Slice node '${node.id}' output shape exceeds its input selection.`);
    }
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  return {
    rank,
    elements: outputElements,
    outputShape: [...output.shape, ...new Array(WEBGPU_SLICE_MAX_RANK - rank).fill(1)],
    starts: [...starts, ...new Array(WEBGPU_SLICE_MAX_RANK - rank).fill(0)],
    steps: [...steps, ...new Array(WEBGPU_SLICE_MAX_RANK - rank).fill(1)],
    inputStrides: [...inputStrides, ...new Array(WEBGPU_SLICE_MAX_RANK - rank).fill(1)],
  };
}

function u32ElementCount(shape) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = product(shape);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32 ? elements : null;
}

function typedDtypeCode(dtype) {
  return Object.prototype.hasOwnProperty.call(WEBGPU_TYPED_DTYPE_CODES, dtype)
    ? WEBGPU_TYPED_DTYPE_CODES[dtype]
    : -1;
}

function typedElementCount(tensor) {
  const elements = u32ElementCount(tensor?.shape);
  const bytes = WEBGPU_TYPED_DTYPE_BYTES[tensor?.dtype];
  if (!tensor || elements == null || bytes == null) return null;
  const expectedBytes = elements * bytes;
  return Number.isSafeInteger(expectedBytes) && tensor.sizeBytes === expectedBytes ? elements : null;
}

function castBackwardDescriptor(node, output) {
  const input = nodeInput(node);
  const inputElements = typedElementCount(input);
  const outputElements = typedElementCount(output);
  if (!input || !output || inputElements == null || outputElements == null ||
      inputElements !== outputElements) {
    throw new Error(`WebGPU Cast backward requires equal-size F32/I32/I8/U8 input and output storage (node ${node.id}).`);
  }
  return { input, inputElements, differentiable: input.dtype === 'float32' && output.dtype === 'float32' };
}

function dequantizeLinearBackwardDescriptor(node, output) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const scale = node.inputs.scale;
  const zeroPoint = node.inputs.zero_point || null;
  const inputElements = typedElementCount(input);
  const scaleElements = typedElementCount(scale);
  const zeroPointElements = zeroPoint ? typedElementCount(zeroPoint) : 1;
  const outputElements = typedElementCount(output);
  if (!input || !scale || !output || inputElements == null || outputElements == null ||
      output.dtype !== 'float32' || scale.dtype !== 'float32' || scaleElements !== 1 ||
      zeroPointElements !== 1 || inputElements !== outputElements) {
    throw new Error(`WebGPU DequantizeLinear backward requires F32 output/scalar scale and equal-size F32/I32/I8/U8 input/output storage (node ${node.id}).`);
  }
  const raw = new Uint32Array([
    inputElements,
    typedDtypeCode(input.dtype),
    zeroPoint ? typedDtypeCode(zeroPoint.dtype) : WEBGPU_TYPED_DTYPE_CODES.float32,
    zeroPoint ? 1 : 0,
  ]);
  return { raw, input, scale, zeroPoint, inputElements };
}

function gatherBackwardDescriptor(node, input, indices, output) {
  if (!input || !indices || !output || input.dtype !== 'float32' ||
      indices.dtype !== 'int32' || output.dtype !== 'float32') {
    throw new Error(`WebGPU Gather backward requires F32 data/output tensors and I32 indices (node ${node.id}).`);
  }
  const rank = input.shape?.length;
  let axis = node.params?.axis ?? 0;
  if (!Number.isInteger(axis)) {
    throw new Error(`WebGPU Gather node ${node.id} has a non-integer axis.`);
  }
  if (axis < 0) axis += rank;
  if (!Number.isInteger(rank) || rank < 1 || rank > WEBGPU_GATHER_MAX_RANK ||
      !Array.isArray(indices.shape) || indices.shape.length > WEBGPU_GATHER_MAX_RANK ||
      axis < 0 || axis >= rank) {
    throw new Error(`WebGPU Gather node ${node.id} requires rank-1..8 data, rank-0..8 indices, and a valid axis.`);
  }
  const expectedShape = [
    ...input.shape.slice(0, axis), ...indices.shape, ...input.shape.slice(axis + 1),
  ];
  if (!sameShape(output.shape, expectedShape) || expectedShape.length > WEBGPU_GATHER_MAX_RANK) {
    throw new Error(`WebGPU Gather node ${node.id} output shape must be input[:axis] + indices.shape + input[axis + 1:].`);
  }
  const outer = u32ElementCount(input.shape.slice(0, axis));
  const inner = u32ElementCount(input.shape.slice(axis + 1));
  const indicesElements = u32ElementCount(indices.shape);
  const inputElements = u32ElementCount(input.shape);
  const outputElements = u32ElementCount(output.shape);
  const axisSize = input.shape[axis];
  if (![outer, inner, indicesElements, inputElements, outputElements, axisSize]
    .every((value) => Number.isInteger(value) && value > 0 && value <= MAX_U32) ||
    inputElements !== outer * axisSize * inner ||
    outputElements !== outer * indicesElements * inner) {
    throw new Error(`WebGPU Gather node ${node.id} has an invalid element count.`);
  }
  return { axis, outer, axisSize, inner, indicesElements, inputElements, outputElements };
}

function gatherElementsBackwardDescriptor(node, input, indices, output) {
  if (!input || !indices || !output || input.dtype !== 'float32' ||
      indices.dtype !== 'int32' || output.dtype !== 'float32') {
    throw new Error(`WebGPU GatherElements backward requires F32 data/output tensors and I32 indices (node ${node.id}).`);
  }
  const rank = input.shape?.length;
  let axis = node.params?.axis ?? 0;
  if (!Number.isInteger(axis)) {
    throw new Error(`WebGPU GatherElements node ${node.id} has a non-integer axis.`);
  }
  if (axis < 0) axis += rank;
  const inputElements = u32ElementCount(input.shape);
  const outputElements = u32ElementCount(output.shape);
  if (!Number.isInteger(rank) || rank < 1 || rank > WEBGPU_GATHER_MAX_RANK ||
      !Array.isArray(indices.shape) || indices.shape.length !== rank ||
      !sameShape(output.shape, indices.shape) || axis < 0 || axis >= rank ||
      indices.shape.some((dimension, index) => index !== axis && dimension > input.shape[index]) ||
      !inputElements || !outputElements) {
    throw new Error(`WebGPU GatherElements node ${node.id} requires matching rank-1..8 index/output shapes and a valid axis.`);
  }
  if (u32ElementCount(indices.shape) !== outputElements) {
    throw new Error(`WebGPU GatherElements node ${node.id} has an invalid output element count.`);
  }
  return { rank, axis, inputElements, outputElements };
}

function visionDescriptor(node, input, output) {
  if (!input || input.dtype !== 'float32' || output?.dtype !== 'float32' || input.shape.length !== 4 ||
      output.shape.length !== 3) {
    throw new Error(`${node.opType} backward requires NHWC F32 input and rank-3 F32 output (node ${node.id}).`);
  }
  const [batch, height, width, channels] = input.shape;
  const expected = node.opType === 'ProfileX' ? [batch, 2 * channels, width]
    : node.opType === 'ProfileY' ? [batch, 2 * channels, height]
      : [batch, channels, width];
  if (!sameShape(output.shape, expected)) {
    throw new Error(`${node.opType} node ${node.id} has an incompatible output shape.`);
  }
  return { batch, height, width, channels };
}

function broadcastStrides(shape, outputShape) {
  if (shape.length > outputShape.length) return null;
  const padded = new Array(outputShape.length).fill(1);
  for (let index = 0; index < shape.length; index++) {
    padded[outputShape.length - shape.length + index] = shape[index];
  }
  const strides = contiguousStrides(padded);
  for (let index = 0; index < outputShape.length; index++) {
    if (padded[index] !== outputShape[index] && padded[index] !== 1) return null;
    if (padded[index] === 1 && outputShape[index] !== 1) strides[index] = 0;
  }
  return strides;
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

function encodeAttentionDropout(node, context, nodeIndex, u32, f32, offset) {
  const probability = attentionDropoutProbability(node);
  if (probability > 0 && !context) {
    throw new Error(`Attention node ${node.id} training dropout is missing its RNG context.`);
  }
  u32[offset] = dropoutThreshold(probability);
  u32[offset + 1] = context
    ? attentionDropoutEffectiveSeed(node, context, nodeIndex)
    : 0;
  u32[offset + 2] = context?.counter >>> 0 || 0;
  f32[offset + 3] = 1 / (1 - probability);
}

function sdpaTrainingParameters(node, context, nodeIndex) {
  const qkv = node.inputs.qkv;
  const output = firstOutput(node);
  const rank = qkv?.shape?.length;
  const batch = rank === 2 ? 1 : qkv?.shape?.[0];
  const outputBatch = output?.shape?.length === 2 ? 1 : output?.shape?.[0];
  if (!qkv || !output || qkv.dtype !== 'float32' || output.dtype !== 'float32' ||
      (rank !== 2 && rank !== 3) || output.shape.length !== rank || batch !== outputBatch) {
    throw new Error(`WebGPU SDPA training requires compatible rank-2 or rank-3 F32 tensors (node ${node.id}).`);
  }
  const seq = qkv.shape[rank - 2];
  const dModel = output.shape[rank - 1];
  const heads = node.params?.heads || 8;
  const headDim = dModel / heads;
  const mask = node.inputs.mask;
  const maskMode = attentionMaskMode(mask, batch, seq, seq, node.id);
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      heads * headDim !== dModel || headDim > 64 || qkv.shape[rank - 1] !== 3 * dModel ||
      output.shape[rank - 2] !== seq) {
    throw new Error(`SDPA node ${node.id} has incompatible head or tensor dimensions.`);
  }
  const raw = new ArrayBuffer(48);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32.set([seq, dModel, heads, headDim, batch]);
  f32[5] = node.params?.scale ?? 1 / Math.sqrt(headDim);
  u32[6] = node.params?.causal === false ? 0 : 1;
  u32[7] = maskMode;
  encodeAttentionDropout(node, context, nodeIndex, u32, f32, 8);
  return { raw, qkv, output, mask, seq, heads, batch };
}

function crossSdpaTrainingParameters(node, context, nodeIndex) {
  const q = node.inputs.q;
  const k = node.inputs.k;
  const v = node.inputs.v;
  const output = firstOutput(node);
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const batchOf = (tensor) => tensor?.shape?.length === 2 ? 1 : tensor?.shape?.[0];
  if (!q || !k || !v || !output || q.dtype !== 'float32' || k.dtype !== 'float32' ||
      v.dtype !== 'float32' || output.dtype !== 'float32' || (rank !== 2 && rank !== 3) ||
      k.shape.length !== rank || v.shape.length !== rank || output.shape.length !== rank ||
      batchOf(k) !== batch || batchOf(v) !== batch || batchOf(output) !== batch) {
    throw new Error(`WebGPU CrossSDPA training requires compatible rank-2 or rank-3 F32 tensors (node ${node.id}).`);
  }
  const seqQ = q.shape[rank - 2];
  const seqKV = k.shape[rank - 2];
  const dModel = q.shape[rank - 1];
  const heads = node.params?.heads || 8;
  const headDim = dModel / heads;
  const mask = node.inputs.mask;
  const maskMode = attentionMaskMode(mask, batch, seqQ, seqKV, node.id);
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      headDim > 64 || k.shape[rank - 1] !== dModel || v.shape[rank - 2] !== seqKV ||
      v.shape[rank - 1] !== dModel || output.shape[rank - 2] !== seqQ ||
      output.shape[rank - 1] !== dModel) {
    throw new Error(`CrossSDPA node ${node.id} has incompatible head or tensor dimensions.`);
  }
  const raw = new ArrayBuffer(64);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32.set([seqQ, seqKV, dModel, heads, headDim, batch]);
  f32[6] = node.params?.scale ?? 1 / Math.sqrt(headDim);
  u32[7] = node.params?.causal === true ? 1 : 0;
  u32[8] = maskMode;
  encodeAttentionDropout(node, context, nodeIndex, u32, f32, 9);
  return { raw, q, k, v, output, mask, seqQ, seqKV, heads, batch };
}

function crossAttentionBackwardDescriptor(node) {
  const q = node.inputs.q;
  const kv = node.inputs.kv;
  const weight = node.inputs.weight;
  const scale = node.inputs.scale || null;
  const bias = node.inputs.bias || null;
  const output = firstOutput(node);
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const batchOf = (tensor) => tensor?.shape?.length === 2 ? 1 : tensor?.shape?.[0];
  if (!q || !kv || !weight || !output || q.dtype !== 'float32' || kv.dtype !== 'float32' ||
      weight.dtype !== 'float32' || output.dtype !== 'float32' ||
      (scale && scale.dtype !== 'float32') || (bias && bias.dtype !== 'float32') ||
      (rank !== 2 && rank !== 3) || kv.shape?.length !== rank || output.shape?.length !== rank ||
      batchOf(kv) !== batch || batchOf(output) !== batch) {
    throw new Error(`WebGPU CrossAttention backward requires compatible rank-2 or rank-3 F32 Q/KV/projection/output tensors (node ${node.id}).`);
  }

  const seqQ = q.shape[rank - 2];
  const seqKV = kv.shape[rank - 2];
  const dModel = output.shape[rank - 1];
  if (![batch, seqQ, seqKV, dModel].every((value) =>
    Number.isInteger(value) && value > 0 && value <= MAX_U32) ||
      q.shape[rank - 1] !== dModel || kv.shape[rank - 1] !== dModel ||
      output.shape[rank - 2] !== seqQ || !sameShape(q.shape, output.shape)) {
    throw new Error(`CrossAttention node ${node.id} has incompatible Q/KV/output dimensions.`);
  }

  const heads = node.params?.heads || 8;
  const headDim = dModel / heads;
  // Keep this aligned with the authoritative F32 forward shader, whose local
  // projection arrays deliberately cap the model width at 64.
  if (dModel > 64) {
    throw new Error(`CrossAttention node ${node.id} F32 WebGPU kernel requires d_model <= 64.`);
  }
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      headDim > 64 || heads * headDim !== dModel) {
    throw new Error(`CrossAttention node ${node.id} has incompatible head dimensions.`);
  }

  const projectionWidth = 3 * dModel;
  const projectionElements = projectionWidth * dModel;
  const qElements = u32ElementCount(q.shape);
  const kvElements = u32ElementCount(kv.shape);
  const outputElements = u32ElementCount(output.shape);
  const weightElements = u32ElementCount(weight.shape);
  if (!Number.isSafeInteger(projectionWidth) || !Number.isSafeInteger(projectionElements) ||
      !qElements || !kvElements || !outputElements || !weightElements ||
      qElements !== batch * seqQ * dModel || kvElements !== batch * seqKV * dModel ||
      outputElements !== qElements || weight.shape?.length !== 2 ||
      weight.shape?.[0] !== projectionWidth || weight.shape?.[1] !== dModel ||
      weightElements !== projectionElements ||
      (scale && u32ElementCount(scale.shape) !== projectionWidth) ||
      (bias && u32ElementCount(bias.shape) !== projectionWidth)) {
    throw new Error(`CrossAttention node ${node.id} requires [3*d_model, d_model] F32 projections and optional 3*d_model F32 scale/bias tensors.`);
  }

  const raw = new ArrayBuffer(64);
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  u32.set([
    seqQ, seqKV, dModel, heads, headDim, batch,
    qElements, kvElements, weightElements, projectionWidth,
    scale ? 1 : 0, bias ? 1 : 0,
  ]);
  f32[12] = 1 / Math.sqrt(headDim);
  return {
    raw, q, kv, weight, scale, bias, output,
    seqQ, seqKV, dModel, heads, headDim, batch,
    qElements, kvElements, weightElements, projectionWidth,
  };
}

/**
 * Opt-in WebGPU reverse-mode trainer. Forward-only users never import the WGSL
 * training library and never allocate gradient buffers or backward pipelines.
 */
/** Full-profile forward owner. The inference executor never enters this mode. */
class TrainerOwnedGraphExecutor extends GraphExecutor {
  override async compile(): Promise<void> {
    this._assertPortableQuantizedGraph(this.graph as RuntimeGraph);
    this.resetDecodeCache();
    await this.graphCompiler.ensureShaderLibrary();
    this.resources.resetCompilationResources();
    this.pipelines = [];
    this.decodeState.resetCompilation();
    this.compiledGraphPlan = compileWebGPUGraphPlan(this.graph as RuntimeGraph);
    this._allocateBuffers();
    for (let nodeIndex = 0; nodeIndex < this.graph.nodes.length; nodeIndex++) {
      const start = this.pipelines.length;
      await this._buildNodePipeline(this.graph.nodes[nodeIndex]);
      for (let index = start; index < this.pipelines.length; index++) {
        this.pipelines[index].graphNodeIndex = nodeIndex;
      }
    }
    this.compiledWeightRevision = this.graph.weightRevision || 0;
    this.compiledTopologyRevision = this.graph.topologyRevision || 0;
  }

  executeTraining(
    inputs: Record<string, RuntimeTypedArray>,
    pipelines: ReadonlyMap<number, WebGPUTrainingDispatch>,
  ): Promise<GPUBuffer | undefined> {
    return this.dispatch.execute(inputs, { adapter: null }, pipelines);
  }
}

export class WebGPUAutograd {
  declare device: GPUDevice;
  declare graph: TrainingGraph;
  declare ownsExecutor: boolean;
  declare executor: WebGPUTrainingForwardExecutor;
  declare gradientBuffers: Map<string, GPUBuffer>;
  declare gradientCapacityBytes: Map<string, number>;
  declare pipelineCache: Map<string, GPUComputePipeline>;
  declare moduleCache: Map<string, GPUShaderModule>;
  readonly backendPlanCache: BackendPlanCache<WebGPUBackendPlanRecipe>;
  declare backendTopologyIdentity: string | null;
  declare currentBackendPlanKey: string | null;
  declare currentBackendPlanRecipe: WebGPUBackendPlanRecipe | null;
  declare dummyBuffers: Map<number, GPUBuffer>;
  declare gradientTopologyRevision: number;
  declare _activeResources: Set<GPUBuffer> | null;
  declare _training: boolean;

  constructor(
    device: GPUDevice,
    graph: RuntimeGraph,
    executor: WebGPUTrainingForwardExecutor | null = null,
    cacheOptions: BackendPlanCacheOptions = {},
  ) {
    if (!device) throw new Error('WebGPUAutograd requires a GPUDevice.');
    if (!graph) throw new Error('WebGPUAutograd requires a RuntimeGraph.');
    ensureTrainingGraphState(graph);
    this.device = device;
    this.graph = graph as TrainingGraph;
    this.ownsExecutor = executor == null;
    this.executor = executor || new TrainerOwnedGraphExecutor(device, graph);
    this.gradientBuffers = new Map();
    this.gradientCapacityBytes = new Map();
    this.pipelineCache = new Map();
    this.moduleCache = new Map();
    this.backendPlanCache = new BackendPlanCache('webgpu', cacheOptions);
    this.backendTopologyIdentity = null;
    this.currentBackendPlanKey = null;
    this.currentBackendPlanRecipe = null;
    this.dummyBuffers = new Map();
    this.gradientTopologyRevision = graph.topologyRevision || 0;
    this._activeResources = null;
    this._training = false;
  }

  async _loadShaders(): Promise<void> {
    if (!TrainingShaders) {
      ({ TrainingShaderLibrary: TrainingShaders } = await import('./TrainingShaderLibrary.js'));
    }
  }

  async _pipeline(shaderName: keyof TrainingShaderMap, entryPoint = 'main'): Promise<GPUComputePipeline> {
    await this._loadShaders();
    const key = `${shaderName}:${entryPoint}`;
    if (this.pipelineCache.has(key)) return this.pipelineCache.get(key)!;
    let module = this.moduleCache.get(shaderName);
    if (!module) {
      const code = TrainingShaders![shaderName];
      if (!code) throw new Error(`Missing WebGPU training shader '${shaderName}'.`);
      module = this.device.createShaderModule({ label: `Training_${shaderName}`, code });
      this.moduleCache.set(shaderName, module);
    }
    const pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: { module, entryPoint },
    });
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  _buffer(tensor: Tensor | null | undefined): GPUBuffer {
    const buffer = tensor && this.executor.gpuBuffers.get(tensor.name);
    if (!buffer) throw new Error(`WebGPU tensor '${tensor?.name}' is not allocated.`);
    return buffer;
  }

  _gradient(tensor: Tensor | null | undefined): GPUBuffer | null {
    if (!tensor || tensor.dtype !== 'float32') return null;
    let buffer = this.gradientBuffers.get(tensor.name);
    const requiredBytes = Math.max(4, Math.ceil(tensor.sizeBytes / 4) * 4);
    const priorCapacity = this.gradientCapacityBytes.get(tensor.name) ??
      (buffer ? requiredBytes : 0);
    if (!buffer || priorCapacity < requiredBytes) {
      const capacityBytes = Math.max(requiredBytes, priorCapacity > 0 ? priorCapacity * 2 : 0);
      buffer?.destroy?.();
      buffer = this.device.createBuffer({
        label: `Gradient_${tensor.name}`,
        size: capacityBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.gradientBuffers.set(tensor.name, buffer);
      this.gradientCapacityBytes.set(tensor.name, capacityBytes);
    }
    return buffer;
  }

  _resetGradientsForTopology() {
    const revision = this.graph.topologyRevision || 0;
    if (revision === this.gradientTopologyRevision) return;
    for (const buffer of this.gradientBuffers.values()) buffer.destroy?.();
    this.gradientBuffers.clear();
    this.gradientCapacityBytes.clear();
    this.gradientTopologyRevision = revision;
  }

  async rebind(
    graph: RuntimeGraph,
    binding: { readonly shapeSignature: string; readonly tacticSignature?: string },
  ): Promise<void> {
    if (this._training) throw new Error('WebGPUAutograd cannot rebind during a trainStep.');
    await this._bindBackendPlan(graph, binding);
  }

  async _bindBackendPlan(
    graph: RuntimeGraph,
    binding: { readonly shapeSignature?: string; readonly tacticSignature?: string } = {},
  ): Promise<void> {
    ensureTrainingGraphState(graph);
    const signatures = concreteTrainingSignatures(graph, binding);
    const key = backendPlanCacheKey(signatures.shapeSignature, signatures.tacticSignature);
    const topologyIdentity = trainingGraphTopologyIdentity(graph);
    const topologyChanged = this.backendTopologyIdentity !== null &&
      this.backendTopologyIdentity !== topologyIdentity;
    const cached = topologyChanged ? null : this.backendPlanCache.peek(key);
    const recipe = cached || buildWebGPUBackendPlanRecipe(
      graph,
      signatures.shapeSignature,
      signatures.tacticSignature,
      topologyIdentity,
    );
    assertWebGPUBackendPlanRecipe(
      graph,
      recipe,
      signatures.shapeSignature,
      signatures.tacticSignature,
    );
    await this.executor.rebindGraph(graph, {
      shapeSignature: signatures.shapeSignature,
      precompiledGraphPlan: recipe.forwardPlan,
      forwardNodeRecipe: recipe.forwardNodes,
      includeDropoutNodes: true,
    });
    if (topologyChanged) this.backendPlanCache.invalidateTopology();
    if (cached) {
      this.backendPlanCache.recordHit(key);
    } else {
      this.backendPlanCache.recordMiss();
      this.backendPlanCache.recordBuild();
      this.backendPlanCache.publish(key, recipe, recipe.metadataBytes);
    }
    this.backendPlanCache.recordMaterialization({ forward: true, backward: false });
    this.backendTopologyIdentity = topologyIdentity;
    this.currentBackendPlanKey = key;
    this.currentBackendPlanRecipe = recipe;
    this.graph = graph as TrainingGraph;
    this._resetGradientsForTopology();
  }

  inspectPlanCache(): Readonly<BackendPlanCacheInspection> {
    return this.backendPlanCache.inspect();
  }

  _floatGradient(tensor: Tensor | null | undefined, node: any, role: string): GPUBuffer {
    if (!tensor || tensor.dtype !== 'float32') {
      throw new Error(`WebGPU backward requires F32 ${role} at node ${node.id}.`);
    }
    return this._gradient(tensor)!;
  }

  _dummy(size = 4): GPUBuffer {
    const bytes = Math.max(4, Math.ceil(size / 4) * 4);
    let buffer = this.dummyBuffers.get(bytes);
    if (!buffer) {
      buffer = this.device.createBuffer({
        label: `TrainingDummy_${bytes}`,
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.dummyBuffers.set(bytes, buffer);
    }
    return buffer;
  }

  _parameterBuffer(value: ArrayBuffer | ArrayBufferView, storage = false): GPUBuffer {
    const bytes = value instanceof ArrayBuffer
      ? new Uint8Array(value)
      : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    const size = Math.max(16, Math.ceil(bytes.byteLength / 16) * 16);
    const buffer = this.device.createBuffer({
      size,
      usage: (storage ? GPUBufferUsage.STORAGE : GPUBufferUsage.UNIFORM) | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, bytes);
    this._activeResources?.add(buffer);
    return buffer;
  }

  _temporaryBuffer(size: number, label: string): GPUBuffer {
    const buffer = this.device.createBuffer({
      label,
      size: Math.max(4, Math.ceil(size / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this._activeResources?.add(buffer);
    return buffer;
  }

  async _readFloatBuffer(gpuBuffer: GPUBuffer, sizeBytes: number): Promise<Float32Array> {
    const staging = this.device.createBuffer({
      label: 'TrainingReadback',
      size: Math.max(4, Math.ceil(sizeBytes / 4) * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, staging, 0, sizeBytes);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return new Float32Array(staging.getMappedRange(0, sizeBytes).slice(0));
    } finally {
      try { staging.unmap(); } catch { /* It may not have reached the mapped state. */ }
      staging.destroy?.();
    }
  }

  async _dispatch(
    shaderName: keyof TrainingShaderMap,
    entryPoint: string,
    entries: Array<[number, GPUBuffer]>,
    workgroupCount: number[],
    resources: GPUBuffer[] = [],
  ): Promise<WebGPUTrainingDispatch> {
    const seen = new Set<number>();
    for (const [binding, buffer] of entries) {
      if (!Number.isInteger(binding) || binding < 0 || seen.has(binding)) {
        throw new Error(`Invalid or duplicate binding ${binding} for ${shaderName}.${entryPoint}.`);
      }
      if (!buffer) throw new Error(`Missing buffer at binding ${binding} for ${shaderName}.${entryPoint}.`);
      seen.add(binding);
    }
    const dispatchLimit = this.device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
    if (!Array.isArray(workgroupCount) || workgroupCount.length !== 3 ||
        workgroupCount.some((count) => !Number.isSafeInteger(count) || count <= 0 || count > dispatchLimit)) {
      throw new Error(`Invalid workgroup count for ${shaderName}.${entryPoint}.`);
    }
    const pipeline = await this._pipeline(shaderName, entryPoint);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    return {
      pipeline,
      bindGroup,
      workgroupCount: workgroupCount as [number, number, number],
      resources,
    };
  }

  _dropoutParameterBuffer(node: any, context: TrainingDropoutContext, nodeIndex: number): GPUBuffer {
    const probability = dropoutProbability(node);
    const raw = new ArrayBuffer(32);
    const u32 = new Uint32Array(raw);
    const f32 = new Float32Array(raw);
    u32[0] = product(firstOutput(node).shape);
    u32[1] = dropoutThreshold(probability);
    u32[2] = dropoutEffectiveSeed(node, context, nodeIndex);
    u32[3] = context.counter >>> 0;
    f32[4] = 1 / (1 - probability);
    return this._parameterBuffer(raw);
  }

  async _buildDropoutForwardOverrides(
    context: TrainingDropoutContext,
  ): Promise<Map<number, WebGPUTrainingDispatch>> {
    const overrides = new Map<number, WebGPUTrainingDispatch>();
    for (let nodeIndex = 0; nodeIndex < this.graph.nodes.length; nodeIndex++) {
      const node: any = this.graph.nodes[nodeIndex];
      const attentionProbability = node.opType === 'SDPA' || node.opType === 'CrossSDPA'
        ? attentionDropoutProbability(node)
        : 0;
      if (node.opType !== 'Dropout' && attentionProbability === 0) continue;
      const pipelineIndices: number[] = [];
      for (let index = 0; index < this.executor.pipelines.length; index++) {
        if (this.executor.pipelines[index].graphNodeIndex === nodeIndex) pipelineIndices.push(index);
      }
      if (pipelineIndices.length !== 1) {
        throw new Error(`Training dropout node ${node.id} must compile to exactly one inference pipeline.`);
      }
      let dispatch: WebGPUTrainingDispatch;
      if (node.opType === 'Dropout') {
        const input = nodeInput(node);
        const output = firstOutput(node);
        if (input?.dtype !== 'float32' || output?.dtype !== 'float32' ||
            product(input.shape) !== product(output.shape)) {
          throw new Error(`Dropout node ${node.id} requires equal-size F32 input/output tensors.`);
        }
        const params = this._dropoutParameterBuffer(node, context, nodeIndex);
        dispatch = await this._dispatch('dropout', 'main', [
          [0, this._buffer(input)],
          [1, this._buffer(output)],
          [2, params],
        ], [Math.ceil(product(output.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'SDPA') {
        const descriptor = sdpaTrainingParameters(node, context, nodeIndex);
        const params = this._parameterBuffer(descriptor.raw);
        dispatch = await this._dispatch('sdpaTraining', 'main', [
          [0, this._buffer(descriptor.qkv)],
          [1, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.qkv)],
          [2, this._buffer(descriptor.output)],
          [3, params],
        ], [Math.ceil(descriptor.seq / 64), descriptor.heads, descriptor.batch], [params]);
      } else {
        const descriptor = crossSdpaTrainingParameters(node, context, nodeIndex);
        const params = this._parameterBuffer(descriptor.raw);
        dispatch = await this._dispatch('crossSdpaTraining', 'main', [
          [0, this._buffer(descriptor.q)],
          [1, this._buffer(descriptor.k)],
          [2, this._buffer(descriptor.v)],
          [3, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.q)],
          [4, this._buffer(descriptor.output)],
          [5, params],
        ], [Math.ceil(descriptor.seqQ / 64), descriptor.heads, descriptor.batch], [params]);
      }
      overrides.set(pipelineIndices[0], dispatch);
    }
    return overrides;
  }

  async _buildBackwardDispatches(
    dropoutTraining: TrainingDropoutContext | null = null,
  ): Promise<WebGPUTrainingDispatch[]> {
    const dispatches: WebGPUTrainingDispatch[] = [];
    const add = async (
      shader: keyof TrainingShaderMap,
      entry: string,
      entries: any[],
      count: number[],
      resources: GPUBuffer[] = [],
    ): Promise<void> => {
      dispatches.push(await this._dispatch(
        shader,
        entry,
        entries as Array<[number, GPUBuffer]>,
        count,
        resources,
      ));
    };

    const backwardNodes = this.currentBackendPlanRecipe?.backwardNodes ??
      this.graph.nodes.map((node, nodeIndex) => ({
        nodeIndex,
        nodeId: node.id,
        opType: node.opType,
      })).reverse();
    for (const descriptor of backwardNodes) {
      const ni = descriptor.nodeIndex;
      const node: any = this.graph.nodes[ni];
      if (!node || node.id !== descriptor.nodeId || node.opType !== descriptor.opType) {
        throw new Error(`WebGPU backward plan node ${ni} changed identity or operator.`);
      }
      if (node.opType === 'Split') {
        const input = nodeInput(node);
        const outputKeys = Object.keys(node.outputs || {});
        const activeOutputs = outputKeys.filter((key) => this.gradientBuffers.has(node.outputs[key].name));
        if (!activeOutputs.length) continue;
        if (!input || input.dtype !== 'float32' || !outputKeys.length) {
          throw new Error(`Split backward requires an F32 input and at least one output (node ${node.id}).`);
        }
        let axis = node.params?.axis ?? 0;
        if (axis < 0) axis += input.shape.length;
        if (!Number.isInteger(axis) || axis < 0 || axis >= input.shape.length ||
            input.shape[axis] % outputKeys.length) {
          throw new Error(`Split node ${node.id} has an invalid axis or unequal split.`);
        }
        const splitSize = input.shape[axis] / outputKeys.length;
        const inner = product(input.shape.slice(axis + 1));
        const gi = this._floatGradient(input, node, 'Split input');
        for (const key of activeOutputs) {
          const outputIndex = outputKeys.indexOf(key);
          const splitOutput = node.outputs[key];
          if (splitOutput.dtype !== 'float32' || splitOutput.shape[axis] !== splitSize ||
              product(splitOutput.shape) * outputKeys.length !== product(input.shape)) {
            throw new Error(`Split node ${node.id} has an incompatible output '${key}'.`);
          }
          const total = product(splitOutput.shape);
          const params = this._parameterBuffer(new Uint32Array([
            total, inner, splitSize, input.shape[axis], outputIndex * splitSize,
          ]));
          await add('splitBackward', 'main', [[0, this.gradientBuffers.get(splitOutput.name)], [1, gi], [2, params]],
            [Math.ceil(total / 64), 1, 1], [params]);
        }
        continue;
      }
      const output = node.opType === 'MoERouter'
        ? (node.outputs.weights || node.outputs.expert_weights)
        : firstOutput(node);
      const go = output && this.gradientBuffers.get(output.name);
      if (!go) {
        const hasOtherLiveOutput = (Object.values(node.outputs || {}) as any[])
          .some((tensor) => tensor && this.gradientBuffers.has(tensor.name));
        if (hasOtherLiveOutput) {
          throw new Error(`WebGPU backward for a non-primary output of '${node.opType}' is not implemented (node ${node.id}).`);
        }
        continue;
      }

      if (node.opType === 'MatMul' || node.opType === 'Linear' || node.opType === 'Gemm') {
        const input = nodeInput(node);
        const weight = node.inputs.weight || node.inputs.b;
        const bias = node.inputs.bias;
        if (!input || !weight || !output || input.dtype !== 'float32' || weight.dtype !== 'float32' ||
            output.dtype !== 'float32' || (bias && bias.dtype !== 'float32')) {
          throw new Error(`WebGPU backward requires F32 linear tensors at node ${node.id}.`);
        }
        const rows = product(input.shape.slice(0, -1));
        const dIn = input.shape[input.shape.length - 1];
        const dOut = output.shape[output.shape.length - 1];
        if (node.wLayout !== 'din' && node.wLayout !== 'dout') {
          throw new Error(
            `WebGPU linear node ${node.id} requires a normalized 'din' or 'dout' weight layout.`,
          );
        }
        const canonicalDinLayout = node.wLayout === 'din' ? 1 : 0;
        const expectedWeightShape = canonicalDinLayout
          ? [dIn, dOut]
          : [dOut, dIn];
        const params = this._parameterBuffer(new Uint32Array([
          rows, dIn, dOut, bias ? 1 : 0,
          canonicalDinLayout, canonicalDinLayout, 0, 0,
        ]));
        if (!sameShape(weight.shape, expectedWeightShape) ||
            product(output.shape) !== rows * dOut ||
            (bias && product(bias.shape) !== dOut)) {
          throw new Error(`Linear node ${node.id} has incompatible input, weight, or output shapes.`);
        }
        const gi = this._floatGradient(input, node, 'linear input');
        const gw = this._floatGradient(weight, node, 'linear weight');
        const gb = bias ? this._floatGradient(bias, node, 'linear bias') : this._dummy(dOut * 4);
        await add('matMulBackward', 'input_main', [[1, this._buffer(weight)], [2, go], [3, gi], [6, params]],
          [Math.ceil(dIn / 16), Math.ceil(rows / 16), 1], [params]);
        await add('matMulBackward', 'weight_main', [[0, this._buffer(input)], [2, go], [4, gw], [6, params]],
          [Math.ceil(dOut / 16), Math.ceil(dIn / 16), 1], [params]);
        if (bias) {
          await add('matMulBackward', 'bias_main', [[2, go], [5, gb], [6, params]],
            [Math.ceil(dOut / 64), 1, 1], [params]);
        }
      } else if (['Add', 'Mul', 'Sub', 'Div'].includes(node.opType)) {
        const a = node.inputs.a || nodeInput(node);
        const b = node.inputs.b;
        if (!a || !b) throw new Error(`${node.opType} backward requires two inputs.`);
        const rank = output.shape.length;
        if (rank > 8) throw new Error(`${node.opType} backward supports at most 8 dimensions.`);
        const outputStrides = contiguousStrides(output.shape);
        const aStrides = broadcastStrides(a.shape, output.shape);
        const bStrides = broadcastStrides(b.shape, output.shape);
        if (!aStrides || !bStrides) {
          throw new Error(`${node.opType} node ${node.id} has incompatible broadcast shapes.`);
        }
        const raw = new Uint32Array(32);
        raw.set([rank, product(output.shape), product(a.shape), product(b.shape),
          { Add: 0, Mul: 1, Sub: 2, Div: 3 }[node.opType]], 0);
        raw.set(outputStrides, 8);
        raw.set(aStrides, 16);
        raw.set(bStrides, 24);
        const params = this._parameterBuffer(raw, true);
        const ga = this._floatGradient(a, node, 'left operand');
        const gb = this._floatGradient(b, node, 'right operand');
        await add('basicBackward', 'a_main', [[1, this._buffer(b)], [2, go], [3, ga], [5, params]],
          [Math.ceil(product(a.shape) / 64), 1, 1], [params]);
        await add('basicBackward', 'b_main', [[0, this._buffer(a)], [2, go], [4, gb], [5, params]],
          [Math.ceil(product(b.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'Where' || node.opType === 'Mask') {
        const condition = node.inputs.cond || node.inputs.condition || node.inputs.mask;
        const a = node.inputs.x || node.inputs.a || nodeInput(node);
        const b = node.inputs.y || node.inputs.b;
        if (!condition || !a || !b || (condition.dtype !== 'float32' && condition.dtype !== 'int32') ||
            a.dtype !== 'float32' || b.dtype !== 'float32' || output.dtype !== 'float32' ||
            !sameShape(condition.shape, output.shape) || !sameShape(a.shape, output.shape) ||
            !sameShape(b.shape, output.shape)) {
          throw new Error(`${node.opType} backward requires exact-shape F32 operands and an F32/I32 condition (node ${node.id}).`);
        }
        const params = this._parameterBuffer(new Uint32Array([
          product(output.shape), condition.dtype === 'float32' ? 1 : 0, 0, 0,
        ]));
        await add('whereBackward', 'main', [
          [0, this._buffer(condition)],
          [1, go],
          [2, this._floatGradient(a, node, `${node.opType} x operand`)],
          [3, this._floatGradient(b, node, `${node.opType} y operand`)],
          [4, params],
        ], [Math.ceil(product(output.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'Slice') {
        const input = nodeInput(node);
        if (!input || input.dtype !== 'float32' || output.dtype !== 'float32') {
          throw new Error(`WebGPU Slice backward requires F32 input/output tensors (node ${node.id}).`);
        }
        const descriptor = positiveSliceDescriptor(node, input, output);
        const params = new Uint32Array(36);
        params.set([descriptor.rank, descriptor.elements, 0, 0]);
        params.set(descriptor.outputShape, 4);
        params.set(descriptor.starts, 12);
        params.set(descriptor.steps, 20);
        params.set(descriptor.inputStrides, 28);
        const parameterBuffer = this._parameterBuffer(params);
        await add('sliceBackward', 'main', [
          [0, go],
          [1, this._floatGradient(input, node, 'Slice input')],
          [2, parameterBuffer],
        ], [Math.ceil(descriptor.elements / 64), 1, 1], [parameterBuffer]);
      } else if (node.opType === 'Gather') {
        const input = nodeInput(node);
        const indices = node.inputs.indices;
        const descriptor = gatherBackwardDescriptor(node, input, indices, output);
        const params = this._parameterBuffer(new Uint32Array([
          descriptor.outer, descriptor.axisSize, descriptor.inner, descriptor.indicesElements,
        ]));
        // I32 indices are intentionally non-differentiable. The shader reduces
        // all repeated selections into each F32 data-gradient element.
        await add('gatherBackward', 'main', [
          [0, this._buffer(indices)],
          [1, go],
          [2, this._floatGradient(input, node, 'Gather data')],
          [3, params],
        ], [Math.ceil(descriptor.inputElements / 64), 1, 1], [params]);
      } else if (node.opType === 'GatherElements') {
        const input = nodeInput(node);
        const indices = node.inputs.indices;
        const descriptor = gatherElementsBackwardDescriptor(node, input, indices, output);
        const raw = new Uint32Array(20);
        raw.set([descriptor.rank, descriptor.axis, descriptor.inputElements, descriptor.outputElements]);
        raw.set(input.shape, 4);
        raw.set(indices.shape, 12);
        const params = this._parameterBuffer(raw);
        // I32 indices are intentionally non-differentiable. Negative values
        // are normalized by the shader according to GatherElements semantics.
        await add('gatherElementsBackward', 'main', [
          [0, this._buffer(indices)],
          [1, go],
          [2, this._floatGradient(input, node, 'GatherElements data')],
          [3, params],
        ], [Math.ceil(descriptor.inputElements / 64), 1, 1], [params]);
      } else if (node.opType === 'MeanHeight') {
        const input = nodeInput(node);
        const { batch, height, width, channels } = visionDescriptor(node, input, output);
        const params = this._parameterBuffer(new Uint32Array([height, width, channels, batch]));
        await add('visionBackward', 'mean_height_main', [
          [2, go],
          [3, this._floatGradient(input, node, 'MeanHeight input')],
          [4, params],
        ], [Math.ceil(width / 64), channels, batch], [params]);
      } else if (node.opType === 'ProfileX' || node.opType === 'ProfileY') {
        const input = nodeInput(node);
        const { batch, height, width, channels } = visionDescriptor(node, input, output);
        const params = this._parameterBuffer(new Uint32Array([height, width, channels, batch]));
        const entry = node.opType === 'ProfileX' ? 'profile_x_main' : 'profile_y_main';
        const extent = node.opType === 'ProfileX' ? width : height;
        await add('visionBackward', entry, [
          [0, this._buffer(input)],
          [2, go],
          [3, this._floatGradient(input, node, `${node.opType} input`)],
          [4, params],
        ], [Math.ceil(extent / 64), channels, batch], [params]);
      } else if (node.opType === 'SpatialSoftargmaxY') {
        const input = nodeInput(node);
        const { batch, height, width, channels } = visionDescriptor(node, input, output);
        const params = this._parameterBuffer(new Uint32Array([height, width, channels, batch]));
        await add('visionBackward', 'spatial_softargmax_y_main', [
          [0, this._buffer(input)],
          [1, this._buffer(output)],
          [2, go],
          [3, this._floatGradient(input, node, 'SpatialSoftargmaxY input')],
          [4, params],
        ], [Math.ceil(width / 64), channels, batch], [params]);
      } else if (node.opType === 'Expand' || node.opType === 'Broadcast') {
        const input = nodeInput(node);
        if (!input || input.dtype !== 'float32' || output.dtype !== 'float32' ||
            input.shape.length === 0 || input.shape.length > output.shape.length || output.shape.length > 8) {
          throw new Error(`${node.opType} backward requires rank-1..8 F32 broadcast tensors.`);
        }
        const offset = output.shape.length - input.shape.length;
        if (output.shape.some((dimension, index) => {
          const source = index < offset ? 1 : input.shape[index - offset];
          return source !== 1 && source !== dimension;
        })) throw new Error(`${node.opType} node ${node.id} has incompatible broadcast shapes.`);
        const raw = new Uint32Array(20);
        raw.set([input.shape.length, output.shape.length, product(input.shape), product(output.shape)]);
        raw.set(input.shape, 4);
        raw.set(output.shape, 12);
        const params = this._parameterBuffer(raw, true);
        await add('expandBackward', 'main', [
          [0, go], [1, this._floatGradient(input, node, `${node.opType} input`)], [2, params],
        ], [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
      } else if(node.opType==='Pad'){
        const input=nodeInput(node),pads=node.params?.pads||[];
        if(!input||input.dtype!=='float32'||input.shape.length!==4||output.shape.length!==4||(pads.length!==4&&pads.length!==8)) throw new Error(`Pad backward requires rank-4 NHWC tensors and 4 or 8 pads.`);
        const [pt,pl,pb,pr]=pads.length===8?[pads[1],pads[2],pads[5],pads[6]]:pads,[batch,height,width,channels]=input.shape;
        if(![pt,pl,pb,pr].every(v=>Number.isInteger(v)&&v>=0)||output.shape[0]!==batch||output.shape[1]!==height+pt+pb||output.shape[2]!==width+pl+pr||output.shape[3]!==channels) throw new Error(`Pad node ${node.id} has incompatible dimensions.`);
        const params=this._parameterBuffer(new Uint32Array([batch,height,width,channels,output.shape[1],output.shape[2],channels,0,pt,pl,pb,pr]));
        await add('padBackward','main',[[0,go],[1,this._floatGradient(input,node,'Pad input')],[2,params]],[Math.ceil(product(input.shape)/64),1,1],[params]);
      } else if(node.opType==='Interpolate1D'||node.opType==='Interp1D'||node.opType==='InterpLinear1D'){
        const input=nodeInput(node);if(!input||input.dtype!=='float32'||input.shape.length!==3||output.shape.length!==3||output.shape[0]!==input.shape[0]||output.shape[1]!==input.shape[1])throw new Error(`${node.opType} backward requires matching rank-3 NCL tensors.`);
        const params=this._parameterBuffer(new Uint32Array([input.shape[0],input.shape[1],input.shape[2],output.shape[2]]));await add('interp1DBackward','main',[[0,go],[1,this._floatGradient(input,node,`${node.opType} input`)],[2,params]],[Math.ceil(product(input.shape)/64),1,1],[params]);
      } else if (node.opType === 'Dropout') {
        if (!dropoutTraining) throw new Error(`Dropout node ${node.id} backward is missing its training RNG context.`);
        const input = nodeInput(node);
        if (input?.dtype !== 'float32' || product(input.shape) !== product(output.shape)) {
          throw new Error(`Dropout node ${node.id} requires equal-size F32 input/output tensors.`);
        }
        const params = this._dropoutParameterBuffer(node, dropoutTraining, ni);
        await add('dropoutBackward', 'main', [
          [0, go],
          [1, this._floatGradient(input, node, 'Dropout input')],
          [2, params],
        ], [Math.ceil(product(output.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'Cast') {
        const descriptor = castBackwardDescriptor(node, output);
        // F32-to-F32 is an identity. Every conversion involving integer
        // storage deliberately stops gradients, matching the CPU trainer.
        if (descriptor.differentiable) {
          const params = this._parameterBuffer(new Uint32Array([descriptor.inputElements, 0, 0, 0]));
          await add('copyBackward', 'main', [
            [0, go],
            [1, this._floatGradient(descriptor.input, node, 'Cast input')],
            [2, params],
          ], [Math.ceil(descriptor.inputElements / 64), 1, 1], [params]);
        }
      } else if (node.opType === 'DequantizeLinear') {
        const descriptor = dequantizeLinearBackwardDescriptor(node, output);
        const params = this._parameterBuffer(descriptor.raw);
        if (descriptor.input.dtype === 'float32') {
          await add('dequantizeLinearBackward', 'input_main', [
            [2, this._buffer(descriptor.scale)],
            [3, go],
            [4, this._floatGradient(descriptor.input, node, 'DequantizeLinear F32 input')],
            [6, params],
          ], [Math.ceil(descriptor.inputElements / 64), 1, 1], [params]);
        }
        // The scalar F32 scale is differentiable for all portable input
        // dtypes. The raw input decoder handles F32/I32/I8/U8 faithfully.
        await add('dequantizeLinearBackward', 'scale_main', [
          [0, this._buffer(descriptor.input)],
          [1, descriptor.zeroPoint ? this._buffer(descriptor.zeroPoint) : this._dummy()],
          [3, go],
          [5, this._floatGradient(descriptor.scale, node, 'DequantizeLinear scale')],
          [6, params],
        ], [1, 1, 1], [params]);
      } else if (node.opType === 'ReduceSum' || node.opType === 'ReduceMean') {
        const input = nodeInput(node);
        const rowWidth = input?.shape?.at(-1);
        const length = product(input?.shape || []);
        const rows = length / rowWidth;
        if (input?.dtype !== 'float32' || !Number.isSafeInteger(rowWidth) || rowWidth <= 0 ||
            !Number.isSafeInteger(rows) || product(output.shape) !== rows) {
          throw new Error(`${node.opType} node ${node.id} must reduce the last axis into one F32 value per outer row.`);
        }
        const raw = new ArrayBuffer(16);
        const u32 = new Uint32Array(raw);
        const f32 = new Float32Array(raw);
        u32[0] = length;
        u32[1] = rowWidth;
        f32[2] = node.opType === 'ReduceMean' ? 1 / rowWidth : 1;
        const params = this._parameterBuffer(raw);
        await add('reduceBackward', 'main', [
          [0, go],
          [1, this._floatGradient(input, node, `${node.opType} input`)],
          [2, params],
        ], [Math.ceil(length / 64), 1, 1], [params]);
      } else if (['Identity', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze'].includes(node.opType)) {
        const input = nodeInput(node);
        const gi = this._floatGradient(input, node, 'shape-op input');
        const params = this._parameterBuffer(new Uint32Array([product(output.shape), 0, 0, 0]));
        await add('copyBackward', 'main', [[0, go], [1, gi], [2, params]],
          [Math.ceil(product(output.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'Transpose') {
        const input = nodeInput(node);
        const rank = input.shape.length;
        if (rank > 8) throw new Error(`Transpose backward supports at most 8 dimensions (node ${node.id}).`);
        const perm = node.params?.perm || [...Array(rank).keys()].reverse();
        if (perm.length !== rank || new Set(perm).size !== rank ||
            perm.some((dimension) => !Number.isInteger(dimension) || dimension < 0 || dimension >= rank)) {
          throw new Error(`Transpose node ${node.id} has an invalid permutation.`);
        }
        const inputStrides = new Array(8).fill(1);
        const outputShape = perm.map((p) => input.shape[p]);
        const outputStrides = new Array(8).fill(1);
        let stride = 1;
        for (let i = rank - 1; i >= 0; i--) { inputStrides[i] = stride; stride *= input.shape[i]; }
        stride = 1;
        for (let i = rank - 1; i >= 0; i--) { outputStrides[i] = stride; stride *= outputShape[i]; }
        const raw = new Uint32Array(34);
        raw[0] = rank; raw[1] = product(outputShape);
        raw.set([...input.shape, ...new Array(8 - rank).fill(1)], 2);
        raw.set([...perm, ...new Array(8 - rank).fill(0)], 10);
        raw.set(inputStrides, 18); raw.set(outputStrides, 26);
        const params = this._parameterBuffer(raw, true);
        await add('transposeBackward', 'main', [[0, go], [1, this._floatGradient(input, node, 'transpose input')], [2, params]],
          [Math.ceil(product(outputShape) / 64), 1, 1], [params]);
      } else if (['ReLU', 'GELU', 'SiLU', 'Swish', 'Sigmoid', 'Tanh', 'LeakyReLU',
        'HardSigmoid', 'HardSwish', 'Clip'].includes(node.opType)) {
        const input = nodeInput(node);
        const kinds = {
          ReLU: 0, GELU: 1, SiLU: 2, Swish: 2, Sigmoid: 3, Tanh: 4,
          LeakyReLU: 5, HardSigmoid: 6, HardSwish: 7, Clip: 8,
        };
        const raw = new ArrayBuffer(16);
        const u32 = new Uint32Array(raw); const f32 = new Float32Array(raw);
        u32[0] = product(output.shape);
        u32[1] = node.opType === 'GELU' && geluApproximation(node) === 'tanh'
          ? 9
          : kinds[node.opType];
        if (node.opType === 'Clip') {
          f32[2] = scalarValue(node.inputs.min, node.params?.min ?? -1e9);
          f32[3] = scalarValue(node.inputs.max, node.params?.max ?? 1e9);
        } else {
          f32[2] = node.params?.alpha ?? 0.01;
        }
        const params = this._parameterBuffer(raw);
        await add('activationBackward', 'main', [[0, this._buffer(input)], [1, this._buffer(output)], [2, go],
          [3, this._floatGradient(input, node, 'activation input')], [4, params]],
        [Math.ceil(product(output.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'PReLU') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const length = product(output.shape);
        const channels = input.shape[input.shape.length - 1];
        const weightLength = product(weight?.shape || []);
        if (weightLength !== 1 && weightLength !== channels) {
          throw new Error(`PReLU node ${node.id} weight must contain 1 or ${channels} values.`);
        }
        const params = this._parameterBuffer(new Uint32Array([length, channels, weightLength, 0]));
        await add('preluBackward', 'input_main', [[0, this._buffer(input)], [1, this._buffer(weight)], [2, go],
          [3, this._floatGradient(input, node, 'PReLU input')], [5, params]],
        [Math.ceil(length / 64), 1, 1], [params]);
        await add('preluBackward', 'weight_main', [[0, this._buffer(input)], [2, go],
          [4, this._floatGradient(weight, node, 'PReLU weight')], [5, params]],
        [Math.ceil(weightLength / 64), 1, 1], [params]);
      } else if (node.opType === 'Softmax' || node.opType === 'LogSoftmax') {
        const input = nodeInput(node);
        const width = input.shape[input.shape.length - 1];
        const rows = product(input.shape) / width;
        const params = this._parameterBuffer(new Uint32Array([
          rows, width, node.opType === 'LogSoftmax' ? 1 : 0, 0,
        ]));
        await add('softmaxBackward', 'main', [[0, this._buffer(output)], [1, go],
          [2, this._floatGradient(input, node, `${node.opType} input`)], [3, params]],
        [Math.ceil(rows * width / 64), 1, 1], [params]);
      } else if (node.opType === 'Embedding') {
        const tokens = nodeInput(node);
        const weight = node.inputs.weight;
        if (!tokens || tokens.dtype !== 'int32' || !weight || weight.dtype !== 'float32') {
          throw new Error(`Embedding backward requires int32 token IDs and an F32 weight at node ${node.id}.`);
        }
        const dModel = output.shape[output.shape.length - 1];
        const vocab = weight.shape[0];
        const tokenCount = product(tokens.shape);
        if (product(weight.shape) !== vocab * dModel || product(output.shape) !== tokenCount * dModel) {
          throw new Error(`Embedding node ${node.id} has incompatible tensor shapes.`);
        }
        const params = this._parameterBuffer(new Uint32Array([tokenCount, dModel, vocab, 0]));
        await add('embeddingBackward', 'main', [[0, this._buffer(tokens)], [1, go],
          [2, this._floatGradient(weight, node, 'embedding weight')], [3, params]],
          [Math.ceil(vocab * dModel / 64), 1, 1], [params]);
      } else if (node.opType === 'LayerNorm') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        if (!input || !weight || input.dtype !== 'float32' || weight.dtype !== 'float32' ||
            (bias && bias.dtype !== 'float32')) {
          throw new Error(`LayerNorm backward requires F32 tensors at node ${node.id}.`);
        }
        const dModel = input.shape[input.shape.length - 1];
        if (node.params?.d_model != null && node.params.d_model !== dModel) {
          throw new Error(`LayerNorm node ${node.id} requires params.d_model=${dModel} to match its last dimension.`);
        }
        const rows = product(input.shape.slice(0, -1));
        if (product(weight.shape) !== dModel || (bias && product(bias.shape) !== dModel) ||
            product(output.shape) !== rows * dModel) {
          throw new Error(`LayerNorm node ${node.id} has incompatible tensor shapes.`);
        }
        const raw = new ArrayBuffer(16); const u32 = new Uint32Array(raw); const f32 = new Float32Array(raw);
        u32[0] = rows; u32[1] = dModel; f32[2] = node.params?.eps ?? 1e-6; u32[3] = bias ? 1 : 0;
        const params = this._parameterBuffer(raw);
        await add('layerNormBackward', 'input_main', [[0, this._buffer(input)], [1, this._buffer(weight)], [2, go],
          [3, this._floatGradient(input, node, 'LayerNorm input')], [6, params]],
        [Math.ceil(rows * dModel / 64), 1, 1], [params]);
        const paramEntries = [[0, this._buffer(input)], [2, go],
          [4, this._floatGradient(weight, node, 'LayerNorm weight')], [6, params]];
        if (bias) paramEntries.push([5, this._floatGradient(bias, node, 'LayerNorm bias')]);
        else paramEntries.push([5, this._dummy(dModel * 4)]);
        await add('layerNormBackward', 'param_main', paramEntries,
          [Math.ceil(dModel / 64), 1, 1], [params]);
      } else if (node.opType === 'GroupNorm') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        if (!input || !weight || !bias || !output || input.dtype !== 'float32' ||
            weight.dtype !== 'float32' || bias.dtype !== 'float32' || output.dtype !== 'float32') {
          throw new Error(`GroupNorm backward requires F32 input, weight, bias, and output tensors at node ${node.id}.`);
        }
        if (input.shape.length !== 4 || output.shape.length !== 4 ||
            input.shape.some((dimension, index) => output.shape[index] !== dimension)) {
          throw new Error(`GroupNorm node ${node.id} requires matching rank-4 NHWC input and output tensors.`);
        }
        const [batch, height, width, channels] = input.shape;
        const numGroups = node.params?.num_groups;
        const eps = node.params?.eps ?? 1e-5;
        if (!Number.isInteger(numGroups) || numGroups <= 0 || channels % numGroups !== 0) {
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
        u32.set([batch, height, width, channels, numGroups, 1]);
        f32[6] = eps;
        const params = this._parameterBuffer(raw);
        await add('groupNormBackward', 'input_main', [
          [0, this._buffer(input)],
          [1, this._buffer(weight)],
          [2, go],
          [3, this._floatGradient(input, node, 'GroupNorm input')],
          [6, params],
        ], [Math.ceil(batch * numGroups / 64), 1, 1], [params]);
        await add('groupNormBackward', 'param_main', [
          [0, this._buffer(input)],
          [2, go],
          [4, this._floatGradient(weight, node, 'GroupNorm weight')],
          [5, this._floatGradient(bias, node, 'GroupNorm bias')],
          [6, params],
        ], [Math.ceil(channels / 64), 1, 1], [params]);
      } else if (node.opType === 'RMSNorm') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        if (!input || !weight || input.dtype !== 'float32' || weight.dtype !== 'float32') {
          throw new Error(`RMSNorm backward requires F32 tensors at node ${node.id}.`);
        }
        const dModel = node.params?.d_model || input.shape[input.shape.length - 1];
        const inputLength = product(input.shape);
        if (!Number.isInteger(dModel) || dModel <= 0 || inputLength % dModel) {
          throw new Error(`RMSNorm node ${node.id} has an invalid d_model.`);
        }
        const rows = inputLength / dModel;
        if (product(weight.shape) !== dModel || product(output.shape) !== rows * dModel) {
          throw new Error(`RMSNorm node ${node.id} has incompatible tensor shapes.`);
        }
        const raw = new ArrayBuffer(16); const u32 = new Uint32Array(raw); const f32 = new Float32Array(raw);
        u32[0] = rows; u32[1] = dModel; f32[2] = node.params?.eps ?? 1e-6;
        const params = this._parameterBuffer(raw);
        await add('rmsNormBackward', 'input_main', [[0, this._buffer(input)], [1, this._buffer(weight)], [2, go],
          [3, this._floatGradient(input, node, 'RMSNorm input')], [5, params]],
        [Math.ceil(rows * dModel / 64), 1, 1], [params]);
        await add('rmsNormBackward', 'weight_main', [[0, this._buffer(input)], [2, go],
          [4, this._floatGradient(weight, node, 'RMSNorm weight')], [5, params]],
        [Math.ceil(dModel / 64), 1, 1], [params]);
      } else if (node.opType === 'BatchNorm2D') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        const mean = node.inputs.running_mean;
        const variance = node.inputs.running_var;
        const values = [input, weight, bias, mean, variance, output];
        if (values.some((value) => !value || value.dtype !== 'float32') || input.shape.length !== 4 ||
            product(output.shape) !== product(input.shape)) {
          throw new Error(`BatchNorm2D backward requires rank-4 F32 input/output and F32 parameters (node ${node.id}).`);
        }
        const channels = input.shape[3];
        if ([weight, bias, mean, variance].some((value) => product(value.shape) < channels)) {
          throw new Error(`BatchNorm2D node ${node.id} parameter length is smaller than ${channels}.`);
        }
        const length = product(input.shape);
        const raw = new ArrayBuffer(16); const u32 = new Uint32Array(raw); const f32 = new Float32Array(raw);
        u32[0] = length; u32[1] = channels; f32[2] = node.params?.eps ?? 1e-5;
        const params = this._parameterBuffer(raw);
        await add('batchNorm2DBackward', 'input_main', [[1, this._buffer(weight)],
          [3, this._buffer(variance)], [4, go],
          [5, this._floatGradient(input, node, 'BatchNorm2D input')], [8, params]],
        [Math.ceil(length / 64), 1, 1], [params]);
        await add('batchNorm2DBackward', 'param_main', [[0, this._buffer(input)], [2, this._buffer(mean)],
          [3, this._buffer(variance)], [4, go],
          [6, this._floatGradient(weight, node, 'BatchNorm2D weight')],
          [7, this._floatGradient(bias, node, 'BatchNorm2D bias')], [8, params]],
        [Math.ceil(channels / 64), 1, 1], [params]);
      } else if (node.opType === 'GlobalAveragePool') {
        const input = nodeInput(node);
        if (!input || input.dtype !== 'float32' || input.shape.length !== 4) {
          throw new Error(`GlobalAveragePool backward requires a rank-4 F32 input (node ${node.id}).`);
        }
        const [n, h, w, c] = input.shape;
        if (product(output.shape) !== n * c) {
          throw new Error(`GlobalAveragePool node ${node.id} has an incompatible output shape.`);
        }
        const params = this._parameterBuffer(new Uint32Array([n, h, w, c, 0, 0, 0, 0, 0, 0, 0, 0]));
        await add('poolingBackward', 'global_average_main', [[1, go],
          [2, this._floatGradient(input, node, 'GlobalAveragePool input')], [3, params]],
        [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'MaxPool2D') {
        const input = nodeInput(node);
        if (!input || input.dtype !== 'float32' || output.dtype !== 'float32' ||
            input.shape.length !== 4 || output.shape.length !== 4 ||
            input.shape[0] !== output.shape[0] || input.shape[3] !== output.shape[3]) {
          throw new Error(`WebGPU MaxPool2D backward requires compatible rank-4 F32 tensors (node ${node.id}).`);
        }
        const [n, h, w, c] = input.shape;
        const [, outH, outW] = output.shape;
        const paramsRecord = node.params || {};
        const [ky, kx] = spatialPair(paramsRecord.kernel, 1, 'MaxPool2D', 'kernel', false, true);
        const [sy, sx] = spatialPair(paramsRecord.stride, 1, 'MaxPool2D', 'stride', false);
        const pads = fullSpatialPads(paramsRecord, 'MaxPool2D');
        const expectedOutH = checkedWindowOutput(
          h, ky, sy, pads[0], pads[2], 1, `WebGPU MaxPool2D output height at node ${node.id}`,
        );
        const expectedOutW = checkedWindowOutput(
          w, kx, sx, pads[1], pads[3], 1, `WebGPU MaxPool2D output width at node ${node.id}`,
        );
        if (outH !== expectedOutH || outW !== expectedOutW) {
          throw new Error(
            `MaxPool2D node ${node.id} output shape is incompatible with its canonical pooling parameters.`,
          );
        }
        const params = this._parameterBuffer(new Uint32Array([
          n, h, w, c, outH, outW, ky, kx, sy, sx, pads[0], pads[1],
        ]));
        await add('poolingBackward', 'max_pool_main', [[0, this._buffer(input)], [1, go],
          [2, this._floatGradient(input, node, 'MaxPool2D input')], [3, params]],
        [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
      } else if (['Resize', 'ResizeNearest2D', 'UpsampleNearest2D', 'Upsample2x'].includes(node.opType)) {
        const input = nodeInput(node);
        if (!input || input.dtype !== 'float32' || input.shape.length !== 4 || output.shape.length !== 4 ||
            input.shape[0] !== output.shape[0] || input.shape[3] !== output.shape[3]) {
          throw new Error(`${node.opType} backward requires compatible rank-4 F32 tensors (node ${node.id}).`);
        }
        const [n, h, w, c] = input.shape;
        const [, outH, outW] = output.shape;
        const nearest = node.opType !== 'Resize' || node.params?.mode === 'nearest';
        const params = this._parameterBuffer(new Uint32Array([n, h, w, c, outH, outW, nearest ? 0 : 1, 0]));
        await add('resizeBackward', 'main', [[0, go],
          [1, this._floatGradient(input, node, `${node.opType} input`)], [2, params]],
        [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
      } else if (node.opType === 'Concat') {
        const orderedInputs = concatInputs(node);
        if (!orderedInputs.length || orderedInputs.some((input) => input.dtype !== 'float32')) {
          throw new Error(`Concat backward requires at least one F32 input (node ${node.id}).`);
        }
        const rank = output.shape.length;
        let axis = node.params?.axis ?? 0;
        if (!Number.isInteger(axis)) throw new Error(`Concat node ${node.id} has a non-integer axis.`);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank || orderedInputs.some((input) => input.shape.length !== rank)) {
          throw new Error(`Concat node ${node.id} has incompatible ranks or axis.`);
        }
        const inner = product(output.shape.slice(axis + 1));
        let axisOffset = 0;
        for (const input of orderedInputs) {
          for (let dim = 0; dim < rank; dim++) {
            if (dim !== axis && input.shape[dim] !== output.shape[dim]) {
              throw new Error(`Concat node ${node.id} has incompatible input shapes.`);
            }
          }
          const length = product(input.shape);
          const inputAxis = input.shape[axis];
          const params = this._parameterBuffer(new Uint32Array([
            length, axisOffset, inputAxis, output.shape[axis], inner, 0, 0, 0,
          ]));
          await add('concatBackward', 'main', [[0, go],
            [1, this._floatGradient(input, node, 'Concat input')], [2, params]],
          [Math.ceil(length / 64), 1, 1], [params]);
          axisOffset += inputAxis;
        }
        if (axisOffset !== output.shape[axis]) {
          throw new Error(`Concat node ${node.id} input axes do not match its output.`);
        }
      } else if (node.opType === 'ConvTranspose2D') {
        const input = nodeInput(node), weight = node.inputs.weight, bias = node.inputs.bias;
        if (!input || !weight || input.dtype !== 'float32' || weight.dtype !== 'float32' || (bias && bias.dtype !== 'float32') || input.shape.length !== 4 || weight.shape.length !== 4 || output.shape.length !== 4) throw new Error(`ConvTranspose2D backward requires rank-4 NHWC F32 tensors at node ${node.id}.`);
        // HWIO weights [kh, kw, in_c, out_c].
        const [batch,inHeight,inWidth,inChannels]=input.shape,[kernelY,kernelX,weightIn,outChannels]=weight.shape,[,outHeight,outWidth,outputChannels]=output.shape,[strideY,strideX]=normalizeSpatialPair(node.params?.stride,1),[padY,padX]=normalizeSpatialPair(node.params?.padding,0);
        if(weightIn!==inChannels||outputChannels!==outChannels||output.shape[0]!==batch||!Number.isInteger(strideY)||!Number.isInteger(strideX)||strideY<=0||strideX<=0||!Number.isInteger(padY)||!Number.isInteger(padX)||padY<0||padX<0||outHeight!==(inHeight-1)*strideY+kernelY-2*padY||outWidth!==(inWidth-1)*strideX+kernelX-2*padX||(bias&&product(bias.shape)!==outChannels)) throw new Error(`ConvTranspose2D node ${node.id} has incompatible dimensions or parameters.`);
        const params=this._parameterBuffer(new Uint32Array([batch,inHeight,inWidth,inChannels,outHeight,outWidth,outChannels,strideX,kernelY,kernelX,strideY,padY,padX,0,0,0]));
        const gi=this._floatGradient(input,node,'ConvTranspose2D input'),gw=this._floatGradient(weight,node,'ConvTranspose2D weight'),gb=bias?this._floatGradient(bias,node,'ConvTranspose2D bias'):this._dummy(outChannels*4);
        await add('convTranspose2DBackward','input_main',[[1,this._buffer(weight)],[2,go],[3,gi],[6,params]],[Math.ceil(product(input.shape)/64),1,1],[params]);
        await add('convTranspose2DBackward','weight_main',[[0,this._buffer(input)],[2,go],[4,gw],[6,params]],[Math.ceil(product(weight.shape)/64),1,1],[params]);
        if(bias) await add('convTranspose2DBackward','bias_main',[[2,go],[5,gb],[6,params]],[Math.ceil(outChannels/64),1,1],[params]);
      } else if (node.opType === 'Conv1D') {
        const input = nodeInput(node), weight = node.inputs.weight, bias = node.inputs.bias;
        if (!input || !weight || input.dtype !== 'float32' || weight.dtype !== 'float32' ||
            (bias && bias.dtype !== 'float32') || input.shape.length !== 3 || weight.shape.length !== 3 || output.shape.length !== 3) {
          throw new Error(`Conv1D backward requires rank-3 NLC F32 tensors at node ${node.id}.`);
        }
        // NLC activations [batch, l, c]; WIO weights [k, in_per_group, out_c].
        const [batch, inputLength, inChannels] = input.shape, [kernel, inputPerGroup, outChannels] = weight.shape;
        const [, outputLength, outputChannels] = output.shape;
        const groups = node.params?.groups ?? 1, stride = normalizeSpatialPair(node.params?.stride, 1)[0], padding = normalizeSpatialPair(node.params?.padding, 0)[0];
        if (output.shape[0] !== batch || outputChannels !== outChannels || !Number.isInteger(groups) || groups <= 0 ||
            inChannels % groups || outChannels % groups || inputPerGroup !== inChannels / groups ||
            !Number.isInteger(stride) || stride <= 0 || !Number.isInteger(padding) || padding < 0 ||
            inputLength + 2 * padding < kernel || Math.floor((inputLength + 2 * padding - kernel) / stride) + 1 !== outputLength ||
            (bias && product(bias.shape) !== outChannels)) throw new Error(`Conv1D node ${node.id} has incompatible grouped NLC dimensions.`);
        const params = this._parameterBuffer(new Uint32Array([
          inChannels, inputLength, outChannels, kernel, stride, padding, node.params?.relu ? 1 : 0, batch,
          groups, inputPerGroup, outputLength, 0,
        ]));
        const gi = this._floatGradient(input, node, 'Conv1D input'), gw = this._floatGradient(weight, node, 'Conv1D weight');
        const gb = bias ? this._floatGradient(bias, node, 'Conv1D bias') : this._dummy(outChannels * 4);
        await add('conv1DBackward', 'input_main', [[1, this._buffer(weight)], [2, this._buffer(output)], [3, go], [4, gi], [7, params]], [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
        await add('conv1DBackward', 'weight_main', [[0, this._buffer(input)], [2, this._buffer(output)], [3, go], [5, gw], [7, params]], [Math.ceil(product(weight.shape) / 64), 1, 1], [params]);
        if (bias) await add('conv1DBackward', 'bias_main', [[2, this._buffer(output)], [3, go], [6, gb], [7, params]], [Math.ceil(outChannels / 64), 1, 1], [params]);
      } else if (node.opType === 'Conv2D') {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        if (!input || !weight || input.dtype !== 'float32' || weight.dtype !== 'float32' ||
            (bias && bias.dtype !== 'float32') || input.shape.length !== 4 || weight.shape.length !== 4 ||
            output.shape.length !== 4) {
          throw new Error(`Conv2D backward requires rank-4 F32 tensors at node ${node.id}.`);
        }
        const [n, inH, inW, inC] = input.shape;
        const [kh, kw, weightIn, weightOut] = weight.shape;
        const [, outH, outW, outC] = output.shape;
        const [sy, sx] = normalizeSpatialPair(node.params?.stride, 1);
        const [py, px] = normalizeSpatialPair(node.params?.padding, 0);
        const pads = node.params?.pads || [py, px, py, px];
        const [dy, dx] = normalizeSpatialPair(node.params?.dilation, 1);
        const groups = node.params?.groups || 1;
        if (output.shape[0] !== n || (bias && product(bias.shape) !== outC) ||
            !Number.isInteger(groups) || groups <= 0 || inC % groups || outC % groups ||
            weightOut !== (groups === inC ? outC / inC : outC) ||
            weightIn !== (groups === inC ? inC : inC / groups)) {
          throw new Error(`Conv2D node ${node.id} has an invalid group or weight layout.`);
        }
        if (![sy, sx, dy, dx].every((value) => Number.isInteger(value) && value > 0) ||
            !pads.slice(0, 2).every((value) => Number.isInteger(value) && value >= 0)) {
          throw new Error(`Conv2D node ${node.id} has invalid stride, dilation, or padding.`);
        }
        const p = new Uint32Array(20);
        p.set([n, inH, inW, inC, outH, outW, outC, kh, kw, sy, sx, pads[0], pads[1], groups,
          dy, dx, node.params?.relu || 0, bias ? 1 : 0, product(weight.shape), 0]);
        const params = this._parameterBuffer(p);
        await add('conv2DBackward', 'input_main', [[1, this._buffer(weight)], [2, this._buffer(output)], [3, go],
          [4, this._floatGradient(input, node, 'Conv2D input')], [7, params]],
        [Math.ceil(product(input.shape) / 64), 1, 1], [params]);
        await add('conv2DBackward', 'weight_main', [[0, this._buffer(input)], [2, this._buffer(output)], [3, go],
          [5, this._floatGradient(weight, node, 'Conv2D weight')], [7, params]],
        [Math.ceil(product(weight.shape) / 64), 1, 1], [params]);
        if (bias) await add('conv2DBackward', 'bias_main', [[2, this._buffer(output)], [3, go],
          [6, this._floatGradient(bias, node, 'Conv2D bias')], [7, params]],
        [Math.ceil(outC / 64), 1, 1], [params]);
      } else if (node.opType === 'SDPA') {
        const descriptor = sdpaTrainingParameters(node, dropoutTraining, ni);
        const params = this._parameterBuffer(descriptor.raw);
        await add('sdpaBackward', 'main', [[0, this._buffer(descriptor.qkv)],
          [1, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.qkv)],
          [2, go], [3, this._floatGradient(descriptor.qkv, node, 'SDPA qkv')], [4, params]],
          [descriptor.seq, descriptor.heads, descriptor.batch * 3], [params]);
      } else if (node.opType === 'CrossSDPA') {
        const descriptor = crossSdpaTrainingParameters(node, dropoutTraining, ni);
        const params = this._parameterBuffer(descriptor.raw);
        await add('crossSdpaBackward', 'q_main', [[0, this._buffer(descriptor.q)],
          [1, this._buffer(descriptor.k)], [2, this._buffer(descriptor.v)],
          [3, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.q)], [4, go],
          [5, this._floatGradient(descriptor.q, node, 'CrossSDPA q')], [8, params]],
          [descriptor.seqQ, descriptor.heads, descriptor.batch], [params]);
        await add('crossSdpaBackward', 'k_main', [[0, this._buffer(descriptor.q)],
          [1, this._buffer(descriptor.k)], [2, this._buffer(descriptor.v)],
          [3, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.q)], [4, go],
          [6, this._floatGradient(descriptor.k, node, 'CrossSDPA k')], [8, params]],
          [descriptor.seqKV, descriptor.heads, descriptor.batch], [params]);
        await add('crossSdpaBackward', 'v_main', [[0, this._buffer(descriptor.q)],
          [1, this._buffer(descriptor.k)],
          [3, descriptor.mask ? this._buffer(descriptor.mask) : this._buffer(descriptor.q)], [4, go],
          [7, this._floatGradient(descriptor.v, node, 'CrossSDPA v')], [8, params]],
          [descriptor.seqKV, descriptor.heads, descriptor.batch], [params]);
      } else if (node.opType === 'CrossAttention') {
        const descriptor = crossAttentionBackwardDescriptor(node);
        const params = this._parameterBuffer(descriptor.raw);
        const scale = descriptor.scale ? this._buffer(descriptor.scale) : this._dummy();
        const bias = descriptor.bias ? this._buffer(descriptor.bias) : this._dummy();
        const common = [
          [0, this._buffer(descriptor.q)],
          [1, this._buffer(descriptor.kv)],
          [2, this._buffer(descriptor.weight)],
          [3, scale],
          [4, bias],
          [5, go],
        ];
        await add('crossAttentionBackward', 'q_main', [
          ...common,
          [6, this._floatGradient(descriptor.q, node, 'CrossAttention q')],
          [11, params],
        ], [Math.ceil(descriptor.qElements / 64), 1, 1], [params]);
        await add('crossAttentionBackward', 'kv_main', [
          ...common,
          [7, this._floatGradient(descriptor.kv, node, 'CrossAttention kv')],
          [11, params],
        ], [Math.ceil(descriptor.kvElements / 64), 1, 1], [params]);
        await add('crossAttentionBackward', 'weight_main', [
          ...common,
          [8, this._floatGradient(descriptor.weight, node, 'CrossAttention projection weight')],
          [11, params],
        ], [Math.ceil(descriptor.weightElements / 64), 1, 1], [params]);
        if (descriptor.scale) {
          await add('crossAttentionBackward', 'scale_main', [
            ...common,
            [9, this._floatGradient(descriptor.scale, node, 'CrossAttention projection scale')],
            [11, params],
          ], [Math.ceil(descriptor.projectionWidth / 64), 1, 1], [params]);
        }
        if (descriptor.bias) {
          await add('crossAttentionBackward', 'bias_main', [
            ...common,
            [10, this._floatGradient(descriptor.bias, node, 'CrossAttention projection bias')],
            [11, params],
          ], [Math.ceil(descriptor.projectionWidth / 64), 1, 1], [params]);
        }
      } else if (node.opType === 'MoELinear') {
        const input = nodeInput(node);
        const expertWeight = node.inputs.expert_weight || node.inputs.weight;
        const expertBias = node.inputs.expert_bias || node.inputs.bias;
        const indices = node.inputs.route_indices || node.inputs.indices;
        const gates = node.inputs.route_weights || node.inputs.weights;
        if (!input || !expertWeight || !indices || !gates || input.dtype !== 'float32' ||
            expertWeight.dtype !== 'float32' || indices.dtype !== 'float32' || gates.dtype !== 'float32' ||
            (expertBias && expertBias.dtype !== 'float32') || expertWeight.shape.length !== 3) {
          throw new Error(`MoELinear backward requires F32 tensors and rank-3 expert weights at node ${node.id}.`);
        }
        const rows = product(input.shape.slice(0, -1));
        const dIn = input.shape[input.shape.length - 1];
        const dOut = output.shape[output.shape.length - 1];
        const experts = expertWeight.shape[0];
        const topK = indices.shape[indices.shape.length - 1];
        if (expertWeight.shape[1] !== dIn || expertWeight.shape[2] !== dOut ||
            product(indices.shape) !== rows * topK || product(gates.shape) !== rows * topK ||
            product(output.shape) !== rows * dOut ||
            (expertBias && product(expertBias.shape) !== experts * dOut) ||
            !Number.isInteger(topK) || topK <= 0) {
          throw new Error(`MoELinear node ${node.id} has incompatible expert or routing shapes.`);
        }
        // A partially resident bank routes by global slot id, so the shader
        // needs both directions of the staged-row mapping.
        const residentSlots = node.residentSlots;
        const residentSlotDomain = node.residentSlotDomain;
        if (residentSlots === undefined && residentSlotDomain !== undefined) {
          throw new Error(
            `MoELinear node ${node.id} has a resident slot domain without a slot table.`);
        }
        if (residentSlots !== undefined && residentSlots.length !== experts) {
          throw new Error(
            `MoELinear node ${node.id} lists ${residentSlots.length} resident slots ` +
            `but stages ${experts} experts.`);
        }
        if (residentSlots !== undefined &&
            (!Number.isSafeInteger(residentSlotDomain) || residentSlotDomain <= 0 ||
             residentSlotDomain > MAX_U32 ||
             !residentSlots.every((slot, row) =>
               Number.isSafeInteger(slot) && slot >= 0 && slot < residentSlotDomain &&
               (row === 0 || slot > residentSlots[row - 1])))) {
          throw new Error(
            `MoELinear node ${node.id} has invalid resident slot-domain metadata.`);
        }
        // top-k is bounded by the staged expert rows, not by their largest
        // global slot id. This is the same contract used by forward and the
        // native training planner.
        if (topK > experts) {
          throw new Error(`MoELinear node ${node.id} has incompatible expert or routing shapes.`);
        }
        const slotDomain = residentSlots === undefined ? 0 : residentSlotDomain;
        const slotRowsTable = new Uint32Array(Math.max(1, slotDomain)).fill(0xffffffff);
        const rowSlotsTable = new Uint32Array(Math.max(1, experts));
        if (residentSlots !== undefined) {
          for (let row = 0; row < residentSlots.length; row++) {
            slotRowsTable[residentSlots[row]] = row;
            rowSlotsTable[row] = residentSlots[row];
          }
        }
        const slotRows = this._parameterBuffer(slotRowsTable, true);
        const rowSlots = this._parameterBuffer(rowSlotsTable, true);
        const banks: Array<[number, GPUBuffer]> = [[10, slotRows], [11, rowSlots]];
        const params = this._parameterBuffer(
          new Uint32Array([rows, dIn, dOut, experts, topK, expertBias ? 1 : 0, slotDomain, 0]));
        await add('moeLinearBackward', 'input_main', [[1, this._buffer(expertWeight)], [3, this._buffer(indices)],
          [4, this._buffer(gates)], [5, go], [6, this._floatGradient(input, node, 'MoELinear input')],
          ...banks, [12, params]],
          [Math.ceil(rows * dIn / 64), 1, 1], [params]);
        await add('moeLinearBackward', 'weight_main', [[0, this._buffer(input)], [3, this._buffer(indices)],
          [4, this._buffer(gates)], [5, go],
          [7, this._floatGradient(expertWeight, node, 'MoELinear expert weight')],
          ...banks, [12, params]],
          [Math.ceil(product(expertWeight.shape) / 64), 1, 1], [params]);
        if (expertBias) await add('moeLinearBackward', 'bias_main', [[3, this._buffer(indices)], [4, this._buffer(gates)],
          [5, go], [8, this._floatGradient(expertBias, node, 'MoELinear expert bias')],
          ...banks, [12, params]],
          [Math.ceil(product(expertBias.shape) / 64), 1, 1], [params]);
        await add('moeLinearBackward', 'route_main', [[0, this._buffer(input)], [1, this._buffer(expertWeight)],
          [2, expertBias ? this._buffer(expertBias) : this._dummy(experts * dOut * 4)], [3, this._buffer(indices)],
          [5, go], [9, this._floatGradient(gates, node, 'MoELinear route weights')],
          ...banks, [12, params]],
        [Math.ceil(rows * topK / 64), 1, 1], [params]);
      } else if (node.opType === 'MoERouter') {
        const input = nodeInput(node);
        const weight = node.inputs.weight || node.inputs.router_weight;
        const bias = node.inputs.bias;
        const indices = node.outputs.indices || node.outputs.expert_indices;
        const gates = node.outputs.weights || node.outputs.expert_weights;
        if (!input || !weight || !indices || !gates || input.dtype !== 'float32' ||
            weight.dtype !== 'float32' || indices.dtype !== 'float32' || gates.dtype !== 'float32' ||
            (bias && bias.dtype !== 'float32')) {
          throw new Error(`MoERouter backward requires F32 tensors at node ${node.id}.`);
        }
        const rows = product(input.shape.slice(0, -1));
        const dModel = input.shape[input.shape.length - 1];
        const experts = node.params?.num_experts ?? weight.shape[weight.shape.length - 1];
        const topK = indices.shape[indices.shape.length - 1];
        const temperature = node.params?.temperature ?? 1;
        if (weight.shape.length !== 2 || weight.shape[0] !== dModel || weight.shape[1] !== experts ||
            !Number.isInteger(experts) || experts <= 0 || !Number.isInteger(topK) || topK <= 0 ||
            topK > experts || topK > 8 || product(indices.shape) !== rows * topK ||
            product(gates.shape) !== rows * topK || (bias && product(bias.shape) < experts) ||
            !(temperature > 0) || !Number.isFinite(temperature)) {
          throw new Error(`MoERouter node ${node.id} has incompatible routing dimensions or parameters.`);
        }
        const raw = new ArrayBuffer(32); const u32 = new Uint32Array(raw); const f32 = new Float32Array(raw);
        u32.set([rows, dModel, experts, topK, node.params?.normalize === false ? 0 : 1, bias ? 1 : 0]);
        f32[6] = temperature;
        const params = this._parameterBuffer(raw);
        const gradLogits = this._temporaryBuffer(rows * experts * 4, `MoERouterGradLogits_${node.id}`);
        const biasBuffer = bias ? this._buffer(bias) : this._dummy(experts * 4);
        await add('moeRouterBackward', 'logit_main', [[0, this._buffer(input)], [1, this._buffer(weight)],
          [2, biasBuffer], [3, this._buffer(indices)], [4, this._buffer(gates)], [5, go], [6, gradLogits],
          [10, params]], [Math.ceil(rows * experts / 64), 1, 1], [params, gradLogits]);
        await add('moeRouterBackward', 'input_main', [[1, this._buffer(weight)], [6, gradLogits],
          [7, this._floatGradient(input, node, 'MoERouter input')], [10, params]],
        [Math.ceil(rows * dModel / 64), 1, 1], [params, gradLogits]);
        await add('moeRouterBackward', 'weight_main', [[0, this._buffer(input)], [6, gradLogits],
          [8, this._floatGradient(weight, node, 'MoERouter weight')], [10, params]],
        [Math.ceil(dModel * experts / 64), 1, 1], [params, gradLogits]);
        if (bias) await add('moeRouterBackward', 'bias_main', [[6, gradLogits],
          [9, this._floatGradient(bias, node, 'MoERouter bias')], [10, params]],
        [Math.ceil(experts / 64), 1, 1], [params, gradLogits]);
      } else {
        throw new Error(`WebGPU backward for '${node.opType}' is not implemented (node ${node.id}).`);
      }
    }
    if (this.currentBackendPlanRecipe) {
      this.backendPlanCache.recordMaterialization({ forward: false, backward: true });
    }
    return dispatches;
  }

  async _ensureForwardCompiled() {
    if (this.executor.compiledTopologyRevision == null) {
      await this._bindBackendPlan(this.graph, concreteTrainingSignatures(this.graph));
    }
  }

  _assertStepState(topologyRevision: number, weightRevision: number): void {
    this.graph.assertTopologyRevision?.(topologyRevision, 'WebGPU training');
    if ((this.graph.weightRevision || 0) !== weightRevision) {
      throw new Error('WebGPU training weights changed concurrently during trainStep.');
    }
  }

  _uploadUpdatedTensor(tensor: Tensor): void {
    const values = tensor.buffer as Float32Array;
    const gpu = this._buffer(tensor);
    this.device.queue.writeBuffer(gpu, 0, values.buffer, values.byteOffset, values.byteLength);
  }

  async trainStep(options: WebGPUTrainStepOptions = {}) {
    if (this._training) throw new Error('WebGPUAutograd does not support concurrent trainStep calls.');
    this._training = true;
    const resources = new Set<GPUBuffer>();
    this._activeResources = resources;
    try {
      return await this._trainStepImpl(options);
    } finally {
      // Parameter and scratch buffers must stay alive until every submitted
      // dispatch/readback that references them has completed.
      try { await this.device.queue.onSubmittedWorkDone?.(); } catch { /* Device-loss errors surface elsewhere. */ }
      for (const resource of resources) resource.destroy?.();
      this._activeResources = null;
      this._training = false;
    }
  }

  async _trainStepImpl({
    inputs = {},
    targets,
    logitsTensor,
    losses = null,
    trainableTensors,
    optimizer = {},
    updateMode = 'adamw',
    lastToken = null,
    ignoreIndex = null,
    lossMask = null,
    dropout = {},
    gradientAccumulationSteps = 1,
    flushGradientAccumulation = false,
    resetGradientAccumulation = false,
    shapeSignature = this.graph.trainingShapeSignature,
    tacticSignature = this.graph.trainingTacticSignature,
  }: WebGPUTrainStepOptions = {}) {
    if (!trainableTensors?.length) throw new Error('WebGPUAutograd.trainStep requires trainableTensors.');
    if (new Set(trainableTensors).size !== trainableTensors.length) {
      throw new Error('WebGPUAutograd.trainStep trainableTensors must be unique.');
    }
    const normalizedUpdateMode = normalizeTrainingUpdateMode(updateMode);
    validateOptimizerOptions(optimizer, normalizedUpdateMode);
    const optimizerDescriptor = canonicalOptimizerDescriptor(normalizedUpdateMode, optimizer);
    const defaultLogitsName = logitsTensor || this.graph.outputNames?.[0] ||
      firstOutput(this.graph.nodes.at(-1))?.name;
    const lossDescriptors = normalizeCrossEntropyLosses({
      targets,
      logitsTensor,
      losses,
      ignoreIndex,
      lossMask,
      lastToken,
    }, defaultLogitsName);
    if (!Number.isSafeInteger(gradientAccumulationSteps) || gradientAccumulationSteps <= 0) {
      throw new Error('WebGPUAutograd.trainStep gradientAccumulationSteps must be a positive safe integer.');
    }
    if (gradientAccumulationSteps > 1 && lossDescriptors.length > 1 &&
        lossDescriptors.some((loss) => loss.normalizer == null)) {
      throw new Error(
        'Multiple losses with gradient accumulation require an explicit full-window normalizer for every loss.',
      );
    }
    if (typeof flushGradientAccumulation !== 'boolean' || typeof resetGradientAccumulation !== 'boolean') {
      throw new Error('WebGPUAutograd.trainStep accumulation flush/reset options must be boolean.');
    }
    if (resetGradientAccumulation) resetPendingGradients(this.graph);
    const currentTrainingStep = this.graph.trainingStep ?? 0;
    const nextTrainingStep = optimizer.step ?? (currentTrainingStep + 1);
    if (!Number.isSafeInteger(currentTrainingStep) || currentTrainingStep < 0 ||
        !Number.isSafeInteger(nextTrainingStep) || nextTrainingStep <= currentTrainingStep) {
      throw new Error('WebGPUAutograd.trainStep optimizer step must be a safe integer greater than graph.trainingStep.');
    }
    const trainables = trainableTensors.map((name) => {
      const tensor = this.graph.getTensor(name);
      if (!tensor?.isWeight || tensor.dtype !== 'float32' || !(tensor.buffer instanceof Float32Array)) {
        throw new Error(`Trainable tensor '${name}' must be an initialized F32 graph weight.`);
      }
      return tensor;
    });
    for (const node of this.graph.nodes) {
      if (node.opType === 'SDPA' || node.opType === 'CrossSDPA') attentionDropoutProbability(node);
    }
    await this._ensureForwardCompiled();
    this.graph.assertTopologyRevision?.(
      this.executor.compiledTopologyRevision as number,
      'WebGPU training',
    );
    this._resetGradientsForTopology();
    const topologyRevision = this.graph.topologyRevision || 0;
    let expectedWeightRevision = this.graph.weightRevision || 0;
    if (!dropout || typeof dropout !== 'object' || Array.isArray(dropout)) {
      throw new Error('WebGPUAutograd.trainStep dropout options must be an object.');
    }
    const accumulationIndex = gradientAccumulationIndex(this.graph);
    const defaultDropoutCounter = Number(
      (BigInt(nextTrainingStep - 1) * BigInt(gradientAccumulationSteps) + BigInt(accumulationIndex + 1)) & 0xffffffffn,
    );
    const trainingDropout = dropoutContext({
      seed: dropout.seed ?? 0,
      counter: dropout.counter ?? defaultDropoutCounter,
      shapeSignature,
    });
    const pipelineOverrides = await this._buildDropoutForwardOverrides(trainingDropout);
    if (typeof this.executor.executeTraining === 'function') {
      await this.executor.executeTraining(inputs, pipelineOverrides);
    } else {
      if (pipelineOverrides.size > 0) {
        throw new Error('WebGPU Trainer requires a Trainer-owned forward executor for Dropout.');
      }
      await this.executor.execute(inputs, { adapter: null });
    }
    this._assertStepState(topologyRevision, expectedWeightRevision);

    for (const [name, buffer] of this.gradientBuffers) {
      const tensor = this.graph.getTensor(name);
      this.device.queue.writeBuffer(buffer, 0, new Uint8Array(Math.max(4, tensor!.sizeBytes)));
    }
    const logitsGradients = new Map<string, Float32Array>();
    const lossMetrics: WebGPULossMetric[] = [];
    let loss = 0;
    let correct = 0;
    let count = 0;
    for (const descriptor of lossDescriptors) {
      const logits = this.graph.getTensor(descriptor.logitsTensor);
      if (!logits || logits.dtype !== 'float32') {
        throw new Error(`Logits tensor '${descriptor.logitsTensor}' must be F32.`);
      }
      const logitsValues = await this._readFloatBuffer(this._buffer(logits), logits.sizeBytes);
      this._assertStepState(topologyRevision, expectedWeightRevision);
      const metric = crossEntropyGradient(logits, logitsValues, descriptor);
      let gradient = logitsGradients.get(logits.name);
      if (!gradient) {
        gradient = new Float32Array(metric.gradient.length);
        logitsGradients.set(logits.name, gradient);
      }
      addGradient(gradient, metric.gradient);
      loss += metric.loss;
      correct += metric.correct;
      count += metric.examples;
      lossMetrics.push({
        name: descriptor.name,
        logitsTensor: descriptor.logitsTensor,
        loss: metric.loss,
        correct: metric.correct,
        examples: metric.examples,
        normalizer: metric.normalizer,
        weight: descriptor.weight,
      });
    }
    const gradients = new Map<string, Float32Array>();
    const hasContribution = lossMetrics.some((metric) => metric.examples > 0 && metric.weight > 0);
    if (hasContribution) {
      for (const [name, gradient] of logitsGradients) {
        this.device.queue.writeBuffer(this._gradient(this.graph.getTensor(name))!, 0, gradient);
      }

      const dispatches = await this._buildBackwardDispatches(trainingDropout);
      this._assertStepState(topologyRevision, expectedWeightRevision);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass({ label: 'VolvoxAI_Backward' });
      for (const dispatch of dispatches) {
        pass.setPipeline(dispatch.pipeline);
        pass.setBindGroup(0, dispatch.bindGroup);
        pass.dispatchWorkgroups(...dispatch.workgroupCount);
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      for (const tensor of trainables) {
        const gradientBuffer = this.gradientBuffers.get(tensor.name);
        if (!gradientBuffer) throw new Error(`No gradient reached trainable tensor '${tensor.name}'.`);
        const gradient = await this._readFloatBuffer(gradientBuffer, tensor.sizeBytes);
        this._assertStepState(topologyRevision, expectedWeightRevision);
        for (let index = 0; index < gradient.length; index++) {
          if (!Number.isFinite(gradient[index])) {
            throw new Error(`Gradient for trainable tensor '${tensor.name}' is non-finite at index ${index}.`);
          }
        }
        gradients.set(tensor.name, gradient);
      }
      this._assertStepState(topologyRevision, expectedWeightRevision);
    } else {
      for (const tensor of trainables) {
        gradients.set(tensor.name, new Float32Array((tensor.buffer as Float32Array).length));
      }
    }
    const accumulationSignature = JSON.stringify({
      backend: 'webgpu',
      activationShapeSignature: shapeSignature,
      tacticSignature,
      topologyRevision,
      weightRevision: expectedWeightRevision,
      trainableTensors,
      mode: normalizedUpdateMode,
      optimizerDescriptor,
      nextTrainingStep,
      losses: lossDescriptors.map(({ name, logitsTensor, weight, normalizer }) => ({
        name, logitsTensor, weight, normalizer,
      })),
    });
    const accumulationAggregation = lossDescriptors.length > 1 ||
      lossDescriptors.every((descriptor) => descriptor.normalizer != null)
      ? 'sum'
      : 'mean_by_examples';
    const accumulation = accumulateGradients(this.graph, {
      backend: 'webgpu',
      gradients,
      examples: count,
      loss,
      correct,
      trainableNames: trainableTensors,
      accumulationSteps: gradientAccumulationSteps,
      flush: flushGradientAccumulation,
      signature: accumulationSignature,
      aggregation: accumulationAggregation,
      hasContribution,
      metrics: lossMetrics as any,
    });
    if (!accumulation.apply) {
      return {
        loss: accumulation.loss,
        correct: accumulation.correct,
        examples: accumulation.examples,
        losses: accumulation.metrics ?? lossMetrics,
        updatedTensors: [],
        gradients: accumulation.gradients ?? gradients,
        accumulating: !accumulation.complete,
        accumulationStep: accumulation.microbatches,
        gradientAccumulationSteps,
      };
    }

    const clipping = clipGradientsGlobal(
      accumulation.gradients,
      optimizerDescriptor.optimizer.maxGradNorm,
    );
    const prepared: Array<[Tensor, Float32Array]> = trainables.map((tensor) => [
      tensor,
      accumulation.gradients.get(tensor.name)!,
    ]);
    const updatedTensors: Tensor[] = [];
    for (const [tensor, gradient] of prepared) {
      const updated = this.graph.applyTensorUpdate(tensor.name, gradient, {
        ...optimizer,
        maxGradNorm: 0,
        mode: normalizedUpdateMode,
        step: nextTrainingStep,
      });
      expectedWeightRevision = this.graph.weightRevision || 0;
      updatedTensors.push(updated);
      this._uploadUpdatedTensor(updated);
    }
    this.executor.compiledWeightRevision = this.graph.weightRevision || 0;
    this.graph.trainingStep = nextTrainingStep;
    if (normalizedUpdateMode !== 'adamw') this.graph.optimizerState = null;
    this.graph.optimizerDescriptor = optimizerDescriptor;
    synchronizeTrainingGraphState(this.graph);
    return {
      loss: accumulation.loss,
      correct: accumulation.correct,
      examples: accumulation.examples,
      losses: accumulation.metrics ?? lossMetrics,
      updatedTensors,
      gradients: accumulation.gradients,
      accumulating: false,
      accumulationStep: accumulation.microbatches,
      gradientAccumulationSteps,
      globalGradNorm: clipping.norm,
      gradientScale: clipping.scale,
    };
  }

  dispose() {
    if (this._training) throw new Error('Cannot dispose WebGPUAutograd during a trainStep.');
    for (const buffer of this.gradientBuffers.values()) buffer.destroy?.();
    for (const buffer of this.dummyBuffers.values()) buffer.destroy?.();
    this.gradientBuffers.clear();
    this.gradientCapacityBytes.clear();
    this.gradientTopologyRevision = this.graph.topologyRevision || 0;
    this.dummyBuffers.clear();
    this.pipelineCache.clear();
    this.moduleCache.clear();
    this.backendPlanCache.clear();
    this.backendTopologyIdentity = null;
    this.currentBackendPlanKey = null;
    this.currentBackendPlanRecipe = null;
    if (this.ownsExecutor) this.executor.dispose?.();
  }
}
