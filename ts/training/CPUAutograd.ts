import { CPUEngine } from '../backends/CPUEngine.js';
import type { Graph } from '../core/Graph.js';
import type { Tensor } from '../core/Tensor.js';
import { normalizeSpatialPair } from '../ops/spatialParameters.js';
import { geluApproximation, geluDerivative } from '../ops/gELU.js';
import { _cpuAttentionMask } from '../ops/sDPA.js';
import { _cpuDropout, dropoutContext, dropoutMultiplier } from '../ops/dropout.js';
import { attentionDropout, attentionProbabilityIndex } from '../ops/attentionDropout.js';
import {
  canonicalOptimizerDescriptor,
  normalizeTrainingUpdateMode,
  validateOptimizerOptions,
} from './TrainingOptimizer.js';
import {
  ensureTrainingGraphState,
  synchronizeTrainingGraphState,
  type StatefulTrainingGraph,
  type TrainingTensorUpdateOptions,
} from './TrainingGraph.js';
import {
  addGradient,
  crossEntropyGradient,
  normalizeCrossEntropyLosses,
} from './TrainingLosses.js';
import {
  accumulateGradients,
  clipGradientsGlobal,
  gradientAccumulationIndex,
  resetGradientAccumulation as resetPendingGradients,
} from './GradientAccumulation.js';
import type { RuntimeTypedArray } from '../types.js';
import type { TrainingStepOptions } from './TrainingStep.js';

type TrainingDropoutContext = ReturnType<typeof dropoutContext>;
type CrossEntropyGradientResult = ReturnType<typeof crossEntropyGradient>;
type GradientMap = Map<string, Float32Array>;

interface AcceleratedTrainingExecution {
  backend?: string;
  preflight?(graph: StatefulTrainingGraph, options: TrainingStepOptions): void | Promise<void>;
  forward?(
    graph: StatefulTrainingGraph,
    inputs: Record<string, RuntimeTypedArray>,
    options: { dropout: TrainingDropoutContext },
  ): void | Promise<void>;
  crossEntropyGradient?(
    logits: Tensor,
    values: Float32Array,
    descriptor: Parameters<typeof crossEntropyGradient>[2],
  ): CrossEntropyGradientResult;
  addGradient?(destination: Float32Array, source: Float32Array): unknown;
  backwardNode?(context: any): boolean | Promise<boolean>;
  allFinite?(gradient: Float32Array): boolean;
  clipGradients?(gradients: GradientMap, maxGradNorm: number): { norm: number; scale: number };
  applyTensorUpdate?(
    graph: StatefulTrainingGraph,
    tensor: Tensor,
    gradient: Float32Array,
    options: TrainingTensorUpdateOptions,
  ): Tensor;
}

interface TrainingLossMetric {
  name: string;
  logitsTensor: string;
  loss: number;
  correct: number;
  examples: number;
  normalizer: number;
  weight: number;
}

function firstOutput(node) {
  return node.outputs.out || Object.values(node.outputs || {})[0];
}

function nodeInput(node) {
  return node.inputs.input || node.inputs.x || node.inputs.data || Object.values(node.inputs || {})[0];
}

function gradFor(grads, tensor) {
  if (!tensor || tensor.dtype !== "float32") return null;
  let grad = grads.get(tensor.name);
  if (!grad) {
    grad = new Float32Array(tensor.buffer.length);
    grads.set(tensor.name, grad);
  }
  return grad;
}

function backwardFailure(node, detail) {
  const id = node?.id != null ? ` '${node.id}'` : "";
  throw new Error(`CPU backward for ${node?.opType || "unknown"} node${id} is unsupported: ${detail}.`);
}

function elementCount(shape) {
  return shape.reduce((count, dim) => count * dim, 1);
}

function broadcastIndexer(node, tensor, output) {
  if (!tensor?.shape || !output?.shape || !tensor.buffer) {
    backwardFailure(node, "missing broadcast tensor metadata or storage");
  }
  if (elementCount(tensor.shape) !== tensor.buffer.length ||
      elementCount(output.shape) !== output.buffer.length) {
    backwardFailure(node, "broadcast tensor storage does not match its shape");
  }
  if (tensor.shape.length > output.shape.length) {
    backwardFailure(node, "an input rank exceeds the output rank");
  }
  const rank = output.shape.length;
  const offset = rank - tensor.shape.length;
  const strides = new Array(tensor.shape.length);
  let stride = 1;
  for (let dim = tensor.shape.length - 1; dim >= 0; dim--) {
    strides[dim] = stride;
    stride *= tensor.shape[dim];
  }
  for (let dim = 0; dim < rank; dim++) {
    const inputDim = dim < offset ? 1 : tensor.shape[dim - offset];
    if (inputDim !== 1 && inputDim !== output.shape[dim]) {
      backwardFailure(node, `input shape [${tensor.shape}] cannot broadcast to [${output.shape}]`);
    }
  }
  return (outputIndex) => {
    let remaining = outputIndex;
    let inputIndex = 0;
    for (let dim = rank - 1; dim >= 0; dim--) {
      const coordinate = remaining % output.shape[dim];
      remaining = Math.floor(remaining / output.shape[dim]);
      if (dim >= offset && tensor.shape[dim - offset] !== 1) {
        inputIndex += coordinate * strides[dim - offset];
      }
    }
    return inputIndex;
  };
}

function addBroadcast(node, tensor, output, dst, src) {
  const index = broadcastIndexer(node, tensor, output);
  for (let i = 0; i < src.length; i++) {
    dst[index(i)] += src[i];
  }
}

function concatInputEntries(node) {
  const preferred = ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const seen = new Set<string>();
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
      const li = /^input(\d+)$/.exec(left);
      const ri = /^input(\d+)$/.exec(right);
      if (li && ri) return Number(li[1]) - Number(ri[1]);
      return left.localeCompare(right);
    }));
  return entries;
}

function backwardConcat(node, go, grads) {
  const output = firstOutput(node);
  const entries = concatInputEntries(node);
  const rank = output?.shape?.length ?? 0;
  let axis = node.params?.axis ?? 0;
  if (axis < 0) axis += rank;
  if (!output || output.dtype !== "float32" || !Number.isInteger(axis) || axis < 0 || axis >= rank || entries.length === 0) {
    backwardFailure(node, "Concat requires an F32 output, inputs, and a valid axis");
  }
  const inner = output.shape.slice(axis + 1).reduce((count, dimension) => count * dimension, 1);
  const outer = output.shape.slice(0, axis).reduce((count, dimension) => count * dimension, 1);
  const outputAxis = output.shape[axis];
  let totalAxis = 0;
  for (const [, input] of entries) {
    if (input.shape.length !== rank ||
        input.shape.some((dimension, index) => index !== axis && dimension !== output.shape[index])) {
      backwardFailure(node, "Concat input shapes are incompatible");
    }
    totalAxis += input.shape[axis];
  }
  if (totalAxis !== outputAxis) backwardFailure(node, "Concat input axes do not match the output");

  let axisOffset = 0;
  let differentiable = false;
  for (const [, input] of entries) {
    const inputAxis = input.shape[axis];
    if (input.dtype === "float32") {
      const gi = gradFor(grads, input);
      const block = inputAxis * inner;
      for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
        const inputBase = outerIndex * block;
        const outputBase = (outerIndex * outputAxis + axisOffset) * inner;
        for (let offset = 0; offset < block; offset++) {
          let gradient = go[outputBase + offset];
          if (node.params?.sigmoid) {
            const sigmoid = 1 / (1 + Math.exp(-(input.buffer as Float32Array)[inputBase + offset]));
            gradient *= sigmoid * (1 - sigmoid);
          }
          gi[inputBase + offset] += gradient;
        }
      }
      differentiable = true;
    }
    axisOffset += inputAxis;
  }
  if (!differentiable) backwardFailure(node, "Concat has no differentiable F32 input");
}

function backwardEmbedding(node, go, grads) {
  const tokens = node.inputs.input;
  const weight = node.inputs.weight;
  const out = firstOutput(node);
  if (tokens?.dtype !== "int32" || !(tokens.buffer instanceof Int32Array) ||
      weight?.dtype !== "float32" || !weight.buffer || !out) {
    backwardFailure(node, "embedding requires int32 token storage and an F32 table");
  }
  const gw = gradFor(grads, weight);
  const width = out.shape[out.shape.length - 1];
  const rows = out.buffer.length / width;
  for (let row = 0; row < rows; row++) {
    const token = tokens.buffer[row];
    if (!Number.isInteger(token) || token < 0 || token * width + width > gw.length) {
      backwardFailure(node, `token id ${token} is outside the embedding table`);
    }
    for (let col = 0; col < width; col++) gw[token * width + col] += go[row * width + col];
  }
}

function backwardLayerNorm(node, go, grads) {
  const input = nodeInput(node);
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  if (input?.dtype !== "float32" || weight?.dtype !== "float32" ||
      !input.buffer || !weight.buffer) {
    backwardFailure(node, "LayerNorm requires F32 input and scale storage");
  }
  const gi = gradFor(grads, input);
  const gw = gradFor(grads, weight);
  const gb = bias?.dtype === "float32" ? gradFor(grads, bias) : null;
  const width = node.params?.d_model ?? input.shape[input.shape.length - 1];
  const rows = input.buffer.length / width;
  const eps = node.params?.eps ?? 1e-6;
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let mean = 0;
    for (let col = 0; col < width; col++) mean += input.buffer[offset + col];
    mean /= width;
    let variance = 0;
    for (let col = 0; col < width; col++) {
      const centered = input.buffer[offset + col] - mean;
      variance += centered * centered;
    }
    const invStd = 1 / Math.sqrt(variance / width + eps);
    let sumDxhat = 0;
    let sumDxhatXhat = 0;
    for (let col = 0; col < width; col++) {
      const xhat = (input.buffer[offset + col] - mean) * invStd;
      const dxhat = go[offset + col] * weight.buffer[col];
      gw[col] += go[offset + col] * xhat;
      if (gb) gb[col] += go[offset + col];
      sumDxhat += dxhat;
      sumDxhatXhat += dxhat * xhat;
    }
    for (let col = 0; col < width; col++) {
      const xhat = (input.buffer[offset + col] - mean) * invStd;
      const dxhat = go[offset + col] * weight.buffer[col];
      gi[offset + col] += invStd * (dxhat - sumDxhat / width - xhat * sumDxhatXhat / width);
    }
  }
}

function backwardGroupNorm(node, go, grads) {
  const input = nodeInput(node);
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = firstOutput(node);
  if (!input || !weight || !bias || !output || input.dtype !== "float32" ||
      weight.dtype !== "float32" || bias.dtype !== "float32" || output.dtype !== "float32" ||
      !(input.buffer instanceof Float32Array) || !(weight.buffer instanceof Float32Array) ||
      !(bias.buffer instanceof Float32Array)) {
    backwardFailure(node, "GroupNorm requires F32 input, weight, bias, and output storage");
  }
  if (input.shape.length !== 4 || output.shape.length !== 4 ||
      input.shape.some((dimension, index) => output.shape[index] !== dimension) ||
      go.length !== elementCount(output.shape)) {
    backwardFailure(node, "GroupNorm requires matching rank-4 NHWC input, output, and output-gradient shapes");
  }

  const [batch, height, width, channels] = input.shape;
  const numGroups = node.params?.num_groups;
  const eps = node.params?.eps ?? 1e-5;
  if (!Number.isInteger(numGroups) || numGroups <= 0 || channels % numGroups !== 0) {
    backwardFailure(node, `num_groups must be a positive divisor of ${channels} channels`);
  }
  if (weight.shape.length !== 1 || weight.shape[0] !== channels ||
      bias.shape.length !== 1 || bias.shape[0] !== channels ||
      weight.buffer.length !== channels || bias.buffer.length !== channels) {
    backwardFailure(node, `weight and bias must have shape [${channels}]`);
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    backwardFailure(node, "eps must be a positive finite number");
  }

  const gi = gradFor(grads, input);
  const gw = gradFor(grads, weight);
  const gb = gradFor(grads, bias);
  const channelsPerGroup = channels / numGroups;
  const spatialSize = height * width;
  const valuesPerGroup = spatialSize * channelsPerGroup;
  const sampleStride = spatialSize * channels;

  for (let n = 0; n < batch; n++) {
    const sampleOffset = n * sampleStride;
    for (let group = 0; group < numGroups; group++) {
      const firstChannel = group * channelsPerGroup;
      let mean = 0;
      for (let spatial = 0; spatial < spatialSize; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          mean += input.buffer[offset + localChannel];
        }
      }
      mean /= valuesPerGroup;

      let variance = 0;
      for (let spatial = 0; spatial < spatialSize; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const centered = input.buffer[offset + localChannel] - mean;
          variance += centered * centered;
        }
      }
      const invStd = 1 / Math.sqrt(variance / valuesPerGroup + eps);

      let sumDxhat = 0;
      let sumDxhatXhat = 0;
      for (let spatial = 0; spatial < spatialSize; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const channel = firstChannel + localChannel;
          const index = offset + localChannel;
          const xhat = (input.buffer[index] - mean) * invStd;
          const dxhat = go[index] * weight.buffer[channel];
          sumDxhat += dxhat;
          sumDxhatXhat += dxhat * xhat;
          gw[channel] += go[index] * xhat;
          gb[channel] += go[index];
        }
      }

      for (let spatial = 0; spatial < spatialSize; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const channel = firstChannel + localChannel;
          const index = offset + localChannel;
          const xhat = (input.buffer[index] - mean) * invStd;
          const dxhat = go[index] * weight.buffer[channel];
          gi[index] += invStd * (dxhat - sumDxhat / valuesPerGroup - xhat * sumDxhatXhat / valuesPerGroup);
        }
      }
    }
  }
}

function backwardRMSNorm(node, go, grads) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight;
  if (input?.dtype !== "float32" || weight?.dtype !== "float32" ||
      !input.buffer || !weight.buffer) {
    backwardFailure(node, "RMSNorm requires F32 input and scale storage");
  }
  const gi = gradFor(grads, input);
  const gw = gradFor(grads, weight);
  const width = node.params?.d_model ?? input.shape[input.shape.length - 1];
  const rows = input.buffer.length / width;
  const eps = node.params?.eps ?? 1e-6;
  for (let row = 0; row < rows; row++) {
    const offset = row * width;
    let meanSquare = 0;
    for (let col = 0; col < width; col++) {
      const x = input.buffer[offset + col];
      meanSquare += x * x;
    }
    meanSquare /= width;
    const invRms = 1 / Math.sqrt(meanSquare + eps);
    let projection = 0;
    for (let col = 0; col < width; col++) {
      const x = input.buffer[offset + col];
      const dxScaled = go[offset + col] * weight.buffer[col];
      gw[col] += go[offset + col] * x * invRms;
      projection += dxScaled * x;
    }
    const correction = projection * invRms * invRms * invRms / width;
    for (let col = 0; col < width; col++) {
      gi[offset + col] += go[offset + col] * weight.buffer[col] * invRms - input.buffer[offset + col] * correction;
    }
  }
}

function backwardActivation(node, go, grads) {
  const input = nodeInput(node);
  const out = firstOutput(node);
  if (input?.dtype !== "float32" || !input.buffer || !out?.buffer || input.buffer.length !== go.length) {
    backwardFailure(node, "activation requires matching F32 input/output storage");
  }
  const gi = gradFor(grads, input);
  const alpha = node.params?.alpha !== undefined ? node.params.alpha : 0.01;
  for (let i = 0; i < go.length; i++) {
    const x = input.buffer[i];
    let derivative;
    if (node.opType === "ReLU") {
      derivative = x > 0 ? 1 : 0;
    } else if (node.opType === "LeakyReLU") {
      derivative = x > 0 ? 1 : alpha;
    } else if (node.opType === "GELU") {
      derivative = geluDerivative(x, geluApproximation(node));
    } else if (node.opType === "SiLU" || node.opType === "Swish") {
      const sigmoid = 1 / (1 + Math.exp(-x));
      derivative = sigmoid + x * sigmoid * (1 - sigmoid);
    } else if (node.opType === "Sigmoid") {
      const y = out.buffer[i];
      derivative = y * (1 - y);
    } else if (node.opType === "Tanh") {
      const y = out.buffer[i];
      derivative = 1 - y * y;
    } else if (node.opType === "HardSigmoid") {
      derivative = x > -3 && x < 3 ? 1 / 6 : 0;
    } else if (node.opType === "HardSwish") {
      derivative = x <= -3 ? 0 : (x >= 3 ? 1 : x / 3 + 0.5);
    } else if (node.opType === "Clip") {
      const minimum = node.params?.min ?? -1e9, maximum = node.params?.max ?? 1e9;
      derivative = x > minimum && x < maximum ? 1 : 0;
    } else {
      continue;
    }
    gi[i] += go[i] * derivative;
  }
}

function backwardSDPA(
  node,
  go,
  grads,
  dropoutTraining: TrainingDropoutContext | null = null,
  nodeIndex = 0,
) {
  const qkv = node.inputs.qkv;
  const out = firstOutput(node);
  const rank = qkv?.shape?.length;
  const batch = rank === 2 ? 1 : qkv?.shape?.[0];
  const outputBatch = out?.shape?.length === 2 ? 1 : out?.shape?.[0];
  if (qkv?.dtype !== "float32" || !qkv.buffer || !out?.buffer ||
      (rank !== 2 && rank !== 3) || out.shape.length !== rank || batch !== outputBatch) {
    backwardFailure(node, "SDPA requires compatible rank-2 or rank-3 F32 QKV/output storage");
  }
  const gqkv = gradFor(grads, qkv);
  const seq = qkv.shape[rank - 2];
  const model = out.shape[out.shape.length - 1];
  const heads = node.params?.heads || 8;
  const headDim = model / heads;
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      qkv.shape[rank - 1] !== model * 3 || out.shape[rank - 2] !== seq) {
    backwardFailure(node, "SDPA has incompatible head or tensor dimensions");
  }
  const scale = node.params?.scale !== undefined ? node.params.scale : 1 / Math.sqrt(headDim);
  const causal = node.params?.causal !== false;
  const keeps = _cpuAttentionMask(node, batch, seq, seq);
  const probabilityDropout = attentionDropout(node, dropoutTraining, nodeIndex);
  const stride = model * 3;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qkvBase = batchIndex * seq * stride;
    const outputBase = batchIndex * seq * model;
    for (let head = 0; head < heads; head++) {
      for (let query = 0; query < seq; query++) {
        const probabilities = new Float64Array(seq);
        let max = -Infinity;
        let visibleKeys = 0;
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          let score = 0;
          for (let dim = 0; dim < headDim; dim++) {
            const qi = qkvBase + query * stride + head * headDim + dim;
            const ki = qkvBase + key * stride + model + head * headDim + dim;
            score += qkv.buffer[qi] * qkv.buffer[ki];
          }
          score *= scale;
          probabilities[key] = score;
          if (score > max) max = score;
          visibleKeys++;
        }
        if (visibleKeys === 0) continue;

        let denominator = 0;
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          probabilities[key] = Math.exp(probabilities[key] - max);
          denominator += probabilities[key];
        }
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          probabilities[key] /= denominator;
        }

        const dProbability = new Float64Array(seq);
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          const dropoutScale = probabilityDropout(attentionProbabilityIndex(
            batchIndex, head, query, key, heads, seq, seq,
          ));
          let value = 0;
          for (let dim = 0; dim < headDim; dim++) {
            const goValue = go[outputBase + query * model + head * headDim + dim];
            const vi = qkvBase + key * stride + model * 2 + head * headDim + dim;
            value += goValue * qkv.buffer[vi];
            gqkv[vi] += probabilities[key] * dropoutScale * goValue;
          }
          dProbability[key] = value * dropoutScale;
        }
        let dot = 0;
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          dot += probabilities[key] * dProbability[key];
        }
        for (let key = 0; key < seq; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          const dScore = probabilities[key] * (dProbability[key] - dot);
          for (let dim = 0; dim < headDim; dim++) {
            const qi = qkvBase + query * stride + head * headDim + dim;
            const ki = qkvBase + key * stride + model + head * headDim + dim;
            gqkv[qi] += dScore * scale * qkv.buffer[ki];
            gqkv[ki] += dScore * scale * qkv.buffer[qi];
          }
        }
      }
    }
  }
}

function backwardCrossSDPA(
  node,
  go,
  grads,
  dropoutTraining: TrainingDropoutContext | null = null,
  nodeIndex = 0,
) {
  const q = node.inputs.q;
  const k = node.inputs.k;
  const v = node.inputs.v;
  const out = firstOutput(node);
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const batchOf = (tensor) => tensor?.shape?.length === 2 ? 1 : tensor?.shape?.[0];
  if (q?.dtype !== "float32" || k?.dtype !== "float32" || v?.dtype !== "float32" ||
      !q.buffer || !k.buffer || !v.buffer || !out?.buffer ||
      (rank !== 2 && rank !== 3) || k.shape.length !== rank || v.shape.length !== rank ||
      out.shape.length !== rank || batchOf(k) !== batch || batchOf(v) !== batch || batchOf(out) !== batch) {
    backwardFailure(node, "CrossSDPA requires compatible rank-2 or rank-3 F32 Q/K/V/output storage");
  }
  const gq = gradFor(grads, q);
  const gk = gradFor(grads, k);
  const gv = gradFor(grads, v);
  const seqQ = q.shape[rank - 2];
  const seqKV = k.shape[rank - 2];
  const model = out.shape[out.shape.length - 1];
  const heads = node.params?.heads || 8;
  const headDim = model / heads;
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      q.shape[rank - 1] !== model || k.shape[rank - 1] !== model || v.shape[rank - 1] !== model ||
      v.shape[rank - 2] !== seqKV || out.shape[rank - 2] !== seqQ) {
    backwardFailure(node, "CrossSDPA has incompatible head or tensor dimensions");
  }
  const scale = node.params?.scale ?? 1 / Math.sqrt(headDim);
  const causal = node.params?.causal === true;
  const keeps = _cpuAttentionMask(node, batch, seqQ, seqKV);
  const probabilityDropout = attentionDropout(node, dropoutTraining, nodeIndex);
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qBase = batchIndex * seqQ * model;
    const kvBase = batchIndex * seqKV * model;
    const outputBase = qBase;
    for (let head = 0; head < heads; head++) {
      for (let query = 0; query < seqQ; query++) {
        const probabilities = new Float64Array(seqKV);
        let max = -Infinity;
        let visibleKeys = 0;
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          let score = 0;
          for (let dim = 0; dim < headDim; dim++) {
            const qi = qBase + query * model + head * headDim + dim;
            const ki = kvBase + key * model + head * headDim + dim;
            score += q.buffer[qi] * k.buffer[ki];
          }
          score *= scale;
          probabilities[key] = score;
          if (score > max) max = score;
          visibleKeys++;
        }
        if (visibleKeys === 0) continue;

        let denominator = 0;
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          probabilities[key] = Math.exp(probabilities[key] - max);
          denominator += probabilities[key];
        }
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          probabilities[key] /= denominator;
        }
        const dProbability = new Float64Array(seqKV);
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          const dropoutScale = probabilityDropout(attentionProbabilityIndex(
            batchIndex, head, query, key, heads, seqQ, seqKV,
          ));
          let value = 0;
          for (let dim = 0; dim < headDim; dim++) {
            const oi = outputBase + query * model + head * headDim + dim;
            const vi = kvBase + key * model + head * headDim + dim;
            value += go[oi] * v.buffer[vi];
            gv[vi] += probabilities[key] * dropoutScale * go[oi];
          }
          dProbability[key] = value * dropoutScale;
        }
        let dot = 0;
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          dot += probabilities[key] * dProbability[key];
        }
        for (let key = 0; key < seqKV; key++) {
          if ((causal && key > query) || !keeps(batchIndex, query, key)) continue;
          const dScore = probabilities[key] * (dProbability[key] - dot);
          for (let dim = 0; dim < headDim; dim++) {
            const qi = qBase + query * model + head * headDim + dim;
            const ki = kvBase + key * model + head * headDim + dim;
            gq[qi] += dScore * scale * k.buffer[ki];
            gk[ki] += dScore * scale * q.buffer[qi];
          }
        }
      }
    }
  }
}

function backwardCrossAttention(node, go, grads) {
  const q = node.inputs.q;
  const kv = node.inputs.kv;
  const weight = node.inputs.weight;
  const scale = node.inputs.scale || null;
  const bias = node.inputs.bias || null;
  const out = firstOutput(node);
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const batchOf = (tensor) => tensor?.shape?.length === 2 ? 1 : tensor?.shape?.[0];
  if (q?.dtype !== 'float32' || kv?.dtype !== 'float32' || weight?.dtype !== 'float32' ||
      !q.buffer || !kv.buffer || !weight.buffer || !out?.buffer ||
      (scale && (scale.dtype !== 'float32' || !scale.buffer)) ||
      (bias && (bias.dtype !== 'float32' || !bias.buffer)) ||
      (rank !== 2 && rank !== 3) || kv.shape.length !== rank || out.shape.length !== rank ||
      batchOf(kv) !== batch || batchOf(out) !== batch) {
    backwardFailure(node, 'CrossAttention requires compatible rank-2 or rank-3 F32 Q/KV/projection/output storage');
  }
  const seqQ = q.shape[rank - 2];
  const seqKV = kv.shape[rank - 2];
  const dModel = out.shape[rank - 1];
  const heads = node.params?.heads || 8;
  const headDim = dModel / heads;
  const projectionWidth = 3 * dModel;
  if (!Number.isInteger(heads) || heads <= 0 || !Number.isInteger(headDim) || headDim <= 0 ||
      q.shape[rank - 1] !== dModel || kv.shape[rank - 1] !== dModel ||
      out.shape[rank - 2] !== seqQ || weight.shape.length !== 2 ||
      weight.shape[0] !== projectionWidth || weight.shape[1] !== dModel ||
      (scale && elementCount(scale.shape) !== projectionWidth) ||
      (bias && elementCount(bias.shape) !== projectionWidth)) {
    backwardFailure(node, 'CrossAttention has incompatible projection or attention dimensions');
  }
  const gq = gradFor(grads, q);
  const gkv = gradFor(grads, kv);
  const gw = gradFor(grads, weight);
  const gs = scale ? gradFor(grads, scale) : null;
  const gb = bias ? gradFor(grads, bias) : null;
  const attentionScale = 1 / Math.sqrt(headDim);
  const affineScale = (column) => scale ? scale.buffer[column] : 1;

  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qBase = batchIndex * seqQ * dModel;
    const kvBase = batchIndex * seqKV * dModel;
    for (let head = 0; head < heads; head++) {
      for (let query = 0; query < seqQ; query++) {
        const qRaw = new Float64Array(headDim);
        const qProjected = new Float64Array(headDim);
        for (let dimension = 0; dimension < headDim; dimension++) {
          const column = head * headDim + dimension;
          let raw = 0;
          for (let inputDimension = 0; inputDimension < dModel; inputDimension++) {
            raw += q.buffer[qBase + query * dModel + inputDimension] *
              weight.buffer[column * dModel + inputDimension];
          }
          qRaw[dimension] = raw;
          qProjected[dimension] = raw * affineScale(column) + (bias ? bias.buffer[column] : 0);
        }

        const kRaw = new Float64Array(seqKV * headDim);
        const kProjected = new Float64Array(seqKV * headDim);
        const vRaw = new Float64Array(seqKV * headDim);
        const vProjected = new Float64Array(seqKV * headDim);
        for (let key = 0; key < seqKV; key++) {
          for (let dimension = 0; dimension < headDim; dimension++) {
            const kColumn = dModel + head * headDim + dimension;
            const vColumn = 2 * dModel + head * headDim + dimension;
            let kValue = 0;
            let vValue = 0;
            for (let inputDimension = 0; inputDimension < dModel; inputDimension++) {
              const source = kv.buffer[kvBase + key * dModel + inputDimension];
              kValue += source * weight.buffer[kColumn * dModel + inputDimension];
              vValue += source * weight.buffer[vColumn * dModel + inputDimension];
            }
            const offset = key * headDim + dimension;
            kRaw[offset] = kValue;
            vRaw[offset] = vValue;
            kProjected[offset] = kValue * affineScale(kColumn) + (bias ? bias.buffer[kColumn] : 0);
            vProjected[offset] = vValue * affineScale(vColumn) + (bias ? bias.buffer[vColumn] : 0);
          }
        }

        const probabilities = new Float64Array(seqKV);
        let maximum = -Infinity;
        for (let key = 0; key < seqKV; key++) {
          let score = 0;
          for (let dimension = 0; dimension < headDim; dimension++) {
            score += qProjected[dimension] * kProjected[key * headDim + dimension];
          }
          score *= attentionScale;
          probabilities[key] = score;
          if (score > maximum) maximum = score;
        }
        let denominator = 0;
        for (let key = 0; key < seqKV; key++) {
          probabilities[key] = Math.exp(probabilities[key] - maximum);
          denominator += probabilities[key];
        }
        for (let key = 0; key < seqKV; key++) probabilities[key] /= denominator;

        const dProbability = new Float64Array(seqKV);
        for (let key = 0; key < seqKV; key++) {
          let value = 0;
          for (let dimension = 0; dimension < headDim; dimension++) {
            value += go[qBase + query * dModel + head * headDim + dimension] *
              vProjected[key * headDim + dimension];
          }
          dProbability[key] = value;
        }
        let probabilityDot = 0;
        for (let key = 0; key < seqKV; key++) probabilityDot += probabilities[key] * dProbability[key];
        const dScore = new Float64Array(seqKV);
        for (let key = 0; key < seqKV; key++) {
          dScore[key] = probabilities[key] * (dProbability[key] - probabilityDot);
        }

        const dQProjected = new Float64Array(headDim);
        for (let dimension = 0; dimension < headDim; dimension++) {
          for (let key = 0; key < seqKV; key++) {
            dQProjected[dimension] += dScore[key] * attentionScale * kProjected[key * headDim + dimension];
          }
        }
        for (let dimension = 0; dimension < headDim; dimension++) {
          const column = head * headDim + dimension;
          const projectedGradient = dQProjected[dimension];
          const rawGradient = projectedGradient * affineScale(column);
          if (gs) gs[column] += projectedGradient * qRaw[dimension];
          if (gb) gb[column] += projectedGradient;
          for (let inputDimension = 0; inputDimension < dModel; inputDimension++) {
            const sourceIndex = qBase + query * dModel + inputDimension;
            const weightIndex = column * dModel + inputDimension;
            gq[sourceIndex] += rawGradient * weight.buffer[weightIndex];
            gw[weightIndex] += rawGradient * q.buffer[sourceIndex];
          }
        }

        for (let key = 0; key < seqKV; key++) {
          for (let dimension = 0; dimension < headDim; dimension++) {
            const kColumn = dModel + head * headDim + dimension;
            const vColumn = 2 * dModel + head * headDim + dimension;
            const kProjectedGradient = dScore[key] * attentionScale * qProjected[dimension];
            const vProjectedGradient = probabilities[key] *
              go[qBase + query * dModel + head * headDim + dimension];
            const kRawGradient = kProjectedGradient * affineScale(kColumn);
            const vRawGradient = vProjectedGradient * affineScale(vColumn);
            if (gs) {
              gs[kColumn] += kProjectedGradient * kRaw[key * headDim + dimension];
              gs[vColumn] += vProjectedGradient * vRaw[key * headDim + dimension];
            }
            if (gb) {
              gb[kColumn] += kProjectedGradient;
              gb[vColumn] += vProjectedGradient;
            }
            for (let inputDimension = 0; inputDimension < dModel; inputDimension++) {
              const sourceIndex = kvBase + key * dModel + inputDimension;
              const kWeightIndex = kColumn * dModel + inputDimension;
              const vWeightIndex = vColumn * dModel + inputDimension;
              const source = kv.buffer[sourceIndex];
              gkv[sourceIndex] += kRawGradient * weight.buffer[kWeightIndex] +
                vRawGradient * weight.buffer[vWeightIndex];
              gw[kWeightIndex] += kRawGradient * source;
              gw[vWeightIndex] += vRawGradient * source;
            }
          }
        }
      }
    }
  }
}

function backwardConv2D(node, go, grads) {
  const input = node.inputs.input;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = firstOutput(node);
  if (input?.dtype !== "float32" || weight?.dtype !== "float32" ||
      !input.buffer || !weight.buffer || !output?.buffer ||
      input.shape.length !== 4 || weight.shape.length !== 4 || output.shape.length !== 4) {
    backwardFailure(node, "Conv2D requires rank-4 NHWC/HWIO F32 storage");
  }
  const gi = gradFor(grads, input);
  const gw = gradFor(grads, weight);
  const gb = bias?.dtype === "float32" ? gradFor(grads, bias) : null;
  const [batch, inH, inW, inC] = input.shape;
  const [kernelH, kernelW] = weight.shape;
  const outH = output.shape[1];
  const outW = output.shape[2];
  const outC = output.shape[3];
  const [strideY, strideX] = normalizeSpatialPair(node.params?.stride, 1);
  const [padY, padX] = normalizeSpatialPair(node.params?.padding, 0);
  const pads = node.params?.pads || [padY, padX, padY, padX];
  const [dilationY, dilationX] = normalizeSpatialPair(node.params?.dilation, 1);
  const groups = node.params?.groups || 1;
  const groupOut = outC / groups;
  const groupIn = weight.shape[2];
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        for (let oc = 0; oc < outC; oc++) {
          const outIndex = ((batchIndex * outH + oh) * outW + ow) * outC + oc;
          let dy = go[outIndex];
          if (node.params?.relu === 1 && output.buffer[outIndex] <= 0) dy = 0;
          else if (node.params?.relu >= 2 && (output.buffer[outIndex] <= 0 || output.buffer[outIndex] >= 6)) dy = 0;
          if (gb) gb[oc] += dy;
          if (groups === inC) {
            const multiplier = outC / inC;
            const ic = Math.floor(oc / multiplier);
            const channelMultiplier = oc - ic * multiplier;
            for (let kh = 0; kh < kernelH; kh++) {
              for (let kw = 0; kw < kernelW; kw++) {
                const ih = oh * strideY + kh * dilationY - pads[0];
                const iw = ow * strideX + kw * dilationX - pads[1];
                if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                const inputIndex = ((batchIndex * inH + ih) * inW + iw) * inC + ic;
                const weightIndex = (((kh * kernelW + kw) * inC + ic) * multiplier) + channelMultiplier;
                gi[inputIndex] += dy * weight.buffer[weightIndex];
                gw[weightIndex] += dy * input.buffer[inputIndex];
              }
            }
          } else {
            const group = Math.floor(oc / groupOut);
            const inputStart = group * groupIn;
            for (let localInput = 0; localInput < groupIn; localInput++) {
              const ic = inputStart + localInput;
              for (let kh = 0; kh < kernelH; kh++) {
                for (let kw = 0; kw < kernelW; kw++) {
                  const ih = oh * strideY + kh * dilationY - pads[0];
                  const iw = ow * strideX + kw * dilationX - pads[1];
                  if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                  const inputIndex = ((batchIndex * inH + ih) * inW + iw) * inC + ic;
                  const weightIndex = (((kh * kernelW + kw) * groupIn + localInput) * outC) + oc;
                  gi[inputIndex] += dy * weight.buffer[weightIndex];
                  gw[weightIndex] += dy * input.buffer[inputIndex];
                }
              }
            }
          }
        }
      }
    }
  }
}

function backwardConv1D(node, go, grads) {
  const input = node.inputs.input;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = firstOutput(node);
  if (input?.dtype !== 'float32' || weight?.dtype !== 'float32' || !input.buffer || !weight.buffer ||
      !output?.buffer || input.shape.length !== 3 || weight.shape.length !== 3 || output.shape.length !== 3) {
    backwardFailure(node, 'Conv1D requires rank-3 NCL F32 input, weight, and output storage');
  }
  const [batch, inChannels, inputLength] = input.shape;
  const [outChannels, inputPerGroup, kernel] = weight.shape;
  const outputLength = output.shape[2];
  const groups = node.params?.groups ?? 1;
  const stride = normalizeSpatialPair(node.params?.stride, 1)[0];
  const padding = normalizeSpatialPair(node.params?.padding, 0)[0];
  if (!Number.isInteger(groups) || groups <= 0 || inChannels % groups || outChannels % groups ||
      inputPerGroup !== inChannels / groups || output.shape[0] !== batch || output.shape[1] !== outChannels) {
    backwardFailure(node, 'Conv1D has incompatible grouped NCL dimensions');
  }
  const gi = gradFor(grads, input), gw = gradFor(grads, weight);
  const gb = bias?.dtype === 'float32' ? gradFor(grads, bias) : null;
  const groupOut = outChannels / groups;
  for (let b = 0; b < batch; b++) for (let oc = 0; oc < outChannels; oc++) for (let ox = 0; ox < outputLength; ox++) {
    const outputIndex = (b * outChannels + oc) * outputLength + ox;
    const gradient = node.params?.relu && output.buffer[outputIndex] <= 0 ? 0 : go[outputIndex];
    if (gb) gb[oc] += gradient;
    const inputStart = Math.floor(oc / groupOut) * inputPerGroup;
    for (let localIc = 0; localIc < inputPerGroup; localIc++) for (let kk = 0; kk < kernel; kk++) {
      const ix = ox * stride + kk - padding;
      if (ix < 0 || ix >= inputLength) continue;
      const inputIndex = (b * inChannels + inputStart + localIc) * inputLength + ix;
      const weightIndex = (oc * inputPerGroup + localIc) * kernel + kk;
      gi[inputIndex] += gradient * weight.buffer[weightIndex];
      gw[weightIndex] += gradient * input.buffer[inputIndex];
    }
  }
}

function backwardConvTranspose2D(node, go, grads) {
  const input = node.inputs.input || node.inputs.x, weight = node.inputs.weight, bias = node.inputs.bias, output = firstOutput(node);
  if (input?.dtype !== 'float32' || weight?.dtype !== 'float32' || !input.buffer || !weight.buffer || !output?.buffer ||
      input.shape.length !== 4 || weight.shape.length !== 4 || output.shape.length !== 4) {
    backwardFailure(node, 'ConvTranspose2D requires NHWC input/output and rank-4 F32 weights');
  }
  const [batch, inHeight, inWidth, inChannels] = input.shape, [, outHeight, outWidth, outChannels] = output.shape;
  const [, weightOutChannels, kernelY, kernelX] = weight.shape;
  const [strideY, strideX] = normalizeSpatialPair(node.params?.stride, 1);
  const [padY, padX] = normalizeSpatialPair(node.params?.padding, 0);
  if (weight.shape[0] !== inChannels || weightOutChannels !== outChannels ||
      outHeight !== (inHeight - 1) * strideY + kernelY - 2 * padY || outWidth !== (inWidth - 1) * strideX + kernelX - 2 * padX) {
    backwardFailure(node, 'ConvTranspose2D has incompatible dimensions or output shape');
  }
  const gi = gradFor(grads, input), gw = gradFor(grads, weight), gb = bias?.dtype === 'float32' ? gradFor(grads, bias) : null;
  for (let b = 0; b < batch; b++) for (let iy = 0; iy < inHeight; iy++) for (let ix = 0; ix < inWidth; ix++) for (let ic = 0; ic < inChannels; ic++) for (let oc = 0; oc < outChannels; oc++) for (let ky = 0; ky < kernelY; ky++) for (let kx = 0; kx < kernelX; kx++) {
    const oy = iy * strideY - padY + ky, ox = ix * strideX - padX + kx;
    if (oy < 0 || oy >= outHeight || ox < 0 || ox >= outWidth) continue;
    const inputIndex = ((b * inHeight + iy) * inWidth + ix) * inChannels + ic;
    const weightIndex = ((ic * outChannels + oc) * kernelY + ky) * kernelX + kx;
    const outputIndex = ((b * outHeight + oy) * outWidth + ox) * outChannels + oc;
    gi[inputIndex] += go[outputIndex] * weight.buffer[weightIndex];
    gw[weightIndex] += go[outputIndex] * input.buffer[inputIndex];
    if (gb) gb[oc] += go[outputIndex];
  }
}

function backwardMoELinear(node, go, grads) {
  const input = node.inputs.input || node.inputs.x;
  const expertWeight = node.inputs.expert_weight || node.inputs.weight;
  const expertBias = node.inputs.expert_bias || node.inputs.bias;
  const routeIndices = node.inputs.route_indices || node.inputs.indices;
  const routeWeights = node.inputs.route_weights || node.inputs.weights;
  const output = firstOutput(node);
  if (input?.dtype !== "float32" || expertWeight?.dtype !== "float32" ||
      routeWeights?.dtype !== "float32" || !input.buffer || !expertWeight.buffer ||
      !routeWeights.buffer || !routeIndices?.buffer || !output?.buffer) {
    backwardFailure(node, "MoELinear requires F32 inputs, expert weights, and routing weights");
  }
  const gi = gradFor(grads, input);
  const gw = gradFor(grads, expertWeight);
  const gb = expertBias?.dtype === "float32" ? gradFor(grads, expertBias) : null;
  const gr = gradFor(grads, routeWeights);
  const dIn = input.shape[input.shape.length - 1];
  const dOut = output.shape[output.shape.length - 1];
  const rows = input.buffer.length / dIn;
  const topK = routeIndices.shape[routeIndices.shape.length - 1];
  for (let row = 0; row < rows; row++) {
    for (let slot = 0; slot < topK; slot++) {
      const routeOffset = row * topK + slot;
      const expert = Math.trunc(routeIndices.buffer[routeOffset]);
      if (!Number.isInteger(expert) || expert < 0 || expert >= expertWeight.shape[0]) {
        backwardFailure(node, `expert index ${routeIndices.buffer[routeOffset]} is out of range`);
      }
      const gate = routeWeights.buffer[routeOffset];
      const expertBase = expert * dIn * dOut;
      for (let col = 0; col < dOut; col++) {
        const dy = go[row * dOut + col];
        let expertValue = expertBias ? expertBias.buffer[expert * dOut + col] : 0;
        for (let dim = 0; dim < dIn; dim++) {
          const inputValue = input.buffer[row * dIn + dim];
          const weightIndex = expertBase + dim * dOut + col;
          expertValue += inputValue * expertWeight.buffer[weightIndex];
          gi[row * dIn + dim] += dy * gate * expertWeight.buffer[weightIndex];
          gw[weightIndex] += dy * gate * inputValue;
        }
        if (gb) gb[expert * dOut + col] += dy * gate;
        gr[routeOffset] += dy * expertValue;
      }
    }
  }
}

function backwardMoERouter(node, gateGrad, grads) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight || node.inputs.router_weight;
  const bias = node.inputs.bias;
  const indices = node.outputs.indices || node.outputs.expert_indices;
  const gates = node.outputs.weights || node.outputs.expert_weights;
  if (input?.dtype !== "float32" || weight?.dtype !== "float32" ||
      !input.buffer || !weight.buffer || !indices?.buffer || !gates?.buffer) {
    backwardFailure(node, "MoERouter requires F32 input/router storage and materialized routes");
  }
  const gi = gradFor(grads, input);
  const gw = gradFor(grads, weight);
  const gb = bias?.dtype === "float32" ? gradFor(grads, bias) : null;
  const dModel = input.shape[input.shape.length - 1];
  const rows = input.buffer.length / dModel;
  const experts = node.params?.num_experts ?? weight.shape[weight.shape.length - 1];
  const topK = node.params?.top_k ?? indices.shape[indices.shape.length - 1];
  const temperature = node.params?.temperature ?? 1;
  const normalize = node.params?.normalize !== false;
  for (let row = 0; row < rows; row++) {
    if (!normalize) {
      // Non-renormalized routing exposes the selected probabilities from a
      // softmax over every expert. The selected gates therefore also carry a
      // denominator gradient into experts that were not selected.
      const probabilities = new Float64Array(experts);
      let max = -Infinity;
      for (let expert = 0; expert < experts; expert++) {
        let logit = bias ? bias.buffer[expert] : 0;
        for (let dim = 0; dim < dModel; dim++) {
          logit += input.buffer[row * dModel + dim] * weight.buffer[dim * experts + expert];
        }
        logit /= temperature;
        probabilities[expert] = logit;
        if (logit > max) max = logit;
      }
      let denominator = 0;
      for (let expert = 0; expert < experts; expert++) {
        denominator += probabilities[expert] = Math.exp(probabilities[expert] - max);
      }
      for (let expert = 0; expert < experts; expert++) probabilities[expert] /= denominator;
      let probabilityDot = 0;
      for (let slot = 0; slot < topK; slot++) {
        const offset = row * topK + slot;
        const expert = Math.trunc(indices.buffer[offset]);
        probabilityDot += probabilities[expert] * gateGrad[offset];
      }
      for (let expert = 0; expert < experts; expert++) {
        let upstream = 0;
        for (let slot = 0; slot < topK; slot++) {
          const offset = row * topK + slot;
          if (Math.trunc(indices.buffer[offset]) === expert) upstream += gateGrad[offset];
        }
        const dLogit = probabilities[expert] * (upstream - probabilityDot) / temperature;
        if (gb) gb[expert] += dLogit;
        for (let dim = 0; dim < dModel; dim++) {
          const inputIndex = row * dModel + dim;
          const weightIndex = dim * experts + expert;
          gi[inputIndex] += dLogit * weight.buffer[weightIndex];
          gw[weightIndex] += dLogit * input.buffer[inputIndex];
        }
      }
      continue;
    }
    let gateDot = 0;
    for (let slot = 0; slot < topK; slot++) {
      const offset = row * topK + slot;
      gateDot += gateGrad[offset] * gates.buffer[offset];
    }
    for (let slot = 0; slot < topK; slot++) {
      const offset = row * topK + slot;
      const expert = Math.trunc(indices.buffer[offset]);
      if (expert < 0 || expert >= experts) continue;
      const dLogit = gates.buffer[offset] * (gateGrad[offset] - gateDot) / temperature;
      if (gb) gb[expert] += dLogit;
      for (let dim = 0; dim < dModel; dim++) {
        const inputIndex = row * dModel + dim;
        const weightIndex = dim * experts + expert;
        gi[inputIndex] += dLogit * weight.buffer[weightIndex];
        gw[weightIndex] += dLogit * input.buffer[inputIndex];
      }
    }
  }
}

/** Trainer-owned CPU forward engine. Inference CPUEngine never sees RNG state. */
class CPUTrainingForwardEngine extends CPUEngine {
  readonly #dropout: TrainingDropoutContext;

  constructor(dropout: TrainingDropoutContext) {
    super();
    this.#dropout = dropout;
  }

  override _runNode(node: any, execution: any = {}) {
    const nodeIndex = execution.nodeIndex ?? 0;
    if (node.opType === 'Dropout') {
      return _cpuDropout(node, {
        training: { dropout: this.#dropout },
        nodeIndex,
      });
    }
    if (node.opType === 'SDPA') {
      return this._cpuSDPA(node, {
        probabilityMultiplier: attentionDropout(node, this.#dropout, nodeIndex),
      });
    }
    if (node.opType === 'CrossSDPA') {
      return this._cpuCrossSDPA(node, {
        probabilityMultiplier: attentionDropout(node, this.#dropout, nodeIndex),
      });
    }
    return super._runNode(node, execution);
  }
}

export class CPUAutograd {
  /** @internal Trainer-owned forward entry used by gradient verification tests. */
  static async _forward(
    graph: Graph,
    inputs: Record<string, RuntimeTypedArray>,
    dropout: TrainingDropoutContext,
  ): Promise<void> {
    const engine = new CPUTrainingForwardEngine(dropout);
    engine.allocateGraph(graph);
    await engine.execute(graph, inputs, { adapter: null });
  }

  static async trainStep(graph: Graph, options: TrainingStepOptions = {}) {
    return this._trainStepWithExecution(ensureTrainingGraphState(graph), options, null);
  }

  /** Internal execution hook used by strict accelerated training backends. */
  static async _trainStepWithExecution(
    graph: any,
    options: TrainingStepOptions = {},
    execution: AcceleratedTrainingExecution | null = null,
  ) {
    const {
      inputs = {},
      targets,
      logitsTensor,
      losses = null,
      trainableTensors,
      optimizer = {},
      updateMode = "adamw",
      lastToken = null,
      ignoreIndex = null,
      lossMask = null,
      dropout = {},
      gradientAccumulationSteps = 1,
      flushGradientAccumulation = false,
      resetGradientAccumulation = false,
    } = options;
    if (!graph) throw new Error("CPUAutograd.trainStep requires a graph.");
    ensureTrainingGraphState(graph);
    if (!trainableTensors || trainableTensors.length === 0) {
      throw new Error("CPUAutograd.trainStep requires trainableTensors.");
    }
    if (new Set(trainableTensors).size !== trainableTensors.length) {
      throw new Error("CPUAutograd.trainStep trainableTensors must be unique.");
    }
    const mode = normalizeTrainingUpdateMode(updateMode);
    validateOptimizerOptions(optimizer, mode);
    const optimizerDescriptor = canonicalOptimizerDescriptor(mode, optimizer);
    const defaultLogitsName = logitsTensor || graph.outputNames?.[0] ||
      firstOutput(graph.nodes[graph.nodes.length - 1])?.name;
    const lossDescriptors = normalizeCrossEntropyLosses({
      targets,
      logitsTensor,
      losses,
      ignoreIndex,
      lossMask,
      lastToken,
    }, defaultLogitsName);
    if (!Number.isSafeInteger(gradientAccumulationSteps) || gradientAccumulationSteps <= 0) {
      throw new Error("CPUAutograd.trainStep gradientAccumulationSteps must be a positive safe integer.");
    }
    if (gradientAccumulationSteps > 1 && lossDescriptors.length > 1 &&
        lossDescriptors.some((loss) => loss.normalizer == null)) {
      throw new Error(
        "Multiple losses with gradient accumulation require an explicit full-window normalizer for every loss.",
      );
    }
    if (typeof flushGradientAccumulation !== "boolean" || typeof resetGradientAccumulation !== "boolean") {
      throw new Error("CPUAutograd.trainStep accumulation flush/reset options must be boolean.");
    }
    await execution?.preflight?.(graph, options);
    if (resetGradientAccumulation) resetPendingGradients(graph);
    const currentTrainingStep = graph.trainingStep ?? 0;
    const nextTrainingStep = optimizer.step ?? (currentTrainingStep + 1);
    if (!Number.isSafeInteger(currentTrainingStep) || currentTrainingStep < 0 ||
        !Number.isSafeInteger(nextTrainingStep) || nextTrainingStep <= currentTrainingStep) {
      throw new Error("CPUAutograd.trainStep optimizer step must be a safe integer greater than graph.trainingStep.");
    }
    const trainables = trainableTensors.map((name) => {
      const tensor = graph.getTensor(name);
      if (!tensor?.isWeight || tensor.dtype !== "float32" || !(tensor.buffer instanceof Float32Array)) {
        throw new Error(`Trainable tensor '${name}' must be an initialized F32 graph weight.`);
      }
      if (tensor.isWeight) graph.adapters?._assertBaseMutationAllowed();
      return tensor;
    });
    const topologyRevision = graph.topologyRevision;
    const weightRevision = graph.weightRevision;
    const accumulationIndex = gradientAccumulationIndex(graph);
    if (!dropout || typeof dropout !== "object" || Array.isArray(dropout)) {
      throw new Error("CPUAutograd.trainStep dropout options must be an object.");
    }
    const defaultDropoutCounter = Number(
      (BigInt(nextTrainingStep - 1) * BigInt(gradientAccumulationSteps) + BigInt(accumulationIndex + 1)) & 0xffffffffn,
    );
    const trainingDropout = dropoutContext({
      seed: dropout.seed ?? 0,
      counter: dropout.counter ?? defaultDropoutCounter,
    });

    if (execution?.forward) {
      await execution.forward(graph, inputs, { dropout: trainingDropout });
    } else {
      await this._forward(graph, inputs, trainingDropout);
    }
    const backendName = execution?.backend || "cpu";
    const backendResult = execution?.backend ? { backend: backendName } : {};
    graph.assertTopologyRevision?.(topologyRevision, `${backendName.toUpperCase()} training`);
    if (graph.weightRevision !== weightRevision) {
      throw new Error(`${backendName.toUpperCase()} training graph weights changed during the forward pass.`);
    }

    const grads: GradientMap = new Map();
    const lossMetrics: TrainingLossMetric[] = [];
    let loss = 0;
    let correct = 0;
    let count = 0;
    for (const descriptor of lossDescriptors) {
      const logits = graph.getTensor(descriptor.logitsTensor);
      if (!logits || logits.dtype !== "float32" || !(logits.buffer instanceof Float32Array)) {
        throw new Error(`Logits tensor '${descriptor.logitsTensor}' must be initialized F32.`);
      }
      const metric = execution?.crossEntropyGradient
        ? execution.crossEntropyGradient(logits, logits.buffer, descriptor)
        : crossEntropyGradient(logits, logits.buffer, descriptor);
      if (execution?.addGradient) execution.addGradient(gradFor(grads, logits)!, metric.gradient);
      else addGradient(gradFor(grads, logits)!, metric.gradient);
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
    const hasContribution = lossMetrics.some((metric) => metric.examples > 0 && metric.weight > 0);
    if (!hasContribution) {
      for (const tensor of trainables) gradFor(grads, tensor);
    }

    if (hasContribution) for (let ni = graph.nodes.length - 1; ni >= 0; ni--) {
      const node = graph.nodes[ni];
      const out = firstOutput(node);
      const liveOutputGradients = (Object.values(node.outputs || {}) as Tensor[])
        .filter((tensor) => tensor && grads.has(tensor.name));
      let go = out ? grads.get(out.name) : null;
      if (node.opType === "MoERouter") {
        const gates = node.outputs.weights || node.outputs.expert_weights;
        go = gates ? grads.get(gates.name) : null;
      }
      if (node.opType === "Split") {
        if (!liveOutputGradients.length) continue;
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length === 0) backwardFailure(node, "Split requires F32 input");
        if (execution?.backwardNode) {
          const handled = await execution.backwardNode({ node, nodeIndex: ni, output: out, outputGradient: go, outputGradients: new Map(liveOutputGradients.map((tensor) => [tensor.name, grads.get(tensor.name)])), gradients: grads, gradientFor: (tensor) => gradFor(grads, tensor), trainingDropout });
          if (!handled) backwardFailure(node, `the operation is not supported by the ${backendName} training backend`);
          continue;
        }
        let axis=node.params?.axis ?? 0; if(axis<0)axis+=input.shape.length;
        const outputs=(Object.values(node.outputs||{}) as Tensor[]);
        if(!Number.isInteger(axis)||axis<0||axis>=input.shape.length||input.shape[axis]%outputs.length) backwardFailure(node,"Split has invalid axis or output sizes");
        const split=input.shape[axis]/outputs.length, inner=input.shape.slice(axis+1).reduce((a,b)=>a*b,1), outer=input.shape.slice(0,axis).reduce((a,b)=>a*b,1), gi=gradFor(grads,input);
        for(let oi=0;oi<outputs.length;oi++){const dy=grads.get(outputs[oi].name);if(!dy)continue;for(let group=0;group<outer;group++)for(let index=0;index<split*inner;index++)gi[(group*input.shape[axis]+oi*split)*inner+index]+=dy[group*split*inner+index];}
        continue;
      }
      if (!go) {
        if (liveOutputGradients.length) {
          backwardFailure(node, "a non-primary output carries a gradient but has no backward rule");
        }
        continue;
      }
      if (execution?.backwardNode) {
        const handled = await execution.backwardNode({
          node,
          nodeIndex: ni,
          output: out,
          outputGradient: go,
          gradients: grads,
          gradientFor: (tensor) => gradFor(grads, tensor),
          trainingDropout,
        });
        if (!handled) backwardFailure(node, `the operation is not supported by the ${backendName} training backend`);
        continue;
      }
      if (node.opType === "MatMul" || node.opType === "Linear" || node.opType === "Gemm") {
        const input = nodeInput(node);
        const weight = node.inputs.weight;
        const bias = node.inputs.bias;
        if (!input || !weight || input.dtype !== "float32" || weight.dtype !== "float32" ||
            !input.buffer || !weight.buffer || !out?.buffer) {
          backwardFailure(node, "linear backward requires F32 input, weight, and output storage");
        }
        const gi = gradFor(grads, input);
        const gw = gradFor(grads, weight);
        const gb = bias?.dtype === "float32" ? gradFor(grads, bias) : null;
        const k = input.shape[input.shape.length - 1];
        const n = out.shape[out.shape.length - 1];
        const m = out.buffer.length / n;
        const doutFirst = node.inputs.scale ? true : (node.wLayout ? node.wLayout === "dout" : weight.shape[0] === n && weight.shape[1] === k);
        for (let row = 0; row < m; row++) {
          for (let j = 0; j < n; j++) {
            const dy = go[row * n + j];
            if (gb) gb[j] += dy;
            for (let kk = 0; kk < k; kk++) {
              const x = input.buffer[row * k + kk];
              if (doutFirst) {
                gi[row * k + kk] += dy * weight.buffer[j * k + kk];
                gw[j * k + kk] += x * dy;
              } else {
                gi[row * k + kk] += dy * weight.buffer[kk * n + j];
                gw[kk * n + j] += x * dy;
              }
            }
          }
        }
      } else if (node.opType === "Embedding") {
        backwardEmbedding(node, go, grads);
      } else if (node.opType === "LayerNorm") {
        backwardLayerNorm(node, go, grads);
      } else if (node.opType === "GroupNorm") {
        backwardGroupNorm(node, go, grads);
      } else if (node.opType === "RMSNorm") {
        backwardRMSNorm(node, go, grads);
      } else if (node.opType === "ReLU" || node.opType === "GELU" || node.opType === "SiLU" ||
                 node.opType === "Swish" || node.opType === "Sigmoid" || node.opType === "Tanh" ||
                 node.opType === "LeakyReLU" || node.opType === "HardSigmoid" || node.opType === "HardSwish" || node.opType === "Clip") {
        backwardActivation(node, go, grads);
      } else if (node.opType === "Softmax" || node.opType === "LogSoftmax") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || out?.dtype !== "float32" ||
            !(input.buffer instanceof Float32Array) || !(out.buffer instanceof Float32Array) ||
            input.buffer.length !== out.buffer.length) {
          backwardFailure(node, `${node.opType} requires matching F32 input/output storage`);
        }
        const width = input.shape.at(-1);
        const rows = input.buffer.length / width;
        const axis = node.params?.axis ?? -1;
        if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(rows) ||
            (axis !== -1 && axis !== input.shape.length - 1)) {
          backwardFailure(node, `${node.opType} supports the last axis only`);
        }
        const gi = gradFor(grads, input);
        for (let row = 0; row < rows; row++) {
          const offset = row * width;
          let sum = 0;
          if (node.opType === "LogSoftmax") {
            for (let column = 0; column < width; column++) sum += go[offset + column];
            for (let column = 0; column < width; column++) {
              gi[offset + column] += go[offset + column] - Math.exp(out.buffer[offset + column]) * sum;
            }
          } else {
            for (let column = 0; column < width; column++) sum += go[offset + column] * out.buffer[offset + column];
            for (let column = 0; column < width; column++) {
              gi[offset + column] += out.buffer[offset + column] * (go[offset + column] - sum);
            }
          }
        }
      } else if (node.opType === "PReLU") {
        const input = nodeInput(node); const slope = node.inputs.slope || node.inputs.weight;
        if (input?.dtype !== "float32" || slope?.dtype !== "float32" || !(input.buffer instanceof Float32Array) || !(slope.buffer instanceof Float32Array)) backwardFailure(node, "PReLU requires F32 input and slope");
        const gi = gradFor(grads, input), gs = gradFor(grads, slope), channels = input.shape.length === 4 ? input.shape[3] : slope.buffer.length;
        for (let i=0;i<go.length;i++) { const si=slope.buffer.length===channels?i%channels:i%slope.buffer.length; if(input.buffer[i]<0){gi[i]+=go[i]*slope.buffer[si];gs[si]+=go[i]*input.buffer[i];}else gi[i]+=go[i]; }
      } else if (node.opType === "GlobalAveragePool") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || go.length !== input.shape[0] * input.shape[3]) backwardFailure(node, "GlobalAveragePool requires rank-4 NHWC F32 tensors");
        const gi = gradFor(grads, input); const [batch, height, width, channels] = input.shape; const scale = 1 / (height * width);
        for (let n=0;n<batch;n++) for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<channels;c++) gi[((n*height+y)*width+x)*channels+c] += go[n*channels+c]*scale;
      } else if (node.opType === "BatchNorm2D") {
        // BatchNorm2D inference intentionally uses fixed running statistics.
        // Those statistics are constants for training, so this is an affine
        // channel-wise derivative rather than batch-statistics backpropagation.
        const input = nodeInput(node);
        const weight = node.inputs.weight || node.inputs.scale;
        const bias = node.inputs.bias || node.inputs.b;
        const mean = node.inputs.running_mean || node.inputs.mean;
        const variance = node.inputs.running_var || node.inputs.var;
        if (input?.dtype !== "float32" || weight?.dtype !== "float32" || bias?.dtype !== "float32" ||
            mean?.dtype !== "float32" || variance?.dtype !== "float32" || input.shape.length !== 4 ||
            !(input.buffer instanceof Float32Array) || !(weight.buffer instanceof Float32Array) ||
            !(bias.buffer instanceof Float32Array) || !(mean.buffer instanceof Float32Array) ||
            !(variance.buffer instanceof Float32Array)) backwardFailure(node, "BatchNorm2D requires NHWC F32 input and channel parameters");
        const channels = input.shape[3];
        if (weight.buffer.length !== channels || bias.buffer.length !== channels || mean.buffer.length !== channels || variance.buffer.length !== channels) backwardFailure(node, "BatchNorm2D channel parameter length does not match input");
        const epsilon = node.params?.eps ?? 1e-5;
        if (!(epsilon > 0)) backwardFailure(node, "BatchNorm2D epsilon must be positive");
        const gi = gradFor(grads, input), gw = gradFor(grads, weight), gb = gradFor(grads, bias);
        for (let i = 0; i < go.length; i++) {
          const c = i % channels;
          const inv = 1 / Math.sqrt(variance.buffer[c] + epsilon);
          gi[i] += go[i] * weight.buffer[c] * inv;
          gw[c] += go[i] * (input.buffer[i] - mean.buffer[c]) * inv;
          gb[c] += go[i];
        }
      } else if (node.opType === "MaxPool2D" || node.opType === "AveragePool" || node.opType === "AveragePool2D") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || out?.dtype !== "float32") backwardFailure(node, `${node.opType} requires rank-4 NHWC F32 tensors`);
        const [batch, height, width, channels] = input.shape;
        const [, outHeight, outWidth] = out.shape;
        const [kernelY, kernelX] = node.params?.kernel || [];
        const [strideY, strideX] = node.params?.stride || [1, 1];
        const [padY, padX] = node.params?.padding || [0, 0];
        if (![kernelY, kernelX, strideY, strideX, padY, padX].every(Number.isInteger) || kernelY <= 0 || kernelX <= 0 || strideY <= 0 || strideX <= 0 || padY < 0 || padX < 0) backwardFailure(node, `${node.opType} has unsupported pooling parameters`);
        const gi = gradFor(grads, input);
        for (let n=0;n<batch;n++) for (let oy=0;oy<outHeight;oy++) for (let ox=0;ox<outWidth;ox++) {
          if (node.opType === "MaxPool2D") {
            for (let c=0;c<channels;c++) {
              let best = -Infinity, bestIndex = -1;
              for (let ky=0;ky<kernelY;ky++) for (let kx=0;kx<kernelX;kx++) { const iy=oy*strideY+ky-padY, ix=ox*strideX+kx-padX; if(iy>=0&&iy<height&&ix>=0&&ix<width){const index=((n*height+iy)*width+ix)*channels+c; if(input.buffer[index]>best){best=input.buffer[index];bestIndex=index;}} }
              gi[bestIndex] += go[((n*outHeight+oy)*outWidth+ox)*channels+c];
            }
          } else {
            let count = 0;
            for (let ky=0;ky<kernelY;ky++) for (let kx=0;kx<kernelX;kx++) { const iy=oy*strideY+ky-padY, ix=ox*strideX+kx-padX; if(iy>=0&&iy<height&&ix>=0&&ix<width) count++; }
            for (let ky=0;ky<kernelY;ky++) for (let kx=0;kx<kernelX;kx++) { const iy=oy*strideY+ky-padY, ix=ox*strideX+kx-padX; if(iy>=0&&iy<height&&ix>=0&&ix<width) for(let c=0;c<channels;c++) gi[((n*height+iy)*width+ix)*channels+c] += go[((n*outHeight+oy)*outWidth+ox)*channels+c] / count; }
          }
        }
      } else if (node.opType === "Resize" || node.opType === "ResizeNearest2D" || node.opType === "UpsampleNearest2D" || node.opType === "Upsample2x") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || out?.dtype !== "float32") backwardFailure(node, `${node.opType} requires rank-4 NHWC F32 tensors`);
        const [batch, inHeight, inWidth, channels] = input.shape;
        const [, outHeight, outWidth] = out.shape;
        const nearest = node.opType === "ResizeNearest2D" || node.opType === "UpsampleNearest2D" || node.opType === "Upsample2x" || node.params?.mode === "nearest";
        const gi = gradFor(grads, input);
        for (let n=0;n<batch;n++) for (let oy=0;oy<outHeight;oy++) for (let ox=0;ox<outWidth;ox++) {
          if (nearest) {
            const iy=Math.min(inHeight-1, Math.floor(oy*inHeight/outHeight)), ix=Math.min(inWidth-1, Math.floor(ox*inWidth/outWidth));
            for(let c=0;c<channels;c++) gi[((n*inHeight+iy)*inWidth+ix)*channels+c] += go[((n*outHeight+oy)*outWidth+ox)*channels+c];
          } else {
            let fy=(oy+.5)*inHeight/outHeight-.5, fx=(ox+.5)*inWidth/outWidth-.5; if(fy<0)fy=0;if(fx<0)fx=0;
            const y0=Math.min(inHeight-1,Math.floor(fy)), x0=Math.min(inWidth-1,Math.floor(fx)), y1=Math.min(inHeight-1,y0+1), x1=Math.min(inWidth-1,x0+1), wy=fy-y0, wx=fx-x0;
            for(let c=0;c<channels;c++){const g=go[((n*outHeight+oy)*outWidth+ox)*channels+c];gi[((n*inHeight+y0)*inWidth+x0)*channels+c]+=g*(1-wy)*(1-wx);gi[((n*inHeight+y0)*inWidth+x1)*channels+c]+=g*(1-wy)*wx;gi[((n*inHeight+y1)*inWidth+x0)*channels+c]+=g*wy*(1-wx);gi[((n*inHeight+y1)*inWidth+x1)*channels+c]+=g*wy*wx;}
          }
        }
      } else if (node.opType === "SDPA") {
        backwardSDPA(node, go, grads, trainingDropout, ni);
      } else if (node.opType === "CrossSDPA") {
        backwardCrossSDPA(node, go, grads, trainingDropout, ni);
      } else if (node.opType === "CrossAttention") {
        backwardCrossAttention(node, go, grads);
      } else if (node.opType === "Conv2D") {
        backwardConv2D(node, go, grads);
      } else if (node.opType === "Conv1D") {
        backwardConv1D(node, go, grads);
      } else if (node.opType === "ConvTranspose2D") {
        backwardConvTranspose2D(node, go, grads);
      } else if (node.opType === "MoELinear") {
        backwardMoELinear(node, go, grads);
      } else if (node.opType === "MoERouter") {
        backwardMoERouter(node, go, grads);
      } else if (node.opType === "Add") {
        const a = node.inputs.a || nodeInput(node);
        const b = node.inputs.b;
        if (!a || !b || !out) backwardFailure(node, "Add requires two inputs and one output");
        let differentiable = false;
        if (a.dtype === "float32") {
          addBroadcast(node, a, out, gradFor(grads, a), go);
          differentiable = true;
        }
        if (b.dtype === "float32") {
          addBroadcast(node, b, out, gradFor(grads, b), go);
          differentiable = true;
        }
        if (!differentiable) backwardFailure(node, "Add has no differentiable F32 input");
      } else if (node.opType === "Mul" || node.opType === "Sub" || node.opType === "Div") {
        const a = node.inputs.a || nodeInput(node);
        const b = node.inputs.b;
        if (a?.dtype !== "float32" || b?.dtype !== "float32" || !a.buffer || !b.buffer || !out) {
          backwardFailure(node, `${node.opType} backward requires two F32 inputs and one output`);
        }
        const ga = gradFor(grads, a);
        const gb = gradFor(grads, b);
        const aIndex = broadcastIndexer(node, a, out);
        const bIndex = broadcastIndexer(node, b, out);
        for (let i = 0; i < go.length; i++) {
          const ai = aIndex(i);
          const bi = bIndex(i);
          if (node.opType === "Mul") {
            ga[ai] += go[i] * b.buffer[bi];
            gb[bi] += go[i] * a.buffer[ai];
          } else if (node.opType === "Sub") {
            ga[ai] += go[i];
            gb[bi] -= go[i];
          } else {
            ga[ai] += go[i] / b.buffer[bi];
            gb[bi] -= go[i] * a.buffer[ai] / (b.buffer[bi] * b.buffer[bi]);
          }
        }
      } else if (node.opType === "Where" || node.opType === "Mask") {
        const condition = node.inputs.cond || node.inputs.condition || node.inputs.mask;
        const a = node.inputs.x || node.inputs.a || nodeInput(node);
        const b = node.inputs.y || node.inputs.b;
        if (!condition?.buffer || !a || !b || a.dtype !== "float32" || b.dtype !== "float32" ||
            !out || condition.buffer.length !== go.length || a.buffer?.length !== go.length ||
            b.buffer?.length !== go.length) {
          backwardFailure(node, `${node.opType} requires exact-shape condition and F32 operands`);
        }
        const da = gradFor(grads, a);
        const db = gradFor(grads, b);
        for (let i = 0; i < go.length; i++) {
          if (condition.buffer[i] !== 0) da[i] += go[i];
          else db[i] += go[i];
        }
      } else if (node.opType === "Slice") {
        const input = nodeInput(node);
        const rank = input?.shape?.length ?? 0;
        const startsInput = node.params?.starts ?? [];
        const stepsInput = node.params?.steps ?? startsInput.map(() => 1);
        const axesInput = node.params?.axes ?? startsInput.map((_, index) => index);
        if (input?.dtype !== "float32" || !out || rank === 0 || rank > 8 ||
            out.shape.length !== rank || !Array.isArray(startsInput) ||
            !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
            startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
          backwardFailure(node, "Slice requires matching rank-1..8 F32 tensors and parameter arrays");
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
              !Number.isInteger(start) || !Number.isInteger(step) || step <= 0) {
            backwardFailure(node, "Slice requires unique axes and positive integer steps");
          }
          if (start < 0) start += input.shape[axis];
          if (start < 0 || start >= input.shape[axis]) {
            backwardFailure(node, "Slice start is outside its input axis");
          }
          starts[axis] = start;
          steps[axis] = step;
          seen.add(axis);
        }
        const inputStrides = new Array(rank);
        let stride = 1;
        for (let axis = rank - 1; axis >= 0; axis--) {
          const outputLength = out.shape[axis];
          if (!Number.isInteger(outputLength) || outputLength <= 0 ||
              starts[axis] + (outputLength - 1) * steps[axis] >= input.shape[axis]) {
            backwardFailure(node, "Slice output shape exceeds its input selection");
          }
          inputStrides[axis] = stride;
          stride *= input.shape[axis];
        }
        const gi = gradFor(grads, input);
        for (let outputIndex = 0; outputIndex < go.length; outputIndex++) {
          let remaining = outputIndex;
          let inputIndex = 0;
          for (let axis = rank - 1; axis >= 0; axis--) {
            const coordinate = remaining % out.shape[axis];
            remaining = Math.floor(remaining / out.shape[axis]);
            inputIndex += (starts[axis] + coordinate * steps[axis]) * inputStrides[axis];
          }
          gi[inputIndex] += go[outputIndex];
        }
      } else if (node.opType === "Gather") {
        const input = nodeInput(node);
        const indices = node.inputs.indices;
        const rank = input?.shape?.length ?? 0;
        let axis = node.params?.axis ?? 0;
        if (axis < 0) axis += rank;
        if (input?.dtype !== "float32" || !indices?.buffer || !out || rank === 0 || rank > 8 ||
            !Number.isInteger(axis) || axis < 0 || axis >= rank) {
          backwardFailure(node, "Gather requires F32 data, initialized indices, and a valid axis");
        }
        const expectedShape = [...input.shape.slice(0, axis), ...indices.shape, ...input.shape.slice(axis + 1)];
        if (out.shape.length !== expectedShape.length ||
            out.shape.some((dimension, index) => dimension !== expectedShape[index])) {
          backwardFailure(node, "Gather output shape does not replace the selected axis with indices");
        }
        const outer = elementCount(input.shape.slice(0, axis));
        const inner = elementCount(input.shape.slice(axis + 1));
        const indexCount = elementCount(indices.shape);
        if (go.length !== outer * indexCount * inner) {
          backwardFailure(node, "Gather output storage does not match its shapes");
        }
        const gi = gradFor(grads, input);
        for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
          for (let indexPosition = 0; indexPosition < indexCount; indexPosition++) {
            let selected = indices.buffer[indexPosition];
            if (!Number.isInteger(selected)) {
              backwardFailure(node, "Gather index is outside the selected axis");
            }
            if (selected < 0) selected += input.shape[axis];
            if (selected < 0 || selected >= input.shape[axis]) {
              backwardFailure(node, "Gather index is outside the selected axis");
            }
            for (let innerIndex = 0; innerIndex < inner; innerIndex++) {
              const outputIndex = (outerIndex * indexCount + indexPosition) * inner + innerIndex;
              const inputIndex = (outerIndex * input.shape[axis] + selected) * inner + innerIndex;
              gi[inputIndex] += go[outputIndex];
            }
          }
        }
      } else if (node.opType === "GatherElements") {
        const input = nodeInput(node);
        const indices = node.inputs.indices;
        const rank = input?.shape?.length ?? 0;
        let axis = node.params?.axis ?? 0;
        if (axis < 0) axis += rank;
        if (input?.dtype !== "float32" || !indices?.buffer || !out || rank === 0 || rank > 8 ||
            indices.shape.length !== rank || out.shape.length !== rank ||
            !Number.isInteger(axis) || axis < 0 || axis >= rank ||
            out.shape.some((dimension, index) => dimension !== indices.shape[index]) ||
            indices.shape.some((dimension, index) => index !== axis && dimension > input.shape[index])) {
          backwardFailure(node, "GatherElements has incompatible data, indices, output, or axis");
        }
        const inputStrides = new Array(rank);
        const indexStrides = new Array(rank);
        let inputStride = 1;
        let indexStride = 1;
        for (let dimension = rank - 1; dimension >= 0; dimension--) {
          inputStrides[dimension] = inputStride;
          indexStrides[dimension] = indexStride;
          inputStride *= input.shape[dimension];
          indexStride *= indices.shape[dimension];
        }
        const gi = gradFor(grads, input);
        for (let outputIndex = 0; outputIndex < go.length; outputIndex++) {
          let remaining = outputIndex;
          let inputIndex = 0;
          for (let dimension = 0; dimension < rank; dimension++) {
            const coordinate = Math.floor(remaining / indexStrides[dimension]);
            remaining %= indexStrides[dimension];
            if (dimension === axis) {
              let selected = indices.buffer[outputIndex];
              if (!Number.isInteger(selected)) backwardFailure(node, "GatherElements index must be integral");
              if (selected < 0) selected += input.shape[dimension];
              if (selected < 0 || selected >= input.shape[dimension]) {
                backwardFailure(node, "GatherElements index is outside the selected axis");
              }
              inputIndex += selected * inputStrides[dimension];
            } else {
              inputIndex += coordinate * inputStrides[dimension];
            }
          }
          gi[inputIndex] += go[outputIndex];
        }
      } else if (node.opType === "MeanHeight") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || !out ||
            out.shape.length !== 3 || out.shape[0] !== input.shape[0] ||
            out.shape[1] !== input.shape[3] || out.shape[2] !== input.shape[2]) {
          backwardFailure(node, "MeanHeight requires NHWC F32 input and [batch, channels, width] output");
        }
        const [batch, height, width, channels] = input.shape;
        const gi = gradFor(grads, input);
        for (let b = 0; b < batch; b++) for (let channel = 0; channel < channels; channel++) {
          for (let x = 0; x < width; x++) {
            const gradient = go[(b * channels + channel) * width + x] / height;
            for (let y = 0; y < height; y++) {
              gi[((b * height + y) * width + x) * channels + channel] += gradient;
            }
          }
        }
      } else if (node.opType === "ProfileX" || node.opType === "ProfileY") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || !out || out.shape.length !== 3) {
          backwardFailure(node, `${node.opType} requires NHWC F32 input and rank-3 output`);
        }
        const [batch, height, width, channels] = input.shape;
        const expected = node.opType === "ProfileX" ? [batch, 2 * channels, width] : [batch, 2 * channels, height];
        if (out.shape.some((dimension, index) => dimension !== expected[index])) {
          backwardFailure(node, `${node.opType} output shape is incompatible with its input`);
        }
        const gi = gradFor(grads, input);
        if (node.opType === "ProfileX") {
          for (let b = 0; b < batch; b++) for (let channel = 0; channel < channels; channel++) {
            for (let x = 0; x < width; x++) {
              let maxY = 0;
              let maximum = input.buffer[((b * height) * width + x) * channels + channel];
              for (let y = 1; y < height; y++) {
                const value = input.buffer[((b * height + y) * width + x) * channels + channel];
                if (value > maximum) { maximum = value; maxY = y; }
              }
              const outputBase = b * 2 * channels * width;
              const maxGradient = go[outputBase + channel * width + x];
              const meanGradient = go[outputBase + (channels + channel) * width + x] / height;
              for (let y = 0; y < height; y++) {
                gi[((b * height + y) * width + x) * channels + channel] += meanGradient;
              }
              gi[((b * height + maxY) * width + x) * channels + channel] += maxGradient;
            }
          }
        } else {
          for (let b = 0; b < batch; b++) for (let channel = 0; channel < channels; channel++) {
            for (let y = 0; y < height; y++) {
              let maxX = 0;
              let maximum = input.buffer[((b * height + y) * width) * channels + channel];
              for (let x = 1; x < width; x++) {
                const value = input.buffer[((b * height + y) * width + x) * channels + channel];
                if (value > maximum) { maximum = value; maxX = x; }
              }
              const outputBase = b * 2 * channels * height;
              const maxGradient = go[outputBase + channel * height + y];
              const meanGradient = go[outputBase + (channels + channel) * height + y] / width;
              for (let x = 0; x < width; x++) {
                gi[((b * height + y) * width + x) * channels + channel] += meanGradient;
              }
              gi[((b * height + y) * width + maxX) * channels + channel] += maxGradient;
            }
          }
        }
      } else if (node.opType === "SpatialSoftargmaxY") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || input.shape.length !== 4 || !out ||
            out.shape.length !== 3 || out.shape[0] !== input.shape[0] ||
            out.shape[1] !== input.shape[3] || out.shape[2] !== input.shape[2]) {
          backwardFailure(node, "SpatialSoftargmaxY requires NHWC F32 input and [batch, channels, width] output");
        }
        const [batch, height, width, channels] = input.shape;
        const gi = gradFor(grads, input);
        for (let b = 0; b < batch; b++) for (let channel = 0; channel < channels; channel++) {
          for (let x = 0; x < width; x++) {
            let maximum = input.buffer[((b * height) * width + x) * channels + channel];
            for (let y = 1; y < height; y++) {
              maximum = Math.max(maximum, input.buffer[((b * height + y) * width + x) * channels + channel]);
            }
            let denominator = 0;
            for (let y = 0; y < height; y++) {
              denominator += Math.exp(input.buffer[((b * height + y) * width + x) * channels + channel] - maximum);
            }
            if (!(denominator > 0)) continue;
            const outputIndex = (b * channels + channel) * width + x;
            for (let y = 0; y < height; y++) {
              const inputIndex = ((b * height + y) * width + x) * channels + channel;
              const probability = Math.exp(input.buffer[inputIndex] - maximum) / denominator;
              gi[inputIndex] += go[outputIndex] * probability * ((y + 0.5) / height - out.buffer[outputIndex]);
            }
          }
        }
      } else if (node.opType === "Cast") {
        const input = nodeInput(node);
        if (!input || !out || input.buffer?.length !== out.buffer?.length) {
          backwardFailure(node, 'Cast requires equal-size initialized input and output storage');
        }
        // A floating-point cast is an identity in the differentiable graph.
        // Integer conversions intentionally stop gradients rather than silently
        // applying a straight-through estimator.
        if (input.dtype === 'float32' && out.dtype === 'float32') {
          const gi = gradFor(grads, input);
          for (let i = 0; i < go.length; i++) gi[i] += go[i];
        }
      } else if (node.opType === "DequantizeLinear") {
        const input = node.inputs.input || node.inputs.x;
        const scale = node.inputs.scale;
        const zeroPoint = node.inputs.zero_point || null;
        if (!input?.buffer || !scale?.buffer || !out || out.dtype !== 'float32' ||
            scale.dtype !== 'float32' || scale.buffer.length !== 1 ||
            (zeroPoint && (!zeroPoint.buffer || zeroPoint.buffer.length !== 1)) ||
            input.buffer.length !== go.length || out.buffer.length !== go.length) {
          backwardFailure(node, 'DequantizeLinear requires scalar F32 scale and matching initialized storage');
        }
        const zero = zeroPoint ? Number(zeroPoint.buffer[0]) : 0;
        const scaleValue = scale.buffer[0];
        if (input.dtype === 'float32') {
          const gi = gradFor(grads, input);
          for (let i = 0; i < go.length; i++) gi[i] += go[i] * scaleValue;
        }
        if (scale.dtype === 'float32') {
          const gs = gradFor(grads, scale);
          for (let i = 0; i < go.length; i++) gs[0] += go[i] * (Number(input.buffer[i]) - zero);
        }
      } else if (node.opType === "Expand" || node.opType === "Broadcast") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || !input.buffer || !out) {
          backwardFailure(node, `${node.opType} backward requires an F32 input and output`);
        }
        addBroadcast(node, input, out, gradFor(grads, input), go);
      } else if (node.opType === "Pad") {
        const input = nodeInput(node);
        if (input?.dtype !== 'float32' || !input.buffer || input.shape.length !== 4 || out?.shape?.length !== 4) {
          backwardFailure(node, 'Pad backward requires rank-4 NHWC F32 input/output storage');
        }
        const pads = node.params?.pads || [];
        const [padTop, padLeft, padBottom, padRight] = pads.length === 8
          ? [pads[1], pads[2], pads[5], pads[6]] : pads;
        const [batch, height, width, channels] = input.shape;
        if (![padTop, padBottom, padLeft, padRight].every((value) => Number.isInteger(value) && value >= 0) ||
            out.shape[0] !== batch || out.shape[1] !== height + padTop + padBottom ||
            out.shape[2] !== width + padLeft + padRight || out.shape[3] !== channels) {
          backwardFailure(node, 'Pad has incompatible constant-padding dimensions');
        }
        const gi = gradFor(grads, input), outHeight = out.shape[1], outWidth = out.shape[2];
        for (let b = 0; b < batch; b++) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < channels; c++) {
          gi[((b * height + y) * width + x) * channels + c] += go[((b * outHeight + y + padTop) * outWidth + x + padLeft) * channels + c];
        }
      } else if(node.opType==='Interpolate1D'||node.opType==='Interp1D'||node.opType==='InterpLinear1D'){
        const input=nodeInput(node);if(input?.dtype!=='float32'||input.shape.length!==3||out?.shape?.length!==3)backwardFailure(node,'Interpolate1D backward requires rank-3 NCL F32 tensors');
        const [batch,channels,inputLength]=input.shape,outputLength=out.shape[2],gi=gradFor(grads,input),scale=inputLength/outputLength;
        for(let b=0;b<batch;b++)for(let c=0;c<channels;c++)for(let x=0;x<outputLength;x++){let position=(x+.5)*scale-.5;position=Math.max(0,Math.min(inputLength-1,position));const x0=Math.floor(position),x1=Math.min(inputLength-1,x0+1),fraction=position-x0,g=go[(b*channels+c)*outputLength+x],base=(b*channels+c)*inputLength;gi[base+x0]+=g*(1-fraction);gi[base+x1]+=g*fraction;}
      } else if (node.opType === "Concat" || node.opType === "Concat2") {
        backwardConcat(node, go, grads);
      } else if (node.opType === "Dropout") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || !input.buffer || input.buffer.length !== go.length) {
          backwardFailure(node, "Dropout requires equal-size F32 input and output-gradient storage");
        }
        const gi = gradFor(grads, input);
        for (let i = 0; i < go.length; i++) {
          gi[i] += go[i] * dropoutMultiplier(node, trainingDropout, i, ni);
        }
      } else if (node.opType === "ReduceSum" || node.opType === "ReduceMean") {
        const input = nodeInput(node);
        const rowWidth = input?.shape?.at(-1);
        const rows = input?.buffer?.length / rowWidth;
        if (input?.dtype !== "float32" || !(input.buffer instanceof Float32Array) ||
            !Number.isSafeInteger(rowWidth) || rowWidth <= 0 || !Number.isSafeInteger(rows) ||
            go.length !== rows) {
          backwardFailure(node, `${node.opType} requires an F32 input and one output gradient per last-axis row`);
        }
        const gi = gradFor(grads, input);
        const scale = node.opType === "ReduceMean" ? 1 / rowWidth : 1;
        for (let index = 0; index < gi.length; index++) gi[index] += go[Math.floor(index / rowWidth)] * scale;
      } else if (node.opType === "Identity" || node.opType === "Reshape" || node.opType === "Flatten" ||
                 node.opType === "Squeeze" || node.opType === "Unsqueeze") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || !input.buffer || input.buffer.length !== go.length) {
          backwardFailure(node, "shape/copy backward requires equal-size F32 storage");
        }
        const gi = gradFor(grads, input);
        for (let i = 0; i < go.length; i++) gi[i] += go[i];
      } else if (node.opType === "Transpose") {
        const input = nodeInput(node);
        if (input?.dtype !== "float32" || !input.buffer) {
          backwardFailure(node, "Transpose backward requires F32 input storage");
        }
        const gi = gradFor(grads, input);
        const rank = input.shape.length;
        const perm = node.params?.perm || [...Array(rank).keys()].reverse();
        const inStrides = new Array(rank);
        const outShape = perm.map((p) => input.shape[p]);
        const outStrides = new Array(rank);
        let s = 1;
        for (let i = rank - 1; i >= 0; i--) { inStrides[i] = s; s *= input.shape[i]; }
        s = 1;
        for (let i = rank - 1; i >= 0; i--) { outStrides[i] = s; s *= outShape[i]; }
        for (let oi = 0; oi < go.length; oi++) {
          let tmp = oi;
          let ii = 0;
          for (let d = 0; d < rank; d++) {
            const coord = Math.floor(tmp / outStrides[d]);
            tmp %= outStrides[d];
            ii += coord * inStrides[perm[d]];
          }
          gi[ii] += go[oi];
        }
      } else {
        backwardFailure(node, "the operation is not in the CPU training allowlist");
      }
    }

    graph.assertTopologyRevision?.(topologyRevision, `${backendName.toUpperCase()} training`);
    if (graph.weightRevision !== weightRevision) {
      throw new Error(`${backendName.toUpperCase()} training graph weights changed before optimizer application.`);
    }
    const currentGradients: GradientMap = new Map();
    for (const tensor of trainables) {
      const grad = grads.get(tensor.name);
      if (!grad || grad.length !== tensor.buffer.length) {
        throw new Error(`No valid gradient reached trainable tensor '${tensor.name}'.`);
      }
      if (execution?.allFinite) {
        if (!execution.allFinite(grad)) {
          throw new Error(`Gradient for trainable tensor '${tensor.name}' is non-finite.`);
        }
      } else {
        for (let index = 0; index < grad.length; index++) {
          if (!Number.isFinite(grad[index])) {
            throw new Error(`Gradient for trainable tensor '${tensor.name}' is non-finite at index ${index}.`);
          }
        }
      }
      currentGradients.set(tensor.name, grad);
    }
    const accumulationSignature = JSON.stringify({
      backend: backendName,
      topologyRevision,
      weightRevision,
      trainableTensors,
      mode,
      optimizerDescriptor,
      nextTrainingStep,
      losses: lossDescriptors.map(({ name, logitsTensor, weight, normalizer }) => ({
        name, logitsTensor, weight, normalizer,
      })),
    });
    const accumulationAggregation = lossDescriptors.length > 1 ||
      lossDescriptors.every((descriptor) => descriptor.normalizer != null)
      ? "sum"
      : "mean_by_examples";
    const accumulation = accumulateGradients(graph, {
      backend: backendName,
      gradients: currentGradients,
      examples: count,
      loss,
      correct,
      trainableNames: trainableTensors,
      accumulationSteps: gradientAccumulationSteps,
      flush: flushGradientAccumulation,
      signature: accumulationSignature,
      aggregation: accumulationAggregation,
      hasContribution,
      metrics: lossMetrics,
    });
    if (!accumulation.apply) {
      return {
        ...backendResult,
        loss: accumulation.loss,
        correct: accumulation.correct,
        examples: accumulation.examples,
        losses: accumulation.metrics ?? lossMetrics,
        updatedTensors: [],
        gradients: accumulation.gradients ?? currentGradients,
        accumulating: !accumulation.complete,
        accumulationStep: accumulation.microbatches,
        gradientAccumulationSteps,
      };
    }

    const clipping = execution?.clipGradients
      ? execution.clipGradients(accumulation.gradients, optimizerDescriptor.optimizer.maxGradNorm)
      : clipGradientsGlobal(accumulation.gradients, optimizerDescriptor.optimizer.maxGradNorm);
    const prepared: Array<[Tensor, Float32Array]> = trainables.map((tensor) => [
      tensor,
      accumulation.gradients.get(tensor.name)!,
    ]);
    const updatedTensors: Tensor[] = [];
    for (const [trainable, grad] of prepared) {
      const updateOptions = {
        ...optimizer,
        maxGradNorm: 0,
        mode,
        step: nextTrainingStep,
      };
      const tensor = execution?.applyTensorUpdate
        ? execution.applyTensorUpdate(graph, trainable, grad, updateOptions)
        : graph.applyTensorUpdate(trainable.name, grad, updateOptions);
      updatedTensors.push(tensor);
    }
    graph.trainingStep = nextTrainingStep;
    if (mode !== "adamw") graph.optimizerState = null;
    graph.optimizerDescriptor = optimizerDescriptor;
    synchronizeTrainingGraphState(graph);

    return {
      ...backendResult,
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
}
