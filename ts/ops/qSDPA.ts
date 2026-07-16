import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type { PerTensorQuantization, TensorQuantization } from '../types.js';

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function tensorElements(tensor, label, node) {
  if (!tensor || !Array.isArray(tensor.shape) ||
      tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} has an invalid element count.`);
  }
  return elements;
}

function matchesByteStorage(tensor) {
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  if (tensor.dtype === 'uint8') {
    return tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray;
  }
  return false;
}

function requireByteTensor(tensor, label, node) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesByteStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes || tensor.buffer.length !== elements) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requireInt32Tensor(tensor, label, node) {
  if (!tensor || tensor.dtype !== 'int32' || !(tensor.buffer instanceof Int32Array)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} requires I32 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes('int32');
  if (tensor.sizeBytes !== expectedBytes || tensor.buffer.byteLength !== expectedBytes ||
      tensor.buffer.length !== elements) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} has incompatible I32 storage.`);
  }
  return elements;
}

function requirePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QSDPA node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} requires per_tensor quantization.`);
  }
  return descriptor;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function storageRangesOverlap(left, right) {
  const leftStorage = left.buffer;
  const rightStorage = right.buffer;
  if (leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

function canonicalInputs(node) {
  const inputs = node?.inputs;
  if (!isRecord(inputs)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires q, k, and v inputs.`);
  }
  const names = Object.keys(inputs).sort();
  const withoutMask = names.length === 3 && names[0] === 'k' && names[1] === 'q' && names[2] === 'v';
  const withMask = names.length === 4 && names[0] === 'k' && names[1] === 'mask' &&
    names[2] === 'q' && names[3] === 'v';
  if (!withoutMask && !withMask) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires exactly q, k, v, and optional mask inputs.`);
  }
  return { q: inputs.q, k: inputs.k, v: inputs.v, mask: withMask ? inputs.mask : null };
}

function canonicalOutput(node) {
  const outputs = node?.outputs;
  if (!isRecord(outputs) || Object.keys(outputs).length !== 1) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires exactly one output.`);
  }
  return outputs.out ?? Object.values(outputs)[0];
}

function parseParams(node, dModel) {
  const params = node?.params;
  if (!isRecord(params)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} parameters must be an object.`);
  }
  for (const name of Object.keys(params)) {
    if (!['heads', 'causal', 'scale'].includes(name)) {
      throw new Error(`QSDPA node ${nodeLabel(node)} does not accept parameter '${name}'.`);
    }
  }
  if (!Object.prototype.hasOwnProperty.call(params, 'heads') ||
      !Number.isInteger(params.heads) || params.heads <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} heads must be a positive integer.`);
  }
  if (!Object.prototype.hasOwnProperty.call(params, 'causal') || typeof params.causal !== 'boolean') {
    throw new Error(`QSDPA node ${nodeLabel(node)} causal must be an explicit boolean.`);
  }
  const heads = params.heads;
  const headDim = dModel / heads;
  if (!Number.isInteger(headDim) || headDim <= 0 || dModel % 4 !== 0 || headDim % 4 !== 0 ||
      headDim > 64) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires D/head dimensions divisible by 4 with head_dim <= 64.`);
  }
  const scaleSource = params.scale ?? 1 / Math.sqrt(headDim);
  if (typeof scaleSource !== 'number' || !Number.isFinite(scaleSource) || scaleSource <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} scale must be a positive finite F32 value.`);
  }
  const scale = Math.fround(scaleSource);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} scale must be a positive finite F32 value.`);
  }
  return { heads, headDim, causal: params.causal, scale };
}

function maskMode(mask, batch, queries, keys, node) {
  if (!mask) return 0;
  requireInt32Tensor(mask, 'mask', node);
  const shape = mask.shape;
  if (shape.length === 1 && shape[0] === keys) return 1;
  if (shape.length === 2 && shape[1] === keys) {
    if (shape[0] === batch) return 2;
    if (shape[0] === queries) return 3;
  }
  if (shape.length === 3 && shape[0] === batch && shape[1] === queries && shape[2] === keys) return 4;
  throw new Error(`QSDPA node ${nodeLabel(node)} mask must have shape [K], [B,K], [Q,K], or [B,Q,K].`);
}

function maskKeeps(mask, mode, batchIndex, query, key, queries, keys) {
  if (mode === 0) return true;
  let index = key;
  if (mode === 2) index = batchIndex * keys + key;
  else if (mode === 3) index = query * keys + key;
  else if (mode === 4) index = (batchIndex * queries + query) * keys + key;
  return mask.buffer[index] !== 0;
}

function requantize(value, outputScale, outputZeroPoint, minimum, maximum) {
  const transformed = Math.fround(Math.fround(value / outputScale) + outputZeroPoint);
  if (Number.isNaN(transformed)) return outputZeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

function centeredRaw(value, zeroPoint) {
  return value - zeroPoint;
}

function maximumCenteredMagnitude(dtype, zeroPoint) {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}

// Canonical byte-domain scaled dot-product attention. Q/K use an exact I32
// raw-byte dot product for each head; the online F32 softmax keeps only scalar
// state plus a private <=64-value accumulator, never an F32 graph tensor or
// score matrix. Validation finishes before the first output byte is assigned.
export function _cpuQSDPA(node) {
  const { q, k, v, mask } = canonicalInputs(node);
  const output = canonicalOutput(node);
  const qElements = requireByteTensor(q, 'q', node);
  const kElements = requireByteTensor(k, 'k', node);
  const vElements = requireByteTensor(v, 'v', node);
  const outputElements = requireByteTensor(output, 'output', node);
  const rank = q.shape.length;
  if ((rank !== 2 && rank !== 3) || k.shape.length !== rank || v.shape.length !== rank ||
      output.shape.length !== rank || !sameShape(q.shape, output.shape)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires q/out [Q,D] or [B,Q,D] and rank-matched k/v tensors.`);
  }

  const batch = rank === 2 ? 1 : q.shape[0];
  const queries = q.shape[rank - 2];
  const keys = k.shape[rank - 2];
  const dModel = q.shape[rank - 1];
  const kBatch = rank === 2 ? 1 : k.shape[0];
  const vBatch = rank === 2 ? 1 : v.shape[0];
  if (kBatch !== batch || vBatch !== batch || k.shape[rank - 1] !== dModel ||
      v.shape[rank - 1] !== dModel || v.shape[rank - 2] !== keys ||
      qElements !== batch * queries * dModel || kElements !== batch * keys * dModel ||
      vElements !== batch * keys * dModel || outputElements !== qElements) {
    throw new Error(`QSDPA node ${nodeLabel(node)} has incompatible Q/K/V/output dimensions.`);
  }

  const { heads, headDim, causal, scale } = parseParams(node, dModel);
  const maskLayout = maskMode(mask, batch, queries, keys, node);
  if (storageRangesOverlap(q, output) || storageRangesOverlap(k, output) ||
      storageRangesOverlap(v, output) || (mask && storageRangesOverlap(mask, output))) {
    throw new Error(`QSDPA node ${nodeLabel(node)} requires output storage distinct from every input.`);
  }

  const qQuantization = requirePerTensorQuantization(q, 'q', node);
  const kQuantization = requirePerTensorQuantization(k, 'k', node);
  const vQuantization = requirePerTensorQuantization(v, 'v', node);
  const outputQuantization = requirePerTensorQuantization(output, 'output', node);
  const qScale = f32Scale(qQuantization.scale, 'q', node);
  const kScale = f32Scale(kQuantization.scale, 'k', node);
  const vScale = f32Scale(vQuantization.scale, 'v', node);
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const scoreMultiplier = Math.fround(Math.fround(qScale * kScale) * scale);
  if (!Number.isFinite(scoreMultiplier) || scoreMultiplier <= 0) {
    throw new Error(`QSDPA node ${nodeLabel(node)} q_scale * k_scale * scale must be finite positive F32.`);
  }
  const maximumRawDot = headDim * maximumCenteredMagnitude(q.dtype, qQuantization.zero_point) *
    maximumCenteredMagnitude(k.dtype, kQuantization.zero_point);
  const maximumScore = Math.fround(Math.fround(maximumRawDot) * scoreMultiplier);
  if (!Number.isFinite(maximumScore)) {
    throw new Error(`QSDPA node ${nodeLabel(node)} score range is not representable as finite F32.`);
  }
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;
  const accumulators = new Float32Array(headDim);

  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qBase = batchIndex * queries * dModel;
    const kvBase = batchIndex * keys * dModel;
    for (let head = 0; head < heads; head++) {
      const headOffset = head * headDim;
      for (let query = 0; query < queries; query++) {
        accumulators.fill(0);
        let runningMax = -Infinity;
        let runningSum = Math.fround(0);
        let hasVisibleKey = false;

        for (let key = 0; key < keys; key++) {
          if ((causal && key > query) || !maskKeeps(mask, maskLayout, batchIndex, query, key, queries, keys)) {
            continue;
          }
          let rawDot = 0;
          const qOffset = qBase + query * dModel + headOffset;
          const kOffset = kvBase + key * dModel + headOffset;
          for (let channel = 0; channel < headDim; channel++) {
            rawDot += centeredRaw(q.buffer[qOffset + channel], qQuantization.zero_point) *
              centeredRaw(k.buffer[kOffset + channel], kQuantization.zero_point);
          }
          const score = Math.fround(Math.fround(rawDot) * scoreMultiplier);
          const vOffset = kvBase + key * dModel + headOffset;

          if (!hasVisibleKey) {
            runningMax = score;
            runningSum = Math.fround(1);
            for (let channel = 0; channel < headDim; channel++) {
              accumulators[channel] = Math.fround(
                centeredRaw(v.buffer[vOffset + channel], vQuantization.zero_point) * vScale,
              );
            }
            hasVisibleKey = true;
          } else if (score > runningMax) {
            const previousWeight = Math.fround(Math.exp(Math.fround(runningMax - score)));
            runningSum = Math.fround(Math.fround(runningSum * previousWeight) + 1);
            for (let channel = 0; channel < headDim; channel++) {
              const value = Math.fround(
                centeredRaw(v.buffer[vOffset + channel], vQuantization.zero_point) * vScale,
              );
              accumulators[channel] = Math.fround(
                Math.fround(accumulators[channel] * previousWeight) + value,
              );
            }
            runningMax = score;
          } else {
            // Equal infinities are a real tie (one more unit of mass), not an
            // exp(-inf - -inf) NaN. The same first-valid-key discipline makes
            // completely masked rows stay at sum=0 below.
            const currentWeight = score === runningMax
              ? Math.fround(1)
              : Math.fround(Math.exp(Math.fround(score - runningMax)));
            runningSum = Math.fround(runningSum + currentWeight);
            for (let channel = 0; channel < headDim; channel++) {
              const value = Math.fround(
                centeredRaw(v.buffer[vOffset + channel], vQuantization.zero_point) * vScale,
              );
              accumulators[channel] = Math.fround(
                accumulators[channel] + Math.fround(currentWeight * value),
              );
            }
          }
        }

        const outputOffset = qBase + query * dModel + headOffset;
        if (!hasVisibleKey || runningSum === 0) {
          for (let channel = 0; channel < headDim; channel++) {
            output.buffer[outputOffset + channel] = outputQuantization.zero_point;
          }
          continue;
        }
        for (let channel = 0; channel < headDim; channel++) {
          const value = Math.fround(accumulators[channel] / runningSum);
          output.buffer[outputOffset + channel] = requantize(
            value, outputScale, outputQuantization.zero_point, minimum, maximum,
          );
        }
      }
    }
  }
}
