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
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} has an invalid element count.`);
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
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesByteStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes || tensor.buffer.length !== elements) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requirePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QLayerNorm node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} requires per_tensor quantization.`);
  }
  return descriptor;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requireAffineTensor(tensor, label, dModel, node) {
  if (!tensor || tensor.dtype !== 'float32' || !Array.isArray(tensor.shape) ||
      tensor.shape.length !== 1 || tensor.shape[0] !== dModel ||
      !(tensor.buffer instanceof Float32Array)) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} requires F32 shape [${dModel}].`);
  }
  const expectedBytes = dModel * Tensor.dtypeBytes('float32');
  if (tensor.sizeBytes !== expectedBytes || tensor.buffer.byteLength !== expectedBytes ||
      tensor.buffer.length !== dModel) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label} has incompatible F32 storage.`);
  }
  for (let channel = 0; channel < dModel; channel++) {
    if (!Number.isFinite(tensor.buffer[channel])) {
      throw new Error(`QLayerNorm node ${nodeLabel(node)} ${label}[${channel}] must be finite.`);
    }
  }
  return tensor.buffer;
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
    throw new Error(`QLayerNorm node ${nodeLabel(node)} requires input, weight, and bias inputs.`);
  }
  const names = Object.keys(inputs).sort();
  if (names.length !== 3 || names[0] !== 'bias' || names[1] !== 'input' || names[2] !== 'weight') {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} requires exactly input, weight, and bias inputs.`);
  }
  return { input: inputs.input, weight: inputs.weight, bias: inputs.bias };
}

function canonicalOutput(node) {
  const outputs = node?.outputs;
  if (!isRecord(outputs) || Object.keys(outputs).length !== 1) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} requires exactly one output.`);
  }
  return outputs.out ?? Object.values(outputs)[0];
}

function parseParams(node, dModel) {
  const params = node?.params ?? {};
  if (!isRecord(params)) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} parameters must be an object.`);
  }
  for (const name of Object.keys(params)) {
    if (!['eps', 'd_model'].includes(name)) {
      throw new Error(`QLayerNorm node ${nodeLabel(node)} does not accept parameter '${name}'.`);
    }
  }
  const epsilonSource = params.eps ?? 1e-5;
  if (typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource)) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} eps must be a positive finite F32 value.`);
  }
  const epsilon = Math.fround(epsilonSource);
  if (!Number.isFinite(epsilon) || epsilon <= 0) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} eps must be a positive finite F32 value.`);
  }
  if (params.d_model != null && (!Number.isInteger(params.d_model) || params.d_model !== dModel)) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} d_model must match the last input dimension (${dModel}).`);
  }
  return { epsilon };
}

function requantize(value, outputScale, outputZeroPoint, minimum, maximum) {
  const transformed = Math.fround(Math.fround(value / outputScale) + outputZeroPoint);
  if (Number.isNaN(transformed)) return outputZeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

function centeredRaw(raw, zeroPoint) {
  return Math.fround(raw - zeroPoint);
}

// Canonical byte-domain LayerNorm for [..., D] activations. The reduction
// stays in raw-byte coordinates until a stable centered variance is known;
// only scalar F32 temporaries are decoded and no F32 activation is created.
// Validate every descriptor before the first output assignment so malformed
// nodes cannot leave a partially rewritten byte-domain boundary.
export function _cpuQLayerNorm(node) {
  const { input, weight, bias } = canonicalInputs(node);
  const output = canonicalOutput(node);
  const inputElements = requireByteTensor(input, 'input', node);
  const outputElements = requireByteTensor(output, 'output', node);
  if (input.shape.length < 1 || output.shape.length < 1 || !sameShape(input.shape, output.shape) ||
      inputElements !== outputElements) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} requires matching rank-at-least-1 input and output tensors.`);
  }

  const dModel = input.shape.at(-1);
  const rows = inputElements / dModel;
  if (!Number.isSafeInteger(rows) || rows <= 0) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} has an invalid row count.`);
  }
  const { epsilon } = parseParams(node, dModel);
  const gamma = requireAffineTensor(weight, 'weight', dModel, node);
  const beta = requireAffineTensor(bias, 'bias', dModel, node);
  if (storageRangesOverlap(input, output) || storageRangesOverlap(weight, output) ||
      storageRangesOverlap(bias, output)) {
    throw new Error(`QLayerNorm node ${nodeLabel(node)} requires output storage distinct from every input.`);
  }

  const inputQuantization = requirePerTensorQuantization(input, 'input', node);
  const outputQuantization = requirePerTensorQuantization(output, 'output', node);
  const inputScale = f32Scale(inputQuantization.scale, 'input', node);
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;

  for (let row = 0; row < rows; row++) {
    const offset = row * dModel;
    let meanRaw = Math.fround(0);
    for (let channel = 0; channel < dModel; channel++) {
      const raw = centeredRaw(input.buffer[offset + channel], inputQuantization.zero_point);
      meanRaw = Math.fround(meanRaw + raw);
    }
    meanRaw = Math.fround(meanRaw / dModel);

    // Center first in raw-byte units. E[x^2] - E[x]^2 can erase a valid small
    // variance for high U8 inputs, whereas this biased two-pass form preserves
    // it and matches the paired WebGPU statistics pass.
    let varianceRaw = Math.fround(0);
    for (let channel = 0; channel < dModel; channel++) {
      const raw = centeredRaw(input.buffer[offset + channel], inputQuantization.zero_point);
      const centered = Math.fround(raw - meanRaw);
      varianceRaw = Math.fround(varianceRaw + Math.fround(centered * centered));
    }
    varianceRaw = Math.fround(varianceRaw / dModel);
    const variance = Math.fround(Math.fround(Math.max(0, varianceRaw) * inputScale) * inputScale);
    const invStd = Math.fround(1 / Math.sqrt(Math.fround(variance + epsilon)));

    for (let channel = 0; channel < dModel; channel++) {
      const raw = centeredRaw(input.buffer[offset + channel], inputQuantization.zero_point);
      const normalized = Math.fround(
        Math.fround(Math.fround(raw - meanRaw) * inputScale) * invStd,
      );
      const affine = Math.fround(Math.fround(normalized * gamma[channel]) + beta[channel]);
      output.buffer[offset + channel] = requantize(
        affine, outputScale, outputQuantization.zero_point, minimum, maximum,
      );
    }
  }
}
