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
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} has an invalid element count.`);
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
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesByteStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes || tensor.buffer.length !== elements) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requirePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QGroupNorm node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} requires per_tensor quantization.`);
  }
  return descriptor;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requireAffineTensor(tensor, label, channels, node) {
  if (!tensor || tensor.dtype !== 'float32' || !Array.isArray(tensor.shape) ||
      tensor.shape.length !== 1 || tensor.shape[0] !== channels ||
      !(tensor.buffer instanceof Float32Array)) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} requires F32 shape [${channels}].`);
  }
  const expectedBytes = channels * Tensor.dtypeBytes('float32');
  if (tensor.sizeBytes !== expectedBytes || tensor.buffer.byteLength !== expectedBytes ||
      tensor.buffer.length !== channels) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label} has incompatible F32 storage.`);
  }
  for (let channel = 0; channel < channels; channel++) {
    if (!Number.isFinite(tensor.buffer[channel])) {
      throw new Error(`QGroupNorm node ${nodeLabel(node)} ${label}[${channel}] must be finite.`);
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
    throw new Error(`QGroupNorm node ${nodeLabel(node)} requires input, weight, and bias inputs.`);
  }
  const names = Object.keys(inputs).sort();
  if (names.length !== 3 || names[0] !== 'bias' || names[1] !== 'input' || names[2] !== 'weight') {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} requires exactly input, weight, and bias inputs.`);
  }
  return { input: inputs.input, weight: inputs.weight, bias: inputs.bias };
}

function canonicalOutput(node) {
  const outputs = node?.outputs;
  if (!isRecord(outputs) || Object.keys(outputs).length !== 1) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} requires exactly one output.`);
  }
  return outputs.out ?? Object.values(outputs)[0];
}

function parseParams(node) {
  const params = node?.params ?? {};
  if (!isRecord(params)) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} parameters must be an object.`);
  }
  for (const name of Object.keys(params)) {
    if (!['num_groups', 'eps', 'data_layout'].includes(name)) {
      throw new Error(`QGroupNorm node ${nodeLabel(node)} does not accept parameter '${name}'.`);
    }
  }
  const numGroups = params.num_groups;
  if (!Number.isInteger(numGroups) || numGroups <= 0) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} num_groups must be a positive integer.`);
  }
  if (params.data_layout != null && params.data_layout !== 'NHWC') {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} supports NHWC data_layout only.`);
  }
  const sourceEpsilon = params.eps ?? 1e-5;
  if (typeof sourceEpsilon !== 'number' || !Number.isFinite(sourceEpsilon)) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} eps must be a positive finite F32 value.`);
  }
  const epsilon = Math.fround(sourceEpsilon);
  if (!Number.isFinite(epsilon) || epsilon <= 0) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} eps must be a positive finite F32 value.`);
  }
  return { numGroups, epsilon };
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

// Canonical byte-domain GroupNorm for NHWC activations. Input and output own
// their per-tensor quantization mappings; statistics and affine work use only
// scalar F32 temporaries, so this portable implementation never materializes
// an F32 activation tensor. Validate every descriptor before the first output
// assignment so a rejected node cannot expose a partially rewritten boundary.
export function _cpuQGroupNorm(node) {
  const { numGroups, epsilon } = parseParams(node);
  const { input, weight, bias } = canonicalInputs(node);
  const output = canonicalOutput(node);
  const inputElements = requireByteTensor(input, 'input', node);
  const outputElements = requireByteTensor(output, 'output', node);
  if (input.shape.length !== 4 || output.shape.length !== 4 || !sameShape(input.shape, output.shape) ||
      inputElements !== outputElements) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} requires matching rank-4 NHWC input and output tensors.`);
  }

  const [batch, height, width, channels] = input.shape;
  if (channels % numGroups !== 0) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} num_groups must divide ${channels} channels.`);
  }
  const gamma = requireAffineTensor(weight, 'weight', channels, node);
  const beta = requireAffineTensor(bias, 'bias', channels, node);
  if (storageRangesOverlap(input, output) || storageRangesOverlap(weight, output) ||
      storageRangesOverlap(bias, output)) {
    throw new Error(`QGroupNorm node ${nodeLabel(node)} requires output storage distinct from every input.`);
  }

  const inputQuantization = requirePerTensorQuantization(input, 'input', node);
  const outputQuantization = requirePerTensorQuantization(output, 'output', node);
  const inputScale = f32Scale(inputQuantization.scale, 'input', node);
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const channelsPerGroup = channels / numGroups;
  const area = height * width;
  const valuesPerGroup = area * channelsPerGroup;
  const sampleStride = area * channels;
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;

  for (let sample = 0; sample < batch; sample++) {
    const sampleOffset = sample * sampleStride;
    for (let group = 0; group < numGroups; group++) {
      const firstChannel = group * channelsPerGroup;
      let meanRaw = Math.fround(0);
      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const raw = centeredRaw(input.buffer[offset + localChannel], inputQuantization.zero_point);
          meanRaw = Math.fround(meanRaw + raw);
        }
      }
      meanRaw = Math.fround(meanRaw / valuesPerGroup);

      // Keep the variance in raw-byte units until after the centered-square
      // reduction. Besides matching the GPU stats pass, this avoids the
      // catastrophic E[x^2] - E[x]^2 cancellation that nearly constant
      // high-valued U8 activations can trigger.
      let varianceRaw = Math.fround(0);
      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const raw = centeredRaw(input.buffer[offset + localChannel], inputQuantization.zero_point);
          const centered = Math.fround(raw - meanRaw);
          varianceRaw = Math.fround(varianceRaw + Math.fround(centered * centered));
        }
      }
      varianceRaw = Math.fround(varianceRaw / valuesPerGroup);
      const variance = Math.fround(Math.fround(Math.max(0, varianceRaw) * inputScale) * inputScale);
      const invStd = Math.fround(1 / Math.sqrt(Math.fround(variance + epsilon)));

      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const channel = firstChannel + localChannel;
          const raw = centeredRaw(input.buffer[offset + localChannel], inputQuantization.zero_point);
          const normalized = Math.fround(
            Math.fround(Math.fround(raw - meanRaw) * inputScale) * invStd,
          );
          const affine = Math.fround(Math.fround(normalized * gamma[channel]) + beta[channel]);
          output.buffer[offset + localChannel] = requantize(
            affine, outputScale, outputQuantization.zero_point, minimum, maximum,
          );
        }
      }
    }
  }
}
