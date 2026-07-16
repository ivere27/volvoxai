import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type { PerTensorQuantization, TensorQuantization } from '../types.js';

const INT32_MAX = 0x7fffffff;

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function elementCount(shape, label, node) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} has an unsupported element count.`);
  }
  return elements;
}

function matchesByteStorage(tensor) {
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  return tensor.dtype === 'uint8' &&
    (tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray);
}

function requireByteTensor(tensor, label, node) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype)) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = elementCount(tensor.shape, label, node);
  if (tensor.sizeBytes !== elements || !matchesByteStorage(tensor) ||
      tensor.buffer.byteLength !== elements || tensor.buffer.length !== elements) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requireMask(tensor, node) {
  if (!tensor || tensor.dtype !== 'int32' || !(tensor.buffer instanceof Int32Array)) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} mask requires unquantized I32 storage.`);
  }
  const elements = elementCount(tensor.shape, 'mask', node);
  if (tensor.quantization != null || tensor.sizeBytes !== elements * Int32Array.BYTES_PER_ELEMENT ||
      tensor.buffer.byteLength !== tensor.sizeBytes || tensor.buffer.length !== elements) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} mask requires unquantized I32 typed storage.`);
  }
  return elements;
}

function requireImmutablePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor?.quantization || !Object.isFrozen(tensor.quantization)) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} requires immutable per_tensor quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QMaskedMean node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} requires immutable per_tensor quantization metadata.`);
  }
  return descriptor;
}

function f32Positive(value, label, node) {
  const scalar = Math.fround(value);
  if (!Number.isFinite(scalar) || scalar <= 0) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scalar;
}

function maximumCenteredMagnitude(dtype, zeroPoint) {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}

function storageRangesOverlap(left, right) {
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

function canonicalDescriptor(node) {
  if (!isRecord(node?.inputs) || Object.keys(node.inputs).length !== 2 ||
      !node.inputs.input || !node.inputs.mask) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} requires exactly { input, mask }.`);
  }
  if (!isRecord(node?.outputs) || Object.keys(node.outputs).length !== 1 || !node.outputs.out) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} requires exactly { out }.`);
  }
  const parameterNames = Object.keys(node.params || {});
  if (!isRecord(node.params || {}) || parameterNames.length !== 0) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} accepts no parameters.`);
  }
  const { input, mask } = node.inputs;
  const output = node.outputs.out;
  const inputElements = requireByteTensor(input, 'input', node);
  const maskElements = requireMask(mask, node);
  const outputElements = requireByteTensor(output, 'output', node);
  if (input.shape.length !== 3 || mask.shape.length !== 2 || output.shape.length !== 2) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} requires input [B,S,D], mask [B,S], and output [B,D].`);
  }
  const [batch, sequence, width] = input.shape;
  if (mask.shape[0] !== batch || mask.shape[1] !== sequence ||
      output.shape[0] !== batch || output.shape[1] !== width ||
      inputElements !== batch * sequence * width || maskElements !== batch * sequence ||
      outputElements !== batch * width) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} has incompatible [B,S,D], [B,S], or [B,D] dimensions.`);
  }
  const inputQuantization = requireImmutablePerTensorQuantization(input, 'input', node);
  const outputQuantization = requireImmutablePerTensorQuantization(output, 'output', node);
  const inputScale = f32Positive(inputQuantization.scale, 'input', node);
  const outputScale = f32Positive(outputQuantization.scale, 'output', node);
  const multiplier = Math.fround(inputScale / outputScale);
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} input_scale / output_scale must be finite positive F32.`);
  }
  const maximumSum = sequence * maximumCenteredMagnitude(input.dtype, inputQuantization.zero_point);
  if (!Number.isSafeInteger(maximumSum) || maximumSum > INT32_MAX) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} sequence is too large for an exact I32 centered sum.`);
  }
  if (input === output || storageRangesOverlap(input, output) || storageRangesOverlap(mask, output)) {
    throw new Error(`QMaskedMean node ${nodeLabel(node)} requires output storage distinct from input and mask.`);
  }
  return {
    input, mask, output, batch, sequence, width, inputQuantization, outputQuantization, multiplier,
  };
}

function requantize(transformed, outputZeroPoint, minimum, maximum) {
  if (Number.isNaN(transformed)) return outputZeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

// Canonical byte-domain masked mean for hard routing and sequence pooling. The
// mask is I32 and nonzero values keep a token. The per-row sum remains exact
// I32; conversion/requantization is scalar private math, never an F32 graph
// edge.
export function _cpuQMaskedMean(node) {
  const {
    input, mask, output, batch, sequence, width, inputQuantization, outputQuantization, multiplier,
  } = canonicalDescriptor(node);
  const counts = new Int32Array(batch);
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    let count = 0;
    const maskOffset = batchIndex * sequence;
    for (let token = 0; token < sequence; token++) {
      count += mask.buffer[maskOffset + token] !== 0 ? 1 : 0;
    }
    counts[batchIndex] = count;
  }
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const count = counts[batchIndex];
    const inputBase = batchIndex * sequence * width;
    const maskBase = batchIndex * sequence;
    const outputBase = batchIndex * width;
    for (let column = 0; column < width; column++) {
      let sum = 0;
      if (count !== 0) {
        for (let token = 0; token < sequence; token++) {
          if (mask.buffer[maskBase + token] !== 0) {
            sum += input.buffer[inputBase + token * width + column] - inputQuantization.zero_point;
          }
        }
      }
      const mean = count === 0 ? 0 : Math.fround(Math.fround(sum) / Math.fround(count));
      // Preserve the portable kernel's F32 rounding schedule: mean, product,
      // then zero-point addition are distinct operations. This matters near a
      // ties-to-even quantization boundary.
      const scaled = Math.fround(mean * multiplier);
      const transformed = Math.fround(scaled + outputQuantization.zero_point);
      output.buffer[outputBase + column] = requantize(
        transformed, outputQuantization.zero_point, minimum, maximum,
      );
    }
  }
}
