import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type {
  PerAxisQuantization,
  PerTensorQuantization,
  TensorQuantization,
} from '../types.js';

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function tensorElements(tensor, label, node, { positiveRank = false } = {}) {
  if (!tensor || !Array.isArray(tensor.shape) || (positiveRank && tensor.shape.length === 0) ||
      tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} requires a valid${positiveRank ? ' positive-rank' : ''} shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} has an invalid element count.`);
  }
  return elements;
}

function matchesTypedStorage(tensor) {
  if (tensor.dtype === 'int32') return tensor.buffer instanceof Int32Array;
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  if (tensor.dtype === 'uint8') {
    return tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray;
  }
  return false;
}

function requireRawTensor(tensor, label, node, allowedDtypes, options) {
  if (!tensor || !allowedDtypes.includes(tensor.dtype)) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} requires ${allowedDtypes.join('/')} storage.`);
  }
  const elements = tensorElements(tensor, label, node, options);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesTypedStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requireQuantization(
  tensor: any,
  scheme: 'per_tensor',
  label: string,
  node: any,
): PerTensorQuantization;
function requireQuantization(
  tensor: any,
  scheme: 'per_axis',
  label: string,
  node: any,
): PerAxisQuantization;
function requireQuantization(
  tensor: any,
  scheme: 'per_tensor' | 'per_axis',
  label: string,
  node: any,
): TensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const quantization = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QEmbedding node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!quantization || quantization.scheme !== scheme) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} requires ${scheme} quantization.`);
  }
  return quantization;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requantize(value, scale, zeroPoint, minimum, maximum) {
  const transformed = Math.fround(Math.fround(value / scale) + zeroPoint);
  if (Number.isNaN(transformed)) return zeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

// Canonical W8A8 table lookup. Token IDs are ordinary I32 values; the table
// stores one I8/U8 quantization mapping per vocabulary row, and the result is
// immediately requantized into one per-tensor activation domain. Validate the
// entire ID vector before writing so an out-of-range ID cannot leave a partial
// result in the output tensor.
export function _cpuQEmbedding(node) {
  const input = node?.inputs?.input;
  const weight = node?.inputs?.weight;
  const output = node?.outputs?.out;

  const tokens = requireRawTensor(input, 'input IDs', node, ['int32'], { positiveRank: true });
  if (input.isInput !== true) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} requires token IDs to be a graph input so they can be preflighted before output writes.`);
  }
  const weightElements = requireRawTensor(weight, 'weight', node, ['int8', 'uint8'], { positiveRank: true });
  const outputElements = requireRawTensor(output, 'output', node, ['int8', 'uint8'], { positiveRank: true });
  if (weight.shape.length !== 2) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} requires rank-2 [vocab,hidden] weights.`);
  }
  const [vocab, hidden] = weight.shape;
  const expectedOutputShape = [...input.shape, hidden];
  if (output.shape.length !== expectedOutputShape.length ||
      output.shape.some((dimension, index) => dimension !== expectedOutputShape[index]) ||
      weightElements !== vocab * hidden || outputElements !== tokens * hidden) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} has incompatible ID, [vocab,hidden] weight, or output dimensions.`);
  }

  const outputQuantization = requireQuantization(output, 'per_tensor', 'output', node);
  const weightQuantization = requireQuantization(weight, 'per_axis', 'weight', node);
  if (weightQuantization.axis !== 0 || weightQuantization.scales.length !== vocab ||
      weightQuantization.zero_points.length !== vocab) {
    throw new Error(`QEmbedding node ${nodeLabel(node)} requires per_axis weight quantization along axis 0.`);
  }
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const rowScales = new Float32Array(vocab);
  for (let row = 0; row < vocab; row++) {
    rowScales[row] = f32Scale(weightQuantization.scales[row], `weight[${row}]`, node);
  }

  // This is deliberately before the first output assignment. It is observable
  // to callers that reuse an activation buffer after a rejected token batch.
  for (let token = 0; token < tokens; token++) {
    const id = input.buffer[token];
    if (id < 0 || id >= vocab) {
      throw new Error(`QEmbedding node ${nodeLabel(node)} token id ${id} is outside vocabulary size ${vocab}.`);
    }
  }

  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;
  for (let token = 0; token < tokens; token++) {
    const row = input.buffer[token];
    const rowScale = rowScales[row];
    const rowZeroPoint = weightQuantization.zero_points[row];
    const weightOffset = row * hidden;
    const outputOffset = token * hidden;
    for (let column = 0; column < hidden; column++) {
      const centered = Math.fround(weight.buffer[weightOffset + column] - rowZeroPoint);
      const dequantized = Math.fround(centered * rowScale);
      output.buffer[outputOffset + column] = requantize(
        dequantized, outputScale, outputQuantization.zero_point, minimum, maximum,
      );
    }
  }
}
