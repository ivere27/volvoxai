import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type { PerTensorQuantization, TensorLike, TensorQuantization } from '../types.js';

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function tensorElements(tensor, label, node) {
  if (!tensor || !Array.isArray(tensor.shape) ||
      tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} has an invalid element count.`);
  }
  return elements;
}

function matchesTypedStorage(tensor) {
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  if (tensor.dtype === 'uint8') {
    return tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray;
  }
  return false;
}

function requireByteTensor(tensor, label, node) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype)) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesTypedStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requirePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QSiLU node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} requires per_tensor quantization.`);
  }
  return descriptor;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QSiLU node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requantize(value, outputScale, outputZeroPoint, minimum, maximum) {
  const transformed = Math.fround(Math.fround(value / outputScale) + outputZeroPoint);
  if (Number.isNaN(transformed)) return outputZeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

function canonicalInput(node) {
  const aliases = ['input', 'x', 'data'];
  const names = Object.keys(node?.inputs || {});
  const matches = aliases.filter((name) => node.inputs?.[name]);
  if (names.length !== 1 || matches.length !== 1) {
    throw new Error(`QSiLU node ${nodeLabel(node)} requires exactly one activation input (input/x/data).`);
  }
  return node.inputs[matches[0]];
}

function canonicalOutput(node): TensorLike {
  const outputs = Object.values(node?.outputs || {}).filter(Boolean) as TensorLike[];
  if (outputs.length !== 1) {
    throw new Error(`QSiLU node ${nodeLabel(node)} requires exactly one output.`);
  }
  return outputs[0];
}

// Canonical byte-domain SiLU.  The activation mapping belongs to each tensor:
// x = (raw - input_zero_point) * input_scale, then x * sigmoid(x), followed
// by a direct ties-to-even, saturating requantization into the output bytes.
// Validate the whole descriptor before the first output assignment so malformed
// calls cannot expose a partially rewritten activation boundary.
export function _cpuQSiLU(node) {
  if (Object.keys(node?.params || {}).length !== 0) {
    throw new Error(`QSiLU node ${nodeLabel(node)} does not accept parameters.`);
  }
  const input = canonicalInput(node);
  const output = canonicalOutput(node);
  const inputElements = requireByteTensor(input, 'input', node);
  const outputElements = requireByteTensor(output, 'output', node);
  if (inputElements !== outputElements || !sameShape(input.shape, output.shape)) {
    throw new Error(`QSiLU node ${nodeLabel(node)} requires same-shape input and output tensors.`);
  }
  if (input.buffer === output.buffer) {
    throw new Error(`QSiLU node ${nodeLabel(node)} requires distinct input and output byte storage.`);
  }
  const inputQuantization = requirePerTensorQuantization(input, 'input', node);
  const outputQuantization = requirePerTensorQuantization(output, 'output', node);
  const inputScale = f32Scale(inputQuantization.scale, 'input', node);
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;
  const inputBuffer = input.buffer as Int8Array | Uint8Array | Uint8ClampedArray;
  const outputBuffer = output.buffer as Int8Array | Uint8Array | Uint8ClampedArray;

  for (let index = 0; index < inputElements; index++) {
    const centered = Math.fround(inputBuffer[index] - inputQuantization.zero_point);
    const x = Math.fround(centered * inputScale);
    const denominator = Math.fround(1 + Math.fround(Math.exp(-x)));
    const silu = Math.fround(x / denominator);
    outputBuffer[index] = requantize(
      silu, outputScale, outputQuantization.zero_point, minimum, maximum,
    );
  }
}
