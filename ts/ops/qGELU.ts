import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type { PerTensorQuantization, TensorLike, TensorQuantization } from '../types.js';

const SQRT_1_2 = Math.fround(0.7071067811865476);

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
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} has an invalid element count.`);
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
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} requires I8/U8 storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesTypedStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes) {
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
  }
  return elements;
}

function requirePerTensorQuantization(tensor, label, node): PerTensorQuantization {
  if (!tensor.quantization) {
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QGELU node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} requires per_tensor quantization.`);
  }
  return descriptor;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QGELU node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requantize(value, outputScale, outputZeroPoint, minimum, maximum) {
  const scaled = Math.fround(value / outputScale);
  const transformed = Math.fround(scaled + outputZeroPoint);
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
    throw new Error(`QGELU node ${nodeLabel(node)} requires exactly one activation input (input/x/data).`);
  }
  return node.inputs[matches[0]];
}

function canonicalOutput(node): TensorLike {
  const outputs = Object.values(node?.outputs || {}).filter(Boolean) as TensorLike[];
  if (outputs.length !== 1) {
    throw new Error(`QGELU node ${nodeLabel(node)} requires exactly one output.`);
  }
  return outputs[0];
}

function hasCanonicalParameters(params) {
  const values = params || {};
  const names = Object.keys(values);
  return names.length === 0 ||
    (names.length === 1 && names[0] === 'approximate' && values.approximate === 'none');
}

// This is deliberately the same Abramowitz-Stegun 7.1.26 polynomial and F32
// operation order as qGELUInt8.wgsl and qgelu_i8u8. QGELU accepts either no
// parameters or exactly approximate="none"; both spellings select its one
// portable erf semantic without depending on host erff().
export function qgeluErfApproxF32(value) {
  const sign = value >= 0 ? Math.fround(1) : Math.fround(-1);
  const x = Math.fround(Math.abs(value));
  const tDenominator = Math.fround(Math.fround(1) + Math.fround(Math.fround(0.3275911) * x));
  const t = Math.fround(Math.fround(1) / tDenominator);
  let polynomial = Math.fround(Math.fround(1.061405429) * t);
  polynomial = Math.fround(polynomial - Math.fround(1.453152027));
  polynomial = Math.fround(polynomial * t);
  polynomial = Math.fround(polynomial + Math.fround(1.421413741));
  polynomial = Math.fround(polynomial * t);
  polynomial = Math.fround(polynomial - Math.fround(0.284496736));
  polynomial = Math.fround(polynomial * t);
  polynomial = Math.fround(polynomial + Math.fround(0.254829592));
  polynomial = Math.fround(polynomial * t);
  const squared = Math.fround(x * x);
  const tail = Math.fround(Math.exp(Math.fround(-squared)));
  return Math.fround(sign * Math.fround(Math.fround(1) - Math.fround(polynomial * tail)));
}

export function qgeluValueF32(value) {
  const x = Math.fround(value);
  const erfInput = Math.fround(x * SQRT_1_2);
  const cdf = Math.fround(Math.fround(0.5) * Math.fround(Math.fround(1) + qgeluErfApproxF32(erfInput)));
  return Math.fround(x * cdf);
}

// Canonical byte-domain GELU. The input and output retain their declared I8/U8
// representations; only one scalar is decoded to F32 at a time and immediately
// requantized with ties-to-even, saturating output semantics. Validate the full
// descriptor before any output write so malformed calls cannot partly rewrite a
// byte-domain activation boundary.
export function _cpuQGELU(node) {
  if (!hasCanonicalParameters(node?.params)) {
    throw new Error(`QGELU node ${nodeLabel(node)} accepts only omitted parameters or approximate='none'.`);
  }
  const input = canonicalInput(node);
  const output = canonicalOutput(node);
  const inputElements = requireByteTensor(input, 'input', node);
  const outputElements = requireByteTensor(output, 'output', node);
  if (inputElements !== outputElements || !sameShape(input.shape, output.shape)) {
    throw new Error(`QGELU node ${nodeLabel(node)} requires same-shape input and output tensors.`);
  }
  if (input.buffer === output.buffer) {
    throw new Error(`QGELU node ${nodeLabel(node)} requires distinct input and output byte storage.`);
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
    outputBuffer[index] = requantize(
      qgeluValueF32(x), outputScale, outputQuantization.zero_point, minimum, maximum,
    );
  }
}
