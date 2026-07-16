import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type {
  PerAxisQuantization,
  PerTensorQuantization,
  TensorQuantization,
} from '../types.js';

const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function tensorElements(tensor, label, node) {
  if (!tensor || !Array.isArray(tensor.shape) || tensor.shape.length === 0 ||
      tensor.shape.some((dimension) => !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} requires a positive-rank shape.`);
  }
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} has an invalid element count.`);
  }
  return elements;
}

function matchesTypedStorage(tensor) {
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  if (tensor.dtype === 'uint8') {
    return tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray;
  }
  if (tensor.dtype === 'int32') return tensor.buffer instanceof Int32Array;
  return false;
}

function requireRawTensor(tensor, label, node, allowedDtypes) {
  if (!tensor || !allowedDtypes.includes(tensor.dtype)) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} requires ${allowedDtypes.join('/')} storage.`);
  }
  const elements = tensorElements(tensor, label, node);
  const expectedBytes = elements * Tensor.dtypeBytes(tensor.dtype);
  if (tensor.sizeBytes !== expectedBytes || !matchesTypedStorage(tensor) ||
      tensor.buffer.byteLength !== expectedBytes) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} has incompatible typed storage.`);
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
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} requires immutable quantization metadata.`);
  }
  const quantization = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QLinear node ${nodeLabel(node)} ${label} quantization`) as TensorQuantization | null;
  if (!quantization || quantization.scheme !== scheme) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} requires ${scheme} quantization.`);
  }
  return quantization;
}

function f32Scale(value, label, node) {
  const scale = Math.fround(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`QLinear node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`);
  }
  return scale;
}

function requantize(accumulator, multiplier, zeroPoint, minimum, maximum) {
  const scaled = Math.fround(Math.fround(accumulator) * multiplier);
  const transformed = Math.fround(scaled + zeroPoint);
  if (Number.isNaN(transformed)) return zeroPoint;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

// Canonical W8A8 dense reference. The graph owns all quantization metadata:
// activation tensors are per-tensor I8/U8, while the [d_out,d_in] weight uses
// per-output-channel quantization along axis 0. Bias is already quantized in
// the I32 accumulator domain (input_scale * weight_scale[output_channel]).
export function _cpuQLinear(node) {
  const input = node?.inputs?.input;
  const weight = node?.inputs?.weight;
  const bias = node?.inputs?.bias;
  const output = node?.outputs?.out;

  const inputElements = requireRawTensor(input, 'input', node, ['int8', 'uint8']);
  const weightElements = requireRawTensor(weight, 'weight', node, ['int8', 'uint8']);
  const outputElements = requireRawTensor(output, 'output', node, ['int8', 'uint8']);
  const biasElements = requireRawTensor(bias, 'bias', node, ['int32']);

  if (input.shape.length !== output.shape.length || input.shape.length < 1 ||
      input.shape.slice(0, -1).some((dimension, index) => dimension !== output.shape[index])) {
    throw new Error(`QLinear node ${nodeLabel(node)} requires matching input/output outer dimensions.`);
  }
  if (weight.shape.length !== 2) {
    throw new Error(`QLinear node ${nodeLabel(node)} requires rank-2 [d_out,d_in] weights.`);
  }
  const dIn = input.shape.at(-1);
  const dOut = output.shape.at(-1);
  const rows = inputElements / dIn;
  if (!Number.isInteger(rows) || weight.shape[0] !== dOut || weight.shape[1] !== dIn ||
      weightElements !== dOut * dIn || outputElements !== rows * dOut ||
      bias.shape.length !== 1 || bias.shape[0] !== dOut || biasElements !== dOut) {
    throw new Error(`QLinear node ${nodeLabel(node)} has incompatible [d_out,d_in] dimensions or I32 bias.`);
  }

  const inputQuantization = requireQuantization(input, 'per_tensor', 'input', node);
  const outputQuantization = requireQuantization(output, 'per_tensor', 'output', node);
  const weightQuantization = requireQuantization(weight, 'per_axis', 'weight', node);
  if (weightQuantization.axis !== 0 || weightQuantization.scales.length !== dOut ||
      weightQuantization.zero_points.length !== dOut) {
    throw new Error(`QLinear node ${nodeLabel(node)} requires per_axis weight quantization along axis 0.`);
  }

  const inputScale = f32Scale(inputQuantization.scale, 'input', node);
  const outputScale = f32Scale(outputQuantization.scale, 'output', node);
  const outputMinimum = output.dtype === 'int8' ? -128 : 0;
  const outputMaximum = output.dtype === 'int8' ? 127 : 255;
  for (let outputChannel = 0; outputChannel < dOut; outputChannel++) {
    const weightScale = f32Scale(weightQuantization.scales[outputChannel], `weight[${outputChannel}]`, node);
    const multiplier = Math.fround(Math.fround(inputScale * weightScale) / outputScale);
    const weightZeroPoint = weightQuantization.zero_points[outputChannel];
    for (let row = 0; row < rows; row++) {
      let accumulator = bias.buffer[outputChannel];
      const inputBase = row * dIn;
      const weightBase = outputChannel * dIn;
      for (let inputChannel = 0; inputChannel < dIn; inputChannel++) {
        accumulator += (input.buffer[inputBase + inputChannel] - inputQuantization.zero_point) *
          (weight.buffer[weightBase + inputChannel] - weightZeroPoint);
        if (accumulator < INT32_MIN || accumulator > INT32_MAX) {
          throw new Error(`QLinear node ${nodeLabel(node)} I32 accumulator overflows at row ${row}, output ${outputChannel}.`);
        }
      }
      output.buffer[row * dOut + outputChannel] = requantize(
        accumulator, multiplier, outputQuantization.zero_point, outputMinimum, outputMaximum,
      );
    }
  }
}
