import { roundTiesToEven } from './quantizeLinear.js';

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function quantization(tensor, label) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) ||
      tensor.quantization?.scheme !== 'per_tensor') {
    throw new Error(`${label} must be an I8/U8 tensor with per_tensor quantization metadata.`);
  }
  return tensor.quantization;
}

function quantizeToTensor(value, tensor, descriptor) {
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  const transformed = Math.fround(Math.fround(value / descriptor.scale) + descriptor.zero_point);
  if (Number.isNaN(transformed)) return descriptor.zero_point;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

function reluClamp(value, tensor, descriptor, relu) {
  if (!relu) return value;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  let clamped = Math.max(value, descriptor.zero_point);
  if (relu >= 2) {
    const transformed = Math.fround(Math.fround(6 / descriptor.scale) + descriptor.zero_point);
    const upper = transformed <= minimum ? minimum : transformed >= maximum ? maximum : roundTiesToEven(transformed);
    clamped = Math.min(clamped, upper);
  }
  return clamped;
}

// Exact-shape canonical QAdd. Broadcasting is deliberately excluded from the
// first W8A8 contract: exporters must lower a broadcast to an explicit typed
// expansion before QAdd so every backend has identical descriptor semantics.
export function _cpuQAdd(node) {
  const a = node.inputs.a || node.inputs.input || node.inputs.x;
  const b = node.inputs.b || node.inputs.y;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (!a || !b || !output || !sameShape(a.shape, b.shape) || !sameShape(a.shape, output.shape) ||
      a.buffer?.length !== b.buffer?.length || a.buffer?.length !== output.buffer?.length) {
    throw new Error(`QAdd node ${node.id} requires exact-shape input/output tensors with matching storage lengths.`);
  }
  const qa = quantization(a, `QAdd node ${node.id} input a`);
  const qb = quantization(b, `QAdd node ${node.id} input b`);
  const qo = quantization(output, `QAdd node ${node.id} output`);
  const relu = node.params?.relu ?? 0;
  if (!Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error(`QAdd node ${node.id} supports relu values 0 (none), 1 (ReLU), or 2 (ReLU6).`);
  }
  for (let index = 0; index < output.buffer.length; index++) {
    const av = Math.fround(Math.fround(a.buffer[index] - qa.zero_point) * qa.scale);
    const bv = Math.fround(Math.fround(b.buffer[index] - qb.zero_point) * qb.scale);
    output.buffer[index] = reluClamp(quantizeToTensor(Math.fround(av + bv), output, qo), output, qo, relu);
  }
}
