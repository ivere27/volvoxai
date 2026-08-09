import type { RuntimeDType } from '../types.js';
import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';

function strides(shape) {
  const out = new Array(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index--) {
    out[index] = stride;
    stride *= shape[index];
  }
  return out;
}

export function cpuBroadcastBinary(
  node,
  operation,
  operationName,
  options: Readonly<{ dtypes?: readonly RuntimeDType[]; maximumRank?: number }> = {},
) {
  const a = node.inputs.a;
  const b = node.inputs.b;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const dtypes = options.dtypes;
  assertShapeKernelTensor(a, `${operationName} input a`, {
    ...(dtypes === undefined ? {} : { dtypes }),
    ...(options.maximumRank === undefined ? {} : { maximumRank: options.maximumRank }),
  });
  assertShapeKernelTensor(b, `${operationName} input b`, {
    dtypes: [a.dtype],
    ...(options.maximumRank === undefined ? {} : { maximumRank: options.maximumRank }),
  });
  if (a.quantization != null || b.quantization != null) {
    throw new Error(`${operationName} does not accept quantized arithmetic inputs.`);
  }
  const rank = Math.max(a.shape.length, b.shape.length);
  const aOffset = rank - a.shape.length;
  const bOffset = rank - b.shape.length;
  const expectedShape = new Array(rank);
  for (let dimension = 0; dimension < rank; dimension++) {
    const ad = dimension < aOffset ? 1 : a.shape[dimension - aOffset];
    const bd = dimension < bOffset ? 1 : b.shape[dimension - bOffset];
    const expected = Math.max(ad, bd);
    if (ad !== 1 && bd !== 1 && ad !== bd) {
      throw new Error(`${operationName} shapes [${a.shape}] and [${b.shape}] do not broadcast to [${output.shape}].`);
    }
    expectedShape[dimension] = expected;
  }
  assertShapeKernelOutput(output, expectedShape, a.dtype, undefined, operationName);

  if (a.buffer.length === output.buffer.length && b.buffer.length === output.buffer.length) {
    for (let index = 0; index < output.buffer.length; index++) {
      output.buffer[index] = operation(a.buffer[index], b.buffer[index]);
    }
    return;
  }
  if (a.buffer.length === 1 && b.buffer.length === output.buffer.length) {
    const scalar = a.buffer[0];
    for (let index = 0; index < output.buffer.length; index++) output.buffer[index] = operation(scalar, b.buffer[index]);
    return;
  }
  if (b.buffer.length === 1 && a.buffer.length === output.buffer.length) {
    const scalar = b.buffer[0];
    for (let index = 0; index < output.buffer.length; index++) output.buffer[index] = operation(a.buffer[index], scalar);
    return;
  }

  const aStrides = strides(a.shape);
  const bStrides = strides(b.shape);
  for (let outputIndex = 0; outputIndex < output.buffer.length; outputIndex++) {
    let remaining = outputIndex;
    let aIndex = 0;
    let bIndex = 0;
    for (let dimension = rank - 1; dimension >= 0; dimension--) {
      const coordinate = remaining % output.shape[dimension];
      remaining = Math.floor(remaining / output.shape[dimension]);
      if (dimension >= aOffset && a.shape[dimension - aOffset] !== 1) {
        aIndex += coordinate * aStrides[dimension - aOffset];
      }
      if (dimension >= bOffset && b.shape[dimension - bOffset] !== 1) {
        bIndex += coordinate * bStrides[dimension - bOffset];
      }
    }
    output.buffer[outputIndex] = operation(a.buffer[aIndex], b.buffer[bIndex]);
  }
}
