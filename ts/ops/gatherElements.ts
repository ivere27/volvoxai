import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
  sliceShapeKernelQuantization,
} from './shapeKernelValidation.js';

function strides(shape) {
  const result = new Array(shape.length);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    result[axis] = stride;
    stride *= shape[axis];
  }
  return result;
}

export function _cpuGatherElements(node) {
  const input = node.inputs?.input || node.inputs?.data;
  const indices = node.inputs?.indices;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  assertShapeKernelTensor(input, 'GatherElements input', { minimumRank: 1 });
  const outputElements = assertShapeKernelTensor(indices, 'GatherElements indices', {
    dtypes: ['int32'], minimumRank: 1,
  });
  if (indices.quantization != null) {
    throw new Error('GatherElements indices must be unquantized I32.');
  }
  if (indices.shape.length !== input.shape.length) {
    throw new Error('GatherElements input and indices must have equal rank.');
  }
  const params = assertShapeKernelParams(node, ['axis'], 'GatherElements');
  const axis = normalizeShapeKernelAxis(
    params.axis,
    input.shape.length,
    0,
    'GatherElements',
  );
  for (let dimension = 0; dimension < input.shape.length; dimension++) {
    if (dimension !== axis && indices.shape[dimension] > input.shape[dimension]) {
      throw new Error(
        `GatherElements indices axis ${dimension} exceeds the corresponding input extent.`,
      );
    }
  }
  let expectedQuantization = input.quantization;
  if (input.quantization?.scheme === 'per_axis') {
    if (input.quantization.axis === axis) {
      throw new Error('GatherElements cannot reorder a per-axis quantization dimension.');
    }
    const extent = indices.shape[input.quantization.axis];
    expectedQuantization = sliceShapeKernelQuantization(
      input.quantization,
      input.quantization.axis,
      0,
      extent,
    );
  }
  assertShapeKernelOutput(
    output,
    indices.shape,
    input.dtype,
    expectedQuantization,
    'GatherElements',
  );

  const axisExtent = input.shape[axis];
  for (let index = 0; index < indices.buffer.length; index++) {
    const normalized = indices.buffer[index] < 0
      ? indices.buffer[index] + axisExtent
      : indices.buffer[index];
    if (normalized < 0 || normalized >= axisExtent) {
      throw new Error(
        `GatherElements index ${indices.buffer[index]} is outside axis extent ${axisExtent}.`,
      );
    }
  }

  const inputStrides = strides(input.shape);
  const indexStrides = strides(indices.shape);
  const coordinates = new Array(indices.shape.length);
  for (let linear = 0; linear < outputElements; linear++) {
    let remaining = linear;
    for (let dimension = 0; dimension < indices.shape.length; dimension++) {
      coordinates[dimension] = Math.floor(remaining / indexStrides[dimension]);
      remaining %= indexStrides[dimension];
    }
    let gathered = indices.buffer[linear];
    if (gathered < 0) gathered += axisExtent;
    let inputOffset = 0;
    for (let dimension = 0; dimension < input.shape.length; dimension++) {
      inputOffset += (dimension === axis ? gathered : coordinates[dimension]) *
        inputStrides[dimension];
    }
    output.buffer[linear] = input.buffer[inputOffset];
  }
}
