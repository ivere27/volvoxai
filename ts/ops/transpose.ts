import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  remapShapeKernelQuantization,
} from './shapeKernelValidation.js';

export function _cpuTranspose(node) {
  const input = node.inputs?.input || node.inputs?.x || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const elements = assertShapeKernelTensor(input, 'Transpose input', {
    minimumRank: 1, maximumRank: 8,
  });
  const params = assertShapeKernelParams(node, ['perm'], 'Transpose');
  const permutation = params.perm ?? [...Array(input.shape.length).keys()].reverse();
  if (!Array.isArray(permutation) || permutation.length !== input.shape.length ||
      new Set(permutation).size !== permutation.length ||
      permutation.some((axis) => !Number.isInteger(axis) || axis < 0 || axis >= input.shape.length)) {
    throw new Error(`Transpose node ${node.id ?? '<unnamed>'} has an invalid permutation.`);
  }
  const outputShape = permutation.map((axis) => input.shape[axis]);
  const outputQuantization = input.quantization?.scheme === 'per_axis'
    ? remapShapeKernelQuantization(
      input.quantization,
      permutation.indexOf(input.quantization.axis),
    )
    : input.quantization;
  assertShapeKernelOutput(
    output,
    outputShape,
    input.dtype,
    outputQuantization,
    'Transpose',
  );

  const inputStrides = new Array(input.shape.length);
  let stride = 1;
  for (let axis = input.shape.length - 1; axis >= 0; axis--) {
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  const outputStrides = new Array(outputShape.length);
  stride = 1;
  for (let axis = outputShape.length - 1; axis >= 0; axis--) {
    outputStrides[axis] = stride;
    stride *= outputShape[axis];
  }
  for (let outputIndex = 0; outputIndex < elements; outputIndex++) {
    let inputIndex = 0;
    let remaining = outputIndex;
    for (let axis = 0; axis < outputShape.length; axis++) {
      const coordinate = Math.floor(remaining / outputStrides[axis]);
      remaining %= outputStrides[axis];
      inputIndex += coordinate * inputStrides[permutation[axis]];
    }
    output.buffer[outputIndex] = input.buffer[inputIndex];
  }
}
