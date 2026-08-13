import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
  sameShape,
} from './shapeKernelValidation.js';

function reductionDescriptor(node, operation) {
  const input = node.inputs?.input || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const elements = assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 1,
  });
  if (input.quantization != null) throw new Error(`${operation} input must be unquantized.`);
  const params = assertShapeKernelParams(node, ['axis', 'keepdims'], operation);
  const axis = normalizeShapeKernelAxis(params.axis, input.shape.length, -1, operation);
  if (axis !== input.shape.length - 1) throw new Error(`${operation} must reduce the last axis.`);
  const droppedShape = [...input.shape.slice(0, -1)];
  const keepShape = [...droppedShape, 1];
  // Legacy concrete graphs encoded keepdims only through their output shape.
  // Logical v1 supplies the canonical boolean/default before this kernel runs.
  const keepdims = params.keepdims ?? !sameShape(output?.shape ?? [], droppedShape);
  if (typeof keepdims !== 'boolean') throw new Error(`${operation} keepdims must be boolean.`);
  const expectedShape = keepdims ? keepShape : droppedShape;
  assertShapeKernelOutput(output, expectedShape, 'float32', undefined, operation);
  const axisSize = input.shape.at(-1);
  return { input, output, axisSize, rows: elements / axisSize };
}

export function _cpuReduceMean(node) {
  const { input, output, axisSize, rows } = reductionDescriptor(node, 'ReduceMean');
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let column = 0; column < axisSize; column++) {
      sum += input.buffer[row * axisSize + column];
    }
    output.buffer[row] = sum / axisSize;
  }
}
