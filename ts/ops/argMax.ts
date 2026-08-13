import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
} from './shapeKernelValidation.js';

export function _cpuArgMax(node) {
  const input = node.inputs?.input || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  assertShapeKernelTensor(input, 'ArgMax input', {
    dtypes: ['float32'], minimumRank: 1,
  });
  if (input.quantization != null) throw new Error('ArgMax input must be unquantized F32.');
  const params = assertShapeKernelParams(
    node,
    ['axis', 'keepdims', 'select_last_index'],
    'ArgMax',
  );
  const axis = normalizeShapeKernelAxis(params.axis, input.shape.length, 0, 'ArgMax');
  const keepdims = params.keepdims ?? true;
  if (typeof keepdims !== 'boolean') throw new Error('ArgMax keepdims must be boolean.');
  if ((params.select_last_index ?? 0) !== 0) {
    throw new Error('ArgMax select_last_index must be 0 so ties select the first index.');
  }
  const expectedShape = [
    ...input.shape.slice(0, axis),
    ...(keepdims ? [1] : []),
    ...input.shape.slice(axis + 1),
  ];
  assertShapeKernelOutput(output, expectedShape, 'int32', undefined, 'ArgMax');

  const axisSize = input.shape[axis];
  const innerBlock = input.shape.slice(axis + 1)
    .reduce((product, dimension) => product * dimension, 1);
  const outerBlock = input.shape.slice(0, axis)
    .reduce((product, dimension) => product * dimension, 1);
  let outputIndex = 0;
  for (let outer = 0; outer < outerBlock; outer++) {
    for (let inner = 0; inner < innerBlock; inner++) {
      const base = outer * axisSize * innerBlock + inner;
      let best = input.buffer[base];
      let bestIndex = 0;
      for (let candidate = 1; candidate < axisSize; candidate++) {
        const value = input.buffer[base + candidate * innerBlock];
        if (value > best) {
          best = value;
          bestIndex = candidate;
        }
      }
      output.buffer[outputIndex++] = bestIndex;
    }
  }
}
