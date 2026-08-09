import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';

function normalizedPads(value, rank) {
  if (!Array.isArray(value) ||
      value.some((amount) => !Number.isSafeInteger(amount) || amount < 0)) {
    throw new Error('Pad pads must be non-negative safe integers.');
  }
  if (value.length === rank * 2) {
    return { before: value.slice(0, rank), after: value.slice(rank) };
  }
  // Legacy NHWC static graphs used [top, left, bottom, right]. Bounded-shape
  // v1 always supplies the rank-general 2*rank form above.
  if (rank === 4 && value.length === 4) {
    return {
      before: [0, value[0], value[1], 0],
      after: [0, value[2], value[3], 0],
    };
  }
  throw new Error(`Pad pads must contain exactly ${rank * 2} entries.`);
}

export function _cpuPad(node) {
  const input = node.inputs?.input || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = assertShapeKernelTensor(input, 'Pad input', { minimumRank: 1 });
  const params = assertShapeKernelParams(node, ['pads', 'value'], 'Pad');
  const pads = normalizedPads(params.pads, input.shape.length);
  const value = params.value ?? 0;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error('Pad value must be finite.');
  }
  const expectedShape = input.shape.map((dimension, axis) => {
    const result = dimension + pads.before[axis] + pads.after[axis];
    if (!Number.isSafeInteger(result) || result <= 0) {
      throw new Error(`Pad axis ${axis} exceeds the safe integer range.`);
    }
    return result;
  });
  if (input.quantization?.scheme === 'per_axis' &&
      (pads.before[input.quantization.axis] !== 0 || pads.after[input.quantization.axis] !== 0)) {
    throw new Error('Pad must not extend a per-axis quantization dimension.');
  }
  assertShapeKernelOutput(
    output,
    expectedShape,
    input.dtype,
    input.quantization,
    'Pad',
  );

  output.buffer.fill(value);
  const inputStrides = new Array(input.shape.length);
  const outputStrides = new Array(output.shape.length);
  let inputStride = 1;
  let outputStride = 1;
  for (let axis = input.shape.length - 1; axis >= 0; axis--) {
    inputStrides[axis] = inputStride;
    outputStrides[axis] = outputStride;
    inputStride *= input.shape[axis];
    outputStride *= output.shape[axis];
  }
  for (let inputIndex = 0; inputIndex < inputElements; inputIndex++) {
    let remaining = inputIndex;
    let outputIndex = 0;
    for (let axis = 0; axis < input.shape.length; axis++) {
      const coordinate = Math.floor(remaining / inputStrides[axis]);
      remaining %= inputStrides[axis];
      outputIndex += (coordinate + pads.before[axis]) * outputStrides[axis];
    }
    output.buffer[outputIndex] = input.buffer[inputIndex];
  }
}
