import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  sliceShapeKernelQuantization,
} from './shapeKernelValidation.js';

function safeIntegerArray(value, label, { positive = false } = {}) {
  if (!Array.isArray(value) || value.length === 0 ||
      value.some((item) => !Number.isSafeInteger(item) || (positive && item <= 0))) {
    throw new Error(`${label} must be a non-empty array of ${positive ? 'positive ' : ''}safe integers.`);
  }
  return value;
}

function sliceAxes(value, rank) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('Slice axes must be a non-empty array.');
  }
  const axes = value.map((raw, index) => {
    if (!Number.isInteger(raw)) throw new Error(`Slice axis ${index} must be an integer.`);
    const axis = raw < 0 ? raw + rank : raw;
    if (axis < 0 || axis >= rank) throw new Error(`Slice axis ${index} is outside rank ${rank}.`);
    return axis;
  });
  if (new Set(axes).size !== axes.length) throw new Error('Slice axes must be unique.');
  return axes;
}

function canonicalPositiveSlice(node, input, output) {
  assertShapeKernelTensor(input, 'Slice input', { minimumRank: 1, maximumRank: 8 });
  const params = assertShapeKernelParams(node, ['starts', 'ends', 'axes', 'steps'], 'Slice');
  const startsInput = safeIntegerArray(params.starts, 'Slice starts');
  const endsInput = safeIntegerArray(params.ends, 'Slice ends');
  if (startsInput.length !== endsInput.length) {
    throw new Error('Slice starts and ends must have equal lengths.');
  }
  const rawAxes = params.axes ?? startsInput.map((_value, index) => index);
  const axes = sliceAxes(rawAxes, input.shape.length);
  const stepsInput = params.steps === undefined
    ? startsInput.map(() => 1)
    : safeIntegerArray(params.steps, 'Slice steps', { positive: true });
  if (axes.length !== startsInput.length || stepsInput.length !== startsInput.length) {
    throw new Error('Slice starts, ends, axes, and steps must have equal lengths.');
  }

  const starts = new Array(input.shape.length).fill(0);
  const ends = [...input.shape];
  const steps = new Array(input.shape.length).fill(1);
  const expectedShape = [...input.shape];
  for (let index = 0; index < axes.length; index++) {
    const axis = axes[index];
    const extent = input.shape[axis];
    const rawStart = startsInput[index];
    const rawEnd = endsInput[index];
    const start = Math.min(extent, Math.max(0, rawStart < 0 ? rawStart + extent : rawStart));
    const end = Math.min(extent, Math.max(0, rawEnd < 0 ? rawEnd + extent : rawEnd));
    const step = stepsInput[index];
    if (end <= start) throw new Error(`Slice axis ${axis} selects an empty extent.`);
    starts[axis] = start;
    ends[axis] = end;
    steps[axis] = step;
    expectedShape[axis] = Math.floor((end - start - 1) / step) + 1;
  }
  const expectedQuantization = input.quantization?.scheme === 'per_axis'
    ? sliceShapeKernelQuantization(
      input.quantization,
      input.quantization.axis,
      starts[input.quantization.axis],
      ends[input.quantization.axis],
      steps[input.quantization.axis],
    )
    : input.quantization;
  const outputElements = assertShapeKernelOutput(
    output,
    expectedShape,
    input.dtype,
    expectedQuantization,
    'Slice',
  );

  const inputStrides = new Array(input.shape.length);
  let stride = 1;
  for (let axis = input.shape.length - 1; axis >= 0; axis--) {
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  return { starts, steps, outputElements, inputStrides };
}

export function _cpuSlice(node) {
  const input = node.inputs?.input || node.inputs?.data || node.inputs?.x;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const descriptor = canonicalPositiveSlice(node, input, output);
  for (let outputIndex = 0; outputIndex < descriptor.outputElements; outputIndex++) {
    let remaining = outputIndex;
    let inputIndex = 0;
    for (let axis = output.shape.length - 1; axis >= 0; axis--) {
      const coordinate = remaining % output.shape[axis];
      remaining = Math.floor(remaining / output.shape[axis]);
      inputIndex += (descriptor.starts[axis] + coordinate * descriptor.steps[axis]) *
        descriptor.inputStrides[axis];
    }
    output.buffer[outputIndex] = input.buffer[inputIndex];
  }
}
