import { assertRawQuantizedShapeTensors } from './quantizedShape.js';

const MAX_SLICE_RANK = 8;
const MAX_U32 = 0xffffffff;

function elementCount(shape) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = shape.reduce((count, dimension) => count * dimension, 1);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32 ? elements : null;
}

function canonicalPositiveSlice(node, input, output) {
  const rank = input?.shape?.length;
  const inputElements = elementCount(input?.shape);
  const outputElements = elementCount(output?.shape);
  const startsInput = node.params?.starts ?? [];
  const stepsInput = node.params?.steps ??
    (Array.isArray(startsInput) ? startsInput.map(() => 1) : null);
  const axesInput = node.params?.axes ??
    (Array.isArray(startsInput) ? startsInput.map((_, index) => index) : null);
  if (!input || !output || inputElements == null || outputElements == null ||
      !['float32', 'int32', 'int8', 'uint8'].includes(input.dtype) ||
      output.dtype !== input.dtype ||
      !Number.isInteger(rank) || rank < 1 || rank > MAX_SLICE_RANK ||
      output.shape.length !== rank || !Array.isArray(startsInput) ||
      !Array.isArray(stepsInput) || !Array.isArray(axesInput) ||
      startsInput.length !== stepsInput.length || startsInput.length !== axesInput.length) {
    throw new Error(`Slice node '${node.id}' requires rank 1..8 input/output tensors with matching F32/I32/I8/U8 dtype and parameter arrays.`);
  }
  // Slice selects a subset of elements without touching their values, so a
  // quantized input passes through with its descriptor unchanged.
  assertRawQuantizedShapeTensors(node, [input, output], 'Slice');
  if (!input.buffer || !output.buffer || input.buffer.length !== inputElements ||
      output.buffer.length !== outputElements) {
    throw new Error(`Slice node '${node.id}' tensor storage does not match its shape.`);
  }

  const starts = new Array(rank).fill(0);
  const steps = new Array(rank).fill(1);
  const seen = new Set();
  for (let index = 0; index < axesInput.length; index++) {
    let axis = axesInput[index];
    if (axis < 0) axis += rank;
    let start = startsInput[index];
    const step = stepsInput[index];
    if (!Number.isInteger(axis) || axis < 0 || axis >= rank || seen.has(axis) ||
        !Number.isSafeInteger(start) || !Number.isSafeInteger(step) ||
        step <= 0 || step > MAX_U32) {
      throw new Error(`Slice node '${node.id}' requires unique axes and positive integer steps.`);
    }
    if (start < 0) start += input.shape[axis];
    if (start < 0 || start >= input.shape[axis]) {
      throw new Error(`Slice node '${node.id}' start is outside its input axis.`);
    }
    starts[axis] = start;
    steps[axis] = step;
    seen.add(axis);
  }

  const inputStrides = new Array(rank);
  let stride = 1;
  for (let axis = rank - 1; axis >= 0; axis--) {
    const outputLength = output.shape[axis];
    const remaining = input.shape[axis] - 1 - starts[axis];
    if (outputLength - 1 > Math.floor(remaining / steps[axis])) {
      throw new Error(`Slice node '${node.id}' output shape exceeds its input selection.`);
    }
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  return { rank, inputStrides, outputElements, starts, steps };
}

// Canonical positive-step Slice for rank 1..8. This intentionally matches
// the full-WASM and WebGPU descriptor: axes and negative starts are normalized
// once, omitted axes are identity selections, and output shapes must be within
// the selected input extent.
export function _cpuSlice(node) {
  const input = node.inputs.input || node.inputs.data || node.inputs.x;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const descriptor = canonicalPositiveSlice(node, input, output);

  for (let outputIndex = 0; outputIndex < descriptor.outputElements; outputIndex++) {
    let remaining = outputIndex;
    let inputIndex = 0;
    for (let axis = descriptor.rank - 1; axis >= 0; axis--) {
      const coordinate = remaining % output.shape[axis];
      remaining = Math.floor(remaining / output.shape[axis]);
      inputIndex += (descriptor.starts[axis] + coordinate * descriptor.steps[axis]) *
        descriptor.inputStrides[axis];
    }
    output.buffer[outputIndex] = input.buffer[inputIndex];
  }
}
