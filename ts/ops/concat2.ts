import type { TensorQuantization } from '../types.js';
import { sameQuantizationDescriptor } from './quantizedShape.js';
import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
} from './shapeKernelValidation.js';

function orderedInputs(node): readonly any[] {
  const entries = Object.entries(node.inputs || {}).filter(([, tensor]) => tensor);
  if (entries.length < 2) throw new Error('Concat requires at least two input tensors.');
  if (entries.every(([name]) => /^input(0|[1-9][0-9]*)$/.test(name))) {
    entries.sort(([left], [right]) => Number(left.slice(5)) - Number(right.slice(5)));
    if (entries.some(([name], index) => name !== `input${index}`)) {
      throw new Error("Concat canonical 'inputN' ports must be contiguous.");
    }
    return entries.map(([, tensor]) => tensor);
  }
  const preferred = new Map(
    ['input', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].map((name, index) => [name, index]),
  );
  entries.sort(([left], [right]) => {
    const leftOrder = preferred.get(left) ?? Number.POSITIVE_INFINITY;
    const rightOrder = preferred.get(right) ?? Number.POSITIVE_INFINITY;
    return leftOrder - rightOrder || left.localeCompare(right);
  });
  return entries.map(([, tensor]) => tensor);
}

function concatQuantization(inputs: readonly any[], axis: number): TensorQuantization | undefined {
  const first = inputs[0].quantization;
  if (inputs.some((input) => !sameQuantizationDescriptor(input.quantization, first))) {
    if (first?.scheme !== 'per_axis' || first.axis !== axis ||
        inputs.some((input) => input.quantization?.scheme !== 'per_axis' ||
          input.quantization.axis !== axis)) {
      throw new Error(
        'Concat inputs need identical quantization unless concatenating their common per-axis dimension.',
      );
    }
  }
  if (first == null) return undefined;
  if (first.scheme === 'per_tensor' || first.axis !== axis) return first;
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  for (const input of inputs) {
    if (input.quantization?.scheme !== 'per_axis' || input.quantization.axis !== axis) {
      throw new Error('Concat inputs have incompatible per-axis quantization metadata.');
    }
    scales.push(...input.quantization.scales);
    zeroPoints.push(...input.quantization.zero_points);
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze(scales),
    zero_points: Object.freeze(zeroPoints),
  });
}

export function _cpuConcat2(node) {
  const operation = node.opType || 'Concat';
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputs = orderedInputs(node);
  const first = inputs[0];
  assertShapeKernelTensor(first, `${operation} input0`, {
    minimumRank: 1, maximumRank: 8,
  });
  const rank = first.shape.length;
  for (let index = 1; index < inputs.length; index++) {
    assertShapeKernelTensor(inputs[index], `${operation} input${index}`, {
      dtypes: [first.dtype], minimumRank: rank, maximumRank: rank,
    });
  }
  const params = operation === 'Concat2'
    ? assertShapeKernelParams(node, ['axis', 'sigmoid'], operation)
    : assertShapeKernelParams(node, ['axis'], operation);
  const axis = normalizeShapeKernelAxis(params.axis, rank, 0, operation);
  const expectedShape = [...first.shape];
  expectedShape[axis] = 0;
  for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
    const input = inputs[inputIndex];
    for (let dimension = 0; dimension < rank; dimension++) {
      if (dimension !== axis && input.shape[dimension] !== first.shape[dimension]) {
        throw new Error(`${operation} input${inputIndex} shape is incompatible outside axis ${axis}.`);
      }
    }
    expectedShape[axis] += input.shape[axis];
    if (!Number.isSafeInteger(expectedShape[axis])) {
      throw new Error(`${operation} axis extent exceeds the safe integer range.`);
    }
  }
  const expectedQuantization = concatQuantization(inputs, axis);
  assertShapeKernelOutput(
    output,
    expectedShape,
    first.dtype,
    expectedQuantization,
    operation,
  );
  const sigmoid = params.sigmoid ?? false;
  if (sigmoid && first.dtype !== 'float32') {
    throw new Error(`${operation} cannot fuse sigmoid into ${first.dtype} storage.`);
  }

  const inner = expectedShape.slice(axis + 1)
    .reduce((count, dimension) => count * dimension, 1);
  const outer = expectedShape.slice(0, axis)
    .reduce((count, dimension) => count * dimension, 1);
  const outputAxis = expectedShape[axis];
  for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
    let axisOffset = 0;
    for (const input of inputs) {
      const inputAxis = input.shape[axis];
      const block = inputAxis * inner;
      const sourceStart = outerIndex * block;
      const destinationStart = (outerIndex * outputAxis + axisOffset) * inner;
      if (sigmoid) {
        for (let index = 0; index < block; index++) {
          output.buffer[destinationStart + index] =
            1 / (1 + Math.exp(-input.buffer[sourceStart + index]));
        }
      } else {
        output.buffer.set(
          input.buffer.subarray(sourceStart, sourceStart + block),
          destinationStart,
        );
      }
      axisOffset += inputAxis;
    }
  }
}
