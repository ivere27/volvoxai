import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
  sliceShapeKernelQuantization,
} from './shapeKernelValidation.js';

function orderedOutputKeys(node): readonly string[] {
  const keys = Object.keys(node.outputs || {});
  if (keys.length === 0) throw new Error('Split requires at least one output tensor.');
  if (keys.every((name) => /^out(0|[1-9][0-9]*)$/.test(name))) {
    const ordered = [...keys].sort((left, right) => Number(left.slice(3)) - Number(right.slice(3)));
    if (ordered.some((name, index) => name !== `out${index}`)) {
      throw new Error("Split canonical 'outN' ports must be contiguous.");
    }
    return ordered;
  }
  // Static legacy Graphs used arbitrary insertion-ordered output names. The
  // logical v1 path always uses contiguous outN ports.
  return keys;
}

function splitSizes(params, axisExtent, outputCount) {
  if (params.split !== undefined && params.num_outputs !== undefined) {
    throw new Error('Split must specify split or num_outputs, not both.');
  }
  if (params.split !== undefined) {
    if (!Array.isArray(params.split) || params.split.length !== outputCount ||
        params.split.some((size) => !Number.isSafeInteger(size) || size <= 0)) {
      throw new Error('Split params.split must contain one positive size per output.');
    }
    return [...params.split];
  }
  const count = params.num_outputs ?? outputCount;
  if (!Number.isSafeInteger(count) || count <= 0 || count !== outputCount) {
    throw new Error('Split num_outputs must equal the concrete output count.');
  }
  if (axisExtent % count !== 0) {
    throw new Error(`Split num_outputs must divide axis extent ${axisExtent}.`);
  }
  return new Array(count).fill(axisExtent / count);
}

export function _cpuSplit(node) {
  const input = node.inputs?.input || node.inputs?.x || node.inputs?.data;
  assertShapeKernelTensor(input, 'Split input', { minimumRank: 1 });
  const params = assertShapeKernelParams(node, ['axis', 'split', 'num_outputs'], 'Split');
  const axis = normalizeShapeKernelAxis(params.axis, input.shape.length, 0, 'Split');
  const outputKeys = orderedOutputKeys(node);
  const sizes = splitSizes(params, input.shape[axis], outputKeys.length);
  const sum = sizes.reduce((total, size) => total + size, 0);
  if (!Number.isSafeInteger(sum) || sum !== input.shape[axis]) {
    throw new Error(`Split sizes sum to ${sum}, expected ${input.shape[axis]}.`);
  }

  let quantizationOffset = 0;
  const descriptors = outputKeys.map((key, index) => {
    const expectedShape = [...input.shape];
    expectedShape[axis] = sizes[index];
    const expectedQuantization = input.quantization?.scheme === 'per_axis' &&
      input.quantization.axis === axis
      ? sliceShapeKernelQuantization(
        input.quantization,
        axis,
        quantizationOffset,
        quantizationOffset + sizes[index],
      )
      : input.quantization;
    quantizationOffset += sizes[index];
    const output = node.outputs[key];
    assertShapeKernelOutput(
      output,
      expectedShape,
      input.dtype,
      expectedQuantization,
      `Split ${key}`,
    );
    return { output, size: sizes[index] };
  });

  const outer = input.shape.slice(0, axis)
    .reduce((product, dimension) => product * dimension, 1);
  const inner = input.shape.slice(axis + 1)
    .reduce((product, dimension) => product * dimension, 1);
  let axisOffset = 0;
  for (const descriptor of descriptors) {
    const chunk = descriptor.size * inner;
    for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
      const inputOffset = (outerIndex * input.shape[axis] + axisOffset) * inner;
      descriptor.output.buffer.set(
        input.buffer.subarray(inputOffset, inputOffset + chunk),
        outerIndex * chunk,
      );
    }
    axisOffset += descriptor.size;
  }
}
