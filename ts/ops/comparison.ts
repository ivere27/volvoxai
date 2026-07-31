const MAX_COMPARISON_RANK = 8;
const MAX_U32 = 0xffffffff;

function elementCount(shape) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32
    ? elements
    : null;
}

function tensorElements(tensor) {
  const elements = elementCount(tensor?.shape);
  if (elements == null) return null;
  if (Number.isSafeInteger(tensor?.sizeBytes) && tensor.sizeBytes !== elements * 4) return null;
  if (tensor?.buffer && tensor.buffer.length !== elements) return null;
  return elements;
}

function contiguousStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    strides[axis] = stride;
    stride *= shape[axis];
  }
  return strides;
}

export function comparisonDescriptor(node) {
  const a = node.inputs?.a;
  const b = node.inputs?.b;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const operation = node.opType === 'Equal' ? 0 :
    node.opType === 'GreaterOrEqual' ? 1 : -1;
  const aRank = a?.shape?.length;
  const bRank = b?.shape?.length;
  const outputRank = output?.shape?.length;
  const aElements = tensorElements(a);
  const bElements = tensorElements(b);
  const outputElements = tensorElements(output);
  if (operation < 0 || inputNames.length !== 2 || inputNames[0] !== 'a' ||
      inputNames[1] !== 'b' || outputNames.length !== 1 || !a || !b || !output ||
      a.dtype !== 'int32' || b.dtype !== 'int32' || output.dtype !== 'int32' ||
      aElements == null || bElements == null || outputElements == null ||
      !Number.isInteger(aRank) || !Number.isInteger(bRank) ||
      !Number.isInteger(outputRank) || aRank < 0 || bRank < 0 ||
      aRank > MAX_COMPARISON_RANK || bRank > MAX_COMPARISON_RANK ||
      outputRank !== Math.max(aRank, bRank) || node.params == null ||
      typeof node.params !== 'object' || Array.isArray(node.params) ||
      Object.keys(node.params).length !== 0) {
    throw new Error(
      `${node.opType} node ${node.id ?? '<unnamed>'} requires exactly I32 { a, b } -> { out } with rank-0..8 ONNX broadcasting and no parameters.`,
    );
  }

  const paddedA = [...new Array(outputRank - aRank).fill(1), ...a.shape];
  const paddedB = [...new Array(outputRank - bRank).fill(1), ...b.shape];
  const expectedShape = new Array(outputRank);
  for (let axis = 0; axis < outputRank; axis++) {
    const left = paddedA[axis];
    const right = paddedB[axis];
    if (left !== right && left !== 1 && right !== 1) {
      throw new Error(`${node.opType} node ${node.id ?? '<unnamed>'} has incompatible broadcast shapes.`);
    }
    expectedShape[axis] = Math.max(left, right);
  }
  if (output.shape.length !== expectedShape.length ||
      output.shape.some((dimension, axis) => dimension !== expectedShape[axis])) {
    throw new Error(
      `${node.opType} node ${node.id ?? '<unnamed>'} output shape must be [${expectedShape}].`,
    );
  }

  const outputStrides = contiguousStrides(expectedShape);
  const aContiguous = contiguousStrides(paddedA);
  const bContiguous = contiguousStrides(paddedB);
  const aStrides = paddedA.map((dimension, axis) =>
    dimension === 1 && expectedShape[axis] !== 1 ? 0 : aContiguous[axis]);
  const bStrides = paddedB.map((dimension, axis) =>
    dimension === 1 && expectedShape[axis] !== 1 ? 0 : bContiguous[axis]);
  if (outputElements !== elementCount(expectedShape)) {
    throw new Error(`${node.opType} node ${node.id ?? '<unnamed>'} has invalid output storage.`);
  }
  return {
    a, b, output, rank: outputRank, elements: outputElements, operation,
    outputShape: expectedShape, outputStrides, aStrides, bStrides,
  };
}

export function _cpuComparison(node) {
  const descriptor = comparisonDescriptor(node);
  if (!(descriptor.a.buffer instanceof Int32Array) ||
      !(descriptor.b.buffer instanceof Int32Array) ||
      !(descriptor.output.buffer instanceof Int32Array)) {
    throw new Error(`${node.opType} node ${node.id ?? '<unnamed>'} requires physical I32 storage.`);
  }
  for (let index = 0; index < descriptor.elements; index++) {
    let remaining = index;
    let aIndex = 0;
    let bIndex = 0;
    for (let axis = 0; axis < descriptor.rank; axis++) {
      const coordinate = Math.floor(remaining / descriptor.outputStrides[axis]);
      remaining %= descriptor.outputStrides[axis];
      aIndex += coordinate * descriptor.aStrides[axis];
      bIndex += coordinate * descriptor.bStrides[axis];
    }
    descriptor.output.buffer[index] = descriptor.operation === 0
      ? Number(descriptor.a.buffer[aIndex] === descriptor.b.buffer[bIndex])
      : Number(descriptor.a.buffer[aIndex] >= descriptor.b.buffer[bIndex]);
  }
}

export function logicalNotDescriptor(node) {
  const input = node.inputs?.input;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputNames = Object.keys(node.inputs || {});
  const outputNames = Object.keys(node.outputs || {});
  const elements = tensorElements(input);
  if (inputNames.length !== 1 || inputNames[0] !== 'input' ||
      outputNames.length !== 1 || !input || !output ||
      input.dtype !== 'int32' || output.dtype !== 'int32' ||
      elements == null || tensorElements(output) !== elements ||
      input.shape.length !== output.shape.length ||
      input.shape.some((dimension, axis) => dimension !== output.shape[axis]) ||
      node.params == null || typeof node.params !== 'object' ||
      Array.isArray(node.params) || Object.keys(node.params).length !== 0) {
    throw new Error(
      `Not node ${node.id ?? '<unnamed>'} requires exactly same-shape I32 { input } -> { out } and no parameters.`,
    );
  }
  return { input, output, elements };
}

export function _cpuNot(node) {
  const descriptor = logicalNotDescriptor(node);
  if (!(descriptor.input.buffer instanceof Int32Array) ||
      !(descriptor.output.buffer instanceof Int32Array)) {
    throw new Error(`Not node ${node.id ?? '<unnamed>'} requires physical I32 storage.`);
  }
  for (let index = 0; index < descriptor.elements; index++) {
    descriptor.output.buffer[index] = Number(descriptor.input.buffer[index] === 0);
  }
}
