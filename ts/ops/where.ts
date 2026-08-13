const MAX_WHERE_RANK = 8;
const MAX_U32 = 0xffffffff;

function elementCount(shape) {
  if (!Array.isArray(shape) || shape.length > MAX_WHERE_RANK || shape.some((dimension) =>
    !Number.isSafeInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32
    ? elements
    : null;
}

function tensorElements(tensor) {
  const elements = elementCount(tensor?.shape);
  if (elements == null) return null;
  if (Number.isSafeInteger(tensor?.sizeBytes) && tensor.sizeBytes !== elements * 4) return null;
  if (tensor?.buffer?.length !== elements) return null;
  return elements;
}

function physicalStorageMatches(tensor) {
  return tensor?.dtype === 'float32'
    ? tensor.buffer instanceof Float32Array
    : tensor?.dtype === 'int32' && tensor.buffer instanceof Int32Array;
}

function contiguousStrides(shape) {
  const result = new Array(shape.length);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    result[axis] = stride;
    stride *= shape[axis];
  }
  return result;
}

function broadcastDescriptor(tensor, outputShape) {
  const paddedShape = [
    ...new Array(outputShape.length - tensor.shape.length).fill(1),
    ...tensor.shape,
  ];
  const contiguous = contiguousStrides(paddedShape);
  return paddedShape.map((dimension, axis) =>
    dimension === 1 && outputShape[axis] !== 1 ? 0 : contiguous[axis]);
}

export function whereDescriptor(node) {
  const condition = node.inputs?.cond || node.inputs?.condition || node.inputs?.mask;
  const a = node.inputs?.x || node.inputs?.a || node.inputs?.input;
  const b = node.inputs?.y || node.inputs?.b;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const conditionElements = tensorElements(condition);
  const aElements = tensorElements(a);
  const bElements = tensorElements(b);
  const outputElements = tensorElements(output);
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const rank = Math.max(
    condition?.shape?.length ?? -1,
    a?.shape?.length ?? -1,
    b?.shape?.length ?? -1,
  );
  const isMask = node.opType === 'Mask';
  const maskParams = node.params ?? {};
  if (isMask && (inputNames.length !== 3 || inputNames[0] !== 'a' ||
      inputNames[1] !== 'b' || inputNames[2] !== 'mask' ||
      outputNames.length !== 1 || outputNames[0] !== 'out' ||
      typeof maskParams !== 'object' || Array.isArray(maskParams) ||
      Reflect.ownKeys(maskParams).length !== 0)) {
    throw new Error('Mask requires exactly { mask, a, b } -> { out } and no parameters.');
  }
  if (!condition || !a || !b || !output ||
      (condition.dtype !== 'float32' && condition.dtype !== 'int32') ||
      (a.dtype !== 'float32' && a.dtype !== 'int32') ||
      b.dtype !== a.dtype || output.dtype !== a.dtype ||
      conditionElements == null || aElements == null || bElements == null ||
      outputElements == null || rank < 0 || rank > MAX_WHERE_RANK ||
      !physicalStorageMatches(condition) || !physicalStorageMatches(a) ||
      !physicalStorageMatches(b) || !physicalStorageMatches(output)) {
    throw new Error(
      `${node.opType || 'Where'} requires rank-0..8 broadcast-compatible same-dtype F32/I32 data/output and an F32 or I32 condition with matching physical storage.`,
    );
  }
  if (isMask && (condition.quantization != null || a.quantization != null ||
      b.quantization != null || output.quantization != null)) {
    throw new Error('Mask does not accept quantization metadata.');
  }

  const inputs = [condition, a, b];
  if (isMask && inputs.some((tensor) =>
    tensor.shape.length !== output.shape.length ||
    tensor.shape.some((dimension, axis) => dimension !== output.shape[axis]))) {
    throw new Error(
      `Mask input shapes [${condition.shape}], [${a.shape}], and [${b.shape}] must exactly match output [${output.shape}].`,
    );
  }
  if (!isMask && output.shape.length !== rank) {
    throw new Error(
      `Where input shapes [${condition.shape}], [${a.shape}], and [${b.shape}] do not broadcast output [${output.shape}].`,
    );
  }
  const paddedShapes = inputs.map((tensor) => [
    ...new Array(rank - tensor.shape.length).fill(1),
    ...tensor.shape,
  ]);
  const expectedShape = new Array(rank);
  for (let axis = 0; axis < rank; axis++) {
    const nonSingleton = paddedShapes
      .map((shape) => shape[axis])
      .filter((dimension) => dimension !== 1);
    const expected = nonSingleton[0] ?? 1;
    if (nonSingleton.some((dimension) => dimension !== expected) ||
        output.shape[axis] !== expected) {
      throw new Error(
        `${node.opType || 'Where'} input shapes [${condition.shape}], [${a.shape}], and [${b.shape}] do not broadcast output [${output.shape}].`,
      );
    }
    expectedShape[axis] = expected;
  }
  if (outputElements !== elementCount(expectedShape)) {
    throw new Error(`${node.opType || 'Where'} output storage does not match its broadcast shape.`);
  }

  return {
    condition,
    a,
    b,
    output,
    rank,
    elements: outputElements,
    outputStrides: contiguousStrides(expectedShape),
    conditionStrides: broadcastDescriptor(condition, expectedShape),
    aStrides: broadcastDescriptor(a, expectedShape),
    bStrides: broadcastDescriptor(b, expectedShape),
  };
}

export function _cpuWhere(node) {
  const descriptor = whereDescriptor(node);
  for (let index = 0; index < descriptor.elements; index++) {
    let remaining = index;
    let conditionIndex = 0;
    let aIndex = 0;
    let bIndex = 0;
    for (let axis = 0; axis < descriptor.rank; axis++) {
      const coordinate = Math.floor(remaining / descriptor.outputStrides[axis]);
      remaining %= descriptor.outputStrides[axis];
      conditionIndex += coordinate * descriptor.conditionStrides[axis];
      aIndex += coordinate * descriptor.aStrides[axis];
      bIndex += coordinate * descriptor.bStrides[axis];
    }
    // Integer conditions must remain integers: e.g. Int32.MIN_VALUE is true,
    // rather than being accidentally interpreted as the F32 bit pattern -0.
    descriptor.output.buffer[index] = descriptor.condition.buffer[conditionIndex] !== 0
      ? descriptor.a.buffer[aIndex]
      : descriptor.b.buffer[bIndex];
  }
}
