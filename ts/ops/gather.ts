import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxis,
  remapShapeKernelQuantization,
} from './shapeKernelValidation.js';

function strides(shape) {
  const result = new Array(shape.length);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    result[axis] = stride;
    stride *= shape[axis];
  }
  return result;
}

export function _cpuGather(node) {
  const input = node.inputs?.input;
  const indices = node.inputs?.indices;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  assertShapeKernelTensor(input, 'Gather input', { minimumRank: 1 });
  assertShapeKernelTensor(indices, 'Gather indices', { dtypes: ['int32'] });
  if (indices.quantization != null) throw new Error('Gather indices must be unquantized I32.');
  const params = assertShapeKernelParams(node, ['axis'], 'Gather');
  const axis = normalizeShapeKernelAxis(params.axis, input.shape.length, 0, 'Gather');
  const expectedShape = [
    ...input.shape.slice(0, axis),
    ...indices.shape,
    ...input.shape.slice(axis + 1),
  ];
  let expectedQuantization = input.quantization;
  if (input.quantization?.scheme === 'per_axis') {
    if (input.quantization.axis === axis) {
      throw new Error('Gather cannot reorder a per-axis quantization dimension.');
    }
    const outputAxis = input.quantization.axis < axis
      ? input.quantization.axis
      : input.quantization.axis + indices.shape.length - 1;
    expectedQuantization = remapShapeKernelQuantization(input.quantization, outputAxis);
  }
  const outputElements = assertShapeKernelOutput(
    output,
    expectedShape,
    input.dtype,
    expectedQuantization,
    'Gather',
  );

  const axisExtent = input.shape[axis];
  // Selecting along a partially resident bank's slot axis keeps the model's
  // global slot ids, so map them to staged rows exactly as the MoE kernels do.
  const residentSlots: readonly number[] | null =
    axis === 0 && node.residentSlots ? node.residentSlots : null;
  const slotDomain = residentSlots === null ? axisExtent : node.residentSlotDomain;
  const slotToRow: Map<number, number> | null = residentSlots === null
    ? null
    : new Map(residentSlots.map((slot, row) => [slot, row] as [number, number]));
  if (residentSlots !== null && residentSlots.length !== axisExtent) {
    throw new Error(
      `Gather lists ${residentSlots.length} resident slots but the staged axis ` +
      `holds ${axisExtent}.`);
  }
  if (!Number.isSafeInteger(slotDomain) || slotDomain <= 0 ||
      (residentSlots !== null && residentSlots.some((slot, index) =>
        !Number.isSafeInteger(slot) || slot < 0 || slot >= slotDomain ||
        (index > 0 && slot <= residentSlots[index - 1])))) {
    throw new Error('Gather has invalid resident slot-domain metadata.');
  }
  for (let index = 0; index < indices.buffer.length; index++) {
    const raw = indices.buffer[index];
    const normalized = raw < 0 ? raw + slotDomain : raw;
    if (normalized < 0 || normalized >= slotDomain) {
      throw new Error(`Gather index ${raw} is outside axis extent ${slotDomain}.`);
    }
    if (slotToRow !== null) {
      if (!slotToRow.has(normalized)) {
        throw new Error(`Gather slot ${normalized} is not resident in this context.`);
      }
      continue;
    }
  }

  const inputStrides = strides(input.shape);
  const outputStrides = strides(expectedShape);
  const indexStrides = strides(indices.shape);
  for (let outputIndex = 0; outputIndex < outputElements; outputIndex++) {
    let remaining = outputIndex;
    const coordinates = new Array(expectedShape.length);
    for (let dimension = 0; dimension < expectedShape.length; dimension++) {
      coordinates[dimension] = Math.floor(remaining / outputStrides[dimension]);
      remaining %= outputStrides[dimension];
    }
    let indicesOffset = 0;
    for (let dimension = 0; dimension < indices.shape.length; dimension++) {
      indicesOffset += coordinates[axis + dimension] * indexStrides[dimension];
    }
    let gathered = indices.buffer[indicesOffset];
    if (gathered < 0) gathered += slotDomain;
    if (slotToRow !== null) gathered = slotToRow.get(gathered)!;
    let inputOffset = 0;
    for (let dimension = 0; dimension < axis; dimension++) {
      inputOffset += coordinates[dimension] * inputStrides[dimension];
    }
    inputOffset += gathered * inputStrides[axis];
    for (let dimension = axis + 1; dimension < input.shape.length; dimension++) {
      inputOffset += coordinates[dimension - 1 + indices.shape.length] * inputStrides[dimension];
    }
    output.buffer[outputIndex] = input.buffer[inputOffset];
  }
}
