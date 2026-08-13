import { SHAPE_SYMBOL_PATTERN } from './shapeSystem.js';
import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  normalizeShapeKernelAxes,
  reshapeShapeKernelQuantization,
  sameShape,
} from './shapeKernelValidation.js';

function product(shape) {
  return shape.reduce((value, dimension) => value * dimension, 1);
}

function reshapeTarget(params, output, operation) {
  // Legacy concrete Graphs did not carry an explicit target. Dynamic v1 does;
  // retaining the output descriptor fallback does not widen the logical v1
  // contract because graph-wide shape preflight has already required it.
  const target = params.shape ?? output.shape;
  if (!Array.isArray(target) || target.length !== output.shape.length) {
    throw new Error(`${operation} params.shape must match the concrete output rank.`);
  }
  const symbols = new Map();
  for (let axis = 0; axis < target.length; axis++) {
    const dimension = target[axis];
    if (typeof dimension === 'number') {
      if (!Number.isSafeInteger(dimension) || dimension <= 0 || dimension !== output.shape[axis]) {
        throw new Error(`${operation} params.shape[${axis}] does not match the concrete output.`);
      }
      continue;
    }
    if (typeof dimension !== 'string' || !SHAPE_SYMBOL_PATTERN.test(dimension)) {
      throw new Error(`${operation} params.shape[${axis}] must be a positive integer or symbol.`);
    }
    const previous = symbols.get(dimension);
    if (previous !== undefined && previous !== output.shape[axis]) {
      throw new Error(`${operation} symbol '${dimension}' has conflicting concrete values.`);
    }
    symbols.set(dimension, output.shape[axis]);
  }
  return output.shape;
}

function structuralOutputShape(node, input, output, operation) {
  if (operation === 'Identity') {
    assertShapeKernelParams(node, [], operation);
    return input.shape;
  }
  if (operation === 'Reshape') {
    const params = assertShapeKernelParams(node, ['shape'], operation);
    return reshapeTarget(params, output, operation);
  }
  if (operation === 'Flatten') {
    const params = assertShapeKernelParams(node, ['axis'], operation);
    const rawAxisValue = params.axis ?? 1;
    if (!Number.isInteger(rawAxisValue)) throw new Error('Flatten axis must be an integer.');
    const rawAxis = rawAxisValue as number;
    const axis = rawAxis < 0 ? rawAxis + input.shape.length : rawAxis;
    if (axis < 0 || axis > input.shape.length) {
      throw new Error(`Flatten axis must resolve to a boundary in [0, ${input.shape.length}].`);
    }
    return [product(input.shape.slice(0, axis)), product(input.shape.slice(axis))];
  }
  if (operation === 'Squeeze') {
    const params = assertShapeKernelParams(node, ['axes'], operation);
    const axes = normalizeShapeKernelAxes(params.axes, input.shape.length, operation);
    for (const axis of axes) {
      if (input.shape[axis] !== 1) throw new Error(`Squeeze input axis ${axis} must be 1.`);
    }
    const removed = new Set(axes);
    return input.shape.filter((_dimension, axis) => !removed.has(axis));
  }
  if (operation === 'Unsqueeze') {
    const params = assertShapeKernelParams(node, ['axes'], operation);
    if (!Array.isArray(params.axes) || params.axes.length === 0) {
      throw new Error('Unsqueeze axes must be a non-empty array.');
    }
    const outputRank = input.shape.length + params.axes.length;
    const axes = normalizeShapeKernelAxes(params.axes, outputRank, operation);
    const inserted = new Set(axes);
    const shape: number[] = [];
    let inputAxis = 0;
    for (let outputAxis = 0; outputAxis < outputRank; outputAxis++) {
      shape.push(inserted.has(outputAxis) ? 1 : input.shape[inputAxis++]);
    }
    return shape;
  }
  throw new Error(`Unsupported structural operation '${operation}'.`);
}

export function _cpuReshape(node) {
  const operation = node.opType || 'Reshape';
  const input = node.inputs?.input || node.inputs?.x || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = assertShapeKernelTensor(input, `${operation} input`);
  if (!output) throw new Error(`${operation} requires one output tensor.`);
  const expectedShape = structuralOutputShape(node, input, output, operation);
  if (inputElements !== product(expectedShape)) {
    throw new Error(`${operation} must preserve the exact element count.`);
  }
  const expectedQuantization = operation === 'Identity'
    ? input.quantization
    : reshapeShapeKernelQuantization(
      input.quantization,
      input.shape,
      expectedShape,
      operation,
    );
  assertShapeKernelOutput(
    output,
    expectedShape,
    input.dtype,
    expectedQuantization,
    operation,
  );
  if (!sameShape(output.shape, expectedShape)) {
    throw new Error(`${operation} output shape is incompatible.`);
  }
  output.buffer.set(input.buffer);
}
