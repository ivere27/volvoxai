import { SHAPE_SYMBOL_PATTERN } from './shapeSystem.js';
import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
  remapShapeKernelQuantization,
} from './shapeKernelValidation.js';

function concreteTarget(node, output) {
  const params = assertShapeKernelParams(node, ['shape'], 'Expand');
  // Static legacy graphs use their already-concrete output descriptor. The v1
  // logical path always supplies params.shape and is checked below.
  const target = params.shape ?? output?.shape;
  if (!Array.isArray(target) || !output || target.length !== output.shape.length) {
    throw new Error('Expand params.shape must match the concrete output rank.');
  }
  const symbols = new Map();
  for (let axis = 0; axis < target.length; axis++) {
    const dimension = target[axis];
    if (typeof dimension === 'number') {
      if (!Number.isSafeInteger(dimension) || dimension <= 0 || dimension !== output.shape[axis]) {
        throw new Error(`Expand params.shape[${axis}] does not match the concrete output.`);
      }
    } else if (typeof dimension === 'string' && SHAPE_SYMBOL_PATTERN.test(dimension)) {
      const previous = symbols.get(dimension);
      if (previous !== undefined && previous !== output.shape[axis]) {
        throw new Error(`Expand symbol '${dimension}' has conflicting concrete values.`);
      }
      symbols.set(dimension, output.shape[axis]);
    } else {
      throw new Error(`Expand params.shape[${axis}] must be a positive integer or symbol.`);
    }
  }
  return output.shape;
}

export function _cpuExpand(node) {
  const input = node.inputs?.input || node.inputs?.x || node.inputs?.data;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  assertShapeKernelTensor(input, 'Expand input', { maximumRank: 8 });
  const target = concreteTarget(node, output);
  if (target.length < input.shape.length || target.length > 8) {
    throw new Error('Expand output rank must be between input rank and 8.');
  }
  const offset = target.length - input.shape.length;
  for (let axis = 0; axis < target.length; axis++) {
    const inputDimension = axis < offset ? 1 : input.shape[axis - offset];
    if (inputDimension !== 1 && inputDimension !== target[axis]) {
      throw new Error(`Expand input shape [${input.shape}] cannot broadcast to [${target}].`);
    }
  }
  let expectedQuantization = input.quantization;
  if (input.quantization?.scheme === 'per_axis') {
    const outputAxis = input.quantization.axis + offset;
    if (input.shape[input.quantization.axis] !== target[outputAxis]) {
      throw new Error('Expand must not expand the per-axis quantization extent.');
    }
    expectedQuantization = remapShapeKernelQuantization(input.quantization, outputAxis);
  }
  const outputElements = assertShapeKernelOutput(
    output,
    target,
    input.dtype,
    expectedQuantization,
    'Expand',
  );

  const inputStrides = new Array(input.shape.length);
  let stride = 1;
  for (let axis = input.shape.length - 1; axis >= 0; axis--) {
    inputStrides[axis] = stride;
    stride *= input.shape[axis];
  }
  for (let outputIndex = 0; outputIndex < outputElements; outputIndex++) {
    let remaining = outputIndex;
    let inputIndex = 0;
    for (let axis = target.length - 1; axis >= 0; axis--) {
      const coordinate = remaining % target[axis];
      remaining = Math.floor(remaining / target[axis]);
      if (axis >= offset && input.shape[axis - offset] !== 1) {
        inputIndex += coordinate * inputStrides[axis - offset];
      }
    }
    output.buffer[outputIndex] = input.buffer[inputIndex];
  }
}
