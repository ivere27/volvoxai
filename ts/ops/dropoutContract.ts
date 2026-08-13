import {
  assertShapeKernelOutput,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';

export function dropoutUint32(name, value) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`Dropout ${name} must be an unsigned 32-bit integer.`);
  }
  return value >>> 0;
}

export function dropoutProbability(node) {
  const params = node?.params ?? {};
  const names = ['ratio', 'p', 'probability'].filter((name) => params[name] !== undefined);
  if (names.length > 1) {
    throw new Error(`Dropout node ${node?.id ?? '<unnamed>'} must specify at most one probability field.`);
  }
  const probability = names.length === 0 ? 0.5 : params[names[0]];
  if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0 || probability >= 1) {
    throw new Error(`Dropout node ${node?.id ?? '<unnamed>'} probability must be finite and in [0, 1).`);
  }
  return probability;
}

/** Validate the inference identity contract without importing training RNG code. */
export function dropoutDescriptor(node) {
  const input = node.inputs?.input || node.inputs?.x;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  assertShapeKernelTensor(input, `Dropout node ${node.id ?? '<unnamed>'} input`, {
    dtypes: ['float32'], maximumRank: 8,
  });
  assertShapeKernelOutput(
    output, input.shape, 'float32', undefined,
    `Dropout node ${node.id ?? '<unnamed>'}`,
  );
  dropoutProbability(node);
  dropoutUint32('node seed', node?.params?.seed ?? 0);
  return Object.freeze({ input, output });
}
