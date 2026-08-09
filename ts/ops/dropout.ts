import {
  dropoutDescriptor,
  dropoutProbability,
  dropoutUint32,
} from './dropoutContract.js';

export { dropoutDescriptor, dropoutProbability } from './dropoutContract.js';

export function dropoutThreshold(probability) {
  if (probability <= 0) return 0;
  return Math.min(0xffffffff, Math.floor(probability * 0x100000000)) >>> 0;
}

export function dropoutHash(index, seed, counter, stream) {
  let value = (index >>> 0) ^ (seed >>> 0) ^
    Math.imul(counter >>> 0, 0x9e3779b9) ^ Math.imul((stream + 1) >>> 0, 0x9e3779b9);
  value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb352d);
  value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}

/** Stable UTF-16-independent hash used to bind RNG streams to a concrete shape. */
export function dropoutShapeSignatureHash(signature = '') {
  if (typeof signature !== 'string') throw new Error('Dropout shapeSignature must be a string.');
  const bytes = new TextEncoder().encode(signature);
  let hash = 0x811c9dc5;
  for (const value of bytes) {
    hash ^= value;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return signature.length === 0 ? 0 : hash >>> 0;
}

export function dropoutEffectiveSeed(node, context, stream = 0) {
  return (dropoutUint32("seed", context?.seed ?? 0) ^ dropoutUint32("node seed", node?.params?.seed ?? 0) ^
    dropoutUint32("shape hash", context?.shapeHash ?? 0) ^
    Math.imul((stream + 1) >>> 0, 0x9e3779b9)) >>> 0;
}

export function dropoutMultiplier(node, context, index, stream = 0) {
  const probability = dropoutProbability(node);
  if (!context || probability === 0) return 1;
  const counter = dropoutUint32("counter", context.counter ?? 0);
  return dropoutHash(index, dropoutEffectiveSeed(node, context, stream), counter, -1) >= dropoutThreshold(probability)
    ? 1 / (1 - probability)
    : 0;
}

export function dropoutContext({ seed = 0, counter = 0, shapeSignature = '' } = {}) {
  return Object.freeze({
    seed: dropoutUint32("seed", seed),
    counter: dropoutUint32("counter", counter),
    shapeSignature,
    shapeHash: dropoutShapeSignatureHash(shapeSignature),
  });
}

/** Inference is an identity. A context is supplied only by opt-in autograd. */
export function _cpuDropout(node, execution: {
  training?: { dropout?: any };
  nodeIndex?: number;
} = {}) {
  const { input, output } = dropoutDescriptor(node);
  const context = execution.training?.dropout || null;
  const stream = execution.nodeIndex ?? 0;
  if (!context || dropoutProbability(node) === 0) {
    // Preserve Dropout as a true inference alias: no allocation, copy, mask, or
    // loop. A later opt-in training call detaches the output below.
    output.buffer = input.buffer;
    return;
  }
  if (output.buffer === input.buffer) output.buffer = new Float32Array(input.buffer.length);
  for (let index = 0; index < input.buffer.length; index++) {
    output.buffer[index] = input.buffer[index] * dropoutMultiplier(node, context, index, stream);
  }
}
