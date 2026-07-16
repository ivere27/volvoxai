function uint32(name, value) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Dropout ${name} must be a non-negative safe integer.`);
  }
  return value >>> 0;
}

export function dropoutProbability(node) {
  const probability = node?.params?.ratio ?? node?.params?.p ?? node?.params?.probability ?? 0.5;
  if (typeof probability !== "number" || !Number.isFinite(probability) || probability < 0 || probability >= 1) {
    throw new Error(`Dropout node ${node?.id ?? "<unnamed>"} probability must be finite and in [0, 1).`);
  }
  return probability;
}

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

export function dropoutEffectiveSeed(node, context, stream = 0) {
  return (uint32("seed", context?.seed ?? 0) ^ uint32("node seed", node?.params?.seed ?? 0) ^
    Math.imul((stream + 1) >>> 0, 0x9e3779b9)) >>> 0;
}

export function dropoutMultiplier(node, context, index, stream = 0) {
  const probability = dropoutProbability(node);
  if (!context || probability === 0) return 1;
  const counter = uint32("counter", context.counter ?? 0);
  return dropoutHash(index, dropoutEffectiveSeed(node, context, stream), counter, -1) >= dropoutThreshold(probability)
    ? 1 / (1 - probability)
    : 0;
}

export function dropoutContext({ seed = 0, counter = 0 } = {}) {
  return Object.freeze({ seed: uint32("seed", seed), counter: uint32("counter", counter) });
}

/** Inference is an identity. A context is supplied only by opt-in autograd. */
export function _cpuDropout(node, execution: {
  training?: { dropout?: any };
  nodeIndex?: number;
} = {}) {
  const input = node.inputs.input || node.inputs.x;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (input?.dtype !== "float32" || output?.dtype !== "float32" ||
      !(input.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array) ||
      input.buffer.length !== output.buffer.length) {
    throw new Error(`Dropout node ${node.id ?? "<unnamed>"} requires equal-size F32 input/output tensors.`);
  }
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
