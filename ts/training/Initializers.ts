import type { RuntimeDType } from '../types.js';

export interface InitializerSpec {
  type?: string;
  kind?: string;
  name?: string;
  seed?: string | number;
  mean?: number;
  stddev?: number;
  std?: number;
  gain?: number;
}

export type TensorInitializer = string | InitializerSpec;

function elementCount(shape: readonly number[]) {
  if (!Array.isArray(shape) || shape.length === 0) throw new Error("Initializer shape must be non-empty.");
  let count = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim <= 0) throw new Error("Initializer shape dimensions must be positive integers.");
    count *= dim;
    if (!Number.isSafeInteger(count)) throw new Error("Initializer tensor is too large.");
  }
  return count;
}

function seedValue(seed: unknown) {
  if (typeof seed === "number") {
    if (!Number.isFinite(seed)) throw new Error("Initializer seed must be finite.");
    return Math.trunc(seed) >>> 0;
  }
  const text = String(seed ?? 0);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** Deterministic Mulberry32 generator used by every built-in initializer. */
export function createSeededRandom(seed: unknown = 0): () => number {
  let state = seedValue(seed);
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normalizedSpec(spec: TensorInitializer): InitializerSpec & { type: string } {
  const options = typeof spec === "string" ? { type: spec } : { ...(spec || {}) };
  const type = String(options.type || options.kind || options.name || "zeros")
    .toLowerCase()
    .replace(/[-_\s]/g, "");
  return { ...options, type };
}

function fanInOut(shape: readonly number[]): [number, number] {
  if (shape.length === 1) return [shape[0], shape[0]];
  let receptiveField = 1;
  for (let index = 0; index < shape.length - 2; index++) receptiveField *= shape[index];
  return [shape[shape.length - 2] * receptiveField, shape[shape.length - 1] * receptiveField];
}

function fillNormal(
  output: Float32Array,
  random: () => number,
  mean: number,
  stddev: number,
) {
  for (let index = 0; index < output.length; index += 2) {
    const u1 = Math.max(random(), Number.EPSILON);
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    output[index] = mean + stddev * radius * Math.cos(angle);
    if (index + 1 < output.length) output[index + 1] = mean + stddev * radius * Math.sin(angle);
  }
}

/**
 * Allocate initialized runtime storage for a trainable graph weight.
 * Supported types: zeros, ones, normal, xavier/xavierUniform, xavierNormal.
 */
export function initializeTensor(
  shape: readonly number[],
  dtype: RuntimeDType = "float32",
  initializer: TensorInitializer = "zeros",
): Float32Array {
  if (dtype !== "float32") {
    throw new Error(`Built-in initializers require float32 weights, got '${dtype}'.`);
  }
  const output = new Float32Array(elementCount(shape));
  const options = normalizedSpec(initializer);
  if (options.type === "zeros" || options.type === "zero") return output;
  if (options.type === "ones" || options.type === "one") {
    output.fill(1);
    return output;
  }

  const random = createSeededRandom(options.seed ?? 0);
  if (options.type === "normal" || options.type === "gaussian") {
    const mean = options.mean ?? 0;
    const stddev = options.stddev ?? options.std ?? 0.02;
    if (!Number.isFinite(mean) || !Number.isFinite(stddev) || stddev < 0) {
      throw new Error("Normal initializer mean/stddev must be finite and stddev must be non-negative.");
    }
    fillNormal(output, random, mean, stddev);
    return output;
  }

  const [fanIn, fanOut] = fanInOut(shape);
  const gain = options.gain ?? 1;
  if (!Number.isFinite(gain) || gain < 0) throw new Error("Xavier initializer gain must be finite and non-negative.");
  if (options.type === "xavier" || options.type === "xavieruniform" || options.type === "glorotuniform") {
    const limit = gain * Math.sqrt(6 / (fanIn + fanOut));
    for (let index = 0; index < output.length; index++) output[index] = (random() * 2 - 1) * limit;
    return output;
  }
  if (options.type === "xaviernormal" || options.type === "glorotnormal") {
    fillNormal(output, random, 0, gain * Math.sqrt(2 / (fanIn + fanOut)));
    return output;
  }
  throw new Error(`Unsupported initializer '${options.type}'.`);
}
