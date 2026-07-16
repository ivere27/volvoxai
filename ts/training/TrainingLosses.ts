import type { TensorLike } from '../types.js';

type LossArray = ArrayLike<number | boolean> | ArrayBufferView;
type LossArraySource = LossArray | { buffer?: LossArray };

export interface CrossEntropyLossInput {
  name?: string;
  logitsTensor?: string;
  logits?: string;
  targets?: LossArraySource;
  ignoreIndex?: number | null;
  lossMask?: LossArraySource | null;
  lastToken?: number | null;
  weight?: number;
  normalizer?: number | null;
}

export interface CrossEntropyTrainingOptions extends CrossEntropyLossInput {
  losses?: CrossEntropyLossInput[] | null;
}

export interface CrossEntropyLossDescriptor {
  readonly name: string;
  readonly logitsTensor: string;
  readonly targets: readonly number[];
  readonly ignoreIndex: number | null;
  readonly lastToken: number | null;
  readonly lossMask: readonly (number | boolean)[] | null;
  readonly weight: number;
  readonly normalizer: number | null;
}

function arrayValues(value: LossArraySource | undefined, label: string): Array<number | boolean> {
  const storage = Array.isArray(value) || ArrayBuffer.isView(value)
    ? value
    : (value as { buffer?: LossArray } | undefined)?.buffer;
  if (!storage || (!Array.isArray(storage) && !ArrayBuffer.isView(storage)) || storage instanceof DataView) {
    throw new Error(`${label} must be an array or typed array.`);
  }
  return Array.from(storage as ArrayLike<number | boolean>);
}

function finiteNonNegative(value: unknown, label: string, fallback: number): number;
function finiteNonNegative(value: unknown, label: string, fallback: null): number | null;
function finiteNonNegative(value: unknown, label: string, fallback: number | null): number | null {
  const resolved = value == null ? fallback : value;
  if (typeof resolved !== "number" || !Number.isFinite(resolved) || resolved < 0) {
    throw new Error(`${label} must be finite and non-negative.`);
  }
  return resolved;
}

/** Normalize the legacy single-CE fields and the generic weighted loss list. */
export function normalizeCrossEntropyLosses(
  options: CrossEntropyTrainingOptions = {},
  defaultLogitsTensor: string | null = null,
): Readonly<CrossEntropyLossDescriptor>[] {
  const explicit = options.losses;
  if (explicit != null && !Array.isArray(explicit)) {
    throw new Error("trainStep losses must be an array.");
  }
  if (Array.isArray(explicit) && explicit.length === 0) {
    throw new Error("trainStep losses must not be empty.");
  }
  if (explicit?.length && options.targets != null) {
    throw new Error("trainStep accepts either losses or the legacy targets fields, not both.");
  }
  const source = explicit != null
    ? explicit
    : [{
      logitsTensor: options.logitsTensor ?? defaultLogitsTensor,
      targets: options.targets,
      ignoreIndex: options.ignoreIndex,
      lossMask: options.lossMask,
      lastToken: options.lastToken,
    }];
  if (source.length === 0) throw new Error("trainStep losses must not be empty.");

  const names = new Set();
  return source.map((entry, index) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error(`trainStep loss ${index} must be an object.`);
    }
    const name = entry.name ?? `loss_${index}`;
    if (typeof name !== "string" || !name || names.has(name)) {
      throw new Error("trainStep loss names must be unique non-empty strings.");
    }
    names.add(name);
    const logitsTensor = entry.logitsTensor ?? entry.logits ?? defaultLogitsTensor;
    if (typeof logitsTensor !== "string" || !logitsTensor) {
      throw new Error(`trainStep loss '${name}' requires logitsTensor.`);
    }
    const targets = arrayValues(entry.targets, `trainStep loss '${name}' targets`) as number[];
    if (targets.length === 0) throw new Error(`trainStep loss '${name}' targets must not be empty.`);
    const ignoreIndex = entry.ignoreIndex ?? null;
    if (ignoreIndex != null && !Number.isInteger(ignoreIndex)) {
      throw new Error(`trainStep loss '${name}' ignoreIndex must be an integer or null.`);
    }
    const lastToken = entry.lastToken ?? null;
    if (lastToken != null && (!Number.isSafeInteger(lastToken) || lastToken < 0)) {
      throw new Error(`trainStep loss '${name}' lastToken must be a non-negative safe integer or null.`);
    }
    let lossMask: Array<number | boolean> | null = null;
    if (entry.lossMask != null) {
      lossMask = arrayValues(entry.lossMask, `trainStep loss '${name}' lossMask`);
      if (lossMask.length !== targets.length) {
        throw new Error(`trainStep loss '${name}' lossMask length must match targets length.`);
      }
      if (lossMask.some((value) => value !== true && value !== false && value !== 0 && value !== 1)) {
        throw new Error(`trainStep loss '${name}' lossMask values must be boolean or 0/1.`);
      }
    }
    const weight = finiteNonNegative(entry.weight, `trainStep loss '${name}' weight`, 1);
    const normalizer = entry.normalizer == null
      ? null
      : finiteNonNegative(entry.normalizer, `trainStep loss '${name}' normalizer`, null);
    if (normalizer === 0) throw new Error(`trainStep loss '${name}' normalizer must be positive.`);
    return Object.freeze({
      name,
      logitsTensor,
      targets,
      ignoreIndex,
      lastToken,
      lossMask,
      weight,
      normalizer,
    });
  });
}

/** Build a weighted CE gradient for one logits tensor without mutating graph state. */
export function crossEntropyGradient(
  logits: TensorLike,
  logitsValues: Float32Array,
  descriptor: CrossEntropyLossDescriptor,
) {
  if (!logits || logits.dtype !== "float32" || !Array.isArray(logits.shape) || logits.shape.length === 0) {
    throw new Error(`Logits tensor '${descriptor.logitsTensor}' must be F32.`);
  }
  if (!(logitsValues instanceof Float32Array) || logitsValues.length === 0) {
    throw new Error(`Logits tensor '${descriptor.logitsTensor}' requires non-empty Float32 storage.`);
  }
  const classes = logits.shape.at(-1);
  if (typeof classes !== 'number') {
    throw new Error(`Logits tensor '${descriptor.logitsTensor}' has no class dimension.`);
  }
  const rows = logitsValues.length / classes;
  if (!Number.isSafeInteger(classes) || classes <= 0 || !Number.isSafeInteger(rows) || rows <= 0) {
    throw new Error(`Logits tensor '${descriptor.logitsTensor}' has invalid class dimensions.`);
  }
  const candidateCount = descriptor.targets.length === rows ? rows : descriptor.targets.length;
  if (candidateCount <= 0 || candidateCount > rows) {
    throw new Error(`Target count for loss '${descriptor.name}' does not match logits rows.`);
  }
  const batch = logits.shape.length >= 3 ? logits.shape[0] : 1;
  const rowsPerBatch = Number.isSafeInteger(batch) && batch > 0 && rows % batch === 0
    ? rows / batch
    : 0;
  const perBatchTargets = batch > 1 && descriptor.targets.length === batch && rowsPerBatch > 0;
  let row0 = descriptor.targets.length === rows ? 0 : rows - candidateCount;
  let position = rowsPerBatch - 1;
  if (descriptor.lastToken != null && (descriptor.targets.length === 1 || perBatchTargets)) {
    const limit = perBatchTargets ? rowsPerBatch : rows;
    if (descriptor.lastToken >= limit) {
      throw new Error(
        `lastToken ${descriptor.lastToken} is out of range for ${limit} logits rows` +
        `${perBatchTargets ? " per batch" : ""}.`,
      );
    }
    if (perBatchTargets) position = descriptor.lastToken;
    else row0 = descriptor.lastToken;
  }

  const examples: Array<{ row: number; target: number }> = [];
  for (let index = 0; index < candidateCount; index++) {
    if (descriptor.lossMask && !descriptor.lossMask[index]) continue;
    const target = descriptor.targets[index];
    if (descriptor.ignoreIndex != null && target === descriptor.ignoreIndex) continue;
    if (!Number.isInteger(target) || target < 0 || target >= classes) {
      throw new Error(`Target id ${target} for loss '${descriptor.name}' is out of range for ${classes} classes.`);
    }
    examples.push({
      row: perBatchTargets ? index * rowsPerBatch + position : row0 + index,
      target,
    });
  }
  const activeCount = examples.length;
  const normalizer = descriptor.normalizer ?? activeCount;
  if (activeCount > 0 && normalizer <= 0) {
    throw new Error(`Loss '${descriptor.name}' requires a positive normalizer.`);
  }
  const gradient = new Float32Array(logitsValues.length);
  if (activeCount === 0 || descriptor.weight === 0) {
    return {
      gradient,
      loss: 0,
      unweightedLossSum: 0,
      correct: 0,
      examples: activeCount,
      normalizer: descriptor.normalizer ?? 0,
    };
  }

  let lossSum = 0;
  let correct = 0;
  const gradientScale = descriptor.weight / normalizer;
  for (const { row, target } of examples) {
    const offset = row * classes;
    let maximum = logitsValues[offset];
    let argmax = 0;
    for (let col = 1; col < classes; col++) {
      const value = logitsValues[offset + col];
      if (value > maximum) {
        maximum = value;
        argmax = col;
      }
    }
    let denominator = 0;
    for (let col = 0; col < classes; col++) {
      denominator += Math.exp(logitsValues[offset + col] - maximum);
    }
    lossSum += maximum + Math.log(denominator) - logitsValues[offset + target];
    if (argmax === target) correct++;
    for (let col = 0; col < classes; col++) {
      const probability = Math.exp(logitsValues[offset + col] - maximum) / denominator;
      gradient[offset + col] += (probability - (col === target ? 1 : 0)) * gradientScale;
    }
  }
  return {
    gradient,
    loss: descriptor.weight * lossSum / normalizer,
    unweightedLossSum: lossSum,
    correct,
    examples: activeCount,
    normalizer,
  };
}

export function addGradient(destination: Float32Array, source: Float32Array): Float32Array {
  if (!(destination instanceof Float32Array) || !(source instanceof Float32Array) ||
      destination.length !== source.length) {
    throw new Error("Cannot add incompatible gradients.");
  }
  for (let index = 0; index < destination.length; index++) destination[index] += source[index];
  return destination;
}
