type GradientMap = Map<string, Float32Array>;

interface AccumulatedLossMetric {
  name: string;
  logitsTensor: string;
  weight: number;
  normalizer: number;
  loss: number;
  correct: number;
  examples: number;
}

interface PendingLossMetric extends Omit<AccumulatedLossMetric, 'loss'> {
  lossSum: number;
}

interface AccumulationState {
  backend: string;
  signature: string;
  trainableNames: string[];
  accumulationSteps: number;
  aggregation: 'mean_by_examples' | 'sum';
  microbatches: number;
  examples: number;
  lossSum: number;
  correct: number;
  hasContribution: boolean;
  lossMetrics: PendingLossMetric[] | null;
  sums: GradientMap;
}

const ACCUMULATION_STATE: unique symbol = Symbol('volvoxai.gradientAccumulationState');

interface AccumulationGraph {
  _pendingGradientAccumulation?: boolean;
  [ACCUMULATION_STATE]?: AccumulationState;
}

export interface GradientAccumulationOptions {
  backend?: string;
  gradients?: GradientMap;
  examples?: number;
  loss?: number;
  correct?: number;
  trainableNames?: string[];
  accumulationSteps?: number;
  flush?: boolean;
  signature?: string;
  aggregation?: 'mean_by_examples' | 'sum';
  hasContribution?: boolean;
  metrics?: AccumulatedLossMetric[];
}

function accumulatedState(graph: object): AccumulationState | undefined {
  return (graph as AccumulationGraph)[ACCUMULATION_STATE];
}

function retainAccumulatedState(graph: object, state: AccumulationState): void {
  Object.defineProperty(graph, ACCUMULATION_STATE, {
    configurable: true,
    writable: true,
    enumerable: false,
    value: state,
  });
}

function releaseAccumulatedState(graph: object): void {
  delete (graph as AccumulationGraph)[ACCUMULATION_STATE];
}

function positiveInteger(name: string, value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive safe integer.`);
  }
  return value;
}

function sameNames(left: readonly string[], right: readonly string[]) {
  return left.length === right.length && left.every((name, index) => name === right[index]);
}

function copyScaledGradient(gradient: Float32Array, scale: number) {
  const out = new Float32Array(gradient.length);
  for (let index = 0; index < gradient.length; index++) out[index] = gradient[index] * scale;
  return out;
}

function accumulateLossMetrics(
  state: AccumulationState,
  metrics: AccumulatedLossMetric[],
  aggregation: 'mean_by_examples' | 'sum',
) {
  if (!Array.isArray(metrics)) throw new Error("Gradient accumulation loss metrics must be an array.");
  if (!state.lossMetrics) {
    state.lossMetrics = metrics.map((metric) => ({
      name: metric.name,
      logitsTensor: metric.logitsTensor,
      weight: metric.weight,
      normalizer: metric.normalizer,
      lossSum: 0,
      correct: 0,
      examples: 0,
    }));
  }
  if (state.lossMetrics.length !== metrics.length ||
      state.lossMetrics.some((metric, index) => metric.name !== metrics[index]?.name)) {
    throw new Error("Gradient accumulation loss metrics changed before the pending update was applied.");
  }
  for (let index = 0; index < metrics.length; index++) {
    const source = metrics[index];
    const destination = state.lossMetrics[index];
    destination.lossSum += aggregation === "sum" ? source.loss : source.loss * source.examples;
    destination.correct += source.correct;
    destination.examples += source.examples;
  }
  return state.lossMetrics.map((metric) => ({
    name: metric.name,
    logitsTensor: metric.logitsTensor,
    loss: aggregation === "sum" || metric.examples === 0
      ? metric.lossSum
      : metric.lossSum / metric.examples,
    correct: metric.correct,
    examples: metric.examples,
    normalizer: aggregation === "sum" ? metric.normalizer : metric.examples,
    weight: metric.weight,
  }));
}

/** Number of already accumulated microbatches for the graph. */
export function gradientAccumulationIndex(graph: object) {
  return accumulatedState(graph)?.microbatches ?? 0;
}

export function hasPendingGradientAccumulation(graph: object) {
  return accumulatedState(graph) != null;
}

export function getGradientAccumulationState(graph: object) {
  const state = accumulatedState(graph);
  return state ? Object.freeze({
    pending: true,
    microbatches: state.microbatches,
    accumulationSteps: state.accumulationSteps,
    examples: state.examples,
  }) : Object.freeze({ pending: false, microbatches: 0, accumulationSteps: 0, examples: 0 });
}

export function resetGradientAccumulation(graph: object) {
  releaseAccumulatedState(graph);
  if (graph && typeof graph === "object") (graph as AccumulationGraph)._pendingGradientAccumulation = false;
}

/**
 * Accumulate mean gradients using the number of active examples as the weight.
 * This preserves the same gradient as one combined padded batch even when the
 * microbatches contain different numbers of ignored labels.
 */
export function accumulateGradients(graph: object, {
  backend,
  gradients,
  examples,
  loss,
  correct,
  trainableNames,
  accumulationSteps = 1,
  flush = false,
  signature,
  aggregation = "mean_by_examples",
  hasContribution = (examples ?? 0) > 0,
  metrics = [],
}: GradientAccumulationOptions = {}) {
  positiveInteger("gradientAccumulationSteps", accumulationSteps);
  if (!(gradients instanceof Map) || typeof examples !== 'number' ||
      !Number.isSafeInteger(examples) || examples < 0) {
    throw new Error("Gradient accumulation requires a gradient map and non-negative example count.");
  }
  if (typeof hasContribution !== "boolean") {
    throw new Error("Gradient accumulation hasContribution must be boolean.");
  }
  if (!Array.isArray(trainableNames) || trainableNames.length === 0) {
    throw new Error("Gradient accumulation requires trainable tensor names.");
  }
  if (typeof backend !== "string" || !backend || typeof signature !== "string") {
    throw new Error("Gradient accumulation requires backend and signature identifiers.");
  }
  if (aggregation !== "mean_by_examples" && aggregation !== "sum") {
    throw new Error("Gradient accumulation aggregation must be 'mean_by_examples' or 'sum'.");
  }
  const lossValue = loss as number;
  const correctCount = correct as number;

  let state = accumulatedState(graph);
  if (!state && accumulationSteps === 1) {
    return {
      apply: hasContribution,
      complete: true,
      empty: !hasContribution,
      microbatches: 1,
      accumulationSteps: 1,
      examples,
      loss: lossValue,
      correct: correctCount,
      gradients,
      metrics,
    };
  }
  if (!state) {
    state = {
      backend,
      signature,
      trainableNames: [...trainableNames],
      accumulationSteps,
      aggregation,
      microbatches: 0,
      examples: 0,
      lossSum: 0,
      correct: 0,
      hasContribution: false,
      lossMetrics: null,
      sums: new Map(),
    };
    retainAccumulatedState(graph, state);
    Object.defineProperty(graph, "_pendingGradientAccumulation", {
      configurable: true,
      writable: true,
      enumerable: false,
      value: true,
    });
  } else if (state.backend !== backend || state.signature !== signature ||
      state.accumulationSteps !== accumulationSteps || state.aggregation !== aggregation ||
      !sameNames(state.trainableNames, trainableNames)) {
    throw new Error("Gradient accumulation options or trainable tensors changed before the pending update was applied; reset the accumulation first.");
  }

  for (const name of trainableNames) {
    const gradient = gradients.get(name);
    if (!(gradient instanceof Float32Array)) {
      throw new Error(`Gradient accumulation is missing F32 gradient '${name}'.`);
    }
    let sum = state.sums.get(name);
    if (!sum) {
      sum = copyScaledGradient(gradient, aggregation === "sum" ? 1 : examples);
      state.sums.set(name, sum);
    } else {
      if (sum.length !== gradient.length) {
        throw new Error(`Gradient accumulation shape changed for '${name}'.`);
      }
      const scale = aggregation === "sum" ? 1 : examples;
      for (let index = 0; index < sum.length; index++) sum[index] += gradient[index] * scale;
    }
  }
  state.microbatches++;
  state.examples += examples;
  state.lossSum += aggregation === "sum" ? lossValue : lossValue * examples;
  state.correct += correctCount;
  state.hasContribution ||= hasContribution;
  const accumulatedMetrics = accumulateLossMetrics(state, metrics, aggregation);

  const apply = flush || state.microbatches >= accumulationSteps;
  if (!apply) {
    return {
      apply: false,
      complete: false,
      empty: !state.hasContribution,
      microbatches: state.microbatches,
      accumulationSteps,
      examples: state.examples,
      loss: aggregation === "sum" || state.examples === 0 ? state.lossSum : state.lossSum / state.examples,
      correct: state.correct,
      gradients,
      metrics: accumulatedMetrics,
    };
  }

  const averaged: GradientMap = new Map();
  for (const name of trainableNames) {
    const sum = state.sums.get(name)!;
    const gradient = new Float32Array(sum.length);
    const divisor = aggregation === "sum" ? 1 : state.examples;
    for (let index = 0; index < sum.length; index++) gradient[index] = sum[index] / divisor;
    averaged.set(name, gradient);
  }
  const result = {
    apply: state.hasContribution,
    complete: true,
    empty: !state.hasContribution,
    microbatches: state.microbatches,
    accumulationSteps,
    examples: state.examples,
    loss: aggregation === "sum" || state.examples === 0 ? state.lossSum : state.lossSum / state.examples,
    correct: state.correct,
    gradients: averaged,
    metrics: accumulatedMetrics,
  };
  releaseAccumulatedState(graph);
  (graph as AccumulationGraph)._pendingGradientAccumulation = false;
  return result;
}

/** Clip one combined norm over every trainable gradient. */
export function clipGradientsGlobal(gradients: GradientMap, maxGradNorm = 0) {
  if (typeof maxGradNorm !== "number" || !Number.isFinite(maxGradNorm) || maxGradNorm < 0) {
    throw new Error("maxGradNorm must be finite and non-negative.");
  }
  let sumSquares = 0;
  for (const [name, gradient] of gradients) {
    if (!(gradient instanceof Float32Array)) throw new Error(`Gradient '${name}' must be F32.`);
    for (let index = 0; index < gradient.length; index++) {
      const value = gradient[index];
      if (!Number.isFinite(value)) throw new Error(`Gradient for '${name}' is non-finite at index ${index}.`);
      sumSquares += value * value;
    }
  }
  const norm = Math.sqrt(sumSquares);
  const scale = maxGradNorm > 0 && norm > maxGradNorm
    ? maxGradNorm / (norm + 1e-12)
    : 1;
  if (scale !== 1) {
    for (const gradient of gradients.values()) {
      for (let index = 0; index < gradient.length; index++) gradient[index] *= scale;
    }
  }
  return { norm, scale };
}
