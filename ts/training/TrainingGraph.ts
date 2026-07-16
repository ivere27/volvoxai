import { Graph } from '../core/Graph.js';
import type { Tensor } from '../core/Tensor.js';
import type { GraphInspection, GraphInspectionOptions } from '../types.js';
import type {
  TrainingOptimizerDescriptor,
  TrainingOptimizerOptions,
} from './TrainingOptimizer.js';

interface OptimizerMoments {
  m: Float32Array;
  v: Float32Array;
  step: number;
}

export interface TrainingGraphState {
  optimizerState: Map<string, OptimizerMoments> | null;
  optimizerDescriptor: TrainingOptimizerDescriptor | null;
  trainingStep: number;
  trainingMetadata: unknown;
}

export type StatefulTrainingGraph = Graph & TrainingGraphState;

export interface TrainingTensorUpdateOptions extends TrainingOptimizerOptions {
  mode?: 'assign' | 'add' | 'sgd' | 'adamw';
}

type TrainingTensorUpdate = Float32Array | ArrayBuffer | ArrayBufferView | ArrayLike<number>;

const optimizerTensorReferences = new WeakMap<Graph, Map<string, Tensor>>();

function defineState(graph: object, name: keyof TrainingGraphState, value: unknown) {
  if (name in graph) return;
  Object.defineProperty(graph, name, {
    configurable: true,
    enumerable: true,
    writable: true,
    value,
  });
}

function floatStorage(value: TrainingTensorUpdate | undefined): Float32Array {
  if (value == null) return new Float32Array();
  if (value instanceof Float32Array) return value;
  if (ArrayBuffer.isView(value)) {
    return new Float32Array(value.buffer, value.byteOffset, value.byteLength / Float32Array.BYTES_PER_ELEMENT);
  }
  return new Float32Array(value);
}

function tensorLength(tensor: Tensor | undefined): number {
  try {
    return floatStorage(tensor?.buffer).length;
  } catch {
    return -1;
  }
}

/**
 * Add full-profile state and methods to a core graph supplied directly to a
 * trainer. The inference Graph class remains free of these fields and methods.
 */
export function ensureTrainingGraphState<TGraph extends Graph>(graph: TGraph): TGraph & TrainingGraphState {
  if (!graph || (typeof graph !== 'object' && typeof graph !== 'function')) {
    throw new Error('Training requires a graph object.');
  }
  defineState(graph, 'optimizerState', null);
  defineState(graph, 'optimizerDescriptor', null);
  defineState(graph, 'trainingStep', 0);
  defineState(graph, 'trainingMetadata', null);

  if (graph instanceof Graph && graph.applyTensorUpdate === Graph.prototype.applyTensorUpdate) {
    Object.defineProperty(graph, 'applyTensorUpdate', {
      configurable: true,
      value(name: string, update: TrainingTensorUpdate, options: TrainingTensorUpdateOptions = {}) {
        const mode = options.mode || 'sgd';
        if (mode === 'assign' || mode === 'add') {
          return Graph.prototype.applyTensorUpdate.call(this, name, update, options);
        }
        return applyTrainingTensorUpdate(this, name, update, { ...options, mode });
      },
    });
  }
  if (graph instanceof Graph && graph.inspect === Graph.prototype.inspect) {
    Object.defineProperty(graph, 'inspect', {
      configurable: true,
      value(options: GraphInspectionOptions = {}) {
        const trainingGraph = this as StatefulTrainingGraph;
        return { ...Graph.prototype.inspect.call(this, options), trainingStep: trainingGraph.trainingStep };
      },
    });
  }
  return graph as TGraph & TrainingGraphState;
}

/** Drop stale optimizer moments after structural tensor edits. */
export function synchronizeTrainingGraphState(graph: Graph): StatefulTrainingGraph {
  const trainingGraph = ensureTrainingGraphState(graph);
  if (!(trainingGraph.optimizerState instanceof Map)) return trainingGraph;
  let references = optimizerTensorReferences.get(trainingGraph);
  if (!references) {
    references = new Map();
    optimizerTensorReferences.set(trainingGraph, references);
  }
  for (const [name, state] of trainingGraph.optimizerState) {
    const tensor = trainingGraph.getTensor?.(name);
    const previous = references.get(name);
    const valid = tensor?.isWeight && tensor.dtype === 'float32' &&
      state?.m instanceof Float32Array && state?.v instanceof Float32Array &&
      state.m.length === tensorLength(tensor) && state.v.length === tensorLength(tensor) &&
      (!previous || previous === tensor);
    if (!valid) {
      trainingGraph.optimizerState.delete(name);
      references.delete(name);
    } else {
      references.set(name, tensor);
    }
  }
  return trainingGraph;
}

/** Apply an SGD or AdamW parameter update owned entirely by the full profile. */
export function applyTrainingTensorUpdate(
  graph: Graph,
  name: string,
  update: TrainingTensorUpdate,
  options: TrainingTensorUpdateOptions = {},
): Tensor {
  const trainingGraph = synchronizeTrainingGraphState(graph);
  const tensor = trainingGraph.getTensor?.(name);
  if (!tensor) throw new Error(`Tensor '${name}' not found.`);
  if (tensor.dtype !== 'float32') {
    throw new Error(`Tensor '${name}' update requires float32, got '${tensor.dtype}'.`);
  }
  if (!tensor.buffer) throw new Error(`Tensor '${name}' has no CPU buffer to update.`);
  const target = floatStorage(tensor.buffer);
  const source = floatStorage(update);
  if (source.length !== target.length) {
    throw new Error(`Tensor '${name}' update length mismatch: got ${source.length}, want ${target.length}.`);
  }

  const mode = options.mode || 'sgd';
  if (mode !== 'sgd' && mode !== 'adamw') {
    throw new Error(`Unsupported training tensor update mode '${mode}'.`);
  }
  const learningRate = options.learningRate ?? options.lr ?? 1e-3;
  const weightDecay = options.weightDecay ?? 0;
  const maxGradNorm = options.maxGradNorm ?? 0;
  let scale = 1;
  if (maxGradNorm > 0) {
    let sumSquares = 0;
    for (let index = 0; index < source.length; index++) sumSquares += source[index] * source[index];
    const norm = Math.sqrt(sumSquares);
    if (norm > maxGradNorm) scale = maxGradNorm / (norm + 1e-12);
  }

  const next = new Float32Array(target);
  let nextState: OptimizerMoments | null = null;
  if (mode === 'sgd') {
    for (let index = 0; index < next.length; index++) {
      let gradient = source[index] * scale;
      if (weightDecay) gradient += weightDecay * next[index];
      next[index] -= learningRate * gradient;
    }
  } else {
    const existing = trainingGraph.optimizerState?.get(name);
    const firstMoment = existing?.m?.length === next.length
      ? new Float32Array(existing.m) : new Float32Array(next.length);
    const secondMoment = existing?.v?.length === next.length
      ? new Float32Array(existing.v) : new Float32Array(next.length);
    const step = options.step ?? ((existing?.step ?? 0) + 1);
    const beta1 = options.beta1 ?? 0.9;
    const beta2 = options.beta2 ?? 0.999;
    const epsilon = options.epsilon ?? 1e-8;
    const correction1 = 1 - Math.pow(beta1, step);
    const correction2 = 1 - Math.pow(beta2, step);
    for (let index = 0; index < next.length; index++) {
      const gradient = source[index] * scale;
      firstMoment[index] = beta1 * firstMoment[index] + (1 - beta1) * gradient;
      secondMoment[index] = beta2 * secondMoment[index] + (1 - beta2) * gradient * gradient;
      if (weightDecay) next[index] -= learningRate * weightDecay * next[index];
      next[index] -= learningRate * (firstMoment[index] / correction1) /
        (Math.sqrt(secondMoment[index] / correction2) + epsilon);
    }
    nextState = { m: firstMoment, v: secondMoment, step };
  }

  const result = Graph.prototype.applyTensorUpdate.call(trainingGraph, name, next, { mode: 'assign' });
  if (nextState) {
    trainingGraph.optimizerState ||= new Map();
    trainingGraph.optimizerState.set(name, nextState);
    let references = optimizerTensorReferences.get(trainingGraph);
    if (!references) {
      references = new Map();
      optimizerTensorReferences.set(trainingGraph, references);
    }
    references.set(name, tensor);
  }
  return result;
}

/** Graph variant exported by the full profile. */
export class TrainingGraph extends Graph {
  declare optimizerState: Map<string, OptimizerMoments> | null;
  declare optimizerDescriptor: TrainingOptimizerDescriptor | null;
  declare trainingStep: number;
  declare trainingMetadata: unknown;

  constructor() {
    super();
    ensureTrainingGraphState(this);
  }

  applyTensorUpdate(
    name: string,
    update: TrainingTensorUpdate,
    options: TrainingTensorUpdateOptions = {},
  ): Tensor {
    const mode = options.mode || 'sgd';
    if (mode === 'assign' || mode === 'add') return super.applyTensorUpdate(name, update, options);
    return applyTrainingTensorUpdate(this, name, update, { ...options, mode });
  }

  inspect(options: GraphInspectionOptions = {}): GraphInspection & { trainingStep: number } {
    return { ...super.inspect(options), trainingStep: this.trainingStep };
  }
}
