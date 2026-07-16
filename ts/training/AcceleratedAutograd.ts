import type { Graph } from '../core/Graph.js';
import type { Tensor } from '../core/Tensor.js';
import type { RuntimeTypedArray } from '../types.js';
import { dropoutContext } from '../ops/dropout.js';
import {
  canonicalOptimizerDescriptor,
  normalizeTrainingUpdateMode,
  validateOptimizerOptions,
} from './TrainingOptimizer.js';
import {
  ensureTrainingGraphState,
  synchronizeTrainingGraphState,
  type StatefulTrainingGraph,
  type TrainingTensorUpdateOptions,
} from './TrainingGraph.js';
import {
  crossEntropyGradient,
  normalizeCrossEntropyLosses,
} from './TrainingLosses.js';
import {
  accumulateGradients,
  gradientAccumulationIndex,
  resetGradientAccumulation as resetPendingGradients,
} from './GradientAccumulation.js';
import type { TrainingStepOptions } from './TrainingStep.js';

type GradientMap = Map<string, Float32Array>;
type TrainingDropoutContext = ReturnType<typeof dropoutContext>;
type CrossEntropyGradientResult = ReturnType<typeof crossEntropyGradient>;

interface TrainingLossMetric {
  name: string;
  logitsTensor: string;
  loss: number;
  correct: number;
  examples: number;
  normalizer: number;
  weight: number;
}

/**
 * Required numerical hooks for a strict accelerated training backend.
 *
 * Unlike CPUAutograd's compatibility driver, this contract has no JavaScript
 * numerical fallbacks. A backend either implements the entire accepted graph
 * or rejects it during preflight.
 */
export interface AcceleratedTrainingExecution {
  readonly backend: string;
  preflight(graph: StatefulTrainingGraph, options: TrainingStepOptions): void | Promise<void>;
  forward(
    graph: StatefulTrainingGraph,
    inputs: Record<string, RuntimeTypedArray>,
    options: { dropout: TrainingDropoutContext },
  ): void | Promise<void>;
  crossEntropyGradient(
    logits: Tensor,
    values: Float32Array,
    descriptor: Parameters<typeof crossEntropyGradient>[2],
  ): CrossEntropyGradientResult;
  addGradient(destination: Float32Array, source: Float32Array): unknown;
  backwardNode(context: any): boolean | Promise<boolean>;
  allFinite(gradient: Float32Array): boolean;
  clipGradients(gradients: GradientMap, maxGradNorm: number): { norm: number; scale: number };
  applyTensorUpdate(
    graph: StatefulTrainingGraph,
    tensor: Tensor,
    gradient: Float32Array,
    options: TrainingTensorUpdateOptions,
  ): Tensor;
}

function firstOutput(node: any): Tensor | undefined {
  return node?.outputs?.out || Object.values(node?.outputs || {})[0] as Tensor | undefined;
}

function nodeInput(node: any): Tensor | undefined {
  return node?.inputs?.input || node?.inputs?.x || node?.inputs?.data ||
    Object.values(node?.inputs || {})[0] as Tensor | undefined;
}

function gradientFor(grads: GradientMap, tensor: Tensor | null | undefined): Float32Array | null {
  if (!tensor || tensor.dtype !== 'float32') return null;
  let gradient = grads.get(tensor.name);
  if (!gradient) {
    gradient = new Float32Array((tensor.buffer as Float32Array).length);
    grads.set(tensor.name, gradient);
  }
  return gradient;
}

function backwardFailure(node: any, detail: string): never {
  const id = node?.id != null ? ` '${node.id}'` : '';
  throw new Error(
    `${String(node?.opType || 'unknown')} backward for node${id} is unsupported: ${detail}.`,
  );
}

/** Run one all-accelerated train step without importing CPUAutograd/CPUEngine. */
export async function acceleratedTrainStep(
  inputGraph: Graph,
  options: TrainingStepOptions,
  execution: AcceleratedTrainingExecution,
) {
  const graph = ensureTrainingGraphState(inputGraph);
  const {
    inputs = {},
    targets,
    logitsTensor,
    losses = null,
    trainableTensors,
    optimizer = {},
    updateMode = 'adamw',
    lastToken = null,
    ignoreIndex = null,
    lossMask = null,
    dropout = {},
    gradientAccumulationSteps = 1,
    flushGradientAccumulation = false,
    resetGradientAccumulation = false,
  } = options;
  const backendName = execution?.backend;
  if (!graph) throw new Error('Accelerated training requires a graph.');
  if (typeof backendName !== 'string' || backendName.length === 0) {
    throw new Error('Accelerated training requires a named backend execution.');
  }
  if (!trainableTensors || trainableTensors.length === 0) {
    throw new Error('Accelerated training requires trainableTensors.');
  }
  if (new Set(trainableTensors).size !== trainableTensors.length) {
    throw new Error('Accelerated training trainableTensors must be unique.');
  }
  const mode = normalizeTrainingUpdateMode(updateMode);
  validateOptimizerOptions(optimizer, mode);
  const optimizerDescriptor = canonicalOptimizerDescriptor(mode, optimizer);
  const defaultLogitsName = logitsTensor || graph.outputNames?.[0] ||
    firstOutput(graph.nodes[graph.nodes.length - 1])?.name;
  const lossDescriptors = normalizeCrossEntropyLosses({
    targets,
    logitsTensor,
    losses,
    ignoreIndex,
    lossMask,
    lastToken,
  }, defaultLogitsName);
  if (!Number.isSafeInteger(gradientAccumulationSteps) || gradientAccumulationSteps <= 0) {
    throw new Error('Accelerated training gradientAccumulationSteps must be a positive safe integer.');
  }
  if (gradientAccumulationSteps > 1 && lossDescriptors.length > 1 &&
      lossDescriptors.some((loss) => loss.normalizer == null)) {
    throw new Error(
      'Multiple losses with gradient accumulation require an explicit full-window normalizer for every loss.',
    );
  }
  if (typeof flushGradientAccumulation !== 'boolean' ||
      typeof resetGradientAccumulation !== 'boolean') {
    throw new Error('Accelerated training accumulation flush/reset options must be boolean.');
  }

  await execution.preflight(graph, options);
  if (resetGradientAccumulation) resetPendingGradients(graph);
  const currentTrainingStep = graph.trainingStep ?? 0;
  const nextTrainingStep = optimizer.step ?? (currentTrainingStep + 1);
  if (!Number.isSafeInteger(currentTrainingStep) || currentTrainingStep < 0 ||
      !Number.isSafeInteger(nextTrainingStep) || nextTrainingStep <= currentTrainingStep) {
    throw new Error(
      'Accelerated training optimizer step must be a safe integer greater than graph.trainingStep.',
    );
  }
  const trainables = trainableTensors.map((name) => {
    const tensor = graph.getTensor(name);
    if (!tensor?.isWeight || tensor.dtype !== 'float32' ||
        !(tensor.buffer instanceof Float32Array)) {
      throw new Error(`Trainable tensor '${name}' must be an initialized F32 graph weight.`);
    }
    graph.adapters?._assertBaseMutationAllowed();
    return tensor;
  });
  const topologyRevision = graph.topologyRevision;
  const weightRevision = graph.weightRevision;
  const accumulationIndex = gradientAccumulationIndex(graph);
  if (!dropout || typeof dropout !== 'object' || Array.isArray(dropout)) {
    throw new Error('Accelerated training dropout options must be an object.');
  }
  const defaultDropoutCounter = Number(
    (BigInt(nextTrainingStep - 1) * BigInt(gradientAccumulationSteps) +
      BigInt(accumulationIndex + 1)) & 0xffffffffn,
  );
  const trainingDropout = dropoutContext({
    seed: dropout.seed ?? 0,
    counter: dropout.counter ?? defaultDropoutCounter,
  });

  await execution.forward(graph, inputs, { dropout: trainingDropout });
  graph.assertTopologyRevision?.(topologyRevision, `${backendName.toUpperCase()} training`);
  if (graph.weightRevision !== weightRevision) {
    throw new Error(`${backendName.toUpperCase()} training graph weights changed during the forward pass.`);
  }

  const grads: GradientMap = new Map();
  const lossMetrics: TrainingLossMetric[] = [];
  let loss = 0;
  let correct = 0;
  let count = 0;
  for (const descriptor of lossDescriptors) {
    const logits = graph.getTensor(descriptor.logitsTensor);
    if (!logits || logits.dtype !== 'float32' || !(logits.buffer instanceof Float32Array)) {
      throw new Error(`Logits tensor '${descriptor.logitsTensor}' must be initialized F32.`);
    }
    const metric = execution.crossEntropyGradient(logits, logits.buffer, descriptor);
    execution.addGradient(gradientFor(grads, logits)!, metric.gradient);
    loss += metric.loss;
    correct += metric.correct;
    count += metric.examples;
    lossMetrics.push({
      name: descriptor.name,
      logitsTensor: descriptor.logitsTensor,
      loss: metric.loss,
      correct: metric.correct,
      examples: metric.examples,
      normalizer: metric.normalizer,
      weight: descriptor.weight,
    });
  }
  const hasContribution = lossMetrics.some((metric) => metric.examples > 0 && metric.weight > 0);
  if (!hasContribution) {
    for (const tensor of trainables) gradientFor(grads, tensor);
  }

  if (hasContribution) {
    for (let nodeIndex = graph.nodes.length - 1; nodeIndex >= 0; nodeIndex--) {
      const node = graph.nodes[nodeIndex];
      const output = firstOutput(node);
      const liveOutputGradients = (Object.values(node.outputs || {}) as Tensor[])
        .filter((tensor) => tensor && grads.has(tensor.name));
      let outputGradient = output ? grads.get(output.name) : null;
      if (node.opType === 'MoERouter') {
        const gates = node.outputs.weights || node.outputs.expert_weights;
        outputGradient = gates ? grads.get(gates.name) : null;
      }
      if (node.opType === 'Split') {
        if (!liveOutputGradients.length) continue;
        const input = nodeInput(node);
        if (input?.dtype !== 'float32' || input.shape.length === 0) {
          backwardFailure(node, 'Split requires F32 input');
        }
        const handled = await execution.backwardNode({
          node,
          nodeIndex,
          output,
          outputGradient,
          outputGradients: new Map(liveOutputGradients.map((tensor) => [
            tensor.name,
            grads.get(tensor.name),
          ])),
          gradients: grads,
          gradientFor: (tensor: Tensor) => gradientFor(grads, tensor),
          trainingDropout,
        });
        if (!handled) {
          backwardFailure(node, `the operation is not supported by the ${backendName} training backend`);
        }
        continue;
      }
      if (!outputGradient) {
        if (liveOutputGradients.length) {
          backwardFailure(node, 'a non-primary output carries a gradient but has no backward rule');
        }
        continue;
      }
      const handled = await execution.backwardNode({
        node,
        nodeIndex,
        output,
        outputGradient,
        gradients: grads,
        gradientFor: (tensor: Tensor) => gradientFor(grads, tensor),
        trainingDropout,
      });
      if (!handled) {
        backwardFailure(node, `the operation is not supported by the ${backendName} training backend`);
      }
    }
  }

  graph.assertTopologyRevision?.(topologyRevision, `${backendName.toUpperCase()} training`);
  if (graph.weightRevision !== weightRevision) {
    throw new Error(
      `${backendName.toUpperCase()} training graph weights changed before optimizer application.`,
    );
  }
  const currentGradients: GradientMap = new Map();
  for (const tensor of trainables) {
    const gradient = grads.get(tensor.name);
    if (!gradient || gradient.length !== (tensor.buffer as Float32Array).length) {
      throw new Error(`No valid gradient reached trainable tensor '${tensor.name}'.`);
    }
    if (!execution.allFinite(gradient)) {
      throw new Error(`Gradient for trainable tensor '${tensor.name}' is non-finite.`);
    }
    currentGradients.set(tensor.name, gradient);
  }
  const accumulationSignature = JSON.stringify({
    backend: backendName,
    topologyRevision,
    weightRevision,
    trainableTensors,
    mode,
    optimizerDescriptor,
    nextTrainingStep,
    losses: lossDescriptors.map(({ name, logitsTensor: nameOfLogits, weight, normalizer }) => ({
      name,
      logitsTensor: nameOfLogits,
      weight,
      normalizer,
    })),
  });
  const accumulationAggregation = lossDescriptors.length > 1 ||
    lossDescriptors.every((descriptor) => descriptor.normalizer != null)
    ? 'sum'
    : 'mean_by_examples';
  const accumulation = accumulateGradients(graph, {
    backend: backendName,
    gradients: currentGradients,
    examples: count,
    loss,
    correct,
    trainableNames: trainableTensors,
    accumulationSteps: gradientAccumulationSteps,
    flush: flushGradientAccumulation,
    signature: accumulationSignature,
    aggregation: accumulationAggregation,
    hasContribution,
    metrics: lossMetrics,
  });
  if (!accumulation.apply) {
    return {
      backend: backendName,
      loss: accumulation.loss,
      correct: accumulation.correct,
      examples: accumulation.examples,
      losses: accumulation.metrics ?? lossMetrics,
      updatedTensors: [],
      gradients: accumulation.gradients ?? currentGradients,
      accumulating: !accumulation.complete,
      accumulationStep: accumulation.microbatches,
      gradientAccumulationSteps,
    };
  }

  const clipping = execution.clipGradients(
    accumulation.gradients,
    optimizerDescriptor.optimizer.maxGradNorm,
  );
  const updatedTensors: Tensor[] = [];
  for (const trainable of trainables) {
    const gradient = accumulation.gradients.get(trainable.name)!;
    const updateOptions = {
      ...optimizer,
      maxGradNorm: 0,
      mode,
      step: nextTrainingStep,
    };
    updatedTensors.push(execution.applyTensorUpdate(graph, trainable, gradient, updateOptions));
  }
  graph.trainingStep = nextTrainingStep;
  if (mode !== 'adamw') graph.optimizerState = null;
  graph.optimizerDescriptor = optimizerDescriptor;
  synchronizeTrainingGraphState(graph);

  return {
    backend: backendName,
    loss: accumulation.loss,
    correct: accumulation.correct,
    examples: accumulation.examples,
    losses: accumulation.metrics ?? lossMetrics,
    updatedTensors,
    gradients: accumulation.gradients,
    accumulating: false,
    accumulationStep: accumulation.microbatches,
    gradientAccumulationSteps,
    globalGradNorm: clipping.norm,
    gradientScale: clipping.scale,
  };
}
