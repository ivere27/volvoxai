import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/index.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';
import { exportModelCheckpoint } from '../ts/training/ModelCheckpoint.js';
import { dropoutContext, dropoutMultiplier } from '../ts/ops/dropout.js';
import {
  accumulateGradients,
  clipGradientsGlobal,
  hasPendingGradientAccumulation,
  resetGradientAccumulation,
} from '../ts/training/GradientAccumulation.js';

function weight(graph, name, shape, values) {
  const tensor = graph.addWeight(name, shape);
  tensor.buffer = Float32Array.from(values);
  return tensor;
}

function close(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

async function trainingCrossEntropy(graph, targets, context) {
  await CPUAutograd._forward(graph, {}, context);
  const logits = graph.getTensor(graph.outputNames[0]);
  const classes = logits.shape.at(-1);
  let loss = 0;
  for (let row = 0; row < targets.length; row++) {
    let maximum = -Infinity;
    for (let col = 0; col < classes; col++) maximum = Math.max(maximum, logits.buffer[row * classes + col]);
    let denominator = 0;
    for (let col = 0; col < classes; col++) denominator += Math.exp(logits.buffer[row * classes + col] - maximum);
    loss += maximum + Math.log(denominator) - logits.buffer[row * classes + targets[row]];
  }
  return loss / targets.length;
}

async function numericTrainingGradient(graph, tensorName, index, targets, context) {
  const tensor = graph.getTensor(tensorName);
  const value = tensor.buffer[index];
  const epsilon = 1e-3;
  tensor.buffer[index] = value + epsilon;
  const positive = await trainingCrossEntropy(graph, targets, context);
  tensor.buffer[index] = value - epsilon;
  const negative = await trainingCrossEntropy(graph, targets, context);
  tensor.buffer[index] = value;
  return (positive - negative) / (2 * epsilon);
}

test('CPU Dropout is an inference identity and uses one deterministic inverted mask in forward/backward', async () => {
  const graph = new Graph();
  const parameter = weight(graph, 'parameter', [8, 2], [
    0.2, -0.4, 0.1, 0.7, -0.3, 0.6, 0.8, -0.2,
    0.5, 0.1, -0.6, 0.3, 0.9, -0.8, 0.4, -0.1,
  ]);
  const dropoutNode = graph.addNode({
    id: 'dropout',
    opType: 'Dropout',
    inputs: { input: parameter },
    outputs: { out: { name: 'logits', shape: [8, 2] } },
    params: { p: 0.5, seed: 9 },
  });
  const logits = dropoutNode.outputs.out;
  graph.setOutputs([logits.name]);
  const targets = [0, 1, 0, 1, 1, 0, 1, 0];
  const context = { seed: 17, counter: 23 };
  const before = new Float32Array(parameter.buffer);

  const trained = await CPUAutograd.trainStep(graph, {
    targets,
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    dropout: context,
  });
  const trainingOutput = new Float32Array(logits.buffer);
  const analytic = trained.gradients.get('parameter');
  for (let row = 0; row < 8; row++) {
    const m0 = dropoutMultiplier(dropoutNode, context, row * 2, 0);
    const m1 = dropoutMultiplier(dropoutNode, context, row * 2 + 1, 0);
    close(trainingOutput[row * 2], before[row * 2] * m0);
    close(trainingOutput[row * 2 + 1], before[row * 2 + 1] * m1);
    const z0 = trainingOutput[row * 2];
    const z1 = trainingOutput[row * 2 + 1];
    const maximum = Math.max(z0, z1);
    const e0 = Math.exp(z0 - maximum);
    const e1 = Math.exp(z1 - maximum);
    const probability0 = e0 / (e0 + e1);
    close(analytic[row * 2], (probability0 - (targets[row] === 0 ? 1 : 0)) * m0 / 8);
    close(analytic[row * 2 + 1], ((1 - probability0) - (targets[row] === 1 ? 1 : 0)) * m1 / 8);
  }

  const repeated = await CPUAutograd.trainStep(graph, {
    targets,
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    dropout: context,
  });
  assert.deepEqual([...logits.buffer], [...trainingOutput]);
  assert.deepEqual([...repeated.gradients.get('parameter')], [...analytic]);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await assert.rejects(
    engine.execute({}, { training: { dropout: context } }),
    /does not accept training or Dropout RNG options/,
  );
  await engine.execute({});
  assert.equal(logits.buffer, parameter.buffer, 'inference Dropout must be a zero-copy alias');
  assert.deepEqual([...logits.buffer], [...before]);

  await CPUAutograd.trainStep(graph, {
    targets,
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    dropout: { ...context, counter: context.counter + 1 },
  });
  assert.notEqual(logits.buffer, parameter.buffer, 'training must detach the Dropout output');
  assert.notDeepEqual([...logits.buffer], [...trainingOutput]);
  assert.equal(Object.hasOwn(dropoutNode, 'mask'), false);
});

test('CPU trainStep clips one global norm over all trainables', async () => {
  const graph = new Graph();
  const left = weight(graph, 'left', [1, 2], [0, 0]);
  const right = weight(graph, 'right', [1, 2], [0, 0]);
  const logits = graph.addOp('Add', { a: left, b: right }, { out: [1, 2] }).out;
  graph.setOutputs([logits.name]);

  const result = await CPUAutograd.trainStep(graph, {
    targets: [0],
    trainableTensors: ['left', 'right'],
    updateMode: 'sgd',
    optimizer: { learningRate: 1, maxGradNorm: 0.1 },
  });
  close(result.globalGradNorm, 1);
  close(result.gradientScale, 0.1);
  close(left.buffer[0], 0.05);
  close(left.buffer[1], -0.05);
  close(right.buffer[0], 0.05);
  close(right.buffer[1], -0.05);
});

test('CPU SDPA and CrossSDPA reuse deterministic attention-probability dropout masks in backward', async (t) => {
  const rng = { seed: 31, counter: 7 };
  const executionContext = dropoutContext(rng);

  await t.test('SDPA', async () => {
    const graph = new Graph();
    const qkv = weight(graph, 'qkv', [2, 6], [
      0.4, -0.2, 0.1, 0.5, 0.7, -0.3,
      -0.1, 0.6, 0.3, -0.4, 0.2, 0.8,
    ]);
    const logits = graph.addOp('SDPA', { qkv }, { out: [2, 2] }, {
      heads: 1, causal: false, dropout: 0.5, dropout_seed: 13,
    }).out;
    graph.setOutputs([logits.name]);
    const targets = [0, 1];
    const result = await CPUAutograd.trainStep(graph, {
      targets, trainableTensors: ['qkv'], updateMode: 'sgd',
      optimizer: { learningRate: 0 }, dropout: rng,
    });
    for (let index = 0; index < qkv.buffer.length; index++) {
      const numeric = await numericTrainingGradient(graph, 'qkv', index, targets, executionContext);
      close(result.gradients.get('qkv')[index], numeric, 4e-4);
    }
  });

  await t.test('CrossSDPA', async () => {
    const graph = new Graph();
    const q = weight(graph, 'q', [2, 2], [0.4, -0.2, -0.1, 0.6]);
    const k = weight(graph, 'k', [2, 2], [0.1, 0.5, 0.3, -0.4]);
    const v = weight(graph, 'v', [2, 2], [0.7, -0.3, 0.2, 0.8]);
    const logits = graph.addOp('CrossSDPA', { q, k, v }, { out: [2, 2] }, {
      heads: 1, dropout: 0.5, dropout_seed: 13,
    }).out;
    graph.setOutputs([logits.name]);
    const targets = [0, 1];
    const result = await CPUAutograd.trainStep(graph, {
      targets, trainableTensors: ['q', 'k', 'v'], updateMode: 'sgd',
      optimizer: { learningRate: 0 }, dropout: rng,
    });
    for (const name of ['q', 'k', 'v']) {
      const tensor = graph.getTensor(name);
      for (let index = 0; index < tensor.buffer.length; index++) {
        const numeric = await numericTrainingGradient(graph, name, index, targets, executionContext);
        close(result.gradients.get(name)[index], numeric, 4e-4);
      }
    }
  });
});

test('CPU trainStep accumulates microbatch gradients and advances the optimizer step only on apply', async () => {
  const graph = new Graph();
  const logits = weight(graph, 'logits', [1, 2], [0, 0]);
  graph.setOutputs([logits.name]);
  const revision = graph.weightRevision;
  const common = {
    trainableTensors: ['logits'],
    updateMode: 'sgd',
    optimizer: { learningRate: 1 },
    gradientAccumulationSteps: 2,
  };

  const first = await CPUAutograd.trainStep(graph, { ...common, targets: [0] });
  assert.equal(first.accumulating, true);
  assert.equal(first.accumulationStep, 1);
  assert.equal(first.updatedTensors.length, 0);
  assert.equal(graph.trainingStep, 0);
  assert.equal(graph.weightRevision, revision);
  assert.equal(hasPendingGradientAccumulation(graph), true);
  assert.throws(() => exportModelCheckpoint(graph), /gradient accumulation is pending/);

  const second = await CPUAutograd.trainStep(graph, { ...common, targets: [1] });
  assert.equal(second.accumulating, false);
  assert.equal(second.accumulationStep, 2);
  assert.equal(second.examples, 2);
  assert.equal(second.correct, 1);
  assert.equal(graph.trainingStep, 1);
  close(logits.buffer[0], 0);
  close(logits.buffer[1], 0);
  assert.equal(hasPendingGradientAccumulation(graph), false);
});

test('empty microbatches count toward a window without applying an empty optimizer step', async () => {
  const graph = new Graph();
  const logits = weight(graph, 'logits', [1, 2], [0, 0]);
  graph.setOutputs([logits.name]);
  const common = {
    trainableTensors: ['logits'],
    updateMode: 'sgd',
    optimizer: { learningRate: 1 },
    gradientAccumulationSteps: 2,
    ignoreIndex: -100,
  };

  const empty = await CPUAutograd.trainStep(graph, { ...common, targets: [-100] });
  assert.equal(empty.accumulating, true);
  assert.equal(empty.accumulationStep, 1);
  assert.equal(graph.trainingStep, 0);
  const applied = await CPUAutograd.trainStep(graph, { ...common, targets: [1] });
  assert.equal(applied.accumulating, false);
  assert.equal(applied.examples, 1);
  assert.equal(graph.trainingStep, 1);
  close(logits.buffer[0], -0.5);
  close(logits.buffer[1], 0.5);

  const allEmpty = new Graph();
  const allEmptyLogits = weight(allEmpty, 'all-empty.logits', [1, 2], [0, 0]);
  allEmpty.setOutputs([allEmptyLogits.name]);
  const emptyCommon = { ...common, trainableTensors: ['all-empty.logits'] };
  await CPUAutograd.trainStep(allEmpty, { ...emptyCommon, targets: [-100] });
  const completed = await CPUAutograd.trainStep(allEmpty, { ...emptyCommon, targets: [-100] });
  assert.equal(completed.accumulating, false);
  assert.equal(completed.updatedTensors.length, 0);
  assert.equal(allEmpty.trainingStep, 0);
  assert.equal(hasPendingGradientAccumulation(allEmpty), false);
  assert.deepEqual([...allEmptyLogits.buffer], [0, 0]);
});

test('gradient accumulation validates signatures, supports reset/flush, and global clipping is reusable', () => {
  const graph = {};
  const gradient = new Map([['w', Float32Array.of(3, 4)]]);
  const first = accumulateGradients(graph, {
    backend: 'cpu', gradients: gradient, examples: 1, loss: 2, correct: 0,
    trainableNames: ['w'], accumulationSteps: 3, signature: 'a',
  });
  assert.equal(first.apply, false);
  assert.throws(() => accumulateGradients(graph, {
    backend: 'cpu', gradients: gradient, examples: 1, loss: 2, correct: 0,
    trainableNames: ['w'], accumulationSteps: 3, signature: 'b',
  }), /changed before/);
  resetGradientAccumulation(graph);
  const flushed = accumulateGradients(graph, {
    backend: 'cpu', gradients: gradient, examples: 1, loss: 2, correct: 0,
    trainableNames: ['w'], accumulationSteps: 3, flush: true, signature: 'b',
  });
  assert.equal(flushed.apply, true);
  const clipped = clipGradientsGlobal(flushed.gradients, 2.5);
  close(clipped.norm, 5);
  close(clipped.scale, 0.5);
  assert.deepEqual([...flushed.gradients.get('w')], [1.5, 2]);

  const summedGraph = {};
  const left = accumulateGradients(summedGraph, {
    backend: 'cpu', gradients: new Map([['w', Float32Array.of(0.25, -0.25)]]),
    examples: 2, loss: 0.4, correct: 1, trainableNames: ['w'], accumulationSteps: 2,
    aggregation: 'sum', signature: 'full-window-normalized',
  });
  assert.equal(left.apply, false);
  const summed = accumulateGradients(summedGraph, {
    backend: 'cpu', gradients: new Map([['w', Float32Array.of(-0.1, 0.1)]]),
    examples: 1, loss: 0.2, correct: 1, trainableNames: ['w'], accumulationSteps: 2,
    aggregation: 'sum', signature: 'full-window-normalized',
  });
  assert.equal(summed.apply, true);
  assert.deepEqual([...summed.gradients.get('w')], [...Float32Array.of(0.15, -0.15)]);
  close(summed.loss, 0.6);
});
