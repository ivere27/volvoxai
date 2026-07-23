import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/index.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

function graphFor(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 3, 2, 1], 'float32', {
    buffer: Float32Array.of(0.1, 0.5, 0.7, -0.4, -0.2, 0.3),
  });
  const outputShape = opType === 'ProfileX' ? [1, 2, 2]
    : opType === 'ProfileY' ? [1, 2, 3] : [1, 1, 2];
  const { out } = graph.addOp(opType, { input }, { out: outputShape });
  graph.setOutputs([out.name]);
  return graph;
}

function targetsFor(opType) {
  return opType === 'ProfileX' || opType === 'ProfileY' ? [0, 1] : [0];
}

async function loss(graph, engine, targets) {
  await engine.execute({});
  const logits = graph.getTensor(graph.outputNames[0]);
  const classes = logits.shape.at(-1);
  let total = 0;
  for (let row = 0; row < targets.length; row++) {
    const offset = row * classes;
    let maximum = -Infinity;
    for (let column = 0; column < classes; column++) maximum = Math.max(maximum, logits.buffer[offset + column]);
    let denominator = 0;
    for (let column = 0; column < classes; column++) denominator += Math.exp(logits.buffer[offset + column] - maximum);
    total += maximum + Math.log(denominator) - logits.buffer[offset + targets[row]];
  }
  return total / targets.length;
}

async function numericGradient(graph, engine, index, targets, epsilon = 1e-3) {
  const input = graph.getTensor('input');
  const original = input.buffer[index];
  input.buffer[index] = original + epsilon;
  const positive = await loss(graph, engine, targets);
  input.buffer[index] = original - epsilon;
  const negative = await loss(graph, engine, targets);
  input.buffer[index] = original;
  return (positive - negative) / (2 * epsilon);
}

function close(actual, expected, label) {
  const tolerance = 4e-4 + 4e-3 * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: analytic=${actual}, numeric=${expected}, tolerance=${tolerance}`);
}

test('vision profile and softargmax gradients match finite differences', async (t) => {
  for (const opType of ['MeanHeight', 'ProfileX', 'ProfileY', 'SpatialSoftargmaxY']) {
    await t.test(opType, async () => {
      const graph = graphFor(opType);
      const targets = targetsFor(opType);
      const result = await CPUAutograd.trainStep(graph, {
        targets,
        trainableTensors: ['input'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0 },
      });
      const engine = new CPUEngine();
      engine.allocateGraph(graph);
      for (let index = 0; index < graph.getTensor('input').buffer.length; index++) {
        close(result.gradients.get('input')[index], await numericGradient(graph, engine, index, targets),
          `${opType} input[${index}]`);
      }
    });
  }
});
