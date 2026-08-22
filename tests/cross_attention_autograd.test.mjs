import test from 'node:test';
import assert from 'node:assert/strict';

import { TrainingGraph as Graph } from '../ts/training/TrainingGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

function crossAttentionGraph() {
  const graph = new Graph();
  const q = graph.addWeight('q', [1, 2, 2], 'float32', {
    buffer: Float32Array.of(0.2, -0.4, 0.6, 0.1),
  });
  const kv = graph.addWeight('kv', [1, 2, 2], 'float32', {
    buffer: Float32Array.of(-0.3, 0.7, 0.5, -0.2),
  });
  const weight = graph.addWeight('weight', [6, 2], 'float32', {
    buffer: Float32Array.of(
      0.2, -0.1, 0.3, 0.4,
      -0.2, 0.1, 0.5, -0.3,
      0.4, 0.2, -0.1, 0.6,
    ),
  });
  const scale = graph.addWeight('scale', [6], 'float32', {
    buffer: Float32Array.of(1.1, 0.9, 1.2, 0.8, 1.05, 0.95),
  });
  const bias = graph.addWeight('bias', [6], 'float32', {
    buffer: Float32Array.of(0.02, -0.03, 0.04, 0.01, -0.02, 0.03),
  });
  const { out } = graph.addOp('CrossAttention', { q, kv, weight, scale, bias }, { out: [1, 2, 2] }, {
    heads: 1,
  });
  graph.setOutputs([out.name]);
  return graph;
}

async function loss(graph, engine, targets) {
  await engine.execute({});
  const logits = graph.getTensor(graph.outputNames[0]).buffer;
  let total = 0;
  for (let row = 0; row < targets.length; row++) {
    const start = row * 2;
    const maximum = Math.max(logits[start], logits[start + 1]);
    const denominator = Math.exp(logits[start] - maximum) + Math.exp(logits[start + 1] - maximum);
    total += maximum + Math.log(denominator) - logits[start + targets[row]];
  }
  return total / targets.length;
}

async function numericGradient(graph, engine, name, index, targets, epsilon = 1e-3) {
  const tensor = graph.getTensor(name);
  const original = tensor.buffer[index];
  tensor.buffer[index] = original + epsilon;
  const positive = await loss(graph, engine, targets);
  tensor.buffer[index] = original - epsilon;
  const negative = await loss(graph, engine, targets);
  tensor.buffer[index] = original;
  return (positive - negative) / (2 * epsilon);
}

function close(actual, expected, label) {
  const tolerance = 4e-4 + 4e-3 * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: analytic=${actual}, numeric=${expected}, tolerance=${tolerance}`);
}

test('CrossAttention backpropagates projected Q/K/V and affine projection parameters', async () => {
  const graph = crossAttentionGraph();
  const targets = [0, 1];
  const result = await CPUAutograd.trainStep(graph, {
    targets,
    trainableTensors: ['q', 'kv', 'weight', 'scale', 'bias'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  for (const [name, index] of [['q', 0], ['kv', 1], ['weight', 0], ['scale', 0], ['bias', 0]]) {
    const numeric = await numericGradient(graph, engine, name, index, targets);
    close(result.gradients.get(name)[index], numeric, `CrossAttention ${name}[${index}]`);
  }
});
