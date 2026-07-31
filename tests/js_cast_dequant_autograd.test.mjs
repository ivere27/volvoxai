import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/index.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

async function loss(graph, engine, targets) {
  await engine.execute({});
  const values = graph.getTensor(graph.outputNames[0]).buffer;
  let total = 0;
  for (let row = 0; row < targets.length; row++) {
    const offset = row * 2;
    const maximum = Math.max(values[offset], values[offset + 1]);
    total += maximum + Math.log(Math.exp(values[offset] - maximum) + Math.exp(values[offset + 1] - maximum)) -
      values[offset + targets[row]];
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
  const tolerance = 3e-4 + 3e-3 * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: analytic=${actual}, numeric=${expected}, tolerance=${tolerance}`);
}

test('floating Cast propagates its identity gradient', async () => {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', { buffer: Float32Array.of(0.2, -0.4) });
  const { out } = graph.addOp('Cast', { input }, { out: [1, 2] }, { to: 'float32' });
  graph.setOutputs([out.name]);
  const result = await CPUAutograd.trainStep(graph, {
    targets: [1], trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0 },
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  close(result.gradients.get('input')[0], await numericGradient(graph, engine, 'input', 0, [1]), 'Cast input[0]');
});

test('DequantizeLinear propagates F32 input and scalar-scale gradients', async () => {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', { buffer: Float32Array.of(0.5, -0.25) });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.8) });
  const zeroPoint = graph.addWeight('zero', [1], 'float32', { buffer: Float32Array.of(0.1) });
  const { out } = graph.addOp('DequantizeLinear', { input, scale, zero_point: zeroPoint }, { out: [1, 2] });
  graph.setOutputs([out.name]);
  const result = await CPUAutograd.trainStep(graph, {
    targets: [0], trainableTensors: ['input', 'scale'], updateMode: 'sgd', optimizer: { learningRate: 0 },
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  close(result.gradients.get('input')[1], await numericGradient(graph, engine, 'input', 1, [0]), 'Dequant input[1]');
  close(result.gradients.get('scale')[0], await numericGradient(graph, engine, 'scale', 0, [0]), 'Dequant scale');
});
