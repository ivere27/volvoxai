import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';

test('CPU QConv2D executes canonical NHWC/OHWI W8A8 with per-output weight scales', async () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 2, 2], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  const weight = graph.addWeight('weight', [2, 1, 1, 2], 'int8', {
    buffer: Int8Array.of(1, -1, 2, 1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25, 0.5], zero_points: [0, 0] },
  });
  const bias = graph.addWeight('bias', [2], 'int32', { buffer: Int32Array.of(1, -2) });
  const { out } = graph.addOp('QConv2D', { input, weight, bias }, {
    out: {
      name: 'out', shape: [1, 2, 2, 2], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  }, { weight_layout: 'OHWI', data_layout: 'NHWC' });
  graph.setOutputs([out.name]);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(2, 4, 6, 8, 10, 12, 14, 16) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [0, 6, 0, 18, 0, 30, 0, 42]);
});

test('CPU QConv2D applies quantized ReLU at the output descriptor zero point', async () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 1, 2, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  const weight = graph.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.5], zero_points: [0] },
  });
  const { out } = graph.addOp('QConv2D', { input, weight }, {
    out: {
      name: 'out', shape: [1, 1, 2, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  }, { weight_layout: 'OHWI', data_layout: 'NHWC', relu: 1 });
  graph.setOutputs([out.name]);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(-4, 4) });
  assert.deepEqual([...result.out], [0, 4]);
});

test('CPU QConv2D rejects a per-tensor weight descriptor', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 1, 1, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  const weight = graph.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  graph.addOp('QConv2D', { input, weight }, {
    out: {
      name: 'out', shape: [1, 1, 1, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  }, { weight_layout: 'OHWI', data_layout: 'NHWC' });
  const engine = new CPUEngine();
  assert.throws(
    () => engine.allocateGraph(graph),
    /axis-0 per-channel weight metadata/,
  );
});

test('CPU QConv2D maps an overflowing zero accumulator to its output zero point', async () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 1, 1, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 3e38, zero_point: 1 },
  });
  const weight = graph.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [3e38], zero_points: [1] },
  });
  const { out } = graph.addOp('QConv2D', { input, weight }, {
    out: {
      name: 'out', shape: [1, 1, 1, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 1, zero_point: -7 },
    },
  }, { weight_layout: 'OHWI', data_layout: 'NHWC' });
  graph.setOutputs([out.name]);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(1) });
  assert.deepEqual([...result.out], [-7]);
});
