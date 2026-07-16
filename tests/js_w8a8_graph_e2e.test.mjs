import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';

const qInput = { scheme: 'per_tensor', scale: 0.25, zero_point: 128 };
const qActivation = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };

function typedOutput(name, shape, dtype = 'int8', quantization = qActivation) {
  return {
    outputs_shape: { out: shape },
    outputs_dtype: { out: dtype },
    ...(quantization ? { outputs_quantization: { out: quantization } } : {}),
    outputs: { out: name },
  };
}

test('canonical W8A8 blueprint preserves physical bytes through a browser CPU inference island', async () => {
  const graph = new Graph();
  const tensors = new Map();
  const addInput = (name, shape, dtype, options) => {
    const tensor = graph.addInput(name, shape, dtype, options);
    tensors.set(name, tensor);
    return tensor;
  };
  const addWeight = (name, shape, dtype, options) => {
    const tensor = graph.addWeight(name, shape, dtype, options);
    tensors.set(name, tensor);
    return tensor;
  };
  addInput('input', [1, 2, 2, 1], 'uint8', { quantization: qInput });
  addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(2),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.5], zero_points: [0] },
  });
  addWeight('bias', [1], 'int32', { buffer: Int32Array.of(0) });
  addWeight('out_scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  addWeight('out_zero_point', [1], 'int8', { buffer: Int8Array.of(0) });

  const config = {
    nodes: [
      {
        opType: 'RequantizeLinear', inputs: { input: 'input' },
        ...typedOutput('qx', [1, 2, 2, 1]),
      },
      {
        opType: 'QConv2D', inputs: { input: 'qx', weight: 'weight', bias: 'bias' },
        ...typedOutput('conv', [1, 2, 2, 1]),
        params: { data_layout: 'NHWC', weight_layout: 'OHWI', stride: [1, 1], dilation: [1, 1], pads: [0, 0, 0, 0] },
      },
      {
        opType: 'QAdd', inputs: { a: 'conv', b: 'conv' },
        ...typedOutput('added', [1, 2, 2, 1]),
      },
      {
        opType: 'MaxPool2D', inputs: { input: 'added' },
        ...typedOutput('pooled', [1, 1, 1, 1]),
        params: { kernel: [2, 2], stride: [2, 2], padding: [0, 0] },
      },
      {
        opType: 'ResizeNearest2D', inputs: { input: 'pooled' },
        ...typedOutput('resized', [1, 1, 2, 1]),
        params: { mode: 'nearest' },
      },
      {
        opType: 'Reshape', inputs: { input: 'resized' },
        ...typedOutput('flat', [1, 2]),
      },
      {
        opType: 'Concat', inputs: { input0: 'flat', input1: 'flat' },
        ...typedOutput('concat', [1, 4]),
        params: { axis: 1, count: 2 },
      },
      {
        opType: 'DequantizeLinear', inputs: { input: 'concat', scale: 'out_scale', zero_point: 'out_zero_point' },
        ...typedOutput('result', [1, 4], 'float32', null),
      },
    ],
  };
  GraphLoader._buildFromBlueprint(graph, config, tensors);
  assert.deepEqual(graph.outputNames, ['result']);
  graph.assertValid();
  assert.equal(graph.getTensor('qx').dtype, 'int8');
  assert.deepEqual(graph.getTensor('weight').quantization, {
    scheme: 'per_axis', axis: 0, scales: [0.5], zero_points: [0],
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Uint8Array.of(128, 129, 130, 131) });
  assert.ok(result.result instanceof Float32Array);
  assert.deepEqual([...result.result], [1.5, 1.5, 1.5, 1.5]);
});

test('canonical W8A8 decoder island keeps embedding, attention, logits, and token selection typed', async () => {
  const graph = new Graph();
  const tensors = new Map();
  const addInput = (name, shape, dtype, options) => {
    const tensor = graph.addInput(name, shape, dtype, options);
    tensors.set(name, tensor);
    return tensor;
  };
  const addWeight = (name, shape, dtype, options) => {
    const tensor = graph.addWeight(name, shape, dtype, options);
    tensors.set(name, tensor);
    return tensor;
  };
  const activation = { scheme: 'per_tensor', scale: 1, zero_point: 0 };
  const rows = { scheme: 'per_axis', axis: 0, scales: [1, 1, 1, 1], zero_points: [0, 0, 0, 0] };
  addInput('ids', [2], 'int32');
  addWeight('table', [4, 4], 'int8', {
    buffer: Int8Array.of(
      0, 0, 0, 0,
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
    ),
    quantization: rows,
  });
  addWeight('identity', [4, 4], 'int8', {
    buffer: Int8Array.of(
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ),
    quantization: rows,
  });
  addWeight('identity_bias', [4], 'int32', { buffer: Int32Array.of(0, 0, 0, 0) });
  addWeight('logit_weight', [3, 4], 'int8', {
    buffer: Int8Array.of(
      1, 0, 0, 0,
      0, 2, 0, 0,
      -1, -1, 0, 0,
    ),
    quantization: { scheme: 'per_axis', axis: 0, scales: [1, 1, 1], zero_points: [0, 0, 0] },
  });
  addWeight('logit_bias', [3], 'int32', { buffer: Int32Array.of(0, 0, 0) });

  const typed = (name, shape, dtype = 'int8', quantization = activation) => ({
    outputs: { out: name },
    outputs_shape: { out: shape },
    outputs_dtype: { out: dtype },
    ...(quantization ? { outputs_quantization: { out: quantization } } : {}),
  });
  const config = {
    nodes: [
      { opType: 'QEmbedding', inputs: { input: 'ids', weight: 'table' }, ...typed('embedded', [2, 4]) },
      { opType: 'QLinear', inputs: { input: 'embedded', weight: 'identity', bias: 'identity_bias' }, ...typed('q', [2, 4]) },
      { opType: 'QLinear', inputs: { input: 'embedded', weight: 'identity', bias: 'identity_bias' }, ...typed('k', [2, 4]) },
      { opType: 'QLinear', inputs: { input: 'embedded', weight: 'identity', bias: 'identity_bias' }, ...typed('v', [2, 4]) },
      {
        opType: 'QSDPA', inputs: { q: 'q', k: 'k', v: 'v' }, ...typed('attended', [2, 4]),
        params: { heads: 1, causal: true, scale: 0.5 },
      },
      {
        opType: 'QLinear', inputs: { input: 'attended', weight: 'logit_weight', bias: 'logit_bias' },
        ...typed('logits', [2, 3]),
      },
      {
        opType: 'QArgMax', inputs: { input: 'logits' },
        outputs: { out: 'token_ids' }, outputs_shape: { out: [2] }, outputs_dtype: { out: 'int32' },
        params: { axis: -1 },
      },
    ],
  };
  GraphLoader._buildFromBlueprint(graph, config, tensors);
  graph.assertValid();
  for (const name of ['embedded', 'q', 'k', 'v', 'attended', 'logits']) {
    assert.equal(graph.getTensor(name).dtype, 'int8');
    assert.ok(graph.getTensor(name).quantization);
  }
  assert.equal(graph.getTensor('token_ids').dtype, 'int32');
  assert.equal(graph.getTensor('token_ids').quantization, null);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ ids: Int32Array.of(1, 2) });
  assert.ok(result.token_ids instanceof Int32Array);
  assert.deepEqual([...result.token_ids], [0, 1]);
});
