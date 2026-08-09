import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';

const qInput = { scheme: 'per_tensor', scale: 0.25, zero_point: 128 };
const qActivation = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };

function typedOutput(name, shape, dtype = 'int8') {
  return {
    outputs_shape: { out: shape },
    outputs_dtype: { out: dtype },
    outputs: { out: name },
  };
}

test('canonical W8A8 graph document preserves physical bytes through a browser CPU inference island', async () => {
  const graph = new RuntimeGraph();
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
  addWeight('input_scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  addWeight('input_zero_point', [1], 'uint8', { buffer: Uint8Array.of(128) });
  addWeight('weight_scale', [1], 'float32', { buffer: Float32Array.of(0.5) });
  addWeight('weight_zero_point', [1], 'int8', { buffer: Int8Array.of(0) });
  addWeight('out_scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  addWeight('out_zero_point', [1], 'int8', { buffer: Int8Array.of(0) });

  const document = {
    format: 'volvox-graph/v1',
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        input: {
          scheme: 'per_tensor', scale_tensor: 'input_scale',
          zero_point_tensor: 'input_zero_point',
        },
        weight: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'weight_scale',
          zero_point_tensor: 'weight_zero_point',
        },
        ...Object.fromEntries(
          ['qx', 'conv', 'added', 'pooled', 'resized', 'flat', 'concat'].map((name) => [name, {
            scheme: 'per_tensor', scale_tensor: 'out_scale',
            zero_point_tensor: 'out_zero_point',
          }]),
        ),
      },
    },
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
        params: { shape: [1, 2] },
      },
      {
        opType: 'Concat', inputs: { input0: 'flat', input1: 'flat' },
        ...typedOutput('concat', [1, 4]),
        params: { axis: 1 },
      },
      {
        opType: 'DequantizeLinear', inputs: { input: 'concat', scale: 'out_scale', zero_point: 'out_zero_point' },
        ...typedOutput('result', [1, 4], 'float32'),
      },
    ],
    outputs: ['result'],
  };
  RuntimeGraphLoader._buildFromGraphDocument(graph, document, tensors, {
    quantizationByTensor: Object.fromEntries(
      ['qx', 'conv', 'added', 'pooled', 'resized', 'flat', 'concat']
        .map((name) => [name, qActivation]),
    ),
    reservedNames: new Set([
      'input_scale', 'input_zero_point', 'weight_scale', 'weight_zero_point',
      'out_scale', 'out_zero_point',
    ]),
  });
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
  const graph = new RuntimeGraph();
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
  addWeight('activation_scale', [1], 'float32', { buffer: Float32Array.of(1) });
  addWeight('activation_zero_point', [1], 'int8', { buffer: Int8Array.of(0) });
  addWeight('row_scale', [4], 'float32', { buffer: Float32Array.of(1, 1, 1, 1) });
  addWeight('row_zero_point', [4], 'int8', { buffer: Int8Array.of(0, 0, 0, 0) });
  addWeight('logit_scale', [3], 'float32', { buffer: Float32Array.of(1, 1, 1) });
  addWeight('logit_zero_point', [3], 'int8', { buffer: Int8Array.of(0, 0, 0) });

  const typed = (name, shape, dtype = 'int8') => ({
    outputs: { out: name },
    outputs_shape: { out: shape },
    outputs_dtype: { out: dtype },
  });
  const document = {
    format: 'volvox-graph/v1',
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        table: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'row_scale',
          zero_point_tensor: 'row_zero_point',
        },
        identity: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'row_scale',
          zero_point_tensor: 'row_zero_point',
        },
        logit_weight: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'logit_scale',
          zero_point_tensor: 'logit_zero_point',
        },
        ...Object.fromEntries(
          ['embedded', 'q', 'k', 'v', 'attended', 'logits'].map((name) => [name, {
            scheme: 'per_tensor', scale_tensor: 'activation_scale',
            zero_point_tensor: 'activation_zero_point',
          }]),
        ),
      },
    },
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
    outputs: ['token_ids'],
  };
  RuntimeGraphLoader._buildFromGraphDocument(graph, document, tensors, {
    quantizationByTensor: Object.fromEntries(
      ['embedded', 'q', 'k', 'v', 'attended', 'logits']
        .map((name) => [name, activation]),
    ),
    reservedNames: new Set([
      'activation_scale', 'activation_zero_point', 'row_scale', 'row_zero_point',
      'logit_scale', 'logit_zero_point',
    ]),
  });
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
