import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { _cpuQMaskedMean } from '../ts/ops/qMaskedMean.js';

function qMaskedMeanGraph({
  inputDtype = 'int8',
  outputDtype = 'int8',
  inputShape = [2, 3, 2],
  inputValues = [1, -3, 100, 100, 5, 1, -2, 3, 4, 5, 6, 7],
  maskValues = [1, 0, 1, 0, 0, 0],
  inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
  outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 2 },
  outputFill = 71,
  params = {},
  maskShape = null,
  outputShape = null,
} = {}) {
  const [batch, sequence, width] = inputShape;
  const graph = new Graph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: inputDtype === 'int8' ? Int8Array.from(inputValues) : Uint8Array.from(inputValues),
    quantization: inputQuantization,
  });
  const mask = graph.addInput('mask', maskShape || [batch, sequence], 'int32', {
    buffer: Int32Array.from(maskValues),
  });
  const output = graph.addOp('QMaskedMean', { input, mask }, {
    out: {
      name: 'out', shape: outputShape || [batch, width], dtype: outputDtype,
      buffer: outputDtype === 'int8'
        ? new Int8Array((outputShape || [batch, width]).reduce((total, value) => total * value, 1)).fill(outputFill)
        : new Uint8Array((outputShape || [batch, width]).reduce((total, value) => total * value, 1)).fill(outputFill),
      quantization: outputQuantization,
    },
  }, params).out;
  return { graph, node: graph.nodes[0], input, mask, output };
}

test('QMaskedMean averages only nonzero-mask I8 tokens and writes zero point for an empty row', () => {
  const { node, output } = qMaskedMeanGraph();
  _cpuQMaskedMean(node);
  assert.ok(output.buffer instanceof Int8Array);
  assert.deepEqual([...output.buffer], [10, 2, 2, 2]);
});

test('QMaskedMean supports independent asymmetric U8 input and output descriptors', () => {
  const { node, output } = qMaskedMeanGraph({
    inputDtype: 'uint8', outputDtype: 'uint8', inputShape: [1, 2, 2],
    inputValues: [132, 124, 128, 136], maskValues: [1, 1],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 10 },
  });
  _cpuQMaskedMean(node);
  assert.ok(output.buffer instanceof Uint8Array);
  assert.deepEqual([...output.buffer], [11, 11]);
});

test('QMaskedMean uses ties-to-even requantization after the raw-domain mean', () => {
  const { node, output } = qMaskedMeanGraph({
    inputShape: [1, 2, 2], inputValues: [0, 1, 1, 2], maskValues: [1, 1],
    inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  _cpuQMaskedMean(node);
  assert.deepEqual([...output.buffer], [0, 2]);
});

test('QMaskedMean keeps the staged F32 product/add schedule at an adversarial half step', () => {
  // With FMA contraction this valid descriptor rounds to -63. The portable
  // contract rounds the product to F32 before adding the zero point, giving
  // exactly -62.5 and ties-to-even -62.
  const sequence = 127;
  const { node, output } = qMaskedMeanGraph({
    inputShape: [1, sequence, 1], inputValues: new Array(sequence).fill(-128),
    maskValues: new Array(sequence).fill(1),
    inputQuantization: { scheme: 'per_tensor', scale: 0.001, zero_point: 127 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.01, zero_point: -37 },
  });
  _cpuQMaskedMean(node);
  assert.deepEqual([...output.buffer], [-62]);
});

test('QMaskedMean validates complete descriptors before output writes', () => {
  const malformedMask = qMaskedMeanGraph({ outputFill: 73, maskShape: [1, 6] });
  assert.throws(() => _cpuQMaskedMean(malformedMask.node), /\[B,S,D\], \[B,S\], or \[B,D\]/);
  assert.deepEqual([...malformedMask.output.buffer], new Array(4).fill(73));

  const parameterized = qMaskedMeanGraph({ outputFill: 73, params: { axis: 1 } });
  assert.throws(() => _cpuQMaskedMean(parameterized.node), /accepts no parameters/);
  assert.deepEqual([...parameterized.output.buffer], new Array(4).fill(73));

  const mutable = qMaskedMeanGraph({ outputFill: 73 });
  mutable.input.quantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -1 };
  assert.throws(() => _cpuQMaskedMean(mutable.node), /immutable per_tensor/);
  assert.deepEqual([...mutable.output.buffer], new Array(4).fill(73));
});

test('GraphLoader and CPUEngine retain QMaskedMean as a typed byte-domain router edge', async () => {
  const graph = new Graph();
  const tensors = new Map();
  const input = graph.addInput('router_tokens', [1, 3, 2], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
  });
  const mask = graph.addInput('router_keep', [1, 3], 'int32');
  tensors.set(input.name, input);
  tensors.set(mask.name, mask);
  GraphLoader._buildFromBlueprint(graph, {
    nodes: [{
      opType: 'QMaskedMean', inputs: { input: 'router_tokens', mask: 'router_keep' },
      outputs: { out: 'router_mean' }, outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'uint8' },
      outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.5, zero_point: 128 } },
      params: {},
    }],
  }, tensors);
  graph.assertValid();
  assert.equal(graph.getTensor('router_mean').dtype, 'uint8');
  assert.ok(graph.getTensor('router_mean').quantization);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({
    router_tokens: Int8Array.of(1, -3, 100, 100, 5, 1),
    router_keep: Int32Array.of(1, 0, 1),
  });
  assert.ok(result.router_mean instanceof Uint8Array);
  assert.deepEqual([...result.router_mean], [132, 128]);
});
