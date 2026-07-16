import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { _cpuQArgMax } from '../ts/ops/qArgMax.js';

function byteValues(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function normalizedAxis(axis, rank) {
  return axis < 0 ? axis + rank : axis;
}

function qArgMaxGraph({
  shape = [2, 3, 2],
  dtype = 'int8',
  values = [1, 9, 5, 9, 5, 4, 2, 0, 8, 7, 6, 10],
  quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
  axis = 1,
  outputShape = null,
  outputFill = 71,
} = {}) {
  const graph = new Graph();
  const input = graph.addInput('input', shape, dtype, {
    buffer: byteValues(dtype, values), quantization,
  });
  const resolvedAxis = normalizedAxis(axis, shape.length);
  const expectedShape = [...shape.slice(0, resolvedAxis), ...shape.slice(resolvedAxis + 1)];
  const output = graph.addOp('QArgMax', { input }, {
    out: {
      name: 'out', shape: outputShape || expectedShape, dtype: 'int32',
      buffer: new Int32Array((outputShape || expectedShape).reduce((count, dimension) => count * dimension, 1)).fill(outputFill),
    },
  }, { axis }).out;
  return { graph, node: graph.nodes[0], input, output };
}

test('QArgMax compares raw I8 values over non-final axes and keeps first ties', () => {
  const { node, output } = qArgMaxGraph();
  _cpuQArgMax(node);
  assert.ok(output.buffer instanceof Int32Array);
  assert.deepEqual([...output.buffer], [1, 0, 1, 2]);
});

test('QArgMax accepts U8 input and a normalized negative axis without decoding logits', () => {
  // [outer=2, axis=3, inner=2]; intentionally asymmetric metadata proves
  // that raw byte order, not zero-point-centered/F32 values, selects winners.
  const { node, output } = qArgMaxGraph({
    shape: [2, 3, 2], dtype: 'uint8', axis: -2,
    values: [128, 7, 255, 7, 255, 6, 1, 4, 2, 9, 2, 8],
    quantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 201 },
  });
  _cpuQArgMax(node);
  assert.deepEqual([...output.buffer], [1, 0, 1, 1]);
});

test('QArgMax supports rank eight while keeping only I32 indices visible', () => {
  const shape = [1, 1, 1, 1, 1, 1, 1, 4];
  const { node, output } = qArgMaxGraph({
    shape, axis: -1, values: [-3, 5, 5, 2], quantization: {
      scheme: 'per_tensor', scale: 0.5, zero_point: 17,
    },
  });
  _cpuQArgMax(node);
  assert.deepEqual([...output.buffer], [1]);
  assert.equal(output.quantization, null);
});

test('QArgMax rejects noncanonical descriptors and aliases before output writes', () => {
  const malformed = qArgMaxGraph();
  malformed.node.params.extra = true;
  assert.throws(() => _cpuQArgMax(malformed.node), /exactly \{ axis: integer \}/);
  assert.deepEqual([...malformed.output.buffer], new Array(4).fill(71));

  const keepdims = qArgMaxGraph({ outputShape: [2, 1, 2] });
  assert.throws(() => _cpuQArgMax(keepdims.node), /incompatible axis or output dimensions/);
  assert.deepEqual([...keepdims.output.buffer], new Array(4).fill(71));

  const perAxis = qArgMaxGraph();
  perAxis.input.quantization = Object.freeze({
    scheme: 'per_axis', axis: 1, scales: Object.freeze([0.25, 0.25, 0.25]),
    zero_points: Object.freeze([0, 0, 0]),
  });
  assert.throws(() => _cpuQArgMax(perAxis.node), /immutable per_tensor/);
  assert.deepEqual([...perAxis.output.buffer], new Array(4).fill(71));

  const mutable = qArgMaxGraph();
  mutable.input.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 };
  assert.throws(() => _cpuQArgMax(mutable.node), /immutable per_tensor/);
  assert.deepEqual([...mutable.output.buffer], new Array(4).fill(71));

  const aliased = qArgMaxGraph({ shape: [2, 4], axis: 1, values: [1, 2, 3, 4, 4, 3, 2, 1] });
  const shared = new ArrayBuffer(8);
  aliased.input.buffer = new Int8Array(shared);
  aliased.output.buffer = new Int32Array(shared);
  assert.throws(() => _cpuQArgMax(aliased.node), /distinct from input storage/);
});
