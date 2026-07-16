import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Graph } from '../ts/core/Graph.js';

const q = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 };

function output(name, shape, dtype = 'int8', quantization = q) {
  return { name, shape, dtype, quantization };
}

test('CPU keeps raw W8A8 bytes through MaxPool, nearest resize, reshape, and concat', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2, 2, 1], 'int8', { quantization: q });
  const { out: pooled } = graph.addOp('MaxPool2D', { input }, {
    out: output('pooled', [1, 1, 1, 1]),
  }, { kernel: [2, 2], stride: [2, 2], padding: [0, 0] });
  const { out: resized } = graph.addOp('ResizeNearest2D', { input: pooled }, {
    out: output('resized', [1, 1, 2, 1]),
  }, { mode: 'nearest' });
  const { out: flat } = graph.addOp('Reshape', { input: resized }, {
    out: output('flat', [1, 2]),
  });
  const { out } = graph.addOp('Concat', { input0: flat, input1: flat }, {
    out: output('out', [1, 4]),
  }, { axis: 1, count: 2 });
  graph.outputNames = [out.name];
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(-10, 2, 5, -2) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [5, 5, 5, 5]);
});

test('CPU refuses a raw shape operation that would change a byte tensor interpretation', () => {
  const graph = new Graph();
  const input = graph.addInput('input', [1], 'int8', { quantization: q });
  graph.addOp('Identity', { input }, {
    out: output('out', [1], 'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: -3 }),
  });
  const engine = new CPUEngine();
  assert.throws(
    () => engine.allocateGraph(graph),
    /identical per-tensor quantization metadata/,
  );
});

test('CPU refuses bilinear resize and fused sigmoid concat for byte activations', () => {
  const resizeGraph = new Graph();
  const input = resizeGraph.addInput('input', [1, 1, 1, 1], 'int8', { quantization: q });
  resizeGraph.addOp('Resize', { input }, {
    out: output('resized', [1, 2, 2, 1]),
  }, { mode: 'linear' });
  assert.throws(
    () => new CPUEngine().allocateGraph(resizeGraph),
    /mode "nearest"/,
  );

  const concatGraph = new Graph();
  const concatenatedInput = concatGraph.addInput('input', [1, 2, 2, 1], 'int8', {
    quantization: q,
  });
  const { out } = concatGraph.addOp('Concat', {
    input0: concatenatedInput, input1: concatenatedInput,
  }, {
    out: output('out', [1, 2, 2, 2]),
  }, { axis: 3, count: 2, sigmoid: 1 });
  assert.throws(
    () => new CPUEngine().allocateGraph(concatGraph),
    /no fused sigmoid/,
  );
  assert.equal(out.dtype, 'int8');
});

test('CPU typed pool and nearest resize reject non-canonical semantics', () => {
  const dilationGraph = new Graph();
  const dilationInput = dilationGraph.addInput('input', [1, 3, 3, 1], 'int8', { quantization: q });
  dilationGraph.addOp('MaxPool2D', { input: dilationInput }, {
    out: output('pooled', [1, 2, 2, 1]),
  }, { kernel: [2, 2], stride: [2, 2], pads: [0, 0, 1, 1], dilation: [2, 1] });
  assert.throws(
    () => new CPUEngine().allocateGraph(dilationGraph),
    /only with unit dilation/,
  );

  const ceilGraph = new Graph();
  const ceilInput = ceilGraph.addInput('input', [1, 3, 3, 1], 'int8', { quantization: q });
  ceilGraph.addOp('MaxPool2D', { input: ceilInput }, {
    out: output('pooled', [1, 2, 2, 1]),
  }, { kernel: [2, 2], stride: [2, 2], pads: [0, 0, 1, 1], ceil_mode: true });
  assert.throws(
    () => new CPUEngine().allocateGraph(ceilGraph),
    /does not support ceil_mode/,
  );

  const resizeGraph = new Graph();
  const resizeInput = resizeGraph.addInput('input', [1, 2, 2, 1], 'int8', { quantization: q });
  const { out: resized } = resizeGraph.addOp('Resize', { input: resizeInput }, {
    out: output('resized', [1, 4, 4, 1]),
  }, { mode: 'nearest', coordinate_transformation_mode: 'half_pixel' });
  assert.throws(
    () => new CPUEngine().allocateGraph(resizeGraph),
    /coordinate_transformation_mode "asymmetric"/,
  );
  assert.equal(resized.dtype, 'int8');
});
