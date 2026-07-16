import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Graph } from '../ts/core/Graph.js';

function typedInput(graph, name, dtype, quantization) {
  return graph.addInput(name, [5], dtype, { quantization });
}

test('CPU QAdd requantizes differing I8/U8 activation descriptors', async () => {
  const graph = new Graph();
  const a = typedInput(graph, 'a', 'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: -2 });
  const b = typedInput(graph, 'b', 'uint8', { scheme: 'per_tensor', scale: 0.25, zero_point: 128 });
  const { out } = graph.addOp('QAdd', { a, b }, {
    out: {
      name: 'out', shape: [5], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 3 },
    },
  });
  graph.outputNames = [out.name];
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({
    a: Int8Array.of(-2, 0, 2, 10, -128),
    b: Uint8Array.of(128, 132, 120, 255, 0),
  });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [3, 7, 3, 78, -128]);
});

test('CPU QAdd refuses a graph with non-authoritative output quantization metadata', () => {
  const graph = new Graph();
  const a = typedInput(graph, 'a', 'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: 0 });
  const b = typedInput(graph, 'b', 'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: 0 });
  const { out } = graph.addOp('QAdd', { a, b }, { out: { name: 'out', shape: [5], dtype: 'int8' } });
  const engine = new CPUEngine();
  assert.throws(
    () => engine.allocateGraph(graph),
    /exact-shape canonical per-tensor I8\/U8 edges/,
  );
  assert.equal(out.quantization, null);
});

test('CPU QAdd clamps fused ReLU and ReLU6 in the output quantization domain', async () => {
  const graph = new Graph();
  const q = { scheme: 'per_tensor', scale: 0.5, zero_point: -2 };
  const a = graph.addInput('a', [3], 'int8', { quantization: q });
  const b = graph.addInput('b', [3], 'int8', { quantization: q });
  const { out: relu } = graph.addOp('QAdd', { a, b }, {
    out: { name: 'relu', shape: [3], dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -2 } },
  }, { relu: 1 });
  const { out } = graph.addOp('QAdd', { a: relu, b: relu }, {
    out: { name: 'out', shape: [3], dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -2 } },
  }, { relu: 2 });
  graph.outputNames = [out.name];
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ a: Int8Array.of(-10, 2, 20), b: Int8Array.of(-10, 2, 20) });
  assert.deepEqual([...result.out], [-2, 10, 10]);
});

test('CPU QAdd maps extreme non-finite real sums to the output zero point', async () => {
  const graph = new Graph();
  const q = { scheme: 'per_tensor', scale: 3e38, zero_point: 0 };
  const a = graph.addInput('a', [1], 'int8', { quantization: q });
  const b = graph.addInput('b', [1], 'int8', { quantization: q });
  const { out } = graph.addOp('QAdd', { a, b }, {
    out: { name: 'out', shape: [1], dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 1, zero_point: 7 } },
  });
  graph.outputNames = [out.name];
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ a: Int8Array.of(127), b: Int8Array.of(-128) });
  assert.deepEqual([...result.out], [7]);
});
