import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function batchMatMulGraph() {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 1, 2, 3]);
  const b = graph.addInput('b', [1, 2, 3, 2]);
  const { out } = graph.addOp('BatchMatMul', { a, b }, {
    out: { name: 'out', shape: [2, 2, 2, 2] },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function referenceBatchMatMul(a, b) {
  const output = [];
  for (let leftBatch = 0; leftBatch < 2; leftBatch++) {
    for (let rightBatch = 0; rightBatch < 2; rightBatch++) {
      for (let row = 0; row < 2; row++) {
        for (let column = 0; column < 2; column++) {
          let sum = 0;
          for (let inner = 0; inner < 3; inner++) {
            sum += a[(leftBatch * 2 + row) * 3 + inner] *
              b[(rightBatch * 3 + inner) * 2 + column];
          }
          output.push(sum);
        }
      }
    }
  }
  return output;
}

function logicalGraph() {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 1, 3], 'int32');
  const b = graph.addInput('b', [1, 2, 1], 'int32');
  const equal = graph.addOp('Equal', { a, b }, {
    out: { name: 'equal', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  const greaterOrEqual = graph.addOp('GreaterOrEqual', { a, b }, {
    out: { name: 'greaterOrEqual', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  const notEqual = graph.addOp('Not', { input: equal }, {
    out: { name: 'notEqual', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  graph.setOutputs([equal.name, greaterOrEqual.name, notEqual.name]);
  return graph;
}

function typedSelectionGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [6], 'int32');
  const condition = graph.addInput('condition', [6], 'int32');
  const clipped = graph.addOp('Clip', { input }, {
    out: { name: 'clipped', shape: [6], dtype: 'int32' },
  }, { min: -2, max: 5 }).out;
  const fallback = graph.addWeight('fallback', [6], 'int32', {
    buffer: Int32Array.of(90, 91, 92, 93, 94, 95),
  });
  const selected = graph.addOp('Where', {
    condition, x: clipped, y: fallback,
  }, {
    out: { name: 'selected', shape: [6], dtype: 'int32' },
  }).out;
  graph.setOutputs([clipped.name, selected.name]);
  return graph;
}

function typedShapeGraph() {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 2], 'int32');
  const b = graph.addInput('b', [2, 1], 'int32');
  const joined = graph.addOp('Concat', { input0: a, input1: b }, {
    out: { name: 'joined', shape: [2, 3], dtype: 'int32' },
  }, { axis: 1, count: 2 }).out;
  const reshaped = graph.addOp('Reshape', { input: joined }, {
    out: { name: 'reshaped', shape: [3, 2], dtype: 'int32' },
  }).out;
  const transposed = graph.addOp('Transpose', { input: reshaped }, {
    out: { name: 'transposed', shape: [2, 3], dtype: 'int32' },
  }, { perm: [1, 0] }).out;
  const sliced = graph.addOp('Slice', { input: transposed }, {
    out: { name: 'sliced', shape: [2, 2], dtype: 'int32' },
  }, { axes: [1], starts: [1], steps: [1] }).out;
  const shaped = graph.addOp('Reshape', { input: sliced }, {
    out: { name: 'shaped', shape: [2, 1, 2], dtype: 'int32' },
  }).out;
  const expanded = graph.addOp('Expand', { input: shaped }, {
    out: { name: 'expanded', shape: [2, 3, 2], dtype: 'int32' },
  }).out;
  const split = graph.addOp('Split', { input: expanded }, {
    out0: { name: 'out0', shape: [2, 1, 2], dtype: 'int32' },
    out1: { name: 'out1', shape: [2, 1, 2], dtype: 'int32' },
    out2: { name: 'out2', shape: [2, 1, 2], dtype: 'int32' },
  }, { axis: 1 });
  graph.setOutputs([split.out0.name, split.out1.name, split.out2.name]);
  return graph;
}

async function execute(graph, inputs) {
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute(inputs);
}

test('CPU Softmax normalizes every last-axis row of a rank-4 tensor', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2, 2, 3]);
  const out = graph.addOp('Softmax', { input }, {
    out: { name: 'out', shape: [1, 2, 2, 3] },
  }, { axis: -1 }).out;
  graph.setOutputs([out.name]);
  const values = Float32Array.of(
    0, 1, 2,
    2, 1, 0,
    -4, -4, -4,
    100, 101, 99,
  );
  const result = await execute(graph, { input: values });
  for (let row = 0; row < 4; row++) {
    const offset = row * 3;
    const maximum = Math.max(...values.slice(offset, offset + 3));
    const exponentials = [...values.slice(offset, offset + 3)]
      .map((value) => Math.exp(value - maximum));
    const sum = exponentials.reduce((left, right) => left + right, 0);
    for (let column = 0; column < 3; column++) {
      assert.ok(Math.abs(result.out[offset + column] - exponentials[column] / sum) < 1e-6);
    }
  }
});

test('CPU LayerNorm preserves high-offset low-variance rows', async () => {
  const dModel = 127;
  const values = new Float32Array(dModel);
  values.fill(10_000);
  values[0] -= 2 ** -10;
  values[dModel - 1] += 2 ** -10;

  const graph = new Graph();
  const input = graph.addInput('input', [1, dModel]);
  const weight = graph.addWeight('weight', [dModel], 'float32', {
    buffer: new Float32Array(dModel).fill(1),
  });
  const bias = graph.addWeight('bias', [dModel], 'float32', {
    buffer: new Float32Array(dModel),
  });
  const out = graph.addOp('LayerNorm', { input, weight, bias }, {
    out: { name: 'out', shape: [1, dModel] },
  }, { d_model: dModel, eps: 1e-12 }).out;
  graph.setOutputs([out.name]);

  const result = await execute(graph, { input: values });
  assert.ok(Math.abs(result.out[0] + 7.9684234) < 1e-5);
  assert.ok(Math.abs(result.out[dModel - 1] - 7.9684234) < 1e-5);
  assert.equal(result.out[1], 0);
});

test('CPU BatchMatMul matches ONNX broadcast-batch matrix semantics', async () => {
  const a = Float32Array.from({ length: 12 }, (_, index) => index - 4);
  const b = Float32Array.from({ length: 12 }, (_, index) => (index % 5) - 2);
  const result = await execute(batchMatMulGraph(), { a, b });
  assert.deepEqual([...result.out], referenceBatchMatMul(a, b));
});

test('CPU I32 Equal/GreaterOrEqual broadcast and Not produce canonical 0/1 storage', async () => {
  const result = await execute(logicalGraph(), {
    a: Int32Array.of(1, 2, 3, 4, 5, 6),
    b: Int32Array.of(2, 5),
  });
  assert.deepEqual([...result.equal], [
    0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0,
  ]);
  assert.deepEqual([...result.greaterOrEqual], [
    0, 1, 1, 0, 0, 0,
    1, 1, 1, 0, 1, 1,
  ]);
  assert.deepEqual([...result.notEqual], [...result.equal].map((value) => 1 - value));
});

test('CPU I32 Clip and exact-shape Where retain integer values', async () => {
  const result = await execute(typedSelectionGraph(), {
    input: Int32Array.of(-9, -2, 0, 5, 6, 2147483647),
    condition: Int32Array.of(1, 0, -2147483648, 0, 1, 1),
  });
  assert.deepEqual([...result.clipped], [-2, -2, 0, 5, 5, 5]);
  assert.deepEqual([...result.selected], [-2, 91, 0, 93, 5, 5]);
});

test('CPU Gather normalizes ONNX negative indices and keeps invalid selections fail-closed', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3, 2]);
  const indices = graph.addInput('indices', [5], 'int32');
  const out = graph.addOp('Gather', { input, indices }, {
    out: { name: 'out', shape: [2, 5, 2] },
  }, { axis: 1 }).out;
  graph.setOutputs([out.name]);

  const result = await execute(graph, {
    input: Float32Array.from({ length: 12 }, (_, index) => index + 1),
    indices: Int32Array.of(-1, -3, -4, 3, 0),
  });
  assert.deepEqual([...result.out], [
    5, 6, 1, 2, -1, -1, -1, -1, 1, 2,
    11, 12, 7, 8, -1, -1, -1, -1, 7, 8,
  ]);
});

test('CPU storage-only shape operators preserve I32 through a complete chain', async () => {
  const result = await execute(typedShapeGraph(), {
    a: Int32Array.of(1, 2, 3, 4),
    b: Int32Array.of(5, 6),
  });
  for (const name of ['out0', 'out1', 'out2']) {
    assert.ok(result[name] instanceof Int32Array);
    assert.deepEqual([...result[name]], [5, 4, 3, 6]);
  }
});

test('BatchMatMul rejects incompatible broadcast geometry instead of flattening it', async () => {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 2, 3]);
  const b = graph.addInput('b', [3, 3, 2]);
  const out = graph.addOp('BatchMatMul', { a, b }, {
    out: { name: 'out', shape: [3, 2, 2] },
  }).out;
  graph.setOutputs([out.name]);
  await assert.rejects(
    execute(graph, {
      a: new Float32Array(12),
      b: new Float32Array(18),
    }),
    /incompatible batch dimensions/,
  );
});
