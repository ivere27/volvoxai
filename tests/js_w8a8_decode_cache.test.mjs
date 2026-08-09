import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { quantizedRowNode } from '../ts/backends/quantizedRowExecution.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
const SEQUENCE = 3;
const WIDTH = 4;
const MEMORY = 2;

function perTensor() {
  return { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
}

function byteWeight(graph, name, rows, columns, seed = 0) {
  const values = new Int8Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      values[row * columns + column] = ((row * 3 + column * 5 + seed) % 9) - 4;
    }
  }
  return graph.addWeight(name, [rows, columns], 'int8', {
    buffer: values,
    quantization: {
      scheme: 'per_axis', axis: 0,
      scales: new Array(rows).fill(0.125), zero_points: new Array(rows).fill(0),
    },
  });
}

function qlinear(graph, input, name, outputWidth, seed) {
  const inputWidth = input.shape.at(-1);
  const weight = byteWeight(graph, `${name}.weight`, outputWidth, inputWidth, seed);
  const bias = graph.addWeight(`${name}.bias`, [outputWidth], 'int32', {
    buffer: new Int32Array(outputWidth),
  });
  return graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name, shape: [1, SEQUENCE, outputWidth], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
}

function floatWeight(graph, name, shape, seed = 0, scale = 0.04) {
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return graph.addWeight(name, shape, 'float32', {
    buffer: Float32Array.from({ length: elements }, (_, index) =>
      ((((index * 7) + seed * 5) % 17) - 8) * scale),
  });
}

function floatLinear(graph, input, name, outputWidth, seed, opType = 'Linear') {
  const inputWidth = input.shape.at(-1);
  const weight = floatWeight(graph, `${name}.weight`, [outputWidth, inputWidth], seed);
  const bias = floatWeight(graph, `${name}.bias`, [outputWidth], seed + 13, 0.01);
  return graph.addOp(opType, { input, weight, bias }, {
    out: { name, shape: [1, SEQUENCE, outputWidth], dtype: 'float32' },
  }, { weight_layout: 'OUT_IN' }).out;
}

function reshapeActivation(graph, input, name, shape) {
  const descriptor = { name, shape, dtype: input.dtype };
  if (input.quantization) descriptor.quantization = input.quantization;
  return graph.addOp('Reshape', { input }, { out: descriptor }, {}).out;
}

function byteBatchAdapter(graph, input, name) {
  const downWeight = graph.addWeight(`${name}.down.weight`, [1, WIDTH, 2], 'int8', {
    buffer: Int8Array.from({ length: WIDTH * 2 }, (_, index) => (index % 5) - 2),
    quantization: perTensor(),
  });
  const down = graph.addOp('QBatchMatMul', { a: input, b: downWeight }, {
    out: {
      name: `${name}.down`, shape: [1, SEQUENCE, 2], dtype: 'int8',
      quantization: perTensor(),
    },
  }, {}).out;
  const upWeight = graph.addWeight(`${name}.up.weight`, [1, 2, WIDTH], 'int8', {
    buffer: Int8Array.from({ length: 2 * WIDTH }, (_, index) => 2 - (index % 5)),
    quantization: perTensor(),
  });
  return graph.addOp('QBatchMatMul', { a: down, b: upWeight }, {
    out: {
      name, shape: [1, SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }, {}).out;
}

function floatBatchAdapter(graph, input, name) {
  const downWeight = floatWeight(graph, `${name}.down.weight`, [1, WIDTH, 2], 17, 0.03);
  const down = graph.addOp('BatchMatMul', { a: input, b: downWeight }, {
    out: { name: `${name}.down`, shape: [1, SEQUENCE, 2], dtype: 'float32' },
  }, {}).out;
  const upWeight = floatWeight(graph, `${name}.up.weight`, [1, 2, WIDTH], 18, 0.03);
  return graph.addOp('BatchMatMul', { a: down, b: upWeight }, {
    out: { name, shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }, {}).out;
}

function decoderGraph() {
  const graph = new RuntimeGraph();
  const yIds = graph.addInput('y_ids', [1, SEQUENCE], 'int32');
  const yKeep = graph.addInput('y_keep', [1, SEQUENCE], 'int32');
  const memoryK = graph.addInput('memory_k', [1, MEMORY, WIDTH], 'int8', { quantization: perTensor() });
  const memoryV = graph.addInput('memory_v', [1, MEMORY, WIDTH], 'int8', { quantization: perTensor() });
  const memoryKeep = graph.addInput('memory_keep', [1, MEMORY], 'int32');

  const embeddingWeight = byteWeight(graph, 'embedding.weight', 7, WIDTH, 1);
  let hidden = graph.addOp('QEmbedding', { input: yIds, weight: embeddingWeight }, {
    out: {
      name: 'embed', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
  const position = graph.addWeight('position', [1, SEQUENCE, WIDTH], 'int8', {
    buffer: Int8Array.from({ length: SEQUENCE * WIDTH }, (_, index) => (index % 5) - 2),
    quantization: perTensor(),
  });
  hidden = graph.addOp('QAdd', { a: hidden, b: position }, {
    out: {
      name: 'positioned', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;

  const gamma = graph.addWeight('norm.weight', [WIDTH], 'float32', {
    buffer: Float32Array.of(1, 0.75, 1.25, 0.5),
  });
  const beta = graph.addWeight('norm.bias', [WIDTH], 'float32', {
    buffer: Float32Array.of(0.125, -0.125, 0.25, 0),
  });
  const normalized = graph.addOp('QLayerNorm', { input: hidden, weight: gamma, bias: beta }, {
    out: {
      name: 'normalized', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }, { eps: 1e-5, d_model: WIDTH }).out;

  const qProjected = qlinear(graph, normalized, 'self.q.projected', WIDTH, 2);
  const qSingleton = reshapeActivation(
    graph, qProjected, 'self.q.singleton', [SEQUENCE, 1, WIDTH],
  );
  const q = reshapeActivation(graph, qSingleton, 'self.q', [SEQUENCE, WIDTH]);
  const k = reshapeActivation(
    graph, qlinear(graph, normalized, 'self.k.projected', WIDTH, 3),
    'self.k', [SEQUENCE, WIDTH],
  );
  const v = reshapeActivation(
    graph, qlinear(graph, normalized, 'self.v.projected', WIDTH, 4),
    'self.v', [SEQUENCE, WIDTH],
  );
  const attended = graph.addOp('QSDPA', { q, k, v, mask: yKeep }, {
    out: {
      name: 'self.attention.2d', shape: [SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  const attended3d = reshapeActivation(
    graph, attended, 'self.attention', [1, SEQUENCE, WIDTH],
  );
  hidden = graph.addOp('QAdd', { a: hidden, b: attended3d }, {
    out: {
      name: 'self.residual', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
  hidden = byteBatchAdapter(graph, hidden, 'adapter.output');

  const memoryK2d = reshapeActivation(graph, memoryK, 'memory.k.2d', [MEMORY, WIDTH]);
  const memoryV2d = reshapeActivation(graph, memoryV, 'memory.v.2d', [MEMORY, WIDTH]);
  const crossQ = reshapeActivation(
    graph, qlinear(graph, hidden, 'cross.q.projected', WIDTH, 5),
    'cross.q', [SEQUENCE, WIDTH],
  );
  const crossed = graph.addOp('QSDPA', {
    q: crossQ, k: memoryK2d, v: memoryV2d, mask: memoryKeep,
  }, {
    out: {
      name: 'cross.attention.2d', shape: [SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }, { heads: 1, causal: false, scale: 0.5 }).out;
  const crossed3d = reshapeActivation(
    graph, crossed, 'cross.attention', [1, SEQUENCE, WIDTH],
  );
  hidden = graph.addOp('QAdd', { a: hidden, b: crossed3d }, {
    out: {
      name: 'cross.residual', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;

  hidden = qlinear(graph, hidden, 'ffn.linear', WIDTH, 6);
  hidden = graph.addOp('QGELU', { input: hidden }, {
    out: {
      name: 'ffn.gelu', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
  hidden = graph.addOp('QSiLU', { input: hidden }, {
    out: {
      name: 'ffn.silu', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;
  const logits = qlinear(graph, hidden, 'logits', 7, 7);
  const tokenIds = graph.addOp('QArgMax', { input: logits }, {
    out: { name: 'token_ids', shape: [1, SEQUENCE], dtype: 'int32' },
  }, { axis: -1 }).out;
  graph.setOutputs([tokenIds.name]);
  return graph;
}

function mixedDecoderGraph() {
  const graph = new RuntimeGraph();
  const yIds = graph.addInput('y_ids', [1, SEQUENCE], 'int32');
  const yKeep = graph.addInput('y_keep', [1, SEQUENCE], 'int32');
  const memoryK = graph.addInput('memory_k', [1, MEMORY, WIDTH], 'float32');
  const memoryV = graph.addInput('memory_v', [1, MEMORY, WIDTH], 'float32');
  const memoryKeep = graph.addInput('memory_keep', [1, MEMORY], 'int32');

  const embeddingWeight = floatWeight(graph, 'mixed.embedding.weight', [7, WIDTH], 1, 0.08);
  let hidden = graph.addOp('Embedding', { input: yIds, weight: embeddingWeight }, {
    out: { name: 'mixed.embed', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  const position = floatWeight(graph, 'mixed.position', [1, SEQUENCE, WIDTH], 2, 0.025);
  hidden = graph.addOp('Add', { a: hidden, b: position }, {
    out: { name: 'mixed.positioned', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  // This rank-one operand must remain unsliced and broadcast over the local
  // [1,1,D] row. WIDTH intentionally differs from SEQUENCE: right alignment,
  // not a coincidental dimension value, identifies features.
  const featureBias = floatWeight(graph, 'mixed.feature_bias', [WIDTH], 3, 0.015);
  hidden = graph.addOp('Add', { a: hidden, b: featureBias }, {
    out: { name: 'mixed.biased', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;

  const gamma = graph.addWeight('mixed.norm.weight', [WIDTH], 'float32', {
    buffer: Float32Array.of(1, 0.75, 1.25, 0.5),
  });
  const beta = graph.addWeight('mixed.norm.bias', [WIDTH], 'float32', {
    buffer: Float32Array.of(0.125, -0.125, 0.25, 0),
  });
  const normalized = graph.addOp('LayerNorm', { input: hidden, weight: gamma, bias: beta }, {
    out: { name: 'mixed.normalized', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { eps: 1e-5, d_model: WIDTH }).out;

  const qProjected = floatLinear(
    graph, normalized, 'mixed.self.q.projected', WIDTH, 4, 'Linear',
  );
  const qSingleton = reshapeActivation(
    graph, qProjected, 'mixed.self.q.singleton', [SEQUENCE, 1, WIDTH],
  );
  const q2d = reshapeActivation(
    graph, qSingleton, 'mixed.self.q.2d', [SEQUENCE, WIDTH],
  );
  const queryScale = graph.addWeight('mixed.self.q.scale', [1], 'float32', {
    buffer: Float32Array.of(0.75),
  });
  const q = graph.addOp('Mul', { a: q2d, b: queryScale }, {
    out: { name: 'mixed.self.q', shape: [SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  const k = reshapeActivation(
    graph, floatLinear(graph, normalized, 'mixed.self.k.projected', WIDTH, 5, 'MatMul'),
    'mixed.self.k', [SEQUENCE, WIDTH],
  );
  const v = reshapeActivation(
    graph, floatLinear(graph, normalized, 'mixed.self.v.projected', WIDTH, 6, 'Gemm'),
    'mixed.self.v', [SEQUENCE, WIDTH],
  );
  const attended = graph.addOp('CrossSDPA', { q, k, v, mask: yKeep }, {
    out: { name: 'mixed.self.attention.2d', shape: [SEQUENCE, WIDTH], dtype: 'float32' },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  const attended3d = reshapeActivation(
    graph, attended, 'mixed.self.attention', [1, SEQUENCE, WIDTH],
  );
  hidden = graph.addOp('Add', { a: hidden, b: attended3d }, {
    out: { name: 'mixed.self.residual', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  hidden = floatBatchAdapter(graph, hidden, 'mixed.adapter.output');

  const memoryK2d = reshapeActivation(
    graph, memoryK, 'mixed.memory.k.2d', [MEMORY, WIDTH],
  );
  const memoryV2d = reshapeActivation(
    graph, memoryV, 'mixed.memory.v.2d', [MEMORY, WIDTH],
  );
  const crossQ = reshapeActivation(
    graph, floatLinear(graph, hidden, 'mixed.cross.q.projected', WIDTH, 7),
    'mixed.cross.q', [SEQUENCE, WIDTH],
  );
  const crossed = graph.addOp('CrossSDPA', {
    q: crossQ, k: memoryK2d, v: memoryV2d, mask: memoryKeep,
  }, {
    out: { name: 'mixed.cross.attention.2d', shape: [SEQUENCE, WIDTH], dtype: 'float32' },
  }, { heads: 1, causal: false, scale: 0.5 }).out;
  const crossed3d = reshapeActivation(
    graph, crossed, 'mixed.cross.attention', [1, SEQUENCE, WIDTH],
  );
  hidden = graph.addOp('Add', { a: hidden, b: crossed3d }, {
    out: { name: 'mixed.cross.residual', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;

  hidden = floatLinear(graph, hidden, 'mixed.ffn.linear', WIDTH, 8);
  hidden = graph.addOp('GELU', { input: hidden }, {
    out: { name: 'mixed.ffn.gelu', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }, { approximate: 'none' }).out;
  hidden = graph.addOp('SiLU', { input: hidden }, {
    out: { name: 'mixed.ffn.silu', shape: [1, SEQUENCE, WIDTH], dtype: 'float32' },
  }).out;
  const logits = floatLinear(graph, hidden, 'mixed.logits', 7, 9);
  const scale = graph.addWeight('mixed.logits.scale', [1], 'float32', {
    buffer: Float32Array.of(0.03125),
  });
  const zeroPoint = graph.addWeight('mixed.logits.zero_point', [1], 'int8', {
    buffer: Int8Array.of(0),
  });
  const quantized = graph.addOp('QuantizeLinear', {
    input: logits, scale, zero_point: zeroPoint,
  }, {
    out: {
      name: 'mixed.logits.quantized', shape: [1, SEQUENCE, 7], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 0 },
    },
  }).out;
  const tokenIds = graph.addOp('QArgMax', { input: quantized }, {
    out: { name: 'token_ids', shape: [1, SEQUENCE], dtype: 'int32' },
  }, { axis: -1 }).out;
  graph.setOutputs([tokenIds.name]);
  return graph;
}

function decoderInputs(yIds, yKeep) {
  return {
    y_ids: yIds,
    y_keep: yKeep,
    memory_k: Int8Array.of(3, -2, 1, 4, -1, 2, -3, 1),
    memory_v: Int8Array.of(4, 1, -2, 3, -3, 2, 4, -1),
    memory_keep: Int32Array.of(1, 1),
  };
}

function mixedDecoderInputs(yIds, yKeep) {
  return {
    y_ids: yIds,
    y_keep: yKeep,
    memory_k: Float32Array.of(0.25, -0.5, 0.75, 0.125, -0.25, 0.5, -0.75, 0.375),
    memory_v: Float32Array.of(0.5, 0.125, -0.25, 0.75, -0.375, 0.25, 0.625, -0.5),
    memory_keep: Int32Array.of(1, 1),
  };
}

function sequenceRowWidth(tensor) {
  if (tensor?.shape?.length >= 2 && tensor.shape[0] === 1 &&
      tensor.shape[1] === SEQUENCE) {
    return tensor.shape.slice(2).reduce((product, dimension) => product * dimension, 1);
  }
  if ((tensor?.shape?.length === 2 ||
      (tensor?.shape?.length === 3 && tensor.shape[1] === 1)) &&
      tensor.shape[0] === SEQUENCE) {
    return tensor.shape.slice(1).reduce((product, dimension) => product * dimension, 1);
  }
  return null;
}

function tensorRow(tensor, position) {
  const width = sequenceRowWidth(tensor);
  assert.ok(Number.isInteger(width) && width > 0, `${tensor?.name} logical row width`);
  return [...tensor.buffer.subarray(position * width, (position + 1) * width)];
}

function compareCurrentRows(actualGraph, expectedGraph, position) {
  for (const expectedNode of expectedGraph.nodes) {
    const expected = expectedNode.outputs.out || Object.values(expectedNode.outputs || {})[0];
    if (sequenceRowWidth(expected) == null) continue;
    const actual = actualGraph.tensors.get(expected.name);
    assert.deepEqual(tensorRow(actual, position), tensorRow(expected, position),
      `${expectedNode.opType} ${expectedNode.id} row ${position}`);
  }
}

function assertRowsClose(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label} row width`);
  for (let index = 0; index < actual.length; index++) {
    if (Number.isInteger(expected[index]) && Number.isInteger(actual[index])) {
      assert.equal(actual[index], expected[index], `${label}[${index}]`);
      continue;
    }
    const tolerance = 5e-4 + 5e-4 * Math.abs(expected[index]);
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `${label}[${index}] expected ${expected[index]}, received ${actual[index]}`);
  }
}

function compareMixedCurrentRows(actualGraph, expectedGraph, position) {
  for (const expectedNode of expectedGraph.nodes) {
    const expected = expectedNode.outputs.out || Object.values(expectedNode.outputs || {})[0];
    if (sequenceRowWidth(expected) == null) continue;
    const actual = actualGraph.tensors.get(expected.name);
    assertRowsClose(tensorRow(actual, position), tensorRow(expected, position),
      `${expectedNode.opType} ${expectedNode.id} row ${position}`);
  }
}

async function verifyRowDecode(engine, graph, { useDecodeSession = false } = {}) {
  const referenceGraph = decoderGraph();
  const reference = new CPUEngine();
  reference.allocateGraph(referenceGraph);
  const yIds = Int32Array.of(1, 0, 0);
  const yKeep = Int32Array.of(1, 0, 0);
  const names = Object.keys(decoderInputs(yIds, yKeep));

  const decode = useDecodeSession
    ? engine.createDecodeSession({ changedInputs: ['y_ids', 'y_keep'], rowMode: 'required' })
    : null;
  try {
    const seedInputs = decoderInputs(yIds, yKeep);
    const expectedSeed = await reference.execute(seedInputs);
    const actualSeed = decode
      ? await decode.seed(seedInputs)
      : await engine.execute(seedInputs, {
          incremental: true, incrementalReset: true, changedInputs: names,
        });
    assert.equal(actualSeed.token_ids[0], expectedSeed.token_ids[0]);

    for (const [position, id] of [[1, 3], [2, 5]]) {
      yIds[position] = id;
      yKeep[position] = 1;
      const inputs = decoderInputs(yIds, yKeep);
      const expected = await reference.execute(inputs);
      const actual = decode
        ? await decode.step(inputs, { position })
        : await engine.execute(inputs, {
            incremental: true,
            changedInputs: ['y_ids', 'y_keep'],
            incrementalRowPosition: position,
          });
      assert.equal(actual.token_ids[position], expected.token_ids[position]);
      compareCurrentRows(graph, referenceGraph, position);
      // These are the physical self-attention K/V cache rows. Every completed
      // prefix row must remain identical to a full recompute.
      for (const name of ['self.k', 'self.v']) {
        const cached = graph.tensors.get(name);
        const full = referenceGraph.tensors.get(name);
        for (let row = 0; row <= position; row++) {
          assert.deepEqual(tensorRow(cached, row), tensorRow(full, row), `${name} cache row ${row}`);
        }
      }
    }
  } finally {
    await decode?.close();
  }
}

async function verifyMixedRowDecode(engine, graph, { referenceEngine = null } = {}) {
  const referenceGraph = mixedDecoderGraph();
  const reference = referenceEngine || new CPUEngine();
  reference.allocateGraph(referenceGraph);
  const yIds = Int32Array.of(1, 0, 0);
  const yKeep = Int32Array.of(1, 0, 0);
  const names = Object.keys(mixedDecoderInputs(yIds, yKeep));
  const decode = engine.createDecodeSession({
    changedInputs: ['y_ids', 'y_keep'], rowMode: 'required',
  });
  try {
    const seedInputs = mixedDecoderInputs(yIds, yKeep);
    const expectedSeed = await reference.execute(seedInputs);
    const actualSeed = await decode.seed(seedInputs);
    assert.equal(actualSeed.token_ids[0], expectedSeed.token_ids[0]);
    const untouchedFutureEmbedding = tensorRow(graph.tensors.get('mixed.embed'), 2);

    for (const [position, id] of [[1, 3], [2, 6]]) {
      yIds[position] = id;
      yKeep[position] = 1;
      if (position === 1) yIds[2] = 5;
      const inputs = mixedDecoderInputs(yIds, yKeep);
      const expected = await reference.execute(inputs);
      const actual = await decode.step(inputs, { position });
      assert.equal(actual.token_ids[position], expected.token_ids[position]);
      compareMixedCurrentRows(graph, referenceGraph, position);
      if (position === 1) {
        // A full-sequence Embedding call would observe the deliberately changed
        // future ID. The incremental contract must leave that row untouched.
        assert.deepEqual(tensorRow(graph.tensors.get('mixed.embed'), 2), untouchedFutureEmbedding);
      }
      for (const name of ['mixed.self.k', 'mixed.self.v']) {
        const cached = graph.tensors.get(name);
        const full = referenceGraph.tensors.get(name);
        for (let row = 0; row <= position; row++) {
          assertRowsClose(tensorRow(cached, row), tensorRow(full, row), `${name} cache row ${row}`);
        }
      }
    }
  } finally {
    await decode.close();
  }
}

test('CPU W8A8 incremental decode updates one row and retains self/cross K/V', async () => {
  const graph = decoderGraph();
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  assert.equal(cpu.capabilities.incrementalRows, true);
  await verifyRowDecode(cpu, graph, { useDecodeSession: true });
});

test('CPU mixed-precision incremental decode slices float rows and safe Add broadcasts', async () => {
  const graph = mixedDecoderGraph();
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  await verifyMixedRowDecode(cpu, graph);
});

test('mixed-precision row Add fails closed on a non-broadcast operand', () => {
  const graph = new RuntimeGraph();
  const a = graph.addInput('a', [1, SEQUENCE, WIDTH], 'float32', {
    buffer: new Float32Array(SEQUENCE * WIDTH),
  });
  const b = graph.addWeight('b', [2, WIDTH], 'float32', {
    buffer: new Float32Array(2 * WIDTH),
  });
  graph.addOp('Add', { a, b }, {
    out: {
      name: 'out', shape: [1, SEQUENCE, WIDTH], dtype: 'float32',
      buffer: new Float32Array(SEQUENCE * WIDTH),
    },
  });
  assert.throws(() => quantizedRowNode(graph.nodes[0], 1),
    /input b does not broadcast to the fixed-sequence Add output/);
});

test('byte Expand row keeps an invariant [1,1,D] source and slices only output', () => {
  const graph = new RuntimeGraph();
  const source = graph.addWeight('source', [1, 1, WIDTH], 'int8', {
    buffer: Int8Array.of(-3, -2, -1, 0), quantization: perTensor(),
  });
  graph.addOp('Expand', { input: source }, {
    out: {
      name: 'expanded', shape: [1, SEQUENCE, WIDTH], dtype: 'int8',
      buffer: new Int8Array(SEQUENCE * WIDTH), quantization: perTensor(),
    },
  });
  const row = quantizedRowNode(graph.nodes[0], 2);
  assert.deepEqual(row.inputs.input.shape, [1, 1, WIDTH]);
  assert.equal(row.inputs.input.buffer, source.buffer);
  assert.deepEqual(row.outputs.out.shape, [1, 1, WIDTH]);
  assert.equal(row.outputs.out.buffer.length, WIDTH);
  assert.equal(row.outputs.out.buffer.byteOffset,
    graph.nodes[0].outputs.out.buffer.byteOffset + 2 * WIDTH);
});

test('Reshape row accepts only layout-preserving contiguous sequence views', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, SEQUENCE, WIDTH], 'float32', {
    buffer: Float32Array.from({ length: SEQUENCE * WIDTH }, (_, index) => index),
  });
  graph.addOp('Reshape', { input }, {
    out: {
      name: 'valid', shape: [SEQUENCE, 1, WIDTH], dtype: 'float32',
      buffer: new Float32Array(SEQUENCE * WIDTH),
    },
  }, {});
  const row = quantizedRowNode(graph.nodes[0], 2);
  assert.deepEqual(row.inputs.input.shape, [1, 1, WIDTH]);
  assert.deepEqual(row.outputs.out.shape, [1, 1, WIDTH]);
  assert.equal(row.inputs.input.buffer.byteOffset,
    input.buffer.byteOffset + 2 * WIDTH * Float32Array.BYTES_PER_ELEMENT);

  const invalid = new RuntimeGraph();
  const invalidInput = invalid.addInput('input', [1, SEQUENCE, WIDTH], 'float32', {
    buffer: new Float32Array(SEQUENCE * WIDTH),
  });
  invalid.addOp('Reshape', { input: invalidInput }, {
    out: {
      name: 'invalid', shape: [1, SEQUENCE, 2, 2], dtype: 'float32',
      buffer: new Float32Array(SEQUENCE * WIDTH),
    },
  }, {});
  assert.throws(() => quantizedRowNode(invalid.nodes[0], 1),
    /output must use \[1,3,D\], \[3,1,D\], or \[3,D\] storage/);
});

for (const opType of ['Add', 'Mul']) {
  test(`${opType} row rejects a dirty broadcast scalar before input or output writes`, async () => {
    const graph = new RuntimeGraph();
    const activation = graph.addInput('activation', [SEQUENCE, WIDTH], 'float32');
    const scale = graph.addInput('scale', [1], 'float32');
    const output = graph.addOp(opType, { a: activation, b: scale }, {
      out: { name: 'out', shape: [SEQUENCE, WIDTH], dtype: 'float32' },
    }).out;
    graph.setOutputs([output.name]);
    const cpu = new CPUEngine();
    cpu.allocateGraph(graph);
    const activationValues = Float32Array.from(
      { length: SEQUENCE * WIDTH }, (_, index) => index - 3,
    );
    await cpu.execute({ activation: activationValues, scale: Float32Array.of(0.5) }, {
      incremental: true, incrementalReset: true,
      changedInputs: ['activation', 'scale'],
    });
    const scaleBefore = [...graph.tensors.get('scale').buffer];
    const outputBefore = [...graph.tensors.get('out').buffer];
    await assert.rejects(cpu.execute({
      activation: activationValues, scale: Float32Array.of(0.75),
    }, {
      incremental: true, changedInputs: ['scale'], incrementalRowPosition: 1,
    }), /broadcast input b must remain invariant during incremental row execution/);
    assert.deepEqual([...graph.tensors.get('scale').buffer], scaleBefore);
    assert.deepEqual([...graph.tensors.get('out').buffer], outputBefore);
    assert.equal(cpu._incrementalCacheValid, false);
  });
}

test('Expand row rejects a dirty broadcast source before input or output writes', async () => {
  const graph = new RuntimeGraph();
  const source = graph.addInput('source', [1, 1, WIDTH], 'int8', {
    quantization: perTensor(),
  });
  const output = graph.addOp('Expand', { input: source }, {
    out: {
      name: 'expanded', shape: [1, SEQUENCE, WIDTH], dtype: 'int8',
      quantization: perTensor(),
    },
  }).out;
  graph.setOutputs([output.name]);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  const seed = Int8Array.of(-3, -2, -1, 0);
  await cpu.execute({ source: seed }, {
    incremental: true, incrementalReset: true, changedInputs: ['source'],
  });
  const inputBefore = [...graph.tensors.get('source').buffer];
  const outputBefore = [...graph.tensors.get('expanded').buffer];
  await assert.rejects(cpu.execute({ source: Int8Array.of(0, 1, 2, 3) }, {
    incremental: true, changedInputs: ['source'], incrementalRowPosition: 1,
  }), /broadcast input must remain invariant during incremental row execution/);
  assert.deepEqual([...graph.tensors.get('source').buffer], inputBefore);
  assert.deepEqual([...graph.tensors.get('expanded').buffer], outputBefore);
  assert.equal(cpu._incrementalCacheValid, false);
});

for (const [opType, dtype, TypedArray] of [
  ['BatchMatMul', 'float32', Float32Array],
  ['QBatchMatMul', 'int8', Int8Array],
]) {
  test(`${opType} row rejects a dirty right-hand matrix before input or output writes`, async () => {
    const graph = new RuntimeGraph();
    const tensorOptions = dtype === 'int8' ? { quantization: perTensor() } : {};
    const a = graph.addInput('a', [1, SEQUENCE, 2], dtype, tensorOptions);
    const b = graph.addInput('b', [1, 2, WIDTH], dtype, tensorOptions);
    const out = graph.addOp(opType, { a, b }, {
      out: {
        name: 'out', shape: [1, SEQUENCE, WIDTH], dtype,
        ...tensorOptions,
      },
    }, {}).out;
    graph.setOutputs([out.name]);
    const cpu = new CPUEngine();
    cpu.allocateGraph(graph);
    const seedA = TypedArray.from({ length: SEQUENCE * 2 }, (_, index) => (index % 5) - 2);
    const seedB = TypedArray.from({ length: 2 * WIDTH }, (_, index) => (index % 3) - 1);
    await cpu.execute({ a: seedA, b: seedB }, {
      incremental: true, incrementalReset: true, changedInputs: ['a', 'b'],
    });
    const inputBefore = [...graph.tensors.get('b').buffer];
    const outputBefore = [...graph.tensors.get('out').buffer];
    const changedB = TypedArray.from(seedB, (value) => value + 1);
    await assert.rejects(cpu.execute({ a: seedA, b: changedB }, {
      incremental: true, changedInputs: ['b'], incrementalRowPosition: 1,
    }), /input b must remain invariant during incremental row execution/);
    assert.deepEqual([...graph.tensors.get('b').buffer], inputBefore);
    assert.deepEqual([...graph.tensors.get('out').buffer], outputBefore);
    assert.equal(cpu._incrementalCacheValid, false);
  });
}

for (const [inputName, label, replacement] of [
  ['memory_k', 'k', (values) => Int8Array.from(values, (value) => value + 1)],
  ['memory_v', 'v', (values) => Int8Array.from(values, (value) => value - 1)],
  ['memory_keep', 'mask', () => Int32Array.of(1, 0)],
]) {
  test(`noncausal attention rejects dirty ${label} before copying changed memory`, async () => {
    const graph = decoderGraph();
    const cpu = new CPUEngine();
    cpu.allocateGraph(graph);
    const yIds = Int32Array.of(1, 0, 0);
    const yKeep = Int32Array.of(1, 0, 0);
    const seed = decoderInputs(yIds, yKeep);
    await cpu.execute(seed, {
      incremental: true, incrementalReset: true, changedInputs: Object.keys(seed),
    });
    const inputBefore = [...graph.tensors.get(inputName).buffer];
    const outputBefore = [...graph.tensors.get('cross.attention.2d').buffer];
    const changed = decoderInputs(yIds, yKeep);
    changed[inputName] = replacement(changed[inputName]);
    await assert.rejects(cpu.execute(changed, {
      incremental: true, changedInputs: [inputName], incrementalRowPosition: 1,
    }), new RegExp(
      `noncausal input ${label} must remain invariant during incremental row execution`,
    ));
    assert.deepEqual([...graph.tensors.get(inputName).buffer], inputBefore);
    assert.deepEqual([...graph.tensors.get('cross.attention.2d').buffer], outputBefore);
    assert.equal(cpu._incrementalCacheValid, false);
  });
}

test('W8A8 decode row rejects unsupported descendants and invalidates the seed cache', async () => {
  const graph = decoderGraph();
  const source = graph.tensors.get('token_ids');
  // Softmax, not Identity: Identity is a re-view and is now row-local, so it
  // no longer stands in for an operator without a row proof. Softmax genuinely
  // has none — a row cannot be normalized without seeing the others.
  const unsupported = graph.addOp('Softmax', { input: source }, {
    out: { name: 'unsupported', shape: [1, SEQUENCE], dtype: 'int32' },
  }).out;
  graph.setOutputs([unsupported.name]);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  const yIds = Int32Array.of(1, 0, 0);
  const yKeep = Int32Array.of(1, 0, 0);
  const inputs = decoderInputs(yIds, yKeep);
  await cpu.execute(inputs, {
    incremental: true, incrementalReset: true, changedInputs: Object.keys(inputs),
  });
  yIds[1] = 3;
  yKeep[1] = 1;
  await assert.rejects(cpu.execute(decoderInputs(yIds, yKeep), {
    incremental: true, changedInputs: ['y_ids', 'y_keep'], incrementalRowPosition: 1,
  }), /unsupported op 'Softmax'/);
  assert.equal(cpu._incrementalCacheValid, false);
});

test('compiled C/WASM quantized and mixed kernels execute row decode without JS fallback', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-decode-cache-'));
  try {
    const output = join(directory, 'volvoxai.wasm');
    await run(clang, [
      '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
      '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
      '-o', output, 'native/src/kernels/kernels.c',
    ], { cwd: repositoryRoot });
    const wasm = await WasmEngine.init(output);
    assert.ok(wasm);
    const graph = decoderGraph();
    wasm.compile(graph);
    for (const name of [
      '_cpuQEmbedding', '_cpuQLinear', '_cpuQAdd', '_cpuQLayerNorm', '_cpuQGELU',
      '_cpuQSiLU', '_cpuQSDPA', '_cpuQArgMax', '_cpuQBatchMatMul', '_cpuReshape',
    ]) {
      wasm[name] = () => { throw new Error(`WASM row decode called JS fallback ${name}`); };
    }
    await verifyRowDecode(wasm, graph, { useDecodeSession: true });

    const mixedGraph = mixedDecoderGraph();
    wasm.compile(mixedGraph);
    for (const name of [
      '_cpuEmbedding', '_cpuAdd', '_cpuLayerNorm', '_cpuMatMul', '_cpuCrossSDPA',
      '_cpuGELU', '_cpuSiLU', '_cpuMul', '_cpuBatchMatMul', '_cpuReshape',
      '_cpuQuantizeLinear', '_cpuQArgMax',
    ]) {
      wasm[name] = () => { throw new Error(`WASM mixed row decode called JS fallback ${name}`); };
    }
    const mixedReference = await wasm.fork();
    await verifyMixedRowDecode(wasm, mixedGraph, { referenceEngine: mixedReference });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
