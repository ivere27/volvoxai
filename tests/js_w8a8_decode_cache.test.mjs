import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/core/Graph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

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

function decoderGraph() {
  const graph = new Graph();
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

  const q = qlinear(graph, normalized, 'self.q', WIDTH, 2);
  const k = qlinear(graph, normalized, 'self.k', WIDTH, 3);
  const v = qlinear(graph, normalized, 'self.v', WIDTH, 4);
  const attended = graph.addOp('QSDPA', { q, k, v, mask: yKeep }, {
    out: {
      name: 'self.attention', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  hidden = graph.addOp('QAdd', { a: hidden, b: attended }, {
    out: {
      name: 'self.residual', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }).out;

  const crossQ = qlinear(graph, hidden, 'cross.q', WIDTH, 5);
  const crossed = graph.addOp('QSDPA', {
    q: crossQ, k: memoryK, v: memoryV, mask: memoryKeep,
  }, {
    out: {
      name: 'cross.attention', shape: [1, SEQUENCE, WIDTH], dtype: 'int8', quantization: perTensor(),
    },
  }, { heads: 1, causal: false, scale: 0.5 }).out;
  hidden = graph.addOp('QAdd', { a: hidden, b: crossed }, {
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
  graph.outputNames = [tokenIds.name];
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

function tensorRow(tensor, position) {
  const width = tensor.shape.slice(2).reduce((product, dimension) => product * dimension, 1);
  return [...tensor.buffer.subarray(position * width, (position + 1) * width)];
}

function compareCurrentRows(actualGraph, expectedGraph, position) {
  for (const expectedNode of expectedGraph.nodes) {
    const expected = expectedNode.outputs.out || Object.values(expectedNode.outputs || {})[0];
    if (expected?.shape?.[0] !== 1 || expected.shape[1] !== SEQUENCE) continue;
    const actual = actualGraph.tensors.get(expected.name);
    assert.deepEqual(tensorRow(actual, position), tensorRow(expected, position),
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

test('CPU W8A8 incremental decode updates one row and retains self/cross K/V', async () => {
  const graph = decoderGraph();
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  assert.equal(cpu.supportsIncrementalRows, true);
  await verifyRowDecode(cpu, graph, { useDecodeSession: true });
});

test('W8A8 decode row rejects unsupported descendants and invalidates the seed cache', async () => {
  const graph = decoderGraph();
  const source = graph.tensors.get('token_ids');
  const unsupported = graph.addOp('Identity', { input: source }, {
    out: { name: 'unsupported', shape: [1, SEQUENCE], dtype: 'int32' },
  }).out;
  graph.outputNames = [unsupported.name];
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
  }), /unsupported op 'Identity'/);
  assert.equal(cpu._incrementalCacheValid, false);
});

test('compiled C/WASM W8A8 kernels execute row decode without JS operator fallback', {
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
      '_cpuQSiLU', '_cpuQSDPA', '_cpuQArgMax',
    ]) {
      wasm[name] = () => { throw new Error(`WASM row decode called JS fallback ${name}`); };
    }
    await verifyRowDecode(wasm, graph, { useDecodeSession: true });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
