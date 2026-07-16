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

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

function bytes(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function qEmbeddingGraph({
  ids = [2, 0, 1],
  idsShape = [3],
  weightDtype = 'int8',
  weightValues = [-1, 0, 1, 2, 0, -2, 5, 1, -3],
  weightShape = [3, 3],
  weightQuantization = {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25, 0.125], zero_points: [0, 0, 1],
  },
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
} = {}) {
  const graph = new Graph();
  const input = graph.addInput('ids', idsShape, 'int32');
  const weight = graph.addWeight('table', weightShape, weightDtype, {
    buffer: bytes(weightDtype, weightValues), quantization: weightQuantization,
  });
  const { out } = graph.addOp('QEmbedding', { input, weight }, {
    out: {
      name: 'out', shape: [...idsShape, weightShape[1]], dtype: outputDtype,
      quantization: outputQuantization,
    },
  });
  graph.outputNames = [out.name];
  return { graph, input: Int32Array.from(ids), out };
}

async function cpuResult(spec) {
  const { graph, input } = qEmbeddingGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute({ ids: input });
}

test('portable WASM QEmbedding preserves W8A8 storage and preflights every ID', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qembedding-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qembedding_i8u8, 'function', 'forward WASM exports qembedding_i8u8');
    wasm._cpuQEmbedding = () => { throw new Error('WASM QEmbedding must not use the JS reference'); };

    await t.test('I8 table to I8 output matches the CPU reference', async () => {
      const spec = {};
      const cpu = await cpuResult(spec);
      const { graph, input } = qEmbeddingGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ ids: input });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [1, -1, -3, -3, -1, 1, 1, -1, -3]);
    });

    await t.test('ties-to-even and typed saturation match the CPU reference', async () => {
      const spec = {
        ids: [0, 1], idsShape: [2], weightShape: [2, 3],
        weightValues: [1, 3, -3, 127, -128, 0],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.25, 1], zero_points: [0, 0],
        },
        outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
      };
      const cpu = await cpuResult(spec);
      const { graph, input } = qEmbeddingGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ ids: input });
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 2, -2, 127, -128, 0]);
    });

    await t.test('mixed U8 table and U8 output stay in physical byte storage', async () => {
      const spec = {
        weightDtype: 'uint8', weightValues: [127, 128, 129, 22, 20, 18, 205, 201, 197],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.5, 0.25, 0.125], zero_points: [128, 20, 201],
        },
        outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 100 },
      };
      const cpu = await cpuResult(spec);
      const { graph, input } = qEmbeddingGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ ids: input });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [101, 100, 99, 99, 100, 101, 101, 100, 99]);
    });

    await t.test('cross-signedness I8/U8 table/output combinations match the CPU reference', async () => {
      const variants = [
        {
          weightDtype: 'int8', outputDtype: 'uint8',
          outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 100 },
        },
        {
          weightDtype: 'uint8', weightValues: [127, 128, 129, 22, 20, 18, 205, 201, 197],
          weightQuantization: {
            scheme: 'per_axis', axis: 0, scales: [0.5, 0.25, 0.125], zero_points: [128, 20, 201],
          },
          outputDtype: 'int8',
          outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        },
      ];
      for (const spec of variants) {
        const cpu = await cpuResult(spec);
        const { graph, input } = qEmbeddingGraph(spec);
        wasm.compile(graph);
        const result = await wasm.execute({ ids: input });
        assert.ok(spec.outputDtype === 'int8' ? result.out instanceof Int8Array : result.out instanceof Uint8Array);
        assert.deepEqual([...result.out], [...cpu.out]);
      }
    });

    await t.test('out-of-vocabulary IDs fail before a C kernel writes the output buffer', async () => {
      const { graph, input, out } = qEmbeddingGraph({ ids: [0, 9], idsShape: [2] });
      wasm.compile(graph);
      out.buffer.fill(73);
      const logError = console.error;
      console.error = () => {};
      try {
        await assert.rejects(
          () => wasm.execute({ ids: input }),
          /QEmbedding node .*rejected.*token IDs/,
        );
      } finally {
        console.error = logError;
      }
      assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
    });

    await t.test('compile rejects a table that is not axis-0 per-row quantized', () => {
      const invalid = qEmbeddingGraph({
        weightQuantization: { scheme: 'per_axis', axis: 1, scales: [0.25, 0.25, 0.25], zero_points: [0, 0, 0] },
      });
      assert.throws(() => wasm.compile(invalid.graph), /axis-0 per-output-channel quantization metadata/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
