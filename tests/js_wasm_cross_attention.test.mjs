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

function ramp(length, scale = 0.1, offset = 0) {
  return Float32Array.from({ length }, (_, index) => ((index % 11) - 5) * scale + offset);
}

function crossAttentionGraph({ batch, useAffine }) {
  const graph = new RuntimeGraph();
  const qShape = batch === 1 ? [2, 4] : [batch, 2, 4];
  const kvShape = batch === 1 ? [3, 4] : [batch, 3, 4];
  const outputShape = batch === 1 ? [2, 4] : [batch, 2, 4];
  const q = graph.addInput('q', qShape);
  const kv = graph.addInput('kv', kvShape);
  const weight = graph.addWeight('weight', [12, 4], 'float32', { buffer: ramp(48, 0.07) });
  const inputs = { q, kv, weight };
  if (useAffine) {
    inputs.scale = graph.addWeight('scale', [12], 'float32', { buffer: ramp(12, 0.03, 1) });
    inputs.bias = graph.addWeight('bias', [12], 'float32', { buffer: ramp(12, 0.02) });
  }
  const { out } = graph.addOp('CrossAttention', inputs, {
    out: { name: 'out', shape: outputShape },
  }, { heads: 2 });
  graph.setOutputs([out.name]);
  return graph;
}

async function cpuResult(graph, inputs) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

function assertClose(actual, expected, label, tolerance = 2e-5) {
  assert.equal(actual.length, expected.length, `${label} output length`);
  for (let index = 0; index < actual.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    assert.ok(difference <= tolerance,
      `${label}[${index}] differs by ${difference}; got ${actual[index]}, expected ${expected[index]}`);
  }
}

test('portable WASM CrossAttention matches the CPU reference for rank-2 and batched rank-3 inputs', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      t.skip(`${clang} is unavailable`);
      return;
    }
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-cross-attention-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');

    await t.test('rank-2 Q/KV without affine projections', async () => {
      const inputs = { q: ramp(8, 0.11), kv: ramp(12, 0.09, 0.02) };
      const cpu = await cpuResult(crossAttentionGraph({ batch: 1, useAffine: false }), inputs);
      const graph = crossAttentionGraph({ batch: 1, useAffine: false });
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assertClose(result.out, cpu.out, 'rank-2 CrossAttention');
    });

    await t.test('rank-3 batches with optional scale and bias stay independent', async () => {
      const inputs = { q: ramp(16, 0.11), kv: ramp(24, 0.09, 0.02) };
      const cpu = await cpuResult(crossAttentionGraph({ batch: 2, useAffine: true }), inputs);
      const graph = crossAttentionGraph({ batch: 2, useAffine: true });
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assertClose(result.out, cpu.out, 'batched affine CrossAttention');
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
