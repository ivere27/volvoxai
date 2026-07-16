import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/index.js';
import { TrainingVolvoxAI } from '../ts/training/TrainingVolvoxAI.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

function closeArray(actual, expected, label, tolerance = 5e-5) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

async function buildFullWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output,
    'native/src/kernels/kernels.c', 'native/src/kernels/training_kernels.c',
  ], { cwd: repositoryRoot });
}

function values(length, stride, offset = 0) {
  return Float32Array.from({ length }, (_, index) => (((index * stride) % 19) - 9) / 17 + offset);
}

function crossAttentionGraph({
  rank3,
  scale: useScale,
  bias: useBias,
  sharedQKV = false,
  dModel = 4,
  heads = 2,
}) {
  const graph = new Graph();
  const batch = rank3 ? 2 : 1;
  const seqQ = 2;
  const seqKV = sharedQKV ? seqQ : 3;
  const qShape = rank3 ? [batch, seqQ, dModel] : [seqQ, dModel];
  const kvShape = rank3 ? [batch, seqKV, dModel] : [seqKV, dModel];
  const q = graph.addWeight('q', qShape, 'float32', {
    buffer: values(batch * seqQ * dModel, 3),
  });
  const kv = sharedQKV ? q : graph.addWeight('kv', kvShape, 'float32', {
    buffer: values(batch * seqKV * dModel, 5, 0.04),
  });
  const weight = graph.addWeight('weight', [3 * dModel, dModel], 'float32', {
    buffer: values(3 * dModel * dModel, 7, 0.02),
  });
  const inputs = { q, kv, weight };
  if (useScale) {
    inputs.scale = graph.addWeight('scale', [3 * dModel], 'float32', {
      buffer: values(3 * dModel, 2, 1),
    });
  }
  if (useBias) {
    inputs.bias = graph.addWeight('bias', [3 * dModel], 'float32', {
      buffer: values(3 * dModel, 11, 0.01),
    });
  }
  const { out } = graph.addOp('CrossAttention', inputs, {
    out: rank3 ? [batch, seqQ, dModel] : [seqQ, dModel],
  }, { heads });
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    targetCount: batch * seqQ,
    classes: dModel,
    trainable: ['q', ...(sharedQKV ? [] : ['kv']), 'weight', ...(useScale ? ['scale'] : []), ...(useBias ? ['bias'] : [])],
  };
}

async function trainParity(api, makeGraph, label) {
  const wasmCase = makeGraph();
  const cpuCase = makeGraph();
  const options = {
    targets: Array.from({ length: wasmCase.targetCount }, (_, index) => index % wasmCase.classes),
    trainableTensors: wasmCase.trainable,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  };
  const wasm = await api.trainStep(wasmCase.graph, { ...options, backend: 'wasm' });
  const cpu = await CPUAutograd.trainStep(cpuCase.graph, options);
  assert.equal(wasm.backend, 'wasm');
  assert.ok(Math.abs(wasm.loss - cpu.loss) <= 5e-5, `${label} loss`);
  closeArray(wasmCase.graph.getTensor(wasmCase.output).buffer,
    cpuCase.graph.getTensor(cpuCase.output).buffer, `${label} output`);
  for (const name of wasmCase.trainable) {
    closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `${label} ${name} gradient`);
  }
}

test('strict full-WASM CrossAttention matches CPU forward and projected backward gradients', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-cross-attention-training-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);

    await trainParity(api, () => crossAttentionGraph({ rank3: true, scale: true, bias: true }),
      'batched affine CrossAttention');
    await trainParity(api, () => crossAttentionGraph({ rank3: false, scale: false, bias: false }),
      'rank-2 CrossAttention');
    await trainParity(api, () => crossAttentionGraph({ rank3: true, scale: true, bias: false }),
      'scale-only CrossAttention');
    await trainParity(api, () => crossAttentionGraph({ rank3: false, scale: false, bias: true }),
      'bias-only CrossAttention');
    await trainParity(api, () => crossAttentionGraph({
      rank3: true, scale: true, bias: true, dModel: 6, heads: 3,
    }), 'three-head CrossAttention');
    await trainParity(api, () => crossAttentionGraph({ rank3: true, scale: true, bias: true, sharedQKV: true }),
      'shared Q/KV CrossAttention');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
