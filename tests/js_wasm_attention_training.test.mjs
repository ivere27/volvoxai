import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { TrainingGraph as Graph } from '../ts/training/TrainingGraph.js';
import { createWasmStepRunner } from './helpers/training_session.mjs';
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

function values(length, salt) {
  return Float32Array.from({ length }, (_, index) => (((index * salt) % 23) - 11) / 19);
}

function maskValues(mode, batch, queries, keys) {
  if (mode === 'K') return Int32Array.from([1, 1, 0]);
  if (mode === 'BK') return Int32Array.from([
    1, 1, 0,
    1, 0, 1,
  ].slice(0, batch * keys));
  if (mode === 'QK') return Int32Array.from([
    1, 1, 0,
    1, 0, 1,
    0, 0, 0,
  ].slice(0, queries * keys));
  return Int32Array.from([
    1, 1, 0,
    1, 0, 1,
    0, 0, 0,
    1, 1, 1,
    0, 1, 1,
    1, 0, 1,
  ].slice(0, batch * queries * keys));
}

function maskShape(mode, batch, queries, keys) {
  if (mode === 'K') return [keys];
  if (mode === 'BK') return [batch, keys];
  if (mode === 'QK') return [queries, keys];
  return [batch, queries, keys];
}

function selfAttentionGraph({ mode, rank3, causal, dropout = 0.35 }) {
  const graph = new Graph();
  const batch = rank3 ? 2 : 1;
  const queries = 3;
  const dModel = 2;
  const qkv = graph.addWeight('qkv', rank3 ? [batch, queries, dModel * 3] : [queries, dModel * 3], 'float32', {
    buffer: values(batch * queries * dModel * 3, 7),
  });
  const inputs = { qkv };
  const executionInputs = {};
  if (mode !== 'none') {
    const mask = maskValues(mode, batch, queries, queries);
    inputs.mask = graph.addInput('mask', maskShape(mode, batch, queries, queries), 'int32', {
      buffer: mask,
    });
    executionInputs.mask = mask;
  }
  const params = { heads: 1, dropout, dropout_seed: 29 };
  if (causal !== undefined) params.causal = causal;
  const { out } = graph.addOp('SDPA', inputs, {
    out: rank3 ? [batch, queries, dModel] : [queries, dModel],
  }, params);
  graph.setOutputs([out.name]);
  return { graph, output: out.name, trainable: ['qkv'], targetCount: batch * queries, executionInputs };
}

function crossAttentionGraph({ mode, rank3, causal, dropout = 0.3, shared = false }) {
  const graph = new Graph();
  const batch = rank3 ? 2 : 1;
  const queries = 3;
  const keys = 3;
  const dModel = 2;
  const shapeQ = rank3 ? [batch, queries, dModel] : [queries, dModel];
  const shapeKV = rank3 ? [batch, keys, dModel] : [keys, dModel];
  const q = graph.addWeight('q', shapeQ, 'float32', { buffer: values(batch * queries * dModel, 5) });
  const k = shared ? q : graph.addWeight('k', shapeKV, 'float32', { buffer: values(batch * keys * dModel, 9) });
  const v = shared ? q : graph.addWeight('v', shapeKV, 'float32', { buffer: values(batch * keys * dModel, 13) });
  const inputs = { q, k, v };
  const executionInputs = {};
  if (mode !== 'none') {
    const mask = maskValues(mode, batch, queries, keys);
    inputs.mask = graph.addInput('mask', maskShape(mode, batch, queries, keys), 'int32', {
      buffer: mask,
    });
    executionInputs.mask = mask;
  }
  const params = { heads: 1, attention_dropout: dropout, training_seed: 17 };
  if (causal !== undefined) params.causal = causal;
  const { out } = graph.addOp('CrossSDPA', inputs, { out: shapeQ }, params);
  graph.setOutputs([out.name]);
  return {
    graph,
    output: out.name,
    trainable: shared ? ['q'] : ['q', 'k', 'v'],
    targetCount: batch * queries,
    executionInputs,
  };
}

async function trainParity(runWasmStep, makeGraph, counter) {
  const wasmCase = makeGraph();
  const cpuCase = makeGraph();
  const targets = Array.from({ length: wasmCase.targetCount }, (_, index) => index & 1);
  const options = {
    targets,
    trainableTensors: wasmCase.trainable,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    dropout: { seed: 41, counter },
  };
  const wasm = await runWasmStep(wasmCase.graph, {
    ...options, inputs: wasmCase.executionInputs, backend: 'wasm',
  });
  const cpu = await CPUAutograd.trainStep(cpuCase.graph, {
    ...options, inputs: cpuCase.executionInputs, shapeSignature: wasm.shapeSignature,
  });
  assert.equal(wasm.backend, 'wasm');
  assert.ok(Math.abs(wasm.loss - cpu.loss) <= 5e-5,
    `loss at dropout counter ${counter} for ${wasmCase.graph.nodes[0].opType} ` +
    `${JSON.stringify(wasmCase.graph.nodes[0].params)}: ${wasm.loss} != ${cpu.loss}`);
  for (const name of wasmCase.trainable) {
    closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `${name} gradient at counter ${counter}`);
  }
  return {
    gradients: new Map(wasmCase.trainable.map((name) => [name, Float32Array.from(wasm.gradients.get(name))])),
  };
}

test('strict full-WASM SDPA and CrossSDPA match CPU masks, causal defaults, and attention dropout', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-attention-training-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);

    // [K] also exercises rank-2 planning. The remaining layouts use B=2 to
    // validate batch strides and the [B,K] versus [Q,K] distinction.
    for (const entry of [
      { mode: 'none', rank3: true },
      { mode: 'K', rank3: false },
      { mode: 'BK', rank3: true },
      { mode: 'QK', rank3: true, causal: false },
      { mode: 'BQK', rank3: true },
    ]) await trainParity(runWasmStep, () => selfAttentionGraph(entry), 5);

    // Cross attention is non-causal by default; the QK case explicitly flips
    // it to causal so both default policies are covered in the strict path.
    for (const entry of [
      { mode: 'none', rank3: true },
      { mode: 'K', rank3: false },
      { mode: 'BK', rank3: true },
      { mode: 'QK', rank3: true, causal: true },
      { mode: 'BQK', rank3: true },
      { mode: 'none', rank3: true, shared: true },
    ]) await trainParity(runWasmStep, () => crossAttentionGraph(entry), 5);

    const descriptor = { mode: 'BQK', rank3: true };
    const first = await trainParity(runWasmStep, () => selfAttentionGraph(descriptor), 11);
    const repeated = await trainParity(runWasmStep, () => selfAttentionGraph(descriptor), 11);
    closeArray(repeated.gradients.get('qkv'), first.gradients.get('qkv'), 'repeat attention dropout gradient', 0);

    const advanced = await trainParity(runWasmStep, () => selfAttentionGraph(descriptor), 12);
    assert.ok(first.gradients.get('qkv').some(
      (value, index) => value !== advanced.gradients.get('qkv')[index]),
      'changing the dropout counter changes the attention-probability mask');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
