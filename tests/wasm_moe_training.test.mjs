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

function moeGraph({ normalize, routerBias: includeRouterBias, expertBias: includeExpertBias, tied = false }) {
  const graph = new Graph();
  const input = graph.addWeight('input', [2, 2], 'float32', {
    buffer: Float32Array.from([0.7, -0.2, 0.1, 0.8]),
  });
  const router = graph.addWeight('router', [2, 3], 'float32', {
    buffer: tied ? new Float32Array(6) : Float32Array.from([0.9, 0.3, -0.8, -0.4, 0.7, 0.2]),
  });
  const routerBias = includeRouterBias ? graph.addWeight('router_bias', [3], 'float32', {
    buffer: tied ? new Float32Array(3) : Float32Array.from([0.1, -0.05, 0]),
  }) : null;
  const routes = graph.addOp('MoERouter', {
    input,
    weight: router,
    ...(routerBias ? { bias: routerBias } : {}),
  }, {
    indices: [2, 2],
    weights: [2, 2],
  }, {
    num_experts: 3,
    top_k: 2,
    normalize,
    temperature: normalize ? 0.5 : 2,
  });
  const experts = graph.addWeight('experts', [3, 2, 2], 'float32', {
    buffer: Float32Array.from([
      0.4, -0.2, 0.1, 0.6,
      -0.3, 0.8, 0.5, -0.1,
      0.9, 0.2, -0.7, 0.3,
    ]),
  });
  const expertBias = includeExpertBias ? graph.addWeight('expert_bias', [3, 2], 'float32', {
    buffer: Float32Array.from([0.1, 0, -0.1, 0.2, 0.05, -0.2]),
  }) : null;
  const { out } = graph.addOp('MoELinear', {
    input,
    expert_weight: experts,
    route_indices: routes.indices,
    route_weights: routes.weights,
    ...(expertBias ? { expert_bias: expertBias } : {}),
  }, { out: [2, 2] });
  graph.setOutputs([out.name]);
  return {
    graph,
    output: out.name,
    routes,
    trainable: ['input', 'router', ...(routerBias ? ['router_bias'] : []), 'experts', ...(expertBias ? ['expert_bias'] : [])],
    executionInputs: {},
  };
}

function routeWeightGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [2, 2], 'float32', {
    buffer: Float32Array.from([0.6, -0.3, 0.2, 0.5]),
  });
  const experts = graph.addWeight('experts', [3, 2, 2], 'float32', {
    buffer: Float32Array.from([
      0.3, -0.1, 0.2, 0.5,
      -0.2, 0.6, 0.4, -0.3,
      0.8, 0.2, -0.5, 0.4,
    ]),
  });
  const expertBias = graph.addWeight('expert_bias', [3, 2], 'float32', {
    buffer: Float32Array.from([0.05, -0.1, -0.2, 0.15, 0.1, 0.2]),
  });
  const indexValues = Float32Array.from([0, 2, 1, 2]);
  const indices = graph.addInput('indices', [2, 2], 'float32', {
    buffer: indexValues,
  });
  const weights = graph.addWeight('route_weights', [2, 2], 'float32', {
    buffer: Float32Array.from([0.65, 0.35, 0.2, 0.8]),
  });
  const { out } = graph.addOp('MoELinear', {
    input,
    expert_weight: experts,
    expert_bias: expertBias,
    route_indices: indices,
    route_weights: weights,
  }, { out: [2, 2] });
  graph.setOutputs([out.name]);
  return {
    graph,
    output: out.name,
    trainable: ['input', 'experts', 'expert_bias', 'route_weights'],
    executionInputs: { indices: indexValues },
  };
}

async function trainParity(runWasmStep, makeGraph) {
  const wasmCase = makeGraph();
  const cpuCase = makeGraph();
  const options = {
    targets: [0, 1],
    trainableTensors: wasmCase.trainable,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  };
  const wasm = await runWasmStep(wasmCase.graph, {
    ...options, inputs: wasmCase.executionInputs, backend: 'wasm',
  });
  const cpu = await CPUAutograd.trainStep(cpuCase.graph, {
    ...options, inputs: cpuCase.executionInputs,
  });
  assert.equal(wasm.backend, 'wasm');
  assert.ok(Math.abs(wasm.loss - cpu.loss) <= 5e-5, 'MoE loss');
  for (const name of wasmCase.trainable) {
    closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `MoE ${name} gradient`);
  }
  return { wasmCase, cpuCase, wasm, cpu };
}

test('strict full-WASM MoERouter and MoELinear match CPU routing and all gradients', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-moe-training-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);

    const normalized = await trainParity(runWasmStep, () => moeGraph({
      normalize: true, routerBias: true, expertBias: true, tied: true,
    }));
    assert.ok(normalized.cpuCase.routes.indices.buffer instanceof Float32Array,
      'portable router indices use exact F32 integer storage');
    assert.ok(normalized.cpuCase.routes.weights.buffer instanceof Float32Array,
      'portable router gates use F32 storage');
    assert.deepEqual([...normalized.cpuCase.routes.indices.buffer], [0, 1, 0, 1],
      'equal router logits choose lower expert indices first');

    const fullSoftmax = await trainParity(runWasmStep, () => moeGraph({
      normalize: false, routerBias: false, expertBias: false,
    }));
    assert.notEqual(fullSoftmax.wasm.gradients.get('router')[2], 0,
      'non-normalized selected gates propagate denominator gradients to unselected router experts');

    await trainParity(runWasmStep, () => moeGraph({
      normalize: true, routerBias: true, expertBias: false,
    }));
    await trainParity(runWasmStep, routeWeightGraph);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
