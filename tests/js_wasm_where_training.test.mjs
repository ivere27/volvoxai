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

function whereGraph(opType, conditionDtype) {
  const graph = new Graph();
  const condition = graph.addInput('condition', [1, 3], conditionDtype);
  const left = graph.addWeight('left', [1, 3], 'float32', {
    buffer: Float32Array.of(0.2, -0.4, 0.7),
  });
  const right = graph.addWeight('right', [1, 3], 'float32', {
    buffer: Float32Array.of(-0.3, 0.6, 0.1),
  });
  const { out } = graph.addOp(opType, {
    condition,
    x: left,
    y: right,
  }, { out: [1, 3] });
  graph.outputNames = [out.name];
  return graph;
}

function sliceGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [2, 4], 'float32', {
    buffer: Float32Array.of(0.2, -0.4, 0.7, 0.1, -0.3, 0.6, 0.5, -0.2),
  });
  const { out } = graph.addOp('Slice', { input }, { out: [2, 2] }, {
    axes: [1],
    starts: [1],
    steps: [2],
  });
  graph.outputNames = [out.name];
  return graph;
}

function gatherGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [2, 3], 'float32', {
    buffer: Float32Array.of(0.2, -0.4, 0.7, -0.3, 0.6, 0.1),
  });
  const indicesShape = opType === 'Gather' ? [2] : [2, 2];
  const indices = graph.addInput('indices', indicesShape, 'int32');
  const outputShape = [2, 2];
  const { out } = graph.addOp(opType, { input, indices }, { out: outputShape }, { axis: 1 });
  graph.outputNames = [out.name];
  return graph;
}

function visionProfileGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2, 2, 1], 'float32', {
    buffer: Float32Array.of(0.2, -0.4, 0.7, 0.1),
  });
  const outputShape = opType === 'ProfileX' || opType === 'ProfileY' ? [1, 2, 2] : [1, 1, 2];
  const { out } = graph.addOp(opType, { input }, { out: outputShape });
  graph.outputNames = [out.name];
  return graph;
}

function closeArray(actual, expected, label, tolerance = 2e-5) {
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

test('WASM training matches CPU branch routing for Where and Mask', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-where-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);

    for (const [opType, conditionDtype, condition] of [
      ['Where', 'int32', Int32Array.of(1, 0, -2)],
      ['Mask', 'float32', Float32Array.of(0, 2, 0)],
    ]) {
      const wasmGraph = whereGraph(opType, conditionDtype);
      const cpuGraph = whereGraph(opType, conditionDtype);
      const options = {
        inputs: { condition },
        targets: [2],
        trainableTensors: ['left', 'right'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0.01 },
      };
      const wasm = await api.trainStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.equal(wasm.backend, 'wasm');
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, `${opType} loss`);
      for (const name of ['left', 'right']) {
        closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `${opType} ${name} gradient`);
        closeArray(wasmGraph.getTensor(name).buffer, cpuGraph.getTensor(name).buffer,
          `${opType} ${name} update`);
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training scatters positive-step Slice gradients to the selected values', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-slice-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);
    const wasmGraph = sliceGraph();
    const cpuGraph = sliceGraph();
    const options = {
      targets: [0, 1],
      trainableTensors: ['input'],
      updateMode: 'sgd',
      optimizer: { learningRate: 0.01 },
    };
    const wasm = await api.trainStep(wasmGraph, { ...options, backend: 'wasm' });
    const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.equal(wasm.backend, 'wasm');
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, 'Slice loss');
    closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), 'Slice input gradient');
    closeArray(wasmGraph.getTensor('input').buffer, cpuGraph.getTensor('input').buffer,
      'Slice input update');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU scatter-add gradients for Gather and GatherElements', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-gather-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);
    for (const [opType, indices] of [
      ['Gather', Int32Array.of(2, 0)],
      ['GatherElements', Int32Array.of(-1, 0, 1, 2)],
    ]) {
      const wasmGraph = gatherGraph(opType);
      const cpuGraph = gatherGraph(opType);
      const options = {
        inputs: { indices },
        targets: [0, 1],
        trainableTensors: ['input'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0.01 },
      };
      const wasm = await api.trainStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.equal(wasm.backend, 'wasm');
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, `${opType} loss`);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} input gradient`);
      closeArray(wasmGraph.getTensor('input').buffer, cpuGraph.getTensor('input').buffer,
        `${opType} input update`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for NHWC vision profile primitive gradients', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-vision-profile-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);
    for (const opType of ['MeanHeight', 'ProfileX', 'ProfileY', 'SpatialSoftargmaxY']) {
      const wasmGraph = visionProfileGraph(opType);
      const cpuGraph = visionProfileGraph(opType);
      const options = {
        targets: opType === 'ProfileX' || opType === 'ProfileY' ? [0, 1] : [0],
        trainableTensors: ['input'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0.01 },
      };
      const wasm = await api.trainStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.equal(wasm.backend, 'wasm');
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 3e-5, `${opType} loss`);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} input gradient`, 4e-5);
      closeArray(wasmGraph.getTensor('input').buffer, cpuGraph.getTensor('input').buffer,
        `${opType} input update`, 4e-5);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
