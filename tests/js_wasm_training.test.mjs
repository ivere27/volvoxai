import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/index.js';
import { createWasmStepRunner } from './helpers/training_session.mjs';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

function linearGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2]);
  const weight = graph.addWeight('weight', [2, 2], 'float32', {
    buffer: Float32Array.from([0.2, -0.1, 0.3, 0.4]),
  });
  const bias = graph.addWeight('bias', [2], 'float32', {
    buffer: Float32Array.from([0.01, -0.02]),
  });
  const { out } = graph.addNode({
    opType: 'MatMul',
    inputs: { input, weight, bias },
    outputs: { out: [1, 2] },
    wLayout: 'din',
  }).outputs;
  graph.setOutputs([out.name]);
  return graph;
}

function dropoutBroadcastGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2]);
  const bias = graph.addWeight('bias', [2], 'float32', {
    buffer: Float32Array.from([0.1, -0.2]),
  });
  const { out: added } = graph.addOp('Add', { a: input, b: bias }, { out: [1, 2] });
  const { out } = graph.addOp('Dropout', { input: added }, { out: [1, 2] }, { p: 0.25 });
  graph.setOutputs([out.name]);
  return graph;
}

function binaryBroadcastGraph(opType) {
  const graph = new Graph();
  const left = graph.addWeight('left', [2, 2], 'float32', {
    buffer: Float32Array.from([0.8, -0.4, 0.6, 1.2]),
  });
  const right = graph.addWeight('right', [2], 'float32', {
    buffer: Float32Array.from([0.5, -0.25]),
  });
  const { out } = graph.addOp(opType, { a: left, b: right }, { out: [2, 2] });
  graph.setOutputs([out.name]);
  return graph;
}

function expandGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', {
    buffer: Float32Array.from([0.2, -0.4]),
  });
  const { out } = graph.addOp('Expand', { input }, { out: [2, 2] });
  graph.setOutputs([out.name]);
  return graph;
}

function groupNormGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 1, 2, 2]);
  const weight = graph.addWeight('weight', [2], 'float32', {
    buffer: Float32Array.from([1.2, 0.8]),
  });
  const bias = graph.addWeight('bias', [2], 'float32', {
    buffer: Float32Array.from([0.1, -0.2]),
  });
  const { out } = graph.addOp('GroupNorm', { input, weight, bias }, { out: [1, 1, 2, 2] }, {
    num_groups: 1,
    eps: 1e-5,
  });
  graph.setOutputs([out.name]);
  return graph;
}

function softmaxGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 3], 'float32', {
    buffer: Float32Array.from([0.2, -0.4, 0.7]),
  });
  const { out } = graph.addOp(opType, { input }, { out: [1, 3] });
  graph.setOutputs([out.name]);
  return graph;
}

function conv2dGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 3, 3, 1]);
  const weight = graph.addWeight('weight', [2, 2, 1, 2], 'float32', { buffer: Float32Array.from([0.2, -0.1, 0.3, 0.4, 0.1, 0.2, -0.2, 0.3]) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.from([0.1, -0.1]) });
  const { out } = graph.addOp('Conv2D', { input, weight, bias }, { out: [1, 2, 2, 2] }, { stride: [1, 1], padding: [0, 0] });
  graph.setOutputs([out.name]);
  return graph;
}

function conv1dGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2, 4], 'float32', { buffer: Float32Array.from([.2,-.1,.4,.3, .5,.1,-.2,.6]) });
  const weight = graph.addWeight('weight', [2, 2, 3], 'float32', { buffer: Float32Array.from([.1,.2,-.1, .3,-.2,.4, -.2,.1,.3, .4,.2,-.3]) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.from([.05,-.1]) });
  const { out } = graph.addOp('Conv1D', { input, weight, bias }, { out: [1, 2, 4] }, { stride: 1, padding: 1 });
  graph.setOutputs([out.name]);
  return graph;
}

function convTranspose2dGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2, 2, 1], 'float32', { buffer: Float32Array.from([.2,-.1,.4,.3]) });
  const weight = graph.addWeight('weight', [1, 2, 2, 2], 'float32', { buffer: Float32Array.from([.1,.2,-.1,.3, .2,-.2,.4,.1]) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.from([.05,-.03]) });
  const { out } = graph.addOp('ConvTranspose2D', { input, weight, bias }, { out: [1, 3, 3, 2] }, { kernel: [2, 2], stride: [1, 1], padding: [0, 0] });
  graph.setOutputs([out.name]);
  return graph;
}

function padGraph() {
  const graph=new Graph(); const input=graph.addWeight('input',[1,2,2,2],'float32',{buffer:Float32Array.from([.2,-.1,.4,.3,.5,.1,-.2,.6])});
  const {out}=graph.addOp('Pad',{input},{out:[1,4,5,2]},{pads:[1,2,1,1],value:-.25}); graph.setOutputs([out.name]); return graph;
}

function interp1dGraph(opType='Interpolate1D'){
  const graph=new Graph(),input=graph.addWeight('input',[2,1,2],'float32',{buffer:Float32Array.from([0,2,10,14])});
  const {out}=graph.addOp(opType,{input},{out:[2,1,4]},{size:4});graph.setOutputs([out.name]);return graph;
}

function depthwiseConv2dGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 3, 3, 2]);
  const weight = graph.addWeight('weight', [2, 2, 2, 1], 'float32', { buffer: Float32Array.from([0.2, 0.1, -0.1, 0.3, 0.4, -0.2, 0.1, 0.2]) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.from([0.1, -0.1]) });
  const { out } = graph.addOp('Conv2D', { input, weight, bias }, { out: [1, 2, 2, 2] }, { groups: 2 });
  graph.setOutputs([out.name]);
  return graph;
}

function batchNorm2dGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2, 2, 2]);
  const weight = graph.addWeight('weight', [2], 'float32', { buffer: Float32Array.from([1.2, 0.8]) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.from([0.1, -0.2]) });
  const running_mean = graph.addWeight('running_mean', [2], 'float32', { buffer: Float32Array.from([0.3, -0.1]) });
  const running_var = graph.addWeight('running_var', [2], 'float32', { buffer: Float32Array.from([0.7, 1.1]) });
  const { out } = graph.addOp('BatchNorm2D', { input, weight, bias, running_mean, running_var }, { out: [1, 2, 2, 2] }, { eps: 1e-5 });
  graph.setOutputs([out.name]);
  return graph;
}

function pool2dGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 3, 3, 2], 'float32', {
    buffer: Float32Array.from([.1,.7,.3,.2,.5,.4,.9,.6,.8,.2,.1,.5,.4,.3,.7,.6,.2,.8]),
  });
  const { out } = graph.addOp(opType, { input }, { out: [1, 2, 2, 2] }, { kernel: [2, 2], stride: [1, 1], padding: [0, 0] });
  graph.setOutputs([out.name]);
  return graph;
}

function resize2dGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2, 2, 2], 'float32', { buffer: Float32Array.from([.1,.7,.3,.2,.5,.4,.9,.6]) });
  const outShape = ['UpsampleNearest2D', 'Upsample2x'].includes(opType) ? [1, 4, 4, 2] : [1, 3, 3, 2];
  const { out } = graph.addOp(opType, { input }, { out: outShape }, opType === 'Resize' ? { mode: 'bilinear' } : {});
  graph.setOutputs([out.name]);
  return graph;
}

function concatGraph() {
  const graph = new Graph();
  const left = graph.addWeight('left', [1, 1], 'float32', { buffer: Float32Array.from([0.2]) });
  const right = graph.addWeight('right', [1, 1], 'float32', { buffer: Float32Array.from([-0.4]) });
  const { out } = graph.addOp('Concat', { a: left, b: right }, { out: [1, 2] }, { axis: 1 });
  graph.setOutputs([out.name]);
  return graph;
}

function preluGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', { buffer: Float32Array.from([-0.5, 0.7]) });
  const slope = graph.addWeight('slope', [2], 'float32', { buffer: Float32Array.from([0.2, 0.3]) });
  const { out } = graph.addOp('PReLU', { input, slope }, { out: [1, 2] });
  graph.setOutputs([out.name]);
  return graph;
}

function splitGraph() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 4], 'float32', { buffer: Float32Array.from([.2, -.1, .4, .7]) });
  const outputs = graph.addNode({ opType: 'Split', inputs: { input }, outputs: { out0: [1, 2], out1: [1, 2] }, params: { axis: 1 } }).outputs;
  graph.setOutputs([outputs.out1.name]);
  return graph;
}

function hardActivationGraph(opType) {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', { buffer: Float32Array.from([-1, 1]) });
  const { out } = graph.addOp(opType, { input }, { out: [1, 2] }, opType === 'Clip' ? { min: -0.5, max: 0.5 } : {});
  graph.setOutputs([out.name]);
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

test('explicit WASM training uses C backward and matches CPU linear SGD and AdamW', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-training-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const input = Float32Array.from([1, -2]);

    for (const updateMode of ['sgd', 'adamw']) {
      const wasmGraph = linearGraph();
      const cpuGraph = linearGraph();
      const options = {
        inputs: { input },
        targets: [1],
        trainableTensors: ['weight', 'bias'],
        updateMode,
        optimizer: { learningRate: 0.05, weightDecay: 0.01 },
      };
      const before = Float32Array.from(wasmGraph.getTensor('weight').buffer);
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);

      assert.equal(wasm.backend, 'wasm');
      assert.equal(wasmGraph.trainingStep, 0, 'checkpoint authoring does not apply a training step');
      assert.deepEqual(wasmGraph.getTensor('weight').buffer, before);
      assert.equal(wasm.publishedGraph.trainingStep, 1);
      assert.ok(before.some((value, index) => value !== wasm.publishedGraph.getTensor('weight').buffer[index]));
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
      closeArray(wasm.gradients.get('weight'), cpu.gradients.get('weight'), `${updateMode} weight gradient`);
      closeArray(wasm.gradients.get('bias'), cpu.gradients.get('bias'), `${updateMode} bias gradient`);
      closeArray(wasm.publishedGraph.getTensor('weight').buffer, cpuGraph.getTensor('weight').buffer, `${updateMode} weight`);
      closeArray(wasm.publishedGraph.getTensor('bias').buffer, cpuGraph.getTensor('bias').buffer, `${updateMode} bias`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for broadcast Add and deterministic Dropout', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-dropout-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = dropoutBroadcastGraph();
    const cpuGraph = dropoutBroadcastGraph();
    const options = {
      inputs: { input: Float32Array.from([0.5, -0.4]) },
      targets: [1],
      trainableTensors: ['bias'],
      updateMode: 'sgd',
      optimizer: { learningRate: 0.01 },
      dropout: { seed: 3, counter: 4 },
    };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
    const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.equal(wasm.backend, 'wasm');
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('bias'), cpu.gradients.get('bias'), 'broadcast/dropout gradient');
    closeArray(wasm.publishedGraph.getTensor('bias').buffer, cpuGraph.getTensor('bias').buffer, 'broadcast/dropout update');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for broadcast Sub and Div', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-binary-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    for (const opType of ['Sub', 'Div']) {
      const wasmGraph = binaryBroadcastGraph(opType);
      const cpuGraph = binaryBroadcastGraph(opType);
      const options = {
        targets: [1, 0],
        trainableTensors: ['left', 'right'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0.01 },
      };
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.equal(wasm.backend, 'wasm');
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, `${opType} loss`);
      for (const name of ['left', 'right']) {
        closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `${opType} ${name} gradient`);
        closeArray(wasm.publishedGraph.getTensor(name).buffer, cpuGraph.getTensor(name).buffer, `${opType} ${name} update`);
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for Expand forward and broadcast-reduction backward', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-expand-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = expandGraph();
    const cpuGraph = expandGraph();
    const options = { targets: [0, 1], trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0.01 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
    const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.equal(wasm.backend, 'wasm');
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), 'Expand input gradient');
    closeArray(wasm.publishedGraph.getTensor('input').buffer, cpuGraph.getTensor('input').buffer, 'Expand input update');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for NHWC GroupNorm', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-groupnorm-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = groupNormGraph();
    const cpuGraph = groupNormGraph();
    const options = {
      inputs: { input: Float32Array.from([0.1, 0.4, -0.2, 0.8]) },
      targets: [1],
      trainableTensors: ['weight', 'bias'],
      updateMode: 'sgd',
      optimizer: { learningRate: 0.01 },
    };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
    const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.equal(wasm.backend, 'wasm');
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('weight'), cpu.gradients.get('weight'), 'GroupNorm weight gradient');
    closeArray(wasm.gradients.get('bias'), cpu.gradients.get('bias'), 'GroupNorm bias gradient');
    closeArray(wasm.publishedGraph.getTensor('weight').buffer, cpuGraph.getTensor('weight').buffer, 'GroupNorm weight update');
    closeArray(wasm.publishedGraph.getTensor('bias').buffer, cpuGraph.getTensor('bias').buffer, 'GroupNorm bias update');
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('CPU and WASM training both cover Softmax and LogSoftmax backward', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-softmax-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    for (const opType of ['Softmax', 'LogSoftmax']) {
      const wasmGraph = softmaxGraph(opType);
      const cpuGraph = softmaxGraph(opType);
      const options = {
        targets: [1],
        trainableTensors: ['input'],
        updateMode: 'sgd',
        optimizer: { learningRate: 0 },
      };
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
      const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.equal(wasm.backend, 'wasm');
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, opType);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} gradient`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM training matches CPU for Conv2D weight and bias updates', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-conv-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = conv2dGraph(), cpuGraph = conv2dGraph();
    const options = { inputs: { input: Float32Array.from([.1,.2,.3,.4,.5,.6,.7,.8,.9]) }, targets: [0, 1, 0, 1], trainableTensors: ['weight', 'bias'], updateMode: 'sgd', optimizer: { learningRate: .01 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' });
    const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.equal(wasm.backend, 'wasm'); assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('weight'), cpu.gradients.get('weight'), 'Conv2D weight gradient');
    closeArray(wasm.publishedGraph.getTensor('weight').buffer, cpuGraph.getTensor('weight').buffer, 'Conv2D weight update');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for Conv1D input, weight, and bias updates', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-conv1d-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = conv1dGraph(), cpuGraph = conv1dGraph();
    const options = { targets: [0, 1], trainableTensors: ['input', 'weight', 'bias'], updateMode: 'sgd', optimizer: { learningRate: .01 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    for (const name of ['input', 'weight', 'bias']) {
      closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `Conv1D ${name} gradient`);
      closeArray(wasm.publishedGraph.getTensor(name).buffer, cpuGraph.getTensor(name).buffer, `Conv1D ${name} update`);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for ConvTranspose2D gradients', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-convtranspose-'));
  try {
    const wasmPath=join(directory,'volvoxai.full.wasm'); await buildFullWasm(wasmPath); const runWasmStep=createWasmStepRunner(wasmPath);
    const wasmGraph=convTranspose2dGraph(),cpuGraph=convTranspose2dGraph(); const options={targets:[0,1,0,1,0,1,0,1,0],trainableTensors:['input','weight','bias'],updateMode:'sgd',optimizer:{learningRate:.01}};
    const wasm=await runWasmStep(wasmGraph,{...options,backend:'wasm'}),cpu=await CPUAutograd.trainStep(cpuGraph,options); assert.ok(Math.abs(wasm.loss-cpu.loss)<2e-5);
    for(const name of ['input','weight','bias']) { closeArray(wasm.gradients.get(name),cpu.gradients.get(name),`ConvTranspose2D ${name} gradient`); closeArray(wasm.publishedGraph.getTensor(name).buffer,cpuGraph.getTensor(name).buffer,`ConvTranspose2D ${name} update`); }
  } finally { await rm(directory,{recursive:true,force:true}); }
});

test('WASM training matches CPU for constant Pad forward and backward', {timeout:120_000}, async(t)=>{
  try{await run(clang,['--version']);}catch(error){if(error?.code==='ENOENT')return t.skip(`${clang} is unavailable`);throw error;} const directory=await mkdtemp(join(tmpdir(),'volvoxai-wasm-pad-'));
  try{const wasmPath=join(directory,'volvoxai.full.wasm');await buildFullWasm(wasmPath);const runWasmStep=createWasmStepRunner(wasmPath),wasmGraph=padGraph(),cpuGraph=padGraph(),options={targets:Array.from({length:20},(_,i)=>i%2),trainableTensors:['input'],updateMode:'sgd',optimizer:{learningRate:.01}};
    const wasm=await runWasmStep(wasmGraph,{...options,backend:'wasm'}),cpu=await CPUAutograd.trainStep(cpuGraph,options);assert.ok(Math.abs(wasm.loss-cpu.loss)<2e-5);closeArray(wasm.gradients.get('input'),cpu.gradients.get('input'),'Pad input gradient');closeArray(wasm.publishedGraph.getTensor('input').buffer,cpuGraph.getTensor('input').buffer,'Pad input update');
  }finally{await rm(directory,{recursive:true,force:true});}
});

test('WASM training matches CPU for batched Interpolate1D gradients',{timeout:120_000},async(t)=>{
  try{await run(clang,['--version']);}catch(error){if(error?.code==='ENOENT')return t.skip(`${clang} is unavailable`);throw error;}const directory=await mkdtemp(join(tmpdir(),'volvoxai-wasm-interp1d-'));
  try{const wasmPath=join(directory,'volvoxai.full.wasm');await buildFullWasm(wasmPath);const runWasmStep=createWasmStepRunner(wasmPath);
    {const opType='Interpolate1D',wasmGraph=interp1dGraph(opType),cpuGraph=interp1dGraph(opType),options={targets:[1,2],trainableTensors:['input'],updateMode:'sgd',optimizer:{learningRate:.01}};const wasm=await runWasmStep(wasmGraph,{...options,backend:'wasm'}),cpu=await CPUAutograd.trainStep(cpuGraph,options);assert.ok(Math.abs(wasm.loss-cpu.loss)<2e-5,opType);closeArray(wasm.gradients.get('input'),cpu.gradients.get('input'),`${opType} input gradient`);closeArray(wasm.publishedGraph.getTensor('input').buffer,cpuGraph.getTensor('input').buffer,`${opType} input update`);}
  }finally{await rm(directory,{recursive:true,force:true});}
});

test('WASM training matches CPU for depthwise Conv2D', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-depthwise-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = depthwiseConv2dGraph(), cpuGraph = depthwiseConv2dGraph();
    const options = { inputs: { input: Float32Array.from([...Array(18)].map((_, index) => (index + 1) / 20)) }, targets: [0, 1, 0, 1], trainableTensors: ['weight', 'bias'], updateMode: 'sgd', optimizer: { learningRate: .01 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5); closeArray(wasm.gradients.get('weight'), cpu.gradients.get('weight'), 'depthwise Conv2D gradient');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for fixed-statistics BatchNorm2D', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-batchnorm-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = batchNorm2dGraph(), cpuGraph = batchNorm2dGraph();
    const options = { inputs: { input: Float32Array.from([.1,.2,.3,.4,.5,.6,.7,.8]) }, targets: [0, 1, 0, 1], trainableTensors: ['weight', 'bias'], updateMode: 'sgd', optimizer: { learningRate: .01 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('weight'), cpu.gradients.get('weight'), 'BatchNorm2D weight gradient');
    closeArray(wasm.gradients.get('bias'), cpu.gradients.get('bias'), 'BatchNorm2D bias gradient');
    closeArray(wasm.publishedGraph.getTensor('weight').buffer, cpuGraph.getTensor('weight').buffer, 'BatchNorm2D weight update');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for MaxPool2D and AveragePool2D', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-pool-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    for (const opType of ['MaxPool2D', 'AveragePool2D']) {
      const wasmGraph = pool2dGraph(opType), cpuGraph = pool2dGraph(opType);
      const options = { targets: [0, 1, 0, 1], trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, opType);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} input gradient`);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for bilinear, nearest, and canonical 2x upsample', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-resize-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    for (const opType of ['Resize', 'ResizeNearest2D', 'UpsampleNearest2D']) {
      const wasmGraph = resize2dGraph(opType), cpuGraph = resize2dGraph(opType);
      const options = { targets: Array.from({ length: opType === 'UpsampleNearest2D' ? 16 : 9 }, (_, index) => index % 2), trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, opType);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} input gradient`);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for Concat gradients', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-concat-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = concatGraph(), cpuGraph = concatGraph();
    const options = { targets: [1], trainableTensors: ['left', 'right'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('left'), cpu.gradients.get('left'), 'Concat left gradient');
    closeArray(wasm.gradients.get('right'), cpu.gradients.get('right'), 'Concat right gradient');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for PReLU input and slope gradients', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-prelu-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = preluGraph(), cpuGraph = preluGraph();
    const options = { targets: [1], trainableTensors: ['input', 'slope'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), 'PReLU input gradient');
    closeArray(wasm.gradients.get('slope'), cpu.gradients.get('slope'), 'PReLU slope gradient');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU when only a non-primary Split output is live', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-split-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    const wasmGraph = splitGraph(), cpuGraph = splitGraph();
    const options = { targets: [1], trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
    const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
    assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5);
    closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), 'Split input gradient');
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('WASM training matches CPU for HardSigmoid and HardSwish', { timeout: 120_000 }, async (t) => {
  try { await run(clang, ['--version']); } catch (error) { if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`); throw error; }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-hard-activation-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm'); await buildFullWasm(wasmPath);
    const runWasmStep = createWasmStepRunner(wasmPath);
    for (const opType of ['HardSigmoid', 'HardSwish', 'Clip']) {
      const wasmGraph = hardActivationGraph(opType), cpuGraph = hardActivationGraph(opType);
      const options = { targets: [1], trainableTensors: ['input'], updateMode: 'sgd', optimizer: { learningRate: 0 } };
      const wasm = await runWasmStep(wasmGraph, { ...options, backend: 'wasm' }); const cpu = await CPUAutograd.trainStep(cpuGraph, options);
      assert.ok(Math.abs(wasm.loss - cpu.loss) < 2e-5, opType);
      closeArray(wasm.gradients.get('input'), cpu.gradients.get('input'), `${opType} input gradient`);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
