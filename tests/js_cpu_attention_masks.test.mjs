import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine, Graph } from '../ts/index.js';
import { CPUAutograd } from '../ts/training/index.js';

function addWeight(graph, name, shape, values) {
  const tensor = graph.addWeight(name, shape);
  tensor.buffer = Float32Array.from(values);
  return tensor;
}

function close(actual, expected, label, absolute = 3e-4, relative = 3e-3) {
  const tolerance = absolute + relative * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: analytic=${actual}, numeric=${expected}, tolerance=${tolerance}`);
}

async function crossEntropy(graph, targets, inputs = {}) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await engine.execute(inputs);
  const logits = graph.getTensor(graph.outputNames[0]);
  const classes = logits.shape[logits.shape.length - 1];
  let loss = 0;
  for (let row = 0; row < targets.length; row++) {
    let maximum = -Infinity;
    for (let col = 0; col < classes; col++) {
      maximum = Math.max(maximum, logits.buffer[row * classes + col]);
    }
    let denominator = 0;
    for (let col = 0; col < classes; col++) {
      denominator += Math.exp(logits.buffer[row * classes + col] - maximum);
    }
    loss += maximum + Math.log(denominator) - logits.buffer[row * classes + targets[row]];
  }
  return loss / targets.length;
}

async function numericGradient(graph, tensorName, index, targets, inputs) {
  const tensor = graph.getTensor(tensorName);
  const original = tensor.buffer[index];
  const epsilon = 1e-3;
  tensor.buffer[index] = original + epsilon;
  const positive = await crossEntropy(graph, targets, inputs);
  tensor.buffer[index] = original - epsilon;
  const negative = await crossEntropy(graph, targets, inputs);
  tensor.buffer[index] = original;
  return (positive - negative) / (2 * epsilon);
}

async function zeroRateGradients(graph, trainableTensors, targets, inputs = {}, options = {}) {
  return CPUAutograd.trainStep(graph, {
    inputs,
    targets,
    trainableTensors,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    ...options,
  });
}

test('CPU SDPA supports full attention and masks forward/backward consistently', async () => {
  const graph = new Graph();
  const qkv = addWeight(graph, 'qkv', [1, 3, 6], [
    0.2, -0.1, 0.3, 0.4, 0.5, -0.2,
    -0.3, 0.6, 0.1, -0.5, 0.7, 0.2,
    0.4, 0.2, -0.6, 0.3, -0.4, 0.9,
  ]);
  const mask = graph.addInput('mask', [3], 'int32', { buffer: new Int32Array(3) });
  const { out } = graph.addOp(
    'SDPA', { qkv, mask }, { out: [1, 3, 2] }, { heads: 1, scale: 0.43, causal: false },
  );
  graph.outputNames = [out.name];
  const inputs = { mask: Int32Array.of(1, 0, 1) };
  const targets = [0, 1, 0];

  const result = await zeroRateGradients(graph, ['qkv'], targets, inputs);
  const gradient = result.gradients.get('qkv');
  for (const index of [0, 14, 16]) {
    close(gradient[index], await numericGradient(graph, 'qkv', index, targets, inputs),
      `masked full SDPA qkv[${index}]`);
  }
  assert.deepEqual([...gradient.slice(8, 12)], [0, 0, 0, 0],
    'the masked key contributes neither K nor V gradients');

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await engine.execute(inputs);
  assert.notDeepEqual([...out.buffer.slice(0, 2)], [0.5, -0.2],
    'causal:false lets the first query attend beyond the first value');

  await engine.execute({ mask: Int32Array.of(0, 0, 0) });
  assert.deepEqual([...out.buffer], [0, 0, 0, 0, 0, 0]);
  const fullyMasked = await zeroRateGradients(
    graph, ['qkv'], targets, { mask: Int32Array.of(0, 0, 0) },
  );
  assert.ok(fullyMasked.gradients.get('qkv').every((value) => value === 0));
});

test('CPU CrossSDPA applies query masks and defaults to non-causal attention', async () => {
  const graph = new Graph();
  const q = addWeight(graph, 'q', [1, 2, 2], [0.2, -0.1, -0.3, 0.6]);
  const k = addWeight(graph, 'k', [1, 3, 2], [0.3, 0.4, 0.1, -0.5, -0.6, 0.3]);
  const v = addWeight(graph, 'v', [1, 3, 2], [0.5, -0.2, 0.7, 0.2, -0.4, 0.9]);
  const mask = graph.addInput('mask', [2, 3], 'int32', { buffer: new Int32Array(6) });
  const { out } = graph.addOp(
    'CrossSDPA', { q, k, v, mask }, { out: [1, 2, 2] }, { heads: 1, scale: 0.37 },
  );
  graph.outputNames = [out.name];
  const inputs = { mask: Int32Array.of(1, 0, 1, 0, 0, 0) };
  const targets = [0, 1];

  const result = await zeroRateGradients(graph, ['q', 'k', 'v'], targets, inputs);
  for (const [name, index] of [['q', 0], ['k', 4], ['v', 4]]) {
    close(result.gradients.get(name)[index],
      await numericGradient(graph, name, index, targets, inputs),
      `masked CrossSDPA ${name}[${index}]`);
  }
  assert.deepEqual([...result.gradients.get('q').slice(2, 4)], [0, 0]);
  assert.deepEqual([...result.gradients.get('k').slice(2, 4)], [0, 0]);
  assert.deepEqual([...result.gradients.get('v').slice(2, 4)], [0, 0]);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await engine.execute(inputs);
  assert.deepEqual([...out.buffer.slice(2, 4)], [0, 0]);
  assert.notDeepEqual([...out.buffer.slice(0, 2)], [0.5, -0.2],
    'CrossSDPA is non-causal unless causal:true is requested');
  graph.nodes[0].params.causal = true;
  await engine.execute(inputs);
  close(out.buffer[0], 0.5, 'causal CrossSDPA first value', 1e-6, 1e-6);
  close(out.buffer[1], -0.2, 'causal CrossSDPA second value', 1e-6, 1e-6);
});

test('CPU attention accepts batch and batch-query mask layouts', async (t) => {
  for (const [label, maskShape] of [['BK', [2, 1]], ['BQK', [2, 1, 1]]]) {
    await t.test(label, async () => {
      const graph = new Graph();
      const qkv = addWeight(graph, 'qkv', [2, 1, 6], [
        0.2, 0.1, -0.3, 0.4, 0.5, -0.2,
        -0.4, 0.7, 0.2, -0.1, 0.3, 0.6,
      ]);
      const mask = graph.addInput('mask', maskShape, 'int32', {
        buffer: new Int32Array(2),
      });
      const { out } = graph.addOp(
        'SDPA', { qkv, mask }, { out: [2, 1, 2] }, { heads: 1, causal: false },
      );
      graph.outputNames = [out.name];
      const engine = new CPUEngine();
      engine.allocateGraph(graph);
      await engine.execute({ mask: Int32Array.of(1, 0) });
      close(out.buffer[0], 0.5, `${label} first output`, 1e-7, 1e-7);
      close(out.buffer[1], -0.2, `${label} second output`, 1e-7, 1e-7);
      assert.deepEqual([...out.buffer.slice(2, 4)], [0, 0]);
    });
  }
});

test('CPU trainStep honors ignoreIndex and lossMask and advances successful steps once', async () => {
  const graph = new Graph();
  const logits = addWeight(graph, 'logits', [1, 4, 3], new Array(12).fill(0));
  graph.outputNames = [logits.name];
  const result = await zeroRateGradients(
    graph,
    ['logits'],
    Int32Array.of(2, -100, 0, 1),
    {},
    { ignoreIndex: -100, lossMask: Uint8Array.of(1, 1, 0, 1) },
  );

  close(result.loss, Math.log(3), 'masked cross entropy loss', 1e-7, 1e-7);
  assert.equal(result.examples, 2);
  assert.equal(result.correct, 0);
  const expectedGradient = [
    1 / 6, 1 / 6, -1 / 3,
    0, 0, 0,
    0, 0, 0,
    1 / 6, -1 / 3, 1 / 6,
  ];
  expectedGradient.forEach((expected, index) => {
    close(result.gradients.get('logits')[index], expected, `loss gradient[${index}]`, 1e-7, 1e-7);
  });
  assert.equal(graph.trainingStep, 1);

  const noOp = await zeroRateGradients(
    graph,
    ['logits'],
    Int32Array.of(99, 99, 99, 99),
    {},
    { lossMask: Uint8Array.of(0, 0, 0, 0) },
  );
  assert.equal(noOp.loss, 0);
  assert.equal(noOp.examples, 0);
  assert.deepEqual(noOp.updatedTensors, []);
  assert.equal(graph.trainingStep, 1, 'an all-masked no-op is not an optimizer step');

  await zeroRateGradients(
    graph,
    ['logits'],
    Int32Array.of(2, 1, 0, 1),
    {},
    { optimizer: { learningRate: 0, step: 7 } },
  );
  assert.equal(graph.trainingStep, 7);
  assert.equal(graph.optimizerDescriptor.updateMode, 'sgd');
  await assert.rejects(
    () => zeroRateGradients(
      graph,
      ['logits'],
      Int32Array.of(2, 1, 0, 1),
      {},
      { optimizer: { learningRate: 0, step: 7 } },
    ),
    /greater than graph\.trainingStep/,
  );
});

test('CPU embedding training accepts conventional Int32 token IDs', async () => {
  const graph = new Graph();
  const tokens = graph.addInput('tokens', [2], 'int32', { buffer: new Int32Array(2) });
  const table = addWeight(graph, 'table', [3, 3], [
    0.1, 0.2, 0.3,
    -0.2, 0.4, 0.7,
    0.9, -0.1, 0.5,
  ]);
  const { out } = graph.addOp('Embedding', { input: tokens, weight: table }, { out: [2, 3] });
  graph.outputNames = [out.name];
  const result = await zeroRateGradients(
    graph, ['table'], Int32Array.of(1, 2), { tokens: Int32Array.of(0, 2) },
  );
  assert.deepEqual([...result.gradients.get('table').slice(3, 6)], [0, 0, 0]);
  assert.ok(result.gradients.get('table').slice(0, 3).some((value) => value !== 0));
  assert.ok(result.gradients.get('table').slice(6, 9).some((value) => value !== 0));
});

test('CPU optimizer descriptors are validated before any mutation', async () => {
  const graph = new Graph();
  const logits = addWeight(graph, 'logits', [1, 2], [0.25, -0.25]);
  graph.outputNames = [logits.name];
  const before = new Float32Array(logits.buffer);
  await assert.rejects(
    () => CPUAutograd.trainStep(graph, {
      targets: [1],
      trainableTensors: ['logits'],
      updateMode: 'adamw',
      optimizer: [],
    }),
    /must be an object/,
  );
  assert.deepEqual(logits.buffer, before);
  assert.equal(graph.trainingStep, 0);
  assert.equal(graph.optimizerState, null);
  assert.equal(graph.optimizerDescriptor, null);
});

test('CPU inputs require exact graph dtype and cannot target parameters', async () => {
  const graph = new Graph();
  const ids = graph.addInput('ids', [1], 'int32');
  const table = addWeight(graph, 'table', [2, 2], [1, 0, 0, 1]);
  const { out } = graph.addOp('Embedding', { input: ids, weight: table }, { out: [1, 2] });
  graph.outputNames = [out.name];
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await assert.rejects(() => engine.execute({ ids: Float32Array.of(1) }), /typed storage/);
  await assert.rejects(() => engine.execute({ table: new Float32Array(4) }), /Unknown graph input/);
  await assert.rejects(
    () => CPUAutograd.trainStep(graph, {
      inputs: { ids: Int32Array.of(1) },
      targets: [0],
      trainableTensors: ['ids'],
      optimizer: { learningRate: 0 },
    }),
    /graph weight/,
  );
});
