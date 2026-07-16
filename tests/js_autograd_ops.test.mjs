import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine, Graph } from '../ts/index.js';
import { CPUAutograd } from '../ts/training/index.js';
import { geluValue } from '../ts/ops/gELU.js';

function addWeight(graph, name, shape, values) {
  const tensor = graph.addWeight(name, shape);
  tensor.buffer = Float32Array.from(values);
  return tensor;
}

function close(actual, expected, label, absolute = 2e-4, relative = 2e-3) {
  const tolerance = absolute + relative * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: analytic=${actual}, numeric=${expected}, tolerance=${tolerance}`);
}

async function crossEntropy(graph, engine, targets, inputs = {}) {
  await engine.execute(inputs);
  const logits = graph.getTensor(graph.outputNames[0]);
  const classes = logits.shape[logits.shape.length - 1];
  const rows = logits.buffer.length / classes;
  assert.equal(targets.length, rows);
  let loss = 0;
  for (let row = 0; row < rows; row++) {
    let max = -Infinity;
    for (let col = 0; col < classes; col++) max = Math.max(max, logits.buffer[row * classes + col]);
    let denominator = 0;
    for (let col = 0; col < classes; col++) denominator += Math.exp(logits.buffer[row * classes + col] - max);
    loss += max + Math.log(denominator) - logits.buffer[row * classes + targets[row]];
  }
  return loss / rows;
}

async function numericGradient(graph, engine, tensorName, index, targets, inputs = {}, epsilon = 1e-3) {
  const tensor = graph.getTensor(tensorName);
  const original = tensor.buffer[index];
  tensor.buffer[index] = original + epsilon;
  const positive = await crossEntropy(graph, engine, targets, inputs);
  tensor.buffer[index] = original - epsilon;
  const negative = await crossEntropy(graph, engine, targets, inputs);
  tensor.buffer[index] = original;
  return (positive - negative) / (2 * epsilon);
}

async function gradients(graph, trainableTensors, targets, inputs = {}) {
  const result = await CPUAutograd.trainStep(graph, {
    inputs,
    targets,
    trainableTensors,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return { result, engine };
}

test('CPU training selects one sequence row per batch target', async () => {
  const graph = new Graph();
  const logits = addWeight(graph, 'logits', [2, 2, 2], new Array(8).fill(0));
  graph.outputNames = [logits.name];
  const options = {
    targets: [1, 0],
    trainableTensors: ['logits'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  };

  const finalPosition = await CPUAutograd.trainStep(graph, options);
  assert.deepEqual([...finalPosition.gradients.get('logits')], [
    0, 0, 0.25, -0.25,
    0, 0, -0.25, 0.25,
  ]);

  const firstPosition = await CPUAutograd.trainStep(graph, { ...options, lastToken: 0 });
  assert.deepEqual([...firstPosition.gradients.get('logits')], [
    0.25, -0.25, 0, 0,
    -0.25, 0.25, 0, 0,
  ]);

  await assert.rejects(
    () => CPUAutograd.trainStep(graph, { ...options, lastToken: 2 }),
    /2 logits rows per batch/,
  );
});

test('activation backward formulas match finite differences', async (t) => {
  for (const opType of ['ReLU', 'GELU', 'SiLU', 'Swish', 'Sigmoid', 'Tanh', 'LeakyReLU']) {
    await t.test(opType, async () => {
      const graph = new Graph();
      const parameter = addWeight(graph, 'parameter', [1, 3], [-0.7, 0.2, 1.1]);
      const { out } = graph.addOp(opType, { input: parameter }, { out: [1, 3] }, { alpha: 0.2 });
      graph.outputNames = [out.name];
      const targets = [1];
      const { result, engine } = await gradients(graph, ['parameter'], targets);
      for (let index = 0; index < parameter.buffer.length; index++) {
        const numeric = await numericGradient(graph, engine, 'parameter', index, targets);
        close(result.gradients.get('parameter')[index], numeric, `${opType}[${index}]`);
      }
    });
  }
});

test('CPU GELU defaults to erf semantics and retains explicit tanh approximation', async () => {
  const values = Float32Array.of(-3, -1, 0, 1, 3);
  const run = async (params) => {
    const graph = new Graph();
    const input = graph.addInput('input', [values.length]);
    const { out } = graph.addOp('GELU', { input }, { out: [values.length] }, params);
    graph.outputNames = [out.name];
    const engine = new CPUEngine();
    engine.allocateGraph(graph);
    await engine.execute({ input: values });
    return [...out.buffer];
  };
  const exact = await run({});
  const tanh = await run({ approximate: 'tanh' });
  for (let index = 0; index < values.length; index++) {
    close(exact[index], geluValue(values[index], 'none'), `exact GELU[${index}]`, 2e-7, 2e-7);
    close(tanh[index], geluValue(values[index], 'tanh'), `tanh GELU[${index}]`, 2e-7, 2e-7);
  }
  assert.ok(Math.abs(exact[1] - tanh[1]) > 1e-4, 'the two GELU modes must remain distinguishable');
  await assert.rejects(() => run({ approximate: 'fast' }), /'none' or 'tanh'/);
});

test('Embedding scatters gradients only into selected table rows', async () => {
  const graph = new Graph();
  const token = graph.addInput('token', [1], 'int32');
  const table = addWeight(graph, 'table', [3, 3], [
    0.1, 0.2, 0.3,
    -0.2, 0.4, 0.7,
    0.9, -0.1, 0.5,
  ]);
  const { out } = graph.addOp('Embedding', { input: token, weight: table }, { out: [1, 3] });
  graph.outputNames = [out.name];
  const inputs = { token: Int32Array.of(1) };
  const targets = [2];
  const { result, engine } = await gradients(graph, ['table'], targets, inputs);
  const gradient = result.gradients.get('table');
  assert.deepEqual([...gradient.slice(0, 3)], [0, 0, 0]);
  assert.deepEqual([...gradient.slice(6, 9)], [0, 0, 0]);
  for (let index = 3; index < 6; index++) {
    const numeric = await numericGradient(graph, engine, 'table', index, targets, inputs);
    close(gradient[index], numeric, `Embedding table[${index}]`);
  }
});

test('LayerNorm and RMSNorm propagate input and affine gradients', async (t) => {
  await t.test('LayerNorm', async () => {
    const graph = new Graph();
    const input = addWeight(graph, 'input', [1, 3], [-0.4, 0.2, 1.3]);
    const weight = addWeight(graph, 'weight', [3], [1.2, 0.8, 1.1]);
    const bias = addWeight(graph, 'bias', [3], [0.1, -0.2, 0.3]);
    const { out } = graph.addOp('LayerNorm', { input, weight, bias }, { out: [1, 3] }, { d_model: 3, eps: 0.2 });
    graph.outputNames = [out.name];
    const targets = [1];
    const { result, engine } = await gradients(graph, ['input', 'weight', 'bias'], targets);
    for (const [name, index] of [['input', 1], ['weight', 1], ['bias', 1]]) {
      const numeric = await numericGradient(graph, engine, name, index, targets);
      close(result.gradients.get(name)[index], numeric, `LayerNorm ${name}[${index}]`);
    }
  });

  await t.test('RMSNorm', async () => {
    const graph = new Graph();
    const input = addWeight(graph, 'input', [1, 3], [-0.4, 0.2, 1.3]);
    const weight = addWeight(graph, 'weight', [3], [1.2, 0.8, 1.1]);
    const { out } = graph.addOp('RMSNorm', { input, weight }, { out: [1, 3] }, { d_model: 3, eps: 0.15 });
    graph.outputNames = [out.name];
    const targets = [1];
    const { result, engine } = await gradients(graph, ['input', 'weight'], targets);
    for (const [name, index] of [['input', 1], ['weight', 1]]) {
      const numeric = await numericGradient(graph, engine, name, index, targets);
      close(result.gradients.get(name)[index], numeric, `RMSNorm ${name}[${index}]`);
    }
  });
});

test('last-axis ReduceSum and ReduceMean propagate every outer-row gradient', async (t) => {
  for (const opType of ['ReduceSum', 'ReduceMean']) {
    await t.test(opType, async () => {
      const graph = new Graph();
      const parameter = addWeight(graph, 'parameter', [2, 2, 3], [
        0.1, -0.2, 0.3, 0.4, 0.2, -0.5,
        -0.3, 0.6, 0.7, 0.8, -0.4, 0.2,
      ]);
      const logits = graph.addOp(opType, { input: parameter }, { out: [2, 2] }).out;
      graph.outputNames = [logits.name];
      const targets = [1, 0];
      const { result, engine } = await gradients(graph, ['parameter'], targets);
      for (let index = 0; index < parameter.buffer.length; index++) {
        const numeric = await numericGradient(graph, engine, 'parameter', index, targets);
        close(result.gradients.get('parameter')[index], numeric, `${opType}[${index}]`);
      }
    });
  }
});

test('SDPA and CrossSDPA gradients match finite differences', async (t) => {
  await t.test('causal SDPA', async () => {
    const graph = new Graph();
    const qkv = addWeight(graph, 'qkv', [1, 2, 6], [
      0.2, -0.1, 0.3, 0.4, 0.5, -0.2,
      -0.3, 0.6, 0.1, -0.5, 0.7, 0.2,
    ]);
    const { out } = graph.addOp('SDPA', { qkv }, { out: [1, 2, 2] }, { heads: 1 });
    graph.outputNames = [out.name];
    const targets = [0, 1];
    const { result, engine } = await gradients(graph, ['qkv'], targets);
    for (const index of [6, 8, 9, 11]) {
      const numeric = await numericGradient(graph, engine, 'qkv', index, targets);
      close(result.gradients.get('qkv')[index], numeric, `SDPA qkv[${index}]`, 3e-4, 3e-3);
    }
  });

  await t.test('CrossSDPA', async () => {
    const graph = new Graph();
    const q = addWeight(graph, 'q', [1, 2, 2], [0.2, -0.1, -0.3, 0.6]);
    const k = addWeight(graph, 'k', [1, 2, 2], [0.3, 0.4, 0.1, -0.5]);
    const v = addWeight(graph, 'v', [1, 2, 2], [0.5, -0.2, 0.7, 0.2]);
    const { out } = graph.addOp('CrossSDPA', { q, k, v }, { out: [1, 2, 2] }, { heads: 1, scale: 0.37 });
    graph.outputNames = [out.name];
    const targets = [0, 1];
    const { result, engine } = await gradients(graph, ['q', 'k', 'v'], targets);
    for (const [name, index] of [['q', 3], ['k', 0], ['v', 3]]) {
      const numeric = await numericGradient(graph, engine, name, index, targets);
      close(result.gradients.get(name)[index], numeric, `CrossSDPA ${name}[${index}]`, 3e-4, 3e-3);
    }
  });
});

test('JavaScript attention trains independent examples in a batch', async (t) => {
  await t.test('SDPA', async () => {
    const graph = new Graph();
    const qkv = addWeight(graph, 'qkv', [2, 1, 6], [
      0.2, 0.1, -0.3, 0.4, 0.5, -0.2,
      -0.4, 0.7, 0.2, -0.1, 0.3, 0.6,
    ]);
    const { out } = graph.addOp('SDPA', { qkv }, { out: [2, 1, 2] }, { heads: 1 });
    graph.outputNames = [out.name];

    const targets = [0, 1];
    const { result, engine } = await gradients(graph, ['qkv'], targets);
    for (const index of [4, 10]) {
      const numeric = await numericGradient(graph, engine, 'qkv', index, targets);
      close(result.gradients.get('qkv')[index], numeric,
        `batched SDPA qkv[${index}]`, 3e-4, 3e-3);
    }
  });

  await t.test('CrossSDPA', async () => {
    const graph = new Graph();
    const q = addWeight(graph, 'q', [2, 2, 2], [
      0.2, -0.1, -0.3, 0.6,
      -0.4, 0.7, 0.8, -0.2,
    ]);
    const k = addWeight(graph, 'k', [2, 3, 2], [
      0.3, 0.4, 0.1, -0.5, -0.2, 0.8,
      0.6, -0.1, -0.4, 0.2, 0.7, 0.3,
    ]);
    const v = addWeight(graph, 'v', [2, 3, 2], [
      0.5, -0.2, 0.7, 0.2, -0.6, 0.1,
      -0.3, 0.9, 0.4, 0.6, 0.2, -0.7,
    ]);
    const { out } = graph.addOp(
      'CrossSDPA', { q, k, v }, { out: [2, 2, 2] }, { heads: 1, scale: 0.37 },
    );
    graph.outputNames = [out.name];

    const targets = [0, 1, 1, 0];
    const { result, engine } = await gradients(graph, ['q', 'k', 'v'], targets);
    for (const [name, index] of [['q', 7], ['k', 8], ['v', 10]]) {
      const numeric = await numericGradient(graph, engine, name, index, targets);
      close(result.gradients.get(name)[index], numeric,
        `batched CrossSDPA ${name}[${index}]`, 3e-4, 3e-3);
    }
  });
});

async function convolutionCase({ depthwise }) {
  const graph = new Graph();
  const inputValues = depthwise
    ? [0.5, -0.4, 0.8, 0.3]
    : [0.5, -0.4, 0.2, 0.7, 0.8, 0.3, -0.6, 0.1];
  const input = addWeight(graph, 'input', [1, 1, 2, depthwise ? 2 : 4], inputValues);
  const weightShape = depthwise ? [1, 1, 2, 2] : [1, 1, 2, 4];
  const weightSize = weightShape.reduce((a, b) => a * b, 1);
  const weight = addWeight(graph, 'weight', weightShape,
    Array.from({ length: weightSize }, (_, index) => (index % 7 - 3) * 0.11));
  const bias = addWeight(graph, 'bias', [4], [0.2, 0.1, 0.3, 0.4]);
  const { out } = graph.addOp('Conv2D', { input, weight, bias }, { out: [1, 1, 2, 4] }, {
    groups: 2,
    relu: depthwise ? 2 : 1,
  });
  graph.outputNames = [out.name];
  const targets = [0, 3];
  const { result, engine } = await gradients(graph, ['input', 'weight', 'bias'], targets);
  const probes = depthwise
    ? [['input', 0], ['weight', 0], ['bias', 2]]
    : [['input', 2], ['weight', 2], ['bias', 3]];
  for (const [name, index] of probes) {
    const numeric = await numericGradient(graph, engine, name, index, targets);
    close(result.gradients.get(name)[index], numeric,
      `${depthwise ? 'depthwise' : 'grouped'} Conv2D ${name}[${index}]`, 3e-4, 3e-3);
  }
}

test('Conv2D backward covers grouped, depthwise, and fused activation paths', async (t) => {
  await t.test('grouped with fused ReLU', () => convolutionCase({ depthwise: false }));
  await t.test('depthwise with fused ReLU6', () => convolutionCase({ depthwise: true }));
});

test('MoERouter and MoELinear jointly backpropagate selected expert routes', async () => {
  const graph = new Graph();
  const input = addWeight(graph, 'input', [1, 2], [0.7, -0.2]);
  const router = addWeight(graph, 'router', [2, 3], [0.9, 0.3, -0.8, -0.4, 0.7, 0.2]);
  const routerBias = addWeight(graph, 'router_bias', [3], [0.1, -0.05, 0]);
  const routes = graph.addOp('MoERouter', { input, weight: router, bias: routerBias }, {
    indices: [1, 2],
    weights: [1, 2],
  }, { num_experts: 3, top_k: 2 });
  const experts = addWeight(graph, 'experts', [3, 2, 2], [
    0.4, -0.2, 0.1, 0.6,
    -0.3, 0.8, 0.5, -0.1,
    0.9, 0.2, -0.7, 0.3,
  ]);
  const expertBias = addWeight(graph, 'expert_bias', [3, 2], [0.1, 0, -0.1, 0.2, 0.05, -0.2]);
  const { out } = graph.addOp('MoELinear', {
    input,
    expert_weight: experts,
    expert_bias: expertBias,
    route_indices: routes.indices,
    route_weights: routes.weights,
  }, { out: [1, 2] });
  graph.outputNames = [out.name];

  const names = ['input', 'router', 'router_bias', 'experts', 'expert_bias'];
  const targets = [1];
  const { result, engine } = await gradients(graph, names, targets);
  assert.deepEqual([...routes.indices.buffer], [0, 1]);
  for (const [name, index] of [
    ['input', 0], ['router', 0], ['router_bias', 0], ['experts', 0], ['expert_bias', 0],
  ]) {
    const numeric = await numericGradient(graph, engine, name, index, targets);
    close(result.gradients.get(name)[index], numeric, `MoE ${name}[${index}]`, 3e-4, 3e-3);
  }
  // The hard top-k decision is not differentiated; an unselected expert has
  // no router or expert-parameter gradient for this token.
  assert.equal(result.gradients.get('router')[2], 0);
  assert.equal(result.gradients.get('expert_bias')[4], 0);

  graph.nodes[0].params.normalize = false;
  const fullSoftmax = await gradients(graph, names, targets);
  const unselectedNumeric = await numericGradient(graph, fullSoftmax.engine, 'router', 2, targets);
  close(fullSoftmax.result.gradients.get('router')[2], unselectedNumeric,
    'MoE full-softmax unselected router gradient', 3e-4, 3e-3);
  assert.notEqual(fullSoftmax.result.gradients.get('router')[2], 0);
});

test('explicit graph LoRA A/B branches are trainable', async () => {
  const graph = new Graph();
  const input = graph.addInput('x', [1, 2]);
  const base = addWeight(graph, 'base', [2, 2], [1, 0, 0, 1]);
  const a = addWeight(graph, 'lora_a', [2, 1], [0.4, -0.3]);
  const b = addWeight(graph, 'lora_b', [1, 2], [0.2, -0.5]);
  const baseOut = graph.addOp('MatMul', { input, weight: base }, { out: [1, 2] }).out;
  graph.nodes.at(-1).wLayout = 'din';
  const lowRank = graph.addOp('MatMul', { input, weight: a }, { out: [1, 1] }).out;
  graph.nodes.at(-1).wLayout = 'din';
  const delta = graph.addOp('MatMul', { input: lowRank, weight: b }, { out: [1, 2] }).out;
  graph.nodes.at(-1).wLayout = 'din';
  const logits = graph.addOp('Add', { a: baseOut, b: delta }, { out: [1, 2] }).out;
  graph.outputNames = [logits.name];

  const beforeA = new Float32Array(a.buffer);
  const beforeB = new Float32Array(b.buffer);
  const result = await CPUAutograd.trainStep(graph, {
    inputs: { x: Float32Array.from([0.7, -0.2]) },
    targets: [1],
    trainableTensors: ['lora_a', 'lora_b'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.05 },
  });
  assert.equal(result.updatedTensors.length, 2);
  assert.notDeepEqual(a.buffer, beforeA);
  assert.notDeepEqual(b.buffer, beforeB);
});

test('CPU training uses the base route when a staged adapter is active', async () => {
  const graph = new Graph();
  const input = addWeight(graph, 'input', [1, 2], [0.7, -0.2]);
  const weight = addWeight(graph, 'weight', [2, 2], [1, 0, 0, 1]);
  const { out } = graph.addOp('MatMul', { input, weight }, { out: [1, 2] });
  graph.nodes[0].wLayout = 'din';
  graph.outputNames = [out.name];
  graph.stageAdapter('active', {
    kind: 'lora',
    targets: [{
      weight: 'weight', rank: 1, alpha: 1,
      A: Float32Array.from([8, -6]), B: Float32Array.from([5, -7]),
    }],
  }, { activate: true });

  const trained = await gradients(graph, ['input'], [1]);
  graph.activateAdapter(null);
  const numeric = await numericGradient(graph, trained.engine, 'input', 0, [1]);
  close(trained.result.gradients.get('input')[0], numeric, 'base-only active-adapter gradient');
});

test('CPU training validates optimizer and trainables before mutation', async () => {
  const graph = new Graph();
  const parameter = addWeight(graph, 'parameter', [1, 2], [0.2, -0.4]);
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.outputNames = [out.name];
  const before = new Float32Array(parameter.buffer);

  await assert.rejects(() => CPUAutograd.trainStep(graph, {
    targets: [0], trainableTensors: ['parameter', 'missing'], updateMode: 'sgd',
  }), /Trainable tensor 'missing'/);
  assert.deepEqual(parameter.buffer, before);
  await assert.rejects(() => CPUAutograd.trainStep(graph, {
    targets: [0], trainableTensors: ['parameter', 'parameter'], updateMode: 'sgd',
  }), /must be unique/);
  await assert.rejects(() => CPUAutograd.trainStep(graph, {
    targets: [0], trainableTensors: ['parameter'], updateMode: 'assign',
  }), /must be SGD\/AdamW/);
  assert.deepEqual(parameter.buffer, before);
});

test('CPU training propagates through a reachable Softmax branch before updating weights', async () => {
  const graph = new Graph();
  const parameter = addWeight(graph, 'parameter', [1, 3], [0.2, -0.4, 0.7]);
  const supported = graph.addOp('ReLU', { input: parameter }, { out: [1, 3] }).out;
  const unsupported = graph.addOp('Softmax', { input: parameter }, { out: [1, 3] }).out;
  const logits = graph.addOp('Add', { a: supported, b: unsupported }, { out: [1, 3] }).out;
  graph.outputNames = [logits.name];
  const before = new Float32Array(parameter.buffer);

  const result = await CPUAutograd.trainStep(graph, {
    targets: [1],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.1 },
  });
  assert.equal(result.examples, 1);
  assert.notDeepEqual(parameter.buffer, before);
});

test('CPU training propagates through a reachable non-primary Split output', async () => {
  const graph = new Graph();
  const parameter = addWeight(graph, 'parameter', [1, 4], [0.2, -0.4, 0.7, -0.1]);
  const split = graph.addOp(
    'Split',
    { input: parameter },
    { left: [1, 2], right: [1, 2] },
    { axis: 1 },
  );
  const logits = graph.addOp('Identity', { input: split.right }, { out: [1, 2] }).out;
  graph.outputNames = [logits.name];
  const before = new Float32Array(parameter.buffer);

  const result = await CPUAutograd.trainStep(graph, {
    targets: [1], trainableTensors: ['parameter'], updateMode: 'sgd', optimizer: { learningRate: 0.1 },
  });
  assert.equal(result.examples, 1);
  assert.notDeepEqual(parameter.buffer, before);
});

test('CPU training rejects non-finite gradients before mutating a trainable', async () => {
  const graph = new Graph();
  const parameter = addWeight(graph, 'parameter', [1, 2], [Number.NaN, 0.4]);
  const logits = graph.addOp('Identity', { input: parameter }, { out: [1, 2] }).out;
  graph.outputNames = [logits.name];
  const revision = graph.weightRevision;
  const finiteValue = parameter.buffer[1];

  await assert.rejects(() => CPUAutograd.trainStep(graph, {
    targets: [1], trainableTensors: ['parameter'], updateMode: 'sgd', optimizer: { learningRate: 0.1 },
  }), /Gradient.*non-finite/);
  assert.ok(Number.isNaN(parameter.buffer[0]));
  assert.equal(parameter.buffer[1], finiteValue);
  assert.equal(graph.weightRevision, revision);
});
