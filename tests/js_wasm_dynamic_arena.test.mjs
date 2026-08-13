import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { createBackendCompileInput } from '../ts/backends/BackendProvider.js';
import { CPUBackendProvider } from '../ts/backends/CPUBackendProvider.js';
import { WasmBackendProvider } from '../ts/backends/WasmBackendProvider.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { Runtime } from '../ts/core/ContextRuntime.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';
import { geluValue } from '../ts/ops/gELU.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

async function buildForwardWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
}

function dynamicLinearGraph(rows) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [rows, 3]);
  const residual = graph.addInput('residual', [rows, 2]);
  const weight = graph.addWeight('weight', [3, 2], 'float32', {
    buffer: Float32Array.of(0.25, -0.5, 1.5, 0.75, -1, 0.5),
  });
  const bias = graph.addWeight('bias', [2], 'float32', {
    buffer: Float32Array.of(0.125, -0.25),
  });
  const { out: projected } = graph.addOp('Linear', { input, weight, bias }, {
    out: { name: 'projected', shape: [rows, 2] },
  }, { weight_layout: 'IN_OUT' });
  const { out } = graph.addOp('Add', { a: projected, b: residual }, {
    out: { name: 'out', shape: [rows, 2] },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function livenessWaveGraph(rows, { publicTap = false, identityView = false } = {}) {
  const graph = new RuntimeGraph();
  const shape = [rows, 4];
  const x = graph.addInput('live.x', shape);
  const y = graph.addInput('live.y', shape);
  const { out: sum } = graph.addOp('Add', { a: x, b: y }, {
    out: { name: 'live.sum', shape },
  });
  const { out: relu } = graph.addOp('ReLU', { input: sum }, {
    out: { name: 'live.relu', shape },
  });
  const geluInput = identityView
    ? graph.addOp('Identity', { input: relu }, {
      out: { name: 'live.view', shape },
    }).out
    : relu;
  const { out: gelu } = graph.addOp('GELU', { input: geluInput }, {
    out: { name: 'live.gelu', shape },
  }, { approximate: 'none' });
  const { out } = graph.addOp('Add', { a: gelu, b: y }, {
    out: { name: 'live.out', shape },
  });
  graph.setOutputs(publicTap ? [relu.name, out.name] : [out.name]);
  return graph;
}

function invariantWeightViewGraph() {
  const graph = new RuntimeGraph();
  const constant = graph.addWeight('constant', [2, 2], 'float32', {
    buffer: Float32Array.of(1.25, -2.5, 3.75, -4),
  });
  const { out: identity } = graph.addOp('Identity', { input: constant }, {
    out: { name: 'constant.view', shape: [2, 2] },
  });
  const { out: reshaped } = graph.addOp('Reshape', { input: identity }, {
    out: { name: 'constant.reshaped', shape: [4] },
  }, { shape: [4] });
  graph.setOutputs([reshaped.name]);
  return graph;
}

function livenessInputs(rows, salt) {
  return {
    'live.x': Float32Array.from(
      { length: rows * 4 }, (_, index) => Math.fround(((index + salt) % 13 - 6) / 7),
    ),
    'live.y': Float32Array.from(
      { length: rows * 4 }, (_, index) => Math.fround(((index * 3 + salt) % 17 - 8) / 9),
    ),
  };
}

async function cpuLivenessOutputs(graph, values) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(values);
}

function dynamicIdentitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      S: { min: 1, max: 3 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S'] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function staticIdentitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2, 2] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2, 2] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function partiallyResidentBankSnapshot() {
  const expertCount = 4;
  const inputFeatures = 2;
  const outputFeatures = 1;
  const descriptor = {
    name: 'experts', dtype: 'float32',
    shape: [expertCount, inputFeatures, outputFeatures],
  };
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { F: { min: 1, max: expertCount } },
    banks: { experts: 'F' },
    inputs: {
      x: { dtype: 'float32', shape: [1, inputFeatures] },
      route_indices: { dtype: 'float32', shape: [1, 1] },
      route_weights: { dtype: 'float32', shape: [1, 1] },
    },
    nodes: [{
      id: 'mix',
      opType: 'MoELinear',
      inputs: {
        input: 'x',
        expert_weight: 'experts',
        route_indices: 'route_indices',
        route_weights: 'route_weights',
      },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: [1, outputFeatures] },
      },
      params: {},
    }],
    outputs: ['y'],
  }, [descriptor]);
  const data = new Float32Array(expertCount * inputFeatures * outputFeatures);
  for (let expert = 0; expert < expertCount; expert++) {
    data[expert * inputFeatures] = expert + 1;
    data[expert * inputFeatures + 1] = expert + 1;
  }
  return Model.capture({
    graph,
    weights: { experts: { ...descriptor, data } },
  });
}

function bankRouteInputs(expert) {
  return {
    x: { data: Float32Array.of(1, 0), shape: [1, 2] },
    route_indices: { data: Float32Array.of(expert), shape: [1, 1] },
    route_weights: { data: Float32Array.of(1), shape: [1, 1] },
  };
}

function dynamicDecodeGeluSnapshot(approximation = 'none') {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { S: { min: 2, max: 8 } },
    inputs: { x: { dtype: 'float32', shape: [1, 'S', 4] } },
    nodes: [{
      id: 'gelu',
      opType: 'GELU',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 'S', 4] } },
      params: { approximate: approximation },
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function decodeGeluInputs(sequence, salt = 0) {
  return {
    x: {
      data: Float32Array.from(
        { length: sequence * 4 },
        (_, index) => Math.fround(((index + salt) % 17 - 8) / 5),
      ),
      shape: [1, sequence, 4],
    },
  };
}

function replaceDecodeGeluRow(inputs, position, salt) {
  const data = new Float32Array(inputs.x.data);
  for (let feature = 0; feature < 4; feature++) {
    data[position * 4 + feature] = Math.fround((salt + feature) / 9);
  }
  return { x: { data, shape: inputs.x.shape } };
}

function referenceDecodeGelu(inputs, approximation = 'none') {
  return Float32Array.from(inputs.x.data,
    (value) => Math.fround(geluValue(value, approximation)));
}

function adversarialIdentitySnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 9 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 1] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 1] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function boundedIdentitySnapshot(dimensions, shape) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions,
    inputs: { x: { dtype: 'float32', shape } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function oversizedPlanMetadataSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [{
      id: 'i'.repeat(1024 * 1024 + 64),
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function dynamicLogicalLinearSnapshot() {
  const definitions = [
    {
      name: 'weight', dtype: 'float32', shape: [3, 2],
      data: Float32Array.of(0.25, -0.5, 1.5, 0.75, -1, 0.5),
    },
    {
      name: 'bias', dtype: 'float32', shape: [2],
      data: Float32Array.of(0.125, -0.25),
    },
  ];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { input: { dtype: 'float32', shape: ['B', 3] } },
    nodes: [{
      id: 'linear',
      opType: 'Linear',
      inputs: { input: 'input', weight: 'weight', bias: 'bias' },
      outputs: { out: { tensor: 'out', dtype: 'float32', shape: ['B', 2] } },
      params: { weight_layout: 'din_dout' },
    }],
    outputs: ['out'],
  }, definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })));
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [
      definition.name,
      { ...definition },
    ])),
  });
}

function dynamicWaveBSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      S: { min: 1, max: 5 },
    },
    inputs: {
      a: { dtype: 'float32', shape: ['B', 1, 4] },
      b: { dtype: 'float32', shape: [1, 'S', 4] },
      condition: { dtype: 'int32', shape: ['B', 'S', 1] },
    },
    nodes: [{
      id: 'subtract',
      opType: 'Sub',
      inputs: { a: 'a', b: 'b' },
      outputs: { out: { tensor: 'difference', dtype: 'float32', shape: ['B', 'S', 4] } },
      params: {},
    }, {
      id: 'select',
      opType: 'Where',
      inputs: { condition: 'condition', a: 'difference', b: 'a' },
      outputs: { out: { tensor: 'selected', dtype: 'float32', shape: ['B', 'S', 4] } },
      params: {},
    }, {
      id: 'mean',
      opType: 'ReduceMean',
      inputs: { input: 'selected' },
      outputs: { out: { tensor: 'mean', dtype: 'float32', shape: ['B', 'S', 1] } },
      params: { axis: -1, keepdims: true },
    }, {
      id: 'transpose',
      opType: 'Transpose',
      inputs: { input: 'selected' },
      outputs: { out: { tensor: 'transposed', dtype: 'float32', shape: ['S', 'B', 4] } },
      params: { perm: [1, 0, 2] },
    }],
    outputs: ['mean', 'transposed'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function dynamicConvSnapshot() {
  const definitions = [{
    name: 'weight', dtype: 'float32', shape: [3, 3, 2, 3],
    data: Float32Array.from(
      { length: 54 }, (_, index) => Math.fround((((index * 11) % 29) - 14) / 31),
    ),
  }, {
    name: 'bias', dtype: 'float32', shape: [3],
    data: Float32Array.of(0.125, -0.25, 0.375),
  }];
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 2 },
      H: { min: 3, max: 8 },
      W: { min: 3, max: 9 },
    },
    inputs: { input: { dtype: 'float32', shape: ['B', 'H', 'W', 2] } },
    nodes: [{
      id: 'conv',
      opType: 'Conv2D',
      inputs: { input: 'input', weight: 'weight', bias: 'bias' },
      outputs: { out: { tensor: 'out', dtype: 'float32', shape: ['B', 'H', 'W', 3] } },
      params: {
        stride: [1, 1], padding: [1, 1], dilation: [1, 1],
        groups: 1, weight_layout: 'HWIO', relu: 0,
      },
    }],
    outputs: ['out'],
  }, definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })));
  return Model.capture({
    graph,
    weights: Object.fromEntries(definitions.map((definition) => [
      definition.name, { ...definition },
    ])),
  });
}

function dynamicAttentionSnapshot() {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 2 },
      S: { min: 1, max: 4 },
      Q: { min: 1, max: 3 },
      K: { min: 1, max: 5 },
    },
    inputs: {
      qkv: { dtype: 'float32', shape: ['B', 'S', 12] },
      self_mask: { dtype: 'int32', shape: ['B', 'S', 'S'] },
      position_ids: { dtype: 'int32', shape: ['B', 'S'] },
      q: { dtype: 'float32', shape: ['B', 'Q', 4] },
      k: { dtype: 'float32', shape: ['B', 'K', 4] },
      v: { dtype: 'float32', shape: ['B', 'K', 4] },
      cross_mask: { dtype: 'int32', shape: ['B', 'Q', 'K'] },
    },
    nodes: [{
      id: 'self-attention',
      opType: 'SDPA',
      inputs: { qkv: 'qkv', mask: 'self_mask' },
      outputs: { out: { tensor: 'self_attended', dtype: 'float32', shape: ['B', 'S', 4] } },
      params: { heads: 2, causal: false },
    }, {
      id: 'rotary',
      opType: 'RoPE',
      inputs: { input: 'self_attended', position_ids: 'position_ids' },
      outputs: { out: { tensor: 'rotated', dtype: 'float32', shape: ['B', 'S', 4] } },
      params: { rotary_dim: 4, theta: 10000, position_offset: 0, interleaved: false },
    }, {
      id: 'cross-attention',
      opType: 'CrossSDPA',
      inputs: { q: 'q', k: 'k', v: 'v', mask: 'cross_mask' },
      outputs: { out: { tensor: 'cross', dtype: 'float32', shape: ['B', 'Q', 4] } },
      params: { heads: 2, causal: false },
    }],
    outputs: ['rotated', 'cross'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function referenceLinear(values) {
  const result = new Float32Array(values.length / 3 * 2);
  for (let row = 0; row < values.length / 3; row++) {
    const a = values[row * 3];
    const b = values[row * 3 + 1];
    const c = values[row * 3 + 2];
    result[row * 2] = Math.fround(a * 0.25 + b * 1.5 - c + 0.125);
    result[row * 2 + 1] = Math.fround(a * -0.5 + b * 0.75 + c * 0.5 - 0.25);
  }
  return result;
}

function dynamicQLinearGraph(rows) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [rows, 3], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const weight = graph.addWeight('weight', [2, 3], 'int8', {
    buffer: Int8Array.of(2, 0, -1, -2, 1, 3),
    quantization: {
      scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2],
    },
  });
  const bias = graph.addWeight('bias', [2], 'int32', {
    buffer: Int32Array.of(2, -4),
  });
  const { out } = graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name: 'out', shape: [rows, 2], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
    },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function qlinearInputs(rows, offset) {
  return {
    input: Int8Array.from(
      { length: rows * 3 }, (_, index) => ((index * 5 + offset) % 17) - 8,
    ),
  };
}

function waveBInputs(batch, sequence, salt) {
  return {
    a: {
      data: Float32Array.from(
        { length: batch * 4 }, (_, index) => Math.fround((index + salt) / 9),
      ),
      shape: [batch, 1, 4],
    },
    b: {
      data: Float32Array.from(
        { length: sequence * 4 }, (_, index) => Math.fround((salt - index) / 11),
      ),
      shape: [1, sequence, 4],
    },
    condition: {
      data: Int32Array.from(
        { length: batch * sequence }, (_, index) => (index + salt) % 3 === 0 ? 0 : 1,
      ),
      shape: [batch, sequence, 1],
    },
  };
}

function convInputs(batch, height, width, salt) {
  return {
    input: {
      data: Float32Array.from(
        { length: batch * height * width * 2 },
        (_, index) => Math.fround((((index * 7 + salt) % 41) - 20) / 23),
      ),
      shape: [batch, height, width, 2],
    },
  };
}

function attentionInputs(batch, sequence, queries, keys, salt) {
  const floats = (length, multiplier) => Float32Array.from(
    { length },
    (_, index) => Math.fround((((index * multiplier + salt) % 31) - 15) / 19),
  );
  return {
    qkv: { data: floats(batch * sequence * 12, 7), shape: [batch, sequence, 12] },
    self_mask: {
      data: Int32Array.from(
        { length: batch * sequence * sequence },
        (_, index) => (index + salt) % 5 === 0 ? 0 : 1,
      ),
      shape: [batch, sequence, sequence],
    },
    position_ids: {
      data: Int32Array.from(
        { length: batch * sequence },
        (_, index) => index % sequence + (index >= sequence ? 1 : 0),
      ),
      shape: [batch, sequence],
    },
    q: { data: floats(batch * queries * 4, 11), shape: [batch, queries, 4] },
    k: { data: floats(batch * keys * 4, 13), shape: [batch, keys, 4] },
    v: { data: floats(batch * keys * 4, 17), shape: [batch, keys, 4] },
    cross_mask: {
      data: Int32Array.from(
        { length: batch * queries * keys },
        (_, index) => (index + salt) % 7 === 0 ? 0 : 1,
      ),
      shape: [batch, queries, keys],
    },
  };
}

async function cpuQLinearOutput(rows, values) {
  const engine = new CPUEngine();
  const graph = dynamicQLinearGraph(rows);
  engine.allocateGraph(graph);
  return (await engine.execute(values)).out;
}

function inputs(rows, offset) {
  return {
    input: Float32Array.from(
      { length: rows * 3 }, (_, index) => Math.fround((index + offset) / 7),
    ),
    residual: Float32Array.from(
      { length: rows * 2 }, (_, index) => Math.fround((offset - index) / 11),
    ),
  };
}

test('WASM provider rejects a stale sidecar without the reusable-arena ABI', async () => {
  const stale = new WasmEngine({ instance: { exports: {
    memory: new WebAssembly.Memory({ initial: 1 }),
    alloc_bytes: () => 16,
    reset_heap: () => {},
  } } });
  const provider = new WasmBackendProvider(stale);
  try {
    await assert.rejects(
      provider.compile(createBackendCompileInput(staticIdentitySnapshot()), {
        operatorFallback: 'forbid',
      }),
      /does not expose runtime ABI 1; rebuild the sidecar/,
    );
  } finally {
    await provider.close();
  }
});

async function cpuOutput(rows, values) {
  const engine = new CPUEngine();
  const graph = dynamicLinearGraph(rows);
  engine.allocateGraph(graph);
  return (await engine.execute(values)).out;
}

function assertClose(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    assert.ok(difference <= 5e-5 * Math.max(1, Math.abs(expected[index])),
      `${label}[${index}] got ${actual[index]}, expected ${expected[index]}`);
  }
}

test('WASM reusable arena alternates concrete shapes without repacking or leaking', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-dynamic-arena-'));
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await buildForwardWasm(wasmPath);
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm);
    assert.equal(wasm.api.wasm_runtime_abi_version(), 1);
    assert.equal(typeof wasm.api.heap_mark, 'function');
    assert.equal(typeof wasm.api.heap_rewind, 'function');

    await t.test('ordinary liveness arenas reuse only disjoint lifetimes and roll back atomically',
      async () => {
        const live = await WasmEngine.init(wasmPath);
        assert.ok(live);
        const small = livenessWaveGraph(1);
        const large = livenessWaveGraph(4);
        const maximums = Object.freeze(Object.fromEntries(
          [...large.tensors].map(([name, tensor]) => [name, tensor.sizeBytes]),
        ));
        live.compile(small, live.prepareGraph(small), {
          tensorMaximumBytes: maximums,
          activationStorage: 'liveness',
          shapeSignature: 'live-rows=1',
        });

        assert.equal(live.pointers.get('live.x'), live.pointers.get('live.relu'),
          'an input may be reused only after its final consumer');
        assert.equal(live.pointers.get('live.x'), live.pointers.get('live.out'),
          'the final output may reuse a topology-dead input region');
        assert.equal(live.pointers.get('live.sum'), live.pointers.get('live.gelu'));
        for (const [left, right] of [
          ['live.x', 'live.sum'],
          ['live.sum', 'live.relu'],
          ['live.relu', 'live.gelu'],
          ['live.y', 'live.out'],
          ['live.gelu', 'live.out'],
        ]) {
          assert.notEqual(live.pointers.get(left), live.pointers.get(right),
            `${left}/${right} overlap at a kernel boundary and need disjoint storage`);
        }
        const smallValues = livenessInputs(1, 3);
        const smallExpected = await cpuLivenessOutputs(livenessWaveGraph(1), smallValues);
        assertClose((await live.execute(smallValues))['live.out'], smallExpected['live.out'],
          'small liveness wave');
        await assert.rejects(
          live.execute(smallValues, {
            incremental: true,
            incrementalReset: true,
            changedInputs: ['live.x', 'live.y'],
          }),
          /incremental execution requires persistent activation storage/,
        );
        const smallArena = live.inspectArena();
        assert.deepEqual({
          storage: smallArena.activationStorage,
          logical: smallArena.logicalActivationBytes,
          required: smallArena.activationRequiredBytes,
          capacity: smallArena.activationCapacityBytes,
          reused: smallArena.activationReuseBytes,
          arenas: smallArena.activationArenaCount,
        }, {
          storage: 'liveness', logical: 96, required: 48, capacity: 48,
          reused: 48, arenas: 1,
        });

        live.rebindGraph(large, live.prepareGraph(large), {
          shapeSignature: 'live-rows=4',
        });
        const largeValues = livenessInputs(4, 7);
        const largeExpected = await cpuLivenessOutputs(livenessWaveGraph(4), largeValues);
        assertClose((await live.execute(largeValues))['live.out'], largeExpected['live.out'],
          'large liveness wave');
        assert.deepEqual(
          [live.inspectArena().activationRequiredBytes,
            live.inspectArena().activationCapacityBytes,
            live.inspectArena().activationGrowCount],
          [192, 192, 1],
        );

        const commit = live._commitStagedVariantResources;
        let failCommit = true;
        live._commitStagedVariantResources = function failAfterLivenessCommit(...args) {
          commit.apply(this, args);
          if (failCommit) {
            failCommit = false;
            throw new Error('injected liveness commit failure');
          }
        };
        try {
          assert.throws(
            () => live.rebindGraph(small, live.prepareGraph(small), {
              shapeSignature: 'live-rows=1-failed',
            }),
            /injected liveness commit failure/,
          );
        } finally {
          live._commitStagedVariantResources = commit;
        }
        assert.equal(live.inspectArena().currentSignature, 'live-rows=4');
        assert.deepEqual(
          [live.inspectArena().activationRequiredBytes,
            live.inspectArena().activationCapacityBytes],
          [192, 192],
        );
        assertClose((await live.execute(largeValues))['live.out'], largeExpected['live.out'],
          'restored large liveness wave');

        const outputs = await WasmEngine.init(wasmPath);
        assert.ok(outputs);
        const publicGraph = livenessWaveGraph(1, { publicTap: true });
        outputs.compile(publicGraph, outputs.prepareGraph(publicGraph), {
          activationStorage: 'liveness', shapeSignature: 'public-lifetimes',
        });
        assert.notEqual(outputs.pointers.get('live.relu'), outputs.pointers.get('live.out'),
          'all public outputs remain live through the snapshot boundary');
        const publicExpected = await cpuLivenessOutputs(
          livenessWaveGraph(1, { publicTap: true }), smallValues,
        );
        const publicActual = await outputs.execute(smallValues);
        assertClose(publicActual['live.relu'], publicExpected['live.relu'], 'public early output');
        assertClose(publicActual['live.out'], publicExpected['live.out'], 'public final output');
        assert.equal(outputs.inspectArena().activationRequiredBytes, 64);

        const views = await WasmEngine.init(wasmPath);
        assert.ok(views);
        const viewGraph = livenessWaveGraph(1, { identityView: true });
        views.compile(viewGraph, views.prepareGraph(viewGraph), {
          activationStorage: 'liveness', shapeSignature: 'view-lifetimes',
        });
        assert.equal(views.pointers.get('live.relu'), views.pointers.get('live.view'));
        const viewExpected = await cpuLivenessOutputs(
          livenessWaveGraph(1, { identityView: true }), smallValues,
        );
        assertClose((await views.execute(smallValues))['live.out'], viewExpected['live.out'],
          'storage-view lifetime union');
        assert.equal(views.inspectArena().activationRequiredBytes, 48);

        const invariantViews = await WasmEngine.init(wasmPath);
        assert.ok(invariantViews);
        const invariantViewGraph = invariantWeightViewGraph();
        invariantViews.compile(
          invariantViewGraph,
          invariantViews.prepareGraph(invariantViewGraph),
          { activationStorage: 'liveness', shapeSignature: 'invariant-weight-view' },
        );
        assert.equal(invariantViews.pointers.get('constant.view'),
          invariantViews.pointers.get('constant'));
        assert.equal(invariantViews.pointers.get('constant.reshaped'),
          invariantViews.pointers.get('constant'));
        assert.deepEqual(
          {
            persistent: invariantViews.inspectArena().persistentWeightBytes,
            logical: invariantViews.inspectArena().logicalActivationBytes,
            required: invariantViews.inspectArena().activationRequiredBytes,
            capacity: invariantViews.inspectArena().activationCapacityBytes,
            arenas: invariantViews.inspectArena().activationArenaCount,
          },
          { persistent: 16, logical: 32, required: 0, capacity: 0, arenas: 0 },
          'the view chain reuses the invariant prefix instead of entering an activation arena',
        );
        assert.deepEqual(
          (await invariantViews.execute({}))['constant.reshaped'],
          Float32Array.of(1.25, -2.5, 3.75, -4),
        );
      });

    const capacities = Object.freeze({
      input: 4 * 3 * Float32Array.BYTES_PER_ELEMENT,
      residual: 4 * 2 * Float32Array.BYTES_PER_ELEMENT,
      weight: 3 * 2 * Float32Array.BYTES_PER_ELEMENT,
      bias: 2 * Float32Array.BYTES_PER_ELEMENT,
      projected: 4 * 2 * Float32Array.BYTES_PER_ELEMENT,
      out: 4 * 2 * Float32Array.BYTES_PER_ELEMENT,
    });

    const small = dynamicLinearGraph(1);
    wasm.compile(small, wasm.prepareGraph(small), {
      tensorCapacityBytes: capacities,
      shapeSignature: 'rows=1',
    });
    const smallInputs = inputs(1, 2);
    assertClose((await wasm.execute(smallInputs)).out,
      await cpuOutput(1, smallInputs), 'initial small');
    const initialInspection = wasm.inspectArena();
    assert.equal(initialInspection.reusable, true);
    assert.equal(initialInspection.activationStorage, 'persistent',
      'the direct engine keeps conservative retained storage unless explicitly opted in');
    assert.equal(initialInspection.f32PackCount, 1);
    assert.equal(initialInspection.activationCapacityBytes,
      capacities.input + capacities.residual + capacities.projected + capacities.out);

    /* Explicit growth detaches every prior JS view. Rebinding must refresh all
     * tensors before metadata validation or dispatch. */
    wasm.mem.grow(1);
    const large = dynamicLinearGraph(4);
    wasm.rebindGraph(large, wasm.prepareGraph(large), { shapeSignature: 'rows=4' });
    const largeInputs = inputs(4, 5);
    assertClose((await wasm.execute(largeInputs)).out,
      await cpuOutput(4, largeInputs), 'large after grow');

    const heapBytes = wasm.mem.buffer.byteLength;
    const buildVariantResourcesForCount = wasm._buildVariantResources;
    let variantResourceBuilds = 0;
    wasm._buildVariantResources = function countedVariantBuild(graph, prepared) {
      variantResourceBuilds++;
      return buildVariantResourcesForCount.call(this, graph, prepared);
    };
    try {
      for (let iteration = 0; iteration < 8; iteration++) {
        const rows = iteration % 2 === 0 ? 1 : 4;
        const graph = rows === 1 ? small : large;
        wasm.rebindGraph(graph, wasm.prepareGraph(graph), {
          shapeSignature: `rows=${rows}`,
        });
        const values = inputs(rows, 20 + iteration);
        assertClose((await wasm.execute(values)).out,
          await cpuOutput(rows, values), `alternate ${iteration}`);
        assert.equal(wasm.mem.buffer.byteLength, heapBytes,
          'variant rebuilding must reuse the same bounded heap region');
      }
    } finally {
      wasm._buildVariantResources = buildVariantResourcesForCount;
    }
    assert.equal(variantResourceBuilds, 8,
      'each rebind must construct descriptors once, then replay staged metadata at commit');

    const retainedBeforeInjectedFailure = inputs(4, 37);
    const growFor = wasm._growFor;
    let rejectMeasuredGrowth = true;
    wasm._growFor = function injectedGrowFailure(needed) {
      if (rejectMeasuredGrowth) {
        rejectMeasuredGrowth = false;
        throw new Error('injected WASM growth failure');
      }
      return growFor.call(this, needed);
    };
    try {
      assert.throws(
        () => wasm.rebindGraph(small, wasm.prepareGraph(small), {
          shapeSignature: 'rows=1-injected-grow-failure',
        }),
        /injected WASM growth failure/,
      );
    } finally {
      wasm._growFor = growFor;
    }
    assert.equal(wasm.inspectArena().currentSignature, 'rows=4');
    assertClose((await wasm.execute(retainedBeforeInjectedFailure)).out,
      await cpuOutput(4, retainedBeforeInjectedFailure),
      'previous variant after pre-commit growth failure');

    const commitStagedVariantResources = wasm._commitStagedVariantResources;
    let rejectActualCommit = true;
    wasm._commitStagedVariantResources = function injectedCommitFailure(...args) {
      if (rejectActualCommit) {
        rejectActualCommit = false;
        commitStagedVariantResources.apply(this, args);
        throw new Error('injected WASM post-rewind specialization failure');
      }
      return commitStagedVariantResources.apply(this, args);
    };
    try {
      assert.throws(
        () => wasm.rebindGraph(small, wasm.prepareGraph(small), {
          shapeSignature: 'rows=1-injected-commit-failure',
        }),
        /injected WASM post-rewind specialization failure/,
      );
    } finally {
      wasm._commitStagedVariantResources = commitStagedVariantResources;
    }
    assert.equal(wasm.inspectArena().currentSignature, 'rows=4');
    assertClose((await wasm.execute(retainedBeforeInjectedFailure)).out,
      await cpuOutput(4, retainedBeforeInjectedFailure),
      'previous variant after post-rewind specialization failure');

    const beforeFailure = wasm.inspectArena();
    const oversized = dynamicLinearGraph(5);
    assert.throws(
      () => wasm.rebindGraph(oversized, wasm.prepareGraph(oversized), {
        shapeSignature: 'rows=5',
      }),
      /exceeds or disagrees|incompatible with the compiled arena/,
    );
    const retainedInputs = inputs(4, 41);
    assertClose((await wasm.execute(retainedInputs)).out,
      await cpuOutput(4, retainedInputs), 'previous variant after failed preflight');

    const inspection = wasm.inspectArena();
    assert.equal(inspection.currentSignature, 'rows=4');
    assert.equal(inspection.f32PackCount, 1, 'shape changes must not repack F32 weights');
    assert.equal(inspection.q8PackCount, 0);
    assert.equal(inspection.variantRebindCount, 9);
    assert.equal(inspection.variantBytes, inspection.variantHighWaterBytes);
    assert.equal(inspection.heapBytes, heapBytes);
    assert.ok(inspection.variantHighWaterBytes > 0);
    assert.deepEqual(
      [inspection.persistentWeightBytes, inspection.activationCapacityBytes],
      [capacities.weight + capacities.bias,
        capacities.input + capacities.residual + capacities.projected + capacities.out],
    );
    assert.equal(beforeFailure.heapBytes, inspection.heapBytes);

    await t.test('quantized invariant panels survive alternating concrete shapes', async () => {
      const quantized = await WasmEngine.init(wasmPath);
      assert.ok(quantized);
      const smallQ = dynamicQLinearGraph(1);
      const largeQ = dynamicQLinearGraph(4);
      const capacities = Object.freeze({ input: 12, weight: 6, bias: 8, out: 8 });
      quantized.compile(smallQ, quantized.prepareGraph(smallQ), {
        tensorCapacityBytes: capacities,
        tensorMaximumBytes: capacities,
        shapeSignature: 'qrows=1',
      });
      for (let iteration = 0; iteration < 6; iteration++) {
        const rows = iteration % 2 === 0 ? 4 : 1;
        const graph = rows === 4 ? largeQ : smallQ;
        quantized.rebindGraph(graph, quantized.prepareGraph(graph), {
          shapeSignature: `qrows=${rows}`,
        });
        const values = qlinearInputs(rows, iteration + 3);
        assert.deepEqual(
          [...(await quantized.execute(values)).out],
          [...await cpuQLinearOutput(rows, values)],
        );
      }
      const arena = quantized.inspectArena();
      assert.equal(arena.q8PackCount, 1);
      assert.ok(arena.packedWeightBytes > 0);
      assert.equal(arena.activationGrowCount, 0);
    });

    await t.test('provider eagerly slices partial banks and isolates sibling residencies',
      async () => {
        const source = await WasmEngine.init(wasmPath);
        assert.ok(source);
        const forked = [];
        const originalFork = source.fork;
        source.fork = async function observedFork(options) {
          const engine = await originalFork.call(this, options);
          forked.push(engine);
          return engine;
        };
        const runtime = new Runtime();
        runtime._addProvider('wasm', new WasmBackendProvider(source));
        const results = [];
        let compiled;
        let first;
        let second;
        try {
          compiled = await runtime.compile(partiallyResidentBankSnapshot(), {
            backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
          });
          first = await compiled.createContext({ bankResidency: { experts: [1] } });
          second = await compiled.createContext({ bankResidency: { experts: [3] } });

          assert.equal(forked.length, 2);
          assert.notEqual(forked[0].mem, forked[1].mem,
            'sibling contexts must own distinct linear memories');
          for (const engine of forked) {
            assert.equal(engine.inspectArena().persistentWeightBytes, 2 * 4,
              'preload allocates one F32 [1, 2, 1] slice, not the full bank');
            assert.equal(engine.tensorMaximumBytes.get('experts'), 2 * 4,
              'a resident weight uses its exact sliced size as the context maximum');
          }
          assert.deepEqual(
            Array.from(forked[0].graph.getTensor('experts').buffer), [2, 2],
          );
          assert.deepEqual(
            Array.from(forked[1].graph.getTensor('experts').buffer), [4, 4],
          );

          const left = await first.execute(bankRouteInputs(1));
          const right = await second.execute(bankRouteInputs(3));
          results.push(left, right);
          assert.deepEqual(await left.output('y').read(), Float32Array.of(2));
          assert.deepEqual(await right.output('y').read(), Float32Array.of(4));
          assert.equal(left.report.backendReport.specializationCacheHit, true,
            'the first request hits the resident variant compiled by createContext');
          assert.equal(right.report.backendReport.specializationCacheHit, true);

          await assert.rejects(second.execute(bankRouteInputs(1)), (error) => {
            const messages = [];
            let cause = error;
            while (cause) {
              messages.push(cause.message);
              cause = cause.cause;
            }
            assert.ok(
              messages.some((message) => /expert that is not resident/.test(message)),
              messages.join(' <- '),
            );
            return true;
          });
          assert.deepEqual(
            Array.from(forked[1].graph.getTensor('experts').buffer), [4, 4],
            'a failed cross-residency route cannot replace the sibling weight slice',
          );
        } finally {
          source.fork = originalFork;
          await first?.close();
          await second?.close();
          await compiled?.close();
          await runtime.close();
          await Promise.all(results.map((result) => result.close()));
        }
      });

    await t.test('provider disposes a fork exactly once when eager preload compilation fails',
      async () => {
        const source = await WasmEngine.init(wasmPath);
        assert.ok(source);
        const originalFork = source.fork;
        let forked = null;
        let disposeCalls = 0;
        source.fork = async function failingPreloadFork(options) {
          const engine = await originalFork.call(this, options);
          forked = engine;
          const originalDispose = engine.dispose;
          engine.compile = function injectedPreloadCompileFailure() {
            throw new Error('injected eager preload compile failure');
          };
          engine.dispose = function countedForkDispose() {
            disposeCalls++;
            return originalDispose.call(this);
          };
          return engine;
        };
        const runtime = new Runtime();
        runtime._addProvider('wasm', new WasmBackendProvider(source));
        let compiled;
        let returnedContext;
        try {
          compiled = await runtime.compile(staticIdentitySnapshot(), {
            backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
          });
          await assert.rejects(async () => {
            returnedContext = await compiled.createContext();
          }, /injected eager preload compile failure/);
          assert.equal(returnedContext, undefined,
            'a context cannot escape after its eager specialization fails');
          assert.ok(forked);
          assert.equal(disposeCalls, 1,
            'the provider owns and disposes the failed fork exactly once');
          assert.equal(forked._disposed, true);
        } finally {
          source.fork = originalFork;
          await returnedContext?.close();
          await compiled?.close();
          await runtime.close();
        }
        assert.equal(disposeCalls, 1,
          'compiled-model/runtime cleanup must not redispose the rejected fork');
      });

    await t.test('public provider binds small-large-small once and transfers exact outputs', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      const results = [];
      let compiled;
      let context;
      try {
        compiled = await runtime.compile(dynamicIdentitySnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        for (const [shape, values] of [
          [[1, 2], [1, 2]],
          [[4, 3], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]],
          [[1, 2], [21, 22]],
        ]) {
          results.push(await context.execute({
            x: { data: Float32Array.from(values), shape },
          }));
        }
        assert.deepEqual(await results[0].output('y').read(), Float32Array.of(1, 2));
        assert.equal(results[0].report.backendReport.activationStorage, 'liveness');
        assert.equal(results[0].report.backendReport.activationRequiredBytes, 8);
        assert.equal(results[0].report.backendReport.activationReuseBytes, 8);
        assert.equal(results[0].report.backendReport.activationArenaCount, 1);
        assert.equal(results[0].report.backendReport.activationCapacityBytes, 8,
          'first shape starts at its active bytes instead of the declared maximum');
        assert.equal(results[0].report.backendReport.activationGrowCount, 1,
          'context preload starts at the smaller legal minimum before the first active shape');
        assert.equal(results[0].report.backendReport.specializationCacheHit, false,
          'a non-minimum first request does not alias the exact minimum preload signature');
        assert.equal(results[0].report.backendReport.specializationCacheEntries, 2,
          'minimum preload and the non-minimum first request remain separate variants');
        assert.deepEqual(await results[1].output('y').read(),
          Float32Array.of(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14));
        assert.equal(results[1].report.backendReport.specializationCacheHit, false);
        assert.equal(results[1].report.backendReport.specializationCacheEntries, 3);
        assert.equal(results[1].report.backendReport.activationCapacityBytes, 48);
        assert.equal(results[1].report.backendReport.activationCapacityHighWaterBytes, 48);
        assert.equal(results[1].report.backendReport.activationGrowCount, 2);
        assert.deepEqual(await results[2].output('y').read(), Float32Array.of(21, 22));
        assert.equal(results[2].report.backendReport.specializationCacheHit, true);
        assert.equal(results[2].report.backendReport.specializationCacheEntries, 3,
          'the preload minimum remains alongside both requested signatures');
        assert.equal(results[2].report.backendReport.invariantSchedulePrepareCount, 1);
        assert.equal(results[2].report.backendReport.invariantF32PackCount, 0);
        assert.equal(results[2].report.backendReport.logicalActivationBytes, 16);
        assert.equal(results[2].report.backendReport.activationCapacityBytes, 48);
        assert.equal(results[2].report.backendReport.activationGrowCount, 2);
        await context.close();
        context = null;
        assert.deepEqual(await results[0].output('y').read(), Float32Array.of(1, 2),
          'result storage must outlive arena reuse and context closure');
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('real WASM GELU tanh matches the portable numerical contract', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      const results = [];
      let compiled;
      let context;
      try {
        compiled = await runtime.compile(dynamicDecodeGeluSnapshot('tanh'), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        const inputs = {
          x: {
            data: Float32Array.of(-3, -2, -1, -0.5, 0, 0.5, 1, 3),
            shape: [1, 2, 4],
          },
        };
        const result = await context.execute(inputs);
        results.push(result);
        assertClose(
          await result.output('y').read(),
          referenceDecodeGelu(inputs, 'tanh'),
          'WASM GELU tanh',
        );
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('public WASM decode preserves row state across token steps and resets on prompt growth',
      async () => {
        const source = await WasmEngine.init(wasmPath);
        assert.ok(source);
        const runtime = new Runtime();
        runtime._addProvider('wasm', new WasmBackendProvider(source));
        const results = [];
        let compiled;
        let context;
        try {
          compiled = await runtime.compile(dynamicDecodeGeluSnapshot(), {
            backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
          });
          context = await compiled.createContext({
            decode: {
              changedInputs: ['x'],
              rowMode: 'required',
              requireIncremental: true,
            },
          });

          const seedInputs = decodeGeluInputs(4, 1);
          const seed = await context.decode.seed(seedInputs);
          results.push(seed);
          assertClose(await seed.output('y').read(), referenceDecodeGelu(seedInputs),
            'WASM decode seed');
          assert.equal(seed.report.decodeState.mode, 'incremental-seed');
          assert.equal(seed.report.backendReport.activationStorage, 'persistent');
          assert.equal(seed.report.backendReport.activationReuseBytes, 0);
          assert.equal(seed.report.backendReport.activationRequiredBytes,
            seed.report.backendReport.logicalActivationBytes);
          assert.equal(seed.report.decodeState.activeSequenceLength, 1);
          assert.equal(seed.report.decodeState.kvCapacity, 4);
          assert.equal(seed.report.decodeState.kvCapacityClass, 4);
          const generation = seed.report.decodeState.cacheGeneration;
          const semanticSignature = seed.report.decodeState.semanticSeedSignature;
          const rebindCount = seed.report.backendReport.specializationRebindCount;
          const growCount = seed.report.backendReport.activationGrowCount;

          const stepInputs = replaceDecodeGeluRow(seedInputs, 1, 30);
          const step = await context.decode.step(stepInputs, { position: 1 });
          results.push(step);
          assertClose(await step.output('y').read(), referenceDecodeGelu(stepInputs),
            'WASM decode row step');
          assert.equal(step.report.decodeState.mode, 'incremental-row');
          assert.equal(step.report.decodeState.activeSequenceLength, 2);
          assert.equal(step.report.decodeState.cacheGeneration, generation);
          assert.equal(step.report.decodeState.semanticSeedSignature, semanticSignature);
          assert.equal(step.report.backendReport.specializationCacheHit, true);
          assert.equal(step.report.backendReport.specializationRebindCount, rebindCount);
          assert.equal(step.report.backendReport.activationGrowCount, growCount,
            'WASM token steps must not grow or rebind the arena');

          const grownInputs = decodeGeluInputs(6, 50);
          const grown = await context.decode.seed(grownInputs);
          results.push(grown);
          assertClose(await grown.output('y').read(), referenceDecodeGelu(grownInputs),
            'WASM decode grown prompt seed');
          assert.equal((await grown.output('y').read()).length, 24);
          assert.equal(grown.report.decodeState.kvCapacity, 6);
          assert.equal(grown.report.decodeState.kvCapacityClass, 8);
          assert.equal(grown.report.decodeState.automaticReset, true);
          assert.equal(grown.report.decodeState.automaticResetReason,
            'kv-capacity-class-changed');
          assert.notEqual(grown.report.decodeState.semanticSeedSignature, semanticSignature);
          assert.ok(grown.report.backendReport.specializationRebindCount > rebindCount);

          await context.decode.reset();
          await assert.rejects(context.decode.step(grownInputs, { position: 1 }),
            (error) => error?.code === 'INVALID_ARGUMENT' && /successful seed/.test(error.message));
        } finally {
          await context?.close();
          await compiled?.close();
          await runtime.close();
          await Promise.all(results.map((result) => result.close()));
        }
      });

    await t.test('public provider grows active Linear extents without repacking weights', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      const results = [];
      let compiled;
      let context;
      try {
        compiled = await runtime.compile(dynamicLogicalLinearSnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        for (const values of [
          Float32Array.of(1, 2, 3),
          Float32Array.from({ length: 12 }, (_, index) => Math.fround((index - 3) / 5)),
          Float32Array.of(-2, 4, 0.5),
        ]) {
          const rows = values.length / 3;
          results.push(await context.execute({ input: { data: values, shape: [rows, 3] } }));
          assertClose(await results.at(-1).output('out').read(), referenceLinear(values),
            `public Linear rows=${rows}`);
        }
        for (const result of results) {
          assert.equal(result.report.backendReport.invariantF32PackCount, 1);
          assert.equal(result.report.backendReport.invariantSchedulePrepareCount, 1);
        }
        assert.equal(results[0].report.backendReport.activationGrowCount, 0);
        assert.equal(results[1].report.backendReport.activationGrowCount, 1);
        assert.equal(results[2].report.backendReport.activationGrowCount, 1);
        assert.equal(results[2].report.backendReport.specializationCacheHit, true);
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('compiled model creates isolated arenas for concurrent shape contexts', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      let compiled;
      let smallContext;
      let largeContext;
      const results = [];
      try {
        compiled = await runtime.compile(dynamicLogicalLinearSnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        [smallContext, largeContext] = await Promise.all([
          compiled.createContext(), compiled.createContext(),
        ]);
        const smallValues = Float32Array.of(1, 2, 3);
        const largeValues = Float32Array.from(
          { length: 12 }, (_, index) => Math.fround((index - 5) / 7),
        );
        results.push(...await Promise.all([
          smallContext.execute({ input: { data: smallValues, shape: [1, 3] } }),
          largeContext.execute({ input: { data: largeValues, shape: [4, 3] } }),
        ]));
        assertClose(await results[0].output('out').read(), referenceLinear(smallValues),
          'concurrent small context');
        assertClose(await results[1].output('out').read(), referenceLinear(largeValues),
          'concurrent large context');
        for (const [index, result] of results.entries()) {
          assert.equal(result.report.backendReport.invariantSchedulePrepareCount, 1);
          assert.equal(result.report.backendReport.invariantF32PackCount, 1);
          assert.equal(result.report.backendReport.specializationCacheEntries, index + 1,
            'only the large context adds a signature beyond its preloaded minimum');
        }
        await smallContext.close();
        smallContext = null;
        const survivor = await largeContext.execute({
          input: { data: Float32Array.of(2, 1, 0), shape: [1, 3] },
        });
        results.push(survivor);
        assertClose(await survivor.output('out').read(),
          referenceLinear(Float32Array.of(2, 1, 0)), 'surviving context');
      } finally {
        await smallContext?.close();
        await largeContext?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('public dynamic operator waves retain CPU parity across rebinds', async (t) => {
      for (const [label, snapshot, requests, outputNames] of [
        [
          'Wave-B broadcast/Where/reduction/transpose',
          dynamicWaveBSnapshot(),
          [waveBInputs(1, 2, 3), waveBInputs(4, 5, 7), waveBInputs(1, 2, 11)],
          ['mean', 'transposed'],
        ],
        [
          'spatial Conv2D',
          dynamicConvSnapshot(),
          [convInputs(1, 3, 4, 2), convInputs(2, 8, 9, 5), convInputs(1, 3, 4, 9)],
          ['out'],
        ],
        [
          'SDPA/CrossSDPA/RoPE attention family',
          dynamicAttentionSnapshot(),
          [
            attentionInputs(1, 2, 2, 3, 3),
            attentionInputs(2, 4, 3, 5, 7),
            attentionInputs(1, 2, 2, 3, 11),
          ],
          ['rotated', 'cross'],
        ],
      ]) {
        await t.test(label, async () => {
          const source = await WasmEngine.init(wasmPath);
          assert.ok(source);
          const wasmRuntime = new Runtime();
          const cpuRuntime = new Runtime();
          wasmRuntime._addProvider('wasm', new WasmBackendProvider(source));
          cpuRuntime._addProvider('cpu', new CPUBackendProvider(new CPUEngine()));
          let wasmCompiled;
          let cpuCompiled;
          let wasmContext;
          let cpuContext;
          const results = [];
          try {
            [wasmCompiled, cpuCompiled] = await Promise.all([
              wasmRuntime.compile(snapshot, {
                backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
              }),
              cpuRuntime.compile(snapshot, {
                backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
              }),
            ]);
            [wasmContext, cpuContext] = await Promise.all([
              wasmCompiled.createContext(), cpuCompiled.createContext(),
            ]);
            for (const request of requests) {
              const [wasmResult, cpuResult] = await Promise.all([
                wasmContext.execute(request), cpuContext.execute(request),
              ]);
              results.push(wasmResult, cpuResult);
              for (const name of outputNames) {
                assertClose(
                  await wasmResult.output(name).read(),
                  await cpuResult.output(name).read(),
                  `${label} ${name}`,
                );
              }
            }
            const finalReport = results.at(-2).report.backendReport;
            assert.equal(finalReport.specializationCacheHit, true);
            assert.equal(finalReport.specializationCacheEntries, 3,
              'the canonical preload minimum is retained with both request shapes');
            assert.equal(finalReport.invariantSchedulePrepareCount, 1);
            assert.equal(finalReport.specializationRebindCount, 3);
            assert.equal(finalReport.activationGrowCount, 2,
              'minimum preload and the larger request each grow the active arena once');
          } finally {
            await wasmContext?.close();
            await cpuContext?.close();
            await wasmCompiled?.close();
            await cpuCompiled?.close();
            await wasmRuntime.close();
            await cpuRuntime.close();
            await Promise.all(results.map((result) => result.close()));
          }
        });
      }
    });

    await t.test('static provider prebinding avoids shape specialization after first execution', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      let compiled;
      let context;
      const results = [];
      try {
        compiled = await runtime.compile(staticIdentitySnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        results.push(await context.execute({
          x: { data: Float32Array.of(1, 2, 3, 4), shape: [2, 2] },
        }));
        results.push(await context.execute({
          x: { data: Float32Array.of(5, 6, 7, 8), shape: [2, 2] },
        }));
        assert.deepEqual(await results[1].output('y').read(), Float32Array.of(5, 6, 7, 8));
        assert.equal(results[0].report.backendReport.specializationCacheHit, true,
          'a static request hits the variant prepared during context creation');
        assert.equal(results[1].report.backendReport.specializationCacheHit, true);
        assert.equal(results[1].report.backendReport.specializationCacheEntries, 1);
        assert.equal(results[1].report.backendReport.invariantSchedulePrepareCount, 1);
        assert.equal(results[1].report.backendReport.specializationRebindCount, 0);
        assert.equal(results[1].report.backendReport.activationGrowCount, 0);
        await assert.rejects(
          context.decode.seed({
            x: { data: Float32Array.of(9, 10, 11, 12), shape: [2, 2] },
          }),
          (error) => error?.code === 'BACKEND_UNSUPPORTED' &&
            /retained activation storage is explicit/.test(error.message),
        );
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('provider plan LRU evicts deterministically under adversarial signatures', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      let compiled;
      let context;
      const results = [];
      try {
        compiled = await runtime.compile(adversarialIdentitySnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        for (let rows = 1; rows <= 9; rows++) {
          results.push(await context.execute({
            x: {
              data: Float32Array.from({ length: rows }, (_, index) => rows * 10 + index),
              shape: [rows, 1],
            },
          }));
        }
        assert.equal(results[0].report.backendReport.specializationCacheHit, true,
          'the exact dynamic minimum request hits the variant preloaded at context creation');
        const highWaterHeap = results.at(-1).report.backendReport.wasmHeapBytes;
        results.push(await context.execute({
          x: { data: Float32Array.of(101), shape: [1, 1] },
        }));
        const revisitedOne = results.at(-1);
        assert.equal(revisitedOne.report.backendReport.specializationCacheHit, false,
          'the oldest of nine signatures must be evicted from an eight-entry LRU');
        assert.equal(revisitedOne.report.backendReport.specializationCacheEntries, 8);
        assert.equal(revisitedOne.report.backendReport.specializationCacheMisses, 9,
          'the preloaded minimum is a hit before eight new signatures and one revisit miss');
        assert.equal(revisitedOne.report.backendReport.specializationCacheEvictions, 2);
        assert.equal(revisitedOne.report.backendReport.activationCapacityHighWaterBytes, 36);
        assert.equal(revisitedOne.report.backendReport.wasmHeapBytes, highWaterHeap);
        assert.deepEqual(await revisitedOne.output('y').read(), Float32Array.of(101));

        results.push(await context.execute({
          x: { data: Float32Array.of(30, 31, 32), shape: [3, 1] },
        }));
        assert.equal(results.at(-1).report.backendReport.specializationCacheHit, true,
          'a non-evicted signature must remain a cache hit after the LRU rollover');
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('provider refuses to retain a plan larger than its metadata budget', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const runtime = new Runtime();
      runtime._addProvider('wasm', new WasmBackendProvider(source));
      let compiled;
      let context;
      const results = [];
      try {
        compiled = await runtime.compile(oversizedPlanMetadataSnapshot(), {
          backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
        });
        context = await compiled.createContext();
        for (const value of [3, 7]) {
          results.push(await context.execute({
            x: { data: Float32Array.of(value), shape: [1] },
          }));
        }
        assert.deepEqual(await results[1].output('y').read(), Float32Array.of(7));
        assert.equal(results[0].report.backendReport.specializationCacheEntries, 0);
        assert.equal(results[1].report.backendReport.specializationCacheEntries, 0);
        assert.equal(results[1].report.backendReport.specializationCacheMetadataBytes, 0);
        assert.equal(results[1].report.backendReport.specializationCacheMisses, 2);
        assert.equal(results[1].report.backendReport.invariantSchedulePrepareCount, 1);
        assert.equal(results[1].report.backendReport.specializationRebindCount, 0);
      } finally {
        await context?.close();
        await compiled?.close();
        await runtime.close();
        await Promise.all(results.map((result) => result.close()));
      }
    });

    await t.test('provider rejects unrepresentable bounded domains before context creation', async () => {
      const source = await WasmEngine.init(wasmPath);
      assert.ok(source);
      const provider = new WasmBackendProvider(source);
      try {
        const rankNine = boundedIdentitySnapshot({}, Array(9).fill(1));
        await assert.rejects(
          provider.compile(createBackendCompileInput(rankNine), {
            operatorFallback: 'forbid',
          }),
          /rank 9 exceeds 8/,
        );

        const signedExtentOverflow = boundedIdentitySnapshot({
          B: { min: 1, max: 0x80000000 },
        }, ['B']);
        await assert.rejects(
          provider.compile(createBackendCompileInput(signedExtentOverflow), {
            operatorFallback: 'forbid',
          }),
          /cannot represent the bounded extent/,
        );

        const residentOverflow = boundedIdentitySnapshot({
          B: { min: 1, max: 400_000_000 },
        }, ['B']);
        await assert.rejects(
          provider.compile(createBackendCompileInput(residentOverflow), {
            operatorFallback: 'forbid',
          }),
          /bounded-domain resources require.*exceeding/,
        );
      } finally {
        await provider.close();
      }
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
