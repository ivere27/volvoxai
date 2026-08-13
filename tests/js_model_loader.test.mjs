import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { ReadOnlySafetensorsCache } from '../ts/core/RuntimeGraphLoader.js';
import {
  ModelLoader,
} from '../ts/core/ModelLoader.js';
import { GraphError } from '../ts/core/Graph.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';
import { Tensor } from '../ts/core/Tensor.js';

function constantDocument(overrides = {}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1, 4] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: [1, 4] },
      },
      params: {},
    }],
    outputs: ['y'],
    ...overrides,
  };
}

function symbolicDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 8 },
      S: { min: 2, max: 32, multiple_of: 2 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: ['B', 'S'] },
      },
      params: {},
    }],
    outputs: ['y'],
  };
}

function quantizedDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 8 } },
    inputs: { x: { dtype: 'int8', shape: ['B', 4] } },
    nodes: [{
      id: 'linear',
      opType: 'QLinear',
      inputs: { input: 'x', weight: 'w' },
      outputs: {
        out: { tensor: 'y', dtype: 'uint8', shape: ['B', 3] },
      },
      params: {},
    }],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        x: {
          scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx',
        },
        w: {
          scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw',
        },
        y: {
          scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy',
        },
      },
    },
  };
}

function makeSafetensors(definitions, metadata = {}) {
  const file = SafetensorsFile.empty({ metadata });
  for (const [name, dtype, shape, values] of definitions) {
    file.addTensor(name, dtype, shape, values);
  }
  return file.toArrayBuffer();
}

function constantWeights(value = 3) {
  return makeSafetensors([
    ['weight', 'F32', [1], Float32Array.of(value)],
  ], { fixture: 'logical-loader' });
}

function quantizedWeights(overrides = {}) {
  const definitions = {
    w: ['I8', [3, 4], Int8Array.from({ length: 12 }, (_, index) => index - 6)],
    sx: ['F32', [1], Float32Array.of(0.125)],
    zx: ['I8', [1], Int8Array.of(-3)],
    sw: ['F32', [3], Float32Array.of(0.25, 1 / 3, 0.5)],
    zw: ['I8', [3], Int8Array.of(-1, 0, 1)],
    sy: ['F32', [1], Float32Array.of(0.5)],
    zy: ['U8', [1], Uint8Array.of(127)],
    ...overrides,
  };
  return makeSafetensors(
    Object.entries(definitions).map(([name, definition]) => [name, ...definition]),
  );
}

function packageFetch(graph, weightsBySource) {
  const graphText = typeof graph === 'string' ? graph : JSON.stringify(graph);
  return async (source) => {
    if (source.endsWith('graph.json') || /\.graph\.json(?:[?#]|$)/.test(source)) {
      return { ok: true, text: async () => graphText };
    }
    if (Object.hasOwn(weightsBySource, source)) {
      return { ok: true, arrayBuffer: async () => weightsBySource[source] };
    }
    return { ok: false, statusText: 'not found' };
  };
}

test('logical package loader loads constant and symbolic documents with canonical source resolution', async () => {
  const buffer = constantWeights();
  const requested = [];
  const baseFetch = packageFetch(constantDocument(), {
    'models/model.safetensors?revision=7': buffer,
  });
  const loaded = await ModelLoader.load(
    'models/model.safetensors?revision=7',
    { fetch: async (source) => {
      requested.push(source);
      return baseFetch(source);
    } },
  );

  assert.deepEqual(requested, [
    'models/graph.json',
    'models/model.safetensors?revision=7',
  ]);
  assert.deepEqual(loaded.graph.inputs.x.shape, [1, 4]);
  assert.deepEqual(loaded.graph.nodes[0].outputs.out.shape, [1, 4]);
  assert.deepEqual(loaded.source, {
    graphUrl: 'models/graph.json',
    weightSources: ['models/model.safetensors?revision=7'],
  });
  assert.deepEqual(loaded.weightFiles[0], {
    source: 'models/model.safetensors?revision=7',
    sourceIndex: 0,
    sizeBytes: buffer.byteLength,
    metadata: { fixture: 'logical-loader' },
    tensorNames: ['weight'],
  });
  assert.deepEqual([...loaded.weights.weight.data], [3]);

  const symbolic = await ModelLoader.load('weights.safetensors', {
    graphUrl: 'router.graph.json?revision=2',
    fetch: packageFetch(symbolicDocument(), {
      'weights.safetensors': constantWeights(4),
    }),
  });
  assert.deepEqual(symbolic.graph.dimensions.B, {
    name: 'B', min: 1, max: 8, multiple_of: 1,
  });
  assert.deepEqual(symbolic.graph.dimensions.S, {
    name: 'S', min: 2, max: 32, multiple_of: 2,
  });
  assert.deepEqual(symbolic.graph.inputs.x.shape, ['B', 'S']);
  assert.equal(symbolic.source.graphUrl, 'router.graph.json?revision=2');
});

test('explicit graphUrl supports weightless input-only packages without a dummy file', async () => {
  const requested = [];
  const loaded = await ModelLoader.load([], {
    graphUrl: 'graph.json',
    fetch: async (source) => {
      requested.push(source);
      return { ok: true, text: async () => JSON.stringify(constantDocument()) };
    },
  });

  assert.deepEqual(requested, ['graph.json']);
  assert.deepEqual(loaded.weights, {});
  assert.deepEqual(loaded.weightFiles, []);
  assert.deepEqual(loaded.source, { graphUrl: 'graph.json', weightSources: [] });
  assert.equal(Object.isFrozen(loaded.weights), true);
  assert.equal(Object.isFrozen(loaded.weightFiles), true);
  assert.equal(Object.isFrozen(loaded.source.weightSources), true);

  await assert.rejects(
    ModelLoader.load([], {
      fetch: async () => {
        throw new Error('must not fetch');
      },
    }),
    /graphUrl is required when no safetensors sources are supplied/,
  );
});

test('logical weights own copied payloads and expose only defensive data copies', async () => {
  const sourceBuffer = constantWeights(7);
  const loaded = await ModelLoader.load('model.safetensors', {
    fetch: packageFetch(constantDocument(), { 'model.safetensors': sourceBuffer }),
  });
  const weight = loaded.weights.weight;

  for (const value of [
    loaded,
    loaded.weights,
    weight,
    weight.shape,
    loaded.weightFiles,
    loaded.weightFiles[0],
    loaded.weightFiles[0].metadata,
    loaded.weightFiles[0].tensorNames,
    loaded.source,
    loaded.source.weightSources,
  ]) assert.equal(Object.isFrozen(value), true);

  assert.equal(weight instanceof Tensor, false);
  assert.equal(weight.dtype, 'float32');
  assert.equal(weight.safetensorsDtype, 'F32');
  assert.equal(weight.sizeBytes, 4);
  assert.equal(weight.safetensorsSizeBytes, 4);
  assert.equal(weight.role, 'model_weight');

  const firstRead = weight.data;
  const secondRead = weight.copyData();
  assert.notStrictEqual(firstRead.buffer, secondRead.buffer);
  firstRead[0] = 91;
  secondRead[0] = 92;
  weight.copyBytes().fill(0);
  new Uint8Array(sourceBuffer).fill(0);
  assert.deepEqual([...weight.data], [7]);
  assert.deepEqual([...weight.copyData()], [7]);

  const source = await readFile(
    new URL('../ts/core/ModelLoader.ts', import.meta.url),
    'utf8',
  );
  assert.doesNotMatch(source, /from ['"]\.\/(?:RuntimeGraph|Tensor)\.js['"]/);
  assert.doesNotMatch(source, /new (?:RuntimeGraph|Tensor)\s*\(/);
});

test('logical package loader hydrates deeply immutable per-tensor and per-axis affine metadata', async () => {
  const loaded = await ModelLoader.load('model.safetensors', {
    fetch: packageFetch(quantizedDocument(), {
      'model.safetensors': quantizedWeights(),
    }),
  });

  assert.deepEqual(loaded.quantizationByTensor.x, {
    scheme: 'per_tensor', scale: 0.125, zero_point: -3,
  });
  assert.deepEqual(loaded.quantizationByTensor.w, {
    scheme: 'per_axis',
    axis: 0,
    scales: [0.25, Math.fround(1 / 3), 0.5],
    zero_points: [-1, 0, 1],
  });
  assert.deepEqual(loaded.quantizationByTensor.y, {
    scheme: 'per_tensor', scale: 0.5, zero_point: 127,
  });
  assert.deepEqual(loaded.quantizationParameterNames, [
    'sw', 'sx', 'sy', 'zw', 'zx', 'zy',
  ]);
  assert.equal(loaded.weights.w.role, 'model_weight');
  for (const name of loaded.quantizationParameterNames) {
    assert.equal(loaded.weights[name].role, 'quantization_parameter');
  }
  assert.deepEqual(loaded.graph.weights.w.quantization, {
    scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw',
  });

  for (const value of [
    loaded.quantizationByTensor,
    loaded.quantizationByTensor.x,
    loaded.quantizationByTensor.w,
    loaded.quantizationByTensor.w.scales,
    loaded.quantizationByTensor.w.zero_points,
    loaded.quantizationParameterNames,
  ]) assert.equal(Object.isFrozen(value), true);
});

test('logical package loader rejects duplicate graph keys and cross-file tensor names', async () => {
  const duplicateGraph = JSON.stringify(constantDocument()).replace(
    '{"format":"volvox-graph/v1",',
    '{"format":"volvox-graph/v1","format":"volvox-graph/v1",',
  );
  let weightFetches = 0;
  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: async (source) => {
        if (source === 'graph.json') {
          return { ok: true, text: async () => duplicateGraph };
        }
        weightFetches++;
        return { ok: true, arrayBuffer: async () => constantWeights() };
      },
    }),
    (error) => {
      assert.equal(error.code, 'ERR_STRICT_JSON_DUPLICATE_KEY');
      assert.match(error.message, /duplicate object key "format"/);
      return true;
    },
  );
  assert.equal(weightFetches, 0);

  const first = makeSafetensors([
    ['duplicate', 'F32', [1], Float32Array.of(1)],
  ]);
  const second = makeSafetensors([
    ['duplicate', 'F32', [1], Float32Array.of(2)],
  ]);
  await assert.rejects(
    ModelLoader.load(['first.safetensors', 'second.safetensors'], {
      fetch: packageFetch(constantDocument(), {
        'first.safetensors': first,
        'second.safetensors': second,
      }),
    }),
    /Tensor 'duplicate' occurs in more than one safetensors source/,
  );
});

test('logical package loader enforces canonical filenames and the closed graph root before weights', async (t) => {
  let fetches = 0;
  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      graphUrl: 'model.json',
      fetch: async () => {
        fetches++;
        throw new Error('must not fetch');
      },
    }),
    /graphUrl must name graph\.json or a named \*\.graph\.json/,
  );
  assert.equal(fetches, 0);

  const cases = [
    ['wrong format', { ...constantDocument(), format: 'volvox-graph/v2' }, 'INVALID_GRAPH'],
    ['retired shape system field', {
      ...constantDocument(), shape_system: 'volvox-bounded-shape/v1',
    }, 'INVALID_GRAPH'],
  ];
  for (const [name, document, diagnostic] of cases) {
    await t.test(name, async () => {
      let attemptedWeights = 0;
      await assert.rejects(
        ModelLoader.load('model.safetensors', {
          fetch: async (source) => {
            if (source === 'graph.json') {
              return { ok: true, text: async () => JSON.stringify(document) };
            }
            attemptedWeights++;
            return { ok: true, arrayBuffer: async () => constantWeights() };
          },
        }),
        (error) => {
          assert.ok(error instanceof GraphError);
          assert.equal(error.diagnostic, diagnostic);
          return true;
        },
      );
      assert.equal(attemptedWeights, 0);
    });
  }
});

test('logical package loader rejects lossy decoded JSON and malformed quantization payload values', async () => {
  const unsafe = constantDocument({
    dimensions: { B: { min: 1, max: Number.MAX_SAFE_INTEGER + 1 } },
  });
  let weightFetches = 0;
  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: async (source) => {
        if (source === 'graph.json') return { ok: true, json: async () => unsafe };
        weightFetches++;
        return { ok: true, arrayBuffer: async () => constantWeights() };
      },
    }),
    /integer outside JSON's safe range/,
  );
  assert.equal(weightFetches, 0);

  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: packageFetch(quantizedDocument(), {
        'model.safetensors': quantizedWeights({
          sx: ['F32', [1], Float32Array.of(Number.NaN)],
        }),
      }),
    }),
    /Quantization scale 0 for 'x'.*finite, positive.*F32/,
  );

  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: packageFetch(quantizedDocument(), {
        'model.safetensors': quantizedWeights({
          sx: ['F16', [1], Uint16Array.of(0x3800)],
        }),
      }),
    }),
    /Scale tensor 'sx' for 'x' must use canonical F32 storage/,
  );

  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: packageFetch(quantizedDocument(), {
        'model.safetensors': quantizedWeights({
          zy: ['I8', [1], Int8Array.of(127)],
        }),
      }),
    }),
    /'zy' dtype must match target dtype 'uint8'/,
  );
});

test('logical parser validates QuantizeLinear parameter inputs during package hydration', async () => {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input: { dtype: 'float32', shape: [2] } },
    nodes: [{
      id: 'quantize',
      opType: 'QuantizeLinear',
      inputs: { input: 'input', scale: 'wrong_scale', zero_point: 'zero' },
      outputs: {
        out: { tensor: 'output', dtype: 'uint8', shape: [2] },
      },
      params: {},
    }],
    outputs: ['output'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        output: {
          scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero',
        },
      },
    },
  };
  const weights = makeSafetensors([
    ['scale', 'F32', [1], Float32Array.of(0.25)],
    ['wrong_scale', 'F32', [1], Float32Array.of(0.5)],
    ['zero', 'U8', [1], Uint8Array.of(127)],
  ]);
  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      fetch: packageFetch(document, { 'model.safetensors': weights }),
    }),
    /QuantizeLinear parameter inputs must match/,
  );
});

test('read-only safetensors cache deduplicates payload loading while packages stay isolated', async () => {
  const buffer = constantWeights(11);
  const counts = { graph: 0, weights: 0 };
  const cache = new ReadOnlySafetensorsCache();
  const fetch = async (source) => {
    if (source === 'graph.json') {
      counts.graph++;
      return { ok: true, text: async () => JSON.stringify(constantDocument()) };
    }
    if (source === 'model.safetensors') {
      counts.weights++;
      await Promise.resolve();
      return { ok: true, arrayBuffer: async () => buffer };
    }
    return { ok: false, statusText: 'not found' };
  };
  const load = () => ModelLoader.load('model.safetensors', {
    graphUrl: 'graph.json', fetch, safetensorsCache: cache,
  });
  const [first, second] = await Promise.all([load(), load()]);

  assert.equal(counts.graph, 2);
  assert.equal(counts.weights, 1);
  assert.equal(cache.size, 1);
  assert.notStrictEqual(first, second);
  assert.notStrictEqual(first.weights.weight, second.weights.weight);
  const changed = first.weights.weight.data;
  changed[0] = 99;
  assert.deepEqual([...first.weights.weight.data], [11]);
  assert.deepEqual([...second.weights.weight.data], [11]);

  await assert.rejects(
    ModelLoader.load('model.safetensors', {
      graphUrl: 'graph.json', fetch, safetensorsCache: cache,
      safetensors: { writable: true },
    }),
    /cannot be used with writable safetensors options/,
  );
});
