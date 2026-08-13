import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader, ReadOnlySafetensorsCache } from '../ts/core/RuntimeGraphLoader.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

function fixture() {
  const weights = SafetensorsFile.empty({ metadata: { fixture: 'cache' } });
  weights.addTensor('weight', 'F32', [1], Float32Array.of(3));
  const buffer = weights.toArrayBuffer();
  const graphText = JSON.stringify({
    format: 'volvox-graph/v1',
    inputs: { input: { shape: [1], dtype: 'float32' } },
    nodes: [{
      opType: 'Identity',
      inputs: { input: 'input' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1] },
      outputs_dtype: { out: 'float32' },
    }],
    outputs: ['output'],
  });
  return { buffer, graphText };
}

test('graph loader rejects every non-canonical graph format before loading weights', async (t) => {
  const cases = [
    ['missing', undefined],
    ['wrong version', 'volvox-graph/v2'],
    ['wrong case', 'Volvox-Graph/v1'],
    ['wrong type', 1],
  ];
  for (const [name, format] of cases) {
    await t.test(name, async () => {
      const graph = new RuntimeGraph();
      let weightFetches = 0;
      const document = {
        inputs: { input: { shape: [1], dtype: 'float32' } },
        nodes: [],
      };
      if (format !== undefined) document.format = format;
      const fetch = async (url) => {
        if (url === 'graph.json') {
          return { ok: true, text: async () => JSON.stringify(document) };
        }
        weightFetches++;
        throw new Error(`unexpected weight fetch: ${url}`);
      };

      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        /graph\.json format must be 'volvox-graph\/v1'/,
      );
      assert.equal(weightFetches, 0);
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('graph loader rejects object-form graph outputs before loading weights', async () => {
  const graph = new RuntimeGraph();
  let weightFetches = 0;
  const document = {
    format: 'volvox-graph/v1',
    inputs: { input: { shape: [1], dtype: 'float32' } },
    nodes: [],
    outputs: { output: 'output' },
  };
  const fetch = async (url) => {
    if (url === 'graph.json') {
      return { ok: true, text: async () => JSON.stringify(document) };
    }
    weightFetches++;
    throw new Error(`unexpected weight fetch: ${url}`);
  };

  await assert.rejects(
    RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
    /graph outputs must be a non-empty array/,
  );
  assert.equal(weightFetches, 0);
  assert.equal(graph.nodes.length, 0);
  assert.equal(graph.tensors.size, 0);
});

test('graph loader requires declared non-empty graph outputs before loading weights', async (t) => {
  for (const [name, outputs] of [['missing', undefined], ['empty', []]]) {
    await t.test(name, async () => {
      const graph = new RuntimeGraph();
      let weightFetches = 0;
      const document = {
        format: 'volvox-graph/v1',
        inputs: { input: { shape: [1], dtype: 'float32' } },
        nodes: [],
      };
      if (outputs !== undefined) document.outputs = outputs;
      const fetch = async (url) => {
        if (url === 'graph.json') {
          return { ok: true, text: async () => JSON.stringify(document) };
        }
        weightFetches++;
        throw new Error(`unexpected weight fetch: ${url}`);
      };

      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        /graph outputs must be a non-empty array/,
      );
      assert.equal(weightFetches, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('graph loader resolves only the canonical graph.json package entry', async () => {
  const { buffer, graphText } = fixture();
  const requested = [];
  const fetch = async (url) => {
    requested.push(url);
    if (url === 'models/graph.json') return { ok: true, text: async () => graphText };
    if (url === 'models/demo_weights.safetensors?revision=7') {
      return { ok: true, arrayBuffer: async () => buffer };
    }
    return { ok: false, statusText: 'not found' };
  };

  await RuntimeGraphLoader.load(new RuntimeGraph(), 'models/demo_weights.safetensors?revision=7', { fetch });

  assert.deepEqual(requested, [
    'models/graph.json',
    'models/demo_weights.safetensors?revision=7',
  ]);
});

test('graph loader rejects non-graph package basenames before fetching', async () => {
  let fetches = 0;
  const fetch = async () => {
    fetches++;
    throw new Error('fetch must not run');
  };
  await assert.rejects(
    RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', {
      graphUrl: 'models/config.json',
      fetch,
    }),
    /must name graph\.json or a named \*\.graph\.json document/,
  );
  assert.equal(fetches, 0);
});

test('graph loader requires an explicit canonical runtime dtype for every graph input', async (t) => {
  const { buffer } = fixture();
  for (const [name, dtype] of [['missing', undefined], ['uppercase', 'F32'], ['storage-only', 'float16']]) {
    await t.test(name, async () => {
      const input = { shape: [1] };
      if (dtype !== undefined) input.dtype = dtype;
      const graphText = JSON.stringify({
        format: 'volvox-graph/v1',
        inputs: { input },
        nodes: [],
        outputs: ['input'],
      });
      const fetch = async (url) => url === 'graph.json'
        ? { ok: true, text: async () => graphText }
        : { ok: true, arrayBuffer: async () => buffer };
      const graph = new RuntimeGraph();
      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        /requires an explicit canonical dtype/,
      );
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('graph loader rejects implicit output dtypes and the retired op alias before loading weights', async (t) => {
  const { buffer } = fixture();
  const base = {
    format: 'volvox-graph/v1',
    inputs: { input: { shape: [1], dtype: 'float32' } },
    nodes: [{
      opType: 'Identity',
      inputs: { input: 'input' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1] },
      outputs_dtype: { out: 'float32' },
    }],
    outputs: ['output'],
  };
  const cases = [
    ['missing output dtype', (document) => { delete document.nodes[0].outputs_dtype; }, /outputs_dtype must exactly describe/],
    ['incomplete output dtype', (document) => { document.nodes[0].outputs_dtype = {}; }, /outputs_dtype must exactly describe/],
    ['retired op alias', (document) => { document.nodes[0].op = 'Identity'; }, /unsupported field 'op'/],
  ];
  for (const [name, mutate, expected] of cases) {
    await t.test(name, async () => {
      const document = structuredClone(base);
      mutate(document);
      let weightFetches = 0;
      const fetch = async (url) => {
        if (url === 'graph.json') {
          return { ok: true, text: async () => JSON.stringify(document) };
        }
        weightFetches++;
        return { ok: true, arrayBuffer: async () => buffer };
      };
      await assert.rejects(
        RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        expected,
      );
      assert.equal(weightFetches, 0);
    });
  }
});

test('graph loader rejects unknown operators before loading weights', async (t) => {
  const { buffer } = fixture();
  for (const opType of ['DefinitelyNotAnOperator']) {
    await t.test(opType, async () => {
      const document = {
        format: 'volvox-graph/v1',
        inputs: { input: { shape: [1], dtype: 'float32' } },
        nodes: [{
          opType,
          inputs: { input: 'input' },
          outputs: { out: 'output' },
          outputs_shape: { out: [1] },
          outputs_dtype: { out: 'float32' },
        }],
        outputs: ['output'],
      };
      const graph = new RuntimeGraph();
      let weightFetches = 0;
      const fetch = async (url) => {
        if (url === 'graph.json') {
          return { ok: true, text: async () => JSON.stringify(document) };
        }
        weightFetches++;
        return { ok: true, arrayBuffer: async () => buffer };
      };
      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        /unknown or offline-only current-v1 opType/,
      );
      assert.equal(weightFetches, 0);
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('graph loader validates json-only response numbers losslessly before loading weights', async (t) => {
  for (const [label, value, expected] of [
    ['non-finite', Number.POSITIVE_INFINITY, /non-finite JSON number/],
    ['unsafe integer', Number.MAX_SAFE_INTEGER + 1, /outside JSON's safe range/],
  ]) {
    await t.test(label, async () => {
      const document = {
        format: 'volvox-graph/v1',
        inputs: { input: { shape: [1], dtype: 'float32' } },
        nodes: [],
        outputs: ['input'],
        source: { invalid_numeric_evidence: value },
      };
      const graph = new RuntimeGraph();
      let weightFetches = 0;
      const fetch = async (url) => {
        if (url === 'graph.json') return { ok: true, json: async () => document };
        weightFetches++;
        throw new Error('weights must not be fetched');
      };
      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        expected,
      );
      assert.equal(weightFetches, 0);
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('graph loader rejects unconstructable current-v1 tensor names and sizes before weights', async (t) => {
  const cases = [
    ['constructor name', () => ({
      inputs: Object.fromEntries([['constructor', { shape: [1], dtype: 'float32' }]]),
      outputs: ['constructor'],
    }), /not a valid current-v1 name/],
    ['__proto__ name', () => ({
      inputs: Object.fromEntries([['__proto__', { shape: [1], dtype: 'float32' }]]),
      outputs: ['__proto__'],
    }), /not a valid current-v1 name/],
    ['unsafe element count', () => ({
      inputs: { input: { shape: [1 << 27, 1 << 27], dtype: 'float32' } },
      outputs: ['input'],
    }), /element count exceeds JSON's safe integer range/],
    ['unsafe byte count', () => ({
      inputs: { input: { shape: [Number.MAX_SAFE_INTEGER], dtype: 'float32' } },
      outputs: ['input'],
    }), /byte size exceeds JSON's safe integer range/],
  ];
  for (const [label, fields, expected] of cases) {
    await t.test(label, async () => {
      const graph = new RuntimeGraph();
      const document = {
        format: 'volvox-graph/v1',
        nodes: [],
        ...fields(),
      };
      let weightFetches = 0;
      const fetch = async (url) => {
        if (url === 'graph.json') {
          return { ok: true, text: async () => JSON.stringify(document) };
        }
        weightFetches++;
        throw new Error('weights must not be fetched');
      };
      await assert.rejects(
        RuntimeGraphLoader.load(graph, 'model.safetensors', { graphUrl: 'graph.json', fetch }),
        expected,
      );
      assert.equal(weightFetches, 0);
      assert.equal(graph.nodes.length, 0);
      assert.equal(graph.tensors.size, 0);
    });
  }
});

test('read-only SafeTensors cache deduplicates concurrent fetch/parse while isolating graph wrappers', async () => {
  const { buffer, graphText } = fixture();
  const counts = { graph: 0, weights: 0 };
  const fetch = async (url) => {
    if (url === 'graph.json') {
      counts.graph++;
      return { ok: true, text: async () => graphText };
    }
    if (url === 'model.safetensors') {
      counts.weights++;
      await Promise.resolve();
      return { ok: true, arrayBuffer: async () => buffer };
    }
    return { ok: false, statusText: 'not found' };
  };
  const cache = new ReadOnlySafetensorsCache();
  const load = () => RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', {
    graphUrl: 'graph.json', fetch, safetensorsCache: cache,
  });
  const [first, second] = await Promise.all([load(), load()]);

  assert.equal(counts.graph, 2, 'each graph still owns and validates its document');
  assert.equal(counts.weights, 1, 'model bytes are fetched and parsed once');
  assert.equal(cache.size, 1);
  assert.notEqual(first, second);
  assert.equal(first.topologyRevision, 1, 'weights, inputs, and nodes commit atomically once');
  assert.equal(second.topologyRevision, 1);
  assert.notEqual(first.getTensor('weight'), second.getTensor('weight'));
  assert.notEqual(first.weightFiles[0], second.weightFiles[0]);
  assert.notEqual(first.weightFiles[0].getTensor('weight'), second.weightFiles[0].getTensor('weight'));
  first.weightFiles[0].metadata.fixture = 'first-only';
  first.weightFiles[0].getTensor('weight').shape[0] = 99;
  assert.equal(second.weightFiles[0].metadata.fixture, 'cache');
  assert.deepEqual(second.weightFiles[0].getTensor('weight').shape, [1]);
  assert.throws(
    () => second.weightFiles[0].getTensor('weight').getMutableBytes(),
    /not opened writable/,
  );

  cache.clear();
  await load();
  assert.equal(counts.weights, 2);
});

test('read-only SafeTensors cache evicts failed loads and rejects writable graph loading', async () => {
  const { buffer, graphText } = fixture();
  let weightAttempts = 0;
  const fetch = async (url) => {
    if (url === 'graph.json') return { ok: true, text: async () => graphText };
    weightAttempts++;
    if (weightAttempts === 1) return { ok: false, statusText: 'temporary failure' };
    return { ok: true, arrayBuffer: async () => buffer };
  };
  const cache = new ReadOnlySafetensorsCache();
  const options = { graphUrl: 'graph.json', fetch, safetensorsCache: cache };
  await assert.rejects(
    RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', options),
    /temporary failure/,
  );
  assert.equal(cache.size, 0);
  await RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', options);
  assert.equal(weightAttempts, 2);
  assert.equal(cache.size, 1);
  await assert.rejects(
    RuntimeGraphLoader.load(new RuntimeGraph(), 'model.safetensors', {
      ...options, safetensors: { writable: true },
    }),
    /cannot be used with writable safetensors options/,
  );
});
