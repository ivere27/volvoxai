import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader, ReadOnlySafetensorsCache } from '../ts/core/GraphLoader.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

function fixture() {
  const weights = SafetensorsFile.empty({ metadata: { fixture: 'cache' } });
  weights.addTensor('weight', 'F32', [1], Float32Array.of(3));
  const buffer = weights.toArrayBuffer();
  const config = JSON.stringify({
    inputs: { input: { shape: [1], dtype: 'float32' } },
    nodes: [{
      op: 'Identity',
      inputs: { input: 'input' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1] },
    }],
    outputs: ['output'],
  });
  return { buffer, config };
}

test('read-only SafeTensors cache deduplicates concurrent fetch/parse while isolating graph wrappers', async () => {
  const { buffer, config } = fixture();
  const counts = { config: 0, weights: 0 };
  const fetch = async (url) => {
    if (url === 'config.json') {
      counts.config++;
      return { ok: true, text: async () => config };
    }
    if (url === 'model.safetensors') {
      counts.weights++;
      await Promise.resolve();
      return { ok: true, arrayBuffer: async () => buffer };
    }
    return { ok: false, statusText: 'not found' };
  };
  const cache = new ReadOnlySafetensorsCache();
  const load = () => GraphLoader.load(new Graph(), 'model.safetensors', {
    configUrl: 'config.json', fetch, safetensorsCache: cache,
  });
  const [first, second] = await Promise.all([load(), load()]);

  assert.equal(counts.config, 2, 'each graph still owns and validates its config');
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
  const { buffer, config } = fixture();
  let weightAttempts = 0;
  const fetch = async (url) => {
    if (url === 'config.json') return { ok: true, text: async () => config };
    weightAttempts++;
    if (weightAttempts === 1) return { ok: false, statusText: 'temporary failure' };
    return { ok: true, arrayBuffer: async () => buffer };
  };
  const cache = new ReadOnlySafetensorsCache();
  const options = { configUrl: 'config.json', fetch, safetensorsCache: cache };
  await assert.rejects(
    GraphLoader.load(new Graph(), 'model.safetensors', options),
    /temporary failure/,
  );
  assert.equal(cache.size, 0);
  await GraphLoader.load(new Graph(), 'model.safetensors', options);
  assert.equal(weightAttempts, 2);
  assert.equal(cache.size, 1);
  await assert.rejects(
    GraphLoader.load(new Graph(), 'model.safetensors', {
      ...options, safetensors: { writable: true },
    }),
    /cannot be used with writable safetensors options/,
  );
});
