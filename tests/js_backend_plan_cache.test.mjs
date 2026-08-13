import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BackendPlanCache,
  backendPlanCacheKey,
  trainingGraphTopologyIdentity,
} from '../ts/training/BackendPlanCache.js';

test('backend plan cache budgets retained key bytes and reports oversize skips', () => {
  const cache = new BackendPlanCache('wasm', {
    planCacheEntries: 2,
    planCacheMetadataBytes: 32,
  });
  const first = Object.freeze({ id: 'first' });
  const second = Object.freeze({ id: 'second' });

  cache.recordMiss();
  cache.recordBuild();
  assert.equal(cache.publish('1234567890', first, 20), true);
  assert.equal(cache.inspect().metadataBytes, 30);

  cache.recordMiss();
  cache.recordBuild();
  assert.equal(cache.publish('b', second, 20), true);
  let inspection = cache.inspect();
  assert.equal(inspection.entries, 1);
  assert.equal(inspection.evictions, 1);
  assert.equal(inspection.metadataBytes, 21);

  cache.recordMiss();
  cache.recordBuild();
  assert.equal(cache.publish('oversized-key-12345', first, 20), false);
  inspection = cache.inspect();
  assert.equal(inspection.entries, 1);
  assert.equal(inspection.oversizeSkips, 1);
  assert.ok(inspection.metadataBytes <= inspection.metadataLimitBytes);
});

test('backend plan cache hit publication is transactional and topology invalidation is bounded', () => {
  const cache = new BackendPlanCache('webgpu', {
    planCacheEntries: 2,
    planCacheMetadataBytes: 256,
  });
  const key = backendPlanCacheKey('B=4', 'linear-4');
  const recipe = Object.freeze({ id: 'linear-4' });
  cache.recordMiss();
  cache.recordBuild();
  assert.equal(cache.publish(key, recipe, 16), true);

  assert.equal(cache.peek(key), recipe);
  assert.equal(cache.inspect().hits, 0, 'peek must not mutate hit telemetry or LRU state');
  cache.recordHit(key);
  assert.equal(cache.inspect().hits, 1);

  cache.invalidateTopology();
  const inspection = cache.inspect();
  assert.equal(inspection.entries, 0);
  assert.equal(inspection.evictions, 1);
});

test('backend topology ownership changes on tensor kind, activation dtype, or rank', () => {
  const graph = (dtype, shape, inputKind = true) => ({
    tensors: new Map([
      ['x', { name: 'x', dtype, shape, ...(inputKind ? { isInput: true } : {}) }],
      ['weight', { name: 'weight', dtype: 'float32', shape: [2, 2], isWeight: true }],
      ['y', { name: 'y', dtype: 'float32', shape }],
    ]),
    nodes: [{
      id: 'linear',
      opType: 'Linear',
      inputs: { input: { name: 'x' }, weight: { name: 'weight' } },
      outputs: { out: { name: 'y' } },
      params: { weight_layout: 'din_dout' },
    }],
    outputNames: ['y'],
  });
  const base = trainingGraphTopologyIdentity(graph('float32', [1, 2]));
  assert.notEqual(base, trainingGraphTopologyIdentity(graph('int32', [1, 2])));
  assert.notEqual(base, trainingGraphTopologyIdentity(graph('float32', [1, 1, 2])));
  assert.notEqual(base, trainingGraphTopologyIdentity(graph('float32', [1, 2], false)));
});
