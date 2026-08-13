import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CPUShapeExecutionContext,
  CPUShapeExecutionContextError,
  cpuActivationArenaMetadataBytes,
  cpuShapePlanMetadataBytes,
} from '../ts/core/CPUShapeExecutionContext.js';
import {
  concreteCPUActivationArenaTensors,
  maximumCPUActivationArenaTensors,
  planCPUActivationArena,
  planCPUActivationArenaWithinMaximum,
} from '../ts/core/CPUActivationArenaPlan.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';

function typedData(dtype, values) {
  if (dtype === 'float32') return Float32Array.from(values);
  if (dtype === 'int32') return Int32Array.from(values);
  if (dtype === 'int8') return Int8Array.from(values);
  return Uint8Array.from(values);
}

function makeSnapshot(document, definitions = []) {
  const graph = parseGraphDocument(
    document,
    definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
  );
  const weights = Object.fromEntries(definitions.map((definition) => [
    definition.name,
    {
      name: definition.name,
      dtype: definition.dtype,
      shape: [...definition.shape],
      data: typedData(definition.dtype, definition.values),
    },
  ]));
  return Model.capture({ graph, weights });
}

function dynamicIdentitySnapshot({ max = 8 } = {}) {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max },
      S: { min: 1, max },
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
  });
}

function staticIdentitySnapshot() {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2, 3] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2, 3] } },
      params: {},
    }],
    outputs: ['y'],
  });
}

function mixedDynamicFixedSnapshot() {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: {
      a: { dtype: 'float32', shape: ['B'] },
      z: { dtype: 'float32', shape: [4] },
    },
    nodes: [{
      id: 'dynamic', opType: 'Identity', inputs: { input: 'a' },
      outputs: { out: { tensor: 'b', dtype: 'float32', shape: ['B'] } }, params: {},
    }, {
      id: 'fixed', opType: 'Identity', inputs: { input: 'z' },
      outputs: { out: { tensor: 'zz', dtype: 'float32', shape: [4] } }, params: {},
    }],
    outputs: ['b', 'zz'],
  });
}

function mixedDTypeDynamicFixedSnapshot() {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: {
      a: { dtype: 'float32', shape: ['B'] },
      z: { dtype: 'int32', shape: [4] },
    },
    nodes: [{
      id: 'dynamic', opType: 'Identity', inputs: { input: 'a' },
      outputs: { out: { tensor: 'b', dtype: 'float32', shape: ['B'] } }, params: {},
    }, {
      id: 'fixed', opType: 'Identity', inputs: { input: 'z' },
      outputs: { out: { tensor: 'zz', dtype: 'int32', shape: [4] } }, params: {},
    }],
    outputs: ['b', 'zz'],
  });
}

function mixedDTypeDynamicSnapshot() {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: {
      a: { dtype: 'float32', shape: ['B'] },
      z: { dtype: 'int32', shape: ['B'] },
    },
    nodes: [{
      id: 'float', opType: 'Identity', inputs: { input: 'a' },
      outputs: { out: { tensor: 'b', dtype: 'float32', shape: ['B'] } }, params: {},
    }, {
      id: 'integer', opType: 'Identity', inputs: { input: 'z' },
      outputs: { out: { tensor: 'zz', dtype: 'int32', shape: ['B'] } }, params: {},
    }],
    outputs: ['b', 'zz'],
  });
}

function dropoutFanoutSnapshot() {
  return makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B'] } },
    nodes: [{
      id: 'dropout', opType: 'Dropout', inputs: { input: 'x' },
      outputs: { out: { tensor: 'd', dtype: 'float32', shape: ['B'] } },
      params: { p: 0.5 },
    }, {
      id: 'relu', opType: 'ReLU', inputs: { input: 'd' },
      outputs: { out: { tensor: 'a', dtype: 'float32', shape: ['B'] } }, params: {},
    }, {
      id: 'fanout', opType: 'Add', inputs: { a: 'd', b: 'a' },
      outputs: { out: { tensor: 'b', dtype: 'float32', shape: ['B'] } }, params: {},
    }],
    outputs: ['d', 'b'],
  });
}

function transformerSliceSnapshot() {
  const embedding = Array.from({ length: 8 * 4 }, (_, index) => {
    const token = Math.floor(index / 4);
    const feature = index % 4;
    return Math.fround((token - 3) * 0.17 + (feature - 1.5) * 0.09);
  });
  const norm = [1, 0.5, 1.5, 2];
  const projection = [
    0.5, -0.25, 0.75, 0.125,
    -0.4, 0.6, 0.2, -0.3,
    0.1, 0.2, -0.5, 0.9,
  ];
  const bias = [0.05, -0.1, 0.2];
  const snapshot = makeSnapshot({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 3 },
      S: { min: 1, max: 4 },
      H: { min: 2, max: 8 },
      O: { min: 1, max: 6 },
    },
    inputs: { ids: { dtype: 'int32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'embedding',
      opType: 'Embedding',
      inputs: { input: 'ids', weight: 'embedding.weight' },
      outputs: {
        out: { tensor: 'embedded', dtype: 'float32', shape: ['B', 'S', 'H'] },
      },
      params: {},
    }, {
      id: 'activation',
      opType: 'GELU',
      inputs: { input: 'embedded' },
      outputs: {
        out: { tensor: 'activated', dtype: 'float32', shape: ['B', 'S', 'H'] },
      },
      params: { approximate: 'tanh' },
    }, {
      id: 'normalization',
      opType: 'RMSNorm',
      inputs: { input: 'activated', weight: 'norm.weight' },
      outputs: {
        out: { tensor: 'normalized', dtype: 'float32', shape: ['B', 'S', 'H'] },
      },
      params: { eps: 1e-5, d_model: 4 },
    }, {
      id: 'projection',
      opType: 'Linear',
      inputs: {
        input: 'normalized',
        weight: 'projection.weight',
        bias: 'projection.bias',
      },
      outputs: {
        out: { tensor: 'logits', dtype: 'float32', shape: ['B', 'S', 'O'] },
      },
      params: { weight_layout: 'dout_din' },
    }],
    outputs: ['logits'],
  }, [{
    name: 'embedding.weight', dtype: 'float32', shape: [8, 4], values: embedding,
  }, {
    name: 'norm.weight', dtype: 'float32', shape: [4], values: norm,
  }, {
    name: 'projection.weight', dtype: 'float32', shape: [3, 4], values: projection,
  }, {
    name: 'projection.bias', dtype: 'float32', shape: [3], values: bias,
  }]);
  return { snapshot, embedding, norm, projection, bias };
}

function transformerReference(ids, weights) {
  const activated = new Float32Array(ids.length * 4);
  const normalized = new Float32Array(ids.length * 4);
  const output = new Float32Array(ids.length * 3);
  const c = 0.7978845608028654;
  for (let token = 0; token < ids.length; token++) {
    for (let feature = 0; feature < 4; feature++) {
      const x = weights.embedding[ids[token] * 4 + feature];
      activated[token * 4 + feature] = Math.fround(
        0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x))),
      );
    }
    let squareSum = 0;
    for (let feature = 0; feature < 4; feature++) {
      const value = activated[token * 4 + feature];
      squareSum += value * value;
    }
    const rms = Math.sqrt(squareSum / 4 + 1e-5);
    for (let feature = 0; feature < 4; feature++) {
      normalized[token * 4 + feature] = Math.fround(
        activated[token * 4 + feature] / rms * weights.norm[feature],
      );
    }
    for (let projected = 0; projected < 3; projected++) {
      let sum = 0;
      for (let feature = 0; feature < 4; feature++) {
        sum += normalized[token * 4 + feature] * weights.projection[projected * 4 + feature];
      }
      output[token * 3 + projected] = Math.fround(sum + weights.bias[projected]);
    }
  }
  return output;
}

function assertClose(actual, expected, tolerance = 2e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `index ${index}: expected ${expected[index]}, got ${actual[index]}`,
    );
  }
}

function allocateForRequest(request) {
  const elements = request.targetCapacityBytes / (request.dtype === 'float32' || request.dtype === 'int32' ? 4 : 1);
  if (request.dtype === 'float32') return new Float32Array(elements);
  if (request.dtype === 'int32') return new Int32Array(elements);
  if (request.dtype === 'int8') return new Int8Array(elements);
  return new Uint8Array(elements);
}

function preparedMetadataBytes(snapshot, plan) {
  const maximum = planCPUActivationArena(
    snapshot.graph,
    `${snapshot.definitionFingerprint}|maximum-domain`,
    maximumCPUActivationArenaTensors(snapshot.graph),
  );
  const layout = planCPUActivationArenaWithinMaximum(
    snapshot.graph,
    plan.signature,
    concreteCPUActivationArenaTensors(plan),
    maximum,
  );
  return cpuShapePlanMetadataBytes(plan) + cpuActivationArenaMetadataBytes(layout);
}

test('smaller best-fit fragmentation falls back to the proved maximum layout', () => {
  const graph = Object.freeze({
    fingerprint: 'fragmentation-fixture',
    dimensions: Object.freeze({}),
    tensors: Object.freeze({}),
    outputs: Object.freeze([]),
    nodes: Object.freeze([
      Object.freeze({ id: 'birth-a', inputs: Object.freeze({}),
        outputs: Object.freeze({ out: Object.freeze({ tensor: 'A' }) }) }),
      Object.freeze({ id: 'birth-bd', inputs: Object.freeze({}),
        outputs: Object.freeze({
          b: Object.freeze({ tensor: 'B' }),
          d: Object.freeze({ tensor: 'D' }),
        }) }),
      Object.freeze({ id: 'use-b', inputs: Object.freeze({ input: 'B' }), outputs: Object.freeze({}) }),
      Object.freeze({ id: 'use-d', inputs: Object.freeze({ input: 'D' }), outputs: Object.freeze({}) }),
      Object.freeze({ id: 'gap-4', inputs: Object.freeze({}), outputs: Object.freeze({}) }),
      Object.freeze({ id: 'gap-5', inputs: Object.freeze({}), outputs: Object.freeze({}) }),
      Object.freeze({ id: 'birth-c', inputs: Object.freeze({}),
        outputs: Object.freeze({ out: Object.freeze({ tensor: 'C' }) }) }),
      Object.freeze({ id: 'gap-7', inputs: Object.freeze({}), outputs: Object.freeze({}) }),
      Object.freeze({ id: 'use-c', inputs: Object.freeze({ input: 'C' }), outputs: Object.freeze({}) }),
    ]),
  });
  const tensors = (sizes) => Object.freeze(Object.fromEntries(
    Object.entries(sizes).map(([name, sizeBytes]) => [name, Object.freeze({
      name, kind: 'value', dtype: 'uint8', sizeBytes,
    })]),
  ));
  const maximum = planCPUActivationArena(
    graph,
    'maximum',
    tensors({ A: 10, B: 8, C: 15, D: 12 }),
  );
  const fragmented = planCPUActivationArena(
    graph,
    'smaller',
    tensors({ A: 5, B: 8, C: 4, D: 12 }),
  );
  const bounded = planCPUActivationArenaWithinMaximum(
    graph,
    'smaller',
    tensors({ A: 5, B: 8, C: 4, D: 12 }),
    maximum,
  );

  assert.equal(maximum.capacityBytes, 22);
  assert.deepEqual(Object.fromEntries(
    Object.entries(maximum.regions).map(([name, region]) => [name, region.offsetBytes]),
  ), { A: 0, B: 0, C: 0, D: 10 });
  assert.equal(fragmented.capacityBytes, 25,
    'concrete best-fit is non-monotone under smaller interval sizes');
  assert.equal(bounded.capacityBytes, 22);
  assert.deepEqual(Object.fromEntries(
    Object.entries(bounded.regions).map(([name, region]) => [name, region.offsetBytes]),
  ), { A: 0, B: 0, C: 0, D: 10 });
});

test('dynamic CPU slice matches a reference and reuses one capacity pool small -> large -> small', async () => {
  const weights = transformerSliceSnapshot();
  const context = new CPUShapeExecutionContext(weights.snapshot, {
    planCacheEntries: 2,
    planCacheMetadataBytes: 1024 * 1024,
    maxCapacityBytes: 4096,
  });
  const definitionId = weights.snapshot.definitionId;
  const weightRevisionId = weights.snapshot.weightRevisionId;

  const smallIds = Int32Array.of(1, 5);
  const small = await context.execute({ ids: { data: smallIds, shape: [1, 2] } });
  assert.deepEqual(small.get('logits').shape, [1, 2, 3]);
  assert.equal(small.get('logits').data.byteLength, 1 * 2 * 3 * 4);
  assertClose(small.get('logits').data, transformerReference(smallIds, weights));
  const stableSmallBytes = new Uint8Array(small.get('logits').data.buffer).slice();
  let inspection = context.inspect();
  assert.equal(inspection.capacities.currentBytes, 72);
  assert.equal(inspection.capacities.highWaterBytes, 72);
  assert.equal(inspection.capacities.growEvents, 1);
  assert.equal(inspection.capacities.grownTensorCount, 2);
  assert.equal(inspection.planCache.misses, 1);
  assert.deepEqual(inspection.invariantWeights, { tensorCount: 4, sizeBytes: 204 });

  const largeIds = Int32Array.of(0, 1, 2, 3, 4, 5, 6, 7);
  const large = await context.execute({ ids: { data: largeIds, shape: [2, 4] } });
  assert.deepEqual(large.get('logits').shape, [2, 4, 3]);
  assertClose(large.get('logits').data, transformerReference(largeIds, weights));
  inspection = context.inspect();
  assert.equal(inspection.capacities.currentBytes, 288);
  assert.equal(inspection.capacities.highWaterBytes, 288);
  assert.equal(inspection.capacities.growEvents, 2);
  assert.equal(inspection.capacities.grownTensorCount, 4);
  assert.equal(inspection.planCache.misses, 2);
  assert.deepEqual(inspection.invariantWeights, { tensorCount: 4, sizeBytes: 204 });

  const smallAgainIds = Int32Array.of(7, 2);
  const smallAgain = await context.execute({
    ids: { data: smallAgainIds, shape: [1, 2] },
  });
  assert.deepEqual(smallAgain.get('logits').shape, [1, 2, 3]);
  assert.equal(smallAgain.get('logits').data.length, 6);
  assertClose(smallAgain.get('logits').data, transformerReference(smallAgainIds, weights));
  inspection = context.inspect();
  assert.equal(inspection.capacities.currentBytes, 288);
  assert.equal(inspection.capacities.growEvents, 2);
  assert.equal(inspection.planCache.hits, 1);
  assert.equal(inspection.planCache.misses, 2);
  assert.deepEqual(inspection.planCache.signatures, [large.signature, small.signature]);
  assert.equal(weights.snapshot.definitionId, definitionId);
  assert.equal(weights.snapshot.weightRevisionId, weightRevisionId);
  assert.deepEqual(small.revision, context.revision);
  assert.deepEqual([...new Uint8Array(small.get('logits').data.buffer)], [...stableSmallBytes]);
  await context.close();
});

test('default CPU metadata budget is bounded for eight production-sized plans', async () => {
  const context = new CPUShapeExecutionContext(dynamicIdentitySnapshot());
  assert.equal(context.inspect().limits.planCacheEntries, 8);
  assert.equal(context.inspect().limits.planCacheMetadataBytes, 16 * 1024 * 1024);
  await context.close();
});

test('liveness arena preserves Dropout fanout and public outputs across later reuse', async () => {
  const context = new CPUShapeExecutionContext(dropoutFanoutSnapshot());
  const cases = [
    { input: Float32Array.of(-2), d: [-2], b: [-2] },
    { input: Float32Array.of(-4, -1, 2, 5), d: [-4, -1, 2, 5], b: [-4, -1, 4, 10] },
    { input: Float32Array.of(3, -7), d: [3, -7], b: [6, -7] },
  ];
  for (const fixture of cases) {
    const result = await context.execute({
      x: { data: fixture.input, shape: [fixture.input.length] },
    });
    assert.deepEqual([...result.get('d').data], fixture.d);
    assert.deepEqual([...result.get('b').data], fixture.b);
    assert.equal(result.get('d').data.length, fixture.input.length,
      'exact output views never expose a larger capacity tail');
  }
  assert.equal(context.inspect().capacities.growEvents, 2);
  await context.close();
});

test('metadata-only LRU has deterministic recency and entry-count eviction', async () => {
  const context = new CPUShapeExecutionContext(dynamicIdentitySnapshot(), {
    planCacheEntries: 2,
    planCacheMetadataBytes: 1024 * 1024,
  });
  const a = await context.execute({ x: { data: Float32Array.of(1), shape: [1, 1] } });
  const b = await context.execute({ x: { data: Float32Array.of(2, 3), shape: [1, 2] } });
  await context.execute({ x: { data: Float32Array.of(4), shape: [1, 1] } });
  assert.deepEqual(context.inspect().planCache.signatures, [b.signature, a.signature]);
  const c = await context.execute({ x: { data: Float32Array.of(5, 6, 7), shape: [1, 3] } });
  const inspection = context.inspect();
  assert.deepEqual(inspection.planCache.signatures, [a.signature, c.signature]);
  assert.equal(inspection.planCache.entryCount, 2);
  assert.equal(inspection.planCache.hits, 1);
  assert.equal(inspection.planCache.misses, 3);
  assert.equal(inspection.planCache.evictions, 1);
  assert.ok(inspection.planCache.metadataBytes <= inspection.limits.planCacheMetadataBytes);
  assert.equal(inspection.capacities.currentBytes, 32);
  assert.deepEqual(inspection.capacities.tensors.map(({ name, capacityBytes }) =>
    [name, capacityBytes]), [['@arena/float32', 32]]);
  assert.equal(c.get('y').data.byteLength, 12);
  assert.equal('graph' in inspection.planCache, false);
  assert.equal('arena' in inspection.planCache, false);
  await context.close();
});

test('metadata byte budget independently evicts plans and charges canonical bytes', async () => {
  const snapshot = dynamicIdentitySnapshot({ max: 16 });
  const aInputs = { x: { data: Float32Array.of(1), shape: [1, 1] } };
  const bInputs = { x: { data: new Float32Array(16), shape: [1, 16] } };
  const aBytes = preparedMetadataBytes(snapshot, snapshot.bindShapes(aInputs));
  const bBytes = preparedMetadataBytes(snapshot, snapshot.bindShapes(bInputs));
  const budget = Math.max(aBytes, bBytes);
  const context = new CPUShapeExecutionContext(snapshot, {
    planCacheEntries: 10,
    planCacheMetadataBytes: budget,
  });

  const a = await context.execute(aInputs);
  const b = await context.execute(bInputs);
  let inspection = context.inspect();
  assert.deepEqual(inspection.planCache.signatures, [b.signature]);
  assert.equal(inspection.planCache.metadataBytes, bBytes);
  assert.equal(inspection.planCache.evictions, 1);
  await context.execute(aInputs);
  inspection = context.inspect();
  assert.deepEqual(inspection.planCache.signatures, [a.signature]);
  assert.equal(inspection.planCache.metadataBytes, aBytes);
  assert.equal(inspection.planCache.evictions, 2);
  await context.close();
});

test('failed shaped bind and capacity-limit candidate leave the committed binding untouched', async () => {
  const context = new CPUShapeExecutionContext(dynamicIdentitySnapshot({ max: 4 }), {
    planCacheEntries: 4,
    maxCapacityBytes: 8,
  });
  const smallInputs = { x: { data: Float32Array.of(7), shape: [1, 1] } };
  const first = await context.execute(smallInputs);
  const committed = context.inspect();

  await assert.rejects(
    context.execute({ x: { data: Float32Array.of(1, 2), shape: [1, 1] } }),
    (error) => error?.code === 'BYTE_LENGTH_MISMATCH',
  );
  await assert.rejects(
    context.execute({ x: { data: new Float32Array(4), shape: [2, 2] } }),
    (error) => error instanceof CPUShapeExecutionContextError &&
      error.code === 'CAPACITY_LIMIT_EXCEEDED',
  );
  const afterFailures = context.inspect();
  assert.equal(afterFailures.currentSignature, first.signature);
  assert.deepEqual(afterFailures.planCache.signatures, committed.planCache.signatures);
  assert.equal(afterFailures.planCache.metadataBytes, committed.planCache.metadataBytes);
  assert.deepEqual(afterFailures.capacities.tensors, committed.capacities.tensors);
  assert.equal(afterFailures.capacities.currentBytes, committed.capacities.currentBytes);
  assert.equal(afterFailures.capacities.growEvents, committed.capacities.growEvents);
  assert.equal(afterFailures.executions.failed, 2);

  const later = await context.execute(smallInputs);
  assert.deepEqual([...later.get('y').data], [7]);
  await context.close();
});

test('partially allocated growth candidate rolls back on allocator failure', async () => {
  let rejectLargeIntegerArena = false;
  const allocations = [];
  const context = new CPUShapeExecutionContext(mixedDTypeDynamicSnapshot(), {
    maxCapacityBytes: 1024,
    capacityAllocator(request) {
      allocations.push([request.name, request.targetCapacityBytes]);
      if (rejectLargeIntegerArena && request.name === '@arena/int32' &&
          request.targetCapacityBytes > 8) {
        throw new Error('injected integer-arena allocation failure');
      }
      return allocateForRequest(request);
    },
  });
  const small = {
    a: { data: Float32Array.of(3), shape: [1] },
    z: { data: Int32Array.of(7), shape: [1] },
  };
  await context.execute(small);
  const committed = context.inspect();
  rejectLargeIntegerArena = true;
  await assert.rejects(
    context.execute({
      a: { data: new Float32Array(4), shape: [4] },
      z: { data: new Int32Array(4), shape: [4] },
    }),
    (error) => error instanceof CPUShapeExecutionContextError &&
      error.code === 'CAPACITY_ALLOCATION_FAILED',
  );
  const rolledBack = context.inspect();
  assert.equal(rolledBack.currentSignature, committed.currentSignature);
  assert.deepEqual(rolledBack.capacities.tensors, committed.capacities.tensors);
  assert.equal(rolledBack.capacities.currentBytes, committed.capacities.currentBytes);
  assert.equal(rolledBack.capacities.growEvents, committed.capacities.growEvents);
  assert.deepEqual(rolledBack.planCache.signatures, committed.planCache.signatures);
  assert.ok(allocations.some(([name, bytes]) => name === '@arena/float32' && bytes > 8));
  assert.ok(allocations.some(([name, bytes]) => name === '@arena/int32' && bytes > 8));

  rejectLargeIntegerArena = false;
  const recovered = await context.execute(small);
  assert.deepEqual([...recovered.get('b').data], [3]);
  assert.deepEqual([...recovered.get('zz').data], [7]);
  await context.close();
});

test('growth rejects a new slot that aliases a later reused capacity', async () => {
  const committedStorage = new Map();
  let injectOverlap = false;
  const context = new CPUShapeExecutionContext(mixedDTypeDynamicFixedSnapshot(), {
    maxCapacityBytes: 1024,
    capacityAllocator(request) {
      if (injectOverlap && request.name === '@arena/float32') {
        const fixed = committedStorage.get('@arena/int32');
        return new Float32Array(
          fixed.buffer,
          fixed.byteOffset,
          request.targetCapacityBytes / 4,
        );
      }
      const storage = allocateForRequest(request);
      committedStorage.set(request.name, storage);
      return storage;
    },
  });
  const fixed = Int32Array.of(10, 20, 30, 40);
  await context.execute({
    a: { data: Float32Array.of(1), shape: [1] },
    z: { data: fixed, shape: [4] },
  });
  const committed = context.inspect();
  injectOverlap = true;
  await assert.rejects(
    context.execute({
      a: { data: Float32Array.of(1, 2), shape: [2] },
      z: { data: fixed, shape: [4] },
    }),
    (error) => error instanceof CPUShapeExecutionContextError &&
      error.code === 'CAPACITY_STORAGE_MISMATCH' && /overlaps/.test(error.message),
  );
  assert.equal(context.inspect().currentSignature, committed.currentSignature);
  assert.deepEqual(context.inspect().capacities.tensors, committed.capacities.tensors);
  await context.close();
});

test('static context validates shaped views but bypasses dynamic plan lookup', async () => {
  const snapshot = staticIdentitySnapshot();
  const context = new CPUShapeExecutionContext(snapshot);
  const firstInput = Float32Array.of(1, 2, 3, 4, 5, 6);
  const first = await context.execute({ x: { data: firstInput, shape: [2, 3] } });
  const second = await context.execute({
    x: { data: Float32Array.of(6, 5, 4, 3, 2, 1), shape: [2, 3] },
  });
  const inspection = context.inspect();
  assert.equal(inspection.staticFastPath, true);
  assert.equal(inspection.currentSignature, snapshot.staticShapePlan.signature);
  assert.equal(inspection.planCache.entryCount, 0);
  assert.equal(inspection.planCache.hits, 0);
  assert.equal(inspection.planCache.misses, 0);
  assert.equal(inspection.capacities.growEvents, 1);
  assert.deepEqual([...first.get('y').data], [...firstInput]);
  assert.deepEqual([...second.get('y').data], [6, 5, 4, 3, 2, 1]);
  await assert.rejects(
    context.execute({ x: firstInput }),
    (error) => error?.code === 'INVALID_TENSOR_VIEW',
  );
  await context.close();
});

test('static Identity aliases its validated input capacity while results remain owned snapshots', async () => {
  const allocated = new Map();
  const context = new CPUShapeExecutionContext(staticIdentitySnapshot(), {
    capacityAllocator(request) {
      const storage = new Float32Array(request.targetCapacityBytes / Float32Array.BYTES_PER_ELEMENT);
      allocated.set(request.name, storage);
      return storage;
    },
  });
  const first = await context.execute({
    x: { data: Float32Array.of(1, 2, 3, 4, 5, 6), shape: [2, 3] },
  });
  assert.deepEqual([...allocated.get('x')], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...allocated.get('y')], [0, 0, 0, 0, 0, 0]);
  assert.deepEqual([...first.get('y').data], [1, 2, 3, 4, 5, 6]);

  await context.execute({
    x: { data: Float32Array.of(6, 5, 4, 3, 2, 1), shape: [2, 3] },
  });
  assert.deepEqual([...first.get('y').data], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...allocated.get('y')], [0, 0, 0, 0, 0, 0]);
  await context.close();
});

test('different contexts execute concurrently without sharing mutable binding or results', async () => {
  const snapshot = dynamicIdentitySnapshot({ max: 8 });
  const left = new CPUShapeExecutionContext(snapshot);
  const right = new CPUShapeExecutionContext(snapshot);
  const [leftResult, rightResult] = await Promise.all([
    left.execute({ x: { data: Float32Array.of(1, 2), shape: [1, 2] } }),
    right.execute({ x: { data: Float32Array.of(9, 8, 7, 6), shape: [2, 2] } }),
  ]);
  assert.deepEqual([...leftResult.get('y').data], [1, 2]);
  assert.deepEqual([...rightResult.get('y').data], [9, 8, 7, 6]);
  assert.notEqual(left.inspect().currentSignature, right.inspect().currentSignature);
  assert.deepEqual(leftResult.revision, rightResult.revision);

  leftResult.get('y').data[0] = -100;
  const rightAgain = await right.execute({
    x: { data: Float32Array.of(5, 4, 3, 2), shape: [2, 2] },
  });
  assert.deepEqual([...rightAgain.get('y').data], [5, 4, 3, 2]);
  assert.deepEqual([...rightResult.get('y').data], [9, 8, 7, 6]);
  await Promise.all([left.close(), right.close()]);
});

test('FIFO close drains accepted work, is idempotent, rejects new work, and preserves results', async () => {
  const context = new CPUShapeExecutionContext(dynamicIdentitySnapshot({ max: 4 }));
  const completionOrder = [];
  const firstPromise = context.execute({
    x: { data: Float32Array.of(1), shape: [1, 1] },
  }).then((result) => {
    completionOrder.push(1);
    return result;
  });
  const secondPromise = context.execute({
    x: { data: Float32Array.of(2, 3), shape: [1, 2] },
  }).then((result) => {
    completionOrder.push(2);
    return result;
  });
  const close = context.close();
  assert.strictEqual(context.close(), close);
  await assert.rejects(
    context.execute({ x: { data: Float32Array.of(4), shape: [1, 1] } }),
    (error) => error instanceof CPUShapeExecutionContextError && error.code === 'CONTEXT_CLOSED',
  );
  const [first, second] = await Promise.all([firstPromise, secondPromise]);
  await close;
  assert.deepEqual(completionOrder, [1, 2]);
  assert.equal(context.state, 'closed');
  assert.equal(context.inspect().capacities.currentBytes, 0);
  assert.equal(context.inspect().planCache.entryCount, 0);
  assert.deepEqual([...first.get('y').data], [1]);
  assert.deepEqual([...second.get('y').data], [2, 3]);
  second.get('y').data[0] = 99;
  assert.deepEqual([...first.get('y').data], [1]);
});
