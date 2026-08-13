import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BoundExecutionGraphError,
  createBoundExecutionGraph,
  createBoundExecutionGraphMaterializer,
} from '../ts/core/BoundExecutionGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';
import { resolveGraphShapes } from '../ts/core/ResolvedShapePlan.js';
import {
  concreteCPUActivationArenaTensors,
  planCPUActivationArena,
} from '../ts/core/CPUActivationArenaPlan.js';

function cloneStorage(value) {
  if (value instanceof Float32Array) return new Float32Array(value);
  if (value instanceof Int32Array) return new Int32Array(value);
  if (value instanceof Int8Array) return new Int8Array(value);
  if (value instanceof Uint8ClampedArray) return new Uint8ClampedArray(value);
  return new Uint8Array(value);
}

function safetensorsDType(dtype) {
  if (dtype === 'float32') return 'F32';
  if (dtype === 'int32') return 'I32';
  if (dtype === 'int8') return 'I8';
  return 'U8';
}

function makeLoadedWeight(definition, role) {
  const owned = cloneStorage(definition.data);
  const shape = Object.freeze([...definition.shape]);
  return Object.freeze({
    name: definition.name,
    dtype: definition.dtype,
    shape,
    sizeBytes: owned.byteLength,
    safetensorsSizeBytes: owned.byteLength,
    safetensorsDtype: safetensorsDType(definition.dtype),
    source: 'fixture.safetensors',
    sourceIndex: 0,
    role,
    get data() {
      return cloneStorage(owned);
    },
    copyData() {
      return cloneStorage(owned);
    },
    copyBytes() {
      return new Uint8Array(
        new Uint8Array(owned.buffer, owned.byteOffset, owned.byteLength),
      );
    },
  });
}

function freezeQuantization(quantizationByTensor) {
  return Object.freeze(Object.fromEntries(Object.entries(quantizationByTensor).map(
    ([name, value]) => value.scheme === 'per_tensor'
      ? [name, Object.freeze({ ...value })]
      : [name, Object.freeze({
        ...value,
        scales: Object.freeze([...value.scales]),
        zero_points: Object.freeze([...value.zero_points]),
      })],
  )));
}

function makePackage(document, definitions = [], quantizationByTensor = {}) {
  const descriptors = definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  const graph = parseGraphDocument(document, descriptors);
  const quantizationParameterNames = Object.freeze(graph.quantization === null
    ? []
    : [...new Set(Object.values(graph.quantization.tensors).flatMap((reference) =>
      [reference.scale_tensor, reference.zero_point_tensor]))].sort());
  const parameters = new Set(quantizationParameterNames);
  const definitionsByName = new Map(definitions.map((definition) =>
    [definition.name, definition]));
  const weights = Object.freeze(Object.fromEntries(Object.keys(graph.weights).map((name) => [
    name,
    makeLoadedWeight(
      definitionsByName.get(name),
      parameters.has(name) ? 'quantization_parameter' : 'model_weight',
    ),
  ])));
  return Object.freeze({
    graph,
    weights,
    quantizationByTensor: freezeQuantization(quantizationByTensor),
    quantizationParameterNames,
    weightFiles: Object.freeze([]),
    source: Object.freeze({
      graphUrl: 'graph.json',
      weightSources: Object.freeze(definitions.length === 0 ? [] : ['fixture.safetensors']),
    }),
  });
}

function constantIdentityDocument(shape = [2, 3], overrides = {}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape } },
      params: {},
    }],
    outputs: ['y'],
    ...overrides,
  };
}

function symbolicIdentityDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      S: { min: 2, max: 8, multiple_of: 2 },
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
  };
}

function quantizeDequantizeDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
    nodes: [{
      id: 'quantize',
      opType: 'QuantizeLinear',
      inputs: { input: 'x', scale: 'scale', zero_point: 'zero' },
      outputs: { out: { tensor: 'q', dtype: 'int8', shape: ['B', 4] } },
      params: {},
    }, {
      id: 'dequantize',
      opType: 'DequantizeLinear',
      inputs: { input: 'q', scale: 'scale', zero_point: 'zero' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 4] } },
      params: {},
    }],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        q: {
          scheme: 'per_tensor',
          scale_tensor: 'scale',
          zero_point_tensor: 'zero',
        },
      },
    },
  };
}

function assertBoundError(operation, code, pattern = undefined) {
  let caught;
  try {
    operation();
  } catch (error) {
    caught = error;
  }
  assert.ok(caught instanceof BoundExecutionGraphError,
    `expected BoundExecutionGraphError, got ${String(caught)}`);
  assert.equal(caught.code, code);
  assert.equal(typeof caught.path, 'string');
  assert.notEqual(caught.path.length, 0);
  if (pattern !== undefined) assert.match(caught.message, pattern);
  return caught;
}

function replacePlanTensor(plan, name, patch) {
  const replacement = Object.freeze({ ...plan.tensors[name], ...patch });
  const tensors = Object.freeze({ ...plan.tensors, [name]: replacement });
  const nodes = Object.freeze(plan.nodes.map((node) => Object.freeze({
    ...node,
    inputs: Object.freeze(Object.fromEntries(Object.entries(node.inputs).map(
      ([port, descriptor]) => [port, descriptor.name === name ? replacement : descriptor],
    ))),
    outputs: Object.freeze(Object.fromEntries(Object.entries(node.outputs).map(
      ([port, descriptor]) => [port, descriptor.name === name ? replacement : descriptor],
    ))),
  })));
  const outputs = Object.freeze(plan.outputs.map((descriptor) =>
    descriptor.name === name ? replacement : descriptor));
  return Object.freeze({ ...plan, tensors, nodes, outputs });
}

test('constant binding publishes a frozen wrapper around a concrete CPU-parity graph', async () => {
  const weightDefinition = {
    name: 'unused.weight',
    dtype: 'float32',
    shape: [2],
    data: Float32Array.of(7, 9),
  };
  const source = makePackage(constantIdentityDocument(), [weightDefinition]);
  const values = Float32Array.of(1, 2, 3, 4, 5, 6);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: values, shape: [2, 3] },
  });
  const bound = createBoundExecutionGraph(source, plan);

  assert.equal(Object.isFrozen(bound), true);
  assert.equal(Object.isFrozen(bound.logicalBytes), true);
  assert.equal(Object.isFrozen(bound.graph), false);
  assert.strictEqual(bound.plan, plan);
  assert.equal(bound.snapshot, null);
  assert.deepEqual(bound.revision, {
    definitionFingerprint: source.graph.fingerprint,
    definitionId: null,
    topologyRevision: null,
    weightRevision: null,
    weightRevisionId: null,
  });
  assert.equal(bound.graphFingerprint, source.graph.fingerprint);
  assert.equal(bound.signature, 'v1|1:x|2:2,3');
  assert.deepEqual(bound.inputs, [{
    name: 'x', kind: 'input', dtype: 'float32', shape: [2, 3],
    elementCount: 6, sizeBytes: 24,
  }]);
  assert.deepEqual(bound.outputs, [{
    name: 'y', kind: 'value', dtype: 'float32', shape: [2, 3],
    elementCount: 6, sizeBytes: 24,
  }]);
  assert.deepEqual(bound.logicalBytes, { activations: 48, weights: 8, total: 56 });
  assert.equal(bound.graph.getTensor('x').buffer, undefined);
  assert.equal(bound.graph.getTensor('y').buffer, undefined);
  assert.deepEqual([...bound.graph.getTensor('unused.weight').buffer], [7, 9]);
  assert.deepEqual(bound.graph.outputNames, ['y']);
  assert.deepEqual(bound.graph.nodes[0].outputs.out.shape, [2, 3]);

  values.fill(99);
  const input = Float32Array.of(-1, 0, 1, 2, 3, 4);
  const cpu = new CPUEngine();
  cpu.allocateGraph(bound.graph);
  const result = await cpu.execute({ x: input });
  assert.deepEqual([...result.y], [...input]);
});

test('two symbolic bindings keep one logical identity but independent concrete graphs and storage', async () => {
  const source = makePackage(symbolicIdentityDocument());
  const smallPlan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const largePlan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(24), shape: [3, 8] },
  });
  const small = createBoundExecutionGraph(source, smallPlan);
  const large = createBoundExecutionGraph(source, largePlan);

  assert.equal(small.graphFingerprint, large.graphFingerprint);
  assert.notEqual(small.signature, large.signature);
  assert.notStrictEqual(small.graph, large.graph);
  for (const name of ['x', 'y']) {
    assert.notStrictEqual(small.graph.getTensor(name), large.graph.getTensor(name));
  }
  assert.deepEqual(small.graph.getTensor('y').shape, [1, 2]);
  assert.deepEqual(large.graph.getTensor('y').shape, [3, 8]);

  const smallCpu = new CPUEngine();
  const largeCpu = new CPUEngine();
  smallCpu.allocateGraph(small.graph);
  largeCpu.allocateGraph(large.graph);
  assert.notStrictEqual(small.graph.getTensor('x').buffer, large.graph.getTensor('x').buffer);
  assert.notStrictEqual(small.graph.getTensor('y').buffer, large.graph.getTensor('y').buffer);
  const smallInput = Float32Array.of(3, 4);
  const largeInput = Float32Array.from({ length: 24 }, (_, index) => index - 12);
  const [smallResult, largeResult] = await Promise.all([
    smallCpu.execute({ x: smallInput }),
    largeCpu.execute({ x: largeInput }),
  ]);
  assert.deepEqual([...smallResult.y], [...smallInput]);
  assert.deepEqual([...largeResult.y], [...largeInput]);
});

test('repeated materialization privately copies every weight and never routes weights through activation storage', () => {
  const definition = {
    name: 'private.weight',
    dtype: 'float32',
    shape: [3],
    data: Float32Array.of(2, 4, 8),
  };
  const source = makePackage(constantIdentityDocument([1, 2]), [definition]);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const requests = [];
  const options = {
    activationStorageFactory(request) {
      requests.push(request.name);
      return undefined;
    },
  };
  const first = createBoundExecutionGraph(source, plan, options);
  const second = createBoundExecutionGraph(source, plan, options);
  const firstWeight = first.graph.getTensor('private.weight').buffer;
  const secondWeight = second.graph.getTensor('private.weight').buffer;

  assert.deepEqual(requests, ['x', 'y', 'x', 'y']);
  assert.notStrictEqual(firstWeight, secondWeight);
  assert.notStrictEqual(firstWeight.buffer, secondWeight.buffer);
  firstWeight[0] = 99;
  assert.deepEqual([...secondWeight], [2, 4, 8]);
  assert.deepEqual([...source.weights['private.weight'].data], [2, 4, 8]);
});

test('snapshot context can explicitly reuse one private invariant weight revision across shapes', () => {
  const source = makePackage(symbolicIdentityDocument(), [{
    name: 'private.weight',
    dtype: 'float32',
    shape: [3],
    data: Float32Array.of(2, 4, 8),
  }]);
  const snapshot = Model.capture(source);
  const smallPlan = snapshot.bindShapes({
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const largePlan = snapshot.bindShapes({
    x: { data: new Float32Array(24), shape: [3, 8] },
  });
  const first = createBoundExecutionGraph(snapshot, smallPlan);
  const invariant = first.graph.getTensor('private.weight').buffer;
  const requests = [];
  const second = createBoundExecutionGraph(snapshot, largePlan, {
    invariantWeightStorageFactory(request) {
      requests.push(request);
      return invariant;
    },
  });

  assert.strictEqual(second.graph.getTensor('private.weight').buffer, invariant);
  assert.deepEqual(requests.map((request) => ({
    graphFingerprint: request.graphFingerprint,
    definitionId: request.definitionId,
    weightRevisionId: request.weightRevisionId,
    name: request.name,
    dtype: request.dtype,
    shape: request.shape,
    elementCount: request.elementCount,
    sizeBytes: request.sizeBytes,
    role: request.role,
  })), [{
    graphFingerprint: snapshot.definitionFingerprint,
    definitionId: snapshot.definitionId,
    weightRevisionId: snapshot.weightRevisionId,
    name: 'private.weight',
    dtype: 'float32',
    shape: [3],
    elementCount: 3,
    sizeBytes: 12,
    role: 'model_weight',
  }]);
  assert.equal(Object.isFrozen(requests[0]), true);
  assert.equal(Object.isFrozen(requests[0].shape), true);

  assertBoundError(
    () => createBoundExecutionGraph(snapshot, largePlan, {
      invariantWeightStorageFactory: () => new Float32Array(2),
    }),
    'WEIGHT_PAYLOAD_MISMATCH',
    /exact float32 storage with 12 bytes/,
  );
  assertBoundError(
    () => createBoundExecutionGraph(source, largePlan, {
      invariantWeightStorageFactory: () => invariant,
    }),
    'INVALID_PACKAGE',
    /Model/,
  );
});

test('snapshot materializer validates invariant weights once and fully rejects forged plan lookalikes', () => {
  const source = makePackage(symbolicIdentityDocument(), [{
    name: 'private.weight',
    dtype: 'float32',
    shape: [3],
    data: Float32Array.of(2, 4, 8),
  }]);
  const snapshot = Model.capture(source);
  const invariant = snapshot.copyWeightData('private.weight');
  let weightRequests = 0;
  const materializer = createBoundExecutionGraphMaterializer(snapshot, {
    invariantWeightStorageFactory(request) {
      weightRequests++;
      assert.equal(request.name, 'private.weight');
      return invariant;
    },
  });
  const smallPlan = snapshot.bindShapes({
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const largePlan = snapshot.bindShapes({
    x: { data: new Float32Array(24), shape: [3, 8] },
  });
  const small = materializer.materialize(smallPlan);
  const large = materializer.materialize(largePlan);

  assert.equal(weightRequests, 1);
  assert.notStrictEqual(small.graph, large.graph);
  assert.strictEqual(small.graph.getTensor('private.weight').buffer, invariant);
  assert.strictEqual(large.graph.getTensor('private.weight').buffer, invariant);
  assert.deepEqual(small.graph.getTensor('y').shape, [1, 2]);
  assert.deepEqual(large.graph.getTensor('y').shape, [3, 8]);

  const forged = replacePlanTensor(largePlan, 'x', { elementCount: 99 });
  assertBoundError(
    () => materializer.materialize(forged),
    'PLAN_DESCRIPTOR_MISMATCH',
    /element\/byte totals/,
  );
  assert.equal(weightRequests, 1,
    'a forged plan must fail complete validation before invariant storage is revisited');
});

test('shared invariant weight storage accepts adjacent views and rejects exact byte overlap', () => {
  const definitions = ['alpha', 'beta', 'gamma'].map((name, index) => ({
    name,
    dtype: 'float32',
    shape: [2],
    data: Float32Array.of(index + 1, index + 2),
  }));
  const snapshot = Model.capture(makePackage(constantIdentityDocument([1, 2]), definitions));
  const plan = snapshot.bindShapes({
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const BufferType = typeof SharedArrayBuffer === 'function'
    ? SharedArrayBuffer
    : ArrayBuffer;
  const adjacent = new BufferType(24);
  const offsets = new Map([['alpha', 0], ['beta', 8], ['gamma', 16]]);
  const bound = createBoundExecutionGraph(snapshot, plan, {
    invariantWeightStorageFactory({ name }) {
      return new Float32Array(adjacent, offsets.get(name), 2);
    },
  });

  assert.strictEqual(bound.graph.getTensor('alpha').buffer.buffer, adjacent);
  assert.strictEqual(bound.graph.getTensor('beta').buffer.buffer, adjacent);
  assert.strictEqual(bound.graph.getTensor('gamma').buffer.buffer, adjacent);

  assertBoundError(
    () => createBoundExecutionGraph(snapshot, plan, {
      invariantWeightStorageFactory({ name }) {
        return new Float32Array(adjacent, name === 'beta' ? 4 : offsets.get(name), 2);
      },
    }),
    'WEIGHT_PAYLOAD_MISMATCH',
    /invariant weight storage 'beta'.*overlaps storage for 'alpha'/s,
  );

  const nestedDefinitions = [{
    name: 'alpha', dtype: 'float32', shape: [4], data: new Float32Array(4),
  }, {
    name: 'beta', dtype: 'float32', shape: [2], data: new Float32Array(2),
  }, {
    name: 'gamma', dtype: 'float32', shape: [2], data: new Float32Array(2),
  }];
  const nestedSnapshot = Model.capture(makePackage(
    constantIdentityDocument([1, 2]), nestedDefinitions,
  ));
  const nestedPlan = nestedSnapshot.bindShapes({
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  assertBoundError(
    () => createBoundExecutionGraph(nestedSnapshot, nestedPlan, {
      invariantWeightStorageFactory({ name }) {
        const offset = name === 'alpha' ? 0 : name === 'beta' ? 16 : 4;
        const length = name === 'alpha' ? 4 : 2;
        return new Float32Array(adjacent, offset, length);
      },
    }),
    'WEIGHT_PAYLOAD_MISMATCH',
    /invariant weight storage 'gamma'.*overlaps storage for 'alpha'/s,
  );
});

test('capacity providers can publish exact logical views without exposing capacity tails', async () => {
  const source = makePackage(symbolicIdentityDocument());
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(8), shape: [2, 4] },
  });
  const capacity = new Float32Array(64);
  const requests = [];
  let offset = 5;
  const bound = createBoundExecutionGraph(source, plan, {
    activationStorageFactory(request) {
      requests.push(request);
      const view = new Float32Array(capacity.buffer, offset * 4, request.elementCount);
      offset += request.elementCount + 3;
      return view;
    },
  });

  assert.deepEqual(requests.map((request) => request.name), ['x', 'y']);
  for (const request of requests) {
    assert.equal(Object.isFrozen(request), true);
    assert.equal(Object.isFrozen(request.shape), true);
    assert.equal(request.signature, plan.signature);
    assert.equal(request.sizeBytes, 32);
  }
  const inputView = bound.graph.getTensor('x').buffer;
  const outputView = bound.graph.getTensor('y').buffer;
  assert.strictEqual(inputView.buffer, capacity.buffer);
  assert.strictEqual(outputView.buffer, capacity.buffer);
  assert.equal(inputView.byteLength, 32);
  assert.equal(outputView.byteLength, 32);
  assert.notEqual(inputView.byteOffset, outputView.byteOffset);

  const values = Float32Array.from({ length: 8 }, (_, index) => index + 0.5);
  const cpu = new CPUEngine();
  cpu.allocateGraph(bound.graph);
  const result = await cpu.execute({ x: values });
  assert.strictEqual(result.y, outputView);
  assert.deepEqual([...result.y], [...values]);
});

test('plan, descriptor, and weight mismatches fail before the first activation factory call', () => {
  const definition = {
    name: 'weight', dtype: 'float32', shape: [1], data: Float32Array.of(3),
  };
  const source = makePackage(constantIdentityDocument([1, 2]), [definition]);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  let calls = 0;
  const options = { activationStorageFactory() { calls++; return undefined; } };

  const wrongFingerprint = Object.freeze({
    ...plan,
    graphFingerprint: 'volvox-logical-graph/v1|wrong',
  });
  assertBoundError(
    () => createBoundExecutionGraph(source, wrongFingerprint, options),
    'PLAN_FINGERPRINT_MISMATCH',
  );
  assert.equal(calls, 0);

  const wrongElements = replacePlanTensor(plan, 'x', { elementCount: 99 });
  assertBoundError(
    () => createBoundExecutionGraph(source, wrongElements, options),
    'PLAN_DESCRIPTOR_MISMATCH',
    /element\/byte totals/,
  );
  assert.equal(calls, 0);

  const badWeight = Object.freeze({ ...source.weights.weight, sizeBytes: 8 });
  const wrongPackage = Object.freeze({
    ...source,
    weights: Object.freeze({ ...source.weights, weight: badWeight }),
  });
  assertBoundError(
    () => createBoundExecutionGraph(wrongPackage, plan, options),
    'WEIGHT_PAYLOAD_MISMATCH',
  );
  assert.equal(calls, 0);
});

test('invalid activation factory views never produce a partially published wrapper', () => {
  const source = makePackage(constantIdentityDocument([1, 4]));
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(4), shape: [1, 4] },
  });
  let published;
  assertBoundError(
    () => {
      published = createBoundExecutionGraph(source, plan, {
        activationStorageFactory() {
          return new Int32Array(4);
        },
      });
    },
    'ACTIVATION_STORAGE_MISMATCH',
    /exact float32 typed-array view/,
  );
  assert.equal(published, undefined);

  assertBoundError(
    () => createBoundExecutionGraph(source, plan, {
      activationStorageFactory() {
        return new Float32Array(5);
      },
    }),
    'ACTIVATION_STORAGE_MISMATCH',
    /16 bytes/,
  );

  const overlappingCapacity = new Float32Array(4);
  assertBoundError(
    () => createBoundExecutionGraph(source, plan, {
      activationStorageFactory() {
        return overlappingCapacity;
      },
    }),
    'ACTIVATION_STORAGE_MISMATCH',
    /overlaps supplied storage.*non-aliased/s,
  );
});

test('fingerprint/signature-bound liveness layouts permit only topology-disjoint aliases', () => {
  const source = makePackage(symbolicIdentityDocument());
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(4), shape: [1, 4] },
  });
  const layout = planCPUActivationArena(
    source.graph,
    plan.signature,
    concreteCPUActivationArenaTensors(plan),
  );
  const arena = new Float32Array(layout.capacityByArena.float32 / 4);
  const materialize = (candidate) => createBoundExecutionGraph(source, plan, {
    activationStorageLayout: candidate,
    activationStorageFactory(request) {
      const region = candidate.regions[request.name];
      return new Float32Array(
        arena.buffer,
        region.offsetBytes,
        request.elementCount,
      );
    },
  });
  const bound = materialize(layout);
  assert.notEqual(
    bound.graph.getTensor('x').buffer.byteOffset,
    bound.graph.getTensor('y').buffer.byteOffset,
    'one node must read its input while its output region is already live',
  );

  const overlappingY = Object.freeze({
    ...layout.regions.y,
    offsetBytes: layout.regions.x.offsetBytes,
  });
  const invalidRegions = Object.freeze({ ...layout.regions, y: overlappingY });
  const invalid = Object.freeze({ ...layout, regions: invalidRegions });
  assertBoundError(
    () => materialize(invalid),
    'ACTIVATION_STORAGE_MISMATCH',
    /overlaps simultaneously live activation/,
  );
  assertBoundError(
    () => materialize(Object.freeze({ ...layout, signature: `${layout.signature}|wrong` })),
    'ACTIVATION_STORAGE_MISMATCH',
    /exact graph fingerprint and signature/,
  );
});

test('snapshot overload retains and materializes one exact derived weight revision', () => {
  const document = constantIdentityDocument([1, 2]);
  const initialPackage = makePackage(document, [{
    name: 'revision.weight',
    dtype: 'float32',
    shape: [2],
    data: Float32Array.of(1, 2),
  }]);
  const initial = Model.capture(initialPackage);
  const derivedPackage = makePackage(document, [{
    name: 'revision.weight',
    dtype: 'float32',
    shape: [2],
    data: Float32Array.of(13, 21),
  }]);
  const derived = Model.derive(derivedPackage, initial);
  const plan = derived.bindShapes({
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const requested = [];
  const bound = createBoundExecutionGraph(derived, plan, {
    activationStorageFactory(request) {
      requested.push(request.name);
      return undefined;
    },
  });

  assert.strictEqual(bound.snapshot, derived);
  assert.deepEqual(bound.revision, {
    definitionFingerprint: derived.definitionFingerprint,
    definitionId: derived.definitionId,
    topologyRevision: derived.topologyRevision,
    weightRevision: derived.weightRevision,
    weightRevisionId: derived.weightRevisionId,
  });
  assert.equal(derived.definitionId, initial.definitionId);
  assert.equal(derived.weightRevision, initial.weightRevision + 1);
  assert.notEqual(derived.weightRevisionId, initial.weightRevisionId);
  assert.deepEqual(requested, ['x', 'y']);
  assert.deepEqual([...bound.graph.getTensor('revision.weight').buffer], [13, 21]);
  assert.deepEqual([...initial.copyWeightData('revision.weight')], [1, 2]);
  bound.graph.getTensor('revision.weight').buffer[0] = 99;
  assert.deepEqual([...derived.copyWeightData('revision.weight')], [13, 21]);
});

test('kernel-facing dense layout preserves canonical Linear semantics for square weights', async () => {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
    nodes: [{
      id: 'linear',
      opType: 'Linear',
      inputs: { input: 'x', weight: 'weight' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
      params: {},
    }],
    outputs: ['y'],
  };
  const source = makePackage(document, [{
    name: 'weight',
    dtype: 'float32',
    shape: [2, 2],
    // Canonical Linear default is output-major [d_out, d_in].
    data: Float32Array.of(1, 2, 3, 4),
  }]);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(2), shape: [1, 2] },
  });
  const bound = createBoundExecutionGraph(source, plan);

  assert.equal(bound.graph.nodes[0].wLayout, 'dout');
  const cpu = new CPUEngine();
  cpu.allocateGraph(bound.graph);
  const result = await cpu.execute({ x: Float32Array.of(1, 10) });
  assert.deepEqual([...result.y], [21, 43]);
});

test('selected output order is exact and concrete graph params are private mutable copies', () => {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 2 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 3] } },
    nodes: [{
      id: 'first',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'first.value', dtype: 'float32', shape: ['B', 3] } },
      params: {},
    }, {
      id: 'second',
      opType: 'LeakyReLU',
      inputs: { input: 'first.value' },
      outputs: { out: { tensor: 'second.value', dtype: 'float32', shape: ['B', 3] } },
      params: { alpha: 0.2 },
    }],
    outputs: ['second.value', 'first.value'],
  };
  const source = makePackage(document);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(6), shape: [2, 3] },
  });
  const bound = createBoundExecutionGraph(source, plan);

  assert.deepEqual(bound.outputs.map((output) => output.name),
    ['second.value', 'first.value']);
  assert.deepEqual(bound.graph.outputNames, ['second.value', 'first.value']);
  assert.deepEqual(Object.keys(bound.graph.nodes[0].outputs), ['out']);
  assert.equal('outputs_shape' in bound.graph.nodes[0], false);
  assert.deepEqual(bound.graph.nodes[1].params, { alpha: 0.2 });
  assert.notStrictEqual(bound.graph.nodes[1].params, source.graph.nodes[1].params);
  bound.graph.nodes[1].params.alpha = 0.9;
  assert.deepEqual(source.graph.nodes[1].params, { alpha: 0.2 });
  assert.deepEqual(plan.nodes[1].params, { alpha: 0.2 });
});

test('Q/DQ materialization preserves hydrated edge metadata and private parameter weights', async () => {
  const definitions = [{
    name: 'scale', dtype: 'float32', shape: [1], data: Float32Array.of(0.25),
  }, {
    name: 'zero', dtype: 'int8', shape: [1], data: Int8Array.of(0),
  }];
  const quantization = Object.freeze({
    q: Object.freeze({ scheme: 'per_tensor', scale: 0.25, zero_point: 0 }),
  });
  const source = makePackage(quantizeDequantizeDocument(), definitions, quantization);
  const plan = resolveGraphShapes(source.graph, {
    x: { data: new Float32Array(4), shape: [1, 4] },
  }, source.quantizationByTensor);
  const requested = [];
  const bound = createBoundExecutionGraph(source, plan, {
    activationStorageFactory(request) {
      requested.push(request.name);
      return undefined;
    },
  });

  assert.deepEqual(requested, ['q', 'x', 'y']);
  assert.deepEqual(bound.graph.getTensor('q').quantization, {
    scheme: 'per_tensor', scale: 0.25, zero_point: 0,
  });
  assert.equal(bound.graph.getTensor('x').quantization, null);
  assert.equal(bound.graph.getTensor('y').quantization, null);
  assert.strictEqual(bound.graph.nodes[1].inputs.input, bound.graph.getTensor('q'));
  assert.deepEqual([...bound.graph.getTensor('scale').buffer], [0.25]);
  assert.deepEqual([...bound.graph.getTensor('zero').buffer], [0]);

  const cpu = new CPUEngine();
  cpu.allocateGraph(bound.graph);
  const result = await cpu.execute({ x: Float32Array.of(-1, -0.1, 0.1, 1) });
  assert.deepEqual([...result.y], [-1, 0, 0, 1]);
});
