import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Model,
  ModelError,
} from '../ts/core/Model.js';
import { ModelLoader } from '../ts/core/ModelLoader.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { ResolvedShapePlanError } from '../ts/core/ResolvedShapePlan.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';

function identityDocument({ dimensions = {}, shape = [2, 3] } = {}) {
  return {
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
  };
}

function quantizedDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 8 } },
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

function typedData(dtype, values) {
  if (dtype === 'float32') return Float32Array.from(values);
  if (dtype === 'int32') return Int32Array.from(values);
  if (dtype === 'int8') return Int8Array.from(values);
  return Uint8Array.from(values);
}

function structuralSource(document, definitions = [], quantizationByTensor = {}) {
  const descriptors = definitions.map(({ name, dtype, shape }) => ({ name, dtype, shape }));
  const graph = parseGraphDocument(document, descriptors);
  const weights = Object.fromEntries(definitions.map((definition) => [
    definition.name,
    {
      name: definition.name,
      dtype: definition.dtype,
      shape: [...definition.shape],
      data: typedData(definition.dtype, definition.values),
      role: definition.role,
    },
  ]));
  return { graph, weights, quantizationByTensor };
}

function assertDeepFrozen(value, seen = new Set()) {
  if (value == null || typeof value !== 'object' || seen.has(value) || ArrayBuffer.isView(value)) {
    return;
  }
  seen.add(value);
  assert.equal(Object.isFrozen(value), true, `expected frozen ${Object.prototype.toString.call(value)}`);
  for (const key of Reflect.ownKeys(value)) assertDeepFrozen(value[key], seen);
}

function reachesReference(root, target, seen = new Set()) {
  if (root === target) return true;
  if (root == null || typeof root !== 'object' || seen.has(root)) return false;
  seen.add(root);
  return Reflect.ownKeys(root).some((key) => reachesReference(root[key], target, seen));
}

test('capture accepts a loaded logical package and owns graph and fixed-weight bytes', async () => {
  const file = SafetensorsFile.empty({ metadata: { fixture: 'logical-snapshot' } });
  file.addTensor('weight', 'F32', [2], Float32Array.of(3, 5));
  const safetensors = file.toArrayBuffer();
  const loaded = await ModelLoader.load('model.safetensors', {
    fetch: async (source) => source === 'graph.json'
      ? { ok: true, text: async () => JSON.stringify(identityDocument()) }
      : { ok: true, arrayBuffer: async () => safetensors },
  });
  const packageData = loaded.weights.weight.data;
  const snapshot = Model.capture(loaded);

  assert.notEqual(snapshot.graph, loaded.graph);
  assert.equal(snapshot.graph.fingerprint, loaded.graph.fingerprint);
  assert.deepEqual(snapshot.weightNames, ['weight']);
  assert.deepEqual(snapshot.weightDescriptors, [{
    name: 'weight',
    dtype: 'float32',
    shape: [2],
    sizeBytes: 8,
    role: 'model_weight',
    bank: null,
  }]);
  packageData.fill(99);
  assert.deepEqual([...snapshot.copyWeightData('weight')], [3, 5]);
  const firstCopy = snapshot.copyWeightData('weight');
  firstCopy[0] = -100;
  assert.deepEqual([...snapshot.copyWeightData('weight')], [3, 5]);
  const bytes = snapshot.copyWeightBytes('weight');
  bytes.fill(0);
  assert.deepEqual([...snapshot.copyWeightData('weight')], [3, 5]);
  assert.equal(reachesReference(snapshot, packageData), false);
  assertDeepFrozen(snapshot);
});

test('logical introspection and fixed-domain static prebinding are immutable and allocation-free', () => {
  const source = structuralSource(identityDocument({
    dimensions: { S: { min: 2, max: 3, multiple_of: 2 } },
    shape: ['S', 4],
  }));
  const snapshot = Model.capture(source);

  assert.equal(snapshot.isStatic, true);
  assert.equal(snapshot.staticShapePlan.signature, 'v1|1:x|2:2,4');
  assert.deepEqual(snapshot.staticShapePlan.symbols, { S: 2 });
  assert.deepEqual(snapshot.inputNames, ['x']);
  assert.deepEqual(snapshot.inputDescriptors.map(({ name, dtype, shape }) => ({ name, dtype, shape })), [{
    name: 'x', dtype: 'float32', shape: ['S', 4],
  }]);
  assert.deepEqual(snapshot.outputNames, ['y']);
  assert.deepEqual(snapshot.outputDescriptors.map(({ name, dtype, shape }) => ({ name, dtype, shape })), [{
    name: 'y', dtype: 'float32', shape: ['S', 4],
  }]);
  assert.equal(snapshot.definitionFingerprint, source.graph.fingerprint);
  assert.equal(snapshot.shapeDomainProof.supported, true);
  assertDeepFrozen(snapshot.inputDescriptors);
  assertDeepFrozen(snapshot.outputDescriptors);
  assertDeepFrozen(snapshot.staticShapePlan);

  const definitionId = snapshot.definitionId;
  const weightRevisionId = snapshot.weightRevisionId;
  const plan = snapshot.bindShapes({
    x: { data: new Float32Array(8), shape: [2, 4] },
  });
  assert.equal(plan, snapshot.staticShapePlan);
  assert.equal(snapshot.definitionId, definitionId);
  assert.equal(snapshot.weightRevisionId, weightRevisionId);
  assert.throws(
    () => snapshot.bindShapes({ x: { data: new Float32Array(7), shape: [2, 4] } }),
    (error) => error instanceof ResolvedShapePlanError && error.code === 'INPUT_BINDING_FAILED',
  );
});

test('weight-only constant graph pre-resolves the empty public-input signature', () => {
  const source = structuralSource({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: {},
    nodes: [],
    outputs: ['constant'],
  }, [{
    name: 'constant', dtype: 'float32', shape: [3], values: [2, 4, 8],
  }]);
  const snapshot = Model.capture(source);

  assert.equal(snapshot.isStatic, true);
  assert.equal(snapshot.staticShapePlan.signature, 'v1|');
  assert.deepEqual(snapshot.staticShapePlan.outputs.map((output) => output.name), ['constant']);
  assert.equal(snapshot.staticShapePlan.weightBytes, 12);
  assert.equal(snapshot.staticShapePlan.logicalActivationBytes, 0);
  assert.equal(snapshot.bindShapes({}), snapshot.staticShapePlan);
  assert.throws(
    () => snapshot.bindShapes({ extra: { data: Float32Array.of(1), shape: [1] } }),
    (error) => error instanceof ResolvedShapePlanError && error.code === 'INPUT_BINDING_FAILED',
  );
});

test('two dynamic bindings are independent and cannot retain caller data', () => {
  const source = structuralSource(identityDocument({
    dimensions: {
      B: { min: 1, max: 4 },
      S: { min: 1, max: 16 },
    },
    shape: ['B', 'S'],
  }));
  const snapshot = Model.capture(source);
  const definitionId = snapshot.definitionId;
  const weightRevisionId = snapshot.weightRevisionId;
  const firstData = new Float32Array(6);
  const secondData = new Float32Array(8);
  const firstShape = [2, 3];
  const secondShape = [1, 8];

  const first = snapshot.bindShapes({ x: { data: firstData, shape: firstShape } });
  const second = snapshot.bindShapes({ x: { data: secondData, shape: secondShape } });

  assert.equal(snapshot.isStatic, false);
  assert.equal(snapshot.staticShapePlan, null);
  assert.notEqual(first, second);
  assert.deepEqual(first.outputs[0].shape, [2, 3]);
  assert.deepEqual(second.outputs[0].shape, [1, 8]);
  assert.equal(first.signature, 'v1|1:x|2:2,3');
  assert.equal(second.signature, 'v1|1:x|2:1,8');
  assert.equal(reachesReference(first, firstData), false);
  assert.equal(reachesReference(first, firstShape), false);
  assert.equal(reachesReference(second, secondData), false);
  assert.equal(reachesReference(second, secondShape), false);
  assert.equal(snapshot.definitionId, definitionId);
  assert.equal(snapshot.weightRevisionId, weightRevisionId);
});

test('derive preserves only matching definition identity and always owns a new weight revision', () => {
  const firstData = Float32Array.of(1, 2);
  const definition = identityDocument({
    dimensions: { B: { min: 1, max: 4 } },
    shape: ['B', 3],
  });
  const firstSource = structuralSource(definition, [{
    name: 'weight', dtype: 'float32', shape: [2], values: firstData,
  }]);
  const first = Model.capture(firstSource);
  const secondSource = structuralSource(definition, [{
    name: 'weight', dtype: 'float32', shape: [2], values: [7, 11],
  }]);
  const second = Model.derive(secondSource, first);

  assert.equal(first.topologyRevision, 1);
  assert.equal(first.weightRevision, 1);
  assert.equal(second.definitionId, first.definitionId);
  assert.equal(second.topologyRevision, first.topologyRevision);
  assert.equal(second.weightRevision, 2);
  assert.notEqual(second.weightRevisionId, first.weightRevisionId);
  assert.match(second.weightRevisionId, new RegExp(`^${first.definitionId}:weight:2:`));
  firstData.fill(90);
  secondSource.weights.weight.data.fill(80);
  assert.deepEqual([...first.copyWeightData('weight')], [1, 2]);
  assert.deepEqual([...second.copyWeightData('weight')], [7, 11]);

  const changed = structuralSource(identityDocument({
    dimensions: { B: { min: 1, max: 5 } },
    shape: ['B', 3],
  }), [{
    name: 'weight', dtype: 'float32', shape: [2], values: [13, 17],
  }]);
  const third = Model.derive(changed, second);
  assert.notEqual(third.definitionFingerprint, second.definitionFingerprint);
  assert.notEqual(third.definitionId, second.definitionId);
  assert.equal(third.topologyRevision, second.topologyRevision + 1);
  assert.equal(third.weightRevision, second.weightRevision + 1);
  assert.notEqual(third.weightRevisionId, second.weightRevisionId);
  assert.equal(second.matchesDefinition(secondSource.graph), true);
  assert.equal(second.matchesDefinition(changed.graph), false);
});

test('hydrated quantization is captured defensively and participates in binding', () => {
  const quantization = {
    q: { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
  };
  const source = structuralSource(quantizedDocument(), [{
    name: 'scale',
    dtype: 'float32',
    shape: [1],
    values: [0.25],
    role: 'quantization_parameter',
  }, {
    name: 'zero',
    dtype: 'int8',
    shape: [1],
    values: [-3],
    role: 'quantization_parameter',
  }], quantization);
  const snapshot = Model.capture(source);

  quantization.q.scale = 0.5;
  quantization.q.zero_point = 7;
  assert.deepEqual(snapshot.quantizationByTensor.q, {
    scheme: 'per_tensor', scale: 0.25, zero_point: -3,
  });
  const plan = snapshot.bindShapes({
    x: { data: new Float32Array(8), shape: [2, 4] },
  });
  assert.deepEqual(plan.tensors.q.quantization, {
    scheme: 'per_tensor', scale: 0.25, zero_point: -3,
  });
  assertDeepFrozen(snapshot.quantizationByTensor);
});

test('programmatic hydrated quantization rejects incomplete, extra, and invalid values at capture', () => {
  const source = structuralSource(quantizedDocument(), [{
    name: 'scale', dtype: 'float32', shape: [1], values: [0.25],
  }, {
    name: 'zero', dtype: 'int8', shape: [1], values: [-3],
  }], {
    q: { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
  });
  const invalidMetadata = [{
    q: { scheme: 'per_tensor', scale: 0.25 },
  }, {
    q: { scheme: 'per_tensor', scale: 0.25, zero_point: -3, legacy: true },
  }, {
    q: { scheme: 'per_tensor', scale: Number.NaN, zero_point: -3 },
  }, {
    q: { scheme: 'per_tensor', scale: Number.MIN_VALUE, zero_point: -3 },
  }, {
    q: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  }, {}, {
    q: { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
    undeclared: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  }];

  for (const quantizationByTensor of invalidMetadata) {
    assert.throws(
      () => Model.capture({ ...source, quantizationByTensor }),
      (error) => error instanceof ModelError && error.code === 'INVALID_SOURCE',
    );
  }
});

test('capture rejects incomplete weight sets, byte mismatches, and unsupported whole domains', () => {
  const weighted = structuralSource(identityDocument(), [{
    name: 'weight', dtype: 'float32', shape: [2], values: [1, 2],
  }]);
  assert.throws(
    () => Model.capture({ ...weighted, weights: {} }),
    (error) => error instanceof ModelError && error.code === 'WEIGHT_SET_MISMATCH',
  );
  const shortWeight = {
    ...weighted,
    weights: {
      weight: { ...weighted.weights.weight, data: Float32Array.of(1) },
    },
  };
  assert.throws(
    () => Model.capture(shortWeight),
    (error) => error instanceof ModelError && error.code === 'WEIGHT_PAYLOAD_MISMATCH',
  );

  const unsupported = structuralSource({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      K: { min: 1, max: 4 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'K'] } },
    nodes: [{
      id: 'projection',
      opType: 'MatMul',
      inputs: { input: 'x', weight: 'weight' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 3] } },
      params: {},
    }],
    outputs: ['y'],
  }, [{
    name: 'weight', dtype: 'float32', shape: [4, 3], values: new Float32Array(12),
  }]);
  assert.throws(
    () => Model.capture(unsupported),
    (error) => error instanceof ModelError &&
      error.code === 'SHAPE_DOMAIN_UNSUPPORTED' &&
      /UNPROVABLE_DYNAMIC_CONTRACTION/.test(error.message),
  );
});
