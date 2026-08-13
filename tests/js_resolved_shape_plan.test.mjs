import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ResolvedShapePlanError,
  proveGraphShapeDomain,
  resolveGraphShapes,
  resolveMinimumGraphShapes,
  resolveStaticGraphShapes,
} from '../ts/core/ResolvedShapePlan.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

function assertPlanError(operation, code, pattern = undefined) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof ResolvedShapePlanError, `expected ResolvedShapePlanError, got ${String(error)}`);
  assert.equal(error.code, code);
  assert.equal(typeof error.path, 'string');
  assert.notEqual(error.path.length, 0);
  if (pattern !== undefined) assert.match(error.message, pattern);
  return error;
}

function assertDeepFrozen(value, seen = new Set()) {
  if (value == null || typeof value !== 'object' || seen.has(value)) return;
  seen.add(value);
  assert.equal(Object.isFrozen(value), true, `expected frozen value: ${Object.prototype.toString.call(value)}`);
  for (const key of Reflect.ownKeys(value)) assertDeepFrozen(value[key], seen);
}

function reachesReference(root, target, seen = new Set()) {
  if (root === target) return true;
  if (root == null || typeof root !== 'object' || seen.has(root)) return false;
  seen.add(root);
  return Reflect.ownKeys(root).some((key) => reachesReference(root[key], target, seen));
}

function constantIdentityDocument(overrides = {}) {
  return {
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
    ...overrides,
  };
}

function symbolicChainDocument(overrides = {}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      H: { min: 2, max: 8, multiple_of: 2 },
      O: { min: 1, max: 8 },
      S: { min: 1, max: 16 },
    },
    inputs: { ids: { dtype: 'int32', shape: ['B', 'S'] } },
    nodes: [{
      id: 'embedding',
      opType: 'Embedding',
      inputs: { input: 'ids', weight: 'embedding.weight' },
      outputs: {
        out: { tensor: 'hidden', dtype: 'float32', shape: ['B', 'S', 'H'] },
      },
      params: {},
    }, {
      id: 'normalization',
      opType: 'LayerNorm',
      inputs: {
        input: 'hidden',
        weight: 'norm.weight',
        bias: 'norm.bias',
      },
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
        out: { tensor: 'projected', dtype: 'float32', shape: ['B', 'S', 'O'] },
      },
      params: {},
    }, {
      id: 'activation',
      opType: 'ReLU',
      inputs: { input: 'projected' },
      outputs: {
        out: { tensor: 'scores', dtype: 'float32', shape: ['B', 'S', 'O'] },
      },
      params: {},
    }],
    outputs: ['scores'],
    ...overrides,
  };
}

function symbolicChainWeights() {
  return [
    { name: 'embedding.weight', dtype: 'float32', shape: [32, 4] },
    { name: 'norm.weight', dtype: 'float32', shape: [4] },
    { name: 'norm.bias', dtype: 'float32', shape: [4] },
    { name: 'projection.weight', dtype: 'float32', shape: [3, 4] },
    { name: 'projection.bias', dtype: 'float32', shape: [3] },
  ];
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
      outputs: {
        out: { tensor: 'q', dtype: 'int8', shape: ['B', 4] },
      },
      params: {},
    }, {
      id: 'dequantize',
      opType: 'DequantizeLinear',
      inputs: { input: 'q', scale: 'scale', zero_point: 'zero' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: ['B', 4] },
      },
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

const quantizedWeights = Object.freeze([
  Object.freeze({ name: 'scale', dtype: 'float32', shape: Object.freeze([1]) }),
  Object.freeze({ name: 'zero', dtype: 'int8', shape: Object.freeze([1]) }),
]);

const perTensorI8 = Object.freeze({
  scheme: 'per_tensor',
  scale: 0.25,
  zero_point: 0,
});

test('constant graph produces a deep-frozen descriptor-only plan with checked totals', () => {
  const graph = parseGraphDocument(constantIdentityDocument());
  const callerShape = [2, 3];
  const callerData = new Float32Array([1, 2, 3, 4, 5, 6]);
  const callerView = { data: callerData, shape: callerShape };
  const inputs = { x: callerView };
  const beforeData = [...callerData];
  const beforeShape = [...callerShape];

  const plan = resolveGraphShapes(graph, inputs);

  assert.equal(plan.signature, 'v1|1:x|2:2,3');
  assert.equal(plan.graphFingerprint, graph.fingerprint);
  assert.deepEqual(plan.symbols, {});
  assert.deepEqual(plan.tensors.x, {
    name: 'x',
    kind: 'input',
    dtype: 'float32',
    shape: [2, 3],
    elementCount: 6,
    sizeBytes: 24,
  });
  assert.deepEqual(plan.outputs.map((output) => output.name), ['y']);
  assert.equal(plan.logicalActivationBytes, 48);
  assert.equal(plan.weightBytes, 0);
  assert.deepEqual(callerShape, beforeShape);
  assert.deepEqual([...callerData], beforeData);
  assert.equal(reachesReference(plan, callerView), false);
  assert.equal(reachesReference(plan, callerShape), false);
  assert.equal(reachesReference(plan, callerData), false);
  assert.equal('data' in plan.tensors.x, false);
  assert.equal('buffer' in plan.tensors.x, false);
  assertDeepFrozen(plan);

  const serialized = JSON.stringify(plan);
  callerShape[0] = 99;
  callerData.fill(99);
  assert.deepEqual(plan.tensors.x.shape, [2, 3]);
  assert.equal(plan.tensors.x.sizeBytes, 24);
  assert.equal(JSON.stringify(plan), serialized);
  assert.throws(() => {
    plan.tensors.x.shape[0] = 9;
  }, TypeError);
});

test('static resolver prebinds metadata without request storage and rejects a dynamic public domain', () => {
  const largeStaticGraph = parseGraphDocument(constantIdentityDocument({
    inputs: { x: { dtype: 'float32', shape: [100_000_000] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: [100_000_000] },
      },
      params: {},
    }],
  }));

  const plan = resolveStaticGraphShapes(largeStaticGraph);
  assert.equal(plan.signature, 'v1|1:x|1:100000000');
  assert.equal(plan.tensors.x.sizeBytes, 400_000_000);
  assert.equal(plan.logicalActivationBytes, 800_000_000);
  assert.equal('data' in plan.tensors.x, false);
  assertDeepFrozen(plan);

  const dynamicGraph = parseGraphDocument(symbolicChainDocument(), symbolicChainWeights());
  assertPlanError(
    () => resolveStaticGraphShapes(dynamicGraph),
    'INPUT_BINDING_FAILED',
    /more than one legal value/,
  );
});

test('minimum resolver builds the canonical smallest legal metadata-only plan', () => {
  const graph = parseGraphDocument(symbolicChainDocument(), symbolicChainWeights());
  const minimum = resolveMinimumGraphShapes(graph);
  const explicit = resolveGraphShapes(graph, {
    ids: { data: new Int32Array(1), shape: [1, 1] },
  });

  assert.equal(minimum.signature, explicit.signature);
  assert.deepEqual(minimum.symbols, explicit.symbols);
  assert.deepEqual(minimum.tensors, explicit.tensors);
  assert.equal(minimum.logicalActivationBytes, explicit.logicalActivationBytes);
  assert.equal('data' in minimum.tensors.ids, false);
  assertDeepFrozen(minimum);

  const alignedGraph = parseGraphDocument(constantIdentityDocument({
    dimensions: { S: { min: 3, max: 8, multiple_of: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['S'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['S'] } },
      params: {},
    }],
  }));
  const aligned = resolveMinimumGraphShapes(alignedGraph);
  assert.deepEqual(aligned.symbols, { S: 4 });
  assert.deepEqual(aligned.tensors.x.shape, [4]);
});

test('symbolic embedding-normalization-dense chain binds output-only symbols and all bytes', () => {
  const graph = parseGraphDocument(symbolicChainDocument(), symbolicChainWeights());
  const plan = resolveGraphShapes(graph, {
    ids: { data: new Int32Array(2 * 5), shape: [2, 5] },
  });

  assert.equal(plan.signature, 'v1|3:ids|2:2,5');
  assert.deepEqual(plan.symbols, { B: 2, H: 4, O: 3, S: 5 });
  assert.deepEqual(plan.tensors.hidden.shape, [2, 5, 4]);
  assert.deepEqual(plan.tensors.normalized.shape, [2, 5, 4]);
  assert.deepEqual(plan.tensors.projected.shape, [2, 5, 3]);
  assert.deepEqual(plan.outputs[0].shape, [2, 5, 3]);
  assert.equal(plan.logicalActivationBytes, 600);
  assert.equal(plan.weightBytes, 604);
  assert.deepEqual(
    plan.nodes.map((node) => node.shapeFunctionId),
    [
      'volvox.shape.embedding-prefix.v1',
      'volvox.shape.feature-norm.v1',
      'volvox.shape.dense-last-axis.v1',
      'volvox.shape.activation-preserve.v1',
    ],
  );
  assert.strictEqual(plan.nodes[1].inputs.input, plan.tensors.hidden);
  assert.strictEqual(plan.nodes[2].outputs.out, plan.tensors.projected);
  assertDeepFrozen(plan);
});

test('one immutable logical graph resolves two exact public signatures without retaining either view', () => {
  const graph = parseGraphDocument(symbolicChainDocument(), symbolicChainWeights());
  const fingerprint = graph.fingerprint;
  const firstShape = [1, 3];
  const secondShape = [4, 2];
  const firstData = new Int32Array(3);
  const secondData = new Int32Array(8);
  const first = resolveGraphShapes(graph, {
    ids: { data: firstData, shape: firstShape },
  });
  const second = resolveGraphShapes(graph, {
    ids: { data: secondData, shape: secondShape },
  });

  assert.notEqual(first.signature, second.signature);
  assert.deepEqual(first.outputs[0].shape, [1, 3, 3]);
  assert.deepEqual(second.outputs[0].shape, [4, 2, 3]);
  assert.deepEqual(first.symbols, { B: 1, H: 4, O: 3, S: 3 });
  assert.deepEqual(second.symbols, { B: 4, H: 4, O: 3, S: 2 });
  assert.equal(graph.fingerprint, fingerprint);
  assert.equal(Object.isFrozen(graph), true);
  assert.equal(reachesReference(first, firstData), false);
  assert.equal(reachesReference(second, secondData), false);
});

test('tensor records use unsigned UTF-8 name order while the signature remains public-input-only', () => {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: {
      'é': { dtype: 'float32', shape: [1] },
      a: { dtype: 'float32', shape: [1] },
      z: { dtype: 'float32', shape: [1] },
    },
    nodes: [],
    outputs: ['a'],
  });
  const plan = resolveGraphShapes(graph, {
    z: { data: new Float32Array(1), shape: [1] },
    'é': { data: new Float32Array(1), shape: [1] },
    a: { data: new Float32Array(1), shape: [1] },
  });

  assert.deepEqual(Object.keys(plan.tensors), ['a', 'z', 'é']);
  assert.equal(plan.signature, 'v1|1:a|1:1|1:z|1:1|2:é|1:1');
  assert.equal(plan.signature.includes(graph.fingerprint), false);
});

test('output assertions reject conflicting output-only bindings atomically', () => {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      O: { min: 1, max: 4 },
      T: { min: 1, max: 4 },
    },
    inputs: {
      x: { dtype: 'float32', shape: ['B', 4] },
      z: { dtype: 'float32', shape: ['T', 4] },
    },
    nodes: [{
      id: 'first',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['O', 4] } },
      params: {},
    }, {
      id: 'second',
      opType: 'Identity',
      inputs: { input: 'z' },
      outputs: { out: { tensor: 'q', dtype: 'float32', shape: ['O', 4] } },
      params: {},
    }],
    outputs: ['q'],
  });
  const inputs = {
    x: { data: new Float32Array(2 * 4), shape: [2, 4] },
    z: { data: new Float32Array(3 * 4), shape: [3, 4] },
  };
  const before = JSON.stringify(inputs, (_key, value) => ArrayBuffer.isView(value) ? [...value] : value);

  const error = assertPlanError(
    () => resolveGraphShapes(graph, inputs),
    'SYMBOL_CONFLICT',
    /already bound to 2/,
  );
  assert.equal(error.path, 'nodes[1].outputs.out.shape[0]');
  assert.equal(
    JSON.stringify(inputs, (_key, value) => ArrayBuffer.isView(value) ? [...value] : value),
    before,
  );
});

test('ordinary node input rejects a logical symbol that no earlier binding defines', () => {
  const base = parseGraphDocument({
    ...constantIdentityDocument(),
    nodes: [{
      id: 'first',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2, 3] } },
      params: {},
    }, {
      id: 'second',
      opType: 'Identity',
      inputs: { input: 'y' },
      outputs: { out: { tensor: 'z', dtype: 'float32', shape: [2, 3] } },
      params: {},
    }],
    outputs: ['z'],
  });
  const environment = new ShapeEnvironment([{ name: 'U', min: 1, max: 8 }]);
  const poisonedY = Object.freeze({
    ...base.tensors.y,
    shape: Object.freeze(['U', 3]),
  });
  const poisoned = Object.freeze({
    ...base,
    environment,
    dimensions: Object.freeze({
      U: Object.freeze({ name: 'U', min: 1, max: 8, multiple_of: 1 }),
    }),
    tensors: Object.freeze({ ...base.tensors, y: poisonedY }),
  });

  const error = assertPlanError(
    () => resolveGraphShapes(poisoned, {
      x: { data: new Float32Array(6), shape: [2, 3] },
    }),
    'UNBOUND_SYMBOL',
    /before an input or earlier output binds it/,
  );
  assert.equal(error.path, 'nodes[1].inputs.input.shape[0]');
});

test('dtype, same-byte wrong-shape, byte-length, and safe-integer failures remain distinct', () => {
  const graph = parseGraphDocument(constantIdentityDocument());
  assertPlanError(
    () => resolveGraphShapes(graph, {
      x: { data: new Int32Array(6), shape: [2, 3] },
    }),
    'INPUT_BINDING_FAILED',
    /dtype 'int32'.*requires 'float32'/,
  );
  assertPlanError(
    () => resolveGraphShapes(graph, {
      x: { data: new Float32Array(6), shape: [3, 2] },
    }),
    'INPUT_BINDING_FAILED',
    /requires constant 2/,
  );
  assertPlanError(
    () => resolveGraphShapes(graph, {
      x: { data: new Float32Array(5), shape: [2, 3] },
    }),
    'INPUT_BINDING_FAILED',
    /20 bytes.*require 24/,
  );

  const wrongDType = parseGraphDocument(constantIdentityDocument({
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'int32', shape: [2, 3] } },
      params: {},
    }],
  }));
  assertPlanError(
    () => resolveGraphShapes(wrongDType, {
      x: { data: new Float32Array(6), shape: [2, 3] },
    }),
    'OUTPUT_DTYPE_MISMATCH',
    /inferred 'float32'.*asserts 'int32'/,
  );

  const overflow = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { N: { min: 1, max: Number.MAX_SAFE_INTEGER } },
    inputs: { x: { dtype: 'float32', shape: ['N'] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['N'] } },
      params: {},
    }],
    outputs: ['y'],
  });
  assertPlanError(
    () => resolveGraphShapes(overflow, {
      x: { data: new Float32Array(0), shape: [Number.MAX_SAFE_INTEGER] },
    }),
    'ARITHMETIC_OVERFLOW',
    /safe integer range/,
  );
});

test('Q/DQ resolution hydrates affine metadata before inference and enforces agreement', () => {
  const graph = parseGraphDocument(quantizedDocument(), quantizedWeights);
  const metadata = { q: perTensorI8 };
  const plan = resolveGraphShapes(graph, {
    x: { data: new Float32Array(3 * 4), shape: [3, 4] },
  }, metadata);

  assert.deepEqual(plan.tensors.q, {
    name: 'q',
    kind: 'value',
    dtype: 'int8',
    shape: [3, 4],
    elementCount: 12,
    sizeBytes: 12,
    quantization: perTensorI8,
    producerNodeId: 'quantize',
    producerPort: 'out',
  });
  assert.equal(plan.tensors.y.quantization, undefined);
  assert.equal(plan.nodes[0].outputs.out.quantization.scheme, 'per_tensor');
  assert.notStrictEqual(plan.tensors.q.quantization, metadata.q);
  assert.equal(plan.weightBytes, 5);
  assert.equal(reachesReference(plan, metadata.q), false);

  assertPlanError(
    () => resolveGraphShapes(graph, {
      x: { data: new Float32Array(4), shape: [1, 4] },
    }),
    'QUANTIZATION_METADATA_MISSING',
    /missing target metadata for \[q\]/,
  );
  assertPlanError(
    () => resolveGraphShapes(graph, {
      x: { data: new Float32Array(4), shape: [1, 4] },
    }, {
      q: {
        scheme: 'per_axis',
        axis: 1,
        scales: [0.25, 0.25, 0.25, 0.25],
        zero_points: [0, 0, 0, 0],
      },
    }),
    'QUANTIZATION_MISMATCH',
    /logical reference requires 'per_tensor'/,
  );
});

test('whole-domain proof accepts the symbolic Wave-A chain and output-only aliases', () => {
  const graph = parseGraphDocument(symbolicChainDocument(), symbolicChainWeights());
  const fingerprint = graph.fingerprint;
  const proof = proveGraphShapeDomain(graph);

  assert.equal(proof.supported, true);
  assert.equal(proof.graphFingerprint, fingerprint);
  assert.deepEqual(proof.symbolRelations, { B: 'B', H: 4, O: 3, S: 'S' });
  assert.deepEqual(proof.outputs[0].shape, ['B', 'S', 3]);
  assert.equal(proof.nodes.length, 4);
  assert.match(proof.nodes[2].facts.join(' '), /contracted extent 4/);
  assert.equal(graph.fingerprint, fingerprint);
  assertDeepFrozen(proof);
});

test('whole-domain proof supports per-tensor Q/DQ and is pure over hydrated metadata', () => {
  const graph = parseGraphDocument(quantizedDocument(), quantizedWeights);
  const metadata = { q: { scheme: 'per_tensor', scale: 0.1, zero_point: 0 } };
  const before = structuredClone(metadata);
  const proof = proveGraphShapeDomain(graph, metadata);

  assert.equal(proof.supported, true);
  assert.deepEqual(proof.outputs[0].shape, ['B', 4]);
  assert.equal(proof.nodes[0].outputs.out.quantization.scale, Math.fround(0.1));
  assert.deepEqual(metadata, before);
  assert.notStrictEqual(proof.nodes[0].outputs.out.quantization, metadata.q);
});

test('Resize and ResizeNearest2D receive their authored declared output descriptors', () => {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      H: { min: 2, max: 16 },
      W: { min: 2, max: 16 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'H', 'W', 3] } },
    nodes: [{
      id: 'linear-resize',
      opType: 'Resize',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'mid', dtype: 'float32', shape: ['B', 5, 7, 3] },
      },
      params: { mode: 'linear' },
    }, {
      id: 'nearest-resize',
      opType: 'ResizeNearest2D',
      inputs: { input: 'mid' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: ['B', 9, 11, 3] },
      },
      params: {},
    }],
    outputs: ['y'],
  });

  const proof = proveGraphShapeDomain(graph);
  assert.equal(proof.supported, true);
  assert.deepEqual(proof.nodes[0].outputs.out.shape, ['B', 5, 7, 3]);
  assert.deepEqual(proof.nodes[1].outputs.out.shape, ['B', 9, 11, 3]);

  const plan = resolveGraphShapes(graph, {
    x: { data: new Float32Array(2 * 4 * 6 * 3), shape: [2, 4, 6, 3] },
  });
  assert.deepEqual(plan.tensors.mid.shape, [2, 5, 7, 3]);
  assert.deepEqual(plan.tensors.y.shape, [2, 9, 11, 3]);
});

test('whole-domain proof fails closed for registry gaps and unprovable contractions', () => {
  const unknown = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 4] } },
    nodes: [{
      id: 'nms',
      opType: 'NonMaxSuppression',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 4] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  const unknownProof = proveGraphShapeDomain(unknown);
  assert.equal(unknownProof.supported, false);
  assert.equal(unknownProof.code, 'UNKNOWN_OPERATOR');
  assert.equal(unknownProof.path, 'nodes[0].opType');

  const dynamicContraction = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 4 },
      K: { min: 2, max: 3 },
    },
    inputs: { x: { dtype: 'float32', shape: ['B', 'K'] } },
    nodes: [{
      id: 'matmul',
      opType: 'MatMul',
      inputs: { input: 'x', weight: 'weight' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 4] } },
      params: {},
    }],
    outputs: ['y'],
  }, [{ name: 'weight', dtype: 'float32', shape: [3, 4] }]);
  const contractionProof = proveGraphShapeDomain(dynamicContraction);
  assert.equal(contractionProof.supported, false);
  assert.equal(contractionProof.code, 'OPERATOR_DOMAIN_UNSUPPORTED');
  assert.equal(contractionProof.path, 'nodes[0]');
  assert.match(contractionProof.reason, /UNPROVABLE_DYNAMIC_CONTRACTION/);
});

test('whole-domain proof rejects an output-only constraint that cannot contain inferred output', () => {
  const document = symbolicChainDocument();
  document.dimensions.O = { min: 1, max: 2 };
  const graph = parseGraphDocument(document, symbolicChainWeights());
  const proof = proveGraphShapeDomain(graph);

  assert.equal(proof.supported, false);
  assert.equal(proof.code, 'BOUND_VIOLATION');
  assert.equal(proof.path, 'nodes[2].outputs.out.shape[2]');
  assert.match(proof.reason, /constant 3.*outside 'O' bounds/);
});
