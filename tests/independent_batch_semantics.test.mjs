import test from 'node:test';
import assert from 'node:assert/strict';

import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model, ModelError } from '../ts/core/Model.js';
import {
  hasConservativeIndependentPublicBatchSemantics,
  inspectIndependentPublicBatchSemantics,
} from '../ts/ops/independentBatchSemantics.js';

function capture(document, definitions = [], weights = {}) {
  return Model.capture({
    graph: parseGraphDocument(document, definitions),
    weights,
  });
}

function graphDocument({
  shape,
  nodes,
  outputs = ['y'],
  maxBatch = 8,
  inputs = { x: { dtype: 'float32', shape } },
  dimensions = { B: { min: 1, max: maxBatch } },
}) {
  return {
    format: 'volvox-graph/v1',
    dimensions,
    inputs,
    nodes,
    outputs,
  };
}

function identityNode(shape, input = 'x') {
  return {
    id: 'identity',
    opType: 'Identity',
    inputs: { input },
    outputs: { out: { tensor: 'y', dtype: 'float32', shape } },
    params: {},
  };
}

function floatWeight(name, shape) {
  return Object.freeze({
    name,
    dtype: 'float32',
    shape: Object.freeze(shape),
    data: new Float32Array(shape.reduce((product, value) => product * value, 1)),
  });
}

function intWeight(name, values) {
  return Object.freeze({
    name,
    dtype: 'int32',
    shape: Object.freeze([values.length]),
    data: Int32Array.from(values),
  });
}

function assertRejected(snapshot, failedNode) {
  const evidence = inspectIndependentPublicBatchSemantics(snapshot);
  assert.equal(evidence.supported, false);
  assert.equal(evidence.failedNode, failedNode);
  assert.equal(evidence.graphFingerprint, snapshot.definitionFingerprint);
  assert.equal(hasConservativeIndependentPublicBatchSemantics(snapshot), false);
  return evidence;
}

test('typed proof follows a batch axis through permutation, reshape, squeeze, and reduction', () => {
  const snapshot = capture(graphDocument({
    shape: ['B', 2, 3],
    nodes: [
      {
        id: 'move_batch', opType: 'Transpose', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [2, 'B', 3] } },
        params: { perm: [1, 0, 2] },
      },
      {
        id: 'preserving_reshape', opType: 'Reshape', inputs: { input: 'a' },
        outputs: { out: { tensor: 'b', dtype: 'float32', shape: [2, 'B', 1, 3] } },
        params: { shape: [2, 'B', 1, 3] },
      },
      {
        id: 'remove_unit', opType: 'Squeeze', inputs: { input: 'b' },
        outputs: { out: { tensor: 'c', dtype: 'float32', shape: [2, 'B', 3] } },
        params: { axes: [2] },
      },
      {
        id: 'restore_batch', opType: 'Transpose', inputs: { input: 'c' },
        outputs: { out: { tensor: 'd', dtype: 'float32', shape: ['B', 2, 3] } },
        params: { perm: [1, 0, 2] },
      },
      {
        id: 'lane_softmax', opType: 'Softmax', inputs: { input: 'd' },
        outputs: { out: { tensor: 'e', dtype: 'float32', shape: ['B', 2, 3] } },
        params: { axis: 2 },
      },
      {
        id: 'lane_reduce', opType: 'ReduceSum', inputs: { input: 'e' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2, 1] } },
        params: { axis: 2, keepdims: true },
      },
    ],
  }));

  const evidence = inspectIndependentPublicBatchSemantics(snapshot);
  assert.deepEqual(evidence, {
    protocol: 'typed-independent-batch-proof/v1',
    graphFingerprint: snapshot.definitionFingerprint,
    supported: true,
    batchSymbol: 'B',
    coveredNodes: 6,
    reason: null,
    failedNode: null,
    failedTensor: null,
  });
  assert.equal(Object.isFrozen(evidence), true);
  assert.equal(hasConservativeIndependentPublicBatchSemantics(snapshot), true);
});

test('typed proof covers fixed-table gather and the structural VQA encoder pattern', () => {
  const table = floatWeight('table', [4, 3]);
  const other = floatWeight('other', [4, 3]);
  const snapshot = capture(graphDocument({
    shape: ['B'],
    inputs: { ids: { dtype: 'int32', shape: ['B'] } },
    outputs: ['y'],
    nodes: [
      {
        id: 'embedding', opType: 'Embedding', inputs: { input: 'ids', weight: 'table' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: ['B', 3] } }, params: {},
      },
      {
        id: 'gather', opType: 'Gather', inputs: { input: 'other', indices: 'ids' },
        outputs: { out: { tensor: 'b', dtype: 'float32', shape: ['B', 3] } },
        params: { axis: 0 },
      },
      {
        id: 'sum', opType: 'Add', inputs: { a: 'a', b: 'b' },
        outputs: { out: { tensor: 'c', dtype: 'float32', shape: ['B', 3] } }, params: {},
      },
      {
        id: 'softmax', opType: 'Softmax', inputs: { input: 'c' },
        outputs: { out: { tensor: 'd', dtype: 'float32', shape: ['B', 3] } },
        params: { axis: -1 },
      },
      {
        id: 'argmax', opType: 'ArgMax', inputs: { input: 'd' },
        outputs: { out: { tensor: 'y', dtype: 'int32', shape: ['B'] } },
        params: { axis: -1, keepdims: false, select_last_index: 0 },
      },
    ],
  }), [
    { name: 'table', dtype: 'float32', shape: [4, 3] },
    { name: 'other', dtype: 'float32', shape: [4, 3] },
  ], { table, other });

  const evidence = inspectIndependentPublicBatchSemantics(snapshot);
  assert.equal(evidence.supported, true);
  assert.equal(evidence.coveredNodes, 5);
});

test('reshape and Softmax cannot reinterpret or reduce the request axis', () => {
  const mixedReshape = capture(graphDocument({
    shape: ['B', 2],
    nodes: [
      {
        id: 'mix_lanes', opType: 'Reshape', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [2, 'B'] } },
        params: { shape: [2, 'B'] },
      },
      {
        id: 'restore', opType: 'Reshape', inputs: { input: 'a' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
        params: { shape: ['B', 2] },
      },
    ],
  }));
  assertRejected(mixedReshape, 'mix_lanes');

  const softmaxAcrossBatch = capture(graphDocument({
    shape: ['B', 2],
    nodes: [
      {
        id: 'move', opType: 'Transpose', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [2, 'B'] } },
        params: { perm: [1, 0] },
      },
      {
        id: 'mix_lanes', opType: 'Softmax', inputs: { input: 'a' },
        outputs: { out: { tensor: 'b', dtype: 'float32', shape: [2, 'B'] } },
        params: { axis: 1 },
      },
      {
        id: 'restore', opType: 'Transpose', inputs: { input: 'b' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
        params: { perm: [1, 0] },
      },
    ],
  }));
  assertRejected(softmaxAcrossBatch, 'mix_lanes');
});

test('Gather, Concat, and BatchMatMul reject request-axis indexing or matrix use', () => {
  const indices = intWeight('indices', [0]);
  const gather = capture(graphDocument({
    shape: ['B', 2],
    nodes: [
      {
        id: 'gather_batch', opType: 'Gather', inputs: { input: 'x', indices: 'indices' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: [1, 2] } },
        params: { axis: 0 },
      },
      identityNode(['B', 2]),
    ],
  }), [{ name: 'indices', dtype: 'int32', shape: [1] }], { indices });
  assertRejected(gather, 'gather_batch');

  const concat = capture(graphDocument({
    shape: ['B', 2],
    maxBatch: 1,
    nodes: [
      {
        id: 'concat_batch', opType: 'Concat', inputs: { input0: 'x', input1: 'x' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: [2, 2] } },
        params: { axis: 0 },
      },
      identityNode(['B', 2]),
    ],
  }));
  assertRejected(concat, 'concat_batch');

  const weight = floatWeight('w', [1, 3, 4]);
  const matrixBatch = capture(graphDocument({
    shape: ['B', 3],
    nodes: [
      {
        id: 'insert_prefix', opType: 'Unsqueeze', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [1, 'B', 3] } },
        params: { axes: [0] },
      },
      {
        id: 'matrix_batch', opType: 'BatchMatMul', inputs: { a: 'a', b: 'w' },
        outputs: { out: { tensor: 'b', dtype: 'float32', shape: [1, 'B', 4] } },
        params: {},
      },
      {
        id: 'remove_prefix', opType: 'Squeeze', inputs: { input: 'b' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 4] } },
        params: { axes: [0] },
      },
    ],
  }), [{ name: 'w', dtype: 'float32', shape: [1, 3, 4] }], { w: weight });
  assertRejected(matrixBatch, 'matrix_batch');
});

test('canonical graph proof rejects axis-0 reduction/slice and reverse slice before batching proof', () => {
  const unsafeDocuments = [
    graphDocument({
      shape: ['B', 2],
      nodes: [{
        id: 'reduce_batch', opType: 'ReduceSum', inputs: { input: 'x' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: [2] } },
        params: { axis: 0, keepdims: false },
      }, identityNode(['B', 2])],
    }),
    graphDocument({
      shape: ['B', 2],
      nodes: [{
        id: 'slice_batch', opType: 'Slice', inputs: { input: 'x' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: [1, 2] } },
        params: { starts: [0], ends: [1], axes: [0], steps: [1] },
      }, identityNode(['B', 2])],
    }),
    graphDocument({
      shape: ['B', 3],
      nodes: [{
        id: 'reverse', opType: 'Slice', inputs: { input: 'x' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 3] } },
        params: { starts: [2], ends: [-4], axes: [1], steps: [-1] },
      }],
    }),
  ];

  for (const document of unsafeDocuments) {
    assert.throws(() => capture(document), (error) =>
      error instanceof ModelError && error.code === 'SHAPE_DOMAIN_UNSUPPORTED');
  }
});

test('per-axis quantization cannot follow a transposed request axis', () => {
  const snapshot = capture(graphDocument({
    shape: ['B', 2],
    nodes: [
      {
        id: 'move', opType: 'Transpose', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [2, 'B'] } },
        params: { perm: [1, 0] },
      },
      {
        id: 'restore', opType: 'Transpose', inputs: { input: 'a' },
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', 2] } },
        params: { perm: [1, 0] },
      },
    ],
  }));

  // Canonical model validation already prevents a dynamic per-axis table from
  // targeting B. Forge only the immutable descriptor view to regression-test
  // this proof's own axis-provenance guard after B moved from axis 0 to axis 1.
  const moved = snapshot.graph.tensors.a;
  const hostile = {
    definitionFingerprint: snapshot.definitionFingerprint,
    inputDescriptors: snapshot.inputDescriptors,
    outputDescriptors: snapshot.outputDescriptors,
    graph: {
      ...snapshot.graph,
      tensors: {
        ...snapshot.graph.tensors,
        a: Object.freeze({
          ...moved,
          quantization: Object.freeze({
            scheme: 'per_axis', axis: 1,
            scale_tensor: 'scale', zero_point_tensor: 'zero_point',
          }),
        }),
      },
    },
  };
  const evidence = inspectIndependentPublicBatchSemantics(hostile);
  assert.equal(evidence.supported, false);
  assert.equal(evidence.failedTensor, 'a');
  assert.match(evidence.reason, /quantization.*batch axis/i);
});

test('VQA encoder B=1 folds fail closed at the exact unsafe Reshape', () => {
  const fp32 = capture(graphDocument({
    shape: ['B', 'M', 320],
    maxBatch: 1,
    dimensions: {
      B: { min: 1, max: 1 },
      M: { min: 211, max: 402 },
    },
    nodes: [
      {
        id: 'node_149', opType: 'Reshape', inputs: { input: 'x' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: ['M', 'B', 320] } },
        params: { shape: ['M', 'B', 320] },
      },
      identityNode(['B', 'M', 320]),
    ],
  }));
  const fp32Evidence = assertRejected(fp32, 'node_149');
  assert.match(fp32Evidence.reason, /row-major products/);

  const int8 = capture(graphDocument({
    shape: ['B', 8, 1, 'M'],
    maxBatch: 1,
    dimensions: {
      B: { min: 1, max: 1 },
      M: { min: 211, max: 402 },
    },
    nodes: [
      {
        id: 'node_44', opType: 'Reshape', inputs: { input: 'x' },
        outputs: { out: { tensor: 'dead', dtype: 'float32', shape: [1, 8, 1, 'M'] } },
        params: { shape: [1, 8, 1, 'M'] },
      },
      identityNode(['B', 8, 1, 'M']),
    ],
  }));
  const int8Evidence = assertRejected(int8, 'node_44');
  assert.match(int8Evidence.reason, /row-major products/);
});
