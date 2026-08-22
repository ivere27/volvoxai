import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ModelBuilder,
  ModelBuilderError,
} from '../ts/core/ModelBuilder.js';
import { GraphError } from '../ts/core/Graph.js';

function assertDeepFrozen(value, seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) return;
  seen.add(value);
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) assertDeepFrozen(child, seen);
}

function constantBuilder() {
  return new ModelBuilder({
    dimensions: {},
    inputs: {
      x: { dtype: 'float32', shape: [1, 4] },
    },
    weights: [{ name: 'fixed.weight', dtype: 'float32', shape: [4, 4] }],
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: [1, 4] },
      },
    }],
    outputs: ['y'],
  });
}

test('builder publishes immutable constant logical and decoded-document snapshots', () => {
  const inputShape = [1, 4];
  const weightShape = [4, 4];
  const builder = new ModelBuilder({
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: inputShape } },
    weights: [{ name: 'fixed.weight', dtype: 'float32', shape: weightShape }],
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 4] } },
    }],
    outputs: ['y'],
  });

  inputShape[0] = 9;
  weightShape[0] = 9;

  const logical = builder.snapshot();
  const document = builder.documentSnapshot();
  const weights = builder.weightDescriptorsSnapshot();
  assert.equal(document.format, 'volvox-graph/v1');
  assert.equal('shape_system' in document, false);
  assert.deepEqual(document.dimensions, {});
  assert.deepEqual(document.inputs.x, { dtype: 'float32', shape: [1, 4] });
  assert.deepEqual(document.nodes[0].outputs.out, {
    tensor: 'y', dtype: 'float32', shape: [1, 4],
  });
  assert.deepEqual(document.outputs, ['y']);
  assert.equal('weights' in document, false);
  assert.deepEqual(weights, [{ name: 'fixed.weight', dtype: 'float32', shape: [4, 4] }]);
  assert.deepEqual(logical.inputs.x.shape, [1, 4]);
  assert.deepEqual(logical.weights['fixed.weight'].shape, [4, 4]);
  assert.equal('buffer' in logical.inputs.x, false);
  assert.equal('data' in logical.weights['fixed.weight'], false);
  assert.match(builder.fingerprint, /^volvox-logical-graph\/v1\|/);

  assertDeepFrozen(logical);
  assertDeepFrozen(document);
  assertDeepFrozen(weights);
});

test('symbolic construction supports fixed weights and generates only omitted IDs deterministically', () => {
  const builder = new ModelBuilder({
    dimensions: {
      S: { min: 2, max: 16, multiple_of: 2 },
      B: { min: 1, max: 4 },
    },
    inputs: { ids: { dtype: 'int32', shape: ['B', 'S'] } },
    weights: [{ name: 'token.weight', dtype: 'float32', shape: [64, 8] }],
    outputs: ['ids'],
  });

  const prior = builder.snapshot();
  const generated = builder.topologyTransaction((edit) => {
    assert.strictEqual(edit.snapshot(), prior);
    const id = edit.addNode({
      opType: 'Embedding',
      inputs: { input: 'ids', weight: 'token.weight' },
      outputs: {
        out: { tensor: 'hidden', dtype: 'float32', shape: ['B', 'S', 8] },
      },
      params: {},
    });
    edit.selectOutputs(['hidden']);
    return id;
  });
  assert.equal(generated, 'node_0');
  assert.deepEqual(Object.keys(builder.snapshot().dimensions), ['B', 'S']);
  assert.deepEqual(builder.snapshot().nodes[0], {
    id: 'node_0',
    opType: 'Embedding',
    inputs: { input: 'ids', weight: 'token.weight' },
    outputs: {
      out: { tensor: 'hidden', dtype: 'float32', shape: ['B', 'S', 8] },
    },
    params: {},
  });

  const explicit = builder.addNode({
    id: 'caller_identity',
    opType: 'Identity',
    inputs: { input: 'hidden' },
    outputs: {
      out: { tensor: 'echo', dtype: 'float32', shape: ['B', 'S', 8] },
    },
  });
  assert.equal(explicit, 'caller_identity');
  assert.equal(builder.snapshot().nodes[1].id, 'caller_identity');
});

test('failed generated-node edits preserve exact snapshots and do not consume IDs', () => {
  const builder = new ModelBuilder({
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2] } },
    outputs: ['x'],
  });
  const priorGraph = builder.snapshot();
  const priorDocument = builder.documentSnapshot();
  const priorWeights = builder.weightDescriptorsSnapshot();
  const priorFingerprint = builder.fingerprint;
  let rejectedId;

  assert.throws(() => builder.topologyTransaction((edit) => {
    rejectedId = edit.addNode({
      opType: 'Identity',
      inputs: { input: 'missing' },
      outputs: { out: { tensor: 'bad', dtype: 'float32', shape: [2] } },
    });
    edit.selectOutputs(['bad']);
  }), GraphError);

  assert.equal(rejectedId, 'node_0');
  assert.strictEqual(builder.snapshot(), priorGraph);
  assert.strictEqual(builder.documentSnapshot(), priorDocument);
  assert.strictEqual(builder.weightDescriptorsSnapshot(), priorWeights);
  assert.equal(builder.fingerprint, priorFingerprint);

  const acceptedId = builder.topologyTransaction((edit) => {
    const id = edit.addNode({
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } },
    });
    edit.selectOutputs(['y']);
    return id;
  });
  assert.equal(acceptedId, 'node_0');
  assert.notEqual(builder.fingerprint, priorFingerprint);
});

test('grouped topology edits may repair temporary dangling references atomically', () => {
  const builder = new ModelBuilder({
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [3] } },
    outputs: ['x'],
  });
  const priorGraph = builder.snapshot();
  const priorDocument = builder.documentSnapshot();

  assert.throws(() => builder.removeInput('x'), GraphError);
  assert.strictEqual(builder.snapshot(), priorGraph);
  assert.strictEqual(builder.documentSnapshot(), priorDocument);

  builder.topologyTransaction((edit) => {
    assert.equal(edit.removeInput('x'), true);
    edit.setInput('z', { dtype: 'int32', shape: [3] });
    edit.selectOutputs(['z']);
    assert.strictEqual(edit.snapshot(), priorGraph);
  });
  assert.deepEqual(builder.documentSnapshot().inputs, {
    z: { dtype: 'int32', shape: [3] },
  });
  assert.deepEqual(builder.documentSnapshot().outputs, ['z']);

  const committedGraph = builder.snapshot();
  const committedDocument = builder.documentSnapshot();
  assert.throws(() => builder.topologyTransaction(() => {
    throw new Error('caller abort');
  }), /caller abort/);
  assert.strictEqual(builder.snapshot(), committedGraph);
  assert.strictEqual(builder.documentSnapshot(), committedDocument);
});

test('central affine quantization and fixed parameter descriptors commit together', () => {
  const builder = new ModelBuilder({
    dimensions: { B: { min: 1, max: 8 } },
    inputs: { q: { dtype: 'uint8', shape: ['B', 4] } },
    weights: [
      { name: 'q.scale', dtype: 'float32', shape: [1] },
      { name: 'q.zero', dtype: 'uint8', shape: [1] },
    ],
    outputs: ['q'],
  });
  builder.setQuantization({
    format: 'volvox-affine-safetensors/v1',
    tensors: {
      q: {
        scheme: 'per_tensor',
        scale_tensor: 'q.scale',
        zero_point_tensor: 'q.zero',
      },
    },
  });

  assert.deepEqual(builder.snapshot().inputs.q.quantization, {
    scheme: 'per_tensor', scale_tensor: 'q.scale', zero_point_tensor: 'q.zero',
  });
  assert.deepEqual(builder.documentSnapshot().quantization, {
    format: 'volvox-affine-safetensors/v1',
    tensors: {
      q: {
        scheme: 'per_tensor', scale_tensor: 'q.scale', zero_point_tensor: 'q.zero',
      },
    },
  });
  assert.equal('quantization' in builder.documentSnapshot().inputs.q, false);

  const priorGraph = builder.snapshot();
  const priorDocument = builder.documentSnapshot();
  const priorWeights = builder.weightDescriptorsSnapshot();
  assert.throws(() => builder.removeWeight('q.scale'), GraphError);
  assert.strictEqual(builder.snapshot(), priorGraph);
  assert.strictEqual(builder.documentSnapshot(), priorDocument);
  assert.strictEqual(builder.weightDescriptorsSnapshot(), priorWeights);
});

test('fromDocument enforces the closed graph schema and rejects async or nested transactions', () => {
  const document = constantBuilder().documentSnapshot();
  const adopted = ModelBuilder.fromDocument(document, [
    { name: 'fixed.weight', dtype: 'float32', shape: [4, 4] },
  ]);
  assert.deepEqual(adopted.documentSnapshot(), document);

  const extended = JSON.parse(JSON.stringify(document));
  extended.shape_system = 'volvox-bounded-shape/v1';
  assert.throws(
    () => ModelBuilder.fromDocument(extended),
    (error) => error instanceof GraphError &&
      error.diagnostic === 'INVALID_GRAPH' &&
      error.path === 'graph',
  );

  assert.strictEqual(
    adopted.topologyTransaction((edit) => edit.setDimension('Unused', { min: 1, max: 2 })),
    adopted,
  );
  const prior = adopted.snapshot();
  assert.throws(
    () => adopted.topologyTransaction(() => Promise.resolve()),
    ModelBuilderError,
  );
  assert.strictEqual(adopted.snapshot(), prior);
  assert.throws(
    () => adopted.topologyTransaction((outer) => outer.topologyTransaction(() => {})),
    ModelBuilderError,
  );
  assert.strictEqual(adopted.snapshot(), prior);
});

test('a rejected async callback cannot resume and mutate through its transaction editor', async () => {
  const builder = new ModelBuilder({
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [2] } },
    outputs: ['x'],
  });
  const priorGraph = builder.snapshot();
  const priorDocument = builder.documentSnapshot();
  const priorWeights = builder.weightDescriptorsSnapshot();
  const priorFingerprint = builder.fingerprint;
  let finishContinuation;
  const continuationFinished = new Promise((resolve) => {
    finishContinuation = resolve;
  });
  const lateErrors = [];

  assert.throws(() => builder.topologyTransaction(async (edit) => {
    const capturedSetInput = edit.setInput;
    const fluentEditor = edit.setDimension('Unused', { min: 1, max: 2 });
    assert.strictEqual(fluentEditor, edit);
    await Promise.resolve();
    for (const operation of [
      () => edit.setInput('late', { dtype: 'float32', shape: [2] }),
      () => capturedSetInput('captured', { dtype: 'float32', shape: [2] }),
      () => fluentEditor.selectOutputs(['late']),
    ]) {
      try {
        operation();
      } catch (error) {
        lateErrors.push(error);
      }
    }
    finishContinuation();
  }), ModelBuilderError);

  await continuationFinished;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(lateErrors.length, 3);
  for (const error of lateErrors) {
    assert.ok(error instanceof ModelBuilderError);
    assert.match(error.message, /no longer active|late asynchronous edits/i);
  }
  assert.strictEqual(builder.snapshot(), priorGraph);
  assert.strictEqual(builder.documentSnapshot(), priorDocument);
  assert.strictEqual(builder.weightDescriptorsSnapshot(), priorWeights);
  assert.equal(builder.fingerprint, priorFingerprint);
});
