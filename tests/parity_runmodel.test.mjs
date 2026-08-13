import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildInputs,
  captureStableResult,
  concreteExecutionInputs,
  runtimeFailureMessage,
} from './parity/lib/runmodel.mjs';

test('parity fixtures always expose explicit concrete shapes', () => {
  const inputs = buildInputs({
    inputs: [{
      name: 'pixels',
      gen: 'seededU8',
      dtype: 'u8',
      shape: [1, 2, 3],
      seed: 7,
    }],
  });

  assert.ok(inputs.pixels.data instanceof Uint8Array);
  assert.equal(inputs.pixels.data.length, 6);
  assert.deepEqual(inputs.pixels.shape, [1, 2, 3]);
  assert.equal(Object.isFrozen(inputs.pixels), true);
  assert.equal(Object.isFrozen(inputs.pixels.shape), true);
  assert.throws(
    () => buildInputs({
      inputs: [{ name: 'pixels', gen: 'seededU8', dtype: 'u8', seed: 7 }],
    }),
    /requires one explicit positive shape/,
  );
});

test('dynamic parity execution rejects unshaped typed storage', () => {
  const dynamicSnapshot = {
    inputNames: ['tokens'],
    staticShapePlan: null,
  };
  const data = Int32Array.of(1, 2, 3);
  assert.throws(
    () => concreteExecutionInputs(dynamicSnapshot, { tokens: data }),
    /requires an explicit \{ data, shape \} fixture/,
  );

  const shaped = Object.freeze({ data, shape: Object.freeze([1, 3]) });
  assert.strictEqual(
    concreteExecutionInputs(dynamicSnapshot, { tokens: shaped }).tokens,
    shaped,
  );
});

test('parity runtime errors surface the provider compilation report', () => {
  assert.equal(runtimeFailureMessage({
    message: "Required backend 'webgpu' could not compile the model.",
    report: {
      candidates: [{
        backend: 'webgpu',
        outcome: 'unsupported',
        message: "WebGPU cannot prove MatMul at node 'projection'.",
      }],
    },
  }), "Required backend 'webgpu' could not compile the model. " +
    "webgpu unsupported: WebGPU cannot prove MatMul at node 'projection'.");
});

test('stable parity capture validates symbolic outputs against concrete results', async () => {
  const values = Float32Array.of(1, 2, 3, 4, 5, 6);
  const graph = {
    tensors: {
      logits: { dtype: 'float32', shape: [1, 'S', 2] },
    },
  };
  let contextClosed = false;
  let resultClosed = false;
  const tensorResult = {
    dtype: 'float32',
    shape: [1, 3, 2],
    location: 'host',
    async read() {
      return new Float32Array(values);
    },
  };
  const result = {
    report: { contextId: 'context-1', backend: 'cpu-js' },
    output(name) {
      assert.equal(name, 'logits');
      return tensorResult;
    },
    async close() {
      resultClosed = true;
    },
  };
  const context = {
    id: 'context-1',
    async close() {
      contextClosed = true;
    },
  };

  const captured = await captureStableResult(
    graph,
    result,
    context,
    'cpu-js',
    ['logits'],
  );
  assert.deepEqual([...captured.outputs.logits], [...values]);
  assert.deepEqual(captured.stableResult.outputs[0].shape, [1, 3, 2]);
  assert.equal(contextClosed, true);
  assert.equal(resultClosed, true);
});
