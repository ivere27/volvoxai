import test from 'node:test';
import assert from 'node:assert/strict';

import {
  Backend,
  DataType,
  Empty,
  GraphNode,
  OpType,
  PingResponse,
  RunRequest,
  StringEntry,
  Tensor,
  TensorShape,
  TensorShapeEntry,
} from '../runtime/generated/typescript/volvoxai_lite.js';
import { VolvoxAiServiceFfi } from '../runtime/generated/typescript/volvoxai_ffi.js';

test('generated Synurang TypeScript messages round-trip the VolvoxAI wire contract', () => {
  const request = new RunRequest({
    modelId: 'model-7',
    inputs: [new Tensor({
      name: 'tokens',
      shape: [1n, 3n],
      dtype: DataType.DATA_TYPE_I32,
      data: new Uint8Array([7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0]),
    })],
    outputNames: ['logits'],
    lastToken: 2,
  });

  const decoded = RunRequest.fromBinary(request.toBinary());
  assert.equal(decoded.modelId, 'model-7');
  assert.deepEqual(decoded.outputNames, ['logits']);
  assert.equal(decoded.lastToken, 2);
  assert.equal(decoded.inputs[0].name, 'tokens');
  assert.deepEqual(decoded.inputs[0].shape, [1n, 3n]);
  assert.equal(decoded.inputs[0].dtype, DataType.DATA_TYPE_I32);
  assert.deepEqual(decoded.inputs[0].data, request.inputs[0].data);
});

test('generated entry messages preserve map-compatible fields and local Empty', () => {
  const node = new GraphNode({
    index: 3,
    id: 'linear',
    op: OpType.OP_LINEAR,
    opName: 'Linear',
    inputs: [new StringEntry({ key: 'input', value: 'hidden' })],
    outputs: [new StringEntry({ key: 'out', value: 'logits' })],
    outputShapes: [new TensorShapeEntry({
      key: 'out',
      value: new TensorShape({ dims: [1n, 32000n] }),
    })],
  });

  const decoded = GraphNode.fromBinary(node.toBinary());
  assert.deepEqual(decoded.inputs.map(({ key, value }) => [key, value]), [['input', 'hidden']]);
  assert.deepEqual(decoded.outputs.map(({ key, value }) => [key, value]), [['out', 'logits']]);
  assert.deepEqual(decoded.outputShapes[0].value?.dims, [1n, 32000n]);
  assert.deepEqual(new Empty().toBinary(), new Uint8Array());
});

test('generated Synurang TypeScript FFI client uses the canonical service route', () => {
  const calls = [];
  const client = new VolvoxAiServiceFfi({
    invoke(serviceName, methodName, data) {
      calls.push({ serviceName, methodName, data });
      return new PingResponse({
        version: '0.2.0',
        availableBackends: [Backend.BACKEND_CPU],
      }).toBinary();
    },
    openStream() {
      throw new Error('unexpected stream');
    },
  });

  const response = client.ping(new Empty());
  assert.equal(response.version, '0.2.0');
  assert.deepEqual(response.availableBackends, [Backend.BACKEND_CPU]);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].serviceName, 'VolvoxAiService');
  assert.equal(calls[0].methodName, '/volvoxai.v1.VolvoxAiService/Ping');
  assert.deepEqual(calls[0].data, new Uint8Array());
});
