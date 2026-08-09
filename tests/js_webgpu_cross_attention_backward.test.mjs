import test from 'node:test';
import assert from 'node:assert/strict';

import { WebGPUAutograd } from '../ts/training/WebGPUAutograd.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});

function tensor(name, shape, dtype = 'float32') {
  return {
    name,
    shape,
    dtype,
    sizeBytes: shape.reduce((size, dimension) => size * dimension, 1) * 4,
  };
}

function mockDevice() {
  return {
    createBuffer(descriptor) {
      return { descriptor, bytes: new Uint8Array(descriptor.size), destroy() {} };
    },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        const sourceBytes = source instanceof ArrayBuffer
          ? new Uint8Array(source)
          : new Uint8Array(source.buffer, source.byteOffset, source.byteLength);
        destination.bytes.set(sourceBytes.subarray(sourceOffset, size == null ? undefined : sourceOffset + size), offset);
      },
    },
  };
}

class DispatchRecorder extends WebGPUAutograd {
  async _dispatch(shaderName, entryPoint, entries, workgroupCount, resources = []) {
    return { shaderName, entryPoint, entries, workgroupCount, resources };
  }
}

function makeTrainer(node, tensors) {
  const device = mockDevice();
  const gpuBuffers = new Map(tensors.map((value) => [value.name, { tensor: value.name }]));
  return new DispatchRecorder(device, { nodes: [node] }, { gpuBuffers });
}

function entry(dispatch, binding) {
  return dispatch.entries.find(([index]) => index === binding)[1];
}

test('WebGPU CrossAttention backward dispatches F32 Q/K/V projection and affine gradients', async () => {
  const q = tensor('q', [2, 3, 4]);
  const kv = tensor('kv', [2, 2, 4]);
  const weight = tensor('weight', [12, 4]);
  const scale = tensor('scale', [12]);
  const bias = tensor('bias', [12]);
  const output = tensor('out', [2, 3, 4]);
  const node = {
    id: 'cross_attention_f32', opType: 'CrossAttention',
    inputs: { q, kv, weight, scale, bias }, outputs: { out: output }, params: { heads: 2 },
  };
  const trainer = makeTrainer(node, [q, kv, weight, scale, bias, output]);
  const gradOutput = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOutput);

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map((dispatch) => [dispatch.shaderName, dispatch.entryPoint]), [
    ['crossAttentionBackward', 'q_main'],
    ['crossAttentionBackward', 'kv_main'],
    ['crossAttentionBackward', 'weight_main'],
    ['crossAttentionBackward', 'scale_main'],
    ['crossAttentionBackward', 'bias_main'],
  ]);
  assert.deepEqual(dispatches.map((dispatch) => dispatch.entries.map(([binding]) => binding)), [
    [0, 1, 2, 3, 4, 5, 6, 11],
    [0, 1, 2, 3, 4, 5, 7, 11],
    [0, 1, 2, 3, 4, 5, 8, 11],
    [0, 1, 2, 3, 4, 5, 9, 11],
    [0, 1, 2, 3, 4, 5, 10, 11],
  ]);
  assert.equal(entry(dispatches[0], 0).tensor, q.name);
  assert.equal(entry(dispatches[0], 1).tensor, kv.name);
  assert.equal(entry(dispatches[0], 2).tensor, weight.name);
  assert.equal(entry(dispatches[0], 3).tensor, scale.name);
  assert.equal(entry(dispatches[0], 4).tensor, bias.name);
  assert.equal(entry(dispatches[0], 5), gradOutput);
  assert.equal(entry(dispatches[0], 6), trainer.gradientBuffers.get(q.name));
  assert.equal(entry(dispatches[1], 7), trainer.gradientBuffers.get(kv.name));
  assert.equal(entry(dispatches[2], 8), trainer.gradientBuffers.get(weight.name));
  assert.equal(entry(dispatches[3], 9), trainer.gradientBuffers.get(scale.name));
  assert.equal(entry(dispatches[4], 10), trainer.gradientBuffers.get(bias.name));
  assert.deepEqual(dispatches.map((dispatch) => dispatch.workgroupCount), [
    [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
  ]);

  const params = entry(dispatches[0], 11);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 12)], [
    3, 2, 4, 2, 2, 2, 24, 16, 48, 12, 1, 1,
  ]);
  assert.ok(Math.abs(new Float32Array(params.bytes.buffer)[12] - 1 / Math.sqrt(2)) < 1e-7);
});

test('WebGPU CrossAttention backward uses identity/no-bias affine semantics when parameters are absent', async () => {
  const q = tensor('q', [3, 4]);
  const kv = tensor('kv', [2, 4]);
  const weight = tensor('weight', [12, 4]);
  const output = tensor('out', [3, 4]);
  const node = {
    id: 'cross_attention_no_affine', opType: 'CrossAttention',
    inputs: { q, kv, weight }, outputs: { out: output }, params: { heads: 2 },
  };
  const trainer = makeTrainer(node, [q, kv, weight, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map((dispatch) => dispatch.entryPoint), ['q_main', 'kv_main', 'weight_main']);
  const params = entry(dispatches[0], 11);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 12)], [
    3, 2, 4, 2, 2, 1, 12, 8, 48, 12, 0, 0,
  ]);
  assert.match(entry(dispatches[0], 3).descriptor.label, /TrainingDummy/);
  assert.match(entry(dispatches[0], 4).descriptor.label, /TrainingDummy/);
});

test('WebGPU CrossAttention backward rejects packed-I8 weights and the F32 forward width limit', async () => {
  const q = tensor('q', [2, 4]);
  const kv = tensor('kv', [2, 4]);
  const int8Weight = tensor('weight_i8', [12, 4], 'int8');
  const output = tensor('out', [2, 4]);
  const packedNode = {
    id: 'cross_attention_i8', opType: 'CrossAttention',
    inputs: { q, kv, weight: int8Weight }, outputs: { out: output }, params: { heads: 2 },
  };
  const packedTrainer = makeTrainer(packedNode, [q, kv, int8Weight, output]);
  packedTrainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });
  await assert.rejects(() => packedTrainer._buildBackwardDispatches(), /F32 Q\/KV\/projection\/output/);

  const wideQ = tensor('wide_q', [1, 128]);
  const wideKv = tensor('wide_kv', [1, 128]);
  const wideWeight = tensor('wide_weight', [384, 128]);
  const wideOutput = tensor('wide_out', [1, 128]);
  const wideNode = {
    id: 'cross_attention_wide', opType: 'CrossAttention',
    inputs: { q: wideQ, kv: wideKv, weight: wideWeight }, outputs: { out: wideOutput }, params: { heads: 2 },
  };
  const wideTrainer = makeTrainer(wideNode, [wideQ, wideKv, wideWeight, wideOutput]);
  wideTrainer.gradientBuffers.set(wideOutput.name, { tensor: 'grad_out' });
  await assert.rejects(() => wideTrainer._buildBackwardDispatches(), /d_model <= 64/);
});
