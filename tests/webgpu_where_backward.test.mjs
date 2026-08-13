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
  const graph = { nodes: [node] };
  return new DispatchRecorder(device, graph, { gpuBuffers });
}

function entry(dispatch, binding) {
  return dispatch.entries.find(([index]) => index === binding)[1];
}

test('WebGPU Where backward dispatches exact-shape F32 conditions without a condition gradient', async () => {
  const condition = tensor('condition', [2, 2]);
  const x = tensor('x', [2, 2]);
  const y = tensor('y', [2, 2]);
  const output = tensor('out', [2, 2]);
  const node = {
    id: 'where_f32', opType: 'Where',
    inputs: { cond: condition, x, y }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [condition, x, y, output]);
  const gradOut = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOut);

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'whereBackward');
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2, 3, 4]);
  assert.equal(entry(dispatch, 0).tensor, condition.name);
  assert.equal(entry(dispatch, 1), gradOut);
  assert.equal(entry(dispatch, 2), trainer.gradientBuffers.get(x.name));
  assert.equal(entry(dispatch, 3), trainer.gradientBuffers.get(y.name));
  assert.equal(trainer.gradientBuffers.has(condition.name), false);
  assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 4).bytes.buffer).slice(0, 2)], [4, 1]);
});

test('WebGPU Mask backward dispatches exact-shape I32 conditions', async () => {
  const mask = tensor('mask', [2, 2], 'int32');
  const a = tensor('a', [2, 2]);
  const b = tensor('b', [2, 2]);
  const output = tensor('out', [2, 2]);
  const node = {
    id: 'mask_i32', opType: 'Mask',
    inputs: { mask, a, b }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [mask, a, b, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'whereBackward');
  assert.equal(entry(dispatch, 0).tensor, mask.name);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 4).bytes.buffer).slice(0, 2)], [4, 0]);
  assert.equal(trainer.gradientBuffers.has(mask.name), false);
});

test('WebGPU Where backward rejects equal-element-count but non-exact shapes', async () => {
  const condition = tensor('condition', [1, 4], 'int32');
  const x = tensor('x', [2, 2]);
  const y = tensor('y', [2, 2]);
  const output = tensor('out', [2, 2]);
  const node = {
    id: 'where_shape', opType: 'Where',
    inputs: { condition, x, y }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [condition, x, y, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  await assert.rejects(() => trainer._buildBackwardDispatches(), /exact-shape F32 operands/);
});
