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

test('WebGPU Slice backward dispatches positive-step rank-2 metadata', async () => {
  const input = tensor('input', [2, 5]);
  const output = tensor('out', [2, 2]);
  const node = {
    id: 'slice', opType: 'Slice', inputs: { input }, outputs: { out: output },
    params: { axes: [-1], starts: [-4], steps: [2] },
  };
  const trainer = makeTrainer(node, [input, output]);
  const gradOut = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOut);

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'sliceBackward');
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2]);
  assert.equal(entry(dispatch, 0), gradOut);
  assert.equal(entry(dispatch, 1), trainer.gradientBuffers.get(input.name));
  assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 2).bytes.buffer).slice(0, 36)], [
    2, 4, 0, 0,
    2, 2, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 0, 0,
    1, 2, 1, 1, 1, 1, 1, 1,
    5, 1, 1, 1, 1, 1, 1, 1,
  ]);
});

test('WebGPU Slice backward rejects negative or zero steps', async () => {
  const input = tensor('input', [4]);
  const output = tensor('out', [2]);
  const node = {
    id: 'slice_step', opType: 'Slice', inputs: { input }, outputs: { out: output },
    params: { axes: [0], starts: [3], steps: [-1] },
  };
  const trainer = makeTrainer(node, [input, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  await assert.rejects(() => trainer._buildBackwardDispatches(), /positive integer steps/);
});

test('WebGPU Slice backward dispatches normalized rank-five metadata', async () => {
  const input = tensor('input', [2, 3, 4, 5, 6]);
  const output = tensor('out', [1, 3, 2, 5, 3]);
  const node = {
    id: 'slice_rank_five', opType: 'Slice', inputs: { input }, outputs: { out: output },
    params: { axes: [-5, -3, -1], starts: [-1, 1, -5], steps: [1, 2, 2] },
  };
  const trainer = makeTrainer(node, [input, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'sliceBackward');
  assert.deepEqual([...new Uint32Array(entry(dispatch, 2).bytes.buffer).slice(0, 36)], [
    5, 90, 0, 0,
    1, 3, 2, 5, 3, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 0, 0,
    1, 1, 2, 1, 2, 1, 1, 1,
    360, 120, 30, 6, 1, 1, 1, 1,
  ]);
});

test('WebGPU Slice backward accepts the rank-eight upper bound', async () => {
  const input = tensor('input', [1, 1, 1, 1, 1, 1, 2, 3]);
  const output = tensor('out', [1, 1, 1, 1, 1, 1, 1, 2]);
  const node = {
    id: 'slice_rank_eight', opType: 'Slice', inputs: { input }, outputs: { out: output },
    params: { axes: [-2, -1], starts: [-1, -3], steps: [1, 2] },
  };
  const trainer = makeTrainer(node, [input, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'sliceBackward');
  assert.deepEqual([...new Uint32Array(entry(dispatch, 2).bytes.buffer).slice(0, 36)], [
    8, 2, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 2,
    0, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 2,
    6, 6, 6, 6, 6, 6, 3, 1,
  ]);
});

test('WebGPU Slice backward rejects rank beyond the canonical forward contract', async () => {
  const input = tensor('input', new Array(9).fill(1));
  const output = tensor('out', new Array(9).fill(1));
  const node = {
    id: 'slice_rank_nine', opType: 'Slice', inputs: { input }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [input, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  await assert.rejects(() => trainer._buildBackwardDispatches(), /rank 1\.\.8 F32 input\/output tensors/);
});
