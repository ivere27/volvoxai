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

function tensor(name, shape, dtype = 'float32', buffer = null) {
  const bytes = dtype === 'int8' || dtype === 'uint8' ? 1 : 4;
  return {
    name,
    shape,
    dtype,
    buffer,
    sizeBytes: shape.reduce((size, dimension) => size * dimension, 1) * bytes,
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

test('WebGPU Gather backward dispatches canonical I32 arbitrary-axis scatter-add metadata', async () => {
  const data = tensor('data', [2, 3, 4]);
  const indices = tensor('indices', [2, 2], 'int32');
  const output = tensor('out', [2, 2, 2, 4]);
  const node = {
    id: 'gather_axis_one', opType: 'Gather', inputs: { input: data, indices }, outputs: { out: output },
    params: { axis: -2 },
  };
  const trainer = makeTrainer(node, [data, indices, output]);
  const gradOut = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOut);

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'gatherBackward');
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2, 3]);
  assert.equal(entry(dispatch, 0).tensor, indices.name);
  assert.equal(entry(dispatch, 1), gradOut);
  assert.equal(entry(dispatch, 2), trainer.gradientBuffers.get(data.name));
  assert.equal(trainer.gradientBuffers.has(indices.name), false);
  assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 3).bytes.buffer).slice(0, 4)], [2, 3, 4, 4]);
});

test('WebGPU Gather backward supports data rank eight and rejects float indices', async () => {
  const data = tensor('rank_eight', [2, 1, 1, 1, 1, 1, 1, 3]);
  const indices = tensor('rank_eight_indices', [2], 'int32');
  const output = tensor('rank_eight_out', [2, 1, 1, 1, 1, 1, 1, 2]);
  const node = {
    id: 'gather_rank_eight', opType: 'Gather', inputs: { input: data, indices }, outputs: { out: output },
    params: { axis: -1 },
  };
  const trainer = makeTrainer(node, [data, indices, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });
  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.deepEqual([...new Uint32Array(entry(dispatch, 3).bytes.buffer).slice(0, 4)], [2, 3, 1, 2]);

  const floatIndices = tensor('float_indices', [2], 'float32');
  const rejectedOutput = tensor('rejected_out', [2, 3]);
  const rejectedNode = {
    id: 'gather_float_indices', opType: 'Gather',
    inputs: { input: tensor('rejected_data', [4, 3]), indices: floatIndices },
    outputs: { out: rejectedOutput }, params: { axis: 0 },
  };
  const rejectedTrainer = makeTrainer(rejectedNode, [rejectedNode.inputs.input, floatIndices, rejectedOutput]);
  rejectedTrainer.gradientBuffers.set(rejectedOutput.name, { tensor: 'grad_out' });
  await assert.rejects(() => rejectedTrainer._buildBackwardDispatches(), /I32 indices/);
});

test('WebGPU GatherElements backward encodes negative-index-compatible metadata', async () => {
  const data = tensor('data', [2, 3, 4]);
  // The dispatch must retain canonical I32 storage; the shader normalizes the
  // negative values instead of creating an index gradient.
  const indices = tensor('indices', [2, 2, 4], 'int32', Int32Array.of(
    -3, -1, 0, 1, 2, -2, 0, -1,
    -3, -1, 0, 1, 2, -2, 0, -1,
  ));
  const output = tensor('out', [2, 2, 4]);
  const node = {
    id: 'gather_elements', opType: 'GatherElements',
    inputs: { input: data, indices }, outputs: { out: output }, params: { axis: -2 },
  };
  const trainer = makeTrainer(node, [data, indices, output]);
  const gradOut = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOut);

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'gatherElementsBackward');
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2, 3]);
  assert.equal(entry(dispatch, 0).tensor, indices.name);
  assert.equal(entry(dispatch, 1), gradOut);
  assert.equal(entry(dispatch, 2), trainer.gradientBuffers.get(data.name));
  assert.equal(trainer.gradientBuffers.has(indices.name), false);
  assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
  const params = new Uint32Array(entry(dispatch, 3).bytes.buffer);
  assert.deepEqual([...params.slice(0, 4)], [3, 1, 24, 16]);
  assert.deepEqual([...params.slice(4, 7)], [2, 3, 4]);
  assert.deepEqual([...params.slice(12, 15)], [2, 2, 4]);

  const floatIndices = tensor('float_indices', [2, 2, 4]);
  const rejectedOutput = tensor('rejected_out', [2, 2, 4]);
  const rejectedNode = {
    id: 'gather_elements_float_indices', opType: 'GatherElements',
    inputs: { input: data, indices: floatIndices }, outputs: { out: rejectedOutput }, params: { axis: 1 },
  };
  const rejectedTrainer = makeTrainer(rejectedNode, [data, floatIndices, rejectedOutput]);
  rejectedTrainer.gradientBuffers.set(rejectedOutput.name, { tensor: 'grad_out' });
  await assert.rejects(() => rejectedTrainer._buildBackwardDispatches(), /I32 indices/);
});
