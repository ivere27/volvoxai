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

function tensor(name, shape) {
  return {
    name,
    shape,
    dtype: 'float32',
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

function visionCase(opType, outputShape) {
  const input = tensor('input', [2, 3, 5, 4]);
  const output = tensor('out', outputShape);
  const node = { id: opType, opType, inputs: { input }, outputs: { out: output }, params: {} };
  const trainer = makeTrainer(node, [input, output]);
  const gradOut = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOut);
  return { input, output, trainer, gradOut };
}

test('WebGPU MeanHeight backward encodes NHWC dimensions and excludes unused forward buffers', async () => {
  const { input, trainer, gradOut } = visionCase('MeanHeight', [2, 4, 5]);
  const [dispatch] = await trainer._buildBackwardDispatches();

  assert.equal(dispatch.shaderName, 'visionBackward');
  assert.equal(dispatch.entryPoint, 'mean_height_main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [2, 3, 4]);
  assert.equal(entry(dispatch, 2), gradOut);
  assert.equal(entry(dispatch, 3), trainer.gradientBuffers.get(input.name));
  assert.deepEqual(dispatch.workgroupCount, [1, 4, 2]);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 4).bytes.buffer).slice(0, 4)], [3, 5, 4, 2]);
});

test('WebGPU ProfileX and ProfileY backward dispatch their strict-first-max kernels', async () => {
  const profileX = visionCase('ProfileX', [2, 8, 5]);
  const [xDispatch] = await profileX.trainer._buildBackwardDispatches();
  assert.equal(xDispatch.entryPoint, 'profile_x_main');
  assert.deepEqual(xDispatch.entries.map(([binding]) => binding), [0, 2, 3, 4]);
  assert.equal(entry(xDispatch, 0).tensor, profileX.input.name);
  assert.deepEqual(xDispatch.workgroupCount, [1, 4, 2]);

  const profileY = visionCase('ProfileY', [2, 8, 3]);
  const [yDispatch] = await profileY.trainer._buildBackwardDispatches();
  assert.equal(yDispatch.entryPoint, 'profile_y_main');
  assert.deepEqual(yDispatch.entries.map(([binding]) => binding), [0, 2, 3, 4]);
  assert.equal(entry(yDispatch, 0).tensor, profileY.input.name);
  assert.deepEqual(yDispatch.workgroupCount, [1, 4, 2]);
});

test('WebGPU SpatialSoftargmaxY backward binds the forward expectation', async () => {
  const { input, output, trainer, gradOut } = visionCase('SpatialSoftargmaxY', [2, 4, 5]);
  const [dispatch] = await trainer._buildBackwardDispatches();

  assert.equal(dispatch.entryPoint, 'spatial_softargmax_y_main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2, 3, 4]);
  assert.equal(entry(dispatch, 0).tensor, input.name);
  assert.equal(entry(dispatch, 1).tensor, output.name);
  assert.equal(entry(dispatch, 2), gradOut);
  assert.deepEqual(dispatch.workgroupCount, [1, 4, 2]);
});

test('WebGPU vision backward rejects an incompatible profile output shape', async () => {
  const { trainer } = visionCase('ProfileX', [2, 4, 5]);
  await assert.rejects(() => trainer._buildBackwardDispatches(), /incompatible output shape/);
});
