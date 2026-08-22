import test from 'node:test';
import assert from 'node:assert/strict';

import { WebGPUAutograd } from '../ts/training/WebGPUAutograd.js';
import { DataType } from '../ts/generated/volvoxaiEnums.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});

const dtypeBytes = { float32: 4, int32: 4, int8: 1, uint8: 1 };

function tensor(name, shape, dtype = 'float32') {
  const elements = shape.reduce((size, dimension) => size * dimension, 1);
  return { name, shape, dtype, sizeBytes: elements * dtypeBytes[dtype] };
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

test('WebGPU Cast backward copies an F32-to-F32 gradient and stops integer conversions', async () => {
  const input = tensor('input', [2, 2]);
  const output = tensor('out', [4]);
  const node = {
    id: 'cast_f32', opType: 'Cast', inputs: { input }, outputs: { out: output }, params: { to: 'float32' },
  };
  const trainer = makeTrainer(node, [input, output]);
  const gradOutput = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOutput);

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'copyBackward');
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2]);
  assert.equal(entry(dispatch, 0), gradOutput);
  assert.equal(entry(dispatch, 1), trainer.gradientBuffers.get(input.name));
  assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(entry(dispatch, 2).bytes.buffer).slice(0, 4)], [4, 0, 0, 0]);

  for (const [castInput, castOutput] of [
    [tensor('int_input', [4], 'int8'), tensor('float_out', [4])],
    [tensor('float_input', [4]), tensor('int_out', [4], 'int32')],
  ]) {
    const integerNode = {
      id: `cast_${castInput.dtype}_${castOutput.dtype}`, opType: 'Cast',
      inputs: { input: castInput }, outputs: { out: castOutput }, params: {},
    };
    const integerTrainer = makeTrainer(integerNode, [castInput, castOutput]);
    integerTrainer.gradientBuffers.set(castOutput.name, { tensor: 'synthetic_grad_out' });
    assert.deepEqual(await integerTrainer._buildBackwardDispatches(), []);
    assert.equal(integerTrainer.gradientBuffers.has(castInput.name), false);
  }
});

test('WebGPU DequantizeLinear backward dispatches F32 input and scalar-scale gradients', async () => {
  const input = tensor('input', [2, 2]);
  const scale = tensor('scale', [1]);
  const zeroPoint = tensor('zero', [1]);
  const output = tensor('out', [2, 2]);
  const node = {
    id: 'dequant_f32', opType: 'DequantizeLinear',
    inputs: { input, scale, zero_point: zeroPoint }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [input, scale, zeroPoint, output]);
  const gradOutput = { tensor: 'grad_out' };
  trainer.gradientBuffers.set(output.name, gradOutput);

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map((dispatch) => [dispatch.shaderName, dispatch.entryPoint]), [
    ['dequantizeLinearBackward', 'input_main'],
    ['dequantizeLinearBackward', 'scale_main'],
  ]);
  assert.deepEqual(dispatches[0].entries.map(([binding]) => binding), [2, 3, 4, 6]);
  assert.deepEqual(dispatches[1].entries.map(([binding]) => binding), [0, 1, 3, 5, 6]);
  assert.equal(entry(dispatches[0], 2).tensor, scale.name);
  assert.equal(entry(dispatches[0], 3), gradOutput);
  assert.equal(entry(dispatches[0], 4), trainer.gradientBuffers.get(input.name));
  assert.equal(entry(dispatches[1], 0).tensor, input.name);
  assert.equal(entry(dispatches[1], 1).tensor, zeroPoint.name);
  assert.equal(entry(dispatches[1], 5), trainer.gradientBuffers.get(scale.name));
  assert.equal(trainer.gradientBuffers.has(zeroPoint.name), false);
  assert.deepEqual(
    [...new Uint32Array(entry(dispatches[0], 6).bytes.buffer).slice(0, 4)],
    [4, DataType.F32, DataType.F32, 1],
  );
});

test('WebGPU DequantizeLinear keeps quantized inputs non-differentiable while training F32 scale', async () => {
  const input = tensor('input', [5], 'uint8');
  const scale = tensor('scale', [1]);
  const zeroPoint = tensor('zero', [1], 'int8');
  const output = tensor('out', [5]);
  const node = {
    id: 'dequant_u8', opType: 'DequantizeLinear',
    inputs: { input, scale, zero_point: zeroPoint }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [input, scale, zeroPoint, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'dequantizeLinearBackward');
  assert.equal(dispatch.entryPoint, 'scale_main');
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 3, 5, 6]);
  assert.equal(trainer.gradientBuffers.has(input.name), false);
  assert.equal(trainer.gradientBuffers.has(zeroPoint.name), false);
  assert.equal(entry(dispatch, 5), trainer.gradientBuffers.get(scale.name));
  assert.deepEqual(
    [...new Uint32Array(entry(dispatch, 6).bytes.buffer).slice(0, 4)],
    [5, DataType.U8, DataType.I8, 1],
  );

  const noZeroNode = {
    id: 'dequant_i32_no_zero', opType: 'DequantizeLinear',
    inputs: { input: tensor('i32_input', [2], 'int32'), scale: tensor('no_zero_scale', [1]) },
    outputs: { out: tensor('no_zero_out', [2]) }, params: {},
  };
  const noZeroTensors = [noZeroNode.inputs.input, noZeroNode.inputs.scale, noZeroNode.outputs.out];
  const noZeroTrainer = makeTrainer(noZeroNode, noZeroTensors);
  noZeroTrainer.gradientBuffers.set(noZeroNode.outputs.out.name, { tensor: 'grad_out' });
  const [noZeroDispatch] = await noZeroTrainer._buildBackwardDispatches();
  assert.match(entry(noZeroDispatch, 1).descriptor.label, /TrainingDummy/);
  assert.deepEqual(
    [...new Uint32Array(entry(noZeroDispatch, 6).bytes.buffer).slice(0, 4)],
    [2, DataType.I32, DataType.F32, 0],
  );
});

test('WebGPU DequantizeLinear scale gradient retains raw I32 zero-point metadata', async () => {
  const input = tensor('input', [2], 'int32');
  const scale = tensor('scale', [1]);
  const zeroPoint = tensor('zero', [1], 'int32');
  const output = tensor('out', [2]);
  const node = {
    id: 'dequant_i32_high_range', opType: 'DequantizeLinear',
    inputs: { input, scale, zero_point: zeroPoint }, outputs: { out: output }, params: {},
  };
  const trainer = makeTrainer(node, [input, scale, zeroPoint, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.shaderName, 'dequantizeLinearBackward');
  assert.equal(dispatch.entryPoint, 'scale_main');
  // [size, input dtype, zero-point dtype, has zero-point]. The raw integer
  // decoder uses these flags before it performs the high-range subtraction.
  assert.deepEqual(
    [...new Uint32Array(entry(dispatch, 6).bytes.buffer).slice(0, 4)],
    [2, DataType.I32, DataType.I32, 1],
  );

  const floatZeroPoint = tensor('float_zero', [1]);
  const floatZeroNode = {
    id: 'dequant_i32_f32_zero', opType: 'DequantizeLinear',
    inputs: { input, scale, zero_point: floatZeroPoint }, outputs: { out: output }, params: {},
  };
  const floatZeroTrainer = makeTrainer(floatZeroNode, [input, scale, floatZeroPoint, output]);
  floatZeroTrainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });
  const [floatZeroDispatch] = await floatZeroTrainer._buildBackwardDispatches();
  assert.deepEqual(
    [...new Uint32Array(entry(floatZeroDispatch, 6).bytes.buffer).slice(0, 4)],
    [2, DataType.I32, DataType.F32, 1],
  );
});

test('WebGPU Cast and DequantizeLinear backward reject invalid typed storage contracts', async () => {
  const castInput = tensor('cast_input', [2]);
  const castOutput = tensor('cast_out', [3]);
  const castNode = {
    id: 'cast_bad_shape', opType: 'Cast', inputs: { input: castInput }, outputs: { out: castOutput }, params: {},
  };
  const castTrainer = makeTrainer(castNode, [castInput, castOutput]);
  castTrainer.gradientBuffers.set(castOutput.name, { tensor: 'grad_out' });
  await assert.rejects(() => castTrainer._buildBackwardDispatches(), /equal-size F32\/I32\/I8\/U8/);

  const input = tensor('input', [2], 'int8');
  const vectorScale = tensor('scale', [2]);
  const output = tensor('out', [2]);
  const dequantNode = {
    id: 'dequant_bad_scale', opType: 'DequantizeLinear',
    inputs: { input, scale: vectorScale }, outputs: { out: output }, params: {},
  };
  const dequantTrainer = makeTrainer(dequantNode, [input, vectorScale, output]);
  dequantTrainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });
  await assert.rejects(() => dequantTrainer._buildBackwardDispatches(), /F32 output\/scalar scale/);
});
