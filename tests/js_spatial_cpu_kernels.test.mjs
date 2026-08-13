import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { _cpuBatchMatMul } from '../ts/ops/batchMatMul.js';
import { _cpuConv1D } from '../ts/ops/conv1D.js';
import { _cpuConv2D } from '../ts/ops/conv2D.js';
import { _cpuConvTranspose2D } from '../ts/ops/convTranspose2D.js';
import { _cpuMaxPool2D } from '../ts/ops/maxPool2D.js';
import { _cpuAveragePool2D } from '../ts/ops/averagePool2D.js';
import { _cpuGlobalAveragePool } from '../ts/ops/globalAveragePool.js';
import { _cpuResize } from '../ts/ops/resize.js';
import { _cpuUpsample2x } from '../ts/ops/upsample2x.js';

const constructors = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

const kernels = Object.freeze({
  BatchMatMul: _cpuBatchMatMul,
  Conv1D: _cpuConv1D,
  Conv2D: _cpuConv2D,
  ConvTranspose2D: _cpuConvTranspose2D,
  MaxPool2D: _cpuMaxPool2D,
  AveragePool2D: _cpuAveragePool2D,
  GlobalAveragePool: _cpuGlobalAveragePool,
  Resize: _cpuResize,
  ResizeNearest2D: _cpuResize,
  UpsampleNearest2D: _cpuUpsample2x,
});

const corpus = JSON.parse(readFileSync(
  new URL('./operator_shape_contract_spatial_vectors.json', import.meta.url),
  'utf8',
));

function tensor(shape, dtype = 'float32', values = undefined, quantization = undefined) {
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  const Constructor = constructors[dtype];
  const buffer = values === undefined ? new Constructor(elements) : Constructor.from(values);
  assert.equal(buffer.length, elements, `fixture [${shape}] length`);
  return {
    shape: [...shape],
    dtype,
    buffer,
    sizeBytes: buffer.byteLength,
    ...(quantization === undefined ? {} : { quantization: structuredClone(quantization) }),
  };
}

function materialize(descriptor, values = undefined) {
  return tensor(
    descriptor.shape,
    descriptor.dtype,
    values,
    descriptor.quantization,
  );
}

function corpusNode(vector, invalidOutput = false) {
  const outputDescriptor = structuredClone(vector.expected.out);
  if (invalidOutput) outputDescriptor.shape[0]++;
  return {
    id: vector.id,
    opType: vector.operator,
    inputs: Object.fromEntries(
      Object.entries(vector.request.inputs).map(([name, descriptor]) => [name, materialize(descriptor)]),
    ),
    outputs: { out: materialize(outputDescriptor) },
    params: structuredClone(vector.request.params ?? {}),
  };
}

test('spatial CPU kernels execute both shared legal shapes for every canonical operator', () => {
  const counts = new Map(Object.keys(kernels).map((operator) => [operator, 0]));
  for (const vector of corpus.success_cases) {
    kernels[vector.operator](corpusNode(vector));
    counts.set(vector.operator, counts.get(vector.operator) + 1);
  }
  for (const [operator, count] of counts) {
    assert.ok(count >= 2, `${operator} legal CPU coverage`);
  }
});

test('every spatial CPU kernel rejects a descriptor-valid but semantically wrong output shape', () => {
  for (const operator of Object.keys(kernels)) {
    const vector = corpus.success_cases.find((candidate) => candidate.operator === operator);
    assert.throws(
      () => kernels[operator](corpusNode(vector, true)),
      /output|preserve NHWC/,
      operator,
    );
  }
});

test('spatial CPU kernels reject overlapping typed-array views of one backing buffer', () => {
  const backing = new ArrayBuffer(5 * Float32Array.BYTES_PER_ELEMENT);
  const input = {
    shape: [1, 3, 1],
    dtype: 'float32',
    buffer: new Float32Array(backing, 0, 3),
    sizeBytes: 3 * Float32Array.BYTES_PER_ELEMENT,
  };
  const output = {
    shape: [1, 3, 1],
    dtype: 'float32',
    buffer: new Float32Array(backing, 2 * Float32Array.BYTES_PER_ELEMENT, 3),
    sizeBytes: 3 * Float32Array.BYTES_PER_ELEMENT,
  };
  assert.throws(() => _cpuConv1D({
    opType: 'Conv1D',
    inputs: { input, weight: tensor([1, 1, 1], 'float32', [1]) },
    outputs: { out: output },
    params: {},
  }), /must not alias/);
});

test('Conv1D normalizes one-element spatial parameters and implements ReLU6 exactly', () => {
  const output = tensor([1, 3, 1]);
  _cpuConv1D({
    opType: 'Conv1D',
    inputs: {
      input: tensor([1, 3, 1], 'float32', [-2, 2, 4]),
      weight: tensor([1, 1, 1], 'float32', [2]),
    },
    outputs: { out: output },
    params: { stride: [1], padding: [0], relu: 2 },
  });
  assert.deepEqual([...output.buffer], [0, 4, 6]);
});

test('Conv2D uses explicit asymmetric top/left/bottom/right pads and rejects contradiction', () => {
  const output = tensor([1, 2, 2, 1]);
  _cpuConv2D({
    opType: 'Conv2D',
    inputs: {
      input: tensor([1, 1, 1, 1], 'float32', [2]),
      weight: tensor([1, 1, 1, 1], 'float32', [3]),
    },
    outputs: { out: output },
    params: { pads: [1, 0, 0, 1], weight_layout: 'HWCM' },
  });
  assert.deepEqual([...output.buffer], [0, 0, 6, 0]);

  assert.throws(() => _cpuConv2D({
    opType: 'Conv2D',
    inputs: {
      input: tensor([1, 2, 2, 1]),
      weight: tensor([1, 1, 1, 1]),
    },
    outputs: { out: tensor([1, 4, 4, 1]) },
    params: { padding: [1, 1], pads: [1, 1, 2, 2] },
  }), /same symmetric padding/);
});

test('ConvTranspose2D normalizes scalar and one-element spatial parameters', () => {
  const output = tensor([1, 2, 2, 1]);
  _cpuConvTranspose2D({
    opType: 'ConvTranspose2D',
    inputs: {
      input: tensor([1, 1, 1, 1], 'float32', [2]),
      weight: tensor([2, 2, 1, 1], 'float32', [1, 1, 1, 1]),
    },
    outputs: { out: output },
    params: { kernel: 2, stride: [1], padding: [0] },
  });
  assert.deepEqual([...output.buffer], [2, 2, 2, 2]);
});

test('MaxPool2D applies explicit asymmetric pads for F32 as well as byte storage', () => {
  const output = tensor([1, 2, 2, 1]);
  _cpuMaxPool2D({
    opType: 'MaxPool2D',
    inputs: { input: tensor([1, 2, 2, 1], 'float32', [1, 2, 3, 4]) },
    outputs: { out: output },
    params: { kernel: 2, pads: [1, 0, 0, 1] },
  });
  assert.deepEqual([...output.buffer], [2, 2, 4, 4]);
});

test('AveragePool2D honors symmetric pads while excluding padded cells from the divisor', () => {
  const output = tensor([1, 3, 3, 1]);
  _cpuAveragePool2D({
    opType: 'AveragePool2D',
    inputs: { input: tensor([1, 2, 2, 1], 'float32', [1, 2, 3, 4]) },
    outputs: { out: output },
    params: { kernel: 2, pads: [1, 1, 1, 1], count_include_pad: 0 },
  });
  assert.deepEqual([...output.buffer], [1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4]);
  assert.throws(() => _cpuAveragePool2D({
    opType: 'AveragePool2D',
    inputs: { input: tensor([1, 2, 2, 1]) },
    outputs: { out: tensor([1, 2, 2, 1]) },
    params: { kernel: 2, pads: [1, 0, 0, 1] },
  }), /pads must be symmetric/);
});

test('GlobalAveragePool reduces each batch independently', () => {
  const output = tensor([2, 1, 1, 1]);
  _cpuGlobalAveragePool({
    opType: 'GlobalAveragePool',
    inputs: { input: tensor([2, 1, 2, 1], 'float32', [1, 3, 10, 14]) },
    outputs: { out: output },
    params: {},
  });
  assert.deepEqual([...output.buffer], [2, 12]);
});

test('Resize executes canonical linear and nearest coordinate rules', () => {
  const linear = tensor([1, 1, 4, 1]);
  _cpuResize({
    opType: 'Resize',
    inputs: { input: tensor([1, 1, 2, 1], 'float32', [0, 10]) },
    outputs: { out: linear },
    params: { mode: 'linear', coordinate_transformation_mode: 'half_pixel' },
  });
  assert.deepEqual([...linear.buffer], [0, 2.5, 7.5, 10]);

  const nearest = tensor([1, 1, 3, 1]);
  _cpuResize({
    opType: 'ResizeNearest2D',
    inputs: { input: tensor([1, 1, 2, 1], 'float32', [0, 10]) },
    outputs: { out: nearest },
    params: {},
  });
  assert.deepEqual([...nearest.buffer], [0, 0, 10]);
});

test('UpsampleNearest2D duplicates every NHWC pixel into an exact 2x2 block', () => {
  const output = tensor([1, 2, 4, 1]);
  _cpuUpsample2x({
    opType: 'UpsampleNearest2D',
    inputs: { input: tensor([1, 1, 2, 1], 'float32', [3, 7]) },
    outputs: { out: output },
    params: {},
  });
  assert.deepEqual([...output.buffer], [3, 3, 7, 7, 3, 3, 7, 7]);
});
