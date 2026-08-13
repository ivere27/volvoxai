import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuSub } from '../ts/ops/sub.js';
import { _cpuDiv } from '../ts/ops/div.js';
import { _cpuComparison } from '../ts/ops/comparison.js';
import { _cpuReduceSum } from '../ts/ops/reduceSum.js';
import { _cpuReduceMean } from '../ts/ops/reduceMean.js';
import { _cpuArgMax } from '../ts/ops/argMax.js';
import { _cpuTranspose } from '../ts/ops/transpose.js';
import { _cpuReshape } from '../ts/ops/reshape.js';
import { _cpuExpand } from '../ts/ops/expand.js';
import { _cpuConcat2 } from '../ts/ops/concat2.js';
import { _cpuSplit } from '../ts/ops/split.js';
import { _cpuSlice } from '../ts/ops/slice.js';
import { _cpuPad } from '../ts/ops/pad.js';
import { _cpuGather } from '../ts/ops/gather.js';
import { _cpuGatherElements } from '../ts/ops/gatherElements.js';

const constructors = {
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

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
    ...(quantization === undefined ? {} : { quantization }),
  };
}

function perAxis(axis, scales, zeroPoints = new Array(scales.length).fill(0)) {
  return { scheme: 'per_axis', axis, scales, zero_points: zeroPoints };
}

test('CPU Sub and Div execute rank-aligned and scalar F32 broadcasts and reject noncanonical storage', () => {
  const left = tensor([2, 1], 'float32', [10, 20]);
  const right = tensor([3], 'float32', [1, 2, 4]);
  const difference = tensor([2, 3]);
  _cpuSub({ opType: 'Sub', inputs: { a: left, b: right }, outputs: { out: difference }, params: {} });
  assert.deepEqual([...difference.buffer], [9, 8, 6, 19, 18, 16]);

  const numerator = tensor([], 'float32', [12]);
  const denominator = tensor([2, 2], 'float32', [2, 3, 4, 6]);
  const quotient = tensor([2, 2]);
  _cpuDiv({ opType: 'Div', inputs: { a: numerator, b: denominator }, outputs: { out: quotient } });
  assert.deepEqual([...quotient.buffer], [6, 4, 3, 2]);

  assert.throws(() => _cpuSub({
    opType: 'Sub',
    inputs: { a: tensor([2], 'int32', [1, 2]), b: tensor([2], 'int32', [1, 1]) },
    outputs: { out: tensor([2], 'int32') },
    params: {},
  }), /supported dtypes.*float32/);
});

test('CPU Equal and GreaterOrEqual execute I32 broadcasts including rank zero and reject bad shapes', () => {
  const equal = tensor([2, 3], 'int32');
  _cpuComparison({
    opType: 'Equal', inputs: {
      a: tensor([2, 1], 'int32', [2, 3]),
      b: tensor([3], 'int32', [2, 3, 4]),
    }, outputs: { out: equal }, params: {},
  });
  assert.deepEqual([...equal.buffer], [1, 0, 0, 0, 1, 0]);

  const greater = tensor([2], 'int32');
  _cpuComparison({
    opType: 'GreaterOrEqual',
    inputs: { a: tensor([], 'int32', [5]), b: tensor([2], 'int32', [5, 7]) },
    outputs: { out: greater },
  });
  assert.deepEqual([...greater.buffer], [1, 0]);

  assert.throws(() => _cpuComparison({
    opType: 'Equal',
    inputs: { a: tensor([2, 1], 'int32'), b: tensor([3], 'int32') },
    outputs: { out: tensor([2, 2], 'int32') }, params: {},
  }), /output shape must be/);
});

test('CPU reductions honor last-axis keepdims and ArgMax emits first-tie I32 indices', () => {
  const sum = tensor([2]);
  _cpuReduceSum({
    opType: 'ReduceSum', inputs: { input: tensor([2, 3], 'float32', [1, 2, 3, 4, 5, 6]) },
    outputs: { out: sum }, params: { axis: -1, keepdims: false },
  });
  assert.deepEqual([...sum.buffer], [6, 15]);

  const mean = tensor([2, 2, 1]);
  _cpuReduceMean({
    opType: 'ReduceMean',
    inputs: { input: tensor([2, 2, 2], 'float32', [1, 3, 2, 6, 10, 14, 8, 12]) },
    outputs: { out: mean }, params: {},
  });
  assert.deepEqual([...mean.buffer], [2, 4, 12, 10]);

  const firstTie = tensor([2], 'int32');
  _cpuArgMax({
    opType: 'ArgMax',
    inputs: { input: tensor([2, 3], 'float32', [5, 5, 1, -1, 3, 2]) },
    outputs: { out: firstTie }, params: { axis: 1, keepdims: false, select_last_index: 0 },
  });
  assert.deepEqual([...firstTie.buffer], [0, 1]);

  const middleAxis = tensor([2, 1, 2], 'int32');
  _cpuArgMax({
    opType: 'ArgMax',
    inputs: { input: tensor([2, 2, 2], 'float32', [1, 9, 3, 4, 8, 1, 2, 7]) },
    outputs: { out: middleAxis }, params: { axis: -2 },
  });
  assert.deepEqual([...middleAxis.buffer], [1, 0, 0, 1]);

  assert.throws(() => _cpuReduceSum({
    opType: 'ReduceSum', inputs: { input: tensor([2, 3]) },
    outputs: { out: tensor([3]) }, params: { axis: 0, keepdims: false },
  }), /last axis/);
  assert.throws(() => _cpuArgMax({
    opType: 'ArgMax', inputs: { input: tensor([2, 3]) },
    outputs: { out: tensor([2]) }, params: { axis: 1, keepdims: false },
  }), /supported dtypes.*int32/);
});

test('CPU Transpose and reshape-family kernels preserve bytes with canonical per-axis remapping', () => {
  const transposed = tensor([3, 2]);
  _cpuTranspose({
    opType: 'Transpose', inputs: { input: tensor([2, 3], 'float32', [1, 2, 3, 4, 5, 6]) },
    outputs: { out: transposed }, params: { perm: [1, 0] },
  });
  assert.deepEqual([...transposed.buffer], [1, 4, 2, 5, 3, 6]);

  const transposeInputQuantization = perAxis(1, [0.5, 0.25, 0.125]);
  const transposeOutputQuantization = perAxis(2, [0.5, 0.25, 0.125]);
  const byteTranspose = tensor([1, 2, 3], 'int8', undefined, transposeOutputQuantization);
  _cpuTranspose({
    opType: 'Transpose',
    inputs: { input: tensor([2, 3, 1], 'int8', [1, 2, 3, 4, 5, 6], transposeInputQuantization) },
    outputs: { out: byteTranspose }, params: { perm: [2, 0, 1] },
  });
  assert.deepEqual([...byteTranspose.buffer], [1, 2, 3, 4, 5, 6]);

  const flattened = tensor([2, 12]);
  _cpuReshape({
    opType: 'Flatten', inputs: { input: tensor([2, 3, 4], 'float32', [...Array(24).keys()]) },
    outputs: { out: flattened }, params: { axis: 1 },
  });
  assert.equal(flattened.buffer[23], 23);

  const squeezed = tensor([2, 3], 'int32');
  _cpuReshape({
    opType: 'Squeeze', inputs: { input: tensor([2, 1, 3], 'int32', [1, 2, 3, 4, 5, 6]) },
    outputs: { out: squeezed }, params: { axes: [-2] },
  });
  assert.deepEqual([...squeezed.buffer], [1, 2, 3, 4, 5, 6]);

  const unsqueezedQuantization = perAxis(2, [0.5, 0.25, 0.125]);
  const unsqueezed = tensor([1, 2, 3], 'uint8', undefined, unsqueezedQuantization);
  _cpuReshape({
    opType: 'Unsqueeze',
    inputs: { input: tensor([2, 3], 'uint8', [1, 2, 3, 4, 5, 6], perAxis(1, [0.5, 0.25, 0.125])) },
    outputs: { out: unsqueezed }, params: { axes: [0] },
  });

  const reshaped = tensor([3, 2], 'int32');
  _cpuReshape({
    opType: 'Reshape', inputs: { input: tensor([2, 3], 'int32', [1, 2, 3, 4, 5, 6]) },
    outputs: { out: reshaped }, params: { shape: [3, 2] },
  });
  assert.deepEqual([...reshaped.buffer], [1, 2, 3, 4, 5, 6]);

  assert.throws(() => _cpuReshape({
    opType: 'Squeeze', inputs: { input: tensor([2, 3]) },
    outputs: { out: tensor([2]) }, params: { axes: [1] },
  }), /must be 1/);
  assert.throws(() => _cpuTranspose({
    opType: 'Transpose',
    inputs: { input: tensor([2, 3, 1], 'int8', undefined, transposeInputQuantization) },
    outputs: { out: tensor([1, 2, 3], 'int8', undefined, perAxis(1, [0.5, 0.25])) },
    params: { perm: [2, 0, 1] },
  }), /quantization metadata/);
});

test('CPU Expand broadcasts F32 and remaps a shifted byte quantization axis', () => {
  const expanded = tensor([3, 2]);
  _cpuExpand({
    opType: 'Expand', inputs: { input: tensor([1, 2], 'float32', [4, 7]) },
    outputs: { out: expanded }, params: { shape: [3, 2] },
  });
  assert.deepEqual([...expanded.buffer], [4, 7, 4, 7, 4, 7]);

  const byteInput = tensor(
    [2, 3], 'int8', [1, 2, 3, 4, 5, 6], perAxis(1, [0.5, 0.25, 0.125]),
  );
  const byteOutput = tensor(
    [4, 2, 3], 'int8', undefined, perAxis(2, [0.5, 0.25, 0.125]),
  );
  _cpuExpand({
    opType: 'Expand', inputs: { input: byteInput }, outputs: { out: byteOutput },
    params: { shape: [4, 2, 3] },
  });
  assert.deepEqual([...byteOutput.buffer.slice(0, 12)], [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);

  assert.throws(() => _cpuExpand({
    opType: 'Expand',
    inputs: { input: tensor([2, 1], 'int8', [1, 2], perAxis(1, [0.5])) },
    outputs: { out: tensor([2, 3], 'int8', undefined, perAxis(1, [0.5, 0.5, 0.5])) },
    params: { shape: [2, 3] },
  }), /must not expand the per-axis/);
});

test('CPU Concat and Split handle variadic/uneven byte slices and validate affine metadata', () => {
  const joined = tensor([2, 3]);
  _cpuConcat2({
    opType: 'Concat',
    inputs: {
      input0: tensor([2, 1], 'float32', [1, 10]),
      input1: tensor([2, 2], 'float32', [2, 3, 20, 30]),
    }, outputs: { out: joined }, params: { axis: 1 },
  });
  assert.deepEqual([...joined.buffer], [1, 2, 3, 10, 20, 30]);

  const quantizedJoin = tensor(
    [1, 3], 'int8', undefined, perAxis(1, [0.5, 0.25, 0.125]),
  );
  _cpuConcat2({
    opType: 'Concat',
    inputs: {
      input0: tensor([1, 2], 'int8', [4, 5], perAxis(1, [0.5, 0.25])),
      input1: tensor([1, 1], 'int8', [6], perAxis(1, [0.125])),
    }, outputs: { out: quantizedJoin }, params: { axis: -1 },
  });
  assert.deepEqual([...quantizedJoin.buffer], [4, 5, 6]);

  const splitInput = tensor([2, 5], 'float32', [...Array(10).keys()]);
  const first = tensor([2, 2]);
  const second = tensor([2, 3]);
  _cpuSplit({
    opType: 'Split', inputs: { input: splitInput }, outputs: { out0: first, out1: second },
    params: { axis: 1, split: [2, 3] },
  });
  assert.deepEqual([...first.buffer], [0, 1, 5, 6]);
  assert.deepEqual([...second.buffer], [2, 3, 4, 7, 8, 9]);

  const splitQuantization = perAxis(1, [0.5, 0.25, 0.125, 0.0625]);
  const byteFirst = tensor([1, 1], 'uint8', undefined, perAxis(1, [0.5]));
  const byteSecond = tensor([1, 3], 'uint8', undefined, perAxis(1, [0.25, 0.125, 0.0625]));
  _cpuSplit({
    opType: 'Split',
    inputs: { input: tensor([1, 4], 'uint8', [1, 2, 3, 4], splitQuantization) },
    outputs: { out0: byteFirst, out1: byteSecond },
    params: { axis: 1, split: [1, 3] },
  });
  assert.deepEqual([...byteSecond.buffer], [2, 3, 4]);

  assert.throws(() => _cpuSplit({
    opType: 'Split', inputs: { input: splitInput }, outputs: { out0: first, out1: second },
    params: { axis: 1, split: [1, 4], num_outputs: 2 },
  }), /not both/);
  assert.throws(() => _cpuConcat2({
    opType: 'Concat',
    inputs: {
      input0: tensor([1, 2], 'int8', undefined, perAxis(1, [0.5, 0.25])),
      input1: tensor([1, 1], 'int8', undefined, perAxis(1, [0.125])),
    }, outputs: { out: tensor([1, 3], 'int8', undefined, perAxis(1, [1, 1, 1])) },
    params: { axis: 1 },
  }), /quantization metadata/);
});

test('CPU Slice and Pad implement exact rank-general geometry with quantized metadata transforms', () => {
  const sliced = tensor([3]);
  _cpuSlice({
    opType: 'Slice', inputs: { input: tensor([6], 'float32', [0, 1, 2, 3, 4, 5]) },
    outputs: { out: sliced }, params: { starts: [1], ends: [6], axes: [0], steps: [2] },
  });
  assert.deepEqual([...sliced.buffer], [1, 3, 5]);

  const sliceQuantization = perAxis(1, [0.5, 0.25, 0.125, 0.0625]);
  const byteSlice = tensor([2, 3], 'int8', undefined, perAxis(1, [0.25, 0.125, 0.0625]));
  _cpuSlice({
    opType: 'Slice',
    inputs: { input: tensor([2, 4], 'int8', [1, 2, 3, 4, 5, 6, 7, 8], sliceQuantization) },
    outputs: { out: byteSlice }, params: { starts: [1], ends: [4], axes: [1] },
  });
  assert.deepEqual([...byteSlice.buffer], [2, 3, 4, 6, 7, 8]);

  const padded = tensor([6], 'int32');
  _cpuPad({
    opType: 'Pad', inputs: { input: tensor([3], 'int32', [2, 3, 4]) },
    outputs: { out: padded }, params: { pads: [1, 2], value: -1 },
  });
  assert.deepEqual([...padded.buffer], [-1, 2, 3, 4, -1, -1]);

  const padQuantization = perAxis(2, [0.5, 0.25]);
  const bytePad = tensor([2, 3, 2], 'uint8', undefined, padQuantization);
  _cpuPad({
    opType: 'Pad',
    inputs: { input: tensor([1, 2, 2], 'uint8', [1, 2, 3, 4], padQuantization) },
    outputs: { out: bytePad }, params: { pads: [1, 0, 0, 0, 1, 0], value: 0 },
  });
  assert.deepEqual([...bytePad.buffer.slice(6, 10)], [1, 2, 3, 4]);

  assert.throws(() => _cpuSlice({
    opType: 'Slice', inputs: { input: tensor([6]) }, outputs: { out: tensor([2]) },
    params: { starts: [1], ends: [6], axes: [0], steps: [2] },
  }), /output shape/);
  assert.throws(() => _cpuPad({
    opType: 'Pad',
    inputs: { input: tensor([1, 2], 'int8', undefined, perAxis(1, [0.5, 0.25])) },
    outputs: { out: tensor([1, 3], 'int8', undefined, perAxis(1, [0.5, 0.25, 0.125])) },
    params: { pads: [0, 0, 0, 1] },
  }), /must not extend a per-axis/);
});

test('CPU Gather and GatherElements validate indices before writes and remap safe byte metadata', () => {
  const gathered = tensor([2, 2]);
  _cpuGather({
    opType: 'Gather',
    inputs: {
      input: tensor([2, 3], 'float32', [1, 2, 3, 4, 5, 6]),
      indices: tensor([2], 'int32', [2, 0]),
    }, outputs: { out: gathered }, params: { axis: 1 },
  });
  assert.deepEqual([...gathered.buffer], [3, 1, 6, 4]);

  const bankGather = tensor([1, 2]);
  _cpuGather({
    opType: 'Gather',
    inputs: {
      input: tensor([1, 2], 'float32', [7, 8]),
      indices: tensor([1], 'int32', [-1]),
    }, outputs: { out: bankGather }, params: { axis: 0 },
    residentSlots: [3], residentSlotDomain: 4,
  });
  assert.deepEqual([...bankGather.buffer], [7, 8],
    'negative bank indices normalize against the complete global domain');
  assert.throws(() => _cpuGather({
    opType: 'Gather',
    inputs: {
      input: tensor([1, 2], 'float32', [7, 8]),
      indices: tensor([1], 'int32', [2]),
    }, outputs: { out: tensor([1, 2]) }, params: { axis: 0 },
    residentSlots: [3], residentSlotDomain: 4,
  }), /slot 2 is not resident/);

  const gatherInputQuantization = perAxis(2, [0.5, 0.25]);
  const gatherOutputQuantization = perAxis(1, [0.5, 0.25]);
  const scalarGather = tensor([2, 2], 'int8', undefined, gatherOutputQuantization);
  _cpuGather({
    opType: 'Gather',
    inputs: {
      input: tensor([2, 3, 2], 'int8', [...Array(12).keys()], gatherInputQuantization),
      indices: tensor([], 'int32', [1]),
    }, outputs: { out: scalarGather }, params: { axis: 1 },
  });
  assert.deepEqual([...scalarGather.buffer], [2, 3, 8, 9]);

  const elementGather = tensor([2, 2], 'int32');
  _cpuGatherElements({
    opType: 'GatherElements',
    inputs: {
      input: tensor([2, 3], 'int32', [10, 11, 12, 20, 21, 22]),
      indices: tensor([2, 2], 'int32', [0, 2, 1, 0]),
    }, outputs: { out: elementGather }, params: { axis: 1 },
  });
  assert.deepEqual([...elementGather.buffer], [10, 12, 21, 20]);

  const elementQuantization = perAxis(0, [0.5, 0.25, 0.125]);
  const narrowedQuantization = perAxis(0, [0.5, 0.25]);
  const byteElements = tensor([2, 2, 2], 'uint8', undefined, narrowedQuantization);
  _cpuGatherElements({
    opType: 'GatherElements',
    inputs: {
      input: tensor([3, 2, 2], 'uint8', [...Array(12).keys()], elementQuantization),
      indices: tensor([2, 2, 2], 'int32', [0, 1, 1, 0, 1, 0, 0, 1]),
    }, outputs: { out: byteElements }, params: { axis: 2 },
  });
  assert.deepEqual(byteElements.quantization.scales, [0.5, 0.25]);

  const untouched = tensor([2], 'float32', [99, 99]);
  assert.throws(() => _cpuGather({
    opType: 'Gather',
    inputs: { input: tensor([2], 'float32', [1, 2]), indices: tensor([2], 'int32', [0, 2]) },
    outputs: { out: untouched }, params: { axis: 0 },
  }), /outside axis extent/);
  assert.deepEqual([...untouched.buffer], [99, 99]);
  assert.throws(() => _cpuGatherElements({
    opType: 'GatherElements',
    inputs: {
      input: tensor([2, 2], 'float32'),
      indices: tensor([3, 2], 'int32'),
    }, outputs: { out: tensor([3, 2]) }, params: { axis: 1 },
  }), /exceeds the corresponding input extent/);
});
