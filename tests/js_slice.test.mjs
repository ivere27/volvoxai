import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuSlice } from '../ts/ops/slice.js';

function tensor(shape, values = null) {
  const elements = shape.reduce((count, dimension) => count * dimension, 1);
  return {
    shape,
    dtype: 'float32',
    buffer: values ? new Float32Array(values) : new Float32Array(elements),
  };
}

test('CPU Slice maps canonical rank-five normalized axes and starts', () => {
  const input = tensor([2, 3, 4, 5, 6], Array.from({ length: 720 }, (_, index) => index));
  const out = tensor([1, 3, 2, 5, 3]);
  _cpuSlice({
    id: 'slice_rank_five',
    inputs: { input },
    outputs: { out },
    params: {
      axes: [-5, -3, -1], starts: [-1, 1, -5], ends: [2, 4, 6], steps: [1, 2, 2],
    },
  });

  const expected = [];
  for (let axis1 = 0; axis1 < 3; axis1++) {
    for (const axis2 of [1, 3]) {
      for (let axis3 = 0; axis3 < 5; axis3++) {
        for (const axis4 of [1, 3, 5]) {
          expected.push(((((1 * 3 + axis1) * 4 + axis2) * 5 + axis3) * 6 + axis4));
        }
      }
    }
  }
  assert.deepEqual([...out.buffer], expected);
});

test('CPU Slice accepts the rank-eight upper bound', () => {
  const input = tensor([1, 1, 1, 1, 1, 1, 2, 3], [0, 1, 2, 3, 4, 5]);
  const out = tensor([1, 1, 1, 1, 1, 1, 1, 2]);
  _cpuSlice({
    id: 'slice_rank_eight',
    inputs: { input },
    outputs: { out },
    params: { axes: [-2, -1], starts: [-1, -3], ends: [2, 3], steps: [1, 2] },
  });

  assert.deepEqual([...out.buffer], [3, 5]);
});

test('CPU Slice rejects non-positive steps, impossible output selections, and rank nine', () => {
  const input = tensor([4], [0, 1, 2, 3]);
  const out = tensor([2]);
  assert.throws(() => _cpuSlice({
    id: 'slice_step', inputs: { input }, outputs: { out },
    params: { axes: [0], starts: [0], ends: [4], steps: [0] },
  }), /positive safe integers/);
  assert.throws(() => _cpuSlice({
    id: 'slice_bounds', inputs: { input }, outputs: { out },
    params: { axes: [0], starts: [3], ends: [4], steps: [1] },
  }), /output shape .* does not match/);
  const rankNineInput = tensor(new Array(9).fill(1), [1]);
  const rankNineOut = tensor(new Array(9).fill(1));
  assert.throws(() => _cpuSlice({
    id: 'slice_rank_nine', inputs: { input: rankNineInput }, outputs: { out: rankNineOut }, params: {},
  }), /input rank must be in \[1, 8\]/);
});
