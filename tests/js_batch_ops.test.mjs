import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuConcat2 } from '../ts/ops/concat2.js';
import { _cpuAdd } from '../ts/ops/add.js';
import { _cpuConv1D } from '../ts/ops/conv1D.js';
import { _cpuCrossAttention } from '../ts/ops/crossAttention.js';
import { _cpuMeanHeight } from '../ts/ops/meanHeight.js';
import { _cpuProfileX } from '../ts/ops/profileX.js';
import { _cpuProfileY } from '../ts/ops/profileY.js';
import { _cpuSpatialSoftargmaxY } from '../ts/ops/spatialSoftargmaxY.js';
import { _cpuMul } from '../ts/ops/mul.js';
import { _cpuExpand } from '../ts/ops/expand.js';
import { _cpuInterp1D } from '../ts/ops/interp1D.js';

function tensor(shape, values = null) {
  const size = shape.reduce((count, dim) => count * dim, 1);
  return { shape, buffer: values ? new Float32Array(values) : new Float32Array(size) };
}

function close(actual, expected, tolerance = 1e-6) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

test('Conv1D and projected CrossAttention keep batch rows independent', () => {
  const convInput = tensor([2, 1, 3], [1, 2, 3, 10, 20, 30]);
  const convWeight = tensor([1, 1, 1], [2]);
  const convBias = tensor([1], [1]);
  const convOutput = tensor([2, 1, 3]);
  _cpuConv1D({
    inputs: { input: convInput, weight: convWeight, bias: convBias },
    outputs: { out: convOutput },
    params: { stride: 1, padding: 0 },
  });
  assert.deepEqual([...convOutput.buffer], [3, 5, 7, 21, 41, 61]);

  const q = tensor([2, 1, 1], [1, -1]);
  const kv = tensor([2, 1, 1], [3, 4]);
  const weight = tensor([3, 1], [1, 1, 2]);
  const output = tensor([2, 1, 1]);
  _cpuCrossAttention({
    inputs: { q, kv, weight }, outputs: { out: output }, params: { heads: 1 },
  });
  assert.deepEqual([...output.buffer], [6, 8]);
});

test('vision profile primitives process every NHWC batch row', () => {
  const input = tensor([2, 2, 1, 1], [1, 3, 10, 14]);

  const profileX = tensor([2, 2, 1]);
  _cpuProfileX({ inputs: { input }, outputs: { out: profileX } });
  assert.deepEqual([...profileX.buffer], [3, 2, 14, 12]);

  const profileY = tensor([2, 2, 2]);
  _cpuProfileY({ inputs: { input }, outputs: { out: profileY } });
  assert.deepEqual([...profileY.buffer], [1, 3, 1, 3, 10, 14, 10, 14]);

  const mean = tensor([2, 1, 1]);
  _cpuMeanHeight({ inputs: { input }, outputs: { out: mean } });
  assert.deepEqual([...mean.buffer], [2, 12]);

  const softargmax = tensor([2, 1, 1]);
  _cpuSpatialSoftargmaxY({ inputs: { input }, outputs: { out: softargmax } });
  const expected = (a, b) => {
    const ea = Math.exp(a - b);
    return (ea * 0.25 + 0.75) / (ea + 1);
  };
  close(softargmax.buffer[0], expected(1, 3));
  close(softargmax.buffer[1], expected(10, 14));
});

test('axis-aware Concat interleaves inputs independently for each batch', () => {
  const a = tensor([2, 1], [1, 10]);
  const b = tensor([2, 2], [2, 3, 20, 30]);
  const out = tensor([2, 3]);
  _cpuConcat2({
    id: 'batched_concat', inputs: { a, b }, outputs: { out }, params: { axis: 1 },
  });
  assert.deepEqual([...out.buffer], [1, 2, 3, 10, 20, 30]);
});

test('CPU Add and Mul apply right-aligned broadcasting to rank-3 channel affine tensors', () => {
  const activations = tensor([2, 2, 3], [
    1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12,
  ]);
  const channels = tensor([3], [10, 20, 30]);
  const added = tensor([2, 2, 3]);
  _cpuAdd({ inputs: { a: activations, b: channels }, outputs: { out: added } });
  assert.deepEqual([...added.buffer], [
    11, 22, 33, 14, 25, 36,
    17, 28, 39, 20, 31, 42,
  ]);

  const scale = tensor([1, 1, 3], [2, 3, 4]);
  const multiplied = tensor([2, 2, 3]);
  _cpuMul({ inputs: { a: activations, b: scale }, outputs: { out: multiplied } });
  assert.deepEqual([...multiplied.buffer], [
    2, 6, 12, 8, 15, 24,
    14, 24, 36, 20, 33, 48,
  ]);
});

test('CPU Add applies fused ReLU and ReLU6 after broadcasting', () => {
  const activations = tensor([2, 2], [-4, 1, 5, 10]);
  const channels = tensor([2], [1, -3]);

  const plain = tensor([2, 2]);
  _cpuAdd({
    inputs: { a: activations, b: channels }, outputs: { out: plain }, params: { relu: 0 },
  });
  assert.deepEqual([...plain.buffer], [-3, -2, 6, 7]);

  const relu = tensor([2, 2]);
  _cpuAdd({
    inputs: { a: activations, b: channels }, outputs: { out: relu }, params: { relu: 1 },
  });
  assert.deepEqual([...relu.buffer], [0, 0, 6, 7]);

  const relu6 = tensor([2, 2]);
  _cpuAdd({
    inputs: { a: activations, b: channels }, outputs: { out: relu6 }, params: { relu: 2 },
  });
  assert.deepEqual([...relu6.buffer], [0, 0, 6, 6]);

  assert.throws(() => _cpuAdd({
    inputs: { a: activations, b: channels }, outputs: { out: tensor([2, 2]) },
    params: { relu: 3 },
  }), /supports relu values 0.*1.*2/);
});

test('CPU Expand applies standard right-aligned broadcasting beyond rank four', () => {
  const input = tensor([1, 2, 1, 2, 1], [1, 2, 3, 4]);
  const out = tensor([3, 2, 2, 2, 2]);
  _cpuExpand({ inputs: { input }, outputs: { out } });
  for (let outer = 0; outer < 3; outer++) {
    for (let channel = 0; channel < 2; channel++) {
      for (let width = 0; width < 2; width++) {
        for (let lane = 0; lane < 2; lane++) {
          const index = ((((outer * 2 + channel) * 2 + 0) * 2 + width) * 2 + lane);
          assert.equal(out.buffer[index], input.buffer[channel * 2 + width]);
        }
      }
    }
  }
  assert.throws(() => _cpuExpand({
    inputs: { input: tensor([2, 2], [1, 2, 3, 4]) }, outputs: { out: tensor([2, 3]) },
  }), /cannot broadcast/);
});

test('CPU Interp1D keeps NCL batch rows independent', () => {
  const input=tensor([2,1,2],[0,2,10,14]),out=tensor([2,1,4]);
  _cpuInterp1D({inputs:{input},outputs:{out},params:{size:4}});
  assert.deepEqual([...out.buffer],[0,.5,1.5,2,10,11,13,14]);
});

test('CPU Add rejects incompatible broadcast shapes instead of leaving stale output', () => {
  const a = tensor([2, 2, 3]);
  const b = tensor([2, 2]);
  const out = tensor([2, 2, 3]);
  assert.throws(
    () => _cpuAdd({ inputs: { a, b }, outputs: { out } }),
    /do not broadcast/,
  );
});
