import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuCrossSDPA } from '../ts/ops/crossSDPA.js';
import { _cpuRoPE } from '../ts/ops/roPE.js';
import { _cpuSDPA } from '../ts/ops/sDPA.js';

function elements(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function f32(shape, values = undefined) {
  const count = elements(shape);
  return {
    shape,
    dtype: 'float32',
    buffer: values === undefined ? new Float32Array(count) : Float32Array.from(values),
  };
}

function i32(shape, values) {
  return { shape, dtype: 'int32', buffer: Int32Array.from(values) };
}

function close(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `index ${index}: ${actual[index]} != ${expected[index]}`);
  }
}

test('CPU SDPA executes rank-2 and rank-3 shapes and rejects invalid packed geometry before writes', () => {
  const rank2Output = f32([2, 2]);
  _cpuSDPA({
    id: 'sdpa-rank2',
    inputs: {
      qkv: f32([2, 6], [
        0, 0, 0, 0, 1, 2,
        0, 0, 0, 0, 3, 4,
      ]),
    },
    outputs: { out: rank2Output },
    params: { heads: 1, causal: false },
  });
  close(rank2Output.buffer, [2, 3, 2, 3]);

  const rank3Output = f32([2, 2, 2]);
  _cpuSDPA({
    id: 'sdpa-rank3',
    inputs: {
      qkv: f32([2, 2, 6], [
        0, 0, 0, 0, 1, 2,
        0, 0, 0, 0, 3, 4,
        0, 0, 0, 0, 10, 20,
        0, 0, 0, 0, 30, 40,
      ]),
    },
    outputs: { out: rank3Output },
    params: { heads: 1, causal: false },
  });
  close(rank3Output.buffer, [2, 3, 2, 3, 20, 30, 20, 30]);

  const invalidOutput = f32([2, 2], [91, 92, 93, 94]);
  assert.throws(
    () => _cpuSDPA({
      id: 'sdpa-invalid',
      inputs: { qkv: f32([2, 7]) },
      outputs: { out: invalidOutput },
      params: { heads: 1, causal: false },
    }),
    /compatible.*input.*output/,
  );
  assert.deepEqual([...invalidOutput.buffer], [91, 92, 93, 94]);
});

test('CPU CrossSDPA executes rank-2 and rank-3 shapes and rejects mismatched K/V before writes', () => {
  const rank2Output = f32([2, 2]);
  _cpuCrossSDPA({
    id: 'cross-rank2',
    inputs: {
      q: f32([2, 2]),
      k: f32([3, 2]),
      v: f32([3, 2], [1, 2, 3, 4, 5, 6]),
    },
    outputs: { out: rank2Output },
    params: { heads: 1, causal: false },
  });
  close(rank2Output.buffer, [3, 4, 3, 4]);

  const rank3Output = f32([2, 2, 2]);
  _cpuCrossSDPA({
    id: 'cross-rank3',
    inputs: {
      q: f32([2, 2, 2]),
      k: f32([2, 2, 2]),
      v: f32([2, 2, 2], [1, 2, 3, 4, 10, 20, 30, 40]),
    },
    outputs: { out: rank3Output },
    params: { heads: 1, causal: false },
  });
  close(rank3Output.buffer, [2, 3, 2, 3, 20, 30, 20, 30]);

  const invalidOutput = f32([2, 2], [81, 82, 83, 84]);
  assert.throws(
    () => _cpuCrossSDPA({
      id: 'cross-invalid',
      inputs: { q: f32([2, 2]), k: f32([3, 2]), v: f32([2, 2]) },
      outputs: { out: invalidOutput },
      params: { heads: 1, causal: false },
    }),
    /compatible.*Q\/K\/V\/output/,
  );
  assert.deepEqual([...invalidOutput.buffer], [81, 82, 83, 84]);
});

test('CPU ambiguous rank-2 attention masks use BK precedence when B equals Q', () => {
  const output = f32([2, 2, 2]);
  _cpuCrossSDPA({
    id: 'cross-bk-precedence',
    inputs: {
      q: f32([2, 2, 2]),
      k: f32([2, 2, 2]),
      v: f32([2, 2, 2], [1, 0, 3, 0, 10, 0, 30, 0]),
      mask: i32([2, 2], [1, 0, 0, 1]),
    },
    outputs: { out: output },
    params: { heads: 1, causal: false },
  });
  close(output.buffer, [1, 0, 1, 0, 30, 0, 30, 0]);
});

test('CPU RoPE executes rank-2 and rank-3 shapes and rejects invalid rotary width before writes', () => {
  const rank2Input = f32([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
  const rank2Output = f32([2, 4]);
  _cpuRoPE({
    id: 'rope-rank2',
    inputs: { input: rank2Input },
    outputs: { out: rank2Output },
    params: { rotary_dim: 4, position_offset: 0, interleaved: false },
  });
  close(rank2Output.buffer.subarray(0, 4), rank2Input.buffer.subarray(0, 4));
  assert.ok(rank2Output.buffer.every(Number.isFinite));

  const rank3Input = f32(
    [2, 2, 6],
    Array.from({ length: 24 }, (_, index) => index * 0.25 - 2),
  );
  const rank3Output = f32([2, 2, 6]);
  _cpuRoPE({
    id: 'rope-rank3',
    inputs: {
      input: rank3Input,
      position_ids: i32([2, 2], [0, 2, 1, 3]),
    },
    outputs: { out: rank3Output },
    params: { rotary_dim: 4, theta: 10, interleaved: true },
  });
  assert.ok(rank3Output.buffer.every(Number.isFinite));
  for (let row = 0; row < 4; row++) {
    assert.equal(rank3Output.buffer[row * 6 + 4], rank3Input.buffer[row * 6 + 4]);
    assert.equal(rank3Output.buffer[row * 6 + 5], rank3Input.buffer[row * 6 + 5]);
  }

  const invalidOutput = f32([2, 4], new Array(8).fill(71));
  assert.throws(
    () => _cpuRoPE({
      id: 'rope-invalid',
      inputs: { input: rank2Input },
      outputs: { out: invalidOutput },
      params: { rotary_dim: 3 },
    }),
    /valid rotary parameters/,
  );
  assert.deepEqual([...invalidOutput.buffer], new Array(8).fill(71));
});
