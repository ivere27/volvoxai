import test from 'node:test';
import assert from 'node:assert/strict';

import { _cpuWhere } from '../ts/ops/where.js';

function tensor(shape, dtype, buffer) {
  return { shape, dtype, buffer };
}

test('CPU Mask accepts the mask alias and preserves I32 nonzero semantics', () => {
  const mask = tensor([4], 'int32', Int32Array.of(0, 1, -7, -2147483648));
  const a = tensor([4], 'float32', Float32Array.of(10, 20, 30, 40));
  const b = tensor([4], 'float32', Float32Array.of(-1, -2, -3, -4));
  const out = tensor([4], 'float32', new Float32Array(4));

  _cpuWhere({ id: 'mask_i32', opType: 'Mask', inputs: { mask, a, b }, outputs: { out } });

  assert.deepEqual([...out.buffer], [-1, 20, 30, 40]);
});

test('CPU Where requires canonical exact-shape F32 operands and F32/I32 conditions', () => {
  const condition = tensor([1, 4], 'int32', Int32Array.of(1, 0, 1, 0));
  const a = tensor([2, 2], 'float32', Float32Array.of(1, 2, 3, 4));
  const b = tensor([2, 2], 'float32', Float32Array.of(5, 6, 7, 8));
  const out = tensor([2, 2], 'float32', new Float32Array(4));

  assert.throws(
    () => _cpuWhere({ id: 'where_bad_shape', opType: 'Where', inputs: { condition, a, b }, outputs: { out } }),
    /exact-shape F32 operands\/output and an F32 or I32 condition/,
  );
});
