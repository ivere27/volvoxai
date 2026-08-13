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
  assert.throws(
    () => _cpuWhere({
      id: 'mask_no_broadcast', opType: 'Mask',
      inputs: { mask: tensor([1, 4], 'int32', mask.buffer), a, b },
      outputs: { out },
    }),
    /must exactly match output/,
  );
});

test('CPU Where broadcasts condition and data inputs independently', () => {
  const condition = tensor([2, 1], 'int32', Int32Array.of(1, 0));
  const a = tensor([1, 3], 'float32', Float32Array.of(10, 20, 30));
  const b = tensor([2, 3], 'float32', Float32Array.of(-1, -2, -3, -4, -5, -6));
  const out = tensor([2, 3], 'float32', new Float32Array(6));

  _cpuWhere({ id: 'where_broadcast', opType: 'Where', inputs: { condition, a, b }, outputs: { out } });

  assert.deepEqual([...out.buffer], [10, 20, 30, -4, -5, -6]);
});

test('CPU Where supports rank-0 conditions and broadcasts both I32 data branches', () => {
  const condition = tensor([], 'float32', Float32Array.of(0));
  const a = tensor([2, 1], 'int32', Int32Array.of(10, 20));
  const b = tensor([1, 3], 'int32', Int32Array.of(-1, -2, -3));
  const out = tensor([2, 3], 'int32', new Int32Array(6));

  _cpuWhere({ id: 'where_scalar', opType: 'Where', inputs: { condition, a, b }, outputs: { out } });

  assert.deepEqual([...out.buffer], [-1, -2, -3, -1, -2, -3]);
});

test('CPU Where rejects incompatible broadcasts and mismatched physical storage', () => {
  const condition = tensor([1, 4], 'int32', Int32Array.of(1, 0, 1, 0));
  const a = tensor([2, 2], 'float32', Float32Array.of(1, 2, 3, 4));
  const b = tensor([2, 2], 'float32', Float32Array.of(5, 6, 7, 8));
  const out = tensor([2, 2], 'float32', new Float32Array(4));

  assert.throws(
    () => _cpuWhere({ id: 'where_bad_shape', opType: 'Where', inputs: { condition, a, b }, outputs: { out } }),
    /do not broadcast output/,
  );
  assert.throws(
    () => _cpuWhere({
      id: 'where_bad_storage', opType: 'Where',
      inputs: {
        condition: tensor([1], 'int32', Int32Array.of(1)),
        a: tensor([1], 'float32', Int32Array.of(1)),
        b: tensor([1], 'float32', Float32Array.of(2)),
      },
      outputs: { out: tensor([1], 'float32', new Float32Array(1)) },
    }),
    /matching physical storage/,
  );
});
