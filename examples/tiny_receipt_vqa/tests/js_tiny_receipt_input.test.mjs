import test from 'node:test';
import assert from 'node:assert/strict';

import * as tinyReceiptInput from '../TinyReceiptInput.js';

const { preprocessTinyReceiptImage } = tinyReceiptInput;

test('TinyReceipt input utility exports no whole-model session surface', () => {
  assert.deepEqual(Object.keys(tinyReceiptInput), ['preprocessTinyReceiptImage']);
});

test('TinyReceipt preprocessing creates evaluator-compatible grayscale normalized F32', async () => {
  const output = await preprocessTinyReceiptImage({
    data: Uint8Array.of(0, 255, 0, 255),
    width: 2,
    height: 2,
    channels: 1,
  }, { width: 1, height: 1 });
  assert.ok(output instanceof Float32Array);
  assert.equal(output.length, 1);
  assert.ok(Math.abs(output[0] - ((128 / 255) * 2 - 1)) < 1e-7);

  const rgb = await preprocessTinyReceiptImage({
    data: Uint8Array.of(255, 0, 0, 0, 0, 255),
    width: 2,
    height: 1,
    channels: 3,
  }, { width: 1, height: 1 });
  assert.ok(Math.abs(rgb[0] - ((53 / 255) * 2 - 1)) < 1e-7);
});

test('TinyReceipt normalization rounds at the canonical F32 operation boundaries', async () => {
  const output = await preprocessTinyReceiptImage({
    data: Uint8Array.of(64, 128, 191),
    width: 3,
    height: 1,
    channels: 1,
  }, { width: 3, height: 1 });
  const expected = [64, 128, 191].map((value) => {
    const unit = Math.fround(Math.fround(value) / Math.fround(255));
    return Math.fround(Math.fround(unit * Math.fround(2)) - Math.fround(1));
  });
  assert.deepEqual([...output], expected);
});
