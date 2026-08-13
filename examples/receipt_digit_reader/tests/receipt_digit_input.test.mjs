import test from 'node:test';
import assert from 'node:assert/strict';

import {
  luminance,
  normalizeReceiptPixels,
  prepareReceiptImage,
  resizeBilinear,
  toGrayscalePlane,
} from '../ReceiptDigitInput.js';

test('grayscale reduction matches BT.601 luma and ignores alpha', () => {
  const rgb = toGrayscalePlane({
    data: Uint8Array.from([255, 0, 0, 0, 255, 0]), width: 2, height: 1, channels: 3,
  });
  // The plane is F32 storage, so compare against the rounded luma rather than
  // the F64 reference value.
  assert.equal(rgb[0], Math.fround(luminance(255, 0, 0)));
  assert.equal(rgb[1], Math.fround(luminance(0, 255, 0)));

  // An opaque and a fully transparent pixel with identical colour must reduce
  // identically: compositing would change values the model was trained on.
  const rgba = toGrayscalePlane({
    data: Uint8Array.from([10, 20, 30, 255, 10, 20, 30, 0]), width: 2, height: 1, channels: 4,
  });
  assert.equal(rgba[0], rgba[1]);
});

test('grayscale reduction rejects unsupported channel counts', () => {
  assert.throws(() => toGrayscalePlane({
    data: Uint8Array.from([1, 2]), width: 1, height: 1, channels: 2,
  }), /1, 3, or 4/);
  assert.throws(() => toGrayscalePlane({ data: Uint8Array.from([1]), width: 0, height: 1 }),
    /width, height/);
});

test('bilinear resize uses half-pixel centers', () => {
  // Downscaling 4->2 with half-pixel centers samples at source x = 0.5 and 2.5,
  // averaging neighbours. The corner convention would return the endpoints
  // 0 and 3 instead, shifting thin strokes by up to half a pixel.
  const plane = Float32Array.from([0, 1, 2, 3]);
  const resized = resizeBilinear(plane, 4, 1, 2, 1);
  assert.deepEqual([...resized], [0.5, 2.5]);
});

test('bilinear resize is exact for an identity target and clamps at edges', () => {
  const plane = Float32Array.from([1, 2, 3, 4]);
  assert.deepEqual([...resizeBilinear(plane, 2, 2, 2, 2)], [1, 2, 3, 4]);
  // Upscaling must stay inside the source range rather than extrapolating.
  const upscaled = resizeBilinear(Float32Array.from([0, 10]), 2, 1, 4, 1);
  assert.ok(Math.min(...upscaled) >= 0 && Math.max(...upscaled) <= 10);
});

test('normalization applies the producer symmetric transform', () => {
  const values = normalizeReceiptPixels(Float32Array.from([0, 127.5, 255]));
  assert.deepEqual([...values].map((value) => Number(value.toFixed(6))), [-1, 0, 1]);
});

test('prepareReceiptImage composes grayscale, resize, and normalization', () => {
  const data = normalizeReceiptPixels(Float32Array.from([0, 255, 255, 0]));
  const prepared = prepareReceiptImage(
    { data: Uint8Array.from([0, 255, 255, 0]), width: 2, height: 2, channels: 1 },
    { width: 2, height: 2 },
  );
  assert.deepEqual([...prepared], [...data]);
  assert.equal(prepared.length, 4);
});
