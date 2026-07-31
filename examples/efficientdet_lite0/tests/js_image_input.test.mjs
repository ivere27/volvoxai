import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { efficientDetInputFromRgba } from '../image_input.js';

const rgba = Uint8ClampedArray.of(
  0, 127, 255, 19,
  128, 64, 200, 255,
);

function tensor(dtype, quantization = null) {
  return { shape: [1, 1, 2, 3], dtype, quantization };
}

test('EfficientDet image input normalizes RGB values to zero-one for F32 packages', () => {
  const input = efficientDetInputFromRgba(rgba, tensor('float32'));
  assert.ok(input instanceof Float32Array);
  assert.deepEqual(
    [...input],
    [...Float32Array.from([0, 127, 255, 128, 64, 200], (value) => value / 255)],
  );
});

test('EfficientDet image input preserves physical bytes for canonical U8 packages', () => {
  const input = efficientDetInputFromRgba(rgba, tensor('uint8', {
    scheme: 'per_tensor', scale: 0.0078125, zero_point: 127,
  }));
  assert.ok(input instanceof Uint8Array);
  assert.deepEqual([...input], [0, 127, 255, 128, 64, 200]);
});

test('EfficientDet image input centers source levels for canonical I8 packages', () => {
  const input = efficientDetInputFromRgba(rgba, tensor('int8', {
    scheme: 'per_tensor', scale: 0.0078125, zero_point: 0,
  }));
  assert.ok(input instanceof Int8Array);
  assert.deepEqual([...input], [-128, -1, 127, 0, -64, 72]);
});

test('EfficientDet image input rejects malformed typed image contracts', () => {
  assert.throws(
    () => efficientDetInputFromRgba(rgba, tensor('uint8')),
    /requires per-tensor quantization metadata/,
  );
  assert.throws(
    () => efficientDetInputFromRgba(rgba, { shape: [1, 3, 1, 3], dtype: 'float32' }),
    /expected 12/,
  );
  assert.throws(
    () => efficientDetInputFromRgba(rgba, tensor('int32')),
    /dtype 'int32' is unsupported/,
  );
});

test('EfficientDet browser demo exposes WebGPU through public backend selection', async () => {
  const html = await readFile(new URL('../../efficientdet_lite0.html', import.meta.url), 'utf8');
  assert.match(html, /<option value="webgpu">WebGPU<\/option>/);
  assert.match(html, /VolvoxAI\.createRuntime\(\{/);
  assert.match(html, /model\.compile\(\{/);
  assert.match(html, /compiled\.createContext\(\)/);
  assert.match(html, /result\.output\('scores'\)\.read\(\)/);
});
