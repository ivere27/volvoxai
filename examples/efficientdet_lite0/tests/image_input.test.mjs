import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { efficientDetInputViewFromRgba } from '../image_input.js';

const rgba = Uint8ClampedArray.of(
  0, 127, 255, 19,
  128, 64, 200, 255,
);

function tensor(dtype, quantization = null) {
  return { shape: [1, 1, 2, 3], dtype, quantization };
}

test('EfficientDet image input normalizes RGB values to zero-one for F32 packages', () => {
  const input = efficientDetInputViewFromRgba(rgba, tensor('float32'));
  assert.ok(input.data instanceof Float32Array);
  assert.deepEqual(input.shape, [1, 1, 2, 3]);
  assert.deepEqual(
    [...input.data],
    [...Float32Array.from([0, 127, 255, 128, 64, 200], (value) => value / 255)],
  );
});

test('EfficientDet image input preserves bytes for canonical U8 packages', () => {
  const input = efficientDetInputViewFromRgba(rgba, tensor('uint8', {
    scheme: 'per_tensor', scale: 0.0078125, zero_point: 127,
  }));
  assert.ok(input.data instanceof Uint8Array);
  assert.deepEqual(input.shape, [1, 1, 2, 3]);
  assert.deepEqual([...input.data], [0, 127, 255, 128, 64, 200]);
});

test('EfficientDet image input centers source levels for canonical I8 packages', () => {
  const input = efficientDetInputViewFromRgba(rgba, tensor('int8', {
    scheme: 'per_tensor', scale: 0.0078125, zero_point: 0,
  }));
  assert.ok(input.data instanceof Int8Array);
  assert.deepEqual(input.shape, [1, 1, 2, 3]);
  assert.deepEqual([...input.data], [-128, -1, 127, 0, -64, 72]);
});

test('EfficientDet image input rejects malformed typed image contracts', () => {
  assert.throws(
    () => efficientDetInputViewFromRgba(rgba, tensor('uint8')),
    /requires per-tensor quantization metadata/,
  );
  assert.throws(
    () => efficientDetInputViewFromRgba(rgba, { shape: [1, 3, 1, 3], dtype: 'float32' }),
    /expected 12/,
  );
  assert.throws(
    () => efficientDetInputViewFromRgba(rgba, tensor('int32')),
    /dtype 'int32' is unsupported/,
  );
});

test('EfficientDet browser demo exposes WebGPU through public backend selection', async () => {
  const html = await readFile(new URL('../../efficientdet_lite0.html', import.meta.url), 'utf8');
  assert.match(html, /<option value="webgpu">WebGPU<\/option>/);
  assert.match(html, /VolvoxAI\.createRuntime\(\{/);
  assert.match(html, /runtime\.compile\(snapshot, \{/);
  assert.match(html, /compiled\.createContext\(\)/);
  assert.match(html, /efficientDetInputViewFromRgba/);
  assert.match(html, /context\.execute\(\{ input0 \}\)/);
  assert.match(html, /result\.output\('scores'\)\.read\(\)/);
});
