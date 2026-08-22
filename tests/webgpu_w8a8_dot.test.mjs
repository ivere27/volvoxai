import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const root = new URL('../', import.meta.url);

async function source(path) {
  return readFile(new URL(path, root), 'utf8');
}

function signedByte(value) {
  return value >= 128 ? value - 256 : value;
}

function typedByte(value, dtype) {
  return dtype === 'int8' ? signedByte(value) : value;
}

function packedDotCorrection(input, weight, inputDtype, weightDtype, inputZeroPoint, weightZeroPoint) {
  const inputOffset = inputDtype === 'uint8' ? 128 : 0;
  const weightOffset = weightDtype === 'uint8' ? 128 : 0;
  const inputZero = inputZeroPoint - inputOffset;
  const weightZero = weightZeroPoint - weightOffset;
  const signedInput = input.map((value) => typedByte(value, inputDtype) -
    (inputDtype === 'uint8' ? 128 : 0));
  const signedWeight = weight.map((value) => typedByte(value, weightDtype) -
    (weightDtype === 'uint8' ? 128 : 0));
  const dot = signedInput.reduce((sum, value, index) => sum + value * signedWeight[index], 0);
  const inputSum = signedInput.reduce((sum, value) => sum + value, 0);
  const weightSum = signedWeight.reduce((sum, value) => sum + value, 0);
  return dot - weightZero * inputSum - inputZero * weightSum + 4 * inputZero * weightZero;
}

test('WebGPU packed-dot shaders alone require the optional WGSL language feature', async () => {
  const [linear, linearTiled, conv, batch, linearFallback, linearTiledFallback,
    convFallback, batchFallback, executor] =
    await Promise.all([
      source('shaders/inference/qLinearInt8Dot.wgsl'),
      source('shaders/inference/qLinearInt8DotTiled.wgsl'),
      source('shaders/inference/qConv2DInt8DotTiled.wgsl'),
      source('shaders/inference/qBatchMatMulDot.wgsl'),
      source('shaders/inference/qLinearInt8.wgsl'),
      source('shaders/inference/qLinearInt8Tiled.wgsl'),
      source('shaders/inference/qConv2DInt8.wgsl'),
      source('shaders/inference/qBatchMatMul.wgsl'),
      source('ts/backends/GraphExecutor.ts'),
    ]);

  for (const shader of [linear, linearTiled, conv, batch]) {
    assert.match(shader,
      /^\/\/ @volvoxai-native-spv-only\nrequires packed_4x8_integer_dot_product;/);
    assert.match(shader, /dot4I8Packed\(/);
  }
  for (const shader of [linearFallback, linearTiledFallback, convFallback, batchFallback]) {
    assert.doesNotMatch(shader, /requires packed_4x8_integer_dot_product;/);
    assert.doesNotMatch(shader, /dot4I8Packed\(/);
  }
  assert.match(executor,
    /wgslLanguageFeatures\s*=\s*globalThis\.navigator\?\.gpu\?\.wgslLanguageFeatures/);
  assert.match(executor,
    /wgslLanguageFeatures\?\.has\?\.\('packed_4x8_integer_dot_product'\)\s*===\s*true/);
  assert.match(batch, /inner \+ 4u <= params\.k/);
  assert.match(batch, /for \(; inner < params\.k; inner = inner \+ 1u\)/);
  assert.match(batch, /fn b_column_word\(/);
});

test('WebGPU tiled W8A8 convolution cooperatively caches input and weight reduction tiles', async () => {
  const conv = await source('shaders/inference/qConv2DInt8DotTiled.wgsl');
  assert.match(conv, /@compute @workgroup_size\(8, 4, 1\)/);
  assert.match(conv, /var<workgroup> input_tile\s*:\s*array<u32, 32>/);
  assert.match(conv, /var<workgroup> weight_tile\s*:\s*array<u32, 256>/);
  assert.ok((conv.match(/workgroupBarrier\(\)/g) || []).length >= 2);
  assert.match(conv, /reduction_base = reduction_base \+ 32u/);
});

test('packed I8 dot correction exactly matches canonical asymmetric W8A8 products and tails', () => {
  let state = 0x6d2b79f5;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
  for (const inputDtype of ['int8', 'uint8']) {
    for (const weightDtype of ['int8', 'uint8']) {
      const inputZeroPoint = inputDtype === 'int8' ? -37 : 139;
      const weightZeroPoint = weightDtype === 'int8' ? 23 : 117;
      for (let iteration = 0; iteration < 200; iteration++) {
        const input = Array.from({ length: 4 }, () => random() & 255);
        const weight = Array.from({ length: 4 }, () => random() & 255);
        const corrected = packedDotCorrection(
          input, weight, inputDtype, weightDtype, inputZeroPoint, weightZeroPoint,
        );
        const expected = input.reduce((sum, raw, index) =>
          sum + (typedByte(raw, inputDtype) - inputZeroPoint) *
            (typedByte(weight[index], weightDtype) - weightZeroPoint), 0);
        assert.equal(corrected, expected);

        for (const tail of [1, 2, 3]) {
          const paddedInput = input.map((value, index) => index < tail ? value :
            (inputZeroPoint & 255));
          const paddedWeight = weight.map((value, index) => index < tail ? value :
            (weightZeroPoint & 255));
          const correctedTail = packedDotCorrection(
            paddedInput, paddedWeight, inputDtype, weightDtype, inputZeroPoint, weightZeroPoint,
          );
          const expectedTail = input.slice(0, tail).reduce((sum, raw, index) =>
            sum + (typedByte(raw, inputDtype) - inputZeroPoint) *
              (typedByte(weight[index], weightDtype) - weightZeroPoint), 0);
          assert.equal(correctedTail, expectedTail);
        }
      }
    }
  }
});

test('packed QBatchMatMul chunks plus scalar tail match arbitrary-K asymmetric products', () => {
  let state = 0x91e10da5;
  const random = () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return state;
  };
  for (const inputDtype of ['int8', 'uint8']) {
    for (const weightDtype of ['int8', 'uint8']) {
      const inputZeroPoint = inputDtype === 'int8' ? -53 : 173;
      const weightZeroPoint = weightDtype === 'int8' ? 41 : 89;
      for (let k = 1; k <= 19; k++) {
        const input = Array.from({ length: k }, () => random() & 255);
        const weight = Array.from({ length: k }, () => random() & 255);
        let packed = 0;
        let inner = 0;
        for (; inner + 4 <= k; inner += 4) {
          packed += packedDotCorrection(
            input.slice(inner, inner + 4), weight.slice(inner, inner + 4),
            inputDtype, weightDtype, inputZeroPoint, weightZeroPoint,
          );
        }
        for (; inner < k; inner++) {
          packed += (typedByte(input[inner], inputDtype) - inputZeroPoint) *
            (typedByte(weight[inner], weightDtype) - weightZeroPoint);
        }
        const expected = input.reduce((sum, raw, index) =>
          sum + (typedByte(raw, inputDtype) - inputZeroPoint) *
            (typedByte(weight[index], weightDtype) - weightZeroPoint), 0);
        assert.equal(packed, expected, `${inputDtype}/${weightDtype} K=${k}`);
      }
    }
  }
});
