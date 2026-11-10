#!/usr/bin/env node

import assert from 'node:assert/strict';
import { basename } from 'node:path';
import { loadWasmKernelFixture } from './build_wasm_kernel_fixture.mjs';

// Canonical protobuf DataType values used by the C/WASM kernel ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_F32 = 18;
const OUTPUT_SENTINEL = 0xa5;
const PACKED_Q8_MAGIC = 0x32513856;

const artifactPaths = process.argv.slice(2);
if (artifactPaths.length === 0) {
  throw new Error(
    'usage: node tools/test_wasm_baseline_simd.mjs <kernel-fixture.wasm>',
  );
}

function assertKernelExports(api, artifactPath) {
  for (const name of [
    'memory',
    'alloc_bytes',
    'reset_heap',
    'vx_packed_q8_weight_size',
    'vx_pack_q8_weight',
    'qlinear_i8u8',
    'vx_qlinear_i8u8_packed',
    'qbatch_matmul_i8u8',
    'vx_qbatch_matmul_i8u8_simd128',
    'qsdpa_i8u8',
  ]) {
    assert.ok(api[name], `${artifactPath} must export '${name}'`);
  }
  assert.ok(api.memory instanceof WebAssembly.Memory);
}

function arena(api) {
  const allocate = (byteLength) => {
    assert.ok(Number.isSafeInteger(byteLength) && byteLength > 0);
    const pointer = Number(api.alloc_bytes(byteLength));
    assert.ok(Number.isSafeInteger(pointer) && pointer > 0);
    const required = pointer + byteLength;
    if (required > api.memory.buffer.byteLength) {
      api.memory.grow(
        Math.ceil((required - api.memory.buffer.byteLength) / 65536),
      );
    }
    return pointer;
  };
  const write = (values) => {
    const pointer = allocate(values.byteLength);
    new Uint8Array(api.memory.buffer, pointer, values.byteLength).set(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    );
    return pointer;
  };
  const bytes = (pointer, byteLength) =>
    new Uint8Array(api.memory.buffer, pointer, byteLength).slice();
  return Object.freeze({ allocate, write, bytes });
}

function verifyPackedQLinear(api, artifactPath) {
  api.reset_heap();
  const memory = arena(api);
  // N>=8 and a widened pair pack admit baseline SIMD128. Odd M/K/N exercise
  // every final tile in the developer fixture built from release kernel flags.
  const rows = 5;
  const dIn = 33;
  const dOut = 9;
  const input = Uint8Array.from(
    { length: rows * dIn },
    (_, index) => (index * 29 + 17) & 0xff,
  );
  const weights = Int8Array.from(
    { length: dOut * dIn },
    (_, index) => ((index * 37 + 11) % 255) - 127,
  );
  const bias = Int32Array.from(
    { length: dOut },
    (_, column) => column * 31 - 113,
  );
  const weightScales = Float32Array.from(
    { length: dOut },
    (_, column) => 0.00390625 * (1 + column % 7),
  );
  const weightZeroPoints = new Int32Array(dOut);

  const inputPointer = memory.write(input);
  const weightPointer = memory.write(weights);
  const biasPointer = memory.write(bias);
  const scalePointer = memory.write(weightScales);
  const zeroPointPointer = memory.write(weightZeroPoints);
  const packedBytes = Number(api.vx_packed_q8_weight_size(dIn, dOut));
  assert.ok(packedBytes > 48);
  const packedPointer = memory.allocate(packedBytes);
  const portableOutputPointer = memory.allocate(rows * dOut);
  const packedOutputPointer = memory.allocate(rows * dOut);

  assert.equal(api.vx_pack_q8_weight(
    packedPointer, packedBytes, weightPointer, dIn, dOut, VX_DTYPE_I8, 1,
  ), 1, `${artifactPath}: create widened W8A8 pack`);
  const header = new DataView(api.memory.buffer, packedPointer, 48);
  assert.equal(header.getUint32(0, true), PACKED_Q8_MAGIC);
  assert.equal(header.getUint32(32, true), Math.ceil(dOut / 8));
  assert.equal(header.getUint32(36, true), Math.ceil(dIn / 2));
  assert.ok(header.getUint32(40, true) < packedBytes);

  const common = [
    rows,
    dIn,
    dOut,
    0.03125,
    139,
    0.0625,
    -7,
    VX_DTYPE_U8,
    VX_DTYPE_I8,
    VX_DTYPE_I8,
  ];
  const portableCall = () => api.qlinear_i8u8(
    inputPointer,
    weightPointer,
    biasPointer,
    scalePointer,
    zeroPointPointer,
    portableOutputPointer,
    ...common,
  );
  const packedCall = () => api.vx_qlinear_i8u8_packed(
    inputPointer,
    packedPointer,
    biasPointer,
    scalePointer,
    zeroPointPointer,
    packedOutputPointer,
    ...common,
  );

  new Uint8Array(api.memory.buffer, portableOutputPointer, rows * dOut)
    .fill(OUTPUT_SENTINEL);
  new Uint8Array(api.memory.buffer, packedOutputPointer, rows * dOut)
    .fill(OUTPUT_SENTINEL);
  assert.equal(portableCall(), 1, `${artifactPath}: portable QLinear`);
  assert.equal(packedCall(), 1, `${artifactPath}: packed QLinear`);
  const expected = memory.bytes(portableOutputPointer, rows * dOut);
  assert.deepEqual(
    memory.bytes(packedOutputPointer, rows * dOut),
    expected,
    `${artifactPath}: baseline SIMD packed QLinear must be byte exact`,
  );

  header.setUint32(0, 0, true);
  new Uint8Array(api.memory.buffer, packedOutputPointer, rows * dOut)
    .fill(OUTPUT_SENTINEL);
  assert.equal(packedCall(), 0, `${artifactPath}: reject corrupt packed header`);
  assert.deepEqual(
    memory.bytes(packedOutputPointer, rows * dOut),
    new Uint8Array(rows * dOut).fill(OUTPUT_SENTINEL),
    `${artifactPath}: packed rejection must precede output writes`,
  );
  header.setUint32(0, PACKED_Q8_MAGIC, true);
  assert.equal(packedCall(), 1, `${artifactPath}: packed QLinear remains reusable`);
  assert.deepEqual(memory.bytes(packedOutputPointer, rows * dOut), expected);
}

function verifySymmetricPackedQLinear(api, artifactPath) {
  // Exercise both sides of N8 admission, MR5, activation-pair packing and the
  // 4096-pair panel limit. Affine U8 fixtures do not select this I8 path.
  const shapes = [
    [1, 15, 8], [4, 16, 8], [5, 17, 8], [6, 33, 16],
    [5, 16, 7], [5, 16, 9],
    [5, 8191, 8], [6, 8192, 8], [6, 8193, 8],
  ];
  const cases = [
    // Canonical scale order gives an accumulator product of exactly +/-4.5.
    // Dividing input_scale first changes this to +/-4.500000476837158.
    { name: 'halfway', inputScale: 0.001, weightScale: 0.003,
      outputScale: 0.01, bias: 15000, expected: 4 },
    // All three scales are finite normal F32 values. Their canonical
    // multiplier is 8, but input_scale/output_scale alone overflows.
    { name: 'finite scale overflow', inputScale: 2 ** 64,
      weightScale: 2 ** -126, outputScale: 2 ** -65, bias: 1, expected: 8 },
    { name: 'nonzero pair packing', inputScale: 0.03125,
      weightScale: 0.0625, outputScale: 2 },
  ];
  for (const test of cases) {
    for (const [rows, dIn, dOut] of shapes) {
      api.reset_heap();
      const memory = arena(api);
      const label = `${artifactPath}: symmetric I8 ${test.name} ` +
        `M=${rows} K=${dIn} N=${dOut}`;
      const fixed = test.expected !== undefined;
      const input = Int8Array.from({ length: rows * dIn }, (_, index) =>
        fixed ? 0 : ((index * 29 + 17) % 255) - 127);
      const weights = Int8Array.from({ length: dOut * dIn }, (_, index) =>
        fixed ? 0 : ((index * 37 + 11) % 255) - 127);
      const bias = Int32Array.from({ length: dOut }, (_, column) =>
        fixed ? (column % 2 ? -test.bias : test.bias) : column * 31 - 113);
      const inputPointer = memory.write(input);
      const weightPointer = memory.write(weights);
      const biasPointer = memory.write(bias);
      const scalePointer = memory.write(
        new Float32Array(dOut).fill(test.weightScale));
      const zeroPointer = memory.write(new Int32Array(dOut));
      const packedBytes = Number(api.vx_packed_q8_weight_size(dIn, dOut));
      const packedPointer = memory.allocate(packedBytes);
      const referencePointer = memory.allocate(rows * dOut);
      const outputPointer = memory.write(
        new Uint8Array(rows * dOut + 16).fill(OUTPUT_SENTINEL));
      assert.equal(api.vx_pack_q8_weight(packedPointer, packedBytes,
        weightPointer, dIn, dOut, VX_DTYPE_I8, 1), 1, `${label}: pack`);
      const common = [rows, dIn, dOut, test.inputScale, 0,
        test.outputScale, 0, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8];
      assert.equal(api.qlinear_i8u8(inputPointer, weightPointer,
        biasPointer, scalePointer, zeroPointer, referencePointer, ...common),
        1, `${label}: portable reference`);
      assert.equal(api.vx_qlinear_i8u8_packed(inputPointer, packedPointer,
        biasPointer, scalePointer, zeroPointer, outputPointer, ...common),
        1, `${label}: packed`);
      const expected = memory.bytes(referencePointer, rows * dOut);
      if (fixed) {
        const golden = Int8Array.from({ length: rows * dOut }, (_, index) =>
          index % dOut % 2 ? -test.expected : test.expected);
        assert.deepEqual(expected, new Uint8Array(golden.buffer),
          `${label}: golden`);
      }
      assert.deepEqual(memory.bytes(outputPointer, rows * dOut), expected,
        `${label}: byte exact`);
      assert.deepEqual(memory.bytes(outputPointer + rows * dOut, 16),
        new Uint8Array(16).fill(OUTPUT_SENTINEL), `${label}: output bounds`);
    }
  }
}

function verifyDirectPackedQConv(api, artifactPath) {
  // Batch crossing, MR/N tails, odd channels, dilation, stride and asymmetric
  // padding are compared with canonical Conv, then rerun with changed input.
  const shapes = [
    [1, 4, 4, 8, 1, 1, 1, 0, 0],
    [1, 5, 16, 22, 1, 1, 1, 0, 0],
    [1, 6, 3, 9, 3, 1, 1, 1, 1],
    [2, 3, 3, 7, 3, 1, 3, 3, 3],
    [1, 33, 16, 16, 3, 2, 3, 2, 4],
    [1, 37, 128, 128, 11, 1, 1, 5, 5],
    [1, 37, 128, 22, 11, 1, 3, 15, 15],
    [1, 37, 128, 22, 11, 1, 5, 25, 25],
    [1, 1, 1, 1, 3, 1, 5, 12, 3],
    [1, 1, 8192, 8, 1, 1, 1, 0, 0],
    [1, 1, 8193, 8, 1, 1, 1, 0, 0],
  ];
  const cases = shapes.map((shape) => ({ shape,
    inputScale: 0.03125, weightScale: 0.00390625, outputScale: 0.5 }));
  cases.push({ shape: [1, 6, 2, 22, 3, 1, 1, 1, 1],
    inputScale: 0.001, weightScale: 0.003, outputScale: 0.01,
    bias: 15000, expected: 4 });
  cases.push({ shape: [1, 6, 2, 22, 3, 1, 1, 1, 1],
    inputScale: 2 ** 64, weightScale: 2 ** -126, outputScale: 2 ** -65,
    bias: 1, expected: 8 });
  for (const test of cases) {
    api.reset_heap();
    const memory = arena(api);
    const [batch, width, channels, outputs, kernel, stride, dilation, left, right] = test.shape;
    const outputWidth = Math.floor((width + left + right - (kernel - 1) * dilation - 1) /
      stride) + 1;
    const columns = kernel * channels;
    const count = batch * outputWidth * outputs;
    const label = `${artifactPath}: direct Conv ${test.shape.join('/')}`;
    const fixed = test.expected !== undefined;
    const input = Int8Array.from({ length: batch * width * channels }, (_, index) =>
      fixed ? 0 : ((index * 29 + 17) & 255) - 128);
    const weights = Int8Array.from({ length: columns * outputs }, (_, index) =>
      fixed ? 0 : ((index * 37 + 11) & 255) - 128);
    const bias = Int32Array.from({ length: outputs }, (_, index) =>
      fixed ? (index % 2 ? -test.bias : test.bias) : index * 31 - 113);
    const inputPointer = memory.write(input);
    const weightPointer = memory.write(weights);
    const biasPointer = memory.write(bias);
    const scalePointer = memory.write(new Float32Array(outputs).fill(test.weightScale));
    const zeroPointer = memory.write(new Int32Array(outputs));
    const packedBytes = Number(api.vx_packed_q8_weight_size(columns, outputs));
    const packedPointer = memory.allocate(packedBytes);
    const referencePointer = memory.allocate(count);
    const outputPointer = memory.allocate(count + 16);
    assert.equal(api.vx_pack_q8_weight(packedPointer, packedBytes, weightPointer,
      columns, outputs, VX_DTYPE_I8, 1), 1, `${label}: pack`);
    const direct = (destination = outputPointer, extent = outputWidth) =>
      api.vx_qconv1d_i8_wasm_packed(inputPointer, packedPointer, biasPointer,
        scalePointer, zeroPointer, destination, batch, width, channels, extent,
        outputs, kernel, stride, dilation, left, right, test.inputScale, test.outputScale);
    const resetOutput = () => new Uint8Array(api.memory.buffer, outputPointer,
      count + 16).fill(OUTPUT_SENTINEL);
    for (let request = 0; request < 2; request++) {
      if (request && !fixed) {
        for (let index = 0; index < input.length; index++) input[index] = -input[index] - 1;
        new Int8Array(api.memory.buffer, inputPointer, input.length).set(input);
      }
      resetOutput();
      assert.equal(api.qconv2d_i8u8(inputPointer, weightPointer, biasPointer,
        scalePointer, zeroPointer, referencePointer, batch, 1, width, channels,
        1, outputWidth, outputs, 1, kernel, channels, 1, stride, 1, dilation,
        0, left, 0, right, 1, 0, test.inputScale, 0, test.outputScale, 0,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8), 1, `${label}: canonical`);
      if (columns > 8192) {
        assert.equal(direct(), 0, `${label}: panel capacity rejection`);
        assert.deepEqual(memory.bytes(outputPointer, count),
          new Uint8Array(count).fill(OUTPUT_SENTINEL), `${label}: no rejected writes`);
        const patchPointer = memory.allocate(batch * outputWidth * columns);
        assert.equal(api.qconv2d_im2col_i8u8(inputPointer, patchPointer,
          biasPointer, zeroPointer, batch, 1, width, channels, 1, outputWidth,
          outputs, 1, kernel, 1, stride, 1, dilation, 0, left, 0, right,
          0, VX_DTYPE_I8, VX_DTYPE_I8), 1, `${label}: fallback im2col`);
        assert.equal(api.vx_qlinear_i8u8_packed(patchPointer, packedPointer,
          biasPointer, scalePointer, zeroPointer, outputPointer,
          batch * outputWidth, columns, outputs, test.inputScale, 0,
          test.outputScale, 0, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8),
          1, `${label}: fallback GEMM`);
      } else {
        assert.equal(direct(), 1, `${label}: admitted`);
      }
      const expected = memory.bytes(referencePointer, count);
      if (fixed) {
        const golden = Int8Array.from({ length: count }, (_, index) =>
          index % outputs % 2 ? -test.expected : test.expected);
        assert.deepEqual(expected, new Uint8Array(golden.buffer), `${label}: golden`);
      }
      assert.deepEqual(memory.bytes(outputPointer, count), expected,
        `${label}: current input request ${request}`);
      assert.deepEqual(memory.bytes(outputPointer + count, 16),
        new Uint8Array(16).fill(OUTPUT_SENTINEL), `${label}: output bounds`);
    }
    resetOutput();
    assert.equal(direct(outputPointer, outputWidth + 1), 0, `${label}: geometry rejection`);
    new Int32Array(api.memory.buffer, zeroPointer, outputs)[0] = 1;
    assert.equal(direct(), 0, `${label}: asymmetric weight rejection`);
    new Int32Array(api.memory.buffer, zeroPointer, outputs)[0] = 0;
    new Int32Array(api.memory.buffer, biasPointer, outputs)[0] = 0x7fffffff;
    assert.equal(direct(), 0, `${label}: accumulator overflow rejection`);
    new Int32Array(api.memory.buffer, biasPointer, outputs).set(bias);
    assert.equal(direct(inputPointer), 0, `${label}: overlapping input rejection`);
    assert.deepEqual(memory.bytes(outputPointer, count + 16),
      new Uint8Array(count + 16).fill(OUTPUT_SENTINEL), `${label}: reject before writing`);
  }
}

function verifyQBatchMatMul(api, artifactPath) {
  api.reset_heap();
  const memory = arena(api);
  const m = 17;
  const k = 65;
  const n = 63;
  const left = Uint8Array.from(
    { length: m * k },
    (_, index) => (index * 29 + 17) & 0xff,
  );
  const right = Int8Array.from(
    { length: k * n },
    (_, index) => ((index * 37 + 11) % 255) - 127,
  );
  const leftPointer = memory.write(left);
  const rightPointer = memory.write(right);
  const scalarOutputPointer = memory.allocate(m * n);
  const simdOutputPointer = memory.allocate(m * n);
  const common = [
    m,
    k,
    n,
    0.03125,
    139,
    0.0078125,
    -13,
    0.0625,
    -7,
    VX_DTYPE_U8,
    VX_DTYPE_I8,
    VX_DTYPE_I8,
  ];
  const scalarCall = () => api.qbatch_matmul_i8u8(
    leftPointer, rightPointer, scalarOutputPointer, ...common,
  );
  const simdCall = () => api.vx_qbatch_matmul_i8u8_simd128(
    leftPointer, rightPointer, simdOutputPointer, ...common,
  );

  assert.equal(scalarCall(), 1, `${artifactPath}: scalar QBatchMatMul`);
  assert.equal(simdCall(), 1, `${artifactPath}: SIMD128 QBatchMatMul`);
  const expected = memory.bytes(scalarOutputPointer, m * n);
  assert.deepEqual(
    memory.bytes(simdOutputPointer, m * n),
    expected,
    `${artifactPath}: SIMD128 QBatchMatMul odd tails must be byte exact`,
  );

  new Uint8Array(api.memory.buffer, simdOutputPointer, m * n)
    .fill(OUTPUT_SENTINEL);
  assert.equal(api.vx_qbatch_matmul_i8u8_simd128(
    leftPointer,
    rightPointer,
    simdOutputPointer,
    ...common.slice(0, -1),
    VX_DTYPE_F32,
  ), 0, `${artifactPath}: reject a non-byte QBatchMatMul output`);
  assert.deepEqual(
    memory.bytes(simdOutputPointer, m * n),
    new Uint8Array(m * n).fill(OUTPUT_SENTINEL),
    `${artifactPath}: QBatchMatMul rejection must precede output writes`,
  );
  assert.equal(simdCall(), 1, `${artifactPath}: QBatchMatMul remains reusable`);
  assert.deepEqual(memory.bytes(simdOutputPointer, m * n), expected);
}

function verifyQSDPA(api, artifactPath) {
  api.reset_heap();
  const memory = arena(api);
  // D=40 exercises five SIMD dot/accumulation blocks. This fixture and golden
  // originate from the portable-C QSDPA contract retained before CPU-JS was
  // removed; it is intentionally small enough to diagnose byte differences.
  const q = Int8Array.from(
    { length: 40 },
    (_, index) => ((index * 17 + 11) % 101) - 53,
  );
  const k = Uint8Array.from(
    { length: 5 * 40 },
    (_, index) => 127 + (((index * 13 + 7) % 91) - 45),
  );
  const v = Uint8Array.from(
    { length: 5 * 40 },
    (_, index) => 120 + (((index * 19 + 5) % 81) - 40),
  );
  const mask = Int32Array.of(1, 1, 0, 1, 1);
  const expected = Int8Array.of(
    -24, -5, -1, 1, -29, -10, -6, -4, -11, -15,
    4, -9, -16, -20, -1, 3, 5, -25, -6, -2,
    0, -7, -11, 8, -5, -12, -16, 3, -10, -17,
    -21, -2, 2, 4, -26, -7, -3, -1, -8, -12,
  );
  const qPointer = memory.write(q);
  const kPointer = memory.write(k);
  const vPointer = memory.write(v);
  const maskPointer = memory.write(mask);
  const outputPointer = memory.allocate(q.byteLength);
  const common = [
    outputPointer,
    1,
    1,
    5,
    40,
    1,
    0.03125,
    -3,
    0.015625,
    127,
    0.0625,
    120,
    0.0625,
    -7,
    0.125,
    VX_DTYPE_I8,
    VX_DTYPE_U8,
    VX_DTYPE_U8,
    VX_DTYPE_I8,
    0,
    1,
  ];
  const call = (maskArgument = maskPointer) => api.qsdpa_i8u8(
    qPointer, kPointer, vPointer, maskArgument, ...common,
  );

  assert.equal(call(), 1, `${artifactPath}: mixed I8/U8 QSDPA`);
  assert.deepEqual(
    new Int8Array(api.memory.buffer, outputPointer, q.length).slice(),
    expected,
    `${artifactPath}: D=40 masked QSDPA golden`,
  );

  new Uint8Array(api.memory.buffer, outputPointer, q.length).fill(OUTPUT_SENTINEL);
  assert.equal(call(0), 0, `${artifactPath}: required QSDPA mask is missing`);
  assert.deepEqual(
    memory.bytes(outputPointer, q.length),
    new Uint8Array(q.length).fill(OUTPUT_SENTINEL),
    `${artifactPath}: QSDPA rejection must precede output writes`,
  );

  new Int32Array(api.memory.buffer, maskPointer, mask.length).fill(0);
  assert.equal(call(), 1, `${artifactPath}: all-masked QSDPA`);
  assert.deepEqual(
    new Int8Array(api.memory.buffer, outputPointer, q.length).slice(),
    new Int8Array(q.length).fill(-7),
    `${artifactPath}: all-masked QSDPA returns the output zero point`,
  );
  new Int32Array(api.memory.buffer, maskPointer, mask.length).set(mask);
  assert.equal(call(), 1, `${artifactPath}: QSDPA remains reusable`);
  assert.deepEqual(
    new Int8Array(api.memory.buffer, outputPointer, q.length).slice(),
    expected,
  );
}

for (const artifactPath of artifactPaths) {
  const { exports: api } = await loadWasmKernelFixture(artifactPath);
  assertKernelExports(api, artifactPath);
  verifyPackedQLinear(api, artifactPath);
  verifySymmetricPackedQLinear(api, artifactPath);
  verifyDirectPackedQConv(api, artifactPath);
  verifyQBatchMatMul(api, artifactPath);
  verifyQSDPA(api, artifactPath);
  console.log(
    `Verified baseline WASM kernels in ${basename(artifactPath)}: ` +
    'affine/symmetric packed QLinear, direct QConv, QBatchMatMul SIMD128, and quantized attention.',
  );
}
