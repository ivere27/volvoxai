#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { wasmToolImports } from './wasm_host_imports.mjs';

// Canonical protobuf DataType values used by the C/WASM kernel ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_F32 = 18;
const OUTPUT_SENTINEL = 0xa5;
const PACKED_Q8_MAGIC = 0x32513856;

const artifactPaths = process.argv.slice(2);
if (artifactPaths.length === 0) {
  throw new Error(
    'usage: node tools/test_wasm_baseline_simd.mjs <parent.wasm> [...]',
  );
}

function assertKernelExports(api, artifactPath) {
  for (const name of [
    'memory',
    'alloc_bytes',
    'reset_heap',
    'packed_q8_weight_size',
    'pack_q8_weight',
    'qlinear_i8u8',
    'qlinear_i8u8_packed',
    'qbatch_matmul_i8u8',
    'qbatch_matmul_i8u8_simd128',
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
  // every final tile without relying on a test-only instrumented sidecar.
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
  const packedBytes = Number(api.packed_q8_weight_size(dIn, dOut));
  assert.ok(packedBytes > 48);
  const packedPointer = memory.allocate(packedBytes);
  const portableOutputPointer = memory.allocate(rows * dOut);
  const packedOutputPointer = memory.allocate(rows * dOut);

  assert.equal(api.pack_q8_weight(
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
  const packedCall = () => api.qlinear_i8u8_packed(
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
  const simdCall = () => api.qbatch_matmul_i8u8_simd128(
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
  assert.equal(api.qbatch_matmul_i8u8_simd128(
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
  const module = await WebAssembly.compile(await readFile(artifactPath));
  const { exports: api } = await WebAssembly.instantiate(module, wasmToolImports());
  assertKernelExports(api, artifactPath);
  verifyPackedQLinear(api, artifactPath);
  verifyQBatchMatMul(api, artifactPath);
  verifyQSDPA(api, artifactPath);
  console.log(
    `Verified baseline WASM kernels in ${basename(artifactPath)}: ` +
    'packed QLinear, QBatchMatMul SIMD128, and quantized attention.',
  );
}
