import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
// Canonical protobuf DataType values used by the native/WASM ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_I32 = 16;
const VX_DTYPE_F32 = 18;

async function buildWasm(directory) {
  const output = join(directory, 'dequantize-linear-simd.wasm');
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_DEQUANTIZE_TESTING=1',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

async function instantiate(path) {
  const environment = {
    expf: Math.exp,
    tanhf: Math.tanh,
    logf: Math.log,
    sinf: Math.sin,
    cosf: Math.cos,
    powf: Math.pow,
  };
  const bytes = await readFile(path);
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: environment,
    math: environment,
  });
  return instance.exports;
}

function StorageFor(dtype) {
  if (dtype === VX_DTYPE_I8) return Int8Array;
  if (dtype === VX_DTYPE_U8) return Uint8Array;
  if (dtype === VX_DTYPE_I32) return Int32Array;
  if (dtype === VX_DTYPE_F32) return Float32Array;
  throw new Error(`unsupported test dtype ${dtype}`);
}

function makeHarness(api, capacity) {
  api.reset_heap();
  // Reserve four bytes per source so this harness can also prove that the
  // wider canonical dtypes stay on the scalar route. Outputs are F32-aligned
  // but deliberately not v128-aligned.
  const input = api.alloc_bytes(capacity * 4 + 16);
  const scale = api.alloc_bytes(4);
  const zero = api.alloc_bytes(4);
  const simd = api.alloc_bytes(capacity * 4 + 16) + 4;
  const scalar = api.alloc_bytes(capacity * 4 + 16) + 8;
  const requiredBytes = scalar + capacity * 4;
  const currentBytes = api.memory.buffer.byteLength;
  if (currentBytes < requiredBytes) {
    api.memory.grow(Math.ceil((requiredBytes - currentBytes) / 65_536));
  }

  return {
    run(values, inputDtype, scaleValue, zeroValue, zeroDtype = inputDtype,
      { includeZero = true } = {}) {
      assert.ok(values.length <= capacity);
      const InputStorage = StorageFor(inputDtype);
      new InputStorage(api.memory.buffer, input, values.length).set(values);
      new Float32Array(api.memory.buffer, scale, 1)[0] = scaleValue;
      if (includeZero) {
        const ZeroStorage = StorageFor(zeroDtype);
        new ZeroStorage(api.memory.buffer, zero, 1)[0] = zeroValue;
      }
      new Uint8Array(api.memory.buffer, simd, values.length * 4).fill(0xa5);
      new Uint8Array(api.memory.buffer, scalar, values.length * 4).fill(0x5a);
      api.reset_dequantize_linear_wasm_simd_blocks();
      const zeroPointer = includeZero ? zero : 0;
      const zeroType = includeZero ? zeroDtype : 0;
      const simdStatus = api.dequantize_linear_typed(
        input, inputDtype, scale, zeroPointer, zeroType, simd, values.length,
      );
      const scalarStatus = api.dequantize_linear_typed_scalar_reference(
        input, inputDtype, scale, zeroPointer, zeroType, scalar, values.length,
      );
      return {
        simdStatus,
        scalarStatus,
        // Compare the stored IEEE-754 representation, not JS numeric equality.
        simdBits: Array.from(new Uint32Array(api.memory.buffer, simd, values.length)),
        scalarBits: Array.from(new Uint32Array(api.memory.buffer, scalar, values.length)),
        blocks: api.dequantize_linear_wasm_simd_blocks(),
        pointers: { input, scale, zero, simd, scalar },
      };
    },
  };
}

test('WASM SIMD DequantizeLinear is bit-exact with the canonical double oracle', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-dequantize-simd-'));
  try {
    const api = await instantiate(await buildWasm(directory));
    assert.equal(typeof api.dequantize_linear_typed, 'function');
    assert.equal(typeof api.dequantize_linear_typed_scalar_reference, 'function');
    assert.equal(typeof api.dequantize_linear_wasm_simd_blocks, 'function');
    const harness = makeHarness(api, 100_128);

    await t.test('all I8/U8 values and every SIMD tail match for finite scales', () => {
      const scales = [
        0, -0, 1, -2.75, 0.03125, Math.fround(Math.PI),
        1.401298464324817e-45, 3.4028234663852886e38,
      ];
      for (const descriptor of [
        { dtype: VX_DTYPE_I8, zero: -113, values: Int8Array.from({ length: 256 }, (_, i) => i - 128) },
        { dtype: VX_DTYPE_U8, zero: 239, values: Uint8Array.from({ length: 256 }, (_, i) => i) },
      ]) {
        for (const scaleValue of scales) {
          for (let length = 0; length <= 47; length++) {
            const values = new (StorageFor(descriptor.dtype))(length);
            for (let index = 0; index < length; index++) {
              values[index] = descriptor.values[(index * 37 + length) & 255];
            }
            const result = harness.run(values, descriptor.dtype, scaleValue, descriptor.zero);
            assert.equal(result.simdStatus, 1);
            assert.equal(result.scalarStatus, 1);
            assert.deepEqual(result.simdBits, result.scalarBits,
              `dtype ${descriptor.dtype}, scale ${Object.is(scaleValue, -0) ? '-0' : scaleValue}, length ${length}`);
            assert.equal(result.blocks, Math.floor(length / 16));
          }
          const result = harness.run(descriptor.values, descriptor.dtype, scaleValue, descriptor.zero);
          assert.deepEqual(result.simdBits, result.scalarBits);
          assert.equal(result.blocks, 16);
        }
      }
    });

    await t.test('100003 pseudo-random bytes match for model and extreme scales', () => {
      let state = 0x243f6a88;
      const raw = Uint8Array.from({ length: 100_003 }, () => {
        state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
        return state >>> 24;
      });
      for (const descriptor of [
        { dtype: VX_DTYPE_I8, zero: -128, values: new Int8Array(raw.buffer) },
        { dtype: VX_DTYPE_U8, zero: 128, values: raw },
      ]) {
        for (const scaleValue of [0.02, -0.375, 1.401298464324817e-45, 3.4028234663852886e38]) {
          const result = harness.run(descriptor.values, descriptor.dtype, scaleValue, descriptor.zero);
          assert.equal(result.simdStatus, 1);
          assert.equal(result.scalarStatus, 1);
          assert.deepEqual(result.simdBits, result.scalarBits,
            `dtype ${descriptor.dtype}, scale ${scaleValue}`);
          assert.equal(result.blocks, Math.floor(raw.length / 16));
        }
      }
    });

    await t.test('8192 finite raw F32 scale patterns round to identical bits', () => {
      const scaleValue = new Float32Array(1);
      const scaleBits = new Uint32Array(scaleValue.buffer);
      let state = 0x9e3779b9;
      for (let sample = 0; sample < 8_192; sample++) {
        state = (Math.imul(state, 22_695_477) + 1) >>> 0;
        scaleBits[0] = state;
        if ((scaleBits[0] & 0x7f80_0000) === 0x7f80_0000) {
          scaleBits[0] ^= 0x0080_0000;
        }
        const dtype = sample & 1 ? VX_DTYPE_I8 : VX_DTYPE_U8;
        const Values = StorageFor(dtype);
        const values = Values.from({ length: 33 }, (_, index) => (index * 67 + sample) & 255);
        const zeroValue = dtype === VX_DTYPE_I8 ? -91 : 173;
        const result = harness.run(values, dtype, scaleValue[0], zeroValue);
        assert.deepEqual(result.simdBits, result.scalarBits,
          `scale bits 0x${scaleBits[0].toString(16).padStart(8, '0')}`);
        assert.equal(result.blocks, 2);
      }
    });

    await t.test('absent and cross-dtype integral zero points retain exact SIMD results', () => {
      const values = Uint8Array.from({ length: 65 }, (_, index) => (index * 29) & 255);
      for (const descriptor of [
        { zero: 0, dtype: VX_DTYPE_U8, includeZero: false },
        { zero: -17, dtype: VX_DTYPE_I8, includeZero: true },
        { zero: 101, dtype: VX_DTYPE_I32, includeZero: true },
        { zero: -31, dtype: VX_DTYPE_F32, includeZero: true },
      ]) {
        const result = harness.run(values, VX_DTYPE_U8, 0.125,
          descriptor.zero, descriptor.dtype, { includeZero: descriptor.includeZero });
        assert.deepEqual(result.simdBits, result.scalarBits);
        assert.equal(result.blocks, 4);
      }
    });

    await t.test('non-finite scales and unproved zero points preserve scalar bits', () => {
      const values = Uint8Array.from({ length: 97 }, (_, index) => (index * 43) & 255);
      for (const scaleValue of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
        const result = harness.run(values, VX_DTYPE_U8, scaleValue, 128);
        assert.equal(result.simdStatus, 1, 'non-finite dequantization scales remain accepted');
        assert.deepEqual(result.simdBits, result.scalarBits);
        assert.equal(result.blocks, 0);
      }
      for (const descriptor of [
        { zero: 0.5, dtype: VX_DTYPE_F32 },
        { zero: Number.NaN, dtype: VX_DTYPE_F32 },
        { zero: Number.POSITIVE_INFINITY, dtype: VX_DTYPE_F32 },
        { zero: 20_000_000, dtype: VX_DTYPE_I32 },
        { zero: -20_000_000, dtype: VX_DTYPE_I32 },
      ]) {
        const result = harness.run(values, VX_DTYPE_U8, 0.2,
          descriptor.zero, descriptor.dtype);
        assert.equal(result.simdStatus, 1);
        assert.deepEqual(result.simdBits, result.scalarBits);
        assert.equal(result.blocks, 0);
      }
    });

    await t.test('F32/I32 sources including NaN and infinity stay scalar and exact', () => {
      const f32 = Float32Array.of(
        Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
        -0, 0, 1.25, -3.5, 3.4028234663852886e38,
      );
      const f32Result = harness.run(f32, VX_DTYPE_F32, 0.25, -0, VX_DTYPE_F32);
      assert.deepEqual(f32Result.simdBits, f32Result.scalarBits);
      assert.equal(f32Result.blocks, 0);

      const i32 = Int32Array.of(
        -2_147_483_648, -16_777_217, -1, 0, 1, 16_777_217, 2_147_483_647,
      );
      const i32Result = harness.run(i32, VX_DTYPE_I32, 0.125, 16_777_216, VX_DTYPE_I32);
      assert.deepEqual(i32Result.simdBits, i32Result.scalarBits);
      assert.equal(i32Result.blocks, 0);
    });

    await t.test('invalid descriptors fail before writing', () => {
      const values = Uint8Array.from({ length: 33 }, (_, index) => index);
      const valid = harness.run(values, VX_DTYPE_U8, 0.25, 3);
      const { input, scale, zero, simd } = valid.pointers;
      const sentinel = new Array(values.length * 4).fill(0x6d);
      for (const invoke of [
        () => api.dequantize_linear_typed(input, 99, scale, zero, VX_DTYPE_U8, simd, values.length),
        () => api.dequantize_linear_typed(input, VX_DTYPE_U8, scale, zero, 99, simd, values.length),
        () => api.dequantize_linear_typed(input, VX_DTYPE_U8, scale, zero, VX_DTYPE_U8, simd, -1),
        () => api.dequantize_linear_typed(0, VX_DTYPE_U8, scale, zero, VX_DTYPE_U8, simd, values.length),
        () => api.dequantize_linear_typed(input, VX_DTYPE_U8, 0, zero, VX_DTYPE_U8, simd, values.length),
        () => api.dequantize_linear_typed(input, VX_DTYPE_U8, scale, zero, VX_DTYPE_U8, 0, values.length),
      ]) {
        new Uint8Array(api.memory.buffer, simd, values.length * 4).fill(0x6d);
        api.reset_dequantize_linear_wasm_simd_blocks();
        assert.equal(invoke(), 0);
        assert.deepEqual(Array.from(new Uint8Array(api.memory.buffer, simd, values.length * 4)), sentinel);
        assert.equal(api.dequantize_linear_wasm_simd_blocks(), 0);
      }
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
