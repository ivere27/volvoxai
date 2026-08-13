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

async function buildWasm(directory) {
  const output = join(directory, 'quantize-linear-simd.wasm');
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_QUANTIZE_TESTING=1',
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

function roundTiesToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return (lower & 1) === 0 ? lower : lower + 1;
}

function expectedFromTransformed(value, minimum, maximum, zero) {
  if (Number.isNaN(value)) return zero;
  if (value <= minimum) return minimum;
  if (value >= maximum) return maximum;
  if (!Number.isFinite(value)) return value < 0 ? minimum : maximum;
  return roundTiesToEven(value);
}

function makeHarness(api, capacity) {
  api.reset_heap();
  // Deliberately keep the F32 input off a SIMD boundary and byte outputs at
  // different unaligned addresses. WebAssembly SIMD loads/stores must retain
  // the portable unaligned-memory contract.
  const input = api.alloc_bytes(capacity * 4 + 16) + 4;
  const scale = api.alloc_bytes(4);
  const zero = api.alloc_bytes(1);
  const simd = api.alloc_bytes(capacity + 16) + 3;
  const scalar = api.alloc_bytes(capacity + 16) + 5;
  const requiredBytes = scalar + capacity;
  const currentBytes = api.memory.buffer.byteLength;
  if (currentBytes < requiredBytes) {
    api.memory.grow(Math.ceil((requiredBytes - currentBytes) / 65_536));
  }
  return {
    run(values, dtype, scaleValue, zeroValue, { includeZero = true } = {}) {
      assert.ok(values.length <= capacity);
      new Float32Array(api.memory.buffer, input, values.length).set(values);
      new Float32Array(api.memory.buffer, scale, 1)[0] = scaleValue;
      if (dtype === VX_DTYPE_I8) new Int8Array(api.memory.buffer, zero, 1)[0] = zeroValue;
      else new Uint8Array(api.memory.buffer, zero, 1)[0] = zeroValue;
      new Uint8Array(api.memory.buffer, simd, values.length).fill(0xa5);
      new Uint8Array(api.memory.buffer, scalar, values.length).fill(0x5a);
      api.reset_quantize_linear_wasm_simd_blocks();
      const zeroPointer = includeZero ? zero : 0;
      const zeroDtype = includeZero ? dtype : 0;
      const simdStatus = api.quantize_linear_typed(
        input, scale, zeroPointer, zeroDtype, simd, dtype, values.length,
      );
      const scalarStatus = api.quantize_linear_typed_scalar_reference(
        input, scale, zeroPointer, zeroDtype, scalar, dtype, values.length,
      );
      const Storage = dtype === VX_DTYPE_I8 ? Int8Array : Uint8Array;
      return {
        simdStatus,
        scalarStatus,
        simd: Array.from(new Storage(api.memory.buffer, simd, values.length)),
        scalar: Array.from(new Storage(api.memory.buffer, scalar, values.length)),
        blocks: api.quantize_linear_wasm_simd_blocks(),
        pointers: { input, scale, zero, simd, scalar },
      };
    },
  };
}

test('WASM SIMD QuantizeLinear is byte-exact with its canonical scalar fallback', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-quantize-simd-'));
  try {
    const api = await instantiate(await buildWasm(directory));
    assert.equal(typeof api.quantize_linear_typed, 'function');
    assert.equal(typeof api.quantize_linear_typed_scalar_reference, 'function');
    assert.equal(typeof api.quantize_linear_wasm_simd_blocks, 'function');
    const harness = makeHarness(api, 100_128);

    await t.test('every I8/U8 integer, half tie, quarter, and saturation edge matches', () => {
      for (const descriptor of [
        { dtype: VX_DTYPE_I8, minimum: -128, maximum: 127, zero: -101 },
        { dtype: VX_DTYPE_U8, minimum: 0, maximum: 255, zero: 231 },
      ]) {
        const transformed = [];
        for (let base = descriptor.minimum - 4; base <= descriptor.maximum + 4; base++) {
          transformed.push(base, base + 0.25, base + 0.5, base + 0.75);
        }
        const input = Float32Array.from(transformed,
          (value) => value - descriptor.zero);
        const result = harness.run(input, descriptor.dtype, 1, descriptor.zero);
        assert.equal(result.simdStatus, 1);
        assert.equal(result.scalarStatus, 1);
        assert.deepEqual(result.simd, result.scalar);
        assert.deepEqual(result.simd, transformed.map((value) => expectedFromTransformed(
          value, descriptor.minimum, descriptor.maximum, descriptor.zero,
        )));
        assert.equal(result.blocks, Math.floor(input.length / 16));
      }
    });

    await t.test('NaNs map to zero-point and infinities saturate for every tail length', () => {
      const edges = Float32Array.of(
        Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
        0, -0, 0.5, -0.5, 1.5, -1.5,
        3.4028234663852886e38, -3.4028234663852886e38,
        1.401298464324817e-45, -1.401298464324817e-45,
      );
      for (const descriptor of [
        { dtype: VX_DTYPE_I8, zero: -17, minimum: -128, maximum: 127 },
        { dtype: VX_DTYPE_U8, zero: 139, minimum: 0, maximum: 255 },
      ]) {
        for (let length = 0; length <= 47; length++) {
          const input = Float32Array.from({ length }, (_, index) => edges[index % edges.length]);
          const result = harness.run(input, descriptor.dtype, 1, descriptor.zero);
          assert.equal(result.simdStatus, 1);
          assert.equal(result.scalarStatus, 1);
          assert.deepEqual(result.simd, result.scalar, `${descriptor.dtype} length ${length}`);
          assert.equal(result.blocks, Math.floor(length / 16));
          if (length >= 3) {
            assert.deepEqual(result.simd.slice(0, 3), [
              descriptor.zero, descriptor.maximum, descriptor.minimum,
            ]);
          }
        }
      }
    });

    await t.test('100003 raw F32 bit patterns match over ordinary and extreme scales', () => {
      const input = new Float32Array(100_003);
      const bits = new Uint32Array(input.buffer);
      let state = 0x243f6a88;
      for (let index = 0; index < bits.length; index++) {
        state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
        bits[index] = state;
      }
      input.set([
        Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
        -0, 0, 0.5, -0.5, 126.5, 127.5, -127.5, -128.5,
      ]);
      for (const descriptor of [
        { dtype: VX_DTYPE_I8, zero: -23 },
        { dtype: VX_DTYPE_U8, zero: 129 },
      ]) {
        for (const scale of [
          1, 0.5, 0.03125, 1.401298464324817e-45,
          Math.fround(Math.PI), 3.4028234663852886e38,
        ]) {
          const result = harness.run(input, descriptor.dtype, scale, descriptor.zero);
          assert.equal(result.simdStatus, 1);
          assert.equal(result.scalarStatus, 1);
          assert.deepEqual(result.simd, result.scalar,
            `${descriptor.dtype} scale ${scale}`);
          assert.equal(result.blocks, Math.floor(input.length / 16));
        }
      }
    });

    await t.test('absent zero-point stays zero and invalid descriptors write nothing', () => {
      const values = Float32Array.from({ length: 33 }, (_, index) => index - 16.5);
      for (const dtype of [VX_DTYPE_I8, VX_DTYPE_U8]) {
        const result = harness.run(values, dtype, 1, 0, { includeZero: false });
        assert.equal(result.simdStatus, 1);
        assert.equal(result.scalarStatus, 1);
        assert.deepEqual(result.simd, result.scalar);
        assert.equal(result.blocks, 2);
      }

      const valid = harness.run(values, VX_DTYPE_I8, 1, -3);
      const { input, scale, zero, simd } = valid.pointers;
      for (const invalidScale of [0, -0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
        new Float32Array(api.memory.buffer, scale, 1)[0] = invalidScale;
        new Uint8Array(api.memory.buffer, simd, values.length).fill(0x6d);
        api.reset_quantize_linear_wasm_simd_blocks();
        assert.equal(api.quantize_linear_typed(
          input, scale, zero, VX_DTYPE_I8, simd, VX_DTYPE_I8, values.length,
        ), 0);
        assert.deepEqual(Array.from(new Uint8Array(api.memory.buffer, simd, values.length)),
          new Array(values.length).fill(0x6d));
        assert.equal(api.quantize_linear_wasm_simd_blocks(), 0);
      }
      new Float32Array(api.memory.buffer, scale, 1)[0] = 1;
      assert.equal(api.quantize_linear_typed(
        input, scale, zero, VX_DTYPE_U8, simd, VX_DTYPE_I8, values.length,
      ), 0, 'mismatched zero-point dtype');
      assert.equal(api.quantize_linear_typed(
        input, scale, zero, VX_DTYPE_I8, simd, 18, values.length,
      ), 0, 'unsupported F32 output');
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
