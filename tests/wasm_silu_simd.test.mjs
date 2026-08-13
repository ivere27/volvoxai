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

async function buildWasm(directory) {
  const output = join(directory, 'silu-simd.wasm');
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_SILU_TESTING=1',
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

function makeHarness(api, capacity) {
  api.reset_heap();
  // Four-byte aligned but deliberately not SIMD-aligned.
  const input = Number(api.alloc_bytes(capacity * 4 + 32)) + 4;
  const simd = Number(api.alloc_bytes(capacity * 4 + 32)) + 4;
  const scalar = Number(api.alloc_bytes(capacity * 4 + 32)) + 8;
  const requiredBytes = scalar + capacity * 4;
  if (api.memory.buffer.byteLength < requiredBytes) {
    api.memory.grow(Math.ceil(
      (requiredBytes - api.memory.buffer.byteLength) / 65_536,
    ));
  }
  return {
    compareBits(inputBits) {
      assert.ok(inputBits.length <= capacity);
      new Uint32Array(api.memory.buffer, input, inputBits.length).set(inputBits);
      new Uint32Array(api.memory.buffer, simd, inputBits.length).fill(0xa5a5a5a5);
      new Uint32Array(api.memory.buffer, scalar, inputBits.length).fill(0x5a5a5a5a);
      api.reset_silu_wasm_simd_blocks();
      api.silu_f32(input, simd, inputBits.length);
      const blocks = Number(api.silu_wasm_simd_blocks());
      api.silu_f32_scalar_reference(input, scalar, inputBits.length);
      const actual = Array.from(
        new Uint32Array(api.memory.buffer, simd, inputBits.length),
      );
      const expected = Array.from(
        new Uint32Array(api.memory.buffer, scalar, inputBits.length),
      );
      assert.deepEqual(actual, expected);
      return blocks;
    },
  };
}

function f32Bits(values) {
  const storage = new ArrayBuffer(values.length * 4);
  new Float32Array(storage).set(values);
  return Array.from(new Uint32Array(storage));
}

test('WASM SIMD SiLU is bit-exact with the canonical scalar polynomial', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-silu-simd-'));
  try {
    const api = await instantiate(await buildWasm(directory));
    assert.equal(typeof api.silu_f32, 'function');
    assert.equal(typeof api.silu_f32_scalar_reference, 'function');
    const harness = makeHarness(api, 100_128);

    const boundaryValues = [
      -0, 0, -1, 1, -87, 87, -88, 88, -89, 89,
      -1e-30, 1e-30, -3.4028234663852886e38, 3.4028234663852886e38,
      -12.5, 12.5, -0.5, 0.5, -6, 6, -20, 20, -64, 64,
      -86.99999, 86.99999, -87.00001, 87.00001, -88.00001, 88.00001,
      -1000, 1000,
    ];
    assert.equal(harness.compareBits(f32Bits(boundaryValues)), 2);

    // Exercise every tail while still proving that a complete SIMD prefix ran.
    for (let length = 16; length < 48; length++) {
      const values = Array.from({ length }, (_, index) =>
        Math.fround(((index * 37) % 101 - 50) / 7));
      assert.equal(harness.compareBits(f32Bits(values)), Math.floor(length / 16));
    }

    // Sample raw finite F32 encodings across signs, exponents and mantissas.
    const finiteBits = [];
    let state = 0x6d2b79f5;
    while (finiteBits.length < 100_003) {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
      if ((state & 0x7f80_0000) !== 0x7f80_0000) finiteBits.push(state);
    }
    assert.equal(
      harness.compareBits(finiteBits),
      Math.floor(finiteBits.length / 16),
    );

    // A non-finite lane makes that block and the remainder use the exact scalar
    // path, including infinities and NaN payload handling.
    const exceptional = f32Bits(Array.from({ length: 48 }, (_, index) => index - 24));
    exceptional[18] = 0x7f80_0000;
    exceptional[19] = 0xff80_0000;
    exceptional[20] = 0x7fc0_1234;
    assert.equal(harness.compareBits(exceptional), 1);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
