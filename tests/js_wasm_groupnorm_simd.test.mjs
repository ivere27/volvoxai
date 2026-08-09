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

async function buildWasm(output, definitions) {
  await run(clang, [
    '--target=wasm32', '-std=c11', '-O3', '-msimd128', '-nostdlib',
    ...definitions,
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
}

async function instantiate(path) {
  const environment = {
    expf: Math.exp,
    logf: Math.log,
    powf: Math.pow,
    sqrtf: Math.sqrt,
    tanhf: Math.tanh,
    sinf: Math.sin,
    cosf: Math.cos,
  };
  const bytes = await readFile(path);
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: environment,
    math: environment,
  });
  return instance.exports;
}

function allocate(api, bytes) {
  const pointer = Number(api.alloc_bytes(bytes));
  assert.ok(Number.isSafeInteger(pointer) && pointer > 0);
  const required = pointer + bytes;
  if (required > api.memory.buffer.byteLength) {
    api.memory.grow(Math.ceil((required - api.memory.buffer.byteLength) / 65_536));
  }
  return pointer;
}

function prepare(api, specification) {
  api.reset_heap();
  const elements = specification.batch * specification.height *
    specification.width * specification.channels;
  const input = allocate(api, elements * Float32Array.BYTES_PER_ELEMENT);
  const weight = allocate(api, specification.channels * Float32Array.BYTES_PER_ELEMENT);
  const bias = specification.bias === false
    ? 0
    : allocate(api, specification.channels * Float32Array.BYTES_PER_ELEMENT);
  const output = allocate(api, elements * Float32Array.BYTES_PER_ELEMENT);
  const inputValues = new Float32Array(api.memory.buffer, input, elements);
  for (let index = 0; index < elements; index++) {
    inputValues[index] = specification.largeOffset
      ? Math.fround(10_000_000 + (index % 7))
      : Math.fround(((index * 37 + 19) % 509 - 254) / 31);
  }
  const weightValues = new Float32Array(
    api.memory.buffer, weight, specification.channels,
  );
  for (let channel = 0; channel < specification.channels; channel++) {
    weightValues[channel] = Math.fround(((channel * 13 + 7) % 43 - 21) / 19);
  }
  if (bias) {
    const biasValues = new Float32Array(
      api.memory.buffer, bias, specification.channels,
    );
    for (let channel = 0; channel < specification.channels; channel++) {
      biasValues[channel] = Math.fround(((channel * 11 + 5) % 31 - 15) / 17);
    }
  }
  return {
    input,
    weight,
    bias,
    output,
    elements,
    call(outputPointer = output, overrides = {}) {
      return api.groupnorm_f32(
        input, weight, bias, outputPointer,
        overrides.batch ?? specification.batch,
        overrides.height ?? specification.height,
        overrides.width ?? specification.width,
        overrides.channels ?? specification.channels,
        overrides.groups ?? specification.groups,
        overrides.epsilon ?? specification.epsilon ?? 1e-5,
      );
    },
  };
}

function outputBytes(api, prepared) {
  return new Uint8Array(
    api.memory.buffer,
    prepared.output,
    prepared.elements * Float32Array.BYTES_PER_ELEMENT,
  ).slice();
}

test('WASM SIMD128 GroupNorm is bit-exact for group, channel, and spatial tails', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-groupnorm-'));
  try {
    const scalarPath = join(directory, 'scalar.wasm');
    const simdPath = join(directory, 'simd.wasm');
    await Promise.all([
      buildWasm(scalarPath, ['-DVOLVOXAI_DISABLE_GROUPNORM_WASM_SIMD=1']),
      buildWasm(simdPath, ['-DVOLVOXAI_GROUPNORM_WASM_SIMD_TESTING=1']),
    ]);
    const [scalar, simd] = await Promise.all([
      instantiate(scalarPath), instantiate(simdPath),
    ]);
    assert.equal(typeof simd.groupnorm_wasm_simd_calls, 'function');
    assert.equal(typeof simd.reset_groupnorm_wasm_simd_calls, 'function');

    const cases = [
      { name: 'one value', batch: 1, height: 1, width: 1, channels: 1, groups: 1 },
      { name: 'odd groups and cpg', batch: 2, height: 3, width: 5, channels: 15, groups: 3 },
      { name: 'odd groups and even cpg', batch: 2, height: 2, width: 7, channels: 12, groups: 3 },
      { name: 'tile tail without bias', batch: 1, height: 1, width: 17, channels: 21, groups: 7, bias: false },
      { name: 'maximum streamed groups', batch: 1, height: 1, width: 2, channels: 512, groups: 512 },
      {
        name: 'large offset and sub-F32 epsilon',
        batch: 1,
        height: 1,
        width: 11,
        channels: 6,
        groups: 2,
        epsilon: 1e-50,
        largeOffset: true,
      },
    ];
    for (const specification of cases) {
      simd.reset_groupnorm_wasm_simd_calls();
      const expected = prepare(scalar, specification);
      const actual = prepare(simd, specification);
      assert.equal(expected.call(), 1, `${specification.name} scalar status`);
      assert.equal(actual.call(), 1, `${specification.name} SIMD status`);
      assert.deepEqual(outputBytes(simd, actual), outputBytes(scalar, expected),
        `${specification.name} output bits`);
      assert.equal(simd.groupnorm_wasm_simd_calls(), 1,
        `${specification.name} SIMD dispatch count`);
    }

    simd.reset_groupnorm_wasm_simd_calls();
    const wideGroups = {
      name: 'more than streamed group bound',
      batch: 1,
      height: 1,
      width: 2,
      channels: 513,
      groups: 513,
    };
    const expected = prepare(scalar, wideGroups);
    const actual = prepare(simd, wideGroups);
    assert.equal(expected.call(), 1);
    assert.equal(actual.call(), 1);
    assert.deepEqual(outputBytes(simd, actual), outputBytes(scalar, expected));
    assert.equal(simd.groupnorm_wasm_simd_calls(), 0,
      'groups beyond the bounded stack route retain the scalar fallback');

    simd.reset_groupnorm_wasm_simd_calls();
    const inPlaceSpec = {
      name: 'in-place input/output',
      batch: 2,
      height: 3,
      width: 5,
      channels: 15,
      groups: 3,
    };
    const inPlaceExpected = prepare(scalar, inPlaceSpec);
    const inPlaceActual = prepare(simd, inPlaceSpec);
    assert.equal(inPlaceExpected.call(), 1);
    assert.equal(inPlaceActual.call(inPlaceActual.input), 1);
    assert.deepEqual(
      new Uint8Array(
        simd.memory.buffer,
        inPlaceActual.input,
        inPlaceActual.elements * Float32Array.BYTES_PER_ELEMENT,
      ).slice(),
      outputBytes(scalar, inPlaceExpected),
      'the historical safe input/output alias remains bit-exact',
    );
    assert.equal(simd.groupnorm_wasm_simd_calls(), 1);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('WASM GroupNorm rejects invalid descriptors before writes', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-groupnorm-invalid-'));
  try {
    const wasmPath = join(directory, 'groupnorm.wasm');
    await buildWasm(wasmPath, []);
    const api = await instantiate(wasmPath);
    const specification = {
      batch: 1, height: 2, width: 3, channels: 6, groups: 3,
    };

    for (const overrides of [
      { groups: 4 },
      { epsilon: 0 },
      { epsilon: Number.POSITIVE_INFINITY },
      { height: 0 },
    ]) {
      const prepared = prepare(api, specification);
      new Uint32Array(api.memory.buffer, prepared.output, prepared.elements).fill(0xdeadbeef);
      const before = outputBytes(api, prepared);
      assert.equal(prepared.call(prepared.output, overrides), 0);
      assert.deepEqual(outputBytes(api, prepared), before,
        `invalid descriptor ${JSON.stringify(overrides)} must not write`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
