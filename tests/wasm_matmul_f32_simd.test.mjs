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

function importsFor(module) {
  const imports = {};
  for (const descriptor of WebAssembly.Module.imports(module)) {
    assert.equal(descriptor.kind, 'function',
      `unexpected ${descriptor.kind} import ${descriptor.module}.${descriptor.name}`);
    imports[descriptor.module] ||= {};
    imports[descriptor.module][descriptor.name] = () => 0;
  }
  return imports;
}

async function instantiate(path) {
  const module = await WebAssembly.compile(await readFile(path));
  const { exports: api } = await WebAssembly.instantiate(module, importsFor(module));
  assert.ok(api.memory instanceof WebAssembly.Memory);
  return api;
}

function makeAllocator(api) {
  return (bytes) => {
    const pointer = Number(api.alloc_bytes(bytes));
    assert.ok(Number.isInteger(pointer) && pointer > 0,
      `failed to allocate ${bytes} bytes`);
    const end = pointer + bytes;
    if (end > api.memory.buffer.byteLength) {
      api.memory.grow(Math.ceil((end - api.memory.buffer.byteLength) / 65536));
    }
    return pointer;
  };
}

function writeF32(api, allocate, values) {
  const pointer = allocate(values.byteLength);
  new Float32Array(api.memory.buffer, pointer, values.length).set(values);
  return pointer;
}

function values(length, salt) {
  return Float32Array.from({ length }, (_, index) =>
    (((index * (17 + salt) + salt * 13) % 251) - 125) / 127);
}

function execute(api, { m, k, n, bias }) {
  api.reset_heap();
  const allocate = makeAllocator(api);
  const a = writeF32(api, allocate, values(m * k, 1));
  const b = writeF32(api, allocate, values(k * n, 3));
  const biasPointer = bias ? writeF32(api, allocate, values(n, 7)) : 0;
  const output = allocate(m * n * Float32Array.BYTES_PER_ELEMENT);
  api.matmul_f32(a, b, biasPointer, output, m, k, n);
  return new Uint8Array(
    api.memory.buffer, output, m * n * Float32Array.BYTES_PER_ELEMENT,
  ).slice();
}

test('WASM SIMD128 K-major MatMul is bit-exact and keeps ragged fallbacks', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-matmul-f32-'));
  try {
    const scalarPath = join(directory, 'scalar.wasm');
    const simdPath = join(directory, 'simd.wasm');
    await Promise.all([
      buildWasm(scalarPath, ['-DVOLVOXAI_DISABLE_MATMUL_WASM_SIMD=1']),
      buildWasm(simdPath, ['-DVOLVOXAI_MATMUL_WASM_SIMD_TESTING=1']),
    ]);
    const [scalar, simd] = await Promise.all([
      instantiate(scalarPath), instantiate(simdPath),
    ]);
    assert.equal(typeof simd.matmul_wasm_simd_calls, 'function');
    assert.equal(typeof simd.reset_matmul_wasm_simd_calls, 'function');

    const cases = [
      { m: 37, k: 29, n: 40, bias: false, expectedSimdCalls: 1 },
      { m: 5, k: 17, n: 13, bias: true, expectedSimdCalls: 1 },
      { m: 3, k: 5, n: 8, bias: true, expectedSimdCalls: 1 },
      { m: 7, k: 9, n: 7, bias: false, expectedSimdCalls: 0 },
    ];
    for (const spec of cases) {
      simd.reset_matmul_wasm_simd_calls();
      const expected = execute(scalar, spec);
      const actual = execute(simd, spec);
      assert.deepEqual(actual, expected,
        `M=${spec.m}, K=${spec.k}, N=${spec.n}, bias=${spec.bias}`);
      assert.equal(simd.matmul_wasm_simd_calls(), spec.expectedSimdCalls,
        `M=${spec.m}, K=${spec.k}, N=${spec.n} dispatch count`);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
