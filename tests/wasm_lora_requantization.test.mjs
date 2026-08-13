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

async function compileWasm(output, sources) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-Inative/include', '-o', output, ...sources,
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
}

function importsFor(module) {
  const imports = {};
  for (const descriptor of WebAssembly.Module.imports(module)) {
    assert.equal(descriptor.kind, 'function');
    imports[descriptor.module] ||= {};
    imports[descriptor.module][descriptor.name] = () => 0;
  }
  return imports;
}

function allocate(api, bytes) {
  const pointer = Number(api.alloc_bytes(bytes));
  assert.ok(pointer > 0, `failed to allocate ${bytes} WASM bytes`);
  return pointer;
}

function closeArray(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `index ${index}: ${actual[index]} != ${expected[index]}`);
  }
}

test('full WASM reuses native PTQ math for transpose-aware F32 LoRA synchronization', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      t.skip(`${clang} is unavailable`);
      return;
    }
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-lora-requantize-'));
  try {
    const inferencePath = join(directory, 'volvoxai.wasm');
    const fullPath = join(directory, 'volvoxai.full.wasm');
    await compileWasm(inferencePath, ['native/src/kernels/kernels.c']);
    await compileWasm(fullPath, [
      'native/src/kernels/kernels.c',
      'native/src/kernels/training_kernels.c',
      'native/src/training/quantization.c',
    ]);

    const inferenceModule = await WebAssembly.compile(await readFile(inferencePath));
    const fullModule = await WebAssembly.compile(await readFile(fullPath));
    const inferenceNames = new Set(WebAssembly.Module.exports(inferenceModule).map(({ name }) => name));
    assert.equal(inferenceNames.has('volvoxai_training_quantize_weight_f32_to_i8'), false);
    assert.equal(inferenceNames.has('volvoxai_training_dequantize_weight_i8_to_f32'), false);
    assert.equal([...inferenceNames].some((name) => name.startsWith('volvoxai_ptq_')), false);

    const { exports: api } = await WebAssembly.instantiate(fullModule, importsFor(fullModule));
    assert.equal(typeof api.volvoxai_ptq_pack_weight_i8, 'function');
    assert.equal(typeof api.volvoxai_training_quantize_weight_f32_to_i8, 'function');
    assert.equal(typeof api.volvoxai_training_dequantize_weight_i8_to_f32, 'function');

    const sourcePointer = allocate(api, 6 * Float32Array.BYTES_PER_ELEMENT);
    const quantizedPointer = allocate(api, 6);
    const scalesPointer = allocate(api, 2 * Float32Array.BYTES_PER_ELEMENT);
    const restoredPointer = allocate(api, 6 * Float32Array.BYTES_PER_ELEMENT);
    const saturationPointer = allocate(api, BigUint64Array.BYTES_PER_ELEMENT);

    // Canonical OUT_IN rows [-1,0,1] and [-2,0,2], physically stored as the
    // IN_OUT master [-1,-2, 0,0, 1,2].
    new Float32Array(api.memory.buffer, sourcePointer, 6).set([-1, -2, 0, 0, 1, 2]);
    assert.equal(api.volvoxai_training_quantize_weight_f32_to_i8(
      sourcePointer, quantizedPointer, scalesPointer,
      2, 3, 1, 0, saturationPointer,
    ), 1);
    assert.deepEqual([...new Int8Array(api.memory.buffer, quantizedPointer, 6)],
      [-127, 0, 127, -127, 0, 127]);
    closeArray(new Float32Array(api.memory.buffer, scalesPointer, 2), [1 / 127, 2 / 127]);
    assert.equal(new BigUint64Array(api.memory.buffer, saturationPointer, 1)[0], 0n);

    assert.equal(api.volvoxai_training_dequantize_weight_i8_to_f32(
      quantizedPointer, scalesPointer, restoredPointer, 2, 3, 1,
    ), 1);
    closeArray(new Float32Array(api.memory.buffer, restoredPointer, 6), [-1, -2, 0, 0, 1, 2]);

    // Preserve policy leaves descriptor scales untouched and reports clipping.
    new Float32Array(api.memory.buffer, sourcePointer, 6).set([-100, -0.25, 0.25, 100, 1, 2]);
    new Float32Array(api.memory.buffer, scalesPointer, 2).set([0.5, 1]);
    assert.equal(api.volvoxai_training_quantize_weight_f32_to_i8(
      sourcePointer, quantizedPointer, scalesPointer,
      2, 3, 0, 1, saturationPointer,
    ), 1);
    assert.deepEqual([...new Int8Array(api.memory.buffer, quantizedPointer, 6)],
      [-127, 0, 0, 100, 1, 2]);
    assert.deepEqual([...new Float32Array(api.memory.buffer, scalesPointer, 2)], [0.5, 1]);
    assert.equal(new BigUint64Array(api.memory.buffer, saturationPointer, 1)[0], 1n);

    // Recompute validates every source value before mutating either output.
    new Float32Array(api.memory.buffer, sourcePointer, 6).set([1, 2, Number.NaN, 4, 5, 6]);
    new Int8Array(api.memory.buffer, quantizedPointer, 6).fill(7);
    new Float32Array(api.memory.buffer, scalesPointer, 2).set([11, 13]);
    assert.equal(api.volvoxai_training_quantize_weight_f32_to_i8(
      sourcePointer, quantizedPointer, scalesPointer,
      2, 3, 0, 0, saturationPointer,
    ), 0);
    assert.deepEqual([...new Int8Array(api.memory.buffer, quantizedPointer, 6)], [7, 7, 7, 7, 7, 7]);
    assert.deepEqual([...new Float32Array(api.memory.buffer, scalesPointer, 2)], [11, 13]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
