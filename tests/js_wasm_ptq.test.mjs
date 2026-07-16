import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import {
  WASM_PTQ_ABI_VERSION,
  WasmEngine,
  WasmPTQ,
  WasmVolvoxAI,
  createPTQ,
  createWasmPTQ,
} from '../ts/wasm.js';

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

async function compileStablePTQAbi(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--allow-undefined',
    '-Wl,--export=alloc_bytes', '-Wl,--export=reset_heap', '-Wl,--export-memory',
    '-Inative/include', '-o', output,
    'native/src/kernels/core.c',
    'native/src/training/quantization.c',
  ], { cwd: repositoryRoot, maxBuffer: 1024 * 1024 });
}

function close(actual, expected, tolerance = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance * (1 + Math.abs(actual) + Math.abs(expected)),
    `${actual} != ${expected}`,
  );
}

test('WasmPTQ exposes generic C PTQ through an isolated scratch instance', {
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

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-ptq-'));
  try {
    const inferencePath = join(directory, 'volvoxai.wasm');
    const fullPath = join(directory, 'volvoxai.full.wasm');
    const stableAbiPath = join(directory, 'volvoxai.ptq-abi.wasm');
    await Promise.all([
      compileWasm(inferencePath, ['native/src/kernels/kernels.c']),
      compileWasm(fullPath, [
        'native/src/kernels/kernels.c',
        'native/src/kernels/training_kernels.c',
        'native/src/training/quantization.c',
      ]),
      compileStablePTQAbi(stableAbiPath),
    ]);

    const inferenceModule = await WebAssembly.compile(await readFile(inferencePath));
    const fullModule = await WebAssembly.compile(await readFile(fullPath));
    const stableAbiModule = await WebAssembly.compile(await readFile(stableAbiPath));
    const inferenceNames = new Set(
      WebAssembly.Module.exports(inferenceModule).map(({ name }) => name),
    );
    const fullNames = new Set(WebAssembly.Module.exports(fullModule).map(({ name }) => name));
    assert.deepEqual(
      [...inferenceNames].filter((name) => name.startsWith('volvoxai_ptq_')),
      [],
    );
    const ptqExports = [
      'volvoxai_ptq_abi_version',
      'volvoxai_ptq_capabilities',
      'volvoxai_ptq_observer_reset',
      'volvoxai_ptq_observer_observe_f32',
      'volvoxai_ptq_calculate_params',
      'volvoxai_ptq_quantize_f32',
      'volvoxai_ptq_pack_weight_i8',
      'volvoxai_ptq_pack_bias_i32',
    ].sort();
    for (const name of ptqExports) {
      assert.equal(fullNames.has(name), true, `full WASM is missing ${name}`);
    }
    assert.deepEqual(
      WebAssembly.Module.exports(stableAbiModule)
        .map(({ name }) => name)
        .filter((name) => name.startsWith('volvoxai_ptq_'))
        .sort(),
      ptqExports,
      'the stable PTQ ABI must not depend on --export-all',
    );

    const engine = await WasmEngine.init(fullPath);
    assert.ok(engine);
    const sentinelPointer = Number(engine.api.alloc_bytes(64));
    const sentinel = new Uint8Array(engine.mem.buffer, sentinelPointer, 16);
    sentinel.set([1, 3, 5, 7, 9, 11, 13, 15]);

    const originalFork = engine.fork.bind(engine);
    let ptqForkOptions = null;
    engine.fork = async (options) => {
      ptqForkOptions = options;
      return originalFork(options);
    };
    const ptq = await createWasmPTQ(engine);
    assert.ok(ptq instanceof WasmPTQ);
    assert.deepEqual(ptqForkOptions, { relaxedSimd: false });
    assert.equal(ptq.abiVersion, WASM_PTQ_ABI_VERSION);
    assert.equal(ptq.capabilities, 0x1f);
    assert.equal(typeof createPTQ, 'function');
    assert.equal(typeof WasmVolvoxAI.prototype.createPTQ, 'function');
    assert.equal(typeof WasmVolvoxAI.prototype.createWasmPTQ, 'function');

    const observer = ptq.createObserver();
    observer.observe(Float32Array.of(-2, 1));
    observer.observe(Float32Array.of(3, -1));
    assert.deepEqual(observer.snapshot(), { minimum: -2, maximum: 3, sampleCount: 4 });
    const beforeInvalid = observer.snapshot();
    assert.throws(
      () => observer.observe(Float32Array.of(1, Number.NaN)),
      /non-finite/,
    );
    assert.deepEqual(observer.snapshot(), beforeInvalid);

    const symmetric = observer.parameters({ dtype: 'int8', scheme: 'symmetric' });
    assert.equal(symmetric.zero_point, 0);
    close(symmetric.scale, Math.fround(3 / 127));
    const quantizedI8 = ptq.quantize(Float32Array.of(-3, 0, 3), symmetric);
    assert.ok(quantizedI8.data instanceof Int8Array);
    assert.deepEqual([...quantizedI8.data], [-127, 0, 127]);
    assert.equal(quantizedI8.saturationCount, 0);

    const unitScale = ptq.parameters(
      { minimum: -127, maximum: 127, sampleCount: 2 },
      { dtype: 'int8', scheme: 'symmetric' },
    );
    const fullRange = ptq.quantize(Float32Array.of(-129, -128, 127, 128), unitScale);
    assert.deepEqual([...fullRange.data], [-128, -128, 127, 127]);
    assert.equal(fullRange.saturationCount, 2,
      'generic C/WASM I8 quantization uses the full physical [-128,127] range');

    const asymmetric = ptq.parameters(
      { minimum: -1, maximum: 3, sampleCount: 4 },
      { dtype: 'uint8', scheme: 'asymmetric' },
    );
    assert.equal(asymmetric.zero_point, 64);
    close(asymmetric.scale, Math.fround(4 / 255));
    const quantizedU8 = ptq.quantize(Float32Array.of(-1, 0, 3), asymmetric);
    assert.ok(quantizedU8.data instanceof Uint8Array);
    assert.deepEqual([...quantizedU8.data], [0, 64, 255]);

    // Rank-3 axis -2 maps channels across both outer batches and inner pairs.
    const packed = ptq.packWeight(
      Float32Array.of(-1, 1, -2, 2, -0.5, 0.5, -4, 4),
      [2, 2, 2],
      { axis: -2, name: 'adapter.weight' },
    );
    assert.equal(packed.name, 'adapter.weight');
    assert.deepEqual(packed.shape, [2, 2, 2]);
    assert.equal(packed.quantization.axis, 1);
    assert.deepEqual(packed.quantization.zero_points, [0, 0]);
    close(packed.scales[0], Math.fround(1 / 127));
    close(packed.scales[1], Math.fround(4 / 127));
    assert.deepEqual([...packed.data], [-127, 127, -64, 64, -64, 64, -127, 127]);
    assert.equal(packed.saturationCount, 0);

    const narrow = ptq.packWeight(Float32Array.of(-128, 128), [1, 2]);
    assert.deepEqual([...narrow.data], [-127, 127],
      'symmetric C/WASM weights reserve -128 and use narrow range [-127,127]');

    assert.deepEqual(
      [...ptq.packBias(
        Float32Array.of(0.125, -0.5),
        0.5,
        Float32Array.of(0.25, 0.5),
      )],
      [1, -2],
    );
    assert.throws(
      () => ptq.packWeight(Float32Array.of(1, 2), [1, 2], { axis: 2 }),
      /outside rank/,
    );

    // Scratch resets must not reset or overwrite the canonical graph heap.
    assert.deepEqual([...sentinel.slice(0, 8)], [1, 3, 5, 7, 9, 11, 13, 15]);
    const nextCanonicalPointer = Number(engine.api.alloc_bytes(16));
    assert.ok(nextCanonicalPointer >= sentinelPointer + 64);

    const snapshotBeforeDispose = observer.snapshot();
    ptq.dispose();
    ptq.dispose();
    assert.deepEqual(observer.snapshot(), snapshotBeforeDispose,
      'caller-owned observer state remains readable after disposal');
    assert.deepEqual([...packed.data], [-127, 127, -64, 64, -64, 64, -127, 127],
      'caller-owned PTQ results remain valid after disposal');
    assert.throws(() => ptq.createObserver(), /disposed/);
    assert.throws(() => observer.reset(), /disposed/);
    assert.throws(() => observer.observe(new Float32Array()), /disposed/);
    assert.throws(() => observer.parameters(), /disposed/);
    assert.throws(() => ptq.quantize(new Float32Array(), symmetric), /disposed/);
    assert.throws(
      () => ptq.packWeight(Float32Array.of(1), [1]),
      /disposed/,
    );
    assert.throws(
      () => ptq.packBias(new Float32Array(), 1, new Float32Array()),
      /disposed/,
    );

    const inferenceEngine = await WasmEngine.init(inferencePath);
    assert.ok(inferenceEngine);
    await assert.rejects(
      createWasmPTQ(inferenceEngine),
      /missing PTQ export/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
