import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'esbuild';

const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const minimalMemoryWasm = Uint8Array.from([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x03, 0x01, 0x00, 0x01,
  0x07, 0x0a, 0x01, 0x06, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x02, 0x00,
]);

async function bundle(entryPoint, { browserOnly = false } = {}) {
  return build({
    entryPoints: [entryPoint],
    absWorkingDir: repositoryRoot,
    bundle: true,
    format: 'esm',
    target: ['es2022'],
    loader: { '.wgsl': 'text' },
    define: browserOnly ? { __VOLVOXAI_BROWSER_ONLY__: 'true' } : {},
    minifySyntax: browserOnly,
    metafile: true,
    write: false,
  });
}

function trainingInputs(result) {
  return Object.keys(result.metafile.inputs).filter((path) =>
    path.startsWith('ts/training/') || path.startsWith('shaders/training/'));
}

test('browser profiles stay monolithic and preserve their dependency boundaries', async () => {
  const [inference, full, wasm] = await Promise.all([
    bundle('ts/index.ts'),
    bundle('ts/full.ts'),
    bundle('ts/wasm.ts', { browserOnly: true }),
  ]);

  assert.equal(inference.outputFiles.length, 1);
  assert.equal(full.outputFiles.length, 1);
  assert.equal(wasm.outputFiles.length, 1);
  assert.deepEqual(trainingInputs(inference), []);
  const inferenceInputs = Object.keys(inference.metafile.inputs);
  for (const forbiddenInput of [
    'ts/core/RuntimeGraphLoader.ts',
    'ts/core/RuntimeGraphBuilder.ts',
  ]) {
    assert.equal(
      inferenceInputs.includes(forbiddenInput),
      false,
      `inference bundle unexpectedly contains legacy model input ${forbiddenInput}`,
    );
  }
  for (const forbiddenInput of [
    'ts/ops/dropout.ts',
    'ts/ops/attentionDropout.ts',
  ]) {
    assert.equal(
      Object.keys(inference.metafile.inputs).includes(forbiddenInput),
      false,
      `inference bundle unexpectedly contains training RNG module ${forbiddenInput}`,
    );
  }
  assert.equal(
    Object.keys(inference.metafile.inputs).some((path) => path.startsWith('examples/')),
    false,
  );
  assert.equal(
    Object.keys(full.metafile.inputs).some((path) => path.startsWith('examples/')),
    false,
  );
  assert.ok(trainingInputs(full).includes('ts/training/Trainer.ts'));
  assert.ok(trainingInputs(full).includes('ts/training/Initializers.ts'));
  assert.ok(trainingInputs(full).includes('shaders/training/matMulBackward.wgsl'));

  const wasmInputs = Object.keys(wasm.metafile.inputs);
  for (const requiredInput of [
    'ts/backends/WasmEngine.ts',
    'ts/training/AcceleratedAutograd.ts',
    'ts/training/TrainingGraph.ts',
    'ts/training/WasmAutograd.ts',
    'ts/training/WasmTrainer.ts',
    'ts/training/WasmPTQ.ts',
    'ts/training/WasmTrainingKernels.ts',
  ]) {
    assert.ok(
      wasmInputs.includes(requiredInput),
      `WASM-only bundle is missing ${requiredInput}`,
    );
  }
  for (const forbiddenInput of [
    'ts/backends/CPUEngine.ts',
    'ts/backends/GraphExecutor.ts',
    'ts/backends/WebGPUEngine.ts',
    'ts/backends/WebNNEngine.ts',
    'ts/training/CPUAutograd.ts',
    'ts/training/Quantization.ts',
    'ts/training/Trainer.ts',
    'ts/training/TrainingShaderLibrary.ts',
    'ts/training/WebGPUAutograd.ts',
    'ts/training/WasmQuantizedLoRATrainer.ts',
  ]) {
    assert.equal(
      wasmInputs.includes(forbiddenInput),
      false,
      `WASM-only bundle unexpectedly contains ${forbiddenInput}`,
    );
  }
  assert.equal(
    wasmInputs.some((path) => path.startsWith('shaders/') || path.endsWith('.wgsl')),
    false,
    'WASM-only bundle unexpectedly contains WGSL',
  );
  assert.equal(
    wasmInputs.some((path) => path.startsWith('examples/')),
    false,
    'WASM-only bundle unexpectedly contains example code',
  );

  const inferenceSource = inference.outputFiles[0].text;
  const fullSource = full.outputFiles[0].text;
  const wasmSource = wasm.outputFiles[0].text;
  for (const trainingHook of ['prepareForTraining', 'pipelineOverrides', 'dropoutMultiplier']) {
    assert.equal(
      inferenceSource.includes(trainingHook),
      false,
      `inference bundle unexpectedly contains training hook ${trainingHook}`,
    );
  }
  assert.equal(
    /\bimport\s*\(/.test(wasmSource),
    false,
    'browser-only WASM bundle unexpectedly contains a dynamic import',
  );
  for (const trainingSymbol of [
    'optimizerState',
    'optimizerDescriptor',
    'trainingStep',
    'trainingMetadata',
    'initializeTensor',
  ]) {
    assert.equal(
      inferenceSource.includes(trainingSymbol),
      false,
      `inference bundle unexpectedly contains ${trainingSymbol}`,
    );
    assert.equal(
      fullSource.includes(trainingSymbol),
      true,
      `full bundle is missing ${trainingSymbol}`,
    );
  }
});

test('package exports use readable defaults and explicit minified variants', async () => {
  const packageJson = JSON.parse(
    await readFile(new URL('../package.json', import.meta.url), 'utf8'),
  );
  const root = `./dist/${packageJson.version}`;
  assert.equal(packageJson.main, `${root}/volvoxai.js`);
  assert.equal(packageJson.exports['.'], `${root}/volvoxai.js`);
  assert.equal(packageJson.exports['./min'], `${root}/volvoxai.min.js`);
  assert.equal(packageJson.exports['./full'], `${root}/volvoxai.full.js`);
  assert.equal(packageJson.exports['./full/min'], `${root}/volvoxai.full.min.js`);
  assert.equal(packageJson.exports['./wasm'], `${root}/volvoxai.wasm.js`);
  assert.equal(packageJson.exports['./wasm/min'], `${root}/volvoxai.wasm.min.js`);
  assert.equal(packageJson.exports['./volvoxai.wasm'], `${root}/volvoxai.wasm`);
  assert.equal(packageJson.exports['./volvoxai.full.wasm'], `${root}/volvoxai.full.wasm`);
  assert.deepEqual(packageJson.files, ['bin/', `dist/${packageJson.version}/`]);
});

test('the shipped CLI uses only the logical v1 lifecycle', async () => {
  const source = await readFile(new URL('../bin/volvox.js', import.meta.url), 'utf8');
  assert.match(source, /\bModelLoader\.load\s*\(modelUrl\.href\)/);
  assert.match(source, /\bModel\.capture\s*\(/);
  assert.match(source, /\bruntime\.compile\s*\(/);
  assert.doesNotMatch(source, /\bruntime\.loadModel\s*\(/);
  assert.doesNotMatch(source, /\bmodel\.compile\s*\(/);
});

test('the bundled inference entry resolves its adjacent WASM sidecar', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-profile-'));
  try {
    const bundlePath = join(directory, 'volvoxai.mjs');
    await build({
      entryPoints: ['ts/index.ts'],
      absWorkingDir: repositoryRoot,
      bundle: true,
      format: 'esm',
      loader: { '.wgsl': 'text' },
      outfile: bundlePath,
    });
    // Minimal valid module exporting one memory. Initialization does not need
    // operator exports until a graph is compiled.
    await writeFile(join(directory, 'volvoxai.wasm'), minimalMemoryWasm);

    const module = await import(`${pathToFileURL(bundlePath).href}?test=${Date.now()}`);
    for (const hidden of [
      'TrainingGraph', 'TrainingModelBuilder', 'WasmTrainer',
      'CPUAutograd', 'WebGPUAutograd', 'WasmAutograd',
      'WasmQuantizedLoRATrainer', 'accumulateGradients', 'resolveTrainingOptimizer',
      'WasmPTQ', 'WasmPTQObserver', 'createWasmPTQ',
      'normalizeCrossEntropyLosses', 'crossEntropyGradient', 'addGradient',
      'AdapterManager', 'ModelSnapshot', 'runtimeError', 'parseStrictJSON',
    ]) assert.equal(module[hidden], undefined);
    assert.equal(module.Trainer, undefined);
    const runtime = await module.VolvoxAI.createRuntime({ backends: ['wasm'] });
    assert.deepEqual(runtime.listBackends(), ['wasm']);
    await runtime.close();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('the bundled full entry resolves its adjacent full WASM sidecar', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-full-wasm-profile-'));
  try {
    const bundlePath = join(directory, 'volvoxai.full.mjs');
    await build({
      entryPoints: ['ts/full.ts'],
      absWorkingDir: repositoryRoot,
      bundle: true,
      format: 'esm',
      loader: { '.wgsl': 'text' },
      outfile: bundlePath,
    });
    await writeFile(join(directory, 'volvoxai.full.wasm'), minimalMemoryWasm);

    const module = await import(`${pathToFileURL(bundlePath).href}?test=${Date.now()}`);
    for (const hidden of [
      'TrainingGraph', 'TrainingModelBuilder', 'WasmTrainer',
      'CPUAutograd', 'WebGPUAutograd', 'WasmAutograd',
      'WasmQuantizedLoRATrainer', 'accumulateGradients', 'resolveTrainingOptimizer',
      'WasmPTQ', 'WasmPTQObserver', 'createWasmPTQ',
      'normalizeCrossEntropyLosses', 'crossEntropyGradient', 'addGradient',
      'AdapterManager', 'ModelSnapshot', 'runtimeError', 'parseStrictJSON',
    ]) assert.equal(module[hidden], undefined);
    assert.equal(typeof module.Trainer.create, 'function');
    const runtime = await module.VolvoxAI.createRuntime({ backends: ['wasm'] });
    assert.deepEqual(runtime.listBackends(), ['wasm']);
    await runtime.close();
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('the browser-only WASM entry resolves its adjacent full WASM sidecar', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-only-profile-'));
  const originalFetch = globalThis.fetch;
  try {
    const bundlePath = join(directory, 'volvoxai.wasm.mjs');
    await build({
      entryPoints: ['ts/wasm.ts'],
      absWorkingDir: repositoryRoot,
      bundle: true,
      format: 'esm',
      target: ['es2022'],
      loader: { '.wgsl': 'text' },
      define: { __VOLVOXAI_BROWSER_ONLY__: 'true' },
      minifySyntax: true,
      outfile: bundlePath,
    });

    let fetchedUrl = null;
    globalThis.fetch = async (url) => {
      fetchedUrl = new URL(url);
      return {
        ok: true,
        arrayBuffer: async () => minimalMemoryWasm.slice().buffer,
      };
    };

    const module = await import(`${pathToFileURL(bundlePath).href}?test=${Date.now()}`);
    for (const hidden of [
      'TrainingGraph', 'TrainingModelBuilder', 'WasmTrainer',
      'CPUAutograd', 'WebGPUAutograd', 'WasmAutograd',
      'WasmQuantizedLoRATrainer', 'accumulateGradients', 'resolveTrainingOptimizer',
      'WasmPTQ', 'WasmPTQObserver', 'createWasmPTQ',
      'normalizeCrossEntropyLosses', 'crossEntropyGradient', 'addGradient',
      'executionIdentity', 'normalizeBackendReport', 'releaseBackendExecutionSnapshot',
      'createExecutionResult',
      'cloneRuntimeArray', 'AdapterManager', 'ModelSnapshot', 'runtimeError',
      'parseStrictJSON',
    ]) assert.equal(module[hidden], undefined);
    assert.equal(typeof module.Trainer.create, 'function');
    assert.equal(typeof module.PTQ.create, 'function');
    assert.equal(typeof module.createPTQ, 'function');
    const runtime = await module.VolvoxAI.createRuntime();
    assert.deepEqual(runtime.listBackends(), ['wasm']);
    await runtime.close();
    assert.equal(
      fetchedUrl?.href,
      new URL('./volvoxai.full.wasm', pathToFileURL(bundlePath)).href,
    );
  } finally {
    globalThis.fetch = originalFetch;
    await rm(directory, { recursive: true, force: true });
  }
});
