import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { prepareNpmPackages } from '../tools/publish_npm.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const run = promisify(execFile);
const MAX_OUTPUT = 16 * 1024 * 1024;

async function inspectArchive(release, expectedRuntimeFiles, directory, packageJson) {
  const expectedFiles = ['LICENSE', 'README.md', 'package.json', ...expectedRuntimeFiles].sort();
  assert.equal(release.version, packageJson.version);
  assert.equal(path.basename(release.tarballPath), release.filename);
  assert.match(release.integrity, /^sha512-/);
  assert.deepEqual(release.files.map(file => file.path).sort(), expectedFiles);

  const listing = await run('tar', ['-tzf', release.tarballPath], { maxBuffer: MAX_OUTPUT });
  assert.deepEqual(listing.stdout.trim().split('\n').sort(), expectedFiles.map(file => `package/${file}`).sort());
  await fs.mkdir(directory);
  await run('tar', ['-xzf', release.tarballPath, '-C', directory], { maxBuffer: MAX_OUTPUT });
  const packageRoot = path.join(directory, 'package');
  const manifest = JSON.parse(await fs.readFile(path.join(packageRoot, 'package.json'), 'utf8'));
  assert.equal(manifest.name, release.name);
  assert.equal(manifest.version, packageJson.version);
  assert.equal(manifest.type, 'module');
  for (const file of expectedRuntimeFiles) {
    const [packed, built] = await Promise.all([
      fs.readFile(path.join(packageRoot, file)), fs.readFile(path.join(ROOT, file)),
    ]);
    assert.deepEqual(packed, built, `${release.name}: ${file} differs from its qualified build`);
  }
  assert.deepEqual(await fs.readFile(path.join(packageRoot, 'LICENSE')), await fs.readFile(path.join(ROOT, 'LICENSE')));
  return manifest;
}

// Written into each isolated consumer so imports resolve through npm's actual
// installed manifest and the host must discover its adjacent WASM by itself.
async function smoke(specifier, lite, packageName) {
  const api = await import(specifier);
  assert.equal(typeof api.EngineHost, 'function');
  for (const name of ['FullEngineHost', 'VxTrainingServiceClient', 'VxPlanningServiceClient']) {
    assert.equal(name in api, !lite, `${specifier}: ${name}`);
  }
  const sidecar = lite ? 'volvoxai.lite.wasm' : 'volvoxai.wasm';
  assert.ok(import.meta.resolve(`${packageName}/${sidecar}`).endsWith(`/${sidecar}`));
  const { pb: p } = api;
  const host = new api.EngineHost();
  const inference = new api.VxInferenceServiceClient(host);
  try {
    const platform = await new api.VxPlatformServiceClient(host).getPlatformInfo(new p.Empty());
    assert.equal(platform.compiledBackends.includes('wasm'), true);
    assert.equal(platform.compiledBackends.includes('webgpu'), !lite);
    const runtime = await inference.createRuntime(new p.CreateRuntimeRequest());
    const graph = {
      format: 'volvox-graph/v1', dimensions: {},
      inputs: { x: { dtype: 'float32', shape: [4] } },
      nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'x' }, params: {},
        outputs: { out: { tensor: 'y', dtype: 'float32', shape: [4] } } }],
      outputs: ['y'],
      execution: { stateless: {}, entrypoints: [{ name: 'evaluate', effect: 1,
        allowed_ranges: [1], inputs: ['x'], outputs: ['y'] }] },
    };
    const model = await inference.loadModel(new p.LoadModelRequest({
      runtimeId: runtime.runtimeId,
      package: new p.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(graph)) }),
    }));
    const compiled = await inference.compileModel(new p.CompileModelRequest({ modelId: model.modelId }));
    const context = await inference.createExecutionContext(new p.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution(),
    }));
    const result = await inference.execute(new p.ExecuteRequest({
      contextId: context.contextId,
      inputs: new p.InputBindings({ values: [new p.Tensor({
        name: 'x', shape: [4n], dtype: p.DataType.DATA_TYPE_F32,
        inline: new Uint8Array(Float32Array.of(-2, 0, 3, 7).buffer),
      })] }),
      scope: new p.ExecutionScope({ entrypoint: 'evaluate', full: new p.Empty() }),
    }));
    const output = result.outputs.find(tensor => tensor.name === 'y');
    assert.ok(output?.inline, `${specifier}: missing inference output`);
    assert.deepEqual(Array.from(new Float32Array(output.inline.slice().buffer)), [0, 0, 3, 7]);
    await inference.releaseExecutionContext(new p.ExecutionContextRef({ contextId: context.contextId }));
    await inference.releaseCompiledModel(new p.CompiledModelRef({ compiledModelId: compiled.compiledModelId }));
    await inference.releaseModel(new p.ModelRef({ modelId: model.modelId }));
    await inference.releaseRuntime(new p.RuntimeRef({ runtimeId: runtime.runtimeId }));
  } finally {
    await host.close();
  }
}

async function installAndSmoke(release, directory, cases) {
  await fs.mkdir(directory);
  await fs.writeFile(path.join(directory, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  await run('npm', ['install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund',
    '--package-lock=false', '--cache', path.join(directory, 'cache'), release.tarballPath], {
    cwd: directory, maxBuffer: MAX_OUTPUT,
  });
  const installed = (await fs.readdir(path.join(directory, 'node_modules'))).filter(name => !name.startsWith('.'));
  assert.deepEqual(installed, [release.name], 'the package must install without runtime dependencies');
  const script = `import assert from 'node:assert/strict';\n${smoke.toString()}\n` +
    `for (const args of ${JSON.stringify(cases)}) await smoke(...args);\n`;
  await fs.writeFile(path.join(directory, 'smoke.mjs'), script);
  await run(process.execPath, ['smoke.mjs'], { cwd: directory, maxBuffer: MAX_OUTPUT });
}

test('full and lite npm archives install independently and execute their advertised profiles',
  { timeout: 180_000 }, async t => {
    const packageJson = JSON.parse(await fs.readFile(path.join(ROOT, 'package.json'), 'utf8'));
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'volvoxai-npm-packages-'));
    t.after(() => fs.rm(directory, { recursive: true, force: true }));
    const releases = await prepareNpmPackages({
      repositoryRoot: ROOT, outputDirectory: path.join(directory, 'packages'),
    });
    assert.deepEqual(releases.map(release => release.name).sort(), ['volvoxai', 'volvoxai-lite']);
    const full = releases.find(release => release.name === 'volvoxai');
    const lite = releases.find(release => release.name === 'volvoxai-lite');
    const base = `dist/${packageJson.version}`;
    const liteFiles = ['volvoxai.lite.js', 'volvoxai.lite.min.js', 'volvoxai.lite.wasm'].map(name => `${base}/${name}`);
    const fullFiles = [...liteFiles, ...['volvoxai.js', 'volvoxai.min.js', 'volvoxai.wasm'].map(name => `${base}/${name}`)];

    await t.test('lite archive contains exactly its three built runtime files and package metadata', async () => {
      const manifest = await inspectArchive(lite, liteFiles, path.join(directory, 'lite-archive'), packageJson);
      assert.equal(manifest.main, `./${base}/volvoxai.lite.js`);
      assert.deepEqual(manifest.exports, {
        '.': `./${base}/volvoxai.lite.js`,
        './min': `./${base}/volvoxai.lite.min.js`,
        './volvoxai.lite.wasm': `./${base}/volvoxai.lite.wasm`,
      });
      for (const field of ['dependencies', 'optionalDependencies', 'peerDependencies', 'devDependencies']) {
        assert.equal(Object.keys(manifest[field] ?? {}).length, 0, `unexpected lite ${field}`);
      }
      assert.equal(Object.keys(manifest.scripts ?? {}).length, 0, 'staged lite must not inherit repository lifecycle scripts');
    });

    await t.test('full archive preserves all six built runtime files and existing exports', async () => {
      const manifest = await inspectArchive(full, fullFiles, path.join(directory, 'full-archive'), packageJson);
      assert.deepEqual(manifest.exports, packageJson.exports);
      assert.equal(manifest.main, packageJson.main);
    });

    await t.test('lite alone installs offline and both entries find their default WASM and infer', async () => {
      await installAndSmoke(lite, path.join(directory, 'lite-consumer'), [
        ['volvoxai-lite', true, 'volvoxai-lite'], ['volvoxai-lite/min', true, 'volvoxai-lite'],
      ]);
    });

    await t.test('full package retains normal, minified and legacy lite entry behavior', async () => {
      await installAndSmoke(full, path.join(directory, 'full-consumer'), [
        ['volvoxai', false, 'volvoxai'], ['volvoxai/min', false, 'volvoxai'],
        ['volvoxai/lite', true, 'volvoxai'], ['volvoxai/lite/min', true, 'volvoxai'],
      ]);
    });
  });
