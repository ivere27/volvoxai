import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';
import { readPackageVersion } from '../tools/release_version.mjs';

const repositoryRoot = fileURLToPath(new URL('..', import.meta.url));
const tools = [
  { file: 'tools/llm_baseline/verify_webgpu_generation_lifecycle.mjs',
    start: 'const version =', end: 'const selected =', minified: false, requiresOut: true },
  { file: 'tools/llm_baseline/verify_modern_webgpu.mjs',
    start: 'const root = process.cwd();', end: 'const sha256 =', minified: false, requiresOut: true },
  { file: 'tests/parity/external/webgpu_composed_runtime.mjs',
    start: 'function options(argv)', end: '/* One graph,', minified: true, requiresOut: false },
];

function fixture(t) {
  const root = mkdtempSync(path.join(tmpdir(), 'volvoxai-webgpu-paths-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const cwd = path.join(root, 'caller');
  mkdirSync(cwd);
  const setVersion = version => writeFileSync(path.join(root, 'package.json'), JSON.stringify({ version }));
  setVersion('9.7.1');
  return { root, cwd, setVersion };
}

function parse(tool, fixture, argv, deno = true) {
  const url = new URL(`../${tool.file}`, import.meta.url);
  const source = readFileSync(url, 'utf8');
  const start = source.indexOf(tool.start), end = source.indexOf(tool.end, start);
  assert.ok(start >= 0 && end > start, `${tool.file}: argument setup boundaries`);
  // Run the production argument setup before any browser/GPU side effects.
  // Only package metadata and the caller's CWD are supplied by the fixture.
  const setup = source.slice(start, end).replaceAll('import.meta.url', 'scriptUrl');
  const value = tool.requiresOut ? 'options' : 'options(argv)';
  return JSON.parse(runInNewContext(`${setup}\nJSON.stringify(${value})`, {
    assert, URL, fileURLToPath, scriptUrl: url.href,
    readPackageVersion: () => readPackageVersion(fixture.root),
    path: { ...path, resolve: (...parts) => path.resolve(fixture.cwd, ...parts) },
    process: { cwd: () => fixture.cwd, argv: ['node', fileURLToPath(url), ...argv] },
    ...(deno ? { Deno: { args: argv } } : {}),
    argv,
    fail: message => { throw new Error(message); },
  }, { filename: fileURLToPath(url) }));
}

for (const tool of tools) {
  test(`${tool.file} follows package metadata independently of the caller's CWD`, t => {
    const data = fixture(t);
    const argv = tool.requiresOut ? ['--out', 'receipt.json'] : [];
    for (const version of ['9.7.1', '10.2.3-rc.1']) {
      data.setVersion(version);
      const actual = parse(tool, data, argv);
      assert.equal(actual.bundle, path.join(repositoryRoot, 'dist', version,
        tool.minified ? 'volvoxai.min.js' : 'volvoxai.js'));
      assert.equal(actual.wasm, path.join(repositoryRoot, 'dist', version, 'volvoxai.wasm'));
      if (tool.requiresOut) assert.equal(actual.out, path.join(data.cwd, 'receipt.json'));
    }
  });

  test(`${tool.file} preserves explicit paths relative to the caller's CWD`, t => {
    const data = fixture(t);
    const argv = ['--bundle', 'custom/runtime.js', '--wasm', 'custom/runtime.wasm',
      ...(tool.requiresOut ? ['--out', 'receipt.json'] : [])];
    for (const deno of [true, false]) {
      const actual = parse(tool, data, argv, deno);
      assert.equal(actual.bundle, path.join(data.cwd, 'custom/runtime.js'));
      assert.equal(actual.wasm, path.join(data.cwd, 'custom/runtime.wasm'));
    }
    const mixed = parse(tool, data, ['--bundle', '/opt/volvoxai/runtime.js',
      ...(tool.requiresOut ? ['--out', 'receipt.json'] : [])]);
    assert.equal(mixed.bundle, '/opt/volvoxai/runtime.js');
    assert.equal(mixed.wasm, path.join(repositoryRoot, 'dist', '9.7.1', 'volvoxai.wasm'));
  });

  test(`${tool.file} continues to reject invalid arguments`, t => {
    const data = fixture(t);
    const required = tool.requiresOut ? ['--out', 'receipt.json'] : [];
    assert.throws(() => parse(tool, data, [...required, '--unknown']), /unknown argument/);
    for (const flag of ['--bundle', '--wasm']) {
      assert.throws(() => parse(tool, data, [...required, flag]), /requires a value/);
      assert.throws(() => parse(tool, data, [...required, flag, '--bundle']), /requires a value/);
    }
    if (tool.requiresOut) assert.throws(() => parse(tool, data, []), /--out is required/);
  });
}

test('default release paths preserve browser verification options', t => {
  const data = fixture(t);
  const lifecycle = parse(tools[0], data, ['--out', 'receipt.json', '--adapter', 'software-unit', '--only', 'cancel-pending']);
  assert.equal(lifecycle.adapter, 'software-unit');
  assert.equal(lifecycle.only, 'cancel-pending');
  const modern = parse(tools[1], data, ['--out', 'receipt.json', '--suite', 'generate', '--memory', '--only', 'llama']);
  assert.equal(modern.suite, 'generate');
  assert.equal(modern.memory, true);
  assert.equal(modern.only, 'llama');
  const composed = parse(tools[2], data, ['--recreate-hosts', '--recreate-gpu-hosts']);
  assert.equal(composed.recreateHosts, true);
  assert.equal(composed.recreateGpuHosts, true);
});
