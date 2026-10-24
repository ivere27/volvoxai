import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { prepareNpmPackages, publishNpmPackages } from '../tools/publish_npm.mjs';
import { DIST_RELEASE_FILENAMES } from '../tools/release_profiles.mjs';

const digest = bytes => `sha512-${createHash('sha512').update(bytes).digest('base64')}`;
const realNpm = spawnSync('which', ['npm'], { encoding: 'utf8' }).stdout.trim();
const stub = `#!${process.execPath}
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const args = process.argv.slice(2);
const config = JSON.parse(fs.readFileSync(process.env.PUBLISH_TEST_CONFIG));
fs.appendFileSync(process.env.PUBLISH_TEST_LOG, JSON.stringify(args) + '\\n');
if (args[0] === 'pack') {
  const name = JSON.parse(fs.readFileSync('package.json')).name;
  if (config.failPack === name) process.exit(7);
  const result = spawnSync(process.env.PUBLISH_TEST_REAL_NPM, args, {
    stdio: 'inherit', env: { ...process.env, npm_config_offline: 'true',
      npm_config_cache: process.env.PUBLISH_TEST_CACHE,
      npm_config_update_notifier: 'false' },
  });
  process.exit(result.status ?? 1);
} else if (args[0] === 'view') {
  const value = config.registry?.[args[1]];
  if (value === undefined) {
    console.log(JSON.stringify({ error: { code: 'E404' } }));
    process.exit(1);
  }
  console.log(JSON.stringify(value));
} else if (args[0] === 'whoami') {
  if (config.authFailure) process.exit(1);
  console.log('fixture-user');
} else if (args[0] === 'publish') {
  if (path.basename(args[1]) === config.failPublish) process.exit(7);
} else {
  throw new Error('Unexpected npm command: ' + args.join(' '));
}
`;

async function fixture(t) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'volvoxai-npm-publish-'));
  const bin = path.join(directory, 'bin');
  await fs.mkdir(bin);
  await fs.writeFile(path.join(bin, 'package.json'), '{"type":"commonjs"}');
  await fs.writeFile(path.join(bin, 'npm'), stub, { mode: 0o755 });
  const configPath = path.join(directory, 'config.json');
  const log = path.join(directory, 'calls.jsonl');
  await fs.writeFile(configPath, '{}');
  const values = {
    PATH: `${bin}${path.delimiter}${process.env.PATH}`,
    PUBLISH_TEST_CONFIG: configPath, PUBLISH_TEST_LOG: log,
    PUBLISH_TEST_REAL_NPM: realNpm, PUBLISH_TEST_CACHE: path.join(directory, 'cache'),
  };
  const previous = Object.fromEntries(Object.keys(values).map(key => [key, process.env[key]]));
  Object.assign(process.env, values);
  t.after(async () => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    await fs.rm(directory, { recursive: true, force: true });
  });
  const packages = [];
  for (const name of ['volvoxai', 'volvoxai-lite']) {
    const bytes = Buffer.from(`${name} fixture`);
    const filename = `${name}-1.2.3.tgz`;
    const tarballPath = path.join(directory, filename);
    await fs.writeFile(tarballPath, bytes);
    packages.push({ name, version: '1.2.3', filename, tarballPath, integrity: digest(bytes) });
  }
  return {
    directory, packages,
    config: value => fs.writeFile(configPath, JSON.stringify(value)),
    calls: async () => {
      const contents = await fs.readFile(log, 'utf8').catch(error => {
        if (error.code === 'ENOENT') return ''; throw error;
      });
      return contents.trim() ? contents.trim().split('\n').map(JSON.parse) : [];
    },
  };
}

async function writePackageFixture(directory) {
  const version = '1.2.3';
  const files = {
    'package.json': JSON.stringify({ name: 'volvoxai', version, type: 'module',
      files: [`dist/${version}/`],
      scripts: { prepack: 'node tools/check_release_artifacts.mjs --web-only' } }),
    'README.md': '# Full fixture\n', LICENSE: 'Fixture license\n',
    'packages/lite/README.md': '# Lite fixture\n',
    'tools/check_release_artifacts.mjs': 'console.log("fixture release check passed");\n',
    ...Object.fromEntries(DIST_RELEASE_FILENAMES.map(name => [`dist/${version}/${name}`, name])),
  };
  for (const [name, contents] of Object.entries(files)) {
    const destination = path.join(directory, name);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.writeFile(destination, contents);
  }
}

test('dry preparation validates both actual tarballs without registry, authentication or uploads', async t => {
  const f = await fixture(t);
  await writePackageFixture(f.directory);
  const packages = await prepareNpmPackages({ repositoryRoot: f.directory });
  assert.deepEqual(packages.map(item => item.name), ['volvoxai', 'volvoxai-lite']);
  assert.deepEqual(packages.map(item => item.files.length), [9, 6]);
  assert.deepEqual((await f.calls()).map(args => args[0]), ['pack', 'pack']);
  for (const item of packages) assert.equal(digest(await fs.readFile(item.tarballPath)), item.integrity);
  const lite = packages[1];
  const manifest = JSON.parse(spawnSync('tar', ['-xOf', lite.tarballPath, 'package/package.json'],
    { encoding: 'utf8' }).stdout);
  assert.equal(manifest.exports['.'], `./dist/${lite.version}/volvoxai.lite.js`);
  assert.equal(manifest.dependencies, undefined);
  assert.equal(manifest.scripts, undefined);
});

test('failure preparing the second package leaves no completed release manifest', async t => {
  const f = await fixture(t);
  await writePackageFixture(f.directory);
  await f.config({ failPack: 'volvoxai-lite' });
  await assert.rejects(prepareNpmPackages({ repositoryRoot: f.directory }), /failed/);
  assert.deepEqual(await fs.readdir(path.join(f.directory, 'build/npm/1.2.3')), []);
  assert.deepEqual((await f.calls()).map(args => args[0]), ['pack', 'pack']);
});

test('both registry versions are checked before authentication and uploads', async t => {
  const f = await fixture(t);
  await publishNpmPackages(f.packages, { repositoryRoot: f.directory });
  const calls = await f.calls();
  assert.deepEqual(calls.map(args => args[0]), ['view', 'view', 'whoami', 'publish', 'publish']);
  assert.deepEqual(calls.filter(args => args[0] === 'publish').map(args => args[1]),
    f.packages.map(item => item.tarballPath));
});

test('an identical existing full package is skipped while lite is published', async t => {
  const f = await fixture(t);
  await f.config({ registry: { 'volvoxai@1.2.3': f.packages[0].integrity } });
  await publishNpmPackages(f.packages, { repositoryRoot: f.directory });
  assert.deepEqual((await f.calls()).filter(args => args[0] === 'publish').map(args => args[1]),
    [f.packages[1].tarballPath]);
});

test('two identical existing packages need no authentication or upload', async t => {
  const f = await fixture(t);
  await f.config({ registry: Object.fromEntries(f.packages.map(item =>
    [`${item.name}@${item.version}`, item.integrity])) });
  await publishNpmPackages(f.packages, { repositoryRoot: f.directory });
  assert.deepEqual((await f.calls()).map(args => args[0]), ['view', 'view']);
});

test('a conflicting second version prevents every upload', async t => {
  const f = await fixture(t);
  await f.config({ registry: { 'volvoxai-lite@1.2.3': 'sha512-conflict' } });
  await assert.rejects(publishNpmPackages(f.packages, { repositoryRoot: f.directory }), /different bytes/);
  assert.deepEqual((await f.calls()).map(args => args[0]), ['view', 'view']);
});

test('a changed second tarball is rejected before any registry request or upload', async t => {
  const f = await fixture(t);
  await fs.writeFile(f.packages[1].tarballPath, 'changed');
  await assert.rejects(publishNpmPackages(f.packages, { repositoryRoot: f.directory }), /changed/);
  assert.deepEqual(await f.calls(), []);
});

test('an upload failure stops and identifies the already published package', async t => {
  const f = await fixture(t);
  await f.config({ failPublish: f.packages[1].filename });
  await assert.rejects(publishNpmPackages(f.packages, { repositoryRoot: f.directory }),
    /Publishing volvoxai-lite@1\.2\.3 failed\. Published in this run: volvoxai@1\.2\.3/);
  assert.equal((await f.calls()).filter(args => args[0] === 'publish').length, 2);
});
