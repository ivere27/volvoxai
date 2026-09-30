import assert from 'node:assert/strict';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { readPackageVersion, validatePackageVersion } from '../tools/release_version.mjs';
import { syncReleaseVersion } from '../tools/set_version.mjs';

function fixture(t) {
  const root = mkdtempSync(path.join(tmpdir(), 'volvoxai-version-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  const lock = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
  for (const [name, value] of [['package.json', pkg], ['package-lock.json', lock]]) {
    writeFileSync(path.join(root, name), `${JSON.stringify(value, null, 2)}\n`);
  }
  return { root, pkg, lock };
}

test('a version update changes package paths and root lock versions together', t => {
  const { root, pkg, lock } = fixture(t);
  syncReleaseVersion({ repositoryRoot: root, version: '0.8.1' });
  const updated = JSON.parse(readFileSync(path.join(root, 'package.json')));
  const updatedLock = JSON.parse(readFileSync(path.join(root, 'package-lock.json')));
  assert.equal(readPackageVersion(root), '0.8.1');
  assert.equal(updated.main, './dist/0.8.1/volvoxai.js');
  assert.deepEqual(updated.files, ['dist/0.8.1/']);
  assert.equal(Object.keys(updated.exports).length, Object.keys(pkg.exports).length);
  assert.ok(Object.values(updated.exports).every(value => value.startsWith('./dist/0.8.1/')));
  assert.deepEqual(updated.devDependencies, pkg.devDependencies);
  assert.equal(updatedLock.version, '0.8.1');
  assert.equal(updatedLock.packages[''].version, '0.8.1');
  delete updatedLock.packages[''];
  delete lock.packages[''];
  assert.deepEqual(updatedLock.packages, lock.packages);
  assert.deepEqual(syncReleaseVersion({ repositoryRoot: root, check: true }).changedFiles, []);
});

test('metadata synchronization follows a manually changed package version', t => {
  const { root, pkg } = fixture(t);
  pkg.version = '0.8.2';
  writeFileSync(path.join(root, 'package.json'), JSON.stringify(pkg));
  const before = readFileSync(path.join(root, 'package.json'), 'utf8');
  assert.throws(() => syncReleaseVersion({ repositoryRoot: root, check: true }), /metadata is stale/);
  assert.equal(readFileSync(path.join(root, 'package.json'), 'utf8'), before);
  syncReleaseVersion({ repositoryRoot: root });
  assert.equal(readPackageVersion(root), '0.8.2');
  assert.deepEqual(syncReleaseVersion({ repositoryRoot: root, check: true }).changedFiles, []);
});

test('invalid versions fail before changing either manifest', t => {
  const { root } = fixture(t);
  const before = ['package.json', 'package-lock.json'].map(name => readFileSync(path.join(root, name), 'utf8'));
  for (const version of ['../0.8.0', '0.8', '0.8.0"', '0.8.0\n',
    '1.2.3-..', '1.2.3-a..b', '1.2.3-01', '1.2.3+..']) {
    assert.throws(() => syncReleaseVersion({ repositoryRoot: root, version }), /Invalid package release version/);
  }
  assert.deepEqual(['package.json', 'package-lock.json'].map(name => readFileSync(path.join(root, name), 'utf8')), before);
});

test('valid prerelease and build identifiers retain their exact version', () => {
  for (const version of ['1.2.3-0', '1.2.3-rc.2', '1.2.3-00alpha', '1.2.3+build.01',
    '1.2.3-rc.2+build.01']) {
    assert.equal(validatePackageVersion(version), version);
  }
});

test('the npm version hook synchronizes paths without creating a Git tag', t => {
  const { root } = fixture(t);
  syncReleaseVersion({ repositoryRoot: root, version: '1.2.3' });
  for (const file of ['set_version.mjs', 'release_version.mjs', 'release_profiles.mjs',
    'wasm_host_imports.mjs', 'generated/wasmInternalAbi.mjs']) {
    const destination = path.join(root, 'tools', file);
    mkdirSync(path.dirname(destination), { recursive: true });
    copyFileSync(new URL(`../tools/${file}`, import.meta.url), destination);
  }
  const result = spawnSync('npm', ['version', '1.2.4', '--no-git-tag-version'], {
    cwd: root, encoding: 'utf8',
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  const updated = JSON.parse(readFileSync(path.join(root, 'package.json')));
  assert.equal(updated.version, '1.2.4');
  assert.equal(updated.main, './dist/1.2.4/volvoxai.js');
  assert.deepEqual(syncReleaseVersion({ repositoryRoot: root, check: true }).changedFiles, []);
});
