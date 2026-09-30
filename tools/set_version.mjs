#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RELEASE_PROFILES } from './release_profiles.mjs';
import { readPackageVersion, validatePackageVersion } from './release_version.mjs';

const defaultRepositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/** Synchronize npm's required metadata from the one package version. */
export function syncReleaseVersion({ repositoryRoot = defaultRepositoryRoot, version, check = false } = {}) {
  const selected = validatePackageVersion(version ?? readPackageVersion(repositoryRoot));
  const packagePath = path.join(repositoryRoot, 'package.json');
  const lockPath = path.join(repositoryRoot, 'package-lock.json');
  const pkg = JSON.parse(readFileSync(packagePath, 'utf8'));
  const lock = JSON.parse(readFileSync(lockPath, 'utf8'));
  if (!lock.packages?.['']) throw new Error('package-lock.json must contain the root package metadata.');

  const releaseRoot = `./dist/${selected}`;
  pkg.version = selected;
  pkg.exports = {};
  for (const profile of RELEASE_PROFILES.browser) {
    pkg.exports[profile.packageExports.readable] = `${releaseRoot}/${profile.readable}`;
    pkg.exports[profile.packageExports.minified] = `${releaseRoot}/${profile.minified}`;
  }
  for (const profile of RELEASE_PROFILES.wasm) {
    pkg.exports[`./${profile.filename}`] = `${releaseRoot}/${profile.filename}`;
  }
  pkg.main = pkg.exports['.'];
  pkg.files = [`dist/${selected}/`];
  lock.version = selected;
  lock.packages[''].version = selected;

  const changes = [[packagePath, pkg], [lockPath, lock]].filter(([file, expected]) =>
    JSON.stringify(JSON.parse(readFileSync(file, 'utf8'))) !== JSON.stringify(expected));
  if (check && changes.length > 0) {
    throw new Error('Release metadata is stale. Run npm run version:set to synchronize package paths and lockfile versions.');
  }
  if (!check) for (const [file, contents] of changes) {
    writeFileSync(file, `${JSON.stringify(contents, null, 2)}\n`);
  }
  return { version: selected, changedFiles: changes.map(([file]) => path.relative(repositoryRoot, file)) };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = process.argv.slice(2);
    if (args.length > 1 || (args[0]?.startsWith('--') && args[0] !== '--check')) {
      throw new Error('Usage: node tools/set_version.mjs [version | --check]');
    }
    const check = args[0] === '--check';
    const result = syncReleaseVersion({ version: check ? undefined : args[0], check });
    console.log(`${check ? 'Verified' : 'Synchronized'} release version ${result.version}.`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
