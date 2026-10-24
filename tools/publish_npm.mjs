#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RELEASE_PROFILES, DIST_RELEASE_FILENAMES } from './release_profiles.mjs';
import { readPackageVersion } from './release_version.mjs';

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const integrity = bytes => `sha512-${createHash('sha512').update(bytes).digest('base64')}`;

function command(program, args, cwd, inherit = false) {
  const result = spawnSync(program, args, {
    cwd, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
    stdio: inherit ? 'inherit' : ['inherit', 'pipe', 'pipe'],
  });
  if (result.error) throw result.error;
  return result;
}

function checkedCommand(program, args, cwd, inherit = false) {
  const result = command(program, args, cwd, inherit);
  if (result.status !== 0) {
    throw new Error(`${program} ${args.join(' ')} failed (${result.status ?? result.signal}).` +
      (inherit ? '' : `\n${result.stderr ?? ''}${result.stdout ?? ''}`));
  }
  return result;
}

function npmJSON(output) {
  // npm forwards prepack stdout before its JSON result, even with --json.
  const text = output.trim();
  for (const start of [0, ...[...text.matchAll(/\n(?=[\[{])/g)].map(match => match.index + 1)]) {
    try { return JSON.parse(text.slice(start)); } catch { /* Try the final document. */ }
  }
  throw new Error('npm did not return a complete JSON result.');
}

async function packPackage(directory, destination, expected, version) {
  const result = checkedCommand(npm, ['pack', '--json', '--pack-destination', destination], directory);
  const [packed, ...extra] = npmJSON(result.stdout);
  if (extra.length || packed?.name !== expected.name || packed.version !== version ||
      packed.filename !== `${expected.name}-${version}.tgz`) {
    throw new Error('npm packed an unexpected package name, version or filename.');
  }
  const actualFiles = packed.files.map(file => file.path).sort();
  const expectedFiles = [...expected.contents.keys()].sort();
  if (JSON.stringify(actualFiles) !== JSON.stringify(expectedFiles)) {
    throw new Error(`${expected.name} package inventory differs: ${actualFiles.join(', ')}`);
  }
  const tarballPath = path.join(destination, packed.filename);
  const bytes = await fs.readFile(tarballPath);
  if (packed.integrity !== integrity(bytes)) throw new Error(`${expected.name} tarball integrity differs.`);
  for (const [filename, content] of expected.contents) {
    const extracted = spawnSync('tar', ['-xOf', tarballPath, `package/${filename}`], {
      maxBuffer: 16 * 1024 * 1024,
    });
    if (extracted.error) throw extracted.error;
    if (extracted.status !== 0 || !extracted.stdout.equals(content)) {
      throw new Error(`${expected.name}: packed ${filename} differs from the validated source.`);
    }
  }
  console.log(`${packed.name}@${packed.version}: ${packed.files.length} files, ` +
    `${packed.size} bytes packed, ${packed.unpackedSize} bytes unpacked.`);
  return { name: packed.name, version, filename: packed.filename, tarballPath,
    integrity: packed.integrity, files: packed.files, size: packed.size,
    unpackedSize: packed.unpackedSize };
}

/** Package the already-validated full and inference profiles without rebuilding either. */
export async function prepareNpmPackages({
  repositoryRoot = defaultRoot, outputDirectory,
} = {}) {
  const version = readPackageVersion(repositoryRoot);
  const rootPackage = JSON.parse(await fs.readFile(path.join(repositoryRoot, 'package.json'), 'utf8'));
  if (rootPackage.name !== 'volvoxai') throw new Error('Expected the volvoxai root package.');
  checkedCommand(process.execPath, ['tools/check_release_artifacts.mjs', '--web-only'], repositoryRoot, true);
  const destination = path.resolve(outputDirectory ?? path.join(repositoryRoot, 'build', 'npm', version));
  await fs.mkdir(destination, { recursive: true });
  const temporary = await fs.mkdtemp(path.join(destination, '.prepare-'));
  try {
    const releasePath = `dist/${version}`;
    const profile = RELEASE_PROFILES.browser.find(item => item.id === 'inference');
    const liteFiles = [profile.readable, profile.minified, profile.sidecar]
      .map(filename => `${releasePath}/${filename}`);
    const fullContents = new Map(await Promise.all([
      'package.json', 'README.md', 'LICENSE',
      ...DIST_RELEASE_FILENAMES.map(filename => `${releasePath}/${filename}`),
    ].map(async filename => [filename, await fs.readFile(path.join(repositoryRoot, filename))])));
    const litePackage = {
      name: 'volvoxai-lite', version,
      description: 'VolvoxAI WASM CPU inference runtime with one generated protobuf API',
      ...Object.fromEntries(['repository', 'homepage', 'bugs', 'author', 'license']
        .filter(key => rootPackage[key] !== undefined).map(key => [key, rootPackage[key]])),
      type: 'module', main: `./${releasePath}/${profile.readable}`,
      exports: {
        '.': `./${releasePath}/${profile.readable}`,
        './min': `./${releasePath}/${profile.minified}`,
        [`./${profile.sidecar}`]: `./${releasePath}/${profile.sidecar}`,
      },
      files: liteFiles,
    };
    const liteContents = new Map([
      ['package.json', Buffer.from(`${JSON.stringify(litePackage, null, 2)}\n`)],
      ['README.md', await fs.readFile(path.join(repositoryRoot, 'packages', 'lite', 'README.md'))],
      ['LICENSE', fullContents.get('LICENSE')],
      ...liteFiles.map(filename => [filename, fullContents.get(filename)]),
    ]);
    const liteDirectory = path.join(temporary, 'lite');
    for (const [filename, content] of liteContents) {
      const target = path.join(liteDirectory, filename);
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, content);
    }
    const packages = [];
    for (const [directory, name, contents] of [
      [repositoryRoot, 'volvoxai', fullContents],
      [liteDirectory, 'volvoxai-lite', liteContents],
    ]) packages.push(await packPackage(directory, temporary, { name, contents }, version));
    // Publish only complete, checked tarballs; never re-pack a mutable directory during upload.
    for (const item of packages) {
      const finalPath = path.join(destination, item.filename);
      await fs.rename(item.tarballPath, finalPath);
      item.tarballPath = finalPath;
    }
    await fs.writeFile(path.join(destination, 'npm-packages.json'), `${JSON.stringify(packages, null, 2)}\n`);
    return packages;
  } finally {
    await fs.rm(temporary, { recursive: true, force: true });
  }
}

/** Check both registry versions before the first upload; exact published tarballs are reusable. */
export async function publishNpmPackages(packages, { repositoryRoot = defaultRoot } = {}) {
  for (const item of packages) {
    if (integrity(await fs.readFile(item.tarballPath)) !== item.integrity) {
      throw new Error(`${item.name}@${item.version}: prepared tarball changed before publication.`);
    }
  }
  const pending = [];
  for (const item of packages) {
    const spec = `${item.name}@${item.version}`;
    const result = command(npm, ['view', spec, 'dist.integrity', '--json'], repositoryRoot);
    let registry;
    try { registry = npmJSON(result.stdout); } catch (error) {
      throw new Error(`Cannot check ${spec}: ${result.stderr || error.message}`);
    }
    if (result.status !== 0) {
      if (registry?.error?.code !== 'E404') {
        throw new Error(`Cannot check ${spec}: ${result.stderr || result.stdout}`);
      }
      pending.push(item);
    } else if (registry === item.integrity) {
      console.log(`${spec} is already published with identical bytes; skipping.`);
    } else {
      throw new Error(`${spec} already exists with different bytes. Select a new release version before publishing.`);
    }
  }
  if (pending.length === 0) return;
  const authenticated = command(npm, ['whoami'], repositoryRoot);
  if (authenticated.status !== 0) throw new Error('npm is not authenticated. Run npm login before publishing.');
  const published = [];
  for (const item of pending) {
    try {
      if (integrity(await fs.readFile(item.tarballPath)) !== item.integrity) {
        throw new Error('The prepared tarball changed before upload.');
      }
      checkedCommand(npm, ['publish', item.tarballPath, '--access', 'public'], repositoryRoot, true);
      published.push(`${item.name}@${item.version}`);
    } catch (error) {
      throw new Error(`Publishing ${item.name}@${item.version} failed. ` +
        `Published in this run: ${published.join(', ') || 'none'}. ${error.message}`);
    }
  }
  console.log(`npm publish complete: ${published.join(', ')}.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = process.argv.slice(2);
    if (args.length > 1 || (args.length && args[0] !== '--dry-run')) {
      throw new Error('Usage: node tools/publish_npm.mjs [--dry-run]');
    }
    const packages = await prepareNpmPackages();
    if (args[0] === '--dry-run') {
      console.log('npm dry run passed for volvoxai and volvoxai-lite; no upload performed.');
    } else {
      await publishNpmPackages(packages);
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
