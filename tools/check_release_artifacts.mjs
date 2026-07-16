import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(await fs.readFile(
  path.join(repositoryRoot, 'package.json'),
  'utf8',
));
const releaseDirectory = path.join(repositoryRoot, 'dist', packageJson.version);
const expected = [
  'volvoxai.js',
  'volvoxai.min.js',
  'volvoxai.full.js',
  'volvoxai.full.min.js',
  'volvoxai.wasm.js',
  'volvoxai.wasm.min.js',
  'volvoxai.wasm',
  'volvoxai.full.wasm',
].sort();

let actual = [];
try {
  actual = (await fs.readdir(releaseDirectory)).sort();
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
}

const missing = expected.filter((name) => !actual.includes(name));
const unexpected = actual.filter((name) => !expected.includes(name));
if (missing.length || unexpected.length) {
  const details = [
    ...missing.map((name) => `missing: dist/${packageJson.version}/${name}`),
    ...unexpected.map((name) => `unexpected: dist/${packageJson.version}/${name}`),
  ].join('\n');
  throw new Error(
    `Release artifacts are incomplete or stale:\n${details}\n` +
    'Run make build_web before npm pack or npm publish.',
  );
}

console.log(`Verified ${expected.length} release artifacts in dist/${packageJson.version}.`);
