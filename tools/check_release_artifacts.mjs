import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(await fs.readFile(
  path.join(repositoryRoot, 'package.json'),
  'utf8',
));
const releaseDirectory = path.join(repositoryRoot, 'dist', packageJson.version);
const expectedDist = [
  'volvoxai.js',
  'volvoxai.min.js',
  'volvoxai.full.js',
  'volvoxai.full.min.js',
  'volvoxai.wasm.js',
  'volvoxai.wasm.min.js',
  'volvoxai.wasm',
  'volvoxai.full.wasm',
].sort();
const expectedNative = [
  'native/volvoxai',
  'native/volvoxai-full',
];

let actual = [];
try {
  actual = (await fs.readdir(releaseDirectory)).sort();
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
}

const missing = expectedDist.filter((name) => !actual.includes(name));
const unexpected = actual.filter((name) => !expectedDist.includes(name));
const invalidNative = [];
for (const filename of expectedNative) {
  try {
    const info = await fs.stat(path.join(repositoryRoot, filename));
    if (!info.isFile() || info.size === 0) invalidNative.push(`invalid: ${filename}`);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    invalidNative.push(`missing: ${filename}`);
  }
}
if (missing.length || unexpected.length || invalidNative.length) {
  const details = [
    ...missing.map((name) => `missing: dist/${packageJson.version}/${name}`),
    ...unexpected.map((name) => `unexpected: dist/${packageJson.version}/${name}`),
    ...invalidNative,
  ].join('\n');
  throw new Error(
    `Release artifacts are incomplete or stale:\n${details}\n` +
    'Run make build_web and make build_native before npm pack or npm publish.',
  );
}

console.log(`Verified all ${expectedDist.length + expectedNative.length} fixed release artifacts.`);
