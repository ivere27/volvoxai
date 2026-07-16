import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(
  await fs.readFile(path.join(repositoryRoot, 'package.json'), 'utf8'),
);
const distRoot = path.join(repositoryRoot, 'dist');
const outputDirectory = path.join(distRoot, packageJson.version);
const retainedWasm = new Set(['volvoxai.wasm', 'volvoxai.full.wasm']);

async function cleanDist() {
  await fs.mkdir(outputDirectory, { recursive: true });
  for (const entry of await fs.readdir(distRoot, { withFileTypes: true })) {
    // Keep older version directories as local release snapshots, but remove
    // obsolete flat artifacts from the pre-versioned layout.
    if (!entry.isDirectory()) {
      await fs.rm(path.join(distRoot, entry.name), { force: true });
    }
  }
  for (const entry of await fs.readdir(outputDirectory)) {
    if (!retainedWasm.has(entry)) {
      await fs.rm(path.join(outputDirectory, entry), { recursive: true, force: true });
    }
  }
}

async function bundle(entryPoint, outfile, minify, { browserOnly = false } = {}) {
  await build({
    absWorkingDir: repositoryRoot,
    entryPoints: [entryPoint],
    outfile: path.join(outputDirectory, outfile),
    bundle: true,
    format: 'esm',
    target: ['es2022'],
    loader: { '.wgsl': 'text' },
    define: browserOnly ? { __VOLVOXAI_BROWSER_ONLY__: 'true' } : {},
    minify,
    minifySyntax: browserOnly || minify,
    logLevel: 'info',
  });
}

async function buildNormal() {
  await Promise.all([
    bundle('ts/index.ts', 'volvoxai.js', false),
    bundle('ts/full.ts', 'volvoxai.full.js', false),
    bundle('ts/wasm.ts', 'volvoxai.wasm.js', false, { browserOnly: true }),
  ]);
}

async function buildMinified() {
  await Promise.all([
    bundle('ts/index.ts', 'volvoxai.min.js', true),
    bundle('ts/full.ts', 'volvoxai.full.min.js', true),
    bundle('ts/wasm.ts', 'volvoxai.wasm.min.js', true, { browserOnly: true }),
  ]);
}

const mode = process.argv[2] || 'all';
if (mode === 'clean') {
  await cleanDist();
} else if (mode === 'normal') {
  await cleanDist();
  await buildNormal();
} else if (mode === 'min') {
  await fs.mkdir(outputDirectory, { recursive: true });
  await buildMinified();
} else if (mode === 'all') {
  await cleanDist();
  await Promise.all([buildNormal(), buildMinified()]);
} else {
  throw new Error(`Unknown web build mode '${mode}'. Use clean, normal, min, or all.`);
}
