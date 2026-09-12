import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  RELEASE_PROFILES,
  WASM_RELEASE_FILENAMES,
  buildBrowserReleaseBundle,
  removeInvalidWasmSidecars,
  validateReleaseDeclarations} from './release_profiles.mjs';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(
  await fs.readFile(path.join(repositoryRoot, 'package.json'), 'utf8'),
);
const distRoot = path.join(repositoryRoot, 'dist');
const outputDirectory = path.join(distRoot, packageJson.version);
const retainedWasm = new Set(WASM_RELEASE_FILENAMES);

async function removeStaleWasmSidecars() {
  const removed = await removeInvalidWasmSidecars(
    repositoryRoot,
    outputDirectory,
    packageJson.version,
  );
  for (const { filename, reason } of removed) {
    console.warn(
      `[VolvoxAI] Removed stale WASM sidecar '${filename}': ${reason}`,
    );
  }
}

async function cleanDist() {
  await fs.mkdir(outputDirectory, { recursive: true });
  for (const entry of await fs.readdir(distRoot, { withFileTypes: true })) {
    // Version directories are independent release snapshots; only files at
    // the dist root are outside the fixed release layout.
    if (!entry.isDirectory()) {
      await fs.rm(path.join(distRoot, entry.name), { force: true });
    }
  }
  for (const entry of await fs.readdir(outputDirectory)) {
    if (!retainedWasm.has(entry)) {
      await fs.rm(path.join(outputDirectory, entry), { recursive: true, force: true });
    }
  }
  // A JS-only rebuild may preserve an expensive sidecar only when its embedded
  // source, recipe, ABI-contract, and payload hashes are current. Old sidecars
  // are removed explicitly instead of being silently paired with new JS.
  await removeStaleWasmSidecars();
}

async function bundle(profile, outfile, minify) {
  await buildBrowserReleaseBundle(
    repositoryRoot,
    profile.id,
    packageJson.version,
    {
      outfile: path.join(outputDirectory, outfile),
      minify,
      logLevel: 'info',
    },
  );
}

async function buildNormal() {
  await Promise.all(RELEASE_PROFILES.browser.map((profile) =>
    bundle(profile, profile.readable, false)));
}

async function buildMinified() {
  await Promise.all(RELEASE_PROFILES.browser.map((profile) =>
    bundle(profile, profile.minified, true)));
}

const declarations = await validateReleaseDeclarations(repositoryRoot);
if (declarations.errors.length > 0) {
  throw new Error(
    `Release declarations are inconsistent:\n${declarations.errors.join('\n')}`,
  );
}

const mode = process.argv[2] || 'all';
if (mode === 'clean') {
  await cleanDist();
} else if (mode === 'normal') {
  await cleanDist();
  await buildNormal();
} else if (mode === 'min') {
  await fs.mkdir(outputDirectory, { recursive: true });
  await removeStaleWasmSidecars();
  await buildMinified();
} else if (mode === 'all') {
  await cleanDist();
  await Promise.all([buildNormal(), buildMinified()]);
} else {
  throw new Error(`Unknown web build mode '${mode}'. Use clean, normal, min, or all.`);
}
