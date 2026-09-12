import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import {
  validateWasmReleaseBuildSnapshot,
  wasmReleaseEvidencePath} from './build_wasm_release.mjs';
import {
  DIST_RELEASE_FILENAMES,
  NATIVE_RELEASE_FILENAMES,
  RELEASE_PROFILES,
  browserProvenance,
  parseBrowserProvenance,
  stableJSON,
  validateBrowserReleasePayload,
  validateReleaseDeclarations} from './release_profiles.mjs';

const executeFile = promisify(execFile);
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const options = new Set(process.argv.slice(2));
for (const option of options) {
  if (option !== '--web-only') throw new Error(`Unknown release-check option '${option}'.`);
}
const webOnly = options.has('--web-only');

const declarations = await validateReleaseDeclarations(repositoryRoot);
if (declarations.errors.length > 0) {
  throw new Error(
    `Release declarations are inconsistent:\n${declarations.errors.join('\n')}`,
  );
}
const { packageJson } = declarations;
const releaseDirectory = path.join(repositoryRoot, 'dist', packageJson.version);

let actual = [];
try {
  actual = (await fs.readdir(releaseDirectory)).sort();
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
}

const expectedDist = [...DIST_RELEASE_FILENAMES].sort();
const errors = [
  ...expectedDist.filter((name) => !actual.includes(name))
    .map((name) => `missing: dist/${packageJson.version}/${name}`),
  ...actual.filter((name) => !expectedDist.includes(name))
    .map((name) => `unexpected: dist/${packageJson.version}/${name}`),
];

const wasmResults = new Map();
for (const profile of RELEASE_PROFILES.wasm) {
  const artifactPath = path.join(releaseDirectory, profile.filename);
  if (!actual.includes(profile.filename)) continue;
  try {
    const info = await fs.stat(artifactPath);
    if (!info.isFile() || info.size === 0) {
      errors.push(`invalid: dist/${packageJson.version}/${profile.filename}`);
      continue;
    }
    const evidencePath = wasmReleaseEvidencePath(repositoryRoot, profile);
    const snapshot = await validateWasmReleaseBuildSnapshot({
      repositoryRoot,
      artifact: artifactPath,
      evidence: evidencePath,
      profile,
      packageVersion: packageJson.version,
      // This read-only package gate may run on a publishing host without the
      // build image. Fresh provenance replay performs the live executable and
      // Clang resource-header rehash; the exact recorded identities are still
      // schema-checked against the central recipe here.
      verifyToolchainInputs: false,
    });
    const { recorded, objectGraph, componentBinding } = snapshot;
    const validated = snapshot.validatedArtifact;
    wasmResults.set(profile.filename, Object.freeze({
      ...validated,
      buildEvidence: Object.freeze({
        path: path.relative(repositoryRoot, evidencePath).replaceAll('\\', '/'),
        rawBytes: recorded.rawBytes,
        sha256: recorded.sha256,
        objectGraph,
        componentBinding,
      }),
    }));
  } catch (error) {
    errors.push(`invalid: dist/${packageJson.version}/${profile.filename}: ${error.message}`);
  }
}
if (wasmResults.size === RELEASE_PROFILES.wasm.length) {
  const relaxedChildren = new Set([...wasmResults.values()]
    .map((result) => result.boundary.relaxedSimdSha256).filter(Boolean));
  if (relaxedChildren.size > 1) {
    errors.push('inference/full WASM relaxed-SIMD accelerator payloads differ');
  }
}

for (const profile of RELEASE_PROFILES.browser) {
  const expected = await browserProvenance(repositoryRoot, profile.id, packageJson.version);
  for (const filename of [profile.readable, profile.minified]) {
    if (!actual.includes(filename)) continue;
    try {
      const artifactPath = path.join(releaseDirectory, filename);
      const info = await fs.stat(artifactPath);
      if (!info.isFile() || info.size === 0) {
        errors.push(`invalid: dist/${packageJson.version}/${filename}`);
        continue;
      }
      const artifactBytes = await fs.readFile(artifactPath);
      const provenance = parseBrowserProvenance(artifactBytes.toString('utf8'), filename);
      if (stableJSON(provenance) !== stableJSON(expected)) {
        errors.push(
          `invalid: dist/${packageJson.version}/${filename}: ` +
          'browser provenance differs from its central profile',
        );
      }
      const sidecar = wasmResults.get(profile.sidecar)?.provenance;
      if (sidecar && (
        provenance.wasmSourceSha256 !== sidecar.sourceSha256 ||
        provenance.wasmRecipeSha256 !== sidecar.recipeSha256 ||
        provenance.wasmContractSha256 !== sidecar.contractSha256
      )) {
        errors.push(
          `invalid: dist/${packageJson.version}/${filename}: ` +
          `provenance does not match ${profile.sidecar}`,
        );
      }
      await validateBrowserReleasePayload(
        repositoryRoot,
        artifactPath,
        profile.id,
        packageJson.version,
        { minify: filename === profile.minified },
      );
    } catch (error) {
      errors.push(`invalid: dist/${packageJson.version}/${filename}: ${error.message}`);
    }
  }
}

if (!webOnly) {
  const validNative = [];
  const nativeVersions = new Map();
  for (const profile of RELEASE_PROFILES.native) {
    const filename = profile.filename;
    try {
      const artifactPath = path.join(repositoryRoot, filename);
      const info = await fs.stat(artifactPath);
      if (!info.isFile() || info.size === 0) {
        errors.push(`invalid: ${filename}`);
      } else {
        validNative.push(filename);
        try {
          const { stdout: versionOutput } = await executeFile(
            artifactPath,
            ['--version'],
            { cwd: repositoryRoot, maxBuffer: 1024 * 1024 },
          );
          const version = versionOutput.trim();
          if (!version.startsWith(`VolvoxAI Native Engine ${packageJson.version} (`)) {
            errors.push(
              `invalid: ${filename}: native version does not match package ` +
              `${packageJson.version}`,
            );
          }
          nativeVersions.set(filename, version);
        } catch (error) {
          errors.push(`invalid: ${filename}: native ABI/version query failed: ${error.message}`);
        }
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      errors.push(`missing: ${filename}`);
    }
  }
  if (nativeVersions.size === RELEASE_PROFILES.native.length &&
      new Set(nativeVersions.values()).size !== 1) {
    errors.push('native inference/full release metadata comes from different builds');
  }
  if (validNative.length === NATIVE_RELEASE_FILENAMES.length) {
    const boundaryCheck = path.join(repositoryRoot, RELEASE_PROFILES.nativeBoundaryCheck);
    try {
      await executeFile(boundaryCheck, NATIVE_RELEASE_FILENAMES.map((filename) =>
        path.join(repositoryRoot, filename)), {
        cwd: repositoryRoot,
        maxBuffer: 16 * 1024 * 1024,
      });
    } catch (error) {
      const details = [error.stdout, error.stderr].filter(Boolean).join('\n').trim();
      errors.push(
        `native inference/full boundary check failed${details ? `:\n${details}` : ''}`,
      );
    }
  }
}

if (errors.length > 0) {
  throw new Error(
    `Release artifacts are incomplete, stale, or profile-incompatible:\n${errors.join('\n')}\n` +
    'Run make build_web and make build_native before npm pack or publishing a release.',
  );
}

const total = DIST_RELEASE_FILENAMES.length + (webOnly ? 0 : NATIVE_RELEASE_FILENAMES.length);
console.log(
  `Verified all ${total} fixed release artifacts, WASM provenance/ABI boundaries` +
  `${webOnly ? '.' : ', and native inference/full symbol boundaries.'}`,
);
