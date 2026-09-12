#!/usr/bin/env node
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  buildWasmProfile,
  readWasmBuildEvidenceRecord,
  validateWasmBuildEvidenceObjects,
  validateWasmReleaseBuildSnapshot,
  wasmReleaseBuildRoot,
  wasmReleaseEvidencePath} from './build_wasm_release.mjs';
import {
  WASM_PTQ_AUTHORING_SECTION,
  WASM_RELAXED_SIMD_SECTION,
  createWasmProvenance,
  stableJSON,
  wasmProfile,
  withoutWasmCustomSection} from './release_profiles.mjs';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function option(args, name) {
  const index = args.indexOf(name);
  if (index < 0 || index + 1 >= args.length) {
    throw new Error(`Missing required ${name} argument.`);
  }
  return args[index + 1];
}

function optionalOption(args, name) {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  if (index + 1 >= args.length) throw new Error(`Missing required ${name} argument.`);
  return args[index + 1];
}

async function packageVersion() {
  const packageJson = JSON.parse(await fs.readFile(
    path.join(repositoryRoot, 'package.json'),
    'utf8',
  ));
  return packageJson.version;
}

function stripReleaseSections(bytes, profile) {
  let stripped = bytes;
  for (const section of profile.recipe.customSections) {
    stripped = withoutWasmCustomSection(stripped, section);
  }
  return Buffer.from(stripped);
}

async function verifyRecipePayload(artifactPath, buildEvidencePath) {
  const profile = wasmProfile(path.basename(artifactPath));
  const directory = await fs.mkdtemp(path.join(tmpdir(), 'volvoxai-release-wasm-'));
  try {
    const recorded = await readWasmBuildEvidenceRecord(buildEvidencePath, profile);
    const recordedEvidence = recorded.document;
    await validateWasmBuildEvidenceObjects(
      wasmReleaseBuildRoot(repositoryRoot),
      recordedEvidence,
      profile,
    );
    const relaxedPath = path.join(directory, 'relaxed-simd.wasm');
    const ptqAuthoringPath = path.join(directory, 'ptq-authoring.wasm');
    const fresh = await buildWasmProfile({
      repositoryRoot,
      profile,
      outputDirectory: path.join(directory, 'dist'),
      buildRoot: path.join(directory, 'build'),
      evidenceDirectory: path.join(directory, 'evidence'),
      relaxedOutput: relaxedPath,
      ptqOutput: ptqAuthoringPath,
    });
    const baselinePath = path.join(directory, 'dist', profile.filename);
    const [artifact, baseline, freshEvidence] = await Promise.all([
      fs.readFile(artifactPath),
      fs.readFile(baselinePath),
      fs.readFile(fresh.evidencePath),
    ]);
    const actualEvidence = Buffer.from(`${stableJSON(recordedEvidence)}\n`);
    if (!stripReleaseSections(artifact, profile).equals(
      stripReleaseSections(baseline, profile),
    )) {
      throw new Error(
        `${profile.filename} payload does not match a fresh central-recipe build.`,
      );
    }
    const module = new WebAssembly.Module(artifact);
    const relaxedSections = WebAssembly.Module.customSections(
      module,
      WASM_RELAXED_SIMD_SECTION,
    );
    if (profile.recipe.relaxed === null) {
      if (relaxedSections.length !== 0) {
        throw new Error(`${profile.filename} must not embed a relaxed-SIMD child.`);
      }
    } else {
      const relaxed = await fs.readFile(relaxedPath);
      if (relaxedSections.length !== 1 ||
          !Buffer.from(relaxedSections[0]).equals(relaxed)) {
        throw new Error(
          `${profile.filename} relaxed-SIMD child does not match its central recipe.`,
        );
      }
    }
    const ptqAuthoringSections = WebAssembly.Module.customSections(
      module,
      WASM_PTQ_AUTHORING_SECTION,
    );
    if (profile.recipe.ptqAuthoring === null) {
      if (ptqAuthoringSections.length !== 0) {
        throw new Error(
          `${profile.filename} must not embed the full-profile PTQ authoring child.`,
        );
      }
    } else {
      const ptqAuthoring = await fs.readFile(ptqAuthoringPath);
      if (ptqAuthoringSections.length !== 1 ||
          !Buffer.from(ptqAuthoringSections[0]).equals(ptqAuthoring)) {
        throw new Error(
          `${profile.filename} PTQ authoring child does not match its central recipe.`,
        );
      }
    }
    if (!actualEvidence.equals(freshEvidence)) {
      throw new Error(
        `${profile.filename} object/toolchain evidence does not match a fresh ` +
        'central-recipe build.',
      );
    }
    return recorded.sha256;
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
}

async function main() {
  const [command, ...args] = process.argv.slice(2);
  const artifactPath = path.resolve(repositoryRoot, option(args, '--artifact'));
  const selectedBuildEvidence = optionalOption(args, '--build-evidence');
  const buildEvidencePath = selectedBuildEvidence === undefined
    ? wasmReleaseEvidencePath(repositoryRoot, path.basename(artifactPath))
    : path.resolve(repositoryRoot, selectedBuildEvidence);
  const version = await packageVersion();
  if (command === 'write') {
    const outputPath = path.resolve(repositoryRoot, option(args, '--output'));
    const buildEvidenceSha256 = await verifyRecipePayload(
      artifactPath,
      buildEvidencePath,
    );
    const provenance = await createWasmProvenance(
      repositoryRoot,
      artifactPath,
      version,
      { buildEvidenceSha256 },
    );
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, `${stableJSON(provenance)}\n`);
    console.log(`Wrote ${path.relative(repositoryRoot, outputPath)} for ${path.basename(artifactPath)}.`);
    return;
  }
  if (command === 'check') {
    const profile = wasmProfile(path.basename(artifactPath));
    const snapshot = await validateWasmReleaseBuildSnapshot({
      repositoryRoot,
      artifact: artifactPath,
      evidence: buildEvidencePath,
      profile,
      packageVersion: version,
    });
    const result = snapshot.validatedArtifact;
    console.log(
      `Verified ${path.basename(artifactPath)} (${result.artifactSha256}, ` +
      `${result.boundary.exports.length} exports).`,
    );
    return;
  }
  throw new Error(
    "Use 'write --artifact <wasm> --output <json> [--build-evidence <json>]' or " +
    "'check --artifact <wasm> [--build-evidence <json>]'.",
  );
}

await main();
