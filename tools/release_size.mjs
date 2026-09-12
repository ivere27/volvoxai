import { execFile, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

import {
  validateWasmReleaseBuildSnapshot,
  wasmReleaseEvidencePath} from './build_wasm_release.mjs';
import {
  DIST_RELEASE_FILENAMES,
  FIXED_RELEASE_FILENAMES,
  NATIVE_RELEASE_FILENAMES,
  RELEASE_PROFILES,
  inspectBrowserReleaseBundle} from './release_profiles.mjs';
import {
  WASM_INTERNAL_ABI_FORMAT,
  WASM_INTERNAL_ABI_MANIFEST_SHA256} from './generated/wasmInternalAbi.mjs';

const executeFile = promisify(execFile);

export const RELEASE_SIZE_REPORT_FORMAT = 'volvoxai-release-size-report/v1';
export const RELEASE_SIZE_BUDGET_FORMAT = 'volvoxai-release-size-budgets/v1';
export const NATIVE_RELEASE_FINALIZATION_FORMAT =
  'volvoxai-native-release-finalization/v1';

const WASM_SECTION_NAMES = Object.freeze({
  0: 'custom',
  1: 'type',
  2: 'import',
  3: 'function',
  4: 'table',
  5: 'memory',
  6: 'global',
  7: 'export',
  8: 'start',
  9: 'element',
  10: 'code',
  11: 'data',
  12: 'data-count',
  13: 'tag',
});

const MARKDOWN_NUMBER = new Intl.NumberFormat('en-US');

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function normalizedPath(value) {
  return value.split(path.sep).join('/');
}

function artifactIdForDist(filename) {
  return `dist/<package-version>/${filename}`;
}

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort()
      .map((key) => [key, stableValue(value[key])]));
  }
  return value;
}

export function stablePrettyJSON(value) {
  return `${JSON.stringify(stableValue(value), null, 2)}\n`;
}

/**
 * Return `gzip -9 -n` bytes. No filename or wall-clock timestamp is stored,
 * matching the release plan's compressed-size definition exactly.
 */
export function deterministicGzip(bytes) {
  const executable = process.env.GZIP ?? 'gzip';
  const result = spawnSync(executable, ['-9', '-n', '-c'], {
    input: bytes,
    maxBuffer: Math.max(16 * 1024 * 1024, bytes.byteLength * 2),
  });
  if (result.error !== undefined || result.status !== 0) {
    const detail = result.stderr?.toString('utf8').trim();
    throw new Error(
      `Deterministic compression requires '${executable} -9 -n -c'` +
      `${detail ? `: ${detail}` : '.'}`,
      { cause: result.error },
    );
  }
  return Buffer.from(result.stdout);
}

export function deterministicGzipSize(bytes) {
  return deterministicGzip(bytes).byteLength;
}

function readVarUint32(bytes, start) {
  let value = 0;
  let multiplier = 1;
  let offset = start;
  for (let count = 0; count < 5 && offset < bytes.byteLength; count++) {
    const byte = bytes[offset++];
    value += (byte & 0x7f) * multiplier;
    if ((byte & 0x80) === 0) {
      if (!Number.isSafeInteger(value) || value > 0xffff_ffff) {
        throw new Error(`WebAssembly varuint32 overflows at byte ${start}.`);
      }
      return Object.freeze({ value, offset });
    }
    multiplier *= 128;
  }
  throw new Error(`Invalid WebAssembly varuint32 at byte ${start}.`);
}

/** Parse section sizes without depending on an external WebAssembly tool. */
export function parseWasmSections(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const expectedHeader = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
  if (bytes.byteLength < expectedHeader.length ||
      expectedHeader.some((value, index) => bytes[index] !== value)) {
    throw new Error('Input is not a version-1 WebAssembly module.');
  }

  const decoder = new TextDecoder('utf-8', { fatal: true });
  const sections = [];
  let offset = expectedHeader.length;
  while (offset < bytes.byteLength) {
    const sectionStart = offset;
    const id = bytes[offset++];
    const decodedSize = readVarUint32(bytes, offset);
    offset = decodedSize.offset;
    const payloadStart = offset;
    const sectionEnd = payloadStart + decodedSize.value;
    if (sectionEnd > bytes.byteLength) {
      throw new Error(`WebAssembly section at byte ${sectionStart} exceeds the module.`);
    }

    let name = WASM_SECTION_NAMES[id] ?? `unknown-${id}`;
    let customName = null;
    let contentBytes = decodedSize.value;
    let contentOffset = payloadStart;
    if (id === 0) {
      const decodedNameSize = readVarUint32(bytes, payloadStart);
      const nameEnd = decodedNameSize.offset + decodedNameSize.value;
      if (nameEnd > sectionEnd) {
        throw new Error(`WebAssembly custom-section name at byte ${sectionStart} exceeds its payload.`);
      }
      customName = decoder.decode(bytes.subarray(decodedNameSize.offset, nameEnd));
      name = `custom:${customName}`;
      contentBytes = sectionEnd - nameEnd;
      contentOffset = nameEnd;
    }

    sections.push(Object.freeze({
      index: sections.length,
      id,
      name,
      ...(customName === null ? {} : { customName }),
      payloadBytes: decodedSize.value,
      contentBytes,
      contentOffset,
      totalBytes: sectionEnd - sectionStart,
      offset: sectionStart,
    }));
    offset = sectionEnd;
  }
  return Object.freeze(sections);
}

export function inspectWasmBytes(input) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  let module;
  try {
    module = new WebAssembly.Module(bytes);
  } catch (error) {
    throw new Error('Artifact is not a valid WebAssembly module.', { cause: error });
  }
  const sections = parseWasmSections(bytes);
  const imports = WebAssembly.Module.imports(module)
    .map(({ module: namespace, name, kind }) => `${namespace}.${name}:${kind}`)
    .sort();
  const exports = WebAssembly.Module.exports(module)
    .map(({ name, kind }) => `${name}:${kind}`)
    .sort();
  const customSections = sections.filter(({ id }) => id === 0)
    .map(({ index, customName, contentBytes, contentOffset, payloadBytes, totalBytes }) =>
      Object.freeze({
      index,
      name: customName,
      contentBytes,
      payloadBytes,
      totalBytes,
      contentSha256: sha256(bytes.subarray(contentOffset, contentOffset + contentBytes)),
    }));
  const sectionBytes = Object.fromEntries(sections
    .filter(({ id }) => id !== 0)
    .map(({ name, totalBytes }) => [name, totalBytes]));
  return Object.freeze({
    sectionBytes: Object.freeze(sectionBytes),
    sections,
    customSections: Object.freeze(customSections),
    importCount: imports.length,
    imports: Object.freeze(imports),
    importSetSha256: sha256(Buffer.from(`${imports.join('\n')}\n`)),
    exportCount: exports.length,
    exports: Object.freeze(exports),
    exportSetSha256: sha256(Buffer.from(`${exports.join('\n')}\n`)),
  });
}

function wasmExportKind(name) {
  if (name === 'memory') return 'memory';
  if (name === '__heap_base') return 'global';
  return 'function';
}

function normalizeWasmExportManifest(manifestOrExports) {
  if (Array.isArray(manifestOrExports)) {
    return Object.freeze({
      authority: Object.freeze({
        kind: 'release-profile-projection',
        source: 'tools/release_profiles.mjs#RELEASE_PROFILES.wasm.requiredExports',
        generated: false,
      }),
      exports: manifestOrExports,
      exportKinds: Object.fromEntries(manifestOrExports
        .map((name) => [name, wasmExportKind(name)])),
    });
  }
  if (manifestOrExports === null || typeof manifestOrExports !== 'object' ||
      !Array.isArray(manifestOrExports.exports) ||
      manifestOrExports.exportKinds === null ||
      typeof manifestOrExports.exportKinds !== 'object') {
    throw new Error('WASM export ownership requires an export-manifest projection.');
  }
  return manifestOrExports;
}

/**
 * Keep linked exports distinct from the required ABI projection. The
 * projection includes its own authority so generated manifests can replace a
 * hand-maintained source without changing the report schema.
 */
export function describeWasmExportOwnership(inspection, manifestOrExports) {
  const manifest = normalizeWasmExportManifest(manifestOrExports);
  const actual = new Map(inspection.exports.map((entry) => {
    const separator = entry.lastIndexOf(':');
    return [entry.slice(0, separator), entry.slice(separator + 1)];
  }));
  const requiredNames = new Set(manifest.exports);
  const required = [...requiredNames].sort().map((name) => {
    const kind = manifest.exportKinds[name];
    if (kind !== 'function' && kind !== 'global' && kind !== 'memory') {
      throw new Error(`WASM export manifest has no valid kind for '${name}'.`);
    }
    return `${name}:${kind}`;
  });
  const missingRequired = required.filter((entry) => {
    const separator = entry.lastIndexOf(':');
    return actual.get(entry.slice(0, separator)) !== entry.slice(separator + 1);
  });
  const unmanaged = inspection.exports.filter((entry) => {
    const separator = entry.lastIndexOf(':');
    return !requiredNames.has(entry.slice(0, separator));
  });
  return Object.freeze({
    authority: manifest.authority,
    linked: Object.freeze({
      count: inspection.exportCount,
      entries: inspection.exports,
      setSha256: inspection.exportSetSha256,
    }),
    manifestRequired: Object.freeze({
      count: required.length,
      entries: Object.freeze(required),
      setSha256: sha256(Buffer.from(`${required.join('\n')}\n`)),
    }),
    missingRequired: Object.freeze(missingRequired),
    unmanaged: Object.freeze({
      count: unmanaged.length,
      entries: Object.freeze(unmanaged),
      setSha256: sha256(Buffer.from(`${unmanaged.join('\n')}\n`)),
    }),
  });
}

export function classifyBrowserInput(inputPath) {
  const input = inputPath.replaceAll('\\', '/');
  if (/(^|\/)runtime\/generated\//.test(input) || /(^|\/)ts\/generated\//.test(input)) {
    return 'generated';
  }
  if (/\.wgsl$/.test(input) || /(^|\/)shaders\//.test(input) ||
      /(^|\/)ts\/backends\/WebGPUHostBridge\.ts$/.test(input)) {
    return 'webgpu-wgsl';
  }
  if (/(^|\/)ts\/core\/(?:ModelControlWasm|WasmReleaseModule)\.ts$/.test(input)) {
    return 'wasm-portable-control';
  }
  if (/(^|\/)ts\/(?:host|core|backends)\//.test(input)) return 'host-runtime';
  return 'other';
}

export function aggregateBrowserMetafile(metafile, outputBytes) {
  const outputs = Object.values(metafile?.outputs ?? {});
  if (outputs.length !== 1) {
    throw new Error(`Expected one esbuild output in metafile; found ${outputs.length}.`);
  }
  const categories = new Map();
  const entries = [];
  let attributedBytes = 0;
  for (const [input, contribution] of Object.entries(outputs[0].inputs ?? {})) {
    const bytesInOutput = Number(contribution.bytesInOutput ?? 0);
    if (!Number.isSafeInteger(bytesInOutput) || bytesInOutput < 0) {
      throw new Error(`Invalid esbuild bytesInOutput for '${input}'.`);
    }
    const category = classifyBrowserInput(input);
    categories.set(category, (categories.get(category) ?? 0) + bytesInOutput);
    attributedBytes += bytesInOutput;
    entries.push(Object.freeze({
      input: input.replaceAll('\\', '/'),
      category,
      bytesInOutput,
    }));
  }
  const overhead = outputBytes - attributedBytes;
  if (!Number.isSafeInteger(outputBytes) || outputBytes < 0 || overhead < 0) {
    throw new Error(
      `Esbuild attributed ${attributedBytes} bytes into a ${outputBytes}-byte output.`,
    );
  }
  categories.set('bundler-overhead', overhead);
  return Object.freeze({
    outputBytes,
    attributedBytes,
    categories: Object.freeze(Object.fromEntries([...categories].sort())),
    inputs: Object.freeze(entries.sort((left, right) =>
      right.bytesInOutput - left.bytesInOutput || left.input.localeCompare(right.input, 'en'))),
  });
}

async function inspectBrowserMinifiedBundle(repositoryRoot, profile, packageVersion, artifactPath) {
  const result = await inspectBrowserReleaseBundle(
    repositoryRoot,
    profile.id,
    packageVersion,
    { minify: true, outfile: artifactPath, logLevel: 'silent' },
  );
  const actual = await fs.readFile(artifactPath);
  const expected = Buffer.from(result.contents);
  if (!actual.equals(expected)) {
    throw new Error(
      `${normalizedPath(path.relative(repositoryRoot, artifactPath))} differs from its ` +
      'release-size esbuild analysis. Run the central browser release build first or update ' +
      'the analysis projection with the central builder.',
    );
  }
  return Object.freeze({
    esbuildVersion: result.esbuildVersion,
    ...aggregateBrowserMetafile(result.metafile, actual.byteLength),
  });
}

async function executableOnPath(candidate) {
  if (candidate.includes(path.sep)) {
    try {
      await fs.access(candidate, fs.constants.X_OK);
      return path.resolve(candidate);
    } catch {
      return null;
    }
  }
  for (const directory of (process.env.PATH ?? '').split(path.delimiter)) {
    if (!directory) continue;
    const resolved = path.join(directory, candidate);
    try {
      await fs.access(resolved, fs.constants.X_OK);
      return resolved;
    } catch {
      // Continue to the next PATH component.
    }
  }
  return null;
}

async function requireLlvmTool(environmentName, candidates) {
  const configured = process.env[environmentName];
  const attempts = [...(configured ? [configured] : []), ...candidates];
  for (const candidate of attempts) {
    const resolved = await executableOnPath(candidate);
    if (resolved !== null) return resolved;
  }
  throw new Error(
    `Native size inspection requires ${candidates.join(' or ')}. ` +
    `Install the pinned LLVM tools or set ${environmentName} explicitly.`,
  );
}

async function toolVersionEvidence(candidate, arguments_ = ['--version']) {
  const resolved = await executableOnPath(candidate);
  if (resolved === null) {
    return Object.freeze({
      status: 'unavailable',
      candidate,
      reason: `Executable '${candidate}' was not found on PATH.`,
    });
  }
  const version = (await commandText(resolved, arguments_)).split(/\r?\n/, 1)[0].trim();
  return Object.freeze({
    status: 'available',
    path: normalizedPath(resolved),
    version,
  });
}

async function commandText(executable, arguments_, options = {}) {
  try {
    const { stdout } = await executeFile(executable, arguments_, {
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      ...options,
    });
    return stdout;
  } catch (error) {
    const detail = [error?.stdout, error?.stderr].filter(Boolean).join('\n').trim();
    throw new Error(
      `${path.basename(executable)} ${arguments_.join(' ')} failed` +
      `${detail ? `:\n${detail}` : '.'}`,
      { cause: error },
    );
  }
}

export function parseLlvmReadobjSections(output) {
  const sections = [];
  for (const line of output.split(/\r?\n/)) {
    const match = line.match(
      /^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+([0-9A-Fa-f]+)\b/,
    );
    if (!match || match[1] === 'NULL') continue;
    const sizeBytes = Number.parseInt(match[2], 16);
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes < 0) continue;
    sections.push(Object.freeze({ name: match[1], sizeBytes }));
  }
  if (sections.length === 0) {
    throw new Error('llvm-readobj did not emit parseable GNU-style ELF sections.');
  }
  return Object.freeze(sections);
}

export function parseLlvmNmSymbols(output, limit = 30) {
  const symbols = [];
  for (const line of output.split(/\r?\n/)) {
    const fields = line.trim().split(/\s+/);
    if (fields.length < 4) continue;
    const sizeBytes = Number(fields.at(-1));
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes <= 0) continue;
    const value = fields.at(-2);
    const type = fields.at(-3);
    const name = fields.slice(0, -3).join(' ');
    if (!name) continue;
    symbols.push(Object.freeze({ name, type, value, sizeBytes }));
  }
  return Object.freeze(symbols
    .sort((left, right) => right.sizeBytes - left.sizeBytes ||
      left.name.localeCompare(right.name, 'en'))
    .slice(0, limit));
}

export function parseLlvmReadobjBuildId(output) {
  const buildIds = [...output.matchAll(/\bBuild ID:\s*([0-9A-Fa-f]+)\b/gu)]
    .map((match) => match[1].toLowerCase());
  if (buildIds.length !== 1) {
    throw new Error(
      `llvm-readobj did not emit exactly one GNU build ID; found ${buildIds.length}.`,
    );
  }
  return buildIds[0];
}

export function validateStrippedNativeSections(sections, label = 'Native release ELF') {
  const names = new Set(sections.map(({ name }) => name));
  const forbidden = [...names].filter((name) =>
    name === '.symtab' || name === '.strtab' || name.startsWith('.debug'));
  if (forbidden.length > 0) {
    throw new Error(
      `${label} retains stripped symbol/debug sections: ${forbidden.sort().join(', ')}.`,
    );
  }
  const required = ['.dynsym', '.dynstr', '.gnu_debuglink'];
  const missing = required.filter((name) => !names.has(name));
  if (missing.length > 0) {
    throw new Error(
      `${label} is missing required runtime/debuglink sections: ${missing.join(', ')}.`,
    );
  }
  return Object.freeze({
    stripped: true,
    requiredSections: Object.freeze(required),
    forbiddenSections: Object.freeze([]),
  });
}

export function validateNativeDebugSections(sections, label = 'Native debug sidecar') {
  const names = new Set(sections.map(({ name }) => name));
  const required = ['.symtab', '.strtab', '.debug_info'];
  const missing = required.filter((name) => !names.has(name));
  if (missing.length > 0) {
    throw new Error(
      `${label} is missing required symbol/debug sections: ${missing.join(', ')}.`,
    );
  }
  return Object.freeze({
    symbolized: true,
    requiredSections: Object.freeze(required),
  });
}

function requireNativeArtifactFingerprint(artifact, expectedPath, label) {
  if (artifact === null || typeof artifact !== 'object') {
    throw new Error(`${label} fingerprint is missing.`);
  }
  if (artifact.path !== normalizedPath(expectedPath)) {
    throw new Error(
      `${label} path '${artifact.path}' does not match '${normalizedPath(expectedPath)}'.`,
    );
  }
  if (!Number.isSafeInteger(artifact.rawBytes) || artifact.rawBytes <= 0) {
    throw new Error(`${label} rawBytes must be a positive safe integer.`);
  }
  if (!/^[0-9a-f]{64}$/u.test(artifact.sha256)) {
    throw new Error(`${label} SHA-256 is invalid.`);
  }
}

function requireNativeBuildId(value, label) {
  if (typeof value !== 'string' || !/^[0-9a-f]+$/u.test(value)) {
    throw new Error(`${label} is not a lowercase hexadecimal build ID.`);
  }
  return value;
}

function sameStableValue(left, right) {
  return JSON.stringify(stableValue(left)) === JSON.stringify(stableValue(right));
}

export function validateNativeReleaseFinalization({
  profile,
  releaseArtifact,
  debugArtifact,
  evidenceArtifact,
  buildArtifact,
  buildDebugArtifact,
  buildEvidenceArtifact,
  buildDirectory,
  linkMap,
  evidence,
  releaseBuildId,
  debugBuildId,
}) {
  if (profile === null || typeof profile !== 'object' ||
      typeof profile.filename !== 'string' ||
      typeof profile.debugArtifact !== 'string' ||
      typeof profile.debugEvidence !== 'string') {
    throw new Error('Native release profile has no debug artifact/evidence contract.');
  }
  requireNativeArtifactFingerprint(releaseArtifact, profile.filename, 'Native release');
  requireNativeArtifactFingerprint(debugArtifact, profile.debugArtifact, 'Native debug sidecar');
  requireNativeArtifactFingerprint(
    evidenceArtifact,
    profile.debugEvidence,
    'Native finalization evidence',
  );
  for (const [artifact, label] of [
    [buildArtifact, 'Build-private native release'],
    [buildDebugArtifact, 'Build-private native debug sidecar'],
    [buildEvidenceArtifact, 'Build-private native finalization evidence'],
  ]) {
    if (artifact === null || typeof artifact !== 'object' ||
        typeof artifact.path !== 'string' ||
        !Number.isSafeInteger(artifact.rawBytes) || artifact.rawBytes <= 0 ||
        !/^[0-9a-f]{64}$/u.test(artifact.sha256)) {
      throw new Error(`${label} fingerprint is missing or invalid.`);
    }
  }
  if (typeof buildDirectory !== 'string' || buildDirectory.length === 0) {
    throw new Error('Selected native CMake build directory is missing.');
  }
  const expectedBuildArtifact = normalizedPath(path.join(
    buildDirectory,
    'native',
    'release-link',
    path.basename(profile.filename),
  ));
  const expectedBuildDebug = normalizedPath(path.join(
    buildDirectory,
    'native',
    'release-link',
    '.debug',
    path.basename(profile.debugArtifact),
  ));
  const expectedBuildEvidence = normalizedPath(path.join(
    buildDirectory,
    'native',
    'release-link',
    '.debug',
    path.basename(profile.debugEvidence),
  ));
  for (const [actual, expected, label] of [
    [buildArtifact.path, expectedBuildArtifact, 'release artifact'],
    [buildDebugArtifact.path, expectedBuildDebug, 'debug sidecar'],
    [buildEvidenceArtifact.path, expectedBuildEvidence, 'finalization evidence'],
  ]) {
    if (normalizedPath(actual) !== expected) {
      throw new Error(
        `Build-private native ${label} '${actual}' is outside selected build ` +
        `directory '${buildDirectory}'.`,
      );
    }
  }
  if (evidence === null || typeof evidence !== 'object' || Array.isArray(evidence)) {
    throw new Error('Native finalization evidence must be a JSON object.');
  }
  if (evidence.format !== NATIVE_RELEASE_FINALIZATION_FORMAT) {
    throw new Error(
      `Native finalization evidence format '${evidence.format}' is not ` +
      `'${NATIVE_RELEASE_FINALIZATION_FORMAT}'.`,
    );
  }
  if (evidence.artifact !== normalizedPath(profile.filename) ||
      evidence.debugArtifact !== normalizedPath(profile.debugArtifact)) {
    throw new Error(
      'Native finalization evidence paths do not match the release profile.',
    );
  }
  if (!sameStableValue(evidence.before, evidence.after)) {
    throw new Error('Native finalization evidence pre/post snapshots differ.');
  }
  if (typeof evidence.before?.version !== 'string' ||
      evidence.before.version.length === 0) {
    throw new Error('Native finalization evidence has no verified version provenance.');
  }

  const build = evidence.build;
  if (build === null || typeof build !== 'object' || Array.isArray(build) ||
      build.directory !== normalizedPath(buildDirectory)) {
    throw new Error('Native finalization evidence belongs to another build directory.');
  }
  if (build.artifact?.path !== expectedBuildArtifact ||
      build.artifact?.rawBytes !== buildArtifact.rawBytes ||
      build.artifact?.sha256 !== buildArtifact.sha256) {
    throw new Error(
      'Native finalization evidence build artifact does not match the selected build tree.',
    );
  }
  if (linkMap === null || typeof linkMap !== 'object' ||
      build.linkMap?.path !== linkMap.path ||
      build.linkMap?.rawBytes !== linkMap.rawBytes ||
      build.linkMap?.sha256 !== linkMap.mapSha256) {
    throw new Error(
      'Native finalization evidence linker map does not match the selected build tree.',
    );
  }

  const actualReleaseBuildId = requireNativeBuildId(
    releaseBuildId,
    'Native release build ID',
  );
  const actualDebugBuildId = requireNativeBuildId(
    debugBuildId,
    'Native debug build ID',
  );
  const beforeBuildId = requireNativeBuildId(
    evidence.before?.buildId,
    'Native finalization pre-strip build ID',
  );
  const afterBuildId = requireNativeBuildId(
    evidence.after?.buildId,
    'Native finalization post-strip build ID',
  );
  const declaredDebugBuildId = requireNativeBuildId(
    evidence.debug?.buildId,
    'Native finalization debug build ID',
  );
  const buildIds = new Set([
    actualReleaseBuildId,
    actualDebugBuildId,
    beforeBuildId,
    afterBuildId,
    declaredDebugBuildId,
  ]);
  if (buildIds.size !== 1) {
    throw new Error('Native release, debug sidecar, and evidence build IDs differ.');
  }
  if (evidence.release?.rawBytes !== releaseArtifact.rawBytes ||
      evidence.release?.sha256 !== releaseArtifact.sha256) {
    throw new Error(
      'Native finalization evidence release size/SHA-256 does not match the artifact.',
    );
  }
  if (evidence.debug?.rawBytes !== debugArtifact.rawBytes ||
      evidence.debug?.sha256 !== debugArtifact.sha256) {
    throw new Error(
      'Native finalization evidence debug size/SHA-256 does not match the sidecar.',
    );
  }
  if (buildArtifact.rawBytes !== releaseArtifact.rawBytes ||
      buildArtifact.sha256 !== releaseArtifact.sha256 ||
      buildDebugArtifact.rawBytes !== debugArtifact.rawBytes ||
      buildDebugArtifact.sha256 !== debugArtifact.sha256 ||
      buildEvidenceArtifact.rawBytes !== evidenceArtifact.rawBytes ||
      buildEvidenceArtifact.sha256 !== evidenceArtifact.sha256) {
    throw new Error(
      'Published native release bundle differs from the selected build tree.',
    );
  }
  if (evidence.release?.hasGnuDebuglink !== true) {
    throw new Error('Native finalization evidence does not prove a GNU debuglink.');
  }
  if (!Array.isArray(evidence.release?.definedDynamicExports) ||
      evidence.release.definedDynamicExports.length !== 0) {
    throw new Error(
      'Native release executable must have definedDynamicExports=[].',
    );
  }

  return Object.freeze({
    ...stableValue(evidence),
    path: evidenceArtifact.path,
    rawBytes: evidenceArtifact.rawBytes,
    sha256: evidenceArtifact.sha256,
    buildId: actualReleaseBuildId,
    beforeEqualsAfter: true,
  });
}

function nativeObjectGroup(objectPath) {
  const object = objectPath.replaceAll('\\', '/');
  if (/runtime\/generated\/|\/gen\/api\//.test(object)) return 'generated-api';
  if (/\/src\/training\/|vx_api_(?:training|quantization)/.test(object)) {
    return 'training-ptq';
  }
  if (/\/src\/kernels\//.test(object)) return 'kernels';
  if (/\/src\/backends\//.test(object)) return 'backends';
  if (/embedded_(?:shaders|cuda)|shader_store|xz-embedded/.test(object)) return 'shaders';
  if (/\/src\/runtime\//.test(object)) return 'runtime-control';
  if (/\/src\/api\/|\/cli\//.test(object)) return 'api-cli';
  if (/\/third_party\//.test(object)) return 'third-party';
  return 'other';
}

// GNU ld attributes linker-created dynamic/relocation rows to an arbitrary
// input, commonly Scrt1.o. Treating those rows as object ownership makes that
// tiny startup object appear hundreds of KiB larger than it is. Object
// attribution therefore covers file-backed, allocatable input sections only;
// ELF section totals report relocations and symbol tables separately. NOBITS
// sections such as .bss/.tbss do not contribute artifact bytes.
function isFileBackedNativeObjectSection(section) {
  return /^(?:\.text|\.rodata|\.data|\.sdata|\.tdata|\.init|\.fini|\.ctors|\.dtors|\.preinit_array|\.init_array|\.fini_array|\.eh_frame|\.gcc_except_table|\.ARM\.extab|\.ARM\.exidx|\.gnu\.linkonce\.(?:t|r|d|s|td))(?:\.|$)/
    .test(section);
}

const NATIVE_OBJECT_TOKEN = String.raw`\S+(?:\.a\([^)]+\.(?:o|obj|oS)\)|\.(?:o|obj))`;

/** Parse file-backed input-section contributions from GNU ld or lld map text. */
export function parseNativeLinkMapObjects(output, limit = 30) {
  const hasGnuMarker = output.includes('Linker script and memory map');
  let inMemoryMap = !hasGnuMarker;
  let pendingGnuSection = null;
  const objects = new Map();
  for (const line of output.split(/\r?\n/)) {
    if (line.includes('Linker script and memory map')) {
      inMemoryMap = true;
      pendingGnuSection = null;
      continue;
    }
    if (!inMemoryMap) continue;
    if (/^OUTPUT\(/.test(line)) break;
    let match = line.match(new RegExp(
      `^\\s*(?:(\\.[^\\s]+)\\s+)?0x[0-9A-Fa-f]+\\s+` +
      `0x([0-9A-Fa-f]+)\\s+(${NATIVE_OBJECT_TOKEN})(?:\\s|$)`,
    ));
    let sizeBytes;
    let object;
    let section;
    if (match) {
      section = match[1] ?? pendingGnuSection;
      sizeBytes = Number.parseInt(match[2], 16);
      object = match[3];
    } else {
      // lld: VMA LMA Size Align Out/In. Input rows end in object:(section).
      match = line.match(new RegExp(
        `^\\s*[0-9A-Fa-f]+\\s+[0-9A-Fa-f]+\\s+([0-9A-Fa-f]+)\\s+` +
        `\\d+\\s+(${NATIVE_OBJECT_TOKEN}):\\((\\.[^)]+)\\)`,
      ));
      if (!match) {
        const pending = line.match(/^\s+(\.[^\s]+)\s*$/);
        pendingGnuSection = pending?.[1] ?? null;
        continue;
      }
      sizeBytes = Number.parseInt(match[1], 16);
      object = match[2];
      section = match[3];
    }
    pendingGnuSection = null;
    if (!isFileBackedNativeObjectSection(section ?? '')) continue;
    if (!Number.isSafeInteger(sizeBytes) || sizeBytes <= 0) continue;
    const normalized = object.replaceAll('\\', '/');
    objects.set(normalized, (objects.get(normalized) ?? 0) + sizeBytes);
  }
  const entries = [...objects].map(([object, sizeBytes]) => Object.freeze({
    object,
    group: nativeObjectGroup(object),
    sizeBytes,
  })).sort((left, right) => right.sizeBytes - left.sizeBytes ||
    left.object.localeCompare(right.object, 'en'));
  const groups = new Map();
  for (const entry of entries) {
    groups.set(entry.group, (groups.get(entry.group) ?? 0) + entry.sizeBytes);
  }
  return Object.freeze({
    attributionBasis: 'file-backed allocatable object input sections',
    objectCount: entries.length,
    attributedBytes: entries.reduce((total, entry) => total + entry.sizeBytes, 0),
    groups: Object.freeze(Object.fromEntries([...groups].sort())),
    topObjects: Object.freeze(entries.slice(0, limit)),
  });
}

export function validateNativeLinkMapTarget(
  output,
  linkDirectory,
  artifactPath,
  label = 'Native linker map',
) {
  const outputs = [...output.matchAll(/^OUTPUT\((\S+)(?:\s+[^)]*)?\)\s*$/gm)]
    .map((match) => match[1]);
  if (outputs.length !== 1) {
    throw new Error(
      `${label} must name exactly one OUTPUT; found ${outputs.length}.`,
    );
  }
  const linkedOutput = path.resolve(linkDirectory, outputs[0]);
  if (linkedOutput !== path.resolve(artifactPath)) {
    throw new Error(
      `${label} describes '${normalizedPath(linkedOutput)}', not release artifact ` +
      `'${normalizedPath(path.resolve(artifactPath))}'. Rebuild the release profile after ` +
      'any test relinks.',
    );
  }
  return linkedOutput;
}

async function inspectNativeLinkMap(
  repositoryRoot,
  mapPath,
  artifactPath,
  linkDirectory,
) {
  const relativePath = normalizedPath(path.relative(repositoryRoot, mapPath));
  let contents;
  try {
    contents = await fs.readFile(mapPath);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    throw new Error(
      `Missing native release linker map '${relativePath}'; rebuild both native profiles.`,
    );
  }
  const text = contents.toString('utf8');
  const linkedOutput = validateNativeLinkMapTarget(
    text,
    linkDirectory,
    artifactPath,
    `Native linker map '${relativePath}'`,
  );
  const attribution = parseNativeLinkMapObjects(text);
  if (attribution.objectCount === 0) {
    throw new Error(
      `Native linker map '${relativePath}' contained no file-backed object contributions.`,
    );
  }
  return Object.freeze({
    status: 'available',
    path: relativePath,
    rawBytes: contents.byteLength,
    linkedOutput: normalizedPath(path.relative(repositoryRoot, linkedOutput)),
    mapSha256: sha256(contents),
    ...attribution,
  });
}

async function inspectNativeElf({
  releaseArtifact,
  debugArtifact,
  evidenceArtifact,
  buildArtifact,
  buildDebugArtifact,
  buildEvidenceArtifact,
  evidence,
  profile,
  tools,
  repositoryRoot,
  buildDirectory,
  mapPath,
  linkArtifactPath,
  linkDirectory,
}) {
  const releasePath = path.join(repositoryRoot, releaseArtifact.path);
  const debugPath = path.join(repositoryRoot, debugArtifact.path);
  const [
    releaseSectionsOutput,
    debugSectionsOutput,
    symbolsOutput,
    releaseNotesOutput,
    debugNotesOutput,
    objectAttribution,
  ] = await Promise.all([
    commandText(tools.readobj, ['--sections', '--elf-output-style=GNU', releasePath]),
    commandText(tools.readobj, ['--sections', '--elf-output-style=GNU', debugPath]),
    commandText(tools.nm, [
      '--defined-only', '--print-size', '--size-sort', '--format=posix',
      '--radix=d', debugPath,
    ]),
    commandText(tools.readobj, ['--notes', releasePath]),
    commandText(tools.readobj, ['--notes', debugPath]),
    inspectNativeLinkMap(
      repositoryRoot,
      mapPath,
      linkArtifactPath,
      linkDirectory,
    ),
  ]);
  const sections = parseLlvmReadobjSections(releaseSectionsOutput);
  const debugSections = parseLlvmReadobjSections(debugSectionsOutput);
  const releaseSectionPolicy = validateStrippedNativeSections(
    sections,
    `Native release ELF '${releaseArtifact.path}'`,
  );
  const debugSectionPolicy = validateNativeDebugSections(
    debugSections,
    `Native debug sidecar '${debugArtifact.path}'`,
  );
  const topSymbols = parseLlvmNmSymbols(symbolsOutput);
  if (topSymbols.length === 0) {
    throw new Error(
      `${debugArtifact.path} has no inspectable symbols for size attribution.`,
    );
  }
  const releaseBuildId = parseLlvmReadobjBuildId(releaseNotesOutput);
  const debugBuildId = parseLlvmReadobjBuildId(debugNotesOutput);
  const finalizationEvidence = validateNativeReleaseFinalization({
    profile,
    releaseArtifact,
    debugArtifact,
    evidenceArtifact,
    buildArtifact,
    buildDebugArtifact,
    buildEvidenceArtifact,
    buildDirectory: normalizedPath(path.relative(repositoryRoot, buildDirectory)),
    linkMap: objectAttribution,
    evidence,
    releaseBuildId,
    debugBuildId,
  });
  const named = Object.fromEntries(sections.map(({ name, sizeBytes }) => [name, sizeBytes]));
  const relocationBytes = sections
    .filter(({ name }) => /^\.rela?(?:\.|$)/.test(name))
    .reduce((total, { sizeBytes }) => total + sizeBytes, 0);
  return Object.freeze({
    sectionBytes: Object.freeze({
      text: named['.text'] ?? 0,
      rodata: named['.rodata'] ?? 0,
      data: named['.data'] ?? 0,
      dataRelRo: named['.data.rel.ro'] ?? 0,
      ehFrame: (named['.eh_frame'] ?? 0) + (named['.eh_frame_hdr'] ?? 0),
      dynsym: named['.dynsym'] ?? 0,
      dynstr: named['.dynstr'] ?? 0,
      gnuDebuglink: named['.gnu_debuglink'] ?? 0,
      symtab: named['.symtab'] ?? 0,
      strtab: named['.strtab'] ?? 0,
      relocations: relocationBytes,
    }),
    sections,
    sectionSource: releaseArtifact.path,
    topSymbols,
    topSymbolSource: debugArtifact.path,
    releaseElf: Object.freeze({
      path: releaseArtifact.path,
      rawBytes: releaseArtifact.rawBytes,
      sha256: releaseArtifact.sha256,
      buildId: releaseBuildId,
      sectionPolicy: releaseSectionPolicy,
    }),
    debugSidecar: Object.freeze({
      path: debugArtifact.path,
      rawBytes: debugArtifact.rawBytes,
      sha256: debugArtifact.sha256,
      buildId: debugBuildId,
      sectionPolicy: debugSectionPolicy,
    }),
    finalizationEvidence,
    objectAttribution,
  });
}

async function basicArtifact(repositoryRoot, id, relativePath, kind, profile, variant) {
  const absolutePath = path.join(repositoryRoot, relativePath);
  let bytes;
  try {
    bytes = await fs.readFile(absolutePath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`Missing fixed release artifact: ${normalizedPath(relativePath)}.`);
    }
    throw error;
  }
  if (bytes.byteLength === 0) {
    throw new Error(`Fixed release artifact is empty: ${normalizedPath(relativePath)}.`);
  }
  return {
    id,
    path: normalizedPath(relativePath),
    kind,
    profile,
    variant,
    rawBytes: bytes.byteLength,
    gzipBytes: deterministicGzipSize(bytes),
    sha256: sha256(bytes),
    bytes,
  };
}

async function provenanceArtifact(repositoryRoot, relativePath, label) {
  const absolutePath = path.join(repositoryRoot, relativePath);
  let bytes;
  try {
    bytes = await fs.readFile(absolutePath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`Missing ${label}: ${normalizedPath(relativePath)}.`);
    }
    throw error;
  }
  if (bytes.byteLength === 0) {
    throw new Error(`${label} is empty: ${normalizedPath(relativePath)}.`);
  }
  return Object.freeze({
    path: normalizedPath(relativePath),
    rawBytes: bytes.byteLength,
    sha256: sha256(bytes),
    bytes,
  });
}

async function gitEvidence(repositoryRoot) {
  try {
    const [commit, status] = await Promise.all([
      commandText('git', ['rev-parse', 'HEAD'], { cwd: repositoryRoot }),
      commandText('git', ['status', '--porcelain=v1'], { cwd: repositoryRoot }),
    ]);
    return Object.freeze({ commit: commit.trim(), dirty: status.trim().length > 0 });
  } catch {
    return Object.freeze({ commit: 'unknown', dirty: null });
  }
}

function stableHash(value) {
  return sha256(Buffer.from(JSON.stringify(stableValue(value))));
}

async function cmakeCacheEvidence(repositoryRoot, buildDirectory) {
  const cachePath = path.join(buildDirectory, 'CMakeCache.txt');
  const relative = normalizedPath(path.relative(repositoryRoot, cachePath));
  let bytes;
  try {
    bytes = await fs.readFile(cachePath);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return Object.freeze({
      status: 'unavailable',
      expectedPath: relative,
      reason: 'Configure the native CMake build before collecting full size evidence.',
    });
  }
  const entries = new Map();
  for (const line of bytes.toString('utf8').split(/\r?\n/)) {
    const match = line.match(/^([^#/:][^:]*):[^=]+=(.*)$/);
    if (match) entries.set(match[1], match[2]);
  }
  const selectedNames = [
    'CMAKE_C_COMPILER',
    'CMAKE_LINKER',
    'CMAKE_C_COMPILER_ID',
    'CMAKE_C_COMPILER_VERSION',
    'CMAKE_SYSTEM_NAME',
    'CMAKE_SYSTEM_PROCESSOR',
    'VOLVOXAI_CPU_TARGET',
    'VOLVOXAI_ENABLE_VULKAN',
    'VOLVOXAI_ENABLE_OPENGL',
    'VOLVOXAI_ENABLE_CUDA',
    'VOLVOXAI_ENABLE_METAL',
    'VOLVOXAI_CUDA_ARCH',
    'VOLVOXAI_CUDA_FAST_FP32',
  ];
  return Object.freeze({
    status: 'available',
    path: relative,
    sha256: sha256(bytes),
    entryCount: entries.size,
    selected: Object.freeze(Object.fromEntries(selectedNames
      .filter((name) => entries.has(name))
      .map((name) => [name, entries.get(name)]))),
  });
}

async function compileCommandsEvidence(repositoryRoot, buildDirectory) {
  const commandsPath = path.join(buildDirectory, 'compile_commands.json');
  const relative = normalizedPath(path.relative(repositoryRoot, commandsPath));
  let bytes;
  try {
    bytes = await fs.readFile(commandsPath);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return Object.freeze({
      status: 'unavailable',
      expectedPath: relative,
      reason: 'Reconfigure with CMAKE_EXPORT_COMPILE_COMMANDS=ON and rebuild native profiles.',
    });
  }
  let commands;
  try {
    commands = JSON.parse(bytes.toString('utf8'));
  } catch (error) {
    throw new Error(`${relative} is not valid JSON.`, { cause: error });
  }
  if (!Array.isArray(commands)) throw new Error(`${relative} must contain a JSON array.`);
  const groupPatterns = Object.freeze([
    ['inferenceHot',
      /CMakeFiles\/volvoxai_inference_release_[^/]+_hot_objects\.dir\//u],
    ['inferenceCold',
      /CMakeFiles\/volvoxai_inference_release_[^/]+_cold_objects\.dir\//u],
    ['fullHot', /CMakeFiles\/volvoxai_full_release_[^/]+_hot_objects\.dir\//u],
    ['fullCold', /CMakeFiles\/volvoxai_full_release_[^/]+_cold_objects\.dir\//u],
    ['releaseArmHot', /CMakeFiles\/volvox_release_arm_(?:dotprod|i8mm)\.dir\//u],
    ['inferenceLibrary', /CMakeFiles\/volvoxai_objects\.dir\//u],
    ['fullLibrary', /CMakeFiles\/volvoxai-full_objects\.dir\//u],
    ['inferenceCli',
      /CMakeFiles\/volvoxai_inference_release_cli_object\.dir\//u],
    ['fullCli', /CMakeFiles\/volvoxai_full_release_cli_object\.dir\//u],
  ]);
  const grouped = Object.fromEntries(groupPatterns.map(([name]) => [name, []]));
  for (const entry of commands) {
    const command = Array.isArray(entry.arguments)
      ? entry.arguments.join(' ')
      : String(entry.command ?? '');
    const output = `${entry.output ?? ''} ${command}`.replaceAll('\\', '/');
    const match = groupPatterns.find(([, pattern]) => pattern.test(output));
    if (match === undefined) continue;
    const tokens = Array.isArray(entry.arguments)
      ? entry.arguments.map(String)
      : command.trim().split(/\s+/u);
    const optimizations = tokens.filter((token) => /^-O(?:[0-3]|fast|g|s|z)$/u.test(token));
    grouped[match[0]].push(Object.freeze({
      source: normalizedPath(path.resolve(
        entry.directory ?? repositoryRoot,
        entry.file ?? '',
      )),
      compiler: normalizedPath(tokens[0] ?? ''),
      optimization: optimizations.at(-1) ?? null,
      hasDebugInfo: tokens.includes('-g'),
      hasFunctionSections: tokens.includes('-ffunction-sections'),
      hasDataSections: tokens.includes('-fdata-sections'),
      hasDebugPrefixMap: tokens.some((token) => token.startsWith('-fdebug-prefix-map=')),
    }));
  }

  const requireGroup = (name, expectedOptimization, releaseExecutable) => {
    const entries = grouped[name];
    if (entries.length === 0) {
      throw new Error(`Native compile evidence has no '${name}' commands.`);
    }
    const sources = new Set();
    for (const entry of entries) {
      if (sources.has(entry.source)) {
        throw new Error(`Native compile group '${name}' repeats ${entry.source}.`);
      }
      sources.add(entry.source);
      const expected = expectedOptimization === 'size'
        ? (/\bclang(?:-[0-9]+)?$/u.test(entry.compiler) ? '-Oz' : '-Os')
        : expectedOptimization;
      if (entry.optimization !== expected) {
        throw new Error(
          `Native compile group '${name}' uses ${entry.optimization ?? 'no optimization'} ` +
          `for ${entry.source}; expected ${expected}.`,
        );
      }
      if (releaseExecutable && (!entry.hasDebugInfo || !entry.hasFunctionSections ||
          !entry.hasDataSections || !entry.hasDebugPrefixMap)) {
        throw new Error(
          `Native release compile group '${name}' is missing debug/section provenance ` +
          `flags for ${entry.source}.`,
        );
      }
    }
    return Object.freeze({
      commandCount: entries.length,
      sourceCount: sources.size,
      optimization: expectedOptimization,
      debugInfo: releaseExecutable,
      functionSections: releaseExecutable,
      dataSections: releaseExecutable,
      debugPrefixMap: releaseExecutable,
    });
  };
  const releaseGroups = Object.freeze({
    inferenceHot: requireGroup('inferenceHot', '-O3', true),
    inferenceCold: requireGroup('inferenceCold', 'size', true),
    inferenceCli: requireGroup('inferenceCli', 'size', true),
    fullHot: requireGroup('fullHot', '-O3', true),
    fullCold: requireGroup('fullCold', 'size', true),
    fullCli: requireGroup('fullCli', 'size', true),
    armHot: grouped.releaseArmHot.length === 0
      ? Object.freeze({
        commandCount: 0,
        sourceCount: 0,
        optimization: '-O3',
        debugInfo: true,
        functionSections: true,
        dataSections: true,
        debugPrefixMap: true,
        status: 'not-applicable',
      })
      : Object.freeze({
        ...requireGroup('releaseArmHot', '-O3', true),
        status: 'available',
      }),
  });
  const libraryGroups = Object.freeze({
    inference: requireGroup('inferenceLibrary', '-O3', false),
    full: requireGroup('fullLibrary', '-O3', false),
  });
  for (const profile of ['inference', 'full']) {
    const hot = new Set(grouped[`${profile}Hot`].map(({ source }) => source));
    const overlap = grouped[`${profile}Cold`]
      .map(({ source }) => source)
      .filter((source) => hot.has(source));
    if (overlap.length > 0) {
      throw new Error(
        `Native ${profile} hot/cold compile groups overlap: ${overlap.join(', ')}.`,
      );
    }
  }
  return Object.freeze({
    status: 'available',
    path: relative,
    sha256: sha256(bytes),
    commandCount: commands.length,
    sourceCount: new Set(commands.map((entry) =>
      normalizedPath(path.resolve(entry.directory ?? repositoryRoot, entry.file ?? '')))).size,
    releaseGroups,
    libraryGroups,
  });
}

async function nativeLinkCommandsEvidence(repositoryRoot, buildDirectory) {
  const definitions = Object.freeze([
    ['inferenceExecutable', 'volvoxai.dir/link.txt', 'executable', 'volvoxai'],
    ['fullExecutable', 'volvoxai-full.dir/link.txt', 'full-executable', 'volvoxai-full'],
    ['inferenceShared', 'volvoxai_shared.dir/link.txt', 'shared', null],
    ['fullShared', 'volvoxai-full_shared.dir/link.txt', 'shared', null],
    ['inferenceStatic', 'volvoxai_static.dir/link.txt', 'static', null],
    ['fullStatic', 'volvoxai-full_static.dir/link.txt', 'static', null],
  ]);
  const results = {};
  const trainingControlSymbols = [
    'vx_training_control_core_abi_version',
    'vx_training_control_state_init_v1',
    'vx_training_control_state_validate_v1',
    'vx_training_control_step_begin_v1',
    'vx_training_control_step_finish_v1',
    'vx_training_control_prepare_commit_v1',
    'vx_training_control_prepare_rollback_v1',
    'vx_training_control_prepare_reset_accumulation_v1',
  ];
  for (const [name, relative, kind, releaseName] of definitions) {
    const absolute = path.join(buildDirectory, 'native', 'CMakeFiles', relative);
    let bytes;
    try {
      bytes = await fs.readFile(absolute);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      throw new Error(
        `Missing native link-command evidence '${normalizedPath(path.relative(
          repositoryRoot, absolute,
        ))}'; build executables and both library forms.`,
      );
    }
    const text = bytes.toString('utf8');
    const forbiddenPackaging = [
      '--strip-all', '-dead_strip', '--gc-sections',
    ].filter((flag) => text.includes(flag));
    const versionScriptCount = (text.match(/--version-script=/gu) ?? []).length;
    if (kind.includes('executable')) {
      const normalizedCommand = text.replaceAll('\\', '/');
      if (!new RegExp(
        `(?:^|\\s)-o\\s+(?:\\S*/)?release-link/${releaseName}(?:\\s|$)`,
        'u',
      ).test(normalizedCommand)) {
        throw new Error(
          `Native ${name} must link to its build-private release-link directory.`,
        );
      }
      for (const required of ['--gc-sections', '-Map,', '--build-id=sha1']) {
        if (!text.includes(required)) {
          throw new Error(`Native ${name} link command is missing ${required}.`);
        }
      }
      if (versionScriptCount !== 0 || /(?:^|\s)-rdynamic(?:\s|$)/u.test(text)) {
        throw new Error(`Native ${name} must not export an executable ABI.`);
      }
      const missingAnchors = trainingControlSymbols.filter((symbol) =>
        !text.includes(`-u,${symbol}`));
      if (kind === 'full-executable' && missingAnchors.length > 0) {
        throw new Error(
          `Native full executable link command is missing training-control anchors: ` +
          missingAnchors.join(', '),
        );
      }
      if (kind === 'executable' && missingAnchors.length !== trainingControlSymbols.length) {
        throw new Error('Native inference executable unexpectedly anchors training control.');
      }
    } else {
      if (forbiddenPackaging.length > 0 || /(?:^|\s)-s(?:\s|$)/u.test(text)) {
        throw new Error(
          `Native ${name} link command contains executable-only GC/strip flags: ` +
          forbiddenPackaging.join(', '),
        );
      }
      if (kind === 'shared' && versionScriptCount !== 1) {
        throw new Error(
          `Native ${name} must contain exactly one version script; found ` +
          `${versionScriptCount}.`,
        );
      }
      if (kind === 'static' && versionScriptCount !== 0) {
        throw new Error(`Native ${name} static archive unexpectedly uses a version script.`);
      }
    }
    results[name] = Object.freeze({
      path: normalizedPath(path.relative(repositoryRoot, absolute)),
      sha256: sha256(bytes),
      kind,
      versionScriptCount,
    });
  }
  return Object.freeze({ status: 'available', commands: Object.freeze(results) });
}

async function releaseBuildProvenance(repositoryRoot, nativeBuildDirectory, webOnly) {
  const wasmProfiles = RELEASE_PROFILES.wasm.map((profile) => {
    const recipe = stableValue(profile.recipe);
    return Object.freeze({
      id: profile.id,
      artifact: artifactIdForDist(profile.filename),
      capability: profile.capability,
      sourceEntries: Object.freeze([...profile.sourceEntries]),
      recipe: Object.freeze(recipe),
      recipeSha256: stableHash(recipe),
    });
  });
  const browserProfiles = RELEASE_PROFILES.browser.map((profile) => Object.freeze({
    id: profile.id,
    capability: profile.capability,
    entryPoint: profile.entryPoint,
    backends: Object.freeze([...profile.backends]),
    sidecar: profile.sidecar === null ? null : artifactIdForDist(profile.sidecar),
    readable: artifactIdForDist(profile.readable),
    minified: artifactIdForDist(profile.minified),
  }));
  let native = Object.freeze({ status: 'out-of-scope' });
  if (!webOnly) {
    const [cmakeCache, compileCommands, linkCommands] = await Promise.all([
      cmakeCacheEvidence(repositoryRoot, nativeBuildDirectory),
      compileCommandsEvidence(repositoryRoot, nativeBuildDirectory),
      nativeLinkCommandsEvidence(repositoryRoot, nativeBuildDirectory),
    ]);
    for (const [label, evidence] of [
      ['CMake cache', cmakeCache],
      ['compile commands', compileCommands],
    ]) {
      if (evidence.status !== 'available') {
        throw new Error(`Native ${label} evidence is unavailable: ${evidence.reason}`);
      }
    }
    native = Object.freeze({
      status: 'available',
      buildDirectory: normalizedPath(path.relative(repositoryRoot, nativeBuildDirectory)),
      profiles: Object.freeze(RELEASE_PROFILES.native.map((profile) => Object.freeze({
        id: profile.id,
        capability: profile.capability,
        artifact: profile.filename,
        debugArtifact: profile.debugArtifact,
        debugEvidence: profile.debugEvidence,
      }))),
      cmakeCache,
      compileCommands,
      linkCommands,
    });
  }
  return Object.freeze({
    authority: 'tools/release_profiles.mjs',
    browserProfiles: Object.freeze(browserProfiles),
    wasmProfiles: Object.freeze(wasmProfiles),
    native,
  });
}

function withoutBytes(artifact) {
  const { bytes: _bytes, ...rest } = artifact;
  return Object.freeze(rest);
}

/**
 * Inspect the fixed release inventory. This function never writes an artifact.
 * Browser analysis rebuilds minified entries with write=false solely to obtain
 * esbuild's contribution metafile and verifies those bytes against dist.
 */
export async function collectReleaseSizes({
  repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'),
  webOnly = false,
  generatedAt = new Date().toISOString(),
} = {}) {
  repositoryRoot = path.resolve(repositoryRoot);
  const packageJson = JSON.parse(await fs.readFile(
    path.join(repositoryRoot, 'package.json'),
    'utf8',
  ));
  const version = packageJson.version;
  const distDirectory = path.join('dist', version);

  const browserByFilename = new Map();
  for (const profile of RELEASE_PROFILES.browser) {
    browserByFilename.set(profile.readable, { profile, variant: 'readable' });
    browserByFilename.set(profile.minified, { profile, variant: 'minified' });
  }
  const wasmByFilename = new Map(RELEASE_PROFILES.wasm
    .map((profile) => [profile.filename, profile]));

  const distArtifacts = await Promise.all(DIST_RELEASE_FILENAMES.map(async (filename) => {
    const browser = browserByFilename.get(filename);
    const wasm = wasmByFilename.get(filename);
    if (browser !== undefined) {
      return basicArtifact(
        repositoryRoot,
        artifactIdForDist(filename),
        path.join(distDirectory, filename),
        'browser',
        browser.profile.id,
        browser.variant,
      );
    }
    if (wasm !== undefined) {
      const absoluteArtifactPath = path.join(repositoryRoot, distDirectory, filename);
      const artifact = await basicArtifact(
        repositoryRoot,
        artifactIdForDist(filename),
        path.join(distDirectory, filename),
        'wasm',
        wasm.id,
        'release',
      );
      const evidencePath = wasmReleaseEvidencePath(repositoryRoot, wasm);
      const snapshot = await validateWasmReleaseBuildSnapshot({
        repositoryRoot,
        artifact: absoluteArtifactPath,
        evidence: evidencePath,
        profile: wasm,
        packageVersion: version,
        // Keep host-side reporting independent of the build image. Fresh
        // provenance replay is responsible for rehashing tool executables and
        // Clang resource headers; this path still verifies repository inputs,
        // retained objects, and the recipe-bound recorded identities.
        verifyToolchainInputs: false,
      });
      const { recorded, objectGraph, componentBinding, validatedArtifact } = snapshot;
      const buildEvidence = recorded.document;
      if (snapshot.artifactSha256 !== artifact.sha256) {
        throw new Error(`${filename} changed while release-size evidence was collected.`);
      }
      const inspection = inspectWasmBytes(artifact.bytes);
      if (!Array.isArray(wasm.requiredExports) || wasm.requiredExportKinds === undefined) {
        throw new Error(`Release profile '${wasm.id}' has no generated ABI projection.`);
      }
      const exportOwnership = describeWasmExportOwnership(inspection, {
        authority: Object.freeze({
          kind: 'generated-abi-manifest',
          source: 'tools/generated/wasmInternalAbi.mjs',
          projection: 'tools/release_profiles.mjs#RELEASE_PROFILES.wasm',
          format: WASM_INTERNAL_ABI_FORMAT,
          manifestSha256: WASM_INTERNAL_ABI_MANIFEST_SHA256,
          profile: wasm.id,
          generated: true,
        }),
        exports: wasm.requiredExports,
        exportKinds: wasm.requiredExportKinds,
      });
      if (exportOwnership.missingRequired.length > 0) {
        throw new Error(
          `${filename} is missing generated private ABI exports: ` +
          exportOwnership.missingRequired.join(', '),
        );
      }
      artifact.wasm = Object.freeze({
        ...inspection,
        exportOwnership,
        provenance: validatedArtifact.provenance,
        buildEvidence: Object.freeze({
          path: normalizedPath(path.relative(repositoryRoot, evidencePath)),
          rawBytes: recorded.rawBytes,
          evidenceSha256: recorded.sha256,
          objectGraph,
          componentBinding,
          evidence: buildEvidence,
        }),
      });
      return artifact;
    }
    throw new Error(`Central release profiles do not classify '${filename}'.`);
  }));

  let llvmTools = null;
  let nativeArtifacts = [];
  const nativeBuildDirectory = path.resolve(
    repositoryRoot,
    process.env.CMAKE_BUILD_DIR ?? 'build/cmake',
  );
  if (!webOnly) {
    llvmTools = Object.freeze({
      readobj: await requireLlvmTool('LLVM_READOBJ', ['llvm-readobj-17', 'llvm-readobj']),
      nm: await requireLlvmTool('LLVM_NM', ['llvm-nm-17', 'llvm-nm']),
    });
    const nativeProfiles = new Map(RELEASE_PROFILES.native
      .map((profile) => [profile.filename, profile]));
    nativeArtifacts = await Promise.all(NATIVE_RELEASE_FILENAMES.map(async (filename) => {
      const profile = nativeProfiles.get(filename);
      if (profile === undefined) {
        throw new Error(`Central release profiles do not classify '${filename}'.`);
      }
      const artifact = await basicArtifact(
        repositoryRoot,
        filename,
        filename,
        'native',
        profile.id,
        'release',
      );
      const buildArtifactPath = path.join(
        nativeBuildDirectory,
        'native',
        'release-link',
        path.basename(filename),
      );
      const buildDebugPath = path.join(
        nativeBuildDirectory,
        'native',
        'release-link',
        '.debug',
        path.basename(profile.debugArtifact),
      );
      const buildEvidencePath = path.join(
        nativeBuildDirectory,
        'native',
        'release-link',
        '.debug',
        path.basename(profile.debugEvidence),
      );
      const [
        debugArtifact,
        evidenceArtifact,
        buildArtifact,
        buildDebugArtifact,
        buildEvidenceArtifact,
      ] = await Promise.all([
        provenanceArtifact(repositoryRoot, profile.debugArtifact, 'native debug sidecar'),
        provenanceArtifact(
          repositoryRoot,
          profile.debugEvidence,
          'native finalization evidence',
        ),
        provenanceArtifact(
          repositoryRoot,
          path.relative(repositoryRoot, buildArtifactPath),
          'build-private native release',
        ),
        provenanceArtifact(
          repositoryRoot,
          path.relative(repositoryRoot, buildDebugPath),
          'build-private native debug sidecar',
        ),
        provenanceArtifact(
          repositoryRoot,
          path.relative(repositoryRoot, buildEvidencePath),
          'build-private native finalization evidence',
        ),
      ]);
      let evidence;
      try {
        evidence = JSON.parse(evidenceArtifact.bytes.toString('utf8'));
      } catch (error) {
        throw new Error(
          `${profile.debugEvidence} is not valid native finalization evidence JSON.`,
          { cause: error },
        );
      }
      artifact.native = await inspectNativeElf({
        releaseArtifact: artifact,
        debugArtifact,
        evidenceArtifact,
        buildArtifact,
        buildDebugArtifact,
        buildEvidenceArtifact,
        evidence,
        profile,
        tools: llvmTools,
        repositoryRoot,
        buildDirectory: nativeBuildDirectory,
        mapPath: path.join(
          nativeBuildDirectory,
          'size-maps',
          `${path.basename(filename)}.map`,
        ),
        linkArtifactPath: buildArtifactPath,
        linkDirectory: path.join(nativeBuildDirectory, 'native'),
      });
      return artifact;
    }));
  }

  const allArtifacts = [...distArtifacts, ...nativeArtifacts];
  const artifactIndex = new Map(allArtifacts.map((artifact) => [artifact.id, artifact]));
  const browserProfiles = await Promise.all(RELEASE_PROFILES.browser.map(async (profile) => {
    const minifiedId = artifactIdForDist(profile.minified);
    const sidecarId = artifactIdForDist(profile.sidecar);
    const minified = artifactIndex.get(minifiedId);
    const sidecar = artifactIndex.get(sidecarId);
    if (minified === undefined || sidecar === undefined) {
      throw new Error(`Browser profile '${profile.id}' is missing its release pair.`);
    }
    const metafile = await inspectBrowserMinifiedBundle(
      repositoryRoot,
      profile,
      version,
      path.join(repositoryRoot, minified.path),
    );
    minified.browser = metafile;
    return Object.freeze({
      id: profile.id,
      capability: profile.capability,
      minifiedArtifact: minifiedId,
      sidecarArtifact: sidecarId,
      pair: Object.freeze({
        rawBytes: minified.rawBytes + sidecar.rawBytes,
        gzipBytes: minified.gzipBytes + sidecar.gzipBytes,
      }),
      metafile,
    });
  }));

  const [git, buildProvenance] = await Promise.all([
    gitEvidence(repositoryRoot),
    releaseBuildProvenance(repositoryRoot, nativeBuildDirectory, webOnly),
  ]);
  const toolchain = {
    node: process.version,
    esbuild: browserProfiles[0]?.metafile.esbuildVersion ?? 'unknown',
    gzip: (await commandText(process.env.GZIP ?? 'gzip', ['--version']))
      .split(/\r?\n/, 1)[0].trim(),
  };
  const wasmToolchains = distArtifacts
    .filter(({ kind }) => kind === 'wasm')
    .map(({ wasm }) => wasm.buildEvidence.evidence.toolchain);
  if (wasmToolchains.length > 0) {
    if (new Set(wasmToolchains.map(stableHash)).size !== 1) {
      throw new Error('Inference/full WASM build evidence used different toolchains.');
    }
    toolchain.wasmCompiler = Object.freeze({
      status: 'recipe-bound',
      ...wasmToolchains[0].compiler,
    });
    toolchain.wasmLinker = Object.freeze({
      status: 'recipe-bound',
      ...wasmToolchains[0].linker,
    });
  }
  if (!webOnly) {
    const selected = buildProvenance.native.cmakeCache.selected ?? {};
    toolchain.nativeCompiler = await toolVersionEvidence(
      selected.CMAKE_C_COMPILER ?? 'clang',
    );
    toolchain.nativeLinker = await toolVersionEvidence(
      selected.CMAKE_LINKER ?? 'ld.lld',
    );
  }
  if (llvmTools !== null) {
    toolchain.llvmReadobj = Object.freeze({
      path: normalizedPath(llvmTools.readobj),
      version: (await commandText(llvmTools.readobj, ['--version'])).split(/\r?\n/, 1)[0].trim(),
    });
    toolchain.llvmNm = Object.freeze({
      path: normalizedPath(llvmTools.nm),
      version: (await commandText(llvmTools.nm, ['--version'])).split(/\r?\n/, 1)[0].trim(),
    });
  }

  return Object.freeze({
    format: RELEASE_SIZE_REPORT_FORMAT,
    generatedAt,
    scope: webOnly ? 'web-only' : 'full',
    packageVersion: version,
    source: git,
    environment: Object.freeze({
      platform: process.platform,
      arch: process.arch,
      osRelease: os.release(),
      toolchain: Object.freeze(toolchain),
    }),
    buildProvenance,
    fixedArtifactCount: allArtifacts.length,
    artifacts: Object.freeze(allArtifacts.map(withoutBytes)),
    browserProfiles: Object.freeze(browserProfiles),
    extensionArchive: Object.freeze({
      applicable: false,
      status: 'deferred',
      milestone: 'M3',
      reason: 'No extension ZIP/CRX is part of the fixed release inventory yet.',
    }),
  });
}

function requireFiniteNonnegative(value, label) {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a finite non-negative number.`);
  }
  return value;
}

export function regressionAllowance(baseline, policy) {
  requireFiniteNonnegative(baseline, 'baseline');
  const relative = requireFiniteNonnegative(policy.relativeIncrease, 'relativeIncrease');
  const absolute = requireFiniteNonnegative(policy.absoluteIncreaseBytes, 'absoluteIncreaseBytes');
  return Math.max(Math.ceil(baseline * relative), Math.ceil(absolute));
}

function assertExactBudgetKeys(actual, expected, label) {
  const actualKeys = Object.keys(actual ?? {}).sort();
  const expectedKeys = [...expected].sort();
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    const expectedSet = new Set(expectedKeys);
    const actualSet = new Set(actualKeys);
    const missing = expectedKeys.filter((key) => !actualSet.has(key));
    const extra = actualKeys.filter((key) => !expectedSet.has(key));
    throw new Error(
      `${label} must exactly cover the fixed release inventory` +
      `${missing.length > 0 ? `; missing: ${missing.join(', ')}` : ''}` +
      `${extra.length > 0 ? `; unexpected: ${extra.join(', ')}` : ''}.`,
    );
  }
}

function assertBudgetMetricSchema(records, expectedMetrics, label, validateValue) {
  for (const [id, metrics] of Object.entries(records ?? {})) {
    if (metrics === null || typeof metrics !== 'object' || Array.isArray(metrics)) {
      throw new Error(`${label} '${id}' must be an object.`);
    }
    assertExactBudgetKeys(metrics, expectedMetrics, `${label} '${id}' metrics`);
    for (const [metric, value] of Object.entries(metrics)) {
      if (!validateValue(metric, value)) {
        throw new Error(`${label} '${id}' has invalid ${metric} value '${String(value)}'.`);
      }
    }
  }
}

/** Reject a baseline that silently leaves any fixed artifact/profile ungated. */
export function validateReleaseSizeBudgetInventory(budgets) {
  if (budgets?.format !== RELEASE_SIZE_BUDGET_FORMAT) {
    throw new Error(`Size budget format must be '${RELEASE_SIZE_BUDGET_FORMAT}'.`);
  }
  assertExactBudgetKeys(
    budgets?.baselines?.artifacts,
    FIXED_RELEASE_FILENAMES,
    'Size artifact baselines',
  );
  assertExactBudgetKeys(
    budgets?.baselines?.browserPairs,
    RELEASE_PROFILES.browser.map(({ id }) => id),
    'Browser-pair baselines',
  );
  assertExactBudgetKeys(
    budgets?.baselines?.wasmContracts,
    RELEASE_PROFILES.wasm.map(({ filename }) => artifactIdForDist(filename)),
    'WASM-contract baselines',
  );
  const byteMetric = (_metric, value) => Number.isSafeInteger(value) && value >= 0;
  assertBudgetMetricSchema(
    budgets.baselines.artifacts,
    ['rawBytes', 'gzipBytes'],
    'Size artifact baseline',
    byteMetric,
  );
  assertBudgetMetricSchema(
    budgets.baselines.browserPairs,
    ['rawBytes', 'gzipBytes'],
    'Browser-pair baseline',
    byteMetric,
  );
  assertBudgetMetricSchema(
    budgets.baselines.wasmContracts,
    ['importCount', 'importSetSha256', 'exportCount', 'exportSetSha256'],
    'WASM-contract baseline',
    (metric, value) => metric.endsWith('Count')
      ? Number.isSafeInteger(value) && value >= 0
      : typeof value === 'string' && /^[0-9a-f]{64}$/.test(value),
  );
}

function plannedTargetResults(report, budgets, artifactMap) {
  return Object.freeze(Object.entries(budgets.plannedTargets ?? {}).map(([id, target]) => {
    if (target.status === 'deferred' || target.artifact === undefined) {
      return Object.freeze({ id, ...target, enforced: false, status: 'deferred' });
    }
    const artifact = artifactMap.get(target.artifact);
    if (artifact === undefined) {
      return Object.freeze({
        id,
        ...target,
        enforced: false,
        status: report.scope === 'web-only' && target.artifact.startsWith('native/')
          ? 'out-of-scope'
          : 'unavailable',
      });
    }
    const metrics = Object.fromEntries(Object.entries(target.limits ?? {}).map(
      ([metric, maximumBytes]) => [metric, Object.freeze({
        actualBytes: artifact[metric],
        maximumBytes,
        meetsPlannedTarget: artifact[metric] <= maximumBytes,
      })],
    ));
    return Object.freeze({
      id,
      ...target,
      enforced: false,
      status: 'informational',
      metrics: Object.freeze(metrics),
    });
  }));
}

/** Compare only committed baselines. Planned targets are informational. */
export function evaluateSizeBudgets(report, budgets) {
  if (budgets?.format !== RELEASE_SIZE_BUDGET_FORMAT) {
    throw new Error(
      `Size budget format must be '${RELEASE_SIZE_BUDGET_FORMAT}'.`,
    );
  }
  const policy = budgets.policy ?? {};
  regressionAllowance(0, policy);
  const artifactMap = new Map(report.artifacts.map((artifact) => [artifact.id, artifact]));
  const pairMap = new Map(report.browserProfiles.map((profile) => [profile.id, profile.pair]));
  const checks = [];

  const inspectGroup = (group, baselines, actualMap) => {
    for (const [id, metrics] of Object.entries(baselines ?? {})) {
      const actual = actualMap.get(id);
      const outOfScope = report.scope === 'web-only' && id.startsWith('native/');
      if (actual === undefined && outOfScope) continue;
      for (const [metric, baseline] of Object.entries(metrics)) {
        if (actual === undefined || !Number.isSafeInteger(actual[metric])) {
          checks.push(Object.freeze({
            group, id, metric, baselineBytes: baseline,
            actualBytes: null, allowanceBytes: regressionAllowance(baseline, policy),
            limitBytes: baseline + regressionAllowance(baseline, policy),
            deltaBytes: null, passed: false, reason: 'missing actual metric',
          }));
          continue;
        }
        const allowance = regressionAllowance(baseline, policy);
        const limit = baseline + allowance;
        checks.push(Object.freeze({
          group,
          id,
          metric,
          baselineBytes: baseline,
          actualBytes: actual[metric],
          allowanceBytes: allowance,
          limitBytes: limit,
          deltaBytes: actual[metric] - baseline,
          passed: actual[metric] <= limit,
        }));
      }
    }
  };

  inspectGroup('artifact', budgets.baselines?.artifacts, artifactMap);
  inspectGroup('browser-pair', budgets.baselines?.browserPairs, pairMap);

  for (const [id, baseline] of Object.entries(budgets.baselines?.wasmContracts ?? {})) {
    const actual = artifactMap.get(id)?.wasm;
    for (const [metric, baselineValue] of Object.entries(baseline)) {
      const hasActual = actual !== undefined &&
        Object.prototype.hasOwnProperty.call(actual, metric);
      const actualValue = hasActual ? actual[metric] : null;
      checks.push(Object.freeze({
        group: 'wasm-contract',
        id,
        metric,
        comparison: 'exact',
        baselineValue,
        actualValue,
        passed: hasActual && actualValue === baselineValue,
        ...(hasActual ? {} : { reason: 'missing actual contract metric' }),
      }));
    }
  }

  const failures = checks.filter(({ passed }) => !passed);
  return Object.freeze({
    format: budgets.format,
    policy: Object.freeze({
      relativeIncrease: policy.relativeIncrease,
      absoluteIncreaseBytes: policy.absoluteIncreaseBytes,
      rule: 'baseline + max(relativeIncrease * baseline, absoluteIncreaseBytes)',
      wasmContractRule: 'exact linked import/export count and sorted-set SHA-256',
    }),
    passed: failures.length === 0,
    checks: Object.freeze(checks),
    failures: Object.freeze(failures),
    plannedTargets: plannedTargetResults(report, budgets, artifactMap),
  });
}

function bytes(value) {
  return value === null || value === undefined ? 'n/a' : MARKDOWN_NUMBER.format(value);
}

function markdownCell(value) {
  return String(value).replaceAll('|', '\\|').replaceAll('\n', '<br>');
}

export function formatBudgetFailure(failure) {
  if (failure.comparison === 'exact') {
    return `\`${failure.id}\` ${failure.metric}: ` +
      `\`${failure.actualValue ?? 'missing'}\` differs from exact baseline ` +
      `\`${failure.baselineValue}\`.`;
  }
  return `\`${failure.id}\` ${failure.metric}: ${bytes(failure.actualBytes)} B exceeds ` +
    `${bytes(failure.limitBytes)} B.`;
}

function checkIndex(evaluation, group) {
  return new Map(evaluation.checks
    .filter((check) => check.group === group)
    .map((check) => [`${check.id}\0${check.metric}`, check]));
}

export function renderReleaseSizeMarkdown(document) {
  const { budgetEvaluation: evaluation } = document;
  const artifactChecks = checkIndex(evaluation, 'artifact');
  const pairChecks = checkIndex(evaluation, 'browser-pair');
  const wasmContractChecks = checkIndex(evaluation, 'wasm-contract');
  const lines = [
    '# VolvoxAI release size report',
    '',
    `- Scope: \`${document.scope}\``,
    `- Package: \`${document.packageVersion}\``,
    `- Commit: \`${document.source.commit}${document.source.dirty ? '-dirty' : ''}\``,
    `- Generated: \`${document.generatedAt}\``,
    `- Regression gate: **${evaluation.passed ? 'PASS' : 'FAIL'}**`,
    `- Policy: baseline + max(${evaluation.policy.relativeIncrease * 100}%, ` +
      `${bytes(evaluation.policy.absoluteIncreaseBytes)} B)`,
    `- Linked WASM contract: ${evaluation.policy.wasmContractRule}`,
    '',
    '## Build provenance',
    '',
    `- Composition authority: \`${document.buildProvenance.authority}\``,
    `- Node: \`${document.environment.toolchain.node}\``,
    `- esbuild: \`${document.environment.toolchain.esbuild}\``,
    `- gzip: \`${markdownCell(document.environment.toolchain.gzip)}\``,
  ];
  for (const [label, key] of [
    ['WASM compiler', 'wasmCompiler'],
    ['WASM linker', 'wasmLinker'],
    ['Native compiler', 'nativeCompiler'],
    ['Native linker', 'nativeLinker'],
  ]) {
    const tool = document.environment.toolchain[key];
    if (tool === undefined) continue;
    if (tool.status === 'available') {
      lines.push(`- ${label}: \`${markdownCell(tool.version)}\` (\`${markdownCell(tool.path)}\`)`);
    } else if (tool.status === 'recipe-bound') {
      lines.push(
        `- ${label}: \`${markdownCell(tool.version)}\` (\`${markdownCell(tool.path)}\`) — ` +
        'recipe-bound evidence; not live-queried by this host report',
      );
    } else {
      lines.push(`- ${label}: unavailable (${markdownCell(tool.reason)})`);
    }
  }
  for (const profile of document.buildProvenance.wasmProfiles) {
    const objects = profile.recipe.parent.objects;
    const hotObjects = objects.filter(({ optimization }) => optimization === '-O3').length;
    const coldObjects = objects.filter(({ optimization }) => optimization === '-Oz').length;
    lines.push(
      `- WASM ${profile.id}: \`${profile.recipe.toolchain.compiler}\` + ` +
      `\`${profile.recipe.toolchain.linker}\`, ${objects.length} ordered parent objects ` +
      `(${hotObjects} O3, ${coldObjects} Oz), ` +
      `recipe \`${profile.recipeSha256}\``,
    );
  }
  if (document.buildProvenance.native.status === 'available') {
    const { cmakeCache, compileCommands, linkCommands } = document.buildProvenance.native;
    const groups = compileCommands.releaseGroups;
    lines.push(
      `- Native CMake cache: ${cmakeCache.status}` +
        `${cmakeCache.sha256 ? ` (\`${cmakeCache.sha256}\`)` : ''}`,
      `- Native compile commands: ${compileCommands.status}` +
        `${compileCommands.sha256 ? ` (${compileCommands.commandCount} commands, ` +
          `\`${compileCommands.sha256}\`)` : ''}`,
      `- Native inference release objects: ${groups.inferenceHot.commandCount} hot O3, ` +
        `${groups.inferenceCold.commandCount} cold size-opt, ` +
        `${groups.inferenceCli.commandCount} CLI size-opt`,
      `- Native full release objects: ${groups.fullHot.commandCount} hot O3, ` +
        `${groups.fullCold.commandCount} cold size-opt, ` +
        `${groups.fullCli.commandCount} CLI size-opt`,
      `- Native release-only Arm ISA objects: ${groups.armHot.commandCount} O3 ` +
        `(${groups.armHot.status})`,
      `- Native embedding-library objects: ` +
        `${compileCommands.libraryGroups.inference.commandCount} inference O3, ` +
        `${compileCommands.libraryGroups.full.commandCount} full O3`,
      `- Native link-command evidence: ${linkCommands.status} ` +
        `(${Object.keys(linkCommands.commands).length} executable/library commands)`,
    );
  } else {
    lines.push('- Native build evidence: out-of-scope');
  }
  lines.push(
    '',
    '## Fixed artifacts',
    '',
    '| Artifact | Raw B | Gzip B | Raw baseline/limit | Gzip baseline/limit | Gate |',
    '|---|---:|---:|---:|---:|:---:|',
  );
  for (const artifact of document.artifacts) {
    const raw = artifactChecks.get(`${artifact.id}\0rawBytes`);
    const gzip = artifactChecks.get(`${artifact.id}\0gzipBytes`);
    const passed = (raw?.passed ?? true) && (gzip?.passed ?? true);
    lines.push(
      `| \`${markdownCell(artifact.id)}\` | ${bytes(artifact.rawBytes)} | ` +
      `${bytes(artifact.gzipBytes)} | ${bytes(raw?.baselineBytes)}/${bytes(raw?.limitBytes)} | ` +
      `${bytes(gzip?.baselineBytes)}/${bytes(gzip?.limitBytes)} | ${passed ? 'PASS' : 'FAIL'} |`,
    );
  }

  lines.push(
    '',
    '## Browser minified pairs',
    '',
    '| Profile | Minified JS + sidecar | Raw B | Gzip B | Gate |',
    '|---|---|---:|---:|:---:|',
  );
  for (const profile of document.browserProfiles) {
    const raw = pairChecks.get(`${profile.id}\0rawBytes`);
    const gzip = pairChecks.get(`${profile.id}\0gzipBytes`);
    const passed = (raw?.passed ?? true) && (gzip?.passed ?? true);
    lines.push(
      `| ${markdownCell(profile.id)} | \`${profile.minifiedArtifact}\` + ` +
      `\`${profile.sidecarArtifact}\` | ${bytes(profile.pair.rawBytes)} | ` +
      `${bytes(profile.pair.gzipBytes)} | ${passed ? 'PASS' : 'FAIL'} |`,
    );
  }

  lines.push('', '## Browser metafile categories', '');
  for (const profile of document.browserProfiles) {
    lines.push(`### ${profile.id}`, '', '| Category | Bytes in minified output |', '|---|---:|');
    for (const [category, categoryBytes] of Object.entries(profile.metafile.categories)) {
      lines.push(`| ${markdownCell(category)} | ${bytes(categoryBytes)} |`);
    }
    lines.push('');
  }

  const wasmArtifacts = document.artifacts.filter(({ kind }) => kind === 'wasm');
  if (wasmArtifacts.length > 0) {
    lines.push('## WebAssembly composition', '');
    for (const artifact of wasmArtifacts) {
      const provenance = artifact.wasm.provenance;
      const objectGraph = artifact.wasm.buildEvidence.objectGraph;
      lines.push(
        `### ${artifact.id}`,
        '',
        `- ABI authority: \`${artifact.wasm.exportOwnership.authority.source}\` ` +
          `(\`${artifact.wasm.exportOwnership.authority.manifestSha256 ?? 'projection'}\`)`,
        `- Imports: ${artifact.wasm.importCount}`,
        `- Exports: ${artifact.wasm.exportCount} ` +
          `(required ${artifact.wasm.exportOwnership.manifestRequired.count}, ` +
          `unmanaged ${artifact.wasm.exportOwnership.unmanaged.count})`,
        `- Export-set SHA-256: \`${artifact.wasm.exportSetSha256}\``,
        `- Object build evidence: \`${artifact.wasm.buildEvidence.path}\` ` +
          `(${bytes(artifact.wasm.buildEvidence.rawBytes)} B, SHA-256 ` +
          `\`${artifact.wasm.buildEvidence.evidenceSha256}\`)`,
        `- Verified object graph: ${objectGraph.objectCount} objects, ` +
          `${objectGraph.dependencyCount} dependency records, SHA-256 ` +
          `\`${objectGraph.objectGraphSha256}\`` +
          (objectGraph.repositoryDependencyCount === undefined
            ? ''
            : ` (${objectGraph.repositoryDependencyCount} repository + ` +
              `${objectGraph.toolchainDependencyCount} toolchain; ` +
              (objectGraph.toolchainInputsVerified
                ? 'all inputs live-rehashed)'
                : 'toolchain inputs recipe-bound/recorded, not live-rehashed on this host)')),
        `- Source / recipe / contract: \`${provenance.sourceSha256}\` / ` +
          `\`${provenance.recipeSha256}\` / \`${provenance.contractSha256}\``,
        `- Payload / build evidence: \`${provenance.payloadSha256}\` / ` +
          `\`${provenance.buildEvidenceSha256}\``,
        '',
        '| Linked ABI metric | Actual | Exact baseline | Gate |',
        '|---|---|---|:---:|',
      );
      for (const metric of [
        'importCount',
        'importSetSha256',
        'exportCount',
        'exportSetSha256',
      ]) {
        const check = wasmContractChecks.get(`${artifact.id}\0${metric}`);
        lines.push(
          `| ${metric} | \`${markdownCell(artifact.wasm[metric])}\` | ` +
          `\`${markdownCell(check?.baselineValue ?? 'n/a')}\` | ` +
          `${check === undefined ? 'n/a' : check.passed ? 'PASS' : 'FAIL'} |`,
        );
      }
      lines.push(
        '',
        '| Section | Total B |',
        '|---|---:|',
      );
      for (const section of artifact.wasm.sections) {
        lines.push(`| ${markdownCell(section.name)} | ${bytes(section.totalBytes)} |`);
      }
      lines.push('', '| Custom index | Name | Content B | Content SHA-256 |',
        '|---:|---|---:|---|');
      for (const section of artifact.wasm.customSections) {
        lines.push(
          `| ${section.index} | \`${markdownCell(section.name)}\` | ` +
          `${bytes(section.contentBytes)} | \`${section.contentSha256}\` |`,
        );
      }
      lines.push('',
        '| Component/object | Category | Optimization | Raw B | SHA-256 | Flags SHA-256 |',
        '|---|---|:---:|---:|---|---|');
      for (const component of artifact.wasm.buildEvidence.evidence.components) {
        lines.push(
          `| **${markdownCell(component.kind)} linked** ` +
          `(${component.link.orderedObjectIds.map(markdownCell).join(' → ')}) | ` +
          `linked artifact | — | ${bytes(component.artifact.rawBytes)} | ` +
          `\`${component.artifact.sha256}\` | \`${component.link.flagsSha256}\` |`,
        );
        for (const object of component.orderedObjects) {
          lines.push(
            `| ${object.index}: \`${markdownCell(object.id)}\` ` +
            `(\`${markdownCell(object.source)}\`) | ${markdownCell(object.category)} | ` +
            `\`${object.optimization}\` | ${bytes(object.rawBytes)} | ` +
            `\`${object.sha256}\` | \`${object.flagsSha256}\` |`,
          );
        }
      }
      lines.push('');
    }
  }

  const nativeArtifacts = document.artifacts.filter(({ kind }) => kind === 'native');
  if (nativeArtifacts.length > 0) {
    lines.push('## Native ELF composition', '');
    for (const artifact of nativeArtifacts) {
      const {
        releaseElf,
        debugSidecar,
        finalizationEvidence,
      } = artifact.native;
      lines.push(
        `### ${artifact.id}`,
        '',
        `- Release ELF / section source: \`${markdownCell(releaseElf.path)}\` ` +
          `(${bytes(releaseElf.rawBytes)} B, SHA-256 \`${releaseElf.sha256}\`, ` +
          `build ID \`${releaseElf.buildId}\`)`,
        `- Debug sidecar / top-symbol source: \`${markdownCell(debugSidecar.path)}\` ` +
          `(${bytes(debugSidecar.rawBytes)} B, SHA-256 \`${debugSidecar.sha256}\`, ` +
          `build ID \`${debugSidecar.buildId}\`)`,
        `- Finalization evidence: \`${markdownCell(finalizationEvidence.path)}\` ` +
          `(${bytes(finalizationEvidence.rawBytes)} B, SHA-256 ` +
          `\`${finalizationEvidence.sha256}\`, format ` +
          `\`${markdownCell(finalizationEvidence.format)}\`)`,
        `- Finalization invariants: pre/post identical; GNU debuglink present; ` +
          `${finalizationEvidence.release.definedDynamicExports.length} defined dynamic exports`,
        '',
        '| Release section category | Bytes |',
        '|---|---:|',
      );
      for (const [name, sizeBytes] of Object.entries(artifact.native.sectionBytes)) {
        lines.push(`| ${markdownCell(name)} | ${bytes(sizeBytes)} |`);
      }
      const attribution = artifact.native.objectAttribution;
      if (attribution.status === 'available') {
        lines.push(
          '',
          `- Object attribution basis: ${attribution.attributionBasis}`,
          '',
          '| Object group | Attributed bytes |',
          '|---|---:|',
        );
        for (const [group, sizeBytes] of Object.entries(attribution.groups)) {
          lines.push(`| ${markdownCell(group)} | ${bytes(sizeBytes)} |`);
        }
        lines.push('', '| Top object | Group | Attributed bytes |', '|---|---|---:|');
        for (const object of attribution.topObjects) {
          lines.push(
            `| \`${markdownCell(object.object)}\` | ${markdownCell(object.group)} | ` +
            `${bytes(object.sizeBytes)} |`,
          );
        }
      } else {
        lines.push(
          '',
          `Object attribution: **unavailable** — ${markdownCell(attribution.reason)}`,
        );
      }
      lines.push(
        '',
        `Top symbols are read from \`${markdownCell(artifact.native.topSymbolSource)}\`.`,
        '',
        '| Top symbol | Type | Bytes |',
        '|---|:---:|---:|',
      );
      for (const symbol of artifact.native.topSymbols) {
        lines.push(
          `| \`${markdownCell(symbol.name)}\` | ${markdownCell(symbol.type)} | ` +
          `${bytes(symbol.sizeBytes)} |`,
        );
      }
      lines.push('');
    }
  }

  lines.push(
    '## Planned targets (informational only)',
    '',
    'These targets come from the refactoring plan. They do not participate in the regression gate.',
    '',
    '| Target | Status | Measurements |',
    '|---|---|---|',
  );
  for (const target of evaluation.plannedTargets) {
    const measurements = target.metrics === undefined
      ? target.reason ?? 'n/a'
      : Object.entries(target.metrics).map(([metric, value]) =>
        `${metric}: ${bytes(value.actualBytes)}/${bytes(value.maximumBytes)} B ` +
        `(${value.meetsPlannedTarget ? 'meets' : 'above'})`).join('<br>');
    lines.push(
      `| ${markdownCell(target.id)} | ${markdownCell(target.status)} | ` +
      `${markdownCell(measurements)} |`,
    );
  }

  lines.push(
    '',
    '## Extension archive',
    '',
    `Status: **${document.extensionArchive.status}** (${document.extensionArchive.reason})`,
  );
  if (!evaluation.passed) {
    lines.push('', '## Regression failures', '');
    for (const failure of evaluation.failures) {
      lines.push(`- ${formatBudgetFailure(failure)}`);
    }
  }
  return `${lines.join('\n')}\n`;
}

export async function loadReleaseSizeBudgets(budgetPath) {
  return JSON.parse(await fs.readFile(budgetPath, 'utf8'));
}

export async function createReleaseSizeDocument({
  repositoryRoot,
  budgetPath,
  webOnly = false,
  generatedAt,
} = {}) {
  const [report, budgets] = await Promise.all([
    collectReleaseSizes({ repositoryRoot, webOnly, generatedAt }),
    loadReleaseSizeBudgets(budgetPath),
  ]);
  validateReleaseSizeBudgetInventory(budgets);
  return Object.freeze({
    ...report,
    budgetEvaluation: evaluateSizeBudgets(report, budgets),
  });
}
