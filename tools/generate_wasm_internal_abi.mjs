#!/usr/bin/env node
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  GPU_BRIDGE_ABI,
  WASM_INTERNAL_ABI_FORMAT,
  WASM_INTERNAL_ABI_MANIFEST} from './wasm_internal_abi_manifest.mjs';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROFILE_NAMES = Object.freeze(['inference', 'full']);
const EXPECTED_PROFILE_COUNTS = Object.freeze({ inference: 143, full: 241 });
const ENTRY_KEYS = Object.freeze([
  'consumer', 'group', 'kind', 'name', 'owner', 'profiles', 'tsSignature',
]);
const VALID_KINDS = new Set(['function', 'global', 'memory']);
const VALID_TS_SIGNATURES = new Set([
  'global', 'gpuBridgeImport', 'i64Result', 'mathImport', 'memory', 'mixedI64',
  'numeric', 'wasmFunction',
]);
const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

function stableObject(value) {
  if (Array.isArray(value)) return value.map(stableObject);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort()
      .map((key) => [key, stableObject(value[key])]));
  }
  return value;
}

function stableJSON(value) {
  return JSON.stringify(stableObject(value));
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function fail(message) {
  throw new Error(`Invalid internal WASM ABI manifest: ${message}`);
}

function sameMembers(left, right) {
  return stableJSON([...left].sort()) === stableJSON([...right].sort());
}

function validateProfiles(profiles, label) {
  if (!Array.isArray(profiles) || profiles.length === 0 ||
      new Set(profiles).size !== profiles.length ||
      profiles.some((profile) => !PROFILE_NAMES.includes(profile))) {
    fail(`${label} has invalid profiles.`);
  }
}

function validateEntry(entry, label, { importEntry = false } = {}) {
  if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
    fail(`${label} must be an object.`);
  }
  const expectedKeys = importEntry ? [...ENTRY_KEYS, 'module'].sort() : ENTRY_KEYS;
  if (!sameMembers(Object.keys(entry), expectedKeys)) {
    fail(`${label} fields must be exactly ${expectedKeys.join(', ')}.`);
  }
  if (!NAME.test(entry.name) || (importEntry && !NAME.test(entry.module))) {
    fail(`${label} has an invalid module or symbol name.`);
  }
  for (const field of ['owner', 'consumer', 'group']) {
    if (typeof entry[field] !== 'string' || entry[field].length === 0) {
      fail(`${label}.${field} must be a non-empty string.`);
    }
  }
  validateProfiles(entry.profiles, label);
  if (!VALID_KINDS.has(entry.kind) || !VALID_TS_SIGNATURES.has(entry.tsSignature)) {
    fail(`${label} has an invalid kind or TypeScript signature.`);
  }
  if ((entry.kind === 'memory') !== (entry.tsSignature === 'memory') ||
      (entry.kind === 'global') !== (entry.tsSignature === 'global') ||
      (entry.kind === 'function') === ['memory', 'global'].includes(entry.tsSignature)) {
    fail(`${label} kind and TypeScript signature disagree.`);
  }
}

function profileExports(profile) {
  return WASM_INTERNAL_ABI_MANIFEST.parentExports
    .filter((entry) => entry.profiles.includes(profile))
    .sort((left, right) => compareText(left.name, right.name));
}

function validateManifest() {
  const manifest = WASM_INTERNAL_ABI_MANIFEST;
  if (manifest.format !== WASM_INTERNAL_ABI_FORMAT ||
      WASM_INTERNAL_ABI_FORMAT !== 'volvoxai-wasm-internal-abi/v1') {
    fail('format is unsupported.');
  }
  if (!sameMembers(manifest.profiles, PROFILE_NAMES)) fail('profile inventory differs.');
  /* The monotonic clock is unconditional. The GPU bridge is
   * full-profile only, so the inference parent keeps the smaller closure. */
  const gpuImports = manifest.parentImports.filter(
    (entry) => entry.group === 'gpu-bridge-import');
  const clockImports = manifest.parentImports.filter(
    (entry) => entry.group === 'clock-import');
  const entropyImports = manifest.parentImports.filter(
    (entry) => entry.group === 'entropy-import');
  const callImports = manifest.parentImports.filter(
    (entry) => entry.group === 'call-import');
  if (gpuImports.some((entry) => !sameMembers(entry.profiles, ['full']))) {
    fail('GPU imports belong only to the full profile.');
  }
  if (!Array.isArray(manifest.parentImports) ||
      clockImports.length !== 1 ||
      entropyImports.length !== 1 ||
      callImports.length !== 1 || callImports[0].module !== 'synurang' ||
      callImports[0].name !== 'wakeup' ||
      clockImports.length + entropyImports.length + callImports.length + gpuImports.length !== manifest.parentImports.length) {
    fail('the release parent may import only clock, entropy, call wakeup, and profile-selected GPU transport.');
  }
  const importKeys = new Set();
  manifest.parentImports.forEach((entry, index) => {
    validateEntry(entry, `parentImports[${index}]`, { importEntry: true });
    const key = `${entry.module}.${entry.name}:${entry.kind}`;
    if (importKeys.has(key)) fail(`duplicate parent import '${key}'.`);
    importKeys.add(key);
  });
  if (!Array.isArray(manifest.parentExports)) fail('parentExports must be an array.');
  const exportNames = new Set();
  manifest.parentExports.forEach((entry, index) => {
    validateEntry(entry, `parentExports[${index}]`);
    if (exportNames.has(entry.name)) fail(`duplicate parent export '${entry.name}'.`);
    exportNames.add(entry.name);
  });
  const byProfile = Object.fromEntries(PROFILE_NAMES.map((profile) => [
    profile, profileExports(profile),
  ]));
  for (const profile of PROFILE_NAMES) {
    if (byProfile[profile].length !== EXPECTED_PROFILE_COUNTS[profile]) {
      fail(`${profile} exports ${byProfile[profile].length} symbols; expected ` +
        `${EXPECTED_PROFILE_COUNTS[profile]}.`);
    }
  }
  const fullNames = new Set(byProfile.full.map(({ name }) => name));
  for (const { name } of byProfile.inference) {
    if (!fullNames.has(name)) fail(`inference export '${name}' is absent from full.`);
  }
  for (const entry of byProfile.inference) {
    if (['training', 'ptq', 'training-control', 'gpu-control'].includes(entry.group)) {
      fail(`inference contains full-only group '${entry.group}'.`);
    }
  }
  if (!Array.isArray(manifest.abiVersions)) fail('abiVersions must be an array.');
  const versionIds = new Set();
  for (const version of manifest.abiVersions) {
    if (version === null || typeof version !== 'object' || Array.isArray(version) ||
        !sameMembers(Object.keys(version), ['export', 'id', 'profiles', 'value']) ||
        typeof version.id !== 'string' || !NAME.test(version.export) ||
        !Number.isSafeInteger(version.value) || version.value <= 0) {
      fail('ABI version entry is malformed.');
    }
    validateProfiles(version.profiles, `ABI version '${version.id}'`);
    if (versionIds.has(version.id)) fail(`duplicate ABI version id '${version.id}'.`);
    versionIds.add(version.id);
    for (const profile of version.profiles) {
      const target = byProfile[profile].find(({ name }) => name === version.export);
      if (!target || target.kind !== 'function') {
        fail(`ABI version '${version.id}' export is absent from ${profile}.`);
      }
    }
  }
  for (const profile of PROFILE_NAMES) {
    const patterns = manifest.forbiddenProfilePatterns?.[profile];
    if (!Array.isArray(patterns) || patterns.some((pattern) => {
      try { new RegExp(pattern); return false; } catch { return true; }
    })) fail(`${profile} forbidden export patterns are invalid.`);
  }
  for (const [name, child] of Object.entries(manifest.children ?? {})) {
    if (child === null || typeof child !== 'object' || !NAME.test(name) ||
        typeof child.section !== 'string' || child.section.length === 0 ||
        !Array.isArray(child.imports) || !Array.isArray(child.exports)) {
      fail(`child '${name}' is malformed.`);
    }
    validateProfiles(child.profiles, `child '${name}'`);
    const childImports = new Set();
    child.imports.forEach((entry, index) => {
      validateEntry(entry, `children.${name}.imports[${index}]`, { importEntry: true });
      const key = `${entry.module}.${entry.name}:${entry.kind}`;
      if (childImports.has(key)) fail(`child '${name}' has duplicate import '${key}'.`);
      childImports.add(key);
    });
    const childExports = new Set();
    child.exports.forEach((entry, index) => {
      validateEntry(entry, `children.${name}.exports[${index}]`);
      if (childExports.has(entry.name)) {
        fail(`child '${name}' has duplicate export '${entry.name}'.`);
      }
      childExports.add(entry.name);
    });
  }
  return byProfile;
}

function quote(value) {
  return JSON.stringify(value);
}

function gpuBridgeHash() {
  return createHash('sha256').update(stableJSON(GPU_BRIDGE_ABI)).digest('hex');
}

function renderGpuBridgeC() {
  const abi = GPU_BRIDGE_ABI;
  return `/* DO NOT EDIT: generated by tools/generate_wasm_internal_abi.mjs. */
#ifndef VOLVOXAI_BACKENDS_GPU_BRIDGE_H
#define VOLVOXAI_BACKENDS_GPU_BRIDGE_H
#include <stdint.h>
#define VX_GPU_BRIDGE_ABI_VERSION ${abi.version}u
#define VX_GPU_BRIDGE_ABI_HASH "${gpuBridgeHash()}"
${Object.entries(abi.constants).map(([name, value]) => `#define VX_GPU_${name} (${value})`).join('\n')}
${Object.entries(abi.structs).map(([name, fields]) =>
    `typedef struct {\n${fields.map(field => `    uint32_t ${field};`).join('\n')}\n} ${name};`).join('\n')}
#if defined(__wasm__)
#define VX_GPU_IMPORT __attribute__((import_module("gpu")))
#else
#define VX_GPU_IMPORT
#endif
${Object.entries(abi.functions).map(([name, [result, ...args]]) =>
    `VX_GPU_IMPORT ${result} ${name}(${args.join(', ') || 'void'});`).join('\n')}
#endif
`;
}

function renderGpuBridgeTs() {
  const abi = GPU_BRIDGE_ABI;
  return `// DO NOT EDIT: generated by tools/generate_wasm_internal_abi.mjs.
export const GPU_BRIDGE_ABI_VERSION = ${abi.version};
export const GPU_BRIDGE_ABI_HASH = "${gpuBridgeHash()}";
export const GPU_LIMIT_NAMES = ${JSON.stringify(abi.structs.VxGpuLimits)} as const;
${Object.entries(abi.constants).map(([name, value]) => `export const GPU_${name} = ${value};`).join('\n')}
${Object.entries(abi.structs).map(([name, fields]) =>
    `export const ${name.replace('VxGpu', 'GPU_').toUpperCase()}_WORDS = ${fields.reduce((count, field) => count + Number(field.match(/\[(\d+)\]/)?.[1] ?? 1), 0)};`).join('\n')}
export interface WasmGpuBridge {
${Object.entries(abi.functions).map(([name, [result, ...args]]) =>
    `  ${name}(${args.map(arg => arg.split(' ').at(-1) + ': number').join(', ')}): ${result === 'void' ? 'void' : 'number'};`).join('\n')}
}
`;
}

function renderStringArray(name, values, { exported = true } = {}) {
  return `${exported ? 'export ' : ''}const ${name} = /* @__PURE__ */ Object.freeze([\n` +
    values.map((value) => `  ${quote(value)},`).join('\n') + '\n] as const);\n';
}

function tsType(entry) {
  return {
    global: 'WebAssembly.Global',
    i64Result: 'WasmI64ResultExport',
    mathImport: 'WasmMathImport',
    gpuBridgeImport: 'WasmGpuBridgeImport',
    memory: 'WebAssembly.Memory',
    mixedI64: 'WasmMixedI64Export',
    numeric: 'WasmNumericExport',
    wasmFunction: 'WasmGenericExport',
  }[entry.tsSignature];
}

function renderTsInterface(name, entries, extension) {
  return `export interface ${name} extends ${extension} {\n` +
    entries.map((entry) => `  readonly ${entry.name}: ${tsType(entry)};`).join('\n') +
    '\n}\n';
}

function runtimeGroup(group) {
  return WASM_INTERNAL_ABI_MANIFEST.parentExports
    .filter((entry) => entry.group === group)
    .sort((left, right) => compareText(left.name, right.name));
}

function renderTypeScript(byProfile, manifestSha256) {
  const inference = byProfile.inference;
  const inferenceNames = new Set(inference.map(({ name }) => name));
  const fullOnly = byProfile.full.filter(({ name }) => !inferenceNames.has(name));
  const modelControlNames = new Set([
    'memory', '__heap_base', 'alloc_bytes', 'heap_mark',
    ...WASM_INTERNAL_ABI_MANIFEST.parentExports
      .filter(({ owner }) => owner === 'synurang-call').map(({ name }) => name),
    'vx_wasm_inference_path_preflight_v1',
    'vx_wasm_mount_file_abort', 'vx_wasm_mount_file_begin',
    'vx_wasm_mount_file_finish', 'vx_wasm_mount_file_write',
    'vx_wasm_unmount_file',
  ]);
  const modelControl = inference.filter(({ name }) => modelControlNames.has(name));
  const child = WASM_INTERNAL_ABI_MANIFEST.children;
  return `// DO NOT EDIT: generated by tools/generate_wasm_internal_abi.mjs.\n` +
    `// Manifest SHA-256: ${manifestSha256}\n` +
    `export const WASM_INTERNAL_ABI_MANIFEST_SHA256 = ${quote(manifestSha256)};\n\n` +
    `export const RELAXED_SIMD_CHILD_SECTION = ${quote(child.relaxedSimd.section)};\n` +
    `export const PTQ_AUTHORING_CHILD_SECTION = ${quote(child.ptqAuthoring.section)};\n` +
    `export const PTQ_AUTHORING_CHILD_ABI_VERSION = ` +
      `${child.ptqAuthoring.abiVersion.value};\n\n` +
    `export const RELAXED_SIMD_CHILD_PROBE = /* @__PURE__ */ Object.freeze({\n` +
    `  export: ${quote(child.relaxedSimd.probe.export)},\n` +
    `  arity: ${child.relaxedSimd.probe.arity},\n` +
    `  zeroArgumentsResult: ${child.relaxedSimd.probe.zeroArgumentsResult},\n` +
    `});\n\n` +
    `export type WasmNumericExport = (...args: Array<number | undefined>) => number;\n` +
    `export type WasmI64ResultExport = (...args: Array<number | bigint>) => bigint;\n` +
    `export type WasmMixedI64Export = (...args: Array<number | bigint>) => number;\n` +
    `export type WasmGenericExport = ` +
      `(...args: Array<number | bigint>) => number | bigint | void;\n` +
    `export type WasmMathImport = (value: number, second?: number) => number;\n\n` +
    renderTsInterface('WasmInferenceParentExports', inference, 'WebAssembly.Exports') + '\n' +
    renderTsInterface('WasmFullParentExports', fullOnly, 'WasmInferenceParentExports') + '\n' +
    renderStringArray('WASM_PARENT_IMPORTS', WASM_INTERNAL_ABI_MANIFEST.parentImports
      .map(({ module, name, kind }) => `${module}.${name}:${kind}`).sort()) + '\n' +
    `export const WASM_PROFILE_IMPORTS = /* @__PURE__ */ Object.freeze({\n` +
    PROFILE_NAMES.map((profile) =>
      `  ${profile.includes('-') ? quote(profile) : profile}: ` +
      `/* @__PURE__ */ Object.freeze([\n` +
      WASM_INTERNAL_ABI_MANIFEST.parentImports
        .filter((entry) => entry.profiles.includes(profile))
        .map(({ module, name, kind }) => `${module}.${name}:${kind}`)
        .sort()
        .map((entry) => `    ${quote(entry)},`).join('\n') +
      `\n  ] as const),`).join('\n') +
    `\n} as const);\n\n` +
    `export const MODEL_CONTROL_WASM_REQUIRED_EXPORTS = /* @__PURE__ */ Object.freeze([\n` +
    modelControl.map(({ name, kind }) =>
      `  /* @__PURE__ */ Object.freeze({ ` +
        `name: ${quote(name)}, kind: ${quote(kind)} as const }),`).join('\n') +
    `\n] as const);\n\n` +
    renderStringArray('FULL_WASM_TRAINING_EXPORTS',
      runtimeGroup('training').map(({ name }) => name)) + '\n' +
    renderStringArray('FULL_WASM_PTQ_EXPORTS',
      runtimeGroup('ptq').map(({ name }) => name)) + '\n' +
    renderStringArray('FULL_WASM_TRAINING_CONTROL_EXPORTS',
      runtimeGroup('training-control').map(({ name }) => name)) + '\n' +
    renderStringArray('RELAXED_SIMD_CHILD_IMPORTS', child.relaxedSimd.imports
      .map(({ module, name, kind }) => `${module}.${name}:${kind}`).sort()) + '\n' +
    renderStringArray('RELAXED_SIMD_CHILD_EXPORTS', child.relaxedSimd.exports
      .map(({ name, kind }) => `${name}:${kind}`).sort()) + '\n' +
    renderStringArray('PTQ_AUTHORING_CHILD_IMPORTS', child.ptqAuthoring.imports
      .map(({ module, name, kind }) => `${module}.${name}:${kind}`).sort()) + '\n' +
    renderStringArray('PTQ_AUTHORING_CHILD_EXPORTS', child.ptqAuthoring.exports
      .map(({ name, kind }) => `${name}:${kind}`).sort());
}

function renderToolsModule(byProfile, manifestSha256) {
  const profileObject = Object.fromEntries(PROFILE_NAMES.map((profile) => [
    profile, byProfile[profile].map(({ name }) => name),
  ]));
  const kindObject = Object.fromEntries(PROFILE_NAMES.map((profile) => [
    profile, Object.fromEntries(byProfile[profile].map(({ name, kind }) => [name, kind])),
  ]));
  const importList = WASM_INTERNAL_ABI_MANIFEST.parentImports
    .map(({ module, name, kind }) => `${module}.${name}:${kind}`).sort();
  const abiVersions = Object.fromEntries(PROFILE_NAMES.map((profile) => [
    profile,
    Object.fromEntries(WASM_INTERNAL_ABI_MANIFEST.abiVersions
      .filter((entry) => entry.profiles.includes(profile))
      .map(({ id, export: exportName, value }) => [id, { export: exportName, value }])),
  ]));
  const children = Object.fromEntries(Object.entries(WASM_INTERNAL_ABI_MANIFEST.children)
    .map(([name, child]) => [name, {
      section: child.section,
      profiles: child.profiles,
      optional: child.optional,
      memoryMode: child.memoryMode,
      imports: child.imports.map((entry) => `${entry.module}.${entry.name}:${entry.kind}`).sort(),
      exports: child.exports.map((entry) => `${entry.name}:${entry.kind}`).sort(),
      ...(child.abiVersion ? { abiVersion: child.abiVersion } : {}),
      ...(child.probe ? { probe: child.probe } : {}),
    }]));
  /* Imports are profile-scoped for the same reason exports are: the ordinary
   * inference parent is WASM-only and never takes the GPU device bridge. */
  const profileImports = Object.fromEntries(PROFILE_NAMES.map((profile) => [
    profile,
    WASM_INTERNAL_ABI_MANIFEST.parentImports
      .filter((entry) => entry.profiles.includes(profile))
      .map(({ module, name, kind }) => `${module}.${name}:${kind}`)
      .sort(),
  ]));
  const generated = {
    format: WASM_INTERNAL_ABI_FORMAT,
    manifestSha256,
    parentImports: importList,
    profileImports,
    profileExports: profileObject,
    profileExportKinds: kindObject,
    profileAbiVersions: abiVersions,
    forbiddenProfilePatterns: WASM_INTERNAL_ABI_MANIFEST.forbiddenProfilePatterns,
    children,
  };
  return `// DO NOT EDIT: generated by tools/generate_wasm_internal_abi.mjs.\n` +
    `const ABI = ${JSON.stringify(generated, null, 2)};\n\n` +
    `function deepFreeze(value) {\n` +
    `  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {\n` +
    `    Object.freeze(value);\n` +
    `    for (const child of Object.values(value)) deepFreeze(child);\n` +
    `  }\n` +
    `  return value;\n` +
    `}\n\n` +
    `deepFreeze(ABI);\n\n` +
    `export const WASM_INTERNAL_ABI_FORMAT = ABI.format;\n` +
    `export const WASM_INTERNAL_ABI_MANIFEST_SHA256 = ABI.manifestSha256;\n` +
    `export const WASM_PARENT_IMPORTS = Object.freeze(ABI.parentImports);\n` +
    `export const WASM_PROFILE_IMPORTS = Object.freeze(ABI.profileImports);\n` +
    `export const WASM_PROFILE_EXPORTS = Object.freeze(ABI.profileExports);\n` +
    `export const WASM_PROFILE_EXPORT_KINDS = Object.freeze(ABI.profileExportKinds);\n` +
    `export const WASM_PROFILE_ABI_VERSIONS = Object.freeze(ABI.profileAbiVersions);\n` +
    `export const WASM_FORBIDDEN_PROFILE_PATTERNS = ` +
      `Object.freeze(ABI.forbiddenProfilePatterns);\n` +
    `export const WASM_CHILD_CONTRACTS = Object.freeze(ABI.children);\n`;
}

function renderResponse(entries) {
  const flags = entries.map(({ name, kind }) => kind === 'memory'
    ? '-Wl,--export-memory'
    : `-Wl,--export=${name}`);
  return `${[...new Set(flags)].sort((left, right) => {
    if (left === '-Wl,--export-memory') return -1;
    if (right === '-Wl,--export-memory') return 1;
    return compareText(left, right);
  }).join('\n')}\n`;
}

async function writeAtomic(output, contents) {
  await fs.mkdir(path.dirname(output), { recursive: true });
  const temporary = path.join(
    path.dirname(output), `.${path.basename(output)}.${process.pid}.tmp`,
  );
  try {
    await fs.writeFile(temporary, contents);
    await fs.rename(temporary, output);
  } finally {
    await fs.rm(temporary, { force: true });
  }
}

function parseArguments(arguments_) {
  let check = false;
  let outputRoot = repositoryRoot;
  for (let index = 0; index < arguments_.length; index++) {
    const argument = arguments_[index];
    if (argument === '--check') {
      check = true;
    } else if (argument === '--output-root' && index + 1 < arguments_.length) {
      outputRoot = path.resolve(arguments_[++index]);
    } else {
      throw new Error(`Unknown argument '${argument}'.`);
    }
  }
  return { check, outputRoot };
}

async function main() {
  const { check, outputRoot } = parseArguments(process.argv.slice(2));
  const byProfile = validateManifest();
  const canonical = stableJSON(WASM_INTERNAL_ABI_MANIFEST);
  const manifestSha256 = createHash('sha256').update(canonical).digest('hex');
  const outputs = new Map([
    ['native/src/backends/gpu_bridge.h', renderGpuBridgeC()],
    ['ts/generated/gpuBridge.ts', renderGpuBridgeTs()],
    ['ts/generated/wasmInternalAbi.ts', renderTypeScript(byProfile, manifestSha256)],
    ['tools/generated/wasmInternalAbi.mjs', renderToolsModule(byProfile, manifestSha256)],
    ['runtime/generated/wasm/inference.exports.rsp', renderResponse(byProfile.inference)],
    ['runtime/generated/wasm/full.exports.rsp', renderResponse(byProfile.full)],
  ]);
  if (check) {
    const stale = [];
    for (const [relative, expected] of outputs) {
      const output = path.join(outputRoot, relative);
      try {
        if (await fs.readFile(output, 'utf8') !== expected) stale.push(relative);
      } catch (error) {
        if (error?.code !== 'ENOENT') throw error;
        stale.push(relative);
      }
    }
    if (stale.length > 0) {
      throw new Error(
        `Stale generated internal WASM ABI projections: ${stale.join(', ')}. ` +
        'Run node tools/generate_wasm_internal_abi.mjs.',
      );
    }
    console.log(
      `Verified generated internal WASM ABI projections (${EXPECTED_PROFILE_COUNTS.inference}/${EXPECTED_PROFILE_COUNTS.full} exports).`,
    );
    return;
  }
  for (const [relative, contents] of outputs) {
    await writeAtomic(path.join(outputRoot, relative), contents);
  }
  console.log(
    `Generated internal WASM ABI projections (${EXPECTED_PROFILE_COUNTS.inference}/${EXPECTED_PROFILE_COUNTS.full} exports).`,
  );
}

await main();
