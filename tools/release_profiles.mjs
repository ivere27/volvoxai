import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { wasmToolImports } from './wasm_host_imports.mjs';

import {
  WASM_CHILD_CONTRACTS,
  WASM_FORBIDDEN_PROFILE_PATTERNS,
  WASM_PARENT_IMPORTS,
  WASM_PROFILE_IMPORTS,
  WASM_PROFILE_ABI_VERSIONS,
  WASM_PROFILE_EXPORT_KINDS,
  WASM_PROFILE_EXPORTS} from './generated/wasmInternalAbi.mjs';

const executeFile = promisify(execFile);

export const RELEASE_PROFILE_FORMAT = 'volvoxai-release-profiles/v1';
export const WASM_PROVENANCE_FORMAT = 'volvoxai-wasm-provenance/v1';
export const WASM_BUILD_EVIDENCE_FORMAT = 'volvoxai-wasm-build-evidence/v1';
export const WASM_PROVENANCE_SECTION = 'volvoxai.release.provenance.v1';
export const WASM_RELAXED_SIMD_SECTION = WASM_CHILD_CONTRACTS.relaxedSimd.section;
export const WASM_PTQ_AUTHORING_SECTION = WASM_CHILD_CONTRACTS.ptqAuthoring.section;
export const BROWSER_PROVENANCE_FORMAT = 'volvoxai-browser-provenance/v1';
export const BROWSER_PROVENANCE_MARKER = 'volvoxai-browser-provenance-v1';

const ALLOWED_WASM_IMPORTS = WASM_PARENT_IMPORTS;
const PTQ_AUTHORING_CHILD_IMPORTS = WASM_CHILD_CONTRACTS.ptqAuthoring.imports;
const PTQ_AUTHORING_CHILD_EXPORTS = WASM_CHILD_CONTRACTS.ptqAuthoring.exports;

function abiVersionValues(profile) {
  return Object.freeze(Object.fromEntries(Object.entries(profile)
    .map(([id, contract]) => [id, contract.value])));
}

const WASM_INFERENCE_SOURCES = Object.freeze([
  'native/src/kernels/kernels.c',
  'native/src/runtime/portable_control_wasm.c',
  'native/src/runtime/scheduler_policy.c',
]);
const WASM_LIBM_SOURCES = Object.freeze([
  'native/src/runtime/wasm_math.c',
]);
const WASM_FULL_SOURCES = Object.freeze([
  ...WASM_INFERENCE_SOURCES,
  'native/src/kernels/training_kernels.c',
  'native/src/training/quantization.c',
  'native/src/training/trainer_api.c',
  'native/src/training/ptq_api.c',
  'native/src/training/quantization_runtime.c',
  'native/src/training/quantization_package.c',
  'native/src/training/ptq_names.c',
  'native/src/training/ptq_authoring.c',
  'native/src/training/training_control.c',
]);
const WASM_RELAXED_SOURCES = Object.freeze([
  'native/src/kernels/qlinear_w8a8_wasm_relaxed.c',
]);
const WASM_PTQ_AUTHORING_SOURCES = Object.freeze([
  'native/src/training/ptq_authoring_wasm.c',
  'native/src/training/ptq_authoring.c',
  'native/src/training/ptq_names.c',
  'native/src/runtime/wasm_libc.c',
  'native/third_party/cJSON.c',
  'native/src/runtime/safetensors_bytes.c',
  'native/src/runtime/safetensors_header.c',
  'native/src/generated/operator_vocabulary.c',
]);
const WASM_FREESTANDING_SOURCE_ENTRIES = Object.freeze([
  // These are selected through -I and angle includes, so the quoted-include
  // closure cannot discover them. They affect every parent and the PTQ child.
  'native/src/runtime/wasm_freestanding/include/ctype.h',
  'native/src/runtime/wasm_freestanding/include/math.h',
  'native/src/runtime/wasm_freestanding/include/stdio.h',
  'native/src/runtime/wasm_freestanding/include/stdlib.h',
  'native/src/runtime/wasm_freestanding/include/string.h',
]);
const WASM_PTQ_AUTHORING_SOURCE_ENTRIES = Object.freeze([
  ...WASM_PTQ_AUTHORING_SOURCES,
  ...WASM_FREESTANDING_SOURCE_ENTRIES,
]);
function pinnedToolRuntime(path, rawBytes, sha256) {
  return Object.freeze({ path, rawBytes, sha256 });
}

// Canonical realpaths from the Dockerfile's pinned Ubuntu/LLVM loader closure. The
// virtual linux-vdso has no file to pin; every file-backed loader/library is
// included. Clang adds libclang-cpp to the common LLVM/LLD closure below.
const WASM_LLVM_RUNTIME_DEPENDENCIES = Object.freeze([
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2',
    240936,
    '41daf7836f7fce21a40ca63a00c5caf21b05a35a76ccc075d86f1716212cdf01',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libbsd.so.0.11.5',
    89096,
    'e4ad21c203d8e8b9b7f49071311339af6b438026e1ad236403e3b379b022f23d',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libc.so.6',
    2220400,
    'b2cf6c33b74d2f22543b7a469a75b538911e690f769d0b238843a49465b83793',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libedit.so.2.0.68',
    216376,
    '81dbdf431b1d828bad9e2511b562310bd3f0bdb1bad998775406d2784426fd66',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libffi.so.8.1.0',
    47688,
    '247da4d5d34a91cadcdd6282be4c4644fcb8af001334d2b8a82ecda435418cbf',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1',
    125488,
    'fc9d43b2f6c20e53b009238f767c5b949d202389e20de9e202ea684b4ba3729a',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libicudata.so.70.1',
    29476472,
    'c1404396288e178c8db2f30203cb8150e4d2c14cacee9a2883e0092f45399cc8',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libicuuc.so.70.1',
    2062664,
    '4683f3623dc47a92e862dc3c8770caff81c9f4c7e116bb672fc439737b06d88a',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libLLVM-17.so.1',
    123909048,
    'ea4b74072fdd895a63c3356e034f21f22e2fe7858545f2c5aaa22f05683f4657',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/liblzma.so.5.2.5',
    170456,
    '7fa51c1500cb9fcc4c7d69b7fe0a6d7203cfe6af64373f6c411c5445ef1b402c',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libm.so.6',
    940560,
    '3dd5511ae94785c9f921429b0f2b2f7aabb461b6f0e6de6dfdbef15f24bdfee6',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libmd.so.0.0.5',
    47472,
    'bb376813876bb214535bbbdac70f2c0c4b00baecb81557dab78a79518248df06',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30',
    2260296,
    'ff0825e113603c3866680d5d52216bc6d8eedf3a59f52a0aef67ff01994db128',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libtinfo.so.6.3',
    200136,
    '856b8451914abc2277c3d91c64ba3e72b2e0d519806397afecae92fda1a6b10c',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libxml2.so.2.9.13',
    1967384,
    '5295d822fae471706cab38d47c986c85d7e3b69efc56b2d5f7d4f2b5b5b958f7',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libz.so.1.2.11',
    108936,
    '64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370',
  ),
  pinnedToolRuntime(
    '/usr/lib/x86_64-linux-gnu/libzstd.so.1.4.8',
    841808,
    '5df4f4df42d76270bb6981fabc7c1fdccd8ad28a23d84d67f73203fb3f537667',
  ),
]);
const WASM_CLANG_RUNTIME_DEPENDENCIES = Object.freeze([
  pinnedToolRuntime(
    '/usr/lib/llvm-17/lib/libclang-cpp.so.17',
    62775536,
    '32d430ebff126fa3e5b65021e33516d5d23afd8809f05faa6b9bd030e0298cda',
  ),
  ...WASM_LLVM_RUNTIME_DEPENDENCIES,
]);
const WASM_TOOLCHAIN = Object.freeze({
  compiler: 'clang-17',
  compilerInvocationPath: '/usr/bin/clang-17',
  compilerRealPath: '/usr/lib/llvm-17/bin/clang',
  compilerVersion:
    'Ubuntu clang version 17.0.6 ' +
    '(++20231209124227+6009708b4367-1~exp1~20231209124336.77)',
  compilerSha256: '6f673bb8b459659cdefed06f2348b3ad47a13fc84e5736df38a83ffc62532e70',
  compilerRuntimeDependencies: WASM_CLANG_RUNTIME_DEPENDENCIES,
  resourceDependencyRoot: '/usr/lib/llvm-17/lib/clang/17/include',
  linker: 'wasm-ld-17',
  linkerInvocationPath: '/usr/bin/wasm-ld-17',
  linkerRealPath: '/usr/lib/llvm-17/bin/lld',
  linkerVersion: 'Ubuntu LLD 17.0.6',
  linkerSha256: 'aa716ebc4baaeb6a1bcf6cf0f875f59ad93d49590757a1c1c8a5acfbffa41702',
  linkerRuntimeDependencies: WASM_LLVM_RUNTIME_DEPENDENCIES,
});
const WASM_PARENT_COMPILE_FLAGS = Object.freeze([
  '--target=wasm32',
  '-msimd128',
  '-nostdlib',
  '-Inative/src/runtime/wasm_freestanding/include',
  '-Inative/include',
  '-Inative/src',
  '-Inative/src/runtime',
  '-Inative/src/kernels',
  '-Inative/src/backends',
  '-Inative/third_party',
  '-Inative/third_party/synurang/include',
  '-DVOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC=1',
  '-DSYNURANG_RUNTIME_NO_THREADS=1',
]);
const WASM_FULL_INCLUDE_FLAGS = Object.freeze([
  '-Inative/third_party/xz-embedded',
  '-Inative/src/training',
  '-DVOLVOXAI_ENABLE_TRAINING=1',
]);
const WASM_PARENT_BASE_LINK_FLAGS = Object.freeze([
  '-m',
  'wasm32',
  '-L/usr/lib',
  '--no-entry',
  '--allow-undefined',
  '--strip-all',
]);

/**
 * Convert WASM_PROFILE_EXPORTS entries into direct wasm-ld flags.
 * 'memory' becomes '--export-memory'; all others become '--export=<name>'.
 */
function wasmParentLinkFlags(profileExports) {
  const exportFlags = profileExports.map((name) =>
    name === 'memory' ? '--export-memory' : `--export=${name}`);
  return Object.freeze([...WASM_PARENT_BASE_LINK_FLAGS, ...exportFlags]);
}
const WASM_RELAXED_COMPILE_FLAGS = Object.freeze([
  '--target=wasm32',
  '-std=c11',
  '-Wall',
  '-Wextra',
  '-Werror',
  '-msimd128',
  '-mrelaxed-simd',
  '-nostdlib',
  '-ffreestanding',
]);
const WASM_RELAXED_LINK_FLAGS = Object.freeze([
  '-m',
  'wasm32',
  '-L/usr/lib',
  '--no-entry',
  '--import-memory',
  '--strip-all',
]);
const WASM_PTQ_AUTHORING_COMPILE_FLAGS = Object.freeze([
  '--target=wasm32',
  '-std=c11',
  '-nostdlib',
  '-ffreestanding',
  '-DVOLVOXAI_WASM_FREESTANDING_EXTENDED_LIBC=1',
  '-DVOLVOXAI_PTQ_AUTHORING_NO_FILES=1',
  '-DVOLVOXAI_PTQ_HOST_FORMAT=1',
  '-Inative/src/runtime/wasm_freestanding/include',
  '-Inative/include',
  '-Inative/src',
  '-Inative/src/training',
  '-Inative/src/runtime',
  '-Inative/third_party',
]);
const WASM_PTQ_AUTHORING_LINK_FLAGS = Object.freeze([
  '-m',
  'wasm32',
  '-L/usr/lib',
  '--no-entry',
  '--export-memory',
  '--allow-undefined',
  '--strip-all',
]);

function wasmObject(id, source, category, optimization, flags) {
  return Object.freeze({
    id,
    source,
    category,
    optimization,
    flags: Object.freeze([...flags]),
  });
}

function parentObjects(full) {
  const flags = [
    ...WASM_PARENT_COMPILE_FLAGS,
    `-DVOLVOXAI_ENABLE_WEBGPU=${full ? 1 : 0}`,
    full ? '-Iruntime/generated/c' : '-Iruntime/generated/c/inference',
    ...(full ? WASM_FULL_INCLUDE_FLAGS : []),
  ];
  return Object.freeze([
    wasmObject('kernels', WASM_INFERENCE_SOURCES[0], 'numerical-kernel', '-O3', flags),
    wasmObject(
      'portable-control',
      WASM_INFERENCE_SOURCES[1],
      'portable-control',
      '-Oz',
      flags,
    ),
    wasmObject(
      'scheduler-policy',
      WASM_INFERENCE_SOURCES[2],
      'portable-control',
      '-Oz',
      flags,
    ),
    ...['call', 'wasm'].map((name) => wasmObject(
      `synurang-${name}`, `native/third_party/synurang/src/${name}.c`,
      'portable-control', '-Oz', flags)),
    ...(full ? [
      wasmObject(
        'training-kernels',
        WASM_FULL_SOURCES[3],
        'numerical-kernel',
        '-O3',
        flags,
      ),
      wasmObject(
        'quantization',
        WASM_FULL_SOURCES[4],
        'numerical-quantization',
        '-O3',
        flags,
      ),
      ...WASM_FULL_SOURCES.slice(5).map((source) => wasmObject(
        'service-' + path.basename(source, '.c').replaceAll('_', '-'), source,
        'training-control', '-Oz', flags)),
    ] : []),
    ...WASM_LIBM_SOURCES.map((source) => wasmObject(
      `libm-${path.basename(source, '.c')}`, source, 'numerical-kernel', '-O3',
      [...flags, '-ffp-contract=off'])),
  ]);
}

function ptqAuthoringObjects() {
  const ids = [
    'ptq-authoring-wasm',
    'ptq-authoring',
    'ptq-names',
    'wasm-libc',
    'cjson',
    'safetensors-bytes',
    'safetensors-header',
    'operator-vocabulary',
  ];
  return Object.freeze(WASM_PTQ_AUTHORING_SOURCES.map((source, index) =>
    wasmObject(
      ids[index],
      source,
      'ptq-authoring',
      '-Oz',
      WASM_PTQ_AUTHORING_COMPILE_FLAGS,
    )));
}

function wasmRecipe({ full = false } = {}) {
  const profileExports = full
    ? WASM_PROFILE_EXPORTS.full
    : WASM_PROFILE_EXPORTS.inference;
  return Object.freeze({
    toolchain: WASM_TOOLCHAIN,
    parent: Object.freeze({ objects: parentObjects(full), linkFlags: wasmParentLinkFlags(profileExports) }),
    relaxed: null,
    ptqAuthoring: null,
    customSections: Object.freeze([WASM_PROVENANCE_SECTION]),
  });
}

/**
 * The one release-composition authority. Keep this data-only: build and gate
 * tools import it, while package.json and Makefile are checked projections.
 */
export const RELEASE_PROFILES = Object.freeze({
  format: RELEASE_PROFILE_FORMAT,
  browser: Object.freeze([
    Object.freeze({
      id: 'inference',
      capability: 'inference',
      backends: Object.freeze(['wasm']),
      entryPoint: 'ts/index.ts',
      readable: 'volvoxai.lite.js',
      minified: 'volvoxai.lite.min.js',
      sidecar: 'volvoxai.lite.wasm',
      browserOnly: false,
      packageExports: Object.freeze({ readable: './lite', minified: './lite/min' }),
      publicTraining: false,
      publicPTQ: false,
      publicPtqAuthoring: false,
    }),
    Object.freeze({
      id: 'full',
      capability: 'full',
      backends: Object.freeze(['wasm', 'webgpu']),
      entryPoint: 'ts/full.ts',
      readable: 'volvoxai.js',
      minified: 'volvoxai.min.js',
      sidecar: 'volvoxai.wasm',
      browserOnly: false,
      packageExports: Object.freeze({ readable: '.', minified: './min' }),
      publicTraining: true,
      publicPTQ: true,
      publicPtqAuthoring: true,
    }),
  ]),
  wasm: Object.freeze([
    Object.freeze({
      id: 'inference',
      capability: 'inference',
      publicPtqAuthoring: false,
      filename: 'volvoxai.lite.wasm',
      consumers: Object.freeze(['inference']),
      sourceEntries: Object.freeze([
        ...parentObjects(false).map(({ source }) => source),
        ...WASM_FREESTANDING_SOURCE_ENTRIES,
      ]),
      forbiddenDependencyPrefixes: Object.freeze([
        'native/src/training/',
      ]),
      forbiddenDependencyPatterns: Object.freeze([
        // Cover training/PTQ headers and backend fragments as well as .c
        // roots. Delimiters prevent inference words such as "pretraining"
        // from becoming accidental matches.
        '(?:^|[/_.-])training(?:$|[/_.-])',
        '(?:^|[/_.-])ptq(?:$|[/_.-])',
      ]),
      forbiddenDependencyFiles: Object.freeze([
        'native/src/backends/webgpu_backend.c',
        'native/src/backends/gpu_bridge.h',
        'native/src/backends/shader_catalog_profile.h',
        'native/src/backends/shader_catalog_inference.h',
        'native/src/backends/shader_catalog.h',
        'native/src/kernels/training_kernels.c',
        'native/src/kernels/training_kernels.h',
        'native/src/api/vx_api_quantization.c',
        'runtime/generated/c/volvoxai_ffi.c',
        'runtime/generated/c/volvoxai_ffi.h',
        'runtime/generated/c/volvoxai_lite.c',
        'runtime/generated/c/volvoxai_lite.h',
      ]),
      capabilityExportSources: Object.freeze([]),
      requiredExports: WASM_PROFILE_EXPORTS.inference,
      requiredExportKinds: WASM_PROFILE_EXPORT_KINDS.inference,
      forbiddenExportPatterns: WASM_FORBIDDEN_PROFILE_PATTERNS.inference,
      abiVersionExports: WASM_PROFILE_ABI_VERSIONS.inference,
      abiVersions: abiVersionValues(WASM_PROFILE_ABI_VERSIONS.inference),
      recipe: wasmRecipe(),
    }),
    Object.freeze({
      id: 'full',
      capability: 'full',
      publicPtqAuthoring: true,
      filename: 'volvoxai.wasm',
      consumers: Object.freeze(['full']),
      sourceEntries: Object.freeze([
        ...parentObjects(true).map(({ source }) => source),
        ...WASM_FREESTANDING_SOURCE_ENTRIES,
      ]),
      forbiddenDependencyPrefixes: Object.freeze([]),
      forbiddenDependencyPatterns: Object.freeze([]),
      forbiddenDependencyFiles: Object.freeze([]),
      capabilityExportSources: Object.freeze([
        'native/src/kernels/training_kernels.c',
        'native/src/training/quantization.c',
            ]),
      requiredExports: WASM_PROFILE_EXPORTS.full,
      requiredExportKinds: WASM_PROFILE_EXPORT_KINDS.full,
      forbiddenExportPatterns: WASM_FORBIDDEN_PROFILE_PATTERNS.full,
      abiVersionExports: WASM_PROFILE_ABI_VERSIONS.full,
      abiVersions: abiVersionValues(WASM_PROFILE_ABI_VERSIONS.full),
      recipe: wasmRecipe({ full: true }),
    }),
  ]),
  native: Object.freeze([
    Object.freeze({
      id: 'inference',
      capability: 'inference',
      filename: 'native/volvoxai-lite',
      debugArtifact: 'native/.debug/volvoxai-lite.debug',
      debugEvidence: 'native/.debug/volvoxai-lite.debug.json',
      publicTraining: false,
      publicPTQ: false,
      publicPtqAuthoring: false,
    }),
    Object.freeze({
      id: 'full',
      capability: 'full',
      filename: 'native/volvoxai',
      debugArtifact: 'native/.debug/volvoxai.debug',
      debugEvidence: 'native/.debug/volvoxai.debug.json',
      publicTraining: true,
      publicPTQ: true,
      publicPtqAuthoring: true,
    }),
  ]),
  nativeBoundaryCheck: 'native/cmake/check_profile_boundaries.sh',
});

export const BROWSER_RELEASE_FILENAMES = Object.freeze(RELEASE_PROFILES.browser
  .flatMap((profile) => [profile.readable, profile.minified]));
export const WASM_RELEASE_FILENAMES = Object.freeze(RELEASE_PROFILES.wasm
  .map((profile) => profile.filename));
export const DIST_RELEASE_FILENAMES = Object.freeze([
  ...BROWSER_RELEASE_FILENAMES,
  ...WASM_RELEASE_FILENAMES,
]);
export const NATIVE_RELEASE_FILENAMES = Object.freeze(RELEASE_PROFILES.native
  .map((profile) => profile.filename));
export const NATIVE_DEBUG_ARTIFACTS = Object.freeze(RELEASE_PROFILES.native
  .map((profile) => profile.debugArtifact));
export const NATIVE_DEBUG_EVIDENCE = Object.freeze(RELEASE_PROFILES.native
  .map((profile) => profile.debugEvidence));
export const FIXED_RELEASE_FILENAMES = Object.freeze([
  ...DIST_RELEASE_FILENAMES.map((filename) => `dist/<package-version>/${filename}`),
  ...NATIVE_RELEASE_FILENAMES,
]);

export function browserProfile(profileId) {
  const profile = RELEASE_PROFILES.browser.find(({ id }) => id === profileId);
  if (!profile) throw new Error(`Unknown browser release profile '${profileId}'.`);
  return profile;
}

export function wasmProfile(filename) {
  const profile = RELEASE_PROFILES.wasm.find((candidate) =>
    candidate.filename === filename || candidate.id === filename);
  if (!profile) throw new Error(`Unknown WASM release profile '${filename}'.`);
  return profile;
}

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function stableObject(value) {
  if (Array.isArray(value)) return value.map(stableObject);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort()
      .map((key) => [key, stableObject(value[key])]));
  }
  return value;
}

export function stableJSON(value) {
  return JSON.stringify(stableObject(value));
}

async function resolveQuotedInclude(repositoryRoot, sourcePath, includeName) {
  const candidates = [
    path.resolve(path.dirname(sourcePath), includeName),
    path.resolve(repositoryRoot, 'native/include', includeName),
    path.resolve(repositoryRoot, 'native/src', includeName),
    path.resolve(repositoryRoot, 'native/src/runtime', includeName),
    path.resolve(repositoryRoot, 'native/src/kernels', includeName),
    path.resolve(repositoryRoot, 'native/src/backends', includeName),
    path.resolve(repositoryRoot, 'native/src/training', includeName),
    path.resolve(repositoryRoot, 'native/third_party', includeName),
    path.resolve(repositoryRoot, 'native/third_party/synurang/include', includeName),
    path.resolve(repositoryRoot, 'native/third_party/xz-embedded', includeName),
    path.resolve(repositoryRoot, 'runtime/generated/c/inference', includeName),
    path.resolve(repositoryRoot, 'runtime/generated/c', includeName),
  ];
  for (const candidate of candidates) {
    try {
      if ((await fs.stat(candidate)).isFile()) return candidate;
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
    }
  }
  throw new Error(
    `Cannot resolve quoted include '${includeName}' from ${path.relative(repositoryRoot, sourcePath)}.`,
  );
}

async function collectSourceClosure(repositoryRoot, entries) {
  const pending = entries.map((entry) => path.resolve(repositoryRoot, entry));
  const visited = new Map();
  while (pending.length > 0) {
    const sourcePath = pending.pop();
    const relative = path.relative(repositoryRoot, sourcePath).split(path.sep).join('/');
    if (relative.startsWith('../') || path.isAbsolute(relative)) {
      throw new Error(`WASM provenance source escapes the repository: ${sourcePath}`);
    }
    if (visited.has(relative)) continue;
    const contents = await fs.readFile(sourcePath);
    visited.set(relative, contents);
    const text = contents.toString('utf8');
    for (const match of text.matchAll(/^\s*#\s*include\s*"([^"]+)"/gm)) {
      pending.push(await resolveQuotedInclude(repositoryRoot, sourcePath, match[1]));
    }
  }
  return [...visited].sort(([left], [right]) => left.localeCompare(right, 'en'));
}

function wasmComponentSources(component) {
  return component.objects.map(({ source }) => source);
}

async function declaredCapabilityExports(repositoryRoot, profile) {
  const names = new Set();
  for (const source of profile.capabilityExportSources) {
    const text = await fs.readFile(path.join(repositoryRoot, source), 'utf8');
    for (const match of text.matchAll(
      /\bVX_(?:TRAINING|PTQ)_EXPORT\("([A-Za-z0-9_]+)"\)/g,
    )) names.add(match[1]);
  }
  return [...names].sort();
}

function annotatedExportsFromSources(sources) {
  const names = new Set();
  for (const [, contents] of sources) {
    const source = contents.toString('utf8');
    const stack = [];
    let active = true;
    const releaseLines = [];
    for (const line of source.split(/\r?\n/)) {
      const conditional = line.match(/^\s*#\s*(if|ifdef|ifndef)\b(.*)$/);
      if (conditional) {
        const [, directive, expression] = conditional;
        const testing = /\b[A-Za-z0-9_]*TESTING\b/.test(expression);
        const negated = directive === 'ifndef' || /!\s*defined\s*\(/.test(expression);
        const branch = testing ? negated : true;
        stack.push({ parent: active, testing, branch });
        active = active && branch;
        continue;
      }
      if (/^\s*#\s*(?:else|elif)\b/.test(line)) {
        const frame = stack.at(-1);
        if (frame?.testing) frame.branch = !frame.branch;
        if (frame) active = frame.parent && frame.branch;
        continue;
      }
      if (/^\s*#\s*endif\b/.test(line)) {
        const frame = stack.pop();
        if (frame) active = frame.parent;
        continue;
      }
      if (active) releaseLines.push(line);
    }
    const text = releaseLines.join('\n');
    for (const match of text.matchAll(
      /\b(?:WASM_EXPORT|VX_TRAINING_EXPORT|VX_PTQ_EXPORT)\("([A-Za-z0-9_]+)"\)/g,
    )) names.add(match[1]);
  }
  return [...names].sort();
}

async function declaredProfileExports(repositoryRoot, profile) {
  // Embedded children have independent ABIs and are validated from their
  // custom sections below. Only annotations reachable from the parent link
  // recipe can declare exports on the parent module.
  return annotatedExportsFromSources(
    await collectSourceClosure(repositoryRoot, wasmComponentSources(profile.recipe.parent)),
  );
}

function expectedExportKind(profile, name) {
  const kind = profile.requiredExportKinds[name];
  if (kind === undefined) {
    throw new Error(
      `${profile.filename} export '${name}' is not owned by the generated internal ABI manifest.`,
    );
  }
  return kind;
}

function wasmContract(profile, capabilityExports, declaredExports) {
  const requiredExports = [...new Set([
    ...profile.requiredExports,
    ...declaredExports,
  ])].sort();
  return {
    format: 'volvoxai-wasm-release-contract/v1',
    profile: profile.id,
    capability: profile.capability,
    requiredExports,
    requiredExportKinds: Object.fromEntries(requiredExports
      .map((name) => [name, expectedExportKind(profile, name)])),
    capabilityExports,
    forbiddenExportPatterns: [...profile.forbiddenExportPatterns],
    forbiddenDependencyPrefixes: [...profile.forbiddenDependencyPrefixes],
    forbiddenDependencyPatterns: [...profile.forbiddenDependencyPatterns],
    forbiddenDependencyFiles: [...profile.forbiddenDependencyFiles],
    abiVersions: profile.abiVersions,
    allowedImports: ALLOWED_WASM_IMPORTS,
  };
}

export async function expectedWasmProvenance(repositoryRoot, filename, packageVersion) {
  const profile = wasmProfile(filename);
  const [sources, parentSources, capabilityExports] = await Promise.all([
    collectSourceClosure(repositoryRoot, profile.sourceEntries),
    collectSourceClosure(repositoryRoot, wasmComponentSources(profile.recipe.parent)),
    declaredCapabilityExports(repositoryRoot, profile),
  ]);
  const declaredExports = annotatedExportsFromSources(parentSources);
  const stableExports = new Set(profile.requiredExports);
  const unownedExports = declaredExports.filter((name) => !stableExports.has(name));
  if (unownedExports.length > 0) {
    throw new Error(
      `${profile.filename} declares exports outside its versioned central ABI: ` +
      unownedExports.join(', '),
    );
  }
  const sourceHasher = createHash('sha256');
  sourceHasher.update('volvoxai-wasm-source-set/v1\0');
  for (const [relative, contents] of sources) {
    sourceHasher.update(relative);
    sourceHasher.update('\0');
    sourceHasher.update(sha256(contents));
    sourceHasher.update('\0');
  }
  return Object.freeze({
    format: WASM_PROVENANCE_FORMAT,
    packageVersion,
    artifact: profile.filename,
    profile: profile.id,
    capability: profile.capability,
    sourceSha256: sourceHasher.digest('hex'),
    sourceFileCount: sources.length,
    recipeSha256: sha256(stableJSON(profile.recipe)),
    contractSha256: sha256(stableJSON(
      wasmContract(profile, capabilityExports, declaredExports),
    )),
  });
}

function decodeVarUint32(bytes, start, end = bytes.length) {
  let value = 0;
  let offset = start;
  for (let index = 0; index < 5 && offset < end; index += 1) {
    const byte = bytes[offset++];
    if (index === 4 && (byte & 0xf0) !== 0) break;
    value += (byte & 0x7f) * (2 ** (index * 7));
    if ((byte & 0x80) === 0) return { value, offset };
  }
  throw new Error(`Invalid WebAssembly varuint32 at byte ${start}.`);
}

function requireWasmHeader(input, label = 'Input') {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (bytes.length < 8 || bytes[0] !== 0x00 || bytes[1] !== 0x61 ||
      bytes[2] !== 0x73 || bytes[3] !== 0x6d || bytes[4] !== 0x01 ||
      bytes[5] !== 0x00 || bytes[6] !== 0x00 || bytes[7] !== 0x00) {
    throw new Error(`${label} is not a version-1 WebAssembly module.`);
  }
  return bytes;
}

function decodeWasmString(bytes, start, end, label) {
  const decoded = decodeVarUint32(bytes, start, end);
  const stringEnd = decoded.offset + decoded.value;
  if (stringEnd > end) throw new Error(`${label} exceeds its WebAssembly section.`);
  let value;
  try {
    value = new TextDecoder('utf-8', { fatal: true })
      .decode(bytes.subarray(decoded.offset, stringEnd));
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8.`, { cause: error });
  }
  return Object.freeze({ value, offset: stringEnd });
}

function decodeCustomSectionName(bytes, start, end) {
  const decoded = decodeVarUint32(bytes, start);
  const nameEnd = decoded.offset + decoded.value;
  if (nameEnd > end) throw new Error('WebAssembly custom-section name exceeds its section.');
  return new TextDecoder().decode(bytes.subarray(decoded.offset, nameEnd));
}

export function withoutWasmCustomSection(input, sectionName) {
  const bytes = requireWasmHeader(input);
  const retained = [bytes.subarray(0, 8)];
  let offset = 8;
  while (offset < bytes.length) {
    const sectionStart = offset;
    const sectionId = bytes[offset++];
    const decoded = decodeVarUint32(bytes, offset);
    offset = decoded.offset;
    const sectionEnd = offset + decoded.value;
    if (sectionEnd > bytes.length) {
      throw new Error(`WebAssembly section at byte ${sectionStart} exceeds the module.`);
    }
    const name = sectionId === 0
      ? decodeCustomSectionName(bytes, offset, sectionEnd)
      : null;
    if (name !== sectionName) retained.push(bytes.subarray(sectionStart, sectionEnd));
    offset = sectionEnd;
  }
  const size = retained.reduce((total, chunk) => total + chunk.byteLength, 0);
  const output = new Uint8Array(size);
  let cursor = 0;
  for (const chunk of retained) {
    output.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return output;
}

export function orderedWasmCustomSections(input) {
  const bytes = requireWasmHeader(input, 'Custom-section input');
  const sections = [];
  let offset = 8;
  let index = 0;
  while (offset < bytes.length) {
    const sectionId = bytes[offset++];
    const decoded = decodeVarUint32(bytes, offset, bytes.length);
    offset = decoded.offset;
    const sectionEnd = offset + decoded.value;
    if (sectionEnd > bytes.length) throw new Error('WebAssembly section exceeds the module.');
    if (sectionId === 0) {
      const name = decodeWasmString(
        bytes,
        offset,
        sectionEnd,
        `WebAssembly custom section ${index} name`,
      );
      sections.push(Object.freeze({
        index,
        name: name.value,
        payload: Buffer.from(bytes.subarray(name.offset, sectionEnd)),
      }));
    }
    offset = sectionEnd;
    index += 1;
  }
  return Object.freeze(sections);
}

function wasmCustomSectionNames(input) {
  return orderedWasmCustomSections(input).map(({ name }) => name);
}

function skipWasmLimits(bytes, start, end, label) {
  const flags = decodeVarUint32(bytes, start, end);
  if ((flags.value & ~0x03) !== 0) {
    throw new Error(`${label} uses unsupported limits flags ${flags.value}.`);
  }
  const minimum = decodeVarUint32(bytes, flags.offset, end);
  return (flags.value & 0x01) === 0
    ? minimum.offset
    : decodeVarUint32(bytes, minimum.offset, end).offset;
}

/** Parse imports/exports without validating proposal opcodes in the code section. */
export function inspectWasmInterface(input, label = 'WebAssembly module') {
  const bytes = requireWasmHeader(input, label);
  const imports = [];
  const exports = [];
  const kindNames = ['function', 'table', 'memory', 'global', 'tag'];
  const seenInterfaceSections = new Set();
  let offset = 8;
  while (offset < bytes.length) {
    const sectionId = bytes[offset++];
    const decoded = decodeVarUint32(bytes, offset, bytes.length);
    const sectionStart = decoded.offset;
    const sectionEnd = sectionStart + decoded.value;
    if (sectionEnd > bytes.length) throw new Error(`${label} has an oversized section.`);
    if (sectionId === 2) {
      if (seenInterfaceSections.has(sectionId)) {
        throw new Error(`${label} repeats its import section.`);
      }
      seenInterfaceSections.add(sectionId);
      let cursor = sectionStart;
      const count = decodeVarUint32(bytes, cursor, sectionEnd);
      cursor = count.offset;
      for (let index = 0; index < count.value; index += 1) {
        const namespace = decodeWasmString(bytes, cursor, sectionEnd, `${label} import module`);
        const name = decodeWasmString(bytes, namespace.offset, sectionEnd, `${label} import name`);
        cursor = name.offset;
        if (cursor >= sectionEnd) throw new Error(`${label} import ${index} has no kind.`);
        const kind = bytes[cursor++];
        const kindName = kindNames[kind];
        if (kindName === undefined) throw new Error(`${label} import ${index} has kind ${kind}.`);
        if (kind === 0) {
          cursor = decodeVarUint32(bytes, cursor, sectionEnd).offset;
        } else if (kind === 1) {
          if (cursor >= sectionEnd) throw new Error(`${label} table import is truncated.`);
          cursor = skipWasmLimits(bytes, cursor + 1, sectionEnd, `${label} table import`);
        } else if (kind === 2) {
          cursor = skipWasmLimits(bytes, cursor, sectionEnd, `${label} memory import`);
        } else if (kind === 3) {
          if (cursor + 2 > sectionEnd) throw new Error(`${label} global import is truncated.`);
          cursor += 2;
        } else {
          if (cursor >= sectionEnd) throw new Error(`${label} tag import is truncated.`);
          cursor = decodeVarUint32(bytes, cursor + 1, sectionEnd).offset;
        }
        imports.push(`${namespace.value}.${name.value}:${kindName}`);
      }
      if (cursor !== sectionEnd) throw new Error(`${label} import section has trailing bytes.`);
    } else if (sectionId === 7) {
      if (seenInterfaceSections.has(sectionId)) {
        throw new Error(`${label} repeats its export section.`);
      }
      seenInterfaceSections.add(sectionId);
      let cursor = sectionStart;
      const count = decodeVarUint32(bytes, cursor, sectionEnd);
      cursor = count.offset;
      for (let index = 0; index < count.value; index += 1) {
        const name = decodeWasmString(bytes, cursor, sectionEnd, `${label} export name`);
        cursor = name.offset;
        if (cursor >= sectionEnd) throw new Error(`${label} export ${index} has no kind.`);
        const kind = bytes[cursor++];
        const kindName = kindNames[kind];
        if (kindName === undefined) throw new Error(`${label} export ${index} has kind ${kind}.`);
        cursor = decodeVarUint32(bytes, cursor, sectionEnd).offset;
        exports.push(`${name.value}:${kindName}`);
      }
      if (cursor !== sectionEnd) throw new Error(`${label} export section has trailing bytes.`);
    }
    offset = sectionEnd;
  }
  return Object.freeze({
    imports: Object.freeze(imports.sort()),
    exports: Object.freeze(exports.sort()),
  });
}

function assertReleaseCustomSectionOrder(bytes, profile, { allowMissingProvenance = false } = {}) {
  const names = wasmCustomSectionNames(bytes);
  const debugSections = names.filter((name) => name === 'name' || name.startsWith('.debug_'));
  if (debugSections.length > 0) {
    throw new Error(`${profile.filename} retains debug sections: ${debugSections.join(', ')}.`);
  }
  const actual = names.filter((name) => name.startsWith('volvoxai.'));
  const expected = [...profile.recipe.customSections];
  const preProvenance = expected.filter((name) => name !== WASM_PROVENANCE_SECTION);
  if (stableJSON(actual) !== stableJSON(expected) &&
      !(allowMissingProvenance && stableJSON(actual) === stableJSON(preProvenance))) {
    throw new Error(
      `${profile.filename} release-owned custom sections are ${actual.join(', ')}; ` +
      `expected ${expected.join(', ')} in that order.`,
    );
  }
}

function requireSha256(value, label) {
  if (typeof value !== 'string' || !/^[0-9a-f]{64}$/u.test(value)) {
    throw new Error(`${label} must be a lowercase SHA-256.`);
  }
  return value;
}

function mathImports(module, profileId) {
  const imports = WebAssembly.Module.imports(module);
  const actual = imports.map(({ module: namespace, name, kind }) =>
    `${namespace}.${name}:${kind}`).sort();
  const expected = profileId === undefined
    ? ALLOWED_WASM_IMPORTS
    : WASM_PROFILE_IMPORTS[profileId];
  if (!Array.isArray(expected)) {
    throw new Error(`No declared WASM import set for profile '${profileId}'.`);
  }
  if (stableJSON(actual) !== stableJSON(expected)) {
    throw new Error(
      `WASM imports differ from the release ABI for ${profileId ?? 'parent'}: ` +
      `${actual.join(', ')}; expected ${expected.join(', ')}.`,
    );
  }
  return wasmToolImports();
}

function moduleInterface(module) {
  return Object.freeze({
    imports: Object.freeze(WebAssembly.Module.imports(module)
      .map(({ module: namespace, name, kind }) => `${namespace}.${name}:${kind}`)
      .sort()),
    exports: Object.freeze(WebAssembly.Module.exports(module)
      .map(({ name, kind }) => `${name}:${kind}`)
      .sort()),
  });
}

async function validateRelaxedSimdModule(child, label) {
  try {
    return moduleInterface(new WebAssembly.Module(child));
  } catch (baselineError) {
    const inspector = [
      'const bytes = Buffer.from(process.argv[1], "base64");',
      'const module = new WebAssembly.Module(bytes);',
      'const imports = WebAssembly.Module.imports(module)',
      '  .map(({module, name, kind}) => `${module}.${name}:${kind}`).sort();',
      'const exports = WebAssembly.Module.exports(module)',
      '  .map(({name, kind}) => `${name}:${kind}`).sort();',
      'process.stdout.write(JSON.stringify({imports, exports}));',
    ].join('\n');
    let result;
    try {
      result = await executeFile(process.execPath, [
        '--experimental-wasm-relaxed-simd',
        '--eval',
        inspector,
        child.toString('base64'),
      ], { encoding: 'utf8', maxBuffer: 1024 * 1024 });
    } catch (error) {
      const detail = [error?.stdout, error?.stderr].filter(Boolean).join('\n').trim();
      throw new Error(
        `${label} is not valid Relaxed-SIMD WebAssembly${detail ? `:\n${detail}` : '.'}`,
        { cause: baselineError },
      );
    }
    let interface_;
    try {
      interface_ = JSON.parse(result.stdout);
    } catch (error) {
      throw new Error(`${label} validator returned invalid interface JSON.`, { cause: error });
    }
    if (!Array.isArray(interface_.imports) || !Array.isArray(interface_.exports) ||
        interface_.imports.some((entry) => typeof entry !== 'string') ||
        interface_.exports.some((entry) => typeof entry !== 'string')) {
      throw new Error(`${label} validator returned an invalid interface.`);
    }
    return Object.freeze({
      imports: Object.freeze([...interface_.imports]),
      exports: Object.freeze([...interface_.exports]),
    });
  }
}

export async function validateWasmModuleStructure(
  input,
  { relaxedSimd = false, label = 'WebAssembly module' } = {},
) {
  const bytes = Buffer.from(input);
  if (relaxedSimd) return validateRelaxedSimdModule(bytes, label);
  try {
    return moduleInterface(new WebAssembly.Module(bytes));
  } catch (error) {
    throw new Error(`${label} is not valid baseline WebAssembly.`, { cause: error });
  }
}

async function assertRelaxedSimdSection(module, profile) {
  if (!profile.recipe.customSections.includes(WASM_RELAXED_SIMD_SECTION)) return undefined;
  const sections = WebAssembly.Module.customSections(module, WASM_RELAXED_SIMD_SECTION);
  if (sections.length !== 1) {
    throw new Error(
      `${profile.filename} must contain exactly one '${WASM_RELAXED_SIMD_SECTION}' section.`,
    );
  }
  const child = Buffer.from(sections[0]);
  const wasmHeader = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
  if (!child.subarray(0, wasmHeader.length).equals(wasmHeader)) {
    throw new Error(`${profile.filename} relaxed-SIMD section is not a version-1 WASM child.`);
  }
  const interface_ = await validateWasmModuleStructure(
    child,
    { relaxedSimd: true, label: `${profile.filename} relaxed-SIMD child` },
  );
  if (stableJSON(interface_.imports) !== stableJSON(profile.recipe.relaxed.imports) ||
      stableJSON(interface_.exports) !== stableJSON(profile.recipe.relaxed.exports)) {
    throw new Error(
      `${profile.filename} relaxed-SIMD child interface differs from its recipe: ` +
      `imports ${interface_.imports.join(', ')}, exports ${interface_.exports.join(', ')}.`,
    );
  }
  return sha256(child);
}

function assertPtqAuthoringSection(module, profile) {
  const recipe = profile.recipe.ptqAuthoring;
  const sections = WebAssembly.Module.customSections(
    module,
    WASM_PTQ_AUTHORING_SECTION,
  );
  if (recipe === null) {
    if (sections.length !== 0) {
      throw new Error(
        `${profile.filename} inference profile must not embed PTQ authoring.`,
      );
    }
    return undefined;
  }
  if (sections.length !== 1 || sections[0].byteLength === 0) {
    throw new Error(
      `${profile.filename} must contain exactly one non-empty ` +
      `'${WASM_PTQ_AUTHORING_SECTION}' section.`,
    );
  }

  let childModule;
  try {
    childModule = new WebAssembly.Module(sections[0]);
  } catch (error) {
    throw new Error(`${profile.filename} PTQ authoring child is not valid WASM.`, {
      cause: error,
    });
  }
  const actualImports = WebAssembly.Module.imports(childModule)
    .map(({ module: namespace, name, kind }) => `${namespace}.${name}:${kind}`)
    .sort();
  if (stableJSON(actualImports) !== stableJSON(recipe.imports)) {
    throw new Error(
      `${profile.filename} PTQ authoring imports differ from its child ABI: ` +
      actualImports.join(', '),
    );
  }
  const actualExports = WebAssembly.Module.exports(childModule)
    .map(({ name, kind }) => `${name}:${kind}`)
    .sort();
  if (stableJSON(actualExports) !== stableJSON(recipe.exports)) {
    throw new Error(
      `${profile.filename} PTQ authoring exports differ from its child ABI: ` +
      actualExports.join(', '),
    );
  }

  const child = new WebAssembly.Instance(childModule, {
    env: {
      vx_ptq_host_format_double: () => -1,
      vx_ptq_host_parse_double: () => 0,
    },
  });
  const abiVersion = child.exports.vx_ptq_wasm_abi_version?.();
  if (abiVersion !== recipe.abiVersion) {
    throw new Error(
      `${profile.filename} PTQ authoring child exposes ABI ` +
      `${String(abiVersion)}; expected ${recipe.abiVersion}.`,
    );
  }
  return sha256(new Uint8Array(sections[0]));
}

async function assertWasmBoundary(module, profile, capabilityExports, declaredExports) {
  const exports = WebAssembly.Module.exports(module);
  const names = new Map(exports.map((entry) => [entry.name, entry.kind]));
  const requiredExports = new Set([...profile.requiredExports, ...declaredExports]);
  for (const required of requiredExports) {
    if (!names.has(required)) {
      throw new Error(`${profile.filename} is missing required export '${required}'.`);
    }
    const expectedKind = expectedExportKind(profile, required);
    if (names.get(required) !== expectedKind) {
      throw new Error(
        `${profile.filename} export '${required}' has kind '${names.get(required)}'; ` +
        `expected '${expectedKind}'.`,
      );
    }
  }
  for (const required of capabilityExports) {
    if (names.get(required) !== 'function') {
      throw new Error(`${profile.filename} is missing capability export '${required}'.`);
    }
  }
  for (const pattern of profile.forbiddenExportPatterns) {
    const expression = new RegExp(pattern);
    const leaked = [...names.keys()].find((name) => expression.test(name));
    if (leaked) {
      throw new Error(`${profile.filename} leaks forbidden export '${leaked}'.`);
    }
  }
  const relaxedSimdSha256 = await assertRelaxedSimdSection(module, profile);
  const ptqAuthoringSha256 = assertPtqAuthoringSection(module, profile);
  const instance = new WebAssembly.Instance(module, mathImports(module, profile.id));
  const versions = {};
  for (const [name, { export: exportName, value: expected }] of
    Object.entries(profile.abiVersionExports)) {
    const actual = instance.exports[exportName]?.();
    if (actual !== expected) {
      throw new Error(
        `${profile.filename} exposes ${name} ABI ${String(actual)}; expected ${expected}.`,
      );
    }
    versions[name] = actual;
  }
  return Object.freeze({
    exports: Object.freeze(exports.map(({ name, kind }) => Object.freeze({ name, kind }))),
    abiVersions: Object.freeze(versions),
    relaxedSimdSha256,
    ptqAuthoringSha256,
  });
}

export async function createWasmProvenance(
  repositoryRoot,
  artifactPath,
  packageVersion,
  { buildEvidenceSha256 } = {},
) {
  const filename = path.basename(artifactPath);
  const [expected, capabilityExports, declaredExports] = await Promise.all([
    expectedWasmProvenance(repositoryRoot, filename, packageVersion),
    declaredCapabilityExports(repositoryRoot, wasmProfile(filename)),
    declaredProfileExports(repositoryRoot, wasmProfile(filename)),
  ]);
  requireSha256(buildEvidenceSha256, `${filename} build evidence SHA-256`);
  const bytes = new Uint8Array(await fs.readFile(artifactPath));
  const module = new WebAssembly.Module(bytes);
  assertReleaseCustomSectionOrder(bytes, wasmProfile(filename), {
    allowMissingProvenance: true,
  });
  await assertWasmBoundary(module, wasmProfile(filename), capabilityExports, declaredExports);
  const payload = withoutWasmCustomSection(bytes, WASM_PROVENANCE_SECTION);
  return Object.freeze({
    ...expected,
    payloadSha256: sha256(payload),
    buildEvidenceSha256,
  });
}

export function parseCanonicalWasmProvenance(contents, filename = 'WASM artifact') {
  const bytes = Buffer.from(contents);
  let provenance;
  try {
    provenance = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (error) {
    throw new Error(`${filename} has invalid JSON release provenance.`, { cause: error });
  }
  if (provenance === null || typeof provenance !== 'object' || Array.isArray(provenance)) {
    throw new Error(`${filename} release provenance must be a JSON object.`);
  }
  const canonical = Buffer.from(`${stableJSON(provenance)}\n`);
  if (!bytes.equals(canonical)) {
    throw new Error(`${filename} release provenance is not canonical JSON.`);
  }
  return Object.freeze(provenance);
}

export async function validateWasmArtifact(
  repositoryRoot,
  artifactPath,
  packageVersion,
  { buildEvidenceSha256 } = {},
) {
  const filename = path.basename(artifactPath);
  const profile = wasmProfile(filename);
  const [expected, capabilityExports, declaredExports] = await Promise.all([
    expectedWasmProvenance(repositoryRoot, filename, packageVersion),
    declaredCapabilityExports(repositoryRoot, profile),
    declaredProfileExports(repositoryRoot, profile),
  ]);
  const bytes = new Uint8Array(await fs.readFile(artifactPath));
  const module = new WebAssembly.Module(bytes);
  assertReleaseCustomSectionOrder(bytes, profile);
  const boundary = await assertWasmBoundary(
    module,
    profile,
    capabilityExports,
    declaredExports,
  );
  const sections = WebAssembly.Module.customSections(module, WASM_PROVENANCE_SECTION);
  if (sections.length !== 1) {
    throw new Error(
      `${filename} must contain exactly one '${WASM_PROVENANCE_SECTION}' custom section.`,
    );
  }
  const provenance = parseCanonicalWasmProvenance(sections[0], filename);
  for (const [key, value] of Object.entries(expected)) {
    if (provenance[key] !== value) {
      throw new Error(
        `${filename} provenance '${key}' is ${JSON.stringify(provenance[key])}; ` +
        `expected ${JSON.stringify(value)}.`,
      );
    }
  }
  const expectedKeys = [
    ...Object.keys(expected),
    'payloadSha256',
    'buildEvidenceSha256',
  ].sort();
  if (stableJSON(Object.keys(provenance).sort()) !== stableJSON(expectedKeys)) {
    throw new Error(`${filename} provenance contains unexpected or missing fields.`);
  }
  requireSha256(
    provenance.buildEvidenceSha256,
    `${filename} provenance buildEvidenceSha256`,
  );
  const payloadSha256 = sha256(withoutWasmCustomSection(bytes, WASM_PROVENANCE_SECTION));
  if (provenance.payloadSha256 !== payloadSha256) {
    throw new Error(
      `${filename} payload hash is ${payloadSha256}; provenance records ` +
      `${String(provenance.payloadSha256)}.`,
    );
  }
  if (buildEvidenceSha256 !== undefined) {
    requireSha256(buildEvidenceSha256, `${filename} expected build evidence SHA-256`);
    if (provenance.buildEvidenceSha256 !== buildEvidenceSha256) {
      throw new Error(
        `${filename} build-evidence hash is ${buildEvidenceSha256}; provenance ` +
        `records ${provenance.buildEvidenceSha256}.`,
      );
    }
  }
  return Object.freeze({
    profile,
    provenance: Object.freeze(provenance),
    boundary,
    artifactSha256: sha256(bytes),
  });
}

export async function removeInvalidWasmSidecars(
  repositoryRoot,
  outputDirectory,
  packageVersion,
) {
  const removed = [];
  for (const filename of WASM_RELEASE_FILENAMES) {
    const artifactPath = path.join(outputDirectory, filename);
    try {
      await validateWasmArtifact(repositoryRoot, artifactPath, packageVersion);
    } catch (error) {
      if (error?.code === 'ENOENT') continue;
      await fs.rm(artifactPath, { force: true });
      removed.push(Object.freeze({ filename, reason: error.message }));
    }
  }
  return Object.freeze(removed);
}

export async function browserProvenance(repositoryRoot, profileId, packageVersion) {
  const profile = browserProfile(profileId);
  // A sidecar hash ties a JS bundle to the WASM it loads. A profile that loads
  // none records no hash rather than borrowing another profile's, which would
  // claim a dependency it does not have.
  if (profile.sidecar === null) {
    return Object.freeze({
      format: BROWSER_PROVENANCE_FORMAT,
      packageVersion,
      profile: profile.id,
      capability: profile.capability,
      backends: profile.backends,
      sidecar: null,
    });
  }
  const sidecar = await expectedWasmProvenance(
    repositoryRoot,
    profile.sidecar,
    packageVersion,
  );
  return Object.freeze({
    format: BROWSER_PROVENANCE_FORMAT,
    packageVersion,
    profile: profile.id,
    capability: profile.capability,
    backends: profile.backends,
    sidecar: profile.sidecar,
    wasmSourceSha256: sidecar.sourceSha256,
    wasmRecipeSha256: sidecar.recipeSha256,
    wasmContractSha256: sidecar.contractSha256,
  });
}

export async function browserProvenanceBanner(repositoryRoot, profileId, packageVersion) {
  return `/*! ${BROWSER_PROVENANCE_MARKER} ${stableJSON(
    await browserProvenance(repositoryRoot, profileId, packageVersion),
  )} */`;
}

function compactWGSLTextPlugin() {
  return {
    name: 'compact-wgsl-text',
    setup(buildContext) {
      buildContext.onLoad({ filter: /\.wgsl$/, namespace: 'file' }, async (args) => {
        const source = await fs.readFile(args.path, 'utf8');
        const contents = source
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0 && !line.startsWith('//'))
          .join('\n');
        return { contents, loader: 'text' };
      });
    },
  };
}

const INFERENCE_BROWSER_FORBIDDEN_INPUTS = Object.freeze([
  /(^|\/)ts\/training\//,
  /(^|\/)ts\/host\/FullEngineHostCore\.ts$/,
  /(^|\/)ts\/host\/EngineHostCore\.ts$/,
  /(^|\/)ts\/host\/PtqAuthoringWasm\.ts$/,
  /(^|\/)ts\/(?:full|wasm)\.ts$/,
  /(^|\/)ts\/(?:InferenceProfile|RuntimeComposition)\.ts$/,
  /(^|\/)ts\/generated\/[^/]*Full[^/]*\.ts$/,
  /(^|\/)runtime\/generated\/typescript\/volvoxai_(?:ffi|lite)\.ts$/,
  // The ordinary entry is a thin generated-proto-to-C bridge. Reintroducing
  // the TypeScript graph executor would duplicate the native WASM authority.
  /(^|\/)ts\/backends\/(?:GraphExecutor|WasmBackendProvider|WasmEngine)\.ts$/,
  /(^|\/)ts\/core\/ContextRuntime\.ts$/,
  // Inference uses its own catalogue with no training shader text or layouts.
  /(^|\/)ts\/generated\/shaderCatalog\.ts$/,
  /(^|\/)ts\/(?:backends\/ShaderLibrary|training\/TrainingShaderLibrary)\.ts$/,
  /(^|\/)ts\/VolvoxAI\.ts$/,
  /(^|\/)shaders\/training\/[^/]*\.wgsl$/,
]);

const INFERENCE_BROWSER_FORBIDDEN_CONTENT = Object.freeze([
  'volvoxai.v1.CreateTrainerRequest',
  'volvoxai.v1.TrainStepRequest',
  'volvoxai.v1.AuthorPtqTemplateRequest',
  'volvoxai.v1.CreatePtqPlanRequest',
  'volvoxai.v1.VxTrainingService',
  'volvoxai.v1.VxQuantizationService',
]);

const INFERENCE_BROWSER_FORBIDDEN_INTERNAL_WASM_CONTENT = Object.freeze([
  'volvoxai_training_',
  'volvoxai_ptq_',
  'vx_training_control_',
  'vx_ptq_wasm_',
]);

// These modules encoded private planning wire formats whose native receivers
// no longer exist. Planning now crosses the generated VxPlanningService
// boundary, so no browser profile may retain executable bytes from them.
const RETIRED_BROWSER_PLANNING_INPUTS = Object.freeze([
  'ts/backends/WasmActivationPlan.ts',
  'ts/backends/WasmGraphPlan.ts',
  'ts/backends/WasmInferencePlan.ts',
  'ts/backends/WasmShapeContract.ts',
  'ts/core/IncrementalDependencyLiveness.ts',
  'ts/core/PortableCallSequencePolicy.ts',
  'ts/core/PortableGraphBind.ts',
  'ts/core/PortableGraphDomain.ts',
  'ts/core/PortableGraphPlanWire.ts',
  'ts/core/PortableGraphShapeResolver.ts',
  'ts/core/PortableIndependentBatch.ts',
  'ts/core/PortableSafetensorsHeader.ts',
  'ts/core/ResolvedShapePlanProvenance.ts',
  'ts/backends/incrementalExecution.ts',
]);

// Full-only parameter names and alias-group ids must enter only full/strict
// compositions. GraphPlan projection is profile-injected; an inference or
// authoring bundle retaining either module would silently widen its contract.
const FULL_ONLY_PLANNING_PARAMETER_INPUTS = Object.freeze([
  'ts/core/GraphPlanningParameterRegistryFull.ts',
  'ts/generated/operatorParamRegistryFull.ts',
]);

const FULL_ONLY_PLANNING_PARAMETER_CONTENT = Object.freeze([
  'attention_dropout',
  'dropout_seed',
  'training_seed',
  'attention-dropout-probability',
  'attention-dropout-seed',
]);

const FULL_PROTOBUF_CODEC_INPUTS = Object.freeze([
  'runtime/generated/typescript/volvoxai_ffi.ts',
  'runtime/generated/typescript/volvoxai_lite.ts',
]);

const INFERENCE_PROTOBUF_CODEC_INPUTS = Object.freeze([
  'runtime/generated/typescript/inference/volvoxai_ffi.ts',
  'runtime/generated/typescript/inference/volvoxai_lite.ts',
]);

function retainedInputBytes(metafile, candidates) {
  const retained = new Map();
  for (const output of Object.values(metafile?.outputs ?? {})) {
    for (const [rawInput, contribution] of Object.entries(output.inputs ?? {})) {
      const input = rawInput.replaceAll('\\', '/');
      const candidate = candidates.find((value) =>
        input === value || input.endsWith(`/${value}`));
      const bytes = contribution?.bytesInOutput ?? 0;
      if (candidate !== undefined && bytes > 0) {
        retained.set(candidate, (retained.get(candidate) ?? 0) + bytes);
      }
    }
  }
  return retained;
}

function validateBrowserProfileBoundary(profile, metafile, contents) {
  const inputs = Object.keys(metafile?.inputs ?? {}).map((input) =>
    input.replaceAll('\\', '/'));
  if (!profile.backends.includes('webgpu')) {
    const gpuInputs = inputs.filter((input) =>
      /(?:^|\/)ts\/backends\/WebGPUHostBridge\.ts$/.test(input) ||
      /(?:^|\/)ts\/generated\/(?:shaderCatalog[^/]*|gpuBridge)\.ts$/.test(input) ||
      input.endsWith('.wgsl'));
    if (gpuInputs.length > 0) {
      throw new Error(`Browser ${profile.id} profile imports WebGPU dependencies:\n` +
        gpuInputs.map((input) => `  ${input}`).join('\n'));
    }
  }
  const retainedPlanningInputs = new Map();
  for (const output of Object.values(metafile?.outputs ?? {})) {
    for (const [rawInput, contribution] of Object.entries(output.inputs ?? {})) {
      const input = rawInput.replaceAll('\\', '/');
      const retired = RETIRED_BROWSER_PLANNING_INPUTS.find((candidate) =>
        input === candidate || input.endsWith(`/${candidate}`));
      const bytes = contribution?.bytesInOutput ?? 0;
      if (retired !== undefined && bytes > 0) {
        retainedPlanningInputs.set(retired,
          (retainedPlanningInputs.get(retired) ?? 0) + bytes);
      }
    }
  }
  if (retainedPlanningInputs.size > 0) {
    throw new Error(
      `Browser ${profile.id} profile retains retired planning wire code:\n` +
      [...retainedPlanningInputs.entries()]
        .sort(([left], [right]) => left.localeCompare(right, 'en'))
        .map(([input, bytes]) => `  ${input}: ${bytes} B`)
        .join('\n'),
    );
  }
  const wrongProjectionInputs = profile.capability === 'full'
    ? INFERENCE_PROTOBUF_CODEC_INPUTS
    : FULL_PROTOBUF_CODEC_INPUTS;
  const retainedWrongProjection = retainedInputBytes(metafile, wrongProjectionInputs);
  if (retainedWrongProjection.size > 0) {
    throw new Error(
      `Browser ${profile.id} profile retains the wrong protobuf projection:\n` +
      [...retainedWrongProjection.entries()]
        .sort(([left], [right]) => left.localeCompare(right, 'en'))
        .map(([input, bytes]) => `  ${input}: ${bytes} B`)
        .join('\n'),
    );
  }
  if (!profile.publicTraining) {
    const retainedFullParameterInputs = new Map();
    for (const output of Object.values(metafile?.outputs ?? {})) {
      for (const [rawInput, contribution] of Object.entries(output.inputs ?? {})) {
        const input = rawInput.replaceAll('\\', '/');
        const fullOnly = FULL_ONLY_PLANNING_PARAMETER_INPUTS.find((candidate) =>
          input === candidate || input.endsWith(`/${candidate}`));
        const bytes = contribution?.bytesInOutput ?? 0;
        if (fullOnly !== undefined && bytes > 0) {
          retainedFullParameterInputs.set(
            fullOnly,
            (retainedFullParameterInputs.get(fullOnly) ?? 0) + bytes,
          );
        }
      }
    }
    if (retainedFullParameterInputs.size > 0) {
      throw new Error(
        `Browser ${profile.id} profile retains full-only Planning parameter code:\n` +
        [...retainedFullParameterInputs.entries()]
          .sort(([left], [right]) => left.localeCompare(right, 'en'))
          .map(([input, bytes]) => `  ${input}: ${bytes} B`)
          .join('\n'),
      );
    }
    const source = new TextDecoder().decode(contents);
    const leakedParameters = FULL_ONLY_PLANNING_PARAMETER_CONTENT.filter((marker) =>
      source.includes(marker));
    if (leakedParameters.length > 0) {
      throw new Error(
        `Browser ${profile.id} profile contains full-only Planning parameter strings: ` +
        leakedParameters.join(', '),
      );
    }
  }
  const embeddedModelControl = inputs.find((input) =>
    /(^|\/)ts\/generated\/modelControlWasm\.ts$/.test(input));
  if (embeddedModelControl !== undefined) {
    throw new Error(
      `Browser ${profile.id} profile embeds the removed model-control payload: ` +
      embeddedModelControl,
    );
  }
  if (profile.id === 'inference') {
    const forbiddenInputs = inputs.filter((input) =>
      INFERENCE_BROWSER_FORBIDDEN_INPUTS.some((pattern) => pattern.test(input)));
    if (forbiddenInputs.length > 0) {
      throw new Error(
        "Browser inference profile includes forbidden execution/profile inputs:\n" +
        forbiddenInputs.sort().map((input) => `  ${input}`).join('\n'),
      );
    }

    const source = new TextDecoder().decode(contents);
    const forbiddenContent = INFERENCE_BROWSER_FORBIDDEN_CONTENT.filter((marker) =>
      source.includes(marker));
    if (forbiddenContent.length > 0) {
      throw new Error(
        'Browser inference profile contains full protobuf symbols: ' +
        forbiddenContent.join(', '),
      );
    }
  }

  if (!profile.publicTraining && !profile.publicPTQ) {
    const source = new TextDecoder().decode(contents);
    const leakedInternalAbi = INFERENCE_BROWSER_FORBIDDEN_INTERNAL_WASM_CONTENT
      .filter((marker) => source.includes(marker));
    if (leakedInternalAbi.length > 0) {
      throw new Error(
        `Browser ${profile.id} profile contains full-only internal WASM ABI strings: ` +
        leakedInternalAbi.join(', '),
      );
    }
  }

}

/** Inspect one browser release with the exact central build configuration. */
export async function inspectBrowserReleaseBundle(
  repositoryRoot,
  profileId,
  packageVersion,
  { minify = false, outfile, logLevel = 'silent' } = {},
) {
  const profile = browserProfile(profileId);
  // Release declarations and WASM provenance are dependency-free. Load
  // esbuild only at the browser-build boundary so a clean `make build_wasm`
  // does not require node_modules before provenance can be written.
  const { build, version: esbuildVersion } = await import('esbuild');
  const result = await build({
    absWorkingDir: repositoryRoot,
    entryPoints: [profile.entryPoint],
    ...(outfile === undefined ? {} : { outfile }),
    bundle: true,
    format: 'esm',
    target: ['es2022'],
    loader: { '.wgsl': 'text' },
    ...(minify ? { plugins: [compactWGSLTextPlugin()] } : {}),
    define: {
      __VOLVOXAI_VERSION__: JSON.stringify(packageVersion),
      ...(profile.browserOnly ? { __VOLVOXAI_BROWSER_ONLY__: 'true' } : {}),
    },
    banner: {
      js: await browserProvenanceBanner(repositoryRoot, profile.id, packageVersion),
    },
    // The upstream codec resolves nested messages by stable protobuf typeName.
    keepNames: false,
    minify,
    minifySyntax: profile.browserOnly || minify,
    logLevel,
    metafile: true,
    // Inspect the complete dependency graph and bytes before publishing them.
    // This also keeps the mutating builder and read-only reproducibility gate
    // on one exact boundary implementation.
    write: false,
  });
  if (result.outputFiles?.length !== 1) {
    throw new Error(`Browser profile '${profileId}' did not produce exactly one bundle.`);
  }
  const contents = result.outputFiles[0].contents;
  validateBrowserProfileBoundary(profile, result.metafile, contents);
  return Object.freeze({ contents, metafile: result.metafile, esbuildVersion });
}

export async function buildBrowserReleaseBundle(
  repositoryRoot,
  profileId,
  packageVersion,
  { minify = false, outfile, write = outfile !== undefined, logLevel = 'silent' } = {},
) {
  if (write && outfile === undefined) {
    throw new Error(`Browser profile '${profileId}' requires an output path when write=true.`);
  }
  const { contents } = await inspectBrowserReleaseBundle(
    repositoryRoot,
    profileId,
    packageVersion,
    { minify, outfile, logLevel },
  );
  if (!write) return contents;
  await fs.mkdir(path.dirname(outfile), { recursive: true });
  await fs.writeFile(outfile, contents);
  return undefined;
}

export async function validateBrowserReleasePayload(
  repositoryRoot,
  artifactPath,
  profileId,
  packageVersion,
  { minify = false } = {},
) {
  const [actual, expected] = await Promise.all([
    fs.readFile(artifactPath),
    buildBrowserReleaseBundle(repositoryRoot, profileId, packageVersion, {
      outfile: artifactPath,
      minify,
      write: false,
    }),
  ]);
  if (!actual.equals(expected)) {
    throw new Error(
      `browser payload hash ${sha256(actual)} differs from reproducible ` +
      `profile build ${sha256(expected)}`,
    );
  }
  return Object.freeze({
    artifactSha256: sha256(actual),
    byteLength: actual.byteLength,
  });
}

export function parseBrowserProvenance(source, filename = 'browser bundle') {
  const prefix = `/*! ${BROWSER_PROVENANCE_MARKER} `;
  const start = source.indexOf(prefix);
  if (start < 0) throw new Error(`${filename} is missing release provenance.`);
  if (source.indexOf(prefix, start + prefix.length) >= 0) {
    throw new Error(`${filename} contains duplicate release provenance.`);
  }
  const end = source.indexOf(' */', start + prefix.length);
  if (end < 0) throw new Error(`${filename} has unterminated release provenance.`);
  try {
    return JSON.parse(source.slice(start + prefix.length, end));
  } catch (error) {
    throw new Error(`${filename} has invalid JSON release provenance.`, { cause: error });
  }
}

function makeVariable(source, name) {
  const lines = source.split(/\r?\n/);
  for (let index = 0; index < lines.length; index++) {
    const match = lines[index].match(
      new RegExp(`^(?:override\\s+)?${name}\\s*(?::|\\?|\\+)?=\\s*(.*)$`),
    );
    if (!match) continue;
    let value = match[1];
    while (/\\\s*$/.test(value)) {
      value = value.replace(/\\\s*$/, ' ');
      index += 1;
      value += lines[index]?.trim() ?? '';
    }
    return value.trim().split(/\s+/).filter(Boolean);
  }
  return null;
}

function makeAssignmentCount(source, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const global = new RegExp(
    `^(?:override\\s+)?${escaped}\\s*(?::|\\?|\\+)?=`,
  );
  const targetSpecific = new RegExp(
    `^[^#\\t][^=]*:\\s*(?:override\\s+)?${escaped}\\s*(?::|\\?|\\+)?=`,
  );
  return source.split(/\r?\n/)
    .filter((line) => global.test(line) || targetSpecific.test(line))
    .length;
}

function expandedMakeVariable(source, name, parents = new Set()) {
  if (parents.has(name)) throw new Error(`Recursive Makefile variable '${name}'.`);
  const value = makeVariable(source, name);
  if (value === null) return null;
  const nextParents = new Set(parents);
  nextParents.add(name);
  return value.flatMap((token) => {
    const reference = token.match(/^\$\(([A-Za-z0-9_]+)\)$/);
    if (!reference) return [token];
    const expanded = expandedMakeVariable(source, reference[1], nextParents);
    return expanded ?? [token];
  });
}

function normalizedMakeRecipeLines(source) {
  return source.replace(/\\\r?\n[ \t]*/g, ' ')
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/[ \t]+/g, ' '))
    .filter(Boolean);
}

function makeTargetRecipeLines(source, target) {
  const escaped = target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = source.match(new RegExp(
    `^${escaped}:[^\\r\\n]*\\r?\\n((?:\\t[^\\r\\n]*(?:\\r?\\n|$))+)`,
    'm',
  ));
  return match === null ? null : normalizedMakeRecipeLines(match[1]);
}

function sameMembers(actual, expected) {
  return stableJSON([...actual].sort()) === stableJSON([...expected].sort());
}

function containsOrderedSubsequence(actual, expected) {
  let cursor = 0;
  for (const entry of actual) {
    if (entry === expected[cursor]) cursor += 1;
  }
  return cursor === expected.length;
}

export async function validateReleaseDeclarations(repositoryRoot) {
  const errors = [];
  const packageJson = JSON.parse(await fs.readFile(
    path.join(repositoryRoot, 'package.json'),
    'utf8',
  ));
  const releaseRoot = `./dist/${packageJson.version}`;
  const expectedExports = {};
  for (const profile of RELEASE_PROFILES.browser) {
    expectedExports[profile.packageExports.readable] = `${releaseRoot}/${profile.readable}`;
    expectedExports[profile.packageExports.minified] = `${releaseRoot}/${profile.minified}`;
  }
  for (const profile of RELEASE_PROFILES.wasm) {
    expectedExports[`./${profile.filename}`] = `${releaseRoot}/${profile.filename}`;
  }
  for (const wasm of RELEASE_PROFILES.wasm) {
    const expectedConsumers = RELEASE_PROFILES.browser
      .filter(({ sidecar }) => sidecar === wasm.filename)
      .map(({ id }) => id);
    if (!sameMembers(wasm.consumers, expectedConsumers)) {
      errors.push(
        `${wasm.filename} consumers differ from browser sidecar declarations.`,
      );
    }
  }
  for (const browser of RELEASE_PROFILES.browser) {
    if (browser.sidecar !== null &&
        !RELEASE_PROFILES.wasm.some(({ filename }) => filename === browser.sidecar)) {
      errors.push(
        `Browser profile '${browser.id}' names unknown WASM sidecar '${browser.sidecar}'.`,
      );
    }
  }
  if (packageJson.main !== expectedExports['.']) {
    errors.push(`package.json main must be '${expectedExports['.']}'.`);
  }
  if (stableJSON(packageJson.exports) !== stableJSON(expectedExports)) {
    errors.push('package.json exports differ from the central release profiles.');
  }
  const expectedPackageFiles = [`dist/${packageJson.version}/`];
  if (!sameMembers(packageJson.files ?? [], expectedPackageFiles)) {
    errors.push('package.json files differ from the central release layout.');
  }
  if (packageJson.scripts?.['check:release'] !== 'node tools/check_release_artifacts.mjs') {
    errors.push('package.json check:release must remain the read-only artifact gate.');
  }
  if (packageJson.scripts?.['verify:release'] !== 'make verify_release') {
    errors.push('package.json verify:release must invoke the unified Make gate.');
  }
  const expectedSizeScripts = {
    'size:report': 'node tools/report_release_size.mjs report',
    'size:check': 'node tools/report_release_size.mjs check',
    'test:size': 'node --test tests/wasm_internal_abi.test.mjs tests/release_size.test.mjs',
  };
  for (const [name, expected] of Object.entries(expectedSizeScripts)) {
    if (packageJson.scripts?.[name] !== expected) {
      errors.push(`package.json ${name} must be '${expected}'.`);
    }
  }
  /* Every generated projection tsc consumes is verified before tsc runs, so a
   * stale checked-in file fails as a codegen drift rather than as a confusing
   * type error somewhere downstream. */
  const generationChecks = [
    'node tools/generate_wasm_internal_abi.mjs --check',
    'node tools/generate_shader_catalog.mjs --check',
    'node tools/generate_webgpu_dispatch.mjs --check',
  ].join(' && ');
  const typecheck = packageJson.scripts?.typecheck ?? '';
  if (!typecheck.includes(`${generationChecks} && tsc --project tsconfig.json`)) {
    errors.push(
      'package.json typecheck must verify the generated internal WASM ABI, ' +
      'shader catalogue and operator dispatch table before tsc.');
  }
  if (packageJson.scripts?.['test:native-cpu-selection'] !==
      'node native/tests/run_native_cpu_call_sequence_selection_test.mjs') {
    errors.push(
      'package.json test:native-cpu-selection must invoke the native CPU selection gate.',
    );
  }

  const makefile = await fs.readFile(path.join(repositoryRoot, 'Makefile'), 'utf8');
  const inferenceWasm = wasmProfile('inference');
  const fullWasm = wasmProfile('full');
  const expectedMake = {
    WEB_JS_ARTIFACTS: BROWSER_RELEASE_FILENAMES
      .map((filename) => `$(WEB_DIST_DIR)/${filename}`),
    WEB_WASM_ARTIFACTS: WASM_RELEASE_FILENAMES
      .map((filename) => `$(WEB_DIST_DIR)/${filename}`),
    NATIVE_RELEASE_ARTIFACTS: NATIVE_RELEASE_FILENAMES,
    NATIVE_DEBUG_ARTIFACTS,
    NATIVE_DEBUG_EVIDENCE,
    WASM_RELEASE_BUILDER: ['node', 'tools/build_wasm_release.mjs'],
    WASM_RELEASE_OBJECT_ROOT: ['build/wasm/release'],
  };
  const unorderedMakeVariables = new Set([
    'WEB_JS_ARTIFACTS',
    'WEB_WASM_ARTIFACTS',
    'NATIVE_RELEASE_ARTIFACTS',
    'NATIVE_DEBUG_ARTIFACTS',
    'NATIVE_DEBUG_EVIDENCE',
  ]);
  const protectedRecipeVariables = new Set([
    'WASM_RELEASE_BUILDER',
    'WASM_RELEASE_OBJECT_ROOT',
  ]);
  for (const [name, expected] of Object.entries(expectedMake)) {
    const actual = expandedMakeVariable(makefile, name);
    if (actual === null) {
      errors.push(`Makefile is missing ${name}.`);
    } else if (unorderedMakeVariables.has(name)
      ? !sameMembers(actual, expected)
      : stableJSON(actual) !== stableJSON(expected)) {
      errors.push(`Makefile ${name} differs from the central release profiles.`);
    }
  }
  for (const name of protectedRecipeVariables) {
    if (!new RegExp(`^override\\s+${name}\\s*:=`, 'm').test(makefile)) {
      errors.push(`Makefile ${name} must be a non-overridable release projection.`);
    }
    if (makeAssignmentCount(makefile, name) !== 1) {
      errors.push(`Makefile ${name} must have exactly one global release assignment.`);
    }
  }
  const section = makeVariable(makefile, 'WASM_PROVENANCE_SECTION_NAME');
  if (stableJSON(section) !== stableJSON([WASM_PROVENANCE_SECTION])) {
    errors.push('Makefile WASM_PROVENANCE_SECTION_NAME differs from the central release profiles.');
  }
  const recipeLines = new Set(normalizedMakeRecipeLines(makefile));
  const nativeSelectionCommand =
    '$(DOCKER_RUN) npm run test:native-cpu-selection -- ' +
    '--native-build-dir="$(CMAKE_BUILD_DIR)"';
  if (!/^test_native_cpu_selection:\s+build_native\s*$/m.test(makefile) ||
      !recipeLines.has(nativeSelectionCommand) ||
      !/^test_native:\s+test_native_cpu_selection\s+test_native_invariants\s+verify_native_isa\s*$/m.test(makefile) ||
      !/^verify_release:\s+build_web\s+test_native\s*$/m.test(makefile) ||
      !recipeLines.has('$(DOCKER_RUN) npm run check:release')) {
    errors.push(
      'Makefile verify_release must rebuild web/native profiles, run native ' +
      'CPU selection/invariant/ISA tests, then run check:release.',
    );
  }
  /* Every generated projection the WASM build depends on is checked before it
   * compiles: the private JS<->WASM ABI, the shader catalogue that owns
   * `shader_id` and the compressed WGSL block, and the operator dispatch table
   * that names those ids. The catalogue and the table regenerate together
   * because the table's rows are only meaningful against the catalogue's ids. */
  const codegenTargets = [
    ['wasm_abi_codegen', 'wasm_abi_codegen_check',
      ['node tools/generate_wasm_internal_abi.mjs']],
    ['shader_catalog_codegen', 'shader_catalog_codegen_check',
      ['node tools/generate_shader_catalog.mjs',
       'node tools/generate_webgpu_dispatch.mjs']],
  ];
  const codegenConsistent = codegenTargets.every(([generate, verify, commands]) =>
    stableJSON(makeTargetRecipeLines(makefile, generate)) === stableJSON(commands) &&
    stableJSON(makeTargetRecipeLines(makefile, verify)) ===
      stableJSON(commands.map((command) => `${command} --check`)));
  if (!codegenConsistent ||
      !/^build_wasm:\s+build_docker\s+wasm_abi_codegen_check\s+shader_catalog_codegen_check\s*$/m
        .test(makefile)) {
    errors.push(
      'Makefile build_wasm must check the generated internal WASM ABI, shader ' +
      'catalogue and operator dispatch table before compiling.',
    );
  }
  const verifyRecipe = makeTargetRecipeLines(makefile, 'verify_release') ?? [];
  const verifySizeTest = verifyRecipe.indexOf('$(DOCKER_RUN) npm run test:size');
  const verifyArtifactCheck = verifyRecipe.indexOf('$(DOCKER_RUN) npm run check:release');
  const verifySizeReport = verifyRecipe.findIndex((line) => line.includes('npm run size:report'));
  if (verifySizeTest < 0 || verifyArtifactCheck <= verifySizeTest ||
      verifySizeReport <= verifyArtifactCheck) {
    errors.push(
      'Makefile verify_release must run size tests, artifact validation, then the size report.',
    );
  }
  const sizeReportRecipe = makeTargetRecipeLines(makefile, 'size_report') ?? [];
  const sizeCheckRecipe = makeTargetRecipeLines(makefile, 'size_check') ?? [];
  if (!/^size_report:\s+build_docker\s*$/m.test(makefile) ||
      !sizeReportRecipe.some((line) =>
        line.includes('CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:report')) ||
      !/^size_check:\s+build_docker\s*$/m.test(makefile) ||
      !sizeCheckRecipe.includes(
        '$(DOCKER_RUN) env CMAKE_BUILD_DIR="$(CMAKE_BUILD_DIR)" npm run size:check',
      )) {
    errors.push('Makefile size_report/size_check must use the selected native build evidence.');
  }
  if (!/^override\s+CMAKE_EVIDENCE_CONFIG\s*:=\s+-DCMAKE_EXPORT_COMPILE_COMMANDS=ON\s*$/m
    .test(makefile) ||
      [...makefile.matchAll(/cmake -S \. -B \$\(CMAKE_BUILD_DIR\)([^;]*)/g)]
        .some(([, arguments_]) => !arguments_.includes('$(CMAKE_EVIDENCE_CONFIG)'))) {
    errors.push('Every native configure must emit compile_commands.json size evidence.');
  }
  const wasmBuildCommand =
    '$(DOCKER_RUN) $(WASM_RELEASE_BUILDER) all ' +
    '--output-dir $(WEB_DIST_DIR) ' +
    '--build-root $(WASM_RELEASE_OBJECT_ROOT) ' +
    '--evidence-dir $(WASM_PROVENANCE_DIR)';
  const inferenceProvenanceWriteCommand =
    '$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write ' +
    '--artifact $(WEB_DIST_DIR)/volvoxai.lite.wasm ' +
    '--output $(WASM_INFERENCE_PROVENANCE) ' +
    '--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)';
  const fullProvenanceWriteCommand =
    '$(DOCKER_RUN) node tools/write_wasm_provenance.mjs write ' +
    '--artifact $(WEB_DIST_DIR)/volvoxai.wasm ' +
    '--output $(WASM_FULL_PROVENANCE) ' +
    '--build-evidence $(WASM_FULL_BUILD_EVIDENCE)';
  const inferenceProvenanceEmbedCommand =
    '$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py ' +
    '--section-name $(WASM_PROVENANCE_SECTION_NAME) ' +
    '--payload $(WASM_INFERENCE_PROVENANCE) ' +
    '--input $(WEB_DIST_DIR)/volvoxai.lite.wasm ' +
    '--output $(WEB_DIST_DIR)/volvoxai.lite.wasm';
  const fullProvenanceEmbedCommand =
    '$(DOCKER_RUN) $(PY) tools/embed_wasm_custom_section.py ' +
    '--section-name $(WASM_PROVENANCE_SECTION_NAME) ' +
    '--payload $(WASM_FULL_PROVENANCE) ' +
    '--input $(WEB_DIST_DIR)/volvoxai.wasm ' +
    '--output $(WEB_DIST_DIR)/volvoxai.wasm';
  const inferenceProvenanceCheckCommand =
    '$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check ' +
    '--artifact $(WEB_DIST_DIR)/volvoxai.lite.wasm ' +
    '--build-evidence $(WASM_INFERENCE_BUILD_EVIDENCE)';
  const fullProvenanceCheckCommand =
    '$(DOCKER_RUN) node tools/write_wasm_provenance.mjs check ' +
    '--artifact $(WEB_DIST_DIR)/volvoxai.wasm ' +
    '--build-evidence $(WASM_FULL_BUILD_EVIDENCE)';
  const buildWasmRecipe = makeTargetRecipeLines(makefile, 'build_wasm') ?? [];
  const criticalBuildWasmSequence = [
    wasmBuildCommand,
    inferenceProvenanceWriteCommand,
    fullProvenanceWriteCommand,
    inferenceProvenanceEmbedCommand,
    fullProvenanceEmbedCommand,
    inferenceProvenanceCheckCommand,
    fullProvenanceCheckCommand,
  ];
  if (!containsOrderedSubsequence(buildWasmRecipe, criticalBuildWasmSequence) ||
      criticalBuildWasmSequence.some((command_) =>
        buildWasmRecipe.filter((line) => line === command_).length !== 1)) {
    errors.push(
      'Makefile WASM targets must run the central C object build, ' +
      'provenance binding, and final checks exactly once in release order.',
    );
  }
  for (const obsolete of [
    'WASM_INFERENCE_SOURCES',
    'WASM_LIBM_SOURCES',
    'WASM_FULL_SOURCES',
    'WASM_RELAXED_SOURCES',
    'WASM_PTQ_AUTHORING_SOURCES',
    'WASM_BASELINE_FLAGS',
    'WASM_FULL_FLAGS',
    'WASM_RELAXED_FLAGS',
    'WASM_PTQ_AUTHORING_FLAGS',
    'WASM_RELEASE_CC',
    'WASM_RELEASE_RELAXED_CC',
  ]) {
    if (makeAssignmentCount(makefile, obsolete) !== 0) {
      errors.push(`Makefile must not duplicate central WASM recipe variable ${obsolete}.`);
    }
  }
  const abiProjections = [
    {
      file: 'native/src/kernels/wasm_heap_arena.c',
      expression: /#define\s+VOLVOXAI_WASM_RUNTIME_ABI_VERSION\s+(\d+)u/,
      expected: inferenceWasm.abiVersions.runtime,
      label: 'C runtime',
    },
    {
      file: 'native/src/runtime/scheduler_policy.h',
      expression: /#define\s+VX_SCHEDULER_POLICY_ABI_VERSION\s+(\d+)u/,
      expected: inferenceWasm.abiVersions.scheduler,
      label: 'C scheduler policy',
    },
    {
      file: 'native/src/kernels/training_kernels.h',
      expression: /#define\s+VOLVOXAI_TRAINING_ABI_VERSION\s+(\d+)u/,
      expected: fullWasm.abiVersions.training,
      label: 'C training',
    },
    {
      file: 'native/src/training/training_core.h',
      expression: /#define\s+VOLVOXAI_PTQ_ABI_VERSION\s+(\d+)u/,
      expected: fullWasm.abiVersions.ptq,
      label: 'C PTQ',
    },
  ];
  for (const projection of abiProjections) {
    let source;
    try {
      source = await fs.readFile(path.join(repositoryRoot, projection.file), 'utf8');
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      errors.push(`Missing ${projection.label} ABI projection ${projection.file}.`);
      continue;
    }
    const match = source.match(projection.expression);
    if (!match || Number(match[1]) !== projection.expected) {
      errors.push(
        `${projection.label} ABI projection in ${projection.file} must equal ` +
        `${projection.expected}.`,
      );
    }
  }
  return Object.freeze({ packageJson, errors: Object.freeze(errors) });
}
