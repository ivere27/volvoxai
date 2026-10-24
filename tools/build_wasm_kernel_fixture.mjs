#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildWasmComponent } from './build_wasm_release.mjs';
import { wasmProfile } from './release_profiles.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const WASM_KERNEL_FIXTURE_PATH = 'build/wasm/kernel-fixture/kernels.wasm';

// These are developer test roots, never part of the release ABI. Keep scalar
// references available for byte comparisons with the selected SIMD128 routes.
export const WASM_KERNEL_FIXTURE_EXPORTS = Object.freeze([
  'memory',
  'wasm_runtime_abi_version',
  'alloc_bytes',
  'reset_heap',
  'vx_packed_q8_weight_size',
  'vx_pack_q8_weight',
  'qlinear_i8u8',
  'vx_qlinear_i8u8_packed',
  'qbatch_matmul_i8u8',
  'vx_qbatch_matmul_i8u8_simd128',
  'qsdpa_i8u8',
  'qconv2d_i8u8',
  'qconv2d_im2col_i8u8',
]);

export function wasmKernelFixtureRecipe(repositoryRoot = ROOT) {
  const inference = wasmProfile('inference', repositoryRoot);
  const parent = inference.recipe.parent;
  return Object.freeze({
    id: 'kernel-fixture',
    filename: 'kernels.wasm',
    forbiddenDependencyPrefixes: inference.forbiddenDependencyPrefixes,
    forbiddenDependencyFiles: inference.forbiddenDependencyFiles,
    forbiddenDependencyPatterns: inference.forbiddenDependencyPatterns,
    recipe: Object.freeze({
      ...inference.recipe,
      parent: Object.freeze({
        ...parent,
        // Reuse the exact release compilation and optimization policy. The
        // numerical translation unit already supplies its small heap helpers.
        objects: Object.freeze(parent.objects.filter(
          ({ category }) => category === 'numerical-kernel')),
        linkFlags: Object.freeze([
          // Unlike the runtime, this fixture has no host imports. Missing
          // exports or accidental libc dependencies must fail at link time.
          ...parent.linkFlags.filter((flag) =>
            !flag.startsWith('--export') && flag !== '--allow-undefined'),
          ...WASM_KERNEL_FIXTURE_EXPORTS.map((name) =>
            name === 'memory' ? '--export-memory' : `--export=${name}`),
        ]),
      }),
      customSections: Object.freeze([]),
    }),
  });
}

export function assertWasmKernelFixture(module, label = 'WASM kernel fixture') {
  const actual = WebAssembly.Module.exports(module)
    .map(({ name, kind }) => `${name}:${kind}`).sort();
  const expected = WASM_KERNEL_FIXTURE_EXPORTS
    .map((name) => `${name}:${name === 'memory' ? 'memory' : 'function'}`).sort();
  assert.deepEqual(actual, expected,
    `${label}: use the developer fixture built by 'make build_wasm_kernel_fixture'; ` +
    'release WASM exposes the generated API, not direct kernel calls');
  assert.deepEqual(WebAssembly.Module.imports(module), [],
    `${label}: kernel fixture must be self-contained`);
}

export async function loadWasmKernelFixture(artifactPath) {
  const module = await WebAssembly.compile(await readFile(artifactPath));
  assertWasmKernelFixture(module, artifactPath);
  return WebAssembly.instantiate(module);
}

export async function buildWasmKernelFixture(repositoryRoot = ROOT) {
  const result = await buildWasmComponent({
    repositoryRoot,
    profile: wasmKernelFixtureRecipe(repositoryRoot),
    kind: 'parent',
    output: WASM_KERNEL_FIXTURE_PATH,
    buildRoot: 'build/wasm/kernel-fixture',
    evidence: 'build/wasm/kernel-fixture/kernels.build.json',
  });
  assertWasmKernelFixture(new WebAssembly.Module(result.artifactContents));
  return result;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length !== 2) {
    throw new Error('usage: node tools/build_wasm_kernel_fixture.mjs');
  }
  const result = await buildWasmKernelFixture();
  console.log(`Built developer kernel fixture: ${WASM_KERNEL_FIXTURE_PATH} ` +
    `(${result.artifactContents.byteLength} bytes; excluded from releases).`);
}
