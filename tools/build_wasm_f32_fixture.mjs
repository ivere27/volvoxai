#!/usr/bin/env node
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildWasmComponent} from './build_wasm_release.mjs';
import {wasmKernelFixtureRecipe} from './build_wasm_kernel_fixture.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputRoot = process.argv[2] ?? 'build/wasm/f32-fixture';
const base = wasmKernelFixtureRecipe(root);
const parent = base.recipe.parent;
// Root calloc too: LLVM can introduce that libcall after LTO dead stripping
// in this deliberately small libc fixture. The release control TU already
// retains it through its ordinary object/runtime construction paths.
const exports = ['memory', 'calloc', 'test_f32_alloc', 'test_f32_free', 'test_f32_prepare',
  'test_f32_find', 'test_f32_clear', 'test_f32_bytes', 'test_f32_live_bytes',
  'test_f32_fail_after', 'test_f32_bound_add',
  'matmul_f32', 'matmul_f32_out_in', 'vx_gemm_f32_run_packed',
  'lstm_f32', 'lstm_f32_packed'];
const profile = {...base, id: 'f32-fixture', filename: 'f32.wasm', recipe: {
  ...base.recipe, parent: {...parent,
    objects: [...parent.objects, ...[
      ['libc', 'native/src/runtime/wasm_libc.c'],
      ['prepared-weights', 'native/src/runtime/compiled_weight_resources.c'],
      ['test-roots', 'tools/tests/wasm_f32_fixture.c'],
    ].map(([id, source]) => ({id, source, category: 'numerical-kernel',
      optimization: id === 'libc' ? '-Oz' : '-O3', flags: [
        ...parent.objects[0].flags,
        ...(id === 'prepared-weights' ? ['-Dmalloc=test_f32_pack_malloc',
          '-Dcalloc=test_f32_pack_calloc'] : []),
      ]}))],
    linkFlags: [...parent.linkFlags.filter(flag => !flag.startsWith('--export')),
      ...exports.map(name => name === 'memory' ? '--export-memory' : `--export=${name}`)],
  },
}};
const result = await buildWasmComponent({repositoryRoot: root, profile, kind: 'parent',
  output: `${outputRoot}/f32.wasm`, buildRoot: outputRoot,
  evidence: `${outputRoot}/f32.build.json`});
console.log(`Built ${result.artifactContents.byteLength} bytes at ${outputRoot}/f32.wasm`);
