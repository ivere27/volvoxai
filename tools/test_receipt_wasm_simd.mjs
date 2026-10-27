#!/usr/bin/env node

// Reuse the native PackedLinear and packed Conv oracles in actual WebAssembly.
// Run with the repository's pinned Docker toolchain (make build_docker first):
//   docker run --rm -v "$PWD":/workspace -w /workspace volvoxai-build:latest \
//     node tools/test_receipt_wasm_simd.mjs
// No release artifact is replaced. Generated adapters, source/toolchain build
// evidence and results remain in a unique build/wasm/receipt-simd-tests/run-*.
// PackedLinear runs with and without SIMD128; Conv runs with SIMD128 against
// its existing independent scalar integer oracle. This is not a benchmark.

import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildWasmComponent } from './build_wasm_release.mjs';
import { wasmKernelFixtureRecipe } from './build_wasm_kernel_fixture.mjs';
import { sha256 } from './release_profiles.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LINEAR_TESTS = [
  'test_packed_linear_i3_unpack',
  'test_packed_linear_i8',
  'test_packed_linear_i8_tiles',
  'test_packed_linear_exact_bounds',
  'test_packed_linear_lowbit_arithmetic',
  'test_packed_linear_row_tiles',
];
const LIBC_SOURCES = [
  'native/src/runtime/wasm_libc.c',
  'native/src/kernels/wasm_heap_arena.c',
];

function recipe(id, sources, exports, simd) {
  const base = wasmKernelFixtureRecipe(ROOT);
  const parent = base.recipe.parent;
  const kernel = parent.objects.find(({ id }) => id === 'kernels');
  assert.ok(kernel, 'release numerical kernel recipe must exist');
  const flags = [
    // Machine assembly makes the SIMD instruction audit inspectable; -flto
    // would make Clang's -S audit emit LLVM IR instead. These test-only
    // objects otherwise retain the shared numerical compilation policy.
    ...kernel.flags.filter((flag) => flag !== '-flto' && (simd || flag !== '-msimd128')),
    '-ffp-contract=off',
  ];
  return {
    ...base,
    id,
    filename: `${id}.wasm`,
    recipe: {
      ...base.recipe,
      parent: {
        ...parent,
        objects: sources.map((source, index) => ({
          ...kernel, id: `test-${index}`, source, flags,
        })),
        linkFlags: [
          ...parent.linkFlags.filter((flag) => !flag.startsWith('--export')),
          '--export-memory', '--initial-memory=33554432',
          ...exports.map((name) => `--export=${name}`),
        ],
        optimizationFlags: parent.optimizationFlags.filter((flag) =>
          simd || flag !== '--enable-simd'),
        requiredInstructions: simd ? ['i32x4.dot_i16x8_s'] : [],
      },
    },
  };
}

async function main() {
  if (process.argv.length !== 2) {
    throw new Error('usage: node tools/test_receipt_wasm_simd.mjs (inside the pinned build image)');
  }
  const parent = path.join(ROOT, 'build/wasm/receipt-simd-tests');
  await mkdir(parent, { recursive: true });
  const directory = await mkdtemp(path.join(parent, 'run-'));
  const relative = (file) => path.relative(ROOT, file).split(path.sep).join('/');
  const wrapper = path.join(directory, 'conv-test.c');
  const include = path.relative(directory, path.join(ROOT, 'native/tests/test_qconv4.c'))
    .split(path.sep).join('/');
  // Include the original test without rewriting its cases, oracle or CHECK.
  // Only its CLI entry/logging/exit are adapted for a freestanding test module.
  await writeFile(wrapper, `#include <stdio.h>
#include <stdlib.h>
#include "qconv4_ops.inc"
static int receipt_failure_line;
int get_failure_line(void) { return receipt_failure_line; }
static int receipt_test_log(const char* format, ...) { (void)format; return 0; }
#define printf receipt_test_log
#define fprintf(stream, ...) receipt_test_log(__VA_ARGS__)
#define exit(code) do { (void)(code); receipt_failure_line=__LINE__; __builtin_trap(); } while (0)
#define main test_packed_conv
#include "${include}"
`);
  const cases = [
    { id: 'linear-scalar', simd: false, tests: LINEAR_TESTS,
      sources: ['native/tests/test_packed_linear_i8.c', ...LIBC_SOURCES] },
    { id: 'linear-simd', simd: true, tests: LINEAR_TESTS,
      sources: ['native/tests/test_packed_linear_i8.c', ...LIBC_SOURCES] },
    { id: 'conv-simd', simd: true, tests: ['test_packed_conv'],
      sources: [relative(wrapper), 'native/src/kernels/packed_quant_gemm.c', ...LIBC_SOURCES] },
  ];
  const report = { format: 'volvox-receipt-wasm-numerical/v1', status: 'running',
    node: process.version, cases: [] };
  try {
    for (const item of cases) {
      const exports = [...item.tests, ...(item.id === 'conv-simd' ? ['get_failure_line'] : [])];
      const result = await buildWasmComponent({
        repositoryRoot: ROOT, profile: recipe(item.id, item.sources, exports, item.simd),
        kind: 'parent', output: relative(path.join(directory, `${item.id}.wasm`)),
        buildRoot: relative(path.join(directory, item.id)),
        evidence: relative(path.join(directory, `${item.id}.build.json`)),
      });
      const module = await WebAssembly.compile(result.artifactContents);
      assert.deepEqual(WebAssembly.Module.imports(module), [], `${item.id}: unexpected host imports`);
      const instance = await WebAssembly.instantiate(module);
      const entry = { id: item.id, wasm_sha256: sha256(result.artifactContents), tests: {} };
      report.cases.push(entry);
      for (const name of item.tests) {
        try {
          const code = instance.exports[name]();
          entry.tests[name] = code;
          assert.equal(code, 0, `${item.id}/${name}: failed at native test line ${code}`);
        } catch (error) {
          const line = instance.exports.get_failure_line?.();
          throw new Error(`${item.id}/${name}${line ? ` at native test line ${line}` : ''}: ${error}`, { cause: error });
        }
      }
      console.log(`${item.id}: ${item.tests.length} numerical test entries passed`);
    }
    report.status = 'passed';
  } catch (error) {
    report.status = 'failed'; report.error = String(error); throw error;
  } finally {
    await writeFile(path.join(directory, 'results.json'), `${JSON.stringify(report, null, 2)}\n`);
    console.log(`Numerical evidence: ${relative(path.join(directory, 'results.json'))}`);
  }
}

await main();
