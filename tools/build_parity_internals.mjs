// Bundle the paged-KV GPU harness entry point.
//
// Separate from tools/build_web.mjs because this is not a release artifact:
// it is a test-only bundle that re-exports backend internals so a physical-GPU
// campaign can drive a page table the public API has no way to express. Same
// esbuild configuration as the shipped bundles, so the harness runs the code
// the release runs.
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/* No top-level await: a stale Node on a GPU host reports it as a bare syntax
 * error, which hides the real problem (esbuild is a dev dependency and is not
 * installed there). Build the bundle where the dev dependencies live. */
build({
  absWorkingDir: repositoryRoot,
  entryPoints: ['tests/parity/kvcache/paged_kv_internals.ts'],
  outfile: 'tests/parity/kvcache/out/paged_kv_internals.js',
  bundle: true,
  format: 'esm',
  target: ['es2022'],
  loader: { '.wgsl': 'text' },
  logLevel: 'info',
}).catch((error) => {
  console.error(error);
  process.exit(1);
});
