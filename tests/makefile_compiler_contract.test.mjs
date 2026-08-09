import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));

test('native ISA verification inherits the configured WASM compiler', async () => {
  const compiler = 'volvoxai-clang-contract-sentinel';
  const { stdout } = await run('make', [
    '--no-print-directory',
    '--dry-run',
    'verify_native_isa',
    `WASM_CC=${compiler}`,
  ], { cwd: repositoryRoot });

  const commands = stdout.replace(/\\\s*\n\s*/g, ' ');
  assert.match(
    commands,
    new RegExp(`\\benv CLANG="${compiler}"\\s+ctest\\s+--test-dir[^\n]+-L verify\\b`),
  );
});
