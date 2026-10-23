import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runChrome } from '../tools/llm_baseline/verify_packed_webgpu.mjs';

test('PackedLinear bounded shared I8 groups preserve row arithmetic and partial writes', {
  skip: !process.env.VOLVOXAI_TEST_CHROME, timeout: 120000,
}, async () => {
  const candidate = await readFile(new URL('../shaders/inference/packedLinear.wgsl', import.meta.url), 'utf8');
  // Independent F32/integer oracle; software adapters are permitted for this
  // operator unit test. Physical performance qualification is a separate run.
  const result = await runChrome({ sources: { candidate }, requirePhysical: false }, process.env.VOLVOXAI_TEST_CHROME, true);
  assert.equal(result.passed, true);
  assert.equal(result.numerical.length, 132);
  assert.ok(result.numerical.every(item => Number.isFinite(item.maximum_absolute_error)));
});
