import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { runChrome } from '../tools/llm_baseline/verify_packed_webgpu.mjs';

test('PackedLinear shared I8 and admitted MR groups preserve row arithmetic and partial writes', {
  skip: !process.env.VOLVOXAI_TEST_CHROME, timeout: 120000,
}, async () => {
  const candidate = await readFile(new URL('../shaders/inference/packedLinear.wgsl', import.meta.url), 'utf8');
  // Independent F32/integer oracle; software adapters are permitted for this
  // operator unit test. Physical performance qualification is a separate run.
  const result = await runChrome({ sources: { candidate }, requirePhysical: false }, process.env.VOLVOXAI_TEST_CHROME, true);
  assert.equal(result.passed, true);
  assert.equal(result.numerical.length, 159);
  assert.ok(result.numerical.some(item => item.M === 63 && item.N === 2048));
  assert.ok(result.numerical.some(item => item.M === 64 && item.N === 1536));
  assert.ok(result.numerical.some(item => item.M === 97 && item.N === 1024));
  assert.ok(result.numerical.some(item => item.first === 1024 && item.pattern === 'mixed-nonfinite'));
  assert.ok(result.numerical.every(item => Number.isFinite(item.maximum_absolute_error)));
});
