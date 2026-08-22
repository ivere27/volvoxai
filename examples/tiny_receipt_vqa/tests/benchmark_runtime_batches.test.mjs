import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_MAX_BATCH_DELAY_MS,
  parseArguments,
} from '../tools/benchmark_runtime_batches.mjs';

const REQUIRED = [
  '--package', 'package', '--role', 'encoder', '--inputs', 'inputs.json',
  '--output-directory', 'outputs', '--backend', 'cpu-js',
];

test('component-batch benchmark retains and exposes its historical delay default', () => {
  const options = parseArguments(REQUIRED);
  assert.equal(DEFAULT_MAX_BATCH_DELAY_MS, 10);
  assert.equal(options.maxBatchDelayMs, 10);
});

test('component-batch benchmark accepts an explicit zero-delay scheduler', () => {
  const options = parseArguments([...REQUIRED, '--max-batch-delay-ms', '0']);
  assert.equal(options.maxBatchDelayMs, 0);
});

test('component-batch benchmark rejects an invalid scheduler delay', () => {
  for (const value of ['-1', 'NaN', '60001']) {
    assert.throws(
      () => parseArguments([...REQUIRED, '--max-batch-delay-ms', value]),
      /--max-batch-delay-ms/,
    );
  }
});

test('machine paths and output digests require an explicitly private option', () => {
  assert.equal(parseArguments(REQUIRED).includePrivateArtifacts, false);
  assert.equal(
    parseArguments([...REQUIRED, '--include-private-artifacts']).includePrivateArtifacts,
    true,
  );
});
