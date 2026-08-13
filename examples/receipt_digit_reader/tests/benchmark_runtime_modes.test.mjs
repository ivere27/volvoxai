import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  DEFAULT_MAX_BATCH_DELAY_MS,
  digestPackageArtifacts,
  parseArguments,
  summarizePublicLanes,
} from '../tools/benchmark_runtime_modes.mjs';

const REQUIRED = [
  '--package', 'package', '--raw', 'lane0.f32', '--backend', 'cpu-js',
];

test('runtime-mode benchmark retains and exposes its historical delay default', () => {
  const options = parseArguments(REQUIRED);
  assert.equal(DEFAULT_MAX_BATCH_DELAY_MS, 1);
  assert.equal(options.maxBatchDelayMs, 1);
});

test('runtime-mode benchmark accepts an explicit zero-delay scheduler', () => {
  const options = parseArguments([...REQUIRED, '--max-batch-delay-ms', '0']);
  assert.equal(options.maxBatchDelayMs, 0);
});

test('runtime-mode benchmark rejects an invalid scheduler delay', () => {
  for (const value of ['-1', 'Infinity', '60001']) {
    assert.throws(
      () => parseArguments([...REQUIRED, '--max-batch-delay-ms', value]),
      /--max-batch-delay-ms/,
    );
  }
});

test('public lane summaries contain parity but no receipt values or fingerprints', () => {
  const summary = summarizePublicLanes(
    [new Set(['private decoded value'])],
    [new Set(['private output digest'])],
    [{ close: true, exact: true }],
    [true],
  );
  assert.deepEqual(summary, [{
    laneId: 'lane-01',
    decodedRecordStableAcrossMeasuredGroups: true,
    outputStableAcrossMeasuredGroups: true,
    outputMatchesIndependentReference: true,
    outputExactlyMatchesIndependentReference: true,
    decodedRecordMatchesIndependentReference: true,
  }]);
  assert.doesNotMatch(JSON.stringify(summary), /private/);
});

test('private logits require an explicitly private CLI option', () => {
  const options = parseArguments([...REQUIRED, '--include-private-logits']);
  assert.equal(options.includePrivateLogits, true);
  assert.throws(() => parseArguments([...REQUIRED, '--include-logits']), /unknown option/);
});

test('package artifact proof hashes fixed payloads without publishing their path', () => {
  const directory = mkdtempSync(join(tmpdir(), 'private-receipt-package-'));
  const payloads = new Map([
    ['manifest.json', 'manifest'],
    ['graph.json', 'graph'],
    ['model.safetensors', 'weights'],
  ]);
  try {
    for (const [filename, payload] of payloads) writeFileSync(join(directory, filename), payload);
    const hashes = digestPackageArtifacts(directory);
    assert.deepEqual(hashes, Object.fromEntries([...payloads].map(([filename, payload]) => [
      filename, createHash('sha256').update(payload).digest('hex'),
    ])));
    assert.doesNotMatch(JSON.stringify(hashes), new RegExp(directory));
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
