import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function latencyReference(t, medianMs, recordingRuns = 1) {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'volvox-runtime-baseline-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const filename = path.join(directory, 'reference.json');
  writeFileSync(filename, JSON.stringify({
    schema: 'volvoxai.runtime-latency-reference/v1',
    recordedAt: '2026-07-21T00:00:00Z',
    environment: {
      nodeMajor: Number(process.versions.node.split('.')[0]),
      platform: process.platform,
      architecture: process.arch,
    },
    workload: {
      operation: 'Identity', dtype: 'float32', width: 32,
      warmupExecutions: 1, measuredExecutions: 3,
    },
    latency: { medianMs, maximumRegressionPercent: 5, recordingRuns },
  }));
  return filename;
}

test('portable runtime baseline emits bounded CPU lifecycle evidence', (t) => {
  const reference = latencyReference(t, 1_000, 2);
  const stdout = execFileSync(process.execPath, [
    '--expose-gc',
    '--import',
    'tsx',
    'tools/runtime_baseline.mjs',
    '--samples=3',
    '--warmup=1',
    '--width=32',
    `--reference=${reference}`,
  ], { cwd: ROOT, encoding: 'utf8' });
  const evidence = JSON.parse(stdout);

  assert.equal(evidence.schema, 'volvoxai.runtime-baseline/v1');
  assert.equal(evidence.environment.backend, 'cpu');
  assert.equal(evidence.environment.explicitGc, true);
  assert.deepEqual(evidence.workload.shape, [1, 32]);
  assert.equal(evidence.workload.measurementRuns, 2);
  assert.equal(evidence.latency.samplesMs.length, 6);
  assert.equal(evidence.latency.runMediansMs.length, 2);
  assert.ok(evidence.latency.samplesMs.every((value) => Number.isFinite(value) && value >= 0));
  assert.ok(Number.isFinite(evidence.latency.medianMs) && evidence.latency.medianMs >= 0);
  assert.equal(evidence.latency.gate.referenceSchema, 'volvoxai.runtime-latency-reference/v1');
  assert.equal(evidence.latency.gate.maximumRegressionPercent, 5);
  assert.equal(evidence.latency.gate.passed, true);
  assert.equal(evidence.snapshot.outputBytes, 128);
  assert.equal(evidence.snapshot.freshCallerOwnedReads, true);
  assert.equal(evidence.snapshot.readableAfterContextClose, true);
  assert.deepEqual(evidence.disposal, {
    acceptedWorkDrained: true,
    contextCloseIsIdempotent: true,
    resultCloseIsIdempotent: true,
    allOwnersClosed: true,
  });
  assert.equal(evidence.runtime.compilation.selectedBackend, 'cpu');
  assert.equal(evidence.runtime.execution.backend, 'cpu');
  assert.match(evidence.runtime.execution.contextId, /^context-/);
  assert.equal(evidence.budgets.twoIdleContextsToOneMutableStorageRatio, 2.2);
  const idleOneBytes = evidence.memory.deltasFromRetainedBaseline.oneIdleContext.arrayBufferBytes;
  const idleTwoBytes = evidence.memory.deltasFromRetainedBaseline.twoIdleContexts.arrayBufferBytes;
  assert.ok(idleOneBytes > 0);
  assert.ok(idleTwoBytes <= idleOneBytes * evidence.budgets.twoIdleContextsToOneMutableStorageRatio);
  assert.ok(
    evidence.memory.stableResultDeltas.allocated.arrayBufferBytes <=
      evidence.snapshot.alignedOutputBytes *
      (1 + evidence.budgets.stableSnapshotBookkeepingPercent / 100),
  );
  assert.ok(evidence.memory.deltasFromRetainedBaseline.idleContextsClosed.arrayBufferBytes <= 0);
  assert.ok(evidence.memory.deltasFromRetainedBaseline.allOwnersClosed.arrayBufferBytes <= 0);
  for (const stage of [
    'retainedBaseline', 'oneIdleContext', 'twoIdleContexts', 'idleContextsClosed',
    'beforeStableResult', 'stableResultAllocated', 'stableResultWithCallerReads',
    'stableResultAfterContextClose',
    'stableResultClosed', 'allOwnersClosed',
  ]) {
    assert.ok(Number.isSafeInteger(evidence.memory[stage].heapUsedBytes));
    assert.ok(Number.isSafeInteger(evidence.memory[stage].arrayBufferBytes));
  }
});

test('portable runtime baseline fails when the committed median budget is exceeded', (t) => {
  const reference = latencyReference(t, 0.000001);
  const result = spawnSync(process.execPath, [
    '--expose-gc',
    '--import',
    'tsx',
    'tools/runtime_baseline.mjs',
    '--samples=3',
    '--warmup=1',
    '--width=32',
    `--reference=${reference}`,
  ], { cwd: ROOT, encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /exceeds the committed .* budget/);
  const evidence = JSON.parse(result.stdout);
  assert.equal(evidence.latency.gate.passed, false);
  assert.ok(evidence.latency.medianMs > evidence.latency.gate.maximumMedianMs);
});
