import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('WebGPU baseline exercises the dynamic-v1 public lifecycle', () => {
  const source = readFileSync(
    path.join(ROOT, 'tools', 'webgpu_runtime_baseline.html'),
    'utf8',
  );

  assert.match(source, /volvoxai\.webgpu-runtime-baseline\/v2/);
  assert.match(source, /Model/);
  assert.match(source, /runtime\.compile\(snapshot/);
  assert.match(source, /shape: Object\.freeze\(\[1, length\]\)/);
  assert.doesNotMatch(source, /new Graph\(/);
  assert.doesNotMatch(source, /createModel\(/);
});

test('physical WebGPU operator harness authors only redesigned logical snapshots', () => {
  const source = readFileSync(
    path.join(ROOT, 'tools', 'webgpu_op_tests.html'),
    'utf8',
  );

  assert.match(source, /Model/);
  assert.match(source, /parseGraphDocument/);
  assert.match(source, /runtime\.compile\(model\.snapshot/);
  assert.match(source, /shape: descriptor\.shape/);
  assert.doesNotMatch(source, /\bnew Graph\s*\(/);
  assert.doesNotMatch(source, /\bcreateModel\s*\(/);
  assert.doesNotMatch(source, /\bbuildGraph\b/);
});

test('native dynamic baseline uses only the public shaped execution ABI', () => {
  const runner = readFileSync(
    path.join(ROOT, 'tools', 'native_dynamic_shape_performance.mjs'),
    'utf8',
  );
  const helper = readFileSync(
    path.join(ROOT, 'tools', 'native_dynamic_shape_performance.c'),
    'utf8',
  );

  assert.match(runner, /volvoxai\.native-dynamic-shape-performance\/v1/);
  assert.doesNotMatch(runner, /shape_system/);
  assert.match(runner, /independentlyCompiledPaddedMaximum/);
  assert.match(helper, /vx_execution_context_execute/);
  assert.match(helper, /shape_plan=/);
  assert.match(helper, /arena_high_water=/);
  assert.doesNotMatch(helper, /vx_public_api_test_/);
});

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
  assert.ok(idleOneBytes >= 0);
  assert.ok(idleTwoBytes <= idleOneBytes * evidence.budgets.twoIdleContextsToOneMutableStorageRatio);
  assert.ok(
    evidence.memory.stableResultDeltas.allocated.arrayBufferBytes <=
    evidence.snapshot.alignedOutputBytes *
      (1 + evidence.budgets.stableSnapshotBookkeepingPercent / 100),
  );
  assert.equal(evidence.budgets.idleContextAllocation, 'lazy');
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

test('padded-static baseline records batch, sequence, and spatial parity', () => {
  const stdout = execFileSync(process.execPath, [
    '--import',
    'tsx',
    'tools/padded_static_baseline.mjs',
    '--samples=5',
    '--warmup=1',
  ], { cwd: ROOT, encoding: 'utf8' });
  const evidence = JSON.parse(stdout);

  assert.equal(evidence.schema, 'volvoxai.padded-static-baseline/v1');
  assert.deepEqual(evidence.protocol, {
    backend: 'cpu', samples: 5, warmup: 1, order: 'alternating',
  });
  assert.deepEqual(evidence.results.map((result) => result.name), [
    'batch-linear', 'sequence-linear', 'spatial-conv2d',
  ]);
  assert.deepEqual(evidence.results.map((result) => result.paddedToActiveElementRatio), [8, 8, 9]);
  for (const result of evidence.results) {
    assert.ok(result.paddedLogicalInputBytes > result.activeLogicalInputBytes);
    assert.ok(Number.isFinite(result.active.executionP50Ms));
    assert.ok(Number.isFinite(result.active.executionP95Ms));
    assert.ok(Number.isFinite(result.padded.executionP50Ms));
    assert.ok(Number.isFinite(result.padded.executionP95Ms));
    assert.equal(result.activeRegionParity.equalLength, true);
    assert.equal(result.activeRegionParity.maximumAbsoluteError, 0);
    assert.equal(result.activeRegionParity.pass, true);
  }
});

test('dynamic-shape performance protocol separates cold, warm, alternating, and adversarial evidence', () => {
  const stdout = execFileSync(process.execPath, [
    '--import',
    'tsx',
    'tools/dynamic_shape_performance.mjs',
    '--backend=cpu',
    '--samples=3',
    '--warmup=0',
  ], { cwd: ROOT, encoding: 'utf8' });
  const evidence = JSON.parse(stdout);

  assert.equal(evidence.schema, 'volvoxai.dynamic-shape-performance/v1');
  assert.equal(evidence.environment.backend, 'cpu');
  assert.deepEqual(evidence.results.map((result) => result.name), [
    'batch-linear', 'sequence-linear', 'spatial-conv2d', 'multi-input-exact-add',
  ]);
  for (const result of evidence.results) {
    assert.equal(result.warm.dynamicActive.samples, 3);
    assert.equal(result.alternating.samples, 3);
    assert.equal(result.adversarial.signatureCount, 12);
    assert.ok(result.logicalInputBytes.padded > result.logicalInputBytes.active);
    assert.ok(Number.isFinite(result.compile.dynamicWallTimeMs));
    assert.ok(Number.isFinite(result.coldSpecialization.active.shapeBindTimeMs));
    assert.ok(Number.isFinite(result.coldSpecialization.active.providerTimeMs));
    assert.equal(result.coldSpecialization.active.specializationCacheHit, false);
    assert.equal(result.parity.dynamicActiveMaximumAbsoluteError, 0);
    assert.equal(result.parity.dynamicPaddedMaximumAbsoluteError, 0);
    assert.equal(result.parity.paddedActiveRegionMaximumAbsoluteError, 0);
    assert.ok(result.adversarial.finalTelemetry.specializationCacheEntries <= 8);
    assert.ok(result.adversarial.finalTelemetry.specializationCacheEvictions >= 1);
  }
});

test('CPU shape-context baseline exposes a static fast path and exact snapshots', () => {
  const stdout = execFileSync(process.execPath, [
    '--import',
    'tsx',
    'tools/cpu_shape_context_baseline.mjs',
    '--samples=3',
    '--warmup=1',
    '--width=32',
    '--runs=1',
    '--gate=off',
  ], { cwd: ROOT, encoding: 'utf8' });
  const evidence = JSON.parse(stdout);

  assert.equal(evidence.schema, 'volvoxai.cpu-shape-context-baseline/v1');
  assert.equal(evidence.environment.backend, 'cpu-shape-v1');
  assert.deepEqual(evidence.workload.shape, [1, 32]);
  assert.equal(evidence.latency.samplesMs.length, 3);
  assert.equal(evidence.latency.gate, null);
  assert.equal(evidence.parity.byteExactIdentity, true);
  assert.equal(evidence.parity.stableReadableAfterClose, true);
  assert.equal(evidence.lifecycle.staticFastPath, true);
  assert.equal(evidence.lifecycle.planCacheEntries, 0);
  assert.equal(evidence.lifecycle.capacityBytes, 256);
  assert.equal(evidence.lifecycle.capacityHighWaterBytes, 256);
  assert.equal(evidence.lifecycle.growEvents, 1);
  assert.equal(evidence.lifecycle.afterCloseCapacityBytes, 0);
});
