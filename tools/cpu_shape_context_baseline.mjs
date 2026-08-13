#!/usr/bin/env node

import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { CPUShapeExecutionContext } from '../ts/core/CPUShapeExecutionContext.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import { Model } from '../ts/core/Model.js';

const SCHEMA = 'volvoxai.cpu-shape-context-baseline/v1';
const REFERENCE_SCHEMA = 'volvoxai.runtime-latency-reference/v1';
const DEFAULT_REFERENCE = fileURLToPath(
  new URL('../tests/baselines/runtime_cpu_identity.json', import.meta.url),
);

function integerArgument(name, fallback, minimum, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw === undefined ? fallback : Number(raw.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function stringArgument(name, fallback) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw === undefined ? fallback : raw.slice(prefix.length);
  if (!value) throw new Error(`--${name} must not be empty`);
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['samples', 'warmup', 'width', 'runs', 'gate', 'reference', 'worker']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
}

function byteDigest(value) {
  const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  let digest = 0x811c9dc5;
  for (const byte of bytes) {
    digest ^= byte;
    digest = Math.imul(digest, 0x01000193) >>> 0;
  }
  return digest;
}

function staticIdentitySnapshot(width) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1, width] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, width] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

async function loadReference(filename, { samples, warmup, width }) {
  const resolved = path.resolve(filename);
  const reference = JSON.parse(await readFile(resolved, 'utf8'));
  if (reference?.schema !== REFERENCE_SCHEMA ||
      reference?.workload?.operation !== 'Identity' ||
      reference?.workload?.dtype !== 'float32' ||
      reference?.workload?.width !== width ||
      reference?.workload?.warmupExecutions !== warmup ||
      reference?.workload?.measuredExecutions !== samples ||
      !Number.isFinite(reference?.latency?.medianMs) ||
      reference.latency.medianMs <= 0 ||
      reference.latency.maximumRegressionPercent !== 5) {
    throw new Error(`reference '${resolved}' does not match the requested Identity workload`);
  }
  return Object.freeze({
    filename: resolved,
    recordedAt: reference.recordedAt,
    medianMs: reference.latency.medianMs,
    maximumRegressionPercent: reference.latency.maximumRegressionPercent,
  });
}

function installDiagnosticConsole() {
  const original = console.log;
  console.log = (...values) => process.stderr.write(`${values.map(String).join(' ')}\n`);
  return () => { console.log = original; };
}

async function collectGarbageBeforeLatencyContext() {
  if (typeof globalThis.gc !== 'function') {
    throw new Error('CPU shape-context latency workers require --expose-gc');
  }
  // Match the committed runtime baseline protocol: compilation/primer state is
  // retained, but unrelated model-construction garbage is collected before the
  // measured context is created. Otherwise this gate compares a dirty young
  // generation against the clean-heap runtime reference.
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
}

async function measureRun(snapshot, input, warmup, samples) {
  const createStarted = performance.now();
  const context = new CPUShapeExecutionContext(snapshot);
  const contextCreationMs = performance.now() - createStarted;
  let stableResult;
  try {
    const coldStarted = performance.now();
    stableResult = await context.execute({ x: { data: input, shape: [1, input.length] } });
    const coldExecutionMs = performance.now() - coldStarted;
    const stableDigest = byteDigest(stableResult.get('y').data);
    for (let index = 0; index < warmup; index++) {
      await context.execute({ x: { data: input, shape: [1, input.length] } });
    }
    const samplesMs = [];
    for (let index = 0; index < samples; index++) {
      const started = performance.now();
      await context.execute({ x: { data: input, shape: [1, input.length] } });
      samplesMs.push(performance.now() - started);
    }
    const beforeClose = context.inspect();
    await context.close();
    const afterClose = context.inspect();
    return {
      samplesMs,
      contextCreationMs,
      coldExecutionMs,
      stableDigest,
      stableReadableAfterClose: byteDigest(stableResult.get('y').data) === stableDigest,
      beforeClose,
      afterClose,
    };
  } finally {
    await context.close();
  }
}

async function runWorker(samples, warmup, width) {
  const input = Float32Array.from({ length: width }, (_, index) => (index % 257) / 257);
  const snapshotStarted = performance.now();
  const snapshot = staticIdentitySnapshot(width);
  const snapshotCaptureMs = performance.now() - snapshotStarted;
  const restoreConsole = installDiagnosticConsole();
  try {
    await collectGarbageBeforeLatencyContext();
    const measurement = await measureRun(snapshot, input, warmup, samples);
    process.stdout.write(`${JSON.stringify({ ...measurement, snapshotCaptureMs })}\n`);
  } finally {
    restoreConsole();
  }
}

function independentMeasurements(runs, samples, warmup, width) {
  const script = fileURLToPath(import.meta.url);
  const measurements = [];
  for (let index = 0; index < runs; index++) {
    const child = spawnSync(process.execPath, [
      ...process.execArgv.filter((argument) => argument !== '--expose-gc'),
      '--expose-gc',
      script,
      '--worker=1',
      `--samples=${samples}`,
      `--warmup=${warmup}`,
      `--width=${width}`,
    ], {
      cwd: process.cwd(),
      encoding: 'utf8',
      maxBuffer: 1024 * 1024,
    });
    if (child.error) throw child.error;
    if (child.status !== 0) {
      throw new Error(
        `CPU shape-context worker ${index + 1} failed: ` +
        `${child.stderr.trim() || child.stdout.trim() || `exit ${child.status}`}`,
      );
    }
    let measurement;
    try {
      measurement = JSON.parse(child.stdout);
    } catch {
      throw new Error(`CPU shape-context worker ${index + 1} returned invalid JSON`);
    }
    if (!Array.isArray(measurement?.samplesMs) || measurement.samplesMs.length !== samples ||
        measurement.samplesMs.some((value) => !Number.isFinite(value) || value < 0) ||
        measurement?.beforeClose?.staticFastPath !== true ||
        measurement?.stableReadableAfterClose !== true) {
      throw new Error(`CPU shape-context worker ${index + 1} returned invalid evidence`);
    }
    measurements.push(measurement);
  }
  return measurements;
}

async function main() {
  rejectUnknownArguments();
  const samples = integerArgument('samples', 51, 3, 10001);
  const warmup = integerArgument('warmup', 10, 0, 10000);
  const width = integerArgument('width', 65536, 1, 16_777_216);
  const worker = integerArgument('worker', 0, 0, 1);
  if (worker === 1) {
    await runWorker(samples, warmup, width);
    return;
  }
  const runs = integerArgument('runs', 5, 1, 20);
  const gate = stringArgument('gate', 'committed');
  if (!['committed', 'off'].includes(gate)) {
    throw new Error("--gate must be 'committed' or 'off'");
  }
  const reference = gate === 'committed'
    ? await loadReference(stringArgument('reference', DEFAULT_REFERENCE), { samples, warmup, width })
    : null;
  const input = Float32Array.from({ length: width }, (_, index) => (index % 257) / 257);
  const expectedDigest = byteDigest(input);
  const measurements = independentMeasurements(runs, samples, warmup, width);

  const latencyRunsMs = measurements.map((measurement) => measurement.samplesMs);
  const samplesMs = latencyRunsMs.flat();
  const runMediansMs = latencyRunsMs.map(median);
  const medianOfRunMediansMs = median(runMediansMs);
  const maximumMedianMs = reference === null
    ? null
    : reference.medianMs * (1 + reference.maximumRegressionPercent / 100);
  const gatePassed = reference === null ? null : medianOfRunMediansMs <= maximumMedianMs;
  const first = measurements[0];
  const evidence = {
    schema: SCHEMA,
    recordedAt: new Date().toISOString(),
    environment: {
      runtime: 'node', version: process.version,
      platform: process.platform, architecture: process.architecture ?? process.arch,
      backend: 'cpu-shape-v1',
    },
    workload: {
      operation: 'Identity', dtype: 'float32', shape: [1, width],
      warmupExecutions: warmup, measuredExecutions: samples, measurementRuns: runs,
      logicalInputBytes: input.byteLength, logicalOutputBytes: input.byteLength,
    },
    preparation: {
      snapshotCaptureSamplesMs: measurements.map((measurement) => measurement.snapshotCaptureMs),
      contextCreationSamplesMs: measurements.map((measurement) => measurement.contextCreationMs),
      coldExecutionSamplesMs: measurements.map((measurement) => measurement.coldExecutionMs),
    },
    latency: {
      timingScope: 'mandatory shaped-view validation through exact result-owned output snapshot',
      samplesMs,
      runMediansMs,
      medianOfRunMediansMs,
      p50Ms: percentile(samplesMs, 0.5),
      p95Ms: percentile(samplesMs, 0.95),
      minimumMs: Math.min(...samplesMs),
      maximumMs: Math.max(...samplesMs),
      gate: reference === null ? null : {
        referenceSchema: REFERENCE_SCHEMA,
        referenceFile: path.relative(process.cwd(), reference.filename),
        referenceRecordedAt: reference.recordedAt,
        referenceMedianMs: reference.medianMs,
        maximumRegressionPercent: reference.maximumRegressionPercent,
        maximumMedianMs,
        regressionPercent: ((medianOfRunMediansMs / reference.medianMs) - 1) * 100,
        passed: gatePassed,
      },
    },
    parity: {
      expectedDigest,
      runDigests: measurements.map((measurement) => measurement.stableDigest),
      byteExactIdentity: measurements.every((measurement) =>
        measurement.stableDigest === expectedDigest),
      stableReadableAfterClose: measurements.every((measurement) =>
        measurement.stableReadableAfterClose),
    },
    lifecycle: {
      staticFastPath: first.beforeClose.staticFastPath,
      planCacheEntries: first.beforeClose.planCache.entryCount,
      capacityBytes: first.beforeClose.capacities.currentBytes,
      capacityHighWaterBytes: first.beforeClose.capacities.highWaterBytes,
      growEvents: first.beforeClose.capacities.growEvents,
      afterCloseCapacityBytes: first.afterClose.capacities.currentBytes,
    },
  };
  process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
  if (!evidence.parity.byteExactIdentity || !evidence.parity.stableReadableAfterClose ||
      evidence.lifecycle.staticFastPath !== true || evidence.lifecycle.planCacheEntries !== 0) {
    throw new Error('CPU shape-context correctness/lifecycle baseline failed');
  }
  if (gatePassed === false) {
    throw new Error(
      `CPU shape-context median ${medianOfRunMediansMs} ms exceeds committed ` +
      `${maximumMedianMs} ms budget`,
    );
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
});
