#!/usr/bin/env node

import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createRuntime } from '../ts/VolvoxAI.js';
import { Model } from '../ts/core/Model.js';
import { parseGraphDocument } from '../ts/core/Graph.js';

const SCHEMA = 'volvoxai.cpu-public-runtime-baseline/v1';
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

async function collectGarbageBeforeContext() {
  if (typeof globalThis.gc !== 'function') {
    throw new Error('public CPU latency workers require --expose-gc');
  }
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
}

async function measureWorker(samples, warmup, width) {
  const input = Float32Array.from({ length: width }, (_, index) => (index % 257) / 257);
  const expectedDigest = byteDigest(input);
  const snapshot = staticIdentitySnapshot(width);
  const shapedInputs = Object.freeze({
    x: Object.freeze({ data: input, shape: Object.freeze([1, width]) }),
  });
  const runtime = await createRuntime({ backends: ['cpu-js'] });
  let compiled;
  let context;
  let retainedResult;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
    });
    await collectGarbageBeforeContext();
    const contextStarted = performance.now();
    context = await compiled.createContext();
    const contextCreationMs = performance.now() - contextStarted;

    const coldStarted = performance.now();
    retainedResult = await context.execute(shapedInputs);
    const coldExecutionMs = performance.now() - coldStarted;
    const firstOutput = await retainedResult.output('y').read();
    const observedDigest = byteDigest(firstOutput);
    const shapeSignature = retainedResult.report.shapeSignature;
    if (retainedResult.backend !== 'cpu-js' || typeof shapeSignature !== 'string' ||
        shapeSignature.length === 0 || observedDigest !== expectedDigest) {
      throw new Error('public CPU cold execution returned invalid route or parity evidence');
    }

    for (let index = 0; index < warmup; index++) {
      const result = await context.execute(shapedInputs);
      await result.close();
    }
    const samplesMs = [];
    for (let index = 0; index < samples; index++) {
      const started = performance.now();
      const result = await context.execute(shapedInputs);
      samplesMs.push(performance.now() - started);
      await result.close();
    }

    await context.close();
    context = undefined;
    const stableReadableAfterClose =
      byteDigest(await retainedResult.output('y').read()) === expectedDigest;
    await retainedResult.close();
    retainedResult = undefined;
    return Object.freeze({
      samplesMs,
      contextCreationMs,
      coldExecutionMs,
      shapeSignature,
      observedDigest,
      stableReadableAfterClose,
    });
  } finally {
    await context?.close();
    await retainedResult?.close();
    await compiled?.close();
    await runtime.close();
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
        `public CPU worker ${index + 1} failed: ` +
        `${child.stderr.trim() || child.stdout.trim() || `exit ${child.status}`}`,
      );
    }
    let measurement;
    try {
      measurement = JSON.parse(child.stdout);
    } catch {
      throw new Error(`public CPU worker ${index + 1} returned invalid JSON`);
    }
    if (!Array.isArray(measurement?.samplesMs) || measurement.samplesMs.length !== samples ||
        measurement.samplesMs.some((value) => !Number.isFinite(value) || value < 0) ||
        measurement?.stableReadableAfterClose !== true) {
      throw new Error(`public CPU worker ${index + 1} returned invalid evidence`);
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
  const restoreConsole = installDiagnosticConsole();
  try {
    if (worker === 1) {
      process.stdout.write(`${JSON.stringify(await measureWorker(samples, warmup, width))}\n`);
      return;
    }
  } finally {
    restoreConsole();
  }

  const runs = integerArgument('runs', 5, 1, 20);
  const gate = stringArgument('gate', 'committed');
  if (gate !== 'committed' && gate !== 'off') {
    throw new Error("--gate must be 'committed' or 'off'");
  }
  const reference = gate === 'committed'
    ? await loadReference(stringArgument('reference', DEFAULT_REFERENCE), { samples, warmup, width })
    : null;
  const measurements = independentMeasurements(runs, samples, warmup, width);
  const runMediansMs = measurements.map(({ samplesMs }) => median(samplesMs));
  const medianOfRunMediansMs = median(runMediansMs);
  const maximumMedianMs = reference === null
    ? null
    : reference.medianMs * (1 + reference.maximumRegressionPercent / 100);
  const passed = maximumMedianMs === null || medianOfRunMediansMs <= maximumMedianMs;
  const report = Object.freeze({
    schema: SCHEMA,
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      runtime: 'node',
      version: process.version,
      platform: process.platform,
      architecture: process.arch,
      backend: 'public-runtime/cpu-js-provider',
    }),
    workload: Object.freeze({
      operation: 'Identity',
      dtype: 'float32',
      shape: Object.freeze([1, width]),
      warmupExecutions: warmup,
      measuredExecutions: samples,
      measurementRuns: runs,
      logicalInputBytes: width * 4,
      logicalOutputBytes: width * 4,
    }),
    preparation: Object.freeze({
      contextCreationSamplesMs: Object.freeze(
        measurements.map(({ contextCreationMs }) => contextCreationMs),
      ),
      coldExecutionSamplesMs: Object.freeze(
        measurements.map(({ coldExecutionMs }) => coldExecutionMs),
      ),
    }),
    latency: Object.freeze({
      timingScope: 'public ExecutionContext.execute through exact result-owned output snapshot',
      runMediansMs: Object.freeze(runMediansMs),
      medianOfRunMediansMs,
      gate: reference === null ? null : Object.freeze({
        referenceSchema: REFERENCE_SCHEMA,
        referenceFile: reference.filename,
        referenceRecordedAt: reference.recordedAt,
        referenceMedianMs: reference.medianMs,
        maximumRegressionPercent: reference.maximumRegressionPercent,
        maximumMedianMs,
        regressionPercent: ((medianOfRunMediansMs / reference.medianMs) - 1) * 100,
        passed,
      }),
    }),
    parity: Object.freeze({
      expectedDigest: byteDigest(
        Float32Array.from({ length: width }, (_, index) => (index % 257) / 257),
      ),
      runDigests: Object.freeze(measurements.map(({ observedDigest }) => observedDigest)),
      shapeSignatures: Object.freeze(measurements.map(({ shapeSignature }) => shapeSignature)),
      stableReadableAfterClose: measurements.every(
        ({ stableReadableAfterClose }) => stableReadableAfterClose,
      ),
    }),
  });
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!passed) process.exitCode = 1;
}

await main();
