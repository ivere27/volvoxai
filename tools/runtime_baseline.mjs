#!/usr/bin/env node

import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Graph, VolvoxAI } from '../ts/index.js';

const SCHEMA = 'volvoxai.runtime-baseline/v1';
const REFERENCE_SCHEMA = 'volvoxai.runtime-latency-reference/v1';
const DEFAULT_REFERENCE = fileURLToPath(
  new URL('../tests/baselines/runtime_cpu_identity.json', import.meta.url),
);

function integerArgument(name, fallback, minimum, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw == null ? fallback : Number(raw.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['samples', 'warmup', 'width', 'reference', 'latency-worker']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function stringArgument(name, fallback) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw == null ? fallback : raw.slice(prefix.length);
  if (!value) throw new Error(`--${name} must not be empty`);
  return value;
}

async function latencyReference(filename, { samples, warmup, width }) {
  const resolved = path.resolve(filename);
  let reference;
  try {
    reference = JSON.parse(await readFile(resolved, 'utf8'));
  } catch (error) {
    throw new Error(`cannot read latency reference '${resolved}': ${error.message}`);
  }
  if (!reference || reference.schema !== REFERENCE_SCHEMA ||
      !reference.environment || !reference.workload || !reference.latency) {
    throw new Error(`latency reference '${resolved}' does not match ${REFERENCE_SCHEMA}`);
  }
  const nodeMajor = Number(process.versions.node.split('.')[0]);
  const expectedEnvironment = reference.environment;
  if (expectedEnvironment.nodeMajor !== nodeMajor ||
      expectedEnvironment.platform !== process.platform ||
      expectedEnvironment.architecture !== process.arch) {
    throw new Error(
      `latency reference '${resolved}' targets Node ${expectedEnvironment.nodeMajor} ` +
      `${expectedEnvironment.platform}/${expectedEnvironment.architecture}, not Node ${nodeMajor} ` +
      `${process.platform}/${process.arch}`,
    );
  }
  const expectedWorkload = reference.workload;
  if (expectedWorkload.operation !== 'Identity' || expectedWorkload.dtype !== 'float32' ||
      expectedWorkload.width !== width || expectedWorkload.warmupExecutions !== warmup ||
      expectedWorkload.measuredExecutions !== samples) {
    throw new Error(
      `latency reference '${resolved}' does not match --width=${width}, ` +
      `--warmup=${warmup}, --samples=${samples}`,
    );
  }
  if (!Number.isFinite(reference.latency.medianMs) || reference.latency.medianMs <= 0 ||
      reference.latency.maximumRegressionPercent !== 5) {
    throw new Error(`latency reference '${resolved}' has an invalid 5% median budget`);
  }
  const measurementRuns = reference.latency.recordingRuns ?? 1;
  if (!Number.isSafeInteger(measurementRuns) || measurementRuns < 1 || measurementRuns > 20) {
    throw new Error(`latency reference '${resolved}' has an invalid recordingRuns count`);
  }
  return Object.freeze({
    filename: resolved,
    recordedAt: reference.recordedAt,
    medianMs: reference.latency.medianMs,
    maximumRegressionPercent: reference.latency.maximumRegressionPercent,
    measurementRuns,
  });
}

function graphFor(width) {
  const graph = new Graph();
  const input = graph.addInput('x', [1, width]);
  const output = graph.addOp('Identity', { input }, {
    out: { name: 'y', shape: [1, width], dtype: 'float32' },
  }).out;
  graph.setOutputs(output);
  return graph;
}

function memoryFields() {
  const value = process.memoryUsage();
  return {
    rssBytes: value.rss,
    heapUsedBytes: value.heapUsed,
    externalBytes: value.external,
    arrayBufferBytes: value.arrayBuffers,
  };
}

async function measuredMemory() {
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  return memoryFields();
}

function memoryDelta(current, baseline) {
  return Object.fromEntries(Object.keys(baseline).map((key) => [key, current[key] - baseline[key]]));
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
}

function sameBytes(left, right) {
  if (!ArrayBuffer.isView(left) || !ArrayBuffer.isView(right) ||
      left.byteLength !== right.byteLength) return false;
  const a = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const b = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return false;
  }
  return true;
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

async function stableHostReadEvidence(result, outputName) {
  let first = await result.output(outputName).read();
  let second = await result.output(outputName).read();
  const evidence = {
    freshCallerOwnedReads: first !== second && first.buffer !== second.buffer &&
      sameBytes(first, second),
    outputBytes: first.byteLength,
    digest: byteDigest(first),
  };
  first = null;
  second = null;
  return evidence;
}

async function resultDigest(result, outputName) {
  let value = await result.output(outputName).read();
  const digest = byteDigest(value);
  value = null;
  return digest;
}

async function collectRuntimeEvidence({ samples, warmup, width }) {
  const input = Float32Array.from({ length: width }, (_, index) => (index % 257) / 257);
  const graph = graphFor(width);
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  let compiled;
  let idleOne;
  let idleTwo;
  let latencyContext;
  let stableContext;
  let stableResult;
  try {
    compiled = await model.compile({
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    });
    const compilationReport = compiled.report;

    // Initialize model-level caches before taking the retained-runtime baseline.
    const primerContext = await compiled.createContext();
    for (let index = 0; index < 4; index++) {
      const primerResult = await primerContext.execute({ x: input });
      await primerResult.output('y').read();
      await primerResult.output('y').read();
      await primerResult.close();
    }
    await primerContext.close();
    const retainedBaseline = await measuredMemory();

    idleOne = await compiled.createContext();
    const oneIdleContext = await measuredMemory();
    idleTwo = await compiled.createContext();
    const twoIdleContexts = await measuredMemory();
    await idleOne.close();
    await idleTwo.close();
    idleOne = null;
    idleTwo = null;
    const idleContextsClosed = await measuredMemory();

    latencyContext = await compiled.createContext();
    for (let index = 0; index < warmup; index++) {
      const result = await latencyContext.execute({ x: input });
      await result.close();
    }
    const latencySamplesMs = [];
    for (let index = 0; index < samples; index++) {
      const started = performance.now();
      const result = await latencyContext.execute({ x: input });
      await result.close();
      latencySamplesMs.push(performance.now() - started);
    }
    await latencyContext.close();
    latencyContext = null;

    stableContext = await compiled.createContext();
    const beforeStableResult = await measuredMemory();
    stableResult = await stableContext.execute({ x: input });
    const executionReport = stableResult.report;
    const stableResultAllocated = await measuredMemory();
    const stableRead = await stableHostReadEvidence(stableResult, 'y');
    const freshCallerOwnedReads = stableRead.freshCallerOwnedReads;
    const outputBytes = stableRead.outputBytes;
    const stableBytesDigest = stableRead.digest;
    const stableResultWithCallerReads = await measuredMemory();

    // Closing immediately after enqueue must drain the already accepted execution.
    const acceptedExecution = stableContext.execute({ x: input });
    const firstClose = stableContext.close();
    const secondClose = stableContext.close();
    const acceptedResult = await acceptedExecution;
    await acceptedResult.close();
    await Promise.all([firstClose, secondClose]);
    const readableAfterContextClose =
      await resultDigest(stableResult, 'y') === stableBytesDigest;
    const stableResultAfterContextClose = await measuredMemory();
    stableContext = null;

    const firstResultClose = stableResult.close();
    const secondResultClose = stableResult.close();
    const resultCloseIsIdempotent = firstResultClose === secondResultClose;
    await Promise.all([firstResultClose, secondResultClose]);
    stableResult = null;
    const stableResultClosed = await measuredMemory();

    await compiled.close();
    await model.close();
    await runtime.close();
    compiled = null;
    return {
      compilationReport,
      executionReport,
      retainedBaseline,
      oneIdleContext,
      twoIdleContexts,
      idleContextsClosed,
      beforeStableResult,
      stableResultAllocated,
      stableResultWithCallerReads,
      stableResultAfterContextClose,
      stableResultClosed,
      latencySamplesMs,
      outputBytes,
      freshCallerOwnedReads,
      readableAfterContextClose,
      acceptedWorkDrained: true,
      contextCloseIsIdempotent: firstClose === secondClose,
      resultCloseIsIdempotent,
    };
  } finally {
    await stableResult?.close();
    await stableContext?.close();
    await latencyContext?.close();
    await idleTwo?.close();
    await idleOne?.close();
    await compiled?.close();
    await model.close();
    await runtime.close();
  }
}

function installDiagnosticConsole() {
  const methods = { log: console.log, info: console.info, warn: console.warn };
  const diagnostic = (...values) => {
    process.stderr.write(`${values.map((value) => String(value)).join(' ')}\n`);
  };
  console.log = diagnostic;
  console.info = diagnostic;
  console.warn = diagnostic;
  return () => {
    console.log = methods.log;
    console.info = methods.info;
    console.warn = methods.warn;
  };
}

async function runLatencyWorker(samples, warmup, width) {
  const restoreConsole = installDiagnosticConsole();
  try {
    const measurement = await collectRuntimeEvidence({ samples, warmup, width });
    process.stdout.write(`${JSON.stringify({
      samplesMs: measurement.latencySamplesMs,
      medianMs: median(measurement.latencySamplesMs),
    })}\n`);
  } finally {
    restoreConsole();
  }
}

function additionalLatencyRuns(count, samples, warmup, width) {
  const runs = [];
  const script = fileURLToPath(import.meta.url);
  for (let index = 1; index < count; index++) {
    const child = spawnSync(process.execPath, [
      ...process.execArgv,
      script,
      '--latency-worker=1',
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
        `latency measurement worker ${index + 1} failed: ${child.stderr.trim() || `exit ${child.status}`}`,
      );
    }
    let run;
    try {
      run = JSON.parse(child.stdout);
    } catch {
      throw new Error(`latency measurement worker ${index + 1} returned invalid JSON`);
    }
    if (!Array.isArray(run.samplesMs) || run.samplesMs.length !== samples ||
        run.samplesMs.some((value) => !Number.isFinite(value) || value < 0) ||
        !Number.isFinite(run.medianMs)) {
      throw new Error(`latency measurement worker ${index + 1} returned invalid samples`);
    }
    runs.push(run.samplesMs);
  }
  return runs;
}

async function main() {
  rejectUnknownArguments();
  if (typeof globalThis.gc !== 'function') {
    throw new Error('explicit GC is required; run through npm run baseline:runtime');
  }
  const samples = integerArgument('samples', 51, 3, 101);
  const warmup = integerArgument('warmup', 10, 0, 20);
  const width = integerArgument('width', 65_536, 1, 1_048_576);
  const workerArgument = process.argv.slice(2).find((argument) =>
    argument.startsWith('--latency-worker='));
  if (workerArgument != null) {
    if (workerArgument !== '--latency-worker=1') {
      throw new Error('--latency-worker is an internal flag and must equal 1');
    }
    await runLatencyWorker(samples, warmup, width);
    return;
  }
  const reference = await latencyReference(
    stringArgument('reference', DEFAULT_REFERENCE),
    { samples, warmup, width },
  );
  const restoreConsole = installDiagnosticConsole();
  try {
    const measurements = await collectRuntimeEvidence({ samples, warmup, width });
    // Collect after the lifecycle's async frame has returned. Await expressions
    // may retain their resolved caller-owned arrays until that frame completes;
    // those arrays are not runtime resources and must not contaminate this gate.
    const allOwnersClosed = await measuredMemory();
    const {
      compilationReport,
      executionReport,
      retainedBaseline,
      oneIdleContext,
      twoIdleContexts,
      idleContextsClosed,
      beforeStableResult,
      stableResultAllocated,
      stableResultWithCallerReads,
      stableResultAfterContextClose,
      stableResultClosed,
      latencySamplesMs: firstLatencySamplesMs,
      outputBytes,
      freshCallerOwnedReads,
      readableAfterContextClose,
      acceptedWorkDrained,
      contextCloseIsIdempotent,
      resultCloseIsIdempotent,
    } = measurements;

    const latencyRunsMs = [
      firstLatencySamplesMs,
      ...additionalLatencyRuns(reference.measurementRuns, samples, warmup, width),
    ];
    const latencyRunMediansMs = latencyRunsMs.map(median);
    const latencySamplesMs = latencyRunsMs.flat();
    const medianMs = median(latencyRunMediansMs);
    const maximumMedianMs = reference.medianMs *
      (1 + reference.maximumRegressionPercent / 100);
    const regressionPercent = ((medianMs / reference.medianMs) - 1) * 100;
    const latencyGatePassed = medianMs <= maximumMedianMs;
    const alignedOutputBytes = Math.ceil(outputBytes / 256) * 256;
    const oneIdleBytes = memoryDelta(oneIdleContext, retainedBaseline).arrayBufferBytes;
    const twoIdleBytes = memoryDelta(twoIdleContexts, retainedBaseline).arrayBufferBytes;
    const idleContextsClosedBytes =
      memoryDelta(idleContextsClosed, retainedBaseline).arrayBufferBytes;
    const allOwnersClosedBytes = memoryDelta(allOwnersClosed, retainedBaseline).arrayBufferBytes;
    const stableSnapshotBytes =
      memoryDelta(stableResultAllocated, beforeStableResult).arrayBufferBytes;
    const memoryGatePassed = oneIdleBytes > 0 &&
      twoIdleBytes <= oneIdleBytes * 2.2 &&
      stableSnapshotBytes <= alignedOutputBytes * 1.1 &&
      idleContextsClosedBytes <= 0 &&
      allOwnersClosedBytes <= 0;
    const disposalGatePassed = acceptedWorkDrained && contextCloseIsIdempotent &&
      resultCloseIsIdempotent && freshCallerOwnedReads && readableAfterContextClose;

    const evidence = {
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        runtime: 'node',
        version: process.version,
        platform: process.platform,
        architecture: process.arch,
        backend: 'cpu',
        explicitGc: true,
      },
      workload: {
        operation: 'Identity',
        dtype: 'float32',
        shape: [1, width],
        warmupExecutions: warmup,
        measuredExecutions: samples,
        measurementRuns: reference.measurementRuns,
      },
      latency: {
        samplesMs: latencySamplesMs,
        runMediansMs: latencyRunMediansMs,
        medianMs,
        minimumMs: Math.min(...latencySamplesMs),
        maximumMs: Math.max(...latencySamplesMs),
        gate: {
          referenceSchema: REFERENCE_SCHEMA,
          referenceFile: path.relative(process.cwd(), reference.filename) || '.',
          referenceRecordedAt: reference.recordedAt,
          referenceMedianMs: reference.medianMs,
          maximumRegressionPercent: reference.maximumRegressionPercent,
          maximumMedianMs,
          regressionPercent,
          passed: latencyGatePassed,
        },
      },
      memory: {
        retainedBaseline,
        oneIdleContext,
        twoIdleContexts,
        idleContextsClosed,
        beforeStableResult,
        stableResultAllocated,
        stableResultWithCallerReads,
        stableResultAfterContextClose,
        stableResultClosed,
        allOwnersClosed,
        deltasFromRetainedBaseline: {
          oneIdleContext: memoryDelta(oneIdleContext, retainedBaseline),
          twoIdleContexts: memoryDelta(twoIdleContexts, retainedBaseline),
          idleContextsClosed: memoryDelta(idleContextsClosed, retainedBaseline),
          allOwnersClosed: memoryDelta(allOwnersClosed, retainedBaseline),
        },
        stableResultDeltas: {
          allocated: memoryDelta(stableResultAllocated, beforeStableResult),
          withCallerReads: memoryDelta(stableResultWithCallerReads, beforeStableResult),
          afterContextClose: memoryDelta(stableResultAfterContextClose, beforeStableResult),
          closed: memoryDelta(stableResultClosed, beforeStableResult),
        },
      },
      snapshot: {
        outputBytes,
        alignedOutputBytes,
        freshCallerOwnedReads,
        readableAfterContextClose,
      },
      disposal: {
        acceptedWorkDrained,
        contextCloseIsIdempotent,
        resultCloseIsIdempotent,
        allOwnersClosed: allOwnersClosedBytes <= 0,
      },
      runtime: {
        compilation: compilationReport,
        execution: executionReport,
      },
      budgets: {
        medianOneContextLatencyRegressionPercent: reference.maximumRegressionPercent,
        twoIdleContextsToOneMutableStorageRatio: 2.2,
        stableSnapshotBookkeepingPercent: 10,
        closedUnretainedResourcesReturnToBaseline: true,
      },
    };
    process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
    if (!latencyGatePassed) {
      throw new Error(
        `median one-context latency ${medianMs.toFixed(6)} ms exceeds the committed ` +
        `${maximumMedianMs.toFixed(6)} ms budget`,
      );
    }
    if (!memoryGatePassed) {
      throw new Error(
        'runtime ownership memory budget failed: ' +
        `oneIdle=${oneIdleBytes}, twoIdle=${twoIdleBytes}, ` +
        `stableSnapshot=${stableSnapshotBytes}, idleClosed=${idleContextsClosedBytes}, ` +
        `allClosed=${allOwnersClosedBytes} array-buffer bytes`,
      );
    }
    if (!disposalGatePassed) {
      throw new Error('runtime lifecycle/disposal evidence failed');
    }
  } finally {
    restoreConsole();
  }
}

main().catch((error) => {
  process.stderr.write(`runtime baseline failed: ${String(error?.message || error)}\n`);
  process.exitCode = 1;
});
