#!/usr/bin/env node

import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  Model,
  VolvoxAI,
  parseGraphDocument,
} from '../ts/index.js';

const SCHEMA = 'volvoxai.runtime-baseline/v1';
const REFERENCE_SCHEMA = 'volvoxai.runtime-latency-reference/v1';
const DEFAULT_REFERENCE = fileURLToPath(
  new URL('../tests/baselines/runtime_cpu_identity.json', import.meta.url),
);
const REPOSITORY_ROOT = fileURLToPath(new URL('../', import.meta.url));
const MEASUREMENT_RUNS = 5;

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
  const known = new Set([
    'backend', 'samples', 'warmup', 'width', 'reference', 'wasm',
    'native-build-dir', 'latency-worker',
  ]);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function backendArgument() {
  const backend = stringArgument('backend', 'cpu');
  if (!['cpu', 'wasm', 'native-cpu'].includes(backend)) {
    throw new Error("--backend must be 'cpu', 'wasm', or 'native-cpu'");
  }
  return backend;
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

function percentile(values, fraction) {
  if (!Array.isArray(values) || values.length === 0 ||
      !Number.isFinite(fraction) || fraction <= 0 || fraction > 1) {
    throw new Error('percentile requires samples and a fraction in (0, 1]');
  }
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(fraction * ordered.length) - 1];
}

function populationVariance(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error('variance requires at least one sample');
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
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

async function stableHostReadEvidence(result, outputName, expected) {
  let first = await result.output(outputName).read();
  let second = await result.output(outputName).read();
  const evidence = {
    freshCallerOwnedReads: first !== second && first.buffer !== second.buffer &&
      sameBytes(first, second),
    byteExactIdentity: sameBytes(first, expected),
    outputBytes: first.byteLength,
    digest: byteDigest(first),
    expectedDigest: byteDigest(expected),
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

async function collectRuntimeEvidence({ backend, samples, warmup, width, wasmUrl }) {
  const input = Float32Array.from({ length: width }, (_, index) => (index % 257) / 257);
  const inputView = Object.freeze({ data: input, shape: Object.freeze([1, width]) });
  const snapshot = graphFor(width);
  const profileWasm = backend === 'wasm';
  if (profileWasm) {
    globalThis.__VOLVOX_WASM_COMPILE_PROFILE = true;
    globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULT = null;
    globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS = [];
  }
  const runtime = await VolvoxAI.createRuntime({
    backends: [backend],
    ...(profileWasm ? { wasmUrl } : {}),
  });
  let compiled;
  let idleOne;
  let idleTwo;
  let latencyContext;
  let stableContext;
  let stableResult;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    const compilationReport = compiled.report;

    // Initialize model-level caches before taking the retained-runtime baseline.
    const primerContext = await compiled.createContext();
    for (let index = 0; index < 4; index++) {
      const primerResult = await primerContext.execute({ x: inputView });
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
      const result = await latencyContext.execute({ x: inputView });
      await result.close();
    }
    const latencySamplesMs = [];
    for (let index = 0; index < samples; index++) {
      const started = performance.now();
      const result = await latencyContext.execute({ x: inputView });
      await result.close();
      latencySamplesMs.push(performance.now() - started);
    }
    await latencyContext.close();
    latencyContext = null;

    stableContext = await compiled.createContext();
    // Provider contexts allocate activation capacity on first bind. Prime that
    // context before isolating result-owned snapshot bytes.
    const stablePrimer = await stableContext.execute({ x: inputView });
    await stablePrimer.close();
    const beforeStableResult = await measuredMemory();
    stableResult = await stableContext.execute({ x: inputView });
    const executionReport = stableResult.report;
    const stableResultAllocated = await measuredMemory();
    const stableRead = await stableHostReadEvidence(stableResult, 'y', input);
    const freshCallerOwnedReads = stableRead.freshCallerOwnedReads;
    const byteExactIdentity = stableRead.byteExactIdentity;
    const outputBytes = stableRead.outputBytes;
    const stableBytesDigest = stableRead.digest;
    const expectedBytesDigest = stableRead.expectedDigest;
    const stableResultWithCallerReads = await measuredMemory();

    // Closing immediately after enqueue must drain the already accepted execution.
    const acceptedExecution = stableContext.execute({ x: inputView });
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
    await runtime.close();
    compiled = null;
    const wasmCompileProfiles = profileWasm
      ? [...(globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS || [])]
          .map((profile) => ({ ...profile }))
      : [];
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
      byteExactIdentity,
      stableBytesDigest,
      expectedBytesDigest,
      wasmCompileProfiles,
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
    await runtime.close();
    if (profileWasm) {
      delete globalThis.__VOLVOX_WASM_COMPILE_PROFILE;
      delete globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULT;
      delete globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS;
    }
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

async function runLatencyWorker(backend, samples, warmup, width, wasmUrl) {
  const restoreConsole = installDiagnosticConsole();
  try {
    const measurement = await collectRuntimeEvidence({
      backend, samples, warmup, width, wasmUrl,
    });
    process.stdout.write(`${JSON.stringify({
      samplesMs: measurement.latencySamplesMs,
      medianMs: median(measurement.latencySamplesMs),
    })}\n`);
  } finally {
    restoreConsole();
  }
}

function additionalLatencyRuns(count, backend, samples, warmup, width, wasmUrl) {
  const runs = [];
  const script = fileURLToPath(import.meta.url);
  for (let index = 1; index < count; index++) {
    const child = spawnSync(process.execPath, [
      ...process.execArgv,
      script,
      '--latency-worker=1',
      `--backend=${backend}`,
      `--samples=${samples}`,
      `--warmup=${warmup}`,
      `--width=${width}`,
      ...(backend === 'wasm' ? [`--wasm=${wasmUrl}`] : []),
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

function checkedSpawn(command, args, options, label) {
  const child = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
    ...options,
  });
  if (child.error) throw child.error;
  if (child.status !== 0) {
    throw new Error(
      `${label} failed: ${child.stderr?.trim() || child.stdout?.trim() || `exit ${child.status}`}`,
    );
  }
  return child;
}

async function buildNativeBaselineWorker(buildDirectory, temporaryDirectory) {
  const nativeBuildDirectory = path.join(buildDirectory, 'native');
  const linkFile = path.join(
    nativeBuildDirectory,
    'CMakeFiles',
    'volvoxai_cpu_only.dir',
    'link.txt',
  );
  let linkCommand;
  try {
    linkCommand = (await readFile(linkFile, 'utf8')).trim();
  } catch (error) {
    throw new Error(
      `cannot read '${linkFile}'; run 'make build_native' first: ${error.message}`,
    );
  }
  const tokens = (linkCommand.match(/[^\s"']+|"[^"]*"|'[^']*'/gu) || []).map(
    (token) => (/^(["']).*\1$/u.test(token) ? token.slice(1, -1) : token),
  );
  const compiler = tokens.shift();
  if (!compiler || tokens.some((token) => /["']/.test(token))) {
    throw new Error(`native CPU link command '${linkFile}' is unsupported`);
  }
  const mainObjectIndex = tokens.findIndex((token) => token.endsWith('/cli/main.c.o'));
  const outputIndex = tokens.indexOf('-o');
  if (mainObjectIndex < 0 || outputIndex < 0 || outputIndex + 1 >= tokens.length) {
    throw new Error(`native CPU link command '${linkFile}' has no replaceable CLI entry`);
  }

  const source = path.join(REPOSITORY_ROOT, 'tools', 'native_runtime_baseline.c');
  const object = path.join(temporaryDirectory, 'native_runtime_baseline.o');
  const executable = path.join(temporaryDirectory, 'native_runtime_baseline');
  checkedSpawn(compiler, [
    '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
    '-I', path.join(REPOSITORY_ROOT, 'native', 'include'),
    '-c', source, '-o', object,
  ], { cwd: REPOSITORY_ROOT }, 'native baseline helper compilation');

  tokens.splice(mainObjectIndex, 1);
  const adjustedOutputIndex = tokens.indexOf('-o');
  tokens.splice(adjustedOutputIndex, 0, object);
  tokens[adjustedOutputIndex + 2] = executable;
  checkedSpawn(compiler, tokens, { cwd: nativeBuildDirectory },
    'native baseline helper link');
  return Object.freeze({ executable, compiler, linkFile });
}

function nativeGraphDocument(width) {
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { shape: [1, width], dtype: 'float32' } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, width] } },
      params: {},
    }],
    outputs: ['y'],
  };
}

function parseNativeWorker(child, runIndex, samples) {
  const lines = child.stdout.trim().split(/\r?\n/u).filter(Boolean);
  if (lines.length === 0) {
    throw new Error(`native baseline worker ${runIndex + 1} returned no JSON`);
  }
  if (lines.length > 1) {
    process.stderr.write(`${lines.slice(0, -1).join('\n')}\n`);
  }
  let result;
  try {
    result = JSON.parse(lines.at(-1));
  } catch {
    throw new Error(`native baseline worker ${runIndex + 1} returned invalid JSON`);
  }
  if (result?.schema !== 'volvoxai.native-runtime-baseline-worker/v1' ||
      !Array.isArray(result?.latency?.samplesMs) ||
      result.latency.samplesMs.length !== samples ||
      result.latency.samplesMs.some((value) => !Number.isFinite(value) || value < 0) ||
      !Number.isFinite(result?.compilation?.compileTimeMs) ||
      result.compilation.compileTimeMs < 0 ||
      result?.parity?.byteExactIdentity !== true) {
    throw new Error(`native baseline worker ${runIndex + 1} returned invalid evidence`);
  }
  return result;
}

async function nativeRuntimeBaseline({ samples, warmup, width, buildDirectory }) {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), 'volvoxai-native-baseline-'));
  try {
    const graphPath = path.join(temporaryDirectory, 'graph.json');
    await writeFile(graphPath, `${JSON.stringify(nativeGraphDocument(width))}\n`, 'utf8');
    const worker = await buildNativeBaselineWorker(buildDirectory, temporaryDirectory);
    const runs = [];
    for (let runIndex = 0; runIndex < MEASUREMENT_RUNS; runIndex++) {
      const child = checkedSpawn(worker.executable, [
        graphPath, String(width), String(warmup), String(samples),
      ], { cwd: REPOSITORY_ROOT }, `native baseline worker ${runIndex + 1}`);
      runs.push(parseNativeWorker(child, runIndex, samples));
    }
    const latencyRunsMs = runs.map((run) => run.latency.samplesMs);
    const latencySamplesMs = latencyRunsMs.flat();
    const compileSamplesMs = runs.map((run) => run.compilation.compileTimeMs);
    const runMediansMs = latencyRunsMs.map(median);
    const first = runs[0];
    const expectedDigest = first.parity.expectedDigest;
    if (runs.some((run) => run.parity.expectedDigest !== expectedDigest ||
        run.parity.actualDigest !== expectedDigest)) {
      throw new Error('native baseline runs disagree on Identity parity digest');
    }
    return {
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        runtime: 'native',
        version: `native-api-v${first.nativeApiVersion}`,
        platform: process.platform,
        architecture: process.arch,
        backend: 'native-cpu',
        cpuThreads: 1,
        buildDirectory: path.relative(REPOSITORY_ROOT, buildDirectory) || '.',
        linkDescription: path.relative(REPOSITORY_ROOT, worker.linkFile),
        compiler: worker.compiler,
      },
      workload: {
        ...first.workload,
        measurementRuns: MEASUREMENT_RUNS,
      },
      latency: {
        samplesMs: latencySamplesMs,
        runMediansMs,
        p50Ms: percentile(latencySamplesMs, 0.5),
        p95Ms: percentile(latencySamplesMs, 0.95),
        varianceMs2: populationVariance(latencySamplesMs),
        medianOfRunMediansMs: median(runMediansMs),
        minimumMs: Math.min(...latencySamplesMs),
        maximumMs: Math.max(...latencySamplesMs),
        source: 'VxReport.execution_time_ms',
        gate: null,
      },
      compilation: {
        samplesMs: compileSamplesMs,
        p50Ms: percentile(compileSamplesMs, 0.5),
        p95Ms: percentile(compileSamplesMs, 0.95),
        minimumMs: Math.min(...compileSamplesMs),
        maximumMs: Math.max(...compileSamplesMs),
        reportedAllocationBytes: runs.map((run) =>
          run.compilation.reportedAllocationBytes),
      },
      allocation: {
        ...first.allocation,
        semantics: {
          reported: 'public VxReport retained lineage; execution includes the owned result snapshot',
          logical: 'sum of concrete public input and output tensor bytes',
        },
      },
      memory: {
        metric: 'process-rss',
        runs: runs.map((run) => run.memory),
        maximumSampledCurrentBytes: Math.max(...runs.flatMap((run) =>
          Object.values(run.memory.currentBytes))),
        maximumSampledCurrentDeltaFromRetainedBaselineBytes: Math.max(
          ...runs.map((run) => Math.max(...Object.values(run.memory.currentBytes)) -
            run.memory.currentBytes.retainedBaseline),
        ),
        maximumHighWaterBytes: Math.max(...runs.map((run) => run.memory.highWaterBytes)),
        maximumHighWaterDeltaBytes: Math.max(
          ...runs.map((run) => run.memory.highWaterDeltaBytes),
        ),
        limitation: 'RSS is process-level; allocator-exact native tensor high-water is not exposed before the redesign.',
      },
      snapshot: {
        outputBytes: first.allocation.resultSnapshotBytes,
        byteExactIdentity: true,
        expectedDigest,
        actualDigest: expectedDigest,
      },
      runtime: {
        runs: runs.map((run) => ({
          compilation: run.compilation,
          allocation: run.allocation,
        })),
      },
    };
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

function sampledMemoryHighWater(snapshots) {
  const fields = Object.keys(snapshots[0][1]);
  return Object.fromEntries(fields.map((field) => [
    field,
    Math.max(...snapshots.map(([, snapshot]) => snapshot[field])),
  ]));
}

async function javascriptRuntimeBaseline({ backend, samples, warmup, width, wasmUrl }) {
  const reference = backend === 'cpu'
    ? await latencyReference(
        stringArgument('reference', DEFAULT_REFERENCE),
        { samples, warmup, width },
      )
    : null;
  const measurementRuns = reference?.measurementRuns || MEASUREMENT_RUNS;
  const restoreConsole = installDiagnosticConsole();
  try {
    const measurements = await collectRuntimeEvidence({
      backend, samples, warmup, width, wasmUrl,
    });
    // Collect after the lifecycle's async frame has returned. Await expressions
    // may retain their resolved caller-owned arrays until that frame completes;
    // those arrays are not runtime resources and must not contaminate the CPU gate.
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
      byteExactIdentity,
      stableBytesDigest,
      expectedBytesDigest,
      wasmCompileProfiles,
      freshCallerOwnedReads,
      readableAfterContextClose,
      acceptedWorkDrained,
      contextCloseIsIdempotent,
      resultCloseIsIdempotent,
    } = measurements;

    const latencyRunsMs = [
      firstLatencySamplesMs,
      ...additionalLatencyRuns(
        measurementRuns, backend, samples, warmup, width, wasmUrl,
      ),
    ];
    const latencyRunMediansMs = latencyRunsMs.map(median);
    const latencySamplesMs = latencyRunsMs.flat();
    const medianMs = median(latencyRunMediansMs);
    const maximumMedianMs = reference
      ? reference.medianMs * (1 + reference.maximumRegressionPercent / 100)
      : null;
    const regressionPercent = reference
      ? ((medianMs / reference.medianMs) - 1) * 100
      : null;
    const latencyGatePassed = reference ? medianMs <= maximumMedianMs : null;
    const alignedOutputBytes = Math.ceil(outputBytes / 256) * 256;
    const oneIdleBytes = memoryDelta(oneIdleContext, retainedBaseline).arrayBufferBytes;
    const twoIdleBytes = memoryDelta(twoIdleContexts, retainedBaseline).arrayBufferBytes;
    const idleContextsClosedBytes =
      memoryDelta(idleContextsClosed, retainedBaseline).arrayBufferBytes;
    const allOwnersClosedBytes = memoryDelta(allOwnersClosed, retainedBaseline).arrayBufferBytes;
    const stableSnapshotBytes =
      memoryDelta(stableResultAllocated, beforeStableResult).arrayBufferBytes;
    const memoryGatePassed = backend === 'cpu' ? oneIdleBytes >= 0 &&
      twoIdleBytes <= Math.max(oneIdleBytes, 0) * 2.2 &&
      stableSnapshotBytes <= alignedOutputBytes * 1.1 &&
      idleContextsClosedBytes <= 0 &&
      allOwnersClosedBytes <= 0 : null;
    const disposalGatePassed = acceptedWorkDrained && contextCloseIsIdempotent &&
      resultCloseIsIdempotent && freshCallerOwnedReads && readableAfterContextClose &&
      byteExactIdentity && stableBytesDigest === expectedBytesDigest;
    const memorySnapshots = [
      ['retainedBaseline', retainedBaseline],
      ['oneIdleContext', oneIdleContext],
      ['twoIdleContexts', twoIdleContexts],
      ['idleContextsClosed', idleContextsClosed],
      ['beforeStableResult', beforeStableResult],
      ['stableResultAllocated', stableResultAllocated],
      ['stableResultWithCallerReads', stableResultWithCallerReads],
      ['stableResultAfterContextClose', stableResultAfterContextClose],
      ['stableResultClosed', stableResultClosed],
      ['allOwnersClosed', allOwnersClosed],
    ];

    const evidence = {
      schema: SCHEMA,
      recordedAt: new Date().toISOString(),
      environment: {
        runtime: 'node',
        version: process.version,
        platform: process.platform,
        architecture: process.arch,
        backend,
        explicitGc: true,
        ...(backend === 'wasm' ? { wasmArtifact: path.relative(process.cwd(), wasmUrl) } : {}),
      },
      workload: {
        operation: 'Identity',
        dtype: 'float32',
        shape: [1, width],
        warmupExecutions: warmup,
        measuredExecutions: samples,
        measurementRuns,
        logicalInputBytes: width * Float32Array.BYTES_PER_ELEMENT,
        logicalOutputBytes: width * Float32Array.BYTES_PER_ELEMENT,
        logicalLiveTensorBytes: width * Float32Array.BYTES_PER_ELEMENT * 2,
      },
      latency: {
        samplesMs: latencySamplesMs,
        runMediansMs: latencyRunMediansMs,
        p50Ms: percentile(latencySamplesMs, 0.5),
        p95Ms: percentile(latencySamplesMs, 0.95),
        varianceMs2: populationVariance(latencySamplesMs),
        medianMs,
        minimumMs: Math.min(...latencySamplesMs),
        maximumMs: Math.max(...latencySamplesMs),
        gate: reference ? {
          referenceSchema: REFERENCE_SCHEMA,
          referenceFile: path.relative(process.cwd(), reference.filename) || '.',
          referenceRecordedAt: reference.recordedAt,
          referenceMedianMs: reference.medianMs,
          maximumRegressionPercent: reference.maximumRegressionPercent,
          maximumMedianMs,
          regressionPercent,
          passed: latencyGatePassed,
        } : null,
      },
      allocation: {
        compilationReportedBytes: compilationReport.allocationBytes,
        logicalInputBytes: width * Float32Array.BYTES_PER_ELEMENT,
        logicalOutputBytes: outputBytes,
        logicalLiveTensorBytes: width * Float32Array.BYTES_PER_ELEMENT + outputBytes,
        resultSnapshotBytes: outputBytes,
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
        sampledProcessHighWater: sampledMemoryHighWater(memorySnapshots),
        sampledProcessHighWaterSemantics:
          'maximum of the forced-GC process snapshots above, not an allocator event trace',
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
        ...(backend === 'wasm' ? {
          wasmLinearMemory: {
            contextProfiles: wasmCompileProfiles,
            maximumContextHeapBytes: Math.max(
              0, ...wasmCompileProfiles.map((profile) => profile.finalHeapBytes || 0),
            ),
            twoConcurrentIdleContextHeapBytes: wasmCompileProfiles.length >= 3
              ? wasmCompileProfiles[1].finalHeapBytes + wasmCompileProfiles[2].finalHeapBytes
              : null,
            totalObservedGrowCount: wasmCompileProfiles.reduce(
              (sum, profile) => sum + (profile.growCount || 0), 0,
            ),
            semantics:
              'opt-in per-context WasmEngine allocation profiles; closed heaps are not counted as live capacity',
          },
        } : {}),
      },
      snapshot: {
        outputBytes,
        alignedOutputBytes,
        freshCallerOwnedReads,
        readableAfterContextClose,
        byteExactIdentity,
        expectedDigest: expectedBytesDigest,
        actualDigest: stableBytesDigest,
      },
      disposal: {
        acceptedWorkDrained,
        contextCloseIsIdempotent,
        resultCloseIsIdempotent,
        allOwnersClosed: backend === 'cpu' ? allOwnersClosedBytes <= 0 : null,
      },
      runtime: {
        compilation: compilationReport,
        execution: executionReport,
      },
      budgets: backend === 'cpu' ? {
        medianOneContextLatencyRegressionPercent: reference.maximumRegressionPercent,
        twoIdleContextsToOneMutableStorageRatio: 2.2,
        idleContextAllocation: 'lazy',
        stableSnapshotBookkeepingPercent: 10,
        closedUnretainedResourcesReturnToBaseline: true,
      } : null,
    };
    process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
    if (backend === 'cpu' && !latencyGatePassed) {
      throw new Error(
        `median one-context latency ${medianMs.toFixed(6)} ms exceeds the committed ` +
        `${maximumMedianMs.toFixed(6)} ms budget`,
      );
    }
    if (backend === 'cpu' && !memoryGatePassed) {
      throw new Error(
        'runtime ownership memory budget failed: ' +
        `oneIdle=${oneIdleBytes}, twoIdle=${twoIdleBytes}, ` +
        `stableSnapshot=${stableSnapshotBytes}, idleClosed=${idleContextsClosedBytes}, ` +
        `allClosed=${allOwnersClosedBytes} array-buffer bytes`,
      );
    }
    if (!disposalGatePassed) {
      throw new Error('runtime lifecycle/disposal or byte-parity evidence failed');
    }
  } finally {
    restoreConsole();
  }
}

async function main() {
  rejectUnknownArguments();
  const backend = backendArgument();
  const samples = integerArgument('samples', 51, 3, 101);
  const warmup = integerArgument('warmup', 10, 0, 20);
  const width = integerArgument('width', 65_536, 1, 1_048_576);
  const wasmUrl = backend === 'wasm'
    ? path.resolve(stringArgument('wasm', ''))
    : null;
  const workerArgument = process.argv.slice(2).find((argument) =>
    argument.startsWith('--latency-worker='));
  if (workerArgument != null) {
    if (workerArgument !== '--latency-worker=1') {
      throw new Error('--latency-worker is an internal flag and must equal 1');
    }
    if (backend === 'native-cpu') {
      throw new Error('--latency-worker does not support native-cpu');
    }
    if (typeof globalThis.gc !== 'function') {
      throw new Error('explicit GC is required; run through npm run baseline:runtime');
    }
    await runLatencyWorker(backend, samples, warmup, width, wasmUrl);
    return;
  }
  if (backend === 'native-cpu') {
    const evidence = await nativeRuntimeBaseline({
      samples,
      warmup,
      width,
      buildDirectory: path.resolve(
        stringArgument('native-build-dir', 'build/cmake'),
      ),
    });
    process.stdout.write(`${JSON.stringify(evidence, null, 2)}\n`);
    return;
  }
  if (typeof globalThis.gc !== 'function') {
    throw new Error('explicit GC is required; run through npm run baseline:runtime');
  }
  await javascriptRuntimeBaseline({ backend, samples, warmup, width, wasmUrl });
}

main().catch((error) => {
  process.stderr.write(`runtime baseline failed: ${String(error?.message || error)}\n`);
  process.exitCode = 1;
});
