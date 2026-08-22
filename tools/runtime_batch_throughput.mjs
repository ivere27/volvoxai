#!/usr/bin/env node

import process from 'node:process';
import { fileURLToPath } from 'node:url';

import {
  ExecutionMode,
  Model,
  VolvoxAI,
  executionModes,
  parseGraphDocument,
} from '../ts/index.js';

const SCHEMA = 'volvoxai.runtime-batch-throughput/v2';
const DIRECT_MODE = executionModes[ExecutionMode.Direct];
const SCHEDULED_MODE = executionModes[ExecutionMode.Scheduled];
const DEFAULT_WASM = fileURLToPath(
  new URL('../dist/0.4.0/volvoxai.wasm', import.meta.url),
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
  const known = new Set(['backend', 'batch', 'inner', 'output', 'samples', 'warmup', 'wasm']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function backendArgument() {
  const backend = stringArgument('backend', 'cpu-js');
  if (backend !== 'cpu-js' && backend !== 'wasm') {
    throw new Error("--backend must be 'cpu-js' or 'wasm'");
  }
  return backend;
}

function percentile(values, fraction) {
  if (values.length === 0) throw new Error('latency measurement produced no samples');
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function latencySummary(values) {
  return Object.freeze({
    samples: values.length,
    median: percentile(values, 0.5),
    p95: percentile(values, 0.95),
  });
}

function memoryFields() {
  const usage = process.memoryUsage();
  return Object.freeze({
    heapUsedBytes: usage.heapUsed,
    arrayBufferBytes: usage.arrayBuffers,
  });
}

function memoryDelta(current, baseline) {
  return Object.freeze({
    heapUsedBytes: current.heapUsedBytes - baseline.heapUsedBytes,
    arrayBufferBytes: current.arrayBufferBytes - baseline.arrayBufferBytes,
  });
}

async function forceGarbageCollection() {
  if (typeof globalThis.gc !== 'function') {
    throw new Error(
      'memory deltas require `node --expose-gc --import tsx ' +
      'tools/runtime_batch_throughput.mjs`',
    );
  }
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
}

async function retainedMemory() {
  await forceGarbageCollection();
  return memoryFields();
}

function deterministicValues(length, multiplier, divisor) {
  return Float32Array.from(
    { length },
    (_, index) => (((index * multiplier) % 257) - 128) / divisor,
  );
}

function matmulSnapshot(batchSize, innerSize, outputSize) {
  const descriptor = Object.freeze({
    name: 'weight',
    dtype: 'float32',
    shape: Object.freeze([innerSize, outputSize]),
  });
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: batchSize } },
    inputs: { x: { dtype: 'float32', shape: ['B', innerSize] } },
    nodes: [{
      id: 'matmul',
      opType: 'MatMul',
      inputs: { input: 'x', weight: 'weight' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: ['B', outputSize] },
      },
      params: {},
    }],
    outputs: ['y'],
  }, [descriptor]);
  return Model.capture({
    graph,
    weights: {
      weight: {
        ...descriptor,
        data: deterministicValues(innerSize * outputSize, 29, 512),
      },
    },
  });
}

function laneInputs(batchSize, innerSize) {
  return Object.freeze(Array.from({ length: batchSize }, (_, lane) => Object.freeze({
    x: Object.freeze({
      data: Float32Array.from(
        { length: innerSize },
        (_, index) => ((((index * 17) + (lane * 31)) % 257) - 128) / 256,
      ),
      shape: Object.freeze([1, innerSize]),
    }),
  })));
}

function diagnosticCounter() {
  let providerInvocations = 0;
  const shapeSignatures = new Set();
  return Object.freeze({
    onDiagnostic(event) {
      if (event.kind !== 'execution') return;
      providerInvocations++;
      if (event.report.shapeSignature) shapeSignatures.add(event.report.shapeSignature);
    },
    reset() {
      providerInvocations = 0;
      shapeSignatures.clear();
    },
    snapshot(logicalRequests) {
      return Object.freeze({
        source: 'public Runtime onDiagnostic execution reports',
        logicalRequests,
        providerInvocations,
        logicalRequestsPerProviderInvocation: providerInvocations === 0
          ? null
          : logicalRequests / providerInvocations,
        providerShapeSignatures: Object.freeze([...shapeSignatures].sort()),
      });
    },
  });
}

function schedulingCounter() {
  let resultReports = 0;
  let trueBackendInvocations = 0;
  const batchSizes = new Set();
  const invocationsByDispatch = new Map();
  return Object.freeze({
    add(result) {
      const scheduling = result.report.backendReport?.scheduling;
      if (!scheduling || typeof scheduling !== 'object') return;
      resultReports++;
      if (Number.isSafeInteger(scheduling.batchSize)) batchSizes.add(scheduling.batchSize);
      if (typeof scheduling.dispatchId === 'string' && scheduling.dispatchId.length > 0 &&
          Number.isSafeInteger(scheduling.trueBackendInvocations) &&
          scheduling.trueBackendInvocations >= 0) {
        trueBackendInvocations += scheduling.trueBackendInvocations;
        invocationsByDispatch.set(
          scheduling.dispatchId,
          (invocationsByDispatch.get(scheduling.dispatchId) ?? 0) +
            scheduling.trueBackendInvocations,
        );
      }
    },
    snapshot() {
      return Object.freeze({
        resultReports,
        batchSizes: Object.freeze([...batchSizes].sort((left, right) => left - right)),
        dispatches: invocationsByDispatch.size,
        trueBackendInvocations,
        trueBackendInvocationsPerDispatch: Object.freeze(
          [...invocationsByDispatch.values()].sort((left, right) => left - right),
        ),
      });
    },
  });
}

async function createHarness({ backend, wasmUrl, snapshot, mode, batchSize }) {
  const diagnostics = diagnosticCounter();
  const runtime = await VolvoxAI.createRuntime({
    backends: [backend],
    ...(backend === 'wasm' ? { wasmUrl } : {}),
    onDiagnostic: diagnostics.onDiagnostic,
    execution: { mode, scheduler: { maxBatchSize: batchSize } },
  });
  try {
    const compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    return Object.freeze({ runtime, compiled, diagnostics });
  } catch (error) {
    await runtime.close();
    throw error;
  }
}

async function readOutputs(results) {
  return Promise.all(results.map((result) => result.output('y').read()));
}

async function closeResults(results) {
  await Promise.all(results.map((result) => result.close()));
}

async function executeIndependentGroup(runtime, compiled, inputs) {
  const results = [];
  const requestLatencyMs = [];
  const groupStarted = performance.now();
  for (const input of inputs) {
    const requestStarted = performance.now();
    const result = await runtime.run(compiled, input);
    requestLatencyMs.push(performance.now() - requestStarted);
    results.push(result);
  }
  return Object.freeze({
    results: Object.freeze(results),
    requestLatencyMs: Object.freeze(requestLatencyMs),
    groupLatencyMs: performance.now() - groupStarted,
  });
}

async function executeScheduledGroup(runtime, compiled, inputs) {
  const groupStarted = performance.now();
  const handles = inputs.map((input) => runtime.submit(compiled, input));
  const completed = await Promise.all(handles.map(async (handle) => {
    const result = await handle.result;
    return Object.freeze({ result, latencyMs: performance.now() - groupStarted });
  }));
  return Object.freeze({
    results: Object.freeze(completed.map(({ result }) => result)),
    requestLatencyMs: Object.freeze(completed.map(({ latencyMs }) => latencyMs)),
    groupLatencyMs: performance.now() - groupStarted,
  });
}

function throughputEvidence(groupLatencyMs, logicalRequests, innerSize, outputSize) {
  const measuredMilliseconds = groupLatencyMs.reduce((sum, value) => sum + value, 0);
  const usefulFlops = logicalRequests * 2 * innerSize * outputSize;
  return Object.freeze({
    measuredMilliseconds,
    logicalRequestsPerSecond: logicalRequests * 1000 / measuredMilliseconds,
    usefulGflopsPerSecond: usefulFlops / (measuredMilliseconds * 1_000_000),
  });
}

async function measureIndependent(options) {
  const harness = await createHarness({ ...options, mode: DIRECT_MODE });
  try {
    let referenceOutputs;
    for (let index = 0; index < options.warmup; index++) {
      const group = await executeIndependentGroup(
        harness.runtime,
        harness.compiled,
        options.inputs,
      );
      if (index === 0) referenceOutputs = await readOutputs(group.results);
      await closeResults(group.results);
    }
    harness.diagnostics.reset();
    const memoryBefore = await retainedMemory();
    const requestLatencyMs = [];
    const groupLatencyMs = [];
    for (let index = 0; index < options.samples; index++) {
      const group = await executeIndependentGroup(
        harness.runtime,
        harness.compiled,
        options.inputs,
      );
      requestLatencyMs.push(...group.requestLatencyMs);
      groupLatencyMs.push(group.groupLatencyMs);
      await closeResults(group.results);
    }
    const memoryAfterOperations = memoryFields();
    const memoryAfterGc = await retainedMemory();
    const logicalRequests = options.samples * options.batchSize;
    const providerEvidence = harness.diagnostics.snapshot(logicalRequests);
    if (providerEvidence.providerInvocations !== logicalRequests) {
      throw new Error(
        `independent B=1 expected ${logicalRequests} provider invocations, ` +
        `observed ${providerEvidence.providerInvocations}`,
      );
    }
    return Object.freeze({
      referenceOutputs: Object.freeze(referenceOutputs),
      report: Object.freeze({
        name: 'independent-direct-b1',
        measuredGroups: options.samples,
        measuredLogicalRequests: logicalRequests,
        logicalRequestLatencyMs: latencySummary(requestLatencyMs),
        groupLatencyMs: latencySummary(groupLatencyMs),
        throughput: throughputEvidence(
          groupLatencyMs,
          logicalRequests,
          options.innerSize,
          options.outputSize,
        ),
        providerEvidence,
        executionInspection: harness.runtime.inspectExecution(),
        memory: Object.freeze({
          afterOperationsDelta: memoryDelta(memoryAfterOperations, memoryBefore),
          retainedAfterGcDelta: memoryDelta(memoryAfterGc, memoryBefore),
        }),
      }),
    });
  } finally {
    await harness.compiled.close();
    await harness.runtime.close();
  }
}

function compareOutputs(reference, observed) {
  if (reference.length !== observed.length) {
    throw new Error('independent and scheduled result lane counts differ');
  }
  const absoluteTolerance = 1e-5;
  const relativeTolerance = 1e-5;
  let totalElements = 0;
  let exactElements = 0;
  let maximumAbsoluteError = 0;
  let maximumRelativeError = 0;
  for (let lane = 0; lane < reference.length; lane++) {
    const expected = reference[lane];
    const actual = observed[lane];
    if (expected.length !== actual.length) {
      throw new Error(`independent and scheduled lane ${lane} output lengths differ`);
    }
    for (let index = 0; index < expected.length; index++) {
      const absoluteError = Math.abs(actual[index] - expected[index]);
      const relativeError = absoluteError / Math.max(Math.abs(expected[index]), 1e-12);
      totalElements++;
      if (Object.is(actual[index], expected[index])) exactElements++;
      maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
      maximumRelativeError = Math.max(maximumRelativeError, relativeError);
      if (absoluteError > absoluteTolerance + relativeTolerance * Math.abs(expected[index])) {
        throw new Error(
          `scheduled batch differs at lane ${lane}, element ${index}: ` +
          `${actual[index]} versus ${expected[index]}`,
        );
      }
    }
  }
  return Object.freeze({
    totalElements,
    exactElements,
    maximumAbsoluteError,
    maximumRelativeError,
    absoluteTolerance,
    relativeTolerance,
  });
}

async function measureScheduled(options, referenceOutputs) {
  const harness = await createHarness({ ...options, mode: SCHEDULED_MODE });
  try {
    let numericalEquality;
    for (let index = 0; index < options.warmup; index++) {
      const group = await executeScheduledGroup(
        harness.runtime,
        harness.compiled,
        options.inputs,
      );
      if (index === 0) {
        numericalEquality = compareOutputs(referenceOutputs, await readOutputs(group.results));
      }
      await closeResults(group.results);
    }
    harness.diagnostics.reset();
    const memoryBefore = await retainedMemory();
    const scheduling = schedulingCounter();
    const requestLatencyMs = [];
    const groupLatencyMs = [];
    for (let index = 0; index < options.samples; index++) {
      const group = await executeScheduledGroup(
        harness.runtime,
        harness.compiled,
        options.inputs,
      );
      requestLatencyMs.push(...group.requestLatencyMs);
      groupLatencyMs.push(group.groupLatencyMs);
      for (const result of group.results) scheduling.add(result);
      await closeResults(group.results);
    }
    const memoryAfterOperations = memoryFields();
    const memoryAfterGc = await retainedMemory();
    const logicalRequests = options.samples * options.batchSize;
    const providerEvidence = harness.diagnostics.snapshot(logicalRequests);
    const schedulerBatchEvidence = scheduling.snapshot();
    if (providerEvidence.providerInvocations !== options.samples) {
      throw new Error(
        `SCHEDULED B=${options.batchSize} expected ${options.samples} provider invocations, ` +
        `observed ${providerEvidence.providerInvocations}`,
      );
    }
    if (schedulerBatchEvidence.resultReports !== logicalRequests ||
        schedulerBatchEvidence.batchSizes.length !== 1 ||
        schedulerBatchEvidence.batchSizes[0] !== options.batchSize ||
        schedulerBatchEvidence.dispatches !== options.samples ||
        schedulerBatchEvidence.trueBackendInvocations !== options.samples ||
        schedulerBatchEvidence.trueBackendInvocationsPerDispatch.some((count) => count !== 1)) {
      throw new Error('Runtime SCHEDULED results lack exact true-batch provider evidence');
    }
    return Object.freeze({
      numericalEquality,
      report: Object.freeze({
        name: 'runtime-scheduled-true-batch',
        measuredGroups: options.samples,
        measuredLogicalRequests: logicalRequests,
        logicalRequestLatencyMs: latencySummary(requestLatencyMs),
        groupLatencyMs: latencySummary(groupLatencyMs),
        throughput: throughputEvidence(
          groupLatencyMs,
          logicalRequests,
          options.innerSize,
          options.outputSize,
        ),
        providerEvidence,
        schedulerBatchEvidence,
        executionInspection: harness.runtime.inspectExecution(),
        memory: Object.freeze({
          afterOperationsDelta: memoryDelta(memoryAfterOperations, memoryBefore),
          retainedAfterGcDelta: memoryDelta(memoryAfterGc, memoryBefore),
        }),
      }),
    });
  } finally {
    await harness.compiled.close();
    await harness.runtime.close();
  }
}

function installDiagnosticConsole() {
  const original = console.log;
  console.log = (...values) => process.stderr.write(`${values.map(String).join(' ')}\n`);
  return () => { console.log = original; };
}

async function main() {
  rejectUnknownArguments();
  const backend = backendArgument();
  const batchSize = integerArgument('batch', 8, 2, 32);
  const innerSize = integerArgument('inner', 512, 8, 4096);
  const outputSize = integerArgument('output', 512, 8, 4096);
  const samples = integerArgument('samples', 21, 3, 1001);
  const warmup = integerArgument('warmup', 5, 1, 100);
  const wasmUrl = stringArgument('wasm', DEFAULT_WASM);
  if (typeof globalThis.gc !== 'function') await forceGarbageCollection();

  const snapshot = matmulSnapshot(batchSize, innerSize, outputSize);
  const inputs = laneInputs(batchSize, innerSize);
  const options = Object.freeze({
    backend,
    wasmUrl,
    snapshot,
    inputs,
    batchSize,
    innerSize,
    outputSize,
    samples,
    warmup,
  });
  const restoreConsole = installDiagnosticConsole();
  let independent;
  let scheduled;
  try {
    independent = await measureIndependent(options);
    scheduled = await measureScheduled(options, independent.referenceOutputs);
  } finally {
    restoreConsole();
  }
  const independentThroughput = independent.report.throughput.logicalRequestsPerSecond;
  const scheduledThroughput = scheduled.report.throughput.logicalRequestsPerSecond;
  process.stdout.write(`${JSON.stringify(Object.freeze({
    schema: SCHEMA,
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      node: process.version,
      platform: process.platform,
      architecture: process.arch,
      backend,
      ...(backend === 'wasm' ? { wasmArtifact: wasmUrl } : {}),
      gcExposed: true,
    }),
    workload: Object.freeze({
      operation: 'MatMul',
      dtype: 'float32',
      logicalInputShape: Object.freeze([1, innerSize]),
      weightShape: Object.freeze([innerSize, outputSize]),
      logicalOutputShape: Object.freeze([1, outputSize]),
      explicitBatchDomain: Object.freeze({ minimum: 1, maximum: batchSize }),
      usefulFlopsPerLogicalRequest: 2 * innerSize * outputSize,
      warmupGroups: warmup,
      measuredGroups: samples,
    }),
    numericalEquality: scheduled.numericalEquality,
    scenarios: Object.freeze([independent.report, scheduled.report]),
    observedComparison: Object.freeze({
      logicalThroughputRatioScheduledOverIndependent:
        scheduledThroughput / independentThroughput,
      medianGroupLatencyRatioScheduledOverIndependent:
        scheduled.report.groupLatencyMs.median / independent.report.groupLatencyMs.median,
      performanceThresholdApplied: false,
    }),
  }), null, 2)}\n`);
}

await main().catch((error) => {
  process.stderr.write(`runtime batch throughput failed: ${String(error?.message || error)}\n`);
  process.exitCode = 1;
});
