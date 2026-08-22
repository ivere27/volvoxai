#!/usr/bin/env node

import assert from 'node:assert/strict';
import process from 'node:process';

import {
  ExecutionMode,
  Model,
  VolvoxAI,
  executionModes,
  parseGraphDocument,
} from '../ts/index.js';

const SCHEMA = 'volvoxai.runtime-scheduler-baseline/v2';
const DIRECT_MODE = executionModes[ExecutionMode.Direct];
const SCHEDULED_MODE = executionModes[ExecutionMode.Scheduled];

function integerArgument(name, fallback, minimum, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw === undefined ? fallback : Number(raw.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['samples', 'warmup', 'width', 'concurrency']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
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
      'allocation evidence requires V8 GC exposure; run ' +
      '`node --expose-gc --import tsx tools/runtime_scheduler_baseline.mjs`',
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

function explicitBatchIdentitySnapshot(width, maximumBatchSize) {
  const graph = parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: maximumBatchSize } },
    inputs: { x: { dtype: 'float32', shape: ['B', width] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: ['B', width] } },
      params: {},
    }],
    outputs: ['y'],
  }, []);
  return Model.capture({ graph, weights: {} });
}

function logicalInput(width, lane = 0) {
  const data = Float32Array.from(
    { length: width },
    (_, index) => lane + ((index % 257) / 257),
  );
  return Object.freeze({
    x: Object.freeze({ data, shape: Object.freeze([1, width]) }),
  });
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

async function assertIdentity(result, inputs) {
  const observed = await result.output('y').read();
  if (byteDigest(observed) !== byteDigest(inputs.x.data)) {
    throw new Error('CPU explicit-batch Identity returned incorrect output');
  }
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
  let reportCount = 0;
  let trueBackendInvocations = 0;
  const batchSizes = new Set();
  const invocationsByDispatch = new Map();
  return Object.freeze({
    add(result) {
      const scheduling = result.report.backendReport?.scheduling;
      if (!scheduling || typeof scheduling !== 'object') return;
      reportCount++;
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
        resultReports: reportCount,
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

async function createCPUHarness(snapshot, mode, maximumBatchSize) {
  const diagnostics = diagnosticCounter();
  const runtime = await VolvoxAI.createRuntime({
    backends: ['cpu-js'],
    onDiagnostic: diagnostics.onDiagnostic,
    execution: { mode, scheduler: { maxBatchSize: maximumBatchSize } },
  });
  try {
    const compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'cpu-js', operatorFallback: 'forbid' },
    });
    return Object.freeze({ runtime, compiled, diagnostics });
  } catch (error) {
    await runtime.close();
    throw error;
  }
}

async function measureSequential({
  name,
  snapshot,
  input,
  mode,
  samples,
  warmup,
  maximumBatchSize,
  directContext,
}) {
  const harness = await createCPUHarness(snapshot, mode, maximumBatchSize);
  let context;
  try {
    if (directContext) {
      if (typeof harness.compiled.createContext !== 'function') {
        return Object.freeze({
          name,
          available: false,
          reason: 'CompiledModel.createContext is not exposed by this build',
        });
      }
      context = await harness.compiled.createContext();
    }
    const execute = directContext
      ? () => context.execute(input)
      : () => harness.runtime.run(harness.compiled, input);

    for (let index = 0; index < warmup; index++) {
      const result = await execute();
      if (index === 0) await assertIdentity(result, input);
      await result.close();
    }
    const inspectionBefore = harness.runtime.inspectExecution();
    harness.diagnostics.reset();
    const memoryBefore = await retainedMemory();
    const scheduling = schedulingCounter();
    const latencyMs = [];
    for (let index = 0; index < samples; index++) {
      const started = performance.now();
      const result = await execute();
      latencyMs.push(performance.now() - started);
      scheduling.add(result);
      await result.close();
    }
    const memoryAfterOperations = memoryFields();
    const memoryAfterGc = await retainedMemory();
    const inspectionAfter = harness.runtime.inspectExecution();
    if (name === 'runtime-direct-b1') {
      assert.equal(inspectionBefore.schedulerAllocated, false,
        'DIRECT allocated RuntimeScheduler during warmup');
      assert.equal(inspectionAfter.schedulerAllocated, false,
        'DIRECT allocated RuntimeScheduler during measured execution');
    }
    return Object.freeze({
      name,
      available: true,
      logicalBatchSize: 1,
      measuredLogicalRequests: samples,
      latencyMs: latencySummary(latencyMs),
      providerEvidence: harness.diagnostics.snapshot(samples),
      schedulerBatchEvidence: scheduling.snapshot(),
      executionInspection: Object.freeze({ before: inspectionBefore, after: inspectionAfter }),
      memory: Object.freeze({
        afterOperationsDelta: memoryDelta(memoryAfterOperations, memoryBefore),
        retainedAfterGcDelta: memoryDelta(memoryAfterGc, memoryBefore),
      }),
    });
  } finally {
    await context?.close();
    await harness.compiled.close();
    await harness.runtime.close();
  }
}

async function executeConcurrentBurst(runtime, compiled, inputs) {
  const started = performance.now();
  const handles = inputs.map((input) => runtime.submit(compiled, input));
  const completed = await Promise.all(handles.map(async (handle) => {
    const result = await handle.result;
    return Object.freeze({ result, latencyMs: performance.now() - started });
  }));
  return Object.freeze({ completed, burstLatencyMs: performance.now() - started });
}

async function closeCompleted(completed) {
  await Promise.all(completed.map(({ result }) => result.close()));
}

async function measureConcurrent({
  snapshot,
  inputs,
  samples,
  warmup,
  maximumBatchSize,
}) {
  const harness = await createCPUHarness(snapshot, SCHEDULED_MODE, maximumBatchSize);
  try {
    for (let index = 0; index < warmup; index++) {
      const burst = await executeConcurrentBurst(harness.runtime, harness.compiled, inputs);
      if (index === 0) {
        await Promise.all(burst.completed.map(({ result }, lane) =>
          assertIdentity(result, inputs[lane])));
      }
      await closeCompleted(burst.completed);
    }
    const inspectionBefore = harness.runtime.inspectExecution();
    harness.diagnostics.reset();
    const memoryBefore = await retainedMemory();
    const scheduling = schedulingCounter();
    const logicalLatencyMs = [];
    const burstLatencyMs = [];
    for (let index = 0; index < samples; index++) {
      const burst = await executeConcurrentBurst(harness.runtime, harness.compiled, inputs);
      burstLatencyMs.push(burst.burstLatencyMs);
      for (const { result, latencyMs } of burst.completed) {
        logicalLatencyMs.push(latencyMs);
        scheduling.add(result);
      }
      await closeCompleted(burst.completed);
    }
    const memoryAfterOperations = memoryFields();
    const memoryAfterGc = await retainedMemory();
    const inspectionAfter = harness.runtime.inspectExecution();
    const batchEvidence = scheduling.snapshot();
    if (!batchEvidence.batchSizes.includes(inputs.length)) {
      throw new Error(
        `SCHEDULED concurrent route did not produce the requested B=${inputs.length} batch`,
      );
    }
    return Object.freeze({
      name: 'runtime-scheduled-concurrent',
      available: true,
      logicalBatchSize: 1,
      concurrentLogicalRequests: inputs.length,
      measuredBursts: samples,
      measuredLogicalRequests: samples * inputs.length,
      logicalRequestLatencyMs: latencySummary(logicalLatencyMs),
      burstLatencyMs: latencySummary(burstLatencyMs),
      providerEvidence: harness.diagnostics.snapshot(samples * inputs.length),
      schedulerBatchEvidence: batchEvidence,
      executionInspection: Object.freeze({ before: inspectionBefore, after: inspectionAfter }),
      memory: Object.freeze({
        afterOperationsDelta: memoryDelta(memoryAfterOperations, memoryBefore),
        retainedAfterGcDelta: memoryDelta(memoryAfterGc, memoryBefore),
      }),
    });
  } finally {
    await harness.compiled.close();
    await harness.runtime.close();
  }
}

async function main() {
  rejectUnknownArguments();
  const samples = integerArgument('samples', 101, 3, 10001);
  const warmup = integerArgument('warmup', 20, 1, 10000);
  const width = integerArgument('width', 4096, 1, 1_048_576);
  const concurrency = integerArgument('concurrency', 4, 2, 32);
  if (typeof globalThis.gc !== 'function') await forceGarbageCollection();

  const snapshot = explicitBatchIdentitySnapshot(width, concurrency);
  const input = logicalInput(width);
  const concurrentInputs = Object.freeze(
    Array.from({ length: concurrency }, (_, lane) => logicalInput(width, lane)),
  );
  const direct = await measureSequential({
    name: 'direct-warm-context-b1',
    snapshot,
    input,
    mode: DIRECT_MODE,
    samples,
    warmup,
    maximumBatchSize: concurrency,
    directContext: true,
  });
  const directRuntime = await measureSequential({
    name: 'runtime-direct-b1',
    snapshot,
    input,
    mode: DIRECT_MODE,
    samples,
    warmup,
    maximumBatchSize: concurrency,
    directContext: false,
  });
  const scheduledOne = await measureSequential({
    name: 'runtime-scheduled-b1',
    snapshot,
    input,
    mode: SCHEDULED_MODE,
    samples,
    warmup,
    maximumBatchSize: concurrency,
    directContext: false,
  });
  const scheduledBatch = await measureConcurrent({
    snapshot,
    inputs: concurrentInputs,
    samples,
    warmup,
    maximumBatchSize: concurrency,
  });

  process.stderr.write(
    `DIRECT schedulerAllocated=${directRuntime.executionInspection.after.schedulerAllocated}\n`,
  );
  process.stdout.write(`${JSON.stringify(Object.freeze({
    schema: SCHEMA,
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      node: process.version,
      platform: process.platform,
      architecture: process.arch,
      backend: 'public-runtime/cpu-js',
      gcExposed: true,
    }),
    workload: Object.freeze({
      operation: 'Identity',
      dtype: 'float32',
      shape: Object.freeze([1, width]),
      explicitBatchDomain: Object.freeze({ minimum: 1, maximum: concurrency }),
      warmupExecutionsOrBursts: warmup,
      measuredExecutionsOrBursts: samples,
    }),
    scenarios: Object.freeze([direct, directRuntime, scheduledOne, scheduledBatch]),
  }), null, 2)}\n`);
}

await main();
