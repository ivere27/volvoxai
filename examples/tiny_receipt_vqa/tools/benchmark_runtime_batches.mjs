#!/usr/bin/env node

/*
 * Prove component-level Runtime batching for an imported TinyReceipt VQA
 * encoder or one-token decoder. The input fixture is intentionally external:
 * benchmark_runtime_batches.py authors distinct lanes, runs independent ORT
 * B=1 references, and consumes the full raw outputs written by this worker.
 *
 * This is not a TinyReceiptSplitSession benchmark. That high-level session
 * owns private execution contexts and serializes autoregressive B=1 work.
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { requireTinyReceiptPhysicalAdapterIdentity } from '../TinyReceiptSplitE2E.js';

export const DEFAULT_MAX_BATCH_DELAY_MS = 10;

function fail(message) {
  throw new Error(`[tiny_receipt_vqa/benchmark_runtime_batches] ${message}`);
}

function parseArguments(argv) {
  const options = {
    package: null,
    role: null,
    inputs: null,
    outputDirectory: null,
    backend: null,
    mode: null,
    concurrency: 2,
    maxBatchDelayMs: DEFAULT_MAX_BATCH_DELAY_MS,
    warmup: 1,
    repeat: 1,
    api: null,
    wasm: null,
    report: null,
    includePrivateArtifacts: false,
    adapter: 'high-performance',
    requireAdapter: null,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const take = () => {
      const value = argv[++index];
      if (value === undefined) fail(`${flag} needs a value`);
      return value;
    };
    if (flag === '--package') options.package = take();
    else if (flag === '--role') options.role = take();
    else if (flag === '--inputs') options.inputs = take();
    else if (flag === '--output-directory') options.outputDirectory = take();
    else if (flag === '--backend') options.backend = take();
    else if (flag === '--mode') options.mode = take();
    else if (flag === '--concurrency') options.concurrency = Number(take());
    else if (flag === '--max-batch-delay-ms') options.maxBatchDelayMs = Number(take());
    else if (flag === '--warmup') options.warmup = Number(take());
    else if (flag === '--repeat') options.repeat = Number(take());
    else if (flag === '--api') options.api = take();
    else if (flag === '--wasm') options.wasm = take();
    else if (flag === '--report') options.report = take();
    else if (flag === '--include-private-artifacts') options.includePrivateArtifacts = true;
    else if (flag === '--adapter') options.adapter = take();
    else if (flag === '--require-adapter') options.requireAdapter = take();
    else fail(`unknown option ${flag}`);
  }
  for (const name of ['package', 'role', 'inputs', 'outputDirectory', 'backend']) {
    if (!options[name]) fail(`--${name.replaceAll(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)} is required`);
  }
  if (!['encoder', 'decoder'].includes(options.role)) {
    fail("--role must be 'encoder' or 'decoder'");
  }
  if (!['cpu-js', 'wasm', 'webgpu'].includes(options.backend)) {
    fail("--backend must be 'cpu-js', 'wasm', or 'webgpu'");
  }
  for (const [name, minimum] of [['concurrency', 1], ['warmup', 0], ['repeat', 1]]) {
    if (!Number.isSafeInteger(options[name]) || options[name] < minimum) {
      fail(`--${name} must be an integer >= ${minimum}`);
    }
  }
  if (!Number.isFinite(options.maxBatchDelayMs) || options.maxBatchDelayMs < 0 ||
      options.maxBatchDelayMs > 60_000) {
    fail('--max-batch-delay-ms must be finite and between 0 and 60000');
  }
  if (options.backend === 'wasm' && !options.wasm) fail('--wasm is required for WASM');
  if (!['default', 'high-performance', 'low-power'].includes(options.adapter)) {
    fail("--adapter must be 'default', 'high-performance', or 'low-power'");
  }
  if (options.backend === 'webgpu') {
    if (typeof options.requireAdapter !== 'string' || options.requireAdapter.length === 0) {
      fail('--require-adapter is required for physical WebGPU measurement');
    }
  } else if (options.requireAdapter !== null || options.adapter !== 'high-performance') {
    fail('--adapter and --require-adapter apply only to WebGPU');
  }
  return Object.freeze(options);
}

export { parseArguments };

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha256File(path) {
  return sha256Bytes(readFileSync(path));
}

function runtimeIdentity(backend) {
  const deno = globalThis.Deno;
  if (!deno) return Object.freeze({ name: 'node', version: process.version });
  const identity = {
    name: 'deno',
    version: deno.version.deno,
    v8: deno.version.v8,
    typescript: deno.version.typescript,
    target: deno.build.target,
  };
  if (backend === 'webgpu') {
    let configured;
    try {
      configured = deno.env.get('DENO_WEBGPU_BACKEND');
    } catch {
      fail('Deno WebGPU measurement requires --allow-env=DENO_WEBGPU_BACKEND');
    }
    if (configured !== 'vulkan') {
      fail('Deno WebGPU measurement requires DENO_WEBGPU_BACKEND=vulkan');
    }
    identity.webgpuBackend = configured;
  }
  return Object.freeze(identity);
}

function overrideWebGPUAdapterPreference(gpu, preference) {
  if (preference === 'default') return () => {};
  const descriptor = Object.getOwnPropertyDescriptor(gpu, 'requestAdapter');
  const original = gpu?.requestAdapter;
  if (typeof original !== 'function') fail('WebGPU requestAdapter is unavailable');
  const bound = original.bind(gpu);
  gpu.requestAdapter = (request = {}) => bound({ ...request, powerPreference: preference });
  return () => {
    if (descriptor) Object.defineProperty(gpu, 'requestAdapter', descriptor);
    else if (!delete gpu.requestAdapter) fail('could not restore WebGPU requestAdapter');
  };
}

function compactBatchSemantics(compiled) {
  const evidence = compiled?.report?.batchSemantics;
  if (!evidence || typeof evidence !== 'object' ||
      typeof evidence.protocol !== 'string' ||
      typeof evidence.supported !== 'boolean' ||
      !Number.isSafeInteger(evidence.coveredNodes) ||
      typeof evidence.graphFingerprint !== 'string') {
    fail('compiled report does not expose valid typed batch-semantics evidence');
  }
  return Object.freeze({
    protocol: evidence.protocol,
    supported: evidence.supported,
    batchSymbol: evidence.batchSymbol ?? null,
    coveredNodes: evidence.coveredNodes,
    graphFingerprintSha256: sha256Bytes(Buffer.from(evidence.graphFingerprint, 'utf8')),
    reason: evidence.reason ?? null,
    failedNode: evidence.failedNode ?? null,
    failedTensor: evidence.failedTensor ?? null,
  });
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function latencySummary(values) {
  return Object.freeze({
    samples: values.length,
    minMs: Math.min(...values),
    medianMs: percentile(values, 0.5),
    maxMs: Math.max(...values),
  });
}

const fileFetch = async (source) => {
  const buffer = readFileSync(source.startsWith('file:') ? fileURLToPath(source) : source);
  return {
    ok: true,
    text: async () => buffer.toString('utf8'),
    json: async () => JSON.parse(buffer.toString('utf8')),
    arrayBuffer: async () => buffer.buffer.slice(
      buffer.byteOffset, buffer.byteOffset + buffer.byteLength,
    ),
  };
};

const ARRAY_TYPES = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

function product(shape) {
  return shape.reduce((total, extent) => total * extent, 1);
}

function loadArray(descriptor) {
  if (!descriptor || typeof descriptor !== 'object' ||
      !Array.isArray(descriptor.shape) || !descriptor.path || !descriptor.dtype) {
    fail('input tensor descriptor is malformed');
  }
  const Type = ARRAY_TYPES[descriptor.dtype];
  if (!Type) fail(`unsupported fixture dtype '${descriptor.dtype}'`);
  const source = readFileSync(descriptor.path);
  if (source.byteLength % Type.BYTES_PER_ELEMENT !== 0) {
    fail(`input ${descriptor.path} byte length is not aligned to ${descriptor.dtype}`);
  }
  const data = new Type(
    source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength),
  );
  if (data.length !== product(descriptor.shape)) {
    fail(`input ${descriptor.path} length does not match its shape`);
  }
  return Object.freeze({ data, shape: Object.freeze([...descriptor.shape]) });
}

function loadInputFixture(path, concurrency) {
  const document = JSON.parse(readFileSync(path, 'utf8'));
  if (document.schema !== 'volvoxai.tiny-receipt-vqa-runtime-batch-inputs/v1' ||
      !Array.isArray(document.outputs) || !Array.isArray(document.lanes) ||
      document.lanes.length !== concurrency) {
    fail('input fixture has the wrong schema or lane count');
  }
  const lanes = document.lanes.map((lane) => Object.freeze(Object.fromEntries(
    Object.entries(lane).map(([name, descriptor]) => [name, loadArray(descriptor)]),
  )));
  return Object.freeze({
    outputs: Object.freeze([...document.outputs]),
    lanes: Object.freeze(lanes),
  });
}

function createDiagnosticCounter() {
  let executions = [];
  return Object.freeze({
    onDiagnostic(event) {
      if (event.kind !== 'execution') return;
      executions.push(Object.freeze({
        backend: event.backend ?? event.report?.backend ?? null,
        shapeSignature: event.report?.shapeSignature ?? null,
        scheduling: event.report?.scheduling ?? null,
      }));
    },
    reset() { executions = []; },
    snapshot(logicalRequests) {
      return Object.freeze({
        source: 'Runtime onDiagnostic execution reports',
        logicalRequests,
        physicalInvocations: executions.length,
        logicalRequestsPerPhysicalInvocation: executions.length === 0
          ? null : logicalRequests / executions.length,
        executions: Object.freeze([...executions]),
      });
    },
  });
}

function createSchedulingCounter() {
  const dispatches = new Map();
  const batchSizes = new Set();
  let resultReports = 0;
  let trueBackendInvocations = 0;
  return Object.freeze({
    add(result) {
      const scheduling = result.report.backendReport?.scheduling;
      if (!scheduling || typeof scheduling !== 'object') return;
      resultReports++;
      if (Number.isSafeInteger(scheduling.batchSize)) batchSizes.add(scheduling.batchSize);
      if (typeof scheduling.dispatchId === 'string' && scheduling.dispatchId.length > 0) {
        const current = dispatches.get(scheduling.dispatchId) ?? {
          resultReports: 0, trueBackendInvocations: 0, batchSize: scheduling.batchSize ?? null,
        };
        current.resultReports++;
        if (Number.isSafeInteger(scheduling.trueBackendInvocations)) {
          current.trueBackendInvocations += scheduling.trueBackendInvocations;
          trueBackendInvocations += scheduling.trueBackendInvocations;
        }
        dispatches.set(scheduling.dispatchId, current);
      }
    },
    snapshot() {
      return Object.freeze({
        source: 'ExecutionResult backendReport.scheduling (additive across lane reports)',
        resultReports,
        observedBatchSizes: Object.freeze([...batchSizes].sort((a, b) => a - b)),
        dispatchCount: dispatches.size,
        dispatches: Object.freeze(Object.fromEntries(dispatches)),
        trueBackendInvocations,
      });
    },
  });
}

async function executeGroup(compiled, lanes, mode, directMode) {
  const started = performance.now();
  if (mode === directMode) {
    const results = [];
    for (const inputs of lanes) {
      results.push(await compiled.run(inputs, { mode: directMode }));
    }
    return Object.freeze({ results: Object.freeze(results), elapsedMs: performance.now() - started });
  }
  const handles = lanes.map((inputs) => compiled.submit(inputs));
  const results = await Promise.all(handles.map((handle) => handle.result));
  return Object.freeze({ results: Object.freeze(results), elapsedMs: performance.now() - started });
}

function cloneArray(value) {
  return new value.constructor(value);
}

async function readGroup(results, outputNames, scheduling) {
  try {
    return await Promise.all(results.map(async (result) => {
      scheduling.add(result);
      const outputs = {};
      for (const name of outputNames) {
        const tensor = result.output(name);
        const data = await tensor.read();
        outputs[name] = Object.freeze({
          data: cloneArray(data),
          shape: Object.freeze([...tensor.shape]),
          dtype: tensor.dtype,
        });
      }
      return Object.freeze(outputs);
    }));
  } finally {
    await Promise.all(results.map((result) => result.close()));
  }
}

function compareOutputSets(reference, actual) {
  let maximumAbsoluteDifference = 0;
  let maximumRelativeDifference = 0;
  const lanes = [];
  for (let lane = 0; lane < reference.length; lane++) {
    const outputs = {};
    for (const [name, left] of Object.entries(reference[lane])) {
      const right = actual[lane]?.[name];
      if (!right || left.dtype !== right.dtype ||
          left.shape.join(',') !== right.shape.join(',') ||
          left.data.length !== right.data.length) {
        fail(`same-backend lane ${lane} output '${name}' changed ABI under scheduling`);
      }
      let maxAbs = 0;
      let maxRel = 0;
      let exact = true;
      for (let index = 0; index < left.data.length; index++) {
        if (left.data[index] !== right.data[index]) exact = false;
        if (left.dtype === 'float32') {
          if (!Number.isFinite(left.data[index]) || !Number.isFinite(right.data[index])) {
            fail(`same-backend lane ${lane} output '${name}' is non-finite`);
          }
          const absolute = Math.abs(left.data[index] - right.data[index]);
          const relative = absolute / Math.max(Math.abs(left.data[index]), 1e-6);
          maxAbs = Math.max(maxAbs, absolute);
          maxRel = Math.max(maxRel, relative);
        }
      }
      if (left.dtype !== 'float32' && !exact) {
        fail(`same-backend lane ${lane} integer output '${name}' changed under scheduling`);
      }
      if (left.dtype === 'float32' && maxAbs > 1e-5 && maxRel > 1e-5) {
        fail(`same-backend lane ${lane} output '${name}' differs by abs=${maxAbs}, rel=${maxRel}`);
      }
      maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, maxAbs);
      maximumRelativeDifference = Math.max(maximumRelativeDifference, maxRel);
      outputs[name] = Object.freeze({ exact, maxAbs, maxRel });
    }
    lanes.push(Object.freeze(outputs));
  }
  return Object.freeze({
    status: 'passed',
    maximumAbsoluteDifference,
    maximumRelativeDifference,
    lanes: Object.freeze(lanes),
  });
}

function writeOutputs(route, outputs, outputDirectory, includePrivateArtifacts) {
  const lanes = [];
  for (let lane = 0; lane < outputs.length; lane++) {
    const descriptors = {};
    for (const [name, tensor] of Object.entries(outputs[lane])) {
      const filename = `${route}-lane${lane}-${name}.raw`;
      const path = resolve(outputDirectory, filename);
      const bytes = Buffer.from(
        tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength,
      );
      writeFileSync(path, bytes);
      descriptors[name] = Object.freeze({
        dtype: tensor.dtype,
        shape: tensor.shape,
        elements: tensor.data.length,
        bytes: bytes.byteLength,
        ...(includePrivateArtifacts ? {
          path,
          sha256: sha256Bytes(bytes),
        } : {}),
      });
    }
    lanes.push(Object.freeze(descriptors));
  }
  return Object.freeze(lanes);
}

async function runRoute({
  compiled, diagnostics, fixture, mode, directMode, warmup, repeat, backend,
}) {
  for (let index = 0; index < warmup; index++) {
    const group = await executeGroup(compiled, fixture.lanes, mode, directMode);
    await readGroup(group.results, fixture.outputs, createSchedulingCounter());
  }
  diagnostics.reset();
  const scheduling = createSchedulingCounter();
  const latencies = [];
  let representative = null;
  let deterministic = true;
  for (let index = 0; index < repeat; index++) {
    const synchronizedStarted = performance.now();
    const group = await executeGroup(compiled, fixture.lanes, mode, directMode);
    const outputs = await readGroup(group.results, fixture.outputs, scheduling);
    latencies.push(backend === 'webgpu'
      ? performance.now() - synchronizedStarted
      : group.elapsedMs);
    if (representative === null) representative = outputs;
    else if (compareOutputSets(representative, outputs).maximumAbsoluteDifference !== 0) {
      deterministic = false;
    }
  }
  return Object.freeze({
    outputs: representative,
    latency: latencySummary(latencies),
    deterministic,
    physicalEvidence: diagnostics.snapshot(repeat * fixture.lanes.length),
    schedulingEvidence: scheduling.snapshot(),
  });
}

function validatePhysicalEvidence(route, { mode, directMode, concurrency, repeat, maxBatchSize }) {
  const shouldBatch = mode !== directMode && concurrency > 1 && maxBatchSize >= concurrency;
  const expectedInvocations = repeat * (shouldBatch ? 1 : concurrency);
  if (route.physicalEvidence.physicalInvocations !== expectedInvocations) {
    fail(`expected ${expectedInvocations} physical invocations, observed ${route.physicalEvidence.physicalInvocations}`);
  }
  const expectedBatch = shouldBatch ? concurrency : 1;
  // maxBatchSize=1 routes are intentionally unable to create a scheduler
  // batch and may go straight to the provider without a per-result scheduling
  // record. Their public execution diagnostics remain the physical proof.
  const schedulingRequired = shouldBatch;
  if (schedulingRequired &&
      route.schedulingEvidence.observedBatchSizes.join(',') !== String(expectedBatch)) {
    fail(`expected scheduling batchSize=${expectedBatch}, observed ${route.schedulingEvidence.observedBatchSizes}`);
  }
  if (schedulingRequired &&
      route.schedulingEvidence.trueBackendInvocations !== expectedInvocations) {
    fail(`expected additive trueBackendInvocations=${expectedInvocations}, observed ${route.schedulingEvidence.trueBackendInvocations}`);
  }
  return Object.freeze({
    shouldBatch,
    expectedInvocations,
    expectedBatch,
    schedulingRequired,
  });
}

export async function benchmarkRuntimeBatches(options) {
  const runtimeInfo = runtimeIdentity(options.backend);
  const packageDirectory = resolve(options.package);
  const manifestPath = resolve(packageDirectory, 'package_manifest.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const graph = manifest.graphs?.[options.role];
  const maxBatchSize = manifest.shape_contract?.dimensions?.B?.max;
  if (!graph || !Number.isSafeInteger(maxBatchSize) || maxBatchSize < 1) {
    fail('package manifest does not expose the selected graph and B contract');
  }
  if (options.concurrency > 8) fail('--concurrency exceeds the supported audit bound 8');
  if (options.concurrency > maxBatchSize) {
    fail(`--concurrency exceeds the package batch maximum ${maxBatchSize}`);
  }
  const fixture = loadInputFixture(options.inputs, options.concurrency);
  mkdirSync(options.outputDirectory, { recursive: true });
  const apiPath = options.api ? resolve(options.api) : null;
  const wasmPath = options.backend === 'wasm' ? resolve(options.wasm) : null;
  const releasePackage = JSON.parse(readFileSync(
    fileURLToPath(new URL('../../../package.json', import.meta.url)),
    'utf8',
  ));

  const api = options.api
    ? await import(pathToFileURL(apiPath).href)
    : await import('../../../ts/index.js');
  const directMode = api.executionModes?.[api.ExecutionMode?.Direct];
  const scheduledMode = api.executionModes?.[api.ExecutionMode?.Scheduled];
  const mode = options.mode ?? scheduledMode;
  if (!Array.isArray(api.executionModes) || !api.executionModes.includes(mode) ||
      directMode === undefined || scheduledMode === undefined) {
    fail(`--mode must be one of the generated execution modes: ${
      api.executionModes?.join(', ') ?? 'unavailable API contract'}`);
  }
  const diagnostics = createDiagnosticCounter();
  let restoreAdapter = () => {};
  if (options.backend === 'webgpu') {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) fail('WebGPU is unavailable');
    restoreAdapter = overrideWebGPUAdapterPreference(gpu, options.adapter);
  }
  let runtime;
  let compiled;
  let selectedDevice = null;
  const scheduler = Object.freeze({
    maxBatchSize,
    maxBatchDelayMs: options.maxBatchDelayMs,
  });
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [options.backend],
      ...(options.backend === 'wasm'
        ? { wasmUrl: pathToFileURL(resolve(options.wasm)).href }
        : {}),
      execution: {
        mode,
        scheduler,
      },
      onDiagnostic: diagnostics.onDiagnostic,
    });
    const snapshot = await api.Model.load(
      pathToFileURL(resolve(packageDirectory, graph.weights.path)).href,
      {
        graphUrl: pathToFileURL(resolve(packageDirectory, graph.graph.path)).href,
        fetch: fileFetch,
      },
    );
    compiled = await runtime.compile(snapshot, {
      backend: {
        mode: 'require', backend: options.backend, operatorFallback: 'forbid',
      },
    });
    if (options.backend === 'webgpu') {
      if (compiled.backend !== 'webgpu') {
        fail(`required WebGPU compiled on '${compiled.backend ?? 'unknown'}'`);
      }
      selectedDevice = requireTinyReceiptPhysicalAdapterIdentity(
        compiled.report?.selectedDevice,
        options.requireAdapter,
      );
    }
    const batchSemantics = compactBatchSemantics(compiled);
    const independent = await runRoute({
      compiled, diagnostics, fixture, mode: directMode, directMode,
      warmup: options.warmup, repeat: options.repeat, backend: options.backend,
    });
    const independentContract = validatePhysicalEvidence(independent, {
      mode: directMode, directMode, concurrency: options.concurrency, repeat: options.repeat,
      maxBatchSize,
    });
    const scheduled = await runRoute({
      compiled, diagnostics, fixture, mode, directMode,
      warmup: options.warmup, repeat: options.repeat, backend: options.backend,
    });
    const scheduledContract = validatePhysicalEvidence(scheduled, {
      mode, directMode, concurrency: options.concurrency, repeat: options.repeat,
      maxBatchSize,
    });
    const parity = compareOutputSets(independent.outputs, scheduled.outputs);
    const report = Object.freeze({
      schema: 'volvoxai.tiny-receipt-vqa-runtime-batches/v2',
      scope: 'component-only; TinyReceiptSplitSession remains serialized B=1',
      environment: Object.freeze({
        runtime: runtimeInfo,
        platform: process.platform,
        architecture: process.arch,
        backend: options.backend,
        selectedDevice,
        releaseVersion: releasePackage.version ?? null,
        apiSha256: apiPath ? sha256File(apiPath) : null,
        wasmSha256: wasmPath ? sha256File(wasmPath) : null,
        ...(options.includePrivateArtifacts ? {
          api: apiPath ?? 'ts/index.js',
          wasm: wasmPath,
        } : {}),
      }),
      package: Object.freeze({
        manifestSha256: sha256File(manifestPath),
        variant: manifest.variant,
        role: options.role,
        graphSha256: sha256File(resolve(packageDirectory, graph.graph.path)),
        weightsSha256: sha256File(resolve(packageDirectory, graph.weights.path)),
        contractMaxBatchSize: maxBatchSize,
        ...(options.includePrivateArtifacts ? { path: packageDirectory } : {}),
      }),
      batchSemantics,
      mode,
      scheduler,
      concurrency: options.concurrency,
      warmupGroupsPerRoute: options.warmup,
      measuredGroupsPerRoute: options.repeat,
      timingBoundary: options.backend === 'webgpu'
        ? 'all required output readbacks and result close'
        : 'provider execution result publication',
      independent: Object.freeze({
        contract: independentContract,
        latency: independent.latency,
        deterministic: independent.deterministic,
        physicalEvidence: independent.physicalEvidence,
        schedulingEvidence: independent.schedulingEvidence,
        outputs: writeOutputs(
          'independent', independent.outputs, options.outputDirectory,
          options.includePrivateArtifacts,
        ),
      }),
      scheduled: Object.freeze({
        contract: scheduledContract,
        latency: scheduled.latency,
        deterministic: scheduled.deterministic,
        physicalEvidence: scheduled.physicalEvidence,
        schedulingEvidence: scheduled.schedulingEvidence,
        outputs: writeOutputs(
          'scheduled', scheduled.outputs, options.outputDirectory,
          options.includePrivateArtifacts,
        ),
      }),
      sameBackendParity: parity,
      privacy: Object.freeze({
        machinePathsIncluded: options.includePrivateArtifacts,
        outputDigestsIncluded: options.includePrivateArtifacts,
        rawOutputFilesWritten: true,
      }),
      executionInspection: runtime.inspectExecution(),
    });
    if (options.report) {
      mkdirSync(dirname(resolve(options.report)), { recursive: true });
      writeFileSync(resolve(options.report), `${JSON.stringify(report, null, 2)}\n`);
    }
    return report;
  } finally {
    await compiled?.close().catch(() => undefined);
    await runtime?.close().catch(() => undefined);
    restoreAdapter();
  }
}

const runtimeArguments = globalThis.Deno?.args ?? process.argv.slice(2);
const invokedPath = globalThis.Deno
  ? globalThis.Deno.mainModule
  : pathToFileURL(process.argv[1] ?? '').href;
if (import.meta.url === invokedPath) {
  const options = parseArguments(runtimeArguments);
  const originalLog = console.log;
  console.log = (...values) => process.stderr.write(`${values.map(String).join(' ')}\n`);
  benchmarkRuntimeBatches(options).then((report) => {
    console.log = originalLog;
    process.stdout.write(`${JSON.stringify(report)}\n`);
  }).catch((error) => {
    console.log = originalLog;
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
