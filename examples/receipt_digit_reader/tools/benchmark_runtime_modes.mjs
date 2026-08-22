#!/usr/bin/env node

/*
 * Compare Runtime DIRECT and SCHEDULED execution on one imported
 * receipt-digit-reader package.  The provider execution diagnostic and each
 * result's scheduling report are counted together: elapsed time alone cannot
 * establish that concurrent logical requests became one physical batch.
 *
 * A DIRECT group executes its logical B=1 calls sequentially. SCHEDULED
 * submits the whole group concurrently. The package's proved batch
 * contract decides whether that group is one physical B=N call or N B=1
 * calls; this harness never infers batching from concurrency.
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { decodeSlots } from '../ReceiptDigitSession.js';

export const DEFAULT_MAX_BATCH_DELAY_MS = 1;

function fail(message) {
  throw new Error(`[receipt_digit_reader/benchmark_runtime_modes] ${message}`);
}

function parseArguments(argv) {
  const options = {
    package: null,
    raws: [],
    backend: null,
    wasmUrl: null,
    api: null,
    mode: null,
    concurrency: 4,
    maxBatchDelayMs: DEFAULT_MAX_BATCH_DELAY_MS,
    repeat: 10,
    warmup: 2,
    out: null,
    includePrivateLogits: false,
    adapter: 'high-performance',
    requireAdapter: null,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const value = () => {
      const next = argv[++index];
      if (next === undefined) fail(`${flag} needs a value`);
      return next;
    };
    if (flag === '--package') options.package = value();
    else if (flag === '--raw') options.raws.push(value());
    else if (flag === '--backend') options.backend = value();
    else if (flag === '--wasm-url') options.wasmUrl = value();
    else if (flag === '--api') options.api = value();
    else if (flag === '--mode') options.mode = value();
    else if (flag === '--concurrency') options.concurrency = Number(value());
    else if (flag === '--max-batch-delay-ms') options.maxBatchDelayMs = Number(value());
    else if (flag === '--repeat') options.repeat = Number(value());
    else if (flag === '--warmup') options.warmup = Number(value());
    else if (flag === '--out') options.out = value();
    else if (flag === '--adapter') options.adapter = value();
    else if (flag === '--require-adapter') options.requireAdapter = value();
    else if (flag === '--include-private-logits') options.includePrivateLogits = true;
    else fail(`unknown option ${flag}`);
  }
  if (!options.package || options.raws.length === 0 || !options.backend) {
    fail('--package, --raw, and --backend are required');
  }
  if (!['cpu-js', 'wasm', 'webgpu'].includes(options.backend)) {
    fail("--backend must be 'cpu-js', 'wasm', or 'webgpu'");
  }
  for (const [name, minimum] of [['concurrency', 1], ['repeat', 1], ['warmup', 0]]) {
    if (!Number.isSafeInteger(options[name]) || options[name] < minimum) {
      fail(`--${name} must be an integer >= ${minimum}`);
    }
  }
  if (!Number.isFinite(options.maxBatchDelayMs) || options.maxBatchDelayMs < 0 ||
      options.maxBatchDelayMs > 60_000) {
    fail('--max-batch-delay-ms must be finite and between 0 and 60000');
  }
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

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.ceil(sorted.length * fraction) - 1];
}

function latencySummary(values) {
  return Object.freeze({
    median_ms: percentile(values, 0.5),
    p95_ms: percentile(values, 0.95),
    min_ms: Math.min(...values),
    max_ms: Math.max(...values),
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

function digestTypedArray(value) {
  return createHash('sha256')
    .update(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
    .digest('hex');
}

function digestFile(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

export function digestPackageArtifacts(packageDirectory) {
  return Object.freeze(Object.fromEntries(
    ['manifest.json', 'graph.json', 'model.safetensors'].map((filename) => [
      filename, digestFile(`${packageDirectory}/${filename}`),
    ]),
  ));
}

function compactBatchSemantics(compiled) {
  const evidence = compiled?.report?.batchSemantics;
  if (!evidence || typeof evidence !== 'object' ||
      typeof evidence.protocol !== 'string' ||
      typeof evidence.supported !== 'boolean' ||
      !Number.isSafeInteger(evidence.coveredNodes) ||
      typeof evidence.graphFingerprint !== 'string') {
    fail('compiled report does not expose typed batch-semantics evidence');
  }
  return Object.freeze({
    protocol: evidence.protocol,
    supported: evidence.supported,
    batchSymbol: evidence.batchSymbol ?? null,
    coveredNodes: evidence.coveredNodes,
    graphFingerprintSha256: createHash('sha256')
      .update(evidence.graphFingerprint, 'utf8').digest('hex'),
    reason: evidence.reason ?? null,
    failedNode: evidence.failedNode ?? null,
    failedTensor: evidence.failedTensor ?? null,
  });
}

function validatePhysicalEvidence(
  options, manifest, physical, scheduling, batchSemantics, scheduled,
) {
  const maximum = manifest.abi?.batch?.max ?? 1;
  if (!Number.isSafeInteger(maximum) || options.concurrency > maximum) {
    fail(`concurrency ${options.concurrency} exceeds the package batch maximum ${maximum}`);
  }
  if (scheduled && !batchSemantics.supported) {
    fail(`scheduled B>1 requires typed batch semantics (${batchSemantics.reason ?? 'unsupported'})`);
  }
  const expectedInvocations = options.repeat * (scheduled ? 1 : options.concurrency);
  if (physical.physicalInvocations !== expectedInvocations) {
    fail(`expected ${expectedInvocations} physical invocations, observed ${physical.physicalInvocations}`);
  }
  if (scheduled) {
    const batches = [...scheduling.batchSizes];
    if (batches.length !== 1 || batches[0] !== options.concurrency ||
        scheduling.dispatchIds.size !== options.repeat ||
        scheduling.trueBackendInvocations !== options.repeat) {
      fail('scheduled execution did not attest one B=N provider invocation per group');
    }
  }
  return Object.freeze({
    scheduled,
    expectedInvocations,
    expectedBatchSize: scheduled ? options.concurrency : 1,
    resultReadbackIncluded: options.backend === 'webgpu',
  });
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

function requirePhysicalWebGPU(compiled, requiredIdentity) {
  if (compiled?.backend !== 'webgpu') {
    fail(`required WebGPU compiled on '${compiled?.backend ?? 'unknown'}'`);
  }
  const info = compiled.report?.selectedDevice;
  if (!info || typeof info !== 'object' || Array.isArray(info)) {
    fail('WebGPU adapter identity is unavailable');
  }
  const description = Object.values(info)
    .filter((value) => typeof value === 'string' && value.trim())
    .join(' ');
  if (!description || /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|cpu)\b/i.test(description)) {
    fail(`software or unidentified WebGPU adapter rejected (${description || 'empty'})`);
  }
  if (!description.toLowerCase().includes(requiredIdentity.toLowerCase())) {
    fail(`WebGPU adapter '${description}' does not match '${requiredIdentity}'`);
  }
  return Object.freeze({ ...info });
}

function createInvocationCounter() {
  let physicalInvocations = 0;
  const shapeSignatures = new Set();
  return Object.freeze({
    onDiagnostic(event) {
      if (event.kind !== 'execution') return;
      physicalInvocations++;
      if (typeof event.report?.shapeSignature === 'string') {
        shapeSignatures.add(event.report.shapeSignature);
      }
    },
    reset() {
      physicalInvocations = 0;
      shapeSignatures.clear();
    },
    snapshot(logicalRequests) {
      return Object.freeze({
        source: 'Runtime onDiagnostic execution reports',
        logicalRequests,
        physicalInvocations,
        logicalRequestsPerPhysicalInvocation: logicalRequests / physicalInvocations,
        shapeSignatures: Object.freeze([...shapeSignatures].sort()),
      });
    },
  });
}

async function executeGroup(compiled, inputs, mode, directMode) {
  const started = performance.now();
  if (mode === directMode) {
    const results = [];
    for (const laneInputs of inputs) {
      results.push(await compiled.run(laneInputs, { mode: directMode }));
    }
    return Object.freeze({ results: Object.freeze(results), elapsedMs: performance.now() - started });
  }
  const handles = Array.from(
    { length: inputs.length },
    (_, lane) => compiled.submit(inputs[lane]),
  );
  const results = await Promise.all(handles.map((handle) => handle.result));
  return Object.freeze({ results: Object.freeze(results), elapsedMs: performance.now() - started });
}

async function inspectAndClose(
  results, manifest, outputDigests, records, scheduling, representativeLogits = null,
) {
  try {
    for (let lane = 0; lane < results.length; lane++) {
      const result = results[lane];
      const logits = await result.output(manifest.abi.output).read();
      if (representativeLogits) representativeLogits[lane] = Float32Array.from(logits);
      outputDigests[lane].add(digestTypedArray(logits));
      const decoded = decodeSlots(logits, manifest.decode);
      records[lane].add(`${decoded.phone}/${decoded.street}`);
      const report = result.report.backendReport?.scheduling;
      if (report && typeof report === 'object') {
        if (Number.isSafeInteger(report.batchSize)) scheduling.batchSizes.add(report.batchSize);
        if (typeof report.dispatchId === 'string') scheduling.dispatchIds.add(report.dispatchId);
        if (Number.isSafeInteger(report.trueBackendInvocations)) {
          scheduling.trueBackendInvocations += report.trueBackendInvocations;
        }
      }
    }
  } finally {
    await Promise.all(results.map((result) => result.close()));
  }
}

function compareLaneOutputs(reference, actual) {
  if (!(reference instanceof Float32Array) || !(actual instanceof Float32Array) ||
      reference.length !== actual.length) {
    fail('independent and scheduled lane outputs have incompatible storage');
  }
  let maximumAbsoluteDifference = 0;
  let maximumRelativeDifference = 0;
  let close = true;
  for (let index = 0; index < reference.length; index++) {
    const absolute = Math.abs(actual[index] - reference[index]);
    const relative = absolute / Math.max(Math.abs(reference[index]), 1.0e-12);
    maximumAbsoluteDifference = Math.max(maximumAbsoluteDifference, absolute);
    maximumRelativeDifference = Math.max(maximumRelativeDifference, relative);
    if (absolute > 1.0e-5 + 1.0e-5 * Math.abs(reference[index])) close = false;
  }
  return Object.freeze({
    exact: digestTypedArray(reference) === digestTypedArray(actual),
    close,
    maximumAbsoluteDifference,
    maximumRelativeDifference,
  });
}

export function summarizePublicLanes(
  records, outputDigests, laneParity = null, referenceRecordMatches = null,
) {
  return Object.freeze(records.map((recordValues, lane) => Object.freeze({
    laneId: `lane-${String(lane + 1).padStart(2, '0')}`,
    decodedRecordStableAcrossMeasuredGroups: recordValues.size === 1,
    outputStableAcrossMeasuredGroups: outputDigests[lane]?.size === 1,
    ...(laneParity === null ? {} : {
      outputMatchesIndependentReference: laneParity[lane]?.close === true,
      outputExactlyMatchesIndependentReference: laneParity[lane]?.exact === true,
      decodedRecordMatchesIndependentReference: referenceRecordMatches?.[lane] === true,
    }),
  })));
}

export async function benchmarkRuntimeMode(options) {
  const runtimeInfo = runtimeIdentity(options.backend);
  const manifest = JSON.parse(readFileSync(`${options.package}/manifest.json`, 'utf8'));
  const expectedElements = manifest.abi.input.shape.reduce((total, extent) => total * extent, 1);
  if (options.raws.length !== 1 && options.raws.length < options.concurrency) {
    fail('--raw must be provided once for a shared input or at least once per concurrent lane');
  }
  const lanePaths = Array.from(
    { length: options.concurrency },
    (_, lane) => options.raws.length === 1 ? options.raws[0] : options.raws[lane],
  );
  const pixels = lanePaths.map((path) => {
    const raw = readFileSync(path);
    const view = new Float32Array(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
    );
    if (view.length !== expectedElements) {
      fail(`raw input ${path} has ${view.length} F32 values, expected ${expectedElements}`);
    }
    return view;
  });

  const api = options.api
    ? await import(pathToFileURL(options.api).href)
    : await import('../../../ts/index.js');
  const directMode = api.executionModes?.[api.ExecutionMode?.Direct];
  const scheduledMode = api.executionModes?.[api.ExecutionMode?.Scheduled];
  const mode = options.mode ?? scheduledMode;
  if (!Array.isArray(api.executionModes) || !api.executionModes.includes(mode) ||
      directMode === undefined || scheduledMode === undefined) {
    fail(`--mode must be one of the generated execution modes: ${
      api.executionModes?.join(', ') ?? 'unavailable API contract'}`);
  }
  const scheduled = mode === scheduledMode && options.concurrency > 1;
  const invocations = createInvocationCounter();
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
    maxBatchSize: options.concurrency,
    maxBatchDelayMs: options.maxBatchDelayMs,
  });
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [options.backend],
      ...(options.backend === 'wasm' && options.wasmUrl
        ? { wasmUrl: pathToFileURL(options.wasmUrl).href }
        : {}),
      execution: {
        mode,
        scheduler,
      },
      onDiagnostic: invocations.onDiagnostic,
    });
    const snapshot = await api.Model.load(
      pathToFileURL(`${options.package}/model.safetensors`).href,
      {
        graphUrl: pathToFileURL(`${options.package}/graph.json`).href,
        fetch: fileFetch,
      },
    );
    compiled = await runtime.compile(snapshot, {
      backend: {
        mode: 'require', backend: options.backend, operatorFallback: 'forbid',
      },
    });
    if (options.backend === 'webgpu') {
      selectedDevice = requirePhysicalWebGPU(compiled, options.requireAdapter);
    }
    const batchSemantics = compactBatchSemantics(compiled);
    const inputs = Object.freeze(pixels.map((data) => Object.freeze({
      [manifest.abi.input.name]: Object.freeze({
        data,
        shape: Object.freeze([...manifest.abi.input.shape]),
      }),
    })));
    let independentReference = null;
    if (scheduled) {
      const referenceDigests = Array.from(
        { length: options.concurrency }, () => new Set(),
      );
      const referenceRecords = Array.from(
        { length: options.concurrency }, () => new Set(),
      );
      const referenceLogits = Array(options.concurrency).fill(null);
      invocations.reset();
      const referenceGroup = await executeGroup(compiled, inputs, directMode, directMode);
      await inspectAndClose(
        referenceGroup.results,
        manifest,
        referenceDigests,
        referenceRecords,
        { batchSizes: new Set(), dispatchIds: new Set(), trueBackendInvocations: 0 },
        referenceLogits,
      );
      const referencePhysical = invocations.snapshot(options.concurrency);
      if (referencePhysical.physicalInvocations !== options.concurrency) {
        fail('independent B=1 reference did not execute once per logical lane');
      }
      independentReference = Object.freeze({
        physicalEvidence: referencePhysical,
        decodedRecords: Object.freeze(referenceRecords.map((values) => [...values][0])),
        logits: Object.freeze(referenceLogits),
      });
    }
    const newLaneSets = () => Array.from(
      { length: options.concurrency }, () => new Set(),
    );
    const warmupDigests = newLaneSets();
    const warmupRecords = newLaneSets();
    for (let index = 0; index < options.warmup; index++) {
      const group = await executeGroup(compiled, inputs, mode, directMode);
      await inspectAndClose(group.results, manifest, warmupDigests, warmupRecords, {
        batchSizes: new Set(), dispatchIds: new Set(), trueBackendInvocations: 0,
      });
    }

    invocations.reset();
    const groupLatencyMs = [];
    const outputDigests = newLaneSets();
    const records = newLaneSets();
    const scheduling = {
      batchSizes: new Set(), dispatchIds: new Set(), trueBackendInvocations: 0,
    };
    const representativeLogits = Array(options.concurrency).fill(null);
    for (let index = 0; index < options.repeat; index++) {
      const synchronizedStarted = performance.now();
      const group = await executeGroup(compiled, inputs, mode, directMode);
      await inspectAndClose(
        group.results, manifest, outputDigests, records, scheduling, representativeLogits,
      );
      groupLatencyMs.push(options.backend === 'webgpu'
        ? performance.now() - synchronizedStarted
        : group.elapsedMs);
    }
    const logicalRequests = options.repeat * options.concurrency;
    const physical = invocations.snapshot(logicalRequests);
    if (records.some((values) => values.size !== 1)
        || outputDigests.some((values) => values.size !== 1)) {
      fail('decoded record or numerical output changed between requests');
    }
    const laneParity = independentReference === null
      ? null
      : Object.freeze(representativeLogits.map((values, lane) =>
        compareLaneOutputs(independentReference.logits[lane], values)));
    const referenceRecordMatches = independentReference === null
      ? null
      : Object.freeze(records.map((values, lane) =>
        independentReference.decodedRecords[lane] === [...values][0]));
    if (laneParity?.some((comparison, lane) =>
      !comparison.close || referenceRecordMatches[lane] !== true)) {
      fail('scheduled output does not match the same-backend independent B=1 reference');
    }
    const physicalContract = validatePhysicalEvidence(
      options, manifest, physical, scheduling, batchSemantics, scheduled,
    );
    return Object.freeze({
      schema: 'volvoxai.receipt-digit-runtime-modes/v2',
      environment: Object.freeze({
        runtime: runtimeInfo,
        platform: process.platform,
        architecture: process.arch,
        backend: options.backend,
        apiSha256: options.api ? digestFile(options.api) : null,
        selectedDevice,
        ...(options.backend === 'wasm' && options.wasmUrl
          ? { wasmSha256: digestFile(options.wasmUrl) }
          : {}),
      }),
      packageManifestSha256: digestFile(`${options.package}/manifest.json`),
      packageArtifacts: digestPackageArtifacts(options.package),
      variant: manifest.variant,
      batchSemantics,
      inputShape: manifest.abi.input.shape,
      lanes: summarizePublicLanes(
        records, outputDigests, laneParity, referenceRecordMatches,
      ),
      distinctLaneInputs:
        new Set(pixels.map((values) => digestTypedArray(values))).size === pixels.length,
      mode,
      scheduler,
      concurrency: options.concurrency,
      warmupGroups: options.warmup,
      measuredGroups: options.repeat,
      independentReference: independentReference === null ? null : Object.freeze({
        physicalEvidence: independentReference.physicalEvidence,
      }),
      laneParity,
      privacy: Object.freeze({
        inputPathsIncluded: false,
        decodedRecordsIncluded: false,
        outputDigestsIncluded: false,
        privateLogitsIncluded: options.includePrivateLogits,
      }),
      ...(options.includePrivateLogits ? {
        laneLogits: Object.freeze(representativeLogits.map((values) =>
          Object.freeze(Array.from(values ?? [])))),
      } : {}),
      groupLatency: latencySummary(groupLatencyMs),
      timingBoundary: options.backend === 'webgpu'
        ? 'all required output readbacks and result close'
        : 'provider execution result publication',
      logicalRequestsPerSecond:
        logicalRequests * 1000 / groupLatencyMs.reduce((sum, value) => sum + value, 0),
      physicalEvidence: physical,
      physicalContract,
      schedulingEvidence: Object.freeze({
        observedBatchSizes: Object.freeze([...scheduling.batchSizes].sort((a, b) => a - b)),
        dispatches: scheduling.dispatchIds.size,
        trueBackendInvocations: scheduling.trueBackendInvocations,
      }),
      executionInspection: runtime.inspectExecution(),
    });
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
  const cliOptions = parseArguments(runtimeArguments);
  const originalLog = console.log;
  // Provider startup diagnostics are useful provenance, but stdout is the
  // machine-readable report consumed by the Python benchmark wrapper.
  console.log = (...values) => process.stderr.write(`${values.map(String).join(' ')}\n`);
  benchmarkRuntimeMode(cliOptions)
    .then((report) => {
      console.log = originalLog;
      const payload = `${JSON.stringify(report, null, 2)}\n`;
      if (cliOptions.out) writeFileSync(cliOptions.out, payload);
      process.stdout.write(payload);
    })
    .catch((error) => {
      console.log = originalLog;
      process.stderr.write(`${String(error?.stack || error?.message || error)}\n`);
      process.exitCode = 1;
    });
}
