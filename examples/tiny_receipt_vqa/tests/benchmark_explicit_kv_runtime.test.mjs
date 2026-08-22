import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import {
  CANONICAL_INPUT_F32_SHA256,
  CANONICAL_PIXEL_SHA256,
  DYNAMIC_QUALIFICATION_PREFIX,
  closeRuntimeSampleResources,
  createCanonicalTinyReceiptImage,
  executionPhaseBreakdown,
  finiteMilliseconds,
  overrideWebGPUAdapterPreference,
  parseArguments,
  selectComponentTiming,
  summarizeDecoderSteps,
  validateCompilation,
  validateDynamicExecutionGroups,
  validateExplicitKVAnswer,
  validateSynchronizedTiming,
  validateWebGPUCacheEvidence,
} from '../tools/benchmark_explicit_kv_runtime.mjs';
import { preprocessTinyReceiptImage } from '../TinyReceiptInput.js';
import {
  DYNAMIC_REBIND_QUALIFICATION_SCHEMA,
  qualifyDynamicRebind,
} from '../tools/qualify_dynamic_rebind.mjs';

function qualificationAnswer({ prompt, shapeMode, tokenIds = [4, 5] }) {
  const logicalQ = prompt === 'x' ? 2 : 4;
  const logicalShape = { B: 1, Q: logicalQ, M: logicalQ + 210, T: 5 };
  return {
    execution: 'explicit-kv-cache',
    decodeMode: 'explicit-kv-cache',
    tokenIds,
    family: 'phone',
    familyId: 0,
    requestedFamily: 'phone',
    stoppedAtEos: false,
    decoderSeedExecutions: 1,
    decoderOrdinaryExecutions: tokenIds.length,
    decoderCacheStepExecutions: tokenIds.length - 1,
    shapeMode,
    questionTokenIds: Array.from({ length: logicalQ }, (_value, index) => index + 1),
    activeShape: shapeMode === 'active'
      ? logicalShape : { B: 1, Q: 192, M: 402, T: 5 },
    logicalShape,
    cacheShape: {
      initialPastLength: 1,
      finalPastLength: tokenIds.length + 1,
      sentinelSlots: 1,
    },
    decodeReports: tokenIds.map((_token, index) => ({
      operation: index === 0 ? 'explicit-kv-seed' : 'explicit-kv-step',
      position: index,
      pastLength: index + 1,
      presentLength: index + 2,
      sentinelMaskValue: 1,
    })),
  };
}

function strictExecution(contextId) {
  return {
    backend: 'wasm',
    contextId,
    outcome: 'success',
    operatorFallback: 'none',
    routeEvidence: {
      tierFallback: false,
      operator: {
        attestation: 'none',
        used: false,
        offendingNode: null,
      },
    },
  };
}

test('runtime runner accepts only a bounded explicit-cache command', () => {
  const parsed = parseArguments([
    '--package', 'package', '--image=image.png', '--prompt', 'phone number?',
    '--backend', 'wasm', '--family', 'phone', '--max-new', '4',
  ]);
  assert.equal(parsed.backend, 'wasm');
  assert.equal(parsed.maxNewTokens, 4);
  assert.equal(parsed.family, 'phone');
  assert.equal(parsed.qualification, null);
  assert.equal(parsed.executionWarmup, 0);
  assert.equal(parsed.canonicalImage, false);
  const denoWebGPU = parseArguments([
    '--package', 'package', '--canonical-image', '--prompt', 'phone number last one',
    '--backend', 'webgpu', '--family', 'phone', '--max-new', '4',
    '--execution-warmup', '1', '--adapter', 'high-performance',
    '--require-adapter', 'Xclipse 540',
  ]);
  assert.equal(denoWebGPU.backend, 'webgpu');
  assert.equal(denoWebGPU.canonicalImage, true);
  assert.equal(denoWebGPU.imagePath, null);
  assert.equal(denoWebGPU.executionWarmup, 1);
  assert.equal(denoWebGPU.requiredAdapter, 'Xclipse 540');
  const qualification = parseArguments([
    '--package', 'package', '--image=image.png', '--qualification', 'dynamic-rebind',
  ]);
  assert.equal(qualification.prompt, 'phone number last one');
  assert.equal(qualification.qualification, 'dynamic-rebind');
  assert.equal(DYNAMIC_QUALIFICATION_PREFIX,
    'TINYRECEIPT_DYNAMIC_REBIND_QUALIFICATION ');
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--image', 'i', '--qualification', 'dynamic-rebind',
      '--prompt', 'different prompt',
    ]),
    /requires the canonical prompt/,
  );
  assert.throws(
    () => parseArguments(['--package', 'p', '--image', 'i']),
    /--prompt is required/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--image', 'i', '--qualification', 'dynamic-rebind',
      '--max-new', '5',
    ]),
    /requires --max-new=4/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--image', 'i', '--qualification', 'something-else',
    ]),
    /must be 'dynamic-rebind'/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--image', 'i', '--prompt', 'q', '--backend', 'cpu-js',
    ]),
    /JS CPU is outside/,
  );
  assert.throws(
    () => parseArguments(['--package', 'p', '--image', 'i', '--prompt', 'q', '--max-new', '1']),
    /\[2, 191\]/,
  );
  assert.throws(
    () => parseArguments(['--package', 'p', '--image', 'i', '--prompt', 'q', '--unsupported-mode']),
    /unknown option/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--image', 'i', '--canonical-image', '--prompt', 'q',
    ]),
    /exactly one/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--canonical-image', '--prompt', 'q', '--backend', 'webgpu',
      '--execution-warmup', '1',
    ]),
    /--require-adapter is required/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--canonical-image', '--prompt', 'q', '--backend', 'webgpu',
      '--require-adapter', 'GPU',
    ]),
    /execution-warmup >= 1/,
  );
  assert.throws(
    () => parseArguments([
      '--package', 'p', '--canonical-image', '--prompt', 'different',
    ]),
    /--canonical-image requires/,
  );
});

test('runtime strict compilation binds the selected candidate to one device', () => {
  const routeEvidence = {
    tierFallback: false,
    operator: { attestation: 'none', used: false, offendingNode: null },
  };
  const device = { backend: 'webgpu', description: 'physical' };
  const report = {
    requestedPolicy: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
    selectedBackend: 'webgpu',
    selectedDevice: device,
    routeEvidence,
    candidates: [{
      backend: 'webgpu', outcome: 'selected', device: { ...device }, routeEvidence,
    }],
  };
  assert.doesNotThrow(() => validateCompilation(report, 'webgpu', 'encoder'));
  report.candidates[0].device = { backend: 'webgpu', description: 'other' };
  assert.throws(
    () => validateCompilation(report, 'webgpu', 'encoder'),
    /did not strictly select webgpu/,
  );
});

test('runtime adapter override restores inherited requestAdapter exactly', async () => {
  const calls = [];
  const prototype = {
    requestAdapter(options) {
      calls.push(options);
      return Promise.resolve('adapter');
    },
  };
  const gpu = Object.create(prototype);
  assert.equal(Object.hasOwn(gpu, 'requestAdapter'), false);
  const restore = overrideWebGPUAdapterPreference(gpu, 'high-performance');
  assert.equal(Object.hasOwn(gpu, 'requestAdapter'), true);
  assert.equal(await gpu.requestAdapter({ forceFallbackAdapter: false }), 'adapter');
  assert.deepEqual(calls, [{ forceFallbackAdapter: false, powerPreference: 'high-performance' }]);
  restore();
  assert.equal(Object.hasOwn(gpu, 'requestAdapter'), false);
  assert.equal(gpu.requestAdapter, prototype.requestAdapter);
});

test('runtime component boundary preserves WASM diagnostics and synchronizes WebGPU', () => {
  const synchronized = {
    encoderMs: 12,
    decoder: summarizeDecoderSteps([4, 3, 2, 1]),
  };
  const encoderPhases = { executionMs: 10 };
  const decoderPhases = [4, 2, 2, 1].map((executionMs) => ({ executionMs }));
  const wasm = selectComponentTiming('wasm', synchronized, encoderPhases, decoderPhases);
  assert.equal(wasm.boundary, 'runtime-execution-diagnostic');
  assert.equal(wasm.encoderMs, 10);
  assert.equal(wasm.decoder.totalMs, 9);
  const webgpu = selectComponentTiming('webgpu', synchronized, encoderPhases, decoderPhases);
  assert.equal(webgpu.boundary, 'required-small-output-readback');
  assert.equal(webgpu.encoderMs, 12);
  assert.equal(webgpu.decoder.totalMs, 10);
});

test('runtime WebGPU evidence requires synchronized output and readback-free measured KV', () => {
  const reports = {
    encoder: { executionTimeMs: 8 },
    decoder: [3, 2, 2, 1].map((executionTimeMs) => ({ executionTimeMs })),
  };
  const answer = {
    synchronizedTiming: {
      completion: 'required-small-output-readback',
      applicationValidationIncluded: false,
      cacheQualificationReadbackIncluded: false,
      encoderExecutionMs: 9,
      decoderStepMs: [4, 3, 2, 1],
    },
    gpuResidentKv: {
      enabled: true,
      mode: 'device-resident',
      crossCacheOutputs: 8,
      presentCacheOutputsPerStep: 8,
      encoderCrossCacheHandoffs: 32,
      decoderCacheHandoffs: 24,
      runtimeValidatedDeviceInputs: true,
      encoderResultRetainedThroughDecode: true,
      decoderResultRetainedUntilSuccessorExecution: true,
      encoderMemoryReadback: false,
      encoderMemoryReadbackValidated: false,
      encoderCrossCacheReadbackValidated: false,
      cachePrefixReadbackValidated: false,
      appendedCacheReadbackValidated: false,
      cacheReadbackFree: true,
    },
  };
  const timing = validateSynchronizedTiming(answer, reports, 'webgpu', 'measured');
  assert.equal(timing.encoderMs, 9);
  assert.equal(timing.decoder.totalMs, 10);
  assert.doesNotThrow(() => validateWebGPUCacheEvidence(answer, 4, false, 'measured'));
  answer.gpuResidentKv.cacheReadbackFree = false;
  assert.throws(
    () => validateWebGPUCacheEvidence(answer, 4, false, 'measured'),
    /performed a KV cache readback/,
  );
  answer.synchronizedTiming.applicationValidationIncluded = true;
  assert.throws(
    () => validateSynchronizedTiming(answer, reports, 'webgpu', 'measured'),
    /includes application or cache qualification readback/,
  );
});

test('runtime cleanup attempts every owner and preserves operation failure', async () => {
  const operationError = new Error('operation');
  const sessionError = new Error('session close');
  let runtimeClosed = 0;
  let adapterRestored = 0;
  await assert.rejects(
    closeRuntimeSampleResources({
      session: { close: async () => { throw sessionError; } },
      runtime: { close: async () => { runtimeClosed++; } },
      restoreAdapter: () => { adapterRestored++; },
      operationError,
    }),
    (error) => error instanceof AggregateError &&
      error.errors.length === 2 && error.errors[0] === operationError &&
      error.errors[1] === sessionError,
  );
  assert.equal(runtimeClosed, 1);
  assert.equal(adapterRestored, 1);
});

test('runtime canonical generator preserves the benchmark pixel and F32 identities', async () => {
  const image = createCanonicalTinyReceiptImage();
  assert.deepEqual(
    { width: image.width, height: image.height, channels: image.channels },
    { width: 672, height: 320, channels: 1 },
  );
  assert.equal(createHash('sha256').update(image.data).digest('hex'), CANONICAL_PIXEL_SHA256);
  const input = await preprocessTinyReceiptImage(image, { width: 672, height: 320 });
  const bytes = new Uint8Array(input.length * 4);
  const view = new DataView(bytes.buffer);
  input.forEach((value, index) => view.setFloat32(index * 4, value, true));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), CANONICAL_INPUT_F32_SHA256);
});

test('runtime qualification proves strict same-context executions for every run', () => {
  const labels = [
    'short-before',
    'representative-active',
    'representative-maximum-padded',
    'short-after',
  ];
  const runs = labels.map((label) => ({ label, tokenIds: [4, 5] }));
  const groups = labels.map((label) => ({
    label,
    reports: [
      strictExecution('encoder-context'),
      strictExecution('decoder-context'),
      strictExecution('decoder-context'),
    ],
  }));
  const evidence = validateDynamicExecutionGroups(groups, runs, 'wasm');
  assert.equal(evidence.encoderContextId, 'encoder-context');
  assert.equal(evidence.decoderContextId, 'decoder-context');
  assert.equal(evidence.attestations.length, 4);
  assert.ok(evidence.attestations.every(({ encoder, decoder }) =>
    encoder.contextId === 'encoder-context' && decoder.length === 2 &&
    decoder.every(({ contextId }) => contextId === 'decoder-context')));

  const changed = structuredClone(groups);
  changed[3].reports[0].contextId = 'replacement-encoder';
  assert.throws(
    () => validateDynamicExecutionGroups(changed, runs, 'wasm'),
    /did not reuse the same contexts/,
  );
  const fallback = structuredClone(groups);
  fallback[1].reports[2].operatorFallback = 'cpu';
  assert.throws(
    () => validateDynamicExecutionGroups(fallback, runs, 'wasm'),
    /did not execute successfully on strict wasm/,
  );
});

test('runtime timing separates the P=1 seed from steady cache steps', () => {
  assert.equal(finiteMilliseconds(0, 'encoder execution'), 0);
  assert.equal(finiteMilliseconds(1.25, 'encoder execution'), 1.25);
  assert.throws(() => finiteMilliseconds(Number.NaN, 'encoder execution'), /finite non-negative/);
  assert.throws(() => finiteMilliseconds(-1, 'encoder execution'), /finite non-negative/);
  assert.deepEqual(summarizeDecoderSteps([4, 2, 3]), {
    seedMs: 4,
    steadySteps: 2,
    steadyTotalMs: 5,
    steadyMeanMs: 2.5,
    steadyTokensPerSecond: 400,
    totalMs: 9,
    steps: [4, 2, 3],
  });
  assert.throws(() => summarizeDecoderSteps([4]), /at least two/);
});

test('runtime timing preserves shape, provider, and framework phase evidence', () => {
  assert.deepEqual(executionPhaseBreakdown({
    executionTimeMs: 12,
    shapeBindTimeMs: 3,
    providerTimeMs: 8,
    backendReport: { specializationCacheHit: true },
  }, 'decoder'), {
    executionMs: 12,
    shapeBindMs: 3,
    providerMs: 8,
    frameworkMs: 1,
    specializationCacheHit: true,
  });
  assert.throws(
    () => executionPhaseBreakdown({
      executionTimeMs: 2,
      shapeBindTimeMs: 1.5,
      providerTimeMs: 1,
    }, 'decoder'),
    /phase timings exceed/,
  );
});

test('runtime answer evidence requires exact P to P+1 transitions', () => {
  const answer = {
    execution: 'explicit-kv-cache',
    decodeMode: 'explicit-kv-cache',
    tokenIds: [4, 5, 6],
    decoderSeedExecutions: 1,
    decoderRowExecutions: 0,
    decoderDependencyExecutions: 0,
    decoderOrdinaryExecutions: 3,
    decoderCacheStepExecutions: 2,
    shapeMode: 'active',
    questionTokenIds: [11, 12, 2],
    activeShape: { B: 1, Q: 3, M: 213, T: 5 },
    logicalShape: { B: 1, Q: 3, M: 213, T: 5 },
    cacheShape: { initialPastLength: 1, finalPastLength: 4, sentinelSlots: 1 },
    decodeReports: [
      { operation: 'explicit-kv-seed', position: 0, pastLength: 1, presentLength: 2,
        sentinelMaskValue: 1 },
      { operation: 'explicit-kv-step', position: 1, pastLength: 2, presentLength: 3,
        sentinelMaskValue: 1 },
      { operation: 'explicit-kv-step', position: 2, pastLength: 3, presentLength: 4,
        sentinelMaskValue: 1 },
    ],
  };
  assert.deepEqual(validateExplicitKVAnswer(answer, 4), [4, 5, 6]);
  answer.decodeReports[1].presentLength = 4;
  assert.throws(() => validateExplicitKVAnswer(answer, 4), /cache transition 1/);
  answer.decodeReports[1].presentLength = 3;
  answer.activeShape.Q = 4;
  assert.throws(() => validateExplicitKVAnswer(answer, 4), /exact active/);
});

test('untimed qualification reuses one session across grow, maximum, and shrink', async () => {
  const calls = [];
  const hooks = [];
  const session = {
    async generate(options) {
      calls.push(options);
      return qualificationAnswer(options);
    },
  };
  const result = await qualifyDynamicRebind({
    session,
    image: new Float32Array(320 * 672),
    prompt: 'phone number last one',
    family: 'phone',
    maximumNewTokens: 4,
    generationOptions: { kvTransferMode: 'device-qualified' },
    beforeRun: ({ label }) => hooks.push(`before:${label}`),
    afterRun: ({ label }) => hooks.push(`after:${label}`),
  });
  assert.equal(
    DYNAMIC_REBIND_QUALIFICATION_SCHEMA,
    'volvoxai.tiny-receipt-dynamic-shape-qualification/v1',
  );
  assert.equal(result.schema, DYNAMIC_REBIND_QUALIFICATION_SCHEMA);
  assert.equal(result.timed, false);
  assert.equal(result.sameSession, true);
  assert.deepEqual(calls.map(({ prompt, shapeMode }) => ({ prompt, shapeMode })), [
    { prompt: 'x', shapeMode: 'active' },
    { prompt: 'phone number last one', shapeMode: 'active' },
    { prompt: 'phone number last one', shapeMode: 'maximum-padded' },
    { prompt: 'x', shapeMode: 'active' },
  ]);
  assert.ok(calls.every(({ kvTransferMode }) => kvTransferMode === 'device-qualified'));
  assert.deepEqual(result.runs.map(({ activeShape }) => [activeShape.Q, activeShape.M]), [
    [2, 212], [4, 214], [192, 402], [2, 212],
  ]);
  assert.equal(result.checks.decoderGrowShrink, true);
  assert.equal(hooks.length, 8);

  let invocation = 0;
  await assert.rejects(qualifyDynamicRebind({
    session: {
      async generate(options) {
        invocation++;
        return qualificationAnswer({
          ...options,
          tokenIds: invocation === 4 ? [4, 6] : [4, 5],
        });
      },
    },
    image: new Float32Array(320 * 672),
    prompt: 'phone number last one',
    maximumNewTokens: 4,
  }), /short request changed/);
});
