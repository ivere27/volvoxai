import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DYNAMIC_QUALIFICATION_PREFIX,
  executionPhaseBreakdown,
  finiteMilliseconds,
  parseArguments,
  summarizeDecoderSteps,
  validateDynamicExecutionGroups,
  validateExplicitKVAnswer,
} from '../tools/benchmark_explicit_kv_runtime.mjs';
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
      '--package', 'p', '--image', 'i', '--prompt', 'q', '--backend', 'cpu',
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
