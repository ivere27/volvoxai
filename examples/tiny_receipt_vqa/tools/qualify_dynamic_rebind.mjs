// Browser- and Node-neutral TinyReceipt qualification orchestration. This is
// intentionally outside VolvoxAI: the sequence, shape limits, and parity
// decisions are properties of the TinyReceipt package contract.

import { validateDynamicRebindAnswer } from './benchmark_explicit_kv_contract.mjs';

export const DYNAMIC_REBIND_QUALIFICATION_SCHEMA =
  'volvoxai.tiny-receipt-dynamic-shape-qualification/v1';

const MAX_Q = 192;
const IMAGE_TOKENS = 210;

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function summarizeAnswer(label, answer, maximumNewTokens) {
  const tokenIds = validateDynamicRebindAnswer(
    answer,
    maximumNewTokens,
    answer.shapeMode,
  );
  const gpuResidentKv = answer?.gpuResidentKv;
  return Object.freeze({
    label,
    shapeMode: answer.shapeMode,
    family: answer.family,
    familyId: answer.familyId,
    requestedFamily: answer.requestedFamily,
    questionTokenIds: Object.freeze([...answer.questionTokenIds]),
    tokenIds,
    stoppedAtEos: answer.stoppedAtEos === true,
    activeShape: Object.freeze({ ...answer.activeShape }),
    logicalShape: Object.freeze({ ...answer.logicalShape }),
    cache: Object.freeze({
      initialPastLength: answer.cacheShape.initialPastLength,
      finalPastLength: answer.cacheShape.finalPastLength,
      sentinelMaskValue: answer.decodeReports[0].sentinelMaskValue,
      transitions: Object.freeze(answer.decodeReports.map(
        ({ position, pastLength, presentLength }) => Object.freeze({
          position, pastLength, presentLength,
        }),
      )),
    }),
    ...(gpuResidentKv && typeof gpuResidentKv === 'object'
      ? { gpuResidentKv: Object.freeze({ ...gpuResidentKv }) }
      : {}),
  });
}

function sameDecision(left, right) {
  return left.family === right.family && left.familyId === right.familyId &&
    sameArray(left.tokenIds, right.tokenIds);
}

/**
 * Exercise one loaded session through bounded grow/shrink rebinds. Hooks let a
 * backend harness attach execution reports without changing this scenario.
 */
export async function qualifyDynamicRebind({
  session,
  image,
  prompt,
  family = 'phone',
  maximumNewTokens = 4,
  shortPrompt = 'x',
  beforeRun = null,
  afterRun = null,
  generationOptions = null,
}) {
  if (!session || typeof session.generate !== 'function') {
    throw new Error('dynamic rebind qualification requires one loaded session');
  }
  if (!(image instanceof Float32Array) || image.length !== 320 * 672) {
    throw new Error('dynamic rebind qualification requires preprocessed F32 [1,1,320,672]');
  }
  if (typeof prompt !== 'string' || prompt.length === 0 ||
      typeof shortPrompt !== 'string' || shortPrompt.length === 0) {
    throw new Error('dynamic rebind qualification prompts must be non-empty strings');
  }
  if (!Number.isSafeInteger(maximumNewTokens) ||
      maximumNewTokens < 2 || maximumNewTokens > 191) {
    throw new Error('dynamic rebind qualification token bound must be in [2,191]');
  }
  const requests = [
    ['short-before', shortPrompt, 'active'],
    ['representative-active', prompt, 'active'],
    ['representative-maximum-padded', prompt, 'maximum-padded'],
    ['short-after', shortPrompt, 'active'],
  ];
  const runs = [];
  for (const [index, [label, requestPrompt, shapeMode]] of requests.entries()) {
    await beforeRun?.({ index, label, shapeMode });
    let answer;
    try {
      answer = await session.generate({
        ...(generationOptions || {}),
        image,
        prompt: requestPrompt,
        family,
        maxNewTokens: maximumNewTokens,
        preprocessed: true,
        shapeMode,
      });
    } finally {
      await afterRun?.({ index, label, shapeMode });
    }
    runs.push(summarizeAnswer(label, answer, maximumNewTokens));
  }

  const [shortBefore, representative, maximumPadded, shortAfter] = runs;
  if (shortBefore.logicalShape.Q >= representative.logicalShape.Q) {
    throw new Error('qualification did not vary the active question/memory extent');
  }
  if (!sameArray(representative.questionTokenIds, maximumPadded.questionTokenIds) ||
      representative.logicalShape.Q !== maximumPadded.logicalShape.Q ||
      representative.logicalShape.M !== maximumPadded.logicalShape.M ||
      maximumPadded.activeShape.Q !== MAX_Q ||
      maximumPadded.activeShape.M !== MAX_Q + IMAGE_TOKENS) {
    throw new Error('maximum-padded run is not the representative logical request');
  }
  if (!sameDecision(shortBefore, shortAfter)) {
    throw new Error('short request changed after grow/shrink context reuse');
  }
  if (!sameDecision(representative, maximumPadded)) {
    throw new Error('representative active and maximum-padded decisions differ');
  }

  return Object.freeze({
    schema: DYNAMIC_REBIND_QUALIFICATION_SCHEMA,
    timed: false,
    boundedDecoderMaximumNewTokens: maximumNewTokens,
    maximumLegalEncoderBinding: Object.freeze({ B: 1, Q: MAX_Q, M: MAX_Q + IMAGE_TOKENS }),
    sameSession: true,
    checks: Object.freeze({
      exactOutputShapes: true,
      finiteOutputs: true,
      cachePrefixPreserved: true,
      appendedCacheRowFinite: true,
      selectedFamilyParity: true,
      tokenParity: true,
      encoderGrowShrink: true,
      decoderGrowShrink: true,
    }),
    runs: Object.freeze(runs),
  });
}
