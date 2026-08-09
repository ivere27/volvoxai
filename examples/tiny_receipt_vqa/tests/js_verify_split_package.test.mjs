import assert from 'node:assert/strict';
import test from 'node:test';

import {
  runActivePaddedOracle,
  verifyActivePaddedDecisionOracle,
} from '../tools/verify_split_package.mjs';

const LOGICAL_SHAPE = Object.freeze({ B: 1, Q: 16, T: 4, M: 226 });
const MAXIMUM_SHAPE = Object.freeze({ B: 1, Q: 192, T: 192, M: 402 });

function explicitKVAnswer({
  shapeMode,
  activeShape,
  tokenIds = [7, 8, 2],
  family = 'phone',
  familyId = 0,
} = {}) {
  return {
    family,
    familyId,
    questionTokenIds: [...Array(LOGICAL_SHAPE.Q).keys()],
    tokenIds: [...tokenIds],
    text: tokenIds.join(','),
    execution: 'explicit-kv-cache',
    decodeMode: 'explicit-kv-cache',
    decoderSeedExecutions: 1,
    decoderOrdinaryExecutions: tokenIds.length,
    decoderCacheStepExecutions: tokenIds.length - 1,
    activeShape: { ...activeShape },
    logicalShape: { ...LOGICAL_SHAPE },
    cacheShape: {
      initialPastLength: 1,
      finalPastLength: tokenIds.length + 1,
      sentinelSlots: 1,
    },
    shapeMode,
    decodeReports: tokenIds.map((_token, index) => ({
      operation: index === 0 ? 'explicit-kv-seed' : 'explicit-kv-step',
      position: index,
      pastLength: index + 1,
      presentLength: index + 2,
      sentinelMaskValue: 1,
      backendDecodeState: null,
    })),
  };
}

test('verify accepts exact explicit-KV active and maximum-padded views', () => {
  const result = verifyActivePaddedDecisionOracle(
    explicitKVAnswer({ shapeMode: 'active', activeShape: LOGICAL_SHAPE }),
    explicitKVAnswer({
      shapeMode: 'maximum-padded',
      activeShape: { ...MAXIMUM_SHAPE, T: LOGICAL_SHAPE.T },
    }),
  );

  assert.deepEqual(result, {
    exact: true,
    familyMatch: true,
    tokensMatch: true,
    textMatch: true,
    firstTokenDifference: null,
  });
});

test('verify reports the first active/padded greedy decision difference', () => {
  const result = verifyActivePaddedDecisionOracle(
    explicitKVAnswer({ shapeMode: 'active', activeShape: LOGICAL_SHAPE }),
    explicitKVAnswer({
      shapeMode: 'maximum-padded',
      activeShape: { ...MAXIMUM_SHAPE, T: LOGICAL_SHAPE.T },
      tokenIds: [7, 9, 2],
    }),
  );

  assert.equal(result.exact, false);
  assert.equal(result.familyMatch, true);
  assert.equal(result.tokensMatch, false);
  assert.equal(result.firstTokenDifference, 1);
});

test('verify rejects malformed explicit KV cache growth evidence', () => {
  const active = explicitKVAnswer({ shapeMode: 'active', activeShape: LOGICAL_SHAPE });
  active.decodeReports[1].presentLength = 9;
  assert.throws(() => verifyActivePaddedDecisionOracle(
    active,
    explicitKVAnswer({
      shapeMode: 'maximum-padded',
      activeShape: { ...MAXIMUM_SHAPE, T: LOGICAL_SHAPE.T },
    }),
  ), /explicit KV report 1/);
});

test('paired runner requests active and explicitly labelled maximum-padded shapes', async () => {
  const calls = [];
  const session = {
    async generate(options) {
      calls.push(options);
      return options.shapeMode === 'active'
        ? explicitKVAnswer({ shapeMode: 'active', activeShape: LOGICAL_SHAPE })
        : explicitKVAnswer({
          shapeMode: 'maximum-padded',
          activeShape: { ...MAXIMUM_SHAPE, T: LOGICAL_SHAPE.T },
        });
    },
  };
  const image = new Float32Array(320 * 672);
  const result = await runActivePaddedOracle({
    session,
    image,
    prompt: 'phone number last one',
    maxNewTokens: 3,
  });

  assert.equal(result.comparison.exact, true);
  assert.deepEqual(calls.map((call) => call.shapeMode), ['active', 'maximum-padded']);
  for (const call of calls) {
    assert.equal(call.image, image);
    assert.equal(call.preprocessed, true);
    assert.equal(call.maxNewTokens, 3);
  }
});
