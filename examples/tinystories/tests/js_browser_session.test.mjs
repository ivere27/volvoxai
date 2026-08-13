import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  createTinyStoriesDecodeState,
  resolveTinyStoriesSequenceContract,
} from '../browser_session.js';

function snapshot({ minimum = 1, maximum = 256, multipleOf = 1 } = {}) {
  return {
    outputNames: ['logits'],
    graph: {
      dimensions: {
        S: { name: 'S', min: minimum, max: maximum, multiple_of: multipleOf },
      },
      inputs: {
        tokens: { dtype: 'int32', shape: [1, 'S'] },
        positions: { dtype: 'int32', shape: [1, 'S'] },
      },
      tensors: {
        logits: { dtype: 'float32', shape: [1, 'S', 17] },
      },
    },
  };
}

test('TinyStories browser contract requires one bounded active sequence symbol', () => {
  assert.deepEqual(resolveTinyStoriesSequenceContract(snapshot({
    minimum: 3, maximum: 15, multipleOf: 4,
  })), {
    outputName: 'logits',
    sequenceSymbol: 'S',
    minimumSequenceCapacity: 4,
    maximumSequenceCapacity: 12,
    sequenceMultipleOf: 4,
    vocabularySize: 17,
  });

  const fixed = snapshot();
  fixed.graph.inputs.tokens.shape = [1, 256];
  fixed.graph.inputs.positions.shape = [1, 256];
  assert.throws(
    () => resolveTinyStoriesSequenceContract(fixed),
    /sharing one bounded symbol/,
  );
});

test('TinyStories decode state uses request-sized explicit views instead of maximum padding', () => {
  const contract = resolveTinyStoriesSequenceContract(snapshot({ maximum: 16 }));
  const state = createTinyStoriesDecodeState([3, 4, 5], {
    requestedNewTokens: 4,
    eosTokenId: 16,
    contract,
  });

  assert.equal(state.promptLength, 3);
  assert.equal(state.generationLimit, 4);
  assert.equal(state.sequenceCapacity, 7);
  assert.deepEqual(state.inputs.tokens.shape, [1, 7]);
  assert.deepEqual(state.inputs.positions.shape, [1, 7]);
  assert.strictEqual(state.inputs.tokens.data, state.tokens);
  assert.strictEqual(state.inputs.positions.data, state.positions);
  assert.deepEqual([...state.tokens], [3, 4, 5, 16, 16, 16, 16]);
  assert.deepEqual([...state.positions], [0, 1, 2, 3, 4, 5, 6]);

  state.tokens[3] = 9;
  assert.equal(state.inputs.tokens.data[3], 9, 'decode steps update the retained shaped view');
});

test('TinyStories decode state respects legal capacity multiples and maximum bounds', () => {
  const contract = resolveTinyStoriesSequenceContract(snapshot({
    minimum: 3, maximum: 15, multipleOf: 4,
  }));
  const state = createTinyStoriesDecodeState([1, 2, 3, 4, 5], {
    requestedNewTokens: 99,
    eosTokenId: 16,
    contract,
  });
  assert.equal(state.generationLimit, 7);
  assert.equal(state.sequenceCapacity, 12);
  assert.throws(() => createTinyStoriesDecodeState([], {
    requestedNewTokens: 1, eosTokenId: 16, contract,
  }), /at least one token/);
  assert.throws(() => createTinyStoriesDecodeState([1], {
    requestedNewTokens: 1.5, eosTokenId: 16, contract,
  }), /non-negative safe integer/);
});

test('TinyStories browser demo uses the decode lifecycle without static ordinary execution', async () => {
  const html = await readFile(new URL('../../tinystories.html', import.meta.url), 'utf8');
  assert.match(html, /resolveTinyStoriesSequenceContract\(snapshot\)/);
  assert.match(html, /createTinyStoriesDecodeState\(promptTokenIds/);
  assert.match(html, /context\.decode\.seed\(decodeState\.inputs/);
  assert.match(html, /context\.decode\.step\(/);
  assert.doesNotMatch(html, /context\.execute\(/);
  assert.doesNotMatch(html, /\bSEQ\b/);
});
