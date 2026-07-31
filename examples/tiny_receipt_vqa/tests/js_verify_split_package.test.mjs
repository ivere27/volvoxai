import assert from 'node:assert/strict';
import test from 'node:test';

import {
  generateAutoregressiveTokens,
  resolveFixtureInputSemantics,
} from '../tools/verify_split_package.mjs';

test('verify fixture bindings honor source_name aliases for generated runtime inputs', () => {
  assert.deepEqual(resolveFixtureInputSemantics({
    inputs: {
      input0: { source_name: 'decoder_input_ids' },
      input1: { source_name: 'memory' },
      input2: { source_name: 'memory_padding_mask' },
      input3: { source_name: 'family_ids' },
      '@runtime/158:QBatchMatMul.keep': { source_name: 'v4_keep' },
    },
  }, 'decoder'), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    family_ids: 'input3',
    v4_keep: '@runtime/158:QBatchMatMul.keep',
  });
});

test('verify generation executes each updated prefix instead of reading future BOS-only rows',
  async () => {
    const snapshots = [];
    const decisions = [7, 8, 2];
    const generated = await generateAutoregressiveTokens({
      contract: { kind: 'token_ids', name: 'token_ids', length: 192 },
      decoderInputs: {
        decoder_input_ids: 'input0',
        v4_keep: '@runtime/keep',
      },
      baseInputs: { memory: Float32Array.of(1) },
      tokenCount: 8,
      padTokenId: 0,
      bosTokenId: 1,
      eosTokenId: 2,
      execute: async (inputs) => {
        const step = snapshots.length;
        snapshots.push({
          ids: [...inputs.input0.slice(0, 5)],
          keep: [...inputs['@runtime/keep'].slice(0, 5)],
        });
        const output = new Int32Array(192);
        output[step] = decisions[step];
        return { outputs: { token_ids: output }, elapsed: step + 1 };
      },
    });

    assert.deepEqual(generated, { tokens: [7, 8, 2], elapsed: 6 });
    assert.deepEqual(snapshots, [
      { ids: [1, 0, 0, 0, 0], keep: [1, 0, 0, 0, 0] },
      { ids: [1, 7, 0, 0, 0], keep: [1, 1, 0, 0, 0] },
      { ids: [1, 7, 8, 0, 0], keep: [1, 1, 1, 0, 0] },
    ]);
  });

test('verify generation caps new tokens at decoder length minus one', async () => {
  await assert.rejects(generateAutoregressiveTokens({
    contract: { kind: 'token_ids', name: 'token_ids', length: 192 },
    decoderInputs: { decoder_input_ids: 'ids' },
    baseInputs: {},
    execute: async () => ({ outputs: { token_ids: new Int32Array(192) } }),
    tokenCount: 192,
    padTokenId: 0,
    bosTokenId: 1,
    eosTokenId: 2,
  }), /integer in \[1, 191\]/);
});
