import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

import {
  TinyReceiptByteFallbackBPEVocab,
  TinyReceiptSplitSession,
  TINY_RECEIPT_SPLIT_FAMILY_ORDER,
  TINY_RECEIPT_SPLIT_PACKAGE_FORMAT,
} from '../TinyReceiptSplitSession.js';


const PACKAGE_URL = 'https://example.test/tiny-split/package_manifest.json';
const SHA = 'a'.repeat(64);
const BPE_HASH = '3d38e3d05ed255191d504c02c91048795c403aec544c952de0b94f3c8fa082b5';

function asset(path) {
  return { path, bytes: 1, sha256: SHA };
}

function manifest({ routingMode = 'specialized', specializedFamilyId = 0 } = {}) {
  const value = {
    format: TINY_RECEIPT_SPLIT_PACKAGE_FORMAT,
    assets: {
      config: asset('config.json'),
      vocab: asset('vocab.json'),
    },
    tokenizer: {
      type: 'char-vocab',
      version: 1,
      itos_key: 'itos',
      token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
    },
    preprocessing: {
      layout: 'NCHW',
      shape: [1, 1, 320, 672],
      color: 'grayscale',
    },
    families: {
      auto_id: -1,
      ordered_names: [...TINY_RECEIPT_SPLIT_FAMILY_ORDER],
      name_to_id: Object.fromEntries(
        TINY_RECEIPT_SPLIT_FAMILY_ORDER.map((name, index) => [name, index]),
      ),
    },
    generation: {
      strategy: 'greedy-autoregressive',
      decoder_input_length: 192,
      maximum_new_tokens: 191,
      bos_token_id: 1,
      eos_token_id: 2,
      pad_token_id: 0,
      decoder_output: 'token_ids',
      token_ids_row: 'prefix_length_minus_one',
      tie_policy: 'first-index',
    },
    graphs: {
      encoder: {
        graph: asset('encoder/graph.json'),
        weights: asset('encoder/model.safetensors'),
        export_report: asset('encoder/export_report.json'),
        inputs: {
          image: 'input0',
          question_ids: 'input1',
        },
        outputs: {
          memory: 'memory',
          memory_padding_mask: 'memory_padding_mask',
          router_logits: 'router_logits',
          selected_family_ids: 'selected_family_ids',
        },
      },
      decoder: {
        graph: asset('decoder/graph.json'),
        weights: asset('decoder/model.safetensors'),
        export_report: asset('decoder/export_report.json'),
        inputs: {
          decoder_input_ids: 'input0',
          memory: 'input1',
          memory_padding_mask: 'input2',
          v4_keep: 'v4_keep',
        },
        outputs: { token_ids: 'token_ids' },
      },
    },
    mask_semantics: { memory_padding_mask: 'nonzero_means_blocked' },
  };
  if (routingMode === 'runtime') {
    value.graphs.encoder.inputs.family_ids = 'input2';
    value.graphs.decoder.inputs.family_ids = 'input3';
    value.routing = {
      mode: 'runtime',
      family_inputs: { encoder: 'input2', decoder: 'input3' },
    };
  } else {
    value.routing = { mode: 'specialized', family_id: specializedFamilyId };
  }
  return value;
}

function bpeVocabulary() {
  const structural = [
    '<field>', '</field>', '<value>', '</value>', '<op>', '</op>', '<answer>', '</answer>',
  ];
  const digits = [...Array(10).keys()].map(String);
  const bytes = [...Array(256).keys()]
    .map((value) => `<0x${value.toString(16).toUpperCase().padStart(2, '0')}>`);
  const itos = [
    '<pad>', '<bos>', '<eos>', '<unk>', ...structural, ...digits, ...bytes,
    'a', 'b', 'c', 'C', 'f', 'é', ' ', '\t', 'bc', 'abc', 'ab',
  ];
  const unused = [];
  while (itos.length < 1536) {
    const token = `<unused_${String(unused.length).padStart(4, '0')}>`;
    unused.push(token);
    itos.push(token);
  }
  return {
    type: 'byte_fallback_bpe',
    version: 1,
    vocab_size: 1536,
    itos,
    merges: [['b', 'c'], ['a', 'bc'], ['a', 'b']],
    normalization: 'NFC',
    atomic_tokens: [...structural, ...digits],
    byte_tokens: bytes,
    unused_tokens: unused,
    special_tokens: { pad: '<pad>', bos: '<bos>', eos: '<eos>', unk: '<unk>' },
    tokenizer_hash: BPE_HASH,
  };
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
}

function rehashBpe(value) {
  const payload = structuredClone(value);
  delete payload.tokenizer_hash;
  value.tokenizer_hash = createHash('sha256').update(canonicalJson(payload)).digest('hex');
  return value;
}

function bpeManifest({ hoistedKeep = false } = {}) {
  const value = manifest({ routingMode: 'runtime' });
  value.tokenizer = {
    type: 'byte_fallback_bpe',
    version: 1,
    vocab_size: 1536,
    normalization: 'NFC',
    tokenizer_hash: BPE_HASH,
    itos_key: 'itos',
    merges_key: 'merges',
    token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
  };
  delete value.generation.decoder_output;
  delete value.generation.token_ids_row;
  value.generation.logits_row = 'prefix_length_minus_one';
  if (!hoistedKeep) delete value.graphs.decoder.inputs.v4_keep;
  value.graphs.decoder.outputs = { logits: 'logits' };
  return value;
}

function tensor(name, shape, dtype, isInput = false) {
  return { name, shape, dtype, isInput, quantization: null };
}

function encoderGraph(routingMode = 'specialized') {
  const tensors = [
    tensor('input0', [1, 1, 320, 672], 'float32', true),
    tensor('input1', [1, 192], 'int32', true),
    tensor('memory', [1, 402, 320], 'float32'),
    tensor('memory_padding_mask', [1, 402], 'int32'),
    tensor('router_logits', [1, 8], 'float32'),
    tensor('selected_family_ids', [1], 'int32'),
  ];
  if (routingMode === 'runtime') {
    tensors.push(tensor('input2', [1], 'int32', true));
  }
  return {
    kind: 'encoder',
    tensors: new Map(tensors.map((value) => [value.name, value])),
    nodes: [],
    outputNames: [
      'memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids',
    ],
  };
}

function decoderGraph(routingMode = 'specialized') {
  const tensors = [
    tensor('input0', [1, 192], 'int32', true),
    tensor('input1', [1, 402, 320], 'float32', true),
    tensor('input2', [1, 402], 'int32', true),
    tensor('v4_keep', [1, 192], 'int32', true),
    tensor('token_ids', [1, 192], 'int32'),
  ];
  if (routingMode === 'runtime') {
    tensors.push(tensor('input3', [1], 'int32', true));
  }
  return {
    kind: 'decoder',
    tensors: new Map(tensors.map((value) => [value.name, value])),
    nodes: [],
    outputNames: ['token_ids'],
  };
}

function logitsDecoderGraph(vocabSize = 1536, { hoistedKeep = false } = {}) {
  const graph = decoderGraph('runtime');
  if (!hoistedKeep) graph.tensors.delete('v4_keep');
  graph.tensors.delete('token_ids');
  graph.tensors.set('logits', tensor('logits', [1, 192, vocabSize], 'float32'));
  graph.outputNames = ['logits'];
  return graph;
}

function fakeFetch(resources) {
  return async (url) => {
    const value = resources.get(String(url));
    if (value == null) return { ok: false, status: 404 };
    return { ok: true, json: async () => value };
  };
}

function copyInputs(inputs) {
  return Object.fromEntries(
    Object.entries(inputs).map(([name, value]) => [name, value.slice()]),
  );
}

function executionResult(outputs) {
  return {
    output(name) {
      assert.ok(outputs[name]);
      return { read: async () => outputs[name] };
    },
    async close() {
      for (const value of Object.values(outputs)) value.fill(-99);
    },
  };
}

function retainedRowContext(execute, {
  onReset = () => {},
  onClose = () => {},
} = {}) {
  let seeded = false;
  return {
    async execute() {
      throw new Error('decoder ordinary execution must not be used');
    },
    decode: {
      async reset() {
        seeded = false;
        onReset();
      },
      async seed(inputs) {
        assert.equal(seeded, false, 'decoder must reset before seed');
        seeded = true;
        return execute(inputs, { operation: 'seed', position: 0 });
      },
      async step(inputs, { position } = {}) {
        assert.equal(seeded, true, 'decoder step requires a seed');
        assert.ok(Number.isInteger(position) && position >= 1);
        return execute(inputs, { operation: 'step', position });
      },
    },
    async close() {
      seeded = false;
      onClose();
    },
  };
}

async function syntheticSession({
  routingMode = 'specialized',
  specializedFamilyId = 0,
  autoSelectedFamilyId = 0,
  backend = 'wasm',
  decodePolicy = 'auto',
} = {}) {
  const vocabulary = ['<pad>', '<bos>', '<eos>', '<unk>', 'A'];
  while (vocabulary.length < 760) vocabulary.push(`token-${vocabulary.length}`);
  const resources = new Map([
    [PACKAGE_URL, manifest({ routingMode, specializedFamilyId })],
    ['https://example.test/tiny-split/vocab.json', { itos: vocabulary }],
  ]);
  const calls = [];
  const loads = [];
  const contextOptions = [];
  const decodeResets = [];
  const closed = { models: 0, compiled: 0, contexts: 0 };
  const control = { failNextDecoder: false };
  let decoderStep = 0;
  const runtime = {
    createModel(graph) {
      return {
        async compile() {
          return {
            backend,
            async createContext(options) {
              contextOptions.push({ kind: graph.kind, options });
              if (graph.kind === 'encoder') {
                return {
                  async execute(inputs) {
                    calls.push({ kind: graph.kind, inputs: copyInputs(inputs) });
                    const requestedFamilyId = inputs.input2?.[0];
                    const selectedFamilyId = requestedFamilyId == null || requestedFamilyId === -1
                      ? autoSelectedFamilyId
                      : requestedFamilyId;
                    const memory = new Float32Array(402 * 320);
                    memory[0] = 7;
                    const mask = new Int32Array(402);
                    mask[400] = 1;
                    return executionResult({
                      memory,
                      memory_padding_mask: mask,
                      router_logits: Float32Array.of(0, 1, 5, 1, 0, 0, 0, 0),
                      selected_family_ids: Int32Array.of(selectedFamilyId),
                    });
                  },
                  async close() { closed.contexts++; },
                };
              }
              const executeDecoder = async (inputs, decode) => {
                calls.push({
                  kind: graph.kind,
                  inputs: copyInputs(inputs),
                  ...decode,
                });
                if (control.failNextDecoder) {
                  control.failNextDecoder = false;
                  throw new Error('synthetic decoder failure');
                }
                const tokenIds = new Int32Array(192);
                if (decoderStep === 0) {
                  tokenIds[0] = 4;
                  tokenIds[1] = 9;
                } else {
                  tokenIds[1] = 2;
                }
                decoderStep++;
                return executionResult({ token_ids: tokenIds });
              };
              if (!options?.decode) {
                return {
                  execute: (inputs) => executeDecoder(inputs, {
                    operation: 'execute',
                    position: null,
                  }),
                  async close() { closed.contexts++; },
                };
              }
              return retainedRowContext(executeDecoder, {
                onReset() {
                  decoderStep = 0;
                  decodeResets.push('reset');
                },
                onClose() { closed.contexts++; },
              });
            },
            async close() { closed.compiled++; },
          };
        },
        async close() { closed.models++; },
      };
    },
  };
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    decodePolicy,
    graphLoader: async ({ kind, graphUrl, weightsUrl }) => {
      loads.push({ kind, graphUrl, weightsUrl });
      return kind === 'encoder' ? encoderGraph(routingMode) : decoderGraph(routingMode);
    },
  });
  return { session, calls, loads, contextOptions, decodeResets, closed, control };
}

test('byte-fallback BPE matches Python NFC, ranked-merge, boundary, and UTF-8 vectors', async () => {
  const value = bpeVocabulary();
  const vocab = await TinyReceiptByteFallbackBPEVocab.fromJSON(
    value,
    { pad: 0, bos: 1, eos: 2, unk: 3 },
    { vocabSize: 1536, normalization: 'NFC', tokenizerHash: BPE_HASH },
  );
  const vectors = [
    ['abc', [287]],
    ['Cafe\u0301 90', [281, 278, 282, 283, 284, 21, 12]],
    ['<field>abc</field>\t🙂', [4, 287, 5, 285, 262, 181, 175, 152]],
    ['literal <0xEA> text', [130, 127, 138, 123, 136, 278, 130, 284,
      82, 12, 142, 91, 87, 84, 284, 138, 123, 142, 138]],
    ['a\u00a0b\u3000c', [278, 216, 182, 279, 249, 150, 150, 280]],
  ];
  for (const [text, expected] of vectors) {
    assert.deepEqual(vocab.encode(text), expected);
    assert.equal(vocab.decode(expected), text.normalize('NFC'));
  }
  assert.deepEqual(
    vocab.encode('Cafe\u0301 90', { addEos: true, maxLength: 7 }),
    [281, 278, 282, 283, 284, 21, 2],
  );
  assert.deepEqual(
    vocab.encode('unseen🙂', { addBos: true, addEos: true, maxLength: 4 }),
    [1, 139, 132, 2],
  );
  // Raw BPE preserves explicit boundaries; deployed question encoding first
  // applies Python clean_text (trim + collapse all Python `\s` characters).
  assert.deepEqual(vocab.encode('a   b'), [278, 284, 284, 284, 279]);
  assert.deepEqual(
    vocab.encodeQuestion(' \ta  \n\u001Cb\u3000 ', 192),
    [278, 284, 279, 2],
  );

  const tampered = structuredClone(value);
  tampered.itos[278] = 'changed';
  await assert.rejects(
    TinyReceiptByteFallbackBPEVocab.fromJSON(
      tampered,
      { pad: 0, bos: 1, eos: 2, unk: 3 },
    ),
    /tokenizer_hash does not match/,
  );
});

test('BPE release vocabulary rejects weakened atomic, byte, special, and unused declarations', async () => {
  const tokenIds = { pad: 0, bos: 1, eos: 2, unk: 3 };

  const missingBytes = bpeVocabulary();
  delete missingBytes.byte_tokens;
  await assert.rejects(
    TinyReceiptByteFallbackBPEVocab.fromJSON(missingBytes, tokenIds),
    /must contain exactly/,
  );

  const changedAtomic = bpeVocabulary();
  changedAtomic.atomic_tokens = changedAtomic.atomic_tokens.slice(1);
  rehashBpe(changedAtomic);
  await assert.rejects(
    TinyReceiptByteFallbackBPEVocab.fromJSON(changedAtomic, tokenIds),
    /eight structural tokens followed by digits/,
  );

  const changedSpecial = bpeVocabulary();
  changedSpecial.special_tokens.eos = '<end>';
  rehashBpe(changedSpecial);
  await assert.rejects(
    TinyReceiptByteFallbackBPEVocab.fromJSON(changedSpecial, tokenIds),
    /canonical literals/,
  );

  const duplicateUnused = bpeVocabulary();
  duplicateUnused.unused_tokens[1] = duplicateUnused.unused_tokens[0];
  rehashBpe(duplicateUnused);
  await assert.rejects(
    TinyReceiptByteFallbackBPEVocab.fromJSON(duplicateUnused, tokenIds),
    /unique non-empty strings/,
  );
});

test('BPE package tokenizer requires the exact published eight-field contract', async () => {
  const value = bpeManifest();
  delete value.tokenizer.merges_key;
  const resources = new Map([
    [PACKAGE_URL, value],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() { throw new Error('must not compile'); } },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /tokenizer must contain exactly/);
});

test('BPE1536 split session derives logits width and runs the raw decoder ABI', async () => {
  const resources = new Map([
    [PACKAGE_URL, bpeManifest()],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  const calls = [];
  const contextOptions = [];
  let decoderStep = 0;
  const runtime = {
    createModel(graph) {
      return {
        async compile() {
          return {
            backend: 'wasm',
            async createContext(options) {
              contextOptions.push({ kind: graph.kind, options });
              if (graph.kind === 'encoder') {
                return {
                  async execute(inputs) {
                    calls.push({ kind: graph.kind, inputs: copyInputs(inputs) });
                    return executionResult({
                      memory: new Float32Array(402 * 320),
                      memory_padding_mask: new Int32Array(402),
                      router_logits: Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
                      selected_family_ids: Int32Array.of(2),
                    });
                  },
                  async close() {},
                };
              }
              return retainedRowContext(async (inputs, decode) => {
                calls.push({ kind: graph.kind, inputs: copyInputs(inputs), ...decode });
                const logits = new Float32Array(192 * 1536);
                if (decoderStep++ === 0) {
                  // Equal maxima prove the declared first-index tie policy.
                  logits[287] = 9;
                  logits[288] = 9;
                } else {
                  logits[1536 + 2] = 10;
                }
                return executionResult({ logits });
              }, { onReset() { decoderStep = 0; } });
            },
            async close() {},
          };
        },
        async close() {},
      };
    },
  };
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async ({ kind }) => kind === 'encoder'
      ? encoderGraph('runtime')
      : logitsDecoderGraph(),
  });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'Cafe\u0301 90',
    family: 'auto',
    maxNewTokens: 3,
  });
  assert.deepEqual(result.questionTokenIds, [281, 278, 282, 283, 284, 21, 12, 2]);
  assert.deepEqual(result.tokenIds, [287, 2]);
  assert.equal(result.text, 'abc');
  assert.equal(result.family, 'store');
  assert.equal(result.execution, 'context-decode-retained-row');
  assert.equal(result.decodeMode, 'incremental-row-required');
  assert.equal(result.decoderSeedExecutions, 1);
  assert.equal(result.decoderRowExecutions, 1);
  assert.equal(result.decoderOrdinaryExecutions, 0);
  assert.deepEqual(contextOptions, [
    { kind: 'encoder', options: undefined },
    {
      kind: 'decoder',
      options: {
        decode: {
          changedInputs: ['input0'],
          rowMode: 'required',
          requireIncremental: true,
        },
      },
    },
  ]);
  assert.deepEqual(Object.keys(calls[1].inputs).sort(), ['input0', 'input1', 'input2', 'input3']);
  assert.deepEqual(
    calls.slice(1).map(({ operation, position }) => ({ operation, position })),
    [
      { operation: 'seed', position: 0 },
      { operation: 'step', position: 1 },
    ],
  );
  assert.deepEqual(Object.keys(calls[2].inputs), ['input0']);
  assert.deepEqual([...calls[2].inputs.input0.slice(0, 3)], [1, 287, 0]);
  await session.close();
});

test('logits decoder accepts and updates a manifest-declared hoisted v4_keep input', async () => {
  const resources = new Map([
    [PACKAGE_URL, bpeManifest({ hoistedKeep: true })],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  const decoderCalls = [];
  const runtime = {
    createModel(graph) {
      return {
        async compile() {
          return {
            backend: 'wasm',
            async createContext() {
              if (graph.kind === 'encoder') {
                return {
                  async execute(inputs) {
                    assert.ok(inputs);
                    return executionResult({
                      memory: new Float32Array(402 * 320),
                      memory_padding_mask: new Int32Array(402),
                      router_logits: new Float32Array(8),
                      selected_family_ids: Int32Array.of(0),
                    });
                  },
                  async close() {},
                };
              }
              return retainedRowContext(async (inputs, decode) => {
                decoderCalls.push({ inputs: copyInputs(inputs), ...decode });
                const logits = new Float32Array(192 * 1536);
                logits[(decoderCalls.length - 1) * 1536 +
                  (decoderCalls.length === 1 ? 287 : 2)] = 1;
                return executionResult({ logits });
              });
            },
            async close() {},
          };
        },
        async close() {},
      };
    },
  };
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async ({ kind }) => kind === 'encoder'
      ? encoderGraph('runtime')
      : logitsDecoderGraph(1536, { hoistedKeep: true }),
  });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'abc',
    maxNewTokens: 2,
  });
  assert.deepEqual(result.tokenIds, [287, 2]);
  assert.deepEqual(
    decoderCalls.map(({ operation, position }) => ({ operation, position })),
    [
      { operation: 'seed', position: 0 },
      { operation: 'step', position: 1 },
    ],
  );
  assert.deepEqual(
    Object.keys(decoderCalls[0].inputs).sort(),
    ['input0', 'input1', 'input2', 'input3', 'v4_keep'],
  );
  assert.deepEqual(Object.keys(decoderCalls[1].inputs).sort(), ['input0', 'v4_keep']);
  assert.deepEqual([...decoderCalls[0].inputs.v4_keep.slice(0, 3)], [1, 0, 0]);
  assert.deepEqual([...decoderCalls[1].inputs.v4_keep.slice(0, 3)], [1, 1, 0]);
  await session.close();
});

test('BPE1536 raw decoder rejects a logits width that disagrees with vocab.json', async () => {
  const resources = new Map([
    [PACKAGE_URL, bpeManifest()],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  const session = await TinyReceiptSplitSession.load({
    runtime: {
      createModel() {
        return {
          async compile() {
            return {
              async createContext() { return { async close() {} }; },
              async close() {},
            };
          },
          async close() {},
        };
      },
    },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async ({ kind }) => kind === 'encoder'
      ? encoderGraph('runtime')
      : logitsDecoderGraph(760),
  });
  await assert.rejects(session.preload(), /float32 \[1,192,1536\]/);
  await session.close();
});

test('split session seeds once and runs retained decoder rows with blocked-mask semantics', async () => {
  const {
    session,
    calls,
    loads,
    contextOptions,
    decodeResets,
    closed,
  } = await syntheticSession();
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'A',
    family: 'auto',
    maxNewTokens: 3,
  });

  assert.deepEqual(loads.map(({ kind }) => kind), ['encoder', 'decoder']);
  assert.deepEqual(calls.map(({ kind }) => kind), ['encoder', 'decoder', 'decoder']);
  assert.deepEqual(result.tokenIds, [4, 2]);
  assert.equal(result.text, 'A');
  assert.equal(result.family, 'phone');
  assert.equal(result.familyId, 0);
  assert.equal(result.requestedFamily, 'auto');
  assert.equal(result.stoppedAtEos, true);
  assert.equal(result.execution, 'context-decode-retained-row');
  assert.equal(result.decodeMode, 'incremental-row-required');
  assert.equal(result.decoderSeedExecutions, 1);
  assert.equal(result.decoderRowExecutions, 1);
  assert.equal(result.decoderOrdinaryExecutions, 0);
  assert.deepEqual(decodeResets, ['reset']);
  assert.deepEqual(contextOptions, [
    { kind: 'encoder', options: undefined },
    {
      kind: 'decoder',
      options: {
        decode: {
          changedInputs: ['input0', 'v4_keep'],
          rowMode: 'required',
          requireIncremental: true,
        },
      },
    },
  ]);
  assert.deepEqual(Object.keys(calls[0].inputs).sort(), ['input0', 'input1']);
  assert.deepEqual([...calls[0].inputs.input1.slice(0, 4)], [4, 2, 0, 0]);
  assert.deepEqual(Object.keys(calls[1].inputs).sort(), ['input0', 'input1', 'input2', 'v4_keep']);
  assert.deepEqual(
    calls.slice(1).map(({ operation, position }) => ({ operation, position })),
    [
      { operation: 'seed', position: 0 },
      { operation: 'step', position: 1 },
    ],
  );
  assert.equal(calls[1].inputs.input1[0], 7);
  assert.equal(calls[1].inputs.input2[400], 1);
  assert.deepEqual([...calls[1].inputs.input0.slice(0, 3)], [1, 0, 0]);
  assert.deepEqual([...calls[1].inputs.v4_keep.slice(0, 3)], [1, 0, 0]);
  assert.deepEqual(Object.keys(calls[2].inputs).sort(), ['input0', 'v4_keep']);
  assert.deepEqual([...calls[2].inputs.input0.slice(0, 3)], [1, 4, 0]);
  assert.deepEqual([...calls[2].inputs.v4_keep.slice(0, 3)], [1, 1, 0]);
  // Stable host copies survive fake ExecutionResult.close() overwriting its buffers.
  assert.equal(result.routerLogits[2], 5);

  await session.close();
  assert.deepEqual(closed, { models: 2, compiled: 2, contexts: 2 });
});

test('retained decoder resets for zero-token, repeated, and recovered generations', async () => {
  const {
    session,
    calls,
    decodeResets,
    control,
  } = await syntheticSession();
  const request = {
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'A',
    family: 'auto',
  };

  const empty = await session.generate({ ...request, maxNewTokens: 0 });
  assert.deepEqual(empty.tokenIds, []);
  assert.equal(empty.text, '');
  assert.equal(empty.stoppedAtEos, false);
  assert.equal(empty.execution, 'context-decode-retained-row');
  assert.equal(empty.decoderSeedExecutions, 0);
  assert.equal(empty.decoderRowExecutions, 0);
  assert.equal(empty.decoderOrdinaryExecutions, 0);
  assert.deepEqual(calls.map(({ kind }) => kind), ['encoder']);
  assert.deepEqual(decodeResets, ['reset']);

  const first = await session.generate({ ...request, maxNewTokens: 3 });
  const second = await session.generate({ ...request, maxNewTokens: 3 });
  assert.deepEqual(first.tokenIds, [4, 2]);
  assert.deepEqual(second.tokenIds, first.tokenIds);
  assert.equal(second.text, first.text);
  assert.deepEqual(decodeResets, ['reset', 'reset', 'reset']);

  control.failNextDecoder = true;
  await assert.rejects(
    session.generate({ ...request, maxNewTokens: 3 }),
    /synthetic decoder failure/,
  );
  const recovered = await session.generate({ ...request, maxNewTokens: 3 });
  assert.deepEqual(recovered.tokenIds, [4, 2]);
  assert.equal(recovered.text, 'A');
  assert.deepEqual(decodeResets, ['reset', 'reset', 'reset', 'reset', 'reset']);
  await session.close();
});

test('auto decode policy keeps non-WASM backends on ordinary fixed-length forwards', async () => {
  const {
    session,
    calls,
    contextOptions,
    decodeResets,
  } = await syntheticSession({ backend: 'webgpu' });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'A',
    maxNewTokens: 3,
  });

  assert.deepEqual(result.tokenIds, [4, 2]);
  assert.equal(result.execution, 'fixed-length-ordinary-forward');
  assert.equal(result.decodeMode, 'ordinary-forward');
  assert.equal(result.decoderSeedExecutions, 0);
  assert.equal(result.decoderRowExecutions, 0);
  assert.equal(result.decoderOrdinaryExecutions, 2);
  assert.deepEqual(decodeResets, []);
  assert.deepEqual(contextOptions, [
    { kind: 'encoder', options: undefined },
    { kind: 'decoder', options: undefined },
  ]);
  assert.deepEqual(
    calls.slice(1).map(({ operation, position }) => ({ operation, position })),
    [
      { operation: 'execute', position: null },
      { operation: 'execute', position: null },
    ],
  );
  assert.deepEqual(
    calls.slice(1).map(({ inputs }) => Object.keys(inputs).sort()),
    [
      ['input0', 'input1', 'input2', 'v4_keep'],
      ['input0', 'input1', 'input2', 'v4_keep'],
    ],
  );
  await session.close();
});

test('ordinary decode policy disables retained rows even when the backend is WASM', async () => {
  const {
    session,
    calls,
    contextOptions,
    decodeResets,
  } = await syntheticSession({ backend: 'wasm', decodePolicy: 'ordinary' });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'A',
    maxNewTokens: 3,
  });

  assert.deepEqual(result.tokenIds, [4, 2]);
  assert.equal(result.execution, 'fixed-length-ordinary-forward');
  assert.equal(result.decodeMode, 'ordinary-forward');
  assert.equal(result.decoderSeedExecutions, 0);
  assert.equal(result.decoderRowExecutions, 0);
  assert.equal(result.decoderOrdinaryExecutions, 2);
  assert.deepEqual(decodeResets, []);
  assert.deepEqual(contextOptions, [
    { kind: 'encoder', options: undefined },
    { kind: 'decoder', options: undefined },
  ]);
  assert.deepEqual(
    calls.slice(1).map(({ operation }) => operation),
    ['execute', 'execute'],
  );
  await session.close();
});

test('split session rejects an unknown decode policy before graph compilation', async () => {
  await assert.rejects(
    syntheticSession({ decodePolicy: 'unsupported' }),
    /decodePolicy must be 'auto', 'required', or 'ordinary'/,
  );
});

test('split session preserves the specialized ABI and validates its fixed family', async () => {
  const { session, calls } = await syntheticSession();
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: '',
    family: 'phone',
    maxNewTokens: 1,
  });
  assert.deepEqual(Object.keys(calls[0].inputs).sort(), ['input0', 'input1']);
  assert.deepEqual(Object.keys(calls[1].inputs).sort(), ['input0', 'input1', 'input2', 'v4_keep']);
  assert.equal(result.family, 'phone');
  assert.equal(result.requestedFamily, 'phone');
});

test('specialized split session rejects a selected family that contradicts its manifest', async () => {
  const { session, calls } = await syntheticSession({
    specializedFamilyId: 1,
    autoSelectedFamilyId: 0,
  });
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: '',
    family: 'phone',
    maxNewTokens: 1,
  }), /fixed to family 'address'/);
  assert.equal(calls.length, 0);

  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: '',
    family: 'auto',
    maxNewTokens: 1,
  }), /does not match specialized package routing/);
  await session.close();
});

test('runtime-selectable split session sends AUTO and the selected family through both graphs', async () => {
  const { session, calls } = await syntheticSession({
    routingMode: 'runtime',
    autoSelectedFamilyId: 2,
  });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: '',
    family: 'auto',
    maxNewTokens: 1,
  });

  assert.deepEqual(Object.keys(calls[0].inputs).sort(), ['input0', 'input1', 'input2']);
  assert.deepEqual([...calls[0].inputs.input2], [-1]);
  assert.deepEqual(
    Object.keys(calls[1].inputs).sort(),
    ['input0', 'input1', 'input2', 'input3', 'v4_keep'],
  );
  assert.deepEqual([...calls[1].inputs.input3], [2]);
  assert.equal(result.family, 'store');
  assert.equal(result.familyId, 2);
  assert.equal(result.requestedFamily, 'auto');
  await session.close();
});

test('runtime-selectable split session sends an explicit family to encoder and decoder', async () => {
  const { session, calls } = await syntheticSession({
    routingMode: 'runtime',
    autoSelectedFamilyId: 2,
  });
  const result = await session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: '',
    family: 'address',
    maxNewTokens: 1,
  });

  assert.deepEqual([...calls[0].inputs.input2], [1]);
  assert.deepEqual([...calls[1].inputs.input3], [1]);
  assert.equal(result.family, 'address');
  assert.equal(result.familyId, 1);
  assert.equal(result.requestedFamily, 'address');
  await session.close();
});

test('split session rejects mixed specialized and runtime-selectable graph ABIs', async () => {
  const runtimeEncoder = manifest();
  runtimeEncoder.graphs.encoder.inputs.family_ids = 'input2';
  const runtimeDecoder = manifest();
  runtimeDecoder.graphs.decoder.inputs.family_ids = 'input3';

  for (const mixed of [runtimeEncoder, runtimeDecoder]) {
    await assert.rejects(TinyReceiptSplitSession.load({
      runtime: { createModel() {} },
      packageUrl: PACKAGE_URL,
      fetch: fakeFetch(new Map([[PACKAGE_URL, mixed]])),
      graphLoader: async () => { throw new Error('must not load'); },
    }), /graphs\.(encoder|decoder)\.inputs must contain exactly/);
  }
});

test('split session rejects legacy and physically inconsistent routing metadata', async () => {
  const legacy = manifest();
  delete legacy.routing;
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(new Map([[PACKAGE_URL, legacy]])),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /requires explicit routing/);

  const inconsistent = manifest({ routingMode: 'runtime' });
  inconsistent.routing.family_inputs.encoder = 'family_ids';
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(new Map([[PACKAGE_URL, inconsistent]])),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /must match graphs\.encoder\.inputs\.family_ids/);
});

test('split session rejects logits output without the matching generation contract', async () => {
  const invalidLogits = manifest();
  delete invalidLogits.graphs.decoder.outputs.token_ids;
  invalidLogits.graphs.decoder.outputs.logits = 'logits';

  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(new Map([[PACKAGE_URL, invalidLogits]])),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /logits decoder generation must select prefix_length_minus_one/);
});

test('split session rejects an undeclared family tensor in a loaded graph', async () => {
  const vocabulary = ['<pad>', '<bos>', '<eos>', '<unk>'];
  while (vocabulary.length < 760) vocabulary.push(`token-${vocabulary.length}`);
  const resources = new Map([
    [PACKAGE_URL, manifest()],
    ['https://example.test/tiny-split/vocab.json', { itos: vocabulary }],
  ]);
  const graph = encoderGraph();
  const legacyFamily = tensor('family_ids', [1], 'int32', true);
  graph.tensors.set(legacyFamily.name, legacyFamily);
  const session = await TinyReceiptSplitSession.load({
    runtime: {
      createModel() {
        throw new Error('model creation must not begin');
      },
    },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async () => graph,
  });
  await assert.rejects(session.preload(), /encoder graph inputs must be exactly input0, input1/);
  await session.close();
});

test('split session rejects legacy mask semantics and more than T-1 generated tokens', async () => {
  const invalid = manifest();
  invalid.mask_semantics.memory_padding_mask = 'nonzero_means_keep';
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(new Map([[PACKAGE_URL, invalid]])),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /nonzero_means_blocked/);

  const { session } = await syntheticSession();
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672),
    preprocessed: true,
    prompt: 'A',
    maxNewTokens: 192,
  }), /0 through 191/);
});

test('legacy char-vocab size is derived while its JSON shape stays strict', async () => {
  const truncated = ['<pad>', '<bos>', '<eos>', '<unk>', 'A'];
  const resources = new Map([
    [PACKAGE_URL, manifest()],
    ['https://example.test/tiny-split/vocab.json', { itos: truncated }],
  ]);
  const session = await TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async () => { throw new Error('must not load'); },
  });
  assert.equal(session.vocab.itos.length, 5);
  await session.close();

  resources.set('https://example.test/tiny-split/vocab.json', {
    itos: truncated,
    private_metadata: 'unexpected',
  });
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { createModel() {} },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async () => { throw new Error('must not load'); },
  }), /vocab\.json must contain exactly itos/);
});

test('split session keeps every manifest asset inside the fixed ASCII package path', async () => {
  for (const path of [
    '%2e%2e/vocab.json',
    'http:evil',
    'data:text/plain,evil',
    'a:b',
    'unicode-\u2603.json',
  ]) {
    const invalid = manifest();
    invalid.assets.config.path = path;
    await assert.rejects(TinyReceiptSplitSession.load({
      runtime: { createModel() {} },
      packageUrl: PACKAGE_URL,
      fetch: fakeFetch(new Map([[PACKAGE_URL, invalid]])),
      graphLoader: async () => { throw new Error('must not load'); },
    }), /package-relative asset path/, path);
  }
});

test('split session requires package outputs to be declared graph outputs', async () => {
  const vocabulary = ['<pad>', '<bos>', '<eos>', '<unk>'];
  while (vocabulary.length < 760) vocabulary.push(`token-${vocabulary.length}`);
  const resources = new Map([
    [PACKAGE_URL, manifest()],
    ['https://example.test/tiny-split/vocab.json', { itos: vocabulary }],
  ]);
  const graph = encoderGraph();
  delete graph.outputNames;
  const session = await TinyReceiptSplitSession.load({
    runtime: {
      createModel() {
        throw new Error('model creation must not begin');
      },
    },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    graphLoader: async () => graph,
  });
  await assert.rejects(session.preload(), /encoder graph outputs must be exactly/);
  await session.close();
});
