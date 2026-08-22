import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

import {
  TinyReceiptByteFallbackBPEVocab,
  TinyReceiptSplitSession,
  TINY_RECEIPT_SPLIT_FAMILY_ORDER,
  TINY_RECEIPT_SPLIT_KV_PACKAGE_FORMAT,
} from '../TinyReceiptSplitSession.js';


const PACKAGE_URL = 'https://example.test/tiny-split/package_manifest.json';
const SHA = 'a'.repeat(64);
const BPE_HASH = '3d38e3d05ed255191d504c02c91048795c403aec544c952de0b94f3c8fa082b5';
const KV_CROSS_NAMES = [...Array(4).keys()].flatMap((layer) =>
  [`cross_k_${layer}`, `cross_v_${layer}`]);
const KV_PAST_NAMES = [...Array(4).keys()].flatMap((layer) =>
  [`past_k_${layer}`, `past_v_${layer}`]);
const KV_PRESENT_NAMES = [...Array(4).keys()].flatMap((layer) =>
  [`present_k_${layer}`, `present_v_${layer}`]);

function asset(path) {
  return { path, bytes: 1, sha256: SHA };
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

function kvManifest() {
  const encoderInputs = {
    image: 'input0',
    question_ids: 'input1',
    family_ids: 'input2',
    question_position_ids: 'input3',
  };
  const decoderInputs = {
    decoder_input_ids: 'input0',
    position_ids: 'input1',
    family_ids: 'input2',
    memory_padding_mask: 'input3',
    past_padding_mask: 'input4',
  };
  [...KV_CROSS_NAMES, ...KV_PAST_NAMES].forEach((name, index) => {
    decoderInputs[name] = `input${index + 5}`;
  });
  return {
    format: TINY_RECEIPT_SPLIT_KV_PACKAGE_FORMAT,
    assets: { config: asset('config.json'), vocab: asset('vocab.json') },
    tokenizer: {
      type: 'byte_fallback_bpe', version: 1, vocab_size: 1536,
      normalization: 'NFC', tokenizer_hash: BPE_HASH,
      itos_key: 'itos', merges_key: 'merges',
      token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
    },
    preprocessing: { layout: 'NCHW', shape: [1, 1, 320, 672], color: 'grayscale' },
    families: {
      auto_id: -1,
      ordered_names: [...TINY_RECEIPT_SPLIT_FAMILY_ORDER],
      name_to_id: Object.fromEntries(
        TINY_RECEIPT_SPLIT_FAMILY_ORDER.map((name, index) => [name, index]),
      ),
    },
    routing: {
      mode: 'runtime', family_inputs: { encoder: 'input2', decoder: 'input2' },
    },
    generation: {
      strategy: 'greedy-autoregressive-explicit-kv',
      maximum_target_length: 192, maximum_new_tokens: 191,
      bos_token_id: 1, eos_token_id: 2, pad_token_id: 0,
      logits_row: 'current_token', tie_policy: 'first-index',
    },
    shape_contract: {
      graph_shape_mode: 'bounded-explicit-kv-v2',
      dimensions: {
        B: { min: 1, max: 1 }, Q: { min: 1, max: 192 },
        M: { min: 211, max: 402 }, P: { min: 1, max: 191 },
        R: { min: 2, max: 192 },
      },
      fixed_geometry: {
        image: [1, 1, 320, 672], image_tokens: 210, feature_width: 320,
        attention_heads: 8, attention_head_width: 40, decoder_layers: 4,
        adapter_families: 8,
      },
      relations: {
        encoder_memory: {
          operator: 'Concat', axis: 1, fixed_image_tokens: 210,
          dynamic_question_dimension: 'Q', derived_memory_dimension: 'M',
        },
        present_cache: {
          operator: 'Concat', axis: 2, past_dimension: 'P',
          fixed_current_tokens: 1, derived_present_dimension: 'R',
        },
      },
      semantic_inputs: {
        question_position_ids: { shape: ['B', 'Q'], values: 'zero_based_contiguous' },
      },
    },
    cache_contract: {
      format: 'masked-zero-sentinel-v1', layers: 4, heads: 8, head_width: 40,
      past_dimension: 'P', present_dimension: 'R', initial_past_length: 1,
      sentinel_mask_value: 1, cache_dtype: 'float32',
    },
    graphs: {
      encoder: {
        graph: asset('encoder/graph.json'), weights: asset('encoder/model.safetensors'),
        export_report: asset('encoder/export_report.json'), inputs: encoderInputs,
        outputs: Object.fromEntries([
          'memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids',
          ...KV_CROSS_NAMES,
        ].map((name) => [name, name])),
      },
      decoder: {
        graph: asset('decoder/graph.json'), weights: asset('decoder/model.safetensors'),
        export_report: asset('decoder/export_report.json'), inputs: decoderInputs,
        outputs: Object.fromEntries([
          'logits', 'present_padding_mask', ...KV_PRESENT_NAMES,
        ].map((name) => [name, name])),
      },
    },
    mask_semantics: {
      memory_padding_mask: 'nonzero_means_blocked',
      past_padding_mask: 'nonzero_means_blocked',
    },
  };
}

function tensor(name, shape, dtype, isInput = false) {
  return { name, shape, dtype, kind: isInput ? 'input' : 'value' };
}

function logicalGraph(kind, tensors, outputs, dimensions, nodes = []) {
  const table = Object.fromEntries(tensors.map((value) => [value.name, value]));
  return {
    kind,
    format: 'volvox-graph/v1',
    dimensions,
    inputs: Object.fromEntries(
      tensors.filter((value) => value.kind === 'input').map((value) => [value.name, value]),
    ),
    tensors: table,
    nodes,
    outputs,
  };
}

function snapshot(graph) {
  return {
    graph,
    inputNames: Object.keys(graph.inputs),
    outputNames: [...graph.outputs],
  };
}

function kvEncoderGraph() {
  const tensors = [
    tensor('input0', ['B', 1, 320, 672], 'float32', true),
    tensor('input1', ['B', 'Q'], 'int32', true),
    tensor('input2', ['B'], 'int32', true),
    tensor('input3', ['B', 'Q'], 'int32', true),
    tensor('fixture_image_tokens', ['B', 210, 320], 'float32'),
    tensor('fixture_question_tokens', ['B', 'Q', 320], 'float32'),
    tensor('fixture_encoded_sequence', ['B', 'M', 320], 'float32'),
    tensor('memory', ['B', 'M', 320], 'float32'),
    tensor('memory_padding_mask', ['B', 'M'], 'int32'),
    tensor('router_logits', ['B', 8], 'float32'),
    tensor('selected_family_ids', ['B'], 'int32'),
    ...KV_CROSS_NAMES.map((name) =>
      tensor(name, ['B', 8, 'M', 40], 'float32')),
  ];
  return logicalGraph('encoder', tensors, [
    'memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids',
    ...KV_CROSS_NAMES,
  ], {
    B: { name: 'B', min: 1, max: 1, multiple_of: 1 },
    Q: { name: 'Q', min: 1, max: 192, multiple_of: 1 },
    M: { name: 'M', min: 211, max: 402, multiple_of: 1 },
  }, [{
    id: 'fixture-memory-concat', opType: 'Concat',
    inputs: { input0: 'fixture_image_tokens', input1: 'fixture_question_tokens' },
    outputs: { out: {
      tensor: 'fixture_encoded_sequence', shape: ['B', 'M', 320], dtype: 'float32',
    } },
    params: { axis: 1 },
  }]);
}

function kvDecoderGraph() {
  const tensors = [
    tensor('input0', ['B', 1], 'int32', true),
    tensor('input1', ['B'], 'int32', true),
    tensor('input2', ['B'], 'int32', true),
    tensor('input3', ['B', 'M'], 'int32', true),
    tensor('input4', ['B', 'P'], 'int32', true),
  ];
  KV_CROSS_NAMES.forEach((name, index) => {
    tensors.push(tensor(`input${index + 5}`, ['B', 8, 'M', 40], 'float32', true));
  });
  KV_PAST_NAMES.forEach((name, index) => {
    tensors.push(tensor(`input${index + 13}`, ['B', 8, 'P', 40], 'float32', true));
  });
  tensors.push(tensor('logits', ['B', 1, 1536], 'float32'));
  tensors.push(tensor('present_padding_mask', ['B', 'R'], 'int32'));
  const nodes = [];
  KV_PRESENT_NAMES.forEach((name, index) => {
    const current = `fixture_current_${index}`;
    tensors.push(tensor(current, ['B', 8, 1, 40], 'float32'));
    tensors.push(tensor(name, ['B', 8, 'R', 40], 'float32'));
    nodes.push({
      id: `fixture-present-concat-${index}`, opType: 'Concat',
      inputs: { input0: `input${index + 13}`, input1: current },
      outputs: { out: {
        tensor: name, shape: ['B', 8, 'R', 40], dtype: 'float32',
      } },
      params: { axis: 2 },
    });
  });
  return logicalGraph('decoder', tensors, [
    'logits', 'present_padding_mask', ...KV_PRESENT_NAMES,
  ], {
    B: { name: 'B', min: 1, max: 1, multiple_of: 1 },
    M: { name: 'M', min: 211, max: 402, multiple_of: 1 },
    P: { name: 'P', min: 1, max: 191, multiple_of: 1 },
    R: { name: 'R', min: 2, max: 192, multiple_of: 1 },
  }, nodes);
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
    Object.entries(inputs).map(([name, value]) => [name, {
      data: fixtureInputData(value).slice(),
      shape: [...value.shape],
    }]),
  );
}

function fixtureInputData(value) {
  if (ArrayBuffer.isView(value?.data) && !(value.data instanceof DataView)) {
    return value.data;
  }
  if (typeof value?.data?._fixtureData === 'function') return value.data._fixtureData();
  throw new Error('fixture input has no host or live synthetic-device data');
}

function executionResult(outputs, {
  device = false,
  id = 'host-result',
  lifecycle = [],
  readCounts = new Map(),
} = {}) {
  let closed = false;
  const tensors = Object.fromEntries(Object.entries(outputs).map(([name, value]) => {
    if (!device) {
      return [name, {
        shape: value.shape,
        read: async () => value.data,
      }];
    }
    const dtype = value.data instanceof Float32Array ? 'float32' : 'int32';
    return [name, {
      name,
      shape: value.shape,
      dtype,
      location: 'device',
      async read() {
        assert.equal(closed, false, `${id}/${name} read after producer close`);
        readCounts.set(name, (readCounts.get(name) || 0) + 1);
        return value.data;
      },
      _fixtureData() {
        assert.equal(closed, false, `${id}/${name} consumed after producer close`);
        return value.data;
      },
    }];
  }));
  return {
    backend: device ? 'webgpu' : 'wasm',
    output(name) {
      assert.ok(tensors[name]);
      return tensors[name];
    },
    report: null,
    async close() {
      if (closed) return;
      closed = true;
      lifecycle.push(`close:${id}`);
      for (const value of Object.values(outputs)) value.data.fill(-99);
    },
  };
}

function output(data, shape) {
  return { data, shape };
}

async function syntheticKVSession({
  manifestValue = kvManifest(),
  mutateEncoderGraph = () => {},
  mutateDecoderGraph = () => {},
  decoderTokens = [300, 2],
  closeFailures = {},
  failCreateContextKind = null,
  deviceResults = false,
} = {}) {
  const resources = new Map([
    [PACKAGE_URL, manifestValue],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  const calls = [];
  const contextOptions = [];
  const closed = { compiled: 0, contexts: 0 };
  const lifecycle = [];
  const deviceReadCounts = { encoder: new Map(), decoder: new Map() };
  const closeContext = (kind) => {
    closed.contexts++;
    if (closeFailures.contexts?.includes(kind)) {
      throw new Error(`synthetic ${kind} context close failure`);
    }
  };
  const closeCompiled = (kind) => {
    closed.compiled++;
    if (closeFailures.compiled?.includes(kind)) {
      throw new Error(`synthetic ${kind} compiled close failure`);
    }
  };
  const control = {
    corruptCachePrefix: false,
    corruptCacheAppend: false,
    corruptCrossCache: false,
    corruptMemory: false,
    corruptRouterLogits: false,
    corruptMaskPrefix: false,
  };
  let expectedPast = null;
  const runtime = {
    async compile(loadedSnapshot) {
      const graph = loadedSnapshot.graph;
      return {
        backend: 'wasm',
        async createContext(options) {
          contextOptions.push({ kind: graph.kind, options });
          if (failCreateContextKind === graph.kind) {
            throw new Error(`synthetic ${graph.kind} create failure`);
          }
          if (graph.kind === 'encoder') {
            return {
              async execute(inputs) {
                calls.push({ kind: 'encoder', inputs: copyInputs(inputs) });
                const questionLength = inputs.input1.shape[1];
                const memoryLength = questionLength + 210;
                const outputs = {
                  memory: output(new Float32Array(memoryLength * 320),
                    [1, memoryLength, 320]),
                  memory_padding_mask: output(new Int32Array(memoryLength),
                    [1, memoryLength]),
                  router_logits: output(Float32Array.of(0, 1, 5, 1, 0, 0, 0, 0), [1, 8]),
                  selected_family_ids: output(Int32Array.of(2), [1]),
                };
                if (control.corruptMemory) outputs.memory.data[0] = Number.NaN;
                if (control.corruptRouterLogits) {
                  outputs.router_logits.data[0] = Number.POSITIVE_INFINITY;
                }
                KV_CROSS_NAMES.forEach((name, index) => {
                  const values = new Float32Array(8 * memoryLength * 40);
                  values.fill(index + 1);
                  if (control.corruptCrossCache && index === 0) values[0] = Number.NaN;
                  outputs[name] = output(values, [1, 8, memoryLength, 40]);
                });
                return executionResult(outputs, {
                  device: deviceResults,
                  id: 'encoder',
                  lifecycle,
                  readCounts: deviceReadCounts.encoder,
                });
              },
              async close() { closeContext(graph.kind); },
            };
          }
          assert.equal(options?.decode, undefined,
            'explicit KV decoder must use ordinary context execution');
          return {
            async execute(inputs) {
              const copied = copyInputs(inputs);
              lifecycle.push(`execute:decoder:${inputs.input1.data[0]}`);
              calls.push({ kind: 'decoder', inputs: copied });
              const position = inputs.input1.data[0];
              const pastLength = inputs.input4.shape[1];
              assert.deepEqual(inputs.input0.shape, [1, 1]);
              assert.deepEqual(inputs.input1.shape, [1]);
              assert.deepEqual(inputs.input4.shape, [1, pastLength]);
              assert.equal(inputs.input4.data[0], 1);
              assert.ok([...inputs.input4.data.slice(1)].every((value) => value === 0));
              if (position === 0) {
                expectedPast = null;
                assert.equal(inputs.input0.data[0], 1);
                assert.equal(pastLength, 1);
                for (let index = 0; index < KV_PAST_NAMES.length; index++) {
                  assert.ok(fixtureInputData(inputs[`input${index + 13}`])
                    .every((value) => value === 0));
                }
              } else {
                assert.equal(inputs.input0.data[0], decoderTokens[position - 1]);
                assert.equal(position, 1);
                assert.equal(pastLength, 2);
                for (let index = 0; index < KV_PAST_NAMES.length; index++) {
                  assert.deepEqual(
                    [...fixtureInputData(inputs[`input${index + 13}`])],
                    [...expectedPast[index]],
                  );
                }
              }
              for (let index = 0; index < KV_CROSS_NAMES.length; index++) {
                assert.ok(fixtureInputData(inputs[`input${index + 5}`]).every(
                  (value) => value === index + 1));
              }
              const presentLength = pastLength + 1;
              const outputs = {
                logits: output(new Float32Array(1536), [1, 1, 1536]),
                present_padding_mask: output(
                  Int32Array.from({ length: presentLength }, (_, index) =>
                    index === 0 || (index === pastLength && inputs.input0.data[0] === 0) ? 1 : 0),
                  [1, presentLength],
                ),
              };
              if (control.corruptMaskPrefix) outputs.present_padding_mask.data[0] = 0;
              outputs.logits.data[decoderTokens[position]] = 10;
              const nextExpected = [];
              KV_PRESENT_NAMES.forEach((name, index) => {
                const past = fixtureInputData(inputs[`input${index + 13}`]);
                const values = new Float32Array(8 * presentLength * 40);
                for (let head = 0; head < 8; head++) {
                  values.set(
                    past.subarray(head * pastLength * 40, (head + 1) * pastLength * 40),
                    head * presentLength * 40,
                  );
                  values.fill(
                    100 + index * 10 + position,
                    (head * presentLength + pastLength) * 40,
                    (head * presentLength + pastLength + 1) * 40,
                  );
                }
                if (control.corruptCachePrefix && index === 0) values[0] = 7;
                if (control.corruptCacheAppend && index === 0) {
                  values[pastLength * 40] = Number.POSITIVE_INFINITY;
                }
                outputs[name] = output(values, [1, 8, presentLength, 40]);
                nextExpected[index] = values.slice();
              });
              expectedPast = nextExpected;
              return executionResult(outputs, {
                device: deviceResults,
                id: `decoder-${position}`,
                lifecycle,
                readCounts: deviceReadCounts.decoder,
              });
            },
            async close() { closeContext(graph.kind); },
          };
        },
        async close() { closeCompiled(graph.kind); },
      };
    },
  };
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    snapshotLoader: async ({ kind }) => {
      const graph = kind === 'encoder' ? kvEncoderGraph() : kvDecoderGraph();
      (kind === 'encoder' ? mutateEncoderGraph : mutateDecoderGraph)(graph);
      return snapshot(graph);
    },
  });
  return {
    session,
    calls,
    contextOptions,
    control,
    closed,
    lifecycle,
    deviceReadCounts,
  };
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

test('explicit-KV package tokenizer requires the exact published eight-field contract', async () => {
  const value = kvManifest();
  delete value.tokenizer.merges_key;
  const resources = new Map([
    [PACKAGE_URL, value],
    ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
  ]);
  await assert.rejects(TinyReceiptSplitSession.load({
    runtime: { compile() { throw new Error('must not compile'); } },
    packageUrl: PACKAGE_URL,
    fetch: fakeFetch(resources),
    snapshotLoader: async () => { throw new Error('must not load'); },
  }), /tokenizer must contain exactly/);
});

test('explicit KV package carries a blocked positive sentinel through one-token cache steps', async () => {
  const { session, calls, contextOptions } = await syntheticKVSession();
  const answer = await session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    family: 'auto',
    maxNewTokens: 2,
    preprocessed: true,
  });
  assert.deepEqual(answer.tokenIds, [300, 2]);
  assert.equal(answer.familyId, 2);
  assert.equal(answer.execution, 'explicit-kv-cache');
  assert.equal(answer.decodeMode, 'explicit-kv-cache');
  assert.equal(answer.decoderSeedExecutions, 1);
  assert.equal(answer.decoderOrdinaryExecutions, 2);
  assert.equal(answer.decoderCacheStepExecutions, 1);
  assert.deepEqual(answer.cacheShape, {
    initialPastLength: 1, finalPastLength: 3, sentinelSlots: 1,
  });
  assert.deepEqual(answer.decodeReports.map(({ position, pastLength, presentLength }) =>
    ({ position, pastLength, presentLength })), [
    { position: 0, pastLength: 1, presentLength: 2 },
    { position: 1, pastLength: 2, presentLength: 3 },
  ]);
  const encoderCall = calls.find(({ kind }) => kind === 'encoder');
  assert.equal(encoderCall.inputs.input2.data[0], -1);
  assert.deepEqual([...encoderCall.inputs.input3.data], [0, 1]);
  const decoderCalls = calls.filter(({ kind }) => kind === 'decoder');
  assert.equal(decoderCalls.length, 2);
  assert.equal(decoderCalls[0].inputs.input2.data[0], 2);
  assert.deepEqual(decoderCalls.map(({ inputs }) => inputs.input4.shape), [[1, 1], [1, 2]]);
  assert.deepEqual(decoderCalls.map(({ inputs }) => inputs.input1.data[0]), [0, 1]);
  assert.deepEqual(decoderCalls.map(({ inputs }) => inputs.input0.data[0]), [1, 300]);
  assert.equal(contextOptions.find(({ kind }) => kind === 'decoder').options?.decode, undefined);

  const beforeZero = calls.length;
  const zero = await session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 0, preprocessed: true,
  });
  assert.deepEqual(zero.tokenIds, []);
  assert.equal(calls.slice(beforeZero).filter(({ kind }) => kind === 'decoder').length, 0);
  await session.close();

  const padFixture = await syntheticKVSession({ decoderTokens: [0, 2] });
  const padAnswer = await padFixture.session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 2,
    preprocessed: true,
  });
  assert.deepEqual(padAnswer.tokenIds, [0, 2]);
  const padCalls = padFixture.calls.filter(({ kind }) => kind === 'decoder');
  assert.deepEqual([...padCalls[1].inputs.input4.data], [1, 0]);
  await padFixture.session.close();
});

test('WebGPU explicit KV handoff keeps caches on device and preserves producer ownership', async () => {
  const fixture = await syntheticKVSession({ deviceResults: true });
  const answer = await fixture.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    family: 'auto',
    maxNewTokens: 2,
    preprocessed: true,
    kvTransferMode: 'device-resident',
  });
  assert.deepEqual(answer.tokenIds, [300, 2]);
  assert.equal(answer.gpuResidentKv.enabled, true);
  assert.equal(answer.gpuResidentKv.mode, 'device-resident');
  assert.equal(answer.gpuResidentKv.encoderCrossCacheHandoffs, 16);
  assert.equal(answer.gpuResidentKv.decoderCacheHandoffs, 8);
  assert.equal(answer.gpuResidentKv.runtimeValidatedDeviceInputs, true);
  assert.equal(answer.gpuResidentKv.encoderResultRetainedThroughDecode, true);
  assert.equal(answer.gpuResidentKv.decoderResultRetainedUntilSuccessorExecution, true);
  assert.equal(answer.gpuResidentKv.encoderMemoryReadback, false);
  assert.equal(answer.gpuResidentKv.encoderMemoryReadbackValidated, false);
  assert.equal(answer.gpuResidentKv.encoderCrossCacheReadbackValidated, false);
  assert.equal(answer.gpuResidentKv.cachePrefixReadbackValidated, false);
  assert.equal(answer.gpuResidentKv.appendedCacheReadbackValidated, false);
  assert.equal(answer.gpuResidentKv.cacheReadbackFree, true);
  assert.equal(answer.synchronizedTiming.completion, 'required-small-output-readback');
  assert.equal(answer.synchronizedTiming.applicationValidationIncluded, false);
  assert.equal(answer.synchronizedTiming.cacheQualificationReadbackIncluded, false);
  assert.ok(Number.isFinite(answer.synchronizedTiming.encoderExecutionMs));
  assert.equal(answer.synchronizedTiming.decoderStepMs.length, 2);
  assert.ok(answer.synchronizedTiming.decoderStepMs.every(Number.isFinite));
  assert.equal(fixture.deviceReadCounts.encoder.get('memory') || 0, 0);
  for (const key of KV_CROSS_NAMES) {
    assert.equal(fixture.deviceReadCounts.encoder.get(key) || 0, 0);
  }
  assert.equal(fixture.deviceReadCounts.encoder.get('memory_padding_mask'), 1);
  assert.equal(fixture.deviceReadCounts.encoder.get('router_logits'), 1);
  assert.equal(fixture.deviceReadCounts.encoder.get('selected_family_ids'), 1);
  assert.equal(fixture.deviceReadCounts.decoder.get('logits'), 2);
  assert.equal(fixture.deviceReadCounts.decoder.get('present_padding_mask'), 2);
  for (const key of KV_PRESENT_NAMES) {
    assert.equal(fixture.deviceReadCounts.decoder.get(key) || 0, 0);
  }
  assert.ok(
    fixture.lifecycle.indexOf('execute:decoder:1') <
      fixture.lifecycle.indexOf('close:decoder-0'),
    'the successor execution must accept D2D inputs before its producer closes',
  );
  assert.ok(
    fixture.lifecycle.indexOf('close:decoder-1') < fixture.lifecycle.indexOf('close:encoder'),
    'encoder cross-cache owner must outlive all decoder executions',
  );
  await fixture.session.close();
});

test('WebGPU explicit KV qualification reads and verifies cache transitions only when opted in', async () => {
  const fixture = await syntheticKVSession({ deviceResults: true });
  const answer = await fixture.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 2,
    preprocessed: true,
    kvTransferMode: 'device-qualified',
  });
  assert.equal(answer.gpuResidentKv.mode, 'device-qualified');
  assert.equal(answer.gpuResidentKv.encoderMemoryReadback, true);
  assert.equal(answer.gpuResidentKv.encoderMemoryReadbackValidated, true);
  assert.equal(answer.gpuResidentKv.encoderCrossCacheReadbackValidated, true);
  assert.equal(answer.gpuResidentKv.cachePrefixReadbackValidated, true);
  assert.equal(answer.gpuResidentKv.appendedCacheReadbackValidated, true);
  assert.equal(answer.gpuResidentKv.cacheReadbackFree, false);
  assert.equal(fixture.deviceReadCounts.encoder.get('memory'), 1);
  for (const key of KV_CROSS_NAMES) {
    assert.equal(fixture.deviceReadCounts.encoder.get(key), 1);
  }
  for (const key of KV_PRESENT_NAMES) {
    assert.equal(fixture.deviceReadCounts.decoder.get(key), 2);
  }
  await fixture.session.close();

  const corrupt = await syntheticKVSession({ deviceResults: true });
  corrupt.control.corruptCacheAppend = true;
  await assert.rejects(corrupt.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 1,
    preprocessed: true,
    kvTransferMode: 'device-qualified',
  }), /appended a non-finite cache value/);
  corrupt.control.corruptCacheAppend = false;
  corrupt.control.corruptMemory = true;
  await assert.rejects(corrupt.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 1,
    preprocessed: true,
    kvTransferMode: 'device-qualified',
  }), /encoder memory contains a non-finite value/);
  await corrupt.session.close();
});

test('device KV transfer is guarded and the host path remains the default', async () => {
  const fixture = await syntheticKVSession();
  await assert.rejects(fixture.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 1,
    preprocessed: true,
    kvTransferMode: 'device-resident',
  }), /device-resident KV requires a WebGPU execution result/);
  await assert.rejects(fixture.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 1,
    preprocessed: true,
    kvTransferMode: 'typo',
  }), /kvTransferMode must be one of/);
  const recovered = await fixture.session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  });
  assert.equal(recovered.gpuResidentKv.mode, 'host-validated');
  await fixture.session.close();
});

test('explicit KV package rejects weakened sentinel, dimension, mask, and output contracts', async () => {
  const cases = [
    [
      (value) => { value.shape_contract.shape_system = 'volvox-bounded-shape/v1'; },
      /shape_contract must contain exactly/,
    ],
    [
      (value) => { value.cache_contract.initial_past_length = 0; },
      /qualified positive-shape sentinel ABI/,
    ],
    [
      (value) => { value.shape_contract.dimensions.P.min = 0; },
      /shape_contract\.dimensions\.P must be bounded to \[1,191\]/,
    ],
    [
      (value) => { value.shape_contract.dimensions.B.max = 9; },
      /shape_contract\.dimensions\.B must be bounded to \[1,N\]/,
    ],
    [
      (value) => { value.mask_semantics.past_padding_mask = 'unsupported'; },
      /memory\/past padding-mask semantics/,
    ],
    [
      (value) => {
        value.mask_semantics.past_padding_mask = 'zero_means_blocked';
        value.cache_contract.sentinel_mask_value = 0;
      },
      /memory\/past padding-mask semantics/,
    ],
    [
      (value) => {
        delete value.graphs.encoder.inputs.question_position_ids;
        delete value.shape_contract.semantic_inputs.question_position_ids;
      },
      /shape_contract\.semantic_inputs must contain exactly/,
    ],
    [
      (value) => { delete value.graphs.decoder.outputs.present_v_3; },
      /graphs\.decoder\.outputs must contain exactly/,
    ],
    [
      (value) => {
        value.graphs.decoder.inputs.past_v_3 = value.graphs.decoder.inputs.past_k_3;
      },
      /inputs must map every semantic input to a distinct graph tensor/,
    ],
    [
      (value) => { value.graphs.encoder.graph.path = '../encoder.json'; },
      /must be a package-relative asset path/,
    ],
  ];
  for (const [mutate, pattern] of cases) {
    const value = kvManifest();
    mutate(value);
    const resources = new Map([
      [PACKAGE_URL, value],
      ['https://example.test/tiny-split/vocab.json', bpeVocabulary()],
    ]);
    await assert.rejects(TinyReceiptSplitSession.load({
      runtime: { compile() { throw new Error('must not compile'); } },
      packageUrl: PACKAGE_URL,
      fetch: fakeFetch(resources),
      snapshotLoader: async () => { throw new Error('must not load'); },
    }), pattern);
  }
});

test('opt-in B2 manifest and both graphs are one checked package contract', async () => {
  const manifest = kvManifest();
  manifest.shape_contract.dimensions.B.max = 2;
  const widenBatch = (graph) => { graph.dimensions.B.max = 2; };
  const fixture = await syntheticKVSession({
    manifestValue: manifest,
    mutateEncoderGraph: widenBatch,
    mutateDecoderGraph: widenBatch,
  });
  await fixture.session.preload();
  const result = await fixture.session.generate({
    image: new Float32Array(320 * 672),
    prompt: 'a',
    maxNewTokens: 1,
    preprocessed: true,
  });
  assert.equal(result.tokenIds.length, 1);
  assert.deepEqual(fixture.calls.map(({ kind }) => kind), ['encoder', 'decoder']);
  assert.ok(fixture.calls.every(({ inputs }) =>
    Object.values(inputs).every(({ shape }) => shape[0] === 1)));
  await fixture.session.close();
});

test('opt-in B2 manifest rejects a B1 encoder or decoder graph', async () => {
  for (const role of ['encoder', 'decoder']) {
    const manifest = kvManifest();
    manifest.shape_contract.dimensions.B.max = 2;
    const fixture = await syntheticKVSession({
      manifestValue: manifest,
      mutateEncoderGraph(graph) {
        if (role === 'decoder') graph.dimensions.B.max = 2;
      },
      mutateDecoderGraph(graph) {
        if (role === 'encoder') graph.dimensions.B.max = 2;
      },
    });
    await assert.rejects(
      fixture.session.preload(),
      new RegExp(`${role} KV graph dimension B has the wrong bounds`),
    );
    await fixture.session.close();
  }
});

test('explicit KV graph requires eight R=P+1 witnesses and detects sentinel corruption', async () => {
  const malformed = await syntheticKVSession({
    mutateDecoderGraph(graph) {
      graph.nodes.at(-1).params.axis = 1;
    },
  });
  await assert.rejects(malformed.session.preload(), /R=P\+1 Concat witness/);
  await malformed.session.close();

  const { session, control } = await syntheticKVSession();
  control.corruptCachePrefix = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /corrupted its persistent past-cache prefix/);
  control.corruptCachePrefix = false;
  control.corruptMaskPrefix = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /present_padding_mask corrupted its past prefix or current token/);
  control.corruptMaskPrefix = false;
  control.corruptCacheAppend = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /appended a non-finite cache value/);
  control.corruptCacheAppend = false;
  control.corruptCrossCache = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /contains a non-finite cache value/);
  control.corruptCrossCache = false;
  control.corruptMemory = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /encoder memory contains a non-finite value/);
  control.corruptMemory = false;
  control.corruptRouterLogits = true;
  await assert.rejects(session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  }), /encoder router_logits contains a non-finite value/);
  control.corruptRouterLogits = false;
  const recovered = await session.generate({
    image: new Float32Array(320 * 672), prompt: 'a', maxNewTokens: 1,
    preprocessed: true,
  });
  assert.deepEqual(recovered.tokenIds, [300]);
  assert.equal(recovered.cacheShape.finalPastLength, 2);
  await session.close();
});

test('session cleanup drains every explicit-KV owner and preserves setup failures', async () => {
  const closing = await syntheticKVSession({
    closeFailures: {
      contexts: ['encoder'],
      compiled: ['encoder'],
    },
  });
  await closing.session.preload();
  const closePromise = closing.session.close();
  await assert.rejects(closePromise, (error) => {
    assert.ok(error instanceof AggregateError);
    assert.deepEqual(error.errors.map(({ message }) => message), [
      'synthetic encoder context close failure',
      'synthetic encoder compiled close failure',
    ]);
    return true;
  });
  assert.deepEqual(closing.closed, { compiled: 2, contexts: 2 });
  assert.equal(closing.session._records.size, 0);
  assert.equal(closing.session.close(), closePromise);

  const setup = await syntheticKVSession({
    failCreateContextKind: 'decoder',
    closeFailures: { compiled: ['decoder'] },
  });
  await assert.rejects(setup.session.preload(), (error) => {
    assert.ok(error instanceof AggregateError);
    assert.deepEqual(error.errors.map(({ message }) => message), [
      'synthetic decoder create failure',
      'synthetic decoder compiled close failure',
    ]);
    return true;
  });
  await setup.session.close();
  assert.deepEqual(setup.closed, { compiled: 2, contexts: 1 });
});
