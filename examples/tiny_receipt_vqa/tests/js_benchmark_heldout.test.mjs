import assert from 'node:assert/strict';
import test from 'node:test';

import {
  compareGeneratedText,
  createBenchmarkVocabulary,
  encodeQuestion,
  extractStructuredAnswer,
  nextDecoderToken,
  requireSameVocabulary,
  resolveFamilySelection,
  resolveGenerationSettings,
  resolveDecoderInputs,
  resolveDecoderOutput,
} from '../tools/benchmark_heldout.mjs';
import { TinyReceiptCharVocab } from '../TinyReceiptW8A8Session.js';

function vocabulary() {
  return new TinyReceiptCharVocab(
    ['<pad>', '<bos>', '<eos>', '<unk>', 'é', ' ', 'x'],
    { pad: 0, bos: 1, eos: 2, unk: 3 },
  );
}

function currentManifest() {
  return {
    format: 'volvoxai-tiny-receipt-vqa-split-onnx-package-v1',
    generation: {
      decoder_output: 'token_ids',
      token_ids_row: 'prefix_length_minus_one',
      tie_policy: 'first-index',
      decoder_input_length: 192,
    },
    graphs: { decoder: { outputs: { token_ids: 'token_ids' } } },
  };
}

function currentDecoder() {
  return {
    format: 'volvox-graph/v1',
    inputs: {
      input0: { source_name: 'decoder_input_ids', shape: [1, 192], dtype: 'int32' },
      input1: { source_name: 'memory', shape: [1, 402, 320], dtype: 'float32' },
      input2: { source_name: 'memory_padding_mask', shape: [1, 402], dtype: 'int32' },
      v4_keep: { shape: [1, 192], dtype: 'int32' },
    },
    outputs: ['token_ids'],
    nodes: [{
      id: 'select-token',
      opType: 'QArgMax',
      inputs: { input: 'logits_byte' },
      outputs: { out: 'token_ids' },
      outputs_shape: { out: [1, 192] },
      outputs_dtype: { out: 'int32' },
      params: { axis: -1 },
    }],
  };
}

function logitsManifest() {
  return {
    format: 'volvoxai-tiny-receipt-vqa-split-onnx-package-v1',
    generation: {
      logits_row: 'prefix_length_minus_one',
      tie_policy: 'first-index',
      decoder_input_length: 192,
    },
    graphs: { decoder: { outputs: { logits: 'logits' } } },
  };
}

function logitsDecoder(vocabularySize = 1536, { hoistedKeep = false } = {}) {
  const result = {
    format: 'volvox-graph/v1',
    inputs: {
      input0: { source_name: 'decoder_input_ids', shape: [1, 192], dtype: 'int32' },
      input1: { source_name: 'memory', shape: [1, 402, 320], dtype: 'float32' },
      input2: { source_name: 'memory_padding_mask', shape: [1, 402], dtype: 'int32' },
      input3: { source_name: 'family_ids', shape: [1], dtype: 'int32' },
    },
    outputs: ['logits'],
    nodes: [{
      id: 'decoder-head',
      opType: 'MatMul',
      inputs: { a: 'hidden', b: 'head' },
      outputs: { out: 'logits' },
      outputs_shape: { out: [1, 192, vocabularySize] },
      outputs_dtype: { out: 'float32' },
      params: {},
    }],
  };
  if (hoistedKeep) {
    result.inputs.v4_keep = { shape: [1, 192], dtype: 'int32' };
  }
  return result;
}

test('encodeQuestion shares deployed NFC, whitespace, trim, padding, and EOS semantics', () => {
  const ids = encodeQuestion(' \t e\u0301\n  x  ', vocabulary());
  assert.deepEqual([...ids.slice(0, 5)], [4, 5, 6, 2, 0]);
});

test('requireSameVocabulary proves token-ID semantics instead of trusting a build directory', () => {
  const reference = { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'x'] };
  assert.equal(requireSameVocabulary(reference, { itos: [...reference.itos] }), reference.itos);
  assert.throws(
    () => requireSameVocabulary(reference, { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'y'] }),
    /identical token-ID semantics/,
  );
  assert.throws(() => requireSameVocabulary({}, reference), /reference vocabulary/);

  const bpe = {
    type: 'byte_fallback_bpe', version: 1, vocab_size: 5, normalization: 'NFC',
    tokenizer_hash: 'a'.repeat(64), itos: reference.itos,
  };
  assert.equal(requireSameVocabulary(bpe, structuredClone(bpe)), bpe.itos);
  const changedHash = structuredClone(bpe);
  changedHash.tokenizer_hash = 'b'.repeat(64);
  assert.throws(() => requireSameVocabulary(bpe, changedHash), /identical token-ID semantics/);
});

test('benchmark char vocabulary remains dynamically sized', async () => {
  const value = { itos: ['<pad>', '<bos>', '<eos>', '<unk>', 'x'] };
  const vocab = await createBenchmarkVocabulary(value, {
    type: 'char-vocab', version: 1, itos_key: 'itos',
    token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
  });
  assert.equal(vocab.itos.length, 5);
  assert.deepEqual([...encodeQuestion('x', vocab).slice(0, 3)], [4, 2, 0]);
});

test('extractStructuredAnswer prefers answer and tolerates a missing close tag', () => {
  assert.equal(
    extractStructuredAnswer('<value>wrong</value><answer>101</answer>'),
    '101',
  );
  assert.equal(extractStructuredAnswer('<value>wrong</value><answer>101'), '101');
});

test('extractStructuredAnswer never mistakes rationale value or raw text for the answer', () => {
  assert.equal(extractStructuredAnswer('<field>addr</field><value>775'), null);
  assert.equal(extractStructuredAnswer('  unstructured answer  '), null);
});

test('accuracy generation is complete unless a short run is explicitly latency-only', () => {
  assert.deepEqual(resolveGenerationSettings({}), { maxTokens: 191, latencyOnly: false });
  assert.throws(
    () => resolveGenerationSettings({ tokens: '32' }),
    /can truncate the <answer> field/,
  );
  assert.deepEqual(resolveGenerationSettings({ tokens: '32', 'latency-only': true }), {
    maxTokens: 32,
    latencyOnly: true,
  });
  assert.throws(() => resolveGenerationSettings({ tokens: '192' }), /integer in \[1, 191\]/);
});

test('family qualification is explicit, ordered, and fail-closed', () => {
  assert.deepEqual(resolveFamilySelection('phone,address'), [0, 1]);
  assert.deepEqual(resolveFamilySelection('address'), [1]);
  assert.throws(() => resolveFamilySelection('phone,phone'), /distinct families/);
  assert.throws(() => resolveFamilySelection('future'), /unknown family/);
});

test('answer agreement does not claim full structured-text agreement', () => {
  const comparison = compareGeneratedText(
    '<value>Mikpo St. 101</value><answer>101</answer>',
    '<value>Mokpo St. 101</value><answer>101</answer>',
  );
  assert.deepEqual(comparison, {
    splitAnswer: '101',
    legacyAnswer: '101',
    answerAgreement: true,
    structuredTextAgreement: false,
  });
});

test('identical text agrees at both levels and different answers agree at neither', () => {
  assert.deepEqual(compareGeneratedText('<answer>7</answer>', '<answer>7</answer>'), {
    splitAnswer: '7',
    legacyAnswer: '7',
    answerAgreement: true,
    structuredTextAgreement: true,
  });
  assert.deepEqual(compareGeneratedText('<answer>7</answer>', '<answer>5</answer>'), {
    splitAnswer: '7',
    legacyAnswer: '5',
    answerAgreement: false,
    structuredTextAgreement: false,
  });
  assert.deepEqual(compareGeneratedText('<value>7</value>', '<value>7</value>'), {
    splitAnswer: null,
    legacyAnswer: null,
    answerAgreement: false,
    structuredTextAgreement: true,
  });
});

test('resolveDecoderOutput accepts the optimized device token-ID ABI', () => {
  assert.deepEqual(resolveDecoderOutput(currentDecoder(), currentManifest()), {
    kind: 'token_ids',
    name: 'token_ids',
    length: 192,
  });
  const extraOutput = currentDecoder();
  extraOutput.outputs = ['token_ids', 'debug'];
  assert.throws(
    () => resolveDecoderOutput(extraOutput, currentManifest()),
    /only 'token_ids'/,
  );
  const legacyManifest = currentManifest();
  legacyManifest.generation.decoder_output = 'logits';
  assert.throws(
    () => resolveDecoderOutput(currentDecoder(), legacyManifest),
    /supported decoder ABI/,
  );
});

test('resolveDecoderOutput accepts raw BPE1536 logits and proves their width', () => {
  assert.deepEqual(resolveDecoderOutput(logitsDecoder(), logitsManifest(), 1536), {
    kind: 'logits',
    name: 'logits',
    length: 192,
    vocabularySize: 1536,
  });
  assert.throws(
    () => resolveDecoderOutput(logitsDecoder(760), logitsManifest(), 1536),
    /F32 \[1,192,1536\]/,
  );
});

test('resolveDecoderOutput proves a unique canonical I32 [1,192] QArgMax producer', () => {
  for (const mutate of [
    (graph) => { graph.nodes[0].opType = 'ArgMax'; },
    (graph) => { graph.nodes[0].outputs_dtype.out = 'float32'; },
    (graph) => { graph.nodes[0].outputs_shape.out = [1, 191]; },
    (graph) => { graph.nodes[0].params.axis = 1; },
  ]) {
    const graph = currentDecoder();
    mutate(graph);
    assert.throws(
      () => resolveDecoderOutput(graph, currentManifest()),
      /canonical QArgMax/,
    );
  }
  const duplicate = currentDecoder();
  duplicate.nodes.push({ ...duplicate.nodes[0], id: 'duplicate' });
  assert.throws(
    () => resolveDecoderOutput(duplicate, currentManifest()),
    /exactly one producer/,
  );
});

test('resolveDecoderInputs accepts only typed semantic inputs and exact v4_keep', () => {
  assert.deepEqual(resolveDecoderInputs(currentDecoder()), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    v4_keep: 'v4_keep',
  });

  const withFamily = currentDecoder();
  withFamily.inputs.input3 = {
    source_name: 'family_ids', shape: [1], dtype: 'int32',
  };
  assert.deepEqual(resolveDecoderInputs(withFamily), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    family_ids: 'input3',
    v4_keep: 'v4_keep',
  });

  assert.deepEqual(resolveDecoderInputs(logitsDecoder(), 'logits'), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    family_ids: 'input3',
  });
  assert.deepEqual(resolveDecoderInputs(logitsDecoder(1536, { hoistedKeep: true }), 'logits'), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    family_ids: 'input3',
    v4_keep: 'v4_keep',
  });

  const unknown = currentDecoder();
  unknown.inputs.future_mask = { shape: [1, 192], dtype: 'int32' };
  assert.throws(() => resolveDecoderInputs(unknown), /unknown decoder input/);

  const aliasedKeep = currentDecoder();
  delete aliasedKeep.inputs.v4_keep;
  aliasedKeep.inputs['@runtime/158:QBatchMatMul.keep'] = {
    source_name: 'v4_keep', shape: [1, 192], dtype: 'int32',
  };
  assert.deepEqual(resolveDecoderInputs(aliasedKeep), {
    decoder_input_ids: 'input0',
    memory: 'input1',
    memory_padding_mask: 'input2',
    v4_keep: '@runtime/158:QBatchMatMul.keep',
  });
});

test('nextDecoderToken reads I32 decisions without host argmax', () => {
  const contract = { kind: 'token_ids', name: 'token_ids', length: 192 };
  const values = new Int32Array(192);
  values[1] = 7;
  assert.equal(nextDecoderToken(contract, values, 1), 7);
  assert.throws(() => nextDecoderToken(contract, Int32Array.of(4), -1), /non-negative/);
  assert.throws(() => nextDecoderToken(contract, Int32Array.of(4), 0), /I32 \[1,192\]/);
});

test('nextDecoderToken applies first-index argmax to raw logits', () => {
  const contract = {
    kind: 'logits', name: 'logits', length: 192, vocabularySize: 3,
  };
  const values = new Float32Array(192 * 3);
  values.set([1, 5, 5], 3);
  assert.equal(nextDecoderToken(contract, values, 1), 1);
  values[3] = Number.NaN;
  assert.throws(() => nextDecoderToken(contract, values, 1), /non-finite/);
});
