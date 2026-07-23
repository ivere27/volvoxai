import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  CALIBRATION_FORMAT,
  assertCalibrationRoutingRecords,
  assertCompleteCalibrationRanges,
  bindDecoderCalibrationInputs,
  bindEncoderCalibrationInputs,
  buildCalibrationProvenance,
  buildCalibrationReport,
  buildTeacherForcedPrefix,
  calibrationGraphAbi,
  calibrationPackageContract,
  calibrationRouteExecutions,
  calibrationRoutingModeForGraph,
  cleanCalibrationText,
  encodeCalibrationQuestion,
  encodeCalibrationTarget,
  fileIdentity,
  normalizeCalibrationManifest,
  normalizeCalibrationRouteSweep,
  normalizeCalibrationRoutingMode,
  observeFiniteRange,
  observeInputs,
  packageIdentity,
  proveAdditiveAttentionMaskExclusions,
  runtimeF32CandidateTensors,
  selectCalibrationRecords,
  sha256Hex,
  teacherForcedPrefixLengths,
  validateSelectedCalibrationFamily,
  widenImmutableEmbeddingRanges,
} from '../tools/calibrate_split_activations.mjs';

const FAMILY_NAMES = [
  'phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other',
];

async function calibrationContract({ questionLength = 8, decoderLength = 8 } = {}) {
  const itos = ['<pad>', '<bos>', '<eos>', '<unk>', 'a', 'b', ' ', '<', '>', '/'];
  return calibrationPackageContract({
    tokenizer: {
      type: 'char-vocab',
      version: 1,
      itos_key: 'itos',
      token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
    },
    families: {
      auto_id: -1,
      ordered_names: FAMILY_NAMES,
      name_to_id: Object.fromEntries(FAMILY_NAMES.map((name, id) => [name, id])),
    },
    generation: { decoder_input_length: decoderLength },
  }, { itos }, { max_q_len: questionLength });
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
}

function bpeCalibrationDocuments({ questionLength = 8, decoderLength = 8 } = {}) {
  const structural = [
    '<field>', '</field>', '<value>', '</value>', '<op>', '</op>', '<answer>', '</answer>',
  ];
  const digits = [...Array(10).keys()].map(String);
  const bytes = [...Array(256).keys()]
    .map((value) => `<0x${value.toString(16).toUpperCase().padStart(2, '0')}>`);
  const itos = [
    '<pad>', '<bos>', '<eos>', '<unk>', ...structural, ...digits, ...bytes,
    'a', 'b', ' ', 'é', 'ab',
  ];
  const unused = [];
  while (itos.length < 1536) {
    const token = `<unused_${String(unused.length).padStart(4, '0')}>`;
    unused.push(token);
    itos.push(token);
  }
  const vocabulary = {
    type: 'byte_fallback_bpe',
    version: 1,
    vocab_size: 1536,
    itos,
    merges: [['a', 'b']],
    normalization: 'NFC',
    atomic_tokens: [...structural, ...digits],
    byte_tokens: bytes,
    unused_tokens: unused,
    special_tokens: { pad: '<pad>', bos: '<bos>', eos: '<eos>', unk: '<unk>' },
  };
  vocabulary.tokenizer_hash = createHash('sha256')
    .update(canonicalJson(vocabulary))
    .digest('hex');
  const packageManifest = {
    tokenizer: {
      type: 'byte_fallback_bpe',
      version: 1,
      vocab_size: 1536,
      normalization: 'NFC',
      tokenizer_hash: vocabulary.tokenizer_hash,
      itos_key: 'itos',
      merges_key: 'merges',
      token_ids: { pad: 0, bos: 1, eos: 2, unk: 3 },
    },
    families: {
      auto_id: -1,
      ordered_names: FAMILY_NAMES,
      name_to_id: Object.fromEntries(FAMILY_NAMES.map((name, id) => [name, id])),
    },
    generation: { decoder_input_length: decoderLength },
  };
  return { packageManifest, vocabulary, config: { max_q_len: questionLength } };
}

function record(id, family, overrides = {}) {
  return {
    id,
    image: `images/${id}.jpg`,
    question: 'a?',
    target: '<a>b</a>',
    family,
    ...overrides,
  };
}

test('TinyReceipt calibration stratifies real record families without relabeling', async () => {
  const contract = await calibrationContract();
  const normalized = normalizeCalibrationManifest({
    format: CALIBRATION_FORMAT,
    required_families: ['address', 0],
    records: [
      record('phone-0', 'phone'),
      record('phone-1', 'phone'),
      record('address-0', 'address'),
      record('address-1', 1),
      record('store-0', 'store'),
    ],
  }, contract);
  const selected = selectCalibrationRecords(
    normalized.records, 4, contract.familyNames.length, normalized.requiredFamilyIds,
  );

  assert.deepEqual(
    selected.map((sample) => [sample.id, sample.expectedFamilyId]),
    [['address-0', 1], ['phone-0', 0], ['store-0', 2], ['address-1', 1]],
  );
  assert.ok(selected.every(Object.isFrozen));
});

test('TinyReceipt all-public route sweep expands executions without relabeling records', async () => {
  const contract = await calibrationContract();
  const samples = [
    Object.freeze({ id: 'phone-0', expectedFamilyId: 0 }),
    Object.freeze({ id: 'store-0', expectedFamilyId: 2 }),
  ];
  const executions = calibrationRouteExecutions(samples, contract, 'all-public');

  assert.equal(executions.length, samples.length * FAMILY_NAMES.length);
  assert.deepEqual(
    executions.slice(0, FAMILY_NAMES.length).map((sample) => sample.routeFamilyId),
    FAMILY_NAMES.map((_, familyId) => familyId),
  );
  assert.ok(executions.slice(0, FAMILY_NAMES.length).every(
    (sample) => sample.id === 'phone-0' && sample.expectedFamilyId === 0,
  ));
  assert.deepEqual(samples, [
    { id: 'phone-0', expectedFamilyId: 0 },
    { id: 'store-0', expectedFamilyId: 2 },
  ]);
  assert.ok(executions.every(Object.isFrozen));
  const balancedSamples = Array.from({ length: FAMILY_NAMES.length * 2 }, (_, index) => (
    Object.freeze({ id: `sample-${index}`, expectedFamilyId: index % 3 })
  ));
  const balanced = calibrationRouteExecutions(
    balancedSamples, contract, 'balanced-public',
  );
  assert.equal(balanced.length, balancedSamples.length);
  assert.deepEqual(
    balanced.map((sample) => sample.routeFamilyId),
    [...FAMILY_NAMES.keys(), ...FAMILY_NAMES.keys()],
  );
  assert.deepEqual(
    balanced.map((sample) => sample.expectedFamilyId),
    balancedSamples.map((sample) => sample.expectedFamilyId),
  );
  assert.equal(normalizeCalibrationRouteSweep(), 'none');
  assert.throws(
    () => normalizeCalibrationRouteSweep('public-ish'),
    /--route-sweep must be 'none', 'balanced-public', or 'all-public'/,
  );
});

test('TinyReceipt calibration rejects guessed or incomplete record provenance', async () => {
  const contract = await calibrationContract();
  assert.throws(
    () => normalizeCalibrationManifest({
      format: CALIBRATION_FORMAT,
      records: [record('missing-target', 'phone', { target: undefined })],
    }, contract),
    /records\[0\]\.target must be a non-empty string/,
  );
  assert.throws(
    () => normalizeCalibrationManifest({
      format: CALIBRATION_FORMAT,
      records: [record('guessed-family', 'not-a-family')],
    }, contract),
    /canonical family name or ID/,
  );
  assert.throws(
    () => normalizeCalibrationManifest({
      format: CALIBRATION_FORMAT,
      required_families: ['address'],
      records: [record('only-phone', 'phone')],
    }, contract),
    /do not cover required families: address/,
  );
});

test('TinyReceipt calibration encodes normalized questions and bounded BOS/target/EOS', async () => {
  const contract = await calibrationContract({ questionLength: 6, decoderLength: 8 });
  assert.deepEqual(
    [...encodeCalibrationQuestion('  a\n  b  ', contract)],
    [4, 6, 5, 2, 0, 0],
  );
  assert.deepEqual(
    [...encodeCalibrationTarget('ab', contract)],
    [1, 4, 5, 2],
  );
  assert.deepEqual(
    [...encodeCalibrationTarget('abababab', contract)],
    [1, 4, 5, 4, 5, 4, 5, 2],
  );
});

test('calibration clean_text matches Python whitespace and NFC ordering', () => {
  assert.equal(
    cleanCalibrationText('\uFEFF  e\u0301\u001C\tb  \uFEFF'),
    '\uFEFF é b \uFEFF',
  );
  assert.equal(cleanCalibrationText(null), '');
});

test('TinyReceipt calibration uses the validated BPE1536 tokenizer for questions and targets',
  async () => {
    const documents = bpeCalibrationDocuments({ questionLength: 6, decoderLength: 8 });
    const contract = await calibrationPackageContract(
      documents.packageManifest,
      documents.vocabulary,
      documents.config,
    );

    assert.equal(contract.tokenizerKind, 'byte_fallback_bpe');
    assert.deepEqual(
      [...encodeCalibrationQuestion('  a\n  b  ', contract)],
      [278, 280, 279, 2, 0, 0],
    );
    assert.deepEqual(
      [...encodeCalibrationTarget('ab', contract)],
      [1, 282, 2],
    );

    const tampered = structuredClone(documents.vocabulary);
    tampered.itos[278] = 'changed';
    await assert.rejects(
      calibrationPackageContract(documents.packageManifest, tampered, documents.config),
      /tokenizer_hash does not match/,
    );
  });

test('TinyReceipt calibration builds deterministic fixed-shape teacher-forced prefixes', async () => {
  const contract = await calibrationContract({ decoderLength: 8 });
  const target = Int32Array.of(1, 4, 5, 4, 5, 4, 5, 2);
  assert.deepEqual(teacherForcedPrefixLengths(target, 3), [1, 4, 7]);
  assert.deepEqual(teacherForcedPrefixLengths(Int32Array.of(1, 2), 3), [1]);

  const prefix = buildTeacherForcedPrefix(target, 4, contract);
  assert.deepEqual([...prefix.decoderIds], [1, 4, 5, 4, 0, 0, 0, 0]);
  assert.deepEqual([...prefix.decoderKeep], [1, 1, 1, 1, 0, 0, 0, 0]);
});

test('TinyReceipt calibration binds exact runtime-selectable and specialized split ABIs', () => {
  const sample = {
    image: new Float32Array(4),
    questionIds: Int32Array.of(4, 2, 0, 0),
    expectedFamilyId: 1,
  };
  const autoEncoder = { format: 'volvox-graph/v1', inputs: {
    input0: { source_name: 'image', dtype: 'float32' },
    input1: { source_name: 'question_ids', dtype: 'int32' },
    input2: { source_name: 'family_ids', dtype: 'int32' },
  } };
  const encoderInputs = bindEncoderCalibrationInputs(autoEncoder, sample, -1);
  assert.equal(encoderInputs.input0, sample.image);
  assert.deepEqual([...encoderInputs.input2], [-1]);
  assert.equal(calibrationGraphAbi(autoEncoder, 'encoder').routingMode, 'auto');
  const explicitEncoderInputs = bindEncoderCalibrationInputs(
    autoEncoder, sample, -1, 'explicit',
  );
  assert.deepEqual([...explicitEncoderInputs.input2], [1]);
  const sweptEncoderInputs = bindEncoderCalibrationInputs(
    autoEncoder, sample, -1, 'auto', 7,
  );
  assert.deepEqual([...sweptEncoderInputs.input2], [7]);
  assert.equal(calibrationRoutingModeForGraph('auto', 'explicit'), 'explicit');
  assert.equal(calibrationRoutingModeForGraph('specialized', 'explicit'), 'specialized');
  assert.equal(normalizeCalibrationRoutingMode(), 'auto');
  assert.throws(
    () => normalizeCalibrationRoutingMode('guessed'),
    /--routing-mode must be 'auto' or 'explicit'/,
  );

  const specializedEncoder = structuredClone(autoEncoder);
  delete specializedEncoder.inputs.input2;
  const specializedInputs = bindEncoderCalibrationInputs(specializedEncoder, sample, -1);
  assert.deepEqual(Object.keys(specializedInputs).sort(), ['input0', 'input1']);
  assert.deepEqual(
    Object.keys(bindEncoderCalibrationInputs(specializedEncoder, sample, -1, 'explicit')).sort(),
    ['input0', 'input1'],
  );
  assert.throws(
    () => bindEncoderCalibrationInputs(specializedEncoder, sample, -1, 'auto', 7),
    /public route sweep requires a runtime-selectable family input/,
  );
  assert.equal(calibrationGraphAbi(specializedEncoder, 'encoder').routingMode, 'specialized');
  assert.throws(
    () => bindEncoderCalibrationInputs({
      ...specializedEncoder,
      inputs: { ...specializedEncoder.inputs, obsolete: { dtype: 'int32' } },
    }, sample, -1),
    /do not match the current specialized.*or AUTO.*ABI/,
  );

  const produced = {
    memory: new Float32Array(6),
    memoryMask: Int32Array.of(0, 1),
    familyIds: Int32Array.of(1),
    routingMode: 'auto',
  };
  const prefix = {
    decoderIds: Int32Array.of(1, 4, 0, 0),
    decoderKeep: Int32Array.of(1, 1, 0, 0),
  };
  const autoDecoder = { format: 'volvox-graph/v1', inputs: {
    input0: { source_name: 'decoder_input_ids', dtype: 'int32' },
    input1: { source_name: 'memory', dtype: 'float32' },
    input2: { source_name: 'memory_padding_mask', dtype: 'int32' },
    input3: { source_name: 'family_ids', dtype: 'int32' },
    v4_keep: { dtype: 'int32', shape: [1, prefix.decoderIds.length] },
  } };
  const autoDecoderInputs = bindDecoderCalibrationInputs(autoDecoder, produced, prefix);
  assert.equal(autoDecoderInputs.input0, prefix.decoderIds);
  assert.equal(autoDecoderInputs.input3, produced.familyIds);
  assert.equal(autoDecoderInputs.v4_keep, prefix.decoderKeep);
  assert.equal(calibrationGraphAbi(autoDecoder, 'decoder').routingMode, 'auto');
  const explicitDecoderInputs = bindDecoderCalibrationInputs(
    autoDecoder, { ...produced, routingMode: 'explicit' }, prefix,
  );
  assert.equal(explicitDecoderInputs.input3, produced.familyIds);

  const unhoistedAutoDecoder = structuredClone(autoDecoder);
  delete unhoistedAutoDecoder.inputs.v4_keep;
  assert.throws(
    () => bindDecoderCalibrationInputs(unhoistedAutoDecoder, produced, prefix),
    /do not match the current specialized.*or AUTO.*ABI/,
  );

  const specializedDecoder = structuredClone(autoDecoder);
  delete specializedDecoder.inputs.input3;
  const specializedDecoderInputs = bindDecoderCalibrationInputs(
    specializedDecoder, { ...produced, routingMode: 'specialized' }, prefix,
  );
  assert.equal(specializedDecoderInputs.input0, prefix.decoderIds);
  assert.equal(specializedDecoderInputs.v4_keep, prefix.decoderKeep);
  assert.equal(calibrationGraphAbi(specializedDecoder, 'decoder').routingMode, 'specialized');
  assert.throws(
    () => bindDecoderCalibrationInputs(specializedDecoder, produced, prefix),
    /routing mode mismatch: encoder is auto, decoder is specialized/,
  );
  assert.throws(
    () => bindDecoderCalibrationInputs(
      specializedDecoder, { ...produced, routingMode: 'explicit' }, prefix,
    ),
    /routing mode mismatch: encoder is explicit, decoder is specialized/,
  );
});

test('TinyReceipt specialized calibration rejects records for another frozen family', async () => {
  const contract = await calibrationContract();
  assert.equal(
    assertCalibrationRoutingRecords([
      { expectedFamilyId: 0 }, { expectedFamilyId: 0 },
    ], contract, 'specialized'),
    0,
  );
  assert.throws(
    () => assertCalibrationRoutingRecords([
      { expectedFamilyId: 0 }, { expectedFamilyId: 1 },
    ], contract, 'specialized'),
    /must all target the one frozen package family/,
  );
  assert.equal(assertCalibrationRoutingRecords([
    { expectedFamilyId: 0 }, { expectedFamilyId: 1 },
  ], contract, 'auto'), null);
  assert.equal(
    validateSelectedCalibrationFamily(
      { id: 'phone-0', expectedFamilyId: 0 }, Int32Array.of(0), contract, 'specialized',
    ),
    0,
  );
  assert.equal(
    validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 }, Int32Array.of(1), contract, 'explicit',
    ),
    1,
  );
  assert.throws(
    () => validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 }, Int32Array.of(0), contract, 'explicit',
    ),
    /explicit encoder selects phone.*record address-0 is address/,
  );
  assert.throws(
    () => validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 }, Int32Array.of(0), contract, 'specialized',
    ),
    /specialized encoder selects phone.*record address-0 is address/,
  );
  // AUTO routing remains observable provenance rather than relabeling records.
  assert.equal(
    validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 }, Int32Array.of(0), contract, 'auto',
    ),
    0,
  );
  assert.equal(
    validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 },
      Int32Array.of(7), contract, 'explicit', 7,
    ),
    7,
  );
  assert.throws(
    () => validateSelectedCalibrationFamily(
      { id: 'address-0', expectedFamilyId: 1 },
      Int32Array.of(6), contract, 'explicit', 7,
    ),
    /route sweep requested other but encoder selected math/,
  );
});

test('TinyReceipt calibration covers the complete immutable Embedding table', () => {
  const document = {
    nodes: [{
      opType: 'Embedding',
      inputs: { input: 'ids', weight: 'table' },
      outputs: { out: 'embedded' },
    }],
  };
  const observations = { embedded: { min: -0.1, max: 0.2 } };
  const table = {
    isWeight: true,
    dtype: 'float32',
    buffer: Float32Array.of(-1.25, 0.5, 2.5, -0.75),
  };

  assert.equal(
    widenImmutableEmbeddingRanges(
      document, (name) => name === 'table' ? table : null, observations,
    ),
    1,
  );
  assert.deepEqual(observations.embedded, { min: -1.25, max: 2.5 });
});

test('TinyReceipt calibration fails closed on a mutable or non-finite Embedding table', () => {
  const document = {
    nodes: [{
      opType: 'Embedding',
      inputs: { input: 'ids', weight: 'table' },
      outputs: { out: 'embedded' },
    }],
  };
  assert.throws(
    () => widenImmutableEmbeddingRanges(
      document,
      () => ({
        isWeight: true, dtype: 'float32', buffer: Float32Array.of(0, Number.NaN),
      }),
      {},
    ),
    /contains non-finite values/,
  );
});

test('TinyReceipt calibration fails closed on non-finite F32 inputs and outputs', () => {
  const document = { inputs: {
    image: { dtype: 'float32' },
    ids: { dtype: 'int32' },
  } };
  const observations = { image: { min: -1, max: 1 } };
  const original = structuredClone(observations);

  assert.throws(
    () => observeInputs(document, {
      image: Float32Array.of(0, Number.NaN),
      ids: Int32Array.of(0, -1),
    }, observations),
    /graph input 'image' contains a non-finite value at index 1/,
  );
  assert.deepEqual(observations, original);
  assert.throws(
    () => observeFiniteRange(
      observations, 'hidden', Float32Array.of(1, Number.POSITIVE_INFINITY),
      'encoder runtime output',
    ),
    /encoder runtime output 'hidden' contains a non-finite value at index 1/,
  );
  assert.equal(Object.hasOwn(observations, 'hidden'), false);
  assert.throws(
    () => observeFiniteRange(observations, 'empty', new Float32Array(), 'runtime output'),
    /runtime output 'empty' has no values to observe/,
  );

  observeInputs(document, {
    image: Float32Array.of(-2, 0.5),
    ids: Int32Array.of(-2147483648, 2147483647),
  }, observations);
  assert.deepEqual(observations, { image: { min: -2, max: 1 } });
});

test('TinyReceipt calibration requires every runtime F32 candidate and F32 graph input', () => {
  const document = {
    inputs: {
      image: { dtype: 'float32' },
      memory: {},
      ids: { dtype: 'int32' },
    },
    outputs: ['logits', 'token_ids'],
    nodes: [
      {
        outputs: { out: 'hidden' },
        // Loaded runtime metadata is authoritative for calibration coverage.
        outputs_dtype: { out: 'int32' },
      },
      {
        outputs: { out: 'token_ids' },
        outputs_dtype: { out: 'int32' },
      },
    ],
  };
  const tensors = new Map([
    ['hidden', { dtype: 'float32' }],
    ['logits', { dtype: 'float32' }],
    ['token_ids', { dtype: 'int32' }],
  ]);
  const candidates = runtimeF32CandidateTensors(document, (name) => tensors.get(name));
  assert.deepEqual(candidates, ['logits', 'hidden']);

  const complete = {
    image: { min: -1, max: 1 },
    memory: { min: -2, max: 2 },
    logits: { min: -3, max: 3 },
    hidden: { min: -4, max: 4 },
    unrelated: { min: 0, max: 0 },
  };
  assert.equal(
    assertCompleteCalibrationRanges(document, candidates, complete, 'encoder'),
    4,
  );
  assert.throws(
    () => assertCompleteCalibrationRanges(
      document, candidates, { ...complete, hidden: undefined }, 'encoder',
    ),
    /encoder: incomplete F32 calibration ranges \(missing: hidden\)/,
  );
  assert.throws(
    () => assertCompleteCalibrationRanges(
      document, candidates, { ...complete, memory: { min: Number.NaN, max: 1 } }, 'encoder',
    ),
    /invalid: memory/,
  );
});

test('TinyReceipt calibration excludes only proven {0,-Inf} attention-mask domains', () => {
  const tensor = (dtype = 'float32') => ({ dtype });
  const weight = (...values) => ({
    isWeight: true, dtype: 'float32', buffer: Float32Array.of(...values),
  });
  const tensors = new Map([
    ['condition', tensor('int32')],
    ['zero', weight(0, 0)],
    ['padding_sentinel', weight(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY)],
    ['causal_sentinel', weight(0, Number.NEGATIVE_INFINITY)],
    ['padding_mask', tensor()],
    ['combined_mask', tensor()],
    ['viewed_mask', tensor()],
    ['logits', tensor()],
    ['masked_logits', tensor()],
    ['probabilities', tensor()],
  ]);
  const document = {
    outputs: ['probabilities'],
    nodes: [
      {
        opType: 'Where',
        inputs: { condition: 'condition', x: 'padding_sentinel', y: 'zero' },
        outputs: { out: 'padding_mask' },
      },
      {
        opType: 'Add',
        inputs: { a: 'causal_sentinel', b: 'padding_mask' },
        outputs: { out: 'combined_mask' },
      },
      {
        opType: 'Reshape', inputs: { input: 'combined_mask' },
        outputs: { out: 'viewed_mask' },
      },
      {
        opType: 'Add', inputs: { a: 'logits', b: 'viewed_mask' },
        outputs: { out: 'masked_logits' },
      },
      {
        opType: 'Softmax', inputs: { input: 'masked_logits' },
        outputs: { out: 'probabilities' },
      },
    ],
  };
  const proof = proveAdditiveAttentionMaskExclusions(
    document, (name) => tensors.get(name),
  );
  assert.deepEqual(proof.tensors, [
    'combined_mask', 'masked_logits', 'padding_mask', 'viewed_mask',
  ]);
  assert.equal(proof.reasons.padding_mask, 'zero-negative-infinity-where-mask');
  assert.equal(proof.reasons.combined_mask, 'combined-attention-mask');
  assert.equal(proof.reasons.viewed_mask, 'shape-only-attention-mask');
  assert.equal(
    proof.reasons.masked_logits, 'additive-attention-logits-before-softmax',
  );
  assert.deepEqual(
    runtimeF32CandidateTensors(document, (name) => tensors.get(name), proof.tensors),
    ['probabilities'],
  );
});

test('TinyReceipt attention-mask proof rejects ambiguous infinity fanout', () => {
  const tensors = new Map([
    ['condition', { dtype: 'int32' }],
    ['zero', { isWeight: true, dtype: 'float32', buffer: Float32Array.of(0) }],
    ['sentinel', {
      isWeight: true,
      dtype: 'float32',
      buffer: Float32Array.of(Number.NEGATIVE_INFINITY),
    }],
    ['mask', { dtype: 'float32' }],
    ['logits', { dtype: 'float32' }],
    ['masked', { dtype: 'float32' }],
    ['escaped', { dtype: 'float32' }],
  ]);
  assert.throws(
    () => proveAdditiveAttentionMaskExclusions({
      outputs: ['escaped'],
      nodes: [
        {
          opType: 'Where',
          inputs: { condition: 'condition', x: 'sentinel', y: 'zero' },
          outputs: { out: 'mask' },
        },
        {
          opType: 'Add', inputs: { a: 'logits', b: 'mask' },
          outputs: { out: 'masked' },
        },
        {
          opType: 'Identity', inputs: { input: 'masked' },
          outputs: { out: 'escaped' },
        },
      ],
    }, (name) => tensors.get(name)),
    /attention mask 'mask' has an ambiguous numeric Add at 'masked'/,
  );
});

test('TinyReceipt calibration provenance binds records, routing, settings, and artifacts', async () => {
  const contract = await calibrationContract();
  const samples = [
    { id: 'phone-0', expectedFamilyId: 0 },
    { id: 'address-0', expectedFamilyId: 1 },
    { id: 'phone-1', expectedFamilyId: 0 },
  ];
  const routeExecutions = [
    { ...samples[0], familyIds: Int32Array.of(1), routingMode: 'auto' },
    { ...samples[1], familyIds: Int32Array.of(1), routingMode: 'auto' },
    { ...samples[2], familyIds: Int32Array.of(0), routingMode: 'auto' },
  ];
  const calibrationManifest = Object.freeze({
    path: 'calibration-records.json',
    sha256: 'a'.repeat(64),
    bytes: 128,
  });
  const packageArtifact = Object.freeze({
    graphs: Object.freeze({
      encoder: Object.freeze({
        graph: Object.freeze({
          path: 'encoder/graph.json', sha256: 'b'.repeat(64), bytes: 256,
        }),
        weights: Object.freeze({
          path: 'encoder/model.safetensors', sha256: 'c'.repeat(64), bytes: 512,
        }),
      }),
    }),
  });
  const calibratorSource = Object.freeze({
    path: 'examples/tiny_receipt_vqa/tools/calibrate_split_activations.mjs',
    sha256: 'd'.repeat(64),
    bytes: 1024,
  });

  const provenance = buildCalibrationProvenance({
    calibrationManifest,
    samples,
    requiredFamilyIds: [0, 1],
    routeExecutions,
    contract,
    prefixesPerRecord: 3,
    backend: 'cpu',
    batch: 24,
    kinds: ['encoder', 'decoder'],
    routingMode: 'auto',
    packageArtifact,
    calibratorSource,
    affineExclusions: {
      encoder: { semantic_domain: 'additive-attention-mask-{0,-Infinity}/v1' },
    },
  });

  assert.deepEqual(provenance.calibration_manifest, calibrationManifest);
  assert.equal(provenance.fixtures, null);
  assert.deepEqual(provenance.selected_records, {
    count: 3,
    ids: ['phone-0', 'address-0', 'phone-1'],
  });
  assert.deepEqual(provenance.family_counts.required, {
    phone: 1, address: 1, store: 0, item_row: 0,
    item_math: 0, item_lookup: 0, math: 0, other: 0,
  });
  assert.deepEqual(provenance.family_counts.record, {
    phone: 2, address: 1, store: 0, item_row: 0,
    item_math: 0, item_lookup: 0, math: 0, other: 0,
  });
  assert.deepEqual(provenance.route_executions, {
    mode: 'graph-routing',
    count: 3,
    family_counts: {
      phone: 1, address: 2, store: 0, item_row: 0,
      item_math: 0, item_lookup: 0, math: 0, other: 0,
    },
  });
  assert.deepEqual(provenance.settings, {
    samples: 3,
    prefixes_per_record: 3,
    backend: 'cpu',
    batch: 24,
    graphs: ['encoder', 'decoder'],
    routing_mode: 'auto',
  });
  assert.deepEqual(provenance.package, packageArtifact);
  assert.deepEqual(provenance.calibrator_source, calibratorSource);
  assert.deepEqual(provenance.affine_exclusions, {
    encoder: { semantic_domain: 'additive-attention-mask-{0,-Infinity}/v1' },
  });
  const explicitProvenance = buildCalibrationProvenance({
    calibrationManifest,
    samples,
    requiredFamilyIds: [0, 1],
    routeExecutions: samples.map((sample) => ({
      ...sample,
      familyIds: Int32Array.of(sample.expectedFamilyId),
      routingMode: 'explicit',
    })),
    contract,
    prefixesPerRecord: 3,
    backend: 'cpu',
    batch: 24,
    kinds: ['encoder', 'decoder'],
    routingMode: 'explicit',
    packageArtifact,
    calibratorSource,
  });
  assert.equal(explicitProvenance.settings.routing_mode, 'explicit');
  assert.deepEqual(explicitProvenance.route_executions, {
    mode: 'graph-routing',
    count: 3,
    family_counts: {
      phone: 2, address: 1, store: 0, item_row: 0,
      item_math: 0, item_lookup: 0, math: 0, other: 0,
    },
  });
  const swept = calibrationRouteExecutions(samples, contract, 'all-public');
  const sweptProvenance = buildCalibrationProvenance({
    calibrationManifest,
    samples,
    requiredFamilyIds: [0, 1],
    routeExecutions: swept.map((execution) => ({
      ...execution,
      familyIds: Int32Array.of(execution.routeFamilyId),
      routingMode: 'explicit',
    })),
    routeSweep: 'all-public',
    contract,
    prefixesPerRecord: 3,
    backend: 'cpu',
    batch: 24,
    kinds: ['encoder', 'decoder'],
    routingMode: 'explicit',
    packageArtifact,
    calibratorSource,
  });
  assert.deepEqual(sweptProvenance.family_counts.record, {
    phone: 2, address: 1, store: 0, item_row: 0,
    item_math: 0, item_lookup: 0, math: 0, other: 0,
  });
  assert.deepEqual(sweptProvenance.route_executions, {
    mode: 'all-public',
    count: samples.length * FAMILY_NAMES.length,
    family_counts: Object.fromEntries(FAMILY_NAMES.map((name) => [name, samples.length])),
  });
  const balancedSamples = Array.from({ length: 16 }, (_, index) => ({
    ...samples[index % samples.length],
    id: `balanced-${index}`,
  }));
  const balanced = calibrationRouteExecutions(
    balancedSamples, contract, 'balanced-public',
  );
  const balancedProvenance = buildCalibrationProvenance({
    calibrationManifest,
    samples: balancedSamples,
    requiredFamilyIds: [0, 1],
    routeExecutions: balanced.map((execution) => ({
      ...execution,
      familyIds: Int32Array.of(execution.routeFamilyId),
      routingMode: 'explicit',
    })),
    routeSweep: 'balanced-public',
    contract,
    prefixesPerRecord: 3,
    backend: 'wasm',
    batch: 512,
    kinds: ['encoder', 'decoder'],
    routingMode: 'explicit',
    packageArtifact,
    calibratorSource,
  });
  assert.deepEqual(balancedProvenance.route_executions, {
    mode: 'balanced-public',
    count: 16,
    family_counts: Object.fromEntries(FAMILY_NAMES.map((name) => [name, 2])),
  });
  assert.throws(
    () => buildCalibrationProvenance({ routingMode: 'runtime' }),
    /must be auto, explicit, specialized, or fixture/,
  );
  assert.throws(
    () => buildCalibrationProvenance({
      calibrationManifest: {
        ...calibrationManifest,
        path: '/absolute/calibration-records.json',
      },
      samples,
      requiredFamilyIds: [0, 1],
      routeExecutions,
      contract,
      prefixesPerRecord: 3,
      backend: 'cpu',
      batch: 24,
      kinds: ['encoder'],
      routingMode: 'auto',
      packageArtifact,
      calibratorSource,
    }),
    /normalized logical relative path/,
  );
  const fixtureIdentity = Object.freeze({
    files: Object.freeze({
      'image.f32': Object.freeze({
        path: 'fixtures/image.f32', sha256: 'e'.repeat(64), bytes: 64,
      }),
    }),
  });
  const fixtureProvenance = buildCalibrationProvenance({
    calibrationManifest: null,
    fixturesArtifact: fixtureIdentity,
    samples: [],
    requiredFamilyIds: [],
    routeExecutions: null,
    contract,
    prefixesPerRecord: 0,
    backend: 'cpu',
    batch: 24,
    kinds: ['encoder'],
    routingMode: 'fixture',
    packageArtifact,
    calibratorSource,
  });
  assert.deepEqual(fixtureProvenance.fixtures, fixtureIdentity);
  const encoder = { image: { min: -1, max: 1 } };
  const decoder = { logits: { min: -2, max: 2 } };
  const report = buildCalibrationReport(provenance, { encoder, decoder });
  assert.deepEqual(Object.keys(report), ['provenance', 'encoder', 'decoder']);
  assert.equal(JSON.stringify(report).includes('/absolute/'), false);
  assert.equal(report.encoder, encoder);
  assert.equal(report.decoder, decoder);
  assert.equal(
    sha256Hex('abc'),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  );
});

test('TinyReceipt calibration artifact identities expose only logical relative paths', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'volvox-calibration-identity-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const kind of ['encoder', 'decoder']) {
    await mkdir(join(root, kind));
    await writeFile(join(root, kind, 'graph.json'), `{\"kind\":\"${kind}\"}\n`);
    await writeFile(join(root, kind, 'model.safetensors'), `${kind}-weights`);
  }
  const source = await fileIdentity(
    join(root, 'encoder', 'graph.json'), 'logical/graph.json',
  );
  assert.equal(source.path, 'logical/graph.json');
  assert.equal(source.bytes, 19);
  assert.match(source.sha256, /^[0-9a-f]{64}$/);
  const artifact = await packageIdentity(root);
  assert.deepEqual(
    Object.values(artifact.graphs).flatMap((entry) => [entry.graph.path, entry.weights.path]),
    [
      'encoder/graph.json', 'encoder/model.safetensors',
      'decoder/graph.json', 'decoder/model.safetensors',
    ],
  );
  assert.equal(JSON.stringify(artifact).includes(root), false);
  await assert.rejects(
    fileIdentity(join(root, 'encoder', 'graph.json'), '/absolute/graph.json'),
    /normalized logical relative path/,
  );
});
