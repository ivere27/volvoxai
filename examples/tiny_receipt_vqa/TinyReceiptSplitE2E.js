/*
 * Deterministic end-to-end validation for the split TinyReceipt package.
 *
 * This file is deliberately example-owned and environment-neutral. Node,
 * Deno/WebGPU, and native-orchestration tools consume the same workload and
 * reference contract without putting machine-local paths in saved artifacts.
 */

import {
  TinyReceiptCharVocab,
} from './TinyReceiptW8A8Session.js';
import {
  TinyReceiptSplitSession,
  TINY_RECEIPT_SPLIT_PACKAGE_FORMAT,
} from './TinyReceiptSplitSession.js';

const RESULT_SCHEMA = 'volvoxai.tiny-receipt-split-e2e-result/v1';
const REFERENCE_SCHEMA = 'volvoxai.tiny-receipt-split-e2e-reference/v1';
const FIXTURE_SCHEMA = 'volvoxai.tiny-receipt-split-e2e-fixture/v1';
const WORKLOAD_ID = 'synthetic-exact-f32-v1';
const BACKENDS = Object.freeze(['cpu', 'wasm', 'webgpu']);
const FAMILY_ORDER = Object.freeze([
  'phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other',
]);
const IMAGE_WIDTH = 672;
const IMAGE_HEIGHT = 320;
const QUESTION_LENGTH = 192;
const DECODER_LENGTH = 192;
const IMAGE_SHA256 = '028acedd12b13cfcd706b8c364f41e82dd80fd34218e61612e31c0c9ad5474fa';

export const TINY_RECEIPT_SPLIT_E2E_WORKLOAD = Object.freeze({
  id: WORKLOAD_ID,
  prompt: 'phone number last one',
  family: 'auto',
  maxNewTokens: 4,
  minimumDecoderSteps: 2,
  image: Object.freeze({
    generator: 'q=(17*x+29*y+7*(x^y))&255; f32=(q-128)/128',
    dtype: 'float32',
    shape: Object.freeze([1, 1, IMAGE_HEIGHT, IMAGE_WIDTH]),
    byteOrder: 'little',
  }),
});

function fail(message) {
  throw new Error(`[TinyReceiptSplitE2E] ${message}`);
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, keys, label) {
  if (!isRecord(value) ||
      Object.keys(value).sort().join('\0') !== [...keys].sort().join('\0')) {
    fail(`${label} must contain exactly ${keys.join(', ')}.`);
  }
  return value;
}

function nonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} must be non-empty.`);
  return value;
}

function sha256String(value, label) {
  const result = nonEmptyString(value, label);
  if (!/^[0-9a-f]{64}$/.test(result)) fail(`${label} must be a lowercase SHA-256 digest.`);
  return result;
}

function nonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) fail(`${label} must be a non-negative integer.`);
  return value;
}

function finiteNumber(value, label) {
  if (!Number.isFinite(value)) fail(`${label} must be finite.`);
  return value;
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function bytesToHex(bytes) {
  let result = '';
  for (const value of bytes) result += value.toString(16).padStart(2, '0');
  return result;
}

async function sha256(bytes) {
  if (!(bytes instanceof Uint8Array)) fail('SHA-256 input must be Uint8Array.');
  const digest = await globalThis.crypto?.subtle?.digest('SHA-256', bytes);
  if (!digest) fail('Web Crypto SHA-256 is unavailable.');
  return bytesToHex(new Uint8Array(digest));
}

function float32Bytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index++) {
    view.setFloat32(index * 4, values[index], true);
  }
  return bytes;
}

function int32Bytes(values) {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index++) {
    view.setInt32(index * 4, values[index], true);
  }
  return bytes;
}

function numericSummary(values, label) {
  if (!ArrayBuffer.isView(values) || values.length === 0) {
    fail(`${label} must be a non-empty typed array.`);
  }
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let absoluteSum = 0;
  let sumSquares = 0;
  for (let index = 0; index < values.length; index++) {
    const value = finiteNumber(Number(values[index]), `${label}[${index}]`);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    absoluteSum += Math.abs(value);
    sumSquares += value * value;
  }
  return Object.freeze({
    count: values.length,
    min: minimum,
    max: maximum,
    sum,
    absSum: absoluteSum,
    sumSquares,
  });
}

function normalizedWorkload(value = TINY_RECEIPT_SPLIT_E2E_WORKLOAD) {
  if (!isRecord(value) || value.id !== WORKLOAD_ID ||
      value.prompt !== TINY_RECEIPT_SPLIT_E2E_WORKLOAD.prompt ||
      value.family !== 'auto' ||
      value.maxNewTokens !== TINY_RECEIPT_SPLIT_E2E_WORKLOAD.maxNewTokens ||
      value.minimumDecoderSteps !== TINY_RECEIPT_SPLIT_E2E_WORKLOAD.minimumDecoderSteps ||
      canonicalJson(value.image) !== canonicalJson(TINY_RECEIPT_SPLIT_E2E_WORKLOAD.image)) {
    fail(`workload must match the fixed '${WORKLOAD_ID}' contract.`);
  }
  return TINY_RECEIPT_SPLIT_E2E_WORKLOAD;
}

export function createTinyReceiptSplitE2EImage() {
  const image = new Float32Array(IMAGE_HEIGHT * IMAGE_WIDTH);
  for (let y = 0; y < IMAGE_HEIGHT; y++) {
    for (let x = 0; x < IMAGE_WIDTH; x++) {
      const quantized = (17 * x + 29 * y + 7 * (x ^ y)) & 255;
      // Division by 128 is exact in binary, so C, Python, JS, and WGSL fixture
      // readers receive the same IEEE-754 F32 bit pattern.
      image[y * IMAGE_WIDTH + x] = (quantized - 128) / 128;
    }
  }
  return image;
}

export function requireTinyReceiptPhysicalAdapterIdentity(
  value,
  requiredIdentity,
) {
  const required = nonEmptyString(requiredIdentity, 'required adapter identity').trim();
  if (!required) fail('required adapter identity must be non-empty.');
  if (!isRecord(value)) fail('selected WebGPU adapter identity is unavailable.');
  const identity = Object.fromEntries(Object.entries(value)
    .filter(([, entry]) => typeof entry === 'string' && entry.trim())
    .map(([key, entry]) => [key, entry.trim()]));
  const description = Object.values(identity).join(' ');
  if (!description ||
      /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software rasterizer|microsoft basic render|cpu)\b/i
        .test(description)) {
    fail('selected WebGPU adapter is missing or identifies a software device.');
  }
  if (!description.toLowerCase().includes(required.toLowerCase())) {
    fail('selected WebGPU adapter does not match the required identity.');
  }
  return Object.freeze(identity);
}

function checkedTokenIds(
  values,
  label,
  maximumLength = DECODER_LENGTH,
  vocabularySize = null,
) {
  if (!Array.isArray(values) || values.length > maximumLength ||
      values.some((value) => !Number.isSafeInteger(value) || value < 0 ||
        (vocabularySize != null && value >= vocabularySize))) {
    const suffix = vocabularySize == null ? 'non-negative token IDs'
      : `token IDs from 0 through ${vocabularySize - 1}`;
    fail(`${label} must contain ${suffix}.`);
  }
  return values;
}

function familyId(value) {
  if (value === 'auto') return -1;
  const result = FAMILY_ORDER.indexOf(value);
  if (result < 0) fail(`family must be 'auto' or one of ${FAMILY_ORDER.join(', ')}.`);
  return result;
}

export function createTinyReceiptSplitE2EBindings({
  vocab: suppliedVocab = null,
  itos,
  tokenIds = { pad: 0, bos: 1, eos: 2, unk: 3 },
  workload = TINY_RECEIPT_SPLIT_E2E_WORKLOAD,
  decoderPrefix = [],
} = {}) {
  const normalized = normalizedWorkload(workload);
  const vocab = suppliedVocab ?? new TinyReceiptCharVocab(itos, tokenIds);
  if (!Array.isArray(vocab?.itos) || vocab.itos.length === 0 ||
      typeof vocab.encodeQuestion !== 'function' ||
      !Number.isInteger(vocab.pad) || !Number.isInteger(vocab.bos)) {
    fail('vocab must expose itos, PAD/BOS IDs, and encodeQuestion().');
  }
  const question = vocab.encodeQuestion(normalized.prompt, QUESTION_LENGTH);
  const questionIds = new Int32Array(QUESTION_LENGTH);
  questionIds.fill(vocab.pad);
  questionIds.set(question);
  const prefix = checkedTokenIds(
    [...decoderPrefix],
    'decoderPrefix',
    DECODER_LENGTH - 1,
    vocab.itos.length,
  );
  const decoderInputIds = new Int32Array(DECODER_LENGTH);
  decoderInputIds.fill(vocab.pad);
  decoderInputIds[0] = vocab.bos;
  decoderInputIds.set(prefix, 1);
  return Object.freeze({
    image: createTinyReceiptSplitE2EImage(),
    questionTokenIds: Object.freeze([...question]),
    questionIds,
    familyIds: new Int32Array([familyId(normalized.family)]),
    decoderInputIds,
  });
}

async function verifiedAssetDigest(fetchImpl, record, url, label) {
  let response;
  try {
    response = await fetchImpl(url);
  } catch (error) {
    fail(`${label} could not be fetched for package verification: ${error?.message || error}`);
  }
  if (!response || response.ok === false || typeof response.arrayBuffer !== 'function') {
    fail(`${label} did not return a readable byte response for package verification.`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength !== record.bytes) {
    fail(`${label} has ${bytes.byteLength} bytes; its manifest declares ${record.bytes}.`);
  }
  const digest = await sha256(bytes);
  if (digest !== record.sha256) fail(`${label} does not match its manifest SHA-256.`);
  return digest;
}

async function packageIdentity(session, fetchImpl, verifyAssets) {
  const packageInfo = session?.package;
  if (!packageInfo || !session?.vocab?.itos) fail('session omitted package or vocabulary identity.');
  const definitions = [
    ['encoderGraph', packageInfo.encoder?.graph, packageInfo.encoder?.graphUrl, 'encoder graph'],
    ['encoderWeights', packageInfo.encoder?.weights, packageInfo.encoder?.weightsUrl, 'encoder weights'],
    ['decoderGraph', packageInfo.decoder?.graph, packageInfo.decoder?.graphUrl, 'decoder graph'],
    ['decoderWeights', packageInfo.decoder?.weights, packageInfo.decoder?.weightsUrl, 'decoder weights'],
  ];
  const entries = [];
  for (const [name, record, url, label] of definitions) {
    const digest = sha256String(record?.sha256, `${label} SHA-256`);
    if (verifyAssets) {
      if (!Number.isSafeInteger(record?.bytes) || record.bytes <= 0) {
        fail(`${label} must declare positive bytes.`);
      }
      nonEmptyString(url, `${label} URL`);
      await verifiedAssetDigest(fetchImpl, record, url, label);
    }
    entries.push([name, digest]);
  }
  const assets = Object.freeze(Object.fromEntries(entries));
  const vocabularyBytes = new TextEncoder().encode(canonicalJson(session.vocab.itos));
  const semanticTokenizerHash = session.vocab.tokenizerHash == null
    ? null
    : sha256String(session.vocab.tokenizerHash, 'BPE tokenizer hash');
  return Object.freeze({
    format: TINY_RECEIPT_SPLIT_PACKAGE_FORMAT,
    assets,
    // BPE's release fingerprint covers itos, ranked merges, boundaries, and
    // byte/unused declarations. Legacy char-vocab references retain their
    // historical canonical-itos digest.
    vocabularySha256: semanticTokenizerHash || await sha256(vocabularyBytes),
  });
}

function strictRoute(value, label) {
  if (!isRecord(value) || value.tierFallback !== false ||
      !isRecord(value.operator) || value.operator.attestation !== 'none' ||
      value.operator.used !== false || value.operator.offendingNode != null) {
    fail(`${label} does not prove a strict no-fallback route.`);
  }
}

function strictProviderSummary(diagnostics, backend, decoderSteps, minimumDecoderSteps) {
  const compilation = diagnostics.filter(({ kind }) => kind === 'compilation')
    .map(({ report }) => report);
  const execution = diagnostics.filter(({ kind }) => kind === 'execution')
    .map(({ report }) => report);
  const failures = diagnostics.filter(({ kind }) => kind === 'execution-error');
  if (failures.length !== 0) fail('runtime emitted an execution-error diagnostic.');
  if (compilation.length !== 2) {
    fail(`expected two compilations (encoder and decoder), received ${compilation.length}.`);
  }
  for (const [index, report] of compilation.entries()) {
    const policy = report?.requestedPolicy;
    if (policy?.mode !== 'require' || policy.backend !== backend ||
        policy.operatorFallback !== 'forbid' || report.selectedBackend !== backend ||
        !Array.isArray(report.candidates) || report.candidates.length !== 1 ||
        report.candidates[0]?.backend !== backend ||
        report.candidates[0]?.outcome !== 'selected' ||
        canonicalJson(report.candidates[0]?.device) !== canonicalJson(report.selectedDevice)) {
      fail(`compilation ${index} did not strictly select '${backend}'.`);
    }
    strictRoute(report.routeEvidence, `compilation ${index}`);
    strictRoute(report.candidates[0].routeEvidence, `compilation candidate ${index}`);
  }
  if (decoderSteps < minimumDecoderSteps) {
    fail(`decoder executed ${decoderSteps} step(s), fewer than required ${minimumDecoderSteps}.`);
  }
  if (execution.length !== decoderSteps + 1) {
    fail(`expected one encoder plus ${decoderSteps} decoder executions, received ${execution.length}.`);
  }
  for (const [index, report] of execution.entries()) {
    if (report?.outcome !== 'success' || report.backend !== backend ||
        report.operatorFallback !== 'none' ||
        report.decodeState?.operation !== 'execute') {
      fail(`execution ${index} did not remain on strict '${backend}'.`);
    }
    strictRoute(report.routeEvidence, `execution ${index}`);
  }
  const encoderContext = execution[0].contextId;
  const decoderContext = execution[1].contextId;
  if (!nonEmptyString(encoderContext, 'encoder context ID') ||
      !nonEmptyString(decoderContext, 'decoder context ID') ||
      encoderContext === decoderContext ||
      execution.slice(1).some((report) => report.contextId !== decoderContext)) {
    fail('execution diagnostics do not prove one isolated encoder and one isolated decoder context.');
  }
  const deviceKeys = new Map();
  for (const report of compilation) {
    const identity = report.selectedDevice == null ? null : cloneJson(report.selectedDevice);
    deviceKeys.set(canonicalJson(identity), identity);
  }
  if (deviceKeys.size !== 1) {
    fail('encoder and decoder compilations selected different device identities.');
  }
  const selectedDeviceKey = canonicalJson(compilation[0].selectedDevice);
  if (execution.some((report) => canonicalJson(report.device) !== selectedDeviceKey)) {
    fail('execution device identity differs from its strict compilation.');
  }
  const executionTimes = execution.map(({ executionTimeMs }) =>
    finiteNumber(executionTimeMs, 'executionTimeMs'));
  return Object.freeze({
    strictNoFallback: true,
    compilationCount: compilation.length,
    executionCount: execution.length,
    encoderExecutions: 1,
    decoderExecutions: decoderSteps,
    contextCount: 2,
    devices: Object.freeze([...deviceKeys.values()]),
    executionTimeMs: Object.freeze({
      encoder: executionTimes[0],
      decoderTotal: executionTimes.slice(1).reduce((sum, value) => sum + value, 0),
      decoderMean: executionTimes.slice(1).reduce((sum, value) => sum + value, 0) /
        decoderSteps,
    }),
  });
}

async function outputSummary(answer, vocabularySize) {
  const questionTokenIds = checkedTokenIds(
    [...(answer?.questionTokenIds || [])],
    'answer.questionTokenIds',
    QUESTION_LENGTH,
    vocabularySize,
  );
  const tokenIds = checkedTokenIds(
    [...(answer?.tokenIds || [])],
    'answer.tokenIds',
    TINY_RECEIPT_SPLIT_E2E_WORKLOAD.maxNewTokens,
    vocabularySize,
  );
  if (!(answer.routerLogits instanceof Float32Array) || answer.routerLogits.length !== 8) {
    fail('answer.routerLogits must be F32[8].');
  }
  const router = answer.routerLogits.slice();
  const text = typeof answer.text === 'string' ? answer.text : fail('answer.text must be a string.');
  return Object.freeze({
    family: nonEmptyString(answer.family, 'answer.family'),
    familyId: nonNegativeInteger(answer.familyId, 'answer.familyId'),
    requestedFamily: answer.requestedFamily,
    requestedFamilyId: answer.requestedFamilyId,
    questionTokenIds: Object.freeze(questionTokenIds),
    questionTokenIdsSha256: await sha256(int32Bytes(questionTokenIds)),
    tokenIds: Object.freeze(tokenIds),
    tokenIdsSha256: await sha256(int32Bytes(tokenIds)),
    text,
    textUtf8Sha256: await sha256(new TextEncoder().encode(text)),
    stoppedAtEos: answer.stoppedAtEos === true,
    routerLogits: Object.freeze({
      values: Object.freeze([...router]),
      sha256: await sha256(float32Bytes(router)),
      summary: numericSummary(router, 'routerLogits'),
    }),
  });
}

function validateReferenceShape(reference) {
  exactKeys(reference, ['schema', 'provenance', 'package', 'workload', 'expected'], 'reference');
  if (reference.schema !== REFERENCE_SCHEMA) fail(`reference schema must be '${REFERENCE_SCHEMA}'.`);
  exactKeys(reference.provenance, [
    'kind', 'sourceFormat', 'sourceVariant', 'provider', 'description',
  ], 'reference.provenance');
  if (reference.provenance.kind !== 'onnx-runtime-oracle' ||
      reference.provenance.sourceFormat !== 'tiny_receipt_vqa_split_onnx_v1' ||
      reference.provenance.sourceVariant !== 'int8-w8a8') {
    fail('reference provenance must identify the split INT8 ONNX Runtime oracle.');
  }
  nonEmptyString(reference.provenance.provider, 'reference.provenance.provider');
  nonEmptyString(reference.provenance.description, 'reference.provenance.description');
  exactKeys(reference.package, ['format', 'assets', 'vocabularySha256'], 'reference.package');
  if (reference.package.format !== TINY_RECEIPT_SPLIT_PACKAGE_FORMAT) {
    fail('reference package format is invalid.');
  }
  exactKeys(reference.package.assets, [
    'encoderGraph', 'encoderWeights', 'decoderGraph', 'decoderWeights',
  ], 'reference.package.assets');
  for (const [name, digest] of Object.entries(reference.package.assets)) {
    sha256String(digest, `reference.package.assets.${name}`);
  }
  sha256String(reference.package.vocabularySha256, 'reference.package.vocabularySha256');
  exactKeys(reference.workload, [
    'id', 'prompt', 'family', 'maxNewTokens', 'minimumDecoderSteps', 'image',
  ], 'reference.workload');
  const image = exactKeys(
    reference.workload.image,
    ['generator', 'dtype', 'shape', 'byteOrder', 'sha256'],
    'reference.workload.image',
  );
  const baseImage = TINY_RECEIPT_SPLIT_E2E_WORKLOAD.image;
  if (reference.workload.id !== WORKLOAD_ID ||
      reference.workload.prompt !== TINY_RECEIPT_SPLIT_E2E_WORKLOAD.prompt ||
      reference.workload.family !== 'auto' ||
      reference.workload.maxNewTokens !== 4 ||
      reference.workload.minimumDecoderSteps !== 2 ||
      image.generator !== baseImage.generator || image.dtype !== 'float32' ||
      image.byteOrder !== 'little' || !sameArray(image.shape, baseImage.shape)) {
    fail('reference workload does not match the fixed deterministic workload.');
  }
  sha256String(image.sha256, 'reference.workload.image.sha256');
  exactKeys(reference.expected, [
    'family', 'familyId', 'requestedFamily', 'requestedFamilyId',
    'questionTokenIds', 'questionTokenIdsSha256', 'tokenIds', 'tokenIdsSha256',
    'textUtf8Sha256', 'stoppedAtEos', 'routerLogits',
  ], 'reference.expected');
  checkedTokenIds(reference.expected.questionTokenIds, 'reference expected questionTokenIds');
  checkedTokenIds(reference.expected.tokenIds, 'reference expected tokenIds', 4);
  if (reference.expected.questionTokenIds.length === 0 ||
      reference.expected.tokenIds.length < reference.workload.minimumDecoderSteps ||
      FAMILY_ORDER[reference.expected.familyId] !== reference.expected.family ||
      reference.expected.requestedFamily !== 'auto' ||
      reference.expected.requestedFamilyId !== -1 ||
      typeof reference.expected.stoppedAtEos !== 'boolean') {
    fail('reference expected routing/token contract is invalid.');
  }
  sha256String(reference.expected.questionTokenIdsSha256, 'reference questionTokenIdsSha256');
  sha256String(reference.expected.tokenIdsSha256, 'reference tokenIdsSha256');
  sha256String(reference.expected.textUtf8Sha256, 'reference textUtf8Sha256');
  const router = exactKeys(
    reference.expected.routerLogits,
    ['values', 'sha256', 'summary', 'atol', 'rtol'],
    'reference.expected.routerLogits',
  );
  if (!Array.isArray(router.values) || router.values.length !== 8) {
    fail('reference router logits must contain eight values.');
  }
  router.values.forEach((value, index) => finiteNumber(value, `reference router[${index}]`));
  sha256String(router.sha256, 'reference router SHA-256');
  exactKeys(router.summary, ['count', 'min', 'max', 'sum', 'absSum', 'sumSquares'],
    'reference router summary');
  if (router.summary.count !== 8) fail('reference router summary count must be eight.');
  for (const name of ['min', 'max', 'sum', 'absSum', 'sumSquares']) {
    finiteNumber(router.summary[name], `reference router summary ${name}`);
  }
  if (!Number.isFinite(router.atol) || router.atol < 0 ||
      !Number.isFinite(router.rtol) || router.rtol < 0) {
    fail('reference router tolerances must be finite and non-negative.');
  }
  return reference;
}

async function validateReferenceIntegrity(reference) {
  validateReferenceShape(reference);
  const questionDigest = await sha256(int32Bytes(reference.expected.questionTokenIds));
  const tokenDigest = await sha256(int32Bytes(reference.expected.tokenIds));
  const routerValues = Float32Array.from(reference.expected.routerLogits.values);
  const routerDigest = await sha256(float32Bytes(routerValues));
  const routerSummary = numericSummary(routerValues, 'reference routerLogits');
  const imageDigest = await sha256(float32Bytes(createTinyReceiptSplitE2EImage()));
  if (questionDigest !== reference.expected.questionTokenIdsSha256 ||
      tokenDigest !== reference.expected.tokenIdsSha256 ||
      routerDigest !== reference.expected.routerLogits.sha256 ||
      canonicalJson(routerSummary) !== canonicalJson(reference.expected.routerLogits.summary) ||
      imageDigest !== IMAGE_SHA256 ||
      reference.workload.image.sha256 !== IMAGE_SHA256) {
    fail('reference hashes or numeric summaries are internally inconsistent.');
  }
  return reference;
}

function compareReference(report, reference) {
  validateReferenceShape(reference);
  if (canonicalJson(report.package) !== canonicalJson(reference.package)) {
    fail('runtime package identity does not match the oracle reference.');
  }
  const expectedWorkload = {
    ...TINY_RECEIPT_SPLIT_E2E_WORKLOAD,
    image: {
      ...TINY_RECEIPT_SPLIT_E2E_WORKLOAD.image,
      sha256: report.workload.image.sha256,
    },
  };
  if (canonicalJson(reference.workload) !== canonicalJson(expectedWorkload)) {
    fail('runtime workload identity does not match the oracle reference.');
  }
  const actual = report.output;
  const expected = reference.expected;
  for (const name of [
    'family', 'familyId', 'requestedFamily', 'requestedFamilyId',
    'questionTokenIdsSha256', 'tokenIdsSha256', 'textUtf8Sha256', 'stoppedAtEos',
  ]) {
    if (actual[name] !== expected[name]) fail(`output ${name} does not match the oracle.`);
  }
  if (!sameArray(actual.questionTokenIds, expected.questionTokenIds) ||
      !sameArray(actual.tokenIds, expected.tokenIds)) {
    fail('output token IDs do not match the oracle.');
  }
  if (actual.tokenIds.length < reference.workload.minimumDecoderSteps) {
    fail('oracle result does not exercise multiple decoder steps.');
  }
  for (let index = 0; index < expected.routerLogits.values.length; index++) {
    const wanted = expected.routerLogits.values[index];
    const received = actual.routerLogits.values[index];
    const tolerance = expected.routerLogits.atol + expected.routerLogits.rtol * Math.abs(wanted);
    if (Math.abs(received - wanted) > tolerance) {
      fail(`router logit ${index} differs from the oracle by more than ${tolerance}.`);
    }
  }
  return Object.freeze({
    schema: reference.schema,
    provenanceKind: reference.provenance.kind,
    matched: true,
    exactTokenIds: true,
    routerWithinTolerance: true,
  });
}

function apiContracts(api) {
  if (!api?.VolvoxAI || typeof api.VolvoxAI.createRuntime !== 'function' ||
      typeof api.Graph !== 'function' || typeof api.GraphLoader?.load !== 'function' ||
      typeof api.ReadOnlySafetensorsCache !== 'function') {
    fail('api must expose VolvoxAI, Graph, GraphLoader, and ReadOnlySafetensorsCache.');
  }
}

/**
 * Own the complete Runtime -> two Models -> two Contexts lifecycle, execute the
 * fixed workload, validate strict provider evidence, and compare with an
 * independently produced ONNX Runtime oracle.
 */
export async function runTinyReceiptSplitE2E({
  api,
  backend,
  packageUrl,
  fetch: fetchImpl = globalThis.fetch?.bind(globalThis),
  wasmUrl,
  reference,
  workload = TINY_RECEIPT_SPLIT_E2E_WORKLOAD,
  sessionClass = TinyReceiptSplitSession,
  verifyPackageAssets = true,
} = {}) {
  apiContracts(api);
  const normalized = normalizedWorkload(workload);
  if (!BACKENDS.includes(backend)) fail(`backend must be one of ${BACKENDS.join(', ')}.`);
  if (packageUrl == null) fail('packageUrl is required.');
  if (typeof fetchImpl !== 'function') fail('fetch is required.');
  if (typeof verifyPackageAssets !== 'boolean') fail('verifyPackageAssets must be boolean.');
  if (reference != null) await validateReferenceIntegrity(reference);

  const diagnostics = [];
  let cache;
  let runtime;
  let session;
  let report;
  let operationError = null;
  const lifecycle = {
    sessionClosed: false,
    runtimeClosed: false,
    cacheCleared: false,
  };
  try {
    runtime = await api.VolvoxAI.createRuntime({
      backends: [backend],
      ...(wasmUrl == null ? {} : { wasmUrl }),
      onDiagnostic(event) {
        diagnostics.push(event);
      },
    });
    cache = new api.ReadOnlySafetensorsCache();
    session = await sessionClass.load({
      runtime,
      packageUrl,
      fetch: fetchImpl,
      safetensorsCache: cache,
      compileOptions: {
        backend: { mode: 'require', backend, operatorFallback: 'forbid' },
      },
      graphLoader: ({ graphUrl, weightsUrl, fetch, safetensorsCache }) =>
        api.GraphLoader.load(new api.Graph(), weightsUrl, {
          graphUrl,
          fetch,
          safetensorsCache,
        }),
    });
    const image = createTinyReceiptSplitE2EImage();
    const answer = await session.generate({
      image,
      prompt: normalized.prompt,
      family: normalized.family,
      maxNewTokens: normalized.maxNewTokens,
      preprocessed: true,
    });
    const output = await outputSummary(answer, session.vocab.itos.length);
    const provider = strictProviderSummary(
      diagnostics,
      backend,
      output.tokenIds.length,
      normalized.minimumDecoderSteps,
    );
    report = {
      schema: RESULT_SCHEMA,
      status: 'pass',
      backend,
      package: await packageIdentity(session, fetchImpl, verifyPackageAssets),
      workload: {
        ...cloneJson(normalized),
        image: {
          ...cloneJson(normalized.image),
          sha256: await sha256(float32Bytes(image)),
          summary: numericSummary(image, 'image'),
        },
      },
      output,
      provider,
      lifecycle,
      reference: null,
    };
    if (reference != null) report.reference = compareReference(report, reference);
  } catch (error) {
    operationError = error;
  }

  const cleanupErrors = [];
  if (session) {
    try {
      await session.close();
      lifecycle.sessionClosed = true;
    } catch (error) {
      cleanupErrors.push(error);
    }
  } else {
    lifecycle.sessionClosed = true;
  }
  if (runtime) {
    try {
      await runtime.close();
      lifecycle.runtimeClosed = true;
    } catch (error) {
      cleanupErrors.push(error);
    }
  } else {
    lifecycle.runtimeClosed = true;
  }
  try {
    cache?.clear();
    lifecycle.cacheCleared = true;
  } catch (error) {
    cleanupErrors.push(error);
  }
  if (operationError && cleanupErrors.length !== 0) {
    throw new AggregateError(
      [operationError, ...cleanupErrors],
      `[TinyReceiptSplitE2E] operation failed and ${cleanupErrors.length} lifecycle cleanup ` +
        `operation(s) also failed: ${operationError?.message || operationError}`,
    );
  }
  if (operationError) throw operationError;
  if (cleanupErrors.length !== 0) {
    throw new AggregateError(cleanupErrors, '[TinyReceiptSplitE2E] lifecycle cleanup failed.');
  }
  report.lifecycle = Object.freeze({ ...lifecycle });
  return Object.freeze(report);
}

/**
 * Convert one unverified candidate report into a reference skeleton. Callers
 * must supply explicit independent oracle provenance; self-recording cannot
 * masquerade as an ONNX Runtime oracle.
 */
export function createTinyReceiptSplitE2EReference(report, {
  provenance,
  routerAtol,
  routerRtol,
} = {}) {
  if (report?.schema !== RESULT_SCHEMA || report.status !== 'pass') {
    fail('a passing E2E candidate report is required.');
  }
  exactKeys(provenance, [
    'kind', 'sourceFormat', 'sourceVariant', 'provider', 'description',
  ], 'provenance');
  if (provenance.kind !== 'onnx-runtime-oracle') {
    fail("reference provenance kind must be 'onnx-runtime-oracle'.");
  }
  if (!Number.isFinite(routerAtol) || routerAtol < 0 ||
      !Number.isFinite(routerRtol) || routerRtol < 0) {
    fail('non-negative routerAtol and routerRtol are required.');
  }
  const expected = report.output;
  const result = {
    schema: REFERENCE_SCHEMA,
    provenance: cloneJson(provenance),
    package: cloneJson(report.package),
    workload: {
      id: report.workload.id,
      prompt: report.workload.prompt,
      family: report.workload.family,
      maxNewTokens: report.workload.maxNewTokens,
      minimumDecoderSteps: report.workload.minimumDecoderSteps,
      image: {
        generator: report.workload.image.generator,
        dtype: report.workload.image.dtype,
        shape: [...report.workload.image.shape],
        byteOrder: report.workload.image.byteOrder,
        sha256: report.workload.image.sha256,
      },
    },
    expected: {
      family: expected.family,
      familyId: expected.familyId,
      requestedFamily: expected.requestedFamily,
      requestedFamilyId: expected.requestedFamilyId,
      questionTokenIds: [...expected.questionTokenIds],
      questionTokenIdsSha256: expected.questionTokenIdsSha256,
      tokenIds: [...expected.tokenIds],
      tokenIdsSha256: expected.tokenIdsSha256,
      textUtf8Sha256: expected.textUtf8Sha256,
      stoppedAtEos: expected.stoppedAtEos,
      routerLogits: {
        values: [...expected.routerLogits.values],
        sha256: expected.routerLogits.sha256,
        summary: cloneJson(expected.routerLogits.summary),
        atol: routerAtol,
        rtol: routerRtol,
      },
    },
  };
  return validateReferenceShape(result);
}

export async function createTinyReceiptSplitE2EFixtureManifest(bindings, fileRecords) {
  if (!(bindings?.image instanceof Float32Array) ||
      !(bindings?.questionIds instanceof Int32Array) ||
      !(bindings?.familyIds instanceof Int32Array) ||
      !(bindings?.decoderInputIds instanceof Int32Array)) {
    fail('fixture bindings are invalid.');
  }
  const definitions = [
    ['image', bindings.image, 'float32', [1, 1, IMAGE_HEIGHT, IMAGE_WIDTH], float32Bytes],
    ['question_ids', bindings.questionIds, 'int32', [1, QUESTION_LENGTH], int32Bytes],
    ['family_ids', bindings.familyIds, 'int32', [1], int32Bytes],
    ['decoder_input_ids', bindings.decoderInputIds, 'int32', [1, DECODER_LENGTH], int32Bytes],
  ];
  const tensors = {};
  for (const [name, value, dtype, shape, encoder] of definitions) {
    const record = fileRecords?.[name];
    exactKeys(record, ['path', 'bytes', 'sha256'], `fileRecords.${name}`);
    if (typeof record.path !== 'string' || !/^[A-Za-z0-9._-]+$/.test(record.path)) {
      fail(`fileRecords.${name}.path must be one relative filename.`);
    }
    const bytes = encoder(value);
    if (record.bytes !== bytes.byteLength ||
        record.sha256 !== await sha256(bytes)) {
      fail(`fileRecords.${name} does not describe the binding bytes.`);
    }
    tensors[name] = {
      ...record,
      dtype,
      shape,
      byteOrder: 'little',
      encoding: 'raw',
    };
  }
  return Object.freeze({
    schema: FIXTURE_SCHEMA,
    workload: {
      id: WORKLOAD_ID,
      prompt: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.prompt,
      family: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.family,
      maxNewTokens: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.maxNewTokens,
      minimumDecoderSteps: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.minimumDecoderSteps,
      questionTokenIds: [...bindings.questionTokenIds],
    },
    tensors,
  });
}

export function tinyReceiptSplitE2ERawBytes(bindings) {
  return Object.freeze({
    image: float32Bytes(bindings.image),
    question_ids: int32Bytes(bindings.questionIds),
    family_ids: int32Bytes(bindings.familyIds),
    decoder_input_ids: int32Bytes(bindings.decoderInputIds),
  });
}

export {
  FIXTURE_SCHEMA as TINY_RECEIPT_SPLIT_E2E_FIXTURE_SCHEMA,
  REFERENCE_SCHEMA as TINY_RECEIPT_SPLIT_E2E_REFERENCE_SCHEMA,
  RESULT_SCHEMA as TINY_RECEIPT_SPLIT_E2E_RESULT_SCHEMA,
  validateReferenceIntegrity as validateTinyReceiptSplitE2EReference,
};
