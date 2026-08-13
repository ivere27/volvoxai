/*
 * Browser/native-JS host session for the typed TinyReceipt explicit-KV package.
 *
 * The encoder binds the exact active question extent Q and produces M=Q+210
 * memory rows. The decoder executes one token at a time with explicit
 * cross-attention and per-layer self-attention K/V tensors, starting from a
 * blocked, all-zero P=1 sentinel because Volvox graphs require positive
 * dynamic extents. It exposes current-token logits and uses first-index greedy
 * selection on the host.
 */

import { preprocessTinyReceiptImage } from './TinyReceiptInput.js';

const KV_PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-kv-onnx-package-v1';
const BPE_VOCAB_SIZE = 1536;
const IMAGE_TOKENS = 210;
const MAX_Q = 192;
const MAX_T = 192;
const CACHE_LAYERS = 4;
const CACHE_HEADS = 8;
const CACHE_HEAD_WIDTH = 40;
const CACHE_SEED_FORMAT = 'masked-zero-sentinel-v1';
const KV_TRANSFER_MODES = Object.freeze([
  'host-validated',
  'device-resident',
  'device-qualified',
]);
const CACHE_INPUT_KEYS = Object.freeze([
  ...[...Array(CACHE_LAYERS).keys()].flatMap((layer) => [
    `cross_k_${layer}`, `cross_v_${layer}`,
  ]),
  ...[...Array(CACHE_LAYERS).keys()].flatMap((layer) => [
    `past_k_${layer}`, `past_v_${layer}`,
  ]),
]);
const CACHE_OUTPUT_KEYS = Object.freeze([
  ...[...Array(CACHE_LAYERS).keys()].flatMap((layer) => [
    `present_k_${layer}`, `present_v_${layer}`,
  ]),
]);
const FAMILY_ORDER = Object.freeze([
  'phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other',
]);
const SPECIAL_TOKENS = Object.freeze(['<pad>', '<bos>', '<eos>', '<unk>']);
const STRUCTURAL_TOKENS = Object.freeze([
  '<field>', '</field>', '<value>', '</value>', '<op>', '</op>', '<answer>', '</answer>',
]);
const DIGIT_TOKENS = Object.freeze([...Array(10).keys()].map(String));
const DEFAULT_ATOMIC_TOKENS = Object.freeze([...STRUCTURAL_TOKENS, ...DIGIT_TOKENS]);
const DEFAULT_BYTE_TOKENS = Object.freeze(
  [...Array(256).keys()].map((value) => `<0x${value.toString(16).toUpperCase().padStart(2, '0')}>`),
);
const WHITESPACE_CODEPOINTS = new Set([
  0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0, 0x1680,
  0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
  ...[...Array(11).keys()].map((offset) => 0x2000 + offset),
]);
const CLEAN_TEXT_WHITESPACE_CODEPOINTS = new Set([
  ...WHITESPACE_CODEPOINTS,
  0x001C, 0x001D, 0x001E, 0x001F,
]);

function fail(message) {
  throw new Error(`[TinyReceiptSplitSession] ${message}`);
}

async function closeResources(resources) {
  const errors = [];
  for (const resource of resources) {
    if (!resource || typeof resource.close !== 'function') continue;
    try {
      await resource.close();
    } catch (error) {
      errors.push(error);
    }
  }
  return errors;
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function nonEmptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} must be a non-empty string.`);
  return value;
}

function integer(value, label) {
  if (!Number.isInteger(value)) fail(`${label} must be an integer.`);
  return value;
}

function dimension(value, name, minimum, maximum, label) {
  if (!isRecord(value) || value.min !== minimum || value.max !== maximum ||
      (value.multiple_of != null && value.multiple_of !== 1)) {
    fail(`${label}.${name} must be bounded to [${minimum},${maximum}].`);
  }
  return Object.freeze({ min: minimum, max: maximum });
}

function normalizeKVShapeContract(value) {
  exactRecordKeys(value, [
    'graph_shape_mode', 'dimensions', 'fixed_geometry', 'relations', 'semantic_inputs',
  ], 'shape_contract');
  if (!isRecord(value) || value.graph_shape_mode !== 'bounded-explicit-kv-v1') {
    fail('KV shape_contract must use bounded-explicit-kv-v1.');
  }
  exactRecordKeys(value.dimensions, ['B', 'Q', 'M', 'P', 'R'], 'shape_contract.dimensions');
  const dimensions = Object.freeze({
    B: dimension(value.dimensions.B, 'B', 1, 1, 'shape_contract.dimensions'),
    Q: dimension(value.dimensions.Q, 'Q', 1, MAX_Q, 'shape_contract.dimensions'),
    M: dimension(
      value.dimensions.M,
      'M',
      IMAGE_TOKENS + 1,
      IMAGE_TOKENS + MAX_Q,
      'shape_contract.dimensions',
    ),
    // Volvox bounded shapes stay positive. P=1 is the manifest-declared
    // masked sentinel used to adapt the producer's empty P=0 seed.
    P: dimension(value.dimensions.P, 'P', 1, MAX_T - 1, 'shape_contract.dimensions'),
    R: dimension(value.dimensions.R, 'R', 2, MAX_T, 'shape_contract.dimensions'),
  });
  if (!isRecord(value.fixed_geometry) ||
      !sameShape(value.fixed_geometry.image, [1, 1, 320, 672]) ||
      value.fixed_geometry.image_tokens !== IMAGE_TOKENS ||
      value.fixed_geometry.feature_width !== 320 ||
      value.fixed_geometry.attention_heads !== CACHE_HEADS ||
      value.fixed_geometry.attention_head_width !== CACHE_HEAD_WIDTH ||
      value.fixed_geometry.decoder_layers !== CACHE_LAYERS ||
      value.fixed_geometry.adapter_families !== 8) {
    fail('KV shape_contract.fixed_geometry is not the TinyReceipt v1 geometry.');
  }
  if (!isRecord(value.relations?.encoder_memory) ||
      value.relations.encoder_memory.operator !== 'Concat' ||
      value.relations.encoder_memory.fixed_image_tokens !== IMAGE_TOKENS ||
      value.relations.encoder_memory.dynamic_question_dimension !== 'Q' ||
      value.relations.encoder_memory.derived_memory_dimension !== 'M' ||
      !isRecord(value.relations?.present_cache) ||
      value.relations.present_cache.operator !== 'Concat' ||
      value.relations.present_cache.past_dimension !== 'P' ||
      value.relations.present_cache.fixed_current_tokens !== 1 ||
      value.relations.present_cache.derived_present_dimension !== 'R') {
    fail('KV shape_contract must prove M=Q+210 and R=P+1 with canonical Concat witnesses.');
  }
  const semanticInputs = value.semantic_inputs;
  exactRecordKeys(
    semanticInputs,
    ['question_position_ids'],
    'shape_contract.semantic_inputs',
  );
  if (!sameShape(semanticInputs.question_position_ids?.shape, ['B', 'Q']) ||
      semanticInputs.question_position_ids?.values !== 'zero_based_contiguous') {
    fail('KV question_position_ids must use the canonical [B,Q] ABI.');
  }
  return Object.freeze({ dimensions });
}

function normalizeKVCacheContract(value, maskSemantics) {
  exactRecordKeys(value, [
    'format', 'layers', 'heads', 'head_width', 'past_dimension', 'present_dimension',
    'initial_past_length', 'sentinel_mask_value', 'cache_dtype',
  ], 'cache_contract');
  if (value.format !== CACHE_SEED_FORMAT || value.layers !== CACHE_LAYERS ||
      value.heads !== CACHE_HEADS || value.head_width !== CACHE_HEAD_WIDTH ||
      value.past_dimension !== 'P' || value.present_dimension !== 'R' ||
      value.initial_past_length !== 1 || value.cache_dtype !== 'float32' ||
      !Number.isInteger(value.sentinel_mask_value) ||
      ![0, 1].includes(value.sentinel_mask_value)) {
    fail('cache_contract is not the qualified positive-shape sentinel ABI.');
  }
  const expected = 1;
  if (maskSemantics !== 'nonzero_means_blocked' ||
      value.sentinel_mask_value !== expected) {
    fail('cache_contract sentinel value does not match past_padding_mask semantics.');
  }
  return Object.freeze({
    mode: 'explicit-kv',
    format: value.format,
    layers: CACHE_LAYERS,
    heads: CACHE_HEADS,
    headWidth: CACHE_HEAD_WIDTH,
    initialPastLength: 1,
    sentinelMaskValue: expected,
  });
}

function shaped(data, shape, label) {
  if (!ArrayBuffer.isView(data) || data instanceof DataView) {
    fail(`${label} data must be a runtime typed array.`);
  }
  let elements = 1;
  for (const [axis, extent] of shape.entries()) {
    if (!Number.isSafeInteger(extent) || extent <= 0 ||
        !Number.isSafeInteger(elements * extent)) {
      fail(`${label} shape axis ${axis} is invalid.`);
    }
    elements *= extent;
  }
  if (data.length !== elements) {
    fail(`${label} data length ${data.length} does not match [${shape.join(',')}].`);
  }
  return Object.freeze({ data, shape: Object.freeze([...shape]) });
}

function deviceShaped(data, shape, label) {
  if (!isRecord(data) || data.location !== 'device' || data.dtype !== 'float32' ||
      !sameShape(data.shape, shape) || typeof data.read !== 'function') {
    fail(`${label} must be a live device F32 result with shape [${shape.join(',')}].`);
  }
  return Object.freeze({ data, shape: Object.freeze([...shape]) });
}

function monotonicMilliseconds() {
  return globalThis.performance?.now?.() ?? Date.now();
}

function normalizeKVTransferMode(value) {
  const mode = value ?? KV_TRANSFER_MODES[0];
  if (!KV_TRANSFER_MODES.includes(mode)) {
    fail(`kvTransferMode must be one of ${KV_TRANSFER_MODES.join(', ')}.`);
  }
  return mode;
}

function activePositionIds(length) {
  const result = new Int32Array(length);
  for (let index = 0; index < length; index++) result[index] = index;
  return result;
}

function assetPath(value, label) {
  const path = nonEmptyString(value, label);
  if (path.split('/').some((part) =>
    part === '.' || part === '..' || !/^[A-Za-z0-9._-]+$/.test(part))) {
    fail(`${label} must be a package-relative asset path.`);
  }
  return path;
}

function assetRecord(value, label) {
  if (!isRecord(value)) fail(`${label} must be an asset record.`);
  const path = assetPath(value.path, `${label}.path`);
  if (!Number.isInteger(value.bytes) || value.bytes <= 0) fail(`${label}.bytes must be positive.`);
  if (typeof value.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(value.sha256)) {
    fail(`${label}.sha256 must be a lowercase SHA-256 digest.`);
  }
  return Object.freeze({ path, bytes: value.bytes, sha256: value.sha256 });
}

function packageManifestUrl(value) {
  if (value instanceof URL) return value;
  const source = nonEmptyString(value, 'packageUrl');
  const base = globalThis.location?.href || import.meta.url;
  const url = new URL(source, base);
  return url.pathname.endsWith('/') ? new URL('package_manifest.json', url) : url;
}

function assetUrl(manifestUrl, record) {
  return new URL(record.path, new URL('.', manifestUrl)).toString();
}

async function fetchJson(fetchImpl, url, label) {
  if (typeof fetchImpl !== 'function') fail(`no fetch implementation can load ${label}.`);
  let response;
  try {
    response = await fetchImpl(url);
  } catch (error) {
    fail(`could not load ${label}: ${error?.message || error}`);
  }
  if (!response || response.ok === false) {
    fail(`could not load ${label}: ${response?.statusText || response?.status || 'request failed'}.`);
  }
  try {
    return await response.json();
  } catch (error) {
    fail(`could not parse ${label}: ${error?.message || error}`);
  }
}

function exactRecordKeys(value, expected, label) {
  if (!isRecord(value) ||
      Object.keys(value).sort().join('\0') !== [...expected].sort().join('\0')) {
    fail(`${label} must contain exactly ${[...expected].join(', ')}.`);
  }
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
}

function bytesToHex(bytes) {
  let result = '';
  for (const value of bytes) result += value.toString(16).padStart(2, '0');
  return result;
}

async function tokenizerFingerprint(value) {
  const unhashed = Object.fromEntries(
    Object.entries(value).filter(([key]) => key !== 'tokenizer_hash'),
  );
  const bytes = new TextEncoder().encode(canonicalJson(unhashed));
  const digest = await globalThis.crypto?.subtle?.digest('SHA-256', bytes);
  if (!digest) fail('Web Crypto SHA-256 is required to verify the BPE tokenizer.');
  return bytesToHex(new Uint8Array(digest));
}

function stringArray(value, label, { nonEmpty = false } = {}) {
  if (!Array.isArray(value) || (nonEmpty && value.length === 0) ||
      value.some((entry) => typeof entry !== 'string')) {
    fail(`${label} must be ${nonEmpty ? 'a non-empty' : 'an'} array of strings.`);
  }
  return value;
}

function isWhitespace(character) {
  return WHITESPACE_CODEPOINTS.has(character.codePointAt(0));
}

function strictUtf8Bytes(value) {
  for (const character of value) {
    const codepoint = character.codePointAt(0);
    if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
      fail('BPE input must contain valid Unicode scalar values.');
    }
  }
  return new TextEncoder().encode(value);
}

function cleanQuestionText(value) {
  const result = [];
  let pendingSpace = false;
  for (const character of String(value ?? '')) {
    if (CLEAN_TEXT_WHITESPACE_CODEPOINTS.has(character.codePointAt(0))) {
      pendingSpace = result.length > 0;
      continue;
    }
    if (pendingSpace) result.push(' ');
    result.push(character);
    pendingSpace = false;
  }
  return result.join('').normalize('NFC');
}

/** Dependency-free runtime for the release byte_fallback_bpe v1 contract. */
export class TinyReceiptByteFallbackBPEVocab {
  static async fromJSON(value, tokenIds, manifestTokenizer = null) {
    if (!isRecord(value) || value.type !== 'byte_fallback_bpe' || value.version !== 1) {
      fail('vocab.json must use the byte_fallback_bpe v1 contract.');
    }
    exactRecordKeys(value, [
      'type', 'version', 'vocab_size', 'itos', 'merges', 'normalization',
      'atomic_tokens', 'byte_tokens', 'unused_tokens', 'special_tokens',
      'tokenizer_hash',
    ], 'BPE vocab.json');
    if (typeof value.tokenizer_hash !== 'string' ||
        !/^[0-9a-f]{64}$/.test(value.tokenizer_hash)) {
      fail('BPE vocab.json requires a lowercase tokenizer_hash.');
    }
    const fingerprint = await tokenizerFingerprint(value);
    if (fingerprint !== value.tokenizer_hash) {
      fail('BPE vocab.json tokenizer_hash does not match its contents.');
    }
    if (manifestTokenizer && (
      manifestTokenizer.vocabSize !== value.vocab_size ||
      manifestTokenizer.normalization !== value.normalization ||
      manifestTokenizer.tokenizerHash !== value.tokenizer_hash
    )) {
      fail('package tokenizer contract does not match BPE vocab.json.');
    }
    return new TinyReceiptByteFallbackBPEVocab(value, tokenIds);
  }

  constructor(value, tokenIds) {
    if (value.normalization !== 'NFC') {
      fail('byte_fallback_bpe v1 requires NFC normalization.');
    }
    const itos = stringArray(value.itos, 'BPE vocab.json itos', { nonEmpty: true });
    if (itos.some((token) => token.length === 0)) {
      fail('BPE vocab.json itos tokens must be non-empty strings.');
    }
    if (value.vocab_size !== BPE_VOCAB_SIZE || value.vocab_size !== itos.length) {
      fail(`BPE vocab_size and itos length must both equal ${BPE_VOCAB_SIZE}.`);
    }
    this.itos = Object.freeze([...itos]);
    this.stoi = new Map(this.itos.map((token, index) => [token, index]));
    if (this.stoi.size !== this.itos.length) fail('BPE vocab.json itos contains duplicates.');
    if (!SPECIAL_TOKENS.every((token, index) => this.itos[index] === token)) {
      fail('BPE vocabulary must begin with <pad>, <bos>, <eos>, <unk>.');
    }

    const atomic = stringArray(value.atomic_tokens, 'BPE atomic_tokens');
    const bytes = stringArray(value.byte_tokens, 'BPE byte_tokens');
    const unused = stringArray(value.unused_tokens, 'BPE unused_tokens');
    if (!sameShape(atomic, DEFAULT_ATOMIC_TOKENS)) {
      fail('BPE atomic_tokens must contain the eight structural tokens followed by digits 0-9.');
    }
    if (!sameShape(bytes, DEFAULT_BYTE_TOKENS)) {
      fail('BPE byte_tokens must contain <0x00> through <0xFF> in order.');
    }
    exactRecordKeys(value.special_tokens, ['pad', 'bos', 'eos', 'unk'], 'BPE special_tokens');
    if (value.special_tokens.pad !== '<pad>' || value.special_tokens.bos !== '<bos>' ||
        value.special_tokens.eos !== '<eos>' || value.special_tokens.unk !== '<unk>') {
      fail('BPE special_tokens must map pad/bos/eos/unk to their canonical literals.');
    }
    if (unused.some((token) => token.length === 0) || new Set(unused).size !== unused.length) {
      fail('BPE unused_tokens must contain unique non-empty strings.');
    }
    this.atomicTokens = Object.freeze([...atomic]);
    this.byteTokens = DEFAULT_BYTE_TOKENS;
    this.unusedTokens = Object.freeze([...unused]);
    this._atomicSet = new Set(this.atomicTokens);
    this._byteSet = new Set(this.byteTokens);
    this._unusedSet = new Set(this.unusedTokens);
    const missing = [...SPECIAL_TOKENS, ...this.atomicTokens, ...this.byteTokens]
      .filter((token) => !this.stoi.has(token));
    if (missing.length !== 0) fail(`BPE vocabulary is missing required token '${missing[0]}'.`);
    if (this.unusedTokens.some((token) => !this.stoi.has(token))) {
      fail('BPE unused_tokens contains a token absent from itos.');
    }

    if (!Array.isArray(value.merges)) fail('BPE merges must be an array.');
    this.merges = Object.freeze(value.merges.map((pair, index) => {
      if (!Array.isArray(pair) || pair.length !== 2 ||
          pair.some((token) => typeof token !== 'string' || token.length === 0)) {
        fail(`BPE merge ${index} must contain two non-empty strings.`);
      }
      return Object.freeze([...pair]);
    }));
    this._mergeRanks = new Map();
    for (let rank = 0; rank < this.merges.length; rank++) {
      const [left, right] = this.merges[rank];
      const key = `${left}\0${right}`;
      if (this._mergeRanks.has(key)) fail('duplicate BPE merge pairs are not allowed.');
      const merged = left + right;
      if (!this.stoi.has(left) || !this.stoi.has(right) || !this.stoi.has(merged)) {
        fail(`BPE merge ${rank} references a token absent from itos.`);
      }
      if (this._byteSet.has(left) || this._byteSet.has(right) ||
          this._atomicSet.has(left) || this._atomicSet.has(right) ||
          SPECIAL_TOKENS.includes(left) || SPECIAL_TOKENS.includes(right)) {
        fail('BPE special, atomic, and byte fallback tokens must never participate in merges.');
      }
      if ([...merged].some((character) =>
        DIGIT_TOKENS.includes(character) || isWhitespace(character))) {
        fail('BPE digits and whitespace must never participate in merges.');
      }
      this._mergeRanks.set(key, rank);
    }
    this._tagsLongestFirst = this.atomicTokens
      .map((token, index) => ({ token, index }))
      .filter(({ token }) => [...token].length > 1)
      .sort((left, right) =>
        [...right.token].length - [...left.token].length || left.index - right.index)
      .map(({ token }) => token);

    this.pad = integer(tokenIds.pad, 'tokenizer.token_ids.pad');
    this.bos = integer(tokenIds.bos, 'tokenizer.token_ids.bos');
    this.eos = integer(tokenIds.eos, 'tokenizer.token_ids.eos');
    this.unk = integer(tokenIds.unk, 'tokenizer.token_ids.unk');
    for (const [name, id, literal] of [
      ['pad', this.pad, '<pad>'], ['bos', this.bos, '<bos>'],
      ['eos', this.eos, '<eos>'], ['unk', this.unk, '<unk>'],
    ]) {
      if (id < 0 || this.itos[id] !== literal) {
        fail(`tokenizer.token_ids.${name} must identify ${literal} in vocab.json.`);
      }
    }
    this.type = 'byte_fallback_bpe';
    this.version = 1;
    this.normalization = 'NFC';
    this.tokenizerHash = value.tokenizer_hash;
  }

  _fallbackTokens(character) {
    return [...strictUtf8Bytes(character)]
      .map((byte) => `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`);
  }

  _applyBPE(initial) {
    let tokens = initial;
    while (tokens.length > 1) {
      let bestPair = null;
      let bestRank = null;
      for (let index = 0; index + 1 < tokens.length; index++) {
        const pair = [tokens[index], tokens[index + 1]];
        const rank = this._mergeRanks.get(`${pair[0]}\0${pair[1]}`);
        if (rank !== undefined && (bestRank === null || rank < bestRank)) {
          bestPair = pair;
          bestRank = rank;
        }
      }
      if (bestPair === null) break;
      const merged = [];
      for (let index = 0; index < tokens.length;) {
        if (index + 1 < tokens.length && tokens[index] === bestPair[0] &&
            tokens[index + 1] === bestPair[1]) {
          merged.push(tokens[index] + tokens[index + 1]);
          index += 2;
        } else {
          merged.push(tokens[index++]);
        }
      }
      tokens = merged;
    }
    return tokens;
  }

  _encodeMergeableSpan(span) {
    const initial = [];
    for (const character of span) {
      if (this.stoi.has(character) && !this._byteSet.has(character) &&
          !this._unusedSet.has(character)) {
        initial.push(character);
      } else {
        initial.push(...this._fallbackTokens(character));
      }
    }
    return this._applyBPE(initial).map((token) => this.stoi.get(token));
  }

  encode(value, { addBos = false, addEos = false, maxLength = 0 } = {}) {
    if (!Number.isInteger(maxLength) || maxLength < 0) {
      fail('BPE maxLength must be a non-negative integer.');
    }
    const normalized = String(value).normalize('NFC');
    // TextEncoder replaces lone UTF-16 surrogates. Python's strict UTF-8 source
    // contract rejects them, so validate before any direct-token shortcut.
    strictUtf8Bytes(normalized);
    const ids = [];
    let spanStart = 0;
    let cursor = 0;
    while (cursor < normalized.length) {
      const tag = this._tagsLongestFirst.find((token) => normalized.startsWith(token, cursor));
      const character = String.fromCodePoint(normalized.codePointAt(cursor));
      const isDigit = DIGIT_TOKENS.includes(character);
      const isSpace = isWhitespace(character);
      if (tag === undefined && !isDigit && !isSpace) {
        cursor += character.length;
        continue;
      }
      if (spanStart < cursor) ids.push(...this._encodeMergeableSpan(normalized.slice(spanStart, cursor)));
      if (tag !== undefined) {
        ids.push(this.stoi.get(tag));
        cursor += tag.length;
      } else if (isDigit) {
        ids.push(this.stoi.get(character));
        cursor += character.length;
      } else {
        if (this.stoi.has(character) && !this._byteSet.has(character)) {
          ids.push(this.stoi.get(character));
        } else {
          ids.push(...this._fallbackTokens(character).map((token) => this.stoi.get(token)));
        }
        cursor += character.length;
      }
      spanStart = cursor;
    }
    if (spanStart < normalized.length) {
      ids.push(...this._encodeMergeableSpan(normalized.slice(spanStart)));
    }
    if (addBos) ids.unshift(this.bos);
    if (addEos) ids.push(this.eos);
    if (maxLength > 0 && ids.length > maxLength) {
      ids.length = maxLength;
      if (addEos) ids[maxLength - 1] = this.eos;
    }
    return ids;
  }

  encodeQuestion(value, capacity) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      fail('question capacity must be a positive integer.');
    }
    return this.encode(cleanQuestionText(value), { addEos: true, maxLength: capacity });
  }

  decode(tokenIds, { errors = 'replace' } = {}) {
    if (errors !== 'replace' && errors !== 'strict') {
      fail("BPE decode errors must be 'replace' or 'strict'.");
    }
    const output = [];
    for (const rawId of tokenIds) {
      const tokenId = Number(rawId);
      if (tokenId === this.eos) break;
      if (tokenId === this.pad || tokenId === this.bos ||
          !Number.isInteger(tokenId) || tokenId < 0 || tokenId >= this.itos.length) continue;
      const token = this.itos[tokenId];
      if (this._unusedSet.has(token)) continue;
      if (this._byteSet.has(token)) output.push(Number.parseInt(token.slice(3, 5), 16));
      else output.push(...strictUtf8Bytes(token));
    }
    return new TextDecoder('utf-8', { fatal: errors === 'strict' })
      .decode(Uint8Array.from(output));
  }
}

function normalizeTokenIds(value, label) {
  if (!isRecord(value)) fail(`${label} must be an object.`);
  const result = Object.freeze({
    pad: integer(value.pad, `${label}.pad`),
    bos: integer(value.bos, `${label}.bos`),
    eos: integer(value.eos, `${label}.eos`),
    unk: integer(value.unk, `${label}.unk`),
  });
  if (result.pad !== 0 || result.bos !== 1 || result.eos !== 2 || result.unk !== 3) {
    fail('token IDs must be pad=0, bos=1, eos=2, and unk=3.');
  }
  return result;
}

function normalizeTokenizer(value) {
  exactRecordKeys(value, [
    'type', 'version', 'vocab_size', 'normalization', 'tokenizer_hash',
    'itos_key', 'merges_key', 'token_ids',
  ], 'tokenizer');
  if (value.type !== 'byte_fallback_bpe' || value.version !== 1 ||
      value.vocab_size !== BPE_VOCAB_SIZE || value.normalization !== 'NFC' ||
      typeof value.tokenizer_hash !== 'string' ||
      !/^[0-9a-f]{64}$/.test(value.tokenizer_hash) ||
      value.itos_key !== 'itos' || value.merges_key !== 'merges' ||
      !isRecord(value.token_ids)) {
    fail('tokenizer must be the canonical byte_fallback_bpe v1 1536-token NFC contract.');
  }
  return Object.freeze({
    type: value.type,
    version: value.version,
    itosKey: 'itos',
    mergesKey: 'merges',
    vocabSize: value.vocab_size,
    normalization: value.normalization,
    tokenizerHash: value.tokenizer_hash,
    tokenIds: normalizeTokenIds(value.token_ids, 'tokenizer.token_ids'),
  });
}

function normalizeKVGeneration(value, tokenIds) {
  if (!isRecord(value) || value.strategy !== 'greedy-autoregressive-explicit-kv' ||
      value.maximum_target_length !== MAX_T ||
      value.maximum_new_tokens !== MAX_T - 1 || value.logits_row !== 'current_token' ||
      value.tie_policy !== 'first-index' ||
      value.pad_token_id !== tokenIds.pad || value.bos_token_id !== tokenIds.bos ||
      value.eos_token_id !== tokenIds.eos) {
    fail('KV generation must use one-token first-index greedy decoding.');
  }
  return Object.freeze({
    ...value,
    decoderOutput: 'logits',
    outputKeys: Object.freeze(['logits', 'present_padding_mask', ...CACHE_OUTPUT_KEYS]),
  });
}

function normalizeGraph(value, manifestUrl, kind, inputKeys, outputKeys) {
  if (!isRecord(value)) fail(`graphs.${kind} must be an object.`);
  const graph = assetRecord(value.graph, `graphs.${kind}.graph`);
  const weights = assetRecord(value.weights, `graphs.${kind}.weights`);
  const report = assetRecord(value.export_report, `graphs.${kind}.export_report`);
  exactRecordKeys(value.inputs, inputKeys, `graphs.${kind}.inputs`);
  exactRecordKeys(value.outputs, outputKeys, `graphs.${kind}.outputs`);
  const inputs = {};
  const outputs = {};
  for (const name of inputKeys) {
    inputs[name] = nonEmptyString(value.inputs[name], `graphs.${kind}.inputs.${name}`);
  }
  for (const name of outputKeys) {
    outputs[name] = nonEmptyString(value.outputs[name], `graphs.${kind}.outputs.${name}`);
  }
  if (new Set(Object.values(inputs)).size !== inputKeys.length) {
    fail(`graphs.${kind}.inputs must map every semantic input to a distinct graph tensor.`);
  }
  if (new Set(Object.values(outputs)).size !== outputKeys.length) {
    fail(`graphs.${kind}.outputs must map every semantic output to a distinct graph tensor.`);
  }
  return Object.freeze({
    graphUrl: assetUrl(manifestUrl, graph),
    weightsUrl: assetUrl(manifestUrl, weights),
    reportUrl: assetUrl(manifestUrl, report),
    graph,
    weights,
    report,
    inputs: Object.freeze(inputs),
    outputs: Object.freeze(outputs),
  });
}

function normalizeManifest(value, url) {
  if (!isRecord(value) || value.format !== KV_PACKAGE_FORMAT) {
    fail(`package manifest must use format '${KV_PACKAGE_FORMAT}'.`);
  }
  const tokenizer = normalizeTokenizer(value.tokenizer);
  const tokenIds = tokenizer.tokenIds;
  if (!isRecord(value.assets)) fail('KV package manifest requires assets.');
  if (!isRecord(value.graphs)) fail('KV package manifest requires encoder and decoder graphs.');
  const config = assetRecord(value.assets.config, 'assets.config');
  const vocab = assetRecord(value.assets.vocab, 'assets.vocab');
  const preprocessing = value.preprocessing;
  if (!isRecord(preprocessing) || preprocessing.layout !== 'NCHW' ||
      !sameShape(preprocessing.shape, [1, 1, 320, 672]) ||
      preprocessing.color !== 'grayscale') {
    fail('KV preprocessing must be grayscale F32 NCHW [1,1,320,672].');
  }
  const families = value.families;
  if (!isRecord(families) || families.auto_id !== -1 ||
      !sameShape(families.ordered_names, FAMILY_ORDER) ||
      !isRecord(families.name_to_id) ||
      FAMILY_ORDER.some((name, index) => families.name_to_id[name] !== index)) {
    fail('KV families must preserve the canonical eight-family order and AUTO=-1.');
  }
  if (!isRecord(value.routing) || value.routing.mode !== 'runtime' ||
      !isRecord(value.routing.family_inputs) ||
      value.routing.family_inputs.encoder !== value.graphs?.encoder?.inputs?.family_ids ||
      value.routing.family_inputs.decoder !== value.graphs?.decoder?.inputs?.family_ids) {
    fail('KV routing must retain one explicit family_ids input on both graphs.');
  }
  const pastMaskSemantics = value.mask_semantics?.past_padding_mask;
  if (value.mask_semantics?.memory_padding_mask !== 'nonzero_means_blocked' ||
      pastMaskSemantics !== 'nonzero_means_blocked') {
    fail('KV memory/past padding-mask semantics are not explicit.');
  }
  const generation = normalizeKVGeneration(value.generation, tokenIds);
  const shapeContract = normalizeKVShapeContract(value.shape_contract);
  const cache = normalizeKVCacheContract(value.cache_contract, pastMaskSemantics);
  const encoderInputKeys = [
    'image', 'question_ids', 'question_position_ids', 'family_ids',
  ];
  const encoderOutputKeys = [
    'memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids',
    ...CACHE_INPUT_KEYS.filter((name) => name.startsWith('cross_')),
  ];
  const decoderInputKeys = [
    'decoder_input_ids', 'position_ids', 'family_ids', 'memory_padding_mask',
    'past_padding_mask', ...CACHE_INPUT_KEYS,
  ];
  return Object.freeze({
    url,
    format: KV_PACKAGE_FORMAT,
    configUrl: assetUrl(url, config),
    vocabUrl: assetUrl(url, vocab),
    tokenizer,
    tokenIds,
    preprocessing: Object.freeze({
      width: 672,
      height: 320,
      shape: Object.freeze([1, 1, 320, 672]),
    }),
    generation,
    shapeContract,
    cache: Object.freeze({ ...cache, pastMaskSemantics }),
    routing: Object.freeze({ mode: 'runtime', familyId: null }),
    encoder: normalizeGraph(
      value.graphs.encoder,
      url,
      'encoder',
      encoderInputKeys,
      encoderOutputKeys,
    ),
    decoder: normalizeGraph(
      value.graphs.decoder,
      url,
      'decoder',
      decoderInputKeys,
      generation.outputKeys,
    ),
  });
}

function graphTensor(graph, name, label) {
  const tensor = graph?.tensors?.[name];
  if (!tensor) fail(`${label} '${name}' is absent from its graph.`);
  return tensor;
}

function requireTensor(graph, name, { shape, dtype, input = false, output = false }, label) {
  const tensor = graphTensor(graph, name, label);
  if (tensor.dtype !== dtype || tensor.quantization != null || !sameShape(tensor.shape, shape)) {
    fail(`${label} '${name}' must be unquantized ${dtype} [${shape.join(',')}].`);
  }
  if (input && tensor.kind !== 'input') fail(`${label} '${name}' must be a graph input.`);
  if (output && (!Array.isArray(graph.outputs) || !graph.outputs.includes(name))) {
    fail(`${label} '${name}' must be a declared graph output.`);
  }
  return tensor;
}

function requireExactGraphInterface(graph, inputNames, outputNames, label) {
  const actualInputs = Object.keys(graph.inputs || {});
  if (actualInputs.length !== inputNames.length ||
      inputNames.some((name) => !actualInputs.includes(name))) {
    fail(`${label} graph inputs must be exactly ${inputNames.join(', ')}.`);
  }
  if (!Array.isArray(graph.outputs) || graph.outputs.length !== outputNames.length ||
      outputNames.some((name) => !graph.outputs.includes(name))) {
    fail(`${label} graph outputs must be exactly ${outputNames.join(', ')}.`);
  }
}

function requireGraphDimensions(graph, expected, label) {
  const dimensions = graph?.dimensions;
  const wanted = {
    B: [1, 1], Q: [1, MAX_Q], M: [IMAGE_TOKENS + 1, IMAGE_TOKENS + MAX_Q],
    P: [1, MAX_T - 1], R: [2, MAX_T],
  };
  const bankDimensions = new Set(
    isRecord(graph?.banks)
      ? Object.values(graph.banks).map((bank) => (isRecord(bank) ? bank.dimension : bank))
      : [],
  );
  const requestDimensions = isRecord(dimensions)
    ? Object.keys(dimensions).filter((name) => !bankDimensions.has(name)) : [];
  if (!isRecord(dimensions) ||
      requestDimensions.sort().join('\0') !== [...expected].sort().join('\0')) {
    fail(`${label} KV graph dimensions must be exactly ${expected.join(', ')}.`);
  }
  for (const name of expected) {
    const descriptor = dimensions[name];
    if (!wanted[name] || !isRecord(descriptor) || descriptor.min !== wanted[name][0] ||
        descriptor.max !== wanted[name][1] ||
        (descriptor.multiple_of != null && descriptor.multiple_of !== 1)) {
      fail(`${label} KV graph dimension ${name} has the wrong bounds.`);
    }
  }
}

function requireEncoderConcatWitness(graph, memoryOutput) {
  const witnesses = graph.nodes.filter((node) => {
    if (node?.opType !== 'Concat' || node.params?.axis !== 1) return false;
    const inputNames = Object.values(node.inputs || {});
    const outputs = Object.values(node.outputs || {});
    if (inputNames.length !== 2 || outputs.length !== 1) return false;
    const inputShapes = inputNames.map((name) => graphTensor(graph, name, 'encoder Concat').shape);
    const output = outputs[0];
    return sameShape(inputShapes[0], ['B', 210, 320]) &&
      sameShape(inputShapes[1], ['B', 'Q', 320]) &&
      isRecord(output) && sameShape(output.shape, ['B', 'M', 320]) &&
      output.dtype === 'float32';
  });
  if (witnesses.length !== 1 || !sameShape(graphTensor(
    graph,
    memoryOutput,
    'encoder memory',
  ).shape, ['B', 'M', 320])) {
    fail('encoder graph must retain exactly one qualified M=Q+210 Concat witness.');
  }
}

function validateEncoderGraph(graph, definition) {
  requireGraphDimensions(graph, ['B', 'Q', 'M'], 'encoder');
  const inputNames = [definition.inputs.image, definition.inputs.question_ids];
  inputNames.push(definition.inputs.question_position_ids);
  inputNames.push(definition.inputs.family_ids);
  const crossKeys = CACHE_INPUT_KEYS.filter((name) => name.startsWith('cross_'));
  const outputNames = [
    definition.outputs.memory,
    definition.outputs.memory_padding_mask,
    definition.outputs.router_logits,
    definition.outputs.selected_family_ids,
    ...crossKeys.map((key) => definition.outputs[key]),
  ];
  requireExactGraphInterface(graph, inputNames, outputNames, 'encoder');
  requireTensor(graph, definition.inputs.image,
    { shape: ['B', 1, 320, 672], dtype: 'float32', input: true }, 'encoder image');
  requireTensor(graph, definition.inputs.question_ids,
    { shape: ['B', 'Q'], dtype: 'int32', input: true }, 'encoder question_ids');
  requireTensor(graph, definition.inputs.question_position_ids,
    { shape: ['B', 'Q'], dtype: 'int32', input: true }, 'encoder question_position_ids');
  requireTensor(graph, definition.inputs.family_ids,
    { shape: ['B'], dtype: 'int32', input: true }, 'encoder family_ids');
  requireTensor(graph, definition.outputs.memory,
    { shape: ['B', 'M', 320], dtype: 'float32', output: true }, 'encoder memory');
  requireTensor(graph, definition.outputs.memory_padding_mask,
    { shape: ['B', 'M'], dtype: 'int32', output: true }, 'encoder memory_padding_mask');
  requireTensor(graph, definition.outputs.router_logits,
    { shape: ['B', 8], dtype: 'float32', output: true }, 'encoder router_logits');
  requireTensor(graph, definition.outputs.selected_family_ids,
    { shape: ['B'], dtype: 'int32', output: true }, 'encoder selected_family_ids');
  for (const key of crossKeys) {
    requireTensor(graph, definition.outputs[key], {
      shape: ['B', CACHE_HEADS, 'M', CACHE_HEAD_WIDTH],
      dtype: 'float32',
      output: true,
    }, `encoder ${key}`);
  }
  requireEncoderConcatWitness(graph, definition.outputs.memory);
}

function validateDecoderGraph(graph, definition, vocabSize) {
  requireGraphDimensions(graph, ['B', 'M', 'P', 'R'], 'decoder');
  const inputKeys = [
    'decoder_input_ids', 'position_ids', 'family_ids', 'memory_padding_mask',
    'past_padding_mask', ...CACHE_INPUT_KEYS,
  ];
  const outputKeys = ['logits', 'present_padding_mask', ...CACHE_OUTPUT_KEYS];
  requireExactGraphInterface(
    graph,
    inputKeys.map((key) => definition.inputs[key]),
    outputKeys.map((key) => definition.outputs[key]),
    'decoder',
  );
  requireTensor(graph, definition.inputs.decoder_input_ids,
    { shape: ['B', 1], dtype: 'int32', input: true }, 'decoder decoder_input_ids');
  requireTensor(graph, definition.inputs.position_ids,
    { shape: ['B'], dtype: 'int32', input: true }, 'decoder position_ids');
  requireTensor(graph, definition.inputs.family_ids,
    { shape: ['B'], dtype: 'int32', input: true }, 'decoder family_ids');
  requireTensor(graph, definition.inputs.memory_padding_mask,
    { shape: ['B', 'M'], dtype: 'int32', input: true }, 'decoder memory_padding_mask');
  requireTensor(graph, definition.inputs.past_padding_mask,
    { shape: ['B', 'P'], dtype: 'int32', input: true }, 'decoder past_padding_mask');
  for (const key of CACHE_INPUT_KEYS) {
    const shape = key.startsWith('cross_')
      ? ['B', CACHE_HEADS, 'M', CACHE_HEAD_WIDTH]
      : ['B', CACHE_HEADS, 'P', CACHE_HEAD_WIDTH];
    requireTensor(graph, definition.inputs[key], {
      shape, dtype: 'float32', input: true,
    }, `decoder ${key}`);
  }
  requireTensor(graph, definition.outputs.logits,
    { shape: ['B', 1, vocabSize], dtype: 'float32', output: true }, 'decoder logits');
  requireTensor(graph, definition.outputs.present_padding_mask,
    { shape: ['B', 'R'], dtype: 'int32', output: true }, 'decoder present_padding_mask');
  for (const key of CACHE_OUTPUT_KEYS) {
    requireTensor(graph, definition.outputs[key], {
      shape: ['B', CACHE_HEADS, 'R', CACHE_HEAD_WIDTH],
      dtype: 'float32',
      output: true,
    }, `decoder ${key}`);
  }
  const producers = new Map();
  for (const node of graph.nodes || []) {
    for (const descriptor of Object.values(node?.outputs || {})) {
      if (isRecord(descriptor) && typeof descriptor.tensor === 'string') {
        producers.set(descriptor.tensor, node);
      }
    }
  }
  const witnessIds = new Set();
  for (const key of CACHE_OUTPUT_KEYS) {
    let tensorName = definition.outputs[key];
    let node = producers.get(tensorName);
    while (['Identity', 'QuantizeLinear', 'DequantizeLinear', 'Cast'].includes(node?.opType)) {
      const dataInput = node.inputs?.input;
      if (typeof dataInput !== 'string') break;
      tensorName = dataInput;
      node = producers.get(tensorName);
    }
    const inputNames = Object.values(node?.inputs || {});
    const outputs = Object.values(node?.outputs || {});
    if (node?.opType !== 'Concat' || node.params?.axis !== 2 ||
        inputNames.length !== 2 || outputs.length !== 1 ||
        !sameShape(graphTensor(graph, inputNames[0], `decoder ${key} Concat`).shape,
          ['B', CACHE_HEADS, 'P', CACHE_HEAD_WIDTH]) ||
        !sameShape(graphTensor(graph, inputNames[1], `decoder ${key} Concat`).shape,
          ['B', CACHE_HEADS, 1, CACHE_HEAD_WIDTH]) ||
        !sameShape(outputs[0]?.shape, ['B', CACHE_HEADS, 'R', CACHE_HEAD_WIDTH]) ||
        outputs[0]?.dtype !== 'float32') {
      fail(`decoder ${key} must retain a qualified R=P+1 Concat witness.`);
    }
    witnessIds.add(node.id);
  }
  if (witnessIds.size !== CACHE_OUTPUT_KEYS.length) {
    fail('decoder present caches must retain eight distinct R=P+1 Concat witnesses.');
  }
}

async function copyOutput(result, name, Type, expectedShape, label) {
  const output = result.output(name);
  if (Array.isArray(output.shape) && !sameShape(output.shape, expectedShape)) {
    fail(`${label} '${name}' returned shape [${output.shape.join(',')}], expected ` +
      `[${expectedShape.join(',')}].`);
  }
  const value = await output.read();
  const expectedLength = expectedShape.reduce((product, extent) => product * extent, 1);
  if (!(value instanceof Type) || value.length !== expectedLength) {
    fail(`${label} '${name}' returned the wrong typed length.`);
  }
  return value.slice();
}

function deviceOutput(result, name, expectedShape, label) {
  if (result?.backend !== 'webgpu') {
    fail(`${label} device-resident KV requires a WebGPU execution result.`);
  }
  const output = result.output(name);
  if (output.location !== 'device' || output.dtype !== 'float32' ||
      !sameShape(output.shape, expectedShape) || typeof output.read !== 'function') {
    fail(`${label} '${name}' is not a device-resident F32 output with shape ` +
      `[${expectedShape.join(',')}].`);
  }
  return output;
}

function firstIndexArgmax(values, row, width) {
  const start = row * width;
  let bestIndex = 0;
  let bestValue = values[start];
  if (!Number.isFinite(bestValue)) fail(`decoder logits row ${row} contains a non-finite value.`);
  for (let index = 1; index < width; index++) {
    const value = values[start + index];
    if (!Number.isFinite(value)) fail(`decoder logits row ${row} contains a non-finite value.`);
    // Strictly greater preserves the declared first-index tie policy.
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function isReadOnlySafetensorsCache(value) {
  return !!value && typeof value.load === 'function' && typeof value.clear === 'function' &&
    Number.isInteger(value.size) && value.size >= 0;
}

export class TinyReceiptSplitSession {
  static async load(options = {}) {
    const runtime = options.runtime;
    if (!runtime || typeof runtime.compile !== 'function') {
      fail('load requires a Runtime handle with compile().');
    }
    if (typeof options.snapshotLoader !== 'function') {
      fail('load requires a snapshotLoader function.');
    }
    const url = packageManifestUrl(options.packageUrl);
    const fetchImpl = options.fetch || globalThis.fetch?.bind(globalThis);
    const manifest = normalizeManifest(
      await fetchJson(fetchImpl, url.toString(), 'package_manifest.json'),
      url,
    );
    const rawVocab = await fetchJson(fetchImpl, manifest.vocabUrl, 'vocab.json');
    const vocab = await TinyReceiptByteFallbackBPEVocab.fromJSON(
      rawVocab,
      manifest.tokenIds,
      manifest.tokenizer,
    );
    const cache = options.safetensorsCache ?? null;
    if (cache != null && !isReadOnlySafetensorsCache(cache)) {
      fail('safetensorsCache must be a ReadOnlySafetensorsCache when provided.');
    }
    return new TinyReceiptSplitSession({
      runtime,
      manifest,
      vocab,
      snapshotLoader: options.snapshotLoader,
      fetchImpl,
      cache,
      compileOptions: options.compileOptions || {},
    });
  }

  constructor({
    runtime,
    manifest,
    vocab,
    snapshotLoader,
    fetchImpl,
    cache,
    compileOptions,
  }) {
    this.runtime = runtime;
    this.package = manifest;
    this.vocab = vocab;
    this._snapshotLoader = snapshotLoader;
    this._fetch = fetchImpl;
    this._cache = cache;
    this._compileOptions = compileOptions;
    this._records = new Map();
    this._tail = Promise.resolve();
    this._closed = false;
    this._closePromise = null;
  }

  _exclusive(task) {
    const previous = this._tail;
    let release;
    this._tail = new Promise((resolve) => { release = resolve; });
    return previous.catch(() => undefined).then(task).finally(release);
  }

  async _record(kind) {
    if (this._closed) fail('session is closed.');
    const cached = this._records.get(kind);
    if (cached) return cached;
    const definition = this.package[kind];
    const snapshot = await this._snapshotLoader({
      kind,
      graphUrl: definition.graphUrl,
      weightsUrl: definition.weightsUrl,
      graphAsset: definition.graph,
      weightsAsset: definition.weights,
      fetch: this._fetch,
      safetensorsCache: this._cache,
    });
    const graph = snapshot?.graph;
    if (!graph || !graph.tensors || !Array.isArray(graph.nodes)
        || !Array.isArray(snapshot.inputNames) || !Array.isArray(snapshot.outputNames)) {
      fail(`${kind} snapshot loader returned an invalid logical model snapshot.`);
    }
    if (kind === 'encoder') validateEncoderGraph(graph, definition);
    else validateDecoderGraph(graph, definition, this.vocab.itos.length);
    let compiled;
    let context;
    try {
      compiled = await this.runtime.compile(snapshot, this._compileOptions);
      context = await compiled.createContext();
    } catch (error) {
      const cleanupErrors = await closeResources([context, compiled]);
      if (cleanupErrors.length > 0) {
        throw new AggregateError(
          [error, ...cleanupErrors],
          `[TinyReceiptSplitSession] ${kind} setup and cleanup both failed.`,
        );
      }
      throw error;
    }
    const record = {
      graph,
      snapshot,
      compiled,
      context,
      decoderExecution: kind === 'decoder' ? 'explicit-kv-cache' : null,
    };
    this._records.set(kind, record);
    return record;
  }

  async preload() {
    return this._exclusive(async () => {
      await this._record('encoder');
      await this._record('decoder');
      return Object.freeze({ encoder: true, decoder: true });
    });
  }

  async _prepareImage(image, preprocessed) {
    const length = 320 * 672;
    if (preprocessed === true || image instanceof Float32Array || image?.preprocessed === true) {
      const value = image instanceof Float32Array ? image : image?.data;
      if (!(value instanceof Float32Array) || value.length !== length) {
        fail('preprocessed image must be F32 NCHW [1,1,320,672].');
      }
      return value;
    }
    const value = await preprocessTinyReceiptImage(image, this.package.preprocessing);
    if (!(value instanceof Float32Array) || value.length !== length) {
      fail('image preprocessing returned the wrong F32 NCHW length.');
    }
    // With one grayscale channel NHWC and NCHW have identical flat storage.
    return value;
  }

  _familyId(value) {
    if (value == null || value === 'auto') return -1;
    if (typeof value === 'string') {
      const result = FAMILY_ORDER.indexOf(value);
      if (result < 0) fail(`family must be 'auto' or one of ${FAMILY_ORDER.join(', ')}.`);
      return result;
    }
    if (!Number.isInteger(value) || value < 0 || value >= FAMILY_ORDER.length) {
      fail('numeric family must be an ID from 0 through 7.');
    }
    return value;
  }

  async _encoderForward({
    image,
    prompt,
    family,
    preprocessed,
    paddedMaximum,
    kvTransferMode,
  }) {
    const requestedFamilyId = this._familyId(family);
    const encoder = await this._record('encoder');
    const definition = this.package.encoder;
    const encodedQuestion = this.vocab.encodeQuestion(prompt, MAX_Q);
    const questionLength = encodedQuestion.length;
    const boundQuestionLength = paddedMaximum ? MAX_Q : questionLength;
    const memoryLength = boundQuestionLength + IMAGE_TOKENS;
    const questionIds = new Int32Array(boundQuestionLength);
    questionIds.fill(this.vocab.pad);
    questionIds.set(encodedQuestion);
    const inputs = {
      [definition.inputs.image]: shaped(
        await this._prepareImage(image, preprocessed),
        [1, 1, 320, 672],
        'encoder image',
      ),
      [definition.inputs.question_ids]: shaped(
        questionIds,
        [1, boundQuestionLength],
        'encoder question_ids',
      ),
      [definition.inputs.question_position_ids]: shaped(
        activePositionIds(boundQuestionLength),
        [1, boundQuestionLength],
        'encoder question_position_ids',
      ),
      [definition.inputs.family_ids]: shaped(
        Int32Array.of(requestedFamilyId),
        [1],
        'encoder family_ids',
      ),
    };
    const deviceKV = kvTransferMode !== 'host-validated';
    const qualifyDeviceKV = kvTransferMode === 'device-qualified';
    const executionStarted = monotonicMilliseconds();
    const execution = await encoder.context.execute(inputs);
    let keepExecution = false;
    try {
      const memory = deviceKV ? null : await copyOutput(
        execution,
        definition.outputs.memory,
        Float32Array,
        [1, memoryLength, 320],
        'encoder memory',
      );
      const [memoryPaddingMask, routerLogits, selected] = deviceKV
        ? await Promise.all([
          copyOutput(
            execution,
            definition.outputs.memory_padding_mask,
            Int32Array,
            [1, memoryLength],
            'encoder memory_padding_mask',
          ),
          copyOutput(
            execution,
            definition.outputs.router_logits,
            Float32Array,
            [1, 8],
            'encoder router_logits',
          ),
          copyOutput(
            execution,
            definition.outputs.selected_family_ids,
            Int32Array,
            [1],
            'encoder selected_family_ids',
          ),
        ])
        : [
          await copyOutput(
            execution,
            definition.outputs.memory_padding_mask,
            Int32Array,
            [1, memoryLength],
            'encoder memory_padding_mask',
          ),
          await copyOutput(
            execution,
            definition.outputs.router_logits,
            Float32Array,
            [1, 8],
            'encoder router_logits',
          ),
          await copyOutput(
            execution,
            definition.outputs.selected_family_ids,
            Int32Array,
            [1],
            'encoder selected_family_ids',
          ),
        ];
      // Device-resident execution is synchronized by only the small public
      // values required by application control flow. Capture the component
      // boundary before argmax/validation and before any optional qualification
      // readback. The host path still needs every cache on the host and records
      // its completion after those required transfers below.
      const synchronizedEncoderExecutionMs = deviceKV
        ? monotonicMilliseconds() - executionStarted
        : null;
      if (memory?.some((value) => !Number.isFinite(value))) {
        fail('encoder memory contains a non-finite value.');
      }
      if (routerLogits.some((value) => !Number.isFinite(value))) {
        fail('encoder router_logits contains a non-finite value.');
      }
      if (memoryPaddingMask.some((value) => value !== 0 && value !== 1)) {
        fail('encoder memory_padding_mask must contain I32 0/1 blocked flags.');
      }
      const selectedFamilyId = selected[0];
      if (!Number.isInteger(selectedFamilyId) ||
          selectedFamilyId < 0 || selectedFamilyId >= FAMILY_ORDER.length) {
        fail('encoder selected_family_ids returned an invalid family.');
      }
      if (requestedFamilyId >= 0 && selectedFamilyId !== requestedFamilyId) {
        fail('encoder did not preserve the explicitly requested family.');
      }
      if (qualifyDeviceKV) {
        const qualifiedMemory = await copyOutput(
          execution,
          definition.outputs.memory,
          Float32Array,
          [1, memoryLength, 320],
          'encoder memory',
        );
        if (qualifiedMemory.some((value) => !Number.isFinite(value))) {
          fail('encoder memory contains a non-finite value.');
        }
      }
      const crossCache = {};
      for (const key of CACHE_INPUT_KEYS.filter((name) => name.startsWith('cross_'))) {
        const shape = [1, CACHE_HEADS, memoryLength, CACHE_HEAD_WIDTH];
        if (deviceKV) {
          const output = deviceOutput(
            execution,
            definition.outputs[key],
            shape,
            `encoder ${key}`,
          );
          if (qualifyDeviceKV) {
            const values = await copyOutput(
              execution,
              definition.outputs[key],
              Float32Array,
              shape,
              `encoder ${key}`,
            );
            if (values.some((value) => !Number.isFinite(value))) {
              fail(`encoder ${key} contains a non-finite cache value.`);
            }
          }
          crossCache[key] = output;
        } else {
          const values = await copyOutput(
            execution,
            definition.outputs[key],
            Float32Array,
            shape,
            `encoder ${key}`,
          );
          if (values.some((value) => !Number.isFinite(value))) {
            fail(`encoder ${key} contains a non-finite cache value.`);
          }
          crossCache[key] = values;
        }
      }
      const encoderExecutionMs = synchronizedEncoderExecutionMs ??
        monotonicMilliseconds() - executionStarted;
      keepExecution = deviceKV;
      return {
        memory,
        memoryPaddingMask,
        routerLogits,
        selectedFamilyId,
        requestedFamilyId,
        questionLength,
        boundQuestionLength,
        memoryLength,
        questionTokenIds: Object.freeze([...encodedQuestion]),
        crossCache: Object.freeze(crossCache),
        cacheExecution: deviceKV ? execution : null,
        encoderExecutionMs,
        encoderMemoryReadbackValidated: qualifyDeviceKV,
      };
    } finally {
      if (!keepExecution) await execution.close();
    }
  }

  async _generateExplicitKVHost({ encoded, limit, shapeMode }) {
    const decoder = await this._record('decoder');
    const definition = this.package.decoder;
    const cache = this.package.cache;
    const selectedFamilyIds = Int32Array.of(encoded.selectedFamilyId);
    const pastKeys = CACHE_INPUT_KEYS.filter((name) => name.startsWith('past_'));
    let pastLength = cache.initialPastLength;
    let pastPaddingMask = Int32Array.of(cache.sentinelMaskValue);
    let pastCache = Object.fromEntries(pastKeys.map((key) => [
      key,
      new Float32Array(CACHE_HEADS * pastLength * CACHE_HEAD_WIDTH),
    ]));
    let currentToken = this.vocab.bos;
    const tokenIds = [];
    const decodeReports = [];
    let stoppedAtEos = false;
    for (let position = 0; position < limit; position++) {
      const inputs = {
        [definition.inputs.decoder_input_ids]: shaped(
          Int32Array.of(currentToken),
          [1, 1],
          'decoder decoder_input_ids',
        ),
        [definition.inputs.position_ids]: shaped(
          Int32Array.of(position),
          [1],
          'decoder position_ids',
        ),
        [definition.inputs.family_ids]: shaped(
          selectedFamilyIds,
          [1],
          'decoder family_ids',
        ),
        [definition.inputs.memory_padding_mask]: shaped(
          encoded.memoryPaddingMask,
          [1, encoded.memoryLength],
          'decoder memory_padding_mask',
        ),
        [definition.inputs.past_padding_mask]: shaped(
          pastPaddingMask,
          [1, pastLength],
          'decoder past_padding_mask',
        ),
      };
      for (const key of CACHE_INPUT_KEYS.filter((name) => name.startsWith('cross_'))) {
        inputs[definition.inputs[key]] = shaped(
          encoded.crossCache[key],
          [1, CACHE_HEADS, encoded.memoryLength, CACHE_HEAD_WIDTH],
          `decoder ${key}`,
        );
      }
      for (const key of pastKeys) {
        inputs[definition.inputs[key]] = shaped(
          pastCache[key],
          [1, CACHE_HEADS, pastLength, CACHE_HEAD_WIDTH],
          `decoder ${key}`,
        );
      }
      const executionStarted = monotonicMilliseconds();
      const execution = await decoder.context.execute(inputs);
      const presentLength = pastLength + 1;
      let next;
      let nextMask;
      const nextCache = {};
      try {
        const logits = await copyOutput(
          execution,
          definition.outputs.logits,
          Float32Array,
          [1, 1, this.vocab.itos.length],
          'decoder logits',
        );
        next = firstIndexArgmax(logits, 0, this.vocab.itos.length);
        nextMask = await copyOutput(
          execution,
          definition.outputs.present_padding_mask,
          Int32Array,
          [1, presentLength],
          'decoder present_padding_mask',
        );
        const expectedCurrentMask = currentToken === this.vocab.pad ? 1 : 0;
        if (nextMask.some((value) => value !== 0 && value !== 1) ||
            nextMask[0] !== cache.sentinelMaskValue ||
            nextMask[pastLength] !== expectedCurrentMask ||
            pastPaddingMask.some((value, index) => nextMask[index] !== value)) {
          fail('decoder present_padding_mask corrupted its past prefix or current token.');
        }
        for (const outputKey of CACHE_OUTPUT_KEYS) {
          const pastKey = outputKey.replace(/^present_/, 'past_');
          nextCache[pastKey] = await copyOutput(
            execution,
            definition.outputs[outputKey],
            Float32Array,
            [1, CACHE_HEADS, presentLength, CACHE_HEAD_WIDTH],
            `decoder ${outputKey}`,
          );
          const values = nextCache[pastKey];
          for (let head = 0; head < CACHE_HEADS; head++) {
            for (let past = 0; past < pastLength; past++) {
              const oldOffset = (head * pastLength + past) * CACHE_HEAD_WIDTH;
              const newOffset = (head * presentLength + past) * CACHE_HEAD_WIDTH;
              for (let width = 0; width < CACHE_HEAD_WIDTH; width++) {
                if (values[newOffset + width] !== pastCache[pastKey][oldOffset + width]) {
                  fail(`decoder ${outputKey} corrupted its persistent past-cache prefix.`);
                }
              }
            }
            const appendOffset = (head * presentLength + pastLength) * CACHE_HEAD_WIDTH;
            for (let width = 0; width < CACHE_HEAD_WIDTH; width++) {
              if (!Number.isFinite(values[appendOffset + width])) {
                fail(`decoder ${outputKey} appended a non-finite cache value.`);
              }
            }
          }
        }
        decodeReports.push(Object.freeze({
          operation: position === 0 ? 'explicit-kv-seed' : 'explicit-kv-step',
          position,
          pastLength,
          presentLength,
          sentinelMaskValue: cache.sentinelMaskValue,
          backendDecodeState: execution.report?.decodeState || null,
          wallTimeMs: monotonicMilliseconds() - executionStarted,
        }));
      } finally {
        await execution.close();
      }
      if (!Number.isInteger(next) || next < 0 || next >= this.vocab.itos.length) {
        fail(`decoder returned invalid vocabulary index ${next}.`);
      }
      tokenIds.push(next);
      pastPaddingMask = nextMask;
      pastCache = nextCache;
      pastLength = presentLength;
      if (next === this.vocab.eos) {
        stoppedAtEos = true;
        break;
      }
      currentToken = next;
    }
    const logicalTargetLength = limit + 1;
    return Object.freeze({
      family: FAMILY_ORDER[encoded.selectedFamilyId],
      familyId: encoded.selectedFamilyId,
      requestedFamily: encoded.requestedFamilyId < 0
        ? 'auto' : FAMILY_ORDER[encoded.requestedFamilyId],
      requestedFamilyId: encoded.requestedFamilyId,
      questionTokenIds: encoded.questionTokenIds,
      tokenIds: Object.freeze([...tokenIds]),
      text: this.vocab.decode(tokenIds),
      stoppedAtEos,
      execution: decoder.decoderExecution,
      decodeMode: 'explicit-kv-cache',
      decoderSeedExecutions: tokenIds.length > 0 ? 1 : 0,
      decoderOrdinaryExecutions: tokenIds.length,
      decoderCacheStepExecutions: Math.max(0, tokenIds.length - 1),
      activeShape: Object.freeze({
        B: 1,
        Q: encoded.boundQuestionLength,
        T: logicalTargetLength,
        M: encoded.memoryLength,
      }),
      logicalShape: Object.freeze({
        B: 1,
        Q: encoded.questionLength,
        T: logicalTargetLength,
        M: encoded.questionLength + IMAGE_TOKENS,
      }),
      cacheShape: Object.freeze({
        initialPastLength: cache.initialPastLength,
        finalPastLength: pastLength,
        sentinelSlots: 1,
      }),
      shapeMode,
      decodeReports: Object.freeze(decodeReports),
      routerLogits: encoded.routerLogits.slice(),
      synchronizedTiming: Object.freeze({
        completion: 'required-output-readback',
        encoderExecutionMs: encoded.encoderExecutionMs,
        decoderStepMs: Object.freeze(decodeReports.map(({ wallTimeMs }) => wallTimeMs)),
      }),
      gpuResidentKv: Object.freeze({
        enabled: false,
        mode: 'host-validated',
        crossCacheOutputs: CACHE_OUTPUT_KEYS.length,
        presentCacheOutputsPerStep: CACHE_OUTPUT_KEYS.length,
        encoderCrossCacheHandoffs: 0,
        decoderCacheHandoffs: 0,
        runtimeValidatedDeviceInputs: false,
        encoderCrossCacheReadbackValidated: true,
        cachePrefixReadbackValidated: true,
        appendedCacheReadbackValidated: true,
        cacheReadbackFree: false,
      }),
    });
  }

  async _generateExplicitKVDevice({ encoded, limit, shapeMode, qualify }) {
    if (encoded.cacheExecution?.backend !== 'webgpu') {
      fail('device-resident KV requires a live WebGPU encoder result.');
    }
    const decoder = await this._record('decoder');
    const definition = this.package.decoder;
    const cache = this.package.cache;
    const selectedFamilyIds = Int32Array.of(encoded.selectedFamilyId);
    const crossKeys = CACHE_INPUT_KEYS.filter((name) => name.startsWith('cross_'));
    const pastKeys = CACHE_INPUT_KEYS.filter((name) => name.startsWith('past_'));
    let pastLength = cache.initialPastLength;
    let pastPaddingMask = Int32Array.of(cache.sentinelMaskValue);
    let pastCache = Object.fromEntries(pastKeys.map((key) => [
      key,
      new Float32Array(CACHE_HEADS * pastLength * CACHE_HEAD_WIDTH),
    ]));
    let qualifiedPastCache = qualify ? pastCache : null;
    let previousExecution = null;
    let currentToken = this.vocab.bos;
    const tokenIds = [];
    const decodeReports = [];
    let stoppedAtEos = false;
    let encoderCrossCacheHandoffs = 0;
    let decoderCacheHandoffs = 0;
    try {
      for (let position = 0; position < limit; position++) {
        const inputs = {
          [definition.inputs.decoder_input_ids]: shaped(
            Int32Array.of(currentToken),
            [1, 1],
            'decoder decoder_input_ids',
          ),
          [definition.inputs.position_ids]: shaped(
            Int32Array.of(position),
            [1],
            'decoder position_ids',
          ),
          [definition.inputs.family_ids]: shaped(
            selectedFamilyIds,
            [1],
            'decoder family_ids',
          ),
          [definition.inputs.memory_padding_mask]: shaped(
            encoded.memoryPaddingMask,
            [1, encoded.memoryLength],
            'decoder memory_padding_mask',
          ),
          [definition.inputs.past_padding_mask]: shaped(
            pastPaddingMask,
            [1, pastLength],
            'decoder past_padding_mask',
          ),
        };
        for (const key of crossKeys) {
          inputs[definition.inputs[key]] = deviceShaped(
            encoded.crossCache[key],
            [1, CACHE_HEADS, encoded.memoryLength, CACHE_HEAD_WIDTH],
            `decoder ${key}`,
          );
          encoderCrossCacheHandoffs++;
        }
        for (const key of pastKeys) {
          const shape = [1, CACHE_HEADS, pastLength, CACHE_HEAD_WIDTH];
          inputs[definition.inputs[key]] = position === 0
            ? shaped(pastCache[key], shape, `decoder ${key}`)
            : deviceShaped(pastCache[key], shape, `decoder ${key}`);
          if (position > 0) decoderCacheHandoffs++;
        }

        const executionStarted = monotonicMilliseconds();
        const execution = await decoder.context.execute(inputs);
        const retiredExecution = previousExecution;
        previousExecution = execution;
        const presentLength = pastLength + 1;
        let logits;
        let nextMask;
        let wallTimeMs;
        try {
          [logits, nextMask] = await Promise.all([
            copyOutput(
              execution,
              definition.outputs.logits,
              Float32Array,
              [1, 1, this.vocab.itos.length],
              'decoder logits',
            ),
            copyOutput(
              execution,
              definition.outputs.present_padding_mask,
              Int32Array,
              [1, presentLength],
              'decoder present_padding_mask',
            ),
          ]);
          // Reading logits and the present mask fences this execution. Keep
          // application validation, argmax, cache-handle bookkeeping, optional
          // KV qualification, and result retirement outside the component time.
          wallTimeMs = monotonicMilliseconds() - executionStarted;
        } finally {
          // The prior result owns the self-cache copied by this execution. Its
          // storage remains live through the required-output synchronization,
          // including when either small-output read fails.
          if (retiredExecution) await retiredExecution.close();
        }
        const next = firstIndexArgmax(logits, 0, this.vocab.itos.length);
        const expectedCurrentMask = currentToken === this.vocab.pad ? 1 : 0;
        if (nextMask.some((value) => value !== 0 && value !== 1) ||
            nextMask[0] !== cache.sentinelMaskValue ||
            nextMask[pastLength] !== expectedCurrentMask ||
            pastPaddingMask.some((value, index) => nextMask[index] !== value)) {
          fail('decoder present_padding_mask corrupted its past prefix or current token.');
        }

        const nextCache = {};
        const nextQualifiedCache = qualify ? {} : null;
        for (const outputKey of CACHE_OUTPUT_KEYS) {
          const pastKey = outputKey.replace(/^present_/, 'past_');
          const shape = [1, CACHE_HEADS, presentLength, CACHE_HEAD_WIDTH];
          nextCache[pastKey] = deviceOutput(
            execution,
            definition.outputs[outputKey],
            shape,
            `decoder ${outputKey}`,
          );
          if (!qualify) continue;
          const values = await copyOutput(
            execution,
            definition.outputs[outputKey],
            Float32Array,
            shape,
            `decoder ${outputKey}`,
          );
          const qualifiedPast = qualifiedPastCache[pastKey];
          for (let head = 0; head < CACHE_HEADS; head++) {
            for (let past = 0; past < pastLength; past++) {
              const oldOffset = (head * pastLength + past) * CACHE_HEAD_WIDTH;
              const newOffset = (head * presentLength + past) * CACHE_HEAD_WIDTH;
              for (let width = 0; width < CACHE_HEAD_WIDTH; width++) {
                if (values[newOffset + width] !== qualifiedPast[oldOffset + width]) {
                  fail(`decoder ${outputKey} corrupted its persistent past-cache prefix.`);
                }
              }
            }
            const appendOffset = (head * presentLength + pastLength) * CACHE_HEAD_WIDTH;
            for (let width = 0; width < CACHE_HEAD_WIDTH; width++) {
              if (!Number.isFinite(values[appendOffset + width])) {
                fail(`decoder ${outputKey} appended a non-finite cache value.`);
              }
            }
          }
          nextQualifiedCache[pastKey] = values;
        }
        decodeReports.push(Object.freeze({
          operation: position === 0 ? 'explicit-kv-seed' : 'explicit-kv-step',
          position,
          pastLength,
          presentLength,
          sentinelMaskValue: cache.sentinelMaskValue,
          backendDecodeState: execution.report?.decodeState || null,
          wallTimeMs,
        }));
        if (!Number.isInteger(next) || next < 0 || next >= this.vocab.itos.length) {
          fail(`decoder returned invalid vocabulary index ${next}.`);
        }
        tokenIds.push(next);
        pastPaddingMask = nextMask;
        pastCache = nextCache;
        if (qualify) qualifiedPastCache = nextQualifiedCache;
        pastLength = presentLength;
        if (next === this.vocab.eos) {
          stoppedAtEos = true;
          break;
        }
        currentToken = next;
      }
    } finally {
      await previousExecution?.close();
    }

    const logicalTargetLength = limit + 1;
    return Object.freeze({
      family: FAMILY_ORDER[encoded.selectedFamilyId],
      familyId: encoded.selectedFamilyId,
      requestedFamily: encoded.requestedFamilyId < 0
        ? 'auto' : FAMILY_ORDER[encoded.requestedFamilyId],
      requestedFamilyId: encoded.requestedFamilyId,
      questionTokenIds: encoded.questionTokenIds,
      tokenIds: Object.freeze([...tokenIds]),
      text: this.vocab.decode(tokenIds),
      stoppedAtEos,
      execution: decoder.decoderExecution,
      decodeMode: 'explicit-kv-cache',
      decoderSeedExecutions: tokenIds.length > 0 ? 1 : 0,
      decoderOrdinaryExecutions: tokenIds.length,
      decoderCacheStepExecutions: Math.max(0, tokenIds.length - 1),
      activeShape: Object.freeze({
        B: 1,
        Q: encoded.boundQuestionLength,
        T: logicalTargetLength,
        M: encoded.memoryLength,
      }),
      logicalShape: Object.freeze({
        B: 1,
        Q: encoded.questionLength,
        T: logicalTargetLength,
        M: encoded.questionLength + IMAGE_TOKENS,
      }),
      cacheShape: Object.freeze({
        initialPastLength: cache.initialPastLength,
        finalPastLength: pastLength,
        sentinelSlots: 1,
      }),
      shapeMode,
      decodeReports: Object.freeze(decodeReports),
      routerLogits: encoded.routerLogits.slice(),
      synchronizedTiming: Object.freeze({
        completion: 'required-small-output-readback',
        applicationValidationIncluded: false,
        cacheQualificationReadbackIncluded: false,
        encoderExecutionMs: encoded.encoderExecutionMs,
        decoderStepMs: Object.freeze(decodeReports.map(({ wallTimeMs }) => wallTimeMs)),
      }),
      gpuResidentKv: Object.freeze({
        enabled: true,
        mode: qualify ? 'device-qualified' : 'device-resident',
        crossCacheOutputs: crossKeys.length,
        presentCacheOutputsPerStep: CACHE_OUTPUT_KEYS.length,
        encoderCrossCacheHandoffs,
        decoderCacheHandoffs,
        runtimeValidatedDeviceInputs: tokenIds.length > 0,
        encoderResultRetainedThroughDecode: true,
        decoderResultRetainedUntilSuccessorExecution: true,
        encoderMemoryReadback: qualify,
        encoderMemoryReadbackValidated: encoded.encoderMemoryReadbackValidated === true,
        encoderCrossCacheReadbackValidated: qualify,
        cachePrefixReadbackValidated: qualify,
        appendedCacheReadbackValidated: qualify,
        cacheReadbackFree: !qualify,
      }),
    });
  }

  async _generateImpl({
    image,
    prompt,
    family = 'auto',
    maxNewTokens = null,
    preprocessed = false,
    shapeMode = 'active',
    kvTransferMode = 'host-validated',
  } = {}) {
    const limit = maxNewTokens == null
      ? this.package.generation.maximum_new_tokens
      : maxNewTokens;
    if (!Number.isInteger(limit) || limit < 0 ||
        limit > this.package.generation.maximum_new_tokens) {
      fail('maxNewTokens must be an integer from 0 through 191.');
    }
    if (shapeMode !== 'active' && shapeMode !== 'maximum-padded') {
      fail("shapeMode must be 'active' or 'maximum-padded'.");
    }
    const normalizedKVTransferMode = normalizeKVTransferMode(kvTransferMode);
    const paddedMaximum = shapeMode === 'maximum-padded';
    const encoded = await this._encoderForward({
      image,
      prompt,
      family,
      preprocessed,
      paddedMaximum,
      kvTransferMode: normalizedKVTransferMode,
    });
    try {
      if (normalizedKVTransferMode === 'host-validated') {
        return await this._generateExplicitKVHost({ encoded, limit, shapeMode });
      }
      return await this._generateExplicitKVDevice({
        encoded,
        limit,
        shapeMode,
        qualify: normalizedKVTransferMode === 'device-qualified',
      });
    } finally {
      await encoded.cacheExecution?.close();
    }
  }

  generate(options = {}) {
    return this._exclusive(() => this._generateImpl(options));
  }

  close() {
    if (this._closePromise) return this._closePromise;
    this._closePromise = this._exclusive(async () => {
      this._closed = true;
      const records = [...this._records.values()];
      const errors = [];
      try {
        errors.push(...await closeResources(records.map(({ context }) => context)));
        errors.push(...await closeResources(records.map(({ compiled }) => compiled)));
      } finally {
        this._records.clear();
      }
      if (errors.length > 0) {
        throw new AggregateError(
          errors,
          '[TinyReceiptSplitSession] one or more owned resources failed to close.',
        );
      }
    });
    return this._closePromise;
  }
}

export {
  FAMILY_ORDER as TINY_RECEIPT_SPLIT_FAMILY_ORDER,
  KV_PACKAGE_FORMAT as TINY_RECEIPT_SPLIT_KV_PACKAGE_FORMAT,
};
