/*
 * Browser/native-JS host session for the typed TinyReceipt split-ONNX package.
 *
 * The encoder runs once. On WASM, the decoder seeds its fixed [1,192] graph
 * and then executes one retained row at prefixLength - 1; other backends keep
 * ordinary fixed-length forwards unless the caller explicitly requires row
 * decode. Optimized packages expose in-graph QArgMax token IDs;
 * source-preserving packages expose logits and use the same first-index greedy
 * policy on the host.
 */

import {
  TinyReceiptCharVocab,
  preprocessTinyReceiptImage,
} from './TinyReceiptW8A8Session.js';

const PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-onnx-package-v1';
const BPE_VOCAB_SIZE = 1536;
const DECODE_POLICIES = new Set(['auto', 'required', 'ordinary']);
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
  if (!isRecord(value) || value.version !== 1) {
    fail('tokenizer must declare a supported version 1 contract.');
  }
  if (value.type === 'char-vocab') {
    if (value.itos_key !== 'itos') fail("char-vocab tokenizer itos_key must be 'itos'.");
    return Object.freeze({
      type: value.type,
      version: value.version,
      itosKey: 'itos',
      mergesKey: null,
      vocabSize: null,
      normalization: null,
      tokenizerHash: null,
      tokenIds: normalizeTokenIds(value.token_ids, 'tokenizer.token_ids'),
    });
  }
  if (value.type === 'byte_fallback_bpe') {
    exactRecordKeys(value, [
      'type', 'version', 'vocab_size', 'normalization', 'tokenizer_hash',
      'itos_key', 'merges_key', 'token_ids',
    ], 'tokenizer');
    if (value.vocab_size !== BPE_VOCAB_SIZE ||
        value.normalization !== 'NFC' ||
        typeof value.tokenizer_hash !== 'string' ||
        !/^[0-9a-f]{64}$/.test(value.tokenizer_hash) ||
        value.itos_key !== 'itos' || value.merges_key !== 'merges' ||
        !isRecord(value.token_ids)) {
      fail('byte_fallback_bpe tokenizer must declare vocab_size, NFC, and tokenizer_hash.');
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
  fail("tokenizer.type must be 'char-vocab' or 'byte_fallback_bpe'.");
}

function normalizeGeneration(value, tokenIds, decoderGraph) {
  if (!isRecord(value) || value.strategy !== 'greedy-autoregressive' ||
      value.decoder_input_length !== 192 || value.maximum_new_tokens !== 191 ||
      value.tie_policy !== 'first-index' ||
      value.pad_token_id !== tokenIds.pad || value.bos_token_id !== tokenIds.bos ||
      value.eos_token_id !== tokenIds.eos) {
    fail('generation must use the fixed-length first-index greedy contract.');
  }
  const outputs = decoderGraph?.outputs;
  if (isRecord(outputs) && Object.keys(outputs).length === 1 &&
      typeof outputs.logits === 'string') {
    if ((value.decoder_output != null && value.decoder_output !== 'logits') ||
        value.logits_row !== 'prefix_length_minus_one') {
      fail('logits decoder generation must select prefix_length_minus_one.');
    }
    return Object.freeze({ ...value, decoderOutput: 'logits', outputKeys: ['logits'] });
  }
  if (isRecord(outputs) && Object.keys(outputs).length === 1 &&
      typeof outputs.token_ids === 'string') {
    if (value.decoder_output !== 'token_ids' ||
        value.token_ids_row !== 'prefix_length_minus_one') {
      fail('token_ids decoder generation must select prefix_length_minus_one.');
    }
    return Object.freeze({ ...value, decoderOutput: 'token_ids', outputKeys: ['token_ids'] });
  }
  fail('decoder outputs must declare exactly logits or token_ids.');
}

function normalizeRouting(value, graphs) {
  const coreEncoder = ['image', 'question_ids'];
  const coreDecoder = ['decoder_input_ids', 'memory', 'memory_padding_mask'];
  // Hoisting the causal keep mask is independent of whether the decoder
  // publishes logits or in-graph token IDs. The manifest is authoritative.
  if (isRecord(graphs?.decoder?.inputs) &&
      Object.hasOwn(graphs.decoder.inputs, 'v4_keep')) {
    coreDecoder.push('v4_keep');
  }
  if (!isRecord(value)) fail('package manifest requires explicit routing.');
  if (value.mode === 'specialized') {
    exactRecordKeys(value, ['mode', 'family_id'], 'routing');
    const familyId = integer(value.family_id, 'routing.family_id');
    if (familyId < 0 || familyId >= FAMILY_ORDER.length) {
      fail('routing.family_id must be an ID from 0 through 7.');
    }
    exactRecordKeys(graphs?.encoder?.inputs, coreEncoder, 'graphs.encoder.inputs');
    exactRecordKeys(graphs?.decoder?.inputs, coreDecoder, 'graphs.decoder.inputs');
    return Object.freeze({
      mode: 'specialized',
      familyId,
      encoder: coreEncoder,
      decoder: coreDecoder,
    });
  }
  if (value.mode === 'runtime') {
    exactRecordKeys(value, ['mode', 'family_inputs'], 'routing');
    exactRecordKeys(value.family_inputs, ['encoder', 'decoder'], 'routing.family_inputs');
    const encoder = [...coreEncoder, 'family_ids'];
    const decoder = [...coreDecoder, 'family_ids'];
    exactRecordKeys(graphs?.encoder?.inputs, encoder, 'graphs.encoder.inputs');
    exactRecordKeys(graphs?.decoder?.inputs, decoder, 'graphs.decoder.inputs');
    for (const kind of ['encoder', 'decoder']) {
      const declared = nonEmptyString(
        value.family_inputs[kind],
        `routing.family_inputs.${kind}`,
      );
      if (declared !== graphs[kind].inputs.family_ids) {
        fail(`routing.family_inputs.${kind} must match graphs.${kind}.inputs.family_ids.`);
      }
    }
    return Object.freeze({ mode: 'runtime', familyId: null, encoder, decoder });
  }
  fail("routing.mode must be 'specialized' or 'runtime'.");
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
  if (!isRecord(value) || value.format !== PACKAGE_FORMAT) {
    fail(`package manifest must use format '${PACKAGE_FORMAT}'.`);
  }
  const tokenizer = normalizeTokenizer(value.tokenizer);
  const tokenIds = tokenizer.tokenIds;
  if (!isRecord(value.assets)) fail('package manifest requires assets.');
  const config = assetRecord(value.assets.config, 'assets.config');
  const vocab = assetRecord(value.assets.vocab, 'assets.vocab');
  const preprocessing = value.preprocessing;
  if (!isRecord(preprocessing) || preprocessing.layout !== 'NCHW' ||
      !sameShape(preprocessing.shape, [1, 1, 320, 672]) ||
      preprocessing.color !== 'grayscale') {
    fail('preprocessing must be grayscale F32 NCHW [1,1,320,672].');
  }
  const families = value.families;
  if (!isRecord(families) || families.auto_id !== -1 ||
      !sameShape(families.ordered_names, FAMILY_ORDER) ||
      !isRecord(families.name_to_id) ||
      FAMILY_ORDER.some((name, index) => families.name_to_id[name] !== index)) {
    fail('families must preserve the canonical eight-family order and AUTO=-1.');
  }
  const generation = normalizeGeneration(value.generation, tokenIds, value.graphs?.decoder);
  if (value.mask_semantics?.memory_padding_mask !== 'nonzero_means_blocked') {
    fail('memory_padding_mask must declare nonzero_means_blocked.');
  }
  if (!isRecord(value.graphs)) fail('package manifest requires encoder and decoder graphs.');
  const routing = normalizeRouting(value.routing, value.graphs);
  return Object.freeze({
    url,
    configUrl: assetUrl(url, config),
    vocabUrl: assetUrl(url, vocab),
    tokenizer,
    tokenIds,
    preprocessing: Object.freeze({
      width: 672,
      height: 320,
      shape: Object.freeze([1, 1, 320, 672]),
    }),
    generation: Object.freeze({ ...generation }),
    routing: Object.freeze({ mode: routing.mode, familyId: routing.familyId }),
    encoder: normalizeGraph(
      value.graphs.encoder,
      url,
      'encoder',
      routing.encoder,
      ['memory', 'memory_padding_mask', 'router_logits', 'selected_family_ids'],
    ),
    decoder: normalizeGraph(
      value.graphs.decoder,
      url,
      'decoder',
      routing.decoder,
      generation.outputKeys,
    ),
  });
}

function graphTensor(graph, name, label) {
  const tensor = graph?.getTensor?.(name) || graph?.tensors?.get?.(name);
  if (!tensor) fail(`${label} '${name}' is absent from its graph.`);
  return tensor;
}

function requireTensor(graph, name, { shape, dtype, input = false, output = false }, label) {
  const tensor = graphTensor(graph, name, label);
  if (tensor.dtype !== dtype || tensor.quantization != null || !sameShape(tensor.shape, shape)) {
    fail(`${label} '${name}' must be unquantized ${dtype} [${shape.join(',')}].`);
  }
  if (input && tensor.isInput !== true) fail(`${label} '${name}' must be a graph input.`);
  if (output && (!Array.isArray(graph.outputNames) || !graph.outputNames.includes(name))) {
    fail(`${label} '${name}' must be a declared graph output.`);
  }
  return tensor;
}

function requireExactGraphInterface(graph, inputNames, outputNames, label) {
  const actualInputs = [...graph.tensors.values()]
    .filter((tensor) => tensor.isInput === true)
    .map((tensor) => tensor.name);
  if (actualInputs.length !== inputNames.length ||
      inputNames.some((name) => !actualInputs.includes(name))) {
    fail(`${label} graph inputs must be exactly ${inputNames.join(', ')}.`);
  }
  if (!Array.isArray(graph.outputNames) || graph.outputNames.length !== outputNames.length ||
      outputNames.some((name) => !graph.outputNames.includes(name))) {
    fail(`${label} graph outputs must be exactly ${outputNames.join(', ')}.`);
  }
}

function validateEncoderGraph(graph, definition) {
  const inputNames = [definition.inputs.image, definition.inputs.question_ids];
  if (definition.inputs.family_ids) inputNames.push(definition.inputs.family_ids);
  requireExactGraphInterface(
    graph,
    inputNames,
    [
      definition.outputs.memory,
      definition.outputs.memory_padding_mask,
      definition.outputs.router_logits,
      definition.outputs.selected_family_ids,
    ],
    'encoder',
  );
  requireTensor(graph, definition.inputs.image,
    { shape: [1, 1, 320, 672], dtype: 'float32', input: true }, 'encoder image');
  requireTensor(graph, definition.inputs.question_ids,
    { shape: [1, 192], dtype: 'int32', input: true }, 'encoder question_ids');
  if (definition.inputs.family_ids) {
    requireTensor(graph, definition.inputs.family_ids,
      { shape: [1], dtype: 'int32', input: true }, 'encoder family_ids');
  }
  requireTensor(graph, definition.outputs.memory,
    { shape: [1, 402, 320], dtype: 'float32', output: true }, 'encoder memory');
  requireTensor(graph, definition.outputs.memory_padding_mask,
    { shape: [1, 402], dtype: 'int32', output: true }, 'encoder memory_padding_mask');
  requireTensor(graph, definition.outputs.router_logits,
    { shape: [1, 8], dtype: 'float32', output: true }, 'encoder router_logits');
  requireTensor(graph, definition.outputs.selected_family_ids,
    { shape: [1], dtype: 'int32', output: true }, 'encoder selected_family_ids');
}

function validateDecoderGraph(graph, definition, generation, vocabSize) {
  const inputNames = [
    definition.inputs.decoder_input_ids,
    definition.inputs.memory,
    definition.inputs.memory_padding_mask,
  ];
  if (definition.inputs.v4_keep) inputNames.push(definition.inputs.v4_keep);
  if (definition.inputs.family_ids) inputNames.push(definition.inputs.family_ids);
  const outputName = generation.decoderOutput === 'token_ids'
    ? definition.outputs.token_ids
    : definition.outputs.logits;
  requireExactGraphInterface(
    graph,
    inputNames,
    [outputName],
    'decoder',
  );
  requireTensor(graph, definition.inputs.decoder_input_ids,
    { shape: [1, 192], dtype: 'int32', input: true }, 'decoder decoder_input_ids');
  requireTensor(graph, definition.inputs.memory,
    { shape: [1, 402, 320], dtype: 'float32', input: true }, 'decoder memory');
  requireTensor(graph, definition.inputs.memory_padding_mask,
    { shape: [1, 402], dtype: 'int32', input: true }, 'decoder memory_padding_mask');
  if (definition.inputs.v4_keep) {
    requireTensor(graph, definition.inputs.v4_keep,
      { shape: [1, 192], dtype: 'int32', input: true }, 'decoder v4_keep');
  }
  if (definition.inputs.family_ids) {
    requireTensor(graph, definition.inputs.family_ids,
      { shape: [1], dtype: 'int32', input: true }, 'decoder family_ids');
  }
  if (generation.decoderOutput === 'token_ids') {
    requireTensor(graph, definition.outputs.token_ids,
      { shape: [1, 192], dtype: 'int32', output: true }, 'decoder token_ids');
  } else {
    requireTensor(graph, definition.outputs.logits,
      { shape: [1, 192, vocabSize], dtype: 'float32', output: true }, 'decoder logits');
  }
}

async function copyOutput(result, name, Type, expectedLength, label) {
  const value = await result.output(name).read();
  if (!(value instanceof Type) || value.length !== expectedLength) {
    fail(`${label} '${name}' returned the wrong typed length.`);
  }
  return value.slice();
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
    if (!runtime || typeof runtime.createModel !== 'function') {
      fail('load requires a Runtime handle with createModel().');
    }
    if (typeof options.graphLoader !== 'function') {
      fail('load requires a graphLoader function.');
    }
    const url = packageManifestUrl(options.packageUrl);
    const fetchImpl = options.fetch || globalThis.fetch?.bind(globalThis);
    const manifest = normalizeManifest(
      await fetchJson(fetchImpl, url.toString(), 'package_manifest.json'),
      url,
    );
    const rawVocab = await fetchJson(fetchImpl, manifest.vocabUrl, 'vocab.json');
    let vocab;
    if (manifest.tokenizer.type === 'char-vocab') {
      exactRecordKeys(rawVocab, ['itos'], 'vocab.json');
      vocab = new TinyReceiptCharVocab(rawVocab?.itos, manifest.tokenIds);
    } else {
      vocab = await TinyReceiptByteFallbackBPEVocab.fromJSON(
        rawVocab,
        manifest.tokenIds,
        manifest.tokenizer,
      );
    }
    const cache = options.safetensorsCache ?? null;
    if (cache != null && !isReadOnlySafetensorsCache(cache)) {
      fail('safetensorsCache must be a ReadOnlySafetensorsCache when provided.');
    }
    const decodePolicy = options.decodePolicy ?? 'auto';
    if (!DECODE_POLICIES.has(decodePolicy)) {
      fail("decodePolicy must be 'auto', 'required', or 'ordinary'.");
    }
    return new TinyReceiptSplitSession({
      runtime,
      manifest,
      vocab,
      graphLoader: options.graphLoader,
      fetchImpl,
      cache,
      compileOptions: options.compileOptions || {},
      decodePolicy,
    });
  }

  constructor({
    runtime,
    manifest,
    vocab,
    graphLoader,
    fetchImpl,
    cache,
    compileOptions,
    decodePolicy,
  }) {
    this.runtime = runtime;
    this.package = manifest;
    this.vocab = vocab;
    this._graphLoader = graphLoader;
    this._fetch = fetchImpl;
    this._cache = cache;
    this._compileOptions = compileOptions;
    this._decodePolicy = decodePolicy;
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
    const graph = await this._graphLoader({
      kind,
      graphUrl: definition.graphUrl,
      weightsUrl: definition.weightsUrl,
      graphAsset: definition.graph,
      weightsAsset: definition.weights,
      fetch: this._fetch,
      safetensorsCache: this._cache,
    });
    if (!graph || !graph.tensors || !Array.isArray(graph.nodes)) {
      fail(`${kind} graph loader returned an invalid graph.`);
    }
    if (kind === 'encoder') validateEncoderGraph(graph, definition);
    else validateDecoderGraph(
      graph,
      definition,
      this.package.generation,
      this.vocab.itos.length,
    );
    const model = this.runtime.createModel(graph);
    let compiled;
    let context;
    let requireRetainedRow = false;
    try {
      compiled = await model.compile(this._compileOptions);
      requireRetainedRow = kind === 'decoder' &&
        (this._decodePolicy === 'required' ||
          (this._decodePolicy === 'auto' && compiled.backend === 'wasm'));
      if (requireRetainedRow) {
        const changedInputs = [definition.inputs.decoder_input_ids];
        if (definition.inputs.v4_keep) changedInputs.push(definition.inputs.v4_keep);
        context = await compiled.createContext({
          decode: {
            changedInputs,
            rowMode: 'required',
            requireIncremental: true,
          },
        });
      } else {
        context = await compiled.createContext();
      }
    } catch (error) {
      await context?.close();
      await compiled?.close();
      await model.close();
      throw error;
    }
    const record = {
      graph,
      model,
      compiled,
      context,
      decoderExecution: kind !== 'decoder'
        ? null
        : requireRetainedRow
          ? 'context-decode-retained-row'
          : 'fixed-length-ordinary-forward',
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

  async _encoderForward({ image, prompt, family, preprocessed }) {
    const requestedFamilyId = this._familyId(family);
    if (this.package.routing.mode === 'specialized' && requestedFamilyId >= 0 &&
        requestedFamilyId !== this.package.routing.familyId) {
      fail(`specialized package is fixed to family '${
        FAMILY_ORDER[this.package.routing.familyId]}'.`);
    }
    const encoder = await this._record('encoder');
    const definition = this.package.encoder;
    const encodedQuestion = this.vocab.encodeQuestion(prompt, 192);
    const questionIds = new Int32Array(192);
    questionIds.fill(this.vocab.pad);
    questionIds.set(encodedQuestion);
    const inputs = {
      [definition.inputs.image]: await this._prepareImage(image, preprocessed),
      [definition.inputs.question_ids]: questionIds,
    };
    if (definition.inputs.family_ids) {
      inputs[definition.inputs.family_ids] = Int32Array.of(requestedFamilyId);
    }
    const execution = await encoder.context.execute(inputs);
    try {
      const memory = await copyOutput(
        execution,
        definition.outputs.memory,
        Float32Array,
        402 * 320,
        'encoder memory',
      );
      const memoryPaddingMask = await copyOutput(
        execution,
        definition.outputs.memory_padding_mask,
        Int32Array,
        402,
        'encoder memory_padding_mask',
      );
      const routerLogits = await copyOutput(
        execution,
        definition.outputs.router_logits,
        Float32Array,
        8,
        'encoder router_logits',
      );
      const selected = await copyOutput(
        execution,
        definition.outputs.selected_family_ids,
        Int32Array,
        1,
        'encoder selected_family_ids',
      );
      if (memoryPaddingMask.some((value) => value !== 0 && value !== 1)) {
        fail('encoder memory_padding_mask must contain I32 0/1 blocked flags.');
      }
      const selectedFamilyId = selected[0];
      if (!Number.isInteger(selectedFamilyId) ||
          selectedFamilyId < 0 || selectedFamilyId >= FAMILY_ORDER.length) {
        fail('encoder selected_family_ids returned an invalid family.');
      }
      if (this.package.routing.mode === 'specialized' &&
          selectedFamilyId !== this.package.routing.familyId) {
        fail('encoder selected family does not match specialized package routing.');
      }
      if (this.package.routing.mode === 'runtime' && requestedFamilyId >= 0 &&
          selectedFamilyId !== requestedFamilyId) {
        fail('encoder did not preserve the explicitly requested family.');
      }
      return {
        memory,
        memoryPaddingMask,
        routerLogits,
        selectedFamilyId,
        requestedFamilyId,
        questionTokenIds: Object.freeze([...encodedQuestion]),
      };
    } finally {
      await execution.close();
    }
  }

  async _generateImpl({
    image,
    prompt,
    family = 'auto',
    maxNewTokens = null,
    preprocessed = false,
  } = {}) {
    const limit = maxNewTokens == null
      ? this.package.generation.maximum_new_tokens
      : maxNewTokens;
    if (!Number.isInteger(limit) || limit < 0 ||
        limit > this.package.generation.maximum_new_tokens) {
      fail('maxNewTokens must be an integer from 0 through 191.');
    }
    const encoded = await this._encoderForward({ image, prompt, family, preprocessed });
    const decoder = await this._record('decoder');
    const definition = this.package.decoder;
    const decoderIds = new Int32Array(192);
    decoderIds.fill(this.vocab.pad);
    decoderIds[0] = this.vocab.bos;
    const decoderKeep = definition.inputs.v4_keep ? new Int32Array(192) : null;
    if (decoderKeep) decoderKeep[0] = 1;
    const selectedFamilyIds = Int32Array.of(encoded.selectedFamilyId);
    const seedInputs = {
      [definition.inputs.decoder_input_ids]: decoderIds,
      [definition.inputs.memory]: encoded.memory,
      // The source and package both use nonzero=true=blocked. Do not invert it.
      [definition.inputs.memory_padding_mask]: encoded.memoryPaddingMask,
    };
    if (definition.inputs.v4_keep) {
      seedInputs[definition.inputs.v4_keep] = decoderKeep;
    }
    if (definition.inputs.family_ids) {
      seedInputs[definition.inputs.family_ids] = selectedFamilyIds;
    }
    const retainedRow = decoder.decoderExecution === 'context-decode-retained-row';
    const tokenIds = [];
    let prefixLength = 1;
    let stoppedAtEos = false;
    let decoderSeedExecutions = 0;
    let decoderRowExecutions = 0;
    let decoderOrdinaryExecutions = 0;
    if (retainedRow) await decoder.context.decode.reset();
    for (let generated = 0; generated < limit; generated++) {
      const row = prefixLength - 1;
      let execution;
      if (!retainedRow) {
        execution = await decoder.context.execute(seedInputs);
        decoderOrdinaryExecutions++;
      } else if (row === 0) {
        execution = await decoder.context.decode.seed(seedInputs);
        decoderSeedExecutions++;
      } else {
        execution = await decoder.context.decode.step({
          [definition.inputs.decoder_input_ids]: decoderIds,
          ...(definition.inputs.v4_keep
            ? { [definition.inputs.v4_keep]: decoderKeep }
            : {}),
        }, { position: row });
        decoderRowExecutions++;
      }
      let next;
      try {
        if (this.package.generation.decoderOutput === 'token_ids') {
          const selected = await copyOutput(
            execution,
            definition.outputs.token_ids,
            Int32Array,
            192,
            'decoder token_ids',
          );
          next = selected[row];
        } else {
          const width = this.vocab.itos.length;
          const logits = await copyOutput(
            execution,
            definition.outputs.logits,
            Float32Array,
            192 * width,
            'decoder logits',
          );
          next = firstIndexArgmax(logits, row, width);
        }
      } finally {
        await execution.close();
      }
      if (!Number.isInteger(next) || next < 0 || next >= this.vocab.itos.length) {
        fail(`decoder returned invalid vocabulary index ${next}.`);
      }
      tokenIds.push(next);
      if (next === this.vocab.eos) {
        stoppedAtEos = true;
        break;
      }
      decoderIds[prefixLength] = next;
      if (decoderKeep) decoderKeep[prefixLength] = 1;
      prefixLength++;
    }
    return Object.freeze({
      family: FAMILY_ORDER[encoded.selectedFamilyId],
      familyId: encoded.selectedFamilyId,
      requestedFamily: encoded.requestedFamilyId < 0
        ? 'auto'
        : FAMILY_ORDER[encoded.requestedFamilyId],
      requestedFamilyId: encoded.requestedFamilyId,
      questionTokenIds: encoded.questionTokenIds,
      tokenIds: Object.freeze([...tokenIds]),
      text: this.vocab.decode(tokenIds),
      stoppedAtEos,
      execution: decoder.decoderExecution,
      decodeMode: retainedRow ? 'incremental-row-required' : 'ordinary-forward',
      decoderSeedExecutions,
      decoderRowExecutions,
      decoderOrdinaryExecutions,
      routerLogits: encoded.routerLogits.slice(),
    });
  }

  generate(options = {}) {
    return this._exclusive(() => this._generateImpl(options));
  }

  close() {
    if (this._closePromise) return this._closePromise;
    this._closePromise = this._exclusive(async () => {
      this._closed = true;
      const records = [...this._records.values()];
      for (const { context } of records) await context.close();
      for (const { compiled } of records) await compiled.close();
      for (const { model } of records) await model.close();
      this._records.clear();
    });
    return this._closePromise;
  }
}

export {
  FAMILY_ORDER as TINY_RECEIPT_SPLIT_FAMILY_ORDER,
  PACKAGE_FORMAT as TINY_RECEIPT_SPLIT_PACKAGE_FORMAT,
};
