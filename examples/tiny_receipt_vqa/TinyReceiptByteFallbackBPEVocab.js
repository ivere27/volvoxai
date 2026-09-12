const BPE_VOCAB_SIZE = 1536;
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
  throw new Error(`[TinyReceiptByteFallbackBPEVocab] ${message}`);
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function exactRecordKeys(value, keys, label) {
  if (!isRecord(value) ||
      Object.keys(value).sort().join('\0') !== [...keys].sort().join('\0')) {
    fail(`${label} must contain exactly ${keys.join(', ')}.`);
  }
  return value;
}

function integer(value, label) {
  if (!Number.isInteger(value)) fail(`${label} must be an integer.`);
  return value;
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function bytesToHex(bytes) {
  let result = '';
  for (const byte of bytes) result += byte.toString(16).padStart(2, '0');
  return result;
}

function canonicalJson(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
}

async function tokenizerFingerprint(value) {
  const unhashed = Object.freeze({
    type: value.type,
    version: value.version,
    vocab_size: value.vocab_size,
    itos: value.itos,
    merges: value.merges,
    normalization: value.normalization,
    atomic_tokens: value.atomic_tokens,
    byte_tokens: value.byte_tokens,
    unused_tokens: value.unused_tokens,
    special_tokens: value.special_tokens,
  });
  const bytes = new TextEncoder().encode(canonicalJson(unhashed));
  const digest = await globalThis.crypto?.subtle?.digest('SHA-256', bytes);
  if (!digest) fail('Web Crypto SHA-256 is required to verify the tokenizer.');
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
      fail('tokenizer input must contain valid Unicode scalar values.');
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
    if (!sameArray(atomic, DEFAULT_ATOMIC_TOKENS)) {
      fail('BPE atomic_tokens must contain the eight structural tokens followed by digits 0-9.');
    }
    if (!sameArray(bytes, DEFAULT_BYTE_TOKENS)) {
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
      ['pad', this.pad, '<pad>'],
      ['bos', this.bos, '<bos>'],
      ['eos', this.eos, '<eos>'],
      ['unk', this.unk, '<unk>'],
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
