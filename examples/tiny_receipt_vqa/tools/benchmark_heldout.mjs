#!/usr/bin/env node
/**
 * Latency and accuracy over N real heldout questions: split vs legacy.
 *
 * A single-image measurement cannot separate a small real difference from run
 * to run variance — the split/legacy gap measured that way sat around 6%, which
 * is the same order as the spread between consecutive runs. This runs the real
 * eval set through both packages on identical inputs and reports the
 * distribution, so "6% slower" can be confirmed or dismissed.
 *
 * Both sides are treated identically:
 *   * either the same per-family specialized artifacts or one runtime-routed
 *     split package, selected by the same router decision, so neither pays a
 *     routing cost the other avoids;
 *   * the same preprocessed image and the same tokenized question;
 *   * incremental row decode on both, seeded once then stepped.
 *
 * Accuracy is reported against the annotations' ground-truth answer. The two
 * engines are also compared at two intentionally distinct levels: extracted
 * answer agreement catches decision regressions, while full structured-text
 * agreement exposes any difference in the generated rationale/OCR text.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/benchmark_heldout.mjs \
 *     --eval "$RECEIPT_VQA_DATA_ROOT/eval/heldout" --count 100 \
 *     --split build/fam --legacy build/legacy-w8a8-current
 *
 * Use `--split-package build/runtime-package` instead of `--split` when one
 * package accepts `family_ids` for every selected family.
 *
 * Accuracy runs generate the full 191-token autoregressive budget by default.
 * Shorter `--tokens` runs are useful for profiling only and must opt in with
 * `--latency-only`; they never publish a ground-truth score.
 */

import { execFileSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { Graph, GraphLoader, VolvoxAI } from '../../../ts/index.js';
import { TinyReceiptCharVocab } from '../TinyReceiptW8A8Session.js';
import { TinyReceiptByteFallbackBPEVocab } from '../TinyReceiptSplitSession.js';

const FAMILY_ORDER = ['phone', 'address', 'store', 'item_row', 'item_math',
  'item_lookup', 'math', 'other'];
const SPLIT_PACKAGE_FORMAT = 'volvoxai-tiny-receipt-vqa-split-onnx-package-v1';
const MEMORY_LENGTH = 402;
const QUESTION_LENGTH = 192;
const DECODER_LENGTH = 192;

const DECODER_INPUT_ABI = Object.freeze({
  decoder_input_ids: Object.freeze({ shape: [1, DECODER_LENGTH], dtype: 'int32' }),
  memory: Object.freeze({ shape: [1, MEMORY_LENGTH, 320], dtype: 'float32' }),
  memory_padding_mask: Object.freeze({ shape: [1, MEMORY_LENGTH], dtype: 'int32' }),
  family_ids: Object.freeze({ shape: [1], dtype: 'int32' }),
});

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const name = values[index].replace(/^--/, '');
    const value = values[index + 1];
    if (value === undefined || value.startsWith('--')) result[name] = true;
    else {
      result[name] = value;
      index++;
    }
  }
  return result;
}

export function resolveGenerationSettings(options = {}) {
  const maximumNewTokens = DECODER_LENGTH - 1;
  const maxTokens = Number(options.tokens ?? maximumNewTokens);
  const latencyOnly = options['latency-only'] === true;
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > maximumNewTokens) {
    throw new Error(`--tokens must be an integer in [1, ${maximumNewTokens}]`);
  }
  if (maxTokens < maximumNewTokens && !latencyOnly) {
    throw new Error(`--tokens ${maxTokens} can truncate the <answer> field; `
      + `use ${maximumNewTokens} tokens for accuracy or pass --latency-only`);
  }
  return Object.freeze({ maxTokens, latencyOnly });
}

export function resolveFamilySelection(value) {
  if (value === undefined) return Object.freeze(FAMILY_ORDER.map((_name, index) => index));
  if (typeof value !== 'string') throw new Error('--families must be a comma-separated list');
  const requested = value.split(',').map((name) => name.trim()).filter(Boolean);
  if (requested.length === 0 || new Set(requested).size !== requested.length) {
    throw new Error('--families must name one or more distinct families');
  }
  const indices = requested.map((name) => {
    const index = FAMILY_ORDER.indexOf(name);
    if (index < 0) throw new Error(`unknown family ${JSON.stringify(name)}`);
    return index;
  });
  return Object.freeze(indices);
}

async function loadGraph(graphPath, weightsPath) {
  const graphBytes = await readFile(graphPath);
  const document = JSON.parse(graphBytes);
  const weights = await readFile(weightsPath);
  const graph = new Graph();
  await GraphLoader.load(graph, 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch: async (url) => url === 'graph.json'
      ? { ok: true, json: async () => document }
      : {
        ok: true,
        arrayBuffer: async () => weights.buffer.slice(
          weights.byteOffset, weights.byteOffset + weights.byteLength,
        ),
      },
  });
  return {
    graph,
    document,
    identity: Object.freeze({
      graph: Object.freeze({
        bytes: graphBytes.byteLength,
        sha256: createHash('sha256').update(graphBytes).digest('hex'),
      }),
      weights: Object.freeze({
        bytes: weights.byteLength,
        sha256: createHash('sha256').update(weights).digest('hex'),
      }),
    }),
  };
}

async function fileIdentity(filename) {
  const bytes = await readFile(filename);
  return Object.freeze({
    bytes: bytes.byteLength,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  });
}

/** Pillow-equivalent decode + the package's declared preprocessing. */
function preprocess(imagePath) {
  const script = [
    'from PIL import Image',
    'import struct,sys',
    "im=Image.open(sys.argv[1]).convert('L').resize((672,320), Image.BILINEAR)",
    'sys.stdout.buffer.write(im.tobytes())',
  ].join('\n');
  const raw = execFileSync('python3', ['-c', script, imagePath],
    { maxBuffer: 64 * 1024 * 1024 });
  const pixels = new Float32Array(raw.length);
  for (let index = 0; index < raw.length; index++) {
    pixels[index] = (raw[index] / 255 - 0.5) / 0.5;
  }
  return pixels;
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function exactKeys(value, expected) {
  return isRecord(value)
    && Object.keys(value).sort().join('\0') === [...expected].sort().join('\0');
}

/** Use the deployed split session's tokenizer normalization and EOS policy. */
export function encodeQuestion(text, vocab) {
  if (!(vocab instanceof TinyReceiptCharVocab) &&
      !(vocab instanceof TinyReceiptByteFallbackBPEVocab)) {
    throw new Error('question encoder requires a supported TinyReceipt vocabulary');
  }
  const ids = new Int32Array(QUESTION_LENGTH);
  ids.fill(vocab.pad);
  ids.set(vocab.encodeQuestion(text, QUESTION_LENGTH));
  return ids;
}

/** Require byte-for-byte token-ID semantics across every compared package. */
export function requireSameVocabulary(reference, candidate, label = 'candidate vocabulary') {
  if (!isRecord(reference) || !Array.isArray(reference.itos)
      || reference.itos.length === 0
      || reference.itos.some((token) => typeof token !== 'string')) {
    throw new Error('reference vocabulary must contain a non-empty string itos array');
  }
  if (!isRecord(candidate) || !Array.isArray(candidate.itos)
      || candidate.itos.length !== reference.itos.length
      || candidate.itos.some((token, index) => token !== reference.itos[index])) {
    throw new Error(`${label} does not have identical token-ID semantics`);
  }
  const referenceBPE = reference.type === 'byte_fallback_bpe';
  const candidateBPE = candidate.type === 'byte_fallback_bpe';
  if (referenceBPE || candidateBPE) {
    if (!referenceBPE || !candidateBPE || reference.version !== 1 || candidate.version !== 1 ||
        reference.vocab_size !== candidate.vocab_size ||
        reference.normalization !== candidate.normalization ||
        !/^[0-9a-f]{64}$/.test(reference.tokenizer_hash || '') ||
        reference.tokenizer_hash !== candidate.tokenizer_hash) {
      throw new Error(`${label} does not have identical token-ID semantics`);
    }
  }
  return reference.itos;
}

export async function createBenchmarkVocabulary(value, tokenizerContract = null) {
  const tokenIds = tokenizerContract?.token_ids ?? { pad: 0, bos: 1, eos: 2, unk: 3 };
  if (value?.type === 'byte_fallback_bpe') {
    if (!exactKeys(tokenizerContract, [
      'type', 'version', 'vocab_size', 'normalization', 'tokenizer_hash',
      'itos_key', 'merges_key', 'token_ids',
    ]) || tokenizerContract.type !== 'byte_fallback_bpe' ||
        tokenizerContract.version !== 1 || tokenizerContract.itos_key !== 'itos' ||
        tokenizerContract.merges_key !== 'merges' ||
        !exactKeys(tokenizerContract.token_ids, ['pad', 'bos', 'eos', 'unk'])) {
      throw new Error('BPE benchmark package requires the exact published tokenizer contract');
    }
    return TinyReceiptByteFallbackBPEVocab.fromJSON(value, tokenIds, {
      vocabSize: tokenizerContract?.vocab_size,
      normalization: tokenizerContract?.normalization,
      tokenizerHash: tokenizerContract?.tokenizer_hash,
    });
  }
  if (!isRecord(value) || !exactKeys(value, ['itos'])) {
    throw new Error('char-vocab benchmark vocabulary must contain exactly itos');
  }
  return new TinyReceiptCharVocab(value.itos, tokenIds);
}

function percentile(sorted, fraction) {
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function report(label, samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((total, value) => total + value, 0) / samples.length;
  return `${label.padEnd(22)} n=${samples.length}  median ${percentile(sorted, 0.5).toFixed(0).padStart(5)} ms`
    + `  mean ${mean.toFixed(0).padStart(5)}  p10 ${percentile(sorted, 0.1).toFixed(0).padStart(5)}`
    + `  p90 ${percentile(sorted, 0.9).toFixed(0).padStart(5)}`;
}

/** Extract the scored field. The rationale's <value> field is not the answer. */
export function extractStructuredAnswer(text) {
  const answer = text.match(/<answer>([^<]*)/)?.[1];
  return answer === undefined ? null : answer.trim();
}

/** Keep answer agreement separate from byte-for-byte generated-text equality. */
export function compareGeneratedText(splitText, legacyText) {
  const splitAnswer = extractStructuredAnswer(splitText);
  const legacyAnswer = extractStructuredAnswer(legacyText);
  return Object.freeze({
    splitAnswer,
    legacyAnswer,
    answerAgreement: splitAnswer !== null && legacyAnswer !== null
      && splitAnswer === legacyAnswer,
    structuredTextAgreement: splitText === legacyText,
  });
}

/** Resolve the exact supported split-decoder input ABI without guessing. */
export function resolveDecoderInputs(document, decoderOutput = 'token_ids') {
  if (!['token_ids', 'logits'].includes(decoderOutput)) {
    throw new Error("decoder output kind must be 'token_ids' or 'logits'");
  }
  if (document?.format !== 'volvox-graph/v1') {
    throw new Error("decoder graph must use format 'volvox-graph/v1'");
  }
  if (!isRecord(document?.inputs)) {
    throw new Error('decoder graph must declare an input descriptor object');
  }
  const resolved = {};
  for (const [name, descriptor] of Object.entries(document.inputs)) {
    if (!isRecord(descriptor)) {
      throw new Error(`decoder input ${JSON.stringify(name)} must be a descriptor object`);
    }
    const semantic = descriptor.source_name ?? name;
    if (semantic === 'v4_keep') {
      if (!sameShape(descriptor.shape, [1, DECODER_LENGTH]) || descriptor.dtype !== 'int32') {
        throw new Error('decoder input v4_keep must be I32 [1,192]');
      }
      if (resolved.v4_keep !== undefined) {
        throw new Error('decoder input semantic "v4_keep" is declared more than once');
      }
      resolved.v4_keep = name;
      continue;
    }

    const contract = DECODER_INPUT_ABI[semantic];
    if (!contract) {
      throw new Error(
        `unknown decoder input ${JSON.stringify(name)} (source ${JSON.stringify(semantic)})`,
      );
    }
    if (resolved[semantic] !== undefined) {
      throw new Error(`decoder input semantic ${JSON.stringify(semantic)} is declared more than once`);
    }
    if (!sameShape(descriptor.shape, contract.shape) || descriptor.dtype !== contract.dtype) {
      throw new Error(
        `decoder input ${JSON.stringify(name)} must be ${contract.dtype} ${JSON.stringify(contract.shape)}`,
      );
    }
    resolved[semantic] = name;
  }

  for (const semantic of ['decoder_input_ids', 'memory', 'memory_padding_mask']) {
    if (resolved[semantic] === undefined) {
      throw new Error(`decoder graph is missing semantic input ${JSON.stringify(semantic)}`);
    }
  }
  // `v4_keep` is an independently hoistable input for either output ABI.
  // The runtime tensor name may be generated; source_name carries the stable
  // semantic used by the package manifest and fixture tooling.
  return Object.freeze(resolved);
}

/** Resolve either the optimized token-ID or source-preserving logits ABI. */
export function resolveDecoderOutput(document, manifest, vocabularySize = null) {
  if (document?.format !== 'volvox-graph/v1') {
    throw new Error("decoder graph must use format 'volvox-graph/v1'");
  }
  const common = manifest?.format === SPLIT_PACKAGE_FORMAT
    && manifest?.generation?.tie_policy === 'first-index'
    && manifest?.generation?.decoder_input_length === DECODER_LENGTH;
  const tokenIdsContract = common
    && manifest.generation.decoder_output === 'token_ids'
    && manifest.generation.token_ids_row === 'prefix_length_minus_one'
    && exactKeys(manifest?.graphs?.decoder?.outputs, ['token_ids'])
    && manifest.graphs.decoder.outputs.token_ids === 'token_ids';
  const logitsContract = common
    && (manifest.generation.decoder_output == null ||
      manifest.generation.decoder_output === 'logits')
    && manifest.generation.logits_row === 'prefix_length_minus_one'
    && exactKeys(manifest?.graphs?.decoder?.outputs, ['logits'])
    && manifest.graphs.decoder.outputs.logits === 'logits';
  if (!tokenIdsContract && !logitsContract) {
    throw new Error('split package manifest does not declare a supported decoder ABI');
  }
  const outputs = document?.outputs;
  const outputName = tokenIdsContract ? 'token_ids' : 'logits';
  if (!Array.isArray(outputs) || outputs.length !== 1 || outputs[0] !== outputName) {
    throw new Error(
      `TinyReceipt decoder must expose only '${outputName}'; got ${JSON.stringify(outputs)}`,
    );
  }
  if (!Array.isArray(document?.nodes)) {
    throw new Error('decoder graph must declare a node array');
  }
  const producers = document.nodes.filter((node) => isRecord(node?.outputs)
    && Object.values(node.outputs).includes(outputName));
  if (producers.length !== 1) {
    throw new Error(`decoder ${outputName} must have exactly one producer; got ${producers.length}`);
  }
  const producer = producers[0];
  if (logitsContract) {
    if (!Number.isInteger(vocabularySize) || vocabularySize <= 0 ||
        !Object.entries(producer.outputs).some(([port, name]) => name === 'logits'
          && producer.outputs_dtype?.[port] === 'float32'
          && sameShape(producer.outputs_shape?.[port], [1, DECODER_LENGTH, vocabularySize]))) {
      throw new Error(`decoder logits must be F32 [1,192,${vocabularySize ?? 'vocab'}]`);
    }
    return Object.freeze({
      kind: 'logits', name: 'logits', length: DECODER_LENGTH, vocabularySize,
    });
  }
  if (producer.opType !== 'QArgMax'
      || !exactKeys(producer.inputs, ['input']) || typeof producer.inputs.input !== 'string'
      || !exactKeys(producer.outputs, ['out']) || producer.outputs.out !== 'token_ids'
      || !exactKeys(producer.outputs_dtype, ['out']) || producer.outputs_dtype.out !== 'int32'
      || !exactKeys(producer.outputs_shape, ['out'])
      || !sameShape(producer.outputs_shape.out, [1, DECODER_LENGTH])
      || !exactKeys(producer.params, ['axis']) || producer.params.axis !== -1) {
    throw new Error('decoder token_ids must be produced by canonical QArgMax(axis=-1) as I32 [1,192]');
  }
  return Object.freeze({ kind: 'token_ids', name: 'token_ids', length: DECODER_LENGTH });
}

/** Read one decode decision, preserving first-index tie behavior for logits. */
export function nextDecoderToken(contract, values, step) {
  if (!Number.isSafeInteger(step) || step < 0) throw new Error('decode step must be non-negative');
  if (contract.kind === 'token_ids' && contract.name === 'token_ids'
      && contract.length === DECODER_LENGTH) {
    if (!(values instanceof Int32Array) || values.length !== DECODER_LENGTH) {
      throw new Error('decoder token_ids execution output must be I32 [1,192]');
    }
    if (step >= DECODER_LENGTH) throw new Error(`token_ids output has no decode step ${step}`);
    return Number(values[step]);
  }
  if (contract.kind === 'logits' && contract.name === 'logits'
      && contract.length === DECODER_LENGTH && Number.isInteger(contract.vocabularySize)
      && contract.vocabularySize > 0) {
    const width = contract.vocabularySize;
    if (!(values instanceof Float32Array) || values.length !== DECODER_LENGTH * width) {
      throw new Error(`decoder logits execution output must be F32 [1,192,${width}]`);
    }
    if (step >= DECODER_LENGTH) throw new Error(`logits output has no decode step ${step}`);
    const offset = step * width;
    let best = 0;
    let bestValue = values[offset];
    if (!Number.isFinite(bestValue)) throw new Error('decoder logits contain non-finite values');
    for (let index = 1; index < width; index++) {
      const value = values[offset + index];
      if (!Number.isFinite(value)) throw new Error('decoder logits contain non-finite values');
      if (value > bestValue) {
        best = index;
        bestValue = value;
      }
    }
    return best;
  }
  throw new Error('decoder output contract must be I32 token_ids or F32 logits');
}

function printOperatorProfile(label, profile) {
  if (!profile) return;
  const total = [...profile.values()].reduce((sum, entry) => sum + entry.ms, 0);
  console.log(`\n${label} operator profile (${total.toFixed(0)} ms attributed)`);
  console.log('operator              count        ms      %');
  for (const [op, { ms, count }] of [...profile.entries()].sort((a, b) => b[1].ms - a[1].ms)) {
    console.log(`${op.padEnd(21)} ${String(count).padStart(5)} ${ms.toFixed(0).padStart(9)}`
      + ` ${(100 * ms / total).toFixed(1).padStart(6)}`);
  }
}

async function main() {
  const runId = randomUUID();
  const startedAt = new Date().toISOString();
  const options = parseArguments(process.argv.slice(2));
  const receiptDataRoot = process.env.RECEIPT_VQA_DATA_ROOT;
  const evalRoot = options.eval
    ?? (receiptDataRoot ? join(receiptDataRoot, 'eval', 'heldout') : null);
  if (!evalRoot) {
    throw new Error('pass --eval or set RECEIPT_VQA_DATA_ROOT=/path/to/receipt-vqa-data');
  }
  const count = Number(options.count ?? 100);
  const splitPrefix = options.split ?? 'build/fam';
  const splitPackage = options['split-package'] == null
    ? null
    : String(options['split-package']);
  if (splitPackage && options.split != null) {
    throw new Error('pass either --split <family-prefix> or --split-package <dir>, not both');
  }
  const packageDirForFamily = (family) => splitPackage ?? `${splitPrefix}${family}`;
  const legacyDir = options.legacy ?? 'build/legacy-w8a8-current';
  const { maxTokens, latencyOnly } = resolveGenerationSettings(options);
  const selectedFamilies = resolveFamilySelection(options.families);

  const firstPackageDir = packageDirForFamily(selectedFamilies[0]);
  const vocabularyPath = join(firstPackageDir, 'vocab.json');
  const firstManifest = JSON.parse(
    await readFile(join(firstPackageDir, 'package_manifest.json'), 'utf8'),
  );
  const legacyVocabularyPath = join(legacyDir, 'vocab.json');
  const vocabulary = JSON.parse(await readFile(vocabularyPath, 'utf8'));
  const legacyVocabulary = JSON.parse(await readFile(legacyVocabularyPath, 'utf8'));
  const itos = requireSameVocabulary(vocabulary, legacyVocabulary, 'legacy package vocabulary');
  const PAD = 0, BOS = 1, EOS = 2, UNK = 3;
  if (itos[PAD] !== '<pad>' || itos[BOS] !== '<bos>'
      || itos[EOS] !== '<eos>' || itos[UNK] !== '<unk>') {
    throw new Error('package vocabulary does not declare the required PAD/BOS/EOS/UNK IDs');
  }
  const vocab = await createBenchmarkVocabulary(vocabulary, firstManifest.tokenizer);

  const requestedIds = options.ids
    ? new Set(String(options.ids).split(',').map((value) => value.trim()).filter(Boolean))
    : null;
  const names = (await readdir(join(evalRoot, 'annotations'))).sort();
  const cases = [];
  for (const name of names) {
    const record = JSON.parse(await readFile(join(evalRoot, 'annotations', name), 'utf8'));
    if (requestedIds && !requestedIds.has(String(record.id))) continue;
    cases.push({
      id: record.id,
      question: record.question,
      answer: String(record.answer ?? ''),
      image: join(evalRoot, 'images', `${record.id}.jpg`),
    });
    if (!requestedIds && cases.length >= count) break;
  }
  if (requestedIds) {
    const found = new Set(cases.map((testCase) => String(testCase.id)));
    const missing = [...requestedIds].filter((id) => !found.has(id));
    if (missing.length > 0) {
      throw new Error(`requested heldout IDs not found: ${missing.join(', ')}`);
    }
  }
  process.stderr.write(`loaded ${cases.length} cases\n`);

  const runtime = await VolvoxAI.createRuntime({
    backends: ['wasm'],
    wasmUrl: pathToFileURL(resolve('dist/0.3.0/volvoxai.wasm')),
  });
  const policy = { backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' } };

  // Router: 8 nodes over question text only, shared by both engines so neither
  // is charged for a decision the other gets for free.
  const router = await loadGraph(join(legacyDir, 'router.graph.json'),
    join(legacyDir, 'model.safetensors'));
  const routerModel = runtime.createModel(router.graph);
  const routerCompiled = await routerModel.compile(policy);
  const routerContext = await routerCompiled.createContext();

  const splitEngines = [];
  const splitEngineCache = new Map();
  const legacyEngines = [];
  const splitPackageIdentities = [];
  const legacyGraphIdentities = [];
  for (const family of selectedFamilies) {
    const packageDir = packageDirForFamily(family);
    let splitEngine = splitEngineCache.get(packageDir);
    if (!splitEngine) {
      const packageVocabulary = JSON.parse(
        await readFile(join(packageDir, 'vocab.json'), 'utf8'),
      );
      requireSameVocabulary(vocabulary, packageVocabulary, `split family ${family} vocabulary`);
      const manifest = JSON.parse(
        await readFile(join(packageDir, 'package_manifest.json'), 'utf8'),
      );
      // Besides validating the shared token IDs above, bind this package's
      // manifest tokenizer fingerprint to its own vocab.json.
      await createBenchmarkVocabulary(packageVocabulary, manifest.tokenizer);
      const encoder = await loadGraph(join(packageDir, 'encoder/graph.json'),
        join(packageDir, 'encoder/model.safetensors'));
      const decoder = await loadGraph(join(packageDir, 'decoder/graph.json'),
        join(packageDir, 'decoder/model.safetensors'));
      const decoderOutput = resolveDecoderOutput(decoder.document, manifest, itos.length);
      const decoderInputs = resolveDecoderInputs(decoder.document, decoderOutput.kind);
      const encoderModel = runtime.createModel(encoder.graph);
      const encoderCompiled = await encoderModel.compile(policy);
      const decoderModel = runtime.createModel(decoder.graph);
      const decoderCompiled = await decoderModel.compile(policy);
      const incremental = decoderOutput.kind === 'token_ids';
      const changed = [decoderInputs.decoder_input_ids];
      if (decoderInputs.v4_keep) changed.push(decoderInputs.v4_keep);
      splitEngine = {
        encoder, decoder, decoderInputs, decoderOutput, incremental, manifest,
        encoderContext: await encoderCompiled.createContext(),
        decoderContext: incremental
          ? await decoderCompiled.createContext({
            decode: { changedInputs: changed, rowMode: 'auto' },
          })
          : await decoderCompiled.createContext(),
      };
      splitEngineCache.set(packageDir, splitEngine);
    }
    splitEngines[family] = splitEngine;
    splitPackageIdentities.push(Object.freeze({
      family: FAMILY_ORDER[family],
      package_manifest: await fileIdentity(join(packageDir, 'package_manifest.json')),
      vocabulary: await fileIdentity(join(packageDir, 'vocab.json')),
      graphs: Object.freeze({
        encoder: splitEngine.encoder.identity,
        decoder: splitEngine.decoder.identity,
      }),
      calibration_provenance: splitEngine.manifest.calibration_provenance ?? null,
      precision_policy: splitEngine.manifest?.variant?.precision_policy ?? null,
    }));

    const legacyGraph = await loadGraph(
      join(legacyDir, `explicit_family_${FAMILY_ORDER[family]}.graph.json`),
      join(legacyDir, 'model.safetensors'));
    const legacyModel = runtime.createModel(legacyGraph.graph);
    const legacyCompiled = await legacyModel.compile(policy);
    legacyEngines[family] = {
      graph: legacyGraph,
      context: await legacyCompiled.createContext({
        decode: { changedInputs: ['y_ids', 'y_keep'], rowMode: 'auto' },
      }),
    };
    legacyGraphIdentities.push(Object.freeze({
      family: FAMILY_ORDER[family],
      graph: legacyGraph.identity.graph,
    }));
  }
  process.stderr.write(`compiled ${splitEngineCache.size} split package(s) + `
    + `${selectedFamilies.length} legacy family engines\n`);

  const splitTimes = [];
  const legacyTimes = [];
  const splitProfile = options.profile ? new Map() : null;
  const legacyProfile = options.profile ? new Map() : null;
  globalThis.__VOLVOX_WASM_PROFILE_SHAPES = options['profile-shapes'] === true;
  let splitCorrect = 0, legacyCorrect = 0;
  let answerAgree = 0, structuredTextAgree = 0;
  const caseResults = [];

  for (const [index, testCase] of cases.entries()) {
    const image = preprocess(testCase.image);
    const questionIds = encodeQuestion(testCase.question, vocab);
    const routerKeep = Int32Array.from(questionIds, (value) => (value !== PAD ? 1 : 0));

    const routed = await routerContext.execute({ q_ids: questionIds, router_keep: routerKeep });
    const family = Number((await routed.output('router_family').read())[0]);
    await routed.close?.();
    const split = splitEngines[family];
    const legacy = legacyEngines[family];
    if (!split || !legacy) {
      throw new Error(`router selected unqualified family ${FAMILY_ORDER[family] ?? family}; `
        + 'include it in --families and provide its calibrated package');
    }

    // ---- split: encode once, then incremental decode ----
    let started = performance.now();
    globalThis.__VOLVOX_WASM_PROFILE = splitProfile ?? undefined;
    const encoderInputs = {};
    for (const [name, descriptor] of Object.entries(split.encoder.document.inputs)) {
      const source = descriptor.source_name || name;
      if (source === 'image') encoderInputs[name] = image;
      if (source === 'question_ids') encoderInputs[name] = questionIds;
      if (source === 'family_ids') encoderInputs[name] = Int32Array.of(family);
    }
    const encoded = await split.encoderContext.execute(encoderInputs);
    const memory = Float32Array.from(await encoded.output('memory').read());
    const memoryMask = Int32Array.from(await encoded.output('memory_padding_mask').read());
    await encoded.close?.();

    const decoderIds = new Int32Array(DECODER_LENGTH).fill(PAD);
    decoderIds[0] = BOS;
    const decoderKeep = new Int32Array(DECODER_LENGTH);
    decoderKeep[0] = 1;
    const decoderInputs = {
      [split.decoderInputs.decoder_input_ids]: decoderIds,
      [split.decoderInputs.memory]: memory,
      [split.decoderInputs.memory_padding_mask]: memoryMask,
    };
    if (split.decoderInputs.v4_keep) {
      decoderInputs[split.decoderInputs.v4_keep] = decoderKeep;
    }
    if (split.decoderInputs.family_ids) {
      decoderInputs[split.decoderInputs.family_ids] = Int32Array.of(family);
    }
    if (split.incremental) await split.decoderContext.decode.reset();
    const splitTokens = [];
    for (let step = 0; step < maxTokens; step++) {
      const execution = split.incremental
        ? (step === 0
          ? await split.decoderContext.decode.seed(decoderInputs)
          : await split.decoderContext.decode.step(decoderInputs, { position: step }))
        : await split.decoderContext.execute(decoderInputs);
      const decoderOutput = await execution.output(split.decoderOutput.name).read();
      const best = nextDecoderToken(split.decoderOutput, decoderOutput, step);
      await execution.close?.();
      if (best === EOS && !options['fixed-steps']) break;
      splitTokens.push(best);
      if (step + 1 < DECODER_LENGTH) {
        decoderIds[step + 1] = best;
        decoderKeep[step + 1] = 1;
      }
    }
    splitTimes.push(performance.now() - started);
    globalThis.__VOLVOX_WASM_PROFILE = undefined;

    // ---- legacy: whole model, incremental decode ----
    started = performance.now();
    globalThis.__VOLVOX_WASM_PROFILE = legacyProfile ?? undefined;
    const memoryKeep = new Int32Array(MEMORY_LENGTH);
    memoryKeep.fill(1, 0, MEMORY_LENGTH - QUESTION_LENGTH);
    memoryKeep.set(routerKeep, MEMORY_LENGTH - QUESTION_LENGTH);
    const legacyIds = new Int32Array(DECODER_LENGTH).fill(PAD);
    legacyIds[0] = BOS;
    const legacyKeep = new Int32Array(DECODER_LENGTH);
    legacyKeep[0] = 1;
    const legacyInputs = {
      image, q_ids: questionIds, router_keep: routerKeep, memory_keep: memoryKeep,
      y_ids: legacyIds, y_keep: legacyKeep,
    };
    await legacy.context.decode.reset();
    const legacyTokens = [];
    for (let step = 0; step < maxTokens; step++) {
      const execution = step === 0
        ? await legacy.context.decode.seed(legacyInputs)
        : await legacy.context.decode.step(legacyInputs, { position: step });
      const ids = await execution.output('token_ids').read();
      const next = Number(ids.length === 1 ? ids[0] : ids[step]);
      await execution.close?.();
      if (next === EOS && !options['fixed-steps']) break;
      legacyTokens.push(next);
      if (step + 1 < DECODER_LENGTH) {
        legacyIds[step + 1] = next;
        legacyKeep[step + 1] = 1;
      }
    }
    legacyTimes.push(performance.now() - started);
    globalThis.__VOLVOX_WASM_PROFILE = undefined;

    const splitText = vocab.decode(splitTokens).trim();
    const legacyText = vocab.decode(legacyTokens).trim();
    const comparison = compareGeneratedText(splitText, legacyText);
    const { splitAnswer, legacyAnswer } = comparison;
    if (!latencyOnly && splitAnswer === testCase.answer) splitCorrect++;
    if (!latencyOnly && legacyAnswer === testCase.answer) legacyCorrect++;
    if (comparison.answerAgreement) answerAgree++;
    if (comparison.structuredTextAgreement) structuredTextAgree++;
    caseResults.push(Object.freeze({
      id: String(testCase.id),
      family: FAMILY_ORDER[family],
      truth: testCase.answer,
      split_text: splitText,
      legacy_text: legacyText,
      split_answer: splitAnswer,
      legacy_answer: legacyAnswer,
      split_correct: latencyOnly ? null : splitAnswer === testCase.answer,
      legacy_correct: latencyOnly ? null : legacyAnswer === testCase.answer,
      answer_agreement: comparison.answerAgreement,
      structured_text_agreement: comparison.structuredTextAgreement,
      split_ms: splitTimes[splitTimes.length - 1],
      legacy_ms: legacyTimes[legacyTimes.length - 1],
    }));
    if (options.verbose) {
      process.stderr.write(`  [${testCase.id}] fam=${FAMILY_ORDER[family]} truth=${JSON.stringify(testCase.answer)}\n`
        + `      split =${JSON.stringify(splitText)}\n      legacy=${JSON.stringify(legacyText)}\n`);
    }
    if ((index + 1) % 10 === 0) process.stderr.write(`  ${index + 1}/${cases.length}\n`);
  }

  console.log(`\n=== ${cases.length} heldout questions, WASM, ${maxTokens} max tokens ===`);
  console.log(report(`split (${splitEngineCache.size} package(s))`, splitTimes));
  console.log(report(`legacy (${selectedFamilies.length} families)`, legacyTimes));
  const splitMedian = [...splitTimes].sort((a, b) => a - b)[Math.floor(splitTimes.length / 2)];
  const legacyMedian = [...legacyTimes].sort((a, b) => a - b)[Math.floor(legacyTimes.length / 2)];
  console.log(`\nmedian ratio legacy/split: ${(legacyMedian / splitMedian).toFixed(3)}x`);
  let splitWins = 0;
  for (let index = 0; index < splitTimes.length; index++) {
    if (splitTimes[index] < legacyTimes[index]) splitWins++;
  }
  console.log(`paired: split faster on ${splitWins}/${splitTimes.length} questions`);
  if (latencyOnly) {
    console.log('\nexact-match vs ground truth: not measured (--latency-only)');
  } else {
    console.log(`\nexact-match vs ground truth: split ${splitCorrect}/${cases.length}`
      + `, legacy ${legacyCorrect}/${cases.length}`);
  }
  console.log(`split vs legacy extracted-answer agreement: ${answerAgree}/${cases.length}`);
  console.log(`split vs legacy full structured-text agreement: ${structuredTextAgree}/${cases.length}`);
  if (options.report) {
    const benchmarkReport = {
      format: 'volvox-tinyreceipt-heldout-benchmark/v1',
      settings: {
        eval: resolve(evalRoot),
        count: cases.length,
        max_tokens: maxTokens,
        accuracy_mode: latencyOnly ? 'latency-only' : 'full-generation',
        backend: 'wasm',
        split_prefix: splitPackage ? null : resolve(splitPrefix),
        split_package: splitPackage ? resolve(splitPackage) : null,
        legacy_dir: resolve(legacyDir),
        selection: requestedIds ? 'explicit-ids' : 'sorted-annotation-filenames',
        qualified_families: selectedFamilies.map((family) => FAMILY_ORDER[family]),
      },
      provenance: {
        run_id: runId,
        started_at: startedAt,
        split_packages: splitPackageIdentities,
        legacy: {
          vocabulary: await fileIdentity(legacyVocabularyPath),
          router: router.identity.graph,
          weights: router.identity.weights,
          family_graphs: legacyGraphIdentities,
        },
      },
      summary: {
        split_median_ms: splitMedian,
        legacy_median_ms: legacyMedian,
        legacy_over_split_median_ratio: legacyMedian / splitMedian,
        split_paired_wins: splitWins,
        split_exact_match: latencyOnly ? null : splitCorrect,
        legacy_exact_match: latencyOnly ? null : legacyCorrect,
        extracted_answer_agreement: answerAgree,
        structured_text_agreement: structuredTextAgree,
      },
      cases: caseResults,
    };
    await writeFile(options.report, `${JSON.stringify(benchmarkReport, null, 2)}\n`);
    console.log(`wrote ${options.report}`);
  }
  printOperatorProfile('split', splitProfile);
  printOperatorProfile('legacy', legacyProfile);
  globalThis.__VOLVOX_WASM_PROFILE_SHAPES = undefined;
  process.exit(0);
}

if (process.argv[1]
    && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
