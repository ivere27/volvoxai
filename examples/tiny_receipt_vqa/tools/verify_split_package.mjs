#!/usr/bin/env node
/**
 * Verify one or two TinyReceipt explicit-KV packages on the sealed
 * deterministic workload.
 *
 * Every package is executed twice through TinyReceiptSplitSession: once with
 * exact active B/Q/T/M views and once with the explicitly labelled
 * maximum-padded reference binding. Every decoder decision must use an
 * ordinary one-token execution with the explicit P=1 blocked-zero sentinel and
 * growing K/V cache. The router family and every greedy token decision must
 * agree between the two bindings and, when supplied, with the reference
 * package.
 *
 *   node --import tsx examples/tiny_receipt_vqa/tools/verify_split_package.mjs \
 *     --package build/tiny-receipt-int8-from-fp32 \
 *     [--reference build/tiny-receipt-fp32] [--tokens 8] \
 *     [--backend cpu|wasm] [--wasm dist/0.4.0/volvoxai.wasm]
 */

import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  ModelLoader,
  Model,
  VolvoxAI,
} from '../../../ts/index.js';
import {
  createTinyReceiptSplitE2EImage,
  TINY_RECEIPT_SPLIT_E2E_WORKLOAD,
} from '../TinyReceiptSplitE2E.js';
import { TinyReceiptSplitSession } from '../TinyReceiptSplitSession.js';

const MAXIMUM_ENCODER_SHAPE = Object.freeze({ B: 1, Q: 192, M: 402 });
const IMAGE_TOKENS = 210;
const ALLOWED_OPTIONS = new Set(['package', 'reference', 'tokens', 'backend', 'wasm']);

function parseArguments(values) {
  const result = {};
  for (let index = 0; index < values.length; index++) {
    const raw = values[index];
    if (!raw.startsWith('--')) throw new Error(`unexpected argument ${JSON.stringify(raw)}`);
    const name = raw.slice(2);
    if (!ALLOWED_OPTIONS.has(name)) throw new Error(`unknown option --${name}`);
    if (Object.hasOwn(result, name)) throw new Error(`duplicate option --${name}`);
    const value = values[index + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`option --${name} requires a value`);
    }
    result[name] = value;
    index++;
  }
  return result;
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function sameShape(left, right) {
  return isRecord(left) && isRecord(right)
    && ['B', 'Q', 'T', 'M'].every((name) => left[name] === right[name]);
}

function exactShape(value, expected, label) {
  if (!sameShape(value, expected)) {
    throw new Error(`${label} must be B/Q/T/M=${shapeLabel(expected)}`);
  }
}

function shapeLabel(value) {
  return `${value.B}/${value.Q}/${value.T}/${value.M}`;
}

function firstTokenDifference(left, right) {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index++) {
    if (left[index] !== right[index]) return index;
  }
  return null;
}

function requireExplicitKVDecode(answer, label) {
  if (!isRecord(answer) || answer.execution !== 'explicit-kv-cache'
      || answer.decodeMode !== 'explicit-kv-cache'
      || !Array.isArray(answer.tokenIds) || answer.tokenIds.length < 1
      || answer.decoderOrdinaryExecutions !== answer.tokenIds.length
      || answer.decoderSeedExecutions !== 1
      || answer.decoderCacheStepExecutions !== answer.tokenIds.length - 1
      || !Array.isArray(answer.decodeReports)
      || answer.decodeReports.length !== answer.tokenIds.length
      || !isRecord(answer.cacheShape)
      || answer.cacheShape.initialPastLength !== 1
      || answer.cacheShape.finalPastLength !== answer.tokenIds.length + 1
      || answer.cacheShape.sentinelSlots !== 1) {
    throw new Error(`${label} must use one-token explicit KV runs from one blocked sentinel`);
  }
  for (const [index, report] of answer.decodeReports.entries()) {
    if (!isRecord(report)
        || report.operation !== (index === 0 ? 'explicit-kv-seed' : 'explicit-kv-step')
        || report.position !== index
        || report.pastLength !== index + 1
        || report.presentLength !== index + 2
        || report.sentinelMaskValue !== 1) {
      throw new Error(`${label} explicit KV report ${index} is not the exact v1 contract`);
    }
  }
}

/**
 * Validate and compare one exact-active answer with its maximum-padded oracle.
 * The returned mismatch is a semantic result; malformed lifecycle or shape
 * evidence throws.
 */
export function verifyActivePaddedDecisionOracle(active, padded) {
  if (active?.shapeMode !== 'active' || padded?.shapeMode !== 'maximum-padded') {
    throw new Error('oracle requires one active run and one maximum-padded run');
  }
  if (!isRecord(active.logicalShape) || active.logicalShape.B !== 1
      || active.logicalShape.Q < 1 || active.logicalShape.Q > 192
      || active.logicalShape.T < 2 || active.logicalShape.T > 192
      || active.logicalShape.M !== active.logicalShape.Q + IMAGE_TOKENS) {
    throw new Error('active logical shape must satisfy B=1, M=Q+210, and bounded Q/T');
  }
  exactShape(active.activeShape, active.logicalShape, 'active binding');
  exactShape(padded.logicalShape, active.logicalShape, 'maximum-padded logical request');
  if (!sameArray(active.questionTokenIds, padded.questionTokenIds)
      || active.questionTokenIds.length !== active.logicalShape.Q) {
    throw new Error('active and maximum-padded runs must encode the same exact active question');
  }
  requireExplicitKVDecode(active, 'active run');
  requireExplicitKVDecode(padded, 'maximum-padded run');
  exactShape(
    padded.activeShape,
    { ...MAXIMUM_ENCODER_SHAPE, T: active.logicalShape.T },
    'maximum-padded binding',
  );

  const activeTokens = [...active.tokenIds];
  const paddedTokens = [...padded.tokenIds];
  const firstDifference = firstTokenDifference(activeTokens, paddedTokens);
  const familyMatch = active.family === padded.family && active.familyId === padded.familyId;
  const tokensMatch = firstDifference === null;
  return Object.freeze({
    exact: familyMatch && tokensMatch,
    familyMatch,
    tokensMatch,
    textMatch: active.text === padded.text,
    firstTokenDifference: firstDifference,
  });
}

/** Execute both explicit-KV shape modes against one already-loaded session. */
export async function runActivePaddedOracle({
  session,
  image,
  prompt,
  family = 'auto',
  maxNewTokens,
}) {
  if (!session || typeof session.generate !== 'function') {
    throw new Error('oracle requires a loaded TinyReceipt split session');
  }
  if (!(image instanceof Float32Array) || image.length !== 320 * 672) {
    throw new Error('oracle image must be preprocessed F32 [1,1,320,672]');
  }
  if (typeof prompt !== 'string' || prompt.length === 0) {
    throw new Error('oracle prompt must be a non-empty string');
  }
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 191) {
    throw new Error('oracle maxNewTokens must be an integer in [1, 191]');
  }
  const runs = {};
  for (const [name, shapeMode] of [
    ['active', 'active'],
    ['padded', 'maximum-padded'],
  ]) {
    const started = performance.now();
    const answer = await session.generate({
      image,
      prompt,
      family,
      maxNewTokens,
      preprocessed: true,
      shapeMode,
    });
    runs[name] = Object.freeze({ answer, elapsed: performance.now() - started });
  }
  const comparison = verifyActivePaddedDecisionOracle(
    runs.active.answer,
    runs.padded.answer,
  );
  return Object.freeze({
    active: runs.active,
    padded: runs.padded,
    comparison,
  });
}

async function fileFetch(input) {
  let url;
  try {
    url = input instanceof URL ? input : new URL(String(input));
  } catch (error) {
    return { ok: false, status: 400, statusText: error?.message || String(error) };
  }
  if (url.protocol !== 'file:') {
    return { ok: false, status: 400, statusText: `unsupported protocol ${url.protocol}` };
  }
  try {
    const bytes = await readFile(fileURLToPath(url));
    return {
      ok: true,
      status: 200,
      json: async () => JSON.parse(bytes.toString('utf8')),
      arrayBuffer: async () => bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ),
    };
  } catch (error) {
    return { ok: false, status: 404, statusText: error?.message || String(error) };
  }
}

async function snapshotLoader({ graphUrl, weightsUrl, fetch }) {
  const logicalPackage = await ModelLoader.load(weightsUrl, { graphUrl, fetch });
  return Model.capture(logicalPackage);
}

async function runPackage(packageDir, backend, wasmPath, maxNewTokens) {
  const runtime = await VolvoxAI.createRuntime({
    backends: [backend],
    ...(backend === 'wasm' ? { wasmUrl: pathToFileURL(resolve(wasmPath)) } : {}),
  });
  let session;
  try {
    session = await TinyReceiptSplitSession.load({
      runtime,
      packageUrl: pathToFileURL(resolve(packageDir, 'package_manifest.json')),
      fetch: fileFetch,
      snapshotLoader,
      compileOptions: {
        backend: { mode: 'require', backend, operatorFallback: 'forbid' },
      },
    });
    if (maxNewTokens > session.package.generation.maximum_new_tokens) {
      throw new Error(`--tokens exceeds package maximum ${
        session.package.generation.maximum_new_tokens}`);
    }
    await session.preload();
    const oracle = await runActivePaddedOracle({
      session,
      image: createTinyReceiptSplitE2EImage(),
      prompt: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.prompt,
      family: TINY_RECEIPT_SPLIT_E2E_WORKLOAD.family,
      maxNewTokens,
    });
    return Object.freeze({
      oracle,
      vocabulary: Object.freeze([...session.vocab.itos]),
    });
  } finally {
    try {
      await session?.close();
    } finally {
      await runtime.close();
    }
  }
}

function printRun(label, value) {
  const answer = value.answer;
  console.log(`  ${label.padEnd(14)} B/Q/T/M ${shapeLabel(answer.activeShape)}`
    + `  family ${answer.familyId}:${answer.family}`
    + `  tokens ${JSON.stringify(answer.tokenIds)}`);
  console.log(`  ${''.padEnd(14)} ${value.elapsed.toFixed(0)} ms  decoder ordinary/cache-step `
    + `${answer.decoderOrdinaryExecutions}/${answer.decoderCacheStepExecutions}`);
}

function printPackage(label, packageDir, backend, result) {
  console.log(`${label} ${packageDir} [${backend}]`);
  printRun('active', result.oracle.active);
  printRun('max-padded ref', result.oracle.padded);
  const speedup = result.oracle.padded.elapsed / result.oracle.active.elapsed;
  console.log(`  active/padded decision ${result.oracle.comparison.exact ? 'MATCH' : 'DIFFERS'}`
    + `  padded/active time ${speedup.toFixed(2)}x`);
}

async function defaultWasmPath() {
  const metadata = JSON.parse(
    await readFile(new URL('../../../package.json', import.meta.url), 'utf8'),
  );
  if (typeof metadata.version !== 'string' || metadata.version.length === 0) {
    throw new Error('package.json must declare a version');
  }
  const repository = fileURLToPath(new URL('../../../', import.meta.url));
  return resolve(repository, 'dist', metadata.version, 'volvoxai.wasm');
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (typeof options.package !== 'string') throw new Error('pass --package <dir>');
  const backend = options.backend ?? 'cpu';
  if (backend !== 'cpu' && backend !== 'wasm') {
    throw new Error("--backend must be 'cpu' or 'wasm'");
  }
  const maxNewTokens = Number(options.tokens ?? TINY_RECEIPT_SPLIT_E2E_WORKLOAD.maxNewTokens);
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 191) {
    throw new Error('--tokens must be an integer in [1, 191]');
  }
  const wasmPath = options.wasm ?? await defaultWasmPath();
  const actual = await runPackage(options.package, backend, wasmPath, maxNewTokens);
  printPackage('package  ', options.package, backend, actual);
  let exact = actual.oracle.comparison.exact;

  if (options.reference) {
    const reference = await runPackage(options.reference, backend, wasmPath, maxNewTokens);
    printPackage('reference', options.reference, backend, reference);
    if (!sameArray(actual.vocabulary, reference.vocabulary)) {
      throw new Error('package and reference vocabularies do not have identical token-ID semantics');
    }
    const actualAnswer = actual.oracle.active.answer;
    const referenceAnswer = reference.oracle.active.answer;
    const tokenDifference = firstTokenDifference(
      [...actualAnswer.tokenIds],
      [...referenceAnswer.tokenIds],
    );
    const familyMatch = actualAnswer.familyId === referenceAnswer.familyId;
    const packageMatch = familyMatch && tokenDifference === null;
    console.log(`package/reference family ${familyMatch ? 'MATCH' : 'DIFFERS'}`
      + ` | greedy tokens ${tokenDifference === null ? 'MATCH' : `DIFFER AT ${tokenDifference}`}`);
    exact &&= reference.oracle.comparison.exact && packageMatch;
  }
  if (!exact) process.exitCode = 1;
}

if (process.argv[1]
    && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
