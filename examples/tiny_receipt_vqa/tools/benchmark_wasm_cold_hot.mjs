#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

import {
  Graph,
  GraphLoader,
  ReadOnlySafetensorsCache,
  VolvoxAI,
} from '../../../ts/index.js';
import {
  TinyReceiptSplitSession,
} from '../TinyReceiptSplitSession.js';
import { preprocessTinyReceiptImage } from '../TinyReceiptW8A8Session.js';

const repository = fileURLToPath(new URL('../../../', import.meta.url));
const packageDir = process.argv[2] || '/tmp/volvoxai-bpe1536-int8-canonical-root';
const receiptDataRoot = process.env.RECEIPT_VQA_DATA_ROOT;
const imagePath = process.argv[3] || (receiptDataRoot
  ? join(receiptDataRoot, 'eval', 'heldout', 'images', '00002.jpg')
  : null);
if (!imagePath) {
  throw new Error(
    'pass an image path as argument 2 or set '
    + 'RECEIPT_VQA_DATA_ROOT=/path/to/receipt-vqa-data',
  );
}
const prompt = process.argv[4] || 'phone number last one';
const wasmPath = process.argv[5] || `${repository}/dist/0.3.0/volvoxai.wasm`;
const maxNewTokens = Number(process.argv[6] || 191);
if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 191) {
  throw new Error('max-new-tokens argument must be an integer in [1, 191]');
}

const fetchRecords = new Map();
async function fileFetch(input) {
  const url = input instanceof URL ? input : new URL(String(input));
  const key = url.toString();
  try {
    const bytes = await readFile(url);
    const record = fetchRecords.get(key) || { requests: 0, bytes: bytes.byteLength };
    record.requests++;
    record.bytes = bytes.byteLength;
    fetchRecords.set(key, record);
    return new Response(bytes, { status: 200, statusText: 'OK' });
  } catch (error) {
    return new Response(String(error), { status: 404, statusText: 'Not Found' });
  }
}

// Keep image decoding outside the zero-dependency inference module while
// matching the source evaluator's Pillow RGB decode boundary exactly.
const python = [
  'from PIL import Image',
  'import struct,sys',
  "im=Image.open(sys.argv[1]).convert('RGB')",
  "sys.stdout.buffer.write(struct.pack('<II', *im.size))",
  'sys.stdout.buffer.write(im.tobytes())',
].join('\n');
const decoded = execFileSync('python3', ['-c', python, imagePath], {
  maxBuffer: 32 * 1024 * 1024,
});
const sourceWidth = decoded.readUInt32LE(0);
const sourceHeight = decoded.readUInt32LE(4);
const rawImage = {
  data: new Uint8Array(decoded.buffer, decoded.byteOffset + 8, decoded.byteLength - 8),
  width: sourceWidth,
  height: sourceHeight,
  channels: 3,
};

const graphLoads = [];
let activeRun = null;
const compilations = [];
const contextLabels = new Map();
const executions = { encoder: [[], []], decoder: [[], []] };
const runtimeInitStarted = performance.now();
const runtime = await VolvoxAI.createRuntime({
  backends: ['wasm'],
  wasmUrl: pathToFileURL(wasmPath),
  onDiagnostic(event) {
    if (event.kind === 'compilation') {
      compilations.push({
        label: compilations.length === 0 ? 'encoder' : 'decoder',
        report: event.report,
      });
      return;
    }
    if (event.kind !== 'execution' || activeRun == null) return;
    if (!contextLabels.has(event.report.contextId)) {
      contextLabels.set(event.report.contextId, contextLabels.size === 0 ? 'encoder' : 'decoder');
    }
    const label = contextLabels.get(event.report.contextId);
    executions[label][activeRun].push({
      ms: event.report.executionTimeMs ?? 0,
      executionId: event.report.executionId,
      contextId: event.report.contextId,
      backend: event.report.backend,
      device: event.report.device,
      outcome: event.report.outcome,
      operatorFallback: event.report.operatorFallback,
      routeEvidence: event.report.routeEvidence,
      decodeState: event.report.decodeState,
    });
  },
});
const runtimeInitMs = performance.now() - runtimeInitStarted;
const sessionCache = new ReadOnlySafetensorsCache();
const sessionLoadStarted = performance.now();
const session = await TinyReceiptSplitSession.load({
  runtime,
  packageUrl: pathToFileURL(`${packageDir}/package_manifest.json`),
  fetch: fileFetch,
  safetensorsCache: sessionCache,
  decodePolicy: 'required',
  compileOptions: {
    backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
  },
  graphLoader: async ({ weightsUrl, graphUrl, kind }) => {
    const started = performance.now();
    try {
      const graph = new Graph();
      await GraphLoader.load(graph, weightsUrl, {
        graphUrl,
        fetch: fileFetch,
        safetensorsCache: sessionCache,
      });
      return graph;
    } finally {
      graphLoads.push({ graphUrl, kind, ms: performance.now() - started });
    }
  },
});
const sessionLoadMs = performance.now() - sessionLoadStarted;

const preprocessStarted = performance.now();
const image = await preprocessTinyReceiptImage(rawImage, session.package.preprocessing);
const preprocessMs = performance.now() - preprocessStarted;

const runs = [];
for (let runIndex = 0; runIndex < 2; runIndex++) {
  activeRun = runIndex;
  const started = performance.now();
  let answer;
  try {
    answer = await session.generate({
      image,
      prompt,
      family: 'auto',
      preprocessed: true,
      maxNewTokens,
    });
  } finally {
    activeRun = null;
  }
  runs.push({
    name: runIndex === 0 ? 'cold-graphs' : 'hot-graphs',
    generationMs: performance.now() - started,
    answer,
  });
}

function strictRoute(route, label) {
  if (route?.tierFallback !== false ||
      route?.operator?.attestation !== 'none' ||
      route.operator.used !== false ||
      route.operator.offendingNode != null) {
    throw new Error(`${label} does not prove a strict WASM no-fallback route`);
  }
}

function answerSignature(answer) {
  return JSON.stringify({
    family: answer.family,
    familyId: answer.familyId,
    requestedFamily: answer.requestedFamily,
    requestedFamilyId: answer.requestedFamilyId,
    questionTokenIds: [...answer.questionTokenIds],
    tokenIds: [...answer.tokenIds],
    text: answer.text,
    stoppedAtEos: answer.stoppedAtEos,
    execution: answer.execution,
    decodeMode: answer.decodeMode,
    decoderSeedExecutions: answer.decoderSeedExecutions,
    decoderRowExecutions: answer.decoderRowExecutions,
    decoderOrdinaryExecutions: answer.decoderOrdinaryExecutions,
    routerLogits: [...answer.routerLogits],
  });
}

if (answerSignature(runs[0].answer) !== answerSignature(runs[1].answer)) {
  throw new Error('cold and hot generations produced different exact answers');
}

if (compilations.length !== 2) {
  throw new Error(`expected two WASM compilations, received ${compilations.length}`);
}
for (const [index, { label, report }] of compilations.entries()) {
  const expectedLabel = index === 0 ? 'encoder' : 'decoder';
  if (label !== expectedLabel ||
      report?.requestedPolicy?.mode !== 'require' ||
      report.requestedPolicy.backend !== 'wasm' ||
      report.requestedPolicy.operatorFallback !== 'forbid' ||
      report.selectedBackend !== 'wasm' ||
      !Array.isArray(report.candidates) ||
      report.candidates.length !== 1 ||
      report.candidates[0]?.backend !== 'wasm' ||
      report.candidates[0]?.outcome !== 'selected') {
    throw new Error(`${expectedLabel} compilation did not strictly select WASM`);
  }
  strictRoute(report.routeEvidence, `${expectedLabel} compilation`);
  strictRoute(report.candidates[0].routeEvidence, `${expectedLabel} compilation candidate`);
}

function strictExecution(record, label) {
  if (record?.backend !== 'wasm' ||
      record.outcome !== 'success' ||
      record.operatorFallback !== 'none') {
    throw new Error(`${label} did not execute successfully on strict WASM`);
  }
  strictRoute(record.routeEvidence, label);
}

function executionSummary(runIndex, endToEndMs) {
  const encoder = executions.encoder[runIndex];
  const decoder = executions.decoder[runIndex];
  const answer = runs[runIndex].answer;
  if (answer.execution !== 'context-decode-retained-row' ||
      answer.decodeMode !== 'incremental-row-required' ||
      answer.decoderSeedExecutions !== 1 ||
      answer.decoderRowExecutions !== Math.max(0, answer.tokenIds.length - 1) ||
      answer.decoderOrdinaryExecutions !== 0) {
    throw new Error(`run ${runIndex} did not report the required retained-row contract`);
  }
  if (encoder.length !== 1 || decoder.length !== answer.tokenIds.length || decoder.length < 1) {
    throw new Error(
      `run ${runIndex} expected one encoder and ${answer.tokenIds.length} decoder executions`,
    );
  }
  strictExecution(encoder[0], `run ${runIndex} encoder`);
  if (encoder[0].decodeState?.operation !== 'execute' ||
      encoder[0].decodeState.mode != null ||
      encoder[0].decodeState.position != null) {
    throw new Error(`run ${runIndex} encoder did not use an ordinary execution`);
  }
  const seed = decoder[0];
  strictExecution(seed, `run ${runIndex} decoder seed`);
  if (seed.decodeState?.operation !== 'seed' ||
      seed.decodeState.mode !== 'incremental-seed' ||
      seed.decodeState.cacheState !== 'seeded' ||
      seed.decodeState.position != null) {
    throw new Error(`run ${runIndex} decoder did not begin with one incremental seed`);
  }
  const rows = decoder.slice(1);
  for (const [index, row] of rows.entries()) {
    const position = index + 1;
    strictExecution(row, `run ${runIndex} decoder row ${position}`);
    if (row.decodeState?.operation !== 'step' ||
        row.decodeState.mode !== 'incremental-row' ||
        row.decodeState.cacheState !== 'advanced' ||
        row.decodeState.position !== position) {
      throw new Error(`run ${runIndex} decoder row ${position} was not retained-row execution`);
    }
  }
  if (encoder[0].contextId === seed.contextId ||
      decoder.some(({ contextId }) => contextId !== seed.contextId)) {
    throw new Error(`run ${runIndex} did not isolate encoder and decoder contexts`);
  }

  const steadyTotal = rows.reduce((sum, { ms }) => sum + ms, 0);
  const steadyMean = rows.length
    ? steadyTotal / rows.length
    : 0;
  const decoderTotal = decoder.reduce((sum, { ms }) => sum + ms, 0);
  return {
    encoderMs: encoder.reduce((sum, value) => sum + value.ms, 0),
    decoderSeedMs: seed.ms,
    decoderSteadySteps: rows.length,
    decoderSteadyTotalMs: steadyTotal,
    decoderSteadyMeanMs: steadyMean,
    decoderSteadyTokensPerSecond: steadyMean ? 1000 / steadyMean : 0,
    decoderTotalMs: decoderTotal,
    endToEndMs,
    // Retain the original field names for existing report consumers.
    decoderFirstStepMs: seed.ms,
    steadySteps: rows.length,
    steadyMeanMs: steadyMean,
    steadyTokensPerSecond: steadyMean ? 1000 / steadyMean : 0,
    executionEvidence: {
      encoderOperation: encoder[0].decodeState.operation,
      decoderSeedMode: seed.decodeState.mode,
      decoderSteadyMode: rows.length ? 'incremental-row' : null,
      strictNoFallback: true,
    },
  };
}

const executionSummaries = runs.map((run, index) =>
  executionSummary(index, run.generationMs));

const output = {
  provenance: {
    packageDir,
    imagePath,
    prompt,
    maxNewTokens,
    wasmPath,
    node: process.version,
    imageDecode: 'Pillow RGB',
    preprocessing: 'TinyReceipt split-session JavaScript',
    coldDefinition: 'runtime and manifest/vocab loaded; encoder/decoder models not compiled',
    endToEndDefinition: 'TinyReceiptSplitSession.generate on the preprocessed image',
    decodePolicy: 'required retained-row WASM',
  },
  sourceImage: { width: sourceWidth, height: sourceHeight },
  runtimeInitMs,
  sessionLoadMs,
  preprocessMs,
  fetches: [...fetchRecords.entries()].map(([url, record]) => ({ url, ...record })),
  graphLoads: graphLoads.map((record) => ({ ...record })),
  compilations: compilations.map(({ label, report }) => ({ label, report })),
  runs: runs.map((run, index) => ({
    name: run.name,
    generationMs: run.generationMs,
    family: run.answer.family,
    familyId: run.answer.familyId,
    requestedFamily: run.answer.requestedFamily,
    text: run.answer.text,
    tokenIds: [...run.answer.tokenIds],
    tokens: run.answer.tokenIds.length,
    stoppedAtEos: run.answer.stoppedAtEos,
    execution: run.answer.execution,
    decodeMode: run.answer.decodeMode,
    decoderSeedExecutions: run.answer.decoderSeedExecutions,
    decoderRowExecutions: run.answer.decoderRowExecutions,
    decoderOrdinaryExecutions: run.answer.decoderOrdinaryExecutions,
    ...executionSummaries[index],
  })),
  coldHotExactAnswerMatch: true,
  coldMinusHotMs: runs[0].generationMs - runs[1].generationMs,
  coldToHotRatio: runs[0].generationMs / runs[1].generationMs,
};

await session.close();
await runtime.close();

console.log('WASM_TINYRECEIPT_COLD_HOT ' + JSON.stringify(output, (_key, value) =>
  typeof value === 'number' ? Number(value.toFixed(3)) : value, 2));
