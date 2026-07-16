#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

import { VolvoxAI } from '../../../ts/VolvoxAI.js';
import { WasmEngine } from '../../../ts/backends/WasmEngine.js';
import {
  TinyReceiptW8A8Session,
  preprocessTinyReceiptImage,
} from '../TinyReceiptW8A8Session.js';

const repository = fileURLToPath(new URL('../../../', import.meta.url));
const packageDir = process.argv[2] || '/tmp/volvoxai-tinyreceipt-safetensors-v1';
const navercapRoot = process.env.NAVERCAP_ROOT;
const imagePath = process.argv[3] || (navercapRoot
  ? join(navercapRoot, 'eval', 'heldout', 'images', '00002.jpg')
  : null);
if (!imagePath) {
  throw new Error(
    'pass an image path as argument 2 or set NAVERCAP_ROOT=/path/to/navercap',
  );
}
const prompt = process.argv[4] || 'phone number last one';
const wasmPath = process.argv[5] || `${repository}/dist/0.2.0/volvoxai.wasm`;

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

const baseRuntime = new VolvoxAI();
const graphLoads = [];
const runtime = {
  async loadGraph(sources, options) {
    const started = performance.now();
    try {
      return await baseRuntime.loadGraph(sources, options);
    } finally {
      graphLoads.push({
        configUrl: options.configUrl,
        ms: performance.now() - started,
      });
    }
  },
  compile() {
    throw new Error('benchmark compileGraph override was not installed');
  },
};

let activeRun = null;
const engines = [];
const sessionLoadStarted = performance.now();
const session = await TinyReceiptW8A8Session.load({
  runtime,
  packageUrl: pathToFileURL(`${packageDir}/package_manifest.json`),
  fetch: fileFetch,
  // One backend instance per graph keeps router and family allocations live
  // simultaneously; this is the intended host contract for this benchmark.
  compileGraph: async (graph) => {
    const label = graph.tensors.has('image') ? 'family' : 'router';
    const initStarted = performance.now();
    const engine = await WasmEngine.init(pathToFileURL(wasmPath));
    const initMs = performance.now() - initStarted;
    if (!engine) throw new Error('strict WASM initialization failed');
    const compileStarted = performance.now();
    engine.allocateGraph(graph);
    const compileMs = performance.now() - compileStarted;
    const record = {
      label,
      initMs,
      compileMs,
      relaxedSimdEnabled: engine.relaxedSimdEnabled,
      heapMiB: engine.mem.buffer.byteLength / 1048576,
      executions: [[], []],
    };
    const execute = engine.execute.bind(engine);
    engine.execute = async (inputs, options) => {
      const started = performance.now();
      try {
        return await execute(inputs, options);
      } finally {
        if (activeRun != null) {
          record.executions[activeRun].push({
            ms: performance.now() - started,
            reset: options?.incrementalReset === true,
            row: options?.incrementalRowPosition ?? null,
          });
        }
      }
    };
    engines.push(record);
    return engine;
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
      incremental: true,
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

if (runs[0].answer.family !== runs[1].answer.family ||
    runs[0].answer.text !== runs[1].answer.text ||
    runs[0].answer.tokenIds.length !== runs[1].answer.tokenIds.length) {
  throw new Error('cold and hot generations produced different answers');
}

function executionSummary(runIndex) {
  const router = engines.find((record) => record.label === 'router')?.executions[runIndex] || [];
  const family = engines.find((record) => record.label === 'family')?.executions[runIndex] || [];
  const steady = family.slice(1).map(({ ms }) => ms);
  const steadyMean = steady.length
    ? steady.reduce((sum, value) => sum + value, 0) / steady.length
    : 0;
  return {
    routerMs: router.reduce((sum, value) => sum + value.ms, 0),
    familySeedMs: family[0]?.ms || 0,
    steadySteps: steady.length,
    steadyMeanMs: steadyMean,
    steadyTokensPerSecond: steadyMean ? 1000 / steadyMean : 0,
  };
}

const output = {
  provenance: {
    packageDir,
    imagePath,
    prompt,
    wasmPath,
    node: process.version,
    imageDecode: 'Pillow RGB',
    preprocessing: 'TinyReceiptW8A8Session JavaScript',
    coldDefinition: 'session manifest/vocab loaded; router/family graphs and WASM executors absent',
  },
  sourceImage: { width: sourceWidth, height: sourceHeight },
  sessionLoadMs,
  preprocessMs,
  fetches: [...fetchRecords.entries()].map(([url, record]) => ({ url, ...record })),
  graphLoads: graphLoads.map((record) => ({ ...record })),
  engines: engines.map((record) => ({
    label: record.label,
    initMs: record.initMs,
    compileMs: record.compileMs,
    relaxedSimdEnabled: record.relaxedSimdEnabled,
    heapMiB: record.heapMiB,
  })),
  runs: runs.map((run, index) => ({
    name: run.name,
    generationMs: run.generationMs,
    family: run.answer.family,
    text: run.answer.text,
    tokens: run.answer.tokenIds.length,
    execution: run.answer.execution,
    ...executionSummary(index),
  })),
  coldMinusHotMs: runs[0].generationMs - runs[1].generationMs,
  coldToHotRatio: runs[0].generationMs / runs[1].generationMs,
};

console.log('WASM_TINYRECEIPT_COLD_HOT ' + JSON.stringify(output, (_key, value) =>
  typeof value === 'number' ? Number(value.toFixed(3)) : value, 2));
