#!/usr/bin/env node
/**
 * Cold-path profiler for the TinyReceiptVQA WASM INT8 deployment.
 *
 * Decomposes a single *cold* inference — the whole process-start-to-answer
 * budget of track A — into its phases, and folds in the two opt-in engine
 * profiles so compile() and graph load are broken down from the inside:
 *
 *   image decode · runtime init · session load · preprocess · cold generation
 *     ├ graph load    → __VOLVOX_GRAPH_LOAD_PROFILE   (8 phases)
 *     └ compile       → __VOLVOX_WASM_COMPILE_PROFILE (6 phases + mem.grow)
 *
 * "Cold" is a property of the process, so `--repeat N` re-executes this script
 * as N child processes and reports the median of each field rather than
 * looping in-process (which would only measure the first run cold).
 *
 * This is a measurement tool, not a correctness gate: it reports the answer and
 * the strict-route attestation but does not compare against a golden answer.
 *
 *   node examples/tiny_receipt_vqa/tools/profile_cold_start.mjs \
 *     --package /tmp/volvoxai-bpe1536-int8-canonical-root \
 *     --image path/to/receipt.jpg --repeat 6
 */
import { execFileSync, execFileSync as run } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

const repository = fileURLToPath(new URL('../../../', import.meta.url));

function parseArgs(argv) {
  const options = {
    package: '/tmp/volvoxai-bpe1536-int8-canonical-root',
    image: null,
    prompt: 'phone number last one',
    wasm: `${repository}dist/0.4.0/volvoxai.wasm`,
    maxNewTokens: 191,
    repeat: 1,
    json: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    const value = argv[index + 1];
    switch (flag) {
      case '--package': options.package = value; index++; break;
      case '--image': options.image = value; index++; break;
      case '--prompt': options.prompt = value; index++; break;
      case '--wasm': options.wasm = value; index++; break;
      case '--max-new-tokens': options.maxNewTokens = Number(value); index++; break;
      case '--repeat': options.repeat = Number(value); index++; break;
      case '--json': options.json = true; break;
      default: throw new Error(`unknown argument '${flag}'`);
    }
  }
  if (!options.image) {
    throw new Error('--image <path> is required (the release receipt image is not in-tree)');
  }
  if (!Number.isInteger(options.maxNewTokens) ||
      options.maxNewTokens < 1 || options.maxNewTokens > 191) {
    throw new Error('--max-new-tokens must be an integer in [1, 191]');
  }
  if (!Number.isInteger(options.repeat) || options.repeat < 1) {
    throw new Error('--repeat must be a positive integer');
  }
  return options;
}

/** One cold run in this process. Returns the flat record printed as JSON. */
async function profileOnce(options) {
  // Both engine profiles are opt-in globals; set before anything loads a graph.
  globalThis.__VOLVOX_GRAPH_LOAD_PROFILE = true;
  globalThis.__VOLVOX_WASM_COMPILE_PROFILE = true;

  const { ModelLoader, Model, VolvoxAI } =
    await import(`${repository}ts/index.js`);
  const { TinyReceiptSplitSession } =
    await import(`${repository}examples/tiny_receipt_vqa/TinyReceiptSplitSession.js`);
  const { preprocessTinyReceiptImage } =
    await import(`${repository}examples/tiny_receipt_vqa/TinyReceiptInput.js`);

  async function fileFetch(input) {
    const url = input instanceof URL ? input : new URL(String(input));
    try {
      return new Response(await readFile(url), { status: 200, statusText: 'OK' });
    } catch (error) {
      return new Response(String(error), { status: 404, statusText: 'Not Found' });
    }
  }

  // Matches the release harness's Pillow RGB decode boundary exactly, so the
  // decode cost is comparable with the published figure.
  const decodeStarted = performance.now();
  const python = [
    'from PIL import Image',
    'import struct,sys',
    "im=Image.open(sys.argv[1]).convert('RGB')",
    "sys.stdout.buffer.write(struct.pack('<II', *im.size))",
    'sys.stdout.buffer.write(im.tobytes())',
  ].join('\n');
  const decoded = run('python3', ['-c', python, options.image], {
    maxBuffer: 32 * 1024 * 1024,
  });
  const imageDecodeMs = performance.now() - decodeStarted;
  const rawImage = {
    data: new Uint8Array(decoded.buffer, decoded.byteOffset + 8, decoded.byteLength - 8),
    width: decoded.readUInt32LE(0),
    height: decoded.readUInt32LE(4),
    channels: 3,
  };

  const runtimeInitStarted = performance.now();
  const runtime = await VolvoxAI.createRuntime({
    backends: ['wasm'],
    wasmUrl: pathToFileURL(options.wasm),
  });
  const runtimeInitMs = performance.now() - runtimeInitStarted;

  // Graph load and compile happen inside session load; capture each engine
  // profile as it is published so the encoder and decoder are kept apart.
  const graphLoadProfiles = [];
  const compileProfiles = [];
  globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS = [];
  globalThis.__VOLVOX_GRAPH_LOAD_PROFILE_RESULTS = [];
  const sessionLoadStarted = performance.now();
  const session = await TinyReceiptSplitSession.load({
    runtime,
    packageUrl: pathToFileURL(`${options.package}/package_manifest.json`),
    fetch: fileFetch,
    compileOptions: {
      backend: { mode: 'require', backend: 'wasm', operatorFallback: 'forbid' },
    },
    snapshotLoader: async ({ weightsUrl, graphUrl, kind }) => {
      const started = performance.now();
      const logicalPackage = await ModelLoader.load(weightsUrl, {
        graphUrl, fetch: fileFetch,
      });
      graphLoadProfiles.push({ kind, totalMs: performance.now() - started });
      return Model.capture(logicalPackage);
    },
  });
  const sessionLoadMs = performance.now() - sessionLoadStarted;

  const preprocessStarted = performance.now();
  const image = await preprocessTinyReceiptImage(rawImage, session.package.preprocessing);
  const preprocessMs = performance.now() - preprocessStarted;

  const generationStarted = performance.now();
  const answer = await session.generate({
    image, prompt: options.prompt, family: 'auto',
    preprocessed: true, maxNewTokens: options.maxNewTokens,
  });
  const coldGenerationMs = performance.now() - generationStarted;

  // Every compile() appends, so this covers encoder + decoder whether they
  // compiled during session load or lazily on first execute.
  compileProfiles.push(...(globalThis.__VOLVOX_WASM_COMPILE_PROFILE_RESULTS || []));

  return {
    imageDecodeMs, runtimeInitMs, sessionLoadMs, preprocessMs, coldGenerationMs,
    measuredTotalMs: imageDecodeMs + runtimeInitMs + sessionLoadMs +
      preprocessMs + coldGenerationMs,
    answer: typeof answer === 'string' ? answer : JSON.stringify(answer),
    graphLoadProfiles, compileProfiles,
  };
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const ms = (value) => (value == null ? '     —' : value.toFixed(1).padStart(7));

function reportTable(records) {
  const pick = (key) => median(records.map((record) => record[key]));
  const total = pick('measuredTotalMs');
  console.log(`\ncold path — median of ${records.length} process(es)\n`);
  console.log('| phase              |      ms |     % |');
  console.log('| ---                | ------: | ----: |');
  for (const [label, key] of [
    ['image decode', 'imageDecodeMs'], ['runtime init', 'runtimeInitMs'],
    ['session load', 'sessionLoadMs'], ['preprocess', 'preprocessMs'],
    ['cold generation', 'coldGenerationMs'],
  ]) {
    const value = pick(key);
    console.log(`| ${label.padEnd(18)} | ${ms(value)} | ${(100 * value / total).toFixed(1).padStart(5)} |`);
  }
  console.log(`| ${'measured total'.padEnd(18)} | ${ms(total)} | 100.0 |`);

  // Engine profiles: sum the per-graph records within a run, then take the
  // median across runs, so encoder+decoder are reported as one cold budget.
  const sumOver = (list, key) => list.reduce((total, entry) => total + (entry[key] || 0), 0);
  const compileKeys = ['descriptorMs', 'tensorAllocMs', 'packWeightMs', 'viewRefreshMs',
    'metadataAllocMs', 'preflightMs', 'unattributedMs', 'totalMs'];
  const compileCounters = ['growCount', 'growPages', 'growMs', 'allocBytesCount',
    'viewRefreshCount', 'descriptorCount', 'tensorCopyBytes', 'packBytes', 'finalHeapBytes'];
  if (records[0].compileProfiles.length) {
    const compileTotal = median(records.map((r) => sumOver(r.compileProfiles, 'totalMs')));
    console.log(`\ncompile() ${compileTotal.toFixed(1)} ms — breakdown\n`);
    console.log('| item               |      ms |     % |');
    console.log('| ---                | ------: | ----: |');
    for (const key of compileKeys.filter((k) => k !== 'totalMs')) {
      const value = median(records.map((r) => sumOver(r.compileProfiles, key)));
      console.log(`| ${key.padEnd(18)} | ${ms(value)} | ${(100 * value / compileTotal).toFixed(1).padStart(5)} |`);
    }
    const growMs = median(records.map((r) => sumOver(r.compileProfiles, 'growMs')));
    console.log(`| ${'↳ of which mem.grow'.padEnd(18)} | ${ms(growMs)} | ${(100 * growMs / compileTotal).toFixed(1).padStart(5)} |`);
    console.log('\ncounters: ' + compileCounters
      .map((key) => `${key}=${median(records.map((r) => sumOver(r.compileProfiles, key)))}`)
      .join(' '));
  }
  if (records[0].graphLoadProfiles.length) {
    const loadTotal = median(records.map((r) => sumOver(r.graphLoadProfiles, 'totalMs')));
    console.log(`\ngraph load ${loadTotal.toFixed(1)} ms — breakdown\n`);
    console.log('| item               |      ms |     % |');
    console.log('| ---                | ------: | ----: |');
    for (const key of ['graphFetchMs', 'graphParseMs', 'graphValidateMs', 'weightFetchMs',
      'weightDecodeMs', 'tensorBuildMs', 'nodeBuildMs', 'postProcessMs', 'unattributedMs']) {
      const value = median(records.map((r) => sumOver(r.graphLoadProfiles, key)));
      console.log(`| ${key.padEnd(18)} | ${ms(value)} | ${(100 * value / loadTotal).toFixed(1).padStart(5)} |`);
    }
    console.log('\ncounters: ' + ['graphBytes', 'weightBytes', 'tensorCount', 'nodeCount']
      .map((key) => `${key}=${median(records.map((r) => sumOver(r.graphLoadProfiles, key)))}`)
      .join(' '));
  }
  console.log(`\nanswer: ${JSON.stringify(records[0].answer)}`);
}

const options = parseArgs(process.argv.slice(2));
if (options.repeat > 1) {
  // Each repetition must be a fresh process to stay cold. Carry this process's
  // execArgv and env so a child resolves modules the same way the parent does
  // (this script is normally driven through tsx, which bare node cannot follow).
  const records = [];
  const childArgs = [...process.execArgv, process.argv[1],
    '--package', options.package, '--image', options.image,
    '--prompt', options.prompt, '--wasm', options.wasm,
    '--max-new-tokens', String(options.maxNewTokens), '--json'];
  for (let attempt = 0; attempt < options.repeat; attempt++) {
    const stdout = execFileSync(process.execPath, childArgs, {
      maxBuffer: 64 * 1024 * 1024, encoding: 'utf8', env: process.env,
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    const line = stdout.trim().split('\n').at(-1);
    records.push(JSON.parse(line));
    process.stderr.write(`run ${attempt + 1}/${options.repeat}: ` +
      `${records.at(-1).measuredTotalMs.toFixed(1)} ms\n`);
  }
  reportTable(records);
} else {
  const record = await profileOnce(options);
  if (options.json) console.log(JSON.stringify(record));
  else reportTable([record]);
}
