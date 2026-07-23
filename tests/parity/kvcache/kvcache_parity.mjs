// KV-cache decode-path parity. The row/KV-cache incremental decode (DecodeSession,
// rowMode:'required') is the efficient generation path that ships for the W8A8
// encoder-decoder — entirely separate code from a full forward, with its own cache
// indexing. This drives a small W8A8 decoder (QEmbedding/QLayerNorm/QLinear/QSDPA
// self+cross/QGELU/QSiLU) through the KV-cache decode and checks its final retained
// self-attention K/V against a CPU full recompute. The graph mirrors
// tests/js_w8a8_decode_cache.test.mjs.
//
//   node tests/parity/kvcache/kvcache_parity.mjs [backend ...]   # default cpu wasm
//   node tests/parity/kvcache/kvcache_parity.mjs compare
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import {
  atomicWriteJsonSync,
  createRuntimeEvidence,
  createRunManifest,
  finalizeRunManifest,
  readRunArtifactSync,
  readRunManifestSync,
  recordRunResult,
  removeParityOutputsSync,
  writeRunArtifactSync,
  writeRunManifestSync,
} from '../lib/artifact.mjs';
import { requirePhysicalAdapterIdentity, requirePhysicalWebGPU } from '../lib/backend.mjs';
import { kvCacheCampaignFingerprint } from '../lib/gpu_campaign.mjs';
import { captureStableResult } from '../lib/runmodel.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');
const OUT = path.join(HERE, 'out');
const CASE_ID = 'w8a8-kvcache';
const MANIFEST_FILE = path.join(OUT, 'run.json');
globalThis.window ??= globalThis; globalThis.self ??= globalThis;
globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
const nf = globalThis.fetch;
globalThis.fetch = async (u, i) => { const h = typeof u === 'string' ? u : u?.url; if (h && h.startsWith('file://')) return new Response(await fs.promises.readFile(fileURLToPath(h))); return nf(u, i); };

const SEQ = 3, WIDTH = 4, MEMORY = 2, VOCAB = 7;
const KNOWN_BACKENDS = ['cpu', 'wasm', 'webgpu'];
const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
const bundlePath = path.join(ROOT, 'dist', ver, 'volvoxai.js');
const perTensor = () => ({ scheme: 'per_tensor', scale: 0.125, zero_point: 0 });

function atomicWriteText(file, content) {
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${globalThis.process?.pid ?? 'deno'}.${randomUUID()}.tmp`,
  );
  try {
    fs.writeFileSync(temporary, content, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

function byteWeight(graph, name, rows, cols, seed = 0) {
  const v = new Int8Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) v[r * cols + c] = ((r * 3 + c * 5 + seed) % 9) - 4;
  return graph.addWeight(name, [rows, cols], 'int8', { buffer: v, quantization: { scheme: 'per_axis', axis: 0, scales: new Array(rows).fill(0.125), zero_points: new Array(rows).fill(0) } });
}
function qlinear(graph, input, name, outW, seed) {
  const inW = input.shape.at(-1);
  const weight = byteWeight(graph, `${name}.weight`, outW, inW, seed);
  const bias = graph.addWeight(`${name}.bias`, [outW], 'int32', { buffer: new Int32Array(outW) });
  return graph.addOp('QLinear', { input, weight, bias }, { out: { name, shape: [1, SEQ, outW], dtype: 'int8', quantization: perTensor() } }).out;
}
function decoderGraph(Graph) {
  const g = new Graph();
  const yIds = g.addInput('y_ids', [1, SEQ], 'int32');
  const yKeep = g.addInput('y_keep', [1, SEQ], 'int32');
  const memoryK = g.addInput('memory_k', [1, MEMORY, WIDTH], 'int8', { quantization: perTensor() });
  const memoryV = g.addInput('memory_v', [1, MEMORY, WIDTH], 'int8', { quantization: perTensor() });
  const memoryKeep = g.addInput('memory_keep', [1, MEMORY], 'int32');
  const emb = byteWeight(g, 'embedding.weight', VOCAB, WIDTH, 1);
  let h = g.addOp('QEmbedding', { input: yIds, weight: emb }, { out: { name: 'embed', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  const pos = g.addWeight('position', [1, SEQ, WIDTH], 'int8', { buffer: Int8Array.from({ length: SEQ * WIDTH }, (_, i) => (i % 5) - 2), quantization: perTensor() });
  h = g.addOp('QAdd', { a: h, b: pos }, { out: { name: 'positioned', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  const gamma = g.addWeight('norm.weight', [WIDTH], 'float32', { buffer: Float32Array.of(1, 0.75, 1.25, 0.5) });
  const beta = g.addWeight('norm.bias', [WIDTH], 'float32', { buffer: Float32Array.of(0.125, -0.125, 0.25, 0) });
  const norm = g.addOp('QLayerNorm', { input: h, weight: gamma, bias: beta }, { out: { name: 'normalized', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }, { eps: 1e-5, d_model: WIDTH }).out;
  const q = qlinear(g, norm, 'self.q', WIDTH, 2), k = qlinear(g, norm, 'self.k', WIDTH, 3), v = qlinear(g, norm, 'self.v', WIDTH, 4);
  const att = g.addOp('QSDPA', { q, k, v, mask: yKeep }, { out: { name: 'self.attention', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }, { heads: 1, causal: true, scale: 0.5 }).out;
  h = g.addOp('QAdd', { a: h, b: att }, { out: { name: 'self.residual', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  const cq = qlinear(g, h, 'cross.q', WIDTH, 5);
  const cr = g.addOp('QSDPA', { q: cq, k: memoryK, v: memoryV, mask: memoryKeep }, { out: { name: 'cross.attention', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }, { heads: 1, causal: false, scale: 0.5 }).out;
  h = g.addOp('QAdd', { a: h, b: cr }, { out: { name: 'cross.residual', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  h = qlinear(g, h, 'ffn.linear', WIDTH, 6);
  h = g.addOp('QGELU', { input: h }, { out: { name: 'ffn.gelu', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  h = g.addOp('QSiLU', { input: h }, { out: { name: 'ffn.silu', shape: [1, SEQ, WIDTH], dtype: 'int8', quantization: perTensor() } }).out;
  const logits = qlinear(g, h, 'logits', VOCAB, 7);
  // Keep the terminal projection in the selected decoder closure. Seven logits
  // deliberately exercise WebGPU's unaligned packed-byte row-copy path.
  g.setOutputs([logits.name, 'self.k', 'self.v']);
  return g;
}
// The physical self-attention K/V cache — the actual retained state a KV-cache decode
// must keep identical to a full recompute (int8, non-degenerate). Read from the graph
// outputs from the stable execution result.
const CACHE_TENSORS = ['self.k', 'self.v'];
const cacheFromOutputs = (outputs) => Object.fromEntries(
  CACHE_TENSORS.map((name) => [name, [...outputs[name]]]),
);
const decoderInputs = (yIds, yKeep) => ({ y_ids: yIds, y_keep: yKeep, memory_k: Int8Array.of(3, -2, 1, 4, -1, 2, -3, 1), memory_v: Int8Array.of(4, 1, -2, 3, -3, 2, 4, -1), memory_keep: Int32Array.of(1, 1) });
const STEPS = [[1, 3], [2, 5]]; // [position, new token id]

// Full-recompute ground truth: one causal forward with the final y_ids fills every row.
async function referenceCache(module) {
  const g = decoderGraph(module.Graph);
  const runtime = await module.VolvoxAI.createRuntime({ backends: ['cpu'], wasmUrl: wasmPath });
  const model = runtime.createModel(g);
  let compiled;
  let context;
  try {
    compiled = await model.compile({
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    const yIds = Int32Array.of(1, 0, 0), yKeep = Int32Array.of(1, 0, 0);
    for (const [p, id] of STEPS) { yIds[p] = id; yKeep[p] = 1; }
    const result = await context.execute(decoderInputs(yIds, yKeep));
    const execution = result.report;
    const captured = await captureStableResult(g, result, context, 'cpu', g.outputNames);
    context = null;
    return {
      cache: cacheFromOutputs(captured.outputs),
      runtimeEvidence: createRuntimeEvidence({
        compilation: compiled.report,
        execution,
        stableResult: captured.stableResult,
      }),
    };
  } finally {
    await context?.close();
    await compiled?.close();
    await model.close();
    await runtime.close();
  }
}

// KV-cache decode: seed(row 0) then step() per row; the retained self-K/V cache must
// end identical to a full recompute.
async function kvcacheCache(module, backend) {
  const g = decoderGraph(module.Graph);
  const runtime = await module.VolvoxAI.createRuntime({ backends: [backend], wasmUrl: wasmPath });
  const model = runtime.createModel(g);
  let compiled;
  let context;
  try {
    compiled = await model.compile({
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    if (compiled.backend !== backend) {
      throw new Error(`requested backend '${backend}' compiled on fallback '${compiled.backend}'`);
    }
    const adapterInfo = backend === 'webgpu'
      ? requirePhysicalWebGPU(compiled, 'KV-cache WebGPU parity')
      : null;
    context = await compiled.createContext({
      decode: { changedInputs: ['y_ids', 'y_keep'], rowMode: 'required' },
    });
    const yIds = Int32Array.of(1, 0, 0), yKeep = Int32Array.of(1, 0, 0);
    const seed = await context.decode.seed(decoderInputs(yIds, yKeep));
    await seed.close();
    let result = null;
    for (const [p, id] of STEPS) {
      yIds[p] = id;
      yKeep[p] = 1;
      await result?.close();
      result = await context.decode.step(decoderInputs(yIds, yKeep), { position: p });
    }
    const execution = result.report;
    const captured = await captureStableResult(g, result, context, backend, g.outputNames);
    context = null;
    return {
      cache: cacheFromOutputs(captured.outputs),
      adapterInfo,
      runtimeEvidence: createRuntimeEvidence({
        compilation: compiled.report,
        execution,
        stableResult: captured.stableResult,
      }),
    };
  } finally {
    await context?.close();
    await compiled?.close();
    await model.close();
    await runtime.close();
  }
}

const flatCache = (c) => CACHE_TENSORS.flatMap((n) => c[n] ?? []);
function validateCachePayload(payload, expectedBackend) {
  if (expectedBackend !== 'reference' && payload?.backend !== expectedBackend) {
    throw new Error(`${expectedBackend}: artifact backend label does not match`);
  }
  if (payload?.cache == null || typeof payload.cache !== 'object') {
    throw new Error(`${expectedBackend}: cache payload is missing`);
  }
  for (const name of CACHE_TENSORS) {
    const values = payload.cache[name];
    if (!Array.isArray(values) || values.length !== SEQ * WIDTH) {
      throw new Error(`${expectedBackend}: ${name} must contain exactly ${SEQ * WIDTH} values`);
    }
    for (let i = 0; i < values.length; i++) {
      if (!Number.isInteger(values[i]) || values[i] < -128 || values[i] > 127) {
        throw new Error(`${expectedBackend}: ${name}[${i}] is not int8`);
      }
    }
  }
  if (expectedBackend === 'webgpu') {
    payload.adapterInfo = requirePhysicalAdapterIdentity(
      payload.adapterInfo,
      'KV-cache WebGPU artifact',
    );
  }
  return payload;
}

async function produce(backends) {
  if (backends.length === 0 || backends.some((backend) => !KNOWN_BACKENDS.includes(backend)) ||
      new Set(backends).size !== backends.length) {
    throw new Error(`backends must be a unique non-empty subset of: ${KNOWN_BACKENDS.join(', ')}`);
  }
  const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
  fs.mkdirSync(OUT, { recursive: true });
  removeParityOutputsSync([
    'reference.json', 'cpu.json', 'wasm.json', 'webgpu.json',
    'run.json', 'kvcache_matrix.md', 'kvcache_matrix.json',
  ], { outputRoot: OUT });
  const fingerprint = kvCacheCampaignFingerprint();
  const manifest = createRunManifest({
    command: `kvcache produce ${backends.join(' ')}`,
    fingerprint,
    jobs: [
      { case: CASE_ID, tier: 'reference', expectation: 'required' },
      ...backends.map((tier) => ({ case: CASE_ID, tier, expectation: 'required' })),
    ],
    producer: {
      kind: globalThis.Deno ? 'deno' : 'node',
      backends,
      runtimeEvidenceRequired: true,
    },
  });
  let failures = 0;
  let ref = null;
  try {
    ref = await referenceCache(module);
    const payload = validateCachePayload({
      cache: ref.cache,
      runtimeEvidence: ref.runtimeEvidence,
    }, 'reference');
    writeRunArtifactSync({
      manifest,
      file: 'reference.json',
      case: CASE_ID,
      tier: 'reference',
      payload,
      metadata: { runtimeEvidence: ref.runtimeEvidence },
      outputRoot: OUT,
    });
    console.log(`  [reference] full-recompute K/V cache: ${JSON.stringify(ref.cache)}`);
  } catch (error) {
    ref = null;
    failures++;
    recordRunResult(manifest, { case: CASE_ID, tier: 'reference', status: 'error', error });
    console.error(`  [reference] ERROR ${String(error.message || error).slice(0, 200)}`);
  }
  for (const be of backends) {
    if (ref == null) {
      failures++;
      recordRunResult(manifest, {
        case: CASE_ID,
        tier: be,
        status: 'error',
        error: 'full-recompute reference was not produced',
      });
      continue;
    }
    try {
      const result = await kvcacheCache(module, be);
      const payload = validateCachePayload({
        backend: be,
        cache: result.cache,
        runtimeEvidence: result.runtimeEvidence,
        ...(result.adapterInfo ? { adapterInfo: result.adapterInfo } : {}),
      }, be);
      writeRunArtifactSync({
        manifest,
        file: `${be}.json`,
        case: CASE_ID,
        tier: be,
        payload,
        metadata: { runtimeEvidence: result.runtimeEvidence },
        outputRoot: OUT,
      });
      console.log(`  [${be}] kv-cache K/V: ${JSON.stringify(result.cache)}`);
    } catch (e) {
      failures++;
      recordRunResult(manifest, { case: CASE_ID, tier: be, status: 'error', error: e });
      console.error(`  [${be}] ERROR ${String(e.message || e).slice(0, 200)}`);
    }
  }
  finalizeRunManifest(manifest);
  writeRunManifestSync(MANIFEST_FILE, manifest, { outputRoot: OUT });
  return failures === 0 && manifest.outcome === 'success';
}

function compare(requiredBackends) {
  if (requiredBackends.length === 0 ||
      requiredBackends.some((backend) => !KNOWN_BACKENDS.includes(backend)) ||
      new Set(requiredBackends).size !== requiredBackends.length) {
    throw new Error(`required backends must be a unique non-empty subset of: ${KNOWN_BACKENDS.join(', ')}`);
  }
  let manifest;
  try {
    manifest = readRunManifestSync(MANIFEST_FILE, {
      outputRoot: OUT,
      requireComplete: true,
      requireFinalized: true,
      expectedFingerprint: kvCacheCampaignFingerprint(),
    });
  } catch (error) {
    console.error(`current kv-cache run is unavailable: ${error.message}`);
    return 2;
  }
  if (manifest.outcome !== 'success') {
    console.error('current kv-cache producer manifest did not succeed');
    return 2;
  }
  const selected = manifest.selection.jobs
    .filter((job) => job.case === CASE_ID && KNOWN_BACKENDS.includes(job.tier))
    .map((job) => job.tier);
  if (selected.length !== requiredBackends.length ||
      requiredBackends.some((backend) => !selected.includes(backend))) {
    console.error(`compare requested [${requiredBackends}] but current run produced [${selected}]`);
    return 2;
  }
  let ref;
  const current = {};
  try {
    ref = validateCachePayload(readRunArtifactSync('reference.json', manifest, {
      case: CASE_ID, tier: 'reference', outputRoot: OUT,
    }), 'reference');
    for (const backend of requiredBackends) {
      current[backend] = validateCachePayload(readRunArtifactSync(`${backend}.json`, manifest, {
        case: CASE_ID, tier: backend, outputRoot: OUT,
      }), backend);
    }
  } catch (error) {
    console.error(`current kv-cache artifact is invalid: ${error.message}`);
    return 2;
  }
  const refFlat = flatCache(ref.cache);
  const lines = ['# KV-cache decode parity (W8A8 row decode vs full recompute)', '',
    'The retained self-attention K/V cache after a seeded + stepped row decode must equal',
    'the K/V a full recompute produces — a direct check the cache holds correct rows.', '',
    `nonzero cache values: ${refFlat.filter((v) => v !== 0).length}/${refFlat.length}`, '', '| tier | K/V cache vs full-recompute |', '|---|---|'];
  let fail = 0;
  const required = new Set(requiredBackends);
  for (const be of KNOWN_BACKENDS) {
    if (!required.has(be)) { lines.push(`| ${be} | not requested |`); continue; }
    const d = current[be];
    const df = flatCache(d.cache);
    const match = df.length === refFlat.length && df.every((t, i) => t === refFlat[i]);
    lines.push(`| ${be} | ${match ? 'MATCH (exact int8)' : 'DIFF'} |`);
    if (!match) fail++;
  }
  atomicWriteJsonSync('kvcache_matrix.json', {
    runId: manifest.runId,
    fingerprint: manifest.fingerprint.digest,
    requiredBackends,
    passed: fail === 0,
    ...(current.webgpu?.adapterInfo ? { webgpuAdapterInfo: current.webgpu.adapterInfo } : {}),
  }, { outputRoot: OUT });
  atomicWriteText(path.join(OUT, 'kvcache_matrix.md'), lines.join('\n') + '\n');
  console.log(lines.join('\n'));
  if (fail > 0) { console.error(`\n${fail} required kv-cache parity check(s) failed`); return 1; }
  console.log('\nkv-cache parity OK');
  return 0;
}

function exitWith(code) {
  if (globalThis.Deno?.exit) globalThis.Deno.exit(code);
  globalThis.process?.exit(code);
  if (code !== 0) throw new Error(`process failed with exit code ${code}`);
}

const args = globalThis.Deno?.args ?? globalThis.process?.argv.slice(2) ?? [];
if (args[0] === 'compare') exitWith(compare(args.slice(1).length ? args.slice(1) : ['cpu', 'wasm']));
else if (!await produce(args.length ? args : ['cpu', 'wasm'])) exitWith(1);
