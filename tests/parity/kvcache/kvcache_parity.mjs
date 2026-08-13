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
  canonicalJson,
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
import { captureStableResult, concreteExecutionInputs } from '../lib/runmodel.mjs';

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

function decoderSnapshot(module) {
  const definitions = [];
  const weights = {};
  const quantizationByTensor = {};
  const nodes = [];
  const addWeight = (name, shape, dtype, data, quantization = null) => {
    definitions.push({ name, dtype, shape });
    weights[name] = { name, dtype, shape, data };
    if (quantization) quantizationByTensor[name] = quantization;
    return name;
  };
  const byteWeight = (name, rows, columns, seed = 0) => {
    const data = new Int8Array(rows * columns);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        data[row * columns + column] = ((row * 3 + column * 5 + seed) % 9) - 4;
      }
    }
    return addWeight(name, [rows, columns], 'int8', data, {
      scheme: 'per_axis', axis: 0,
      scales: new Array(rows).fill(0.125), zero_points: new Array(rows).fill(0),
    });
  };
  const addNode = (opType, inputs, name, shape, params = {}) => {
    nodes.push({
      id: name,
      opType,
      inputs,
      outputs: { out: { tensor: name, dtype: 'int8', shape } },
      params,
    });
    quantizationByTensor[name] = perTensor();
    return name;
  };
  const qlinear = (input, name, outputWidth, seed) => {
    const weight = byteWeight(`${name}.weight`, outputWidth, WIDTH, seed);
    const bias = addWeight(
      `${name}.bias`, [outputWidth], 'int32', new Int32Array(outputWidth),
    );
    return addNode(
      'QLinear', { input, weight, bias }, name, [1, SEQ, outputWidth], {},
    );
  };

  const inputs = {
    y_ids: { dtype: 'int32', shape: [1, SEQ] },
    y_keep: { dtype: 'int32', shape: [1, SEQ] },
    memory_k: { dtype: 'int8', shape: [1, MEMORY, WIDTH] },
    memory_v: { dtype: 'int8', shape: [1, MEMORY, WIDTH] },
    memory_keep: { dtype: 'int32', shape: [1, MEMORY] },
  };
  quantizationByTensor.memory_k = perTensor();
  quantizationByTensor.memory_v = perTensor();

  const embedding = byteWeight('embedding.weight', VOCAB, WIDTH, 1);
  let hidden = addNode(
    'QEmbedding', { input: 'y_ids', weight: embedding },
    'embed', [1, SEQ, WIDTH], {},
  );
  const position = addWeight(
    'position', [1, SEQ, WIDTH], 'int8',
    Int8Array.from({ length: SEQ * WIDTH }, (_, index) => (index % 5) - 2),
    perTensor(),
  );
  hidden = addNode('QAdd', { a: hidden, b: position },
    'positioned', [1, SEQ, WIDTH], {});
  const gamma = addWeight('norm.weight', [WIDTH], 'float32',
    Float32Array.of(1, 0.75, 1.25, 0.5));
  const beta = addWeight('norm.bias', [WIDTH], 'float32',
    Float32Array.of(0.125, -0.125, 0.25, 0));
  const normalized = addNode(
    'QLayerNorm', { input: hidden, weight: gamma, bias: beta },
    'normalized', [1, SEQ, WIDTH], { eps: 1e-5, d_model: WIDTH },
  );
  const q = qlinear(normalized, 'self.q', WIDTH, 2);
  const k = qlinear(normalized, 'self.k', WIDTH, 3);
  const v = qlinear(normalized, 'self.v', WIDTH, 4);
  const attended = addNode(
    'QSDPA', { q, k, v, mask: 'y_keep' },
    'self.attention', [1, SEQ, WIDTH], { heads: 1, causal: true, scale: 0.5 },
  );
  hidden = addNode('QAdd', { a: hidden, b: attended },
    'self.residual', [1, SEQ, WIDTH], {});
  const crossQ = qlinear(hidden, 'cross.q', WIDTH, 5);
  const crossed = addNode(
    'QSDPA', { q: crossQ, k: 'memory_k', v: 'memory_v', mask: 'memory_keep' },
    'cross.attention', [1, SEQ, WIDTH], { heads: 1, causal: false, scale: 0.5 },
  );
  hidden = addNode('QAdd', { a: hidden, b: crossed },
    'cross.residual', [1, SEQ, WIDTH], {});
  hidden = qlinear(hidden, 'ffn.linear', WIDTH, 6);
  hidden = addNode('QGELU', { input: hidden }, 'ffn.gelu', [1, SEQ, WIDTH], {});
  hidden = addNode('QSiLU', { input: hidden }, 'ffn.silu', [1, SEQ, WIDTH], {});
  const logits = qlinear(hidden, 'logits', VOCAB, 7);

  const quantizationReferences = {};
  let quantizationOrdinal = 0;
  for (const [name, metadata] of Object.entries(quantizationByTensor)) {
    const prefix = `__quantization.${quantizationOrdinal++}`;
    const scales = metadata.scheme === 'per_axis' ? metadata.scales : [metadata.scale];
    const zeroPoints = metadata.scheme === 'per_axis'
      ? metadata.zero_points
      : [metadata.zero_point];
    const scale = addWeight(
      `${prefix}.scale`, [scales.length], 'float32', Float32Array.from(scales),
    );
    const zeroPoint = addWeight(
      `${prefix}.zero_point`, [zeroPoints.length], 'int8', Int8Array.from(zeroPoints),
    );
    quantizationReferences[name] = metadata.scheme === 'per_axis'
      ? { scheme: 'per_axis', axis: metadata.axis, scale_tensor: scale,
          zero_point_tensor: zeroPoint }
      : { scheme: 'per_tensor', scale_tensor: scale, zero_point_tensor: zeroPoint };
  }

  const graph = module.parseGraphDocument({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs,
    nodes,
    // Keep the terminal projection in the selected decoder closure. Seven
    // logits deliberately exercise WebGPU's unaligned packed-byte row copy.
    outputs: [logits, 'self.k', 'self.v'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: quantizationReferences,
    },
  }, definitions);
  return module.Model.capture({ graph, weights, quantizationByTensor });
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

function requireIncrementalRowDecodeState(value, position, label) {
  if (value?.operation !== 'step' || value.mode !== 'incremental-row' ||
      value.cacheState !== 'advanced' || value.cacheGeneration !== 1 ||
      value.position !== position || value.activeSequenceLength !== position + 1) {
    throw new Error(
      `${label}: expected step/incremental-row/advanced generation 1 at position ${position} ` +
      `with active length ${position + 1}`,
    );
  }
  return {
    operation: value.operation,
    mode: value.mode,
    cacheState: value.cacheState,
    cacheGeneration: value.cacheGeneration,
    position: value.position,
    activeSequenceLength: value.activeSequenceLength,
  };
}

function requireRuntimeDecodeState(value, position, label) {
  if (value?.operation !== 'step' || value.mode !== 'incremental-row' ||
      value.cacheState !== 'advanced' || value.cacheGeneration !== 1 ||
      value.position !== position) {
    throw new Error(
      `${label}: expected step/incremental-row/advanced generation 1 at position ${position}`,
    );
  }
  return {
    operation: value.operation,
    mode: value.mode,
    cacheState: value.cacheState,
    cacheGeneration: value.cacheGeneration,
    position: value.position,
  };
}

function requireManifestRuntimeEvidence(manifest, tier, payload) {
  const result = manifest.results.find((candidate) =>
    candidate.case === CASE_ID && candidate.tier === tier);
  const metadataEvidence = result?.metadata?.runtimeEvidence;
  if (metadataEvidence == null || payload?.runtimeEvidence == null ||
      canonicalJson(metadataEvidence) !== canonicalJson(payload.runtimeEvidence)) {
    throw new Error(`${tier}: manifest runtime evidence does not match its artifact payload`);
  }
}

// Full-recompute ground truth: one causal forward with the final y_ids fills every row.
async function referenceCache(module) {
  const snapshot = decoderSnapshot(module);
  const runtime = await module.VolvoxAI.createRuntime({ backends: ['cpu'], wasmUrl: wasmPath });
  let compiled;
  let context;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    const yIds = Int32Array.of(1, 0, 0), yKeep = Int32Array.of(1, 0, 0);
    for (const [p, id] of STEPS) { yIds[p] = id; yKeep[p] = 1; }
    const result = await context.execute(concreteExecutionInputs(
      snapshot, decoderInputs(yIds, yKeep),
    ));
    const execution = result.report;
    const captured = await captureStableResult(
      snapshot.graph, result, context, 'cpu', snapshot.outputNames,
    );
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
    await runtime.close();
  }
}

// KV-cache decode: seed(row 0) then step() per row; the retained self-K/V cache must
// end identical to a full recompute.
async function kvcacheCache(module, backend) {
  const snapshot = decoderSnapshot(module);
  const runtime = await module.VolvoxAI.createRuntime({ backends: [backend], wasmUrl: wasmPath });
  let compiled;
  let context;
  try {
    compiled = await runtime.compile(snapshot, {
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
    const seed = await context.decode.seed(concreteExecutionInputs(
      snapshot, decoderInputs(yIds, yKeep),
    ));
    await seed.close();
    let result = null;
    const steps = [];
    for (const [p, id] of STEPS) {
      yIds[p] = id;
      yKeep[p] = 1;
      await result?.close();
      result = await context.decode.step(concreteExecutionInputs(
        snapshot, decoderInputs(yIds, yKeep),
      ), { position: p });
      const decodeState = requireIncrementalRowDecodeState(
        result.report.decodeState, p, `${backend} step ${p}`,
      );
      const cache = {};
      for (const name of CACHE_TENSORS) cache[name] = await result.output(name).read();
      steps.push({ position: p, decodeState, cache: cacheFromOutputs(cache) });
    }
    const execution = result.report;
    const finalPosition = STEPS.at(-1)[0];
    const decodeState = requireIncrementalRowDecodeState(
      execution.decodeState, finalPosition, `${backend} final result`,
    );
    const captured = await captureStableResult(
      snapshot.graph, result, context, backend, snapshot.outputNames,
    );
    const runtimeEvidence = createRuntimeEvidence({
      compilation: compiled.report,
      execution,
      stableResult: captured.stableResult,
    });
    requireRuntimeDecodeState(
      runtimeEvidence.execution.decodeState, finalPosition, `${backend} runtime evidence`,
    );
    context = null;
    return {
      cache: cacheFromOutputs(captured.outputs),
      steps,
      decodeState,
      adapterInfo,
      runtimeEvidence,
    };
  } finally {
    await context?.close();
    await compiled?.close();
    await runtime.close();
  }
}

const flatCache = (c) => CACHE_TENSORS.flatMap((n) => c[n] ?? []);
function validateCache(cache, label) {
  if (cache == null || typeof cache !== 'object') {
    throw new Error(`${label}: cache payload is missing`);
  }
  for (const name of CACHE_TENSORS) {
    const values = cache[name];
    if (!Array.isArray(values) || values.length !== SEQ * WIDTH) {
      throw new Error(`${label}: ${name} must contain exactly ${SEQ * WIDTH} values`);
    }
    for (let i = 0; i < values.length; i++) {
      if (!Number.isInteger(values[i]) || values[i] < -128 || values[i] > 127) {
        throw new Error(`${label}: ${name}[${i}] is not int8`);
      }
    }
  }
}

function validateCachePayload(payload, expectedBackend) {
  if (expectedBackend !== 'reference' && payload?.backend !== expectedBackend) {
    throw new Error(`${expectedBackend}: artifact backend label does not match`);
  }
  validateCache(payload?.cache, expectedBackend);
  if (expectedBackend !== 'reference') {
    if (!Array.isArray(payload.steps) || payload.steps.length !== STEPS.length) {
      throw new Error(`${expectedBackend}: every incremental step must be sealed`);
    }
    for (let index = 0; index < STEPS.length; index++) {
      const expectedPosition = STEPS[index][0];
      if (payload.steps[index]?.position !== expectedPosition) {
        throw new Error(`${expectedBackend}: incremental step ${index} has the wrong position`);
      }
      requireIncrementalRowDecodeState(
        payload.steps[index]?.decodeState,
        expectedPosition,
        `${expectedBackend} step ${expectedPosition}`,
      );
      validateCache(payload.steps[index]?.cache, `${expectedBackend} step ${expectedPosition}`);
    }
    const finalPosition = STEPS.at(-1)[0];
    const decodeState = requireIncrementalRowDecodeState(
      payload.decodeState, finalPosition, `${expectedBackend} final decode state`,
    );
    const evidenceState = requireRuntimeDecodeState(
      payload.runtimeEvidence?.execution?.decodeState,
      finalPosition,
      `${expectedBackend} runtime evidence`,
    );
    const projectedDecodeState = {
      operation: decodeState.operation,
      mode: decodeState.mode,
      cacheState: decodeState.cacheState,
      cacheGeneration: decodeState.cacheGeneration,
      position: decodeState.position,
    };
    if (canonicalJson(projectedDecodeState) !== canonicalJson(evidenceState)) {
      throw new Error(`${expectedBackend}: payload and runtime-evidence decode states differ`);
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
    requireManifestRuntimeEvidence(manifest, 'reference', payload);
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
        steps: result.steps,
        decodeState: result.decodeState,
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
      requireManifestRuntimeEvidence(manifest, be, payload);
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
    requireManifestRuntimeEvidence(manifest, 'reference', ref);
    for (const backend of requiredBackends) {
      current[backend] = validateCachePayload(readRunArtifactSync(`${backend}.json`, manifest, {
        case: CASE_ID, tier: backend, outputRoot: OUT,
      }), backend);
      requireManifestRuntimeEvidence(manifest, backend, current[backend]);
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
    const finalMatch = df.length === refFlat.length && df.every((t, i) => t === refFlat[i]);
    const stepMatch = d.steps.every(({ position, cache }) => CACHE_TENSORS.every((name) => {
      const prefixElements = (position + 1) * WIDTH;
      return cache[name].slice(0, prefixElements).every(
        (value, index) => value === ref.cache[name][index],
      );
    }));
    const match = finalMatch && stepMatch;
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
