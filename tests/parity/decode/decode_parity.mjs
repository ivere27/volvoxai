// Autoregressive decode-path parity for TinyStories-1M.
//
//   node tests/parity/decode/decode_parity.mjs [backend ...]   # default cpu-js wasm; produce token seqs
//   node tests/parity/decode/decode_parity.mjs compare         # compare tiers + PyTorch greedy generate
//
// The whole-model campaign keeps a separate explicit [1,256] maximum-capacity
// oracle. This campaign exercises the deployment contract instead: allocate one
// request-sized capacity, seed the prompt once, then retain state across steps.
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import {
  atomicWriteJsonSync,
  createParityFingerprintSync,
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
import { requirePhysicalWebGPU } from '../lib/backend.mjs';
import { captureStableResult } from '../lib/runmodel.mjs';
import {
  createTinyStoriesDecodeState,
  resolveTinyStoriesSequenceContract,
} from '../../../examples/tinystories/browser_session.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');
const OUT = path.join(HERE, 'out');
const CASE_ID = 'tinystories-decode';
const MANIFEST_FILE = path.join(OUT, 'run.json');
const TORCH_MANIFEST_FILE = path.join(OUT, 'torch_run.json');
const KNOWN_BACKENDS = ['cpu-js', 'wasm', 'webgpu'];
globalThis.window ??= globalThis; globalThis.self ??= globalThis;
globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
const nf = globalThis.fetch;
globalThis.fetch = async (u, i) => { const h = typeof u === 'string' ? u : u?.url; if (h && h.startsWith('file://')) return new Response(await fs.promises.readFile(fileURLToPath(h))); return nf(u, i); };

const MODEL = 'models/tinystories_1m';
const FIXTURE_SEQUENCE_CAPACITY = 256;
const VOCAB = 50257, EOS = 50256, PROMPT_LEN = 8, N_NEW = 16;
const REQUEST_SEQUENCE_CAPACITY = PROMPT_LEN + N_NEW;
const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const modelUrl = pathToFileURL(path.join(ROOT, MODEL, 'model.safetensors')).href;
const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
const bundlePath = path.join(ROOT, 'dist', ver, 'volvoxai.js');

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

function campaignFingerprint() {
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: path.join(ROOT, 'tests', 'parity', 'policy.json'),
    selectedCases: ['tinystories_1m'],
    sourceFiles: [
      fileURLToPath(import.meta.url),
      path.join(HERE, 'decode_torch_oracle.py'),
      path.join(HERE, '..', 'lib', 'artifact.mjs'),
      path.join(HERE, '..', 'lib', 'backend.mjs'),
      path.join(HERE, '..', 'lib', 'runmodel.mjs'),
      path.join(ROOT, 'examples', 'tinystories', 'browser_session.js'),
    ],
    buildFiles: [bundlePath, wasmPath],
    // Deno's GPU command intentionally has no --allow-run, so fingerprint the
    // exact scripts/build/model files instead of spawning git for source state.
    source: { revision: null, state: 'explicit-files', statusSha256: null },
  });
}

function selectedBackends(args) {
  if (args.some((backend) => !KNOWN_BACKENDS.includes(backend)) ||
      new Set(args).size !== args.length) {
    throw new Error(`backends must be a unique subset of: ${KNOWN_BACKENDS.join(', ')}`);
  }
  // CPU-JS is the reference and WASM is the default required portability tier.
  // Asking for WebGPU adds it to the same fresh campaign; it never reuses an
  // earlier CPU-JS reference.
  return ['cpu-js', 'wasm', ...(args.includes('webgpu') ? ['webgpu'] : [])];
}

function validateTokens(payload, backend) {
  if (payload?.backend !== backend || !Array.isArray(payload.tokens) || payload.tokens.length !== N_NEW) {
    throw new Error(`${backend}: expected exactly ${N_NEW} generated tokens`);
  }
  for (let i = 0; i < payload.tokens.length; i++) {
    const token = payload.tokens[i];
    if (!Number.isSafeInteger(token) || token < 0 || token >= VOCAB) {
      throw new Error(`${backend}: invalid token ${token} at generated position ${i}`);
    }
  }
  return payload;
}

function readTokensFixture() {
  const b = fs.readFileSync(path.join(ROOT, MODEL, 'tokens.i32'));
  const tokens = new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  if (tokens.length !== FIXTURE_SEQUENCE_CAPACITY) {
    throw new Error(
      `TinyStories oracle fixture must contain ${FIXTURE_SEQUENCE_CAPACITY} I32 tokens`,
    );
  }
  return tokens;
}

async function decode(module, backend, prompt) {
  const snapshot = module.Model.capture(
    await module.ModelLoader.load(modelUrl),
  );
  const runtime = await module.VolvoxAI.createRuntime({ backends: [backend], wasmUrl: wasmPath });
  let compiled;
  let context;
  try {
    compiled = await runtime.compile(snapshot, {
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    if (compiled.backend !== backend) {
      throw new Error(`requested backend '${backend}' compiled on '${compiled.backend || 'unknown'}'`);
    }
    if (backend === 'webgpu') requirePhysicalWebGPU(compiled, 'decode WebGPU parity');
    const contract = resolveTinyStoriesSequenceContract(snapshot);
    if (contract.vocabularySize !== VOCAB ||
        contract.maximumSequenceCapacity !== FIXTURE_SEQUENCE_CAPACITY) {
      throw new Error('TinyStories package differs from the qualified vocabulary/capacity contract');
    }
    const state = createTinyStoriesDecodeState(prompt, {
      requestedNewTokens: N_NEW,
      eosTokenId: EOS,
      contract,
    });
    if (state.sequenceCapacity !== REQUEST_SEQUENCE_CAPACITY) {
      throw new Error(
        `decode request resolved capacity ${state.sequenceCapacity}; expected ${REQUEST_SEQUENCE_CAPACITY}`,
      );
    }
    context = await compiled.createContext({
      decode: {
        changedInputs: ['tokens'],
        rowMode: 'auto',
        requireIncremental: true,
      },
    });
    const generated = [];
    let runtimeEvidence = null;
    let last = state.promptLength - 1;
    for (let step = 0; step < N_NEW; step++) {
      const result = step === 0
        ? await context.decode.seed(state.inputs, { position: last })
        : await context.decode.step(
          { tokens: state.inputs.tokens },
          { position: last },
        );
      let flat;
      if (step === N_NEW - 1) {
        const execution = result.report;
        const captured = await captureStableResult(
          snapshot.graph,
          result,
          context,
          backend,
          snapshot.outputNames,
        );
        context = null;
        flat = captured.outputs[contract.outputName];
        runtimeEvidence = createRuntimeEvidence({
          compilation: compiled.report,
          execution,
          stableResult: captured.stableResult,
        });
      } else {
        const values = await result.output(contract.outputName).read();
        flat = values instanceof Float32Array ? values : Float32Array.from(values);
        await result.close();
      }
      const expectedBytes = state.sequenceCapacity * contract.vocabularySize * 4;
      if (flat.byteLength !== expectedBytes) {
        throw new Error(`${backend}: logits output has ${flat.byteLength} bytes, expected ${expectedBytes}`);
      }
      const base = last * VOCAB;
      let best = 0, bestV = -Infinity;
      for (let c = 0; c < VOCAB; c++) {
        const value = flat[base + c];
        if (value > bestV) { bestV = value; best = c; }
      }
      generated.push(best);
      if (step + 1 < N_NEW) {
        last++;
        state.tokens[last] = best;
      }
    }
    return { tokens: generated, runtimeEvidence };
  } finally {
    await context?.close();
    await compiled?.close();
    await runtime.close();
  }
}

async function produce(backends) {
  const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
  fs.mkdirSync(OUT, { recursive: true });
  removeParityOutputsSync([
    'prompt.json', 'cpu-js.json', 'wasm.json', 'webgpu.json', 'torch.json',
    'run.json', 'torch_run.json', 'decode_matrix.md', 'decode_matrix.json',
  ], { outputRoot: OUT });
  const fingerprint = campaignFingerprint();
  const manifest = createRunManifest({
    command: `decode produce ${backends.join(' ')}`,
    fingerprint,
    jobs: [
      { case: CASE_ID, tier: 'prompt', expectation: 'required' },
      ...backends.map((tier) => ({ case: CASE_ID, tier, expectation: 'required' })),
    ],
    producer: {
      kind: globalThis.Deno ? 'deno' : 'node',
      backends,
      runtimeEvidenceTiers: backends,
      executionContract: 'execution-context-retained-seed-step-v1',
      shapedInputs: true,
      promptActiveLength: PROMPT_LEN,
      requestSequenceCapacity: REQUEST_SEQUENCE_CAPACITY,
      maximumFixtureSequenceCapacity: FIXTURE_SEQUENCE_CAPACITY,
    },
  });
  const prompt = [...readTokensFixture().slice(0, PROMPT_LEN)];
  let promptReady = false;
  try {
    writeRunArtifactSync({
      manifest,
      file: 'prompt.json',
      case: CASE_ID,
      tier: 'prompt',
      payload: {
        prompt,
        nNew: N_NEW,
        executionContract: 'execution-context-retained-seed-step-v1',
        requestSequenceCapacity: REQUEST_SEQUENCE_CAPACITY,
        maximumFixtureSequenceCapacity: FIXTURE_SEQUENCE_CAPACITY,
      },
      outputRoot: OUT,
    });
    promptReady = true;
  } catch (error) {
    recordRunResult(manifest, { case: CASE_ID, tier: 'prompt', status: 'error', error });
    console.error(`  [prompt] ERROR ${String(error.message || error).slice(0, 200)}`);
  }
  let failed = 0;
  for (const be of backends) {
    if (!promptReady) {
      failed++;
      recordRunResult(manifest, {
        case: CASE_ID,
        tier: be,
        status: 'error',
        error: 'prompt artifact was not produced',
      });
      continue;
    }
    try {
      const decoded = await decode(module, be, prompt);
      const payload = validateTokens({
        backend: be,
        tokens: decoded.tokens,
        runtimeEvidence: decoded.runtimeEvidence,
      }, be);
      writeRunArtifactSync({
        manifest,
        file: `${be}.json`,
        case: CASE_ID,
        tier: be,
        payload,
        metadata: { runtimeEvidence: decoded.runtimeEvidence },
        outputRoot: OUT,
      });
      console.log(`  [${be}] generated ${decoded.tokens.length}: ${decoded.tokens.join(',')}`);
    } catch (e) {
      failed++;
      recordRunResult(manifest, { case: CASE_ID, tier: be, status: 'error', error: e });
      console.error(`  [${be}] ERROR ${String(e.message || e).slice(0, 200)}`);
    }
  }
  finalizeRunManifest(manifest);
  writeRunManifestSync(MANIFEST_FILE, manifest, { outputRoot: OUT });
  if (!promptReady || failed > 0 || manifest.outcome !== 'success') {
    console.error(`${failed} requested decode backend(s) failed`);
    return false;
  }
  return true;
}

function compare() {
  const fingerprint = campaignFingerprint();
  const manifest = readRunManifestSync(MANIFEST_FILE, {
    outputRoot: OUT,
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: fingerprint,
  });
  if (manifest.outcome !== 'success') throw new Error('current decode producer manifest did not succeed');
  const selected = manifest.selection.jobs
    .filter((job) => job.case === CASE_ID && KNOWN_BACKENDS.includes(job.tier))
    .map((job) => job.tier);
  if (!selected.includes('cpu-js') || !selected.includes('wasm') ||
      selected.some((tier) => !KNOWN_BACKENDS.includes(tier))) {
    throw new Error('current decode manifest must require CPU-JS and WASM, with optional WebGPU');
  }
  const prompt = readRunArtifactSync('prompt.json', manifest, {
    case: CASE_ID, tier: 'prompt', outputRoot: OUT,
  });
  if (!Array.isArray(prompt?.prompt) || prompt.prompt.length !== PROMPT_LEN ||
      prompt.nNew !== N_NEW ||
      prompt.executionContract !== 'execution-context-retained-seed-step-v1' ||
      prompt.requestSequenceCapacity !== REQUEST_SEQUENCE_CAPACITY ||
      prompt.maximumFixtureSequenceCapacity !== FIXTURE_SEQUENCE_CAPACITY) {
    throw new Error('current decode prompt artifact has the wrong shape');
  }
  const current = {};
  for (const tier of selected) {
    current[tier] = validateTokens(readRunArtifactSync(`${tier}.json`, manifest, {
      case: CASE_ID, tier, outputRoot: OUT,
    }), tier);
  }
  const cpu = current['cpu-js'];
  const rows = [];
  let fail = 0;
  for (const tier of selected.filter((candidate) => candidate !== 'cpu-js')) {
    const d = current[tier];
    const match = d.tokens.length === cpu.tokens.length && d.tokens.every((t, i) => t === cpu.tokens[i]);
    const firstDiff = d.tokens.findIndex((t, i) => t !== cpu.tokens[i]);
    rows.push(`| ${tier} vs cpu-js | ${match ? 'MATCH' : `DIFF@${firstDiff}`} |`);
    if (!match) fail++;
  }

  const torchArtifactExists = fs.existsSync(path.join(OUT, 'torch.json'));
  const torchManifestExists = fs.existsSync(TORCH_MANIFEST_FILE);
  if (torchArtifactExists !== torchManifestExists) {
    throw new Error('optional Torch result is incomplete; artifact and manifest must appear together');
  }
  if (torchManifestExists) {
    const torchManifest = readRunManifestSync(TORCH_MANIFEST_FILE, {
      outputRoot: OUT,
      requireComplete: true,
      requireFinalized: true,
      expectedFingerprint: fingerprint,
    });
    if (torchManifest.outcome !== 'success' || torchManifest.producer?.parentRunId !== manifest.runId) {
      throw new Error('Torch result does not belong to the current decode producer run');
    }
    const torch = validateTokens(readRunArtifactSync('torch.json', torchManifest, {
      case: CASE_ID, tier: 'torch', outputRoot: OUT,
    }), 'torch');
    const match = torch.tokens.length === cpu.tokens.length && torch.tokens.every((token, i) => token === cpu.tokens[i]);
    const firstDiff = torch.tokens.findIndex((token, i) => token !== cpu.tokens[i]);
    rows.push(`| torch vs cpu-js | ${match ? 'MATCH' : `DIFF@${firstDiff}`} |`);
    if (!match) fail++;
  } else {
    rows.push('| torch vs cpu-js | — (optional oracle absent) |');
  }
  const lines = ['# Autoregressive decode-path parity (TinyStories greedy)', '',
    `prompt ${PROMPT_LEN} tokens, ${N_NEW} new; exact token-sequence match vs pure-JS cpu-js (torch = external).`, '',
    `run: ${manifest.runId}`, `cpu-js: ${cpu.tokens.join(',')}`, '', '| comparison | result |', '|---|---|', ...rows];
  atomicWriteJsonSync('decode_matrix.json', { runId: manifest.runId, rows, fail }, { outputRoot: OUT });
  atomicWriteText(path.join(OUT, 'decode_matrix.md'), lines.join('\n') + '\n');
  console.log(lines.join('\n'));
  if (fail > 0) { console.error(`\n${fail} decode parity check(s) failed`); return 1; }
  console.log('\ndecode parity OK');
  return 0;
}

function exitWith(code) {
  if (globalThis.Deno?.exit) globalThis.Deno.exit(code);
  globalThis.process?.exit(code);
  if (code !== 0) throw new Error(`process failed with exit code ${code}`);
}

const args = globalThis.Deno?.args ?? globalThis.process?.argv.slice(2) ?? [];
try {
  if (args[0] === 'compare') {
    if (args.length !== 1) throw new Error('decode compare takes no backend arguments; it uses the current run manifest');
    exitWith(compare());
  } else {
    const backends = selectedBackends(args);
    if (!await produce(backends)) exitWith(1);
  }
} catch (error) {
  console.error(`decode parity ERROR: ${String(error.message || error).slice(0, 400)}`);
  exitWith(2);
}
