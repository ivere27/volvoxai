// Level 1-B / 2-B: cross-tier BACKWARD (gradient) parity.
//
//   node tests/parity/backward/run_backward.mjs           # produce cpu+wasm grads + dump tensors
//   node tests/parity/backward/run_backward.mjs compare   # compare cpu/wasm/torch grad signatures
//
// The engine ships autograd + optimizer + loss but nothing checked gradients across
// tiers or against an external oracle. Each case runs one forward + cross-entropy +
// backward via Runtime -> Model -> Trainer on cpu and wasm (webgpu on a GPU
// box), and dumps the exact weights/inputs/targets so backward_torch_oracle.py can
// compute the PyTorch reference. Gates wasm vs cpu (consistency) + both vs PyTorch.
import { signature, compareSignature } from '../lib/extract.mjs';
import {
  atomicWriteJsonSync,
  createParityFingerprintSync,
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
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');
const OUT = path.join(HERE, 'out');
const KNOWN_TIERS = ['cpu', 'wasm', 'webgpu'];
const RESULT_TIERS = [...KNOWN_TIERS, 'torch'];
const MANIFEST_DIR = 'manifests';

function atomicWriteText(file, content) {
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, content, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}
// Browser-global shims + file:// fetch shim so the full bundle (and WebGPUAutograd
// under Deno on a GPU box) load cleanly headless.
globalThis.window ??= globalThis;
globalThis.self ??= globalThis;
globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
const nf = globalThis.fetch;
globalThis.fetch = async (u, i) => { const h = typeof u === 'string' ? u : u?.url; if (h && h.startsWith('file://')) return new Response(await fs.promises.readFile(fileURLToPath(h))); return nf(u, i); };

const prod = (s) => s.reduce((a, b) => a * b, 1);
function seeded(n, seed, lo = -1, hi = 1) { const a = new Float32Array(n); let x = seed >>> 0 || 1; const span = hi - lo; for (let i = 0; i < n; i++) { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; a[i] = lo + ((x >>> 8) / 0x1000000) * span; } return a; }

// Declarative cases; the same spec drives the JS graph and the PyTorch oracle.
const CASES = [
  { id: 'linear', arch: 'linear', B: 4, D: 8, C: 5,
    input: { x: { shape: [4, 8], seed: 33 } },
    weights: { W: { shape: [8, 5], seed: 11 }, b: { shape: [5], seed: 22 } },
    targets: [1, 3, 0, 4], trainable: ['W', 'b'] },
  { id: 'mlp', arch: 'mlp', B: 4, D: 8, H: 6, C: 5,
    input: { x: { shape: [4, 8], seed: 45 } },
    weights: { W1: { shape: [8, 6], seed: 41 }, b1: { shape: [6], seed: 42 }, W2: { shape: [6, 5], seed: 43 }, b2: { shape: [5], seed: 44 } },
    targets: [2, 0, 4, 1], trainable: ['W1', 'b1', 'W2', 'b2'] },
  { id: 'layernorm', arch: 'layernorm', B: 4, D: 6, C: 5,
    input: { x: { shape: [4, 6], seed: 55 } },
    weights: { g: { shape: [6], seed: 51, lo: 0.5, hi: 1.5 }, be: { shape: [6], seed: 52 }, W: { shape: [6, 5], seed: 53 }, b: { shape: [5], seed: 54 } },
    targets: [0, 3, 2, 4], trainable: ['g', 'be', 'W', 'b'] },
];

const LR = 0.1; // SGD learning rate for the optimizer-step check

function buildGraph(Graph, c) {
  const g = new Graph();
  const [xName, xd] = Object.entries(c.input)[0];
  const x = g.addInput(xName, xd.shape);
  const xbuf = seeded(prod(xd.shape), xd.seed, xd.lo, xd.hi);
  const w = {}, dump = { [xName]: { buf: xbuf, shape: xd.shape } };
  for (const [name, ws] of Object.entries(c.weights)) {
    // Pass the buffer into addWeight (not set afterward) so GPU tiers upload the
    // right values at allocation time, not a stale/zero initial buffer.
    const buf = seeded(prod(ws.shape), ws.seed, ws.lo, ws.hi);
    const t = g.addWeight(name, ws.shape, 'float32', buf);
    w[name] = t; dump[name] = { buf, shape: ws.shape };
  }
  let out;
  if (c.arch === 'linear') out = g.addOp('Linear', { input: x, weight: w.W, bias: w.b }, { out: [c.B, c.C] }).out;
  else if (c.arch === 'mlp') {
    const h1 = g.addOp('Linear', { input: x, weight: w.W1, bias: w.b1 }, { out: [c.B, c.H] }).out;
    const a = g.addOp('GELU', { input: h1 }, { out: [c.B, c.H] }).out;
    out = g.addOp('Linear', { input: a, weight: w.W2, bias: w.b2 }, { out: [c.B, c.C] }).out;
  } else if (c.arch === 'layernorm') {
    const ln = g.addOp('LayerNorm', { input: x, weight: w.g, bias: w.be }, { out: [c.B, c.D] }, { eps: 1e-5 }).out;
    out = g.addOp('Linear', { input: ln, weight: w.W, bias: w.b }, { out: [c.B, c.C] }).out;
  }
  g.setOutputs([out.name]);
  return { g, xName, xbuf, dump };
}

function manifestFile(tier) {
  return path.join(MANIFEST_DIR, `${tier}.json`);
}

function artifactFile(caseId, tier) {
  return `${caseId}.${tier}.json`;
}

function fixtureFiles(c) {
  return [
    path.join(c.id, 'meta.json'),
    ...Object.keys(c.input).map((name) => path.join(c.id, `${name}.f32`)),
    ...Object.keys(c.weights).map((name) => path.join(c.id, `${name}.f32`)),
  ];
}

function exactOutputFiles(tiers, { includeFixtures = false, includeReport = false } = {}) {
  const files = [];
  for (const tier of tiers) {
    files.push(manifestFile(tier));
    for (const c of CASES) files.push(artifactFile(c.id, tier));
  }
  if (includeFixtures) for (const c of CASES) files.push(...fixtureFiles(c));
  if (includeReport) files.push('backward_matrix.md');
  return files;
}

function refreshFixtures(Graph) {
  removeParityOutputsSync(
    CASES.flatMap((c) => fixtureFiles(c)),
    { outputRoot: OUT },
  );
  for (const c of CASES) {
    const { xName, dump } = buildGraph(Graph, c);
    const dir = path.join(OUT, c.id);
    fs.mkdirSync(dir, { recursive: true });
    for (const [name, tensor] of Object.entries(dump)) {
      fs.writeFileSync(
        path.join(dir, `${name}.f32`),
        Buffer.from(tensor.buf.buffer, tensor.buf.byteOffset, tensor.buf.byteLength),
      );
    }
    atomicWriteJsonSync(path.join(c.id, 'meta.json'), {
      id: c.id,
      arch: c.arch,
      dims: Object.fromEntries(
        Object.entries({ B: c.B, D: c.D, H: c.H, C: c.C }).filter(([, value]) => value != null),
      ),
      input: xName,
      targets: c.targets,
      trainable: c.trainable,
      shapes: Object.fromEntries(Object.entries(dump).map(([name, tensor]) => [name, tensor.shape])),
    }, { outputRoot: OUT });
  }
}

function backwardFingerprint(version) {
  return createParityFingerprintSync({
    repoRoot: ROOT,
    policyFile: path.join(ROOT, 'tests', 'parity', 'policy.json'),
    selectedCases: [],
    fixtureFiles: CASES.flatMap((c) => fixtureFiles(c).map((file) => path.join(OUT, file))),
    buildFiles: [
      path.join(ROOT, 'dist', version, 'volvoxai.full.js'),
      path.join(ROOT, 'dist', version, 'volvoxai.full.wasm'),
    ],
    sourceFiles: [
      fileURLToPath(import.meta.url),
      path.join(HERE, 'backward_torch_oracle.py'),
      path.join(HERE, '..', 'external', 'sigutil.py'),
      path.join(HERE, '..', 'lib', 'artifact.mjs'),
      path.join(HERE, '..', 'lib', 'extract.mjs'),
      path.join(HERE, '..', 'lib', 'backend.mjs'),
    ],
    // Deno's normal WebGPU invocation has no --allow-run permission for Git.
    // The exact harness files above are stronger and portable across hosts.
    source: { revision: null, state: 'explicit-files', statusSha256: null },
  });
}

function tierJobs(tier) {
  return CASES.map((c) => ({ case: c.id, tier, expectation: 'required' }));
}

function readTierManifest(tier, fingerprint, runId) {
  const manifest = readRunManifestSync(manifestFile(tier), {
    outputRoot: OUT,
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: fingerprint,
    expectedRunId: runId,
  });
  if (manifest.outcome !== 'success') {
    throw new Error(`${tier} backward producer recorded an error`);
  }
  return manifest;
}

function readTierResults(tier, manifest) {
  return Object.fromEntries(CASES.map((c) => [
    c.id,
    readRunArtifactSync(artifactFile(c.id, tier), manifest, {
      case: c.id,
      tier,
      outputRoot: OUT,
    }),
  ]));
}

async function produce(tiers) {
  const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
  const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.full.wasm'); // training exports live in the full profile
  const requested = [...new Set(tiers)];
  for (const tier of requested) {
    if (!KNOWN_TIERS.includes(tier)) throw new Error(`unknown backward parity tier: ${tier}`);
  }
  if (requested.length === 0) throw new Error('at least one backward parity tier is required');

  const attachWebgpu = requested.length === 1 && requested[0] === 'webgpu';
  const invalidatedTiers = attachWebgpu ? requested : RESULT_TIERS;
  removeParityOutputsSync(exactOutputFiles(invalidatedTiers, {
    includeFixtures: !attachWebgpu,
    includeReport: true,
  }), { outputRoot: OUT });
  const full = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.full.js')).href);
  refreshFixtures(full.Graph);
  const fingerprint = backwardFingerprint(ver);

  let campaignRunId;
  if (attachWebgpu) {
    const cpuManifest = readTierManifest('cpu', fingerprint);
    readTierManifest('wasm', fingerprint, cpuManifest.runId);
    campaignRunId = cpuManifest.runId;
  }

  const manifests = new Map();
  for (const tier of requested) {
    const manifest = createRunManifest({
      command: `backward produce ${tier}`,
      fingerprint,
      jobs: tierJobs(tier),
      producer: { kind: globalThis.Deno ? 'deno' : 'node', backend: tier },
      ...(campaignRunId ? { runId: campaignRunId } : {}),
    });
    campaignRunId ??= manifest.runId;
    manifests.set(tier, manifest);
  }

  const runtimes = new Map();
  const initializationErrors = new Map();
  let failures = 0;
  for (const tier of requested) {
    let runtime;
    let probeModel;
    let probeCompiled;
    try {
      runtime = await full.VolvoxAI.createRuntime({ backends: [tier], wasmUrl: wasmPath });
      const probe = buildGraph(full.Graph, CASES[0]);
      probeModel = runtime.createModel(probe.g);
      probeCompiled = await probeModel.compile({
        backend: { mode: 'require', backend: tier, operatorFallback: 'forbid' },
      });
      if (probeCompiled.backend !== tier) {
        throw new Error(`requested backend '${tier}' compiled on '${probeCompiled.backend || 'unknown'}'`);
      }
      if (tier === 'webgpu') requirePhysicalWebGPU(probeCompiled, 'backward WebGPU parity');
      runtimes.set(tier, runtime);
    } catch (e) {
      failures++;
      const message = String(e.message || e).slice(0, 200);
      initializationErrors.set(tier, message);
      console.error(`  [${tier}] initialization ERROR ${message}`);
    } finally {
      await probeCompiled?.close().catch(() => undefined);
      await probeModel?.close().catch(() => undefined);
      if (initializationErrors.has(tier)) await runtime?.close().catch(() => undefined);
    }
  }
  const runTier = async (tier, graph, options, { commit = false } = {}) => {
    const runtime = runtimes.get(tier);
    if (!runtime) throw new Error(`training runtime '${tier}' is unavailable`);
    const model = runtime.createModel(graph);
    let trainer;
    try {
      trainer = await full.VolvoxAI.createTrainer(model, {
        backend: tier,
        wasmUrl: wasmPath,
      });
      if (trainer.backend !== tier) {
        throw new Error(`requested training backend '${tier}' created '${trainer.backend || 'unknown'}'`);
      }
      const result = await trainer.trainStep(options);
      if (commit) await trainer.commit();
      const checkpoint = await trainer.exportCheckpoint({ includeOptimizerState: false });
      return {
        result,
        trainedGraph: full.importModelCheckpoint(checkpoint).graph,
      };
    } finally {
      await trainer?.close();
      await model.close();
    }
  };
  for (const c of CASES) {
    for (const tier of requested) {
      const manifest = manifests.get(tier);
      if (initializationErrors.has(tier)) {
        recordRunResult(manifest, {
          case: c.id,
          tier,
          status: 'error',
          error: `backend initialization failed: ${initializationErrors.get(tier)}`,
        });
        continue;
      }
      const { g, xName, xbuf } = buildGraph(full.Graph, c);
      try {
        const { result: res } = await runTier(tier, g, { inputs: { [xName]: xbuf }, targets: c.targets, trainableTensors: c.trainable, updateMode: 'sgd', optimizer: { learningRate: 0 } });
        const sigs = {};
        for (const name of c.trainable) {
          sigs[name] = signature(res.gradients.get(name), { topk: 0, shape: c.weights[name].shape });
        }
        // One SGD step (lr>0) on a fresh graph: compare the *updated weights* — exercises
        // each tier's optimizer-apply path (a GPU update shader can be wrong independent
        // of the gradient), not just gradient computation.
        const step = buildGraph(full.Graph, c);
        const { trainedGraph } = await runTier(
          tier,
          step.g,
          { inputs: { [step.xName]: step.xbuf }, targets: c.targets, trainableTensors: c.trainable, updateMode: 'sgd', optimizer: { learningRate: LR } },
          { commit: true },
        );
        const stepSigs = {};
        for (const name of c.trainable) {
          stepSigs[name] = signature(trainedGraph.getTensor(name).buffer, { topk: 0, shape: c.weights[name].shape });
        }
        writeRunArtifactSync({
          manifest,
          file: artifactFile(c.id, tier),
          case: c.id,
          tier,
          payload: { id: c.id, tier, loss: res.loss, sigs, stepSigs },
          outputRoot: OUT,
        });
        console.log(`  [${tier}] ${c.id}: loss=${res.loss.toFixed(6)} grads + sgd-step(lr=${LR})`);
      } catch (e) {
        failures++;
        const message = String(e.message || e).slice(0, 200);
        recordRunResult(manifest, { case: c.id, tier, status: 'error', error: message });
        console.error(`  [${tier}] ${c.id}: ERROR ${message}`);
      }
    }
  }
  const produced = [];
  for (const tier of requested) {
    const manifest = finalizeRunManifest(manifests.get(tier));
    writeRunManifestSync(manifestFile(tier), manifest, { outputRoot: OUT });
    if (manifest.outcome === 'success') produced.push(tier);
  }
  await Promise.allSettled([...runtimes.values()].map((runtime) => runtime.close()));
  console.log(
    `${produced.length ? `produced ${produced.join('+')}` : 'produced no'} backward signatures ->`,
    path.relative(ROOT, OUT),
  );
  if (failures > 0) {
    console.error(`${failures} requested backward producer operation(s) failed`);
    if (globalThis.Deno?.exit) globalThis.Deno.exit(1);
    process.exitCode = 1;
  }
}

function compare() {
  const policy = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'policy.json'), 'utf8'));
  const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
  const fingerprint = backwardFingerprint(ver);
  const cpuManifest = readTierManifest('cpu', fingerprint);
  const runId = cpuManifest.runId;
  const manifests = {
    cpu: cpuManifest,
    wasm: readTierManifest('wasm', fingerprint, runId),
  };
  for (const tier of ['webgpu', 'torch']) {
    if (fs.existsSync(path.join(OUT, manifestFile(tier)))) {
      manifests[tier] = readTierManifest(tier, fingerprint, runId);
    }
  }
  const results = Object.fromEntries(
    Object.entries(manifests).map(([tier, manifest]) => [tier, readTierResults(tier, manifest)]),
  );
  let fail = 0, ran = 0;
  const lines = ['# Backward + optimizer-step cross-tier parity', '',
    'oracle: pure-JS (cpu). wasm/webgpu vs cpu gate (when present); torch = external correctness (gates cpu).'];
  // Two tables: gradients (`sigs`) and one SGD step's updated weights (`stepSigs`).
  for (const [key, title] of [['sigs', 'Gradients'], ['stepSigs', `Updated weights after one SGD step (lr=${LR})`]]) {
    lines.push('', `## ${title}`, '', '| case | tensor | wasm vs cpu | webgpu vs cpu | torch (ext) |', '|---|---|---|---|---|');
    for (const c of CASES) {
      const cpu = results.cpu[c.id];
      for (const name of c.trainable) {
        const cell = { wasm: '—', webgpu: '—', torch: '—' };
        for (const tier of ['wasm', 'webgpu']) {
          const data = results[tier]?.[c.id];
          if (data) {
            ran++;
            const r = compareSignature(data[key]?.[name], cpu[key]?.[name], policy.tolerances.fp32);
            cell[tier] = `${r.pass ? 'ok' : 'FAIL'} ${r.numeric.maxAbs.toExponential(1)}`;
            if (!r.pass) fail++;
          }
        }
        const torch = results.torch?.[c.id];
        if (torch) {
          ran++;
          const r = compareSignature(cpu[key]?.[name], torch[key]?.[name], policy.tolerances.external);
          cell.torch = `${r.pass ? 'ok' : 'FAIL'} ${r.numeric.maxAbs.toExponential(1)}`;
          if (!r.pass) fail++;
        }
        lines.push(`| ${c.id} | ${name} | ${cell.wasm} | ${cell.webgpu} | ${cell.torch} |`);
      }
    }
  }
  atomicWriteText(path.join(OUT, 'backward_matrix.md'), lines.join('\n') + '\n');
  console.log(lines.join('\n'));
  if (ran === 0) { console.error('\nno backward results — run produce first'); process.exit(2); }
  if (fail > 0) { console.error(`\n${fail} backward parity check(s) failed`); process.exit(1); }
  console.log('\nbackward parity OK');
}

const args = process.argv.slice(2);
if (args[0] === 'compare') compare();
else await produce(args.length ? args : ['cpu', 'wasm']); // e.g. `produce webgpu` on a GPU box
