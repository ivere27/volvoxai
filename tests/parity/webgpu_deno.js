// Real-GPU headless WebGPU parity, run under Deno. One harness for every policy
// model, single- or multi-output.
//
//   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi --allow-run=git \
//     tests/parity/webgpu_deno.js [model ...]           # default: all policy models
//
// Why Deno: its built-in WebGPU (wgpu) does *surfaceless* Vulkan compute, so it
// drives a real GPU with no display and no root. Headless Chrome can't on a bare
// server (its Vulkan needs a WSI surface -> navigator.gpu falls back to SwiftShader).
//
// For each model it runs cpu (pure-JS oracle), wasm, and webgpu, then gates webgpu
// (and wasm) against cpu per output at the model's dtype tolerance. Every declared
// output is read through its stable ExecutionResult tensor. When an external oracle signature exists
// (out/<model>.onnx.json), webgpu is also gated against it at `external` tol.
import { signature, compareSignature } from './lib/extract.mjs';
import { formatAdapterIdentity, requirePhysicalWebGPU } from './lib/backend.mjs';
import {
  buildInputs,
  captureStableResult,
  concreteExecutionInputs,
  runtimeFailureMessage,
} from './lib/runmodel.mjs';
import {
  createRuntimeEvidence,
  createParityFingerprintSync,
  createRunManifest,
  finalizeRunManifest,
  readRunArtifactSync,
  readRunManifestSync,
  recordRunResult,
  removeParityOutputsSync,
  writeRunArtifactSync,
  writeRunManifestSync,
} from './lib/artifact.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
globalThis.window ??= globalThis;
globalThis.self ??= globalThis;
globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => {
  const href = typeof url === 'string' ? url : url?.url;
  if (href && href.startsWith('file://')) return new Response(await fs.promises.readFile(fileURLToPath(href)));
  return nativeFetch(url, init);
};

const sliceForSpec = (flat, spec) => spec.kind === 'row' ? flat.subarray(spec.row * spec.cols, spec.row * spec.cols + spec.cols) : flat;

const policy = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/parity/policy.json'), 'utf8'));
const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
const OUT = path.join(ROOT, 'tests/parity/out');
const WEBGPU_MANIFEST = path.join(OUT, 'manifests/whole-webgpu.json');

function atomicWriteBytes(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${globalThis.process?.pid ?? 'deno'}.tmp`);
  try {
    fs.rmSync(temporary, { force: true });
    fs.writeFileSync(temporary, Buffer.from(value.buffer, value.byteOffset, value.byteLength), { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

function generatedFixtureFiles(names) {
  return names.flatMap((name) => (policy.models[name].inputs || [])
    .filter((input) => input.gen)
    .map((input) => path.join(ROOT, 'tests/parity/out/fixtures', name,
      `${input.name}.${input.dtype === 'f32' ? 'f32' : input.dtype}`)));
}

function materializeFixtures(names) {
  for (const name of names) {
    const model = policy.models[name];
    const values = buildInputs(model);
    for (const input of model.inputs.filter((candidate) => candidate.gen)) {
      atomicWriteBytes(path.join(OUT, 'fixtures', name,
        `${input.name}.${input.dtype === 'f32' ? 'f32' : input.dtype}`),
      values[input.name].data);
    }
  }
}

function webgpuFingerprint(names) {
  return createParityFingerprintSync({
    selectedCases: names,
    fixtureFiles: generatedFixtureFiles(names),
    buildFiles: [path.join(ROOT, 'dist', ver, 'volvoxai.js')],
  });
}

function loadCurrentExternalManifest() {
  const entries = Object.entries(policy.models).filter(([, model]) =>
    model.external?.onnx || model.external?.source === 'pytorch');
  const manifestFile = path.join(ROOT, 'tests/parity/out/manifests/whole-external.json');
  if (!fs.existsSync(manifestFile)) return null;
  const fingerprint = createParityFingerprintSync({
    selectedCases: entries.map(([name]) => name),
    fixtureFiles: generatedFixtureFiles(entries.map(([name]) => name)),
    buildFiles: [
      path.join(ROOT, 'tests/parity/external/onnx_oracle.py'),
      path.join(ROOT, 'tests/parity/external/tinystories_torch_oracle.py'),
      path.join(ROOT, 'tests/parity/external/sigutil.py'),
    ],
    extraModelFiles: entries.flatMap(([, model]) => model.external?.onnx
      ? [path.join(ROOT, model.external.onnx)] : []),
  });
  return readRunManifestSync(manifestFile, {
    requireComplete: true,
    requireFinalized: true,
    expectedFingerprint: fingerprint,
  });
}

let externalManifest = null;
try {
  externalManifest = loadCurrentExternalManifest();
} catch (error) {
  console.error(`external manifest invalid: ${String(error.message || error).slice(0, 300)}`);
  Deno.exit(1);
}

// Run one model on one backend; return { outputs: {name: {flat, len}}, ms }.
async function run(model, backend) {
  const modelUrl = pathToFileURL(path.join(ROOT, model.dir, 'model.safetensors')).href;
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
    const adapterInfo = backend === 'webgpu'
      ? requirePhysicalWebGPU(compiled, 'whole-model WebGPU parity')
      : null;
    context = await compiled.createContext();
    const inputs = concreteExecutionInputs(snapshot, buildInputs(model));
    const warmup = await context.execute(inputs);
    await warmup.close();
    const t0 = performance.now();
    const result = await context.execute(inputs);
    const ms = performance.now() - t0;
    const execution = result.report;
    const captured = await captureStableResult(
      snapshot.graph,
      result,
      context,
      backend,
      model.outputs.map((output) => output.name),
    );
    const outputs = Object.fromEntries(Object.entries(captured.outputs).map(([name, flat]) => [
      name,
      { flat, len: flat.length },
    ]));
    return {
      outputs,
      ms,
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
    await runtime.close();
  }
}

const wanted = Deno.args.length ? [...new Set(Deno.args)] : Object.keys(policy.models);
for (const name of wanted) {
  if (!policy.models[name]) {
    console.error(`unknown parity model: ${name}`);
    Deno.exit(2);
  }
}
materializeFixtures(wanted);
removeParityOutputsSync([
  WEBGPU_MANIFEST,
  ...wanted.map((name) => path.join(OUT, `${name}.webgpu.json`)),
]);
const webgpuManifest = createRunManifest({
  command: 'webgpu produce',
  fingerprint: webgpuFingerprint(wanted),
  jobs: wanted.map((name) => ({ case: name, tier: 'webgpu', expectation: 'required' })),
  producer: {
    kind: 'deno',
    backend: 'webgpu',
    strictBackend: true,
    physicalAdapterRequired: true,
    runtimeEvidenceRequired: true,
  },
});
let failed = 0;
for (const name of wanted) {
  const model = policy.models[name];
  const modelErrors = [];
  const tol = policy.tolerances[model.dtype] || policy.tolerances.fp32;
  console.log(`\n## ${name} [${model.dtype}]`);
  const res = {};
  for (const be of ['cpu', 'wasm', 'webgpu']) {
    try {
      res[be] = await run(model, be);
      const lens = model.outputs.map((s) => `${s.name}=${res[be].outputs[s.name].len}`).join(' ');
      console.log(`  [${be}] ok ${res[be].ms.toFixed(1)}ms ${lens}`);
      if (res[be].adapterInfo) console.log(`    adapter: ${formatAdapterIdentity(res[be].adapterInfo)}`);
    } catch (e) {
      const message = runtimeFailureMessage(e).slice(0, 1000);
      console.log(`  [${be}] ERROR ${message}`);
      modelErrors.push(`${be}: ${message}`);
    }
  }
  if (!res.cpu) modelErrors.push('strict CPU reference was not produced');

  // gate each output vs the pure-JS oracle. A true-int8 GPU backend legitimately
  // differs from the int8-folded-to-fp32 CPU reference in raw magnitudes (and on a
  // *random* input the argmax is meaningless), so int8-GPU-vs-CPU is DIAGNOSTIC, not
  // a gate -- the real int8 GPU check is GPU-vs-GPU (webgpu == opengl == vulkan,
  // bit-identical) on a real image; see the native cross-GPU comparison / README.
  const sig = (r, spec) => signature(sliceForSpec(r.outputs[spec.name].flat, spec), {
    topk: spec.topk ?? 10,
    shape: spec.shape,
    sampleAxes: spec.sampleAxes ?? [],
  });
  for (const be of ['wasm', 'webgpu']) {
    if (!res.cpu || !res[be]) { console.log(`  [${be}] SKIP`); continue; }
    const diag = model.dtype === 'int8' && be === 'webgpu';
    for (const spec of model.outputs) {
      const c = compareSignature(sig(res[be], spec), sig(res.cpu, spec), tol);
      const tag = diag ? '[diag]' : (c.pass ? 'PASS' : 'FAIL');
      console.log(`  ${be} vs cpu  ${spec.name}: ${tag} maxAbs=${c.numeric.maxAbs}${spec.topk ? ` top1=${c.task.top1Match}` : ''}${diag ? '  (true-int8 vs folded-fp32; gate is GPU-vs-GPU)' : ''}`);
      if (!diag && !c.pass) {
        console.log(`     problems: ${c.problems.join('; ')}`);
        modelErrors.push(`${be}/${spec.name}: ${c.problems.join('; ')}`);
      }
    }
  }

  // external oracle (PyTorch/ONNX signature) vs webgpu, fp32 only (the int8 external
  // gate needs a real image + decision metric -> handled by the native GPU path).
  const externalTier = model.external?.onnx ? 'onnx'
    : model.external?.source === 'pytorch' ? 'torch' : null;
  const oraclePath = externalTier
    ? path.join(ROOT, 'tests/parity/out', `${name}.${externalTier}.json`)
    : null;
  const externalResult = externalManifest?.results.find((entry) =>
    entry.case === name && entry.tier === externalTier && entry.status === 'success');
  if (res.webgpu && model.dtype !== 'int8' && oraclePath && externalResult) {
    const ext = readRunArtifactSync(oraclePath, externalManifest, {
      case: name,
      tier: externalTier,
    });
    for (const spec of model.outputs) {
      const gold = ext.sigs?.[spec.name];
      if (!gold) continue;
      const c = compareSignature(sig(res.webgpu, spec), gold, policy.tolerances.external);
      console.log(`  webgpu vs ${ext.backend || 'external'}  ${spec.name}: ${c.pass ? 'PASS' : 'FAIL'} maxAbs=${c.numeric.maxAbs}${spec.topk ? ` top1=${c.task.top1Match}` : ''}`);
      if (!c.pass) {
        console.log(`     problems: ${c.problems.join('; ')}`);
        modelErrors.push(`${externalTier}/${spec.name}: ${c.problems.join('; ')}`);
      }
    }
  }
  if (modelErrors.length === 0 && res.webgpu) {
    const sigs = Object.fromEntries(model.outputs.map((spec) => [spec.name, sig(res.webgpu, spec)]));
    writeRunArtifactSync({
      manifest: webgpuManifest,
      file: path.join(OUT, `${name}.webgpu.json`),
      case: name,
      tier: 'webgpu',
      payload: {
        model: name,
        backend: 'webgpu',
        ms: Number(res.webgpu.ms.toFixed(2)),
        adapterInfo: res.webgpu.adapterInfo,
        sigs,
        runtimeEvidence: res.webgpu.runtimeEvidence,
      },
      metadata: {
        adapterInfo: res.webgpu.adapterInfo,
        runtimeEvidence: res.webgpu.runtimeEvidence,
      },
    });
  } else {
    failed++;
    recordRunResult(webgpuManifest, {
      case: name,
      tier: 'webgpu',
      status: 'error',
      error: modelErrors.join(' | ') || 'WebGPU result was not produced',
    });
  }
}
finalizeRunManifest(webgpuManifest);
writeRunManifestSync(WEBGPU_MANIFEST, webgpuManifest);
console.log(`\n${failed === 0 ? 'ALL_PASS' : `FAIL (${failed})`}`);
Deno.exit(failed === 0 && webgpuManifest.outcome === 'success' ? 0 : 1);
