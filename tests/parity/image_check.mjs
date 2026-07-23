#!/usr/bin/env node
// Real-image EfficientDet parity: decode dog.jpg/cat.jpg once (via image_oracle.py),
// feed the IDENTICAL input to VolvoxAI cpu/wasm/native + ONNX Runtime. VolvoxAI
// tiers compare every output element. The fp32 ONNX oracle does too; the int8
// ONNX graph is a different true-quantized numeric domain, so it gates on the
// selected detections while reporting raw-tensor deltas as diagnostics.
//
//   node tests/parity/image_check.mjs examples/efficientdet_lite0/assets/dog.jpg int8
import fs from 'fs';
import path from 'path';
import { pathToFileURL, fileURLToPath } from 'url';
import { execFileSync } from 'child_process';
import { readTensor } from './lib/tensorio.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const [imgRel, model] = process.argv.slice(2);
if (!imgRel || !['int8', 'fp32'].includes(model)) {
  console.error('usage: image_check.mjs <image.jpg> <int8|fp32>'); process.exit(2);
}
const labels = fs.readFileSync(path.join(ROOT, 'models/efficientdet_lite0_int8/labels.txt'), 'utf8').split(/\r?\n/);
const policy = JSON.parse(fs.readFileSync(path.join(ROOT, 'tests/parity/policy.json'), 'utf8'));
const modelPolicy = policy.models[`efficientdet_lite0_${model}`];
fs.mkdirSync(path.join(ROOT, 'tests/parity/out'), { recursive: true });
const runDir = fs.mkdtempSync(path.join(ROOT, 'tests/parity/out/', 'image-check-'));
process.on('exit', () => fs.rmSync(runDir, { recursive: true, force: true }));

const nf = globalThis.fetch;
globalThis.fetch = async (u, i) => {
  const h = typeof u === 'string' ? u : u?.url;
  if (h && h.startsWith('file://')) return new Response(await fs.promises.readFile(fileURLToPath(h)), { status: 200 });
  return nf(u, i);
};

// 1) decode image + run ONNX (writes fixture + onnx.{scores,boxes}.f32 + prints ONNX detections)
execFileSync('python3', ['tests/parity/external/image_oracle.py', imgRel, model, runDir], { cwd: ROOT, stdio: 'inherit' });
const dtype = model === 'int8' ? 'u8' : 'f32';
const fix = path.join(runDir, `input0.${dtype}`);
const input = readTensor(fix, dtype);
const modelDir = path.join(ROOT, `models/efficientdet_lite0_${model}`);

const volvox = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
async function runVolvox(backend) {
  const runtime = await volvox.VolvoxAI.createRuntime({
    backends: [backend],
    wasmUrl: path.join(ROOT, 'dist', ver, 'volvoxai.wasm'),
  });
  const url = pathToFileURL(path.join(modelDir, 'model.safetensors')).href;
  const graph = new volvox.Graph();
  await volvox.GraphLoader.load(graph, url);
  const model = runtime.createModel(graph);
  let compiled;
  let context;
  try {
    compiled = await model.compile({
      backend: { mode: 'require', backend, operatorFallback: 'forbid' },
    });
    context = await compiled.createContext();
    const result = await context.execute({ input0: input });
    try {
      const checked = async (name) => {
        const tensor = graph.getTensor(name);
        const values = await result.output(name).read();
        if (!tensor || !ArrayBuffer.isView(values) || values.byteLength !== tensor.sizeBytes) {
          throw new Error(`${backend}: invalid output '${name}'`);
        }
        return values instanceof Float32Array ? values : Float32Array.from(values);
      };
      return { scores: await checked('scores'), boxes: await checked('boxes') };
    } finally {
      await result.close();
    }
  } finally {
    await context?.close();
    await compiled?.close();
    await model.close();
    await runtime.close();
  }
}

function runNative() {
  const bin = path.join(ROOT, 'native', 'volvoxai');
  if (!fs.existsSync(bin)) throw new Error(`required native binary is missing: ${bin}`);
  const so = path.join(runDir, 'native.scores.f32');
  const bo = path.join(runDir, 'native.boxes.f32');
  const route = execFileSync(
    bin,
    ['run', modelDir, '--input', `input0=${fix}`, '--output', `scores=${so}`, '--output', `boxes=${bo}`],
    { encoding: 'utf8' },
  );
  if (!/^Backend: cpu\r?$/m.test(route)) {
    throw new Error(`native CPU run did not report the requested route: ${route.slice(0, 200)}`);
  }
  return { scores: readTensor(so, 'f32'), boxes: readTensor(bo, 'f32') };
}

const EXPECTED = { scores: 1 * 19206 * 90, boxes: 1 * 19206 * 4 };
function validateOutput(tier, output) {
  for (const name of ['scores', 'boxes']) {
    const values = output?.[name];
    if (!ArrayBuffer.isView(values) || values.length !== EXPECTED[name]) {
      throw new Error(`${tier}/${name}: got ${values?.length ?? 'invalid'} elements, expected ${EXPECTED[name]}`);
    }
    for (let i = 0; i < values.length; i++) {
      if (!Number.isFinite(values[i])) throw new Error(`${tier}/${name}: non-finite value at ${i}`);
    }
  }
  return output;
}

function compareFull(candidate, reference, tolerance) {
  let maxAbs = 0;
  let exceed = 0;
  for (let i = 0; i < candidate.length; i++) {
    const delta = Math.abs(candidate[i] - reference[i]);
    maxAbs = Math.max(maxAbs, delta);
    if (delta > tolerance.atol + tolerance.rtol * Math.abs(reference[i])) exceed++;
  }
  return { pass: exceed === 0, maxAbs, exceed };
}

// top detections: per anchor best class, then top-k anchors by score
function topDetections(scores, boxes, k = 5) {
  const N = boxes.length / 4, C = scores.length / N;
  const best = new Array(N);
  for (let a = 0; a < N; a++) {
    let bc = 0, bs = -1e30;
    for (let c = 0; c < C; c++) {
      const v = scores[a * C + c];
      if (v > bs) { bs = v; bc = c; }
    }
    best[a] = { a, c: bc, s: bs, box: [...boxes.subarray(a * 4, a * 4 + 4)] };
  }
  return best.sort((x, y) => y.s - x.s).slice(0, k)
}

function detectionTask(candidate, reference, tolerance, k = 5) {
  const cand = topDetections(candidate.scores, candidate.boxes, k);
  const refTop = topDetections(reference.scores, reference.boxes, k);
  const byIdentity = new Map(cand.map((item) => [`${item.a}:${item.c}`, item]));
  const top1Match = cand[0]?.a === refTop[0]?.a && cand[0]?.c === refTop[0]?.c;
  let missing = 0, scoreExceed = 0, boxExceed = 0, maxScoreAbs = 0, maxBoxAbs = 0;
  for (const expected of refTop) {
    const actual = byIdentity.get(`${expected.a}:${expected.c}`);
    if (!actual) { missing++; continue; }
    const scoreDelta = Math.abs(actual.s - expected.s);
    maxScoreAbs = Math.max(maxScoreAbs, scoreDelta);
    if (scoreDelta > tolerance.atol + tolerance.rtol * Math.abs(expected.s)) scoreExceed++;
    for (let i = 0; i < 4; i++) {
      const boxDelta = Math.abs(actual.box[i] - expected.box[i]);
      maxBoxAbs = Math.max(maxBoxAbs, boxDelta);
      if (boxDelta > tolerance.atol + tolerance.rtol * Math.abs(expected.box[i])) boxExceed++;
    }
  }
  const top1ScoreDelta = top1Match ? Math.abs(cand[0].s - refTop[0].s) : Infinity;
  const top1ScorePass = top1Match &&
    top1ScoreDelta <= tolerance.atol + tolerance.rtol * Math.abs(refTop[0].s);
  const topKOverlap = (k - missing) / k;
  return {
    // The true-int8 ONNX graph and VolvoxAI's folded-fp32 reference do not
    // promise raw regression-value parity. Gate the externally meaningful
    // decision: exact top-1 identity/confidence and at least 4/5 matching
    // anchor/class detections. Near-tied quantized candidates may swap at the
    // tail, so require bounded overlap rather than an arbitrary total order.
    // Keep all selected score/box deltas visible.
    pass: top1ScorePass && topKOverlap >= 0.8,
    top1Match,
    top1ScorePass,
    top1ScoreDelta,
    topKOverlap,
    missing,
    scoreExceed,
    boxExceed,
    maxScoreAbs,
    maxBoxAbs,
  };
}

function formatDetections(output) {
  return topDetections(output.scores, output.boxes)
    .map((d) => `${(labels[d.c] || d.c)}@${d.a}:${d.s.toFixed(3)}`);
}

const onnx = validateOutput('onnx', {
  scores: readTensor(path.join(runDir, 'onnx.scores.f32'), 'f32'),
  boxes: readTensor(path.join(runDir, 'onnx.boxes.f32'), 'f32'),
});
const runs = {
  cpu: validateOutput('cpu', await runVolvox('cpu')),
  wasm: validateOutput('wasm', await runVolvox('wasm')),
  'native-cpu': validateOutput('native-cpu', runNative()),
  onnx,
};

const ref = runs.cpu;
console.log(`\n=== ${path.basename(imgRel)} · efficientdet ${model} · all ${ref.boxes.length / 4} anchors ===`);
console.log('| tier | max|Δscores| vs cpu | max|Δboxes| vs cpu | top detections |');
console.log('|---|---|---|---|');
let failed = 0;
for (const [tier, r] of Object.entries(runs)) {
  const baseTolerance = tier === 'onnx' ? policy.tolerances.external
    : policy.tolerances[model === 'int8' ? 'int8' : 'fp32'];
  const scoresTolerance = {
    ...baseTolerance,
    ...(modelPolicy.toleranceOverrides?.[tier]?.scores ?? {}),
  };
  const boxesTolerance = {
    ...baseTolerance,
    ...(modelPolicy.toleranceOverrides?.[tier]?.boxes ?? {}),
  };
  const scores = tier === 'cpu' ? { pass: true, maxAbs: 0, exceed: 0 }
    : compareFull(r.scores, ref.scores, scoresTolerance);
  const boxes = tier === 'cpu' ? { pass: true, maxAbs: 0, exceed: 0 }
    : compareFull(r.boxes, ref.boxes, boxesTolerance);
  const task = tier === 'onnx' && model === 'int8'
    ? detectionTask(r, ref, baseTolerance)
    : null;
  const pass = task ? task.pass : scores.pass && boxes.pass;
  if (!pass) failed++;
  const overrideReason = modelPolicy.toleranceOverrides?.[tier]?.scores?.reason;
  const mode = task
    ? `task ${task.top1ScorePass ? 'top1' : 'top1-mismatch'}, top5 overlap=${task.topKOverlap.toFixed(2)}, selected max score/box Δ=${task.maxScoreAbs.toExponential(2)}/${task.maxBoxAbs.toExponential(2)} (diagnostic)`
    : `full tensor${overrideReason ? ' (declared score approximation)' : ''}`;
  console.log(`| ${tier} | ${scores.maxAbs.toExponential(2)} (${scores.exceed} exceed) | ${boxes.maxAbs.toExponential(2)} (${boxes.exceed} exceed) | ${pass ? 'PASS' : 'FAIL'} (${mode}) · ${formatDetections(r).join(', ')} |`);
}
if (failed) {
  console.error(`${failed} image parity tier(s) failed their declared full-tensor/task contract`);
  process.exitCode = 1;
}
