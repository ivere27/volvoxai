// WebGPU producer for the L1 op / L2 graph matrices, run under Deno on a real GPU.
// Reads the packages already authored by `run.mjs opmatrix|graphmatrix` (so inputs
// and configs are byte-identical to the other tiers) and emits a webgpu signature
// per case in the same format the matrices consume:  <sigDir>/<id>.webgpu.json.
//
//   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
//     tests/parity/webgpu_matrix_deno.js <packagesDir> <sigDir>
//   e.g.  ... out/opcases   out/ops
//         ... out/graphcases out/graphsigs
import { signature } from './lib/extract.mjs';
import { requirePhysicalWebGPU } from './lib/backend.mjs';
import { createRuntimeEvidence } from './lib/artifact.mjs';
import { captureStableResult } from './lib/runmodel.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
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

const [pkgDirArg, sigDirArg] = Deno.args;
if (!pkgDirArg || !sigDirArg) { console.error('usage: webgpu_matrix_deno.js <packagesDir> <sigDir>'); Deno.exit(2); }
const pkgDir = path.isAbsolute(pkgDirArg) ? pkgDirArg : path.join(ROOT, pkgDirArg);
const sigDir = path.isAbsolute(sigDirArg) ? sigDirArg : path.join(ROOT, sigDirArg);
fs.mkdirSync(sigDir, { recursive: true });

function atomicWriteJson(file, value) {
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${globalThis.process?.pid ?? 'deno'}.${randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value)}\n`, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

const TA = { f32: Float32Array, i32: Int32Array, u8: Uint8Array, i8: Int8Array };
// Inputs were materialized as inputs/<name>.<ext>; pick whichever dtype exists.
function loadInput(pkg, name) {
  for (const ext of ['f32', 'i32', 'i8', 'u8']) {
    const f = path.join(pkg, 'inputs', `${name}.${ext}`);
    if (fs.existsSync(f)) { const b = fs.readFileSync(f); const C = TA[ext]; return new C(b.buffer, b.byteOffset, b.byteLength / C.BYTES_PER_ELEMENT); }
  }
  throw new Error(`no input file for "${name}" in ${pkg}/inputs`);
}

const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');

const ids = fs.readdirSync(pkgDir).filter((d) => fs.existsSync(path.join(pkgDir, d, 'meta.json')));
// Reuse one Runtime/provider for every case so all compiled models share one
// explicitly owned WebGPU device.
const runtime = await module.VolvoxAI.createRuntime({ backends: ['webgpu'], wasmUrl: wasmPath });
let ok = 0, fail = 0, skipped = 0;
for (const id of ids) {
  const pkg = path.join(pkgDir, id);
  try {
    const meta = JSON.parse(fs.readFileSync(path.join(pkg, 'meta.json'), 'utf8'));
    if (meta.skip?.includes('webgpu')) {
      console.log(`  webgpu ${id}: declared skip`);
      skipped++;
      continue;
    }
    const inputs = {};
    for (const name of meta.inputs) inputs[name] = loadInput(pkg, name);
    const modelUrl = pathToFileURL(path.join(pkg, 'model.safetensors')).href;
    const graph = new module.Graph();
    await module.GraphLoader.load(graph, modelUrl);
    const model = runtime.createModel(graph);
    let compiled;
    let context;
    let result;
    try {
      compiled = await model.compile({
        backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
      });
      const adapterInfo = requirePhysicalWebGPU(compiled, 'WebGPU matrix');
      context = await compiled.createContext();
      result = await context.execute(inputs);
      const name = (graph.outputNames && graph.outputNames[0]) || 'y';
      const tensor = graph.getTensor(name);
      if (!tensor) throw new Error(`graph has no output tensor '${name}'`);
      const execution = result.report;
      const captured = await captureStableResult(
        graph,
        result,
        context,
        'webgpu',
        graph.outputNames,
      );
      result = null;
      const values = captured.outputs[name];
      const flat = values instanceof Float32Array ? values : Float32Array.from(values);
      const outputShape = meta.outputShape ?? tensor.shape;
      atomicWriteJson(path.join(sigDir, `${id}.webgpu.json`), {
        id,
        tier: 'webgpu',
        backend: 'webgpu',
        adapterInfo,
        sig: signature(flat, { topk: 0, shape: outputShape }),
        runtimeEvidence: createRuntimeEvidence({
          compilation: compiled.report,
          execution,
          stableResult: captured.stableResult,
        }),
      });
      ok++;
    } finally {
      await result?.close();
      await context?.close();
      await compiled?.close();
      await model.close();
    }
  } catch (e) { console.error(`  webgpu ${id}: ${String(e.message || e).split('\n')[0].slice(0, 160)}`); fail++; }
}
await runtime.close();
console.log(`webgpu matrix: ${ok} ok, ${skipped} declared skip, ${fail} failed -> ${path.relative(ROOT, sigDir)}`);
Deno.exit(fail === 0 ? 0 : 1);
