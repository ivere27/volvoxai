// WebGPU EfficientDet parity on a real GPU, under Deno. EfficientDet has TWO
// graph outputs (scores, boxes); both are read through the stable result owner.
// Compares against a native reference the
// caller passes (OpenGL, which matches ONNX to ~6e-7 on fp32).
//
//   deno run --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
//     tests/parity/webgpu_efficientdet_deno.js <model> <inputFile> <u8|f32> <outDir>
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { formatAdapterIdentity, requirePhysicalWebGPU } from './lib/backend.mjs';
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

function atomicWriteJson(file, value) {
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${globalThis.process?.pid ?? 'deno'}.${randomUUID()}.tmp`,
  );
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value)}\n`, { flag: 'wx' });
    fs.renameSync(temporary, file);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

const [modelName = 'efficientdet_lite0_fp32', inputFile = 'dog_fp32.f32', dtype = 'f32', outDir = '/tmp'] = Deno.args;
const TA = { f32: Float32Array, u8: Uint8Array, i8: Int8Array, i32: Int32Array };
const InputArray = TA[dtype];
if (!InputArray) throw new Error(`unsupported input dtype '${dtype}'`);
const inBuf = fs.readFileSync(path.join(ROOT, inputFile));
if (inBuf.byteLength % InputArray.BYTES_PER_ELEMENT !== 0) {
  throw new Error(`input byte length ${inBuf.byteLength} is not aligned for ${dtype}`);
}
const input0 = new InputArray(inBuf.buffer, inBuf.byteOffset, inBuf.byteLength / InputArray.BYTES_PER_ELEMENT);

const ver = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const module = await import(pathToFileURL(path.join(ROOT, 'dist', ver, 'volvoxai.js')).href);
const wasmPath = path.join(ROOT, 'dist', ver, 'volvoxai.wasm');
const modelUrl = pathToFileURL(path.join(ROOT, 'models', modelName, 'model.safetensors')).href;

const graph = new module.Graph();
await module.GraphLoader.load(graph, modelUrl);
const runtime = await module.VolvoxAI.createRuntime({ backends: ['webgpu'], wasmUrl: wasmPath });
const model = runtime.createModel(graph);
let compiled;
let context;
let execution;
try {
  compiled = await model.compile({
    backend: { mode: 'require', backend: 'webgpu', operatorFallback: 'forbid' },
  });
  const adapterInfo = requirePhysicalWebGPU(compiled, 'EfficientDet GPU consensus');
  console.log(`Adapter: ${formatAdapterIdentity(adapterInfo)}`);
  atomicWriteJson(path.join(outDir, 'webgpu_adapter.json'), adapterInfo);
  context = await compiled.createContext();
  const warmup = await context.execute({ input0 });
  await warmup.close();
  execution = await context.execute({ input0 });

  const outNames = graph.outputNames || [];
  if (outNames.length === 0) throw new Error('WebGPU graph has no declared outputs');
  console.log(`outputNames=${JSON.stringify(outNames)}`);

  const outputs = {};
  for (const name of outNames) {
    const tensor = graph.getTensor(name);
    if (!tensor) throw new Error(`graph has no output tensor '${name}'`);
    const arr = await execution.output(name).read();
    if (!ArrayBuffer.isView(arr) || arr.byteLength !== tensor.sizeBytes) {
      throw new Error(`WebGPU output '${name}' readback has the wrong logical size`);
    }
    const flat = arr instanceof Float32Array ? arr : Float32Array.from(arr);
    outputs[name] = flat;
    const dst = path.join(outDir, `wg_${name}.f32`);
    const temporary = path.join(path.dirname(dst), `.${path.basename(dst)}.${globalThis.process?.pid ?? 'deno'}.${randomUUID()}.tmp`);
    try {
      fs.writeFileSync(temporary, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength), { flag: 'wx' });
      fs.renameSync(temporary, dst);
    } catch (error) {
      fs.rmSync(temporary, { force: true });
      throw error;
    }
    console.log(`  [${name}] len=${flat.length} -> ${dst}`);
  }
  console.log(`WROTE ${Object.keys(outputs).join(',')}`);
} finally {
  await execution?.close();
  await context?.close();
  await compiled?.close();
  await model.close();
  await runtime.close();
}
