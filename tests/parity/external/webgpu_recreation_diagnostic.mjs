/** Diagnose native reclamation separately from logical GPUDevice destruction.
 * GC is an explicit diagnostic mode, never a product workaround or a passing
 * release qualification. Run each mode in a fresh Deno process.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const args = new Map();
for (let index = 0; index < Deno.args.length; index += 2) {
  args.set(Deno.args[index], Deno.args[index + 1]);
}
const mode = args.get('--mode') ?? 'raw';
const cleanup = args.get('--cleanup') ?? 'none';
const retention = args.get('--retain') ?? 'none';
assert.ok(['raw', 'proto'].includes(mode));
assert.ok(['none', 'tick', 'delay', 'gc'].includes(cleanup));
assert.ok(['none', mode === 'raw' ? 'device' : 'host'].includes(retention));
if (cleanup === 'gc') assert.equal(typeof globalThis.gc, 'function', 'use --v8-flags=--expose-gc for this diagnostic only');

const retained = [];
const tracked = [];
let iteration = 0, phase = 'start', failurePhase;
const event = (name, fields = {}) => console.log(JSON.stringify({ mode, cleanup, retention, iteration, event: name, ...fields }));
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const watchdog = setTimeout(() => { event('timeout', { phase }); Deno.exit(124); }, 30_000);
const gpu = navigator.gpu;
assert.ok(gpu, 'WebGPU is required');
const originalRequestAdapter = gpu.requestAdapter.bind(gpu);
gpu.requestAdapter = async (...values) => {
  const adapter = await originalRequestAdapter(...values);
  assert.ok(adapter);
  const requestDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = async (...values) => {
    const device = await requestDevice(...values);
    tracked.push({ kind: 'device', iteration, weak: new WeakRef(device) });
    const createBuffer = device.createBuffer.bind(device);
    device.createBuffer = (...values) => {
      const buffer = createBuffer(...values);
      tracked.push({ kind: 'buffer', iteration, weak: new WeakRef(buffer) });
      return buffer;
    };
    event('device-created', { adapter: adapter.info.description });
    return device;
  };
  return adapter;
};

async function rawCycle() {
  phase = 'request-device';
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  const device = await adapter.requestDevice();
  let source, destination;
  try {
    phase = 'copy';
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    source = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    destination = device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const input = Float32Array.from({ length: 12 }, (_, index) => index);
    device.queue.writeBuffer(source, 0, input);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, destination, 0, 48);
    (await device.queue.submit([encoder.finish()]));
    const errors = (await Promise.all([
      device.popErrorScope(), device.popErrorScope(),
      destination.mapAsync(GPUMapMode.READ).then(() => null, error => error),
    ])).filter(Boolean).map(error => error.message);
    event('copy-completed', { errors });
    assert.deepEqual(errors, []);
    assert.deepEqual(new Float32Array(destination.getMappedRange()), input);
    destination.unmap();
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    failurePhase = phase;
    throw error;
  } finally {
    source?.destroy(); destination?.destroy();
    phase = 'device-destroy';
    device.destroy();
    const lost = await device.lost;
    event('device-destroyed', { reason: lost.reason });
    if (retention === 'device') retained.push(device);
  }
}

let api, wasmUrl;
if (mode === 'proto') {
  api = await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
  wasmUrl = pathToFileURL(path.resolve(args.get('--wasm')));
  const fetch = globalThis.fetch;
  globalThis.fetch = async (source, options) => String(source).startsWith('file:')
    ? new Response(await readFile(fileURLToPath(source))) : fetch(source, options);
}
async function protoCycle() {
  const p = api.pb;
  const ok = value => {
    const report = value.report ?? value;
    assert.equal(report.status, p.NativeStatus.NATIVE_STATUS_OK, `${report.code}: ${report.message}`);
    return value;
  };
  const graph = JSON.stringify({ format: 'volvox-graph/v1', dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1, 4] } }, outputs: ['y'],
    nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'x' }, params: {},
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 4] } } }],
  });
  const host = new api.FullEngineHost({ wasmUrl,
    onDiagnostic: message => event('bridge-diagnostic', { message }),
    fetch: async () => ({ ok: true, text: async () => graph }),
  });
  tracked.push({ kind: 'host', iteration, weak: new WeakRef(host) });
  try {
    const inference = new api.VxInferenceServiceClient(host);
    phase = 'load';
    const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
    const model = ok(await inference.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'graph.json' })));
    phase = 'compile';
    const compiled = ok(await inference.compileModel(new p.CompileModelRequest({ modelId: model.modelId,
      policy: new p.BackendPolicy({ mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['webgpu'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID }),
    })));
    const context = ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId })));
    phase = 'execute';
    const result = ok(await inference.execute(new p.ExecuteRequest({ contextId: context.contextId,
      inputs: [new p.Tensor({ name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [1n, 4n],
        inline: new Uint8Array(Float32Array.of(-2, 0, 1, 3).buffer) })],
    })));
    phase = 'read';
    const output = ok(await inference.readOutput(new p.ReadOutputRequest({ resultId: result.resultId, name: 'y' })));
    assert.equal(result.report.route.provider, 'webgpu');
    const bytes = output.tensor.inline;
    assert.deepEqual([...new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))], [0, 0, 1, 3]);
    event('inference-completed', { route: result.report.route.provider });
  } catch (error) {
    failurePhase = phase;
    throw error;
  } finally {
    phase = 'host-close';
    await host.close();
    event('host-closed');
    if (retention === 'host') retained.push(host);
  }
}

try {
  for (iteration = 0; iteration < 3; iteration++) {
    await (mode === 'raw' ? rawCycle() : protoCycle());
    phase = 'after-close';
    // GC and WeakRef dereferencing must happen after the previous JS job.
    // Dereferencing before GC would keep its target alive until that job ends.
    if (cleanup === 'tick' || cleanup === 'gc') await tick();
    if (cleanup === 'delay') await new Promise(resolve => setTimeout(resolve, 2_000));
    if (cleanup === 'gc') {
      globalThis.gc(); await tick(); globalThis.gc(); await tick(); globalThis.gc();
    }
    const alive = {};
    for (const value of tracked) if (value.weak.deref()) alive[value.kind] = (alive[value.kind] ?? 0) + 1;
    event('after-cleanup', { alive, retained: retained.length });
  }
  event('passed');
} catch (error) {
  event('failed', { phase: failurePhase ?? phase, message: error.message });
  Deno.exitCode = 1;
} finally {
  clearTimeout(watchdog);
}
