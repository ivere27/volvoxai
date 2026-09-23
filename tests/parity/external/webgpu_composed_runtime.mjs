/**
 * Exercise the minified full profile through generated API calls on one C owner.
 * CPU and required-WebGPU results must agree, with explicit GPU route evidence.
 * --recreate-hosts also closes the CPU host before acquiring the GPU host.
 * --recreate-gpu-hosts additionally closes an actual GPU host and repeats GPU execution.
 *
 * Run on a physical adapter:
 *   deno run --unstable-webgpu --allow-read --allow-env --allow-ffi \
 *     tests/parity/external/webgpu_composed_runtime.mjs \
 *     --bundle dist/0.6.0/volvoxai.min.js \
 *     --wasm dist/0.6.0/volvoxai.wasm
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';

function fail(message) {
  console.error(`webgpu-composed-runtime FAIL ${message}`);
  Deno.exit(1);
}

function options(argv) {
  const parsed = { bundle: null, wasm: null, recreateHosts: false, recreateGpuHosts: false };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === '--recreate-hosts') { parsed.recreateHosts = true; continue; }
    if (flag === '--recreate-gpu-hosts') { parsed.recreateGpuHosts = true; continue; }
    if (flag === '--bundle') { parsed.bundle = path.resolve(argv[++index]); continue; }
    if (flag === '--wasm') { parsed.wasm = path.resolve(argv[++index]); continue; }
    fail(`unknown argument '${flag}'`);
  }
  if (!parsed.bundle || !parsed.wasm) fail('--bundle and --wasm are required');
  return parsed;
}

/* One graph, small and entirely ordinary: the question here is which planner
 * ran it, not which operator. */
const SHAPE = [1, 2, 3, 2];
const INPUT = Float32Array.from(
  { length: 12 }, (_, index) => (index % 7) - 3 + index * 0.25);

const GRAPH = {
  format: 'volvox-graph/v1',
  dimensions: {},
  inputs: { input0: { shape: SHAPE, dtype: 'float32' } },
  outputs: ['out0'],
  nodes: [
    { id: 'a', opType: 'ReLU', inputs: { input: 'input0' },
      outputs: { out: { tensor: 'mid', dtype: 'float32', shape: SHAPE } },
      params: {} },
    { id: 'b', opType: 'Sigmoid', inputs: { input: 'mid' },
      outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
      params: {} },
  ],
};

function emptySafetensors() {
  const text = new TextEncoder().encode('{}');
  const padding = (8 - (text.byteLength % 8)) % 8;
  const bytes = new Uint8Array(8 + text.byteLength + padding);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(text.byteLength + padding), true);
  bytes.set(text, 8);
  bytes.fill(0x20, 8 + text.byteLength);
  return bytes;
}

function createHost(api, opts, wasmBytes) {
  const graphText = JSON.stringify(GRAPH);
  const weights = emptySafetensors();
  return new api.FullEngineHost({
    onDiagnostic: message => console.error(message),
    wasmUrl: pathToFileURL(opts.wasm),
    fetch: async (source) => {
      const name = String(source);
      if (name === 'graph.json') {
        return { ok: true, text: async () => graphText,
          json: async () => JSON.parse(graphText) };
      }
      if (name === 'model.safetensors') {
        return { ok: true, arrayBuffer: async () => weights.buffer.slice(
          weights.byteOffset, weights.byteOffset + weights.byteLength) };
      }
      if (name === opts.wasm || name.endsWith('.wasm')) {
        return { ok: true, arrayBuffer: async () => wasmBytes.buffer.slice(
          wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) };
      }
      return { ok: false, status: 404, statusText: name };
    },
    resolveModelSource: (graphPath, weightPaths) => ({
      graphUrl: graphPath, weightSources: weightPaths,
    }),
  });
}

/** Compile one graph under a policy that requires exactly one backend. */
async function runGraph(api, host, backend) {
  const inference = new api.VxInferenceServiceClient(host);
  const ok = api.pb.NativeStatus.NATIVE_STATUS_OK;
  const say = (stage, report) => {
    /* A compile report carries one candidate per backend the policy named;
     * the top-level message only says the policy was not satisfied. */
    const candidates = (report?.compilation?.candidates ?? []).map((candidate) =>
      `${candidate.backend}:${candidate.outcome}:${candidate.reason}`).join(' | ');
    return `${backend}/${stage} [${report?.code}] ${report?.message}` +
      (candidates ? ` || ${candidates}` : '');
  };

  const runtime = await inference.createRuntime(new api.pb.CreateRuntimeRequest());
  if (runtime.report?.status !== ok) fail(say('createRuntime', runtime.report));
  const model = await inference.loadModel(new api.pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: 'graph.json',
    weightPaths: ['model.safetensors'],
  }));
  if (model.report?.status !== ok) fail(say('loadModel', model.report));
  const compiled = await inference.compileModel(new api.pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new api.pb.BackendPolicy({
      mode: api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: [backend],
      operatorFallback: api.pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
    }),
  }));
  if (compiled.report?.status !== ok) fail(say('compileModel', compiled.report));
  const context = await inference.createExecutionContext(
    new api.pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId,
    }));
  if (context.report?.status !== ok) fail(say('createContext', context.report));

  const result = await inference.execute(new api.pb.ExecuteRequest({
    contextId: context.contextId,
    inputs: [new api.pb.Tensor({
      name: 'input0',
      dtype: api.pb.DataType.DATA_TYPE_F32,
      shape: SHAPE.map(BigInt),
      inline: new Uint8Array(INPUT.buffer.slice(0)),
    })],
  }));
  if (result.report?.status !== ok) fail(say('execute', result.report));

  const deadline = performance.now() + 30000;
  for (;;) {
    const completed = await inference.getResult(new api.pb.ResultRef(result));
    if (completed.report?.status !== ok) fail(say('getResult', completed.report));
    if (completed.state !== api.pb.ResultState.RESULT_STATE_PENDING) {
      if (completed.state !== api.pb.ResultState.RESULT_STATE_READY ||
          completed.executionId !== result.executionId) fail(`${backend}: invalid completion`);
      break;
    }
    if (performance.now() >= deadline) fail(`${backend}: completion timed out`);
    await new Promise(resolve => setTimeout(resolve, 1));
  }

  const read = await inference.readOutput(new api.pb.ReadOutputRequest({
    resultId: result.resultId, name: 'out0',
  }));
  if (read.report?.status !== ok) fail(say('readOutput', read.report));
  const payload = read.tensor?.inline;
  if (!payload?.byteLength) fail(`${backend}: output had no bytes`);
  return {
    values: new Float32Array(payload.buffer.slice(
      payload.byteOffset, payload.byteOffset + payload.byteLength)),
    route: result.report?.route?.provider ?? null,
  };
}

async function main() {
  const opts = options(Deno.args);
  const gpu = navigator.gpu;
  if (!gpu) fail('no WebGPU implementation');
  const originalRequestAdapter = gpu.requestAdapter;
  let devicesRequested = 0;
  gpu.requestAdapter = async (...args) => {
    const adapter = await originalRequestAdapter.apply(gpu, args);
    if (adapter) {
      const requestDevice = adapter.requestDevice.bind(adapter);
      adapter.requestDevice = async (...args) => {
        const device = await requestDevice(...args);
        console.log(`webgpu-composed-runtime requestDevice: ${++devicesRequested}`);
        return device;
      };
    }
    return adapter;
  };
  const adapter = await navigator.gpu?.requestAdapter?.({
    powerPreference: 'high-performance',
  });
  if (!adapter) fail('no WebGPU adapter; this proof needs a physical device');
  const info = adapter.info ?? {};
  console.log(`webgpu-composed-runtime adapter: ${info.vendor ?? '?'} ` +
    `${info.device ?? '?'} ${info.description ?? ''}`.trim());

  const api = await import(`file://${opts.bundle}`);
  const wasmBytes = await readFile(opts.wasm);
  const fetchOriginal = globalThis.fetch;
  globalThis.fetch = async (url, init) => String(url).startsWith('file:')
    ? new Response(await readFile(fileURLToPath(url))) : fetchOriginal(url, init);

  let engine = createHost(api, opts, wasmBytes);
  const closedHosts = [];
  let host, device;
  try {
    await new api.VxPlatformServiceClient(engine).getPlatformInfo(new api.pb.Empty());
    if (devicesRequested !== 0) fail('platform information acquired a GPU device');
    host = await runGraph(api, engine, 'wasm');
    if (devicesRequested !== 0) fail('required CPU execution acquired a GPU device');
    console.log('webgpu-composed-runtime metadata and CPU device count: 0');
    if (opts.recreateHosts) {
      await engine.close();
      closedHosts.push(engine);
      engine = createHost(api, opts, wasmBytes);
    }
    device = await runGraph(api, engine, 'webgpu');
    if (devicesRequested !== 1) fail(`first GPU execution acquired ${devicesRequested} devices`);
    if (opts.recreateGpuHosts) {
      await engine.close();
      closedHosts.push(engine);
      engine = createHost(api, opts, wasmBytes);
      device = await runGraph(api, engine, 'webgpu');
      if (devicesRequested !== 2) fail(`second GPU execution acquired ${devicesRequested} devices`);
    }
  } finally {
    await engine.close();
    globalThis.fetch = fetchOriginal;
    gpu.requestAdapter = originalRequestAdapter;
  }

  if (device.values.length !== host.values.length) {
    fail(`length ${device.values.length} vs ${host.values.length}`);
  }
  let worst = 0;
  for (let index = 0; index < host.values.length; index++) {
    const expected = host.values[index];
    const actual = device.values[index];
    const scale = Math.max(1e-6, Math.abs(expected));
    worst = Math.max(worst, Math.abs(actual - expected) / scale);
  }
  if (!(worst <= 1e-4)) fail(`values differ, max_rel=${worst.toExponential(3)}`);

  /* The route is the evidence. Matching values would also come back from a
   * provider that never touched the device, which is precisely what this
   * proof exists to rule out. */
  if (device.route !== 'webgpu') {
    fail(`compiled under 'webgpu' but ran on route '${device.route}'`);
  }
  console.log(`webgpu-composed-runtime PASS (route=${device.route}, ` +
    `max_rel=${worst.toExponential(3)}, recreated=${opts.recreateHosts}, ` +
    `gpu_recreated=${opts.recreateGpuHosts}, devices=${devicesRequested}, ` +
    `retained_closed_hosts=${closedHosts.length}) -- the selected release profile reaches the C planner`);
}

await main();
