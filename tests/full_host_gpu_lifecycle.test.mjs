import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const releaseRoot = new URL(`../dist/${version}/`, import.meta.url);
const { VxInferenceServiceClient,
  VxPlatformServiceClient, VxProfilingServiceClient, pb } = await import(new URL('volvoxai.js', releaseRoot));
const graph = JSON.stringify({
  format: 'volvox-graph/v1', dimensions: {},
  inputs: { x: { shape: [1, 4], dtype: 'float32' } }, outputs: ['y'],
  nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'x' },
    outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 4] } }, params: {} }],
});
const weights = new Uint8Array(16);
new DataView(weights.buffer).setBigUint64(0, 8n, true);
weights.set(new TextEncoder().encode('{}      '), 8);

async function createHost(filename) {
  const api = await import(new URL(filename, releaseRoot));
  const full = !filename.includes('.lite');
  const Host = full ? api.FullEngineHost : api.EngineHost;
  const wasmUrl = new URL(full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm', releaseRoot);
  return new Host({ wasmUrl, fetch: async source => {
    if (String(source) === 'graph.json') return { ok: true, text: async () => graph };
    if (String(source) === 'weights.safetensors') {
      return { ok: true, arrayBuffer: async () => weights.buffer.slice(0) };
    }
    return { ok: false, status: 404 };
  } });
}

const required = (...backends) => new pb.BackendPolicy({
  mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends,
  operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID,
});
function accepted(response) {
  assert.equal(response.report?.status, pb.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
  return response;
}

for (const filename of ['volvoxai.lite.js', 'volvoxai.lite.min.js', 'volvoxai.js', 'volvoxai.min.js'])
test(`${filename}: only full prepares a GPU for a valid live model with a GPU policy`, async (t) => {
  const profile = filename.includes('.lite') ? 'inference' : 'full';
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let adapters = 0;
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: {
    requestAdapter: async () => { adapters++; return null; },
  } } });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, 'navigator', original);
    else delete globalThis.navigator;
  });

  const unopened = await createHost(filename);
  await unopened.close();
  assert.equal(adapters, 0);
  const host = await createHost(filename);
  const inference = new VxInferenceServiceClient(reportTransport(host));
  try {
    const info = await new VxPlatformServiceClient(reportTransport(host)).getPlatformInfo(new pb.Empty());
    assert.equal(info.profile, profile === 'full' ? pb.BuildProfile.BUILD_PROFILE_FULL : pb.BuildProfile.BUILD_PROFILE_INFERENCE);
    const runtime = accepted(await inference.createRuntime(new pb.CreateRuntimeRequest()));
    const profiling = new VxProfilingServiceClient(reportTransport(host));
    const trace = accepted(await profiling.startTrace(new pb.StartTraceRequest({
      runtimeId: runtime.runtimeId, deviceTiming: true,
    })));
    assert.equal(adapters, 0, 'device timing alone must not acquire a GPU');
    accepted(await profiling.stopTrace(new pb.TraceRef(trace)));
    await profiling.releaseTrace(new pb.TraceRef(trace));
    const model = accepted(await inference.loadModel(new pb.LoadModelRequest({
      runtimeId: runtime.runtimeId, graphPath: 'graph.json', weightPaths: ['weights.safetensors'],
    })));
    const defaults = accepted(await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId })));
    const cpu = accepted(await inference.compileModel(new pb.CompileModelRequest({
      modelId: model.modelId, policy: required('wasm'),
    })));
    assert.notEqual(cpu.compiledModelId, defaults.compiledModelId);
    const context = accepted(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: cpu.compiledModelId,
    })));

    const executeCpu = async () => {
      const result = accepted(await inference.execute(new pb.ExecuteRequest({
        contextId: context.contextId, inputs: [new pb.Tensor({ name: 'x',
          dtype: pb.DataType.DATA_TYPE_F32, shape: [1n, 4n],
          inline: new Uint8Array(Float32Array.of(-2, 0, 1, 3).buffer),
        })],
      })));
      assert.equal(result.report.route?.provider, 'wasm');
      const read = accepted(await inference.readOutput(new pb.ReadOutputRequest({
        resultId: result.resultId, name: 'y',
      })));
      const bytes = read.tensor.inline;
      assert.deepEqual([...new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))], [0, 0, 1, 3]);
      const released = await inference.releaseResult(new pb.ResultRef({ resultId: result.resultId }));
      assert.equal(released.status, pb.NativeStatus.NATIVE_STATUS_OK);
    };
    await executeCpu();
    assert.equal(adapters, 0, 'metadata, load, default compile, and required CPU execution need no GPU');

    const malformed = await host.openWithRequest({
      path: '/volvoxai.v1.VxInferenceService/CompileModel', requestStream: false, responseStream: false,
    }, () => Uint8Array.of(0x80));
    try { await assert.rejects(malformed.recv(), error => error.code === 3); }
    finally { await malformed.close(); }
    const missing = await inference.compileModel(new pb.CompileModelRequest({ modelId: 0n, policy: required('webgpu') }));
    assert.equal(missing.report?.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
    const invalid = await inference.compileModel(new pb.CompileModelRequest({
      modelId: model.modelId, policy: required('webgpu', 'wasm'),
    }));
    assert.equal(invalid.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(adapters, 0, 'C rejects invalid GPU requests before device acquisition');

    for (let attempt = 0; attempt < 2; attempt++) {
      const gpu = await inference.compileModel(new pb.CompileModelRequest({
        modelId: model.modelId, policy: required('webgpu'),
      }));
      assert.equal(gpu.compiledModelId, 0n);
      assert.notEqual(gpu.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
      assert.equal(gpu.report?.compilation?.candidates[0]?.backend, 'webgpu');
      assert.equal(gpu.report?.compilation?.candidates[0]?.outcome,
        pb.CandidateOutcome.CANDIDATE_OUTCOME_UNAVAILABLE);
    }
    assert.equal(adapters, profile === 'full' ? 1 : 0,
      'only full may acquire a device, once after C requests it');
    const preferred = accepted(await inference.compileModel(new pb.CompileModelRequest({
      modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_PREFER,
        backends: ['webgpu', 'wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }),
    })));
    assert.equal(preferred.report.route?.provider, 'wasm');
    await executeCpu();
  } finally {
    await host.close();
  }
  assert.equal(adapters, profile === 'full' ? 1 : 0, 'closing does not acquire or retry a device');
  await assert.rejects((inference.createRuntime(new pb.CreateRuntimeRequest())), /closed/);
});
