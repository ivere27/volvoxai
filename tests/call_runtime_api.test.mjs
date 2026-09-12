import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { assertOperationResponse } from '../ts/host/OperationReports.js';
import { VolvoxAIError } from '../ts/core/RuntimeErrors.js';
import * as inferencePb from '../runtime/generated/typescript/inference/volvoxai_lite.js';
import * as fullPb from '../runtime/generated/typescript/volvoxai_lite.js';
import { PROTO_METHOD_RESPONSES as inferenceMethods } from '../ts/generated/protoMethods.js';
import { PROTO_METHOD_RESPONSES as fullMethods } from '../ts/generated/protoMethodsFull.js';

const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
const releaseRoot = new URL(`../dist/${version}/`, import.meta.url);

for (const [profile, p, methods] of [['inference', inferencePb, inferenceMethods], ['full', fullPb, fullMethods]]) {
  test(`${profile}: schema-selected reports preserve every typed domain error`, () => {
    for (const [path, typeName] of Object.entries(methods)) {
      const Response = p[typeName];
      const direct = Response === p.OperationReport;
      const hasReport = Response.fields.some(field => field.messageType === p.OperationReport.typeName);
      const ok = new p.OperationReport();
      assert.doesNotThrow(() => assertOperationResponse(p, methods, path,
        (direct ? ok : new Response(hasReport ? { report: ok } : {})).toBinary()));
      if (!direct && !hasReport) continue;
      const report = new p.OperationReport({
        status: p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
        code: p.OperationCode.OPERATION_CODE_INVALID_ARGUMENT,
        stage: p.OperationStage.OPERATION_STAGE_EXECUTE,
        message: 'invalid input',
        inputIssue: new p.InputValidationIssue({ inputName: 'pixels' }),
      });
      assert.throws(() => assertOperationResponse(p, methods, path,
        (direct ? report : new Response({ report })).toBinary()), error => {
        assert.ok(error instanceof VolvoxAIError);
        assert.equal(error.operation, path);
        assert.ok(error.response instanceof Response);
        assert.ok(error.report instanceof p.OperationReport);
        assert.equal(error.report.inputIssue.inputName, 'pixels');
        assert.equal(error.code, p.OperationCode.OPERATION_CODE_INVALID_ARGUMENT);
        return true;
      });
    }
  });

  test(`${profile}: report selection distinguishes query state, missing reports and pending work`, () => {
    const path = name => Object.keys(methods).find(method => method.endsWith(`/${name}`));
    const ok = new p.OperationReport();
    assert.doesNotThrow(() => assertOperationResponse(p, methods, path('PollRequest'),
      new p.RequestInfo({ status: p.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED, report: ok }).toBinary()));
    assert.doesNotThrow(() => assertOperationResponse(p, methods, path('Run'),
      new p.ExecutionResultHandle({ state: p.ResultState.RESULT_STATE_PENDING, report: ok }).toBinary()));
    assert.doesNotThrow(() => assertOperationResponse(p, methods, path('DescribeStatus'),
      new p.StatusDescription({ status: p.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED }).toBinary()));
    assert.throws(() => assertOperationResponse(p, methods, path('PollRequest'), new p.RequestInfo().toBinary()),
      error => error instanceof VolvoxAIError && error.report === null && /no OperationReport/.test(error.message));
  });
}

for (const filename of ['volvoxai.js', 'volvoxai.min.js', 'volvoxai.full.js', 'volvoxai.full.min.js']) {
  const full = filename.includes('.full');
  const moduleUrl = new URL(filename, releaseRoot);
  const wasmUrl = new URL(full ? 'volvoxai.full.wasm' : 'volvoxai.wasm', releaseRoot);

  test(`${filename}: generated calls snapshot requests before lazy initialization`, async () => {
    const api = await import(moduleUrl);
    const { pb: p } = api;
    const Host = full ? api.FullEngineHost : api.EngineHost;
    const host = new Host({ wasmUrl });
    try {
      const request = new p.DescribeStatusRequest({ status: p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT });
      const pending = new api.VxPlatformServiceClient(host).describeStatus(request);
      request.status = p.NativeStatus.NATIVE_STATUS_INTERNAL;
      assert.ok(pending instanceof Promise);
      assert.equal((await pending).status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
      assert.equal('check' in api, false);
      assert.equal('VxInferenceServiceFfiAsync' in api, false);
      assert.equal('VxTrainingServiceClient' in api, full);
    } finally { await host.close(); }
  });

  test(`${filename}: calls execute in C and report failures without check()`, async () => {
    const api = await import(moduleUrl);
    const { pb: p } = api;
    const Host = full ? api.FullEngineHost : api.EngineHost;
    const host = new Host({ wasmUrl });
    const inference = new api.VxInferenceServiceClient(host);
    try {
      const runtime = await inference.createRuntime(new p.CreateRuntimeRequest());
      await assert.rejects(inference.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId })),
        error => error instanceof api.VolvoxAIError && error.code === p.OperationCode.OPERATION_CODE_INVALID_MODEL_SOURCE);
      const graphDocument = new TextEncoder().encode(JSON.stringify({
        format: 'volvox-graph/v1', dimensions: {},
        inputs: { x: { dtype: 'float32', shape: [2] } }, nodes: [], outputs: ['x'],
      }));
      const model = await inference.loadModel(new p.LoadModelRequest({
        runtimeId: runtime.runtimeId, package: new p.ModelPackage({ graphDocument }),
      }));
      const compiled = await inference.compileModel(new p.CompileModelRequest({
        modelId: model.modelId, policy: new p.BackendPolicy({
          mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'],
        }),
      }));
      const inline = new Uint8Array(Float32Array.of(3, -7).buffer);
      const tensor = new p.Tensor({ name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [1n], inline });
      await assert.rejects(inference.run(new p.RunRequest({ compiledModelId: compiled.compiledModelId, inputs: [tensor] })), error => {
        assert.ok(error instanceof api.VolvoxAIError);
        assert.equal(error.report.inputIssue.inputName, 'x');
        assert.equal(error.response.resultId, 0n);
        return true;
      });
      tensor.shape = [2n];
      const result = await inference.run(new p.RunRequest({ compiledModelId: compiled.compiledModelId, inputs: [tensor] }));
      const output = await inference.readOutput(new p.ReadOutputRequest({ resultId: result.resultId, name: 'x' }));
      assert.deepEqual(output.tensor.inline, inline);
      await inference.releaseResult(new p.ResultRef(result));
      const other = new Host({ wasmUrl });
      try {
        await assert.rejects(new api.VxInferenceServiceClient(other).getModelInfo(new p.ModelRef(model)),
          error => error instanceof api.VolvoxAIError && error.status === p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
      } finally { await other.close(); }
    } finally { await host.close(); }
    await assert.rejects(inference.createRuntime(new p.CreateRuntimeRequest()), /closed/);
  });

  test(`${filename}: call deadlines and host close interrupt package preparation`, async () => {
    const api = await import(moduleUrl);
    const { pb: p } = api;
    const Host = full ? api.FullEngineHost : api.EngineHost;
    const host = new Host({ wasmUrl, fetch: () => new Promise(() => {}) });
    const inference = new api.VxInferenceServiceClient(host);
    try {
      const runtime = await inference.createRuntime(new p.CreateRuntimeRequest());
      const request = new p.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'graph.json' });
      await assert.rejects(inference.loadModel(request, { timeoutMs: 10 }),
        error => error instanceof api.RpcError && error.code === 4);
      const pending = inference.loadModel(request);
      const rejected = assert.rejects(pending, error => error instanceof api.RpcError && error.code === 1);
      await new Promise(resolve => setTimeout(resolve, 10));
      await host.close();
      await rejected;
    } finally { await host.close(); }
  });
}


test('cancelling package transfer unmounts completed files and cancels the active reader', async () => {
  const { EngineHost } = await import('../ts/host/EngineHost.js');
  const { ModelControlWasm } = await import('../ts/core/ModelControlWasm.js');
  const { VxInferenceServiceClient, VxPlatformServiceClient } = await import('../runtime/generated/typescript/inference/volvoxai_ffi.js');
  const p = inferencePb;
  const mounted = new Set();
  const mount = ModelControlWasm.prototype.mountFileChunks;
  const unmount = ModelControlWasm.prototype.unmountFile;
  ModelControlWasm.prototype.mountFileChunks = function(path, chunks) { mount.call(this, path, chunks); mounted.add(path); };
  ModelControlWasm.prototype.unmountFile = function(path) { unmount.call(this, path); mounted.delete(path); };
  let entered;
  const reading = new Promise(resolve => { entered = resolve; });
  let signal;
  let readerCancelled = false;
  const host = new EngineHost({ wasmUrl: new URL('volvoxai.wasm', releaseRoot),
    fetch: async (source, options) => {
      if (source === 'graph.json') return { ok: true, text: async () => '{}' };
      signal = options.signal;
      return { ok: true, body: new ReadableStream({
        start() { entered(); },
        cancel() { readerCancelled = true; },
      }) };
    },
  });
  try {
    const inference = new VxInferenceServiceClient(host);
    const runtime = await inference.createRuntime(new p.CreateRuntimeRequest());
    const controller = new AbortController();
    const loading = inference.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId,
      graphPath: 'graph.json', weightPaths: ['weights.safetensors'] }), { signal: controller.signal });
    const rejected = assert.rejects(loading, error => error.name === 'RpcError' && error.code === 1);
    await reading;
    assert.equal(mounted.size, 1);
    controller.abort();
    await rejected;
    assert.equal(signal.aborted, true);
    assert.equal(readerCancelled, true);
    assert.equal(mounted.size, 0, 'cancelled call releases its VFS names while the owner remains open');
    await new VxPlatformServiceClient(host).getPlatformInfo(new p.Empty());
  } finally {
    await host.close();
    ModelControlWasm.prototype.mountFileChunks = mount;
    ModelControlWasm.prototype.unmountFile = unmount;
  }
});

test('initialization observes deadlines and host close without cancelling another host', async () => {
  const { EngineHost } = await import('../ts/host/EngineHost.js');
  const { VxPlatformServiceClient } = await import('../runtime/generated/typescript/inference/volvoxai_ffi.js');
  const bytes = await readFile(new URL('volvoxai.wasm', releaseRoot));
  const originalFetch = globalThis.fetch;
  let finish;
  globalThis.fetch = async () => new Promise(resolve => { finish = () => resolve({ ok: true,
    arrayBuffer: async () => bytes.slice().buffer }); });
  const url = `https://volvoxai-test.invalid/shared-${Date.now()}.wasm`;
  const cancelled = new EngineHost({ wasmUrl: url });
  const retained = new EngineHost({ wasmUrl: url });
  try {
    const first = new VxPlatformServiceClient(cancelled);
    await assert.rejects(first.getPlatformInfo(new inferencePb.Empty(), { timeoutMs: 10 }),
      error => error.name === 'RpcError' && error.code === 4);
    const interrupted = first.getPlatformInfo(new inferencePb.Empty());
    const rejection = assert.rejects(interrupted, error => error.name === 'RpcError' && error.code === 1);
    const surviving = new VxPlatformServiceClient(retained).getPlatformInfo(new inferencePb.Empty());
    await cancelled.close();
    await rejection;
    finish();
    assert.equal((await surviving).profile, inferencePb.BuildProfile.BUILD_PROFILE_INFERENCE);
  } finally {
    finish?.();
    await cancelled.close();
    await retained.close();
    globalThis.fetch = originalFetch;
  }
});
