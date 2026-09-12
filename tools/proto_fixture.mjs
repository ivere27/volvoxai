import { reportTransport, checkedReport } from './proto_report_fixture.mjs';
/** Development fixtures over the generated API; never included in a release. */
import assert from 'node:assert/strict';
import { EngineHost } from '../ts/host/EngineHost.js';
import { FullEngineHost } from '../ts/full.js';
import * as p from '../runtime/generated/typescript/volvoxai_lite.js';
import { VxInferenceServiceClient, VxSchedulerServiceClient,
  VxPlanningServiceClient, VxQuantizationServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
export { p };
export function ok(value) {
  const report = value.report ?? value;
  assert.equal(report.status, p.NativeStatus.NATIVE_STATUS_OK, report.message);
  return value;
}
export function tensors(inputs) {
  return Object.entries(inputs).map(([name, {data, shape}]) => new p.Tensor({name,
    shape: shape.map(BigInt), dtype: data instanceof Int32Array ? p.DataType.DATA_TYPE_I32 : p.DataType.DATA_TYPE_F32,
    inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength)}));
}
export function safetensors(weights = []) {
  let offset = 0;
  const header = {};
  for (const weight of weights) {
    header[weight.name] = {dtype: 'F32', shape: weight.shape,
      data_offsets: [offset, offset + weight.data.byteLength]};
    offset += weight.data.byteLength;
  }
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - new TextEncoder().encode(json).length % 8) % 8);
  const prefix = new TextEncoder().encode(json), result = new Uint8Array(8 + prefix.length + offset);
  new DataView(result.buffer).setBigUint64(0, BigInt(prefix.length), true);
  result.set(prefix, 8); offset = 8 + prefix.length;
  for (const weight of weights) {
    result.set(new Uint8Array(weight.data.buffer, weight.data.byteOffset, weight.data.byteLength), offset);
    offset += weight.data.byteLength;
  }
  return result;
}
export async function fixture({wasmUrl, scheduled = false, full = false} = {}) {
  const assets = new Map(); let sequence = 0;
  const host = new (full ? FullEngineHost : EngineHost)({wasmUrl, fetch: async url => {
    const data = assets.get(String(url));
    assert.ok(data, `unknown fixture asset ${url}`);
    return {ok: true, arrayBuffer: async () => data.slice().buffer};
  }});
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const scheduler = new VxSchedulerServiceClient(reportTransport(host));
  const planning = new VxPlanningServiceClient(reportTransport(host));
  const quantization = new VxQuantizationServiceClient(reportTransport(host));
  const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({executionMode:
    scheduled ? p.ExecutionMode.EXECUTION_MODE_SCHEDULED : p.ExecutionMode.EXECUTION_MODE_DIRECT})));
  return {host, inference, scheduler, planning, quantization, runtime,
    async load(document, weights = new Uint8Array()) {
      const graphPath = `fixture-${++sequence}/graph.json`, weightPath = `fixture-${sequence}/model.safetensors`;
      assets.set(graphPath, new TextEncoder().encode(JSON.stringify(document)));
      if (weights.length) assets.set(weightPath, weights);
      try {
        return ok(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
          graphPath, weightPaths: weights.length ? [weightPath] : []})));
      } finally { assets.delete(graphPath); assets.delete(weightPath); }
    },
    async compile(model, backend = 'wasm') {
      return ok(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
        policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: [backend], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
    },
    async read(result, name) {
      const ready = ok(await inference.getResult(new p.ResultRef(result)));
      assert.equal(ready.state, p.ResultState.RESULT_STATE_READY);
      const bytes = ok(await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name}))).tensor.inline;
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    },
    close: async () => (await host.close()),
  };
}
