import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { statelessGraph } from '../tools/proto_fixture.mjs';
import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { EngineHost } from '../ts/host/EngineHost.js';
import { FullEngineHost } from '../ts/full.js';
import { VxInferenceServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import * as p from '../runtime/generated/typescript/volvoxai_lite.js';

const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const H = 4, D = 128;
const node = (name, opType, inputs, shape, params = {}) => ({
  id: name, opType, inputs, params,
  outputs: { out: { tensor: name, shape, dtype: 'float32' } },
});
const graph = statelessGraph({
  format: 'volvox-graph/v1', dimensions: { S: { min: 1, max: 5 } },
  inputs: { x: { shape: [1, 'S', H * D], dtype: 'float32' },
    positions: { shape: ['S'], dtype: 'int32' }, norm: { shape: [D], dtype: 'float32' } },
  nodes: [
    node('split', 'Reshape', { input: 'x' }, ['S', H, D], { shape: ['S', H, D] }),
    node('head_major', 'Transpose', { input: 'split' }, [H, 'S', D], { perm: [1, 0, 2] }),
    node('normalized', 'RMSNorm', { input: 'head_major', weight: 'norm' }, [H, 'S', D], { eps: 1e-6 }),
    node('rotated', 'RoPE', { input: 'normalized', position_ids: 'positions' }, [H, 'S', D],
      { rotary_dim: D, theta: 1000000, interleaved: false }),
    node('row_major', 'Transpose', { input: 'rotated' }, ['S', H, D], { perm: [1, 0, 2] }),
    node('y', 'Reshape', { input: 'row_major' }, [1, 'S', H * D], { shape: [1, 'S', H * D] }),
  ], outputs: ['y'],
});
const graphBytes = new TextEncoder().encode(JSON.stringify(graph));
const ok = response => {
  assert.equal(response.report?.status ?? response.status, p.NativeStatus.NATIVE_STATUS_OK,
    response.report?.message);
  return response;
};

function values(S) {
  const x = Float32Array.from({ length: S * H * D }, (_, i) =>
    Math.sin(i * .17) + .25 * Math.cos(i * .31));
  const positions = Int32Array.from({ length: S }, (_, i) => 7 + i * 3);
  const norm = Float32Array.from({ length: D }, (_, i) => .7 + .6 * i / (D - 1));
  return { x, positions, norm };
}

function reference({ x, positions, norm }) {
  // Independent row-major formula; no runtime reshape or RoPE helper is used.
  const out = new Float32Array(x.length), half = D / 2;
  for (let s = 0; s < positions.length; s++) for (let h = 0; h < H; h++) {
    const base = (s * H + h) * D;
    let variance = 0;
    for (let d = 0; d < D; d++) variance += x[base + d] ** 2;
    const inverse = 1 / Math.sqrt(variance / D + 1e-6);
    for (let d = 0; d < half; d++) {
      const angle = positions[s] / 1000000 ** (2 * d / D);
      const left = x[base + d] * inverse * norm[d];
      const right = x[base + d + half] * inverse * norm[d + half];
      out[base + d] = left * Math.cos(angle) - right * Math.sin(angle);
      out[base + d + half] = right * Math.cos(angle) + left * Math.sin(angle);
    }
  }
  return out;
}

for (const profile of ['inference', 'full']) {
  test(`${profile} WASM: dynamic head norm and half-split RoPE D128`, async () => {
    const wasmUrl = new URL(`../dist/${version}/volvoxai${profile === 'full' ? '' : '.lite'}.wasm`, import.meta.url);
    const host = profile === 'full' ? new FullEngineHost({ wasmUrl }) : new EngineHost({ wasmUrl });
    try {
      const engine = new VxInferenceServiceClient(reportTransport(host));
      const runtime = ok(await engine.createRuntime(new p.CreateRuntimeRequest()));
      const model = ok(await engine.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId,
        package: new p.ModelPackage({ graphDocument: graphBytes }) })));
      const compiled = ok(await engine.compileModel(new p.CompileModelRequest({ modelId: model.modelId,
        policy: new p.BackendPolicy({ mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: ['wasm'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
      assert.equal(compiled.report.backend, 'wasm');
      assert.equal(compiled.report.compilation.tierFallbackUsed, false);
      const freshContext = async () => ok(await engine.createExecutionContext(new p.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution() })));
      let context = await freshContext();
      for (const S of [1, 3, 5, 1]) {
        const data = values(S);
        const execute = async positions => engine.execute(new p.ExecuteRequest({
          contextId: context.contextId,
          inputs: new p.InputBindings({ values: [
            new p.Tensor({ name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [1n, BigInt(S), BigInt(H * D)],
              inline: new Uint8Array(data.x.buffer) }),
            new p.Tensor({ name: 'positions', dtype: p.DataType.DATA_TYPE_I32, shape: [BigInt(S)],
              inline: new Uint8Array(positions.buffer) }),
            new p.Tensor({ name: 'norm', dtype: p.DataType.DATA_TYPE_F32, shape: [BigInt(D)],
              inline: new Uint8Array(data.norm.buffer) }),
          ] }), scope: new p.ExecutionScope({ entrypoint: 'evaluate', full: new p.Empty() }),
          outputs: new p.OutputOptions({ storage: p.OutputStorage.OUTPUT_STORAGE_INLINE }),
        }));
        if (S === 3) {
          const invalid = data.positions.slice(); invalid[1] = -1;
          const failure = await execute(invalid);
          assert.equal(failure.report.status, p.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED);
          assert.equal(failure.report.stage, p.OperationStage.OPERATION_STAGE_EXECUTE);
          const state = ok(await engine.getExecutionContextInfo(new p.ExecutionContextRef(context)));
          assert.equal(state.state, p.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
          ok(await engine.releaseExecutionContext(new p.ExecutionContextRef(context)));
          context = await freshContext();
        }
        const result = ok(await execute(data.positions));
        assert.equal(result.report.backend, 'wasm');
        assert.equal(result.report.fallback.operatorFallbackUsed, false);
        assert.deepEqual(result.outputs[0].shape, [1n, BigInt(S), BigInt(H * D)]);
        const actual = new Float32Array(result.outputs[0].inline.slice().buffer);
        const expected = reference(data);
        assert.equal(actual.length, expected.length);
        for (let i = 0; i < actual.length; i++)
          assert.ok(Number.isFinite(actual[i]) && Math.abs(actual[i] - expected[i]) <=
            2e-5 + 2e-5 * Math.abs(expected[i]), `S=${S} element=${i}: ${actual[i]} vs ${expected[i]}`);
      }
      ok(await engine.releaseExecutionContext(new p.ExecutionContextRef(context)));
      ok(await engine.releaseCompiledModel(new p.CompiledModelRef(compiled)));
      ok(await engine.releaseModel(new p.ModelRef(model)));
      ok(await engine.releaseRuntime(new p.RuntimeRef(runtime)));
    } finally { await host.close(); }
  });
}
