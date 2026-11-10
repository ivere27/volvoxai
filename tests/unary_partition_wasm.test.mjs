import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {declareStateless, graphRequest, inlineOutput} from './parity/external/execution_fixture.mjs';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));

for (const stem of ['volvoxai.lite', 'volvoxai']) {
  test(`${stem}: unary SIMD results survive tails and input permutation`, async () => {
    const api = await import(`../dist/${version}/${stem}.js`), p = api.pb;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (source, init) => {
      const url = source instanceof URL ? source : new URL(source.url ?? source);
      return url.protocol === 'file:' ? new Response(await readFile(url)) : originalFetch(source, init);
    };
    const Host = api.FullEngineHost ?? api.EngineHost;
    const host = new Host({wasmUrl: new URL(`../dist/${version}/${stem}.wasm`, import.meta.url)});
    const client = new api.VxInferenceServiceClient(host);
    const checked = response => {
      assert.equal(response.report?.status ?? response.status, p.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
      return response;
    };
    try {
      const runtime = checked(await client.createRuntime(new p.CreateRuntimeRequest()));
      for (const op of ['Sin', 'Cos', 'Tanh', 'Sigmoid', 'SiLU', 'GELU']) {
        const graph = declareStateless({format: 'volvox-graph/v1', dimensions: {N: {min: 1, max: 65}},
          inputs: {x: {dtype: 'float32', shape: ['N']}}, outputs: ['y'], nodes: [{id: 'unary', opType: op,
            inputs: {input: 'x'}, outputs: {out: {tensor: 'y', dtype: 'float32', shape: ['N']}}, params: {}}]});
        const model = checked(await client.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
          package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(graph))})})));
        const compiled = checked(await client.compileModel(new p.CompileModelRequest({modelId: model.modelId,
          policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'],
            operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
        const context = checked(await client.createExecutionContext(new p.CreateExecutionContextRequest({
          compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution()})));
        const run = async values => inlineOutput(checked(await client.execute(graphRequest(p, graph, context.contextId,
          [new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [BigInt(values.length)],
            inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength)})],
          {outputs: new p.OutputOptions({storage: p.OutputStorage.OUTPUT_STORAGE_INLINE})}))), 'y');
        const input = Float32Array.from({length: 65}, (_, i) => (i - 32) * 0.137);
        input.set([0, -0, 1e-40, -1e-40, 1e-8, -1e-8, Infinity, -Infinity, NaN, 1e10, -1e10]);
        const expected = await run(input);
        for (const length of [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 65]) {
          const actual = await run(input.slice(0, length));
          assert.deepEqual(new Uint32Array(actual.buffer), new Uint32Array(expected.slice(0, length).buffer), `${op}: length ${length}`);
        }
        const reversed = await run(input.slice().reverse());
        assert.deepEqual(new Uint32Array(reversed.reverse().buffer), new Uint32Array(expected.buffer), `${op}: reversed`);
        for (let i = 0; i < input.length; i++) {
          const actual = await run(input.slice(i, i + 1));
          assert.deepEqual(new Uint32Array(actual.buffer), new Uint32Array(expected.slice(i, i + 1).buffer), `${op}: scalar ${i}`);
        }
        checked(await client.releaseExecutionContext(new p.ExecutionContextRef({contextId: context.contextId})));
        checked(await client.releaseCompiledModel(new p.CompiledModelRef({compiledModelId: compiled.compiledModelId})));
        checked(await client.releaseModel(new p.ModelRef({modelId: model.modelId})));
      }
      checked(await client.releaseRuntime(new p.RuntimeRef({runtimeId: runtime.runtimeId})));
    } finally {
      await host.close();
      globalThis.fetch = originalFetch;
    }
  });
}
