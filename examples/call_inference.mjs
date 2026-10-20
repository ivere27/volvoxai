import { loadRuntimeRelease } from './common/RuntimeRelease.js';

const { api, wasmUrl } = await loadRuntimeRelease('inference');
const { EngineHost, VxInferenceServiceClient, pb } = api;

const host = new EngineHost({ wasmUrl });
const inference = new VxInferenceServiceClient(host);
try {
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1', dimensions: {},
    execution: {stateless: {}, entrypoints: [{name: 'evaluate', effect: 1, allowed_ranges: [1], inputs: ['x'], outputs: ['x']}]},
    inputs: { x: { dtype: 'float32', shape: [2] } }, nodes: [], outputs: ['x'],
  }));
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({ graphDocument }),
  }));
  const compiled = await inference.compileModel(new pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new pb.BackendPolicy({
      mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'],
    }),
  }));
  const context = await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
    compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution(),
  }));
  const result = await inference.execute(new pb.ExecuteRequest({
    contextId: context.contextId,
    scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}),
    inputs: new pb.InputBindings({values: [new pb.Tensor({
      name: 'x', dtype: pb.DataType.DATA_TYPE_F32, shape: [2n],
      inline: new Uint8Array(Float32Array.of(3, -7).buffer),
    })]}),
  }), { timeoutMs: 1000 });
  const output = result.outputs.find(tensor => tensor.name === 'x');
  console.log(Array.from(new Float32Array(output.inline.slice().buffer)));
} finally {
  // Close cancels calls and retires every remaining C handle owned by this host.
  await host.close();
}
