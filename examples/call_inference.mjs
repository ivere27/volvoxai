import { EngineHost, VxInferenceServiceClient, pb } from '../dist/0.5.0/volvoxai.js';

const host = new EngineHost();
const inference = new VxInferenceServiceClient(host);
try {
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1', dimensions: {},
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
  const result = await inference.run(new pb.RunRequest({
    compiledModelId: compiled.compiledModelId,
    inputs: [new pb.Tensor({
      name: 'x', dtype: pb.DataType.DATA_TYPE_F32, shape: [2n],
      inline: new Uint8Array(Float32Array.of(3, -7).buffer),
    })],
  }), { timeoutMs: 1000 });
  const output = await inference.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'x' }));
  console.log(Array.from(new Float32Array(output.tensor.inline.slice().buffer)));
} finally {
  // Close cancels calls and retires every remaining C handle owned by this host.
  await host.close();
}
