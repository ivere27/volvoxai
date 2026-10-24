/** Replay independent NumPy primitive oracles through both public WASM profiles.
 * Create the input corpus with the opt-in public Python numerical fixture.
 * This validates the shared CPU implementation, not WebGPU or browser capacity.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { readPackageVersion } from '../release_version.mjs';

const root = process.cwd();
const version = readPackageVersion(root);
const corpusPath = process.argv[2] ?? 'build/llm/modern-primitives-public/wasm-cases.json';
const receiptPath = process.argv[3] ?? 'build/llm/architecture-expansion/wasm-modern.json';
const generationPath = process.argv[4] ?? 'build/llm/architecture-expansion/wasm-generation-cases.json';
const document = JSON.parse(await fs.readFile(corpusPath, 'utf8'));
const cases = Array.isArray(document) ? document : document.cases;
assert.ok(Array.isArray(cases) && cases.length > 0, 'a nonempty independent oracle corpus is required');
const generationCases = JSON.parse(await fs.readFile(generationPath, 'utf8')).cases;
assert.ok(Array.isArray(generationCases) && generationCases.length > 0, 'a nonempty HF generation corpus is required');
const bytes = data => new Uint8Array(Buffer.from(data, 'base64'));
const results = [];

function numeric(data, dtype) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const width = dtype === 'F64' ? 8 : 4;
  assert.equal(data.byteLength % width, 0);
  return Array.from({ length: data.byteLength / width }, (_, index) =>
    dtype === 'I32' ? view.getInt32(index * width, true) :
      dtype === 'F64' ? view.getFloat64(index * width, true) : view.getFloat32(index * width, true));
}

for (const profile of ['inference', 'full']) {
  const stem = `volvoxai${profile === 'inference' ? '.lite' : ''}`;
  const api = await import(pathToFileURL(path.join(root, 'dist', version, `${stem}.js`)));
  const pb = api.pb;
  const host = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl: path.join(root, 'dist', version, `${stem}.wasm`) });
  const client = new api.VxInferenceServiceClient(host);
  const checked = packet => {
    assert.equal(packet.report?.status ?? packet.status, pb.NativeStatus.NATIVE_STATUS_OK, packet.report?.message);
    return packet;
  };
  const policy = () => new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
    backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID });
  const tensor = value => new pb.Tensor({ name: value.name, shape: value.shape.map(BigInt),
    dtype: ['I32', 'int32'].includes(value.dtype) ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32, inline: bytes(value.data) });
  try {
    const runtime = checked(await client.createRuntime(new pb.CreateRuntimeRequest()));
    for (const fixture of cases) {
      const model = checked(await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)),
          weightShards: fixture.weight_shards.map(bytes) }) })));
      const compiled = checked(await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId, policy: policy(),
        packedArithmetic: fixture.packed_arithmetic ?? pb.PackedArithmetic.PACKED_ARITHMETIC_F32 })));
      const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution() })));
      const packet = checked(await client.execute(new pb.ExecuteRequest({ contextId: context.contextId,
        inputs: new pb.InputBindings({ values: fixture.inputs.map(tensor) }),
        scope: new pb.ExecutionScope({ entrypoint: 'evaluate', full: new pb.Empty() }),
        outputs: new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE }) })));
      assert.equal(packet.report.backend, 'wasm');
      assert.equal(packet.report.route.attested, true);
      assert.equal(packet.report.route.selectedNodes, packet.report.route.activeNodes);
      assert.equal(packet.report.route.fallbackNodes, 0);
      assert.equal(packet.report.route.missingNodes, 0);
      assert.equal(packet.report.fallback?.operatorFallbackUsed ?? false, false);
      let maximumError = 0;
      for (const expected of fixture.expected) {
        const actual = packet.outputs.find(output => output.name === expected.name);
        assert.ok(actual, `${fixture.name}/${expected.name}`);
        assert.deepEqual(actual.shape, expected.shape.map(BigInt));
        assert.equal(actual.dtype, expected.dtype === 'I32' ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32);
        const observed = numeric(actual.inline, expected.dtype === 'I32' ? 'I32' : 'F32');
        const oracle = numeric(bytes(expected.data), expected.dtype);
        assert.equal(observed.length, oracle.length);
        for (let index = 0; index < observed.length; index++) {
          if (!Number.isFinite(oracle[index])) {
            assert.ok(Number.isNaN(oracle[index]) ? Number.isNaN(observed[index]) :
              Object.is(observed[index], oracle[index]), `${profile}/${fixture.name}/${expected.name}/${index}: nonfinite contract`);
            continue;
          }
          const error = Math.abs(observed[index] - oracle[index]);
          maximumError = Math.max(maximumError, error);
          assert.ok(Number.isFinite(observed[index]) && (expected.dtype === 'I32' ? error === 0 :
            error <= 1e-5 + 1e-5 * Math.abs(oracle[index])), `${profile}/${fixture.name}/${expected.name}/${index}`);
        }
      }
      results.push({ profile, name: fixture.name, maximum_error: maximumError, pass: true });
      checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId })));
      checked(await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId })));
      checked(await client.releaseModel(new pb.ModelRef({ modelId: model.modelId })));
    }
    for (const fixture of generationCases) {
      const model = checked(await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)),
          weightShards: fixture.weight_shards.map(bytes) }) })));
      const compiled = checked(await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId, policy: policy(),
        packedArithmetic: fixture.packed_arithmetic ?? pb.PackedArithmetic.PACKED_ARITHMETIC_F32 })));
      const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, autoregressive: new pb.AutoregressiveExecution({
          sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY, maxSteps: BigInt(fixture.expected_ids.length) }) })));
      const generated = [];
      for (let index = 0; index < fixture.budgets.length; index++) {
        const packet = checked(await client.generate(new pb.GenerateRequest({ contextId: context.contextId,
          initialInputs: index === 0 ? new pb.GenerationInputs({ values: fixture.inputs.map(tensor) }) : undefined,
          budget: new pb.GenerationBudget({ maxSteps: fixture.budgets[index] }),
          outputs: new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE }) })));
        assert.equal(packet.report.backend, 'wasm');
        assert.equal(packet.report.fallback?.operatorFallbackUsed ?? false, false);
        const ids = packet.outputs.find(output => output.name === 'generated_ids');
        assert.ok(ids, `${fixture.name} generated IDs`);
        assert.equal(ids.dtype, pb.DataType.DATA_TYPE_I32);
        generated.push(...numeric(ids.inline, 'I32'));
      }
      assert.deepEqual(generated, fixture.expected_ids, `${profile}/${fixture.name} HF greedy feedback`);
      results.push({ profile, name: fixture.name, kind: 'full-prefix-generation', generated_ids: generated, pass: true });
      checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId })));
      checked(await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId })));
      checked(await client.releaseModel(new pb.ModelRef({ modelId: model.modelId })));
    }
    checked(await client.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId })));
  } finally {
    await host.close();
  }
}
await fs.writeFile(receiptPath, JSON.stringify({ status: 'PASS', corpus: corpusPath, generation_corpus: generationPath,
  oracle: 'independent NumPy F64/I32 expectations and official HF repeated-prefix greedy IDs', results }, null, 2) + '\n');
console.log(JSON.stringify({ status: 'PASS', cases: results.length }));
