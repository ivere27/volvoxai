/** Current WASM control/memory baseline using one retained tiny HF fixture.
 * This is not a full-checkpoint, browser or performance regression qualifier.
 * Invoke each profile in a separate process; never mix instrumented timings.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

const [profile, mode, corpusPath, outputPath] = process.argv.slice(2);
assert.ok(['inference', 'full'].includes(profile));
assert.ok(['timing', 'memory'].includes(mode));
assert.ok(corpusPath && outputPath, 'usage: PROFILE timing|memory GENERATION_CORPUS NEW_JSON');
try { await fs.stat(outputPath); throw new Error('refusing to overwrite evidence'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const root = process.cwd();
const version = JSON.parse(await fs.readFile('package.json', 'utf8')).version;
const stem = profile === 'inference' ? 'volvoxai.lite' : 'volvoxai';
const entry = path.join(root, 'dist', version, `${stem}.js`);
const wasm = path.join(root, 'dist', version, `${stem}.wasm`);
const files = [entry, wasm, path.resolve(corpusPath), path.resolve('package.json'), path.resolve(process.argv[1])];
const hashes = async () => Object.fromEntries(await Promise.all(files.map(async file =>
  [file, crypto.createHash('sha256').update(await fs.readFile(file)).digest('hex')])));
const before = await hashes();
const corpus = JSON.parse(await fs.readFile(corpusPath, 'utf8'));
assert.equal(corpus.format, 'volvox-hf-generation-fixtures/v1');
const fixture = corpus.cases.find(value => value.name === 'qwen3_5_text');
assert.ok(fixture, 'fixed tiny qwen3_5_text fixture is required');
assert.deepEqual(fixture.budgets, [1, 2]);
assert.equal(fixture.expected_ids.length, 3);
const result = { format: 'volvox-wasm-control-baseline/v1', status: 'failed', profile, mode,
  fixture: fixture.name, sha256_before: before, cold: {}, cycles: [], resource_snapshots: [],
  numerical_tensor_capture: false, service_actions: 0, whole_peak_qualified: false,
  whole_model_qualified: false, browser_qualified: false, regression_allowance: null,
  measurement_policy: { warmup: 2, repeats: 5, chunks: [1, 2], output: 'INLINE I32',
    timing_scope: 'Completed public API calls; Node process, uncontrolled background and OS cache',
    memory_scope: 'Exact public WASM partition at boundaries, observed boundary maxima; not an allocated-byte peak between observations' } };
const api = await import(pathToFileURL(entry));
const pb = api.pb;
const start = performance.now();
const host = new (api.FullEngineHost ?? api.EngineHost)({ wasmUrl: wasm });
const client = new api.VxInferenceServiceClient(host);
const profiling = new api.VxProfilingServiceClient(host);
const checked = packet => {
  assert.equal(packet.report?.status ?? packet.status, pb.NativeStatus.NATIVE_STATUS_OK, packet.report?.message);
  return packet;
};
const route = report => {
  assert.equal(report.backend, 'wasm');
  assert.ok(report.route.builtin && report.route.attested);
  assert.equal(report.route.selectedNodes, report.route.activeNodes);
  assert.ok(report.route.activeNodes > 0);
  assert.equal(report.route.fallbackNodes, 0);
  assert.equal(report.route.missingNodes, 0);
  assert.equal(report.fallback?.operatorFallbackUsed ?? false, false);
};
const snapshot = async phase => {
  if (mode !== 'memory') return;
  const value = checked(await profiling.getResourceSnapshot(new pb.GetResourceSnapshotRequest({ module: new pb.Empty() }))).snapshot;
  assert.equal(value.wasm.status, pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE);
  assert.equal(value.wasm.linearBytes, value.wasm.allocatedBytes + value.wasm.freeBytes +
    value.wasm.allocatorMetadataBytes + value.wasm.modulePrefixBytes + value.wasm.pageSlackBytes + value.wasm.untrackedBytes);
  result.resource_snapshots.push({ phase, snapshot: value.toJson(), node_process: process.memoryUsage() });
};
const timed = async (name, action) => {
  const start = performance.now();
  const value = checked(await action());
  if (mode === 'timing') result.cold[name] = performance.now() - start;
  await snapshot(name);
  return value;
};
const decode = value => new Uint8Array(Buffer.from(value, 'base64'));
let runtime, model, compiled;
let failure;
try {
  runtime = await timed('runtime_create_ms', () => client.createRuntime(new pb.CreateRuntimeRequest()));
  if (mode === 'timing') result.cold.host_and_runtime_ms = performance.now() - start;
  model = await timed('load_ms', () => client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
    package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)),
      weightShards: fixture.weight_shards.map(decode) }) })));
  compiled = await timed('compile_ms', () => client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
    policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
  route(compiled.report);
  const initial = fixture.inputs.map(value => new pb.Tensor({ name: value.name, shape: value.shape.map(BigInt),
    dtype: pb.DataType.DATA_TYPE_I32, inline: decode(value.data) }));
  const cycle = async ordinal => {
    const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, autoregressive: new pb.AutoregressiveExecution({
        maxSteps: 3n, sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY }) })));
    const ids = [], calls = [];
    try {
      await snapshot(`cycle${ordinal}:context_created`);
      for (let index = 0; index < fixture.budgets.length; index++) {
        const start = performance.now();
        const packet = checked(await client.generate(new pb.GenerateRequest({ contextId: context.contextId,
          initialInputs: index === 0 ? new pb.GenerationInputs({ values: initial }) : undefined,
          budget: new pb.GenerationBudget({ maxSteps: BigInt(fixture.budgets[index]) }),
          outputs: new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE }) })));
        const duration = performance.now() - start;
        route(packet.report);
        assert.equal(packet.outputs.length, 1);
        const output = packet.outputs[0];
        assert.equal(output.dtype, pb.DataType.DATA_TYPE_I32);
        assert.equal(output.inline.byteLength, fixture.budgets[index] * 4);
        const view = new DataView(output.inline.buffer, output.inline.byteOffset, output.inline.byteLength);
        for (let offset = 0; offset < output.inline.byteLength; offset += 4) ids.push(view.getInt32(offset, true));
        calls.push({ steps: fixture.budgets[index], call_ms: mode === 'timing' ? duration : null });
        await snapshot(`cycle${ordinal}:chunk${index}`);
      }
      assert.deepEqual(ids, fixture.expected_ids);
    } finally {
      checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId })));
    }
    await snapshot(`cycle${ordinal}:context_released`);
    return { ordinal, ids, calls };
  };
  result.first = await cycle('first');
  if (mode === 'timing') for (let i = -2; i < 5; i++) {
    const value = await cycle(i);
    if (i >= 0) result.cycles.push(value);
  } else result.cycles.push(await cycle(0));
  result.status = 'observed';
} catch (error) {
  failure = error;
  result.error = { message: error.message, stack: error.stack };
} finally {
  const cleanup = async (phase, action) => {
    try { await action(); }
    catch (error) {
      failure ??= error;
      result.status = 'failed';
      (result.cleanup_errors ??= []).push({ phase, message: error.message });
    }
  };
  if (compiled) await cleanup('compiled', async () => checked(await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId }))));
  if (model) await cleanup('model', async () => checked(await client.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  if (runtime) await cleanup('runtime', async () => checked(await client.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))));
  await cleanup('retirement_snapshot', () => snapshot('all_runtime_model_context_owners_released'));
  if (mode === 'memory') result.observed_linear_max_bytes = Math.max(...result.resource_snapshots.map(value =>
    Number(value.snapshot.wasm?.linearBytes ?? 0)));
  await cleanup('host', () => host.close());
  result.sha256_after = await hashes();
  try { assert.deepEqual(result.sha256_after, before); }
  catch (error) { result.status = 'failed'; result.identity_error = error.message; failure ??= error; }
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ status: result.status, profile, mode, fixture: result.fixture }));
}
if (failure) throw failure;
