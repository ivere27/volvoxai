/** Validate required cached contexts against retained independent HF prefixes.
 * replay: row preservation and interleaved greedy continuation (WASM/WebGPU).
 * policies: capacity refusal/retry and context-owned RANDOM sampling (WASM).
 * boundaries: capacity refusal/retry and retirement for all 11 families (physical WebGPU).
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const options = { mode: 'replay', backend: 'wasm', only: '' };
const argv = globalThis.Deno?.args ?? process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const key = argv[i].slice(2);
  assert.ok(['bundle', 'wasm', 'fixtures', 'out', 'backend', 'only', 'mode'].includes(key), argv[i]);
  assert.ok(argv[i + 1] && !argv[i + 1].startsWith('--'));
  options[key] = argv[++i];
}
for (const key of ['bundle', 'wasm', 'fixtures', 'out']) assert.ok(options[key], `--${key} required`);
assert.ok(['wasm', 'webgpu'].includes(options.backend));
assert.ok(['replay', 'policies', 'boundaries'].includes(options.mode));
assert.ok(options.mode !== 'policies' || options.backend === 'wasm', 'policies require --backend wasm');
assert.ok(options.mode !== 'boundaries' || options.backend === 'webgpu', 'boundaries require --backend webgpu');
assert.ok(options.mode !== 'boundaries' || !options.only, 'boundaries require the complete 11-family corpus');
const digest = value => createHash('sha256').update(value).digest('hex');
const bytes = value => new Uint8Array(Buffer.from(value, 'base64'));
const trackedPaths = Object.fromEntries(['bundle', 'wasm', 'fixtures'].map(key => [key, options[key]]));
trackedPaths.runner = fileURLToPath(import.meta.url);
const tracked = Object.fromEntries(await Promise.all(Object.entries(trackedPaths).map(async ([key, value]) =>
  [key, digest(await readFile(value))])));
const document = JSON.parse(await readFile(options.fixtures, 'utf8'));
assert.equal(document.format, 'volvox-modern-incremental/v1');
let cases = document.cases.filter(value => !options.only || value.name.includes(options.only));
if (options.mode === 'boundaries') {
  const families = new Set();
  cases = cases.filter(value => {
    const family = value.name.split('-prompt')[0];
    if (families.has(family)) return false;
    families.add(family); return true;
  });
  assert.equal(cases.length, 11);
}
assert.ok(cases.length);
const adapterWitnesses = [];
let gpu, originalRequestAdapter, devicesCreated = 0;
if (options.backend === 'webgpu') {
  gpu = globalThis.navigator?.gpu;
  assert.ok(gpu, 'navigator.gpu required');
  originalRequestAdapter = gpu.requestAdapter;
  gpu.requestAdapter = async function (...args) {
    const adapter = await originalRequestAdapter.apply(gpu, args);
    assert.ok(adapter, 'hardware adapter required');
    const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
    const witness = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, info[key]])
      .filter(([, value]) => typeof value === 'string' && value));
    assert.ok(Object.values(witness).join(' '), 'physical adapter identity required');
    assert.doesNotMatch(Object.values(witness).join(' '), /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|cpu)\b/i);
    assert.notEqual(info.isFallbackAdapter ?? adapter.isFallbackAdapter, true);
    if (adapterWitnesses.length) assert.deepEqual(witness, adapterWitnesses[0]);
    adapterWitnesses.push(witness);
    const originalRequestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (...deviceArgs) => {
      const device = await originalRequestDevice(...deviceArgs);
      devicesCreated++;
      return device;
    };
    return adapter;
  };
}
const api = await import(pathToFileURL(path.resolve(options.bundle)));
const pb = api.pb;
const Host = options.mode === 'boundaries' ? api.FullEngineHost : (api.FullEngineHost ?? api.EngineHost);
const host = new Host({ wasmUrl: path.resolve(options.wasm) });
const client = new api.VxInferenceServiceClient(host);
const receipt = { status: 'FAIL', mode: options.mode, backend: options.backend, sha256: tracked, results: [],
  reference_policy: document.reference_policy ?? 'raw-hf',
  adapter_witnesses: adapterWitnesses, route_attestations: 0, timing_claim: false,
  physical_memory_peak_claim: false, independent_random_oracle: false,
  cross_backend_random_equality_claim: false };
if (options.mode === 'boundaries') receipt.random_claim = false;
const checked = value => {
  assert.equal(value.report?.status ?? value.status, pb.NativeStatus.NATIVE_STATUS_OK, value.report?.message);
  return value;
};
let device;
function attest(report) {
  assert.equal(report.backend, options.backend);
  assert.equal(report.route?.attested, true);
  assert.ok(report.route.activeNodes > 0);
  assert.equal(report.route.selectedNodes, report.route.activeNodes);
  assert.equal(report.route.fallbackNodes, 0);
  assert.equal(report.route.missingNodes, 0);
  assert.equal(report.fallback?.operatorFallbackUsed ?? false, false);
  receipt.route_attestations++;
  if (options.backend === 'webgpu') {
    assert.ok(report.device);
    assert.doesNotMatch(report.device, /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|cpu)\b/i);
    if (device) assert.equal(report.device, device); else device = report.device;
  }
}
function tensor(value) {
  return new pb.Tensor({ name: value.name, shape: value.shape.map(BigInt), inline: value.data,
    dtype: ['I32', 'int32'].includes(value.dtype) ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32 });
}
function bindings(inputs) {
  return inputs.map(value => tensor({ ...value, data: bytes(value.data) }));
}
function numbers(data, integer = false) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  assert.equal(data.byteLength % 4, 0);
  return Array.from({ length: data.byteLength / 4 }, (_, i) => integer ? view.getInt32(i * 4, true) : view.getFloat32(i * 4, true));
}
function setInteger(input, index, value) {
  new DataView(input.data.buffer, input.data.byteOffset, input.data.byteLength).setInt32(index * 4, value, true);
}
const inline = () => new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE });
const retained = [];

async function replay(fixture, compiled, ownedContexts) {
  const initial = fixture.inputs.map(value => ({ ...value, data: bytes(value.data) }));
  const values = initial.map(value => ({ ...value, data: value.data.slice() }));
  const find = name => values.find(value => value.name === name);
  const promptLength = numbers(find('active_lengths').data, true)[0];
  const invariants = fixture.graph.execution.invariant_inputs;
  const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
    compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution({ slots: 1,
      rowMode: pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED, requireIncremental: true }) })));
  ownedContexts.push(context.contextId);
  let previous, maximumError = 0;
  for (let step = 0; step < fixture.expected_logits.length; step++) {
    const row = promptLength - 1 + step;
    if (step) {
      setInteger(find('token_ids'), row, fixture.expected_ids[step - 1]);
      setInteger(find('attention_mask'), row, 1);
      for (const name of ['image_feature_ids', 'image_token_mask']) {
        if (find(name)) setInteger(find(name), row, 0);
      }
    }
    const packet = checked(await client.execute(new pb.ExecuteRequest({ contextId: context.contextId,
      inputs: new pb.InputBindings({ values: values.filter(value => !step || !invariants.includes(value.name)).map(tensor),
        reuse: step ? invariants : [] }),
      scope: new pb.ExecutionScope(step ? { entrypoint: 'transition',
        slotRows: new pb.SlotRows({ slots: [new pb.SlotRow({ position: row })] }) } :
        { entrypoint: 'initialize', full: new pb.Empty() }),
      outputs: new pb.OutputOptions({ ...inline(), selection: new pb.TensorOutputSelection({ names: ['logits'] }) }) })));
    attest(packet.report);
    const output = packet.outputs.find(value => value.name === 'logits');
    assert.ok(output);
    const actual = numbers(output.inline), expected = fixture.expected_logits[step];
    const offset = row * expected.length;
    if (!step && fixture.expected_prefill_logits) {
      assert.equal(fixture.expected_prefill_logits.length, 1);
      const prefill = fixture.expected_prefill_logits[0];
      assert.equal(prefill.length, promptLength);
      for (let active = 0; active < promptLength; active++) {
        assert.equal(prefill[active].length, expected.length);
        for (let i = 0; i < expected.length; i++) {
          const reference = prefill[active][i], value = actual[active * expected.length + i];
          const error = Math.abs(value - reference);
          maximumError = Math.max(maximumError, error);
          assert.ok(Number.isFinite(value) && Number.isFinite(reference) &&
            error <= 2e-4 + 2e-4 * Math.abs(reference),
            `${fixture.name}/prefill${active}/${i}: ${value} != ${reference}`);
        }
      }
    }
    for (let i = 0; i < expected.length; i++) {
      const error = Math.abs(actual[offset + i] - expected[i]);
      maximumError = Math.max(maximumError, error);
      assert.ok(Number.isFinite(actual[offset + i]) && error <= 2e-4 + 2e-4 * Math.abs(expected[i]),
        `${fixture.name}/row${row}/${i}: ${actual[offset + i]} != ${expected[i]}`);
    }
    if (previous) assert.deepEqual(output.inline.slice(0, offset * 4), previous.slice(0, offset * 4));
    previous = output.inline.slice();
  }
  const generation = [], observed = [[], []];
  for (let owner = 0; owner < 2; owner++) {
    const next = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, autoregressive: new pb.AutoregressiveExecution({
        sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY, maxSteps: BigInt(fixture.expected_ids.length) }) })));
    ownedContexts.push(next.contextId); generation.push(next.contextId);
  }
  for (let chunk = 0; chunk < fixture.budgets.length; chunk++) {
    for (let owner = 0; owner < 2; owner++) {
      const packet = checked(await client.generate(new pb.GenerateRequest({ contextId: generation[owner],
        initialInputs: chunk === 0 ? new pb.GenerationInputs({ values: initial.map(tensor) }) : undefined,
        budget: new pb.GenerationBudget({ maxSteps: fixture.budgets[chunk] }), outputs: inline() })));
      attest(packet.report);
      const ids = packet.outputs.find(value => value.name === 'generated_ids');
      assert.ok(ids); observed[owner].push(...numbers(ids.inline, true));
    }
  }
  assert.deepEqual(observed, [fixture.expected_ids, fixture.expected_ids]);
  const route = compiled.report.route;
  return { name: fixture.name, pass: true, maximum_error: maximumError,
    prefill_rows_checked: fixture.expected_prefill_logits ? promptLength : 0, generation_ids: observed,
    route: { provider: route.provider, attested: route.attested, active_nodes: route.activeNodes,
      selected_nodes: route.selectedNodes, fallback_nodes: route.fallbackNodes, missing_nodes: route.missingNodes,
      digest: String(route.digest) } };
}

async function checkPolicies(fixture, compiled, ownedContexts, seen) {
  const record = { name: fixture.name, pass: false };
  async function context(maxSteps, sampling, samplingOptions = {}) {
    const owner = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId,
      autoregressive: new pb.AutoregressiveExecution({ maxSteps: BigInt(maxSteps), sampling, ...samplingOptions }) })));
    ownedContexts.push(owner.contextId);
    return owner.contextId;
  }
  async function generate(contextId, count, initial) {
    const packet = await client.generate(new pb.GenerateRequest({ contextId,
      initialInputs: initial ? new pb.GenerationInputs({ values: bindings(initial) }) : undefined,
      budget: new pb.GenerationBudget({ maxSteps: count }),
      outputs: new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE }) }));
    checked(packet); attest(packet.report);
    assert.equal(packet.outputs.length, 1);
    const output = packet.outputs[0];
    assert.equal(output.dtype, pb.DataType.DATA_TYPE_I32);
    assert.deepEqual(output.shape.map(Number), [1, count]);
    assert.ok(!output.buffer);
    assert.equal(output.inline.byteLength, count * 4);
    const view = new DataView(output.inline.buffer, output.inline.byteOffset, output.inline.byteLength);
    retained.push([output.inline, output.inline.slice()]);
    return Array.from({ length: count }, (_, i) => view.getInt32(i * 4, true));
  }
  async function state(contextId) {
    const info = checked(await client.getExecutionContextInfo(new pb.ExecutionContextRef({ contextId })));
    return { initialized: info.initialized ?? false, total_steps: String(info.totalSteps), state: info.state };
  }
  async function refusal(contextId, count, initial) {
    const before = await state(contextId);
    await assert.rejects(client.generate(new pb.GenerateRequest({ contextId,
      initialInputs: initial ? new pb.GenerationInputs({ values: bindings(initial) }) : undefined,
      budget: new pb.GenerationBudget({ maxSteps: count }),
      outputs: options.mode === 'boundaries' ? inline() : undefined })), error =>
      error.status === pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT &&
      error.response?.progress !== undefined && error.report?.lineage?.contextId === contextId);
    assert.deepEqual(await state(contextId), before);
    return { requested_steps: count, state_before_after: before, status: 'INVALID_ARGUMENT' };
  }
  const family = fixture.name.split('-prompt')[0];
  if (!seen.has(family)) {
    const boundary = fixture.boundary;
    assert.equal(boundary.expected_ids.length, 2);
    const owner = await context(4, pb.SamplingMode.SAMPLING_MODE_GREEDY);
    let capacity;
    if (options.mode === 'boundaries') {
      capacity = boundary.inputs.find(value => value.name === 'token_ids').shape[1];
      const lengths = bytes(boundary.inputs.find(value => value.name === 'active_lengths').data);
      assert.equal(numbers(lengths, true)[0], capacity - 1);
    }
    const refusals = [await refusal(owner, 3, boundary.inputs)];
    const ids = await generate(owner, 1, boundary.inputs);
    assert.deepEqual(ids, boundary.expected_ids.slice(0, 1));
    refusals.push(await refusal(owner, 2));
    ids.push(...await generate(owner, 1));
    assert.deepEqual(ids, boundary.expected_ids);
    refusals.push(await refusal(owner, 1));
    const result = { generated_ids: ids, independent_hf_ids: boundary.expected_ids, refusals };
    if (options.mode === 'boundaries') {
      Object.assign(record, result, { capacity, prompt_length: capacity - 1,
        retry_after_initial_and_continuation_refusal: true });
    } else record.boundary = result;
    seen.add(family);
  }
  if (options.mode === 'policies') {
    const samplingOptions = [{ seed: 7n, temperature: 0.7, topK: 5, topP: 0.8 },
      { seed: 18446744073709551615n, temperature: 1.5, topK: 0, topP: 1 }];
    const steps = fixture.expected_ids.length;
    const whole = [];
    for (const samplingOption of samplingOptions) {
      whole.push(await generate(await context(steps, pb.SamplingMode.SAMPLING_MODE_RANDOM, samplingOption), steps, fixture.inputs));
    }
    const owners = [];
    for (const index of [0, 1, 0]) owners.push(await context(steps, pb.SamplingMode.SAMPLING_MODE_RANDOM, samplingOptions[index]));
    const split = [[], [], []];
    for (let chunk = 0; chunk < fixture.budgets.length; chunk++) {
      for (let owner = 0; owner < owners.length; owner++) {
        split[owner].push(...await generate(owners[owner], fixture.budgets[chunk], chunk ? undefined : fixture.inputs));
      }
    }
    assert.deepEqual(split, [whole[0], whole[1], whole[0]]);
    assert.ok(split.flat().every(value => value >= 0 && value < fixture.expected_logits[0].length));
    record.random = { single_chunk_ids: whole, interleaved_chunk_ids: split, budgets: fixture.budgets,
      sampling_options: samplingOptions.map(value => ({ ...value, seed: String(value.seed) })) };
  }
  record.pass = true; return record;
}

try {
  const runtime = checked(await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 })));
  try {
    const seen = new Set();
    for (const fixture of cases) {
      const ownedContexts = [];
      let model, compiled;
      try {
        model = checked(await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
          package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)),
            weightShards: fixture.weight_shards.map(bytes) }) })));
        if (options.mode === 'replay') {
          const info = checked(await client.getModelInfo(new pb.ModelRef({ modelId: model.modelId })));
          assert.equal(info.execution.incremental.positionInput,
            fixture.graph.execution.incremental.position_input ?? '');
          const controls = fixture.graph.execution.generation.incremental_controls;
          if (controls) assert.equal(info.execution.generation.incrementalControls.positionInput,
            controls.position_input);
          else assert.equal(info.execution.generation.incrementalControls, undefined);
        }
        compiled = checked(await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
          policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
            backends: [options.backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
        attest(compiled.report);
        const result = options.mode === 'replay' ? await replay(fixture, compiled, ownedContexts) :
          await checkPolicies(fixture, compiled, ownedContexts, seen);
        result.reference_policy = fixture.reference_policy ?? document.reference_policy ?? 'raw-hf';
        receipt.results.push(result);
        console.log('PASS', options.backend, options.mode, fixture.name);
      } finally {
        for (const contextId of ownedContexts.reverse()) checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId })));
        if (compiled) checked(await client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId: compiled.compiledModelId })));
        if (model) checked(await client.releaseModel(new pb.ModelRef({ modelId: model.modelId })));
      }
    }
  } finally { checked(await client.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))); }
  for (const [key, hash] of Object.entries(tracked)) assert.equal(digest(await readFile(trackedPaths[key])), hash);
  if (options.backend === 'webgpu') {
    assert.ok(adapterWitnesses.length && devicesCreated > 0, 'actual bridge adapter/device witness required');
    receipt.devices_created = devicesCreated;
  }
  receipt.device = device; receipt.status = 'PASS';
} catch (error) { receipt.error = error.stack ?? String(error); throw error; }
finally {
  await host.close();
  if (gpu) gpu.requestAdapter = originalRequestAdapter;
  if (options.mode !== 'replay') {
    for (const [value, snapshot] of retained) assert.deepEqual(value, snapshot);
    receipt.inline_outputs_survive_owner_retirement = true;
  }
  await mkdir(path.dirname(options.out), { recursive: true });
  await writeFile(options.out, JSON.stringify(receipt, null, 2) + '\n');
}
