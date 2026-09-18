/** Retained-session receipt latency through the released proto/WASM host. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const options = { profile: 'inference', warmup: '3', repeat: '15' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  assert.ok(['spec', 'model', 'variant', 'backend', 'profile', 'warmup', 'repeat', 'out'].includes(key));
  assert.ok(process.argv[i + 1]);
  options[key] = process.argv[i + 1];
}
assert.ok(options.spec && options.out);
assert.ok(['digit', 'vqa'].includes(options.model));
assert.ok(['fp32', 'int8', 'ptq', 'ptq-wasm'].includes(options.variant));
assert.ok(['wasm', 'webgpu'].includes(options.backend));
assert.ok(['inference', 'full'].includes(options.profile));
assert.ok(options.backend !== 'webgpu' || options.profile === 'full');
const warmup = Number(options.warmup), repeat = Number(options.repeat);
assert.ok(Number.isInteger(warmup) && warmup >= 1 && Number.isInteger(repeat) && repeat >= 1);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const signature = value => hash(JSON.stringify(value));
const specBytes = readFileSync(options.spec), spec = JSON.parse(specBytes);
const prefix = spec.reference_kind === 'historical-four-token-prefix';
const generation = spec.generation ?? { max_new_tokens: 191, require_eos: true };
assert.deepEqual(generation, prefix ? { max_new_tokens: 4, require_eos: false } : { max_new_tokens: 191, require_eos: true });
assert.ok(!prefix || options.model === 'vqa');
const version = JSON.parse(readFileSync(path.join(root, 'package.json'))).version;
const stem = options.profile === 'full' ? 'volvoxai' : 'volvoxai.lite';
const bundle = path.join(root, 'dist', version, `${stem}.js`);
const wasm = path.join(root, 'dist', version, `${stem}.wasm`);
const api = await import(pathToFileURL(bundle));
const { pb, VxInferenceServiceClient } = api;
const contracts = spec.models[options.model].variants[options.variant];
const f32 = (data, shape) => ({ data, shape, dtype: pb.DataType.DATA_TYPE_F32 });
const i32 = (values, shape) => ({ data: Int32Array.from(values), shape, dtype: pb.DataType.DATA_TYPE_I32 });
const cases = spec.cases.map(item => {
  const bytes = readFileSync(path.join(root, item.pixels));
  assert.equal(hash(bytes), item.sha256);
  assert.equal(bytes.length, 320 * 672 * 4);
  return { ...item, pixels: f32(new Float32Array(Uint8Array.from(bytes).buffer), [1, 1, 320, 672]) };
});
const routeKey = [options.model, options.backend, options.profile, options.variant].join('/');
const expected = spec.expected[routeKey];
assert.ok(expected, 'the frozen 2000-case reference is required');
let adapterInfo, restoreAdapter;
if (options.backend === 'webgpu') {
  const gpu = globalThis.navigator?.gpu;
  assert.ok(gpu, 'WebGPU is unavailable');
  const original = gpu.requestAdapter;
  restoreAdapter = () => { gpu.requestAdapter = original; };
  gpu.requestAdapter = async (...args) => {
    const adapter = await original.apply(gpu, args);
    assert.ok(adapter, 'no physical WebGPU adapter');
    adapterInfo = Object.fromEntries(['vendor', 'architecture', 'device', 'description']
      .map(name => [name, String(adapter.info?.[name] ?? '')]));
    const identity = Object.values(adapterInfo).join(' ');
    assert.match(identity, /nvidia|3090/i);
    assert.doesNotMatch(identity, /swiftshader|lavapipe|llvmpipe|software/i);
    return adapter;
  };
}
function route(report) {
  assert.equal(report.backend, options.backend);
  assert.equal(report.route?.provider, options.backend);
  assert.equal(report.route?.attested, true);
  assert.equal(report.route.activeNodes, report.route.selectedNodes);
  assert.equal(report.route.fallbackNodes, 0);
  assert.equal(report.route.missingNodes, 0);
  assert.ok(!report.fallback?.operatorFallbackUsed && !report.compilation?.tierFallbackUsed);
  return { backend: options.backend, attested: true, active_nodes: report.route.activeNodes,
    selected_nodes: report.route.selectedNodes, fallback_nodes: 0, missing_nodes: 0 };
}
function stats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const quantile = q => {
    const at = (sorted.length - 1) * q, lo = Math.floor(at);
    return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (at - lo);
  };
  return { n: values.length, median_ms: quantile(.5), p95_ms: quantile(.95),
    mean_ms: values.reduce((a, b) => a + b, 0) / values.length, min_ms: sorted[0], max_ms: sorted.at(-1) };
}
function argmax(values, begin = 0, length = values.length) {
  let best = begin;
  for (let i = begin + 1; i < begin + length; i++) if (values[i] > values[best]) best = i;
  return best - begin;
}
function decoderFeed(encoded) {
  const feed = Object.fromEntries(Object.entries(encoded).filter(([name]) => name.startsWith('cross_') || name === 'memory_padding_mask'));
  Object.assign(feed, { decoder_input_ids: i32([1], [1, 1]), position_ids: i32([0], [1]),
    family_ids: encoded.selected_family_ids, past_padding_mask: i32([1], [1, 1]) });
  for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) {
    feed[`past_${kind}_${layer}`] = f32(new Float32Array(320), [1, 8, 1, 40]);
  }
  return feed;
}
const Constructor = options.profile === 'full' ? api.FullEngineHost : api.EngineHost;
const host = new Constructor({ wasmUrl: pathToFileURL(wasm) });
const client = new VxInferenceServiceClient(host);
const contexts = {}, routes = {}, pendingReports = [], pendingTimings = [];
function executionMetrics() {
  const output = {};
  for (const [role, values] of pendingTimings) for (const [name, value] of Object.entries(values)) {
    output[name] = (output[name] ?? 0) + value;
    if (role !== 'digit') output[`${role}_${name}`] = (output[`${role}_${name}`] ?? 0) + value;
  }
  pendingTimings.length = 0;
  return output;
}
try {
  const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 }));
  for (const [role, contract] of Object.entries(contracts)) {
    const graph = readFileSync(path.join(root, contract.graph));
    const weights = readFileSync(path.join(root, contract.weights));
    assert.equal(hash(graph), spec.file_hashes[contract.graph]);
    assert.equal(hash(weights), spec.file_hashes[contract.weights]);
    const model = await client.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
      package: new pb.ModelPackage({ graphDocument: graph, weightShards: [weights] }) }));
    const compiled = await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: [options.backend], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) }));
    routes[role] = { compilation: route(compiled.report) };
    contexts[role] = (await client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: compiled.compiledModelId }))).contextId;
  }
  async function execute(role, feed, needCache = true) {
    const contract = contracts[role];
    const request = new pb.ExecuteRequest({ contextId: contexts[role],
      inputs: Object.entries(feed).map(([name, value]) => new pb.Tensor({ name: contract.inputs[name],
        shape: value.shape.map(BigInt), dtype: value.dtype,
        inline: new Uint8Array(value.data.buffer, value.data.byteOffset, value.data.byteLength) })) });
    const started = performance.now();
    const execution = await client.execute(request);
    const output = {};
    let report = execution.report, executeMs, releaseMs, readOutputMs = 0, getResultCalls = 0, readOutputCalls = 0;
    try {
      const deadline = performance.now() + 120000;
      let state = execution.state;
      while (state === pb.ResultState.RESULT_STATE_PENDING) {
        if (performance.now() >= deadline) throw new Error('GPU result timed out');
        const info = await client.getResult(new pb.ResultRef({ resultId: execution.resultId }));
        getResultCalls++;
        state = info.state;
        report = info.report;
        if (state === pb.ResultState.RESULT_STATE_PENDING) await new Promise(resolve => setTimeout(resolve, 1));
      }
      assert.equal(state, pb.ResultState.RESULT_STATE_READY);
      executeMs = performance.now() - started;
      async function read(semantic) {
        const before = performance.now();
        const tensor = (await client.readOutput(new pb.ReadOutputRequest({ resultId: execution.resultId,
          name: contract.outputs[semantic] }))).tensor;
        readOutputMs += performance.now() - before;
        readOutputCalls++;
        assert.ok([pb.DataType.DATA_TYPE_F32, pb.DataType.DATA_TYPE_I32].includes(tensor.dtype));
        const data = Uint8Array.from(tensor.inline);
        output[semantic] = { shape: tensor.shape.map(Number), dtype: tensor.dtype,
          data: tensor.dtype === pb.DataType.DATA_TYPE_F32 ? new Float32Array(data.buffer) : new Int32Array(data.buffer) };
      }
      if (role === 'digit') {
        await read('slot_logits');
      } else if (role === 'encoder') {
        for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) await read(`cross_${kind}_${layer}`);
        await read('memory_padding_mask');
        await read('selected_family_ids');
      } else {
        await read('logits');
        // An EOS or capped final step has no following cache consumer.
        if (needCache && argmax(output.logits.data) !== 2) {
          for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) await read(`present_${kind}_${layer}`);
          await read('present_padding_mask');
        }
      }
    } finally {
      const before = performance.now();
      await client.releaseResult(new pb.ResultRef({ resultId: execution.resultId }));
      releaseMs = performance.now() - before;
    }
    const elapsed = performance.now() - started;
    pendingReports.push([role, report]);
    pendingTimings.push([role, { execute_ms: executeMs, read_output_ms: readOutputMs,
      release_result_ms: releaseMs, runtime_execution_ms: execution.report.timings.executionTimeMs,
      execute_calls: 1, get_result_calls: getResultCalls, read_output_calls: readOutputCalls, release_result_calls: 1 }]);
    return [output, elapsed];
  }
  async function request(item) {
    if (options.model === 'digit') {
      const start = performance.now();
      const [output, elapsed] = await execute('digit', { image: item.pixels });
      const total = performance.now() - start;
      assert.ok(output.slot_logits.data.every(Number.isFinite));
      return [Array.from({ length: 16 }, (_, i) => argmax(output.slot_logits.data, i * 11, 11)),
        { request_ms: total, component_ms: elapsed, ...executionMetrics() }];
    }
    const count = item.question_ids.length;
    const feed = { image: item.pixels, question_ids: i32(item.question_ids, [1, count]),
      question_position_ids: i32(Array.from({ length: count }, (_, i) => i), [1, count]), family_ids: i32([item.family_id ?? -1], [1]) };
    const start = performance.now();
    const [encoded, encoderMs] = await execute('encoder', feed);
    const decFeed = decoderFeed(encoded), tokens = [1], decoderMs = [];
    let ttft;
    for (let position = 0; position < generation.max_new_tokens; position++) {
      const [output, elapsed] = await execute('decoder', decFeed, position + 1 < generation.max_new_tokens);
      decoderMs.push(elapsed);
      const token = argmax(output.logits.data);
      tokens.push(token);
      ttft ??= performance.now() - start;
      if (token === 2 || position + 1 === generation.max_new_tokens) break;
      for (let layer = 0; layer < 4; layer++) for (const kind of ['k', 'v']) decFeed[`past_${kind}_${layer}`] = output[`present_${kind}_${layer}`];
      decFeed.past_padding_mask = output.present_padding_mask;
      decFeed.decoder_input_ids = i32([token], [1, 1]);
      decFeed.position_ids = i32([position + 1], [1]);
    }
    const total = performance.now() - start;
    if (generation.require_eos) assert.equal(tokens.at(-1), 2);
    const sum = decoderMs.reduce((a, b) => a + b, 0);
    return [[tokens, encoded.selected_family_ids.data[0]], { request_ms: total, component_ms: encoderMs + sum,
      encoder_ms: encoderMs, decoder_ms: sum, ttft_ms: ttft, decoder_steps: decoderMs.length,
      decode_ms_per_token: decoderMs.slice(1).reduce((a, b) => a + b, 0) / Math.max(1, decoderMs.length - 1), ...executionMetrics() }];
  }
  const samples = [], signatures = {};
  for (let iteration = 0; iteration < warmup + repeat; iteration++) {
    const order = Array.from(cases.keys());
    if (iteration % 2) order.reverse();
    for (const index of order) {
      const [actual, timing] = await request(cases[index]);
      for (const [role, report] of pendingReports) routes[role].execution = route(report);
      pendingReports.length = 0;
      assert.deepEqual(actual, expected[String(cases[index].index)], 'output differs from frozen 2000-case reference');
      const sig = signature(actual);
      if (index in signatures) assert.equal(sig, signatures[index], 'output changed between benchmark requests');
      signatures[index] = sig;
      if (iteration >= warmup) samples.push({ ...timing, case_index: cases[index].index });
    }
  }
  const metrics = Object.fromEntries(Object.keys(samples[0]).filter(name => name.endsWith('_ms') || name === 'decode_ms_per_token')
    .map(name => [name, stats(samples.map(s => s[name]))]));
  const totalMs = samples.reduce((total, s) => total + s.request_ms, 0);
  const report = { schema: prefix ? 'volvoxai.receipt-proto-prefix-latency/v1' : 'volvoxai.receipt-proto-latency/v1', status: 'measured', host_quiet_verified: false, base_head: spec.base_head,
    model: options.model, backend: options.backend, profile: options.profile, variant: options.variant,
    cpu_threads: 1, warmup_per_case: warmup, repeat_per_case: repeat, case_indices: cases.map(c => c.index),
    order: 'alternating-forward-reverse', spec_sha256: hash(specBytes), harness_sha256: hash(readFileSync(fileURLToPath(import.meta.url))),
    runtime: globalThis.Deno?.version ?? { node: process.version }, driver: 'javascript-proto-c-wasm',
    artifacts: { [path.basename(bundle)]: hash(readFileSync(bundle)), [path.basename(wasm)]: hash(readFileSync(wasm)) },
    output_matches_2000_reference: !prefix, output_matches_benchmark_reference: true,
    repeated_outputs_identical: true, output_sha256: signatures,
    routes, metrics, samples, serial_requests_per_second: 1000 * samples.length / totalMs };
  report.timing_contract = {
    execute_ms: 'Execute through READY, including pending-result polling; excludes ReadOutput and ReleaseResult',
    runtime_execution_ms: 'Execute report interval; asynchronous WebGPU completion is measured by execute_ms; not GPU kernel-only time',
    read_output_ms: 'needed ReadOutput calls returning inline tensor data',
    component_ms: 'execution, needed output reads/materialization/selection and result release; excludes input construction',
    request_ms: 'complete request including input construction, token selection and cache rebinding',
    tensor_transport: 'inline tensors through the public WASM transport',
  };
  if (options.model === 'vqa') report.serial_generated_tokens_per_second = 1000 * samples.reduce((n, s) => n + s.decoder_steps, 0) / totalMs;
  if (adapterInfo) report.adapter = adapterInfo;
  if (prefix) Object.assign(report, { reference_kind: spec.reference_kind, generation,
    historical_reference: spec.historical_reference, stopped_at_eos: false, complete_answer: false });
  mkdirSync(path.dirname(path.resolve(options.out)), { recursive: true });
  writeFileSync(options.out, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ route: routeKey, measured_requests: samples.length, request: metrics.request_ms }));
} finally {
  await host.close();
  restoreAdapter?.();
}
