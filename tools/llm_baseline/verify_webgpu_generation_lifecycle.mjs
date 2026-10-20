#!/usr/bin/env -S deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi
/** Actual-device whole-prefix Generate lifetime regression.
 *
 * deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
 *   tools/llm_baseline/verify_webgpu_generation_lifecycle.mjs \
 *   --bundle dist/0.7.0/volvoxai.js --wasm dist/0.7.0/volvoxai.wasm \
 *   --out build/llm/portable-gpu/webgpu-generation-lifecycle.json
 *
 * The immutable DeepSeek V3 fixture supplies independent HF IDs. All compute,
 * copies and mappings use the real WebGPU device. One real mapAsync completion
 * can be held to make cancellation/retirement boundaries deterministic. This
 * measures correctness and explicit GPUBuffer retirement, not performance.
 * --only <check-name> isolates one check in a fresh process, allowing device
 * recreation failures to be distinguished from generation lifetime failures.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const options = {
  generationCorpus: path.resolve('build/llm/architecture-expansion/wasm-generation-cases.json'),
  adapter: 'hardware', only: '',
};
const argv = globalThis.Deno?.args ?? process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const flag = argv[i];
  assert.ok(['--bundle', '--wasm', '--out', '--generation-corpus', '--adapter', '--only'].includes(flag), `unknown argument ${flag}`);
  const value = argv[++i]; assert.ok(value && !value.startsWith('--'), `${flag} requires a value`);
  const key = flag === '--generation-corpus' ? 'generationCorpus' : flag.slice(2);
  options[key] = ['adapter', 'only'].includes(key) ? value : path.resolve(value);
}
for (const key of ['bundle', 'wasm', 'out']) assert.ok(options[key], `--${key} is required`);
assert.ok(['hardware', 'software-unit'].includes(options.adapter));
const checkNames = ['expired-before-commit', 'cancel-pending', 'deadline-after-commit',
  'release-all-handles-pending', 'abort-pending-transport', 'close-pending-owner'];
assert.ok(!options.only || checkNames.includes(options.only), 'unknown lifecycle check');
const selected = name => !options.only || options.only === name;
const hash = value => createHash('sha256').update(value).digest('hex');
const decode = value => new Uint8Array(Buffer.from(value, 'base64'));
const plain = value => JSON.parse(JSON.stringify(value, (_, item) => typeof item === 'bigint' ? String(item) : item));
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const bounded = (promise, label, ms = 30000) => {
  let timer;
  return Promise.race([promise, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out`)), ms);
  })]).finally(() => clearTimeout(timer));
};
const software = /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|microsoft basic render|cpu)\b/i;
let api, pb, fixture, pins = {}, adapterIdentity = null, phase = 'initialize', failure = null;
const diagnostics = [], checks = [], witnesses = [], sessions = new Set();
const held = new Set();
let armed = null;
const totals = { devices: 0, buffersCreated: 0, buffersDestroyed: 0, scalarMaps: 0, gates: 0 };
const live = new Set(), staging = new Set();
const gpu = globalThis.navigator?.gpu;
assert.ok(gpu, 'navigator.gpu unavailable; Deno --unstable-webgpu is required');
const originalRequestAdapter = gpu.requestAdapter;

function holdNextScalar() {
  assert.equal(armed, null, 'a scalar gate is already armed');
  let start, open;
  const gate = { started: new Promise(resolve => { start = resolve; }),
    released: new Promise(resolve => { open = resolve; }), used: false, start,
    release() { open(); held.delete(gate); if (armed === gate) armed = null; } };
  armed = gate; held.add(gate); totals.gates++;
  return gate;
}

// Preserve original GPU objects and all numeric work. Override methods on the
// actual instances so WebGPU receives its own buffers without proxy branding.
gpu.requestAdapter = async function (...args) {
  const adapter = await originalRequestAdapter.apply(gpu, args);
  if (!adapter) return adapter;
  const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
  const identity = Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(key => [key, info[key]])
    .filter(([, value]) => typeof value === 'string' && value));
  const description = Object.values(identity).join(' ');
  assert.ok(description, 'adapter identity required');
  if (options.adapter === 'hardware') assert.doesNotMatch(description, software, 'software adapter refused');
  if (adapterIdentity) assert.deepEqual(identity, adapterIdentity, 'adapter changed');
  else adapterIdentity = identity;
  const requestDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = async (...deviceArgs) => {
    const device = await requestDevice(...deviceArgs); totals.devices++;
    const createBuffer = device.createBuffer.bind(device);
    device.createBuffer = descriptor => {
      const buffer = createBuffer(descriptor); totals.buffersCreated++; live.add(buffer);
      const readback = descriptor.label === 'vx.readback';
      if (readback) staging.add(buffer);
      const destroy = buffer.destroy.bind(buffer);
      buffer.destroy = () => {
        if (live.delete(buffer)) totals.buffersDestroyed++;
        staging.delete(buffer); destroy();
      };
      if (readback) {
        const mapAsync = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = (...mapArgs) => {
          assert.equal(buffer.size, 4, 'Generate maps exactly one categorical I32 ID');
          totals.scalarMaps++;
          const realMapping = mapAsync(...mapArgs);
          const gate = armed;
          if (!gate) return realMapping;
          armed = null; gate.used = true; gate.start();
          // Begin the actual map before delaying its observable completion.
          return Promise.all([realMapping, gate.released]).then(() => undefined);
        };
      }
      return buffer;
    };
    return device;
  };
  return adapter;
};

function checked(response, label) {
  const report = response.report ?? response;
  assert.equal(report.status, pb.NativeStatus.NATIVE_STATUS_OK, `${label}: ${report.message ?? ''}`);
  return response;
}
async function response(call) {
  try { return await call; }
  catch (error) { if (!error.report && !error.response?.report) throw error; return error.response ?? { report: error.report, outputs: [] }; }
}
function attest(report, label) {
  assert.equal(report.backend, 'webgpu', `${label}: backend`);
  assert.equal(report.route?.attested, true, `${label}: route attestation`);
  assert.ok(report.route.activeNodes > 0, `${label}: active nodes`);
  assert.equal(report.route.selectedNodes, report.route.activeNodes, `${label}: active route coverage`);
  assert.equal(report.route.fallbackNodes, 0, `${label}: fallback nodes`);
  assert.equal(report.route.missingNodes, 0, `${label}: missing nodes`);
  assert.equal(report.fallback?.operatorFallbackUsed ?? false, false, `${label}: fallback`);
  assert.equal(report.compilation?.tierFallbackUsed ?? false, false, `${label}: tier fallback`);
}
function ids(packet) {
  assert.equal(packet.outputs.length, 1);
  const output = packet.outputs[0];
  assert.equal(output.name, 'generated_ids'); assert.equal(output.dtype, pb.DataType.DATA_TYPE_I32);
  assert.ok(output.inline instanceof Uint8Array);
  const view = new DataView(output.inline.buffer, output.inline.byteOffset, output.inline.byteLength);
  const values = Array.from({ length: output.inline.byteLength / 4 }, (_, i) => view.getInt32(i * 4, true));
  assert.deepEqual(output.shape, [1n, BigInt(values.length)]);
  witnesses.push({ original: output.inline, expected: output.inline.slice() });
  return values;
}
function request(contextId, first = true, deadline) {
  return new pb.GenerateRequest({ contextId,
    ...(first ? { initialInputs: new pb.GenerationInputs({ values: fixture.inputs.map(value => new pb.Tensor({ name: value.name,
      shape: value.shape.map(BigInt), dtype: ['I32', 'int32'].includes(value.dtype) ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32,
      inline: decode(value.data) })) }) } : {}),
    budget: new pb.GenerationBudget({ maxSteps: 3, ...(deadline === undefined ? {} : { deadlineMonotonicNs: deadline }) }),
    outputs: new pb.OutputOptions({ storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE }) });
}
async function openSession() {
  const host = new api.FullEngineHost({ wasmUrl: pathToFileURL(options.wasm),
    onDiagnostic: message => { diagnostics.push(message); console.error(message); } });
  const client = new api.VxInferenceServiceClient(host), platform = new api.VxPlatformServiceClient(host);
  const session = { host, client, platform, contexts: new Set(), runtimeId: 0n, modelId: 0n, compiledModelId: 0n, closed: false };
  sessions.add(session);
  session.runtimeId = checked(await client.createRuntime(new pb.CreateRuntimeRequest()), 'CreateRuntime').runtimeId;
  session.modelId = checked(await client.loadModel(new pb.LoadModelRequest({ runtimeId: session.runtimeId,
    package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)),
      weightShards: fixture.weight_shards.map(decode) }) })), 'LoadModel').modelId;
  const compiled = checked(await client.compileModel(new pb.CompileModelRequest({ modelId: session.modelId,
    policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['webgpu'],
      operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })), 'CompileModel');
  attest(compiled.report, 'CompileModel');
  assert.equal(compiled.report.compilation.policyMode, pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE);
  assert.equal(compiled.report.compilation.operatorFallback, pb.OperatorFallback.OPERATOR_FALLBACK_FORBID);
  session.compiledModelId = compiled.compiledModelId;
  return session;
}
async function context(session) {
  const created = checked(await session.client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: session.compiledModelId,
    autoregressive: new pb.AutoregressiveExecution({ sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY, maxSteps: 3n }) })), 'CreateExecutionContext');
  session.contexts.add(created.contextId); return created.contextId;
}
const info = (s, contextId) => s.client.getExecutionContextInfo(new pb.ExecutionContextRef({ contextId }));
async function releaseContext(s, contextId) {
  checked(await s.client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId })), 'ReleaseExecutionContext'); s.contexts.delete(contextId);
}
async function retireHandles(s) {
  for (const contextId of [...s.contexts]) await releaseContext(s, contextId);
  for (const [key, method, type] of [['compiledModelId', 'releaseCompiledModel', 'CompiledModelRef'], ['modelId', 'releaseModel', 'ModelRef'], ['runtimeId', 'releaseRuntime', 'RuntimeRef']]) {
    if (s[key]) { checked(await s.client[method](new pb[type]({ [key]: s[key] })), method); s[key] = 0n; }
  }
}
async function drainStaging(label) {
  const deadline = performance.now() + 30000;
  while (staging.size && performance.now() < deadline) await delay(5);
  assert.equal(staging.size, 0, `${label}: retained readback ticket storage`);
}
async function closeSession(s) {
  if (s.closed) return;
  await retireHandles(s); await bounded(s.host.close(), 'host.close'); s.closed = true; sessions.delete(s);
  await drainStaging('host.close'); assert.equal(live.size, 0, 'host.close destroys every observed GPUBuffer');
}
async function begin(s, contextId, deadline, callOptions) {
  const gate = holdNextScalar();
  const pending = { settled: false, promise: s.client.generate(request(contextId, true, deadline), callOptions) };
  pending.observed = pending.promise.then(value => { pending.settled = true; return { value }; }, error => { pending.settled = true; return { error }; });
  await bounded(gate.started, `${phase}: first scalar map`);
  assert.equal(gate.used, true); assert.equal(pending.settled, false, `${phase}: call must be pending`);
  const state = checked(await info(s, contextId), 'pending context info');
  assert.equal(state.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING);
  assert.equal(state.totalSteps, 0n);
  return { gate, pending, maps: totals.scalarMaps };
}
async function finish(started) {
  started.gate.release();
  const result = await bounded(started.pending.observed, `${phase}: Generate completion`);
  if (result.error) return response(Promise.reject(result.error));
  return result.value;
}
function pass(name, details = {}) { checks.push({ name, ...details, pass: true }); console.log(`PASS ${name}`); }

try {
  globalThis.window ??= globalThis; globalThis.self ??= globalThis;
  globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (source, init) => {
    const href = source instanceof URL ? source.href : typeof source === 'string' ? source : source?.url;
    return href?.startsWith('file://') ? new Response(await readFile(fileURLToPath(href))) : originalFetch(source, init);
  };
  const [corpusBytes, bundleBytes, wasmBytes] = await Promise.all([readFile(options.generationCorpus), readFile(options.bundle), readFile(options.wasm)]);
  pins = { generationCorpusSha256: hash(corpusBytes), bundleSha256: hash(bundleBytes), wasmSha256: hash(wasmBytes) };
  fixture = JSON.parse(corpusBytes).cases.find(value => value.name === 'deepseek_v3');
  assert.ok(fixture, 'immutable DeepSeek V3 fixture required'); assert.deepEqual(fixture.expected_ids, [30, 10, 8]);
  api = await import(pathToFileURL(options.bundle).href); pb = api.pb;
  if (checkNames.slice(0, 4).some(selected)) {
  const s = await openSession();

  if (selected('expired-before-commit')) {
  phase = 'expired-before-commit';
  const expiredContext = await context(s), before = plain(checked(await info(s, expiredContext), 'initial context info'));
  const mapsBefore = totals.scalarMaps;
  const expired = await response(s.client.generate(request(expiredContext, true, 1n)));
  assert.equal(expired.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED); assert.equal(expired.outputs?.length ?? 0, 0);
  assert.deepEqual(plain(checked(await info(s, expiredContext), 'refused context info')), before);
  assert.equal(totals.scalarMaps, mapsBefore, 'expired admission dispatches no GPU work');
  const baseline = checked(await s.client.generate(request(expiredContext)), 'reuse after expired admission');
  attest(baseline.report, 'baseline'); assert.deepEqual(ids(baseline), fixture.expected_ids);
  await drainStaging('baseline'); await releaseContext(s, expiredContext);
  pass(phase, { unchangedContext: true, reusedIds: fixture.expected_ids });
  }

  if (selected('cancel-pending')) {
  phase = 'cancel-pending';
  const cancelledContext = await context(s), cancelled = await begin(s, cancelledContext);
  const busy = await response(s.client.generate(request(cancelledContext, false)));
  assert.equal(busy.report.status, pb.NativeStatus.NATIVE_STATUS_BUSY);
  for (let repeat = 0; repeat < 2; repeat++) checked(await s.client.cancelExecutionContext(new pb.ExecutionContextRef({ contextId: cancelledContext })), 'CancelExecutionContext');
  const cancelledPacket = checked(await finish(cancelled), 'cancelled Generate');
  attest(cancelledPacket.report, phase); assert.deepEqual(ids(cancelledPacket), fixture.expected_ids.slice(0, 1));
  assert.equal(cancelledPacket.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_CANCELLED);
  assert.equal(cancelledPacket.progress.stopReason, pb.ExecutionStopReason.EXECUTION_STOP_REASON_CANCELLED);
  assert.equal(cancelledPacket.progress.totalSteps, 1n);
  assert.equal(totals.scalarMaps, cancelled.maps, 'cancel submits no next prefix');
  const terminal = plain(checked(await info(s, cancelledContext), 'cancelled context info'));
  const retry = await response(s.client.generate(request(cancelledContext, false)));
  assert.equal(retry.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  assert.deepEqual(plain(checked(await info(s, cancelledContext), 'cancelled context remains terminal')), terminal);
  await drainStaging(phase); await releaseContext(s, cancelledContext);
  pass(phase, { generatedIds: fixture.expected_ids.slice(0, 1), noFurtherPrefix: true });
  }

  if (selected('deadline-after-commit')) {
  phase = 'deadline-after-commit';
  const deadlineContext = await context(s);
  const deadline = (await s.platform.getMonotonicTime(new pb.Empty())).nanoseconds + 2_000_000_000n;
  const late = await begin(s, deadlineContext, deadline);
  while ((await s.platform.getMonotonicTime(new pb.Empty())).nanoseconds <= deadline) await delay(20);
  const latePacket = await finish(late);
  assert.equal(latePacket.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
  attest(latePacket.report, phase);
  assert.equal(latePacket.outputs?.length ?? 0, 0, 'committed deadline transfers no IDs');
  assert.equal(latePacket.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
  assert.equal(totals.scalarMaps, late.maps, 'deadline submits no next prefix');
  assert.equal(checked(await info(s, deadlineContext), 'failed deadline context').state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
  assert.equal((await response(s.client.generate(request(deadlineContext, false)))).report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  await drainStaging(phase); await releaseContext(s, deadlineContext);
  pass(phase, { noOutputHandoff: true, noFurtherPrefix: true });
  }

  if (selected('release-all-handles-pending')) {
  phase = 'release-all-handles-pending';
  const releasedContext = await context(s), retained = await begin(s, releasedContext);
  await retireHandles(s);
  assert.equal((await response(info(s, releasedContext))).report.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  assert.equal(retained.pending.settled, false, 'handle retirement preserves the in-flight result owner');
  const retainedPacket = checked(await finish(retained), 'retained Generate');
  attest(retainedPacket.report, phase); assert.deepEqual(ids(retainedPacket), fixture.expected_ids);
  assert.equal(retainedPacket.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
  await drainStaging(phase); await closeSession(s);
  pass(phase, { generatedIds: fixture.expected_ids, independentInlineLifetime: true });
  }
  await closeSession(s);
  }

  if (selected('abort-pending-transport')) {
  phase = 'abort-pending-transport';
  const a = await openSession(), abortedContext = await context(a), abort = new AbortController();
  const aborted = await begin(a, abortedContext, undefined, { signal: abort.signal }); abort.abort();
  const abortedResult = await bounded(aborted.pending.observed, 'aborted transport');
  assert.ok(abortedResult.error, 'AbortSignal rejects the pending transport');
  /* Transport cancellation abandons delivery while the committed chunk drains.
   * Logical cancellation is the CancelExecutionContext check above. Keep its
   * handle live here so the finished engine state can be observed publicly. */
  aborted.gate.release();
  let abortedInfo;
  const abortDeadline = performance.now() + 30000;
  do {
    abortedInfo = checked(await info(a, abortedContext), 'abandoned Generate context');
    if (abortedInfo.state !== pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING) break;
    await delay(5);
  } while (performance.now() < abortDeadline);
  assert.equal(abortedInfo.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
  assert.equal(abortedInfo.totalSteps, 3n, 'abandoned transport drains its committed chunk');
  await drainStaging(phase);
  assert.equal(totals.scalarMaps, aborted.maps + 2, 'only the admitted chunk continues after delivery is abandoned');
  const survivorContext = await context(a);
  const survivor = checked(await a.client.generate(request(survivorContext)), 'independent Generate after abandoned delivery');
  attest(survivor.report, phase); assert.deepEqual(ids(survivor), fixture.expected_ids);
  await closeSession(a);
  pass(phase, { deliveryAbandoned: true, committedChunkDrained: true, retiredReadbackStorage: true, independentContextReusable: true });
  }

  if (selected('close-pending-owner')) {
  phase = 'close-pending-owner';
  const c = await openSession(), closingContext = await context(c), closing = await begin(c, closingContext);
  let closeDone = false;
  const closed = c.host.close().then(() => { closeDone = true; });
  await delay(0); assert.equal(closeDone, false, 'host.close waits for the retained actual mapping');
  const closingResult = await bounded(closing.pending.observed, 'closed pending transport');
  assert.ok(closingResult.error, 'owner close rejects the pending transport');
  closing.gate.release(); await bounded(closed, 'pending host.close'); c.closed = true; sessions.delete(c);
  await drainStaging(phase); assert.equal(live.size, 0, 'owner close destroys all observed GPUBuffer storage');
  assert.equal(totals.scalarMaps, closing.maps + 2, 'owner close drains only its admitted chunk before retirement');
  pass(phase, { deliveryAbandoned: true, committedChunkDrained: true, allObservedGpuBuffersDestroyed: true });
  }
  assert.equal(checks.length, options.only ? 1 : checkNames.length, 'all selected lifecycle checks executed');
  for (const witness of witnesses) assert.deepEqual(witness.original, witness.expected, 'INLINE IDs changed after owner retirement');
  phase = 'complete';
} catch (error) {
  failure = { message: error.message, stack: error.stack, report: plain(error.report ?? error.response?.report ?? null) };
} finally {
  for (const gate of held) gate.release();
  for (const session of sessions) {
    try { await closeSession(session); } catch (error) { diagnostics.push(`cleanup: ${error.message}`); failure ??= { message: error.message }; }
  }
  gpu.requestAdapter = originalRequestAdapter;
  const receipt = { status: failure ? 'FAIL' : 'PASS', scope: 'actual WebGPU public Generate lifecycle, immutable DeepSeek V3 whole-prefix graph',
    selectedChecks: options.only ? [options.only] : checkNames,
    options, pins, adapter: adapterIdentity, qualification: options.adapter,
    oracle: 'independent immutable HF categorical IDs [30,10,8]',
    completionControl: 'one real mapAsync is invoked normally and its completion promise is gated; numerical work and bytes are never replaced',
    bufferObservation: 'explicit creation/destruction of actual GPUBuffer objects; readback staging lifetime witnesses ticket retirement',
    checks, totals, liveBuffers: live.size, liveReadbackBuffers: staging.size, independentInlineLifetime: !failure,
    diagnostics, phase, error: failure };
  await mkdir(path.dirname(options.out), { recursive: true }); await writeFile(options.out, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ status: receipt.status, checks: checks.length, totals, out: options.out }));
}
if (failure) throw new Error(failure.message);
