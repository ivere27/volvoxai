#!/usr/bin/env -S deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi
/** Replay immutable F64/I32 primitive and HF generation oracles over public WebGPU.
 *
 * deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi \
 *   tools/llm_baseline/verify_modern_webgpu.mjs \
 *   --out build/llm/portable-gpu/webgpu.json
 *
 * --bundle and --wasm default to this repository's package-version release.
 * Hardware is required by default. --adapter software-unit is explicitly
 * software correctness evidence. OWNED_BUFFER is an existing WebGPU refusal;
 * the suite checks that refusal before replaying INLINE on the same context.
 * --memory records partial requested allocator capacities, not performance or
 * physical device memory. The generated client settles pending C snapshots.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { readPackageVersion } from '../release_version.mjs';

const root = process.cwd();
const version = readPackageVersion();
const options = {
  bundle: fileURLToPath(new URL(`../../dist/${version}/volvoxai.js`, import.meta.url)),
  wasm: fileURLToPath(new URL(`../../dist/${version}/volvoxai.wasm`, import.meta.url)),
  corpus: path.join(root, 'build/llm/modern-primitives-public/wasm-cases-with-moe.json'),
  generationCorpus: path.join(root, 'build/llm/architecture-expansion/wasm-generation-cases.json'),
  adapter: 'hardware', memory: false, only: '', suite: 'all',
};
const argv = globalThis.Deno?.args ?? process.argv.slice(2);
for (let index = 0; index < argv.length; index++) {
  const flag = argv[index];
  if (flag === '--memory') { assert.equal(options.memory, false); options.memory = true; continue; }
  assert.ok(['--bundle', '--wasm', '--out', '--corpus', '--generation-corpus', '--adapter', '--only', '--suite'].includes(flag), `unknown argument ${flag}`);
  const value = argv[++index]; assert.ok(value && !value.startsWith('--'), `${flag} requires a value`);
  const key = flag === '--generation-corpus' ? 'generationCorpus' : flag.slice(2);
  options[key] = ['adapter', 'only', 'suite'].includes(key) ? value : path.resolve(value);
}
assert.ok(options.out, '--out is required');
assert.ok(['hardware', 'software-unit'].includes(options.adapter));
assert.ok(['all', 'execute', 'generate'].includes(options.suite));
const sha256 = value => createHash('sha256').update(value).digest('hex');
const bytes = value => new Uint8Array(Buffer.from(value, 'base64'));
const serializable = value => JSON.parse(JSON.stringify(value, (_, item) => typeof item === 'bigint' ? String(item) : item));
const selected = fixture => !options.only || fixture.name.includes(options.only);
const softwarePattern = /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|microsoft basic render|cpu)\b/i;
const results = [], diagnostics = [], retained = [], cleanupErrors = [];
let api, pb, host, client, profiling, runtimeId = 0n, traceId = 0n, current = null;
let status = 'FAIL', error = null, adapter = null, adapterWitness = null, trace = null;
const activeContexts = new Set(), activeCompiled = new Set(), activeModels = new Set();
const counts = { executeCalls: 0, generateCalls: 0, routeAttestations: 0, generatedIds: 0, ownedRefusals: 0, randomRefusals: 0, snapshotsRetained: 0 };
const inputs = fixture => fixture.inputs.map(value => new pb.Tensor({ name: value.name,
  shape: value.shape.map(BigInt), dtype: ['I32', 'int32'].includes(value.dtype) ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32,
  inline: bytes(value.data) }));
function checked(response, label) {
  const report = response.report ?? response;
  assert.equal(report.status, pb.NativeStatus.NATIVE_STATUS_OK, `${label}: ${report.message ?? ''}`);
  return response;
}
async function refused(call) {
  try { return await call; }
  catch (failure) {
    if (!failure.report && !failure.response?.report) throw failure;
    return failure.response ?? { report: failure.report, outputs: [] };
  }
}
function identity(info) {
  if (typeof info === 'string') {
    return Object.fromEntries(info.split(',').map(part => {
      const at = part.indexOf('='); return at < 0 ? ['description', part.trim()] : [part.slice(0, at).trim(), part.slice(at + 1).trim()];
    }).filter(([, value]) => value));
  }
  return info ? serializable(info) : {};
}
function attest(report, label, compilation = false) {
  assert.equal(report.backend, 'webgpu', `${label}: backend`);
  assert.equal(report.route?.attested, true, `${label}: attestation`);
  assert.equal(report.route?.selectedNodes, report.route?.activeNodes, `${label}: selected/active`);
  assert.ok(report.route?.activeNodes > 0, `${label}: no active nodes`);
  assert.equal(report.route?.fallbackNodes, 0, `${label}: fallback nodes`);
  assert.equal(report.route?.missingNodes, 0, `${label}: missing nodes`);
  assert.equal(report.fallback?.operatorFallbackUsed ?? false, false, `${label}: operator fallback`);
  assert.equal(report.compilation?.tierFallbackUsed ?? false, false, `${label}: tier fallback`);
  if (compilation) {
    assert.equal(report.compilation.policyMode, pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE);
    assert.equal(report.compilation.operatorFallback, pb.OperatorFallback.OPERATOR_FALLBACK_FORBID);
    assert.equal(report.compilation.candidates.length, 1);
    assert.equal(report.compilation.candidates[0].backend, 'webgpu');
    assert.equal(report.compilation.candidates[0].outcome, pb.CandidateOutcome.CANDIDATE_OUTCOME_SELECTED);
    const observed = identity(report.device), description = Object.values(observed).join(' ');
    assert.ok(description, `${label}: no device identity`);
    if (options.adapter === 'hardware') assert.doesNotMatch(description, softwarePattern, `${label}: software backend`);
    if (adapter) assert.deepEqual(observed, adapter, `${label}: adapter changed`);
    else adapter = observed;
  } else if (adapter) assert.deepEqual(identity(report.device), adapter, `${label}: device differs from compile`);
  counts.routeAttestations++;
}
function numeric(data, dtype) {
  assert.ok(data instanceof Uint8Array, 'completed INLINE payload required');
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const width = dtype === 'F64' ? 8 : 4;
  assert.equal(data.byteLength % width, 0);
  return Array.from({ length: data.byteLength / width }, (_, index) =>
    dtype === 'I32' ? view.getInt32(index * width, true) : dtype === 'F64' ? view.getFloat64(index * width, true) : view.getFloat32(index * width, true));
}
function compare(packet, fixture) {
  assert.deepEqual(packet.outputs.map(value => value.name).sort(), fixture.expected.map(value => value.name).sort());
  let maximumError = 0;
  for (const expected of fixture.expected) {
    const actual = packet.outputs.find(value => value.name === expected.name);
    assert.deepEqual(actual.shape, expected.shape.map(BigInt));
    assert.equal(actual.dtype, expected.dtype === 'I32' ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32);
    const observed = numeric(actual.inline, expected.dtype === 'I32' ? 'I32' : 'F32');
    const oracle = numeric(bytes(expected.data), expected.dtype);
    assert.equal(observed.length, oracle.length);
    for (let index = 0; index < observed.length; index++) {
      if (!Number.isFinite(oracle[index])) {
        assert.ok(Number.isNaN(oracle[index]) ? Number.isNaN(observed[index]) :
          Object.is(observed[index], oracle[index]), `${fixture.name}/${expected.name}/${index}: nonfinite contract`);
        continue;
      }
      const difference = Math.abs(observed[index] - oracle[index]);
      maximumError = Math.max(maximumError, difference);
      assert.ok(Number.isFinite(observed[index]) && (expected.dtype === 'I32' ? difference === 0 : difference <= 1e-5 + 1e-5 * Math.abs(oracle[index])),
        `${fixture.name}/${expected.name}/${index}: ${observed[index]} vs ${oracle[index]}`);
    }
  }
  return maximumError;
}
function retain(packet, label) {
  const snapshots = packet.outputs.map(value => ({ original: value.inline, expected: value.inline.slice() }));
  retained.push({ label, snapshots }); counts.snapshotsRetained++;
}
function contextState(info) {
  return serializable({ state: info.state, initialized: info.initialized, totalSteps: info.totalSteps, stopReason: info.stopReason,
    activeLengths: info.activeLengths, empty: info.empty, cacheGeneration: info.cacheGeneration, bindingsAvailable: info.bindingsAvailable });
}
async function contextInfo(contextId) {
  return checked(await client.getExecutionContextInfo(new pb.ExecutionContextRef({ contextId })), 'GetExecutionContextInfo');
}
async function releaseContext(contextId) {
  checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId })), 'ReleaseExecutionContext'); activeContexts.delete(contextId);
}
async function modelFor(fixture) {
  const model = checked(await client.loadModel(new pb.LoadModelRequest({ runtimeId,
    package: new pb.ModelPackage({ graphDocument: new TextEncoder().encode(JSON.stringify(fixture.graph)), weightShards: fixture.weight_shards.map(bytes) }) })), `${fixture.name}/LoadModel`);
  activeModels.add(model.modelId);
  const compiled = checked(await client.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
    packedArithmetic: fixture.packed_arithmetic ?? pb.PackedArithmetic.PACKED_ARITHMETIC_F32,
    policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['webgpu'],
      operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })), `${fixture.name}/CompileModel`);
  activeCompiled.add(compiled.compiledModelId); attest(compiled.report, `${fixture.name}/Compile`, true);
  return { modelId: model.modelId, compiledModelId: compiled.compiledModelId };
}
async function retireModel(handles) {
  checked(await client.releaseCompiledModel(new pb.CompiledModelRef(handles)), 'ReleaseCompiledModel'); activeCompiled.delete(handles.compiledModelId);
  checked(await client.releaseModel(new pb.ModelRef(handles)), 'ReleaseModel'); activeModels.delete(handles.modelId);
}
async function contextFor(compiledModelId, autoregressive) {
  const context = checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId,
    ...(autoregressive ? { autoregressive: new pb.AutoregressiveExecution(autoregressive) } : { graph: new pb.GraphExecution() }) })), 'CreateExecutionContext');
  activeContexts.add(context.contextId); return context.contextId;
}
const outputOptions = owned => new pb.OutputOptions({ storage: owned ? pb.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER : pb.OutputStorage.OUTPUT_STORAGE_INLINE });
function executeRequest(fixture, contextId, owned) {
  return new pb.ExecuteRequest({ contextId, inputs: new pb.InputBindings({ values: inputs(fixture) }),
    scope: new pb.ExecutionScope({ entrypoint: 'evaluate', full: new pb.Empty() }), outputs: outputOptions(owned) });
}
function generateRequest(fixture, contextId, budget, first, owned = false) {
  return new pb.GenerateRequest({ contextId, ...(first ? { initialInputs: new pb.GenerationInputs({ values: inputs(fixture) }) } : {}),
    budget: new pb.GenerationBudget({ maxSteps: budget }), outputs: outputOptions(owned) });
}
async function executeFixture(fixture) {
  current = { kind: 'Execute', name: fixture.name };
  const handles = await modelFor(fixture), contextId = await contextFor(handles.compiledModelId);
  const before = contextState(await contextInfo(contextId));
  const denied = await refused(client.execute(executeRequest(fixture, contextId, true)));
  assert.equal(denied.report.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED, `${fixture.name}/OWNED Execute refusal`);
  assert.equal(denied.outputs?.length ?? 0, 0);
  assert.deepEqual(contextState(await contextInfo(contextId)), before, `${fixture.name}/OWNED Execute mutated state`);
  counts.ownedRefusals++;
  const packet = checked(await client.execute(executeRequest(fixture, contextId, false)), `${fixture.name}/Execute`);
  counts.executeCalls++; attest(packet.report, `${fixture.name}/Execute`);
  const maximumError = compare(packet, fixture); retain(packet, fixture.name);
  await releaseContext(contextId); await retireModel(handles);
  results.push({ kind: 'Execute', name: fixture.name, maximumError, ownedRefusal: true, pass: true });
  console.log(`Execute PASS ${fixture.name} error=${maximumError}`);
}
async function generateVariant(fixture, compiledModelId, kind, configuration, budgets, expected, stopReason, testOwned = false) {
  current = { kind: `Generate/${kind}`, name: fixture.name };
  const contextId = await contextFor(compiledModelId, { sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY,
    maxSteps: BigInt(fixture.expected_ids.length), ...configuration });
  if (testOwned) {
    const before = contextState(await contextInfo(contextId));
    const denied = await refused(client.generate(generateRequest(fixture, contextId, budgets[0], true, true)));
    assert.equal(denied.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT, `${fixture.name}/OWNED Generate refusal`);
    assert.equal(denied.outputs?.length ?? 0, 0);
    assert.deepEqual(contextState(await contextInfo(contextId)), before, `${fixture.name}/OWNED Generate mutated state`);
    counts.ownedRefusals++;
  }
  const generated = [];
  let last;
  for (const budget of budgets) {
    last = checked(await client.generate(generateRequest(fixture, contextId, budget, generated.length === 0)), `${fixture.name}/${kind}/Generate`);
    counts.generateCalls++; attest(last.report, `${fixture.name}/${kind}/Generate`);
    const ids = last.outputs.find(value => value.name === 'generated_ids');
    assert.equal(last.outputs.length, 1); assert.ok(ids, `${fixture.name}: generated IDs`);
    assert.equal(ids.dtype, pb.DataType.DATA_TYPE_I32);
    const chunk = numeric(ids.inline, 'I32'); assert.ok(chunk.length > 0 && chunk.length <= budget);
    assert.deepEqual(ids.shape, [1n, BigInt(chunk.length)]);
    generated.push(...chunk); counts.generatedIds += chunk.length;
    assert.equal(last.progress.steps, BigInt(chunk.length)); assert.equal(last.progress.totalSteps, BigInt(generated.length));
    retain(last, `${fixture.name}/${kind}/${generated.length}`);
    if (last.progress.state === pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED) break;
  }
  assert.deepEqual(generated, expected, `${fixture.name}/${kind}: HF greedy feedback`);
  assert.equal(last.progress.stopReason, stopReason, `${fixture.name}/${kind}: stop reason`);
  const info = await contextInfo(contextId);
  assert.equal(info.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
  assert.equal(info.totalSteps, BigInt(expected.length));
  await releaseContext(contextId);
  results.push({ kind: `Generate/${kind}`, name: fixture.name, generatedIds: generated, pass: true });
  console.log(`Generate PASS ${fixture.name}/${kind} IDs=${JSON.stringify(generated)}`);
}
async function generationFixture(fixture) {
  current = { kind: 'Generate/Compile', name: fixture.name };
  const handles = await modelFor(fixture);
  const random = await refused(client.createExecutionContext(new pb.CreateExecutionContextRequest({ compiledModelId: handles.compiledModelId,
    autoregressive: new pb.AutoregressiveExecution({ sampling: pb.SamplingMode.SAMPLING_MODE_RANDOM, seed: 42n, maxSteps: 3n }) })));
  assert.equal(random.report.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED, `${fixture.name}/RANDOM context refusal`);
  assert.equal(random.contextId ?? 0n, 0n); counts.randomRefusals++;
  await generateVariant(fixture, handles.compiledModelId, 'greedy', {}, fixture.budgets, fixture.expected_ids,
    pb.ExecutionStopReason.EXECUTION_STOP_REASON_STEP_LIMIT, true);
  await generateVariant(fixture, handles.compiledModelId, 'eos', { eosIds: [fixture.expected_ids[0]], stopIds: [fixture.expected_ids[0]] },
    [fixture.expected_ids.length], fixture.expected_ids.slice(0, 1), pb.ExecutionStopReason.EXECUTION_STOP_REASON_EOS);
  await generateVariant(fixture, handles.compiledModelId, 'stop-sequence', { stopSequences: [new pb.IdStopSequence({ ids: fixture.expected_ids.slice(0, 2) })] },
    [1, 2], fixture.expected_ids.slice(0, 2), pb.ExecutionStopReason.EXECUTION_STOP_REASON_ID_STOP);
  await retireModel(handles);
}

let sourcePins = {}, corpusCounts = {};
try {
  assert.ok(globalThis.navigator?.gpu, 'navigator.gpu unavailable; Deno --unstable-webgpu is required');
  globalThis.window ??= globalThis; globalThis.self ??= globalThis;
  globalThis.document ??= { createElement: () => ({}), querySelector: () => null };
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = async (source, init) => {
    const href = source instanceof URL ? source.href : typeof source === 'string' ? source : source?.url;
    if (href?.startsWith('file://')) return new Response(await readFile(fileURLToPath(href)));
    return originalFetch(source, init);
  };
  const [corpusBytes, generationBytes, bundleBytes, wasmBytes] = await Promise.all([
    readFile(options.corpus), readFile(options.generationCorpus), readFile(options.bundle), readFile(options.wasm),
  ]);
  sourcePins = { corpusSha256: sha256(corpusBytes), generationCorpusSha256: sha256(generationBytes), bundleSha256: sha256(bundleBytes), wasmSha256: sha256(wasmBytes) };
  const cases = JSON.parse(corpusBytes).cases, generationCases = JSON.parse(generationBytes).cases;
  assert.ok(Array.isArray(cases) && cases.length > 0, 'nonempty primitive corpus required');
  assert.ok(Array.isArray(generationCases) && generationCases.length > 0, 'nonempty generation corpus required');
  for (const corpus of [cases, generationCases]) {
    assert.equal(new Set(corpus.map(fixture => fixture.name)).size, corpus.length, 'fixture names must be unique');
  }
  corpusCounts = { primitives: cases.length, generationFamilies: generationCases.length };
  const observedAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  assert.ok(observedAdapter, 'WebGPU adapter unavailable');
  const witnessInfo = observedAdapter.info ?? await observedAdapter.requestAdapterInfo?.() ?? {};
  adapterWitness = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'deviceType', 'driver'].map(key => [key, witnessInfo[key]]).filter(([, value]) => typeof value === 'string' && value));
  const witnessDescription = Object.values(adapterWitness).join(' ');
  assert.ok(witnessDescription, 'WebGPU adapter identity unavailable');
  if (options.adapter === 'hardware') assert.doesNotMatch(witnessDescription, softwarePattern, 'software adapter refused');
  api = await import(pathToFileURL(options.bundle).href); pb = api.pb;
  assert.equal(typeof api.FullEngineHost, 'function'); assert.equal(typeof api.VxInferenceServiceClient, 'function');
  host = new api.FullEngineHost({ wasmUrl: pathToFileURL(options.wasm), onDiagnostic: message => { diagnostics.push(message); console.error(message); } });
  client = new api.VxInferenceServiceClient(host);
  runtimeId = checked(await client.createRuntime(new pb.CreateRuntimeRequest()), 'CreateRuntime').runtimeId;
  if (options.memory) {
    profiling = new api.VxProfilingServiceClient(host);
    const started = checked(await profiling.startTrace(new pb.StartTraceRequest({ runtimeId,
      options: new pb.TraceOptions({ memory: true, executionPlans: true, deviceTiming: false, capacityBytes: 4n * 1024n * 1024n }) })), 'StartTrace');
    traceId = started.traceId;
  }
  if (options.suite !== 'generate') for (const fixture of cases.filter(selected)) await executeFixture(fixture);
  if (options.suite !== 'execute') for (const fixture of generationCases.filter(selected)) await generationFixture(fixture);
  checked(await client.releaseRuntime(new pb.RuntimeRef({ runtimeId })), 'ReleaseRuntime'); runtimeId = 0n;
  for (const entry of retained) for (const snapshot of entry.snapshots) assert.deepEqual(snapshot.original, snapshot.expected, `${entry.label}: INLINE snapshot changed after owner retirement`);
  assert.ok(results.length > 0, 'selected suite is empty');
  status = 'PASS'; current = null;
} catch (failure) {
  error = { message: failure.message, stack: failure.stack, report: serializable(failure.report ?? failure.response?.report ?? null) };
  throw failure;
} finally {
  if (client && pb) {
    const cleanup = async (call, label) => { try { checked(await call(), label); } catch (failure) { cleanupErrors.push(`${label}: ${failure.message}`); } };
    for (const contextId of activeContexts) await cleanup(() => client.releaseExecutionContext(new pb.ExecutionContextRef({ contextId })), 'cleanup context');
    for (const compiledModelId of activeCompiled) await cleanup(() => client.releaseCompiledModel(new pb.CompiledModelRef({ compiledModelId })), 'cleanup compiled');
    for (const modelId of activeModels) await cleanup(() => client.releaseModel(new pb.ModelRef({ modelId })), 'cleanup model');
    if (runtimeId) await cleanup(() => client.releaseRuntime(new pb.RuntimeRef({ runtimeId })), 'cleanup runtime');
  }
  if (profiling && traceId) {
    try {
      let info = checked(await profiling.stopTrace(new pb.TraceRef({ traceId })), 'StopTrace');
      const deadline = performance.now() + 30000;
      while (info.state !== pb.TraceState.TRACE_STATE_READY && performance.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 10)); info = checked(await profiling.getTrace(new pb.TraceRef({ traceId })), 'GetTrace');
      }
      assert.equal(info.state, pb.TraceState.TRACE_STATE_READY, 'memory trace did not drain');
      trace = serializable(info); checked(await profiling.releaseTrace(new pb.TraceRef({ traceId })), 'ReleaseTrace');
    } catch (failure) { cleanupErrors.push(`trace: ${failure.message}`); }
  }
  if (host) { try { await host.close(); } catch (failure) { cleanupErrors.push(`host close: ${failure.message}`); } }
  if (cleanupErrors.length) status = 'FAIL';
  const receipt = { status, scope: options.only || options.suite !== 'all' ? 'selected public API WebGPU subset' : 'complete supplied public API WebGPU corpora',
    options, sourcePins, corpusCounts, adapter, adapterWitness, qualification: options.adapter,
    oracle: 'immutable independent NumPy F64/I32 primitive expectations and official HF greedy IDs', tolerances: { absolute: 1e-5, relative: 1e-5, integer: 0 },
    counts, results, retainedInlineAfterAllOwnerRetirement: status === 'PASS', ownedContract: 'Execute BACKEND_UNSUPPORTED; Generate INVALID_ARGUMENT, before state mutation',
    memory: trace ? { scope: 'partial requested allocator capacities; no speed or physical memory claim', trace } : null,
    diagnostics, current, error, cleanupErrors };
  await mkdir(path.dirname(options.out), { recursive: true }); await writeFile(options.out, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ status, results: results.length, counts, out: options.out }));
  if (cleanupErrors.length && !error) throw new Error(cleanupErrors.join('\n'));
}
