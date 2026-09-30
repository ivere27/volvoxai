import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {checkTraceMemory} from '../tools/trace_memory_evidence.mjs';
import { fixture, p, ok, tensors, safetensors } from '../tools/proto_fixture.mjs';
import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { VxProfilingServiceClient, VxPlatformServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import { MemoryEvidenceValidatingTransport } from '../runtime/typescript/MemoryEvidenceValidatingTransport.js';
import { validateMemoryBounds } from '../runtime/typescript/MemoryEvidenceValidation.js';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.lite.wasm`, import.meta.url).pathname;
const graph = {
  format: 'volvox-graph/v1', dimensions: {},
  inputs: {x: {dtype: 'float32', shape: [4]}},
  nodes: [
    {id: 'relu', opType: 'ReLU', inputs: {input: 'x'},
      outputs: {out: {tensor: 'first', dtype: 'float32', shape: [4]}}, params: {}},
    {id: 'sum', opType: 'Add', inputs: {a: 'first', b: 'first'},
      outputs: {out: {tensor: 'out"한글', dtype: 'float32', shape: [4]}}, params: {}},
  ], outputs: ['out"한글'],
};
async function setup(t) {
  const f = await fixture({wasmUrl});
  t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(new MemoryEvidenceValidatingTransport(reportTransport(f.host)));
  const model = await f.load(graph);
  const compiled = await f.compile(model);
  validateMemoryBounds(compiled.memoryBounds);
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
  return {...f, profiling, model, compiled, context};
}
async function run(f) {
  const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: f.context.contextId,
    inputs: tensors({x: {data: Float32Array.of(-1, 2, -3, 4), shape: [4]}})})));
  const output = await f.read(result, 'out"한글');
  assert.deepEqual([...output], [0, 4, 0, 8]);
  assert.equal(result.metrics.outputBytes, 16n);
  const info = ok(await f.inference.getResult(new p.ResultRef(result)));
  assert.deepEqual(info.metrics, result.metrics);
  await f.inference.releaseResult(new p.ResultRef(result));
  return result;
}
const start = (profiling, runtimeId, options = {}) =>
  profiling.startTrace(new p.StartTraceRequest({runtimeId, options: new p.TraceOptions(options)}));
/** Reads every AIP-158 page; repeating a page returns the same records. */
async function listEvents(profiling, traceId, pageSize = 1) {
  const events = []; let pageToken = '';
  do {
    const request = new p.ListTraceEventsRequest({traceId, pageSize, pageToken});
    const page = ok(await profiling.listTraceEvents(request));
    assert.deepEqual(ok(await profiling.listTraceEvents(request)).events, page.events);
    events.push(...page.events); pageToken = page.nextPageToken;
  } while (pageToken);
  return events;
}
async function listResources(profiling, traceId) {
  return ok(await profiling.listTraceResourceSnapshots(new p.ListTraceResourceSnapshotsRequest({traceId, pageSize: 4096}))).resourceSnapshots;
}
async function listPlans(profiling, traceId, count) {
  const plans = [];
  for (let planId = 1n; planId <= count; planId++)
    plans.push(ok(await profiling.getTracePlan(new p.GetTracePlanRequest({traceId, planId}))).plan);
  return plans;
}
async function exportJson(profiling, traceId, pageSize = 1) {
  const chunks = []; let pageToken = '';
  do {
    const request = new p.ExportChromeTraceRequest({traceId, pageSize, pageToken});
    const page = ok(await profiling.exportChromeTrace(request));
    // Export reads are non-consuming and reproducible, including escaped Unicode metadata.
    const repeat = ok(await profiling.exportChromeTrace(request));
    assert.deepEqual(repeat.data, page.data);
    chunks.push(Buffer.from(page.data)); pageToken = page.nextPageToken;
  } while (pageToken);
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

test('default capture is basic host work; invalid detail is rejected before attachment', async t => {
  const f = await setup(t);
  for (const detail of [-1, 2]) {
    const invalid = await start(f.profiling, f.runtime.runtimeId, {detail});
    assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(invalid.traceId, 0n);
  }
  const trace = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  assert.equal(trace.options.detail, p.TraceDetail.TRACE_DETAIL_BASIC);
  assert.equal(trace.options.deviceTiming, false);
  assert.equal(trace.options.memory, false);
  assert.equal(trace.options.capacityBytes, 4194304n);
  assert.equal(trace.options.sampleIntervalNs, 100000000n);
  await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  assert.equal(stopped.state, p.TraceState.TRACE_STATE_READY);
  assert.equal(stopped.events.count, 1n);
  assert.deepEqual(stopped.devices, []);
  assert.deepEqual(await listResources(f.profiling, trace.traceId), []);
  const [event] = await listEvents(f.profiling, trace.traceId);
  assert.ok(event.host);
  assert.equal(event.node, undefined);
  assert.equal(event.activity, p.TraceActivity.TRACE_ACTIVITY_COMPUTE);
  const json = await exportJson(f.profiling, trace.traceId);
  assert.equal(json.otherData.format, 'volvoxai-trace/v8');
  assert.equal(json.otherData.detail, 'basic');
  assert.equal(json.otherData.deviceTiming, false);
  assert.equal(json.otherData.memory, false);
  await f.profiling.releaseTrace(new p.TraceRef(trace));
});

test('empty and bounded export pages validate page tokens and sizes', async t => {
  const f = await setup(t);
  const trace = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  const busy = await f.profiling.exportChromeTrace(new p.ExportChromeTraceRequest({traceId: trace.traceId}));
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  const info = ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  assert.equal(info.events.count, 0n);
  const json = await exportJson(f.profiling, trace.traceId);
  assert.ok(json.traceEvents.every(event => event.ph === 'M'));
  for (const options of [{pageSize: 0}, {pageSize: 1025}, {pageToken: '1'}, {pageToken: '-1'}, {pageToken: 'next'}]) {
    const invalid = await f.profiling.exportChromeTrace(new p.ExportChromeTraceRequest({traceId: trace.traceId, ...options}));
    assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(invalid.data?.length ?? 0, 0);
  }
  await f.profiling.releaseTrace(new p.TraceRef(trace));
});

test('real node records, independent executions, immutable pages and Chrome JSON', async t => {
  const f = await setup(t);
  await run(f); // Collection is absent during warmup.
  assert.ok(f.compiled.memoryBounds.bounds.length);
  const capture = ok(await start(f.profiling, f.runtime.runtimeId, {detail: p.TraceDetail.TRACE_DETAIL_NODES, memory: true}));
  assert.equal(capture.options.deviceTiming, false);
  assert.deepEqual(capture.devices, []);
  assert.equal(ok(await f.profiling.getTrace(new p.TraceRef(capture))).state, p.TraceState.TRACE_STATE_COLLECTING);
  const duplicate = await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId}));
  assert.equal(duplicate.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  assert.equal(duplicate.traceId, 0n);
  const busy = await f.profiling.listTraceEvents(new p.ListTraceEventsRequest({traceId: capture.traceId}));
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  const first = await run(f), second = await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(capture)));
  assert.equal(stopped.state, p.TraceState.TRACE_STATE_READY);
  assert.equal(stopped.events.dropped, 0n);
  await run(f); // Closed admission must not append another execution.
  const again = ok(await f.profiling.getTrace(new p.TraceRef(capture)));
  assert.equal(again.events.count, stopped.events.count);
  const records = await listEvents(f.profiling, capture.traceId);
  assert.equal(BigInt(records.length), stopped.events.count);
  const memory = await listResources(f.profiling, capture.traceId);
  assert.ok(memory.length >= 2);
  assert.ok(checkTraceMemory(records, stopped, p).length > 0);
  const nodes = records.filter(event => event.host !== undefined && event.node !== undefined);
  assert.equal(nodes.length, 4);
  assert.ok(nodes.every(event => event.phase === p.TracePhase.TRACE_PHASE_FORWARD && !event.program));
  assert.equal(records.filter(event => event.host && !event.node).length, 2);
  assert.deepEqual(new Set(nodes.map(event => event.node.scheduleIndex)), new Set([0, 1]));
  assert.deepEqual(new Set(nodes.map(event => event.lineage.executionId)), new Set([first.executionId, second.executionId]));
  for (const event of records.filter(event => event.host)) {
    assert.ok(event.host); assert.equal(event.device, undefined);
    assert.equal(event.lineage.runtimeId, f.runtime.runtimeId);
    assert.equal(event.lineage.modelId, f.model.modelId);
    assert.equal(event.lineage.compiledModelId, f.compiled.compiledModelId);
    assert.equal(event.lineage.contextId, f.context.contextId);
    assert.ok(event.host.durationNs >= 0n);
    assert.ok(memory[0].observationEndNs <= event.host.startNs);
    assert.ok(memory.at(-1).observationStartNs >= event.host.startNs + event.host.durationNs);
  }
  // Metadata survives context/model retirement without retaining their data.
  await f.inference.releaseExecutionContext(new p.ExecutionContextRef(f.context));
  const exported = await exportJson(f.profiling, capture.traceId);
  const singlePage = await exportJson(f.profiling, capture.traceId, 1024);
  assert.deepEqual(singlePage.otherData, exported.otherData);
  assert.deepEqual(singlePage.traceEvents.filter(event => event.ph !== 'M'),
    exported.traceEvents.filter(event => event.ph !== 'M'));
  const last = ok(await f.profiling.exportChromeTrace(new p.ExportChromeTraceRequest({
    traceId: capture.traceId, pageToken: String(stopped.events.count),
  })));
  assert.equal(last.nextPageToken, '');
  assert.equal(exported.otherData.detail, 'nodes');
  assert.equal(exported.otherData.deviceTiming, false);
  assert.equal(exported.otherData.memory, true);
  const slices = exported.traceEvents.filter(event => event.ph === 'X');
  assert.equal(slices.length, records.filter(event => event.host).length);
  const escaped = slices.find(event => event.args.output === 'out"한글');
  assert.ok(escaped);
  assert.equal(typeof escaped.args.executionId, 'string');
  const invalid = await f.profiling.listTraceEvents(new p.ListTraceEventsRequest({traceId: capture.traceId,
    pageToken: String(stopped.events.count + 1n)}));
  assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  await f.profiling.releaseTrace(new p.TraceRef(capture));
  await f.profiling.releaseTrace(new p.TraceRef(capture));
  const stale = await f.profiling.stopTrace(new p.TraceRef(capture));
  assert.equal(stale.report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
});

test('bounded overflow preserves numerics and reports loss; trace handles are owner scoped', async t => {
  const f = await setup(t);
  const other = await fixture({wasmUrl});
  t.after(() => other.close());
  const capture = ok(await start(f.profiling, f.runtime.runtimeId, {detail: p.TraceDetail.TRACE_DETAIL_NODES, capacityBytes: 4096n}));
  const foreign = new VxProfilingServiceClient(reportTransport(other.host));
  const refused = await foreign.stopTrace(new p.TraceRef(capture));
  assert.equal(refused.report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  for (let i = 0; i < 20; i++) await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(capture)));
  assert.ok(stopped.events.dropped > 0n);
  const json = await exportJson(f.profiling, capture.traceId, 1024);
  assert.equal(BigInt(json.otherData.droppedEvents), stopped.events.dropped);
  const next = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  await f.profiling.releaseTrace(new p.TraceRef(next)); // Release also stops admission.
  await run(f);
  const final = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  await f.profiling.releaseTrace(new p.TraceRef(final));
});

test('resource scopes expose exact WASM partition, ownership and unsupported CPU/GPU telemetry', async t => {
  const f = await setup(t);
  await run(f);
  const observed = ok(await f.profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest({contextId: f.context.contextId, includeDevice: true}))).snapshot;
  assert.equal(observed.memory.subject.ownerId, String(f.context.contextId));
  assert.equal(observed.memory.inventory, p.MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL);
  assert.ok(observed.memory.allocations.some(a => a.role === p.MemoryResourceRole.MEMORY_RESOURCE_ROLE_ARENA));
  for (const value of [observed.process.resident, observed.process.peakResident]) {
    assert.equal(value.status, p.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED);
    assert.equal(value.toJson().bytes, undefined);
  }
  assert.equal(observed.cpu.status, p.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED);
  assert.equal(observed.gpu.status, p.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED);
  const wasm = observed.wasm;
  assert.equal(wasm.status, p.ObservationStatus.OBSERVATION_STATUS_AVAILABLE);
  assert.equal(wasm.linearBytes, wasm.allocatedBytes + wasm.freeBytes + wasm.allocatorMetadataBytes +
    wasm.modulePrefixBytes + wasm.pageSlackBytes + wasm.untrackedBytes);
  assert.ok(wasm.largestFreeBlockBytes <= wasm.freeBytes);
  assert.ok(observed.observationEndNs >= observed.observationStartNs);
  const limited = ok(await f.profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest({module: new p.Empty(), maxAllocations: 1}))).snapshot;
  assert.equal(limited.memory.subject.kind, p.MemoryOwnerKind.MEMORY_OWNER_KIND_MODULE);
  assert.equal(limited.memory.allocations.length, 1);
  assert.equal(limited.memory.truncated, true);
  assert.equal(limited.gpu.status, p.ObservationStatus.OBSERVATION_STATUS_NOT_COLLECTED);
  const invalid = await f.profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest());
  assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
});

test('wire presence distinguishes schedule index zero from an operation without a node', () => {
  const span = new p.TraceHostSpan({startNs: 123n, durationNs: 456n});
  const operation = p.TraceEvent.fromBinary(new p.TraceEvent({host: span}).toBinary());
  const node = p.TraceEvent.fromBinary(new p.TraceEvent({host: span,
    node: new p.TraceNode({scheduleIndex: 0, outputName: 'first'})}).toBinary());
  assert.equal(operation.node, undefined);
  assert.equal(node.node.scheduleIndex, 0);
  assert.equal(node.node.outputName, 'first');
  assert.equal(node.host.durationNs, 456n);
  assert.equal(node.device, undefined);
  const device = p.TraceEvent.fromBinary(new p.TraceEvent({
    device: new p.TraceDeviceInterval({hostObservedNs: 123n, elapsedNs: 999n}),
  }).toBinary());
  assert.equal(device.host, undefined);
  assert.equal(device.device.elapsedNs, 999n);
  const program = p.TraceEvent.fromBinary(new p.TraceEvent({
    device: new p.TraceDeviceInterval({elapsedNs: 9n}),
    phase: p.TracePhase.TRACE_PHASE_BACKWARD,
    program: new p.TraceProgram({name: 'matMulBackward', entryPoint: 'weight_main'}),
    node: new p.TraceNode({scheduleIndex: 0, outputName: 'logits'}), tensorName: 'weight',
  }).toBinary());
  assert.equal(program.node.scheduleIndex, 0);
  assert.equal(program.phase, p.TracePhase.TRACE_PHASE_BACKWARD);
  assert.equal(program.program.entryPoint, 'weight_main');
  assert.equal(program.tensorName, 'weight');
  assert.equal(device.program, undefined);
});


test('public nanosecond fields and allocation identities preserve uint64 values beyond Number precision', () => {
  const value = (1n << 64n) - 1n;
  for (const [Type, key] of [[p.MonotonicTime, 'nanoseconds'], [p.RuntimeBudget, 'maxBatchDelayNs'],
    [p.SubmitOptions, 'deadlineMonotonicNs'], [p.CreateBatchQueueRequest, 'maxWaitNs'],
    [p.BatchQueueInfo, 'workerBusyNs'], [p.BatchQueueInfo, 'wallNs'],
    [p.CompilationMetrics, 'hostTimeNs'], [p.ShapePlanEvidence, 'bindTimeNs'],
    [p.ExecutionMetrics, 'hostTimeNs']]) {
    const message = Type.fromBinary(new Type({[key]: value}).toBinary());
    assert.equal(message[key], value, key);
  }
  const event = p.TraceEvent.fromBinary(new p.TraceEvent({memory: new p.TraceMemoryEvent({
    timestampNs: value, allocationId: value, bytes: value,
    action: p.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE,
  })}).toBinary());
  assert.equal(event.memory.timestampNs, value);
  assert.equal(event.memory.allocationId, value);
  assert.equal(event.host, undefined);
  assert.equal(event.device, undefined);
});


test('coalescing delay uses nanoseconds with a checked scheduler range', async t => {
  const f = await setup(t);
  for (const maxBatchDelayNs of [1n, 4_294_967_295_000_000n]) {
    const runtime = ok(await f.inference.createRuntime(new p.CreateRuntimeRequest({
      budget: new p.RuntimeBudget({maxBatchDelayNs}),
    })));
    await f.inference.releaseRuntime(new p.RuntimeRef(runtime));
  }
  for (const maxBatchDelayNs of [4_294_967_295_000_001n, (1n << 64n) - 1n]) {
    const invalid = await f.inference.createRuntime(new p.CreateRuntimeRequest({
      budget: new p.RuntimeBudget({maxBatchDelayNs}),
    }));
    assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(invalid.runtimeId, 0n);
  }
});

test('canonical plans and resources survive parent release, join events, and export through protobuf', async t => {
  const f = await setup(t);
  const trace = ok(await start(f.profiling, f.runtime.runtimeId, {
    memory: true, utilization: true, executionPlans: true, detail: p.TraceDetail.TRACE_DETAIL_NODES}));
  await run(f); await run(f);
  const info = ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  assert.equal(info.plans.count, 1n);
  assert.equal(info.plans.dropped, 0n);
  assert.ok(info.collectorBytes <= info.options.capacityBytes);
  await f.inference.releaseExecutionContext(new p.ExecutionContextRef(f.context));
  await f.inference.releaseCompiledModel(new p.CompiledModelRef(f.compiled));
  await f.inference.releaseModel(new p.ModelRef(f.model));
  const [plan] = await listPlans(f.profiling, trace.traceId, info.plans.count);
  assert.deepEqual(plan.steps.map(step => step.sourceNodeIds), [['relu'], ['sum']]);
  assert.equal(plan.sourceMappingComplete, true);
  const tensorNames = references => references.map(id => plan.tensors.find(tensor => tensor.tensorId === id)?.name);
  assert.deepEqual(plan.steps.map(step => tensorNames(step.inputs)), [['x'], ['first', 'first']]);
  assert.deepEqual(plan.steps.map(step => tensorNames(step.outputs)), [['first'], ['out"한글']]);
  assert.ok(plan.tensors.some(tensor => tensor.toJson().allocationId !== undefined && tensor.toJson().offsetBytes !== undefined));
  assert.ok(plan.allocations.some(allocation => allocation.toJson().traceAllocationId !== undefined));
  const events = await listEvents(f.profiling, trace.traceId, 4096);
  for (const event of events.filter(event => event.lineage.executionId)) assert.equal(event.planId, plan.planId);
  const missing = await f.profiling.getTracePlan(new p.GetTracePlanRequest({traceId: trace.traceId, planId: 2n}));
  assert.equal(missing.report.status, p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  const {collectProfile, analyzeProfile} = await import('../tools/profile_report.mjs');
  const artifact = await collectProfile(f.profiling, p, trace.traceId);
  const report = analyzeProfile(artifact);
  assert.equal(report.executions.length, 2);
  assert.equal(report.memory.plans.length, 1);
  assert.equal(report.memory.plans[0].arena.mappedTensorPeakBytes, 48n);
  assert.ok(report.memory.wasm.observedPeakLinearBytes > 0n);
  assert.equal(report.cpu.averageOccupiedCores, null);
  const chrome = await exportJson(f.profiling, trace.traceId, 1024);
  assert.ok(chrome.traceEvents.filter(e => e.name === 'WASM linear memory').length >= 2);
  assert.equal(chrome.otherData.planCount, '1');
  assert.equal(chrome.otherData.utilization, true);
  ok(await f.profiling.releaseTrace(new p.TraceRef(trace)));
});

test('plans follow concrete shapes and arena replacement without stale allocation joins', async t => {
  const f = await fixture({wasmUrl});
  t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const dynamic = structuredClone(graph);
  dynamic.dimensions = {N: {min: 1, max: 1024}};
  dynamic.inputs.x.shape = ['N'];
  for (const node of dynamic.nodes) node.outputs.out.shape = ['N'];
  const model = await f.load(dynamic), compiled = await f.compile(model);
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
  const trace = ok(await start(profiling, f.runtime.runtimeId, {memory: true, executionPlans: true}));
  const ids = [];
  for (const count of [4, 1024, 4, 4]) {
    const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId,
      inputs: tensors({x: {data: new Float32Array(count).fill(3), shape: [count]}})})));
    const output = await f.read(result, 'out"한글');
    assert.equal(output.length, count);
    assert.ok(output.every(value => value === 6));
    ids.push(result.executionId);
    ok(await f.inference.releaseResult(new p.ResultRef(result)));
  }
  ok(await profiling.stopTrace(new p.TraceRef(trace)));
  const {collectProfile} = await import('../tools/profile_report.mjs');
  const artifact = await collectProfile(profiling, p, trace.traceId);
  const operations = ids.map(id => artifact.events.find(event =>
    event.name === 'Execute' && event.lineage?.executionId === String(id)));
  assert.ok(operations.every(Boolean));
  const plans = operations.map(event => artifact.plans.find(plan => plan.planId === event.planId));
  assert.equal(artifact.plans.length, 3);
  assert.equal(plans[0].shapeSignature, plans[2].shapeSignature);
  assert.notEqual(plans[0].planId, plans[2].planId);
  assert.equal(plans[2].planId, plans[3].planId);
  const arena = plan => plan.allocations.find(a => a.allocator === 'host.arena');
  assert.notEqual(arena(plans[0]).traceAllocationId, arena(plans[2]).traceAllocationId);
  assert.equal(arena(plans[1]).traceAllocationId, arena(plans[2]).traceAllocationId);
  for (let i = 0; i < plans.length; i++) {
    assert.equal(plans[i].placementComplete, true);
    assert.deepEqual(plans[i].tensors.find(tensor => tensor.name === 'x').shape, [String([4, 1024, 4, 4][i])]);
  }
  ok(await profiling.releaseTrace(new p.TraceRef(trace)));
});

test('decode plans describe retained tensor storage after leaving the ordinary arena', async t => {
  const f = await fixture({wasmUrl});
  t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const compiled = await f.compile(await f.load(graph));
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest({
    ...compiled, decodeRowMode: p.DecodeRowMode.DECODE_ROW_MODE_AUTO,
  })));
  const trace = ok(await start(profiling, f.runtime.runtimeId, {
    memory: true, executionPlans: true, detail: p.TraceDetail.TRACE_DETAIL_NODES,
  }));
  const executions = [];
  async function execute(kind, values, expected) {
    const fields = {contextId: context.contextId,
      inputs: tensors({x: {data: Float32Array.from(values), shape: [4]}})};
    const request = kind === 'execute' ? new p.ExecuteRequest(fields) :
      kind === 'decodePrefill' ? new p.DecodePrefillRequest(fields) :
        new p.DecodeStepRequest({...fields, dependencyUpdate: new p.Empty()});
    const result = ok(await f.inference[kind](request));
    assert.deepEqual([...await f.read(result, 'out"한글')], expected);
    executions.push(result.executionId);
    ok(await f.inference.releaseResult(new p.ResultRef(result)));
  }
  await execute('execute', [-1, 2, -3, 4], [0, 4, 0, 8]);
  await execute('decodePrefill', [2, -3, 4, -5], [4, 0, 8, 0]);
  const retained = ok(await profiling.getResourceSnapshot(
    new p.GetResourceSnapshotRequest(context))).snapshot.memory.allocations;
  await execute('decodeStep', [-3, 4, -5, 6], [0, 8, 0, 12]);
  await execute('execute', [4, -5, 6, -7], [8, 0, 12, 0]);
  const info = ok(await profiling.stopTrace(new p.TraceRef(trace)));
  const events = await listEvents(profiling, trace.traceId, 4096);
  const plans = await listPlans(profiling, trace.traceId, info.plans.count);
  const nodes = executions.map(id => events.filter(event => event.node && event.lineage.executionId === id));
  assert.ok(nodes.every(execution => execution.length === graph.nodes.length));
  const layouts = nodes.map(execution => plans.find(plan => plan.planId === execution[0].planId));
  assert.ok(layouts.every(plan => plan?.placementComplete));
  assert.notEqual(layouts[0].planId, layouts[1].planId);
  assert.equal(layouts[1].planId, layouts[2].planId);
  assert.notEqual(layouts[2].planId, layouts[3].planId);
  assert.notEqual(layouts[0].planId, layouts[3].planId);
  assert.ok(layouts[0].allocations.some(allocation => allocation.allocator === 'host.arena'));
  assert.ok(layouts[3].allocations.some(allocation => allocation.allocator === 'host.arena'));
  for (const tensor of layouts[1].tensors) {
    const allocation = layouts[1].allocations.find(item => item.allocationId === tensor.allocationId);
    const actual = retained.find(item => item.name === tensor.name && item.allocator === 'host.tensor');
    assert.ok(actual, `missing retained storage for ${tensor.name}`);
    assert.equal(allocation?.allocator, actual.allocator);
    assert.equal(allocation.name, actual.name);
    assert.equal(allocation.capacityBytes, actual.capacityBytes);
    assert.equal(allocation.space, actual.space);
    assert.equal(tensor.offsetBytes, 0n);
  }
  for (let i = 0; i < layouts.length; i++) {
    for (const tensor of layouts[i].tensors) {
      const allocation = layouts[i].allocations.find(item => item.allocationId === tensor.allocationId);
      if (!allocation.traceAllocationId) continue;
      const history = events.filter(event => event.sequence < nodes[i][0].sequence &&
        event.memory?.allocationId === allocation.traceAllocationId);
      assert.ok(history.length, `missing allocator history for ${tensor.name}`);
      assert.notEqual(history.at(-1).memory.action, p.TraceMemoryAction.TRACE_MEMORY_ACTION_FREE,
        `${tensor.name} refers to storage freed before execution`);
    }
  }
  ok(await profiling.releaseTrace(new p.TraceRef(trace)));
});

test('Run inventories private contexts and keeps same-shape compiled plans distinct', async t => {
  const f = await fixture({wasmUrl});
  t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const variant = structuredClone(graph);
  variant.nodes[1].inputs.b = 'x';
  const models = await Promise.all([f.load(graph), f.load(variant)]);
  const compiled = [];
  for (const model of models) compiled.push(await f.compile(model));
  const trace = ok(await start(profiling, f.runtime.runtimeId, {executionPlans: true, memory: true}));
  for (let i = 0; i < compiled.length; i++) {
    const result = ok(await f.inference.run(new p.RunRequest({compiledModelId: compiled[i].compiledModelId,
      inputs: tensors({x: {data: Float32Array.of(-1, 2, -3, 4), shape: [4]}})})));
    assert.deepEqual([...await f.read(result, 'out"한글')], i ? [-1, 4, -3, 8] : [0, 4, 0, 8]);
    ok(await f.inference.releaseResult(new p.ResultRef(result)));
  }
  const snapshot = ok(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest(f.runtime))).snapshot;
  const arenas = snapshot.memory.allocations.filter(a => a.allocator === 'host.arena');
  assert.equal(arenas.length, 2);
  assert.ok(arenas.every(a => a.owner.kind === p.MemoryOwnerKind.MEMORY_OWNER_KIND_COMPILED_MODEL));
  assert.deepEqual(new Set(arenas.map(a => a.owner.ownerId)), new Set(compiled.map(c => String(c.compiledModelId))));
  const stopped = ok(await profiling.stopTrace(new p.TraceRef(trace)));
  const plans = await listPlans(profiling, trace.traceId, stopped.plans.count);
  assert.equal(plans.length, 2);
  assert.notEqual(plans[0].graphFingerprint, plans[1].graphFingerprint);
  assert.deepEqual(new Set(plans.map(plan => plan.lineage.compiledModelId)), new Set(compiled.map(c => c.compiledModelId)));
  ok(await profiling.releaseTrace(new p.TraceRef(trace)));
});

test('WASM VFS weight mappings remain allocations inside linear memory', async t => {
  const f = await fixture({wasmUrl});
  t.after(() => f.close());
  const model = await f.load({format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {dtype: 'float32', shape: [4]}},
    nodes: [{id: 'add', opType: 'Add', inputs: {a: 'x', b: 'weight'},
      outputs: {out: {tensor: 'y', dtype: 'float32', shape: [4]}}, params: {}}], outputs: ['y'],
  }, safetensors([{name: 'weight', shape: [4], data: Float32Array.of(1, 2, 3, 4)}]));
  await f.compile(model);
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const snapshot = ok(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest(f.runtime))).snapshot;
  const weights = snapshot.memory.allocations.filter(a => a.role === p.MemoryResourceRole.MEMORY_RESOURCE_ROLE_WEIGHTS);
  assert.equal(weights.length, 1);
  assert.equal(weights[0].space, p.MemorySpace.MEMORY_SPACE_WASM_LINEAR);
  assert.ok(weights[0].capacityBytes >= 16n);
});

test('application annotations join the trace on the platform clock', async t => {
  const f = await setup(t);
  const platform = new VxPlatformServiceClient(reportTransport(f.host));
  const trace = ok(await start(f.profiling, f.runtime.runtimeId));
  const begin = (await platform.getMonotonicTime(new p.Empty())).nanoseconds;
  await run(f);
  const end = (await platform.getMonotonicTime(new p.Empty())).nanoseconds;
  ok({report: await f.profiling.annotateTrace(new p.AnnotateTraceRequest({traceId: trace.traceId, name: 'frame 1', startNs: begin, endNs: end}))});
  for (const invalid of [{name: ''}, {startNs: end, endNs: begin}]) {
    const refused = await f.profiling.annotateTrace(new p.AnnotateTraceRequest({traceId: trace.traceId, name: 'x', startNs: begin, endNs: end, ...invalid}));
    assert.equal(refused.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  }
  ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  const late = await f.profiling.annotateTrace(new p.AnnotateTraceRequest({traceId: trace.traceId, name: 'late', startNs: begin, endNs: end}));
  assert.equal(late.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  const events = await listEvents(f.profiling, trace.traceId, 16);
  const annotation = events.find(event => event.activity === p.TraceActivity.TRACE_ACTIVITY_ANNOTATION);
  const execute = events.find(event => event.name === 'Execute');
  assert.equal(annotation.name, 'frame 1');
  assert.equal(annotation.host.durationNs, end - begin);
  assert.ok(annotation.host.startNs <= execute.host.startNs);
  const chrome = await exportJson(f.profiling, trace.traceId, 16);
  assert.ok(chrome.traceEvents.some(event => event.cat === 'user.annotation' && event.name === 'frame 1'));
  ok(await f.profiling.releaseTrace(new p.TraceRef(trace)));
});
