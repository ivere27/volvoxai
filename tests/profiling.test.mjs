import assert from 'node:assert/strict';
import test from 'node:test';
import {checkTraceMemory} from '../tools/trace_memory_evidence.mjs';
import { fixture, p, ok, tensors } from '../tools/proto_fixture.mjs';
import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { VxProfilingServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import { MemoryEvidenceValidatingTransport } from '../runtime/typescript/MemoryEvidenceValidatingTransport.js';
import { validateMemoryBounds } from '../runtime/typescript/MemoryEvidenceValidation.js';

const wasmUrl = new URL('../dist/0.6.0/volvoxai.lite.wasm', import.meta.url).pathname;
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
async function exportJson(profiling, traceId, limit = 1) {
  let offset = 0n;
  const chunks = [];
  for (;;) {
    const request = new p.ExportChromeTraceRequest({traceId, offset, limit});
    const page = ok(await profiling.exportChromeTrace(request));
    // Export reads are non-consuming and reproducible, including escaped Unicode metadata.
    const repeat = ok(await profiling.exportChromeTrace(request));
    assert.deepEqual(repeat.data, page.data);
    chunks.push(Buffer.from(page.data));
    assert.ok(page.nextOffset >= offset && page.nextOffset <= offset + BigInt(limit));
    offset = page.nextOffset;
    if (page.eof) break;
    assert.equal(page.nextOffset, request.offset + BigInt(limit));
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

test('default capture is basic host work; invalid detail is rejected before attachment', async t => {
  const f = await setup(t);
  for (const detail of [-1, 2]) {
    const invalid = await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId, detail}));
    assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
    assert.equal(invalid.traceId, 0n);
  }
  const trace = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  assert.equal(trace.detail, p.TraceDetail.TRACE_DETAIL_BASIC);
  assert.equal(trace.deviceTiming, false);
  assert.equal(trace.memory, false);
  assert.equal(trace.capacityBytes, 4194304n);
  await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  assert.equal(stopped.eventCount, 1n);
  assert.deepEqual(stopped.devices, []);
  const page = ok(await f.profiling.readTrace(new p.ReadTraceRequest(trace)));
  assert.deepEqual(page.processMemory, []);
  assert.ok(page.events[0].host);
  assert.equal(page.events[0].node, undefined);
  const json = await exportJson(f.profiling, trace.traceId);
  assert.equal(json.otherData.format, 'volvoxai-trace/v6');
  assert.equal(json.otherData.detail, 'basic');
  assert.equal(json.otherData.deviceTiming, false);
  assert.equal(json.otherData.memory, false);
  await f.profiling.releaseTrace(new p.TraceRef(trace));
});

test('empty and bounded export pages validate event offsets and limits', async t => {
  const f = await setup(t);
  const trace = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  const busy = await f.profiling.exportChromeTrace(new p.ExportChromeTraceRequest({traceId: trace.traceId}));
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  const info = ok(await f.profiling.stopTrace(new p.TraceRef(trace)));
  assert.equal(info.eventCount, 0n);
  const json = await exportJson(f.profiling, trace.traceId);
  assert.ok(json.traceEvents.every(event => event.ph === 'M'));
  for (const options of [{limit: 0}, {limit: 1025}, {offset: 1n}, {offset: 2n ** 63n}]) {
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
  const capture = ok(await f.profiling.startTrace(new p.StartTraceRequest({
    runtimeId: f.runtime.runtimeId, detail: p.TraceDetail.TRACE_DETAIL_NODES, memory: true,
  })));
  assert.equal(capture.deviceTiming, false);
  assert.deepEqual(capture.devices, []);
  assert.equal(ok(await f.profiling.getTrace(new p.TraceRef(capture))).state, p.TraceState.TRACE_STATE_COLLECTING);
  const duplicate = await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId}));
  assert.equal(duplicate.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  assert.equal(duplicate.traceId, 0n);
  const busy = await f.profiling.readTrace(new p.ReadTraceRequest(capture));
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  const first = await run(f), second = await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(capture)));
  assert.equal(stopped.state, p.TraceState.TRACE_STATE_READY);
  assert.equal(stopped.droppedEvents, 0n);
  await run(f); // Closed admission must not append another execution.
  const again = ok(await f.profiling.getTrace(new p.TraceRef(capture)));
  assert.equal(again.eventCount, stopped.eventCount);
  const records = [];
  let memory;
  for (let offset = 0n;;) {
    const page = await f.profiling.readTrace(new p.ReadTraceRequest({traceId: capture.traceId, offset, limit: 1}));
    ok(page);
    records.push(...page.events);
    if (!offset) {
      assert.equal(page.processMemory.length, 2);
      memory = page.processMemory;
      assert.equal(page.processMemory[0].resourceInventory, p.MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL);
    } else assert.equal(page.processMemory.length, 0);
    if (page.eof) break;
    assert.equal(page.nextOffset, offset + 1n);
    offset = page.nextOffset;
  }
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
    assert.ok(memory[1].observationStartNs >= event.host.startNs + event.host.durationNs);
  }
  // Metadata survives context/model retirement without retaining their data.
  await f.inference.releaseExecutionContext(new p.ExecutionContextRef(f.context));
  const exported = await exportJson(f.profiling, capture.traceId);
  const singlePage = await exportJson(f.profiling, capture.traceId, 1024);
  assert.deepEqual(singlePage.otherData, exported.otherData);
  assert.deepEqual(singlePage.traceEvents.filter(event => event.ph !== 'M'),
    exported.traceEvents.filter(event => event.ph !== 'M'));
  const end = ok(await f.profiling.exportChromeTrace(new p.ExportChromeTraceRequest({
    traceId: capture.traceId, offset: stopped.eventCount,
  })));
  assert.equal(end.eof, true);
  assert.equal(end.data?.length ?? 0, 0);
  assert.equal(exported.otherData.detail, 'nodes');
  assert.equal(exported.otherData.deviceTiming, false);
  assert.equal(exported.otherData.memory, true);
  const slices = exported.traceEvents.filter(event => event.ph === 'X');
  assert.equal(slices.length, records.filter(event => event.host).length);
  const escaped = slices.find(event => event.args.output === 'out"한글');
  assert.ok(escaped);
  assert.equal(typeof escaped.args.executionId, 'string');
  const invalid = await f.profiling.readTrace(new p.ReadTraceRequest({traceId: capture.traceId, offset: stopped.eventCount + 1n}));
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
  const capture = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId,
    detail: p.TraceDetail.TRACE_DETAIL_NODES, capacityBytes: 4096n})));
  const foreign = new VxProfilingServiceClient(reportTransport(other.host));
  const refused = await foreign.stopTrace(new p.TraceRef(capture));
  assert.equal(refused.report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  for (let i = 0; i < 20; i++) await run(f);
  const stopped = ok(await f.profiling.stopTrace(new p.TraceRef(capture)));
  assert.ok(stopped.droppedEvents > 0n);
  const json = await exportJson(f.profiling, capture.traceId, 1024);
  assert.equal(BigInt(json.otherData.droppedEvents), stopped.droppedEvents);
  const next = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  await f.profiling.releaseTrace(new p.TraceRef(next)); // Release also stops admission.
  await run(f);
  const final = ok(await f.profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  await f.profiling.releaseTrace(new p.TraceRef(final));
});

test('on-demand memory scopes distinguish counters, process envelopes and compilation bounds', async t => {
  const f = await setup(t);
  const observed = ok(await f.profiling.getMemorySnapshot(new p.GetMemorySnapshotRequest({contextId: f.context.contextId, includeProcess: true})));
  assert.equal(observed.snapshot.subject.ownerId, String(f.context.contextId));
  assert.equal(observed.snapshot.resourceInventory, p.MemoryInventoryKind.MEMORY_INVENTORY_KIND_PARTIAL);
  assert.equal(observed.snapshot.counters[0].name, 'host_arena');
  assert.equal(observed.snapshot.envelopes.length, 2);
  for (const envelope of observed.snapshot.envelopes) {
    assert.equal(envelope.valueRelation, p.MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE);
    assert.equal(envelope.bytes, undefined);
  }
  assert.ok(observed.snapshot.observationEndNs >= observed.snapshot.observationStartNs);
  const invalid = await f.profiling.getMemorySnapshot(new p.GetMemorySnapshotRequest());
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
    [p.CompiledModelHandle, 'compileTimeNs'], [p.ShapePlanEvidence, 'bindTimeNs'],
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
