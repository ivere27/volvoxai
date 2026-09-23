/** Release-artifact profiling qualification, runnable with Node (WASM) or Deno
 * (physical WebGPU). No software adapter or operator fallback is accepted.
 *
 * node tools/qualify_profiling.mjs wasm lite
 * node tools/qualify_profiling.mjs wasm full
 * Optional fifth argument: executable node count, for example 80 for query batching.
 * deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env \
 *   --allow-ffi tools/qualify_profiling.mjs webgpu full
 */
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import process from 'node:process';
import {checkTraceMemory} from './trace_memory_evidence.mjs';

const [backend, profile] = process.argv.slice(2);
assert.ok(['wasm', 'webgpu'].includes(backend));
assert.ok(['lite', 'full'].includes(profile));
assert.ok(backend !== 'webgpu' || profile === 'full', 'WebGPU belongs to the full profile');
const stem = `volvoxai${profile === 'lite' ? '.lite' : ''}`;
const bundle = new URL(`../dist/0.6.0/${stem}.min.js`, import.meta.url);
const wasm = new URL(`../dist/0.6.0/${stem}.wasm`, import.meta.url);
const output = new URL(process.argv[4] ? `${process.argv[4].replace(/\/$/, '')}/` : '../build/profiling-hardware/', process.argv[4] ? new URL(`file://${process.cwd()}/`) : import.meta.url);
const api = await import(bundle.href), p = api.pb;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const ok = value => {
  const report = value.report ?? value;
  assert.equal(report.status, p.NativeStatus.NATIVE_STATUS_OK, JSON.stringify(report,
    (_, v) => typeof v === 'bigint' ? String(v) : v));
  return value;
};
const width = 256, name = 'out"한글';
const nodeCount = Number(process.argv[5] ?? 2);
assert.ok(Number.isInteger(nodeCount) && nodeCount >= 2 && nodeCount <= 1024);
const graph = {format: 'volvox-graph/v1', dimensions: {},
  inputs: {x: {dtype: 'float32', shape: [1, width]}}, nodes: Array.from({length: nodeCount}, (_, index) => ({
    id: `node_${index}`, opType: index ? 'Add' : 'ReLU',
    inputs: index ? {a: `value_${index - 1}`, b: 'x'} : {input: 'x'},
    outputs: {out: {tensor: index === nodeCount - 1 ? name : `value_${index}`,
      dtype: 'float32', shape: [1, width]}}, params: {},
  })), outputs: [name]};

const gpu = globalThis.navigator?.gpu, requestAdapter = gpu?.requestAdapter;
const fetchOriginal = globalThis.fetch;
let submissions = 0, devices = 0, executions = 0, adapterInfo;
let timingQueries = 0, collecting = false, timingEnabled = false;
const timingCreates = {on: [], off: []};
const errors = [];
if (backend === 'webgpu') {
  assert.ok(gpu, 'physical WebGPU is required');
  gpu.requestAdapter = async (...args) => {
    const adapter = await requestAdapter.apply(gpu, args);
    assert.ok(adapter, 'physical adapter unavailable');
    const info = adapter.info ?? await adapter.requestAdapterInfo();
    adapterInfo = {vendor: info.vendor, device: info.device, description: info.description};
    assert.ok(!adapter.isFallbackAdapter && !info.isFallbackAdapter &&
      !/swiftshader|llvmpipe|lavapipe|softpipe|software|microsoft basic render|\bcpu\b/i
        .test(JSON.stringify(adapterInfo)), 'software adapter refused');
    const requestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (...deviceArgs) => {
      const device = await requestDevice(...deviceArgs);
      devices++;
      device.addEventListener('uncapturederror', event => errors.push(event.error.message));
      const createQuerySet = device.createQuerySet.bind(device);
      device.createQuerySet = (...args) => { timingQueries++; return createQuerySet(...args); };
      const submit = device.queue.submit.bind(device.queue);
      device.queue.submit = (...buffers) => { submissions++; return submit(...buffers); };
      return device;
    };
    return adapter;
  };
}
globalThis.fetch = async (url, init) => String(url).startsWith('file:')
  ? new Response(await readFile(fileURLToPath(url))) : fetchOriginal(url, init);
const host = new (profile === 'full' ? api.FullEngineHost : api.EngineHost)({wasmUrl: wasm});
try {
  const inference = new api.VxInferenceServiceClient(host);
  const profiling = new api.VxProfilingServiceClient(host);
  const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({cpuThreads: 1})));
  const model = ok(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(graph))})})));
  const compiled = ok(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
    policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: [backend], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
  assert.equal(compiled.report.backend, backend);
  assert.equal(compiled.report.route.attested, true);
  const context = ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
  async function run() {
    const beforeQueries = timingQueries;
    const values = Float32Array.from({length: width}, (_, i) => (i % 17 - 8 + executions % 5) / 4);
    executions++;
    const result = ok(await inference.execute(new p.ExecuteRequest({contextId: context.contextId,
      inputs: [new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_F32,
        shape: [1n, BigInt(width)], inline: new Uint8Array(values.buffer)})]})));
    assert.equal(result.report.backend, backend);
    assert.equal(result.report.route.attested, true);
    const deadline = performance.now() + 30000;
    for (;;) {
      const ready = ok(await inference.getResult(new p.ResultRef(result)));
      if (ready.state !== p.ResultState.RESULT_STATE_PENDING) {
        assert.equal(ready.state, p.ResultState.RESULT_STATE_READY); break;
      }
      assert.ok(performance.now() < deadline, 'GPU completion timed out');
      await new Promise(resolve => setTimeout(resolve, 1));
    }
    const tensor = ok(await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name}))).tensor;
    const actual = new Float32Array(tensor.inline.slice().buffer);
    assert.deepEqual([...actual], [...values].map(x => Math.max(0, x) + (nodeCount - 1) * x));
    await inference.releaseResult(new p.ResultRef(result));
    const created = timingQueries - beforeQueries;
    timingCreates[collecting ? 'on' : 'off'].push(created);
    if (!collecting || !timingEnabled) assert.equal(created, 0, 'disabled profiling creates no WebGPU queries');
    return result.executionId;
  }
  // Large schedules include node, program, copy and allocation records.
  async function capture(count, capacityBytes = nodeCount > 64 ? 8388608n : 4194304n, deviceTiming = true, detail = p.TraceDetail.TRACE_DETAIL_NODES) {
    const trace = ok(await profiling.startTrace(new p.StartTraceRequest({runtimeId: runtime.runtimeId,
      detail, deviceTiming, capacityBytes, memory: true})));
    const identities = [];
    collecting = true; timingEnabled = deviceTiming;
    for (let i = 0; i < count; i++) identities.push(await run());
    let stopped = ok(await profiling.stopTrace(new p.TraceRef(trace)));
    const deadline = performance.now() + 30000;
    while (stopped.state === p.TraceState.TRACE_STATE_DRAINING) {
      assert.ok(performance.now() < deadline, 'timestamp drain timed out');
      await new Promise(resolve => setTimeout(resolve, 0));
      stopped = ok(await profiling.getTrace(new p.TraceRef(trace)));
    }
    assert.equal(stopped.state, p.TraceState.TRACE_STATE_READY);
    collecting = false;
    await run();
    assert.equal(ok(await profiling.getTrace(new p.TraceRef(trace))).eventCount, stopped.eventCount);
    return {trace, stopped, identities};
  }
  async function inspect({trace, stopped, identities}, phase, overflow = false) {
    const events = [], memory = [];
    for (let offset = 0n;;) {
      const request = new p.ReadTraceRequest({traceId: trace.traceId, offset, limit: 3});
      const page = ok(await profiling.readTrace(request));
      assert.deepEqual(ok(await profiling.readTrace(request)), page);
      if (offset) assert.equal(page.processMemory.length, 0);
      events.push(...page.events); memory.push(...page.processMemory);
      if (page.eof) break;
      assert.equal(page.nextOffset, offset + BigInt(page.events.length));
      offset = page.nextOffset;
    }
    const operations = events.filter(e => e.host !== undefined && e.node === undefined && e.activity === p.TraceActivity.TRACE_ACTIVITY_WORK);
    const nodes = events.filter(e => e.host !== undefined && e.node !== undefined && e.activity === p.TraceActivity.TRACE_ACTIVITY_WORK);
    assert.equal(BigInt(events.length), stopped.eventCount);
    assert.equal(memory.length, 2);
    assert.ok(events.filter(e => !e.memory).every(e => identities.includes(e.lineage.executionId)));
    const device = events.filter(e => e.device !== undefined);
    const programs = device.filter(e => e.program);
    const deviceScopes = device.filter(e => !e.program);
    const activities = events.filter(e => e.host && e.activity !== p.TraceActivity.TRACE_ACTIVITY_WORK);
    const enabled = backend === 'webgpu' && stopped.deviceTiming;
    const nodeTiming = stopped.detail === p.TraceDetail.TRACE_DETAIL_NODES;
    // A tiny budget can fill with initial memory inventory before a device
    // adapter is probed. An empty support inventory then remains unobserved.
    if (!overflow) {
      assert.equal(stopped.devices.some(d => d.support === p.TraceSupport.TRACE_SUPPORT_AVAILABLE), enabled);
      assert.equal(stopped.devices.length, enabled || (backend === 'webgpu' && nodeTiming) ? 1 : 0);
    }
    if (stopped.devices.length && enabled) {
      const coverage = stopped.devices[0];
      assert.equal(coverage.nodeTimingAvailable, true);
      assert.equal(coverage.passIntervals, BigInt(deviceScopes.filter(e => !e.node).length));
      assert.equal(coverage.nodeIntervals, BigInt(deviceScopes.filter(e => e.node).length));
      assert.equal(coverage.programTimingAvailable, true);
      assert.equal(coverage.programIntervals, BigInt(programs.length));
      assert.equal(coverage.failedIntervals, 0n); assert.equal(coverage.unavailablePasses, 0n);
      assert.equal(coverage.splitsPasses, nodeTiming); assert.equal(coverage.addsBarriers, false);
    }
    assert.equal(events.length, operations.length + nodes.length + activities.length + device.length + events.filter(e => e.memory).length);
    if (overflow) assert.ok(stopped.droppedEvents > 0n);
    else {
      assert.equal(stopped.droppedEvents, 0n);
      assert.ok(checkTraceMemory(events, stopped, p).length > 0);
      if (backend === 'webgpu') assert.ok(stopped.allocators.some(a => a.allocator === 'webgpu.buffer'));
      assert.equal(operations.length, identities.length);
      assert.equal(nodes.length, stopped.detail === p.TraceDetail.TRACE_DETAIL_NODES ? identities.length * nodeCount : 0);
      assert.equal(deviceScopes.length, enabled ? identities.length * (nodeTiming ? nodeCount : 1) : 0);
      assert.equal(programs.length, enabled && nodeTiming ? identities.length * nodeCount : 0);
      assert.ok(programs.every(e => e.phase === p.TracePhase.TRACE_PHASE_FORWARD && e.node && e.program.entryPoint));
      assert.ok(device.every(e => (e.node !== undefined) === nodeTiming && e.device.elapsedNs >= 0n));
    }
    if (!overflow && backend === 'webgpu' && nodeTiming) {
      assert.ok(activities.some(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_COPY));
      assert.ok(activities.some(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_SUBMIT));
      assert.ok(activities.some(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_AWAIT));
      assert.ok(!activities.some(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_WAIT));
      assert.ok(activities.every(e => e.queue?.deviceId && e.queue.queueId));
      assert.ok(activities.filter(e => e.copy).every(e => e.copy.bytes > 0n));
      assert.ok(device.every(e => e.queue?.deviceId && e.queue.queueId && e.queue.submissionId));
      assert.ok(device.every(e => e.device.correlation?.method === p.TraceClockMethod.TRACE_CLOCK_METHOD_BOUNDED));
      assert.ok(device.every(e => e.device.correlation.earliestStartNs <= e.device.correlation.latestStartNs));
      const coverage = stopped.devices[0];
      assert.equal(coverage.boundedIntervals, BigInt(device.length));
      assert.equal(coverage.calibratedIntervals, 0n);
      assert.equal(coverage.hostAwaits, BigInt(activities.filter(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_AWAIT).length));
    }
    const chunks = [];
    for (let offset = 0n;;) {
      const request = new p.ExportChromeTraceRequest({traceId: trace.traceId, offset, limit: 3});
      const chunk = ok(await profiling.exportChromeTrace(request));
      assert.deepEqual(ok(await profiling.exportChromeTrace(request)), chunk);
      chunks.push(Buffer.from(chunk.data));
      assert.equal(chunk.nextOffset, offset + (stopped.eventCount - offset < 3n ? stopped.eventCount - offset : 3n));
      offset = chunk.nextOffset;
      if (chunk.eof) { assert.equal(offset, stopped.eventCount); break; }
    }
    const data = Buffer.concat(chunks), exported = JSON.parse(data.toString('utf8'));
    assert.equal(exported.otherData.format, 'volvoxai-trace/v6');
    assert.equal(exported.otherData.detail, nodeTiming ? 'nodes' : 'basic');
    assert.equal(exported.otherData.deviceTiming, stopped.deviceTiming);
    assert.equal(exported.otherData.memory, true);
    assert.deepEqual(exported.otherData.devices.map(d => d.nodeTimingAvailable), stopped.devices.map(d => d.nodeTimingAvailable));
    const awaits = activities.filter(e => e.activity === p.TraceActivity.TRACE_ACTIVITY_AWAIT);
    assert.equal(exported.traceEvents.filter(e => e.ph === 'X').length, operations.length + nodes.length + activities.length - awaits.length);
    const begins = exported.traceEvents.filter(e => e.ph === 'b');
    const ends = exported.traceEvents.filter(e => e.ph === 'e');
    assert.equal(begins.length, awaits.length); assert.equal(ends.length, awaits.length);
    for (const event of awaits) {
      const begin = begins.find(e => e.id === String(event.sequence));
      const end = ends.find(e => e.id === String(event.sequence));
      assert.ok(begin && end); assert.equal(begin.cat, 'host.await');
      assert.equal(begin.pid, 3); assert.equal(end.pid, begin.pid);
      assert.equal(end.name, begin.name);
      assert.ok(Math.abs((end.ts - begin.ts) * 1000 - Number(event.host.durationNs)) < 1);
    }
    const filename = `${backend}-${profile}-${phase}.trace.json`;
    await writeFile(new URL(filename, output), data);
    ok(await profiling.releaseTrace(new p.TraceRef(trace)));
    return {phase, allocators: stopped.allocators.map(a => a.toJson()), memoryEvents: events.filter(e => e.memory).length, detail: stopped.detail, deviceTiming: stopped.deviceTiming, devices: stopped.devices.map(d => d.toJson()), hostOperations: operations.length, hostNodes: nodes.length, deviceIntervals: device.length, deviceDurationNs: device.map(e => String(e.device.elapsedNs)),
      droppedEvents: String(stopped.droppedEvents), trace: filename, traceSha256: sha256(data)};
  }
  await mkdir(output, {recursive: true});
  const cases = [await inspect(await capture(4), 'cold')];
  for (let i = 0; i < 8; i++) await run();
  const warm = await capture(10);
  cases.push(await inspect(await capture(20, 4096n), 'overflow', true));
  cases.push(await inspect(await capture(3, 4194304n, false), 'nodes-host-only'));
  cases.push(await inspect(await capture(3, 4194304n, true, p.TraceDetail.TRACE_DETAIL_BASIC), 'basic-device'));
  cases.push(await inspect(await capture(3, 4194304n, false, p.TraceDetail.TRACE_DETAIL_BASIC), 'basic-host-only'));
  for (let i = 0; i < 8; i++) await run();
  const snapshot = ok(await profiling.getMemorySnapshot(new p.GetMemorySnapshotRequest({
    contextId: context.contextId, includeProcess: true}))).snapshot;
  assert.equal(snapshot.counters[0].name, 'host_arena');
  assert.equal(snapshot.envelopes.length, 2);
  assert.ok(snapshot.envelopes.every(e =>
    e.valueRelation === p.MemoryValueRelation.MEMORY_VALUE_RELATION_UNAVAILABLE && !e.bytes));
  const lifetimeTrace = ok(await profiling.startTrace(new p.StartTraceRequest({
    runtimeId: runtime.runtimeId, detail: p.TraceDetail.TRACE_DETAIL_NODES, memory: true,
  })));
  collecting = true; timingEnabled = false;
  const lifetimeIds = [];
  const buffers = new api.VxBufferServiceClient(host);
  let alias;
  if (backend === 'wasm') {
    const values = new Float32Array(width).fill(2);
    const request = new p.ExecuteTensorsRequest({contextId: context.contextId, inputs: [new p.Tensor({
      name: 'x', shape: [1n, BigInt(width)], dtype: p.DataType.DATA_TYPE_F32, inline: new Uint8Array(values.buffer),
    })]});
    const first = ok(await inference.executeTensors(request));
    alias = ok(await buffers.retainBuffers(new p.BufferRefs({bufferIds: [first.outputs[0].buffer.bufferId]})));
    ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: [first.outputs[0].buffer.bufferId]})));
    const second = ok(await inference.executeTensors(request));
    ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: [second.outputs[0].buffer.bufferId]})));
    const reused = ok(await inference.executeTensors(request));
    const copied = ok(await buffers.copyTensors(new p.CopyTensorsRequest({sources: reused.outputs, inlineResult: true})));
    assert.deepEqual([...new Float32Array(copied.outputs[0].inline.slice().buffer)], Array(width).fill(2 * nodeCount));
    ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: [reused.outputs[0].buffer.bufferId]})));
    lifetimeIds.push(...[first, second, reused].map(r => r.report.lineage.executionId));
    executions += 3;
    const observed = ok(await profiling.getMemorySnapshot(new p.GetMemorySnapshotRequest({contextId: context.contextId}))).snapshot;
    const capacity = new Map(observed.counters.map(c => [c.name, c.measurement.bytes.bytes]));
    assert.equal(capacity.get('retained_result_capacity'), BigInt(2 * width * 4));
    assert.equal(capacity.get('idle_result_capacity'), BigInt(width * 4));
  } else {
    for (let i = 0; i < 3; i++) lifetimeIds.push(await run());
  }
  for (const [method, ref] of [
    ['releaseExecutionContext', new p.ExecutionContextRef(context)],
    ['releaseCompiledModel', new p.CompiledModelRef(compiled)],
    ['releaseModel', new p.ModelRef(model)],
  ]) ok(await inference[method](ref));
  if (alias) {
    const retainedInfo = ok(await profiling.getTrace(new p.TraceRef(lifetimeTrace)));
    assert.equal(retainedInfo.allocators.find(a => a.allocator === 'host.result').liveBytes, BigInt(width * 4));
    ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: [alias.buffers[0].bufferId]})));
  }
  // Context retirement frees numerical buffers. The validated timestamp pool
  // is bridge-owned and remains reusable until host close: two 16 KiB buffers
  // per created batch, separate from model/context lifetime.
  const retainedTimingBytes = BigInt(timingQueries) * 32768n;
  const retirementDeadline = performance.now() + 30000;
  for (;;) {
    const info = ok(await profiling.getTrace(new p.TraceRef(lifetimeTrace)));
    if (info.allocators.every(a => a.liveBytes ===
        (a.allocator === 'webgpu.buffer' ? retainedTimingBytes : 0n))) break;
    assert.ok(performance.now() < retirementDeadline, JSON.stringify(info.allocators.map(a => a.toJson())));
    await new Promise(resolve => setTimeout(resolve, 1));
  }
  const lifetimeStopped = ok(await profiling.stopTrace(new p.TraceRef(lifetimeTrace)));
  assert.equal(lifetimeStopped.state, p.TraceState.TRACE_STATE_READY);
  collecting = false;
  cases.push(await inspect({trace: lifetimeTrace, stopped: lifetimeStopped, identities: lifetimeIds}, 'memory-lifetime'));
  ok(await inference.releaseRuntime(new p.RuntimeRef(runtime)));
  cases.push(await inspect(warm, 'warm-after-retirement'));
  if (backend === 'webgpu') { assert.equal(devices, 1); assert.ok(submissions >= executions); }
  assert.deepEqual(errors, []);
  const report = {backend, profile, nodeCount, adapterInfo, executionsChecked: executions, submissions, devices, timingCreates,
    bundleSha256: sha256(await readFile(bundle)), wasmSha256: sha256(await readFile(wasm)), cases};
  await writeFile(new URL(`${backend}-${profile}.json`, output), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
} finally {
  await host.close();
  globalThis.fetch = fetchOriginal;
  if (gpu) gpu.requestAdapter = requestAdapter;
}
