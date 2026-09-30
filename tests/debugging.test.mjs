import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxBufferServiceClient, VxDebugServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {EngineDebugSession} from '../examples/common/EngineDebugSession.js';

// Node debugging is a full-profile service.
const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const state = p.DebugState, status = p.DebugTensorStatus, reason = p.DebugStopReason;
const invalid = p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT;
function graph(width = 4, depth = 5) {
  const nodes = [];
  for (let i = 0; i < depth; i++) {
    const input = i ? `t${i - 1}` : 'x';
    nodes.push({id: `node${i}`, opType: i % 2 ? 'Add' : 'ReLU',
      inputs: i % 2 ? {a: input, b: input} : {input},
      outputs: {out: {tensor: `t${i}`, dtype: 'float32', shape: [width]}}, params: {}});
  }
  return {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [width]}},
    nodes, outputs: [`t${depth - 1}`]};
}
async function setup(t, document = graph(), options = {}) {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const model = await f.load(document, options.weights);
  const compiled = ok(await f.inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
    preserveNodeBoundaries: options.preserve ?? false})));
  return {...f, debug, model, compiled};
}
function inputs(data = Float32Array.of(-1, 2, -3, 4)) { return tensors({x: {data, shape: [data.length]}}); }
const ref = info => new p.DebugSessionRef({debugSessionId: info.debugSessionId});
async function create(f, capture = {values: true}, limits, data) {
  return ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({
    compiledModelId: f.compiled.compiledModelId, inputs: inputs(data)}), capture: new p.DebugCapture(capture), limits: limits && new p.DebugLimits(limits)})));
}
async function plan(f, info) { return ok(await f.debug.getDebugPlan(ref(info))).plan; }
/** Every published event (AIP-158 pages), and the snapshots they contain. */
async function events(f, info, pageSize = 2) {
  const all = []; let pageToken = '';
  do {
    const page = ok(await f.debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId, pageSize, pageToken})));
    assert.equal(page.totalSize, info.events.count);
    all.push(...page.events); pageToken = page.nextPageToken;
  } while (pageToken);
  return all;
}
async function read(f, info) { return (await events(f, info)).flatMap(event => event.snapshots); }
async function run(f, info, breakBeforeNodes = [], breakOnNonfinite = false) {
  return ok(await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision, breakBeforeNodes, breakOnNonfinite})));
}
async function step(f, info) {
  return ok(await f.debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision})));
}
async function values(f, info, snapshot, readLimit = 3) {
  const parts = []; let readOffset = 0n, size = 1n;
  while (readOffset < size) {
    const request = new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: snapshot.snapshotId, readOffset, readLimit});
    const chunk = ok(await f.debug.readDebugTensor(request));
    assert.equal(chunk.status, status.DEBUG_TENSOR_STATUS_AVAILABLE);
    const repeat = ok(await f.debug.readDebugTensor(request));
    assert.deepEqual(repeat.data, chunk.data);
    parts.push(Buffer.from(chunk.data)); readOffset += BigInt(chunk.data.length); size = chunk.sizeBytes;
  }
  const bytes = Buffer.concat(parts);
  assert.equal(BigInt(bytes.length), snapshot.logicalBytes);
  return [...new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length))];
}

test('Step executes one node, immutable snapshots survive arena reuse and parent handle release', async t => {
  const f = await setup(t);
  const input = Float32Array.of(-1, 2, -3, 4);
  let info = await create(f, {values: true}, undefined, input);
  input.fill(99);
  assert.equal(info.state, state.DEBUG_STATE_PAUSED); assert.equal(info.nextStep, 0);
  assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_ENTRY);
  assert.equal(info.events.count, 0n);
  const layout = await plan(f, info);
  assert.equal(layout.steps.length, 5);
  assert.deepEqual(layout.steps[2].sourceNodeIds, ['node2']);
  assert.ok(layout.lineage.executionId > 0n);
  for (let i = 0; i < 3; i++) {
    const same = ok(await f.debug.getDebugSession(ref(info)));
    assert.equal(same.revision, info.revision); assert.equal(same.events.count, 0n);
  }
  info = await step(f, info);
  assert.equal(info.nextStep, 1); assert.equal(info.events.count, 2n);
  assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_STEP);
  const early = await read(f, info);
  assert.deepEqual((await events(f, info)).map(event => event.point), [p.DebugPoint.DEBUG_POINT_BEFORE, p.DebugPoint.DEBUG_POINT_AFTER]);
  assert.deepEqual(await values(f, info, early[0]), [-1, 2, -3, 4]);
  assert.deepEqual(await values(f, info, early[1]), [0, 2, 0, 4]);
  const ordinary = ok(await f.inference.run(new p.RunRequest({...f.compiled, inputs: inputs()})));
  const reference = [...await f.read(ordinary, 't4')];
  await f.inference.releaseResult(new p.ResultRef(ordinary));
  await f.inference.releaseCompiledModel(new p.CompiledModelRef(f.compiled));
  await f.inference.releaseModel(new p.ModelRef(f.model));
  info = await run(f, info);
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED); assert.equal(info.nextStep, 5);
  assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_UNSPECIFIED);
  assert.equal(info.captureComplete, true);
  assert.deepEqual(await values(f, info, early[1]), [0, 2, 0, 4]);
  const snapshots = await read(f, info);
  assert.deepEqual(await values(f, info, snapshots.at(-1)), reference);
  assert.deepEqual(reference, [0, 8, 0, 16]);
  assert.ok(info.peakBytes <= info.maxBytes);
});

test('revision conflicts, busy and unknown breakpoints do not execute; Continue resumes a breakpoint once', async t => {
  const f = await setup(t); let info = await create(f);
  const missing = await f.debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId}));
  assert.equal(missing.report.status, invalid);
  const stale = await f.debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: 9n}));
  assert.equal(stale.report.status, p.NativeStatus.NATIVE_STATUS_REVISION_CONFLICT);
  assert.equal(stale.report.code, p.OperationCode.OPERATION_CODE_REVISION_CONFLICT);
  const unknown = await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId,
    expectedRevision: info.revision, breakBeforeNodes: ['missing']}));
  assert.equal(unknown.report.status, invalid);
  assert.equal((await f.debug.getDebugSession(ref(info))).nextStep, 0);
  const revision = info.revision;
  info = await run(f, info, ['node2']);
  assert.equal(info.state, state.DEBUG_STATE_PAUSED); assert.equal(info.nextStep, 2);
  assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_BREAKPOINT);
  const old = await f.debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: revision}));
  assert.equal(old.report.status, p.NativeStatus.NATIVE_STATUS_REVISION_CONFLICT);
  info = await run(f, info, ['node2', 'node4']);
  assert.equal(info.nextStep, 4); assert.equal(info.state, state.DEBUG_STATE_PAUSED);
  info = await run(f, info, ['node4']);
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
  const finished = await step(f, info).catch(error => error);
  const again = await f.debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: info.revision}));
  assert.ok(finished instanceof Error || finished.report);
  assert.equal(again.report.code, p.OperationCode.OPERATION_CODE_DEBUG_SESSION_FINISHED);
});

test('break_on_nonfinite pauses after the first step that produces NaN or infinity', async t => {
  const document = graph(2, 3);
  document.nodes[1] = {id: 'node1', opType: 'Div', inputs: {a: 't0', b: 't0'},
    outputs: {out: {tensor: 't1', dtype: 'float32', shape: [2]}}, params: {}};
  const f = await setup(t, document, {preserve: true});
  let info = await create(f, {}, undefined, Float32Array.of(0, 2));
  info = await run(f, info, [], true);
  assert.equal(info.state, state.DEBUG_STATE_PAUSED);
  assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_NONFINITE);
  assert.equal(info.nextStep, 2);
  const last = (await events(f, info)).at(-1);
  assert.equal(last.step, 1); assert.equal(last.point, p.DebugPoint.DEBUG_POINT_AFTER);
  assert.equal(last.snapshots[0].statistics.nanCount, 1n);
  info = await run(f, info);
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
});

test('default capture keeps finite statistics, nonfinite counts and no raw bytes', async t => {
  const f = await setup(t, graph(6, 1));
  let info = await create(f, {tensorNames: ['x']}, undefined, Float32Array.of(-1, 2, NaN, Infinity, -Infinity, 5));
  info = await run(f, info);
  const snapshots = await read(f, info);
  assert.equal(snapshots.length, 1);
  assert.equal(snapshots[0].status, status.DEBUG_TENSOR_STATUS_STATISTICS_ONLY);
  const stats = snapshots[0].statistics;
  assert.equal(stats.finiteCount, 3n); assert.equal(stats.nanCount, 1n);
  assert.equal(stats.positiveInfinityCount, 1n); assert.equal(stats.negativeInfinityCount, 1n);
  assert.equal(stats.minimum, -1); assert.equal(stats.maximum, 5); assert.equal(stats.mean, 2); assert.equal(stats.variance, 6);
  const chunk = ok(await f.debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 1n})));
  assert.equal(chunk.status, status.DEBUG_TENSOR_STATUS_STATISTICS_ONLY); assert.equal(chunk.data?.length ?? 0, 0);
});

test('byte and record limits are enforced while numerical execution completes', async t => {
  const width = 32768, f = await setup(t, graph(width, 3));
  let info = await create(f, {values: true}, {maxBytes: 65536n}, new Float32Array(width).fill(2));
  info = await run(f, info);
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED); assert.equal(info.captureComplete, false);
  assert.ok(info.snapshots.dropped > 0n); assert.ok(info.peakBytes <= 65536n);
  assert.ok((await read(f, info)).every(tensor => tensor.status === status.DEBUG_TENSOR_STATUS_BUDGET_EXCEEDED));
  const small = await setup(t);
  let bounded = await create(small, {values: true}, {maxEvents: 1, maxSnapshots: 1});
  bounded = await run(small, bounded);
  assert.equal(bounded.events.count, 1n); assert.equal(bounded.snapshots.count, 1n);
  assert.ok(bounded.events.dropped > 0n); assert.ok(bounded.snapshots.dropped > 0n);
});

test('selection, pagination and tensor read bounds fail closed', async t => {
  const f = await setup(t);
  for (const capture of [{nodeIds: ['missing']}, {tensorNames: ['missing']}, {tensorNames: ['x\0bad']}]) {
    const bad = await f.debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({compiledModelId: f.compiled.compiledModelId, inputs: inputs()}), capture: new p.DebugCapture(capture)}));
    assert.equal(bad.report.status, invalid); assert.equal(bad.debugSessionId, 0n);
    assert.equal(bad.state, state.DEBUG_STATE_UNSPECIFIED);
  }
  let info = await create(f, {values: true, nodeIds: ['node1'], inputs: false, tensorNames: ['t1']});
  info = await run(f, info);
  assert.equal(info.snapshots.count, 1n); assert.equal(info.events.count, 2n);
  for (const args of [{pageSize: 0}, {pageSize: 4097}, {pageToken: '99'}, {pageToken: 'x'}, {pageToken: '01'}]) {
    const bad = await f.debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId, ...args}));
    assert.equal(bad.report.status, invalid, JSON.stringify(args));
  }
  const unknown = await f.debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 2n}));
  assert.equal(unknown.report.status, p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  for (const args of [{readLimit: 0}, {readOffset: 17n}]) {
    const bad = await f.debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 1n, ...args}));
    assert.equal(bad.report.status, invalid);
  }
});

test('cancel preserves published snapshots and release is idempotent', async t => {
  const f = await setup(t); let info = await create(f);
  info = await step(f, info);
  info = ok(await f.debug.cancelDebugSession(ref(info)));
  assert.equal(info.state, state.DEBUG_STATE_CANCELLED); assert.equal(info.nextStep, 1);
  assert.deepEqual(await values(f, info, (await read(f, info))[1]), [0, 2, 0, 4]);
  const failed = await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: info.revision}));
  assert.equal(failed.report.code, p.OperationCode.OPERATION_CODE_DEBUG_SESSION_FINISHED);
  ok(await f.debug.releaseDebugSession(ref(info)));
  ok(await f.debug.releaseDebugSession(ref(info)));
  const disposed = await f.debug.getDebugSession(ref(info));
  assert.equal(disposed.report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
});

test('source-preserving compilation exposes alias nodes; optimized plans mark missing observations', async t => {
  const document = graph(4, 1);
  document.nodes[0].opType = 'Reshape'; document.nodes[0].params = {shape: [4]};
  for (const preserve of [false, true]) {
    const f = await setup(t, document, {preserve});
    let info = await create(f); const layout = await plan(f, info);
    assert.equal(info.capabilities.preservesNodeBoundaries, preserve);
    assert.equal(layout.steps[0].skipped, !preserve);
    info = await run(f, info);
    const snapshots = await read(f, info);
    if (preserve) assert.deepEqual(await values(f, info, snapshots.at(-1)), [-1, 2, -3, 4]);
    else assert.ok(snapshots.every(tensor => tensor.status === status.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT));
  }
});

test('quantized raw values retain their exact per-tensor affine descriptor', async t => {
  const document = {format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {dtype: 'int8', shape: [4]}}, nodes: [{id: 'view', opType: 'Reshape', inputs: {input: 'x'},
      outputs: {out: {tensor: 'y', dtype: 'int8', shape: [4]}}, params: {shape: [4]}}], outputs: ['y'],
    quantization: {format: 'volvox-affine-safetensors/v1', tensors: {x: {scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero'},
      y: {scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero'}}}};
  const f = await setup(t, document, {preserve: true, weights: safetensors([
    {name: 'scale', shape: [1], data: Float32Array.of(.25)}, {name: 'zero', shape: [1], dtype: 'I8', data: Int8Array.of(-2)}])});
  let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({capture: new p.DebugCapture({values: true}), forward: new p.DebugForward({compiledModelId: f.compiled.compiledModelId,
    inputs: [new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_I8, shape: [4n], inline: new Uint8Array(Int8Array.of(-4, -2, 2, 6).buffer)})]})})));
  info = await run(f, info);
  for (const tensor of await read(f, info)) {
    assert.equal(tensor.dtype, p.DataType.DATA_TYPE_I8);
    assert.equal(tensor.quantization.perTensor.scale, .25); assert.equal(tensor.quantization.perTensor.zeroPoint, -2);
    assert.equal(tensor.statistics.mean, .5);
    // real = 0.25 * (q + 2): [-0.5, 0, 1, 2]
    assert.equal(tensor.realStatistics.mean, .625); assert.equal(tensor.realStatistics.minimum, -.5);
    const chunk = ok(await f.debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: tensor.snapshotId})));
    assert.deepEqual([...new Int8Array(chunk.data.buffer, chunk.data.byteOffset, chunk.data.length)], [-4, -2, 2, 6]);
  }
});

test('quantized weights preserve per-axis parameters and I32 bias statistics', async t => {
  const names = ['x', 'weight', 'y'];
  const quantization = {format: 'volvox-affine-safetensors/v1', tensors: Object.fromEntries(names.map(name => [name,
    {scheme: name === 'weight' ? 'per_axis' : 'per_tensor', ...(name === 'weight' ? {axis: 0} : {}),
      scale_tensor: `${name}.scale`, zero_point_tensor: `${name}.zero`}])),
  };
  const document = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'int8', shape: [1, 2]}},
    nodes: [{id: 'quantized', opType: 'QLinear', inputs: {input: 'x', weight: 'weight', bias: 'bias'},
      outputs: {out: {tensor: 'y', dtype: 'int8', shape: [1, 2]}}, params: {}}], outputs: ['y'], quantization};
  const weights = [
    {name: 'weight', shape: [2, 2], dtype: 'I8', data: Int8Array.of(1, 2, -3, 4)},
    {name: 'bias', shape: [2], dtype: 'I32', data: Int32Array.of(3, -5)},
    {name: 'x.scale', shape: [1], data: Float32Array.of(.125)},
    {name: 'x.zero', shape: [1], dtype: 'I8', data: Int8Array.of(-2)},
    {name: 'weight.scale', shape: [2], data: Float32Array.of(.25, .5)},
    {name: 'weight.zero', shape: [2], dtype: 'I8', data: Int8Array.of(0, 1)},
    {name: 'y.scale', shape: [1], data: Float32Array.of(.25)},
    {name: 'y.zero', shape: [1], dtype: 'I8', data: Int8Array.of(-3)},
  ];
  const f = await setup(t, document, {weights: safetensors(weights), preserve: true});
  let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({compiledModelId: f.compiled.compiledModelId,
    inputs: [new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_I8, shape: [1n, 2n], inline: new Uint8Array(Int8Array.of(-2, 6).buffer)})]})})));
  info = await run(f, info);
  const layout = await plan(f, info);
  const snapshots = await read(f, info);
  const weight = snapshots.find(tensor => layout.tensors[tensor.tensorId].name === 'weight');
  assert.equal(weight.quantization.perAxis.axis, 0);
  assert.deepEqual(weight.quantization.perAxis.scales, [.25, .5]);
  assert.deepEqual(weight.quantization.perAxis.zeroPoints, [0, 1]);
  assert.equal(weight.statistics.mean, 1);
  // Per-axis real values: rows [0.25, 0.5] and [-2, 1.5].
  assert.equal(weight.realStatistics.mean, .0625);
  const bias = snapshots.find(tensor => layout.tensors[tensor.tensorId].name === 'bias');
  assert.equal(bias.dtype, p.DataType.DATA_TYPE_I32);
  assert.equal(bias.statistics.minimum, -5); assert.equal(bias.statistics.maximum, 3);
  assert.equal(bias.statistics.mean, -1); assert.equal(bias.statistics.variance, 16);
});

test('fused Conv+Clip captures its real output; removed intermediate values are never fabricated', async t => {
  const shape = [1, 1, 4, 1];
  const document = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape}},
    nodes: [{id: 'conv', opType: 'Conv2D', inputs: {input: 'x', weight: 'w'},
      outputs: {out: {tensor: 'middle', dtype: 'float32', shape}},
      params: {data_layout: 'NHWC', weight_layout: 'HWIO', stride: [1, 1], dilation: [1, 1], pads: [0, 0, 0, 0], groups: 1}},
    {id: 'clip', opType: 'Clip', inputs: {input: 'middle'}, outputs: {out: {tensor: 'y', dtype: 'float32', shape}}, params: {min: 0, max: 6}}],
    outputs: ['y']};
  for (const preserve of [false, true]) {
    const f = await setup(t, document, {preserve, weights: safetensors([{name: 'w', shape: [1, 1, 1, 1], data: Float32Array.of(2)}])});
    let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({capture: new p.DebugCapture({values: true}),
      forward: new p.DebugForward({compiledModelId: f.compiled.compiledModelId, inputs: tensors({x: {data: Float32Array.of(-1, 2, 7, 4), shape}})})})));
    const layout = await plan(f, info);
    assert.equal(layout.steps[0].fused, !preserve);
    if (!preserve) assert.deepEqual(layout.steps[0].sourceNodeIds, ['conv', 'clip']);
    assert.equal(layout.tensors[layout.steps[0].outputs[0]].name, preserve ? 'middle' : 'y');
    info = await step(f, info);
    let snapshots = await read(f, info);
    assert.deepEqual(await values(f, info, snapshots.at(-1)), preserve ? [-2, 4, 14, 8] : [0, 4, 6, 6]);
    info = await run(f, info); snapshots = await read(f, info);
    if (preserve) assert.deepEqual(await values(f, info, snapshots.at(-1)), [0, 4, 6, 6]);
    else for (const snapshot of snapshots.filter(tensor => layout.tensors[tensor.tensorId].name === 'middle'))
      assert.equal(snapshot.status, status.DEBUG_TENSOR_STATUS_OPTIMIZED_OUT);
  }
});

test('KV cache reads preserve the selected slot through generated messages and pagination', async () => {
  const tokenCount = 4098, rowWidth = 2;
  for (const selectedSlot of [undefined, 2]) {
    const slot = selectedSlot ?? 0, requests = [];
    const client = {async readDebugKVCache(request) {
      const decoded = p.ReadDebugKVCacheRequest.fromBinary(request.toBinary());
      requests.push({sessionId: decoded.debugSessionId, cacheId: decoded.cacheId, slot: decoded.slot,
        offset: decoded.tokenOffset, limit: decoded.tokenLimit});
      const rows = Math.min(decoded.tokenLimit, tokenCount - decoded.tokenOffset);
      const data = Uint8Array.from({length: rows * rowWidth}, (_, i) =>
        (decoded.slot * 53 + decoded.tokenOffset * rowWidth + i) % 256);
      return new p.ReadDebugKVCacheResponse({report: new p.OperationReport(), dtype: p.DataType.DATA_TYPE_U8,
        shape: [BigInt(rows), BigInt(rowWidth)], data, tokenCount});
    }};
    const session = new EngineDebugSession(client, p, new p.DebugSessionInfo({debugSessionId: 7n}));
    const result = await session.readKVCache(3, selectedSlot);
    assert.deepEqual(requests, [0, 4096].map(offset => ({sessionId: 7n, cacheId: 3, slot, offset, limit: 4096})));
    assert.equal(result.dtype, p.DataType.DATA_TYPE_U8);
    assert.deepEqual(result.shape, [BigInt(tokenCount), BigInt(rowWidth)]);
    assert.deepEqual(result.data, Uint8Array.from({length: tokenCount * rowWidth}, (_, i) => (slot * 53 + i) % 256));
  }
});

test('an exported snapshot is an ordinary retained buffer that outlives its session', async t => {
  const f = await setup(t);
  let info = await create(f);
  info = await step(f, info);
  const output = (await read(f, info))[1];
  const exported = ok(await f.debug.exportDebugTensor(new p.ExportDebugTensorRequest({
    debugSessionId: info.debugSessionId, snapshotId: output.snapshotId})));
  const [tensor] = exported.outputs;
  assert.equal(tensor.name, 't0'); assert.deepEqual(tensor.shape, [4n]);
  assert.equal(tensor.buffer.lengthBytes, 16n);
  ok(await f.debug.releaseDebugSession(ref(info)));
  const buffers = new VxBufferServiceClient(reportTransport(f.host));
  const copied = ok(await buffers.copyTensors(new p.CopyTensorsRequest({sources: [tensor], inlineResult: true})));
  assert.deepEqual([...new Float32Array(copied.outputs[0].inline.slice().buffer)], [0, 2, 0, 4]);
  ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: [tensor.buffer.bufferId]})));
  const refused = await f.debug.exportDebugTensor(new p.ExportDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 1n}));
  assert.equal(refused.report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
});
