import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxDebugServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

// A decode target steps the context's real incremental pass; its KV caches are
// read live at every stop. Debugging is a full-profile service.
const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const B = 2, S = 8, D = 8, LAYERS = 2;
const state = p.DebugState, reason = p.DebugStopReason, busy = p.NativeStatus.NATIVE_STATUS_BUSY;

function decoder() {
  const nodes = [], weights = [];
  let input = 'x';
  for (let layer = 0; layer < LAYERS; layer++) {
    for (const [index, port] of ['q', 'k', 'v'].entries()) {
      const id = `${port}${layer}`;
      weights.push({name: `${id}.weight`, shape: [D, D],
        data: Float32Array.from({length: D * D}, (_, i) => i % D === Math.floor(i / D) ? .8 : Math.sin(i + index + layer) * .05)});
      weights.push({name: `${id}.bias`, shape: [D], data: new Float32Array(D)});
      nodes.push({id, opType: 'Linear', inputs: {input, weight: `${id}.weight`, bias: `${id}.bias`},
        outputs: {out: {tensor: id, shape: [B, S, D], dtype: 'float32'}}, params: {}});
    }
    const id = layer === LAYERS - 1 ? 'out' : `attention${layer}`;
    nodes.push({id, opType: 'CrossSDPA', inputs: {q: `q${layer}`, k: `k${layer}`, v: `v${layer}`, mask: 'keep'},
      outputs: {out: {tensor: id, shape: [B, S, D], dtype: 'float32'}}, params: {heads: 2, causal: true}});
    input = id;
  }
  return {graph: {format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {shape: [B, S, D], dtype: 'float32'}, keep: {shape: [B, S], dtype: 'int32'}},
    nodes, outputs: ['out']}, weights: safetensors(weights)};
}

async function setup(t) {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const {graph, weights} = decoder();
  f.debug = new VxDebugServiceClient(reportTransport(f.host));
  f.compiled = await f.compile(await f.load(graph, weights));
  f.context = async () => ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest({
    compiledModelId: f.compiled.compiledModelId, decodeRowMode: p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
    decodeSlots: B, decodeInputs: ['x', 'keep'], requireIncremental: true}))).contextId;
  return f;
}

const lengths0 = [4, 2];
function feed(lengths, seed = 0) {
  const x = Float32Array.from({length: B * S * D}, (_, i) => Math.sin(i * .71 + seed));
  const keep = Int32Array.from({length: B * S}, (_, i) => i % S < lengths[Math.floor(i / S)] ? 1 : 0);
  return {x, keep};
}
const bind = ({x, keep}) => tensors({x: {data: x.slice(), shape: [B, S, D]}, keep: {data: keep.slice(), shape: [B, S]}});
const cursor = positions => new p.DecodeSlotActions({slots: positions.map(position => new p.DecodeSlotAction(
  position === null ? {empty: true} : {position}))});
const ref = info => new p.DebugSessionRef({debugSessionId: info.debugSessionId});

async function prefill(f, contextId, values) {
  const result = ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({contextId, inputs: bind(values),
    slotPositions: new p.DecodeSlotPositions({positions: lengths0.map(length => length - 1)})})));
  ok(await f.inference.releaseResult(new p.ResultRef(result)));
}

async function kv(f, info, cacheId, slot) {
  const rows = ok(await f.debug.readDebugKVCache(new p.ReadDebugKVCacheRequest({
    debugSessionId: info.debugSessionId, cacheId, slot, tokenLimit: 3})));
  let data = rows.data, offset = Number(rows.shape[0]);
  while (offset < rows.tokenCount) { // paged in chunks of three rows
    const next = ok(await f.debug.readDebugKVCache(new p.ReadDebugKVCacheRequest({
      debugSessionId: info.debugSessionId, cacheId, slot, tokenOffset: offset, tokenLimit: 3})));
    data = Uint8Array.of(...data, ...next.data); offset += Number(next.shape[0]);
  }
  assert.equal(rows.dtype, p.DataType.DATA_TYPE_F32);
  assert.equal(offset, rows.tokenCount);
  return new Float32Array(data.slice().buffer);
}

async function stepTo(f, info, predicate) {
  while (info.state === state.DEBUG_STATE_PAUSED && !(await predicate(info)))
    info = ok(await f.debug.stepDebugSession(new p.StepDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
  return info;
}

test('a debugged DecodeStep equals the ordinary step and exposes its KV writes at every stop', async t => {
  const f = await setup(t);
  const [debugged, reference] = [await f.context(), await f.context()];
  const values = feed(lengths0);
  await prefill(f, debugged, values); await prefill(f, reference, values);
  // The step appends row 4 to slot 0 and row 2 to slot 1 with new inputs.
  const next = feed([5, 3], 1);
  const request = contextId => new p.DecodeStepRequest({contextId, slotActions: cursor([4, 2]), inputs: bind(next)});
  const expected = ok(await f.inference.decodeStep(request(reference)));

  let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: request(debugged),
    capture: new p.DebugCapture({values: true})})));
  assert.equal(info.target, p.DebugTarget.DEBUG_TARGET_DECODE_STEP);
  assert.equal(info.state, state.DEBUG_STATE_PAUSED); assert.equal(info.stopReason, reason.DEBUG_STOP_REASON_ENTRY);
  assert.equal(info.report.lineage.contextId, BigInt(debugged));

  // The attached context refuses every other operation until release.
  const refused = await f.inference.decodeStep(request(debugged));
  assert.equal(refused.report.status, busy);
  assert.equal((await f.inference.getDecodeState(new p.ExecutionContextRef({contextId: debugged}))).report.status, busy);

  let decode = ok(await f.debug.getDebugDecodeState(ref(info)));
  assert.deepEqual(decode.slots.map(slot => [slot.lengthBefore, slot.lengthAfter, slot.writeBegin, slot.writeEnd]),
    [[4, 5, 4, 5], [2, 3, 2, 3]]);
  assert.equal(decode.caches.length, 2 * LAYERS);
  const plan = ok(await f.debug.getDebugPlan(ref(info))).plan;
  const name = cache => plan.tensors[cache.tensorId].name;
  assert.deepEqual(decode.caches.map(name), ['k0', 'v0', 'k1', 'v1']);
  const {DEBUG_KV_ROLE_KEY: key, DEBUG_KV_ROLE_VALUE: value} = p.DebugKVRole;
  assert.deepEqual(decode.caches.map(cache => cache.role), [key, value, key, value]);
  for (const cache of decode.caches) {
    assert.equal(cache.tokenCapacity, S); assert.deepEqual(cache.tokenShape, [BigInt(D)]);
    assert.equal(plan.steps[cache.writerStep].sourceNodeIds[0], name(cache));
    assert.equal(cache.written, false);
  }
  const k0 = decode.caches[0];
  const before = await kv(f, info, k0.cacheId, 0);
  assert.equal(before.length, 5 * D);

  // Stop right after the step that writes k0: row 4 changed, rows 0..3 did not.
  info = await stepTo(f, info, async current => current.nextStep > k0.writerStep);
  decode = ok(await f.debug.getDebugDecodeState(ref(info)));
  assert.equal(decode.caches[0].written, true); assert.equal(decode.caches[2].written, false);
  const after = await kv(f, info, k0.cacheId, 0);
  assert.deepEqual(after.subarray(0, 4 * D), before.subarray(0, 4 * D));
  assert.notDeepEqual(after.subarray(4 * D), before.subarray(4 * D));

  // Run to completion; the result equals the ordinary DecodeStep bit for bit.
  info = ok(await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
  assert.ok(info.resultId > 0n);
  assert.deepEqual(await f.read({resultId: info.resultId}, 'out'), await f.read(expected, 'out'));
  // Only the changed closure ran: every executed step appears as an event pair.
  const events = ok(await f.debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId, pageSize: 4096}))).events;
  assert.equal(events.length % 2, 0);
  // The completed cache stays readable until release, and equals the reference's.
  decode = ok(await f.debug.getDebugDecodeState(ref(info)));
  assert.ok(decode.caches.every(cache => cache.written));
  const final = await kv(f, info, decode.caches[3].cacheId, 1);
  ok(await f.debug.releaseDebugSession(ref(info)));

  // Released: the context advanced exactly like the reference and runs again.
  const state0 = ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId: debugged})));
  const state1 = ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId: reference})));
  assert.deepEqual(state0.activeLengths, state1.activeLengths);
  const third = feed([6, 4], 2);
  const again = contextId => new p.DecodeStepRequest({contextId, slotActions: cursor([5, 3]), inputs: bind(third)});
  assert.deepEqual(await f.read(ok(await f.inference.decodeStep(again(debugged))), 'out'),
    await f.read(ok(await f.inference.decodeStep(again(reference))), 'out'));

  // A second session on the reference sees the same completed v1 cache.
  info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: new p.DecodeStepRequest({
    contextId: reference, slotActions: cursor([null, 4]), inputs: bind(third)})})));
  const empty = ok(await f.debug.getDebugDecodeState(ref(info))).slots[0];
  assert.equal(empty.empty, true); assert.equal(empty.writeBegin, empty.writeEnd);
  ok(await f.debug.releaseDebugSession(ref(info)));
  assert.equal(final.length, 3 * D);
});

test('a debugged DecodePrefill writes every prompt row; cancel resets the context like a failed step', async t => {
  const f = await setup(t);
  const contextId = await f.context();
  const values = feed(lengths0);
  let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodePrefill: new p.DecodePrefillRequest({
    contextId, inputs: bind(values), slotPositions: new p.DecodeSlotPositions({positions: lengths0.map(length => length - 1)})})})));
  assert.equal(info.target, p.DebugTarget.DEBUG_TARGET_DECODE_PREFILL);
  let decode = ok(await f.debug.getDebugDecodeState(ref(info)));
  assert.deepEqual(decode.slots.map(slot => [slot.lengthBefore, slot.lengthAfter, slot.writeBegin, slot.writeEnd]),
    [[0, 4, 0, 4], [0, 2, 0, 2]]);
  info = ok(await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
  decode = ok(await f.debug.getDebugDecodeState(ref(info)));
  assert.equal((await kv(f, info, decode.caches[1].cacheId, 0)).length, 4 * D);
  ok(await f.inference.releaseResult(new p.ResultRef({resultId: info.resultId})));
  ok(await f.debug.releaseDebugSession(ref(info)));
  assert.equal(ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId}))).prefilled, true);

  // Cancelling a partially executed step leaves no half-written cache behind.
  info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: new p.DecodeStepRequest({
    contextId, slotActions: cursor([4, 2]), inputs: bind(feed([5, 3], 1))})})));
  info = ok(await f.debug.stepDebugSession(new p.StepDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
  info = ok(await f.debug.cancelDebugSession(ref(info)));
  assert.equal(info.state, state.DEBUG_STATE_CANCELLED); assert.equal(info.resultId, 0n);
  const read = await f.debug.readDebugKVCache(new p.ReadDebugKVCacheRequest({debugSessionId: info.debugSessionId, cacheId: 1}));
  assert.equal(read.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  ok(await f.debug.releaseDebugSession(ref(info)));
  assert.equal(ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId}))).prefilled, false);
});

test('decode debug metadata budget failures report OUT_OF_MEMORY and release the context', async t => {
  const f = await setup(t);
  const contextId = await f.context();
  const request = () => new p.DecodePrefillRequest({contextId, inputs: bind(feed(lengths0)),
    slotPositions: new p.DecodeSlotPositions({positions: lengths0.map(length => length - 1)})});
  // Cover the session, slot and KV metadata admission boundaries without
  // tying the assertion to the size of any private C struct.
  for (let maxBytes = 4096n; maxBytes <= 8192n; maxBytes += 32n) {
    const info = await f.debug.createDebugSession(new p.CreateDebugSessionRequest({
      decodePrefill: request(), limits: new p.DebugLimits({maxBytes})}));
    assert.equal(info.report.status, p.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY, `budget ${maxBytes}`);
    assert.equal(info.report.code, p.OperationCode.OPERATION_CODE_OUT_OF_MEMORY, `budget ${maxBytes}`);
    assert.ok(info.report.message.length > 0);
    assert.equal(info.debugSessionId, 0n);
    assert.equal(info.state, state.DEBUG_STATE_UNSPECIFIED);
    ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId})));
  }
  // The same context can create and complete a session after every failure.
  let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodePrefill: request()})));
  info = ok(await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
  assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
  ok(await f.inference.releaseResult(new p.ResultRef({resultId: info.resultId})));
  ok(await f.debug.releaseDebugSession(ref(info)));
  assert.equal(ok(await f.inference.getDecodeState(new p.ExecutionContextRef({contextId}))).prefilled, true);
});

test('paged KV reads gather through the page table and match a contiguous cache', async t => {
  const f = await setup(t);
  const [paged, dense] = [await f.context(), await f.context()];
  const values = feed(lengths0);
  await prefill(f, paged, values); await prefill(f, dense, values);
  ok(await f.inference.configureDecodeCache(new p.ConfigureDecodeCacheRequest({contextId: paged,
    tensors: ['k0', 'v0', 'k1', 'v1'], pageTokens: 2, maxPages: 8})));
  const next = feed([5, 3], 1);
  const sessions = [];
  for (const contextId of [paged, dense]) {
    let info = ok(await f.debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: new p.DecodeStepRequest({
      contextId, slotActions: cursor([4, 2]), inputs: bind(next)})})));
    info = ok(await f.debug.continueDebugSession(new p.ContinueDebugSessionRequest({...ref(info), expectedRevision: info.revision})));
    assert.equal(info.state, state.DEBUG_STATE_COMPLETED);
    sessions.push(info);
  }
  const [pagedState, denseState] = await Promise.all(sessions.map(info => f.debug.getDebugDecodeState(ref(info))));
  assert.ok(pagedState.caches.every(cache => cache.pageTokens === 2));
  assert.ok(denseState.caches.every(cache => cache.pageTokens === 0));
  for (let index = 0; index < pagedState.caches.length; index++)
    for (let slot = 0; slot < B; slot++)
      assert.deepEqual(await kv(f, sessions[0], pagedState.caches[index].cacheId, slot),
        await kv(f, sessions[1], denseState.caches[index].cacheId, slot));
  for (const info of sessions) {
    ok(await f.inference.releaseResult(new p.ResultRef({resultId: info.resultId})));
    ok(await f.debug.releaseDebugSession(ref(info)));
  }
});
