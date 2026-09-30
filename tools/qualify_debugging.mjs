/** Release WASM and physical WebGPU qualification; Node or Deno. */
import { readPackageVersion } from './release_version.mjs';
import assert from 'node:assert/strict';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {pathToFileURL, fileURLToPath} from 'node:url';
import path from 'node:path';
// Node debugging is a full-profile service; the lite entry does not carry it.
const [backend = 'wasm', output = 'build/debugging'] = process.argv.slice(2);
const profile = 'full';
const root = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
const releaseVersion = readPackageVersion(root);
const api = await import(pathToFileURL(path.join(root, `dist/${releaseVersion}`, profile === 'full' ? 'volvoxai.js' : 'volvoxai.lite.js')));
const p = api.pb;
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => String(url).startsWith('file:') ? new Response(await readFile(fileURLToPath(url))) : originalFetch(url, init);
const host = new api.FullEngineHost();
const ok = value => { assert.equal((value.report ?? value).status, p.NativeStatus.NATIVE_STATUS_OK, value.report?.message); return value; };
try {
  if (backend === 'webgpu') assert.ok(globalThis.navigator?.gpu, 'WebGPU unavailable');
  const inference = new api.VxInferenceServiceClient(host), debug = new api.VxDebugServiceClient(host);
  const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({cpuThreads: 1})));
  const width = 256, depth = 12;
  const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [1, width]}},
    nodes: Array.from({length: depth}, (_, i) => ({id: `node${i}`, opType: i ? 'Add' : 'ReLU',
      inputs: i ? {a: `t${i - 1}`, b: 'x'} : {input: 'x'},
      outputs: {out: {tensor: `t${i}`, dtype: 'float32', shape: [1, width]}}, params: {}})), outputs: [`t${depth - 1}`]};
  const model = ok(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(graph))})})));
  const reports = [];
  for (const preserveNodeBoundaries of [false, true]) {
    const compiled = ok(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId, preserveNodeBoundaries,
      policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: [backend],
        operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
    for (let iteration = 0; iteration < 10; iteration++) {
      const input = Float32Array.from({length: width}, (_, i) => (i % 17 - 8 + iteration) / 4);
      const inputs = [new p.Tensor({name: 'x', dtype: p.DataType.DATA_TYPE_F32, shape: [1n, BigInt(width)], inline: new Uint8Array(input.buffer)})];
      let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({forward: new p.DebugForward({compiledModelId: compiled.compiledModelId, inputs}), capture: new p.DebugCapture({values: true})})));
      const request = new p.DebugSessionRef(info);
      assert.equal(info.nextStep, 0);
      info = ok(await debug.stepDebugSession(new p.StepDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: info.revision})));
      assert.equal(info.nextStep, 1);
      const early = ok(await debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 2n}))).data;
      assert.deepEqual([...new Float32Array(early.slice().buffer)], [...input].map(x => Math.max(x, 0)));
      const normal = ok(await inference.run(new p.RunRequest({...compiled, inputs})));
      const deadline = performance.now() + 30000;
      while (ok(await inference.getResult(new p.ResultRef(normal))).state === p.ResultState.RESULT_STATE_PENDING) {
        assert.ok(performance.now() < deadline); await new Promise(resolve => setTimeout(resolve, 1));
      }
      const expected = ok(await inference.readOutput(new p.ReadOutputRequest({...normal, name: `t${depth - 1}`}))).tensor.inline;
      ok(await inference.releaseResult(new p.ResultRef(normal)));
      info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: info.revision, breakBeforeNodes: ['node5']})));
      assert.equal(info.nextStep, 5); assert.equal(info.stopReason, p.DebugStopReason.DEBUG_STOP_REASON_BREAKPOINT);
      info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({debugSessionId: info.debugSessionId, expectedRevision: info.revision})));
      assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED); assert.equal(info.captureComplete, true);
      const page = ok(await debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId, pageSize: 4096})));
      const last = page.events.at(-1).snapshots.at(-1);
      const actual = ok(await debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: last.snapshotId}))).data;
      assert.deepEqual(actual, expected);
      assert.deepEqual([...new Float32Array(actual.slice().buffer)], [...input].map(x => Math.max(x, 0) + (depth - 1) * x));
      assert.deepEqual(ok(await debug.readDebugTensor(new p.ReadDebugTensorRequest({debugSessionId: info.debugSessionId, snapshotId: 2n}))).data, early);
      assert.ok(info.peakBytes <= info.maxBytes);
      reports.push({preserveNodeBoundaries, iteration, snapshots: Number(info.snapshots.count), peakBytes: Number(info.peakBytes), outputExact: true});
      ok(await debug.releaseDebugSession(request));
    }
    ok(await inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
  }
  const decode = [];
  for (const quantized of [false, true]) decode.push(await qualifyDecode(inference, debug, runtime, quantized));
  await mkdir(output, {recursive: true});
  await writeFile(path.join(output, `${backend}-${profile}.json`), JSON.stringify({backend, profile, sessions: reports, decode}, null, 2));
  console.log(JSON.stringify({backend, profile, sessions: reports.length, outputExact: true, decode}));
} finally { await host.close(); }

/** A debugged DecodeStep must equal the ordinary one and expose its KV writes. */
async function qualifyDecode(inference, debug, runtime, quantized) {
  const slots = 2, sequence = 8, width = 64, layers = 2, nodes = [], header = {}, blobs = [], quantization = {};
  const dtype = quantized ? 'int8' : 'float32', element = quantized ? 1 : 4;
  let offset = 0, source = 'x';
  const add = (tensor, shape, data) => {
    header[tensor] = {dtype: data instanceof Float32Array ? 'F32' : data instanceof Int8Array ? 'I8' : 'I32', shape,
      data_offsets: [offset, offset + data.byteLength]};
    blobs.push(new Uint8Array(data.buffer)); offset += data.byteLength;
  };
  const affine = (name, perAxis) => {
    const count = perAxis ? width : 1;
    add(`${name}.scale`, [count], new Float32Array(count).fill(.05)); add(`${name}.zero`, [count], new Int8Array(count));
    quantization[name] = {scheme: perAxis ? 'per_axis' : 'per_tensor', ...(perAxis ? {axis: 0} : {}),
      scale_tensor: `${name}.scale`, zero_point_tensor: `${name}.zero`};
  };
  if (quantized) affine('x', false);
  for (let layer = 0; layer < layers; layer++) {
    for (const [index, port] of ['q', 'k', 'v'].entries()) {
      const name = `${port}${layer}`;
      if (quantized) {
        add(`${name}.weight`, [width, width], Int8Array.from({length: width * width}, (_, i) =>
          i % width === Math.floor(i / width) ? 19 : (i + index) % 5 - 2));
        add(`${name}.bias`, [width], new Int32Array(width));
        affine(`${name}.weight`, true); affine(name, false);
      } else {
        add(`${name}.weight`, [width, width], Float32Array.from({length: width * width}, (_, i) =>
          i % width === Math.floor(i / width) ? .8 : ((i * 7 + index + layer) % 11 - 5) * .01));
        add(`${name}.bias`, [width], new Float32Array(width));
      }
      nodes.push({id: name, opType: quantized ? 'QLinear' : 'Linear', inputs: {input: source, weight: `${name}.weight`, bias: `${name}.bias`},
        outputs: {out: {tensor: name, shape: [slots, sequence, width], dtype}}, params: {}});
    }
    const name = layer === layers - 1 ? 'out' : `attention${layer}`;
    nodes.push({id: name, opType: quantized ? 'QSDPA' : 'CrossSDPA', inputs: {q: `q${layer}`, k: `k${layer}`, v: `v${layer}`, mask: 'keep'},
      outputs: {out: {tensor: name, shape: [slots, sequence, width], dtype}}, params: {heads: 2, causal: true}});
    if (quantized) affine(name, false);
    source = name;
  }
  const graph = {format: 'volvox-graph/v1', dimensions: {}, nodes, outputs: ['out'],
    inputs: {x: {shape: [slots, sequence, width], dtype}, keep: {shape: [slots, sequence], dtype: 'int32'}},
    ...(quantized ? {quantization: {format: 'volvox-affine-safetensors/v1', tensors: quantization}} : {})};
  let json = JSON.stringify(header); json += ' '.repeat((8 - json.length % 8) % 8);
  const shard = new Uint8Array(8 + json.length + offset);
  new DataView(shard.buffer).setBigUint64(0, BigInt(json.length), true);
  shard.set(new TextEncoder().encode(json), 8); offset = 8 + json.length;
  for (const blob of blobs) { shard.set(blob, offset); offset += blob.length; }
  const model = ok(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(graph)), weightShards: [shard]})})));
  const compiled = ok(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
    policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: [backend],
      operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
  const contexts = [];
  for (let i = 0; i < 2; i++) contexts.push(ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({
    compiledModelId: compiled.compiledModelId, decodeRowMode: p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,
    decodeSlots: slots, decodeInputs: ['x', 'keep'], requireIncremental: true}))).contextId);
  const lengths = [4, 2];
  const feed = (counts, seed) => [
    new p.Tensor({name: 'x', dtype: quantized ? p.DataType.DATA_TYPE_I8 : p.DataType.DATA_TYPE_F32,
      shape: [BigInt(slots), BigInt(sequence), BigInt(width)],
      inline: new Uint8Array((quantized ? Int8Array : Float32Array).from({length: slots * sequence * width},
        (_, i) => ((i * 13 + seed * 7) % 23 - 11) / (quantized ? 1 : 8)).buffer)}),
    new p.Tensor({name: 'keep', dtype: p.DataType.DATA_TYPE_I32, shape: [BigInt(slots), BigInt(sequence)],
      inline: new Uint8Array(Int32Array.from({length: slots * sequence}, (_, i) => i % sequence < counts[Math.floor(i / sequence)] ? 1 : 0).buffer)})];
  const ready = async result => {
    const deadline = performance.now() + 30000;
    while (ok(await inference.getResult(new p.ResultRef(result))).state === p.ResultState.RESULT_STATE_PENDING) {
      assert.ok(performance.now() < deadline); await new Promise(resolve => setTimeout(resolve, 1));
    }
    const data = ok(await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name: 'out'}))).tensor.inline.slice();
    ok(await inference.releaseResult(new p.ResultRef(result)));
    return data;
  };
  for (const contextId of contexts)
    await ready(ok(await inference.decodePrefill(new p.DecodePrefillRequest({contextId, inputs: feed(lengths, 0),
      slotPositions: new p.DecodeSlotPositions({positions: lengths.map(n => n - 1)})}))));
  const actions = positions => new p.DecodeSlotActions({slots: positions.map(position => new p.DecodeSlotAction({position}))});
  const step = contextId => new p.DecodeStepRequest({contextId, slotActions: actions(lengths), inputs: feed(lengths.map(n => n + 1), 1)});
  const rows = async (info, cacheId, slot = 0) => ok(await debug.readDebugKVCache(new p.ReadDebugKVCacheRequest({
    debugSessionId: info.debugSessionId, cacheId, slot, tokenLimit: 4096}))).data.slice();
  const started = performance.now();
  let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: step(contexts[0])})));
  const ref = new p.DebugSessionRef({debugSessionId: info.debugSessionId});
  assert.equal(info.stopReason, p.DebugStopReason.DEBUG_STOP_REASON_ENTRY);
  await assert.rejects(inference.decodeStep(step(contexts[0])), error => error.status === p.NativeStatus.NATIVE_STATUS_BUSY);
  const key = ok(await debug.getDebugDecodeState(ref)).caches[0];
  assert.equal(key.role, p.DebugKVRole.DEBUG_KV_ROLE_KEY); assert.equal(key.written, false);
  const before = await rows(info, key.cacheId);
  while (info.nextStep <= key.writerStep)
    info = ok(await debug.stepDebugSession(new p.StepDebugSessionRequest({...ref, expectedRevision: info.revision})));
  const after = await rows(info, key.cacheId), row = lengths[0] * width * element;
  assert.equal(ok(await debug.getDebugDecodeState(ref)).caches[0].written, true);
  assert.deepEqual(after.subarray(0, row), before.subarray(0, row), 'KV write outside its row');
  assert.notDeepEqual(after.subarray(row), before.subarray(row), 'KV row was not written');
  info = ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({...ref, expectedRevision: info.revision})));
  assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED);
  const debugged = await ready({resultId: info.resultId});
  ok(await debug.releaseDebugSession(ref));
  assert.deepEqual(debugged, await ready(ok(await inference.decodeStep(step(contexts[1])))), 'debugged step differs');
  // Every row the finished step left, read at the next step's ENTRY stop.
  const committed = async contextId => {
    const next = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({decodeStep: new p.DecodeStepRequest({
      contextId, slotActions: actions(lengths.map(n => n + 1)), inputs: feed(lengths.map(n => n + 2), 2)})})));
    const data = [];
    for (const cache of ok(await debug.getDebugDecodeState(new p.DebugSessionRef(next))).caches)
      for (let slot = 0; slot < slots; slot++)
        data.push((await rows(next, cache.cacheId, slot)).subarray(0, (lengths[slot] + 1) * width * element));
    ok(await debug.releaseDebugSession(new p.DebugSessionRef(next)));
    return data;
  };
  assert.deepEqual(await committed(contexts[0]), await committed(contexts[1]), 'debugged KV differs from the ordinary step');
  ok(await inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
  return {dtype: quantized ? 'I8' : 'F32', slots, stepsVisited: Number(info.events.count) / 2, stepCount: info.stepCount,
    elapsedMs: performance.now() - started, outputExact: true, kvExact: true};
}
