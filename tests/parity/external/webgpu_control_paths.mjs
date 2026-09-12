/** Real-device checks for the C dynamic, batching and decode control paths. */
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
const only = args.get('--only');
assert.ok(only === undefined || ['dynamic','independent-batch','required-row'].includes(only),
  '--only must name dynamic, independent-batch or required-row');
const api = await import(pathToFileURL(path.resolve(args.get('--bundle'))).href);
const wasmUrl = path.resolve(args.get('--wasm'));
const p = api.pb;
const ok = value => { assert.equal((value.report ?? value).status, 0, (value.report ?? value).message); return value; };
const relu = (shape, dimensions = {}) => ({ format: 'volvox-graph/v1', dimensions,
  inputs: { x: { dtype: 'float32', shape } }, nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'x' },
  outputs: { out: { tensor: 'y', dtype: 'float32', shape } }, params: {} }], outputs: ['y'] });
const tensor = (shape, values) => new p.Tensor({ name: 'x', shape: shape.map(BigInt),
  dtype: p.DataType.DATA_TYPE_F32, inline: new Uint8Array(Float32Array.from(values).buffer) });
const policy = backend => new p.BackendPolicy({ mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
  backends: [backend], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID });
function equal(actual, expected) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) <= 1e-6,
    `element ${i}: ${actual[i]} != ${expected[i]}`);
}
async function fixture(graph, scheduled = false) {
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  assert.ok(adapter);
  const identity = ['vendor','architecture','device','description']
    .map(key => adapter.info?.[key] ?? '').join(' ').trim();
  assert.ok(identity, 'the physical adapter must report its identity');
  assert.doesNotMatch(identity, /swiftshader|llvmpipe|software/i);
  console.log(`webgpu-control adapter: ${identity}`);
  const device = await adapter.requestDevice();
  const bridge = await api.createWebGPUHostBridge({ device });
  const stats = { encodes: 0 };
  const counted = { ...bridge, imports: { ...bridge.imports, vx_gpu_encode(...args) {
    stats.encodes++; return bridge.imports.vx_gpu_encode(...args);
  } } };
  const host = new api.FullEngineHost({ wasmUrl, gpuBridge: counted,
    fetch: async () => ({ ok: true, arrayBuffer: async () => new TextEncoder().encode(JSON.stringify(graph)).buffer }),
  });
  const inference = new api.VxInferenceServiceClient(host);
  const scheduler = new api.VxSchedulerServiceClient(host);
  const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({
    executionMode: scheduled ? p.ExecutionMode.EXECUTION_MODE_SCHEDULED : p.ExecutionMode.EXECUTION_MODE_DIRECT,
  })));
  const model = ok(await inference.loadModel(new p.LoadModelRequest({ runtimeId: runtime.runtimeId, graphPath: 'graph.json' })));
  const compiled = ok(await inference.compileModel(new p.CompileModelRequest({ modelId: model.modelId, policy: policy('webgpu') })));
  assert.equal(stats.encodes, 0, 'compile must not submit work');
  const context = async (fields = {}, id = compiled.compiledModelId) => ok(await inference.createExecutionContext(
    new p.CreateExecutionContextRequest({ compiledModelId: id, ...fields })));
  const read = async result => {
    const ready = ok(await inference.getResult(new p.ResultRef(result)));
    assert.equal(ready.state, p.ResultState.RESULT_STATE_READY);
    const output = ok(await inference.readOutput(new p.ReadOutputRequest({ resultId: result.resultId, name: 'y' })));
    const bytes = output.tensor.inline;
    return [...new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))];
  };
  return { inference, scheduler, runtime, model, compiled, context, read, stats,
    async close() { await host.close(); device.destroy(); await device.lost; } };
}
if (!only || only === 'dynamic') {
  const f = await fixture(relu(['B','S',5], { B: {min:1,max:4}, S:{min:2,max:8,multiple_of:2} }));
  try {
    const gpu = await f.context();
    const cpuCompiled = ok(await f.inference.compileModel(new p.CompileModelRequest({ modelId:f.model.modelId,policy:policy('wasm') })));
    const cpu = await f.context({}, cpuCompiled.compiledModelId);
    for (let repeat = 0; repeat < 8; repeat++) {
      for (const shape of [[1,2,5],[2,4,5],[4,8,5],[3,6,5]]) {
        const values = Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i + repeat));
        const execute = async context => ok(await f.inference.execute(new p.ExecuteRequest({contextId:context.contextId,inputs:[tensor(shape,values)]})));
        const a=await execute(gpu), b=await execute(cpu);
        equal(await f.read(a),await f.read(b));
        ok(await f.inference.releaseResult(new p.ResultRef(a))); ok(await f.inference.releaseResult(new p.ResultRef(b)));
      }
    }
    console.log('webgpu-control dynamic: PASS (32 changing shapes, full C/WASM parity)');
  } finally { await f.close(); }
}
if (!only || only === 'independent-batch') {
  const f = await fixture(relu(['B',3], {B:{min:1,max:4}}), true);
  try {
    for (const count of [2,4,3]) {
      const requests=[];
      for(let lane=0;lane<count;lane++) requests.push(ok(await f.scheduler.submit(new p.SubmitRequest({
        compiledModelId:f.compiled.compiledModelId,inputs:[tensor([1,3],Array(3).fill(lane+1))]}))));
      const before=f.stats.encodes;
      await f.scheduler.waitRequest(new p.RequestRef(requests[0]));
      assert.equal(f.stats.encodes-before,1,'batch must have one physical invocation');
      for(const [lane,request] of requests.entries()) {
        const state=await f.scheduler.pollRequest(new p.RequestRef(request));
        assert.equal(state.state,p.RequestState.REQUEST_STATE_SUCCEEDED,state.report.message);
        const result=ok(await f.scheduler.takeRequestResult(new p.RequestRef(request)));
        equal(await f.read(result),Array(3).fill(lane+1));
        ok(await f.inference.releaseResult(new p.ResultRef(result)));
        ok(await f.scheduler.releaseRequest(new p.RequestRef(request)));
      }
    }
    console.log('webgpu-control independent-batch: PASS (B=2/4/3, one physical invocation each)');
  } finally {await f.close();}
}
if (!only || only === 'required-row') {
  const shape=[1,4,5];
  const f=await fixture(relu(shape));
  try {
    const context=await f.context({decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true});
    const values=Array.from({length:20},(_,i)=>i%2 ? i : -i);
    const expected=values.map(v=>Math.max(0,v));
    const prefill=ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({contextId:context.contextId,inputs:[tensor(shape,values)]})));
    equal(await f.read(prefill),expected);
    for(const position of [1,2]) {
      values.fill(30+position,position*5,(position+1)*5);
      expected.fill(30+position,position*5,(position+1)*5);
      const before=f.stats.encodes;
      const result=ok(await f.inference.decodeStep(new p.DecodeStepRequest({contextId:context.contextId,position,inputs:[tensor(shape,values)]})));
      assert.equal(f.stats.encodes-before,3,'row gather, numerical dispatch and row scatter');
      equal(await f.read(result),expected);
      ok(await f.inference.releaseResult(new p.ResultRef(result)));
    }
    equal(await f.read(prefill),Array.from({length:20},(_,i)=>i%2?i:0));
    ok(await f.inference.resetDecode(new p.ExecutionContextRef(context)));
    values.fill(7);
    const reset=ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({contextId:context.contextId,inputs:[tensor(shape,values)]})));
    equal(await f.read(reset),Array(20).fill(7));
    ok(await f.inference.releaseResult(new p.ResultRef(reset)));
    ok(await f.inference.releaseResult(new p.ResultRef(prefill)));
    console.log('webgpu-control required-row: PASS (unaligned rows, prefix preservation, reset, old snapshot)');
  } finally {await f.close();}
}
