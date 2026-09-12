/** Multi-input, branching, dense and normalization domain/row qualification. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const args = new Map();
for (let i=2;i<process.argv.length;i+=2) args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href), p=api.pb;
const ok=value=>{assert.equal((value.report??value).status,0,JSON.stringify(value.report??value,(_,v)=>typeof v==='bigint'?String(v):v));return value;};
const policy=backend=>new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
  backends:[backend],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID});
const weights={w:{shape:[3,5],values:Array.from({length:15},(_,i)=>Math.sin(i+1)*.3)},
  bias:{shape:[5],values:[.1,-.2,.3,0,.15]},g:{shape:[5],values:[1,.8,1.1,.9,1.2]},
  beta:{shape:[5],values:[0,.1,-.1,.2,-.2]}};
const floats=values=>new Uint8Array(Float32Array.from(values).buffer);
function shard() {
  const header={},chunks=[];let at=0;
  for(const [name,t] of Object.entries(weights)) {const chunk=floats(t.values);
    header[name]={dtype:'F32',shape:t.shape,data_offsets:[at,at+chunk.length]};chunks.push(chunk);at+=chunk.length;}
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);
  const data=new Uint8Array(8+json.length+at);new DataView(data.buffer).setBigUint64(0,BigInt(json.length),true);
  data.set(new TextEncoder().encode(json),8);at=8+json.length;
  for(const chunk of chunks){data.set(chunk,at);at+=chunk.length;}return data;
}
function graph(shape,dimensions={}) {
  const outShape=[...shape.slice(0,-1),5];
  const node=(id,opType,inputs,outputShape=shape,params={})=>({id,opType,inputs,
    outputs:{out:{tensor:id,dtype:'float32',shape:outputShape}},params});
  return {format:'volvox-graph/v1',dimensions,inputs:{x:{dtype:'float32',shape},z:{dtype:'float32',shape}},
    nodes:[node('a','ReLU',{input:'x'}),node('b','Tanh',{input:'z'}),node('sum','Add',{a:'a',b:'b'}),
      node('dense','MatMul',{input:'sum',weight:'w',bias:'bias'},outShape,{weight_layout:'din_dout'}),
      node('norm','LayerNorm',{input:'dense',weight:'g',bias:'beta'},outShape,{eps:1e-5}),
      node('y','Softmax',{input:'norm'},outShape,{axis:-1})],outputs:['y']};
}
// Independent scalar reference; it does not share C inference implementations.
function expected(x,z) {
  const output=[];
  for(let start=0;start<x.length;start+=3) {
    const h=Array.from({length:3},(_,j)=>Math.max(0,x[start+j])+Math.tanh(z[start+j]));
    const d=weights.bias.values.map((bias,col)=>bias+h.reduce((sum,value,j)=>sum+value*weights.w.values[j*5+col],0));
    const mean=d.reduce((a,b)=>a+b,0)/5,variance=d.reduce((sum,v)=>sum+(v-mean)**2,0)/5;
    const n=d.map((v,j)=>(v-mean)/Math.sqrt(variance+1e-5)*weights.g.values[j]+weights.beta.values[j]);
    const e=n.map(v=>Math.exp(v-Math.max(...n))),total=e.reduce((a,b)=>a+b,0);output.push(...e.map(v=>v/total));
  }return output;
}
function equal(actual,reference) {
  assert.equal(actual.length,reference.length);
  for(let i=0;i<actual.length;i++) assert.ok(Number.isFinite(actual[i])&&Math.abs(actual[i]-reference[i])<2e-5,
    `element ${i}: ${actual[i]} != ${reference[i]}`);
}
const inputs=(shape,x,z)=>['x','z'].map((name,i)=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_F32,
  shape:shape.map(BigInt),inline:floats(i?z:x)}));
async function fixture(modelGraph,scheduled=false,refusal=false) {
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});assert.ok(adapter);
  const identity=['vendor','architecture','device','description'].map(k=>adapter.info?.[k]??'').join(' ').trim();
  assert.ok(identity);assert.doesNotMatch(identity,/swiftshader|llvmpipe|software/i);console.log(`adapter: ${identity}`);
  const device=await adapter.requestDevice(),bridge=await api.createWebGPUHostBridge({device}),stats={encodes:0};
  const counted={...bridge,imports:{...bridge.imports,vx_gpu_encode(...values){stats.encodes++;return bridge.imports.vx_gpu_encode(...values);}}};
  const data=new TextEncoder().encode(JSON.stringify(modelGraph)),weightData=shard();
  const host=new (api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:counted,
    fetch:async url=>new Response(url==='graph.json'?data:weightData)});
  const inference=new api.VxInferenceServiceClient(host),scheduler=new api.VxSchedulerServiceClient(host);
  const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest({executionMode:scheduled?
    p.ExecutionMode.EXECUTION_MODE_SCHEDULED:p.ExecutionMode.EXECUTION_MODE_DIRECT})));
  const loaded=await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']}));
  const candidate=loaded.report.status===0?await inference.compileModel(new p.CompileModelRequest({modelId:loaded.modelId,policy:policy('webgpu')})):loaded;
  if(refusal) {
    assert.notEqual(candidate.report.status,0,'an unproved domain must be refused');assert.equal(stats.encodes,0);
    await host.close();device.destroy();await device.lost;return;
  }
  const model=ok(loaded),compiled=ok(candidate);
  assert.equal(stats.encodes,0);
  const context=async(fields={},id=compiled.compiledModelId)=>ok(await inference.createExecutionContext(
    new p.CreateExecutionContextRequest({compiledModelId:id,...fields})));
  const read=async result=>{
    const deadline=Date.now()+30_000;let state;
    do {state=ok(await inference.getResult(new p.ResultRef(result)));
      assert.ok(Date.now()<deadline);if(state.state===p.ResultState.RESULT_STATE_PENDING)await new Promise(r=>setTimeout(r,1));
    }while(state.state===p.ResultState.RESULT_STATE_PENDING);
    assert.equal(state.state,p.ResultState.RESULT_STATE_READY);
    const t=ok(await inference.readOutput(new p.ReadOutputRequest({resultId:result.resultId,name:'y'}))).tensor;
    return [...new Float32Array(t.inline.slice().buffer)];
  };
  return {inference,scheduler,model,compiled,context,read,stats,
    release:async result=>ok(await inference.releaseResult(new p.ResultRef(result))),
    close:async()=>{await host.close();device.destroy();await device.lost;}};
}
const only=args.get('--only');
if(!only||only==='dynamic') {
  const f=await fixture(graph(['B','S',3],{B:{min:1,max:4},S:{min:2,max:8,multiple_of:2}}));
  try {
    const gpu=await f.context(),cpuCompiled=ok(await f.inference.compileModel(new p.CompileModelRequest({modelId:f.model.modelId,policy:policy('wasm')})));
    const cpu=await f.context({},cpuCompiled.compiledModelId);
    for(let repeat=0;repeat<4;repeat++) for(const shape of [[1,2,3],[2,4,3],[4,8,3],[3,6,3]]) {
      const n=shape.reduce((a,b)=>a*b,1),x=Array.from({length:n},(_,i)=>Math.sin(i+repeat)),z=x.map((_,i)=>Math.cos(i-repeat));
      const a=ok(await f.inference.execute(new p.ExecuteRequest({contextId:gpu.contextId,inputs:inputs(shape,x,z)})));
      const b=ok(await f.inference.execute(new p.ExecuteRequest({contextId:cpu.contextId,inputs:inputs(shape,x,z)})));
      equal(await f.read(a),expected(x,z));equal(await f.read(a),await f.read(b));await f.release(a);await f.release(b);
    }
    console.log('expanded dynamic: PASS (16 changing B/S, two inputs, branches, dense, LayerNorm, Softmax; independent oracle and C parity)');
  } finally {await f.close();}
}
if(!only||only==='independent-batch') {
  const f=await fixture(graph(['B',3],{B:{min:1,max:4}}),true);
  try {
    for(const count of [2,4,3]) {
      const requests=[];
      for(let lane=0;lane<count;lane++) {
        const x=[lane-.5,lane+.2,-lane-1],z=[.2-lane,.4+lane,-.1];
        requests.push({x,z,request:ok(await f.scheduler.submit(new p.SubmitRequest({compiledModelId:f.compiled.compiledModelId,inputs:inputs([1,3],x,z)})))});
      }
      const before=f.stats.encodes;await f.scheduler.waitRequest(new p.RequestRef(requests[0].request));
      assert.equal(f.stats.encodes-before,6,'six graph nodes must execute once for the combined batch');
      for(const {x,z,request} of requests) {
        const result=ok(await f.scheduler.takeRequestResult(new p.RequestRef(request)));
        equal(await f.read(result),expected(x,z));await f.release(result);ok(await f.scheduler.releaseRequest(new p.RequestRef(request)));
      }
    }
    console.log('expanded independent-batch: PASS (B=2/4/3, one six-node invocation each, independent per-lane oracle)');
  } finally {await f.close();}
}
if(!only||only==='required-row') {
  const shape=[1,4,3],f=await fixture(graph(shape));
  try {
    const context=await f.context({decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true});
    const x=Array.from({length:12},(_,i)=>Math.sin(i)),z=x.map((_,i)=>Math.cos(i));
    const prefill=ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({contextId:context.contextId,inputs:inputs(shape,x,z)})));
    const original=expected(x,z);equal(await f.read(prefill),original);
    for(const position of [1,2,3]) {
      for(let j=position*3;j<(position+1)*3;j++){x[j]=Math.cos(j);z[j]=-Math.sin(j);}
      const before=f.stats.encodes;
      const result=ok(await f.inference.decodeStep(new p.DecodeStepRequest({contextId:context.contextId,position,inputs:inputs(shape,x,z)})));
      assert.equal(f.stats.encodes-before,19,'GPU row gathers, six numerical nodes and scatters');
      equal(await f.read(result),expected(x,z));equal(await f.read(prefill),original);await f.release(result);
    }
    ok(await f.inference.resetDecode(new p.ExecutionContextRef(context)));x.fill(.25);z.fill(-.5);
    const reset=ok(await f.inference.decodePrefill(new p.DecodePrefillRequest({contextId:context.contextId,inputs:inputs(shape,x,z)})));
    equal(await f.read(reset),expected(x,z));await f.release(reset);await f.release(prefill);
    console.log('expanded required-row: PASS (two-input row edits, 3-to-5 dense, normalization, prefix, old snapshot and reset)');
  } finally {await f.close();}
}

if(!only||only==='refusals') {
  const unrelated=graph(['B',3],{B:{min:1,max:4},C:{min:1,max:4}});
  unrelated.inputs.z.shape=['C',3];unrelated.nodes[1].outputs.out.shape=['C',3];
  await fixture(unrelated,false,true);
  const crossing=graph(['B','S',3],{B:{min:1,max:4},S:{min:2,max:8,multiple_of:2}});
  crossing.nodes[4].params.d_model=10;await fixture(crossing,false,true);
  const tooLarge=graph(['B',3],{B:{min:1,max:20_000_000}});await fixture(tooLarge,false,true);
  console.log('expanded domain refusals: PASS (independent symbols, cross-row normalization, maximum device buffer demand; zero submissions)');
}
