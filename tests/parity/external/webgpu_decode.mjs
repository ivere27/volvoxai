/** Public decode transactions against independent whole-sequence execution.
 * Every step also checks all untouched rows and the original owned snapshot. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();
for(let i=2;i<process.argv.length;i+=2) args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const ok=(value,label='')=>{assert.equal((value.report??value).status,0,`${label}: ${JSON.stringify(value.report??value,(_,v)=>typeof v==='bigint'?String(v):v)}`);return value;};
const types={float32:[Float32Array,'F32',p.DataType.DATA_TYPE_F32],int32:[Int32Array,'I32',p.DataType.DATA_TYPE_I32],int8:[Int8Array,'I8',p.DataType.DATA_TYPE_I8],uint8:[Uint8Array,'U8',p.DataType.DATA_TYPE_U8]};
const shapeSize=shape=>shape.reduce((a,b)=>a*b,1);
const spec=(shape,dtype='float32')=>({shape,dtype});
const node=(id,opType,inputs,shape,dtype='float32',params={})=>({id,opType,inputs,outputs:{out:{tensor:id,shape,dtype}},params});
const weight=(shape,data,dtype='float32')=>({...spec(shape,dtype),data:types[dtype][0].from(data)});
const policy=backend=>new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[backend],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID});
const fixtures=[];
function add(name,inputs,nodes,weights={},quantization={},changed=Object.keys(inputs)) {
  fixtures.push({name,graph:{format:'volvox-graph/v1',dimensions:{},inputs,nodes,outputs:[nodes.at(-1).id],
    ...(Object.keys(quantization).length?{quantization:{format:'volvox-affine-safetensors/v1',tensors:quantization}}:{})},weights,changed});
}
function quantized(weights,quantization,name,scale=.05,axis) {
  const count=axis===undefined?1:weights[name].shape[axis];
  weights[name+'.scale']=weight([count],Array(count).fill(scale));
  weights[name+'.zero']=weight([count],Array(count).fill(0),'int8');
  quantization[name]={scheme:axis===undefined?'per_tensor':'per_axis',...(axis===undefined?{}:{axis}),scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
}
const S=4,shape=[1,S,3];
for(const op of ['GELU','SiLU','Sub','Div','Where','Equal','GreaterOrEqual']) {
  const comparison=['Equal','GreaterOrEqual'].includes(op),dtype=comparison?'int32':'float32';
  const inputs={x:spec(shape,dtype)},ports={input:'x'};
  if(['Sub','Div', 'Equal','GreaterOrEqual'].includes(op)){inputs.z=spec(shape,dtype);Object.assign(ports,{a:'x',b:'z'});delete ports.input;}
  if(op==='Where'){inputs.z=spec(shape);inputs.condition=spec(shape,'int32');Object.assign(ports,{condition:'condition',a:'x',b:'z'});delete ports.input;}
  add(op,inputs,[node('y',op,ports,shape,dtype)]);
}
add('Embedding',{tokens:spec([1,S],'int32')},[node('y','Embedding',{input:'tokens',weight:'w'},shape)],
  {w:weight([7,3],Array.from({length:21},(_,i)=>Math.sin(i)))});
add('BatchMatMul',{x:spec(shape)},[node('y','BatchMatMul',{a:'x',b:'w'},[1,S,5])],
  {w:weight([1,3,5],Array.from({length:15},(_,i)=>Math.cos(i)*.1))});
add('view-sequence-major',{x:spec([S,1,3])},[
  node('r','Reshape',{input:'x'},[1,S,3],'float32',{shape:[1,S,3]}),
  node('y','SiLU',{input:'r'},[1,S,3])]);
add('invariant-image-branch',{x:spec(shape),image:spec([1,2,2,3])},[
  node('pool','GlobalAveragePool',{input:'image'},[1,1,1,3]),
  node('features','Reshape',{input:'pool'},[1,1,3],'float32',{shape:[1,1,3]}),
  node('y','Add',{a:'x',b:'features'},shape)],{}, {},['x']);
{
  const w={'scale':weight([1],[.05]),zero:weight([1],[0],'int8')},q={};
  q.bytes={scheme:'per_tensor',scale_tensor:'scale',zero_point_tensor:'zero'};
  add('affine-unaligned-bytes',{x:spec(shape)},[
    node('bytes','QuantizeLinear',{input:'x',scale:'scale',zero_point:'zero'},shape,'int8'),
    node('y','DequantizeLinear',{input:'bytes',scale:'scale',zero_point:'zero'},shape)],w,q);
}
for(const op of ['QGELU','QSiLU','QAdd','QArgMax','QLinear','QEmbedding','QBatchMatMul','QLayerNorm']) {
  const w={},q={},inputs={x:spec(shape,'int8')};
  let outputShape=shape,outputType='int8',ports={input:'x'},params={};
  quantized(w,q,'x');
  if(op==='QAdd'){inputs.z=spec(shape,'int8');ports={a:'x',b:'z'};quantized(w,q,'z');}
  if(op==='QArgMax'){outputShape=[1,S];outputType='int32';params={axis:-1};}
  if(op==='QLinear'){w.w=weight([5,3],Array.from({length:15},(_,i)=>i-7),'int8');w.b=weight([5],Array(5).fill(0),'int32');ports.weight='w';ports.bias='b';outputShape=[1,S,5];quantized(w,q,'w',.02,0);}
  if(op==='QEmbedding'){
    delete inputs.x;delete q.x;inputs.tokens=spec([1,S],'int32');ports={input:'tokens',weight:'w'};
    w.w=weight([7,3],Array.from({length:21},(_,i)=>i-10),'int8');quantized(w,q,'w',.03,0);
  }
  if(op==='QBatchMatMul'){w.w=weight([1,3,5],Array.from({length:15},(_,i)=>i-7),'int8');ports={a:'x',b:'w'};outputShape=[1,S,5];quantized(w,q,'w',.03);}
  if(op==='QLayerNorm'){w.g=weight([3],[1,.8,1.2]);w.b=weight([3],[.1,0,-.1]);ports.weight='g';ports.bias='b';params={eps:1e-5};}
  if(outputType==='int8') quantized(w,q,'y',.03);
  add(op,inputs,[node('y',op,ports,outputShape,outputType,params)],w,q);
}
for(const op of ['QSDPA','CrossSDPA']) for(const causal of [true,false]) for(const layout of ['none','K','BK','QK','BQK']) {
  const dtype=op==='QSDPA'?'int8':'float32',w={},q={},D=8,K=causal?S:5;
  const inputs={q:spec([1,S,D],dtype),k:spec([1,K,D],dtype),v:spec([1,K,D],dtype)},ports={q:'q',k:'k',v:'v'};
  if(layout!=='none'){inputs.mask=spec(({K:[K],BK:[1,K],QK:[S,K],BQK:[1,S,K]})[layout],'int32');ports.mask='mask';}
  if(dtype==='int8') for(const name of ['q','k','v','y'])quantized(w,q,name,.025);
  add(`${op}-${causal?'causal':'memory'}-${layout}`,inputs,[node('y',op,ports,[1,S,D],dtype,{heads:2,causal})],w,q,causal?['q','k','v']:['q']);
}
// Lane count is an explicit declaration. Invariant operands keep their
// original shape; only tensors carrying the sequence axis acquire lanes.
for(const original of [...fixtures]) if(['GELU','Where','QEmbedding','QLinear','QAdd',
    'QSDPA-causal-none','QSDPA-causal-BK','QSDPA-causal-BQK','QSDPA-memory-QK',
    'CrossSDPA-causal-BK','CrossSDPA-memory-BQK','invariant-image-branch'].includes(original.name)) {
  const graph=structuredClone(original.graph);
  for(const [name,t] of Object.entries(graph.inputs)) {
    if(t.shape[0]===1 && ((t.shape[1]===S && name!=='image') || ['q','k','v','mask'].includes(name)))t.shape[0]=3;
  }
  for(const n of graph.nodes) for(const t of Object.values(n.outputs))
    if(t.shape[0]===1 && t.shape[1]===S)t.shape[0]=3;
  fixtures.push({...original,name:original.name+'-lanes3',graph,lanes:3});
}
// These cross the old native lane and WebGPU scratch/span constants.
{
  const original=fixtures.find(f=>f.name==='GELU');
  const graph=structuredClone(original.graph);graph.inputs.x.shape[0]=65;graph.nodes[0].outputs.out.shape[0]=65;
  fixtures.push({...original,name:'GELU-lanes65',graph,lanes:65});
  const large=[1,S,70000];
  add('large-row-scratch',{x:spec(large)},[node('y','ReLU',{input:'x'},large)]);
  const chain=[];let input='x';
  for(let i=0;i<128;i++){const id='layer'+i;chain.push(node(id,i%2?'GELU':'SiLU',{input},shape));input=id;}
  add('deep-row-spans',{x:spec(shape)},chain);
}
function shard(weights) {
  const header={},chunks=[];let offset=0;
  for(const [name,t] of Object.entries(weights)) {
    const chunk=new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength);
    header[name]={dtype:types[t.dtype][1],shape:t.shape,data_offsets:[offset,offset+chunk.length]};chunks.push(chunk);offset+=chunk.length;
  }
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);
  const data=new Uint8Array(8+json.length+offset);new DataView(data.buffer).setBigUint64(0,BigInt(json.length),true);data.set(new TextEncoder().encode(json),8);offset=8+json.length;
  for(const chunk of chunks){data.set(chunk,offset);offset+=chunk.length;}return data;
}
function compare(a,b,dtype,label) {
  assert.equal(a.length,b.length,label);
  for(let i=0;i<a.length;i++) assert.ok(Number.isFinite(a[i])&&Math.abs(a[i]-b[i])<=(dtype==='float32'?3e-5:1),`${label}[${i}] ${a[i]} != ${b[i]}`);
}
const results=[];
const decodeBackend=args.get('--backend')??'webgpu';
for(const fixture of fixtures.filter(f=>!args.has('--only')||f.name===args.get('--only'))) {
  const {name,graph,weights,changed}=fixture;
  const lanes=fixture.lanes??1;
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});assert.ok(adapter);
  const device=await adapter.requestDevice(),bridge=await api.createWebGPUHostBridge({device,onDiagnostic:message=>console.error(message)});
  const stats={encodes:0,snapshots:0};
  const counted={...bridge,imports:{...bridge.imports,
    vx_gpu_encode(...a){stats.encodes++;return bridge.imports.vx_gpu_encode(...a);},
    vx_gpu_snapshot(...a){stats.snapshots++;return bridge.imports.vx_gpu_snapshot(...a);}}};
  const graphBytes=new TextEncoder().encode(JSON.stringify(graph)),weightBytes=shard(weights);
  const host=new (api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:counted,
    fetch:async url=>new Response(url==='graph.json'?graphBytes:weightBytes)});
  try {
    const inference=new api.VxInferenceServiceClient(host);
    const call=async(method,request)=>ok(await inference[method](request),`${name}/${method}`);
    const runtime=await call('createRuntime',new p.CreateRuntimeRequest());
    const model=await call('loadModel',new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']}));
    const contexts=[];
    for(const [index,backend] of [decodeBackend,'wasm'].entries()){
      const compiled=await call('compileModel',new p.CompileModelRequest({modelId:model.modelId,policy:policy(backend)}));
      contexts.push(await call('createExecutionContext',new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        ...(index===0?{decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true,decodeLanes:lanes,decodeInputs:changed}:{})})));
    }
    const arrays={};
    for(const [key,t] of Object.entries(graph.inputs)) arrays[key]=types[t.dtype][0].from({length:shapeSize(t.shape)},(_,i)=>
      key==='tokens'?(i*3+1)%7:key==='mask'?(i%5===2?0:1):t.dtype==='int32'?(i%3):t.dtype==='float32'?(Math.sin(i*1.3+.2)*.7+1):((i*7+3)%31-15));
    const bindings=names=>names.map(key=>new p.Tensor({name:key,dtype:types[graph.inputs[key].dtype][2],shape:graph.inputs[key].shape.map(BigInt),inline:new Uint8Array(arrays[key].buffer.slice(0))}));
    const out=graph.nodes.at(-1).outputs.out,ctor=types[out.dtype][0];
    const read=async result=>{
      let state;const deadline=Date.now()+30000;
      do {state=await call('getResult',new p.ResultRef(result));assert.ok(Date.now()<deadline);if(state.state===p.ResultState.RESULT_STATE_PENDING)await new Promise(r=>setTimeout(r,1));}while(state.state===p.ResultState.RESULT_STATE_PENDING);
      assert.equal(state.state,p.ResultState.RESULT_STATE_READY,name);
      const response=await call('readOutput',new p.ReadOutputRequest({resultId:result.resultId,name:out.tensor}));return new ctor(response.tensor.inline.slice().buffer);
    };
    const reference=async()=>{const r=await call('execute',new p.ExecuteRequest({contextId:contexts[1].contextId,inputs:bindings(Object.keys(graph.inputs))}));const a=await read(r);await call('releaseResult',new p.ResultRef(r));return a;};
    const prefill=await call('decodePrefill',new p.DecodePrefillRequest({contextId:contexts[0].contextId,inputs:bindings(Object.keys(graph.inputs))}));
    const shapePlan=prefill.report.route?.shapePlan;
    assert.ok(shapePlan && shapePlan.logicalBytes>0n && shapePlan.logicalBytes<16777216n,name+'/logical-bytes');
    assert.ok(shapePlan.arenaCapacityBytes>=shapePlan.arenaRequiredBytes && shapePlan.arenaHighWaterBytes>=shapePlan.arenaCapacityBytes,name+'/arena-accounting');
    const initial=await read(prefill);compare(initial,await reference(),out.dtype,name+'/prefill');
    const state=async()=>call('getDecodeState',new p.ExecutionContextRef(contexts[0]));
    const firstState=await state();assert.equal(firstState.lanes,lanes);assert.equal(firstState.prefilled,true);
    assert.deepEqual(firstState.activeLengths,Array(lanes).fill(1));
    const retained=initial.slice(),width=retained.length/(lanes*S);
    const patterns=[['parked',1,'idle'],[1,2,'parked'],[2,'idle',1]];
    const schedule=lanes===1?[[1],[2],undefined]:[...patterns.map(pattern=>Array.from({length:lanes},(_,i)=>pattern[i%3])),undefined];
    for(const actions of schedule) {
      const prior=await state(),positions=actions??prior.activeLengths;
      for(const key of changed){const a=arrays[key],rowWidth=a.length/(lanes*S);
        for(let lane=0;lane<lanes;lane++) {
          const position=positions[lane];if(typeof position!=='number')continue;
          for(let j=0;j<rowWidth;j++){const i=(lane*S+position)*rowWidth+j;
            a[i]=key==='tokens'?(a[i]+2)%7:graph.inputs[key].dtype==='float32'?(.35+Math.cos(i+position)*.6):graph.inputs[key].dtype==='int32'?(a[i]+1)%3:((a[i]+7)%29);}
        }
      }
      const cursor=!actions?{}:lanes===1?{position:positions[0]}:{laneActions:new p.DecodeLaneActions({lanes:positions.map(action=>
        new p.DecodeLaneAction(typeof action==='number'?{position:action}:{[action]:true}))})};
      const before=stats.snapshots;
      const step=await call('decodeStep',new p.DecodeStepRequest({contextId:contexts[0].contextId,...cursor,inputs:bindings(changed)}));
      assert.equal(stats.snapshots-before,decodeBackend==='webgpu'?1:0,name+' snapshots only the final output');
      const actual=await read(step),whole=await reference();
      for(let lane=0;lane<lanes;lane++) for(let row=0;row<S;row++) {
        const start=(lane*S+row)*width,end=start+width,position=positions[lane];
        if(typeof position==='number' && row===position)compare(actual.slice(start,end),whole.slice(start,end),out.dtype,name+`/lane${lane}/row${row}`);
        else assert.deepEqual(actual.slice(start,end),retained.slice(start,end),name+`/retained-lane${lane}/row${row}`);
      }
      retained.set(actual);assert.deepEqual(await read(prefill),initial,name+'/old-snapshot');
      const next=await state();assert.deepEqual(next.activeLengths,positions.map((pos,lane)=>typeof pos==='number'?pos+1:prior.activeLengths[lane]));
      assert.deepEqual(next.parked,positions.map(pos=>pos==='parked'));assert.equal(next.cacheGeneration,firstState.cacheGeneration);
      assert.equal(next.report.lineage.contextId,contexts[0].contextId);
      await call('releaseResult',new p.ResultRef(step));
    }
    const preserved=await state();
    for(const cursor of [
      {},
      {laneActions:new p.DecodeLaneActions()},
      {laneActions:new p.DecodeLaneActions({lanes:Array.from({length:lanes},()=>new p.DecodeLaneAction({idle:true}))})},
      {laneActions:new p.DecodeLaneActions({lanes:Array.from({length:lanes},()=>new p.DecodeLaneAction({parked:true}))})},
      {position:0},
    ]) {
      const before=stats.encodes;
      const refusal=await inference.decodeStep(new p.DecodeStepRequest({contextId:contexts[0].contextId,...cursor,inputs:bindings(changed)}));
      assert.notEqual(refusal.report.status,0);assert.equal(refusal.resultId,0n);assert.equal(stats.encodes,before);
      const after=await state();assert.deepEqual(after.activeLengths,preserved.activeLengths);assert.deepEqual(after.parked,preserved.parked);assert.equal(after.cacheGeneration,preserved.cacheGeneration);
    }
    const malformedPrefill=await inference.decodePrefill(new p.DecodePrefillRequest({contextId:contexts[0].contextId,
      lanePositions:new p.DecodeLanePositions({positions:Array(lanes).fill(S)}),inputs:bindings(Object.keys(graph.inputs))}));
    assert.notEqual(malformedPrefill.report.status,0);assert.equal(malformedPrefill.resultId,0n);
    assert.deepEqual((await state()).activeLengths,preserved.activeLengths);
    await call('resetDecode',new p.ExecutionContextRef(contexts[0]));const reset=await state();
    assert.equal(reset.prefilled,false);assert.deepEqual(reset.activeLengths,Array(lanes).fill(0));assert.ok(reset.cacheGeneration>preserved.cacheGeneration);
    if(lanes>1) {
      const prompts=Array.from({length:lanes},(_,i)=>i%3===1?1:0);
      const nextPrefill=await call('decodePrefill',new p.DecodePrefillRequest({contextId:contexts[0].contextId,
        lanePositions:new p.DecodeLanePositions({positions:prompts}),inputs:bindings(Object.keys(graph.inputs))}));
      compare(await read(nextPrefill),await reference(),out.dtype,name+'/staggered-prefill');
      const stateAfterPrefill=await state();assert.deepEqual(stateAfterPrefill.activeLengths,prompts.map(n=>n+1));
      assert.ok(stateAfterPrefill.cacheGeneration>reset.cacheGeneration);
      const nextStep=await call('decodeStep',new p.DecodeStepRequest({contextId:contexts[0].contextId}));
      compare(await read(nextStep),await reference(),out.dtype,name+'/reuse-inputs');assert.deepEqual((await state()).activeLengths,prompts.map(n=>n+2));
      await call('releaseResult',new p.ResultRef(nextStep));await call('releaseResult',new p.ResultRef(nextPrefill));
    }
    await call('releaseResult',new p.ResultRef(prefill));
    results.push({name,status:'pass',backend:decodeBackend,steps:schedule.length,lanes});console.log(`${name}: PASS`);
  } finally {await host.close();device.destroy();await device.lost;}
}
assert.ok(results.length);
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify({status:'pass',cases:results},null,2)+'\n');
