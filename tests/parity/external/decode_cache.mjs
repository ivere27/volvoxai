/** Public paging, prefix sharing and slot retirement on CPU and physical GPU. */
import assert from 'node:assert/strict';
import {declareIncremental,graphRequest,graphContext,inlineOutput} from './execution_fixture.mjs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
// The public clients throw on a refused operation; the report travels with the error.
const refused=async call=>{try{return (await call).report;}catch(error){assert.ok(error.report,String(error));return error.report;}};
const ok=(v,label='')=>{assert.equal((v.report??v).status,0,label+JSON.stringify(v.report??v,(_,x)=>typeof x==='bigint'?String(x):x));return v;};
const B=3,S=8,D=8,results=[];
for(const quantized of [false,true])for(const layers of [1,10])for(const backend of (args.get('--backends')??'wasm,webgpu').split(',')) {
  const name=`${quantized?'I8':'F32'}/${layers}-layers/${backend}`;
  if(args.has('--only')&&!name.includes(args.get('--only')))continue;
  const weights={},quantization={},nodes=[],paged=[];
  const dtype=quantized?'int8':'float32',ArrayType=quantized?Int8Array:Float32Array;
  const addWeight=(name,shape,data)=>weights[name]={shape,data};
  const affine=name=>{
    const axis=name.endsWith('.weight'),count=axis?D:1;
    addWeight(name+'.scale',[count],new Float32Array(count).fill(.05));
    addWeight(name+'.zero',[count],new Int8Array(count));
    quantization[name]={scheme:axis?'per_axis':'per_tensor',...(axis?{axis:0}:{}),scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
  };
  if(quantized)affine('x');
  let input='x';
  for(let layer=0;layer<layers;layer++) {
    for(const [index,port] of ['q','k','v'].entries()) {
      const id=port+layer;
      addWeight(id+'.weight',[D,D],ArrayType.from({length:D*D},(_,i)=>quantized?(i%D===Math.floor(i/D)?19:((i+index)%5-2)):(i%D===Math.floor(i/D)?.8:Math.sin(i+index)*.04)));
      addWeight(id+'.bias',[D],quantized?new Int32Array(D):new Float32Array(D));
      if(quantized){affine(id+'.weight');affine(id);}
      nodes.push({id,opType:quantized?'QLinear':'Linear',inputs:{input,weight:id+'.weight',bias:id+'.bias'},outputs:{out:{tensor:id,shape:[B,S,D],dtype}},params:{}});
      if(port!=='q')paged.push(id);
    }
    const id=layer===layers-1?'out':'attention'+layer;
    nodes.push({id,opType:quantized?'QSDPA':'CrossSDPA',inputs:{q:'q'+layer,k:'k'+layer,v:'v'+layer,mask:'keep'},outputs:{out:{tensor:id,shape:[B,S,D],dtype}},params:{heads:2,causal:true}});
    if(quantized)affine(id);input=id;
  }
  const graph={format:'volvox-graph/v1',dimensions:{},inputs:{x:{shape:[B,S,D],dtype},keep:{shape:[B,S],dtype:'int32'}},nodes,outputs:['out'],...(quantized?{quantization:{format:'volvox-affine-safetensors/v1',tensors:quantization}}:{})};
  declareIncremental(graph,{slots:B,updates:['x','keep']});
  const header={},chunks=[];let offset=0;
  for(const [key,w] of Object.entries(weights)){header[key]={dtype:w.data instanceof Float32Array?'F32':w.data instanceof Int8Array?'I8':'I32',shape:w.shape,data_offsets:[offset,offset+w.data.byteLength]};chunks.push(new Uint8Array(w.data.buffer));offset+=w.data.byteLength;}
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);const shard=new Uint8Array(8+json.length+offset);
  new DataView(shard.buffer).setBigUint64(0,BigInt(json.length),true);shard.set(new TextEncoder().encode(json),8);offset=8+json.length;for(const bytes of chunks){shard.set(bytes,offset);offset+=bytes.length;}
  const adapter=backend==='webgpu'?await navigator.gpu.requestAdapter({powerPreference:'high-performance'}):null;
  if(backend==='webgpu'){assert.ok(adapter);assert.doesNotMatch(JSON.stringify(adapter.info??{}),/swiftshader|llvmpipe|lavapipe|software/i);}
  const device=adapter?await adapter.requestDevice():null,bridge=device?await api.createWebGPUHostBridge({device,onDiagnostic:message=>console.error(message)}):null;
  const stats={snapshots:0,readbacks:0,encodes:0};
  const counted=bridge?{...bridge,imports:{...bridge.imports,
    vx_gpu_snapshot(...a){if(a[2]>0)stats.snapshots++;return bridge.imports.vx_gpu_snapshot(...a);},
    vx_gpu_readback(...a){stats.readbacks++;return bridge.imports.vx_gpu_readback(...a);},
    vx_gpu_encode(...a){stats.encodes++;return bridge.imports.vx_gpu_encode(...a);}}}:undefined;
  const graphBytes=new TextEncoder().encode(JSON.stringify(graph));
  const host=new(api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:counted,fetch:async url=>new Response(url==='graph.json'?graphBytes:shard)});
  try {
    const inference=new api.VxInferenceServiceClient(host),call=async(method,r)=>ok(await inference[method](r),name+'/'+method);
    const runtime=await call('createRuntime',new p.CreateRuntimeRequest());
    const model=await call('loadModel',new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']}));
    const contexts=[],compiledIds=[];
    const cacheOptions={tensors:paged,pageTokens:2,maxPages:8,clearOnRecycle:true};
    for(const [index,target] of [backend,'wasm'].entries()) {
      const compiled=await call('compileModel',new p.CompileModelRequest({modelId:model.modelId,policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[target],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      compiledIds.push(compiled.compiledModelId);
      contexts.push(await call('createExecutionContext',graphContext(p,compiled.compiledModelId,
        index===0?{rowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,slots:B,cache:cacheOptions}:{})));

    }
    const contextId=contexts[0].contextId,ref=new p.ExecutionContextRef({contextId});
    const lengths=[4,2,6],x=ArrayType.from({length:B*S*D},(_,i)=>quantized?(i*7%39)-19:Math.sin(i*.71));
    const keep=Int32Array.from({length:B*S},(_,i)=>i%S<lengths[Math.floor(i/S)]?1:0);
    const bindings=()=>[new p.Tensor({name:'x',dtype:quantized?p.DataType.DATA_TYPE_I8:p.DataType.DATA_TYPE_F32,shape:[BigInt(B),BigInt(S),BigInt(D)],inline:new Uint8Array(x.buffer.slice(0))}),new p.Tensor({name:'keep',dtype:p.DataType.DATA_TYPE_I32,shape:[BigInt(B),BigInt(S)],inline:new Uint8Array(keep.buffer.slice(0))})];
    const read=async result=>inlineOutput(result,'out',ArrayType);
    const ordinary=async()=>read(await call('execute',graphRequest(p,graph,contexts[1].contextId,bindings(),{lengths})));
    const state=()=>call('getExecutionContextInfo',ref);
    const cacheState=async()=>(await state()).cache;
    let refusals=0,steps=0;
    for(const fields of [{pageTokens:0},{maxPages:1},{tensors:['x']},{tensors:['missing']},{tensors:[paged[0],paged[0]]}]) {
      const before={...stats},prior=await state();
      let rejected;
      try {
        const candidate=await inference.createExecutionContext(graphContext(p,compiledIds[0],
          {rowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,slots:B,cache:{...cacheOptions,...fields}}));
        rejected=await refused(inference.execute(graphRequest(p,graph,candidate.contextId,bindings(),{entrypoint:'initialize',lengths})));
        await call('releaseExecutionContext',new p.ExecutionContextRef(candidate));
      } catch(error) {assert.ok(error.report,String(error));rejected=error.report;}
      assert.notEqual(rejected.status,0);assert.deepEqual(stats,before,'invalid cache limits must fail before dispatch');
      assert.deepEqual((await state()).activeLengths,prior.activeLengths);refusals++;
    }
    const prefill=await call('execute',graphRequest(p,graph,contextId,bindings(),{entrypoint:'initialize',lengths}));
    const initial=await read(prefill);let previous=initial.slice();
    let cache=await cacheState();
    assert.deepEqual(cache.slots.map(x=>x.activeLength),lengths);assert.equal(cache.residentPages,6);assert.equal(cache.freePages,2);
    assert.deepEqual(cache.slots[1].pages,[2,-1,-1,-1]);
    assert.equal(cache.poolBytes,BigInt(B*S*D*2*layers*ArrayType.BYTES_PER_ELEMENT));
    assert.equal(cache.reservedPages,0);assert.equal((await state()).report.lineage.contextId,contextId);
    const step=async positions=>{
      const before={...stats},expected=await ordinary();
      const result=await call('execute',graphRequest(p,graph,contextId,bindings(),{entrypoint:'transition',positions}));
      // The ordinary CPU reference creates no GPU work.
      assert.equal(stats.snapshots-before.snapshots,backend==='webgpu'?1:0);
      const actual=await read(result);
      for(let slot=0;slot<B;slot++) {
        const pos=positions[slot]==='recompute'?lengths[slot]-1:positions[slot];
        for(let row=0;row<S;row++)for(let d=0;d<D;d++) {
          const index=(slot*S+row)*D+d;
          if(pos!==null&&row===pos)assert.ok(Math.abs(actual[index]-expected[index])<=(quantized?1:4e-5),`${name}/${steps}/${slot}/${row}/${d}: ${actual[index]} != ${expected[index]}`);
          else assert.equal(actual[index],previous[index],`${name}/untouched/${slot}/${row}/${d}`);
        }
        if(pos!==null)lengths[slot]=pos+1;
      }
      previous=actual;steps++;
      assert.deepEqual((await state()).activeLengths,lengths);
      assert.deepEqual((await cacheState()).slots.map(x=>x.activeLength),lengths);
    };
    keep[4]=1;keep[S+2]=1;await step([4,2,null]);
    cache=await cacheState();assert.equal(cache.freePages,0);
    const before={...stats},mapping=cache.slots.map(x=>[...x.pages]),generation=cache.cacheGeneration;
    assert.notEqual((await refused(inference.execute(graphRequest(p,graph,contextId,bindings(),{entrypoint:'transition',positions:[5,3,6]})))).status,0);
    assert.deepEqual(stats,before);cache=await cacheState();assert.deepEqual(cache.slots.map(x=>[...x.pages]),mapping);assert.equal(cache.cacheGeneration,generation);assert.equal(cache.reservedPages,0);refusals++;
    cache=await call('releaseContextSlot',new p.ContextSlotRef({contextId,slot:2}));lengths[2]=0;assert.equal(cache.slots[2].activeLength,0);
    const key='prefix/'+('한'.repeat(180)),savedX=x.slice(0,4*D),savedKeep=keep.slice(0,4),savedOut=previous.slice(0,4*D);
    const copying={...stats};cache=await call('publishContextPrefix',new p.PublishContextPrefixRequest({contextId,key,slot:0,tokens:4}));
    assert.ok(cache.prefixSnapshotBytes>0n);assert.equal(stats.snapshots,copying.snapshots);assert.equal(stats.readbacks,copying.readbacks);
    await call('releaseContextSlot',new p.ContextSlotRef({contextId,slot:1}));lengths[1]=0;
    cache=await call('reuseContextPrefix',new p.ReuseContextPrefixRequest({contextId,key,slot:1}));lengths[1]=4;
    x.set(savedX,S*D);keep.set(savedKeep,S);previous.set(savedOut,S*D);
    assert.equal(cache.prefixHits,1n);assert.deepEqual(cache.slots[1].pages.slice(0,2),cache.slots[0].pages.slice(0,2));
    x[S*D+3*D]+=quantized?7:.35;keep[5]=1;
    await step([5,'recompute',null]);
    cache=await cacheState();assert.equal(cache.copyOnWrites,1n);assert.notEqual(cache.slots[1].pages[1],cache.slots[0].pages[1]);
    await call('releaseContextSlot',new p.ContextSlotRef({contextId,slot:1}));lengths[1]=0;
    await call('reuseContextPrefix',new p.ReuseContextPrefixRequest({contextId,key,slot:1}));lengths[1]=4;
    x.set(savedX,S*D);keep.set(savedKeep,S);previous.set(savedOut,S*D);keep[S+4]=1;
    await step([null,4,null]);
    // A released slot can start a new request at row zero while its neighbors
    // retain their own prefix. No full-batch prefill may overwrite those slots.
    for(let i=0;i<S*D;i++)x[2*S*D+i]=quantized?(i*3%23)-11:Math.cos(i*.31)*.7;
    keep.fill(0,2*S);keep[2*S]=1;await step([null,null,0]);
    keep[2*S+1]=1;await step([null,null,1]);
    assert.deepEqual(await read(prefill),initial,'old result survives paging, COW and slot replacement');
    for(let slot=0;slot<B;slot++)await call('releaseContextSlot',new p.ContextSlotRef({contextId,slot}));
    cache=await call('evictContextPrefixes',new p.EvictContextPrefixesRequest({contextId,freePages:8}));
    assert.equal(cache.freePages,8);assert.equal(cache.residentPages,0);assert.equal(cache.prefixSnapshotBytes,0n);assert.equal(cache.evictions,1n);
    const fresh=await call('createExecutionContext',graphContext(p,compiledIds[0],{rowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,slots:B,cache:cacheOptions}));
    assert.equal((await call('getExecutionContextInfo',new p.ExecutionContextRef(fresh))).cache?.configured??false,false);
    results.push({name,status:'pass',steps,refusals,pagedTensors:paged.length});console.log(name+': PASS');
  } finally {await host.close();if(device){device.destroy();await device.lost;}}
}
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify({status:'pass',cases:results},null,2)+'\n');
