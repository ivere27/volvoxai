/** C-owned feedback against an ordinary CPU autoregressive loop. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const ok=(v,label='')=>{assert.equal((v.report??v).status,0,label+JSON.stringify(v.report??v,(_,x)=>typeof x==='bigint'?String(x):x));return v;};
const results=[],S=6,D=8,V=7;
const constructors={F32:Float32Array,I8:Int8Array,I32:Int32Array};
for(const quantized of [false,true])for(const backend of (args.get('--backends')??'wasm,webgpu').split(',')) {
  const name=`${quantized?'quantized':'float'}/${backend}`,weights={},quantization={};
  const dtype=quantized?'int8':'float32',nodes=[];
  const addWeight=(name,shape,values,type=quantized?'I8':'F32')=>weights[name]={shape,dtype:type,data:constructors[type].from(values)};
  const affine=name=>{
    const count=name.endsWith('.weight')?weights[name].shape[0]:1;
    addWeight(name+'.scale',[count],Array(count).fill(.05),'F32');addWeight(name+'.zero',[count],Array(count).fill(0),'I8');
    quantization[name]={scheme:count===1?'per_tensor':'per_axis',...(count===1?{}:{axis:0}),scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
  };
  const node=(id,opType,inputs,shape,type=dtype,params={})=>{nodes.push({id,opType,inputs,outputs:{out:{tensor:id,shape,dtype:type}},params});if(quantized&&type==='int8')affine(id);};
  addWeight('embedding.weight',[V,D],Array.from({length:V*D},(_,i)=>quantized?(i*13%23)-11:Math.sin(i*.73)*.8));
  if(quantized)affine('embedding.weight');
  node('embedding',quantized?'QEmbedding':'Embedding',{input:'tokens',weight:'embedding.weight'},[1,S,D]);
  for(const [index,id] of ['q','k','v','scores'].entries()) {
    const width=id==='scores'?V:D;
    addWeight(id+'.weight',[width,D],Array.from({length:width*D},(_,i)=>quantized?(i*7+index*3)%17-8:Math.cos(i*.61+index)*.3));
    addWeight(id+'.bias',[width],Array(width).fill(0),quantized?'I32':'F32');if(quantized)affine(id+'.weight');
    if(id==='scores')node('attention',quantized?'QSDPA':'CrossSDPA',{q:'q',k:'k',v:'v',mask:'keep'},[1,S,D],dtype,{heads:2,causal:true});
    node(id,quantized?'QLinear':'Linear',{input:id==='scores'?'attention':'embedding',weight:id+'.weight',bias:id+'.bias'},[1,S,width]);
  }
  node('generated',quantized?'QArgMax':'ArgMax',{input:'scores'},[1,S],'int32',quantized?{axis:-1}:{axis:-1,keepdims:false});
  const graph={format:'volvox-graph/v1',dimensions:{},inputs:{tokens:{shape:[1,S],dtype:'int32'},keep:{shape:[1,S],dtype:'int32'}},nodes,outputs:['generated','scores'],
    ...(quantized?{quantization:{format:'volvox-affine-safetensors/v1',tensors:quantization}}:{})};
  const header={},chunks=[];let offset=0;
  for(const [key,w] of Object.entries(weights)){header[key]={dtype:w.dtype,shape:w.shape,data_offsets:[offset,offset+w.data.byteLength]};const chunk=new Uint8Array(w.data.buffer);chunks.push(chunk);offset+=chunk.length;}
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);const shard=new Uint8Array(8+json.length+offset);
  new DataView(shard.buffer).setBigUint64(0,BigInt(json.length),true);shard.set(new TextEncoder().encode(json),8);offset=8+json.length;
  for(const chunk of chunks){shard.set(chunk,offset);offset+=chunk.length;}
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}),device=await adapter.requestDevice();
  const bridge=await api.createWebGPUHostBridge({device,onDiagnostic:message=>console.error(message)});
  const stats={snapshots:0,readbacks:0,encodes:0};
  const counted={...bridge,imports:{...bridge.imports,
    vx_gpu_snapshot(...a){stats.snapshots++;return bridge.imports.vx_gpu_snapshot(...a);},
    vx_gpu_readback(...a){stats.readbacks++;return bridge.imports.vx_gpu_readback(...a);},
    vx_gpu_encode(...a){stats.encodes++;return bridge.imports.vx_gpu_encode(...a);}}};
  const graphBytes=new TextEncoder().encode(JSON.stringify(graph));
  const host=new (api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:counted,
    fetch:async url=>new Response(url==='graph.json'?graphBytes:shard)});
  try {
    const inference=new api.VxInferenceServiceClient(host),call=async(method,r)=>ok(await inference[method](r),name+'/'+method);
    const runtime=await call('createRuntime',new p.CreateRuntimeRequest());
    const model=await call('loadModel',new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']}));
    const contexts=[];
    for(const [index,target] of [backend,'wasm'].entries()) {
      const compiled=await call('compileModel',new p.CompileModelRequest({modelId:model.modelId,policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[target],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      contexts.push(await call('createExecutionContext',new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        ...(index===0?{decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true,decodeInputs:['tokens','keep']}:{})})));
    }
    const tokens=new Int32Array(S);tokens[0]=3;const keep=new Int32Array(S);keep[0]=1;
    const bindings=()=>Object.entries({tokens,keep}).map(([key,data])=>new p.Tensor({name:key,dtype:p.DataType.DATA_TYPE_I32,shape:[1n,BigInt(S)],inline:new Uint8Array(data.buffer.slice(0))}));
    const read=async result=>{
      let info;const deadline=Date.now()+30000;
      do {info=await call('getResult',new p.ResultRef(result));assert.ok(Date.now()<deadline);if(info.state===p.ResultState.RESULT_STATE_PENDING)await new Promise(r=>setTimeout(r,1));}while(info.state===p.ResultState.RESULT_STATE_PENDING);
      assert.equal(info.state,p.ResultState.RESULT_STATE_READY);
      const outputs={};
      for(const output of graph.outputs){const value=await call('readOutput',new p.ReadOutputRequest({resultId:result.resultId,name:output}));const Ctor=output==='generated'?Int32Array:quantized?Int8Array:Float32Array;outputs[output]=new Ctor(value.tensor.inline.slice().buffer);}
      return outputs;
    };
    const ordinary=async()=>{const r=await call('execute',new p.ExecuteRequest({contextId:contexts[1].contextId,inputs:bindings()}));const output=await read(r);await call('releaseResult',new p.ResultRef(r));return output;};
    const prefill=await call('decodePrefill',new p.DecodePrefillRequest({contextId:contexts[0].contextId,inputs:bindings()}));
    if(args.get('--paged')==='true')await call('configureDecodeCache',new p.ConfigureDecodeCacheRequest({contextId:contexts[0].contextId,tensors:['k','v'],pageTokens:2}));
    const initial=await read(prefill),state=async()=>call('getDecodeState',new p.ExecutionContextRef(contexts[0]));
    let reference=await ordinary(),length=1,refusals=0;
    const request=fields=>new p.DecodeGenerateRequest({contextId:contexts[0].contextId,tokenInput:'tokens',keepInput:'keep',tokenOutput:'generated',tokenCount:1,...fields});
    for(const count of [2,1,2]) {
      const prior=await state();
      for(const invalid of [{tokenCount:0},{tokenCount:99},{cacheGeneration:prior.cacheGeneration+1n},{tokenInput:'keep'},{tokenOutput:'scores'},{keepInput:'missing'}]) {
        const commands={...stats},response=await inference.decodeGenerate(request(invalid));
        assert.notEqual(response.report.status,0);assert.equal(response.resultId,0n);assert.deepEqual(stats,commands);
        const after=await state();assert.deepEqual(after.activeLengths,prior.activeLengths);assert.equal(after.cacheGeneration,prior.cacheGeneration);refusals++;
      }
      const before={...stats};
      const generated=await call('decodeGenerate',request({tokenCount:count,cacheGeneration:prior.cacheGeneration}));
      assert.equal(stats.snapshots-before.snapshots,backend==='webgpu'?graph.outputs.length:0,'only final snapshots');
      assert.equal(stats.readbacks,before.readbacks,'no intermediate token readback');
      for(let position=length;position<length+count;position++){tokens[position]=reference.generated[position-1];keep[position]=1;reference=await ordinary();}
      const actual=await read(generated);length+=count;
      for(const output of graph.outputs) {
        const width=output==='generated'?1:V;
        for(let i=0;i<length*width;i++) {
          if(output==='generated')assert.equal(actual[output][i],reference[output][i],name+'/'+output+'/'+i);
          else assert.ok(Math.abs(actual[output][i]-reference[output][i])<=(quantized?1:3e-5),name+'/'+output+'/'+i);
        }
        assert.deepEqual(actual[output].slice(length*width),initial[output].slice(length*width),'unwritten suffix');
      }
      assert.deepEqual((await state()).activeLengths,[length]);assert.deepEqual(await read(prefill),initial,'independent prefill result');
      assert.equal(generated.report.lineage.contextId,contexts[0].contextId);
      await call('releaseResult',new p.ResultRef(generated));
    }
    const full=await state(),before={...stats};
    assert.notEqual((await inference.decodeGenerate(request({}))).report.status,0);assert.deepEqual(stats,before);
    assert.deepEqual((await state()).activeLengths,[S]);refusals++;
    await call('resetDecode',new p.ExecutionContextRef(contexts[0]));
    assert.notEqual((await inference.decodeGenerate(request({cacheGeneration:full.cacheGeneration}))).report.status,0);refusals++;
    await call('releaseResult',new p.ResultRef(prefill));
    results.push({name,status:'pass',paged:args.get('--paged')==='true',generatedTokens:5,resumptions:2,refusals});console.log(name+': PASS');
  } finally {await host.close();device.destroy();await device.lost;}
}
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify({status:'pass',cases:results},null,2)+'\n');
