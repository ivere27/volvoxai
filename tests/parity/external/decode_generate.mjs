/** C-owned feedback against an ordinary CPU autoregressive loop. */
import assert from 'node:assert/strict';
import {declareIncremental,graphRequest,graphContext,activeLengths,inlineOutput} from './execution_fixture.mjs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const ok=(v,label='')=>{assert.equal((v.report??v).status,0,label+JSON.stringify(v.report??v,(_,x)=>typeof x==='bigint'?String(x):x));return v;};
// The public clients throw on a refused operation; the error carries the response.
const refused=async call=>{try{return await call;}catch(error){assert.ok(error.report,String(error));return error.response??{report:error.report,outputs:[]};}};
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
  declareIncremental(graph,{generation:{initialize_entrypoint:'initialize',transition_entrypoint:'transition',
    token_input:'tokens',token_output:'generated',keep_input:'keep',logits_output:'scores',logits_axis:2,
    id_domain_size:V,output_name:'generated_ids',sequence_axis:1}});
  const header={},chunks=[];let offset=0;
  for(const [key,w] of Object.entries(weights)){header[key]={dtype:w.dtype,shape:w.shape,data_offsets:[offset,offset+w.data.byteLength]};const chunk=new Uint8Array(w.data.buffer);chunks.push(chunk);offset+=chunk.length;}
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);const shard=new Uint8Array(8+json.length+offset);
  new DataView(shard.buffer).setBigUint64(0,BigInt(json.length),true);shard.set(new TextEncoder().encode(json),8);offset=8+json.length;
  for(const chunk of chunks){shard.set(chunk,offset);offset+=chunk.length;}
  const adapter=backend==='webgpu'?await navigator.gpu.requestAdapter({powerPreference:'high-performance'}):null;
  if(backend==='webgpu'){assert.ok(adapter);assert.doesNotMatch(JSON.stringify(adapter.info??{}),/swiftshader|llvmpipe|lavapipe|software/i);}
  const device=adapter?await adapter.requestDevice():null;
  const bridge=device?await api.createWebGPUHostBridge({device,onDiagnostic:message=>console.error(message)}):null;
  const stats={snapshots:0,readbacks:0,encodes:0};
  const counted=bridge?{...bridge,imports:{...bridge.imports,
    vx_gpu_snapshot(...a){if(a[2]>0)stats.snapshots++;return bridge.imports.vx_gpu_snapshot(...a);},
    vx_gpu_readback(...a){stats.readbacks++;return bridge.imports.vx_gpu_readback(...a);},
    vx_gpu_encode(...a){stats.encodes++;return bridge.imports.vx_gpu_encode(...a);}}}:undefined;
  const graphBytes=new TextEncoder().encode(JSON.stringify(graph));
  const host=new (api.FullEngineHost??api.EngineHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:counted,
    fetch:async url=>new Response(url==='graph.json'?graphBytes:shard)});
  try {
    const inference=new api.VxInferenceServiceClient(host),call=async(method,r)=>ok(await inference[method](r),name+'/'+method);
    const runtime=await call('createRuntime',new p.CreateRuntimeRequest());
    const model=await call('loadModel',new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']}));
    const contexts=[],compiledIds=[];
    for(const [index,target] of [backend,'wasm'].entries()) {
      const compiled=await call('compileModel',new p.CompileModelRequest({modelId:model.modelId,policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[target],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      compiledIds.push(compiled.compiledModelId);
      const request=index===0?new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        autoregressive:new p.AutoregressiveExecution({sampling:p.SamplingMode.SAMPLING_MODE_GREEDY,maxSteps:BigInt(S)}),
        ...(args.get('--paged')==='true'?{limits:new p.ExecutionLimits({cache:new p.ContextCacheOptions({tensors:['k','v'],pageTokens:2})})}:{})})
        :graphContext(p,compiled.compiledModelId);
      contexts.push(await call('createExecutionContext',request));
      if(index===0) {
        const randomRequest=new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
          autoregressive:new p.AutoregressiveExecution({sampling:p.SamplingMode.SAMPLING_MODE_RANDOM,maxSteps:2n,seed:42n})});
        if(quantized||target==='webgpu') {
          const unsupported=await refused(inference.createExecutionContext(randomRequest));
          assert.equal(unsupported.report.status,p.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED);
        } else {
          const admitted=await call('createExecutionContext',randomRequest);
          await call('releaseExecutionContext',new p.ExecutionContextRef(admitted));
        }
      }
    }
    const tokens=new Int32Array(S);tokens[0]=3;const keep=new Int32Array(S);keep[0]=1;
    const bindings=()=>Object.entries({tokens,keep}).map(([key,data])=>new p.Tensor({name:key,dtype:p.DataType.DATA_TYPE_I32,shape:[1n,BigInt(S)],inline:new Uint8Array(data.buffer.slice(0))}));
    const initialBindings=[...bindings(),activeLengths(p,[1])];
    const ordinary=async()=>inlineOutput(await call('execute',graphRequest(p,graph,contexts[1].contextId,bindings())),'generated',Int32Array);
    const state=async()=>call('getExecutionContextInfo',new p.ExecutionContextRef(contexts[0]));
    const request=(count,first=false,fields={})=>new p.GenerateRequest({contextId:contexts[0].contextId,
      ...(first?{initialInputs:new p.GenerationInputs({values:initialBindings})}:{}),budget:new p.GenerationBudget({maxSteps:count}),...fields});
    let reference=await ordinary(),length=0,refusals=0,retained;const greedyIds=[];
    for(const count of [2,1,3]) {
      const prior=await state();
      const invalid=[request(0,length===0),request(1,length===0,{outputs:new p.OutputOptions({selection:new p.TensorOutputSelection({names:['scores']})})})];
      if(length)invalid.push(request(1,true));
      for(const rejected of invalid) {
        const commands={...stats},response=await refused(inference.generate(rejected));
        assert.notEqual(response.report.status,0);assert.equal(response.outputs?.length??0,0);assert.deepEqual(stats,commands);
        const after=await state();assert.deepEqual(after.activeLengths,prior.activeLengths);assert.equal(after.cacheGeneration,prior.cacheGeneration);refusals++;
      }
      const before={...stats},expected=[];
      const generated=await call('generate',request(count,length===0));
      assert.equal(stats.snapshots-before.snapshots,backend==='webgpu'?1:0,'one final ID snapshot per call');
      for(let position=length;position<length+count;position++) {
        expected.push(reference[position]);
        if(position+1<S){tokens[position+1]=reference[position];keep[position+1]=1;reference=await ordinary();}
      }
      const actual=inlineOutput(generated,'generated_ids',Int32Array);
      greedyIds.push(...actual);
      assert.deepEqual([...actual],expected,name+'/IDs');
      assert.deepEqual(generated.outputs[0].shape,[1n,BigInt(count)]);
      assert.equal(generated.progress.steps,BigInt(count));
      length+=count;assert.equal(generated.progress.totalSteps,BigInt(length));
      assert.deepEqual((await state()).activeLengths,[length]);
      if(retained)assert.deepEqual([...inlineOutput(retained.packet,'generated_ids',Int32Array)],retained.ids,'prior call output is immutable');
      retained={packet:generated,ids:[...actual]};
      assert.equal(generated.report.lineage.contextId,contexts[0].contextId);
    }
    const before={...stats};
    assert.equal((await state()).state,p.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
    assert.notEqual((await refused(inference.generate(request(1)))).report.status,0);assert.deepEqual(stats,before);refusals++;
    let samplingCases=0;
    if(!quantized&&backend==='wasm') {
      const initial=()=>{
        const initialTokens=new Int32Array(S),initialKeep=new Int32Array(S);initialTokens[0]=3;initialKeep[0]=1;
        return [new p.Tensor({name:'tokens',dtype:p.DataType.DATA_TYPE_I32,shape:[1n,BigInt(S)],inline:new Uint8Array(initialTokens.buffer)}),
          new p.Tensor({name:'keep',dtype:p.DataType.DATA_TYPE_I32,shape:[1n,BigInt(S)],inline:new Uint8Array(initialKeep.buffer)}),activeLengths(p,[1])];
      };
      const contextFor=async configuration=>call('createExecutionContext',new p.CreateExecutionContextRequest({compiledModelId:compiledIds[0],
        autoregressive:new p.AutoregressiveExecution({sampling:p.SamplingMode.SAMPLING_MODE_RANDOM,maxSteps:BigInt(S),seed:42n,...configuration})}));
      const runChunks=async(configuration,chunks,expected,reason=p.ExecutionStopReason.EXECUTION_STOP_REASON_STEP_LIMIT)=>{
        const context=await contextFor(configuration);const actual=[];let last;
        try {
          for(const count of chunks) {
            last=await call('generate',new p.GenerateRequest({contextId:context.contextId,
              ...(actual.length?{}:{initialInputs:new p.GenerationInputs({values:initial()})}),budget:new p.GenerationBudget({maxSteps:count})}));
            actual.push(...inlineOutput(last,'generated_ids',Int32Array));
            assert.equal(last.progress.totalSteps,BigInt(actual.length));
            if(last.progress.stopReason===reason)break;
          }
          assert.deepEqual(actual,expected,'C sampling/stop IDs');assert.equal(last.progress.stopReason,reason);
          const terminal=await call('getExecutionContextInfo',new p.ExecutionContextRef(context));
          assert.equal(terminal.state,p.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
          assert.equal(terminal.totalSteps,BigInt(expected.length));
          const rejected=await refused(inference.generate(new p.GenerateRequest({contextId:context.contextId,budget:new p.GenerationBudget({maxSteps:1})})));
          assert.notEqual(rejected.report.status,0);samplingCases++;
        } finally {await call('releaseExecutionContext',new p.ExecutionContextRef(context));}
      };
      /* Independently computed SplitMix64(42) draws. Numerical reference logits
       * come from ordinary Graph execution; categorical policy lives only in
       * this test, and never in a product host or application generation loop. */
      const draws=[0.7415648787718233,0.1599103928769201,0.27860113025513866,
        0.34419071652363753,0.03803016854024621,0.8682280765465323];
      tokens.fill(0);tokens[0]=3;keep.fill(0);keep[0]=1;const sampledExpected=[];
      for(let row=0;row<S;row++) {
        const packet=await call('execute',graphRequest(p,graph,contexts[1].contextId,bindings()));
        const scores=inlineOutput(packet,'scores',Float32Array).subarray(row*V,(row+1)*V);
        const candidates=Array.from(scores,(score,id)=>({score,id})).sort((a,b)=>b.score-a.score||a.id-b.id).slice(0,5);
        const maximum=candidates[0].score;
        for(const candidate of candidates)candidate.mass=Math.exp((candidate.score-maximum)/.75);
        const total=candidates.reduce((sum,candidate)=>sum+candidate.mass,0),threshold=Math.fround(.8)*total;
        let retained=0,count=0;do{retained+=candidates[count++].mass;}while(count<candidates.length&&retained<threshold);
        const target=draws[row]*retained;let cumulative=0,selected=candidates[count-1].id;
        for(const candidate of candidates.slice(0,count)){cumulative+=candidate.mass;if(target<cumulative){selected=candidate.id;break;}}
        sampledExpected.push(selected);if(row+1<S){tokens[row+1]=selected;keep[row+1]=1;}
      }
      for(const chunks of [[6],[1,2,3]])await runChunks({temperature:.75,topK:5,topP:.8},chunks,sampledExpected);
      await runChunks({topK:1},[1,2,3],greedyIds);
      await runChunks({sampling:p.SamplingMode.SAMPLING_MODE_GREEDY,seed:undefined,
        eosIds:[greedyIds[0]],stopIds:[greedyIds[0]]},[6],[greedyIds[0]],p.ExecutionStopReason.EXECUTION_STOP_REASON_EOS);
      await runChunks({sampling:p.SamplingMode.SAMPLING_MODE_GREEDY,seed:undefined,
        stopSequences:[new p.IdStopSequence({ids:greedyIds.slice(0,2)})]},[1,1,4],greedyIds.slice(0,2),p.ExecutionStopReason.EXECUTION_STOP_REASON_ID_STOP);
      for(const configuration of [{temperature:0},{topP:0},{topK:V+1},{stopIds:[V]},
        {stopSequences:[new p.IdStopSequence({ids:[]})]}]) {
        const invalid=await refused(contextFor(configuration));assert.equal(invalid.report.status,p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);samplingCases++;
      }
    }
    results.push({name,status:'pass',paged:args.get('--paged')==='true',generatedTokens:6,resumptions:2,refusals,samplingCases});console.log(name+': PASS');
  } finally {await host.close();if(device){device.destroy();await device.lost;}}
}
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify({status:'pass',cases:results},null,2)+'\n');
