/** Raw media caller qualification through the ordinary browser public API. */
import {whisperFeatures,receiptPixels,receiptQuestion} from '/media.mjs';
const config=await(await fetch('/candidate.json')).json();
const receipt={status:'running',backend:config.backend,profile:config.profile,mode:config.mode,diagnostic_only:!!config.diagnostic_only,qualification_claim:!config.diagnostic_only,shader_substitutions:[],
  cases:[],snapshots:[],transfers:[],routes:[],adapter_witnesses:[],device_buffers:[],timings:[],workflow_timings:[],
  memory_scope:'Exact WASM linear allocator partition and tracked GPU API capacities; bounded allocation inventory (128 rows, explicit truncation); external RSS/PSS/Swap sampling is separate.',
  memory_observer_policy:{allocation_rows_per_snapshot:128,correctness_only_compiled_inventory_rows:65536,decoder_snapshots:'first and final token; load, compile, encoder retirement and release boundaries also observed',rss_scope:'Includes the caller and retained bounded observer receipts.'},
  timing_scope:'Awaited public call and caller processor elapsed performance.now; memory observers and numerical oracle absent in timing mode. Cold/warm labels refer to a fresh host first/next workflow, not driver filesystem caches.'};
globalThis.__qualification=receipt;
const require=(value,message)=>{if(!value)throw Error(message);};
const plain=value=>JSON.parse(JSON.stringify(value,(_,v)=>typeof v==='bigint'?String(v):v));
const same=(a,b,message)=>require(JSON.stringify(a)===JSON.stringify(b),message);
const digest=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),v=>v.toString(16).padStart(2,'0')).join('');
const hex=value=>new Uint8Array(value.match(/../g).map(v=>parseInt(v,16)));
const marker=async phase=>{receipt.phase=phase;if(config.observe)await fetch('/marker',{method:'POST',body:JSON.stringify({phase,performance_now_ms:performance.now()})});};
const timed=async(label,action)=>{await marker(label+':begin');const start=performance.now();const value=await action();receipt.timings.push({phase:label,start_ms:start,end_ms:performance.now(),elapsed_ms:performance.now()-start});await marker(label+':end');return value;};
function compare(actual,expected,policy,label){
  require(actual.length===expected.length,label+' extent differs');let maximum=0,sum=0;
  for(let i=0;i<actual.length;i++){const delta=actual[i]-expected[i],error=Math.abs(delta);maximum=Math.max(maximum,error);sum+=delta*delta;
    require(Number.isFinite(actual[i])&&error<=policy.atol+policy.rtol*Math.abs(expected[i]),`${label}/${i}: ${actual[i]} != ${expected[i]}`);}
  const rms=Math.sqrt(sum/actual.length);if(policy.rms!==undefined)require(rms<=policy.rms,label+' RMS '+rms);
  return {elements:actual.length,maximum_absolute_error:maximum,rms_error:rms,...policy};
}
async function checkedFetch(source){const data=await(await fetch(source.url)).arrayBuffer();require(data.byteLength===source.bytes&&await digest(data)===source.sha256,'caller source identity differs '+source.url);return data;}
async function loadOracle(){
  const data=await checkedFetch(config.oracle),length=Number(new DataView(data).getBigUint64(0,true));
  const header=JSON.parse(new TextDecoder().decode(new Uint8Array(data,8,length)));
  receipt.independent_reference_arrays=Object.keys(header).filter(v=>v!=='__metadata__').length;
  return name=>{const row=header[name];require(row,'reference absent '+name);const offset=8+length+row.data_offsets[0],count=(row.data_offsets[1]-row.data_offsets[0])/4;
    require(row.dtype==='F32'||row.dtype==='I32','reference type differs '+name);return row.dtype==='F32'?new Float32Array(data,offset,count):new Int32Array(data,offset,count);};
}
const diagnosticReplacements=[];
if(config.shader_overrides?.length){
 require(config.diagnostic_only,'shader replacement is diagnostic only');const compact=text=>text.split(/\r?\n/).map(line=>line.trim()).filter(line=>line.length&&!line.startsWith('//')).join('\n');
 for(const entry of config.shader_overrides){const original=new TextDecoder().decode(await checkedFetch(entry.original)),replacement=new TextDecoder().decode(await checkedFetch(entry.replacement));
 const row={original_source:entry.original.sha256,replacement_source:entry.replacement.sha256,original_compact_sha256:await digest(new TextEncoder().encode(compact(original))),replacement_compact_sha256:await digest(new TextEncoder().encode(compact(replacement))),matched_calls:0};receipt.shader_substitutions.push(row);diagnosticReplacements.push({original:compact(original),replacement:compact(replacement),row});}
}
let originalRequestAdapter,host;
if(config.backend==='webgpu'){
  require(navigator.gpu,'navigator.gpu absent');originalRequestAdapter=navigator.gpu.requestAdapter;
  navigator.gpu.requestAdapter=async function(...args){
    const adapter=await originalRequestAdapter.apply(navigator.gpu,args);require(adapter,'WebGPU adapter absent');
    const info=adapter.info??await adapter.requestAdapterInfo?.()??{},witness=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info[k]]));
    const identity=Object.values(witness).filter(Boolean).join(' ');require(identity&&!/swiftshader|llvmpipe|lavapipe|softpipe|software|\bcpu\b/i.test(identity),'hardware adapter required '+identity);
    require(!(adapter.isFallbackAdapter??info.isFallbackAdapter??false),'fallback adapter refused');receipt.adapter_witnesses.push(witness);
    if(config.observe){const request=adapter.requestDevice;adapter.requestDevice=async function(...args){
      const device=await request.apply(adapter,args);if(diagnosticReplacements.length){const createShader=device.createShaderModule;device.createShaderModule=function(description){const selected=diagnosticReplacements.find(v=>v.original===description.code);if(selected){selected.row.matched_calls++;return createShader.call(device,{...description,code:selected.replacement});}return createShader.call(device,description);};}
      const observation={created:0,destroyed:0,live_bytes:0,peak_bytes:0,device_destroyed:false};receipt.device_buffers.push(observation);
      const create=device.createBuffer;device.createBuffer=function(description){const buffer=create.call(device,description);let live=true;
        observation.created++;observation.live_bytes+=buffer.size;observation.peak_bytes=Math.max(observation.peak_bytes,observation.live_bytes);
        require(observation.peak_bytes<=config.limits.gpu_api_buffer_bytes,'GPU buffer capacity exceeded');const destroy=buffer.destroy;
        buffer.destroy=function(){if(live){live=false;observation.destroyed++;observation.live_bytes-=buffer.size;}return destroy.call(buffer);};return buffer;};
      const destroy=device.destroy;device.destroy=function(){observation.live_bytes_before_device_destroy=observation.live_bytes;observation.device_destroyed=true;observation.live_bytes=0;return destroy.call(device);};return device;};}
    return adapter;
  };
}
try{
  const api=await import(config.bundle.url),p=api.pb;
  const checked=value=>{const report=value.report??value;require(report.status===0,report.message??'operation failed');return value;};
  const route=report=>{require(report.backend===config.backend&&report.route?.attested&&report.route.provider===config.backend,'selected backend differs');
    require(report.route.activeNodes>0&&report.route.activeNodes===report.route.selectedNodes&&report.route.missingNodes===0&&report.route.fallbackNodes===0&&!report.fallback?.operatorFallbackUsed,'route fallback or missing nodes');
    receipt.routes.push(plain(report.route));};
  const transport=async(input,init)=>{
    const response=await fetch(input,init);if(!config.observe||(!String(input).includes('/model.safetensors')&&!String(input).includes('/graph.json')))return response;
    const observation={url:String(input),received_bytes:0,chunks:0,maximum_chunk_bytes:0,ended:false,cancelled:false};receipt.transfers.push(observation);const reader=response.body.getReader();
    return new Response(new ReadableStream({async pull(controller){const {value,done}=await reader.read();if(done){observation.ended=true;controller.close();return;}
      observation.received_bytes+=value.byteLength;observation.chunks++;observation.maximum_chunk_bytes=Math.max(observation.maximum_chunk_bytes,value.byteLength);controller.enqueue(value);},
      async cancel(reason){observation.cancelled=true;await reader.cancel(reason);}}),{status:response.status,headers:response.headers});
  };
  host=new(config.profile==='inference'?api.EngineHost:api.FullEngineHost)({wasmUrl:new URL(config.wasm.url,location.href),fetch:transport});
  const client=new api.VxInferenceServiceClient(host),buffers=new api.VxBufferServiceClient(host),profiling=new api.VxProfilingServiceClient(host);
  const snapshot=async(phase,maxAllocations=128)=>{await marker(phase);if(!config.observe)return;
    const response=checked(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest({module:new p.Empty(),maxAllocations}))),value=plain(response.snapshot);
    receipt.snapshots.push({phase,snapshot:value,gpu_api_live_bytes:receipt.device_buffers.reduce((n,v)=>n+v.live_bytes,0),
      gpu_api_devices:receipt.device_buffers.map(({live_bytes,peak_bytes})=>({live_bytes,peak_bytes}))});
    require(Number(value.wasm.linearBytes)<=config.limits.wasm_linear_bytes,'WASM capacity exceeded');return value;};
  const runtime=checked(await timed('runtime_create',()=>client.createRuntime(new p.CreateRuntimeRequest({cpuThreads:1}))));
  const baseline=await snapshot('before_models'),models=new Map(),compiled=new Map(),contexts=new Set(),owned=new Set(),held=[];
  const tensor=(name,values,shape,dtype)=>new p.Tensor({name,shape:shape.map(BigInt),dtype:dtype==='F32'?p.DataType.DATA_TYPE_F32:p.DataType.DATA_TYPE_I32,inline:new Uint8Array(values.buffer,values.byteOffset,values.byteLength)});
  const f=(values,shape)=>({values:Float32Array.from(values),shape,dtype:'F32'}),i=(values,shape)=>({values:Int32Array.from(values),shape,dtype:'I32'});
  const tensors=feed=>Object.entries(feed).map(([name,v])=>tensor(name,v.values,v.shape,v.dtype));
  const view=value=>({values:value.dtype===p.DataType.DATA_TYPE_F32?new Float32Array(value.inline.slice().buffer):new Int32Array(value.inline.slice().buffer),shape:value.shape.map(Number),dtype:value.dtype===p.DataType.DATA_TYPE_F32?'F32':'I32'});
  const context=async(role,autoregressive)=>{const value=checked(await client.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.get(role),...(autoregressive?{autoregressive}:{graph:new p.GraphExecution()})})));contexts.add(value.contextId);return value.contextId;};
  const releaseContext=async id=>{checked(await client.releaseExecutionContext(new p.ExecutionContextRef({contextId:id})));contexts.delete(id);};
  const readOwned=async outputs=>{if(outputs.every(v=>v.inline!==undefined))return outputs;const copied=checked(await buffers.copyTensors(new p.CopyTensorsRequest({sources:outputs,inlineResult:true})));return copied.outputs;};
  const releaseOutputs=async outputs=>{const ids=outputs.filter(v=>v.buffer).map(v=>v.buffer.bufferId);if(ids.length)checked(await buffers.releaseBuffers(new p.BufferRefs({bufferIds:ids})));for(const id of ids)owned.delete(id);};
  const execute=async(label,id,feed,{entrypoint='evaluate',reuse=[],ownedOutput=false}={})=>{
    const packet=checked(await timed(label,()=>client.execute(new p.ExecuteRequest({contextId:id,inputs:new p.InputBindings({values:tensors(feed),reuse}),
      scope:new p.ExecutionScope({entrypoint,full:new p.Empty()}),outputs:new p.OutputOptions({storage:ownedOutput&&config.backend==='wasm'?p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER:p.OutputStorage.OUTPUT_STORAGE_INLINE})}))));route(packet.report);
    if(ownedOutput&&config.backend==='wasm')for(const value of packet.outputs)owned.add(value.buffer.bufferId);return packet;};
  for(const [role,packageInfo]of Object.entries(config.packages)){
    const graph=packageInfo.graph,weights=packageInfo.weights;
    const model=checked(await timed(role+':load',()=>client.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:new URL(graph.url,location.href).href,weightPaths:[new URL(weights.url,location.href).href],
      sourceOptions:new p.ModelSourceOptions({maxTotalBytes:BigInt(graph.bytes+weights.bytes),maxShardBytes:BigInt(weights.bytes),graph:new p.ModelSourceIntegrity({byteLength:BigInt(graph.bytes),sha256:hex(graph.sha256)}),weightShards:[new p.ModelSourceIntegrity({byteLength:BigInt(weights.bytes),sha256:hex(weights.sha256)})]})}))));models.set(role,model.modelId);
    await snapshot(role+':loaded');
    const result=checked(await timed(role+':compile',()=>client.compileModel(new p.CompileModelRequest({modelId:model.modelId,policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[config.backend],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})}))));
    route(result.report);compiled.set(role,result.compiledModelId);await snapshot(role+':compiled');
  }
  if(config.mode==='correctness')await snapshot('all_models_compiled_inventory',65536);
  const oracle=config.mode==='correctness'?await loadOracle():null;receipt.oracle_loaded=!!oracle;
  const processor=JSON.parse(new TextDecoder().decode(await checkedFetch(config.processor))).audio.processor_config;
  const vocabulary=JSON.parse(new TextDecoder().decode(await checkedFetch(config.vocabulary)));
  for(let round=0;round<config.rounds;round++){
    const category=round===0?'cold':round<=2?'warmup':'measured',label=category+'-'+round,audio=config.corpus.audio;
    const audioWallStart=performance.now(),audioTimingStart=receipt.timings.length;
    const pcm=new Float32Array(await checkedFetch(config.pcm));require(await digest(pcm)===audio.pcm_sha256,'PCM identity differs');
    const features=await timed(label+':audio:prepare',()=>whisperFeatures(pcm,processor));
    const observation={kind:'raw-PCM-whisper',round,label,checks:{},generated_ids:[],encoder_execute_calls:1,conditioning_handoff:config.backend==='webgpu'?'asynchronous Execute GPU readback -> independent portable INLINE conditioning -> Generate upload':'HOST OWNED encoder snapshot directly bound to Generate',conditioning_device_to_host_bytes:config.backend==='webgpu'?2304000:0};
    if(oracle)observation.checks.features=compare(features.values,oracle('audio_features'),audio.tolerances.features,'audio features');
    pcm.fill(0);const encoder=await context('audio_encoder');
    if(config.backend==='webgpu'&&oracle&&!round){
      const before=plain(checked(await client.getExecutionContextInfo(new p.ExecutionContextRef({contextId:encoder}))));delete before.report;
      const refused=await client.execute(new p.ExecuteRequest({contextId:encoder,inputs:new p.InputBindings({values:tensors({input_features:features})}),scope:new p.ExecutionScope({entrypoint:'evaluate',full:new p.Empty()}),outputs:new p.OutputOptions({storage:p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER})})).then(value=>value,error=>({report:error.report??error.response?.report,error:String(error)}));
      require(refused.report?.status===p.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED,'WebGPU OWNED_BUFFER must be refused before retry');const after=plain(checked(await client.getExecutionContextInfo(new p.ExecutionContextRef({contextId:encoder}))));delete after.report;
      same(before,after,'unsupported OWNED output mutated context before INLINE retry');receipt.webgpu_owned_output_refusal={...plain(refused),before,after,context_unchanged:true};
    }
    const packet=await execute(label+':audio:encoder',encoder,{input_features:features},{ownedOutput:true});features.values.fill(0);
    const original=packet.outputs[0],copied=(await timed(label+':audio:conditioning_copy',()=>readOwned(packet.outputs)))[0],memory=view(copied),memoryHash=await digest(copied.inline);
    if(!round)held.push({tensor:copied,hash:memoryHash});
    if(oracle)observation.checks.encoder=compare(memory.values,oracle('audio_memory'),audio.tolerances.encoder,'audio encoder');
    await releaseContext(encoder);await snapshot(label+':audio:encoder_retired');
    const initial=prefix=>({token_ids:i(prefix,[1,prefix.length]),position_ids:i(prefix.map((_,j)=>j),[prefix.length]),attention_mask:i(prefix.map(()=>1),[1,prefix.length]),encoder_memory:memory});
    if(oracle){const decoder=await context('audio_decoder');let prefix=audio.prompt_ids.slice();observation.checks.logits=[];
      for(let step=0;step<audio.expected_ids.length;step++){const output=await execute(label+':audio:reference_decode:'+step,decoder,initial(prefix));
        const logits=output.outputs.find(v=>v.name==='logits');observation.checks.logits.push(compare(view(logits).values,oracle('audio_logits_'+step),audio.tolerances.logits,'audio logits '+step));
        prefix.push(audio.expected_ids[step]);held.push({tensor:logits,hash:await digest(logits.inline)});}await releaseContext(decoder);}
    const generated=await context('audio_decoder',new p.AutoregressiveExecution({sampling:p.SamplingMode.SAMPLING_MODE_GREEDY,maxSteps:3n,eosIds:[audio.eos_id]}));
    const input=tensors(initial(audio.prompt_ids));if(config.backend==='wasm'){input.pop();input.push(new p.Tensor({...original,name:'encoder_memory'}));}
    for(let index=0;index<2;index++){
      const output=checked(await timed(label+':audio:generate:'+index,()=>client.generate(new p.GenerateRequest({contextId:generated,initialInputs:index?undefined:new p.GenerationInputs({values:input}),budget:new p.GenerationBudget({maxSteps:index?2:1}),outputs:new p.OutputOptions({storage:p.OutputStorage.OUTPUT_STORAGE_INLINE})}))));route(output.report);
      const ids=output.outputs.find(v=>v.name==='generated_ids');require(ids,'generated IDs missing');observation.generated_ids.push(...view(ids).values);held.push({tensor:ids,hash:await digest(ids.inline)});memory.values.fill(0);
      if(!output.progress.canContinue)break;
    }
    same(observation.generated_ids,audio.expected_ids,'Whisper greedy IDs differ');require(await digest((await readOwned([original]))[0].inline)===memoryHash,'independent encoder snapshot changed after mutations/owner release');
    await releaseContext(generated);receipt.cases.push(observation);await snapshot(label+':audio:finished');
    await releaseOutputs(packet.outputs);
    receipt.workflow_timings.push({workflow:'audio',round,category,wall_elapsed_ms:performance.now()-audioWallStart,stage_sum_ms:receipt.timings.slice(audioTimingStart).reduce((sum,v)=>sum+v.elapsed_ms,0),wall_scope:'raw fetch/integrity, caller processor, encoder, conditioning copy, generation, required IDs/lifetimes; model load/compile excluded'});
    const contracts=config.corpus.image.contracts,mapping=contracts.decoder.inputs,invariants=Object.keys(mapping).filter(name=>name.startsWith('cross_')||['family_ids','memory_padding_mask'].includes(name));
    for(const selected of config.corpus.image.cases){
      const imageWallStart=performance.now(),imageTimingStart=receipt.timings.length;
      const jpeg=await checkedFetch(selected.jpeg_source),pixels=await timed(label+':'+selected.key+':prepare',()=>receiptPixels(jpeg));
      // Decoder RGB->L parity is fixed prospectively by the original pixels hash.
      require(await digest(pixels.values)===selected.pixels_sha256,selected.key+' browser JPEG pixels differ from pinned Pillow processor');
      const question=receiptQuestion(selected.question,vocabulary);same(question,selected.question_ids,'caller question IDs differ');
      const ec=await context('image_encoder'),encoderFeed={image:pixels,question_ids:i(question,[1,question.length]),question_position_ids:i(question.map((_,j)=>j),[1,question.length]),family_ids:i([-1],[1])};
      const output=await execute(label+':'+selected.key+':encoder',ec,Object.fromEntries(Object.entries(encoderFeed).map(([name,value])=>[contracts.encoder.inputs[name],value])),{ownedOutput:true});
      const encodedOutputs=await timed(label+':'+selected.key+':conditioning_copy',()=>readOwned(output.outputs)),byName=new Map(encodedOutputs.map(value=>[value.name,value])),encoded={},encoderChecks={},hashes={};
      for(const [name,tensorName]of Object.entries(contracts.encoder.outputs)){const value=byName.get(tensorName);encoded[name]=view(value);hashes[tensorName]=await digest(value.inline);if(!round)held.push({tensor:value,hash:hashes[tensorName]});
        if(oracle){if(encoded[name].dtype==='F32')encoderChecks[name]=compare(encoded[name].values,oracle(selected.key+'_encoder_'+name),config.corpus.image.tolerances,selected.key+' encoder '+name);else same(Array.from(encoded[name].values),Array.from(oracle(selected.key+'_encoder_'+name)),selected.key+' integer encoder '+name);}}
      await releaseContext(ec);pixels.values.fill(0);await snapshot(label+':'+selected.key+':encoder_retired');
      const feed=Object.fromEntries(Object.entries(encoded).filter(([name])=>name.startsWith('cross_')||name==='memory_padding_mask'));
      Object.assign(feed,{decoder_input_ids:i([1],[1,1]),position_ids:i([0],[1]),family_ids:encoded.selected_family_ids,past_padding_mask:i([1],[1,1])});
      for(let layer=0;layer<4;layer++)for(const kind of ['k','v'])feed[`past_${kind}_${layer}`]=f(new Float32Array(320),[1,8,1,40]);
      const dc=await context('image_decoder'),observed={kind:'raw-JPEG-vqa',key:selected.key,round,label,jpeg_decoder:pixels.decoder,pixels_exact_pinned_processor:true,question_ids_exact:true,encoder:encoderChecks,logits:[],generated_ids:[],encoder_execute_calls:1,conditioning_snapshot_and_reuse:true};
      for(let step=0;step<selected.expected_ids.length;step++){
        const bound=Object.fromEntries(Object.entries(feed).filter(([name])=>!step||!invariants.includes(name)).map(([name,value])=>[mapping[name],value]));
        const packet=await execute(label+':'+selected.key+':decode:'+step,dc,bound,{entrypoint:step?'transition':'initialize',reuse:step?invariants.map(name=>mapping[name]):[]}),outputs=new Map(packet.outputs.map(value=>[value.name,value]));
        const actual=Object.fromEntries(Object.entries(contracts.decoder.outputs).map(([name,tensorName])=>[name,view(outputs.get(tensorName))]));
        if(oracle)observed.logits.push(compare(actual.logits.values,oracle(selected.key+'_logits_'+step),config.corpus.image.tolerances,selected.key+' logits '+step));
        let token=0;for(let j=1;j<actual.logits.values.length;j++)if(actual.logits.values[j]>actual.logits.values[token])token=j;
        require(token===selected.expected_ids[step],selected.key+' greedy token differs at '+step);observed.generated_ids.push(token);
        const past=feed.past_padding_mask.shape[1];same(actual.present_padding_mask.shape,[1,past+1],'KV mask shape differs');
        for(let j=0;j<past;j++)require(actual.present_padding_mask.values[j]===feed.past_padding_mask.values[j],'KV mask prefix changed');require(actual.present_padding_mask.values[past]===0,'KV new mask differs');
        for(let layer=0;layer<4;layer++)for(const kind of ['k','v']){const before=feed[`past_${kind}_${layer}`],after=actual[`present_${kind}_${layer}`];same(after.shape,[1,8,past+1,40],'KV shape differs');
          for(let head=0;head<8;head++)for(let pos=0;pos<past;pos++)for(let d=0;d<40;d++)require(after.values[(head*(past+1)+pos)*40+d]===before.values[(head*past+pos)*40+d],'KV prefix changed');feed[`past_${kind}_${layer}`]=after;}
        feed.past_padding_mask=actual.present_padding_mask;feed.decoder_input_ids=i([token],[1,1]);feed.position_ids=i([step+1],[1]);
        if(!step){for(const name of invariants)feed[name].values.fill(0);const logits=outputs.get(contracts.decoder.outputs.logits);held.push({tensor:logits,hash:await digest(logits.inline)});}
        if(!step||step===selected.expected_ids.length-1)await snapshot(label+':'+selected.key+':decode:'+step);
      }
      await releaseContext(dc);const copies=await readOwned(output.outputs);for(const value of copies)require(await digest(value.inline)===hashes[value.name],'independent VQA encoder output mutated');
      await releaseOutputs(output.outputs);
      receipt.cases.push(observed);await snapshot(label+':'+selected.key+':finished');
      receipt.workflow_timings.push({workflow:selected.key,round,category,wall_elapsed_ms:performance.now()-imageWallStart,stage_sum_ms:receipt.timings.slice(imageTimingStart).reduce((sum,v)=>sum+v.elapsed_ms,0),wall_scope:'raw fetch/integrity, caller processor, encoder, conditioning copy, cached graph decode, required IDs/cache/lifetimes; model load/compile excluded'});
    }
  }
  require(!contexts.size&&!owned.size,'workflow contexts or owned buffers still live');
  for(const id of compiled.values())checked(await client.releaseCompiledModel(new p.CompiledModelRef({compiledModelId:id})));
  for(const id of models.values())checked(await client.releaseModel(new p.ModelRef({modelId:id})));
  const released=await snapshot('models_and_contexts_released');
  if(config.observe){require(!released.memory.truncated,'released owner inventory must be complete');require(Number(released.wasm.allocatedBytes)<=Number(baseline.wasm.allocatedBytes)+config.limits.post_release_allocator_growth_bytes,'model-sized allocator storage survived release');}
  checked(await client.releaseRuntime(new p.RuntimeRef({runtimeId:runtime.runtimeId})));await snapshot('runtime_released');
  await timed('host_close',()=>host.close());host=null;
  for(const value of held)require(await digest(value.tensor.inline)===value.hash,'retained INLINE output changed after all owner retirement');
  receipt.retained_inline_outputs=held.length;receipt.inline_survives_all_owner_retirement=true;
  if(config.backend==='webgpu'&&config.observe)require(receipt.device_buffers.length&&receipt.device_buffers.every(v=>v.live_bytes===0&&v.device_destroyed&&v.live_bytes_before_device_destroy===0&&v.created===v.destroyed),'GPU owner remained live');
  require(receipt.shader_substitutions.every(v=>v.matched_calls>0),'diagnostic shader was never matched');receipt.status='PASS';await marker('host_closed');
}catch(error){receipt.status='FAIL';receipt.error=error.stack??String(error);}
finally{
  if(host)try{await host.close();}catch(error){receipt.cleanup_error=String(error);}
  if(originalRequestAdapter)navigator.gpu.requestAdapter=originalRequestAdapter;
  globalThis.__qualification=plain(receipt);await fetch('/result',{method:'POST',body:JSON.stringify(globalThis.__qualification)});
}
