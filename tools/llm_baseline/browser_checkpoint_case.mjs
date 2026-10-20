/** Public RPC replay inside an actual browser; imported only by the qualifier. */
const config = await (await fetch('/candidate.json')).json();
const receipt = {status:'running', backend:config.backend, profile:config.profile,
  cases:[], snapshots:[], transfers:[], routes:[], adapter_witnesses:[], device_buffers:[],
  memory_scope:'Public WASM heap and tracked GPU API buffer capacities; external browser process memory is sampled by the driver.'};
globalThis.__qualification = receipt;
addEventListener('unhandledrejection',event=>{
  receipt.status='FAIL';receipt.error=event.reason?.stack??String(event.reason);
  fetch('/result',{method:'POST',body:JSON.stringify(receipt)});
});
const require = (value,message) => { if(!value) throw Error(message); };
const same = (a,b,message) => require(JSON.stringify(a)===JSON.stringify(b),message);
const plain = value => JSON.parse(JSON.stringify(value,(_,v)=>typeof v==='bigint'?String(v):v));
const marker = async phase => { receipt.phase=phase; await fetch('/marker',{method:'POST',body:JSON.stringify({phase})}); };
const hex = value => new Uint8Array(value.match(/../g).map(v=>parseInt(v,16)));
const digest = async value => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value)),v=>v.toString(16).padStart(2,'0')).join('');
const bytes = values => new Uint8Array(Int32Array.from(values).buffer);
const f32 = value => new Float32Array(value.slice().buffer);
function compare(actual,expected,limits,label) {
  require(actual.length===expected.length,`${label}: tensor extent differs`);
  let maximum=0,sum=0;
  for(let i=0;i<actual.length;i++) {
    const error=Math.abs(actual[i]-expected[i]);maximum=Math.max(maximum,error);sum+=error*error;
    require(Number.isFinite(actual[i])&&error<=limits.absolute+limits.relative*Math.abs(expected[i]),
      `${label}/${i}: ${actual[i]} != ${expected[i]}`);
  }
  const rms=Math.sqrt(sum/actual.length);
  if(limits.rms_max!==undefined)require(rms<=limits.rms_max,`${label}: RMS ${rms}`);
  return {elements:actual.length,maximum_error:maximum,rms};
}
async function oracle(source) {
  const buffer=await(await fetch(source.url)).arrayBuffer();
  require(buffer.byteLength===source.bytes&&await digest(buffer)===source.sha256,'independent oracle identity differs');
  const length=Number(new DataView(buffer).getBigUint64(0,true));
  const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,length)));
  return name=>{
    const row=header[name];require(row,`reference missing ${name}`);
    require(row.dtype==='F32',`reference ${name} dtype differs`);
    return new Float32Array(buffer,8+length+row.data_offsets[0],(row.data_offsets[1]-row.data_offsets[0])/4);
  };
}
let originalRequestAdapter;
if(config.backend==='webgpu') {
  require(navigator.gpu,'browser navigator.gpu is unavailable');
  originalRequestAdapter=navigator.gpu.requestAdapter;
  navigator.gpu.requestAdapter=async function(...args) {
    const adapter=await originalRequestAdapter.apply(navigator.gpu,args);require(adapter,'no WebGPU adapter');
    const info=adapter.info??await adapter.requestAdapterInfo?.()??{};
    const witness=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info[k]]));
    const identity=Object.values(witness).filter(Boolean).join(' ');
    require(identity&&!/swiftshader|llvmpipe|lavapipe|softpipe|software|\bcpu\b/i.test(identity),'hardware adapter required: '+identity);
    require(!(adapter.isFallbackAdapter??info.isFallbackAdapter??false),'fallback adapter refused');
    receipt.adapter_witnesses.push(witness);
    const request=adapter.requestDevice;
    adapter.requestDevice=async function(...deviceArgs) {
      const device=await request.apply(adapter,deviceArgs);
      const observation={created:0,destroyed:0,live_bytes:0,peak_bytes:0,device_destroyed:false};
      receipt.device_buffers.push(observation);
      const create=device.createBuffer;
      device.createBuffer=function(description) {
        const buffer=create.call(device,description);let live=true;
        observation.created++;observation.live_bytes+=buffer.size;
        observation.peak_bytes=Math.max(observation.peak_bytes,observation.live_bytes);
        require(observation.peak_bytes<=config.limits.gpu_api_buffer_bytes,'GPU API buffer budget exceeded');
        const destroy=buffer.destroy;
        buffer.destroy=function(){if(live){live=false;observation.destroyed++;observation.live_bytes-=buffer.size;}return destroy.call(buffer);};
        return buffer;
      };
      const destroy=device.destroy;
      device.destroy=function(){observation.device_destroyed=true;observation.live_bytes=0;return destroy.call(device);};
      return device;
    };
    return adapter;
  };
}
let host;
try {
  const api=await import(config.bundle.url), p=api.pb;
  const checked=value=>{const report=value.report??value;require(report.status===0,report.message??'public operation failed');return value;};
  const route=report=>{
    require(report.backend===config.backend,'required backend changed');
    require(report.route?.attested&&report.route.activeNodes>0&&report.route.selectedNodes===report.route.activeNodes,
      'active graph route not fully attested');
    require(report.route.missingNodes===0&&report.route.fallbackNodes===0&&!report.fallback?.operatorFallbackUsed,'fallback route used');
    receipt.routes.push(plain(report.route));
  };
  const transport=async(input,init)=>{
    const response=await fetch(input,init);
    if(!String(input).includes('/weights.safetensors')&&!String(input).includes('/graph.json'))return response;
    const observation={url:String(input),received_bytes:0,chunks:0,maximum_chunk_bytes:0,ended:false,cancelled:false};
    receipt.transfers.push(observation);
    const reader=response.body.getReader();
    const body=new ReadableStream({async pull(controller){
      const {value,done}=await reader.read();
      if(done){observation.ended=true;controller.close();return;}
      observation.chunks++;observation.received_bytes+=value.byteLength;
      observation.maximum_chunk_bytes=Math.max(observation.maximum_chunk_bytes,value.byteLength);controller.enqueue(value);
    },async cancel(reason){observation.cancelled=true;await reader.cancel(reason);}});
    return new Response(body,{status:response.status,headers:response.headers});
  };
  host=new (config.profile==='inference'?api.EngineHost:api.FullEngineHost)({wasmUrl:new URL(config.wasm.url,location.href),fetch:transport});
  const client=new api.VxInferenceServiceClient(host), profiling=new api.VxProfilingServiceClient(host);
  const snapshot=async phase=>{
    const response=checked(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest({module:new p.Empty(),maxAllocations:65536})));
    const value=plain(response.snapshot);receipt.snapshots.push({phase,snapshot:value});
    require(Number(value.wasm.linearBytes)<=config.limits.wasm_linear_bytes,'WASM linear budget exceeded');
    await marker(phase);return value;
  };
  const runtime=checked(await client.createRuntime(new p.CreateRuntimeRequest({cpuThreads:1})));
  const baseline=await snapshot('before_model');
  const model=checked(await client.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,
    graphPath:new URL(config.graph.url,location.href).href,weightPaths:[new URL(config.weights.url,location.href).href],
    sourceOptions:new p.ModelSourceOptions({maxTotalBytes:BigInt(config.graph.bytes+config.weights.bytes),
      maxShardBytes:BigInt(config.weights.bytes),graph:new p.ModelSourceIntegrity({byteLength:BigInt(config.graph.bytes),sha256:hex(config.graph.sha256)}),
      weightShards:[new p.ModelSourceIntegrity({byteLength:BigInt(config.weights.bytes),sha256:hex(config.weights.sha256)})]})})));
  await snapshot('loaded');
  const compiled=checked(await client.compileModel(new p.CompileModelRequest({modelId:model.modelId,
    policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[config.backend],
      operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
  route(compiled.report);await snapshot('compiled');
  const capacity=config.capacity,positions=Array.from({length:capacity},(_,i)=>i);
  const tensor=(name,data,shape)=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_I32,shape:shape.map(BigInt),inline:bytes(data)});
  const inputs=(ids,active,initial)=>[
    tensor('token_ids',[...ids,...Array(capacity-ids.length).fill(0)],[1,capacity]),
    tensor('position_ids',positions,[capacity]),tensor('attention_mask',positions.map(i=>i<ids.length?1:0),[1,capacity]),
    ...(initial?[tensor('active_lengths',[active],[1])]:[])];
  const retain=[];
  for(const selected of config.cases) {
    await marker(selected.id+':oracle');const ref=await oracle(selected.reference);
    const context=checked(await client.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
      graph:new p.GraphExecution({slots:1,rowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true})})));
    const observation={id:selected.id,graph:[],generation:[],immutable_outputs:true};receipt.cases.push(observation);
    const ids=selected.input_token_ids.slice();let previous;
    try {
      for(let step=0;step<4;step++) {
        if(step)ids.push(selected.greedy_ids[step-1]);
        const row=ids.length-1,prefix=step?'decode.'+String(step-1).padStart(3,'0'):'prefill';
        const packet=checked(await client.execute(new p.ExecuteRequest({contextId:context.contextId,
          inputs:new p.InputBindings({values:inputs(ids,selected.prefix_length,step===0),reuse:step?['active_lengths']:[]}),
          scope:new p.ExecutionScope(step?{entrypoint:'transition',slotRows:new p.SlotRows({slots:[new p.SlotRow({position:row})]})}:{entrypoint:'initialize',full:new p.Empty()}),
          outputs:new p.OutputOptions({storage:p.OutputStorage.OUTPUT_STORAGE_INLINE,
            selection:new p.TensorOutputSelection({names:['logits',...config.cache_names]})})})));
        route(packet.report);
        const outputs=new Map(packet.outputs.map(v=>[v.name,v]));
        const logits=outputs.get('logits'), actual=f32(logits.inline).subarray(row*config.vocab,(row+1)*config.vocab);
        const expected=ref(prefix+'.logits'),expectedRow=step?expected:expected.subarray(row*config.vocab,(row+1)*config.vocab);
        const logitCheck=compare(actual,expectedRow,config.tolerance.logits,selected.id+'/'+prefix+'/logits');
        const cacheChecks=[];
        for(let layer=0;layer<config.layers;layer++)for(const [suffix,key] of [['k_rope','key'],['v','value']]) {
          const compact=f32(outputs.get(`layer.${layer}.${suffix}`).inline),expectedCache=ref(prefix+`.layer.${String(layer).padStart(3,'0')}.${key}`);
          const ordered=new Float32Array(ids.length*config.kv_heads*config.head_dim);
          for(let head=0;head<config.kv_heads;head++)for(let position=0;position<ids.length;position++)for(let d=0;d<config.head_dim;d++)
            ordered[(head*ids.length+position)*config.head_dim+d]=compact[(position*config.kv_heads+head)*config.head_dim+d];
          cacheChecks.push({layer,kind:key,...compare(ordered,expectedCache,config.tolerance.layer_kv,selected.id+'/'+prefix+'/'+layer+'/'+key)});
        }
        if(previous)for(let byte=0;byte<row*config.vocab*4;byte++)
          require(logits.inline[byte]===previous[byte],'previous cached logits changed');
        previous=logits.inline.slice();
        if(step===0)retain.push({tensor:logits,hash:await digest(logits.inline)});
        observation.graph.push({phase:prefix,row,logits:logitCheck,caches:cacheChecks,snapshot_output_bytes:packet.outputs.reduce((n,v)=>n+v.inline.byteLength,0)});
        await snapshot(selected.id+':'+prefix);
      }
    }finally{checked(await client.releaseExecutionContext(new p.ExecutionContextRef({contextId:context.contextId})));}
    for(const chunks of [[4],[1,1,2]]) {
      const owner=checked(await client.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        autoregressive:new p.AutoregressiveExecution({sampling:p.SamplingMode.SAMPLING_MODE_GREEDY,maxSteps:4n})})));
      const generated=[];let index=0;
      try {
        for(const count of chunks) {
          const packet=checked(await client.generate(new p.GenerateRequest({contextId:owner.contextId,
            initialInputs:index++===0?new p.GenerationInputs({values:inputs(selected.input_token_ids,selected.prefix_length,true)}):undefined,
            budget:new p.GenerationBudget({maxSteps:count}),outputs:new p.OutputOptions({storage:p.OutputStorage.OUTPUT_STORAGE_INLINE})})));
          route(packet.report);const output=packet.outputs.find(v=>v.name==='generated_ids');require(output,'missing generated IDs');
          generated.push(...new Int32Array(output.inline.slice().buffer));
        }
        same(generated,selected.greedy_ids,'generated IDs differ from independent HF');
        observation.generation.push({chunks,ids:generated});
      }finally{checked(await client.releaseExecutionContext(new p.ExecutionContextRef({contextId:owner.contextId})));}
      await snapshot(selected.id+':generate-'+chunks.join('-'));
    }
  }
  checked(await client.releaseCompiledModel(new p.CompiledModelRef({compiledModelId:compiled.compiledModelId})));
  checked(await client.releaseModel(new p.ModelRef({modelId:model.modelId})));
  const released=await snapshot('model_and_contexts_released');
  require(Number(released.wasm.allocatedBytes)<=Number(baseline.wasm.allocatedBytes)+config.limits.post_release_allocator_growth_bytes,
    'engine allocator retained model-sized storage after release');
  checked(await client.releaseRuntime(new p.RuntimeRef({runtimeId:runtime.runtimeId})));
  await snapshot('runtime_released');await host.close();host=null;
  for(const held of retain)require(await digest(held.tensor.inline)===held.hash,'INLINE output changed after owner release');
  receipt.retained_inline_outputs=retain.length;
  if(config.backend==='webgpu')require(receipt.device_buffers.length>0&&receipt.device_buffers.every(v=>v.device_destroyed&&v.live_bytes===0),'GPU bridge did not release its device');
  receipt.status='PASS';await marker('host_closed');
}catch(error){receipt.status='FAIL';receipt.error=error.stack??String(error);}
finally{
  if(host)try{await host.close();}catch(error){receipt.cleanup_error=String(error);}
  if(originalRequestAdapter)navigator.gpu.requestAdapter=originalRequestAdapter;
  globalThis.__qualification=plain(receipt);
  await fetch('/result',{method:'POST',body:JSON.stringify(globalThis.__qualification)});
}
