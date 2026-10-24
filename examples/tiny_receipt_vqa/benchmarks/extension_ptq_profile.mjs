/** Headless Chrome WASM profiling in one isolated page, without extension UI/workers.
 * Inputs/spec are prepared by extension_ptq_profile.py. Use --runtime-dir to
 * freeze a baseline release and --puppeteer-root for the extension dependency.
 */
import {createServer} from 'node:http';
import {readFile,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
const options={};
for(let i=2;i<process.argv.length;i+=2)options[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['spec','runtime-dir','puppeteer-root','out'])if(!options[key])throw Error(`--${key} required`);
const hash=data=>createHash('sha256').update(data).digest('hex');
const specBytes=await readFile(options.spec),specSha256=hash(specBytes);
const spec=JSON.parse(specBytes);
const callerSha256=hash(await readFile(fileURLToPath(import.meta.url)));
const inputFileSha256={};
for(const inputs of Object.values(spec.inputs))for(const input of inputs){
 const digest=hash(await readFile(input.file));
 if(digest!==input.sha256)throw Error(`Input bytes do not match spec: ${input.file}`);
 inputFileSha256[input.file]=digest;
}
const previous=options['refresh-profiles']?JSON.parse(await readFile(options['refresh-profiles'])):null;
if(previous){
 if(previous.spec_sha256!==specSha256)throw Error('Profiling refresh changed or lacks original spec binding');
 spec.profilesOnly=true;
 if(options['profile-model'])spec.models=Object.fromEntries(Object.entries(spec.models).filter(([name])=>name===options['profile-model']));
}
const {default:puppeteer}=await import(pathToFileURL(path.join(options['puppeteer-root'],'node_modules/puppeteer/lib/puppeteer/puppeteer.js')));
const routes=new Map(),assetUrls=new Map();let serial=0;
const asset=file=>{const resolved=path.resolve(file);if(assetUrls.has(resolved))return assetUrls.get(resolved);const url=`/asset/${serial++}/${path.basename(file)}`;routes.set(url,resolved);assetUrls.set(resolved,url);return url;};
const bundle='/volvoxai.lite.js',wasm='/volvoxai.lite.wasm';
if(spec.vocab){spec.vocabUrl=asset(spec.vocab);spec.tokenizerUrl=asset(spec.tokenizer);}
routes.set(bundle,path.join(options['runtime-dir'],'volvoxai.lite.min.js'));
routes.set(wasm,path.join(options['runtime-dir'],'volvoxai.lite.wasm'));
const runtimeSha256=Object.fromEntries(await Promise.all([bundle,wasm].map(async name=>[name,hash(await readFile(routes.get(name)))])));
for(const inputs of Object.values(spec.inputs))for(const input of inputs)input.url=asset(input.file);
for(const variants of Object.values(spec.models))for(const model of Object.values(variants))for(const role of Object.values(model.roles)){
 role.graph=asset(path.join(role.path,'graph.json'));role.weights=(role.weight_paths??[path.join(role.path,'model.safetensors')]).map(asset);
}
const server=createServer(async(req,res)=>{
 try{
  if(req.url==='/'){res.writeHead(200,{'Content-Type':'text/html'});res.end('<!doctype html><meta charset="utf-8"><title>Isolated receipt kernel profile</title>');return;}
  const file=routes.get(req.url);if(!file){res.writeHead(404);res.end();return;}
  const bytes=await readFile(file);
  res.writeHead(200,{'Content-Type':file.endsWith('.js')?'text/javascript':req.url.endsWith('.wasm')?'application/wasm':'application/octet-stream'});
  res.end(bytes);
 }catch(e){res.writeHead(500);res.end(String(e));}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try{
 browser=await puppeteer.launch({headless:true,protocolTimeout:600000,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage();page.on('console',msg=>console.log(msg.text()));
 await page.goto(`http://127.0.0.1:${server.address().port}/`);
 let report=await page.evaluate(async(spec)=>{
 try {
  const api=await import('/volvoxai.lite.js');
  const {pb,EngineHost,VxInferenceServiceClient,VxProfilingServiceClient}=api;
  const tokenizer=spec.vocabUrl?await (await import(spec.tokenizerUrl)).TinyReceiptByteFallbackBPEVocab.fromJSON(await(await fetch(spec.vocabUrl)).json(),{pad:0,bos:1,eos:2,unk:3}):null;
  const result={label:spec.label,backend:'wasm',cpu_threads:1,models:{},feedback:spec.feedback,case_index:spec.case_index,
   timing_boundary:'direct generated Execute inline request/result transport; no UI, tabs or shared-worker queue',
   cold_definition:'new host/model/context setup plus first request; module and OS cache may be warm'};
  const tensor=(shape,data)=>({shape:shape.map(BigInt),dtype:data instanceof Float32Array?pb.DataType.DATA_TYPE_F32:pb.DataType.DATA_TYPE_I32,
   inline:new Uint8Array(data.buffer,data.byteOffset,data.byteLength)});
  const i32=(shape,values)=>tensor(shape,new Int32Array(values));
  const cross=Array.from({length:4},(_,i)=>[`cross_k_${i}`,`cross_v_${i}`]).flat();
  const past=cross.map(n=>n.replace('cross','past')),present=cross.map(n=>n.replace('cross','present'));
  const stats=values=>{const sorted=[...values].sort((a,b)=>a-b);return{n:values.length,mean_ms:values.reduce((a,b)=>a+b)/values.length,
   median_ms:(sorted[Math.floor(sorted.length/2)]+sorted[Math.floor((sorted.length-1)/2)])/2,p95_ms:sorted[Math.ceil(sorted.length*.95)-1],min_ms:sorted[0],max_ms:sorted.at(-1),samples_ms:values};};
  for(const [kind,variants] of Object.entries(spec.models)){
   result.models[kind]={};
   const initial={};
   for(const item of spec.inputs[kind]){
    const buffer=await(await fetch(item.url)).arrayBuffer();
    initial[item.name]=tensor(item.shape,item.dtype==='float32'?new Float32Array(buffer):new Int32Array(buffer));
   }
   for(const [variant,model] of Object.entries(variants)){
    const setup=performance.now(),host=new EngineHost({wasmUrl:'/volvoxai.lite.wasm'});
    const client=new VxInferenceServiceClient(host),profiling=new VxProfilingServiceClient(host);
    const entry=result.models[kind][variant]={profiles:{},packed_arithmetic:model.arithmetic,package_bytes_unique:model.package_bytes_unique};
    try{
     console.log('setup',kind,variant);
     const runtime=await client.createRuntime(new pb.CreateRuntimeRequest({cpuThreads:1}));
     const contexts={};
     for(const [role,contract] of Object.entries(model.roles)){
      const loaded=await client.loadModel(new pb.LoadModelRequest({runtimeId:runtime.runtimeId,
       graphPath:new URL(contract.graph,location.href).href,weightPaths:contract.weights.map(url=>new URL(url,location.href).href)}));
      const expectedArithmetic=pb.PackedArithmetic['PACKED_ARITHMETIC_'+model.arithmetic.toUpperCase().replaceAll('-','_')];
      if(expectedArithmetic===undefined)throw Error('Unknown packed arithmetic');
      const compiled=await client.compileModel(new pb.CompileModelRequest({modelId:loaded.modelId,
       packedArithmetic:expectedArithmetic,
       policy:new pb.BackendPolicy({mode:pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:['wasm'],operatorFallback:pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      if(compiled.report.backend!=='wasm')throw Error('Unexpected backend');
      if(compiled.packedArithmetic!==expectedArithmetic)throw Error('Unexpected packed arithmetic');
      contexts[role]=(await client.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,graph:new pb.GraphExecution()}))).contextId;
     }
     entry.setup_ms=performance.now()-setup;
     entry.context_ids=Object.fromEntries(Object.entries(contexts).map(([name,id])=>[name,String(id)]));
     let observed=[],verification=null;
     async function run(role,values,position=0){
      const contract=model.roles[role].inputs;
      let reuse=[],feedback=[];
      const names=role==='digit'?['slot_logits']:role==='encoder'?['memory_padding_mask','selected_family_ids',...cross]:
       spec.feedback?['logits']:['logits','present_padding_mask',...present];
      if(spec.feedback&&role==='decoder'&&position){
       reuse=[...cross,'memory_padding_mask','family_ids'].map(n=>contract[n]);
       feedback=past.map((name,i)=>new pb.TensorFeedback({inputName:contract[name],outputName:present[i]}));
       feedback.push(new pb.TensorFeedback({inputName:contract.past_padding_mask,outputName:'present_padding_mask'}));
       values=Object.fromEntries(['decoder_input_ids','position_ids'].map(n=>[n,values[n]]));
      }
      const t=performance.now();
      const packet=await client.execute(new pb.ExecuteRequest({contextId:contexts[role],
       scope:new pb.ExecutionScope({entrypoint:'evaluate',full:new pb.Empty()}),
       inputs:new pb.InputBindings({values:Object.entries(values).map(([name,v])=>new pb.Tensor({name:contract[name]??name,...v})),reuse,feedback}),
       outputs:new pb.OutputOptions({selection:new pb.TensorOutputSelection({names}),storage:pb.OutputStorage.OUTPUT_STORAGE_INLINE})}));
      const outputs=Object.fromEntries(packet.outputs.map(v=>[v.name,{dtype:v.dtype,shape:v.shape,inline:v.inline}]));
      observed.push({role,position,ms:performance.now()-t,input_bytes:Object.values(values).reduce((s,v)=>s+v.inline.byteLength,0),
       output_bytes:Object.values(outputs).reduce((s,v)=>s+v.inline.byteLength,0)});
      if(verification){
       const hashes={};for(const [name,value] of Object.entries(outputs))hashes[name]=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',value.inline)),v=>v.toString(16).padStart(2,'0')).join('');
       verification.push({role,position,outputs:hashes});
      }
      return outputs;
     }
     function argmax(array){let best=0;for(let i=1;i<array.length;i++)if(array[i]>array[best])best=i;return best;}
     async function request(){
      observed=[];const t=performance.now();let signature;
      if(kind==='digit'){
       const output=await run('digit',initial),logits=new Float32Array(output.slot_logits.inline.slice().buffer);
       signature=Array.from({length:16},(_,i)=>argmax(logits.subarray(i*11,i*11+11)));
      }else{
       const encoded=await run('encoder',initial),feed={memory_padding_mask:encoded.memory_padding_mask,family_ids:encoded.selected_family_ids,
        past_padding_mask:i32([1,1],[1]),...Object.fromEntries(cross.map(name=>[name,encoded[name]]))};
       for(const name of past)feed[name]=tensor([1,8,1,40],new Float32Array(320));
       signature=[1];
       for(let position=0;position<191;position++){
        feed.decoder_input_ids=i32([1,1],[signature.at(-1)]);feed.position_ids=i32([1],[position]);
        const output=await run('decoder',feed,position),token=argmax(new Float32Array(output.logits.inline.slice().buffer));
        signature.push(token);if(token===2)break;
        if(!spec.feedback){feed.past_padding_mask=output.present_padding_mask;past.forEach((name,i)=>{feed[name]=output[present[i]];});}
       }
       if(signature.at(-1)!==2)throw Error('No EOS');
      }
      return{total_ms:performance.now()-t,calls:observed,signature};
     }
     entry.cold=await request();
     if(!spec.profilesOnly){
     for(let i=0;i<spec.warmup;i++)await request();
     entry.warm=[];for(let i=0;i<spec.repeat;i++)entry.warm.push(await request());
     if(entry.warm.some(row=>JSON.stringify(row.signature)!==JSON.stringify(entry.cold.signature)))throw Error('Unstable output');
     entry.total=stats(entry.warm.map(r=>r.total_ms));
     entry.components=Object.fromEntries(Object.keys(model.roles).map(role=>[role,stats(entry.warm.map(r=>r.calls.filter(c=>c.role===role).reduce((s,c)=>s+c.ms,0)))]));
     verification=[];await request();entry.output_sha256=verification;verification=null;
     }
     const trace=await profiling.startTrace(new pb.StartTraceRequest({runtimeId:runtime.runtimeId,
      options:new pb.TraceOptions({detail:pb.TraceDetail.TRACE_DETAIL_NODES,memory:true,executionPlans:true,capacityBytes:33554432n})}));
     entry.profile_request=await request();
     const info=await profiling.stopTrace(new pb.TraceRef({traceId:trace.traceId}));
     entry.profiles.info=info.toJson();
     for(const [name,group] of Object.entries({operators:'OPERATOR',nodes:'NODE',activities:'ACTIVITY',operations:'OPERATION'})){
      entry.profiles[name]=(await profiling.getTraceSummary(new pb.GetTraceSummaryRequest({traceId:trace.traceId,
       groupBy:pb.TraceSummaryGrouping[`TRACE_SUMMARY_GROUPING_${group}`],maxRows:4096}))).toJson();
     }
     await profiling.releaseTrace(new pb.TraceRef({traceId:trace.traceId}));
     if(kind==='vqa'){
      entry.text=tokenizer.decode(entry.cold.signature);entry.correct=entry.text.match(/<answer>([^<>]*)<\/answer>/)?.[1]===spec.expected[kind];
     }else{
      const digits=values=>values.slice(0,values.includes(10)?values.indexOf(10):values.length).join('');
      const slots=entry.cold.signature;entry.correct=digits(slots.slice(0,12))===spec.expected[kind].phone&&digits(slots.slice(12))===spec.expected[kind].street;
     }
     if(!entry.correct)throw Error(`${kind}/${variant}: public example accuracy failure`);
     console.log(kind,variant,spec.profilesOnly?'profile refreshed':entry.total.median_ms);
    }finally{await host.close();}
   }
  }
  return result;
 } catch(error) { return {error:String(error),stack:error?.stack}; }
 },spec);
 if(report?.error)throw Error(JSON.stringify(report));
 report.browser=await browser.version();
 for(const [name,digest] of Object.entries(runtimeSha256))if(hash(await readFile(routes.get(name)))!==digest)throw Error(`Runtime changed during capture: ${name}`);
 for(const [file,digest] of Object.entries(inputFileSha256))if(hash(await readFile(file))!==digest)throw Error(`Input changed during capture: ${file}`);
 if(hash(await readFile(options.spec))!==specSha256)throw Error('Spec changed during capture');
 report.runtime_sha256=runtimeSha256;report.spec_sha256=specSha256;
 report.input_file_sha256=inputFileSha256;report.caller_sha256=callerSha256;
 if(previous){
  if(JSON.stringify(previous.runtime_sha256)!==JSON.stringify(report.runtime_sha256))throw Error('Profiling refresh changed runtime artifacts');
  for(const [kind,variants] of Object.entries(report.models))for(const [variant,item] of Object.entries(variants)){
   const target=previous.models[kind][variant];
   if(JSON.stringify(target.cold.signature)!==JSON.stringify(item.cold.signature))throw Error('Profiling refresh changed output signature');
   Object.assign(target,{profiles:item.profiles,context_ids:item.context_ids,profile_request:item.profile_request,
    profile_refresh:{reason:'Complete context-preserving node summary; operator summary aggregates all contexts',runtime_sha256:report.runtime_sha256,spec_sha256:specSha256,caller_sha256:callerSha256}});
  }
  report=previous;
 }
 await writeFile(options.out,JSON.stringify(report,null,2)+'\n');
}finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
