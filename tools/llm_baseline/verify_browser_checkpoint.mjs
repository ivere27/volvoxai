#!/usr/bin/env node
/** Actual Chrome checkpoint qualification with task-owned profile and cleanup. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import {createReadStream,readFileSync,readdirSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createServer} from 'node:http';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';
import {readPackageVersion} from '../release_version.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const version=readPackageVersion(root);
const options={backend:'wasm',profile:'inference',out:path.join(root,'build/llm/remaining-validation-001/browser/inference-wasm.json'),chrome:'/usr/bin/google-chrome'};
for(let i=2;i<process.argv.length;i+=2){const key=process.argv[i].slice(2);assert.ok(Object.hasOwn(options,key),key);options[key]=process.argv[i+1];}
assert.ok(['wasm','webgpu'].includes(options.backend));assert.ok(['inference','full'].includes(options.profile));
assert.ok(options.backend==='wasm'||options.profile==='full');
options.out=path.resolve(options.out);await fs.mkdir(path.dirname(options.out),{recursive:true});
assert.equal(await fs.access(options.out).then(()=>true,()=>false),false,'use a new receipt');
const identity=async file=>{const hash=createHash('sha256');for await(const bytes of createReadStream(file))hash.update(bytes);return {path:file,bytes:(await fs.stat(file)).size,sha256:hash.digest('hex')};};
const referenceDir=path.join(root,'build/llm/reference-v2/smollm2-135m');
const reference=JSON.parse(await fs.readFile(path.join(referenceDir,'manifest.json'),'utf8'));
assert.equal(reference.status,'complete');
const graph=JSON.parse(await fs.readFile(path.join(root,'build/llm/c-stage/packages/smollm2-135m/execution/graph.json'),'utf8'));
const caches=Array.from({length:30},(_,i)=>[`layer.${i}.k_rope`,`layer.${i}.v`]).flat();
graph.outputs.push(...caches);for(const entry of graph.execution.entrypoints)entry.outputs.push(...caches);
const graphPath=options.out.replace(/\.json$/,'.graph.json');await fs.writeFile(graphPath,JSON.stringify(graph));
const bundlePath=path.join(root,'dist',version,options.profile==='inference'?'volvoxai.lite.js':'volvoxai.js');
const wasmPath=path.join(root,'dist',version,options.profile==='inference'?'volvoxai.lite.wasm':'volvoxai.wasm');
const weightsPath=path.join(root,'build/llm/packages/smollm2-135m/model.safetensors');
const paths={bundle:bundlePath,wasm:wasmPath,weights:weightsPath,graph:graphPath,
  helper:fileURLToPath(import.meta.url),browser_helper:path.join(root,'tools/llm_baseline/browser_checkpoint_case.mjs'),reference_manifest:path.join(referenceDir,'manifest.json')};
const pins=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([key,file])=>[key,await identity(file)])));
const cases=[];
for(const selected of reference.cases.filter(v=>v.prefix_length===16)) {
  const file=path.join(referenceDir,selected.reference_path);const pin=await identity(file);
  assert.equal(pin.sha256,selected.reference_sha256);cases.push({...selected,reference:{...pin,url:'/oracle/'+selected.id+'.safetensors'}});
}
const candidate={format:'volvox-browser-checkpoint-candidate/v1',backend:options.backend,profile:options.profile,model:'smollm2-135m',
  repo_id:reference.repo_id,revision:reference.revision,dtype:'F32',capacity:128,vocab:49152,layers:30,kv_heads:3,head_dim:64,
  cache_names:caches,cases,tolerance:reference.reference_policy.engine_tolerance,
  limits:{wasm_linear_bytes:2147418112,browser_tree_rss_bytes:4294967296,gpu_api_buffer_bytes:2147483648,post_release_allocator_growth_bytes:8388608},
  budget_scope:'Prospective candidate limits. Browser tree RSS is shared-page double-counted; PSS/RSS are sampled, not exact physical peak.',
  bundle:{...pins.bundle,url:'/runtime.js'},wasm:{...pins.wasm,url:'/runtime.wasm'},graph:{...pins.graph,url:'/graph.json'},weights:{...pins.weights,url:'/weights.safetensors'},pins};
const candidatePath=options.out.replace(/\.json$/,'.candidate.json');await fs.writeFile(candidatePath,JSON.stringify(candidate,null,2)+'\n');
const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'volvox-browser-checkpoint-'));
const served=new Map([['/runtime.js',bundlePath],['/runtime.wasm',wasmPath],['/graph.json',graphPath],['/weights.safetensors',weightsPath],
  ['/case.mjs',paths.browser_helper],...cases.map(v=>[v.reference.url,v.reference.path])]);
const receipt={format:'volvox-browser-checkpoint/v1',status:'running',candidate:await identity(candidatePath),pins,markers:[],samples:[],
  sampled_memory_scope:'Sum of Chrome task-owned process tree RSS/PSS/Swap from Linux procfs at nominal100ms. RSS overlaps shared pages; PSS is apportioned resident memory, not exact peak. Driver/server process is excluded.'};
let resultResolve;const resultPromise=new Promise(resolve=>{resultResolve=resolve;});
const server=createServer(async(request,response)=>{
  const url=new URL(request.url,'http://localhost').pathname;
  response.setHeader('cross-origin-opener-policy','same-origin');response.setHeader('cross-origin-embedder-policy','require-corp');response.setHeader('cache-control','no-store');
  if(url==='/'){response.setHeader('content-type','text/html');response.end('<!doctype html><title>Checkpoint qualification</title><script type="module" src="/case.mjs"></script>');return;}
  if(url==='/candidate.json'){response.setHeader('content-type','application/json');response.end(JSON.stringify(candidate));return;}
  if(url==='/marker'||url==='/result') {
    let body='';for await(const chunk of request)body+=chunk;const value=JSON.parse(body);
    if(url==='/marker'){receipt.markers.push({...value,at:Date.now()});sample();}else resultResolve(value);
    response.end('ok');return;
  }
  const file=served.get(url);if(!file){response.writeHead(404).end();return;}
  response.setHeader('content-type',url.endsWith('.wasm')?'application/wasm':url.endsWith('.js')||url.endsWith('.mjs')?'text/javascript':'application/octet-stream');
  response.setHeader('content-length',(await fs.stat(file)).size);createReadStream(file).pipe(response);
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const chromeArgs=['--headless=new','--no-sandbox','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--remote-debugging-port=0',`--user-data-dir=${temporary}`];
if(options.backend==='webgpu')chromeArgs.push('--disable-gpu-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan','--disable-vulkan-surface');
else chromeArgs.push('--disable-gpu');
chromeArgs.push(`http://127.0.0.1:${server.address().port}`);
const child=spawn(options.chrome,chromeArgs,{detached:true,stdio:['ignore','ignore','pipe']});
const childExit=once(child,'exit').catch(error=>[null,null,String(error)]);
let diagnostics='',socket;
child.stderr.on('data',chunk=>{diagnostics=(diagnostics+chunk).slice(-12000);});
let maximumRSS=0,maximumPSS=0,maximumSwap=0,maximumInterval=0,lastSample;
const observedProcesses=new Map();
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const processInfo=pid=>{
  const stat=readFileSync(`/proc/${pid}/stat`,'utf8'),fields=stat.slice(stat.lastIndexOf(')')+2).split(' ');
  return {pid,state:fields[0],ppid:Number(fields[1]),process_group:Number(fields[2]),starttime:fields[19]};
};
function sample(){
  if(!child.pid)return;
  const processes=[];
  for(const name of readdirSync('/proc'))if(/^\d+$/.test(name))try{
    processes.push(processInfo(Number(name)));
  }catch{}
  const selected=new Set([child.pid]);let changed=true;
  while(changed){changed=false;for(const item of processes)if(selected.has(item.ppid)&&!selected.has(item.pid)){selected.add(item.pid);changed=true;}}
  for(const item of processes)if(selected.has(item.pid))observedProcesses.set(item.pid,item);
  let rss=0,pss=0,swap=0,observed=0;
  for(const pid of selected)try{
    const memory=readFileSync(`/proc/${pid}/smaps_rollup`,'utf8');
    const read=key=>Number(memory.match(new RegExp('^'+key+':\\s+(\\d+)','m'))?.[1]??0)*1024;
    rss+=read('Rss');pss+=read('Pss');swap+=read('Swap');observed++;
  }catch{}
  const now=Date.now();if(lastSample)maximumInterval=Math.max(maximumInterval,now-lastSample);lastSample=now;
  maximumRSS=Math.max(maximumRSS,rss);maximumPSS=Math.max(maximumPSS,pss);maximumSwap=Math.max(maximumSwap,swap);
  receipt.samples.push({at:now,phase:receipt.markers.at(-1)?.phase??'startup',processes:observed,rss,pss,swap});
  if(rss>candidate.limits.browser_tree_rss_bytes)child.kill('SIGKILL');
}
const timer=setInterval(sample,100);
const abort=reason=>{receipt.abort_reason=reason;try{if(child.pid)process.kill(-child.pid,'SIGKILL');}catch{}child.kill('SIGKILL');};
const interrupt=()=>abort('task interrupted by signal');
process.on('SIGINT',interrupt);process.on('SIGTERM',interrupt);
const timeout=setTimeout(()=>abort('20 minute task timeout'),20*60*1000);
try{
  const endpoint=await new Promise((resolve,reject)=>{
    child.stderr.on('data',()=>{const found=diagnostics.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(found)resolve(found[1]);});
    child.once('error',reject);child.once('exit',code=>reject(Error('Chrome exited '+code+': '+diagnostics)));
  });
  socket=new WebSocket(endpoint);await once(socket,'open');
  let id=0;const pending=new Map();socket.addEventListener('message',event=>{const value=JSON.parse(event.data);const promise=pending.get(value.id);if(promise){pending.delete(value.id);if(value.error)promise.reject(Error(JSON.stringify(value.error)));else promise.resolve(value.result);}});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const next=++id;pending.set(next,{resolve,reject});socket.send(JSON.stringify({id:next,method,params}));});
  receipt.browser=await call('Browser.getVersion');
  const result=await Promise.race([resultPromise,childExit.then(([code])=>{throw Error('Chrome exited before result: '+code);})]);
  receipt.browser_result=result;assert.equal(result.status,'PASS',result.error);
  sample();assert.ok(receipt.samples.some(v=>v.processes>1&&v.pss>0),'physical browser process samples unavailable');
  assert.ok(maximumRSS<=candidate.limits.browser_tree_rss_bytes,'browser tree RSS budget exceeded');
  for(const[key,pin]of Object.entries(pins))assert.equal((await identity(paths[key])).sha256,pin.sha256,key+' changed during execution');
  receipt.status='PASS';
}catch(error){receipt.status='FAIL';receipt.error=error.stack??String(error);}
finally{
  clearInterval(timer);clearTimeout(timeout);socket?.close();sample();
  try{if(child.pid)process.kill(-child.pid,'SIGKILL');}catch{}
  const killed=[];
  for(const [pid,expected]of observedProcesses)try{
    const actual=processInfo(pid);
    if(actual.starttime===expected.starttime&&actual.state!=='Z'){process.kill(pid,'SIGKILL');killed.push(pid);}
  }catch{}
  child.kill('SIGKILL');await childExit;server.close();
  let alive=[],zombies=[];
  for(let attempt=0;attempt<100;attempt++) {
    alive=[];zombies=[];
    for(const [pid,expected]of observedProcesses)try{
      const actual=processInfo(pid);
      if(actual.starttime===expected.starttime){if(actual.state==='Z')zombies.push(pid);else alive.push(pid);}
    }catch{}
    if(!alive.length)break;await delay(50);
  }
  const before=(await fs.stat(temporary)).size;
  await fs.rm(temporary,{recursive:true,force:true,maxRetries:10,retryDelay:100});
  receipt.cleanup={task_owned_profile_removed:await fs.access(temporary).then(()=>false,()=>true),path:temporary,directory_entry_bytes_before:before,
    task_owned_process_group:child.pid,observed_processes:Array.from(observedProcesses.values()),killed_verified_pids:killed,
    live_processes_after:alive,zombies_after:zombies,process_tree_drained:alive.length===0};
  if(alive.length){receipt.status='FAIL';receipt.cleanup_error='Task-owned browser processes remained alive';}
  process.off('SIGINT',interrupt);process.off('SIGTERM',interrupt);
  receipt.memory={peak_sampled_tree_rss_bytes:maximumRSS,peak_sampled_tree_pss_bytes:maximumPSS,peak_sampled_tree_swap_bytes:maximumSwap,maximum_sample_interval_ms:maximumInterval,sample_count:receipt.samples.length};
  receipt.diagnostics=diagnostics;await fs.writeFile(options.out,JSON.stringify(receipt,null,2)+'\n');
}
console.log(JSON.stringify({status:receipt.status,receipt:options.out,memory:receipt.memory,error:receipt.error}));
if(receipt.status!=='PASS')process.exitCode=1;
