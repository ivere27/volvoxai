#!/usr/bin/env node
// Run the frozen F64 corpus in actual Chrome/Dawn, independently of wgpu-native.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { tmpdir } from 'node:os';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
for (const key of ['corpus', 'shader', 'chrome', 'out']) assert.ok(args.get(`--${key}`), `--${key} required`);
const corpusBytes = await readFile(args.get('--corpus')), shaderBytes = await readFile(args.get('--shader'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const pins = { corpus: {path:path.resolve(args.get('--corpus')), bytes:corpusBytes.length, sha256:sha(corpusBytes)},
  shader: {path:path.resolve(args.get('--shader')), bytes:shaderBytes.length, sha256:sha(shaderBytes)},
  helper: {path:import.meta.filename, sha256:sha(await readFile(import.meta.filename))} };
const page = `<!doctype html><script type="module">
const from64=text=>Uint8Array.from(atob(text), c=>c.charCodeAt(0));
const tensor=value=>{const bytes=from64(value.data);return value.dtype==='<f8'?new Float64Array(bytes.buffer):value.dtype==='<i4'?new Int32Array(bytes.buffer):new Float32Array(bytes.buffer);};
const hash=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
const compare=(actual, expected, atol,rtol)=>{
  let classification=true, finite=true, max=0, count=0;
  for(let i=0;i<expected.length;i++) {
    const a=actual[i], e=expected[i];
    if(Number.isNaN(e)){classification&&=Number.isNaN(a);continue;}
    if(!Number.isFinite(e)){classification&&=Object.is(a,e);continue;}
    count++;const error=Math.abs(a-e);max=Math.max(max,error);
    finite&&=Number.isFinite(a)&&error<=atol+rtol*Math.abs(e);
    classification&&=Number.isFinite(a);
  }
  return {pass:classification&&finite,classification_pass:classification,finite_pass:finite,finite_elements:count,maximum_absolute_error:Number.isFinite(max)?max:null};
};
let device, owned=[];
const receipt={format:'volvox-portable-numeric-observation/v1',runtime:'Chrome/Dawn',status:'FAIL',cases:[],physical_peak_qualified:false,model_accuracy_qualified:false};
try {
  const corpus=await (await fetch('/corpus')).json(), source=await (await fetch('/shader')).text();
  let adapter;receipt.adapter_requests=[];
  for(let attempt=0;attempt<5&&!adapter;attempt++) {
    adapter=await navigator.gpu.requestAdapter();
    receipt.adapter_requests.push({attempt,elapsed_ms:performance.now(),available:!!adapter});
    if(!adapter)await new Promise(resolve=>setTimeout(resolve,1000));
  }
  if(!adapter)throw new Error('No WebGPU adapter');
  const info=adapter.info, description=[info.vendor,info.architecture,info.device,info.description].filter(Boolean).join(' ');
  if(/swiftshader|llvmpipe|lavapipe|software|cpu/i.test(description))throw new Error('Hardware mode refuses software adapter: '+description);
  receipt.adapter={description,vendor:info.vendor,architecture:info.architecture,device:info.device};
  device=await adapter.requestDevice();
  device.pushErrorScope('validation');
  const module=device.createShaderModule({code:source});
  const messages=await module.getCompilationInfo();
  if(messages.messages.some(m=>m.type==='error'))throw new Error(JSON.stringify(messages.messages));
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  for(const c of corpus.cases) {
    const own=b=>{owned.push(b);return b;};
    const buffer=(bytes,usage)=>{const b=own(device.createBuffer({size:bytes.byteLength,usage,mappedAtCreation:true}));new Uint8Array(b.getMappedRange()).set(new Uint8Array(bytes.buffer,bytes.byteOffset,bytes.byteLength));b.unmap();return b;};
    const buffers=c.inputs.map(v=>buffer(from64(v.data),GPUBufferUsage.STORAGE));
    const count=c.output_shape.reduce((n,v)=>n*v,1), tail=tensor(corpus.output_tail);
    const initial=new Float32Array(count+tail.length);initial.fill(-98765.5);initial.set(tail,count);
    const out=buffer(initial,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);buffers.push(out);
    buffers.push(buffer(from64(c.uniform),GPUBufferUsage.UNIFORM));
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((b,i)=>({binding:i,resource:{buffer:b}}))});
    const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(...c.workgroups);pass.end();
    const readback=own(device.createBuffer({size:initial.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));
    encoder.copyBufferToBuffer(out,0,readback,0,initial.byteLength);device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();await readback.mapAsync(GPUMapMode.READ);
    const full=new Float32Array(readback.getMappedRange().slice(0));readback.unmap();
    let actual=full.slice(0,count);
    if(c.sampled_query_rows!==null) {
      const [batch,queries,heads,width]=c.output_shape, sampled=new Float32Array(batch*c.sampled_query_rows.length*heads*width);let at=0;
      for(let b=0;b<batch;b++)for(const row of c.sampled_query_rows){const first=(b*queries+row)*heads*width;sampled.set(actual.subarray(first,first+heads*width),at);at+=heads*width;}
      actual=sampled;
    }
    const expected=tensor(c.expected), check=compare(actual,expected,c.atol,c.rtol);
    const bits=new Uint32Array(full.buffer), tailBits=new Uint32Array(tail.buffer);
    const guard=tailBits.every((v,i)=>bits[count+i]===v);check.pass&&=guard;
    const entry={id:c.id,required:c.required,guard_pass:guard,output_sha256:await hash(full.slice(0,count).buffer),...check};
    if(c.hf_expected)entry.independent_hf=compare(actual,tensor(c.hf_expected),c.atol,c.rtol);
    receipt.cases.push(entry);owned.forEach(b=>b.destroy());owned=[];
  }
  const error=await device.popErrorScope();if(error)throw error;
  receipt.status=receipt.cases.every(c=>c.guard_pass&&(!c.required||c.pass))?'PASS':'FAIL';
  receipt.nonfinite_diagnostics_pass=receipt.cases.filter(c=>!c.required).every(c=>c.pass);
}catch(error){receipt.error=error.stack??String(error);}
finally{owned.forEach(b=>b.destroy());device?.destroy();window.__numericResult=receipt;}
</script>`;
const files = new Map([['/', ['text/html', page]], ['/corpus', ['application/json', corpusBytes]], ['/shader', ['text/plain', shaderBytes]]]);
const server = createServer((req,res)=>{const f=files.get(new URL(req.url,'http://localhost').pathname);if(!f){res.writeHead(404).end();return;}res.writeHead(200,{'content-type':f[0],'cache-control':'no-store','cross-origin-opener-policy':'same-origin','cross-origin-embedder-policy':'require-corp'}).end(f[1]);});
server.listen(0,'127.0.0.1');await once(server,'listening');
const profile=await mkdtemp(path.join(tmpdir(),'volvox-portable-numeric-'));
const child=spawn(args.get('--chrome'),['--headless=new','--no-sandbox','--disable-gpu-sandbox','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan','--disable-vulkan-surface','--remote-debugging-port=0',`--user-data-dir=${profile}`,`http://127.0.0.1:${server.address().port}`],{detached:true,stdio:['ignore','ignore','pipe']});
const exited=once(child,'exit');let socket, log='', rejectDeadline;
const deadline=new Promise((_,reject)=>{rejectDeadline=reject;});
const timeout=setTimeout(()=>rejectDeadline(new Error('Raw Chrome correctness exceeded 120 seconds: '+log)),120000);
child.stderr.on('data',bytes=>{log=(log+bytes).slice(-8000);});
let result;
try {
  const drive=async()=>{
    const endpoint=await new Promise((resolve,reject)=>{const inspect=()=>{const found=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(found)resolve(found[1]);};child.stderr.on('data',inspect);inspect();child.once('error',reject);child.once('exit',code=>reject(new Error('Chrome exited '+code+': '+log)));});
    const target=(await(await fetch(`http://${new URL(endpoint).host}/json/list`)).json()).find(v=>v.type==='page');assert.ok(target);
    socket=new WebSocket(target.webSocketDebuggerUrl);await once(socket,'open');
    let next=0;const pending=new Map();
    socket.addEventListener('message',event=>{const m=JSON.parse(event.data);if(m.id){pending.get(m.id)?.resolve(m);pending.delete(m.id);}});
    socket.addEventListener('close',()=>{for(const p of pending.values())p.reject(new Error('Chrome connection closed'));pending.clear();});
    const send=(method,params)=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
    for(;;){const response=await send('Runtime.evaluate',{expression:'window.__numericResult',returnByValue:true});result=response.result?.result?.value;if(result)return;await new Promise(resolve=>setTimeout(resolve,100));}
  };
  await Promise.race([drive(),deadline]);
}catch(error){result={status:'FAIL',error:error.stack??String(error),diagnostics:log};}
finally{clearTimeout(timeout);socket?.close();try{process.kill(-child.pid,'SIGKILL');}catch(error){if(error.code!=='ESRCH')throw error;}server.close();await exited;await rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:100});}
result={...result,...pins};
if(result.status!=='PASS')result.diagnostics=log;
if(sha(await readFile(args.get('--shader')))!==pins.shader.sha256){result.status='FAIL';result.error='Shader changed during worker';}
await writeFile(args.get('--out'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({status:result.status,cases:result.cases?.length,out:args.get('--out')}));
if(result.status!=='PASS')process.exitCode=1;
