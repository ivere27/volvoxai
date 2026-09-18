#!/usr/bin/env node
/** Launch an isolated Chrome profile and verify packaged MV3 WASM + restart. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const {version}=JSON.parse(await fs.readFile(path.join(ROOT,'package.json'),'utf8'));
const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'volvoxai-mv3-'));
const extension=path.join(temporary,'extension'), profile=path.join(temporary,'chrome');
await fs.mkdir(extension); await fs.mkdir(profile);
for(const file of ['volvoxai.lite.js','volvoxai.lite.wasm','volvoxai.min.js','volvoxai.wasm'])
  await fs.copyFile(path.join(ROOT,'dist',version,file),path.join(extension,file));
await fs.writeFile(path.join(extension,'manifest.json'),JSON.stringify({manifest_version:3,name:'VolvoxAI packaged WASM verification',version:'1.0',
  permissions:['storage'],background:{service_worker:'worker.js',type:'module'},
  content_security_policy:{extension_pages:"script-src 'self' 'wasm-unsafe-eval'; object-src 'self'"}}));
await fs.writeFile(path.join(extension,'page.html'),'<!doctype html><title>VolvoxAI MV3 verification</title>');
await fs.writeFile(path.join(extension,'graph.json'),JSON.stringify({format:'volvox-graph/v1',dimensions:{},
  inputs:{x:{dtype:'float32',shape:[1,2]}},nodes:[{id:'relu',opType:'ReLU',inputs:{input:'x'},
    outputs:{out:{tensor:'y',dtype:'float32',shape:[1,2]}},params:{}}],outputs:['y']}));
await fs.writeFile(path.join(extension,'worker.js'),`
import * as inferenceProfile from './volvoxai.lite.js';
import * as fullProfile from './volvoxai.min.js';
const requireOK=value=>{const r=value.report??value;if(r.status!==0)throw Error(r.message);return value;};
globalThis.verification=(async()=>{
  const previous=(await chrome.storage.local.get('state')).state??{generation:0,handles:{}};
  const current={generation:previous.generation+1,handles:{},profiles:[]};
  for(const [name,api,Host] of [['inference',inferenceProfile,inferenceProfile.EngineHost],['full',fullProfile,fullProfile.FullEngineHost]]) {
    const host=new Host(); const p=api.pb; const client=new api.VxInferenceServiceClient(host);
    const runtime=requireOK(await client.createRuntime(new p.CreateRuntimeRequest()));
    current.handles[name]=String(runtime.runtimeId);
    if(previous.handles[name]) {
      if(previous.handles[name]===String(runtime.runtimeId))throw Error('restarted owner reused a public handle');
      let rejected = false;
      try {
        await client.loadModel(new p.LoadModelRequest({runtimeId:BigInt(previous.handles[name]),graphPath:'unreachable.json'}));
      } catch (error) {
        if (!(error instanceof api.VolvoxAIError) || error.report.status !== p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED) throw error;
        rejected = true;
      }
      if (!rejected) throw Error('stale runtime handle was accepted');
    }
    const model=requireOK(await client.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:new URL('graph.json',import.meta.url).href})));
    const compiled=requireOK(await client.compileModel(new p.CompileModelRequest({modelId:model.modelId,
      policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:['wasm']})})));
    const context=requireOK(await client.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
    const result=requireOK(await client.execute(new p.ExecuteRequest({contextId:context.contextId,inputs:[new p.Tensor({name:'x',shape:[1n,2n],dtype:p.DataType.DATA_TYPE_F32,inline:new Uint8Array(Float32Array.of(-2,3).buffer)})]})));
    const output=requireOK(await client.readOutput(new p.ReadOutputRequest({resultId:result.resultId,name:'y'}))).tensor.inline;
    const values=new Float32Array(output.buffer.slice(output.byteOffset,output.byteOffset+output.byteLength));
    if(values[0]!==0||values[1]!==3)throw Error('packaged C execution produced incorrect output');
    current.profiles.push(name); await host.close();
  }
  await chrome.storage.local.set({state:current}); return current;
})();
chrome.runtime.onMessage.addListener((_message,_sender,respond)=>{verification.then(value=>respond({value}),error=>respond({error:String(error)}));return true;});
`);
const chrome=spawn(process.env.VOLVOXAI_CHROME??'google-chrome',[
  '--headless=new','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check',
  '--enable-unsafe-extension-debugging','--remote-debugging-port=0',`--user-data-dir=${profile}`,
],{stdio:['ignore','ignore','pipe']});
let stderr='';chrome.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-6000);});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let socket, id=0; const pending=new Map();
const call=(method,params={},sessionId)=>new Promise((resolve,reject)=>{
  const requestId=++id; const timeout=setTimeout(()=>{pending.delete(requestId);reject(Error('CDP timeout: '+method));},25000);
  pending.set(requestId,{resolve:v=>{clearTimeout(timeout);resolve(v);},reject:e=>{clearTimeout(timeout);reject(e);}});
  socket.send(JSON.stringify({id:requestId,method,params,...(sessionId?{sessionId}:{})}));
});
try {
  let port;
  for(let attempt=0;attempt<100;attempt++) {
    try {port=(await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n');break;} catch {await delay(50);}
  }
  assert.ok(port,stderr);
  socket=new WebSocket('ws://127.0.0.1:'+port[0]+port[1]);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  socket.onmessage=event=>{const message=JSON.parse(event.data);const entry=pending.get(message.id);if(!entry)return;pending.delete(message.id);
    if(message.error)entry.reject(Error(JSON.stringify(message.error)));else entry.resolve(message.result);};
  const loaded=await call('Extensions.loadUnpacked',{path:extension});
  const extensionId=loaded.id;
  assert.ok(extensionId,JSON.stringify(loaded));
  const page=await call('Target.createTarget',{url:'chrome-extension://'+extensionId+'/page.html'});
  const attached=await call('Target.attachToTarget',{targetId:page.targetId,flatten:true});
  const evaluate=async expression=>{
    const response=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true},attached.sessionId);
    assert.ok(!response.exceptionDetails,JSON.stringify(response.exceptionDetails));return response.result.value;
  };
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await evaluate('typeof chrome.runtime?.sendMessage === "function"')) break;
    if (attempt === 99) throw Error('extension page did not initialize');
    await delay(50);
  }
  const wake=()=>evaluate('chrome.runtime.sendMessage("verify")');
  const first=await wake(); assert.ok(!first.error,first.error); assert.equal(first.value.generation,1);
  assert.deepEqual(first.value.profiles,['inference','full']);
  const targets=await call('Target.getTargets');
  const worker=targets.targetInfos.find(t=>t.type==='service_worker'&&t.url.startsWith('chrome-extension://'+extensionId+'/'));
  assert.ok(worker,'extension worker target missing');
  const closed=await call('Target.closeTarget',{targetId:worker.targetId});assert.equal(closed.success,true);
  await delay(100);
  const second=await wake();assert.ok(!second.error,second.error);assert.equal(second.value.generation,2);
  assert.notEqual(first.value.handles.inference,second.value.handles.inference);
  assert.notEqual(first.value.handles.full,second.value.handles.full);
  console.log('MV3 packaged WASM/CSP: PASS (inference + minified full; worker restart rejects stale handles)');
} catch(error) {
  process.stderr.write(stderr+'\n');throw error;
} finally {
  if(socket?.readyState===WebSocket.OPEN) {try{await call('Browser.close');}catch{}socket.close();}
  chrome.kill('SIGTERM');
  await delay(200);
  await fs.rm(temporary,{recursive:true,force:true,maxRetries:10,retryDelay:100});
}
