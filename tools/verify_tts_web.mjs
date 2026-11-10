#!/usr/bin/env node
/** Replay the shared TTS corpus over public WASM or WebGPU dispatch.
 * node tools/verify_tts_web.mjs --backend wasm --corpus ... --out ...
 * deno run --no-config --unstable-webgpu -A tools/verify_tts_web.mjs --backend webgpu --corpus ... --out ...
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { readPackageVersion } from './release_version.mjs';

const options = { backend: 'wasm' };
const argv = globalThis.Deno?.args ?? process.argv.slice(2);
for (let i=0;i<argv.length;i++) {
  assert.ok(['--backend','--corpus','--out'].includes(argv[i]));
  const key=argv[i].slice(2), value=argv[++i]; assert.ok(value); options[key]=value;
}
assert.ok(['wasm','webgpu'].includes(options.backend) && options.corpus && options.out);
const version=readPackageVersion(), results=[];
const bytes=value=>new Uint8Array(Buffer.from(value,'base64'));
const hash=value=>createHash('sha256').update(value).digest('hex');
const corpusBytes=await fs.readFile(options.corpus), cases=JSON.parse(corpusBytes).cases;
assert.ok(cases.length);
const receipt={backend:options.backend,corpus_sha256:hash(corpusBytes),artifacts:[],results,status:'FAIL'};
const originalFetch=globalThis.fetch;
globalThis.fetch=async (source,init)=> {
  const url=typeof source==='string'?source:source instanceof URL?source.href:source?.url;
  return url?.startsWith('file:')?new Response(await fs.readFile(fileURLToPath(url))):originalFetch(source,init);
};
try {
  if(options.backend==='webgpu') {
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    assert.ok(adapter,'WebGPU adapter required');
    const info=adapter.info??await adapter.requestAdapterInfo?.()??{};
    receipt.device=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,info[k]??'']));
    assert.doesNotMatch(Object.values(receipt.device).join(' '),/swiftshader|llvmpipe|lavapipe|software/i,'physical GPU required');
  }
  for(const profile of options.backend==='wasm'?['inference','full']:['full']) {
    const stem=profile==='inference'?'volvoxai.lite':'volvoxai';
    const bundle=path.resolve('dist',version,stem+'.js'), wasm=path.resolve('dist',version,stem+'.wasm');
    receipt.artifacts.push({profile,bundle_sha256:hash(await fs.readFile(bundle)),wasm_sha256:hash(await fs.readFile(wasm))});
    const api=await import(pathToFileURL(bundle).href), pb=api.pb;
    const host=new (api.FullEngineHost??api.EngineHost)({wasmUrl:pathToFileURL(wasm),onDiagnostic:message=>console.error(message)});
    const client=new api.VxInferenceServiceClient(host);
    const checked=response=>{assert.equal(response.report?.status??response.status,pb.NativeStatus.NATIVE_STATUS_OK,response.report?.message);return response;};
    const attest=report=>{
      assert.equal(report.backend,options.backend); assert.equal(report.route.attested,true);
      assert.equal(report.route.selectedNodes,report.route.activeNodes);
      assert.equal(report.route.fallbackNodes,0); assert.equal(report.route.missingNodes,0);
      assert.equal(report.fallback?.operatorFallbackUsed??false,false);
    };
    try {
      const runtime=checked(await client.createRuntime(new pb.CreateRuntimeRequest()));
      for(const fixture of cases) {
        const model=checked(await client.loadModel(new pb.LoadModelRequest({runtimeId:runtime.runtimeId,
          package:new pb.ModelPackage({graphDocument:new TextEncoder().encode(JSON.stringify(fixture.graph)),weightShards:fixture.weight_shards.map(bytes)})})));
        const compiled=checked(await client.compileModel(new pb.CompileModelRequest({modelId:model.modelId,
          policy:new pb.BackendPolicy({mode:pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[options.backend],operatorFallback:pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
        attest(compiled.report);
        const context=checked(await client.createExecutionContext(new pb.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,graph:new pb.GraphExecution()})));
        const request=new pb.ExecuteRequest({contextId:context.contextId,
          inputs:new pb.InputBindings({values:fixture.inputs.map(t=>new pb.Tensor({name:t.name,shape:t.shape.map(BigInt),dtype:t.dtype==='I32'?pb.DataType.DATA_TYPE_I32:pb.DataType.DATA_TYPE_F32,inline:bytes(t.data)}))}),
          scope:new pb.ExecutionScope({entrypoint:'evaluate',full:new pb.Empty()}),outputs:new pb.OutputOptions({storage:pb.OutputStorage.OUTPUT_STORAGE_INLINE})});
        let maximum=0;
        for(let repeat=0;repeat<2;repeat++) {
          const packet=checked(await client.execute(request)); attest(packet.report);
          for(const expected of fixture.expected) {
            const actual=packet.outputs.find(t=>t.name===expected.name);assert.ok(actual,fixture.name);
            assert.deepEqual(actual.shape,expected.shape.map(BigInt));
            assert.equal(actual.dtype,expected.dtype==='I32'?pb.DataType.DATA_TYPE_I32:pb.DataType.DATA_TYPE_F32);
            const raw=bytes(expected.data);assert.equal(actual.inline.length,raw.length);
            if(fixture.bitwise) assert.deepEqual(actual.inline,raw,`${fixture.name}: exact output bits`);
            const a=new DataView(actual.inline.buffer,actual.inline.byteOffset,actual.inline.length),b=new DataView(raw.buffer,raw.byteOffset,raw.length);
            for(let i=0;i<raw.length;i+=4) {
              const av=expected.dtype==='I32'?a.getInt32(i,true):a.getFloat32(i,true),bv=expected.dtype==='I32'?b.getInt32(i,true):b.getFloat32(i,true);
              if(!Number.isFinite(bv)) { assert.ok(Number.isNaN(bv)?Number.isNaN(av):av===bv,`${fixture.name}: ${av} vs ${bv}`);continue; }
              const error=Math.abs(av-bv);maximum=Math.max(maximum,error);
              assert.ok(Number.isFinite(av)&&error<=(expected.dtype==='I32'?0:1e-5+1e-5*Math.abs(bv)),`${fixture.name}/${i/4}: ${av} vs ${bv}`);
            }
          }
        }
        results.push({profile,name:fixture.name,max_error:maximum,pass:true});console.log('PASS',profile,options.backend,fixture.name);
        checked(await client.releaseExecutionContext(new pb.ExecutionContextRef({contextId:context.contextId})));
        checked(await client.releaseCompiledModel(new pb.CompiledModelRef({compiledModelId:compiled.compiledModelId})));
        checked(await client.releaseModel(new pb.ModelRef({modelId:model.modelId})));
      }
      checked(await client.releaseRuntime(new pb.RuntimeRef({runtimeId:runtime.runtimeId})));
    } finally {await host.close();}
  }
  receipt.status='PASS';
} finally {
  globalThis.fetch=originalFetch;
  await fs.mkdir(path.dirname(options.out),{recursive:true});await fs.writeFile(options.out,JSON.stringify(receipt,null,2)+'\n');
}
