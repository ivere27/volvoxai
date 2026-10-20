import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {EngineHost} from '../ts/host/EngineHost.js';
import {FullEngineHost} from '../ts/full.js';
import {VxInferenceServiceClient} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as p from '../runtime/generated/typescript/inference/volvoxai_lite.js';
import {RpcError} from '../runtime/generated/typescript/inference/synurang_runtime.js';
import {declareStateless} from './parity/external/execution_fixture.mjs';

const {version}=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
const graph=new TextEncoder().encode(JSON.stringify(declareStateless({format:'volvox-graph/v1',dimensions:{},
  inputs:{x:{dtype:'float32',shape:[1]}},nodes:[],outputs:['x']})));
const digest=bytes=>new Uint8Array(createHash('sha256').update(bytes).digest());
const integrity=bytes=>new p.ModelSourceIntegrity({byteLength:BigInt(bytes.byteLength),sha256:digest(bytes)});
const streams=bytes=>new ReadableStream({start(controller){
  for(let offset=0;offset<bytes.byteLength;offset+=13)controller.enqueue(bytes.subarray(offset,offset+13));controller.close();
}});

/* The opt-in boundary fixture retains one transport block, never a weight
 * file or model-sized JavaScript array. Its tensor is valid F32 SafeTensors;
 * loading checks the source without compiling this deliberately wide graph. */
function largeWeightFixture() {
  const elements=16*1024*1024+1,payloadBytes=elements*4;
  const json=new TextEncoder().encode(JSON.stringify({parameter:{dtype:'F32',shape:[elements,1],data_offsets:[0,payloadBytes]}}));
  const headerBytes=Math.ceil(json.byteLength/8)*8;
  const prefix=new Uint8Array(8+headerBytes);
  new DataView(prefix.buffer).setBigUint64(0,BigInt(headerBytes),true);
  prefix.fill(0x20,8);prefix.set(json,8);
  const zeroBlock=new Uint8Array(1024*1024),hash=createHash('sha256');
  hash.update(prefix);
  for(let offset=0;offset<payloadBytes;offset+=zeroBlock.byteLength)
    hash.update(zeroBlock.subarray(0,Math.min(zeroBlock.byteLength,payloadBytes-offset)));
  const byteLength=prefix.byteLength+payloadBytes,sha256=new Uint8Array(hash.digest());
  const document=new TextEncoder().encode(JSON.stringify(declareStateless({format:'volvox-graph/v1',dimensions:{},
    inputs:{x:{dtype:'float32',shape:[1,elements]}},nodes:[{id:'projection',opType:'MatMul',
      inputs:{input:'x',weight:'parameter'},outputs:{out:{tensor:'y',dtype:'float32',shape:[1,1]}},params:{}}],outputs:['y']})));
  return {document,integrity:()=>new p.ModelSourceIntegrity({byteLength:BigInt(byteLength),sha256}),
    response:()=>{let offset=0;return {ok:true,headers:{get:name=>name==='content-length'?String(byteLength):null},
      body:new ReadableStream({pull(controller){
        if(offset===byteLength){controller.close();return;}
        if(offset===0){offset=prefix.byteLength;controller.enqueue(prefix);return;}
        const count=Math.min(zeroBlock.byteLength,byteLength-offset);offset+=count;controller.enqueue(zeroBlock.subarray(0,count));
      }},{highWaterMark:0}),arrayBuffer:async()=>{throw new Error('large source buffering is forbidden');}};}};
}

for(const [profile,Host] of [['inference',EngineHost],['full',FullEngineHost]]) {
  const wasmUrl=new URL(`../dist/${version}/volvoxai${profile==='inference'?'.lite':''}.wasm`,import.meta.url);
  test(`${profile}: inline graph fetches only BF16 shards and checks its manifest before fetch`,async()=>{
    const encoded=new TextEncoder().encode(JSON.stringify({offset:{dtype:'BF16',shape:[1],data_offsets:[0,2]}}));
    const header=new Uint8Array(Math.ceil(encoded.byteLength/8)*8);header.fill(0x20);header.set(encoded);
    const shard=new Uint8Array(8+header.byteLength+2);
    new DataView(shard.buffer).setBigUint64(0,BigInt(header.byteLength),true);
    shard.set(header,8);shard.set([0xc0,0x3f],8+header.byteLength); // BF16 1.5
    const document=new TextEncoder().encode(JSON.stringify(declareStateless({format:'volvox-graph/v1',dimensions:{},
      inputs:{x:{dtype:'float32',shape:[1]}},nodes:[{id:'sum',opType:'Add',inputs:{a:'x',b:'offset'},
        outputs:{out:{tensor:'y',dtype:'float32',shape:[1]}},params:{}}],outputs:['y']})));
    const fetched=[];
    const host=new Host({wasmUrl,fetch:async source=>{
      fetched.push(source);assert.equal(source,'weights.safetensors');return {ok:true,body:streams(shard)};
    },resolveModelSource:(graphPath,weightPaths)=>{
      assert.equal(graphPath,'');return {graphUrl:'',weightSources:weightPaths};
    }});
    const inference=new VxInferenceServiceClient(host);
    try {
      const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
      const sourceOptions=new p.ModelSourceOptions({maxTotalBytes:BigInt(document.byteLength+shard.byteLength),
        maxShardBytes:BigInt(Math.max(document.byteLength,shard.byteLength)),graph:integrity(document),weightShards:[integrity(shard)]});
      const request={runtimeId:runtime.runtimeId,graphDocument:document,weightPaths:['weights.safetensors'],sourceOptions};
      const bad=p.ModelSourceOptions.fromBinary(sourceOptions.toBinary());bad.graph.sha256[0]^=1;
      await assert.rejects(inference.loadModel(new p.LoadModelRequest({...request,sourceOptions:bad})),
        error=>error.report.code===p.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH);
      assert.deepEqual(fetched,[]);
      for(const manifest of [undefined,sourceOptions]) {
        const model=await inference.loadModel(new p.LoadModelRequest({...request,sourceOptions:manifest}));
        const compiled=await inference.compileModel(new p.CompileModelRequest({modelId:model.modelId}));
        const context=await inference.createExecutionContext(new p.CreateExecutionContextRequest({
          compiledModelId:compiled.compiledModelId,graph:new p.GraphExecution()}));
        await inference.releaseModel(new p.ModelRef(model));
        await inference.releaseCompiledModel(new p.CompiledModelRef(compiled));
        const result=await inference.execute(new p.ExecuteRequest({contextId:context.contextId,
          inputs:new p.InputBindings({values:[new p.Tensor({name:'x',shape:[1n],dtype:p.DataType.DATA_TYPE_F32,
            inline:new Uint8Array(Float32Array.of(2).buffer)})]}),
          scope:new p.ExecutionScope({entrypoint:'evaluate',full:new p.Empty()})}));
        assert.deepEqual([...new Float32Array(result.outputs[0].inline.slice().buffer)],[3.5]);
        await inference.releaseExecutionContext(new p.ExecutionContextRef(context));
      }
      assert.deepEqual(fetched,['weights.safetensors','weights.safetensors']);
      const bounded=new Host({wasmUrl,maxPackageBytes:document.byteLength+shard.byteLength-1,
        fetch:async()=>({ok:true,body:streams(shard)})});
      try {
        const limited=new VxInferenceServiceClient(bounded);
        const owner=await limited.createRuntime(new p.CreateRuntimeRequest());
        await assert.rejects(limited.loadModel(new p.LoadModelRequest({runtimeId:owner.runtimeId,
          graphDocument:document,weightPaths:['weights.safetensors']})),
          error=>error.report.code===p.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE);
      } finally {await bounded.close();}
    } finally {await host.close();}
  });
  test(`${profile}: manifest-backed path loading streams, verifies hashes and preserves runtime usability`,async()=>{
    let buffered=0,fetches=0;
    const host=new Host({wasmUrl,fetch:async()=>{fetches++;return {ok:true,body:streams(graph),arrayBuffer:async()=>{buffered++;throw new Error('whole buffering forbidden');}};}});
    const inference=new VxInferenceServiceClient(host);
    try {
      const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
      const sourceOptions=new p.ModelSourceOptions({maxTotalBytes:100n*1024n*1024n,maxShardBytes:80n*1024n*1024n,graph:integrity(graph)});
      const model=await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',sourceOptions}));
      assert.ok(model.modelId);assert.equal(fetches,1);assert.equal(buffered,0);
      await inference.releaseModel(new p.ModelRef(model));
      const bad=integrity(graph);bad.sha256[0]^=1;
      await assert.rejects(inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',
        sourceOptions:new p.ModelSourceOptions({maxTotalBytes:100n*1024n*1024n,maxShardBytes:80n*1024n*1024n,graph:bad})})),
        error=>error.report.code===p.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH);
      assert.ok((await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',sourceOptions}))).modelId);
    } finally {await host.close();}
  });
  for(const mismatch of ['short','long','header','no-stream'])test(`${profile}: ${mismatch} streamed body refuses before model publication`,async()=>{
    let cancelled=false;
    const actual=mismatch==='short'?graph.subarray(0,graph.byteLength-1):mismatch==='long'?new Uint8Array([...graph,0]):graph;
    const host=new Host({wasmUrl,fetch:async()=>({ok:true,
      ...(mismatch==='no-stream'?{arrayBuffer:async()=>actual.buffer}:{body:new ReadableStream({start(controller){controller.enqueue(actual);if(mismatch==='short')controller.close();},cancel(){cancelled=true;}})}),
      ...(mismatch==='header'?{headers:{get:name=>name==='content-length'?String(graph.byteLength+1):null}}:{}),
    })});
    const inference=new VxInferenceServiceClient(host);
    try {
      const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
      /* Short sources end; excessive sources cancel without another read. */
      await assert.rejects(inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',
        sourceOptions:new p.ModelSourceOptions({maxTotalBytes:1024n,maxShardBytes:1024n,graph:integrity(graph)})})),
        error=>error.report.code===(mismatch==='no-stream'?p.OperationCode.OPERATION_CODE_PACKAGE_BODY_UNAVAILABLE:
          p.OperationCode.OPERATION_CODE_MODEL_SOURCE_INTEGRITY_MISMATCH));
      if(mismatch==='long'||mismatch==='header')assert.equal(cancelled,true);
    } finally {await host.close();}
  });
  test(`${profile}: an ordinary upload competing with an active manifest stream reports BUSY`,{timeout:5000},async()=>{
    let entered;const reading=new Promise(resolve=>{entered=resolve;});
    const controller=new AbortController();
    const slow=new ReadableStream({pull(){entered();return new Promise(()=>{});}}, {highWaterMark:0});
    const host=new Host({wasmUrl,fetch:async source=>({ok:true,body:source==='slow.graph.json'?slow:streams(graph)})});
    const inference=new VxInferenceServiceClient(host);
    try {
      const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
      const active=inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'slow.graph.json',
        sourceOptions:new p.ModelSourceOptions({maxTotalBytes:1024n,maxShardBytes:1024n,graph:integrity(graph)})}),{signal:controller.signal});
      const cancelled=assert.rejects(active,error=>error instanceof RpcError&&error.code===1);
      await reading;
      await assert.rejects(inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json'})),
        error=>error.report.status===p.NativeStatus.NATIVE_STATUS_BUSY&&error.report.code===p.OperationCode.OPERATION_CODE_BUSY);
      controller.abort();await cancelled;
      assert.ok((await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json'}))).modelId);
    } finally {controller.abort();await host.close();}
  });
  test(`${profile}: cancelling after fetch accepts a response releases its unread stream`,async()=>{
    for(const manifest of [false,true]) {
      const controller=new AbortController();let cancelled=false;
      const host=new Host({wasmUrl,fetch:async()=>{
        /* Abort after the transport's fulfillment callback accepts the body,
         * but before CallScope.wait accepts its fulfillment. */
        queueMicrotask(()=>queueMicrotask(()=>controller.abort()));
        return {ok:true,body:new ReadableStream({cancel(){cancelled=true;}})};
      }});
      const inference=new VxInferenceServiceClient(host);
      try {
        const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
        await assert.rejects(inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',
          ...(manifest?{sourceOptions:new p.ModelSourceOptions({maxTotalBytes:1024n,maxShardBytes:1024n,graph:integrity(graph)})}:{})}),
          {signal:controller.signal}),error=>error instanceof RpcError&&error.code===1);
        assert.equal(cancelled,true,'the acquired body must be cancelled even when the waiter never receives it');
      } finally {await host.close();}
    }
  });
  test(`${profile}: explicit manifest streams a source beyond the ordinary 64 MiB limit`,
    {skip:process.env.VOLVOXAI_TEST_LARGE_MODEL_SOURCE!=='1'},async()=>{
      const fixture=largeWeightFixture();let weightFetches=0;
      const host=new Host({wasmUrl,fetch:async source=>source==='graph.json'?{ok:true,body:streams(fixture.document)}:
        (weightFetches++,fixture.response())});
      const inference=new VxInferenceServiceClient(host);
      try {
        const runtime=await inference.createRuntime(new p.CreateRuntimeRequest());
        const request={runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']};
        await assert.rejects(inference.loadModel(new p.LoadModelRequest(request)),
          error=>error.report.code===p.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE);
        assert.equal(weightFetches,1,'ordinary paths enforce their existing limit before reading the weight body');
        const sourceOptions=new p.ModelSourceOptions({maxTotalBytes:80n*1024n*1024n,maxShardBytes:80n*1024n*1024n,
          graph:integrity(fixture.document),weightShards:[fixture.integrity()]});
        const model=await inference.loadModel(new p.LoadModelRequest({...request,sourceOptions}));
        assert.ok(model.modelId);assert.equal(weightFetches,2);
        await inference.releaseModel(new p.ModelRef(model));
      } finally {await host.close();}
    });
}
