import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
#!/usr/bin/env -S deno run --unstable-webgpu --allow-read --allow-env
/** Real minified proto/WASM training qualification. */
import assert from 'node:assert/strict';
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
const args = new Map();
for (let i=2;i<process.argv.length;i++) {
  if (process.argv[i]==='--require-physical') args.set(process.argv[i],true);
  else args.set(process.argv[i],process.argv[++i]);
}
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href), p=api.pb;
const bytesOf=values=>new Uint8Array(Float32Array.from(values).buffer);
const initial={w:{shape:[3,4],values:[.2,-.4,.1,.3,.1,.2,-.1,.4,.3,-.2,.2,.1]},
  b:{shape:[4],values:[.1,-.1,0,.2]},v:{shape:[2,4],values:[.2,.1,-.3,.2,-.2,.4,.1,.3]},c:{shape:[2],values:[0,.1]}};
function shard(tensors) {
  const header={},chunks=[]; let at=0;
  for(const [name,tensor] of Object.entries(tensors)) {
    const chunk=bytesOf(tensor.values);
    header[name]={dtype:'F32',shape:tensor.shape,data_offsets:[at,at+chunk.length]};
    chunks.push(chunk);at+=chunk.length;
  }
  let json=JSON.stringify(header);json+=' '.repeat((8-json.length%8)%8);
  const bytes=new Uint8Array(8+json.length+at);
  new DataView(bytes.buffer).setBigUint64(0,BigInt(json.length),true);
  bytes.set(new TextEncoder().encode(json),8);at=8+json.length;
  for(const chunk of chunks){bytes.set(chunk,at);at+=chunk.length;}
  return bytes;
}
const graph=new TextEncoder().encode(JSON.stringify({format:'volvox-graph/v1',dimensions:{B:{min:1,max:4}},
  inputs:{x:{dtype:'float32',shape:['B',3]}},nodes:[
    {id:'first',opType:'MatMul',inputs:{input:'x',weight:'w',bias:'b'},
      outputs:{out:{tensor:'hidden',dtype:'float32',shape:['B',4]}},params:{weight_layout:'din_dout'}},
    {id:'activation',opType:'Tanh',inputs:{input:'hidden'},
      outputs:{out:{tensor:'activated',dtype:'float32',shape:['B',4]}},params:{}},
    {id:'second',opType:'Linear',inputs:{input:'activated',weight:'v',bias:'c'},
      outputs:{out:{tensor:'logits',dtype:'float32',shape:['B',2]}},params:{weight_layout:'dout_din'}},
  ],outputs:['logits']}));
const dropoutDefinition=JSON.parse(new TextDecoder().decode(graph));
dropoutDefinition.nodes[1].opType='Dropout';dropoutDefinition.nodes[1].params={ratio:.5};
const dropoutGraph=new TextEncoder().encode(JSON.stringify(dropoutDefinition));
const weights=shard(initial),originalFetch=globalThis.fetch;
if(args.has('--fixture-dir')) {
  const directory=path.resolve(args.get('--fixture-dir'));await mkdir(directory,{recursive:true});
  await writeFile(path.join(directory,'graph.json'),graph);
  await writeFile(path.join(directory,'model.safetensors'),weights);
  process.exit(0);
}
const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}); assert.ok(adapter);
const identity=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info?.[k]??'']));
if(args.has('--require-physical')) assert.ok(!/swiftshader|llvmpipe/i.test(JSON.stringify(identity)));

globalThis.fetch=async(url,init)=>String(url).startsWith('file:')?new Response(await readFile(fileURLToPath(url))):originalFetch(url,init);
const host=new api.FullEngineHost({wasmUrl:pathToFileURL(path.resolve(args.get('--wasm'))),
  onDiagnostic:message=>console.error(message),fetch:async url=>new Response(url==='graph.json'?graph:url==='dropout.graph.json'?dropoutGraph:weights)});
const inference=new api.VxInferenceServiceClient(reportTransport(host)),training=new api.VxTrainingServiceClient(reportTransport(host));
const ok=value=>{assert.equal((value.report??value).status,0,JSON.stringify(value.report??value,(_,item)=>typeof item==='bigint'?String(item):item));return value;};
const closeTo=(a,b,label)=>{
  assert.equal(a.length,b.length,label);let maximum=0;
  for(let i=0;i<a.length;i++) {const error=Math.abs(a[i]-b[i]);maximum=Math.max(error,maximum);
    assert.ok(Number.isFinite(a[i])&&error<=3e-5+3e-4*Math.abs(b[i]),label+'['+i+']: '+a[i]+' != '+b[i]);}
  return maximum;
};
const exportWeights=async trainerId=>{
  const bytes=ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId}))).shards[0];
  const length=Number(new DataView(bytes.buffer,bytes.byteOffset).getBigUint64(0,true));
  const header=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+length)));
  return Object.fromEntries(Object.keys(initial).map(name=>{const [start,end]=header[name].data_offsets;
    return [name,new Float32Array(bytes.slice(8+length+start,8+length+end).buffer)];}));
};
// Central finite differences of an independent scalar forward are an oracle
// for the first SGD update, not just parity with another runtime backend.
function oracleSgd() {
  const t=structuredClone(initial),x=Array.from({length:3},(_,i)=>Math.sin(i+1)*.5);
  const loss=()=>{
    const h=t.b.values.map((bias,j)=>Math.tanh(bias+x.reduce((sum,v,i)=>sum+v*t.w.values[i*4+j],0)));
    const logits=t.c.values.map((bias,i)=>bias+h.reduce((sum,v,j)=>sum+v*t.v.values[i*4+j],0));
    const m=Math.max(...logits);return Math.log(logits.reduce((sum,v)=>sum+Math.exp(v-m),0))+m-logits[0];
  };
  return Object.fromEntries(Object.entries(t).map(([name,tensor])=>[name,tensor.values.map((value,i)=>{
    const eps=1e-5;tensor.values[i]=value+eps;const plus=loss();tensor.values[i]=value-eps;const minus=loss();tensor.values[i]=value;
    return value-.03*((plus-minus)/(2*eps)+.02*value);
  })]));
}
let gpu;
const ready=async (accepted,trainerId=gpu)=>{
  const deadline=Date.now()+30_000;let result=accepted;
  while(result.state===p.ResultState.RESULT_STATE_PENDING) {
    assert.ok(Date.now()<deadline,'GPU training did not complete');await new Promise(resolve=>setTimeout(resolve,1));
    result=ok(await training.getTrainStep(new p.TrainStepRef({trainerId,microbatchId:accepted.microbatchId})));
  }
  assert.equal(result.state,p.ResultState.RESULT_STATE_READY);return result;
};
try {
  const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
  const model=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['model.safetensors']})));
  const cpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend:'wasm'}))).trainerId;
  gpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend:'webgpu'}))).trainerId;
  const cases=[];let previous,lastOptions;
  for(const [kind,batch,accumulation,clip] of [[1,1,1,0],[1,3,1,.05],[0,2,1,0],[0,4,1,.05],[0,2,2,.05],[0,2,2,.05],[1,1,1,0]]) {
    const optimizerKind=kind?p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW;
    const options={inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[BigInt(batch),3n],
      inline:bytesOf(Array.from({length:batch*3},(_,i)=>Math.sin(i+1)*.5))})],
      losses:[new p.CrossEntropyLoss({name:'classification',logitsName:'logits',targets:Array.from({length:batch},(_,i)=>i%2),normalizer:batch*accumulation})],
      trainableNames:Object.keys(initial),accumulationSteps:accumulation,
      optimizer:new p.TrainerOptimizerOptions({kind:optimizerKind,learningRate:.03,weightDecay:.02,maxGradientNorm:clip})};
    lastOptions=options;
    const reference=ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:cpu})));
    const accepted=ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:gpu})));
    assert.equal(accepted.state,p.ResultState.RESULT_STATE_PENDING);
    assert.equal((await training.trainStep(new p.TrainStepRequest({...options,trainerId:gpu}))).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
    assert.equal((await training.commitTrainer(new p.TrainerRef({trainerId:gpu}))).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
    assert.equal((await training.rollbackTrainer(new p.TrainerRef({trainerId:gpu}))).status,p.NativeStatus.NATIVE_STATUS_BUSY);
    assert.equal((await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:gpu}))).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
    if(previous) assert.equal((await training.getTrainStep(new p.TrainStepRef({trainerId:gpu,microbatchId:previous}))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
    const actual=await ready(accepted);assert.equal(actual.backend,'webgpu');
    assert.equal(actual.updateApplied,reference.updateApplied);assert.equal(actual.optimizerStep,reference.optimizerStep);
    const lossError=closeTo([actual.loss],[reference.loss],'loss');
    assert.equal(actual.metrics[0].correct,reference.metrics[0].correct);assert.equal(actual.metrics[0].examples,reference.metrics[0].examples);
    assert.deepEqual(await training.getTrainStep(new p.TrainStepRef({trainerId:gpu,microbatchId:actual.microbatchId})),actual);
    let weightError=null;
    if(actual.updateApplied) {const expected=await exportWeights(cpu),actualWeights=await exportWeights(gpu);
      weightError=Math.max(...Object.keys(initial).map(name=>closeTo(actualWeights[name],expected[name],name)));
      if(!previous) {const oracle=oracleSgd();for(const name of Object.keys(initial))closeTo(actualWeights[name],oracle[name],'finite-difference '+name);}
    }
    previous=actual.microbatchId;cases.push({optimizer:kind?'sgd':'adamw',batch,accumulation,clip,lossError,weightError});
  }
  ok(await training.commitTrainer(new p.TrainerRef({trainerId:gpu})));
  ok(await training.rollbackTrainer(new p.TrainerRef({trainerId:gpu})));
  const finalWeights=await exportWeights(gpu);assert.ok(finalWeights.w.some((value,i)=>Math.abs(value-initial.w.values[i])>1e-4));
  // Roll back a real uncommitted update, then recover from a failed async step.
  const committed=await exportWeights(gpu);
  await ready(ok(await training.trainStep(new p.TrainStepRequest({...lastOptions,trainerId:gpu}))));
  assert.ok((await exportWeights(gpu)).w.some((value,i)=>value!==committed.w[i]));
  ok(await training.rollbackTrainer(new p.TrainerRef({trainerId:gpu})));
  assert.deepEqual(await exportWeights(gpu),committed);
  const invalid=ok(await training.trainStep(new p.TrainStepRequest({...lastOptions,trainerId:gpu,
    inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[1n,3n],inline:bytesOf([NaN,1,2])})]})));
  let failed=invalid;const deadline=Date.now()+30_000;
  while(failed.state===p.ResultState.RESULT_STATE_PENDING) {
    assert.ok(Date.now()<deadline);await new Promise(r=>setTimeout(r,1));
    failed=await training.getTrainStep(new p.TrainStepRef({trainerId:gpu,microbatchId:invalid.microbatchId}));
  }
  assert.equal(failed.state,p.ResultState.RESULT_STATE_FAILED);assert.notEqual(failed.report.status,0);
  assert.deepEqual(await exportWeights(gpu),committed);
  await ready(ok(await training.trainStep(new p.TrainStepRequest({...lastOptions,trainerId:gpu}))));
  // Retire accepted work before its readback. The surviving trainer stays usable.
  const cancelled=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend:'webgpu'}))).trainerId;
  const pending=ok(await training.trainStep(new p.TrainStepRequest({...lastOptions,trainerId:cancelled})));
  assert.equal(pending.state,p.ResultState.RESULT_STATE_PENDING);
  ok(await training.releaseTrainer(new p.TrainerRef({trainerId:cancelled})));
  assert.equal((await training.getTrainStep(new p.TrainStepRef({trainerId:cancelled,microbatchId:pending.microbatchId}))).report.status,p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  await ready(ok(await training.trainStep(new p.TrainStepRequest({...lastOptions,trainerId:gpu}))));
  const dropoutModel=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,
    graphPath:'dropout.graph.json',weightPaths:['model.safetensors']})));
  const dropoutCpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:dropoutModel.modelId,backend:'wasm',rngSeed:123n}))).trainerId;
  const dropoutGpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:dropoutModel.modelId,backend:'webgpu',rngSeed:123n}))).trainerId;
  for(let microbatch=0;microbatch<3;microbatch++) {
    const options={...lastOptions,accumulationSteps:2,resetAccumulation:microbatch===0,flushAccumulation:microbatch===2};
    const reference=ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:dropoutCpu})));
    const actual=await ready(ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:dropoutGpu}))),dropoutGpu);
    closeTo([actual.loss],[reference.loss],'dropout loss');
    assert.equal(actual.updateApplied,reference.updateApplied);
    if(actual.updateApplied) {
      const expected=await exportWeights(dropoutCpu),updated=await exportWeights(dropoutGpu);
      for(const name of Object.keys(initial))closeTo(updated[name],expected[name],'dropout '+name);
    }
  }
  ok(await training.releaseTrainer(new p.TrainerRef({trainerId:dropoutCpu})));
  ok(await training.releaseTrainer(new p.TrainerRef({trainerId:dropoutGpu})));
  const report={status:'pass',physicalDevice:identity,cases,
    finiteDifferenceOracle:true,rollback:true,nonfiniteFailureRecovery:true,pendingCancellation:true,dropoutTraining:true,
    contract:'C-planned GPU forward, loss, backward, accumulation, clipping, optimizer; proto completion, commit and rollback'};
  if(args.get('--out')) {await mkdir(path.dirname(args.get('--out')),{recursive:true});await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report));
} finally {await host.close();globalThis.fetch=originalFetch;}
