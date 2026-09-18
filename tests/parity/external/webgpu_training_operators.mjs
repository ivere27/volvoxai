/** Operator training through the released proto API, with C CPU parity. */
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const args = new Map();
for (let i=2; i<process.argv.length; i+=2) args.set(process.argv[i], process.argv[i+1]);
const api = await import(pathToFileURL(path.resolve(args.get('--bundle'))).href), p = api.pb;
import { definitions, tensor, shard } from './webgpu_operator_fixtures.mjs';
function unpack(bytes) {
  const size=Number(new DataView(bytes.buffer,bytes.byteOffset).getBigUint64(0,true));
  const header=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+size)));
  return Object.fromEntries(Object.entries(header).filter(([name])=>name!=='__metadata__').map(([name,entry])=>[
    name,new Float32Array(bytes.slice(8+size+entry.data_offsets[0],8+size+entry.data_offsets[1]).buffer)]));
}
const ok=value=>{const report=value.report??value;assert.equal(report.status,0,JSON.stringify(report,(_,v)=>typeof v==='bigint'?String(v):v));return value;};
function compare(a,b,label) {
  assert.equal(a.length,b.length,label);let error=0;
  for(let i=0;i<a.length;i++) {const delta=Math.abs(a[i]-b[i]);assert.ok(Number.isFinite(a[i])&&delta<3e-5+3e-4*Math.abs(b[i]),`${label}[${i}]: ${a[i]} != ${b[i]}`);error=Math.max(error,delta);}
  return error;
}
// A numerical derivative uses only the forward loss. It detects a backward
// formula shared by CPU and GPU being wrong, which backend parity cannot do.
async function gradientOracle(fixture,graph,weights,training,modelId) {
  const stepOptions=learningRate=>({
    inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:fixture.shape.map(BigInt),
      inline:new Uint8Array(tensor(fixture.shape,7).values.buffer)})],
    losses:[new p.CrossEntropyLoss({name:'classification',logitsName:'logits',
      targets:new p.Tensor({shape:[BigInt(fixture.rows)],dtype:p.DataType.DATA_TYPE_I32,inline:new Uint8Array(Int32Array.from({length:fixture.rows},(_,i)=>i%2).buffer)}),normalizer:fixture.rows})],
    trainableNames:Object.keys(fixture.weights).filter(name=>!['lower','upper','running_mean','running_var'].includes(name)&&(fixture.weights[name].dtype??'F32')==='F32'),
    optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,
      learningRate,weightDecay:0,maxGradientNorm:0}),
  });
  const trainerId=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId,backend:'webgpu',rngSeed:171n}))).trainerId;
  try {
    let result=ok(await training.trainStep(new p.TrainStepRequest({...stepOptions(.2),trainerId})));
    const deadline=Date.now()+30_000;
    while(result.state===p.ResultState.RESULT_STATE_PENDING) {
      assert.ok(Date.now()<deadline);await new Promise(r=>setTimeout(r,1));
      result=ok(await training.getTrainStep(new p.TrainStepRef({trainerId,microbatchId:result.microbatchId})));
    }
    assert.equal(result.state,p.ResultState.RESULT_STATE_READY);
    const updated=unpack(ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId}))).shards[0]);
    const lossAt=async(weightBytes)=>{
      const host=new api.FullEngineHost({wasmUrl:pathToFileURL(path.resolve(args.get('--wasm'))),
        fetch:async url=>new Response(String(url).endsWith('graph.json')?graph:weightBytes)});
      try {
        const inference=new api.VxInferenceServiceClient(host),cpuTraining=new api.VxTrainingServiceClient(host);
        const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
        const cpuModel=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']})));
        const cpu=ok(await cpuTraining.createTrainer(new p.CreateTrainerRequest({modelId:cpuModel.modelId,backend:'wasm',rngSeed:171n})));
        return ok(await cpuTraining.trainStep(new p.TrainStepRequest({...stepOptions(0),trainerId:cpu.trainerId}))).loss;
      } finally {await host.close();}
    };
    const headerSize=Number(new DataView(weights.buffer,weights.byteOffset).getBigUint64(0,true));
    const header=JSON.parse(new TextDecoder().decode(weights.subarray(8,8+headerSize)));
    let samples=0,maximumGradientError=0;
    for(const name of stepOptions(0).trainableNames) {
      const values=fixture.weights[name].values;
      for(const i of new Set([Math.floor(values.length*.27),Math.floor(values.length*.73)])) {
        // A 0.005 perturbation crosses a ProfileY maximum in this fixture.
        // Keep the derivative local while remaining above F32 loss roundoff.
        const before=values[i],low=Math.fround(before-.001),high=Math.fround(before+.001);
        const offset=8+headerSize+header[name].data_offsets[0]+4*i;
        const minus=weights.slice(),plus=weights.slice();
        new DataView(minus.buffer).setFloat32(offset,low,true);
        new DataView(plus.buffer).setFloat32(offset,high,true);
        const expected=((await lossAt(plus))-(await lossAt(minus)))/(high-low);
        const actual=(before-updated[name][i])/Math.fround(.2);
        const error=Math.abs(actual-expected);
        assert.ok(Number.isFinite(error)&&error<3e-4+.025*Math.abs(expected),
          `${fixture.name}/${name}[${i}] gradient ${actual}, numerical derivative ${expected}`);
        samples++;maximumGradientError=Math.max(maximumGradientError,error);
      }
    }
    return {samples,maximumGradientError};
  } finally {ok(await training.releaseTrainer(new p.TrainerRef({trainerId})));}
}
const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});assert.ok(adapter);
const identity=Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info?.[k]??'']));
assert.doesNotMatch(JSON.stringify(identity),/swiftshader|llvmpipe|software/i);
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,options)=>String(url).startsWith('file:')?new Response(await readFile(fileURLToPath(url))):originalFetch(url,options);
const cases=definitions.filter(value=>!args.has('--only')||value.name.includes(args.get('--only')));
assert.ok(cases.length,'No training cases matched');
const results=[];
try {
  for(const fixture of cases) {
    const graph=new TextEncoder().encode(JSON.stringify({format:'volvox-graph/v1',dimensions:{},
      inputs:{x:{dtype:'float32',shape:fixture.shape}},nodes:fixture.nodes,outputs:['logits']}));
    const weights=shard(fixture.weights);
    const host=new api.FullEngineHost({wasmUrl:pathToFileURL(path.resolve(args.get('--wasm'))),
      fetch:async url=>new Response(String(url).endsWith('graph.json')?graph:weights)});
    try {
      const inference=new api.VxInferenceServiceClient(host), training=new api.VxTrainingServiceClient(host);
      const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
      const model=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['model.safetensors']})));
      const cpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend:'wasm',rngSeed:171n}))).trainerId;
      const gpu=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend:'webgpu',rngSeed:171n}))).trainerId;
      let maximumLossError=0, maximumWeightError=0;
      for(let step=0;step<4;step++) {
        const values=tensor(fixture.shape,step+7).values;
        const options={inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:fixture.shape.map(BigInt),inline:new Uint8Array(values.buffer)})],
          losses:[new p.CrossEntropyLoss({name:'classification',logitsName:'logits',targets:new p.Tensor({shape:[BigInt(fixture.rows)],dtype:p.DataType.DATA_TYPE_I32,inline:new Uint8Array(Int32Array.from({length:fixture.rows},(_,i)=>i%2).buffer)}),normalizer:fixture.rows*2})],
          trainableNames:Object.keys(fixture.weights).filter(name=>!['lower','upper','running_mean','running_var'].includes(name)&&(fixture.weights[name].dtype??'F32')==='F32'),
          optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:.03,weightDecay:.01,maxGradientNorm:.1}),
          accumulationSteps:2,resetAccumulation:step===0,flushAccumulation:step===3};
        const expected=ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:cpu})));
        let actual=ok(await training.trainStep(new p.TrainStepRequest({...options,trainerId:gpu})));
        const microbatchId=actual.microbatchId, deadline=Date.now()+30_000;
        assert.equal(actual.state,p.ResultState.RESULT_STATE_PENDING);
        while(actual.state===p.ResultState.RESULT_STATE_PENDING) {
          assert.ok(Date.now()<deadline,'training timed out');await new Promise(resolve=>setTimeout(resolve,1));
          actual=ok(await training.getTrainStep(new p.TrainStepRef({trainerId:gpu,microbatchId})));
        }
        assert.equal(actual.state,p.ResultState.RESULT_STATE_READY);assert.equal(actual.backend,'webgpu');
        assert.equal(actual.updateApplied,expected.updateApplied);
        maximumLossError=Math.max(maximumLossError,compare([actual.loss],[expected.loss],fixture.name+' loss'));
        if(actual.updateApplied) {
          const cpuWeights=unpack(ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:cpu}))).shards[0]);
          const gpuWeights=unpack(ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:gpu}))).shards[0]);
          for(const name of options.trainableNames) maximumWeightError=Math.max(maximumWeightError,compare(gpuWeights[name],cpuWeights[name],fixture.name+'/'+name));
        }
      }
      const gradientChecks=args.get('--gradient-oracle')==='true'
        ? await gradientOracle(fixture,graph,weights,training,model.modelId) : undefined;
      results.push({name:fixture.name,steps:4,maximumLossError,maximumWeightError,gradientChecks});
      console.log(JSON.stringify(results.at(-1)));
    } catch(error) {
      const failure={name:fixture.name,status:'fail',error:String(error)};
      results.push(failure);console.error(JSON.stringify(failure));
    } finally {await host.close();}
  }
} finally {globalThis.fetch=originalFetch;}
const failures=results.filter(value=>value.status==='fail');
const report={status:failures.length?'fail':'pass',identity,cases:results};
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify(report,null,2)+'\n');
assert.equal(failures.length,0,`${failures.length} training cases failed`);
console.log(`WebGPU training operators: ${results.length} cases passed`);
