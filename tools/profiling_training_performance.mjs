/** Worker for profiling_training_performance.py; never starts a trace. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {resolve} from 'node:path';
import {createInterface} from 'node:readline';
const [release, backend] = process.argv.slice(2);
const {version} = JSON.parse(await readFile(resolve(release, 'package.json'), 'utf8'));
const reader = createInterface({input: process.stdin});
const commands = reader[Symbol.asyncIterator]();
const api = await import(pathToFileURL(resolve(release, `dist/${version}/volvoxai.js`))), p = api.pb;
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => String(url).startsWith('file:')
  ? new Response(await readFile(fileURLToPath(url))) : originalFetch(url, init);
const host = new api.FullEngineHost({wasmUrl:pathToFileURL(resolve(release,`dist/${version}/volvoxai.wasm`))});
const inference = new api.VxInferenceServiceClient(host), training = new api.VxTrainingServiceClient(host);
const ok = result => {assert.equal((result.report ?? result).status, 0); return result;};
try {
  const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({cpuThreads:1})));
  const graph = {format:'volvox-graph/v1',dimensions:{},inputs:{x:{dtype:'float32',shape:[1,2]}},
    nodes:[{id:'linear',opType:'Linear',inputs:{input:'x',weight:'w'},
      outputs:{out:{tensor:'logits',dtype:'float32',shape:[1,2]}},params:{weight_layout:'din_dout'}}],outputs:['logits']};
  let header=JSON.stringify({w:{dtype:'F32',shape:[2,2],data_offsets:[0,16]}});header+=' '.repeat(-header.length&7);
  const weights=new Uint8Array(8+header.length+16);
  new DataView(weights.buffer).setBigUint64(0,BigInt(header.length),true);
  weights.set(new TextEncoder().encode(header),8);weights.set(new Uint8Array(Float32Array.of(.2,-.4,.1,.3).buffer),8+header.length);
  const model=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,
    package:new p.ModelPackage({graphDocument:new TextEncoder().encode(JSON.stringify(graph)),weightShards:[weights]})})));
  const trainer=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend})));
  const request=new p.TrainStepRequest({trainerId:trainer.trainerId,trainableNames:['w'],
    inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[1n,2n],inline:new Uint8Array(Float32Array.of(1,0).buffer)})],
    losses:[new p.CrossEntropyLoss({name:'ce',logitsName:'logits',targets:new p.Tensor({
      dtype:p.DataType.DATA_TYPE_I32,shape:[1n],inline:new Uint8Array(Int32Array.of(0).buffer)})})],
    optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:1e-5})});
  async function run() {
    let result=ok(await training.trainStep(request));const deadline=performance.now()+30000;
    while(result.state===p.ResultState.RESULT_STATE_PENDING) {
      assert.ok(performance.now()<deadline);await new Promise(r=>setTimeout(r,0));
      result=ok(await training.getTrainStep(new p.TrainStepRef({trainerId:trainer.trainerId,microbatchId:result.microbatchId})));
    }
    assert.equal(result.backend,backend);assert.equal(result.updateApplied,true);assert.ok(Number.isFinite(result.loss));return result;
  }
  const first=await run(), probability=Math.exp(.2)/(Math.exp(.2)+Math.exp(-.4));
  assert.ok(Math.abs(first.loss+Math.log(probability))<1e-6);
  const shard=ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest(trainer))).shards[0];
  const size=Number(new DataView(shard.buffer,shard.byteOffset).getBigUint64(0,true));
  const offset=JSON.parse(new TextDecoder().decode(shard.subarray(8,8+size))).w.data_offsets[0];
  const actual=new Float32Array(shard.slice(8+size+offset,8+size+offset+16).buffer);
  const expected=[.2+1e-5*(1-probability),-.4-1e-5*(1-probability),.1,.3];
  assert.ok(actual.every((v,i)=>Math.abs(v-expected[i])<1e-6));
  for(let i=0;i<1000;i++)await run();
  console.log(JSON.stringify({ready:true}));
  for(let sample=0;sample<21;sample++) {
    assert.equal((await commands.next()).value,'sample');const start=performance.now();
    for(let i=0;i<20;i++)await run();
    console.log(JSON.stringify({sample,wallMs:(performance.now()-start)/20}));
  }
  assert.equal((await commands.next()).value,'done');
} finally {reader.close();await host.close();globalThis.fetch=originalFetch;}
