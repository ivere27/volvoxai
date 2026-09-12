/** Persistent F32 adapter masters -> immutable I8 inference revisions, all via C. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const ok=v=>{assert.equal((v.report??v).status,0,JSON.stringify(v.report??v,(_,v)=>typeof v==='bigint'?String(v):v));return v;};
const bytes=a=>new Uint8Array(a.buffer,a.byteOffset,a.byteLength).slice();
const tensor=(name,shape,data)=>new p.Tensor({name,shape:shape.map(BigInt),dtype:data instanceof Float32Array?p.DataType.DATA_TYPE_F32:data instanceof Int8Array?p.DataType.DATA_TYPE_I8:p.DataType.DATA_TYPE_I32,inline:bytes(data)});
const node=(id,opType,inputs,shape,dtype='float32',params={})=>({id,opType,inputs,outputs:{out:{tensor:id,shape,dtype}},params});
const base=Float32Array.from({length:16},(_,i)=>Math.sin(i*.63)*.15),x=Float32Array.of(.5,-.7,.2,.9,-.3,.4,.8,-.2),results=[];
for(const backend of (args.get('--backends')??'wasm,webgpu').split(',')) {
 const files=new Map();
 const host=new api.FullEngineHost({wasmUrl:path.resolve(args.get('--wasm')),fetch:async url=>{
   const value=files.get(String(url));assert.ok(value,`missing fixture ${url}`);return new Response(value);
 }});
 try {
  const inference=new api.VxInferenceServiceClient(host),training=new api.VxTrainingServiceClient(host),quantization=new api.VxQuantizationServiceClient(host),planning=new api.VxPlanningServiceClient(host);
  const shard=async tensors=>ok(await planning.writeSafetensors(new p.WriteSafetensorsRequest({edits:tensors.map(t=>new p.SafetensorsEdit({setTensor:t}))}))).data;
  const qa=Int8Array.of(7,-3,5,2,-4,9,1,-6),qb=Int8Array.of(3,-2,6,1,-5,4,2,7),sa=Float32Array.of(.03,.04),sb=Float32Array.of(.02,.025,.03,.04);
  const masters=[];
  for(const [name,data,shape,scales] of [['a',qa,[2,4],sa],['b',qb,[4,2],sb]]) {
   const converted=ok(await quantization.dequantizeWeight(new p.DequantizeWeightRequest({weight:tensor(name,shape,data),scales:[...scales],transposeDestination:true})));
   masters.push(new p.Tensor({...converted.tensor,name}));
  }
  const trainingGraph={format:'volvox-graph/v1',dimensions:{},inputs:{x:{shape:[2,4],dtype:'float32'}},outputs:['out'],nodes:[
   node('base','Linear',{input:'x',weight:'w'},[2,4],'float32',{weight_layout:'din_dout'}),
   node('hidden','MatMul',{input:'x',weight:'a'},[2,2],'float32',{weight_layout:'din_dout'}),
   node('delta','MatMul',{input:'hidden',weight:'b'},[2,4],'float32',{weight_layout:'din_dout'}),
   node('out','Add',{a:'base',b:'delta'},[2,4]),
  ]};
  const inferenceGraph={...trainingGraph,nodes:[trainingGraph.nodes[0],
   node('xq','QuantizeLinear',{input:'x',scale:'sx',zero_point:'zx'},[2,4],'int8'),
   node('hq','QLinear',{input:'xq',weight:'qa',bias:'ba'},[2,2],'int8'),
   node('dq','QLinear',{input:'hq',weight:'qb',bias:'bb'},[2,4],'int8'),
   node('delta','DequantizeLinear',{input:'dq',scale:'sd',zero_point:'zd'},[2,4]),trainingGraph.nodes[3]],outputs:['out','hq','dq'],
   quantization:{format:'volvox-affine-safetensors/v1',tensors:{
    xq:{scheme:'per_tensor',scale_tensor:'sx',zero_point_tensor:'zx'},hq:{scheme:'per_tensor',scale_tensor:'sh',zero_point_tensor:'zh'},dq:{scheme:'per_tensor',scale_tensor:'sd',zero_point_tensor:'zd'},
    qa:{scheme:'per_axis',axis:0,scale_tensor:'sa',zero_point_tensor:'za'},qb:{scheme:'per_axis',axis:0,scale_tensor:'sb',zero_point_tensor:'zb'},
   }}};
  const baseWeights=[tensor('w',[4,4],base)],quantizedWeights=[...baseWeights,tensor('qa',[2,4],qa),tensor('qb',[4,2],qb),tensor('sa',[2],sa),tensor('sb',[4],sb),tensor('za',[2],new Int8Array(2)),tensor('zb',[4],new Int8Array(4)),tensor('ba',[2],new Int32Array(2)),tensor('bb',[4],new Int32Array(4))];
  for(const [suffix,scale] of [['x',.01],['h',.004],['d',.002]])quantizedWeights.push(tensor('s'+suffix,[1],Float32Array.of(scale)),tensor('z'+suffix,[1],Int8Array.of(0)));
  const template=await shard(quantizedWeights),trainingWeights=await shard([...baseWeights,...masters]);
  const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
  async function load(graph,weights,id) {
   files.set(id+'.graph.json',new TextEncoder().encode(JSON.stringify(graph)));files.set(id+'.safetensors',weights);
   return ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:id+'.graph.json',weightPaths:[id+'.safetensors']})));
  }
  if(args.get('--debug')==='true')console.log('SOURCE '+JSON.stringify(ok(await planning.readSafetensors(new p.ReadSafetensorsRequest({source:trainingWeights}))).tensors.filter(t=>['a','b'].includes(t.name)).map(t=>[t.name,[...new Float32Array(t.inline.slice().buffer)]])));
  const model=await load(trainingGraph,trainingWeights,'training');
  const trainer=ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend})));
  const bindings=[new p.TrainerQuantizedWeightBinding({masterName:'a',weightName:'qa',scaleName:'sa',zeroPointName:'za',transposeMaster:true}),new p.TrainerQuantizedWeightBinding({masterName:'b',weightName:'qb',scaleName:'sb',zeroPointName:'zb',transposeMaster:true,preserveScales:true})];
  const exportRequest=new p.ExportQuantizedTrainerWeightsRequest({trainerId:trainer.trainerId,templateShards:[template],bindings});
  async function snapshot(){return ok(await quantization.exportQuantizedTrainerWeights(exportRequest));}
  async function instantiate(weights,index) {
   const model=await load(inferenceGraph,weights,'inference-'+index);
   const compiled=ok(await inference.compileModel(new p.CompileModelRequest({modelId:model.modelId,policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[backend]})})));
   const context=ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId})));
   const result=ok(await inference.execute(new p.ExecuteRequest({contextId:context.contextId,inputs:[tensor('x',[2,4],x)]})));
   const deadline=Date.now()+30000;
   while(ok(await inference.getResult(new p.ResultRef(result))).state===p.ResultState.RESULT_STATE_PENDING){assert.ok(Date.now()<deadline);await new Promise(r=>setTimeout(r,1));}
   const read=async()=>{
    const values={};for(const name of ['out','hq','dq']){const t=ok(await inference.readOutput(new p.ReadOutputRequest({resultId:result.resultId,name}))).tensor;values[name]=[...(name==='out'?new Float32Array(t.inline.slice().buffer):new Int8Array(t.inline.slice().buffer))];}return values;
   };
   return {model,result,read,values:await read()};
  }
  // Initializing masters from I8 does not mutate the template or create optimizer state.
  async function debugMasters(label) {
   if(args.get('--debug')!=='true')return;
   const weights=ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))).shards[0];
   console.log(label+' '+JSON.stringify(ok(await planning.readSafetensors(new p.ReadSafetensorsRequest({source:weights}))).tensors.filter(t=>['a','b'].includes(t.name)).map(t=>[t.name,[...new Float32Array(t.inline.slice().buffer)]])));
  }
  await debugMasters('CREATED');
  const initial=await instantiate(template,0),initialTemplate=template.slice(),snapshots=[];
  await debugMasters('AFTER_INFERENCE');
  let previous=initial.values.out;
  for(let step=0;step<3;step++) {
   let trained=ok(await training.trainStep(new p.TrainStepRequest({trainerId:trainer.trainerId,inputs:[tensor('x',[2,4],x)],trainableNames:['a','b'],
    losses:[new p.CrossEntropyLoss({name:'loss',logitsName:'out',targets:[1,3]})],optimizer:new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,learningRate:.3})})));
   if(trained.state===p.ResultState.RESULT_STATE_PENDING) {
    assert.equal((await quantization.exportQuantizedTrainerWeights(exportRequest)).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
    const deadline=Date.now()+30000;
    do{assert.ok(Date.now()<deadline);await new Promise(r=>setTimeout(r,1));trained=ok(await training.getTrainStep(new p.TrainStepRef({trainerId:trainer.trainerId,microbatchId:trained.microbatchId})));}while(trained.state===p.ResultState.RESULT_STATE_PENDING);
   }
   assert.equal(trained.state,p.ResultState.RESULT_STATE_READY);assert.ok(trained.updateApplied);
   const before=ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))),stateBefore=ok(await training.getTrainerState(new p.TrainerRef(trainer)));
   const packed=await snapshot();assert.equal(packed.shards.length,1);assert.equal(packed.report.lineage.modelId,model.modelId);
   assert.deepEqual(ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId:trainer.trainerId}))).shards,before.shards);
   assert.deepEqual(ok(await training.getTrainerState(new p.TrainerRef(trainer))),stateBefore);
   assert.deepEqual(template,initialTemplate);
   if(args.get('--debug')==='true') {
    const current=ok(await planning.readSafetensors(new p.ReadSafetensorsRequest({source:before.shards[0]})));
    const qt=ok(await planning.readSafetensors(new p.ReadSafetensorsRequest({source:packed.shards[0]})));
    const project=t=>[t.name,[...(t.dtype===p.DataType.DATA_TYPE_F32?new Float32Array(t.inline.slice().buffer):new Int8Array(t.inline.slice().buffer))]];
    console.log(JSON.stringify({backend,step,loss:trained.loss,initial:masters.map(project),masters:current.tensors.filter(t=>['a','b'].includes(t.name)).map(project),quantized:qt.tensors.filter(t=>['qa','qb','sa','sb'].includes(t.name)).map(project)}));
   }
   const published=await instantiate(packed.shards[0],step+1);snapshots.push(published.values);
   assert.ok(published.values.out.some((v,i)=>Math.abs(v-previous[i])>.001),`new quantized revision changes inference ${backend}/${step}: ${JSON.stringify({previous,current:published.values})}`);previous=published.values.out;
   assert.deepEqual(await initial.read(),initial.values,'old model/result survive each successor publication');
   // An invalid successor compile/load cannot alter accepted weights or snapshots.
   const broken={...inferenceGraph,nodes:[...inferenceGraph.nodes.slice(0,-1),node('out','MissingOperator',{input:'delta'},[2,4])]};
   files.set('bad.graph.json',new TextEncoder().encode(JSON.stringify(broken)));files.set('bad.safetensors',packed.shards[0]);
   const failed=await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'bad.graph.json',weightPaths:['bad.safetensors']}));
   assert.notEqual(failed.report.status,0);assert.equal(failed.modelId,0n);assert.deepEqual(await initial.read(),initial.values);
  }
  let partial=ok(await training.trainStep(new p.TrainStepRequest({trainerId:trainer.trainerId,inputs:[tensor('x',[2,4],x)],trainableNames:['a','b'],
   losses:[new p.CrossEntropyLoss({name:'loss',logitsName:'out',targets:[1,3],normalizer:4})],accumulationSteps:2})));
  while(partial.state===p.ResultState.RESULT_STATE_PENDING){await new Promise(r=>setTimeout(r,1));partial=ok(await training.getTrainStep(new p.TrainStepRef({trainerId:trainer.trainerId,microbatchId:partial.microbatchId})));}
  assert.equal(partial.updateApplied,false);assert.notEqual((await quantization.exportQuantizedTrainerWeights(exportRequest)).report.status,0);
  ok(await training.resetTrainerAccumulation(new p.TrainerRef(trainer)));
  const again=await snapshot();assert.ok(again.shards[0].length>0);
  results.push({backend,snapshots,steps:3,persistentMasters:true,initialDequantization:true,immutableRevisions:true,busyAndAccumulationGuards:true});console.log('PASS quantized LoRA '+backend);
 } finally {await host.close();}
}
if(results.length===2)for(let s=0;s<3;s++)for(const key of ['out','hq','dq'])for(let i=0;i<results[0].snapshots[s][key].length;i++)
 assert.ok(Math.abs(results[0].snapshots[s][key][i]-results[1].snapshots[s][key][i])<=(key==='out'?.006:1),`backend ${s}/${key}/${i}`);
await writeFile(args.get('--out')??'quantized-lora.json',JSON.stringify({results},null,2));
