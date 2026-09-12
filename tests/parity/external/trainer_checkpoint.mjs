/** Portable checkpoint continuation, rejection atomicity and accumulation state. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {pathToFileURL,fileURLToPath} from 'node:url';
import path from 'node:path';
const args=new Map();
for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const bytes=x=>new Uint8Array(Float32Array.from(x).buffer);
const graph=new TextEncoder().encode(JSON.stringify({format:'volvox-graph/v1',dimensions:{B:{min:1,max:8}},
  inputs:{x:{dtype:'float32',shape:['B',2]}},nodes:[
    {id:'project',opType:'MatMul',inputs:{input:'x',weight:'w'},outputs:{out:{tensor:'project',dtype:'float32',shape:['B',3]}},params:{weight_layout:'din_dout'}},
    {id:'dropout',opType:'Dropout',inputs:{input:'project'},outputs:{out:{tensor:'dropout',dtype:'float32',shape:['B',3]}},params:{ratio:.25}},
    {id:'logits',opType:'MatMul',inputs:{input:'dropout',weight:'head'},outputs:{out:{tensor:'logits',dtype:'float32',shape:['B',2]}},params:{weight_layout:'din_dout'}},
  ],outputs:['logits']}));
const header=new TextEncoder().encode(JSON.stringify({w:{dtype:'F32',shape:[2,3],data_offsets:[0,24]},head:{dtype:'F32',shape:[3,2],data_offsets:[24,48]}}));
const padded=(header.length+7)&~7,weights=new Uint8Array(8+padded+48);
new DataView(weights.buffer).setBigUint64(0,BigInt(padded),true);weights.fill(32,8,8+padded);weights.set(header,8);
weights.set(bytes([.2,-.3,.4,.1,-.2,.3,.3,-.2,.1,.2,-.1,.4]),8+padded);
const ok=r=>{assert.equal((r.report??r).status,0,JSON.stringify(r.report??r,(_,v)=>typeof v==='bigint'?String(v):v));return r;};
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,options)=>String(url).startsWith('file:')?new Response(await readFile(fileURLToPath(url))):originalFetch(url,options);
const results=[];
try {
 for(const backend of (args.get('--backends')??'wasm,webgpu').split(',')) {
  const host=new api.FullEngineHost({wasmUrl:pathToFileURL(path.resolve(args.get('--wasm'))),
    fetch:async url=>new Response(String(url).endsWith('graph.json')?graph:weights)});
  try {
   const inference=new api.VxInferenceServiceClient(host),training=new api.VxTrainingServiceClient(host);
   const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
   const model=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']})));
   const create=async (checkpoint,shapeOptions)=>ok(await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend,rngSeed:1987n,checkpoint,shapeOptions}))).trainerId;
   const source=await create();
   const state=async id=>ok(await training.getTrainerState(new p.TrainerRef({trainerId:id})));
   const checkpoint=async(id,metadata)=>ok(await training.exportTrainerCheckpoint(new p.ExportTrainerCheckpointRequest({trainerId:id,metadata}))).checkpoint;
   const wait=async(id,result)=>{
    const deadline=Date.now()+30_000;
    while(result.state===p.ResultState.RESULT_STATE_PENDING){assert.ok(Date.now()<deadline);await new Promise(r=>setTimeout(r,1));result=ok(await training.getTrainStep(new p.TrainStepRef({trainerId:id,microbatchId:result.microbatchId})));}
    assert.equal(result.state,p.ResultState.RESULT_STATE_READY);return result;
   };
   const configuredOptimizer=new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW,
    learningRate:.02,beta1:.8,beta2:.95,epsilon:1e-7,weightDecay:.01,maxGradientNorm:.15});
   const start=async(id,batch,accumulationSteps=1,optimizer=configuredOptimizer)=>ok(await training.trainStep(new p.TrainStepRequest({trainerId:id,
    inputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[BigInt(batch),2n],inline:bytes(Array.from({length:batch*2},(_,i)=>Math.sin(i+.7)))})],
    losses:[new p.CrossEntropyLoss({name:'ce',logitsName:'logits',targets:Array.from({length:batch},(_,i)=>i%2),normalizer:batch*accumulationSteps})],
    trainableNames:['w','head'],accumulationSteps,
    optimizer})));
   const step=async(id,batch,optimizer)=>wait(id,await start(id,batch,1,optimizer));
   const initialState=await state(source);assert.equal(initialState.optimizerStep,0n);assert.equal(initialState.rngSeed,1987n);
   assert.equal(initialState.report.lineage.modelId,model.modelId);
   assert.equal(initialState.report.lineage.runtimeId,runtime.runtimeId);
   assert.equal(initialState.optimizerConfig.kind,p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW);
   assert.equal(initialState.optimizerConfig.learningRate,Math.fround(.001));
   assert.equal(initialState.shape.signature,'');
   assert.equal(initialState.shape.options.planCacheEntries,8);
   assert.equal(initialState.shape.options.planCacheMetadataBytes,1048576n);
   assert.equal(initialState.shape.options.maxActivationCapacityBytes,536870912n);
   assert.equal(initialState.shape.options.capacityGrowthFactor,2);
   const zero=await checkpoint(source);assert.equal(zero.optimizerStep,0n);assert.ok(zero.optimizer.length>8);
   await step(source,1);await step(source,2);
   const saved=await checkpoint(source,new Uint8Array([0,255,65,0]));
   const decoded=p.TrainerCheckpoint.fromBinary(saved.toBinary());
   assert.deepEqual(decoded.optimizerConfig.toBinary(),configuredOptimizer.toBinary());
   assert.deepEqual(decoded.graph,graph);assert.equal(decoded.optimizerStep,2n);assert.equal(decoded.rngSeed,1987n);
   const restored=await create(decoded);
   assert.equal((await state(restored)).optimizerBytes,96n);
   assert.deepEqual((await checkpoint(restored)).toBinary(),decoded.toBinary());
   await step(source,3);await step(restored,3,null);
   const bound=(await state(restored)).shape;
   assert.ok(bound.signature.length>0);
   assert.equal(bound.parameterBytes,48n);
   assert.ok(bound.activationCapacityBytes>=72n);
   assert.ok(bound.activationHighWaterBytes>=bound.activationCapacityBytes);
   assert.ok(bound.planCacheEntries>0);assert.ok(bound.planCacheMetadataBytes>0n);
   assert.equal(bound.backend,backend);assert.equal(bound.cachedPlans.length,bound.planCacheEntries);
   assert.ok(bound.cachedPlans.some(plan=>plan.signature===bound.signature));
   assert.equal(bound.cachedPlans.reduce((sum,plan)=>sum+plan.metadataBytes,0n),bound.planCacheMetadataBytes);
   assert.ok(bound.planCacheStorageBytes>=bound.planCacheMetadataBytes);
   for(const plan of bound.cachedPlans)assert.ok(plan.arenaBytes>=plan.logicalBytes);
   const unchanged=await state(restored);assert.deepEqual(unchanged.shape,bound,'inspection never touches LRU order');
   assert.deepEqual(bound.activations.find(t=>t.name==='logits').shape,[3n,2n]);
   assert.deepEqual(bound.activationGradients.find(t=>t.name==='logits').shape,[3n,2n]);
   assert.deepEqual(bound.parameterGradients.find(t=>t.name==='w').shape,[2n,3n]);
   const continued=await checkpoint(source),restoredContinued=await checkpoint(restored);
   assert.deepEqual(restoredContinued.weightShards,continued.weightShards,'weights after continuation');
   assert.deepEqual(restoredContinued.optimizer,continued.optimizer,'AdamW moments after continuation');
   // Restore must own its bytes after the request has been discarded or modified.
   decoded.weightShards[0].fill(0);decoded.optimizer.fill(0);decoded.metadata.fill(17);
   ok(await training.rollbackTrainer(new p.TrainerRef({trainerId:restored})));
   assert.deepEqual((await checkpoint(restored)).toBinary(),saved.toBinary(),'rollback restores imported baseline');
   await step(restored,3);
   assert.deepEqual((await checkpoint(restored)).weightShards,continued.weightShards,'RNG resumes from the rollback baseline');
   // Rejected restore cannot return a Trainer ID or alter a live Trainer.
   for(const mutate of [cp=>{cp.optimizerConfig=null;},cp=>{cp.optimizerConfig.learningRate=NaN;},cp=>{cp.optimizerConfig.beta1=undefined;},cp=>{cp.version=999;},cp=>{cp.graph[0]^=1;},cp=>{cp.optimizerStep++;},cp=>{cp.optimizer=cp.optimizer.subarray(0,8);},cp=>{new DataView(cp.weightShards[0].buffer).setFloat32(cp.weightShards[0].length-4,NaN,true);}]){
    const invalid=p.TrainerCheckpoint.fromBinary(saved.toBinary());mutate(invalid);
    const refusal=await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend,checkpoint:invalid}));
    assert.notEqual(refusal.report.status,0);assert.equal(refusal.trainerId,0n);
    assert.deepEqual((await checkpoint(source)).weightShards,continued.weightShards);
   }
   // Partial overrides keep every saved field, and an invalid override leaves
   // the configuration unchanged. Switching optimizer kind uses its defaults.
   const overrideTrainer=await create(saved,new p.TrainerShapeOptions({planCacheEntries:4}));
   await step(overrideTrainer,3,new p.TrainerOptimizerOptions({learningRate:.007}));
   const overrideState=await state(overrideTrainer);
   assert.equal(overrideState.optimizerConfig.learningRate,Math.fround(.007));
   assert.equal(overrideState.optimizerConfig.beta1,Math.fround(.8));
   assert.equal(overrideState.optimizerConfig.weightDecay,Math.fround(.01));
   await assert.rejects(start(overrideTrainer,3,1,new p.TrainerOptimizerOptions({learningRate:NaN})));
   assert.deepEqual((await state(overrideTrainer)).optimizerConfig.toBinary(),overrideState.optimizerConfig.toBinary());
   await step(overrideTrainer,3,new p.TrainerOptimizerOptions({kind:p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD}));
   const switched=await state(overrideTrainer);
   assert.equal(switched.optimizerConfig.kind,p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD);
   assert.equal(switched.optimizerConfig.learningRate,Math.fround(.001));
   assert.equal(switched.optimizerConfig.weightDecay,0);
   assert.ok(switched.shape.planCacheHits>0n);
   for(const batch of [1,2,4,5,6]) await step(overrideTrainer,batch,null);
   const cached=(await state(overrideTrainer)).shape;
   assert.equal(cached.planCacheEntries,4);
   assert.ok(cached.planCacheEvictions>=2n);
   ok(await training.rollbackTrainer(new p.TrainerRef({trainerId:overrideTrainer})));
   assert.deepEqual((await state(overrideTrainer)).optimizerConfig.toBinary(),saved.optimizerConfig.toBinary());
   ok(await training.releaseTrainer(new p.TrainerRef({trainerId:overrideTrainer})));
   // Cache count, metadata bypass and capacity refusal are independent policies.
   const limited=await create(null,new p.TrainerShapeOptions({planCacheEntries:2,capacityGrowthFactor:1.5}));
   let priorCapacity=0n,priorGrows=0n,largestBatch=0;const cachedSignatures=new Map();
   for(const batch of [1,2,1,3,2]) {
    await step(limited,batch);
    const memory=(await state(limited)).shape;cachedSignatures.set(batch,memory.signature);
    assert.ok(memory.activationCapacityBytes>=priorCapacity,'activation storage is retained across steps');
    if(batch<=largestBatch)assert.equal(memory.activationGrowCount,priorGrows,'smaller/repeated shapes reuse capacity');
    priorCapacity=memory.activationCapacityBytes;priorGrows=memory.activationGrowCount;largestBatch=Math.max(batch,largestBatch);
   }
   const limitedState=(await state(limited)).shape;
   assert.equal(limitedState.planCacheEntries,2);assert.equal(limitedState.planCacheHits,1n);
   assert.equal(limitedState.planCacheMisses,4n);assert.equal(limitedState.planCacheEvictions,2n);
   assert.deepEqual(limitedState.cachedPlans.map(plan=>plan.signature),[cachedSignatures.get(3),cachedSignatures.get(2)]);
   assert.ok(limitedState.cachedPlans[0].lastUsed<limitedState.cachedPlans[1].lastUsed);
   assert.equal(limitedState.planCacheBypasses,0n);
   const bypass=await create(null,new p.TrainerShapeOptions({planCacheMetadataBytes:1n}));
   await step(bypass,1);await step(bypass,1);
   const bypassState=(await state(bypass)).shape;
   assert.equal(bypassState.planCacheEntries,0);assert.equal(bypassState.planCacheMetadataBytes,0n);
   assert.equal(bypassState.planCacheMisses,2n);assert.equal(bypassState.planCacheHits,0n);
   assert.equal(bypassState.planCacheOversizeSkips,2n);assert.equal(bypassState.planCacheBypasses,2n);
   assert.equal(bypassState.planCacheStorageBytes,0n);assert.deepEqual(bypassState.cachedPlans,[]);
   // A policy limit is not an allocation request. The old JS Map grew only
   // for observed shapes; a large limit must not eagerly exhaust WASM memory.
   const sparseCache=await create(null,new p.TrainerShapeOptions({planCacheEntries:1000000000}));
   await step(sparseCache,1);await step(sparseCache,1);
   const sparseState=(await state(sparseCache)).shape;
   assert.equal(sparseState.planCacheEntries,1);assert.equal(sparseState.planCacheHits,1n);
   assert.equal(sparseState.cachedPlans.length,1);assert.ok(sparseState.planCacheStorageBytes<65536n);
   ok(await training.releaseTrainer(new p.TrainerRef({trainerId:sparseCache})));
   const probe=await create();await step(probe,1);
   const firstCapacity=(await state(probe)).shape.activationHighWaterBytes;
   const budget=firstCapacity>initialState.shape.activationHighWaterBytes?firstCapacity:initialState.shape.activationHighWaterBytes;
   const bounded=await create(null,new p.TrainerShapeOptions({maxActivationCapacityBytes:budget}));
   await step(bounded,1);
   const beforeBudget=await checkpoint(bounded),beforeShape=(await state(bounded)).shape;
   await assert.rejects(start(bounded,8));
   assert.deepEqual((await checkpoint(bounded)).toBinary(),beforeBudget.toBinary());
   assert.deepEqual((await state(bounded)).shape.toBinary(),beforeShape.toBinary());
   await step(bounded,1);
   for(const shapeOptions of [{planCacheEntries:0},{planCacheMetadataBytes:0n},
    {maxActivationCapacityBytes:0n},{capacityGrowthFactor:1},{capacityGrowthFactor:NaN},
    {capacityGrowthFactor:Infinity},{maxActivationCapacityBytes:1n}]) {
    const refused=await training.createTrainer(new p.CreateTrainerRequest({modelId:model.modelId,backend,
     shapeOptions:new p.TrainerShapeOptions(shapeOptions)}));
    assert.notEqual(refused.report.status,0);assert.equal(refused.trainerId,0n);
   }
   for(const trainerId of [limited,bypass,probe,bounded])ok(await training.releaseTrainer(new p.TrainerRef({trainerId})));
   const accepted=await start(restored,2,3);
   if(backend==='webgpu'){
    assert.equal((await state(restored)).stepPending,true);
    assert.notEqual((await training.resetTrainerAccumulation(new p.TrainerRef({trainerId:restored}))).report.status,0);
    assert.notEqual((await training.exportTrainerCheckpoint(new p.ExportTrainerCheckpointRequest({trainerId:restored}))).report.status,0);
   }
   await wait(restored,accepted);
   const window=await state(restored);
   assert.equal(window.accumulatedMicrobatches,1);assert.equal(window.accumulationSteps,3);assert.equal(window.gradientBytes,48n);
   assert.ok(window.activationSignature);assert.equal(window.accumulatedMetrics.length,1);
   const refusal=await training.exportTrainerCheckpoint(new p.ExportTrainerCheckpointRequest({trainerId:restored}));
   assert.notEqual(refusal.report.status,0);assert.equal(refusal.checkpoint,undefined);
   const reset=ok(await training.resetTrainerAccumulation(new p.TrainerRef({trainerId:restored})));
   assert.equal(reset.accumulatedMicrobatches,0);assert.equal(reset.gradientBytes,0n);
   assert.deepEqual((await checkpoint(restored)).weightShards,continued.weightShards,'reset preserves parameters');
   ok(await training.commitTrainer(new p.TrainerRef({trainerId:restored})));
   const committed=await checkpoint(restored);await step(restored,1);
   ok(await training.rollbackTrainer(new p.TrainerRef({trainerId:restored})));
   assert.deepEqual((await checkpoint(restored)).toBinary(),committed.toBinary(),'rollback after commit');
   results.push({backend,checkpointContinuation:true,dropoutRng:true,adamwMoments:true,rejectionCases:8,optimizerConfiguration:true,shapeInspection:true,planCacheReuse:true,shapePolicy:true,capacityRefusalAtomic:true,accumulationReset:true,rollback:true});
   console.log(JSON.stringify(results.at(-1)));
  } finally {await host.close();}
 }
} finally {globalThis.fetch=originalFetch;}
if(args.has('--out'))await writeFile(args.get('--out'),JSON.stringify({status:'pass',cases:results},null,2)+'\n');
