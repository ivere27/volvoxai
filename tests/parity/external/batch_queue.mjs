/** Generated C policy API versus the unchanged 99dfd8f TypeScript scheduler. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const legacy=await import(pathToFileURL(path.resolve(args.get('--reference'))).href);
const host=new(api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm'))});
const scheduler=new api.VxSchedulerServiceClient(host),results=[];
const encode=value=>new TextEncoder().encode(String(value)),decode=bytes=>new TextDecoder().decode(bytes);
const ok=(value,label='')=>{assert.equal((value.report??value).status,0,label+JSON.stringify(value.report??value,(_,v)=>typeof v==='bigint'?String(v):v));return value;};
const call=async(method,request)=>ok(await scheduler[method](request),method);
const ref=queue=>new p.BatchQueueRef({queueId:queue.queueId});
const workRef=(queue,workId)=>new p.BatchWorkRef({queueId:queue.queueId,workId});
const tensor=(name,values)=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_F32,shape:[BigInt(values.length)],inline:new Uint8Array(Float32Array.from(values).buffer)});
const inputValues=t=>[...new Float32Array(t.inline.slice().buffer)];
const submit=async(queue,options)=>call('submitBatchWork',new p.SubmitBatchWorkRequest({queueId:queue.queueId,
  group:new p.BatchGroup({model:options.modelId??'',...(options.adapterRevision!==undefined?{adapterRevision:options.adapterRevision}:{}),shapeSignature:options.shapeSignatureMinusBatch??''}),
  payload:encode(options.payload??''),...(options.promptTokens?{decode:new p.DecodeBatchWork({promptTokens:options.promptTokens,maxTokens:options.maxTokens,...(options.promptKey?{promptKey:options.promptKey}:{})})}
  :{stateless:new p.StatelessBatchWork({rows:options.rowsPerLane??1,inputs:[tensor('x',[options.payload??0,(options.payload??0)+.5])]})})}));
const complete=async(queue,dispatch,extra={})=>call('completeBatchDispatch',new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:dispatch.dispatchId,
  outcomes:dispatch.work.map(work=>new p.BatchWorkOutcome({value:encode(`${work.position}:${work.tokens}:${decode(work.payload)}`),
    outputs:[tensor('out',[Number(decode(work.payload)),work.position,work.tokens])]})),...extra}));
const kind=['stateless','prefill','decode'];
const cTrace=d=>({kind:kind[d.kind],group:{model:d.group.model,adapter:d.group.toJson().adapterRevision??null,signature:d.group.shapeSignature,rows:d.rowsPerItem},
  useful:d.usefulItems,usefulRows:d.usefulRows,totalRows:d.totalRows,work:d.work.map(w=>({id:w.padding?-1:w.workId,lane:w.lane,generation:w.laneGeneration,
    position:w.position,tokens:w.tokens,generated:w.generated,padding:w.padding,kvLength:w.kvLength,pages:w.pages}))});
const oldTrace=(works,m)=>({kind:m.kind,group:{model:m.groupKey.modelId,adapter:m.groupKey.adapterRevision,signature:m.groupKey.shapeSignatureMinusBatch,rows:m.groupKey.rowsPerLane},
  useful:m.usefulCount,usefulRows:m.usefulRows,totalRows:m.totalRows,work:works.map(w=>({id:w.requestId,lane:w.slot,generation:w.slotGeneration,
    position:w.position,tokens:w.tokens,generated:w.generated,padding:w.padding,kvLength:w.kvPages?.kvLength??0,pages:[...(w.kvPages?.pageTable??[])]}))});
try {
  for(const scenario of [
    {name:'bulk/padding/mixed-keys',options:{maxLanes:4,multipleOf:4,tokenBudgetPerDispatch:12},requests:Array.from({length:11},(_,i)=>({payload:i,rowsPerLane:3,modelId:i%3?'한'.repeat(300):'a|b',adapterRevision:i%2?'':'x',shapeSignatureMinusBatch:i%3?'s':'b|s'}))},
    {name:'bulk/token-budget/65-lanes',options:{maxLanes:65,tokenBudgetPerDispatch:11},requests:Array.from({length:80},(_,i)=>({payload:i,rowsPerLane:i%3+1}))},
    ...['paged','contiguous'].map(policy=>({name:`decode/chunked/${policy}`,options:{maxLanes:3,tokenBudgetPerDispatch:3,cache:new legacy.PagedKVCache({lanes:3,pageTokens:2,laneTokenCapacity:16,policy})},
      requests:[{payload:1,promptTokens:7,maxTokens:3,promptKey:'p'.repeat(300)},{payload:2,promptTokens:5,maxTokens:4},{payload:3,promptTokens:9,maxTokens:2},{payload:4,promptTokens:7,maxTokens:2,promptKey:'p'.repeat(300)}]})),
  ]) {
    const o=scenario.options,old=new legacy.BatchScheduler({...o,maxRetainedResults:200});
    const queue=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxLanes:o.maxLanes,tokenBudget:o.tokenBudgetPerDispatch,multipleOf:o.multipleOf??1,retainedResults:200,
      ...(o.cache?{cache:new p.BatchCacheOptions({lanes:3,pageTokens:2,laneTokenCapacity:16,policy:o.cache.policy==='paged'?0:1})}:{})}));
    const ids=[];
    for(const request of scenario.requests) {
      const oldId=request.promptTokens?old.submitLLM(request):old.submitStateless({...request,inputs:{x:Float32Array.of(request.payload,request.payload+.5)}});
      const created=await submit(queue,request);assert.equal(created.workId,oldId);ids.push(oldId);
    }
    const expected=[];
    await old.runUntilIdle((works,metadata)=>{
      expected.push(oldTrace(works,metadata));
      return works.map(w=>({value:`${w.position}:${w.tokens}:${w.payload}`,outputs:{out:Float32Array.of(w.payload,w.position,w.tokens)}}));
    });
    const actual=[];let idle=0;
    for(let step=0;step<300;step++) {
      const d=await call('nextBatchDispatch',ref(queue));
      if(!d.dispatchId) {const info=await call('getBatchQueue',ref(queue));if(!info.queueDepth&&!info.activeLanes)break;assert.ok(++idle<10);continue;}
      const again=await call('nextBatchDispatch',ref(queue));assert.deepEqual(again,d);
      actual.push(cTrace(d));
      for(const w of d.work)if(kind[d.kind]==='stateless')assert.deepEqual(inputValues(w.inputs[0]),[Number(decode(w.payload)),Number(decode(w.payload))+.5]);
      await complete(queue,d);
    }
    assert.deepEqual(actual,expected,scenario.name);
    for(const id of ids) {
      const info=await call('getBatchWork',workRef(queue,id)),prior=old.results.find(result=>result.requestId===id);
      assert.equal(info.state,p.BatchWorkState.BATCH_WORK_STATE_COMPLETED);assert.equal(info.generated,prior.generated);
      assert.deepEqual(info.values.map(x=>decode(x.data)),prior.values);
      assert.deepEqual(inputValues(info.outputs[0]),[...prior.outputs.out]);
    }
    const info=await call('getBatchQueue',ref(queue)),prior=old.telemetry();
    for(const key of ['dispatches','rowsDispatched','rowsUseful','completed','failed','cancelled','steps','prefillTokens','prefixReuses','prefixPublications','sharedPromptTokens'])
      assert.equal(Number(info[key]),prior[key],scenario.name+'/'+key);
    assert.ok(info.workerBusyMicros<=info.wallMicros);assert.equal(info.reservedPages,0);
    await call('releaseBatchQueue',ref(queue));
    results.push({name:scenario.name,requests:ids.length,dispatches:actual.length,legacyTrace:true});console.log('PASS '+scenario.name);
  }
  // Invalid admission and completion must not consume IDs or mutate the outbox.
  let refusals=0;
  for(const options of [{maxLanes:0},{maxQueueDepth:0},{tokenBudget:0},{multipleOf:3,maxLanes:2},{policy:99},{maxWaitMicros:2n**63n},
    {cache:new p.BatchCacheOptions({lanes:2,pageTokens:0,laneTokenCapacity:8})},{cache:new p.BatchCacheOptions({lanes:2,pageTokens:2,laneTokenCapacity:8,maxPages:1,policy:1})}]) {
    assert.notEqual((await scheduler.createBatchQueue(new p.CreateBatchQueueRequest(options))).report.status,0);refusals++;
  }
  const queue=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxLanes:2,maxQueueDepth:2,retainedResults:2,multipleOf:2,tokenBudget:4}));
  for(const request of [new p.SubmitBatchWorkRequest({queueId:queue.queueId}),new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:3})}),
    new p.SubmitBatchWorkRequest({queueId:queue.queueId,decode:new p.DecodeBatchWork({promptTokens:1,maxTokens:1})}),
    new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:1,inputs:[tensor('x',[1]),tensor('x',[2])]})})]) {
    assert.notEqual((await scheduler.submitBatchWork(request)).report.status,0);refusals++;
  }
  const a=await submit(queue,{payload:5}),b=await submit(queue,{payload:7});assert.equal(a.workId,1);assert.equal(b.workId,2);
  const overflow=await scheduler.submitBatchWork(new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:1})}));
  assert.equal(overflow.report.status,p.NativeStatus.NATIVE_STATUS_OVERLOADED);refusals++;
  const d=await call('nextBatchDispatch',ref(queue));
  for(const request of [new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId+1n,error:'stale'}),
    new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId,outcomes:[]}),
    new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId,outcomes:[new p.BatchWorkOutcome({outputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[2n],inline:new Uint8Array(1)})]}),new p.BatchWorkOutcome()]})]) {
    assert.notEqual((await scheduler.completeBatchDispatch(request)).status,0);assert.deepEqual(await call('nextBatchDispatch',ref(queue)),d);refusals++;
  }
  assert.equal((await scheduler.takeBatchWork(workRef(queue,b.workId))).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
  await call('cancelBatchWork',workRef(queue,a.workId));
  assert.equal((await call('takeBatchWork',workRef(queue,a.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
  assert.equal((await scheduler.getBatchWork(workRef(queue,a.workId))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  await new Promise(r=>setTimeout(r,12));await complete(queue,d);
  assert.equal((await scheduler.getBatchWork(workRef(queue,a.workId))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  const retained=await call('getBatchWork',workRef(queue,b.workId));assert.deepEqual(inputValues(retained.outputs[0]),[7,0,1]);
  retained.outputs[0].inline.fill(0);assert.deepEqual(inputValues((await call('getBatchWork',workRef(queue,b.workId))).outputs[0]),[7,0,1]);
  let info=await call('getBatchQueue',ref(queue));assert.ok(info.workerBusyMicros>=5000n);assert.ok(info.workerBusyMicros<=info.wallMicros);
  for(let i=0;i<20;i++){await submit(queue,{payload:i});await complete(queue,await call('nextBatchDispatch',ref(queue)));}
  info=await call('getBatchQueue',ref(queue));assert.equal(info.retainedResults,2);assert.equal(info.activeRecords,0);
  assert.equal((await scheduler.getBatchWork(workRef(queue,b.workId))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  await call('releaseBatchQueue',ref(queue));await call('releaseBatchQueue',ref(queue));
  assert.equal((await scheduler.getBatchQueue(ref(queue))).report.status,p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  results.push({name:'atomic-refusals/owned-results/bounded-retention',refusals});console.log('PASS refusals/owned-results/retention');

  // Fill-first acts per group; a full group passes an unrelated waiting group.
  const fill=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxLanes:2,policy:p.BatchQueuePolicy.BATCH_QUEUE_POLICY_FILL_FIRST,maxWaitMicros:200000n}));
  await submit(fill,{payload:1,modelId:'short'});assert.equal((await call('nextBatchDispatch',ref(fill))).dispatchId,0n);
  await submit(fill,{payload:2,modelId:'full'});await submit(fill,{payload:3,modelId:'full'});
  let full=await call('nextBatchDispatch',ref(fill));assert.equal(full.group.model,'full');await complete(fill,full);
  await new Promise(r=>setTimeout(r,210));full=await call('nextBatchDispatch',ref(fill));assert.equal(full.group.model,'short');await complete(fill,full);
  await call('releaseBatchQueue',ref(fill));results.push({name:'fill-first/group-local-deadline'});console.log('PASS fill-first');

  for(const drain of [false,true]) {
    const q=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxLanes:2,tokenBudget:2,
      cache:new p.BatchCacheOptions({lanes:2,pageTokens:2,laneTokenCapacity:8})}));
    const a=await submit(q,{payload:1,promptTokens:5,maxTokens:2}),b=await submit(q,{payload:2,promptTokens:3,maxTokens:1}),c=await submit(q,{payload:3,promptTokens:2,maxTokens:1});
    let d=await call('nextBatchDispatch',ref(q)),firstId=d.dispatchId;
    await call('closeBatchQueue',new p.CloseBatchQueueRequest({queueId:q.queueId,drain}));
    assert.equal((await call('getBatchWork',workRef(q,c.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
    if(drain)for(let i=0;i<20;i++){d=await call('nextBatchDispatch',ref(q));if(!d.dispatchId)break;await complete(q,d);}
    else assert.notEqual((await scheduler.completeBatchDispatch(new p.CompleteBatchDispatchRequest({queueId:q.queueId,dispatchId:firstId,error:'late'}))).status,0);
    for(const work of [a,b])assert.equal((await call('getBatchWork',workRef(q,work.workId))).state,
      drain?p.BatchWorkState.BATCH_WORK_STATE_COMPLETED:p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
    const info=await call('getBatchQueue',ref(q));assert.equal(info.closed,true);assert.equal(info.draining,false);assert.equal(info.reservedPages,0);assert.equal(info.residentPages,0);
    await call('releaseBatchQueue',ref(q));results.push({name:`close/drain-${drain}`});console.log('PASS close/drain-'+drain);
  }
  {
  // A failed lane retires its dispatch atomically; another group stays usable.
  const q=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxLanes:2,
    cache:new p.BatchCacheOptions({lanes:2,pageTokens:2,laneTokenCapacity:8})}));
  const a=await submit(q,{payload:1,promptTokens:2,maxTokens:2,modelId:'a'}),b=await submit(q,{payload:2,promptTokens:2,maxTokens:2,modelId:'b'});
  const bad=await call('nextBatchDispatch',ref(q));await complete(q,bad,{error:'worker failure',outcomes:[]});
  assert.equal((await call('getBatchWork',workRef(q,a.workId))).error,'worker failure');
  for(let i=0;i<6;i++){const d=await call('nextBatchDispatch',ref(q));if(!d.dispatchId)break;await complete(q,d);}
  assert.equal((await call('getBatchWork',workRef(q,b.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_COMPLETED);
  info=await call('getBatchQueue',ref(q));assert.equal(info.failed,1n);assert.equal(info.completed,1n);assert.equal(info.reservedPages,0);
  await call('releaseBatchQueue',ref(q));results.push({name:'failure/independent-groups'});console.log('PASS failure containment');
  }
  await writeFile(args.get('--out')??'batch-queue.json',JSON.stringify({wasm:args.get('--wasm'),reference:'99dfd8f',results},null,2));
  console.log(`PASS batch queue: ${results.length} scenarios`);
} finally {await host.close();}
