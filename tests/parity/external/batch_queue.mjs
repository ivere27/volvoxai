/** The public batch queue: admission refusals, owned results, retention, fill-first and draining. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {writeFile} from 'node:fs/promises';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(pathToFileURL(path.resolve(args.get('--bundle'))).href),p=api.pb;
const host=new(api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm'))});
const scheduler=new api.VxSchedulerServiceClient(host),results=[];
const encode=value=>new TextEncoder().encode(String(value)),decode=bytes=>new TextDecoder().decode(bytes);
const ok=(value,label='')=>{assert.equal((value.report??value).status,0,label+JSON.stringify(value.report??value,(_,v)=>typeof v==='bigint'?String(v):v));return value;};
const call=async(method,request)=>ok(await scheduler[method](request),method);
// The public client throws on a refusal; the error carries the response.
const settle=pending=>pending.catch(error=>{assert.ok(error.report,String(error));return error.response??{report:error.report};});
const ref=queue=>new p.BatchQueueRef({queueId:queue.queueId});
const workRef=(queue,workId)=>new p.BatchWorkRef({queueId:queue.queueId,workId});
const tensor=(name,values)=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_F32,shape:[BigInt(values.length)],inline:new Uint8Array(Float32Array.from(values).buffer)});
const inputValues=t=>[...new Float32Array(t.inline.slice().buffer)];
const submit=async(queue,options)=>call('submitBatchWork',new p.SubmitBatchWorkRequest({queueId:queue.queueId,
  group:new p.BatchGroup({model:options.modelId??'',...(options.adapterRevision!==undefined?{adapterRevision:options.adapterRevision}:{}),shapeSignature:options.shapeSignatureMinusBatch??''}),
  payload:encode(options.payload??''),...(options.promptTokens?{decode:new p.DecodeBatchWork({promptTokens:options.promptTokens,maxTokens:options.maxTokens,...(options.promptKey?{promptKey:options.promptKey}:{})})}
  :{stateless:new p.StatelessBatchWork({rows:options.rowsPerItem??1,inputs:[tensor('x',[options.payload??0,(options.payload??0)+.5])]})})}));
const complete=async(queue,dispatch,extra={})=>call('completeBatchDispatch',new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:dispatch.dispatchId,
  outcomes:dispatch.work.map(work=>new p.BatchWorkOutcome({value:encode(`${work.position}:${work.tokens}:${decode(work.payload)}`),
    outputs:[tensor('out',[Number(decode(work.payload)),work.position,work.tokens])]})),...extra}));
try {
  // Invalid admission and completion must not consume IDs or mutate the outbox.
  let refusals=0;
  for(const options of [{maxSlots:0},{maxQueueDepth:0},{tokenBudget:0},{multipleOf:3,maxSlots:2},{policy:99},
    {cache:new p.BatchCacheOptions({slots:2,pageTokens:0,slotTokenCapacity:8})},{cache:new p.BatchCacheOptions({slots:2,pageTokens:2,slotTokenCapacity:8,maxPages:1,policy:1})}]) {
    assert.notEqual((await settle(scheduler.createBatchQueue(new p.CreateBatchQueueRequest(options)))).report.status,0);refusals++;
  }
  const queue=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxSlots:2,maxQueueDepth:2,retainedResults:2,multipleOf:2,tokenBudget:4}));
  for(const request of [new p.SubmitBatchWorkRequest({queueId:queue.queueId}),new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:3})}),
    new p.SubmitBatchWorkRequest({queueId:queue.queueId,decode:new p.DecodeBatchWork({promptTokens:1,maxTokens:1})}),
    new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:1,inputs:[tensor('x',[1]),tensor('x',[2])]})})]) {
    assert.notEqual((await settle(scheduler.submitBatchWork(request))).report.status,0);refusals++;
  }
  const a=await submit(queue,{payload:5}),b=await submit(queue,{payload:7});assert.equal(a.workId,1);assert.equal(b.workId,2);
  const overflow=await settle(scheduler.submitBatchWork(new p.SubmitBatchWorkRequest({queueId:queue.queueId,stateless:new p.StatelessBatchWork({rows:1})})));
  assert.equal(overflow.report.status,p.NativeStatus.NATIVE_STATUS_OVERLOADED);refusals++;
  const d=await call('nextBatchDispatch',ref(queue));
  for(const request of [new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId+1n,error:'stale'}),
    new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId,outcomes:[]}),
    new p.CompleteBatchDispatchRequest({queueId:queue.queueId,dispatchId:d.dispatchId,outcomes:[new p.BatchWorkOutcome({outputs:[new p.Tensor({name:'x',dtype:p.DataType.DATA_TYPE_F32,shape:[2n],inline:new Uint8Array(1)})]}),new p.BatchWorkOutcome()]})]) {
    assert.notEqual((await settle(scheduler.completeBatchDispatch(request))).status,0);assert.deepEqual(await call('nextBatchDispatch',ref(queue)),d);refusals++;
  }
  assert.equal((await settle(scheduler.takeBatchWork(workRef(queue,b.workId)))).report.status,p.NativeStatus.NATIVE_STATUS_BUSY);
  await call('cancelBatchWork',workRef(queue,a.workId));
  assert.equal((await call('takeBatchWork',workRef(queue,a.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
  assert.equal((await settle(scheduler.getBatchWork(workRef(queue,a.workId)))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  await new Promise(r=>setTimeout(r,12));await complete(queue,d);
  assert.equal((await settle(scheduler.getBatchWork(workRef(queue,a.workId)))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  const retained=await call('getBatchWork',workRef(queue,b.workId));assert.deepEqual(inputValues(retained.outputs[0]),[7,0,1]);
  retained.outputs[0].inline.fill(0);assert.deepEqual(inputValues((await call('getBatchWork',workRef(queue,b.workId))).outputs[0]),[7,0,1]);
  let info=await call('getBatchQueue',ref(queue));assert.ok(info.workerBusyNs>=5000000n);assert.ok(info.workerBusyNs<=info.wallNs);
  for(let i=0;i<20;i++){await submit(queue,{payload:i});await complete(queue,await call('nextBatchDispatch',ref(queue)));}
  info=await call('getBatchQueue',ref(queue));assert.equal(info.retainedResults,2);assert.equal(info.activeRecords,0);
  assert.equal((await settle(scheduler.getBatchWork(workRef(queue,b.workId)))).report.status,p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  await call('releaseBatchQueue',ref(queue));await call('releaseBatchQueue',ref(queue));
  assert.equal((await settle(scheduler.getBatchQueue(ref(queue)))).report.status,p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
  results.push({name:'atomic-refusals/owned-results/bounded-retention',refusals});console.log('PASS refusals/owned-results/retention');

  // Fill-first acts per group; a full group passes an unrelated waiting group.
  const fill=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxSlots:2,policy:p.BatchQueuePolicy.BATCH_QUEUE_POLICY_FILL_FIRST,maxWaitNs:200000000n}));
  await submit(fill,{payload:1,modelId:'short'});assert.equal((await call('nextBatchDispatch',ref(fill))).dispatchId,0n);
  await submit(fill,{payload:2,modelId:'full'});await submit(fill,{payload:3,modelId:'full'});
  let full=await call('nextBatchDispatch',ref(fill));assert.equal(full.group.model,'full');await complete(fill,full);
  await new Promise(r=>setTimeout(r,210));full=await call('nextBatchDispatch',ref(fill));assert.equal(full.group.model,'short');await complete(fill,full);
  await call('releaseBatchQueue',ref(fill));results.push({name:'fill-first/group-local-deadline'});console.log('PASS fill-first');

  for(const drain of [false,true]) {
    const q=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxSlots:2,tokenBudget:2,
      cache:new p.BatchCacheOptions({slots:2,pageTokens:2,slotTokenCapacity:8})}));
    const a=await submit(q,{payload:1,promptTokens:5,maxTokens:2}),b=await submit(q,{payload:2,promptTokens:3,maxTokens:1}),c=await submit(q,{payload:3,promptTokens:2,maxTokens:1});
    let d=await call('nextBatchDispatch',ref(q)),firstId=d.dispatchId;
    await call('closeBatchQueue',new p.CloseBatchQueueRequest({queueId:q.queueId,drain}));
    assert.equal((await call('getBatchWork',workRef(q,c.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
    if(drain)for(let i=0;i<20;i++){d=await call('nextBatchDispatch',ref(q));if(!d.dispatchId)break;await complete(q,d);}
    else assert.notEqual((await settle(scheduler.completeBatchDispatch(new p.CompleteBatchDispatchRequest({queueId:q.queueId,dispatchId:firstId,error:'late'})))).status,0);
    for(const work of [a,b])assert.equal((await call('getBatchWork',workRef(q,work.workId))).state,
      drain?p.BatchWorkState.BATCH_WORK_STATE_COMPLETED:p.BatchWorkState.BATCH_WORK_STATE_CANCELLED);
    const info=await call('getBatchQueue',ref(q));assert.equal(info.closed,true);assert.equal(info.draining,false);assert.equal(info.reservedPages,0);assert.equal(info.residentPages,0);
    await call('releaseBatchQueue',ref(q));results.push({name:`close/drain-${drain}`});console.log('PASS close/drain-'+drain);
  }
  {
  // A failed slot retires its dispatch atomically; another group stays usable.
  const q=await call('createBatchQueue',new p.CreateBatchQueueRequest({maxSlots:2,
    cache:new p.BatchCacheOptions({slots:2,pageTokens:2,slotTokenCapacity:8})}));
  const a=await submit(q,{payload:1,promptTokens:2,maxTokens:2,modelId:'a'}),b=await submit(q,{payload:2,promptTokens:2,maxTokens:2,modelId:'b'});
  const bad=await call('nextBatchDispatch',ref(q));await complete(q,bad,{error:'worker failure',outcomes:[]});
  assert.equal((await call('getBatchWork',workRef(q,a.workId))).error,'worker failure');
  for(let i=0;i<6;i++){const d=await call('nextBatchDispatch',ref(q));if(!d.dispatchId)break;await complete(q,d);}
  assert.equal((await call('getBatchWork',workRef(q,b.workId))).state,p.BatchWorkState.BATCH_WORK_STATE_COMPLETED);
  info=await call('getBatchQueue',ref(q));assert.equal(info.failed,1n);assert.equal(info.completed,1n);assert.equal(info.reservedPages,0);
  await call('releaseBatchQueue',ref(q));results.push({name:'failure/independent-groups'});console.log('PASS failure containment');
  }
  await writeFile(args.get('--out')??'batch-queue.json',JSON.stringify({wasm:args.get('--wasm'),results},null,2));
  console.log(`PASS batch queue: ${results.length} scenarios`);
} finally {await host.close();}
