/** Dependency refresh is distinct from cursor advance, through generated RPCs. */
import assert from 'node:assert/strict';
import {declareIncremental,graphRequest,graphContext,inlineOutput} from './execution_fixture.mjs';
import path from 'node:path';
import {requirePhysicalDevice} from './webgpu_device_bridge.mjs';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(`file://${path.resolve(args.get('--bundle'))}`),p=api.pb;
const {device,description}=await requirePhysicalDevice();
const shape=[1,4,2],spec={shape,dtype:'float32'};
const node=(id,opType,inputs)=>({id,opType,inputs,outputs:{out:{tensor:id,...spec}},params:{}});
const document=declareIncremental({format:'volvox-graph/v1',dimensions:{},inputs:{a:spec,b:spec},
  nodes:[node('left','ReLU',{input:'a'}),node('right','ReLU',{input:'b'}),node('sum','Add',{a:'left',b:'right'})],outputs:['sum']});
const graph=new TextEncoder().encode(JSON.stringify(document));
const ok=v=>{assert.equal((v.report??v).status,0,JSON.stringify(v.report??v,(_,v)=>typeof v==='bigint'?String(v):v));return v;};
const results=[];
try {
  for(const backend of ['wasm','webgpu']) {
    let encodes=0;
    const real=await api.createWebGPUHostBridge({device});
    const bridge={...real,imports:{...real.imports,vx_gpu_encode(...args){encodes++;return real.imports.vx_gpu_encode(...args);}}};
    const host=new(api.FullEngineHost??api.InferenceWasmHost)({wasmUrl:path.resolve(args.get('--wasm')),gpuBridge:bridge,fetch:async()=>new Response(graph)});
    try {
      const inference=new api.VxInferenceServiceClient(host);
      const runtime=ok(await inference.createRuntime(new p.CreateRuntimeRequest()));
      const model=ok(await inference.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json'})));
      const compiled=ok(await inference.compileModel(new p.CompileModelRequest({modelId:model.modelId,
        policy:new p.BackendPolicy({mode:p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:[backend],operatorFallback:p.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
      const context=ok(await inference.createExecutionContext(graphContext(p,compiled.compiledModelId,{rowMode:p.DecodeRowMode.DECODE_ROW_MODE_AUTO})));
      const input={a:Float32Array.from({length:8},(_,i)=>i-2),b:Float32Array.from({length:8},(_,i)=>10+i)};
      const tensors=names=>names.map(name=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_F32,shape:shape.map(BigInt),inline:new Uint8Array(input[name].buffer)}));
      const reference=()=>Float32Array.from(input.a,(v,i)=>Math.max(0,v)+Math.max(0,input.b[i]));
      const state=async()=>ok(await inference.getExecutionContextInfo(new p.ExecutionContextRef(context)));
      const read=async result=>inlineOutput(result,'sum');
      const retained=ok(await inference.execute(graphRequest(p,document,context.contextId,tensors(['a','b']),{entrypoint:'initialize'})));
      const original=reference();assert.deepEqual(await read(retained),original);
      input.a[2]=30;input.a[3]=40;
      const row=ok(await inference.execute(graphRequest(p,document,context.contextId,tensors(['a']),{entrypoint:'transition',positions:[1]})));
      assert.deepEqual(await read(row),reference());
      const beforeState=await state();
      for(const names of [['a'],[],['b','a']]) {
        for(const name of names)input[name]=Float32Array.from(input[name],v=>v+5);
        const before=encodes;
        const refreshed=ok(await inference.execute(graphRequest(p,document,context.contextId,tensors(names),{entrypoint:'transition',roots:names})));
        assert.deepEqual(await read(refreshed),reference());
        assert.deepEqual((await state()).activeLengths,beforeState.activeLengths);
        if(backend==='webgpu')assert.equal(encodes-before,names.length?names.length+1:0,'dependency closure dispatch count');
        assert.deepEqual(await read(retained),original);
      }
      // The next explicit row continues advancing after a dependency refresh.
      input.b[4]+=10;input.b[5]+=10;
      const advanced=ok(await inference.execute(graphRequest(p,document,context.contextId,tensors(['b']),{entrypoint:'transition',positions:[2]})));
      assert.deepEqual(await read(advanced),reference());assert.deepEqual((await state()).activeLengths,[3]);
      const required=ok(await inference.createExecutionContext(graphContext(p,compiled.compiledModelId,{rowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED})));
      await read(ok(await inference.execute(graphRequest(p,document,required.contextId,tensors(['a','b']),{entrypoint:'initialize'}))));
      const fresh=ok(await inference.createExecutionContext(graphContext(p,compiled.compiledModelId,{rowMode:p.DecodeRowMode.DECODE_ROW_MODE_AUTO})));
      for(const ctx of [required,fresh]) {
        const before=encodes;
        const rejected=await inference.execute(graphRequest(p,document,ctx.contextId,tensors(['a']),
          {entrypoint:'transition',roots:['a']})).then(()=>assert.fail('dependency update was admitted'),error=>error);
        assert.equal(rejected.report.status,p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);assert.equal(encodes,before);
        assert.deepEqual(await read(retained),original);
      }
      results.push({backend,dependencyUpdates:3,explicitAdvance:true,refusals:2,retainedSnapshots:true});
      console.log(`dependency decode ${backend}: PASS`);
    }finally{await host.close();}
  }
  if(args.has('--out'))await Deno.writeTextFile(args.get('--out'),JSON.stringify({adapter:description,results},null,2));
}finally{device.destroy();await device.lost;}
