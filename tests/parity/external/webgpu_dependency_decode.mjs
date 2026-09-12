/** Dependency refresh is distinct from cursor advance, through generated RPCs. */
import assert from 'node:assert/strict';
import path from 'node:path';
import {requirePhysicalDevice} from './webgpu_device_bridge.mjs';
const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);
const api=await import(`file://${path.resolve(args.get('--bundle'))}`),p=api.pb;
const {device,description}=await requirePhysicalDevice();
const shape=[1,4,2],spec={shape,dtype:'float32'};
const node=(id,opType,inputs)=>({id,opType,inputs,outputs:{out:{tensor:id,...spec}},params:{}});
const graph=new TextEncoder().encode(JSON.stringify({format:'volvox-graph/v1',dimensions:{},inputs:{a:spec,b:spec},
  nodes:[node('left','ReLU',{input:'a'}),node('right','ReLU',{input:'b'}),node('sum','Add',{a:'left',b:'right'})],outputs:['sum']}));
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
      const context=ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_AUTO,decodeInputs:['a','b']})));
      const input={a:Float32Array.from({length:8},(_,i)=>i-2),b:Float32Array.from({length:8},(_,i)=>10+i)};
      const tensors=names=>names.map(name=>new p.Tensor({name,dtype:p.DataType.DATA_TYPE_F32,shape:shape.map(BigInt),inline:new Uint8Array(input[name].buffer)}));
      const reference=()=>Float32Array.from(input.a,(v,i)=>Math.max(0,v)+Math.max(0,input.b[i]));
      const state=async()=>ok(await inference.getDecodeState(new p.ExecutionContextRef(context)));
      const read=async result=>{
        let current;const deadline=Date.now()+30_000;
        do {
          current=ok(await inference.getResult(new p.ResultRef(result)));
          assert.ok(Date.now()<deadline);if(current.state===p.ResultState.RESULT_STATE_PENDING)await new Promise(r=>setTimeout(r,1));
        }while(current.state===p.ResultState.RESULT_STATE_PENDING);
        assert.equal(current.state,p.ResultState.RESULT_STATE_READY);
        const output=ok(await inference.readOutput(new p.ReadOutputRequest({resultId:result.resultId,name:'sum'}))).tensor;
        return new Float32Array(output.inline.slice().buffer);
      };
      const retained=ok(await inference.decodePrefill(new p.DecodePrefillRequest({contextId:context.contextId,inputs:tensors(['a','b'])})));
      const original=reference();assert.deepEqual(await read(retained),original);
      input.a[2]=30;input.a[3]=40;
      const row=ok(await inference.decodeStep(new p.DecodeStepRequest({contextId:context.contextId,position:1,inputs:tensors(['a'])})));
      assert.deepEqual(await read(row),reference());ok(await inference.releaseResult(new p.ResultRef(row)));
      const beforeState=await state();
      for(const names of [['a'],[],['b','a']]) {
        for(const name of names)input[name]=Float32Array.from(input[name],v=>v+5);
        const before=encodes;
        const refreshed=ok(await inference.decodeStep(new p.DecodeStepRequest({contextId:context.contextId,
          dependencyUpdate:new p.Empty(),inputs:tensors(names)})));
        assert.deepEqual(await read(refreshed),reference());
        assert.deepEqual((await state()).activeLengths,beforeState.activeLengths);
        if(backend==='webgpu')assert.equal(encodes-before,names.length?names.length+1:0,'dependency closure dispatch count');
        assert.deepEqual(await read(retained),original);
        ok(await inference.releaseResult(new p.ResultRef(refreshed)));
      }
      // The ordinary omitted cursor continues advancing after a dependency refresh.
      input.b[4]+=10;input.b[5]+=10;
      const advanced=ok(await inference.decodeStep(new p.DecodeStepRequest({contextId:context.contextId,inputs:tensors(['b'])})));
      assert.deepEqual(await read(advanced),reference());assert.deepEqual((await state()).activeLengths,[3]);
      ok(await inference.releaseResult(new p.ResultRef(advanced)));
      const required=ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({compiledModelId:compiled.compiledModelId,
        decodeRowMode:p.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true,decodeInputs:['a','b']})));
      const requiredPrefill=ok(await inference.decodePrefill(new p.DecodePrefillRequest({contextId:required.contextId,inputs:tensors(['a','b'])})));
      await read(requiredPrefill);
      for(const ctx of [required,context]) {
        if(ctx===context)ok(await inference.resetDecode(new p.ExecutionContextRef(ctx)));
        const before=encodes;
        const rejected=await inference.decodeStep(new p.DecodeStepRequest({contextId:ctx.contextId,dependencyUpdate:new p.Empty(),inputs:tensors(['a'])}));
        assert.equal(rejected.report.status,p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);assert.equal(encodes,before);
        assert.deepEqual(await read(retained),original);
      }
      results.push({backend,dependencyUpdates:3,implicitAdvance:true,refusals:2,retainedSnapshots:true});
      console.log(`dependency decode ${backend}: PASS`);
    }finally{await host.close();}
  }
  if(args.has('--out'))await Deno.writeTextFile(args.get('--out'),JSON.stringify({adapter:description,results},null,2));
}finally{device.destroy();await device.lost;}
