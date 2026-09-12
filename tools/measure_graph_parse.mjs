/** Measure actual C parsing/allocation calls around generated CompileModel. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {ModelControlWasmDispatchFactory} from '../ts/core/ModelControlWasm.js';
import {VxInferenceServiceClient} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as p from '../runtime/generated/typescript/inference/volvoxai_lite.js';
const graph = {format:'volvox-graph/v1',dimensions:{},inputs:{x:{dtype:'float32',shape:[1,16]}},
  nodes:Array.from({length:100},(_,i)=>({id:`relu-${i}`,opType:'ReLU',inputs:{input:i?`h${i-1}`:'x'},
    outputs:{out:{tensor:`h${i}`,dtype:'float32',shape:[1,16]}},params:{}})),outputs:['h99']};
const bytes = new TextEncoder().encode(JSON.stringify(graph));
const ok = value => {assert.equal((value.report??value).status,0,(value.report??value).message);return value;};
const results = {};
for (const variant of ['reparse','shared']) {
  const module = new WebAssembly.Module(readFileSync(`build/graph-parse-probe/${variant}.wasm`));
  let exports;
  class ObservedFactory extends ModelControlWasmDispatchFactory {
    instantiate(wakeup) {const instance=super.instantiate(wakeup);exports=instance.exports;return instance;}
  }
  const factory = new ObservedFactory(module);
  const host = factory.create();
  host.mountFile('graph.json', bytes);
  const api = new VxInferenceServiceClient(host);
  try {
    const runtime=ok(await api.createRuntime(new p.CreateRuntimeRequest()));
    const model=ok(await api.loadModel(new p.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json'})));
    const pointer=exports.alloc_bytes(bytes.length+1);
    const memory=new Uint8Array(exports.memory.buffer);memory.set(bytes,pointer);memory[pointer+bytes.length]=0;
    exports.probe_source(pointer);
    const samples=[];
    for(let repeat=0;repeat<10;repeat++) {
      exports.probe_reset();const start=performance.now();
      const compiled=ok(await api.compileModel(new p.CompileModelRequest({modelId:model.modelId})));
      const milliseconds=performance.now()-start;
      samples.push({parseCalls:Number(exports.probe_count(0)),jsonNodeAllocations:Number(exports.probe_count(1)),
        mallocCalls:Number(exports.probe_count(2)),requestedMallocBytes:Number(exports.probe_count(3)),milliseconds});
      ok(await api.releaseCompiledModel(new p.CompiledModelRef(compiled)));
    }
    results[variant]=samples;
  } finally {await host.close();}
}
assert.equal(results.reparse[0].parseCalls, results.shared[0].parseCalls+1);
console.log(JSON.stringify({format:'volvoxai-graph-parse-measurement/v1', nodes:100,
  scope:'CompileModel after LoadModel. Baseline reparses retained source bytes at lowering; excludes the old file read. malloc includes C runtime allocation, not host JS. Instrumented builds only.',
  results},null,2));
