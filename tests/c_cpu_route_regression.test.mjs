import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture, p, ok, tensors} from '../tools/proto_fixture.mjs';
const wasmUrl=process.env.VOLVOXAI_TEST_FULL_WASM??new URL('../dist/0.4.0/volvoxai.full.wasm',import.meta.url);
async function execute(graph,inputs) {
  const f=await fixture({wasmUrl,full:true});
  try {
    const model=await f.load(graph), compiled=await f.compile(model);
    const result=ok(await f.inference.run(new p.RunRequest({compiledModelId:compiled.compiledModelId,inputs:tensors(inputs)})));
    const outputs=[];
    for(const name of graph.outputs) outputs.push([...await f.read(result,name)]);
    return outputs;
  } finally {await f.close();}
}
test('C CPU AveragePool2D is admitted and computes NHWC channel means',async()=>{
  const graph={format:'volvox-graph/v1',dimensions:{},inputs:{x:{dtype:'float32',shape:[1,4,4,2]}},
    nodes:[{id:'pool',opType:'AveragePool2D',inputs:{input:'x'},outputs:{out:{tensor:'y',dtype:'float32',shape:[1,2,2,2]}},
      params:{kernel:[2,2],stride:[2,2],pads:[0,0,0,0],data_layout:'NHWC'}}],outputs:['y']};
  const output=await execute(graph,{x:{data:Float32Array.from({length:32},(_,i)=>i),shape:[1,4,4,2]}});
  assert.deepEqual(output,[[5,6,9,10,21,22,25,26]]);
});
test('C CPU GatherElements admits smaller index extents and exact axis indexing',async()=>{
  const graph={format:'volvox-graph/v1',dimensions:{},inputs:{x:{dtype:'float32',shape:[3,4]},indices:{dtype:'int32',shape:[3,2]}},
    nodes:[{id:'gather',opType:'GatherElements',inputs:{input:'x',indices:'indices'},
      outputs:{out:{tensor:'y',dtype:'float32',shape:[3,2]}},params:{axis:1}}],outputs:['y']};
  const output=await execute(graph,{x:{data:Float32Array.from({length:12},(_,i)=>i*.5-2),shape:[3,4]},
    indices:{data:Int32Array.from([0,3,2,1,3,0]),shape:[3,2]}});
  assert.deepEqual(output,[[-2,-.5,1,.5,3.5,2]]);
});
test('C CPU Split publishes every output snapshot on wasm32',async()=>{
  const graph={format:'volvox-graph/v1',dimensions:{},inputs:{x:{dtype:'float32',shape:[2,4]}},
    nodes:[{id:'split',opType:'Split',inputs:{input:'x'},outputs:{out0:{tensor:'a',dtype:'float32',shape:[2,1]},
      out1:{tensor:'b',dtype:'float32',shape:[2,3]}},params:{axis:1,split:[1,3]}}],outputs:['a','b']};
  const output=await execute(graph,{x:{data:Float32Array.from({length:8},(_,i)=>i),shape:[2,4]}});
  assert.deepEqual(output,[[0,4],[1,2,3,5,6,7]]);
});
