import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture, safetensors} from '../tools/proto_fixture.mjs';
import {observeActivations} from '../tools/calibration/activation_observer.mjs';
test('calibration tooling reads intermediate ranges from the C PTQ service', async () => {
  const graph = {format: 'volvox-graph/v1', dimensions: {},
    inputs: {x: {dtype: 'float32', shape: [1,2]}}, nodes: [{id:'dense',opType:'MatMul',
      inputs:{input:'x',weight:'w'},outputs:{out:{tensor:'y',dtype:'float32',shape:[1,2]}},params:{}}], outputs:['y']};
  const f = await fixture({full: true, wasmUrl: new URL('../dist/0.6.0/volvoxai.wasm', import.meta.url)});
  try {
    const model = await f.load(graph,safetensors([{name:'w',shape:[2,2],data:Float32Array.of(1,0,0,2)}]));
    const observations = {};
    await observeActivations({quantization:f.quantization, modelId:model.modelId, document:graph,
      names:['x','y'], observations, inputSets:[{x:{data:Float32Array.of(-1,3),shape:[1,2]}},
        {x:{data:Float32Array.of(4,-2),shape:[1,2]}}]});
    assert.deepEqual(observations.x,{min:-2,max:4,samples:2,elements:4});
    assert.deepEqual(observations.y,{min:-4,max:6,samples:2,elements:4});
  } finally {await f.close();}
});
