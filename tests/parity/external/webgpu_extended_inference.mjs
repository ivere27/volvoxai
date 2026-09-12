/** Full inference graphs for every differentiable migration fixture. */
import assert from 'node:assert/strict';
import path from 'node:path';
import { definitions, tensor, shard } from './webgpu_operator_fixtures.mjs';
import { runGraph, compare, requirePhysicalDevice } from './webgpu_device_bridge.mjs';
const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
const api = await import(`file://${path.resolve(args.get('--bundle'))}`);
const { device, description } = await requirePhysicalDevice();
const options = { wasm: path.resolve(args.get('--wasm')), device }, records = [];
try {
  for (const fixture of definitions) {
    if (args.has('--only') && !fixture.name.includes(args.get('--only'))) continue;
    const graph = { format:'volvox-graph/v1', dimensions:{}, inputs:{x:{dtype:'float32',shape:fixture.shape}},
      nodes: fixture.nodes, outputs:['logits'], __safetensors:shard(fixture.weights) };
    const batches = [7,8].map(seed => [{name:'x',shape:fixture.shape,data:tensor(fixture.shape,seed).values}]);
    const cpu = await runGraph(api, options, graph, batches, 'wasm');
    const gpu = await runGraph(api, options, graph, batches, 'webgpu');
    for(let round=0;round<batches.length;round++)
      compare(`${fixture.name}/${round}`,gpu.rounds[round].get('logits'),cpu.rounds[round].get('logits'),1e-4);
    assert.ok(gpu.encodes);assert.equal(gpu.route.fallbackNodes,0);
    records.push({name:fixture.name,dispatches:gpu.encodes});
    console.log(`extended inference ${fixture.name}: PASS`);
  }
  assert.ok(records.length);console.log(`extended inference: ${records.length} cases passed`);
  if(args.has('--out'))await Deno.writeTextFile(args.get('--out'),JSON.stringify({adapter:description,records},null,2));
} finally {device.destroy();await device.lost;}
