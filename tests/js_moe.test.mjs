import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph, ModelBuilder } from '../ts/index.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

test('MoERouter selects deterministic top-k experts and MoELinear mixes them', async () => {
  const graph = new Graph();
  const input = graph.addInput('x', [1, 2]);
  const routerWeight = graph.addWeight('router', [2, 3]);
  routerWeight.buffer = Float32Array.from([
    1, 0, -1,
    0, 1, 1,
  ]);
  const routes = graph.addOp('MoERouter', { input, weight: routerWeight }, {
    indices: [1, 2],
    weights: [1, 2],
  }, { num_experts: 3, top_k: 2 });

  const expertWeight = graph.addWeight('experts', [3, 2, 1]);
  expertWeight.buffer = Float32Array.from([
    1, 0,
    0, 2,
    10, 10,
  ]);
  const mixed = graph.addOp('MoELinear', {
    input,
    expert_weight: expertWeight,
    route_indices: routes.indices,
    route_weights: routes.weights,
  }, { out: [1, 1] });
  graph.setOutputs([mixed.out.name]);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ x: Float32Array.from([2, 1]) });

  assert.deepEqual(Array.from(routes.indices.buffer), [0, 1]);
  assert.ok(Math.abs(routes.weights.buffer[0] - 0.7310586) < 1e-6);
  assert.ok(Math.abs(routes.weights.buffer[1] - 0.2689414) < 1e-6);
  assert.ok(Math.abs(result[mixed.out.name][0] - 2) < 1e-6);
});

test('MoERouter rejects invalid top-k', async () => {
  const graph = new Graph();
  const input = graph.addInput('x', [1, 1]);
  const weight = graph.addWeight('router', [1, 2]);
  weight.buffer = Float32Array.from([1, 2]);
  graph.addOp('MoERouter', { input, weight }, { indices: [1, 3], weights: [1, 3] }, {
    num_experts: 2,
    top_k: 3,
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await assert.rejects(() => engine.execute({ x: Float32Array.from([1]) }), /top_k/);
});

test('ModelBuilder exposes MoE and adapter-routing design helpers', () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [2, 4]);
  const router = builder.weight('router', [4, 3], 'float32', new Float32Array(12));
  const experts = builder.weight('experts', [3, 4, 5], 'float32', new Float32Array(60));
  const routes = builder.moeRouter(input, router, { topK: 2, name: 'router_node' });
  const result = builder.moeLinear(input, experts, routes, { name: 'experts_node' });
  builder.outputs(result.out);
  const graph = builder.build();

  assert.deepEqual(graph.nodes.map((node) => node.opType), ['MoERouter', 'MoELinear']);
  assert.deepEqual(routes.indices.shape, [2, 2]);
  assert.deepEqual(result.out.shape, [2, 5]);
  assert.deepEqual(builder.adapterRouting(builder.adapterRoute('tenant', 3, 0.5)), {
    adapters: [{ name: 'tenant', version: 3, scale: 0.5 }],
  });
});
