import test from 'node:test';
import assert from 'node:assert/strict';

import { Trainer } from '../ts/full.js';
import { WebGPUAutograd } from '../ts/training/WebGPUAutograd.js';
import { TrainingGraph as Graph } from '../ts/training/TrainingGraph.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { _cpuDropout, dropoutContext, dropoutHash } from '../ts/ops/dropout.js';
import { attentionProbabilityIndex } from '../ts/ops/attentionDropout.js';
import { logicalSnapshotFromTrainingGraph } from './helpers/training_fixture.mjs';

// Node does not expose WebGPU constants. These values are only bit masks for
// the host-side mock; their numeric values are irrelevant to these tests.
globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});

function tensor(name, shape, { dtype = 'float32', buffer = null, isWeight = false, isInput = false } = {}) {
  const sizeBytes = shape.reduce((a, b) => a * b, 1) * 4;
  return { name, shape, dtype, sizeBytes, buffer, isWeight, isInput };
}

function mockDevice() {
  const state = {
    buffers: [], bindGroups: [], shaderModules: 0, pipelines: 0, workDone: 0,
    failNextBuffer: false,
  };
  const device = {
    state,
    createBuffer(descriptor) {
      if (state.failNextBuffer) {
        state.failNextBuffer = false;
        throw new Error('injected WebGPU allocation failure');
      }
      const value = {
        descriptor,
        size: descriptor.size,
        bytes: new Uint8Array(descriptor.size),
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      state.buffers.push(value);
      return value;
    },
    createShaderModule() { state.shaderModules++; return {}; },
    async createComputePipelineAsync() {
      state.pipelines++;
      return { getBindGroupLayout() { return {}; } };
    },
    createBindGroup(descriptor) {
      state.bindGroups.push(descriptor);
      return descriptor;
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
          };
        },
        finish() { return {}; },
      };
    },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        const baseOffset = source instanceof ArrayBuffer ? sourceOffset : source.byteOffset + sourceOffset;
        const available = (source instanceof ArrayBuffer ? source.byteLength : source.byteLength) - sourceOffset;
        const bytes = new Uint8Array(source instanceof ArrayBuffer ? source : source.buffer,
          baseOffset, size ?? available);
        destination.bytes.set(bytes, offset);
      },
      submit() {},
      async onSubmittedWorkDone() { state.workDone++; },
    },
  };
  return device;
}

class DispatchRecorder extends WebGPUAutograd {
  async _dispatch(shaderName, entryPoint, entries, workgroupCount, resources = []) {
    return { shaderName, entryPoint, entries, workgroupCount, resources };
  }
}

function makeTrainer(nodes, tensors) {
  const device = mockDevice();
  const gpuBuffers = new Map(tensors.map((value) => [value.name, { tensor: value.name }]));
  const graph = {
    nodes,
    getTensor(name) { return tensors.find((value) => value.name === name); },
  };
  const trainer = new DispatchRecorder(device, graph, { gpuBuffers });
  return { trainer, device };
}

test('WebGPUAutograd construction is lazy and allocates no training resources', () => {
  const device = mockDevice();
  const graph = { nodes: [] };
  const trainer = new WebGPUAutograd(device, graph, { gpuBuffers: new Map() });
  assert.equal(device.state.buffers.length, 0);
  assert.equal(device.state.shaderModules, 0);
  assert.equal(device.state.pipelines, 0);
  trainer.dispose();
});

test('Linear backward dispatches odd 16x16 gradient tiles and a 64-wide bias row', async () => {
  const input = tensor('linear_input', [17, 19]);
  const weight = tensor('linear_weight', [23, 19]);
  const bias = tensor('linear_bias', [23]);
  const output = tensor('linear_output', [17, 23]);
  const node = {
    id: 'linear_tiled_backward', opType: 'Linear', wLayout: 'dout',
    inputs: { input, weight, bias }, outputs: { out: output }, params: {},
  };
  const { trainer } = makeTrainer([node], [input, weight, bias, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_output' });

  const dispatches = await trainer._buildBackwardDispatches();

  assert.deepEqual(dispatches.map(({ shaderName, entryPoint, workgroupCount }) => ({
    shader: `${shaderName}.${entryPoint}`, workgroupCount,
  })), [
    { shader: 'matMulBackward.input_main', workgroupCount: [2, 2, 1] },
    { shader: 'matMulBackward.weight_main', workgroupCount: [2, 2, 1] },
    { shader: 'matMulBackward.bias_main', workgroupCount: [1, 1, 1] },
  ]);
  const params = dispatches[0].entries.find(([binding]) => binding === 6)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer)], [17, 19, 23, 1, 0, 0, 0, 0]);
  trainer.dispose();
});

test('WebGPU input-major Linear backward and optimizer upload preserve canonical storage', async () => {
  const input = tensor('linear_input', [2, 3]);
  const weight = tensor('linear_weight', [3, 4], {
    buffer: Float32Array.from({ length: 12 }, (_, index) => index + 1),
    isWeight: true,
  });
  const output = tensor('linear_output', [2, 4]);
  const node = {
    id: 'linear_input_major', opType: 'MatMul', wLayout: 'din',
    inputs: { input, weight }, outputs: { out: output },
    params: { weight_layout: 'din_dout' },
  };
  const { trainer, device } = makeTrainer([node], [input, weight, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_output' });

  const dispatches = await trainer._buildBackwardDispatches();
  const params = dispatches[0].entries.find(([binding]) => binding === 6)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer)], [2, 3, 4, 0, 1, 1, 0, 0]);

  const gpuWeight = device.createBuffer({ size: weight.sizeBytes });
  trainer.executor.gpuBuffers.set(weight.name, gpuWeight);
  trainer._uploadUpdatedTensor(weight);
  assert.deepEqual(
    [...new Float32Array(gpuWeight.bytes.buffer)],
    [...weight.buffer],
    'updated weights must not be transposed away from the direct input-major forward route',
  );
  trainer.dispose();
});

test('WebGPU backward derives layout per consumer of a shared square weight', async () => {
  const dinInput = tensor('din_input', [2, 3]);
  const doutInput = tensor('dout_input', [2, 3]);
  const sharedWeight = tensor('shared_square_weight', [3, 3], { isWeight: true });
  const dinOutput = tensor('din_output', [2, 3]);
  const doutOutput = tensor('dout_output', [2, 3]);
  const dinNode = {
    id: 'shared_din_consumer', opType: 'MatMul', wLayout: 'din',
    inputs: { input: dinInput, weight: sharedWeight }, outputs: { out: dinOutput },
    params: { weight_layout: 'din_dout' },
  };
  const doutNode = {
    id: 'shared_dout_consumer', opType: 'Linear', wLayout: 'dout',
    inputs: { input: doutInput, weight: sharedWeight }, outputs: { out: doutOutput },
    params: { weight_layout: 'dout_din' },
  };
  const { trainer } = makeTrainer(
    [dinNode, doutNode],
    [dinInput, doutInput, sharedWeight, dinOutput, doutOutput],
  );
  trainer.gradientBuffers.set(dinOutput.name, { tensor: 'grad_din_output' });
  trainer.gradientBuffers.set(doutOutput.name, { tensor: 'grad_dout_output' });

  const dispatches = await trainer._buildBackwardDispatches();
  const inputDispatches = dispatches.filter(({ shaderName, entryPoint }) =>
    shaderName === 'matMulBackward' && entryPoint === 'input_main');
  const layoutFlags = inputDispatches.map((dispatch) => {
    const params = dispatch.entries.find(([binding]) => binding === 6)[1];
    return [...new Uint32Array(params.bytes.buffer).slice(4, 6)];
  });

  assert.deepEqual(layoutFlags, [[0, 0], [1, 1]],
    'reverse traversal must preserve each consumer node layout, not tensor-name metadata');
  trainer.dispose();
});

test('full-profile Trainer owns WebGPU training without exposing a concrete graph', async () => {
  const device = mockDevice();
  const graph = new Graph();
  const input = graph.addInput('x', [1]);
  const output = graph.addOp('Identity', { input }, {
    out: { name: 'y', shape: [1] },
  }).out;
  graph.setOutputs(output);

  const snapshot = logicalSnapshotFromTrainingGraph(graph);
  const trainer = await Trainer.create(snapshot, {
    backend: 'webgpu',
    device,
  });
  assert.ok(trainer instanceof Trainer);
  assert.equal(trainer.backend, 'webgpu');
  assert.equal(Object.hasOwn(trainer, 'graph'), false);

  await trainer.close();
  await trainer.close();
});

test('WebGPU training reuses detached forward/backward plans and one growable arena on shape return', async () => {
  const device = mockDevice();
  const makeGraph = (batch, opType = 'Identity') => {
    const graph = new Graph();
    const input = graph.addInput('x', [batch, 2]);
    const output = graph.addOp(opType, { input }, {
      out: { name: 'y', shape: [batch, 2] },
    }).out;
    graph.setOutputs(output);
    return graph;
  };
  const largest = makeGraph(4);
  const executor = new GraphExecutor(device, largest, {
    shaderLibrary: {
      getCopyShader() { return '@compute @workgroup_size(1) fn main() {}'; },
      getCopy32Shader() { return '@compute @workgroup_size(1) fn main() {}'; },
    },
  });
  await executor.compile();
  const trainer = new WebGPUAutograd(device, largest, executor);
  await trainer.rebind(largest, { shapeSignature: 'B=4', tacticSignature: 'identity' });
  await trainer._buildBackwardDispatches();
  const pipelineBuilds = device.state.pipelines;
  const largestBytes = executor.inspectDynamicResources().activationCapacityBytes;

  await trainer.rebind(makeGraph(1), { shapeSignature: 'B=1', tacticSignature: 'identity' });
  await trainer._buildBackwardDispatches();
  const small = executor.inspectDynamicResources();
  await trainer.rebind(makeGraph(4), { shapeSignature: 'B=4', tacticSignature: 'identity' });
  await trainer._buildBackwardDispatches();
  const returned = executor.inspectDynamicResources();
  const planCache = trainer.inspectPlanCache();

  assert.equal(device.state.pipelines, pipelineBuilds,
    'shape return must reuse device-cached compute pipelines');
  assert.equal(small.activationCapacityBytes, largestBytes);
  assert.equal(returned.activationCapacityBytes, largestBytes,
    'one context-owned capacity pool serves both concrete shapes');
  assert.ok(returned.specializationRebindCount >= 2);
  assert.deepEqual({
    entries: planCache.entries,
    hits: planCache.hits,
    misses: planCache.misses,
    recipeBuilds: planCache.recipeBuilds,
    forwardPlanBuilds: planCache.forwardPlanBuilds,
    backwardPlanBuilds: planCache.backwardPlanBuilds,
    forwardMaterializations: planCache.forwardMaterializations,
    backwardMaterializations: planCache.backwardMaterializations,
  }, {
    entries: 2,
    hits: 1,
    misses: 2,
    recipeBuilds: 2,
    forwardPlanBuilds: 2,
    backwardPlanBuilds: 2,
    forwardMaterializations: 3,
    backwardMaterializations: 3,
  });
  assert.ok(planCache.metadataBytes <= planCache.metadataLimitBytes);

  await assert.rejects(
    trainer.rebind(makeGraph(1), { shapeSignature: 'B=4', tacticSignature: 'identity' }),
    /concrete tensor descriptors do not match/i,
  );
  assert.deepEqual(trainer.inspectPlanCache(), planCache,
    'failed cached-plan materialization must preserve telemetry and LRU state');

  const beforeResourceFailure = trainer.inspectPlanCache();
  device.state.failNextBuffer = true;
  await assert.rejects(
    trainer.rebind(makeGraph(8), { shapeSignature: 'B=8', tacticSignature: 'identity' }),
    /injected WebGPU allocation failure/,
  );
  assert.deepEqual(trainer.inspectPlanCache(), beforeResourceFailure,
    'failed resource staging must not publish or reorder a backend plan recipe');
  assert.equal(executor.inspectDynamicResources().shapeSignature, 'B=4',
    'failed resource staging must leave the prior concrete generation executable');

  await trainer.rebind(makeGraph(4, 'Reshape'), {
    shapeSignature: 'B=4',
    tacticSignature: 'reshape',
  });
  const changedTopology = trainer.inspectPlanCache();
  assert.equal(changedTopology.entries, 1);
  assert.equal(changedTopology.evictions, 2,
    'a different logical topology invalidates every prior context-owned recipe');
  trainer.dispose();
  executor.dispose();
});

test('WebGPU inference compiles Dropout away while the Trainer owns its forward pipeline', async () => {
  const device = mockDevice();
  const input = tensor('dropout_input', [16], { isInput: true });
  const output = tensor('dropout_output', [16]);
  const node = {
    id: 'dropout', opType: 'Dropout', inputs: { input }, outputs: { out: output }, params: { p: 0.5 },
  };
  const graph = {
    nodes: [node], tensors: new Map([[input.name, input], [output.name, output]]),
    topologyRevision: 0, weightRevision: 0,
    adapters: null,
  };
  const executor = new GraphExecutor(device, graph, {
    shaderLibrary: {
      getCopyShader() { return '@compute @workgroup_size(1) fn main() {}'; },
      getCopy32Shader() { return '@compute @workgroup_size(1) fn main() {}'; },
    },
  });

  await executor.compile();
  assert.equal(executor.pipelines.length, 0, 'inference must add no Dropout dispatch');
  assert.equal(executor.gpuBuffers.get(output.name), executor.gpuBuffers.get(input.name));

  assert.equal(executor.prepareForTraining, undefined,
    'the inference executor exposes no training preparation');

  const trainer = new WebGPUAutograd(device, graph);
  await trainer._ensureForwardCompiled();
  assert.equal(trainer.executor.pipelines.length, 1,
    'the Trainer-owned executor compiles one replaceable Dropout dispatch');
  assert.notEqual(
    trainer.executor.gpuBuffers.get(output.name),
    trainer.executor.gpuBuffers.get(input.name),
  );
  trainer.dispose();
  executor.dispose();
});

test('WebGPU GELU encodes exact-erf by default and tanh only when requested', async () => {
  const device = mockDevice();
  const input = tensor('gelu_input', [8]);
  const output = tensor('gelu_output', [8]);
  const node = {
    id: 'gelu', opType: 'GELU', inputs: { input }, outputs: { out: output }, params: {},
  };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getGELUShader() { return 'gelu'; } },
  });
  executor.gpuBuffers.set(input.name, { tensor: input.name });
  executor.gpuBuffers.set(output.name, { tensor: output.name });
  await executor._buildNodePipeline(node);
  let params = device.state.bindGroups.at(-1).entries.find(({ binding }) => binding === 2).resource.buffer;
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 2)], [8, 0]);
  node.params.approximate = 'tanh';
  await executor._buildNodePipeline(node);
  params = device.state.bindGroups.at(-1).entries.find(({ binding }) => binding === 2).resource.buffer;
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 2)], [8, 1]);

  const { trainer } = makeTrainer([node], [input, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_output' });
  let backward = await trainer._buildBackwardDispatches();
  params = backward[0].entries.find(([binding]) => binding === 4)[1];
  assert.equal(new Uint32Array(params.bytes.buffer)[1], 9);
  node.params.approximate = 'none';
  backward = await trainer._buildBackwardDispatches();
  params = backward[0].entries.find(([binding]) => binding === 4)[1];
  assert.equal(new Uint32Array(params.bytes.buffer)[1], 1);
  executor.dispose();
  trainer.dispose();
});

test('WebGPU training Dropout encodes the same deterministic mask for forward and backward', async () => {
  const input = tensor('dropout_input', [16]);
  const output = tensor('dropout_output', [16]);
  const node = {
    id: 'dropout', opType: 'Dropout', inputs: { input }, outputs: { out: output },
    params: { p: 0.5, seed: 9 },
  };
  const { trainer } = makeTrainer([node], [input, output]);
  trainer.executor.pipelines = [{ graphNodeIndex: 0 }];
  const context = dropoutContext({ seed: 17, counter: 23 });
  const overrides = await trainer._buildDropoutForwardOverrides(context);
  assert.equal(overrides.size, 1);
  const forward = overrides.get(0);
  assert.equal(forward.shaderName, 'dropout');
  assert.equal(forward.entryPoint, 'main');
  const forwardParams = forward.entries.find(([binding]) => binding === 2)[1];
  const forwardU32 = new Uint32Array(forwardParams.bytes.buffer);
  const forwardF32 = new Float32Array(forwardParams.bytes.buffer);
  const effectiveSeed = (17 ^ 9 ^ Math.imul(1, 0x9e3779b9)) >>> 0;
  assert.deepEqual([...forwardU32.slice(0, 4)], [16, 0x80000000, effectiveSeed, 23]);
  assert.equal(forwardF32[4], 2);

  trainer.gradientBuffers.set(output.name, { tensor: 'grad_dropout_output' });
  const backward = (await trainer._buildBackwardDispatches(context))[0];
  assert.equal(backward.shaderName, 'dropoutBackward');
  const backwardParams = backward.entries.find(([binding]) => binding === 2)[1];
  assert.deepEqual(
    [...new Uint32Array(backwardParams.bytes.buffer).slice(0, 4)],
    [...forwardU32.slice(0, 4)],
  );

  const cpuInput = { name: 'x', shape: [16], dtype: 'float32', buffer: Float32Array.from({ length: 16 }, (_, i) => i + 1) };
  const cpuOutput = { name: 'y', shape: [16], dtype: 'float32', buffer: new Float32Array(16) };
  _cpuDropout({ ...node, inputs: { input: cpuInput }, outputs: { out: cpuOutput } }, {
    training: { dropout: context }, nodeIndex: 0,
  });
  for (let index = 0; index < 16; index++) {
    const shaderKeep = dropoutHash(index, effectiveSeed, forwardU32[3], -1) >= forwardU32[1];
    assert.equal(cpuOutput.buffer[index], shaderKeep ? cpuInput.buffer[index] * 2 : 0);
  }
});

test('dispose releases only a trainer-owned forward executor', () => {
  const device = mockDevice();
  const graph = { nodes: [] };
  const ownedTrainer = new WebGPUAutograd(device, graph);
  const ownedBuffer = { destroyed: false, destroy() { this.destroyed = true; } };
  ownedTrainer.executor.gpuBuffers.set('owned', ownedBuffer);
  ownedTrainer.dispose();
  assert.equal(ownedBuffer.destroyed, true);

  let externalDisposals = 0;
  const external = {
    gpuBuffers: new Map(),
    dispose() { externalDisposals++; },
  };
  const sharedTrainer = new WebGPUAutograd(device, graph, external);
  sharedTrainer.dispose();
  assert.equal(externalDisposals, 0);
});

test('topology changes retire stale gradient buffers before reuse', () => {
  const value = tensor('value', [2]);
  const { trainer } = makeTrainer([], [value]);
  trainer.graph.topologyRevision = 0;
  const stale = trainer._gradient(value);
  assert.equal(stale.destroyed, false);
  trainer.graph.topologyRevision = 1;
  trainer._resetGradientsForTopology();
  assert.equal(stale.destroyed, true);
  assert.equal(trainer.gradientBuffers.size, 0);
  trainer.dispose();
});

test('binary backward records true NumPy broadcast strides', async () => {
  const a = tensor('a', [2, 1]);
  const b = tensor('b', [1, 3]);
  const out = tensor('out', [2, 3]);
  const node = { id: 'add', opType: 'Add', inputs: { a, b }, outputs: { out }, params: {} };
  const { trainer } = makeTrainer([node], [a, b, out]);
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map(({ entryPoint }) => entryPoint), ['a_main', 'b_main']);
  const params = dispatches[0].entries.find(([binding]) => binding === 5)[1];
  const raw = new Uint32Array(params.bytes.buffer);
  assert.deepEqual([...raw.slice(0, 5)], [2, 6, 2, 3, 0]);
  assert.deepEqual([...raw.slice(8, 10)], [3, 1]);
  assert.deepEqual([...raw.slice(16, 18)], [1, 0]);
  assert.deepEqual([...raw.slice(24, 26)], [0, 1]);
});

test('MoE backward materializes router-logit gradients before parameter gradients', async () => {
  const input = tensor('x', [2, 2]);
  const routerWeight = tensor('router_w', [2, 3]);
  const routerBias = tensor('router_b', [3]);
  const indices = tensor('indices', [2, 2]);
  const gates = tensor('gates', [2, 2]);
  const expertWeight = tensor('experts', [3, 2, 1]);
  const output = tensor('out', [2, 1]);
  const router = {
    id: 'router', opType: 'MoERouter', inputs: { input, weight: routerWeight, bias: routerBias },
    outputs: { indices, weights: gates }, params: { num_experts: 3, top_k: 2, normalize: false, temperature: 2 },
  };
  const linear = {
    id: 'experts', opType: 'MoELinear',
    inputs: { input, expert_weight: expertWeight, route_indices: indices, route_weights: gates },
    outputs: { out: output }, params: {},
  };
  const tensors = [input, routerWeight, routerBias, indices, gates, expertWeight, output];
  const { trainer } = makeTrainer([router, linear], tensors);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map(({ shaderName, entryPoint }) => `${shaderName}.${entryPoint}`), [
    'moeLinearBackward.input_main',
    'moeLinearBackward.weight_main',
    'moeLinearBackward.route_main',
    'moeRouterBackward.logit_main',
    'moeRouterBackward.input_main',
    'moeRouterBackward.weight_main',
    'moeRouterBackward.bias_main',
  ]);
  const logit = dispatches[3];
  assert.deepEqual(logit.entries.map(([binding]) => binding), [0, 1, 2, 3, 4, 5, 6, 10]);
  const scratch = logit.entries.find(([binding]) => binding === 6)[1];
  assert.equal(dispatches[4].entries.find(([binding]) => binding === 6)[1], scratch);
  assert.equal(dispatches[5].entries.find(([binding]) => binding === 6)[1], scratch);
  const params = logit.entries.find(([binding]) => binding === 10)[1];
  assert.equal(new Uint32Array(params.bytes.buffer)[4], 0);
  assert.equal(new Float32Array(params.bytes.buffer)[6], 2);
});

test('MoE backward binds an exact bidirectional partial-residency table', async () => {
  const input = tensor('x', [1, 2]);
  const indices = tensor('indices', [1, 2]);
  const gates = tensor('gates', [1, 2]);
  const expertWeight = tensor('experts', [2, 2, 1]);
  const output = tensor('out', [1, 1]);
  const linear = {
    id: 'resident_experts', opType: 'MoELinear',
    residentSlots: [2, 5], residentSlotDomain: 8,
    inputs: { input, expert_weight: expertWeight, route_indices: indices, route_weights: gates },
    outputs: { out: output }, params: {},
  };
  const { trainer } = makeTrainer([linear], [input, indices, gates, expertWeight, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.equal(dispatches.length, 3);
  for (const dispatch of dispatches) {
    assert.ok(dispatch.entries.some(([binding]) => binding === 10));
    assert.ok(dispatch.entries.some(([binding]) => binding === 11));
    assert.ok(dispatch.entries.some(([binding]) => binding === 12));
  }
  const slotRows = dispatches[0].entries.find(([binding]) => binding === 10)[1];
  const rowSlots = dispatches[0].entries.find(([binding]) => binding === 11)[1];
  const params = dispatches[0].entries.find(([binding]) => binding === 12)[1];
  assert.deepEqual([...new Uint32Array(slotRows.bytes.buffer).slice(0, 8)],
    [0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff, 1, 0xffffffff, 0xffffffff]);
  assert.deepEqual([...new Uint32Array(rowSlots.bytes.buffer).slice(0, 2)], [2, 5]);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 8)], [1, 2, 1, 2, 2, 0, 8, 0]);
});

test('MoE backward rejects top-k beyond staged experts despite a high global slot id', async () => {
  const input = tensor('x', [1, 2]);
  const indices = tensor('indices', [1, 2]);
  const gates = tensor('gates', [1, 2]);
  const expertWeight = tensor('experts', [1, 2, 1]);
  const output = tensor('out', [1, 1]);
  const linear = {
    id: 'resident_top_k', opType: 'MoELinear',
    residentSlots: [100], residentSlotDomain: 128,
    inputs: { input, expert_weight: expertWeight, route_indices: indices, route_weights: gates },
    outputs: { out: output }, params: {},
  };
  const { trainer } = makeTrainer([linear], [input, indices, gates, expertWeight, output]);
  trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });

  await assert.rejects(
    trainer._buildBackwardDispatches(),
    /resident_top_k has incompatible expert or routing shapes/,
  );
});

test('MoE backward requires the exact resident slot domain used by forward', async () => {
  const input = tensor('x', [1, 2]);
  const indices = tensor('indices', [1, 1]);
  const gates = tensor('gates', [1, 1]);
  const expertWeight = tensor('experts', [1, 2, 1]);
  const output = tensor('out', [1, 1]);

  for (const residentSlotDomain of [undefined, 5]) {
    const linear = {
      id: 'resident_domain', opType: 'MoELinear', residentSlots: [5],
      ...(residentSlotDomain === undefined ? {} : { residentSlotDomain }),
      inputs: { input, expert_weight: expertWeight, route_indices: indices, route_weights: gates },
      outputs: { out: output }, params: {},
    };
    const { trainer } = makeTrainer(
      [linear], [input, indices, gates, expertWeight, output],
    );
    trainer.gradientBuffers.set(output.name, { tensor: 'grad_out' });
    await assert.rejects(
      trainer._buildBackwardDispatches(),
      /resident_domain has invalid resident slot-domain metadata/,
    );
  }
});

test('batched SDPA backward encodes batch and dispatches every gradient component', async () => {
  const qkv = tensor('qkv', [2, 3, 12]);
  const mask = tensor('mask', [2, 3], { dtype: 'int32' });
  const out = tensor('out', [2, 3, 4]);
  const node = {
    id: 'self', opType: 'SDPA', inputs: { qkv, mask }, outputs: { out },
    params: { heads: 2, scale: 0.25, causal: false },
  };
  const { trainer } = makeTrainer([node], [qkv, mask, out]);
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const [dispatch] = await trainer._buildBackwardDispatches();
  assert.equal(dispatch.entryPoint, 'main');
  assert.deepEqual(dispatch.workgroupCount, [3, 2, 6]);
  assert.deepEqual(dispatch.entries.map(([binding]) => binding), [0, 1, 2, 3, 4]);
  const params = dispatch.entries.find(([binding]) => binding === 4)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 5)], [3, 4, 2, 2, 2]);
  assert.equal(new Float32Array(params.bytes.buffer)[5], 0.25);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(6, 8)], [0, 2]);
});

test('batched CrossSDPA backward creates independent q, k, and v dispatches', async () => {
  const q = tensor('q', [3, 2, 4]);
  const k = tensor('k', [3, 3, 4]);
  const v = tensor('v', [3, 3, 4]);
  const mask = tensor('mask', [3, 2, 3], { dtype: 'int32' });
  const out = tensor('out', [3, 2, 4]);
  const node = {
    id: 'cross', opType: 'CrossSDPA', inputs: { q, k, v, mask }, outputs: { out },
    params: { heads: 2, scale: 0.375, causal: true },
  };
  const { trainer } = makeTrainer([node], [q, k, v, mask, out]);
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map(({ entryPoint, workgroupCount }) => [entryPoint, workgroupCount]), [
    ['q_main', [2, 2, 3]],
    ['k_main', [3, 2, 3]],
    ['v_main', [3, 2, 3]],
  ]);
  assert.deepEqual(dispatches[0].entries.map(([binding]) => binding), [0, 1, 2, 3, 4, 5, 8]);
  assert.deepEqual(dispatches[1].entries.map(([binding]) => binding), [0, 1, 2, 3, 4, 6, 8]);
  assert.deepEqual(dispatches[2].entries.map(([binding]) => binding), [0, 1, 3, 4, 7, 8]);
  const params = dispatches[0].entries.find(([binding]) => binding === 8)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 6)], [2, 3, 4, 2, 2, 3]);
  assert.equal(new Float32Array(params.bytes.buffer)[6], 0.375);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(7, 9)], [1, 4]);
});

test('WebGPU forward attention binds masks and encodes causal/mask modes', async () => {
  const device = mockDevice();
  const qkv = tensor('qkv', [2, 3, 12]);
  const selfMask = tensor('self_mask', [2, 3], { dtype: 'int32' });
  const selfOut = tensor('self_out', [2, 3, 4]);
  const self = {
    id: 'self_forward', opType: 'SDPA', inputs: { qkv, mask: selfMask }, outputs: { out: selfOut },
    params: { heads: 2, scale: 0.25, causal: false },
  };
  const shaderLibrary = {
    getSDPAShader: () => 'sdpa',
    getCrossSDPAShader: () => 'cross-sdpa',
    getEmbeddingShader: () => 'embedding',
  };
  const executor = new GraphExecutor(device, { nodes: [] }, { shaderLibrary });
  const register = (values) => {
    for (const value of values) executor.gpuBuffers.set(value.name, { tensor: value.name });
  };
  register([qkv, selfMask, selfOut]);
  await executor._buildNodePipeline(self);
  const selfEntries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(selfEntries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const selfParams = selfEntries.find(({ binding }) => binding === 3).resource.buffer;
  assert.deepEqual([...new Uint32Array(selfParams.bytes.buffer).slice(0, 5)], [3, 4, 2, 2, 2]);
  assert.equal(new Float32Array(selfParams.bytes.buffer)[5], 0.25);
  assert.deepEqual([...new Uint32Array(selfParams.bytes.buffer).slice(6, 8)], [0, 2]);

  const q = tensor('q', [3, 2, 4]);
  const k = tensor('k', [3, 3, 4]);
  const v = tensor('v', [3, 3, 4]);
  const crossMask = tensor('cross_mask', [3, 2, 3], { dtype: 'int32' });
  const crossOut = tensor('cross_out', [3, 2, 4]);
  const cross = {
    id: 'cross_forward', opType: 'CrossSDPA', inputs: { q, k, v, mask: crossMask },
    outputs: { out: crossOut }, params: { heads: 2, scale: 0.375, causal: true },
  };
  register([q, k, v, crossMask, crossOut]);
  await executor._buildNodePipeline(cross);
  const crossEntries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(crossEntries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5]);
  const crossParams = crossEntries.find(({ binding }) => binding === 5).resource.buffer;
  assert.deepEqual([...new Uint32Array(crossParams.bytes.buffer).slice(0, 6)], [2, 3, 4, 2, 2, 3]);
  assert.equal(new Float32Array(crossParams.bytes.buffer)[6], 0.375);
  assert.deepEqual([...new Uint32Array(crossParams.bytes.buffer).slice(7, 9)], [1, 4]);

  const tokens = tensor('tokens', [2], { dtype: 'int32' });
  const table = tensor('table', [3, 2]);
  const embedded = tensor('embedded', [2, 2]);
  register([tokens, table, embedded]);
  await executor._buildNodePipeline({
    id: 'embedding_forward', opType: 'Embedding', inputs: { input: tokens, weight: table },
    outputs: { out: embedded }, params: {},
  });
  const embeddingEntries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(embeddingEntries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const embeddingParams = embeddingEntries.find(({ binding }) => binding === 3).resource.buffer;
  assert.deepEqual([...new Uint32Array(embeddingParams.bytes.buffer)], [2, 2, 3, 0]);
  executor.dispose();
});

test('RMSNorm backward uses the same default epsilon as WebGPU forward', async () => {
  const input = tensor('x', [2, 4]);
  const weight = tensor('weight', [4]);
  const out = tensor('out', [2, 4]);
  const node = { id: 'rms', opType: 'RMSNorm', inputs: { input, weight }, outputs: { out }, params: {} };
  const { trainer } = makeTrainer([node], [input, weight, out]);
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  const params = dispatches[0].entries.find(([binding]) => binding === 5)[1];
  assert.ok(Math.abs(new Float32Array(params.bytes.buffer)[2] - 1e-6) < 1e-12);
});

test('last-axis WebGPU reductions encode row width and sum/mean backward scale', async () => {
  for (const [opType, expectedScale] of [['ReduceSum', 1], ['ReduceMean', 0.25]]) {
    const input = tensor(`${opType}_input`, [2, 3, 4]);
    const output = tensor(`${opType}_output`, [2, 3]);
    const node = { id: opType, opType, inputs: { input }, outputs: { out: output }, params: {} };
    const { trainer } = makeTrainer([node], [input, output]);
    trainer.gradientBuffers.set(output.name, { tensor: `${opType}_grad_output` });
    const [dispatch] = await trainer._buildBackwardDispatches();
    assert.equal(dispatch.shaderName, 'reduceBackward');
    assert.deepEqual(dispatch.workgroupCount, [1, 1, 1]);
    const params = dispatch.entries.find(([binding]) => binding === 2)[1];
    assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 2)], [24, 4]);
    assert.equal(new Float32Array(params.bytes.buffer)[2], expectedScale);
  }
});

test('LayerNorm backward infers an omitted d_model like WebGPU forward', async () => {
  const input = tensor('x', [2, 4]);
  const weight = tensor('weight', [4]);
  const bias = tensor('bias', [4]);
  const out = tensor('out', [2, 4]);
  const node = {
    id: 'layer', opType: 'LayerNorm', inputs: { input, weight, bias }, outputs: { out }, params: {},
  };
  const { trainer } = makeTrainer([node], [input, weight, bias, out]);
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  const params = dispatches[0].entries.find(([binding]) => binding === 6)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 2)], [2, 4]);
});

test('pooling, resize, and axis-aware concat use dedicated backward dispatches', async () => {
  const poolInput = tensor('pool_in', [2, 2, 2, 1]);
  const poolOut = tensor('pool_out', [2, 1, 1, 1]);
  const pool = {
    id: 'pool', opType: 'MaxPool2D', inputs: { input: poolInput }, outputs: { out: poolOut },
    params: { kernel: [2, 2], stride: [2, 2], pads: [1, 0, 0, 0] },
  };
  const { trainer: poolTrainer } = makeTrainer([pool], [poolInput, poolOut]);
  poolTrainer.gradientBuffers.set(poolOut.name, { tensor: 'grad_pool' });
  const poolDispatch = (await poolTrainer._buildBackwardDispatches())[0];
  assert.equal(poolDispatch.entryPoint, 'max_pool_main');
  assert.deepEqual(poolDispatch.workgroupCount, [1, 1, 1]);
  const poolParams = poolDispatch.entries.find(([binding]) => binding === 3)[1];
  assert.deepEqual([...new Uint32Array(poolParams.bytes.buffer)], [
    2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 0,
  ]);

  const invalidPoolOut = tensor('invalid_pool_out', [2, 2, 1, 1]);
  const invalidPool = {
    ...pool,
    id: 'invalid_pool',
    outputs: { out: invalidPoolOut },
  };
  const { trainer: invalidPoolTrainer } = makeTrainer(
    [invalidPool], [poolInput, invalidPoolOut],
  );
  invalidPoolTrainer.gradientBuffers.set(invalidPoolOut.name, { tensor: 'grad_invalid_pool' });
  await assert.rejects(
    invalidPoolTrainer._buildBackwardDispatches(),
    /output shape is incompatible with its canonical pooling parameters/i,
  );
  assert.equal(invalidPoolTrainer.gradientBuffers.has(poolInput.name), false,
    'invalid output geometry must fail before allocating an input-gradient buffer');

  const resizeInput = tensor('resize_in', [1, 2, 2, 1]);
  const resizeOut = tensor('resize_out', [1, 3, 3, 1]);
  const resize = {
    id: 'resize', opType: 'Resize', inputs: { input: resizeInput }, outputs: { out: resizeOut },
    params: { mode: 'linear' },
  };
  const { trainer: resizeTrainer } = makeTrainer([resize], [resizeInput, resizeOut]);
  resizeTrainer.gradientBuffers.set(resizeOut.name, { tensor: 'grad_resize' });
  const resizeDispatch = (await resizeTrainer._buildBackwardDispatches())[0];
  assert.equal(resizeDispatch.shaderName, 'resizeBackward');
  const resizeParams = resizeDispatch.entries.find(([binding]) => binding === 2)[1];
  assert.equal(new Uint32Array(resizeParams.bytes.buffer)[6], 1);

  const left = tensor('left', [2]);
  const right = tensor('right', [3]);
  const concatOut = tensor('concat_out', [5]);
  const concat = {
    id: 'concat', opType: 'Concat', inputs: { a: left, b: right }, outputs: { out: concatOut }, params: {},
  };
  const { trainer: concatTrainer } = makeTrainer([concat], [left, right, concatOut]);
  concatTrainer.gradientBuffers.set(concatOut.name, { tensor: 'grad_concat' });
  const concatDispatches = await concatTrainer._buildBackwardDispatches();
  assert.deepEqual(concatDispatches.map(({ entryPoint }) => entryPoint), ['main', 'main']);
  const secondParams = concatDispatches[1].entries.find(([binding]) => binding === 2)[1];
  assert.deepEqual([...new Uint32Array(secondParams.bytes.buffer).slice(0, 5)], [3, 2, 3, 5, 1]);

  const channelLeft = tensor('channel_left', [2, 2]);
  const channelRight = tensor('channel_right', [2, 1]);
  const channelOut = tensor('channel_out', [2, 3]);
  const channelConcat = {
    id: 'channel_concat', opType: 'Concat', inputs: { a: channelLeft, b: channelRight },
    outputs: { out: channelOut }, params: { axis: 1 },
  };
  const { trainer: channelTrainer } = makeTrainer(
    [channelConcat], [channelLeft, channelRight, channelOut],
  );
  channelTrainer.gradientBuffers.set(channelOut.name, { tensor: 'grad_channel_concat' });
  const channelDispatches = await channelTrainer._buildBackwardDispatches();
  const channelRightParams = channelDispatches[1].entries.find(([binding]) => binding === 2)[1];
  assert.deepEqual(
    [...new Uint32Array(channelRightParams.bytes.buffer).slice(0, 5)],
    [2, 2, 1, 3, 1],
  );
});

test('Split backward follows every live output, not only the first output', async () => {
  const input = tensor('input', [1, 4]);
  const left = tensor('left', [1, 2]);
  const right = tensor('right', [1, 2]);
  const node = {
    id: 'split', opType: 'Split', inputs: { input }, outputs: { left, right }, params: { axis: 1 },
  };
  const { trainer } = makeTrainer([node], [input, left, right]);
  trainer.gradientBuffers.set(right.name, { tensor: 'grad_right' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.equal(dispatches.length, 1);
  assert.equal(dispatches[0].shaderName, 'splitBackward');
  const params = dispatches[0].entries.find(([binding]) => binding === 2)[1];
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 5)], [2, 1, 2, 4, 2]);
});

test('a reachable unsupported secondary output fails instead of dropping its gradient', async () => {
  const input = tensor('input', [1, 2]);
  const primary = tensor('primary', [1, 2]);
  const secondary = tensor('secondary', [1, 2]);
  const logits = tensor('logits', [1, 2]);
  const producer = {
    id: 'producer', opType: 'CustomMultiOutput', inputs: { input },
    outputs: { primary, secondary }, params: {},
  };
  const consumer = {
    id: 'consumer', opType: 'Identity', inputs: { input: secondary }, outputs: { out: logits }, params: {},
  };
  const { trainer } = makeTrainer([producer, consumer], [input, primary, secondary, logits]);
  trainer.gradientBuffers.set(logits.name, { tensor: 'grad_logits' });

  await assert.rejects(
    () => trainer._buildBackwardDispatches(),
    /non-primary output.*CustomMultiOutput/,
  );
});

test('trainStep is serialized and destroys transient buffers only after queued work', async () => {
  const device = mockDevice();
  let release;
  const blocked = new Promise((resolve) => { release = resolve; });
  class DeferredTrainer extends WebGPUAutograd {
    async _trainStepImpl() {
      this._parameterBuffer(new Uint32Array([1, 2, 3, 4]));
      await blocked;
      return 'done';
    }
  }
  const trainer = new DeferredTrainer(device, { nodes: [] }, { gpuBuffers: new Map() });
  const first = trainer.trainStep();
  await assert.rejects(() => trainer.trainStep(), /concurrent/);
  assert.equal(device.state.buffers[0].destroyed, false);
  release();
  assert.equal(await first, 'done');
  assert.equal(device.state.workDone, 1);
  assert.equal(device.state.buffers[0].destroyed, true);
});

test('trainStep rejects non-optimizer tensor update modes before GPU work', async () => {
  const device = mockDevice();
  let compiled = false;
  const executor = {
    gpuBuffers: new Map(),
    compiledTopologyRevision: null,
    async compile() { compiled = true; },
  };
  const trainer = new WebGPUAutograd(device, { nodes: [] }, executor);
  await assert.rejects(
    () => trainer.trainStep({ targets: [0], trainableTensors: ['weight'], updateMode: 2 }),
    /SGD or AdamW/,
  );
  assert.equal(compiled, false);
  assert.equal(device.state.buffers.length, 0);
  assert.equal(device.state.shaderModules, 0);
});

test('WebGPU SDPA and CrossSDPA training regenerate the CPU attention-dropout mask', async () => {
  const qkv = tensor('qkv', [2, 3, 12]);
  const selfMask = tensor('self_mask', [2, 3], { dtype: 'int32' });
  const selfOut = tensor('self_out', [2, 3, 4]);
  const self = {
    id: 'self_dropout', opType: 'SDPA', inputs: { qkv, mask: selfMask }, outputs: { out: selfOut },
    params: { heads: 2, scale: 0.25, causal: false, dropout: 0.25, dropout_seed: 9 },
  };
  const q = tensor('q', [2, 2, 4]);
  const k = tensor('k', [2, 3, 4]);
  const v = tensor('v', [2, 3, 4]);
  const crossMask = tensor('cross_mask', [2, 2, 3], { dtype: 'int32' });
  const crossOut = tensor('cross_out', [2, 2, 4]);
  const cross = {
    id: 'cross_dropout', opType: 'CrossSDPA',
    inputs: { q, k, v, mask: crossMask }, outputs: { out: crossOut },
    params: { heads: 2, scale: 0.375, dropout: 0.5, dropout_seed: 13 },
  };
  const tensors = [qkv, selfMask, selfOut, q, k, v, crossMask, crossOut];
  const { trainer } = makeTrainer([self, cross], tensors);
  trainer.executor.pipelines = [{ graphNodeIndex: 0 }, { graphNodeIndex: 1 }];
  const context = dropoutContext({ seed: 17, counter: 23 });

  const overrides = await trainer._buildDropoutForwardOverrides(context);
  assert.equal(overrides.size, 2);
  const selfForward = overrides.get(0);
  const crossForward = overrides.get(1);
  assert.equal(selfForward.shaderName, 'sdpaTraining');
  assert.equal(crossForward.shaderName, 'crossSdpaTraining');
  assert.deepEqual(selfForward.workgroupCount, [1, 2, 2]);
  assert.deepEqual(crossForward.workgroupCount, [1, 2, 2]);
  assert.deepEqual(selfForward.entries.map(([binding]) => binding), [0, 1, 2, 3]);
  assert.deepEqual(crossForward.entries.map(([binding]) => binding), [0, 1, 2, 3, 4, 5]);

  const selfParams = selfForward.entries.find(([binding]) => binding === 3)[1];
  const selfU32 = new Uint32Array(selfParams.bytes.buffer);
  const selfF32 = new Float32Array(selfParams.bytes.buffer);
  assert.deepEqual([...selfU32.slice(0, 5)], [3, 4, 2, 2, 2]);
  assert.deepEqual([...selfU32.slice(6, 9)], [0, 2, 0x40000000]);
  assert.equal(selfU32[9], (17 ^ 9 ^ Math.imul(1, 0x9e3779b9)) >>> 0);
  assert.equal(selfU32[10], 23);
  assert.ok(Math.abs(selfF32[11] - 4 / 3) < 1e-6);

  const crossParams = crossForward.entries.find(([binding]) => binding === 5)[1];
  const crossU32 = new Uint32Array(crossParams.bytes.buffer);
  const crossF32 = new Float32Array(crossParams.bytes.buffer);
  assert.deepEqual([...crossU32.slice(0, 6)], [2, 3, 4, 2, 2, 2]);
  assert.deepEqual([...crossU32.slice(7, 10)], [0, 4, 0x80000000]);
  assert.equal(crossU32[10], (17 ^ 13 ^ Math.imul(2, 0x9e3779b9)) >>> 0);
  assert.equal(crossU32[11], 23);
  assert.equal(crossF32[12], 2);

  // The shader flattens [batch, head, query, key] exactly like the CPU path.
  const probabilityIndex = attentionProbabilityIndex(1, 1, 1, 2, 2, 2, 3);
  assert.equal(probabilityIndex, 23);
  const expectedKeep = dropoutHash(probabilityIndex, crossU32[10], crossU32[11], -1) >= crossU32[9];
  assert.equal(typeof expectedKeep, 'boolean');

  trainer.gradientBuffers.set(selfOut.name, { tensor: 'grad_self' });
  trainer.gradientBuffers.set(crossOut.name, { tensor: 'grad_cross' });
  const backward = await trainer._buildBackwardDispatches(context);
  assert.deepEqual(backward.map(({ shaderName, entryPoint }) => `${shaderName}.${entryPoint}`), [
    'crossSdpaBackward.q_main',
    'crossSdpaBackward.k_main',
    'crossSdpaBackward.v_main',
    'sdpaBackward.main',
  ]);
  const crossBackwardParams = backward[0].entries.find(([binding]) => binding === 8)[1];
  const selfBackwardParams = backward[3].entries.find(([binding]) => binding === 4)[1];
  assert.deepEqual(
    [...new Uint32Array(crossBackwardParams.bytes.buffer).slice(9, 13)],
    [...crossU32.slice(9, 13)],
  );
  assert.deepEqual(
    [...new Uint32Array(selfBackwardParams.bytes.buffer).slice(8, 12)],
    [...selfU32.slice(8, 12)],
  );
});

test('WebGPU input uploads require the declared typed-array dtype', async () => {
  const device = mockDevice();
  const ids = tensor('ids', [1], { dtype: 'int32', isInput: true });
  const graph = {
    nodes: [], weightRevision: 0,
    getTensor(name) { return name === 'ids' ? ids : null; },
    assertTopologyRevision() {},
  };
  const executor = new GraphExecutor(device, graph);
  executor.compiledWeightRevision = 0;
  executor.gpuBuffers.set('ids', device.createBuffer({ size: 4 }));
  await assert.rejects(() => executor.execute({ ids: Float32Array.of(1) }), /typed storage/);
  await assert.rejects(() => executor.execute({ missing: Int32Array.of(1) }), /Unknown graph input/);
  executor.dispose();
});

test('WebGPU training selects one sequence row per batch target', async () => {
  const device = mockDevice();
  const logits = tensor('logits', [2, 2, 2], { buffer: new Float32Array(8), isWeight: true });
  const logitsBuffer = { tensor: 'logits' };
  const appliedSteps = [];
  const graph = {
    nodes: [], outputNames: ['logits'], topologyRevision: 0, weightRevision: 0, trainingStep: 0,
    getTensor(name) { return name === 'logits' ? logits : null; },
    assertTopologyRevision() {},
    applyTensorUpdate(name, gradient, update) { appliedSteps.push(update.step); return logits; },
  };
  const executor = {
    gpuBuffers: new Map([['logits', logitsBuffer]]),
    compiledTopologyRevision: 0, compiledWeightRevision: 0,
    async execute() {},
  };
  class GradientReader extends DispatchRecorder {
    async _readFloatBuffer(buffer, sizeBytes) {
      if (buffer === logitsBuffer) return new Float32Array(logits.buffer);
      return new Float32Array(buffer.bytes.buffer.slice(0, sizeBytes));
    }

    _uploadUpdatedTensor() {}
  }
  const trainer = new GradientReader(device, graph, executor);
  const options = {
    targets: [1, 0],
    trainableTensors: ['logits'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  };

  const finalPosition = await trainer.trainStep(options);
  assert.deepEqual([...finalPosition.gradients.get('logits')], [
    0, 0, 0.25, -0.25,
    0, 0, -0.25, 0.25,
  ]);
  assert.equal(graph.trainingStep, 1);
  assert.deepEqual(graph.optimizerDescriptor, {
    updateMode: 'sgd',
    optimizer: { learningRate: 0, weightDecay: 0, maxGradNorm: 0 },
  });

  const firstPosition = await trainer.trainStep({ ...options, lastToken: 0 });
  assert.deepEqual([...firstPosition.gradients.get('logits')], [
    0.25, -0.25, 0, 0,
    -0.25, 0.25, 0, 0,
  ]);
  assert.equal(graph.trainingStep, 2);

  await trainer.trainStep({ ...options, optimizer: { learningRate: 0, step: 7 } });
  assert.equal(graph.trainingStep, 7);
  assert.deepEqual(appliedSteps, [1, 2, 7]);
  await assert.rejects(
    () => trainer.trainStep({ ...options, optimizer: { learningRate: 0, step: 7 } }),
    /greater than graph\.trainingStep/,
  );
  await assert.rejects(
    () => trainer.trainStep({
      ...options,
      optimizer: { learningRate: 0, step: Number.MAX_SAFE_INTEGER + 1 },
    }),
    /Optimizer step is invalid/,
  );

  const ignored = await trainer.trainStep({
    ...options,
    targets: [-1, -1],
    ignoreIndex: -1,
    optimizer: { learningRate: 0, weightDecay: 0.25, step: 8 },
  });
  assert.equal(ignored.examples, 0);
  assert.equal(graph.trainingStep, 7, 'an all-ignored loss is not an optimizer step');
  assert.equal(graph.optimizerDescriptor.optimizer.weightDecay, 0,
    'an all-ignored loss does not replace optimizer resume metadata');
  assert.deepEqual(appliedSteps, [1, 2, 7]);

  await assert.rejects(
    () => trainer.trainStep({ ...options, lastToken: 2 }),
    /2 logits rows per batch/,
  );
  trainer.dispose();
});

test('WebGPU trainStep rejects non-finite gradients before mutating weights', async () => {
  const device = mockDevice();
  const parameter = tensor('parameter', [1, 2], { buffer: Float32Array.of(0.2, -0.4), isWeight: true });
  const logits = tensor('logits', [1, 2], { buffer: new Float32Array(2) });
  const node = {
    id: 'identity', opType: 'Identity', inputs: { input: parameter }, outputs: { out: logits }, params: {},
  };
  let mutated = false;
  const graph = {
    nodes: [node], outputNames: ['logits'], topologyRevision: 0, weightRevision: 0,
    getTensor(name) { return name === 'parameter' ? parameter : name === 'logits' ? logits : null; },
    assertTopologyRevision() {},
    applyTensorUpdate() { mutated = true; throw new Error('unexpected mutation'); },
  };
  const executor = {
    gpuBuffers: new Map([['parameter', { tensor: 'parameter' }], ['logits', { tensor: 'logits' }]]),
    compiledTopologyRevision: 0, compiledWeightRevision: 0,
    async execute() {},
  };
  class NonFiniteTrainer extends DispatchRecorder {
    async _readFloatBuffer() {
      this.readCount = (this.readCount || 0) + 1;
      return this.readCount === 1 ? Float32Array.of(0.2, -0.4) : Float32Array.of(Number.NaN, 0);
    }
  }
  const trainer = new NonFiniteTrainer(device, graph, executor);

  await assert.rejects(() => trainer.trainStep({
    targets: [1], trainableTensors: ['parameter'], updateMode: 'sgd', optimizer: { learningRate: 0.1 },
  }), /Gradient.*non-finite/);
  assert.equal(mutated, false);
  assert.deepEqual([...parameter.buffer], [...Float32Array.of(0.2, -0.4)]);
});
