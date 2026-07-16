import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine, Graph, GraphLoader, ModelBuilder, VolvoxAI } from '../ts/index.js';
import { TrainingModelBuilder } from '../ts/training/TrainingModelBuilder.js';

test('inference builder exposes generic GroupNorm without training authoring helpers', () => {
  const builder = new ModelBuilder();
  const input = builder.input('image', [2, 4, 5, 8]);
  const scale = builder.weight('gn.scale', [8], 'float32', new Float32Array(8).fill(1));
  const bias = builder.weight('gn.bias', [8], 'float32', new Float32Array(8));
  const normalized = builder.groupNorm(input, scale, bias, {
    numGroups: 4,
    epsilon: 1e-5,
    name: 'stem.norm',
  });
  builder.outputs(normalized);

  assert.equal(builder.getNode('stem.norm').opType, 'GroupNorm');
  assert.deepEqual(builder.getNode('stem.norm').params, { num_groups: 4, eps: 1e-5 });
  assert.equal(builder.build().validate().valid, true);
  assert.throws(() => builder.groupNorm(input, scale, bias, { numGroups: 3 }), /divisor/);
  assert.equal(builder.dropout, undefined);
  assert.equal(builder.loraLinear, undefined);
  assert.equal(builder.routedBottleneckAdapter, undefined);
});

test('training builder composes Dropout and routed bottleneck adapters', () => {
  const builder = new TrainingModelBuilder();
  const input = builder.input('tokens', [2, 3, 4]);
  const routerWeight = builder.weight('router.weight', [4, 2], 'float32', new Float32Array(8));
  const routes = builder.moeRouter(input, routerWeight, { topK: 1, name: 'router' });
  const down = builder.weight('adapter.down', [2, 4, 2], 'float32', new Float32Array(16));
  const up = builder.weight('adapter.up', [2, 2, 4], 'float32', new Float32Array(16));
  const output = builder.routedBottleneckAdapter(input, down, up, routes, {
    dropout: 0.1,
    seed: 9,
    name: 'adapter',
  });
  builder.outputs(output);

  assert.deepEqual(output.shape, input.shape);
  assert.equal(builder.getNode('adapter.down').opType, 'MoELinear');
  assert.equal(builder.getNode('adapter.gelu').opType, 'GELU');
  assert.equal(builder.getNode('adapter.up').opType, 'MoELinear');
  assert.equal(builder.getNode('adapter.dropout').opType, 'Dropout');
  assert.equal(builder.getNode('adapter').opType, 'Add');
  assert.equal(builder.build().validate().valid, true);
  assert.throws(() => builder.dropout(input, { probability: 1 }), /\[0, 1\)/);
});

test('training builder composes explicit LoRA factors with a frozen scalar scale', async () => {
  const builder = new TrainingModelBuilder();
  const input = builder.input('x', [1, 3]);
  const weight = builder.weight('projection.weight', [3, 2], 'float32', Float32Array.from([
    1, 0,
    0, 1,
    1, 1,
  ]));
  const bias = builder.weight('projection.bias', [2], 'float32', Float32Array.from([1, -1]));
  const lora = builder.loraLinear(input, weight, {
    rank: 2,
    alpha: 4,
    bias,
    aInitializer: 'ones',
    bInitializer: 'ones',
    name: 'projection',
  });
  builder.outputs(lora.out);

  assert.deepEqual(lora.A.shape, [3, 2]);
  assert.deepEqual(lora.B.shape, [2, 2]);
  assert.deepEqual(lora.trainableTensors, ['projection.lora_a', 'projection.lora_b']);
  assert.equal(lora.scale, 2);
  assert.deepEqual(builder.graph.nodes.map((node) => node.opType), [
    'Linear', 'MatMul', 'MatMul', 'Mul', 'Add',
  ]);
  assert.deepEqual(builder.graph.nodes.slice(0, 3).map((node) => node.wLayout), [
    'din', 'din', 'din',
  ]);
  assert.equal(builder.getNode('projection.lora_scale').inputs.b.name, 'projection.lora_scale');
  assert.deepEqual(builder.graph.listAdapters(), []);
  assert.equal(builder.build().validate().valid, true);

  const engine = new CPUEngine();
  engine.allocateGraph(builder.graph);
  const result = await engine.execute({ x: Float32Array.from([1, 2, 3]) });
  assert.deepEqual([...result['projection.out']], [29, 28]);
});

test('training builder validates LoRA layout, dimensions, and scaling before mutation', () => {
  const create = (shape = [3, 2]) => {
    const builder = new TrainingModelBuilder();
    const input = builder.input('x', [1, 3]);
    const weight = builder.weight('weight', shape, 'float32', new Float32Array(shape[0] * shape[1]));
    return { builder, input, weight };
  };

  {
    const { builder, input, weight } = create();
    const revision = builder.graph.topologyRevision;
    assert.throws(() => builder.loraLinear(input, weight, { rank: 0 }), /rank must be a positive/);
    assert.equal(builder.graph.topologyRevision, revision);
  }
  {
    const { builder, input, weight } = create();
    assert.throws(() => builder.loraLinear(input, weight, { rank: 1, layout: 'peft' }),
      /layout must be 'din' or 'dout'/);
    assert.throws(() => builder.loraLinear(input, weight, { rank: 1, alpha: 0 }),
      /alpha must be positive and finite/);
    assert.throws(() => builder.loraLinear(input, weight, { rank: 1, scale: Number.NaN }),
      /scale must be positive and finite/);
    assert.throws(() => builder.loraLinear(input, weight, { rank: 1, alpha: 1, scale: 1 }),
      /alpha or scale, not both/);
  }
  {
    const { builder, input, weight } = create([2, 3]);
    assert.throws(() => builder.loraLinear(input, weight, { rank: 1 }),
      /din base weight must be shaped \[3,d_out\]/);
    const lora = builder.loraLinear(input, weight, {
      rank: 1,
      scale: 0.5,
      layout: 'dout',
      opType: 'MatMul',
      name: 'dout_projection',
    });
    assert.deepEqual(lora.A.shape, [3, 1]);
    assert.deepEqual(lora.B.shape, [1, 2]);
    assert.equal(builder.getNode('dout_projection.base').wLayout, 'dout');
  }
});

test('ModelBuilder creates an executable model from an empty graph', async () => {
  const builder = new ModelBuilder();
  assert.ok(builder.build() instanceof Graph);

  const input = builder.input('x', [1, 2]);
  const weight = builder.weight('w', [2, 2], 'float32', Float32Array.from([1, 0, 0, 1]));
  const linear = builder.addNode({
    id: 'linear',
    opType: 'MatMul',
    inputs: { input, weight },
    outputs: { out: { name: 'hidden', shape: [1, 2] } },
    wLayout: 'din',
  });
  builder.addNode({
    id: 'activation',
    opType: 'ReLU',
    inputs: { input: linear.outputs.out },
    outputs: { out: { name: 'result', shape: [1, 2] } },
  });

  const graph = builder.build();
  assert.deepEqual(graph.outputNames, ['result']);
  assert.equal(builder.validate().valid, true);
  assert.equal(graph.inspect().topologyRevision, graph.topologyRevision);

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  assert.deepEqual(
    [...(await engine.execute({ x: Float32Array.from([-2, 3]) })).result],
    [0, 3],
  );
});

test('dynamic insertion and patching are ordered, transactional, and update outputs', () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1]);
  const tail = builder.addNode({
    id: 'tail', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'tail_out', shape: [1] } },
  });
  const inserted = builder.insertNode('tail', {
    id: 'head', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'head_out', shape: [1] } },
  });
  assert.deepEqual(builder.graph.nodes.map((node) => node.id), ['head', 'tail']);
  assert.deepEqual(builder.graph.outputNames, ['head_out', 'tail_out']);

  const originalTailHandle = builder.getNode('tail');
  assert.equal(builder.patchNode('tail', { inputs: { input: inserted.outputs.out } }), originalTailHandle);
  assert.deepEqual(builder.graph.outputNames, ['tail_out']);
  const revision = builder.graph.topologyRevision;
  const tensorNames = [...builder.graph.tensors.keys()];

  assert.throws(() => builder.insertNodeAt(0, {
    id: 'invalid',
    opType: 'Identity',
    inputs: { input: tail.outputs.out },
    outputs: { out: { name: 'invalid_out', shape: [1] } },
  }), /before it is produced/);
  assert.equal(builder.graph.topologyRevision, revision);
  assert.deepEqual([...builder.graph.tensors.keys()], tensorNames);
  assert.equal(builder.graph.getNodeById('invalid'), undefined);
});

test('builder topology transactions restore multi-edit failures exactly', async () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1]);
  const originalNode = builder.addNode({
    id: 'identity', opType: 'Identity', inputs: { input },
    outputs: { out: { name: 'y', shape: [1] } },
  });
  builder.outputs(originalNode.outputs.out);
  const originalOutput = originalNode.outputs.out;
  const revision = builder.graph.topologyRevision;

  assert.throws(() => builder.topologyTransaction((edit) => {
    edit.patchNode('identity', { opType: 'ReLU' });
    edit.renameTensor('y', 'renamed');
    edit.input('extra', [1]);
    throw new Error('abort structural edit');
  }), /abort structural edit/);
  assert.equal(builder.graph.topologyRevision, revision);
  assert.strictEqual(builder.getNode('identity'), originalNode);
  assert.strictEqual(builder.getTensor('y'), originalOutput);
  assert.equal(originalNode.opType, 'Identity');
  assert.equal(originalOutput.name, 'y');
  assert.equal(builder.getTensor('renamed'), undefined);
  assert.equal(builder.getTensor('extra'), undefined);
  assert.deepEqual(builder.graph.outputNames, ['y']);
  assert.equal(builder.validate().valid, true);

  await assert.rejects(builder.topologyTransaction(async (edit) => {
    edit.input('async_extra', [1]);
    throw new Error('abort async structural edit');
  }), /abort async structural edit/);
  assert.equal(builder.graph.topologyRevision, revision);
  assert.equal(builder.getTensor('async_extra'), undefined);

  const committed = builder.topologyTransaction((edit) => edit.input('committed', [1]));
  assert.strictEqual(builder.getTensor('committed'), committed);
  assert.ok(builder.graph.topologyRevision > revision);
  assert.throws(() => builder.topologyTransaction(null), /expects a callback/);
});

test('replace and remove support automatic downstream rewrites and safe deletion', () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1]);
  const first = builder.addNode({
    id: 'first', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'middle', shape: [1] } },
  });
  builder.addNode({
    id: 'second', opType: 'Identity', inputs: { input: first.outputs.out }, outputs: { out: { name: 'result', shape: [1] } },
  });

  const replacement = builder.replaceNode('first', { opType: 'ReLU', outputs: { out: [2] } });
  assert.equal(replacement.outputs.out.name, 'middle');
  assert.deepEqual(replacement.outputs.out.shape, [2]);
  assert.equal(builder.getNode('second').inputs.input, replacement.outputs.out);
  assert.equal(builder.validate().valid, true);

  assert.throws(() => builder.removeNode('first'), /is consumed/);
  const removed = builder.removeNode('first', { rewire: { middle: 'x' } });
  assert.equal(removed.id, 'first');
  assert.equal(builder.getNode('second').inputs.input.name, 'x');
  assert.equal(builder.getTensor('middle'), undefined);
  assert.deepEqual(builder.graph.outputNames, ['result']);

  builder.removeNode('second', { cascade: true });
  assert.equal(builder.graph.nodes.length, 0);
  assert.deepEqual(builder.graph.outputNames, []);
  assert.equal(builder.validate().valid, true);
});

test('tensor CRUD maintains references, selected outputs, and topology guards', () => {
  const api = new VolvoxAI();
  const builder = api.createModel();
  assert.ok(builder instanceof ModelBuilder);
  const input = builder.addInput('x', [1]);
  builder.addNode({
    id: 'identity', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'y', shape: [1] } },
  });
  builder.outputs('y');
  const selectedRevision = builder.graph.topologyRevision;
  builder.renameTensor('y', 'prediction');
  assert.deepEqual(builder.graph.outputNames, ['prediction']);
  assert.equal(builder.getNode('identity').outputs.out.name, 'prediction');
  assert.throws(
    () => builder.graph.assertTopologyRevision(selectedRevision, 'Test backend'),
    /Test backend graph topology changed.*recompile/,
  );

  assert.throws(() => builder.removeTensor('x'), /referenced/);
  assert.equal(builder.removeTensor('x', { cascade: true }), true);
  assert.equal(builder.getTensor('x'), undefined);
  assert.equal(builder.graph.nodes.length, 0);
  assert.deepEqual(builder.graph.outputNames, []);
});

test('loaded adapter versions lock structural builder mutations', () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const weight = builder.weight('w', [2, 2], 'float32', Float32Array.from([1, 0, 0, 1]));
  const { out } = builder.addOp('MatMul', { input, weight }, { out: [1, 2] }, {}, { id: 'linear', wLayout: 'din' });
  builder.graph.stageAdapter('locked', {
    kind: 'lora',
    targets: [{ weight: 'w', rank: 1, alpha: 1, A: Float32Array.from([1, 0]), B: Float32Array.from([1, 1]) }],
  });
  assert.throws(() => builder.patchNode('linear', { outputs: { out } }), /Remove all adapter versions/);
  assert.throws(() => builder.input('another', [1]), /Remove all adapter versions/);
});

test('builder exports a blueprint config with stable tensor names', () => {
  const builder = new ModelBuilder();
  const input = builder.input('tokens', [1, 4], 'int32');
  builder.addNode({
    id: 'copy', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'copied', shape: [1, 4], dtype: 'int32' } },
  });
  assert.deepEqual(builder.toConfig(), {
    inputs: { tokens: { shape: [1, 4], dtype: 'int32' } },
    nodes: [{
      id: 'copy',
      opType: 'Identity',
      inputs: { input: 'tokens' },
      outputs: { out: 'copied' },
      outputs_shape: { out: [1, 4] },
      outputs_dtype: { out: 'int32' },
      params: {},
    }],
    outputs: ['copied'],
  });
});

test('blueprint loading preserves an explicit output selection', () => {
  const graph = new Graph();
  const input = graph.addInput('x', [1], 'float32');
  const config = {
    inputs: { x: { shape: [1], dtype: 'float32' } },
    nodes: [
      {
        opType: 'Identity',
        inputs: { input: 'x' },
        outputs: { out: 'visible' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
      {
        opType: 'Identity',
        inputs: { input: 'visible' },
        outputs: { out: 'internal' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
    ],
    outputs: ['visible'],
  };
  GraphLoader._buildFromBlueprint(graph, config, new Map([['x', input]]));
  assert.deepEqual(graph.outputNames, ['visible']);

  const legacyGraph = new Graph();
  const legacyInput = legacyGraph.addInput('x', [1], 'float32');
  GraphLoader._buildFromBlueprint(legacyGraph, {
    ...config,
    outputs: { source_output_name: 'visible' },
  }, new Map([['x', legacyInput]]));
  assert.deepEqual(legacyGraph.outputNames, ['visible']);

  const invalidOutputGraph = new Graph();
  const invalidOutputInput = invalidOutputGraph.addInput('x', [1], 'float32');
  const invalidRevision = invalidOutputGraph.topologyRevision;
  assert.throws(() => GraphLoader._buildFromBlueprint(invalidOutputGraph, {
    ...config,
    outputs: ['missing'],
  }, new Map([['x', invalidOutputInput]])), /config\.outputs/);
  assert.equal(invalidOutputGraph.nodes.length, 0);
  assert.equal(invalidOutputGraph.topologyRevision, invalidRevision);

  for (const outputs of [['visible', 'visible'], { first: 'visible', second: 'visible' }]) {
    const duplicateGraph = new Graph();
    const duplicateInput = duplicateGraph.addInput('x', [1], 'float32');
    assert.throws(() => GraphLoader._buildFromBlueprint(duplicateGraph, {
      ...config,
      outputs,
    }, new Map([['x', duplicateInput]])), /unique graph tensor/);
    assert.equal(duplicateGraph.nodes.length, 0);
  }
});

test('blueprints serialize canonical linear layout for square weights', async () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const weight = builder.weight('w', [2, 2], 'float32', Float32Array.from([1, 2, 3, 4]));
  builder.addNode({
    id: 'square', opType: 'MatMul', inputs: { input, weight },
    outputs: { out: { name: 'y', shape: [1, 2] } }, wLayout: 'din',
  });
  const config = builder.toConfig();
  assert.equal(config.nodes[0].params.weight_layout, 'IN_OUT');
  assert.equal(Object.hasOwn(config.nodes[0], 'wLayout'), false);

  const loaded = new Graph();
  const loadedInput = loaded.addInput('x', [1, 2]);
  const loadedWeight = loaded.addWeight('w', [2, 2], 'float32', {
    buffer: Float32Array.from([1, 2, 3, 4]),
  });
  GraphLoader._buildFromBlueprint(loaded, config, new Map([
    ['x', loadedInput], ['w', loadedWeight],
  ]));
  GraphLoader._resolveMatMulLayouts(loaded);
  loaded.outputNames = ['y'];
  assert.equal(loaded.nodes[0].wLayout, 'din');
  const engine = new CPUEngine();
  engine.allocateGraph(loaded);
  assert.deepEqual([...(await engine.execute({ x: Float32Array.from([1, 0]) })).y], [1, 2]);
});

test('blueprints reject undeclared inputs instead of inventing an image tensor', () => {
  assert.throws(
    () => GraphLoader._buildFromBlueprint(new Graph(), {
      nodes: [{
        id: 'missing_input',
        op: 'Identity',
        inputs: { input: 'undeclared' },
        outputs: { out: 'result' },
        outputs_shape: { out: [1] },
      }],
    }, new Map()),
    /input 'input' references undeclared tensor 'undeclared'/,
  );
});

test('blueprints reject output collisions before mutating the graph', () => {
  const graph = new Graph();
  const input = graph.addInput('x', [1]);
  const tensors = new Map([['x', input]]);
  const revision = graph.topologyRevision;

  assert.throws(
    () => GraphLoader._buildFromBlueprint(graph, {
      nodes: [{
        id: 'overwrite-input',
        op: 'Identity',
        inputs: { input: 'x' },
        outputs: { out: 'x' },
        outputs_shape: { out: [1] },
      }],
    }, tensors),
    /output 'out' collides with existing tensor 'x'/,
  );
  assert.equal(graph.nodes.length, 0);
  assert.equal(graph.tensors.get('x'), input);
  assert.equal(graph.topologyRevision, revision);
  graph.assertValid();

  assert.throws(
    () => GraphLoader._buildFromBlueprint(graph, {
      nodes: [
        {
          id: 'first', op: 'Identity', inputs: { input: 'x' },
          outputs: { out: 'shared' }, outputs_shape: { out: [1] },
        },
        {
          id: 'second', op: 'Identity', inputs: { input: 'shared' },
          outputs: { out: 'shared' }, outputs_shape: { out: [1] },
        },
      ],
    }, tensors),
    /output 'out' collides with existing tensor 'shared'/,
  );
  assert.equal(graph.nodes.length, 0);
  assert.equal(graph.topologyRevision, revision);
});

test('blueprint loading validates and commits one staged topology transaction', () => {
  const graph = new Graph();
  const input = graph.addInput('input', [1]);
  const tensors = new Map([['input', input]]);
  const revision = graph.topologyRevision;
  GraphLoader._buildFromBlueprint(graph, {
    nodes: [
      {
        op: 'Identity', inputs: { input: 'input' }, outputs: { out: 'first' },
        outputs_shape: { out: [1] },
      },
      {
        op: 'Identity', inputs: { input: 'first' }, outputs: { out: 'second' },
        outputs_shape: { out: [1] },
      },
      {
        op: 'Identity', inputs: { input: 'second' }, outputs: { out: 'third' },
        outputs_shape: { out: [1] },
      },
    ],
  }, tensors);
  assert.equal(graph.topologyRevision, revision + 1);
  assert.equal(graph.nodes.length, 3);
  assert.deepEqual(graph.outputNames, ['third']);
  graph.assertValid();

  const failed = new Graph();
  const failedInput = failed.addInput('input', [1]);
  const failedAliases = new Map([['input', failedInput]]);
  const failedRevision = failed.topologyRevision;
  assert.throws(() => GraphLoader._buildFromBlueprint(failed, {
    nodes: [
      {
        op: 'Identity', inputs: { input: 'input' }, outputs: { out: 'staged_only' },
        outputs_shape: { out: [1] },
      },
      {
        op: 'Identity', inputs: { input: 'staged_only' }, outputs: { out: 'invalid' },
        outputs_shape: { out: [0] },
      },
    ],
  }, failedAliases), /positive integer/);
  assert.equal(failed.topologyRevision, failedRevision);
  assert.equal(failed.nodes.length, 0);
  assert.deepEqual([...failed.tensors.keys()], ['input']);
  assert.deepEqual([...failedAliases.keys()], ['input']);
  failed.assertValid();
});

test('blueprints serialize explicit Conv2D image-weight layouts', () => {
  const regular = new ModelBuilder();
  const image = regular.input('image', [1, 2, 2, 2]);
  const regularWeight = regular.weight('regular_weight', [1, 1, 2, 3], 'float32', new Float32Array(6));
  regular.addOp('Conv2D', { input: image, weight: regularWeight }, { out: [1, 2, 2, 3] });
  assert.equal(regular.toConfig().nodes[0].params.weight_layout, 'HWIO');

  const depthwise = new ModelBuilder();
  const depthImage = depthwise.input('image', [1, 2, 2, 2]);
  const depthWeight = depthwise.weight('depth_weight', [1, 1, 2, 1], 'float32', new Float32Array(2));
  depthwise.addOp('Conv2D', { input: depthImage, weight: depthWeight }, { out: [1, 2, 2, 2] }, { groups: 2 });
  assert.equal(depthwise.toConfig().nodes[0].params.weight_layout, 'HWCM');
});

test('browser Conv2D layout normalization transforms shared depthwise weights once', () => {
  const graph = new Graph();
  const input = graph.addInput('image', [1, 1, 1, 2]);
  const weight = graph.addWeight('shared_depthwise', [1, 1, 1, 2], 'float32', {
    buffer: Float32Array.from([2, 3]),
  });
  const first = graph.addOp('Conv2D', { input, weight }, {
    out: { name: 'first', shape: [1, 1, 1, 2] },
  }, { groups: 2, weight_layout: '1HWO' }).out;
  graph.addOp('Conv2D', { input: first, weight }, {
    out: { name: 'second', shape: [1, 1, 1, 2] },
  }, { groups: 2, weight_layout: '1HWO' });

  GraphLoader._normalizeConvWeightsForImageLayout(graph);
  assert.deepEqual(weight.shape, [1, 1, 2, 1]);
  assert.deepEqual([...weight.buffer], [2, 3]);
  assert.deepEqual(graph.nodes.map((node) => node.params.weight_layout), ['HWCM', 'HWCM']);
});

test('tensor payload dtype and byte length are validated transactionally', () => {
  const builder = new ModelBuilder();
  assert.throws(
    () => builder.weight('short', [2, 2], 'float32', new Float32Array(1)),
    /byte length 4 does not match expected 16/,
  );
  assert.equal(builder.getTensor('short'), undefined);
  assert.throws(
    () => builder.weight('wrong_view', [1], 'float32', new Int32Array(1)),
    /typed storage does not match dtype 'float32'/,
  );
  assert.throws(() => builder.input('unknown', [1], 'float64'), /Unsupported runtime tensor dtype/);

  const weight = builder.weight('weight', [2], 'float32', new Float32Array(2));
  const revision = builder.graph.topologyRevision;
  assert.throws(
    () => builder.updateTensor('weight', { shape: [3], buffer: new Float32Array(2) }),
    /byte length 8 does not match expected 12/,
  );
  assert.equal(builder.getTensor('weight'), weight);
  assert.deepEqual(weight.shape, [2]);
  assert.equal(builder.graph.topologyRevision, revision);
});

test('dynamic graph bookkeeping remains compatible with blueprint output renaming', () => {
  const graph = new Graph();
  const tensors = new Map();
  tensors.set('x', graph.addInput('x', [1]));
  GraphLoader._buildFromBlueprint(graph, {
    nodes: [
      { op: 'Identity', inputs: { input: 'x' }, outputs: { out: 'hidden' }, outputs_shape: { out: [1] } },
      { op: 'Identity', inputs: { input: 'hidden' }, outputs: { out: 'result' }, outputs_shape: { out: [1] } },
    ],
  }, tensors);
  graph.outputNames = ['result'];
  assert.equal(graph.validate().valid, true);
  assert.deepEqual(graph.nodes.map((node) => node.outputs.out.name), ['hidden', 'result']);
  assert.deepEqual(graph.nodes.map((node) => node.outputs.out.dtype), ['float32', 'float32']);
});

test('blueprints round-trip declared output dtypes and reject malformed dtype maps', () => {
  const builder = new ModelBuilder();
  const input = builder.input('bytes', [2], 'int8');
  builder.addNode({
    id: 'typed-copy', opType: 'Identity', inputs: { input },
    outputs: { out: { name: 'copy', shape: [2], dtype: 'int8' } },
  });
  const config = builder.toConfig();
  assert.deepEqual(config.nodes[0].outputs_dtype, { out: 'int8' });

  const loaded = new Graph();
  const loadedInput = loaded.addInput('bytes', [2], 'int8');
  GraphLoader._buildFromBlueprint(loaded, config, new Map([['bytes', loadedInput]]));
  assert.equal(loaded.getTensor('copy').dtype, 'int8');

  const invalidBlueprint = (outputs_dtype) => ({
    nodes: [{
      id: 'typed-copy', op: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputs_shape: { out: [1] }, outputs_dtype,
    }],
  });
  const loadInvalid = (outputs_dtype) => {
    const graph = new Graph();
    const inputTensor = graph.addInput('x', [1]);
    GraphLoader._buildFromBlueprint(graph, invalidBlueprint(outputs_dtype), new Map([['x', inputTensor]]));
  };
  assert.throws(() => loadInvalid({}), /output 'out' requires a declared dtype/);
  assert.throws(() => loadInvalid({ out: 'float64' }), /unsupported dtype 'float64'/);
  assert.throws(() => loadInvalid({ out: 'int8', extra: 'int8' }), /declares unknown output 'extra'/);
  assert.throws(() => loadInvalid('int8'), /outputs_dtype must be an object/);
  const conflictingQuantization = {
    nodes: [{
      id: 'ambiguous-q', op: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputs_shape: { out: { shape: [1], quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } } },
      outputs_dtype: { out: 'int8' },
      outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 } },
    }],
  };
  const conflictingGraph = new Graph();
  const conflictingInput = conflictingGraph.addInput('x', [1]);
  assert.throws(
    () => GraphLoader._buildFromBlueprint(conflictingGraph, conflictingQuantization, new Map([['x', conflictingInput]])),
    /cannot declare quantization in both its shape descriptor and outputs_quantization/,
  );
});

test('blueprints preserve immutable quantization descriptors on typed tensor edges', () => {
  const builder = new ModelBuilder();
  const input = builder.input('input', [1, 2], 'uint8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  });
  const weight = builder.weight('weight', [2, 2], 'int8', Int8Array.of(1, 2, 3, 4));
  builder.updateTensor('weight', {
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [0, 0] },
  });
  builder.addNode({
    id: 'typed-copy', opType: 'Identity', inputs: { input },
    outputs: {
      out: {
        name: 'output', shape: [1, 2], dtype: 'uint8',
        quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      },
    },
  });
  const config = builder.toConfig();
  assert.deepEqual(config.inputs.input.quantization, { scheme: 'per_tensor', scale: 0.25, zero_point: 128 });
  assert.deepEqual(config.weights_quantization.weight, {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [0, 0],
  });
  assert.deepEqual(config.nodes[0].outputs_quantization.out, {
    scheme: 'per_tensor', scale: 0.25, zero_point: 128,
  });

  const loaded = new Graph();
  const loadedInput = loaded.addInput('input', [1, 2], 'uint8', {
    quantization: config.inputs.input.quantization,
  });
  GraphLoader._buildFromBlueprint(loaded, config, new Map([['input', loadedInput]]));
  assert.deepEqual(loaded.getTensor('output').quantization, config.nodes[0].outputs_quantization.out);
  assert.equal(Object.isFrozen(loaded.getTensor('output').quantization), true);
  assert.throws(
    () => builder.input('bad', [1], 'float32', { quantization: { scale: 0.25, zero_point: 0 } }),
    /requires int8 or uint8 tensor storage/,
  );
  const badGraph = new Graph();
  const badInput = badGraph.addInput('input', [1]);
  assert.throws(
    () => GraphLoader._buildFromBlueprint(badGraph, {
      nodes: [{
        id: 'bad', op: 'Identity', inputs: { input: 'input' }, outputs: { out: 'out' },
        outputs_shape: { out: [1] },
        outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } },
      }],
    }, new Map([['input', badInput]])),
    /quantization requires an explicit outputs_dtype/,
  );
});

test('browser graph loading accepts canonical physical W8A8 QConv2D and rejects the legacy sidecar form', () => {
  const graph = new Graph();
  const tensors = new Map();
  tensors.set('x', graph.addInput('x', [1, 1, 1, 1], 'float32'));
  tensors.set('input_scale', graph.addWeight('input_scale', [1], 'float32', { buffer: Float32Array.of(0.25) }));
  tensors.set('input_zero_point', graph.addWeight('input_zero_point', [1], 'int8', { buffer: Int8Array.of(0) }));
  tensors.set('w', graph.addWeight('w', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  }));
  tensors.set('bias', graph.addWeight('bias', [1], 'int32', { buffer: Int32Array.of(0) }));
  GraphLoader._buildFromBlueprint(graph, {
    nodes: [
      {
        op: 'QuantizeLinear', inputs: { input: 'x', scale: 'input_scale', zero_point: 'input_zero_point' }, outputs: { out: 'qx' },
        outputs_shape: { out: [1, 1, 1, 1] },
        outputs_dtype: { out: 'int8' },
        outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } },
        params: {},
      },
      {
        op: 'QConv2D', inputs: { input: 'qx', weight: 'w', bias: 'bias' }, outputs: { out: 'y' },
        outputs_shape: { out: [1, 1, 1, 1] },
        outputs_dtype: { out: 'int8' },
        outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } },
        params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
      },
    ],
  }, tensors);
  assert.equal(graph.getTensor('qx').dtype, 'int8');
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(graph));

  const legacy = new Graph();
  const qx = legacy.addInput('qx', [1, 1, 1, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  const weight = legacy.addWeight('w', [1, 1, 1, 1], 'int8', { buffer: Int8Array.of(1) });
  const scale = legacy.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  legacy.addOp('QConv2D', { input: qx, weight, weight_scale: scale }, {
    out: { name: 'y', shape: [1, 1, 1, 1] },
  }, { input_scale: 0.25, input_zero_point: 0, output_scale: 0.25, output_zero_point: 0 });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(legacy),
    /must use canonical physical I8\/U8 NHWC\/OHWI storage/,
  );
});

test('browser W8A8 gate fails closed on generic byte activations but preserves explicit boundaries and W8A32 weights', () => {
  const q = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 };
  const generic = new Graph();
  const typedInput = generic.addInput('typed', [2], 'int8', { quantization: q });
  generic.addOp('ReLU', { input: typedInput }, {
    out: { name: 'typed_out', shape: [2], dtype: 'int8', quantization: q },
  });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(generic),
    /unsupported generic operator.*explicit DequantizeLinear boundary/,
  );

  const bounded = new Graph();
  const f32Input = bounded.addInput('input', [2], 'float32');
  const scale = bounded.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  const zeroPoint = bounded.addWeight('zero_point', [1], 'int8', { buffer: Int8Array.of(-3) });
  const { out: quantized } = bounded.addOp('QuantizeLinear', {
    input: f32Input, scale, zero_point: zeroPoint,
  }, {
    out: { name: 'quantized', shape: [2], dtype: 'int8', quantization: q },
  });
  bounded.addOp('DequantizeLinear', { input: quantized, scale, zero_point: zeroPoint }, {
    out: { name: 'restored', shape: [2], dtype: 'float32' },
  });
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(bounded));

  const weightOnly = new Graph();
  const f32 = weightOnly.addInput('input', [1, 1], 'float32');
  const packedWeight = weightOnly.addWeight('weight', [1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  });
  weightOnly.addOp('MatMul', { input: f32, weight: packedWeight }, {
    out: { name: 'out', shape: [1, 1], dtype: 'float32' },
  });
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(weightOnly));

  const malformedQConv = new Graph();
  const qconvInput = malformedQConv.addInput('input', [1, 1, 1, 1], 'float32');
  const qconvWeight = malformedQConv.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  });
  malformedQConv.addOp('QConv2D', { input: qconvInput, weight: qconvWeight }, {
    out: { name: 'out', shape: [1, 1, 1, 1], dtype: 'float32' },
  });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(malformedQConv),
    /canonical physical I8\/U8 NHWC\/OHWI storage/,
  );
});

test('browser W8A8 gate rejects unsupported typed MaxPool and Resize semantics', () => {
  const q = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
  const maxPool = new Graph();
  const pooledInput = maxPool.addInput('input', [1, 3, 3, 1], 'int8', { quantization: q });
  maxPool.addOp('MaxPool2D', { input: pooledInput }, {
    out: { name: 'out', shape: [1, 2, 2, 1], dtype: 'int8', quantization: q },
  }, { kernel: [2, 2], stride: [1, 1], dilation: [2, 1] });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(maxPool),
    /only with unit dilation/,
  );

  const resize = new Graph();
  const resizeInput = resize.addInput('input', [1, 2, 2, 1], 'uint8', { quantization: q });
  resize.addOp('Resize', { input: resizeInput }, {
    out: { name: 'out', shape: [1, 4, 4, 1], dtype: 'uint8', quantization: q },
  }, { mode: 'nearest', coordinate_transformation_mode: 'half_pixel' });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(resize),
    /coordinate_transformation_mode "asymmetric"/,
  );
});
