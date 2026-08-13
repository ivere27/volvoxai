import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';
import { RuntimeGraphBuilder } from '../ts/core/RuntimeGraphBuilder.js';
import { Tensor } from '../ts/core/Tensor.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { TrainingModelBuilder } from '../ts/training/TrainingModelBuilder.js';

test('inference builder exposes generic GroupNorm without training authoring helpers', () => {
  const builder = new RuntimeGraphBuilder();
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
  for (const undeclaredName of ['addTensor', 'addInput', 'addWeight', 'node', 'op']) {
    assert.equal(builder[undeclaredName], undefined, `${undeclaredName} is not a RuntimeGraphBuilder method`);
  }
});

test('RuntimeGraph and RuntimeGraphBuilder reject undeclared node, storage, and output fields', () => {
  const builder = new RuntimeGraphBuilder();
  assert.throws(
    () => builder.tensor('invalid-data', [1], 'float32', { data: Float32Array.of(1) }),
    /unsupported field 'data'.*use 'buffer'/,
  );
  const input = builder.input('x', [1]);
  for (const spec of [
    { op: 'Identity', inputs: { input }, outputs: { out: [1] } },
    {
      opType: 'Identity', inputs: { input },
      outputs: { out: { name: 'data-out', shape: [1], data: Float32Array.of(1) } },
    },
    {
      opType: 'Identity', inputs: { input },
      outputs: { out: { name: 'replace-out', shape: [1], replace: true } },
    },
    {
      opType: 'Identity', inputs: { input },
      outputs: { out: { name: 'reuse-out', shape: [1], reuse: true } },
    },
  ]) {
    assert.throws(() => builder.addNode(spec), /unsupported field/);
  }
  assert.equal(builder.graph.nodes.length, 0);

  assert.throws(
    () => RuntimeGraphLoader._buildFromGraphDocument(builder.graph, {
      nodes: [{
        op: 'Identity', inputs: { input: 'x' },
        outputs: { out: 'y' }, outputs_shape: { out: [1] },
      }],
    }, new Map([['x', input]])),
    /unsupported field 'op'/,
  );
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

test('RuntimeGraphBuilder creates an executable model from an empty graph', async () => {
  const builder = new RuntimeGraphBuilder();
  assert.ok(builder.build() instanceof RuntimeGraph);

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
  const builder = new RuntimeGraphBuilder();
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

test('RuntimeGraph batch construction validates and publishes atomically', () => {
  const graph = new RuntimeGraph();
  const sources = graph._addTensorsBatch([
    { name: 'x', shape: [1], dtype: 'float32', options: { role: 'input' } },
    {
      name: 'scale', shape: [1], dtype: 'float32',
      options: { role: 'weight', buffer: Float32Array.of(1) },
    },
  ]);
  assert.deepEqual(sources.map((tensor) => tensor.name), ['x', 'scale']);
  assert.equal(graph.topologyRevision, 1);
  const sourceRevision = graph.topologyRevision;
  const sourceEntries = [...graph.tensors.entries()];
  assert.throws(() => graph._addTensorsBatch([
    { name: 'staged_source', shape: [1], dtype: 'float32', options: { role: 'input' } },
    { name: 'x', shape: [1], dtype: 'float32', options: { role: 'input' } },
  ]), /Tensor 'x' already exists/);
  assert.equal(graph.topologyRevision, sourceRevision);
  assert.deepEqual([...graph.tensors.entries()], sourceEntries);
  assert.equal(graph.getTensor('staged_source'), undefined);

  const revision = graph.topologyRevision;
  const added = graph._addNodesBatch([
    {
      id: 'head', opType: 'Identity', inputs: { input: 'x' },
      outputs: { out: { name: 'hidden', shape: [1] } },
    },
    {
      id: 'tail', opType: 'Identity', inputs: { input: 'hidden' },
      outputs: { out: { name: 'result', shape: [1] } },
    },
  ]);
  assert.deepEqual(added.map((node) => node.id), ['head', 'tail']);
  assert.equal(graph.topologyRevision, revision + 1);
  assert.deepEqual(graph.nodes.map((node) => node.id), ['head', 'tail']);
  graph.assertValid();

  const committedRevision = graph.topologyRevision;
  const committedNodes = [...graph.nodes];
  const committedTensors = [...graph.tensors.entries()];
  assert.throws(() => graph._addNodesBatch([
    {
      id: 'staged', opType: 'Identity', inputs: { input: 'result' },
      outputs: { out: { name: 'staged_out', shape: [1] } },
    },
    {
      id: 'invalid', opType: 'Identity', inputs: { input: 'missing' },
      outputs: { out: { name: 'invalid_out', shape: [1] } },
    },
  ]), /references missing tensor 'missing'/);
  assert.equal(graph.topologyRevision, committedRevision);
  assert.deepEqual(graph.nodes, committedNodes);
  assert.deepEqual([...graph.tensors.entries()], committedTensors);
  assert.equal(graph.getTensor('staged_out'), undefined);
  assert.equal(graph.getNodeById('staged'), undefined);
  graph.assertValid();
});

test('quantization normalization reuses only immutable compatible descriptors', () => {
  const immutable = Object.freeze({
    scheme: 'per_axis',
    axis: 0,
    scales: Object.freeze([0.5, 0.25]),
    zero_points: Object.freeze([0, 0]),
  });
  const first = Tensor.normalizeQuantization('int8', [2, 3], immutable);
  const second = Tensor.normalizeQuantization('int8', [2, 7], immutable);
  assert.strictEqual(second, first);
  assert.throws(
    () => Tensor.normalizeQuantization('int8', [3, 3], immutable),
    /scales must contain one value for each axis-0 element/,
  );

  const mutable = {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [0, 0],
  };
  Tensor.normalizeQuantization('int8', [2, 3], mutable);
  mutable.scales[0] = -1;
  assert.throws(
    () => Tensor.normalizeQuantization('int8', [2, 3], mutable),
    /scales\[0\] must be finite, positive/,
  );
});

test('builder topology transactions restore multi-edit failures exactly', async () => {
  const builder = new RuntimeGraphBuilder();
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
  const builder = new RuntimeGraphBuilder();
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
  const builder = new RuntimeGraphBuilder();
  assert.ok(builder instanceof RuntimeGraphBuilder);
  const input = builder.input('x', [1]);
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

test('builder exports a canonical graph document with stable tensor names', () => {
  const builder = new RuntimeGraphBuilder();
  const input = builder.input('tokens', [1, 4], 'int32');
  builder.addNode({
    id: 'copy', opType: 'Identity', inputs: { input }, outputs: { out: { name: 'copied', shape: [1, 4], dtype: 'int32' } },
  });
  assert.deepEqual(builder.toGraphDocument(), {
    format: 'volvox-graph/v1',
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

test('graph document loading preserves an explicit output selection and rejects mappings', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('x', [1], 'float32');
  const document = {
    format: 'volvox-graph/v1',
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
  RuntimeGraphLoader._buildFromGraphDocument(graph, document, new Map([['x', input]]));
  assert.deepEqual(graph.outputNames, ['visible']);

  const mappedGraph = new RuntimeGraph();
  const mappedInput = mappedGraph.addInput('x', [1], 'float32');
  assert.throws(() => RuntimeGraphLoader._buildFromGraphDocument(mappedGraph, {
    ...document,
    outputs: { source_output_name: 'visible' },
  }, new Map([['x', mappedInput]])), /graph outputs must be a non-empty array/);
  assert.equal(mappedGraph.nodes.length, 0);

  const invalidOutputGraph = new RuntimeGraph();
  const invalidOutputInput = invalidOutputGraph.addInput('x', [1], 'float32');
  const invalidRevision = invalidOutputGraph.topologyRevision;
  assert.throws(() => RuntimeGraphLoader._buildFromGraphDocument(invalidOutputGraph, {
    ...document,
    outputs: ['missing'],
  }, new Map([['x', invalidOutputInput]])), /graph outputs/);
  assert.equal(invalidOutputGraph.nodes.length, 0);
  assert.equal(invalidOutputGraph.topologyRevision, invalidRevision);

  for (const outputs of [['visible', 'visible']]) {
    const duplicateGraph = new RuntimeGraph();
    const duplicateInput = duplicateGraph.addInput('x', [1], 'float32');
    assert.throws(() => RuntimeGraphLoader._buildFromGraphDocument(duplicateGraph, {
      ...document,
      outputs,
    }, new Map([['x', duplicateInput]])), /unique, non-empty graph tensor/);
    assert.equal(duplicateGraph.nodes.length, 0);
  }
});

test('graph documents serialize canonical linear layout for square weights', async () => {
  const builder = new RuntimeGraphBuilder();
  const input = builder.input('x', [1, 2]);
  const weight = builder.weight('w', [2, 2], 'float32', Float32Array.from([1, 2, 3, 4]));
  builder.addNode({
    id: 'square', opType: 'MatMul', inputs: { input, weight },
    outputs: { out: { name: 'y', shape: [1, 2] } }, wLayout: 'din',
  });
  const document = builder.toGraphDocument();
  assert.equal(document.nodes[0].params.weight_layout, 'IN_OUT');
  assert.equal(Object.hasOwn(document.nodes[0], 'wLayout'), false);

  const loaded = new RuntimeGraph();
  const loadedInput = loaded.addInput('x', [1, 2]);
  const loadedWeight = loaded.addWeight('w', [2, 2], 'float32', {
    buffer: Float32Array.from([1, 2, 3, 4]),
  });
  RuntimeGraphLoader._buildFromGraphDocument(loaded, document, new Map([
    ['x', loadedInput], ['w', loadedWeight],
  ]));
  RuntimeGraphLoader._resolveMatMulLayouts(loaded);
  loaded.setOutputs(['y']);
  assert.equal(loaded.nodes[0].wLayout, 'din');
  const engine = new CPUEngine();
  engine.allocateGraph(loaded);
  assert.deepEqual([...(await engine.execute({ x: Float32Array.from([1, 0]) })).y], [1, 2]);
});

test('graph documents reject undeclared inputs instead of inventing an image tensor', () => {
  assert.throws(
    () => RuntimeGraphLoader._buildFromGraphDocument(new RuntimeGraph(), {
      nodes: [{
        id: 'missing_input',
        opType: 'Identity',
        inputs: { input: 'undeclared' },
        outputs: { out: 'result' },
        outputs_shape: { out: [1] },
      }],
    }, new Map()),
    /input 'input' references undeclared tensor 'undeclared'/,
  );
});

test('graph documents reject output collisions before mutating the graph', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('x', [1]);
  const tensors = new Map([['x', input]]);
  const revision = graph.topologyRevision;

  assert.throws(
    () => RuntimeGraphLoader._buildFromGraphDocument(graph, {
      nodes: [{
        id: 'overwrite-input',
        opType: 'Identity',
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
    () => RuntimeGraphLoader._buildFromGraphDocument(graph, {
      nodes: [
        {
          id: 'first', opType: 'Identity', inputs: { input: 'x' },
          outputs: { out: 'shared' }, outputs_shape: { out: [1] },
        },
        {
          id: 'second', opType: 'Identity', inputs: { input: 'shared' },
          outputs: { out: 'shared' }, outputs_shape: { out: [1] },
        },
      ],
    }, tensors),
    /output 'out' collides with existing tensor 'shared'/,
  );
  assert.equal(graph.nodes.length, 0);
  assert.equal(graph.topologyRevision, revision);
});

test('graph document loading validates and commits one staged topology transaction', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1]);
  const tensors = new Map([['input', input]]);
  const revision = graph.topologyRevision;
  RuntimeGraphLoader._buildFromGraphDocument(graph, {
    nodes: [
      {
        opType: 'Identity', inputs: { input: 'input' }, outputs: { out: 'first' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
      {
        opType: 'Identity', inputs: { input: 'first' }, outputs: { out: 'second' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
      {
        opType: 'Identity', inputs: { input: 'second' }, outputs: { out: 'third' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
    ],
    outputs: ['third'],
  }, tensors);
  assert.equal(graph.topologyRevision, revision + 1);
  assert.equal(graph.nodes.length, 3);
  assert.deepEqual(graph.outputNames, ['third']);
  graph.assertValid();

  const failed = new RuntimeGraph();
  const failedInput = failed.addInput('input', [1]);
  const failedAliases = new Map([['input', failedInput]]);
  const failedRevision = failed.topologyRevision;
  assert.throws(() => RuntimeGraphLoader._buildFromGraphDocument(failed, {
    nodes: [
      {
        opType: 'Identity', inputs: { input: 'input' }, outputs: { out: 'staged_only' },
        outputs_shape: { out: [1] },
        outputs_dtype: { out: 'float32' },
      },
      {
        opType: 'Identity', inputs: { input: 'staged_only' }, outputs: { out: 'invalid' },
        outputs_shape: { out: [0] },
        outputs_dtype: { out: 'float32' },
      },
    ],
    outputs: ['invalid'],
  }, failedAliases), /positive integer/);
  assert.equal(failed.topologyRevision, failedRevision);
  assert.equal(failed.nodes.length, 0);
  assert.deepEqual([...failed.tensors.keys()], ['input']);
  assert.deepEqual([...failedAliases.keys()], ['input']);
  failed.assertValid();
});

test('graph documents serialize explicit Conv2D image-weight layouts', () => {
  const regular = new RuntimeGraphBuilder();
  const image = regular.input('image', [1, 2, 2, 2]);
  const regularWeight = regular.weight('regular_weight', [1, 1, 2, 3], 'float32', new Float32Array(6));
  regular.addOp('Conv2D', { input: image, weight: regularWeight }, { out: [1, 2, 2, 3] });
  assert.equal(regular.toGraphDocument().nodes[0].params.weight_layout, 'HWIO');

  const depthwise = new RuntimeGraphBuilder();
  const depthImage = depthwise.input('image', [1, 2, 2, 2]);
  const depthWeight = depthwise.weight('depth_weight', [1, 1, 2, 1], 'float32', new Float32Array(2));
  depthwise.addOp('Conv2D', { input: depthImage, weight: depthWeight }, { out: [1, 2, 2, 2] }, {
    groups: 2, weight_layout: 'HWCM',
  });
  assert.equal(depthwise.toGraphDocument().nodes[0].params.weight_layout, 'HWCM');
});

test('browser Conv2D layout normalization transforms shared depthwise weights once', () => {
  const graph = new RuntimeGraph();
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

  RuntimeGraphLoader._normalizeConvWeightsForImageLayout(graph);
  assert.deepEqual(weight.shape, [1, 1, 2, 1]);
  assert.deepEqual([...weight.buffer], [2, 3]);
  assert.deepEqual(graph.nodes.map((node) => node.params.weight_layout), ['HWCM', 'HWCM']);
});

test('tensor payload dtype and byte length are validated transactionally', () => {
  const builder = new RuntimeGraphBuilder();
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

test('dynamic graph bookkeeping preserves explicitly selected outputs across renaming', () => {
  const graph = new RuntimeGraph();
  const tensors = new Map();
  tensors.set('x', graph.addInput('x', [1]));
  RuntimeGraphLoader._buildFromGraphDocument(graph, {
    nodes: [
      {
        opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'hidden' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
      },
      {
        opType: 'Identity', inputs: { input: 'hidden' }, outputs: { out: 'result' },
        outputs_shape: { out: [1] }, outputs_dtype: { out: 'float32' },
      },
    ],
    outputs: ['result'],
  }, tensors);
  graph.setOutputs(['result']);
  assert.equal(graph.validate().valid, true);
  assert.deepEqual(graph.nodes.map((node) => node.outputs.out.name), ['hidden', 'result']);
  assert.deepEqual(graph.nodes.map((node) => node.outputs.out.dtype), ['float32', 'float32']);
  assert.throws(() => { graph.outputNames = ['hidden']; }, TypeError);
  assert.throws(() => { graph.outputNames.push('hidden'); }, TypeError);
});

test('graph documents round-trip declared output dtypes and reject malformed dtype maps', () => {
  const builder = new RuntimeGraphBuilder();
  const input = builder.input('bytes', [2], 'int8');
  builder.addNode({
    id: 'typed-copy', opType: 'Identity', inputs: { input },
    outputs: { out: { name: 'copy', shape: [2], dtype: 'int8' } },
  });
  const document = builder.toGraphDocument();
  assert.deepEqual(document.nodes[0].outputs_dtype, { out: 'int8' });

  const loaded = new RuntimeGraph();
  const loadedInput = loaded.addInput('bytes', [2], 'int8');
  RuntimeGraphLoader._buildFromGraphDocument(loaded, document, new Map([['bytes', loadedInput]]));
  assert.equal(loaded.getTensor('copy').dtype, 'int8');

  const invalidGraphDocument = (outputs_dtype) => ({
    nodes: [{
      id: 'typed-copy', opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputs_shape: { out: [1] }, outputs_dtype,
    }],
    outputs: ['y'],
  });
  const loadInvalid = (outputs_dtype) => {
    const graph = new RuntimeGraph();
    const inputTensor = graph.addInput('x', [1]);
    RuntimeGraphLoader._buildFromGraphDocument(graph, invalidGraphDocument(outputs_dtype), new Map([['x', inputTensor]]));
  };
  assert.throws(() => loadInvalid(undefined), /outputs_dtype must exactly describe/);
  assert.throws(() => loadInvalid({}), /outputs_dtype must exactly describe/);
  assert.throws(() => loadInvalid({ out: 'float64' }), /unsupported dtype 'float64'/);
  assert.throws(() => loadInvalid({ out: 'int8', extra: 'int8' }), /outputs_dtype must exactly describe/);
  assert.throws(() => loadInvalid('int8'), /outputs_dtype must exactly describe/);
  const embeddedDescriptor = {
    nodes: [{
      id: 'embedded-dtype', opType: 'Identity', inputs: { input: 'x' }, outputs: { out: 'y' },
      outputs_shape: { out: { shape: [1], dtype: 'int8' } },
      outputs_dtype: { out: 'float32' },
    }],
    outputs: ['y'],
  };
  const descriptorGraph = new RuntimeGraph();
  const descriptorInput = descriptorGraph.addInput('x', [1]);
  assert.throws(
    () => RuntimeGraphLoader._buildFromGraphDocument(descriptorGraph, embeddedDescriptor, new Map([['x', descriptorInput]])),
    /requires a shape array/,
  );
});

test('graph packages store immutable quantization parameters only in Safetensors', () => {
  const builder = new RuntimeGraphBuilder();
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
  assert.throws(() => builder.toGraphDocument(), /must use toGraphPackage/);
  const { graph: document, quantizationParameters } = builder.toGraphPackage();
  assert.equal(Object.hasOwn(document.inputs.input, 'quantization'), false);
  assert.equal(Object.hasOwn(document, 'weights_quantization'), false);
  assert.equal(Object.hasOwn(document.nodes[0], 'outputs_quantization'), false);
  assert.equal(document.quantization.format, 'volvox-affine-safetensors/v1');
  assert.equal(Object.keys(document.quantization.tensors).length, 3);
  const inputDescriptor = document.quantization.tensors.input;
  const outputDescriptor = document.quantization.tensors.output;
  const weightDescriptor = document.quantization.tensors.weight;
  assert.deepEqual(inputDescriptor, outputDescriptor);
  assert.equal(weightDescriptor.scheme, 'per_axis');
  assert.equal(weightDescriptor.axis, 0);
  assert.deepEqual(
    [...quantizationParameters.toRuntimeTypedArray(
      quantizationParameters.getTensor(inputDescriptor.scale_tensor),
    )],
    [0.25],
  );
  assert.deepEqual(
    [...quantizationParameters.toRuntimeTypedArray(
      quantizationParameters.getTensor(inputDescriptor.zero_point_tensor),
    )],
    [128],
  );
  assert.deepEqual(
    [...quantizationParameters.toRuntimeTypedArray(
      quantizationParameters.getTensor(weightDescriptor.scale_tensor),
    )],
    [0.5, 0.25],
  );
  assert.equal(JSON.stringify(document).includes('"scale":'), false);
  assert.equal(JSON.stringify(document).includes('"zero_point":'), false);

  const loaded = new RuntimeGraph();
  const loadedInput = loaded.addInput('input', [1, 2], 'uint8', {
    quantization: input.quantization,
  });
  RuntimeGraphLoader._buildFromGraphDocument(
    loaded,
    document,
    new Map([['input', loadedInput]]),
    { quantizationByTensor: {
      input: input.quantization,
      output: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      weight: weight.quantization,
    } },
  );
  assert.deepEqual(loaded.getTensor('output').quantization, outputDescriptor.scheme === 'per_tensor'
    ? { scheme: 'per_tensor', scale: 0.25, zero_point: 128 }
    : null);
  assert.equal(Object.isFrozen(loaded.getTensor('output').quantization), true);
  assert.throws(
    () => builder.input('bad', [1], 'float32', { quantization: { scale: 0.25, zero_point: 0 } }),
    /requires int8 or uint8 tensor storage/,
  );
  const badGraph = new RuntimeGraph();
  const badInput = badGraph.addInput('input', [1]);
  assert.throws(
    () => RuntimeGraphLoader._buildFromGraphDocument(badGraph, {
      nodes: [{
        id: 'bad', opType: 'Identity', inputs: { input: 'input' }, outputs: { out: 'out' },
        outputs_shape: { out: [1] },
        outputs_quantization: { out: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } },
      }],
      outputs: ['out'],
    }, new Map([['input', badInput]])),
    /forbidden inline quantization/,
  );
});

test('browser graph loading accepts canonical physical W8A8 QConv2D and rejects noncanonical sidecar metadata', () => {
  const graph = new RuntimeGraph();
  const tensors = new Map();
  tensors.set('x', graph.addInput('x', [1, 1, 1, 1], 'float32'));
  tensors.set('input_scale', graph.addWeight('input_scale', [1], 'float32', { buffer: Float32Array.of(0.25) }));
  tensors.set('input_zero_point', graph.addWeight('input_zero_point', [1], 'int8', { buffer: Int8Array.of(0) }));
  tensors.set('weight_scale', graph.addWeight('weight_scale', [1], 'float32', { buffer: Float32Array.of(0.25) }));
  tensors.set('weight_zero_point', graph.addWeight('weight_zero_point', [1], 'int8', { buffer: Int8Array.of(0) }));
  tensors.set('w', graph.addWeight('w', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  }));
  tensors.set('bias', graph.addWeight('bias', [1], 'int32', { buffer: Int32Array.of(0) }));
  const quantizedDocument = {
    format: 'volvox-graph/v1',
    inputs: { x: { shape: [1, 1, 1, 1], dtype: 'float32' } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        qx: { scheme: 'per_tensor', scale_tensor: 'input_scale', zero_point_tensor: 'input_zero_point' },
        w: { scheme: 'per_axis', axis: 0, scale_tensor: 'weight_scale', zero_point_tensor: 'weight_zero_point' },
        y: { scheme: 'per_tensor', scale_tensor: 'input_scale', zero_point_tensor: 'input_zero_point' },
      },
    },
    nodes: [
      {
        opType: 'QuantizeLinear', inputs: { input: 'x', scale: 'input_scale', zero_point: 'input_zero_point' }, outputs: { out: 'qx' },
        outputs_shape: { out: [1, 1, 1, 1] },
        outputs_dtype: { out: 'int8' },
        params: {},
      },
      {
        opType: 'QConv2D', inputs: { input: 'qx', weight: 'w', bias: 'bias' }, outputs: { out: 'y' },
        outputs_shape: { out: [1, 1, 1, 1] },
        outputs_dtype: { out: 'int8' },
        params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
      },
    ],
    outputs: ['y'],
  };
  RuntimeGraphLoader._buildFromGraphDocument(graph, quantizedDocument, tensors, {
    quantizationByTensor: {
      qx: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      w: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
      y: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  });
  assert.equal(graph.getTensor('qx').dtype, 'int8');
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));

  const noncanonical = new RuntimeGraph();
  const qx = noncanonical.addInput('qx', [1, 1, 1, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  const weight = noncanonical.addWeight('w', [1, 1, 1, 1], 'int8', { buffer: Int8Array.of(1) });
  const scale = noncanonical.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  noncanonical.addOp('QConv2D', { input: qx, weight, weight_scale: scale }, {
    out: { name: 'y', shape: [1, 1, 1, 1] },
  }, { input_scale: 0.25, input_zero_point: 0, output_scale: 0.25, output_zero_point: 0 });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(noncanonical),
    /must use canonical physical I8\/U8 NHWC\/OHWI storage/,
  );
});

test('browser W8A8 gate fails closed on generic byte activations but preserves explicit boundaries and W8A32 weights', () => {
  const q = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 };
  const generic = new RuntimeGraph();
  const typedInput = generic.addInput('typed', [2], 'int8', { quantization: q });
  generic.addOp('ReLU', { input: typedInput }, {
    out: { name: 'typed_out', shape: [2], dtype: 'int8', quantization: q },
  });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(generic),
    /unsupported generic operator.*explicit DequantizeLinear boundary/,
  );

  const bounded = new RuntimeGraph();
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
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(bounded));

  const weightOnly = new RuntimeGraph();
  const f32 = weightOnly.addInput('input', [1, 1], 'float32');
  const packedWeight = weightOnly.addWeight('weight', [1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  });
  weightOnly.addOp('MatMul', { input: f32, weight: packedWeight }, {
    out: { name: 'out', shape: [1, 1], dtype: 'float32' },
  });
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(weightOnly));

  const malformedQConv = new RuntimeGraph();
  const qconvInput = malformedQConv.addInput('input', [1, 1, 1, 1], 'float32');
  const qconvWeight = malformedQConv.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
  });
  malformedQConv.addOp('QConv2D', { input: qconvInput, weight: qconvWeight }, {
    out: { name: 'out', shape: [1, 1, 1, 1], dtype: 'float32' },
  });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(malformedQConv),
    /canonical physical I8\/U8 NHWC\/OHWI storage/,
  );
});

test('browser W8A8 gate rejects unsupported typed MaxPool and Resize semantics', () => {
  const q = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
  const maxPool = new RuntimeGraph();
  const pooledInput = maxPool.addInput('input', [1, 3, 3, 1], 'int8', { quantization: q });
  maxPool.addOp('MaxPool2D', { input: pooledInput }, {
    out: { name: 'out', shape: [1, 2, 2, 1], dtype: 'int8', quantization: q },
  }, { kernel: [2, 2], stride: [1, 1], dilation: [2, 1] });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(maxPool),
    /only with unit dilation/,
  );

  const resize = new RuntimeGraph();
  const resizeInput = resize.addInput('input', [1, 2, 2, 1], 'uint8', { quantization: q });
  resize.addOp('Resize', { input: resizeInput }, {
    out: { name: 'out', shape: [1, 4, 4, 1], dtype: 'uint8', quantization: q },
  }, { mode: 'nearest', coordinate_transformation_mode: 'half_pixel' });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(resize),
    /coordinate_transformation_mode "asymmetric"/,
  );
});

test('browser W8A8 gate validates canonical QBatchMatMul broadcast and F32 multiplier bounds', () => {
  const graph = new RuntimeGraph();
  const a = graph.addInput('a', [2, 1, 2, 3], 'uint8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 10 },
  });
  const b = graph.addInput('b', [1, 2, 3, 2], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  graph.addOp('QBatchMatMul', { a, b }, {
    out: {
      name: 'out',
      shape: [2, 2, 2, 2],
      dtype: 'uint8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 100 },
    },
  });
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));

  const minimumF32 = 1.401298464324817e-45;
  const underflow = new RuntimeGraph();
  const tinyA = underflow.addInput('a', [1, 1], 'uint8', {
    quantization: {
      scheme: 'per_tensor', scale: minimumF32, zero_point: 0,
    },
  });
  const tinyB = underflow.addInput('b', [1, 1], 'int8', {
    quantization: {
      scheme: 'per_tensor', scale: minimumF32, zero_point: 0,
    },
  });
  underflow.addOp('QBatchMatMul', { a: tinyA, b: tinyB }, {
    out: {
      name: 'out',
      shape: [1, 1],
      dtype: 'uint8',
      quantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(underflow),
    /requantization multiplier not representable as positive F32/,
  );
});

test('browser W8A8 gate rejects a non-representable QLinear multiplier', () => {
  const minimumF32 = 1.401298464324817e-45;
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: minimumF32, zero_point: 0 },
  });
  const weight = graph.addWeight('weight', [1, 1], 'int8', {
    buffer: Int8Array.of(1),
    quantization: {
      scheme: 'per_axis', axis: 0, scales: [minimumF32], zero_points: [0],
    },
  });
  const bias = graph.addWeight('bias', [1], 'int32', {
    buffer: Int32Array.of(0),
  });
  graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name: 'out',
      shape: [1, 1],
      dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    },
  });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph),
    /requantization multiplier not representable as positive F32/,
  );
});
