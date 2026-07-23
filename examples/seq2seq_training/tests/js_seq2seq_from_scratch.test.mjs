import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ModelBuilder,
  exportModelCheckpoint,
  importModelCheckpoint,
  initializeTensor,
} from '../../../ts/full.js';
import { CPUEngine } from '../../../ts/backends/CPUEngine.js';
import { createCPUTrainingHarness } from '../../../tests/helpers/training_session.mjs';
import {
  buildEncoderDecoderTransformer,
  createTeacherForcingBatchForGraph,
  validateEncoderDecoderTrainingMetadata,
} from '../Seq2SeqBuilder.js';

test('seeded built-in initializers are deterministic and non-degenerate', () => {
  const first = initializeTensor([4, 6], 'float32', { type: 'xavierUniform', seed: 123 });
  const second = initializeTensor([4, 6], 'float32', { type: 'xavierUniform', seed: 123 });
  const different = initializeTensor([4, 6], 'float32', { type: 'xavierUniform', seed: 124 });
  assert.deepEqual(first, second);
  assert.notDeepEqual(first, different);
  assert.ok(first.some((value) => value !== 0));
  assert.deepEqual(
    initializeTensor([5], 'float32', { type: 'normal', seed: 8, stddev: 0.1 }),
    initializeTensor([5], 'float32', { type: 'normal', seed: 8, stddev: 0.1 }),
  );
  assert.deepEqual(
    initializeTensor([3], 'float32', 'zeros'),
    Float32Array.of(0, 0, 0),
  );
  assert.deepEqual(
    initializeTensor([3], 'float32', 'ones'),
    Float32Array.of(1, 1, 1),
  );
});

test('empty JS model builds, teacher-forces, trains, and checkpoints a full encoder-decoder', async () => {
  const builder = new ModelBuilder();
  assert.equal(builder.graph.nodes.length, 0);
  assert.equal(builder.graph.weightFiles.length, 0);
  const model = buildEncoderDecoderTransformer(builder, {
    name: 'tiny',
    batchSize: 2,
    sourceLength: 3,
    targetLength: 3,
    vocabSize: 7,
    dModel: 4,
    numHeads: 2,
    dFF: 8,
    encoderLayers: 1,
    decoderLayers: 1,
    seed: 9,
    padTokenId: 0,
    bosTokenId: 1,
  });
  const graph = builder.build();
  assert.equal(graph.validate().valid, true);
  assert.ok(graph.nodes.some((node) => node.opType === 'SDPA' && node.params.causal === false));
  assert.ok(graph.nodes.some((node) => node.opType === 'CrossSDPA'));

  const batch = model.teacherForcing(
    [2, 3, 0, 3, 2, 4],
    [4, 5, 0, 5, 4, 3],
  );
  assert.ok(batch.inputs['tiny.source_tokens'] instanceof Int32Array);
  assert.ok(batch.inputs['tiny.decoder_tokens'] instanceof Int32Array);
  assert.deepEqual([...batch.inputs['tiny.source_mask']], [1, 1, 0, 1, 1, 1]);
  assert.deepEqual([...batch.lossMask], [1, 1, 0, 1, 1, 1]);
  assert.throws(
    () => model.teacherForcing([[2, 3], [0, 3, 2, 4]], [[4, 5, 0], [5, 4, 3]]),
    /rectangular \[2,3\]/,
  );

  const probe = graph.getTensor('tiny.encoder.0.self_attention.qkv.weight');
  const before = new Float32Array(probe.buffer);
  const training = createCPUTrainingHarness();
  const result = await training.runStep(graph, {
    ...batch,
    updateMode: 'adamw',
    optimizer: { learningRate: 2e-3 },
  });
  assert.ok(Number.isFinite(result.loss));
  assert.equal(result.examples, 5);
  assert.equal(graph.trainingStep, 0, 'Trainer must not mutate caller-owned graph state');
  assert.deepEqual(probe.buffer, before);
  assert.equal(result.updatedTensorNames.length, model.trainableTensors.length);

  const checkpoint = await training.exportCheckpoint(graph, {
    tokenizerMetadata: { type: 'test-tokenizer', bosTokenId: 1, padTokenId: 0 },
    metadata: { epoch: 1 },
  });
  const trained = importModelCheckpoint(checkpoint).graph;
  const trainedProbe = trained.getTensor(probe.name);
  assert.equal(trained.trainingStep, 1);
  assert.ok(trainedProbe.buffer.some((value, index) => value !== before[index]));
  assert.deepEqual(checkpoint.optimizerDescriptor, {
    updateMode: 'adamw',
    optimizer: {
      learningRate: 2e-3,
      weightDecay: 0,
      maxGradNorm: 0,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
  });
  await assert.rejects(
    () => training.exportCheckpoint(graph, { graph: { inputs: {}, nodes: [], outputs: [] } }),
    /cannot be overridden/,
  );
  await assert.rejects(
    () => training.exportCheckpoint(graph, { trainingStep: 10 }),
    /cannot be overridden/,
  );
  await assert.rejects(() => training.exportCheckpoint(graph, {
    optimizerDescriptor: { updateMode: 'adamw', optimizer: { learningRate: 0.5 } },
  }), /cannot be overridden/);
  assert.throws(
    () => importModelCheckpoint({ ...checkpoint, optimizerEntries: [] }),
    /optimizer state has no entries/,
  );
  assert.throws(
    () => importModelCheckpoint({ ...checkpoint, trainingStep: 0 }),
    /invalid step/,
  );
  const corruptMetadata = structuredClone(checkpoint);
  corruptMetadata.trainingMetadata.teacherForcingSpec.logitsName = probe.name;
  const corruptRestored = importModelCheckpoint(corruptMetadata);
  assert.throws(
    () => validateEncoderDecoderTrainingMetadata(corruptRestored.graph),
    /logitsName/,
  );
  for (const field of ['trainingStep', 'trainingMetadata', 'optimizerDescriptor']) {
    const truncated = structuredClone(checkpoint);
    delete truncated[field];
    assert.throws(() => importModelCheckpoint(truncated), new RegExp(`missing required field '${field}'`));
  }
  const stateForValidation = trained.optimizerState.get(probe.name);
  const savedSecondMoment = stateForValidation.v[0];
  stateForValidation.v[0] = -1;
  assert.throws(() => exportModelCheckpoint(trained), /negative/);
  stateForValidation.v[0] = savedSecondMoment;
  const checkpointWeightBytes = new Uint8Array(checkpoint.weights.slice(0));
  const restored = importModelCheckpoint(checkpoint);
  assert.equal(restored.graph.trainingStep, 1);
  assert.deepEqual(restored.tokenizerMetadata, {
    type: 'test-tokenizer', bosTokenId: 1, padTokenId: 0,
  });
  assert.deepEqual(restored.metadata, { epoch: 1 });
  assert.deepEqual(
    restored.graph.getTensor(probe.name).buffer,
    trainedProbe.buffer,
  );
  assert.equal(restored.graph.optimizerState.size, trained.optimizerState.size);
  assert.equal(restored.graph.optimizerState.get(probe.name).step, 1);
  assert.equal(restored.graph.outputNames[0], model.logits.name);

  const resumedBatch = createTeacherForcingBatchForGraph(
    restored.graph,
    [2, 3, 0, 3, 2, 4],
    [4, 5, 0, 5, 4, 3],
  );
  const uninterrupted = await training.runStep(graph, {
    ...batch,
  });
  const continued = await training.runStep(restored.graph, {
    ...resumedBatch,
  });
  assert.equal(uninterrupted.loss, continued.loss);
  assert.ok(Number.isFinite(continued.loss));
  const uninterruptedGraph = importModelCheckpoint(
    await training.exportCheckpoint(graph),
  ).graph;
  const continuedGraph = importModelCheckpoint(
    await training.exportCheckpoint(restored.graph),
  ).graph;
  assert.equal(graph.trainingStep, 0);
  assert.equal(restored.graph.trainingStep, 1);
  assert.equal(uninterruptedGraph.trainingStep, 2);
  assert.equal(continuedGraph.trainingStep, 2);
  assert.equal(continuedGraph.optimizerState.get(probe.name).step, 2);
  for (const tensor of uninterruptedGraph.tensors.values()) {
    if (!tensor.isWeight) continue;
    assert.deepEqual(continuedGraph.getTensor(tensor.name).buffer, tensor.buffer, tensor.name);
    const expectedState = uninterruptedGraph.optimizerState.get(tensor.name);
    const actualState = continuedGraph.optimizerState.get(tensor.name);
    assert.equal(actualState.step, expectedState.step, `${tensor.name} optimizer step`);
    assert.deepEqual(actualState.m, expectedState.m, `${tensor.name} first moment`);
    assert.deepEqual(actualState.v, expectedState.v, `${tensor.name} second moment`);
  }
  assert.deepEqual(new Uint8Array(checkpoint.weights), checkpointWeightBytes);

  const switched = await training.runStep(graph, {
    ...batch,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  assert.equal(switched.examples, 5);
  const switchedCheckpoint = await training.exportCheckpoint(graph);
  const switchedGraph = importModelCheckpoint(switchedCheckpoint).graph;
  assert.equal(switchedGraph.trainingStep, 3);
  assert.equal(switchedGraph.optimizerState, null,
    'switching away from AdamW drops incompatible moments');
  assert.equal(switchedCheckpoint.optimizerDescriptor.updateMode, 'sgd');
  await training.close();
});

test('encoder-decoder accepts a precomputed feature prefix and shared source/target/head embeddings', async () => {
  const builder = new ModelBuilder();
  const rawImageTokens = builder.input('receipt.raw_image_tokens', [1, 2, 4], 'float32');
  const visionWeight = builder.weight('receipt.vision.weight', [4, 4], 'float32', {
    initializer: { type: 'xavierUniform', seed: 18 },
  });
  const imageTokens = builder.addOp(
    'Linear',
    { input: rawImageTokens, weight: visionWeight },
    { out: { name: 'receipt.image_tokens', shape: [1, 2, 4] } },
    { weight_layout: 'IN_OUT' },
    { id: 'receipt.vision', wLayout: 'din' },
  ).out;
  const model = buildEncoderDecoderTransformer(builder, {
    name: 'receipt',
    batchSize: 1,
    sourceLength: 3,
    targetLength: 2,
    vocabSize: 7,
    dModel: 4,
    numHeads: 2,
    dFF: 8,
    encoderLayers: 1,
    decoderLayers: 1,
    sourceFeatures: imageTokens,
    additionalTrainableTensors: [visionWeight],
    transformMemory(memory, { builder: graphBuilder, registerTrainable }) {
      const bias = graphBuilder.weight('receipt.memory_adapter.bias', [4], 'float32', new Float32Array(4));
      registerTrainable(bias);
      return graphBuilder.addOp(
        'Add',
        { a: memory, b: bias },
        { out: { name: 'receipt.memory_adapter.out', shape: [...memory.shape] } },
        {},
        { id: 'receipt.memory_adapter' },
      ).out;
    },
    tieSourceTargetEmbeddings: true,
    tieTargetEmbeddings: true,
    lmHeadBias: true,
    seed: 19,
    padTokenId: 0,
    bosTokenId: 1,
  });
  const graph = builder.build();

  assert.deepEqual(model.memory.shape, [1, 5, 4]);
  assert.deepEqual(model.inputs.sourceMask.shape, [1, 5]);
  assert.strictEqual(model.inputs.sourceFeatures, imageTokens);
  assert.ok(model.trainableTensors.includes(visionWeight.name));
  assert.ok(model.trainableTensors.includes('receipt.memory_adapter.bias'));
  assert.ok(graph.nodes.some((node) => node.opType === 'Concat' && node.params.axis === 1));
  assert.equal(graph.getTensor('receipt.target_embedding.weight'), undefined);
  assert.ok(graph.getTensor('receipt.lm_head.bias'));
  assert.strictEqual(
    graph.getNode('receipt.lm_head').inputs.weight,
    graph.getTensor('receipt.source_embedding.weight'),
  );

  const batch = model.teacherForcing([2, 0, 3], [4, 0]);
  assert.deepEqual([...batch.inputs['receipt.source_mask']], [1, 1, 1, 0, 1]);
  assert.deepEqual([...batch.inputs['receipt.decoder_tokens']], [1, 4]);
  assert.deepEqual([...batch.lossMask], [1, 0]);
  batch.inputs[rawImageTokens.name] = new Float32Array([
    0.1, 0.2, 0.3, 0.4,
    -0.4, -0.3, -0.2, -0.1,
  ]);

  const training = createCPUTrainingHarness();
  const result = await training.runStep(graph, {
    ...batch,
    updateMode: 'sgd',
    optimizer: { learningRate: 1e-3 },
  });
  assert.ok(Number.isFinite(result.loss));
  assert.equal(result.examples, 1);

  const restored = importModelCheckpoint(await training.exportCheckpoint(graph)).graph;
  const resumed = createTeacherForcingBatchForGraph(restored, [2, 0, 3], [4, 0]);
  assert.deepEqual([...resumed.inputs['receipt.source_mask']], [1, 1, 1, 0, 1]);
  assert.equal(restored.trainingMetadata.teacherForcingSpec.sourceFeatureLength, 2);
  assert.equal(restored.trainingMetadata.teacherForcingSpec.memoryLength, 5);
  await training.close();
});

test('encoder-decoder builder validates special IDs and rolls back late failures', () => {
  const invalid = new ModelBuilder();
  assert.throws(() => buildEncoderDecoderTransformer(invalid, {
    sourceLength: 2,
    targetLength: 2,
    vocabSize: 4,
    dModel: 2,
    numHeads: 1,
    padTokenId: 1,
    bosTokenId: 1,
  }), /must be different/);
  assert.equal(invalid.graph.tensors.size, 0);

  const builder = new ModelBuilder();
  const collision = builder.weight(
    'tiny.target_embedding.weight',
    [4, 2],
    'float32',
    new Float32Array(8),
  );
  const before = {
    nodes: builder.graph.nodes.length,
    tensors: builder.graph.tensors.size,
    outputs: [...builder.graph.outputNames],
    revision: builder.graph.topologyRevision,
  };
  assert.throws(() => buildEncoderDecoderTransformer(builder, {
    name: 'tiny',
    sourceLength: 2,
    targetLength: 2,
    vocabSize: 4,
    dModel: 2,
    numHeads: 1,
  }), /already exists/);
  assert.equal(builder.graph.nodes.length, before.nodes);
  assert.equal(builder.graph.tensors.size, before.tensors);
  assert.deepEqual(builder.graph.outputNames, before.outputs);
  assert.equal(builder.graph.topologyRevision, before.revision);
  assert.strictEqual(builder.graph.getTensor(collision.name), collision);
  assert.equal(builder.graph.validate().valid, true);
  for (const name of ['__proto__', 'constructor', 'toString', '__metadata__']) {
    assert.throws(() => new ModelBuilder().weight(name, [1], 'float32', Float32Array.of(1)), /name/);
  }
});

test('checkpoints preserve explicit square-linear layouts, params, and standalone values', async () => {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const weight = builder.weight('w', [2, 2], 'float32', Float32Array.of(1, 2, 3, 4));
  const { out } = builder.addOp(
    'Linear',
    { input, weight },
    { out: { name: 'y', shape: [1, 2] } },
    { weight_layout: 'IN_OUT', nested: { marker: 7 } },
  );
  builder.outputs(out);
  const graph = builder.build();
  assert.equal(graph.nodes[0].wLayout, 'din');
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const before = new Float32Array((await engine.execute({ x: Float32Array.of(1, 2) })).y);
  assert.deepEqual(before, Float32Array.of(7, 10));

  const checkpoint = exportModelCheckpoint(graph);
  const restored = importModelCheckpoint(checkpoint).graph;
  checkpoint.graph.nodes[0].params.nested.marker = 99;
  assert.equal(restored.nodes[0].params.nested.marker, 7);
  const restoredEngine = new CPUEngine();
  restoredEngine.allocateGraph(restored);
  assert.deepEqual(
    new Float32Array((await restoredEngine.execute({ x: Float32Array.of(1, 2) })).y),
    before,
  );

  const values = new ModelBuilder();
  const constant = values.tensor('constant', [2], 'float32', {
    buffer: Float32Array.of(3, -2),
  });
  values.outputs(constant);
  const valueCheckpoint = exportModelCheckpoint(values.build());
  const valueGraph = importModelCheckpoint(valueCheckpoint).graph;
  const restoredConstant = valueGraph.getTensor('constant');
  assert.equal(restoredConstant.isWeight, false);
  assert.equal(restoredConstant.isInput, false);
  assert.deepEqual(restoredConstant.buffer, constant.buffer);
  assert.deepEqual(valueGraph.outputNames, ['constant']);

  for (const outputs of ['bogus', {}, 42, null]) {
    assert.throws(
      () => importModelCheckpoint({
        ...valueCheckpoint,
        graph: { ...valueCheckpoint.graph, outputs },
      }),
      /outputs and outputsExplicit/,
    );
  }
  for (const field of ['inputs', 'nodes', 'standaloneTensors']) {
    assert.throws(
      () => importModelCheckpoint({
        ...valueCheckpoint,
        graph: { ...valueCheckpoint.graph, [field]: null },
      }),
      /inputs, nodes, and standaloneTensors/,
    );
  }

  const typed = new ModelBuilder();
  const typedInput = typed.input('tokens', [1], 'int32');
  typed.addNode({
    id: 'typed-copy',
    opType: 'Identity',
    inputs: { input: typedInput },
    outputs: { out: { name: 'typed-output', shape: [1], dtype: 'int32' } },
  });
  const typedCheckpoint = exportModelCheckpoint(typed.build());
  const missingInputDType = structuredClone(typedCheckpoint);
  delete missingInputDType.graph.inputs.tokens.dtype;
  assert.throws(() => importModelCheckpoint(missingInputDType), /explicit shape and dtype/);
  const missingOutputDType = structuredClone(typedCheckpoint);
  delete missingOutputDType.graph.nodes[0].outputs_dtype.out;
  assert.throws(() => importModelCheckpoint(missingOutputDType), /explicit shape and dtype/);
  const missingNodeId = structuredClone(typedCheckpoint);
  delete missingNodeId.graph.nodes[0].id;
  assert.throws(() => importModelCheckpoint(missingNodeId), /explicit id and opType/);

  const automatic = new ModelBuilder();
  const autoInput = automatic.input('auto.x', [1]);
  automatic.addOp('Identity', { input: autoInput }, { out: { name: 'auto.a', shape: [1] } });
  assert.equal(automatic.graph._outputsExplicit, false);
  const autoRestored = importModelCheckpoint(exportModelCheckpoint(automatic.build()));
  assert.equal(autoRestored.graph._outputsExplicit, false);
  autoRestored.model.addOp(
    'Identity',
    { input: autoRestored.graph.getTensor('auto.x') },
    { out: { name: 'auto.b', shape: [1] } },
  );
  assert.deepEqual(autoRestored.graph.outputNames, ['auto.a', 'auto.b']);
});
