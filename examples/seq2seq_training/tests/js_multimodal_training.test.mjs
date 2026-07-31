import test from 'node:test';
import assert from 'node:assert/strict';

import { ModelBuilder, importModelCheckpoint } from '../../../ts/full.js';
import { createCPUTrainingHarness } from '../../../tests/helpers/training_session.mjs';
import { buildEncoderDecoderTransformer } from '../Seq2SeqBuilder.js';

test('NHWC Conv2D + GroupNorm image features train through a multimodal encoder-decoder', async () => {
  const batchSize = 2;
  const builder = new ModelBuilder();
  const image = builder.input('vqa.image', [batchSize, 4, 4, 1], 'float32');
  const convolution = builder.weight('vqa.stem.weight', [3, 3, 1, 4], 'float32', {
    initializer: { type: 'xavierUniform', seed: 31 },
  });
  const convolved = builder.addOp(
    'Conv2D',
    { input: image, weight: convolution },
    { out: { name: 'vqa.stem.conv', shape: [batchSize, 2, 2, 4] } },
    { stride: [2, 2], padding: [1, 1], weight_layout: 'HWIO' },
    { id: 'vqa.stem.conv' },
  ).out;
  const groupScale = builder.weight('vqa.stem.norm.weight', [4], 'float32', new Float32Array(4).fill(1));
  const groupBias = builder.weight('vqa.stem.norm.bias', [4], 'float32', new Float32Array(4));
  const normalized = builder.groupNorm(convolved, groupScale, groupBias, {
    numGroups: 2,
    name: 'vqa.stem.norm',
  });
  const activated = builder.addOp(
    'SiLU',
    { input: normalized },
    { out: { name: 'vqa.stem.activation', shape: [...normalized.shape] } },
    {},
    { id: 'vqa.stem.activation' },
  ).out;
  const imageTokens = builder.addOp(
    'Reshape',
    { input: activated },
    { out: { name: 'vqa.image_tokens', shape: [batchSize, 4, 4] } },
    {},
    { id: 'vqa.image_tokens' },
  ).out;

  const model = buildEncoderDecoderTransformer(builder, {
    name: 'vqa.text',
    batchSize,
    sourceLength: 2,
    targetLength: 2,
    vocabSize: 6,
    dModel: 4,
    numHeads: 2,
    dFF: 8,
    encoderLayers: 1,
    decoderLayers: 1,
    sourceFeatures: imageTokens,
    additionalTrainableTensors: [convolution, groupScale, groupBias],
    tieSourceTargetEmbeddings: true,
    tieTargetEmbeddings: true,
    lmHeadBias: true,
    finalEncoderNorm: false,
    dropout: 0.1,
    dropoutSeed: 73,
    seed: 32,
    padTokenId: 0,
    bosTokenId: 1,
  });
  const graph = builder.build();
  assert.equal(graph.validate().valid, true);
  assert.deepEqual(model.memory.shape, [batchSize, 6, 4]);
  assert.ok(graph.nodes.some((node) => node.opType === 'Dropout'));
  assert.ok(graph.nodes.filter((node) => node.opType === 'SDPA' || node.opType === 'CrossSDPA')
    .every((node) => node.params.dropout === 0.1));

  const before = new Float32Array(convolution.buffer);
  const teacher = model.teacherForcing(
    [[2, 0], [3, 4]],
    [[4, 0], [5, 3]],
  );
  teacher.inputs[image.name] = Float32Array.from({ length: batchSize * 4 * 4 }, (_, index) =>
    ((index * 7) % 19 - 9) / 10);
  const training = createCPUTrainingHarness();
  let result;
  let trained;
  try {
    result = await training.runStep(graph, {
      ...teacher,
      updateMode: 'adamw',
      optimizer: { learningRate: 2e-3, maxGradNorm: 1 },
    });
    trained = importModelCheckpoint(await training.exportCheckpoint(graph)).graph;
  } finally {
    await training.close();
  }

  assert.ok(Number.isFinite(result.loss));
  assert.equal(result.examples, 3);
  assert.equal(graph.trainingStep, 0, 'Trainer must not mutate caller-owned graph state');
  assert.deepEqual(convolution.buffer, before);
  assert.equal(trained.trainingStep, 1);
  assert.ok(trained.getTensor(convolution.name).buffer
    .some((value, index) => value !== before[index]));
  assert.equal(result.updatedTensorNames.length, model.trainableTensors.length);
});
