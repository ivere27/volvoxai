import test from 'node:test';
import assert from 'node:assert/strict';

import { ModelBuilder, importModelCheckpoint } from '../ts/full.js';
import { createCPUTrainingHarness } from './helpers/training_session.mjs';

test('maskedMean pools variable-length token rows and backpropagates through reduction', async () => {
  const builder = new ModelBuilder();
  const tokens = builder.input('tokens', [2, 3, 2]);
  const poolingWeights = builder.input('pooling_weights', [2, 3]);
  const scale = builder.weight('scale', [2], 'float32', Float32Array.of(1, 1));
  const scaled = builder.addOp(
    'Mul',
    { a: tokens, b: scale },
    { out: { name: 'scaled', shape: [2, 3, 2] } },
    {},
    { id: 'scale_tokens' },
  ).out;
  const pooled = builder.maskedMean(scaled, poolingWeights, { name: 'question_mean' });
  const head = builder.weight('head', [2, 2], 'float32', Float32Array.of(0.2, -0.1, 0.3, 0.4));
  const logits = builder.addOp(
    'Linear',
    { input: pooled, weight: head },
    { out: { name: 'logits', shape: [2, 2] } },
    { weight_layout: 'IN_OUT' },
    { id: 'head', wLayout: 'din' },
  ).out;
  builder.outputs(pooled, logits);
  const graph = builder.build();
  const before = new Float32Array(scale.buffer);

  const inputs = {
    tokens: Float32Array.of(1, 2, 3, 4, 100, 100, -5, 2, 7, 8, 4, -3),
    pooling_weights: Float32Array.of(0.5, 0.5, 0, 0, 0, 1),
  };
  const training = createCPUTrainingHarness();
  let result;
  let trained;
  try {
    result = await training.runStep(graph, {
      inputs,
      logitsTensor: logits.name,
      targets: Int32Array.of(0, 1),
      trainableTensors: [scale.name, head.name],
      updateMode: 'sgd',
      optimizer: { learningRate: 1e-2 },
    });
    trained = importModelCheckpoint(await training.exportCheckpoint(graph)).graph;
  } finally {
    await training.close();
  }

  assert.ok(Number.isFinite(result.loss));
  assert.equal(result.examples, 2);
  assert.deepEqual(scale.buffer, before);
  assert.ok(trained.getTensor(scale.name).buffer
    .some((value, index) => value !== before[index]));
});
