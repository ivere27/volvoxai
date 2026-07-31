import test from 'node:test';
import assert from 'node:assert/strict';

import { ModelBuilder, importModelCheckpoint } from '../ts/full.js';
import { createCPUTrainingHarness } from './helpers/training_session.mjs';

function buildTwoHeadGraph() {
  const builder = new ModelBuilder();
  const input = builder.input('x', [1, 2]);
  const shared = builder.weight('shared', [2, 2], 'float32', Float32Array.of(0.2, -0.3, 0.4, 0.1));
  const hidden = builder.addOp(
    'Linear', { input, weight: shared }, { out: { name: 'hidden', shape: [1, 2] } },
    { weight_layout: 'IN_OUT' }, { id: 'shared', wLayout: 'din' },
  ).out;
  const lmWeight = builder.weight('lm.weight', [2, 3], 'float32', Float32Array.of(0.1, 0.2, -0.2, 0.4, -0.1, 0.3));
  const lm = builder.addOp(
    'Linear', { input: hidden, weight: lmWeight }, { out: { name: 'lm.logits', shape: [1, 3] } },
    { weight_layout: 'IN_OUT' }, { id: 'lm', wLayout: 'din' },
  ).out;
  const routerWeight = builder.weight('router.weight', [2, 2], 'float32', Float32Array.of(-0.2, 0.3, 0.5, -0.4));
  const router = builder.addOp(
    'Linear', { input: hidden, weight: routerWeight }, { out: { name: 'router.logits', shape: [1, 2] } },
    { weight_layout: 'IN_OUT' }, { id: 'router', wLayout: 'din' },
  ).out;
  builder.outputs(lm, router);
  return {
    graph: builder.build(),
    input,
    lm,
    router,
    trainableTensors: [shared.name, lmWeight.name, routerWeight.name],
  };
}

function objectives(lm, router, lmTarget, routerTarget, normalizer = null) {
  return [
    {
      name: 'language', logitsTensor: lm.name, targets: [lmTarget], weight: 1,
      ...(normalizer == null ? {} : { normalizer }),
    },
    {
      name: 'router', logitsTensor: router.name, targets: [routerTarget], weight: 0.1,
      ...(normalizer == null ? {} : { normalizer }),
    },
  ];
}

function assertArrayClose(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `index ${index}: ${actual[index]} != ${expected[index]}`);
  }
}

test('one train step combines independently normalized weighted CE objectives', async () => {
  const model = buildTwoHeadGraph();
  const training = createCPUTrainingHarness();
  let result;
  try {
    result = await training.runStep(model.graph, {
      inputs: { x: Float32Array.of(1, -0.5) },
      losses: objectives(model.lm, model.router, 2, 1),
      trainableTensors: model.trainableTensors,
      updateMode: 'sgd',
      optimizer: { learningRate: 0 },
    });
  } finally {
    await training.close();
  }

  assert.equal(result.losses.length, 2);
  assert.ok(Math.abs(result.loss - result.losses[0].loss - result.losses[1].loss) < 1e-7);
  assert.equal(result.examples, 2);
  assert.ok(result.gradients.get('router.weight').some((value) => value !== 0));
  assert.ok(result.gradients.get('lm.weight').some((value) => value !== 0));
});

test('full-window normalizers make heterogeneous multi-loss accumulation exact', async () => {
  const training = createCPUTrainingHarness();
  const accumulated = buildTwoHeadGraph();
  const first = await training.runStep(accumulated.graph, {
    inputs: { x: Float32Array.of(1, -1) },
    losses: objectives(accumulated.lm, accumulated.router, 2, 1, 2),
    trainableTensors: accumulated.trainableTensors,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    gradientAccumulationSteps: 2,
  });
  assert.equal(first.accumulating, true);
  assert.equal(accumulated.graph.trainingStep, 0);
  await assert.rejects(
    () => training.exportCheckpoint(accumulated.graph),
    /gradient accumulation is pending/,
  );

  const second = await training.runStep(accumulated.graph, {
    inputs: { x: Float32Array.of(-0.5, 0.25) },
    losses: objectives(accumulated.lm, accumulated.router, 0, 0, 2),
    trainableTensors: accumulated.trainableTensors,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
    gradientAccumulationSteps: 2,
  });
  assert.equal(second.accumulating, false);
  const accumulatedCheckpoint = await training.exportCheckpoint(accumulated.graph);
  assert.equal(importModelCheckpoint(accumulatedCheckpoint).graph.trainingStep, 1);
  assert.equal(accumulated.graph.trainingStep, 0, 'Trainer must not mutate caller-owned graph state');
  assert.equal(second.losses[0].examples, 2);
  assert.equal(second.losses[1].examples, 2);
  assert.ok(Math.abs(second.loss - second.losses[0].loss - second.losses[1].loss) < 1e-7);

  const microA = buildTwoHeadGraph();
  const resultA = await training.runStep(microA.graph, {
    inputs: { x: Float32Array.of(1, -1) },
    losses: objectives(microA.lm, microA.router, 2, 1, 2),
    trainableTensors: microA.trainableTensors,
    updateMode: 'sgd', optimizer: { learningRate: 0 },
  });
  const microB = buildTwoHeadGraph();
  const resultB = await training.runStep(microB.graph, {
    inputs: { x: Float32Array.of(-0.5, 0.25) },
    losses: objectives(microB.lm, microB.router, 0, 0, 2),
    trainableTensors: microB.trainableTensors,
    updateMode: 'sgd', optimizer: { learningRate: 0 },
  });
  for (const name of accumulated.trainableTensors) {
    const expected = Float32Array.from(resultA.gradients.get(name),
      (value, index) => value + resultB.gradients.get(name)[index]);
    assertArrayClose(second.gradients.get(name), expected);
  }

  const invalid = buildTwoHeadGraph();
  await assert.rejects(() => training.runStep(invalid.graph, {
    inputs: { x: Float32Array.of(1, 1) },
    losses: objectives(invalid.lm, invalid.router, 1, 0),
    trainableTensors: invalid.trainableTensors,
    gradientAccumulationSteps: 2,
  }), /full-window normalizer/);
  await training.close();
});
