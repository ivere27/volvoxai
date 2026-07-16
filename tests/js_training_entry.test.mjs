import test from 'node:test';
import assert from 'node:assert/strict';

import * as core from '../ts/index.js';
import * as full from '../ts/full.js';
import * as trainingCompatibility from '../ts/training.js';

test('training classes live in the full entry rather than the inference entry', () => {
  assert.equal(core.CPUAutograd, undefined);
  assert.equal(core.WebGPUAutograd, undefined);
  assert.equal(core.TrainingVolvoxAI, undefined);
  assert.equal(core.TrainingModelBuilder, undefined);
  assert.equal(core.TrainingGraph, undefined);
  assert.equal(core.buildEncoderDecoderTransformer, undefined);
  assert.equal(full.buildEncoderDecoderTransformer, undefined);
  assert.equal(full.ModelBuilder.prototype.encoderDecoderTransformer, undefined);
  assert.equal(full.VolvoxAI.prototype.teacherForcingBatch, undefined);
  assert.equal(core.initializeTensor, undefined);
  assert.equal(core.VolvoxAI.prototype.trainStep, undefined);
  assert.equal(typeof full.CPUAutograd?.trainStep, 'function');
  assert.equal(typeof full.WebGPUAutograd, 'function');
  assert.equal(full.VolvoxAI, full.TrainingVolvoxAI);
  assert.equal(full.Graph, full.TrainingGraph);
  assert.equal(full.ModelBuilder, full.TrainingModelBuilder);
  assert.equal(trainingCompatibility.VolvoxAI, full.VolvoxAI);
});

test('training graph state and optimizer updates exist only in the full profile', () => {
  const inferenceGraph = new core.Graph();
  const inferenceWeight = inferenceGraph.addWeight(
    'weight', [1], 'float32', Float32Array.of(1),
  );
  assert.equal(Object.hasOwn(inferenceGraph, 'trainingStep'), false);
  assert.equal(Object.hasOwn(inferenceGraph, 'optimizerState'), false);
  assert.equal(Object.hasOwn(inferenceGraph.inspect(), 'trainingStep'), false);
  assert.throws(
    () => inferenceGraph.applyTensorUpdate(inferenceWeight.name, Float32Array.of(1), { mode: 'sgd' }),
    /Unsupported generic tensor update mode/,
  );

  const trainingGraph = new full.Graph();
  const trainingWeight = trainingGraph.addWeight(
    'weight', [1], 'float32', Float32Array.of(1),
  );
  trainingGraph.applyTensorUpdate(trainingWeight.name, Float32Array.of(2), {
    mode: 'sgd', learningRate: 0.25,
  });
  assert.equal(trainingWeight.buffer[0], 0.5);
  assert.equal(trainingGraph.trainingStep, 0);
  assert.equal(trainingGraph.optimizerState, null);
  assert.equal(trainingGraph.inspect().trainingStep, 0);
});

test('full static initialization preserves the training facade subclass', async () => {
  const api = await full.VolvoxAI.init('cpu');
  assert.ok(api instanceof full.TrainingVolvoxAI);
  assert.equal(typeof api.trainStep, 'function');
});

test('the full VolvoxAI facade includes the CPU trainer', async () => {
  const api = new full.VolvoxAI();
  const model = api.createModel();
  const parameter = model.weight('parameter', [1, 2], 'float32', Float32Array.from([0.2, -0.4]));
  const { out } = model.addOp('Identity', { input: parameter }, { out: [1, 2] });
  model.outputs(out);
  const result = await api.trainStep(model.build(), {
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  assert.equal(result.examples, 1);
});
