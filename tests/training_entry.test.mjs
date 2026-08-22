import test from 'node:test';
import assert from 'node:assert/strict';

import * as inference from '../ts/index.js';
import * as full from '../ts/full.js';
import { Trainer as StrictWasmTrainer } from '../ts/training/WasmTrainer.js';
import {
  dropoutContext,
  dropoutMultiplier,
} from '../ts/ops/dropout.js';

const INITIAL_PARAMETER = Float32Array.of(0.2, -0.4, 0.1, 0.3);

function makeTrainingSnapshot({ minimum = 1, maximum = 4, dropout = false } = {}) {
  const projectedName = dropout ? 'projected' : 'logits';
  const nodes = [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'parameter' },
    outputs: {
      out: { tensor: projectedName, dtype: 'float32', shape: ['B', 2] },
    },
    params: {},
  }];
  if (dropout) {
    nodes.push({
      id: 'dropout',
      opType: 'Dropout',
      inputs: { input: projectedName },
      outputs: {
        out: { tensor: 'logits', dtype: 'float32', shape: ['B', 2] },
      },
      params: { p: 0.5, seed: 9 },
    });
  }
  const builder = new full.ModelBuilder({
    dimensions: { B: { min: minimum, max: maximum } },
    inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
    weights: [{ name: 'parameter', dtype: 'float32', shape: [2, 2] }],
    nodes,
    outputs: ['logits'],
  });
  return full.Model.capture({
    graph: builder.snapshot(),
    weights: {
      parameter: {
        name: 'parameter',
        dtype: 'float32',
        shape: [2, 2],
        data: INITIAL_PARAMETER,
      },
    },
  });
}

function shapedBatch(batchSize) {
  const values = new Float32Array(batchSize * 2);
  for (let index = 0; index < values.length; index++) {
    values[index] = Math.fround(((index * 3) % 11 - 5) / 7);
  }
  return {
    inputs: { x: { data: values, shape: [batchSize, 2] } },
    targets: Array.from({ length: batchSize }, (_, index) => index % 2),
  };
}

function stepOptions(batchSize, overrides = {}) {
  return {
    ...shapedBatch(batchSize),
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.05 },
    ...overrides,
  };
}

function closeNumber(actual, expected, tolerance = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance * (1 + Math.abs(actual) + Math.abs(expected)),
    `${actual} != ${expected}`,
  );
}

function closeArray(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index++) {
    closeNumber(actual[index], expected[index], tolerance);
  }
}

test('the v1 entries expose logical authoring while concrete/training internals stay private', () => {
  for (const name of [
    'RuntimeGraph',
    'Tensor',
    'RuntimeGraphBuilder',
    'createModel',
    'CPUAutograd',
    'WebGPUAutograd',
    'WasmAutograd',
    'WasmTrainer',
    'TrainingGraph',
    'TrainingModelBuilder',
    'ensureTrainingGraphState',
    'applyTrainingTensorUpdate',
    'accumulateGradients',
    'resolveTrainingOptimizer',
    'runtimeError',
  ]) {
    assert.equal(inference[name], undefined, `${name} leaked from inference`);
    assert.equal(full[name], undefined, `${name} leaked from full`);
  }

  assert.equal(typeof inference.ModelBuilder, 'function');
  assert.equal(typeof inference.Model, 'function');
  assert.equal(inference.Trainer, undefined);
  assert.equal(inference.PTQCalibrator, undefined);
  assert.equal(typeof full.ModelBuilder, 'function');
  assert.equal(typeof full.Model, 'function');
  assert.equal(typeof full.Trainer.create, 'function');
  assert.equal(typeof full.VolvoxAI.createTrainer, 'function');
  assert.equal(typeof full.PTQCalibrator, 'function');
  assert.equal(typeof full.exportModelCheckpoint, 'function');
  assert.equal(typeof full.initializeTensor, 'function');
});

test('Trainer accepts only immutable logical snapshots and closed handles fail predictably', async () => {
  for (const create of [
    () => full.Trainer.create(null),
    () => StrictWasmTrainer.create(null),
  ]) {
    await assert.rejects(create(), (error) => {
      assert.equal(error.code, 'INVALID_ARGUMENT');
      assert.equal(error.phase, 'initialization');
      return true;
    });
  }

  const snapshot = makeTrainingSnapshot();
  await assert.rejects(
    full.Trainer.create(snapshot, { backend: 'remote' }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'selection',
  );
  await assert.rejects(
    full.Trainer.create(snapshot, { backend: 'webgpu', device: null }),
    (error) => error.code === 'BACKEND_UNAVAILABLE' &&
      error.phase === 'initialization' && error.backend === 'webgpu',
  );

  await assert.rejects(
    full.Trainer.create(snapshot, { backend: 'cpu' }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'selection',
  );
  const trainer = await full.Trainer.create(snapshot, { backend: 'cpu-js' });
  assert.equal(trainer.backend, 'cpu-js');
  await trainer.close();
  assert.throws(() => trainer.inspectShapeState(), (error) =>
    error.code === 'HANDLE_DISPOSED' && error.phase === 'lifecycle');
  await assert.rejects(trainer.exportCheckpoint(), (error) =>
    error.code === 'HANDLE_DISPOSED' && error.phase === 'lifecycle');
});

test('dynamic CPU microbatches match equivalent static plans', async () => {
  for (const batchSize of [1, 3]) {
    const dynamic = await full.Trainer.create(makeTrainingSnapshot(), { backend: 'cpu-js' });
    const fixed = await full.Trainer.create(makeTrainingSnapshot({
      minimum: batchSize,
      maximum: batchSize,
    }), { backend: 'cpu-js' });
    try {
      const options = stepOptions(batchSize, { optimizer: { learningRate: 0.025 } });
      const dynamicResult = await dynamic.trainStep(options);
      const fixedResult = await fixed.trainStep(options);
      closeNumber(dynamicResult.loss, fixedResult.loss);
      closeArray(
        dynamicResult.gradients.get('parameter'),
        fixedResult.gradients.get('parameter'),
      );
      assert.deepEqual(dynamicResult.activationShapes.logits, [batchSize, 2]);
      assert.deepEqual(dynamicResult.activationGradientShapes.logits, [batchSize, 2]);
      const dynamicSuccessor = await dynamic.commit();
      const fixedSuccessor = await fixed.commit();
      closeArray(
        dynamicSuccessor.copyWeightData('parameter'),
        fixedSuccessor.copyWeightData('parameter'),
      );
    } finally {
      await Promise.all([dynamic.close(), fixed.close()]);
    }
  }
});

test('different legal microbatch shapes sum to one fixed combined loss and gradient oracle', async () => {
  const totalExamples = 4;
  const firstBatch = shapedBatch(1);
  const secondBatch = shapedBatch(3);
  const combinedValues = new Float32Array(totalExamples * 2);
  combinedValues.set(firstBatch.inputs.x.data, 0);
  combinedValues.set(secondBatch.inputs.x.data, firstBatch.inputs.x.data.length);
  const combinedTargets = [...firstBatch.targets, ...secondBatch.targets];
  const snapshot = makeTrainingSnapshot({ minimum: 1, maximum: totalExamples });
  const first = await full.Trainer.create(snapshot, { backend: 'cpu-js' });
  const second = await full.Trainer.create(snapshot, { backend: 'cpu-js' });
  const combined = await full.Trainer.create(makeTrainingSnapshot({
    minimum: totalExamples,
    maximum: totalExamples,
  }), { backend: 'cpu-js' });
  const options = (batch, targets) => ({
    inputs: { x: batch },
    losses: [{
      name: 'combined-window',
      logitsTensor: 'logits',
      targets,
      normalizer: totalExamples,
    }],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  try {
    const firstResult = await first.trainStep(options(
      firstBatch.inputs.x,
      firstBatch.targets,
    ));
    const secondResult = await second.trainStep(options(
      secondBatch.inputs.x,
      secondBatch.targets,
    ));
    const combinedResult = await combined.trainStep(options(
      { data: combinedValues, shape: [totalExamples, 2] },
      combinedTargets,
    ));

    closeNumber(
      firstResult.loss + secondResult.loss,
      combinedResult.loss,
      2e-6,
    );
    const expectedGradient = Float32Array.from(
      firstResult.gradients.get('parameter'),
      (value, index) => value + secondResult.gradients.get('parameter')[index],
    );
    closeArray(
      expectedGradient,
      combinedResult.gradients.get('parameter'),
      2e-6,
    );
    assert.notEqual(firstResult.shapeSignature, secondResult.shapeSignature);
    assert.deepEqual(combinedResult.activationShapes.logits, [totalExamples, 2]);
  } finally {
    await Promise.all([first.close(), second.close(), combined.close()]);
  }
});

test('shape plans vary while parameter, gradient, and optimizer storage stay fixed', async () => {
  const trainer = await full.Trainer.create(makeTrainingSnapshot(), {
    backend: 'cpu-js',
    planCacheEntries: 2,
  });
  const seen = [];
  try {
    for (const batchSize of [1, 4, 1]) {
      const result = await trainer.trainStep(stepOptions(batchSize, {
        updateMode: 'adamw',
        optimizer: { learningRate: 0.01 },
      }));
      seen.push({ result, inspection: trainer.inspectShapeState() });
    }
    assert.notEqual(seen[0].result.shapeSignature, seen[1].result.shapeSignature);
    assert.equal(seen[0].result.shapeSignature, seen[2].result.shapeSignature);
    for (const { inspection } of seen) {
      assert.equal(inspection.parameterBytes, INITIAL_PARAMETER.byteLength);
      assert.equal(inspection.parameterGradientBytes, INITIAL_PARAMETER.byteLength);
      assert.equal(inspection.optimizerBytes, INITIAL_PARAMETER.byteLength * 2);
    }
    assert.equal(seen[2].inspection.planCacheEntries, 2);
    assert.equal(seen[2].inspection.planCacheMisses, 2);
    assert.equal(seen[2].inspection.planCacheHits, 1);
    assert.ok(seen[1].inspection.activationCapacityBytes >= seen[0].inspection.activationCapacityBytes);
    assert.equal(
      seen[2].inspection.activationCapacityBytes,
      seen[1].inspection.activationCapacityBytes,
      'shrinking a microbatch must reuse the one growable activation pool',
    );
  } finally {
    await trainer.close();
  }
});

test('accumulation rejects a shape change and rollback discards the complete private window', async () => {
  const snapshot = makeTrainingSnapshot();
  const trainer = await full.Trainer.create(snapshot, { backend: 'cpu-js' });
  const baseline = await trainer.exportCheckpoint();
  try {
    const first = await trainer.trainStep(stepOptions(1, {
      gradientAccumulationSteps: 2,
    }));
    assert.equal(first.accumulating, true);
    assert.equal(trainer.getGradientAccumulationState().pending, true);
    const firstSignature = trainer.inspectShapeState().currentShapeSignature;

    await assert.rejects(
      trainer.trainStep(stepOptions(2, { gradientAccumulationSteps: 2 })),
      (error) => error.code === 'INVALID_ARGUMENT' && /exact activation shape signature/i.test(error.message),
    );
    assert.equal(trainer.inspectShapeState().currentShapeSignature, firstSignature);
    assert.equal(trainer.getGradientAccumulationState().microbatches, 1);
    await assert.rejects(trainer.commit(), /accumulation/i);
    await assert.rejects(trainer.exportCheckpoint(), /accumulation/i);

    await trainer.rollback();
    assert.equal(trainer.getGradientAccumulationState().pending, false);
    assert.equal(trainer.inspectShapeState().currentShapeSignature, null);
    assert.equal(trainer.hasUncommittedUpdates, false);
    const restored = await trainer.exportCheckpoint();
    assert.equal(restored.trainingStep, baseline.trainingStep);
    assert.deepEqual(
      new Uint8Array(restored.parameters),
      new Uint8Array(baseline.parameters),
    );

    const changedShape = await trainer.trainStep(stepOptions(2));
    assert.equal(changedShape.accumulating, false);
    assert.deepEqual(changedShape.activationShapes.logits, [2, 2]);
  } finally {
    await trainer.close();
  }
});

async function dropoutRun(batchSize, dropout) {
  const trainer = await full.Trainer.create(makeTrainingSnapshot({ dropout: true }), {
    backend: 'cpu-js',
  });
  try {
    const result = await trainer.trainStep(stepOptions(batchSize, {
      optimizer: { learningRate: 0 },
      dropout,
    }));
    return {
      loss: result.loss,
      gradient: new Float32Array(result.gradients.get('parameter')),
      signature: result.shapeSignature,
    };
  } finally {
    await trainer.close();
  }
}

test('Dropout is reproducible for seed+counter+shape and changes when either component changes', async () => {
  const rng = { seed: 17, counter: 23 };
  const first = await dropoutRun(4, rng);
  const repeated = await dropoutRun(4, rng);
  closeNumber(first.loss, repeated.loss, 0);
  assert.deepEqual(first.gradient, repeated.gradient);
  assert.equal(first.signature, repeated.signature);

  const nextCounter = await dropoutRun(4, { ...rng, counter: rng.counter + 1 });
  assert.notDeepEqual(nextCounter.gradient, first.gradient);

  const otherShape = await dropoutRun(1, rng);
  assert.notEqual(otherShape.signature, first.signature);
  const dropoutNode = { id: 'dropout', params: { p: 0.5, seed: 9 } };
  const firstContext = dropoutContext({ ...rng, shapeSignature: first.signature });
  const otherContext = dropoutContext({ ...rng, shapeSignature: otherShape.signature });
  assert.notEqual(firstContext.shapeHash, otherContext.shapeHash);
  assert.ok(
    Array.from({ length: 32 }, (_, index) => index).some((index) =>
      dropoutMultiplier(dropoutNode, firstContext, index) !==
      dropoutMultiplier(dropoutNode, otherContext, index)),
    'the concrete shape signature must participate in the deterministic Dropout stream',
  );
});

test('commit returns one immutable logical successor and leaves its source revision untouched', async () => {
  const source = makeTrainingSnapshot();
  const sourceBytes = source.copyWeightBytes('parameter');
  const trainer = await full.VolvoxAI.createTrainer(source, { backend: 'cpu-js' });
  try {
    const update = await trainer.trainStep(stepOptions(2));
    assert.deepEqual(update.updatedTensorNames, ['parameter']);
    assert.equal(trainer.hasUncommittedUpdates, true);
    const successor = await trainer.commit();
    assert.ok(successor instanceof full.Model);
    assert.notEqual(successor, source);
    assert.equal(successor.definitionId, source.definitionId);
    assert.equal(successor.definitionFingerprint, source.definitionFingerprint);
    assert.equal(successor.weightRevision, source.weightRevision + 1);
    assert.equal(trainer.snapshot, successor);
    assert.equal(trainer.hasUncommittedUpdates, false);
    assert.deepEqual(source.copyWeightBytes('parameter'), sourceBytes);
    assert.notDeepEqual(successor.copyWeightBytes('parameter'), sourceBytes);
    await assert.rejects(trainer.commit(), (error) => error.code === 'INVALID_ARGUMENT');
  } finally {
    await trainer.close();
  }
});

test('logical checkpoint v1 preserves bounds/fingerprint and rejects legacy or mismatched definitions', async () => {
  const source = makeTrainingSnapshot({ minimum: 1, maximum: 4 });
  const trainer = await full.Trainer.create(source, { backend: 'cpu-js' });
  let checkpoint;
  try {
    await trainer.trainStep(stepOptions(2, {
      updateMode: 'adamw',
      optimizer: { learningRate: 0.01 },
    }));
    checkpoint = await trainer.exportCheckpoint();
  } finally {
    await trainer.close();
  }

  assert.equal(checkpoint.format, 'volvox.training-checkpoint/v1');
  assert.equal(checkpoint.logicalFingerprint, source.definitionFingerprint);
  assert.deepEqual(checkpoint.logicalGraph.dimensions.B, {
    min: 1,
    max: 4,
    multiple_of: 1,
  });
  assert.deepEqual(checkpoint.logicalGraph.inputs.x.shape, ['B', 2]);
  assert.ok(checkpoint.parameters instanceof ArrayBuffer);
  assert.ok(checkpoint.optimizer instanceof ArrayBuffer);
  assert.deepEqual(checkpoint.parameterDescriptors[0].shape, [2, 2]);
  assert.deepEqual(checkpoint.optimizerEntries[0].shape, [2, 2]);

  const imported = full.importModelCheckpoint(checkpoint);
  assert.ok(imported.snapshot instanceof full.Model);
  assert.equal(imported.snapshot.definitionFingerprint, source.definitionFingerprint);
  assert.deepEqual(imported.snapshot.graph.dimensions.B, source.graph.dimensions.B);
  const resumed = await full.Trainer.create(source, { backend: 'cpu-js', checkpoint });
  try {
    assert.equal(resumed.trainingStep, 1);
    const maximum = await resumed.trainStep(stepOptions(4));
    assert.deepEqual(maximum.activationShapes.logits, [4, 2]);
  } finally {
    await resumed.close();
  }

  assert.throws(
    () => full.importModelCheckpoint({ format: 'volvox.checkpoint.v1' }),
    /not loadable by dynamic training v1/i,
  );
  const tampered = {
    ...checkpoint,
    logicalGraph: {
      ...checkpoint.logicalGraph,
      dimensions: { B: { min: 1, max: 5, multiple_of: 1 } },
    },
  };
  assert.throws(() => full.importModelCheckpoint(tampered), /fingerprint.*bounds/i);

  const differentBounds = makeTrainingSnapshot({ minimum: 1, maximum: 3 });
  await assert.rejects(
    full.Trainer.create(differentBounds, { backend: 'cpu-js', checkpoint }),
    (error) => error.code === 'INVALID_ARGUMENT' && /fingerprint|symbolic constraints/i.test(error.message),
  );
});
