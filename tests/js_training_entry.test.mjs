import test from 'node:test';
import assert from 'node:assert/strict';

import * as core from '../ts/index.js';
import * as full from '../ts/full.js';
import { Trainer as StrictWasmTrainer } from '../ts/training/WasmTrainer.js';

test('training classes live in the full entry rather than the inference entry', () => {
  for (const implementationName of [
    'CPUAutograd',
    'WebGPUAutograd',
    'WasmAutograd',
    'WasmTrainer',
    'WasmQuantizedLoRATrainer',
    'WasmPTQ',
    'WasmPTQObserver',
    'createWasmPTQ',
    'TrainingModelBuilder',
    'TrainingGraph',
    'ensureTrainingGraphState',
    'applyTrainingTensorUpdate',
    'accumulateGradients',
    'resolveTrainingOptimizer',
    'normalizeCrossEntropyLosses',
    'crossEntropyGradient',
    'addGradient',
    'AdapterManager',
    'ModelSnapshot',
    'runtimeError',
    'parseStrictJSON',
  ]) {
    assert.equal(core[implementationName], undefined);
    assert.equal(full[implementationName], undefined);
  }
  assert.equal(core.Trainer, undefined);
  assert.equal(core.buildEncoderDecoderTransformer, undefined);
  assert.equal(full.buildEncoderDecoderTransformer, undefined);
  assert.equal(full.ModelBuilder.prototype.encoderDecoderTransformer, undefined);
  assert.equal(core.initializeTensor, undefined);
  assert.equal(typeof core.VolvoxAI.createRuntime, 'function');
  assert.equal(core.VolvoxAI.trainStep, undefined);
  assert.equal(typeof full.VolvoxAI.createRuntime, 'function');
  assert.equal(typeof full.VolvoxAI.createTrainer, 'function');
  assert.equal(typeof full.Trainer.create, 'function');
  assert.notEqual(full.Graph, core.Graph);
  assert.notEqual(full.ModelBuilder, core.ModelBuilder);
  assert.equal(typeof full.PTQCalibrator, 'function');
  assert.equal(typeof full.exportModelCheckpoint, 'function');
  assert.equal(typeof full.initializeTensor, 'function');
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

test('Trainer construction and closed handles expose stable typed failures', async () => {
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

  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const output = graph.addWeight('logits', [1, 2], 'float32', Float32Array.of(0, 0));
  graph.setOutputs(output);
  const model = runtime.createModel(graph);
  await assert.rejects(
    full.Trainer.create(model, { backend: 'remote' }),
    (error) => error.code === 'INVALID_ARGUMENT' && error.phase === 'selection',
  );
  await assert.rejects(
    full.Trainer.create(model, { backend: 'webgpu', device: null }),
    (error) => error.code === 'BACKEND_UNAVAILABLE' &&
      error.phase === 'initialization' && error.backend === 'webgpu',
  );

  const trainer = await full.Trainer.create(model, { backend: 'cpu' });
  await trainer.close();
  assert.throws(() => trainer.getGradientAccumulationState(), (error) =>
    error.code === 'HANDLE_DISPOSED' && error.phase === 'lifecycle');
  await assert.rejects(trainer.exportCheckpoint(), (error) =>
    error.code === 'HANDLE_DISPOSED' && error.phase === 'lifecycle');

  await model.close();
  await runtime.close();
});

test('the full profile creates a retained CPU Trainer over a Model', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const builder = new full.ModelBuilder();
  const parameter = builder.weight('parameter', [1, 2], 'float32', Float32Array.from([0.2, -0.4]));
  const { out } = builder.addOp('Identity', { input: parameter }, { out: [1, 2] });
  builder.outputs(out);
  const graph = builder.build();
  const model = runtime.createModel(graph);
  const trainer = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const result = await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  assert.equal(result.examples, 1);
  const modelClose = model.close();
  let modelClosed = false;
  modelClose.then(() => { modelClosed = true; });
  await Promise.resolve();
  assert.equal(modelClosed, false, 'Model.close should drain its retained Trainer');
  await trainer.close();
  await modelClose;
  await runtime.close();
});

test('Trainer publication preserves definition identity and leaves old contexts pinned', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const parameter = graph.addWeight(
    'parameter', [1, 2], 'float32', Float32Array.from([0.2, -0.4]),
  );
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.setOutputs(out);
  const model = runtime.createModel(graph);
  const definitionId = model.definitionId;
  const originalRevision = model.weightRevisionId;
  const oldCompiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const oldContext = await oldCompiled.createContext();
  const before = await oldContext.execute({});

  const trainer = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const update = await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.1 },
  });
  assert.deepEqual(update.updatedTensorNames, ['parameter']);
  assert.equal(Object.hasOwn(update, 'updatedTensors'), false);
  assert.deepEqual(parameter.buffer, Float32Array.from([0.2, -0.4]));
  assert.equal(model.definitionId, definitionId);
  assert.equal(model.weightRevisionId, originalRevision,
    'a training step must remain private before commit');
  assert.equal(trainer.hasUncommittedUpdates, true);
  const committed = await trainer.commit();
  assert.equal(committed.definitionId, definitionId);
  assert.equal(committed.weightRevisionId, model.weightRevisionId);
  assert.equal(trainer.hasUncommittedUpdates, false);
  assert.notEqual(model.weightRevisionId, originalRevision);

  const newCompiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const newContext = await newCompiled.createContext();
  const after = await newContext.execute({});
  const oldAgain = await oldContext.execute({});
  assert.deepEqual(
    [...await before.output(out.name).read()],
    [...await oldAgain.output(out.name).read()],
  );
  assert.notDeepEqual(
    [...await after.output(out.name).read()],
    [...await before.output(out.name).read()],
  );

  await Promise.all([before.close(), oldAgain.close(), after.close()]);
  await Promise.all([oldContext.close(), newContext.close(), trainer.close()]);
  await Promise.all([oldCompiled.close(), newCompiled.close()]);
  await model.close();
  await runtime.close();
});

test('Trainer commits only applied updates and rejects unfinished accumulation', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const parameter = graph.addWeight(
    'parameter', [1, 2], 'float32', Float32Array.from([0.3, -0.2]),
  );
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.setOutputs(out);
  const model = runtime.createModel(graph);
  const trainer = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const initialRevision = model.weightRevisionId;
  const beforeFailure = await trainer.exportCheckpoint();

  await assert.rejects(
    trainer.trainStep({
      targets: [99],
      trainableTensors: ['parameter'],
      updateMode: 'sgd',
      optimizer: { learningRate: 0.1 },
    }),
    /target/i,
  );
  assert.equal(model.weightRevisionId, initialRevision);
  const afterFailure = await trainer.exportCheckpoint();
  assert.equal(afterFailure.trainingStep, beforeFailure.trainingStep);
  assert.deepEqual(
    new Uint8Array(afterFailure.weights),
    new Uint8Array(beforeFailure.weights),
    'a failed step must leave the Trainer working revision unchanged',
  );

  const accumulating = await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.1 },
    gradientAccumulationSteps: 2,
  });
  assert.equal(accumulating.updatedTensorNames.length, 0);
  assert.equal(model.weightRevisionId, initialRevision);
  await assert.rejects(trainer.commit(), (error) => {
    assert.equal(error.code, 'INVALID_ARGUMENT');
    assert.match(error.message, /accumulation/i);
    return true;
  });

  const applied = await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.1 },
    gradientAccumulationSteps: 2,
  });
  assert.deepEqual(applied.updatedTensorNames, ['parameter']);
  assert.equal(model.weightRevisionId, initialRevision);
  await trainer.commit();
  assert.notEqual(model.weightRevisionId, initialRevision);

  await trainer.close();
  await model.close();
  await runtime.close();
});

test('Trainer exports and resumes its private working revision', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const parameter = graph.addWeight(
    'parameter', [1, 2], 'float32', Float32Array.from([0.4, -0.1]),
  );
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.setOutputs(out);
  const model = runtime.createModel(graph);
  const trainer = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'adamw',
    optimizer: { learningRate: 0.1 },
  });
  const checkpoint = await trainer.exportCheckpoint();
  assert.equal(checkpoint.trainingStep, 1);
  assert.deepEqual(parameter.buffer, Float32Array.from([0.4, -0.1]));
  await trainer.close();
  await model.close();

  const restored = full.importModelCheckpoint(checkpoint);
  const resumedModel = runtime.createModel(restored.graph);
  const resumed = await full.VolvoxAI.createTrainer(resumedModel, {
    backend: 'cpu',
    checkpoint,
  });
  assert.equal(resumed.trainingStep, 1);
  const second = await resumed.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
  });
  assert.equal(resumed.trainingStep, 2);
  assert.deepEqual(second.updatedTensorNames, ['parameter']);
  assert.equal(resumed.hasUncommittedUpdates, true);
  await resumed.commit();
  assert.equal(resumed.hasUncommittedUpdates, false);

  await resumed.close();
  await resumedModel.close();
  await runtime.close();
});

test('Trainer constructors reject checkpoints from a different Model definition', async () => {
  const retainedGraph = new full.Graph();
  const retainedWeight = retainedGraph.addWeight(
    'retained', [1], 'float32', Float32Array.of(3),
  );
  const retainedOutput = retainedGraph.addOp('Identity', { input: retainedWeight }, {
    out: { name: 'retained_output', shape: [1] },
  }).out;
  retainedGraph.setOutputs(retainedOutput);

  const unrelatedGraph = new full.Graph();
  const unrelatedWeight = unrelatedGraph.addWeight(
    'unrelated', [2], 'float32', Float32Array.of(7, 8),
  );
  const unrelatedOutput = unrelatedGraph.addOp('Identity', { input: unrelatedWeight }, {
    out: { name: 'unrelated_output', shape: [2] },
  }).out;
  unrelatedGraph.setOutputs(unrelatedOutput);
  const unrelatedCheckpoint = full.exportModelCheckpoint(unrelatedGraph);

  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(retainedGraph);
  const definitionId = model.definitionId;
  const weightRevisionId = model.weightRevisionId;
  const assertDefinitionMismatch = (error) => {
    assert.equal(error.code, 'INVALID_ARGUMENT');
    assert.match(error.message, /checkpoint definition.*retained Model/i);
    return true;
  };

  await assert.rejects(
    full.Trainer.create(model, { backend: 'cpu', checkpoint: unrelatedCheckpoint }),
    assertDefinitionMismatch,
  );
  await assert.rejects(
    StrictWasmTrainer.create(model, {
      checkpoint: unrelatedCheckpoint,
      wasmUrl: new URL('missing-training-runtime.wasm', import.meta.url),
    }),
    assertDefinitionMismatch,
  );
  assert.equal(model.definitionId, definitionId);
  assert.equal(model.weightRevisionId, weightRevisionId);

  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const result = await context.execute({});
  assert.deepEqual(await result.output('retained_output').read(), Float32Array.of(3));
  assert.throws(() => result.output('unrelated_output'), (error) => error.code === 'INVALID_ARGUMENT');

  await result.close();
  await context.close();
  await compiled.close();
  await model.close();
  await runtime.close();
});

test('concurrent Trainers reject stale successor publication', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const parameter = graph.addWeight(
    'parameter', [1, 2], 'float32', Float32Array.from([0.2, -0.3]),
  );
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.setOutputs(out);
  const model = runtime.createModel(graph);
  const initialRevision = model.weightRevisionId;
  const first = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const stale = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const options = {
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0.1 },
  };

  const staleBefore = await stale.exportCheckpoint();
  await first.trainStep(options);
  assert.equal(model.weightRevisionId, initialRevision);
  await first.commit();
  const publishedRevision = model.weightRevisionId;
  await stale.trainStep(options);
  const staleAfterStep = await stale.exportCheckpoint();
  await assert.rejects(stale.commit(), (error) => {
    assert.equal(error.code, 'EXECUTION_FAILED');
    assert.match(error.message, /revision changed/);
    return true;
  });
  assert.equal(model.weightRevisionId, publishedRevision);
  const staleAfter = await stale.exportCheckpoint();
  assert.equal(staleAfter.trainingStep, staleAfterStep.trainingStep);
  assert.deepEqual(
    new Uint8Array(staleAfter.weights),
    new Uint8Array(staleAfterStep.weights),
    'a stale commit must leave the private successor intact',
  );
  await stale.rollback();
  const staleRolledBack = await stale.exportCheckpoint();
  assert.equal(staleRolledBack.trainingStep, staleBefore.trainingStep);
  assert.deepEqual(new Uint8Array(staleRolledBack.weights), new Uint8Array(staleBefore.weights));
  assert.equal(stale.hasUncommittedUpdates, false);

  await Promise.all([first.close(), stale.close()]);
  await model.close();
  await runtime.close();
});

test('Trainer rollback discards private weights, optimizer state, and publication intent', async () => {
  const runtime = await full.VolvoxAI.createRuntime({ backends: ['cpu'] });
  const graph = new full.Graph();
  const parameter = graph.addWeight(
    'parameter', [1, 2], 'float32', Float32Array.from([0.25, -0.15]),
  );
  const { out } = graph.addOp('Identity', { input: parameter }, { out: [1, 2] });
  graph.setOutputs(out);
  const model = runtime.createModel(graph);
  const initialRevision = model.weightRevisionId;
  const trainer = await full.VolvoxAI.createTrainer(model, { backend: 'cpu' });
  const baseline = await trainer.exportCheckpoint();

  await trainer.trainStep({
    targets: [0],
    trainableTensors: ['parameter'],
    updateMode: 'adamw',
    optimizer: { learningRate: 0.1 },
  });
  assert.equal(trainer.hasUncommittedUpdates, true);
  await trainer.rollback();
  const restored = await trainer.exportCheckpoint();
  assert.equal(trainer.trainingStep, baseline.trainingStep);
  assert.equal(trainer.hasUncommittedUpdates, false);
  assert.equal(model.weightRevisionId, initialRevision);
  assert.deepEqual(new Uint8Array(restored.weights), new Uint8Array(baseline.weights));
  assert.equal(restored.optimizer, null);
  await assert.rejects(trainer.commit(), (error) => error.code === 'INVALID_ARGUMENT');

  await trainer.close();
  await model.close();
  await runtime.close();
});
