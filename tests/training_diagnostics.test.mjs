/** TrainStep numerical diagnostics: typed non-finite failures, the node that
 * first produced one, gradient norms and per-parameter statistics. */
import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, safetensors, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxTrainingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const D = 3, H = 4, C = 2;
// fc1 (Linear) -> act (GELU) -> fc2 (Linear) -> logits; schedule indexes 0, 1, 2.
// GELU propagates NaN; ReLU would turn it into zero.
const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [2, D]}},
  nodes: [
    {id: 'fc1', opType: 'Linear', inputs: {input: 'x', weight: 'fc1.w', bias: 'fc1.b'},
      outputs: {out: {tensor: 'h', dtype: 'float32', shape: [2, H]}}, params: {weight_layout: 'dout_din'}},
    {id: 'act', opType: 'GELU', inputs: {input: 'h'},
      outputs: {out: {tensor: 'r', dtype: 'float32', shape: [2, H]}}, params: {}},
    {id: 'fc2', opType: 'Linear', inputs: {input: 'r', weight: 'fc2.w', bias: 'fc2.b'},
      outputs: {out: {tensor: 'logits', dtype: 'float32', shape: [2, C]}}, params: {weight_layout: 'dout_din'}},
  ], outputs: ['logits']};
const values = (length, offset) => Float32Array.from({length}, (_, i) => Math.cos(i * 1.3 + offset) * .5);
const weights = safetensors([
  {name: 'fc1.w', shape: [H, D], data: values(H * D, 0)}, {name: 'fc1.b', shape: [H], data: values(H, 1)},
  {name: 'fc2.w', shape: [C, H], data: values(C * H, 2)}, {name: 'fc2.b', shape: [C], data: values(C, 3)}]);
const trainable = ['fc1.w', 'fc1.b', 'fc2.w', 'fc2.b'];

async function trainer(t) {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const training = new VxTrainingServiceClient(reportTransport(f.host));
  const model = await f.load(graph, weights);
  const handle = ok(await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId})));
  return {training, trainerId: handle.trainerId};
}
function step(trainerId, x, targets, {diagnostics, maxGradientNorm = 0, learningRate = .1} = {}) {
  return new p.TrainStepRequest({trainerId, inputs: tensors({x: {data: Float32Array.from(x), shape: [2, D]}}),
    losses: [new p.CrossEntropyLoss({name: 'ce', logitsName: 'logits', ignoreIndex: -1,
      targets: new p.Tensor({shape: [2n], dtype: p.DataType.DATA_TYPE_I32,
        inline: new Uint8Array(Int32Array.from(targets).buffer)})})],
    trainableNames: trainable,
    optimizer: new p.TrainerOptimizerOptions({kind: p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD,
      learningRate, maxGradientNorm}),
    diagnostics});
}
const finite = [.3, -.2, .8, -.5, .1, .4];

test('a step reports the global norm, clipping and per-parameter statistics', async t => {
  const {training, trainerId} = await trainer(t);
  const plain = ok(await training.trainStep(step(trainerId, finite, [0, 1])));
  assert.equal(plain.gradients, undefined, 'nothing observed without clipping or diagnostics');

  const clipped = ok(await training.trainStep(step(trainerId, finite, [0, 1], {maxGradientNorm: 1e-3})));
  assert.ok(clipped.gradients.globalNorm > 1e-3);
  assert.ok(Math.abs(clipped.gradients.clipScale - 1e-3 / clipped.gradients.globalNorm) < 1e-6);
  assert.equal(clipped.gradients.parameters.length, 0);
  assert.equal(clipped.gradients.toJson().clipScale !== undefined, true);

  const described = ok(await training.trainStep(step(trainerId, finite, [0, 1],
    {diagnostics: new p.TrainingDiagnostics({parameterStatistics: true})})));
  const {gradients} = described;
  assert.equal(described.updateApplied, true);
  assert.deepEqual(gradients.parameters.map(parameter => parameter.name), trainable);
  const sum = gradients.parameters.reduce((total, parameter) => total + parameter.gradientNorm ** 2, 0);
  assert.ok(Math.abs(Math.sqrt(sum) - gradients.globalNorm) < 1e-9 * Math.max(1, gradients.globalNorm));
  // Optional scalars read as zero in JavaScript; presence is in toJson().
  assert.equal(gradients.toJson().clipScale, undefined, 'no clipping requested');
  for (const parameter of gradients.parameters) {
    assert.equal(parameter.nonfiniteCount, 0n);
    assert.ok(parameter.parameterNorm > 0 && parameter.gradientMaxAbs <= parameter.gradientNorm + 1e-12);
    // SGD without weight decay: the update is exactly learning_rate * gradient.
    assert.ok(Math.abs(parameter.updateNorm - .1 * parameter.gradientNorm) < 1e-6 * Math.max(1, parameter.gradientNorm),
      `${parameter.name}: ${parameter.updateNorm} vs ${parameter.gradientNorm}`);
  }
});

test('a non-finite loss names the loss and the first forward node that produced it', async t => {
  const {training, trainerId} = await trainer(t);
  const input = [...finite]; input[1] = NaN;
  const failed = await training.trainStep(step(trainerId, input, [0, 1],
    {diagnostics: new p.TrainingDiagnostics({locateNonfinite: true})}));
  assert.equal(failed.report.status, p.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED);
  assert.equal(failed.report.code, p.OperationCode.OPERATION_CODE_TRAINING_LOSS_NONFINITE);
  assert.equal(failed.gradients.nonfiniteLoss, 'ce');
  assert.equal(failed.report.offendingNode.index, 0);
  assert.equal(failed.report.offendingNode.op, 'Linear');
  assert.match(failed.report.message, /loss 'ce' is not finite; first produced by forward node 0/);

  // Without locating, the loss is still named; the Trainer is usable afterwards.
  const plain = await training.trainStep(step(trainerId, input, [0, 1]));
  assert.equal(plain.report.code, p.OperationCode.OPERATION_CODE_TRAINING_LOSS_NONFINITE);
  assert.equal(plain.report.offendingNode, undefined);
  ok(await training.trainStep(step(trainerId, finite, [0, 1])));
});

test('a non-finite gradient names the parameter and the first backward node that produced it', async t => {
  const {training, trainerId} = await trainer(t);
  // Row 1 is ignored by the loss, so the loss stays finite, but its NaN
  // activations reach the weight gradients as 0 * NaN.
  const input = [...finite]; input[4] = NaN;
  const failed = await training.trainStep(step(trainerId, input, [0, -1],
    {diagnostics: new p.TrainingDiagnostics({locateNonfinite: true, parameterStatistics: true})}));
  assert.equal(failed.report.code, p.OperationCode.OPERATION_CODE_TRAINING_GRADIENT_NONFINITE, failed.report.message);
  assert.equal(failed.gradients.nonfiniteLoss, '');
  // Parameters are named in request order; fc1's gradients also carry the NaN.
  assert.equal(failed.gradients.firstNonfiniteParameter, 'fc1.w');
  // Backward runs fc2 first; it is where the first non-finite gradient appears.
  assert.equal(failed.report.offendingNode.index, 2);
  assert.match(failed.report.message, /first produced by backward node 2/);
  const counts = Object.fromEntries(failed.gradients.parameters.map(parameter => [parameter.name, parameter.nonfiniteCount]));
  assert.ok(counts['fc1.w'] > 0n && counts['fc2.w'] > 0n);
  assert.equal(counts['fc1.b'] + counts['fc2.b'] >= 0n, true);

  const plain = await training.trainStep(step(trainerId, input, [0, -1]));
  assert.equal(plain.report.code, p.OperationCode.OPERATION_CODE_TRAINING_GRADIENT_NONFINITE);
  assert.equal(plain.gradients.firstNonfiniteParameter, 'fc1.w');
  assert.equal(plain.gradients.parameters.length, 0);
  ok(await training.trainStep(step(trainerId, finite, [0, 1])));
});
