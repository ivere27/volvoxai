import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, safetensors, statelessGraph, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {readPackageVersion} from '../tools/release_version.mjs';
import {VxTrainingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const wasmUrl = new URL(`../dist/${readPackageVersion()}/volvoxai.wasm`, import.meta.url).pathname;
const document = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
  inputs: {x: {dtype: 'float32', shape: [1, 3]}}, nodes: [{id: 'projection', opType: 'Linear',
    inputs: {input: 'x', weight: 'w'}, params: {weight_layout: 'dout_din'},
    outputs: {out: {tensor: 'logits', dtype: 'float32', shape: [1, 2]}}}], outputs: ['logits']});
const graph = new TextEncoder().encode(JSON.stringify(document));
const original = Float32Array.of(.24, .11, -.4, .18, -.2, .15);
const source = safetensors([{name: 'w', shape: [2, 3], data: original}]);
const inputs = tensors({x: {shape: [1, 3], data: Float32Array.of(1, .5, -.25)}});
const losses = [new p.CrossEntropyLoss({name: 'ce', logitsName: 'logits',
  targets: tensors({target: {shape: [1], data: Int32Array.of(0)}})[0]})];
const policy = bits => new p.WeightQuantization({format: new p.WeightFormat({bits, groupSize: 32})});
const options = bits => new p.QatOptions({nodes: [new p.QatNode({nodeId: 'projection', weights: policy(bits)})]});

for (const bits of [2, 3, 4]) test(`W${bits} QAT trains, resumes and exports matching packed weights in WASM`, async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const training = new VxTrainingServiceClient(reportTransport(f.host));
  const model = await f.load(document, source);
  const trainer = ok(await training.createTrainer(new p.CreateTrainerRequest({
    modelId: model.modelId, backend: 'wasm', qat: options(bits)})));
  const step = async id => ok(await training.trainStep(new p.TrainStepRequest({trainerId: id,
    inputs, losses, trainableNames: ['w'], optimizer: new p.TrainerOptimizerOptions({
      kind: p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD, learningRate: .1})})));
  const exported = async id => ok(await training.exportTrainerWeights(
    new p.ExportTrainerWeightsRequest({trainerId: id})));
  const first = await step(trainer.trainerId);
  const high = 2 ** (bits - 1) - 1;
  const reconstructed = Array.from(original, (value, index) => {
    const row = index < 3 ? original.slice(0, 3) : original.slice(3);
    const scale = Math.fround(Math.max(...row.map(Math.abs)) / high);
    // These fixtures deliberately contain no half-way rounding cases.
    return Math.fround(Math.round(Math.fround(value / scale)) * scale);
  });
  const logits = [0, 1].map(row => reconstructed[row * 3] +
    .5 * reconstructed[row * 3 + 1] - .25 * reconstructed[row * 3 + 2]);
  assert.ok(Math.abs(first.loss - Math.log1p(Math.exp(logits[1] - logits[0]))) < 2e-6);
  let last = first;
  for (let i = 0; i < 24; i++) last = await step(trainer.trainerId);
  assert.ok(last.loss < first.loss, `${first.loss} -> ${last.loss}`);
  const state = ok(await training.getTrainerState(new p.TrainerRef(trainer)));
  assert.equal(state.qat.nodes[0].weights.format.bits, bits);
  assert.ok(state.qatWorkspaceHighWaterBytes > 0n);

  const checkpoint = ok(await training.exportTrainerCheckpoint(new p.ExportTrainerCheckpointRequest(trainer))).checkpoint;
  assert.equal(checkpoint.qat.nodes[0].weights.format.bits, bits);
  const resumed = ok(await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId,
    backend: 'wasm', checkpoint})));
  await step(trainer.trainerId); await step(resumed.trainerId);
  assert.deepEqual((await exported(trainer.trainerId)).shards, (await exported(resumed.trainerId)).shards);
  const mismatch = await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId,
    backend: 'wasm', checkpoint, qat: options(bits === 4 ? 3 : 4)}));
  assert.notEqual(mismatch.report.status, p.NativeStatus.NATIVE_STATUS_OK);
  assert.equal(mismatch.trainerId, 0n);

  const masters = await exported(trainer.trainerId);
  const plan = ok(await f.quantization.createQuantizationPlan(new p.CreateQuantizationPlanRequest({
    bytes: new p.ModelBytes({graph, weightShards: masters.shards}),
    config: new p.QuantizationConfig({weights: policy(bits)})})));
  const packed = ok(await f.quantization.exportQuantizedModel(new p.ExportQuantizedModelRequest(plan)));
  const loaded = ok(await f.inference.loadModel(new p.LoadModelRequest({runtimeId: f.runtime.runtimeId,
    package: new p.ModelPackage({graphDocument: packed.bytes.graph, weightShards: packed.bytes.weightShards})})));
  const compiled = await f.compile(loaded);
  const output = await f.evaluate({compiledModelId: compiled.compiledModelId, inputs});
  const y = await f.read(output, 'logits'); await f.releaseOutputs(output);
  const next = await step(trainer.trainerId);
  assert.ok(Math.abs(next.loss - Math.log1p(Math.exp(y[1] - y[0]))) < 3e-6);
  ok(await training.rollbackTrainer(new p.TrainerRef(resumed)));
  assert.deepEqual((await exported(resumed.trainerId)).shards, checkpoint.weightShards);
  const refused = await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId,
    backend: 'webgpu', qat: options(bits)}));
  assert.notEqual(refused.report.status, p.NativeStatus.NATIVE_STATUS_OK);
  assert.equal(refused.trainerId, 0n);
});
