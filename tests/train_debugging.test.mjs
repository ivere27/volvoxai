/** Train-step debug target: one TrainStep stepped through forward nodes, the
 * loss, backward nodes, gradient accumulation and optimizer updates. */
import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, safetensors, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxTrainingServiceClient, VxDebugServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const D = 3, H = 4, C = 2;
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
const finite = [.3, -.2, .8, -.5, .1, .4];

async function setup(t) {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const training = new VxTrainingServiceClient(reportTransport(f.host));
  const debug = new VxDebugServiceClient(reportTransport(f.host));
  const model = await f.load(graph, weights);
  const trainer = async () => ok(await training.createTrainer(new p.CreateTrainerRequest({modelId: model.modelId}))).trainerId;
  return {f, training, debug, trainer};
}
const step = (trainerId, x, targets) => new p.TrainStepRequest({trainerId,
  inputs: tensors({x: {data: Float32Array.from(x), shape: [2, D]}}),
  losses: [new p.CrossEntropyLoss({name: 'ce', logitsName: 'logits', ignoreIndex: -1,
    targets: new p.Tensor({shape: [2n], dtype: p.DataType.DATA_TYPE_I32, inline: new Uint8Array(Int32Array.from(targets).buffer)})})],
  trainableNames: trainable,
  optimizer: new p.TrainerOptimizerOptions({kind: p.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_ADAMW, learningRate: .05})});
/** Exact private weights as a SafeTensors shard. */
async function parameters(training, trainerId) {
  const exported = ok(await training.exportTrainerWeights(new p.ExportTrainerWeightsRequest({trainerId})));
  return [...exported.shards[0]];
}
async function run(debug, info, request = {}) {
  return ok(await debug.continueDebugSession(new p.ContinueDebugSessionRequest({
    debugSessionId: info.debugSessionId, expectedRevision: info.revision, ...request})));
}

test('a debugged train step leaves the Trainer exactly as TrainStep does', async t => {
  const {training, debug, trainer} = await setup(t);
  const ordinary = await trainer(), debugged = await trainer();
  const expected = ok(await training.trainStep(step(ordinary, finite, [0, 1])));
  let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({trainStep: step(debugged, finite, [0, 1])})));
  assert.equal(info.target, p.DebugTarget.DEBUG_TARGET_TRAIN_STEP);
  assert.equal(info.stopReason, p.DebugStopReason.DEBUG_STOP_REASON_ENTRY);
  info = await run(debug, info);
  assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED);
  const result = info.trainStepResult;
  assert.equal(result.loss, expected.loss);
  assert.equal(result.optimizerStep, expected.optimizerStep);
  assert.equal(result.updateApplied, true);
  assert.deepEqual(result.metrics.map(m => m.toJson()), expected.metrics.map(m => m.toJson()));
  ok(await debug.releaseDebugSession(new p.DebugSessionRef(info)));
  assert.deepEqual(await parameters(training, debugged), await parameters(training, ordinary));
  // Both Trainers continue identically, including AdamW moments.
  const next = [.1, .2, -.3, .4, -.5, .6];
  ok(await training.trainStep(step(ordinary, next, [1, 0])));
  ok(await training.trainStep(step(debugged, next, [1, 0])));
  assert.deepEqual(await parameters(training, debugged), await parameters(training, ordinary));
});

test('the plan steps through every phase and exposes gradients', async t => {
  const {training, debug, trainer} = await setup(t);
  const trainerId = await trainer();
  let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({trainStep: step(trainerId, finite, [0, 1]),
    capture: new p.DebugCapture({values: true})})));
  const {plan} = ok(await debug.getDebugPlan(new p.DebugSessionRef(info)));
  const phases = plan.steps.map(s => s.phase);
  const P = p.TracePhase;
  assert.deepEqual(phases, [P.TRACE_PHASE_FORWARD, P.TRACE_PHASE_FORWARD, P.TRACE_PHASE_FORWARD, P.TRACE_PHASE_LOSS,
    P.TRACE_PHASE_BACKWARD, P.TRACE_PHASE_BACKWARD, P.TRACE_PHASE_BACKWARD, P.TRACE_PHASE_GRADIENT,
    P.TRACE_PHASE_GRADIENT, P.TRACE_PHASE_OPTIMIZER, P.TRACE_PHASE_OPTIMIZER, P.TRACE_PHASE_OPTIMIZER,
    P.TRACE_PHASE_OPTIMIZER]);
  assert.deepEqual(plan.steps.slice(4, 7).map(s => s.sourceNodeIds[0]), ['fc2', 'act', 'fc1']);
  const gradient = plan.tensors.find(t => t.name === 'fc2.w.grad');
  assert.equal(plan.tensors[gradient.gradientOf].name, 'fc2.w');
  // While attached, the Trainer refuses other work.
  const busy = await training.trainStep(step(trainerId, finite, [0, 1]));
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  // Paused before fc1's forward, which runs once; the next step executing
  // fc1 is its backward, after fc2 and act.
  info = await run(debug, info, {breakBeforeNodes: ['fc1']});
  assert.equal(info.stopReason, p.DebugStopReason.DEBUG_STOP_REASON_BREAKPOINT);
  assert.equal(info.nextStep, 6);
  assert.equal(plan.steps[info.nextStep].phase, P.TRACE_PHASE_BACKWARD);
  info = await run(debug, info);
  assert.equal(info.state, p.DebugState.DEBUG_STATE_COMPLETED);
  const events = ok(await debug.listDebugEvents(new p.ListDebugEventsRequest({debugSessionId: info.debugSessionId, pageSize: 4096}))).events;
  const fc2Backward = events.find(e => e.step === 4 && e.point === p.DebugPoint.DEBUG_POINT_AFTER);
  const produced = fc2Backward.snapshots.map(s => plan.tensors[s.tensorId].name).sort();
  assert.deepEqual(produced, ['fc2.b.grad', 'fc2.w.grad', 'r.grad']);
  assert.ok(fc2Backward.snapshots.every(s => s.status === p.DebugTensorStatus.DEBUG_TENSOR_STATUS_AVAILABLE &&
    Number(s.statistics.finiteCount) > 0));
  ok(await debug.releaseDebugSession(new p.DebugSessionRef(info)));
  ok(await training.trainStep(step(trainerId, finite, [0, 1])));
});

test('break_on_nonfinite stops where NaN first appears, forward or backward; cancel restores', async t => {
  const {training, debug, trainer} = await setup(t);
  const trainerId = await trainer();
  const before = await parameters(training, trainerId);
  const input = [...finite]; input[4] = NaN; // Row 1 is ignored by the loss, which stays finite.
  const P = p.TracePhase;
  async function stopAt(capture) {
    let info = ok(await debug.createDebugSession(new p.CreateDebugSessionRequest({
      trainStep: step(trainerId, input, [0, -1]), capture})));
    info = await run(debug, info, {breakOnNonfinite: true});
    assert.equal(info.stopReason, p.DebugStopReason.DEBUG_STOP_REASON_NONFINITE);
    const {plan} = ok(await debug.getDebugPlan(new p.DebugSessionRef(info)));
    const stopped = plan.steps[info.nextStep - 1];
    info = ok(await debug.cancelDebugSession(new p.DebugSessionRef(info)));
    assert.equal(info.state, p.DebugState.DEBUG_STATE_CANCELLED);
    ok(await debug.releaseDebugSession(new p.DebugSessionRef(info)));
    return stopped;
  }
  // Every value captured: the NaN input first shows in fc1's forward output.
  const forward = await stopAt(undefined);
  assert.equal(forward.phase, P.TRACE_PHASE_FORWARD);
  assert.deepEqual(forward.sourceNodeIds, ['fc1']);
  // Gradients only, like detect_anomaly: fc2's backward is the first to write one.
  const gradients = await stopAt(new p.DebugCapture({tensorNames: ['r.grad', 'h.grad', ...trainable.map(n => `${n}.grad`)]}));
  assert.equal(gradients.phase, P.TRACE_PHASE_BACKWARD);
  assert.deepEqual(gradients.sourceNodeIds, ['fc2']);
  // Cancelling restored the Trainer each time.
  assert.deepEqual(await parameters(training, trainerId), before);
  const state = ok(await training.getTrainerState(new p.TrainerRef({trainerId})));
  assert.equal(state.optimizerStep, 0n);
  ok(await training.trainStep(step(trainerId, finite, [0, 1])));
});
