import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import { FullEngineHost } from '../ts/full.js';
import {
  VxInferenceServiceClient,
  VxProfilingServiceClient,
  VxTrainingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';

const packageJson = JSON.parse(
  await readFile(new URL('../package.json', import.meta.url), 'utf8'),
);
const fullWasmUrl = process.env.VOLVOXAI_TEST_FULL_WASM ?? new URL(
  `../dist/${packageJson.version}/volvoxai.wasm`,
  import.meta.url,
);

const GRAPH_PATH = 'fixture/graph.json';
const WEIGHTS_PATH = 'fixture/model.safetensors';

const graph = new TextEncoder().encode(JSON.stringify({
  format: 'volvox-graph/v1',
  dimensions: { B: { min: 1, max: 1 } },
  inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'parameter' },
    outputs: {
      out: { tensor: 'logits', dtype: 'float32', shape: ['B', 2] },
    },
    params: {},
  }],
  outputs: ['logits'],
}));

function safetensors() {
  const values = Float32Array.of(0.2, -0.4, 0.1, 0.3);
  let header = new TextEncoder().encode(JSON.stringify({
    parameter: { dtype: 'F32', shape: [2, 2], data_offsets: [0, values.byteLength] },
  }));
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + values.byteLength);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  output.set(new Uint8Array(values.buffer), 8 + header.byteLength);
  return output;
}

const files = new Map([
  [GRAPH_PATH, graph],
  [WEIGHTS_PATH, safetensors()],
]);

async function fixtureFetch(source) {
  const bytes = files.get(String(source));
  if (!bytes) return { ok: false, status: 404 };
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return {
    ok: true,
    status: 200,
    headers: new Headers({ 'content-length': String(bytes.byteLength) }),
    arrayBuffer: async () => buffer,
    text: async () => new TextDecoder().decode(bytes),
    json: async () => JSON.parse(new TextDecoder().decode(bytes)),
  };
}

const bytesOf = (view) => new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
const ok = (report) => report?.status === pb.NativeStatus.NATIVE_STATUS_OK;

test('C trainer requires the requested backend through the proto API', async () => {
  const host = new FullEngineHost({ wasmUrl: fullWasmUrl, fetch: fixtureFetch });
  try {
    const inference = new VxInferenceServiceClient(reportTransport(host));
    const training = new VxTrainingServiceClient(reportTransport(host));
    const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
    assert.ok(ok(runtime.report), runtime.report?.message);
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId: runtime.runtimeId, graphPath: GRAPH_PATH, weightPaths: [WEIGHTS_PATH],
    }));
    assert.ok(ok(model.report), model.report?.message);
    for (const backend of ['missing', 'webgpu']) {
      const trainer = await training.createTrainer(new pb.CreateTrainerRequest({
        modelId: model.modelId, backend,
      }));
      assert.notEqual(trainer.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
      assert.equal(trainer.trainerId, 0n);
    }
  } finally { await host.close(); }
});

test('full proto API trains, commits, rolls back, and closes over C/WASM', {
  timeout: 30_000,
}, async () => {
  const host = new FullEngineHost({
    wasmUrl: fullWasmUrl,
    fetch: fixtureFetch,
  });
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const training = new VxTrainingServiceClient(reportTransport(host));

  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.ok(ok(runtime.report), runtime.report?.message);
  const model = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: GRAPH_PATH,
    weightPaths: [WEIGHTS_PATH],
  }));
  assert.ok(ok(model.report), model.report?.message);

  const trainer = await training.createTrainer(new pb.CreateTrainerRequest({
    modelId: model.modelId,
    rngSeed: 7n,
  }));
  assert.ok(ok(trainer.report), trainer.report?.message);
  assert.equal(trainer.report?.backend, 'wasm');
  assert.deepEqual(trainer.inputs.map(({ name }) => name), ['x']);
  const exportWeights = async () => {
    const result = await training.exportTrainerWeights(new pb.ExportTrainerWeightsRequest({
      trainerId: trainer.trainerId,
    }));
    assert.ok(ok(result.report), result.report?.message);
    assert.equal(result.shards.length, 1);
    const shard = result.shards[0];
    const headerLength = Number(new DataView(shard.buffer, shard.byteOffset).getBigUint64(0, true));
    return new Float32Array(shard.slice(8 + headerLength).buffer);
  };
  const pathExport = await training.exportTrainerWeights(new pb.ExportTrainerWeightsRequest({
    trainerId: trainer.trainerId, outputPaths: ['model.safetensors'],
  }));
  assert.equal(pathExport.report.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
  assert.deepEqual(pathExport.shards, []);
  const initialWeights = await exportWeights();
  assert.deepEqual([...initialWeights], [...Float32Array.of(0.2, -0.4, 0.1, 0.3)]);
  const initialRevision = await inference.getModelRevision(
    new pb.ModelRef({ modelId: model.modelId }),
  );
  assert.ok(ok(initialRevision.report), initialRevision.report?.message);
  const profiling = new VxProfilingServiceClient(host);
  const trace = await profiling.startTrace(new pb.StartTraceRequest({
    runtimeId: runtime.runtimeId, detail: pb.TraceDetail.TRACE_DETAIL_NODES,
  }));

  // Trainer is a child of Model, which is a child of Runtime. Dropping the
  // parent ids must not close the objects retained by Trainer.
  assert.ok(ok(await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }))));
  assert.ok(ok(await inference.releaseRuntime(
    new pb.RuntimeRef({ runtimeId: runtime.runtimeId }))));

  const step = async (kind = pb.TrainingOptimizerKind.TRAINING_OPTIMIZER_KIND_SGD) =>
    (await training.trainStep(new pb.TrainStepRequest({
    trainerId: trainer.trainerId,
    inputs: [new pb.Tensor({
      name: 'x',
      dtype: pb.DataType.DATA_TYPE_F32,
      shape: [1n, 2n],
      inline: bytesOf(Float32Array.of(1, 0)),
    })],
    losses: [new pb.CrossEntropyLoss({
      name: 'classification',
      logitsName: 'logits',
      targets: new pb.Tensor({shape: [1n], dtype: pb.DataType.DATA_TYPE_I32, inline: new Uint8Array(Int32Array.of(0).buffer)}),
    })],
    trainableNames: ['parameter'],
    optimizer: new pb.TrainerOptimizerOptions({
      kind,
      learningRate: 0.1,
    }),
  })));

  const invalid = await step(99);
  assert.equal(invalid.report?.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  const first = await step();
  assert.ok(ok(first.report), first.report?.message);
  assert.equal(first.backend, 'wasm');
  assert.equal(first.updateApplied, true);
  assert.equal(first.optimizerStep, 1n);
  assert.equal(first.metrics.length, 1);

  const firstWeights = await exportWeights();
  const probability = Math.exp(0.2) / (Math.exp(0.2) + Math.exp(-0.4));
  assert.ok(Math.abs(first.loss + Math.log(probability)) < 1e-6);
  assert.ok(firstWeights.some((value, index) => value !== initialWeights[index]));

  const committed = await training.commitTrainer(
    new pb.TrainerRef({ trainerId: trainer.trainerId }));
  assert.ok(ok(committed.report), committed.report?.message);
  assert.equal(committed.weightRevision, initialRevision.weightRevision + 1n);

  const second = await step();
  assert.ok(ok(second.report), second.report?.message);
  assert.equal(second.optimizerStep, 2n);
  assert.ok(ok(await training.rollbackTrainer(
    new pb.TrainerRef({ trainerId: trainer.trainerId }))));

  assert.deepEqual([...(await exportWeights())], [...firstWeights]);
  const stopped = await profiling.stopTrace(new pb.TraceRef(trace));
  assert.equal(stopped.state, pb.TraceState.TRACE_STATE_READY);
  const observations = await profiling.readTrace(new pb.ReadTraceRequest(trace));
  const steps = observations.events.filter(event => event.name === 'TrainStep');
  assert.equal(steps.length, 3); // Includes the refused optimizer request's host work.
  assert.ok(steps.every(event => event.lineage.modelId === model.modelId));
  assert.ok(observations.events.some(event => event.host !== undefined && event.node !== undefined));
  const phases = new Set(observations.events.map(event => event.phase));
  for (const phase of [pb.TracePhase.TRACE_PHASE_FORWARD, pb.TracePhase.TRACE_PHASE_LOSS,
    pb.TracePhase.TRACE_PHASE_BACKWARD, pb.TracePhase.TRACE_PHASE_GRADIENT, pb.TracePhase.TRACE_PHASE_OPTIMIZER]) {
    assert.ok(phases.has(phase), `missing training phase ${phase}`);
  }
  const backward = observations.events.filter(event => event.phase === pb.TracePhase.TRACE_PHASE_BACKWARD);
  assert.ok(backward.every(event => event.host && event.node && !event.program));
  assert.ok(observations.events.some(event => event.phase === pb.TracePhase.TRACE_PHASE_OPTIMIZER && event.tensorName));
  await profiling.releaseTrace(new pb.TraceRef(trace));
  assert.ok(ok(await training.releaseTrainer(
    new pb.TrainerRef({ trainerId: trainer.trainerId }))));
  await host.close();
});
