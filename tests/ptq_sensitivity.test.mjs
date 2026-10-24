import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, safetensors, tensors, statelessGraph} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxInferenceServiceClient, VxQuantizationServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {ptqSensitivity} from '../tools/ptq_sensitivity.mjs';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.wasm`, import.meta.url).pathname;
const D = 8, H = 16;
const ramp = (length, scale, offset = 0) => Float32Array.from({length}, (_, i) => Math.sin(i * 1.7 + offset) * scale);
// fc2 sees a wide activation range, so quantizing it alone costs the most.
const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [1, D]}},
  nodes: [
    {id: 'fc1', opType: 'Linear', inputs: {input: 'x', weight: 'fc1.w', bias: 'fc1.b'},
      outputs: {out: {tensor: 'h1', dtype: 'float32', shape: [1, H]}}, params: {weight_layout: 'dout_din'}},
    {id: 'act', opType: 'GELU', inputs: {input: 'h1'},
      outputs: {out: {tensor: 'h2', dtype: 'float32', shape: [1, H]}}, params: {}},
    {id: 'fc2', opType: 'Linear', inputs: {input: 'h2', weight: 'fc2.w', bias: 'fc2.b'},
      outputs: {out: {tensor: 'y', dtype: 'float32', shape: [1, D]}}, params: {weight_layout: 'dout_din'}},
  ], outputs: ['y']};
const weights = safetensors([
  {name: 'fc1.w', shape: [H, D], data: ramp(H * D, .5)}, {name: 'fc1.b', shape: [H], data: ramp(H, .1, 1)},
  {name: 'fc2.w', shape: [D, H], data: ramp(D * H, 2, 2)}, {name: 'fc2.b', shape: [D], data: ramp(D, .1, 3)}]);
const batches = (count, offset) => Array.from({length: count}, (_, i) =>
  tensors({x: {data: ramp(D, 1.5, offset + i * .7), shape: [1, D]}}));

test('each quantized node is ranked alone and by what keeping it in F32 recovers', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const options = {
    inference: new VxInferenceServiceClient(reportTransport(f.host)),
    quantization: new VxQuantizationServiceClient(reportTransport(f.host)),
    p, runtimeId: f.runtime.runtimeId,
    source: {graph: new TextEncoder().encode(JSON.stringify(statelessGraph(graph))), weights: [weights]},
    calibration: batches(8, 0), evaluation: batches(4, 20), outputs: ['y'],
    config: new p.QuantizationConfig({weights: new p.WeightQuantization({format: new p.WeightFormat({bits: 8})}), staticActivations: new p.StaticActivations({dtype: p.DataType.DATA_TYPE_I8, scheme: p.AffineScheme.AFFINE_SCHEME_ASYMMETRIC})}),
  };
  const report = await ptqSensitivity(options);
  assert.equal(report.format, 'volvoxai-ptq-sensitivity/v1');
  assert.ok(report.baseline.sqnrDb > 10 && report.baseline.sqnrDb < 80, `baseline ${report.baseline.sqnrDb}`);
  assert.deepEqual(report.rows.map(row => row.node), ['fc1', 'fc2']);
  for (const row of report.rows) {
    assert.equal(typeof row.recoveryDb, 'number');
    // One quantized node alone loses less than the whole quantized graph.
    assert.ok(row.isolated.sqnrDb >= report.baseline.sqnrDb - 1, `${row.node} isolated ${row.isolated.sqnrDb}`);
  }
  assert.deepEqual([...report.mostSensitive].sort(), ['fc1', 'fc2']);
  assert.deepEqual([...report.bestToKeepFloat].sort(), ['fc1', 'fc2']);
  // Deterministic: the same inputs give the same ranking and numbers.
  const again = await ptqSensitivity(options);
  assert.deepEqual(again.rows, report.rows);
});

for (const [bits, mode, dtype] of [
  [2, 'DYNAMIC', 'I8'], [3, 'DYNAMIC', 'I8'], [3, 'FLOAT', 'F16'],
  [4, 'DYNAMIC', 'I8'], [4, 'FLOAT', 'F16'], [8, 'DYNAMIC', 'I8'], [8, 'FLOAT', 'F16'],
]) {
  test(`grouped ${bits}-bit ${mode} sensitivity evaluates without calibration`, async t => {
    const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
    const client = new VxQuantizationServiceClient(reportTransport(f.host));
    let planCalls = 0;
    const options = {
      inference: new VxInferenceServiceClient(reportTransport(f.host)),
      quantization: {
        createQuantizationPlan: async request => { planCalls++; return client.createQuantizationPlan(request); },
        exportQuantizedModel: request => client.exportQuantizedModel(request),
        releaseQuantizationPlan: request => client.releaseQuantizationPlan(request),
        calibrateQuantizationPlan: () => { throw new Error('grouped sensitivity attempted calibration'); },
      },
      p, runtimeId: f.runtime.runtimeId,
      source: {graph: new TextEncoder().encode(JSON.stringify(statelessGraph(graph))), weights: [weights]},
      evaluation: batches(3, 20), outputs: ['y'],
      packedArithmetic: mode === 'FLOAT' ? p.PackedArithmetic.PACKED_ARITHMETIC_F16 : p.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8,
      config: new p.QuantizationConfig({weights: new p.WeightQuantization({format: new p.WeightFormat({bits: bits, groupSize: 128})})}),
    };
    const report = await ptqSensitivity(options);
    assert.equal(report.calibrationBatches, 0);
    assert.equal(report.evaluationBatches, 3);
    assert.equal(planCalls, 5);
    assert.deepEqual(report.configuration, p.QuantizationConfig.fromBinary(options.config.toBinary()).toJson());
    assert.equal(report.packedArithmetic, p.PackedArithmetic[options.packedArithmetic]);
    assert.deepEqual(report.rows.map(row => row.node), ['fc1', 'fc2']);
    assert.ok(Number.isFinite(report.baseline.sqnrDb));
    for (const row of report.rows) {
      assert.ok(Number.isFinite(row.isolated.sqnrDb));
      assert.ok(Number.isFinite(row.leaveOneFloat.sqnrDb));
    }
    await assert.rejects(ptqSensitivity({...options, calibration: batches(1, 0)}), /unnecessary for the resolved plan/);
  });
}

test('input-aware low-bit sweep measures explicit W4 replacements', async t => {
  const f = await fixture({wasmUrl, full: true}); t.after(() => f.close());
  const report = await ptqSensitivity({
    inference: new VxInferenceServiceClient(reportTransport(f.host)),
    quantization: new VxQuantizationServiceClient(reportTransport(f.host)), p,
    runtimeId: f.runtime.runtimeId,
    source: {graph: new TextEncoder().encode(JSON.stringify(statelessGraph(graph))), weights: [weights]},
    calibration: batches(8, 0), evaluation: batches(4, 20), outputs: ['y'],
    modes: ['leave-one-higher'], higherWeightBits: 4,
    packedArithmetic: p.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8,
    config: new p.QuantizationConfig({weights: new p.WeightQuantization({
      format: new p.WeightFormat({bits: 3, groupSize: 32}),
      algorithm: p.WeightAlgorithm.WEIGHT_ALGORITHM_IMPORTANCE})}),
  });
  assert.equal(report.higherWeightBits, 4);
  assert.equal(report.calibrationBatches, 8);
  assert.deepEqual(report.rows.map(row => row.node), ['fc1', 'fc2']);
  for (const row of report.rows) {
    assert.ok(Number.isFinite(row.leaveOneHigher.sqnrDb));
    assert.equal(row.higherRecoveryDb, row.leaveOneHigher.sqnrDb - report.baseline.sqnrDb);
  }
  assert.deepEqual(report.bestToKeepHigher,
    [...report.rows].sort((a, b) => b.higherRecoveryDb - a.higherRecoveryDb).map(row => row.node));
});
