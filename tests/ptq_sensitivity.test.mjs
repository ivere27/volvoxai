import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, safetensors, tensors} from '../tools/proto_fixture.mjs';
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
    source: {graph: new TextEncoder().encode(JSON.stringify(graph)), weights: [weights]},
    calibration: batches(8, 0), evaluation: batches(4, 20), outputs: ['y'],
    config: {activationDtype: p.DataType.DATA_TYPE_I8, activationScheme: p.PtqScheme.PTQ_SCHEME_ASYMMETRIC},
  };
  const report = await ptqSensitivity(options);
  assert.equal(report.format, 'volvoxai-ptq-sensitivity/v1');
  assert.ok(report.baseline.sqnrDb > 10 && report.baseline.sqnrDb < 80, `baseline ${report.baseline.sqnrDb}`);
  assert.deepEqual(report.rows.map(row => row.node), ['fc1', 'act', 'fc2']);
  for (const row of report.rows) {
    assert.equal(typeof row.recoveryDb, 'number');
    // A GELU alone has no dense layer; the writer refuses that selection.
    if (row.node === 'act') { assert.equal(row.isolated.sqnrDb, null); assert.match(row.isolated.note, /refused/); continue; }
    // One quantized node alone loses less than the whole quantized graph.
    assert.ok(row.isolated.sqnrDb >= report.baseline.sqnrDb - 1, `${row.node} isolated ${row.isolated.sqnrDb}`);
  }
  assert.deepEqual([...report.mostSensitive].sort(), ['fc1', 'fc2']);
  assert.deepEqual([...report.bestToKeepFloat].sort(), ['act', 'fc1', 'fc2']);
  // Deterministic: the same inputs give the same ranking and numbers.
  const again = await ptqSensitivity(options);
  assert.deepEqual(again.rows, report.rows);
});
