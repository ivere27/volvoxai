/** GetTraceSummary, plan step costs and external-tool options over the
 * inference WASM profile. */
import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxProfilingServiceClient, VxPlatformServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.lite.wasm`, import.meta.url).pathname;
// fc: [2,3] x [4,3]^T -> [2,4] is 24 multiply-accumulates. GELU is elementwise
// with one transcendental per element; Add is elementwise; Reshape moves data.
const mlp = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [2, 3]}},
  nodes: [
    {id: 'fc', opType: 'Linear', inputs: {input: 'x', weight: 'fc.w', bias: 'fc.b'},
      outputs: {out: {tensor: 'h', dtype: 'float32', shape: [2, 4]}}, params: {weight_layout: 'dout_din'}},
    {id: 'act', opType: 'GELU', inputs: {input: 'h'},
      outputs: {out: {tensor: 'g', dtype: 'float32', shape: [2, 4]}}, params: {}},
    {id: 'sum', opType: 'Add', inputs: {a: 'g', b: 'g'},
      outputs: {out: {tensor: 's', dtype: 'float32', shape: [2, 4]}}, params: {}},
    {id: 'flat', opType: 'Reshape', inputs: {input: 's'},
      outputs: {out: {tensor: 'y', dtype: 'float32', shape: [8]}}, params: {shape: [8]}},
  ], outputs: ['y']};
const mlpWeights = safetensors([
  {name: 'fc.w', shape: [4, 3], data: Float32Array.from({length: 12}, (_, i) => Math.sin(i))},
  {name: 'fc.b', shape: [4], data: Float32Array.of(.1, -.2, .3, -.4)}]);
// Conv2D NHWC [1,4,4,2] with HWIO [3,3,2,3] -> [1,2,2,3]: 12 outputs x 18 = 216.
const conv = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [1, 4, 4, 2]}},
  nodes: [{id: 'conv', opType: 'Conv2D', inputs: {input: 'x', weight: 'w'},
    outputs: {out: {tensor: 'y', dtype: 'float32', shape: [1, 2, 2, 3]}},
    params: {data_layout: 'NHWC', weight_layout: 'HWIO', stride: [1, 1]}}], outputs: ['y']};
const convWeights = safetensors([{name: 'w', shape: [3, 3, 2, 3], data: Float32Array.from({length: 54}, (_, i) => i / 54)}]);

async function traced(t, graph, weights, input, {runs = 3, options = {}} = {}) {
  const f = await fixture({wasmUrl}); t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const compiled = await f.compile(await f.load(graph, weights));
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
  const trace = ok(await profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId,
    options: new p.TraceOptions({detail: p.TraceDetail.TRACE_DETAIL_NODES, executionPlans: true, ...options})})));
  for (let i = 0; i < runs; i++) {
    const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId, inputs: input})));
    ok(await f.inference.releaseResult(new p.ResultRef(result)));
  }
  // Closing the host releases the trace with every other handle.
  const info = ok(await profiling.stopTrace(new p.TraceRef(trace)));
  return {f, profiling, trace, info};
}
const summary = (profiling, trace, groupBy, maxRows) => profiling.getTraceSummary(
  new p.GetTraceSummaryRequest({traceId: trace.traceId, groupBy, maxRows}));

test('plan steps carry the work one call implies', async t => {
  const {profiling, trace} = await traced(t, mlp, mlpWeights,
    tensors({x: {data: Float32Array.of(1, -2, 3, -4, 5, -6), shape: [2, 3]}}), {runs: 1});
  const {plan} = ok(await profiling.getTracePlan(new p.GetTracePlanRequest({traceId: trace.traceId, planId: 1n})));
  const byOperator = Object.fromEntries(plan.steps.map(step => [step.operatorName, step.cost]));
  assert.equal(byOperator.Linear.status, p.CostStatus.COST_STATUS_EXACT);
  assert.equal(byOperator.Linear.multiplyAccumulates, 24n);
  assert.equal(byOperator.Linear.inputBytes, (6n + 12n + 4n) * 4n);
  assert.equal(byOperator.Linear.outputBytes, 32n);
  assert.equal(byOperator.GELU.elementwiseOperations, 8n);
  assert.equal(byOperator.GELU.transcendentalOperations, 8n);
  assert.equal(byOperator.Add.elementwiseOperations, 8n);
  assert.equal(byOperator.Add.transcendentalOperations, 0n);
  assert.equal(byOperator.Reshape.multiplyAccumulates, 0n);
  assert.equal(byOperator.Reshape.elementwiseOperations, 0n);
});

test('convolution cost is independent of the weight layout', async t => {
  const {profiling, trace} = await traced(t, conv, convWeights,
    tensors({x: {data: Float32Array.from({length: 32}, (_, i) => i / 32), shape: [1, 4, 4, 2]}}), {runs: 1});
  const {plan} = ok(await profiling.getTracePlan(new p.GetTracePlanRequest({traceId: trace.traceId, planId: 1n})));
  const step = plan.steps.find(step => step.operatorName === 'Conv2D');
  assert.equal(step.cost.status, p.CostStatus.COST_STATUS_EXACT);
  assert.equal(step.cost.multiplyAccumulates, 216n);
});

test('node, operator and call summaries aggregate the recorded events', async t => {
  const runs = 4;
  const {profiling, trace, info} = await traced(t, mlp, mlpWeights,
    tensors({x: {data: Float32Array.of(1, -2, 3, -4, 5, -6), shape: [2, 3]}}), {runs});
  const nodes = ok(await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE));
  assert.equal(nodes.groupBy, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE);
  assert.equal(nodes.events.count, info.events.count);
  assert.equal(nodes.truncated, false);
  for (let i = 1; i < nodes.rows.length; i++) assert.ok(nodes.rows[i - 1].totalNs >= nodes.rows[i].totalNs);
  const linear = nodes.rows.find(row => row.name === 'Linear');
  assert.equal(linear.count, BigInt(runs));
  assert.equal(linear.domain, p.TraceTimeDomain.TRACE_TIME_DOMAIN_HOST);
  assert.deepEqual(linear.sourceNodeIds, ['fc']);
  assert.equal(linear.outputName, 'h');
  assert.equal(linear.costPerCall.multiplyAccumulates, 24n);
  assert.ok(linear.minNs <= linear.medianNs && linear.medianNs <= linear.p95Ns && linear.p95Ns <= linear.maxNs);
  // Host nodes on WASM measure the numerical work, so rates are reported.
  const json = linear.toJson();
  if (linear.totalNs > 0n) {
    assert.ok(json.achievedFlopsPerSecond > 0 && json.achievedBytesPerSecond > 0);
    assert.ok(Math.abs(linear.achievedFlopsPerSecond - 48 * runs / (Number(linear.totalNs) / 1e9)) <
      1e-6 * linear.achievedFlopsPerSecond);
  }
  const reshape = nodes.rows.find(row => row.name === 'Reshape');
  if (reshape) assert.equal(reshape.toJson().achievedFlopsPerSecond, undefined, 'no multiply-accumulates');
  const share = nodes.rows.reduce((sum, row) => sum + row.share, 0);
  assert.ok(Math.abs(share - 1) < 1e-9, `node shares sum to ${share}`);

  const operators = ok(await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATOR));
  assert.deepEqual(new Set(operators.rows.map(row => row.name)), new Set(nodes.rows.map(row => row.name)));
  assert.ok(operators.rows.every(row => row.toJson().scheduleIndex === undefined && row.sourceNodeIds.length === 0));

  const calls = ok(await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATION));
  const execute = calls.rows.find(row => row.name === 'Execute');
  assert.equal(execute.count, BigInt(runs));
  assert.equal(execute.lineage.contextId > 0n, true);

  const limited = ok(await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE, 1));
  assert.equal(limited.rows.length, 1);
  assert.equal(limited.truncated, nodes.rows.length > 1);
  assert.deepEqual(limited.rows[0].toJson(), nodes.rows[0].toJson());

  const invalid = await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_UNSPECIFIED);
  assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
});

test('a summary needs a READY trace', async t => {
  const f = await fixture({wasmUrl}); t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const trace = ok(await profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  const busy = await summary(profiling, trace, p.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE);
  assert.equal(busy.report.status, p.NativeStatus.NATIVE_STATUS_BUSY);
  ok(await profiling.releaseTrace(new p.TraceRef(trace)));
});

test('external annotations are reported, not faked, where no tool mechanism exists', async t => {
  const f = await fixture({wasmUrl}); t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const platform = new VxPlatformServiceClient(reportTransport(f.host));
  const before = (await platform.getMonotonicTime(new p.Empty())).nanoseconds;
  const {trace, info} = await traced(t, mlp, mlpWeights,
    tensors({x: {data: Float32Array.of(1, -2, 3, -4, 5, -6), shape: [2, 3]}}),
    {runs: 1, options: {external: new p.TraceExternalOptions({annotations: true, capture: true})}});
  const after = (await platform.getMonotonicTime(new p.Empty())).nanoseconds;
  assert.equal(info.options.external.annotations, true);
  assert.equal(info.options.external.capture, true);
  // WASM has no NVTX or RenderDoc: each attempted mechanism says so.
  const rows = Object.fromEntries(info.external.map(row => [row.mechanism, row]));
  assert.equal(rows.nvtx.status, p.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED);
  assert.equal(rows.nvtx.ranges, 0n);
  assert.equal(rows.renderdoc.status, p.ObservationStatus.OBSERVATION_STATUS_UNSUPPORTED);
  assert.ok(info.captureOriginNs >= before - 1000000n && info.captureOriginNs <= after, 'origin on the platform clock');
  void trace;
  const plain = ok(await profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId})));
  assert.equal(plain.options.external, undefined);
  assert.equal(plain.external.length, 0);
  ok(await profiling.releaseTrace(new p.TraceRef(plain)));
});
