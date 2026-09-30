/** A READY trace replayed into the Performance timeline lines up with
 * performance.now(), because the WASM host samples that clock. */
import { readPackageVersion } from '../tools/release_version.mjs';
import assert from 'node:assert/strict';
import test from 'node:test';
import {fixture, p, ok, tensors} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxProfilingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';
import {measureEngineTrace} from '../examples/common/EngineTrace.js';

const releaseVersion = readPackageVersion();
const wasmUrl = new URL(`../dist/${releaseVersion}/volvoxai.lite.wasm`, import.meta.url).pathname;
const graph = {format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [4]}},
  nodes: [{id: 'relu', opType: 'ReLU', inputs: {input: 'x'},
    outputs: {out: {tensor: 'y', dtype: 'float32', shape: [4]}}, params: {}}], outputs: ['y']};

test('engine spans land inside the page time they were recorded in', async t => {
  const f = await fixture({wasmUrl}); t.after(() => f.close());
  const profiling = new VxProfilingServiceClient(reportTransport(f.host));
  const compiled = await f.compile(await f.load(graph));
  const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
  const started = performance.now();
  const trace = ok(await profiling.startTrace(new p.StartTraceRequest({runtimeId: f.runtime.runtimeId,
    options: new p.TraceOptions({detail: p.TraceDetail.TRACE_DETAIL_NODES})})));
  const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId,
    inputs: tensors({x: {data: Float32Array.of(-1, 2, -3, 4), shape: [4]}})})));
  ok(await f.inference.releaseResult(new p.ResultRef(result)));
  const info = ok(await profiling.stopTrace(new p.TraceRef(trace)));
  const finished = performance.now();
  const {events} = ok(await profiling.listTraceEvents(new p.ListTraceEventsRequest({traceId: trace.traceId, pageSize: 4096})));
  const measures = [];
  const count = measureEngineTrace({info, events}, {performance: {measure: (name, options) => measures.push({name, ...options})}});
  assert.equal(count, events.filter(e => e.host).length);
  const execute = measures.find(m => m.name === 'Execute');
  const node = measures.find(m => m.name === '0 ReLU');
  for (const measure of [execute, node]) {
    assert.ok(measure.start >= started - 1 && measure.end <= finished + 1, `${measure.name} ${measure.start}..${measure.end}`);
    assert.equal(measure.detail.devtools.track, 'VolvoxAI');
  }
  assert.ok(node.start >= execute.start - 0.001 && node.end <= execute.end + 0.001, 'the node nests in its call');
});
