#!/usr/bin/env node
/** Compare an archived release with the current one in separate processes.
 * node tools/profiling_performance.mjs /path/to/release/volvoxai.lite.js [on]
 * Run alternating baseline/current processes on an otherwise idle CPU. */
import assert from 'node:assert/strict';
import { resolve, dirname, join } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { readFile } from 'node:fs/promises';
import process from 'node:process';

const bundle = resolve(process.argv[2] ?? 'dist/0.6.0/volvoxai.lite.js');
const enabled = process.argv[3] === 'on';
const interactive = process.argv.includes('--paired');
const backend = process.argv.includes('--webgpu') ? 'webgpu' : 'wasm';
const warmup = backend === 'webgpu' ? 1000 : 2000;
const batchArgument = process.argv.find(value => value.startsWith('--batch='));
const batch = batchArgument ? Number(batchArgument.split('=')[1]) : backend === 'webgpu' ? 20 : 50;
assert.ok(Number.isSafeInteger(batch) && batch > 0);
const caseArgument = process.argv.find(value => value.startsWith('--case='));
const caseIndex = caseArgument ? Number(caseArgument.slice(7)) : null;
assert.ok(caseIndex === null || [0, 1, 2].includes(caseIndex));
const input = interactive ? createInterface({input: process.stdin}) : null;
const commands = input?.[Symbol.asyncIterator]();
async function command(expected) {
  const next = await commands.next();
  assert.equal(next.value, expected);
}
const api = await import(pathToFileURL(bundle));
const {pb: p} = api;
const originalFetch = globalThis.fetch;
globalThis.fetch = async (url, init) => String(url).startsWith('file:')
  ? new Response(await readFile(fileURLToPath(url))) : originalFetch(url, init);
if (backend === 'webgpu') {
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  assert.ok(adapter && !adapter.isFallbackAdapter && !adapter.info?.isFallbackAdapter);
  assert.ok(!/swiftshader|llvmpipe|lavapipe|software|\bcpu\b/i.test(JSON.stringify(adapter.info)));
}
const host = new (backend === 'webgpu' ? api.FullEngineHost : api.EngineHost)({
  onDiagnostic: message => console.error(message),
  wasmUrl: pathToFileURL(join(dirname(bundle), backend === 'webgpu' ? 'volvoxai.wasm' : 'volvoxai.lite.wasm'))});
const inference = new api.VxInferenceServiceClient(host);
const runtime = await inference.createRuntime(new p.CreateRuntimeRequest({cpuThreads: 1}));
const cases = [];
const percentile = (values, q) => [...values].sort((a, b) => a - b)[Math.ceil(values.length * q) - 1];
const summarize = values => ({p50Ms: percentile(values, .5), p95Ms: percentile(values, .95), samplesMs: values});
try {
  const workloads = [[8, 64], [128, 64], [64, 8192]];
  for (const [nodes, width] of caseIndex === null ? workloads : [workloads[caseIndex]]) {
    const document = {
      format: 'volvox-graph/v1', dimensions: {}, inputs: {x: {dtype: 'float32', shape: [width]}},
      nodes: Array.from({length: nodes}, (_, i) => ({id: `add${i}`, opType: 'Add',
        inputs: {a: i ? `out${i - 1}` : 'x', b: 'x'},
        outputs: {out: {tensor: `out${i}`, dtype: 'float32', shape: [width]}}, params: {}})),
      outputs: [`out${nodes - 1}`],
    };
    const model = await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
      package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(document))})}));
    const compiled = await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
      policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: [backend],
        operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
    assert.equal(compiled.report.backend, backend);
    assert.equal(compiled.report.route.attested, true);
    const context = await inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled));
    const request = new p.ExecuteRequest({contextId: context.contextId, inputs: [new p.Tensor({name: 'x',
      dtype: p.DataType.DATA_TYPE_F32, shape: [BigInt(width)], inline: new Uint8Array(new Float32Array(width).fill(1).buffer)})]});
    async function run(check = false) {
      const result = await inference.execute(request);
      if (backend === 'webgpu') {
        const deadline = performance.now() + 30000;
        for (;;) {
          const info = await inference.getResult(new p.ResultRef(result));
          if (info.state !== p.ResultState.RESULT_STATE_PENDING) {
            assert.equal(info.state, p.ResultState.RESULT_STATE_READY); break;
          }
          assert.ok(performance.now() < deadline);
          await new Promise(resolve => setTimeout(resolve, 0));
        }
      }
      if (check) {
        const output = await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name: document.outputs[0]}));
        const bytes = output.tensor.inline;
        const values = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        assert.equal(values.length, width);
        assert.ok(values.every(value => value === nodes + 1));
      }
      await inference.releaseResult(new p.ResultRef(result));
      // Each artifact is loaded with its own generated response definitions.
      return result.metrics?.hostTimeNs !== undefined ? Number(result.metrics.hostTimeNs) / 1e6 : result.metrics?.hostTimeMs ?? result.report.timings.executionTimeMs;
    }
    // Warm both the numerical path and V8's tiered host/codec compilation.
    for (let i = 0; i < warmup; i++) await run(i === warmup - 1);
    const profiling = enabled ? new api.VxProfilingServiceClient(host) : null;
    const trace = enabled ? await profiling.startTrace(new p.StartTraceRequest({runtimeId: runtime.runtimeId,
      detail: p.TraceDetail.TRACE_DETAIL_NODES, deviceTiming: true, capacityBytes: 64n * 1024n * 1024n})) : null;
    const wall = [], engine = [];
    if (interactive) console.log(JSON.stringify({ready: {nodes, width}}));
    for (let sample = 0; sample < 21; sample++) {
      if (interactive) await command('sample');
      let elapsed = 0;
      const start = performance.now();
      for (let i = 0; i < batch; i++) elapsed += await run();
      wall.push((performance.now() - start) / batch);
      engine.push(elapsed / batch);
      if (interactive) console.log(JSON.stringify({sample, wallMs: wall.at(-1), engineMs: engine.at(-1)}));
    }
    if (interactive) await command('next');
    let capture;
    if (trace) {
      let info = await profiling.stopTrace(new p.TraceRef(trace));
      const deadline = performance.now() + 30000;
      while (info.state === p.TraceState.TRACE_STATE_DRAINING) {
        assert.ok(performance.now() < deadline);
        await new Promise(resolve => setTimeout(resolve, 0));
        info = await profiling.getTrace(new p.TraceRef(trace));
      }
      assert.equal(info.state, p.TraceState.TRACE_STATE_READY); assert.equal(info.droppedEvents, 0n);
      capture = {eventCount: String(info.eventCount), droppedEvents: String(info.droppedEvents)};
      await profiling.releaseTrace(new p.TraceRef(trace));
    }
    cases.push({nodes, width, wall: summarize(wall), engine: summarize(engine), ...(capture ? {capture} : {})});
    await inference.releaseExecutionContext(new p.ExecutionContextRef(context));
    await inference.releaseCompiledModel(new p.CompiledModelRef(compiled));
    await inference.releaseModel(new p.ModelRef(model));
  }
} finally { input?.close(); await host.close(); globalThis.fetch = originalFetch; }
console.log(JSON.stringify({bundle, backend, profiling: enabled, warmup, samples: 21, batch, cases}));
