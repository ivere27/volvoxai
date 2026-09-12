#!/usr/bin/env node
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { fixture, p, ok, tensors, safetensors } from './proto_fixture.mjs';
const DEFAULT_WASM = fileURLToPath(new URL('../dist/0.4.0/volvoxai.wasm', import.meta.url));
function integerArgument(name, fallback, minimum, maximum) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw === undefined ? fallback : Number(raw.slice(prefix.length));
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`--${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function stringArgument(name, fallback) {
  const prefix = `--${name}=`;
  const raw = process.argv.slice(2).find((argument) => argument.startsWith(prefix));
  const value = raw === undefined ? fallback : raw.slice(prefix.length);
  if (!value) throw new Error(`--${name} must not be empty`);
  return value;
}

function rejectUnknownArguments() {
  const known = new Set(['backend', 'batch', 'inner', 'output', 'samples', 'warmup', 'wasm']);
  for (const argument of process.argv.slice(2)) {
    const match = /^--([^=]+)=/.exec(argument);
    if (!match || !known.has(match[1])) throw new Error(`unknown argument '${argument}'`);
  }
}

function backendArgument() {
  const backend = stringArgument('backend', 'wasm');
  if (backend !== 'wasm') {
    throw new Error("--backend must be 'wasm'");
  }
  return backend;
}

function percentile(values, fraction) {
  if (values.length === 0) throw new Error('latency measurement produced no samples');
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(ordered.length * fraction) - 1];
}

function latencySummary(values) {
  return Object.freeze({
    samples: values.length,
    median: percentile(values, 0.5),
    p95: percentile(values, 0.95),
  });
}

function memoryFields() {
  const usage = process.memoryUsage();
  return Object.freeze({
    heapUsedBytes: usage.heapUsed,
    arrayBufferBytes: usage.arrayBuffers,
  });
}

function memoryDelta(current, baseline) {
  return Object.freeze({
    heapUsedBytes: current.heapUsedBytes - baseline.heapUsedBytes,
    arrayBufferBytes: current.arrayBufferBytes - baseline.arrayBufferBytes,
  });
}

async function forceGarbageCollection() {
  if (typeof globalThis.gc !== 'function') {
    throw new Error(
      'memory deltas require `node --expose-gc --import tsx ' +
      'tools/runtime_batch_throughput.mjs`',
    );
  }
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
  await new Promise((resolve) => setImmediate(resolve));
  globalThis.gc();
}

async function retainedMemory() {
  await forceGarbageCollection();
  return memoryFields();
}

function deterministicValues(length, multiplier, divisor) {
  return Float32Array.from(
    { length },
    (_, index) => (((index * multiplier) % 257) - 128) / divisor,
  );
}

function matmulSnapshot(batchSize, innerSize, outputSize) {
  return {
    document: {format: 'volvox-graph/v1', dimensions: {B: {min: 1, max: batchSize}},
      inputs: {x: {dtype: 'float32', shape: ['B', innerSize]}},
      nodes: [{id: 'matmul', opType: 'MatMul', inputs: {input: 'x', weight: 'weight'},
        outputs: {out: {tensor: 'y', dtype: 'float32', shape: ['B', outputSize]}}, params: {}}], outputs: ['y']},
    weights: safetensors([{name: 'weight', shape: [innerSize, outputSize],
      data: deterministicValues(innerSize * outputSize, 29, 512)}]),
  };
}

function laneInputs(batchSize, innerSize) {
  return Object.freeze(Array.from({ length: batchSize }, (_, lane) => Object.freeze({
    x: Object.freeze({
      data: Float32Array.from(
        { length: innerSize },
        (_, index) => ((((index * 17) + (lane * 31)) % 257) - 128) / 256,
      ),
      shape: Object.freeze([1, innerSize]),
    }),
  })));
}


async function measure(options, scheduled, reference) {
  const f = await fixture({wasmUrl: options.wasmUrl, scheduled});
  try {
    const snapshot = matmulSnapshot(options.batch, options.inner, options.output);
    const model = await f.load(snapshot.document, snapshot.weights), compiled = await f.compile(model);
    const plan = ok(await f.planning.createGraphPlan(new p.CreateGraphPlanRequest({
      modelId: model.modelId,
    })));
    const info = ok(await f.planning.getGraphPlan(new p.GraphPlanRef(plan)));
    assert.ok(info.plan.independentBatch?.supported, 'C must prove independence');
    ok(await f.planning.releaseGraphPlan(new p.GraphPlanRef(plan)));
    const inputs = laneInputs(options.batch, options.inner), latencies = [], outputs = [];
    const before = await retainedMemory();
    for (let group = -options.warmup; group < options.samples; group++) {
      const start = performance.now(), results = [];
      if (scheduled) {
        const requests = [];
        for (const input of inputs) requests.push(ok(await f.scheduler.submit(new p.SubmitRequest({
          compiledModelId: compiled.compiledModelId, inputs: tensors(input)}))));
        for (const request of requests) {
          const state = await f.scheduler.waitRequest(new p.RequestRef(request));
          assert.equal(state.state, p.RequestState.REQUEST_STATE_SUCCEEDED, state.report.message);
          results.push(ok(await f.scheduler.takeRequestResult(new p.RequestRef(request))));
          ok(await f.scheduler.releaseRequest(new p.RequestRef(request)));
        }
      } else {
        for (const input of inputs) results.push(ok(await f.inference.run(new p.RunRequest({
          compiledModelId: compiled.compiledModelId, inputs: tensors(input)}))));
      }
      if (group >= 0) latencies.push(performance.now() - start);
      for (const [lane, result] of results.entries()) {
        const output = await f.read(result, 'y');
        if (reference) assert.deepEqual(output, reference[lane]);
        else if (group === 0) outputs[lane] = output;
        ok(await f.inference.releaseResult(new p.ResultRef(result)));
      }
    }
    const after = await retainedMemory();
    return {outputs, report: {mode: scheduled ? 'scheduled' : 'direct',
      latencyMs: latencySummary(latencies),
      logicalRequestsPerSecond: options.samples * options.batch * 1000 / latencies.reduce((a,b)=>a+b,0),
      independentBatch: info.plan.independentBatch,
      retainedMemoryDelta: memoryDelta(after, before),
      timing: 'generated API, including submission, wait, result transfer and request release; excluding ReadOutput',
    }};
  } finally { await f.close(); }
}
rejectUnknownArguments(); backendArgument();
const options = {wasmUrl: stringArgument('wasm', DEFAULT_WASM), batch: integerArgument('batch',8,2,32),
  inner: integerArgument('inner',512,8,4096), output: integerArgument('output',512,8,4096),
  samples: integerArgument('samples',21,3,1001), warmup: integerArgument('warmup',5,1,100)};
const direct = await measure(options, false), scheduled = await measure(options, true, direct.outputs);
console.log(JSON.stringify({schema: 'volvoxai.runtime-batch-throughput/v3', options,
  runtime: process.version, numericalEquality: true, scenarios: [direct.report, scheduled.report],
  physicalInvocationCount: 'verified separately by C/GPU instrumentation; not inferred from throughput',
}, (_key,value)=>typeof value === 'bigint' ? value.toString() : value, 2));
