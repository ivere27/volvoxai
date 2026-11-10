import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {runGraph} from './parity/external/webgpu_device_bridge.mjs';
import {collectProfile} from '../tools/profile_report.mjs';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const multiply = shape => shape.reduce((a, b) => a * b, 1);
const rounded = Math.fround;
const channelWeights = Float32Array.of(1 + 2 ** -23, -0, 0.5);
const values = Float32Array.of(1 - 2 ** -23, -0, 0, Infinity, -Infinity,
  NaN, 1e-37, -1e-37, 3.25, -7.5, 1);

function fixture(op, {left = false, observable = false, multiuse = false,
  mutable = false, separateRounding = false, minimumWidth = 5} = {}) {
  const shape = ['B', 3, 'W'];
  const nodes = [{id: 'expand', opType: 'Expand', inputs: {input: 'compact'},
    outputs: {out: {tensor: 'expanded', dtype: 'float32', shape}}, params: {shape}}];
  const binary = {a: left ? 'expanded' : 'x', b: left ? 'x' : 'expanded'};
  nodes.push({id: 'binary', opType: op, inputs: binary,
    outputs: {out: {tensor: separateRounding ? 'product' : 'y', dtype: 'float32', shape}}, params: {}});
  if (separateRounding) nodes.push({id: 'add', opType: 'Add',
    inputs: {a: 'product', b: 'bias'},
    outputs: {out: {tensor: 'y', dtype: 'float32', shape}}, params: {}});
  if (multiuse) nodes.push({id: 'other_consumer', opType: 'Sub',
    inputs: {a: 'expanded', b: 'x'},
    outputs: {out: {tensor: 'other', dtype: 'float32', shape}}, params: {}});
  return {format: 'volvox-graph/v1',
    dimensions: {B: {min: 1, max: 2}, W: {min: minimumWidth, max: 17}},
    inputs: {x: {dtype: 'float32', shape},
      ...(mutable ? {compact: {dtype: 'float32', shape: [1, 3, 1]}} : {})},
    outputs: ['y', ...(observable ? ['expanded'] : []), ...(multiuse ? ['other'] : [])],
    nodes, __weights: {
      ...(!mutable ? {compact: {shape: [1, 3, 1], data: channelWeights}} : {}),
      ...(separateRounding ? {bias: {shape: [1], data: Float32Array.of(-1)}} : {})}};
}

function batches(mutable, minimumWidth = 5) {
  return [[1, 3, 5], [2, 3, 17], [1, 3, minimumWidth], [1, 3, 9]].map((shape, round) => {
    const data = Float32Array.from({length: multiply(shape)}, (_, index) =>
      values[(index + round * 3) % values.length]);
    data[0] = values[0];
    return [{name: 'x', shape, data}, ...(mutable ? [{name: 'compact',
      shape: [1, 3, 1], data: Float32Array.from(channelWeights,
        (value, index) => index === 0 ? rounded(value + round * 0.125) : value)}] : [])];
  });
}

function expected(batch, op, options) {
  const x = batch[0], weights = options.mutable ? batch[1].data : channelWeights;
  const expanded = Float32Array.from(x.data, (_, index) =>
    weights[Math.floor(index / x.shape[2]) % 3]);
  const y = Float32Array.from(x.data, (value, index) => {
    const a = options.left ? expanded[index] : value;
    const b = options.left ? value : expanded[index];
    const result = rounded(op === 'Add' ? a + b : op === 'Mul' ? a * b :
      op === 'Sub' ? a - b : a / b);
    return options.separateRounding ? rounded(result - 1) : result;
  });
  return {y, expanded, other: Float32Array.from(expanded,
    (value, index) => rounded(value - x.data[index]))};
}

function exact(actual, reference, message) {
  assert.equal(actual.length, reference.length, message);
  for (let i = 0; i < actual.length; i++) {
    if (Number.isNaN(reference[i])) assert.ok(Number.isNaN(actual[i]), `${message}: NaN ${i}`);
    else assert.ok(Object.is(actual[i], reference[i]),
      `${message}: element ${i}, actual ${actual[i]}, expected ${reference[i]}`);
  }
}

for (const profile of process.env.VX_EXPAND_FUSION_PROFILE
  ? [process.env.VX_EXPAND_FUSION_PROFILE] : ['inference', 'full']) {
  const stem = profile === 'inference' ? 'volvoxai.lite' : 'volvoxai';
  const api = await import(`../dist/${version}/${stem}.js`);
  const wasm = process.env.VX_EXPAND_FUSION_WASM ??
    fileURLToPath(new URL(`../dist/${version}/${stem}.wasm`, import.meta.url));
  for (const [label, op, options] of [
    ['immutable Add', 'Add', {}], ['immutable Mul', 'Mul', {}],
    ['immutable left Sub', 'Sub', {left: true}], ['immutable left Div', 'Div', {left: true}],
    ['observable Expand', 'Mul', {observable: true}],
    ['multiple consumers', 'Mul', {multiuse: true}],
    ['mutable compact input', 'Div', {mutable: true}],
    ['equal minimum extent declines', 'Mul', {minimumWidth: 1}],
    ['separate F32 Mul then Add', 'Mul', {separateRounding: true}],
  ]) {
    test(`${profile}: Expand/binary ${label} survives grow, shrink and rebinding`, async () => {
      let profiler, runtimeId, traceId, trace;
      class TracedClient extends api.VxInferenceServiceClient {
        constructor(host) {
          super(host);
          profiler = new api.VxProfilingServiceClient(host);
        }
        async createRuntime(request) {
          const response = await super.createRuntime(request);
          runtimeId = response.runtimeId;
          return response;
        }
        async createExecutionContext(request) {
          const response = await super.createExecutionContext(request);
          const started = await profiler.startTrace(new api.pb.StartTraceRequest({runtimeId,
            options: new api.pb.TraceOptions({detail: api.pb.TraceDetail.TRACE_DETAIL_NODES,
              executionPlans: true, capacityBytes: 1048576n})}));
          assert.equal(started.report?.status ?? 0, 0, started.report?.message);
          traceId = started.traceId;
          return response;
        }
        async releaseExecutionContext(request) {
          await profiler.stopTrace(new api.pb.TraceRef({traceId}));
          trace = await collectProfile(profiler, api.pb, traceId);
          await profiler.releaseTrace(new api.pb.TraceRef({traceId}));
          return super.releaseExecutionContext(request);
        }
      }
      const runner = {...api, InferenceWasmHost: api.EngineHost,
        VxInferenceServiceClient: TracedClient};
      const graph = fixture(op, options), inputs = batches(options.mutable, options.minimumWidth);
      const result = await runGraph(runner, {wasm}, graph, inputs, 'wasm');
      for (let round = 0; round < inputs.length; round++) {
        const oracle = expected(inputs[round], op, options);
        for (const name of graph.outputs)
          exact(result.rounds[round].get(name), oracle[name], `${label}/${name}/${round}`);
      }
      assert.equal(result.route.fallbackNodes ?? 0, 0);
      const expectedExpands = options.observable || options.multiuse || options.mutable ||
        options.minimumWidth === 1 ? inputs.length : 0;
      assert.equal(trace.events.filter(event => event.name === 'Expand').length,
        expectedExpands, 'trace must prove admission or deliberate decline');
      if (options.separateRounding)
        assert.equal(result.rounds[0].get('y')[0], 0, 'retain intermediate F32 rounding');
    });
  }
}
