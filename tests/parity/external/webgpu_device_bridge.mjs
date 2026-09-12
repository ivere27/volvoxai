#!/usr/bin/env -S deno run --unstable-webgpu --allow-read --allow-env --allow-ffi
/**
 * The device bridge, on a real GPU, against the engine's own CPU route.
 *
 * `gpu_bridge_execution` proves the descriptor is complete by executing it on
 * a CPU interpreter. That answers "does the wire carry enough"; it cannot
 * answer "does the WGSL the engine named compute the right thing on
 * hardware", because the interpreter never runs the shader.
 *
 * Each graph runs on the physical device and is compared with the C CPU route.
 * Three fixtures whose CPU routes are not qualified carry independent expected
 * values. A GPU run without a numerical oracle fails this suite.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';

function fail(message, report) {
  throw Object.assign(new Error(`[webgpu-device-bridge] ${message}`), { report });
}

function parseArguments(argv) {
  const options = { only: null };
  for (let index = 0; index < argv.length; index++) {
    const flag = argv[index];
    if (flag === '--dump') { options.dump = true; continue; }
    if (flag === '--cases') { options.cases = path.resolve(argv[++index]); continue; }
    if (!['--bundle', '--wasm', '--only'].includes(flag)) {
      fail(`unknown argument ${flag}`);
    }
    const value = argv[++index];
    if (!value || value.startsWith('--')) fail(`${flag} requires a value`);
    if (flag === '--only') options.only = value;
    else options[flag.slice(2)] = path.resolve(value);
  }
  for (const name of ['bundle', 'wasm']) {
    if (!options[name]) fail(`--${name} is required`);
  }
  return options;
}

/* Deno and Chrome both ship a software fallback; either would pass this suite
 * while proving nothing about a GPU. */
const SOFTWARE_ADAPTER =
  /\b(?:swiftshader|llvmpipe|lavapipe|softpipe|software|microsoft basic render|cpu)\b/i;

/**
 * A SafeTensors package.
 *
 * Quantized graphs cannot be written inline: an affine descriptor names a
 * scale and a zero-point tensor, and the loader looks both up in the weight
 * files. Building one here is what makes a quantized operator testable at all.
 */
function safetensors(entries = {}) {
  const DTYPE = new Map([
    [Float32Array, 'F32'], [Int8Array, 'I8'], [Uint8Array, 'U8'],
    [Int32Array, 'I32'],
  ]);
  const header = {};
  const blocks = [];
  let offset = 0;
  for (const [name, value] of Object.entries(entries)) {
    const dtype = DTYPE.get(value.data.constructor);
    if (!dtype) fail(`no SafeTensors dtype for ${name}`);
    const bytes = new Uint8Array(value.data.buffer, value.data.byteOffset,
      value.data.byteLength);
    header[name] = { dtype, shape: value.shape,
      data_offsets: [offset, offset + bytes.length] };
    blocks.push(bytes);
    offset += bytes.length;
  }
  const text = new TextEncoder().encode(JSON.stringify(header));
  const padding = (8 - text.length % 8) % 8;
  const total = 8 + text.length + padding + offset;
  const bytes = new Uint8Array(total);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(text.length + padding), true);
  bytes.set(text, 8);
  bytes.fill(0x20, 8 + text.length, 8 + text.length + padding);
  let at = 8 + text.length + padding;
  for (const block of blocks) { bytes.set(block, at); at += block.length; }
  return bytes;
}

/** An affine descriptor for one quantized tensor, per-tensor scheme. */
function affine(name, scale, zeroPoint, ByteArray) {
  return {
    weights: {
      [`${name}.scale`]: { shape: [1], data: Float32Array.from([scale]) },
      [`${name}.zero`]: { shape: [1], data: ByteArray.from([zeroPoint]) },
    },
    descriptor: {
      scheme: 'per_tensor',
      scale_tensor: `${name}.scale`,
      zero_point_tensor: `${name}.zero`,
    },
  };
}

async function requirePhysicalDevice() {
  if (!globalThis.navigator?.gpu) {
    fail('no navigator.gpu; run under Deno --unstable-webgpu');
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });
  if (!adapter) fail('no WebGPU adapter');
  const info = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
  const description = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(' ');
  if (!description || SOFTWARE_ADAPTER.test(description)) {
    fail(`refusing a software adapter: ${description || '<unnamed>'}`);
  }
  const device = await adapter.requestDevice();
  device.addEventListener?.('uncapturederror', (event) => {
    console.error(`[webgpu-device-bridge] uncaptured: ${event.error?.message}`);
  });
  return { device, description };
}

/** The portable built-in route this module ships under; the oracle. */
const HOST_BACKEND = 'wasm';

/**
 * Compile one graph onto one backend and run it to a settled answer.
 *
 * `webgpu` gets the real device; the host route gets the same module with no
 * bridge, so the two differ in nothing but the backend the policy requires.
 */
async function runGraph(api, options, graph, tensors, backend) {
  /* One tensor list, or several to execute in turn on one context. Repeated
   * execution is its own question: the host's residency rule decides whether
   * the second one is even uploaded. */
  const batches = Array.isArray(tensors[0]) ? tensors : [tensors];
  const diagnostics = [];
  /* Counting encodes is the only direct evidence the device was asked to do
   * anything. Matching numbers alone would also be produced by a route that
   * quietly fell back, and route evidence is a summary rather than a count. */
  let encodes = 0;
  let bridge;
  if (backend === 'webgpu') {
    const real = await api.createWebGPUHostBridge({
      device: options.device,
      onDiagnostic: (message) => diagnostics.push(message),
    });
    bridge = {
      imports: Object.freeze({
        ...real.imports,
        vx_gpu_encode: (dispatch) => {
          encodes++;
          return real.imports.vx_gpu_encode(dispatch);
        },
      }),
      attach: (memory) => real.attach(memory),
      waitForCompletion: () => real.waitForCompletion(),
      close: () => real.close(),
    };
  }

  /* `__weights` is the harness's own field, carrying the tensors an affine
   * descriptor names. The loader validates the document's field set, so it
   * must not travel inside the graph. */
  const { __weights: bundle, __safetensors: raw, ...document } = graph;
  const graphText = JSON.stringify(document);
  const weights = raw ?? safetensors(bundle ?? {});
  const wasmBytes = await readFile(options.wasm);
  const host = new (api.FullEngineHost ?? api.InferenceWasmHost)({
    wasmUrl: options.wasm,
    ...(bridge ? { gpuBridge: bridge } : {}),
    fetch: async (source) => {
      const name = String(source);
      if (name === 'graph.json') {
        return { ok: true, text: async () => graphText,
          json: async () => JSON.parse(graphText) };
      }
      if (name === 'model.safetensors') {
        return { ok: true, arrayBuffer: async () => weights.buffer.slice(
          weights.byteOffset, weights.byteOffset + weights.byteLength) };
      }
      if (name === options.wasm || name.endsWith('.wasm')) {
        return { ok: true, arrayBuffer: async () => wasmBytes.buffer.slice(
          wasmBytes.byteOffset, wasmBytes.byteOffset + wasmBytes.byteLength) };
      }
      return { ok: false, status: 404, statusText: name };
    },
    resolveModelSource: (graphPath, weightPaths) => ({
      graphUrl: graphPath, weightSources: weightPaths,
    }),
  });

  const inference = new api.VxInferenceServiceClient(host);
  const ok = api.pb.NativeStatus.NATIVE_STATUS_OK;
  const say = (stage, report) =>
    `${backend}/${stage} [${report?.code}] ${report?.message}` +
    (diagnostics.length ? ` | bridge: ${diagnostics.join(' ; ')}` : '');

  try {
    const runtime = await inference.createRuntime(new api.pb.CreateRuntimeRequest());
    if (runtime.report?.status !== ok) fail(say('createRuntime', runtime.report), runtime.report);
    const model = await inference.loadModel(new api.pb.LoadModelRequest({
      runtimeId: runtime.runtimeId,
      graphPath: 'graph.json',
      weightPaths: ['model.safetensors'],
      bankResidency: (options.bankResidency ?? []).map(value => new api.pb.BankResidency(value)),
    }));
    if (model.report?.status !== ok) fail(say('loadModel', model.report), model.report);
    const compiled = await inference.compileModel(new api.pb.CompileModelRequest({
      modelId: model.modelId,
      policy: new api.pb.BackendPolicy({
        mode: api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: [backend],
      }),
    }));
    if (compiled.report?.status !== ok) fail(say('compileModel', compiled.report), compiled.report);
    const context = await inference.createExecutionContext(
      new api.pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId,
      }));
    if (context.report?.status !== ok) fail(say('createContext', context.report), context.report);

    if (backend === 'webgpu' && options.inspectContext)
      await options.inspectContext({ api, inference, bridge, compiled, context, encodes: () => encodes });
    const rounds = [];
    let pending = 0;
    let route = null;
    for (const batch of batches) {
      const request = new api.pb.ExecuteRequest({
        contextId: context.contextId,
        inputs: batch.map(({ name, shape, data }) => new api.pb.Tensor({
          name,
          dtype: data instanceof Int32Array ? api.pb.DataType.DATA_TYPE_I32
            : data instanceof Int8Array ? api.pb.DataType.DATA_TYPE_I8
            : data instanceof Uint8Array ? api.pb.DataType.DATA_TYPE_U8
            : api.pb.DataType.DATA_TYPE_F32,
          shape: shape.map(BigInt),
          inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        })),
      });

      const result = await inference.execute(request);
      if (result.report?.status !== ok) fail(say('execute', result.report), result.report);
      await options.afterSubmit?.({ host, inference, bridge, result });
      if (result.state === api.pb.ResultState.RESULT_STATE_PENDING) {
        pending++;
        await bridge.waitForCompletion();
        const completed = await inference.getResult(new api.pb.ResultRef({ resultId: result.resultId }));
        if (completed.report?.status !== ok || completed.state !== api.pb.ResultState.RESULT_STATE_READY)
          fail(say('getResult', completed.report), completed.report);
        if (completed.executionId !== result.executionId) fail('completion changed execution identity');
      }
      route = result.report?.route;

      const outputs = new Map();
      for (const name of graph.outputs) {
        const read = await inference.readOutput(new api.pb.ReadOutputRequest({
          resultId: result.resultId, name,
        }));
        if (read.report?.status !== ok) fail(say(`readOutput(${name})`, read.report), read.report);
        const payload = read.tensor?.inline;
        if (!payload?.byteLength) fail(`${backend}: output '${name}' had no bytes`);
        const copy = payload.buffer.slice(
          payload.byteOffset, payload.byteOffset + payload.byteLength);
        const kind = read.tensor.dtype;
        outputs.set(name,
          kind === api.pb.DataType.DATA_TYPE_I32 ? new Int32Array(copy)
          : kind === api.pb.DataType.DATA_TYPE_I8 ? new Int8Array(copy)
          : kind === api.pb.DataType.DATA_TYPE_U8 ? new Uint8Array(copy)
          : new Float32Array(copy));
      }
      rounds.push(outputs);
      await options.afterRead?.({ api, inference, context, result, outputs, encodes: () => encodes });
      await inference.releaseResult(new api.pb.ResultRef({ resultId: result.resultId }));
    }
    await inference.releaseExecutionContext(new api.pb.ExecutionContextRef(context));
    await inference.releaseCompiledModel(new api.pb.CompiledModelRef(compiled));
    await inference.releaseModel(new api.pb.ModelRef(model));
    await inference.releaseRuntime(new api.pb.RuntimeRef(runtime));
    return { outputs: rounds[rounds.length - 1], rounds, route, pending,
      encodes, diagnostics };
  } finally {
    await host.close();
    await bridge?.waitForCompletion();
  }
}

/**
 * Compare one output against the host route.
 *
 * Float outputs are compared relatively. Byte outputs are compared in
 * quantization steps, and one step of disagreement is allowed -- not as
 * slack, but because it is the honest bound.
 *
 * A quantized shader computes `(a - za) * sa + (b - zb) * sb`, divides by the
 * output scale and rounds ties to even. A GPU is free to turn that division
 * into a multiply by the reciprocal, which changes the last bit; where the
 * result then lands exactly on a tie, the two routes round to neighbouring
 * integers. Verified on one element of QAdd: strict f32 gives -2.4999998,
 * whose fraction is just over one half, while the device landed just under.
 *
 * This is a property of the shaders as written, not of this backend -- the
 * TypeScript WebGPU route computes the same way. A wrong scale, a swapped
 * operand or a mis-read zero point moves many elements by much more than one
 * step, so the bound still fails loudly on a real defect, and the count of
 * differing elements is reported either way.
 */
function compare(label, actual, expected, tolerance) {
  if (!actual) fail(`${label}: no output`);
  if (actual.length !== expected.length) {
    fail(`${label}: got ${actual.length} values, expected ${expected.length}`);
  }
  const quantized = expected instanceof Int8Array || expected instanceof Uint8Array;
  let worst = 0;
  let differing = 0;
  for (let index = 0; index < expected.length; index++) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (difference !== 0) differing++;
    worst = Math.max(worst,
      quantized ? difference : difference / Math.max(1, Math.abs(expected[index])));
  }
  const bound = quantized ? 1 : tolerance;
  if (!(worst <= bound)) {
    fail(`${label}: ${quantized ? 'max_steps' : 'max_rel'} ${worst} exceeds ` +
      `${bound} (${differing} of ${expected.length} elements differ)`);
  }
  return quantized ? { steps: worst, differing, total: expected.length } : worst;
}

/* -- graph builders ------------------------------------------------------ */

function graphOf(nodes, inputs, outputs, weights, quantized) {
  const graph = { format: 'volvox-graph/v1', dimensions: {}, inputs, outputs, nodes };
  if (quantized) {
    graph.quantization = {
      format: 'volvox-affine-safetensors/v1', tensors: quantized,
    };
  }
  if (weights) graph.__weights = weights;
  return graph;
}

function unaryGraph(opType, shape, params = {}, dtype = 'float32') {
  return graphOf(
    [{ id: 'n', opType, inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype, shape } }, params }],
    { input0: { shape, dtype } },
    ['out0']);
}

function binaryGraph(opType, shape, dtype = 'float32') {
  return graphOf(
    [{ id: 'n', opType, inputs: { a: 'input0', b: 'input1' },
       outputs: { out: { tensor: 'out0', dtype, shape } }, params: {} }],
    { input0: { shape, dtype }, input1: { shape, dtype } },
    ['out0']);
}

/** A reshape-family node: the same elements under a different declared shape. */
function reshapeGraph(opType, shape, outShape, params = {}) {
  return graphOf(
    [{ id: 'n', opType, inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: outShape } },
       params }],
    { input0: { shape, dtype: 'float32' } },
    ['out0']);
}

const SHAPE = [1, 2, 3, 2];
const COUNT = 12;
const LEFT = Float32Array.from({ length: COUNT }, (_, i) => Math.sin(i) * 3 - 1);
const RIGHT = Float32Array.from({ length: COUNT }, (_, i) => Math.cos(i) * 2 + 3);
const INTS = Int32Array.from({ length: COUNT }, (_, i) => (i % 3) - 1);
const POOL_SHAPE = [1, 4, 4, 2];
const POOL_IN = Float32Array.from({ length: 32 }, (_, i) => Math.cos(i) * 5);

const one = (data, shape = SHAPE) => [{ name: 'input0', shape, data }];
const two = (a, b, shape = SHAPE) => [
  { name: 'input0', shape, data: a },
  { name: 'input1', shape, data: b },
];

/* The last axis of SHAPE is the feature width these normalise over. */
const FEATURE = SHAPE[SHAPE.length - 1];
const WEIGHT = Float32Array.from({ length: FEATURE }, (_, i) => 1 + i * 0.25);
const BIAS = Float32Array.from({ length: FEATURE }, (_, i) => i * 0.5 - 0.25);

function normGraph(opType, withBias) {
  const inputs = { input: 'input0', weight: 'input1' };
  const declared = {
    input0: { shape: SHAPE, dtype: 'float32' },
    input1: { shape: [FEATURE], dtype: 'float32' },
  };
  if (withBias) {
    inputs.bias = 'input2';
    declared.input2 = { shape: [FEATURE], dtype: 'float32' };
  }
  return graphOf(
    [{ id: 'n', opType, inputs,
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
       params: { eps: 1e-5 } }],
    declared, ['out0']);
}

function normInputs(withBias) {
  const tensors = [
    { name: 'input0', shape: SHAPE, data: LEFT },
    { name: 'input1', shape: [FEATURE], data: WEIGHT },
  ];
  if (withBias) tensors.push({ name: 'input2', shape: [FEATURE], data: BIAS });
  return tensors;
}

/* Same rank as the output with unit axes: the canonical shape contract
 * for the arithmetic operators does not admit a shorter operand. */
const LANE = [1, 1, 1, SHAPE[SHAPE.length - 1]];
const LANE_COUNT = LANE[LANE.length - 1];
const LANE_F32 = Float32Array.from({ length: LANE_COUNT }, (_, i) => 2 + i);
const LANE_I32 = Int32Array.from({ length: LANE_COUNT }, (_, i) => i);

function compareGraph(opType) {
  return graphOf(
    [{ id: 'n', opType, inputs: { a: 'input0', b: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'int32', shape: SHAPE } },
       params: {} }],
    { input0: { shape: SHAPE, dtype: 'int32' },
      input1: { shape: LANE, dtype: 'int32' } },
    ['out0']);
}

const compareInputs = () => [
  { name: 'input0', shape: SHAPE, data: INTS },
  { name: 'input1', shape: LANE, data: LANE_I32 },
];

/* The reduce shader folds the last axis, so the output drops it. */
function reduceGraph(opType) {
  const outShape = SHAPE.slice(0, -1);
  return graphOf(
    [{ id: 'n', opType, inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: outShape } },
       params: { axis: -1, keepdims: false } }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['out0']);
}

const BYTES = Int8Array.from({ length: 12 }, (_, i) => (i * 17) % 61 - 30);

/*
 * A byte-domain unary graph.
 *
 * The affine descriptor names a scale and a zero-point tensor and the loader
 * resolves both from the weight file, so a quantized case carries weights even
 * when the operator itself has none.
 */
function quantizedUnary(opType, inScale = 0.05, inZero = 0,
                       outScale = 0.05, outZero = 0) {
  const input = affine('input0', inScale, inZero, Int8Array);
  const output = affine('out0', outScale, outZero, Int8Array);
  return {
    ...graphOf(
      [{ id: 'n', opType, inputs: { input: 'input0' },
         outputs: { out: { tensor: 'out0', dtype: 'int8', shape: SHAPE } },
         params: {} }],
      { input0: { shape: SHAPE, dtype: 'int8' } }, ['out0']),
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: { input0: input.descriptor, out0: output.descriptor },
    },
    __weights: { ...input.weights, ...output.weights },
  };
}

/* Exact multiples of the 0.05 quantization step: no rounding tie. */
const EXACT_STEPS = Float32Array.from(
  { length: 12 }, (_, i) => ((i * 5) % 61 - 30) * 0.05);

const OTHER_BYTES = Int8Array.from({ length: 12 }, (_, i) => (i * 23) % 47 - 20);

function quantizedBinary(opType) {
  const a = affine('input0', 0.05, 0, Int8Array);
  const b = affine('input1', 0.04, 0, Int8Array);
  const out = affine('out0', 0.06, 0, Int8Array);
  return {
    ...graphOf(
      [{ id: 'n', opType, inputs: { a: 'input0', b: 'input1' },
         outputs: { out: { tensor: 'out0', dtype: 'int8', shape: SHAPE } },
         params: {} }],
      { input0: { shape: SHAPE, dtype: 'int8' },
        input1: { shape: SHAPE, dtype: 'int8' } }, ['out0']),
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: { input0: a.descriptor, input1: b.descriptor, out0: out.descriptor },
    },
    __weights: { ...a.weights, ...b.weights, ...out.weights },
  };
}

function poolGraph(opType) {
  return graphOf(
    [{ id: 'n', opType, inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 2, 2, 2] } },
       params: { kernel: [2, 2], stride: [2, 2], pads: [0, 0, 0, 0],
         data_layout: 'NHWC' } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' } },
    ['out0']);
}

/*
 * Every case uses C CPU parity. Three fixtures also carry independent answers.
 */
const CASES = [
  ['ReLU', unaryGraph('ReLU', SHAPE), one(LEFT)],
  ['Sigmoid', unaryGraph('Sigmoid', SHAPE), one(LEFT)],
  ['SiLU', unaryGraph('SiLU', SHAPE), one(LEFT)],
  ['GELU', unaryGraph('GELU', SHAPE), one(LEFT)],
  ['Add', binaryGraph('Add', SHAPE), two(LEFT, RIGHT)],
  ['Sub', binaryGraph('Sub', SHAPE), two(LEFT, RIGHT)],
  ['Mul', binaryGraph('Mul', SHAPE), two(LEFT, RIGHT)],
  ['Div', binaryGraph('Div', SHAPE), two(LEFT, RIGHT)],
  ['Clip', unaryGraph('Clip', SHAPE, { min: -0.5, max: 1.5 }), one(LEFT)],
  /* The engine aliases a shape node's output onto its input, so these reach
   * the device only when it cannot. The planner is written for that case; the
   * route still has to produce the right answer either way. */
  ['Identity', unaryGraph('Identity', SHAPE), one(LEFT), true],
  ['Reshape', reshapeGraph('Reshape', SHAPE, [1, 12], { shape: [1, 12] }), one(LEFT), true],
  ['Flatten', reshapeGraph('Flatten', SHAPE, [1, 12], { axis: 1 }), one(LEFT), true],
  ['MaxPool2D', poolGraph('MaxPool2D'), one(POOL_IN, POOL_SHAPE)],
  ['AveragePool2D', poolGraph('AveragePool2D'), one(POOL_IN, POOL_SHAPE), false,
    new Map([['out0', Float32Array.from([
      [0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15],
      [16, 18, 24, 26], [17, 19, 25, 27], [20, 22, 28, 30], [21, 23, 29, 31],
    ], (window) => window.reduce((sum, index) => sum + POOL_IN[index], 0) / 4)]])],
  ['Not', unaryGraph('Not', SHAPE, {}, 'int32'), one(INTS)],
  ['Softmax', unaryGraph('Softmax', SHAPE, { axis: -1 }), one(LEFT)],
  ['LogSoftmax', unaryGraph('LogSoftmax', SHAPE, { axis: -1 }), one(LEFT)],
  ['RMSNorm', normGraph('RMSNorm', false), normInputs(false)],
  ['LayerNorm', normGraph('LayerNorm', true), normInputs(true)],
  ['Transpose', graphOf(
    [{ id: 'n', opType: 'Transpose', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [2, 3, 2, 1] } },
       params: { perm: [3, 2, 1, 0] } }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['out0']),
    one(LEFT)],
  ...[[Int8Array, 'int8'], [Uint8Array, 'uint8']].map(([ArrayType, dtype]) => {
    const input = ArrayType.from({ length: 15 }, (_, i) => i * 37 - 129);
    const expected = ArrayType.from({ length: 15 }, (_, i) => input[(i % 3) * 5 + Math.floor(i / 3)]);
    const q = affine('input0', 0.02, 0, ArrayType);
    return [`Transpose-${dtype}-packed-tail`, graphOf(
      [{ id: 'n', opType: 'Transpose', inputs: { input: 'input0' },
         outputs: { out: { tensor: 'out0', dtype, shape: [5, 3] } }, params: { perm: [1, 0] } }],
      { input0: { shape: [3, 5], dtype } }, ['out0'], q.weights,
      { input0: q.descriptor, out0: q.descriptor }),
      one(input, [3, 5]), false, new Map([['out0', expected]])];
  }),
  /* Broadcast extents travel as a storage buffer from the C scratch pool. */
  ['Equal', compareGraph('Equal'), compareInputs()],
  ['GreaterOrEqual', compareGraph('GreaterOrEqual'), compareInputs()],
  ...['Equal', 'GreaterOrEqual'].flatMap(op => [[2, 2], [-1, 2]].map(([a, b]) => [
    `${op}/scalar/${a}/${b}`, graphOf(
      [{id: 'n', opType: op, inputs: {a: 'a', b: 'b'},
        outputs: {out: {tensor: 'out0', shape: [], dtype: 'int32'}}, params: {}}],
      {a: {shape: [], dtype: 'int32'}, b: {shape: [], dtype: 'int32'}}, ['out0']),
      [{name: 'a', shape: [], data: Int32Array.of(a)}, {name: 'b', shape: [], data: Int32Array.of(b)}],
      false, new Map([['out0', Int32Array.of(op === 'Equal' ? +(a === b) : +(a >= b))]])
  ])),
  ['PReLU', graphOf(
    [{ id: 'n', opType: 'PReLU', inputs: { input: 'input0', slope: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
       params: {} }],
    { input0: { shape: SHAPE, dtype: 'float32' },
      input1: { shape: [FEATURE], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: SHAPE, data: LEFT },
     { name: 'input1', shape: [FEATURE], data: WEIGHT }]],
  ['ReduceSum', reduceGraph('ReduceSum'), one(LEFT)],
  ['ReduceMean', reduceGraph('ReduceMean'), one(LEFT)],
  ['GlobalAveragePool', graphOf(
    [{ id: 'n', opType: 'GlobalAveragePool', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 1, 1, 2] } },
       params: {} }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' } }, ['out0']),
    one(POOL_IN, POOL_SHAPE)],
  ['Where', graphOf(
    [{ id: 'n', opType: 'Where',
       inputs: { condition: 'input0', a: 'input1', b: 'input2' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
       params: {} }],
    { input0: { shape: SHAPE, dtype: 'int32' },
      input1: { shape: SHAPE, dtype: 'float32' },
      input2: { shape: SHAPE, dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: SHAPE, data: INTS },
     { name: 'input1', shape: SHAPE, data: LEFT },
     { name: 'input2', shape: SHAPE, data: RIGHT }]],
  ['Expand', graphOf(
    [{ id: 'n', opType: 'Expand', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
       params: { shape: SHAPE } }],
    { input0: { shape: LANE, dtype: 'float32' } }, ['out0']),
    one(LANE_F32, LANE)],
  ['BatchNorm2D', graphOf(
    [{ id: 'n', opType: 'BatchNorm2D',
       inputs: { input: 'input0', weight: 'input1', bias: 'input2',
         running_mean: 'input3', running_var: 'input4' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: POOL_SHAPE } },
       params: { eps: 1e-5 } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' },
      input1: { shape: [2], dtype: 'float32' },
      input2: { shape: [2], dtype: 'float32' },
      input3: { shape: [2], dtype: 'float32' },
      input4: { shape: [2], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: POOL_SHAPE, data: POOL_IN },
     { name: 'input1', shape: [2], data: WEIGHT },
     { name: 'input2', shape: [2], data: BIAS },
     { name: 'input3', shape: [2], data: Float32Array.from([0.5, -0.25]) },
     { name: 'input4', shape: [2], data: Float32Array.from([2, 3]) }]],
  ['GroupNorm', graphOf(
    [{ id: 'n', opType: 'GroupNorm',
       inputs: { input: 'input0', weight: 'input1', bias: 'input2' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: POOL_SHAPE } },
       params: { num_groups: 2, eps: 1e-5 } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' },
      input1: { shape: [2], dtype: 'float32' },
      input2: { shape: [2], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: POOL_SHAPE, data: POOL_IN },
     { name: 'input1', shape: [2], data: WEIGHT },
     { name: 'input2', shape: [2], data: BIAS }]],
  ['UpsampleNearest2D', graphOf(
    [{ id: 'n', opType: 'UpsampleNearest2D', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 8, 8, 2] } },
       params: { data_layout: 'NHWC' } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' } }, ['out0']),
    one(POOL_IN, POOL_SHAPE)],
  ['ArgMax', graphOf(
    [{ id: 'n', opType: 'ArgMax', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'int32', shape: [1, 2, 3] } },
       params: { axis: -1, keepdims: false } }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['out0']),
    one(LEFT)],
  ['Embedding', graphOf(
    [{ id: 'n', opType: 'Embedding',
       inputs: { input: 'input0', weight: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 4, 3] } },
       params: {} }],
    { input0: { shape: [1, 4], dtype: 'int32' },
      input1: { shape: [5, 3], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: [1, 4], data: Int32Array.from([0, 2, 4, 1]) },
     { name: 'input1', shape: [5, 3],
       data: Float32Array.from({ length: 15 }, (_, i) => i * 0.5 - 1) }]],
  ['Slice', graphOf(
    [{ id: 'n', opType: 'Slice', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 1, 3, 2] } },
       params: { axes: [1], starts: [1], ends: [2], steps: [1] } }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['out0']),
    one(LEFT)],
  /* A row-major weight with a bias: the shape the dense shader indexes. */
  ['MatMul', graphOf(
    [{ id: 'n', opType: 'MatMul',
       inputs: { input: 'input0', weight: 'input1', bias: 'input2' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 6, 4] } },
       params: { weight_layout: 'din_dout' } }],
    { input0: { shape: [1, 6, 2], dtype: 'float32' },
      input1: { shape: [2, 4], dtype: 'float32' },
      input2: { shape: [4], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: [1, 6, 2], data: LEFT },
     { name: 'input1', shape: [2, 4],
       data: Float32Array.from({ length: 8 }, (_, i) => i * 0.25 - 1) },
     { name: 'input2', shape: [4], data: Float32Array.from([0.5, -1, 2, 0]) }]],
  ['Resize/nearest', graphOf(
    [{ id: 'n', opType: 'Resize', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 8, 8, 2] } },
       params: { mode: 'nearest', data_layout: 'NHWC' } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' } }, ['out0']),
    one(POOL_IN, POOL_SHAPE)],
  ['Resize/linear', graphOf(
    [{ id: 'n', opType: 'Resize', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 8, 8, 2] } },
       params: { mode: 'linear', data_layout: 'NHWC' } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' } }, ['out0']),
    one(POOL_IN, POOL_SHAPE)],
  ['Gather', graphOf(
    [{ id: 'n', opType: 'Gather',
       inputs: { input: 'input0', indices: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [3, 3] } },
       params: { axis: 0 } }],
    { input0: { shape: [5, 3], dtype: 'float32' },
      input1: { shape: [3], dtype: 'int32' } }, ['out0']),
    [{ name: 'input0', shape: [5, 3],
       data: Float32Array.from({ length: 15 }, (_, i) => i * 0.5 - 1) },
     { name: 'input1', shape: [3], data: Int32Array.from([4, 0, 2]) }]],
  /* One dispatch per operand: the pass records them in order. */
  ['Concat', graphOf(
    [{ id: 'n', opType: 'Concat',
       inputs: { input0: 'input0', input1: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 4, 3, 2] } },
       params: { axis: 1 } }],
    { input0: { shape: SHAPE, dtype: 'float32' },
      input1: { shape: SHAPE, dtype: 'float32' } }, ['out0']),
    two(LEFT, RIGHT)],
  ['GatherElements', graphOf(
    [{ id: 'n', opType: 'GatherElements',
       inputs: { input: 'input0', indices: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [3, 2] } },
       params: { axis: 1 } }],
    { input0: { shape: [3, 4], dtype: 'float32' },
      input1: { shape: [3, 2], dtype: 'int32' } }, ['out0']),
    [{ name: 'input0', shape: [3, 4],
       data: Float32Array.from({ length: 12 }, (_, i) => i * 0.5 - 2) },
     { name: 'input1', shape: [3, 2], data: Int32Array.from([0, 3, 2, 1, 3, 0]) }], false,
    new Map([['out0', Float32Array.of(-2, -0.5, 1, 0.5, 3.5, 2)]])],
  ['BatchMatMul', graphOf(
    [{ id: 'n', opType: 'BatchMatMul', inputs: { a: 'input0', b: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [2, 3, 2] } },
       params: {} }],
    { input0: { shape: [2, 3, 4], dtype: 'float32' },
      input1: { shape: [2, 4, 2], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: [2, 3, 4],
       data: Float32Array.from({ length: 24 }, (_, i) => Math.sin(i) * 2) },
     { name: 'input1', shape: [2, 4, 2],
       data: Float32Array.from({ length: 16 }, (_, i) => Math.cos(i) + 0.5) }]],
  /* HWIO weights: the layout the shader indexes without a repack. */
  ['Conv2D', graphOf(
    [{ id: 'n', opType: 'Conv2D',
       inputs: { input: 'input0', weight: 'input1', bias: 'input2' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 3, 3, 3] } },
       params: { stride: [1, 1], pads: [0, 0, 0, 0], dilation: [1, 1],
         groups: 1, weight_layout: 'HWIO', data_layout: 'NHWC' } }],
    { input0: { shape: POOL_SHAPE, dtype: 'float32' },
      input1: { shape: [2, 2, 2, 3], dtype: 'float32' },
      input2: { shape: [3], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: POOL_SHAPE, data: POOL_IN },
     { name: 'input1', shape: [2, 2, 2, 3],
       data: Float32Array.from({ length: 24 }, (_, i) => Math.sin(i) * 0.3) },
     { name: 'input2', shape: [3], data: Float32Array.from([0.1, -0.2, 0.05]) }]],
  ['SDPA', graphOf(
    [{ id: 'n', opType: 'SDPA', inputs: { qkv: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 4, 4] } },
       params: { heads: 2, scale: 1.0, causal: true } }],
    { input0: { shape: [1, 4, 12], dtype: 'float32' } }, ['out0']),
    one(Float32Array.from({ length: 48 }, (_, i) => Math.sin(i * 0.7) * 0.5),
        [1, 4, 12])],
  ['CrossSDPA', graphOf(
    [{ id: 'n', opType: 'CrossSDPA',
       inputs: { q: 'input0', k: 'input1', v: 'input2' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 3, 4] } },
       params: { heads: 2, scale: 1.0, causal: false } }],
    { input0: { shape: [1, 3, 4], dtype: 'float32' },
      input1: { shape: [1, 5, 4], dtype: 'float32' },
      input2: { shape: [1, 5, 4], dtype: 'float32' } }, ['out0']),
    [{ name: 'input0', shape: [1, 3, 4],
       data: Float32Array.from({ length: 12 }, (_, i) => Math.sin(i) * 0.4) },
     { name: 'input1', shape: [1, 5, 4],
       data: Float32Array.from({ length: 20 }, (_, i) => Math.cos(i) * 0.3) },
     { name: 'input2', shape: [1, 5, 4],
       data: Float32Array.from({ length: 20 }, (_, i) => Math.sin(i * 0.5)) }]],
  /* Quantized: the affine descriptors travel in the weight file, which is
   * what a quantized graph needs and an inline one cannot express. */
  ['QSiLU', quantizedUnary('QSiLU'), one(BYTES)],
  ['QGELU', quantizedUnary('QGELU'), one(BYTES)],
  ['QAdd', quantizedBinary('QAdd'),
    [{ name: 'input0', shape: SHAPE, data: BYTES },
     { name: 'input1', shape: SHAPE, data: OTHER_BYTES }]],
  ['RequantizeLinear', quantizedUnary('RequantizeLinear', 0.05, 0, 0.03, 4),
    one(BYTES)],
  /*
   * Quantize and dequantize, round trip.
   *
   * The byte tensor is declared an output and is also consumed by the second
   * node, which is what lets it be one: the engine refuses a quantized tensor
   * that is a graph output and nothing else. Declaring both outputs exercises
   * atomic completion of both output snapshots from one submission.
   *
   * The input values are exact multiples of the scale, so neither route lands
   * on a rounding tie and the two must agree exactly. A tie would be a
   * legitimate one-step disagreement and would say nothing about the planner.
   */
  /* Two float outputs from one node: the multi-output readback path. */
  ['Split', graphOf(
    [{ id: 'n', opType: 'Split', inputs: { input: 'input0' },
       outputs: {
         out0: { tensor: 'left', dtype: 'float32', shape: [1, 1, 3, 2] },
         out1: { tensor: 'right', dtype: 'float32', shape: [1, 1, 3, 2] },
       },
       params: { axis: 1, split: [1, 1] } }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['left', 'right']),
    one(LEFT), false, new Map([['left', LEFT.slice(0, 6)], ['right', LEFT.slice(6)]])],
  /*
   * A byte dense layer. The weight carries a per-axis descriptor -- one scale
   * and zero point per output channel -- which is what makes the requant
   * multiplier an array rather than a number.
   */
  ['QLinear', graphOf(
    [{ id: 'n', opType: 'QLinear',
       inputs: { input: 'input0', weight: 'w', bias: 'b' },
       outputs: { out: { tensor: 'out0', dtype: 'int8', shape: [1, 3, 4] } },
       params: {} }],
    { input0: { shape: [1, 3, 2], dtype: 'int8' } }, ['out0'],
    { w: { shape: [4, 2], data: Int8Array.from([3, -2, 1, 4, -5, 2, 0, 6]) },
      b: { shape: [4], data: Int32Array.from([10, -20, 5, 0]) },
      'input0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'input0.zero': { shape: [1], data: Int8Array.from([0]) },
      'w.scale': { shape: [4],
        data: Float32Array.from([0.01, 0.02, 0.015, 0.008]) },
      'w.zero': { shape: [4], data: Int8Array.from([0, 0, 0, 0]) },
      'out0.scale': { shape: [1], data: Float32Array.from([0.02]) },
      'out0.zero': { shape: [1], data: Int8Array.from([0]) } },
    { input0: { scheme: 'per_tensor', scale_tensor: 'input0.scale',
                zero_point_tensor: 'input0.zero' },
      w: { scheme: 'per_axis', axis: 0, scale_tensor: 'w.scale',
           zero_point_tensor: 'w.zero' },
      out0: { scheme: 'per_tensor', scale_tensor: 'out0.scale',
              zero_point_tensor: 'out0.zero' } }),
    [{ name: 'input0', shape: [1, 3, 2],
       data: Int8Array.from([10, -20, 30, 5, -15, 25]) }]],
  ['QArgMax', graphOf(
    [{ id: 'n', opType: 'QArgMax', inputs: { input: 'input0' },
       outputs: { out: { tensor: 'out0', dtype: 'int32', shape: [1, 2, 3] } },
       params: { axis: -1 } }],
    { input0: { shape: SHAPE, dtype: 'int8' } }, ['out0'],
    { 'input0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'input0.zero': { shape: [1], data: Int8Array.from([0]) } },
    { input0: { scheme: 'per_tensor', scale_tensor: 'input0.scale',
                zero_point_tensor: 'input0.zero' } }),
    one(BYTES)],
  ['QBatchMatMul', graphOf(
    [{ id: 'n', opType: 'QBatchMatMul', inputs: { a: 'input0', b: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'int8', shape: [2, 2, 2] } },
       params: {} }],
    { input0: { shape: [2, 2, 3], dtype: 'int8' },
      input1: { shape: [2, 3, 2], dtype: 'int8' } }, ['out0'],
    { 'input0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'input0.zero': { shape: [1], data: Int8Array.from([0]) },
      'input1.scale': { shape: [1], data: Float32Array.from([0.04]) },
      'input1.zero': { shape: [1], data: Int8Array.from([0]) },
      'out0.scale': { shape: [1], data: Float32Array.from([0.06]) },
      'out0.zero': { shape: [1], data: Int8Array.from([0]) } },
    { input0: { scheme: 'per_tensor', scale_tensor: 'input0.scale',
                zero_point_tensor: 'input0.zero' },
      input1: { scheme: 'per_tensor', scale_tensor: 'input1.scale',
                zero_point_tensor: 'input1.zero' },
      out0: { scheme: 'per_tensor', scale_tensor: 'out0.scale',
              zero_point_tensor: 'out0.zero' } }),
    [{ name: 'input0', shape: [2, 2, 3],
       data: Int8Array.from([4, -3, 2, 5, 1, -6, 7, 2, -1, 3, -4, 8]) },
     { name: 'input1', shape: [2, 3, 2],
       data: Int8Array.from([2, -1, 3, 4, -2, 5, 1, 6, -3, 2, 4, -5]) }]],
  ['QMaskedMean', graphOf(
    [{ id: 'n', opType: 'QMaskedMean',
       inputs: { input: 'input0', mask: 'input1' },
       outputs: { out: { tensor: 'out0', dtype: 'int8', shape: [2, 3] } },
       params: {} }],
    { input0: { shape: [2, 2, 3], dtype: 'int8' },
      input1: { shape: [2, 2], dtype: 'int32' } }, ['out0'],
    { 'input0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'input0.zero': { shape: [1], data: Int8Array.from([0]) },
      'out0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'out0.zero': { shape: [1], data: Int8Array.from([0]) } },
    { input0: { scheme: 'per_tensor', scale_tensor: 'input0.scale',
                zero_point_tensor: 'input0.zero' },
      out0: { scheme: 'per_tensor', scale_tensor: 'out0.scale',
              zero_point_tensor: 'out0.zero' } }),
    [{ name: 'input0', shape: [2, 2, 3],
       data: Int8Array.from([10, -20, 30, 5, -15, 25, 40, -5, 12, -8, 18, 3]) },
     { name: 'input1', shape: [2, 2], data: Int32Array.from([1, 1, 1, 0]) }]],
  /* Two dispatches from one node: reduce, then apply. */
  ['QLayerNorm', graphOf(
    [{ id: 'n', opType: 'QLayerNorm',
       inputs: { input: 'input0', weight: 'gamma', bias: 'beta' },
       outputs: { out: { tensor: 'out0', dtype: 'int8', shape: SHAPE } },
       params: { eps: 1e-5 } }],
    { input0: { shape: SHAPE, dtype: 'int8' } }, ['out0'],
    { gamma: { shape: [2], data: Float32Array.from([1.25, 0.75]) },
      beta: { shape: [2], data: Float32Array.from([0.1, -0.2]) },
      'input0.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'input0.zero': { shape: [1], data: Int8Array.from([0]) },
      'out0.scale': { shape: [1], data: Float32Array.from([0.04]) },
      'out0.zero': { shape: [1], data: Int8Array.from([0]) } },
    { input0: { scheme: 'per_tensor', scale_tensor: 'input0.scale',
                zero_point_tensor: 'input0.zero' },
      out0: { scheme: 'per_tensor', scale_tensor: 'out0.scale',
              zero_point_tensor: 'out0.zero' } }),
    one(BYTES)],
  ['Quantize/Dequantize', graphOf(
    [{ id: 'q', opType: 'QuantizeLinear',
       inputs: { input: 'input0', scale: 'q.scale', zero_point: 'q.zero' },
       outputs: { out: { tensor: 'mid', dtype: 'int8', shape: SHAPE } },
       params: {} },
     { id: 'd', opType: 'DequantizeLinear',
       inputs: { input: 'mid', scale: 'q.scale', zero_point: 'q.zero' },
       outputs: { out: { tensor: 'out0', dtype: 'float32', shape: SHAPE } },
       params: {} }],
    { input0: { shape: SHAPE, dtype: 'float32' } }, ['mid', 'out0'],
    { 'q.scale': { shape: [1], data: Float32Array.from([0.05]) },
      'q.zero': { shape: [1], data: Int8Array.from([0]) } },
    { mid: { scheme: 'per_tensor', scale_tensor: 'q.scale',
             zero_point_tensor: 'q.zero' } }),
    one(EXACT_STEPS)],
];

// Canonical byte attention and per-row embedding exercise the C planners
// with affine metadata, signed values, masks and repeated identifiers.
{
  const weights={},quantization={};
  for(const [name,scale] of [['q',.03],['k',.04],['v',.05],['out0',.02]]) {
    weights[name+'.scale']={shape:[1],data:Float32Array.of(scale)};
    weights[name+'.zero']={shape:[1],data:Int8Array.of(-3)};
    quantization[name]={scheme:'per_tensor',scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
  }
  const inputs={q:{shape:[2,3,8],dtype:'int8'},k:{shape:[2,4,8],dtype:'int8'},v:{shape:[2,4,8],dtype:'int8'},mask:{shape:[2,4],dtype:'int32'}};
  const tensors=Object.entries(inputs).map(([name,descriptor])=>({name,shape:descriptor.shape,data:name==='mask'
    ?Int32Array.of(1,0,1,1,0,1,1,1)
    :Int8Array.from({length:descriptor.shape.reduce((a,b)=>a*b,1)},(_,i)=>(i*7+name.charCodeAt(0))%39-20)}));
  CASES.push(['QSDPA-masked',graphOf([{id:'n',opType:'QSDPA',inputs:{q:'q',k:'k',v:'v',mask:'mask'},
    outputs:{out:{tensor:'out0',dtype:'int8',shape:[2,3,8]}},params:{heads:2,causal:true}}],inputs,['out0'],weights,quantization),tensors]);
}
{
  const weights={w:{shape:[5,3],data:Int8Array.from({length:15},(_,i)=>i-7)},
    'w.scale':{shape:[5],data:Float32Array.of(.01,.02,.03,.04,.05)},
    'w.zero':{shape:[5],data:Int8Array.of(-2,-1,0,1,2)},
    'out0.scale':{shape:[1],data:Float32Array.of(.02)},'out0.zero':{shape:[1],data:Int8Array.of(-3)}};
  CASES.push(['QEmbedding-repeated',graphOf([{id:'n',opType:'QEmbedding',inputs:{input:'input0',weight:'w'},
    outputs:{out:{tensor:'out0',dtype:'int8',shape:[2,3,3]}},params:{}}],
    {input0:{shape:[2,3],dtype:'int32'}},['out0'],weights,
    {w:{scheme:'per_axis',axis:0,scale_tensor:'w.scale',zero_point_tensor:'w.zero'},
     out0:{scheme:'per_tensor',scale_tensor:'out0.scale',zero_point_tensor:'out0.zero'}}),
    [{name:'input0',shape:[2,3],data:Int32Array.of(0,4,2,2,1,4)}]]);
}

// Variant coverage is separate from operator-name coverage. These answers
// are assembled independently of either runtime implementation.
for (const [dtype, Array_] of [['float32',Float32Array],['int32',Int32Array]]) {
  const values=Array_.of(dtype==='int32'?-2147483648:-1e35,-2,0,2,dtype==='int32'?2147483647:1e35);
  CASES.push(['Clip-default-'+dtype,unaryGraph('Clip',[5],{},dtype),one(values,[5]),false,new Map([['out0',values]])]);
  const lower=Array_.of(dtype==='int32'?-1:-.4),upper=Array_.of(dtype==='int32'?1:.6);
  CASES.push(['Clip-tensor-bounds-'+dtype,graphOf([{id:'n',opType:'Clip',inputs:{input:'input0',min:'lower',max:'upper'},
    outputs:{out:{tensor:'out0',shape:[5],dtype}},params:{min:-100,max:100}}],
    {input0:{shape:[5],dtype}},['out0'],{lower:{shape:[1],data:lower},upper:{shape:[1],data:upper}}),
    one(values,[5]),false,new Map([['out0',Array_.from(values,x=>Math.max(lower[0],Math.min(upper[0],x)))]])]);
  for(const sigmoid of dtype==='float32'?[false,true]:[false]) {
    const x=Array_.of(-2,3,4,-1),y=Array_.of(.5,2);
    CASES.push(['Concat2-'+dtype+(sigmoid?'-sigmoid':''),graphOf([{id:'n',opType:'Concat2',inputs:{b:'right',a:'left'},
      outputs:{out:{tensor:'out0',dtype,shape:[2,3]}},params:{axis:-1,sigmoid}}],
      {left:{shape:[2,2],dtype},right:{shape:[2,1],dtype}},['out0']),
      [{name:'left',shape:[2,2],data:x},{name:'right',shape:[2,1],data:y}],false,
      new Map([['out0',Array_.from([x[0],x[1],y[0],x[2],x[3],y[1]],v=>sigmoid?1/(1+Math.exp(-v)):v)]])]);
  }
}
for(const count of [32,33,65,257]) for(const [dtype,Array_] of [['float32',Float32Array],['int32',Int32Array],['int8',Int8Array],['uint8',Uint8Array]]) {
  const inputs={},ports={},tensors=[],weights={},quantization={};
  for(let i=count-1;i>=0;--i){const name='tensor'+i;inputs[name]={shape:[2,1],dtype};ports['input'+i]=name;tensors.push({name,shape:[2,1],data:Array_.of(i+1,i+41)});}
  if(dtype==='int8'||dtype==='uint8')for(const name of [...Object.keys(inputs),'out0']) {
    weights[name+'.scale']={shape:[1],data:Float32Array.of(.02)};
    weights[name+'.zero']={shape:[1],data:Array_.of(0)};
    quantization[name]={scheme:'per_tensor',scale_tensor:name+'.scale',zero_point_tensor:name+'.zero'};
  }
  CASES.push(['Concat-'+count+'-'+dtype,graphOf([{id:'n',opType:'Concat',inputs:ports,
    outputs:{out:{tensor:'out0',dtype,shape:[2,count]}},params:{axis:1}}],inputs,['out0'],weights,
    Object.keys(quantization).length?quantization:undefined),tensors,false,
    new Map([['out0',Array_.from({length:count*2},(_,i)=>i<count?i+1:i-count+41)]])]);
}
for(const scalar of [false,true]) {
  const shape=scalar?[3]:[2,3],ids=scalar?Int32Array.of(-1):Int32Array.of(-1,0),indexShape=scalar?[]:[2];
  const outputShape=scalar?[]:[2,2],values=Float32Array.from({length:scalar?3:6},(_,i)=>i+1);
  CASES.push(['Gather-negative'+(scalar?'-scalar':''),graphOf([{id:'n',opType:'Gather',
    inputs:{input:'input0',indices:'indices'},outputs:{out:{tensor:'out0',dtype:'float32',shape:outputShape}},params:{axis:-1}}],
    {input0:{shape,dtype:'float32'}},['out0'],{indices:{shape:indexShape,data:ids}}),
    one(values,shape),false,new Map([['out0',Float32Array.from(scalar?[3]:[3,1,6,4])]])]);
}
// More than 255 outputs also detects truncation in dependency counters.
for(const count of [65,257]) for(const [dtype,Array_] of [['float32',Float32Array],['int32',Int32Array]]) {
  const outputs={},expected=new Map(),values=Array_.from({length:count*2},(_,i)=>i*3-17);
  for(let i=0;i<count;i++) {
    outputs['out'+i]={tensor:'piece'+i,dtype,shape:[2,1]};
    expected.set('piece'+i,Array_.of(values[i],values[count+i]));
  }
  CASES.push(['Split-'+count+'-'+dtype,graphOf([{id:'n',opType:'Split',inputs:{input:'input0'},
    outputs,params:{axis:1,split:Array(count).fill(1)}}],{input0:{shape:[2,count],dtype}},[...expected.keys()]),
    one(values,[2,count]),false,expected]);
}
{
  const source=CASES.find(([name])=>name==='Conv2D');
  const graph=structuredClone(source[1]);delete graph.nodes[0].inputs.bias;delete graph.inputs.input2;
  CASES.push(['Conv2D-no-bias',graph,source[2].slice(0,2)]);
}

/*
 * A case cut out of an exported model.
 *
 * Some operators are hard to author a graph for by hand -- the engine refuses
 * a spelling, a missing parameter or a relation between quantization scales
 * that no error message names. A node lifted out of a model the engine already
 * runs is valid by construction, so `extract_operator_case.py` writes one and
 * this reads it back.
 */
const DTYPE_ARRAY = Object.freeze({
  float32: Float32Array, int32: Int32Array, int8: Int8Array, uint8: Uint8Array,
});

function extractedCase(directory, stem) {
  const base = path.join(directory, stem);
  let graph;
  let raw;
  let plan;
  try {
    graph = JSON.parse(readFileSync(`${base}.graph.json`, 'utf8'));
    raw = new Uint8Array(readFileSync(`${base}.safetensors`));
    plan = JSON.parse(readFileSync(`${base}.inputs.json`, 'utf8'));
  } catch {
    return null;
  }
  graph.__safetensors = raw;
  const tensors = Object.entries(plan).map(([name, descriptor]) => {
    const Array_ = DTYPE_ARRAY[descriptor.dtype];
    if (!Array_) fail(`${stem}: no array for ${descriptor.dtype}`);
    /* Deterministic and spread across the byte range, so a wrong scale or a
     * transposed weight moves many elements rather than none. */
    return { name, shape: descriptor.shape,
      data: Array_.from({ length: descriptor.elements },
        (_, index) => ((index * 13) % 61) - 30) };
  });
  return [stem, graph, tensors];
}

async function checkResultLifetimes({ api, inference, bridge, compiled, context }) {
  const pb = api.pb;
  const ok = (response) => {
    assert.equal(response.report?.status, pb.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
    return response;
  };
  const submit = async (owner, value) => ok(await inference.execute(new pb.ExecuteRequest({
    contextId: owner.contextId, inputs: [new pb.Tensor({ name: 'input0',
      shape: SHAPE.map(BigInt), dtype: pb.DataType.DATA_TYPE_F32,
      inline: new Uint8Array(new Float32Array(12).fill(value).buffer) })],
  })));
  const first = await submit(context, 2);
  const secondContext = ok(await inference.createExecutionContext(new pb.CreateExecutionContextRequest(compiled)));
  const second = await submit(secondContext, 9);
  const third = await submit(context, 3);
  await inference.releaseExecutionContext(new pb.ExecutionContextRef(secondContext));
  await bridge.waitForCompletion();
  for (const [result, expected] of [[first, 2], [second, 9], [third, 3]]) {
    const ready = ok(await inference.getResult(new pb.ResultRef(result)));
    assert.equal(ready.state, pb.ResultState.RESULT_STATE_READY);
    assert.equal(ready.executionId, result.executionId);
    const read = ok(await inference.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'out0' })));
    const data = read.tensor.inline;
    assert.deepEqual([...new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))],
      Array(12).fill(expected));
    await inference.releaseResult(new pb.ResultRef(result));
  }
  // Release before mapping settles, repeatedly crossing the default 64-result budget.
  for (let index = 0; index < 70; index++) {
    const result = await submit(context, index);
    await inference.releaseResult(new pb.ResultRef(result));
  }
  await bridge.waitForCompletion();
}

/** Inject host API failures while real command submission still uses the GPU.
 * Buffers keep their WebIDL identity; wrapping them in a Proxy would make
 * native WebGPU calls reject them before exercising the intended failure. */
function instrumentDevice(device, mode) {
  const active = new Set();
  const scopes = [];
  let injected = 0;
  let allocations = 0;
  const wrapped = new Proxy(device, {
    get(target, key) {
      if (key === 'limits' && ['device-limits', 'workgroup-storage', 'workgroup-fallback'].includes(mode)) {
        return new Proxy(target.limits, {
          get(limits, name) {
            if (name === 'maxBufferSize' && mode === 'device-limits') { injected++; return 47; }
            if (name === 'maxComputeWorkgroupStorageSize' && mode !== 'device-limits') {
              injected++;
              return 0;
            }
            return Reflect.get(limits, name, limits);
          },
        });
      }
      if (key === 'createCommandEncoder' && mode === 'pass-creation') return () => {
        injected++;
        throw new Error('injected command encoder creation failure');
      };
      if (key === 'createBuffer') return (descriptor) => {
        const readback = (descriptor.usage & GPUBufferUsage.MAP_READ) !== 0;
        if (readback && mode === 'staging-allocation') {
          injected++;
          throw new Error('injected staging allocation failure');
        }
        const buffer = target.createBuffer(descriptor);
        allocations++;
        active.add(buffer);
        const destroy = buffer.destroy.bind(buffer);
        Object.defineProperty(buffer, 'destroy', { value() {
          destroy();
          active.delete(buffer);
        } });
        if (readback && mode === 'map-rejection') {
          Object.defineProperty(buffer, 'mapAsync', { value: async () => {
            injected++;
            throw new Error('injected readback mapping failure');
          } });
        }
        if (readback && mode === 'mapped-range') {
          Object.defineProperty(buffer, 'getMappedRange', { value() {
            injected++;
            throw new Error('injected mapped range failure');
          } });
        }
        return buffer;
      };
      if (key === 'pushErrorScope') return (filter) => {
        target.pushErrorScope(filter);
        scopes.push(filter);
      };
      if (key === 'popErrorScope') return async () => {
        assert.ok(scopes.length > 0, 'error scope popped without a matching push');
        const filter = scopes.pop();
        const error = await target.popErrorScope();
        if (mode === 'out-of-memory' && filter === 'out-of-memory') {
          injected++;
          return error ?? { message: 'injected GPU out of memory' };
        }
        return error;
      };
      const value = Reflect.get(target, key, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return { device: wrapped, active, scopes,
    get injected() { return injected; }, get allocations() { return allocations; } };
}

async function main() {
  const options = parseArguments(globalThis.Deno?.args ?? process.argv.slice(2));
  const { device, description } = await requirePhysicalDevice();
  options.device = device;
  console.log(`webgpu-device-bridge adapter: ${description}`);
  const api = await import(`file://${options.bundle}`);
  if (typeof api.createWebGPUHostBridge !== 'function') {
    fail('the bundle does not export createWebGPUHostBridge');
  }
  const { OperationCode: code, OperationStage: stage, NativeStatus: status } = api.pb;
  const reportedFailure = (expected) => (error) => {
    assert.ok(error.report, error.message);
    assert.notEqual(error.report.status, status.NATIVE_STATUS_OK);
    for (const [field, value] of Object.entries(expected))
      assert.equal(error.report[field], value, `${field}: ${error.message}`);
    return true;
  };

  let failures = 0;
  let ran = 0;
  let fixtureOracles = 0;
  /* Extracted cases are appended so a missing directory simply yields none. */
  const extracted = ['QConv2D', 'QConv2DDepthwise']
    .map((stem) => extractedCase(options.cases ?? 'build/test-reports/cases', stem))
    .filter(Boolean);
  if (options.cases && extracted.length !== 2) fail('explicit case directory must contain both convolution fixtures');
  for (const [name, graph, tensors, aliasable = false, fixtureOracle] of [...CASES, ...extracted]) {
    if (options.only && !name.toLowerCase().includes(options.only.toLowerCase())) {
      continue;
    }
    ran++;
    /* Every case must pass C CPU parity. Three also have independent
     * expectations, so a shared numerical bug cannot pass by agreement. */
    let oracle = null;
    let oracleRefusal = null;
    try {
      oracle = await runGraph(api, options, graph, tensors, HOST_BACKEND);
    } catch (error) {
      oracleRefusal = error.message;
    }
    try {
      if (!oracle) fail(`C CPU oracle failed: ${oracleRefusal}`);
      const actual = await runGraph(api, options, graph, tensors, 'webgpu');
      if (fixtureOracle) {
        fixtureOracles++;
        if (oracle) for (const [output, expected] of fixtureOracle)
          compare(`${name}/CPU/${output}`, oracle.outputs.get(output), expected, 1e-4);
        oracle = { outputs: fixtureOracle };
      }
      if (actual.route?.builtin !== true) {
        fail('the built-in backend did not own the route');
      }
      if (Number(actual.route?.fallbackNodes ?? 0) !== 0) {
        fail('a node fell back to the host, so the device did not run it');
      }
      /*
       * A shape-family node whose output the engine aliased onto its input has
       * nothing to copy, so no dispatch is the right answer. Saying which
       * cases may legitimately reach the device zero times keeps the check
       * strict everywhere else -- values alone would hide a route that never
       * ran, because an aliased no-op produces the correct answer.
       */
      if (actual.encodes === 0 && !aliasable) {
        fail('no dispatch reached the device');
      }
      if (!oracle) {
        fail(`no numerical oracle: ${oracleRefusal}`);
      }
      let worst = 0;
      let quantizedNote = '';
      for (const [output, expected] of oracle.outputs) {
        if (options.dump) {
          console.log(`  ${name}/${output} host   ${[...expected].slice(0, 8)}`);
          console.log(`  ${name}/${output} device ${
            [...actual.outputs.get(output)].slice(0, 8)}`);
        }
        const result = compare(
          `${name}/${output}`, actual.outputs.get(output), expected, 1e-4);
        if (typeof result === 'number') worst = Math.max(worst, result);
        else quantizedNote = `, ${result.differing}/${result.total} elements ` +
          `differ by one step`;
      }
      console.log(`webgpu-device-bridge ${name}: PASS ` +
        `(max_rel=${worst.toExponential(3)}${quantizedNote}, ` +
        `dispatches=${actual.encodes}` +
        `${fixtureOracle ? ', independent fixture oracle' : ''}` +
        `${actual.encodes === 0 ? ', engine aliased the copy' : ''})`);
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge ${name}: FAIL ${error.message}`);
    }
  }

  /*
   * A second execution must answer its own input. The host uploads a span only
   * when the device has not written it, so the rule that decides "written" is
   * what makes repeated execution correct -- invisible to a host that copies
   * unconditionally, which is what the CPU-interpreted proof uses.
   */
  if (!options.only) {
    try {
      const graph = unaryGraph('ReLU', SHAPE);
      /* Both executions land on one context, which is the only arrangement
       * where the second upload can be skipped. */
      const device = await runGraph(
        api, options, graph, [one(LEFT), one(RIGHT)], 'webgpu');
      const firstOracle = await runGraph(api, options, graph, one(LEFT), HOST_BACKEND);
      const secondOracle = await runGraph(api, options, graph, one(RIGHT), HOST_BACKEND);
      compare('repeat #1', device.rounds[0].get('out0'),
        firstOracle.outputs.get('out0'), 1e-4);
      const worst = compare('repeat #2', device.rounds[1].get('out0'),
        secondOracle.outputs.get('out0'), 1e-4);
      console.log('webgpu-device-bridge repeated-execution: PASS ' +
        `(max_rel=${worst.toExponential(3)})`);
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge repeated-execution: FAIL ${error.message}`);
    }
  }

  if (!options.only) {
    try {
      await runGraph(api, { ...options, inspectContext: checkResultLifetimes },
        unaryGraph('ReLU', SHAPE), one(LEFT), 'webgpu');
      console.log('webgpu-device-bridge result-lifetimes: PASS (interleaved contexts, stable snapshots, 70 pending releases)');
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge result-lifetimes: FAIL ${error.message}`);
    }
    try {
      const invalidDevice = new Proxy(device, {
        get(target, key) {
          if (key === 'createShaderModule') return () => target.createShaderModule({ code: 'invalid WGSL' });
          const value = Reflect.get(target, key, target);
          return typeof value === 'function' ? value.bind(target) : value;
        },
      });
      await assert.rejects(runGraph(api, { ...options, device: invalidDevice },
        unaryGraph('ReLU', SHAPE), one(LEFT), 'webgpu'), reportedFailure({
          code: code.OPERATION_CODE_DEVICE_EXECUTION_FAILED,
          stage: stage.OPERATION_STAGE_READBACK,
        }));
      console.log('webgpu-device-bridge asynchronous-validation: PASS (invalid shader cannot publish READY)');
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge asynchronous-validation: FAIL ${error.message}`);
    }
    for (const mode of ['staging-allocation', 'map-rejection', 'mapped-range',
      'out-of-memory', 'pass-creation', 'device-limits', 'workgroup-storage', 'close-pending']) {
      try {
        const tracked = instrumentDevice(device, mode);
        const afterSubmit = mode === 'close-pending' ? async ({ host, result }) => {
          assert.equal(result.state, api.pb.ResultState.RESULT_STATE_PENDING);
          await host.close();
        } : undefined;
        const compileRefusal = mode === 'device-limits' || mode === 'workgroup-storage';
        const expected = mode === 'close-pending' ? /EngineHost is closed/
          : compileRefusal ? reportedFailure({ stage: stage.OPERATION_STAGE_COMPILE })
          : mode === 'staging-allocation' || mode === 'pass-creation'
            ? reportedFailure({ code: code.OPERATION_CODE_EXECUTION_FAILED,
                stage: stage.OPERATION_STAGE_EXECUTE })
            : reportedFailure({ code: code.OPERATION_CODE_DEVICE_EXECUTION_FAILED,
                stage: stage.OPERATION_STAGE_READBACK });
        const [, graph, tensors] = mode === 'workgroup-storage'
          ? CASES.find(([name]) => name === 'QLayerNorm')
          : ['ReLU', unaryGraph('ReLU', SHAPE), one(LEFT)];
        await assert.rejects(runGraph(api, { ...options, device: tracked.device, afterSubmit },
          graph, tensors, 'webgpu'), expected);
        assert.equal(tracked.active.size, 0, 'host close leaked GPU buffers');
        assert.equal(tracked.scopes.length, 0, 'failed operation leaked error scopes');
        if (compileRefusal) assert.equal(tracked.allocations, 0,
          'compile refusal allocated a GPU buffer');
        if (mode !== 'close-pending') assert.ok(tracked.injected > 0, 'fault was not reached');
        console.log(`webgpu-device-bridge ${mode}: PASS (all buffers and error scopes reclaimed)`);
      } catch (error) {
        failures++;
        console.error(`webgpu-device-bridge ${mode}: FAIL ${error.message}`);
      }
    }
    try {
      const tracked = instrumentDevice(device, 'workgroup-fallback');
      const [, graph, tensors] = CASES.find(([name]) => name === 'QLinear');
      const oracle = await runGraph(api, options, graph, tensors, HOST_BACKEND);
      const actual = await runGraph(api, { ...options, device: tracked.device }, graph, tensors, 'webgpu');
      for (const [name, expected] of oracle.outputs)
        compare(`workgroup-fallback/${name}`, actual.outputs.get(name), expected, 1e-4);
      assert.ok(tracked.injected > 0 && actual.encodes > 0);
      assert.equal(actual.route?.fallbackNodes, 0);
      assert.equal(tracked.active.size, 0);
      assert.equal(tracked.scopes.length, 0);
      console.log('webgpu-device-bridge workgroup-fallback: PASS (C selects a GPU variant without shared storage)');
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge workgroup-fallback: FAIL ${error.message}`);
    }
    try {
      const isolated = await requirePhysicalDevice();
      await assert.rejects(runGraph(api, { ...options, device: isolated.device,
        afterSubmit: () => isolated.device.destroy() }, unaryGraph('ReLU', SHAPE), one(LEFT), 'webgpu'),
        reportedFailure({ code: code.OPERATION_CODE_DEVICE_LOST,
          status: status.NATIVE_STATUS_DEVICE_LOST, stage: stage.OPERATION_STAGE_READBACK }));
      console.log('webgpu-device-bridge device-loss: PASS (terminal DEVICE_LOST result)');
    } catch (error) {
      failures++;
      console.error(`webgpu-device-bridge device-loss: FAIL ${error.message}`);
    }
  }

  if (ran === 0) fail('no numerical cases matched the selection');
  if (failures > 0) fail(`${failures} of ${ran} case(s) failed`);
  console.log(`webgpu-device-bridge: ${ran} of ${ran} cases numerically verified ` +
    `on a physical adapter (${ran} C CPU comparisons, ${fixtureOracles} additional independent fixture oracles)`);
}

export { CASES, graphOf, runGraph, compare, extractedCase, requirePhysicalDevice };
if (import.meta.main || (!globalThis.Deno &&
    path.resolve(process.argv[1] ?? '') === new URL(import.meta.url).pathname)) await main();
