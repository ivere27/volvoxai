import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/core/Graph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

async function buildForwardWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
}

async function cpuResult(graph, inputs) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

function assertClose(actual, expected, label, tolerance = 5e-5) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index++) {
    const value = actual[index];
    const reference = expected[index];
    if (Number.isNaN(reference)) {
      assert.ok(Number.isNaN(value), `${label}[${index}] should be NaN`);
      continue;
    }
    if (!Number.isFinite(reference)) {
      assert.equal(value, reference, `${label}[${index}] should preserve infinity`);
      continue;
    }
    const allowed = tolerance * Math.max(1, Math.abs(reference));
    const difference = Math.abs(value - reference);
    assert.ok(difference <= allowed,
      `${label}[${index}] differs by ${difference}; got ${value}, expected ${reference}`);
  }
}

function assertResultsClose(actual, expected, label, tolerance = 5e-5) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${label} output names`);
  for (const name of Object.keys(expected)) {
    assertClose(actual[name], expected[name], `${label}/${name}`, tolerance);
  }
}

function forbidCpuFallbacks(engine) {
  const helpers = [
    '_cpuMatMul', '_cpuRMSNorm', '_cpuGroupNorm', '_cpuMoERouter', '_cpuMoELinear',
    '_cpuResize', '_cpuSiLU', '_cpuTanh', '_cpuAdd', '_cpuMul', '_cpuSub', '_cpuDiv',
    '_cpuSoftmax', '_cpuLogSoftmax', '_cpuTranspose', '_cpuConcat2', '_cpuSplit', '_cpuExpand',
    '_cpuLeakyReLU', '_cpuPReLU', '_cpuUpsample2x', '_cpuInterp1D', '_cpuPad',
    '_cpuAveragePool2D',
  ];
  for (const helper of helpers) {
    engine[helper] = () => {
      throw new Error(`ordinary WASM dispatch must not call CPU fallback ${helper}`);
    };
  }
}

async function assertWasmParity(wasm, label, makeGraph, inputs, tolerance = 5e-5) {
  const expected = await cpuResult(makeGraph(), inputs);
  const graph = makeGraph();
  wasm.compile(graph);
  const actual = await wasm.execute(inputs);
  assertResultsClose(actual, expected, label, tolerance);
}

function linearGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3]);
  const weight = graph.addWeight('weight', [3, 2], 'float32', {
    buffer: Float32Array.of(0.25, -0.5, 1.5, 0.75, -1, 0.5),
  });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.of(0.125, -0.25) });
  const { out } = graph.addOp(opType, { input, weight, bias }, {
    out: { name: 'out', shape: [2, 2] },
  }, { weight_layout: 'IN_OUT' });
  graph.outputNames = [out.name];
  return graph;
}

function relu6Conv2DGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 1, 2, 1]);
  const weight = graph.addWeight('weight', [1, 1, 1, 1], 'float32', {
    buffer: Float32Array.of(4),
  });
  const bias = graph.addWeight('bias', [1], 'float32', {
    buffer: Float32Array.of(0),
  });
  const { out } = graph.addOp('Conv2D', { input, weight, bias }, {
    out: { name: 'out', shape: [1, 1, 2, 1] },
  }, { relu: 2 });
  graph.outputNames = [out.name];
  return graph;
}

function dynamicDoutLinearGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2]);
  const weight = graph.addInput('weight', [2, 2]);
  const { out } = graph.addOp(opType, { input, weight }, {
    out: { name: 'out', shape: [1, 2] },
  }, { weight_layout: 'OUT_IN' });
  graph.outputNames = [out.name];
  return graph;
}

function sharedPackedLinearGraph(opType, rows = 7) {
  const dIn = 17;
  const dOut = 13;
  const graph = new Graph();
  const inputA = graph.addInput('inputA', [rows, dIn]);
  const inputB = graph.addInput('inputB', [rows, dIn]);
  const weight = graph.addWeight('sharedWeight', [dOut, dIn], 'float32', {
    buffer: Float32Array.from({ length: dOut * dIn }, (_, index) =>
      Math.fround((((index * 17) % 29) - 14) / 31)),
  });
  const bias = graph.addWeight('sharedBias', [dOut], 'float32', {
    buffer: Float32Array.from({ length: dOut }, (_, index) => Math.fround((index - 6) / 19)),
  });
  const first = graph.addOp(opType, { input: inputA, weight, bias }, {
    out: { name: 'packedOutA', shape: [rows, dOut] },
  }, { weight_layout: 'OUT_IN' }).out;
  const second = graph.addOp(opType, { input: inputB, weight, bias }, {
    out: { name: 'packedOutB', shape: [rows, dOut] },
  }, { weight_layout: 'OUT_IN' }).out;
  graph.outputNames = [first.name, second.name];
  return graph;
}

function quantizedOddWidthLinearGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 3]);
  const weight = graph.addWeight('weight', [2, 3], 'int8', {
    buffer: Int8Array.of(1, 2, 3, -1, 2, -3),
  });
  const scale = graph.addWeight('scale', [2], 'float32', { buffer: Float32Array.of(0.5, 0.25) });
  const zeroPoint = graph.addWeight('zeroPoint', [2], 'int32', { buffer: Int32Array.of(1, -2) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.of(0.25, -1) });
  const { out } = graph.addOp(opType, { input, weight, scale, zero_point: zeroPoint, bias }, {
    out: { name: 'out', shape: [1, 2] },
  }, { weight_layout: 'OUT_IN' });
  graph.outputNames = [out.name];
  return graph;
}

function quantizedUint8LinearGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 3]);
  const weight = graph.addWeight('weight', [2, 3], 'uint8', {
    buffer: Uint8Array.of(2, 3, 4, 5, 6, 7),
  });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.5) });
  const zeroPoint = graph.addWeight('zeroPoint', [1], 'uint8', { buffer: Uint8Array.of(2) });
  const { out } = graph.addOp(opType, { input, weight, scale, zero_point: zeroPoint }, {
    out: { name: 'out', shape: [1, 2] },
  }, { weight_layout: 'OUT_IN' });
  graph.outputNames = [out.name];
  return graph;
}

function rmsNormGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 3]);
  const weight = graph.addWeight('weight', [3], 'float32', { buffer: Float32Array.of(1.2, 0.75, 1.5) });
  const { out } = graph.addOp('RMSNorm', { input, weight }, {
    out: { name: 'out', shape: [2, 2, 3] },
  }, { d_model: 3, eps: 0.125 });
  graph.outputNames = [out.name];
  return graph;
}

function rmsNormExtremeGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2]);
  const weight = graph.addWeight('weight', [2], 'float32', { buffer: Float32Array.of(1, 1) });
  const { out } = graph.addOp('RMSNorm', { input, weight }, {
    out: { name: 'out', shape: [1, 2] },
  }, { d_model: 2, eps: 1e-6 });
  graph.outputNames = [out.name];
  return graph;
}

function groupNormGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 2, 4]);
  const weight = graph.addWeight('weight', [4], 'float32', { buffer: Float32Array.of(1, 0.75, 1.25, 0.5) });
  const bias = graph.addWeight('bias', [4], 'float32', { buffer: Float32Array.of(0.1, -0.2, 0.3, -0.4) });
  const { out } = graph.addOp('GroupNorm', { input, weight, bias }, {
    out: { name: 'out', shape: [2, 2, 2, 4] },
  }, { num_groups: 2, eps: 0.05 });
  graph.outputNames = [out.name];
  return graph;
}

function groupNormPrecisionGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 1, 1, 2]);
  const weight = graph.addWeight('weight', [2], 'float32', { buffer: Float32Array.of(1, 1) });
  const bias = graph.addWeight('bias', [2], 'float32', { buffer: Float32Array.of(0, 0) });
  const { out } = graph.addOp('GroupNorm', { input, weight, bias }, {
    out: { name: 'out', shape: [1, 1, 1, 2] },
  }, { num_groups: 1, eps: 1e-50 });
  graph.outputNames = [out.name];
  return graph;
}

function moeGraph(temperature = 1.25) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3]);
  const routerWeight = graph.addWeight('routerWeight', [3, 3], 'float32', {
    buffer: Float32Array.of(0.5, -0.25, 0.75, -0.5, 1, 0.25, 0.2, -0.3, 0.6),
  });
  const routerBias = graph.addWeight('routerBias', [3], 'float32', { buffer: Float32Array.of(0.1, -0.2, 0.05) });
  const routes = graph.addOp('MoERouter', { input, weight: routerWeight, bias: routerBias }, {
    indices: { name: 'indices', shape: [2, 2] },
    weights: { name: 'routeWeights', shape: [2, 2] },
  }, { num_experts: 3, top_k: 2, temperature, normalize: true });
  const expertWeight = graph.addWeight('expertWeight', [3, 3, 2], 'float32', {
    buffer: Float32Array.of(
      0.5, -0.5, 1, 0.25, -0.25, 0.75,
      -0.75, 0.5, 0.5, -1, 0.25, 0.125,
      1, 0.5, -0.5, 0.25, 0.75, -0.25,
    ),
  });
  const expertBias = graph.addWeight('expertBias', [3, 2], 'float32', {
    buffer: Float32Array.of(0.1, -0.1, 0.2, 0.05, -0.25, 0.3),
  });
  const { out } = graph.addOp('MoELinear', {
    input,
    expert_weight: expertWeight,
    expert_bias: expertBias,
    route_indices: routes.indices,
    route_weights: routes.weights,
  }, { out: { name: 'out', shape: [2, 2] } });
  graph.outputNames = [routes.indices.name, routes.weights.name, out.name];
  return graph;
}

function resizeNearestGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 3, 2]);
  const { out } = graph.addOp('ResizeNearest2D', { input }, {
    out: { name: 'out', shape: [2, 3, 5, 2] },
  }, { mode: 'nearest' });
  graph.outputNames = [out.name];
  return graph;
}

function resizeModeNearestGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 3, 2]);
  const { out } = graph.addOp('Resize', { input }, {
    out: { name: 'out', shape: [2, 3, 5, 2] },
  }, { mode: 'nearest' });
  graph.outputNames = [out.name];
  return graph;
}

function upsampleGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 2, 2, 1]);
  const { out } = graph.addOp(opType, { input }, {
    out: { name: 'out', shape: [2, 4, 4, 1] },
  });
  graph.outputNames = [out.name];
  return graph;
}

function interpGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 1, 3]);
  const { out } = graph.addOp(opType, { input }, {
    out: { name: 'out', shape: [2, 1, 5] },
  }, { size: 5 });
  graph.outputNames = [out.name];
  return graph;
}

function leakyGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 5]);
  const { out } = graph.addOp('LeakyReLU', { input }, {
    out: { name: 'out', shape: [1, 5] },
  }, { alpha: 0 });
  graph.outputNames = [out.name];
  return graph;
}

function preluGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 5]);
  const slope = graph.addInput('slope', [2]);
  const { out } = graph.addOp('PReLU', { input, slope }, {
    out: { name: 'out', shape: [1, 5] },
  });
  graph.outputNames = [out.name];
  return graph;
}

function unaryGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3]);
  const { out } = graph.addOp(opType, { input }, { out: { name: 'out', shape: [2, 3] } });
  graph.outputNames = [out.name];
  return graph;
}

function binaryBroadcastGraph(opType, params = {}) {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 1, 3]);
  const b = graph.addInput('b', [1, 4, 1]);
  const { out } = graph.addOp(opType, { a, b }, {
    out: { name: 'out', shape: [2, 4, 3] },
  }, params);
  graph.outputNames = [out.name];
  return graph;
}

function softmaxGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 4]);
  const { out } = graph.addOp(opType, { input }, { out: { name: 'out', shape: [2, 4] } });
  graph.outputNames = [out.name];
  return graph;
}

function typedShapeCopyGraph(dtype) {
  const graph = new Graph();
  let value = graph.addInput('values', [1, 2, 2], dtype);
  for (const [index, opType, shape, params] of [
    [0, 'Identity', [1, 2, 2], {}],
    [1, 'Reshape', [1, 4], {}],
    [2, 'Flatten', [1, 4], { axis: 1 }],
    [3, 'Squeeze', [4], { axes: [0] }],
    [4, 'Unsqueeze', [1, 4], { axes: [0] }],
  ]) {
    value = graph.addOp(opType, { input: value }, {
      out: { name: `shape_${index}`, shape, dtype },
    }, params).out;
  }
  graph.outputNames = [value.name];
  return graph;
}

function reductionGraph(opType, inputKey) {
  const graph = new Graph();
  const values = graph.addInput('values', [2, 2, 4]);
  const { out } = graph.addOp(opType, { [inputKey]: values }, {
    out: { name: 'out', shape: [2, 2] },
  });
  graph.outputNames = [out.name];
  return graph;
}

function padGraph(inputKey) {
  const graph = new Graph();
  const values = graph.addInput('values', [1, 2, 2, 1]);
  const { out } = graph.addOp('Pad', { [inputKey]: values }, {
    out: { name: 'out', shape: [1, 3, 4, 1] },
  }, { pads: [1, 1, 0, 1], value: -0.25 });
  graph.outputNames = [out.name];
  return graph;
}

function averagePoolGraph(opType, inputKey) {
  const graph = new Graph();
  const values = graph.addInput('values', [1, 3, 3, 1]);
  const { out } = graph.addOp(opType, { [inputKey]: values }, {
    out: { name: 'out', shape: [1, 2, 2, 1] },
  }, { kernel: [2, 2], stride: [1, 1], padding: [0, 0] });
  graph.outputNames = [out.name];
  return graph;
}

function transposeGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 3, 2]);
  const { out } = graph.addOp('Transpose', { input }, {
    out: { name: 'out', shape: [2, 2, 3] },
  }, { perm: [2, 0, 1] });
  graph.outputNames = [out.name];
  return graph;
}

function concatGraph(opType, params = {}) {
  const graph = new Graph();
  const a = graph.addInput('a', [2, 1, 2]);
  const b = graph.addInput('b', [2, 2, 2]);
  const inputs = opType === 'Concat2' ? { a, b } : {
    a,
    b,
    c: graph.addInput('c', [2, 1, 2]),
  };
  const axis = inputs.c ? 4 : 3;
  const { out } = graph.addOp(opType, inputs, {
    out: { name: 'out', shape: [2, axis, 2] },
  }, { axis: 1, ...params });
  graph.outputNames = [out.name];
  return graph;
}

function splitGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [2, 6, 2]);
  const outputs = graph.addNode({
    opType: 'Split',
    inputs: { input },
    outputs: {
      out0: { name: 'out0', shape: [2, 2, 2] },
      out1: { name: 'out1', shape: [2, 2, 2] },
      out2: { name: 'out2', shape: [2, 2, 2] },
    },
    params: { axis: -2 },
  }).outputs;
  graph.outputNames = [outputs.out0.name, outputs.out1.name, outputs.out2.name];
  return graph;
}

function splitMixedCaseGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 4]);
  const outputs = graph.addNode({
    opType: 'Split',
    inputs: { input },
    outputs: {
      a: { name: 'a', shape: [1, 2] },
      B: { name: 'B', shape: [1, 2] },
    },
    params: { axis: 1 },
  }).outputs;
  graph.outputNames = [outputs.a.name, outputs.B.name];
  return graph;
}

function expandGraph(opType) {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 3]);
  const { out } = graph.addOp(opType, { input }, {
    out: { name: 'out', shape: [2, 4, 3] },
  });
  graph.outputNames = [out.name];
  return graph;
}

test('ordinary forward WASM replaces every former CPU fallback row with direct kernels', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      t.skip(`${clang} is unavailable`);
      return;
    }
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-direct-fallback-'));
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await buildForwardWasm(wasmPath);
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm, 'compiled forward-only WASM module initializes');
    forbidCpuFallbacks(wasm);

    await t.test('Conv2D preserves fused ReLU6 instead of reducing it to ReLU', async () => {
      await assertWasmParity(wasm, 'Conv2D ReLU6', relu6Conv2DGraph, {
        input: Float32Array.of(-2, 2),
      });
    });

    await t.test('Linear and Gemm use canonical F32 and quantized C paths', async () => {
      const inputs = { input: Float32Array.of(1, -2, 3, -4, 5, -6) };
      await assertWasmParity(wasm, 'Linear', () => linearGraph('Linear'), inputs);
      await assertWasmParity(wasm, 'Gemm', () => linearGraph('Gemm'), inputs);
      const dynamicDoutInputs = {
        input: Float32Array.of(2, 3),
        weight: Float32Array.of(1, 4, 2, 5),
      };
      await assertWasmParity(wasm, 'dynamic OUT_IN Linear', () => dynamicDoutLinearGraph('Linear'), dynamicDoutInputs);
      await assertWasmParity(wasm, 'dynamic OUT_IN Gemm', () => dynamicDoutLinearGraph('Gemm'), dynamicDoutInputs);
      const packedInputs = {
        inputA: Float32Array.from({ length: 7 * 17 }, (_, index) => Math.fround(((index * 7) % 23 - 11) / 13)),
        inputB: Float32Array.from({ length: 7 * 17 }, (_, index) => Math.fround(((index * 11) % 31 - 15) / 17)),
      };
      const packedExpected = await cpuResult(sharedPackedLinearGraph('Linear'), packedInputs);
      const packedGraph = sharedPackedLinearGraph('Linear');
      wasm.compile(packedGraph);
      assert.equal(wasm.f32PackedWeights.size, 1, 'shared immutable F32 weight is packed once');
      assert.equal(wasm.f32PackedNodes.size, 2, 'both Linear nodes reuse packed descriptors');
      const packedPointers = [...wasm.f32PackedNodes.values()].map(({ pointer }) => pointer);
      assert.equal(packedPointers[0], packedPointers[1], 'shared nodes use the same packed panel buffer');
      const packedActual = await wasm.execute(packedInputs);
      assertResultsClose(packedActual, packedExpected, 'packed shared odd-tail Linear');
      const packedAgain = await wasm.execute(packedInputs);
      assertResultsClose(packedAgain, packedExpected, 'packed shared odd-tail Linear repeat');
      assert.equal([...wasm.f32PackedWeights.values()][0].pointer, packedPointers[0],
        'repeat execution does not repack immutable weights');
      assert.equal(wasm.api.gemm_f32_tile_mr(), 4);
      assert.equal(wasm.api.gemm_f32_tile_nr(), 8);
      assert.equal(wasm.api.gemm_f32_tile_kc(), 496);
      assert.equal(wasm.api.gemm_f32_cache_budget_bytes(), 24576);
      assert.equal(wasm.api.gemm_f32_working_set_bytes(), 23936);
      const quantizedInputs = { input: Float32Array.of(1, 2, 3) };
      await assertWasmParity(wasm, 'odd-width quantized Linear', () => quantizedOddWidthLinearGraph('Linear'), quantizedInputs);
      await assertWasmParity(wasm, 'odd-width quantized Gemm', () => quantizedOddWidthLinearGraph('Gemm'), quantizedInputs);
      await assertWasmParity(wasm, 'U8 scalar-quantized Linear', () => quantizedUint8LinearGraph('Linear'), quantizedInputs);
    });

    await t.test('RMSNorm and GroupNorm use direct C kernels', async () => {
      await assertWasmParity(wasm, 'RMSNorm', rmsNormGraph, {
        input: Float32Array.of(1, -2, 3, -4, 5, -6, 2, -1, 4, -3, 6, -5),
      });
      await assertWasmParity(wasm, 'GroupNorm', groupNormGraph, {
        input: Float32Array.from({ length: 32 }, (_, index) => ((index * 7) % 13) - 6),
      });
      await assertWasmParity(wasm, 'RMSNorm large F32', rmsNormExtremeGraph, {
        input: Float32Array.of(1e20, 1e20),
      });
      await assertWasmParity(wasm, 'GroupNorm large offset and sub-F32 epsilon', groupNormPrecisionGraph, {
        input: Float32Array.of(10_000_000, 10_000_001),
      });
    });

    await t.test('MoERouter and MoELinear use direct C kernels', async () => {
      await assertWasmParity(wasm, 'MoE', moeGraph, {
        input: Float32Array.of(0.5, -1, 2, -0.75, 1.5, 0.25),
      }, 1e-4);
      await assertWasmParity(wasm, 'MoE f64 temperature', () => moeGraph(1e40), {
        input: Float32Array.of(0.5, -1, 2, -0.75, 1.5, 0.25),
      }, 1e-4);
    });

    await t.test('ResizeNearest2D handles non-2x batched resizing in C', async () => {
      await assertWasmParity(wasm, 'ResizeNearest2D', resizeNearestGraph, {
        input: Float32Array.from({ length: 24 }, (_, index) => index - 9),
      });
    });

    await t.test('nearest Resize, 2x aliases, and Interp1D aliases avoid CPU loops', async () => {
      const imageInput = { input: Float32Array.from({ length: 24 }, (_, index) => index - 9) };
      await assertWasmParity(wasm, 'Resize mode nearest', resizeModeNearestGraph, imageInput);
      const upsampleInput = { input: Float32Array.from({ length: 8 }, (_, index) => index - 3) };
      await assertWasmParity(wasm, 'UpsampleNearest2D', () => upsampleGraph('UpsampleNearest2D'), upsampleInput);
      await assertWasmParity(wasm, 'Upsample2x', () => upsampleGraph('Upsample2x'), upsampleInput);
      const interpInput = { input: Float32Array.from({ length: 6 }, (_, index) => index * 0.5 - 1) };
      await assertWasmParity(wasm, 'InterpLinear1D', () => interpGraph('InterpLinear1D'), interpInput);
      await assertWasmParity(wasm, 'Interp1D', () => interpGraph('Interp1D'), interpInput);
    });

    await t.test('SiLU, Swish, and accurate Tanh use direct C kernels', async () => {
      const inputs = { input: Float32Array.of(-3, -1, 0, 0.5, 1, 3) };
      await assertWasmParity(wasm, 'SiLU', () => unaryGraph('SiLU'), inputs);
      await assertWasmParity(wasm, 'Swish', () => unaryGraph('Swish'), inputs);
      await assertWasmParity(wasm, 'Tanh', () => unaryGraph('Tanh'), inputs, 1e-5);
    });

    await t.test('LeakyReLU alpha zero and generic PReLU slopes use direct C kernels', async () => {
      await assertWasmParity(wasm, 'LeakyReLU alpha zero', leakyGraph, {
        input: Float32Array.of(-3, -1, 0, 0.5, 2),
      });
      await assertWasmParity(wasm, 'PReLU generic slope repetition', preluGraph, {
        input: Float32Array.of(-3, -1, 0, 0.5, -2),
        slope: Float32Array.of(0.25, 0.5),
      });
    });

    await t.test('all binary broadcast operations use direct rank-N C kernels', async () => {
      const inputs = {
        a: Float32Array.of(1, -2, 3, 4, -5, 6),
        b: Float32Array.of(2, 0, -0.5, 3),
      };
      for (const opType of ['Add', 'Mul', 'Sub', 'Div']) {
        await assertWasmParity(wasm, opType, () => binaryBroadcastGraph(opType), inputs);
      }
      await assertWasmParity(
        wasm, 'Add fused ReLU', () => binaryBroadcastGraph('Add', { relu: 1 }), inputs,
      );
      await assertWasmParity(
        wasm, 'Add fused ReLU6', () => binaryBroadcastGraph('Add', { relu: 2 }), inputs,
      );
      const invalid = binaryBroadcastGraph('Add', { relu: 3 });
      wasm.compile(invalid);
      await assert.rejects(
        wasm.execute(inputs),
        /supports relu values 0.*1.*2/,
      );
    });

    await t.test('Softmax and LogSoftmax use direct C rows including extreme negatives', async () => {
      const inputs = {
        input: Float32Array.of(-3e30, -2e30, -2.5e30, -2.25e30, -1, 0, 2, 1),
      };
      await assertWasmParity(wasm, 'Softmax', () => softmaxGraph('Softmax'), inputs);
      await assertWasmParity(wasm, 'LogSoftmax', () => softmaxGraph('LogSoftmax'), inputs);
    });

    await t.test('F32 and I32 shape-copy chains stay inside WASM', async () => {
      await assertWasmParity(
        wasm,
        'F32 Identity/Reshape/Flatten/Squeeze/Unsqueeze',
        () => typedShapeCopyGraph('float32'),
        { values: Float32Array.of(-3.5, 0, 2.25, 9) },
      );
      await assertWasmParity(
        wasm,
        'I32 Identity/Reshape/Flatten/Squeeze/Unsqueeze',
        () => typedShapeCopyGraph('int32'),
        { values: Int32Array.of(-2147483648, -1, 0, 2147483647) },
        0,
      );
    });

    await t.test('ReduceSum and ReduceMean resolve both supported input aliases', async () => {
      const inputs = {
        values: Float32Array.of(
          -3, 2, 7, -1,
          4, -5, 8, 9,
          1, 3, -2, 6,
          -4, 5, 2, -7,
        ),
      };
      for (const opType of ['ReduceSum', 'ReduceMean']) {
        for (const inputKey of ['input', 'data']) {
          await assertWasmParity(
            wasm,
            `${opType} ${inputKey} alias`,
            () => reductionGraph(opType, inputKey),
            inputs,
          );
        }
      }
    });

    await t.test('Pad and average pooling resolve every supported input alias', async () => {
      const padInputs = { values: Float32Array.of(5, -2, 3, 1) };
      for (const inputKey of ['input', 'data']) {
        await assertWasmParity(wasm, `Pad ${inputKey} alias`, () => padGraph(inputKey), padInputs);
      }

      const poolInputs = {
        values: Float32Array.of(5, -2, 3, 1, 4, -6, 8, 2, 7),
      };
      for (const opType of ['AveragePool2D', 'AveragePool']) {
        for (const inputKey of ['input', 'x']) {
          await assertWasmParity(
            wasm,
            `${opType} ${inputKey} alias`,
            () => averagePoolGraph(opType, inputKey),
            poolInputs,
          );
        }
      }
    });

    await t.test('Transpose, Concat, Concat2, and Split use direct shape kernels', async () => {
      await assertWasmParity(wasm, 'Transpose', transposeGraph, {
        input: Float32Array.from({ length: 12 }, (_, index) => index - 4),
      });
      await assertWasmParity(wasm, 'Concat', () => concatGraph('Concat'), {
        a: Float32Array.of(1, 2, 3, 4),
        b: Float32Array.of(5, 6, 7, 8, 9, 10, 11, 12),
        c: Float32Array.of(13, 14, 15, 16),
      });
      await assertWasmParity(wasm, 'Concat2', () => concatGraph('Concat2'), {
        a: Float32Array.of(1, 2, 3, 4),
        b: Float32Array.of(5, 6, 7, 8, 9, 10, 11, 12),
      });
      await assertWasmParity(wasm, 'Concat truthy sigmoid', () => concatGraph('Concat2', { sigmoid: 1 }), {
        a: Float32Array.of(1, 2, 3, 4),
        b: Float32Array.of(5, 6, 7, 8, 9, 10, 11, 12),
      });
      await assertWasmParity(wasm, 'Split', splitGraph, {
        input: Float32Array.from({ length: 24 }, (_, index) => index - 12),
      });
      await assertWasmParity(wasm, 'Split code-unit output order', splitMixedCaseGraph, {
        input: Float32Array.of(1, 2, 3, 4),
      });
    });

    await t.test('Expand and Broadcast use direct right-aligned rank-N C kernels', async () => {
      const inputs = { input: Float32Array.of(-1, 0.5, 2) };
      await assertWasmParity(wasm, 'Expand', () => expandGraph('Expand'), inputs);
      await assertWasmParity(wasm, 'Broadcast', () => expandGraph('Broadcast'), inputs);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
