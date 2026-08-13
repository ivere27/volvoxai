import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function qaddGraph({
  aDtype = 'int8', aValues, aShape, aQuantization,
  bDtype = 'int8', bValues, bShape, bQuantization,
  outputDtype = 'int8', outputShape, outputQuantization,
  params = {},
} = {}) {
  const graph = new RuntimeGraph();
  const a = graph.addInput('a', aShape, aDtype, { quantization: aQuantization });
  const b = graph.addInput('b', bShape, bDtype, { quantization: bQuantization });
  const { out } = graph.addOp('QAdd', { a, b }, {
    out: { name: 'out', shape: outputShape, dtype: outputDtype, quantization: outputQuantization },
  }, params);
  graph.setOutputs([out.name]);
  return {
    graph,
    inputs: { a: byteStorage(aDtype, aValues), b: byteStorage(bDtype, bValues) },
  };
}

function qconvGraph({
  inputDtype = 'int8', inputValues, inputShape, inputQuantization,
  weightDtype = 'int8', weightValues, weightShape, weightQuantization,
  biasValues = null,
  outputDtype = 'int8', outputShape, outputQuantization,
  params = {},
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, { quantization: inputQuantization });
  const weight = graph.addWeight('weight', weightShape, weightDtype, {
    buffer: byteStorage(weightDtype, weightValues), quantization: weightQuantization,
  });
  const inputs = { input, weight };
  if (biasValues) {
    inputs.bias = graph.addWeight('bias', [weightShape[0]], 'int32', {
      buffer: Int32Array.from(biasValues),
    });
  }
  const { out } = graph.addOp('QConv2D', inputs, {
    out: { name: 'out', shape: outputShape, dtype: outputDtype, quantization: outputQuantization },
  }, { data_layout: 'NHWC', weight_layout: 'OHWI', ...params });
  graph.setOutputs([out.name]);
  return { graph, inputs: { input: byteStorage(inputDtype, inputValues) } };
}

function typedW8A8IslandGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 2, 1], 'uint8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  });
  const weight = graph.addWeight('weight', [1, 1, 1, 1], 'int8', {
    buffer: Int8Array.of(2),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.5], zero_points: [0] },
  });
  const bias = graph.addWeight('bias', [1], 'int32', { buffer: Int32Array.of(0) });
  const { out: requantized } = graph.addOp('RequantizeLinear', { input }, {
    out: {
      name: 'requantized', shape: [1, 2, 2, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  });
  const { out: convolved } = graph.addOp('QConv2D', { input: requantized, weight, bias }, {
    out: {
      name: 'convolved', shape: [1, 2, 2, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  }, { data_layout: 'NHWC', weight_layout: 'OHWI' });
  const { out } = graph.addOp('QAdd', { a: convolved, b: convolved }, {
    out: {
      name: 'out', shape: [1, 2, 2, 1], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function expandedQAddGraph() {
  const graph = new RuntimeGraph();
  const left = graph.addInput('left', [2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -2 },
  });
  const right = graph.addInput('right', [1, 4], 'uint8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  });
  const { out: expanded } = graph.addOp('Expand', { input: right }, {
    out: {
      name: 'expanded', shape: [2, 4], dtype: 'uint8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    },
  });
  const { out } = graph.addOp('QAdd', { a: left, b: expanded }, {
    out: {
      name: 'out', shape: [2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 3 },
    },
  });
  graph.setOutputs([out.name]);
  return {
    graph,
    inputs: {
      left: Int8Array.of(-2, 0, 2, 10, -128, 5, 7, 9),
      right: Uint8Array.of(128, 132, 120, 255),
    },
  };
}

async function cpuResult(factory) {
  const { graph, inputs } = factory();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

test('WASM QConv scratch allocation fails closed outside wasm32', () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  const wasm = new WasmEngine({
    instance: {
      exports: {
        memory,
        alloc_bytes: () => 0xfffffff0,
      },
    },
  });
  assert.throws(
    () => wasm._allocScratchBytes(32, 'test scratch'),
    /exceeds the wasm32 address space/,
  );
});

function forbidCpuQuantizedFallbacks(engine) {
  for (const helper of ['_cpuQAdd', '_cpuQConv2D', '_cpuRequantizeLinear', '_cpuExpand']) {
    engine[helper] = () => {
      throw new Error(`portable W8A8 WASM dispatch must not call ${helper}`);
    };
  }
}

test('portable WASM QAdd and QConv2D preserve byte storage', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qadd-qconv-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qadd_i8u8, 'function', 'forward WASM exports qadd_i8u8');
    assert.equal(typeof wasm.api.expand_nd_i8u8, 'function', 'forward WASM exports expand_nd_i8u8');
    assert.equal(typeof wasm.api.qconv2d_i8u8, 'function', 'forward WASM exports qconv2d_i8u8');
    forbidCpuQuantizedFallbacks(wasm);

    await t.test('generic Add cannot bypass the canonical byte-domain operator contract', () => {
      const graph = new RuntimeGraph();
      const quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
      const a = graph.addInput('a', [4], 'int8', { quantization });
      const b = graph.addInput('b', [4], 'int8', { quantization });
      graph.addOp('Add', { a, b }, {
        out: { name: 'out', shape: [4], dtype: 'int8', quantization },
      });
      assert.throws(() => wasm.compile(graph), /unsupported generic operator/);
    });

    await t.test('QAdd requantizes mixed I8/U8 descriptors directly in byte storage', async () => {
      const spec = {
        aDtype: 'int8', aValues: [-2, 0, 2, 10, -128], aShape: [5],
        aQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -2 },
        bDtype: 'uint8', bValues: [128, 132, 120, 255, 0], bShape: [5],
        bQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        outputDtype: 'int8', outputShape: [5],
        outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 3 },
      };
      const cpu = await cpuResult(() => qaddGraph(spec));
      const { graph, inputs } = qaddGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [3, 7, 3, 78, -128]);
    });

    await t.test('QAdd fuses output-domain ReLU and ReLU6 without materializing F32 activations', async () => {
      const base = {
        aValues: [-20, -4, 0, 10, 30], aShape: [5],
        aQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -4 },
        bValues: [-4, -4, -4, 10, 30], bShape: [5],
        bQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -4 },
        outputShape: [5],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -4 },
      };
      for (const [relu, expected] of [[1, [-4, -4, 0, 24, 64]], [2, [-4, -4, 0, 20, 20]]]) {
        const spec = { ...base, params: { relu } };
        const cpu = await cpuResult(() => qaddGraph(spec));
        const { graph, inputs } = qaddGraph(spec);
        wasm.compile(graph);
        const result = await wasm.execute(inputs);
        assert.deepEqual([...result.out], [...cpu.out]);
        assert.deepEqual([...result.out], expected);
      }
    });

    await t.test('descriptor-preserving byte Expand feeds canonical exact-shape QAdd', async () => {
      const cpu = await cpuResult(expandedQAddGraph);
      const { graph, inputs } = expandedQAddGraph();
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [3, 7, 3, 78, -123, 12, 8, 78]);
    });

    await t.test('QConv2D matches per-axis OHWI I8 reference with I32 bias', async () => {
      const spec = {
        inputValues: [2, 4, 6, 8, 10, 12, 14, 16], inputShape: [1, 2, 2, 2],
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
        weightValues: [1, -1, 2, 1], weightShape: [2, 1, 1, 2],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.25, 0.5], zero_points: [0, 0],
        },
        biasValues: [1, -2], outputShape: [1, 2, 2, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 6, 0, 18, 0, 30, 0, 42]);
    });

    await t.test('eligible QConv2D uses reusable im2col plus packed SIMD128 QLinear', async () => {
      const inputValues = Int8Array.from(
        { length: 18 }, (_, index) => ((index * 11 + 5) % 31) - 15,
      );
      const weightValues = Int8Array.from(
        { length: 8 * 2 * 2 * 2 }, (_, index) => ((index * 17 + 3) % 23) - 11,
      );
      const spec = {
        inputValues, inputShape: [1, 3, 3, 2],
        inputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
        weightValues, weightShape: [8, 2, 2, 2],
        weightQuantization: {
          scheme: 'per_axis', axis: 0,
          scales: [0.0625, 0.125, 0.25, 0.0625, 0.125, 0.25, 0.0625, 0.125],
          zero_points: [0, -2, 1, 0, 2, -1, 0, 1],
        },
        biasValues: [13, -7, 5, -3, 11, -17, 19, -23],
        outputShape: [1, 2, 2, 8],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 1 },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
      assert.ok(descriptor.packedWeightPointer > 0,
        'immutable OHWI weights must be packed once at compile time');
      assert.equal(descriptor.im2colBytes, 4 * 8);
      assert.ok(wasm.qconvIm2ColPointer > 0);
      assert.ok(wasm.qconvIm2ColBytes >= descriptor.im2colBytes);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);

      const smallerSpec = {
        inputValues: [7], inputShape: [1, 1, 1, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -2 },
        weightValues: [1, -2, 3, -4, 5, -6, 7, -8],
        weightShape: [8, 1, 1, 1],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: new Array(8).fill(0.125),
          zero_points: new Array(8).fill(0),
        },
        biasValues: new Array(8).fill(0), outputShape: [1, 1, 1, 8],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      };
      const smallerCpu = await cpuResult(() => qconvGraph(smallerSpec));
      const smaller = qconvGraph(smallerSpec);
      wasm.compile(smaller.graph);
      assert.equal(wasm.qconvIm2ColBytes, 1,
        'recompile must replace, rather than retain, the preceding graph scratch size');
      assert.ok(wasm.qconvIm2ColPointer > 0);
      const smallerResult = await wasm.execute(smaller.inputs);
      assert.deepEqual([...smallerResult.out], [...smallerCpu.out]);
    });

    await t.test('packed 3x3 QConv2D preserves padded-row parity', async () => {
      const channels = 19;
      const inputValues = Uint8Array.from(
        { length: 4 * 5 * channels }, (_, index) => (index * 29 + 17) % 251,
      );
      const weightValues = Int8Array.from(
        { length: 8 * 3 * 3 * channels }, (_, index) => ((index * 19 + 7) % 127) - 63,
      );
      const spec = {
        inputDtype: 'uint8', inputValues, inputShape: [1, 4, 5, channels],
        inputQuantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 137 },
        weightValues, weightShape: [8, 3, 3, channels],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: new Array(8).fill(0.015625),
          zero_points: new Array(8).fill(0),
        },
        biasValues: [31, -47, 59, -71, 83, -97, 101, -103],
        outputDtype: 'uint8', outputShape: [1, 4, 5, 8],
        outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 113 },
        params: { stride: [1, 1], dilation: [1, 1], pads: [1, 1, 1, 1] },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
      assert.ok(descriptor.packedWeightPointer > 0);
      assert.equal(descriptor.im2colBytes, 4 * 5 * 3 * 3 * channels);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('accelerated QConv2D preserves U8 asymmetric padding, dilation, and N-tail parity', async () => {
      const inputValues = Uint8Array.from(
        { length: 5 * 6 * 3 }, (_, index) => (index * 29 + 17) % 251,
      );
      const weightValues = Uint8Array.from(
        { length: 9 * 2 * 2 * 3 }, (_, index) => (index * 37 + 11) % 253,
      );
      const spec = {
        inputDtype: 'uint8', inputValues, inputShape: [1, 5, 6, 3],
        inputQuantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 137 },
        weightDtype: 'uint8', weightValues, weightShape: [9, 2, 2, 3],
        weightQuantization: {
          scheme: 'per_axis', axis: 0,
          scales: [0.015625, 0.03125, 0.0625, 0.125, 0.015625,
            0.03125, 0.0625, 0.125, 0.015625],
          zero_points: [127, 3, 200, 19, 241, 0, 128, 77, 252],
        },
        biasValues: [31, -47, 59, -71, 83, -97, 101, -103, 107],
        outputDtype: 'uint8', outputShape: [1, 2, 8, 9],
        outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 113 },
        params: { stride: [2, 1], dilation: [2, 1], pads: [1, 2, 0, 1] },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
      assert.ok(descriptor.packedWeightPointer > 0);
      assert.equal(descriptor.im2colBytes, 2 * 8 * 2 * 2 * 3);

      const originalApi = wasm.api;
      const calls = { im2col: 0, packed: 0, canonical: 0 };
      wasm.api = {
        ...originalApi,
        qconv2d_im2col_i8u8: (...args) => {
          calls.im2col++;
          return originalApi.qconv2d_im2col_i8u8(...args);
        },
        qlinear_i8u8_packed: (...args) => {
          calls.packed++;
          return originalApi.qlinear_i8u8_packed(...args);
        },
        qconv2d_i8u8: (...args) => {
          calls.canonical++;
          return originalApi.qconv2d_i8u8(...args);
        },
      };
      let result;
      try {
        result = await wasm.execute(inputs);
      } finally {
        wasm.api = originalApi;
      }
      assert.deepEqual(calls, { im2col: 1, packed: 1, canonical: 0 });
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('packed QConv2D keeps canonical descriptor-wide I32 overflow rejection', async () => {
      const reduction = 33026;
      const spec = {
        inputDtype: 'uint8', inputValues: new Uint8Array(reduction),
        inputShape: [1, 1, 1, reduction],
        inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
        weightDtype: 'uint8', weightValues: new Uint8Array(8 * reduction),
        weightShape: [8, 1, 1, reduction],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: new Array(8).fill(1),
          zero_points: new Array(8).fill(0),
        },
        biasValues: new Array(8).fill(0), outputDtype: 'uint8',
        outputShape: [1, 1, 1, 8],
        outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      };
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
      assert.ok(descriptor.packedWeightPointer > 0,
        'the regression descriptor must reach the accelerated route preflight');
      await assert.rejects(
        wasm.execute(inputs),
        /rejected its canonical W8A8 descriptor/,
      );
    });

    await t.test('groups, activation, and missing bias independently retain canonical QConv2D', () => {
      const quantization = {
        scheme: 'per_axis', axis: 0, scales: new Array(8).fill(0.125),
        zero_points: new Array(8).fill(0),
      };
      const common = {
        inputValues: [3, -5], inputShape: [1, 1, 1, 2],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        outputShape: [1, 1, 1, 8],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      };
      const cases = [
        ['groups', qconvGraph({
          ...common, weightValues: new Array(8).fill(1), weightShape: [8, 1, 1, 1],
          weightQuantization: quantization, biasValues: new Array(8).fill(0),
          params: { groups: 2 },
        })],
        ['relu', qconvGraph({
          ...common, weightValues: new Array(16).fill(1), weightShape: [8, 1, 1, 2],
          weightQuantization: quantization, biasValues: new Array(8).fill(0),
          params: { relu: 1 },
        })],
        ['missing bias', qconvGraph({
          ...common, weightValues: new Array(16).fill(1), weightShape: [8, 1, 1, 2],
          weightQuantization: quantization, biasValues: null,
        })],
      ];
      for (const [label, { graph }] of cases) {
        wasm.compile(graph);
        const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
        assert.equal(descriptor.packedWeightPointer, 0, `${label} must not pack a weight`);
        assert.equal(descriptor.im2colBytes, 0, `${label} must not reserve im2col scratch`);
      }
    });

    await t.test('whole-output QConv2D im2col scratch is capped at 64 MiB', () => {
      const graph = new RuntimeGraph();
      const width = 64 * 1024 * 1024 + 1;
      const input = graph.addInput('oversized_input', [1, 1, width, 1], 'int8', {
        quantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      });
      const weight = graph.addWeight('oversized_weight', [8, 1, 1, 1], 'int8', {
        buffer: new Int8Array(8),
        quantization: {
          scheme: 'per_axis', axis: 0, scales: new Array(8).fill(1),
          zero_points: new Array(8).fill(0),
        },
      });
      const bias = graph.addWeight('oversized_bias', [8], 'int32', {
        buffer: new Int32Array(8),
      });
      graph.addOp('QConv2D', { input, weight, bias }, {
        out: {
          name: 'oversized_out', shape: [1, 1, width, 8], dtype: 'int8',
          quantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
        },
      }, { data_layout: 'NHWC', weight_layout: 'OHWI' });
      const metadata = wasm._compilePortableMetadata(graph.nodes[0]);
      assert.equal(metadata.packedWeightPointer, 0);
      assert.equal(metadata.im2colBytes, 0);
    });

    await t.test('QConv2D applies output-domain ReLU without a CPU fallback', async () => {
      const spec = {
        inputValues: [-4, 4], inputShape: [1, 1, 2, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
        weightValues: [1], weightShape: [1, 1, 1, 1],
        weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.5], zero_points: [0] },
        outputShape: [1, 1, 2, 1],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        params: { relu: 1 },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 4]);
    });

    await t.test('QConv2D maps an overflowing zero accumulator to the output zero point', async () => {
      const spec = {
        inputValues: [1], inputShape: [1, 1, 1, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 3e38, zero_point: 1 },
        weightValues: [1], weightShape: [1, 1, 1, 1],
        weightQuantization: { scheme: 'per_axis', axis: 0, scales: [3e38], zero_points: [1] },
        outputShape: [1, 1, 1, 1],
        outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: -7 },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...cpu.out], [-7]);
      assert.deepEqual([...result.out], [-7]);
    });

    await t.test('grouped, padded, strided, dilated QConv2D applies ReLU6 in typed output space', async () => {
      const inputValues = Int8Array.from({ length: 100 }, (_, index) => ((index * 13) % 31) - 15);
      const weightValues = Uint8Array.from({ length: 32 }, (_, index) => [255, 0, 128, 3, 200, 10, 64, 240][index % 8]);
      const spec = {
        inputDtype: 'int8', inputValues, inputShape: [1, 5, 5, 4],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        weightDtype: 'uint8', weightValues, weightShape: [4, 2, 2, 2],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.25, 0.5, 0.125, 0.25], zero_points: [128, 2, 200, 10],
        },
        biasValues: [0, 1, -20, 35],
        outputDtype: 'int8', outputShape: [1, 3, 5, 4],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -4 },
        params: { groups: 2, stride: [2, 1], dilation: [2, 1], pads: [1, 1, 1, 0], relu: 2 },
      };
      const cpu = await cpuResult(() => qconvGraph(spec));
      const { graph, inputs } = qconvGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.ok([...result.out].every((value) => value >= -4 && value <= 20), 'ReLU6 stays in output quantization range');
    });

    await t.test('RequantizeLinear → QConv2D → QAdd remains a direct byte WASM island', async () => {
      const cpuGraph = typedW8A8IslandGraph();
      const cpu = new CPUEngine();
      cpu.allocateGraph(cpuGraph);
      const expected = await cpu.execute({ input: Uint8Array.of(128, 129, 130, 131) });
      const graph = typedW8A8IslandGraph();
      wasm.compile(graph);
      const result = await wasm.execute({ input: Uint8Array.of(128, 129, 130, 131) });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...expected.out]);
      assert.deepEqual([...result.out], [0, 2, 4, 6]);
    });

    await t.test('QAdd rejects broadcasting and QConv2D rejects non-axis-0 weight quantization', () => {
      const broadcast = qaddGraph({
        aValues: [1, 2], aShape: [1, 2],
        aQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        bValues: [3], bShape: [1, 1],
        bQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      });
      assert.throws(() => wasm.compile(broadcast.graph), /exact-shape.*I8\/U8/);

      const nonAxisZero = qconvGraph({
        inputValues: [1], inputShape: [1, 1, 1, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        weightValues: [1], weightShape: [1, 1, 1, 1],
        weightQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        outputShape: [1, 1, 1, 1],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      });
      assert.throws(() => wasm.compile(nonAxisZero.graph), /axis-0 per-channel weight metadata/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
