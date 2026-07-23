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

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

function castGraph(inputDtype, outputDtype, to) {
  const graph = new Graph();
  const input = graph.addInput('input', [4], inputDtype);
  const { out } = graph.addOp('Cast', { input }, {
    out: { name: 'out', shape: [4], dtype: outputDtype },
  }, { to });
  graph.setOutputs([out.name]);
  return graph;
}

function dequantGraph(inputDtype, zeroDtype) {
  const graph = new Graph();
  const input = graph.addInput('input', [4], inputDtype);
  const zeroPoint = graph.addInput('zero', [1], zeroDtype);
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  const { out } = graph.addOp('DequantizeLinear', { input, scale, zero_point: zeroPoint }, {
    out: { name: 'out', shape: [4], dtype: 'float32' },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function quantizeGraph(outputDtype, zeroDtype, zeroValue) {
  const graph = new Graph();
  const input = graph.addInput('input', [10], 'float32');
  const zeroPoint = graph.addInput('zero', [1], zeroDtype);
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  const { out } = graph.addOp('QuantizeLinear', { input, scale, zero_point: zeroPoint }, {
    out: {
      name: 'out', shape: [10], dtype: outputDtype,
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: zeroValue },
    },
  });
  graph.setOutputs([out.name]);
  return graph;
}

async function cpuResult(graph, inputs) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

test('portable WASM Cast and DequantizeLinear use typed C kernels', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-cast-dequant-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');

    await t.test('Cast preserves integer truncation and typed output storage', async () => {
      const values = Float32Array.of(1.9, -2.7, 3, 4.2);
      const cpu = await cpuResult(castGraph('float32', 'int8', 'int8'), { input: values });
      const graph = castGraph('float32', 'int8', 'int8');
      wasm.compile(graph);
      const result = await wasm.execute({ input: values });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [1, -2, 3, 4]);
    });

    await t.test('Cast converts byte input back to F32 without a JS fallback', async () => {
      const values = Uint8Array.of(0, 17, 200, 255);
      const cpu = await cpuResult(castGraph('uint8', 'float32', 'float32'), { input: values });
      const graph = castGraph('uint8', 'float32', 'float32');
      wasm.compile(graph);
      const result = await wasm.execute({ input: values });
      assert.ok(result.out instanceof Float32Array);
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('Cast preserves high-range I32 bits and F32 TypedArray wrapping', async () => {
      const integerInput = Int32Array.of(16_777_217, -1, 255, 256);
      const integerCpu = await cpuResult(castGraph('int32', 'int8', 'int8'), { input: integerInput });
      const integerGraph = castGraph('int32', 'int8', 'int8');
      wasm.compile(integerGraph);
      const integerResult = await wasm.execute({ input: integerInput });
      assert.deepEqual([...integerResult.out], [...integerCpu.out]);
      assert.deepEqual([...integerResult.out], [1, -1, -1, 0]);

      const wrappingInput = Float32Array.of(2_147_483_648, 4_294_967_808, 0, -1.9);
      const wrappingCpu = await cpuResult(castGraph('float32', 'int32', 'int32'), { input: wrappingInput });
      const wrappingGraph = castGraph('float32', 'int32', 'int32');
      wasm.compile(wrappingGraph);
      const wrappingResult = await wasm.execute({ input: wrappingInput });
      assert.deepEqual([...wrappingResult.out], [...wrappingCpu.out]);
      assert.deepEqual([...wrappingResult.out], [-2_147_483_648, 512, 0, -1]);

      const nonFiniteInput = Float32Array.of(Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, -0);
      const nonFiniteCpu = await cpuResult(castGraph('float32', 'int32', 'int32'), { input: nonFiniteInput });
      const nonFiniteGraph = castGraph('float32', 'int32', 'int32');
      wasm.compile(nonFiniteGraph);
      const nonFiniteResult = await wasm.execute({ input: nonFiniteInput });
      assert.deepEqual([...nonFiniteResult.out], [...nonFiniteCpu.out]);
      assert.deepEqual([...nonFiniteResult.out], [0, 0, 0, 0]);
    });

    await t.test('DequantizeLinear reads quantized input and zero point directly', async () => {
      const inputs = { input: Uint8Array.of(2, 6, 10, 14), zero: Uint8Array.of(2) };
      const cpu = await cpuResult(dequantGraph('uint8', 'uint8'), inputs);
      const graph = dequantGraph('uint8', 'uint8');
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Float32Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 1, 2, 3]);
    });

    await t.test('DequantizeLinear subtracts high-range I32 values before F32 conversion', async () => {
      const inputs = {
        input: Int32Array.of(16_777_217, 16_777_216, 16_777_218, 16_777_215),
        zero: Int32Array.of(16_777_216),
      };
      const cpu = await cpuResult(dequantGraph('int32', 'int32'), inputs);
      const graph = dequantGraph('int32', 'int32');
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0.25, 0, 0.5, -0.25]);
    });

    await t.test('QuantizeLinear writes signed byte activations with ties-to-even rounding', async () => {
      const inputs = {
        input: Float32Array.of(-100, -0.875, -0.625, -0.375, -0.125, 0, 0.125, 0.375, 100, Number.NaN),
        zero: Int8Array.of(0),
      };
      const cpu = await cpuResult(quantizeGraph('int8', 'int8', 0), inputs);
      const graph = quantizeGraph('int8', 'int8', 0);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [-128, -4, -2, -2, 0, 0, 0, 2, 127, 0]);
    });

    await t.test('QuantizeLinear writes asymmetric unsigned byte activations', async () => {
      const inputs = {
        input: Float32Array.of(-100, -32, -31.875, -31.5, 0, 0.125, 0.375, 31.75, 100, Number.POSITIVE_INFINITY),
        zero: Uint8Array.of(128),
      };
      const cpu = await cpuResult(quantizeGraph('uint8', 'uint8', 128), inputs);
      const graph = quantizeGraph('uint8', 'uint8', 128);
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 0, 0, 2, 128, 128, 130, 255, 255, 255]);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
