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

function storage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function requantizeGraph({ inputDtype, inputQuantization, outputDtype, outputQuantization } = {}) {
  const graph = new Graph();
  const input = graph.addInput('input', [7], inputDtype, { quantization: inputQuantization });
  const { out } = graph.addOp('RequantizeLinear', { input }, {
    out: { name: 'out', shape: [7], dtype: outputDtype, quantization: outputQuantization },
  });
  graph.outputNames = [out.name];
  return graph;
}

async function cpuResult(spec, values) {
  const graph = requantizeGraph(spec);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute({ input: storage(spec.inputDtype, values) });
}

test('portable WASM RequantizeLinear uses only typed tensor metadata', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-requantize-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.requantize_linear_i8u8, 'function');
    wasm._cpuRequantizeLinear = () => {
      throw new Error('metadata-only RequantizeLinear WASM dispatch must not call the CPU reference');
    };

    await t.test('asymmetric U8 to I8 conversion preserves the represented real values', async () => {
      const spec = {
        inputDtype: 'uint8', inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
      };
      const values = [0, 124, 127, 128, 129, 132, 255];
      const cpu = await cpuResult(spec, values);
      const graph = requantizeGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: storage(spec.inputDtype, values) });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [-65, -3, -2, -1, 0, 1, 62]);
    });

    await t.test('ties-to-even and destination saturation match the CPU reference', async () => {
      const spec = {
        inputDtype: 'int8', inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
        outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      };
      const values = [-128, -65, -1, 0, 1, 64, 127];
      const cpu = await cpuResult(spec, values);
      const graph = requantizeGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: storage(spec.inputDtype, values) });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 0, 126, 128, 130, 255, 255]);
    });

    await t.test('compile rejects absent immutable output metadata', () => {
      const graph = new Graph();
      const input = graph.addInput('input', [1], 'int8', {
        quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      });
      graph.addOp('RequantizeLinear', { input }, {
        out: { name: 'out', shape: [1], dtype: 'int8' },
      });
      assert.throws(() => wasm.compile(graph), /equal-shape canonical per-tensor I8\/U8 edges/);

      const mutableScale = new Graph();
      const typedInput = mutableScale.addInput('input', [1], 'int8', {
        quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      });
      const scale = mutableScale.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.5) });
      mutableScale.addOp('RequantizeLinear', { input: typedInput, scale }, {
        out: {
          name: 'out', shape: [1], dtype: 'int8',
          quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
        },
      });
      assert.throws(() => wasm.compile(mutableScale), /unsupported canonical W8A8 input 'scale'/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
