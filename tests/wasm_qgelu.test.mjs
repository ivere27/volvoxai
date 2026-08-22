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

function bytes(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function qGELUGraph({
  inputDtype = 'int8',
  inputValues = [-4, -1, 0, 2, 5],
  inputShape = [1, 5],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
  params = {},
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: bytes(inputDtype, inputValues), quantization: inputQuantization,
  });
  const { out } = graph.addOp('QGELU', { input }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype,
      quantization: outputQuantization,
    },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, inputValues: bytes(inputDtype, inputValues), out };
}

async function cpuResult(spec) {
  const { graph, inputValues } = qGELUGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute({ input: inputValues });
}

test('portable WASM QGELU preserves W8A8 storage with the fixed portable-erf ABI', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qgelu-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qgelu_i8u8, 'function', 'forward WASM exports qgelu_i8u8');
    wasm._cpuQGELU = () => { throw new Error('WASM QGELU must not use the JS reference'); };

    await t.test('I8 input/output matches the CPU portable-erf byte-domain reference', async () => {
      const spec = {};
      const cpu = await cpuResult(spec);
      const { graph, inputValues } = qGELUGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [-1, 0, 1, 5, 11]);
    });

    await t.test('all I8/U8 input/output combinations remain bytes', async () => {
      const inputVariants = [
        {
          inputDtype: 'int8', inputValues: [-4, -1, 0, 2, 5],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        },
        {
          inputDtype: 'uint8', inputValues: [124, 127, 128, 130, 133],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 127 },
        },
      ];
      const outputVariants = [
        { outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 } },
        { outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 121 } },
      ];
      for (const inputVariant of inputVariants) {
        for (const outputVariant of outputVariants) {
          const spec = { ...inputVariant, ...outputVariant };
          const cpu = await cpuResult(spec);
          const { graph, inputValues } = qGELUGraph(spec);
          wasm.compile(graph);
          const result = await wasm.execute({ input: inputValues });
          assert.ok(spec.outputDtype === 'int8'
            ? result.out instanceof Int8Array
            : result.out instanceof Uint8Array);
          assert.deepEqual([...result.out], [...cpu.out]);
        }
      }
    });

    await t.test('large calls build and reuse the exact byte LUT', async () => {
      const allI8 = Array.from({ length: 256 }, (_, index) => index - 128);
      const descriptor = {
        inputQuantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 0 },
        outputQuantization: { scheme: 'per_tensor', scale: 0.015625, zero_point: -3 },
      };
      const scalarSpec = {
        ...descriptor,
        inputValues: allI8,
        inputShape: [1, allI8.length],
      };
      const scalarGraph = qGELUGraph(scalarSpec);
      wasm.compile(scalarGraph.graph);
      const scalar = [...(await wasm.execute({ input: scalarGraph.inputValues })).out];

      const lutValues = [...allI8, ...allI8];
      const lutSpec = {
        ...descriptor,
        inputValues: lutValues,
        inputShape: [1, lutValues.length],
      };
      const lutGraph = qGELUGraph(lutSpec);
      wasm.compile(lutGraph.graph);
      const first = [...(await wasm.execute({ input: lutGraph.inputValues })).out];
      const second = [...(await wasm.execute({ input: lutGraph.inputValues })).out];
      assert.deepEqual(first, [...scalar, ...scalar]);
      assert.deepEqual(second, first);
    });

    await t.test('a rejected direct C descriptor leaves the destination bytes untouched', () => {
      const inputPointer = wasm.api.alloc_bytes(4);
      const outputPointer = wasm.api.alloc_bytes(4);
      new Int8Array(wasm.mem.buffer, inputPointer, 4).set(Int8Array.of(-4, -1, 0, 2));
      const output = new Uint8Array(wasm.mem.buffer, outputPointer, 4);
      output.fill(73);
      assert.equal(wasm.api.qgelu_i8u8(
        inputPointer, outputPointer, 4,
        0, 0, 0.125, 0, 2, 2,
      ), 0);
      assert.deepEqual([...output], [73, 73, 73, 73]);
    });

    await t.test('only omitted parameters or approximate=none compile for the fixed portable erf path', () => {
      const explicitNone = qGELUGraph({ params: { approximate: 'none' } });
      assert.doesNotThrow(() => wasm.compile(explicitNone.graph));

      const tanh = qGELUGraph({ params: { approximate: 'tanh' } });
      assert.throws(
        () => wasm.compile(tanh.graph),
        /only omitted parameters or approximate='none'/,
      );
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
