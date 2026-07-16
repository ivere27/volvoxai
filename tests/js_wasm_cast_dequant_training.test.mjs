import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/index.js';
import { TrainingVolvoxAI } from '../ts/training/TrainingVolvoxAI.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';

function closeArray(actual, expected, label, tolerance = 5e-5) {
  assert.equal(actual.length, expected.length, `${label} length`);
  for (let index = 0; index < actual.length; index++) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance,
      `${label}[${index}]: ${actual[index]} != ${expected[index]}`);
  }
}

async function buildFullWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output,
    'native/src/kernels/kernels.c', 'native/src/kernels/training_kernels.c',
  ], { cwd: repositoryRoot });
}

function output(graph, opType, inputs, shape, dtype = 'float32', params = {}) {
  return graph.addOp(opType, inputs, {
    out: { name: 'out', shape, dtype },
  }, params).out;
}

function floatCastCase() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', {
    buffer: Float32Array.of(0.25, -0.7),
  });
  const out = output(graph, 'Cast', { input }, [1, 2], 'float32', { to: 'float32' });
  graph.outputNames = [out.name];
  return { graph, output: out.name, trainable: ['input'], targets: [1], inputs: {} };
}

function f32DequantCase() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', {
    buffer: Float32Array.of(0.5, -0.25),
  });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.8) });
  const zero = graph.addWeight('zero', [1], 'float32', { buffer: Float32Array.of(0.1) });
  const out = output(graph, 'DequantizeLinear', { input, scale, zero_point: zero }, [1, 2]);
  graph.outputNames = [out.name];
  return { graph, output: out.name, trainable: ['input', 'scale'], targets: [0], inputs: {} };
}

function typedDequantCase(inputDtype, inputValues, zeroDtype, zeroValues) {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2], inputDtype);
  // Quantized zero points are static model constants, not optimizer weights.
  // The strict WASM planner must still retain their exact raw typed storage.
  const zero = graph.addWeight('zero', [1], zeroDtype, { buffer: zeroValues });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.25) });
  const out = output(graph, 'DequantizeLinear', { input, scale, zero_point: zero }, [1, 2]);
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    trainable: ['scale'],
    targets: [0],
    inputs: { input: inputValues },
  };
}

function quantizedCastCase(dtype) {
  const graph = new Graph();
  const source = graph.addWeight('source', [1, 2], 'float32', {
    buffer: Float32Array.of(1.9, 2.7),
  });
  const cast = graph.addOp('Cast', { input: source }, {
    out: { name: 'quantized', shape: [1, 2], dtype },
  }, { to: dtype }).out;
  const zero = graph.addWeight('zero', [1], dtype, {
    buffer: dtype === 'int32' ? Int32Array.of(0)
      : dtype === 'int8' ? Int8Array.of(0) : Uint8Array.of(0),
  });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.5) });
  const out = output(graph, 'DequantizeLinear', { input: cast, scale, zero_point: zero }, [1, 2]);
  graph.outputNames = [out.name];
  return { graph, output: out.name, trainable: ['scale'], targets: [1], inputs: {} };
}

function integerToFloatCastCase() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2], 'int8');
  const cast = graph.addOp('Cast', { input }, {
    out: { name: 'cast', shape: [1, 2], dtype: 'float32' },
  }, { to: 'float32' }).out;
  const factor = graph.addWeight('factor', [1], 'float32', { buffer: Float32Array.of(0.3) });
  const out = output(graph, 'Mul', { a: cast, b: factor }, [1, 2]);
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    trainable: ['factor'],
    targets: [0],
    inputs: { input: Int8Array.of(2, -3) },
  };
}

function highRangeIntegerCastCase() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2], 'int32');
  const copied = graph.addOp('Cast', { input }, {
    out: { name: 'i32copy', shape: [1, 2], dtype: 'int32' },
  }, { to: 'int32' }).out;
  const narrowed = graph.addOp('Cast', { input: copied }, {
    out: { name: 'i8copy', shape: [1, 2], dtype: 'int8' },
  }, { to: 'int8' }).out;
  const widened = graph.addOp('Cast', { input: narrowed }, {
    out: { name: 'widened', shape: [1, 2], dtype: 'float32' },
  }, { to: 'float32' }).out;
  const factor = graph.addWeight('factor', [1], 'float32', { buffer: Float32Array.of(0.3) });
  const out = output(graph, 'Mul', { a: widened, b: factor }, [1, 2]);
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    trainable: ['factor'],
    targets: [0],
    inputs: { input: Int32Array.of(16_777_217, -1) },
  };
}

function f32WrappingCastCase() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2], 'float32');
  const wrapped = graph.addOp('Cast', { input }, {
    out: { name: 'wrapped', shape: [1, 2], dtype: 'int32' },
  }, { to: 'int32' }).out;
  const widened = graph.addOp('Cast', { input: wrapped }, {
    out: { name: 'widened', shape: [1, 2], dtype: 'float32' },
  }, { to: 'float32' }).out;
  const factor = graph.addWeight('factor', [1], 'float32', { buffer: Float32Array.of(0) });
  const out = output(graph, 'Mul', { a: widened, b: factor }, [1, 2]);
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    trainable: ['factor'],
    targets: [1],
    inputs: { input: Float32Array.of(2_147_483_648, 4_294_967_808) },
  };
}

function highRangeI32DequantCase() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2], 'int32');
  const zero = graph.addWeight('zero', [1], 'int32', { buffer: Int32Array.of(16_777_216) });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(1) });
  const out = output(graph, 'DequantizeLinear', { input, scale, zero_point: zero }, [1, 2]);
  graph.outputNames = [out.name];
  return {
    graph,
    output: out.name,
    trainable: ['scale'],
    targets: [0],
    inputs: { input: Int32Array.of(16_777_217, 16_777_216) },
  };
}

function stoppedFloatToIntegerCastCase() {
  const graph = new Graph();
  const source = graph.addWeight('source', [1, 2], 'float32', {
    buffer: Float32Array.of(1.9, -2.7),
  });
  const cast = graph.addOp('Cast', { input: source }, {
    out: { name: 'quantized', shape: [1, 2], dtype: 'int8' },
  }, { to: 'int8' }).out;
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.4) });
  const out = output(graph, 'DequantizeLinear', { input: cast, scale }, [1, 2]);
  graph.outputNames = [out.name];
  return { graph, targets: [0] };
}

function stoppedZeroPointCase() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', {
    buffer: Float32Array.of(0.4, -0.2),
  });
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(0.5) });
  const zero = graph.addWeight('zero', [1], 'float32', { buffer: Float32Array.of(0.1) });
  const out = output(graph, 'DequantizeLinear', { input, scale, zero_point: zero }, [1, 2]);
  graph.outputNames = [out.name];
  return { graph, targets: [1] };
}

function badScaleCase() {
  const graph = new Graph();
  const input = graph.addWeight('input', [1, 2], 'float32', {
    buffer: Float32Array.of(1, -1),
  });
  const scale = graph.addWeight('scale', [2], 'float32', { buffer: Float32Array.of(0.25, 0.5) });
  const out = output(graph, 'DequantizeLinear', { input, scale }, [1, 2]);
  graph.outputNames = [out.name];
  return graph;
}

async function trainParity(api, makeCase, label) {
  const wasmCase = makeCase();
  const cpuCase = makeCase();
  const options = {
    targets: wasmCase.targets,
    trainableTensors: wasmCase.trainable,
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  };
  const wasm = await api.trainStep(wasmCase.graph, {
    ...options, inputs: wasmCase.inputs, backend: 'wasm',
  });
  const cpu = await CPUAutograd.trainStep(cpuCase.graph, {
    ...options, inputs: cpuCase.inputs,
  });
  assert.equal(wasm.backend, 'wasm');
  assert.ok(Math.abs(wasm.loss - cpu.loss) <= 5e-5, `${label} loss`);
  closeArray(wasmCase.graph.getTensor(wasmCase.output).buffer,
    cpuCase.graph.getTensor(cpuCase.output).buffer, `${label} output`);
  for (const name of wasmCase.trainable) {
    closeArray(wasm.gradients.get(name), cpu.gradients.get(name), `${label} ${name} gradient`);
  }
  return { wasmCase, cpuCase, wasm, cpu };
}

test('strict full-WASM Cast and DequantizeLinear retain typed storage and CPU gradients', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-cast-dequant-training-'));
  try {
    const wasmPath = join(directory, 'volvoxai.full.wasm');
    await buildFullWasm(wasmPath);
    const api = await TrainingVolvoxAI.init(['wasm'], wasmPath);

    await trainParity(api, floatCastCase, 'F32 Cast identity');
    await trainParity(api, f32DequantCase, 'F32 DequantizeLinear input and scale');
    await trainParity(api, integerToFloatCastCase, 'I8 to F32 Cast stops its input gradient');
    const highIntegerCast = await trainParity(api, highRangeIntegerCastCase,
      'high-range I32 Cast preserves integer bits before narrowing');
    assert.deepEqual([...highIntegerCast.wasmCase.graph.getTensor('i32copy').buffer], [16_777_217, -1]);
    assert.deepEqual([...highIntegerCast.wasmCase.graph.getTensor('i8copy').buffer], [1, -1]);
    const wrappingCast = await trainParity(api, f32WrappingCastCase,
      'F32 to I32 TypedArray modulo wrapping');
    assert.deepEqual([...wrappingCast.wasmCase.graph.getTensor('wrapped').buffer], [-2_147_483_648, 512]);
    const highI32Dequant = await trainParity(api, highRangeI32DequantCase,
      'high-range I32 DequantizeLinear subtraction');
    assert.deepEqual([...highI32Dequant.wasmCase.graph.getTensor(highI32Dequant.wasmCase.output).buffer], [1, 0]);

    for (const entry of [
      { inputDtype: 'int32', input: Int32Array.of(5, -3), zeroDtype: 'int32', zero: Int32Array.of(1) },
      { inputDtype: 'int8', input: Int8Array.of(2, -4), zeroDtype: 'int8', zero: Int8Array.of(-1) },
      { inputDtype: 'uint8', input: Uint8Array.of(2, 6), zeroDtype: 'uint8', zero: Uint8Array.of(2) },
    ]) {
      const result = await trainParity(api, () => typedDequantCase(
        entry.inputDtype, entry.input, entry.zeroDtype, entry.zero,
      ), `${entry.inputDtype} DequantizeLinear with static typed zero point`);
      assert.equal(result.wasmCase.graph.getTensor('zero').isWeight, true);
    }

    for (const dtype of ['int32', 'int8', 'uint8']) {
      await trainParity(api, () => quantizedCastCase(dtype), `F32 to ${dtype} Cast`);
    }

    const wasmStoppedCast = stoppedFloatToIntegerCastCase();
    await assert.rejects(() => api.trainStep(wasmStoppedCast.graph, {
      targets: wasmStoppedCast.targets,
      trainableTensors: ['source'],
      updateMode: 'sgd', optimizer: { learningRate: 0 }, backend: 'wasm',
    }), /No valid gradient reached trainable tensor 'source'/);
    const cpuStoppedCast = stoppedFloatToIntegerCastCase();
    await assert.rejects(() => CPUAutograd.trainStep(cpuStoppedCast.graph, {
      targets: cpuStoppedCast.targets,
      trainableTensors: ['source'],
      updateMode: 'sgd', optimizer: { learningRate: 0 },
    }), /No valid gradient reached trainable tensor 'source'/);

    const wasmStoppedZero = stoppedZeroPointCase();
    await assert.rejects(() => api.trainStep(wasmStoppedZero.graph, {
      targets: wasmStoppedZero.targets,
      trainableTensors: ['zero'],
      updateMode: 'sgd', optimizer: { learningRate: 0 }, backend: 'wasm',
    }), /No valid gradient reached trainable tensor 'zero'/);
    const cpuStoppedZero = stoppedZeroPointCase();
    await assert.rejects(() => CPUAutograd.trainStep(cpuStoppedZero.graph, {
      targets: cpuStoppedZero.targets,
      trainableTensors: ['zero'],
      updateMode: 'sgd', optimizer: { learningRate: 0 },
    }), /No valid gradient reached trainable tensor 'zero'/);

    await assert.rejects(() => api.trainStep(badScaleCase(), {
      targets: [0], trainableTensors: ['scale'], updateMode: 'sgd',
      optimizer: { learningRate: 0 }, backend: 'wasm',
    }), /scalar F32 scale/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
