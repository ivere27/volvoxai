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

function elementCount(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function qLayerNormGraph({
  inputShape = [2, 4],
  inputDtype = 'int8',
  inputValues = [-8, -3, 4, 7, 5, -1, 2, -6],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  gamma = [1, -0.75, 0.5, 1.25],
  beta = [0.25, -0.5, 0.75, -0.25],
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
  eps = 1e-5,
  dModel,
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: bytes(inputDtype, inputValues), quantization: inputQuantization,
  });
  const d = inputShape.at(-1);
  const weight = graph.addWeight('weight', [d], 'float32', { buffer: Float32Array.from(gamma) });
  const bias = graph.addWeight('bias', [d], 'float32', { buffer: Float32Array.from(beta) });
  const params = { eps };
  if (dModel !== undefined) params.d_model = dModel;
  const { out } = graph.addOp('QLayerNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype, quantization: outputQuantization,
    },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, inputValues: bytes(inputDtype, inputValues), input, weight, bias, out };
}

function qLayerNormDynamicAffineGraph() {
  const graph = new RuntimeGraph();
  const inputValues = Int8Array.of(-8, -3, 4, 7, 5, -1, 2, -6);
  const input = graph.addInput('input', [2, 4], 'int8', {
    buffer: inputValues, quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  // Dynamic affine placeholders can be nonfinite at compile time. execute()
  // supplies and validates the actual F32 values before native execution.
  const weight = graph.addInput('weight', [4], 'float32', {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const bias = graph.addInput('bias', [4], 'float32', {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const { out } = graph.addOp('QLayerNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: [2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    },
  }, {});
  graph.setOutputs([out.name]);
  return { graph, inputValues, out };
}

async function cpuResult(spec) {
  const { graph, inputValues } = qLayerNormGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute({ input: inputValues });
}

test('portable WASM QLayerNorm preserves I8/U8 activation storage', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qlayernorm-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qlayernorm_i8u8, 'function',
      'forward WASM exports qlayernorm_i8u8');
    wasm._cpuQLayerNorm = () => {
      throw new Error('WASM QLayerNorm must not use the JavaScript reference');
    };
    wasm._cpuLayerNorm = () => {
      throw new Error('WASM QLayerNorm must not materialize an F32 activation');
    };

    await t.test('canonical signed rows match the byte-domain CPU reference', async () => {
      const cpu = await cpuResult({});
      const { graph, inputValues } = qLayerNormGraph();
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [-12, -4, 6, 7, 9, -6, 5, -20]);
    });

    await t.test('every I8/U8 activation pair remains bytes', async () => {
      const inputs = [
        {
          inputDtype: 'int8', inputValues: [-8, -3, 4, 7, 5, -1, 2, -6],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        },
        {
          inputDtype: 'uint8', inputValues: [120, 125, 132, 135, 133, 127, 130, 122],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        },
      ];
      const outputs = [
        { outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -7 } },
        { outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 123 } },
      ];
      for (const input of inputs) {
        for (const output of outputs) {
          const spec = { ...input, ...output };
          const cpu = await cpuResult(spec);
          const { graph, inputValues } = qLayerNormGraph(spec);
          wasm.compile(graph);
          const result = await wasm.execute({ input: inputValues });
          assert.ok(spec.outputDtype === 'int8'
            ? result.out instanceof Int8Array
            : result.out instanceof Uint8Array);
          assert.deepEqual([...result.out], [...cpu.out]);
        }
      }
    });

    await t.test('D=1 and high near-constant U8 D=320 rows stay in the byte domain', async () => {
      const single = {
        inputShape: [3, 1], inputValues: [-121, 0, 117], gamma: [17], beta: [0.5],
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -2 }, dModel: 1,
      };
      const singleGraph = qLayerNormGraph(single);
      wasm.compile(singleGraph.graph);
      const singleResult = await wasm.execute({ input: singleGraph.inputValues });
      assert.deepEqual([...singleResult.out], [0, 0, 0]);

      const inputShape = [4, 320];
      const inputValues = Array.from({ length: elementCount(inputShape) }, (_, index) =>
        index % 2 === 0 ? 240 : 241,
      );
      const spec = {
        inputShape, inputDtype: 'uint8', inputValues,
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 17 },
        gamma: new Array(320).fill(1), beta: new Array(320).fill(0),
        outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      };
      const { graph, inputValues: physicalInput } = qLayerNormGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: physicalInput });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], inputValues.map((value) => value === 240 ? 124 : 132));
    });

    await t.test('dynamic F32 affine inputs validate execution values, not NaN placeholders', async () => {
      const expected = await cpuResult({});
      const { graph, inputValues } = qLayerNormDynamicAffineGraph();
      assert.doesNotThrow(() => wasm.compile(graph));
      await assert.rejects(
        () => wasm.execute({
          input: inputValues,
          weight: Float32Array.of(1, Number.NaN, 0.5, 1.25),
          bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
        }),
        /weight must contain finite F32 values before execution/,
      );
      await assert.rejects(
        () => wasm.execute({
          input: inputValues, weight: Float32Array.of(1, -0.75, 0.5, 1.25),
        }),
        /graph-input F32 bias supplied on every execution/,
      );
      const result = await wasm.execute({
        input: inputValues,
        weight: Float32Array.of(1, -0.75, 0.5, 1.25),
        bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
      });
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('a rejected direct C descriptor leaves destination bytes untouched', () => {
      const inputPointer = wasm.api.alloc_bytes(4);
      const weightPointer = wasm.api.alloc_bytes(4 * Float32Array.BYTES_PER_ELEMENT);
      const biasPointer = wasm.api.alloc_bytes(4 * Float32Array.BYTES_PER_ELEMENT);
      const outputPointer = wasm.api.alloc_bytes(4);
      new Int8Array(wasm.mem.buffer, inputPointer, 4).set(Int8Array.of(-4, -1, 0, 2));
      new Float32Array(wasm.mem.buffer, weightPointer, 4).set(Float32Array.of(1, 1, 1, 1));
      new Float32Array(wasm.mem.buffer, biasPointer, 4).set(Float32Array.of(0, 0, 0, 0));
      const output = new Uint8Array(wasm.mem.buffer, outputPointer, 4);
      output.fill(73);
      assert.equal(wasm.api.qlayernorm_i8u8(
        inputPointer, weightPointer, biasPointer, outputPointer,
        1, 0, 0.25, 0, 0.125, 0, 1e-5, 2, 2,
      ), 0);
      assert.deepEqual([...output], [73, 73, 73, 73]);
    });

    await t.test('only canonical byte rows and affine storage compile for the W8A8 path', () => {
      const valid = qLayerNormGraph({ dModel: 4 });
      assert.doesNotThrow(() => wasm.compile(valid.graph));

      const badDModel = qLayerNormGraph({ dModel: 3 });
      assert.throws(
        () => wasm.compile(badDModel.graph),
        /QLayerNorm node .*must use exact input\/weight\/bias inputs/,
      );

      const nonFiniteAffine = qLayerNormGraph();
      nonFiniteAffine.weight.buffer[2] = Number.NaN;
      assert.throws(() => wasm.compile(nonFiniteAffine.graph), /finite F32 \[D\] affine tensors/);

      const sharedStorage = qLayerNormGraph();
      const shared = new ArrayBuffer(8);
      const aliasedInput = new Int8Array(shared);
      aliasedInput.set([-8, -3, 4, 7, 5, -1, 2, -6]);
      sharedStorage.input.buffer = aliasedInput;
      sharedStorage.out.buffer = new Int8Array(shared);
      assert.throws(
        () => wasm.compile(sharedStorage.graph),
        /output storage distinct from every input/,
      );

      const rawAffineStorage = qLayerNormGraph();
      rawAffineStorage.weight.buffer = new ArrayBuffer(16);
      assert.throws(
        () => wasm.compile(rawAffineStorage.graph),
        /canonical typed input\/output storage/,
      );
    });

    await t.test('affine validation survives WASM memory growth during graph allocation', () => {
      const { graph } = qLayerNormGraph();
      graph.addInput('unused_growth', [4 * 1024 * 1024], 'int8');
      assert.doesNotThrow(() => wasm.compile(graph));
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
