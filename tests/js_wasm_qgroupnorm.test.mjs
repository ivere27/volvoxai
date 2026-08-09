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

function qGroupNormGraph({
  inputShape = [1, 2, 2, 4],
  inputDtype = 'int8',
  inputValues = [-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  gamma = [1, -0.75, 0.5, 1.25],
  beta = [0.25, -0.5, 0.75, -0.25],
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
  numGroups = 2,
  eps = 1e-5,
  dataLayout,
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: bytes(inputDtype, inputValues), quantization: inputQuantization,
  });
  const weight = graph.addWeight('weight', [inputShape[3]], 'float32', {
    buffer: Float32Array.from(gamma),
  });
  const bias = graph.addWeight('bias', [inputShape[3]], 'float32', {
    buffer: Float32Array.from(beta),
  });
  const params = { num_groups: numGroups, eps };
  if (dataLayout !== undefined) params.data_layout = dataLayout;
  const { out } = graph.addOp('QGroupNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype,
      quantization: outputQuantization,
    },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, inputValues: bytes(inputDtype, inputValues), out };
}

function qGroupNormDynamicAffineGraph() {
  const graph = new RuntimeGraph();
  const inputValues = Int8Array.of(-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2);
  const input = graph.addInput('input', [1, 2, 2, 4], 'int8', {
    buffer: inputValues,
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  // RuntimeGraphLoader intentionally permits dynamic F32 affine values. Their
  // compile-time placeholders may be nonfinite because execute() supplies
  // the actual tensors before qgroupnorm_i8u8 runs.
  const weight = graph.addInput('weight', [4], 'float32', {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const bias = graph.addInput('bias', [4], 'float32', {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const { out } = graph.addOp('QGroupNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: [1, 2, 2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    },
  }, { num_groups: 2, eps: 1e-5 });
  graph.setOutputs([out.name]);
  return { graph, inputValues, out };
}

async function cpuResult(spec) {
  const { graph, inputValues } = qGroupNormGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute({ input: inputValues });
}

test('portable WASM QGroupNorm preserves I8/U8 activation storage', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qgroupnorm-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qgroupnorm_i8u8, 'function',
      'forward WASM exports qgroupnorm_i8u8');
    wasm._cpuQGroupNorm = () => {
      throw new Error('WASM QGroupNorm must not use the JavaScript reference');
    };
    wasm._cpuGroupNorm = () => {
      throw new Error('WASM QGroupNorm must not materialize an F32 activation');
    };

    await t.test('canonical signed NHWC groups match the byte-domain CPU reference', async () => {
      const spec = {};
      const cpu = await cpuResult(spec);
      const { graph, inputValues } = qGroupNormGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [-12, -4, 6, 11, 6, -6, 4, -21,
        -1, -16, -2, 1, -11, -13, 3, -11]);
    });

    await t.test('separate NHWC samples use their own byte-domain group statistics', async () => {
      const spec = {
        inputShape: [2, 1, 1, 3], inputValues: [-5, 0, 4, 6, -2, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
        gamma: [1, 0.5, -0.75], beta: [0.25, -0.5, 0.75],
        outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
        numGroups: 1,
      };
      const cpu = await cpuResult(spec);
      const { graph, inputValues } = qGroupNormGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('every I8/U8 activation pair remains physical bytes', async () => {
      const inputVariants = [
        {
          inputDtype: 'int8',
          inputValues: [-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        },
        {
          inputDtype: 'uint8',
          inputValues: [120, 125, 132, 135, 133, 127, 130, 122,
            128, 136, 124, 131, 121, 134, 129, 126],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        },
      ];
      const outputVariants = [
        { outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -7 } },
        { outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 123 } },
      ];
      for (const inputVariant of inputVariants) {
        for (const outputVariant of outputVariants) {
          const spec = { ...inputVariant, ...outputVariant };
          const cpu = await cpuResult(spec);
          const { graph, inputValues } = qGroupNormGraph(spec);
          wasm.compile(graph);
          const result = await wasm.execute({ input: inputValues });
          assert.ok(spec.outputDtype === 'int8'
            ? result.out instanceof Int8Array
            : result.out instanceof Uint8Array);
          assert.deepEqual([...result.out], [...cpu.out]);
        }
      }
    });

    await t.test('dynamic F32 affine inputs validate their execution values, not NaN placeholders', async () => {
      const expected = await cpuResult({});
      const { graph, inputValues } = qGroupNormDynamicAffineGraph();
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
          input: inputValues,
          weight: Float32Array.of(1, -0.75, 0.5, 1.25),
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

    await t.test('centered raw variance stays stable for nearly constant U8 groups', async () => {
      const inputShape = [1, 16, 16, 4];
      const inputValues = Array.from({ length: 16 * 16 * 4 }, (_, index) =>
        index % 2 === 0 ? 240 : 241);
      const spec = {
        inputShape, inputDtype: 'uint8', inputValues,
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 17 },
        gamma: [1, 1, 1, 1], beta: [0, 0, 0, 0],
        outputDtype: 'uint8',
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        numGroups: 1,
      };
      const { graph, inputValues: physicalInput } = qGroupNormGraph(spec);
      wasm.compile(graph);
      const result = await wasm.execute({ input: physicalInput });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], inputValues.map((value) => value === 240 ? 124 : 132));
    });

    await t.test('a rejected direct C descriptor leaves the destination bytes untouched', () => {
      const inputPointer = wasm.api.alloc_bytes(4);
      const weightPointer = wasm.api.alloc_bytes(4 * Float32Array.BYTES_PER_ELEMENT);
      const biasPointer = wasm.api.alloc_bytes(4 * Float32Array.BYTES_PER_ELEMENT);
      const outputPointer = wasm.api.alloc_bytes(4);
      new Int8Array(wasm.mem.buffer, inputPointer, 4).set(Int8Array.of(-4, -1, 0, 2));
      new Float32Array(wasm.mem.buffer, weightPointer, 4).set(Float32Array.of(1, 1, 1, 1));
      new Float32Array(wasm.mem.buffer, biasPointer, 4).set(Float32Array.of(0, 0, 0, 0));
      const output = new Uint8Array(wasm.mem.buffer, outputPointer, 4);
      output.fill(73);
      assert.equal(wasm.api.qgroupnorm_i8u8(
        inputPointer, weightPointer, biasPointer, outputPointer,
        1, 1, 1, 4, 0,
        0.25, 0, 0.125, 0, 1e-5, 2, 2,
      ), 0);
      assert.deepEqual([...output], [73, 73, 73, 73]);
    });

    await t.test('only canonical rank-4 NHWC descriptors compile for the W8A8 path', () => {
      const valid = qGroupNormGraph({ dataLayout: 'NHWC' });
      assert.doesNotThrow(() => wasm.compile(valid.graph));

      const invalidGroups = qGroupNormGraph({ numGroups: 3 });
      assert.throws(
        () => wasm.compile(invalidGroups.graph),
        /QGroupNorm node .*must use exact input\/weight\/bias inputs/,
      );

      const nonFiniteAffine = qGroupNormGraph();
      nonFiniteAffine.graph.tensors.get('weight').buffer[2] = Number.NaN;
      assert.throws(
        () => wasm.compile(nonFiniteAffine.graph),
        /finite F32 \[C\] affine tensors/,
      );

      const sharedStorage = qGroupNormGraph();
      const shared = new ArrayBuffer(16);
      const aliasedInput = new Int8Array(shared);
      aliasedInput.set([-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2]);
      sharedStorage.graph.tensors.get('input').buffer = aliasedInput;
      sharedStorage.out.buffer = new Int8Array(shared);
      assert.throws(
        () => wasm.compile(sharedStorage.graph),
        /output storage distinct from every input/,
      );

      const rawAffineStorage = qGroupNormGraph();
      rawAffineStorage.graph.tensors.get('weight').buffer = new ArrayBuffer(16);
      assert.throws(
        () => wasm.compile(rawAffineStorage.graph),
        /canonical typed input\/output storage/,
      );
    });

    await t.test('affine validation survives WASM memory growth during graph allocation', () => {
      const { graph } = qGroupNormGraph();
      // Allocation happens after weight/bias in this graph. It detaches their
      // previous TypedArray views, so descriptor validation must read the
      // current WASM heap by pointer rather than a stale host view.
      graph.addInput('unused_growth', [4 * 1024 * 1024], 'int8');
      assert.doesNotThrow(() => wasm.compile(graph));
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
