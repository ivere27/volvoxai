import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { Graph } from '../ts/core/Graph.js';
import { ModelSnapshot } from '../ts/core/ModelSnapshot.js';
import { BuiltInBackendProvider } from '../ts/backends/BackendProvider.js';
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

function qlinearGraph({
  opType = 'QLinear',
  inputDtype = 'int8', inputValues, inputShape, inputQuantization,
  weightDtype = 'int8', weightValues, weightShape, weightQuantization,
  biasValues,
  outputDtype = 'int8', outputShape, outputQuantization,
  dynamicWeight = false,
} = {}) {
  const graph = new Graph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    quantization: inputQuantization,
  });
  const weight = dynamicWeight
    ? graph.addInput('weight', weightShape, weightDtype, { quantization: weightQuantization })
    : graph.addWeight('weight', weightShape, weightDtype, {
      buffer: byteStorage(weightDtype, weightValues),
      quantization: weightQuantization,
    });
  const bias = graph.addWeight('bias', [weightShape[0]], 'int32', {
    buffer: Int32Array.from(biasValues),
  });
  const { out } = graph.addOp(opType, { input, weight, bias }, {
    out: {
      name: 'out', shape: outputShape, dtype: outputDtype,
      quantization: outputQuantization,
    },
  });
  graph.setOutputs([out.name]);
  return {
    graph,
    inputValues: byteStorage(inputDtype, inputValues),
    weightValues: byteStorage(weightDtype, weightValues),
  };
}

async function cpuResult(spec) {
  const { graph, inputValues } = qlinearGraph(spec);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute({ input: inputValues });
}

function forbidCpuDenseFallback(engine) {
  engine._cpuQLinear = () => {
    throw new Error('canonical W8A8 WASM dispatch must not call the CPU QLinear reference');
  };
  engine._cpuMatMul = () => {
    throw new Error('canonical W8A8 WASM dispatch must not call the F32 MatMul reference');
  };
}

test('WASM packed Q8 allocation rejects values outside the signed allocator ABI', () => {
  let allocated = false;
  const memory = new WebAssembly.Memory({ initial: 1 });
  const engine = new WasmEngine({ instance: { exports: {
    memory,
    packed_q8_weight_size: () => 0x7ffffff1,
    pack_q8_weight: () => 1,
    alloc_bytes: () => { allocated = true; return 0; },
  } } });
  engine.pointers.set('weight', 0);
  assert.throws(() => engine._allocPackedQ8Weight(
    { name: 'weight', dtype: 'int8' }, 1, 1, true,
  ), /packed Q8 weight size overflow/);
  assert.equal(allocated, false);
});

test('WASM packed F32 allocation leaves room for allocator alignment', () => {
  let allocated = false;
  const memory = new WebAssembly.Memory({ initial: 1 });
  const engine = new WasmEngine({ instance: { exports: {
    memory,
    gemm_f32_packed_elements: () => 0x1ffffffd,
    gemm_f32_pack_b: () => 1,
    gemm_f32_packed: () => 1,
    alloc_bytes: () => { allocated = true; return 0; },
  } } });
  engine.pointers.set('weight', 0);
  const input = { name: 'input', dtype: 'float32', shape: [1, 1], sizeBytes: 4 };
  const weight = { name: 'weight', dtype: 'float32', shape: [1, 1], sizeBytes: 4, isWeight: true };
  const output = { name: 'output', dtype: 'float32', shape: [1, 1], sizeBytes: 4 };
  assert.throws(() => engine._compilePackedF32Linear({
    id: 'oversized-f32', opType: 'Linear', inputs: { input, weight },
    outputs: { out: output },
  }), /unrepresentable packed F32 weight/);
  assert.equal(allocated, false);
});

test('portable WASM QLinear keeps W8A8 storage and matches the CPU reference', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qlinear-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qlinear_i8u8, 'function', 'forward WASM exports qlinear_i8u8');
    forbidCpuDenseFallback(wasm);

    await t.test('provider captures stable snapshots from transient WASM arena views', async () => {
      const spec = {
        inputValues: [3, -2, 5], inputShape: [1, 3],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        weightValues: [2, 0, -1, -2, 1, 3], weightShape: [2, 3],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2],
        },
        biasValues: [2, -4], outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
      };
      const { graph, inputValues } = qlinearGraph(spec);
      wasm.compile(graph);
      const direct = await wasm.execute({ input: inputValues });
      assert.equal(direct.out.buffer, wasm.mem.buffer,
        'internal WASM execution should publish its arena view without a transient copy');

      const provider = new BuiltInBackendProvider(await wasm.fork());
      let compiled;
      let context;
      try {
        compiled = await provider.compile(ModelSnapshot.capture(graph), {
          operatorFallback: 'forbid',
        });
        context = await compiled.createContext();
        const first = await context.execute({ input: inputValues });
        const firstData = first.outputs.find(({ name }) => name === 'out')?.data;
        assert.ok(firstData instanceof Int8Array);
        const firstValues = [...firstData];

        const replacement = Int8Array.of(-3, 4, -5);
        const second = await context.execute({ input: replacement });
        const secondData = second.outputs.find(({ name }) => name === 'out')?.data;
        assert.ok(secondData instanceof Int8Array);
        assert.notDeepEqual([...secondData], firstValues);
        assert.deepEqual([...firstData], firstValues,
          'a later arena reuse must not mutate an earlier provider snapshot');
      } finally {
        await context?.close();
        await compiled?.close();
        await provider.close();
      }
    });

    const signed = {
      inputValues: [1, -2, 3, 4, 0, -1],
      inputShape: [2, 3],
      inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
      weightValues: [2, 0, -1, -2, 1, 3],
      weightShape: [2, 3],
      weightQuantization: {
        scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2],
      },
      biasValues: [2, -4],
      outputShape: [2, 2],
      outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
    };
    await t.test('non-representable F32 multiplier fails before WASM dispatch', () => {
      const minimumF32 = 1.401298464324817e-45;
      const { graph } = qlinearGraph({
        ...signed,
        inputValues: [1],
        inputShape: [1, 1],
        inputQuantization: {
          scheme: 'per_tensor', scale: minimumF32, zero_point: 0,
        },
        weightValues: [1],
        weightShape: [1, 1],
        weightQuantization: {
          scheme: 'per_axis', axis: 0,
          scales: [minimumF32], zero_points: [0],
        },
        biasValues: [0],
        outputShape: [1, 1],
        outputQuantization: {
          scheme: 'per_tensor', scale: 1, zero_point: 0,
        },
      });
      assert.throws(
        () => wasm.compile(graph),
        /requantization multiplier/,
      );
    });
    for (const opType of ['QLinear', 'QMatMul', 'QGemm']) {
      await t.test(`${opType} uses I8 activation, I32 accumulation, and per-axis W8 requantization`, async () => {
        const cpu = await cpuResult({ ...signed, opType });
        const { graph, inputValues } = qlinearGraph({ ...signed, opType });
        wasm.compile(graph);
        const result = await wasm.execute({ input: inputValues });
        assert.ok(result.out instanceof Int8Array);
        assert.deepEqual([...result.out], [...cpu.out]);
        assert.deepEqual([...result.out], [-3, 6, 6, 0]);
      });
    }

    await t.test('asymmetric U8 inputs and outputs saturate without F32 activation storage', async () => {
      const unsigned = {
        inputDtype: 'uint8', inputValues: [255, 255], inputShape: [1, 2],
        inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 128 },
        weightDtype: 'uint8', weightValues: [255, 0, 128, 128], weightShape: [2, 2],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [128, 128],
        },
        biasValues: [-1000, 1024],
        outputDtype: 'uint8', outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
      };
      const cpu = await cpuResult(unsigned);
      const { graph, inputValues } = qlinearGraph(unsigned);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 255]);
    });

    await t.test('independent I8/U8 storage choices for activation, weight, and output stay byte-typed', async () => {
      const mixed = {
        inputDtype: 'int8', inputValues: [-3, 4, 7], inputShape: [1, 3],
        inputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -2 },
        weightDtype: 'uint8', weightValues: [255, 1, 128, 3, 200, 10], weightShape: [2, 3],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.125, 0.5], zero_points: [128, 10],
        },
        biasValues: [17, -23],
        outputDtype: 'uint8', outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.0625, zero_point: 119 },
      };
      const cpu = await cpuResult(mixed);
      const { graph, inputValues } = qlinearGraph(mixed);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('dynamic byte weights stay on the raw fallback and observe every execution input', async () => {
      const dynamic = {
        dynamicWeight: true,
        inputValues: [3, -2, 5], inputShape: [1, 3],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        weightValues: [2, 0, -1, -2, 1, 3], weightShape: [2, 3],
        weightQuantization: {
          scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2],
        },
        biasValues: [2, -4], outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
      };
      const { graph, inputValues, weightValues } = qlinearGraph(dynamic);
      wasm.compile(graph);
      const descriptor = wasm.nodeMetadata.get(graph.nodes[0]);
      assert.equal(descriptor.packedWeightPointer, 0);
      const first = await wasm.execute({ input: inputValues, weight: weightValues });
      const firstValues = [...first.out];
      const replacement = Int8Array.of(-7, 4, 6, 8, -3, 1);
      const second = await wasm.execute({ input: inputValues, weight: replacement });
      const secondValues = [...second.out];
      assert.notDeepEqual(firstValues, secondValues);
      const cpu = new CPUEngine();
      cpu.allocateGraph(graph);
      const expected = await cpu.execute({ input: inputValues, weight: replacement });
      assert.deepEqual(secondValues, [...expected.out]);
    });

    await t.test('SIMD NR=8 panels preserve odd K/N tails and asymmetric metadata', async () => {
      const dIn = 17;
      const dOut = 9;
      const rows = 2;
      const wide = {
        inputDtype: 'uint8',
        inputValues: Array.from({ length: rows * dIn }, (_, index) => (index * 29 + 7) % 251),
        inputShape: [rows, dIn],
        inputQuantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 127 },
        weightDtype: 'int8',
        weightValues: Array.from({ length: dOut * dIn }, (_, index) => (index * 31) % 127 - 63),
        weightShape: [dOut, dIn],
        weightQuantization: {
          scheme: 'per_axis', axis: 0,
          scales: Array.from({ length: dOut }, (_, index) => 0.0078125 * (1 + index % 5)),
          zero_points: Array.from({ length: dOut }, (_, index) => index % 9 - 4),
        },
        biasValues: Array.from({ length: dOut }, (_, index) => index * 19 - 71),
        outputDtype: 'uint8', outputShape: [rows, dOut],
        outputQuantization: { scheme: 'per_tensor', scale: 0.0625, zero_point: 119 },
      };
      const cpu = await cpuResult(wide);
      const { graph, inputValues } = qlinearGraph(wide);
      wasm.compile(graph);
      const result = await wasm.execute({ input: inputValues });
      assert.deepEqual([...result.out], [...cpu.out]);
    });

    await t.test('compile rejects a non-axis-0 per-channel weight descriptor', () => {
      const invalid = qlinearGraph({
        inputValues: [1, 2, 3], inputShape: [1, 3],
        inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        weightValues: [1, 2, 3, 4, 5, 6], weightShape: [2, 3],
        weightQuantization: {
          scheme: 'per_axis', axis: 1, scales: [0.25, 0.25, 0.25], zero_points: [0, 0, 0],
        },
        biasValues: [0, 0], outputShape: [1, 2],
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      });
      assert.throws(() => wasm.compile(invalid.graph), /axis-0 per-output-channel quantization metadata/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
