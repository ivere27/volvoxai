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
// Canonical protobuf DataType value used by the public native/WASM ABI.
const VX_DTYPE_I8 = 6;

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

function byteValues(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function qArgMaxGraph({
  shape = [2, 3, 2],
  dtype = 'int8',
  values = [1, 9, 5, 9, 5, 4, 2, 0, 8, 7, 6, 10],
  quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
  axis = 1,
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', shape, dtype, {
    buffer: byteValues(dtype, values), quantization,
  });
  const resolvedAxis = axis < 0 ? axis + shape.length : axis;
  const outputShape = [...shape.slice(0, resolvedAxis), ...shape.slice(resolvedAxis + 1)];
  const { out } = graph.addOp('QArgMax', { input }, {
    out: { name: 'out', shape: outputShape, dtype: 'int32' },
  }, { axis });
  graph.setOutputs([out.name]);
  return { graph, input, out, values: byteValues(dtype, values) };
}

async function cpuResult(spec) {
  const values = qArgMaxGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(values.graph);
  return cpu.execute({ input: values.values });
}

test('portable WASM QArgMax preserves canonical raw I8/U8 ordering', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qargmax-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qargmax_i8u8, 'function', 'forward WASM exports qargmax_i8u8');
    wasm._cpuQArgMax = () => { throw new Error('WASM QArgMax must not use the JavaScript reference'); };
    wasm._cpuArgMax = () => { throw new Error('WASM QArgMax must not select generic ArgMax'); };

    await t.test('non-final I8 axis matches CPU and retains first ties', async () => {
      const expected = await cpuResult({});
      const values = qArgMaxGraph();
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.values });
      assert.ok(result.out instanceof Int32Array);
      assert.deepEqual([...result.out], [1, 0, 1, 2]);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('asymmetric U8 and normalized negative axis compare bytes', async () => {
      const spec = {
        shape: [2, 3, 2], dtype: 'uint8', axis: -2,
        values: [128, 7, 255, 7, 255, 6, 1, 4, 2, 9, 2, 8],
        quantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: 201 },
      };
      const expected = await cpuResult(spec);
      const values = qArgMaxGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.values });
      assert.deepEqual([...result.out], [1, 0, 1, 1]);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('rank-eight final-axis descriptor remains entirely typed', async () => {
      const spec = {
        shape: [1, 1, 1, 1, 1, 1, 1, 4], axis: -1,
        values: [-3, 5, 5, 2], quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 17 },
      };
      const values = qArgMaxGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.values });
      assert.deepEqual([...result.out], [1]);
      assert.equal(values.out.quantization, null);
    });

    await t.test('wrong dynamic input type, direct C rejection, and source aliases do not write an index', async () => {
      const values = qArgMaxGraph();
      wasm.compile(values.graph);
      await assert.rejects(
        () => wasm.execute({ input: new Uint8Array(12) }),
        /typed storage does not match dtype/,
      );

      const inputPointer = wasm.api.alloc_bytes(12);
      const outputPointer = wasm.api.alloc_bytes(16);
      new Int8Array(wasm.mem.buffer, inputPointer, 12).set(
        Int8Array.of(1, 9, 5, 9, 5, 4, 2, 0, 8, 7, 6, 10),
      );
      const sentinel = new Int32Array(wasm.mem.buffer, outputPointer, 4);
      sentinel.set([17, 23, 31, 47]);
      assert.equal(wasm.api.qargmax_i8u8(
        inputPointer, outputPointer, 2, 0, 2, VX_DTYPE_I8,
      ), 0);
      assert.deepEqual([...sentinel], [17, 23, 31, 47]);
      assert.equal(wasm.api.qargmax_i8u8(
        inputPointer, outputPointer, 2, 3, 2, 2,
      ), 0, 'legacy compact I8 code is canonical F4 and must be rejected');
      assert.deepEqual([...sentinel], [17, 23, 31, 47]);

      const aliased = qArgMaxGraph({ shape: [2, 4], axis: 1, values: [1, 2, 3, 4, 4, 3, 2, 1] });
      const shared = new ArrayBuffer(8);
      aliased.input.buffer = new Int8Array(shared);
      aliased.out.buffer = new Int32Array(shared);
      assert.throws(() => wasm.compile(aliased.graph), /distinct from input storage/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
