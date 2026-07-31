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
import {
  _cpuQBatchMatMul,
  qBatchMatMulDescriptor,
} from '../ts/ops/qBatchMatMul.js';

const run = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const clang = process.env.CLANG || 'clang';
// Canonical protobuf DataType values used by the public native/WASM ABI.
const VX_DTYPE_U8 = 5;
const VX_DTYPE_I8 = 6;
const VX_DTYPE_F32 = 18;

function byteStorage(dtype, valuesOrLength) {
  const Storage = dtype === 'int8' ? Int8Array : Uint8Array;
  return typeof valuesOrLength === 'number'
    ? new Storage(valuesOrLength)
    : Storage.from(valuesOrLength);
}

function qbatchGraph({
  aDtype = 'uint8',
  aValues,
  aShape,
  aQuantization,
  bDtype = 'int8',
  bValues,
  bShape,
  bQuantization,
  outputDtype = 'uint8',
  outputShape,
  outputQuantization,
} = {}) {
  const graph = new Graph();
  const a = graph.addInput('a', aShape, aDtype, { quantization: aQuantization });
  const b = graph.addInput('b', bShape, bDtype, { quantization: bQuantization });
  const { out } = graph.addOp('QBatchMatMul', { a, b }, {
    out: {
      name: 'out',
      shape: outputShape,
      dtype: outputDtype,
      quantization: outputQuantization,
    },
  });
  graph.setOutputs([out.name]);
  const inputs = {
    a: byteStorage(aDtype, aValues),
    b: byteStorage(bDtype, bValues),
  };
  return { graph, node: graph.nodes[0], inputs };
}

const broadcastSpec = Object.freeze({
  aValues: [
    11, 10, 10,
    10, 11, 10,
    10, 10, 11,
    11, 11, 11,
  ],
  aShape: [2, 1, 2, 3],
  aQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 10 },
  bValues: [
    0, 1,
    2, 3,
    4, 5,
    -2, -3,
    -4, -5,
    -6, -7,
  ],
  bShape: [1, 2, 3, 2],
  bQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  outputShape: [2, 2, 2, 2],
  outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 100 },
});

const broadcastExpected = [
  101, 102, 103, 104,
  99, 98, 97, 96,
  105, 106, 109, 112,
  95, 94, 91, 88,
];

const simdBroadcastSpec = Object.freeze({
  aValues: Array.from({ length: 2 * 1 * 2 * 7 }, (_, index) =>
    (index * 29 + 17) & 0xff),
  aShape: [2, 1, 2, 7],
  aQuantization: { scheme: 'per_tensor', scale: 0.0137, zero_point: 139 },
  bValues: Array.from({ length: 1 * 2 * 7 * 5 }, (_, index) =>
    ((index * 37 + 11) % 255) - 127),
  bShape: [1, 2, 7, 5],
  bQuantization: { scheme: 'per_tensor', scale: 0.0091, zero_point: -13 },
  outputDtype: 'int8',
  outputShape: [2, 2, 2, 5],
  outputQuantization: { scheme: 'per_tensor', scale: 0.077, zero_point: -7 },
});

async function executeCpu(spec) {
  const { graph, inputs } = qbatchGraph(spec);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

test('CPU QBatchMatMul preserves U8S8 bytes across right-aligned ONNX broadcasting', async () => {
  const result = await executeCpu(broadcastSpec);
  assert.ok(result.out instanceof Uint8Array);
  assert.deepEqual([...result.out], broadcastExpected);
});

test('QBatchMatMul requantization uses ties-to-even in the physical byte domain', async () => {
  const result = await executeCpu({
    aValues: [129, 129],
    aShape: [1, 2],
    aQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 128 },
    bValues: [-1, 1, 1, 0, 0, 2],
    bShape: [2, 3],
    bQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
    outputDtype: 'int8',
    outputShape: [1, 3],
    outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [0, 0, 2]);
});

test('QBatchMatMul rejects noncanonical descriptors before executing', () => {
  const incompatible = qbatchGraph({
    aValues: new Array(4).fill(0),
    aShape: [2, 2, 1, 1],
    aQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    bValues: new Array(3).fill(0),
    bShape: [3, 1, 1],
    bQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputShape: [2, 2, 1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  incompatible.node.inputs.a.buffer = incompatible.inputs.a;
  incompatible.node.inputs.b.buffer = incompatible.inputs.b;
  incompatible.node.outputs.out.buffer = new Uint8Array(4);
  assert.throws(
    () => qBatchMatMulDescriptor(incompatible.node),
    /incompatible batch dimensions/,
  );

  const perAxis = qbatchGraph({
    aValues: [1, 2],
    aShape: [1, 2],
    aQuantization: {
      scheme: 'per_axis', axis: 0, scales: [1], zero_points: [0],
    },
    bValues: [1, 2],
    bShape: [2, 1],
    bQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  perAxis.node.inputs.a.buffer = perAxis.inputs.a;
  perAxis.node.inputs.b.buffer = perAxis.inputs.b;
  perAxis.node.outputs.out.buffer = new Uint8Array(1);
  assert.throws(
    () => _cpuQBatchMatMul(perAxis.node),
    /a requires per_tensor quantization/,
  );

  const contraction = 100_000;
  const overflow = qbatchGraph({
    aValues: contraction,
    aShape: [1, contraction],
    aQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    bValues: contraction,
    bShape: [contraction, 1],
    bQuantization: { scheme: 'per_tensor', scale: 1, zero_point: -128 },
    outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  overflow.node.inputs.a.buffer = overflow.inputs.a;
  overflow.node.inputs.b.buffer = overflow.inputs.b;
  overflow.node.outputs.out.buffer = new Uint8Array(1);
  assert.throws(
    () => qBatchMatMulDescriptor(overflow.node),
    /may overflow its defined I32 accumulator/,
  );

  const aliased = qbatchGraph({
    aValues: [1],
    aShape: [1, 1],
    aQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    bValues: [1],
    bShape: [1, 1],
    bQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  const shared = new ArrayBuffer(2);
  aliased.node.inputs.a.buffer = new Uint8Array(shared, 0, 1);
  aliased.node.inputs.b.buffer = new Int8Array(1);
  aliased.node.outputs.out.buffer = new Uint8Array(shared, 0, 1);
  assert.throws(
    () => qBatchMatMulDescriptor(aliased.node),
    /output must not alias an input/,
  );

  const minimumF32 = 1.401298464324817e-45;
  const underflow = qbatchGraph({
    aValues: [1],
    aShape: [1, 1],
    aQuantization: {
      scheme: 'per_tensor', scale: minimumF32, zero_point: 0,
    },
    bValues: [1],
    bShape: [1, 1],
    bQuantization: {
      scheme: 'per_tensor', scale: minimumF32, zero_point: 0,
    },
    outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  underflow.node.inputs.a.buffer = underflow.inputs.a;
  underflow.node.inputs.b.buffer = underflow.inputs.b;
  underflow.node.outputs.out.buffer = new Uint8Array(1);
  assert.throws(
    () => qBatchMatMulDescriptor(underflow.node),
    /requantization multiplier is not representable as positive F32/,
  );
});

async function buildForwardWasm(directory) {
  const output = join(directory, 'volvoxai.wasm');
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-DVOLVOXAI_QBATCH_SIMD_TESTING=1',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
  return output;
}

function allocateBytes(wasm, bytes) {
  const pointer = Number(wasm.api.alloc_bytes(bytes));
  assert.ok(Number.isInteger(pointer) && pointer > 0);
  const required = pointer + bytes;
  if (required > wasm.mem.buffer.byteLength) {
    wasm.mem.grow(Math.ceil((required - wasm.mem.buffer.byteLength) / 65536));
  }
  return pointer;
}

function writeBytes(wasm, values) {
  const pointer = allocateBytes(wasm, values.byteLength);
  new Uint8Array(wasm.mem.buffer, pointer, values.byteLength).set(
    new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
  );
  return pointer;
}

function rawQBatchArguments(aPointer, bPointer, outputPointer, spec) {
  return [
    aPointer,
    bPointer,
    outputPointer,
    spec.m,
    spec.k,
    spec.n,
    spec.aScale,
    spec.aZeroPoint,
    spec.bScale,
    spec.bZeroPoint,
    spec.outputScale,
    spec.outputZeroPoint,
    spec.aDtype,
    spec.bDtype,
    spec.outputDtype,
  ];
}

test('WASM SIMD128 QBatchMatMul is exact, fail-closed, and selected end-to-end', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qbatch-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qbatch_matmul_i8u8, 'function');
    assert.equal(typeof wasm.api.qbatch_matmul_i8u8_simd128, 'function');
    assert.equal(typeof wasm.api.qbatch_wasm_simd_calls, 'function');
    assert.equal(typeof wasm.api.reset_qbatch_wasm_simd_calls, 'function');
    wasm._cpuQBatchMatMul = () => {
      throw new Error('WASM QBatchMatMul must not call the JS reference');
    };
    wasm._cpuMatMul = () => {
      throw new Error('WASM QBatchMatMul must not call the F32 MatMul path');
    };

    // Directly compare the separately exported scalar oracle and SIMD kernel
    // on mixed byte domains with odd K and N tails.
    wasm.api.reset_heap();
    const directSpec = {
      m: 3,
      k: 7,
      n: 5,
      aScale: 0.03125,
      aZeroPoint: 137,
      bScale: 0.125,
      bZeroPoint: -13,
      outputScale: 0.0625,
      outputZeroPoint: -7,
      aDtype: VX_DTYPE_U8,
      bDtype: VX_DTYPE_I8,
      outputDtype: VX_DTYPE_I8,
    };
    const directA = Uint8Array.from(
      { length: directSpec.m * directSpec.k },
      (_, index) => (index * 41 + 19) & 0xff,
    );
    const directB = Int8Array.from(
      { length: directSpec.k * directSpec.n },
      (_, index) => ((index * 31 + 7) % 255) - 127,
    );
    const directAPointer = writeBytes(wasm, directA);
    const directBPointer = writeBytes(wasm, directB);
    const scalarOutputPointer = allocateBytes(wasm, directSpec.m * directSpec.n);
    const simdOutputPointer = allocateBytes(wasm, directSpec.m * directSpec.n);
    const scalarArguments = rawQBatchArguments(
      directAPointer, directBPointer, scalarOutputPointer, directSpec,
    );
    const simdArguments = rawQBatchArguments(
      directAPointer, directBPointer, simdOutputPointer, directSpec,
    );
    wasm.api.reset_qbatch_wasm_simd_calls();
    assert.equal(wasm.api.qbatch_matmul_i8u8(...scalarArguments), 1);
    assert.equal(wasm.api.qbatch_wasm_simd_calls(), 0);
    assert.equal(wasm.api.qbatch_matmul_i8u8_simd128(...simdArguments), 1);
    assert.equal(wasm.api.qbatch_wasm_simd_calls(), 1);
    assert.deepEqual(
      new Uint8Array(
        wasm.mem.buffer, simdOutputPointer, directSpec.m * directSpec.n,
      ).slice(),
      new Uint8Array(
        wasm.mem.buffer, scalarOutputPointer, directSpec.m * directSpec.n,
      ).slice(),
      'mixed U8S8 SIMD output must byte-match the scalar oracle',
    );

    // The first four columns exercise SIMD requantization inputs and the fifth
    // is the scalar N tail.  All are exact ties-to-even boundaries.
    const tiesSpec = {
      ...directSpec,
      m: 1,
      k: 2,
      n: 5,
      aScale: 0.5,
      aZeroPoint: 128,
      bScale: 0.5,
      bZeroPoint: 0,
      outputScale: 0.5,
      outputZeroPoint: 0,
    };
    const tiesAPointer = writeBytes(wasm, Uint8Array.of(129, 129));
    const tiesBPointer = writeBytes(
      wasm, Int8Array.of(-1, 1, 1, -3, 3, 0, 0, 2, 0, 0),
    );
    const tiesOutputPointer = allocateBytes(wasm, tiesSpec.n);
    wasm.api.reset_qbatch_wasm_simd_calls();
    assert.equal(wasm.api.qbatch_matmul_i8u8_simd128(
      ...rawQBatchArguments(
        tiesAPointer, tiesBPointer, tiesOutputPointer, tiesSpec,
      ),
    ), 1);
    assert.equal(wasm.api.qbatch_wasm_simd_calls(), 1);
    assert.deepEqual(
      [...new Int8Array(wasm.mem.buffer, tiesOutputPointer, tiesSpec.n)],
      [0, 0, 2, -2, 2],
    );

    // Malformed descriptors fail before writing even one sentinel byte.
    const invalidOutputPointer = allocateBytes(wasm, directSpec.m * directSpec.n);
    const invalidOutput = new Uint8Array(
      wasm.mem.buffer, invalidOutputPointer, directSpec.m * directSpec.n,
    );
    invalidOutput.fill(0x5a);
    assert.equal(wasm.api.qbatch_matmul_i8u8_simd128(
      ...rawQBatchArguments(
        directAPointer,
        directBPointer,
        invalidOutputPointer,
        { ...directSpec, aDtype: VX_DTYPE_F32 },
      ),
    ), 0);
    assert.ok(invalidOutput.every((value) => value === 0x5a));
    assert.equal(wasm.api.qbatch_matmul_i8u8_simd128(
      ...rawQBatchArguments(
        directAPointer,
        directBPointer,
        invalidOutputPointer,
        { ...directSpec, aZeroPoint: 256 },
      ),
    ), 0);
    assert.ok(invalidOutput.every((value) => value === 0x5a));
    assert.equal(wasm.api.qbatch_matmul_i8u8_simd128(
      ...rawQBatchArguments(
        directAPointer,
        directBPointer,
        invalidOutputPointer,
        {
          ...directSpec,
          m: 1,
          k: 33026,
          n: 1,
          aZeroPoint: 0,
          bZeroPoint: -128,
        },
      ),
    ), 0);
    assert.ok(invalidOutput.every((value) => value === 0x5a));

    // Preserve the existing hand-checked broadcast case.
    const cpu = await executeCpu(broadcastSpec);
    const { graph, inputs } = qbatchGraph(broadcastSpec);
    wasm.compile(graph);
    wasm.api.reset_qbatch_wasm_simd_calls();
    const result = await wasm.execute(inputs);
    assert.equal(
      wasm.api.qbatch_wasm_simd_calls(),
      0,
      'N<4 must retain the exact scalar kernel',
    );
    assert.ok(result.out instanceof Uint8Array);
    assert.deepEqual([...result.out], [...cpu.out]);
    assert.deepEqual([...result.out], broadcastExpected);

    // A rank-4, right-aligned broadcast with odd K/N executes once per output
    // batch.  First force the scalar ABI, then enable the standard-SIMD export
    // and require exact JS-reference parity without any backend fallback.
    const simdCpu = await executeCpu(simdBroadcastSpec);
    const simdGraph = qbatchGraph(simdBroadcastSpec);
    wasm.qbatchMatMulSimdEnabled = false;
    wasm.compile(simdGraph.graph);
    wasm.api.reset_qbatch_wasm_simd_calls();
    const scalarResult = await wasm.execute(simdGraph.inputs);
    assert.equal(wasm.api.qbatch_wasm_simd_calls(), 0);
    assert.deepEqual([...scalarResult.out], [...simdCpu.out]);

    wasm.qbatchMatMulSimdEnabled = true;
    wasm.api.reset_qbatch_wasm_simd_calls();
    const simdResult = await wasm.execute(simdGraph.inputs);
    assert.equal(
      wasm.api.qbatch_wasm_simd_calls(),
      4,
      'four broadcast matrix pairs must enter the SIMD kernel',
    );
    assert.deepEqual([...simdResult.out], [...simdCpu.out]);
    assert.deepEqual([...simdResult.out], [...scalarResult.out]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
