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

async function buildForwardWasm(output) {
  await run(clang, [
    '--target=wasm32', '-O3', '-msimd128', '-nostdlib',
    '-Wl,--no-entry', '-Wl,--export-all', '-Wl,--allow-undefined',
    '-o', output, 'native/src/kernels/kernels.c',
  ], { cwd: repositoryRoot });
}

function graphFixture() {
  const graph = new RuntimeGraph();
  const left = graph.addInput('left', [2, 1, 2, 3]);
  const right = graph.addInput('right', [1, 2, 3, 2]);
  const product = graph.addOp('BatchMatMul', { a: left, b: right }, {
    out: { name: 'product', shape: [2, 2, 2, 2] },
  }).out;
  const sigmoidInput = graph.addInput('sigmoidInput', [9]);
  const sigmoid = graph.addOp('Sigmoid', { input: sigmoidInput }, {
    out: { name: 'sigmoid', shape: [9] },
  }).out;

  const logicalA = graph.addInput('logicalA', [2, 1, 3], 'int32');
  const logicalB = graph.addInput('logicalB', [1, 2, 1], 'int32');
  const equal = graph.addOp('Equal', { a: logicalA, b: logicalB }, {
    out: { name: 'equal', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  const greater = graph.addOp('GreaterOrEqual', { a: logicalA, b: logicalB }, {
    out: { name: 'greater', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  const inverted = graph.addOp('Not', { input: equal }, {
    out: { name: 'inverted', shape: [2, 2, 3], dtype: 'int32' },
  }).out;
  const clipped = graph.addOp('Clip', { input: logicalA }, {
    out: { name: 'clipped', shape: [2, 1, 3], dtype: 'int32' },
  }, { min: 2, max: 5 }).out;
  const alternative = graph.addWeight('alternative', [2, 1, 3], 'int32', {
    buffer: Int32Array.of(-1, -2, -3, -4, -5, -6),
  });
  const condition = graph.addInput('condition', [2, 1, 3], 'int32');
  const selected = graph.addOp('Where', {
    condition, a: clipped, b: alternative,
  }, {
    out: { name: 'selected', shape: [2, 1, 3], dtype: 'int32' },
  }).out;

  const joined = graph.addOp('Concat', { input0: clipped, input1: selected }, {
    out: { name: 'joined', shape: [2, 2, 3], dtype: 'int32' },
  }, { axis: 1 }).out;
  const transposed = graph.addOp('Transpose', { input: joined }, {
    out: { name: 'transposed', shape: [2, 3, 2], dtype: 'int32' },
  }, { perm: [0, 2, 1] }).out;
  const sliced = graph.addOp('Slice', { input: transposed }, {
    out: { name: 'sliced', shape: [2, 2, 2], dtype: 'int32' },
  }, { axes: [1], starts: [1], ends: [3], steps: [1] }).out;
  const reshaped = graph.addOp('Reshape', { input: sliced }, {
    out: { name: 'reshaped', shape: [2, 1, 2, 2], dtype: 'int32' },
  }, { shape: [2, 1, 2, 2] }).out;
  const expanded = graph.addOp('Expand', { input: reshaped }, {
    out: { name: 'expanded', shape: [2, 2, 2, 2], dtype: 'int32' },
  }, { shape: [2, 2, 2, 2] }).out;
  const split = graph.addOp('Split', { input: expanded }, {
    part0: { name: 'part0', shape: [2, 1, 2, 2], dtype: 'int32' },
    part1: { name: 'part1', shape: [2, 1, 2, 2], dtype: 'int32' },
  }, { axis: 1, num_outputs: 2 });
  graph.setOutputs([
    product.name, sigmoid.name, equal.name, greater.name, inverted.name, selected.name,
    split.part0.name, split.part1.name,
  ]);
  return graph;
}

const inputs = {
  left: Float32Array.from({ length: 12 }, (_, index) => index - 4),
  right: Float32Array.from({ length: 12 }, (_, index) => (index % 5) - 2),
  sigmoidInput: Float32Array.of(-12, -4, -1, -0.25, 0, 0.25, 1, 4, 12),
  logicalA: Int32Array.of(1, 2, 3, 4, 5, 6),
  logicalB: Int32Array.of(2, 5),
  condition: Int32Array.of(1, 0, -2147483648, 0, 1, 0),
};

test('forward WASM directly executes split-ONNX canonical F32/I32 kernels', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-onnx-ops-'));
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await buildForwardWasm(wasmPath);
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm);
    for (const name of [
      'matmul_f32', 'compare_broadcast_i32', 'not_i32', 'clip_i32',
      'where_typed_32', 'where_broadcast_32', 'concat_slice_u32', 'transpose_nd_u32',
      'slice_nd_u32', 'expand_nd_u32', 'split_slice_u32',
    ]) {
      assert.equal(typeof wasm.api[name], 'function', `${name} is a direct forward export`);
    }

    const cpuGraph = graphFixture();
    const cpu = new CPUEngine();
    cpu.allocateGraph(cpuGraph);
    const expected = await cpu.execute(inputs);
    const graph = graphFixture();
    wasm.compile(graph);
    const actual = await wasm.execute(inputs);
    assert.deepEqual(Object.keys(actual), Object.keys(expected));
    for (const name of Object.keys(expected)) {
      assert.equal(actual[name].constructor, expected[name].constructor, `${name} dtype`);
      if (name === 'sigmoid') {
        for (let index = 0; index < expected[name].length; index++) {
          assert.ok(
            Math.abs(actual[name][index] - expected[name][index]) <= 2e-6,
            `${name}[${index}] ${actual[name][index]} != ${expected[name][index]}`,
          );
        }
      } else {
        assert.deepEqual([...actual[name]], [...expected[name]], `${name} values`);
      }
    }

    const unknownParamGraph = graphFixture();
    const unknownParamExpand = unknownParamGraph.nodes.find((node) => node.opType === 'Expand');
    unknownParamExpand.params = { shape: [2, 2, 2, 2], unexpected: true };
    const unknownParamWasm = await WasmEngine.init(wasmPath);
    assert.throws(
      () => unknownParamWasm.compile(unknownParamGraph),
      /WASM Expand.*requires distinct rank-1\.\.8/,
    );

    const mismatchedTargetGraph = graphFixture();
    const mismatchedTargetExpand = mismatchedTargetGraph.nodes.find((node) => node.opType === 'Expand');
    mismatchedTargetExpand.params = { shape: [2, 1, 2, 2] };
    const mismatchedTargetWasm = await WasmEngine.init(wasmPath);
    assert.throws(
      () => mismatchedTargetWasm.compile(mismatchedTargetGraph),
      /WASM Expand.*requires distinct rank-1\.\.8/,
    );

    const symbolicTargetGraph = graphFixture();
    const symbolicTargetExpand = symbolicTargetGraph.nodes.find((node) => node.opType === 'Expand');
    symbolicTargetExpand.params = { shape: ['B', 'B', 2, 2] };
    const symbolicTargetWasm = await WasmEngine.init(wasmPath);
    assert.doesNotThrow(() => symbolicTargetWasm.compile(symbolicTargetGraph));

    const conflictingSymbolGraph = new RuntimeGraph();
    const conflictingSymbolInput = conflictingSymbolGraph.addInput(
      'input', [2, 1, 2, 2], 'int32',
    );
    const conflictingSymbolOutput = conflictingSymbolGraph.addOp(
      'Expand',
      { input: conflictingSymbolInput },
      { out: { name: 'out', shape: [2, 3, 2, 2], dtype: 'int32' } },
      { shape: ['B', 'B', 2, 2] },
    ).out;
    conflictingSymbolGraph.setOutputs(conflictingSymbolOutput);
    const conflictingSymbolWasm = await WasmEngine.init(wasmPath);
    assert.throws(
      () => conflictingSymbolWasm.compile(conflictingSymbolGraph),
      /WASM Expand.*requires distinct rank-1\.\.8/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
