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

async function cpuResult(graph, inputs) {
  const cpu = new CPUEngine();
  cpu.allocateGraph(graph);
  return cpu.execute(inputs);
}

function values(length) {
  return Float32Array.from({ length }, (_, index) => index - 7);
}

function whereGraph({ opType, conditionName, conditionDtype }) {
  const graph = new RuntimeGraph();
  const condition = graph.addInput(conditionName, [2, 2], conditionDtype);
  const a = graph.addWeight('a', [2, 2], 'float32', {
    buffer: Float32Array.of(1, 2, 3, 4),
  });
  const b = graph.addWeight('b', [2, 2], 'float32', {
    buffer: Float32Array.of(-1, -2, -3, -4),
  });
  const { out } = graph.addOp(opType, {
    [conditionName]: condition,
    a,
    b,
  }, { out: { name: 'out', shape: [2, 2] } });
  graph.setOutputs([out.name]);
  return graph;
}

function rankFiveSliceGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 3, 2, 2], 'float32');
  const { out } = graph.addOp('Slice', { input }, { out: { name: 'out', shape: [1, 2, 1, 2, 2] } }, {
    axes: [-3], starts: [1], ends: [3], steps: [2],
  });
  graph.setOutputs([out.name]);
  return graph;
}

function contiguousInt32SliceGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [3, 2, 4], 'int32');
  const { out } = graph.addOp('Slice', { input }, {
    out: { name: 'out', shape: [1, 2, 4], dtype: 'int32' },
  }, {
    axes: [0], starts: [1], ends: [2], steps: [1],
  });
  graph.setOutputs([out.name]);
  return graph;
}

function gatherGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [2, 3, 4], 'float32');
  const indices = graph.addInput('indices', [2, 2], 'int32');
  const { out } = graph.addOp('Gather', { input, indices }, { out: { name: 'out', shape: [2, 2, 2, 4] } }, { axis: -2 });
  graph.setOutputs([out.name]);
  return graph;
}

function gatherElementsGraph(indicesDtype = 'int32') {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [2, 3, 4], 'float32');
  const indices = graph.addInput('indices', [2, 2, 4], indicesDtype);
  const { out } = graph.addOp('GatherElements', { input, indices }, { out: { name: 'out', shape: [2, 2, 4] } }, { axis: -2 });
  graph.setOutputs([out.name]);
  return graph;
}

test('forward WASM dispatches canonical Where/Mask, Slice, Gather, and GatherElements kernels', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-portable-shape-'));
  try {
    const wasmPath = join(directory, 'volvoxai.wasm');
    await buildForwardWasm(wasmPath);
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm);
    for (const name of ['where_broadcast_32', 'slice_nd_f32', 'slice_nd_u32', 'gather_i32_f32', 'gather_elements_i32_f32']) {
      assert.equal(typeof wasm.api[name], 'function', `${name} is a direct forward WASM export`);
    }

    await t.test('Where and Mask retain F32/I32 condition storage, including Int32.MIN_VALUE mask', async () => {
      const cases = [
        {
          makeGraph: () => whereGraph({ opType: 'Where', conditionName: 'condition', conditionDtype: 'float32' }),
          inputs: { condition: Float32Array.of(0, -0.5, 2, 0) },
          expected: [-1, 2, 3, -4],
        },
        {
          makeGraph: () => whereGraph({ opType: 'Mask', conditionName: 'mask', conditionDtype: 'int32' }),
          inputs: { mask: Int32Array.of(0, -2_147_483_648, 1, 0) },
          expected: [-1, 2, 3, -4],
        },
      ];
      for (const entry of cases) {
        const cpu = await cpuResult(entry.makeGraph(), entry.inputs);
        const graph = entry.makeGraph();
        wasm.compile(graph);
        const result = await wasm.execute(entry.inputs);
        assert.deepEqual([...result.out], [...cpu.out]);
        assert.deepEqual([...result.out], entry.expected);
      }
    });

    await t.test('Slice handles rank five and a normalized negative axis', async () => {
      const inputs = { input: values(24) };
      const cpu = await cpuResult(rankFiveSliceGraph(), inputs);
      const graph = rankFiveSliceGraph();
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.equal(result.out.length, 8);
    });

    await t.test('Slice copies a complete trailing I32 block without changing bits', async () => {
      const inputs = { input: Int32Array.from({ length: 24 }, (_, index) =>
        index % 3 === 0 ? -2_147_483_648 + index : 1_000_000_000 - index) };
      const cpu = await cpuResult(contiguousInt32SliceGraph(), inputs);
      const graph = contiguousInt32SliceGraph();
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [...inputs.input.slice(8, 16)]);
    });

    await t.test('Gather normalizes ONNX negative indices and rejects invalid selections atomically', async () => {
      const inputs = {
        input: values(24),
        indices: Int32Array.of(-1, -3, 0, 2),
      };
      const cpu = await cpuResult(gatherGraph(), inputs);
      const graph = gatherGraph();
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.equal(result.out.length, 32);
      assert.deepEqual([...result.out], [
        1, 2, 3, 4, -7, -6, -5, -4,
        -7, -6, -5, -4, 1, 2, 3, 4,
        13, 14, 15, 16, 5, 6, 7, 8,
        5, 6, 7, 8, 13, 14, 15, 16,
      ]);
      await assert.rejects(
        () => wasm.execute({ ...inputs, indices: Int32Array.of(-1, -3, -4, 2) }),
        /Gather.*rejected its canonical descriptor/,
      );
      const recovered = await wasm.execute(inputs);
      assert.deepEqual([...recovered.out], [...result.out]);
    });

    await t.test('GatherElements handles non-axis dimensions and negative I32 selections', async () => {
      const inputs = {
        input: values(24),
        indices: Int32Array.of(
          -3, -1, 0, 1,
          2, -2, 0, -1,
          -3, -1, 0, 1,
          2, -2, 0, -1,
        ),
      };
      const cpu = await cpuResult(gatherElementsGraph(), inputs);
      const graph = gatherElementsGraph();
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.equal(result.out[0], -7);
      assert.equal(result.out[1], 2);
    });

    await t.test('Gather rejects F32 indices instead of falling back', async () => {
      assert.throws(() => wasm.compile(gatherElementsGraph('float32')), /I32 index/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
