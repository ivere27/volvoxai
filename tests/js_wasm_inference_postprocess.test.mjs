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

async function cpuResult(graph, inputs) {
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  return engine.execute(inputs);
}

function argMaxGraph({ axis, keepdims }) {
  const graph = new RuntimeGraph();
  const inputShape = [2, 3, 2];
  const normalizedAxis = axis < 0 ? axis + inputShape.length : axis;
  const outputShape = keepdims
    ? inputShape.map((dimension, index) => index === normalizedAxis ? 1 : dimension)
    : inputShape.filter((_, index) => index !== normalizedAxis);
  const input = graph.addInput('input', inputShape, 'float32');
  const { out } = graph.addOp('ArgMax', { input }, {
    out: { name: 'out', shape: outputShape, dtype: 'int32' },
  }, { axis, keepdims, select_last_index: 0 });
  graph.setOutputs([out.name]);
  return graph;
}

function nmsGraph(outputDtype) {
  const graph = new RuntimeGraph();
  const boxes = graph.addInput('boxes', [1, 3, 4]);
  const scores = graph.addInput('scores', [1, 1, 3]);
  const maximum = graph.addInput('maximum', [1], 'int32');
  const iou = graph.addInput('iou', [1]);
  const threshold = graph.addInput('threshold', [1]);
  const { out } = graph.addOp('NonMaxSuppression', {
    boxes,
    scores,
    max_output_boxes_per_class: maximum,
    iou_threshold: iou,
    score_threshold: threshold,
  }, {
    out: { name: 'out', shape: [3, 3], dtype: outputDtype },
  });
  graph.setOutputs([out.name]);
  return graph;
}

test('portable WASM ArgMax and NMS execute C kernels with CPU-equivalent typed outputs', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      t.skip(`${clang} is unavailable`);
      return;
    }
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-postprocess-'));
  try {
    const wasmPath = await buildForwardWasm(directory);
    const wasm = await WasmEngine.init(wasmPath);
    assert.ok(wasm, 'compiled forward WASM module initializes');

    await t.test('ArgMax supports a non-final axis, first-index ties, and typed output', async () => {
      const values = Float32Array.from([
        1, 9, 5, 3, 5, 4,
        2, 0, 8, 7, 6, 10,
      ]);
      const cpu = await cpuResult(argMaxGraph({ axis: 1, keepdims: false }), { input: values });
      const graph = argMaxGraph({ axis: 1, keepdims: false });
      wasm.compile(graph);
      const result = await wasm.execute({ input: values });
      assert.ok(result.out instanceof Int32Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [1, 0, 1, 2]);
    });

    await t.test('ArgMax preserves an explicit keepdims axis with canonical F32 to I32 storage', async () => {
      const values = Float32Array.from([
        1, 9, 5, 3, 5, 4,
        2, 0, 8, 7, 6, 10,
      ]);
      const cpu = await cpuResult(argMaxGraph({ axis: -1, keepdims: true }), { input: values });
      const graph = argMaxGraph({ axis: -1, keepdims: true });
      wasm.compile(graph);
      const result = await wasm.execute({ input: values });
      assert.ok(result.out instanceof Int32Array);
      assert.deepEqual(graph.tensors.get('out').shape, [2, 3, 1]);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [1, 0, 0, 0, 0, 1]);
    });

    await t.test('NMS performs stable greedy selection and fills unused output rows', async () => {
      const inputs = {
        boxes: Float32Array.from([
          0, 0, 1, 1,
          0, 0.1, 1, 1.1,
          2, 2, 3, 3,
        ]),
        scores: Float32Array.from([0.9, 0.9, 0.7]),
        maximum: Int32Array.of(3),
        iou: Float32Array.of(0.5),
        threshold: Float32Array.of(0),
      };
      const cpu = await cpuResult(nmsGraph('int32'), inputs);
      const graph = nmsGraph('int32');
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Int32Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 0, 0, 0, 0, 2, -1, -1, -1]);
    });

    await t.test('NMS preserves Uint8Array fill wrapping', async () => {
      const inputs = {
        boxes: Float32Array.from([
          0, 0, 1, 1,
          0, 0.1, 1, 1.1,
          2, 2, 3, 3,
        ]),
        scores: Float32Array.from([0.9, 0.9, 0.7]),
        maximum: Int32Array.of(3),
        iou: Float32Array.of(0.5),
        threshold: Float32Array.of(0),
      };
      const cpu = await cpuResult(nmsGraph('uint8'), inputs);
      const graph = nmsGraph('uint8');
      wasm.compile(graph);
      const result = await wasm.execute(inputs);
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [...cpu.out]);
      assert.deepEqual([...result.out], [0, 0, 0, 0, 0, 2, 255, 255, 255]);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
