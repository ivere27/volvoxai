import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
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

function quantization(scale, zeroPoint) {
  return { scheme: 'per_tensor', scale, zero_point: zeroPoint };
}

function output(name, shape, dtype, scale, zeroPoint) {
  return {
    name, shape, dtype,
    quantization: quantization(scale, zeroPoint),
  };
}

function forbidCpuFallbacks(engine) {
  for (const helper of [
    '_cpuReshape', '_cpuResize', '_cpuConcat2', '_cpuMaxPool2D', '_cpuTranspose',
  ]) {
    engine[helper] = () => {
      throw new Error(`quantized WASM shape-island dispatch must not call ${helper}`);
    };
  }
}

function byteTransposeGraph(dtype) {
  const graph = new RuntimeGraph();
  const zeroPoint = dtype === 'int8' ? -3 : 123;
  const input = graph.addInput('input', [1, 2, 2, 3], dtype, {
    quantization: quantization(0.125, zeroPoint),
  });
  const { out: nhwc } = graph.addOp('Transpose', { input }, {
    out: output('nhwc', [1, 2, 3, 2], dtype, 0.125, zeroPoint),
  }, { perm: [0, 2, 3, 1] });
  const { out } = graph.addOp('Transpose', { input: nhwc }, {
    out: output('out', [1, 2, 2, 3], dtype, 0.125, zeroPoint),
  }, { perm: [0, 3, 1, 2] });
  graph.setOutputs([nhwc.name, out.name]);
  return graph;
}

function signedShapeIslandGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 2, 2], 'int8', {
    quantization: quantization(0.25, 0),
  });
  const side = graph.addInput('side', [1, 12], 'int8', {
    quantization: quantization(0.25, 0),
  });
  const max = graph.addOp('MaxPool2D', { input }, {
    out: output('max', [1, 1, 1, 2], 'int8', 0.25, 0),
  }, { kernel: [2, 2], stride: [2, 2], padding: [0, 0] });
  const resized = graph.addOp('ResizeNearest2D', { input: max.out }, {
    out: output('resized', [1, 2, 3, 2], 'int8', 0.25, 0),
  }, { mode: 'nearest' });
  const reshaped = graph.addOp('Reshape', { input: resized.out }, {
    out: output('reshaped', [1, 12], 'int8', 0.25, 0),
  });
  const flattened = graph.addOp('Flatten', { input: reshaped.out }, {
    out: output('flattened', [1, 12], 'int8', 0.25, 0),
  });
  const squeezed = graph.addOp('Squeeze', { input: flattened.out }, {
    out: output('squeezed', [12], 'int8', 0.25, 0),
  });
  const unsqueezed = graph.addOp('Unsqueeze', { input: squeezed.out }, {
    out: output('unsqueezed', [1, 12], 'int8', 0.25, 0),
  });
  const identity = graph.addOp('Identity', { input: unsqueezed.out }, {
    out: output('identity', [1, 12], 'int8', 0.25, 0),
  });
  const concatenated = graph.addOp('Concat', { input: identity.out, b: side }, {
    out: output('out', [1, 24], 'int8', 0.25, 0),
  }, { axis: 1 });
  graph.setOutputs([concatenated.out.name]);
  return graph;
}

function unsignedMaxPoolResizeGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 2, 1], 'uint8', {
    quantization: quantization(0.125, 128),
  });
  const max = graph.addOp('MaxPool2D', { input }, {
    out: output('max', [1, 1, 1, 1], 'uint8', 0.125, 128),
  }, { kernel: [2, 2], stride: [2, 2] });
  const resized = graph.addOp('Resize', { input: max.out }, {
    out: output('out', [1, 2, 2, 1], 'uint8', 0.125, 128),
  }, { mode: 'nearest' });
  graph.setOutputs([resized.out.name]);
  return graph;
}

function asymmetricPadMaxPoolGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 3, 3, 1], 'int8', {
    quantization: quantization(0.25, 0),
  });
  const { out } = graph.addOp('MaxPool2D', { input }, {
    out: output('out', [1, 2, 2, 1], 'int8', 0.25, 0),
  }, { kernel: [2, 2], stride: [2, 2], padding: [0, 0], pads: [0, 0, 1, 1] });
  graph.setOutputs([out.name]);
  return graph;
}

function mismatchedConcatGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 1], 'int8', {
    quantization: quantization(0.25, 0),
  });
  const other = graph.addInput('other', [1, 1], 'int8', {
    quantization: quantization(0.5, 0),
  });
  const { out } = graph.addOp('Concat', { input, b: other }, {
    out: output('out', [1, 2], 'int8', 0.25, 0),
  }, { axis: 1 });
  graph.setOutputs([out.name]);
  return graph;
}

test('portable WASM preserves quantized shape-island bytes without CPU fallback', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-quantized-shape-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    forbidCpuFallbacks(wasm);

    await t.test('I8 MaxPool, nearest resize, shape views, and concat preserve quantized bytes', async () => {
      const graph = signedShapeIslandGraph();
      wasm.compile(graph);
      const side = Int8Array.of(-128, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 127);
      const result = await wasm.execute({
        input: Int8Array.of(-10, 3, 5, -20, 7, -4, -8, 12),
        side,
      });
      const expectedPrefix = Array.from({ length: 6 }, () => [7, 12]).flat();
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...expectedPrefix, ...side]);
    });

    await t.test('U8 MaxPool compares unsigned bytes and Resize mode nearest stays quantized', async () => {
      const graph = unsignedMaxPoolResizeGraph();
      wasm.compile(graph);
      const result = await wasm.execute({ input: Uint8Array.of(3, 250, 4, 5) });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [250, 250, 250, 250]);
    });

    await t.test('I8/U8 NCHW↔NHWC Transpose preserves exact bytes', async () => {
      for (const dtype of ['int8', 'uint8']) {
        const graph = byteTransposeGraph(dtype);
        wasm.compile(graph);
        const Storage = dtype === 'int8' ? Int8Array : Uint8Array;
        const values = Storage.from(
          { length: 12 },
          (_, index) => dtype === 'int8' ? index - 6 : index + 117,
        );
        const result = await wasm.execute({ input: values });
        assert.ok(result.nhwc instanceof Storage);
        assert.ok(result.out instanceof Storage);
        assert.deepEqual(
          [...result.nhwc],
          [values[0], values[6], values[1], values[7], values[2], values[8],
            values[3], values[9], values[4], values[10], values[5], values[11]],
        );
        assert.deepEqual([...result.out], [...values]);
      }
    });

    await t.test('I8 MaxPool validates asymmetric bottom/right pads while reading raw bytes', async () => {
      const graph = asymmetricPadMaxPoolGraph();
      wasm.compile(graph);
      const result = await wasm.execute({ input: Int8Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9) });
      assert.deepEqual([...result.out], [5, 6, 8, 9]);
    });

    await t.test('raw quantized concat rejects non-identical immutable descriptors', () => {
      assert.throws(() => wasm.compile(mismatchedConcatGraph()), /identical per-tensor metadata/);
    });

    await t.test('shape-island dispatch rejects mutable quantization metadata', () => {
      const graph = signedShapeIslandGraph();
      graph.getTensor('max').quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
      assert.throws(() => wasm.compile(graph), /immutable quantization descriptors/);
    });

    await t.test('typed spatial descriptors reject non-canonical pool and resize semantics', () => {
      const dilated = asymmetricPadMaxPoolGraph();
      dilated.nodes[0].params.dilation = [2, 1];
      assert.throws(() => wasm.compile(dilated), /only with unit dilation/);

      const ceil = asymmetricPadMaxPoolGraph();
      ceil.nodes[0].params.ceil_mode = true;
      assert.throws(() => wasm.compile(ceil), /does not support ceil_mode/);

      const transformedResize = unsignedMaxPoolResizeGraph();
      transformedResize.nodes[1].params.coordinate_transformation_mode = 'half_pixel';
      assert.throws(() => wasm.compile(transformedResize), /coordinate_transformation_mode "asymmetric"/);

      const misspelledResize = unsignedMaxPoolResizeGraph();
      misspelledResize.nodes[1].params.coordinate_transform_mode = 'asymmetric';
      assert.throws(
        () => wasm.compile(misspelledResize),
        /does not define coordinate_transform_mode; use coordinate_transformation_mode/,
      );
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
