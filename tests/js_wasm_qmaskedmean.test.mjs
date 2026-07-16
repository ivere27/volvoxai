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

function qMaskedMeanGraph({
  inputShape = [2, 3, 2],
  inputDtype = 'int8',
  inputValues = [1, -3, 100, 100, 5, 1, -2, 3, 4, 5, 6, 7],
  inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
  maskValues = [1, 0, 1, 0, 0, 0],
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 2 },
  params = {},
} = {}) {
  const [batch, sequence, width] = inputShape;
  const graph = new Graph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: bytes(inputDtype, inputValues), quantization: inputQuantization,
  });
  const mask = graph.addInput('mask', [batch, sequence], 'int32', {
    buffer: Int32Array.from(maskValues),
  });
  const { out } = graph.addOp('QMaskedMean', { input, mask }, {
    out: {
      name: 'out', shape: [batch, width], dtype: outputDtype,
      quantization: outputQuantization,
    },
  }, params);
  graph.outputNames = [out.name];
  return {
    graph, input, mask, out, inputValues: bytes(inputDtype, inputValues),
    maskValues: Int32Array.from(maskValues),
  };
}

async function cpuResult(spec) {
  const values = qMaskedMeanGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(values.graph);
  return cpu.execute({ input: values.inputValues, mask: values.maskValues });
}

test('portable WASM QMaskedMean preserves canonical byte-domain router storage', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qmaskedmean-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qmaskedmean_i8u8, 'function',
      'forward WASM exports qmaskedmean_i8u8');
    wasm._cpuQMaskedMean = () => {
      throw new Error('WASM QMaskedMean must not use the JavaScript reference');
    };
    wasm._cpuReduceMean = () => {
      throw new Error('WASM QMaskedMean must not materialize an F32 reduction');
    };

    await t.test('signed rows average only nonzero-mask tokens and emit zero point for empty rows', async () => {
      const expected = await cpuResult({});
      const values = qMaskedMeanGraph();
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.inputValues, mask: values.maskValues });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [10, 2, 2, 2]);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('ties-to-even requantization follows the centered raw-domain mean', async () => {
      const spec = {
        inputShape: [1, 2, 2], inputValues: [0, 1, 1, 2], maskValues: [1, 1],
        inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
        outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      };
      const expected = await cpuResult(spec);
      const values = qMaskedMeanGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.inputValues, mask: values.maskValues });
      assert.deepEqual([...result.out], [0, 2]);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('the staged F32 product/add schedule rejects an AVX-FMA half-step drift', async () => {
      const sequence = 127;
      const spec = {
        inputShape: [1, sequence, 1], inputValues: new Array(sequence).fill(-128),
        maskValues: new Array(sequence).fill(1),
        inputQuantization: { scheme: 'per_tensor', scale: 0.001, zero_point: 127 },
        outputQuantization: { scheme: 'per_tensor', scale: 0.01, zero_point: -37 },
      };
      const expected = await cpuResult(spec);
      const values = qMaskedMeanGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({ input: values.inputValues, mask: values.maskValues });
      assert.deepEqual([...result.out], [-62]);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('independent I8/U8 input/output domains stay physical bytes', async () => {
      const inputs = [
        {
          inputDtype: 'int8', inputValues: [-5, -1, 1, 3, 5, 7],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
        },
        {
          inputDtype: 'uint8', inputValues: [132, 124, 128, 136, 126, 130],
          inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        },
      ];
      const outputs = [
        { outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -3 } },
        { outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 10 } },
      ];
      for (const input of inputs) {
        for (const output of outputs) {
          const spec = {
            inputShape: [1, 3, 2], maskValues: [1, 1, 0], ...input, ...output,
          };
          const expected = await cpuResult(spec);
          const values = qMaskedMeanGraph(spec);
          wasm.compile(values.graph);
          const result = await wasm.execute({ input: values.inputValues, mask: values.maskValues });
          assert.ok(spec.outputDtype === 'int8'
            ? result.out instanceof Int8Array
            : result.out instanceof Uint8Array);
          assert.deepEqual([...result.out], [...expected.out]);
        }
      }
    });

    await t.test('a graph-input I32 mask is required and validated for every execution', async () => {
      const values = qMaskedMeanGraph();
      const expected = await cpuResult({});
      wasm.compile(values.graph);
      await assert.rejects(
        () => wasm.execute({ input: values.inputValues }),
        /graph-input I32 mask supplied on every execution/,
      );
      await assert.rejects(
        () => wasm.execute({ input: values.inputValues, mask: new Int8Array(24) }),
        /typed storage does not match dtype/,
      );
      const result = await wasm.execute({ input: values.inputValues, mask: values.maskValues });
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('a rejected direct C descriptor leaves destination bytes untouched', () => {
      const inputPointer = wasm.api.alloc_bytes(2);
      const maskPointer = wasm.api.alloc_bytes(Int32Array.BYTES_PER_ELEMENT);
      const outputPointer = wasm.api.alloc_bytes(2);
      new Int8Array(wasm.mem.buffer, inputPointer, 2).set(Int8Array.of(1, -1));
      new Int32Array(wasm.mem.buffer, maskPointer, 1).set(Int32Array.of(1));
      const output = new Uint8Array(wasm.mem.buffer, outputPointer, 2);
      output.fill(73);
      assert.equal(wasm.api.qmaskedmean_i8u8(
        inputPointer, maskPointer, outputPointer, 1, 0, 2,
        0.25, 0, 0.5, 0, 2, 2,
      ), 0);
      assert.deepEqual([...output], [73, 73]);
    });

    await t.test('source aliases, raw mask storage, and parameters fail before allocation hides them', () => {
      const inputAlias = qMaskedMeanGraph({
        inputShape: [1, 2, 2], inputValues: [1, 2, 3, 4], maskValues: [1, 1],
      });
      const sharedInput = new ArrayBuffer(6);
      inputAlias.input.buffer = new Int8Array(sharedInput, 0, 4);
      inputAlias.out.buffer = new Int8Array(sharedInput, 2, 2);
      assert.throws(() => wasm.compile(inputAlias.graph), /output storage distinct from input and mask/);

      const maskAlias = qMaskedMeanGraph({
        inputShape: [1, 2, 2], inputValues: [1, 2, 3, 4], maskValues: [1, 1],
      });
      const sharedMask = new ArrayBuffer(8);
      maskAlias.mask.buffer = new Int32Array(sharedMask, 0, 2);
      maskAlias.out.buffer = new Int8Array(sharedMask, 0, 2);
      assert.throws(() => wasm.compile(maskAlias.graph), /output storage distinct from input and mask/);

      const rawMask = qMaskedMeanGraph();
      rawMask.mask.buffer = new ArrayBuffer(rawMask.mask.sizeBytes);
      assert.throws(() => wasm.compile(rawMask.graph), /canonical I8\/U8 input\/output and I32 mask storage/);

      const parameterized = qMaskedMeanGraph({ params: { axis: 1 } });
      assert.throws(() => wasm.compile(parameterized.graph), /no parameters/);
    });

    await t.test('typed descriptor validation survives WASM memory growth during graph allocation', () => {
      const { graph } = qMaskedMeanGraph();
      // This tensor is allocated after input/mask/out. Growing the WASM heap
      // detaches those earlier TypedArray views before QMaskedMean metadata is
      // validated, so compile must refresh them from their heap pointers.
      graph.addInput('unused_growth', [4 * 1024 * 1024], 'int8');
      assert.doesNotThrow(() => wasm.compile(graph));
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
