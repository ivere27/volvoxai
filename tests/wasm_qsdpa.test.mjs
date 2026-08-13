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

function qSDPAGraph({
  qShape = [2, 4],
  qDtype = 'int8', qValues = [0, -1, -2, 1, -2, 1, 0, -1],
  qQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  kShape = [2, 4],
  kDtype = 'int8', kValues = [0, 0, -1, -1, -1, 0, 0, -2],
  kQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  vShape = [2, 4],
  vDtype = 'int8', vValues = [3, -3, 0, -1, -5, 1, 2, -2],
  vQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  outputDtype = 'int8', outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  maskShape, maskValues, heads = 1, causal = false, scale = 0.5,
} = {}) {
  const graph = new RuntimeGraph();
  const q = graph.addInput('q', qShape, qDtype, { buffer: bytes(qDtype, qValues), quantization: qQuantization });
  const k = graph.addInput('k', kShape, kDtype, { buffer: bytes(kDtype, kValues), quantization: kQuantization });
  const v = graph.addInput('v', vShape, vDtype, { buffer: bytes(vDtype, vValues), quantization: vQuantization });
  const inputs = { q, k, v };
  let mask = null;
  if (maskShape !== undefined) {
    mask = graph.addInput('mask', maskShape, 'int32', { buffer: Int32Array.from(maskValues) });
    inputs.mask = mask;
  }
  const params = { heads, causal };
  if (scale !== undefined) params.scale = scale;
  const { out } = graph.addOp('QSDPA', inputs, {
    out: { name: 'out', shape: qShape, dtype: outputDtype, quantization: outputQuantization },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, qValues: bytes(qDtype, qValues), kValues: bytes(kDtype, kValues),
    vValues: bytes(vDtype, vValues), maskValues: mask ? Int32Array.from(maskValues) : null, out, q, k, v, mask };
}

async function cpuResult(spec) {
  const values = qSDPAGraph(spec);
  const cpu = new CPUEngine();
  cpu.allocateGraph(values.graph);
  const inputs = { q: values.qValues, k: values.kValues, v: values.vValues };
  if (values.maskValues) inputs.mask = values.maskValues;
  return cpu.execute(inputs);
}

test('portable WASM QSDPA preserves canonical I8/U8 attention storage', {
  timeout: 120_000,
}, async (t) => {
  try {
    await run(clang, ['--version']);
  } catch (error) {
    if (error?.code === 'ENOENT') return t.skip(`${clang} is unavailable`);
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-wasm-qsdpa-'));
  try {
    const wasm = await WasmEngine.init(await buildForwardWasm(directory));
    assert.ok(wasm, 'compiled forward WASM module initializes');
    assert.equal(typeof wasm.api.qsdpa_i8u8, 'function', 'forward WASM exports qsdpa_i8u8');
    wasm._cpuQSDPA = () => { throw new Error('WASM QSDPA must not use the JavaScript reference'); };
    wasm._cpuSDPA = () => { throw new Error('WASM QSDPA must not select packed F32 SDPA'); };
    wasm._cpuCrossSDPA = () => { throw new Error('WASM QSDPA must not select F32 CrossSDPA'); };

    await t.test('canonical I8 noncausal and causal results match the portable CPU', async () => {
      for (const causal of [false, true]) {
        const spec = { causal };
        const expected = await cpuResult(spec);
        const values = qSDPAGraph(spec);
        wasm.compile(values.graph);
        const result = await wasm.execute({ q: values.qValues, k: values.kValues, v: values.vValues });
        assert.ok(result.out instanceof Int8Array);
        assert.deepEqual([...result.out], [...expected.out]);
      }
    });

    await t.test('asymmetric U8, a keep-mask, and all-masked rows stay typed', async () => {
      const base = {
        qShape: [1, 4], qDtype: 'uint8', qValues: [129, 126, 131, 128],
        qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
        kShape: [2, 4], kDtype: 'uint8', kValues: [121, 120, 119, 122, 119, 123, 120, 118],
        kQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 120 },
        vShape: [2, 4], vDtype: 'uint8', vValues: [134, 126, 132, 130, 128, 132, 136, 128],
        vQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 130 },
        outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 127 },
      };
      const masked = qSDPAGraph({ ...base, maskShape: [2], maskValues: [0, 1] });
      wasm.compile(masked.graph);
      const result = await wasm.execute({
        q: masked.qValues, k: masked.kValues, v: masked.vValues, mask: masked.maskValues,
      });
      assert.ok(result.out instanceof Uint8Array);
      assert.deepEqual([...result.out], [125, 129, 133, 125]);

      const allMasked = qSDPAGraph({ ...base, maskShape: [2], maskValues: [0, 0] });
      wasm.compile(allMasked.graph);
      const zeroed = await wasm.execute({
        q: allMasked.qValues, k: allMasked.kValues, v: allMasked.vValues, mask: allMasked.maskValues,
      });
      assert.deepEqual([...zeroed.out], [127, 127, 127, 127]);
    });

    await t.test('every independent I8/U8 q/k/v/output pair matches CPU for a one-key row', async () => {
      const byteTypes = [
        { dtype: 'int8', zeroPoint: -1 }, { dtype: 'uint8', zeroPoint: 128 },
      ];
      const centered = [1, -2, 3, -4];
      const encode = (type, values) => values.map((value) => value + type.zeroPoint);
      for (const qType of byteTypes) for (const kType of byteTypes) {
        for (const vType of byteTypes) for (const outType of byteTypes) {
          const spec = {
            qShape: [1, 4], qDtype: qType.dtype, qValues: encode(qType, centered),
            qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: qType.zeroPoint },
            kShape: [1, 4], kDtype: kType.dtype, kValues: encode(kType, centered),
            kQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: kType.zeroPoint },
            vShape: [1, 4], vDtype: vType.dtype, vValues: encode(vType, centered),
            vQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: vType.zeroPoint },
            outputDtype: outType.dtype,
            outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: outType.zeroPoint },
          };
          const expected = await cpuResult(spec);
          const values = qSDPAGraph(spec);
          wasm.compile(values.graph);
          const result = await wasm.execute({ q: values.qValues, k: values.kValues, v: values.vValues });
          assert.deepEqual([...result.out], [...expected.out]);
        }
      }
    });

    await t.test('SIMD128 full and eight-byte dot tiles preserve a 40-wide mixed-type head', async () => {
      const qValues = Array.from({ length: 3 * 40 }, (_, index) =>
        ((index * 17 + 11) % 101) - 53);
      const kValues = Array.from({ length: 5 * 40 }, (_, index) =>
        127 + (((index * 13 + 7) % 91) - 45));
      const vValues = Array.from({ length: 5 * 40 }, (_, index) =>
        120 + (((index * 19 + 5) % 81) - 40));
      const spec = {
        qShape: [3, 40], qDtype: 'int8', qValues,
        qQuantization: { scheme: 'per_tensor', scale: 0.03125, zero_point: -3 },
        kShape: [5, 40], kDtype: 'uint8', kValues,
        kQuantization: { scheme: 'per_tensor', scale: 0.015625, zero_point: 127 },
        vShape: [5, 40], vDtype: 'uint8', vValues,
        vQuantization: { scheme: 'per_tensor', scale: 0.0625, zero_point: 120 },
        outputDtype: 'int8',
        outputQuantization: { scheme: 'per_tensor', scale: 0.0625, zero_point: -7 },
        maskShape: [5], maskValues: [1, 1, 0, 1, 1], heads: 1,
        causal: false, scale: 0.125,
      };
      const expected = await cpuResult(spec);
      const values = qSDPAGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({
        q: values.qValues, k: values.kValues, v: values.vValues,
        mask: values.maskValues,
      });
      assert.ok(result.out instanceof Int8Array);
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('dynamic graph-input masks are required and validated before native execution', async () => {
      const values = qSDPAGraph({ maskShape: [2], maskValues: [1, 0] });
      const expected = await cpuResult({ maskShape: [2], maskValues: [1, 0] });
      assert.doesNotThrow(() => wasm.compile(values.graph));
      await assert.rejects(
        () => wasm.execute({ q: values.qValues, k: values.kValues, v: values.vValues }),
        /graph-input I32 mask supplied on every execution/,
      );
      await assert.rejects(
        () => wasm.execute({
          q: values.qValues, k: values.kValues, v: values.vValues, mask: new Int8Array(8),
        }),
        /typed storage does not match dtype/,
      );
      const result = await wasm.execute({
        q: values.qValues, k: values.kValues, v: values.vValues, mask: values.maskValues,
      });
      assert.deepEqual([...result.out], [...expected.out]);
    });

    await t.test('an ambiguous rank-2 mask uses established [B,K] precedence when B equals Q', async () => {
      // B and Q are both two.  Interpreting this as [Q,K] would make the
      // second query select key 1 in batch 0; [B,K] keeps each batch on its
      // own selected key for both queries.
      const spec = {
        qShape: [2, 2, 4], qValues: new Array(16).fill(0),
        qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        kShape: [2, 2, 4], kValues: new Array(16).fill(0),
        kQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        vShape: [2, 2, 4], vValues: [
          4, 0, 0, 0, -4, 0, 0, 0,
          8, 0, 0, 0, -8, 0, 0, 0,
        ],
        vQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
        maskShape: [2, 2], maskValues: [1, 0, 0, 1],
      };
      const values = qSDPAGraph(spec);
      wasm.compile(values.graph);
      const result = await wasm.execute({
        q: values.qValues, k: values.kValues, v: values.vValues, mask: values.maskValues,
      });
      assert.deepEqual([...result.out], [
        4, 0, 0, 0, 4, 0, 0, 0,
        -8, 0, 0, 0, -8, 0, 0, 0,
      ]);
    });

    await t.test('a rejected direct C descriptor leaves destination bytes untouched', () => {
      const qPointer = wasm.api.alloc_bytes(4);
      const kPointer = wasm.api.alloc_bytes(4);
      const vPointer = wasm.api.alloc_bytes(4);
      const outputPointer = wasm.api.alloc_bytes(4);
      new Int8Array(wasm.mem.buffer, qPointer, 4).set(Int8Array.of(1, 0, 0, 0));
      new Int8Array(wasm.mem.buffer, kPointer, 4).set(Int8Array.of(1, 0, 0, 0));
      new Int8Array(wasm.mem.buffer, vPointer, 4).set(Int8Array.of(1, 2, 3, 4));
      const output = new Uint8Array(wasm.mem.buffer, outputPointer, 4);
      output.fill(73);
      assert.equal(wasm.api.qsdpa_i8u8(
        qPointer, kPointer, vPointer, 0, outputPointer, 1, 1, 1, 4, 0,
        0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.5, 2, 2, 2, 2, 0, 0,
      ), 0);
      assert.deepEqual([...output], [73, 73, 73, 73]);
    });

    await t.test('source-view aliases and raw mask storage fail before WASM allocation hides them', () => {
      const aliased = qSDPAGraph();
      const shared = new ArrayBuffer(8);
      aliased.q.buffer = new Int8Array(shared);
      aliased.out.buffer = new Int8Array(shared);
      assert.throws(() => wasm.compile(aliased.graph), /output storage distinct from every input/);

      const rawMask = qSDPAGraph({ maskShape: [2], maskValues: [1, 0] });
      rawMask.mask.buffer = new ArrayBuffer(8);
      assert.throws(() => wasm.compile(rawMask.graph), /optional I32 mask storage/);
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
