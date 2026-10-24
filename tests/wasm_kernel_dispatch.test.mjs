/** Quantized kernels stay reachable through the shipped protobuf dispatch. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { graphOf, runGraph } from './parity/external/webgpu_device_bridge.mjs';

const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const product = shape => shape.reduce((a, b) => a * b, 1);
const retiredExport = /^(?:matmul_f32$|pack(?:ed)?_q8_weight(?:_|$)|q(?:batch_matmul|conv2d|linear|sdpa)_|volvoxai_(?:training|ptq)_|vx_scheduler_policy_)/;

function affine(descriptors) {
  const weights = {}, quantization = {};
  for (const [name, { scale, zero, Type }] of Object.entries(descriptors)) {
    weights[`${name}.scale`] = { shape: [1], data: Float32Array.of(scale) };
    weights[`${name}.zero`] = { shape: [1], data: Type.of(zero) };
    quantization[name] = { scheme: 'per_tensor',
      scale_tensor: `${name}.scale`, zero_point_tensor: `${name}.zero` };
  }
  return { weights, quantization };
}

function roundEven(value) {
  const lower = Math.floor(value), fraction = value - lower;
  return fraction < 0.5 ? lower : fraction > 0.5 ? lower + 1
    : lower % 2 === 0 ? lower : lower + 1;
}

function matmulExpected(left, right, shape, descriptors) {
  // Independent scalar affine MatMul, including right-aligned batch broadcast.
  // Powers-of-two scales keep the oracle exact at rounding boundaries.
  const result = [], rank = shape.length;
  const [m, n] = shape.slice(-2), k = left.shape.at(-1);
  const { a, b, y } = descriptors;
  const minimum = y.Type === Int8Array ? -128 : 0;
  for (let batch = 0; batch < product(shape.slice(0, -2)); batch++) {
    let remainder = batch;
    const coordinates = [];
    for (let axis = rank - 3; axis >= 0; axis--) {
      coordinates[axis] = remainder % shape[axis];
      remainder = Math.floor(remainder / shape[axis]);
    }
    const offset = tensor => {
      let at = 0;
      for (let axis = 0; axis < tensor.shape.length - 2; axis++)
        at = at * tensor.shape[axis] + (tensor.shape[axis] === 1 ? 0
          : coordinates[rank - tensor.shape.length + axis]);
      return at * product(tensor.shape.slice(-2));
    };
    const ai = offset(left), bi = offset(right);
    for (let row = 0; row < m; row++) for (let column = 0; column < n; column++) {
      let sum = 0;
      for (let inner = 0; inner < k; inner++)
        sum += (left.data[ai + row * k + inner] - a.zero)
          * (right.data[bi + inner * n + column] - b.zero);
      result.push(Math.max(minimum, Math.min(minimum + 255,
        roundEven(sum * a.scale * b.scale / y.scale + y.zero))));
    }
  }
  return y.Type.from(result);
}

function matmulCase(broadcast) {
  const descriptors = {
    a: { scale: 1 / 32, zero: 139, Type: Uint8Array },
    b: { scale: 1 / 128, zero: -13, Type: Int8Array },
    y: { scale: 1 / 16, zero: broadcast ? 127 : -7,
      Type: broadcast ? Uint8Array : Int8Array },
  };
  const aShape = broadcast ? [2, 1, 3, 5] : [17, 65];
  const bShape = broadcast ? [1, 2, 5, 3] : [65, 63];
  const shape = broadcast ? [2, 2, 3, 3] : [17, 63];
  const left = { name: 'a', shape: aShape,
    data: Uint8Array.from({ length: product(aShape) }, (_, i) => (i * 29 + 17) & 255) };
  const right = { name: 'b', shape: bShape,
    data: Int8Array.from({ length: product(bShape) }, (_, i) => (i * 37 + 11) % 255 - 127) };
  const { weights, quantization } = affine(descriptors);
  const dtype = broadcast ? 'uint8' : 'int8';
  const graph = graphOf([{ id: 'product', opType: 'QBatchMatMul',
    inputs: { a: 'a', b: 'b' }, params: {},
    outputs: { out: { tensor: 'y', shape, dtype } } }],
  { a: { shape: aShape, dtype: 'uint8' }, b: { shape: bShape, dtype: 'int8' } },
  ['y'], weights, quantization);
  return { graph, batches: [[left, right], [left, right]], shape,
    Type: descriptors.y.Type, expected: matmulExpected(left, right, shape, descriptors) };
}

function attentionGolden() {
  const shape = [1, 1, 40], keyShape = [1, 5, 40];
  const { weights, quantization } = affine({
    q: { scale: 1 / 32, zero: -3, Type: Int8Array },
    k: { scale: 1 / 64, zero: 127, Type: Uint8Array },
    v: { scale: 1 / 16, zero: 120, Type: Uint8Array },
    y: { scale: 1 / 16, zero: -7, Type: Int8Array },
  });
  const graph = graphOf([{ id: 'attention', opType: 'QSDPA',
    inputs: { q: 'q', k: 'k', v: 'v', mask: 'mask' },
    outputs: { out: { tensor: 'y', shape, dtype: 'int8' } },
    params: { heads: 1, causal: false, scale: 0.125 } }],
  { q: { shape, dtype: 'int8' }, k: { shape: keyShape, dtype: 'uint8' },
    v: { shape: keyShape, dtype: 'uint8' }, mask: { shape: [5], dtype: 'int32' } },
  ['y'], weights, quantization);
  const inputs = [
    { name: 'q', shape, data: Int8Array.from({ length: 40 }, (_, i) => (i * 17 + 11) % 101 - 53) },
    { name: 'k', shape: keyShape, data: Uint8Array.from({ length: 200 }, (_, i) => 127 + (i * 13 + 7) % 91 - 45) },
    { name: 'v', shape: keyShape, data: Uint8Array.from({ length: 200 }, (_, i) => 120 + (i * 19 + 5) % 81 - 40) },
  ];
  // Retained portable-C contract golden, independent of this release's result.
  const expected = Int8Array.of(-24, -5, -1, 1, -29, -10, -6, -4, -11, -15,
    4, -9, -16, -20, -1, 3, 5, -25, -6, -2, 0, -7, -11, 8, -5, -12,
    -16, 3, -10, -17, -21, -2, 2, 4, -26, -7, -3, -1, -8, -12);
  const masks = [Int32Array.of(1, 1, 0, 1, 1), new Int32Array(5), Int32Array.of(1, 1, 0, 1, 1)];
  return { graph, shape, Type: Int8Array,
    batches: masks.map(data => [...inputs, { name: 'mask', shape: [5], data }]),
    expected: [expected, new Int8Array(40).fill(-7), expected] };
}

function causalAttention() {
  const shape = [2, 3, 8], maskShape = [2, 3, 3];
  const { weights, quantization } = affine({
    q: { scale: 1 / 32, zero: 127, Type: Uint8Array },
    k: { scale: 1 / 16, zero: -2, Type: Int8Array },
    v: { scale: 1 / 8, zero: -3, Type: Int8Array },
    y: { scale: 1 / 8, zero: 128, Type: Uint8Array },
  });
  const graph = graphOf([{ id: 'attention', opType: 'QSDPA',
    inputs: { q: 'q', k: 'k', v: 'v', mask: 'mask' },
    outputs: { out: { tensor: 'y', shape, dtype: 'uint8' } },
    params: { heads: 2, causal: true, scale: 0.5 } }],
  { q: { shape, dtype: 'uint8' }, k: { shape, dtype: 'int8' },
    v: { shape, dtype: 'int8' }, mask: { shape: maskShape, dtype: 'int32' } },
  ['y'], weights, quantization);
  const batches = [[
    { name: 'q', shape, data: new Uint8Array(48).fill(127) },
    { name: 'k', shape, data: Int8Array.from({ length: 48 }, (_, i) => i - 24) },
    { name: 'v', shape, data: Int8Array.from({ length: 48 },
      (_, i) => Math.floor(i / 24) * 30 + Math.floor(i % 24 / 8) * 6 + i % 8 - 3) },
    { name: 'mask', shape: maskShape, data: Int32Array.of(
      1, 1, 1, 1, 1, 1, 0, 1, 1,
      0, 0, 0, 0, 1, 1, 1, 0, 1) },
  ]];
  // Centered Q is zero: attention is the uniform mean of allowed V rows.
  // Causality leaves keys {0}, {0,1}, {1,2}, {}, {1}, {0,2}.
  const expected = Uint8Array.from([128, 131, 137, null, 164, 164]
    .flatMap(base => Array.from({ length: 8 }, (_, d) => base === null ? 128 : base + d)));
  return { graph, shape, Type: Uint8Array, batches, expected };
}

for (const profile of ['inference', 'full']) {
  const stem = `volvoxai${profile === 'inference' ? '.lite' : ''}`;
  const wasmUrl = new URL(`../dist/${version}/${stem}.wasm`, import.meta.url);
  const api = await import(new URL(`../dist/${version}/${stem}.min.js`, import.meta.url));
  // The shared parity runner accepts the older internal host class spelling.
  const runnerApi = { ...api, InferenceWasmHost: api.EngineHost };

  test(`${profile}: shipped WASM exposes transport without direct kernel roots`, async () => {
    const module = await WebAssembly.compile(await readFile(wasmUrl));
    const names = WebAssembly.Module.exports(module).map(entry => entry.name);
    assert.deepEqual(names.filter(name => retiredExport.test(name)), []);
    assert.ok(names.includes('synurang_module_send'));
  });

  for (const [name, fixture] of [
    ['QBatchMatMul mixed bytes and SIMD odd tails', matmulCase(false)],
    ['QBatchMatMul rank-four broadcasting and unsigned output', matmulCase(true)],
    ['QSDPA D40 key mask, all-masked rows, and context reuse', attentionGolden()],
    ['QSDPA two heads, causal BQK mask, and unsigned output', causalAttention()],
  ]) test(`${profile}: public dispatch preserves ${name}`, async () => {
    const result = await runGraph(runnerApi, { wasm: fileURLToPath(wasmUrl),
      afterRead: ({ result }) => {
        assert.equal(result.report.backend, 'wasm');
        assert.equal(result.report.fallback.operatorFallbackUsed, false);
        assert.deepEqual(result.outputs[0].shape, fixture.shape.map(BigInt));
        assert.equal(result.outputs[0].dtype, fixture.Type === Int8Array
          ? api.pb.DataType.DATA_TYPE_I8 : api.pb.DataType.DATA_TYPE_U8);
      },
    }, fixture.graph, fixture.batches, 'wasm');
    for (const [round, outputs] of result.rounds.entries()) {
      const expected = Array.isArray(fixture.expected) ? fixture.expected[round] : fixture.expected;
      assert.deepEqual(outputs.get('y'), expected, `round ${round}`);
    }
  });
}
