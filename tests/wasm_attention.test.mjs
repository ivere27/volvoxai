import test from 'node:test';
import assert from 'node:assert/strict';

import { WasmEngine } from '../ts/backends/WasmEngine.js';

function tensor(name, shape) {
  return { name, shape, dtype: 'float32', sizeBytes: shape.reduce((a, b) => a * b, 1) * 4 };
}

function mockEngine(node, tensors, pointers, api) {
  const memory = { buffer: new ArrayBuffer(4096) };
  /* Exercise the real backend lifecycle: bypassing the constructor would also
     bypass API-v1 capabilities and decode-cache ownership state. */
  const engine = new WasmEngine({ instance: { exports: { memory } } });
  engine.api = api;
  engine.mem = memory;
  engine.pointers = new Map(Object.entries(pointers));
  engine.compiledTopologyRevision = 0;
  engine.compiledWeightRevision = 0;
  engine.graph = {
    nodes: [node],
    tensors: new Map(tensors.map((value) => [value.name, value])),
    outputNames: [node.outputs.out.name],
    topologyRevision: 0,
    weightRevision: 0,
  };
  return engine;
}

test('WASM attention treats rank-2 as implicit batch one and loops rank-3 batches', async (t) => {
  await t.test('SDPA', async () => {
    const qkv = tensor('qkv', [2, 3, 6]);
    const out = tensor('out', [2, 3, 2]);
    const calls = [];
    const engine = mockEngine({
      id: 'sdpa', opType: 'SDPA', inputs: { qkv }, outputs: { out }, params: { heads: 1 },
    }, [qkv, out], { qkv: 64, out: 512 }, {
      sdpa_f32(...args) { calls.push(args); },
    });
    assert.equal(engine.preparedGraph, undefined);
    await engine.execute({});
    const lazilyPrepared = engine.preparedGraph;
    assert.equal(Object.isFrozen(lazilyPrepared), true);
    assert.equal(lazilyPrepared.schedule[0].kernelRoute, 'sdpa');
    assert.deepEqual(calls.map((args) => args.slice(0, 2)), [
      [64, 512],
      [64 + 3 * 6 * 4, 512 + 3 * 2 * 4],
    ]);
    assert.deepEqual(calls.map((args) => args.slice(2, 6)), [[3, 2, 1, 2], [3, 2, 1, 2]]);

    qkv.shape = [3, 6]; qkv.sizeBytes = 3 * 6 * 4;
    out.shape = [3, 2]; out.sizeBytes = 3 * 2 * 4;
    calls.length = 0;
    await engine.execute({});
    assert.equal(engine.preparedGraph, lazilyPrepared,
      'direct execution resolves its schedule once and reuses it');
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0].slice(0, 6), [64, 512, 3, 2, 1, 2]);
  });

  await t.test('CrossSDPA', async () => {
    const q = tensor('q', [2, 3, 2]);
    const k = tensor('k', [2, 4, 2]);
    const v = tensor('v', [2, 4, 2]);
    const out = tensor('out', [2, 3, 2]);
    const calls = [];
    const engine = mockEngine({
      id: 'cross', opType: 'CrossSDPA', inputs: { q, k, v }, outputs: { out }, params: { heads: 1 },
    }, [q, k, v, out], { q: 64, k: 256, v: 512, out: 768 }, {
      cross_sdpa_f32(...args) { calls.push(args); },
    });
    await engine.execute({});
    assert.deepEqual(calls.map((args) => args.slice(0, 4)), [
      [64, 256, 512, 768],
      [64 + 3 * 2 * 4, 256 + 4 * 2 * 4, 512 + 4 * 2 * 4, 768 + 3 * 2 * 4],
    ]);
    assert.deepEqual(calls.map((args) => args.slice(4, 10)), [
      [3, 4, 2, 1, 2, 1 / Math.sqrt(2)],
      [3, 4, 2, 1, 2, 1 / Math.sqrt(2)],
    ]);
  });
});

test('WASM compile aliases chained inference Dropout without allocating or executing outputs', async () => {
  const memory = { buffer: new ArrayBuffer(4096) };
  let nextPointer = 64;
  const allocations = [];
  const engine = new WasmEngine({
    instance: {
      exports: {
        memory,
        reset_heap() { nextPointer = 64; },
        alloc_bytes(bytes) {
          const pointer = nextPointer;
          nextPointer += bytes;
          allocations.push([pointer, bytes]);
          return pointer;
        },
      },
    },
  });
  const input = { ...tensor('input', [4]), isInput: true };
  const first = tensor('first', [4]);
  const second = tensor('second', [4]);
  const graph = {
    nodes: [
      { id: 'dropout.0', opType: 'Dropout', inputs: { input }, outputs: { out: first }, params: { p: 0.5 } },
      { id: 'dropout.1', opType: 'Dropout', inputs: { input: first }, outputs: { out: second }, params: { p: 0.5 } },
    ],
    tensors: new Map([[input.name, input], [first.name, first], [second.name, second]]),
    outputNames: [second.name], topologyRevision: 0, weightRevision: 0,
  };

  engine.compile(graph);
  assert.equal(allocations.length, 1);
  assert.equal(engine.pointers.get(first.name), engine.pointers.get(input.name));
  assert.equal(engine.pointers.get(second.name), engine.pointers.get(input.name));
  const values = Float32Array.of(1, -2, 3, -4);
  const result = await engine.execute({ input: values });
  assert.deepEqual([...result[second.name]], [...values]);
});

test('WASM GELU selects erf by default and tanh only when requested', async () => {
  const input = tensor('input', [4]);
  const output = tensor('output', [4]);
  const node = { id: 'gelu', opType: 'GELU', inputs: { input }, outputs: { out: output }, params: {} };
  const calls = [];
  const engine = mockEngine(node, [input, output], { input: 64, output: 128 }, {
    gelu_f32(...args) { calls.push(['none', ...args]); },
    gelu_tanh_f32(...args) { calls.push(['tanh', ...args]); },
  });
  await engine.execute({});
  node.params.approximate = 'tanh';
  await engine.execute({});
  assert.deepEqual(calls, [
    ['none', 64, 128, 4],
    ['tanh', 64, 128, 4],
  ]);
});
