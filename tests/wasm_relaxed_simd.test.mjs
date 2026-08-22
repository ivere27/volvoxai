import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { WasmEngine } from '../ts/backends/WasmEngine.js';

const textEncoder = new TextEncoder();
const relaxedSectionName = 'volvoxai.relaxed_simd.v1';

function uleb(value) {
  const bytes = [];
  do {
    let byte = value & 0x7f;
    value >>>= 7;
    if (value !== 0) byte |= 0x80;
    bytes.push(byte);
  } while (value !== 0);
  return bytes;
}

function wasmString(value) {
  const bytes = [...textEncoder.encode(value)];
  return [...uleb(bytes.length), ...bytes];
}

function section(id, payload) {
  return [id, ...uleb(payload.length), ...payload];
}

function relaxedChild({ memoryMinimum = 1, parameterTypes = null,
  exportMemory = false } = {}) {
  const parameters = parameterTypes || [
    ...Array(10).fill(0x7f), 0x7d, 0x7f, 0x7d, ...Array(4).fill(0x7f),
  ];
  const type = section(1, [1, 0x60, ...uleb(parameters.length), ...parameters, 1, 0x7f]);
  const imports = section(2, [
    1, ...wasmString('env'), ...wasmString('memory'), 2, 0, ...uleb(memoryMinimum),
  ]);
  const functions = section(3, [1, 0]);
  const exportEntries = [
    ...wasmString('qlinear_i8u8_relaxed'), 0, 0,
  ];
  if (exportMemory) exportEntries.push(...wasmString('memory'), 2, 0);
  const exports = section(7, [exportMemory ? 2 : 1, ...exportEntries]);
  const body = parameters.length > 7
    ? [0, 0x20, 7, 0x0b]
    : [0, 0x41, 0, 0x0b];
  const code = section(10, [1, ...uleb(body.length), ...body]);
  return Uint8Array.from([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...type, ...imports, ...functions, ...exports, ...code,
  ]);
}

function parentModule(children = []) {
  const memory = section(5, [1, 0, 1]);
  const exports = section(7, [1, ...wasmString('memory'), 2, 0]);
  const customSections = children.flatMap((child) => section(0, [
    ...wasmString(relaxedSectionName), ...child,
  ]));
  return Uint8Array.from([
    0, 97, 115, 109, 1, 0, 0, 0,
    ...memory, ...exports, ...customSections,
  ]);
}

async function initFixture(children) {
  const directory = await mkdtemp(join(tmpdir(), 'volvoxai-relaxed-loader-'));
  const path = join(directory, 'volvoxai.wasm');
  await writeFile(path, parentModule(children));
  const engine = await WasmEngine.init(path);
  return { directory, engine };
}

function decodeQLinearGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 3], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const weight = graph.addWeight('weight', [8, 3], 'int8', {
    buffer: Int8Array.from({ length: 24 }, (_, index) => index % 7 - 3),
    quantization: {
      scheme: 'per_axis', axis: 0,
      scales: Array.from({ length: 8 }, (_, index) => 0.125 * (index + 1)),
      zero_points: Array.from({ length: 8 }, (_, index) => index - 4),
    },
  });
  const bias = graph.addWeight('bias', [8], 'int32', {
    buffer: Int32Array.from({ length: 8 }, (_, index) => index * 3 - 7),
  });
  const { out } = graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name: 'out', shape: [1, 2, 8], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
    },
  });
  graph.setOutputs([out.name]);
  return graph;
}

function mockQLinearEngine(relaxedKernel, { packedResult = 1 } = {}) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  let heap = 16;
  const calls = { packed: 0, raw: 0, relaxed: [] };
  const api = {
    memory,
    reset_heap: () => { heap = 16; },
    alloc_bytes: (bytes) => {
      const pointer = heap;
      heap = (heap + bytes + 15) & ~15;
      return pointer;
    },
    packed_q8_weight_size: () => 64,
    pack_q8_weight: () => 1,
    qlinear_i8u8_packed: () => { calls.packed++; return packedResult; },
    qlinear_i8u8: () => { calls.raw++; return 1; },
  };
  const relaxed = (...args) => {
    calls.relaxed.push(args);
    return relaxedKernel(...args);
  };
  const engine = new WasmEngine(
    { instance: { exports: api } },
    { instance: { exports: { qlinear_i8u8_relaxed: relaxed } } },
  );
  return { engine, calls };
}

async function seedAndStep(engine, graph) {
  engine.compile(graph);
  await engine.execute({ input: Int8Array.of(1, -2, 3, 4, 0, -1) }, {
    incremental: true, incrementalReset: true, changedInputs: ['input'],
  });
  await engine.execute({ input: Int8Array.of(1, -2, 3, 7, 5, -6) }, {
    incremental: true, changedInputs: ['input'], incrementalRowPosition: 1,
  });
}

test('WASM Relaxed-SIMD child loading is optional and fail-closed', async (t) => {
  await t.test('a baseline artifact without the custom section stays usable', async () => {
    const { directory, engine } = await initFixture([]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
      assert.equal(engine.relaxedApi, null);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('a malformed embedded child falls back to the baseline module', async () => {
    const { directory, engine } = await initFixture([Uint8Array.of(0)]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('an incompatible memory import falls back to the baseline module', async () => {
    const { directory, engine } = await initFixture([relaxedChild({ memoryMinimum: 2 })]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('duplicate versioned sections are rejected as ambiguous', async () => {
    const child = relaxedChild();
    const { directory, engine } = await initFixture([child, child]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('a versioned child with the wrong function arity is rejected', async () => {
    const { directory, engine } = await initFixture([
      relaxedChild({ parameterTypes: [] }),
    ]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('a versioned child with an unexpected export is rejected', async () => {
    const { directory, engine } = await initFixture([
      relaxedChild({ exportMemory: true }),
    ]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, false);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('a valid child is instantiated with the parent linear memory', async () => {
    const { directory, engine } = await initFixture([relaxedChild()]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, true);
      assert.equal(engine.relaxedApi.qlinear_i8u8_relaxed.length, 17);
      const args = Array(17).fill(0);
      args[7] = 1;
      assert.equal(engine.relaxedApi.qlinear_i8u8_relaxed(...args), 1);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  await t.test('a scratch fork can skip the optional child', async () => {
    const { directory, engine } = await initFixture([relaxedChild()]);
    try {
      assert.ok(engine);
      assert.equal(engine.relaxedSimdEnabled, true);
      const scratch = await engine.fork({ relaxedSimd: false });
      assert.equal(scratch.relaxedSimdEnabled, false);
      assert.equal(scratch.relaxedApi, null);
      assert.notEqual(scratch.mem, engine.mem, 'the scratch fork keeps isolated linear memory');
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

test('WASM Relaxed-SIMD QLinear dispatch is decode-row-only and preserves fallbacks', async (t) => {
  await t.test('an eligible immutable packed row keeps the faster baseline SIMD path', async () => {
    const graph = decodeQLinearGraph();
    const { engine, calls } = mockQLinearEngine(() => 1);
    await seedAndStep(engine, graph);

    assert.equal(calls.packed, 2, 'the full seed and incremental row use packed SIMD');
    assert.equal(calls.raw, 0);
    assert.equal(calls.relaxed.length, 0,
      'the slower optional child is not called after packed SIMD succeeds');
  });

  await t.test('a packed rejection delegates one decode row to the optional child', async () => {
    const graph = decodeQLinearGraph();
    const { engine, calls } = mockQLinearEngine(() => 1, { packedResult: 0 });
    await seedAndStep(engine, graph);

    assert.equal(calls.packed, 2);
    assert.equal(calls.raw, 1, 'the full seed cannot use the row-only child');
    assert.equal(calls.relaxed.length, 1);
    const args = calls.relaxed[0];
    const descriptor = engine.nodeMetadata.get(graph.nodes[0]);
    assert.equal(args.length, 17);
    assert.equal(args[0], engine.pointers.get('input') + 3, 'row input byte offset');
    assert.equal(args[1], engine.pointers.get('weight'), 'raw weight pointer');
    assert.equal(args[2], descriptor.packedWeightPointer, 'packed weight pointer');
    assert.equal(args[6], engine.pointers.get('out') + 8, 'row output byte offset');
    assert.equal(args[7], 1, 'one decode row');
    assert.deepEqual(args.slice(8, 10), [3, 8]);
  });

  await t.test('packed and child descriptor rejection retain the canonical raw fallback', async () => {
    const graph = decodeQLinearGraph();
    const { engine, calls } = mockQLinearEngine(() => 0, { packedResult: 0 });
    await seedAndStep(engine, graph);
    assert.equal(calls.relaxed.length, 1);
    assert.equal(calls.packed, 2);
    assert.equal(calls.raw, 2);
  });

  await t.test('a fallback child trap disables it and retains the canonical raw path', async () => {
    const graph = decodeQLinearGraph();
    const { engine, calls } = mockQLinearEngine(
      () => { throw new Error('child trap'); },
      { packedResult: 0 },
    );
    await seedAndStep(engine, graph);
    assert.equal(calls.relaxed.length, 1);
    assert.equal(calls.packed, 2);
    assert.equal(calls.raw, 2);
    assert.equal(engine.relaxedSimdEnabled, false, 'a trapping child is disabled permanently');
  });
});
