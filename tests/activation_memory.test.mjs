import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { fixture, p, ok, tensors, safetensors } from '../tools/proto_fixture.mjs';
import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { VxProfilingServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const wasmDirectory = process.env.VOLVOXAI_TEST_WASM_DIR ?? path.join(root, 'dist', version);
const length = 262145; // Non-power-of-two storage makes capacity rounding visible.
const mib = 1024 * 1024;

function graph(dynamic) {
  const shape = [1, dynamic ? 'N' : length];
  const nodes = [];
  let previous = 'x';
  for (let index = 0; index < 24; index++) {
    const name = `sum${index}`;
    nodes.push({ id: name, opType: 'Add', inputs: { a: previous, b: 'x' },
      outputs: { out: { tensor: name, dtype: 'float32', shape } }, params: {} });
    previous = name;
  }
  nodes.push({ id: 'view', opType: 'Identity', inputs: { input: 'sum0' },
    outputs: { out: { tensor: 'view', dtype: 'float32', shape } }, params: {} });
  nodes.push({ id: 'branch', opType: 'Add', inputs: { a: previous, b: 'view' },
    outputs: { out: { tensor: 'out', dtype: 'float32', shape } }, params: {} });
  return { format: 'volvox-graph/v1', dimensions: dynamic ? { N: { min: 1, max: length * 2 } } : {},
    inputs: { x: { dtype: 'float32', shape } }, nodes, outputs: ['sum0', 'view', 'out'] };
}

for (const full of [false, true]) {
  test(`${full ? 'full' : 'lite'} host arena shares dtype storage only across disjoint lifetimes`, async t => {
    const f = await fixture({ full, wasmUrl: path.join(wasmDirectory, full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm') });
    t.after(() => f.close());
    const profiling = new VxProfilingServiceClient(reportTransport(f.host));
    for (const retainQuantized of [false, true]) {
      const shape = [1, 'N'];
      const model = await f.load({ format: 'volvox-graph/v1', dimensions: { N: { min: 1, max: length * 2 } },
        inputs: { x: { dtype: 'float32', shape } },
        nodes: [
          { id: 'relu', opType: 'ReLU', inputs: { input: 'x' }, params: {},
            outputs: { out: { tensor: 'a', dtype: 'float32', shape } } },
          { id: 'quantize', opType: 'QuantizeLinear', inputs: { input: 'a', scale: 'scale', zero_point: 'zero' }, params: {},
            outputs: { out: { tensor: 'q', dtype: 'int8', shape } } },
          { id: 'dequantize', opType: 'DequantizeLinear', inputs: { input: 'q', scale: 'scale', zero_point: 'zero' }, params: {},
            outputs: { out: { tensor: 'b', dtype: 'float32', shape } } },
          { id: 'add', opType: 'Add', inputs: { a: 'b', b: 'x' }, params: {},
            outputs: { out: { tensor: 'y', dtype: 'float32', shape } } },
        ], outputs: retainQuantized ? ['y', 'q'] : ['y'],
        quantization: { format: 'volvox-affine-safetensors/v1', tensors: {
          q: { scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero' },
        } },
      }, safetensors([
        { name: 'scale', shape: [1], data: Float32Array.of(0.25) },
        { name: 'zero', shape: [1], dtype: 'I8', data: Int8Array.of(-3) },
      ]));
      const compiled = await f.compile(model);
      const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
      const trace = ok(await profiling.startTrace(new p.StartTraceRequest({ ...f.runtime, options: new p.TraceOptions({ memory: true }) })));
      for (const [count, value] of [[length, 0.5], [length * 2, -1], [3, 0.75], [length, 1]]) {
        const result = ok(await f.inference.execute(new p.ExecuteRequest({ ...context,
          inputs: tensors({ x: { data: new Float32Array(count).fill(value), shape: [1, count] } }),
        })));
        const y = await f.read(result, 'y');
        assert.equal(y.length, count);
        assert.ok(y.every(v => v === Math.max(0, value) + value));
        if (retainQuantized) {
          const { tensor } = ok(await f.inference.readOutput(new p.ReadOutputRequest({ ...result, name: 'q' })));
          const q = new Int8Array(tensor.inline.slice().buffer);
          assert.equal(q.length, count);
          assert.ok(q.every(v => v === Math.max(0, value) * 4 - 3), 'retained byte output was overwritten by F32 storage');
        }
        ok(await f.inference.releaseResult(new p.ResultRef(result)));
        if (count === length && value === 0.5) {
          ok(await profiling.stopTrace(new p.TraceRef(trace)));
          const captured = ok(await profiling.getTrace(new p.TraceRef(trace)));
          const arena = captured.allocators.find(item => item.allocator === 'host.arena');
          assert.ok(arena);
          const maximum = BigInt(length * (retainQuantized ? 13 : 12) + 128);
          assert.ok(arena.liveBytes <= maximum, 'disjoint byte storage must fit inside the F32 arena');
          ok(await profiling.releaseTrace(new p.TraceRef(trace)));
        }
      }
      ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
      ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
      ok(await f.inference.releaseModel(new p.ModelRef(model)));
    }
  });

  test(`${full ? 'full' : 'lite'} byte shape operators bind distinct storage after metadata validation`, async t => {
    const f = await fixture({ full, wasmUrl: path.join(wasmDirectory, full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm') });
    t.after(() => f.close());
    const model = await f.load({ format: 'volvox-graph/v1', dimensions: {},
      inputs: { x: { dtype: 'int8', shape: [2, 1] } },
      nodes: [
        { id: 'expand', opType: 'Expand', inputs: { input: 'x' }, params: { shape: [2, 3] },
          outputs: { out: { tensor: 'expanded', dtype: 'int8', shape: [2, 3] } } },
        { id: 'transpose', opType: 'Transpose', inputs: { input: 'expanded' }, params: { perm: [1, 0] },
          outputs: { out: { tensor: 'out', dtype: 'int8', shape: [3, 2] } } },
      ], outputs: ['out'],
      quantization: { format: 'volvox-affine-safetensors/v1', tensors: Object.fromEntries(
        ['x', 'expanded', 'out'].map(name => [name, { scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero' }])) },
    }, safetensors([
      { name: 'scale', shape: [1], data: Float32Array.of(0.25) },
      { name: 'zero', shape: [1], dtype: 'I8', data: Int8Array.of(-3) },
    ]));
    const compiled = await f.compile(model);
    for (const values of [[2, 5], [-7, 11]]) {
      const result = ok(await f.inference.run(new p.RunRequest({ ...compiled, inputs: [new p.Tensor({
        name: 'x', dtype: p.DataType.DATA_TYPE_I8, shape: [2n, 1n], inline: new Uint8Array(Int8Array.from(values).buffer),
      })] })));
      const { tensor } = ok(await f.inference.readOutput(new p.ReadOutputRequest({ ...result, name: 'out' })));
      assert.deepEqual(Array.from(new Int8Array(tensor.inline.slice().buffer)), [...values, ...values, ...values]);
      ok(await f.inference.releaseResult(new p.ResultRef(result)));
    }
  });

  test(`${full ? 'full' : 'lite'} convolution prepares graph-supplied parameters after binding`, async t => {
    const f = await fixture({ full, wasmUrl: path.join(wasmDirectory, full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm') });
    t.after(() => f.close());
    const shape = [1, 1, 1, 1];
    const model = await f.load({ format: 'volvox-graph/v1', dimensions: {},
      inputs: { x: { dtype: 'float32', shape }, w: { dtype: 'float32', shape }, b: { dtype: 'float32', shape: [1] } },
      nodes: [{ id: 'conv', opType: 'Conv2D', inputs: { input: 'x', weight: 'w', bias: 'b' },
        outputs: { out: { tensor: 'out', dtype: 'float32', shape } }, params: { weight_layout: 'HWIO' } }], outputs: ['out'] });
    const compiled = await f.compile(model);
    for (const value of [2, -3]) {
      const result = ok(await f.inference.run(new p.RunRequest({ ...compiled, inputs: tensors({
        x: { data: Float32Array.of(value), shape }, w: { data: Float32Array.of(value), shape },
        b: { data: Float32Array.of(0.5), shape: [1] },
      }) })));
      assert.deepEqual(Array.from(await f.read(result, 'out')), [value * value + 0.5]);
      ok(await f.inference.releaseResult(new p.ResultRef(result)));
    }
  });

  for (const dynamic of [false, true]) {
    test(`${full ? 'full' : 'lite'} ${dynamic ? 'dynamic' : 'fixed'} activation storage is planned once and preserves retained results`, async t => {
      let memory;
      const Instance = WebAssembly.Instance;
      WebAssembly.Instance = class extends Instance {
        constructor(...args) { super(...args); memory = this.exports.memory; }
      };
      let f;
      try {
        f = await fixture({ full, wasmUrl: path.join(wasmDirectory, full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm') });
      } finally { WebAssembly.Instance = Instance; }
      t.after(() => f.close());
      assert.ok(memory);
      const model = await f.load(graph(dynamic));
      const loadedBytes = memory.buffer.byteLength;
      const compiled = await f.compile(model);
      const context = ok(await f.inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
      assert.ok(memory.buffer.byteLength - loadedBytes < 4 * mib,
        'compile/context creation must not allocate every intermediate tensor');

      const profiling = new VxProfilingServiceClient(reportTransport(f.host));
      const trace = ok(await profiling.startTrace(new p.StartTraceRequest({ ...f.runtime, options: new p.TraceOptions({ memory: true }) })));
      const execute = async (count, value) => ok(await f.inference.execute(new p.ExecuteRequest({
        ...context, inputs: tensors({ x: { data: new Float32Array(count).fill(value), shape: [1, count] } }),
      })));
      const release = async result => ok(await f.inference.releaseResult(new p.ResultRef(result)));
      const check = async (result, count, value) => {
        for (const [name, factor] of [['sum0', 2], ['view', 2], ['out', 27]]) {
          const output = await f.read(result, name);
          assert.equal(output.length, count);
          assert.ok(output.every(element => element === value * factor), `${name} was overwritten`);
        }
      };
      const retained = await execute(length, 0.25);
      await check(retained, length, 0.25);
      ok(await profiling.stopTrace(new p.TraceRef(trace)));
      const captured = ok(await profiling.getTrace(new p.TraceRef(trace)));
      const arena = captured.allocators.find(item => item.allocator === 'host.arena');
      assert.ok(arena);
      assert.ok(arena.liveBytes < BigInt(6 * mib), 'first shape must not round its arena up to 8 MiB');
      ok(await profiling.releaseTrace(new p.TraceRef(trace)));

      for (const value of [-0.5, 1, -2, 0]) {
        const next = await execute(length, value);
        await check(next, length, value);
        await check(retained, length, 0.25);
        await release(next);
      }
      const invalid = await f.inference.execute(new p.ExecuteRequest({
        ...context, inputs: tensors({ x: { data: Float32Array.of(1), shape: [1, length] } }),
      }));
      assert.equal(invalid.report.status, p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
      await check(retained, length, 0.25);
      if (dynamic) {
        for (const count of [length * 2, 3, length]) {
          const next = await execute(count, -0.25);
          await check(next, count, -0.25);
          await check(retained, length, 0.25);
          await release(next);
        }
      }
      // Published result snapshots outlive their context and model owners.
      ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
      ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
      ok(await f.inference.releaseModel(new p.ModelRef(model)));
      await check(retained, length, 0.25);
      await release(retained);
    });
  }
}
