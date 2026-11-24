import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors, statelessGraph} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {VxProfilingServiceClient} from '../runtime/generated/typescript/volvoxai_ffi.js';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const node = (id, opType, inputs, shape, params = {}) => ({id, opType, inputs, params,
  outputs: {out: {tensor: id, dtype: 'float32', shape}}});
const cases = [
  {dtype: 'F16', bits: [0x3c00, 0xc000, 0x0001, 0x7bff, 0x8000, 0x3555,
    0x7c00, 0xfc00, 0x7e00, 0x0400, 0x3c01, 0xbc00],
  values: [1, -2, 2 ** -24, 65504, -0, 0.333251953125,
    Infinity, -Infinity, NaN, 2 ** -14, 1.0009765625, -1]},
  {dtype: 'BF16', bits: [0x3f80, 0xc000, 0x0001, 0x7f7f, 0x8000, 0x3eab,
    0x7f80, 0xff80, 0x7fc0, 0x0080, 0x3f81, 0xbf80],
  values: [1, -2, 2 ** -133, 1.9921875 * 2 ** 127, -0, 0.333984375,
    Infinity, -Infinity, NaN, 2 ** -126, 1.0078125, -1]},
];

for (const full of [false, true]) {
  const profile = full ? 'full' : 'inference';
  const wasmUrl = new URL(`../dist/${version}/volvoxai${full ? '' : '.lite'}.wasm`, import.meta.url);
  for (const {dtype, bits, values} of cases) for (const variant of ['lookup', 'mixed', 'output', 'declared-bank']) {
    test(`${profile} WASM ${dtype} ${variant}: compact lookup storage preserves F32 values`, async t => {
      const f = await fixture({wasmUrl, full});
      t.after(() => f.close());
      const bank = variant === 'declared-bank';
      const expanded = variant === 'mixed' || variant === 'output';
      let nodes = [
        node('e', 'Embedding', {input: 'ids', weight: 'w'}, [2, 3]),
        node('g', 'Gather', {input: 'w', indices: 'ids'}, [2, 3], {axis: 0}),
        node('h', 'Gather', {input: 'w', indices: 'columns'}, [4, 2], {axis: -1}),
      ];
      let outputs = ['e', 'g', 'h'];
      if (bank) { nodes = [nodes[1]]; outputs = ['g']; }
      if (variant === 'mixed') {
        nodes.push(node('extra', 'Identity', {input: 'w'}, [4, 3])); outputs.push('extra');
      } else if (variant === 'output') outputs.push('w');
      const model = await f.load(statelessGraph({format: 'volvox-graph/v1',
        dimensions: bank ? {E: {min: 1, max: 4}} : {}, ...(bank ? {banks: {w: 'E'}} : {}),
        inputs: {ids: {dtype: 'int32', shape: [2]}, columns: {dtype: 'int32', shape: [2]}},
        nodes, outputs,
      }), safetensors([{name: 'w', dtype, shape: [4, 3], data: Uint16Array.from(bits)}]));
      const compiled = await f.compile(model);
      const contexts = [await f.context(compiled), await f.context(compiled)];
      const profiling = new VxProfilingServiceClient(reportTransport(f.host));
      const checkStorage = async context => {
        const snapshot = ok(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest(context))).snapshot;
        const converted = snapshot.memory.allocations.filter(a => a.allocator === 'host.packed_weights' && a.name === 'w');
        assert.equal(converted.reduce((total, a) => total + a.capacityBytes, 0n), expanded ? 48n : 0n);
      };
      await checkStorage(contexts[0]);
      ok(await f.inference.releaseModel(new p.ModelRef(model)));
      ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
      const expectedRows = [...values.slice(9), ...values.slice(3, 6)];
      const expectedColumns = Array.from({length: 4}, (_, row) => [values[row * 3 + 2], values[row * 3]]).flat();
      for (const context of contexts) {
        const result = ok(await f.inference.execute(new p.ExecuteRequest({contextId: context.contextId,
          inputs: new p.InputBindings({values: tensors({
            ids: {data: Int32Array.of(3, 1), shape: [2]},
            columns: {data: Int32Array.of(-1, 0), shape: [2]},
          })}), scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()}),
          outputs: new p.OutputOptions({storage: p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER}),
        })));
        if (!bank) assert.deepEqual(Array.from(await f.read(result, 'e')), expectedRows);
        assert.deepEqual(Array.from(await f.read(result, 'g')), expectedRows);
        if (!bank) assert.deepEqual(Array.from(await f.read(result, 'h')), expectedColumns);
        if (expanded) assert.deepEqual(Array.from(await f.read(result, variant === 'mixed' ? 'extra' : 'w')), values);
        assert.ok(result.outputs.every(output => output.dtype === p.DataType.DATA_TYPE_F32));
        await checkStorage(context);
        ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
        assert.deepEqual(Array.from(await f.read(result, bank ? 'g' : 'e')), expectedRows);
        await f.releaseOutputs(result);
      }
    });
  }

  test(`${profile} WASM refuses an explicit CUDA weight-bank cache`, async t => {
    const f = await fixture({wasmUrl, full});
    t.after(() => f.close());
    const model = await f.load(statelessGraph({format: 'volvox-graph/v1',
      dimensions: {E: {min: 2, max: 2}}, banks: {experts: 'E'},
      inputs: {x: {dtype: 'float32', shape: [1, 2]},
        ids: {dtype: 'float32', shape: [1, 1]}, gates: {dtype: 'float32', shape: [1, 1]}},
      nodes: [node('y', 'MoELinear', {input: 'x', expert_weight: 'experts',
        route_indices: 'ids', route_weights: 'gates'}, [1, 2])], outputs: ['y'],
    }), safetensors([{name: 'experts', shape: [2, 2, 2], data: Float32Array.of(1, 0, 0, 1, 2, 0, 0, 2)}]));
    const ordinary = await f.compile(model);
    ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(ordinary)));
    const rejected = await f.inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
      policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['wasm'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID}),
      weightBankCache: [new p.WeightBankCache({bank: 'experts', residentSlots: 1})],
    }));
    assert.equal(rejected.report.status, p.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED);
    assert.equal(rejected.report.code, p.OperationCode.OPERATION_CODE_BACKEND_UNSUPPORTED);
    assert.equal(rejected.compiledModelId, 0n);
  });
}
