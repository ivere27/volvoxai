import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {fixture, p, ok, tensors, safetensors, statelessGraph} from '../tools/proto_fixture.mjs';
import {reportTransport} from '../tools/proto_report_fixture.mjs';
import {EngineHost} from '../ts/host/EngineHost.js';
import {FullEngineHost} from '../ts/full.js';
import {VxInferenceServiceClient, VxBufferServiceClient, VxProfilingServiceClient}
  from '../runtime/generated/typescript/volvoxai_ffi.js';

const {version} = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const op = (id, opType, inputs, shape, params = {}) => ({id, opType, inputs, params,
  outputs: {out: {tensor: id, dtype: 'float32', shape}}});
const graph = mixed => statelessGraph({format: 'volvox-graph/v1', dimensions: {},
  inputs: {x: {dtype: 'float32', shape: [3, 65]}, mx: {dtype: 'float32', shape: [2, 4]},
    ids: {dtype: 'float32', shape: [2, 2]}, routes: {dtype: 'float32', shape: [2, 2]},
    tokens: {dtype: 'int32', shape: [2]}, edge: {dtype: 'float32', shape: [1, 2]}},
  nodes: [{...op('d0', 'Linear', {input: 'x', weight: 'w', bias: 'b'}, [3, 17]), id: 'w'},
    op('d1', 'MatMul', {input: 'x', weight: 'wt', bias: 'b'}, [3, 17]),
    op('d2', 'Gemm', {input: 'x', weight: 'w'}, [3, 17], {weight_layout: 'dout_din'}),
    op('emb', 'Embedding', {input: 'tokens', weight: 'w'}, [2, 65]),
    op('ml', 'MoELinear', {input: 'mx', expert_weight: 'ew', expert_bias: 'eb',
      route_indices: 'ids', route_weights: 'routes'}, [2, 3]),
    op('mg', 'MoEGatedMLP', {input: 'mx', gate_weight: 'gw', up_weight: 'uw', down_weight: 'dw',
      route_indices: 'ids', route_weights: 'routes'}, [2, 3], {route_input_scale: true, swiglu_limit: 2}),
    op('wide', 'Linear', {input: 'edge', weight: 'Linear'}, [1, 1]),
    ...(mixed ? [op('copy', 'Identity', {input: 'w'}, [17, 65])] : [])],
  outputs: ['d0', 'd1', 'd2', 'emb', 'ml', 'mg', 'wide'],
});
const inputValues = {
  x: {shape: [3, 65], data: Float32Array.from({length: 195}, (_, i) => (i % 13 - 6) * .125)},
  mx: {shape: [2, 4], data: Float32Array.of(.25, -1, .5, 2, -.5, .75, 1, -.25)},
  ids: {shape: [2, 2], data: Float32Array.of(2, 0, 1, 2)},
  routes: {shape: [2, 2], data: Float32Array.of(.75, .25, .5, .25)},
  tokens: {shape: [2], data: Int32Array.of(0, 16)},
  edge: {shape: [1, 2], data: Float32Array.of(2 ** -126, 2 ** 120)},
};
function weights(dtype, expanded) {
  const half = [0xb800, 0xb400, 0xb000, 0, 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00, 0x3d00];
  const brain = [0xbf00, 0xbe80, 0xbe00, 0, 0x3e00, 0x3e80, 0x3f00, 0x3f40, 0x3f80, 0x3fa0];
  const values = [-.5, -.25, -.125, 0, .125, .25, .5, .75, 1, 1.25];
  const names = ['w', 'wt', 'b', 'ew', 'eb', 'gw', 'uw', 'dw'];
  const shapes = [[17, 65], [65, 17], [17], [3, 4, 3], [3, 3], [3, 4, 5], [3, 4, 5], [3, 5, 3]];
  return [...names.map((name, part) => {
    const codes = Array.from({length: shapes[part].reduce((a, b) => a * b)}, (_, i) => {
      const source = part === 1 ? (i % 17) * 65 + Math.floor(i / 17) : i;
      return (source * 7 + (part <= 1 ? 0 : part)) % 10;
    });
    return {name, shape: shapes[part], dtype: expanded ? 'F32' : dtype,
      data: expanded ? Float32Array.from(codes, i => values[i]) :
        Uint16Array.from(codes, i => (dtype === 'F16' ? half : brain)[i])};
  }), {name: 'Linear', shape: [1, 2], dtype: expanded ? 'F32' : 'BF16',
    data: expanded ? Float32Array.of(1.9921875 * 2 ** 127, 2 ** -133) : Uint16Array.of(0x7f7f, 1)}];
}

function adapterPackage() {
  const bytes = safetensors([
    {name: 'a', shape: [2, 1], data: Float32Array.of(.5, 1)},
    {name: 'b', shape: [1, 2], data: Float32Array.of(2, -1)},
  ]);
  const headerSize = Number(new DataView(bytes.buffer).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerSize)));
  header.__metadata__ = {volvox_adapter_manifest: JSON.stringify({format: 'volvox.adapter.v1',
    adapter_id: 'compact-lora', version_id: 'v1', kind: 'lora',
    targets: [{weight: 'w', a: 'a', b: 'b', rank: 1, alpha: 1, layout: 'din_r_r_dout'}]})};
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - new TextEncoder().encode(json).length % 8) % 8);
  const encoded = new TextEncoder().encode(json), payload = bytes.subarray(8 + headerSize);
  const result = new Uint8Array(8 + encoded.length + payload.length);
  new DataView(result.buffer).setBigUint64(0, BigInt(encoded.length), true);
  result.set(encoded, 8); result.set(payload, 8 + encoded.length);
  return result;
}

for (const full of [false, true]) {
  test(`${full ? 'full' : 'inference'} WASM compact BF16 base with published LoRA`, async t => {
    const wasmUrl = new URL(`../dist/${version}/volvoxai${full ? '' : '.lite'}.wasm`, import.meta.url);
    const adapter = adapterPackage(), adapterBefore = adapter.slice();
    const host = new (full ? FullEngineHost : EngineHost)({wasmUrl, fetch: async path => {
      assert.equal(String(path), 'compact-lora.safetensors');
      return {ok: true, arrayBuffer: async () => adapter.slice().buffer};
    }});
    t.after(() => host.close());
    const transport = reportTransport(host), inference = new VxInferenceServiceClient(transport);
    const buffers = new VxBufferServiceClient(transport), profiling = new VxProfilingServiceClient(transport);
    const runtime = ok(await inference.createRuntime(new p.CreateRuntimeRequest({
      executionMode: p.ExecutionMode.EXECUTION_MODE_DIRECT})));
    const document = statelessGraph({format: 'volvox-graph/v1', dimensions: {},
      inputs: {x: {dtype: 'float32', shape: [1, 2]}},
      nodes: [op('linear', 'Linear', {input: 'x', weight: 'w'}, [1, 2])], outputs: ['linear']});
    const source = safetensors([{name: 'w', dtype: 'BF16', shape: [2, 2],
      data: Uint16Array.of(0x3f80, 0x4000, 0x4040, 0x4080)}]), before = source.slice();
    const model = ok(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
      package: new p.ModelPackage({graphDocument: new TextEncoder().encode(JSON.stringify(document)),
        weightShards: [source]})})));
    const revision = ok(await inference.publishAdapter(new p.PublishAdapterRequest({modelId: model.modelId,
      adapterName: 'compact-lora', versionName: 'v1', packagePath: 'compact-lora.safetensors'})));
    const policy = new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['wasm'], operatorFallback: p.OperatorFallback.OPERATOR_FALLBACK_FORBID});
    const contexts = [];
    for (const adapted of [false, true]) {
      const compiled = ok(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
        policy, ...(adapted ? {adapterRevision: new p.AdapterRevisionRef(revision)} : {})})));
      contexts.push(ok(await inference.createExecutionContext(new p.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new p.GraphExecution()}))));
      ok(await inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
    }
    ok(await inference.releaseModel(new p.ModelRef(model)));
    const checkStorage = async context => {
      const snapshot = ok(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest(context))).snapshot;
      assert.ok(snapshot.memory.allocations.every(a => a.allocator !== 'host.packed_weights' || a.capacityBytes === 0n),
        'LoRA must leave the BF16 base compact');
    };
    for (const [index, context] of contexts.entries()) {
      await checkStorage(context);
      const result = ok(await inference.execute(new p.ExecuteRequest({contextId: context.contextId,
        inputs: new p.InputBindings({values: tensors({x: {shape: [1, 2], data: Float32Array.of(2, 3)}})}),
        scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()}),
        outputs: new p.OutputOptions({storage: p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER})})));
      await checkStorage(context);
      ok(await inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
      const output = ok(await buffers.copyTensors(new p.CopyTensorsRequest({
        sources: result.outputs, inlineResult: true}))).outputs[0];
      assert.equal(output.dtype, p.DataType.DATA_TYPE_F32);
      const bytes = output.inline;
      assert.deepEqual(Array.from(new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))),
        index ? [16, 14] : [8, 18]);
      ok(await buffers.releaseBuffers(new p.BufferRefs({bufferIds: result.outputs.map(value => value.buffer.bufferId)})));
    }
    assert.deepEqual(source, before); assert.deepEqual(adapter, adapterBefore);
    ok(await inference.releaseRuntime(new p.RuntimeRef(runtime)));
  });
}

for (const full of [false, true]) for (const dtype of ['F16', 'BF16']) for (const mixed of [false, true]) {
  test(`${full ? 'full' : 'inference'} WASM ${dtype} dense/MoE ${mixed ? 'shared fallback' : 'compact storage'}`, async t => {
    const wasmUrl = new URL(`../dist/${version}/volvoxai${full ? '' : '.lite'}.wasm`, import.meta.url);
    const f = await fixture({wasmUrl, full}); t.after(() => f.close());
    const execute = async context => ok(await f.inference.execute(new p.ExecuteRequest({
      contextId: context.contextId, inputs: new p.InputBindings({values: tensors(inputValues)}),
      scope: new p.ExecutionScope({entrypoint: 'evaluate', full: new p.Empty()}),
      outputs: new p.OutputOptions({storage: p.OutputStorage.OUTPUT_STORAGE_OWNED_BUFFER}),
    })));
    const refModel = await f.load(graph(false), safetensors(weights(dtype, true)));
    const refCompiled = await f.compile(refModel), refContext = await f.context(refCompiled);
    const reference = await execute(refContext);
    ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(refContext)));
    ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(refCompiled)));
    ok(await f.inference.releaseModel(new p.ModelRef(refModel)));

    const model = await f.load(graph(mixed), safetensors(weights(dtype, false)));
    const compiled = await f.compile(model), contexts = [await f.context(compiled), await f.context(compiled)];
    ok(await f.inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
    ok(await f.inference.releaseModel(new p.ModelRef(model)));
    const profiling = new VxProfilingServiceClient(reportTransport(f.host));
    const checkStorage = async context => {
      const snapshot = ok(await profiling.getResourceSnapshot(new p.GetResourceSnapshotRequest(context))).snapshot;
      const packed = snapshot.memory.allocations.filter(a => a.allocator === 'host.packed_weights');
      for (const name of ['wt', 'ew', 'gw', 'uw', 'dw', 'Linear'])
        assert.equal(packed.filter(a => a.name === name).reduce((sum, a) => sum + a.capacityBytes, 0n), 0n, name);
      assert.equal(packed.filter(a => a.name === 'w').reduce((sum, a) => sum + a.capacityBytes, 0n), mixed ? 4420n : 0n);
      if (!mixed) assert.ok(packed.every(a => ['b', 'eb'].includes(a.name)), 'no hidden full F32 prepack');
      assert.equal(packed.filter(a => ['b', 'eb'].includes(a.name)).reduce((sum, a) => sum + a.capacityBytes, 0n), 104n);
    };
    for (const context of contexts) {
      await checkStorage(context);
      const actual = await execute(context);
      await checkStorage(context);
      ok(await f.inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
      assert.ok(actual.outputs.every(output => output.dtype === p.DataType.DATA_TYPE_F32));
      for (const name of graph(false).outputs) {
        const a = await f.read(actual, name), b = await f.read(reference, name);
        assert.equal(a.length, b.length);
        for (let i = 0; i < a.length; i++) assert.ok(Number.isFinite(a[i]) &&
          Math.abs(a[i] - b[i]) <= 1e-5 * (1 + Math.abs(b[i])), `${name}[${i}]: ${a[i]} != ${b[i]}`);
      }
      await f.releaseOutputs(actual);
    }
    await f.releaseOutputs(reference);
  });
}
