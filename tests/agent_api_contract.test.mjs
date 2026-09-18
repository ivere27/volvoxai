import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

const releaseRoot = new URL('../dist/0.5.0/', import.meta.url);
const schemaHash = createHash('sha256').update(await readFile(new URL('../proto/volvoxai.proto', import.meta.url))).digest('hex');
const graphDocument = () => new TextEncoder().encode(JSON.stringify({
  format: 'volvox-graph/v1', dimensions: {B: {min: 1, max: 4}},
  inputs: {x: {dtype: 'float32', shape: ['B', 2]}, y: {dtype: 'float32', shape: ['B', 2]}},
  nodes: [{id: 'projection', opType: 'MatMul', inputs: {input: 'x', weight: 'parameter'},
    outputs: {out: {tensor: 'logits', dtype: 'float32', shape: ['B', 2]}}, params: {}}],
  outputs: ['logits'],
}));
const bytes = values => new Uint8Array(values.buffer, values.byteOffset, values.byteLength).slice();
const floats = data => new Float32Array(data.slice().buffer);

// VxPlanningService is a full-profile authoring surface. Author the package once
// there; every profile must still load the bytes it produces.
async function authorModelPackage() {
  const api = await import(new URL('volvoxai.js', releaseRoot));
  const {pb: p} = api;
  const host = new api.FullEngineHost({wasmUrl: new URL('volvoxai.wasm', releaseRoot)});
  const planning = new api.VxPlanningServiceClient(reportTransport(host));
  try {
    const plan = checkedReport(await planning.createGraphPlan(new p.CreateGraphPlanRequest({graph:
      new p.GraphPlanningSource({graphDocument: graphDocument(), weights: [
        new p.PlanningWeight({name: 'parameter', dtype: p.DataType.DATA_TYPE_F32, shape: [2n, 2n]}),
      ]}),
    })));
    const graph = checkedReport(await planning.exportGraphPlan(new p.GraphPlanRef(plan))).source.graphDocument;
    const shard = checkedReport(await planning.writeSafetensors(new p.WriteSafetensorsRequest({edits: [
      new p.SafetensorsEdit({setTensor: new p.Tensor({name: 'parameter', shape: [2n, 2n],
        dtype: p.DataType.DATA_TYPE_F32, inline: bytes(Float32Array.of(2, 0, 0, 3))})}),
    ]}))).data;
    checkedReport(await planning.releaseGraphPlan(new p.GraphPlanRef(plan)));
    return {graph, shard};
  } finally { await host.close(); }
}
const authored = await authorModelPackage();

for (const filename of ['volvoxai.lite.js', 'volvoxai.lite.min.js', 'volvoxai.js', 'volvoxai.min.js']) {
  test(`${filename}: discovers contracts, loads authored bytes and explains invalid inputs`, async () => {
    const api = await import(new URL(filename, releaseRoot));
    const {pb: p} = api; const check = checkedReport;
    const full = !filename.includes('.lite');
    const Host = full ? api.FullEngineHost : api.EngineHost;
    let fetches = 0;
    const host = new Host({wasmUrl: new URL(full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm', releaseRoot),
      fetch: async () => { fetches++; throw new Error('inline packages must not fetch'); }});
    const inference = new api.VxInferenceServiceClient(reportTransport(host));
    const platform = new api.VxPlatformServiceClient(reportTransport(host));
    const scheduler = new api.VxSchedulerServiceClient(reportTransport(host));
    const F32 = p.DataType.DATA_TYPE_F32;
    const tensor = (name = 'x', changes = {}) => new p.Tensor({name, shape: [2n, 2n], dtype: F32,
      inline: bytes(Float32Array.of(1, 2, 3, 4)), ...changes});
    try {
      const platformInfo = await platform.getPlatformInfo(new p.Empty());
      assert.equal(platformInfo.compiledBackends.includes('webgpu'), full);
      const catalog = check(await platform.describeApi(new p.DescribeApiRequest()));
      assert.equal(catalog.schemaSha256, schemaHash);
      assert.equal(catalog.messages.length, 0);
      assert.ok(catalog.methods.some(m => m.name === 'GetModelInfo'));
      assert.equal(catalog.methods.some(m => m.service === 'VxTrainingService'), full);
      const described = check(await platform.describeApi(new p.DescribeApiRequest({
        service: 'VxInferenceService', method: 'Run', includeTypes: true,
      })));
      assert.deepEqual(described.methods.map(m => m.name), ['Run']);
      const run = described.messages.find(m => m.name.endsWith('.RunRequest'));
      assert.equal(run.fields.find(f => f.name === 'compiled_model_id').handleKind, 'CompiledModel');
      assert.equal(run.fields.find(f => f.name === 'compiled_model_id').required, true);
      assert.ok(described.methods[0].rules.some(r => r.kind === p.ApiRuleKind.API_RULE_KIND_COMPLETE_INPUT_BATCH));
      const tensorType = described.messages.find(m => m.name.endsWith('.Tensor'));
      assert.deepEqual(tensorType.fields.filter(f => f.oneof === 'payload').map(f => f.name), ['inline', 'buffer', 'borrowed']);
      const defaults = check(await platform.describeApi(new p.DescribeApiRequest({
        service: 'VxInferenceService', method: 'CreateRuntime', includeTypes: true,
      })));
      const budget = defaults.messages.find(m => m.name.endsWith('.RuntimeBudget'));
      const limit = budget.fields.find(f => f.name === 'max_scheduled_requests');
      assert.equal(limit.protoDefault, '"0"');
      assert.equal(limit.engineDefault, '64');
      assert.equal(limit.hasPresence, true);
      const load = check(await platform.describeApi(new p.DescribeApiRequest({
        service: 'VxInferenceService', method: 'LoadModel', includeTypes: true,
      })));
      assert.ok(load.messages.find(m => m.name.endsWith('.LoadModelRequest')).rules.some(r =>
        r.kind === p.ApiRuleKind.API_RULE_KIND_EXACTLY_ONE && r.fields.join(',') === 'graph_path,package'));
      for (const filter of [{service: 'MissingService'}, {service: 'VxInferenceService', method: 'MissingMethod'},
        ...(!full ? [{service: 'VxTrainingService'}] : [])]) {
        assert.equal((await platform.describeApi(new p.DescribeApiRequest(filter))).report.status,
          p.NativeStatus.NATIVE_STATUS_NOT_FOUND);
      }
      assert.equal((await platform.describeApi(new p.DescribeApiRequest({method: 'Run'}))).report.status,
        p.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);

      const runtime = check(await inference.createRuntime(new p.CreateRuntimeRequest({
        executionMode: p.ExecutionMode.EXECUTION_MODE_SCHEDULED,
      })));
      // These are actual outputs of the authoring APIs, forwarded directly.
      // Each bundle gets its own copies; the test zeroes them to prove C copied.
      const graph = authored.graph.slice();
      const shard = authored.shard.slice();
      const modelPackage = new p.ModelPackage({graphDocument: graph, weightShards: [shard]});
      for (const source of [
        {package: modelPackage, graphPath: 'graph.json'},
        {package: modelPackage, graphPath: '\0'},
        {package: modelPackage, weightPaths: ['weights.safetensors']},
        {package: new p.ModelPackage()},
        {package: new p.ModelPackage({graphDocument: graph, weightShards: [new Uint8Array()]})},
        {package: new p.ModelPackage({graphDocument: Uint8Array.of(1, 2, 3)})},
      ]) {
        const rejected = await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId, ...source}));
        assert.notEqual(rejected.report.status, 0);
        assert.equal(rejected.modelId, 0n);
      }
      const model = check(await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId, package: modelPackage})));
      graph.fill(0); shard.fill(0);
      const info = check(await inference.getModelInfo(new p.ModelRef(model)));
      assert.equal(info.report.lineage.modelId, model.modelId);
      assert.deepEqual(info.inputs.map(t => t.name), ['x', 'y']);
      assert.deepEqual(info.outputs.map(t => t.name), ['logits']);
      assert.equal(info.inputs[0].dtype, F32);
      assert.equal(info.inputs[0].dimensions[0].symbol, 'B');
      assert.equal(info.inputs[0].dimensions[0].min, 1n);
      assert.equal(info.inputs[0].dimensions[0].max, 4n);
      assert.equal(info.outputs[0].dimensions[0].symbol, 'B');
      const compiled = check(await inference.compileModel(new p.CompileModelRequest({modelId: model.modelId,
        policy: new p.BackendPolicy({mode: p.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm']}),
      })));
      const context = check(await inference.createExecutionContext(new p.CreateExecutionContextRequest(compiled)));
      for (const response of [
        await inference.executeTensors(new p.ExecuteRequest({contextId: context.contextId,
          inputs: [new p.Tensor({name: 'x', dtype: F32, shape: [2n, 2n],
            borrowed: new p.BorrowedBuffer({resource: new p.NativeResource({handle: 1n, sizeBytes: 16n,
              kind: p.NativeResourceKind.NATIVE_RESOURCE_KIND_CUDA}), lengthBytes: 16n})})]})),
      ]) {
        assert.equal(response.report.status, p.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
      }
      check(await inference.releaseModel(new p.ModelRef(model)));
      assert.equal((await inference.getModelInfo(new p.ModelRef(model))).report.status, p.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
      const first = check(await inference.execute(new p.ExecuteRequest({contextId: context.contextId,
        inputs: [tensor(), tensor('y')],
      })));
      const cases = [
        ['DTYPE_MISMATCH', [tensor('x', {dtype: p.DataType.DATA_TYPE_I32}), tensor('y')]],
        ['RANK_MISMATCH', [tensor('x', {shape: [4n]}), tensor('y')]],
        ['DIMENSION_OUT_OF_RANGE', [tensor('x', {shape: [5n, 2n]}), tensor('y')]],
        ['DIMENSION_OUT_OF_RANGE', [tensor('x', {shape: [-1n, 2n]}), tensor('y')]],
        ['BYTE_SIZE_MISMATCH', [tensor('x', {inline: new Uint8Array(4)}), tensor('y')]],
        ['MISSING_INPUT', [tensor()]],
        ['DUPLICATE_INPUT', [tensor(), tensor()]],
        ['UNKNOWN_INPUT', [tensor('unknown'), tensor('y')]],
        ['UNKNOWN_INPUT', [tensor('😀'.repeat(70)), tensor('y')]],
        ['SYMBOL_MISMATCH', [tensor(), tensor('y', {shape: [1n, 2n], inline: new Uint8Array(8)})]],
        ['PAYLOAD_REQUIRED', [new p.Tensor({name: 'x', dtype: F32, shape: [2n, 2n]}), tensor('y')]],
        ['TRANSPORT_UNSUPPORTED', [new p.Tensor({name: 'x', dtype: F32, shape: [2n, 2n],
          borrowed: new p.BorrowedBuffer({resource: new p.NativeResource({kind: p.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST, handle: 1n, sizeBytes: 16n}), lengthBytes: 16n})}), tensor('y')]],
        ['TRANSPORT_UNSUPPORTED', [new p.Tensor({name: 'x', dtype: F32, shape: [2n, 2n],
          borrowed: new p.BorrowedBuffer({resource: new p.NativeResource({kind: p.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST,
            handle: 1n, sizeBytes: 16n}), lengthBytes: 16n})}), tensor('y')]],
        ['INVALID_NAME', [tensor(''), tensor('y')]],
        ['INVALID_NAME', [tensor('x\0extra'), tensor('y')]],
        ['RANK_MISMATCH', [tensor('x', {shape: Array(9).fill(1n)}), tensor('y')]],
      ];
      for (const [code, inputs] of cases) {
        const response = await inference.execute(new p.ExecuteRequest({contextId: context.contextId, inputs}));
        assert.notEqual(response.report.status, 0, code);
        assert.equal(response.resultId, 0n, code);
        const issue = response.report.inputIssue;
        assert.ok(issue, code);
        assert.equal(issue.code, p.InputValidationCode[`INPUT_VALIDATION_CODE_${code}`], code);
        assert.throws(() => check(response), error => error.report.inputIssue === issue);
        if (code === 'MISSING_INPUT') {
          assert.equal(issue.inputName, 'y'); assert.equal(Object.hasOwn(issue.toJson(), 'inputIndex'), false);
          assert.equal(issue.actual, undefined); assert.equal(issue.expected.name, 'y');
        } else {
          assert.ok(issue.actual); assert.equal(Object.hasOwn(issue.toJson(), 'inputIndex'), true);
          assert.equal(issue.actualRank, BigInt(inputs[Number(issue.inputIndex)].shape.length));
        }
        if (code === 'UNKNOWN_INPUT' && inputs[0].name.includes('😀')) {
          assert.equal(issue.namesTruncated, true);
          assert.equal(issue.inputName, '😀'.repeat(63));
        }
        if (code === 'BYTE_SIZE_MISMATCH') {
          assert.equal(issue.expectedByteSize, 16n); assert.equal(issue.actual.byteSize, 4n);
        }
        if (code === 'SYMBOL_MISMATCH') {
          assert.equal(issue.axis, 0); assert.equal(issue.requiredExtent, 2n);
          assert.equal(issue.expected.dimensions[0].symbol, 'B');
        }
      }
      // Run and scheduler admission share the C validator with Execute.
      const badInputs = [tensor('x', {inline: new Uint8Array(3)}), tensor('y')];
      const badRun = await inference.run(new p.RunRequest({compiledModelId: compiled.compiledModelId, inputs: badInputs}));
      const badSubmit = await scheduler.submit(new p.SubmitRequest({compiledModelId: compiled.compiledModelId, inputs: badInputs}));
      for (const response of [badRun, badSubmit])
        assert.equal(response.report.inputIssue.code, p.InputValidationCode.INPUT_VALIDATION_CODE_BYTE_SIZE_MISMATCH);
      assert.equal(badSubmit.requestId, 0n);
      const next = check(await inference.execute(new p.ExecuteRequest({contextId: context.contextId,
        inputs: [tensor('x', {inline: bytes(Float32Array.of(2, 3, 4, 5))}), tensor('y')],
      })));
      for (const [result, expected] of [[first, [2, 6, 6, 12]], [next, [4, 9, 8, 15]]]) {
        const output = check(await inference.readOutput(new p.ReadOutputRequest({resultId: result.resultId, name: 'logits'})));
        assert.deepEqual([...floats(output.tensor.inline)], expected);
        check(await inference.releaseResult(new p.ResultRef(result)));
      }
      check(await inference.releaseExecutionContext(new p.ExecutionContextRef(context)));
      check(await inference.releaseCompiledModel(new p.CompiledModelRef(compiled)));
      check(await inference.releaseRuntime(new p.RuntimeRef(runtime)));
      assert.equal(fetches, 0);
    } finally { await host.close(); }
  });

  test(`${filename}: inline packages respect the host transport budget`, async () => {
    const api = await import(new URL(filename, releaseRoot));
    const {pb: p} = api; const check = checkedReport;
    const full = !filename.includes('.lite');
    const Host = full ? api.FullEngineHost : api.EngineHost;
    const host = new Host({wasmUrl: new URL(full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm', releaseRoot), maxPackageBytes: 16});
    try {
      const inference = new api.VxInferenceServiceClient(reportTransport(host));
      const runtime = check(await inference.createRuntime(new p.CreateRuntimeRequest()));
      const rejected = await inference.loadModel(new p.LoadModelRequest({runtimeId: runtime.runtimeId,
        package: new p.ModelPackage({graphDocument: graphDocument()}),
      }));
      assert.equal(rejected.modelId, 0n);
      assert.equal(rejected.report.code, p.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE);
    } finally { await host.close(); }
  });
}
