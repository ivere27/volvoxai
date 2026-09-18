import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const packageJson = JSON.parse(
  await readFile(new URL('../package.json', import.meta.url), 'utf8'),
);
const releaseRoot = new URL(`../dist/${packageJson.version}/`, import.meta.url);

for (const filename of [
  'volvoxai.lite.js',
  'volvoxai.lite.min.js',
  'volvoxai.js',
  'volvoxai.min.js',
]) {
  test(`${filename} copies oneof payloads without selecting undefined fields`, async () => {
    const {pb} = await import(new URL(filename, releaseRoot));
    const data = new Uint8Array(Float32Array.of(.25, -2.5, 7).buffer);
    const original = new pb.Tensor({name: 'master', dtype: pb.DataType.DATA_TYPE_F32, shape: [3n], inline: data});
    for (const tensor of [original, pb.Tensor.fromBinary(original.toBinary())]) {
      const copied = new pb.Tensor({...tensor, name: 'renamed'});
      const decoded = pb.Tensor.fromBinary(copied.toBinary());
      assert.equal(decoded.payloadCase, pb.TensorPayloadOneofCase.Inline);
      assert.deepEqual(decoded.inline, data);
      assert.equal(decoded.name, 'renamed');
    }
    for (const init of [{position: 0}, {idle: false}, {parked: true}]) {
      const action = new pb.DecodeLaneAction(init);
      assert.deepEqual(new pb.DecodeLaneAction({...action}).toBinary(), action.toBinary());
    }
    const dependency = new pb.DecodeStepRequest({contextId: 7n, dependencyUpdate: new pb.Empty()});
    const decodedDependency = pb.DecodeStepRequest.fromBinary(dependency.toBinary());
    assert.equal(decodedDependency.cursorCase, pb.DecodeStepRequestCursorOneofCase.DependencyUpdate);
    assert.deepEqual(new pb.DecodeStepRequest({...decodedDependency}).toBinary(), dependency.toBinary());
    const none = new pb.CreateBatchQueueRequest({maxLanes: undefined});
    assert.equal(none.toBinary().length, 0);
    // The wire can contain an older member followed by its replacement. Copy
    // the selected member even if the decoded object still holds older data.
    const older = new pb.Tensor({buffer: new pb.BufferView({bufferId: 1n, lengthBytes: 12n})}).toBinary();
    const combined = pb.Tensor.fromBinary(Uint8Array.from([...older, ...original.toBinary()]));
    assert.equal(combined.payloadCase, pb.TensorPayloadOneofCase.Inline);
    assert.deepEqual(pb.Tensor.fromBinary(new pb.Tensor({...combined}).toBinary()).inline, data);
  });
  test(`${filename} preserves nested protobuf message codecs`, async () => {
    const release = await import(new URL(filename, releaseRoot));
    const authoring = !filename.includes('.lite');
    assert.equal('VxPlanningServiceClient' in release, authoring);
    assert.equal('CreateGraphPlanRequest' in release.pb, authoring);
    assert.equal(typeof release.RpcError, 'function');
    const encoded = new release.pb.ExecutionResultHandle({
      resultId: 7n,
      executionId: 11n,
      report: new release.pb.OperationReport({
        status: release.pb.NativeStatus.NATIVE_STATUS_OK,
        lineage: new release.pb.Lineage({ executionId: 11n }),
        route: new release.pb.RouteEvidence({
          provider: 'wasm',
          attested: true,
          shapePlan: new release.pb.ShapePlanEvidence({ signature: 'S=8' }),
        }),
      }),
    }).toBinary();
    const decoded = release.pb.ExecutionResultHandle.fromBinary(encoded);

    assert.ok(decoded.report instanceof release.pb.OperationReport);
    assert.ok(decoded.report.lineage instanceof release.pb.Lineage);
    assert.ok(decoded.report.route instanceof release.pb.RouteEvidence);
    assert.ok(decoded.report.route.shapePlan instanceof release.pb.ShapePlanEvidence);
    assert.equal(decoded.report.lineage.executionId, 11n);
    assert.equal(decoded.report.route.shapePlan.signature, 'S=8');
  });
}

for (const filename of [
  'volvoxai.lite.js',
  'volvoxai.lite.min.js',
  'volvoxai.js',
  'volvoxai.min.js',
]) {
  const full = !filename.includes('.lite');
  // Standalone planning is a full-profile surface; the lite bundles publish no
  // VxPlanningService client. Scheduler coverage below stays on every bundle.
  if (full) test(`${filename} executes standalone planning through the common proto API`, async () => {
    const release = await import(new URL(filename, releaseRoot));
    const Host = release.FullEngineHost;
    assert.equal(typeof Host, 'function');
    assert.equal(typeof release.VxPlanningServiceClient, 'function');
    for (const name of ['ModelBuilder', 'Tokenizer', 'createAuthoring', 'Model', 'Runtime']) {
      assert.equal(name in release, false, name);
    }
    const host = new Host({ wasmUrl: new URL(full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm', releaseRoot) });
    try {
      const planning = new release.VxPlanningServiceClient(host);
      const result = await planning.createGraphPlan(new release.pb.CreateGraphPlanRequest({
        graph: new release.pb.GraphPlanningSource({ graphDocument: new TextEncoder().encode(JSON.stringify({
          format: 'volvox-graph/v1', dimensions: {}, inputs: { x: { dtype: 'float32', shape: [1] } }, nodes: [], outputs: ['x'],
        })) }),
      }));
      assert.ok(result.graphPlanId > 0n);
      await planning.releaseGraphPlan(new release.pb.GraphPlanRef(result));
    } finally { await host.close(); }
  });
  test(`${filename} rejects unknown scheduler requests through the common proto API`, async () => {
    const release = await import(new URL(filename, releaseRoot));
    const {pb, VolvoxAIError} = release;
    const Host = full ? release.FullEngineHost : release.EngineHost;
    const host = new Host({ wasmUrl: new URL(full ? 'volvoxai.wasm' : 'volvoxai.lite.wasm', releaseRoot) });
    try {
      const scheduler = new release.VxSchedulerServiceClient(host);
      for (const operation of ['pollRequest', 'waitRequest']) {
        await assert.rejects(scheduler[operation](new pb.RequestRef({requestId: 0n})), error =>
          error instanceof VolvoxAIError &&
          error.status === pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED &&
          error.code === pb.OperationCode.OPERATION_CODE_HANDLE_DISPOSED &&
          error.report === error.response.report);
      }
    } finally { await host.close(); }
  });
}
