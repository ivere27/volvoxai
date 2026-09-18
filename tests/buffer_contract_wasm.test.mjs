import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { EngineHost } from '../ts/host/EngineHost.js';
import { FullEngineHost } from '../ts/full.js';
import { VxBufferServiceClient, VxInferenceServiceClient } from
  '../runtime/generated/typescript/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';

const { version } = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
const ok = response => assert.equal(response.report?.status ?? response.status,
  pb.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
const graph = new TextEncoder().encode(JSON.stringify({
  format: 'volvox-graph/v1', dimensions: {}, inputs: { x: { dtype: 'float32', shape: [2] } },
  nodes: [{ id: 'double', opType: 'Add', inputs: { a: 'x', b: 'x' },
    outputs: { out: { tensor: 'y', dtype: 'float32', shape: [2] } }, params: {} }],
  outputs: ['y'],
}));

for (const profile of ['inference', 'full']) {
  test(`${profile} WASM retains CPU storage, executes buffer inputs and rejects foreign pointers`, async () => {
    const wasmUrl = new URL(`../dist/${version}/volvoxai${profile === 'full' ? '' : '.lite'}.wasm`, import.meta.url);
    const host = profile === 'full' ? new FullEngineHost({ wasmUrl }) : new EngineHost({ wasmUrl });
    try {
      const buffers = new VxBufferServiceClient(reportTransport(host));
      const engine = new VxInferenceServiceClient(reportTransport(host));
      const values = Float32Array.of(1.25, -3.5);
      const batch = await buffers.copyTensors(new pb.CopyTensorsRequest({ sources: [new pb.Tensor({
        name: 'x', shape: [2n], dtype: pb.DataType.DATA_TYPE_F32, inline: new Uint8Array(values.buffer),
      })] }));
      ok(batch);
      const source = batch.outputs[0];
      const originalId = source.buffer.bufferId;
      const retained = await buffers.retainBuffers(new pb.BufferRefs({ bufferIds: [originalId] }));
      ok(retained);
      const retainedId = retained.buffers[0].bufferId;
      assert.notEqual(retainedId, originalId);
      ok(await buffers.releaseBuffers(new pb.BufferRefs({ bufferIds: [originalId, originalId] })));
      assert.equal((await buffers.getBufferInfo(new pb.BufferHandle({ bufferId: originalId }))).report.status,
        pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
      source.buffer.bufferId = retainedId;
      const info = await buffers.getBufferInfo(new pb.BufferHandle({ bufferId: retainedId }));
      ok(info);
      assert.equal(info.kind, pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST);
      assert.equal(info.sizeBytes, 8n);
      assert.equal((await buffers.beginBufferAccess(new pb.BufferAccessRequest({
        view: source.buffer, hostMapping: true,
      }))).report.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
      const runtime = await engine.createRuntime(new pb.CreateRuntimeRequest());
      ok(runtime);
      const model = await engine.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
        package: new pb.ModelPackage({ graphDocument: graph }) }));
      ok(model);
      const compiled = await engine.compileModel(new pb.CompileModelRequest({ modelId: model.modelId }));
      ok(compiled);
      const context = await engine.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId,
      }));
      ok(context);
      const result = await engine.executeTensors(new pb.ExecuteTensorsRequest({
        contextId: context.contextId, inputs: [source],
      }));
      ok(result);
      ok(await engine.releaseExecutionContext(new pb.ExecutionContextRef({ contextId: context.contextId })));
      const copied = await buffers.copyTensors(new pb.CopyTensorsRequest({ sources: result.outputs, inlineResult: true }));
      ok(copied);
      const data = copied.outputs[0].inline;
      assert.deepEqual([...new Float32Array(data.slice().buffer)], [2.5, -7]);
      const borrowed = new pb.Tensor({ name: 'x', shape: [2n], dtype: pb.DataType.DATA_TYPE_F32,
        borrowed: new pb.BorrowedBuffer({ resource: new pb.NativeResource({
          kind: pb.NativeResourceKind.NATIVE_RESOURCE_KIND_HOST, handle: 1n, sizeBytes: 8n,
        }), lengthBytes: 8n }) });
      assert.equal((await buffers.copyTensors(new pb.CopyTensorsRequest({ sources: [borrowed] }))).report.status,
        pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
      assert.equal((await buffers.importDLPack(new pb.ImportDLPackRequest({ managedTensor: 1n }))).report.status,
        pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
      assert.equal((await buffers.endBufferAccess(new pb.EndBufferAccessRequest({
        cuda: new pb.CudaStreamCompletion({ deviceId: 0, streams: [1n] }),
      }))).status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
      ok(await buffers.releaseBuffers(new pb.BufferRefs({ bufferIds: [retainedId, result.outputs[0].buffer.bufferId] })));
    } finally {
      await host.close();
    }
  });
}
