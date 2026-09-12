import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  loadModelControlWasmDispatchFactory,
  ModelControlWasmDispatchFactory,
} from '../ts/core/ModelControlWasm.js';
import {
  VxInferenceServiceClient,
  VxPlanningServiceClient,
  VxPlatformServiceClient,
  VxSchedulerServiceClient,
} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/inference/volvoxai_lite.js';
import { unary } from '../runtime/generated/typescript/inference/synurang_runtime.js';

const byteCodec = { encode: bytes => bytes, decode: bytes => bytes };

const packageJson = JSON.parse(
  await readFile(new URL('../package.json', import.meta.url), 'utf8'),
);
const wasmUrl = new URL(
  `../dist/${packageJson.version}/volvoxai.wasm`,
  import.meta.url,
);
const GRAPH_PATH = '/model-control-dispatch/graph.json';
const WEIGHTS_PATH = '/model-control-dispatch/model.safetensors';
const GRAPH = new TextEncoder().encode(JSON.stringify({
  format: 'volvox-graph/v1',
  dimensions: { B: { min: 1, max: 1 } },
  inputs: { x: { dtype: 'float32', shape: ['B', 2] } },
  nodes: [{
    id: 'projection',
    opType: 'MatMul',
    inputs: { input: 'x', weight: 'parameter' },
    outputs: {
      out: { tensor: 'logits', dtype: 'float32', shape: ['B', 2] },
    },
    params: {},
  }],
  outputs: ['logits'],
}));

function safetensors() {
  const values = Float32Array.of(0.2, -0.4, 0.1, 0.3);
  let header = new TextEncoder().encode(JSON.stringify({
    parameter: { dtype: 'F32', shape: [2, 2], data_offsets: [0, values.byteLength] },
  }));
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding !== 0) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + values.byteLength);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  output.set(new Uint8Array(values.buffer), 8 + header.byteLength);
  return output;
}

function largeSafetensors(dataBytes) {
  assert.equal(dataBytes % Float32Array.BYTES_PER_ELEMENT, 0);
  let header = new TextEncoder().encode(JSON.stringify({
    unused: {
      dtype: 'F32',
      shape: [dataBytes / Float32Array.BYTES_PER_ELEMENT],
      data_offsets: [0, dataBytes],
    },
  }));
  const padding = (8 - (header.byteLength % 8)) % 8;
  if (padding !== 0) {
    const padded = new Uint8Array(header.byteLength + padding);
    padded.set(header);
    padded.fill(0x20, header.byteLength);
    header = padded;
  }
  const output = new Uint8Array(8 + header.byteLength + dataBytes);
  new DataView(output.buffer).setBigUint64(0, BigInt(header.byteLength), true);
  output.set(header, 8);
  return output;
}

function inputTensor() {
  const values = Float32Array.of(1, 2);
  return new pb.Tensor({
    name: 'x',
    shape: [1n, 2n],
    dtype: pb.DataType.DATA_TYPE_F32,
    inline: new Uint8Array(values.buffer),
  });
}

function wrappingControlFactory(source, wrapExports) {
  return new (class extends ModelControlWasmDispatchFactory {
    instantiate(wakeup) {
      const instance = super.instantiate(wakeup);
      return { exports: wrapExports(instance.exports) };
    }
  })(source.module);
}

test('model-control owner drives generated unary dispatch with private persistent state', async () => {
  const factory = await loadModelControlWasmDispatchFactory(
    wasmUrl,
  );
  const owner = factory.create();
  const platform = new VxPlatformServiceClient(owner);
  const inference = new VxInferenceServiceClient(owner);
  const planning = new VxPlanningServiceClient(owner);
  const scheduler = new VxSchedulerServiceClient(owner);
  let runtimeId = 0n;
  let modelId = 0n;
  let compiledModelId = 0n;
  let contextId = 0n;
  let resultId = 0n;
  let directResultId = 0n;
  let requestId = 0n;
  let scheduledResultId = 0n;
  let graphPlanId = 0n;
  let modelFilesMounted = false;
  try {
    const rawPlatformInfo = (await unary(
      owner, { path: '/volvoxai.v1.VxPlatformService/GetPlatformInfo' },
      new pb.Empty().toBinary(), byteCodec, byteCodec,
    ));
    const retainedPlatformInfo = rawPlatformInfo.slice();
    const info = (await platform.getPlatformInfo(new pb.Empty()));
    assert.equal(info.profile, pb.BuildProfile.BUILD_PROFILE_INFERENCE);
    assert.equal(info.transport, pb.TransportProfile.TRANSPORT_PROFILE_REMOTE);
    assert.deepEqual(
      rawPlatformInfo,
      retainedPlatformInfo,
      'a later dispatch must not mutate a returned response copy',
    );

    const runtime = (await inference.createRuntime(new pb.CreateRuntimeRequest({
      cpuThreads: 1,
      executionMode: pb.ExecutionMode.EXECUTION_MODE_SCHEDULED,
    })));
    assert.equal(runtime.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(runtime.runtimeId > 0n);
    runtimeId = runtime.runtimeId;

    owner.mountFile(GRAPH_PATH, GRAPH);
    const weightBytes = safetensors();
    owner.mountFileChunks(WEIGHTS_PATH, [
      weightBytes.subarray(0, 7),
      weightBytes.subarray(7, 19),
      weightBytes.subarray(19),
    ]);
    modelFilesMounted = true;
    const model = (await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: GRAPH_PATH,
      weightPaths: [WEIGHTS_PATH],
    })));
    assert.equal(model.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(model.modelId > 0n);
    modelId = model.modelId;

    const graphPlan = (await planning.createGraphPlan(new pb.CreateGraphPlanRequest({ modelId })));
    assert.equal(graphPlan.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(graphPlan.graphPlanId > 0n);
    graphPlanId = graphPlan.graphPlanId;
    assert.equal(
      graphPlan.sourceKind,
      pb.GraphPlanSourceKind.GRAPH_PLAN_SOURCE_KIND_MODEL,
    );
    const graphPlanInfo = (await planning.getGraphPlan(new pb.GraphPlanRef({ graphPlanId })));
    assert.equal(graphPlanInfo.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(graphPlanInfo.plan?.planIdentity, graphPlan.plan?.planIdentity);
    const resolution = (await planning.resolveGraphPlan(new pb.ResolveGraphPlanRequest({
      graphPlanId,
      minimum: new pb.Empty(),
    })));
    assert.equal(resolution.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(resolution.planIdentity, graphPlan.plan?.planIdentity);
    assert.equal(
      (await planning.releaseGraphPlan(new pb.GraphPlanRef({ graphPlanId }))).status,
      pb.NativeStatus.NATIVE_STATUS_OK,
    );
    graphPlanId = 0n;
    assert.equal(
      (await planning.getGraphPlan(new pb.GraphPlanRef({ graphPlanId: graphPlan.graphPlanId })))
        .report?.status,
      pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
    );
    assert.equal(
      (await planning.releaseGraphPlan(new pb.GraphPlanRef({ graphPlanId: graphPlan.graphPlanId }))).status,
      pb.NativeStatus.NATIVE_STATUS_OK,
    );

    const compiled = (await inference.compileModel(new pb.CompileModelRequest({ modelId })));
    assert.equal(compiled.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(compiled.compiledModelId > 0n);
    compiledModelId = compiled.compiledModelId;

    const submitted = (await scheduler.submit(new pb.SubmitRequest({
      compiledModelId,
      inputs: [inputTensor()],
    })));
    assert.equal(submitted.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(submitted.requestId > 0n);
    requestId = submitted.requestId;

    const nonblocking = (await scheduler.pollRequest(new pb.RequestRef({ requestId })));
    assert.equal(
      nonblocking.report?.status,
      pb.NativeStatus.NATIVE_STATUS_BUSY,
      'PollRequest must not pump queued work',
    );
    assert.equal(nonblocking.state, pb.RequestState.REQUEST_STATE_QUEUED);

    const waited = (await scheduler.waitRequest(new pb.RequestRef({ requestId })));
    assert.equal(waited.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(waited.state, pb.RequestState.REQUEST_STATE_SUCCEEDED);
    const scheduledResult = (await scheduler.takeRequestResult(
      new pb.RequestRef({ requestId }),
    ));
    assert.equal(
      scheduledResult.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
    );
    assert.ok(scheduledResult.resultId > 0n);
    scheduledResultId = scheduledResult.resultId;

    // The compiled child owns its Model and Runtime. Drop both public parent
    // handles before entering the synchronous threadless Run path to prove
    // that generated handle ownership preserves this ordinary call chain.
    assert.equal((await inference.releaseModel(
      new pb.ModelRef({ modelId }),
    )).status, pb.NativeStatus.NATIVE_STATUS_OK);
    modelId = 0n;
    assert.equal((await inference.releaseRuntime(
      new pb.RuntimeRef({ runtimeId }),
    )).status, pb.NativeStatus.NATIVE_STATUS_OK);
    runtimeId = 0n;

    const directlyRun = (await inference.run(new pb.RunRequest({
      compiledModelId,
      inputs: [inputTensor()],
    })));
    assert.equal(directlyRun.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(directlyRun.resultId > 0n);
    directResultId = directlyRun.resultId;
    const directRead = (await inference.readOutput(new pb.ReadOutputRequest({
      resultId: directResultId,
      name: 'logits',
    })));
    assert.equal(directRead.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(directRead.tensor?.inline?.byteLength, 8);

    const context = (await inference.createExecutionContext(
      new pb.CreateExecutionContextRequest({ compiledModelId }),
    ));
    assert.equal(context.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(context.contextId > 0n);
    contextId = context.contextId;

    const foreignView = new pb.BufferView({
      handle: 0x7fff_fff0n,
      offset: 8n,
      length: 8n,
      space: pb.MemorySpace.MEMORY_SPACE_JS_ARRAY_BUFFER,
    });
    const rejectedViewInput = (await inference.execute(new pb.ExecuteRequest({
      contextId,
      inputs: [new pb.Tensor({
        name: 'x',
        shape: [1n, 2n],
        dtype: pb.DataType.DATA_TYPE_F32,
        view: foreignView,
      })],
    })));
    assert.equal(
      rejectedViewInput.report?.status,
      pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
    );
    assert.equal(rejectedViewInput.report?.code, pb.OperationCode.OPERATION_CODE_TRANSPORT_UNSUPPORTED);
    assert.equal(rejectedViewInput.resultId, 0n);

    const executed = (await inference.execute(new pb.ExecuteRequest({
      contextId,
      inputs: [inputTensor()],
    })));
    assert.equal(executed.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(executed.resultId > 0n);
    resultId = executed.resultId;
    const read = (await inference.readOutput(new pb.ReadOutputRequest({
      resultId,
      name: 'logits',
    })));
    assert.equal(read.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.equal(read.tensor?.inline?.byteLength, 8);
    const rejectedReadInto = (await inference.readOutput(new pb.ReadOutputRequest({
      resultId,
      name: 'logits',
      into: foreignView,
    })));
    assert.equal(
      rejectedReadInto.report?.status,
      pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED,
    );
    assert.equal(rejectedReadInto.report?.code, pb.OperationCode.OPERATION_CODE_TRANSPORT_UNSUPPORTED);

    owner.mountFile('/model-control-dispatch-fixture.bin', Uint8Array.of(1, 2, 3));
    owner.mountFile('/model-control-dispatch-fixture.bin', Uint8Array.of(4, 5));
    owner.unmountFile('/model-control-dispatch-fixture.bin');
    assert.throws(
      () => owner.unmountFile('/model-control-dispatch-fixture.bin'),
      /does not own mount path/,
    );

    (await assert.rejects(
      async () => (await unary(
        owner, { path: '/volvoxai.v1.VxInferenceService/DoesNotExist' },
        new Uint8Array(), byteCodec, byteCodec,
      )),
      /[Mm]ethod|[Uu]nimplemented/,
    ));
  } finally {
    try {
      if (graphPlanId !== 0n) assert.equal((await planning.releaseGraphPlan(
        new pb.GraphPlanRef({ graphPlanId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (scheduledResultId !== 0n) assert.equal((await inference.releaseResult(
        new pb.ResultRef({ resultId: scheduledResultId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (directResultId !== 0n) assert.equal((await inference.releaseResult(
        new pb.ResultRef({ resultId: directResultId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (requestId !== 0n) assert.equal((await scheduler.releaseRequest(
        new pb.RequestRef({ requestId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (resultId !== 0n) assert.equal((await inference.releaseResult(
        new pb.ResultRef({ resultId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (contextId !== 0n) assert.equal((await inference.releaseExecutionContext(
        new pb.ExecutionContextRef({ contextId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (compiledModelId !== 0n) assert.equal((await inference.releaseCompiledModel(
        new pb.CompiledModelRef({ compiledModelId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (modelId !== 0n) assert.equal((await inference.releaseModel(
        new pb.ModelRef({ modelId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      if (modelFilesMounted) {
        owner.unmountFile(WEIGHTS_PATH);
        owner.unmountFile(GRAPH_PATH);
      }
      if (runtimeId !== 0n) assert.equal((await inference.releaseRuntime(
        new pb.RuntimeRef({ runtimeId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
    } finally {
      (await owner.close());
    }
  }
  assert.equal(owner.closed, true);
  (await assert.rejects(
    async () => (await unary(
      owner, { path: '/volvoxai.v1.VxPlatformService/GetPlatformInfo' },
      new Uint8Array(), byteCodec, byteCodec,
    )),
    /requires an open owner/,
  ));
});

test('VFS holds every staged and snapshotted source at the 32-shard model boundary', async () => {
  const factory = await loadModelControlWasmDispatchFactory(
    wasmUrl,
  );
  const owner = factory.create();
  const inference = new VxInferenceServiceClient(owner);
  const graphPath = '/model-control-dispatch/maximum-shards.graph.json';
  const weightPaths = Array.from(
    { length: 32 },
    (_unused, index) =>
      `/model-control-dispatch/maximum-shard-${index}.safetensors`,
  );
  const mounted = [];
  let runtimeId = 0n;
  let modelId = 0n;
  try {
    owner.mountFile(graphPath, GRAPH);
    mounted.push(graphPath);
    const weight = safetensors();
    for (const weightPath of weightPaths) {
      owner.mountFile(weightPath, weight);
      mounted.push(weightPath);
    }

    const runtime = (await inference.createRuntime(new pb.CreateRuntimeRequest({
      cpuThreads: 1,
    })));
    assert.equal(runtime.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    runtimeId = runtime.runtimeId;
    const model = (await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath,
      weightPaths,
    })));
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message,
    );
    assert.ok(model.modelId > 0n);
    modelId = model.modelId;
  } finally {
    try {
      if (modelId !== 0n) assert.equal((await inference.releaseModel(
        new pb.ModelRef({ modelId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
      for (let index = mounted.length - 1; index >= 0; index--) {
        owner.unmountFile(mounted[index]);
      }
      if (runtimeId !== 0n) assert.equal((await inference.releaseRuntime(
        new pb.RuntimeRef({ runtimeId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
    } finally {
      (await owner.close());
    }
  }
});

test('large standalone Planning weight sets remain canonical at model scale', async (t) => {
  const factory = await loadModelControlWasmDispatchFactory(
    wasmUrl,
  );
  const owner = factory.create();
  const planning = new VxPlanningServiceClient(owner);
  const weightCount = 4096;
  const graphDocument = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [],
    outputs: ['x'],
  }));
  const weights = Array.from({ length: weightCount }, (_, index) =>
    new pb.PlanningWeight({
      name: `weight_${String(weightCount - 1 - index).padStart(6, '0')}`,
      dtype: pb.DataType.DATA_TYPE_F16,
      shape: [1n],
    }));
  let graphPlanId = 0n;
  try {
    const started = performance.now();
    const created = (await planning.createGraphPlan(new pb.CreateGraphPlanRequest({
      graph: new pb.GraphPlanningSource({ graphDocument, weights }),
    })));
    const elapsedMilliseconds = performance.now() - started;
    assert.equal(created.report?.status, pb.NativeStatus.NATIVE_STATUS_OK,
      created.report?.message);
    assert.ok(created.graphPlanId > 0n);
    graphPlanId = created.graphPlanId;
    const weightNames = created.plan?.tensors
      .filter(({ kind }) => kind === pb.GraphTensorKind.GRAPH_TENSOR_KIND_WEIGHT)
      .map(({ name }) => name) ?? [];
    assert.equal(weightNames.length, weightCount);
    assert.deepEqual(weightNames, weightNames.toSorted(),
      'the public plan must preserve canonical weight-name order');
    t.diagnostic(
      `${weightCount} reverse-ordered Planning weights: ` +
      `${elapsedMilliseconds.toFixed(3)} ms`,
    );
  } finally {
    if (graphPlanId !== 0n) {
      assert.equal((await planning.releaseGraphPlan(
        new pb.GraphPlanRef({ graphPlanId }),
      )).status, pb.NativeStatus.NATIVE_STATUS_OK);
    }
    (await owner.close());
  }
});

test('model-scale VFS shards use a fixed mailbox beyond the former 64-MiB address ceiling', async () => {
  const sourceFactory = await loadModelControlWasmDispatchFactory(
    wasmUrl,
  );
  const beginSizes = [];
  const chunkSizes = [];
  let memory;
  const factory = wrappingControlFactory(sourceFactory, (exports) => {
    memory = exports.memory;
    return {
      ...exports,
      vx_wasm_mount_file_begin(path, bytes) {
        beginSizes.push(bytes);
        return exports.vx_wasm_mount_file_begin(path, bytes);
      },
      vx_wasm_mount_file_write(data, bytes) {
        chunkSizes.push(bytes);
        return exports.vx_wasm_mount_file_write(data, bytes);
      },
    };
  });
  const owner = factory.create();
  const bytes = largeSafetensors(34 * 1024 * 1024);
  const paths = [
    '/model-control-dispatch/model-00001-of-00002.safetensors',
    '/model-control-dispatch/model-00002-of-00002.safetensors',
  ];
  const graphPath = '/model-control-dispatch/snapshot.graph.json';
  const graph = new TextEncoder().encode(JSON.stringify({
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [],
    outputs: ['x'],
  }));
  const inference = new VxInferenceServiceClient(owner);
  let runtimeId = 0n;
  let modelId = 0n;
  let firstMounted = false;
  let secondMounted = false;
  let graphMounted = false;
  try {
    owner.mountFile(paths[0], bytes);
    firstMounted = true;
    owner.mountFile(paths[1], bytes);
    secondMounted = true;
    assert.deepEqual(beginSizes, [bytes.byteLength, bytes.byteLength]);
    assert.equal(
      chunkSizes.reduce((total, value) => total + value, 0),
      bytes.byteLength * paths.length,
    );
    assert.equal(chunkSizes.length, 70);
    assert.ok(chunkSizes.every((value) => value > 0 && value <= 1024 * 1024));

    // The response allocation now starts above the former 64-MiB address
    // ceiling. A valid signed pointer within current linear memory must work.
    const platform = (await new VxPlatformServiceClient(owner).getPlatformInfo(new pb.Empty()));
    assert.equal(platform.profile, pb.BuildProfile.BUILD_PROFILE_INFERENCE);
    const mountedHighWater = memory.buffer.byteLength;

    /* Leave one original shard live and free the other shard's equal-sized
     * block. The snapshot knows its final size, and SafeTensors borrows that
     * immutable VFS storage. Either geometric snapshot growth or another
     * model-scale parser copy would push the load beyond the bound below. */
    owner.unmountFile(paths[1]);
    secondMounted = false;
    owner.mountFile(graphPath, graph);
    graphMounted = true;
    const runtime = (await inference.createRuntime(new pb.CreateRuntimeRequest({ cpuThreads: 1 })));
    assert.equal(runtime.report?.status, pb.NativeStatus.NATIVE_STATUS_OK);
    runtimeId = runtime.runtimeId;
    const model = (await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath,
      weightPaths: [paths[0]],
    })));
    assert.equal(model.report?.status, pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message);
    modelId = model.modelId;
    assert.ok(memory instanceof WebAssembly.Memory);
    const finalHighWater = memory.buffer.byteLength;
    const allowedHighWater = mountedHighWater + bytes.byteLength + 8 * 1024 * 1024;
    assert.ok(
      finalHighWater <= allowedHighWater,
      `exact snapshot reserve exceeded its bounded high-water: ` +
      `mounted=${mountedHighWater}, final=${finalHighWater}, ` +
      `growth=${finalHighWater - mountedHighWater}, allowed=${allowedHighWater}`,
    );
  } finally {
    if (modelId !== 0n) (await inference.releaseModel(new pb.ModelRef({ modelId })));
    if (runtimeId !== 0n) (await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId })));
    if (graphMounted) owner.unmountFile(graphPath);
    if (secondMounted) owner.unmountFile(paths[1]);
    if (firstMounted) owner.unmountFile(paths[0]);
    (await owner.close());
  }
});

test('an interrupted VFS upload is aborted and never becomes fopen-visible', async () => {
  const sourceFactory = await loadModelControlWasmDispatchFactory(
    wasmUrl,
  );
  let writes = 0;
  let aborts = 0;
  const factory = wrappingControlFactory(sourceFactory, (exports) => ({
    ...exports,
    vx_wasm_mount_file_write(data, bytes) {
      writes++;
      return writes === 2 ? -1 : exports.vx_wasm_mount_file_write(data, bytes);
    },
    vx_wasm_mount_file_abort() {
      aborts++;
      return exports.vx_wasm_mount_file_abort();
    },
  }));
  const owner = factory.create();
  const path = '/model-control-dispatch/interrupted-shard.safetensors';
  try {
    assert.throws(
      () => owner.mountFile(path, new Uint8Array(1024 * 1024 + 1)),
      /rejected a mount chunk/,
    );
    assert.equal(aborts, 1);
    assert.throws(() => owner.unmountFile(path), /does not own mount path/);

    owner.mountFile(path, Uint8Array.of(1, 2, 3));
    owner.unmountFile(path);
  } finally {
    (await owner.close());
  }
});
