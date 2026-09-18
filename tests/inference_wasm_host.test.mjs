import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { reportTransport } from '../tools/proto_report_fixture.mjs';
import { EngineHost } from '../ts/host/EngineHost.js';
import { loadModelControlWasmDispatchFactory } from
  '../ts/core/ModelControlWasm.js';
import {
  VxInferenceServiceClient,
  VxPlatformServiceClient,
} from '../runtime/generated/typescript/inference/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/inference/volvoxai_lite.js';
import { inspectBrowserReleaseBundle } from '../tools/release_profiles.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(
  await readFile(new URL('../package.json', import.meta.url), 'utf8'),
);
const wasmUrl = new URL(
  `../dist/${packageJson.version}/volvoxai.lite.wasm`,
  import.meta.url,
);
const GRAPH_SOURCE = 'fixture/model.graph.json';
const WEIGHT_SOURCE = 'fixture/model.safetensors';
const ADAPTER_SOURCE = 'fixture/adapter.bin';
const GRAPH_TEXT = JSON.stringify({
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
});

function safetensors() {
  const values = Float32Array.of(0.2, -0.4, 0.1, 0.3);
  let header = new TextEncoder().encode(JSON.stringify({
    parameter: {
      dtype: 'F32',
      shape: [2, 2],
      data_offsets: [0, values.byteLength],
    },
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

function exactArrayBuffer(bytes) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function reusingStreamBody(bytes, chunkBytes) {
  const scratch = new Uint8Array(chunkBytes);
  let offset = 0;
  return {
    getReader: () => ({
      read: async () => {
        if (offset === bytes.byteLength) return { done: true };
        const length = Math.min(chunkBytes, bytes.byteLength - offset);
        scratch.fill(0);
        scratch.set(bytes.subarray(offset, offset + length));
        offset += length;
        return { done: false, value: scratch.subarray(0, length) };
      },
      cancel: async () => {},
      releaseLock: () => {},
    }),
    cancel: async () => {},
  };
}

function sourceFetch({ delay = false, calls = [] } = {}) {
  const weights = safetensors();
  let active = 0;
  let maximumActive = 0;
  const fetch = async (source) => {
    calls.push(source);
    active++;
    maximumActive = Math.max(maximumActive, active);
    if (delay) await new Promise((resolve) => setTimeout(resolve, 2));
    active--;
    if (source === GRAPH_SOURCE) {
      // Exercise the graph transport's text fallback. Weight and adapter
      // packages remain byte-exact binary transfers.
      return { ok: true, text: async () => GRAPH_TEXT };
    }
    if (source === WEIGHT_SOURCE) {
      return { ok: true, arrayBuffer: async () => exactArrayBuffer(weights) };
    }
    if (source === ADAPTER_SOURCE) {
      return {
        ok: true,
        arrayBuffer: async () => exactArrayBuffer(Uint8Array.of(1, 2, 3)),
      };
    }
    return { ok: false, status: 404, statusText: 'fixture not found' };
  };
  return {
    fetch,
    maximumActive: () => maximumActive,
  };
}

async function createRuntime(inference) {
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest());
  assert.equal(
    runtime.report?.status,
    pb.NativeStatus.NATIVE_STATUS_OK,
    runtime.report?.message,
  );
  assert.ok(runtime.runtimeId > 0n);
  return runtime.runtimeId;
}

function loadRequest(runtimeId) {
  return new pb.LoadModelRequest({
    runtimeId,
    graphPath: 'public/graph.json',
    weightPaths: ['public/model.safetensors'],
  });
}

test('dispatch-only factory validates the release module without a Planning profile', async (t) => {
  const factory = await loadModelControlWasmDispatchFactory(wasmUrl);
  assert.equal('graphPlanningProfile' in factory, false);
  assert.equal(Object.isFrozen(factory), true);
  const owner = factory.create();
  try {
    const info = (await new VxPlatformServiceClient(owner).getPlatformInfo(new pb.Empty()));
    assert.equal(info.profile, pb.BuildProfile.BUILD_PROFILE_INFERENCE);
    assert.equal(info.transport, pb.TransportProfile.TRANSPORT_PROFILE_REMOTE);
  } finally {
    (await owner.close());
  }
});

test('ordinary release bundle retains no TypeScript Planning registry', async (t) => {
  const release = await inspectBrowserReleaseBundle(
    ROOT,
    'inference',
    packageJson.version,
    { minify: true },
  );
  const forbidden = new Set([
    'ts/core/GraphPlanningParameterRegistry.ts',
    'ts/core/GraphPlanningProfile.ts',
    'ts/generated/operatorParamRegistry.ts',
  ]);
  const retained = [];
  for (const output of Object.values(release.metafile.outputs)) {
    for (const [input, contribution] of Object.entries(output.inputs ?? {})) {
      const normalized = input.replaceAll('\\', '/');
      if (contribution.bytesInOutput > 0 && [...forbidden].some((candidate) =>
        normalized === candidate || normalized.endsWith(`/${candidate}`))) {
        retained.push([normalized, contribution.bytesInOutput]);
      }
    }
  }
  assert.deepEqual(retained, []);
});

test('ordinary EngineHost lets the generated C decoder reject malformed path requests', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({ wasmUrl, fetch: transport.fetch });
  t.after(() => host.close());
  try {
    await assert.rejects(
      async () => {
        const call = await host.open({ path: '/volvoxai.v1.VxInferenceService/LoadModel',
          requestStream: false, responseStream: false });
        try {
          await call.send(Uint8Array.of(0x80));
          await call.halfClose();
          while (await call.recv() !== null) {}
        } finally { await call.close(); }
      },
      error => error.code === 3,
    );
    assert.deepEqual(calls, []);
  } finally {
    await host.close();
  }
});

test('ordinary EngineHost keeps one native owner and stages package paths privately', async (t) => {
  const calls = [];
  const transport = sourceFetch({ delay: true, calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  const models = await Promise.all([
    (await inference.loadModel(loadRequest(runtimeId))),
    (await inference.loadModel(loadRequest(runtimeId))),
  ]);
  try {
    for (const model of models) {
      assert.equal(
        model.report?.status,
        pb.NativeStatus.NATIVE_STATUS_OK,
        model.report?.message,
      );
      assert.ok(model.modelId > 0n);
    }
    assert.equal(
      transport.maximumActive(),
      1,
      'fetch through dispatch must share the native owner queue',
    );
    assert.deepEqual(calls, [
      GRAPH_SOURCE,
      WEIGHT_SOURCE,
      GRAPH_SOURCE,
      WEIGHT_SOURCE,
    ]);

    const adapter = await inference.publishAdapter(new pb.PublishAdapterRequest({
      modelId: models[0].modelId,
      adapterName: 'fixture-adapter',
      packagePath: ADAPTER_SOURCE,
      versionName: 'v1',
    }));
    assert.equal(
      adapter.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      adapter.report?.message,
    );
    assert.ok(adapter.adapterId > 0n);
    assert.equal(calls.at(-1), ADAPTER_SOURCE);
  } finally {
    for (const model of models) {
      await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
    }
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost snapshots reused streaming views into bounded host blocks', async (t) => {
  const graphBytes = new TextEncoder().encode(GRAPH_TEXT);
  const weightBytes = safetensors();
  const host = new EngineHost({
    wasmUrl,
    fetch: async (source) => {
      if (source === GRAPH_SOURCE) {
        return { ok: true, body: reusingStreamBody(graphBytes, 1) };
      }
      if (source === WEIGHT_SOURCE) {
        return { ok: true, body: reusingStreamBody(weightBytes, 3) };
      }
      return { ok: false, status: 404, statusText: 'fixture not found' };
    },
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  let modelId = 0n;
  try {
    const model = await inference.loadModel(loadRequest(runtimeId));
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message,
    );
    assert.ok(model.modelId > 0n);
    modelId = model.modelId;
  } finally {
    if (modelId !== 0n) {
      await inference.releaseModel(new pb.ModelRef({ modelId }));
    }
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
    await host.close();
  }
});

test('ordinary EngineHost returns a typed model response when package fetch fails', async (t) => {
  const host = new EngineHost({
    wasmUrl,
    fetch: async () => ({ ok: false, status: 404, statusText: 'not found' }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  try {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: 'missing.graph.json',
    }));
    assert.equal(model.modelId, 0n);
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_IO_ERROR,
    );
    assert.equal(model.report?.stage, pb.OperationStage.OPERATION_STAGE_MODEL_LOAD);
    assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_PACKAGE_FETCH_FAILED);
    assert.equal(
      model.report?.lineage?.runtimeId,
      runtimeId,
      'transport failure must retain C-validated public lineage',
    );
  } finally {
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost preserves graph-path validation across private staging', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  try {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: 'public/not-a-graph.txt',
      weightPaths: ['public/model.safetensors'],
    }));
    assert.equal(model.modelId, 0n);
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
    );
    assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_INVALID_GRAPH_PATH);
    assert.deepEqual(calls, [], 'native rejection must happen before source fetch');
  } finally {
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost rejects a stale runtime before model package fetch', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));

  const model = await inference.loadModel(loadRequest(runtimeId));
  assert.equal(model.modelId, 0n);
  assert.equal(
    model.report?.status,
    pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
  );
  assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_HANDLE_DISPOSED);
  assert.deepEqual(calls, [], 'a stale native handle must cause no network I/O');
});

test('ordinary EngineHost rejects invalid model metadata before package fetch', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  try {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: 'public/graph.json',
      weightPaths: ['public/model.safetensors'],
      bankResidency: [new pb.BankResidency({
        bank: 'decoder',
        slots: [2, 1],
      })],
    }));
    assert.equal(model.modelId, 0n);
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
    );
    assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_INVALID_BANK_RESIDENCY);
    assert.deepEqual(calls, [], 'invalid native metadata must cause no network I/O');
  } finally {
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('path preflight does not consume a capture for discarded fetch-ready evidence', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new pb.CreateRuntimeRequest({
    memoryCapture: new pb.MemoryCaptureOptions({
      protocol: 'volvoxai-memory-capture/v1',
      includeResourceInventory: true,
    }),
  }));
  assert.match(runtime.report?.memoryEvidence?.captureId ?? '', /-capture-1$/);

  const invalid = await inference.loadModel(new pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: 'public/not-a-graph.txt',
  }));
  assert.equal(invalid.report?.code, pb.OperationCode.OPERATION_CODE_INVALID_GRAPH_PATH);
  assert.equal(invalid.report?.memoryEvidence, undefined);
  assert.deepEqual(calls, []);

  const model = await inference.loadModel(loadRequest(runtime.runtimeId));
  try {
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message,
    );
    assert.match(
      model.report?.memoryEvidence?.captureId ?? '',
      /-capture-2$/,
      'fetch-ready preflight must not consume a capture sequence',
    );
    assert.deepEqual(calls, [GRAPH_SOURCE, WEIGHT_SOURCE]);
  } finally {
    await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
    await inference.releaseRuntime(new pb.RuntimeRef({
      runtimeId: runtime.runtimeId,
    }));
  }
});

test('ordinary EngineHost rejects stale and invalid adapter requests before fetch', async (t) => {
  const calls = [];
  const transport = sourceFetch({ calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  const model = await inference.loadModel(loadRequest(runtimeId));
  assert.equal(
    model.report?.status,
    pb.NativeStatus.NATIVE_STATUS_OK,
    model.report?.message,
  );
  calls.length = 0;
  try {
    const invalid = await inference.publishAdapter(new pb.PublishAdapterRequest({
      modelId: model.modelId,
      adapterName: '__base__',
      packagePath: ADAPTER_SOURCE,
      versionName: 'v1',
    }));
    assert.equal(invalid.adapterId, 0n);
    assert.equal(
      invalid.report?.status,
      pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT,
    );
    assert.equal(invalid.report?.code, pb.OperationCode.OPERATION_CODE_INVALID_ADAPTER_SOURCE);
    assert.deepEqual(calls, [], 'invalid native metadata must cause no network I/O');

    const unavailableSource = 'fixture/missing-adapter.bin';
    const unavailable = await inference.publishAdapter(new pb.PublishAdapterRequest({
      modelId: model.modelId,
      adapterName: 'fixture-adapter',
      packagePath: unavailableSource,
      versionName: 'v1',
    }));
    assert.equal(unavailable.report?.code, pb.OperationCode.OPERATION_CODE_PACKAGE_FETCH_FAILED);
    assert.equal(unavailable.report?.lineage?.runtimeId, runtimeId);
    assert.equal(unavailable.report?.lineage?.modelId, model.modelId);
    assert.deepEqual(calls, [unavailableSource]);
    calls.length = 0;

    await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
    const stale = await inference.publishAdapter(new pb.PublishAdapterRequest({
      modelId: model.modelId,
      adapterName: 'fixture-adapter',
      packagePath: ADAPTER_SOURCE,
      versionName: 'v1',
    }));
    assert.equal(stale.adapterId, 0n);
    assert.equal(
      stale.report?.status,
      pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED,
    );
    assert.equal(stale.report?.code, pb.OperationCode.OPERATION_CODE_HANDLE_DISPOSED);
    assert.deepEqual(calls, [], 'a stale native handle must cause no network I/O');
  } finally {
    await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost keeps descendants after an awaited load and parent release', async (t) => {
  const calls = [];
  const transport = sourceFetch({ delay: true, calls });
  const host = new EngineHost({
    wasmUrl,
    fetch: transport.fetch,
    resolveModelSource: () => ({
      graphUrl: GRAPH_SOURCE,
      weightSources: [WEIGHT_SOURCE],
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);

  const model = await inference.loadModel(loadRequest(runtimeId));
  const released = await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  try {
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message,
    );
    assert.ok(model.modelId > 0n);
    assert.equal(
      released.status,
      pb.NativeStatus.NATIVE_STATUS_OK,
      released.message,
    );
    assert.deepEqual(calls, [GRAPH_SOURCE, WEIGHT_SOURCE]);
  } finally {
    await inference.releaseModel(new pb.ModelRef({ modelId: model.modelId }));
  }
});

test('ordinary EngineHost enforces its package transport budget before reading a declared body', async (t) => {
  let bodyRead = false;
  const host = new EngineHost({
    wasmUrl,
    maxPackageBytes: 4,
    fetch: async () => ({
      ok: true,
      headers: { get: (name) => name.toLowerCase() === 'content-length' ? '5' : null },
      arrayBuffer: async () => {
        bodyRead = true;
        return new ArrayBuffer(5);
      },
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  try {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: 'graph.json',
    }));
    assert.equal(model.modelId, 0n);
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
    );
    assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE);
    assert.equal(bodyRead, false);
  } finally {
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost cancels an undeclared streaming body at its cumulative byte limit', async (t) => {
  let reads = 0;
  let cancelled = false;
  let arrayBufferRead = false;
  const body = {
    getReader: () => ({
      read: async () => {
        reads++;
        if (reads === 1) return { done: false, value: Uint8Array.of(1, 2, 3) };
        return { done: false, value: Uint8Array.of(4, 5) };
      },
      cancel: async () => {
        cancelled = true;
      },
      releaseLock: () => {},
    }),
    cancel: async () => {
      cancelled = true;
    },
  };
  const host = new EngineHost({
    wasmUrl,
    maxPackageBytes: 4,
    fetch: async () => ({
      ok: true,
      body,
      arrayBuffer: async () => {
        arrayBufferRead = true;
        return new ArrayBuffer(0);
      },
    }),
  });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  try {
    const model = await inference.loadModel(new pb.LoadModelRequest({
      runtimeId,
      graphPath: 'graph.json',
    }));
    assert.equal(model.modelId, 0n);
    assert.equal(
      model.report?.status,
      pb.NativeStatus.NATIVE_STATUS_OUT_OF_MEMORY,
    );
    assert.equal(model.report?.code, pb.OperationCode.OPERATION_CODE_PACKAGE_TOO_LARGE);
    assert.equal(reads, 2);
    assert.equal(cancelled, true);
    assert.equal(arrayBufferRead, false);
  } finally {
    await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  }
});

test('ordinary EngineHost does not allow callers to raise the 64-MiB profile ceiling', () => {
  assert.throws(
    () => new EngineHost({ maxPackageBytes: 64 * 1024 * 1024 + 1 }),
    /no larger than 64 MiB/,
  );
});

test('ordinary EngineHost closes its persistent owner after queued work', async (t) => {
  const host = new EngineHost({ wasmUrl });
  t.after(() => host.close());
  const inference = new VxInferenceServiceClient(reportTransport(host));
  const runtimeId = await createRuntime(inference);
  await inference.releaseRuntime(new pb.RuntimeRef({ runtimeId }));
  await host.close();
  await host.close();
  await assert.rejects(
    inference.createRuntime(new pb.CreateRuntimeRequest()),
    /EngineHost is closed/,
  );
});
