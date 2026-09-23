import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { EngineHost } from '../ts/host/EngineHost.js';
import { FullEngineHost } from '../ts/full.js';
import { loadWasmReleaseModule } from '../ts/core/WasmReleaseModule.js';
import { ModelControlWasmDispatchFactory } from '../ts/core/ModelControlWasm.js';
import { VxInferenceServiceClient, VxPlatformServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
const WASM = new URL('../dist/0.6.0/volvoxai.lite.wasm', import.meta.url);
const FULL = new URL('../dist/0.6.0/volvoxai.wasm', import.meta.url);

test('a compiled module is cached while each owner has isolated memory', async () => {
  const module = await loadWasmReleaseModule(WASM);
  assert.equal(await loadWasmReleaseModule(WASM), module);
  const factory = new ModelControlWasmDispatchFactory(module);
  const first = factory.instantiate(), second = factory.instantiate();
  assert.notEqual(first.exports.memory, second.exports.memory);
});

test('a restarted owner rejects old handles even after allocating new ones', async () => {
  let stale = 0n;
  for (let restart = 0; restart < 12; restart++) {
    const host = new EngineHost({ wasmUrl: WASM });
    const api = new VxInferenceServiceClient(reportTransport(host));
    try {
      const created = await api.createRuntime(new pb.CreateRuntimeRequest());
      assert.equal(created.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
      assert.notEqual(created.runtimeId, stale);
      if (stale) {
        const rejected = await api.loadModel(new pb.LoadModelRequest({ runtimeId: stale, graphPath: 'must-not-be-fetched.json' }));
        assert.equal(rejected.report.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
      }
      stale = created.runtimeId;
      const released = await api.releaseRuntime(new pb.RuntimeRef({ runtimeId: created.runtimeId }));
      assert.equal(released.status, pb.NativeStatus.NATIVE_STATUS_OK);
    } finally { await host.close(); }
  }
});

test('the full C owner selects its profile without a TypeScript provider', async () => {
  const host = new FullEngineHost({ wasmUrl: FULL });
  try {
    const platform = await new VxPlatformServiceClient(reportTransport(host)).getPlatformInfo(new pb.Empty());
    assert.equal(platform.profile, pb.BuildProfile.BUILD_PROFILE_FULL);
    const api = new VxInferenceServiceClient(reportTransport(host));
    const runtime = await api.createRuntime(new pb.CreateRuntimeRequest());
    assert.equal(runtime.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
    const released = await api.releaseRuntime(new pb.RuntimeRef({ runtimeId: runtime.runtimeId }));
    assert.equal(released.status, pb.NativeStatus.NATIVE_STATUS_OK);
  } finally { await host.close(); }
});

test('each host rejects the other profile companion before dispatch', async () => {
  for (const [Host, wasmUrl, profile] of [[EngineHost, FULL, 'inference'], [FullEngineHost, WASM, 'full']]) {
    const host = new Host({ wasmUrl });
    try {
      const api = new VxPlatformServiceClient(reportTransport(host));
      await assert.rejects((api.getPlatformInfo(new pb.Empty())),
        new RegExp(`imports do not match the ${profile} profile`));
    } finally { await host.close(); }
  }
});

test('failed module initialization is stable and close retires the host', async () => {
  const host = new FullEngineHost({ wasmUrl: '/no-such-volvoxai-module.wasm' });
  const api = new VxInferenceServiceClient(reportTransport(host));
  await assert.rejects((api.createRuntime(new pb.CreateRuntimeRequest())), /ENOENT|no such file/i);
  await host.close();
  await assert.rejects((api.createRuntime(new pb.CreateRuntimeRequest())), /closed/i);
});

test('a retained closed owner releases its WASM memory', async () => {
  const { execFile } = await import('node:child_process');
  const { promisify } = await import('node:util');
  await promisify(execFile)(process.execPath, ['--expose-gc', '--import', 'tsx', '--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { setImmediate } from 'node:timers/promises';
    import { readFileSync } from 'node:fs';
    import { ModelControlWasmDispatchFactory } from './ts/core/ModelControlWasm.ts';
    import { VxPlatformServiceClient } from './runtime/generated/typescript/inference/volvoxai_ffi.ts';
    import { Empty } from './runtime/generated/typescript/inference/volvoxai_lite.ts';
    let memory;
    class ObservedFactory extends ModelControlWasmDispatchFactory {
      instantiate(wakeup) {
        const instance = super.instantiate(wakeup);
        memory = new WeakRef(instance.exports.memory);
        return instance;
      }
    }
    const owner = new ObservedFactory(new WebAssembly.Module(readFileSync(${JSON.stringify(WASM.pathname)}))).create();
    await new VxPlatformServiceClient(owner).getPlatformInfo(new Empty());
    await owner.close();
    for (let turn = 0; turn < 20; turn++) {
      await setImmediate();
      globalThis.gc();
      if (memory.deref() === undefined) break;
    }
    assert.equal(memory.deref(), undefined, 'closed owner must not retain the upstream Host module');
    assert.equal(owner.closed, true);
  `]);
});
