/**
 * The GPU device bridge, checked against the module that imports it.
 *
 * A device is not needed to prove the parts that matter here: that the full
 * profile asks for exactly the declared bridge, that a host without WebGPU
 * refuses in a way the engine reads as absence rather than failure, and that
 * a descriptor the engine writes is laid out the way the host reads it.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import * as catalog from '../ts/generated/shaderCatalog.js';
import { REFUSING_GPU_BRIDGE } from '../ts/backends/WebGPUHostBridge.js';
import { WASM_PROFILE_IMPORTS } from '../tools/generated/wasmInternalAbi.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = JSON.parse(
  fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'),
);
const FULL_WASM = path.join(ROOT, 'dist', packageJson.version, 'volvoxai.wasm');
const INFERENCE_WASM = path.join(ROOT, 'dist', packageJson.version, 'volvoxai.lite.wasm');

async function completionFenceFixture({validation, pendingScopes = false} = {}) {
  const {createWebGPUHostBridge} = await import('../ts/backends/WebGPUHostBridge.js');
  let finish, reject, lose;
  const queued = new Promise((resolve, fail) => { finish = resolve; reject = fail; });
  const counts = {submits: 0, buffers: 0, wakeups: 0};
  const diagnostics = [];
  const device = {
    lost: new Promise(resolve => { lose = resolve; }),
    queue: {submit() { counts.submits++; }, onSubmittedWorkDone: () => queued},
    pushErrorScope() {},
    popErrorScope: () => pendingScopes ? new Promise(() => {}) : Promise.resolve(validation ?? null),
    createBuffer() { counts.buffers++; assert.fail('completion fences must allocate no GPU storage'); },
    createCommandEncoder: () => ({beginComputePass: () => ({end() {}}), finish: () => ({})}),
  };
  const bridge = await createWebGPUHostBridge({device, catalog,
    onDiagnostic: message => diagnostics.push(message)});
  const memory = new WebAssembly.Memory({initial: 1});
  new Uint8Array(memory.buffer).fill(0x5a);
  bridge.attach(memory);
  bridge.setWakeup(() => { counts.wakeups++; });
  return {bridge, api: bridge.imports, counts, memory, diagnostics, finish, reject, lose};
}

test('completion-only tickets wait without allocating, mapping or copying output storage', async () => {
  const f = await completionFenceFixture();
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_end(), 0);
  const ticket = f.api.vx_gpu_snapshot(0, 0, 0, 0);
  assert.ok(ticket > 0);
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -2);
  assert.equal(f.api.vx_gpu_snapshot(16, 0, 0, 0), -1, 'only the explicit null command is a fence');
  assert.equal(f.api.vx_gpu_snapshot(0, 4, 0, 0), -1);
  assert.equal(f.counts.submits, 1, 'fencing does not add an empty submission');
  assert.equal(f.counts.buffers, 0);
  f.finish();
  await f.bridge.waitForCompletion();
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), 0);
  assert.equal(f.api.vx_gpu_readback(ticket, 16, 0), -1);
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 4), -1);
  assert.ok(new Uint8Array(f.memory.buffer).every(value => value === 0x5a));
  assert.ok(f.counts.wakeups > 0);
  f.api.vx_gpu_readback_release(ticket);
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -1);
  f.bridge.close();
  await f.bridge.waitForCompletion();
});

test('completion-only tickets propagate submission validation and queue failures', async () => {
  for (const failure of ['validation', 'queue']) {
    const f = await completionFenceFixture({validation: failure === 'validation' ? {message: 'bad binding'} : null});
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_end(), 0);
    const ticket = f.api.vx_gpu_snapshot(0, 0, 0, 0);
    assert.ok(ticket > 0);
    if (failure === 'queue') f.reject(new Error('queue failed'));
    else f.finish();
    await f.bridge.waitForCompletion();
    assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -1);
    assert.ok(f.diagnostics.length > 0);
    assert.equal(f.counts.buffers, 0);
    f.api.vx_gpu_readback_release(ticket);
    f.bridge.close();
    await f.bridge.waitForCompletion();
  }
});

test('released completion-only tickets remain drained by owner close', async () => {
  const f = await completionFenceFixture();
  // No live submission record can hide a missing retired-fence retain here.
  const ticket = f.api.vx_gpu_snapshot(0, 0, 0, 0);
  f.api.vx_gpu_readback_release(ticket);
  f.api.vx_gpu_readback_release(ticket);
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -1);
  f.bridge.close();
  let drained = false;
  const closing = f.bridge.waitForCompletion().then(() => { drained = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(drained, false);
  f.finish();
  await closing;
  assert.equal(drained, true);
  assert.equal(f.counts.buffers, 0);
});

test('completion-only tickets drain on device loss with unresolved queue and scopes', async () => {
  const f = await completionFenceFixture({pendingScopes: true});
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_end(), 0);
  const ticket = f.api.vx_gpu_snapshot(0, 0, 0, 0);
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -2);
  f.lose({reason: 'unknown', message: 'test device loss'});
  await f.bridge.waitForCompletion();
  assert.equal(f.api.vx_gpu_readback(ticket, 0, 0), -3);
  f.api.vx_gpu_readback_release(ticket);
  f.bridge.close();
  await f.bridge.waitForCompletion();
  assert.equal(f.counts.buffers, 0);
});

test('completed submissions and buffer retirements have bounded retention on a live device', async () => {
  await promisify(execFile)(process.execPath, ['--expose-gc', '--import', 'tsx',
    'tests/contracts/webgpu_bridge_retention.mjs'], { cwd: ROOT, timeout: 60000 });
});

test('device loss drains submissions and retirements even when queue and error scopes never settle', async (t) => {
  const { createWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  const usage = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  Object.defineProperty(globalThis, 'GPUBufferUsage', { configurable: true,
    value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
  t.after(() => {
    if (usage) Object.defineProperty(globalThis, 'GPUBufferUsage', usage);
    else delete globalThis.GPUBufferUsage;
  });
  let lose, destroyed = 0, wakeups = 0;
  const pending = new Promise(() => {});
  const device = {
    lost: new Promise(resolve => { lose = resolve; }),
    queue: { submit() {}, writeBuffer() {}, onSubmittedWorkDone: () => pending },
    pushErrorScope() {}, popErrorScope: () => pending,
    createBuffer: () => ({ destroy() { destroyed++; } }),
    createCommandEncoder: () => ({ beginComputePass: () => ({ end() {} }), finish: () => ({}) }),
  };
  const bridge = await createWebGPUHostBridge({ device, catalog });
  bridge.attach(new WebAssembly.Memory({ initial: 1 }));
  bridge.setWakeup(() => { wakeups++; });
  bridge.imports.vx_gpu_begin();
  assert.equal(bridge.imports.vx_gpu_ensure(64, 16, 0), 1);
  assert.equal(bridge.imports.vx_gpu_end(), 0);
  bridge.imports.vx_gpu_release(64);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(destroyed, 0, 'the queued buffer is retained until completion or loss');
  lose({ reason: 'unknown', message: 'test device loss' });
  await bridge.waitForCompletion();
  assert.equal(destroyed, 1);
  assert.ok(wakeups > 0);
  assert.equal(bridge.imports.vx_gpu_ensure(64, 16, 0), 0);
  bridge.close();
  await bridge.waitForCompletion();
  assert.equal(destroyed, 1, 'close cannot destroy a retired buffer twice');
});

const BRIDGE_NAMES = Object.freeze([
  'vx_gpu_available', 'vx_gpu_limits', 'vx_gpu_ensure', 'vx_gpu_release',
  'vx_gpu_invalidate', 'vx_gpu_memory_start', 'vx_gpu_memory_stop', 'vx_gpu_begin', 'vx_gpu_begin_activity', 'vx_gpu_begin_trace', 'vx_gpu_trace_node_begin', 'vx_gpu_trace_node_end', 'vx_gpu_trace_program_begin', 'vx_gpu_trace_program_end', 'vx_gpu_debug_label', 'vx_gpu_trace_read', 'vx_gpu_trace_release', 'vx_gpu_await_read', 'vx_gpu_await_release', 'vx_gpu_encode', 'vx_gpu_end',
  'vx_gpu_snapshot', 'vx_gpu_readback', 'vx_gpu_readback_release',
]);

function moduleImports(file) {
  return WebAssembly.Module.imports(new WebAssembly.Module(fs.readFileSync(file)))
    .map(({ module, name, kind }) => `${module}.${name}:${kind}`)
    .sort();
}

test('the full profile imports exactly the declared device bridge', () => {
  const imports = moduleImports(FULL_WASM);
  assert.deepEqual(imports, [...WASM_PROFILE_IMPORTS.full]);
  const bridge = imports.filter((entry) => entry.startsWith('gpu.'));
  assert.deepEqual(bridge, BRIDGE_NAMES.map((name) => `gpu.${name}:function`).sort());
});

test('the inference profile imports no device bridge and instantiates without one', async () => {
  const imports = moduleImports(INFERENCE_WASM);
  assert.deepEqual(imports, [...WASM_PROFILE_IMPORTS.inference]);
  assert.deepEqual(imports.filter((entry) => entry.startsWith('gpu.')), []);
  const { wasmReleaseModuleImports } = await import('../ts/core/WasmReleaseModule.js');
  const hostImports = wasmReleaseModuleImports();
  assert.deepEqual(Object.keys(hostImports).sort(), ['host', 'synurang']);
  const module = new WebAssembly.Module(fs.readFileSync(INFERENCE_WASM));
  const instance = new WebAssembly.Instance(module, hostImports);
  assert.ok(instance.exports.memory instanceof WebAssembly.Memory);
  assert.ok(!Object.keys(instance.exports).some(name => name.includes('gpu')));
});

test('a host without a device leaves the engine on its other backends', async () => {
  const { wasmReleaseModuleImports } = await import('../ts/core/WasmReleaseModule.js');
  const imports = wasmReleaseModuleImports(REFUSING_GPU_BRIDGE);
  assert.deepEqual(Object.keys(imports).sort(), ['gpu', 'host', 'synurang']);
  for (const name of BRIDGE_NAMES) {
    assert.equal(typeof imports.gpu[name], 'function', `${name} must be supplied`);
  }
  /* Absence, not failure: the backend reads this and reports UNAVAILABLE. */
  assert.equal(imports.gpu.vx_gpu_available(), 0);
  assert.equal(imports.gpu.vx_gpu_ensure(1, 1, 0), 0);
  assert.equal(imports.gpu.vx_gpu_encode(0), -1);
});

test('the full module instantiates against the refusing bridge', async () => {
  const { wasmReleaseModuleImports } = await import('../ts/core/WasmReleaseModule.js');
  const module = new WebAssembly.Module(fs.readFileSync(FULL_WASM));
  const instance = new WebAssembly.Instance(module, wasmReleaseModuleImports(REFUSING_GPU_BRIDGE));
  assert.ok(instance.exports.memory instanceof WebAssembly.Memory);
});

test('the host reads a descriptor exactly where the engine writes it', () => {
  /* One VxGpuDispatch with two variants, three bindings and 12 params, laid
   * out by hand against gpu_bridge.h, then read with the same arithmetic
   * WebGPUHostBridge uses. A drift in either side fails here rather than on a
   * device. */
  const HEADER = 5, VARIANT = 4, BINDING = 5;
  const variants = [[69, 7, 5, 3], [70, 1, 1, 1]];
  const bindings = [[0, 0x1000, 0, 64, 0], [1, 0x2000, 0, 32, 1],
    [2, 0x3000, 16, 8, 0]];
  const params = Uint32Array.from({ length: 12 }, (_, index) => index + 100);
  const words = HEADER + variants.length * VARIANT + bindings.length * BINDING
    + params.length;
  const view = new Uint32Array(words);
  view[0] = variants.length;
  view[1] = bindings.length;
  view[2] = params.byteLength;
  view[3] = 42;
  let at = HEADER;
  for (const variant of variants) { view.set(variant, at); at += VARIANT; }
  for (const binding of bindings) { view.set(binding, at); at += BINDING; }
  view.set(params, at);

  const variantBase = HEADER;
  const bindingBase = variantBase + view[0] * VARIANT;
  const paramsBase = bindingBase + view[1] * BINDING;
  assert.equal(view[3], 42, 'node index survives the header');
  assert.deepEqual([...view.subarray(variantBase, variantBase + VARIANT)], variants[0]);
  assert.deepEqual(
    [...view.subarray(bindingBase + BINDING, bindingBase + 2 * BINDING)], bindings[1],
  );
  assert.deepEqual([...view.subarray(paramsBase, paramsBase + params.length)], [...params]);
  assert.equal(paramsBase + params.length, words, 'the tails exactly fill the wire');
});

test('the device rejects mismatched ABI/catalogue pairs and a second memory owner', async () => {
  const { createWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  const { GPU_BRIDGE_ABI_HASH } = await import('../ts/generated/gpuBridge.js');
  const { SHADER_CATALOG_HASH } = await import('../ts/generated/shaderCatalog.js');
  let lose;
  const diagnostics = [];
  let insideImport = false;
  const device = { lost: new Promise((resolve) => { lose = resolve; }) };
  const bridge = await createWebGPUHostBridge({ device, catalog, onDiagnostic: (message) => { assert.equal(insideImport, false, 'user callbacks cannot reenter C'); diagnostics.push(message); } });
  const memory = new WebAssembly.Memory({ initial: 1 });
  bridge.attach(memory);
  const hash = new TextEncoder().encode(SHADER_CATALOG_HASH);
  new Uint8Array(memory.buffer, 128, 64).set(new TextEncoder().encode(GPU_BRIDGE_ABI_HASH));
  new Uint8Array(memory.buffer, 64, 64).set(hash);
  insideImport = true;
  assert.equal(bridge.imports.vx_gpu_available(128, 64), 1);
  assert.equal(bridge.imports.vx_gpu_available(64, 64), 0);
  new Uint8Array(memory.buffer)[64] ^= 1;
  assert.equal(bridge.imports.vx_gpu_available(128, 64), 0);
  insideImport = false;
  assert.equal(diagnostics.length, 0, 'diagnostics wait until the import returns');
  await Promise.resolve();
  assert.equal(diagnostics.length, 1, 'mismatch is reported once');
  assert.throws(() => bridge.attach(new WebAssembly.Memory({ initial: 1 })), /already belongs/);
  new Uint8Array(memory.buffer, 64, 64).set(hash);
  lose({ message: 'test device loss' });
  await Promise.resolve();
  assert.equal(bridge.imports.vx_gpu_available(128, 64), 0);
});

test('an acquired device closes exactly once after its queued work finishes', async () => {
  const {acquireWebGPUHostBridge} = await import('../ts/backends/WebGPUHostBridge.js');
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  let finish, finishDestroy, requested, destroyed = 0;
  const queued = new Promise(resolve => {finish = resolve;});
  const device = {
    lost: new Promise(resolve => {finishDestroy = resolve;}),
    queue: {submit() {}, onSubmittedWorkDone: () => queued},
    pushErrorScope() {}, popErrorScope: async () => null,
    createCommandEncoder: () => ({beginComputePass: () => ({end() {}}), finish: () => ({})}),
    destroy() {destroyed++;},
  };
  Object.defineProperty(globalThis, 'navigator', {configurable:true, value:{gpu:{
    requestAdapter: async () => ({features: new Set(),
      limits: {maxBufferSize: 1024*1024*1024, maxStorageBufferBindingSize: 512*1024*1024},
      requestDevice: async descriptor => {requested=descriptor;return device;}}),
  }}});
  try {
    const bridge = await acquireWebGPUHostBridge(catalog);
    assert.deepEqual(requested.requiredLimits,
      {maxBufferSize:1024*1024*1024,maxStorageBufferBindingSize:512*1024*1024});
    bridge.attach(new WebAssembly.Memory({initial:1}));
    bridge.imports.vx_gpu_begin(); bridge.imports.vx_gpu_end();
    bridge.close(); bridge.close();
    await Promise.resolve();
    assert.equal(destroyed,0);
    finish();
    let closed = false;
    const closing = bridge.waitForCompletion().then(() => {closed = true;});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(destroyed,1);
    assert.equal(closed,false,'close waits for device destruction to settle');
    finishDestroy({reason:'destroyed',message:'closed'}); await closing;
    bridge.close(); await bridge.waitForCompletion();
    assert.equal(destroyed,1);
  } finally {
    if (original) Object.defineProperty(globalThis,'navigator',original);
    else delete globalThis.navigator;
  }
});

function installTestGpu(t, gpu) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu } });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, 'navigator', original);
    else delete globalThis.navigator;
  });
}

test('the deferred bridge acquires once on preparation and refuses again after close', async (t) => {
  const { createDeferredWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  const { GPU_BRIDGE_ABI_HASH } = await import('../ts/generated/gpuBridge.js');
  const { SHADER_CATALOG_HASH } = await import('../ts/generated/shaderCatalog.js');
  let adapters = 0, devices = 0, destroyed = 0, lose;
  const device = {
    features: new Set(),
    lost: new Promise(resolve => { lose = resolve; }),
    destroy() { destroyed++; lose({ reason: 'destroyed', message: 'closed' }); },
  };
  installTestGpu(t, { requestAdapter: async () => {
    adapters++;
    return { features: new Set(), requestDevice: async () => { devices++; return device; } };
  } });
  const bridge = createDeferredWebGPUHostBridge(catalog);
  const imports = bridge.imports;
  const memory = new WebAssembly.Memory({ initial: 1 });
  bridge.attach(memory);
  new Uint8Array(memory.buffer, 64, 64).set(new TextEncoder().encode(SHADER_CATALOG_HASH));
  new Uint8Array(memory.buffer, 128, 64).set(new TextEncoder().encode(GPU_BRIDGE_ABI_HASH));
  assert.equal(imports.vx_gpu_available(128, 64), 0);
  assert.equal(adapters, 0, 'instantiating and attaching must not acquire a device');
  await bridge.prepareTracing();
  assert.equal(adapters, 0, 'a trace started before compilation must not acquire a device');
  assert.throws(() => bridge.attach(new WebAssembly.Memory({ initial: 1 })), /another owner/);
  try {
    await Promise.all([bridge.prepare(), bridge.prepare(), bridge.prepare()]);
    assert.equal(adapters, 1);
    assert.equal(devices, 1);
    assert.equal(bridge.imports, imports, 'WASM imports remain stable after preparation');
    assert.equal(imports.vx_gpu_available(128, 64), 1);
  } finally {
    bridge.close();
    assert.equal(imports.vx_gpu_available(128, 64), 0, 'close refuses immediately while draining');
    await bridge.waitForCompletion();
  }
  assert.equal(imports.vx_gpu_available(128, 64), 0);
  await assert.rejects(bridge.prepare(), /closed/);
  bridge.close();
  await bridge.waitForCompletion();
  assert.equal(destroyed, 1);
});

test('closing a deferred bridge before preparation never requests an adapter', async (t) => {
  const { createDeferredWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  installTestGpu(t, { requestAdapter: () => assert.fail('closed bridge acquired an adapter') });
  const bridge = createDeferredWebGPUHostBridge(catalog);
  bridge.attach(new WebAssembly.Memory({ initial: 1 }));
  bridge.close();
  await bridge.waitForCompletion();
  await assert.rejects(bridge.prepare(), /closed/);
});

test('closing a deferred bridge drains a device that arrives during acquisition', async (t) => {
  const { createDeferredWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  let deliverDevice, lose, destroyed = 0;
  const device = {
    lost: new Promise(resolve => { lose = resolve; }),
    destroy() { destroyed++; },
  };
  installTestGpu(t, { requestAdapter: async () => ({
    features: new Set(),
    requestDevice: () => new Promise(resolve => { deliverDevice = resolve; }),
  }) });
  const bridge = createDeferredWebGPUHostBridge(catalog);
  bridge.attach(new WebAssembly.Memory({ initial: 1 }));
  const preparation = bridge.prepare();
  await new Promise(resolve => setImmediate(resolve));
  bridge.close();
  let closed = false;
  const closing = bridge.waitForCompletion().then(() => { closed = true; });
  deliverDevice(device);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(closed, false, 'close waits for the arriving device to be destroyed');
  assert.equal(destroyed, 1);
  lose({ reason: 'destroyed', message: 'closed' });
  await Promise.all([preparation, closing]);
  assert.equal(bridge.imports.vx_gpu_available(0, 0), 0);
  assert.equal(destroyed, 1);
});

test('large uploads preserve exact bytes without exhausting pending staging storage', async () => {
  const { createWebGPUHostBridge } = await import('../ts/backends/WebGPUHostBridge.js');
  const usage = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  Object.defineProperty(globalThis, 'GPUBufferUsage', { configurable: true,
    value: { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 } });
  const buffers = [], writes = [], submissions = [];
  let pendingBytes = 0, peakBytes = 0;
  const device = {
    lost: new Promise(() => {}),
    createBuffer({ size }) {
      const buffer = { data: new Uint8Array(size), destroy() {} };
      buffers.push(buffer); return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, bytes) {
        pendingBytes += bytes.byteLength;
        assert.ok(pendingBytes <= 32 * 1024 * 1024, 'host-visible staging heap exhausted');
        peakBytes = Math.max(peakBytes, pendingBytes);
        writes.push({ buffer, offset, data: Uint8Array.from(bytes) });
      },
      submit(commands) {
        submissions.push(commands.length);
        for (const { buffer, offset, data } of writes.splice(0)) buffer.data.set(data, offset);
        pendingBytes = 0;
      },
      onSubmittedWorkDone: async () => {},
    },
    pushErrorScope() {}, popErrorScope: async () => null,
    createCommandEncoder: () => ({ beginComputePass: () => ({ end() {} }), finish: () => ({}) }),
  };
  const bridge = await createWebGPUHostBridge({ device, catalog });
  const size = 35 * 1024 * 1024 + 3, pointer = 64;
  const memory = new WebAssembly.Memory({ initial: Math.ceil((size + pointer) / 65536) });
  bridge.attach(memory);
  const source = new Uint8Array(memory.buffer, pointer, size);
  for (let i = 0; i < size; i++) source[i] = (i * 31 + (i >>> 20)) & 255;
  const expected = Uint8Array.from(source);
  try {
    bridge.imports.vx_gpu_begin();
    assert.equal(bridge.imports.vx_gpu_ensure(pointer, size, 1), 1);
    source.fill(0); // Enqueued writes must retain the original call's bytes.
    assert.equal(bridge.imports.vx_gpu_ensure(pointer, size, 1), 1);
    assert.equal(buffers.length, 1, 'immutable residency reuses the existing buffer');
    assert.equal(bridge.imports.vx_gpu_end(), 0);
    await bridge.waitForCompletion();
    assert.deepEqual(buffers[0].data.subarray(0, size), expected);
    assert.equal(buffers[0].data[size], 0, 'the partial word is zero padded');
    assert.ok(submissions.filter(count => count === 0).length >= 2);
    assert.equal(submissions.at(-1), 1, 'compute remains in its original submission');
    assert.ok(peakBytes <= 16 * 1024 * 1024);
    // Explicit host mutation revokes immutable residency, including a weight
    // restored by a trainer or model transaction before a subsequent dispatch.
    bridge.imports.vx_gpu_begin();
    bridge.imports.vx_gpu_invalidate(pointer);
    assert.equal(bridge.imports.vx_gpu_ensure(pointer, size, 1), 1);
    assert.equal(bridge.imports.vx_gpu_end(), 0);
    await bridge.waitForCompletion();
    assert.ok(buffers[0].data.every(value => value === 0), 'invalidated weight uploads current bytes');
    bridge.imports.vx_gpu_begin();
    source.fill(0x6d, 0, 8);
    assert.equal(bridge.imports.vx_gpu_ensure(pointer, 8, 1), 1);
    assert.equal(bridge.imports.vx_gpu_end(), 0);
    await bridge.waitForCompletion();
    assert.equal(buffers.length, 2, 'changed extent obtains a new residency identity');
    assert.deepEqual(buffers[1].data, new Uint8Array(8).fill(0x6d));
  } finally {
    bridge.close(); await bridge.waitForCompletion();
    if (usage) Object.defineProperty(globalThis, 'GPUBufferUsage', usage);
    else delete globalThis.GPUBufferUsage;
  }
});


test('complete output reservations skip uploads without becoming valid activations', async () => {
  const {createWebGPUHostBridge} = await import('../ts/backends/WebGPUHostBridge.js');
  const {GPU_RESIDENCY_WEIGHT: weight, GPU_RESIDENCY_OVERWRITE: overwrite} =
    await import('../ts/generated/gpuBridge.js');
  const usage = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  Object.defineProperty(globalThis, 'GPUBufferUsage', {configurable: true,
    value: {STORAGE: 128, COPY_DST: 8, COPY_SRC: 4}});
  const writes = [], buffers = [];
  const device = {
    lost: new Promise(() => {}),
    createBuffer({size}) {
      const buffer = {data: new Uint8Array(size).fill(0xcd), destroy() {}};
      buffers.push(buffer); return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        writes.push(data.byteLength); buffer.data.set(data, offset);
      },
      submit() {}, onSubmittedWorkDone: async () => {},
    },
    pushErrorScope() {}, popErrorScope: async () => null,
    createCommandEncoder: () => ({beginComputePass: () => ({end() {}}), finish: () => ({})}),
  };
  const bridge = await createWebGPUHostBridge({device, catalog});
  const memory = new WebAssembly.Memory({initial: 1});
  bridge.attach(memory);
  const api = bridge.imports, pointer = 64;
  const source = new Uint8Array(memory.buffer, pointer, 32);
  try {
    source.fill(0x37);
    api.vx_gpu_begin();
    assert.equal(api.vx_gpu_ensure(pointer, 16, overwrite), 1);
    assert.deepEqual(writes, [], 'reserve full output without reading host memory');
    assert.ok(buffers[0].data.every(x => x === 0xcd));
    // A refused plan has not written anything. A subsequent read or partial
    // write must upload host bytes, even with the same address and extent.
    assert.equal(api.vx_gpu_ensure(pointer, 16, 0), 1);
    assert.deepEqual(writes, [16]);
    assert.ok(buffers[0].data.every(x => x === 0x37));
    source.fill(0x59);
    api.vx_gpu_invalidate(pointer);
    assert.equal(api.vx_gpu_ensure(pointer, 16, overwrite), 1);
    assert.deepEqual(writes, [16]);
    assert.equal(api.vx_gpu_ensure(pointer, 16, 0), 1);
    assert.ok(buffers[0].data.every(x => x === 0x59));
    assert.equal(api.vx_gpu_ensure(pointer, 32, overwrite), 1);
    assert.equal(buffers.length, 2, 'extent change retires old storage');
    assert.deepEqual(writes, [16, 16]);
    assert.equal(api.vx_gpu_ensure(pointer, 32, 0), 1);
    assert.ok(buffers[1].data.every(x => x === 0x59));
    api.vx_gpu_release(pointer);
    assert.equal(api.vx_gpu_ensure(pointer, 16, weight), 1);
    assert.equal(api.vx_gpu_ensure(pointer, 16, overwrite), 0, 'immutable weight is not a discardable output');
    assert.equal(api.vx_gpu_ensure(pointer + 64, 16, weight | overwrite), 0);
    assert.equal(api.vx_gpu_ensure(pointer + 64, 15, overwrite), 0, 'partial words need ordinary residency');
    assert.equal(api.vx_gpu_ensure(pointer + 64, 16, 4), 0, 'unknown flags fail closed');
    assert.equal(api.vx_gpu_end(), 0);
    await bridge.waitForCompletion();
  } finally {
    bridge.close(); await bridge.waitForCompletion();
    if (usage) Object.defineProperty(globalThis, 'GPUBufferUsage', usage);
    else delete globalThis.GPUBufferUsage;
  }
});

async function uniformFixture(t, storage = false) {
  const {createWebGPUHostBridge} = await import('../ts/backends/WebGPUHostBridge.js');
  const previous = ['GPUBufferUsage', 'GPUShaderStage', 'GPUMapMode'].map(name =>
    [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  Object.defineProperty(globalThis, 'GPUBufferUsage', {configurable: true,
    value: {STORAGE: 128, COPY_DST: 8, COPY_SRC: 4, UNIFORM: 64, MAP_READ: 1}});
  Object.defineProperty(globalThis, 'GPUShaderStage', {configurable: true, value: {COMPUTE: 1}});
  Object.defineProperty(globalThis, 'GPUMapMode', {configurable: true, value: {READ: 1}});
  const buffers = [], writes = [], submissions = [], waits = [], results = [], storageResults = [], events = [];
  let lose, validation = null;
  const device = {
    features: new Set(),
    lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope() {}, popErrorScope: async () => validation,
    createBuffer(descriptor) {
      const buffer = {size: descriptor.size, data: new Uint8Array(descriptor.size),
        label: descriptor.label, destroyed: 0, destroy() { this.destroyed++; },
        mapAsync: async () => {}, getMappedRange() { return this.data.buffer; }, unmap() {}};
      buffers.push(buffer); return buffer;
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        assert.equal(buffer.destroyed, 0, 'a retired buffer must never receive a queue write');
        const bytes = Uint8Array.from(data);
        buffer.data.set(bytes, offset); writes.push({buffer, offset, bytes});
      },
      submit(commands) { submissions.push({commands, complete: false}); },
      onSubmittedWorkDone() {
        return new Promise((resolve, reject) => waits.push({resolve, reject, count: submissions.length}));
      },
    },
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createComputePipeline: () => ({getBindGroupLayout: () => ({})}),
    createBindGroup: ({entries}) => ({entries}),
    createCommandEncoder() {
      const dispatches = [], copies = [];
      return {
        beginComputePass() {
          let group;
          return {end() {}, setPipeline() {}, setBindGroup(_index, value) { group = value; },
            dispatchWorkgroups() { dispatches.push(group); }};
        },
        copyBufferToBuffer(source, sourceOffset, target, targetOffset, bytes) {
          copies.push({source, sourceOffset, target, targetOffset, bytes});
        },
        finish: () => ({dispatches, copies}),
      };
    },
  };
  const paramsBinding = storage ? 1 : 0;
  const shaderCatalog = {SHADER_NAMES: ['uniform.test'], SHADER_CATALOG_HASH: '0'.repeat(64),
    SHADER_ENTRY_POINTS: ['main'], SHADER_LAYOUTS: [{slots: storage ? [0] : [], writesMask: 0, paramsBinding}],
    loadShaderPack: async () => ['uniform.test']};
  const bridge = await createWebGPUHostBridge({device, catalog: shaderCatalog});
  const memory = new WebAssembly.Memory({initial: 1});
  bridge.attach(memory, (observer, action, resource, low, high) =>
    events.push({observer, action, resource, bytes: BigInt(low) | BigInt(high) << 32n}));
  const api = bridge.imports;
  function encode(values, shader = 0, slot = paramsBinding, written = false) {
    // Mutate the same descriptor for each dispatch. Parameters are live input,
    // and neither the descriptor nor a previous invocation may stand in for it.
    const descriptor = [1, storage ? 1 : 0, values.length * 4, 0, slot, shader, 1, 1, 1];
    if (storage) descriptor.push(0, 512, 0, 16, written ? 1 : 0);
    descriptor.push(...values);
    new Uint32Array(memory.buffer, 64, descriptor.length).set(descriptor);
    return api.vx_gpu_encode(64);
  }
  function complete(count = submissions.length) {
    for (const submission of submissions.slice(0, count)) {
      if (submission.complete) continue;
      submission.complete = true;
      for (const command of submission.commands) {
        for (const group of command.dispatches) {
          const {buffer, offset = 0} = group.entries.find(entry => entry.binding === paramsBinding).resource;
          assert.equal(buffer.destroyed, 0, 'GPU execution retains its parameter storage');
          results.push([...new Uint32Array(buffer.data.buffer, offset, 2)]);
          if (storage) {
            const tensor = group.entries.find(entry => entry.binding === 0).resource;
            assert.equal(tensor.buffer.destroyed, 0, 'GPU execution retains its tensor storage');
            storageResults.push([...new Uint32Array(tensor.buffer.data.buffer, tensor.offset ?? 0, 4)]);
          }
        }
        for (const copy of command.copies) {
          assert.equal(copy.source.destroyed, 0, 'a queued snapshot retains its source');
          assert.equal(copy.target.destroyed, 0, 'a queued snapshot retains its staging buffer');
          copy.target.data.set(copy.source.data.subarray(copy.sourceOffset, copy.sourceOffset + copy.bytes), copy.targetOffset);
        }
      }
    }
    for (const wait of waits.splice(0)) {
      if (wait.count <= count) wait.resolve(); else waits.push(wait);
    }
  }
  t.after(async () => {
    complete(); lose({reason: 'destroyed', message: 'test cleanup'});
    bridge.close(); await bridge.waitForCompletion();
    for (const [name, descriptor] of previous)
      descriptor ? Object.defineProperty(globalThis, name, descriptor) : delete globalThis[name];
  });
  return {bridge, api, memory, device, buffers, writes, submissions, waits, results, storageResults, events,
    encode, complete, lose, failValidation: () => { validation = {message: 'injected validation failure'}; }};
}

test('uniform storage keeps each dispatch and in-flight submission independent, then uploads current values on reuse', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin();
  assert.equal(f.encode([3, 101]), 0);
  assert.equal(f.encode([5, 202]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  assert.equal(f.writes.length, 1, 'dispatches in one pass share a batched parameter upload');
  const d0 = f.submissions[0].commands[0].dispatches[0].entries.find(e => e.binding === 0).resource;
  const d1 = f.submissions[0].commands[0].dispatches[1].entries.find(e => e.binding === 0).resource;
  assert.equal(d0.buffer, d1.buffer, 'dispatches in one pass share the parameter chunk');
  assert.notEqual(d0.offset, d1.offset, 'dispatches in one command buffer need independent parameter offsets');
  f.api.vx_gpu_begin();
  assert.equal(f.encode([7, 303]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  assert.notEqual(f.writes[1].buffer, f.writes[0].buffer, 'pending GPU work cannot return a lease to the pool');
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 0));
  f.complete(); await f.bridge.waitForCompletion();
  assert.deepEqual(f.results, [[3, 101], [5, 202], [7, 303]]);
  const allocated = f.buffers.length;
  f.api.vx_gpu_begin();
  assert.equal(f.encode([11, 404]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  assert.equal(f.buffers.length, allocated, 'completed parameter storage is reused');
  assert.equal(f.writes.length, 3, 'each pass uploads its live parameter bytes in batch');
  f.complete(); await f.bridge.waitForCompletion();
  assert.deepEqual(f.results.at(-1), [11, 404]);
});

test('uniform reuse respects rounded extents and never retains released tensor storage as parameters', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(1024, 16, 0), 1);
  const tensor = f.buffers[0];
  f.api.vx_gpu_release(1024); // Ordinary retirement shares the submission, not the parameter pool.
  assert.equal(f.encode([1, 2]), 0);
  assert.equal(f.encode([3, 4, 5, 6, 7]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(tensor.destroyed, 1);
  const allocated = f.buffers.length;
  f.api.vx_gpu_begin();
  assert.equal(f.encode([8, 9, 10]), 0); // 12 and 8 bytes share the same 16-byte allocation class.
  assert.equal(f.encode([11, 12, 13, 14, 15, 16]), 0); // 24 and 20 bytes round to 32.
  assert.equal(f.api.vx_gpu_end(), 0);
  assert.equal(f.buffers.length, allocated);
  assert.ok(f.writes.slice(-2).every(write => write.buffer !== tensor));
  f.complete(); await f.bridge.waitForCompletion();
  assert.deepEqual(f.results, [[1, 2], [3, 4], [8, 9], [11, 12]]);
});

test('memory inventory includes pooled parameters until actual owner destruction', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.api.vx_gpu_memory_start(9, 1024 * 1024), 1);
  assert.deepEqual(f.events.map(event => [event.action, event.bytes]), [[0, 1048576n]]);
  f.api.vx_gpu_begin(); assert.equal(f.encode([3, 4]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.events.length, 1, 'pool checkout/return does not invent allocation or free events');
  f.bridge.close(); await f.bridge.waitForCompletion();
  assert.equal(f.buffers[0].destroyed, 1);
  assert.deepEqual(f.events.map(event => event.action), [0, 2]);
});

test('closing waits for parameter consumers and destroys each pending and free buffer once', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  f.api.vx_gpu_begin(); assert.equal(f.encode([3, 4]), 0); assert.equal(f.encode([5, 6]), 0);
  f.api.vx_gpu_end();
  f.bridge.close(); f.bridge.close();
  let drained = false;
  const closing = f.bridge.waitForCompletion().then(() => { drained = true; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(drained, false);
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 0));
  f.complete(); await closing;
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.encode([7, 8]), -1);
});

test('failed submissions discard parameter leases instead of returning them to the pool', async t => {
  const f = await uniformFixture(t);
  f.failValidation();
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.buffers[0].destroyed, 1);
});

test('refused dispatches and rejected queue completions release checked-out parameter storage', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin();
  assert.equal(f.encode([1, 2], 1), -1, 'no declared shader is an encoding failure');
  assert.equal(f.api.vx_gpu_end(), -1);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.buffers[0].destroyed, 1);
  f.api.vx_gpu_begin(); assert.equal(f.encode([3, 4]), 0); f.api.vx_gpu_end();
  for (const submission of f.submissions) submission.complete = true;
  for (const wait of f.waits.splice(0)) wait.reject(new Error('injected queue failure'));
  await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
});

test('closing an unsubmitted pass retires its checked-out parameter storage', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0);
  f.bridge.close();
  assert.equal(f.submissions.length, 0, 'close discards unfinished numerical commands');
  assert.equal(f.buffers[0].destroyed, 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.buffers[0].destroyed, 1);
});

test('parameter chunk creation cleans up GPU buffer if chunk registration fails', async t => {
  const f = await uniformFixture(t);
  let armed = false, created = null;
  const origCreateBuffer = f.device.createBuffer.bind(f.device);
  f.device.createBuffer = (desc) => {
    const buf = origCreateBuffer(desc);
    if (desc.label === 'vx.params') {
      created = buf;
      armed = true;
    }
    return buf;
  };
  const realPush = Array.prototype.push;
  Array.prototype.push = function(...args) {
    if (armed) {
      armed = false;
      throw new Error('injected push failure');
    }
    return realPush.apply(this, args);
  };
  try {
    f.api.vx_gpu_begin();
    assert.equal(f.encode([1, 2]), -1, 'chunk allocation failure fails encoding');
    assert.ok(created && created.destroyed === 1, 'GPU buffer was destroyed on chunk allocation failure');
  } finally {
    Array.prototype.push = realPush;
    f.bridge.close();
  }
});

test('cached parameter storage remains owned when active registration fails', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  const cached = f.buffers[0];
  const allocated = f.buffers.length;
  const realPush = Array.prototype.push;
  let injected = false;
  try {
    f.api.vx_gpu_begin();
    Array.prototype.push = function(...args) {
      if (!injected && args.length === 1 && args[0]?.buffer === cached &&
          args[0]?.staging instanceof Uint8Array) {
        injected = true;
        throw new Error('injected cached chunk registration failure');
      }
      return realPush.apply(this, args);
    };
    assert.equal(f.encode([3, 4]), -1);
  } finally {
    Array.prototype.push = realPush;
  }
  assert.equal(injected, true);
  assert.equal(f.api.vx_gpu_end(), -1);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(cached.destroyed, 0, 'failed checkout leaves the completed chunk owned by its pool');

  f.api.vx_gpu_begin(); assert.equal(f.encode([5, 6]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(f.buffers.length, allocated, 'the next request reuses the retained chunk');
  assert.deepEqual(f.results, [[1, 2], [5, 6]]);
  f.bridge.close(); await f.bridge.waitForCompletion();
  assert.equal(cached.destroyed, 1, 'owner close destroys the retained chunk exactly once');
});

for (const [kind, count, words] of [['objects', 130, 8192], ['bytes', 130, 8192]]) {
  test(`completed parameter storage has a bounded ${kind} pool without dropping numerical work`, async t => {
    const f = await uniformFixture(t);
    f.api.vx_gpu_begin();
    const values = new Array(words).fill(0);
    for (let index = 0; index < count; index++) {
      values[0] = index; values[1] = index + 1;
      assert.equal(f.encode(values), 0);
    }
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    assert.equal(f.results.length, count);
    assert.deepEqual(f.results.at(-1), [count - 1, count]);
    const retained = f.buffers.filter(buffer => buffer.destroyed === 0);
    assert.ok(retained.length <= 4);
    assert.ok(retained.reduce((sum, buffer) => sum + buffer.size, 0) <= 4 * 1024 * 1024);
    assert.ok(f.buffers.some(buffer => buffer.destroyed === 1), 'excess storage is destroyed after completion');
  });
}

test('bind groups are cached across identical invocations and invalidated on buffer destruction', async t => {
  const f = await uniformFixture(t);
  let bindGroupCreations = 0;
  const origCreateBindGroup = f.device.createBindGroup;
  f.device.createBindGroup = (desc) => {
    bindGroupCreations++;
    return origCreateBindGroup(desc);
  };
  f.api.vx_gpu_begin();
  assert.equal(f.encode([1, 2]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(bindGroupCreations, 1, 'first dispatch creates a bind group');

  f.api.vx_gpu_begin();
  assert.equal(f.encode([3, 4]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(bindGroupCreations, 1, 'identical dispatch shape/layout/buffers reuses cached bind group');

  // Device loss destroys parameter chunk and evicts cache
  f.lose({reason: 'unknown', message: 'test eviction'});
  await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(b => b.destroyed === 1));
});

test('alternating multi-stage execution reuses inactive tensor buffers without churn', async t => {
  const f = await uniformFixture(t);
  let bindGroupCreations = 0;
  const origCreateBindGroup = f.device.createBindGroup;
  f.device.createBindGroup = (desc) => {
    bindGroupCreations++;
    return origCreateBindGroup(desc);
  };

  // Stage 1 (e.g. duration): 16 bytes
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 1);
  const b1 = f.buffers.at(-1);
  assert.equal(f.encode([1, 2]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(bindGroupCreations, 1);

  // Stage 2 (e.g. acoustic): extent resize to 64 bytes on same host pointer
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 64, 0), 1);
  const b2 = f.buffers.at(-1);
  assert.notEqual(b1, b2);
  assert.equal(f.encode([3, 4]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(b1.destroyed, 0, 'b1 retained in inactive pool rather than destroyed');

  const allocated = f.buffers.length;

  // Next request: Stage 1 runs again, requesting 16 bytes on same host pointer
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 1);
  assert.equal(f.buffers.length, allocated, 'reused inactive buffer without allocating');
  assert.equal(f.encode([5, 6]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(b1.destroyed, 0);
  assert.equal(b2.destroyed, 0);

  // Next request: Stage 2 runs again, requesting 64 bytes
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 64, 0), 1);
  assert.equal(f.buffers.length, allocated, 'reused inactive buffer without allocating');
  assert.equal(f.encode([7, 8]), 0);
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();

  // Explicit release drains both active and inactive buffers
  f.api.vx_gpu_release(512);
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(b1.destroyed, 1, 'explicit release drains inactive buffer');
  assert.equal(b2.destroyed, 1, 'explicit release drains active buffer');
});

test('memory observation preserves inactive reuse, current host bytes and exactly-once destruction', async t => {
  const f = await uniformFixture(t);
  const pointer = 512;
  async function ensure(bytes, value) {
    new Uint8Array(f.memory.buffer, pointer, bytes).fill(value);
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(pointer, bytes, 0), 1);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    const buffer = f.writes.at(-1).buffer;
    assert.deepEqual(buffer.data, new Uint8Array(bytes).fill(value));
    return buffer;
  }
  const small = await ensure(16, 17);
  const large = await ensure(64, 29);
  assert.equal(f.api.vx_gpu_memory_start(7, 32), 1);
  assert.deepEqual(f.events.map(e => [e.action, e.bytes]).sort((a, b) => Number(a[1] - b[1])),
    [[0, 16n], [0, 64n]], 'inventory sees the active and inactive allocations');
  assert.equal(await ensure(16, 43), small);
  assert.equal(f.events.length, 2, 'observation neither allocates nor destroys cached spans');
  f.api.vx_gpu_memory_stop(7);
  assert.equal(await ensure(64, 59), large);
  assert.equal(f.buffers.length, 2, 'turning observation on and off does not change allocation policy');
  f.bridge.close(); f.complete(); await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1), 'owner close drains every allocation once');
});

for (const separateSubmissions of [false, true]) {
  test(`extent reuse preserves tensor inputs until ${separateSubmissions ? 'pending submissions' : 'the open pass'} complete`, async t => {
    const f = await uniformFixture(t, true);
    const bound = [];
    for (const [index, bytes] of [16, 64, 16].entries()) {
      if (separateSubmissions || index === 0) f.api.vx_gpu_begin();
      new Uint32Array(f.memory.buffer, 512, bytes / 4).fill(index + 1);
      assert.equal(f.api.vx_gpu_ensure(512, bytes, 0), 1);
      bound.push(f.writes.at(-1).buffer);
      assert.equal(f.encode([index + 1, 101 + index]), 0);
      if (separateSubmissions) assert.equal(f.api.vx_gpu_end(), 0);
    }
    if (!separateSubmissions) assert.equal(f.api.vx_gpu_end(), 0);
    assert.notEqual(bound[0], bound[2], 'an earlier dispatch still owns the original extent');
    assert.ok(bound.every(buffer => !buffer.destroyed));
    f.complete(); await f.bridge.waitForCompletion();
    assert.deepEqual(f.storageResults, [1, 2, 3].map(value => new Array(4).fill(value)));
    assert.deepEqual(f.results, [[1, 101], [2, 102], [3, 103]]);

    const allocated = f.buffers.length;
    f.api.vx_gpu_begin();
    new Uint32Array(f.memory.buffer, 512, 16).fill(4);
    assert.equal(f.api.vx_gpu_ensure(512, 64, 0), 1);
    assert.equal(f.writes.at(-1).buffer, bound[1], 'completed storage may now be reused');
    assert.equal(f.encode([4, 104]), 0);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    assert.equal(f.buffers.length, allocated);
    assert.deepEqual(f.storageResults.at(-1), new Array(4).fill(4));
  });
}

test('immutable inactive weights preserve uploaded bytes until invalidation or reclassification', async t => {
  const f = await uniformFixture(t, true);
  const tensorWrites = () => f.writes.filter(write => write.buffer.label !== 'vx.params').length;
  async function stage(bytes, flags, hostValue, expectedValue, expectedUploads) {
    new Uint32Array(f.memory.buffer, 512, bytes / 4).fill(hostValue);
    const before = tensorWrites();
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(512, bytes, flags), 1);
    assert.equal(f.encode([expectedValue, expectedValue + 1]), 0);
    assert.equal(f.api.vx_gpu_end(), 0);
    const bound = f.submissions.at(-1).commands[0].dispatches[0].entries.find(entry => entry.binding === 0).resource.buffer;
    f.complete(); await f.bridge.waitForCompletion();
    assert.equal(tensorWrites() - before, expectedUploads);
    assert.deepEqual(f.storageResults.at(-1), new Array(4).fill(expectedValue));
    return bound;
  }
  const small = await stage(16, 1, 11, 11, 1);
  const large = await stage(64, 1, 11, 11, 1);
  const allocated = f.buffers.length;
  // The immutable flag promises unchanged canonical contents. Unannounced host
  // edits cannot silently replace them, just as for an active weight buffer.
  assert.equal(await stage(16, 1, 77, 11, 0), small);
  f.api.vx_gpu_invalidate(512); // Revokes both active and inactive uploaded bits.
  assert.equal(await stage(64, 1, 83, 83, 1), large);
  assert.equal(await stage(16, 1, 97, 97, 1), small);
  // Storage reuse cannot carry the immutable-upload promise across a mutable
  // residency, in either direction and even at the same pointer and extent.
  assert.equal(await stage(64, 0, 101, 101, 1), large);
  assert.equal(await stage(16, 0, 103, 103, 1), small);
  assert.equal(await stage(64, 1, 107, 107, 1), large);
  assert.equal(await stage(16, 1, 109, 109, 1), small);
  assert.equal(f.buffers.length, allocated);
});

test('inactive checkout reclassifies weight and dirty content instead of reusing old logical values', async t => {
  const f = await uniformFixture(t, true);
  async function ensure(bytes, value, flags = 0, written = false) {
    new Uint32Array(f.memory.buffer, 512, bytes / 4).fill(value);
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(512, bytes, flags), 1);
    assert.equal(f.encode([value, value + 1], 0, 1, written), 0);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    assert.deepEqual(f.storageResults.at(-1), new Array(4).fill(value));
    return f.writes.findLast(write => write.buffer.label !== 'vx.params').buffer;
  }
  const small = await ensure(16, 11, 1); // Immutable weight only for this concrete residency.
  await ensure(64, 22);
  assert.equal(await ensure(16, 33), small, 'the storage is reusable as a mutable tensor');
  assert.equal(f.api.vx_gpu_ensure(512, 16, 2), 1, 'old weight metadata cannot refuse a mutable output');
  await ensure(16, 44, 0, true); // Marks the old logical value device-dirty.
  await ensure(64, 55);
  assert.equal(await ensure(16, 66), small, 'reused storage uploads the new host value despite old deviceDirty');
});

test('failed inactive checkout registration returns an error and retains every allocation for cleanup', async t => {
  const f = await uniformFixture(t);
  async function ensure(bytes) {
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(512, bytes, 0), 1);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
  }
  await ensure(16);
  const cached = f.buffers[0];
  await ensure(64);
  const realSet = Map.prototype.set;
  let injected = false;
  f.api.vx_gpu_begin();
  try {
    Map.prototype.set = function(key, value) {
      if (!injected && key === 512 && value?.buffer === cached) {
        injected = true;
        throw new Error('injected restored-span registration failure');
      }
      return realSet.call(this, key, value);
    };
    assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 0, 'allocation failure crosses the private ABI as a status');
  } finally {
    Map.prototype.set = realSet;
  }
  assert.equal(injected, true);
  assert.equal(f.api.vx_gpu_end(), -1);
  f.complete(); await f.bridge.waitForCompletion();
  new Uint8Array(f.memory.buffer, 512, 16).fill(71);
  await ensure(16);
  assert.deepEqual(f.writes.at(-1).buffer.data, new Uint8Array(16).fill(71), 'a subsequent request uses current input');
  f.bridge.close(); f.complete(); await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1), 'failed transfer loses no owner and never frees twice');
});

test('inactive tensor retention has one global byte limit across host pointers', async t => {
  const f = await uniformFixture(t);
  const mib = 1024 * 1024;
  f.memory.grow(32);
  const active = new Map();
  for (let index = 0; index < 6; index++) {
    const pointer = 1024 + index * 64;
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(pointer, mib, 0), 1);
    assert.equal(f.api.vx_gpu_ensure(pointer, 16, 0), 1);
    active.set(pointer, f.writes.at(-1).buffer);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    const idle = f.buffers.filter(buffer => !buffer.destroyed && ![...active.values()].includes(buffer));
    assert.ok(idle.reduce((sum, buffer) => sum + buffer.size, 0) <= 4 * mib,
      'the byte limit applies to all pointers together, independently of parameter storage');
  }
  assert.ok(f.buffers.some(buffer => buffer.size === mib && buffer.destroyed === 1), 'over-budget idle buffers are retired');
  for (const pointer of active.keys()) f.api.vx_gpu_release(pointer);
  f.complete(); await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
});

test('synchronous allocation pressure evicts completed idle tensors and retries once', async t => {
  const f = await uniformFixture(t);
  const create = f.device.createBuffer;
  let refusals = 0;
  f.device.createBuffer = descriptor => {
    const live = f.buffers.filter(buffer => !buffer.destroyed).reduce((sum, buffer) => sum + buffer.size, 0);
    if (live + descriptor.size > 256) { refusals++; throw new Error('injected device capacity pressure'); }
    return create(descriptor);
  };
  for (const [index, bytes] of [64, 80, 96, 112].entries()) {
    new Uint8Array(f.memory.buffer, 512, bytes).fill(index + 1);
    f.api.vx_gpu_begin();
    assert.equal(f.api.vx_gpu_ensure(512, bytes, 0), 1);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
    assert.deepEqual(f.writes.at(-1).buffer.data, new Uint8Array(bytes).fill(index + 1));
  }
  assert.equal(refusals, 1, 'one failed allocation is retried after optional completed storage is evicted');
  assert.equal(f.buffers.filter(buffer => !buffer.destroyed).reduce((sum, buffer) => sum + buffer.size, 0), 112);
  f.bridge.close(); f.complete(); await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
});

test('a failed completion cannot return tensor storage to a later logical residency', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 1);
  const failed = f.writes.at(-1).buffer;
  const pop = f.device.popErrorScope;
  f.device.popErrorScope = async () => ({message: 'injected submission validation failure'});
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  f.device.popErrorScope = pop;
  for (const bytes of [64, 16]) {
    f.api.vx_gpu_begin();
    new Uint8Array(f.memory.buffer, 512, bytes).fill(83);
    assert.equal(f.api.vx_gpu_ensure(512, bytes, 0), 1);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
  }
  assert.notEqual(f.writes.at(-1).buffer, failed, 'failed-owner storage is destroyed rather than checked out');
  assert.equal(failed.destroyed, 1);
  assert.deepEqual(f.writes.at(-1).buffer.data, new Uint8Array(16).fill(83));
});

test('an outstanding snapshot retains an old extent through resize and explicit release', async t => {
  const f = await uniformFixture(t);
  new Uint8Array(f.memory.buffer, 512, 16).fill(97);
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 1);
  const original = f.writes.at(-1).buffer;
  assert.equal(f.api.vx_gpu_end(), 0);
  f.complete(); await f.bridge.waitForCompletion();
  const ticket = f.api.vx_gpu_snapshot(512, 0, 16, 0);
  assert.ok(ticket > 0);
  assert.equal(f.api.vx_gpu_readback(ticket, 1024, 16), -2);
  f.api.vx_gpu_begin();
  assert.equal(f.api.vx_gpu_ensure(512, 64, 0), 1);
  f.api.vx_gpu_release(512);
  assert.equal(f.api.vx_gpu_end(), 0);
  await Promise.resolve();
  assert.equal(original.destroyed, 0, 'completed graph work does not imply that its later snapshot completed');
  f.complete(); await f.bridge.waitForCompletion();
  assert.equal(original.destroyed, 1);
  assert.equal(f.api.vx_gpu_readback(ticket, 1024, 16), 0);
  assert.deepEqual(new Uint8Array(f.memory.buffer, 1024, 16), new Uint8Array(16).fill(97));
  f.api.vx_gpu_readback_release(ticket);
  f.complete(); await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
});

test('releasing a transient tensor removes its bind-group keys from surviving parameter buffers', async t => {
  const f = await uniformFixture(t, true);
  const reverseSets = new Set();
  const realAdd = Set.prototype.add;
  // Observe only cache-key registrations. This checks retained host metadata
  // without exposing an engine API or retaining GPU resources in the product.
  Set.prototype.add = function(value) {
    if (typeof value === 'string' && value.startsWith('0_') && value.includes(':')) {
      realAdd.call(reverseSets, this);
    }
    return realAdd.call(this, value);
  };
  try {
    for (let index = 0; index < 128; index++) {
      f.api.vx_gpu_begin();
      assert.equal(f.api.vx_gpu_ensure(512, 16, 0), 1);
      assert.equal(f.encode([index, index + 1]), 0);
      assert.equal(f.api.vx_gpu_end(), 0);
      f.complete(); await f.bridge.waitForCompletion();
      f.api.vx_gpu_release(512);
      f.complete(); await f.bridge.waitForCompletion();
    }
  } finally {
    Set.prototype.add = realAdd;
  }
  assert.ok(reverseSets.size > 0, 'the probe observed actual cache-key registrations');
  assert.ok([...reverseSets].every(set => set.size === 0), 'no surviving reverse index retains an evicted key');
  assert.deepEqual(f.results, Array.from({length: 128}, (_, index) => [index, index + 1]));
  assert.equal(f.buffers.filter(buffer => buffer.label === 'vx.params' && !buffer.destroyed).length, 1,
    'the parameter buffer is reused, so destroying every buffer cannot hide stale metadata');
});

test('bind-group cache eviction bounds live reverse metadata without dropping dispatches', async t => {
  const f = await uniformFixture(t);
  const reverseSets = new Set();
  const realAdd = Set.prototype.add;
  Set.prototype.add = function(value) {
    if (typeof value === 'string' && value.startsWith('0_') && value.includes(':')) realAdd.call(reverseSets, this);
    return realAdd.call(this, value);
  };
  try {
    f.api.vx_gpu_begin();
    for (let index = 0; index < 4097; index++) assert.equal(f.encode([index, index + 1]), 0);
    assert.equal(f.api.vx_gpu_end(), 0);
    f.complete(); await f.bridge.waitForCompletion();
  } finally {
    Set.prototype.add = realAdd;
  }
  assert.ok(reverseSets.size > 0);
  assert.ok([...reverseSets].reduce((sum, set) => sum + set.size, 0) <= 4096,
    'uniform-only bind groups have one reverse registration each, bounded across all live buffers');
  assert.deepEqual(f.results, Array.from({length: 4097}, (_, index) => [index, index + 1]));
});

test('device loss drains pending and free parameter storage without waiting for queue completion', async t => {
  const f = await uniformFixture(t);
  f.api.vx_gpu_begin(); assert.equal(f.encode([1, 2]), 0); f.api.vx_gpu_end();
  f.complete(); await f.bridge.waitForCompletion();
  f.api.vx_gpu_begin(); assert.equal(f.encode([3, 4, 5, 6, 7]), 0); f.api.vx_gpu_end();
  f.lose({reason: 'unknown', message: 'injected loss'});
  await f.bridge.waitForCompletion();
  assert.ok(f.buffers.every(buffer => buffer.destroyed === 1));
  assert.equal(f.encode([8, 9]), -1);
  // The device no longer executes these commands; settle the fake queue only.
  for (const submission of f.submissions) submission.complete = true;
  f.complete();
});
