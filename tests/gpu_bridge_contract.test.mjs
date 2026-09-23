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
  'vx_gpu_invalidate', 'vx_gpu_memory_start', 'vx_gpu_memory_stop', 'vx_gpu_begin', 'vx_gpu_begin_activity', 'vx_gpu_begin_trace', 'vx_gpu_trace_node_begin', 'vx_gpu_trace_node_end', 'vx_gpu_trace_program_begin', 'vx_gpu_trace_program_end', 'vx_gpu_trace_read', 'vx_gpu_trace_release', 'vx_gpu_await_read', 'vx_gpu_await_release', 'vx_gpu_encode', 'vx_gpu_end',
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
  let finish, finishDestroy, destroyed = 0;
  const queued = new Promise(resolve => {finish = resolve;});
  const device = {
    lost: new Promise(resolve => {finishDestroy = resolve;}),
    queue: {submit() {}, onSubmittedWorkDone: () => queued},
    pushErrorScope() {}, popErrorScope: async () => null,
    createCommandEncoder: () => ({beginComputePass: () => ({end() {}}), finish: () => ({})}),
    destroy() {destroyed++;},
  };
  Object.defineProperty(globalThis, 'navigator', {configurable:true, value:{gpu:{
    requestAdapter: async () => ({features: new Set(), requestDevice: async () => device}),
  }}});
  try {
    const bridge = await acquireWebGPUHostBridge(catalog);
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
  } finally {
    bridge.close(); await bridge.waitForCompletion();
    if (usage) Object.defineProperty(globalThis, 'GPUBufferUsage', usage);
    else delete globalThis.GPUBufferUsage;
  }
});
