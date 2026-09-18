import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers/promises';
import { createWebGPUHostBridge } from '../../ts/backends/WebGPUHostBridge.js';

// Run in a fresh --expose-gc process so collection and the memory budget do not
// depend on concurrent tests. Keep the device alive across both measurements.
let lose, destroyed = 0;
globalThis.GPUBufferUsage = { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4 };
const device = {
  lost: new Promise(resolve => { lose = resolve; }),
  queue: { submit() {}, writeBuffer() {}, onSubmittedWorkDone: async () => {} },
  pushErrorScope() {}, popErrorScope: async () => null,
  createBuffer: () => ({ destroy() { destroyed++; } }),
  createCommandEncoder: () => ({ beginComputePass: () => ({ end() {} }), finish: () => ({}) }),
};
const bridge = await createWebGPUHostBridge({ device, catalog: {
  loadShaderPack: async () => [], SHADER_NAMES: [], SHADER_CATALOG_HASH: '',
  SHADER_ENTRY_POINTS: [], SHADER_LAYOUTS: [],
} });
bridge.attach(new WebAssembly.Memory({ initial: 1 }));
async function exercise(count) {
  for (let i = 0; i < count; i++) {
    bridge.imports.vx_gpu_begin();
    assert.equal(bridge.imports.vx_gpu_ensure(64, 16, 0), 1);
    assert.equal(bridge.imports.vx_gpu_end(), 0);
    bridge.imports.vx_gpu_release(64);
    await bridge.waitForCompletion();
  }
  await setImmediate();
  globalThis.gc();
  return process.memoryUsage().heapUsed;
}
try {
  await exercise(2000);
  const before = await exercise(20000);
  const after = await exercise(20000);
  assert.equal(destroyed, 42000, 'completed buffers are destroyed while the device remains alive');
  assert.ok(after - before < 4 * 1024 * 1024,
    `completed operations retained ${after - before} bytes on a live device`);
  console.log(JSON.stringify({ before, after, growth: after - before, destroyed }));
} finally {
  bridge.close();
  await bridge.waitForCompletion();
  lose({ reason: 'destroyed', message: 'test complete' });
}
