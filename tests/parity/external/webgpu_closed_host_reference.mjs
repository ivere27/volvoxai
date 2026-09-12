import assert from 'node:assert/strict';
const mode = Deno.args[0] ?? 'retain';
const api = await import(Deno.args[1] ?? './volvoxai.full.min.js');
let retainedHost;
async function prepare() {
  const host = new api.FullEngineHost({wasmUrl:new URL('./volvoxai.full.wasm',import.meta.url)});
  await new api.VxPlatformServiceClient(host).getPlatformInfo(new api.pb.Empty());
  await host.close();
  if (mode === 'retain') retainedHost=host;
}
await prepare();
globalThis.gc();
await new Promise(resolve => setTimeout(resolve,0));
globalThis.gc();
console.log(JSON.stringify({mode, event:'gc-after-host-close', retained:!!retainedHost}));
const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
const device=await adapter.requestDevice();
device.pushErrorScope('out-of-memory');
const buffer=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const error=await device.popErrorScope();
console.log(JSON.stringify({mode, error:error?.message??null}));
buffer.destroy();
device.destroy();
await device.lost;
// Keep the closed host reachable through the second allocation.
if (retainedHost) await retainedHost.close();
assert.equal(error,null);
