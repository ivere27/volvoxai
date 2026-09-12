import assert from 'node:assert/strict';
import {WebGPUEngine, RuntimeGraphBuilder} from './old-ts-engine.mjs';

const mode = Deno.args[0] ?? 'single';
let phase = 'start';
const watchdog = setTimeout(() => {
  console.error(JSON.stringify({mode, timeout:phase})); Deno.exit(124);
}, 30_000);
const shape = [1, 2, 3, 2];
const input = Float32Array.from({length:12}, (_, i) => (i % 7) - 3 + i * 0.25);
const expected = Float32Array.from(input, x => 1 / (1 + Math.exp(-Math.max(0,x))));
let devicesRequested = 0;
const originalAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
navigator.gpu.requestAdapter = async (...args) => {
  const adapter = await originalAdapter(...args);
  if (adapter) {
    const originalDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (...args) => {
      const device = await originalDevice(...args);
      console.log(JSON.stringify({mode, event:'requestDevice', count:++devicesRequested}));
      return device;
    };
  }
  return adapter;
};

async function unusedDevice() {
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  const device = await adapter.requestDevice();
  console.log(JSON.stringify({mode, event:'unused-device', adapter:adapter.info.description}));
  device.destroy();
  await device.lost;
}

async function run(iteration, count) {
  phase = `${iteration}:requestDevice`;
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  assert.match(adapter.info.description, /NVIDIA.*3090/);
  const device = await adapter.requestDevice();
  const engine = new WebGPUEngine(device, {adapterInfo:adapter.info});
  try {
    const b = new RuntimeGraphBuilder();
    const x = b.input('input0', shape);
    const mid = b.addOp('ReLU', {input:x}, {out:{name:'mid', shape}}, {}, {id:'a'}).out;
    const out = b.addOp('Sigmoid', {input:mid}, {out:{name:'out0', shape}}, {}, {id:'b'}).out;
    b.outputs(out);
    const graph = b.build();
    phase = `${iteration}:allocateGraph`;
    await engine.allocateGraph(graph, new Map([...graph.tensors.keys()].map(name => [name,name])));
    for (let execution=0; execution<count; execution++) {
      phase = `${iteration}:${execution}:execute`;
      await engine.execute({input0:input});
      phase = `${iteration}:${execution}:readBuffer`;
      const output = await engine.readBuffer(engine.gpuBuffers.get('out0'), input.byteLength);
      let maxAbsoluteError = 0;
      for (let i=0; i<input.length; i++) maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(output[i]-expected[i]));
      assert.ok(maxAbsoluteError < 1e-6, `maximum error ${maxAbsoluteError}`);
      console.log(JSON.stringify({mode, iteration, execution, adapter:adapter.info.description, maxAbsoluteError, status:'PASS'}));
    }
    phase = `${iteration}:queueDone`;
    await device.queue.onSubmittedWorkDone();
  } finally {
    phase = `${iteration}:dispose`;
    engine.dispose();
    device.destroy();
    await device.lost;
  }
}

try {
  if (mode === 'metadata-old' || mode === 'metadata-current') {
    const api = await import(mode === 'metadata-old' ? './old-full.mjs' : './volvoxai.full.min.js');
    const host = new api.FullEngineHost({wasmUrl:new URL('./volvoxai.full.wasm',import.meta.url)});
    const platform = new api.VxPlatformServiceClient(host);
    await platform.getPlatformInfo(new api.pb.Empty());
    console.log(JSON.stringify({mode, event:'platform-info-returned', devicesRequested}));
    await (await (await host.close?.()));
  }
  if (mode === 'pre-device') await unusedDevice();
  for (let iteration=0; iteration<(mode === 'recreate' ? 3 : 1); iteration++)
    await run(iteration, mode === 'single' ? 3 : 1);
  console.log(JSON.stringify({mode, status:'PASS'}));
} finally {
  clearTimeout(watchdog);
}
