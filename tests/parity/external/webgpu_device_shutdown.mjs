/** Device shutdown regression for a real Deno WebGPU adapter.
 * No VolvoxAI, forced GC, loader shim or memory-budget override.
 * Every shutdown scenario is followed by a numerical copy on a fresh device.
 * Destroyed devices and buffers deliberately remain reachable throughout.
 */
import assert from 'node:assert/strict';

const cycles = Number(Deno.args[0] ?? 10);
assert.ok(Number.isSafeInteger(cycles) && cycles > 0 && cycles <= 64);
const retained = [];
const scenarios = ['idle', 'unsubmitted-write', 'submitted-work', 'completed-copy', 'mapped-at-creation'];
let phase = 'start';
const watchdog = setTimeout(() => {
  console.error(`device shutdown timed out at ${phase}`);
  Deno.exit(124);
}, 60_000);

async function acquire() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  assert.ok(adapter, 'physical adapter required');
  assert.match(adapter.info.description, /NVIDIA.*3090/i, 'this qualification requires an NVIDIA RTX 3090');
  return adapter.requestDevice();
}

function buffers(device) {
  return [
    device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
    device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  ];
}

function submit(device, source, destination) {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, destination, 0, 48);
  device.queue.submit([encoder.finish()]);
}

async function copy(device) {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const [source, destination] = buffers(device);
  const input = Float32Array.from({ length: 12 }, (_, i) => i - 6);
  device.queue.writeBuffer(source, 0, input);
  submit(device, source, destination);
  await destination.mapAsync(GPUMapMode.READ);
  assert.deepEqual(new Float32Array(destination.getMappedRange()), input);
  destination.unmap();
  assert.equal(await device.popErrorScope(), null);
  assert.equal(await device.popErrorScope(), null);
  await device.queue.onSubmittedWorkDone();
  // Exercise explicit destruction as well as the implicit destruction below.
  source.destroy();
  destination.destroy();
  retained.push(source, destination);
}

async function close(device) {
  const queue = device.queue;
  const limit = device.limits.maxBufferSize;
  device.destroy();
  assert.equal((await device.lost).reason, 'destroyed');
  device.destroy(); // idempotent; the registry entry must still be valid.
  assert.equal(device.queue, queue);
  assert.equal(device.limits.maxBufferSize, limit);
  assert.ok(device.features);
  const invalid = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST });
  invalid.destroy(); // disposed API objects must remain safe to use/destroy.
  retained.push(device, invalid);
}

try {
  for (let iteration = 0; iteration < cycles; iteration++) {
    const scenario = scenarios[iteration % scenarios.length];
    phase = `${iteration}:${scenario}`;
    const device = await acquire();
    let completion;
    if (scenario === 'mapped-at-creation') {
      const buffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
      new Uint8Array(buffer.getMappedRange()).fill(iteration);
      retained.push(buffer);
    } else if (scenario === 'completed-copy') {
      await copy(device);
    } else if (scenario !== 'idle') {
      const [source, destination] = buffers(device);
      device.queue.writeBuffer(source, 0, new Uint8Array(48).fill(iteration));
      if (scenario === 'submitted-work') {
        submit(device, source, destination);
        // Observe the pending mapping before destroy, without waiting for it.
        // Loss can reject it; either outcome must settle without a crash/hang.
        const mapping = destination.mapAsync(GPUMapMode.READ).then(() => 'mapped', () => 'lost');
        completion = Promise.all([mapping, device.queue.onSubmittedWorkDone()]);
      }
      retained.push(source, destination);
    }
    await close(device);
    if (completion) await completion;

    phase = `${iteration}:fresh-copy`;
    const next = await acquire();
    await copy(next);
    await close(next);
    console.log(JSON.stringify({ iteration, scenario, freshCopy: 'pass', retained: retained.length }));
  }
  console.log(JSON.stringify({ passed: true, cycles, devices: cycles * 2, forcedGc: false }));
} finally {
  clearTimeout(watchdog);
}
