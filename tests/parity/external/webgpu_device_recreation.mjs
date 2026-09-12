/** Standalone device lifecycle reproduction. No VolvoxAI code, WASM or shaders.
 * deno run --unstable-webgpu tests/parity/external/webgpu_device_recreation.mjs
 * A device recreated after destroy must still support a 48-byte buffer copy.
 */
import assert from 'node:assert/strict';

async function bounded(promise, phase) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${phase} timed out after 15 seconds`)), 15_000);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

for (let iteration = 0; iteration < 3; iteration++) {
  // requestDevice consumes an adapter. Reinitialization starts with a fresh adapter.
  const adapter = await bounded(navigator.gpu.requestAdapter({powerPreference:'high-performance'}),
    `device ${iteration} requestAdapter`);
  assert.ok(adapter, 'a physical WebGPU adapter is required');
  const device = await bounded(adapter.requestDevice(), `device ${iteration} requestDevice`);
  let source, destination;
  try {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    source = device.createBuffer({size:48,
      usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    destination = device.createBuffer({size:48,
      usage:GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const input = Float32Array.from({length:12}, (_, index) => index);
    device.queue.writeBuffer(source, 0, input);
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, destination, 0, 48);
    device.queue.submit([encoder.finish()]);
    const mapped = destination.mapAsync(GPUMapMode.READ).then(() => null, error => error);
    const [validation, oom, mapError] = await bounded(Promise.all([
      device.popErrorScope(), device.popErrorScope(), mapped,
    ]), `device ${iteration} map/error scopes`);
    const errors = [validation, oom, mapError].filter(Boolean).map(error => error.message);
    console.log(JSON.stringify({iteration, device:adapter.info.description, errors}));
    assert.deepEqual(errors, [], `device ${iteration} could not copy 48 bytes`);
    assert.deepEqual(new Float32Array(destination.getMappedRange()), input);
    destination.unmap();
    await bounded(device.queue.onSubmittedWorkDone(), `device ${iteration} queue completion`);
  } finally {
    source?.destroy();
    destination?.destroy();
    device.destroy();
    await bounded(device.lost, `device ${iteration} lost after destroy`);
  }
}
