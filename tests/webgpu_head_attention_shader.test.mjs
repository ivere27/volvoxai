import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

// This operator unit test may use a software adapter. Whole-model physical
// admission and residency are exercised by the external WebGPU graph suites.
async function numericalShaders(sources) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error(`WebGPU adapter unavailable: ${location.href}, secure=${isSecureContext}, gpu=${typeof navigator.gpu}`);
  const device = await adapter.requestDevice();
  const records = [];
  const errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const pipelines = {};
  for (const [name, code] of Object.entries(sources)) {
    const module = device.createShaderModule({ code });
    const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (messages.length) throw new Error(`${name}: ${messages.map(message => message.message).join('; ')}`);
    pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  async function dispatch(name, inputs, parameters, groups, count) {
    const resources = inputs.map(data => {
      const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    });
    const output = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uniform = device.createBuffer({ size: parameters.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniform, 0, parameters);
    const staging = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeline = pipelines[name];
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [...resources, output, uniform].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(...groups); pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, count * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    for (const buffer of [...resources, output, uniform, staging]) buffer.destroy();
    return result;
  }
  function compare(name, actual, expected, tolerance = 3e-5) {
    let maximum = 0;
    for (let i = 0; i < expected.length; i++) {
      const error = Math.abs(actual[i] - expected[i]);
      if (!Number.isFinite(actual[i]) || error > tolerance * (1 + Math.abs(expected[i])))
        throw new Error(`${name}[${i}]: ${actual[i]} vs ${expected[i]} (error ${error})`);
      maximum = Math.max(maximum, error);
    }
    records.push({ name, elements: expected.length, maximumAbsoluteError: maximum });
  }
  const values = (count, phase) => Float32Array.from({ length: count }, (_, i) => .7 * Math.sin(i * .17 + phase));
  try {
    // Independent high precision integer remainders construct the immutable
    // input metadata. Expected rotations use JavaScript's separate libm path.
    const decimalScale = 10n ** 60n;
    const tauScaled = 6283185307179586476925286766559005768394338798750211641949889n;
    for (const interleaved of [0, 1]) for (const mode of [0, 1, 2]) for (const large of [false, true]) {
      const B = 2, S = 3, H = 3, D = 128, R = interleaved ? 126 : 128, pairs = R / 2;
      const x = values(B * S * H * D, .3), positions = Int32Array.from({ length: mode === 2 ? B * S : S },
        (_, i) => large ? 2147483647 - i * 256 : 7 + i * 3);
      const metadata = new Float32Array(pairs + 64), theta = large ? 1 : 1000000;
      for (let i = 0; i < pairs; i++) metadata[i] = Math.fround(theta ** Math.fround(Math.fround(2 * i) / R));
      for (let i = 0; i < 32; i++) {
        const remainder = Number(((1n << BigInt(i)) * decimalScale) % tauScaled) / Number(decimalScale);
        const high = Math.fround(remainder);
        metadata[pairs + i * 2] = high; metadata[pairs + i * 2 + 1] = Math.fround(remainder - high);
      }
      const offset = large ? 2147483645 : 19;
      const params = Uint32Array.of(x.length, S, H * D, H, R, offset, interleaved, mode);
      const expected = x.slice();
      for (let b = 0; b < B; b++) for (let s = 0; s < S; s++) for (let h = 0; h < H; h++) {
        const base = ((b * S + s) * H + h) * D;
        const position = mode === 0 ? offset + s : positions[mode === 1 ? s : b * S + s];
        for (let pair = 0; pair < pairs; pair++) {
          const a = interleaved ? pair * 2 : pair, z = interleaved ? a + 1 : pair + pairs;
          const angle = Math.fround(Math.fround(position) / metadata[pair]);
          expected[base + a] = x[base + a] * Math.cos(angle) - x[base + z] * Math.sin(angle);
          expected[base + z] = x[base + z] * Math.cos(angle) + x[base + a] * Math.sin(angle);
        }
      }
      const actual = await dispatch('rope', [x, positions, metadata], params, [Math.ceil(x.length / 64), 1, 1], x.length);
      compare(`RoPE/D128/${interleaved}/${mode}/${large}`, actual, expected);
    }
    for (const KVH of [1, 2, 4]) for (const causal of [0, 1]) for (const maskMode of [0, 1, 2, 3, 4]) {
      const B = 2, Q = 3, K = 5, H = 4, D = 128;
      const q = values(B * Q * H * D, .3), k = values(B * K * KVH * D, .7), v = values(k.length, 1.1);
      const mask = Int32Array.from({ length: B * Q * K }, (_, i) => i % K === 1 || Math.floor(i / K) % Q === 2 ? 0 : 1);
      const scale = Math.fround(1 / Math.sqrt(D));
      const parameters = new ArrayBuffer(48), u = new Uint32Array(parameters), f = new Float32Array(parameters);
      u.set([Q, K, H * D, H, D, B, 0, causal, maskMode, KVH, 0, 0]); f[6] = scale;
      const expected = new Float32Array(q.length);
      for (let b = 0; b < B; b++) for (let s = 0; s < Q; s++) for (let h = 0; h < H; h++) {
        const scores = [], indices = [], kvHead = Math.floor(h / (H / KVH));
        for (let t = 0; t < K; t++) {
          const mi = maskMode === 2 ? b * K + t : maskMode === 3 ? s * K + t : maskMode === 4 ? (b * Q + s) * K + t : t;
          if (causal && t > s || maskMode && !mask[mi]) continue;
          let sum = 0;
          for (let d = 0; d < D; d++) sum += q[((b * Q + s) * H + h) * D + d] * k[((b * K + t) * KVH + kvHead) * D + d];
          scores.push(sum * scale); indices.push(t);
        }
        if (!scores.length) continue;
        const max = Math.max(...scores), weights = scores.map(score => Math.exp(score - max));
        const total = weights.reduce((a, b) => a + b, 0);
        for (let d = 0; d < D; d++) for (let i = 0; i < indices.length; i++)
          expected[((b * Q + s) * H + h) * D + d] += weights[i] / total * v[((b * K + indices[i]) * KVH + kvHead) * D + d];
      }
      const actual = await dispatch('crossSDPA', [q, k, v, mask], new Uint8Array(parameters), [1, H, B], q.length);
      compare(`CrossSDPA/D128/kv-${KVH}/causal-${causal}/mask-${maskMode}`, actual, expected);
    }
    if (errors.length) throw new Error(errors.join('; '));
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
      device: adapter.info.device, description: adapter.info.description }, records };
  } finally { device.destroy(); }
}

test('authoritative WebGPU RoPE and compact GQA D128 shader numerics', {
  skip: !process.env.VOLVOXAI_TEST_CHROME, timeout: 60000,
}, async t => {
  const sources = Object.fromEntries(await Promise.all(['rope', 'crossSDPA'].map(async name =>
    [name, await readFile(new URL(`../shaders/inference/${name}.wgsl`, import.meta.url), 'utf8')])));
  const profile = await mkdtemp(path.join(tmpdir(), 'volvox-head-attention-chrome-'));
  const server = createServer((request, response) => { response.end('<!doctype html><title>WebGPU operator test</title>'); });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const child = spawn(process.env.VOLVOXAI_TEST_CHROME, ['--headless=new', '--no-sandbox', '--disable-gpu-sandbox', '--disable-dev-shm-usage',
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, `http://127.0.0.1:${server.address().port}`], { stdio: ['ignore', 'ignore', 'pipe'] });
  let socket, chromeLog = '';
  const abort = () => { socket?.close(); child.kill('SIGKILL'); };
  t.signal.addEventListener('abort', abort, { once: true });
  child.stderr.on('data', bytes => { chromeLog = (chromeLog + bytes).slice(-8000); });
  try {
    const endpoint = await new Promise((resolve, reject) => {
      let log = '';
      child.stderr.on('data', bytes => { log += bytes; const found = log.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (found) resolve(found[1]); });
      child.once('error', reject); child.once('exit', code => reject(new Error(`Chrome exited ${code}: ${log}`)));
    });
    let target;
    for (let attempt = 0; attempt < 100; attempt++) {
      target = (await (await fetch(`http://${new URL(endpoint).host}/json/list`)).json()).find(value =>
        value.type === 'page' && value.url.startsWith('http://127.0.0.1:'));
      if (target) break;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    assert.ok(target, 'Chrome page target');
    socket = new WebSocket(target.webSocketDebuggerUrl); await once(socket, 'open');
    let next = 0; const waiting = new Map();
    socket.addEventListener('message', event => { const message = JSON.parse(event.data); if (message.id) { waiting.get(message.id)?.resolve(message); waiting.delete(message.id); } });
    socket.addEventListener('close', () => { for (const promise of waiting.values()) promise.reject(new Error('Chrome connection closed')); waiting.clear(); });
    const send = (method, params) => new Promise((resolve, reject) => { const id = ++next; waiting.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
    for (let attempt = 0; attempt < 100; attempt++) {
      const ready = await send('Runtime.evaluate', { expression: 'location.href', returnByValue: true });
      if (ready.result?.result?.value?.startsWith('http://127.0.0.1:')) break;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const result = await send('Runtime.evaluate', { expression: `(${numericalShaders.toString()})(${JSON.stringify(sources)})`, awaitPromise: true, returnByValue: true });
    assert.equal(result.error, undefined, JSON.stringify(result.error));
    assert.equal(result.result?.exceptionDetails, undefined, `${JSON.stringify(result.result?.exceptionDetails)}\n${chromeLog}`);
    assert.equal(result.result.result.value.records.length, 42);
    const numerical = result.result.result.value;
    console.log(JSON.stringify({ adapter: numerical.adapter, cases: numerical.records.length,
      maximumAbsoluteError: Math.max(...numerical.records.map(record => record.maximumAbsoluteError)) }));
  } finally {
    t.signal.removeEventListener('abort', abort);
    socket?.close(); child.kill('SIGKILL'); server.close();
    if (child.exitCode === null) await once(child, 'exit');
    await rm(profile, { recursive: true, force: true });
  }
});
