#!/usr/bin/env node
// Raw authoritative-shader correctness and paired timing. This does not qualify
// the public runtime, whole models, or a device backend absent from the report.
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import process from 'node:process';

// Self-contained so the same test body executes in Chrome or pinned Deno.
export async function packedShaderCases(options) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const info = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']
    .map(key => [key, adapter.info[key]]));
  const software = !!adapter.info.isFallbackAdapter || /swiftshader|llvmpipe|software/i.test(JSON.stringify(info));
  if (options.requirePhysical && software) throw new Error(`Physical adapter required: ${JSON.stringify(info)}`);
  const timestamp = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({ requiredFeatures: timestamp ? ['timestamp-query'] : [] });
  const errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const report = { adapter: info, software_adapter: software, timestamp_query: timestamp,
    limits: { workgroup_storage_bytes: device.limits.maxComputeWorkgroupStorageSize },
    numerical: [], benchmarks: [], scope: 'raw packedLinear WGSL; resident buffers; no model or transport timing' };
  const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
    .map(value => value.toString(16).padStart(2, '0')).join('');
  const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  const f = Math.fround;
  const tiesEven = value => { const base = Math.floor(Math.abs(value)), fraction = Math.abs(value) - base;
    return Math.sign(value) * (base + Number(fraction > .5 || fraction === .5 && base % 2 === 1)); };
  const buffer = (data, usage) => {
    const result = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(result, 0, data); return result;
  };
  const pipelines = {};
  try {
    if (options.probeOnly) return report;
    for (const [name, code] of Object.entries(options.sources)) {
      const module = device.createShaderModule({ code, label: name });
      const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
      if (messages.length) throw new Error(`${name}: ${messages.map(message => message.message).join('; ')}`);
      pipelines[name] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    }
    function fixture(spec) {
      const { M, K, N, G, bits, pattern = 'normal', first = 0, count = M * N - first, mode = 8, bias = true } = spec;
      const x = Float32Array.from({ length: M * K }, (_, i) => ((i * 37 % 257) - 128) / 64);
      if (pattern === 'zero') x.fill(0);
      if (pattern === 'ties') for (let row = 0; row < M; row++) {
        for (let k = 0; k < K; k++) x[row * K + k] = (k % 2 ? -1 : 1) * ((k % 31) + .5);
        x[row * K] = 127;
      }
      if (pattern === 'subnormal') {
        const u = new Uint32Array(x.buffer);
        for (let i = 0; i < u.length; i++) u[i] = (i % 2 ? 0x80000000 : 0) | (i * 101 % 0x800000);
      }
      if (pattern === 'nonfinite' || pattern === 'mixed-nonfinite') {
        const u = new Uint32Array(x.buffer);
        for (let row = 0; row < M; row++) if (pattern === 'nonfinite' || row === 1)
          u[row * K + Math.min(K - 1, 65)] = [0x7fc01234, 0x7f800000, 0xff800000][row % 3];
      }
      const perWord = 32 / bits, words = Math.ceil(K / perWord), groups = Math.ceil(K / G), offset = 2 ** (bits - 1);
      const codes = Int32Array.from({ length: N * K }, (_, i) => (i * 11 + Math.floor(i / K) * 3) % (2 ** bits) - offset);
      const packed = new Uint32Array(N * words);
      for (let n = 0; n < N; n++) for (let k = 0; k < words * perWord; k++)
        packed[n * words + Math.floor(k / perWord)] |= ((k < K ? codes[n * K + k] : 0) + offset) << ((k % perWord) * bits);
      const scales = Float32Array.from({ length: N * groups }, (_, i) => (1 + i % 13) / 97);
      const biases = Float32Array.from({ length: N }, (_, i) => (i % 7 - 3) / 32);
      const initial = new Float32Array(M * N).fill(12345.25), p = new Uint32Array(32);
      p[0] = 1; p[1] = M; p[5] = K; p[6] = N; p[8] = bits; p[9] = G;
      p[17] = mode; p[21] = Number(bias); p[22] = count; p[23] = first;
      const owned = [x, packed, scales, biases, initial].map(data => buffer(data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC));
      owned.push(buffer(p, GPUBufferUsage.UNIFORM));
      const bindings = Object.fromEntries(Object.entries(pipelines).map(([name, pipeline]) => [name,
        device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: owned.map((b, binding) => ({ binding, resource: { buffer: b } })) })]));
      const staging = device.createBuffer({ size: initial.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      let query, resolved, queryRead;
      if (timestamp) {
        query = device.createQuerySet({ type: 'timestamp', count: 2 });
        resolved = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        queryRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      }
      function encode(name, repetitions, read) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass(timestamp ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(pipelines[name]); pass.setBindGroup(0, bindings[name]);
        for (let i = 0; i < repetitions; i++) pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        if (read) encoder.copyBufferToBuffer(owned[4], 0, staging, 0, initial.byteLength);
        if (timestamp) { encoder.resolveQuerySet(query, 0, 2, resolved, 0); encoder.copyBufferToBuffer(resolved, 0, queryRead, 0, 16); }
        return encoder.finish();
      }
      async function run(name, repetitions = 1, read = true) {
        if (read) device.queue.writeBuffer(owned[4], 0, initial);
        const command = encode(name, repetitions, read), start = performance.now();
        device.queue.submit([command]); await device.queue.onSubmittedWorkDone();
        const wall_ms = (performance.now() - start) / repetitions;
        let gpu_ms = null, output;
        if (timestamp) { await queryRead.mapAsync(GPUMapMode.READ); const ts = new BigUint64Array(queryRead.getMappedRange());
          gpu_ms = Number(ts[1] - ts[0]) / 1e6 / repetitions; queryRead.unmap(); }
        if (read) { await staging.mapAsync(GPUMapMode.READ); output = new Float32Array(staging.getMappedRange().slice(0)); staging.unmap(); }
        return { wall_ms, gpu_ms, output };
      }
      function oracle(index) {
        if (index < first || index >= first + count) return initial[index];
        const row = Math.floor(index / N), n = index % N;
        let maximum = 0;
        for (let k = 0; k < K; k++) { const value = x[row * K + k]; if (!Number.isFinite(value)) return NaN; maximum = Math.max(maximum, Math.abs(value)); }
        const as = maximum === 0 ? 1 : Math.max(f(maximum / 127), 2 ** -126);
        let sum = 0;
        if (mode === 8) for (let group = 0; group < groups; group++) {
          let dot = 0;
          for (let k = group * G; k < Math.min(K, (group + 1) * G); k++) dot += Math.max(-127, Math.min(127, tiesEven(f(x[row * K + k] / as)))) * codes[n * K + k];
          sum = f(sum + f(dot * f(as * scales[n * groups + group])));
        } else if (mode === 32 || mode === 16) for (let k = 0; k < K; k++) {
          const a = x[row * K + k], w = f(codes[n * K + k] * scales[n * groups + Math.floor(k / G)]);
          sum = f(sum + f(mode === 16 ? Math.f16round(a) * Math.f16round(w) : a * w));
        }
        else throw new Error('Oracle mode unsupported');
        return bias ? f(sum + biases[n]) : sum;
      }
      return { run, oracle, initial, x, packed, scales, biases, parameters: p,
        close() { for (const b of [...owned, staging, resolved, queryRead]) b?.destroy(); query?.destroy(); } };
    }
    async function check(spec, exhaustive = true) {
      const state = fixture(spec);
      try {
        const actual = await state.run('candidate'), before = pipelines.baseline ? await state.run('baseline') : null;
        const words = new Uint32Array(actual.output.buffer), old = before && new Uint32Array(before.output.buffer);
        let maxError = 0;
        for (let i = 0; i < actual.output.length; i++) {
          if (old && words[i] !== old[i] && !(Number.isNaN(actual.output[i]) && Number.isNaN(before.output[i])))
            throw new Error(`Baseline mismatch ${JSON.stringify(spec)} at ${i}: ${actual.output[i]} / ${before.output[i]}`);
          if (!exhaustive && i % Math.max(1, Math.floor(actual.output.length / 11)) !== 0 && i !== actual.output.length - 1) continue;
          const expected = state.oracle(i), error = Math.abs(actual.output[i] - expected);
          if (Number.isNaN(expected) ? !Number.isNaN(actual.output[i]) : !Number.isFinite(actual.output[i]) || error > 3e-5 * (1 + Math.abs(expected)))
            throw new Error(`Oracle mismatch ${JSON.stringify(spec)} at ${i}: ${actual.output[i]} / ${expected}`);
          if (Number.isFinite(error)) maxError = Math.max(maxError, error);
        }
        if (pipelines.control) {
          const control = await state.run('control'), controlWords = new Uint32Array(control.output.buffer);
          for (let i = 0; i < words.length; i++) if (controlWords[i] !== old[i] && !(Number.isNaN(control.output[i]) && Number.isNaN(before.output[i])))
            throw new Error(`Identical-shader control output mismatch at ${i}`);
        }
        return { ...spec, baseline_bytes_equal: before ? true : null, control_bytes_equal: pipelines.control ? true : null, maximum_absolute_error: maxError,
          output_sha256: await hash(actual.output.buffer), input_sha256: await hash(state.x.buffer),
          packed_weight_sha256: await hash(state.packed.buffer), scale_sha256: await hash(state.scales.buffer),
          bias_sha256: await hash(state.biases.buffer), parameters_sha256: await hash(state.parameters.buffer) };
      } finally { state.close(); }
    }
    for (const bits of [4, 8]) for (const G of [32, 64, 128, 256]) {
      for (const K of [1, G - 1, G, G + 1, 513])
        report.numerical.push(await check({ M: 3, K, N: 64, G, bits, bias: K % 2 === 0 }));
      for (const [N, first, count] of [[128, 64, 137], [128, 0, 1], [128, 192, 1], [128, 1, 189], [65, 64, 129], [7, 2, 11]])
        report.numerical.push(await check({ M: 3, K: G + 3, N, G, bits, first, count }));
      for (const pattern of ['zero', 'ties', 'subnormal', 'nonfinite', 'mixed-nonfinite'])
        report.numerical.push(await check({ M: 3, K: G + 5, N: 128, G, bits, pattern, first: 64, count: 151 }));
    }
    for (const bits of [4, 8]) for (const mode of [16, 32])
      report.numerical.push(await check({ M: 2, K: 35, N: 64, G: 32, bits, mode }));
    // Exercise both sides of MR admission, including its final short row tile
    // and aligned nonzero first. Small cases alone only cover shared_i8.
    for (const bits of [4, 8]) for (const G of [32, 64, 128, 256]) {
      report.numerical.push(await check({ M: 97, K: G + 5, N: 1024, G, bits }));
      report.numerical.push(await check({ M: 98, K: G + 5, N: 1024, G, bits,
        first: 1024, count: 97 * 1024, pattern: 'mixed-nonfinite' }));
    }
    for (const bits of [4, 8]) for (const pattern of ['zero', 'ties', 'subnormal'])
      report.numerical.push(await check({ M: 96, K: 35, N: 1024, G: 32, bits, pattern }));
    for (const [M, N, first, count] of [[63, 2048, 0, 63 * 2048], [64, 1536, 0, 64 * 1536],
      [64, 1536, 64, 64 * 1536 - 64], [307, 320, 0, 307 * 320], [308, 320, 0, 308 * 320]])
      report.numerical.push(await check({ M, K: 35, N, G: 32, bits: 4, first, count }));
    if (options.benchmarks) {
      if (!pipelines.baseline) throw new Error('Paired timings require --baseline');
      const specs = options.benchmarkSpecs || [[1, 320, 320], [1, 1280, 320], [402, 320, 320], [402, 320, 960], [402, 1280, 320]]
        .flatMap(([M, K, N]) => [4, 8].map(bits => ({ M, K, N, G: 32, bits })));
      for (const spec of specs) {
          const checked = await check(spec, false), state = fixture(spec), samples = [], repetitions = spec.M === 1 ? 16 : 1;
          const order = pipelines.control ? ['baseline', 'control', 'candidate'] : ['baseline', 'candidate'];
          try {
            for (let i = 0; i < 2; i++) for (const name of order) await state.run(name, repetitions, false);
            for (let pair = 0; pair < options.pairs; pair++) {
              const sample = { pair, order: pair % 2 ? [...order].reverse() : order };
              for (const name of sample.order) sample[name] = await state.run(name, repetitions, false);
              samples.push(sample);
            }
            const metric = timestamp ? 'gpu_ms' : 'wall_ms';
            report.benchmarks.push({ ...checked, repetitions, samples, metric,
              baseline_median_ms: median(samples.map(s => s.baseline[metric])),
              candidate_median_ms: median(samples.map(s => s.candidate[metric])),
              control_median_ms: pipelines.control ? median(samples.map(s => s.control[metric])) : null,
              identical_control_paired_ratio: pipelines.control ? median(samples.map(s => s.baseline[metric] / s.control[metric])) : null,
              paired_speedup_median: median(samples.map(s => s.baseline[metric] / s.candidate[metric])) });
          } finally { state.close(); }
      }
    }
    if (errors.length) throw new Error(errors.join('; '));
    report.passed = true; return report;
  } finally { device.destroy(); }
}

export async function runChrome(options, executable, software = false) {
  const profile = await mkdtemp(path.join(tmpdir(), 'volvox-packed-webgpu-'));
  const server = createServer((request, response) => response.end('<!doctype html><title>PackedLinear shader validation</title>'));
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const child = spawn(executable, ['--headless=new', '--no-sandbox', '--disable-gpu-sandbox', '--disable-dev-shm-usage',
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', ...(software ? ['--use-webgpu-adapter=swiftshader'] : ['--use-angle=vulkan', '--disable-vulkan-surface']),
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, `http://127.0.0.1:${server.address().port}`], { stdio: ['ignore', 'ignore', 'pipe'] });
  let socket, deadline, log = '';
  child.stderr.on('data', b => { log = (log + b).slice(-16000); });
  try {
    const endpoint = await new Promise((resolve, reject) => {
      child.stderr.on('data', () => { const match = log.match(/DevTools listening on (ws:\/\/\S+)/); if (match) resolve(match[1]); });
      child.once('error', reject); child.once('exit', code => reject(new Error(`Chrome exit ${code}: ${log}`)));
    });
    let target;
    for (let i = 0; i < 100 && !target; i++) {
      target = (await (await fetch(`http://${new URL(endpoint).host}/json/list`)).json()).find(t => t.type === 'page' && t.url.startsWith('http://127.0.0.1:'));
      if (!target) await new Promise(r => setTimeout(r, 20));
    }
    if (!target) throw new Error(`Chrome target missing: ${log}`);
    socket = new WebSocket(target.webSocketDebuggerUrl); await once(socket, 'open');
    let next = 0; const pending = new Map();
    socket.addEventListener('message', e => { const message = JSON.parse(e.data); if (message.id) { pending.get(message.id)?.resolve(message); pending.delete(message.id); } });
    socket.addEventListener('close', () => { for (const p of pending.values()) p.reject(new Error(`Chrome disconnected: ${log}`)); pending.clear(); });
    const send = (method, params) => new Promise((resolve, reject) => { const id = ++next; pending.set(id, { resolve, reject }); socket.send(JSON.stringify({ id, method, params })); });
    const version = await send('Browser.getVersion');
    for (let i = 0; i < 100; i++) {
      const ready = await send('Runtime.evaluate', { expression: 'location.href', returnByValue: true });
      if (ready.result?.result?.value?.startsWith('http://127.0.0.1:')) break;
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const result = await Promise.race([
      send('Runtime.evaluate', { expression: `(${packedShaderCases.toString()})(${JSON.stringify(options)})`, awaitPromise: true, returnByValue: true }),
      new Promise((_, reject) => { deadline = setTimeout(() => reject(new Error('Packed shader check exceeded 180 seconds')), 180000); }),
    ]);
    if (result.error || result.result?.exceptionDetails) throw new Error(`${JSON.stringify(result.error || result.result.exceptionDetails)}\n${log}`);
    return { ...result.result.result.value, browser: version.result };
  } finally {
    clearTimeout(deadline);
    socket?.close(); child.kill('SIGTERM'); await once(child, 'exit').catch(() => {});
    server.close(); await once(server, 'close'); await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  }
}

async function main() {
  const args = process.argv.slice(2), value = (flag, fallback) => args.includes(flag) ? args[args.indexOf(flag) + 1] : fallback;
  const valued = new Set(['--shader', '--baseline', '--chrome', '--out', '--pairs', '--benchmark-cases']);
  const flags = new Set(['--benchmark', '--control', '--probe', '--software', '--allow-software']);
  for (let i = 0; i < args.length; i++) {
    if (valued.has(args[i])) { if (!args[++i] || args[i].startsWith('--')) throw new Error('Option requires a value'); }
    else if (!flags.has(args[i])) throw new Error(`Unknown argument ${args[i]}`);
  }
  const candidate = value('--shader', fileURLToPath(new URL('../../shaders/inference/packedLinear.wgsl', import.meta.url)));
  const baseline = value('--baseline');
  const options = { sources: { candidate: await readFile(candidate, 'utf8') }, benchmarks: args.includes('--benchmark'),
    pairs: Number(value('--pairs', '5')), probeOnly: args.includes('--probe'), requirePhysical: !args.includes('--allow-software') };
  if (!Number.isSafeInteger(options.pairs) || options.pairs < 1 || options.pairs > 100) throw new Error('--pairs must be in 1..100');
  if (baseline) options.sources.baseline = await readFile(baseline, 'utf8');
  if (args.includes('--control')) {
    if (!baseline) throw new Error('--control requires --baseline');
    options.sources.control = options.sources.baseline;
  }
  if (value('--benchmark-cases')) options.benchmarkSpecs = JSON.parse(await readFile(value('--benchmark-cases'), 'utf8'));
  const hostSamples = [], outstanding = new Set();
  const sampleHost = async () => {
    try { const [stat, load] = await Promise.all([readFile('/proc/stat', 'utf8'), readFile('/proc/loadavg', 'utf8')]);
      hostSamples.push({ time: new Date().toISOString(), cpu_ticks: stat.split('\n')[0].trim().split(/\s+/).slice(1).map(Number), loadavg: load.trim().split(/\s+/).slice(0, 3).map(Number) });
    } catch { /* Other hosts lack Linux system counters; leave coverage empty. */ }
  };
  await sampleHost();
  const sampler = setInterval(() => { const pending = sampleHost(); outstanding.add(pending); pending.finally(() => outstanding.delete(pending)); }, 1000);
  let report;
  try {
  report = typeof Deno !== 'undefined'
    ? { ...await packedShaderCases(options), deno: Deno.version }
    : await runChrome(options, value('--chrome', process.env.VOLVOXAI_TEST_CHROME || '/usr/bin/google-chrome'), args.includes('--software'));
  } finally { clearInterval(sampler); await Promise.allSettled([...outstanding]); await sampleHost(); }
  report.host_samples = hostSamples;
  report.host_quiet_verified = false;
  report.harness_sha256 = createHash('sha256').update(await readFile(fileURLToPath(import.meta.url))).digest('hex');
  report.shader_sha256 = Object.fromEntries(Object.entries(options.sources).map(([name, code]) => [name, createHash('sha256').update(code).digest('hex')]));
  report.reproduction = [typeof Deno !== 'undefined' ? 'deno run --no-config --unstable-webgpu --allow-read --allow-write --allow-env --allow-ffi' : 'node',
    path.relative(process.cwd(), process.argv[1]), ...args];
  const out = value('--out'); if (out) await writeFile(out, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ passed: report.passed, adapter: report.adapter, cases: report.numerical.length,
    benchmarks: report.benchmarks.map(({ M, K, N, bits, baseline_median_ms, candidate_median_ms, paired_speedup_median }) =>
      ({ M, K, N, bits, baseline_median_ms, candidate_median_ms, paired_speedup_median })), out }));
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main();
