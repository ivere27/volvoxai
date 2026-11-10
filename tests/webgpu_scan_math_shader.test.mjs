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
async function numericalShaders(sources, unaryModes) {
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
  try {
    {
      const values = [-Infinity, -3.4028234663852886e38, -100, -93.25194549560547,
        -10.001, -10, -9, -1, -0, 0, 1, 9, 10, 10.001, 92.31864929199219,
        93.25194549560547, 100, 3.4028234663852886e38, Infinity, NaN];
      const input = Float32Array.from({length: 130}, (_, i) => values[i % values.length]);
      const actual = await dispatch('tanh', [input], Uint32Array.of(input.length), [1, 3, 1], input.length);
      let maximum = 0;
      for (let i = 0; i < input.length; ++i) {
        const expected = Math.fround(Math.tanh(input[i]));
        if (Number.isNaN(expected)) {
          if (!Number.isNaN(actual[i])) throw new Error(`Tanh/${i}: expected NaN`);
        } else {
          const error = Math.abs(actual[i]-expected); maximum = Math.max(maximum,error);
          if (!Number.isFinite(actual[i]) || error > 2e-6 ||
              (Math.abs(input[i]) >= 10 && actual[i] !== expected) ||
              (input[i] === 0 && !Object.is(actual[i], expected)))
            throw new Error(`Tanh/${i}: ${actual[i]} vs ${expected}`);
        }
      }
      records.push({name: 'Tanh/finite-tails-and-special-values', elements: input.length, maximumAbsoluteError: maximum});
    }
    {
      const input = Float32Array.from([-1e30, -100, -14, -10, -3, -1, -.01, 0, .01, 1, 3, 10, 14, 100, 1e30]);
      const actual = await dispatch('gELU', [input], Uint32Array.of(input.length, 1, 0, 0), [1, 1, 1], input.length);
      let maximum = 0;
      for (let i = 0; i < input.length; ++i) {
        const x = input[i], expected = Math.fround(.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x))));
        const error = Math.abs(actual[i]-expected); maximum = Math.max(maximum,error);
        if (!Number.isFinite(actual[i]) || error > 2e-6*Math.abs(expected)+2e-7)
          throw new Error(`GELU/tanh/${i}: ${actual[i]} vs ${expected}`);
      }
      records.push({name: 'GELU/tanh-large-finite-inputs', elements: input.length, maximumAbsoluteError: maximum});
      const backward = await dispatch('activationBackward', [input, actual, new Float32Array(input.length).fill(1)],
        Uint32Array.of(input.length, 9, 0, 0), [1, 1, 1], input.length);
      let backwardMaximum = 0;
      for (let i = 0; i < input.length; ++i) {
        const x = input[i], t = Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x*x*x));
        const expected = x >= 10 ? 1 : x <= -10 ? 0 :
          .5*(1+t)+.5*x*(1-t*t)*Math.sqrt(2/Math.PI)*(1+3*.044715*x*x);
        const error = Math.abs(backward[i]-expected); backwardMaximum = Math.max(backwardMaximum,error);
        if (!Number.isFinite(backward[i]) || error > 2e-6)
          throw new Error(`GELU/backward/${i}: ${backward[i]} vs ${expected}`);
      }
      records.push({name: 'GELU/tanh-backward-large-finite-inputs', elements: input.length, maximumAbsoluteError: backwardMaximum});
    }
    {
      const values = [-Infinity, -3.4028234663852886e38, -1, -0, 0, 1, 3.4028234663852886e38, Infinity, NaN];
      const input = Float32Array.from({ length: 130 }, (_, i) => values[i % values.length]);
      const actual = await dispatch('reLU', [input], Uint32Array.of(input.length), [1, 3, 1], input.length);
      for (let i = 0; i < input.length; i++) {
        const expected = input[i] > 0 ? input[i] : 0;
        if (!Object.is(actual[i], expected)) throw new Error(`ReLU/${i}: ${actual[i]} vs ${expected}`);
      }
      records.push({ name: 'ReLU/two-dimensional-grid-and-nonfinite', elements: input.length, maximumAbsoluteError: 0 });
    }
    for (const [minimum, maximum] of [[0, 2049], [-2, 3]]) {
      const input = Float32Array.from([-Infinity, -100, -2, -1, 0, 1, 3, 2049, 10000, Infinity]);
      const words = Uint32Array.of(input.length, 0, 0);
      new Float32Array(words.buffer).set([minimum, maximum], 1);
      const actual = await dispatch('clip', [input], words, [1, 1, 1], input.length);
      for (let i = 0; i < input.length; i++) {
        const expected = Math.min(Math.max(input[i], minimum), maximum);
        if (actual[i] !== expected) throw new Error(`Clip/${i}: ${actual[i]} vs ${expected}`);
      }
      records.push({ name: `Clip/${minimum}/${maximum}`, elements: input.length, maximumAbsoluteError: 0 });
    }
    for (const operation of [0, 1, 2]) {
      const input = operation === 0
        ? Float32Array.of(-104, -90, -88, -2, -0, 0, 1, 20, 88, 88.5, 89, Infinity, -Infinity, NaN)
        : Float32Array.of(-3.4028234663852886e38, -1e20, -1e10, -1000, -10, -Math.PI, -1, -0, 0, 1, Math.PI, 10, 1000, 1e10, 1e20, 3.4028234663852886e38, Infinity, -Infinity, NaN);
      const expected = Float32Array.from(input, operation === 0 ? Math.exp : operation === 1 ? Math.sin : Math.cos);
      const actual = await dispatch('unaryMath', [input], Uint32Array.of(input.length, unaryModes[operation], 0, 0), [1, 1, 1], input.length);
      let maximum = 0;
      for (let i = 0; i < expected.length; i++) {
        if (Number.isNaN(expected[i])) {
          if (!Number.isNaN(actual[i])) throw new Error(`unary/${operation}/${i}: expected NaN`);
        } else if (!Number.isFinite(expected[i])) {
          if (actual[i] !== expected[i]) throw new Error(`unary/${operation}/${i}: ${actual[i]} vs ${expected[i]}`);
        } else if (Math.abs(expected[i]) < 2 ** -126 && actual[i] === 0) {
          // Devices may flush subnormal arithmetic; normal finite Exp must not saturate.
        } else if (Math.abs(actual[i] - expected[i]) > 3e-5 * Math.abs(expected[i]) + 1e-7) {
          throw new Error(`unary/${operation}/${i}: ${actual[i]} vs ${expected[i]}`);
        }
        if (Number.isFinite(expected[i])) maximum = Math.max(maximum, Math.abs(actual[i] - expected[i]));
      }
      records.push({ name: `unary/${operation}`, elements: input.length, maximumAbsoluteError: maximum });
    }
    const roundEven = x => {
      if (!(Math.abs(x) < 8388608)) return x;
      const a = Math.abs(x), base = Math.floor(a), part = a - base;
      const value = part < .5 ? base : part > .5 ? base + 1 : base + (base % 2);
      return x < 0 || Object.is(x, -0) ? -value : value;
    };
    const erfReference = x => {
      if (x === 0 || Number.isNaN(x)) return x;
      const a = Math.abs(x);
      if (a >= 6) return x < 0 ? -1 : 1;
      let term = 1, sum = 1;
      for (let n = 1; n < 150; ++n) { term *= 2 * a*a / (2*n+1); sum += term; }
      return Math.sign(x) * 2 / Math.sqrt(Math.PI) * a * Math.exp(-a*a) * sum;
    };
    const functions = [Math.sqrt, Math.log, erfReference, Math.floor, Math.ceil,
      roundEven, Math.atan, x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))), x => -x];
    for (let operation = 3; operation < unaryModes.length; ++operation) {
      const input = Float32Array.from([-100, -40, -20, -5, -2.5, -1.5, -.5, -.001, -0, 0,
        .001, .5, 1, 1.5, 2.5, 5, 20, 100, 8388608, Infinity, -Infinity, NaN]);
      const expected = Float32Array.from(input, functions[operation-3]);
      const actual = await dispatch('unaryMath', [input], Uint32Array.of(input.length, unaryModes[operation], 0, 0), [1,1,1], input.length);
      let maximum = 0;
      for (let i = 0; i < input.length; ++i) {
        if (Number.isNaN(expected[i])) {
          if (!Number.isNaN(actual[i])) throw new Error(`unary/${operation}/${i}: expected NaN, got ${actual[i]}`);
        } else if (!Number.isFinite(expected[i])) {
          if (actual[i] !== expected[i]) throw new Error(`unary/${operation}/${i}: ${actual[i]} vs ${expected[i]}`);
        } else {
          const error = Math.abs(actual[i] - expected[i]); maximum = Math.max(maximum, error);
          if (error > 2e-6 * Math.abs(expected[i]) + 3e-7)
            throw new Error(`unary/${operation}/${i}: ${actual[i]} vs ${expected[i]}`);
          if ([6,7,8,11].includes(operation) && actual[i] !== expected[i])
            throw new Error(`exact unary/${operation}/${i}: ${actual[i]} vs ${expected[i]}`);
        }
      }
      records.push({name: `unary/${operation}`, elements: input.length, maximumAbsoluteError: maximum});
    }
    for (const shape of [[2, 5, 3], [130, 4], [1, 1025], [1, 3]]) {
      const count = shape.reduce((a, b) => a * b, 1);
      for (let axis = 0; axis < shape.length; axis++) for (const integer of [0, 1])
        for (const exclusive of [0, 1]) for (const reverse of [0, 1]) {
          const Ctor = integer ? Int32Array : Float32Array;
          const values = integer ? [2147483647, 1, -1, -2147483648, 17, -23]
            : [16777216, 1, -16777216, .125, -.25, .5];
          const input = Ctor.from({ length: count }, (_, i) => values[i % values.length]);
          const expected = new Ctor(count), length = shape[axis];
          const inner = shape.slice(axis + 1).reduce((a, b) => a * b, 1), scans = count / length;
          for (let lane = 0; lane < scans; lane++) {
            const base = Math.floor(lane / inner) * length * inner + lane % inner;
            let sum = 0;
            for (let t = 0; t < length; t++) {
              const index = base + (reverse ? length - 1 - t : t) * inner;
              const previous = sum;
              sum = integer ? (sum + input[index]) | 0 : Math.fround(sum + input[index]);
              expected[index] = exclusive ? previous : sum;
            }
          }
          const groups = scans > 64 ? [1, Math.ceil(scans / 64), 1] : [1, 1, 1];
          const params = Uint32Array.of(scans, length, inner, exclusive, reverse, integer, 0, 0);
          const raw = await dispatch('cumSum', [input], params, groups, count);
          const actual = integer ? new Int32Array(raw.buffer) : raw;
          for (let i = 0; i < count; i++) if (actual[i] !== expected[i])
            throw new Error(`CumSum/${shape}/${axis}/${integer}/${exclusive}/${reverse}[${i}]: ${actual[i]} vs ${expected[i]}`);
          records.push({ name: `CumSum/${shape}/${axis}/${integer}/${exclusive}/${reverse}`, elements: count, maximumAbsoluteError: 0 });
        }
    }
    for (const [area, channels, groups, offset] of [[8192, 4, 4, 10000], [7921, 128, 128, 0], [3, 4, 2, 1]]) {
      const count = area * channels, perGroup = channels / groups, epsilon = 1e-5;
      const input = Float32Array.from({length: count}, (_, index) => offset === 10000
        ? offset + (Math.floor(index / channels) % 2 ? .25 : -.25)
        : Math.sin(index * .17) * .3 + offset);
      const weight = Float32Array.from({length: channels}, (_, i) => .5 + i * .125);
      const bias = Float32Array.from({length: channels}, (_, i) => -.2 + i * .03);
      const expected = new Float32Array(count);
      for (let group = 0; group < groups; group++) {
        let sum = 0, variance = 0;
        for (let spatial = 0; spatial < area; spatial++) for (let local = 0; local < perGroup; local++)
          sum += input[spatial * channels + group * perGroup + local];
        const mean = sum / (area * perGroup);
        for (let spatial = 0; spatial < area; spatial++) for (let local = 0; local < perGroup; local++)
          variance += (input[spatial * channels + group * perGroup + local] - mean) ** 2;
        const inverse = 1 / Math.sqrt(variance / (area * perGroup) + epsilon);
        for (let spatial = 0; spatial < area; spatial++) for (let local = 0; local < perGroup; local++) {
          const channel = group * perGroup + local, index = spatial * channels + channel;
          expected[index] = (input[index] - mean) * inverse * weight[channel] + bias[channel];
        }
      }
      const params = Uint32Array.of(1, 1, area, channels, groups, 1, 0, 0);
      new Float32Array(params.buffer)[6] = epsilon;
      const actual = await dispatch('groupNorm', [input, weight, bias], params, [Math.ceil(groups / 64), 1, 1], count);
      let maximum = 0;
      for (let i = 0; i < count; i++) {
        const error = Math.abs(actual[i] - expected[i]); maximum = Math.max(maximum, error);
        if (!Number.isFinite(actual[i]) || error > 3e-5 * Math.abs(expected[i]) + 2e-6)
          throw new Error(`GroupNorm/${area}/${channels}/${groups}/${offset}[${i}]: ${actual[i]} vs ${expected[i]}`);
      }
      records.push({name: `GroupNorm/${area}/${channels}/${groups}/${offset}`, elements: count, maximumAbsoluteError: maximum});
    }
    if (errors.length) throw new Error(errors.join('; '));
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
      device: adapter.info.device, description: adapter.info.description }, records };
  } finally { device.destroy(); }
}

test('authoritative WebGPU unary, typed CumSum and centered GroupNorm numerics', {
  skip: !process.env.VOLVOXAI_TEST_CHROME, timeout: 60000,
}, async t => {
  const sources = Object.fromEntries(await Promise.all(['unaryMath', 'cumSum', 'groupNorm', 'reLU', 'clip', 'tanh', 'gELU'].map(async name =>
    [name, await readFile(new URL(`../shaders/inference/${name}.wgsl`, import.meta.url), 'utf8')])));
  sources.activationBackward = await readFile(new URL('../shaders/training/activationBackward.wgsl', import.meta.url), 'utf8');
  // Use the generated runtime selector words, so this numerical test covers
  // the operator-to-shader mapping as well as the shader itself.
  const table = await readFile(new URL('../native/src/backends/webgpu_dispatch.inc', import.meta.url), 'utf8');
  const unaryModes = ['EXP', 'SIN', 'COS', 'SQRT', 'LOG', 'ERF', 'FLOOR', 'CEIL', 'ROUND', 'ATAN', 'SOFTPLUS', 'NEG'].map(operator => {
    const entry = table.match(new RegExp(`\\{ VX_OP_${operator}, VX_SHADER_INFERENCE_UNARY_MATH,\\s*\\w+, \\d+,\\s*(\\w+), 4,`));
    assert.ok(entry, `${operator} dispatch`);
    const parameters = table.match(new RegExp(`${entry[1]}\\[\\] = \\{([\\s\\S]*?)\\n\\};`));
    assert.ok(parameters, `${operator} parameters`);
    const selector = [...parameters[1].matchAll(/\{ ([^}]+) \}/g)][1][1].split(',');
    assert.equal(selector[0].trim(), 'VX_WEBGPU_VALUE_CONST');
    return Number(selector[3]);
  });
  const profile = await mkdtemp(path.join(tmpdir(), 'volvox-scan-math-chrome-'));
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
    const result = await send('Runtime.evaluate', { expression: `(${numericalShaders.toString()})(${JSON.stringify(sources)}, ${JSON.stringify(unaryModes)})`, awaitPromise: true, returnByValue: true });
    assert.equal(result.error, undefined, JSON.stringify(result.error));
    assert.equal(result.result?.exceptionDetails, undefined, `${JSON.stringify(result.result?.exceptionDetails)}\n${chromeLog}`);
    assert.equal(result.result.result.value.records.length, 93);
    const numerical = result.result.result.value;
    console.log(JSON.stringify({ adapter: numerical.adapter, cases: numerical.records.length,
      groupNormMaximumAbsoluteError: Math.max(...numerical.records.filter(record => record.name.startsWith('GroupNorm/')).map(record => record.maximumAbsoluteError)),
      maximumAbsoluteError: Math.max(...numerical.records.map(record => record.maximumAbsoluteError)) }));
  } finally {
    t.signal.removeEventListener('abort', abort);
    socket?.close(); child.kill('SIGKILL'); server.close();
    if (child.exitCode === null) await once(child, 'exit');
    await rm(profile, { recursive: true, force: true });
  }
});
