import { declareStateless, declareIncremental, graphRequest, activeLengths } from './parity/external/execution_fixture.mjs';
import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
import {safetensors} from '../tools/proto_fixture.mjs';
import { FullEngineHost } from '../ts/full.js';
/**
 * The device bridge, driven end to end without a device.
 *
 * `gpu_bridge_contract` proves the two sides agree on the wire. This proves
 * the wire carries enough: the engine plans a real node, the host executes it
 * from the descriptor alone, and the result reaches the caller through the
 * ordinary proto API.
 *
 * The host here is a CPU interpreter rather than a GPU. That is the point --
 * it can only produce the right answer if the descriptor fully describes the
 * work, so a missing binding or a wrong workgroup count fails here instead of
 * on hardware nobody in CI has. What it deliberately does not check is the
 * WGSL itself; that is a shader question, and `test_model_corpus` is where the
 * numbers are compared against native.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { GPU_LIMIT_NAMES } from '../ts/generated/gpuBridge.js';

import {
  SHADER_INFERENCE_ADD_RELU,
  SHADER_INFERENCE_ARG_MAX_F32_I32_NATIVE,
  SHADER_INFERENCE_CROSS_SDPA,
  SHADER_INFERENCE_DECODE_KEEP_MASK,
  SHADER_INFERENCE_EMBEDDING,
  SHADER_INFERENCE_AVERAGE_POOL2_D,
  SHADER_INFERENCE_DIV,
  SHADER_INFERENCE_MUL,
  SHADER_INFERENCE_MAX_POOL2_D,
  SHADER_INFERENCE_RE_LU,
  SHADER_INFERENCE_ROW_INDEX_TRANSFER,
  SHADER_INFERENCE_SI_LU,
  SHADER_INFERENCE_SIGMOID,
  SHADER_INFERENCE_SUB,
  SHADER_INFERENCE_TANH,
  SHADER_NAMES} from '../ts/generated/shaderCatalog.js';

/*
 * What each uniform-elementwise shader computes, as the host would see it.
 *
 * These mirror the WGSL rather than the engine's CPU kernels on purpose: the
 * question this test answers is whether the descriptor names the right shader
 * for the operator, and a reference taken from the shader is what makes a
 * wrong id visible instead of merely a wrong number.
 */
/* What each binary shader computes. ADD_RELU is handled separately because it
 * reads a second uniform field rather than a different operator. */
const BINARY_OPS = {
  [SHADER_INFERENCE_ADD_RELU]: (a, b) => a + b,
  [SHADER_INFERENCE_SUB]: (a, b) => a - b,
  [SHADER_INFERENCE_MUL]: (a, b) => a * b,
  [SHADER_INFERENCE_DIV]: (a, b) => a / b,
};

const UNIFORM_ELEMENTWISE = {
  [SHADER_INFERENCE_RE_LU]: (x) => (x > 0 ? x : 0),
  [SHADER_INFERENCE_SIGMOID]: (x) => 1 / (1 + Math.exp(-x)),
  [SHADER_INFERENCE_TANH]: (x) => {
    const e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  },
  [SHADER_INFERENCE_SI_LU]: (x) => x * (1 / (1 + Math.exp(-x))),
};

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const version = JSON.parse(
  fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const FULL_WASM = path.join(ROOT, 'dist', version, 'volvoxai.wasm');

const N = 1, H = 4, W = 4, C = 2, KY = 2, KX = 2, SY = 2, SX = 2;
const OUT_H = 2, OUT_W = 2;

/** The reference the shader is expected to reproduce, in NHWC. */
function maxPool2dReference(input) {
  const out = new Float32Array(N * OUT_H * OUT_W * C);
  for (let oy = 0; oy < OUT_H; oy++) {
    for (let ox = 0; ox < OUT_W; ox++) {
      for (let c = 0; c < C; c++) {
        let best = -Infinity;
        for (let ky = 0; ky < KY; ky++) {
          for (let kx = 0; kx < KX; kx++) {
            const y = oy * SY + ky;
            const x = ox * SX + kx;
            if (y >= H || x >= W) continue;
            best = Math.max(best, input[((y * W) + x) * C + c]);
          }
        }
        out[((oy * OUT_W) + ox) * C + c] = best;
      }
    }
  }
  return out;
}

/**
 * A host that executes the descriptor on the CPU.
 *
 * It reads the same header, variants, bindings and uniform bytes a real
 * WebGPU host reads, and refuses anything it was not told.
 */
/** VxGpuBinding is five words: slot, host_ptr, offset, bytes, writes. */
const BINDING_WORDS = 5;
/** VxGpuDispatch header: counts, params bytes, node index, params slot. */
const HEADER_WORDS = 5;

const DEVICE_LIMITS = {
  maxBufferSize: 0xfffffffc, maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxUniformBufferBindingSize: 65536, maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12, maxBindingsPerBindGroup: 1000, maxBindGroups: 4,
  minStorageBufferOffsetAlignment: 256, maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256, maxComputeWorkgroupSizeZ: 64, maxComputeWorkgroupsPerDimension: 65535,
};

function createSimulatingBridge(log, limits = DEVICE_LIMITS) {
  const spans = new Map();
  /* Spans already asked for once. A real device answers the first readback
   * with "not yet", so this does too -- otherwise the retry path would never
   * be reached by any test that has no GPU. */
  const jobs = new Map();
  const timingTickets = new Map();
  let timing = false;
  let nextTicket = 1;
  let wakeup = () => {};
  // labels: each vx_gpu_debug_label call; dispatchLabels: the label per dispatch.
  const stats = { pending: 0, calls: [], labels: [], dispatchLabels: [], snapshotSizes: [], readbackSizes: [] };
  let label = null;
  let recorded = null;
  let memory = null;
  const words = () => new Uint32Array(memory.buffer);
  const bytes = (at, length) => new Uint8Array(memory.buffer, at, length);

  const imports = {
    vx_gpu_memory_start: () => 0,
    vx_gpu_memory_stop: () => {},
    vx_gpu_available: () => 1,
    vx_gpu_limits(destination, size) {
      if (size !== GPU_LIMIT_NAMES.length * 4) return 0;
      new Uint32Array(memory.buffer, destination, GPU_LIMIT_NAMES.length)
        .set(GPU_LIMIT_NAMES.map((name) => limits[name]));
      return 1;
    },
    vx_gpu_ensure(hostPointer, size) {
      if (!hostPointer || size <= 0) return 0;
      let span = spans.get(hostPointer);
      if (!span || span.bytes.length !== size) {
        span = { bytes: new Uint8Array(size), deviceDirty: false };
        spans.set(hostPointer, span);
      }
      /* The same rule the real host follows: upload unless the device wrote
       * this span and the host has not read it back. Modelling it here is what
       * lets a stale-input bug fail in CI instead of only on hardware. */
      if (!span.deviceDirty) span.bytes.set(bytes(hostPointer, size));
      return 1;
    },
    vx_gpu_release(hostPointer) { spans.delete(hostPointer); },
    vx_gpu_invalidate(hostPointer) { const span = spans.get(hostPointer); if (span) span.deviceDirty = false; },
    vx_gpu_begin() { recorded = recorded ?? []; },
    vx_gpu_begin_activity() { recorded = recorded ?? []; },
    vx_gpu_await_read() { return -1; },
    vx_gpu_await_release() {},
    vx_gpu_begin_trace() {
      recorded = recorded ?? [];
      if (!timing) return 0;
      const ticket = nextTicket++; timingTickets.set(ticket, {ready: false}); return ticket;
    },
    vx_gpu_trace_node_begin() { return 0; },
    vx_gpu_trace_node_end() {},
    vx_gpu_trace_program_begin() { return 0; },
    vx_gpu_trace_program_end() {},
    vx_gpu_debug_label(text, length) {
      label = length ? new TextDecoder().decode(bytes(text, length)) : null;
      stats.labels.push(label);
      return 1;
    },
    vx_gpu_trace_read(ticket, destination) {
      const job = timingTickets.get(ticket);
      if (!job) return -1;
      if (!job.ready) return -2;
      new Uint8Array(memory.buffer, destination, 40).fill(0);
      new DataView(memory.buffer).setBigUint64(destination, 125000n, true); return 0;
    },
    vx_gpu_trace_release(ticket) { timingTickets.delete(ticket); },
    vx_gpu_encode(dispatch) {
      const view = words();
      const base = dispatch >>> 2;
      const variantCount = view[base];
      const bindingCount = view[base + 1];
      const paramsBytes = view[base + 2];
      const variantBase = base + HEADER_WORDS;
      const bindingBase = variantBase + variantCount * 4;
      const paramsBase = bindingBase + bindingCount * BINDING_WORDS;
      const variants = [];
      for (let i = 0; i < variantCount; i++) {
        const at = variantBase + i * 4;
        variants.push({
          shaderId: view[at],
          workgroup: [view[at + 1], view[at + 2], view[at + 3]],
        });
      }
      const bindings = [];
      for (let i = 0; i < bindingCount; i++) {
        const at = bindingBase + i * BINDING_WORDS;
        bindings.push({
          slot: view[at], hostPointer: view[at + 1],
          offset: view[at + 2], bytes: view[at + 3], writes: view[at + 4],
        });
      }
      const params = Array.from(
        new Uint32Array(view.buffer, paramsBase << 2, paramsBytes >> 2));
      (recorded ??= []).push({ nodeIndex: view[base + 3], variants, bindings, params });
      stats.dispatchLabels.push(label);
      log.push({ variants, bindings, params });

      /* Execute what the descriptor describes. An operator the engine
       * claims but this host cannot run is a planning bug, not a gap, so an
       * unknown shader fails rather than quietly producing nothing. */
      const read = (index, Type = Float32Array) => {
        const span = spans.get(bindings[index].hostPointer);
        if (!span) return null;
        return new Type(span.bytes.buffer, span.bytes.byteOffset + bindings[index].offset,
          bindings[index].bytes / Type.BYTES_PER_ELEMENT);
      };
      const write = (index, values) => {
        const span = spans.get(bindings[index].hostPointer);
        if (!span) return -1;
        if (!bindings[index].writes) return -1;
        span.bytes.set(new Uint8Array(values.buffer, values.byteOffset,
          Math.min(bindings[index].bytes, values.byteLength)), bindings[index].offset);
        span.deviceDirty = true;
        return 0;
      };

      if (variants[0].shaderId === SHADER_INFERENCE_ROW_INDEX_TRANSFER) {
        const [rows, width, extent, scatter] = params;
        const bytes = index => {
          const { hostPointer, offset, bytes } = bindings[index];
          return spans.get(hostPointer).bytes.subarray(offset, offset + bytes);
        };
        const input = bytes(0), output = bytes(2).slice(), indexBytes = bytes(1);
        const indices = new Int32Array(indexBytes.buffer, indexBytes.byteOffset, indexBytes.length / 4);
        for (let row = 0; row < rows; row++) {
          const selected = indices[row];
          if (selected < 0 || selected >= extent) {
            if (!scatter) output.fill(0, row * width, (row + 1) * width);
            continue;
          }
          const source = (scatter ? row : selected) * width;
          output.set(input.subarray(source, source + width), (scatter ? selected : row) * width);
        }
        return write(2, output);
      }

      if (variants[0].shaderId === SHADER_INFERENCE_MAX_POOL2_D ||
          variants[0].shaderId === SHADER_INFERENCE_AVERAGE_POOL2_D) {
        const [pn, ph, pw, pc, pOutH, pOutW, pky, pkx, psy, psx, ppy, ppx] = params;
        const average = variants[0].shaderId === SHADER_INFERENCE_AVERAGE_POOL2_D;
        const input = read(0);
        if (!input) return -1;
        const output = new Float32Array(pn * pOutH * pOutW * pc);
        for (let n = 0; n < pn; n++) {
          for (let oy = 0; oy < pOutH; oy++) {
            for (let ox = 0; ox < pOutW; ox++) {
              for (let c = 0; c < pc; c++) {
                let best = -Infinity;
                let sum = 0;
                let count = 0;
                for (let ky = 0; ky < pky; ky++) {
                  for (let kx = 0; kx < pkx; kx++) {
                    const y = oy * psy + ky - ppy;
                    const x = ox * psx + kx - ppx;
                    if (y < 0 || x < 0 || y >= ph || x >= pw) continue;
                    const value = input[(((n * ph) + y) * pw + x) * pc + c];
                    best = Math.max(best, value);
                    sum += value;
                    count++;
                  }
                }
                output[(((n * pOutH) + oy) * pOutW + ox) * pc + c] =
                  average ? sum / (count || 1) : best;
              }
            }
          }
        }
        return write(1, output);
      }

      if (variants[0].shaderId === SHADER_INFERENCE_ARG_MAX_F32_I32_NATIVE) {
        const [outer, axis, inner, count] = params, input = read(0);
        if (!input || count !== outer * inner) return -1;
        const output = new Int32Array(count);
        for (let index = 0; index < count; index++) {
          const base = Math.floor(index / inner) * axis * inner + index % inner;
          for (let candidate = 1; candidate < axis; candidate++)
            if (input[base + candidate * inner] > input[base + output[index] * inner]) output[index] = candidate;
        }
        return write(1, output);
      }

      if (variants[0].shaderId === SHADER_INFERENCE_EMBEDDING) {
        const [rows, width, vocabulary] = params, tokens = read(0, Int32Array), weights = read(1);
        const output = new Float32Array(rows * width);
        for (let row = 0; row < rows; row++) if (tokens[row] >= 0 && tokens[row] < vocabulary)
          output.set(weights.subarray(tokens[row] * width, (tokens[row] + 1) * width), row * width);
        return write(2, output);
      }

      if (variants[0].shaderId === SHADER_INFERENCE_DECODE_KEEP_MASK) {
        const [lanes, keys, sourceKeys, sourceQueries, mode, causal] = params;
        const source = read(0, Int32Array), positions = read(1, Int32Array), output = new Int32Array(lanes * keys);
        for (let lane = 0; lane < lanes; lane++) for (let key = 0; key < keys; key++) {
          const position = positions[lane];
          if (position < 0 || (causal && key > position)) continue;
          const index = mode === 2 ? lane * sourceKeys + key : mode === 3 ? position * sourceKeys + key
            : mode === 4 ? (lane * sourceQueries + position) * sourceKeys + key : key;
          output[lane * keys + key] = mode === 0 || source[index] !== 0 ? 1 : 0;
        }
        return write(2, output);
      }

      if (variants[0].shaderId === SHADER_INFERENCE_CROSS_SDPA) {
        const [queries, keys, width, heads, headWidth, batches, scaleBits, causal, mode] = params;
        const q = read(0), k = read(1), v = read(2), mask = read(3, Int32Array);
        const scale = new Float32Array(Uint32Array.of(scaleBits).buffer)[0];
        const output = new Float32Array(batches * queries * width);
        for (let batch = 0; batch < batches; batch++) for (let query = 0; query < queries; query++)
          for (let head = 0; head < heads; head++) {
            const scores = [];
            for (let key = 0; key < keys; key++) {
              const index = mode === 2 ? batch * keys + key : mode === 3 ? query * keys + key
                : mode === 4 ? (batch * queries + query) * keys + key : key;
              if ((causal && key > query) || (mode && mask[index] === 0)) continue;
              let score = 0;
              for (let channel = 0; channel < headWidth; channel++)
                score += q[(batch * queries + query) * width + head * headWidth + channel] *
                  k[(batch * keys + key) * width + head * headWidth + channel];
              scores.push({key, value: score * scale});
            }
            const maximum = Math.max(...scores.map(score => score.value));
            const weights = scores.map(score => Math.exp(score.value - maximum));
            const total = weights.reduce((sum, weight) => sum + weight, 0);
            for (let channel = 0; channel < headWidth; channel++)
              for (let index = 0; index < scores.length; index++)
                output[(batch * queries + query) * width + head * headWidth + channel] +=
                  weights[index] / total * v[(batch * keys + scores[index].key) * width + head * headWidth + channel];
          }
        return write(4, output);
      }

      const binary = BINARY_OPS[variants[0].shaderId];
      if (binary) {
        const a = read(0);
        const b = read(1);
        if (!a || !b) return -1;
        const [size] = params;
        const [gx] = variants[0].workgroup;
        if (gx * 64 < size) return -1;
        const output = new Float32Array(size);
        if (variants[0].shaderId === SHADER_INFERENCE_ADD_RELU) {
          const relu = params[1];
          for (let index = 0; index < size; index++) {
            let value = a[index] + b[index];
            if (relu === 1) value = Math.max(value, 0);
            else if (relu >= 2) value = Math.min(Math.max(value, 0), 6);
            output[index] = value;
          }
        } else {
          const [, isBScalar, bSize, aSize, isAScalar] = params;
          for (let index = 0; index < size; index++) {
            const av = isAScalar === 1 ? a[0]
              : (aSize < size && aSize > 0 ? a[index % aSize] : a[index]);
            const bv = isBScalar === 1 ? b[0]
              : (bSize < size && bSize > 0 ? b[index % bSize] : b[index]);
            output[index] = binary(av, bv);
          }
        }
        return write(2, output);
      }

      const activation = UNIFORM_ELEMENTWISE[variants[0].shaderId];
      if (activation) {
        const [size] = params;
        const input = read(0);
        if (!input) return -1;
        /* The grid must reach every element the uniform claims: this is what
         * proves the engine's 2D split, not just its element count. */
        const [gx, gy, gz] = variants[0].workgroup;
        if (gx * 64 * gy * gz < size) return -1;
        const output = new Float32Array(size);
        for (let index = 0; index < size; index++) {
          output[index] = activation(input[index]);
        }
        return write(1, output);
      }
      return -1;
    },
    vx_gpu_end() { return recorded && recorded.length > 0 ? 0 : -1; },
    vx_gpu_snapshot(hostPointer, offset, size) {
      stats.snapshotSizes.push(size);
      const span = hostPointer === 0 && size === 0 ? {bytes: new Uint8Array()} : spans.get(hostPointer);
      if (!span) return -1;
      const ticket = nextTicket++;
      jobs.set(ticket, { bytes: span.bytes.slice(offset, offset + size), ready: false, failed: false });
      stats.pending++;
      return ticket;
    },
    vx_gpu_readback(ticket, destination, size) {
      stats.readbackSizes.push(size);
      const job = jobs.get(ticket);
      if (!job || job.failed || size !== job.bytes.length) return -1;
      if (!job.ready) return -2;
      bytes(destination, size).set(job.bytes);
      return 0;
    },
    vx_gpu_readback_release(ticket) { jobs.delete(ticket); },
  };
  const counted = Object.fromEntries(Object.entries(imports).map(([name, fn]) =>
    [name, (...args) => { stats.calls.push(name); return fn(...args); }]));
  return { imports: counted, stats, jobs, timingTickets, enableTiming() { timing = true; }, setWakeup(callback) { wakeup = callback; }, attach(attached) { memory = attached; },
    complete(failed = false) {
      for (const job of jobs.values()) { job.ready = true; job.failed = failed; }
      for (const job of timingTickets.values()) job.ready = true;
      wakeup();
    },
    async waitForCompletion() { await Promise.resolve(); this.complete(); },
  };
}

function writeSafetensors(file) {
  /* MaxPool2D carries no weights, but a package must still present a valid
   * SafeTensors header. */
  const header = Buffer.from(JSON.stringify({}), 'utf8');
  const padded = Buffer.concat([header, Buffer.alloc((8 - header.length % 8) % 8, 0x20)]);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(padded.length));
  fs.writeFileSync(file, Buffer.concat([length, padded]));
}

test('the engine runs a graph on the WebGPU backend a host supplies', async (t) => {
  if (!fs.existsSync(FULL_WASM)) {
    t.skip('release WASM is not built');
    return;
  }
  const api = await import('../ts/full.js');

  const log = [];
  const bridge = createSimulatingBridge(log);
  assert.equal(bridge.imports.vx_gpu_available(), 1);

  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'volvoxai-gpu-'));
  try {
    const graph = {
      format: "volvox-graph/v1",
      dimensions: {},
      inputs: { input0: { shape: [N, H, W, C], dtype: 'float32' } },
      outputs: ['out0'],
      nodes: [{
        id: 'pool',
        opType: 'MaxPool2D',
        inputs: { input: 'input0' },
        outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [N, OUT_H, OUT_W, C] } },
        params: {
          kernel: [KY, KX], stride: [SY, SX], pads: [0, 0, 0, 0], data_layout: 'NHWC',
        },
      }],
    };
    declareStateless(graph);
    fs.writeFileSync(path.join(directory, 'graph.json'), JSON.stringify(graph));
    writeSafetensors(path.join(directory, 'model.safetensors'));

    const input = Float32Array.from({ length: N * H * W * C }, (_, index) =>
      Math.sin(index) * 10);
    const expected = maxPool2dReference(input);

    /* The full module is the one that carries the WebGPU backend; the
     * ordinary transport is what reaches the engine's C dispatch. Pairing them
     * is how a caller with a device gets planning in C and the device in JS. */
    /* The transport fetches package sources; serving them from memory keeps
     * this test independent of any file protocol the host would otherwise
     * have to resolve. */
    if (!graph.execution) declareStateless(graph);
  const graphText = JSON.stringify(graph);
    const weights = fs.readFileSync(path.join(directory, 'model.safetensors'));
    const host = new FullEngineHost({
      wasmUrl: FULL_WASM,
      gpuBridge: bridge,
      fetch: async (source) => {
        if (String(source) === 'graph.json') {
          return { ok: true, text: async () => graphText,
            json: async () => JSON.parse(graphText) };
        }
        if (String(source) === 'model.safetensors') {
          return { ok: true, arrayBuffer: async () => weights.buffer.slice(
            weights.byteOffset, weights.byteOffset + weights.byteLength) };
        }
        return { ok: false, status: 404, statusText: String(source) };
      },
      resolveModelSource: (graphPath, weightPaths) => ({
        graphUrl: graphPath, weightSources: weightPaths,
      }),
    });
    const inference = new api.VxInferenceServiceClient(reportTransport(host));
    const runtime = await inference.createRuntime(new api.pb.CreateRuntimeRequest());
    assert.equal(runtime.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      runtime.report?.message);
    const model = await inference.loadModel(new api.pb.LoadModelRequest({
      runtimeId: runtime.runtimeId,
      graphPath: 'graph.json',
      weightPaths: ['model.safetensors'],
    }));
    assert.equal(model.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      model.report?.message);
    const compiled = await inference.compileModel(new api.pb.CompileModelRequest({
      modelId: model.modelId,
      policy: new api.pb.BackendPolicy({
        mode: api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['webgpu'],
      }),
    }));
    assert.equal(compiled.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      `compile on webgpu: ${compiled.report?.message}`);

    const context = await inference.createExecutionContext(
      new api.pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new api.pb.GraphExecution(),
      }));
    assert.equal(context.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      context.report?.message);
    const request = graphRequest(api.pb, graph, context.contextId, [new api.pb.Tensor({
      name: 'input0', dtype: api.pb.DataType.DATA_TYPE_F32, shape: [N,H,W,C].map(BigInt),
      inline: new Uint8Array(input.buffer, input.byteOffset, input.byteLength),
    })]);
    const pending = beginCall(bridge, () => inference.execute(request));
    await pending.started;
    assert.equal(pending.settled, false, 'Execute waits for final device completion');
    assert.equal((await inference.getExecutionContextInfo(new api.pb.ExecutionContextRef(context))).state,
      api.pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING);
    bridge.complete();
    const result = await pending.promise;
    assert.equal(result.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK, result.report?.message);

    /*
     * Route evidence, not the answer, is what says the device ran the node.
     * A CPU fallback produces the same numbers, so a test that only compared
     * outputs would pass with the backend doing nothing.
     */
    const route = result.report?.route;
    assert.equal(route?.builtin, true, 'the built-in backend owned the route');
    assert.equal(route?.fallbackNodes, 0, 'no node fell back to the host');
    assert.equal(route?.missingNodes, 0, 'every node was selected');
    assert.equal(route?.selectedNodes, route?.activeNodes,
      'every active node ran on the selected backend');

    /* The bridge is asked in the order the contract describes. */
    const order = ['vx_gpu_begin', 'vx_gpu_ensure', 'vx_gpu_encode',
      'vx_gpu_end', 'vx_gpu_snapshot'];
    let at = 0;
    for (const call of bridge.stats.calls) if (call === order[at]) at++;
    assert.equal(at, order.length,
      `bridge calls did not follow ${order.join(' → ')}: ` +
      `${bridge.stats.calls.join(',')}`);

    const [first] = log;
    assert.ok(first, 'a dispatch reached the host');
    assert.equal(first.variants[0].shaderId, SHADER_INFERENCE_MAX_POOL2_D);
    assert.deepEqual(first.variants[0].workgroup,
      [Math.ceil(OUT_W / 8), Math.ceil(OUT_H / 8), N * C]);
    assert.equal(first.bindings.length, 2, 'input and output are bound');
    assert.deepEqual(first.params,
      [N, H, W, C, OUT_H, OUT_W, KY, KX, SY, SX, 0, 0]);

    const payload = result.outputs.find(tensor => tensor.name === 'out0')?.inline;
    assert.ok(payload && payload.byteLength > 0, 'the output carries bytes');
    const actual = new Float32Array(
      payload.buffer, payload.byteOffset, payload.byteLength >> 2);
    assert.equal(actual.length, expected.length);
    for (let index = 0; index < expected.length; index++) {
      assert.ok(Math.abs(actual[index] - expected[index]) < 1e-6,
        `output[${index}] is ${actual[index]}, expected ${expected[index]}`);
    }
    t.diagnostic(
      `submitted once, ${bridge.stats.pending} pending ` +
      `readbacks, ${log.length} dispatches of ` +
      `${SHADER_NAMES[SHADER_INFERENCE_MAX_POOL2_D]}`);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

/** A SafeTensors package with no tensors: these operators carry no weights. */
function emptySafetensors() {
  const header = Buffer.from(JSON.stringify({}), 'utf8');
  const padded = Buffer.concat([
    header, Buffer.alloc((8 - header.length % 8) % 8, 0x20)]);
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(padded.length));
  return Buffer.concat([length, padded]);
}

/**
 * Compile one graph onto WebGPU and run it to a settled answer.
 *
 * Execute completion includes the pending readbacks, and
 * every operator added to the planner is expected to travel it unchanged.
 */
async function runOnWebGpu(api, graph, inputs, inputShape) {
  const batches = Array.isArray(inputs) ? inputs : [inputs];
  const log = [];
  const bridge = createSimulatingBridge(log);
  if (!graph.execution) declareStateless(graph);
  const graphText = JSON.stringify(graph);
  const weights = emptySafetensors();
  const host = new FullEngineHost({
    wasmUrl: FULL_WASM,
    gpuBridge: bridge,
    fetch: async (source) => {
      if (String(source) === 'graph.json') {
        return { ok: true, text: async () => graphText,
          json: async () => JSON.parse(graphText) };
      }
      if (String(source) === 'model.safetensors') {
        return { ok: true, arrayBuffer: async () => weights.buffer.slice(
          weights.byteOffset, weights.byteOffset + weights.byteLength) };
      }
      return { ok: false, status: 404, statusText: String(source) };
    },
    resolveModelSource: (graphPath, weightPaths) => ({
      graphUrl: graphPath, weightSources: weightPaths,
    }),
  });
  const inference = new api.VxInferenceServiceClient(reportTransport(host));
  const runtime = await inference.createRuntime(new api.pb.CreateRuntimeRequest());
  const model = await inference.loadModel(new api.pb.LoadModelRequest({
    runtimeId: runtime.runtimeId,
    graphPath: 'graph.json',
    weightPaths: ['model.safetensors'],
  }));
  const compiled = await inference.compileModel(new api.pb.CompileModelRequest({
    modelId: model.modelId,
    policy: new api.pb.BackendPolicy({
      mode: api.pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['webgpu'],
    }),
  }));
  assert.equal(compiled.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
    `compile: ${compiled.report?.message} at ` +
    `${JSON.stringify(compiled.report?.offendingNode)}`);
  const context = await inference.createExecutionContext(
    new api.pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new api.pb.GraphExecution(),
    }));
  assert.equal(context.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
    context.report?.message);

  const outputs = [];
  let route = null;
  for (const batch of batches) {
    /* A batch is either the single graph input, or the named tensors of a
     * graph that has more than one. */
    const tensors = batch instanceof Float32Array
      ? [{ name: 'input0', shape: inputShape, data: batch }]
      : batch;
    const request = graphRequest(api.pb, graph, context.contextId, tensors.map(({name, shape, data}) =>
      new api.pb.Tensor({name, dtype: api.pb.DataType.DATA_TYPE_F32, shape: shape.map(BigInt),
        inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength)})));
    const result = await completeCall(bridge, () => inference.execute(request));
    assert.equal(result.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK, result.report?.message);
    route = result.report?.route;
    const payload = result.outputs.find(tensor => tensor.name === 'out0').inline;
    outputs.push(new Float32Array(payload.slice().buffer));
  }
  return { actual: outputs[0], outputs, route, log };
}

/*
 * Every operator the planner claims, run through the same contract.
 *
 * Each case names the shader it must select. Comparing only the numbers would
 * pass with the wrong shader whenever two operators agree on the test data,
 * so the dispatch log is checked as well.
 */
test('each planned operator selects its shader and computes its answer',
     async (t) => {
  if (!fs.existsSync(FULL_WASM)) {
    t.skip('release WASM is not built');
    return;
  }
  const api = await import('../ts/full.js');

  const elementwiseShape = [1, 2, 3, 2];
  const elementwise = Float32Array.from(
    { length: 12 }, (_, index) => Math.sin(index) * 3 - 1);
  /*
   * Tanh is planned but absent here on purpose: its WebGPU row in the
   * generated kernel registry carries exporter_qualified = 0, so compilation
   * refuses it before the planner is consulted. That is the exporter's
   * decision to make, not this test's, and the planner is ready for the day
   * it changes.
   */
  const activations = [
    ['ReLU', SHADER_INFERENCE_RE_LU],
    ['Sigmoid', SHADER_INFERENCE_SIGMOID],
    ['SiLU', SHADER_INFERENCE_SI_LU],
  ];

  for (const [opType, shaderId] of activations) {
    const graph = {
      format: 'volvox-graph/v1',
      dimensions: {},
      inputs: { input0: { shape: elementwiseShape, dtype: 'float32' } },
      outputs: ['out0'],
      nodes: [{
        id: 'activation',
        opType,
        inputs: { input: 'input0' },
        outputs: { out: { tensor: 'out0', dtype: 'float32',
          shape: elementwiseShape } },
        params: {},
      }],
    };
    const { actual, route, log } = await runOnWebGpu(
      api, graph, elementwise, elementwiseShape);
    const reference = UNIFORM_ELEMENTWISE[shaderId];
    assert.equal(actual.length, elementwise.length, opType);
    for (let index = 0; index < elementwise.length; index++) {
      assert.ok(Math.abs(actual[index] - reference(elementwise[index])) < 1e-5,
        `${opType}[${index}] is ${actual[index]}, ` +
        `expected ${reference(elementwise[index])}`);
    }
    assert.equal(route?.fallbackNodes, 0, `${opType} fell back to the host`);
    assert.ok(log.some((entry) => entry.variants[0].shaderId === shaderId),
      `${opType} did not select ${SHADER_NAMES[shaderId]}`);
    t.diagnostic(`${opType} -> ${SHADER_NAMES[shaderId]}`);
  }

  /* AveragePool2D shares MaxPool2D's twelve uniforms; only the shader differs,
   * which is exactly what a wrong id would hide behind similar numbers. */
  const poolShape = [1, 4, 4, 2];
  const poolInput = Float32Array.from(
    { length: 32 }, (_, index) => Math.cos(index) * 5);
  const poolGraph = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input0: { shape: poolShape, dtype: 'float32' } },
    outputs: ['out0'],
    nodes: [{
      id: 'pool',
      opType: 'AveragePool2D',
      inputs: { input: 'input0' },
      outputs: { out: { tensor: 'out0', dtype: 'float32', shape: [1, 2, 2, 2] } },
      params: { kernel: [2, 2], stride: [2, 2], pads: [0, 0, 0, 0],
        data_layout: 'NHWC' },
    }],
  };
  const pooled = await runOnWebGpu(
    api, poolGraph, poolInput, poolShape);
  assert.ok(pooled.log.some((entry) =>
    entry.variants[0].shaderId === SHADER_INFERENCE_AVERAGE_POOL2_D),
    'AveragePool2D did not select its shader');
  assert.equal(pooled.route?.fallbackNodes, 0, 'AveragePool2D fell back');
  for (let oy = 0; oy < 2; oy++) {
    for (let ox = 0; ox < 2; ox++) {
      for (let c = 0; c < 2; c++) {
        let sum = 0;
        for (let ky = 0; ky < 2; ky++) {
          for (let kx = 0; kx < 2; kx++) {
            sum += poolInput[(((oy * 2 + ky) * 4) + (ox * 2 + kx)) * 2 + c];
          }
        }
        const expected = sum / 4;
        const got = pooled.actual[((oy * 2) + ox) * 2 + c];
        assert.ok(Math.abs(got - expected) < 1e-5,
          `AveragePool2D[${oy},${ox},${c}] is ${got}, expected ${expected}`);
      }
    }
  }
  t.diagnostic(`AveragePool2D -> ${SHADER_NAMES[SHADER_INFERENCE_AVERAGE_POOL2_D]}`);
});

/*
 * A second execution must see the second input.
 *
 * The host uploads a span only when the device has not written it. Deciding
 * that from "was it bound" rather than "was it written" makes an input look
 * device-owned after its first dispatch, and every later execution then
 * computes on stale bytes -- with no error anywhere, because the numbers are
 * a perfectly good answer to the previous question.
 */
test('a repeated execution uses the inputs it was given', async (t) => {
  if (!fs.existsSync(FULL_WASM)) {
    t.skip('release WASM is not built');
    return;
  }
  const api = await import('../ts/full.js');

  const shape = [1, 2, 3, 2];
  const first = Float32Array.from({ length: 12 }, (_, i) => Math.sin(i) * 3 - 1);
  const second = Float32Array.from({ length: 12 }, (_, i) => Math.cos(i) * 4 + 1);
  const graph = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { input0: { shape, dtype: 'float32' } },
    outputs: ['out0'],
    nodes: [{
      id: 'activation', opType: 'ReLU',
      inputs: { input: 'input0' },
      outputs: { out: { tensor: 'out0', dtype: 'float32', shape } },
      params: {},
    }],
  };

  const { outputs } = await runOnWebGpu(
    api, graph, [first, second], shape);
  assert.equal(outputs.length, 2);
  for (const [index, source] of [first, second].entries()) {
    const expected = Float32Array.from(source, (x) => (x > 0 ? x : 0));
    for (let at = 0; at < expected.length; at++) {
      assert.ok(Math.abs(outputs[index][at] - expected[at]) < 1e-6,
        `execution ${index + 1} output[${at}] is ${outputs[index][at]}, ` +
        `expected ${expected[at]}`);
    }
  }
  t.diagnostic('both executions answered their own input');
});

/*
 * The binary family, both operands supplied.
 *
 * `run` is handed only a node's primary input, so a binary operator has to
 * resolve its second operand itself. A planner that reads the wrong port would
 * still produce numbers -- the wrong ones, or the same ones when a and b
 * happen to agree -- so the operands here are deliberately different and the
 * shader id is asserted alongside the values.
 */
test('binary operators bind both operands and select their shader',
     async (t) => {
  if (!fs.existsSync(FULL_WASM)) {
    t.skip('release WASM is not built');
    return;
  }
  const api = await import('../ts/full.js');

  const shape = [1, 2, 3, 2];
  const count = 12;
  const left = Float32Array.from({ length: count }, (_, i) => Math.sin(i) * 3 - 1);
  const right = Float32Array.from({ length: count }, (_, i) => Math.cos(i) * 2 + 3);

  const cases = [
    ['Add', SHADER_INFERENCE_ADD_RELU, (a, b) => a + b],
    ['Sub', SHADER_INFERENCE_SUB, (a, b) => a - b],
    ['Mul', SHADER_INFERENCE_MUL, (a, b) => a * b],
    ['Div', SHADER_INFERENCE_DIV, (a, b) => a / b],
  ];

  for (const [opType, shaderId, reference] of cases) {
    const graph = {
      format: 'volvox-graph/v1',
      dimensions: {},
      inputs: {
        input0: { shape, dtype: 'float32' },
        input1: { shape, dtype: 'float32' },
      },
      outputs: ['out0'],
      nodes: [{
        id: 'binary', opType,
        inputs: { a: 'input0', b: 'input1' },
        outputs: { out: { tensor: 'out0', dtype: 'float32', shape } },
        params: {},
      }],
    };
    const { actual, route, log } = await runOnWebGpu(
      api, graph,
      [[{ name: 'input0', shape, data: left },
        { name: 'input1', shape, data: right }]],
      shape);

    assert.equal(actual.length, count, opType);
    for (let index = 0; index < count; index++) {
      const expected = reference(left[index], right[index]);
      assert.ok(Math.abs(actual[index] - expected) < 1e-5,
        `${opType}[${index}] is ${actual[index]}, expected ${expected}`);
    }
    assert.equal(route?.fallbackNodes, 0, `${opType} fell back to the host`);
    assert.ok(log.some((entry) => entry.variants[0].shaderId === shaderId),
      `${opType} did not select ${SHADER_NAMES[shaderId]}`);
    /* Three storage bindings: a, b and the output. A planner that dropped the
     * second operand would bind two and still run. */
    const entry = log.find((item) => item.variants[0].shaderId === shaderId);
    assert.equal(entry.bindings.length, 3, `${opType} bound ${entry.bindings.length}`);
    assert.deepEqual(entry.bindings.map((b) => b.writes), [0, 0, 1],
      `${opType} declared the wrong written binding`);
    t.diagnostic(`${opType} -> ${SHADER_NAMES[shaderId]}`);
  }
});

// Ownership regressions use direct generated dispatch so completion cannot be
// accidentally hidden by a JavaScript retry loop.
import * as pb from '../runtime/generated/typescript/volvoxai_lite.js';
import { VxInferenceServiceClient, VxSchedulerServiceClient, VxPlatformServiceClient, VxProfilingServiceClient, VxDebugServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import { ModelControlWasmDispatchFactory } from '../ts/core/ModelControlWasm.js';
import { wasmReleaseModuleImports } from '../ts/core/WasmReleaseModule.js';

const RESULT_SHAPE = [1, 2, 3, 2];
const RESULT_GRAPH = declareStateless({ format: 'volvox-graph/v1', dimensions: {},
  inputs: { input0: { shape: RESULT_SHAPE, dtype: 'float32' } }, outputs: ['out0'],
  nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'input0' },
    outputs: { out: { tensor: 'out0', shape: RESULT_SHAPE, dtype: 'float32' } }, params: {} }],
});
function accepted(response) {
  assert.equal(response.report?.status, pb.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
  return response;
}
function executeRequest(context, inputs, options = {}) {
  return new pb.ExecuteRequest({contextId: context.contextId,
    inputs: new pb.InputBindings({values: inputs}),
    scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}), ...options});
}
function resultRequest(context, value) {
  return executeRequest(context, [new pb.Tensor({name:'input0', shape:RESULT_SHAPE.map(BigInt),
    dtype:pb.DataType.DATA_TYPE_F32, inline:new Uint8Array(new Float32Array(12).fill(value).buffer)})]);
}
function beginCall(bridge, call) {
  const state = {settled:false};
  const tickets = new Set(bridge.jobs.keys());
  state.promise = call().then(response => {state.settled=true;state.response=response;return response;},
    error => {state.settled=true;throw error;});
  void state.promise.catch(() => {});
  state.started = (async () => {
    for (let turn=0; !state.settled && ![...bridge.jobs.keys()].some(ticket=>!tickets.has(ticket)) && turn<100; turn++)
      await new Promise(resolve => setImmediate(resolve));
    assert.ok(state.settled || [...bridge.jobs.keys()].some(ticket=>!tickets.has(ticket)), 'device call must dispatch or refuse');
  })();
  return state;
}
async function completeCall(bridge, call) {
  const pending=beginCall(bridge,call); await pending.started;
  for(let turn=0; !pending.settled && turn<100; turn++){bridge.complete();await new Promise(resolve=>setImmediate(resolve));}
  return pending.promise;
}
async function beginExecution(fixture, request) {
  const pending=beginCall(fixture.bridge,()=>fixture.inference.execute(request));
  await pending.started;return pending;
}
async function execute(fixture, request) {
  return accepted(await completeCall(fixture.bridge,()=>fixture.inference.execute(request)));
}
function makeResultControl(clock, limits) {
  const log = [];
  const bridge = createSimulatingBridge(log, limits);
  class FixtureFactory extends ModelControlWasmDispatchFactory {
    instantiate(wakeup) {
      if (!clock) return super.instantiate(wakeup);
      const imports = wasmReleaseModuleImports(bridge.imports, wakeup);
      imports.host.vx_host_monotonic_micros_v1 = () => clock.micros;
      const instance = new WebAssembly.Instance(this.module, imports);
      bridge.attach(instance.exports.memory);
      instance.exports.__wasm_call_ctors?.();
      return instance;
    }
  }
  const control = new FixtureFactory(
    new WebAssembly.Module(fs.readFileSync(FULL_WASM)), bridge).create();
  return { control, bridge, log };
}
async function resultFixture({ scheduled = false, graph = RESULT_GRAPH, clock, budget, limits,
  compileStatus = pb.NativeStatus.NATIVE_STATUS_OK, weights = emptySafetensors(), preserveNodeBoundaries = false } = {}) {
  if (!graph.execution) declareStateless(graph);
  const fixture = makeResultControl(clock, limits);
  const { control } = fixture;
  const inference = new VxInferenceServiceClient(reportTransport(control));
  const scheduler = new VxSchedulerServiceClient(reportTransport(control));
  control.mountFileChunks('graph.json', [new TextEncoder().encode(JSON.stringify(graph))]);
  control.mountFileChunks('weights.safetensors', [weights]);
  const runtime = accepted((await inference.createRuntime(new pb.CreateRuntimeRequest({
    executionMode: scheduled ? pb.ExecutionMode.EXECUTION_MODE_SCHEDULED : pb.ExecutionMode.EXECUTION_MODE_DIRECT,
    budget,
  }))));
  const model = accepted((await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
    graphPath: 'graph.json', weightPaths: ['weights.safetensors'] }))));
  const compiled = (await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId, preserveNodeBoundaries,
    policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['webgpu'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
  assert.equal(compiled.report.status, compileStatus, compiled.report.message);
  const contexts = [];
  return { ...fixture, inference, scheduler, runtime, model, compiled, graph,
    async context(options = {}) {
      const context = accepted((await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph:new pb.GraphExecution(), ...options }))));
      contexts.push(context);
      return context;
    },
    async close() {
      fixture.bridge.complete();
      await new Promise(resolve=>setImmediate(resolve));
      for (const context of contexts) (await inference.releaseExecutionContext(new pb.ExecutionContextRef(context)));
      (await inference.releaseCompiledModel(new pb.CompiledModelRef(compiled)));
      (await inference.releaseModel(new pb.ModelRef(model)));
      (await inference.releaseRuntime(new pb.RuntimeRef(runtime)));
      (await control.close());
    },
  };
}
// Begin a real asynchronous wait, then observe the engine's pending state.
// Simulated GPU completion is controlled explicitly by each test.
async function startRequest(fixture, request) {
  const completion = fixture.scheduler.waitRequest(new pb.RequestRef(request));
  void completion.catch(() => {});
  for (let turn = 0; turn < 64; turn++) {
    const state = await fixture.scheduler.pollRequest(new pb.RequestRef(request));
    if (state.state !== pb.RequestState.REQUEST_STATE_QUEUED) return state;
  }
  throw new Error('Scheduled request made no progress while its wait call was open');
}

async function resultValues(inference, result, name = 'out0') {
  const data = result.outputs.find(output => output.name === name).inline;
  return [...new Float32Array(data.slice().buffer)];
}

test('pending executions retain context admission and completed packets retain their values', async () => {
 const f=await resultFixture();
 try {
  const first=await f.context(), second=await f.context();
  const a=await beginExecution(f,resultRequest(first,2));
  const b=await beginExecution(f,resultRequest(second,9));
  assert.equal(f.log.length,2);
  for(let i=0;i<8;i++) {
   assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(first))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING);
   assert.equal(a.settled,false);
  }
  const busy=await f.inference.execute(resultRequest(first,3));
  assert.equal(busy.report.status,pb.NativeStatus.NATIVE_STATUS_BUSY);
  f.bridge.complete();
  const pa=accepted(await a.promise),pbResult=accepted(await b.promise);
  const pc=await execute(f,resultRequest(first,3));
  assert.equal(new Set([pa.executionId,pbResult.executionId,pc.executionId]).size,3);
  await f.inference.releaseExecutionContext(new pb.ExecutionContextRef(first));
  await f.inference.releaseExecutionContext(new pb.ExecutionContextRef(second));
  for(const [packet,value] of [[pa,2],[pbResult,9],[pc,3]])assert.deepEqual(await resultValues(f.inference,packet),Array(12).fill(value));
  assert.equal(f.bridge.jobs.size,0);
 } finally {await f.close();}
});

test('closing a pending Execute transport drains its ticket and permits independent contexts', async () => {
 const f=await resultFixture();
 try {
  for(let i=0;i<8;i++) {
   const context=await f.context(), abort=new AbortController();
   const pending=beginCall(f.bridge,()=>f.inference.execute(resultRequest(context,i),{signal:abort.signal}));
   await pending.started;abort.abort();await assert.rejects(pending.promise);
   f.bridge.complete();
   await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(context));
   await f.inference.releaseExecutionContext(new pb.ExecutionContextRef(context));
   assert.equal(f.bridge.jobs.size,0);
  }
  const survivor=await execute(f,resultRequest(await f.context(),81));
  assert.deepEqual(await resultValues(f.inference,survivor),Array(12).fill(81));
 } finally {await f.close();}
});

test('failed device completion is terminal and never exposes output bytes', async () => {
 const f=await resultFixture();
 try {
  const context=await f.context(), pending=await beginExecution(f,resultRequest(context,7));
  f.bridge.complete(true);const failed=await pending.promise;
  assert.equal(failed.report.status,pb.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED);
  assert.equal(failed.progress.state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
  assert.deepEqual(failed.outputs,[]);assert.ok(failed.executionId>0n);
  for(let i=0;i<2;i++)assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(context))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
  assert.equal(f.bridge.jobs.size,0);
 } finally {await f.close();}
});

test('scheduler remains running until final device completion and retires cancelled requests', async () => {
 const f=await resultFixture({scheduled:true});
 try {
  const submit=async value=>accepted(await f.scheduler.submit(new pb.SubmitRequest({execute:resultRequest(await f.context(),value)})));
  const first=await submit(4), waiting=await startRequest(f,first);
  assert.equal(waiting.state,pb.RequestState.REQUEST_STATE_RUNNING);
  f.bridge.complete();
  assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(first))).state,pb.RequestState.REQUEST_STATE_SUCCEEDED);
  const packet=accepted(await f.scheduler.takeRequestResult(new pb.RequestRef(first)));
  assert.deepEqual(await resultValues(f.inference,packet),Array(12).fill(4));
  assert.equal((await f.scheduler.takeRequestResult(new pb.RequestRef(first))).report.status,pb.NativeStatus.NATIVE_STATUS_NOT_FOUND);
  await f.scheduler.releaseRequest(new pb.RequestRef(first));
  const second=await submit(5);await startRequest(f,second);
  await f.scheduler.cancelRequest(new pb.RequestRef(second));f.bridge.complete();
  assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(second))).state,pb.RequestState.REQUEST_STATE_CANCELLED);
  await f.scheduler.releaseRequest(new pb.RequestRef(second));
  const third=await submit(6);await startRequest(f,third);
  await f.inference.releaseRuntime(new pb.RuntimeRef(f.runtime));
  await f.scheduler.releaseRequest(new pb.RequestRef(third));f.bridge.complete();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.bridge.jobs.size,0);
 } finally {await f.close();}
});

test('full host settles generated Execute calls and releases every readback ticket', async () => {
 const bridge=createSimulatingBridge([]);
 const host=new FullEngineHost({wasmUrl:FULL_WASM,gpuBridge:bridge,fetch:async source=>String(source)==='graph.json'
  ?{ok:true,text:async()=>JSON.stringify(RESULT_GRAPH)}:{ok:true,arrayBuffer:async()=>Uint8Array.from(emptySafetensors()).buffer}});
 const inference=new VxInferenceServiceClient(reportTransport(host));
 try {
  const runtime=accepted(await inference.createRuntime(new pb.CreateRuntimeRequest()));
  const model=accepted(await inference.loadModel(new pb.LoadModelRequest({runtimeId:runtime.runtimeId,graphPath:'graph.json',weightPaths:['weights.safetensors']})));
  const compiled=accepted(await inference.compileModel(new pb.CompileModelRequest({modelId:model.modelId,policy:new pb.BackendPolicy({mode:pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,backends:['webgpu'],operatorFallback:pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})})));
  const context=accepted(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({...compiled,graph:new pb.GraphExecution()})));
  for(let i=0;i<70;i++) {
   const packet=accepted(await completeCall(bridge,()=>inference.execute(resultRequest(context,i))));
   assert.deepEqual(await resultValues(inference,packet),Array(12).fill(i));assert.equal(bridge.jobs.size,0);
  }
  const invalid=resultRequest(context,1);invalid.scope=new pb.ExecutionScope({entrypoint:'transition',slotRows:new pb.SlotRows({slots:[new pb.SlotRow({position:1})]})});
  assert.notEqual((await inference.execute(invalid)).report.status,0,'undeclared transition is refused');
 } finally {await host.close();}
});

test('a node exceeding device bindings refuses the whole graph before portable fallback', async () => {
  const f = makeResultControl(undefined, {...DEVICE_LIMITS, maxStorageBuffersPerShaderStage: 2});
  const inference = new VxInferenceServiceClient(reportTransport(f.control));
  const graph = structuredClone(RESULT_GRAPH);
  graph.nodes[0].outputs.out.tensor = 'mid';
  graph.nodes.push({ id: 'dense', opType: 'MatMul', inputs: { input: 'mid', weight: 'weight', bias: 'bias' },
    outputs: { out: { tensor: 'out0', dtype: 'float32', shape: RESULT_SHAPE } },
    params: { weight_layout: 'dout_din' } });
  f.control.mountFileChunks('graph.json', [new TextEncoder().encode(JSON.stringify(graph))]);
  const header = new TextEncoder().encode(JSON.stringify({
    weight: { dtype: 'F32', shape: [2, 2], data_offsets: [0, 16] },
    bias: { dtype: 'F32', shape: [2], data_offsets: [16, 24] },
  }));
  const headerBytes = Math.ceil(header.length / 8) * 8;
  const weights = new Uint8Array(8 + headerBytes + 24);
  new DataView(weights.buffer).setBigUint64(0, BigInt(headerBytes), true);
  weights.fill(32, 8, 8 + headerBytes);
  weights.set(header, 8);
  weights.set(new Uint8Array(Float32Array.from([1, 0, 0, 1, 0, 0]).buffer), 8 + headerBytes);
  f.control.mountFileChunks('weights.safetensors', [weights]);
  const runtime = accepted((await inference.createRuntime(new pb.CreateRuntimeRequest())));
  const model = accepted((await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
    graphPath: 'graph.json', weightPaths: ['weights.safetensors'] }))));
  let compiled, context;
  try {
    const refused = (await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['webgpu'] }) })));
    assert.equal(refused.report.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED);
    assert.equal(f.log.length, 0, 'qualification submits no partial graph');
    compiled = accepted((await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_PREFER,
        backends: ['webgpu', 'wasm'] }) }))));
    context = accepted((await inference.createExecutionContext(new pb.CreateExecutionContextRequest({...compiled, graph:new pb.GraphExecution()}))));
    const result = accepted((await inference.execute(resultRequest(context, 2))));
    assert.equal(result.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY);
    for (const value of (await resultValues(inference, result))) assert.ok(Math.abs(value - 2) < 1e-6);
    assert.equal(f.log.length, 0, 'the entire model executes in portable C');

  } finally {
    if (context) (await inference.releaseExecutionContext(new pb.ExecutionContextRef(context)));
    if (compiled) (await inference.releaseCompiledModel(new pb.CompiledModelRef(compiled)));
    (await inference.releaseModel(new pb.ModelRef(model)));
    (await inference.releaseRuntime(new pb.RuntimeRef(runtime)));
    (await f.control.close());
  }
});

test('multi-output publication is atomic and context reuse waits for final completion', async () => {
 const graph=structuredClone(RESULT_GRAPH);graph.outputs.push('out1');
 graph.nodes.push({id:'again',opType:'ReLU',inputs:{input:'out0'},outputs:{out:{tensor:'out1',dtype:'float32',shape:RESULT_SHAPE}},params:{}});
 declareStateless(graph);const f=await resultFixture({graph});
 try {
  const context=await f.context(), first=await beginExecution(f,resultRequest(context,2));
  assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length,2);
  f.bridge.jobs.values().next().value.ready=true;
  await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(context));
  assert.equal(first.settled,false);
  assert.equal((await f.inference.execute(resultRequest(context,5))).report.status,pb.NativeStatus.NATIVE_STATUS_BUSY);
  f.bridge.complete();const initial=accepted(await first.promise),second=await execute(f,resultRequest(context,5));
  for(const [packet,value] of [[initial,2],[second,5]])for(const name of ['out0','out1'])assert.deepEqual(await resultValues(f.inference,packet,name),Array(12).fill(value));
  assert.equal(f.bridge.jobs.size,0);
 } finally {await f.close();}
});

test('WASM platform time and scheduled admission share the host monotonic clock', async () => {
  const clock = { micros: 1_234_567 };
  const f = (await resultFixture({ scheduled: true, clock }));
  try {
    const platform = new VxPlatformServiceClient(reportTransport(f.control));
    assert.equal((await platform.getMonotonicTime(new pb.Empty())).nanoseconds, 1_234_567_000n);
    const profiling = new VxProfilingServiceClient(reportTransport(f.control));
    const sample = accepted(await profiling.getResourceSnapshot(new pb.GetResourceSnapshotRequest({module: new pb.Empty()})));
    assert.equal(sample.snapshot.observationStartNs, 1_234_567_000n);
    assert.equal(sample.snapshot.observationEndNs, 1_234_567_000n);
    const expired = (await f.scheduler.submit(new pb.SubmitRequest({
      execute: resultRequest(await f.context(), 1),
      options: new pb.SubmitOptions({ deadlineMonotonicNs: 1_234_567n * 1000n, freshness:pb.RequestFreshness.REQUEST_FRESHNESS_DROP_IF_LATE }),
    })));
    assert.equal(expired.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
    assert.equal(expired.requestId, 0n);
    assert.equal(f.log.length, 0);
    // A positive sub-tick deadline rounds up, never to an already expired tick.
    const rounded = (await f.scheduler.submit(new pb.SubmitRequest({
      execute: resultRequest(await f.context(), 1),
      options: new pb.SubmitOptions({ deadlineMonotonicNs: 1_234_567_001n }),
    })));
    assert.equal(rounded.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
    assert.ok(rounded.requestId > 0n);
    clock.micros += 543;
    assert.equal((await platform.getMonotonicTime(new pb.Empty())).nanoseconds, 1_235_110_000n);
  } finally { (await f.close()); }
});

test('a missing host clock refuses scheduled execution without breaking direct execution', async () => {
  const { control } = makeResultControl({ micros: -1 });
  try {
    const inference = new VxInferenceServiceClient(reportTransport(control));
    const platform = new VxPlatformServiceClient(reportTransport(control));
    assert.equal((await platform.getMonotonicTime(new pb.Empty())).nanoseconds, 0n);
    const scheduled = (await inference.createRuntime(new pb.CreateRuntimeRequest({
      executionMode: pb.ExecutionMode.EXECUTION_MODE_SCHEDULED,
    })));
    assert.equal(scheduled.report.status, pb.NativeStatus.NATIVE_STATUS_TRANSPORT_UNSUPPORTED);
    assert.equal(scheduled.report.code, pb.OperationCode.OPERATION_CODE_SCHEDULER_CLOCK_UNAVAILABLE);
    assert.equal(scheduled.runtimeId, 0n);
    const direct = accepted((await inference.createRuntime(new pb.CreateRuntimeRequest())));
    (await inference.releaseRuntime(new pb.RuntimeRef(direct)));
  } finally { (await control.close()); }
});

test('hard deadlines retire queued and pending GPU work and release bounded admission', async () => {
  const clock = { micros: 1_000_000 };
  const f = (await resultFixture({ scheduled: true, clock, budget: new pb.RuntimeBudget({
    maxScheduledRequests: 1n, maxScheduledInputBytes: 65536n,
    maxUnconsumedResults: 1n, maxUnconsumedResultBytes: 48n,
  }) }));
  try {
    for (let index = 0; index < 24; index++) {
      const request = accepted((await f.scheduler.submit(new pb.SubmitRequest({
        execute: resultRequest(await f.context(), index),
        options: new pb.SubmitOptions({
          deadlineMonotonicNs: BigInt(clock.micros + 500) * 1000n,
          freshness: pb.RequestFreshness.REQUEST_FRESHNESS_DROP_IF_LATE,
        }),
      }))));
      const pending = index % 2 === 1;
      if (pending) {
        await startRequest(f, request);
        assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length, 1);
      }
      const dispatches = f.log.length;
      clock.micros += 500;
      let terminal = await f.scheduler.pollRequest(new pb.RequestRef(request));
      if(pending) {
        assert.equal(terminal.state,pb.RequestState.REQUEST_STATE_RUNNING);
        f.bridge.complete();
        terminal=await f.scheduler.pollRequest(new pb.RequestRef(request));
      }
      assert.equal(terminal.state, pb.RequestState.REQUEST_STATE_FAILED);
      assert.equal(terminal.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
      assert.equal(terminal.deadlineMissed, true);
      assert.equal(terminal.ownedInputBytes, 0n);
      assert.equal(f.log.length, dispatches, 'poll must never dispatch queued work');
      assert.equal(f.bridge.jobs.size, 0);
      const result = (await f.scheduler.takeRequestResult(new pb.RequestRef(request)));
      assert.deepEqual(result.outputs, []);
      assert.equal(result.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
      (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
    }
  } finally { (await f.close()); }
});

test('a committed direct GPU deadline preserves its proved execution evidence', async () => {
  const clock = { micros: 1_000_000 };
  const f = await resultFixture({ clock });
  try {
    const context = await f.context(), request = resultRequest(context, 7);
    request.budget = new pb.ExecutionBudget({ deadlineMonotonicNs: BigInt(clock.micros + 500) * 1000n });
    const pending = await beginExecution(f, request);
    assert.equal(pending.settled, false);
    clock.micros += 500;
    f.bridge.complete();
    const packet = await pending.promise;
    assert.equal(packet.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
    assert.equal(packet.report.code, pb.OperationCode.OPERATION_CODE_DEADLINE_EXCEEDED);
    assert.equal(packet.report.backend, 'webgpu');
    assert.equal(packet.report.route?.attested, true);
    assert.equal(packet.report.route?.selectedNodes, packet.report.route?.activeNodes);
    assert.ok(packet.report.route?.activeNodes > 0);
    assert.equal(packet.report.route?.fallbackNodes, 0);
    assert.equal(packet.report.route?.missingNodes, 0);
    assert.equal(packet.report.compilation?.policyMode, pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE);
    assert.equal(packet.report.compilation?.operatorFallback, pb.OperatorFallback.OPERATOR_FALLBACK_FORBID);
    assert.equal(packet.report.fallback?.operatorFallbackUsed ?? false, false);
    assert.equal(packet.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED);
    assert.deepEqual(packet.outputs, []);
    assert.equal(f.bridge.jobs.size, 0);
  } finally { await f.close(); }
});

test('soft deadlines keep pending results and report a missed target at completion', async () => {
  const clock = { micros: 1_000_000 };
  const f = (await resultFixture({ scheduled: true, clock }));
  try {
    const request = accepted((await f.scheduler.submit(new pb.SubmitRequest({
      execute: resultRequest(await f.context(), 7),
      options: new pb.SubmitOptions({ deadlineMonotonicNs: 1_001_000n * 1000n }),
    }))));
    await startRequest(f, request);
    clock.micros += 2000;
    assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(request))).state,
      pb.RequestState.REQUEST_STATE_RUNNING);
    f.bridge.complete();
    const info = accepted((await f.scheduler.pollRequest(new pb.RequestRef(request))));
    assert.equal(info.state, pb.RequestState.REQUEST_STATE_SUCCEEDED);
    assert.equal(info.deadlineMissed, true);
    const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(request))));
    assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(7));

    (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
  } finally { (await f.close()); }
});

test('LATEST supersedes pending GPU copies and queued requests without retaining their budgets', async () => {
  const f = (await resultFixture({ scheduled: true, budget: new pb.RuntimeBudget({
    maxScheduledRequests: 2n, maxScheduledInputBytes: 131072n,
    maxUnconsumedResults: 2n, maxUnconsumedResultBytes: 96n,
  }) }));
  const submit = async (value) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
    execute: resultRequest(await f.context(), value),
    options: new pb.SubmitOptions({ freshness: pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, streamKey: 9n }),
  }))));
  try {
    for (let index = 0; index < 24; index++) {
      const old = (await submit(1));
      await startRequest(f, old);
      const queued = (await submit(2));
      const latest = (await submit(3));
      assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(old))).state,pb.RequestState.REQUEST_STATE_RUNNING);
      f.bridge.complete();
      for (const obsolete of [old, queued]) {
        assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(obsolete))).state,
          pb.RequestState.REQUEST_STATE_SUPERSEDED);
        (await f.scheduler.releaseRequest(new pb.RequestRef(obsolete)));
      }
      assert.equal(f.bridge.jobs.size, 0);
      await startRequest(f, latest);
      f.bridge.complete();
      const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(latest))));
      assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(3));

      (await f.scheduler.releaseRequest(new pb.RequestRef(latest)));
    }
  } finally { (await f.close()); }
});

test('C scheduling uses host-clock aging before EDF and request order', async () => {
  const clock = { micros: 1_000_000 };
  const f = (await resultFixture({ scheduled: true, clock }));
  const submit = async (value, options = {}) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
    execute: resultRequest(await f.context(), value),
    options: new pb.SubmitOptions(options),
  }))));
  try {
    const old = (await submit(1));
    clock.micros += 20_000;
    const laterDeadline = (await submit(2, { priority: 1, deadlineMonotonicNs: 1_060_000n * 1000n }));
    const earlierDeadline = (await submit(3, { priority: 1, deadlineMonotonicNs: 1_050_000n * 1000n }));
    await startRequest(f, laterDeadline);
    const values = [...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).map(({ bytes }) => new DataView(
      bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(0, true));
    assert.deepEqual(values, [1, 3, 2]);
    f.bridge.complete();
    for (const request of [old, earlierDeadline, laterDeadline]) {
      const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(request))));

      (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
    }
  } finally { (await f.close()); }
});

test('C compilation refuses device resource limits before any GPU allocation or submission', async () => {
  for (const [name, value] of Object.entries({
    maxBufferSize: 47, maxStorageBufferBindingSize: 47,
    maxUniformBufferBindingSize: 15, maxStorageBuffersPerShaderStage: 1,
    maxUniformBuffersPerShaderStage: 0, maxBindingsPerBindGroup: 2, maxBindGroups: 0,
    maxComputeInvocationsPerWorkgroup: 63, maxComputeWorkgroupSizeX: 63,
    maxComputeWorkgroupSizeY: 0, maxComputeWorkgroupSizeZ: 0,
    maxComputeWorkgroupsPerDimension: 0,
  })) {
    const f = (await resultFixture({ limits: { ...DEVICE_LIMITS, [name]: value },
      compileStatus: pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED }));
    try {
      assert.equal(f.compiled.compiledModelId, 0n, name);
      assert.equal(f.log.length, 0, name);
      assert.equal(f.bridge.stats.calls.includes('vx_gpu_ensure'), false, name);
      assert.equal(f.bridge.stats.calls.includes('vx_gpu_begin'), false, name);
    } finally { (await f.close()); }
  }
  const f = (await resultFixture({ limits: { ...DEVICE_LIMITS, maxComputeWorkgroupStorageSize: 0 } }));
  try {
    const result = await execute(f,resultRequest(await f.context(),4));
    f.bridge.complete();
    assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(4));

  } finally { (await f.close()); }
});

import { VxPlanningServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
function boundedReluGraph(dimensions, shape) {
  return { format: 'volvox-graph/v1', dimensions,
    inputs: { input0: { dtype: 'float32', shape } },
    nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'input0' },
      outputs: { out: { tensor: 'out0', dtype: 'float32', shape } }, params: {} }],
    outputs: ['out0'] };
}
function shapedTensor(shape, value) {
  const values = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(value);
  return new pb.Tensor({ name: 'input0', shape: shape.map(BigInt),
    dtype: pb.DataType.DATA_TYPE_F32, inline: new Uint8Array(values.buffer) });
}

test('WebGPU proves positive pointwise domains and executes changing bounds', async () => {
  const graph = boundedReluGraph({ B: { min: 1, max: 4 }, S: { min: 2, max: 8, multiple_of: 2 } }, ['B', 'S', 3]);
  const f = (await resultFixture({ graph }));
  try {
    assert.equal(f.log.length, 0, 'compile never submits GPU work');
    const context = (await f.context());
    for (const shape of [[1, 2, 3], [4, 8, 3], [2, 4, 3], [1, 2, 3]]) {
      const result = await execute(f,executeRequest(context,[shapedTensor(shape,3)]));
      assert.deepEqual((await resultValues(f.inference, result)), Array(shape.reduce((a,b) => a*b,1)).fill(3));

    }
    const refused = (await f.inference.execute(executeRequest(context,[shapedTensor([1,3,3],1)])));
    assert.equal(refused.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  } finally { (await f.close()); }
});

test('WebGPU preserves reused host inputs when growing shapes move arena offsets', async () => {
  const graph = declareStateless({format: 'volvox-graph/v1', dimensions: {S: {min: 1, max: 128}},
    inputs: {input0: {dtype: 'float32', shape: ['S']}, constant: {dtype: 'float32', shape: [32]}},
    nodes: [
      {id: 'growing', opType: 'ReLU', inputs: {input: 'input0'},
        outputs: {out: {tensor: 'out0', dtype: 'float32', shape: ['S']}}, params: {}},
      {id: 'invariant', opType: 'ReLU', inputs: {input: 'constant'},
        outputs: {out: {tensor: 'out1', dtype: 'float32', shape: [32]}}, params: {}},
    ], outputs: ['out0', 'out1']});
  graph.execution.invariant_inputs = ['constant'];
  const f = await resultFixture({graph});
  const constants = Float32Array.from({length: 32}, (_, index) => index + 1001);
  const constant = () => new pb.Tensor({name: 'constant', shape: [32n], dtype: pb.DataType.DATA_TYPE_F32,
    inline: new Uint8Array(constants.buffer)});
  try {
    const reused = await f.context();
    for (const [index, length] of [1, 16, 32, 64, 128, 16].entries()) {
      const control = await f.context();
      const request = executeRequest(reused, [shapedTensor([length], index + 1), ...(index ? [] : [constant()])]);
      if (index) request.inputs.reuse = ['constant'];
      const actual = await execute(f, request);
      const expected = await execute(f, executeRequest(control, [shapedTensor([length], index + 1), constant()]));
      for (const name of ['out0', 'out1']) {
        assert.deepEqual(await resultValues(f.inference, actual, name), await resultValues(f.inference, expected, name),
          `${name} at growing length ${length}`);
      }
      assert.deepEqual(await resultValues(f.inference, actual, 'out1'), [...constants]);
      assert.equal(f.bridge.jobs.size, 0, 'completed executions retire all snapshots and fences');
    }
  } finally {await f.close();}
});

test('one pending WebGPU batch owns shared completion and independent lane snapshots', async () => {
  const f = (await resultFixture({ scheduled: true,
    graph: boundedReluGraph({ B: { min: 1, max: 4 } }, ['B', 3]),
    budget: new pb.RuntimeBudget({ maxUnconsumedResults: 4n, maxUnconsumedResultBytes: 192n }),
  }));
  try {
    const planning = new VxPlanningServiceClient(reportTransport(f.control));
    const planned = accepted((await planning.createGraphPlan(new pb.CreateGraphPlanRequest({ modelId: f.model.modelId }))));
    const contract = (await planning.getGraphPlan(new pb.GraphPlanRef(planned))).plan.independentBatch;
    assert.ok(contract.supported, JSON.stringify(contract));
    (await planning.releaseGraphPlan(new pb.GraphPlanRef(planned)));
    for (const count of [2, 4, 3]) {
      const requests = (await Promise.all(Array.from({ length: count }, async (_, lane) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
        execute: executeRequest(await f.context(), [shapedTensor([1, 3], lane + 1)]) })))))));
      const before = f.log.length;
      assert.equal((await startRequest(f, requests[0])).state, pb.RequestState.REQUEST_STATE_RUNNING);
      assert.equal(f.log.length - before, 1, `all lanes must share one physical dispatch: ${count} lanes; ${JSON.stringify(f.log.slice(before).map(p => p.params))}`);
      assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length, 1, 'all lanes must share one staging snapshot');
      if (count === 3) {
        (await f.scheduler.cancelRequest(new pb.RequestRef(requests[1])));
        assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length, 1, 'cancelling one lane must preserve the others');
      }
      f.bridge.complete();
      for (const [lane, request] of requests.entries()) {
        const info = (await f.scheduler.pollRequest(new pb.RequestRef(request)));
        if (count === 3 && lane === 1) {
          assert.equal(info.state, pb.RequestState.REQUEST_STATE_CANCELLED);
        } else {
          assert.equal(info.state, pb.RequestState.REQUEST_STATE_SUCCEEDED, info.report.message);
          const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(request))));
          assert.deepEqual((await resultValues(f.inference, result)), Array(3).fill(lane + 1));

        }
        (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
      }
      assert.equal(f.bridge.jobs.size, 0);
    }
  } finally { (await f.close()); }
});

test('a failed GPU batch completion retires every lane and its shared reservation', async () => {
  const f = (await resultFixture({scheduled:true,
    graph:boundedReluGraph({B:{min:1,max:4}},['B',3]),
    budget:new pb.RuntimeBudget({maxUnconsumedResults:4n,maxUnconsumedResultBytes:192n})}));
  try {
    for (let repeat=0;repeat<3;repeat++) {
      const requests=(await Promise.all(Array.from({length:4},async (_,i)=>accepted((await f.scheduler.submit(new pb.SubmitRequest({
        execute:executeRequest(await f.context(),[shapedTensor([1,3],i)])})))))));
      await startRequest(f, requests[0]);
      assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length,1);
      f.bridge.complete(true);
      for (const request of requests) {
        const state=(await f.scheduler.pollRequest(new pb.RequestRef(request)));
        assert.equal(state.state,pb.RequestState.REQUEST_STATE_FAILED,state.report.message);
        assert.deepEqual((await f.scheduler.takeRequestResult(new pb.RequestRef(request))).outputs,[]);
        (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
      }
      assert.equal(f.bridge.jobs.size,0);
    }
  } finally {(await f.close());}
});

test('required GPU rows preserve unaligned prefixes and completed packet snapshots', async () => {
 const shape=[1,4,5],graph=declareIncremental(boundedReluGraph({},shape));
 const f=await resultFixture({graph});
 const input=values=>new pb.Tensor({name:'input0',shape:shape.map(BigInt),dtype:pb.DataType.DATA_TYPE_F32,inline:new Uint8Array(Float32Array.from(values).buffer)});
 const request=(context,values,position)=>graphRequest(pb,graph,context.contextId,[input(values)],position===undefined
  ?{entrypoint:'initialize',lengths:[1]}:{entrypoint:'transition',positions:[position]});
 try {
  const options={graph:new pb.GraphExecution({rowMode:pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true})};
  const context=await f.context(options),values=Array.from({length:20},(_,i)=>i%2?i:-i),expected=values.map(v=>Math.max(0,v));
  const initial=await execute(f,request(context,values));
  for(const position of [1,2]) {
   values.fill(40+position,position*5,(position+1)*5);expected.fill(40+position,position*5,(position+1)*5);
   const before=f.log.length,packet=await execute(f,request(context,values,position));
   assert.equal(f.log.length-before,3);assert.deepEqual(await resultValues(f.inference,packet),expected);
  }
  assert.deepEqual(await resultValues(f.inference,initial),Array.from({length:20},(_,i)=>i%2?i:0));
  const fresh=await f.context(options),replayed=await execute(f,request(fresh,Array(20).fill(7)));
  assert.deepEqual(await resultValues(f.inference,replayed),Array(20).fill(7));
 } finally {await f.close();}
});

test('Generate returns byte-exact nonempty ID snapshots across one, two, and three step chunks', async () => {
  const sequence = 6, domain = 3, expected = [2, 2, 2, 2, 2, 2];
  const graph = declareIncremental({format: 'volvox-graph/v1', dimensions: {},
    inputs: {tokens: {dtype: 'int32', shape: [1, sequence]}, keep: {dtype: 'int32', shape: [1, sequence]}},
    nodes: [
      {id: 'embedding', opType: 'Embedding', inputs: {input: 'tokens', weight: 'table'}, params: {},
        outputs: {out: {tensor: 'embedding', dtype: 'float32', shape: [1, sequence, domain]}}},
      ...['q', 'k', 'v'].map(name => ({id: name, opType: 'ReLU', inputs: {input: 'embedding'}, params: {},
        outputs: {out: {tensor: name, dtype: 'float32', shape: [1, sequence, domain]}}})),
      {id: 'attention', opType: 'CrossSDPA', inputs: {q: 'q', k: 'k', v: 'v', mask: 'keep'}, params: {heads: 1, causal: true},
        outputs: {out: {tensor: 'logits', dtype: 'float32', shape: [1, sequence, domain]}}},
      {id: 'argmax', opType: 'ArgMax', inputs: {input: 'logits'}, params: {axis: -1, keepdims: false},
        outputs: {out: {tensor: 'generated', dtype: 'int32', shape: [1, sequence]}}},
    ], outputs: ['logits', 'generated']}, {updates: ['tokens', 'keep'], generation: {initialize_entrypoint: 'initialize',
      transition_entrypoint: 'transition', token_input: 'tokens', token_output: 'generated', keep_input: 'keep',
      logits_output: 'logits', logits_axis: 2, id_domain_size: domain, output_name: 'generated_ids', sequence_axis: 1}});
  const f = await resultFixture({graph, weights: safetensors([
    {name: 'table', shape: [domain, domain], data: Float32Array.of(1, 2, 9, 1, 2, 9, 1, 2, 9)},
  ])});
  try {
    const context = await f.context({graph: undefined, autoregressive: new pb.AutoregressiveExecution({
      sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY, maxSteps: BigInt(sequence)})});
    const tensor = (name, shape, data) => new pb.Tensor({name, shape: shape.map(BigInt),
      dtype: data instanceof Int32Array ? pb.DataType.DATA_TYPE_I32 : pb.DataType.DATA_TYPE_F32,
      inline: new Uint8Array(data.buffer)});
    const values = [tensor('tokens', [1, sequence], new Int32Array(sequence)),
      tensor('keep', [1, sequence], Int32Array.of(1, 0, 0, 0, 0, 0)), activeLengths(pb, [1])];
    const packets = [];
    let offset = 0;
    for (const count of [1, 2, 3]) {
      const snapshots = f.bridge.stats.snapshotSizes.length, readbacks = f.bridge.stats.readbackSizes.length;
      const pending = beginCall(f.bridge, () => f.inference.generate(new pb.GenerateRequest({contextId: context.contextId,
        ...(offset === 0 ? {initialInputs: new pb.GenerationInputs({values})} : {}),
        budget: new pb.GenerationBudget({maxSteps: count})})));
      await pending.started;
      assert.equal(pending.settled, false);
      assert.deepEqual([...f.bridge.jobs.values()].filter(job => job.bytes.length).map(job => job.bytes.length), [count * 4]);
      f.bridge.complete();
      const packet = accepted(await pending.promise), output = packet.outputs[0];
      assert.equal(f.bridge.jobs.size, 0, 'each Generate chunk retires every device ticket');
      assert.deepEqual([...new Int32Array(output.inline.slice().buffer)], expected.slice(offset, offset + count));
      assert.deepEqual(output.shape, [1n, BigInt(count)]);
      assert.equal(packet.progress.steps, BigInt(count));
      offset += count;
      assert.equal(packet.progress.totalSteps, BigInt(offset));
      assert.deepEqual(f.bridge.stats.snapshotSizes.slice(snapshots).filter(size => size > 0), [count * 4]);
      assert.ok(f.bridge.stats.readbackSizes.slice(readbacks).includes(count * 4));
      packets.push(packet);
    }
    assert.equal(packets.at(-1).progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_COMPLETED);
    assert.deepEqual(packets.flatMap(packet => [...new Int32Array(packet.outputs[0].inline.slice().buffer)]), expected);
    const noOutputContext = await f.context({graph: undefined, autoregressive: new pb.AutoregressiveExecution({
      sampling: pb.SamplingMode.SAMPLING_MODE_GREEDY, maxSteps: BigInt(sequence)})});
    const snapshots = f.bridge.stats.snapshotSizes.length;
    const pending = beginCall(f.bridge, () => f.inference.generate(new pb.GenerateRequest({
      contextId: noOutputContext.contextId, initialInputs: new pb.GenerationInputs({values}),
      outputs: new pb.OutputOptions({selection: new pb.TensorOutputSelection({names: []})}),
      budget: new pb.GenerationBudget({maxSteps: 1})})));
    await pending.started;
    assert.equal(pending.settled, false);
    assert.deepEqual(f.bridge.stats.snapshotSizes.slice(snapshots), [0], 'unselected IDs require only a completion fence');
    f.bridge.complete();
    const packet = accepted(await pending.promise);
    assert.deepEqual(packet.outputs, []);
    assert.equal(packet.progress.steps, 1n);
    assert.equal(packet.progress.totalSteps, 1n);
    assert.equal(f.bridge.jobs.size, 0, 'empty selection retires its completion ticket');
  } finally { await f.close(); }
});

test('a draining trace owns its GPU ticket independently of a pending Execute', async () => {
 const f=await resultFixture();f.bridge.enableTiming();const profiling=new VxProfilingServiceClient(reportTransport(f.control));
 try {
  const context=await f.context(),timing=new pb.TraceOptions({deviceTiming:true});
  const trace=accepted(await profiling.startTrace(new pb.StartTraceRequest({runtimeId:f.runtime.runtimeId,options:timing})));
  const pending=await beginExecution(f,resultRequest(context,3));
  assert.equal(pending.settled,false);assert.equal(f.bridge.timingTickets.size,1);
  const stopping=profiling.stopTrace(new pb.TraceRef(trace));
  for(let i=0;i<8;i++)await profiling.getTrace(new pb.TraceRef(trace));
  assert.equal(accepted(await profiling.getTrace(new pb.TraceRef(trace))).state,pb.TraceState.TRACE_STATE_DRAINING);
  await profiling.releaseTrace(new pb.TraceRef(trace));assert.equal(f.bridge.timingTickets.size,0);
  assert.equal(accepted(await stopping).state,pb.TraceState.TRACE_STATE_READY);
  assert.equal(pending.settled,false,'trace release does not publish pending outputs');
  f.bridge.complete();assert.deepEqual(await resultValues(f.inference,accepted(await pending.promise)),Array(12).fill(3));
  const next=accepted(await profiling.startTrace(new pb.StartTraceRequest({runtimeId:f.runtime.runtimeId,options:timing})));
  const second=await execute(f,resultRequest(context,7));assert.deepEqual(await resultValues(f.inference,second),Array(12).fill(7));
  const ready=accepted(await profiling.stopTrace(new pb.TraceRef(next)));
  assert.equal(ready.state,pb.TraceState.TRACE_STATE_READY);assert.ok(ready.devices.some(d=>d.deviceIntervals.passes>0n));
  const page=accepted(await profiling.listTraceEvents(new pb.ListTraceEventsRequest({traceId:next.traceId})));
  assert.equal(page.events.filter(e=>e.device!==undefined).length,1);await profiling.releaseTrace(new pb.TraceRef(next));
 } finally {await f.close();}
});

test('host node tracing leaves device timing disabled even on a timing-capable GPU', async () => {
  const f = await resultFixture();
  f.bridge.enableTiming();
  const profiling = new VxProfilingServiceClient(reportTransport(f.control));
  try {
    const context = await f.context();
    const trace = accepted(await profiling.startTrace(new pb.StartTraceRequest({
      runtimeId: f.runtime.runtimeId, options: new pb.TraceOptions({detail: pb.TraceDetail.TRACE_DETAIL_NODES}),
    })));
    const callsBefore = f.bridge.stats.calls.length;
    const result = await execute(f,resultRequest(context, 5));
    assert.equal(f.bridge.timingTickets.size, 0);
    assert.ok(f.bridge.stats.calls.slice(callsBefore).every(name => !name.includes('trace')));
    const collecting = accepted(await profiling.getTrace(new pb.TraceRef(trace)));
    assert.equal(collecting.state, pb.TraceState.TRACE_STATE_COLLECTING);
    assert.deepEqual(collecting.devices, []);
    f.bridge.complete();

    assert.deepEqual(await resultValues(f.inference, result), Array(12).fill(5));
    const ready = accepted(await profiling.stopTrace(new pb.TraceRef(trace)));
    assert.equal(ready.pendingDeviceIntervals, 0n);
    const page = accepted(await profiling.listTraceEvents(new pb.ListTraceEventsRequest({traceId: trace.traceId})));
    assert.ok(page.events.some(event => event.host && event.node !== undefined));
    assert.ok(page.events.every(event => event.device === undefined));

    await profiling.releaseTrace(new pb.TraceRef(trace));
  } finally { await f.close(); }
});

test('an annotating node trace labels each node\'s WebGPU dispatches, and nothing else does', async () => {
  const f = await resultFixture();
  const profiling = new VxProfilingServiceClient(reportTransport(f.control));
  try {
    const context = await f.context();
    const trace = accepted(await profiling.startTrace(new pb.StartTraceRequest({
      runtimeId: f.runtime.runtimeId, options: new pb.TraceOptions({detail: pb.TraceDetail.TRACE_DETAIL_NODES,
        external: new pb.TraceExternalOptions({annotations: true})}),
    })));
    const result = await execute(f,resultRequest(context, 5));
    f.bridge.complete();

    assert.deepEqual(await resultValues(f.inference, result), Array(12).fill(5));
    assert.deepEqual(f.bridge.stats.labels, ['0 ReLU -> out0', null]);
    assert.deepEqual(f.bridge.stats.dispatchLabels, ['0 ReLU -> out0']);
    const ready = accepted(await profiling.stopTrace(new pb.TraceRef(trace)));
    const row = ready.external.find(r => r.mechanism === 'webgpu_debug_group');
    assert.equal(row.backend, 'webgpu');
    assert.equal(row.status, pb.ObservationStatus.OBSERVATION_STATUS_AVAILABLE);
    assert.equal(row.ranges, 1n);

    await profiling.releaseTrace(new pb.TraceRef(trace));
    // Ordinary execution never labels.
    const plain = await execute(f,resultRequest(context, 2));
    f.bridge.complete();

    assert.equal(f.bridge.stats.labels.length, 2);
    assert.deepEqual(f.bridge.stats.dispatchLabels, ['0 ReLU -> out0', null]);

  } finally { await f.close(); }
});

for (const detail of [pb.TraceDetail.TRACE_DETAIL_BASIC, pb.TraceDetail.TRACE_DETAIL_NODES]) {
  test(`GPU timing without timestamp support retains host detail ${detail} and reports unsupported`, async () => {
    const f = await resultFixture();
    const profiling = new VxProfilingServiceClient(reportTransport(f.control));
    try {
      const context = await f.context();
      const trace = accepted(await profiling.startTrace(new pb.StartTraceRequest({
        runtimeId: f.runtime.runtimeId, options: new pb.TraceOptions({detail, deviceTiming: true}),
      })));
      assert.equal(trace.options.deviceTiming, true);
      const result = await execute(f,resultRequest(context, 6));
      f.bridge.complete();

      assert.deepEqual(await resultValues(f.inference, result), Array(12).fill(6));
      const ready = accepted(await profiling.stopTrace(new pb.TraceRef(trace)));
      assert.equal(ready.state, pb.TraceState.TRACE_STATE_READY);
      assert.equal(ready.events.dropped, 0n);
      assert.equal(ready.pendingDeviceIntervals, 0n);
      assert.equal(ready.devices.length, 1);
      assert.equal(ready.devices[0].support, pb.TraceTimingSupport.TRACE_TIMING_SUPPORT_UNSUPPORTED);
      assert.equal(ready.devices[0].nodeTimingAvailable, false);
      assert.equal(ready.devices[0].deviceIntervals.unsupportedPasses, 1n);
      assert.equal(ready.devices[0].deviceIntervals.nodes, 0n);
      assert.equal(ready.devices[0].splitsPasses, false);
      const page = accepted(await profiling.listTraceEvents(new pb.ListTraceEventsRequest({traceId: trace.traceId})));
      assert.ok(page.events.every(event => event.host && !event.device));
      assert.equal(page.events.some(event => event.node), detail === pb.TraceDetail.TRACE_DETAIL_NODES);

      await profiling.releaseTrace(new pb.TraceRef(trace));
    } finally { await f.close(); }
  });
}

for (const ending of ['complete', 'cancel', 'release', 'transport', 'close', 'failed'])
test(`debug WebGPU readbacks remain asynchronous and terminate safely: ${ending}`, async () => {
  const f = await resultFixture();
  const debug = new VxDebugServiceClient(reportTransport(f.control));
  let closed = false;
  try {
    const created = accepted(await debug.createDebugSession(new pb.CreateDebugSessionRequest({
      execute: resultRequest(await f.context(),7),
      capture: new pb.DebugCapture({values: true}),
    })));
    const abort = new AbortController();
    const completion = debug.stepDebugSession(new pb.StepDebugSessionRequest({debugSessionId: created.debugSessionId,
      expectedRevision: created.revision}), {signal: abort.signal});
    void completion.catch(() => {});
    for (let turn = 0; f.bridge.jobs.size === 0 && turn < 64; turn++)
      await debug.getDebugSession(new pb.DebugSessionRef(created));
    assert.equal([...f.bridge.jobs.values()].filter(job=>job.bytes.length>0).length, 1);
    for (let i = 0; i < 3; i++) {
      const pending = accepted(await debug.getDebugSession(new pb.DebugSessionRef(created)));
      assert.equal(pending.state, pb.DebugState.DEBUG_STATE_RUNNING);
      assert.equal(pending.nextStep, 0, 'step is incomplete until after snapshot publication');
      assert.equal(pending.events.count, 1n, 'only the immutable before event is readable');
    }
    const busy = await debug.stepDebugSession(new pb.StepDebugSessionRequest({debugSessionId: created.debugSessionId,
      expectedRevision: (await debug.getDebugSession(new pb.DebugSessionRef(created))).revision}));
    assert.equal(busy.report.status, pb.NativeStatus.NATIVE_STATUS_BUSY);
    if (ending === 'complete' || ending === 'failed') {
      const finishing=beginCall(f.bridge,()=>completion);
      for(let turn=0; !finishing.settled && turn<100; turn++){f.bridge.complete(ending==='failed');await new Promise(resolve=>setImmediate(resolve));}
      const info = await finishing.promise;
      assert.equal(info.state, ending === 'complete' ? pb.DebugState.DEBUG_STATE_COMPLETED : pb.DebugState.DEBUG_STATE_FAILED);
      // A failing model is session state: the RPC succeeds and `failure` explains the model.
      assert.equal(info.report.status, pb.NativeStatus.NATIVE_STATUS_OK);
      assert.equal(info.failure !== undefined, ending === 'failed');
      const packet=await debug.takeDebugResult(new pb.DebugSessionRef(created));
      assert.equal(packet.outputs.length,ending==='complete'?1:0);
      assert.equal((await debug.takeDebugResult(new pb.DebugSessionRef(created))).report.status,pb.NativeStatus.NATIVE_STATUS_NOT_FOUND);
      if (ending === 'failed') assert.notEqual(info.failure.status, pb.NativeStatus.NATIVE_STATUS_OK);
      const page = accepted(await debug.listDebugEvents(new pb.ListDebugEventsRequest({debugSessionId: created.debugSessionId})));
      const tensors = page.events.flatMap(event => event.snapshots);
      assert.equal(tensors.length, 2);
      if (ending === 'complete') {
        const snapshot = accepted(await debug.readDebugTensor(new pb.ReadDebugTensorRequest({debugSessionId: created.debugSessionId, snapshotId: 2n})));
        assert.deepEqual([...new Float32Array(snapshot.data.slice().buffer)], new Array(12).fill(7));
      } else assert.equal(tensors[1].status, pb.DebugTensorStatus.DEBUG_TENSOR_STATUS_FAILED);
    } else if (ending === 'cancel') {
      const cancelled = accepted(await debug.cancelDebugSession(new pb.DebugSessionRef(created)));
      assert.equal(cancelled.state, pb.DebugState.DEBUG_STATE_RUNNING);
      f.bridge.complete();
      assert.equal((await completion).state, pb.DebugState.DEBUG_STATE_CANCELLED);
    } else if (ending === 'release') {
      await debug.releaseDebugSession(new pb.DebugSessionRef(created));
      f.bridge.complete();
      assert.equal((await completion).state, pb.DebugState.DEBUG_STATE_CANCELLED);
    } else if (ending === 'transport') {
      abort.abort();
      await assert.rejects(completion);
      f.bridge.complete();
      const cancelled = accepted(await debug.getDebugSession(new pb.DebugSessionRef(created)));
      assert.equal(cancelled.state, pb.DebugState.DEBUG_STATE_CANCELLED);
    } else {
      await f.close(); closed = true;
      await assert.rejects(completion);
    }
    assert.equal(f.bridge.jobs.size, 0, 'completion/cancel/release/close retires the staging ticket');
  } finally { if (!closed) await f.close(); }
});

test('no-output Execute waits for the device and propagates late device failure', async () => {
 for(const failed of [false,true]) {
  const f=await resultFixture();
  try {
   const context=await f.context(),request=resultRequest(context,7);
   request.outputs=new pb.OutputOptions({selection:new pb.TensorOutputSelection({names:[]})});
   const pending=await beginExecution(f,request);
   assert.equal(pending.settled,false);
   assert.equal(f.bridge.jobs.size,1,'no-output execution retains only a completion ticket');
   assert.equal([...f.bridge.jobs.values()][0].bytes.length,0,'no selected output is copied');
   assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(context))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING);
   f.bridge.complete(failed);const packet=await pending.promise;
   assert.deepEqual(packet.outputs,[]);
   assert.equal(packet.report.status,failed?pb.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED:pb.NativeStatus.NATIVE_STATUS_OK);
   assert.equal(packet.progress.state,failed?pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_FAILED:pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY);
   assert.equal(f.bridge.jobs.size,0);
  } finally {await f.close();}
 }
});

test('LATEST transfers a single queued result reservation only after incoming admission succeeds', async () => {
 const f=await resultFixture({scheduled:true,budget:new pb.RuntimeBudget({maxScheduledRequests:2n,maxScheduledInputBytes:131072n,maxUnconsumedResults:1n,maxUnconsumedResultBytes:48n})});
 try {
  const oldContext=await f.context(),newContext=await f.context();
  const options=new pb.SubmitOptions({freshness:pb.RequestFreshness.REQUEST_FRESHNESS_LATEST,streamKey:44n});
  const old=accepted(await f.scheduler.submit(new pb.SubmitRequest({execute:resultRequest(oldContext,2),options})));
  const invalid=resultRequest(newContext,9);invalid.inputs.values[0].shape=[1n];
  assert.notEqual((await f.scheduler.submit(new pb.SubmitRequest({execute:invalid,options}))).report.status,0);
  assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(old))).state,pb.RequestState.REQUEST_STATE_QUEUED);
  assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(oldContext))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_QUEUED);
  const latest=accepted(await f.scheduler.submit(new pb.SubmitRequest({execute:resultRequest(newContext,9),options})));
  assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(old))).state,pb.RequestState.REQUEST_STATE_SUPERSEDED);
  assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(oldContext))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_READY);
  assert.equal((await f.inference.getExecutionContextInfo(new pb.ExecutionContextRef(newContext))).state,pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_QUEUED);
  assert.equal(f.log.length,0,'read-only polling does not dispatch the queued replacement');
  await startRequest(f,latest);f.bridge.complete();
  const packet=accepted(await f.scheduler.takeRequestResult(new pb.RequestRef(latest)));
  assert.deepEqual(await resultValues(f.inference,packet),Array(12).fill(9));
  await f.scheduler.releaseRequest(new pb.RequestRef(old));await f.scheduler.releaseRequest(new pb.RequestRef(latest));
 } finally {await f.close();}
});

for (const cancellation of ['context', 'session', 'release'])
for (const ending of ['take', 'release', 'close'])
test(`${cancellation} cancellation drains paused debug GPU work without another command: ${ending}`, async () => {
  const graph = declareStateless({format: 'volvox-graph/v1', dimensions: {},
    inputs: {input0: {shape: RESULT_SHAPE, dtype: 'float32'}}, outputs: ['out0'],
    nodes: ['middle', 'out0'].map((name, index) => ({id: name, opType: 'ReLU',
      inputs: {input: index ? 'middle' : 'input0'}, params: {},
      outputs: {out: {tensor: name, shape: RESULT_SHAPE, dtype: 'float32'}}})),
  });
  const f = await resultFixture({graph, preserveNodeBoundaries: true});
  const debug = new VxDebugServiceClient(reportTransport(f.control));
  let closed = false;
  try {
    const context = await f.context(), contextRef = new pb.ExecutionContextRef(context);
    const retained = await execute(f, resultRequest(context, 7));
    let info = accepted(await debug.createDebugSession(new pb.CreateDebugSessionRequest({
      execute: resultRequest(context, 9), capture: new pb.DebugCapture({inputs: false, outputs: false, values: false}),
    })));
    info = accepted(await debug.stepDebugSession(new pb.StepDebugSessionRequest({
      debugSessionId: info.debugSessionId, expectedRevision: info.revision})));
    assert.equal(info.state, pb.DebugState.DEBUG_STATE_PAUSED);
    assert.equal(f.bridge.jobs.size, 0);
    const ref = new pb.DebugSessionRef(info), snapshots = f.bridge.stats.snapshotSizes.length;
    for (let repeat = 0; repeat < 2; repeat++) {
      if (cancellation === 'context')
        assert.equal((await f.inference.cancelExecutionContext(contextRef)).status, pb.NativeStatus.NATIVE_STATUS_OK);
      else if (cancellation === 'session')
        assert.equal(accepted(await debug.cancelDebugSession(ref)).state, pb.DebugState.DEBUG_STATE_RUNNING);
      else
        assert.equal((await debug.releaseDebugSession(ref)).status, pb.NativeStatus.NATIVE_STATUS_OK);
    }
    assert.deepEqual(f.bridge.stats.snapshotSizes.slice(snapshots), [0], 'one completion fence owns the cancelled work');
    assert.equal(accepted(await f.inference.getExecutionContextInfo(contextRef)).state,
      pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_RUNNING);
    if (ending === 'release') {
      await debug.releaseDebugSession(ref);
      await f.inference.releaseExecutionContext(contextRef);
    }
    if (ending === 'close') {
      closed = true;
      let closeFinished = false;
      const closing = f.control.close().then(() => { closeFinished = true; });
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(closeFinished, false, 'owner close waits for the retained cancellation fence');
      f.bridge.complete();
      await closing;
    } else {
      f.bridge.complete();
      for (let turn = 0; f.bridge.jobs.size && turn < 50; turn++)
        await new Promise(resolve => setImmediate(resolve));
      assert.equal(f.bridge.jobs.size, 0, 'the cancellation observer drained the final fence');
      if (ending === 'take') {
        assert.equal(accepted(await f.inference.getExecutionContextInfo(contextRef)).state,
          pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_CANCELLED);
        if (cancellation === 'release') {
          assert.equal((await debug.getDebugSession(ref)).report.status, pb.NativeStatus.NATIVE_STATUS_HANDLE_DISPOSED);
        } else {
          info = accepted(await debug.getDebugSession(ref));
          assert.equal(info.state, pb.DebugState.DEBUG_STATE_CANCELLED);
          assert.equal(info.resultAvailable, true);
          const packet = accepted(await debug.takeDebugResult(ref));
          assert.deepEqual(packet.outputs, []);
          assert.equal(packet.progress.state, pb.ExecutionContextState.EXECUTION_CONTEXT_STATE_CANCELLED);
          assert.equal((await debug.takeDebugResult(ref)).report.status, pb.NativeStatus.NATIVE_STATUS_NOT_FOUND);
          await debug.releaseDebugSession(ref);
        }
      }
    }
    assert.equal(f.bridge.jobs.size, 0);
    assert.deepEqual(await resultValues(f.inference, retained), Array(12).fill(7));
  } finally { if (!closed) await f.close(); }
});
