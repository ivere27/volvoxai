import { reportTransport, checkedReport } from '../tools/proto_report_fixture.mjs';
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
const FULL_WASM = path.join(ROOT, 'dist', version, 'volvoxai.full.wasm');

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
  let nextTicket = 1;
  let wakeup = () => {};
  const stats = { pending: 0, calls: [] };
  let recorded = null;
  let memory = null;
  const words = () => new Uint32Array(memory.buffer);
  const bytes = (at, length) => new Uint8Array(memory.buffer, at, length);

  const imports = {
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
      log.push({ variants, bindings, params });

      /* Execute what the descriptor describes. An operator the engine
       * claims but this host cannot run is a planning bug, not a gap, so an
       * unknown shader fails rather than quietly producing nothing. */
      const read = (index) => {
        const span = spans.get(bindings[index].hostPointer);
        if (!span) return null;
        return new Float32Array(span.bytes.buffer, span.bytes.byteOffset,
          span.bytes.byteLength >> 2);
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
      const span = spans.get(hostPointer);
      if (!span) return -1;
      const ticket = nextTicket++;
      jobs.set(ticket, { bytes: span.bytes.slice(offset, offset + size), ready: false, failed: false });
      stats.pending++;
      return ticket;
    },
    vx_gpu_readback(ticket, destination, size) {
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
  return { imports: counted, stats, jobs, setWakeup(callback) { wakeup = callback; }, attach(attached) { memory = attached; },
    complete(failed = false) { for (const job of jobs.values()) { job.ready = true; job.failed = failed; } wakeup(); },
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
        compiledModelId: compiled.compiledModelId,
      }));
    assert.equal(context.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      context.report?.message);
    const executeOnce = async () => (await inference.execute(new api.pb.ExecuteRequest({
      contextId: context.contextId,
      inputs: [new api.pb.Tensor({
        name: 'input0',
        dtype: api.pb.DataType.DATA_TYPE_F32,
        shape: [N, H, W, C].map(BigInt),
        inline: new Uint8Array(input.buffer, input.byteOffset, input.byteLength),
      })],
    })));

    const result = await executeOnce();
    assert.equal(result.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK, result.report?.message);
    assert.equal(result.state, api.pb.ResultState.RESULT_STATE_PENDING);
    const pending = await inference.getResult(new api.pb.ResultRef({ resultId: result.resultId }));
    assert.equal(pending.state, api.pb.ResultState.RESULT_STATE_PENDING);
    await bridge.waitForCompletion();
    const ready = await inference.getResult(new api.pb.ResultRef({ resultId: result.resultId }));
    assert.equal(ready.state, api.pb.ResultState.RESULT_STATE_READY);
    assert.equal(ready.executionId, result.executionId);

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

    const output = await inference.readOutput(new api.pb.ReadOutputRequest({
      resultId: result.resultId, name: 'out0',
    }));
    assert.equal(output.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      output.report?.message);
    const payload = output.tensor?.inline;
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
 * GetResult completion is the caller's half of the pending-readback contract, and
 * every operator added to the planner is expected to travel it unchanged.
 */
async function runOnWebGpu(api, graph, inputs, inputShape) {
  const batches = Array.isArray(inputs) ? inputs : [inputs];
  const log = [];
  const bridge = createSimulatingBridge(log);
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
      compiledModelId: compiled.compiledModelId,
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
    const request = new api.pb.ExecuteRequest({
      contextId: context.contextId,
      inputs: tensors.map(({ name, shape, data }) => new api.pb.Tensor({
        name,
        dtype: api.pb.DataType.DATA_TYPE_F32,
        shape: shape.map(BigInt),
        inline: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
      })),
    });
    const result = await inference.execute(request);
    assert.equal(result.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK, result.report?.message);
    await bridge.waitForCompletion();
    const ready = await inference.getResult(new api.pb.ResultRef({ resultId: result.resultId }));
    assert.equal(ready.state, api.pb.ResultState.RESULT_STATE_READY, ready.report?.message);
    route = result.report?.route;

    const output = await inference.readOutput(new api.pb.ReadOutputRequest({
      resultId: result.resultId, name: 'out0',
    }));
    assert.equal(output.report?.status, api.pb.NativeStatus.NATIVE_STATUS_OK,
      output.report?.message);
    const payload = output.tensor?.inline;
    outputs.push(new Float32Array(payload.buffer.slice(
      payload.byteOffset, payload.byteOffset + payload.byteLength)));
    await inference.releaseResult(new api.pb.ResultRef({ resultId: result.resultId }));
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
import { VxInferenceServiceClient, VxSchedulerServiceClient, VxPlatformServiceClient } from '../runtime/generated/typescript/volvoxai_ffi.js';
import { ModelControlWasmDispatchFactory } from '../ts/core/ModelControlWasm.js';
import { wasmReleaseModuleImports } from '../ts/core/WasmReleaseModule.js';

const RESULT_SHAPE = [1, 2, 3, 2];
const RESULT_GRAPH = { format: 'volvox-graph/v1', dimensions: {},
  inputs: { input0: { shape: RESULT_SHAPE, dtype: 'float32' } }, outputs: ['out0'],
  nodes: [{ id: 'relu', opType: 'ReLU', inputs: { input: 'input0' },
    outputs: { out: { tensor: 'out0', shape: RESULT_SHAPE, dtype: 'float32' } }, params: {} }],
};
function accepted(response) {
  assert.equal(response.report?.status, pb.NativeStatus.NATIVE_STATUS_OK, response.report?.message);
  return response;
}
function resultRequest(context, value) {
  return new pb.ExecuteRequest({ contextId: context.contextId, inputs: [new pb.Tensor({
    name: 'input0', shape: RESULT_SHAPE.map(BigInt), dtype: pb.DataType.DATA_TYPE_F32,
    inline: new Uint8Array(new Float32Array(12).fill(value).buffer),
  })] });
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
  compileStatus = pb.NativeStatus.NATIVE_STATUS_OK } = {}) {
  const fixture = makeResultControl(clock, limits);
  const { control } = fixture;
  const inference = new VxInferenceServiceClient(reportTransport(control));
  const scheduler = new VxSchedulerServiceClient(reportTransport(control));
  control.mountFileChunks('graph.json', [new TextEncoder().encode(JSON.stringify(graph))]);
  control.mountFileChunks('weights.safetensors', [emptySafetensors()]);
  const runtime = accepted((await inference.createRuntime(new pb.CreateRuntimeRequest({
    executionMode: scheduled ? pb.ExecutionMode.EXECUTION_MODE_SCHEDULED : pb.ExecutionMode.EXECUTION_MODE_DIRECT,
    budget,
  }))));
  const model = accepted((await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
    graphPath: 'graph.json', weightPaths: ['weights.safetensors'] }))));
  const compiled = (await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
    policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
      backends: ['webgpu'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
  assert.equal(compiled.report.status, compileStatus, compiled.report.message);
  const contexts = [];
  return { ...fixture, inference, scheduler, runtime, model, compiled,
    async context(options = {}) {
      const context = accepted((await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, ...options }))));
      contexts.push(context);
      return context;
    },
    async close() {
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
  const read = accepted((await inference.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name }))));
  const data = read.tensor.inline;
  return [...new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))];
}

test('pending results keep their identity and values across executions and context lifetimes', async () => {
  const f = (await resultFixture());
  try {
    assert.equal(f.log.length, 0, 'compile must not dispatch GPU work');
    const first = (await f.context());
    const a = accepted((await f.inference.execute(resultRequest(first, 2))));
    const second = (await f.context()); // Previously reset the first context's global GPU table.
    const b = accepted((await f.inference.execute(resultRequest(second, 9))));
    const c = accepted((await f.inference.execute(resultRequest(first, 3))));
    assert.equal(f.log.length, 3, 'each submission executes exactly once');
    assert.equal(new Set([a.executionId, b.executionId, c.executionId]).size, 3);
    for (let index = 0; index < 80; index++) {
      const pending = accepted((await f.inference.getResult(new pb.ResultRef(a))));
      assert.equal(pending.state, pb.ResultState.RESULT_STATE_PENDING);
      assert.equal(pending.executionId, a.executionId);
      assert.deepEqual(pending.outputs, []);
    }
    assert.equal((await f.inference.readOutput(new pb.ReadOutputRequest({ resultId: a.resultId, name: 'out0' }))).report.status,
      pb.NativeStatus.NATIVE_STATUS_BUSY);
    (await f.inference.releaseExecutionContext(new pb.ExecutionContextRef(first)));
    (await f.inference.releaseExecutionContext(new pb.ExecutionContextRef(second)));
    f.bridge.complete();
    for (const [result, value] of [[a, 2], [b, 9], [c, 3]]) {
      assert.equal(accepted((await f.inference.getResult(new pb.ResultRef(result)))).state, pb.ResultState.RESULT_STATE_READY);
      assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(value));
      (await f.inference.releaseResult(new pb.ResultRef(result)));
    }
    assert.equal(f.bridge.jobs.size, 0);
  } finally { (await f.close()); }
});

test('releasing pending results reclaims tickets and result budget without touching a later result', async () => {
  const f = (await resultFixture());
  try {
    const context = (await f.context());
    for (let index = 0; index < 80; index++) {
      const result = accepted((await f.inference.execute(resultRequest(context, index))));
      (await f.inference.releaseResult(new pb.ResultRef(result)));
      assert.equal(f.bridge.jobs.size, 0);
    }
    const survivor = accepted((await f.inference.execute(resultRequest(context, 81))));
    f.bridge.complete();
    assert.deepEqual((await resultValues(f.inference, survivor)), Array(12).fill(81));
    (await f.inference.releaseResult(new pb.ResultRef(survivor)));
  } finally { (await f.close()); }
});

test('failed device completion is terminal and never exposes output bytes', async () => {
  const f = (await resultFixture());
  try {
    const result = accepted((await f.inference.execute(resultRequest((await f.context()), 7))));
    f.bridge.complete(true);
    for (let index = 0; index < 2; index++) {
      const failed = (await f.inference.getResult(new pb.ResultRef(result)));
      assert.equal(failed.state, pb.ResultState.RESULT_STATE_FAILED);
      assert.equal(failed.report.status, pb.NativeStatus.NATIVE_STATUS_EXECUTION_FAILED);
      assert.equal(failed.executionId, result.executionId);
      assert.deepEqual(failed.outputs, []);
    }
    assert.equal((await f.inference.readOutput(new pb.ReadOutputRequest({ resultId: result.resultId, name: 'out0' }))).tensor, undefined);
    (await f.inference.releaseResult(new pb.ResultRef(result)));
    assert.equal(f.bridge.jobs.size, 0);
  } finally { (await f.close()); }
});

test('scheduler remains running until snapshot completion, and closes pending requests', async () => {
  const f = (await resultFixture({ scheduled: true }));
  try {
    const submit = async (value) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
      compiledModelId: f.compiled.compiledModelId, inputs: resultRequest({ contextId: 0n }, value).inputs,
    }))));
    const first = (await submit(4));
    const waiting = await startRequest(f, first);
    assert.equal(waiting.state, pb.RequestState.REQUEST_STATE_RUNNING);
    assert.equal(waiting.status, pb.NativeStatus.NATIVE_STATUS_BUSY);
    f.bridge.complete();
    assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(first))).state, pb.RequestState.REQUEST_STATE_SUCCEEDED);
    const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(first))));
    assert.equal(result.state, pb.ResultState.RESULT_STATE_READY);
    assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(4));
    (await f.inference.releaseResult(new pb.ResultRef(result)));
    (await f.scheduler.releaseRequest(new pb.RequestRef(first)));
    const second = (await submit(5));
    await startRequest(f, second);
    (await f.scheduler.cancelRequest(new pb.RequestRef(second)));
    assert.equal((await f.scheduler.pollRequest(new pb.RequestRef(second))).state, pb.RequestState.REQUEST_STATE_CANCELLED);
    assert.equal(f.bridge.jobs.size, 0);
    (await f.scheduler.releaseRequest(new pb.RequestRef(second)));
    const third = (await submit(6));
    await startRequest(f, third);
    (await f.inference.releaseRuntime(new pb.RuntimeRef(f.runtime)));
    (await f.scheduler.releaseRequest(new pb.RequestRef(third)));
    assert.equal(f.bridge.jobs.size, 0);
  } finally { (await f.close()); }
});

test('full host shares C results and releases every readback ticket', async () => {
  const route = { bridge: createSimulatingBridge([]) };
  const fetch = async (source) => String(source) === 'graph.json'
    ? { ok: true, text: async () => JSON.stringify(RESULT_GRAPH) }
    : { ok: true, arrayBuffer: async () => Uint8Array.from(emptySafetensors()).buffer };
  const host = new FullEngineHost({ wasmUrl: FULL_WASM, fetch, gpuBridge: route.bridge });
  const inference = new VxInferenceServiceClient(reportTransport(host));
  let runtime, model, compiled, context;
  try {
    runtime = accepted(await inference.createRuntime(new pb.CreateRuntimeRequest()));
    model = accepted(await inference.loadModel(new pb.LoadModelRequest({ runtimeId: runtime.runtimeId,
      graphPath: 'graph.json', weightPaths: ['weights.safetensors'] })));
    compiled = accepted(await inference.compileModel(new pb.CompileModelRequest({ modelId: model.modelId,
      policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['webgpu'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }) })));
    context = accepted(await inference.createExecutionContext(new pb.CreateExecutionContextRequest(compiled)));
    for (let index = 0; index < 70; index++) {
      const submitted = accepted(await inference.execute(resultRequest(context, index)));
      await route.bridge.waitForCompletion();
      const result = accepted(await inference.getResult(new pb.ResultRef(submitted)));
      assert.equal(result.state, pb.ResultState.RESULT_STATE_READY);
      await inference.releaseResult(new pb.ResultRef(result));
      assert.equal(route.bridge.jobs.size, 0);
    }
    const decode = await inference.decodePrefill(new pb.DecodePrefillRequest(resultRequest(context, 1)));
    assert.equal(decode.report.status, pb.NativeStatus.NATIVE_STATUS_BACKEND_UNSUPPORTED,
      'disabled decode is rejected by the C context contract');
    const enabled = accepted(await inference.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, decodeRowMode: pb.DecodeRowMode.DECODE_ROW_MODE_AUTO,
    })));
    try {
      const prefill = accepted(await inference.decodePrefill(new pb.DecodePrefillRequest(resultRequest(enabled, 1))));
      await route.bridge.waitForCompletion();
      assert.equal((await inference.getResult(new pb.ResultRef(prefill))).state, pb.ResultState.RESULT_STATE_READY);
      await inference.releaseResult(new pb.ResultRef(prefill));
      const step = accepted(await inference.decodeStep(new pb.DecodeStepRequest({
        ...resultRequest(enabled, 2), position: 1,
      })));
      await route.bridge.waitForCompletion();
      assert.equal((await inference.getResult(new pb.ResultRef(step))).state, pb.ResultState.RESULT_STATE_READY);
      await inference.releaseResult(new pb.ResultRef(step));
      assert.equal((await inference.resetDecode(new pb.ExecutionContextRef(enabled))).status, pb.NativeStatus.NATIVE_STATUS_OK);
    } finally { await inference.releaseExecutionContext(new pb.ExecutionContextRef(enabled)); }

  } finally {
    if (context) await inference.releaseExecutionContext(new pb.ExecutionContextRef(context));
    if (compiled) await inference.releaseCompiledModel(new pb.CompiledModelRef(compiled));
    if (model) await inference.releaseModel(new pb.ModelRef(model));
    if (runtime) await inference.releaseRuntime(new pb.RuntimeRef(runtime));
    await host.close();
  }
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
    context = accepted((await inference.createExecutionContext(new pb.CreateExecutionContextRequest(compiled))));
    const result = accepted((await inference.execute(resultRequest(context, 2))));
    assert.equal(result.state, pb.ResultState.RESULT_STATE_READY);
    for (const value of (await resultValues(inference, result))) assert.ok(Math.abs(value - 2) < 1e-6);
    assert.equal(f.log.length, 0, 'the entire model executes in portable C');
    (await inference.releaseResult(new pb.ResultRef(result)));
  } finally {
    if (context) (await inference.releaseExecutionContext(new pb.ExecutionContextRef(context)));
    if (compiled) (await inference.releaseCompiledModel(new pb.CompiledModelRef(compiled)));
    (await inference.releaseModel(new pb.ModelRef(model)));
    (await inference.releaseRuntime(new pb.RuntimeRef(runtime)));
    (await f.control.close());
  }
});

test('multi-output completion stays atomic while a later execution reuses the context', async () => {
  const graph = structuredClone(RESULT_GRAPH);
  graph.outputs.push('out1');
  graph.nodes.push({ id: 'again', opType: 'ReLU', inputs: { input: 'out0' },
    outputs: { out: { tensor: 'out1', dtype: 'float32', shape: RESULT_SHAPE } }, params: {} });
  const f = (await resultFixture({ graph }));
  try {
    const context = (await f.context());
    const first = accepted((await f.inference.execute(resultRequest(context, 2))));
    assert.equal(f.bridge.jobs.size, 2);
    f.bridge.jobs.values().next().value.ready = true;
    const partial = accepted((await f.inference.getResult(new pb.ResultRef(first))));
    assert.equal(partial.state, pb.ResultState.RESULT_STATE_PENDING);
    assert.deepEqual(partial.outputs, []);
    const second = accepted((await f.inference.execute(resultRequest(context, 5))));
    f.bridge.complete();
    for (const [result, expected] of [[first, 2], [second, 5]]) {
      const ready = accepted((await f.inference.getResult(new pb.ResultRef(result))));
      assert.equal(ready.outputs.length, 2);
      for (const name of ['out0', 'out1']) assert.deepEqual((await resultValues(f.inference, result, name)), Array(12).fill(expected));
      (await f.inference.releaseResult(new pb.ResultRef(result)));
    }
    assert.equal(f.bridge.jobs.size, 0);
  } finally { (await f.close()); }
});

test('WASM platform time and scheduled admission share the host monotonic clock', async () => {
  const clock = { micros: 1_234_567 };
  const f = (await resultFixture({ scheduled: true, clock }));
  try {
    const platform = new VxPlatformServiceClient(reportTransport(f.control));
    assert.equal((await platform.getMonotonicTime(new pb.Empty())).microseconds, 1_234_567n);
    const sample = (await platform.sampleProcessMemory(new pb.Empty()));
    assert.equal(sample.hasMonotonicTime, true);
    assert.equal(sample.monotonicNanoseconds, 1_234_567_000n);
    const expired = (await f.scheduler.submit(new pb.SubmitRequest({
      compiledModelId: f.compiled.compiledModelId,
      inputs: resultRequest({ contextId: 0n }, 1).inputs,
      options: new pb.SubmitOptions({ deadlineMonotonicMicros: 1_234_567n }),
    })));
    assert.equal(expired.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
    assert.equal(expired.requestId, 0n);
    assert.equal(f.log.length, 0);
    clock.micros += 543;
    assert.equal((await platform.getMonotonicTime(new pb.Empty())).microseconds, 1_235_110n);
  } finally { (await f.close()); }
});

test('a missing host clock refuses scheduled execution without breaking direct execution', async () => {
  const { control } = makeResultControl({ micros: -1 });
  try {
    const inference = new VxInferenceServiceClient(reportTransport(control));
    const platform = new VxPlatformServiceClient(reportTransport(control));
    assert.equal((await platform.sampleProcessMemory(new pb.Empty())).hasMonotonicTime, false);
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
    maxScheduledRequests: 1n, maxScheduledInputBytes: 4096n,
    maxUnconsumedResults: 1n, maxUnconsumedResultBytes: 48n,
  }) }));
  try {
    for (let index = 0; index < 24; index++) {
      const request = accepted((await f.scheduler.submit(new pb.SubmitRequest({
        compiledModelId: f.compiled.compiledModelId,
        inputs: resultRequest({ contextId: 0n }, index).inputs,
        options: new pb.SubmitOptions({
          deadlineMonotonicMicros: BigInt(clock.micros + 500),
          freshness: pb.RequestFreshness.REQUEST_FRESHNESS_DROP_IF_LATE,
        }),
      }))));
      const pending = index % 2 === 1;
      if (pending) {
        await startRequest(f, request);
        assert.equal(f.bridge.jobs.size, 1);
      }
      const dispatches = f.log.length;
      clock.micros += 500;
      const terminal = (await f.scheduler.pollRequest(new pb.RequestRef(request)));
      assert.equal(terminal.state, pb.RequestState.REQUEST_STATE_FAILED);
      assert.equal(terminal.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
      assert.equal(terminal.deadlineMissed, true);
      assert.equal(terminal.ownedInputBytes, 0n);
      assert.equal(f.log.length, dispatches, 'poll must never dispatch queued work');
      assert.equal(f.bridge.jobs.size, 0);
      const result = (await f.scheduler.takeRequestResult(new pb.RequestRef(request)));
      assert.equal(result.resultId, 0n);
      assert.equal(result.report.status, pb.NativeStatus.NATIVE_STATUS_DEADLINE_EXCEEDED);
      (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
    }
  } finally { (await f.close()); }
});

test('soft deadlines keep pending results and report a missed target at completion', async () => {
  const clock = { micros: 1_000_000 };
  const f = (await resultFixture({ scheduled: true, clock }));
  try {
    const request = accepted((await f.scheduler.submit(new pb.SubmitRequest({
      compiledModelId: f.compiled.compiledModelId,
      inputs: resultRequest({ contextId: 0n }, 7).inputs,
      options: new pb.SubmitOptions({ deadlineMonotonicMicros: 1_001_000n }),
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
    (await f.inference.releaseResult(new pb.ResultRef(result)));
    (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
  } finally { (await f.close()); }
});

test('LATEST supersedes pending GPU copies and queued requests without retaining their budgets', async () => {
  const f = (await resultFixture({ scheduled: true, budget: new pb.RuntimeBudget({
    maxScheduledRequests: 2n, maxScheduledInputBytes: 8192n,
    maxUnconsumedResults: 2n, maxUnconsumedResultBytes: 96n,
  }) }));
  const submit = async (value) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
    compiledModelId: f.compiled.compiledModelId,
    inputs: resultRequest({ contextId: 0n }, value).inputs,
    options: new pb.SubmitOptions({ freshness: pb.RequestFreshness.REQUEST_FRESHNESS_LATEST, streamKey: 9n }),
  }))));
  try {
    for (let index = 0; index < 24; index++) {
      const old = (await submit(1));
      await startRequest(f, old);
      const queued = (await submit(2));
      const latest = (await submit(3));
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
      (await f.inference.releaseResult(new pb.ResultRef(result)));
      (await f.scheduler.releaseRequest(new pb.RequestRef(latest)));
    }
  } finally { (await f.close()); }
});

test('C scheduling uses host-clock aging before EDF and request order', async () => {
  const clock = { micros: 1_000_000 };
  const f = (await resultFixture({ scheduled: true, clock }));
  const submit = async (value, options = {}) => accepted((await f.scheduler.submit(new pb.SubmitRequest({
    compiledModelId: f.compiled.compiledModelId,
    inputs: resultRequest({ contextId: 0n }, value).inputs,
    options: new pb.SubmitOptions(options),
  }))));
  try {
    const old = (await submit(1));
    clock.micros += 20_000;
    const laterDeadline = (await submit(2, { priority: 1, deadlineMonotonicMicros: 1_060_000n }));
    const earlierDeadline = (await submit(3, { priority: 1, deadlineMonotonicMicros: 1_050_000n }));
    await startRequest(f, laterDeadline);
    const values = [...f.bridge.jobs.values()].map(({ bytes }) => new DataView(
      bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(0, true));
    assert.deepEqual(values, [1, 3, 2]);
    f.bridge.complete();
    for (const request of [old, earlierDeadline, laterDeadline]) {
      const result = accepted((await f.scheduler.takeRequestResult(new pb.RequestRef(request))));
      (await f.inference.releaseResult(new pb.ResultRef(result)));
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
    const result = accepted((await f.inference.execute(resultRequest((await f.context()), 4))));
    f.bridge.complete();
    assert.deepEqual((await resultValues(f.inference, result)), Array(12).fill(4));
    (await f.inference.releaseResult(new pb.ResultRef(result)));
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
      const result = accepted((await f.inference.execute(new pb.ExecuteRequest({
        contextId: context.contextId, inputs: [shapedTensor(shape, 3)] }))));
      f.bridge.complete();
      assert.equal(accepted((await f.inference.getResult(new pb.ResultRef(result)))).state, pb.ResultState.RESULT_STATE_READY);
      assert.deepEqual((await resultValues(f.inference, result)), Array(shape.reduce((a,b) => a*b,1)).fill(3));
      (await f.inference.releaseResult(new pb.ResultRef(result)));
    }
    const refused = (await f.inference.execute(new pb.ExecuteRequest({ contextId: context.contextId,
      inputs: [shapedTensor([1, 3, 3], 1)] })));
    assert.equal(refused.report.status, pb.NativeStatus.NATIVE_STATUS_INVALID_ARGUMENT);
  } finally { (await f.close()); }
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
        compiledModelId: f.compiled.compiledModelId, inputs: [shapedTensor([1, 3], lane + 1)] })))))));
      const before = f.log.length;
      assert.equal((await startRequest(f, requests[0])).state, pb.RequestState.REQUEST_STATE_RUNNING);
      assert.equal(f.log.length - before, 1, `all lanes must share one physical dispatch: ${count} lanes; ${JSON.stringify(f.log.slice(before).map(p => p.params))}`);
      assert.equal(f.bridge.jobs.size, 1, 'all lanes must share one staging snapshot');
      if (count === 3) {
        (await f.scheduler.cancelRequest(new pb.RequestRef(requests[1])));
        assert.equal(f.bridge.jobs.size, 1, 'cancelling one lane must preserve the others');
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
          (await f.inference.releaseResult(new pb.ResultRef(result)));
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
        compiledModelId:f.compiled.compiledModelId,inputs:[shapedTensor([1,3],i)]})))))));
      await startRequest(f, requests[0]);
      assert.equal(f.bridge.jobs.size,1);
      f.bridge.complete(true);
      for (const request of requests) {
        const state=(await f.scheduler.pollRequest(new pb.RequestRef(request)));
        assert.equal(state.state,pb.RequestState.REQUEST_STATE_FAILED,state.report.message);
        assert.equal((await f.scheduler.takeRequestResult(new pb.RequestRef(request))).resultId,0n);
        (await f.scheduler.releaseRequest(new pb.RequestRef(request)));
      }
      assert.equal(f.bridge.jobs.size,0);
    }
  } finally {(await f.close());}
});

test('required GPU rows preserve unaligned prefixes, prior snapshots and reset state', async () => {
  const shape=[1,4,5], f=(await resultFixture({graph:boundedReluGraph({},shape)}));
  const input=values=>new pb.Tensor({name:'input0',shape:shape.map(BigInt),
    dtype:pb.DataType.DATA_TYPE_F32,inline:new Uint8Array(Float32Array.from(values).buffer)});
  try {
    const context=(await f.context({decodeRowMode:pb.DecodeRowMode.DECODE_ROW_MODE_REQUIRED,requireIncremental:true}));
    const values=Array.from({length:20},(_,i)=>i%2?i:-i), expected=values.map(v=>Math.max(0,v));
    const initial=accepted((await f.inference.decodePrefill(new pb.DecodePrefillRequest({
      contextId:context.contextId,inputs:[input(values)]}))));
    f.bridge.complete(); accepted((await f.inference.getResult(new pb.ResultRef(initial))));
    for (const position of [1,2]) {
      values.fill(40+position,position*5,(position+1)*5);
      expected.fill(40+position,position*5,(position+1)*5);
      const before=f.log.length;
      const result=accepted((await f.inference.decodeStep(new pb.DecodeStepRequest({
        contextId:context.contextId,position,inputs:[input(values)]}))));
      assert.equal(f.log.length-before,3);
      f.bridge.complete(); accepted((await f.inference.getResult(new pb.ResultRef(result))));
      assert.deepEqual((await resultValues(f.inference,result)),expected);
      (await f.inference.releaseResult(new pb.ResultRef(result)));
    }
    assert.deepEqual((await resultValues(f.inference,initial)),Array.from({length:20},(_,i)=>i%2?i:0));
    assert.equal((await f.inference.resetDecode(new pb.ExecutionContextRef(context))).status,0);
    const reset=accepted((await f.inference.decodePrefill(new pb.DecodePrefillRequest({
      contextId:context.contextId,inputs:[input(Array(20).fill(7))]}))));
    f.bridge.complete(); accepted((await f.inference.getResult(new pb.ResultRef(reset))));
    assert.deepEqual((await resultValues(f.inference,reset)),Array(20).fill(7));
    (await f.inference.releaseResult(new pb.ResultRef(reset))); (await f.inference.releaseResult(new pb.ResultRef(initial)));
  } finally {(await f.close());}
});
