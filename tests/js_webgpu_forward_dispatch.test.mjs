import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import { GraphExecutor as RuntimeGraphExecutor } from '../ts/backends/GraphExecutor.js';
import { compileWebGPUGraphPlan } from '../ts/backends/WebGPUGraphCompiler.js';
import { Graph } from '../ts/core/Graph.js';
import { DataType } from '../ts/generated/volvoxaiEnums.js';

// Node does not expose these WebGPU constants. The mock only needs distinct
// bit values while it records the browser-side dispatch contract.
globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});
globalThis.GPUMapMode ??= Object.freeze({ READ: 1 });

function tensor(name, shape, { dtype = 'float32', buffer = null } = {}) {
  const elementBytes = dtype === 'int8' || dtype === 'uint8' ? 1 : 4;
  return {
    name,
    shape,
    dtype,
    buffer,
    sizeBytes: shape.reduce((product, dimension) => product * dimension, 1) * elementBytes,
  };
}

class GraphExecutor extends RuntimeGraphExecutor {
  async _buildNodePipeline(node) {
    for (const value of [...Object.values(node.inputs || {}), ...Object.values(node.outputs || {})]) {
      if (value && !this.gpuBuffers.has(value.name)) {
        this.gpuBuffers.set(value.name, { tensor: value.name });
      }
    }
    return super._buildNodePipeline(node);
  }
}

function quantizedTensor(name, shape, { dtype = 'int8', scale = 0.25, zeroPoint = 0, buffer = null } = {}) {
  const value = tensor(name, shape, { dtype, buffer });
  value.quantization = { scheme: 'per_tensor', scale, zero_point: zeroPoint };
  return value;
}

function mockDevice() {
  const state = {
    bindGroups: [], shaderCodes: [], buffers: [], copies: [], dispatches: [], writes: [],
    submissions: [], qsdpaSeqKV: [], mapCount: 0,
  };
  return {
    state,
    createBuffer(descriptor) {
      const buffer = {
        descriptor,
        bytes: new Uint8Array(descriptor.size),
        async mapAsync() { state.mapCount++; },
        getMappedRange() { return this.bytes.buffer.slice(0); },
        unmap() { this.unmapped = true; },
        destroy() { this.destroyed = true; },
      };
      state.buffers.push(buffer);
      return buffer;
    },
    createShaderModule({ code }) {
      state.shaderCodes.push(code);
      return { code };
    },
    async createComputePipelineAsync({ compute } = {}) {
      return { code: compute?.module?.code, getBindGroupLayout() { return {}; } };
    },
    createBindGroup(descriptor) {
      state.bindGroups.push(descriptor);
      return descriptor;
    },
    createCommandEncoder() {
      const copies = [];
      const dispatches = [];
      const operations = [];
      return {
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          const operation = { source, sourceOffset, destination, destinationOffset, size };
          copies.push(operation);
          operations.push({ type: 'copy', value: operation });
        },
        beginComputePass() {
          let pipeline = null;
          let bindGroup = null;
          return {
            setPipeline(value) { pipeline = value; },
            setBindGroup(index, value) { bindGroup = { index, value }; },
            dispatchWorkgroups(x, y, z) {
              const operation = { pipeline, bindGroup, x, y, z };
              dispatches.push(operation);
              operations.push({ type: 'dispatch', value: operation });
            },
            end() {},
          };
        },
        finish() { return { copies, dispatches, operations }; },
      };
    },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        const sourceBuffer = source instanceof ArrayBuffer ? source : source.buffer;
        const byteOffset = (source instanceof ArrayBuffer ? 0 : source.byteOffset) + sourceOffset;
        const byteLength = size ?? (source instanceof ArrayBuffer ? source.byteLength : source.byteLength) - sourceOffset;
        destination.bytes.set(new Uint8Array(sourceBuffer, byteOffset, byteLength), offset);
        state.writes.push({ destination, offset, byteOffset, byteLength });
      },
      submit(commandBuffers) {
        state.submissions.push(commandBuffers);
        for (const commandBuffer of commandBuffers) {
          for (const operation of commandBuffer.operations || []) {
            if (operation.type === 'copy') {
              const copy = operation.value;
              copy.destination.bytes.set(
                copy.source.bytes.subarray(copy.sourceOffset, copy.sourceOffset + copy.size),
                copy.destinationOffset,
              );
              state.copies.push(copy);
              continue;
            }
            const dispatch = operation.value;
            const entries = new Map((dispatch.bindGroup?.value?.entries || []).map((entry) =>
              [entry.binding, entry.resource.buffer]));
            if (dispatch.pipeline?.code === 'feedback-embedding') {
              const ids = new Int32Array(entries.get(0).bytes.buffer);
              const output = entries.get(4).bytes;
              const params = new Uint32Array(entries.get(5).bytes.buffer);
              const [rows, vocabulary, hidden] = params;
              output.fill(0);
              for (let row = 0; row < rows; row++) {
                output[row * hidden + ((ids[row] + 1) % vocabulary)] = 100;
              }
            } else if (dispatch.pipeline?.code === 'feedback-qsdpa') {
              const q = entries.get(0).bytes;
              const output = entries.get(4).bytes;
              const params = new Uint32Array(entries.get(5).bytes.buffer);
              output.set(q.subarray(0, params[0] * params[2]));
              state.qsdpaSeqKV.push(params[1]);
            } else if (dispatch.pipeline?.code === 'feedback-qargmax') {
              const input = new Int8Array(entries.get(0).bytes.buffer);
              const output = new Int32Array(entries.get(1).bytes.buffer);
              const params = new Uint32Array(entries.get(2).bytes.buffer);
              const [outer, axisSize, inner] = params;
              for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
                for (let innerIndex = 0; innerIndex < inner; innerIndex++) {
                  let best = 0;
                  let bestValue = -129;
                  for (let axis = 0; axis < axisSize; axis++) {
                    const value = input[(outerIndex * axisSize + axis) * inner + innerIndex];
                    if (value > bestValue) { best = axis; bestValue = value; }
                  }
                  output[outerIndex * inner + innerIndex] = best;
                }
              }
            } else if (dispatch.pipeline?.code === 'incremental-byte-copy') {
              const source = entries.get(0).bytes;
              const destination = entries.get(1).bytes;
              const params = new Uint32Array(entries.get(2).bytes.buffer);
              const [sourceOffset, destinationOffset, size] = params;
              destination.set(source.subarray(sourceOffset, sourceOffset + size), destinationOffset);
            } else if (dispatch.pipeline?.code === 'unaligned-qlinear') {
              const input = entries.get(0).bytes;
              const output = entries.get(5).bytes;
              const params = new Uint32Array(entries.get(6).bytes.buffer);
              const [rows, dIn, dOut] = params;
              for (let row = 0; row < rows; row++) {
                for (let column = 0; column < dOut; column++) {
                  output[row * dOut + column] = input[row * dIn + (column % dIn)] + column;
                }
              }
            } else if (dispatch.pipeline?.code?.includes(
              'struct Params { b : u32, d : u32 }',
            )) {
              const source = dispatch.pipeline.code;
              const input = new Float32Array(entries.get(0).bytes.buffer);
              const output = new Float32Array(entries.get(1).bytes.buffer);
              const [rows, width] = new Uint32Array(entries.get(2).bytes.buffer);
              const rowSeededMaximum = source.includes('var max_val = input[offset];');
              const logOutput = source.includes('let logSum = log(sum);');
              for (let row = 0; row < rows; row++) {
                const offset = row * width;
                let maximum = rowSeededMaximum ? input[offset] : -100000;
                for (let column = rowSeededMaximum ? 1 : 0; column < width; column++) {
                  maximum = Math.max(maximum, input[offset + column]);
                }
                let sum = 0;
                for (let column = 0; column < width; column++) {
                  sum += Math.exp(input[offset + column] - maximum);
                }
                const logSum = Math.log(sum);
                for (let column = 0; column < width; column++) {
                  output[offset + column] = logOutput
                    ? input[offset + column] - maximum - logSum
                    : Math.exp(input[offset + column] - maximum) / sum;
                }
              }
            }
            state.dispatches.push(dispatch);
          }
        }
      },
    },
  };
}

function paramsFrom(device, binding) {
  return device.state.bindGroups.at(-1).entries.find((entry) => entry.binding === binding).resource.buffer;
}

test('WebGPU PReLU accepts the canonical slope input and binds channel parameters', async () => {
  const device = mockDevice();
  const input = tensor('input', [4, 8]);
  const slope = tensor('slope', [8]);
  const out = tensor('out', [4, 8]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getPReLUShader: () => 'prelu' },
  });

  await executor._buildNodePipeline({
    id: 'prelu_slope', opType: 'PReLU', inputs: { input, slope }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['prelu']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.slice(0, 3).map(({ resource }) => resource.buffer.tensor), [
    'input', 'slope', 'out',
  ]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 3).bytes.buffer)], [32, 8, 8, 0]);
  executor.dispose();
});

test('WebGPU typed readback copies padded storage and returns the logical requested dtype', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] });
  const bytes = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  bytes.bytes.set(Uint8Array.of(255, 2, 127, 91));

  const signed = await executor.readBuffer(bytes, 3, 'int8');
  assert.ok(signed instanceof Int8Array);
  assert.deepEqual([...signed], [-1, 2, 127]);
  const unsigned = await executor.readBuffer(bytes, 3, 'uint8');
  assert.ok(unsigned instanceof Uint8Array);
  assert.deepEqual([...unsigned], [255, 2, 127]);
  const middle = await executor.readBufferRange(bytes, 1, 2, 'uint8');
  assert.deepEqual([...middle], [2, 127]);
  assert.deepEqual(device.state.copies.map((copy) => [copy.sourceOffset, copy.size]), [
    [0, 4], [0, 4], [0, 4],
  ]);
  await assert.rejects(
    executor.readBufferRange(bytes, 1, 4, 'int32'),
    /int32-aligned offset/,
  );

  const floats = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  new Float32Array(floats.bytes.buffer)[0] = 1.5;
  const f32 = await executor.readBuffer(floats, 4);
  assert.ok(f32 instanceof Float32Array);
  assert.deepEqual([...f32], [1.5]);
  executor.dispose();
});

test('WebGPU W8A8 incremental rows retain device K/V prefixes and update only the selected row', async () => {
  const graph = new Graph();
  const quantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
  const q = graph.addInput('q', [1, 3, 4], 'int8', { quantization });
  const k = graph.addInput('k', [1, 3, 4], 'int8', { quantization });
  const v = graph.addInput('v', [1, 3, 4], 'int8', { quantization });
  const keep = graph.addInput('keep', [1, 3], 'int32');
  const attended = graph.addOp('QSDPA', { q, k, v, mask: keep }, {
    out: { name: 'attended', shape: [1, 3, 4], dtype: 'int8', quantization },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  graph.addOp('QArgMax', { input: attended }, {
    out: { name: 'tokens', shape: [1, 3], dtype: 'int32' },
  }, { axis: -1 });

  const device = mockDevice();
  const executor = new RuntimeGraphExecutor(device, graph, {
    shaderLibrary: {
      getQSDPAShader: () => 'typed-qsdpa',
      getQArgMaxShader: () => 'typed-qargmax',
    },
  });
  await executor.compile();
  assert.equal(executor.capabilities.incrementalRows, true);
  assert.deepEqual([...executor.incrementalRowCandidates.keys()], [0, 1]);
  assert.equal(executor.incrementalRowPlans.size, 0,
    'row bind groups are compiled lazily for the selected decode closure');

  const inputs = {
    q: new Int8Array(12),
    k: new Int8Array(12),
    v: new Int8Array(12),
    keep: Int32Array.of(1, 1, 1),
  };
  await executor.execute(inputs, {
    incremental: true,
    incrementalReset: true,
    changedInputs: Object.keys(inputs),
  });
  device.state.copies.length = 0;
  device.state.dispatches.length = 0;
  device.state.writes.length = 0;
  device.state.submissions.length = 0;

  inputs.q[8] = 1;
  inputs.k[8] = 2;
  inputs.v[8] = 3;
  await executor.execute(inputs, {
    incremental: true,
    changedInputs: Object.keys(inputs),
    incrementalRowPosition: 2,
  });

  const qsdpaPlan = executor.incrementalRowPlans.get(0);
  assert.deepEqual([...executor.incrementalRowPlans.keys()], [0, 1]);
  assert.deepEqual([...qsdpaPlan.scratchInputs], ['q']);
  assert.deepEqual([...qsdpaPlan.scratchOutputs], ['attended']);
  const inputBuffers = new Set(['q', 'k', 'v', 'keep'].map((name) =>
    executor.gpuBuffers.get(name)));
  assert.deepEqual(device.state.writes.filter(({ destination }) =>
    inputBuffers.has(destination)).map(({ offset, byteLength }) =>
    [offset, byteLength]), [[8, 4], [8, 4], [8, 4], [8, 4]]);
  assert.equal(new Uint32Array(qsdpaPlan.qsdpaParamsBuffer.bytes.buffer)[1], 3,
    'causal QSDPA must expose exactly K/V rows 0..position');
  assert.deepEqual(device.state.copies.map(({ sourceOffset, destinationOffset, size }) =>
    [sourceOffset, destinationOffset, size]), [
    [8, 0, 4],
    [0, 8, 4],
    [8, 0, 4],
    [0, 8, 4],
  ]);
  assert.equal(device.state.dispatches.length, 2);
  assert.equal(device.state.submissions.length, 1,
    'all row copies and kernels must share one GPU submission');
  executor.dispose();
});

test('WebGPU row candidates reject a dirty noncausal QSDPA mask before upload', async () => {
  const graph = new Graph();
  const quantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
  const q = graph.addInput('q', [1, 3, 4], 'int8', { quantization });
  const k = graph.addInput('k', [1, 2, 4], 'int8', { quantization });
  const v = graph.addInput('v', [1, 2, 4], 'int8', { quantization });
  const mask = graph.addInput('mask', [1, 2], 'int32');
  const attended = graph.addOp('QSDPA', { q, k, v, mask }, {
    out: { name: 'attended', shape: [1, 3, 4], dtype: 'int8', quantization },
  }, { heads: 1, causal: false, scale: 0.5 }).out;
  graph.addOp('QArgMax', { input: attended }, {
    out: { name: 'tokens', shape: [1, 3], dtype: 'int32' },
  }, { axis: -1 });

  const device = mockDevice();
  const executor = new RuntimeGraphExecutor(device, graph, {
    shaderLibrary: {
      getQSDPAShader: () => 'typed-cross-qsdpa',
      getQArgMaxShader: () => 'typed-qargmax',
    },
  });
  await executor.compile();
  assert.deepEqual([...executor.incrementalRowCandidates.keys()], [0, 1],
    'candidate discovery must defer changed-input invariance proof until execution');
  assert.deepEqual(
    [...executor.incrementalRowCandidates.get(0).invariantInputs],
    ['k', 'v', 'mask'],
  );

  const inputs = {
    q: new Int8Array(12),
    k: new Int8Array(8),
    v: new Int8Array(8),
    mask: Int32Array.of(1, 1),
  };
  await executor.execute(inputs, {
    incremental: true,
    incrementalReset: true,
    changedInputs: Object.keys(inputs),
  });
  device.state.writes.length = 0;
  device.state.copies.length = 0;
  device.state.dispatches.length = 0;
  device.state.submissions.length = 0;

  await assert.rejects(
    executor.execute(inputs, {
      incremental: true,
      incrementalRowPosition: 1,
      changedInputs: ['mask'],
    }),
    /input 'mask' must remain invariant/,
  );
  assert.equal(device.state.writes.length, 0,
    'dirty invariant K/V/mask must fail before any GPU upload');
  assert.equal(device.state.copies.length, 0);
  assert.equal(device.state.dispatches.length, 0);
  assert.equal(device.state.submissions.length, 0);
  executor.dispose();
});

test('WebGPU W8A8 incremental rows preserve adjacent bytes for unaligned packed rows', async () => {
  const graph = new Graph();
  const activationQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
  const input = graph.addInput('input', [1, 3, 5], 'int8', {
    quantization: activationQuantization,
  });
  const weight = graph.addWeight('weight', [7, 5], 'int8', {
    buffer: new Int8Array(35),
    quantization: {
      scheme: 'per_axis', axis: 0,
      scales: new Array(7).fill(0.125), zero_points: new Array(7).fill(0),
    },
  });
  const bias = graph.addWeight('bias', [7], 'int32', {
    buffer: new Int32Array(7),
  });
  graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name: 'logits', shape: [1, 3, 7], dtype: 'int8',
      quantization: activationQuantization,
    },
  });

  const executor = new RuntimeGraphExecutor(mockDevice(), graph, {
    shaderLibrary: {
      getQLinearShader: () => 'unaligned-qlinear',
      getIncrementalRowByteCopyShader: () => 'incremental-byte-copy',
    },
  });
  await executor.compile();
  assert.equal(executor.capabilities.incrementalRows, true);
  assert.equal(executor.incrementalRowCandidates.has(0), true);

  const values = Int8Array.of(
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    30, 31, 32, 33, 34,
  );
  await executor.execute({ input: values }, {
    incremental: true,
    incrementalReset: true,
    changedInputs: ['input'],
  });
  const logits = executor.gpuBuffers.get('logits');
  const before = logits.bytes.slice(0, 21);
  values.set([40, 41, 42, 43, 44], 5);
  const device = executor.device;
  device.state.dispatches.length = 0;
  device.state.copies.length = 0;
  device.state.writes.length = 0;
  device.state.submissions.length = 0;
  await executor.execute({ input: values }, {
    incremental: true,
    changedInputs: ['input'],
    incrementalRowPosition: 1,
  });

  assert.equal(executor.incrementalRowPlans.has(0), true);
  assert.deepEqual(device.state.writes.filter(({ destination }) =>
    destination === executor.gpuBuffers.get('input')).map(({ offset, byteLength }) =>
    [offset, byteLength]), [[4, 8]],
    'the host upload covers only the minimally aligned span around the five-byte row');
  assert.deepEqual(
    [...executor.gpuBuffers.get('input').bytes.subarray(0, values.length)], [...values],
    'the aligned upload span retains both neighboring input rows',
  );
  assert.deepEqual(device.state.copies, [], 'unaligned device ranges use shader copies');
  assert.deepEqual(device.state.dispatches.map(({ pipeline }) => pipeline.code), [
    'incremental-byte-copy', 'unaligned-qlinear', 'incremental-byte-copy',
  ]);
  assert.equal(device.state.submissions.length, 1);
  assert.deepEqual([...logits.bytes.subarray(0, 7)], [...before.subarray(0, 7)],
    'scatter preserves the preceding packed row');
  assert.deepEqual([...logits.bytes.subarray(7, 14)], [40, 42, 44, 46, 48, 45, 47]);
  assert.deepEqual([...logits.bytes.subarray(14, 21)], [...before.subarray(14, 21)],
    'scatter preserves the following packed row');
  executor.dispose();
});

test('WebGPU device-feedback chunks resume QArgMax IDs and keep rows without per-token maps', async () => {
  const graph = new Graph();
  const sequence = 4;
  const vocabulary = 4;
  const activationQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
  const weightQuantization = {
    scheme: 'per_axis', axis: 0,
    scales: new Array(vocabulary).fill(0.125),
    zero_points: new Array(vocabulary).fill(0),
  };
  const ids = graph.addInput('ids', [1, sequence], 'int32');
  const keep = graph.addInput('keep', [1, sequence], 'int32');
  const embedding = (name) => {
    const weight = graph.addWeight(`${name}_weight`, [vocabulary, vocabulary], 'int8', {
      buffer: new Int8Array(vocabulary * vocabulary), quantization: weightQuantization,
    });
    return graph.addOp('QEmbedding', { input: ids, weight }, {
      out: {
        name, shape: [1, sequence, vocabulary], dtype: 'int8',
        quantization: activationQuantization,
      },
    }).out;
  };
  const q = embedding('q');
  const k = embedding('k');
  const v = embedding('v');
  const attended = graph.addOp('QSDPA', { q, k, v, mask: keep }, {
    out: {
      name: 'attended', shape: [1, sequence, vocabulary], dtype: 'int8',
      quantization: activationQuantization,
    },
  }, { heads: 1, causal: true, scale: 0.5 }).out;
  graph.addOp('QArgMax', { input: attended }, {
    out: { name: 'next_ids', shape: [1, sequence], dtype: 'int32' },
  }, { axis: -1 });

  const device = mockDevice();
  const executor = new RuntimeGraphExecutor(device, graph, {
    shaderLibrary: {
      getQEmbeddingShader: () => 'feedback-embedding',
      getQSDPAShader: () => 'feedback-qsdpa',
      getQArgMaxShader: () => 'feedback-qargmax',
    },
  });
  await executor.compile();
  await assert.rejects(
    executor.executeDeviceFeedbackDecode({
      ids: Int32Array.of(0, 0, 0, 0),
      keep: Int32Array.of(1, 0, 0, 0),
    }, {
      tokenInput: 'ids', keepInput: 'keep', output: 'next_ids',
      endPosition: 2, rowsPerSubmission: 2,
    }),
    /requires rowsPerSubmission = 1/,
  );
  const outputBuffer = await executor.executeDeviceFeedbackDecode({
    ids: Int32Array.of(0, 0, 0, 0),
    keep: Int32Array.of(1, 0, 0, 0),
  }, {
    tokenInput: 'ids',
    keepInput: 'keep',
    output: 'next_ids',
    endPosition: 2,
  });

  assert.equal(device.state.mapCount, 0, 'seed and every feedback row stay unmapped');
  assert.equal(device.state.submissions.length, 2,
    'one seed submission is followed by one ordered first-chunk submission');
  const firstChunk = await executor.readBufferRange(
    outputBuffer, 0, 2 * Int32Array.BYTES_PER_ELEMENT, 'int32',
  );
  assert.deepEqual([...firstChunk], [1, 2]);
  assert.equal(device.state.mapCount, 1, 'the first chunk has one map, not one map per token');

  assert.equal(await executor.executeDeviceFeedbackDecode(null, {
    tokenInput: 'ids',
    keepInput: 'keep',
    output: 'next_ids',
    startPosition: 2,
    endPosition: sequence,
  }), outputBuffer);
  assert.deepEqual([...new Int32Array(executor.gpuBuffers.get('ids').bytes.buffer)], [0, 1, 2, 3]);
  assert.deepEqual([...new Int32Array(executor.gpuBuffers.get('keep').bytes.buffer)], [1, 1, 1, 1]);
  assert.deepEqual(device.state.qsdpaSeqKV, [4, 2, 3, 4],
    'each row observes its own causal prefix despite sharing one params buffer');

  const secondChunk = await executor.readBufferRange(
    outputBuffer, 2 * Int32Array.BYTES_PER_ELEMENT,
    2 * Int32Array.BYTES_PER_ELEMENT, 'int32',
  );
  assert.deepEqual([...secondChunk], [3, 0]);
  assert.equal(device.state.mapCount, 2, 'each two-token chunk is read once');
  await assert.rejects(
    executor.executeDeviceFeedbackDecode(null, {
      tokenInput: 'ids', keepInput: 'keep', output: 'next_ids',
      startPosition: 2, endPosition: sequence,
    }),
    /next position from a valid device-feedback seed\/cache/,
    'a stale or replayed chunk cannot mutate the retained cache',
  );
  executor.dispose();
});

test('WebGPU W8A32 Linear accepts aliases, typed zero points, and an odd feature width', async () => {
  const device = mockDevice();
  const input = tensor('input', [1, 3]);
  const weight = tensor('weight', [2, 3], { dtype: 'uint8' });
  const weightScale = tensor('weight_scale', [2]);
  const zeroPoint = tensor('weight_zero_point', [1], { dtype: 'uint8' });
  const bias = tensor('bias', [2]);
  const out = tensor('out', [1, 2]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getLinearInt8Shader: () => 'linear-w8a32',
      getLinearF32Shader: () => { throw new Error('quantized weights must select the W8A32 shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qlinear_tail', opType: 'Linear',
    inputs: { input, weight, weight_scale: weightScale, weight_zero_point: zeroPoint, bias },
    outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['linear-w8a32']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 6).bytes.buffer).slice(0, 8)], [
    1, 3, 2, DataType.U8, 2, DataType.U8, 1, 0,
  ]);
  executor.dispose();
});

test('WebGPU W8A32 keeps weight_scale graphs in their canonical output-major layout', async () => {
  const device = mockDevice();
  const input = tensor('input', [1, 2]);
  const weight = tensor('weight', [3, 2], { dtype: 'int8', buffer: Int8Array.of(1, 2, 3, 4, 5, 6) });
  const weightScale = tensor('weight_scale', [3], { buffer: Float32Array.of(0.5, 0.25, 0.125) });
  const out = tensor('out', [1, 3]);
  input.isInput = true;
  weight.isWeight = true;
  weightScale.isWeight = true;
  const graph = {
    nodes: [{
      id: 'qlinear_layout', opType: 'Linear', wLayout: 'din',
      inputs: { input, weight, weight_scale: weightScale }, outputs: { out }, params: {},
    }],
    tensors: new Map([[input.name, input], [weight.name, weight], [weightScale.name, weightScale], [out.name, out]]),
  };
  const executor = new GraphExecutor(device, graph, {
    shaderLibrary: { getLinearInt8Shader: () => 'linear-w8a32' },
  });

  await executor.compile();

  assert.equal(executor._dinWeights.has('weight'), false);
  for (const value of [input, weight, weightScale, out]) {
    assert.equal(Object.hasOwn(value, 'gpuBuffer'), false,
      'WebGPU allocation must remain executor-owned');
    assert.ok(executor.gpuBuffers.has(value.name));
  }
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 6).bytes.buffer).slice(0, 8)],
    [1, 2, 3, DataType.I8, 3, DataType.Unspecified, 0, 0],
  );
  executor.dispose();
});

test('WebGPU graph compilation plans expose recursively immutable records', () => {
  const input = tensor('input', [1, 2]);
  const weight = tensor('weight', [2, 3], { buffer: new Float32Array(6) });
  const linearOut = tensor('linear_out', [1, 3]);
  const dropped = tensor('dropped', [1, 3]);
  input.isInput = true;
  weight.isWeight = true;
  const plan = compileWebGPUGraphPlan({
    nodes: [{
      id: 'linear', opType: 'Linear', wLayout: 'din',
      inputs: { input, weight }, outputs: { out: linearOut }, params: {},
    }, {
      id: 'dropout', opType: 'Dropout',
      inputs: { input: linearOut }, outputs: { out: dropped }, params: {},
    }],
    outputNames: ['dropped'],
  });

  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.dinWeights), true);
  assert.equal(Object.isFrozen(plan.dinWeights[0]), true);
  assert.equal(Object.isFrozen(plan.dinWeights[0][1]), true);
  assert.equal(Object.isFrozen(plan.resultCopyTensorNames), true);
  assert.deepEqual(plan.dinWeights, [['weight', { din: 2, dout: 3 }]]);
  assert.deepEqual(plan.resultCopyTensorNames, ['dropped', 'linear_out']);
  assert.equal(plan.dinWeights.set, undefined);
  assert.equal(plan.resultCopyTensorNames.add, undefined);
  assert.throws(() => plan.dinWeights.push(['other', { din: 1, dout: 1 }]), TypeError);
  assert.throws(() => { plan.dinWeights[0][1].din = 9; }, TypeError);
  assert.throws(() => plan.resultCopyTensorNames.push('other'), TypeError);
  assert.throws(() => compileWebGPUGraphPlan({
    nodes: [{
      id: 'noncanonical_resize', opType: 'Resize', inputs: { input }, outputs: { out: dropped },
      params: { mode: 'nearest', coordinate_transform_mode: 'asymmetric' },
    }],
    outputNames: ['dropped'],
  }), /unsupported 'coordinate_transform_mode'.*coordinate_transformation_mode/);
});

test('WebGPU Linear routes odd tiled shapes and preserves every row on the scalar fallback', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getLinearF32Shader: () => 'linear-f32-scalar',
      getLinearF32TiledShader: () => 'linear-f32-tiled',
    },
  });

  const tiledInput = tensor('tiled_input', [17, 19]);
  const tiledWeight = tensor('tiled_weight', [23, 19]);
  const tiledOutput = tensor('tiled_output', [17, 23]);
  await executor._buildNodePipeline({
    id: 'linear_tiled_tails', opType: 'Linear',
    inputs: { input: tiledInput, weight: tiledWeight }, outputs: { out: tiledOutput }, params: {},
  });

  const scalarInput = tensor('scalar_input', [3, 7]);
  const scalarWeight = tensor('scalar_weight', [5, 7]);
  const scalarOutput = tensor('scalar_output', [3, 5]);
  await executor._buildNodePipeline({
    id: 'linear_scalar_rows', opType: 'Linear',
    inputs: { input: scalarInput, weight: scalarWeight }, outputs: { out: scalarOutput }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['linear-f32-tiled', 'linear-f32-scalar']);
  assert.deepEqual(executor.pipelines.map(({ workgroupCount }) => workgroupCount), [
    [2, 2, 1],
    [1, 3, 1],
  ]);
  executor.dispose();
});

test('WebGPU W8A32 and W8A8 Linear tile only above their portable thresholds', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getLinearInt8Shader: () => 'linear-w8a32-scalar',
      getLinearInt8TiledShader: () => 'linear-w8a32-tiled',
      getQLinearShader: () => 'linear-w8a8-scalar',
      getQLinearTiledShader: () => 'linear-w8a8-tiled',
    },
  });

  const w8a32Input = tensor('w8a32_input', [17, 19]);
  const w8a32Weight = tensor('w8a32_weight', [23, 19], { dtype: 'int8' });
  const w8a32Scale = tensor('w8a32_scale', [23]);
  const w8a32Output = tensor('w8a32_output', [17, 23]);
  await executor._buildNodePipeline({
    id: 'w8a32_tiled_tails', opType: 'Linear',
    inputs: { input: w8a32Input, weight: w8a32Weight, weight_scale: w8a32Scale },
    outputs: { out: w8a32Output }, params: {},
  });

  const makeQLinear = (prefix, rows, dIn, dOut) => {
    const input = quantizedTensor(`${prefix}_input`, [rows, dIn], {
      dtype: 'int8', scale: 1, zeroPoint: 0, buffer: new Int8Array(rows * dIn),
    });
    const weight = tensor(`${prefix}_weight`, [dOut, dIn], {
      dtype: 'int8', buffer: new Int8Array(dOut * dIn),
    });
    weight.quantization = {
      scheme: 'per_axis', axis: 0, scales: Array(dOut).fill(1), zero_points: Array(dOut).fill(0),
    };
    const bias = tensor(`${prefix}_bias`, [dOut], {
      dtype: 'int32', buffer: new Int32Array(dOut),
    });
    const out = quantizedTensor(`${prefix}_output`, [rows, dOut], {
      dtype: 'int8', scale: 1, zeroPoint: 0,
    });
    return { id: prefix, opType: 'QLinear', inputs: { input, weight, bias }, outputs: { out }, params: {} };
  };

  await executor._buildNodePipeline(makeQLinear('w8a8_tiled_tails', 9, 19, 36));
  await executor._buildNodePipeline(makeQLinear('w8a8_small_fallback', 3, 15, 28));

  assert.deepEqual(device.state.shaderCodes, [
    'linear-w8a32-tiled', 'linear-w8a8-tiled', 'linear-w8a8-scalar',
  ]);
  assert.deepEqual(executor.pipelines.map(({ workgroupCount }) => workgroupCount), [
    [2, 2, 1],
    [2, 2, 1],
    [1, 1, 1],
  ]);
  executor.dispose();
});

test('WebGPU QLinear binds packed W8A8 metadata and I32 bias without W8A32 fallback', async () => {
  const device = mockDevice();
  const input = tensor('input', [1, 3], { dtype: 'int8', buffer: Int8Array.of(1, -2, 3) });
  const weight = tensor('weight', [2, 3], { dtype: 'int8', buffer: Int8Array.of(2, 0, -1, -2, 1, 3) });
  const bias = tensor('bias', [2], { dtype: 'int32', buffer: Int32Array.of(2, -4) });
  const out = tensor('out', [1, 2], { dtype: 'int8' });
  input.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 };
  weight.quantization = { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2] };
  out.quantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQLinearShader: () => 'w8a8-qlinear',
      getLinearInt8Shader: () => { throw new Error('W8A8 QLinear must not select W8A32'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qlinear', opType: 'QLinear', inputs: { input, weight, bias }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qlinear']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5, 6]);
  const params = entries.find((entry) => entry.binding === 6).resource.buffer;
  const words = new Uint32Array(params.bytes.buffer);
  const signed = new Int32Array(params.bytes.buffer);
  const floats = new Float32Array(params.bytes.buffer);
  assert.deepEqual([...words.slice(0, 6)], [
    1, 3, 2, DataType.I8, DataType.I8, DataType.I8,
  ]);
  assert.deepEqual([...signed.slice(8, 10)], [-1, 0]);
  assert.deepEqual([...floats.slice(12, 14)], [0.25, 0.125]);
  const multiplierBuffer = entries.find((entry) => entry.binding === 2).resource.buffer;
  const zeroBuffer = entries.find((entry) => entry.binding === 3).resource.buffer;
  assert.deepEqual(
    [...new Float32Array(multiplierBuffer.bytes.buffer).slice(0, 2)],
    [1, 0.5],
  );
  assert.deepEqual([...new Int32Array(zeroBuffer.bytes.buffer).slice(0, 2)], [1, -2]);
  executor.dispose();
});

test('WebGPU packed-dot language feature selects DP4a QLinear, QMatMul, and QGemm variants', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(['packed_4x8_integer_dot_product']),
    shaderLibrary: {
      getQLinearShader: () => 'w8a8-scalar-fallback',
      getQLinearTiledShader: () => 'w8a8-tiled-fallback',
      getQLinearDotShader: () => 'w8a8-dot-scalar',
      getQLinearDotTiledShader: () => 'w8a8-dot-tiled',
    },
  });
  const makeNode = (opType, suffix, rows, dIn, dOut) => {
    const input = quantizedTensor(`input_${suffix}`, [rows, dIn], {
      dtype: 'uint8', scale: 0.25, zeroPoint: 127,
    });
    const weight = tensor(`weight_${suffix}`, [dOut, dIn], {
      dtype: 'int8', buffer: new Int8Array(dOut * dIn),
    });
    weight.quantization = {
      scheme: 'per_axis', axis: 0,
      scales: Array(dOut).fill(0.5), zero_points: Array(dOut).fill(-3),
    };
    const bias = tensor(`bias_${suffix}`, [dOut], {
      dtype: 'int32', buffer: new Int32Array(dOut),
    });
    const out = quantizedTensor(`out_${suffix}`, [rows, dOut], {
      dtype: 'int8', scale: 0.125, zeroPoint: -1,
    });
    return { id: suffix, opType, inputs: { input, weight, bias }, outputs: { out }, params: {} };
  };

  await executor._buildNodePipeline(makeNode('QLinear', 'linear', 1, 19, 7));
  await executor._buildNodePipeline(makeNode('QMatMul', 'matmul', 9, 19, 36));
  await executor._buildNodePipeline(makeNode('QGemm', 'gemm', 1, 7, 5));

  assert.equal(executor.hasPackedDot4, true);
  assert.deepEqual(device.state.shaderCodes, [
    'w8a8-dot-scalar', 'w8a8-dot-tiled',
  ]);
  assert.equal(executor.pipelines[0].pipeline, executor.pipelines[2].pipeline);
  assert.deepEqual(executor.pipelines.map(({ workgroupCount }) => workgroupCount), [
    [1, 1, 1], [2, 2, 1], [1, 1, 1],
  ]);
  executor.dispose();
});

test('WebGPU packed-dot QLinear compilation failure caches the feature-free fallback', async () => {
  const device = mockDevice();
  device.createComputePipelineAsync = async ({ compute }) => {
    if (compute.module.code === 'w8a8-dot') throw new Error('driver rejected packed dot');
    return { getBindGroupLayout() { return {}; } };
  };
  const input = quantizedTensor('input', [1, 4], { dtype: 'int8', scale: 1, zeroPoint: 0 });
  const weight = tensor('weight', [4, 4], { dtype: 'int8', buffer: new Int8Array(16) });
  weight.quantization = {
    scheme: 'per_axis', axis: 0, scales: [1, 1, 1, 1], zero_points: [0, 0, 0, 0],
  };
  const bias = tensor('bias', [4], { dtype: 'int32', buffer: new Int32Array(4) });
  const out = quantizedTensor('out', [1, 4], { dtype: 'int8', scale: 1, zeroPoint: 0 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(['packed_4x8_integer_dot_product']),
    shaderLibrary: {
      getQLinearShader: () => 'w8a8-portable',
      getQLinearDotShader: () => 'w8a8-dot',
    },
  });

  await executor._buildNodePipeline({
    id: 'dot_retry', opType: 'QLinear', inputs: { input, weight, bias }, outputs: { out }, params: {},
  });
  await executor._buildNodePipeline({
    id: 'dot_retry_again', opType: 'QLinear', inputs: { input, weight, bias }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-dot', 'w8a8-portable']);
  assert.equal(executor.pipelines.length, 2);
  assert.equal(executor.pipelines[0].pipeline, executor.pipelines[1].pipeline);
  assert.deepEqual([...executor.rejectedSpecializedShaders], ['w8a8-dot']);
  executor.dispose();
});

test('WebGPU QEmbedding binds row quantization metadata and packed byte output', async () => {
  const device = mockDevice();
  const ids = tensor('ids', [2, 2], { dtype: 'int32', buffer: Int32Array.of(2, 0, 1, 2) });
  ids.isInput = true;
  const weight = tensor('table', [3, 3], {
    dtype: 'int8', buffer: Int8Array.of(-1, 0, 1, 2, 0, -2, 5, 1, -3),
  });
  const out = tensor('out', [2, 2, 3], { dtype: 'uint8' });
  weight.quantization = {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25, 0.125], zero_points: [0, 0, 1],
  };
  out.quantization = { scheme: 'per_tensor', scale: 0.5, zero_point: 100 };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQEmbeddingShader: () => 'w8a8-qembedding',
      getEmbeddingShader: () => { throw new Error('QEmbedding must not select the F32 gather shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qembedding', opType: 'QEmbedding', inputs: { input: ids, weight }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qembedding']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5]);
  const params = entries.find((entry) => entry.binding === 5).resource.buffer;
  const words = new Uint32Array(params.bytes.buffer);
  const signed = new Int32Array(params.bytes.buffer);
  const floats = new Float32Array(params.bytes.buffer);
  assert.deepEqual([...words.slice(0, 5)], [
    4, 3, 3, DataType.I8, DataType.U8,
  ]);
  assert.equal(signed[5], 100);
  assert.equal(floats[6], 0.5);
  assert.deepEqual([...new Float32Array(entries.find((entry) => entry.binding === 2).resource.buffer.bytes.buffer).slice(0, 3)], [0.5, 0.25, 0.125]);
  assert.deepEqual([...new Int32Array(entries.find((entry) => entry.binding === 3).resource.buffer.bytes.buffer).slice(0, 3)], [0, 0, 1]);
  executor.dispose();
});

test('WebGPU QEmbedding accepts only vocabulary-bounded internal Clip IDs', async () => {
  const makeGraph = (maximum) => {
    const graph = new Graph();
    const raw = graph.addInput('raw_ids', [2], 'int32');
    const ids = graph.addOp('Clip', { input: raw }, {
      out: { name: 'ids', shape: [2], dtype: 'int32' },
    }, { min: 0, max: maximum }).out;
    const weight = graph.addWeight('table', [3, 2], 'int8', {
      buffer: Int8Array.of(1, -1, 2, -2, 3, -3),
      quantization: {
        scheme: 'per_axis', axis: 0,
        scales: [0.5, 0.25, 0.125], zero_points: [0, 0, 0],
      },
    });
    const out = graph.addOp('QEmbedding', { input: ids, weight }, {
      out: {
        name: 'out', shape: [2, 2], dtype: 'int8',
        quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
      },
    }).out;
    graph.setOutputs([out.name]);
    return graph;
  };

  const graph = makeGraph(2);
  const device = mockDevice();
  const executor = new GraphExecutor(device, graph, {
    shaderLibrary: { getQEmbeddingShader: () => 'bounded-qembedding' },
  });
  await executor._buildNodePipeline(graph.nodes[1]);
  assert.deepEqual(device.state.shaderCodes, ['bounded-qembedding']);
  assert.doesNotThrow(() => executor._preflightQEmbeddingIds({}));
  executor.dispose();

  const invalid = makeGraph(3);
  const rejected = new GraphExecutor(mockDevice(), invalid, {
    shaderLibrary: { getQEmbeddingShader: () => 'unbounded-qembedding' },
  });
  await assert.rejects(
    rejected._buildNodePipeline(invalid.nodes[1]),
    /preflight-complete I32 IDs/,
  );
  rejected.dispose();
});

test('WebGPU QEmbedding preflights graph-input IDs before any output dispatch', async () => {
  const ids = tensor('ids', [2], { dtype: 'int32' });
  ids.isInput = true;
  const weight = tensor('table', [3, 2], { dtype: 'int8' });
  const out = tensor('out', [2, 2], { dtype: 'int8' });
  const node = { id: 'preflight', opType: 'QEmbedding', inputs: { input: ids, weight }, outputs: { out }, params: {} };
  const executor = new GraphExecutor(mockDevice(), { nodes: [node], weightRevision: 0 }, { shaderLibrary: {} });
  executor.compiledWeightRevision = 0;

  assert.doesNotThrow(() => executor._preflightQEmbeddingIds({ ids: Int32Array.of(0, 2) }));
  await assert.rejects(
    () => executor.execute({ ids: Int32Array.of(0, 3) }),
    /token id 3 is outside vocabulary size 3/,
  );
  assert.throws(
    () => executor._preflightQEmbeddingIds({}),
    /requires graph-input I32 IDs supplied on every execution for preflight/,
  );
  executor.dispose();
});

test('WebGPU QConv2D binds canonical NHWC/OHWI W8A8 metadata', async () => {
  const device = mockDevice();
  const input = tensor('input', [1, 2, 2, 2], { dtype: 'int8', buffer: Int8Array.of(2, 4, 6, 8, 10, 12, 14, 16) });
  const weight = tensor('weight', [2, 1, 1, 2], { dtype: 'int8', buffer: Int8Array.of(1, -1, 2, 1) });
  const bias = tensor('bias', [2], { dtype: 'int32', buffer: Int32Array.of(1, -2) });
  const out = tensor('out', [1, 2, 2, 2], { dtype: 'int8' });
  input.quantization = { scheme: 'per_tensor', scale: 0.5, zero_point: 0 };
  weight.quantization = { scheme: 'per_axis', axis: 0, scales: [0.25, 0.5], zero_points: [0, 0] };
  out.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getQConv2DShader: () => 'w8a8-qconv' },
  });

  await executor._buildNodePipeline({
    id: 'qconv', opType: 'QConv2D', inputs: { input, weight, bias }, outputs: { out },
    params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qconv']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5, 6]);
  const params = entries.find((entry) => entry.binding === 6).resource.buffer;
  const words = new Uint32Array(params.bytes.buffer);
  const signed = new Int32Array(params.bytes.buffer);
  const floats = new Float32Array(params.bytes.buffer);
  assert.deepEqual([...words.slice(0, 20)], [
    1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1,
    DataType.I8, DataType.I8, DataType.I8, 0,
  ]);
  assert.deepEqual([...signed.slice(20, 22)], [0, 0]);
  assert.deepEqual([...floats.slice(24, 26)], [0.5, 0.25]);
  assert.deepEqual([...new Float32Array(entries.find((entry) => entry.binding === 2).resource.buffer.bytes.buffer).slice(0, 2)], [0.25, 0.5]);
  executor.dispose();
});

test('WebGPU packed-dot language feature selects cooperative tiled QConv2D', async () => {
  const device = mockDevice();
  const input = quantizedTensor('input', [1, 3, 3, 16], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 128,
  });
  const weight = tensor('weight', [16, 1, 1, 16], {
    dtype: 'int8', buffer: new Int8Array(16 * 16),
  });
  weight.quantization = {
    scheme: 'per_axis', axis: 0,
    scales: Array(16).fill(0.5), zero_points: Array(16).fill(-2),
  };
  const bias = tensor('bias', [16], { dtype: 'int32', buffer: new Int32Array(16) });
  const out = quantizedTensor('out', [1, 3, 3, 16], {
    dtype: 'int8', scale: 0.125, zeroPoint: -1,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(['packed_4x8_integer_dot_product']),
    shaderLibrary: {
      getQConv2DShader: () => 'w8a8-qconv-portable',
      getQConv2DDotTiledShader: () => 'w8a8-qconv-dot-tiled',
    },
  });

  await executor._buildNodePipeline({
    id: 'qconv_dot', opType: 'QConv2D', inputs: { input, weight, bias }, outputs: { out },
    params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qconv-dot-tiled']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 3, 1]);
  executor.dispose();
});

test('WebGPU without packed-dot selects cooperative portable tiled QConv2D', async () => {
  const device = mockDevice();
  const input = quantizedTensor('input', [1, 3, 3, 16], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 128,
  });
  const weight = tensor('weight', [32, 1, 1, 16], {
    dtype: 'int8', buffer: new Int8Array(32 * 16),
  });
  weight.quantization = {
    scheme: 'per_axis', axis: 0,
    scales: Array(32).fill(0.5), zero_points: Array(32).fill(-2),
  };
  const bias = tensor('bias', [32], { dtype: 'int32', buffer: new Int32Array(32) });
  const out = quantizedTensor('out', [1, 3, 3, 32], {
    dtype: 'int8', scale: 0.125, zeroPoint: -1,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(),
    shaderLibrary: {
      getQConv2DShader: () => 'w8a8-qconv-scalar',
      getQConv2DTiledShader: () => 'w8a8-qconv-portable-tiled',
      getQConv2DDotTiledShader: () => { throw new Error('packed dot was not advertised'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qconv_portable_tile', opType: 'QConv2D',
    inputs: { input, weight, bias }, outputs: { out },
    params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qconv-portable-tiled']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 3, 1]);
  executor.dispose();
});

test('WebGPU QConv2D reduction and spatial tails retain the packed-dot tile', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(['packed_4x8_integer_dot_product']),
    shaderLibrary: {
      getQConv2DShader: () => 'w8a8-qconv-portable',
      getQConv2DDotTiledShader: () => 'w8a8-qconv-dot-tiled',
    },
  });
  for (const inputChannels of [17, 18, 19]) {
    const input = quantizedTensor(`input_${inputChannels}`, [1, 3, 5, inputChannels], {
      dtype: 'uint8', scale: 0.02, zeroPoint: 131,
    });
    const weight = tensor(`weight_${inputChannels}`, [16, 1, 1, inputChannels], {
      dtype: 'int8', buffer: new Int8Array(16 * inputChannels),
    });
    weight.quantization = {
      scheme: 'per_axis', axis: 0,
      scales: Array(16).fill(0.03),
      zero_points: Array.from({ length: 16 }, (_, channel) => channel % 7 - 3),
    };
    const bias = tensor(`bias_${inputChannels}`, [16], {
      dtype: 'int32', buffer: new Int32Array(16),
    });
    const out = quantizedTensor(`out_${inputChannels}`, [1, 3, 5, 16], {
      dtype: 'uint8', scale: 0.04, zeroPoint: 113,
    });
    await executor._buildNodePipeline({
      id: `qconv_tail_${inputChannels % 4}`,
      opType: 'QConv2D', inputs: { input, weight, bias }, outputs: { out },
      params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
    });
    const params = paramsFrom(device, 6);
    const signed = new Int32Array(params.bytes.buffer);
    assert.deepEqual([...signed.slice(20, 22)], [131, 113]);
  }

  assert.deepEqual(executor.pipelines.map(({ workgroupCount }) => workgroupCount), [
    [1, 4, 1], [1, 4, 1], [1, 4, 1],
  ]);
  assert.deepEqual(device.state.shaderCodes, ['w8a8-qconv-dot-tiled']);
  assert.ok(executor.pipelines.every(({ pipeline }) => pipeline === executor.pipelines[0].pipeline));
  executor.dispose();
});

test('WebGPU QConv2D falls back when its tiled dispatch exceeds the device dimension limit', async () => {
  const device = mockDevice();
  device.limits = { maxComputeWorkgroupsPerDimension: 2 };
  const input = quantizedTensor('input', [1, 3, 3, 16], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 128,
  });
  const weight = tensor('weight', [16, 1, 1, 16], {
    dtype: 'int8', buffer: new Int8Array(16 * 16),
  });
  weight.quantization = {
    scheme: 'per_axis', axis: 0,
    scales: Array(16).fill(0.5), zero_points: Array(16).fill(0),
  };
  const bias = tensor('bias', [16], { dtype: 'int32', buffer: new Int32Array(16) });
  const out = quantizedTensor('out', [1, 3, 3, 16], {
    dtype: 'int8', scale: 0.125, zeroPoint: 0,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    wgslLanguageFeatures: new Set(['packed_4x8_integer_dot_product']),
    shaderLibrary: {
      getQConv2DShader: () => 'w8a8-qconv-portable',
      getQConv2DDotTiledShader: () => { throw new Error('oversized tile must not be selected'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qconv_limit', opType: 'QConv2D', inputs: { input, weight, bias }, outputs: { out },
    params: { data_layout: 'NHWC', weight_layout: 'OHWI' },
  });

  assert.deepEqual(device.state.shaderCodes, ['w8a8-qconv-portable']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  executor.dispose();
});

test('WebGPU compute pipeline cache keys shader, entry point, and sorted constants', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] });
  const first = await executor._cachedComputePipeline('shared-shader', {
    entryPoint: 'main', constants: { TILE_N: 8, TILE_M: 4 },
  });
  const reordered = await executor._cachedComputePipeline('shared-shader', {
    entryPoint: 'main', constants: { TILE_M: 4, TILE_N: 8 },
  });
  const otherEntry = await executor._cachedComputePipeline('shared-shader', {
    entryPoint: 'secondary', constants: { TILE_M: 4, TILE_N: 8 },
  });
  const otherConstant = await executor._cachedComputePipeline('shared-shader', {
    entryPoint: 'main', constants: { TILE_M: 8, TILE_N: 8 },
  });

  assert.equal(first, reordered);
  assert.notEqual(first, otherEntry);
  assert.notEqual(first, otherConstant);
  assert.deepEqual(device.state.shaderCodes, ['shared-shader', 'shared-shader', 'shared-shader']);
  assert.equal(executor.computePipelineCache.size, 1);
  executor.dispose();
  assert.equal(executor.computePipelineCache.size, 0);
});

test('WebGPU CrossAttention selects the authoritative F32 shader and encodes its contract', async () => {
  const device = mockDevice();
  const q = tensor('q', [2, 3, 4]);
  const kv = tensor('kv', [2, 2, 4]);
  const weight = tensor('weight', [12, 4], { buffer: new Float32Array(48) });
  const scale = tensor('scale', [12], { buffer: new Float32Array(12) });
  const bias = tensor('bias', [12], { buffer: new Float32Array(12) });
  const out = tensor('out', [2, 3, 4]);
  const node = {
    id: 'cross_f32', opType: 'CrossAttention',
    inputs: { q, kv, weight, scale, bias }, outputs: { out }, params: { heads: 2 },
  };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getCrossAttentionF32Shader: () => 'cross-attention-f32',
      getCrossAttentionShader: () => { throw new Error('F32 weights must not select the packed-I8 shader'); },
    },
  });

  await executor._buildNodePipeline(node);

  assert.deepEqual(device.state.shaderCodes, ['cross-attention-f32']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 2, 2]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5, 6]);
  const params = paramsFrom(device, 6);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 5)], [3, 2, 4, 2, 2]);
  assert.ok(Math.abs(new Float32Array(params.bytes.buffer)[5] - 1 / Math.sqrt(2)) < 1e-7);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(6, 9)], [1, 1, 2]);
  executor.dispose();
});

test('WebGPU CrossAttention retains the packed-I8 shader path and rejects unsupported F32 dimensions', async () => {
  const device = mockDevice();
  const q = tensor('q', [2, 4]);
  const kv = tensor('kv', [3, 4]);
  const weight = tensor('weight', [12, 4], { dtype: 'int8', buffer: new Int8Array(48) });
  const out = tensor('out', [2, 4]);
  const node = {
    id: 'cross_i8', opType: 'CrossAttention', inputs: { q, kv, weight }, outputs: { out }, params: { heads: 1 },
  };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getCrossAttentionF32Shader: () => { throw new Error('packed I8 must retain its existing shader'); },
      getCrossAttentionShader: () => 'cross-attention-i8',
    },
  });
  await executor._buildNodePipeline(node);
  assert.deepEqual(device.state.shaderCodes, ['cross-attention-i8']);
  executor.dispose();

  const tooWide = tensor('too_wide', [1, 1, 128]);
  const tooWideKv = tensor('too_wide_kv', [1, 1, 128]);
  const tooWideWeight = tensor('too_wide_weight', [384, 128], { buffer: new Float32Array(384 * 128) });
  const tooWideOut = tensor('too_wide_out', [1, 1, 128]);
  const f32Executor = new GraphExecutor(mockDevice(), { nodes: [] }, {
    shaderLibrary: { getCrossAttentionF32Shader: () => 'cross-attention-f32' },
  });
  await assert.rejects(
    () => f32Executor._buildNodePipeline({
      id: 'cross_f32_too_wide', opType: 'CrossAttention',
      inputs: { q: tooWide, kv: tooWideKv, weight: tooWideWeight }, outputs: { out: tooWideOut }, params: { heads: 2 },
    }),
    /d_model <= 64/,
  );
  f32Executor.dispose();
});

test('WebGPU NonMaxSuppression binds the native-compatible parameter layout', async () => {
  const device = mockDevice();
  const boxes = tensor('boxes', [2, 5, 4]);
  const scores = tensor('scores', [2, 3, 5]);
  const maxOutput = tensor('max_output', [1], { buffer: Float32Array.of(2) });
  const iou = tensor('iou', [1], { buffer: Float32Array.of(0.4) });
  const score = tensor('score', [1], { buffer: Float32Array.of(0.2) });
  const out = tensor('out', [12, 3]);
  const node = {
    id: 'nms', opType: 'NonMaxSuppression',
    inputs: {
      boxes, scores,
      max_output_boxes_per_class: maxOutput,
      iou_threshold: iou,
      score_threshold: score,
    },
    outputs: { out }, params: {},
  };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getNonMaxSuppressionShader: () => 'non-max-suppression' },
  });

  await executor._buildNodePipeline(node);

  assert.deepEqual(device.state.shaderCodes, ['non-max-suppression']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const params = paramsFrom(device, 3);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 5)], [2, 5, 3, 2, 12]);
  const floats = new Float32Array(params.bytes.buffer);
  assert.ok(Math.abs(floats[5] - 0.4) < 1e-7);
  assert.ok(Math.abs(floats[6] - 0.2) < 1e-7);
  executor.dispose();
});

test('WebGPU NonMaxSuppression rejects dynamic scalar parameters that cannot populate its uniform', async () => {
  const device = mockDevice();
  const boxes = tensor('boxes', [1, 2, 4]);
  const scores = tensor('scores', [1, 1, 2]);
  const out = tensor('out', [2, 3]);
  const maxOutput = tensor('max_output', [1]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getNonMaxSuppressionShader: () => 'non-max-suppression' },
  });
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'nms_dynamic', opType: 'NonMaxSuppression',
      inputs: { boxes, scores, max_output_boxes_per_class: maxOutput }, outputs: { out }, params: {},
    }),
    /host-resident scalar/,
  );
  executor.dispose();
});

test('WebGPU Gather uses the rank-aware I32 kernel beyond axis zero', async () => {
  const device = mockDevice();
  const input = tensor('input', [2, 3, 4]);
  const indices = tensor('indices', [2, 2], { dtype: 'int32' });
  const out = tensor('out', [2, 2, 2, 4]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getGatherInt32Shader: () => 'gather-i32',
    },
  });

  await executor._buildNodePipeline({
    id: 'gather_axis_one', opType: 'Gather', inputs: { input, indices }, outputs: { out }, params: { axis: 1 },
  });

  assert.deepEqual(device.state.shaderCodes, ['gather-i32']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const params = paramsFrom(device, 3);
  const words = new Uint32Array(params.bytes.buffer);
  assert.deepEqual([...words.slice(0, 4)], [3, 2, 1, 32]);
  assert.deepEqual([...words.slice(4, 7)], [2, 3, 4]);
  assert.deepEqual([...words.slice(12, 16)], [2, 2, 2, 4]);
  executor.dispose();
});

test('WebGPU Gather rejects F32 indices for every axis', async () => {
  const input = tensor('input', [4, 3]);
  const indices = tensor('indices', [2]);
  const out = tensor('out', [2, 3]);
  const nonAxisOutput = tensor('non_axis_out', [4, 2]);
  for (const [axis, output] of [[0, out], [1, nonAxisOutput]]) {
    const executor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
    await assert.rejects(
      () => executor._buildNodePipeline({
        id: `gather_f32_axis_${axis}`, opType: 'Gather', inputs: { input, indices }, outputs: { out: output }, params: { axis },
      }),
      /requires I32 indices/,
    );
    executor.dispose();
  }
});

test('WebGPU GatherElements binds I32 index metadata and validates matching shapes', async () => {
  const device = mockDevice();
  const input = tensor('input', [2, 3, 4]);
  const indices = tensor('indices', [2, 2, 4], { dtype: 'int32' });
  const out = tensor('out', [2, 2, 4]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getGatherElementsShader: () => 'gather-elements-i32' },
  });

  await executor._buildNodePipeline({
    id: 'gather_elements', opType: 'GatherElements',
    inputs: { input, indices }, outputs: { out }, params: { axis: -2 },
  });

  assert.deepEqual(device.state.shaderCodes, ['gather-elements-i32']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const words = new Uint32Array(paramsFrom(device, 3).bytes.buffer);
  assert.deepEqual([...words.slice(0, 4)], [3, 1, 16, 0]);
  assert.deepEqual([...words.slice(4, 7)], [2, 3, 4]);
  assert.deepEqual([...words.slice(12, 15)], [2, 2, 4]);
  executor.dispose();

  const f32Indices = tensor('f32_indices', [2, 2, 4]);
  const invalidExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => invalidExecutor._buildNodePipeline({
      id: 'gather_elements_f32', opType: 'GatherElements',
      inputs: { input, indices: f32Indices }, outputs: { out }, params: { axis: 1 },
    }),
    /I32 indices/,
  );
  invalidExecutor.dispose();
});

test('WebGPU ArgMax accepts keepdims output shapes and selects the F32 shader', async () => {
  const device = mockDevice();
  const input = tensor('input', [2, 3, 2]);
  const out = tensor('out', [2, 1, 2]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getArgMaxF32Shader: () => 'argmax-f32',
      getArgMaxI32Shader: () => { throw new Error('F32 output must not select the I32 shader'); },
      getArgMaxI8Shader: () => { throw new Error('F32 output must not select the packed byte shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'argmax_keepdims', opType: 'ArgMax', inputs: { input }, outputs: { out }, params: { axis: 1, keepdims: 1 },
  });

  assert.deepEqual(device.state.shaderCodes, ['argmax-f32']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 2).bytes.buffer)], [2, 3, 2, 4]);
  executor.dispose();
});

test('WebGPU ArgMax respects declared I32 and packed byte output storage', async () => {
  const i32Device = mockDevice();
  const i32Input = tensor('input', [2, 3, 2]);
  const i32Out = tensor('out', [2, 2], { dtype: 'int32' });
  const i32Executor = new GraphExecutor(i32Device, { nodes: [] }, {
    shaderLibrary: {
      getArgMaxF32Shader: () => { throw new Error('I32 output must not select the F32 shader'); },
      getArgMaxI32Shader: () => 'argmax-i32',
      getArgMaxI8Shader: () => { throw new Error('I32 output must not select the packed byte shader'); },
    },
  });
  await i32Executor._buildNodePipeline({
    id: 'argmax_i32', opType: 'ArgMax', inputs: { input: i32Input }, outputs: { out: i32Out }, params: { axis: 1 },
  });
  assert.deepEqual(i32Device.state.shaderCodes, ['argmax-i32']);
  assert.deepEqual([...new Uint32Array(paramsFrom(i32Device, 2).bytes.buffer)], [2, 3, 2, 4]);
  i32Executor.dispose();

  const byteDevice = mockDevice();
  const byteInput = tensor('byte_input', [5, 3]);
  const byteOut = tensor('byte_out', [5], { dtype: 'uint8' });
  const byteExecutor = new GraphExecutor(byteDevice, { nodes: [] }, {
    shaderLibrary: {
      getArgMaxF32Shader: () => { throw new Error('byte output must not select the F32 shader'); },
      getArgMaxI32Shader: () => { throw new Error('byte output must not select the I32 shader'); },
      getArgMaxI8Shader: () => 'argmax-packed-i8',
    },
  });
  await byteExecutor._buildNodePipeline({
    id: 'argmax_u8', opType: 'ArgMax', inputs: { input: byteInput }, outputs: { out: byteOut }, params: { axis: -1 },
  });
  assert.deepEqual(byteDevice.state.shaderCodes, ['argmax-packed-i8']);
  assert.deepEqual(byteExecutor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(paramsFrom(byteDevice, 2).bytes.buffer)], [5, 3, 1, 5]);
  byteExecutor.dispose();

  const signedDevice = mockDevice();
  const signedOut = tensor('signed_out', [5], { dtype: 'int8' });
  const signedExecutor = new GraphExecutor(signedDevice, { nodes: [] }, {
    shaderLibrary: { getArgMaxI8Shader: () => 'argmax-packed-i8' },
  });
  await signedExecutor._buildNodePipeline({
    id: 'argmax_i8', opType: 'ArgMax', inputs: { input: byteInput }, outputs: { out: signedOut }, params: { axis: -1 },
  });
  assert.deepEqual(signedDevice.state.shaderCodes, ['argmax-packed-i8']);
  signedExecutor.dispose();
});

test('WebGPU ArgMax rejects a declared output shape incompatible with its axis', async () => {
  const input = tensor('input', [2, 3, 2]);
  const out = tensor('out', [2, 3]);
  const executor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'argmax_bad_shape', opType: 'ArgMax', inputs: { input }, outputs: { out }, params: { axis: 1 },
    }),
    /output must remove its axis or retain it with dimension 1/,
  );
  executor.dispose();
});

test('WebGPU Cast dispatches raw typed storage and packs byte outputs', async () => {
  const device = mockDevice();
  const input = tensor('input', [300]);
  const out = tensor('out', [300], { dtype: 'int8' });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getCastShader: () => 'typed-cast',
      getCopyShader: () => { throw new Error('Cast must not use the F32 copy shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'cast_f32_i8', opType: 'Cast', inputs: { input }, outputs: { out }, params: { to: 'int8' },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-cast']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [2, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2]);
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 2).bytes.buffer).slice(0, 4)],
    [300, DataType.F32, DataType.I8, 0],
  );
  executor.dispose();

  const intDevice = mockDevice();
  const intInput = tensor('int_input', [5], { dtype: 'int8' });
  const intOut = tensor('int_out', [5], { dtype: 'int32' });
  const intExecutor = new GraphExecutor(intDevice, { nodes: [] }, {
    shaderLibrary: { getCastShader: () => 'typed-cast' },
  });
  await intExecutor._buildNodePipeline({
    id: 'cast_i8_i32', opType: 'Cast', inputs: { input: intInput }, outputs: { out: intOut }, params: { to: 'int32' },
  });
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(intDevice, 2).bytes.buffer).slice(0, 4)],
    [5, DataType.I8, DataType.I32, 0],
  );
  intExecutor.dispose();
});

test('WebGPU DequantizeLinear binds typed scalar scale and zero-point metadata', async () => {
  const device = mockDevice();
  const input = tensor('input', [5], { dtype: 'int8' });
  const scale = tensor('scale', [1]);
  const zeroPoint = tensor('zero_point', [1], { dtype: 'uint8' });
  const out = tensor('out', [5]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getDequantizeLinearShader: () => 'typed-dequantize' },
  });

  await executor._buildNodePipeline({
    id: 'dequantize_i8', opType: 'DequantizeLinear',
    inputs: { input, scale, zero_point: zeroPoint }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-dequantize']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3, 4]);
  const words = new Uint32Array(paramsFrom(device, 4).bytes.buffer);
  assert.deepEqual([...words.slice(0, 8)], [
    5, DataType.I8, DataType.F32, DataType.U8, DataType.F32, 1, 0, 0,
  ]);
  executor.dispose();
});

test('WebGPU QuantizeLinear binds typed byte output and packed-word dispatch metadata', async () => {
  const device = mockDevice();
  const input = tensor('input', [5], { buffer: Float32Array.of(-1, -0.5, 0, 0.5, 1) });
  const scale = tensor('scale', [1], { buffer: Float32Array.of(0.25) });
  const zeroPoint = tensor('zero', [1], { dtype: 'int8', buffer: Int8Array.of(0) });
  const out = tensor('out', [5], { dtype: 'int8' });
  out.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getQuantizeLinearShader: () => 'typed-quantize' },
  });

  await executor._buildNodePipeline({
    id: 'quantize_i8', opType: 'QuantizeLinear',
    inputs: { input, scale, zero_point: zeroPoint }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-quantize']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3, 4]);
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 4).bytes.buffer).slice(0, 4)],
    [5, DataType.I8, DataType.I8, 1],
  );
  executor.dispose();
});

test('WebGPU RequantizeLinear uses immutable typed metadata without a scale buffer', async () => {
  const device = mockDevice();
  const input = quantizedTensor('requant_input', [5], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(-3, -1, 0, 2, 10),
  });
  const out = quantizedTensor('requant_out', [5], { dtype: 'uint8', scale: 0.125, zeroPoint: 128 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getRequantizeLinearShader: () => 'typed-requantize' },
  });

  await executor._buildNodePipeline({
    id: 'requant_i8_to_u8', opType: 'RequantizeLinear', inputs: { input }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-requantize']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2]);
  const params = entries.find((entry) => entry.binding === 2).resource.buffer;
  assert.deepEqual(
    [...new Uint32Array(params.bytes.buffer).slice(0, 4)],
    [5, DataType.I8, DataType.U8, 0],
  );
  assert.deepEqual([...new Int32Array(params.bytes.buffer).slice(4, 6)], [-1, 128]);
  assert.deepEqual([...new Float32Array(params.bytes.buffer).slice(8, 9)], [2]);
  executor.dispose();

  const scale = tensor('forbidden_scale', [1], { buffer: Float32Array.of(0.25) });
  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'requant_dynamic_scale', opType: 'RequantizeLinear', inputs: { input, scale }, outputs: { out }, params: {},
    }),
    /scales belong only to immutable tensor metadata/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU QAdd binds mixed byte descriptors and packed-word output metadata', async () => {
  const device = mockDevice();
  const a = tensor('a', [5], { dtype: 'int8', buffer: Int8Array.of(-2, 0, 2, 10, -128) });
  const b = tensor('b', [5], { dtype: 'uint8', buffer: Uint8Array.of(128, 132, 120, 255, 0) });
  const out = tensor('out', [5], { dtype: 'int8' });
  a.quantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -2 };
  b.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 128 };
  out.quantization = { scheme: 'per_tensor', scale: 0.5, zero_point: 3 };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getQAddShader: () => 'typed-qadd' },
  });

  await executor._buildNodePipeline({
    id: 'qadd', opType: 'QAdd', inputs: { a, b }, outputs: { out }, params: { relu: 2 },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qadd']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const words = new Uint32Array(paramsFrom(device, 3).bytes.buffer);
  const signed = new Int32Array(paramsFrom(device, 3).bytes.buffer);
  const floats = new Float32Array(paramsFrom(device, 3).bytes.buffer);
  assert.deepEqual([...words.slice(0, 4)], [
    5, DataType.I8, DataType.U8, DataType.I8,
  ]);
  assert.deepEqual([...signed.slice(4, 7)], [-2, 128, 3]);
  assert.deepEqual([...floats.slice(8, 11)], [0.5, 0.25, 0.5]);
  assert.equal(words[11], 2);
  executor.dispose();
});

test('WebGPU QSiLU binds one packed byte input/output pair and immutable descriptors', async () => {
  const device = mockDevice();
  const input = quantizedTensor('qsilu_input', [5], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 127, buffer: Uint8Array.of(124, 127, 128, 130, 133),
  });
  const out = quantizedTensor('qsilu_out', [5], { dtype: 'int8', scale: 0.125, zeroPoint: -3 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQSiLUShader: () => 'typed-qsilu',
      getSiLUShader: () => { throw new Error('QSiLU must not select the F32 SiLU shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qsilu', opType: 'QSiLU', inputs: { input }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qsilu']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2]);
  const params = entries.find((entry) => entry.binding === 2).resource.buffer;
  assert.deepEqual(
    [...new Uint32Array(params.bytes.buffer).slice(0, 3)],
    [5, DataType.U8, DataType.I8],
  );
  assert.deepEqual([...new Int32Array(params.bytes.buffer).slice(4, 6)], [127, -3]);
  assert.deepEqual([...new Float32Array(params.bytes.buffer).slice(8, 10)], [0.25, 0.125]);
  executor.dispose();

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'qsilu_extra', opType: 'QSiLU', inputs: { input, extra: input }, outputs: { out }, params: {},
    }),
    /distinct same-shape I8\/U8 input\/output tensors and no parameters/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU QGELU binds one packed byte input/output pair with fixed portable-erf semantics', async () => {
  const device = mockDevice();
  const input = quantizedTensor('qgelu_input', [5], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 127, buffer: Uint8Array.of(124, 127, 128, 130, 133),
  });
  const out = quantizedTensor('qgelu_out', [5], { dtype: 'int8', scale: 0.125, zeroPoint: -3 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQGELUShader: () => 'typed-qgelu',
      getGELUShader: () => { throw new Error('QGELU must not select the F32 GELU shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qgelu', opType: 'QGELU', inputs: { input }, outputs: { out }, params: { approximate: 'none' },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qgelu']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2]);
  const params = entries.find((entry) => entry.binding === 2).resource.buffer;
  assert.deepEqual(
    [...new Uint32Array(params.bytes.buffer).slice(0, 3)],
    [5, DataType.U8, DataType.I8],
  );
  assert.deepEqual([...new Int32Array(params.bytes.buffer).slice(4, 6)], [127, -3]);
  assert.deepEqual([...new Float32Array(params.bytes.buffer).slice(8, 10)], [0.25, 0.125]);
  executor.dispose();

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'qgelu_tanh', opType: 'QGELU', inputs: { input }, outputs: { out }, params: { approximate: 'tanh' },
    }),
    /only omitted parameters or approximate='none'/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU QGroupNorm dispatches byte-resident stats then apply with one graph-lifetime stats buffer', async () => {
  const device = mockDevice();
  // C=6/G=2 splits a packed u32 word at channel 3, and B=3 leaves a two-byte
  // tail. This exercises the reason stats/apply must be separate dispatches.
  const input = quantizedTensor('qgroupnorm_input', [3, 1, 1, 6], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(
      -8, -3, 4, 7, 5, -1,
      2, -6, 0, 8, -4, 3,
      -7, 6, 1, -2, 9, -5,
    ),
  });
  const weight = tensor('qgroupnorm_weight', [6], {
    dtype: 'float32', buffer: Float32Array.of(1, -0.75, 0.5, 1.25, -0.5, 0.25),
  });
  const bias = tensor('qgroupnorm_bias', [6], {
    dtype: 'float32', buffer: Float32Array.of(0.25, -0.5, 0.75, -0.25, 0.5, -0.75),
  });
  weight.isWeight = true;
  bias.isWeight = true;
  const out = quantizedTensor('qgroupnorm_out', [3, 1, 1, 6], {
    dtype: 'int8', scale: 0.125, zeroPoint: -3,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQGroupNormStatsShader: () => 'typed-qgroupnorm-stats',
      getQGroupNormApplyShader: () => 'typed-qgroupnorm-apply',
      getGroupNormShader: () => { throw new Error('QGroupNorm must not select the F32 GroupNorm shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qgroupnorm', opType: 'QGroupNorm', inputs: { input, weight, bias }, outputs: { out },
    params: { num_groups: 2, eps: 1e-5, data_layout: 'NHWC' },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qgroupnorm-stats', 'typed-qgroupnorm-apply']);
  assert.equal(executor.pipelines.length, 2);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [6, 1, 1]);
  assert.deepEqual(executor.pipelines[1].workgroupCount, [1, 1, 1]);
  assert.match(executor.pipelines[0].nodeName, /qgroupnorm_stats$/);
  assert.match(executor.pipelines[1].nodeName, /qgroupnorm_apply$/);

  const statsEntries = device.state.bindGroups.at(-2).entries;
  const applyEntries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(statsEntries.map(({ binding }) => binding), [0, 1, 2]);
  assert.deepEqual(applyEntries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5]);
  const statsBuffer = statsEntries.find((entry) => entry.binding === 1).resource.buffer;
  const statsParams = statsEntries.find((entry) => entry.binding === 2).resource.buffer;
  assert.equal(statsBuffer.descriptor.size, 48, 'two F32 statistics for each of six [batch,group] entries');
  assert.equal(applyEntries.find((entry) => entry.binding === 3).resource.buffer, statsBuffer);
  assert.equal(applyEntries.find((entry) => entry.binding === 5).resource.buffer, statsParams);
  assert.deepEqual(
    [...new Uint32Array(statsParams.bytes.buffer).slice(0, 8)],
    [3, 1, 1, 6, 2, DataType.I8, DataType.I8, 0],
  );
  assert.deepEqual([...new Int32Array(statsParams.bytes.buffer).slice(8, 10)], [-1, -3]);
  assert.deepEqual([...new Float32Array(statsParams.bytes.buffer).slice(12, 15)], [0.25, 0.125, Math.fround(1e-5)]);
  assert.ok(executor.auxiliaryBuffers.has(statsBuffer));
  executor.dispose();
  assert.equal(statsBuffer.destroyed, true);
  assert.equal(statsParams.destroyed, true);

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'qgroupnorm_bad_groups', opType: 'QGroupNorm', inputs: { input, weight, bias }, outputs: { out },
      params: { num_groups: 4, eps: 1e-5 },
    }),
    /finite F32 \[C\] weight\/bias tensors and num_groups dividing C/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU QGroupNorm defers dynamic affine finite checks until supplied execution inputs', async () => {
  const device = mockDevice();
  const input = quantizedTensor('dynamic_qgroupnorm_input', [1, 1, 1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(-8, -3, 4, 7),
  });
  const weight = tensor('dynamic_qgroupnorm_weight', [4], {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const bias = tensor('dynamic_qgroupnorm_bias', [4], {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  input.isInput = true;
  weight.isInput = true;
  bias.isInput = true;
  const out = quantizedTensor('dynamic_qgroupnorm_out', [1, 1, 1, 4], {
    dtype: 'int8', scale: 0.125, zeroPoint: -3,
  });
  const node = {
    id: 'dynamic_qgroupnorm', opType: 'QGroupNorm', inputs: { input, weight, bias }, outputs: { out },
    params: { num_groups: 2, eps: 1e-5 },
  };
  const executor = new GraphExecutor(device, { nodes: [node] }, {
    shaderLibrary: {
      getQGroupNormStatsShader: () => 'typed-qgroupnorm-stats',
      getQGroupNormApplyShader: () => 'typed-qgroupnorm-apply',
    },
  });

  await assert.doesNotReject(() => executor._buildNodePipeline(node));
  assert.doesNotThrow(() => executor._preflightQGroupNormDynamicAffines({
    dynamic_qgroupnorm_input: Int8Array.of(-8, -3, 4, 7),
    dynamic_qgroupnorm_weight: Float32Array.of(1, -0.75, 0.5, 1.25),
    dynamic_qgroupnorm_bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
  }));
  assert.throws(
    () => executor._preflightQGroupNormDynamicAffines({
      dynamic_qgroupnorm_input: Int8Array.of(-8, -3, 4, 7),
      dynamic_qgroupnorm_weight: Float32Array.of(1, Number.NaN, 0.5, 1.25),
      dynamic_qgroupnorm_bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
    }),
    /weight must contain finite F32 values before dispatch/,
  );
  assert.throws(
    () => executor._preflightQGroupNormDynamicAffines({
      dynamic_qgroupnorm_input: Int8Array.of(-8, -3, 4, 7),
      dynamic_qgroupnorm_weight: Float32Array.of(1, -0.75, 0.5, 1.25),
    }),
    /graph-input F32 bias supplied on every execution/,
  );
  executor.dispose();
});

test('WebGPU QLayerNorm dispatches byte-resident row stats then packed apply with graph-lifetime scratch', async () => {
  const device = mockDevice();
  // D=5 leaves a three-byte logical tail in the last packed output word.
  const input = quantizedTensor('qlayernorm_input', [3, 5], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1),
  });
  const weight = tensor('qlayernorm_weight', [5], {
    dtype: 'float32', buffer: Float32Array.of(1, -0.75, 0.5, 1.25, -0.5),
  });
  const bias = tensor('qlayernorm_bias', [5], {
    dtype: 'float32', buffer: Float32Array.of(0.25, -0.5, 0.75, -0.25, 0.5),
  });
  weight.isWeight = true;
  bias.isWeight = true;
  const out = quantizedTensor('qlayernorm_out', [3, 5], {
    dtype: 'uint8', scale: 0.125, zeroPoint: 123,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQLayerNormStatsShader: () => 'typed-qlayernorm-stats',
      getQLayerNormApplyShader: () => 'typed-qlayernorm-apply',
      getLayerNormShader: () => { throw new Error('QLayerNorm must not select the F32 LayerNorm shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qlayernorm', opType: 'QLayerNorm', inputs: { input, weight, bias }, outputs: { out },
    params: { eps: 1e-5, d_model: 5 },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qlayernorm-stats', 'typed-qlayernorm-apply']);
  assert.equal(executor.pipelines.length, 2);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [3, 1, 1]);
  assert.deepEqual(executor.pipelines[1].workgroupCount, [1, 1, 1]);
  assert.match(executor.pipelines[0].nodeName, /qlayernorm_stats$/);
  assert.match(executor.pipelines[1].nodeName, /qlayernorm_apply$/);

  const statsEntries = device.state.bindGroups.at(-2).entries;
  const applyEntries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(statsEntries.map(({ binding }) => binding), [0, 1, 2]);
  assert.deepEqual(applyEntries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5]);
  const statsBuffer = statsEntries.find((entry) => entry.binding === 1).resource.buffer;
  const statsParams = statsEntries.find((entry) => entry.binding === 2).resource.buffer;
  assert.equal(statsBuffer.descriptor.size, 24, 'two F32 statistics for each of three rows');
  assert.equal(applyEntries.find((entry) => entry.binding === 3).resource.buffer, statsBuffer);
  assert.equal(applyEntries.find((entry) => entry.binding === 5).resource.buffer, statsParams);
  assert.deepEqual(
    [...new Uint32Array(statsParams.bytes.buffer).slice(0, 4)],
    [3, 5, DataType.I8, DataType.U8],
  );
  assert.deepEqual([...new Int32Array(statsParams.bytes.buffer).slice(4, 6)], [-1, 123]);
  assert.deepEqual([...new Float32Array(statsParams.bytes.buffer).slice(8, 11)], [0.25, 0.125, Math.fround(1e-5)]);
  assert.ok(executor.auxiliaryBuffers.has(statsBuffer));
  executor.dispose();
  assert.equal(statsBuffer.destroyed, true);
  assert.equal(statsParams.destroyed, true);

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'qlayernorm_bad_d', opType: 'QLayerNorm', inputs: { input, weight, bias }, outputs: { out },
      params: { d_model: 4 },
    }),
    /positive eps with optional d_model matching D/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU QLayerNorm defers dynamic affine finite checks until supplied execution inputs', async () => {
  const device = mockDevice();
  const input = quantizedTensor('dynamic_qlayernorm_input', [1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(-8, -3, 4, 7),
  });
  const weight = tensor('dynamic_qlayernorm_weight', [4], {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  const bias = tensor('dynamic_qlayernorm_bias', [4], {
    buffer: Float32Array.of(Number.NaN, Number.NaN, Number.NaN, Number.NaN),
  });
  input.isInput = true;
  weight.isInput = true;
  bias.isInput = true;
  const out = quantizedTensor('dynamic_qlayernorm_out', [1, 4], {
    dtype: 'int8', scale: 0.125, zeroPoint: -3,
  });
  const node = {
    id: 'dynamic_qlayernorm', opType: 'QLayerNorm', inputs: { input, weight, bias }, outputs: { out },
    params: {},
  };
  const executor = new GraphExecutor(device, { nodes: [node] }, {
    shaderLibrary: {
      getQLayerNormStatsShader: () => 'typed-qlayernorm-stats',
      getQLayerNormApplyShader: () => 'typed-qlayernorm-apply',
    },
  });

  await assert.doesNotReject(() => executor._buildNodePipeline(node));
  assert.doesNotThrow(() => executor._preflightQLayerNormDynamicAffines({
    dynamic_qlayernorm_input: Int8Array.of(-8, -3, 4, 7),
    dynamic_qlayernorm_weight: Float32Array.of(1, -0.75, 0.5, 1.25),
    dynamic_qlayernorm_bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
  }));
  assert.throws(
    () => executor._preflightQLayerNormDynamicAffines({
      dynamic_qlayernorm_input: Int8Array.of(-8, -3, 4, 7),
      dynamic_qlayernorm_weight: Float32Array.of(1, Number.NaN, 0.5, 1.25),
      dynamic_qlayernorm_bias: Float32Array.of(0.25, -0.5, 0.75, -0.25),
    }),
    /weight must contain finite F32 values before dispatch/,
  );
  assert.throws(
    () => executor._preflightQLayerNormDynamicAffines({
      dynamic_qlayernorm_input: Int8Array.of(-8, -3, 4, 7),
      dynamic_qlayernorm_weight: Float32Array.of(1, -0.75, 0.5, 1.25),
    }),
    /graph-input F32 bias supplied on every execution/,
  );
  executor.dispose();
});

test('WebGPU QSDPA binds packed byte Q/K/V, a dummy mask, and the 80-byte canonical ABI', async () => {
  const device = mockDevice();
  const q = quantizedTensor('qsdpa_q', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(0, -1, -2, 1, -2, 1, 0, -1),
  });
  const k = quantizedTensor('qsdpa_k', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(0, 0, -1, -1, -1, 0, 0, -2),
  });
  const v = quantizedTensor('qsdpa_v', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1,
    buffer: Int8Array.of(3, -3, 0, -1, -5, 1, 2, -2),
  });
  const out = quantizedTensor('qsdpa_out', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: 0,
  });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQSDPAShader: () => 'typed-qsdpa',
      getSDPAShader: () => { throw new Error('QSDPA must not select packed F32 SDPA'); },
      getCrossSDPAShader: () => { throw new Error('QSDPA must not select F32 CrossSDPA'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qsdpa', opType: 'QSDPA', inputs: { q, k, v }, outputs: { out },
    params: { heads: 1, causal: false, scale: 0.5 },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qsdpa']);
  assert.equal(executor.pipelines.length, 1);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [2, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4, 5]);
  const dummyMask = entries.find((entry) => entry.binding === 3).resource.buffer;
  const params = entries.find((entry) => entry.binding === 5).resource.buffer;
  assert.equal(dummyMask.descriptor.size, 4);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 8)], [
    2, 2, 4, 1, 1, 0, 0,
    DataType.I8 | (DataType.I8 << 8) | (DataType.I8 << 16) | (DataType.I8 << 24),
  ]);
  assert.deepEqual([...new Int32Array(params.bytes.buffer).slice(8, 12)], [-1, -1, -1, 0]);
  assert.deepEqual([...new Float32Array(params.bytes.buffer).slice(12, 17)], [0.25, 0.25, 0.25, 0.25, 0.5]);
  assert.ok(executor.auxiliaryBuffers.has(dummyMask));
  executor.dispose();
  assert.equal(dummyMask.destroyed, true);
  assert.equal(params.destroyed, true);
});

test('WebGPU QSDPA validates dynamic I32 masks before upload and rejects overlapping output views', async () => {
  const device = mockDevice();
  const q = quantizedTensor('dynamic_qsdpa_q', [1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(0, -1, -2, 1),
  });
  const k = quantizedTensor('dynamic_qsdpa_k', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(0, 0, -1, -1, -1, 0, 0, -2),
  });
  const v = quantizedTensor('dynamic_qsdpa_v', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(3, -3, 0, -1, -5, 1, 2, -2),
  });
  const mask = tensor('dynamic_qsdpa_mask', [2], { dtype: 'int32', buffer: Int32Array.of(1, 0) });
  const out = quantizedTensor('dynamic_qsdpa_out', [1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: 0,
  });
  q.isInput = true;
  k.isInput = true;
  v.isInput = true;
  mask.isInput = true;
  const node = {
    id: 'dynamic_qsdpa', opType: 'QSDPA', inputs: { q, k, v, mask }, outputs: { out },
    params: { heads: 1, causal: false },
  };
  const executor = new GraphExecutor(device, { nodes: [node] }, {
    shaderLibrary: { getQSDPAShader: () => 'typed-qsdpa' },
  });
  await assert.doesNotReject(() => executor._buildNodePipeline(node));
  assert.doesNotThrow(() => executor._preflightQSDPAMasks({
    dynamic_qsdpa_mask: Int32Array.of(1, 0),
  }));
  assert.throws(
    () => executor._preflightQSDPAMasks({}),
    /graph-input I32 mask supplied on every execution/,
  );
  assert.throws(
    () => executor._preflightQSDPAMasks({ dynamic_qsdpa_mask: new Int8Array(8) }),
    /typed storage does not match dtype/,
  );
  executor.dispose();

  const shared = new ArrayBuffer(4);
  const aliasedQ = quantizedTensor('aliased_qsdpa_q', [1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: new Int8Array(shared),
  });
  const aliasedOut = quantizedTensor('aliased_qsdpa_out', [1, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: 0, buffer: new Int8Array(shared),
  });
  const rejecting = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejecting._buildNodePipeline({
      id: 'aliased_qsdpa', opType: 'QSDPA', inputs: { q: aliasedQ, k, v }, outputs: { out: aliasedOut },
      params: { heads: 1, causal: false },
    }),
    /output storage overlapping an input/,
  );
  rejecting.dispose();
});

test('WebGPU QMaskedMean binds packed bytes, I32 keep mask, and the 48-byte router ABI', async () => {
  const device = mockDevice();
  const input = quantizedTensor('qmaskedmean_input', [1, 3, 5], {
    dtype: 'int8', scale: 0.5, zeroPoint: -1,
    buffer: Int8Array.of(1, -3, 2, 0, 4, 100, 100, 100, 100, 100, 5, 1, -2, 3, 0),
  });
  const mask = tensor('qmaskedmean_keep', [1, 3], { dtype: 'int32', buffer: Int32Array.of(1, 0, 1) });
  const out = quantizedTensor('qmaskedmean_out', [1, 5], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 2,
  });
  input.quantization = Object.freeze(input.quantization);
  out.quantization = Object.freeze(out.quantization);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQMaskedMeanShader: () => 'typed-qmaskedmean',
      getReduceMeanShader: () => { throw new Error('QMaskedMean must not select an F32 reduction shader'); },
    },
  });
  await executor._buildNodePipeline({
    id: 'qmaskedmean', opType: 'QMaskedMean', inputs: { input, mask }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qmaskedmean']);
  assert.equal(executor.pipelines.length, 1);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3]);
  const params = paramsFrom(device, 3);
  assert.equal(params.descriptor.size, 48);
  assert.deepEqual(
    [...new Uint32Array(params.bytes.buffer).slice(0, 8)],
    [1, 3, 5, DataType.I8, DataType.U8, 0, 0, 0],
  );
  assert.deepEqual([...new Int32Array(params.bytes.buffer).slice(8, 10)], [-1, 2]);
  assert.deepEqual([...new Float32Array(params.bytes.buffer).slice(10, 12)], [0.5, 0.25]);
  executor.dispose();
  assert.equal(params.destroyed, true);
});

test('WebGPU QMaskedMean preflights dynamic masks and rejects aliased output storage', async () => {
  const device = mockDevice();
  const input = quantizedTensor('dynamic_qmaskedmean_input', [1, 2, 2], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: Int8Array.of(1, -3, 5, 1),
  });
  const mask = tensor('dynamic_qmaskedmean_mask', [1, 2], { dtype: 'int32', buffer: Int32Array.of(1, 0) });
  const out = quantizedTensor('dynamic_qmaskedmean_out', [1, 2], {
    dtype: 'int8', scale: 0.25, zeroPoint: 0,
  });
  input.quantization = Object.freeze(input.quantization);
  out.quantization = Object.freeze(out.quantization);
  mask.isInput = true;
  const node = { id: 'dynamic_qmaskedmean', opType: 'QMaskedMean', inputs: { input, mask }, outputs: { out }, params: {} };
  const executor = new GraphExecutor(device, { nodes: [node] }, {
    shaderLibrary: { getQMaskedMeanShader: () => 'typed-qmaskedmean' },
  });
  await assert.doesNotReject(() => executor._buildNodePipeline(node));
  assert.doesNotThrow(() => executor._preflightQMaskedMeanMasks({
    dynamic_qmaskedmean_mask: Int32Array.of(1, 0),
  }));
  assert.throws(() => executor._preflightQMaskedMeanMasks({}), /graph-input I32 mask supplied on every execution/);
  assert.throws(
    () => executor._preflightQMaskedMeanMasks({ dynamic_qmaskedmean_mask: new Int8Array(8) }),
    /typed storage does not match dtype/,
  );
  executor.dispose();

  const shared = new ArrayBuffer(4);
  const aliasedInput = quantizedTensor('aliased_qmaskedmean_input', [1, 2, 2], {
    dtype: 'int8', scale: 0.25, zeroPoint: -1, buffer: new Int8Array(shared),
  });
  const aliasedOutput = quantizedTensor('aliased_qmaskedmean_output', [1, 2], {
    dtype: 'int8', scale: 0.25, zeroPoint: 0, buffer: new Int8Array(shared, 0, 2),
  });
  aliasedInput.quantization = Object.freeze(aliasedInput.quantization);
  aliasedOutput.quantization = Object.freeze(aliasedOutput.quantization);
  const rejecting = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejecting._buildNodePipeline({
      id: 'aliased_qmaskedmean', opType: 'QMaskedMean',
      inputs: { input: aliasedInput, mask }, outputs: { out: aliasedOutput }, params: {},
    }),
    /QMaskedMean/,
  );
  rejecting.dispose();
});

test('WebGPU QArgMax binds packed I8/U8 logits to an unquantized I32 index ABI', async () => {
  const device = mockDevice();
  const input = quantizedTensor('qargmax_input', [2, 3, 2], {
    dtype: 'uint8', scale: 0.03125, zeroPoint: 201,
    buffer: Uint8Array.of(128, 7, 255, 7, 255, 6, 1, 4, 2, 9, 2, 8),
  });
  input.quantization = Object.freeze(input.quantization);
  const out = tensor('qargmax_out', [2, 2], { dtype: 'int32', buffer: new Int32Array(4) });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getQArgMaxShader: () => 'typed-qargmax',
      getArgMaxI32Shader: () => { throw new Error('QArgMax must not select generic F32 ArgMax'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'qargmax', opType: 'QArgMax', inputs: { input }, outputs: { out }, params: { axis: -2 },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-qargmax']);
  assert.equal(executor.pipelines.length, 1);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2]);
  const params = paramsFrom(device, 2);
  assert.equal(params.descriptor.size, 16);
  assert.deepEqual(
    [...new Uint32Array(params.bytes.buffer)],
    [2, 3, 2, DataType.U8],
  );
  executor.dispose();
  assert.equal(params.destroyed, true);
});

test('WebGPU QArgMax rejects overlapping source/output views before dispatch', async () => {
  const shared = new ArrayBuffer(8);
  const input = quantizedTensor('qargmax_alias_input', [2, 4], {
    dtype: 'int8', scale: 0.25, zeroPoint: -3, buffer: new Int8Array(shared),
  });
  input.quantization = Object.freeze(input.quantization);
  const out = tensor('qargmax_alias_out', [2], { dtype: 'int32', buffer: new Int32Array(shared) });
  const executor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'qargmax_alias', opType: 'QArgMax', inputs: { input }, outputs: { out }, params: { axis: 1 },
    }),
    /QArgMax/,
  );
  executor.dispose();
});

test('WebGPU typed Cast and DequantizeLinear reject mismatched storage contracts', async () => {
  const castInput = tensor('cast_input', [2]);
  const castOut = tensor('cast_out', [3], { dtype: 'int32' });
  const castExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => castExecutor._buildNodePipeline({
      id: 'cast_bad_shape', opType: 'Cast', inputs: { input: castInput }, outputs: { out: castOut }, params: {},
    }),
    /equal-size F32\/I32\/I8\/U8/,
  );
  castExecutor.dispose();

  const input = tensor('dequant_input', [2], { dtype: 'uint8' });
  const vectorScale = tensor('vector_scale', [2]);
  const out = tensor('dequant_out', [2]);
  const dequantExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => dequantExecutor._buildNodePipeline({
      id: 'dequant_bad_scale', opType: 'DequantizeLinear',
      inputs: { input, scale: vectorScale }, outputs: { out }, params: {},
    }),
    /F32 output, scalar F32 scale/,
  );
  dequantExecutor.dispose();
});

test('WebGPU DequantizeLinear keeps its scalar scale and output F32', async () => {
  const input = tensor('dequant_input', [2], { dtype: 'int32' });
  const scale = tensor('scale', [1]);
  const out = tensor('out', [2]);

  const integerScaleExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => integerScaleExecutor._buildNodePipeline({
      id: 'dequant_i32_scale', opType: 'DequantizeLinear',
      inputs: { input, scale: tensor('integer_scale', [1], { dtype: 'int32' }) }, outputs: { out }, params: {},
    }),
    /F32 output, scalar F32 scale/,
  );
  integerScaleExecutor.dispose();

  const integerOutputExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => integerOutputExecutor._buildNodePipeline({
      id: 'dequant_i32_output', opType: 'DequantizeLinear',
      inputs: { input, scale }, outputs: { out: tensor('integer_out', [2], { dtype: 'int32' }) }, params: {},
    }),
    /F32 output, scalar F32 scale/,
  );
  integerOutputExecutor.dispose();
});

test('WebGPU Mask uses the browser-only typed shader for I32 mask aliases', async () => {
  const device = mockDevice();
  const mask = tensor('mask', [4], { dtype: 'int32' });
  const a = tensor('a', [4]);
  const b = tensor('b', [4]);
  const out = tensor('out', [4]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getWhereTypedShader: () => 'where-typed',
      getWhereShader: () => { throw new Error('canonical Where/Mask must not use the native F32-only ABI'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'mask_i32', opType: 'Mask', inputs: { mask, a, b }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['where-typed']);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2, 3, 4]);
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 4).bytes.buffer).slice(0, 4)],
    [4, DataType.I32, 0, 0],
  );
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  executor.dispose();
});

test('WebGPU Where accepts F32 conditions and rejects non-exact canonical tensors', async () => {
  const device = mockDevice();
  const condition = tensor('condition', [2, 2]);
  const x = tensor('x', [2, 2]);
  const y = tensor('y', [2, 2]);
  const out = tensor('out', [2, 2]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getWhereTypedShader: () => 'where-typed' },
  });
  await executor._buildNodePipeline({
    id: 'where_f32', opType: 'Where', inputs: { condition, x, y }, outputs: { out }, params: {},
  });
  assert.deepEqual(device.state.shaderCodes, ['where-typed']);
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 4).bytes.buffer).slice(0, 4)],
    [4, DataType.F32, 0, 0],
  );
  executor.dispose();

  const invalidExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => invalidExecutor._buildNodePipeline({
      id: 'where_shape', opType: 'Where',
      inputs: { condition: tensor('bad_condition', [1, 4], { dtype: 'int32' }), x, y },
      outputs: { out }, params: {},
    }),
    /exact-shape same-dtype F32\/I32 operands\/output and an F32 or I32 condition/,
  );
  invalidExecutor.dispose();
});

test('WebGPU Slice uses canonical rank-1..8 metadata with normalized axes and starts', async () => {
  const device = mockDevice();
  const input = tensor('input', [2, 3, 4, 5, 6]);
  const out = tensor('out', [1, 3, 2, 5, 3]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getSliceNdShader: () => 'slice-nd',
      getSliceShader: () => { throw new Error('canonical WebGPU Slice must not use the native 4-D ABI'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'slice_rank_five', opType: 'Slice', inputs: { input }, outputs: { out },
    params: { axes: [-5, -3, -1], starts: [-1, 1, -5], steps: [1, 2, 2] },
  });

  assert.deepEqual(device.state.shaderCodes, ['slice-nd']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [2, 1, 1]);
  assert.deepEqual(device.state.bindGroups.at(-1).entries.map(({ binding }) => binding), [0, 1, 2]);
  const words = new Uint32Array(paramsFrom(device, 2).bytes.buffer);
  assert.deepEqual([...words.slice(0, 4)], [5, 90, 0, 0]);
  assert.deepEqual([...words.slice(4, 12)], [1, 3, 2, 5, 3, 1, 1, 1]);
  assert.deepEqual([...words.slice(12, 20)], [1, 0, 1, 0, 1, 0, 0, 0]);
  assert.deepEqual([...words.slice(20, 28)], [1, 1, 2, 1, 2, 1, 1, 1]);
  assert.deepEqual([...words.slice(28, 36)], [360, 120, 30, 6, 1, 1, 1, 1]);
  executor.dispose();
});

test('WebGPU Slice accepts the rank-eight upper bound', async () => {
  const device = mockDevice();
  const input = tensor('input', [1, 1, 1, 1, 1, 1, 2, 3]);
  const out = tensor('out', [1, 1, 1, 1, 1, 1, 1, 2]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getSliceNdShader: () => 'slice-nd' },
  });

  await executor._buildNodePipeline({
    id: 'slice_rank_eight', opType: 'Slice', inputs: { input }, outputs: { out },
    params: { axes: [-2, -1], starts: [-1, -3], steps: [1, 2] },
  });

  const words = new Uint32Array(paramsFrom(device, 2).bytes.buffer);
  assert.deepEqual([...words.slice(0, 4)], [8, 2, 0, 0]);
  assert.deepEqual([...words.slice(4, 12)], [1, 1, 1, 1, 1, 1, 1, 2]);
  assert.deepEqual([...words.slice(12, 20)], [0, 0, 0, 0, 0, 0, 1, 0]);
  assert.deepEqual([...words.slice(20, 28)], [1, 1, 1, 1, 1, 1, 1, 2]);
  assert.deepEqual([...words.slice(28, 36)], [6, 6, 6, 6, 6, 6, 3, 1]);
  executor.dispose();
});

test('WebGPU dispatches canonical BatchMatMul and I32 logical kernels with typed metadata', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getBatchMatMulShader: () => 'batch-matmul',
      getCompareI32Shader: () => 'compare-i32',
      getNotI32Shader: () => 'not-i32',
      getTypedClipShader: () => 'clip-typed',
      getWhereTypedShader: () => 'where-typed',
    },
  });

  const a = tensor('a', [2, 1, 2, 3]);
  const b = tensor('b', [1, 2, 3, 2]);
  const product = tensor('product', [2, 2, 2, 2]);
  await executor._buildNodePipeline({
    id: 'batch_matmul', opType: 'BatchMatMul',
    inputs: { a, b }, outputs: { out: product }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'batch-matmul');
  assert.deepEqual(executor.pipelines.at(-1).workgroupCount, [1, 1, 4]);
  const batchMetadata = new Uint32Array(
    device.state.bindGroups.at(-1).entries.find(({ binding }) => binding === 3)
      .resource.buffer.bytes.buffer,
  );
  assert.deepEqual([...batchMetadata], [
    2, 2, 3, 2, 4,
    2, 1,
    6, 0,
    0, 6,
  ]);

  const logicalA = tensor('logicalA', [2, 1, 3], { dtype: 'int32' });
  const logicalB = tensor('logicalB', [1, 2, 1], { dtype: 'int32' });
  const equal = tensor('equal', [2, 2, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'equal', opType: 'Equal',
    inputs: { a: logicalA, b: logicalB }, outputs: { out: equal }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'compare-i32');
  const compareMetadata = new Uint32Array(
    device.state.bindGroups.at(-1).entries.find(({ binding }) => binding === 3)
      .resource.buffer.bytes.buffer,
  );
  assert.deepEqual([...compareMetadata], [
    12, 3,
    6, 3, 1,
    3, 0, 1,
    0, 1, 0,
    0,
  ]);

  const inverted = tensor('inverted', [2, 2, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'not', opType: 'Not',
    inputs: { input: equal }, outputs: { out: inverted }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'not-i32');
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 2).bytes.buffer)],
    [12, 0, 0, 0],
  );

  const clipped = tensor('clipped', [2, 1, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'clip', opType: 'Clip',
    inputs: { input: logicalA }, outputs: { out: clipped },
    params: { min: -2, max: 5 },
  });
  assert.equal(device.state.shaderCodes.at(-1), 'clip-typed');
  const clipParams = paramsFrom(device, 2).bytes.buffer;
  assert.deepEqual(
    [...new Uint32Array(clipParams).slice(0, 2)],
    [6, DataType.I32],
  );
  assert.deepEqual([...new Int32Array(clipParams).slice(8, 10)], [-2, 5]);

  const condition = tensor('condition', [2, 1, 3], { dtype: 'int32' });
  const selected = tensor('selected', [2, 1, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'where_i32', opType: 'Where',
    inputs: { condition, x: clipped, y: logicalA },
    outputs: { out: selected }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'where-typed');
  assert.deepEqual(
    [...new Uint32Array(paramsFrom(device, 4).bytes.buffer)],
    [6, DataType.I32, 0, 0],
  );
  executor.dispose();
});

test('WebGPU QBatchMatMul accepts device-only tensors and binds canonical U8S8 metadata', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getQBatchMatMulShader: () => 'qbatch-matmul' },
  });
  const a = quantizedTensor('qa', [2, 1, 2, 3], {
    dtype: 'uint8',
    scale: 0.5,
    zeroPoint: 10,
  });
  const b = quantizedTensor('qb', [1, 2, 3, 2], {
    dtype: 'int8',
    scale: 0.25,
    zeroPoint: -1,
  });
  const out = quantizedTensor('qout', [2, 2, 2, 2], {
    dtype: 'uint8',
    scale: 0.125,
    zeroPoint: 100,
  });
  await executor._buildNodePipeline({
    id: 'qbatch_matmul',
    opType: 'QBatchMatMul',
    inputs: { a, b },
    outputs: { out },
    params: {},
  });

  assert.equal(device.state.shaderCodes.at(-1), 'qbatch-matmul');
  assert.deepEqual(executor.pipelines.at(-1).workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  const metadata = new Uint32Array(
    entries.find(({ binding }) => binding === 3).resource.buffer.bytes.buffer,
  );
  assert.deepEqual([...metadata], [2, 1, 6, 0, 0, 6]);
  const params = entries.find(({ binding }) => binding === 4).resource.buffer.bytes.buffer;
  assert.deepEqual(
    [...new Uint32Array(params).slice(0, 8)],
    [2, 2, 3, 2, 16, DataType.U8, DataType.I8, DataType.U8],
  );
  assert.deepEqual([...new Int32Array(params).slice(8, 11)], [10, -1, 100]);
  assert.deepEqual([...new Float32Array(params).slice(12, 15)], [0.5, 0.25, 0.125]);
  executor.dispose();
});

test('WebGPU storage-only dispatch preserves I32 through concat/shape/slice/expand/split', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getConcatCopy32Shader: () => 'concat-copy-32',
      getCopy32Shader: () => 'copy-32',
      getGeneralTransposeShader: () => 'transpose-32',
      getSliceNdShader: () => 'slice-32',
      getExpandShader: () => 'expand-32',
      getSplitShader: () => 'split-32',
    },
  });
  const a = tensor('a', [2, 2], { dtype: 'int32' });
  const b = tensor('b', [2, 1], { dtype: 'int32' });
  const joined = tensor('joined', [2, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'concat_i32', opType: 'Concat2',
    inputs: { input0: a, input1: b }, outputs: { out: joined },
    params: { axis: 1 },
  });
  assert.deepEqual(device.state.shaderCodes, ['concat-copy-32']);
  assert.equal(executor.pipelines.length, 2);

  const reshaped = tensor('reshaped', [3, 2], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'reshape_i32', opType: 'Reshape',
    inputs: { input: joined }, outputs: { out: reshaped }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'copy-32');

  const transposed = tensor('transposed', [2, 3], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'transpose_i32', opType: 'Transpose',
    inputs: { input: reshaped }, outputs: { out: transposed },
    params: { perm: [1, 0] },
  });
  assert.equal(device.state.shaderCodes.at(-1), 'transpose-32');

  const sliced = tensor('sliced', [2, 2], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'slice_i32', opType: 'Slice',
    inputs: { input: transposed }, outputs: { out: sliced },
    params: { axes: [1], starts: [1], steps: [1] },
  });
  assert.equal(device.state.shaderCodes.at(-1), 'slice-32');

  const shaped = tensor('shaped', [2, 1, 2], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'shape_i32', opType: 'Reshape',
    inputs: { input: sliced }, outputs: { out: shaped }, params: {},
  });
  const expanded = tensor('expanded', [2, 2, 2], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'expand_i32', opType: 'Expand',
    inputs: { input: shaped }, outputs: { out: expanded }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'expand-32');

  const part0 = tensor('part0', [2, 1, 2], { dtype: 'int32' });
  const part1 = tensor('part1', [2, 1, 2], { dtype: 'int32' });
  await executor._buildNodePipeline({
    id: 'split_i32', opType: 'Split',
    inputs: { input: expanded }, outputs: { part0, part1 },
    params: { axis: 1 },
  });
  assert.equal(device.state.shaderCodes.at(-1), 'split-32');
  assert.equal(executor.pipelines.filter(({ nodeName }) =>
    String(nodeName).startsWith('split_i32_split')).length, 2);
  executor.dispose();
});

test('WebGPU Expand uses packed descriptor-preserving I8/U8 dispatch', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getTypedExpandShader: () => 'expand-typed' },
  });
  const input = quantizedTensor('expand_input', [1, 1, 64], {
    dtype: 'uint8', scale: 0.125, zeroPoint: 127,
  });
  const out = quantizedTensor('expand_out', [1, 402, 64], {
    dtype: 'uint8', scale: 0.125, zeroPoint: 127,
  });
  await executor._buildNodePipeline({
    id: 'expand_u8', opType: 'Expand',
    inputs: { input }, outputs: { out }, params: {},
  });
  assert.equal(device.state.shaderCodes.at(-1), 'expand-typed');
  assert.deepEqual(executor.pipelines.at(-1).workgroupCount, [101, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  const metadata = new Uint32Array(
    entries.find(({ binding }) => binding === 2).resource.buffer.bytes.buffer,
  );
  assert.deepEqual([...metadata.slice(0, 7)], [3, 3, 0, 25728, 1, 1, 64]);

  const mismatch = quantizedTensor('expand_mismatch', [1, 402, 64], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 127,
  });
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'expand_bad_affine', opType: 'Expand',
      inputs: { input }, outputs: { out: mismatch }, params: {},
    }),
    /must preserve its I8\/U8 affine descriptor/,
  );
  executor.dispose();
});

test('WebGPU Transpose uses packed descriptor-preserving I8/U8 dispatch', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getTypedTransposeShader: () => 'transpose-typed' },
  });
  const input = quantizedTensor('nchw', [1, 2, 7, 19], {
    dtype: 'uint8', scale: 0.125, zeroPoint: 123,
  });
  const out = quantizedTensor('nhwc', [1, 7, 19, 2], {
    dtype: 'uint8', scale: 0.125, zeroPoint: 123,
  });
  await executor._buildNodePipeline({
    id: 'transpose_u8',
    opType: 'Transpose',
    inputs: { input },
    outputs: { out },
    params: { perm: [0, 2, 3, 1] },
  });
  assert.equal(device.state.shaderCodes.at(-1), 'transpose-typed');
  assert.deepEqual(executor.pipelines.at(-1).workgroupCount, [2, 1, 1]);
  const entries = device.state.bindGroups.at(-1).entries;
  const metadata = new Uint32Array(
    entries.find(({ binding }) => binding === 2).resource.buffer.bytes.buffer,
  );
  assert.deepEqual(
    [...metadata],
    [266, 4, 266, 38, 2, 1, 266, 19, 1, 133],
  );

  const mismatch = quantizedTensor('mismatch', [1, 7, 19, 2], {
    dtype: 'uint8', scale: 0.25, zeroPoint: 123,
  });
  await assert.rejects(
    executor._buildNodePipeline({
      id: 'transpose_bad_descriptor',
      opType: 'Transpose',
      inputs: { input },
      outputs: { out: mismatch },
      params: { perm: [0, 2, 3, 1] },
    }),
    /identical per-tensor metadata/,
  );
  executor.dispose();
});

test('WebGPU Split preserves declared order with twelve outputs', async () => {
  const device = mockDevice();
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getSplitShader: () => 'split-declared-order' },
  });
  const input = tensor('input', [2, 12]);
  const outputs = {};
  for (let index = 0; index < 12; index++) {
    outputs[`out${index}`] = tensor(`slice${index}`, [2, 1]);
  }
  const firstBindGroup = device.state.bindGroups.length;
  await executor._buildNodePipeline({
    id: 'split_many', opType: 'Split',
    inputs: { input }, outputs, params: { axis: 1 },
  });

  const groups = device.state.bindGroups.slice(firstBindGroup);
  assert.equal(groups.length, 12);
  assert.deepEqual(
    groups.map(({ entries }) =>
      entries.find(({ binding }) => binding === 1).resource.buffer.tensor),
    Array.from({ length: 12 }, (_, index) => `slice${index}`),
  );
  assert.deepEqual(
    groups.map(({ entries }) =>
      new Uint32Array(entries.find(({ binding }) => binding === 2)
        .resource.buffer.bytes.buffer)[4]),
    Array.from({ length: 12 }, (_, index) => index),
  );
  executor.dispose();
});

test('WebGPU Softmax shaders seed maxima from finite row data below the old sentinel', async () => {
  const softmaxSource = await readFile(
    new URL('../shaders/inference/softmax.wgsl', import.meta.url),
    'utf8',
  );
  const logSoftmaxSource = await readFile(
    new URL('../shaders/inference/logSoftmax.wgsl', import.meta.url),
    'utf8',
  );
  for (const [opType, source] of [
    ['Softmax', softmaxSource],
    ['LogSoftmax', logSoftmaxSource],
  ]) {
    assert.match(source, /var max_val = input\[offset\];/);
    assert.match(source, /for \(var j = 1u;/);
    assert.doesNotMatch(source, /-100000\.0/);

    const device = mockDevice();
    const executor = new GraphExecutor(device, { nodes: [] }, {
      shaderLibrary: {
        getSoftmaxShader: () => softmaxSource,
        getLogSoftmaxShader: () => logSoftmaxSource,
      },
    });
    const input = tensor(`${opType}_input`, [1, 3], {
      buffer: Float32Array.of(-200003, -200002, -200001),
    });
    const out = tensor(`${opType}_out`, [1, 3]);
    const inputBuffer = device.createBuffer({
      size: input.sizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = device.createBuffer({
      size: out.sizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    new Float32Array(inputBuffer.bytes.buffer).set(input.buffer);
    executor.gpuBuffers.set(input.name, inputBuffer);
    executor.gpuBuffers.set(out.name, outputBuffer);
    await executor._buildNodePipeline({
      id: `${opType}_extreme_negative`,
      opType,
      inputs: { input },
      outputs: { out },
      params: { axis: -1 },
    });
    assert.equal(device.state.shaderCodes.at(-1), source);
    assert.deepEqual(
      [...new Uint32Array(paramsFrom(device, 2).bytes.buffer)],
      [1, 3, 0, 0],
    );
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    const compiled = executor.pipelines.at(-1);
    pass.setPipeline(compiled.pipeline);
    pass.setBindGroup(0, compiled.bindGroup);
    pass.dispatchWorkgroups(...compiled.workgroupCount);
    pass.end();
    device.queue.submit([commandEncoder.finish()]);

    const actual = [...new Float32Array(outputBuffer.bytes.buffer)];
    assert.ok(actual.every(Number.isFinite), `${opType} output must stay finite`);
    const exponentials = [Math.exp(-2), Math.exp(-1), 1];
    const sum = exponentials.reduce((left, right) => left + right, 0);
    const expected = opType === 'Softmax'
      ? exponentials.map((value) => value / sum)
      : [-2 - Math.log(sum), -1 - Math.log(sum), -Math.log(sum)];
    for (let index = 0; index < expected.length; index++) {
      assert.ok(
        Math.abs(actual[index] - expected[index]) <= 1e-6,
        `${opType}[${index}] got ${actual[index]}, expected ${expected[index]}`,
      );
    }
    executor.dispose();
  }
});

test('WebGPU LayerNorm computes variance from centered values in a second pass', async () => {
  const source = await readFile(
    new URL('../shaders/inference/layerNorm.wgsl', import.meta.url),
    'utf8',
  );
  assert.match(source, /let mean = sum \/ f32\(d_model\);/);
  assert.match(source, /var variance_sum : f32 = 0\.0;/);
  assert.match(source, /let centered = input\[offset \+ i\] - mean;/);
  assert.match(source, /variance_sum = variance_sum \+ \(centered \* centered\);/);
  assert.match(source, /let variance = variance_sum \/ f32\(d_model\);/);
  assert.doesNotMatch(source, /sq_sum|mean \* mean/);
});

test('WebGPU Slice rejects non-positive steps and output selections beyond the input', async () => {
  const input = tensor('input', [4]);
  const output = tensor('out', [2]);
  const executor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'slice_zero_step', opType: 'Slice', inputs: { input }, outputs: { out: output },
      params: { axes: [0], starts: [0], steps: [0] },
    }),
    /positive integer steps/,
  );
  await assert.rejects(
    () => executor._buildNodePipeline({
      id: 'slice_out_of_bounds', opType: 'Slice', inputs: { input }, outputs: { out: output },
      params: { axes: [0], starts: [3], steps: [1] },
    }),
    /output shape exceeds its input selection/,
  );
  executor.dispose();
});

test('WebGPU typed shape-only nodes use a physical packed-byte copy', async () => {
  const device = mockDevice();
  const input = quantizedTensor('shape_input', [1, 3], {
    dtype: 'int8', zeroPoint: -3, buffer: Int8Array.of(-3, 4, 7),
  });
  const out = quantizedTensor('shape_out', [3, 1], { dtype: 'int8', zeroPoint: -3 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getTypedCopyShader: () => 'typed-copy',
      getCopyShader: () => { throw new Error('I8 reshape must not use the F32 copy shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'reshape_i8', opType: 'Reshape', inputs: { input }, outputs: { out }, params: {},
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-copy']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 2).bytes.buffer).slice(0, 4)], [3, 0, 0, 0]);
  executor.dispose();
});

test('WebGPU typed concat preserves raw bytes and rejects descriptor changes', async () => {
  const device = mockDevice();
  const left = quantizedTensor('left', [1, 1, 2, 1], { dtype: 'uint8', scale: 0.125, zeroPoint: 17 });
  const right = quantizedTensor('right', [1, 1, 1, 1], { dtype: 'uint8', scale: 0.125, zeroPoint: 17 });
  const out = quantizedTensor('concat_out', [1, 1, 3, 1], { dtype: 'uint8', scale: 0.125, zeroPoint: 17 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getTypedConcatCopyShader: () => 'typed-concat',
      getConcatCopyShader: () => { throw new Error('I8 concat must not use the F32 concat shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'concat_i8', opType: 'Concat', inputs: { input0: left, input1: right }, outputs: { out }, params: { axis: 2 },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-concat']);
  assert.equal(executor.pipelines.length, 2);
  assert.deepEqual(executor.pipelines.map((pipeline) => pipeline.workgroupCount), [[1, 1, 1], [1, 1, 1]]);
  const first = new Uint32Array(device.state.bindGroups[0].entries.find((entry) => entry.binding === 2).resource.buffer.bytes.buffer);
  const second = new Uint32Array(device.state.bindGroups[1].entries.find((entry) => entry.binding === 2).resource.buffer.bytes.buffer);
  assert.deepEqual([...first.slice(0, 5)], [2, 0, 2, 3, 1]);
  assert.deepEqual([...second.slice(0, 5)], [1, 2, 1, 3, 1]);
  executor.dispose();

  const mismatch = quantizedTensor('concat_mismatch', [1, 1, 3, 1], { dtype: 'uint8', scale: 0.25, zeroPoint: 17 });
  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, {
    shaderLibrary: { getTypedConcatCopyShader: () => 'typed-concat' },
  });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'concat_bad_scale', opType: 'Concat', inputs: { input0: left, input1: right },
      outputs: { out: mismatch }, params: { axis: 2 },
    }),
    /identical quantization descriptors/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU F32 concat forwards the fused sigmoid flag to every input copy', async () => {
  const device = mockDevice();
  const left = tensor('left', [1, 2, 2]);
  const right = tensor('right', [1, 1, 2]);
  const out = tensor('concat_out', [1, 3, 2]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getConcatCopyShader: () => 'f32-concat' },
  });

  await executor._buildNodePipeline({
    id: 'concat_sigmoid', opType: 'Concat', inputs: { input0: left, input1: right },
    outputs: { out }, params: { axis: 1, sigmoid: 1 },
  });

  assert.deepEqual(device.state.shaderCodes, ['f32-concat']);
  assert.equal(executor.pipelines.length, 2);
  const parameters = device.state.bindGroups.map((bindGroup) =>
    new Uint32Array(bindGroup.entries.find((entry) => entry.binding === 2).resource.buffer.bytes.buffer));
  assert.deepEqual([...parameters[0]], [4, 0, 2, 3, 2, 1, 0, 0]);
  assert.deepEqual([...parameters[1]], [2, 2, 1, 3, 2, 1, 0, 0]);
  executor.dispose();
});

test('WebGPU F32 Add compiles only its supported fused ReLU modes', async () => {
  const device = mockDevice();
  const a = tensor('a', [1, 4]);
  const b = tensor('b', [4]);
  const out = tensor('out', [1, 4]);
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getBroadcastBinaryShader: (operation) => operation,
    },
  });

  await executor._buildNodePipeline({
    id: 'add_relu6', opType: 'Add', inputs: { a, b }, outputs: { out }, params: { relu: 2 },
  });
  assert.match(device.state.shaderCodes[0], /out_val = av \+ bv/);
  assert.match(device.state.shaderCodes[0], /out_val < 0\.0/);
  assert.match(device.state.shaderCodes[0], /out_val > 6\.0/);

  for (const relu of [-1, 3, 1.5]) {
    await assert.rejects(
      () => executor._buildNodePipeline({
        id: `add_bad_relu_${relu}`, opType: 'Add', inputs: { a, b }, outputs: { out }, params: { relu },
      }),
      /supports relu values 0 \(none\), 1 \(ReLU\), or 2 \(ReLU6\)/,
    );
  }
  executor.dispose();
});

test('WebGPU typed MaxPool2D dispatches signed packed-byte pooling', async () => {
  const device = mockDevice();
  const input = quantizedTensor('pool_input', [1, 3, 3, 1], { dtype: 'int8', scale: 0.5, zeroPoint: -2 });
  const out = quantizedTensor('pool_out', [1, 2, 2, 1], { dtype: 'int8', scale: 0.5, zeroPoint: -2 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getTypedMaxPool2DShader: () => 'typed-maxpool',
      getMaxPool2DShader: () => { throw new Error('I8 MaxPool must not use the F32 shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'pool_i8', opType: 'MaxPool2D', inputs: { input }, outputs: { out },
    params: { kernel: [2, 2], stride: [2, 2], padding: [0, 0], pads: [0, 0, 1, 1] },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-maxpool']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 2).bytes.buffer).slice(0, 13)], [
    1, 3, 3, 1, 2, 2, 2, 2, 2, 2, 0, 0, DataType.I8,
  ]);
  executor.dispose();

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'pool_i8_dilated', opType: 'MaxPool2D', inputs: { input }, outputs: { out },
      params: { kernel: [2, 2], stride: [2, 2], dilation: [2, 1] },
    }),
    /only with unit dilation/,
  );
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'pool_i8_ceil', opType: 'MaxPool2D', inputs: { input }, outputs: { out },
      params: { kernel: [2, 2], stride: [2, 2], ceil_mode: true },
    }),
    /does not support ceil_mode/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU typed Resize uses nearest-neighbor packed-byte forwarding only', async () => {
  const device = mockDevice();
  const input = quantizedTensor('resize_input', [1, 2, 2, 1], { dtype: 'uint8', scale: 0.125, zeroPoint: 9 });
  const out = quantizedTensor('resize_out', [1, 4, 4, 1], { dtype: 'uint8', scale: 0.125, zeroPoint: 9 });
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: {
      getTypedResizeNearestShader: () => 'typed-resize-nearest',
      getResizeShader: () => { throw new Error('U8 nearest resize must not use the F32 shader'); },
    },
  });

  await executor._buildNodePipeline({
    id: 'resize_u8', opType: 'Resize', inputs: { input }, outputs: { out },
    params: { mode: 'nearest', coordinate_transformation_mode: 'asymmetric' },
  });

  assert.deepEqual(device.state.shaderCodes, ['typed-resize-nearest']);
  assert.deepEqual(executor.pipelines[0].workgroupCount, [1, 1, 1]);
  assert.deepEqual([...new Uint32Array(paramsFrom(device, 2).bytes.buffer).slice(0, 8)], [1, 2, 2, 1, 4, 4, 0, 0]);
  executor.dispose();

  const rejectingExecutor = new GraphExecutor(mockDevice(), { nodes: [] }, { shaderLibrary: {} });
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'resize_i8_linear', opType: 'Resize', inputs: { input }, outputs: { out }, params: { mode: 'linear' },
    }),
    /only for nearest-neighbor mode/,
  );
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'resize_i8_half_pixel', opType: 'Resize', inputs: { input }, outputs: { out },
      params: { mode: 'nearest', coordinate_transformation_mode: 'half_pixel' },
    }),
    /coordinate_transformation_mode "asymmetric"/,
  );
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'resize_i8_noncanonical', opType: 'Resize', inputs: { input }, outputs: { out },
      params: { mode: 'nearest', coordinate_transform_mode: 'asymmetric' },
    }),
    /unsupported 'coordinate_transform_mode'.*coordinate_transformation_mode/,
  );
  await assert.rejects(
    () => rejectingExecutor._buildNodePipeline({
      id: 'resize_i8_rounding', opType: 'Resize', inputs: { input }, outputs: { out },
      params: { mode: 'nearest', nearest_mode: 'round_prefer_floor' },
    }),
    /nearest_mode "floor"/,
  );
  rejectingExecutor.dispose();
});

test('WebGPU snapshots every declared output, including input, weight, and Dropout aliases', async () => {
  const device = mockDevice();
  const graph = new Graph();
  const input = graph.addInput('input', [1], 'float32');
  const weight = graph.addWeight('weight', [1], 'float32', Float32Array.of(2));
  const { out: dropout } = graph.addOp('Dropout', { input }, { out: [1] });
  graph.setOutputs(input, weight, dropout);

  const executor = new GraphExecutor(device, graph, { shaderLibrary: {} });
  await executor.compile();

  const inputBuffer = executor.gpuBuffers.get(input.name);
  const weightBuffer = executor.gpuBuffers.get(weight.name);
  const dropoutBuffer = executor.gpuBuffers.get(dropout.name);
  assert.ok(inputBuffer.descriptor.usage & GPUBufferUsage.COPY_SRC);
  assert.ok(weightBuffer.descriptor.usage & GPUBufferUsage.COPY_SRC);
  assert.equal(dropoutBuffer, inputBuffer, 'inference Dropout should alias its input');

  device.queue.writeBuffer(inputBuffer, 0, Float32Array.of(3));
  const snapshots = executor.snapshotOutputs();
  assert.deepEqual([...snapshots.keys()], ['input', 'weight', dropout.name]);
  assert.notEqual(snapshots.get('input').deviceBuffer, inputBuffer);
  assert.notEqual(snapshots.get('weight').deviceBuffer, weightBuffer);
  assert.notEqual(snapshots.get(dropout.name).deviceBuffer, dropoutBuffer);

  device.queue.writeBuffer(inputBuffer, 0, Float32Array.of(9));
  device.queue.writeBuffer(weightBuffer, 0, Float32Array.of(8));
  assert.equal(new Float32Array(snapshots.get('input').deviceBuffer.bytes.buffer)[0], 3);
  assert.equal(new Float32Array(snapshots.get('weight').deviceBuffer.bytes.buffer)[0], 2);
  assert.equal(new Float32Array(snapshots.get(dropout.name).deviceBuffer.bytes.buffer)[0], 3);

  executor.dispose();
  for (const snapshot of snapshots.values()) {
    assert.equal(snapshot.deviceBuffer.destroyed, undefined,
      'executor disposal must not invalidate result-owned buffers');
    snapshot.deviceBuffer.destroy();
    assert.equal(snapshot.deviceBuffer.destroyed, true);
  }
});
