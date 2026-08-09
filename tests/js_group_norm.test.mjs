import test from 'node:test';
import assert from 'node:assert/strict';

import { TrainingGraph as Graph } from '../ts/training/TrainingGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { CPUAutograd } from '../ts/training/CPUAutograd.js';
import { GraphExecutor } from '../ts/backends/GraphExecutor.js';
import { WebGPUAutograd } from '../ts/training/WebGPUAutograd.js';

globalThis.GPUBufferUsage ??= Object.freeze({
  MAP_READ: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  STORAGE: 8,
  UNIFORM: 16,
});

function addWeight(graph, name, shape, values) {
  const tensor = graph.addWeight(name, shape);
  tensor.buffer = Float32Array.from(values);
  return tensor;
}

function referenceGroupNorm(values, shape, weight, bias, groups, eps) {
  const [batch, height, width, channels] = shape;
  const spatialSize = height * width;
  const sampleStride = spatialSize * channels;
  const channelsPerGroup = channels / groups;
  const groupSize = spatialSize * channelsPerGroup;
  const result = new Float32Array(values.length);
  for (let n = 0; n < batch; n++) {
    for (let group = 0; group < groups; group++) {
      const firstChannel = group * channelsPerGroup;
      const indices = [];
      for (let spatial = 0; spatial < spatialSize; spatial++) {
        for (let local = 0; local < channelsPerGroup; local++) {
          indices.push(n * sampleStride + spatial * channels + firstChannel + local);
        }
      }
      const mean = indices.reduce((sum, index) => sum + values[index], 0) / groupSize;
      const variance = indices.reduce((sum, index) => sum + (values[index] - mean) ** 2, 0) / groupSize;
      const invStd = 1 / Math.sqrt(variance + eps);
      for (const index of indices) {
        const channel = index % channels;
        result[index] = (values[index] - mean) * invStd * weight[channel] + bias[channel];
      }
    }
  }
  return result;
}

function close(actual, expected, label, absolute = 3e-4, relative = 3e-3) {
  const tolerance = absolute + relative * Math.max(Math.abs(actual), Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${label}: actual=${actual}, expected=${expected}, tolerance=${tolerance}`);
}

async function crossEntropy(graph, engine, targets) {
  await engine.execute({});
  const logits = graph.getTensor(graph.outputNames[0]);
  const classes = logits.shape.at(-1);
  let loss = 0;
  for (let row = 0; row < targets.length; row++) {
    const offset = row * classes;
    let maximum = -Infinity;
    for (let col = 0; col < classes; col++) maximum = Math.max(maximum, logits.buffer[offset + col]);
    let denominator = 0;
    for (let col = 0; col < classes; col++) denominator += Math.exp(logits.buffer[offset + col] - maximum);
    loss += maximum + Math.log(denominator) - logits.buffer[offset + targets[row]];
  }
  return loss / targets.length;
}

async function numericGradient(graph, engine, tensorName, index, targets, epsilon = 1e-3) {
  const tensor = graph.getTensor(tensorName);
  const original = tensor.buffer[index];
  tensor.buffer[index] = original + epsilon;
  const positive = await crossEntropy(graph, engine, targets);
  tensor.buffer[index] = original - epsilon;
  const negative = await crossEntropy(graph, engine, targets);
  tensor.buffer[index] = original;
  return (positive - negative) / (2 * epsilon);
}

test('CPU GroupNorm normalizes NHWC values independently per sample and group', async () => {
  const graph = new Graph();
  const shape = [2, 1, 2, 4];
  const input = graph.addInput('input', shape);
  const weightValues = [1.2, 0.7, -0.5, 1.5];
  const biasValues = [0.1, -0.2, 0.3, 0.4];
  const weight = addWeight(graph, 'weight', [4], weightValues);
  const bias = addWeight(graph, 'bias', [4], biasValues);
  const { out } = graph.addOp('GroupNorm', { input, weight, bias }, { out: shape }, {
    num_groups: 2,
    eps: 0.05,
  });
  graph.setOutputs([out.name]);

  const values = Float32Array.from([
    1, 2, 3, 4, 5, 6, 7, 8,
    -2, 1, 4, -1, 2, 3, -3, 5,
  ]);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await engine.execute({ input: values });

  const expected = referenceGroupNorm(values, shape, weightValues, biasValues, 2, 0.05);
  for (let index = 0; index < expected.length; index++) {
    close(out.buffer[index], expected[index], `forward[${index}]`, 2e-6, 2e-6);
  }
});

test('CPU GroupNorm backward matches finite differences for input and affine parameters', async () => {
  const graph = new Graph();
  const shape = [1, 1, 2, 4];
  const input = addWeight(graph, 'input', shape, [-0.8, 0.1, 0.5, 1.2, 0.7, -0.3, 1.5, -0.4]);
  const weight = addWeight(graph, 'weight', [4], [1.1, 0.8, -0.6, 1.3]);
  const bias = addWeight(graph, 'bias', [4], [0.2, -0.1, 0.3, -0.2]);
  const { out } = graph.addOp('GroupNorm', { input, weight, bias }, { out: shape }, {
    num_groups: 2,
    eps: 0.07,
  });
  graph.setOutputs([out.name]);
  const targets = [1, 3];

  const analytic = await CPUAutograd.trainStep(graph, {
    targets,
    trainableTensors: ['input', 'weight', 'bias'],
    updateMode: 'sgd',
    optimizer: { learningRate: 0 },
  });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  for (const [name, indices] of [['input', [0, 3, 6]], ['weight', [0, 2]], ['bias', [1, 3]]]) {
    for (const index of indices) {
      const numeric = await numericGradient(graph, engine, name, index, targets);
      close(analytic.gradients.get(name)[index], numeric, `${name}[${index}]`);
    }
  }
});

test('CPU GroupNorm rejects invalid channel groups', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 1, 1, 4]);
  const weight = addWeight(graph, 'weight', [4], [1, 1, 1, 1]);
  const bias = addWeight(graph, 'bias', [4], [0, 0, 0, 0]);
  graph.addOp('GroupNorm', { input, weight, bias }, { out: [1, 1, 1, 4] }, { num_groups: 3 });
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await assert.rejects(() => engine.execute({ input: new Float32Array(4) }), /positive divisor of 4 channels/);
});

function tensor(name, shape, { dtype = 'float32' } = {}) {
  return {
    name,
    shape,
    dtype,
    sizeBytes: shape.reduce((count, dimension) => count * dimension, 1) * 4,
  };
}

function mockDevice() {
  const state = { bindGroups: [] };
  return {
    state,
    createBuffer(descriptor) {
      return { descriptor, bytes: new Uint8Array(descriptor.size), destroy() {} };
    },
    createShaderModule() { return {}; },
    async createComputePipelineAsync() { return { getBindGroupLayout() { return {}; } }; },
    createBindGroup(descriptor) { state.bindGroups.push(descriptor); return descriptor; },
    queue: {
      writeBuffer(destination, offset, source, sourceOffset = 0, size = undefined) {
        const sourceBuffer = source instanceof ArrayBuffer ? source : source.buffer;
        const byteOffset = (source instanceof ArrayBuffer ? 0 : source.byteOffset) + sourceOffset;
        const byteLength = size ?? (source instanceof ArrayBuffer ? source.byteLength : source.byteLength) - sourceOffset;
        destination.bytes.set(new Uint8Array(sourceBuffer, byteOffset, byteLength), offset);
      },
    },
  };
}

class DispatchRecorder extends WebGPUAutograd {
  async _dispatch(shaderName, entryPoint, entries, workgroupCount, resources = []) {
    return { shaderName, entryPoint, entries, workgroupCount, resources };
  }
}

test('WebGPU GroupNorm forward encodes NHWC/group parameters and affine bindings', async () => {
  const device = mockDevice();
  const input = tensor('input', [3, 2, 5, 8]);
  const weight = tensor('weight', [8]);
  const bias = tensor('bias', [8]);
  const out = tensor('out', [3, 2, 5, 8]);
  const node = {
    id: 'group_norm',
    opType: 'GroupNorm',
    inputs: { input, weight, bias },
    outputs: { out },
    params: { num_groups: 4, eps: 0.025 },
  };
  const executor = new GraphExecutor(device, { nodes: [] }, {
    shaderLibrary: { getGroupNormShader: () => 'group-norm' },
  });
  for (const value of [input, weight, bias, out]) {
    executor.gpuBuffers.set(value.name, { tensor: value.name });
  }

  await executor._buildNodePipeline(node);
  const pipeline = executor.pipelines[0];
  assert.deepEqual(pipeline.workgroupCount, [1, 1, 1]);
  const entries = device.state.bindGroups[0].entries;
  assert.deepEqual(entries.map(({ binding }) => binding), [0, 1, 2, 3, 4]);
  const params = entries.find(({ binding }) => binding === 4).resource.buffer;
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 6)], [3, 2, 5, 8, 4, 1]);
  close(new Float32Array(params.bytes.buffer)[6], 0.025, 'forward eps', 1e-8, 1e-6);
  executor.dispose();
});

test('WebGPU GroupNorm backward emits input and affine dispatches with shared parameters', async () => {
  const device = mockDevice();
  const input = tensor('input', [2, 3, 5, 8]);
  const weight = tensor('weight', [8]);
  const bias = tensor('bias', [8]);
  const out = tensor('out', [2, 3, 5, 8]);
  const node = {
    id: 'group_norm',
    opType: 'GroupNorm',
    inputs: { input, weight, bias },
    outputs: { out },
    params: { num_groups: 4, eps: 0.015 },
  };
  const tensors = [input, weight, bias, out];
  const gpuBuffers = new Map(tensors.map((value) => [value.name, { tensor: value.name }]));
  const graph = { nodes: [node], getTensor: (name) => tensors.find((value) => value.name === name) };
  const trainer = new DispatchRecorder(device, graph, { gpuBuffers });
  trainer.gradientBuffers.set(out.name, { tensor: 'grad_out' });

  const dispatches = await trainer._buildBackwardDispatches();
  assert.deepEqual(dispatches.map(({ shaderName, entryPoint, workgroupCount }) =>
    [shaderName, entryPoint, workgroupCount]), [
    ['groupNormBackward', 'input_main', [1, 1, 1]],
    ['groupNormBackward', 'param_main', [1, 1, 1]],
  ]);
  assert.deepEqual(dispatches[0].entries.map(([binding]) => binding), [0, 1, 2, 3, 6]);
  assert.deepEqual(dispatches[1].entries.map(([binding]) => binding), [0, 2, 4, 5, 6]);
  const params = dispatches[0].entries.find(([binding]) => binding === 6)[1];
  assert.equal(dispatches[1].entries.find(([binding]) => binding === 6)[1], params);
  assert.deepEqual([...new Uint32Array(params.bytes.buffer).slice(0, 6)], [2, 3, 5, 8, 4, 1]);
  close(new Float32Array(params.bytes.buffer)[6], 0.015, 'backward eps', 1e-8, 1e-6);
  trainer.dispose();
});
