import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { _cpuQGroupNorm } from '../ts/ops/qGroupNorm.js';

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function elements(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function qGroupNormGraph({
  inputShape = [1, 2, 2, 4],
  inputDtype = 'int8',
  inputValues = [-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  gamma = [1, -0.75, 0.5, 1.25],
  beta = [0.25, -0.5, 0.75, -0.25],
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
  numGroups = 2,
  eps = 1e-5,
  dataLayout,
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: byteStorage(inputDtype, inputValues), quantization: inputQuantization,
  });
  const weight = graph.addWeight('weight', [inputShape[3]], 'float32', {
    buffer: Float32Array.from(gamma),
  });
  const bias = graph.addWeight('bias', [inputShape[3]], 'float32', {
    buffer: Float32Array.from(beta),
  });
  const params = { num_groups: numGroups, eps };
  if (dataLayout !== undefined) params.data_layout = dataLayout;
  const { out } = graph.addOp('QGroupNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype,
      buffer: byteStorage(outputDtype, new Array(elements(inputShape)).fill(0)),
      quantization: outputQuantization,
    },
  }, params);
  return { graph, node: graph.nodes[0], input, weight, bias, out };
}

function roundEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

// Kept independent from the portable operator: this is a straightforward
// mathematical reference over logical NHWC bytes, not a call-through helper.
function referenceQGroupNorm({
  inputShape, inputValues, inputQuantization, gamma, beta,
  outputDtype, outputQuantization, numGroups, eps,
}) {
  const [batch, height, width, channels] = inputShape;
  const channelsPerGroup = channels / numGroups;
  const area = height * width;
  const valuesPerGroup = area * channelsPerGroup;
  const sampleStride = area * channels;
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const minimum = outputDtype === 'int8' ? -128 : 0;
  const maximum = outputDtype === 'int8' ? 127 : 255;
  const result = [];

  for (let sample = 0; sample < batch; sample++) {
    const sampleOffset = sample * sampleStride;
    for (let group = 0; group < numGroups; group++) {
      const firstChannel = group * channelsPerGroup;
      let meanRaw = Math.fround(0);
      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const raw = Math.fround(inputValues[offset + localChannel] - inputQuantization.zero_point);
          meanRaw = Math.fround(meanRaw + raw);
        }
      }
      meanRaw = Math.fround(meanRaw / valuesPerGroup);

      let varianceRaw = Math.fround(0);
      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const raw = Math.fround(inputValues[offset + localChannel] - inputQuantization.zero_point);
          const centered = Math.fround(raw - meanRaw);
          varianceRaw = Math.fround(varianceRaw + Math.fround(centered * centered));
        }
      }
      varianceRaw = Math.fround(varianceRaw / valuesPerGroup);
      const variance = Math.fround(Math.fround(Math.max(0, varianceRaw) * inputScale) * inputScale);
      const invStd = Math.fround(1 / Math.sqrt(Math.fround(
        variance + Math.fround(eps),
      )));

      for (let spatial = 0; spatial < area; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const channel = firstChannel + localChannel;
          const raw = Math.fround(inputValues[offset + localChannel] - inputQuantization.zero_point);
          const normalized = Math.fround(Math.fround(
            Math.fround(raw - meanRaw) * inputScale,
          ) * invStd);
          const value = Math.fround(Math.fround(normalized * Math.fround(gamma[channel])) +
            Math.fround(beta[channel]));
          const transformed = Math.fround(Math.fround(value / outputScale) +
            outputQuantization.zero_point);
          let quantized;
          if (Number.isNaN(transformed)) quantized = outputQuantization.zero_point;
          else if (transformed <= minimum) quantized = minimum;
          else if (transformed >= maximum) quantized = maximum;
          else quantized = roundEven(transformed);
          result[offset + localChannel] = quantized;
        }
      }
    }
  }
  return result;
}

function packLogicalBytes(values) {
  const words = new Uint32Array(Math.ceil(values.length / 4));
  for (let index = 0; index < values.length; index++) {
    words[index >>> 2] |= (values[index] & 255) << ((index & 3) * 8);
  }
  return words;
}

test('QGroupNorm computes canonical signed NHWC groups without an F32 activation output', () => {
  const config = {};
  const { node, out } = qGroupNormGraph(config);
  _cpuQGroupNorm(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], referenceQGroupNorm({
    inputShape: [1, 2, 2, 4],
    inputValues: [-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    gamma: [1, -0.75, 0.5, 1.25], beta: [0.25, -0.5, 0.75, -0.25],
    outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    numGroups: 2, eps: 1e-5,
  }));
});

test('QGroupNorm supports every I8/U8 activation input/output pairing', () => {
  const centeredValues = [-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2];
  const inputs = [
    {
      dtype: 'int8', values: centeredValues,
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
    {
      dtype: 'uint8', values: centeredValues.map((value) => value + 128),
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    },
  ];
  const outputs = [
    { dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -7 } },
    { dtype: 'uint8', quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 123 } },
  ];
  const gamma = [1, -0.75, 0.5, 1.25];
  const beta = [0.25, -0.5, 0.75, -0.25];

  for (const input of inputs) {
    for (const output of outputs) {
      const { node, out } = qGroupNormGraph({
        inputDtype: input.dtype, inputValues: input.values, inputQuantization: input.quantization,
        gamma, beta, outputDtype: output.dtype, outputQuantization: output.quantization,
      });
      _cpuQGroupNorm(node);
      assert.ok(output.dtype === 'int8' ? out.buffer instanceof Int8Array : out.buffer instanceof Uint8Array);
      assert.deepEqual([...out.buffer], referenceQGroupNorm({
        inputShape: [1, 2, 2, 4], inputValues: input.values, inputQuantization: input.quantization,
        gamma, beta, outputDtype: output.dtype, outputQuantization: output.quantization,
        numGroups: 2, eps: 1e-5,
      }));
    }
  }
});

test('QGroupNorm uses centered raw variance for high, nearly constant U8 values', () => {
  // In raw-byte units these are 223/224 after a nonzero zero point. Computing
  // E[raw^2] - E[raw]^2 in F32 loses the 0.25 variance for this 1,024-value
  // group; the centered reduction must instead preserve normalized +/-1.
  const inputShape = [1, 16, 16, 4];
  const inputValues = Array.from({ length: elements(inputShape) }, (_, index) =>
    index % 2 === 0 ? 240 : 241,
  );
  const inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: 17 };
  const gamma = [1, 1, 1, 1];
  const beta = [0, 0, 0, 0];
  const outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 128 };
  const { node, out } = qGroupNormGraph({
    inputShape, inputDtype: 'uint8', inputValues, inputQuantization, gamma, beta,
    outputDtype: 'uint8', outputQuantization, numGroups: 1,
  });
  _cpuQGroupNorm(node);
  const expected = inputValues.map((value) => value === 240 ? 124 : 132);
  assert.deepEqual([...out.buffer], expected);
  assert.deepEqual([...out.buffer], referenceQGroupNorm({
    inputShape, inputValues, inputQuantization, gamma, beta,
    outputDtype: 'uint8', outputQuantization, numGroups: 1, eps: 1e-5,
  }));
});

test('QGroupNorm has logical tail bytes only; an incomplete packed word pads with zero', () => {
  const inputShape = [1, 1, 1, 3];
  const inputValues = [-5, 0, 4];
  const inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -1 };
  const gamma = [1, 0.5, -0.75];
  const beta = [0.25, -0.5, 0.75];
  const outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 };
  const { node, out } = qGroupNormGraph({
    inputShape, inputValues, inputQuantization, gamma, beta, outputQuantization, numGroups: 1,
  });
  _cpuQGroupNorm(node);
  assert.deepEqual([...out.buffer], referenceQGroupNorm({
    inputShape, inputValues, inputQuantization, gamma, beta,
    outputDtype: 'int8', outputQuantization, numGroups: 1, eps: 1e-5,
  }));
  const packed = packLogicalBytes(out.buffer);
  assert.equal(packed.length, 1);
  assert.equal(packed[0] >>> 24, 0);
});

test('QGroupNorm requantizes ties to even and saturates I8 output', () => {
  const { node, out } = qGroupNormGraph({
    inputShape: [1, 1, 1, 6], inputValues: [-3, -2, -1, 1, 2, 3],
    inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    gamma: [0, 0, 0, 0, 0, 0], beta: [0.5, 1.5, -1.5, 1000, -1000, 0],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 }, numGroups: 1,
  });
  _cpuQGroupNorm(node);
  assert.deepEqual([...out.buffer], [0, 2, -2, 127, -128, 0]);
});

test('QGroupNorm validates all descriptors before modifying output storage', () => {
  const { node, weight, out } = qGroupNormGraph();
  out.buffer.fill(73);
  weight.buffer[3] = Number.NaN;
  assert.throws(() => _cpuQGroupNorm(node), /weight\[3\] must be finite/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  weight.buffer[3] = 1;
  node.params.data_layout = 'NCHW';
  assert.throws(() => _cpuQGroupNorm(node), /NHWC data_layout only/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
});

test('QGroupNorm rejects per-axis activation mappings and output/input aliasing', () => {
  const { node, input, out } = qGroupNormGraph({
    inputQuantization: {
      scheme: 'per_axis', axis: 3, scales: [0.25, 0.25, 0.25, 0.25], zero_points: [0, 0, 0, 0],
    },
  });
  out.buffer.fill(29);
  assert.throws(() => _cpuQGroupNorm(node), /input requires per_tensor quantization/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(29));

  const aliasNode = { ...node, outputs: { out: input } };
  assert.throws(() => _cpuQGroupNorm(aliasNode), /output storage distinct/);
});
