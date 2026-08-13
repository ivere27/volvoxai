import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { _cpuQLayerNorm } from '../ts/ops/qLayerNorm.js';

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function elements(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function qLayerNormGraph({
  inputShape = [2, 4],
  inputDtype = 'int8',
  inputValues = [-8, -3, 4, 7, 5, -1, 2, -6],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  gamma = [1, -0.75, 0.5, 1.25],
  beta = [0.25, -0.5, 0.75, -0.25],
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
  eps = 1e-5,
  dModel,
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: byteStorage(inputDtype, inputValues), quantization: inputQuantization,
  });
  const d = inputShape.at(-1);
  const weight = graph.addWeight('weight', [d], 'float32', { buffer: Float32Array.from(gamma) });
  const bias = graph.addWeight('bias', [d], 'float32', { buffer: Float32Array.from(beta) });
  const params = { eps };
  if (dModel !== undefined) params.d_model = dModel;
  const { out } = graph.addOp('QLayerNorm', { input, weight, bias }, {
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

// A separate straightforward reference over logical byte values. It is kept
// independent from the portable operator so all scalar-F32 staging remains
// observable in these regression cases.
function referenceQLayerNorm({
  inputShape, inputValues, inputQuantization, gamma, beta,
  outputDtype, outputQuantization, eps,
}) {
  const dModel = inputShape.at(-1);
  const rows = elements(inputShape) / dModel;
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const epsilon = Math.fround(eps);
  const minimum = outputDtype === 'int8' ? -128 : 0;
  const maximum = outputDtype === 'int8' ? 127 : 255;
  const result = new Array(inputValues.length);

  for (let row = 0; row < rows; row++) {
    const offset = row * dModel;
    let meanRaw = Math.fround(0);
    for (let channel = 0; channel < dModel; channel++) {
      const raw = Math.fround(inputValues[offset + channel] - inputQuantization.zero_point);
      meanRaw = Math.fround(meanRaw + raw);
    }
    meanRaw = Math.fround(meanRaw / dModel);

    let varianceRaw = Math.fround(0);
    for (let channel = 0; channel < dModel; channel++) {
      const raw = Math.fround(inputValues[offset + channel] - inputQuantization.zero_point);
      const centered = Math.fround(raw - meanRaw);
      varianceRaw = Math.fround(varianceRaw + Math.fround(centered * centered));
    }
    varianceRaw = Math.fround(varianceRaw / dModel);
    const variance = Math.fround(Math.fround(Math.max(0, varianceRaw) * inputScale) * inputScale);
    const invStd = Math.fround(1 / Math.sqrt(Math.fround(variance + epsilon)));

    for (let channel = 0; channel < dModel; channel++) {
      const raw = Math.fround(inputValues[offset + channel] - inputQuantization.zero_point);
      const normalized = Math.fround(Math.fround(Math.fround(raw - meanRaw) * inputScale) * invStd);
      const affine = Math.fround(Math.fround(normalized * Math.fround(gamma[channel])) +
        Math.fround(beta[channel]));
      const transformed = Math.fround(Math.fround(affine / outputScale) + outputQuantization.zero_point);
      let quantized;
      if (Number.isNaN(transformed)) quantized = outputQuantization.zero_point;
      else if (transformed <= minimum) quantized = minimum;
      else if (transformed >= maximum) quantized = maximum;
      else quantized = roundEven(transformed);
      result[offset + channel] = quantized;
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

test('QLayerNorm normalizes each byte-domain row without an F32 activation output', () => {
  const { node, out } = qLayerNormGraph();
  _cpuQLayerNorm(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], referenceQLayerNorm({
    inputShape: [2, 4], inputValues: [-8, -3, 4, 7, 5, -1, 2, -6],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    gamma: [1, -0.75, 0.5, 1.25], beta: [0.25, -0.5, 0.75, -0.25],
    outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    eps: 1e-5,
  }));
});

test('QLayerNorm supports every I8/U8 activation input/output pairing', () => {
  const centeredValues = [-8, -3, 4, 7, 5, -1, 2, -6];
  const inputs = [
    { dtype: 'int8', values: centeredValues, quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 } },
    { dtype: 'uint8', values: centeredValues.map((value) => value + 128), quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 } },
  ];
  const outputs = [
    { dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -7 } },
    { dtype: 'uint8', quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 123 } },
  ];
  const gamma = [1, -0.75, 0.5, 1.25];
  const beta = [0.25, -0.5, 0.75, -0.25];

  for (const input of inputs) {
    for (const output of outputs) {
      const { node, out } = qLayerNormGraph({
        inputDtype: input.dtype, inputValues: input.values, inputQuantization: input.quantization,
        gamma, beta, outputDtype: output.dtype, outputQuantization: output.quantization,
      });
      _cpuQLayerNorm(node);
      assert.ok(output.dtype === 'int8' ? out.buffer instanceof Int8Array : out.buffer instanceof Uint8Array);
      assert.deepEqual([...out.buffer], referenceQLayerNorm({
        inputShape: [2, 4], inputValues: input.values, inputQuantization: input.quantization,
        gamma, beta, outputDtype: output.dtype, outputQuantization: output.quantization, eps: 1e-5,
      }));
    }
  }
});

test('QLayerNorm supports D=1 and returns the affine bias domain', () => {
  const beta = [0.5];
  const { node, out } = qLayerNormGraph({
    inputShape: [3, 1], inputValues: [-121, 0, 117], gamma: [17], beta,
    inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -2 }, dModel: 1,
  });
  _cpuQLayerNorm(node);
  assert.deepEqual([...out.buffer], [0, 0, 0]);
});

test('QLayerNorm preserves centered variance for high near-constant U8 D=320 rows', () => {
  const inputShape = [4, 320];
  const inputValues = Array.from({ length: elements(inputShape) }, (_, index) =>
    index % 2 === 0 ? 240 : 241,
  );
  const gamma = new Array(320).fill(1);
  const beta = new Array(320).fill(0);
  const inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: 17 };
  const outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 128 };
  const { node, out } = qLayerNormGraph({
    inputShape, inputDtype: 'uint8', inputValues, inputQuantization, gamma, beta,
    outputDtype: 'uint8', outputQuantization,
  });
  _cpuQLayerNorm(node);
  assert.deepEqual([...out.buffer], inputValues.map((value) => value === 240 ? 124 : 132));
  assert.deepEqual([...out.buffer], referenceQLayerNorm({
    inputShape, inputValues, inputQuantization, gamma, beta,
    outputDtype: 'uint8', outputQuantization, eps: 1e-5,
  }));
});

test('QLayerNorm has logical tail bytes only for odd D', () => {
  const inputShape = [2, 3];
  const inputValues = [-5, 0, 4, 7, -3, 2];
  const inputQuantization = { scheme: 'per_tensor', scale: 0.5, zero_point: -1 };
  const gamma = [1, 0.5, -0.75];
  const beta = [0.25, -0.5, 0.75];
  const outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: -3 };
  const { node, out } = qLayerNormGraph({
    inputShape, inputValues, inputQuantization, gamma, beta, outputQuantization,
  });
  _cpuQLayerNorm(node);
  assert.deepEqual([...out.buffer], referenceQLayerNorm({
    inputShape, inputValues, inputQuantization, gamma, beta,
    outputDtype: 'int8', outputQuantization, eps: 1e-5,
  }));
  const packed = packLogicalBytes(out.buffer);
  assert.equal(packed.length, 2);
  assert.equal(packed[1] >>> 16, 0);
});

test('QLayerNorm requantizes ties to even and saturates I8 output', () => {
  const { node, out } = qLayerNormGraph({
    inputShape: [1, 6], inputValues: [-3, -2, -1, 1, 2, 3],
    inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    gamma: [0, 0, 0, 0, 0, 0], beta: [0.5, 1.5, -1.5, 1000, -1000, 0],
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  _cpuQLayerNorm(node);
  assert.deepEqual([...out.buffer], [0, 2, -2, 127, -128, 0]);
});

test('QLayerNorm rejects aliases and malformed descriptors before writing output', () => {
  const { node, weight, out } = qLayerNormGraph();
  out.buffer.fill(73);
  weight.buffer[3] = Number.NaN;
  assert.throws(() => _cpuQLayerNorm(node), /weight\[3\] must be finite/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  weight.buffer[3] = 1;
  node.params.d_model = 5;
  assert.throws(() => _cpuQLayerNorm(node), /d_model must match/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  node.params.d_model = 4;
  const aliasNode = { ...node, outputs: { out: node.inputs.input } };
  assert.throws(() => _cpuQLayerNorm(aliasNode), /output storage distinct/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
});
