import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { _cpuQLinear } from '../ts/ops/qLinear.js';

function addQLinear(graph, {
  inputDtype = 'int8',
  inputValues,
  inputShape,
  inputQuantization,
  weightDtype = 'int8',
  weightValues,
  weightShape,
  weightQuantization,
  biasValues,
  outputDtype = 'int8',
  outputShape,
  outputQuantization,
} = {}) {
  const inputStorage = inputDtype === 'int8' ? Int8Array.from(inputValues) : Uint8Array.from(inputValues);
  const weightStorage = weightDtype === 'int8' ? Int8Array.from(weightValues) : Uint8Array.from(weightValues);
  const outputElements = outputShape.reduce((product, dimension) => product * dimension, 1);
  const outputStorage = outputDtype === 'int8' ? new Int8Array(outputElements) : new Uint8Array(outputElements);
  const input = graph.addInput('input', inputShape, inputDtype, { quantization: inputQuantization });
  input.buffer = inputStorage;
  const weight = graph.addWeight('weight', weightShape, weightDtype, {
    buffer: weightStorage,
    quantization: weightQuantization,
  });
  const bias = graph.addWeight('bias', [weightShape[0]], 'int32', {
    buffer: Int32Array.from(biasValues),
  });
  const { out } = graph.addOp('QLinear', { input, weight, bias }, {
    out: {
      name: 'out', shape: outputShape, dtype: outputDtype,
      buffer: outputStorage, quantization: outputQuantization,
    },
  });
  return graph.nodes[0];
}

test('QLinear keeps I8 activations and I32 accumulation through per-axis requantization', () => {
  const graph = new RuntimeGraph();
  const node = addQLinear(graph, {
    inputValues: [1, -2, 3, 4, 0, -1],
    inputShape: [2, 3],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    weightValues: [2, 0, -1, -2, 1, 3],
    weightShape: [2, 3],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2] },
    biasValues: [2, -4],
    outputShape: [2, 2],
    outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
  });

  _cpuQLinear(node);
  assert.ok(node.outputs.out.buffer instanceof Int8Array);
  // 13 * 0.5 is the positive tie and -1 * 0.5 is the negative tie.
  assert.deepEqual([...node.outputs.out.buffer], [-3, 6, 6, 0]);
});

test('QLinear supports asymmetric U8 tensors and saturates its quantized output', () => {
  const graph = new RuntimeGraph();
  const node = addQLinear(graph, {
    inputDtype: 'uint8',
    inputValues: [255, 255],
    inputShape: [1, 2],
    inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 128 },
    weightDtype: 'uint8',
    weightValues: [255, 0, 128, 128],
    weightShape: [2, 2],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [128, 128] },
    biasValues: [-1000, 1024],
    outputDtype: 'uint8',
    outputShape: [1, 2],
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  });

  _cpuQLinear(node);
  assert.ok(node.outputs.out.buffer instanceof Uint8Array);
  assert.deepEqual([...node.outputs.out.buffer], [0, 255]);
});

test('QLinear rejects underflowing and overflowing F32 requantization multipliers', () => {
  const minimumF32 = 1.401298464324817e-45;
  for (const [label, inputScale, weightScale] of [
    ['underflowing', minimumF32, minimumF32],
    ['overflowing', 1e30, 1e30],
  ]) {
    const graph = new RuntimeGraph();
    const node = addQLinear(graph, {
      inputValues: [1],
      inputShape: [1, 1],
      inputQuantization: { scheme: 'per_tensor', scale: inputScale, zero_point: 0 },
      weightValues: [1],
      weightShape: [1, 1],
      weightQuantization: {
        scheme: 'per_axis', axis: 0, scales: [weightScale], zero_points: [0],
      },
      biasValues: [0],
      outputShape: [1, 1],
      outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    });
    assert.throws(
      () => _cpuQLinear(node),
      /requantization multiplier is not representable as positive F32/,
      label,
    );
  }
});

test('CPU engine dispatches explicit QLinear without entering the W8A32 MatMul path', async () => {
  const graph = new RuntimeGraph();
  const node = addQLinear(graph, {
    inputValues: [1, -2, 3], inputShape: [1, 3],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    weightValues: [2, 0, -1, -2, 1, 3], weightShape: [2, 3],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [1, -2] },
    biasValues: [2, -4], outputShape: [1, 2],
    outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
  });
  graph.setOutputs([node.outputs.out.name]);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(1, -2, 3) });
  assert.deepEqual([...result.out], [-3, 6]);
});

test('QLinear rejects noncanonical activation, weight, bias, and shape contracts', () => {
  const badInputGraph = new RuntimeGraph();
  const badInput = addQLinear(badInputGraph, {
    inputValues: [1, 2], inputShape: [1, 2],
    inputQuantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
    weightValues: [1, 2], weightShape: [1, 2],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
    biasValues: [0], outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  assert.throws(() => _cpuQLinear(badInput), /input requires per_tensor quantization/);

  const badWeightGraph = new RuntimeGraph();
  const badWeight = addQLinear(badWeightGraph, {
    inputValues: [1, 2, 3], inputShape: [1, 3],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    weightValues: [1, 2, 3, 4, 5, 6], weightShape: [2, 3],
    weightQuantization: { scheme: 'per_axis', axis: 1, scales: [0.25, 0.25, 0.25], zero_points: [0, 0, 0] },
    biasValues: [0, 0], outputShape: [1, 2],
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  assert.throws(() => _cpuQLinear(badWeight), /weight quantization along axis 0/);

  const badBiasGraph = new RuntimeGraph();
  const badBias = addQLinear(badBiasGraph, {
    inputValues: [1, 2], inputShape: [1, 2],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    weightValues: [1, 2], weightShape: [1, 2],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
    biasValues: [0], outputShape: [1, 1],
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  badBias.inputs.bias.dtype = 'float32';
  assert.throws(() => _cpuQLinear(badBias), /bias requires int32 storage/);

  const badShapeGraph = new RuntimeGraph();
  const badShape = addQLinear(badShapeGraph, {
    inputValues: [1, 2], inputShape: [1, 2],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    weightValues: [1, 2], weightShape: [1, 2],
    weightQuantization: { scheme: 'per_axis', axis: 0, scales: [0.25], zero_points: [0] },
    biasValues: [0], outputShape: [1, 2],
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  assert.throws(() => _cpuQLinear(badShape), /incompatible \[d_out,d_in\] dimensions/);
});
