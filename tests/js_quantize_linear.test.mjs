import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { Graph } from '../ts/core/Graph.js';
import { SafetensorsFile } from '../ts/core/Safetensors.js';
import { exportModelCheckpoint, importModelCheckpoint } from '../ts/training/ModelCheckpoint.js';

function quantizeGraph(outputDtype, scaleValue, zeroValue = null) {
  const graph = new Graph();
  const input = graph.addInput('input', [10], 'float32');
  const scale = graph.addWeight('scale', [1], 'float32', { buffer: Float32Array.of(scaleValue) });
  const inputs = { input, scale };
  if (zeroValue != null) {
    const values = outputDtype === 'int8' ? Int8Array.of(zeroValue) : Uint8Array.of(zeroValue);
    inputs.zero_point = graph.addWeight('zero_point', [1], outputDtype, { buffer: values });
  }
  const { out } = graph.addOp('QuantizeLinear', inputs, {
    out: {
      name: 'quantized', shape: [10], dtype: outputDtype,
      quantization: { scheme: 'per_tensor', scale: scaleValue, zero_point: zeroValue ?? 0 },
    },
  });
  graph.setOutputs([out.name]);
  return graph;
}

test('CPU QuantizeLinear writes canonical signed W8A8 activation storage', async () => {
  const graph = quantizeGraph('int8', 0.5, 0);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({
    input: Float32Array.of(-100, -1.5, -1.25, -0.75, -0.5, 0, 0.5, 1.5, 100, Number.NaN),
  });
  assert.ok(result.quantized instanceof Int8Array);
  assert.deepEqual([...result.quantized], [-128, -3, -2, -2, -1, 0, 1, 3, 127, 0]);
});

test('CPU QuantizeLinear supports asymmetric unsigned activations and saturates infinities', async () => {
  const graph = quantizeGraph('uint8', 0.25, 128);
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({
    input: Float32Array.of(-100, -32, -31.875, -31.5, 0, 0.125, 0.375, 31.75, 100, Number.POSITIVE_INFINITY),
  });
  assert.ok(result.quantized instanceof Uint8Array);
  assert.deepEqual([...result.quantized], [0, 0, 0, 2, 128, 128, 130, 255, 255, 255]);
});

test('CPU QuantizeLinear fails closed on an invalid typed quantization descriptor', () => {
  assert.throws(
    () => quantizeGraph('int8', 0, 0),
    /quantization scale must be finite, positive, and representable as F32/,
  );
});

test('CPU QuantizeLinear rejects a runtime scale that disagrees with its output metadata', async () => {
  const graph = quantizeGraph('int8', 0.25, 0);
  graph.getTensor('scale').buffer[0] = 0.5;
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  await assert.rejects(
    () => engine.execute({ input: new Float32Array(10) }),
    /do not match its output quantization metadata/,
  );
});

test('quantized tensor metadata survives a checkpoint round-trip', () => {
  const graph = quantizeGraph('int8', 0.25, 0);
  graph.addWeight('qweight', [2, 2], 'int8', {
    buffer: Int8Array.of(1, 2, 3, 4),
    quantization: { scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [0, 0] },
  });
  const checkpoint = exportModelCheckpoint(graph, { includeOptimizerState: false });
  const descriptor = checkpoint.graph.quantization.tensors.quantized;
  assert.deepEqual(descriptor, {
    scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero_point',
  });
  assert.deepEqual(checkpoint.graph.nodes[0].inputs, {
    input: 'input', scale: 'scale', zero_point: 'zero_point',
  });
  assert.equal(JSON.stringify(checkpoint.graph).includes('"scale":0.25'), false);
  const stored = SafetensorsFile.fromArrayBuffer(checkpoint.weights);
  assert.deepEqual([...stored.toRuntimeTypedArray(stored.getTensor('scale'))], [0.25]);
  assert.deepEqual([...stored.toRuntimeTypedArray(stored.getTensor('zero_point'))], [0]);
  const restored = importModelCheckpoint(checkpoint).graph;
  assert.deepEqual(restored.getTensor('quantized').quantization, {
    scheme: 'per_tensor', scale: 0.25, zero_point: 0,
  });
  assert.deepEqual(restored.getTensor('qweight').quantization, {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25], zero_points: [0, 0],
  });
});
