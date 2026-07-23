import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { _cpuQSiLU } from '../ts/ops/qSiLU.js';

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function roundTiesToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function reference(values, inputQuantization, outputDtype, outputQuantization) {
  const minimum = outputDtype === 'int8' ? -128 : 0;
  const maximum = outputDtype === 'int8' ? 127 : 255;
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  return values.map((raw) => {
    const x = Math.fround(Math.fround(raw - inputQuantization.zero_point) * inputScale);
    const denominator = Math.fround(1 + Math.fround(Math.exp(-x)));
    const silu = Math.fround(x / denominator);
    const transformed = Math.fround(Math.fround(silu / outputScale) + outputQuantization.zero_point);
    if (Number.isNaN(transformed)) return outputQuantization.zero_point;
    if (transformed <= minimum) return minimum;
    if (transformed >= maximum) return maximum;
    return roundTiesToEven(transformed);
  });
}

function qSiLUGraph({
  inputDtype = 'int8',
  inputValues = [-4, -1, 0, 2, 5],
  inputShape = [1, 5],
  inputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.125, zero_point: 0 },
  params = {},
  inputs = null,
} = {}) {
  const graph = new Graph();
  const input = graph.addInput('input', inputShape, inputDtype, {
    buffer: byteStorage(inputDtype, inputValues), quantization: inputQuantization,
  });
  const outputElements = inputShape.reduce((product, dimension) => product * dimension, 1);
  const { out } = graph.addOp('QSiLU', inputs || { input }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype,
      buffer: byteStorage(outputDtype, new Array(outputElements).fill(0)),
      quantization: outputQuantization,
    },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, out };
}

test('QSiLU directly requantizes SiLU into typed I8 storage', () => {
  const { node, out } = qSiLUGraph();
  _cpuQSiLU(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], [-2, 0, 1, 4, 10]);
});

test('QSiLU uses ties-to-even rounding and saturates after the SiLU transform', () => {
  const positiveTie = qSiLUGraph({
    inputValues: [1], inputShape: [1],
    inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.1244918704032898, zero_point: 0 },
  });
  _cpuQSiLU(positiveTie.node);
  // The F32 transform is exactly +2.5, which rounds to the even integer +2.
  assert.deepEqual([...positiveTie.out.buffer], [2]);

  const negativeTie = qSiLUGraph({
    inputValues: [-1], inputShape: [1],
    inputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.07550813257694244, zero_point: 0 },
  });
  _cpuQSiLU(negativeTie.node);
  // The F32 transform is exactly -2.5, which rounds to the even integer -2.
  assert.deepEqual([...negativeTie.out.buffer], [-2]);

  const saturated = qSiLUGraph({
    inputValues: [-1, 127], inputShape: [2],
    inputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.001, zero_point: 0 },
  });
  _cpuQSiLU(saturated.node);
  assert.deepEqual([...saturated.out.buffer], [-128, 127]);
});

test('QSiLU supports every I8/U8 activation storage combination', () => {
  const variants = [
    {
      inputDtype: 'int8', inputValues: [-4, -1, 0, 2, 5],
      inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    },
    {
      inputDtype: 'uint8', inputValues: [124, 127, 128, 130, 133],
      inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 127 },
    },
  ];
  const outputVariants = [
    { outputDtype: 'int8', outputQuantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 } },
    { outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 121 } },
  ];
  for (const inputVariant of variants) {
    for (const outputVariant of outputVariants) {
      const { node, out } = qSiLUGraph({ ...inputVariant, ...outputVariant });
      _cpuQSiLU(node);
      assert.ok(outputVariant.outputDtype === 'int8'
        ? out.buffer instanceof Int8Array
        : out.buffer instanceof Uint8Array);
      assert.deepEqual([...out.buffer], reference(
        inputVariant.inputValues, inputVariant.inputQuantization,
        outputVariant.outputDtype, outputVariant.outputQuantization,
      ));
    }
  }
});

test('QSiLU rejects invalid canonical descriptors before it writes an output byte', () => {
  const { node, input, out } = qSiLUGraph();
  out.buffer.fill(73);
  node.inputs.extra = input;
  assert.throws(() => _cpuQSiLU(node), /exactly one activation input/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
});

test('CPU engine dispatches QSiLU without entering the F32 SiLU path', async () => {
  const { graph, input } = qSiLUGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  engine._cpuSiLU = () => { throw new Error('QSiLU must not use the F32 SiLU fallback'); };
  const result = await engine.execute({ input: Int8Array.of(-4, -1, 0, 2, 5) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [-2, 0, 1, 4, 10]);
});

test('GraphLoader accepts only the parameter-free canonical QSiLU byte boundary', () => {
  const valid = qSiLUGraph();
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(valid.graph));

  const extra = qSiLUGraph();
  extra.node.inputs.extra = extra.input;
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(extra.graph),
    /unsupported canonical W8A8 input 'extra'/,
  );

  const parameterized = qSiLUGraph({ params: { alpha: 0.5 } });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(parameterized.graph),
    /same-shape canonical per-tensor I8\/U8 activation input\/output pair with distinct tensors and no parameters/,
  );
});
