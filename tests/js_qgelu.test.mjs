import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { _cpuQGELU, qgeluValueF32 } from '../ts/ops/qGELU.js';

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

// The QGELU numerical contract intentionally uses this F32 A-S 7.1.26
// polynomial rather than a host erf/erff implementation or GELU-tanh. It is
// duplicated here as a specification-level test oracle for the JS/WASM/WGSL
// byte boundary, including its operation order.
function erfApproxF32(value) {
  const f32 = Math.fround;
  const sign = value >= 0 ? f32(1) : f32(-1);
  const x = f32(Math.abs(value));
  const denominator = f32(f32(1) + f32(f32(0.3275911) * x));
  const t = f32(f32(1) / denominator);
  let polynomial = f32(f32(1.061405429) * t);
  polynomial = f32(polynomial - f32(1.453152027));
  polynomial = f32(polynomial * t);
  polynomial = f32(polynomial + f32(1.421413741));
  polynomial = f32(polynomial * t);
  polynomial = f32(polynomial - f32(0.284496736));
  polynomial = f32(polynomial * t);
  polynomial = f32(polynomial + f32(0.254829592));
  polynomial = f32(polynomial * t);
  const squared = f32(x * x);
  const tail = f32(Math.exp(f32(-squared)));
  return f32(sign * f32(f32(1) - f32(polynomial * tail)));
}

function geluF32(value) {
  const f32 = Math.fround;
  const x = f32(value);
  const erf = erfApproxF32(f32(x * f32(0.7071067811865476)));
  const cdf = f32(f32(0.5) * f32(f32(1) + erf));
  return f32(x * cdf);
}

function reference(values, inputQuantization, outputDtype, outputQuantization) {
  const f32 = Math.fround;
  const minimum = outputDtype === 'int8' ? -128 : 0;
  const maximum = outputDtype === 'int8' ? 127 : 255;
  const inputScale = f32(inputQuantization.scale);
  const outputScale = f32(outputQuantization.scale);
  return values.map((raw) => {
    const x = f32(f32(raw - inputQuantization.zero_point) * inputScale);
    const transformed = f32(f32(geluF32(x) / outputScale) + outputQuantization.zero_point);
    if (Number.isNaN(transformed)) return outputQuantization.zero_point;
    if (transformed <= minimum) return minimum;
    if (transformed >= maximum) return maximum;
    return roundTiesToEven(transformed);
  });
}

function qGELUGraph({
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
  const { out } = graph.addOp('QGELU', inputs || { input }, {
    out: {
      name: 'out', shape: inputShape, dtype: outputDtype,
      buffer: byteStorage(outputDtype, new Array(outputElements).fill(0)),
      quantization: outputQuantization,
    },
  }, params);
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, out };
}

test('QGELU directly requantizes the portable erf GELU into typed I8 storage', () => {
  const { node, out } = qGELUGraph();
  _cpuQGELU(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], [-1, 0, 1, 5, 11]);
  assert.equal(qgeluValueF32(Math.fround(1)), Math.fround(0.8413447141647339));
});

test('QGELU uses ties-to-even rounding and saturates after its fixed GELU transform', () => {
  const positiveTie = qGELUGraph({
    inputValues: [1], inputShape: [1],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.05987062305212021, zero_point: 0 },
  });
  _cpuQGELU(positiveTie.node);
  // The F32 portable-erf transform is exactly +2.5 output units, so it rounds to +2.
  assert.deepEqual([...positiveTie.out.buffer], [2]);

  const negativeTie = qGELUGraph({
    inputValues: [-1], inputShape: [1],
    inputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.040129370987415314, zero_point: 0 },
  });
  _cpuQGELU(negativeTie.node);
  // The F32 portable-erf transform is exactly -2.5 output units, so it rounds to -2.
  assert.deepEqual([...negativeTie.out.buffer], [-2]);

  const saturated = qGELUGraph({
    inputValues: [-1, 127], inputShape: [2],
    inputQuantization: { scheme: 'per_tensor', scale: 0.75, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.001, zero_point: 0 },
  });
  _cpuQGELU(saturated.node);
  assert.deepEqual([...saturated.out.buffer], [-128, 127]);
});

test('QGELU supports every I8/U8 activation storage combination', () => {
  const inputVariants = [
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
  for (const inputVariant of inputVariants) {
    for (const outputVariant of outputVariants) {
      const { node, out } = qGELUGraph({ ...inputVariant, ...outputVariant });
      _cpuQGELU(node);
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

test('QGELU validates the complete canonical descriptor before it writes an output byte', () => {
  const { node, input, out } = qGELUGraph();
  out.buffer.fill(73);
  node.inputs.extra = input;
  assert.throws(() => _cpuQGELU(node), /exactly one activation input/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
});

test('CPU engine dispatches QGELU without entering the F32 GELU path', async () => {
  const { graph } = qGELUGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  engine._cpuGELU = () => { throw new Error('QGELU must not use the F32 GELU fallback'); };
  const result = await engine.execute({ input: Int8Array.of(-4, -1, 0, 2, 5) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [-1, 0, 1, 5, 11]);
});

test('GraphLoader accepts only fixed portable-erf QGELU semantics', () => {
  const omitted = qGELUGraph();
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(omitted.graph));

  const explicitNone = qGELUGraph({ params: { approximate: 'none' } });
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(explicitNone.graph));

  const tanh = qGELUGraph({ params: { approximate: 'tanh' } });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(tanh.graph),
    /only omitted parameters or approximate='none'/,
  );

  const extraParameter = qGELUGraph({ params: { approximate: 'none', alpha: 0.5 } });
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(extraParameter.graph),
    /only omitted parameters or approximate='none'/,
  );
});
