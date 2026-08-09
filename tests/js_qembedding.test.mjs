import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { _cpuQEmbedding } from '../ts/ops/qEmbedding.js';

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function qEmbeddingGraph({
  ids = [2, 0],
  idsShape = [2],
  weightDtype = 'int8',
  weightValues = [-1, 0, 1, 2, 0, -2, 5, 1, -3],
  weightShape = [3, 3],
  weightQuantization = {
    scheme: 'per_axis', axis: 0, scales: [0.5, 0.25, 0.125], zero_points: [0, 0, 1],
  },
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('ids', idsShape, 'int32', { buffer: Int32Array.from(ids) });
  const weight = graph.addWeight('table', weightShape, weightDtype, {
    buffer: byteStorage(weightDtype, weightValues),
    quantization: weightQuantization,
  });
  const outputShape = [...idsShape, weightShape[1]];
  const outputElements = outputShape.reduce((product, dimension) => product * dimension, 1);
  const { out } = graph.addOp('QEmbedding', { input, weight }, {
    out: {
      name: 'out', shape: outputShape, dtype: outputDtype,
      buffer: byteStorage(outputDtype, new Array(outputElements).fill(0)),
      quantization: outputQuantization,
    },
  });
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, weight, out };
}

function clippedQEmbeddingGraph({ maximum = 2 } = {}) {
  const graph = new RuntimeGraph();
  const raw = graph.addInput('raw_ids', [2], 'int32', {
    buffer: Int32Array.of(-9, 17),
  });
  const { out: ids } = graph.addOp('Clip', { input: raw }, {
    out: { name: 'ids', shape: [2], dtype: 'int32' },
  }, { min: 0, max: maximum });
  const weight = graph.addWeight('table', [3, 3], 'int8', {
    buffer: Int8Array.of(-1, 0, 1, 2, 0, -2, 5, 1, -3),
    quantization: {
      scheme: 'per_axis', axis: 0,
      scales: [0.5, 0.25, 0.125], zero_points: [0, 0, 1],
    },
  });
  const { out } = graph.addOp('QEmbedding', { input: ids, weight }, {
    out: {
      name: 'out', shape: [2, 3], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
    },
  });
  graph.setOutputs([out.name]);
  return { graph, raw, ids, out, node: graph.nodes[1] };
}

function reference(ids, weight, hidden, rowScales, rowZeroPoints, outputDtype, outputScale, outputZeroPoint) {
  const minimum = outputDtype === 'int8' ? -128 : 0;
  const maximum = outputDtype === 'int8' ? 127 : 255;
  const result = [];
  for (const id of ids) {
    for (let hiddenIndex = 0; hiddenIndex < hidden; hiddenIndex++) {
      const centered = Math.fround(weight[id * hidden + hiddenIndex] - rowZeroPoints[id]);
      const dequantized = Math.fround(centered * Math.fround(rowScales[id]));
      const transformed = Math.fround(Math.fround(dequantized / Math.fround(outputScale)) + outputZeroPoint);
      if (Number.isNaN(transformed)) result.push(outputZeroPoint);
      else if (transformed <= minimum) result.push(minimum);
      else if (transformed >= maximum) result.push(maximum);
      else {
        const lower = Math.floor(transformed);
        const fraction = transformed - lower;
        result.push(fraction < 0.5 ? lower : fraction > 0.5 ? lower + 1 : (lower % 2 === 0 ? lower : lower + 1));
      }
    }
  }
  return result;
}

test('QEmbedding gathers per-row I8 values and requantizes directly to a typed activation', () => {
  const { node, out } = qEmbeddingGraph();
  _cpuQEmbedding(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], [1, -1, -3, -3, -1, 1]);
});

test('QEmbedding uses ties-to-even rounding and saturates its typed output', () => {
  const { node, out } = qEmbeddingGraph({
    ids: [0, 1], idsShape: [2], weightShape: [2, 3],
    weightValues: [1, 3, -3, 127, -128, 0],
    weightQuantization: {
      scheme: 'per_axis', axis: 0, scales: [0.25, 1], zero_points: [0, 0],
    },
    outputQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
  });
  _cpuQEmbedding(node);
  // 0.5 -> 0, 1.5 -> 2, and -1.5 -> -2; the second row saturates.
  assert.deepEqual([...out.buffer], [0, 2, -2, 127, -128, 0]);
});

test('QEmbedding supports every I8/U8 table/output storage combination', () => {
  const cases = [
    {
      weightDtype: 'int8', weightValues: [-1, 0, 1, 2, 0, -2, 5, 1, -3],
      scales: [0.5, 0.25, 0.125], zeroPoints: [0, 0, 1],
    },
    {
      weightDtype: 'uint8', weightValues: [127, 128, 129, 22, 20, 18, 205, 201, 197],
      scales: [0.5, 0.25, 0.125], zeroPoints: [128, 20, 201],
    },
  ];
  const outputs = [
    { dtype: 'int8', quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 } },
    { dtype: 'uint8', quantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 100 } },
  ];
  for (const table of cases) {
    for (const output of outputs) {
      const ids = [2, 0, 1];
      const weightQuantization = {
        scheme: 'per_axis', axis: 0, scales: table.scales, zero_points: table.zeroPoints,
      };
      const { node, out } = qEmbeddingGraph({
        ids, idsShape: [ids.length], weightDtype: table.weightDtype, weightValues: table.weightValues,
        weightQuantization, outputDtype: output.dtype, outputQuantization: output.quantization,
      });
      _cpuQEmbedding(node);
      assert.ok(output.dtype === 'int8' ? out.buffer instanceof Int8Array : out.buffer instanceof Uint8Array);
      assert.deepEqual([...out.buffer], reference(
        ids, table.weightValues, 3, table.scales, table.zeroPoints, output.dtype,
        output.quantization.scale, output.quantization.zero_point,
      ));
    }
  }
});

test('QEmbedding rejects an invalid ID before it writes any output byte', () => {
  const { node, out } = qEmbeddingGraph({ ids: [0, 9] });
  out.buffer.fill(73);
  assert.throws(() => _cpuQEmbedding(node), /token id 9 is outside vocabulary size 3/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));
});

test('CPU engine dispatches QEmbedding without entering F32 Embedding', async () => {
  const { graph, input } = qEmbeddingGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  engine._cpuEmbedding = () => { throw new Error('QEmbedding must not use the F32 Embedding fallback'); };
  const result = await engine.execute({ [input.name]: Int32Array.of(2, 0) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [1, -1, -3, -3, -1, 1]);
});

test('RuntimeGraphLoader admits public or vocabulary-bounded internal QEmbedding IDs', () => {
  const { graph, node, input } = qEmbeddingGraph({ idsShape: [1, 2], ids: [2, 0] });
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));
  input.isInput = false;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph),
    new RegExp(`QEmbedding node ${node.id} .*preflight-complete I32 IDs`),
  );

  const bounded = clippedQEmbeddingGraph();
  assert.doesNotThrow(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(bounded.graph),
  );
  const outOfRange = clippedQEmbeddingGraph({ maximum: 3 });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(outOfRange.graph),
    /preflight-complete I32 IDs/,
  );
});

test('CPU QEmbedding executes vocabulary-bounded internal Clip IDs', async () => {
  const { graph, raw } = clippedQEmbeddingGraph();
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ [raw.name]: Int32Array.of(-9, 17) });
  assert.deepEqual([...result.out], [-3, -1, 1, 1, -1, -3]);
});
