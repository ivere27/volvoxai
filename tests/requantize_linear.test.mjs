import test from 'node:test';
import assert from 'node:assert/strict';

import { CPUEngine } from '../ts/backends/CPUEngine.js';
import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';

function makeGraph(inputDtype, inputQuantization, outputDtype, outputQuantization) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [7], inputDtype, { quantization: inputQuantization });
  const { out } = graph.addOp('RequantizeLinear', { input }, {
    out: { name: 'out', shape: [7], dtype: outputDtype, quantization: outputQuantization },
  });
  graph.setOutputs([out.name]);
  return graph;
}

test('CPU RequantizeLinear preserves real values across asymmetric U8 to I8 storage', async () => {
  const graph = makeGraph(
    'uint8', { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: -1 },
  );
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Uint8Array.of(0, 124, 127, 128, 129, 132, 255) });
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [-65, -3, -2, -1, 0, 1, 62]);
});

test('CPU RequantizeLinear uses ties-to-even and saturates in the destination range', async () => {
  const graph = makeGraph(
    'int8', { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
    'uint8', { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
  );
  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const result = await engine.execute({ input: Int8Array.of(-128, -65, -1, 0, 1, 64, 127) });
  assert.deepEqual([...result.out], [0, 0, 126, 128, 130, 255, 255]);
});

test('CPU RequantizeLinear rejects a missing typed descriptor', () => {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  });
  const { out } = graph.addOp('RequantizeLinear', { input }, { out: { name: 'out', shape: [1], dtype: 'int8' } });
  const engine = new CPUEngine();
  assert.throws(
    () => engine.allocateGraph(graph),
    /equal-shape canonical per-tensor I8\/U8 edges/,
  );
  assert.equal(out.quantization, null);
});
