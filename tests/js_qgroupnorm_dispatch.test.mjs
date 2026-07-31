import test from 'node:test';
import assert from 'node:assert/strict';

import { Graph } from '../ts/core/Graph.js';
import { GraphLoader } from '../ts/core/GraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function qGroupNormGraph() {
  const graph = new Graph();
  const input = graph.addInput('input', [1, 2, 2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const weight = graph.addWeight('weight', [4], 'float32', {
    buffer: Float32Array.of(1, -0.75, 0.5, 1.25),
  });
  const bias = graph.addWeight('bias', [4], 'float32', {
    buffer: Float32Array.of(0.25, -0.5, 0.75, -0.25),
  });
  const { out } = graph.addOp('QGroupNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: [1, 2, 2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    },
  }, { num_groups: 2, eps: 1e-5, data_layout: 'NHWC' });
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, out };
}

test('CPU dispatches QGroupNorm through the byte-domain kernel, not F32 GroupNorm', async () => {
  const { graph } = qGroupNormGraph();
  assert.doesNotThrow(() => GraphLoader._assertBrowserQuantizationSupported(graph));

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const qGroupNorm = engine._cpuQGroupNorm;
  let qGroupNormCalls = 0;
  engine._cpuQGroupNorm = (node) => {
    qGroupNormCalls++;
    return qGroupNorm(node);
  };
  engine._cpuGroupNorm = () => { throw new Error('QGroupNorm must not materialize an F32 activation'); };

  const result = await engine.execute({
    input: Int8Array.of(-8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2),
  });
  assert.equal(qGroupNormCalls, 1);
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [-12, -4, 6, 11, 6, -6, 4, -21, -1, -16, -2, 1, -11, -13, 3, -11]);
});

test('GraphLoader rejects non-canonical QGroupNorm byte boundaries', () => {
  const invalidGroups = qGroupNormGraph();
  invalidGroups.node.params.num_groups = 3;
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(invalidGroups.graph),
    /rank-4 NHWC per-tensor I8\/U8 activation edges/,
  );

  const unexpectedInput = qGroupNormGraph();
  unexpectedInput.node.inputs.extra = unexpectedInput.input;
  assert.throws(
    () => GraphLoader._assertBrowserQuantizationSupported(unexpectedInput.graph),
    /exact input\/weight\/bias inputs/,
  );
});
