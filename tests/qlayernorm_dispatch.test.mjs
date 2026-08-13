import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function qLayerNormGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const weight = graph.addWeight('weight', [4], 'float32', {
    buffer: Float32Array.of(1, -0.75, 0.5, 1.25),
  });
  const bias = graph.addWeight('bias', [4], 'float32', {
    buffer: Float32Array.of(0.25, -0.5, 0.75, -0.25),
  });
  const { out } = graph.addOp('QLayerNorm', { input, weight, bias }, {
    out: {
      name: 'out', shape: [2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.125, zero_point: -3 },
    },
  }, {});
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, out };
}

test('CPU dispatches QLayerNorm through the byte-domain kernel, not F32 LayerNorm', async () => {
  const { graph } = qLayerNormGraph();
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const qLayerNorm = engine._cpuQLayerNorm;
  let qLayerNormCalls = 0;
  engine._cpuQLayerNorm = (node) => {
    qLayerNormCalls++;
    return qLayerNorm(node);
  };
  engine._cpuLayerNorm = () => { throw new Error('QLayerNorm must not materialize an F32 activation'); };

  const result = await engine.execute({
    input: Int8Array.of(-8, -3, 4, 7, 5, -1, 2, -6),
  });
  assert.equal(qLayerNormCalls, 1);
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [-12, -4, 6, 7, 9, -6, 5, -20]);
});

test('RuntimeGraphLoader rejects non-canonical QLayerNorm byte boundaries', () => {
  const invalidDModel = qLayerNormGraph();
  invalidDModel.node.params.d_model = 3;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(invalidDModel.graph),
    /same-shape rank-at-least-1 per-tensor I8\/U8 activation edges/,
  );

  const unexpectedInput = qLayerNormGraph();
  unexpectedInput.node.inputs.extra = unexpectedInput.input;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(unexpectedInput.graph),
    /exact input\/weight\/bias inputs/,
  );
});
