import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function qSDPAGraph() {
  const graph = new RuntimeGraph();
  const q = graph.addInput('q', [2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const k = graph.addInput('k', [2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const v = graph.addInput('v', [2, 4], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  });
  const { out } = graph.addOp('QSDPA', { q, k, v }, {
    out: {
      name: 'out', shape: [2, 4], dtype: 'int8',
      quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    },
  }, { heads: 1, causal: false, scale: 0.5 });
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], q, k, v, out };
}

test('CPU dispatches QSDPA through the byte-domain kernel, not F32 attention kernels', async () => {
  const { graph } = qSDPAGraph();
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const qsdpa = engine._cpuQSDPA;
  let qsdpaCalls = 0;
  engine._cpuQSDPA = (node) => {
    qsdpaCalls++;
    return qsdpa(node);
  };
  engine._cpuSDPA = () => { throw new Error('QSDPA must not select packed F32 SDPA'); };
  engine._cpuCrossSDPA = () => { throw new Error('QSDPA must not select F32 CrossSDPA'); };

  const result = await engine.execute({
    q: Int8Array.of(0, -1, -2, 1, -2, 1, 0, -1),
    k: Int8Array.of(0, 0, -1, -1, -1, 0, 0, -2),
    v: Int8Array.of(3, -3, 0, -1, -5, 1, 2, -2),
  });
  assert.equal(qsdpaCalls, 1);
  assert.ok(result.out instanceof Int8Array);
  assert.deepEqual([...result.out], [0, 0, 2, 0, 0, 0, 2, -1]);
});

test('RuntimeGraphLoader accepts only canonical QSDPA byte boundaries', () => {
  const explicitNullScale = qSDPAGraph();
  explicitNullScale.node.params.scale = null;
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(explicitNullScale.graph));

  const missingCausal = qSDPAGraph();
  delete missingCausal.node.params.causal;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(missingCausal.graph),
    /explicit heads\/causal/,
  );

  const overflowScale = qSDPAGraph();
  overflowScale.node.params.scale = 3e38;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(overflowScale.graph),
    /finite score scaling/,
  );

  const malformedMask = qSDPAGraph();
  const mask = malformedMask.graph.addInput('mask', [3], 'int32');
  malformedMask.node.inputs.mask = mask;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(malformedMask.graph),
    /optional I32 mask inputs/,
  );

  const generic = qSDPAGraph();
  generic.node.opType = 'CrossSDPA';
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(generic.graph),
    /unsupported generic operator/,
  );
});
