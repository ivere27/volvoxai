import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { RuntimeGraphLoader } from '../ts/core/RuntimeGraphLoader.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function qArgMaxGraph() {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [2, 3, 2], 'int8', {
    quantization: { scheme: 'per_tensor', scale: 0.25, zero_point: -3 },
  });
  const { out } = graph.addOp('QArgMax', { input }, {
    out: { name: 'out', shape: [2, 2], dtype: 'int32' },
  }, { axis: 1 });
  graph.setOutputs([out.name]);
  return { graph, node: graph.nodes[0], input, out };
}

test('CPU dispatches QArgMax through its raw-byte kernel, not generic ArgMax', async () => {
  const { graph } = qArgMaxGraph();
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(graph));

  const engine = new CPUEngine();
  engine.allocateGraph(graph);
  const qargmax = engine._cpuQArgMax;
  let qargmaxCalls = 0;
  engine._cpuQArgMax = (node) => {
    qargmaxCalls++;
    return qargmax(node);
  };
  engine._cpuArgMax = () => { throw new Error('QArgMax must not select generic ArgMax'); };

  const result = await engine.execute({
    input: Int8Array.of(1, 9, 5, 9, 5, 4, 2, 0, 8, 7, 6, 10),
  });
  assert.equal(qargmaxCalls, 1);
  assert.ok(result.out instanceof Int32Array);
  assert.deepEqual([...result.out], [1, 0, 1, 2]);
});

test('RuntimeGraphLoader admits only the canonical QArgMax byte boundary', () => {
  assert.doesNotThrow(() => RuntimeGraphLoader._assertBrowserQuantizationSupported(qArgMaxGraph().graph));

  const badParams = qArgMaxGraph();
  badParams.node.params.keepdims = false;
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(badParams.graph),
    /exact \{ axis: integer \}/,
  );

  const quantizedOutput = qArgMaxGraph();
  quantizedOutput.out.quantization = Object.freeze({ scheme: 'per_tensor', scale: 0.25, zero_point: 0 });
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(quantizedOutput.graph),
    /unquantized I32 output/,
  );

  const wrongName = qArgMaxGraph();
  wrongName.node.inputs = { x: wrongName.input };
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(wrongName.graph),
    /exactly \{ input \}/,
  );

  const mutableDescriptor = qArgMaxGraph();
  mutableDescriptor.input.quantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -3 };
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(mutableDescriptor.graph),
    /immutable per-tensor/,
  );

  const generic = qArgMaxGraph();
  generic.node.opType = 'ArgMax';
  assert.throws(
    () => RuntimeGraphLoader._assertBrowserQuantizationSupported(generic.graph),
    /unsupported generic operator/,
  );
});
