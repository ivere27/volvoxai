import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function quantization(dtype) {
  return {
    scheme: 'per_tensor',
    scale: dtype === 'int8' ? 0.25 : 0.125,
    zero_point: dtype === 'int8' ? -3 : 123,
  };
}

function transposeGraph(dtype, {
  outputShape = [1, 2, 3, 2],
  outputQuantization = quantization(dtype),
  perm = [0, 2, 3, 1],
} = {}) {
  const graph = new RuntimeGraph();
  const input = graph.addInput('input', [1, 2, 2, 3], dtype, {
    quantization: quantization(dtype),
  });
  const { out } = graph.addOp('Transpose', { input }, {
    out: {
      name: 'out',
      shape: outputShape,
      dtype,
      quantization: outputQuantization,
    },
  }, { perm });
  graph.setOutputs([out.name]);
  return graph;
}

test('CPU Transpose preserves exact I8/U8 bytes across NCHW to NHWC', async () => {
  for (const dtype of ['int8', 'uint8']) {
    const Storage = dtype === 'int8' ? Int8Array : Uint8Array;
    const input = Storage.from(
      { length: 12 },
      (_, index) => dtype === 'int8' ? index - 6 : index + 117,
    );
    const engine = new CPUEngine();
    engine.allocateGraph(transposeGraph(dtype));
    const result = await engine.execute({ input });
    assert.ok(result.out instanceof Storage);
    assert.deepEqual(
      [...result.out],
      [input[0], input[6], input[1], input[7], input[2], input[8],
        input[3], input[9], input[4], input[10], input[5], input[11]],
    );
  }
});

test('typed Transpose rejects descriptor, shape, and permutation changes', () => {
  const descriptorMismatch = transposeGraph('uint8', {
    outputQuantization: {
      scheme: 'per_tensor', scale: 0.25, zero_point: 123,
    },
  });
  assert.throws(
    () => new CPUEngine().allocateGraph(descriptorMismatch),
    /descriptor-preserving rank-1\.\.8 I8\/U8 permutation/,
  );

  const shapeMismatch = transposeGraph('uint8', {
    outputShape: [1, 3, 2, 2],
  });
  assert.throws(
    () => new CPUEngine().allocateGraph(shapeMismatch),
    /descriptor-preserving rank-1\.\.8 I8\/U8 permutation/,
  );

  const duplicateAxis = transposeGraph('int8', {
    perm: [0, 2, 2, 1],
  });
  assert.throws(
    () => new CPUEngine().allocateGraph(duplicateAxis),
    /descriptor-preserving rank-1\.\.8 I8\/U8 permutation/,
  );
});
