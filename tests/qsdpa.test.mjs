import test from 'node:test';
import assert from 'node:assert/strict';

import { RuntimeGraph } from '../ts/core/RuntimeGraph.js';
import { _cpuQSDPA } from '../ts/ops/qSDPA.js';

function byteStorage(dtype, values) {
  return dtype === 'int8' ? Int8Array.from(values) : Uint8Array.from(values);
}

function valuesForCentered(dtype, zeroPoint, centered) {
  return centered.map((value) => value + zeroPoint);
}

function elements(shape) {
  return shape.reduce((product, dimension) => product * dimension, 1);
}

function qSDPAGraph({
  qShape = [2, 4],
  qDtype = 'int8',
  qValues = [0, -1, -2, 1, -2, 1, 0, -1],
  qQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  kShape = [2, 4],
  kDtype = 'int8',
  kValues = [0, 0, -1, -1, -1, 0, 0, -2],
  kQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  vShape = [2, 4],
  vDtype = 'int8',
  vValues = [3, -3, 0, -1, -5, 1, 2, -2],
  vQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: -1 },
  outputDtype = 'int8',
  outputQuantization = { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
  maskShape,
  maskValues,
  heads = 1,
  causal = false,
  scale = 0.5,
  outputFill = 0,
} = {}) {
  const graph = new RuntimeGraph();
  const q = graph.addInput('q', qShape, qDtype, {
    buffer: byteStorage(qDtype, qValues), quantization: qQuantization,
  });
  const k = graph.addInput('k', kShape, kDtype, {
    buffer: byteStorage(kDtype, kValues), quantization: kQuantization,
  });
  const v = graph.addInput('v', vShape, vDtype, {
    buffer: byteStorage(vDtype, vValues), quantization: vQuantization,
  });
  const inputs = { q, k, v };
  let mask = null;
  if (maskShape !== undefined) {
    mask = graph.addInput('mask', maskShape, 'int32', { buffer: Int32Array.from(maskValues) });
    inputs.mask = mask;
  }
  const params = { heads, causal };
  if (scale !== undefined) params.scale = scale;
  const { out } = graph.addOp('QSDPA', inputs, {
    out: {
      name: 'out', shape: qShape, dtype: outputDtype,
      buffer: byteStorage(outputDtype, new Array(elements(qShape)).fill(outputFill)),
      quantization: outputQuantization,
    },
  }, params);
  return { graph, node: graph.nodes[0], q, k, v, mask, out };
}

test('QSDPA computes the canonical I8 fixture with noncausal and causal byte outputs', () => {
  const noncausal = qSDPAGraph();
  _cpuQSDPA(noncausal.node);
  assert.ok(noncausal.out.buffer instanceof Int8Array);
  assert.deepEqual([...noncausal.out.buffer], [0, 0, 2, 0, 0, 0, 2, -1]);

  const causal = qSDPAGraph({ causal: true });
  _cpuQSDPA(causal.node);
  assert.deepEqual([...causal.out.buffer], [4, -2, 1, 0, 0, 0, 2, -1]);

  const explicitNullScale = qSDPAGraph({ scale: null });
  _cpuQSDPA(explicitNullScale.node);
  assert.deepEqual([...explicitNullScale.out.buffer], [0, 0, 2, 0, 0, 0, 2, -1]);
});

test('QSDPA supports independently typed I8/U8 q, k, v, and output tensors', () => {
  const centeredQ = [1, -2, 3, -4];
  const centeredK = [-1, 2, -3, 4];
  const centeredV = [1, -2, 3, -4];
  const byteTypes = [
    { dtype: 'int8', zeroPoint: -1 },
    { dtype: 'uint8', zeroPoint: 128 },
  ];
  for (const qType of byteTypes) {
    for (const kType of byteTypes) {
      for (const vType of byteTypes) {
        for (const outType of byteTypes) {
          const { node, out } = qSDPAGraph({
            qShape: [1, 4], qDtype: qType.dtype,
            qValues: valuesForCentered(qType.dtype, qType.zeroPoint, centeredQ),
            qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: qType.zeroPoint },
            kShape: [1, 4], kDtype: kType.dtype,
            kValues: valuesForCentered(kType.dtype, kType.zeroPoint, centeredK),
            kQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: kType.zeroPoint },
            vShape: [1, 4], vDtype: vType.dtype,
            vValues: valuesForCentered(vType.dtype, vType.zeroPoint, centeredV),
            vQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: vType.zeroPoint },
            outputDtype: outType.dtype,
            outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: outType.zeroPoint },
            scale: undefined,
          });
          _cpuQSDPA(node);
          assert.ok(outType.dtype === 'int8' ? out.buffer instanceof Int8Array : out.buffer instanceof Uint8Array);
          assert.deepEqual([...out.buffer], centeredV.map((value) => value * 2 + outType.zeroPoint));
        }
      }
    }
  }
});

test('QSDPA honors asymmetric U8 cross descriptors and mask keep semantics', () => {
  const base = {
    qShape: [1, 4], qDtype: 'uint8', qValues: [129, 126, 131, 128],
    qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 128 },
    kShape: [2, 4], kDtype: 'uint8', kValues: [121, 120, 119, 122, 119, 123, 120, 118],
    kQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 120 },
    vShape: [2, 4], vDtype: 'uint8', vValues: [134, 126, 132, 130, 128, 132, 136, 128],
    vQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 130 },
    outputDtype: 'uint8', outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 127 },
  };
  const noMask = qSDPAGraph(base);
  _cpuQSDPA(noMask.node);
  assert.deepEqual([...noMask.out.buffer], [128, 126, 131, 126]);

  const keepSecond = qSDPAGraph({ ...base, maskShape: [2], maskValues: [0, 1] });
  _cpuQSDPA(keepSecond.node);
  assert.deepEqual([...keepSecond.out.buffer], [125, 129, 133, 125]);

  const none = qSDPAGraph({ ...base, maskShape: [2], maskValues: [0, 0] });
  _cpuQSDPA(none.node);
  assert.deepEqual([...none.out.buffer], [127, 127, 127, 127]);
});

test('QSDPA accepts every existing I32 mask layout and writes zero point for all-masked rows', () => {
  const qShape = [2, 2, 4];
  const qValues = new Array(elements(qShape)).fill(0);
  const kValues = new Array(16).fill(0);
  const vValues = [
    4, 0, 0, 0, 0, 4, 0, 0,
    4, 0, 0, 0, 0, 4, 0, 0,
  ];
  const base = {
    qShape, qValues, qQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    kShape: [2, 2, 4], kValues, kQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    vShape: [2, 2, 4], vValues, vQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: -7 }, scale: 0.5,
  };
  const cases = [
    { shape: [2], values: [2, 0], expected: [-3, -7, -7, -7, -3, -7, -7, -7, -3, -7, -7, -7, -3, -7, -7, -7] },
    { shape: [2, 2], values: [1, 0, 0, 3], expected: [-3, -7, -7, -7, -3, -7, -7, -7, -7, -3, -7, -7, -7, -3, -7, -7] },
    // B==Q is deliberately [B,K] by the existing attention-mask precedence.
    { shape: [2, 2], values: [1, 0, 0, 5], expected: [-3, -7, -7, -7, -3, -7, -7, -7, -7, -3, -7, -7, -7, -3, -7, -7] },
    { shape: [2, 2, 2], values: [1, 0, 0, 1, 0, 1, 1, 0], expected: [-3, -7, -7, -7, -7, -3, -7, -7, -7, -3, -7, -7, -3, -7, -7, -7] },
  ];
  for (const { shape, values, expected } of cases) {
    const { node, out } = qSDPAGraph({ ...base, maskShape: shape, maskValues: values });
    _cpuQSDPA(node);
    assert.deepEqual([...out.buffer], expected);
  }

  const qkMask = qSDPAGraph({
    ...base, qShape: [2, 3, 4], qValues: new Array(2 * 3 * 4).fill(0),
    maskShape: [3, 2], maskValues: [1, 0, 0, 1, 1, 0],
  });
  _cpuQSDPA(qkMask.node);
  assert.deepEqual([...qkMask.out.buffer], [
    -3, -7, -7, -7, -7, -3, -7, -7, -3, -7, -7, -7,
    -3, -7, -7, -7, -7, -3, -7, -7, -3, -7, -7, -7,
  ]);

  const fullyMasked = qSDPAGraph({ ...base, maskShape: [2], maskValues: [0, 0] });
  _cpuQSDPA(fullyMasked.node);
  assert.deepEqual([...fullyMasked.out.buffer], new Array(elements(qShape)).fill(-7));
});

test('QSDPA supports the v1 head_dim=64 maximum without F32 graph activation storage', () => {
  const dModel = 320;
  const qValues = Array.from({ length: dModel }, (_, index) => (index % 5) - 2);
  const kValues = Array.from({ length: dModel }, (_, index) => (index % 7) - 3);
  const vValues = Array.from({ length: dModel }, (_, index) => (index % 31) - 15);
  const { node, out } = qSDPAGraph({
    qShape: [1, dModel], qValues, kShape: [1, dModel], kValues, vShape: [1, dModel], vValues,
    qQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    kQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    vQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 0.25, zero_point: 0 }, heads: 5,
  });
  _cpuQSDPA(node);
  assert.ok(out.buffer instanceof Int8Array);
  assert.deepEqual([...out.buffer], vValues);
});

test('QSDPA requantizes ties to even and saturates typed output', () => {
  const { node, out } = qSDPAGraph({
    qShape: [1, 4], qValues: [1, 0, 0, 0], kShape: [1, 4], kValues: [1, 0, 0, 0],
    vShape: [1, 4], vValues: [1, 3, -3, 127],
    qQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    kQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    vQuantization: { scheme: 'per_tensor', scale: 0.5, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  _cpuQSDPA(node);
  assert.deepEqual([...out.buffer], [0, 2, -2, 64]);

  // Use a separate graph to exercise saturation with a distinct V descriptor.
  const saturating = qSDPAGraph({
    qShape: [1, 4], qValues: [1, 0, 0, 0], kShape: [1, 4], kValues: [1, 0, 0, 0],
    vShape: [1, 4], vValues: [127, -128, 0, 0],
    qQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    kQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
    vQuantization: { scheme: 'per_tensor', scale: 1000, zero_point: 0 },
    outputQuantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
  });
  _cpuQSDPA(saturating.node);
  assert.deepEqual([...saturating.out.buffer], [127, -128, 0, 0]);
});

test('QSDPA rejects malformed descriptors and aliases before writing output', () => {
  const { node, out } = qSDPAGraph({ outputFill: 73 });
  node.params.heads = 3;
  assert.throws(() => _cpuQSDPA(node), /D\/head dimensions divisible by 4/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  node.params.heads = 1;
  node.params.scale = 3e38;
  assert.throws(() => _cpuQSDPA(node), /score range is not representable/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  node.params.scale = 0.5;
  const aliasNode = { ...node, outputs: { out: node.inputs.q } };
  assert.throws(() => _cpuQSDPA(aliasNode), /output storage distinct/);
  assert.deepEqual([...out.buffer], new Array(out.buffer.length).fill(73));

  const malformedMask = qSDPAGraph({ maskShape: [3], maskValues: [1, 1, 1], outputFill: 73 });
  assert.throws(() => _cpuQSDPA(malformedMask.node), /mask must have shape/);
  assert.deepEqual([...malformedMask.out.buffer], new Array(malformedMask.out.buffer.length).fill(73));
});
