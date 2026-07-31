import test from 'node:test';
import assert from 'node:assert/strict';

import * as inference from '../ts/index.js';
import {
  Graph,
  GraphLoader,
  PTQCalibrator,
  PTQObserver,
  SafetensorsFile,
  VolvoxAI,
  derivePTQParameters,
  materializePTQWeights,
  packPTQBias,
  packPTQWeight,
  quantizePTQ,
} from '../ts/full.js';
import { CPUEngine } from '../ts/backends/CPUEngine.js';

function close(left, right, tolerance = 1e-7) {
  return Math.abs(left - right) <= tolerance * (1 + Math.abs(left) + Math.abs(right));
}

test('PTQ observer accumulates ranges and derives symmetric/asymmetric affine parameters', () => {
  const observer = new PTQObserver();
  observer.observe(Float32Array.of(-2, 1));
  observer.observe(Float32Array.of(3, -1));
  assert.deepEqual(observer.snapshot(), { minimum: -2, maximum: 3, sampleCount: 4 });

  const symmetric = observer.parameters({ dtype: 'int8', scheme: 'symmetric' });
  assert.equal(symmetric.zero_point, 0);
  assert.ok(close(symmetric.scale, Math.fround(3 / 127)));
  assert.deepEqual([...quantizePTQ(Float32Array.of(-3, 0, 3), symmetric).data], [-127, 0, 127]);

  const unitScale = derivePTQParameters(
    { minimum: -127, maximum: 127, sampleCount: 2 },
    { dtype: 'int8', scheme: 'symmetric' },
  );
  const fullRange = quantizePTQ(Float32Array.of(-129, -128, 127, 128), unitScale);
  assert.deepEqual([...fullRange.data], [-128, -128, 127, 127]);
  assert.equal(fullRange.saturationCount, 2,
    'generic I8 quantization uses the full physical [-128,127] range');

  const asymmetric = derivePTQParameters(
    { minimum: -1, maximum: 3, sampleCount: 4 },
    { dtype: 'uint8', scheme: 'asymmetric' },
  );
  assert.equal(asymmetric.zero_point, 64);
  assert.ok(close(asymmetric.scale, Math.fround(4 / 255)));
  assert.deepEqual([...quantizePTQ(Float32Array.of(-1, 0, 3), asymmetric).data], [0, 64, 255]);

  const before = observer.snapshot();
  assert.throws(() => observer.observe(Float32Array.of(1, NaN)), /non-finite/);
  assert.deepEqual(observer.snapshot(), before, 'a rejected observation must not partially update the range');
  observer.reset();
  assert.throws(() => observer.snapshot(), /no observations/);
});

test('PTQ packs canonical per-output-channel I8 weights and accumulator-domain bias', () => {
  const packed = packPTQWeight(Float32Array.of(-1, 0, 1, -2, 0, 2), [2, 3]);
  assert.equal(packed.quantization.scheme, 'per_axis');
  assert.equal(packed.quantization.axis, 0);
  assert.deepEqual(packed.quantization.zero_points, [0, 0]);
  assert.ok(close(packed.scales[0], Math.fround(1 / 127)));
  assert.ok(close(packed.scales[1], Math.fround(2 / 127)));
  assert.deepEqual([...packed.data], [-127, 0, 127, -127, 0, 127]);
  assert.equal(packed.saturationCount, 0);

  const narrow = packPTQWeight(Float32Array.of(-128, 128), [1, 2]);
  assert.deepEqual([...narrow.data], [-127, 127],
    'symmetric packed weights reserve -128 and use narrow range [-127,127]');

  const bias = packPTQBias(Float32Array.of(0.125, -0.5), 0.5, Float32Array.of(0.25, 0.5));
  assert.deepEqual([...bias], [1, -2]);
  assert.throws(
    () => packPTQWeight(Float32Array.of(1, NaN), [1, 2]),
    /non-finite/,
  );
});

test('PTQ calibrator observes declared runtime result snapshots across forwards', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [2], 'float32');
  const { out: output } = graph.addOp('ReLU', { input }, {
    out: { name: 'output', shape: [2], dtype: 'float32' },
  });
  graph.setOutputs([output.name]);

  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const model = runtime.createModel(graph);
  const compiled = await model.compile({
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  const calibrator = new PTQCalibrator();
  try {
    for (const sample of [
      { input: Float32Array.of(-1, 2) },
      { input: Float32Array.of(3, -4) },
    ]) {
      const result = await context.execute(sample);
      try {
        const values = await result.output('output').read();
        assert.ok(values instanceof Float32Array);
        calibrator.observe('output', values);
      } finally {
        await result.close();
      }
    }
    assert.deepEqual(calibrator.observers.get('output').snapshot(), {
      minimum: 0, maximum: 3, sampleCount: 4,
    });
    assert.equal(calibrator.parameters().output.zero_point, 0);
  } finally {
    await context.close();
    await compiled.close();
    await model.close();
    await runtime.close();
  }
});

test('PTQ calibrator rejects invalid named observations without mutating state', () => {
  const calibrator = new PTQCalibrator();
  calibrator.observe('output', Float32Array.of(1, 2));
  assert.throws(
    () => calibrator.observe('output', Float32Array.of(1, NaN)),
    /non-finite/,
  );
  assert.deepEqual(calibrator.observers.get('output').snapshot(), {
    minimum: 1, maximum: 2, sampleCount: 2,
  });
  assert.throws(
    () => calibrator.observe('', Float32Array.of(1)),
    /name must be non-empty/,
  );
  assert.deepEqual(calibrator.observers.get('output').snapshot(), {
    minimum: 1, maximum: 2, sampleCount: 2,
  });
});

test('PTQ materialization emits INT8 weights and affine parameters only in safetensors', () => {
  const graph = new Graph();
  graph.addWeight('linear.weight', [2, 3], 'float32', {
    buffer: Float32Array.of(-1, 0, 1, -2, 0, 2),
  });
  graph.addWeight('linear.bias', [2], 'float32', {
    buffer: Float32Array.of(0.5, -1),
  });
  graph.addWeight('norm.weight', [2], 'float32', {
    buffer: Float32Array.of(1, 1),
  });

  const artifact = materializePTQWeights(graph, [{
    name: 'linear.weight',
    bias: 'linear.bias',
    inputScale: 0.5,
  }]);
  const file = SafetensorsFile.fromArrayBuffer(artifact.weights.toArrayBuffer());
  assert.deepEqual(file.listTensorNames(), [
    'norm.weight', 'linear.weight', 'linear.weight.scale',
    'linear.weight.zero_point', 'linear.bias',
  ]);
  assert.equal(file.getTensor('linear.weight').dtype, 'I8');
  assert.equal(file.getTensor('linear.weight.scale').dtype, 'F32');
  assert.equal(file.getTensor('linear.weight.zero_point').dtype, 'I8');
  assert.equal(file.getTensor('linear.bias').dtype, 'I32');
  assert.equal(file.getTensor('norm.weight').dtype, 'F32');
  assert.deepEqual([...file.toRuntimeTypedArray(file.getTensor('linear.weight'))],
    [-127, 0, 127, -127, 0, 127]);
  assert.deepEqual([...file.toRuntimeTypedArray(file.getTensor('linear.bias'))], [127, -127]);
  assert.equal(Object.hasOwn(file.metadata, 'weights_quantization'), false);
  assert.deepEqual(artifact.quantization, {
    format: 'volvox-affine-safetensors/v1',
    tensors: {
      'linear.weight': {
        scheme: 'per_axis', axis: 0,
        scale_tensor: 'linear.weight.scale',
        zero_point_tensor: 'linear.weight.zero_point',
      },
    },
  });
  assert.deepEqual(
    [...file.toRuntimeTypedArray(file.getTensor('linear.weight.zero_point'))],
    [0, 0],
  );
});

test('PTQ materialization rejects graph namespace and prototype collisions', () => {
  const graph = new Graph();
  graph.addInput('input', [2], 'float32');
  graph.addTensor('activation', [2], 'float32');
  graph.addWeight('weight', [2, 2], 'float32', {
    buffer: Float32Array.of(1, 0, 0, 1),
  });
  graph.addWeight('other', [2], 'float32', {
    buffer: Float32Array.of(1, 1),
  });

  assert.throws(
    () => materializePTQWeights(graph, [{ name: 'weight', outputName: 'activation' }]),
    /collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(graph, [{ name: 'weight', scaleName: 'input' }]),
    /PTQ scale .*collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(graph, [{ name: 'weight', outputName: 'other' }]),
    /collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(graph, [{ name: 'weight', outputName: '__proto__' }]),
    /safetensors name/,
  );
});

test('PTQ materialized weights reload and execute through a canonical QLinear graph document', async () => {
  const source = new Graph();
  source.addWeight('linear.weight', [2, 2], 'float32', {
    buffer: Float32Array.of(1, 0, 0, 1),
  });
  source.addWeight('linear.bias', [2], 'float32', {
    buffer: Float32Array.of(0, 0),
  });
  const artifact = materializePTQWeights(source, [{
    name: 'linear.weight',
    bias: 'linear.bias',
    inputScale: 1,
  }]);
  artifact.weights.addTensor('activation.scale', 'F32', [1], Float32Array.of(1));
  artifact.weights.addTensor('activation.zero_point', 'I8', [1], Int8Array.of(0));
  const activationQuantization = {
    scheme: 'per_tensor',
    scale_tensor: 'activation.scale',
    zero_point_tensor: 'activation.zero_point',
  };
  const graphDocument = {
    format: 'volvox-graph/v1',
    inputs: {
      input: {
        shape: [1, 2],
        dtype: 'int8',
      },
    },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        ...artifact.quantization.tensors,
        input: activationQuantization,
        output: activationQuantization,
      },
    },
    nodes: [{
      opType: 'QLinear',
      inputs: { input: 'input', weight: 'linear.weight', bias: 'linear.bias' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'int8' },
    }],
    outputs: ['output'],
  };
  const weights = artifact.weights.toArrayBuffer();
  const fetch = async (url) => url.endsWith('graph.json')
    ? { ok: true, json: async () => graphDocument }
    : { ok: true, arrayBuffer: async () => weights };
  const loaded = await GraphLoader.load(new Graph(), 'model.safetensors', {
    graphUrl: 'graph.json',
    fetch,
  });
  const engine = new CPUEngine();
  engine.allocateGraph(loaded);
  const result = await engine.execute({ input: Int8Array.of(2, -3) });
  assert.deepEqual([...result.output], [2, -3]);
});

test('PTQ authoring APIs stay out of the inference entry', () => {
  assert.equal(inference.PTQObserver, undefined);
  assert.equal(inference.PTQCalibrator, undefined);
  assert.equal(inference.materializePTQWeights, undefined);
});
