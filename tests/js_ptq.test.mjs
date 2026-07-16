import test from 'node:test';
import assert from 'node:assert/strict';

import * as inference from '../ts/index.js';
import {
  CPUEngine,
  Graph,
  GraphLoader,
  PTQCalibrator,
  PTQObserver,
  SafetensorsFile,
  calibratePTQ,
  derivePTQParameters,
  materializePTQWeights,
  packPTQBias,
  packPTQWeight,
  quantizePTQ,
} from '../ts/full.js';

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

test('PTQ calibrator observes named CPU-visible graph tensors across forwards', () => {
  const graph = new Graph();
  const input = graph.addInput('input', [2], 'float32', { buffer: Float32Array.of(-1, 2) });
  const value = graph.addTensor('activation', [2], 'float32', { buffer: Float32Array.of(3, -4) });
  const calibrator = new PTQCalibrator();
  calibrator.observeGraph(graph, [input.name, value.name]);
  input.buffer.set([-5, 1]);
  value.buffer.set([2, 6]);
  calibrator.observeGraph(graph, [input.name, value.name]);
  assert.deepEqual(calibrator.observers.get('input').snapshot(), {
    minimum: -5, maximum: 2, sampleCount: 4,
  });
  assert.deepEqual(calibrator.observers.get('activation').snapshot(), {
    minimum: -4, maximum: 6, sampleCount: 4,
  });
  assert.equal(calibrator.parameters().activation.zero_point, 0);
});

test('calibratePTQ runs named samples and observes each host forward', async () => {
  const graph = new Graph();
  const input = graph.addInput('input', [2], 'float32');
  const { out: output } = graph.addOp('ReLU', { input }, {
    out: { name: 'output', shape: [2], dtype: 'float32' },
  });
  graph.outputNames = [output.name];
  const executor = new CPUEngine().allocateGraph(graph);
  const samples = (async function* namedSamples() {
    yield { input: Float32Array.of(-1, 2) };
    yield { input: Float32Array.of(3, -4) };
  }());
  const calibrator = await calibratePTQ(executor, samples, { tensorNames: ['output'] });
  assert.deepEqual(calibrator.observers.get('output').snapshot(), {
    minimum: 0, maximum: 3, sampleCount: 4,
  });

  await assert.rejects(
    calibratePTQ({
      graph,
      capabilities: { outputLocation: 'device' },
      execute: (values) => executor.execute(values),
    }, [
      { input: Float32Array.of(1, 2) },
    ], { graph, tensorNames: ['output'] }),
    /requires a readback mapping/,
  );
});

test('PTQ materialization emits an INT8 safetensors artifact and config fragment', () => {
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
    'norm.weight', 'linear.weight', 'linear.weight.scale', 'linear.bias',
  ]);
  assert.equal(file.getTensor('linear.weight').dtype, 'I8');
  assert.equal(file.getTensor('linear.weight.scale').dtype, 'F32');
  assert.equal(file.getTensor('linear.bias').dtype, 'I32');
  assert.equal(file.getTensor('norm.weight').dtype, 'F32');
  assert.deepEqual([...file.toRuntimeTypedArray(file.getTensor('linear.weight'))],
    [-127, 0, 127, -127, 0, 127]);
  assert.deepEqual([...file.toRuntimeTypedArray(file.getTensor('linear.bias'))], [127, -127]);
  assert.deepEqual(
    JSON.parse(file.metadata.weights_quantization),
    artifact.weightsQuantization,
  );
  assert.deepEqual(artifact.weightsQuantization['linear.weight'].zero_points, [0, 0]);
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

test('PTQ materialized weights reload and execute through a canonical QLinear blueprint', async () => {
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
  const config = {
    inputs: {
      input: {
        shape: [1, 2],
        dtype: 'int8',
        quantization: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      },
    },
    weights_quantization: artifact.weightsQuantization,
    nodes: [{
      op: 'QLinear',
      inputs: { input: 'input', weight: 'linear.weight', bias: 'linear.bias' },
      outputs: { out: 'output' },
      outputs_shape: { out: [1, 2] },
      outputs_dtype: { out: 'int8' },
      outputs_quantization: {
        out: { scheme: 'per_tensor', scale: 1, zero_point: 0 },
      },
    }],
    outputs: ['output'],
  };
  const weights = artifact.weights.toArrayBuffer();
  const fetch = async (url) => url.endsWith('config.json')
    ? { ok: true, json: async () => config }
    : { ok: true, arrayBuffer: async () => weights };
  const loaded = await GraphLoader.load(new Graph(), 'model.safetensors', {
    configUrl: 'config.json',
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
