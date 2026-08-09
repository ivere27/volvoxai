import test from 'node:test';
import assert from 'node:assert/strict';

import * as inference from '../ts/index.js';
import {
  ModelLoader,
  ModelBuilder,
  Model,
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

function close(left, right, tolerance = 1e-7) {
  return Math.abs(left - right) <= tolerance * (1 + Math.abs(left) + Math.abs(right));
}

function makePTQSnapshot({ features = 3, maximumBatch = 4 } = {}) {
  const weights = [
    {
      name: 'linear.weight',
      dtype: 'float32',
      shape: [2, features],
      data: Float32Array.from({ length: 2 * features }, (_, index) =>
        Math.fround(((index * 5) % 9 - 4) / 4)),
    },
    {
      name: 'linear.bias',
      dtype: 'float32',
      shape: [2],
      data: Float32Array.of(0.5, -1),
    },
    {
      name: 'norm.weight',
      dtype: 'float32',
      shape: [2],
      data: Float32Array.of(1, 1),
    },
  ];
  const builder = new ModelBuilder({
    dimensions: { B: { min: 1, max: maximumBatch } },
    inputs: { input: { dtype: 'float32', shape: ['B', features] } },
    weights: weights.map(({ name, dtype, shape }) => ({ name, dtype, shape })),
    nodes: [{
      id: 'relu',
      opType: 'ReLU',
      inputs: { input: 'input' },
      outputs: {
        out: { tensor: 'activation', dtype: 'float32', shape: ['B', features] },
      },
      params: {},
    }],
    outputs: ['activation'],
  });
  return Model.capture({
    graph: builder.snapshot(),
    weights: Object.fromEntries(weights.map((weight) => [weight.name, weight])),
  });
}

function calibrationBatch(batchId, profile, batchSize, features = 3, sampleOffset = 0) {
  const input = Float32Array.from({ length: batchSize * features }, (_, index) =>
    Math.fround(((index + sampleOffset) % 7 - 3) / 2));
  const activation = Float32Array.from(input, (value) => Math.max(0, value));
  return {
    batchId,
    profile,
    samples: batchSize,
    chunkIndex: 0,
    chunkCount: 1,
    inputs: { input: { data: input, shape: [batchSize, features] } },
    activations: { activation: { data: activation, shape: [batchSize, features] } },
  };
}

function completeCoverage(snapshot) {
  const calibrator = new PTQCalibrator(snapshot, {
    profiles: ['default'],
    activations: ['activation'],
  });
  calibrator.observeBatchChunk(calibrationBatch(
    'default.0', 'default', 1, snapshot.graph.inputs.input.shape[1],
  ));
  return calibrator.coverage();
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
  assert.equal(fullRange.saturationCount, 2);

  const asymmetric = derivePTQParameters(
    { minimum: -1, maximum: 3, sampleCount: 4 },
    { dtype: 'uint8', scheme: 'asymmetric' },
  );
  assert.equal(asymmetric.zero_point, 64);
  assert.ok(close(asymmetric.scale, Math.fround(4 / 255)));
  assert.deepEqual([...quantizePTQ(Float32Array.of(-1, 0, 3), asymmetric).data], [0, 64, 255]);

  const before = observer.snapshot();
  assert.throws(() => observer.observe(Float32Array.of(1, NaN)), /non-finite/);
  assert.deepEqual(observer.snapshot(), before);
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
  assert.deepEqual([...narrow.data], [-127, 127]);
  const bias = packPTQBias(Float32Array.of(0.125, -0.5), 0.5, Float32Array.of(0.25, 0.5));
  assert.deepEqual([...bias], [1, -2]);
  assert.throws(() => packPTQWeight(Float32Array.of(1, NaN), [1, 2]), /non-finite/);
});

test('named shaped calibration tracks profile, signature, symbol, and activation coverage', () => {
  const snapshot = makePTQSnapshot();
  const calibrator = new PTQCalibrator(snapshot, {
    profiles: ['short', 'maximum'],
    activations: ['activation'],
  });
  assert.equal(calibrator.observe, undefined, 'storage-only legacy observation must not exist');
  assert.equal(calibrator.observeBatch, undefined, 'unchunked legacy observation must not exist');

  calibrator.observeBatchChunk(calibrationBatch('short.0', 'short', 1));
  calibrator.observeBatchChunk(calibrationBatch('short.1', 'short', 2, 3, 2));
  let coverage = calibrator.coverage();
  assert.equal(coverage.format, 'volvox.ptq-coverage/v1');
  assert.equal(coverage.logicalFingerprint, snapshot.definitionFingerprint);
  assert.equal(coverage.complete, false);
  assert.equal(coverage.totalBatches, 2);
  assert.equal(coverage.totalSamples, 3);
  assert.deepEqual(coverage.profiles[0].symbols.B, { minimum: 1, maximum: 2 });
  assert.equal(coverage.profiles[0].signatures.length, 2);
  assert.equal(coverage.profiles[0].activationSamples.activation, 9);
  assert.equal(coverage.profiles[1].batches, 0);
  assert.throws(() => calibrator.parameters(), /missing required shape profiles.*maximum/i);
  assert.throws(
    () => materializePTQWeights(snapshot, ['linear.weight'], { coverage }),
    /complete named-profile coverage/,
  );

  const beforeInvalid = calibrator.coverage();
  const invalid = calibrationBatch('maximum.invalid', 'maximum', 4);
  invalid.inputs.input.shape = [2, 6];
  assert.throws(() => calibrator.observeBatchChunk(invalid), /rank|shape|contract|extent|axis/i);
  assert.deepEqual(calibrator.coverage(), beforeInvalid);
  assert.throws(
    () => calibrator.observeBatchChunk(calibrationBatch('unknown.0', 'undeclared', 1)),
    /was not declared/,
  );

  calibrator.observeBatchChunk(calibrationBatch('maximum.0', 'maximum', 4, 3, 4));
  coverage = calibrator.coverage();
  assert.equal(coverage.complete, true);
  assert.equal(coverage.totalBatches, 3);
  assert.equal(coverage.totalSamples, 7);
  assert.deepEqual(coverage.profiles[1].symbols.B, { minimum: 4, maximum: 4 });
  assert.equal(coverage.profiles[1].activationSamples.activation, 12);
  assert.ok(calibrator.parameters().activation.scale > 0);
});

test('PTQ promotion chunks commit one exact logical batch and sample count once', () => {
  const snapshot = makePTQSnapshot();
  const calibrator = new PTQCalibrator(snapshot, {
    profiles: ['short'],
    activations: ['activation', 'input'],
  });
  const first = calibrationBatch('short.chunked', 'short', 2, 3, 2);
  first.chunkCount = 2;
  const inputValues = Float32Array.from(first.inputs.input.data);
  delete first.activations.input;

  calibrator.observeBatchChunk(first);

  let coverage = calibrator.coverage();
  assert.equal(coverage.complete, false);
  assert.equal(coverage.totalBatches, 0);
  assert.equal(coverage.totalSamples, 0);
  assert.deepEqual(coverage.profiles[0].activationSamples, {});
  assert.equal(calibrator.observers.size, 0);
  assert.throws(() => calibrator.parameters(), /incomplete logical batches.*short\.chunked/);

  const second = calibrationBatch('short.chunked', 'short', 2, 3, 2);
  second.chunkIndex = 1;
  second.chunkCount = 2;
  second.activations = {
    input: { data: inputValues, shape: [2, 3] },
  };
  calibrator.observeBatchChunk(second);

  coverage = calibrator.coverage();
  assert.equal(coverage.complete, true);
  assert.equal(coverage.totalBatches, 1);
  assert.equal(coverage.totalSamples, 2);
  assert.equal(coverage.profiles[0].batches, 1);
  assert.equal(coverage.profiles[0].samples, 2);
  assert.deepEqual(coverage.profiles[0].activationSamples, {
    activation: 6,
    input: 6,
  });
  assert.equal(calibrator.observers.get('activation').snapshot().sampleCount, 6);
  assert.equal(calibrator.observers.get('input').snapshot().sampleCount, 6);
  const beforeReuse = calibrator.coverage();
  assert.throws(
    () => calibrator.observeBatchChunk(second),
    /logical batch 'short\.chunked' was already completed/,
  );
  assert.deepEqual(calibrator.coverage(), beforeReuse);
});

test('PTQ repeated chunks fail closed on inconsistent identity, inputs, or partitioning', () => {
  const snapshot = makePTQSnapshot();
  const setup = () => {
    const calibrator = new PTQCalibrator(snapshot, {
      profiles: ['short', 'maximum'],
      activations: ['activation', 'input'],
    });
    const first = calibrationBatch('shared.0', 'short', 1);
    first.chunkCount = 2;
    calibrator.observeBatchChunk(first);
    return calibrator;
  };
  const validSecond = () => {
    const second = calibrationBatch('shared.0', 'short', 1);
    second.chunkIndex = 1;
    second.chunkCount = 2;
    second.activations = {
      input: {
        data: Float32Array.from(second.inputs.input.data),
        shape: [1, 3],
      },
    };
    return second;
  };
  const cases = [
    ['profile', (chunk) => { chunk.profile = 'maximum'; }, /preserve profile/],
    ['samples', (chunk) => { chunk.samples = 2; }, /preserve profile, samples/],
    ['chunk count', (chunk) => {
      chunk.chunkCount = 1;
      chunk.chunkIndex = 0;
    }, /preserve profile, samples, chunk count/],
    ['shape binding', (chunk) => {
      chunk.inputs.input = {
        data: Float32Array.of(-1, 0, 1, 2, 3, 4),
        shape: [2, 3],
      };
      chunk.activations.input = {
        data: Float32Array.of(-1, 0, 1, 2, 3, 4),
        shape: [2, 3],
      };
    }, /shape binding/],
    ['input bytes', (chunk) => {
      chunk.inputs.input.data[0] = Math.fround(chunk.inputs.input.data[0] + 1);
      chunk.activations.input.data[0] = chunk.inputs.input.data[0];
    }, /exact input bytes/],
    ['duplicate chunk index', (chunk) => { chunk.chunkIndex = 0; }, /already observed/],
    ['duplicate activation', (chunk) => {
      chunk.activations = {
        activation: {
          data: Float32Array.from(chunk.inputs.input.data, (value) => Math.max(0, value)),
          shape: [1, 3],
        },
      };
    }, /more than one chunk/],
    ['undeclared activation', (chunk) => {
      chunk.activations = {
        'linear.weight': { data: new Float32Array(6), shape: [2, 3] },
      };
    }, /was not declared/],
  ];
  for (const [label, mutate, expected] of cases) {
    const calibrator = setup();
    const before = calibrator.coverage();
    const chunk = validSecond();
    mutate(chunk);
    assert.throws(() => calibrator.observeBatchChunk(chunk), expected, label);
    assert.deepEqual(calibrator.coverage(), before, label);
    assert.equal(calibrator.observers.size, 0, label);
    calibrator.observeBatchChunk(validSecond());
    assert.equal(calibrator.coverage().totalBatches, 1, label);
  }

  const incomplete = new PTQCalibrator(snapshot, {
    profiles: ['short'],
    activations: ['activation', 'input'],
  });
  const incompleteOnlyChunk = calibrationBatch('incomplete.0', 'short', 1);
  assert.throws(
    () => incomplete.observeBatchChunk(incompleteOnlyChunk),
    /without the exact declared activation universe/,
  );
  assert.equal(incomplete.coverage().totalBatches, 0);
  assert.equal(incomplete.observers.size, 0);
});

test('PTQ shaped-batch observation is atomic across activations and coverage overflow', () => {
  const snapshot = makePTQSnapshot();
  const calibrator = new PTQCalibrator(snapshot, {
    profiles: ['short'],
    activations: ['activation', 'input'],
  });
  const firstValid = calibrationBatch('short.0', 'short', 1);
  firstValid.activations.input = {
    data: Float32Array.from(firstValid.inputs.input.data),
    shape: [...firstValid.inputs.input.shape],
  };
  calibrator.observeBatchChunk(firstValid);
  const beforeFailureCoverage = calibrator.coverage();
  const beforeFailureObserver = calibrator.observers.get('activation').snapshot();
  const beforeFailureInputObserver = calibrator.observers.get('input').snapshot();
  const invalid = calibrationBatch('short.invalid', 'short', 2, 3, 4);
  invalid.activations.input = {
    data: Float32Array.of(0, 1, 2, 3, 4, Number.NaN),
    shape: [2, 3],
  };
  assert.throws(() => calibrator.observeBatchChunk(invalid), /non-finite/);
  assert.deepEqual(calibrator.coverage(), beforeFailureCoverage);
  assert.deepEqual(calibrator.observers.get('activation').snapshot(), beforeFailureObserver);
  assert.deepEqual(calibrator.observers.get('input').snapshot(), beforeFailureInputObserver);

  const overflowing = new PTQCalibrator(snapshot, {
    profiles: ['maximum'],
    activations: ['activation'],
  });
  const first = calibrationBatch('maximum.overflow-base', 'maximum', 1);
  first.samples = Number.MAX_SAFE_INTEGER;
  overflowing.observeBatchChunk(first);
  const beforeOverflowCoverage = overflowing.coverage();
  const beforeOverflowObserver = overflowing.observers.get('activation').snapshot();
  assert.throws(
    () => overflowing.observeBatchChunk(
      calibrationBatch('maximum.overflow', 'maximum', 1, 3, 5),
    ),
    /sample coverage exceeds the safe integer range/,
  );
  assert.deepEqual(overflowing.coverage(), beforeOverflowCoverage);
  assert.deepEqual(overflowing.observers.get('activation').snapshot(), beforeOverflowObserver);
});

test('PTQ materialization requires exact coverage and preserves the logical graph and symbols', () => {
  const snapshot = makePTQSnapshot();
  const calibrator = new PTQCalibrator(snapshot, {
    profiles: ['short', 'maximum'],
    activations: ['activation'],
  });
  calibrator.observeBatchChunk(calibrationBatch('short.0', 'short', 1));
  calibrator.observeBatchChunk(calibrationBatch('maximum.0', 'maximum', 4));
  const coverage = calibrator.coverage();
  const artifact = materializePTQWeights(snapshot, [{
    name: 'linear.weight',
    bias: 'linear.bias',
    inputScale: 0.5,
  }], { coverage });
  const file = SafetensorsFile.fromArrayBuffer(artifact.weights.toArrayBuffer());

  assert.equal(artifact.format, 'volvox.ptq.v1');
  assert.equal(artifact.logicalFingerprint, snapshot.definitionFingerprint);
  assert.equal(artifact.coverage, coverage);
  assert.deepEqual(artifact.logicalGraph.dimensions.B, {
    min: 1,
    max: 4,
    multiple_of: 1,
  });
  assert.deepEqual(artifact.logicalGraph.inputs.input.shape, ['B', 3]);
  assert.deepEqual(file.listTensorNames(), [
    'norm.weight', 'linear.weight', 'linear.weight.scale',
    'linear.weight.zero_point', 'linear.bias',
  ]);
  assert.equal(file.metadata.format, 'volvox.ptq.v1');
  assert.equal(file.metadata.logical_fingerprint, snapshot.definitionFingerprint);
  assert.deepEqual(JSON.parse(file.metadata.profile_coverage), coverage);
  assert.equal(file.getTensor('linear.weight').dtype, 'I8');
  assert.equal(file.getTensor('linear.weight.scale').dtype, 'F32');
  assert.equal(file.getTensor('linear.weight.zero_point').dtype, 'I8');
  assert.equal(file.getTensor('linear.bias').dtype, 'I32');
  assert.equal(file.getTensor('norm.weight').dtype, 'F32');
  assert.deepEqual(
    [...file.toRuntimeTypedArray(file.getTensor('linear.weight.zero_point'))],
    [0, 0],
  );

  const wrongFingerprint = { ...coverage, logicalFingerprint: 'wrong' };
  assert.throws(
    () => materializePTQWeights(snapshot, ['linear.weight'], { coverage: wrongFingerprint }),
    /exact logical fingerprint/,
  );
});

test('PTQ materialization rejects logical namespace and prototype collisions', () => {
  const snapshot = makePTQSnapshot();
  const coverage = completeCoverage(snapshot);
  assert.throws(
    () => materializePTQWeights(snapshot, [{
      name: 'linear.weight', outputName: 'activation',
    }], { coverage }),
    /collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(snapshot, [{
      name: 'linear.weight', scaleName: 'input',
    }], { coverage }),
    /PTQ scale .*collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(snapshot, [{
      name: 'linear.weight', outputName: 'norm.weight',
    }], { coverage }),
    /collides with graph tensor/,
  );
  assert.throws(
    () => materializePTQWeights(snapshot, [{
      name: 'linear.weight', outputName: '__proto__',
    }], { coverage }),
    /safetensors name/,
  );
});

test('materialized weights reload and execute through a dynamic QLinear graph', async () => {
  const source = makePTQSnapshot({ features: 2, maximumBatch: 3 });
  const coverage = completeCoverage(source);
  const artifact = materializePTQWeights(source, [{
    name: 'linear.weight',
    bias: 'linear.bias',
    inputScale: 1,
  }], { coverage });
  artifact.weights.addTensor('activation.scale', 'F32', [1], Float32Array.of(1));
  artifact.weights.addTensor('activation.zero_point', 'I8', [1], Int8Array.of(0));
  const activationQuantization = {
    scheme: 'per_tensor',
    scale_tensor: 'activation.scale',
    zero_point_tensor: 'activation.zero_point',
  };
  const graphDocument = {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 3, multiple_of: 1 } },
    inputs: { input: { shape: ['B', 2], dtype: 'int8' } },
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        ...artifact.quantization.tensors,
        input: activationQuantization,
        output: activationQuantization,
      },
    },
    nodes: [{
      id: 'linear',
      opType: 'QLinear',
      inputs: { input: 'input', weight: 'linear.weight', bias: 'linear.bias' },
      outputs: { out: { tensor: 'output', dtype: 'int8', shape: ['B', 2] } },
      params: {},
    }],
    outputs: ['output'],
  };
  const weights = artifact.weights.toArrayBuffer();
  const fetch = async (url) => url.endsWith('graph.json')
    ? { ok: true, json: async () => graphDocument }
    : { ok: true, arrayBuffer: async () => weights };
  const snapshot = await Model.load('model.safetensors', {
    graphUrl: 'graph.json',
    fetch,
  });
  const runtime = await VolvoxAI.createRuntime({ backends: ['cpu'] });
  const compiled = await runtime.compile(snapshot, {
    backend: { mode: 'require', backend: 'cpu', operatorFallback: 'forbid' },
  });
  const context = await compiled.createContext();
  try {
    const result = await context.execute({
      input: { data: Int8Array.of(2, -3, -1, 4), shape: [2, 2] },
    });
    try {
      const output = await result.output('output').read();
      assert.equal(output.length, 4);
      assert.deepEqual(result.report.shapeSignature, 'v1|5:input|2:2,2');
    } finally {
      await result.close();
    }
  } finally {
    await context.close();
    await compiled.close();
    await runtime.close();
  }
});

test('PTQ authoring stays out of inference and accepts no concrete Graph compatibility path', () => {
  assert.equal(inference.PTQObserver, undefined);
  assert.equal(inference.PTQCalibrator, undefined);
  assert.equal(inference.materializePTQWeights, undefined);
  assert.equal(inference.RuntimeGraph, undefined);
  assert.equal(inference.RuntimeGraphBuilder, undefined);
  assert.throws(
    () => new PTQCalibrator(null, { profiles: ['default'], activations: ['activation'] }),
    /Model/,
  );
  assert.throws(
    () => new PTQCalibrator(makePTQSnapshot(), { profiles: [], activations: ['activation'] }),
    /at least one named shape profile/,
  );
  assert.throws(
    () => new PTQCalibrator(makePTQSnapshot(), { profiles: ['default'], activations: [] }),
    /at least one non-empty activation name/,
  );
  assert.throws(
    () => new PTQCalibrator(makePTQSnapshot(), {
      profiles: ['default'], activations: ['activation', 'activation'],
    }),
    /activation names must be unique/,
  );
  assert.throws(
    () => new PTQCalibrator(makePTQSnapshot(), {
      profiles: ['default'], activations: ['missing'],
    }),
    /must name an F32 logical tensor/,
  );
});
