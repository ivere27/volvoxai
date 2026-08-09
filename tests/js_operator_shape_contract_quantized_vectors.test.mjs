import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  OperatorShapeContractError,
  QUANTIZED_OPERATOR_SHAPE_FUNCTIONS,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

const VECTOR_PATH = new URL('./operator_shape_contract_quantized_vectors.json', import.meta.url);
const EXPECTED_OPERATORS = new Set([
  'QLinear', 'QMatMul', 'QGemm', 'QBatchMatMul', 'QConv2D', 'QAdd',
  'QEmbedding', 'QGELU', 'QSiLU', 'QLayerNorm', 'QGroupNorm',
  'QMaskedMean', 'QSDPA', 'QArgMax',
]);

function canonicalDescriptor(descriptor) {
  const quantization = descriptor.quantization == null
    ? undefined
    : descriptor.quantization.scheme === 'per_tensor'
      ? { ...descriptor.quantization, scale: Math.fround(descriptor.quantization.scale) }
      : {
          ...descriptor.quantization,
          scales: descriptor.quantization.scales.map((scale) => Math.fround(scale)),
        };
  return {
    shape: [...descriptor.shape],
    dtype: descriptor.dtype,
    ...(quantization === undefined ? {} : { quantization }),
  };
}

function loadCorpus() {
  const corpus = JSON.parse(readFileSync(VECTOR_PATH, 'utf8'));
  assert.deepEqual(Object.keys(corpus).sort(), [
    'failure_cases', 'families', 'format', 'success_cases',
  ]);
  assert.equal(corpus.format, 'volvox-operator-shape-vectors/v1');
  const manifest = new Set(corpus.families.flatMap((family) => family.operators));
  assert.deepEqual(manifest, EXPECTED_OPERATORS);
  assert.deepEqual(new Set(Object.keys(QUANTIZED_OPERATOR_SHAPE_FUNCTIONS)), EXPECTED_OPERATORS);
  const legal = new Map([...manifest].map((operator) => [operator, 0]));
  const illegal = new Map([...manifest].map((operator) => [operator, 0]));
  for (const [vectors, coverage] of [
    [corpus.success_cases, legal],
    [corpus.failure_cases, illegal],
  ]) {
    for (const vector of vectors) {
      for (const operator of vector.operators) {
        coverage.set(operator, coverage.get(operator) + 1);
      }
    }
  }
  for (const operator of manifest) {
    assert.ok(legal.get(operator) >= 2, operator + ' needs two legal vectors');
    assert.ok(illegal.get(operator) >= 1, operator + ' needs one illegal vector');
  }
  return corpus;
}

const corpus = loadCorpus();

test('shared quantized legal vectors converge in concrete and bounded TS contracts', () => {
  for (const original of corpus.success_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      const before = structuredClone(vector);
      const contract = getOperatorShapeContract(operator);
      assert.equal(contract.shapeFunctionId, vector.expected_shape_function_id);
      const expected = Object.fromEntries(
        Object.entries(vector.expected).map(([name, descriptor]) =>
          [name, canonicalDescriptor(descriptor)]),
      );
      const outputs = inferConcreteOperatorShapes(operator, vector.request);
      assert.deepEqual(outputs, expected, vector.id + ':' + operator + ' concrete');
      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, true,
        vector.id + ':' + operator + ': ' + (proof.supported ? '' : proof.reason));
      assert.deepEqual(proof.outputs, expected, vector.id + ':' + operator + ' domain');
      assert.deepEqual(vector, before, vector.id + ':' + operator + ' mutated its vector');
    }
  }
});

test('shared quantized illegal vectors have exact concrete codes and paths', () => {
  for (const original of corpus.failure_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      assert.throws(
        () => inferConcreteOperatorShapes(operator, vector.request),
        (error) => {
          assert.ok(error instanceof OperatorShapeContractError);
          assert.equal(error.code, vector.expected_error.code, vector.id + ':' + operator);
          assert.equal(error.path, vector.expected_error.path, vector.id + ':' + operator);
          return true;
        },
      );
      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, false, vector.id + ':' + operator + ' domain accepted invalid input');
    }
  }
});

test('quantized bounded proofs preserve dynamic B/S/Q/K/H/W axes', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'S', min: 1, max: 32 },
    { name: 'Q', min: 1, max: 16 },
    { name: 'K', min: 1, max: 24 },
    { name: 'H', min: 3, max: 32 },
    { name: 'W', min: 3, max: 32 },
  ]);
  const pt = (scale = 0.05, zero_point = 0) => ({
    scheme: 'per_tensor', scale, zero_point,
  });
  const pa = (count) => ({
    scheme: 'per_axis', axis: 0,
    scales: Array.from({ length: count }, (_, index) => 0.01 * (index + 1)),
    zero_points: Array(count).fill(0),
  });
  const byte = (shape, quantization = pt()) => ({ shape, dtype: 'int8', quantization });
  const output = (shape) => ({ out: byte(shape, pt(0.08)) });
  const i32 = (shape) => ({ shape, dtype: 'int32' });
  const f32 = (shape) => ({ shape, dtype: 'float32' });
  const cases = [
    ...['QLinear', 'QMatMul', 'QGemm'].map((operator) => [operator, {
      inputs: { input: byte(['B', 'S', 4]), weight: byte([3, 4], pa(3)), bias: i32([3]) },
      params: {}, declaredOutputs: output(['B', 'S', 3]),
    }]),
    ['QBatchMatMul', {
      inputs: { a: byte(['B', 'Q', 4]), b: byte([1, 4, 3]) },
      params: {}, declaredOutputs: output(['B', 'Q', 3]),
    }],
    ['QConv2D', {
      inputs: { input: byte(['B', 'H', 'W', 2]), weight: byte([3, 3, 3, 2], pa(3)) },
      params: {
        stride: [1, 1], padding: [1, 1], pads: [1, 1, 1, 1],
        data_layout: 'NHWC', weight_layout: 'OHWI',
      },
      declaredOutputs: output(['B', 'H', 'W', 3]),
    }],
    ['QAdd', {
      inputs: { a: byte(['B', 'S', 4]), b: byte(['B', 'S', 4]) },
      params: { relu: 0 }, declaredOutputs: output(['B', 'S', 4]),
    }],
    ['QEmbedding', {
      inputs: { input: i32(['B', 'S']), weight: byte([5, 4], pa(5)) },
      params: {}, declaredOutputs: output(['B', 'S', 4]),
    }],
    ['QGELU', {
      inputs: { input: byte(['B', 'S', 4]) }, params: { approximate: 'none' },
      declaredOutputs: output(['B', 'S', 4]),
    }],
    ['QSiLU', {
      inputs: { input: byte(['B', 'S', 4]) }, params: {},
      declaredOutputs: output(['B', 'S', 4]),
    }],
    ['QLayerNorm', {
      inputs: { input: byte(['B', 'S', 4]), weight: f32([4]), bias: f32([4]) },
      params: { eps: 1e-5, d_model: 4 }, declaredOutputs: output(['B', 'S', 4]),
    }],
    ['QGroupNorm', {
      inputs: { input: byte(['B', 'H', 'W', 4]), weight: f32([4]), bias: f32([4]) },
      params: { num_groups: 2, eps: 1e-5, data_layout: 'NHWC' },
      declaredOutputs: output(['B', 'H', 'W', 4]),
    }],
    ['QMaskedMean', {
      inputs: { input: byte(['B', 'S', 4]), mask: i32(['B', 'S']) },
      params: {}, declaredOutputs: output(['B', 4]),
    }],
    ['QSDPA', {
      inputs: {
        q: byte(['B', 'Q', 8]), k: byte(['B', 'K', 8]), v: byte(['B', 'K', 8]),
        mask: i32(['B', 'Q', 'K']),
      },
      params: { heads: 2, causal: false, scale: 0.5 },
      declaredOutputs: output(['B', 'Q', 8]),
    }],
    ['QArgMax', {
      inputs: { input: byte(['B', 'S', 7]) }, params: { axis: -1 },
    }],
  ];
  for (const [operator, request] of cases) {
    const proof = proveOperatorShapeDomain(operator, { environment, ...request });
    assert.equal(proof.supported, true,
      operator + ': ' + (proof.supported ? '' : proof.reason));
  }
});

test('quantized domain proof rejects an unprovable dynamic contraction', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'D', min: 4, max: 8, multiple_of: 4 },
  ]);
  const pt = { scheme: 'per_tensor', scale: 0.05, zero_point: 0 };
  const proof = proveOperatorShapeDomain('QLinear', {
    environment,
    inputs: {
      input: { shape: ['B', 'D'], dtype: 'int8', quantization: pt },
      weight: {
        shape: [3, 4], dtype: 'int8',
        quantization: {
          scheme: 'per_axis', axis: 0,
          scales: [0.01, 0.02, 0.03], zero_points: [0, 0, 0],
        },
      },
      bias: { shape: [3], dtype: 'int32' },
    },
    params: {},
    declaredOutputs: {
      out: { shape: ['B', 3], dtype: 'int8', quantization: pt },
    },
  });
  assert.equal(proof.supported, false);
  assert.equal(proof.code, 'UNPROVABLE_DYNAMIC_CONTRACTION');
});

test('quantized domain proofs enforce exact I32 and F32 transform boundaries', () => {
  const perTensor = (scale = 1, zero_point = 0) => ({
    scheme: 'per_tensor', scale, zero_point,
  });
  const byte = (shape, quantization = perTensor(), dtype = 'uint8') => ({
    shape, dtype, quantization,
  });
  const i32 = (shape) => ({ shape, dtype: 'int32' });
  const batchRequest = (maximumK) => ({
    environment: new ShapeEnvironment([{ name: 'K', min: 1, max: maximumK }]),
    inputs: {
      a: byte([1, 1, 'K']),
      b: byte([1, 'K', 1]),
    },
    params: {},
    declaredOutputs: { out: byte([1, 1, 1]) },
  });
  const safeBatch = proveOperatorShapeDomain('QBatchMatMul', batchRequest(33_025));
  assert.equal(safeBatch.supported, true,
    safeBatch.supported ? '' : safeBatch.reason);
  const unsafeBatch = proveOperatorShapeDomain('QBatchMatMul', batchRequest(33_026));
  assert.equal(unsafeBatch.supported, false);
  assert.equal(unsafeBatch.code, 'INVALID_DOMAIN');

  const maskedMeanRequest = (maximumS, inputScale = 1, outputScale = 1) => ({
    environment: new ShapeEnvironment([{ name: 'S', min: 1, max: maximumS }]),
    inputs: {
      input: byte([1, 'S', 1], perTensor(inputScale)),
      mask: i32([1, 'S']),
    },
    params: {},
    declaredOutputs: { out: byte([1, 1], perTensor(outputScale)) },
  });
  const safeMean = proveOperatorShapeDomain('QMaskedMean', maskedMeanRequest(8_421_504));
  assert.equal(safeMean.supported, true, safeMean.supported ? '' : safeMean.reason);
  const unsafeMean = proveOperatorShapeDomain('QMaskedMean', maskedMeanRequest(8_421_505));
  assert.equal(unsafeMean.supported, false);
  assert.equal(unsafeMean.code, 'INVALID_DOMAIN');

  const minimumF32 = 1.401298464324817e-45;
  const maximumF32 = 3.4028234663852886e38;
  const underflowMean = proveOperatorShapeDomain(
    'QMaskedMean', maskedMeanRequest(1, minimumF32, maximumF32),
  );
  assert.equal(underflowMean.supported, false);
  assert.equal(underflowMean.code, 'UNSAFE_QUANTIZATION_TRANSFORM');

  const attentionRequest = (qScale, kScale) => ({
    environment: new ShapeEnvironment([
      { name: 'B', min: 1, max: 2 },
      { name: 'Q', min: 1, max: 2 },
      { name: 'K', min: 1, max: 3 },
    ]),
    inputs: {
      q: byte(['B', 'Q', 8], perTensor(qScale), 'int8'),
      k: byte(['B', 'K', 8], perTensor(kScale), 'int8'),
      v: byte(['B', 'K', 8], perTensor(1), 'int8'),
      mask: i32(['B', 'Q', 'K']),
    },
    params: { heads: 2, causal: false, scale: 0.5 },
    declaredOutputs: {
      out: byte(['B', 'Q', 8], perTensor(1), 'int8'),
    },
  });
  const safeAttention = proveOperatorShapeDomain('QSDPA', attentionRequest(1, 1));
  assert.equal(safeAttention.supported, true,
    safeAttention.supported ? '' : safeAttention.reason);
  const overflowAttention = proveOperatorShapeDomain(
    'QSDPA', attentionRequest(maximumF32, maximumF32),
  );
  assert.equal(overflowAttention.supported, false);
  assert.equal(overflowAttention.code, 'UNSAFE_QUANTIZATION_TRANSFORM');
});
