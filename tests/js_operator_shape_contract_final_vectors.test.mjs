import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  FINAL_OPERATOR_SHAPE_FUNCTIONS,
  OperatorShapeContractError,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

const VECTOR_PATH = new URL('./operator_shape_contract_final_vectors.json', import.meta.url);
const EXPECTED_OPERATORS = new Set([
  'MoERouter', 'MoELinear', 'CrossAttention', 'BatchNorm2D', 'Interpolate1D',
  'Not', 'Mask', 'Broadcast', 'Concat2', 'RequantizeLinear', 'SSMScan',
  'SelectiveScan', 'SpatialSoftargmaxY', 'MeanHeight', 'ProfileX', 'ProfileY',
  'Dropout',
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

const corpus = JSON.parse(readFileSync(VECTOR_PATH, 'utf8'));

test('final-v1 shared corpus is closed over all formerly deferred operators', () => {
  assert.equal(corpus.format, 'volvox-operator-shape-vectors/v1');
  const manifest = new Set(corpus.families.flatMap((family) => family.operators));
  assert.deepEqual(manifest, EXPECTED_OPERATORS);
  assert.deepEqual(new Set(Object.keys(FINAL_OPERATOR_SHAPE_FUNCTIONS)), EXPECTED_OPERATORS);
  const success = new Map([...manifest].map((operator) => [operator, 0]));
  const failure = new Map([...manifest].map((operator) => [operator, 0]));
  for (const [vectors, coverage] of [
    [corpus.success_cases, success], [corpus.failure_cases, failure],
  ]) {
    for (const vector of vectors) {
      for (const operator of vector.operators) coverage.set(operator, coverage.get(operator) + 1);
    }
  }
  for (const operator of manifest) {
    assert.ok(success.get(operator) >= 1, `${operator} needs a legal vector`);
    assert.ok(failure.get(operator) >= 1, `${operator} needs an illegal vector`);
  }
});

test('final-v1 shared legal vectors converge in concrete and domain contracts', () => {
  for (const original of corpus.success_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      const before = structuredClone(vector);
      const expected = Object.fromEntries(Object.entries(vector.expected).map(
        ([name, descriptor]) => [name, canonicalDescriptor(descriptor)],
      ));
      assert.equal(
        getOperatorShapeContract(operator).shapeFunctionId,
        vector.expected_shape_function_id,
      );
      assert.deepEqual(
        inferConcreteOperatorShapes(operator, vector.request),
        expected,
        `${vector.id}:${operator}:concrete`,
      );
      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, true,
        `${vector.id}:${operator}:${proof.supported ? '' : proof.reason}`);
      assert.deepEqual(proof.outputs, expected, `${vector.id}:${operator}:domain`);
      assert.deepEqual(vector, before, `${vector.id}:${operator}:mutated`);
    }
  }
});

test('final-v1 shared illegal vectors have stable codes and paths', () => {
  for (const vector of corpus.failure_cases) {
    for (const operator of vector.operators) {
      assert.throws(
        () => inferConcreteOperatorShapes(operator, structuredClone(vector.request)),
        (error) => {
          assert.ok(error instanceof OperatorShapeContractError);
          assert.equal(error.code, vector.expected_error.code, `${vector.id}:${operator}`);
          assert.equal(error.path, vector.expected_error.path, `${vector.id}:${operator}`);
          return true;
        },
      );
      const proof = proveOperatorShapeDomain(operator, {
        ...structuredClone(vector.request),
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, false, `${vector.id}:${operator}:domain accepted`);
    }
  }
});

test('final-v1 domain proof preserves dynamic language, vision, and scan axes', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'S', min: 1, max: 32 },
    { name: 'Q', min: 1, max: 16 },
    { name: 'K', min: 1, max: 24 },
    { name: 'H', min: 2, max: 32 },
    { name: 'W', min: 2, max: 32 },
  ]);
  const f32 = (shape) => ({ shape, dtype: 'float32' });
  const i32 = (shape) => ({ shape, dtype: 'int32' });
  const pt = { scheme: 'per_tensor', scale: 0.5, zero_point: 0 };
  const byte = (shape, dtype = 'int8', quantization = pt) => ({ shape, dtype, quantization });
  const cases = [
    ['MoERouter', {
      inputs: { input: f32(['B', 'S', 4]), weight: f32([4, 3]) }, params: { top_k: 2 },
    }, { indices: ['B', 'S', 2], weights: ['B', 'S', 2] }],
    ['MoELinear', {
      inputs: {
        input: f32(['B', 'S', 4]), expert_weight: f32([3, 4, 5]),
        route_indices: f32(['B', 'S', 2]), route_weights: f32(['B', 'S', 2]),
      }, params: {},
    }, { out: ['B', 'S', 5] }],
    ['CrossAttention', {
      inputs: { q: f32(['B', 'Q', 4]), kv: f32(['B', 'K', 4]), weight: f32([12, 4]) },
      params: { heads: 2 },
    }, { out: ['B', 'Q', 4] }],
    ['BatchNorm2D', {
      inputs: {
        input: f32(['B', 'H', 'W', 5]), weight: f32([5]), bias: f32([5]),
        running_mean: f32([5]), running_var: f32([5]),
      }, params: {},
    }, { out: ['B', 'H', 'W', 5] }],
    ['Interpolate1D', { inputs: { input: f32(['B', 3, 'S']) }, params: { size: 7 } },
      { out: ['B', 3, 7] }],
    ['Not', { inputs: { input: i32(['B', 'S']) }, params: {} }, { out: ['B', 'S'] }],
    ['Mask', {
      inputs: { mask: i32(['B', 'S']), a: f32(['B', 'S']), b: f32(['B', 'S']) }, params: {},
    }, { out: ['B', 'S'] }],
    ['Broadcast', { inputs: { input: f32(['B', 1]) }, params: { shape: ['B', 'S'] } },
      { out: ['B', 'S'] }],
    ['Concat2', { inputs: { a: f32(['B', 2]), b: f32(['B', 3]) }, params: { axis: 1 } },
      { out: ['B', 5] }],
    ['RequantizeLinear', {
      inputs: { input: byte(['B', 'S']) }, params: {},
      declaredOutputs: { out: byte(['B', 'S'], 'uint8', { ...pt, zero_point: 128 }) },
    }, { out: ['B', 'S'] }],
    ['SSMScan', {
      inputs: {
        input: f32(['B', 'S', 4]), delta: f32(['B', 'S', 4]), A: f32([4, 5]),
        B: f32([5]), C: f32(['S', 5]),
      }, params: {},
    }, { out: ['B', 'S', 4], state: ['B', 4, 5] }],
    ['SelectiveScan', {
      inputs: {
        input: f32(['B', 'S', 4]), delta: f32(['B', 'S', 4]), A: f32([4, 5]),
        B: f32(['B', 'S', 5]), C: f32([5]),
      }, params: {},
    }, { out: ['B', 'S', 4], state: ['B', 4, 5] }],
    ...['SpatialSoftargmaxY', 'MeanHeight'].map((operator) => [operator, {
      inputs: { input: f32(['B', 'H', 'W', 5]) }, params: {},
    }, { out: ['B', 5, 'W'] }]),
    ['ProfileX', { inputs: { input: f32(['B', 'H', 'W', 5]) }, params: {} },
      { out: ['B', 10, 'W'] }],
    ['ProfileY', { inputs: { input: f32(['B', 'H', 'W', 5]) }, params: {} },
      { out: ['B', 10, 'H'] }],
    ['Dropout', { inputs: { input: f32(['B', 'S', 4]) }, params: { ratio: 0.25 } },
      { out: ['B', 'S', 4] }],
  ];
  for (const [operator, request, expectedShapes] of cases) {
    const proof = proveOperatorShapeDomain(operator, { environment, ...request });
    assert.equal(proof.supported, true,
      `${operator}: ${proof.supported ? '' : proof.reason}`);
    assert.deepEqual(
      Object.fromEntries(Object.entries(proof.outputs).map(([name, output]) => [name, [...output.shape]])),
      expectedShapes,
      operator,
    );
  }
});

test('RequantizeLinear domain proof accepts and rejects exact positive-F32 ratio boundaries', () => {
  const minimumF32 = 1.401298464324817e-45;
  const maximumF32 = 3.4028234663852886e38;
  const request = (inputScale, outputScale) => ({
    environment: new ShapeEnvironment([{ name: 'S', min: 1, max: 32 }]),
    inputs: {
      input: {
        shape: [1, 'S'], dtype: 'int8',
        quantization: { scheme: 'per_tensor', scale: inputScale, zero_point: 0 },
      },
    },
    params: {},
    declaredOutputs: {
      out: {
        shape: [1, 'S'], dtype: 'uint8',
        quantization: { scheme: 'per_tensor', scale: outputScale, zero_point: 128 },
      },
    },
  });

  const safe = proveOperatorShapeDomain('RequantizeLinear', request(maximumF32, maximumF32));
  assert.equal(safe.supported, true, safe.supported ? '' : safe.reason);

  for (const [label, inputScale, outputScale] of [
    ['underflow', minimumF32, maximumF32],
    ['overflow', maximumF32, minimumF32],
  ]) {
    const proof = proveOperatorShapeDomain(
      'RequantizeLinear', request(inputScale, outputScale),
    );
    assert.equal(proof.supported, false, label);
    assert.equal(proof.code, 'INVALID_QUANTIZATION', label);
  }
});
