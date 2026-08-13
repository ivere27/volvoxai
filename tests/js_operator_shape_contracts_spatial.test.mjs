import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  OperatorShapeContractError,
  WAVE_B_OPERATOR_SHAPE_FUNCTIONS,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import { operatorShapeContracts } from '../ts/generated/kernelRegistry.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

const OPERATORS = Object.freeze([
  'BatchMatMul',
  'Conv1D',
  'Conv2D',
  'ConvTranspose2D',
  'MaxPool2D',
  'AveragePool2D',
  'GlobalAveragePool',
  'Resize',
  'ResizeNearest2D',
  'UpsampleNearest2D',
]);

const corpus = JSON.parse(readFileSync(
  new URL('./operator_shape_contract_spatial_vectors.json', import.meta.url),
  'utf8',
));

function tensor(shape, dtype = 'float32', quantization = undefined) {
  return quantization === undefined ? { shape, dtype } : { shape, dtype, quantization };
}

test('spatial shared corpus is strict and covers two legal plus one illegal shape per operator', () => {
  assert.deepEqual(Object.keys(corpus).sort(), ['failure_cases', 'format', 'success_cases']);
  assert.equal(corpus.format, 'volvox-operator-shape-spatial-vectors/v1');
  const successCounts = new Map(OPERATORS.map((operator) => [operator, 0]));
  const failureCounts = new Map(OPERATORS.map((operator) => [operator, 0]));
  const ids = new Set();
  for (const [kind, cases, counts] of [
    ['success', corpus.success_cases, successCounts],
    ['failure', corpus.failure_cases, failureCounts],
  ]) {
    for (const vector of cases) {
      assert.deepEqual(
        Object.keys(vector).sort(),
        (kind === 'success'
          ? ['expected', 'id', 'operator', 'request']
          : ['expected_error', 'id', 'operator', 'request']).sort(),
      );
      assert.equal(ids.has(vector.id), false, `duplicate id ${vector.id}`);
      ids.add(vector.id);
      assert.equal(counts.has(vector.operator), true, vector.operator);
      counts.set(vector.operator, counts.get(vector.operator) + 1);
      assert.equal(JSON.stringify(vector.request).includes('"buffer"'), false);
      assert.equal(JSON.stringify(vector.request).includes('"data"'), false);
    }
  }
  for (const operator of OPERATORS) {
    assert.ok(successCounts.get(operator) >= 2, `${operator} legal coverage`);
    assert.ok(failureCounts.get(operator) >= 1, `${operator} illegal coverage`);
  }
});

test('spatial shared success vectors match immutable TypeScript inference and generated routes', () => {
  for (const vector of corpus.success_cases) {
    assert.equal(operatorShapeContracts[vector.operator].classification, 'canonical');
    assert.equal(
      getOperatorShapeContract(vector.operator).shapeFunctionId,
      WAVE_B_OPERATOR_SHAPE_FUNCTIONS[vector.operator],
    );
    const request = structuredClone(vector.request);
    const before = structuredClone(request);
    const output = inferConcreteOperatorShapes(vector.operator, request);
    assert.deepEqual(output, vector.expected, vector.id);
    assert.deepEqual(request, before, `${vector.id} mutated its request`);
    assert.equal(Object.isFrozen(output), true);
    assert.equal(Object.isFrozen(output.out), true);
    assert.equal(Object.isFrozen(output.out.shape), true);
  }
});

test('spatial shared failure vectors have deterministic TypeScript codes and paths', () => {
  for (const vector of corpus.failure_cases) {
    assert.throws(
      () => inferConcreteOperatorShapes(vector.operator, structuredClone(vector.request)),
      (error) => {
        assert.ok(error instanceof OperatorShapeContractError, vector.id);
        assert.equal(error.code, vector.expected_error.code, vector.id);
        assert.equal(error.path, vector.expected_error.path, vector.id);
        return true;
      },
    );
  }
});

const environment = new ShapeEnvironment([
  { name: 'B', min: 1, max: 8 },
  { name: 'H', min: 1, max: 32 },
  { name: 'K', min: 1, max: 16 },
  { name: 'S', min: 1, max: 16 },
  { name: 'T', min: 1, max: 20 },
  { name: 'W', min: 1, max: 32 },
]);

function prove(operator, inputs, params = undefined, declaredOutputs = undefined) {
  return proveOperatorShapeDomain(operator, {
    environment,
    inputs,
    ...(params === undefined ? {} : { params }),
    ...(declaredOutputs === undefined ? {} : { declaredOutputs }),
  });
}

test('bounded domain proof accepts exact symbolic identity formulas', () => {
  const cases = [
    ['BatchMatMul', { a: tensor(['B', 2, 'K']), b: tensor([1, 'K', 3]) }, {}, ['B', 2, 3]],
    ['Conv1D', { input: tensor(['B', 'S', 4]), weight: tensor([3, 4, 6]) },
      { padding: 1 }, ['B', 'S', 6]],
    ['Conv2D', { input: tensor(['B', 'H', 'W', 4]), weight: tensor([3, 3, 4, 6]) },
      { padding: [1, 1], weight_layout: 'HWIO' }, ['B', 'H', 'W', 6]],
    ['ConvTranspose2D', { input: tensor(['B', 'H', 'W', 4]), weight: tensor([3, 3, 4, 6]) },
      { kernel: [3, 3], padding: [1, 1] }, ['B', 'H', 'W', 6]],
    ['MaxPool2D', { input: tensor(['B', 'H', 'W', 4]) },
      { kernel: [3, 3], padding: [1, 1] }, ['B', 'H', 'W', 4]],
    ['AveragePool2D', { input: tensor(['B', 'H', 'W', 4]) },
      { kernel: [3, 3], padding: [1, 1] }, ['B', 'H', 'W', 4]],
    ['GlobalAveragePool', { input: tensor(['B', 'H', 'W', 4]) }, {}, ['B', 1, 1, 4]],
    ['UpsampleNearest2D', { input: tensor(['B', 3, 5, 4]) }, {}, ['B', 6, 10, 4]],
  ];
  for (const [operator, inputs, params, expected] of cases) {
    const proof = prove(operator, inputs, params);
    assert.equal(proof.supported, true, `${operator}: ${proof.reason}`);
    assert.deepEqual(proof.outputs.out.shape, expected, operator);
  }

  for (const operator of ['Resize', 'ResizeNearest2D']) {
    const proof = prove(
      operator,
      { input: tensor(['B', 'H', 'W', 4]) },
      operator === 'Resize' ? { mode: 'nearest' } : {},
      { out: tensor(['B', 'S', 'T', 4]) },
    );
    assert.equal(proof.supported, true, `${operator}: ${proof.reason}`);
    assert.deepEqual(proof.outputs.out.shape, ['B', 'S', 'T', 4]);
  }
});

test('bounded domain proof rejects every unprovable or invalid spatial family deterministically', () => {
  const rejections = [
    ['BatchMatMul', { a: tensor(['B', 2, 'K']), b: tensor([1, 'S', 3]) }, {}, undefined,
      'UNPROVABLE_DYNAMIC_CONTRACTION'],
    ['Conv1D', { input: tensor(['B', 'S', 4]), weight: tensor([3, 4, 6]) },
      { stride: 2, padding: 1 }, undefined, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
    ['Conv2D', { input: tensor(['B', 'H', 'W', 4]), weight: tensor([3, 3, 4, 6]) },
      { stride: [2, 2], padding: [1, 1], weight_layout: 'HWIO' }, undefined,
      'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
    ['ConvTranspose2D', { input: tensor(['B', 'H', 'W', 4]), weight: tensor([3, 3, 4, 6]) },
      { kernel: [3, 3], stride: [2, 2], padding: [1, 1] }, undefined,
      'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
    ['MaxPool2D', { input: tensor(['B', 'H', 'W', 4]) },
      { kernel: [2, 2], stride: [2, 2] }, undefined, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
    ['AveragePool2D', { input: tensor(['B', 'H', 'W', 4]) },
      { kernel: [2, 2], stride: [2, 2] }, undefined, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
    ['GlobalAveragePool', { input: tensor(['B', 'H', 4]) }, {}, undefined, 'INVALID_RANK'],
    ['Resize', { input: tensor(['B', 'H', 'W', 4]) }, { mode: 'nearest' },
      { out: tensor(['H', 'S', 'T', 4]) }, 'SHAPE_MISMATCH'],
    ['ResizeNearest2D', { input: tensor(['B', 'H', 'W', 4]) }, { mode: 'linear' },
      { out: tensor(['B', 'S', 'T', 4]) }, 'INVALID_PARAMS'],
    ['UpsampleNearest2D', { input: tensor(['B', 'H', 'W', 4]) }, {}, undefined,
      'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'],
  ];
  for (const [operator, inputs, params, declaredOutputs, code] of rejections) {
    const proof = prove(operator, inputs, params, declaredOutputs);
    assert.equal(proof.supported, false, operator);
    assert.equal(proof.code, code, `${operator}: ${proof.reason}`);
  }
});
