import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  OperatorShapeContractError,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';

const VECTOR_PATH = new URL('./operator_shape_contract_wave_b_vectors.json', import.meta.url);
const CONCAT_AFFINE_VECTOR_PATH = new URL(
  './operator_shape_contract_concat_affine_vectors.json',
  import.meta.url,
);
const EXPECTED_OPERATORS = new Set([
  'Sub', 'Div', 'Equal', 'GreaterOrEqual', 'Where', 'ReduceSum', 'ReduceMean',
  'ArgMax', 'Transpose', 'Flatten', 'Squeeze', 'Unsqueeze', 'Reshape', 'Expand',
  'Concat', 'Split', 'Slice', 'Pad', 'Gather', 'GatherElements',
]);

function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertDescriptor(value, path) {
  assert.equal(isRecord(value), true, `${path} must be an object`);
  assert.deepEqual(
    Object.keys(value).filter((name) => !['shape', 'dtype', 'quantization'].includes(name)),
    [],
    `${path} has unsupported fields`,
  );
  assert.ok(Array.isArray(value.shape), `${path}.shape must be an array`);
  assert.ok(value.shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0));
  assert.ok(['float32', 'int32', 'int8', 'uint8'].includes(value.dtype));
}

function loadCorpus() {
  const corpus = JSON.parse(readFileSync(VECTOR_PATH, 'utf8'));
  assert.deepEqual(Object.keys(corpus).sort(), [
    'failure_cases', 'families', 'format', 'success_cases',
  ]);
  assert.equal(corpus.format, 'volvox-operator-shape-vectors/v1');
  assert.ok(Array.isArray(corpus.families) && corpus.families.length > 0);
  assert.ok(Array.isArray(corpus.success_cases));
  assert.ok(Array.isArray(corpus.failure_cases));

  const familyOperators = new Map();
  const manifest = new Set();
  for (const family of corpus.families) {
    assert.deepEqual(Object.keys(family).sort(), ['id', 'operators']);
    assert.ok(typeof family.id === 'string' && family.id.length > 0);
    assert.equal(familyOperators.has(family.id), false, `duplicate family ${family.id}`);
    assert.ok(Array.isArray(family.operators) && family.operators.length > 0);
    const operators = new Set(family.operators);
    assert.equal(operators.size, family.operators.length);
    for (const operator of operators) {
      assert.equal(manifest.has(operator), false, `${operator} belongs to multiple families`);
      manifest.add(operator);
    }
    familyOperators.set(family.id, operators);
  }
  assert.deepEqual(manifest, EXPECTED_OPERATORS);

  const ids = new Set();
  const legalCoverage = new Map([...manifest].map((operator) => [operator, 0]));
  const illegalCoverage = new Map([...manifest].map((operator) => [operator, 0]));
  for (const [kind, cases, coverage] of [
    ['success', corpus.success_cases, legalCoverage],
    ['failure', corpus.failure_cases, illegalCoverage],
  ]) {
    for (const vector of cases) {
      assert.ok(typeof vector.id === 'string' && vector.id.length > 0);
      assert.equal(ids.has(vector.id), false, `duplicate case ${vector.id}`);
      ids.add(vector.id);
      assert.ok(familyOperators.has(vector.family));
      assert.ok(Array.isArray(vector.operators) && vector.operators.length > 0);
      for (const operator of vector.operators) {
        assert.ok(familyOperators.get(vector.family).has(operator));
        coverage.set(operator, coverage.get(operator) + 1);
      }
      assert.equal(isRecord(vector.request), true);
      assert.equal(isRecord(vector.request.inputs), true);
      for (const [name, descriptor] of Object.entries(vector.request.inputs)) {
        assert.ok(name.length > 0);
        assertDescriptor(descriptor, `${vector.id}.request.inputs.${name}`);
      }
      assert.ok(typeof vector.expected_shape_function_id === 'string');
      if (kind === 'success') {
        assert.equal(isRecord(vector.expected), true);
        assert.ok(Object.keys(vector.expected).length > 0);
        for (const [name, descriptor] of Object.entries(vector.expected)) {
          assert.ok(name.length > 0);
          assertDescriptor(descriptor, `${vector.id}.expected.${name}`);
        }
      } else {
        assert.deepEqual(Object.keys(vector.expected_error).sort(), ['code', 'path']);
      }
    }
  }
  for (const operator of manifest) {
    assert.ok(legalCoverage.get(operator) >= 2, `${operator} needs two legal vectors`);
    assert.ok(illegalCoverage.get(operator) >= 1, `${operator} needs one illegal vector`);
  }
  return corpus;
}

const corpus = loadCorpus();

function loadConcatAffineCorpus() {
  const value = JSON.parse(readFileSync(CONCAT_AFFINE_VECTOR_PATH, 'utf8'));
  assert.deepEqual(Object.keys(value).sort(), ['cases', 'format']);
  assert.equal(value.format, 'volvox-concat-affine-domain-vectors/v1');
  assert.ok(Array.isArray(value.cases) && value.cases.length > 0);
  assert.equal(new Set(value.cases.map((vector) => vector.id)).size, value.cases.length);
  return value;
}

const concatAffineCorpus = loadConcatAffineCorpus();

test('shared non-spatial Wave-B legal vectors converge in concrete and bounded TS contracts', () => {
  for (const original of corpus.success_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      const before = structuredClone(vector);
      const contract = getOperatorShapeContract(operator);
      assert.equal(contract.shapeFunctionId, vector.expected_shape_function_id,
        `${vector.id}:${operator} route`);

      const outputs = inferConcreteOperatorShapes(operator, vector.request);
      assert.deepEqual(outputs, vector.expected, `${vector.id}:${operator} concrete`);
      assert.equal(Object.isFrozen(outputs), true);
      for (const descriptor of Object.values(outputs)) {
        assert.equal(Object.isFrozen(descriptor), true);
        assert.equal(Object.isFrozen(descriptor.shape), true);
      }

      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, true, `${vector.id}:${operator} domain`);
      assert.deepEqual(proof.outputs, vector.expected, `${vector.id}:${operator} domain outputs`);
      assert.deepEqual(vector, before, `${vector.id}:${operator} mutated its vector`);
    }
  }
});

test('literal scalar broadcast preserves a singleton-bound symbol in TypeScript', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 1 },
    { name: 'Q', min: 1, max: 192 },
  ]);
  const proof = proveOperatorShapeDomain('Equal', {
    environment,
    inputs: {
      a: { shape: ['B', 'Q'], dtype: 'int32' },
      b: { shape: [], dtype: 'int32' },
    },
  });

  assert.equal(proof.supported, true, proof.reason ?? '');
  assert.deepEqual(proof.outputs, {
    out: { shape: ['B', 'Q'], dtype: 'int32' },
  });
});

test('shared non-spatial Wave-B illegal vectors have exact TS concrete codes and paths', () => {
  for (const original of corpus.failure_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      const before = structuredClone(vector);
      assert.equal(
        getOperatorShapeContract(operator).shapeFunctionId,
        vector.expected_shape_function_id,
      );
      assert.throws(
        () => inferConcreteOperatorShapes(operator, vector.request),
        (error) => {
          assert.ok(error instanceof OperatorShapeContractError);
          assert.equal(error.code, vector.expected_error.code, `${vector.id}:${operator}`);
          assert.equal(error.path, vector.expected_error.path, `${vector.id}:${operator}`);
          return true;
        },
      );
      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, false, `${vector.id}:${operator} domain accepted invalid request`);
      assert.deepEqual(vector, before, `${vector.id}:${operator} mutated its vector`);
    }
  }
});

test('shared affine Concat vectors prove exactly one dynamic term in TypeScript', () => {
  for (const original of concatAffineCorpus.cases) {
    const vector = structuredClone(original);
    const before = structuredClone(vector);
    const environment = new ShapeEnvironment(Object.entries(vector.environment).map(
      ([name, constraint]) => ({ name, ...constraint }),
    ));
    const proof = proveOperatorShapeDomain('Concat', {
      ...vector.request,
      environment,
    });
    if (vector.expected !== undefined) {
      assert.equal(proof.supported, true, `${vector.id}: ${proof.reason ?? ''}`);
      assert.deepEqual(proof.outputs, vector.expected, vector.id);
      assert.deepEqual(proof.facts, vector.facts, `${vector.id}: facts`);
      assert.deepEqual(
        proof.affineRelations,
        vector.relations,
        `${vector.id}: affine relations`,
      );
    } else {
      assert.equal(proof.supported, false, `${vector.id}: unexpectedly supported`);
      assert.equal(proof.code, vector.expected_error.code, vector.id);
      assert.match(
        proof.reason,
        new RegExp(`^${vector.expected_error.path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}:`),
        vector.id,
      );
    }
    assert.deepEqual(vector, before, `${vector.id}: mutated its vector`);
  }
});
