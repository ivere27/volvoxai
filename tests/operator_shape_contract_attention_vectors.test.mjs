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

const VECTOR_PATH = new URL('./operator_shape_contract_attention_vectors.json', import.meta.url);
const EXPECTED_OPERATORS = new Set(['SDPA', 'CrossSDPA', 'RoPE']);

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

function assertNoTensorValues(value, path = 'corpus') {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoTensorValues(item, `${path}[${index}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const [name, item] of Object.entries(value)) {
    assert.equal(['buffer', 'data', 'value', 'values'].includes(name), false,
      `${path}.${name} is a forbidden tensor-value field`);
    assertNoTensorValues(item, `${path}.${name}`);
  }
}

function loadCorpus() {
  const corpus = JSON.parse(readFileSync(VECTOR_PATH, 'utf8'));
  assert.deepEqual(Object.keys(corpus).sort(), [
    'failure_cases', 'families', 'format', 'success_cases',
  ]);
  assert.equal(corpus.format, 'volvox-operator-shape-vectors/v1');
  assertNoTensorValues(corpus);

  const familyOperators = new Map();
  const manifest = new Set();
  for (const family of corpus.families) {
    assert.deepEqual(Object.keys(family).sort(), ['id', 'operators']);
    assert.equal(familyOperators.has(family.id), false, `duplicate family ${family.id}`);
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
  const legal = new Map([...manifest].map((operator) => [operator, 0]));
  const illegal = new Map([...manifest].map((operator) => [operator, 0]));
  for (const [kind, cases, coverage] of [
    ['success', corpus.success_cases, legal],
    ['failure', corpus.failure_cases, illegal],
  ]) {
    for (const vector of cases) {
      assert.equal(ids.has(vector.id), false, `duplicate case ${vector.id}`);
      ids.add(vector.id);
      assert.ok(familyOperators.has(vector.family));
      for (const operator of vector.operators) {
        assert.ok(familyOperators.get(vector.family).has(operator));
        coverage.set(operator, coverage.get(operator) + 1);
      }
      assert.equal(isRecord(vector.request), true);
      assert.equal(isRecord(vector.request.inputs), true);
      for (const [name, descriptor] of Object.entries(vector.request.inputs)) {
        assertDescriptor(descriptor, `${vector.id}.request.inputs.${name}`);
      }
      if (kind === 'success') {
        for (const [name, descriptor] of Object.entries(vector.expected)) {
          assertDescriptor(descriptor, `${vector.id}.expected.${name}`);
        }
      } else {
        assert.deepEqual(Object.keys(vector.expected_error).sort(), ['code', 'path']);
      }
    }
  }
  for (const operator of manifest) {
    assert.ok(legal.get(operator) >= 2, `${operator} needs two legal vectors`);
    assert.ok(illegal.get(operator) >= 1, `${operator} needs one illegal vector`);
  }
  for (const variant of ['mask-k', 'mask-bk', 'mask-qk', 'mask-bqk']) {
    assert.ok(corpus.success_cases.some((vector) => vector.id === `sdpa-${variant}`));
    assert.ok(corpus.success_cases.some((vector) => vector.id === `cross-sdpa-${variant}`));
  }
  return corpus;
}

const corpus = loadCorpus();

test('shared attention-family legal vectors converge in concrete and bounded TS contracts', () => {
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
        assert.equal(descriptor.quantization, undefined);
      }
      const proof = proveOperatorShapeDomain(operator, {
        ...vector.request,
        environment: new ShapeEnvironment([]),
      });
      assert.equal(proof.supported, true,
        `${vector.id}:${operator}: ${proof.supported ? '' : proof.reason}`);
      assert.deepEqual(proof.outputs, vector.expected, `${vector.id}:${operator} domain`);
      assert.deepEqual(vector, before, `${vector.id}:${operator} mutated its vector`);
    }
  }
});

test('shared attention-family illegal vectors have exact TS codes and paths', () => {
  for (const original of corpus.failure_cases) {
    for (const operator of original.operators) {
      const vector = structuredClone(original);
      const before = structuredClone(vector);
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
      assert.equal(proof.supported, false, `${vector.id}:${operator} domain accepted invalid input`);
      assert.deepEqual(vector, before, `${vector.id}:${operator} mutated its vector`);
    }
  }
});

test('attention-family bounded proofs accept dynamic B/Q/K/S but keep attention widths provable', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'Q', min: 1, max: 8 },
    { name: 'K', min: 1, max: 12 },
    { name: 'S', min: 1, max: 16 },
    { name: 'D', min: 4, max: 12, multiple_of: 2 },
  ]);
  const tensor = (shape, dtype = 'float32') => ({ shape, dtype });

  const sdpa = proveOperatorShapeDomain('SDPA', {
    environment,
    inputs: {
      qkv: tensor(['B', 'S', 24]),
      mask: tensor(['B', 'S', 'S'], 'int32'),
    },
    params: { heads: 2, causal: true },
  });
  assert.equal(sdpa.supported, true, sdpa.supported ? '' : sdpa.reason);
  assert.deepEqual(sdpa.outputs.out.shape, ['B', 'S', 8]);

  const cross = proveOperatorShapeDomain('CrossSDPA', {
    environment,
    inputs: {
      q: tensor(['B', 'Q', 8]),
      k: tensor(['B', 'K', 8]),
      v: tensor(['B', 'K', 8]),
      mask: tensor(['Q', 'K'], 'int32'),
    },
    params: { heads: 4, causal: false },
  });
  assert.equal(cross.supported, true, cross.supported ? '' : cross.reason);
  assert.deepEqual(cross.outputs.out.shape, ['B', 'Q', 8]);

  const rope = proveOperatorShapeDomain('RoPE', {
    environment,
    inputs: {
      input: tensor(['B', 'S', 'D']),
      position_ids: tensor(['B', 'S'], 'int32'),
    },
    params: {},
  });
  assert.equal(rope.supported, true, rope.supported ? '' : rope.reason);
  assert.deepEqual(rope.outputs.out.shape, ['B', 'S', 'D']);
});

test('attention-family bounded proofs reject unprovable widths independently of concrete inference', () => {
  const environment = new ShapeEnvironment([
    { name: 'S', min: 1, max: 16 },
    { name: 'Packed', min: 24, max: 48, multiple_of: 24 },
    { name: 'Feature', min: 8, max: 16, multiple_of: 8 },
    { name: 'OddPossible', min: 4, max: 9 },
  ]);
  const tensor = (shape) => ({ shape, dtype: 'float32' });
  const cases = [
    ['SDPA', {
      inputs: { qkv: tensor(['S', 'Packed']) },
      params: { heads: 2, causal: true },
    }, 'UNPROVABLE_DYNAMIC_FEATURE'],
    ['CrossSDPA', {
      inputs: {
        q: tensor(['S', 'Feature']),
        k: tensor(['S', 'Feature']),
        v: tensor(['S', 'Feature']),
      },
      params: { heads: 2, causal: false },
    }, 'UNPROVABLE_DYNAMIC_FEATURE'],
    ['RoPE', {
      inputs: { input: tensor(['S', 'OddPossible']) },
      params: {},
    }, 'UNPROVABLE_DYNAMIC_FEATURE'],
  ];
  for (const [operator, request, code] of cases) {
    const proof = proveOperatorShapeDomain(operator, { environment, ...request });
    assert.equal(proof.supported, false, `${operator} unexpectedly proved its domain`);
    assert.equal(proof.code, code);
  }
});
