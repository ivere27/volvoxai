import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  OperatorShapeContractError,
  WAVE_A_OPERATOR_SHAPE_FUNCTIONS,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
} from '../ts/ops/operatorShapeContracts.js';

const ROOT_FIELDS = Object.freeze([
  'failure_cases',
  'families',
  'format',
  'success_cases',
]);
const SUCCESS_FIELDS = Object.freeze([
  'expected',
  'expected_shape_function_id',
  'family',
  'id',
  'operators',
  'request',
]);
const FAILURE_FIELDS = Object.freeze([
  'expected_error',
  'expected_shape_function_id',
  'family',
  'id',
  'operators',
  'request',
]);
const FORBIDDEN_VALUE_FIELDS = new Set(['buffer', 'data', 'value', 'values']);
const RUNTIME_DTYPES = new Set(['float32', 'int32', 'int8', 'uint8']);

function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertExactFields(value, expected, path) {
  assert.equal(isRecord(value), true, `${path} must be an object`);
  assert.deepEqual(Object.keys(value).sort(), [...expected].sort(),
    `${path} has an unsupported schema`);
}

function assertNoValues(value, path) {
  if (Array.isArray(value)) {
    value.forEach((child, index) => assertNoValues(child, `${path}[${index}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const [name, child] of Object.entries(value)) {
    assert.equal(FORBIDDEN_VALUE_FIELDS.has(name), false,
      `${path} must not carry tensor values through '${name}'`);
    assertNoValues(child, `${path}.${name}`);
  }
}

function assertDescriptorSchema(descriptor, path) {
  assert.equal(isRecord(descriptor), true, `${path} must be an object`);
  const allowed = new Set(['dtype', 'quantization', 'shape']);
  for (const name of Object.keys(descriptor)) {
    assert.equal(allowed.has(name), true, `${path} has unsupported field '${name}'`);
  }
  assert.ok(Array.isArray(descriptor.shape), `${path}.shape must be an array`);
  for (const dimension of descriptor.shape) {
    assert.equal(Number.isSafeInteger(dimension) && dimension > 0, true,
      `${path}.shape must contain positive safe integers`);
  }
  assert.equal(RUNTIME_DTYPES.has(descriptor.dtype), true,
    `${path}.dtype is unsupported`);
  if (descriptor.quantization === undefined) return;
  assert.equal(isRecord(descriptor.quantization), true,
    `${path}.quantization must be an object`);
  if (descriptor.quantization.scheme === 'per_tensor') {
    assertExactFields(
      descriptor.quantization,
      ['scale', 'scheme', 'zero_point'],
      `${path}.quantization`,
    );
    return;
  }
  assert.equal(descriptor.quantization.scheme, 'per_axis',
    `${path}.quantization has an unsupported scheme`);
  assertExactFields(
    descriptor.quantization,
    ['axis', 'scales', 'scheme', 'zero_points'],
    `${path}.quantization`,
  );
  assert.ok(Array.isArray(descriptor.quantization.scales));
  assert.ok(Array.isArray(descriptor.quantization.zero_points));
}

function assertRequestSchema(request, path) {
  assert.equal(isRecord(request), true, `${path} must be an object`);
  const allowed = new Set(['declaredOutputs', 'inputs', 'params']);
  for (const name of Object.keys(request)) {
    assert.equal(allowed.has(name), true, `${path} has unsupported field '${name}'`);
  }
  assert.equal(isRecord(request.inputs), true, `${path}.inputs must be an object`);
  assert.ok(Object.keys(request.inputs).length > 0, `${path}.inputs must not be empty`);
  for (const [name, descriptor] of Object.entries(request.inputs)) {
    assert.ok(name.length > 0, `${path}.inputs has an empty port name`);
    assertDescriptorSchema(descriptor, `${path}.inputs.${name}`);
  }
  if (request.params !== undefined) {
    assert.equal(isRecord(request.params), true, `${path}.params must be an object`);
  }
  if (request.declaredOutputs !== undefined) {
    assertExactFields(request.declaredOutputs, ['out'], `${path}.declaredOutputs`);
    assertDescriptorSchema(request.declaredOutputs.out, `${path}.declaredOutputs.out`);
  }
  assertNoValues(request, path);
}

function loadAndValidateCorpus() {
  const source = readFileSync(
    new URL('./operator_shape_contract_vectors.json', import.meta.url),
    'utf8',
  );
  const vectors = JSON.parse(source);
  assertExactFields(vectors, ROOT_FIELDS, 'corpus');
  assert.equal(vectors.format, 'volvox-operator-shape-vectors/v1');
  assert.ok(Array.isArray(vectors.families));
  assert.ok(Array.isArray(vectors.success_cases));
  assert.ok(Array.isArray(vectors.failure_cases));

  const familyIds = new Set();
  const operatorsByFamily = new Map();
  const manifestOperators = new Set();
  for (const [index, family] of vectors.families.entries()) {
    const path = `families[${index}]`;
    assertExactFields(family, ['id', 'operators'], path);
    assert.ok(typeof family.id === 'string' && family.id.length > 0, path);
    assert.equal(familyIds.has(family.id), false, `duplicate family id '${family.id}'`);
    familyIds.add(family.id);
    assert.ok(Array.isArray(family.operators) && family.operators.length > 0, path);
    const operators = new Set(family.operators);
    assert.equal(operators.size, family.operators.length, `${path} has duplicate operators`);
    for (const operator of operators) {
      assert.equal(manifestOperators.has(operator), false,
        `operator '${operator}' belongs to multiple families`);
      manifestOperators.add(operator);
    }
    operatorsByFamily.set(family.id, operators);
  }
  assert.deepEqual(manifestOperators, new Set(Object.keys(WAVE_A_OPERATOR_SHAPE_FUNCTIONS)));

  const caseIds = new Set();
  const successesByFamily = new Map([...familyIds].map((family) => [family, 0]));
  const failuresByFamily = new Map([...familyIds].map((family) => [family, 0]));
  const successfulOperators = new Set();
  const failedOperators = new Set();
  const groups = [
    ['success', vectors.success_cases, SUCCESS_FIELDS, successesByFamily, successfulOperators],
    ['failure', vectors.failure_cases, FAILURE_FIELDS, failuresByFamily, failedOperators],
  ];
  for (const [kind, cases, fields, counts, covered] of groups) {
    for (const [index, vectorCase] of cases.entries()) {
      const path = `${kind}_cases[${index}]`;
      assertExactFields(vectorCase, fields, path);
      assert.ok(typeof vectorCase.id === 'string' && vectorCase.id.length > 0, path);
      assert.equal(caseIds.has(vectorCase.id), false,
        `duplicate operator shape vector id '${vectorCase.id}'`);
      caseIds.add(vectorCase.id);
      assert.equal(familyIds.has(vectorCase.family), true, `${path} has unknown family`);
      assert.ok(Array.isArray(vectorCase.operators) && vectorCase.operators.length > 0, path);
      assert.equal(new Set(vectorCase.operators).size, vectorCase.operators.length,
        `${path} has duplicate operators`);
      for (const operator of vectorCase.operators) {
        assert.equal(operatorsByFamily.get(vectorCase.family).has(operator), true,
          `${path} routes '${operator}' through the wrong family`);
        covered.add(operator);
      }
      counts.set(vectorCase.family, counts.get(vectorCase.family) + 1);
      assertRequestSchema(vectorCase.request, `${path}.request`);
      assert.ok(typeof vectorCase.expected_shape_function_id === 'string' &&
        vectorCase.expected_shape_function_id.length > 0, path);
      if (kind === 'success') {
        assertExactFields(vectorCase.expected, ['out'], `${path}.expected`);
        assertDescriptorSchema(vectorCase.expected.out, `${path}.expected.out`);
      } else {
        assertExactFields(vectorCase.expected_error, ['code', 'path'],
          `${path}.expected_error`);
      }
    }
  }
  for (const family of familyIds) {
    assert.ok(successesByFamily.get(family) >= 2,
      `family '${family}' needs at least two legal shapes`);
    assert.ok(failuresByFamily.get(family) >= 1,
      `family '${family}' needs at least one illegal case`);
  }
  assert.deepEqual(successfulOperators, manifestOperators,
    'every Wave-A operator must have a legal shared vector');
  assert.deepEqual(failedOperators, manifestOperators,
    'every Wave-A operator must have an illegal shared vector');
  return vectors;
}

const vectors = loadAndValidateCorpus();

test('shared Wave-A success vectors match generated IDs and immutable TS inference', () => {
  for (const original of vectors.success_cases) {
    for (const operator of original.operators) {
      const vectorCase = structuredClone(original);
      const before = structuredClone(vectorCase);
      const contract = getOperatorShapeContract(operator);
      assert.equal(contract.shapeFunctionId, vectorCase.expected_shape_function_id,
        `${vectorCase.id}:${operator}`);
      const outputs = inferConcreteOperatorShapes(operator, vectorCase.request);
      assert.deepEqual(outputs, vectorCase.expected, `${vectorCase.id}:${operator}`);
      assert.deepEqual(vectorCase, before, `${vectorCase.id}:${operator} mutated its request`);
      assert.equal(Object.isFrozen(outputs), true, `${vectorCase.id}:${operator} outputs`);
      assert.equal(Object.isFrozen(outputs.out), true, `${vectorCase.id}:${operator} out`);
      assert.equal(Object.isFrozen(outputs.out.shape), true, `${vectorCase.id}:${operator} shape`);
      if (outputs.out.quantization !== undefined) {
        assert.equal(Object.isFrozen(outputs.out.quantization), true,
          `${vectorCase.id}:${operator} quantization`);
        if (outputs.out.quantization.scheme === 'per_axis') {
          assert.equal(Object.isFrozen(outputs.out.quantization.scales), true);
          assert.equal(Object.isFrozen(outputs.out.quantization.zero_points), true);
        }
      }
    }
  }
});

test('shared Wave-A failure vectors have exact TS codes and paths', () => {
  for (const original of vectors.failure_cases) {
    for (const operator of original.operators) {
      const vectorCase = structuredClone(original);
      const before = structuredClone(vectorCase);
      const contract = getOperatorShapeContract(operator);
      assert.equal(contract.shapeFunctionId, vectorCase.expected_shape_function_id,
        `${vectorCase.id}:${operator}`);
      assert.throws(
        () => inferConcreteOperatorShapes(operator, vectorCase.request),
        (error) => {
          assert.ok(error instanceof OperatorShapeContractError,
            `${vectorCase.id}:${operator}`);
          assert.equal(error.code, vectorCase.expected_error.code,
            `${vectorCase.id}:${operator}`);
          assert.equal(error.path, vectorCase.expected_error.path,
            `${vectorCase.id}:${operator}`);
          return true;
        },
      );
      assert.deepEqual(vectorCase, before, `${vectorCase.id}:${operator} mutated its request`);
    }
  }
});
