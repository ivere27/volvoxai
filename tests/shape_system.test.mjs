import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  PublicInputShapeContract,
  ShapeContractError,
  ShapeEnvironment,
  bindPublicInputShapes,
  canonicalShapeSignature,
  checkedShapeAdd,
  checkedShapeCeilDivide,
  checkedShapeElementCount,
  checkedShapeFloorDivide,
  checkedShapeMultiply,
  checkedShapeSubtract,
  checkedTensorByteLength,
  createTensorShapeSpec,
} from '../ts/ops/shapeSystem.js';

const SHAPE_VECTOR_FORMAT = 'volvox-shape-system-vectors/v1';
const SHAPE_VECTOR_FIELDS = [
  'failure_cases',
  'format',
  'max_safe_integer',
  'success_cases',
];
const SHAPE_VECTOR_FAILURE_OPERATIONS = new Set([
  'bind',
  'environment',
  'tensor_byte_length',
]);

function validateShapeSystemVectors(value) {
  assert.ok(value != null && typeof value === 'object' && !Array.isArray(value),
    'shape vector corpus must be an object');
  assert.deepEqual(Object.keys(value).sort(), SHAPE_VECTOR_FIELDS,
    'shape vector corpus has an unsupported top-level schema');
  assert.equal(value.format, SHAPE_VECTOR_FORMAT,
    `unsupported shape vector format '${String(value.format)}'`);
  assert.equal(value.max_safe_integer, Number.MAX_SAFE_INTEGER,
    'shape vector corpus uses a different arithmetic limit');
  assert.ok(Array.isArray(value.success_cases), 'success_cases must be an array');
  assert.ok(Array.isArray(value.failure_cases), 'failure_cases must be an array');

  const ids = new Set();
  for (const vectorCase of [...value.success_cases, ...value.failure_cases]) {
    assert.ok(vectorCase != null && typeof vectorCase === 'object' && !Array.isArray(vectorCase),
      'each shape vector case must be an object');
    assert.ok(typeof vectorCase.id === 'string' && vectorCase.id.length > 0,
      'each shape vector case must have a non-empty id');
    assert.equal(ids.has(vectorCase.id), false, `duplicate shape vector id '${vectorCase.id}'`);
    ids.add(vectorCase.id);
  }
  for (const vectorCase of value.failure_cases) {
    assert.equal(
      SHAPE_VECTOR_FAILURE_OPERATIONS.has(vectorCase.operation),
      true,
      `unsupported shape vector operation '${String(vectorCase.operation)}'`,
    );
  }
  return value;
}

function loadShapeSystemVectors() {
  const source = readFileSync(new URL('./shape_system_vectors.json', import.meta.url), 'utf8');
  return validateShapeSystemVectors(JSON.parse(source));
}

const VECTOR_TYPED_ARRAYS = Object.freeze({
  float32: Float32Array,
  int32: Int32Array,
  int8: Int8Array,
  uint8: Uint8Array,
});

function makeVectorViews(rawViews) {
  const views = {};
  const dataByName = {};
  const callerShapes = {};
  for (const [name, rawView] of Object.entries(rawViews)) {
    const TypedArray = VECTOR_TYPED_ARRAYS[rawView.dtype];
    assert.ok(TypedArray, `vector view '${name}' has unsupported dtype '${rawView.dtype}'`);
    assert.equal(
      rawView.byte_length % TypedArray.BYTES_PER_ELEMENT,
      0,
      `vector view '${name}' byte_length is not aligned to dtype '${rawView.dtype}'`,
    );
    const data = new TypedArray(rawView.byte_length / TypedArray.BYTES_PER_ELEMENT);
    const shape = [...rawView.shape];
    views[name] = { data, shape };
    dataByName[name] = data;
    callerShapes[name] = shape;
  }
  return { views, dataByName, callerShapes };
}

function assertShapeError(operation, code, message) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof ShapeContractError);
  assert.equal(error.code, code);
  if (message) assert.match(error.message, message);
  return error;
}

function languageContract(inputOrder = ['tokens', 'mask']) {
  const environment = new ShapeEnvironment([
    { name: 'S', min: 2, max: 16, multiple_of: 2 },
    { name: 'B', min: 1, max: 4 },
  ]);
  const specs = {
    tokens: { name: 'tokens', dtype: 'int32', shape: ['B', 'S'] },
    mask: { name: 'mask', dtype: 'int32', shape: ['B', 'S'] },
  };
  return new PublicInputShapeContract(
    environment,
    inputOrder.map((name) => specs[name]),
  );
}

test('shared v1 shape vectors agree exactly and reject unknown corpus schemas', () => {
  const vectors = loadShapeSystemVectors();
  const vectorsBefore = structuredClone(vectors);

  const unknownFormat = structuredClone(vectors);
  unknownFormat.format = 'volvox-shape-system-vectors/v2';
  assert.throws(() => validateShapeSystemVectors(unknownFormat), /unsupported shape vector format/);

  const extendedSchema = structuredClone(vectors);
  extendedSchema.future_field = true;
  assert.throws(
    () => validateShapeSystemVectors(extendedSchema),
    /unsupported top-level schema/,
  );

  const unknownOperation = structuredClone(vectors);
  unknownOperation.failure_cases[0].operation = 'future_operation';
  assert.throws(
    () => validateShapeSystemVectors(unknownOperation),
    /unsupported shape vector operation/,
  );

  for (const original of vectors.success_cases) {
    const vectorCase = structuredClone(original);
    const caseBefore = structuredClone(vectorCase);
    const environment = new ShapeEnvironment(vectorCase.dimensions);
    const contract = new PublicInputShapeContract(environment, vectorCase.inputs);
    const { views, dataByName, callerShapes } = makeVectorViews(vectorCase.views);
    const dataBefore = Object.fromEntries(
      Object.entries(dataByName).map(([name, data]) => [name, [...data]]),
    );

    const binding = bindPublicInputShapes(contract, views);
    const expected = vectorCase.expected;
    assert.equal(binding.signature, expected.signature, `vector '${vectorCase.id}' signature`);
    assert.deepEqual(
      Object.entries(binding.symbols),
      expected.symbols,
      `vector '${vectorCase.id}' symbols`,
    );
    assert.deepEqual(
      Object.keys(binding.inputs),
      expected.input_order,
      `vector '${vectorCase.id}' input order`,
    );

    const logicalInputs = Object.fromEntries(
      vectorCase.inputs.map((input) => [input.name, input]),
    );
    for (const name of expected.input_order) {
      const bound = binding.inputs[name];
      assert.deepEqual(
        {
          name: bound.name,
          dtype: bound.dtype,
          shape: [...bound.shape],
          elementCount: bound.elementCount,
          sizeBytes: bound.sizeBytes,
        },
        {
          name,
          dtype: logicalInputs[name].dtype,
          shape: vectorCase.views[name].shape,
          elementCount: expected.element_counts[name],
          sizeBytes: expected.byte_lengths[name],
        },
        `vector '${vectorCase.id}' descriptor '${name}'`,
      );
      assert.strictEqual(
        bound.data,
        dataByName[name],
        `vector '${vectorCase.id}' data identity '${name}'`,
      );
    }

    assert.deepEqual(vectorCase, caseBefore, `vector '${vectorCase.id}' source metadata`);
    assert.deepEqual(
      Object.fromEntries(
        Object.entries(callerShapes).map(([name, shape]) => [name, [...shape]]),
      ),
      Object.fromEntries(
        Object.entries(vectorCase.views).map(([name, view]) => [name, view.shape]),
      ),
      `vector '${vectorCase.id}' caller shapes`,
    );
    assert.deepEqual(
      Object.fromEntries(
        Object.entries(dataByName).map(([name, data]) => [name, [...data]]),
      ),
      dataBefore,
      `vector '${vectorCase.id}' caller data`,
    );
  }

  for (const original of vectors.failure_cases) {
    const vectorCase = structuredClone(original);
    const caseBefore = structuredClone(vectorCase);
    let callerState;
    let error;
    try {
      if (vectorCase.operation === 'environment') {
        new ShapeEnvironment(vectorCase.dimensions);
      } else if (vectorCase.operation === 'bind') {
        const environment = new ShapeEnvironment(vectorCase.dimensions);
        const contract = new PublicInputShapeContract(environment, vectorCase.inputs);
        callerState = makeVectorViews(vectorCase.views);
        callerState.dataBefore = Object.fromEntries(
          Object.entries(callerState.dataByName).map(([name, data]) => [name, [...data]]),
        );
        bindPublicInputShapes(contract, callerState.views);
      } else {
        checkedTensorByteLength(vectorCase.shape, vectorCase.dtype, vectorCase.path);
      }
    } catch (caught) {
      error = caught;
    }

    assert.ok(
      error instanceof ShapeContractError,
      `vector '${vectorCase.id}' must fail with ShapeContractError`,
    );
    assert.equal(error.code, vectorCase.expected_error.code, `vector '${vectorCase.id}' code`);
    assert.equal(error.path, vectorCase.expected_error.path, `vector '${vectorCase.id}' path`);
    assert.deepEqual(vectorCase, caseBefore, `vector '${vectorCase.id}' source metadata`);
    if (callerState) {
      assert.deepEqual(
        Object.fromEntries(
          Object.entries(callerState.callerShapes).map(([name, shape]) => [name, [...shape]]),
        ),
        Object.fromEntries(
          Object.entries(vectorCase.views).map(([name, view]) => [name, view.shape]),
        ),
        `vector '${vectorCase.id}' caller shapes`,
      );
      assert.deepEqual(
        Object.fromEntries(
          Object.entries(callerState.dataByName).map(([name, data]) => [name, [...data]]),
        ),
        callerState.dataBefore,
        `vector '${vectorCase.id}' caller data`,
      );
    }
  }

  assert.deepEqual(vectors, vectorsBefore, 'shared shape vector corpus must not be mutated');
});

test('checked shape arithmetic accepts exact safe bounds and rejects overflow', () => {
  assert.equal(checkedShapeAdd(Number.MAX_SAFE_INTEGER - 1, 1), Number.MAX_SAFE_INTEGER);
  assert.equal(checkedShapeMultiply(3, 7), 21);
  assert.equal(checkedShapeMultiply(-3, 7), -21);
  assert.equal(checkedShapeSubtract(3, 7), -4);
  assert.equal(checkedShapeFloorDivide(8, 3), 2);
  assert.equal(checkedShapeCeilDivide(8, 3), 3);
  assert.equal(checkedShapeFloorDivide(-8, 3), -3);
  assert.equal(checkedShapeCeilDivide(-8, 3), -2);
  assert.equal(checkedShapeCeilDivide(Number.MAX_SAFE_INTEGER, 1), Number.MAX_SAFE_INTEGER);
  assert.equal(checkedShapeElementCount([2, 3, 5]), 30);
  assert.equal(checkedTensorByteLength([2, 3, 5], 'float32'), 120);

  assertShapeError(
    () => checkedShapeAdd(Number.MAX_SAFE_INTEGER, 1, 'sum'),
    'ARITHMETIC_OVERFLOW',
    /sum: exceeds/,
  );
  assertShapeError(
    () => checkedShapeMultiply(Number.MAX_SAFE_INTEGER, 2, 'product'),
    'ARITHMETIC_OVERFLOW',
    /product: exceeds/,
  );
  assertShapeError(
    () => checkedShapeSubtract(0, Number.MIN_SAFE_INTEGER, 'difference'),
    'ARITHMETIC_OVERFLOW',
    /difference: exceeds/,
  );
  assertShapeError(
    () => checkedShapeElementCount([Number.MAX_SAFE_INTEGER, 2], 'huge'),
    'ARITHMETIC_OVERFLOW',
    /huge element count/,
  );
  assertShapeError(
    () => checkedTensorByteLength([Number.MAX_SAFE_INTEGER], 'int32', 'huge tensor'),
    'ARITHMETIC_OVERFLOW',
    /huge tensor byte length/,
  );
  assertShapeError(
    () => checkedShapeFloorDivide(4, 0),
    'ARITHMETIC_INVALID',
    /divisor: must be a positive safe integer/,
  );
});

test('constant and symbolic public inputs bind to immutable concrete descriptors', () => {
  const environment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 4 },
    { name: 'S', min: 2, max: 8, multiple_of: 2 },
  ]);
  const contract = new PublicInputShapeContract(environment, [
    { name: 'scale', dtype: 'float32', shape: [1] },
    { name: 'tokens', dtype: 'int32', shape: ['B', 'S'] },
  ]);
  const tokenData = new Int32Array(8);
  const binding = bindPublicInputShapes(contract, {
    tokens: { data: tokenData, shape: [2, 4] },
    scale: { data: Float32Array.of(0.5), shape: [1] },
  });

  assert.deepEqual(binding.symbols, { B: 2, S: 4 });
  assert.deepEqual(binding.inputs.tokens.shape, [2, 4]);
  assert.equal(binding.inputs.tokens.elementCount, 8);
  assert.equal(binding.inputs.tokens.sizeBytes, 32);
  assert.strictEqual(binding.inputs.tokens.data, tokenData);
  assert.deepEqual(binding.inputs.scale.shape, [1]);
  assert.equal(binding.inputs.scale.sizeBytes, 4);
  assert.equal(binding.signature, 'v1|5:scale|1:1|6:tokens|2:2,4');

  assert.equal(Object.isFrozen(environment), true);
  assert.equal(Object.isFrozen(environment.dimensions), true);
  assert.equal(Object.isFrozen(environment.dimensions[0]), true);
  assert.equal(Object.isFrozen(contract), true);
  assert.equal(Object.isFrozen(contract.inputs), true);
  assert.equal(Object.isFrozen(contract.inputs[0].shape), true);
  assert.equal(Object.isFrozen(binding), true);
  assert.equal(Object.isFrozen(binding.inputs), true);
  assert.equal(Object.isFrozen(binding.inputs.tokens), true);
  assert.equal(Object.isFrozen(binding.inputs.tokens.shape), true);
  assert.equal(Object.isFrozen(binding.symbols), true);
});

test('repeated symbols reject conflicting same-byte input shapes', () => {
  const contract = languageContract();
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Int32Array(8), shape: [2, 4] },
      mask: { data: new Int32Array(8), shape: [1, 8] },
    }),
    'SYMBOL_CONFLICT',
    /binds 'B' to 2, but it was already bound to 1|binds 'B' to 1, but it was already bound to 2/,
  );
});

test('constant dimensions reject a wrong shape even when rank and bytes match', () => {
  const contract = new PublicInputShapeContract(new ShapeEnvironment([]), [
    { name: 'image', dtype: 'float32', shape: [2, 2] },
  ]);
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      image: { data: new Float32Array(4), shape: [1, 4] },
    }),
    'DIMENSION_MISMATCH',
    /requires constant 2/,
  );
});

test('binding validates exact names, dtype, rank, bounds, multiples, and bytes', () => {
  const contract = languageContract();
  const validTokens = { data: new Int32Array(8), shape: [2, 4] };
  const validMask = { data: new Int32Array(8), shape: [2, 4] };

  assertShapeError(
    () => bindPublicInputShapes(contract, { tokens: validTokens }),
    'INVALID_INPUT_SET',
    /missing \[mask\]/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: validTokens,
      mask: validMask,
      extra: { data: new Int32Array(1), shape: [1] },
    }),
    'INVALID_INPUT_SET',
    /unexpected \[extra\]/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Float32Array(8), shape: [2, 4] },
      mask: validMask,
    }),
    'DTYPE_MISMATCH',
    /requires 'int32'/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Int32Array(8), shape: [8] },
      mask: validMask,
    }),
    'RANK_MISMATCH',
    /requires rank 2/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Int32Array(15), shape: [3, 5] },
      mask: { data: new Int32Array(15), shape: [3, 5] },
    }),
    'MULTIPLE_OF_VIOLATION',
    /not a multiple of 2/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Int32Array(20), shape: [5, 4] },
      mask: { data: new Int32Array(20), shape: [5, 4] },
    }),
    'BOUND_VIOLATION',
    /outside \[1, 4\]/,
  );
  assertShapeError(
    () => bindPublicInputShapes(contract, {
      tokens: { data: new Int32Array(7), shape: [2, 4] },
      mask: validMask,
    }),
    'BYTE_LENGTH_MISMATCH',
    /has 28 bytes.*require 32/,
  );
});

test('canonical signatures do not depend on descriptor or value insertion order', () => {
  const first = bindPublicInputShapes(languageContract(['tokens', 'mask']), {
    tokens: { data: new Int32Array(8), shape: [2, 4] },
    mask: { data: new Int32Array(8), shape: [2, 4] },
  });
  const second = bindPublicInputShapes(languageContract(['mask', 'tokens']), {
    mask: { data: new Int32Array(8), shape: [2, 4] },
    tokens: { data: new Int32Array(8), shape: [2, 4] },
  });
  assert.equal(first.signature, second.signature);
  assert.deepEqual(Object.keys(first.inputs), ['mask', 'tokens']);

  assert.equal(
    canonicalShapeSignature([
      { name: 'z', shape: [2, 3] },
      { name: 'a', shape: [1] },
    ]),
    canonicalShapeSignature([
      { name: 'a', shape: [1] },
      { name: 'z', shape: [2, 3] },
    ]),
  );
  assert.equal(
    canonicalShapeSignature([
      { name: 'z', shape: [2, 3] },
      { name: 'a', shape: [1] },
    ]),
    'v1|1:a|1:1|1:z|2:2,3',
  );
  assert.notEqual(
    canonicalShapeSignature([{ name: 'a', shape: [1, 4] }]),
    canonicalShapeSignature([{ name: 'a', shape: [2, 2] }]),
  );
});

test('shape construction rejects malformed constraints, duplicates, and unknown symbols', () => {
  assertShapeError(
    () => new ShapeEnvironment([{ name: 'bad-name', min: 1, max: 2 }]),
    'INVALID_CONSTRAINT',
    /must match/,
  );
  assert.doesNotThrow(
    () => new ShapeEnvironment([{ name: `B${'x'.repeat(63)}`, min: 1, max: 2 }]),
  );
  assertShapeError(
    () => new ShapeEnvironment([{ name: `B${'x'.repeat(64)}`, min: 1, max: 2 }]),
    'INVALID_CONSTRAINT',
    /must match/,
  );
  assertShapeError(
    () => new ShapeEnvironment([
      { name: 'B', min: 1, max: 2 },
      { name: 'B', min: 1, max: 4 },
    ]),
    'DUPLICATE_SYMBOL',
    /declared more than once/,
  );
  assertShapeError(
    () => new ShapeEnvironment([{ name: 'B', min: 0, max: 2 }]),
    'INVALID_CONSTRAINT',
    /positive safe integer/,
  );
  assertShapeError(
    () => new ShapeEnvironment([{ name: 'B', min: 3, max: 2 }]),
    'INVALID_CONSTRAINT',
    /min 3 greater than max 2/,
  );
  assertShapeError(
    () => new ShapeEnvironment([{ name: 'B', min: 1, max: 2, multiple_of: 0 }]),
    'INVALID_CONSTRAINT',
    /multiple_of.*positive safe integer/,
  );
  assertShapeError(
    () => new ShapeEnvironment([{ name: 'B', min: 2, max: 3, multiple_of: 4 }]),
    'INVALID_CONSTRAINT',
    /has no multiple of 4/,
  );

  const environment = new ShapeEnvironment([{ name: 'B', min: 1, max: 4 }]);
  assertShapeError(
    () => createTensorShapeSpec(['S'], environment),
    'UNKNOWN_SYMBOL',
    /undeclared symbol 'S'/,
  );
  assertShapeError(
    () => createTensorShapeSpec([0], environment),
    'INVALID_SHAPE_SPEC',
    /positive safe integer/,
  );
  assertShapeError(
    () => new PublicInputShapeContract(environment, [
      { name: 'x', dtype: 'float32', shape: ['B'] },
      { name: 'x', dtype: 'float32', shape: ['B'] },
    ]),
    'DUPLICATE_INPUT',
    /declared more than once/,
  );
});

test('construction and binding clone shape metadata and never mutate caller values', () => {
  const constraint = { name: 'B', min: 1, max: 4 };
  const sourceShape = ['B', 2];
  const environment = new ShapeEnvironment([constraint]);
  const contract = new PublicInputShapeContract(environment, [
    { name: 'x', dtype: 'int8', shape: sourceShape },
  ]);

  constraint.min = 3;
  sourceShape[0] = 4;
  assert.deepEqual(environment.dimensions, [{ name: 'B', min: 1, max: 4 }]);
  assert.deepEqual(contract.inputs[0].shape, ['B', 2]);

  const data = Int8Array.of(1, 2, 3, 4);
  const dataBefore = [...data];
  const callerShape = [2, 2];
  const binding = bindPublicInputShapes(contract, {
    x: { data, shape: callerShape },
  });
  callerShape[0] = 1;
  assert.deepEqual(binding.inputs.x.shape, [2, 2]);
  assert.deepEqual([...data], dataBefore);
});
