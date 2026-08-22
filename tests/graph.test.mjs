import test from 'node:test';
import assert from 'node:assert/strict';

import {
  GraphError,
  parseGraphDocument,
} from '../ts/core/Graph.js';

function constantDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: {
      x: { dtype: 'float32', shape: [1, 4] },
    },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: {
        out: { tensor: 'y', dtype: 'float32', shape: [1, 4] },
      },
      params: { nested: { enabled: true, axes: [1, 3] } },
    }],
    outputs: ['y'],
  };
}

function emptyGraph(overrides = {}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { x: { dtype: 'float32', shape: [1] } },
    nodes: [],
    outputs: ['x'],
    ...overrides,
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function assertLogicalError(operation, options = {}) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof GraphError, `expected GraphError, got ${String(error)}`);
  assert.equal(error.code, 'INVALID_GRAPH');
  if (options.diagnostic) assert.equal(error.diagnostic, options.diagnostic);
  if (options.path) assert.equal(error.path, options.path);
  if (options.message) assert.match(error.message, options.message);
  return error;
}

test('constant documents produce immutable logical descriptors without Tensor allocation', () => {
  const document = constantDocument();
  const weights = [{ name: 'unused.weight', dtype: 'int8', shape: [2, 4] }];
  const logical = parseGraphDocument(document, weights);

  assert.equal(logical.format, 'volvox-graph/v1');
  assert.equal('shape_system' in logical, false);
  assert.deepEqual(Object.keys(logical.dimensions), []);
  assert.deepEqual(logical.inputs.x, {
    kind: 'input', name: 'x', dtype: 'float32', shape: [1, 4],
  });
  assert.deepEqual(logical.weights['unused.weight'], {
    kind: 'weight', name: 'unused.weight', dtype: 'int8', shape: [2, 4], bank: null,
  });
  assert.deepEqual(logical.nodes[0].outputs.out, {
    tensor: 'y', dtype: 'float32', shape: [1, 4],
  });
  assert.deepEqual(logical.tensors.y, {
    kind: 'value',
    name: 'y',
    dtype: 'float32',
    shape: [1, 4],
    producerNodeId: 'identity',
    producerPort: 'out',
  });
  assert.deepEqual(logical.outputs, ['y']);
  assert.match(logical.fingerprint, /^volvox-logical-graph\/v1\|/);
  assert.equal('buffer' in logical.inputs.x, false);
  assert.equal('sizeBytes' in logical.inputs.x, false);

  for (const value of [
    logical,
    logical.dimensions,
    logical.inputs,
    logical.inputs.x,
    logical.inputs.x.shape,
    logical.weights,
    logical.weights['unused.weight'],
    logical.nodes,
    logical.nodes[0],
    logical.nodes[0].inputs,
    logical.nodes[0].outputs,
    logical.nodes[0].outputs.out,
    logical.nodes[0].params,
    logical.nodes[0].params.nested,
    logical.nodes[0].params.nested.axes,
    logical.outputs,
    logical.tensors,
    logical.tensors.y,
  ]) assert.equal(Object.isFrozen(value), true);

  document.inputs.x.shape[0] = 99;
  document.nodes[0].outputs.out.shape[1] = 99;
  document.nodes[0].params.nested.axes[0] = 99;
  weights[0].shape[0] = 99;
  assert.deepEqual(logical.inputs.x.shape, [1, 4]);
  assert.deepEqual(logical.nodes[0].outputs.out.shape, [1, 4]);
  assert.deepEqual(logical.nodes[0].params.nested.axes, [1, 3]);
  assert.deepEqual(logical.weights['unused.weight'].shape, [2, 4]);
});

test('node params is a required exact v1 field even when empty', () => {
  const document = constantDocument();
  delete document.nodes[0].params;
  assertLogicalError(
    () => parseGraphDocument(document),
    { path: 'nodes[0]', message: /requires field 'params'/ },
  );
});

function symbolicDocument(reverseObjectOrder = false) {
  const dimensions = reverseObjectOrder
    ? { S: { multiple_of: 2, max: 16, min: 2 }, B: { max: 4, min: 1 } }
    : { B: { min: 1, max: 4 }, S: { min: 2, max: 16, multiple_of: 2 } };
  const inputs = reverseObjectOrder
    ? {
        mask: { shape: ['B', 'S'], dtype: 'int32' },
        ids: { shape: ['B', 'S'], dtype: 'int32' },
      }
    : {
        ids: { dtype: 'int32', shape: ['B', 'S'] },
        mask: { dtype: 'int32', shape: ['B', 'S'] },
      };
  const nodeInputs = reverseObjectOrder
    ? { weight: 'token.weight', input: 'ids' }
    : { input: 'ids', weight: 'token.weight' };
  const nodeOutputs = reverseObjectOrder
    ? {
        auxiliary: { shape: ['B', 'S'], dtype: 'int32', tensor: 'echo_ids' },
        out: { shape: ['S', 'B', 8], dtype: 'float32', tensor: 'hidden' },
      }
    : {
        out: { tensor: 'hidden', dtype: 'float32', shape: ['S', 'B', 8] },
        auxiliary: { tensor: 'echo_ids', dtype: 'int32', shape: ['B', 'S'] },
      };
  const params = reverseObjectOrder
    ? { nested: { z: 2, a: 1 }, mode: 'lookup' }
    : { mode: 'lookup', nested: { a: 1, z: 2 } };
  const body = {
    dimensions,
    inputs,
    nodes: [{
      id: 'embedding',
      opType: 'Embedding',
      inputs: nodeInputs,
      outputs: nodeOutputs,
      params,
    }],
    outputs: ['hidden', 'echo_ids'],
  };
  return reverseObjectOrder
    ? {
        outputs: body.outputs,
        nodes: body.nodes,
        inputs: body.inputs,
        dimensions: body.dimensions,
        format: 'volvox-graph/v1',
      }
    : {
        format: 'volvox-graph/v1',
        ...body,
      };
}

test('symbolic documents canonicalize maps and keep output shape assertions for DS2', () => {
  const first = parseGraphDocument(symbolicDocument(), [
    { name: 'z.unused', dtype: 'float32', shape: [1] },
    { name: 'token.weight', dtype: 'float32', shape: [64, 8] },
  ]);
  const reordered = parseGraphDocument(symbolicDocument(true), [
    { shape: [64, 8], dtype: 'float32', name: 'token.weight' },
    { shape: [1], dtype: 'float32', name: 'z.unused' },
  ]);

  assert.deepEqual(Object.keys(first.dimensions), ['B', 'S']);
  assert.deepEqual(first.dimensions.B, { name: 'B', min: 1, max: 4, multiple_of: 1 });
  assert.deepEqual(Object.keys(first.inputs), ['ids', 'mask']);
  assert.deepEqual(Object.keys(first.nodes[0].inputs), ['input', 'weight']);
  assert.deepEqual(Object.keys(first.nodes[0].outputs), ['auxiliary', 'out']);
  // Identity with this shape would be contradictory, but DS1 intentionally
  // retains assertions and leaves operator inference to DS2.
  assert.deepEqual(first.nodes[0].outputs.out.shape, ['S', 'B', 8]);
  assert.deepEqual(first.outputs, ['hidden', 'echo_ids']);
  assert.equal(first.fingerprint, reordered.fingerprint);
});

test('canonical fingerprint preserves semantic node-array and public-output-array order', () => {
  const document = emptyGraph({
    nodes: [
      {
        id: 'first', opType: 'Identity', inputs: { input: 'x' },
        outputs: { out: { tensor: 'a', dtype: 'float32', shape: [1] } },
        params: {},
      },
      {
        id: 'second', opType: 'Identity', inputs: { input: 'x' },
        outputs: { out: { tensor: 'b', dtype: 'float32', shape: [1] } },
        params: {},
      },
    ],
    outputs: ['a', 'b'],
  });
  const baseline = parseGraphDocument(document);
  const swappedNodes = clone(document);
  swappedNodes.nodes.reverse();
  const swappedOutputs = clone(document);
  swappedOutputs.outputs.reverse();

  assert.notEqual(baseline.fingerprint, parseGraphDocument(swappedNodes).fingerprint);
  assert.notEqual(baseline.fingerprint, parseGraphDocument(swappedOutputs).fingerprint);
});

test('shape_system is rejected as an unknown closed-schema root field', () => {
  assertLogicalError(
    () => parseGraphDocument({
      ...constantDocument(),
      shape_system: 'volvox-bounded-shape/v1',
    }),
    { diagnostic: 'INVALID_GRAPH', path: 'graph', message: /unsupported field 'shape_system'/ },
  );
});

test('new v1 explicitly rejects legacy split output maps', () => {
  const document = constantDocument();
  document.nodes[0] = {
    id: 'identity',
    opType: 'Identity',
    inputs: { input: 'x' },
    outputs: { out: 'y' },
    outputs_shape: { out: [1, 4] },
    outputs_dtype: { out: 'float32' },
  };
  assertLogicalError(
    () => parseGraphDocument(document),
    { diagnostic: 'REEXPORT_REQUIRED', path: 'nodes[0]', message: /legacy.*outputs_shape.*re-export/i },
  );
});

test('strict schema rejects malformed roots, dimensions, descriptors, and JSON params', () => {
  const cases = [];

  cases.push([emptyGraph({ format: 'volvox-graph/v2' }), /format.*exactly/]);
  const missingDimensions = emptyGraph();
  delete missingDimensions.dimensions;
  cases.push([missingDimensions, /requires field 'dimensions'/]);
  cases.push([{ ...emptyGraph(), extra: true }, /unsupported field 'extra'/]);
  cases.push([emptyGraph({ dimensions: { 'bad-name': { min: 1, max: 2 } } }), /symbol names must match/]);
  cases.push([emptyGraph({
    dimensions: { [`A${'x'.repeat(64)}`]: { min: 1, max: 2 } },
  }), /symbol names must match/]);
  cases.push([emptyGraph({ dimensions: { B: { min: 2, max: 3, multiple_of: 4 } } }), /no multiple of 4/]);
  cases.push([emptyGraph({
    inputs: { x: { dtype: 'float16', shape: [1] } },
  }), /must be one of/]);
  cases.push([emptyGraph({
    inputs: { x: { dtype: 'float32', shape: ['Unknown'] } },
  }), /undeclared symbol 'Unknown'/]);

  const badOperator = constantDocument();
  badOperator.nodes[0].opType = 'MadeUp';
  cases.push([badOperator, /unsupported runtime operator/]);
  const badOutput = constantDocument();
  badOutput.nodes[0].outputs.out.extra = 1;
  cases.push([badOutput, /unsupported field 'extra'/]);
  const nonFinite = constantDocument();
  nonFinite.nodes[0].params.value = Number.NaN;
  cases.push([nonFinite, /finite decoded JSON number/]);

  for (const [document, message] of cases) {
    assertLogicalError(() => parseGraphDocument(document), { message });
  }

  assertLogicalError(
    () => parseGraphDocument(new Date()),
    { path: 'graph', message: /plain decoded JSON object/ },
  );
  const sparse = emptyGraph();
  sparse.nodes = new Array(1);
  assertLogicalError(() => parseGraphDocument(sparse), { message: /array hole/ });
});

test('topological resolution rejects forward references and duplicate identities', () => {
  const forward = emptyGraph({
    nodes: [
      {
        id: 'consumer', opType: 'Identity', inputs: { input: 'later' },
        outputs: { out: { tensor: 'result', dtype: 'float32', shape: [1] } },
        params: {},
      },
      {
        id: 'producer', opType: 'Identity', inputs: { input: 'x' },
        outputs: { out: { tensor: 'later', dtype: 'float32', shape: [1] } },
        params: {},
      },
    ],
    outputs: ['result'],
  });
  assertLogicalError(
    () => parseGraphDocument(forward),
    { path: 'nodes[0].inputs.input', message: /unresolved tensor 'later'.*topological/ },
  );

  const duplicateNode = constantDocument();
  duplicateNode.nodes.push({
    id: 'identity', opType: 'Identity', inputs: { input: 'y' },
    outputs: { out: { tensor: 'z', dtype: 'float32', shape: [1, 4] } },
    params: {},
  });
  duplicateNode.outputs = ['z'];
  assertLogicalError(() => parseGraphDocument(duplicateNode), { message: /duplicates node id/ });

  const duplicateTensor = constantDocument();
  duplicateTensor.nodes[0].outputs = {
    a: { tensor: 'y', dtype: 'float32', shape: [1, 4] },
    b: { tensor: 'y', dtype: 'float32', shape: [1, 4] },
  };
  assertLogicalError(() => parseGraphDocument(duplicateTensor), { message: /duplicates tensor 'y'/ });

  const overwritesInput = constantDocument();
  overwritesInput.nodes[0].outputs.out.tensor = 'x';
  overwritesInput.outputs = ['x'];
  assertLogicalError(() => parseGraphDocument(overwritesInput), { message: /duplicates tensor 'x'/ });

  const blankPort = constantDocument();
  blankPort.nodes[0].outputs = {
    ' ': { tensor: 'y', dtype: 'float32', shape: [1, 4] },
  };
  assertLogicalError(() => parseGraphDocument(blankPort), { message: /port.*non-empty string/ });
});

test('graph outputs must be non-empty, unique, and already resolved', () => {
  assertLogicalError(
    () => parseGraphDocument(emptyGraph({ outputs: [] })),
    { path: 'outputs', message: /non-empty array/ },
  );
  assertLogicalError(
    () => parseGraphDocument(emptyGraph({ outputs: ['x', 'x'] })),
    { path: 'outputs[1]', message: /duplicates graph output/ },
  );
  assertLogicalError(
    () => parseGraphDocument(emptyGraph({ outputs: ['missing'] })),
    { path: 'outputs[0]', message: /unresolved tensor/ },
  );
});

test('weights are fixed, unique, checked, and participate in topology', () => {
  const graph = parseGraphDocument(emptyGraph({
    nodes: [{
      id: 'linear', opType: 'Linear', inputs: { input: 'x', weight: 'w' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1] } },
      params: {},
    }],
    outputs: ['y'],
  }), [{ name: 'w', dtype: 'float32', shape: [1, 1] }]);
  assert.equal(graph.nodes[0].inputs.weight, 'w');

  assertLogicalError(
    () => parseGraphDocument(emptyGraph(), [
      { name: 'w', dtype: 'float32', shape: [1] },
      { name: 'w', dtype: 'float32', shape: [1] },
    ]),
    { message: /duplicates tensor 'w'/ },
  );
  assertLogicalError(
    () => parseGraphDocument(emptyGraph(), [
      { name: 'x', dtype: 'float32', shape: [1] },
    ]),
    { message: /duplicates a public input/ },
  );
  assertLogicalError(
    () => parseGraphDocument(emptyGraph({
      dimensions: { B: { min: 1, max: 4 } },
      inputs: { x: { dtype: 'float32', shape: ['B'] } },
    }), [{ name: 'w', dtype: 'float32', shape: ['B'] }]),
    { message: /weight shapes must contain constant dimensions only/ },
  );
  assertLogicalError(
    () => parseGraphDocument(emptyGraph(), [
      { name: 'huge', dtype: 'float32', shape: [Number.MAX_SAFE_INTEGER] },
    ]),
    { message: /byte length.*safe integer range/ },
  );
});

test('fingerprints normalize omitted multiple_of and reject aliased parameter objects', () => {
  const omitted = emptyGraph({
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B'] } },
  });
  const explicit = clone(omitted);
  explicit.dimensions.B.multiple_of = 1;
  assert.equal(
    parseGraphDocument(omitted).fingerprint,
    parseGraphDocument(explicit).fingerprint,
  );

  const shared = { value: 1 };
  const aliased = constantDocument();
  aliased.nodes[0].params = { left: shared, right: shared };
  assertLogicalError(
    () => parseGraphDocument(aliased),
    { message: /cycles or shared object aliases/ },
  );
});

function quantizedDocument(reverseQuantizationOrder = false) {
  const references = reverseQuantizationOrder
    ? {
        y: {
          zero_point_tensor: 'zy', scale_tensor: 'sy', scheme: 'per_tensor',
        },
        w: {
          zero_point_tensor: 'zw', scale_tensor: 'sw', axis: -2, scheme: 'per_axis',
        },
        x: {
          zero_point_tensor: 'zx', scale_tensor: 'sx', scheme: 'per_tensor',
        },
      }
    : {
        x: {
          scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx',
        },
        w: {
          scheme: 'per_axis', axis: -2, scale_tensor: 'sw', zero_point_tensor: 'zw',
        },
        y: {
          scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy',
        },
      };
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 8 } },
    inputs: { x: { dtype: 'int8', shape: ['B', 4] } },
    nodes: [{
      id: 'quantized-linear',
      opType: 'QLinear',
      inputs: { input: 'x', weight: 'w' },
      outputs: {
        out: { tensor: 'y', dtype: 'uint8', shape: ['B', 3] },
      },
      params: {},
    }],
    outputs: ['y'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: references,
    },
  };
}

function quantizedWeights() {
  return [
    { name: 'w', dtype: 'int8', shape: [3, 4] },
    { name: 'sx', dtype: 'float32', shape: [1] },
    { name: 'zx', dtype: 'int8', shape: [1] },
    { name: 'sw', dtype: 'float32', shape: [3] },
    { name: 'zw', dtype: 'int8', shape: [3] },
    { name: 'sy', dtype: 'float32', shape: [1] },
    { name: 'zy', dtype: 'uint8', shape: [1] },
    { name: 'sx.alternate', dtype: 'float32', shape: [1] },
  ];
}

test('central affine references are canonical, immutable, and attached to target descriptors', () => {
  const document = quantizedDocument();
  const logical = parseGraphDocument(document, quantizedWeights());

  assert.deepEqual(logical.quantization, {
    format: 'volvox-affine-safetensors/v1',
    tensors: {
      w: { scheme: 'per_axis', axis: 0, scale_tensor: 'sw', zero_point_tensor: 'zw' },
      x: { scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx' },
      y: { scheme: 'per_tensor', scale_tensor: 'sy', zero_point_tensor: 'zy' },
    },
  });
  assert.strictEqual(logical.inputs.x.quantization, logical.quantization.tensors.x);
  assert.strictEqual(logical.weights.w.quantization, logical.quantization.tensors.w);
  assert.strictEqual(logical.tensors.y.quantization, logical.quantization.tensors.y);
  assert.strictEqual(
    logical.nodes[0].outputs.out.quantization,
    logical.quantization.tensors.y,
  );
  assert.equal('scale' in logical.quantization.tensors.x, false);
  assert.equal('zero_point' in logical.quantization.tensors.x, false);

  for (const value of [
    logical.quantization,
    logical.quantization.tensors,
    logical.quantization.tensors.x,
    logical.quantization.tensors.w,
    logical.inputs.x,
    logical.weights.w,
    logical.tensors.y,
    logical.nodes[0].outputs.out,
  ]) assert.equal(Object.isFrozen(value), true);

  document.quantization.tensors.x.scale_tensor = 'changed';
  document.quantization.tensors.w.axis = 1;
  assert.equal(logical.quantization.tensors.x.scale_tensor, 'sx');
  assert.equal(logical.quantization.tensors.w.axis, 0);
});

test('quantization references participate canonically in the graph fingerprint', () => {
  const weights = quantizedWeights();
  const baseline = parseGraphDocument(quantizedDocument(), weights);
  const reordered = parseGraphDocument(
    quantizedDocument(true),
    [...weights].reverse(),
  );
  assert.equal(baseline.fingerprint, reordered.fingerprint);

  const changed = quantizedDocument();
  changed.quantization.tensors.x.scale_tensor = 'sx.alternate';
  assert.notEqual(
    baseline.fingerprint,
    parseGraphDocument(changed, weights).fingerprint,
  );
  assert.notEqual(baseline.fingerprint, parseGraphDocument(constantDocument()).fingerprint);
});

test('central affine table validates target and fixed parameter dtype and shape metadata', () => {
  const cases = [];

  const badFormat = quantizedDocument();
  badFormat.quantization.format = 'volvox-affine-safetensors/v2';
  cases.push([badFormat, quantizedWeights(), /quantization\.format.*exactly/]);

  const empty = quantizedDocument();
  empty.quantization.tensors = {};
  cases.push([empty, quantizedWeights(), /quantization\.tensors.*non-empty/]);

  const unknown = quantizedDocument();
  unknown.quantization.tensors.unknown = unknown.quantization.tensors.x;
  delete unknown.quantization.tensors.x;
  cases.push([unknown, quantizedWeights(), /targets unknown tensor 'unknown'/]);

  const floatTarget = quantizedDocument();
  floatTarget.quantization.tensors.sx = floatTarget.quantization.tensors.x;
  delete floatTarget.quantization.tensors.x;
  cases.push([floatTarget, quantizedWeights(), /target 'sx' must have int8 or uint8/]);

  const missingParameter = quantizedWeights().filter((weight) => weight.name !== 'sx');
  cases.push([quantizedDocument(), missingParameter, /must exist in supplied fixed weights/]);

  const wrongScale = quantizedWeights();
  wrongScale.find((weight) => weight.name === 'sx').dtype = 'int32';
  cases.push([quantizedDocument(), wrongScale, /'sx' must have float32 storage/]);

  const wrongZero = quantizedWeights();
  wrongZero.find((weight) => weight.name === 'zy').dtype = 'int8';
  cases.push([quantizedDocument(), wrongZero, /'zy' dtype must match target dtype 'uint8'/]);

  const wrongScalar = quantizedWeights();
  wrongScalar.find((weight) => weight.name === 'sx').shape = [2];
  cases.push([quantizedDocument(), wrongScalar, /scalar shape \[1\]/]);

  const wrongAxisExtent = quantizedWeights();
  wrongAxisExtent.find((weight) => weight.name === 'sw').shape = [4];
  cases.push([quantizedDocument(), wrongAxisExtent, /must both have shape \[3\]/]);

  for (const [document, weights, message] of cases) {
    assertLogicalError(() => parseGraphDocument(document, weights), { message });
  }
});

test('per-axis references reject invalid or symbolic target axes', () => {
  const nonInteger = quantizedDocument();
  nonInteger.quantization.tensors.w.axis = 0.5;
  assertLogicalError(
    () => parseGraphDocument(nonInteger, quantizedWeights()),
    { message: /integer axis/ },
  );

  const outside = quantizedDocument();
  outside.quantization.tensors.w.axis = -3;
  assertLogicalError(
    () => parseGraphDocument(outside, quantizedWeights()),
    { message: /outside target rank 2/ },
  );

  const symbolic = quantizedDocument();
  symbolic.quantization.tensors.x = {
    scheme: 'per_axis', axis: 0, scale_tensor: 'sx', zero_point_tensor: 'zx',
  };
  assertLogicalError(
    () => parseGraphDocument(symbolic, quantizedWeights()),
    { message: /target axis 0 is symbolic/ },
  );
});

test('quantization parameters cannot be targets, public outputs, or node products', () => {
  const selfTarget = quantizedDocument();
  selfTarget.quantization.tensors.zx = {
    scheme: 'per_tensor', scale_tensor: 'sx', zero_point_tensor: 'zx',
  };
  assertLogicalError(
    () => parseGraphDocument(selfTarget, quantizedWeights()),
    { message: /parameter tensor 'zx' cannot itself be a quantization target/ },
  );

  const publicParameter = quantizedDocument();
  publicParameter.outputs = ['sy'];
  assertLogicalError(
    () => parseGraphDocument(publicParameter, quantizedWeights()),
    { path: 'outputs[0]', message: /quantization parameter tensor 'sy'.*public graph output/ },
  );

  const producedParameter = quantizedDocument();
  producedParameter.nodes[0].outputs.out.tensor = 'sx';
  producedParameter.outputs = ['sx'];
  assertLogicalError(
    () => parseGraphDocument(producedParameter, quantizedWeights()),
    { message: /duplicates tensor 'sx'/ },
  );
});

test('quantization schema rejects inline payloads, extra fields, and duplicate parameter references', () => {
  const inlineInput = quantizedDocument();
  inlineInput.inputs.x.quantization = { scale: 1, zero_point: 0 };
  assertLogicalError(
    () => parseGraphDocument(inlineInput, quantizedWeights()),
    { message: /inline quantization is forbidden/ },
  );

  const inlineOutput = quantizedDocument();
  inlineOutput.nodes[0].outputs.out.quantization = { scale: 1, zero_point: 0 };
  assertLogicalError(
    () => parseGraphDocument(inlineOutput, quantizedWeights()),
    { message: /inline quantization is forbidden/ },
  );

  const inlineWeight = quantizedWeights();
  inlineWeight[0].quantization = { scheme: 'per_tensor' };
  assertLogicalError(
    () => parseGraphDocument(quantizedDocument(), inlineWeight),
    { message: /inline quantization is forbidden/ },
  );

  const inlineParams = quantizedDocument();
  inlineParams.nodes[0].params = { nested: { scale_tensor: 'sx' } };
  assertLogicalError(
    () => parseGraphDocument(inlineParams, quantizedWeights()),
    { message: /retired inline affine metadata/ },
  );

  const numericPayload = quantizedDocument();
  numericPayload.quantization.tensors.x.scale = 0.25;
  assertLogicalError(
    () => parseGraphDocument(numericPayload, quantizedWeights()),
    { message: /unsupported field 'scale'/ },
  );

  const sameParameters = quantizedDocument();
  sameParameters.quantization.tensors.x.zero_point_tensor = 'sx';
  assertLogicalError(
    () => parseGraphDocument(sameParameters, quantizedWeights()),
    { message: /requires distinct scale and zero-point/ },
  );
});

test('QuantizeLinear and DequantizeLinear operands must match central references', () => {
  const document = {
    format: 'volvox-graph/v1',
    dimensions: {},
    inputs: { f: { dtype: 'float32', shape: [2] } },
    nodes: [{
      id: 'quantize',
      opType: 'QuantizeLinear',
      inputs: { input: 'f', scale: 'scale', zero_point: 'zero' },
      outputs: { out: { tensor: 'q', dtype: 'uint8', shape: [2] } },
      params: {},
    }],
    outputs: ['q'],
    quantization: {
      format: 'volvox-affine-safetensors/v1',
      tensors: {
        q: {
          scheme: 'per_tensor', scale_tensor: 'scale', zero_point_tensor: 'zero',
        },
      },
    },
  };
  const weights = [
    { name: 'scale', dtype: 'float32', shape: [1] },
    { name: 'wrong.scale', dtype: 'float32', shape: [1] },
    { name: 'zero', dtype: 'uint8', shape: [1] },
  ];
  assert.equal(parseGraphDocument(document, weights).outputs[0], 'q');

  document.nodes[0].inputs.scale = 'wrong.scale';
  assertLogicalError(
    () => parseGraphDocument(document, weights),
    { message: /QuantizeLinear parameter inputs must match/ },
  );
});

test('ill-formed Unicode names and map keys are rejected before UTF-8 canonicalization', () => {
  const badName = emptyGraph({
    inputs: { [`bad\ud800`]: { dtype: 'float32', shape: [1] } },
    outputs: [`bad\ud800`],
  });
  assertLogicalError(
    () => parseGraphDocument(badName),
    { message: /unpaired UTF-16 surrogate/ },
  );

  const badParams = constantDocument();
  badParams.nodes[0].params = { [`bad\udfff`]: true };
  assertLogicalError(
    () => parseGraphDocument(badParams),
    { message: /unpaired UTF-16 surrogate/ },
  );
});

function bankDocument({ dimension = 'F', min = 1, max = 8 } = {}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: { [dimension]: { min, max } },
    banks: { 'adapters.down': dimension },
    inputs: { x: { dtype: 'float32', shape: [1, 4] } },
    nodes: [{
      id: 'identity',
      opType: 'Identity',
      inputs: { input: 'x' },
      outputs: { out: { tensor: 'y', dtype: 'float32', shape: [1, 4] } },
      params: {},
    }],
    outputs: ['y'],
  };
}

function bankWeights(slots = 4) {
  return [{ name: 'adapters.down', dtype: 'float32', shape: [slots, 2, 4] }];
}

test('a declared bank records its slot bounds on the weight descriptor', () => {
  const graph = parseGraphDocument(bankDocument(), bankWeights(4));
  assert.deepEqual(graph.banks['adapters.down'], {
    name: 'adapters.down', dimension: 'F', min: 1, max: 8,
  });
  assert.deepEqual(graph.weights['adapters.down'].bank, {
    name: 'adapters.down', dimension: 'F', min: 1, max: 8,
  });
  // The payload shape stays concrete; only the slot extent is governed.
  assert.deepEqual(graph.weights['adapters.down'].shape, [4, 2, 4]);
});

test('filling a bank slot keeps the definition fingerprint', () => {
  const four = parseGraphDocument(bankDocument(), bankWeights(4));
  const five = parseGraphDocument(bankDocument(), bankWeights(5));
  assert.equal(
    five.fingerprint, four.fingerprint,
    'adding a family within bounds must not force a recompile',
  );

  // A non-bank weight keeps shape identity, so the same change is a new model.
  const plainDocument = bankDocument();
  delete plainDocument.banks;
  assert.notEqual(
    parseGraphDocument(plainDocument, bankWeights(5)).fingerprint,
    parseGraphDocument(plainDocument, bankWeights(4)).fingerprint,
  );
});

test('widening bank bounds is a new definition', () => {
  assert.notEqual(
    parseGraphDocument(bankDocument({ max: 16 }), bankWeights(4)).fingerprint,
    parseGraphDocument(bankDocument({ max: 8 }), bankWeights(4)).fingerprint,
  );
});

test('bank declarations are validated against dimensions and weights', () => {
  const undeclared = bankDocument();
  undeclared.banks = { 'adapters.down': 'Missing' };
  assert.throws(() => parseGraphDocument(undeclared, bankWeights()), (error) => {
    assert.ok(error instanceof GraphError);
    assert.match(error.message, /undeclared dimension 'Missing'/);
    return true;
  });

  assert.throws(
    () => parseGraphDocument(bankDocument(), [
      { name: 'other', dtype: 'float32', shape: [4, 2, 4] },
    ]),
    /not a supplied fixed weight/,
  );

  assert.throws(
    () => parseGraphDocument(bankDocument({ max: 3 }), bankWeights(4)),
    /supplies 4 slots, outside the 'F' bound \[1, 3\]/,
  );

  assert.throws(
    () => parseGraphDocument(bankDocument(), [
      { name: 'adapters.down', dtype: 'float32', shape: [4] },
    ]),
    /needs a slot axis and at least one payload axis/,
  );
});
