import test from 'node:test';
import assert from 'node:assert/strict';

import {
  OperatorShapeContractError,
  WAVE_B_OPERATOR_SHAPE_FUNCTIONS,
  WAVE_B_SHAPE_FUNCTION_IDS,
  getOperatorShapeContract,
  inferConcreteOperatorShapes,
  proveOperatorShapeDomain,
} from '../ts/ops/operatorShapeContracts.js';
import {
  operatorShapeContracts,
  operatorShapeFunctionIds,
} from '../ts/generated/kernelRegistry.js';
import { ShapeEnvironment } from '../ts/ops/shapeSystem.js';
import { parseGraphDocument } from '../ts/core/Graph.js';
import {
  ResolvedShapePlanError,
  proveGraphShapeDomain,
  resolveGraphShapes,
} from '../ts/core/ResolvedShapePlan.js';

function tensor(shape, dtype = 'float32', quantization = undefined) {
  return quantization === undefined ? { shape, dtype } : { shape, dtype, quantization };
}

function contractError(operation, code, pattern = undefined) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof OperatorShapeContractError);
  assert.equal(error.code, code);
  if (pattern !== undefined) assert.match(error.message, pattern);
  return error;
}

const environment = new ShapeEnvironment([
  { name: 'B', min: 1, max: 8 },
  { name: 'S', min: 1, max: 32 },
  { name: 'T', min: 1, max: 4 },
]);

const perAxis3 = Object.freeze({
  scheme: 'per_axis',
  axis: 1,
  scales: [0.5, 0.25, 0.125],
  zero_points: [0, 0, 0],
});

test('Wave-B routes use generated canonical IDs and expose normalized variadic ports', () => {
  assert.equal(WAVE_B_SHAPE_FUNCTION_IDS.broadcastArithmetic, 'volvox.shape.broadcast-arithmetic.v1');
  assert.equal(WAVE_B_SHAPE_FUNCTION_IDS.broadcastComparison, 'volvox.shape.broadcast-comparison.v1');
  assert.equal(WAVE_B_SHAPE_FUNCTION_IDS.transpose, operatorShapeFunctionIds.Transpose);
  assert.equal(WAVE_B_OPERATOR_SHAPE_FUNCTIONS.Sub, operatorShapeFunctionIds.Sub);
  assert.equal(WAVE_B_OPERATOR_SHAPE_FUNCTIONS.ReduceSum, operatorShapeFunctionIds.ReduceSum);
  assert.equal(WAVE_B_OPERATOR_SHAPE_FUNCTIONS.GatherElements, operatorShapeFunctionIds.GatherElements);
  for (const [operator, shapeFunctionId] of Object.entries(WAVE_B_OPERATOR_SHAPE_FUNCTIONS)) {
    assert.equal(shapeFunctionId, operatorShapeFunctionIds[operator]);
    assert.equal(operatorShapeContracts[operator].classification, 'canonical');
  }
  assert.deepEqual(getOperatorShapeContract('Concat').ports.variadicInputs, {
    prefix: 'input', minimum: 2,
  });
  assert.deepEqual(getOperatorShapeContract('Split').ports.variadicOutputs, {
    prefix: 'out', minimum: 1,
  });
  assert.equal(Object.isFrozen(WAVE_B_OPERATOR_SHAPE_FUNCTIONS), true);
});

test('Sub/Div, comparisons, and Where infer right-aligned broadcast shapes while Add stays exact', () => {
  assert.deepEqual(inferConcreteOperatorShapes('Sub', {
    inputs: { a: tensor([2, 1, 4]), b: tensor([1, 3, 4]) },
  }).out, { shape: [2, 3, 4], dtype: 'float32' });
  assert.deepEqual(inferConcreteOperatorShapes('Equal', {
    inputs: { a: tensor([2, 1], 'int32'), b: tensor([3], 'int32') },
  }).out, { shape: [2, 3], dtype: 'int32' });
  const where = inferConcreteOperatorShapes('Where', {
    inputs: {
      condition: tensor([2, 1], 'int32'),
      a: tensor([1, 3]),
      b: tensor([2, 3]),
    },
  });
  assert.deepEqual(where.out, { shape: [2, 3], dtype: 'float32' });
  assert.equal(Object.isFrozen(where.out.shape), true);

  contractError(() => inferConcreteOperatorShapes('Add', {
    inputs: { a: tensor([2, 1]), b: tensor([2, 3]) },
  }), 'SHAPE_MISMATCH', /broadcasting is explicit/);
  contractError(() => inferConcreteOperatorShapes('GreaterOrEqual', {
    inputs: { a: tensor([2]), b: tensor([2], 'int32') },
  }), 'INVALID_DTYPE', /int32/);
});

test('broadcast domain proof accepts shared symbols and rejects merely overlapping intervals', () => {
  const accepted = proveOperatorShapeDomain('Div', {
    environment,
    inputs: { a: tensor(['B', 1, 4]), b: tensor([1, 'S', 4]) },
  });
  assert.equal(accepted.supported, true);
  assert.deepEqual(accepted.outputs.out.shape, ['B', 'S', 4]);

  const rejected = proveOperatorShapeDomain('Sub', {
    environment,
    inputs: { a: tensor(['B']), b: tensor(['S']) },
  });
  assert.equal(rejected.supported, false);
  assert.equal(rejected.code, 'UNPROVABLE_DYNAMIC_BROADCAST');
});

test('last-axis reductions and ArgMax normalize negative axes and keepdims', () => {
  assert.deepEqual(inferConcreteOperatorShapes('ReduceMean', {
    inputs: { input: tensor([2, 5, 7]) },
    params: { axis: -1, keepdims: false },
  }).out.shape, [2, 5]);
  assert.deepEqual(inferConcreteOperatorShapes('ReduceSum', {
    inputs: { input: tensor([2, 5, 7]) },
    params: { axis: 2 },
  }).out.shape, [2, 5, 1]);
  assert.deepEqual(inferConcreteOperatorShapes('ArgMax', {
    inputs: { input: tensor([2, 5, 7]) },
    params: { axis: -2, keepdims: false, select_last_index: 0 },
  }).out, { shape: [2, 7], dtype: 'int32' });

  contractError(() => inferConcreteOperatorShapes('ReduceSum', {
    inputs: { input: tensor([2, 5, 7]) }, params: { axis: 1 },
  }), 'INVALID_PARAMS', /last axis/);
  contractError(() => inferConcreteOperatorShapes('ArgMax', {
    inputs: { input: tensor([2, 5]) }, params: { select_last_index: 1 },
  }), 'INVALID_PARAMS', /first-index/);
});

test('Transpose and singleton structural operations remap fixed per-axis metadata', () => {
  const transpose = proveOperatorShapeDomain('Transpose', {
    environment,
    inputs: { input: tensor(['B', 3, 'S'], 'int8', perAxis3) },
    params: { perm: [0, 2, 1] },
  });
  assert.equal(transpose.supported, true);
  assert.deepEqual(transpose.outputs.out.shape, ['B', 'S', 3]);
  assert.equal(transpose.outputs.out.quantization.axis, 2);

  assert.deepEqual(inferConcreteOperatorShapes('Squeeze', {
    inputs: { input: tensor([2, 1, 3]) }, params: { axes: [-2] },
  }).out.shape, [2, 3]);
  assert.deepEqual(inferConcreteOperatorShapes('Unsqueeze', {
    inputs: { input: tensor([2, 3]) }, params: { axes: [1, -1] },
  }).out.shape, [2, 1, 3, 1]);
  contractError(() => inferConcreteOperatorShapes('Squeeze', {
    inputs: { input: tensor([2, 3]) }, params: { axes: [1] },
  }), 'SHAPE_MISMATCH', /must be 1/);
});

test('Flatten and Reshape conserve products conservatively for bounded symbols', () => {
  const flatten = proveOperatorShapeDomain('Flatten', {
    environment,
    inputs: { input: tensor(['B', 1, 4]) },
    params: { axis: 1 },
  });
  assert.equal(flatten.supported, true);
  assert.deepEqual(flatten.outputs.out.shape, ['B', 4]);

  const unrepresentable = proveOperatorShapeDomain('Flatten', {
    environment,
    inputs: { input: tensor(['B', 'S', 4]) },
    params: { axis: 1 },
  });
  assert.equal(unrepresentable.supported, false);
  assert.equal(unrepresentable.code, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA');

  assert.deepEqual(inferConcreteOperatorShapes('Reshape', {
    inputs: { input: tensor([2, 3, 4]) }, params: { shape: [6, 4] },
  }).out.shape, [6, 4]);
  assert.deepEqual(inferConcreteOperatorShapes('Reshape', {
    inputs: { input: tensor([2, 4]) },
    params: { shape: ['B', 4] },
    declaredOutputs: { out: tensor([2, 4]) },
  }).out.shape, [2, 4]);
  const symbolic = proveOperatorShapeDomain('Reshape', {
    environment,
    inputs: { input: tensor(['B', 'S', 4]) },
    params: { shape: ['B', 'S', 4] },
  });
  assert.equal(symbolic.supported, true);
  const mismatch = proveOperatorShapeDomain('Reshape', {
    environment,
    inputs: { input: tensor(['B', 4]) },
    params: { shape: ['B', 8] },
  });
  assert.equal(mismatch.supported, false);
  assert.equal(mismatch.code, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA');
});

test('Expand uses an explicit normalized target and rejects expansion of a quantized axis', () => {
  const proof = proveOperatorShapeDomain('Expand', {
    environment,
    inputs: { input: tensor(['B', 1, 4]) },
    params: { shape: ['B', 'S', 4] },
  });
  assert.equal(proof.supported, true);
  assert.deepEqual(proof.outputs.out.shape, ['B', 'S', 4]);
  assert.deepEqual(inferConcreteOperatorShapes('Expand', {
    inputs: { input: tensor([2, 1, 4]) },
    params: { shape: ['B', 'S', 4] },
    declaredOutputs: { out: tensor([2, 3, 4]) },
  }).out.shape, [2, 3, 4]);

  const perAxisOne = {
    scheme: 'per_axis', axis: 1, scales: [0.5], zero_points: [0],
  };
  contractError(() => inferConcreteOperatorShapes('Expand', {
    inputs: { input: tensor([2, 1, 4], 'int8', perAxisOne) },
    params: { shape: [2, 3, 4] },
  }), 'UNSAFE_QUANTIZATION_TRANSFORM', /per-axis/);
});

test('variadic Concat and Split use numbered ports, checked extents, and affine slicing', () => {
  assert.deepEqual(inferConcreteOperatorShapes('Concat', {
    inputs: { input0: tensor([2, 3]), input1: tensor([2, 4]) },
    params: { axis: -1 },
  }).out.shape, [2, 7]);
  const concatProof = proveOperatorShapeDomain('Concat', {
    environment,
    inputs: { input0: tensor(['B', 3]), input1: tensor(['B', 4]) },
    params: { axis: 1 },
  });
  assert.equal(concatProof.supported, true);
  assert.deepEqual(concatProof.outputs.out.shape, ['B', 7]);

  const dynamicAxis = proveOperatorShapeDomain('Concat', {
    environment,
    inputs: { input0: tensor(['B', 'S']), input1: tensor(['B', 'S']) },
    params: { axis: 1 },
  });
  assert.equal(dynamicAxis.supported, false);
  assert.equal(dynamicAxis.code, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA');

  const quantization = {
    scheme: 'per_axis', axis: 1,
    scales: [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
    zero_points: [0, 0, 0, 0, 0, 0],
  };
  const split = inferConcreteOperatorShapes('Split', {
    inputs: { input: tensor([2, 6], 'int8', quantization) },
    params: { axis: 1, split: [2, 4] },
  });
  assert.deepEqual(Object.keys(split), ['out0', 'out1']);
  assert.deepEqual(split.out0.shape, [2, 2]);
  assert.deepEqual(split.out1.shape, [2, 4]);
  assert.deepEqual(split.out0.quantization.scales, [0.5, 0.25]);
  assert.deepEqual(split.out1.quantization.scales, [0.125, 0.0625, 0.03125, 0.015625]);
});

test('Slice and Pad normalize static geometry and fail closed on unrepresentable dynamic formulas', () => {
  const quantization = {
    scheme: 'per_axis', axis: 1,
    scales: Array.from({ length: 10 }, (_unused, index) => (index + 1) / 16),
    zero_points: new Array(10).fill(0),
  };
  const slice = inferConcreteOperatorShapes('Slice', {
    inputs: { input: tensor([3, 10], 'int8', quantization) },
    params: { starts: [-7], ends: [9], axes: [1], steps: [2] },
  });
  assert.deepEqual(slice.out.shape, [3, 3]);
  assert.deepEqual(slice.out.quantization.scales, [4 / 16, 6 / 16, 8 / 16]);

  const fullDynamic = proveOperatorShapeDomain('Slice', {
    environment,
    inputs: { input: tensor(['B', 'S']) },
    params: { starts: [0], ends: [32], axes: [1], steps: [1] },
  });
  assert.equal(fullDynamic.supported, true);
  assert.deepEqual(fullDynamic.outputs.out.shape, ['B', 'S']);
  const partialDynamic = proveOperatorShapeDomain('Slice', {
    environment,
    inputs: { input: tensor(['B', 'S']) },
    params: { starts: [1], ends: [32], axes: [1], steps: [1] },
  });
  assert.equal(partialDynamic.supported, false);
  assert.equal(partialDynamic.code, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA');

  assert.deepEqual(inferConcreteOperatorShapes('Pad', {
    inputs: { input: tensor([2, 3]) },
    params: { pads: [1, 0, 2, 1], value: 0 },
  }).out.shape, [5, 4]);
  const padProof = proveOperatorShapeDomain('Pad', {
    environment,
    inputs: { input: tensor(['B', 3]) },
    params: { pads: [0, 1, 0, 2] },
  });
  assert.equal(padProof.supported, true);
  assert.deepEqual(padProof.outputs.out.shape, ['B', 6]);
  const rejectedPad = proveOperatorShapeDomain('Pad', {
    environment,
    inputs: { input: tensor(['B', 3]) },
    params: { pads: [1, 0, 0, 0] },
  });
  assert.equal(rejectedPad.supported, false);
  assert.equal(rejectedPad.code, 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA');
});

test('Gather and GatherElements infer index-spliced shapes with conservative domain inequalities', () => {
  assert.deepEqual(inferConcreteOperatorShapes('Gather', {
    inputs: { input: tensor([2, 5, 3]), indices: tensor([4], 'int32') },
    params: { axis: 1 },
  }).out.shape, [2, 4, 3]);
  const gather = proveOperatorShapeDomain('Gather', {
    environment,
    inputs: { input: tensor(['B', 5, 3]), indices: tensor(['S'], 'int32') },
    params: { axis: -2 },
  });
  assert.equal(gather.supported, true);
  assert.deepEqual(gather.outputs.out.shape, ['B', 'S', 3]);

  const narrowEnvironment = new ShapeEnvironment([
    { name: 'B', min: 1, max: 8 },
    { name: 'T', min: 1, max: 4 },
  ]);
  const elements = proveOperatorShapeDomain('GatherElements', {
    environment: narrowEnvironment,
    inputs: {
      input: tensor(['B', 5, 3]),
      indices: tensor(['B', 'T', 3], 'int32'),
    },
    params: { axis: 1 },
  });
  assert.equal(elements.supported, true);
  assert.deepEqual(elements.outputs.out.shape, ['B', 'T', 3]);

  const tooWide = proveOperatorShapeDomain('GatherElements', {
    environment,
    inputs: {
      input: tensor(['B', 5, 3]),
      indices: tensor(['B', 'S', 3], 'int32'),
    },
    params: { axis: 2 },
  });
  assert.equal(tooWide.supported, false);
  assert.equal(tooWide.code, 'INVALID_DOMAIN');

  contractError(() => inferConcreteOperatorShapes('Gather', {
    inputs: {
      input: tensor([2, 3], 'int8', perAxis3),
      indices: tensor([2], 'int32'),
    },
    params: { axis: 1 },
  }), 'UNSAFE_QUANTIZATION_TRANSFORM', /reorder/);
});

function concatGraphDocument(inputs = { input0: 'left', input1: 'right' }) {
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: {
      left: { dtype: 'float32', shape: ['B', 2] },
      right: { dtype: 'float32', shape: ['B', 3] },
    },
    nodes: [{
      id: 'concat', opType: 'Concat', inputs,
      outputs: { out: { tensor: 'joined', dtype: 'float32', shape: ['B', 5] } },
      params: { axis: 1 },
    }],
    outputs: ['joined'],
  };
}

function affineConcatGraphDocument() {
  return {
    format: 'volvox-graph/v1',
    dimensions: {
      B: { min: 1, max: 1 },
      Q: { min: 1, max: 192 },
      M: { min: 211, max: 402 },
    },
    inputs: {
      imageTokens: { dtype: 'float32', shape: ['B', 210, 320] },
      questionTokens: { dtype: 'float32', shape: ['B', 'Q', 320] },
    },
    nodes: [{
      id: 'memory_concat',
      opType: 'Concat',
      inputs: { input0: 'imageTokens', input1: 'questionTokens' },
      outputs: {
        out: { tensor: 'memory', dtype: 'float32', shape: ['B', 'M', 320] },
      },
      params: { axis: 1 },
    }],
    outputs: ['memory'],
  };
}

function splitGraphDocument(outputs = {
  out0: { tensor: 'first', dtype: 'float32', shape: ['B', 2] },
  out1: { tensor: 'second', dtype: 'float32', shape: ['B', 4] },
}) {
  return {
    format: 'volvox-graph/v1',
    dimensions: { B: { min: 1, max: 4 } },
    inputs: { x: { dtype: 'float32', shape: ['B', 6] } },
    nodes: [{
      id: 'split', opType: 'Split', inputs: { input: 'x' }, outputs,
      params: { axis: 1, split: [2, 4] },
    }],
    outputs: Object.values(outputs).map((output) => output.tensor),
  };
}

function resolvedError(operation, code, pattern) {
  let error;
  try {
    operation();
  } catch (caught) {
    error = caught;
  }
  assert.ok(error instanceof ResolvedShapePlanError);
  assert.equal(error.code, code);
  assert.match(error.message, pattern);
}

test('graph-wide concrete binding and domain proof accept Concat and multi-output Split', () => {
  const concat = parseGraphDocument(concatGraphDocument());
  const concatProof = proveGraphShapeDomain(concat);
  assert.equal(concatProof.supported, true);
  assert.deepEqual(concatProof.outputs[0].shape, ['B', 5]);
  const concatPlan = resolveGraphShapes(concat, {
    left: { shape: [3, 2], data: new Float32Array(6) },
    right: { shape: [3, 3], data: new Float32Array(9) },
  });
  assert.deepEqual(concatPlan.outputs[0].shape, [3, 5]);
  assert.deepEqual(Object.keys(concatPlan.nodes[0].inputs), ['input0', 'input1']);

  const split = parseGraphDocument(splitGraphDocument());
  const splitProof = proveGraphShapeDomain(split);
  assert.equal(splitProof.supported, true);
  assert.deepEqual(splitProof.nodes[0].outputs.out0.shape, ['B', 2]);
  assert.deepEqual(splitProof.nodes[0].outputs.out1.shape, ['B', 4]);
  const splitPlan = resolveGraphShapes(split, {
    x: { shape: [3, 6], data: new Float32Array(18) },
  });
  assert.deepEqual(splitPlan.outputs.map((output) => output.shape), [[3, 2], [3, 4]]);
  assert.deepEqual(Object.keys(splitPlan.nodes[0].outputs), ['out0', 'out1']);
});

test('graph-wide affine Concat proves M=Q+210 and binds each concrete M exactly', () => {
  const graph = parseGraphDocument(affineConcatGraphDocument());
  const proof = proveGraphShapeDomain(graph);
  assert.equal(proof.supported, true);
  assert.deepEqual(proof.outputs[0].shape, ['B', 'M', 320]);
  assert.deepEqual(proof.nodes[0].facts, [
    'M=Q+210 exactly over the complete bounded Concat domain',
  ]);
  assert.deepEqual(proof.affineSymbolRelations, {
    M: { source: 'Q', offset: 210 },
  });

  const questionLength = 17;
  const plan = resolveGraphShapes(graph, {
    imageTokens: {
      shape: [1, 210, 320],
      data: new Float32Array(210 * 320),
    },
    questionTokens: {
      shape: [1, questionLength, 320],
      data: new Float32Array(questionLength * 320),
    },
  });
  assert.equal(plan.symbols.Q, questionLength);
  assert.equal(plan.symbols.M, questionLength + 210);
  assert.deepEqual(plan.outputs[0].shape, [1, questionLength + 210, 320]);

  const preboundDocument = affineConcatGraphDocument();
  preboundDocument.inputs.unrelatedMemoryExtent = {
    dtype: 'float32',
    shape: ['M'],
  };
  const preboundProof = proveGraphShapeDomain(
    parseGraphDocument(preboundDocument),
  );
  assert.equal(preboundProof.supported, false);
  assert.equal(preboundProof.code, 'SYMBOL_CONFLICT');
  assert.match(preboundProof.reason, /public input or non-affine output/);
});

test('graph-wide affine Concat permits identical M reuse and rejects a conflicting source', () => {
  const repeated = affineConcatGraphDocument();
  repeated.inputs.imageMask = { dtype: 'int32', shape: ['B', 210] };
  repeated.inputs.questionMask = { dtype: 'int32', shape: ['B', 'Q'] };
  repeated.nodes.push({
    id: 'mask_concat',
    opType: 'Concat',
    inputs: { input0: 'imageMask', input1: 'questionMask' },
    outputs: {
      out: { tensor: 'memoryMask', dtype: 'int32', shape: ['B', 'M'] },
    },
    params: { axis: 1 },
  });
  repeated.outputs.push('memoryMask');

  const repeatedProof = proveGraphShapeDomain(
    parseGraphDocument(repeated),
  );
  assert.equal(repeatedProof.supported, true);
  assert.deepEqual(repeatedProof.affineSymbolRelations, {
    M: { source: 'Q', offset: 210 },
  });
  assert.deepEqual(repeatedProof.nodes.map((node) => node.facts[0]), [
    'M=Q+210 exactly over the complete bounded Concat domain',
    'M=Q+210 exactly over the complete bounded Concat domain',
  ]);

  const conflicting = structuredClone(repeated);
  conflicting.dimensions.R = { min: 1, max: 192 };
  conflicting.inputs.questionMask.shape = ['B', 'R'];
  const conflictingProof = proveGraphShapeDomain(
    parseGraphDocument(conflicting),
  );
  assert.equal(conflictingProof.supported, false);
  assert.equal(conflictingProof.code, 'SYMBOL_CONFLICT');
  assert.match(conflictingProof.reason, /Q\+210, not R\+210/);
});

test('graph-wide variadic validation rejects gaps and declared/inferred count mismatches', () => {
  const concatGap = parseGraphDocument(concatGraphDocument({
    input0: 'left', input2: 'right',
  }));
  const gapProof = proveGraphShapeDomain(concatGap);
  assert.equal(gapProof.supported, false);
  assert.equal(gapProof.code, 'INPUT_PORT_MISMATCH');
  assert.match(gapProof.reason, /contiguous/);
  resolvedError(() => resolveGraphShapes(concatGap, {
    left: { shape: [2, 2], data: new Float32Array(4) },
    right: { shape: [2, 3], data: new Float32Array(6) },
  }), 'INPUT_PORT_MISMATCH', /contiguous/);

  const concatCount = parseGraphDocument(concatGraphDocument({ input0: 'left' }));
  const concatCountProof = proveGraphShapeDomain(concatCount);
  assert.equal(concatCountProof.supported, false);
  assert.equal(concatCountProof.code, 'INPUT_PORT_MISMATCH');
  assert.match(concatCountProof.reason, /at least 2/);

  const splitGap = parseGraphDocument(splitGraphDocument({
    out0: { tensor: 'first', dtype: 'float32', shape: ['B', 2] },
    out2: { tensor: 'second', dtype: 'float32', shape: ['B', 4] },
  }));
  const splitGapProof = proveGraphShapeDomain(splitGap);
  assert.equal(splitGapProof.supported, false);
  assert.equal(splitGapProof.code, 'OUTPUT_PORT_MISMATCH');
  assert.match(splitGapProof.reason, /contiguous/);

  const splitCount = parseGraphDocument(splitGraphDocument({
    out0: { tensor: 'first', dtype: 'float32', shape: ['B', 2] },
  }));
  const countProof = proveGraphShapeDomain(splitCount);
  assert.equal(countProof.supported, false);
  assert.equal(countProof.code, 'OUTPUT_PORT_MISMATCH');
  assert.match(countProof.reason, /requires exactly/);
});
