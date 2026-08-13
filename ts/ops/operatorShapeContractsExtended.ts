/* Shape contracts for the operators outside the core families: MoE routing
 * and expert linears, CrossAttention, BatchNorm2D, Interpolate1D, logic and
 * masking, Concat2, RequantizeLinear, selective scan, vision profiles and
 * Dropout.
 *
 * Mirrors shape_contract_extended.inc on the native side and
 * tests/operator_shape_contract_extended_vectors.json.
 */
import {
  SHAPE_SYMBOL_PATTERN,
  ShapeEnvironment,
  checkedShapeAdd,
  checkedShapeElementCount,
  checkedShapeFloorDivide,
  checkedShapeMultiply,
  checkedShapeSubtract,
  createTensorShapeSpec,
  type ShapeDimensionSpec,
  type TensorShapeSpec,
} from './shapeSystem.js';
import {
  ACTIVATION_OPERATORS,
  ATTENTION_SHAPE_FUNCTION_IDS,
  AttentionParameters,
  AttentionShapeFunctionId,
  ConcreteOperatorOutputs,
  ConcreteOperatorShapeRequest,
  DIRECT_SHAPE_FUNCTION_IDS,
  DirectShapeFunctionId,
  DomainInference,
  EXTENDED_SHAPE_FUNCTION_IDS,
  I32_ACCUMULATOR_MAX,
  LogicalOperatorOutputs,
  LogicalOperatorTensorDescriptor,
  OperatorAffineDimensionRelation,
  OperatorDomainProof,
  OperatorDomainProofRequest,
  OperatorShapeContract,
  OperatorShapeContractError,
  OperatorShapeFunctionId,
  OperatorShapePorts,
  OperatorTensorDescriptor,
  PerAxisQuantization,
  PerTensorQuantization,
  QUANTIZED_SHAPE_FUNCTION_IDS,
  QuantizedShapeFunctionId,
  SPATIAL_SHAPE_FUNCTION_IDS,
  STRUCTURAL_SHAPE_FUNCTION_IDS,
  SpatialShapeFunctionId,
  StructuralShapeFunctionId,
  activationParam,
  assertAllowedFields,
  assertAttentionMaskDType,
  assertAttentionRank,
  assertAxisZeroByteWeight,
  assertConcreteAttentionMask,
  assertFloatTensor,
  assertI32AccumulatorBound,
  assertI32CenteredSumBound,
  assertLogicalAttentionMask,
  assertLogicalVectorShape,
  assertPerTensorByte,
  assertPositiveF32Ratio,
  assertRank,
  assertRuntimeDType,
  assertSameStorageDType,
  assertUnquantized,
  assertVectorShape,
  attentionParameters,
  booleanParam,
  canonicalLayoutParam,
  centeredMagnitude,
  checkedDimensionsProduct,
  checkedTransposeOutput,
  checkedWindowOutput,
  cloneConcreteOutput,
  cloneConcreteOutputs,
  cloneLogicalOutput,
  cloneLogicalOutputs,
  collapseLogicalDimensions,
  concreteBroadcastShape,
  concreteQuantizedDeclaredOutput,
  concreteTargetShape,
  constantLogicalShape,
  dimensionsProvablyEqual,
  domainInference,
  fail,
  falseOrAbsentParam,
  finitePositiveParam,
  fixedDimensionValue,
  fullSpatialPads,
  integerParam,
  isRecord,
  legalDimensionProgression,
  logicalBroadcastDimension,
  logicalBroadcastShape,
  logicalProductsProvablyEqual,
  logicalShapesProvablyEqual,
  logicalTargetShape,
  logicalTransposeOutput,
  logicalWindowOutput,
  maximumDimension,
  maximumWeightMagnitude,
  normalizeAxes,
  normalizeAxis,
  normalizeConcreteDeclaredOutput,
  normalizeConcreteInputs,
  normalizeConcreteVariadicInputs,
  normalizeLogicalDeclaredOutput,
  normalizeLogicalInputs,
  normalizeLogicalVariadicInputs,
  ownNames,
  paramsRecord,
  positiveSafeIntegerParam,
  quantizationEqual,
  remapPerAxis,
  requireRankRange,
  reshapeQuantization,
  safeIntegerArray,
  sameConcreteShape,
  slicedPerAxis,
  spatialPairParam,
  spatialScalarParam,
  validateActivationParams,
} from './operatorShapeContractsCommon.js';

import {
  inferArgMax,
  inferBroadcastArithmetic,
  inferBroadcastComparison,
  inferConcat,
  inferExpand,
  inferFlatten,
  inferGather,
  inferGatherElements,
  inferPad,
  inferReduction,
  inferReshape,
  inferSlice,
  inferSplit,
  inferSqueeze,
  inferTranspose,
  inferUnsqueeze,
  inferWhere,
  proveArgMax,
  proveBroadcastArithmetic,
  proveBroadcastComparison,
  proveConcat,
  proveExpand,
  proveFlatten,
  proveGather,
  proveGatherElements,
  provePad,
  proveReduction,
  proveReshape,
  proveSlice,
  proveSplit,
  proveSqueeze,
  proveTranspose,
  proveUnsqueeze,
  proveWhere,
} from './operatorShapeContractsStructural.js';

export function inferMoERouter(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 1, 8, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  const feature = inputs.input.shape.at(-1)!;
  if (inputs.weight.shape[0] !== feature) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape[0]", `must equal input feature extent ${feature}.`);
  }
  const experts = inputs.weight.shape[1];
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['num_experts', 'top_k', 'temperature', 'normalize'], 'operator params');
  const numExperts = positiveSafeIntegerParam(params, 'num_experts', experts);
  const topK = positiveSafeIntegerParam(params, 'top_k', 2);
  if (numExperts !== experts) {
    fail('SHAPE_MISMATCH', 'operator params.num_experts', `must equal router weight extent ${experts}.`);
  }
  if (topK > experts) {
    fail('INVALID_PARAMS', 'operator params.top_k', `must be in [1, ${experts}].`);
  }
  finitePositiveParam(params, 'temperature');
  booleanParam(params, 'normalize', true);
  if (inputs.bias !== undefined &&
      (inputs.bias.shape.length !== 1 || inputs.bias.shape[0] !== experts)) {
    fail('SHAPE_MISMATCH', "operator input 'bias'.shape", `must equal [${experts}].`);
  }
  const routeShape = Object.freeze([...inputs.input.shape.slice(0, -1), topK]);
  return cloneConcreteOutputs({
    indices: { shape: routeShape, dtype: 'float32' },
    weights: { shape: routeShape, dtype: 'float32' },
  });
}
export function proveMoERouter(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['input', 'weight'], ['bias'],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 1, 8, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  if (!dimensionsProvablyEqual(
    inputs.input.shape.at(-1)!, inputs.weight.shape[0], request.environment,
  )) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'weight'.shape[0]",
      'must equal the input feature extent over the complete domain.',
    );
  }
  const experts = fixedDimensionValue(inputs.weight.shape[1], request.environment);
  if (experts === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'weight'.shape[1]",
      'the expert count must be fixed over the complete domain.',
    );
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['num_experts', 'top_k', 'temperature', 'normalize'], 'operator params');
  const numExperts = positiveSafeIntegerParam(params, 'num_experts', experts);
  const topK = positiveSafeIntegerParam(params, 'top_k', 2);
  if (numExperts !== experts) {
    fail('SHAPE_MISMATCH', 'operator params.num_experts', `must equal router weight extent ${experts}.`);
  }
  if (topK > experts) {
    fail('INVALID_PARAMS', 'operator params.top_k', `must be in [1, ${experts}].`);
  }
  finitePositiveParam(params, 'temperature');
  booleanParam(params, 'normalize', true);
  if (inputs.bias !== undefined && (inputs.bias.shape.length !== 1 ||
      !dimensionsProvablyEqual(inputs.bias.shape[0], experts, request.environment))) {
    fail('SHAPE_MISMATCH', "operator input 'bias'.shape", `must equal [${experts}].`);
  }
  const routeShape = Object.freeze([...inputs.input.shape.slice(0, -1), topK]);
  return domainInference(
    cloneLogicalOutputs({
      indices: { shape: routeShape, dtype: 'float32' },
      weights: { shape: routeShape, dtype: 'float32' },
    }, request.environment),
    [`the fixed top-k extent ${topK} is valid for ${experts} experts; token-prefix symbols are preserved`],
  );
}
export function inferMoELinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(
    request.inputs,
    ['input', 'expert_weight', 'route_indices', 'route_weights'],
    ['expert_bias'],
  );
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 1, 8, "operator input 'input'.shape");
  assertRank(inputs.expert_weight.shape, 0, 3, "operator input 'expert_weight'.shape");
  const [experts, inputFeature, outputFeature] = inputs.expert_weight.shape;
  if (inputFeature !== inputs.input.shape.at(-1)) {
    fail('SHAPE_MISMATCH', "operator input 'expert_weight'.shape[1]", 'must equal the input feature extent.');
  }
  if (!sameConcreteShape(inputs.route_indices.shape, inputs.route_weights.shape) ||
      inputs.route_indices.shape.length !== inputs.input.shape.length ||
      !inputs.route_indices.shape.slice(0, -1).every(
        (dimension, axis) => dimension === inputs.input.shape[axis],
      )) {
    fail('SHAPE_MISMATCH', 'operator route inputs', 'must share the input token prefix and one common top-k axis.');
  }
  const topK = inputs.route_indices.shape.at(-1)!;
  if (topK > experts) {
    fail('SHAPE_MISMATCH', "operator input 'route_indices'.shape", `top-k extent must not exceed ${experts}.`);
  }
  if (inputs.expert_bias !== undefined &&
      !sameConcreteShape(inputs.expert_bias.shape, [experts, outputFeature])) {
    fail(
      'SHAPE_MISMATCH', "operator input 'expert_bias'.shape",
      `must equal [${experts}, ${outputFeature}].`,
    );
  }
  return cloneConcreteOutput(
    [...inputs.input.shape.slice(0, -1), outputFeature], 'float32',
  );
}
export function proveMoELinear(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'expert_weight', 'route_indices', 'route_weights'],
    ['expert_bias'],
  );
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 1, 8, "operator input 'input'.shape");
  assertRank(inputs.expert_weight.shape, 0, 3, "operator input 'expert_weight'.shape");
  const [expertDimension, inputFeature, outputFeature] = inputs.expert_weight.shape;
  if (!dimensionsProvablyEqual(inputFeature, inputs.input.shape.at(-1)!, request.environment)) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'expert_weight'.shape[1]",
      'must equal the input feature extent over the complete domain.',
    );
  }
  if (!logicalShapesProvablyEqual(
    inputs.route_indices.shape, inputs.route_weights.shape, request.environment,
  ) || inputs.route_indices.shape.length !== inputs.input.shape.length ||
      !inputs.route_indices.shape.slice(0, -1).every((dimension, axis) =>
        dimensionsProvablyEqual(dimension, inputs.input.shape[axis], request.environment))) {
    fail('SHAPE_MISMATCH', 'operator route inputs', 'must provably share the input token prefix and top-k axis.');
  }
  const experts = fixedDimensionValue(expertDimension, request.environment);
  const outputWidth = fixedDimensionValue(outputFeature, request.environment);
  if (experts === undefined || outputWidth === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'expert_weight'.shape",
      'expert count and output width must be fixed over the complete domain.',
    );
  }
  const topKDimension = inputs.route_indices.shape.at(-1)!;
  const topKMaximum = typeof topKDimension === 'number'
    ? topKDimension
    : request.environment.get(topKDimension)!.max;
  if (topKMaximum > experts) {
    fail('SHAPE_MISMATCH', "operator input 'route_indices'.shape", `top-k maximum must not exceed ${experts}.`);
  }
  if (inputs.expert_bias !== undefined && (inputs.expert_bias.shape.length !== 2 ||
      !dimensionsProvablyEqual(inputs.expert_bias.shape[0], experts, request.environment) ||
      !dimensionsProvablyEqual(inputs.expert_bias.shape[1], outputWidth, request.environment))) {
    fail(
      'SHAPE_MISMATCH', "operator input 'expert_bias'.shape",
      `must equal [${experts}, ${outputWidth}].`,
    );
  }
  return domainInference(
    cloneLogicalOutput(
      Object.freeze([...inputs.input.shape.slice(0, -1), outputWidth]),
      'float32', request.environment,
    ),
    [`all routing shapes are valid through top-k maximum ${topKMaximum}; route values remain execution-time checked`],
  );
}
function crossAttentionParams(
  params: Readonly<Record<string, unknown>>,
  feature: number,
): number {
  assertAllowedFields(params, ['heads'], 'operator params');
  const heads = positiveSafeIntegerParam(params, 'heads', 8);
  if (feature % heads !== 0) {
    fail('INVALID_PARAMS', 'operator params.heads', `must divide feature extent ${feature}.`);
  }
  return heads;
}
export function inferCrossAttention(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(
    request.inputs, ['q', 'kv', 'weight'], ['scale', 'bias'],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.q.shape, 2, 3, "operator input 'q'.shape");
  if (inputs.kv.shape.length !== inputs.q.shape.length) {
    fail('INVALID_RANK', "operator input 'kv'.shape", 'must have the same rank as q.');
  }
  const rank = inputs.q.shape.length;
  const feature = inputs.q.shape[rank - 1];
  if (inputs.kv.shape[rank - 1] !== feature ||
      (rank === 3 && inputs.kv.shape[0] !== inputs.q.shape[0])) {
    fail('SHAPE_MISMATCH', "operator input 'kv'.shape", 'must share q batch and feature dimensions.');
  }
  const projection = checkedShapeMultiply(feature, 3, 'CrossAttention projection extent');
  if (!sameConcreteShape(inputs.weight.shape, [projection, feature])) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape", `must equal [${projection}, ${feature}].`);
  }
  for (const name of ['scale', 'bias'] as const) {
    const descriptor = inputs[name];
    if (descriptor !== undefined && !sameConcreteShape(descriptor.shape, [projection])) {
      fail('SHAPE_MISMATCH', `operator input '${name}'.shape`, `must equal [${projection}].`);
    }
  }
  crossAttentionParams(paramsRecord(request.params), feature);
  return cloneConcreteOutput(inputs.q.shape, 'float32');
}
export function proveCrossAttention(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['q', 'kv', 'weight'], ['scale', 'bias'],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.q.shape, 2, 3, "operator input 'q'.shape");
  if (inputs.kv.shape.length !== inputs.q.shape.length) {
    fail('INVALID_RANK', "operator input 'kv'.shape", 'must have the same rank as q.');
  }
  const rank = inputs.q.shape.length;
  const featureDimension = inputs.q.shape[rank - 1];
  if (!dimensionsProvablyEqual(inputs.kv.shape[rank - 1], featureDimension, request.environment) ||
      (rank === 3 && !dimensionsProvablyEqual(
        inputs.kv.shape[0], inputs.q.shape[0], request.environment,
      ))) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'kv'.shape",
      'must share q batch and feature dimensions over the complete domain.',
    );
  }
  const feature = fixedDimensionValue(featureDimension, request.environment);
  if (feature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'q'.shape",
      'CrossAttention requires a fixed projection/head feature extent.',
    );
  }
  const projection = checkedShapeMultiply(feature, 3, 'CrossAttention projection extent');
  if (inputs.weight.shape.length !== 2 ||
      !dimensionsProvablyEqual(inputs.weight.shape[0], projection, request.environment) ||
      !dimensionsProvablyEqual(inputs.weight.shape[1], feature, request.environment)) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape", `must equal [${projection}, ${feature}].`);
  }
  for (const name of ['scale', 'bias'] as const) {
    const descriptor = inputs[name];
    if (descriptor !== undefined && (descriptor.shape.length !== 1 ||
        !dimensionsProvablyEqual(descriptor.shape[0], projection, request.environment))) {
      fail('SHAPE_MISMATCH', `operator input '${name}'.shape`, `must equal [${projection}].`);
    }
  }
  const heads = crossAttentionParams(paramsRecord(request.params), feature);
  return domainInference(
    cloneLogicalOutput(inputs.q.shape, 'float32', request.environment),
    [`batch/query/key symbols may vary; fixed feature ${feature} is divisible by ${heads} heads`],
  );
}
export function inferBatchNorm2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(
    request.inputs, ['input', 'weight', 'bias', 'running_mean', 'running_var'], [],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const channels = inputs.input.shape[3];
  for (const name of ['weight', 'bias', 'running_mean', 'running_var'] as const) {
    if (!sameConcreteShape(inputs[name].shape, [channels])) {
      fail('SHAPE_MISMATCH', `operator input '${name}'.shape`, `must equal [${channels}].`);
    }
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['eps'], 'operator params');
  finitePositiveParam(params, 'eps');
  return cloneConcreteOutput(inputs.input.shape, 'float32');
}
export function proveBatchNorm2D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment,
    ['input', 'weight', 'bias', 'running_mean', 'running_var'], [],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const channel = inputs.input.shape[3];
  for (const name of ['weight', 'bias', 'running_mean', 'running_var'] as const) {
    if (inputs[name].shape.length !== 1 ||
        !dimensionsProvablyEqual(inputs[name].shape[0], channel, request.environment)) {
      fail(
        'UNPROVABLE_DYNAMIC_CHANNEL', `operator input '${name}'.shape`,
        'must equal the NHWC channel extent over the complete domain.',
      );
    }
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['eps'], 'operator params');
  finitePositiveParam(params, 'eps');
  return domainInference(
    cloneLogicalOutput(inputs.input.shape, 'float32', request.environment),
    ['NHWC batch and spatial axes may vary; every parameter vector equals the channel axis'],
  );
}
export function inferInterpolate1D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['size'], 'operator params');
  const size = positiveSafeIntegerParam(params, 'size');
  return cloneConcreteOutput([inputs.input.shape[0], inputs.input.shape[1], size], 'float32');
}
export function proveInterpolate1D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['size'], 'operator params');
  const size = positiveSafeIntegerParam(params, 'size');
  return domainInference(
    cloneLogicalOutput(
      Object.freeze([inputs.input.shape[0], inputs.input.shape[1], size]),
      'float32', request.environment,
    ),
    [`N/C symbols are preserved and the resampled length is the fixed extent ${size}`],
  );
}
export function inferLogicalNot(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertUnquantized(inputs.input, "operator input 'input'");
  if (inputs.input.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int32'.");
  }
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  return cloneConcreteOutput(inputs.input.shape, 'int32');
}
export function proveLogicalNot(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertUnquantized(inputs.input, "operator input 'input'");
  if (inputs.input.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int32'.");
  }
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  return domainInference(
    cloneLogicalOutput(inputs.input.shape, 'int32', request.environment),
    ['logical negation preserves every axis and is independent of tensor values'],
  );
}
export function inferMask(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['mask', 'a', 'b'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertUnquantized(descriptor, `operator input '${name}'`);
  }
  if (inputs.mask.dtype !== 'float32' && inputs.mask.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'mask'.dtype", 'must be float32 or int32.');
  }
  const dtype = assertSameStorageDType(inputs.a, inputs.b, 'operator data inputs');
  if (dtype !== 'float32' && dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'a'.dtype", 'must be float32 or int32.');
  }
  requireRankRange(inputs.a.shape, 0, 8, 'operator data inputs');
  if (!sameConcreteShape(inputs.mask.shape, inputs.a.shape) ||
      !sameConcreteShape(inputs.a.shape, inputs.b.shape)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'Mask requires three exactly equal shapes; it never broadcasts.');
  }
  return cloneConcreteOutput(inputs.a.shape, dtype);
}
export function proveMask(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['mask', 'a', 'b'], [],
  );
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertUnquantized(descriptor, `operator input '${name}'`);
  }
  if (inputs.mask.dtype !== 'float32' && inputs.mask.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'mask'.dtype", 'must be float32 or int32.');
  }
  const dtype = assertSameStorageDType(inputs.a, inputs.b, 'operator data inputs');
  if (dtype !== 'float32' && dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'a'.dtype", 'must be float32 or int32.');
  }
  requireRankRange(inputs.a.shape, 0, 8, 'operator data inputs');
  if (!logicalShapesProvablyEqual(inputs.mask.shape, inputs.a.shape, request.environment) ||
      !logicalShapesProvablyEqual(inputs.a.shape, inputs.b.shape, request.environment)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'Mask shapes must be equal over the complete domain.');
  }
  return domainInference(
    cloneLogicalOutput(inputs.a.shape, dtype, request.environment),
    ['all three exact shapes are equal over the complete domain; Mask never broadcasts'],
  );
}
function concat2Params(
  params: Readonly<Record<string, unknown>>,
): Readonly<{ axis: unknown; sigmoid: boolean }> {
  assertAllowedFields(params, ['axis', 'sigmoid'], 'operator params');
  return Object.freeze({ axis: params.axis, sigmoid: booleanParam(params, 'sigmoid', false) });
}
export function inferConcat2(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  const params = concat2Params(paramsRecord(request.params));
  if (params.sigmoid) {
    assertFloatTensor(inputs.a, "operator input 'a'");
    assertFloatTensor(inputs.b, "operator input 'b'");
  }
  return inferConcat({
    inputs: { input0: inputs.a, input1: inputs.b },
    params: { axis: params.axis },
  });
}
export function proveConcat2(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  const params = concat2Params(paramsRecord(request.params));
  if (params.sigmoid) {
    assertFloatTensor(inputs.a, "operator input 'a'");
    assertFloatTensor(inputs.b, "operator input 'b'");
  }
  const result = proveConcat({
    environment: request.environment,
    inputs: { input0: inputs.a, input1: inputs.b },
    params: { axis: params.axis },
  });
  return domainInference(
    result.outputs,
    [...result.facts, params.sigmoid
      ? 'the fused sigmoid is legal only in the unquantized F32 domain'
      : 'storage and per-axis affine metadata are propagated exactly'],
  );
}
export function inferRequantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const output = normalizeConcreteDeclaredOutput(request.declaredOutputs);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  if (!sameConcreteShape(inputs.input.shape, output.shape)) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must equal the input shape.');
  }
  const ratio = Math.fround(inputQuantization.scale / outputQuantization.scale);
  if (!Number.isFinite(ratio) || ratio <= 0) {
    fail('INVALID_QUANTIZATION', "declared output 'out'.quantization.scale", 'produces a non-representable positive scale ratio.');
  }
  return cloneConcreteOutput(inputs.input.shape, output.dtype, outputQuantization);
}
export function proveRequantizeLinear(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const output = normalizeLogicalDeclaredOutput(request.declaredOutputs, request.environment);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  if (!logicalShapesProvablyEqual(inputs.input.shape, output.shape, request.environment)) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must equal the input shape over the complete domain.');
  }
  const ratio = Math.fround(inputQuantization.scale / outputQuantization.scale);
  if (!Number.isFinite(ratio) || ratio <= 0) {
    fail('INVALID_QUANTIZATION', "declared output 'out'.quantization.scale", 'produces a non-representable positive scale ratio.');
  }
  return domainInference(
    cloneLogicalOutput(inputs.input.shape, output.dtype, request.environment, outputQuantization),
    [`the shape domain is preserved and the fixed affine scale ratio ${ratio} is representable`],
  );
}
function concreteScanBCMode(
  shape: readonly number[],
  inputShape: readonly number[],
  stateWidth: number,
): number {
  const rank = inputShape.length;
  const batch = rank === 3 ? inputShape[0] : 1;
  const sequence = inputShape[rank - 2];
  if (sameConcreteShape(shape, [stateWidth])) return 0;
  if (sameConcreteShape(shape, [sequence, stateWidth])) return 1;
  if (rank === 3 && sameConcreteShape(shape, [batch, sequence, stateWidth])) return 2;
  return -1;
}
function logicalScanBCMode(
  shape: TensorShapeSpec,
  inputShape: TensorShapeSpec,
  stateWidth: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number {
  const rank = inputShape.length;
  const batch = rank === 3 ? inputShape[0] : 1;
  const sequence = inputShape[rank - 2];
  if (logicalShapesProvablyEqual(shape, [stateWidth], environment)) return 0;
  if (logicalShapesProvablyEqual(shape, [sequence, stateWidth], environment)) return 1;
  if (rank === 3 && logicalShapesProvablyEqual(
    shape, [batch, sequence, stateWidth], environment,
  )) return 2;
  return -1;
}
export function inferScan(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(
    request.inputs, ['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 2, 3, "operator input 'input'.shape");
  if (!sameConcreteShape(inputs.delta.shape, inputs.input.shape)) {
    fail('SHAPE_MISMATCH', "operator input 'delta'.shape", 'must equal the input shape.');
  }
  const rank = inputs.input.shape.length;
  const batch = rank === 3 ? inputs.input.shape[0] : 1;
  const channels = inputs.input.shape[rank - 1];
  if (inputs.A.shape.length !== 2 || inputs.A.shape[0] !== channels) {
    fail('SHAPE_MISMATCH', "operator input 'A'.shape", `must be [${channels}, state_width].`);
  }
  const stateWidth = inputs.A.shape[1];
  if (concreteScanBCMode(inputs.B.shape, inputs.input.shape, stateWidth) < 0 ||
      concreteScanBCMode(inputs.C.shape, inputs.input.shape, stateWidth) < 0) {
    fail('SHAPE_MISMATCH', 'operator B/C inputs', 'must use [N], [S,N], or [B,S,N] selective-scan layout.');
  }
  if (inputs.D !== undefined && !sameConcreteShape(inputs.D.shape, [channels])) {
    fail('SHAPE_MISMATCH', "operator input 'D'.shape", `must equal [${channels}].`);
  }
  if (inputs.z !== undefined && !sameConcreteShape(inputs.z.shape, inputs.input.shape)) {
    fail('SHAPE_MISMATCH', "operator input 'z'.shape", 'must equal the input shape.');
  }
  const stateShape = Object.freeze([batch, channels, stateWidth]);
  if (inputs.initial_state !== undefined &&
      !sameConcreteShape(inputs.initial_state.shape, stateShape)) {
    fail('SHAPE_MISMATCH', "operator input 'initial_state'.shape", `must equal [${stateShape}].`);
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['delta_softplus'], 'operator params');
  booleanParam(params, 'delta_softplus', true);
  return cloneConcreteOutputs({
    out: { shape: inputs.input.shape, dtype: 'float32' },
    state: { shape: stateShape, dtype: 'float32' },
  });
}
export function proveScan(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment,
    ['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'],
  );
  for (const [name, descriptor] of Object.entries(inputs)) {
    assertFloatTensor(descriptor, `operator input '${name}'`);
  }
  requireRankRange(inputs.input.shape, 2, 3, "operator input 'input'.shape");
  if (!logicalShapesProvablyEqual(inputs.delta.shape, inputs.input.shape, request.environment)) {
    fail('SHAPE_MISMATCH', "operator input 'delta'.shape", 'must equal the input shape over the complete domain.');
  }
  const rank = inputs.input.shape.length;
  const batch: ShapeDimensionSpec = rank === 3 ? inputs.input.shape[0] : 1;
  const channels = inputs.input.shape[rank - 1];
  if (inputs.A.shape.length !== 2 ||
      !dimensionsProvablyEqual(inputs.A.shape[0], channels, request.environment)) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE', "operator input 'A'.shape",
      'must be [channels, state_width] over the complete domain.',
    );
  }
  const stateWidth = inputs.A.shape[1];
  const bMode = logicalScanBCMode(inputs.B.shape, inputs.input.shape, stateWidth, request.environment);
  const cMode = logicalScanBCMode(inputs.C.shape, inputs.input.shape, stateWidth, request.environment);
  if (bMode < 0 || cMode < 0) {
    fail('SHAPE_MISMATCH', 'operator B/C inputs', 'must provably use [N], [S,N], or [B,S,N] layout.');
  }
  if (inputs.D !== undefined && (inputs.D.shape.length !== 1 ||
      !dimensionsProvablyEqual(inputs.D.shape[0], channels, request.environment))) {
    fail('SHAPE_MISMATCH', "operator input 'D'.shape", 'must equal [channels].');
  }
  if (inputs.z !== undefined &&
      !logicalShapesProvablyEqual(inputs.z.shape, inputs.input.shape, request.environment)) {
    fail('SHAPE_MISMATCH', "operator input 'z'.shape", 'must equal the input shape.');
  }
  const stateShape = Object.freeze([batch, channels, stateWidth]);
  if (inputs.initial_state !== undefined &&
      !logicalShapesProvablyEqual(inputs.initial_state.shape, stateShape, request.environment)) {
    fail('SHAPE_MISMATCH', "operator input 'initial_state'.shape", 'must equal [batch, channels, state_width].');
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['delta_softplus'], 'operator params');
  booleanParam(params, 'delta_softplus', true);
  return domainInference(
    cloneLogicalOutputs({
      out: { shape: inputs.input.shape, dtype: 'float32' },
      state: { shape: stateShape, dtype: 'float32' },
    }, request.environment),
    [`B/S may vary; channel/state equalities and B/C broadcast modes ${bMode}/${cMode} are proved`],
  );
}
type VisionProfileKind = 'mean' | 'softargmax' | 'profile-x' | 'profile-y';
function concreteVisionProfileShape(
  kind: VisionProfileKind,
  inputShape: readonly number[],
): readonly number[] {
  const [batch, height, width, channels] = inputShape;
  if (kind === 'profile-x') {
    return Object.freeze([batch, checkedShapeMultiply(channels, 2, 'ProfileX channel extent'), width]);
  }
  if (kind === 'profile-y') {
    return Object.freeze([batch, checkedShapeMultiply(channels, 2, 'ProfileY channel extent'), height]);
  }
  return Object.freeze([batch, channels, width]);
}
function logicalVisionProfileShape(
  kind: VisionProfileKind,
  inputShape: TensorShapeSpec,
  environment: ShapeEnvironment,
): TensorShapeSpec {
  const [batch, height, width, channels] = inputShape;
  if (kind === 'profile-x' || kind === 'profile-y') {
    const fixedChannels = fixedDimensionValue(channels, environment);
    if (fixedChannels === undefined) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA', "operator input 'input'.shape[3]",
        'the doubled profile channel extent requires a fixed channel dimension in v1.',
      );
    }
    const doubled = checkedShapeMultiply(fixedChannels, 2, 'profile channel extent');
    return Object.freeze([batch, doubled, kind === 'profile-x' ? width : height]);
  }
  return Object.freeze([batch, channels, width]);
}
export function inferVisionProfile(
  kind: VisionProfileKind,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  return cloneConcreteOutput(concreteVisionProfileShape(kind, inputs.input.shape), 'float32');
}
export function proveVisionProfile(
  kind: VisionProfileKind,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  return domainInference(
    cloneLogicalOutput(
      logicalVisionProfileShape(kind, inputs.input.shape, request.environment),
      'float32', request.environment,
    ),
    [kind === 'profile-x' || kind === 'profile-y'
      ? 'batch and retained spatial axes may vary; the fixed channel extent is doubled'
      : 'batch/channel/width symbols are preserved and height is reduced independent of values'],
  );
}
function dropoutParams(params: Readonly<Record<string, unknown>>): void {
  assertAllowedFields(params, ['ratio', 'p', 'probability', 'seed'], 'operator params');
  const probabilityNames = ['ratio', 'p', 'probability'].filter((name) => params[name] !== undefined);
  if (probabilityNames.length > 1) {
    fail('INVALID_PARAMS', 'operator params', 'must specify at most one Dropout probability field.');
  }
  const probability = probabilityNames.length === 0 ? 0.5 : params[probabilityNames[0]];
  if (typeof probability !== 'number' || !Number.isFinite(probability) ||
      probability < 0 || probability >= 1) {
    fail('INVALID_PARAMS', 'operator params.ratio', 'must be finite and in [0, 1).');
  }
  const seed = params.seed ?? 0;
  if (!Number.isSafeInteger(seed) || (seed as number) < 0 || (seed as number) > 0xffffffff) {
    fail('INVALID_PARAMS', 'operator params.seed', 'must be an unsigned 32-bit integer.');
  }
}
export function inferDropout(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  dropoutParams(paramsRecord(request.params));
  return cloneConcreteOutput(inputs.input.shape, 'float32');
}
export function proveDropout(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  dropoutParams(paramsRecord(request.params));
  return domainInference(
    cloneLogicalOutput(inputs.input.shape, 'float32', request.environment),
    ['shape is value-independent; training randomness is keyed by seed, counter, and concrete linear index'],
  );
}
