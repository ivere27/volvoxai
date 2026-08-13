/* Shape contracts for the direct operators: the output shape is the input
 * shape, or a single axis is replaced.  Identity, activations, Cast, the
 * quantize/dequantize boundary, feature and group normalization, dense
 * projections, Embedding and exact-shape binary arithmetic.
 *
 * Mirrors tests/operator_shape_contract_vectors.json.
 */
import type { RuntimeDType, TensorQuantization } from '../types.js';
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


export function inferIdentity(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const input = inputs.input;
  return cloneConcreteOutput(input.shape, input.dtype, input.quantization ?? undefined);
}
export function proveIdentity(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const input = inputs.input;
  return domainInference(
    cloneLogicalOutput(input.shape, input.dtype, request.environment, input.quantization ?? undefined),
    ['output shape, dtype, and quantization equal the input for every legal binding'],
  );
}
export function inferActivation(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const input = inputs.input;
  if (operator === 'Clip' && input.dtype === 'int32') {
    assertUnquantized(input, "operator input 'input'");
  } else {
    assertFloatTensor(input, "operator input 'input'");
  }
  validateActivationParams(operator, paramsRecord(request.params), input.shape.length, input.dtype);
  return cloneConcreteOutput(input.shape, input.dtype);
}
export function proveActivation(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const input = inputs.input;
  if (operator === 'Clip' && input.dtype === 'int32') {
    assertUnquantized(input, "operator input 'input'");
  } else {
    assertFloatTensor(input, "operator input 'input'");
  }
  validateActivationParams(operator, paramsRecord(request.params), input.shape.length, input.dtype);
  return domainInference(
    cloneLogicalOutput(input.shape, input.dtype, request.environment),
    ['activation preserves every input axis exactly'],
  );
}
export function inferPReLU(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'slope'], []);
  const input = inputs.input;
  const slope = inputs.slope;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(slope, "operator input 'slope'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(slope.shape, 0, 1, "operator input 'slope'.shape");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const slopeExtent = slope.shape[0];
  if (slopeExtent !== 1 && slopeExtent !== input.shape.at(-1)) {
    fail(
      'SHAPE_MISMATCH',
      "operator input 'slope'.shape",
      `must be [1] or match the last input extent ${input.shape.at(-1)}.`,
    );
  }
  return cloneConcreteOutput(input.shape, 'float32');
}
export function provePReLU(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'slope'],
    [],
  );
  const input = inputs.input;
  const slope = inputs.slope;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(slope, "operator input 'slope'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(slope.shape, 0, 1, "operator input 'slope'.shape");
  const fixedSlope = constantLogicalShape(slope.shape, "operator input 'slope'.shape");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const slopeExtent = fixedSlope[0];
  if (slopeExtent !== 1) {
    const feature = fixedDimensionValue(input.shape.at(-1)!, request.environment);
    if (feature === undefined) {
      fail(
        'UNPROVABLE_DYNAMIC_FEATURE',
        `operator input 'input'.shape[${input.shape.length - 1}]`,
        'per-feature PReLU requires a fixed last extent over the complete domain.',
      );
    }
    if (feature !== slopeExtent) {
      fail(
        'SHAPE_MISMATCH',
        "operator input 'slope'.shape",
        `has extent ${slopeExtent}, but the fixed input feature extent is ${feature}.`,
      );
    }
  }
  return domainInference(
    cloneLogicalOutput(input.shape, 'float32', request.environment),
    [slopeExtent === 1
      ? 'one fixed scalar slope applies to every legal input binding'
      : `fixed per-feature slope extent ${slopeExtent} matches the input feature axis`],
  );
}
export function inferCast(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const input = inputs.input;
  assertUnquantized(input, "operator input 'input'");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['to'], 'operator params');
  assertRuntimeDType(params.to, 'operator params.to');
  return cloneConcreteOutput(input.shape, params.to);
}
export function proveCast(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const input = inputs.input;
  assertUnquantized(input, "operator input 'input'");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['to'], 'operator params');
  assertRuntimeDType(params.to, 'operator params.to');
  return domainInference(
    cloneLogicalOutput(input.shape, params.to, request.environment),
    ['cast changes storage dtype but preserves every axis'],
  );
}
function validateQuantizationParameterInputs(
  input: { readonly shape: readonly number[] },
  scale: OperatorTensorDescriptor,
  zeroPoint: OperatorTensorDescriptor | undefined,
  dtype: 'int8' | 'uint8',
  quantization: TensorQuantization,
): void {
  assertFloatTensor(scale, "operator input 'scale'");
  if (zeroPoint !== undefined) {
    if (zeroPoint.dtype !== dtype) {
      fail('INVALID_DTYPE', "operator input 'zero_point'.dtype", `must be '${dtype}'.`);
    }
    assertUnquantized(zeroPoint, "operator input 'zero_point'");
  }
  if (quantization.scheme === 'per_tensor') {
    assertVectorShape(scale.shape, 1, "operator input 'scale'.shape");
    if (zeroPoint !== undefined) assertVectorShape(zeroPoint.shape, 1, "operator input 'zero_point'.shape");
    return;
  }
  const extent = input.shape[quantization.axis];
  assertVectorShape(scale.shape, extent, "operator input 'scale'.shape");
  if (zeroPoint !== undefined) {
    assertVectorShape(zeroPoint.shape, extent, "operator input 'zero_point'.shape");
  }
}
function validateLogicalQuantizationParameterInputs(
  input: { readonly shape: TensorShapeSpec },
  scale: LogicalOperatorTensorDescriptor,
  zeroPoint: LogicalOperatorTensorDescriptor | undefined,
  dtype: 'int8' | 'uint8',
  quantization: TensorQuantization,
  environment: ShapeEnvironment,
): void {
  assertFloatTensor(scale, "operator input 'scale'");
  constantLogicalShape(scale.shape, "operator input 'scale'.shape");
  if (zeroPoint !== undefined) {
    if (zeroPoint.dtype !== dtype) {
      fail('INVALID_DTYPE', "operator input 'zero_point'.dtype", `must be '${dtype}'.`);
    }
    assertUnquantized(zeroPoint, "operator input 'zero_point'");
    constantLogicalShape(zeroPoint.shape, "operator input 'zero_point'.shape");
  }
  if (quantization.scheme === 'per_tensor') {
    assertLogicalVectorShape(scale.shape, 1, environment, "operator input 'scale'.shape");
    if (zeroPoint !== undefined) {
      assertLogicalVectorShape(zeroPoint.shape, 1, environment, "operator input 'zero_point'.shape");
    }
    return;
  }
  const extent = fixedDimensionValue(input.shape[quantization.axis], environment);
  if (extent === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT',
      `operator input 'input'.shape[${quantization.axis}]`,
      'per-axis quantization requires one fixed extent over the complete domain.',
    );
  }
  assertLogicalVectorShape(scale.shape, extent, environment, "operator input 'scale'.shape");
  if (zeroPoint !== undefined) {
    assertLogicalVectorShape(zeroPoint.shape, extent, environment, "operator input 'zero_point'.shape");
  }
}
export function inferQuantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'scale'], ['zero_point']);
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const output = normalizeConcreteDeclaredOutput(request.declaredOutputs);
  if (output.dtype !== 'int8' && output.dtype !== 'uint8') {
    fail('INVALID_DTYPE', "declared output 'out'.dtype", "must be 'int8' or 'uint8'.");
  }
  if (output.quantization == null) {
    fail('INVALID_QUANTIZATION', "declared output 'out'.quantization", 'is required.');
  }
  if (!sameConcreteShape(input.shape, output.shape)) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must exactly equal the inferred input shape.');
  }
  validateQuantizationParameterInputs(
    input,
    inputs.scale,
    inputs.zero_point,
    output.dtype,
    output.quantization,
  );
  return cloneConcreteOutput(input.shape, output.dtype, output.quantization);
}
export function proveQuantizeLinear(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'scale'],
    ['zero_point'],
  );
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const output = normalizeLogicalDeclaredOutput(request.declaredOutputs, request.environment);
  if (output.dtype !== 'int8' && output.dtype !== 'uint8') {
    fail('INVALID_DTYPE', "declared output 'out'.dtype", "must be 'int8' or 'uint8'.");
  }
  if (output.quantization == null) {
    fail('INVALID_QUANTIZATION', "declared output 'out'.quantization", 'is required.');
  }
  if (!logicalShapesProvablyEqual(input.shape, output.shape, request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      "declared output 'out'.shape",
      'must equal the inferred input shape over the complete bounded domain.',
    );
  }
  validateLogicalQuantizationParameterInputs(
    input,
    inputs.scale,
    inputs.zero_point,
    output.dtype,
    output.quantization,
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(input.shape, output.dtype, request.environment, output.quantization),
    [output.quantization.scheme === 'per_tensor'
      ? 'per-tensor activation quantization is independent of dynamic extents'
      : `per-axis extent ${output.quantization.scales.length} is fixed over the complete domain`],
  );
}
export function inferDequantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'scale'], ['zero_point']);
  const input = inputs.input;
  if (input.dtype !== 'int8' && input.dtype !== 'uint8') {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int8' or 'uint8'.");
  }
  if (input.quantization == null) {
    fail('INVALID_QUANTIZATION', "operator input 'input'.quantization", 'is required.');
  }
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  validateQuantizationParameterInputs(
    input,
    inputs.scale,
    inputs.zero_point,
    input.dtype,
    input.quantization,
  );
  return cloneConcreteOutput(input.shape, 'float32');
}
export function proveDequantizeLinear(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'scale'],
    ['zero_point'],
  );
  const input = inputs.input;
  if (input.dtype !== 'int8' && input.dtype !== 'uint8') {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int8' or 'uint8'.");
  }
  if (input.quantization == null) {
    fail('INVALID_QUANTIZATION', "operator input 'input'.quantization", 'is required.');
  }
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  validateLogicalQuantizationParameterInputs(
    input,
    inputs.scale,
    inputs.zero_point,
    input.dtype,
    input.quantization,
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(input.shape, 'float32', request.environment),
    [input.quantization.scheme === 'per_tensor'
      ? 'per-tensor dequantization preserves all dynamic axes'
      : `per-axis extent ${input.quantization.scales.length} is fixed over the complete domain`],
  );
}
function normalizationParams(
  params: Readonly<Record<string, unknown>>,
  feature: number,
): void {
  assertAllowedFields(params, ['eps', 'd_model'], 'operator params');
  finitePositiveParam(params, 'eps');
  if (params.d_model !== undefined &&
      (!Number.isSafeInteger(params.d_model) || (params.d_model as number) <= 0)) {
    fail('INVALID_PARAMS', 'operator params.d_model', 'must be a positive safe integer.');
  }
  if (params.d_model !== undefined && params.d_model !== feature) {
    fail('SHAPE_MISMATCH', 'operator params.d_model', `must equal feature extent ${feature}.`);
  }
}
export function inferFeatureNorm(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const optional = operator === 'LayerNorm' ? ['bias'] : [];
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], optional);
  const input = inputs.input;
  const weight = inputs.weight;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  const feature = input.shape.at(-1)!;
  assertVectorShape(weight.shape, feature, "operator input 'weight'.shape");
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, feature, "operator input 'bias'.shape");
  }
  normalizationParams(paramsRecord(request.params), feature);
  return cloneConcreteOutput(input.shape, 'float32');
}
export function proveFeatureNorm(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const optional = operator === 'LayerNorm' ? ['bias'] : [];
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight'],
    optional,
  );
  const input = inputs.input;
  const weight = inputs.weight;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  constantLogicalShape(weight.shape, "operator input 'weight'.shape");
  const feature = fixedDimensionValue(input.shape.at(-1)!, request.environment);
  if (feature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE',
      `operator input 'input'.shape[${input.shape.length - 1}]`,
      `${operator} requires a fixed feature extent over the complete domain.`,
    );
  }
  assertLogicalVectorShape(weight.shape, feature, request.environment, "operator input 'weight'.shape");
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(inputs.bias.shape, feature, request.environment, "operator input 'bias'.shape");
  }
  normalizationParams(paramsRecord(request.params), feature);
  return domainInference(
    cloneLogicalOutput(input.shape, 'float32', request.environment),
    [`feature extent ${feature} is fixed and affine parameters match it`],
  );
}
function groupNormParams(params: Readonly<Record<string, unknown>>, channels: number): number {
  assertAllowedFields(params, ['num_groups', 'eps'], 'operator params');
  finitePositiveParam(params, 'eps');
  if (!Number.isSafeInteger(params.num_groups) || (params.num_groups as number) <= 0) {
    fail('INVALID_PARAMS', 'operator params.num_groups', 'must be a positive safe integer.');
  }
  const groups = params.num_groups as number;
  if (channels % groups !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.num_groups', `must divide ${channels} channels.`);
  }
  return groups;
}
export function inferGroupNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight', 'bias'], []);
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  assertRank(input.shape, 0, 4, "operator input 'input'.shape");
  constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
  const channels = input.shape[3];
  assertVectorShape(inputs.weight.shape, channels, "operator input 'weight'.shape");
  assertVectorShape(inputs.bias.shape, channels, "operator input 'bias'.shape");
  groupNormParams(paramsRecord(request.params), channels);
  return cloneConcreteOutput(input.shape, 'float32');
}
export function proveGroupNorm(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight', 'bias'],
    [],
  );
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  assertRank(input.shape, 0, 4, "operator input 'input'.shape");
  const channels = fixedDimensionValue(input.shape[3], request.environment);
  if (channels === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_CHANNEL',
      "operator input 'input'.shape[3]",
      'GroupNorm requires a fixed NHWC channel extent over the complete domain.',
    );
  }
  assertLogicalVectorShape(inputs.weight.shape, channels, request.environment, "operator input 'weight'.shape");
  assertLogicalVectorShape(inputs.bias.shape, channels, request.environment, "operator input 'bias'.shape");
  const groups = groupNormParams(paramsRecord(request.params), channels);
  return domainInference(
    cloneLogicalOutput(input.shape, 'float32', request.environment),
    [`channel extent ${channels} is fixed and divisible by ${groups} groups`],
  );
}
type DenseWeightLayout = 'din_dout' | 'dout_din';
function denseLayout(
  operator: string,
  params: Readonly<Record<string, unknown>>,
): DenseWeightLayout {
  assertAllowedFields(params, ['weight_layout', 'transB'], 'operator params');
  if (params.weight_layout !== undefined && params.transB !== undefined) {
    fail('INVALID_PARAMS', 'operator params', 'must not specify both weight_layout and transB.');
  }
  if (params.weight_layout !== undefined) {
    if (params.weight_layout !== 'din_dout' && params.weight_layout !== 'dout_din') {
      fail('INVALID_PARAMS', 'operator params.weight_layout', "must be 'din_dout' or 'dout_din'.");
    }
    return params.weight_layout;
  }
  if (params.transB !== undefined) {
    if (typeof params.transB !== 'boolean') {
      fail('INVALID_PARAMS', 'operator params.transB', 'must be boolean.');
    }
    return params.transB ? 'dout_din' : 'din_dout';
  }
  return operator === 'Linear' ? 'dout_din' : 'din_dout';
}
function denseWeightDimensions(
  shape: readonly number[],
  layout: DenseWeightLayout,
): readonly [number, number] {
  return layout === 'din_dout'
    ? [shape[0], shape[1]]
    : [shape[1], shape[0]];
}
export function inferDense(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  const input = inputs.input;
  const weight = inputs.weight;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(weight.shape, 0, 2, "operator input 'weight'.shape");
  const layout = denseLayout(operator, paramsRecord(request.params));
  const [contracted, outputFeature] = denseWeightDimensions(weight.shape, layout);
  if (input.shape.at(-1) !== contracted) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'input'.shape[${input.shape.length - 1}]`,
      `must equal fixed contracted weight extent ${contracted}.`,
    );
  }
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, outputFeature, "operator input 'bias'.shape");
  }
  return cloneConcreteOutput([...input.shape.slice(0, -1), outputFeature], 'float32');
}
export function proveDense(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight'],
    ['bias'],
  );
  const input = inputs.input;
  const weight = inputs.weight;
  assertFloatTensor(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(weight.shape, 0, 2, "operator input 'weight'.shape");
  const fixedWeightShape = constantLogicalShape(weight.shape, "operator input 'weight'.shape");
  const layout = denseLayout(operator, paramsRecord(request.params));
  const [contracted, outputFeature] = denseWeightDimensions(fixedWeightShape, layout);
  const inputContracted = fixedDimensionValue(input.shape.at(-1)!, request.environment);
  if (inputContracted === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_CONTRACTION',
      `operator input 'input'.shape[${input.shape.length - 1}]`,
      'the contracted activation extent must be fixed over the complete domain.',
    );
  }
  if (inputContracted !== contracted) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'input'.shape[${input.shape.length - 1}]`,
      `is fixed at ${inputContracted}, but the weight contracts ${contracted}.`,
    );
  }
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(inputs.bias.shape, outputFeature, request.environment, "operator input 'bias'.shape");
  }
  const outputShape = Object.freeze([...input.shape.slice(0, -1), outputFeature]);
  return domainInference(
    cloneLogicalOutput(outputShape, 'float32', request.environment),
    [`contracted extent ${contracted} and output feature extent ${outputFeature} are fixed`],
  );
}
export function inferEmbedding(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], []);
  const input = inputs.input;
  const weight = inputs.weight;
  if (input.dtype !== 'int32') fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int32'.");
  assertUnquantized(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(weight.shape, 0, 2, "operator input 'weight'.shape");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return cloneConcreteOutput([...input.shape, weight.shape[1]], 'float32');
}
export function proveEmbedding(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input', 'weight'], []);
  const input = inputs.input;
  const weight = inputs.weight;
  if (input.dtype !== 'int32') fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int32'.");
  assertUnquantized(input, "operator input 'input'");
  assertFloatTensor(weight, "operator input 'weight'");
  assertRank(input.shape, 1, null, "operator input 'input'.shape");
  assertRank(weight.shape, 0, 2, "operator input 'weight'.shape");
  const fixedWeightShape = constantLogicalShape(weight.shape, "operator input 'weight'.shape");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const outputShape = Object.freeze([...input.shape, fixedWeightShape[1]]);
  return domainInference(
    cloneLogicalOutput(outputShape, 'float32', request.environment),
    [`embedding preserves the dynamic index prefix and appends fixed width ${fixedWeightShape[1]}`],
  );
}
export function exactBinaryParams(
  operator: string,
  params: Readonly<Record<string, unknown>>,
): void {
  if (operator === 'Add') {
    assertAllowedFields(params, ['relu'], 'operator params');
    const relu = params.relu ?? 0;
    if (!Number.isInteger(relu) || (relu as number) < 0 || (relu as number) > 2) {
      fail('INVALID_PARAMS', 'operator params.relu', 'must be 0, 1, or 2.');
    }
    return;
  }
  assertAllowedFields(params, [], 'operator params');
}
