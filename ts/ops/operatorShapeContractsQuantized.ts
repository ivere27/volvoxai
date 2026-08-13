/* Shape contracts for the W8A8 quantized operators.
 *
 * Same family boundary the native side uses in shape_contract_quantized.inc
 * and the shared corpus in tests/operator_shape_contract_quantized_vectors.json,
 * so the three implementations stay comparable file by file.  Each operator
 * keeps its infer/prove pair together.
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
  booleanParam,
  canonicalLayoutParam,
  centeredMagnitude,
  checkedDimensionsProduct,
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
  quantizationEqual,
  remapPerAxis,
  reshapeQuantization,
  safeIntegerArray,
  sameConcreteShape,
  slicedPerAxis,
  spatialPairParam,
  validateActivationParams,
} from './operatorShapeContractsCommon.js';


function logicalQuantizedDeclaredOutput(
  request: OperatorDomainProofRequest,
  expectedShape: TensorShapeSpec,
): LogicalOperatorTensorDescriptor {
  const output = normalizeLogicalDeclaredOutput(request.declaredOutputs, request.environment);
  const quantization = assertPerTensorByte(output, "declared output 'out'");
  if (!logicalShapesProvablyEqual(output.shape, expectedShape, request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      "declared output 'out'.shape",
      'must equal the inferred output shape over the complete bounded domain.',
    );
  }
  return cloneLogicalOutput(expectedShape, output.dtype, request.environment, quantization).out;
}
function assertI32Tensor(
  descriptor: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  path: string,
): void {
  if (descriptor.dtype !== 'int32') fail('INVALID_DTYPE', `${path}.dtype`, "must be 'int32'.");
  assertUnquantized(descriptor, path);
}
function validateQDenseMultipliers(
  inputQuantization: PerTensorQuantization,
  weightQuantization: PerAxisQuantization,
  outputQuantization: PerTensorQuantization,
  path: string,
): void {
  weightQuantization.scales.forEach((scale, index) => {
    assertPositiveF32Ratio(
      inputQuantization.scale,
      scale,
      outputQuantization.scale,
      `${path}.scales[${index}]`,
    );
  });
}
export function inferQDense(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight', 'bias'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  assertI32Tensor(inputs.bias, "operator input 'bias'");
  const [outputFeature, contracted] = inputs.weight.shape;
  if (inputs.input.shape.at(-1) !== contracted) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
      `must equal fixed contracted weight extent ${contracted}.`,
    );
  }
  assertVectorShape(inputs.bias.shape, outputFeature, "operator input 'bias'.shape");
  const expected = Object.freeze([...inputs.input.shape.slice(0, -1), outputFeature]);
  const output = concreteQuantizedDeclaredOutput(request, expected);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  validateQDenseMultipliers(
    inputQuantization,
    weightQuantization,
    outputQuantization,
    "operator input 'weight'.quantization",
  );
  assertI32AccumulatorBound(
    contracted,
    centeredMagnitude(inputs.input.dtype, inputQuantization.zero_point),
    maximumWeightMagnitude(inputs.weight.dtype, weightQuantization),
    `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
  );
  return Object.freeze({ out: output });
}
export function proveQDense(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight', 'bias'],
    [],
  );
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  assertI32Tensor(inputs.bias, "operator input 'bias'");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
  const [outputFeature, contracted] = weight;
  const inputContracted = fixedDimensionValue(inputs.input.shape.at(-1)!, request.environment);
  if (inputContracted === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_CONTRACTION',
      `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
      `${operator} requires a fixed contracted activation extent.`,
    );
  }
  if (inputContracted !== contracted) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
      `is fixed at ${inputContracted}, but the weight contracts ${contracted}.`,
    );
  }
  assertLogicalVectorShape(
    inputs.bias.shape,
    outputFeature,
    request.environment,
    "operator input 'bias'.shape",
  );
  const expected = Object.freeze([...inputs.input.shape.slice(0, -1), outputFeature]);
  const output = logicalQuantizedDeclaredOutput(request, expected);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  validateQDenseMultipliers(
    inputQuantization,
    weightQuantization,
    outputQuantization,
    "operator input 'weight'.quantization",
  );
  assertI32AccumulatorBound(
    contracted,
    centeredMagnitude(inputs.input.dtype, inputQuantization.zero_point),
    maximumWeightMagnitude(inputs.weight.dtype, weightQuantization),
    `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
  );
  return domainInference(
    Object.freeze({ out: output }),
    [`fixed ${contracted}-term contraction and axis-0 output-channel metadata are proved for ${operator}`],
  );
}
export function inferQBatchMatMul(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.a.shape, 2, null, "operator input 'a'.shape");
  assertRank(inputs.b.shape, 2, null, "operator input 'b'.shape");
  if (inputs.a.shape.length > 8 || inputs.b.shape.length > 8) {
    fail('INVALID_RANK', 'operator inputs', 'QBatchMatMul operand ranks must not exceed 8.');
  }
  const aQuantization = assertPerTensorByte(inputs.a, "operator input 'a'");
  const bQuantization = assertPerTensorByte(inputs.b, "operator input 'b'");
  const contracted = inputs.a.shape.at(-1)!;
  if (contracted !== inputs.b.shape.at(-2)) {
    fail('SHAPE_MISMATCH', "operator input 'b'.shape", 'contracted matrix dimensions must match.');
  }
  const batch = concreteBroadcastShape(inputs.a.shape.slice(0, -2), inputs.b.shape.slice(0, -2));
  const expected = Object.freeze([...batch, inputs.a.shape.at(-2)!, inputs.b.shape.at(-1)!]);
  const output = concreteQuantizedDeclaredOutput(request, expected);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(
    aQuantization.scale,
    bQuantization.scale,
    outputQuantization.scale,
    "declared output 'out'.quantization.scale",
  );
  assertI32AccumulatorBound(
    contracted,
    centeredMagnitude(inputs.a.dtype, aQuantization.zero_point),
    centeredMagnitude(inputs.b.dtype, bQuantization.zero_point),
    `operator input 'a'.shape[${inputs.a.shape.length - 1}]`,
  );
  return Object.freeze({ out: output });
}
export function proveQBatchMatMul(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.a.shape, 2, null, "operator input 'a'.shape");
  assertRank(inputs.b.shape, 2, null, "operator input 'b'.shape");
  if (inputs.a.shape.length > 8 || inputs.b.shape.length > 8) {
    fail('INVALID_RANK', 'operator inputs', 'QBatchMatMul operand ranks must not exceed 8.');
  }
  const aQuantization = assertPerTensorByte(inputs.a, "operator input 'a'");
  const bQuantization = assertPerTensorByte(inputs.b, "operator input 'b'");
  const leftK = inputs.a.shape.at(-1)!;
  const rightK = inputs.b.shape.at(-2)!;
  if (!dimensionsProvablyEqual(leftK, rightK, request.environment)) {
    fail(
      'UNPROVABLE_DYNAMIC_CONTRACTION',
      "operator input 'b'.shape",
      'contracted matrix dimensions are not equal over the complete domain.',
    );
  }
  const batch = logicalBroadcastShape(
    inputs.a.shape.slice(0, -2),
    inputs.b.shape.slice(0, -2),
    request.environment,
  );
  const expected = Object.freeze([...batch, inputs.a.shape.at(-2)!, inputs.b.shape.at(-1)!]);
  const output = logicalQuantizedDeclaredOutput(request, expected);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(
    aQuantization.scale,
    bQuantization.scale,
    outputQuantization.scale,
    "declared output 'out'.quantization.scale",
  );
  assertI32AccumulatorBound(
    maximumDimension(leftK, request.environment),
    centeredMagnitude(inputs.a.dtype, aQuantization.zero_point),
    centeredMagnitude(inputs.b.dtype, bQuantization.zero_point),
    `operator input 'a'.shape[${inputs.a.shape.length - 1}]`,
  );
  return domainInference(
    Object.freeze({ out: output }),
    ['matrix contraction, leading-axis broadcast, quantized output, and maximum I32 bound are proved'],
  );
}
function qConv2DParams(params: Readonly<Record<string, unknown>>) {
  assertAllowedFields(
    params,
    ['stride', 'padding', 'pads', 'dilation', 'groups', 'relu', 'data_layout', 'weight_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  canonicalLayoutParam(params, 'weight_layout', 'OHWI');
  return Object.freeze({
    stride: spatialPairParam(params.stride, 1, 'operator params.stride', false),
    pads: fullSpatialPads(params),
    dilation: spatialPairParam(params.dilation, 1, 'operator params.dilation', false),
    groups: integerParam(params, 'groups', 1),
    relu: activationParam(params),
  });
}
function assertQConvChannels(
  inputChannels: number,
  weight: readonly number[],
  groups: number,
): number {
  const [outputChannels, , , inputPerGroup] = weight;
  if (inputChannels !== inputPerGroup * groups || outputChannels % groups !== 0) {
    fail(
      'SHAPE_MISMATCH',
      "operator input 'weight'.shape",
      'OHWI channels must match input channels and groups.',
    );
  }
  return outputChannels;
}
function validateQConvAccumulator(
  input: { readonly dtype: RuntimeDType },
  inputQuantization: PerTensorQuantization,
  weight: { readonly dtype: RuntimeDType; readonly shape: readonly number[] },
  weightQuantization: PerAxisQuantization,
): void {
  const terms = checkedShapeMultiply(
    checkedShapeMultiply(weight.shape[1], weight.shape[2], 'QConv2D accumulator terms'),
    weight.shape[3],
    'QConv2D accumulator terms',
  );
  assertI32AccumulatorBound(
    terms,
    centeredMagnitude(input.dtype, inputQuantization.zero_point),
    maximumWeightMagnitude(weight.dtype, weightQuantization),
    "operator input 'weight'.shape",
  );
}
export function inferQConv2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  const params = qConv2DParams(paramsRecord(request.params));
  const [batch, height, width, inputChannels] = inputs.input.shape;
  const outputChannels = assertQConvChannels(inputChannels, inputs.weight.shape, params.groups);
  if (inputs.bias !== undefined) {
    assertI32Tensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, outputChannels, "operator input 'bias'.shape");
  }
  const outputHeight = checkedWindowOutput(
    height,
    inputs.weight.shape[1],
    params.stride[0],
    params.pads[0],
    params.pads[2],
    params.dilation[0],
    "operator input 'input'.shape[1]",
  );
  const outputWidth = checkedWindowOutput(
    width,
    inputs.weight.shape[2],
    params.stride[1],
    params.pads[1],
    params.pads[3],
    params.dilation[1],
    "operator input 'input'.shape[2]",
  );
  const output = concreteQuantizedDeclaredOutput(
    request,
    [batch, outputHeight, outputWidth, outputChannels],
  );
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  validateQDenseMultipliers(
    inputQuantization,
    weightQuantization,
    outputQuantization,
    "operator input 'weight'.quantization",
  );
  validateQConvAccumulator(inputs.input, inputQuantization, inputs.weight, weightQuantization);
  return Object.freeze({ out: output });
}
export function proveQConv2D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight'],
    ['bias'],
  );
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  const params = qConv2DParams(paramsRecord(request.params));
  const inputChannels = fixedDimensionValue(inputs.input.shape[3], request.environment);
  if (inputChannels === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_CHANNEL',
      "operator input 'input'.shape[3]",
      'QConv2D requires a fixed NHWC channel extent.',
    );
  }
  const outputChannels = assertQConvChannels(inputChannels, weight, params.groups);
  if (inputs.bias !== undefined) {
    assertI32Tensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(
      inputs.bias.shape,
      outputChannels,
      request.environment,
      "operator input 'bias'.shape",
    );
  }
  const outputHeight = logicalWindowOutput(
    inputs.input.shape[1],
    request.environment,
    weight[1],
    params.stride[0],
    params.pads[0],
    params.pads[2],
    params.dilation[0],
    "operator input 'input'.shape[1]",
  );
  const outputWidth = logicalWindowOutput(
    inputs.input.shape[2],
    request.environment,
    weight[2],
    params.stride[1],
    params.pads[1],
    params.pads[3],
    params.dilation[1],
    "operator input 'input'.shape[2]",
  );
  const output = logicalQuantizedDeclaredOutput(
    request,
    [inputs.input.shape[0], outputHeight, outputWidth, outputChannels],
  );
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  validateQDenseMultipliers(
    inputQuantization,
    weightQuantization,
    outputQuantization,
    "operator input 'weight'.quantization",
  );
  validateQConvAccumulator(
    inputs.input,
    inputQuantization,
    { dtype: inputs.weight.dtype, shape: weight },
    weightQuantization,
  );
  return domainInference(
    Object.freeze({ out: output }),
    ['fixed OHWI channel/kernel geometry, complete NHWC spatial formulas, and I32 dot bound are proved'],
  );
}
export function inferQAdd(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  const aQuantization = assertPerTensorByte(inputs.a, "operator input 'a'");
  const bQuantization = assertPerTensorByte(inputs.b, "operator input 'b'");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['relu'], 'operator params');
  activationParam(params);
  if (!sameConcreteShape(inputs.a.shape, inputs.b.shape)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'QAdd requires exactly equal shapes; Expand is explicit.');
  }
  const output = concreteQuantizedDeclaredOutput(request, inputs.a.shape);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(aQuantization.scale, 1, outputQuantization.scale,
    "operator input 'a'.quantization.scale");
  assertPositiveF32Ratio(bQuantization.scale, 1, outputQuantization.scale,
    "operator input 'b'.quantization.scale");
  return Object.freeze({ out: output });
}
export function proveQAdd(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  const aQuantization = assertPerTensorByte(inputs.a, "operator input 'a'");
  const bQuantization = assertPerTensorByte(inputs.b, "operator input 'b'");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['relu'], 'operator params');
  activationParam(params);
  if (!logicalShapesProvablyEqual(inputs.a.shape, inputs.b.shape, request.environment)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'QAdd inputs are not exact-shape equal over the domain.');
  }
  const output = logicalQuantizedDeclaredOutput(request, inputs.a.shape);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(aQuantization.scale, 1, outputQuantization.scale,
    "operator input 'a'.quantization.scale");
  assertPositiveF32Ratio(bQuantization.scale, 1, outputQuantization.scale,
    "operator input 'b'.quantization.scale");
  return domainInference(
    Object.freeze({ out: output }),
    ['both byte inputs and the output have one exact shape over every legal binding'],
  );
}
function validateQActivationParams(
  operator: string,
  params: Readonly<Record<string, unknown>>,
): void {
  if (operator === 'QGELU') {
    assertAllowedFields(params, ['approximate'], 'operator params');
    if (params.approximate !== undefined && params.approximate !== 'none') {
      fail('INVALID_PARAMS', 'operator params.approximate', "must be 'none'.");
    }
    return;
  }
  assertAllowedFields(params, [], 'operator params');
}
export function inferQActivation(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  validateQActivationParams(operator, paramsRecord(request.params));
  const output = concreteQuantizedDeclaredOutput(request, inputs.input.shape);
  return Object.freeze({ out: output });
}
export function proveQActivation(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  validateQActivationParams(operator, paramsRecord(request.params));
  const output = logicalQuantizedDeclaredOutput(request, inputs.input.shape);
  return domainInference(
    Object.freeze({ out: output }),
    [`${operator} preserves every logical extent and uses per-tensor activation domains`],
  );
}
export function inferQEmbedding(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertI32Tensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  const expected = Object.freeze([...inputs.input.shape, inputs.weight.shape[1]]);
  const output = concreteQuantizedDeclaredOutput(request, expected);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  weightQuantization.scales.forEach((scale, index) =>
    assertPositiveF32Ratio(scale, 1, outputQuantization.scale,
      `operator input 'weight'.quantization.scales[${index}]`));
  return Object.freeze({ out: output });
}
export function proveQEmbedding(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight'],
    [],
  );
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertI32Tensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 2, "operator input 'weight'.shape");
  const weightQuantization = assertAxisZeroByteWeight(inputs.weight, "operator input 'weight'");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  const output = logicalQuantizedDeclaredOutput(request, [...inputs.input.shape, weight[1]]);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  weightQuantization.scales.forEach((scale, index) =>
    assertPositiveF32Ratio(scale, 1, outputQuantization.scale,
      `operator input 'weight'.quantization.scales[${index}]`));
  return domainInference(
    Object.freeze({ out: output }),
    ['the fixed vocabulary/hidden table appends its hidden width to every dynamic token prefix'],
  );
}
function positiveF32Param(
  params: Readonly<Record<string, unknown>>,
  name: string,
  required = false,
): void {
  const value = params[name];
  if (value === undefined && !required) return;
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0 ||
      !Number.isFinite(Math.fround(value)) || Math.fround(value) <= 0) {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be positive and representable as float32.');
  }
}
function qLayerNormParams(
  params: Readonly<Record<string, unknown>>,
  feature: number,
): void {
  assertAllowedFields(params, ['eps', 'd_model'], 'operator params');
  positiveF32Param(params, 'eps');
  if (params.d_model !== undefined && params.d_model !== feature) {
    fail('SHAPE_MISMATCH', 'operator params.d_model', `must equal feature extent ${feature}.`);
  }
  if (params.d_model !== undefined && !Number.isSafeInteger(params.d_model)) {
    fail('INVALID_PARAMS', 'operator params.d_model', 'must be a positive safe integer.');
  }
}
export function inferQLayerNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight', 'bias'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertPerTensorByte(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  const feature = inputs.input.shape.at(-1)!;
  assertVectorShape(inputs.weight.shape, feature, "operator input 'weight'.shape");
  assertVectorShape(inputs.bias.shape, feature, "operator input 'bias'.shape");
  qLayerNormParams(paramsRecord(request.params), feature);
  const output = concreteQuantizedDeclaredOutput(request, inputs.input.shape);
  return Object.freeze({ out: output });
}
export function proveQLayerNorm(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight', 'bias'],
    [],
  );
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertPerTensorByte(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  const feature = fixedDimensionValue(inputs.input.shape.at(-1)!, request.environment);
  if (feature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE',
      `operator input 'input'.shape[${inputs.input.shape.length - 1}]`,
      'QLayerNorm requires a fixed final feature extent.',
    );
  }
  constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
  assertLogicalVectorShape(inputs.weight.shape, feature, request.environment,
    "operator input 'weight'.shape");
  assertLogicalVectorShape(inputs.bias.shape, feature, request.environment,
    "operator input 'bias'.shape");
  qLayerNormParams(paramsRecord(request.params), feature);
  const output = logicalQuantizedDeclaredOutput(request, inputs.input.shape);
  return domainInference(
    Object.freeze({ out: output }),
    [`fixed feature extent ${feature} and both F32 affine vectors are proved`],
  );
}
function qGroupNormParams(
  params: Readonly<Record<string, unknown>>,
  channels: number,
): number {
  assertAllowedFields(params, ['num_groups', 'eps', 'data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  positiveF32Param(params, 'eps');
  if (!Number.isSafeInteger(params.num_groups) || (params.num_groups as number) <= 0) {
    fail('INVALID_PARAMS', 'operator params.num_groups', 'must be a positive safe integer.');
  }
  const groups = params.num_groups as number;
  if (channels % groups !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.num_groups', `must divide ${channels} channels.`);
  }
  return groups;
}
export function inferQGroupNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight', 'bias'], []);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertPerTensorByte(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  const channels = inputs.input.shape[3];
  assertVectorShape(inputs.weight.shape, channels, "operator input 'weight'.shape");
  assertVectorShape(inputs.bias.shape, channels, "operator input 'bias'.shape");
  qGroupNormParams(paramsRecord(request.params), channels);
  const output = concreteQuantizedDeclaredOutput(request, inputs.input.shape);
  return Object.freeze({ out: output });
}
export function proveQGroupNorm(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'weight', 'bias'],
    [],
  );
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertPerTensorByte(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertFloatTensor(inputs.bias, "operator input 'bias'");
  const channels = fixedDimensionValue(inputs.input.shape[3], request.environment);
  if (channels === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_CHANNEL',
      "operator input 'input'.shape[3]",
      'QGroupNorm requires a fixed NHWC channel extent.',
    );
  }
  constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
  assertLogicalVectorShape(inputs.weight.shape, channels, request.environment,
    "operator input 'weight'.shape");
  assertLogicalVectorShape(inputs.bias.shape, channels, request.environment,
    "operator input 'bias'.shape");
  const groups = qGroupNormParams(paramsRecord(request.params), channels);
  const output = logicalQuantizedDeclaredOutput(request, inputs.input.shape);
  return domainInference(
    Object.freeze({ out: output }),
    [`fixed NHWC channel extent ${channels} is divisible by ${groups} groups`],
  );
}
export function inferQMaskedMean(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'mask'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  assertRank(inputs.mask.shape, 0, 2, "operator input 'mask'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  assertI32Tensor(inputs.mask, "operator input 'mask'");
  const [batch, sequence, feature] = inputs.input.shape;
  if (inputs.mask.shape[0] !== batch || inputs.mask.shape[1] !== sequence) {
    fail('SHAPE_MISMATCH', "operator input 'mask'.shape", 'must be [B,S] for input [B,S,D].');
  }
  const output = concreteQuantizedDeclaredOutput(request, [batch, feature]);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(inputQuantization.scale, 1, outputQuantization.scale,
    "declared output 'out'.quantization.scale");
  assertI32CenteredSumBound(
    sequence,
    centeredMagnitude(inputs.input.dtype, inputQuantization.zero_point),
    "operator input 'input'.shape[1]",
  );
  return Object.freeze({ out: output });
}
export function proveQMaskedMean(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input', 'mask'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  assertRank(inputs.mask.shape, 0, 2, "operator input 'mask'.shape");
  const inputQuantization = assertPerTensorByte(inputs.input, "operator input 'input'");
  assertI32Tensor(inputs.mask, "operator input 'mask'");
  if (!dimensionsProvablyEqual(inputs.mask.shape[0], inputs.input.shape[0], request.environment) ||
      !dimensionsProvablyEqual(inputs.mask.shape[1], inputs.input.shape[1], request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      "operator input 'mask'.shape",
      'must be provably [B,S] for input [B,S,D].',
    );
  }
  const output = logicalQuantizedDeclaredOutput(
    request,
    [inputs.input.shape[0], inputs.input.shape[2]],
  );
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertPositiveF32Ratio(inputQuantization.scale, 1, outputQuantization.scale,
    "declared output 'out'.quantization.scale");
  assertI32CenteredSumBound(
    maximumDimension(inputs.input.shape[1], request.environment),
    centeredMagnitude(inputs.input.dtype, inputQuantization.zero_point),
    "operator input 'input'.shape[1]",
  );
  return domainInference(
    Object.freeze({ out: output }),
    ['[B,S] keep-mask geometry and the maximum centered sequence sum are proved'],
  );
}
interface QAttentionParameters {
  readonly heads: number;
  readonly scale: number;
}
function qAttentionParameters(
  params: Readonly<Record<string, unknown>>,
): QAttentionParameters {
  assertAllowedFields(params, ['heads', 'causal', 'scale'], 'operator params');
  if (!Number.isSafeInteger(params.heads) || (params.heads as number) <= 0) {
    fail('INVALID_PARAMS', 'operator params.heads', 'must be a positive safe integer.');
  }
  if (typeof params.causal !== 'boolean') {
    fail('INVALID_PARAMS', 'operator params.causal', 'must be boolean.');
  }
  positiveF32Param(params, 'scale', true);
  return Object.freeze({ heads: params.heads as number, scale: Math.fround(params.scale as number) });
}
function validateQAttentionFeature(feature: number, heads: number, path: string): number {
  if (feature % heads !== 0 || feature % 4 !== 0) {
    fail('SHAPE_MISMATCH', path, 'D must be divisible by heads and by 4.');
  }
  const headDimension = feature / heads;
  if (headDimension % 4 !== 0 || headDimension > 64) {
    fail('SHAPE_MISMATCH', path, 'head_dim must be divisible by 4 and no greater than 64.');
  }
  return headDimension;
}
function validateQAttentionScales(
  q: PerTensorQuantization,
  k: PerTensorQuantization,
  v: PerTensorQuantization,
  output: PerTensorQuantization,
  scale: number,
  maximumDot: number,
): void {
  const scoreMultiplier = Math.fround(Math.fround(q.scale * k.scale) * scale);
  const maximumScore = Math.fround(maximumDot * scoreMultiplier);
  if (!Number.isFinite(scoreMultiplier) || scoreMultiplier <= 0 || !Number.isFinite(maximumScore)) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      'operator params.scale',
      'quantized score scale and maximum score must be finite positive float32 values.',
    );
  }
  assertPositiveF32Ratio(v.scale, 1, output.scale, "declared output 'out'.quantization.scale");
}
export function inferQSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['q', 'k', 'v'], ['mask']);
  const params = qAttentionParameters(paramsRecord(request.params));
  const qQuantization = assertPerTensorByte(inputs.q, "operator input 'q'");
  const kQuantization = assertPerTensorByte(inputs.k, "operator input 'k'");
  const vQuantization = assertPerTensorByte(inputs.v, "operator input 'v'");
  assertAttentionRank(inputs.q.shape, "operator input 'q'.shape");
  const rank = inputs.q.shape.length;
  if (inputs.k.shape.length !== rank || inputs.v.shape.length !== rank) {
    fail('INVALID_RANK', 'operator inputs', 'q, k, and v must have the same rank.');
  }
  const batch = rank === 2 ? 1 : inputs.q.shape[0];
  const queries = inputs.q.shape.at(-2)!;
  const keys = inputs.k.shape.at(-2)!;
  const feature = inputs.q.shape.at(-1)!;
  if ((rank === 3 && (inputs.k.shape[0] !== batch || inputs.v.shape[0] !== batch)) ||
      inputs.k.shape.at(-1) !== feature || inputs.v.shape.at(-1) !== feature ||
      inputs.v.shape.at(-2) !== keys) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'QSDPA q/k/v batch, feature, and K/V sequence geometry must match.');
  }
  const headDimension = validateQAttentionFeature(feature, params.heads,
    `operator input 'q'.shape[${rank - 1}]`);
  assertConcreteAttentionMask(inputs.mask, batch, queries, keys);
  const output = concreteQuantizedDeclaredOutput(request, inputs.q.shape);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertI32AccumulatorBound(
    headDimension,
    centeredMagnitude(inputs.q.dtype, qQuantization.zero_point),
    centeredMagnitude(inputs.k.dtype, kQuantization.zero_point),
    `operator input 'q'.shape[${rank - 1}]`,
  );
  const maximumDot = headDimension *
    centeredMagnitude(inputs.q.dtype, qQuantization.zero_point) *
    centeredMagnitude(inputs.k.dtype, kQuantization.zero_point);
  validateQAttentionScales(
    qQuantization,
    kQuantization,
    vQuantization,
    outputQuantization,
    params.scale,
    maximumDot,
  );
  return Object.freeze({ out: output });
}
export function proveQSDPA(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['q', 'k', 'v'],
    ['mask'],
  );
  const params = qAttentionParameters(paramsRecord(request.params));
  const qQuantization = assertPerTensorByte(inputs.q, "operator input 'q'");
  const kQuantization = assertPerTensorByte(inputs.k, "operator input 'k'");
  const vQuantization = assertPerTensorByte(inputs.v, "operator input 'v'");
  assertAttentionRank(inputs.q.shape, "operator input 'q'.shape");
  const rank = inputs.q.shape.length;
  if (inputs.k.shape.length !== rank || inputs.v.shape.length !== rank) {
    fail('INVALID_RANK', 'operator inputs', 'q, k, and v must have the same rank.');
  }
  const batch = rank === 2 ? 1 : inputs.q.shape[0];
  const queries = inputs.q.shape.at(-2)!;
  const keys = inputs.k.shape.at(-2)!;
  const featureSpec = inputs.q.shape.at(-1)!;
  const sameBatch = rank === 2 || (
    dimensionsProvablyEqual(inputs.k.shape[0], batch, request.environment) &&
    dimensionsProvablyEqual(inputs.v.shape[0], batch, request.environment)
  );
  if (!sameBatch ||
      !dimensionsProvablyEqual(inputs.k.shape.at(-1)!, featureSpec, request.environment) ||
      !dimensionsProvablyEqual(inputs.v.shape.at(-1)!, featureSpec, request.environment) ||
      !dimensionsProvablyEqual(inputs.v.shape.at(-2)!, keys, request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      'operator inputs',
      'QSDPA q/k/v geometry is not equal over the complete domain.',
    );
  }
  const feature = fixedDimensionValue(featureSpec, request.environment);
  if (feature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE',
      `operator input 'q'.shape[${rank - 1}]`,
      'QSDPA requires a fixed D for heads and packed dot products.',
    );
  }
  const headDimension = validateQAttentionFeature(feature, params.heads,
    `operator input 'q'.shape[${rank - 1}]`);
  assertLogicalAttentionMask(inputs.mask, batch, queries, keys, request.environment);
  const output = logicalQuantizedDeclaredOutput(request, inputs.q.shape);
  const outputQuantization = assertPerTensorByte(output, "declared output 'out'");
  assertI32AccumulatorBound(
    headDimension,
    centeredMagnitude(inputs.q.dtype, qQuantization.zero_point),
    centeredMagnitude(inputs.k.dtype, kQuantization.zero_point),
    `operator input 'q'.shape[${rank - 1}]`,
  );
  const maximumDot = headDimension *
    centeredMagnitude(inputs.q.dtype, qQuantization.zero_point) *
    centeredMagnitude(inputs.k.dtype, kQuantization.zero_point);
  validateQAttentionScales(
    qQuantization,
    kQuantization,
    vQuantization,
    outputQuantization,
    params.scale,
    maximumDot,
  );
  return domainInference(
    Object.freeze({ out: output }),
    ['B/Q/K may vary; fixed D/head geometry, masks, I32 dot bound, and score scales are proved'],
  );
}
function qArgMaxAxis(params: Readonly<Record<string, unknown>>, rank: number): number {
  assertAllowedFields(params, ['axis'], 'operator params');
  if (ownNames(params, 'operator params').length !== 1 || params.axis !== -1) {
    fail('INVALID_PARAMS', 'operator params.axis', 'must be exactly -1 for canonical QArgMax.');
  }
  return rank - 1;
}
export function inferQArgMax(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  if (inputs.input.shape.length < 2 || inputs.input.shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank 2 through 8.');
  }
  const axis = qArgMaxAxis(paramsRecord(request.params), inputs.input.shape.length);
  return cloneConcreteOutput(inputs.input.shape.slice(0, axis), 'int32');
}
export function proveQArgMax(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  if (inputs.input.shape.length < 2 || inputs.input.shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank 2 through 8.');
  }
  const axis = qArgMaxAxis(paramsRecord(request.params), inputs.input.shape.length);
  return domainInference(
    cloneLogicalOutput(inputs.input.shape.slice(0, axis), 'int32', request.environment),
    ['the final positive vocabulary axis is removed and every dynamic prefix is preserved'],
  );
}
