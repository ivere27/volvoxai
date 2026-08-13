/* Shape contracts for BatchMatMul and the spatial operators: Conv1D/2D,
 * ConvTranspose2D, pooling, Resize and Upsample.
 *
 * Mirrors shape_contract_spatial.inc on the native side and
 * tests/operator_shape_contract_spatial_vectors.json.
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
  quantizationEqual,
  remapPerAxis,
  reshapeQuantization,
  safeIntegerArray,
  sameConcreteShape,
  slicedPerAxis,
  spatialPairParam,
  spatialScalarParam,
  validateActivationParams,
} from './operatorShapeContractsCommon.js';


export function inferBatchMatMul(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  assertFloatTensor(inputs.a, "operator input 'a'");
  assertFloatTensor(inputs.b, "operator input 'b'");
  assertRank(inputs.a.shape, 2, null, "operator input 'a'.shape");
  assertRank(inputs.b.shape, 2, null, "operator input 'b'.shape");
  if (inputs.a.shape.length > 8 || inputs.b.shape.length > 8) {
    fail('INVALID_RANK', 'operator inputs', 'BatchMatMul operand ranks must not exceed 8.');
  }
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const leftK = inputs.a.shape.at(-1)!;
  const rightK = inputs.b.shape.at(-2)!;
  if (leftK !== rightK) {
    fail('SHAPE_MISMATCH', "operator input 'b'.shape", `contracts ${rightK}, but a contracts ${leftK}.`);
  }
  const batch = concreteBroadcastShape(inputs.a.shape.slice(0, -2), inputs.b.shape.slice(0, -2));
  return cloneConcreteOutput([...batch, inputs.a.shape.at(-2)!, inputs.b.shape.at(-1)!], 'float32');
}
export function proveBatchMatMul(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  assertFloatTensor(inputs.a, "operator input 'a'");
  assertFloatTensor(inputs.b, "operator input 'b'");
  assertRank(inputs.a.shape, 2, null, "operator input 'a'.shape");
  assertRank(inputs.b.shape, 2, null, "operator input 'b'.shape");
  if (inputs.a.shape.length > 8 || inputs.b.shape.length > 8) {
    fail('INVALID_RANK', 'operator inputs', 'BatchMatMul operand ranks must not exceed 8.');
  }
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const leftK = inputs.a.shape.at(-1)!;
  const rightK = inputs.b.shape.at(-2)!;
  if (!dimensionsProvablyEqual(leftK, rightK, request.environment)) {
    fail(
      'UNPROVABLE_DYNAMIC_CONTRACTION',
      "operator input 'b'.shape",
      `contracted dimensions '${String(leftK)}' and '${String(rightK)}' are not equal over the complete domain.`,
    );
  }
  const batch = logicalBroadcastShape(
    inputs.a.shape.slice(0, -2),
    inputs.b.shape.slice(0, -2),
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(
      [...batch, inputs.a.shape.at(-2)!, inputs.b.shape.at(-1)!],
      'float32',
      request.environment,
    ),
    ['contracted matrix dimensions and every right-aligned batch broadcast are proved'],
  );
}
function conv1DParams(params: Readonly<Record<string, unknown>>) {
  assertAllowedFields(
    params,
    ['stride', 'padding', 'groups', 'relu', 'data_layout', 'weight_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NLC');
  canonicalLayoutParam(params, 'weight_layout', 'WIO');
  return Object.freeze({
    stride: spatialScalarParam(params.stride, 1, 'operator params.stride', false),
    padding: spatialScalarParam(params.padding, 0, 'operator params.padding', true),
    groups: integerParam(params, 'groups', 1),
    relu: activationParam(params),
  });
}
export function inferConv1D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 3, "operator input 'weight'.shape");
  const params = conv1DParams(paramsRecord(request.params));
  const [batch, length, inputChannels] = inputs.input.shape;
  const [kernel, weightChannels, outputChannels] = inputs.weight.shape;
  if (inputChannels % params.groups !== 0 || outputChannels % params.groups !== 0 ||
      weightChannels !== inputChannels / params.groups) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape[1]", 'is incompatible with input channels and groups.');
  }
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, outputChannels, "operator input 'bias'.shape");
  }
  const outputLength = checkedWindowOutput(
    length, kernel, params.stride, params.padding, params.padding, 1,
    "operator input 'input'.shape[1]",
  );
  return cloneConcreteOutput([batch, outputLength, outputChannels], 'float32');
}
export function proveConv1D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['input', 'weight'], ['bias'],
  );
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 3, "operator input 'weight'.shape");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  const params = conv1DParams(paramsRecord(request.params));
  const inputChannels = fixedDimensionValue(inputs.input.shape[2], request.environment);
  if (inputChannels === undefined) {
    fail('UNPROVABLE_DYNAMIC_CHANNEL', "operator input 'input'.shape[2]", 'must be fixed for grouped Conv1D.');
  }
  if (inputChannels % params.groups !== 0 || weight[2] % params.groups !== 0 ||
      weight[1] !== inputChannels / params.groups) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape[1]", 'is incompatible with fixed input channels and groups.');
  }
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(inputs.bias.shape, weight[2], request.environment, "operator input 'bias'.shape");
  }
  const outputLength = logicalWindowOutput(
    inputs.input.shape[1], request.environment, weight[0], params.stride,
    params.padding, params.padding, 1, "operator input 'input'.shape[1]",
  );
  return domainInference(
    cloneLogicalOutput([inputs.input.shape[0], outputLength, weight[2]], 'float32', request.environment),
    ['fixed WIO weight/group geometry and the complete NLC spatial formula are proved'],
  );
}
function conv2DParams(params: Readonly<Record<string, unknown>>) {
  assertAllowedFields(
    params,
    ['stride', 'padding', 'pads', 'dilation', 'groups', 'relu', 'data_layout', 'weight_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  const weightLayout = params.weight_layout ?? 'HWIO';
  if (weightLayout !== 'HWIO' && weightLayout !== 'HWCM') {
    fail('INVALID_PARAMS', 'operator params.weight_layout', "must be 'HWIO' or 'HWCM'.");
  }
  return Object.freeze({
    stride: spatialPairParam(params.stride, 1, 'operator params.stride', false),
    pads: fullSpatialPads(params),
    dilation: spatialPairParam(params.dilation, 1, 'operator params.dilation', false),
    groups: integerParam(params, 'groups', 1),
    relu: activationParam(params),
    weightLayout,
  });
}
function conv2DChannels(
  inputChannels: number,
  weight: readonly number[],
  groups: number,
  weightLayout: unknown,
): number {
  if (weightLayout === 'HWCM') {
    if (groups !== inputChannels || weight[2] !== inputChannels) {
      fail('SHAPE_MISMATCH', "operator input 'weight'.shape", 'depthwise Conv2D requires HWCM [kh,kw,input_channels,multiplier].');
    }
    return checkedShapeMultiply(inputChannels, weight[3], 'Conv2D output channels');
  }
  if (weightLayout !== 'HWIO' || inputChannels % groups !== 0 || weight[3] % groups !== 0 ||
      weight[2] !== inputChannels / groups) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape", 'grouped Conv2D requires compatible HWIO [kh,kw,input_channels/groups,output_channels].');
  }
  return weight[3];
}
export function inferConv2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const params = conv2DParams(paramsRecord(request.params));
  const [batch, height, width, channels] = inputs.input.shape;
  const outputChannels = conv2DChannels(
    channels, inputs.weight.shape, params.groups, params.weightLayout,
  );
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, outputChannels, "operator input 'bias'.shape");
  }
  const outHeight = checkedWindowOutput(
    height, inputs.weight.shape[0], params.stride[0], params.pads[0], params.pads[2],
    params.dilation[0], "operator input 'input'.shape[1]",
  );
  const outWidth = checkedWindowOutput(
    width, inputs.weight.shape[1], params.stride[1], params.pads[1], params.pads[3],
    params.dilation[1], "operator input 'input'.shape[2]",
  );
  return cloneConcreteOutput([batch, outHeight, outWidth, outputChannels], 'float32');
}
export function proveConv2D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['input', 'weight'], ['bias'],
  );
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  const params = conv2DParams(paramsRecord(request.params));
  const inputChannels = fixedDimensionValue(inputs.input.shape[3], request.environment);
  if (inputChannels === undefined) {
    fail('UNPROVABLE_DYNAMIC_CHANNEL', "operator input 'input'.shape[3]", 'must be fixed for grouped Conv2D.');
  }
  const outputChannels = conv2DChannels(inputChannels, weight, params.groups, params.weightLayout);
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(inputs.bias.shape, outputChannels, request.environment, "operator input 'bias'.shape");
  }
  const outHeight = logicalWindowOutput(
    inputs.input.shape[1], request.environment, weight[0], params.stride[0],
    params.pads[0], params.pads[2], params.dilation[0], "operator input 'input'.shape[1]",
  );
  const outWidth = logicalWindowOutput(
    inputs.input.shape[2], request.environment, weight[1], params.stride[1],
    params.pads[1], params.pads[3], params.dilation[1], "operator input 'input'.shape[2]",
  );
  return domainInference(
    cloneLogicalOutput(
      [inputs.input.shape[0], outHeight, outWidth, outputChannels],
      'float32',
      request.environment,
    ),
    ['fixed image-layout weight/group geometry and both NHWC spatial formulas are proved'],
  );
}
function convTranspose2DParams(params: Readonly<Record<string, unknown>>) {
  assertAllowedFields(
    params,
    ['kernel', 'stride', 'padding', 'data_layout', 'weight_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  canonicalLayoutParam(params, 'weight_layout', 'HWIO');
  return Object.freeze({
    kernel: spatialPairParam(params.kernel, 1, 'operator params.kernel', false, true),
    stride: spatialPairParam(params.stride, 1, 'operator params.stride', false),
    padding: spatialPairParam(params.padding, 0, 'operator params.padding', true),
  });
}
export function inferConvTranspose2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'weight'], ['bias']);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const params = convTranspose2DParams(paramsRecord(request.params));
  if (params.kernel[0] !== inputs.weight.shape[0] || params.kernel[1] !== inputs.weight.shape[1]) {
    fail('SHAPE_MISMATCH', 'operator params.kernel', 'must equal the HWIO weight kernel extents.');
  }
  if (inputs.weight.shape[2] !== inputs.input.shape[3]) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape[2]", 'must equal input channels.');
  }
  const outputChannels = inputs.weight.shape[3];
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    assertVectorShape(inputs.bias.shape, outputChannels, "operator input 'bias'.shape");
  }
  return cloneConcreteOutput([
    inputs.input.shape[0],
    checkedTransposeOutput(inputs.input.shape[1], params.kernel[0], params.stride[0], params.padding[0], "operator input 'input'.shape[1]"),
    checkedTransposeOutput(inputs.input.shape[2], params.kernel[1], params.stride[1], params.padding[1], "operator input 'input'.shape[2]"),
    outputChannels,
  ], 'float32');
}
export function proveConvTranspose2D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs, request.environment, ['input', 'weight'], ['bias'],
  );
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertFloatTensor(inputs.weight, "operator input 'weight'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  assertRank(inputs.weight.shape, 0, 4, "operator input 'weight'.shape");
  const weight = constantLogicalShape(inputs.weight.shape, "operator input 'weight'.shape");
  const params = convTranspose2DParams(paramsRecord(request.params));
  if (params.kernel[0] !== weight[0] || params.kernel[1] !== weight[1]) {
    fail('SHAPE_MISMATCH', 'operator params.kernel', 'must equal the HWIO weight kernel extents.');
  }
  const inputChannels = fixedDimensionValue(inputs.input.shape[3], request.environment);
  if (inputChannels === undefined) {
    fail('UNPROVABLE_DYNAMIC_CHANNEL', "operator input 'input'.shape[3]", 'must be fixed for ConvTranspose2D.');
  }
  if (weight[2] !== inputChannels) {
    fail('SHAPE_MISMATCH', "operator input 'weight'.shape[2]", 'must equal fixed input channels.');
  }
  if (inputs.bias !== undefined) {
    assertFloatTensor(inputs.bias, "operator input 'bias'");
    constantLogicalShape(inputs.bias.shape, "operator input 'bias'.shape");
    assertLogicalVectorShape(inputs.bias.shape, weight[3], request.environment, "operator input 'bias'.shape");
  }
  return domainInference(
    cloneLogicalOutput([
      inputs.input.shape[0],
      logicalTransposeOutput(inputs.input.shape[1], request.environment, params.kernel[0], params.stride[0], params.padding[0], "operator input 'input'.shape[1]"),
      logicalTransposeOutput(inputs.input.shape[2], request.environment, params.kernel[1], params.stride[1], params.padding[1], "operator input 'input'.shape[2]"),
      weight[3],
    ], 'float32', request.environment),
    ['fixed HWIO channel geometry and both transposed spatial formulas are proved'],
  );
}
function pool2DParams(operator: string, params: Readonly<Record<string, unknown>>) {
  const average = operator === 'AveragePool2D';
  assertAllowedFields(
    params,
    average
      ? ['kernel', 'stride', 'padding', 'pads', 'dilation', 'ceil_mode', 'count_include_pad', 'auto_pad', 'data_layout']
      : ['kernel', 'stride', 'padding', 'pads', 'dilation', 'ceil_mode', 'data_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  falseOrAbsentParam(params.ceil_mode, 'operator params.ceil_mode');
  if (average) {
    falseOrAbsentParam(params.count_include_pad, 'operator params.count_include_pad');
    if (params.auto_pad !== undefined && params.auto_pad !== '' && params.auto_pad !== 'NOTSET') {
      fail('INVALID_PARAMS', 'operator params.auto_pad', "must be 'NOTSET', empty, or absent.");
    }
  }
  const dilation = spatialPairParam(params.dilation, 1, 'operator params.dilation', false);
  if (dilation[0] !== 1 || dilation[1] !== 1) {
    fail('INVALID_PARAMS', 'operator params.dilation', 'dilated pooling is not supported.');
  }
  return Object.freeze({
    kernel: spatialPairParam(params.kernel, 1, 'operator params.kernel', false, true),
    stride: spatialPairParam(params.stride, 1, 'operator params.stride', false),
    pads: fullSpatialPads(params, average),
  });
}
function spatialPoolDType(
  operator: string,
  descriptor: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
): RuntimeDType {
  if (descriptor.dtype === 'float32') {
    assertUnquantized(descriptor, "operator input 'input'");
    return 'float32';
  }
  if (operator !== 'MaxPool2D' || (descriptor.dtype !== 'int8' && descriptor.dtype !== 'uint8')) {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", `${operator} requires float32${operator === 'MaxPool2D' ? ' or I8/U8' : ''}.`);
  }
  if (descriptor.quantization?.scheme !== 'per_tensor') {
    fail('INVALID_QUANTIZATION', "operator input 'input'.quantization", 'raw byte MaxPool2D requires per-tensor quantization.');
  }
  return descriptor.dtype;
}
export function inferPool2D(operator: string, request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const dtype = spatialPoolDType(operator, inputs.input);
  const params = pool2DParams(operator, paramsRecord(request.params));
  const [batch, height, width, channels] = inputs.input.shape;
  return cloneConcreteOutput([
    batch,
    checkedWindowOutput(height, params.kernel[0], params.stride[0], params.pads[0], params.pads[2], 1, "operator input 'input'.shape[1]"),
    checkedWindowOutput(width, params.kernel[1], params.stride[1], params.pads[1], params.pads[3], 1, "operator input 'input'.shape[2]"),
    channels,
  ], dtype, inputs.input.quantization ?? undefined);
}
export function provePool2D(operator: string, request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const dtype = spatialPoolDType(operator, inputs.input);
  const params = pool2DParams(operator, paramsRecord(request.params));
  return domainInference(
    cloneLogicalOutput([
      inputs.input.shape[0],
      logicalWindowOutput(inputs.input.shape[1], request.environment, params.kernel[0], params.stride[0], params.pads[0], params.pads[2], 1, "operator input 'input'.shape[1]"),
      logicalWindowOutput(inputs.input.shape[2], request.environment, params.kernel[1], params.stride[1], params.pads[1], params.pads[3], 1, "operator input 'input'.shape[2]"),
      inputs.input.shape[3],
    ], dtype, request.environment, inputs.input.quantization ?? undefined),
    ['both NHWC floor-window formulas are representable over the complete domain'],
  );
}
export function inferGlobalAveragePool(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  return cloneConcreteOutput([inputs.input.shape[0], 1, 1, inputs.input.shape[3]], 'float32');
}
export function proveGlobalAveragePool(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  return domainInference(
    cloneLogicalOutput([inputs.input.shape[0], 1, 1, inputs.input.shape[3]], 'float32', request.environment),
    ['GlobalAveragePool reduces both positive spatial dimensions to one'],
  );
}
function resizeParams(operator: string, params: Readonly<Record<string, unknown>>, byteStorage: boolean): void {
  assertAllowedFields(
    params,
    ['mode', 'coordinate_transformation_mode', 'coordinate_transform_mode', 'nearest_mode', 'align_corners', 'antialias', 'data_layout'],
    'operator params',
  );
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  if (params.coordinate_transform_mode !== undefined) {
    fail('INVALID_PARAMS', 'operator params.coordinate_transform_mode', 'is not defined; use coordinate_transformation_mode.');
  }
  const nearest = operator === 'ResizeNearest2D' || params.mode === 'nearest';
  if (operator === 'ResizeNearest2D' && params.mode !== undefined && params.mode !== 'nearest') {
    fail('INVALID_PARAMS', 'operator params.mode', "must be 'nearest'.");
  }
  if (operator === 'Resize' && params.mode !== undefined && params.mode !== 'nearest' && params.mode !== 'linear') {
    fail('INVALID_PARAMS', 'operator params.mode', "must be 'nearest' or 'linear'.");
  }
  if (byteStorage && !nearest) {
    fail('INVALID_PARAMS', 'operator params.mode', 'raw I8/U8 resize requires explicit nearest mode.');
  }
  const transform = params.coordinate_transformation_mode;
  if (nearest) {
    if (transform !== undefined && transform !== 'asymmetric') {
      fail('INVALID_PARAMS', 'operator params.coordinate_transformation_mode', "must be 'asymmetric' for nearest resize.");
    }
    if (params.nearest_mode !== undefined && params.nearest_mode !== 'floor') {
      fail('INVALID_PARAMS', 'operator params.nearest_mode', "must be 'floor'.");
    }
  } else {
    if (transform !== undefined && transform !== 'half_pixel') {
      fail('INVALID_PARAMS', 'operator params.coordinate_transformation_mode', "must be 'half_pixel' for linear resize.");
    }
    if (params.nearest_mode !== undefined) {
      fail('INVALID_PARAMS', 'operator params.nearest_mode', 'is valid only for nearest resize.');
    }
  }
  falseOrAbsentParam(params.align_corners, 'operator params.align_corners');
  falseOrAbsentParam(params.antialias, 'operator params.antialias');
}
function resizeOutputDType(
  input: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
  output: OperatorTensorDescriptor | LogicalOperatorTensorDescriptor,
): RuntimeDType {
  if (input.dtype === 'float32') {
    assertUnquantized(input, "operator input 'input'");
    if (output.dtype !== 'float32') fail('INVALID_DTYPE', "declared output 'out'.dtype", "must be 'float32'.");
    assertUnquantized(output, "declared output 'out'");
    return 'float32';
  }
  if ((input.dtype !== 'int8' && input.dtype !== 'uint8') || input.quantization?.scheme !== 'per_tensor') {
    fail('INVALID_QUANTIZATION', "operator input 'input'.quantization", 'raw byte resize requires per-tensor I8/U8 quantization.');
  }
  if (output.dtype !== input.dtype || !quantizationEqual(input.quantization, output.quantization)) {
    fail('INVALID_QUANTIZATION', "declared output 'out'.quantization", 'must preserve the exact input byte domain.');
  }
  return input.dtype;
}
export function inferResize(operator: string, request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const declared = normalizeConcreteDeclaredOutput(request.declaredOutputs);
  assertRank(declared.shape, 0, 4, "declared output 'out'.shape");
  const dtype = resizeOutputDType(inputs.input, declared);
  resizeParams(operator, paramsRecord(request.params), dtype !== 'float32');
  if (declared.shape[0] !== inputs.input.shape[0] || declared.shape[3] !== inputs.input.shape[3]) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must preserve NHWC batch and channel extents.');
  }
  return cloneConcreteOutput(declared.shape, dtype, declared.quantization ?? undefined);
}
export function proveResize(operator: string, request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const declared = normalizeLogicalDeclaredOutput(request.declaredOutputs, request.environment);
  assertRank(declared.shape, 0, 4, "declared output 'out'.shape");
  const dtype = resizeOutputDType(inputs.input, declared);
  resizeParams(operator, paramsRecord(request.params), dtype !== 'float32');
  if (!dimensionsProvablyEqual(declared.shape[0], inputs.input.shape[0], request.environment) ||
      !dimensionsProvablyEqual(declared.shape[3], inputs.input.shape[3], request.environment)) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'does not provably preserve NHWC batch and channel extents.');
  }
  return domainInference(
    cloneLogicalOutput(declared.shape, dtype, request.environment, declared.quantization ?? undefined),
    ['the explicit bounded output target preserves NHWC batch and channels for every binding'],
  );
}
export function inferUpsampleNearest2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  return cloneConcreteOutput([
    inputs.input.shape[0],
    checkedShapeMultiply(inputs.input.shape[1], 2, 'UpsampleNearest2D output height'),
    checkedShapeMultiply(inputs.input.shape[2], 2, 'UpsampleNearest2D output width'),
    inputs.input.shape[3],
  ], 'float32');
}

export function proveUpsampleNearest2D(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  const spatial = [1, 2].map((axis) => {
    const fixed = fixedDimensionValue(inputs.input.shape[axis], request.environment);
    if (fixed === undefined) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `operator input 'input'.shape[${axis}]`,
        `2x output from dynamic '${inputs.input.shape[axis]}' is not one v1 constant-or-symbol dimension.`,
      );
    }
    return checkedShapeMultiply(fixed, 2, `UpsampleNearest2D output axis ${axis}`);
  });
  return domainInference(
    cloneLogicalOutput(
      [inputs.input.shape[0], spatial[0], spatial[1], inputs.input.shape[3]],
      'float32',
      request.environment,
    ),
    ['both fixed spatial extents have exact checked 2x outputs'],
  );
}
