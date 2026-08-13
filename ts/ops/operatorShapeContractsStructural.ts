/* Shape contracts for the broadcasting and structural operators: elementwise
 * broadcast arithmetic and comparison, Where, reductions, ArgMax, and the
 * shape movers (Transpose, Flatten, Squeeze, Unsqueeze, Reshape, Expand,
 * Concat, Split, Slice, Pad, Gather, GatherElements).
 *
 * Mirrors shape_contract_structural.inc on the native side and
 * tests/operator_shape_contract_structural_vectors.json.
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


export function inferBroadcastArithmetic(
  _operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  assertFloatTensor(inputs.a, "operator input 'a'");
  assertFloatTensor(inputs.b, "operator input 'b'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return cloneConcreteOutput(concreteBroadcastShape(inputs.a.shape, inputs.b.shape), 'float32');
}
export function proveBroadcastArithmetic(
  _operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  assertFloatTensor(inputs.a, "operator input 'a'");
  assertFloatTensor(inputs.b, "operator input 'b'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return domainInference(
    cloneLogicalOutput(
      logicalBroadcastShape(inputs.a.shape, inputs.b.shape, request.environment),
      'float32',
      request.environment,
    ),
    ['right-aligned arithmetic broadcasting is proved axis-by-axis over the bounded domain'],
  );
}
export function inferBroadcastComparison(
  _operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  if (inputs.a.dtype !== 'int32' || inputs.b.dtype !== 'int32') {
    fail('INVALID_DTYPE', 'operator inputs', 'comparison inputs must both be int32.');
  }
  assertUnquantized(inputs.a, "operator input 'a'");
  assertUnquantized(inputs.b, "operator input 'b'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return cloneConcreteOutput(concreteBroadcastShape(inputs.a.shape, inputs.b.shape), 'int32');
}
export function proveBroadcastComparison(
  _operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  if (inputs.a.dtype !== 'int32' || inputs.b.dtype !== 'int32') {
    fail('INVALID_DTYPE', 'operator inputs', 'comparison inputs must both be int32.');
  }
  assertUnquantized(inputs.a, "operator input 'a'");
  assertUnquantized(inputs.b, "operator input 'b'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return domainInference(
    cloneLogicalOutput(
      logicalBroadcastShape(inputs.a.shape, inputs.b.shape, request.environment),
      'int32',
      request.environment,
    ),
    ['I32 comparison output uses the proved right-aligned broadcast shape'],
  );
}
function assertWhereTypes(
  condition: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  a: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  b: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
): RuntimeDType {
  if (condition.dtype !== 'float32' && condition.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'condition'.dtype", 'must be float32 or int32.');
  }
  assertUnquantized(condition, "operator input 'condition'");
  const dtype = assertSameStorageDType(a, b, "operator inputs 'a' and 'b'");
  if (dtype !== 'float32' && dtype !== 'int32') {
    fail('INVALID_DTYPE', 'operator data inputs', 'must be float32 or int32.');
  }
  assertUnquantized(a, "operator input 'a'");
  assertUnquantized(b, "operator input 'b'");
  return dtype;
}
export function inferWhere(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['condition', 'a', 'b'], []);
  const dtype = assertWhereTypes(inputs.condition, inputs.a, inputs.b);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const dataShape = concreteBroadcastShape(inputs.a.shape, inputs.b.shape);
  return cloneConcreteOutput(concreteBroadcastShape(inputs.condition.shape, dataShape), dtype);
}
export function proveWhere(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['condition', 'a', 'b'],
    [],
  );
  const dtype = assertWhereTypes(inputs.condition, inputs.a, inputs.b);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const dataShape = logicalBroadcastShape(
    inputs.a.shape,
    inputs.b.shape,
    request.environment,
  );
  const outputShape = logicalBroadcastShape(
    inputs.condition.shape,
    dataShape,
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(outputShape, dtype, request.environment),
    ['condition, true, and false inputs share one proved right-aligned broadcast result'],
  );
}
function reductionParams(
  params: Readonly<Record<string, unknown>>,
  rank: number,
): Readonly<{ axis: number; keepdims: boolean }> {
  assertAllowedFields(params, ['axis', 'keepdims'], 'operator params');
  const axis = normalizeAxis(params.axis, rank, -1);
  if (axis !== rank - 1) {
    fail('INVALID_PARAMS', 'operator params.axis', 'must resolve to the last axis.');
  }
  return Object.freeze({ axis, keepdims: booleanParam(params, 'keepdims', true) });
}
export function inferReduction(
  _operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = reductionParams(paramsRecord(request.params), inputs.input.shape.length);
  const shape = params.keepdims
    ? [...inputs.input.shape.slice(0, -1), 1]
    : [...inputs.input.shape.slice(0, -1)];
  return cloneConcreteOutput(shape, 'float32');
}
export function proveReduction(
  _operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = reductionParams(paramsRecord(request.params), inputs.input.shape.length);
  const shape = Object.freeze(params.keepdims
    ? [...inputs.input.shape.slice(0, -1), 1]
    : [...inputs.input.shape.slice(0, -1)]);
  return domainInference(
    cloneLogicalOutput(shape, 'float32', request.environment),
    ['the canonical reduction removes or singletonizes only the last axis'],
  );
}
function argMaxParams(
  params: Readonly<Record<string, unknown>>,
  rank: number,
): Readonly<{ axis: number; keepdims: boolean }> {
  assertAllowedFields(params, ['axis', 'keepdims', 'select_last_index'], 'operator params');
  const axis = normalizeAxis(params.axis, rank, 0);
  const keepdims = booleanParam(params, 'keepdims', true);
  const selectLastIndex = params.select_last_index ?? 0;
  if (selectLastIndex !== 0) {
    fail('INVALID_PARAMS', 'operator params.select_last_index', 'must be 0 (first-index ties).');
  }
  return Object.freeze({ axis, keepdims });
}
function argMaxOutputShape<T extends number | string>(
  input: readonly T[],
  axis: number,
  keepdims: boolean,
): readonly (T | number)[] {
  return Object.freeze([
    ...input.slice(0, axis),
    ...(keepdims ? [1] : []),
    ...input.slice(axis + 1),
  ]);
}
export function inferArgMax(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = argMaxParams(paramsRecord(request.params), inputs.input.shape.length);
  return cloneConcreteOutput(
    argMaxOutputShape(inputs.input.shape, params.axis, params.keepdims) as readonly number[],
    'int32',
  );
}
export function proveArgMax(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = argMaxParams(paramsRecord(request.params), inputs.input.shape.length);
  return domainInference(
    cloneLogicalOutput(
      argMaxOutputShape(inputs.input.shape, params.axis, params.keepdims),
      'int32',
      request.environment,
    ),
    [`axis ${params.axis} has positive extent for every binding and ties select the first index`],
  );
}
function transposePermutation(
  params: Readonly<Record<string, unknown>>,
  rank: number,
): readonly number[] {
  assertAllowedFields(params, ['perm'], 'operator params');
  const source = params.perm ?? [...Array(rank).keys()].reverse();
  if (!Array.isArray(source) || source.length !== rank ||
      source.some((axis) => !Number.isInteger(axis) || axis < 0 || axis >= rank) ||
      new Set(source).size !== rank) {
    fail('INVALID_PARAMS', 'operator params.perm', `must be a permutation of [0, ${rank}).`);
  }
  return Object.freeze([...source]);
}
export function inferTranspose(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  if (inputs.input.shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank at most 8.');
  }
  const perm = transposePermutation(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = perm.map((axis) => inputs.input.shape[axis]);
  const quantization = inputs.input.quantization?.scheme === 'per_axis'
    ? remapPerAxis(inputs.input.quantization, perm.indexOf(inputs.input.quantization.axis))
    : inputs.input.quantization ?? undefined;
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveTranspose(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  if (inputs.input.shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank at most 8.');
  }
  const perm = transposePermutation(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = Object.freeze(perm.map((axis) => inputs.input.shape[axis]));
  const quantization = inputs.input.quantization?.scheme === 'per_axis'
    ? remapPerAxis(inputs.input.quantization, perm.indexOf(inputs.input.quantization.axis))
    : inputs.input.quantization ?? undefined;
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['transpose permutes fixed-rank axes and remaps a fixed per-axis affine index'],
  );
}
function flattenAxis(params: Readonly<Record<string, unknown>>, rank: number): number {
  assertAllowedFields(params, ['axis'], 'operator params');
  const raw = params.axis ?? 1;
  if (!Number.isInteger(raw)) fail('INVALID_PARAMS', 'operator params.axis', 'must be an integer.');
  const axis = (raw as number) < 0 ? (raw as number) + rank : raw as number;
  if (axis < 0 || axis > rank) {
    fail('INVALID_PARAMS', 'operator params.axis', `must resolve to a boundary in [0, ${rank}].`);
  }
  return axis;
}
export function inferFlatten(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const axis = flattenAxis(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = Object.freeze([
    checkedDimensionsProduct(inputs.input.shape.slice(0, axis), 'Flatten prefix product'),
    checkedDimensionsProduct(inputs.input.shape.slice(axis), 'Flatten suffix product'),
  ]);
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    undefined,
    "operator input 'input'.quantization",
  );
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveFlatten(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const axis = flattenAxis(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = Object.freeze([
    collapseLogicalDimensions(inputs.input.shape.slice(0, axis), request.environment, 'Flatten prefix'),
    collapseLogicalDimensions(inputs.input.shape.slice(axis), request.environment, 'Flatten suffix'),
  ]);
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    request.environment,
    "operator input 'input'.quantization",
  );
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['both flattened products are exactly representable as one v1 constant or symbol'],
  );
}
function squeezeAxesConcrete(
  params: Readonly<Record<string, unknown>>,
  shape: readonly number[],
): readonly number[] {
  assertAllowedFields(params, ['axes'], 'operator params');
  if (params.axes === undefined) {
    fail(
      'INVALID_PARAMS',
      'operator params.axes',
      'is required so Squeeze has one fixed output rank over the complete domain.',
    );
  }
  const axes = normalizeAxes(params.axes, shape.length, 'operator params.axes');
  for (const axis of axes) {
    if (shape[axis] !== 1) {
      fail('SHAPE_MISMATCH', `operator input 'input'.shape[${axis}]`, 'must be 1 to squeeze it.');
    }
  }
  return axes;
}
function squeezeAxesLogical(
  params: Readonly<Record<string, unknown>>,
  shape: TensorShapeSpec,
  environment: ShapeEnvironment,
): readonly number[] {
  assertAllowedFields(params, ['axes'], 'operator params');
  if (params.axes === undefined) {
    fail(
      'INVALID_PARAMS',
      'operator params.axes',
      'is required so Squeeze has one fixed output rank over the complete domain.',
    );
  }
  const axes = normalizeAxes(params.axes, shape.length, 'operator params.axes');
  for (const axis of axes) {
    if (fixedDimensionValue(shape[axis], environment) !== 1) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `operator input 'input'.shape[${axis}]`,
        'must be fixed at 1 over the complete domain to squeeze it.',
      );
    }
  }
  return axes;
}
export function inferSqueeze(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const axes = new Set(squeezeAxesConcrete(paramsRecord(request.params), inputs.input.shape));
  const outputShape = Object.freeze(inputs.input.shape.filter((_dimension, axis) => !axes.has(axis)));
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    undefined,
    "operator input 'input'.quantization",
  );
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveSqueeze(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const axes = new Set(squeezeAxesLogical(
    paramsRecord(request.params),
    inputs.input.shape,
    request.environment,
  ));
  const outputShape = Object.freeze(inputs.input.shape.filter((_dimension, axis) => !axes.has(axis)));
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    request.environment,
    "operator input 'input'.quantization",
  );
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['only axes fixed at one over the complete domain are removed'],
  );
}
function unsqueezeAxes(
  params: Readonly<Record<string, unknown>>,
  inputRank: number,
): readonly number[] {
  assertAllowedFields(params, ['axes'], 'operator params');
  if (params.axes === undefined) {
    fail('INVALID_PARAMS', 'operator params.axes', 'is required.');
  }
  const raw = params.axes;
  if (!Array.isArray(raw)) fail('INVALID_PARAMS', 'operator params.axes', 'must be an array.');
  return normalizeAxes(raw, inputRank, 'operator params.axes', {
    outputRank: inputRank + raw.length,
  });
}
function unsqueezedShape<T extends number | string>(
  input: readonly T[],
  axes: readonly number[],
): readonly (T | number)[] {
  const insertions = new Set(axes);
  const output: Array<T | number> = [];
  let inputAxis = 0;
  for (let outputAxis = 0; outputAxis < input.length + axes.length; outputAxis++) {
    output.push(insertions.has(outputAxis) ? 1 : input[inputAxis++]);
  }
  return Object.freeze(output);
}
export function inferUnsqueeze(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const axes = unsqueezeAxes(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = unsqueezedShape(inputs.input.shape, axes) as readonly number[];
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    undefined,
    "operator input 'input'.quantization",
  );
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveUnsqueeze(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const axes = unsqueezeAxes(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = unsqueezedShape(inputs.input.shape, axes);
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    request.environment,
    "operator input 'input'.quantization",
  );
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['unsqueeze inserts fixed singleton axes and remaps a fixed per-axis affine index'],
  );
}
export function inferReshape(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const outputShape = concreteTargetShape(request.params, request.declaredOutputs, inputs.input.dtype);
  if (checkedShapeElementCount(inputs.input.shape, 'Reshape input') !==
      checkedShapeElementCount(outputShape, 'Reshape output')) {
    fail('SHAPE_MISMATCH', 'operator params.shape', 'must preserve the exact element count.');
  }
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    undefined,
    "operator input 'input'.quantization",
  );
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveReshape(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const outputShape = logicalTargetShape(
    request.params,
    request.environment,
    request.declaredOutputs,
    inputs.input.dtype,
  );
  if (!logicalProductsProvablyEqual(inputs.input.shape, outputShape, request.environment)) {
    fail(
      'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
      'operator params.shape',
      'does not provably preserve the input element product over the complete domain.',
    );
  }
  const quantization = reshapeQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    request.environment,
    "operator input 'input'.quantization",
  );
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['normalized target factors exactly conserve the symbolic input element product'],
  );
}
function assertConcreteExpand(
  input: readonly number[],
  target: readonly number[],
): void {
  if (target.length < input.length || target.length > 8) {
    fail('INVALID_RANK', 'operator params.shape', 'must have rank between input rank and 8.');
  }
  const offset = target.length - input.length;
  for (let axis = 0; axis < target.length; axis++) {
    const inputDimension = axis < offset ? 1 : input[axis - offset];
    if (inputDimension !== 1 && inputDimension !== target[axis]) {
      fail('SHAPE_MISMATCH', `operator params.shape[${axis}]`, 'is not a valid broadcast target.');
    }
  }
}
function assertLogicalExpand(
  input: TensorShapeSpec,
  target: TensorShapeSpec,
  environment: ShapeEnvironment,
): void {
  if (target.length < input.length || target.length > 8) {
    fail('INVALID_RANK', 'operator params.shape', 'must have rank between input rank and 8.');
  }
  const offset = target.length - input.length;
  for (let axis = 0; axis < target.length; axis++) {
    const inputDimension = axis < offset ? 1 : input[axis - offset];
    logicalBroadcastDimension(inputDimension, target[axis], environment, `operator params.shape[${axis}]`);
    const inputFixed = fixedDimensionValue(inputDimension, environment);
    if (inputFixed !== 1 && !dimensionsProvablyEqual(inputDimension, target[axis], environment)) {
      fail(
        'UNPROVABLE_DYNAMIC_BROADCAST',
        `operator params.shape[${axis}]`,
        'the target must equal each non-singleton input dimension over the complete domain.',
      );
    }
  }
}
function expandQuantization(
  quantization: TensorQuantization | null | undefined,
  inputShape: readonly (number | string)[],
  targetShape: readonly (number | string)[],
  environment?: ShapeEnvironment,
): TensorQuantization | undefined {
  if (quantization == null || quantization.scheme === 'per_tensor') return quantization ?? undefined;
  const outputAxis = quantization.axis + targetShape.length - inputShape.length;
  const equal = environment === undefined
    ? inputShape[quantization.axis] === targetShape[outputAxis]
    : dimensionsProvablyEqual(
      inputShape[quantization.axis],
      targetShape[outputAxis],
      environment,
    );
  if (!equal) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      `operator params.shape[${outputAxis}]`,
      'must not expand the per-axis quantization extent.',
    );
  }
  return remapPerAxis(quantization, outputAxis);
}
export function inferExpand(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const outputShape = concreteTargetShape(request.params, request.declaredOutputs, inputs.input.dtype);
  assertConcreteExpand(inputs.input.shape, outputShape);
  const quantization = expandQuantization(inputs.input.quantization, inputs.input.shape, outputShape);
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveExpand(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const outputShape = logicalTargetShape(
    request.params,
    request.environment,
    request.declaredOutputs,
    inputs.input.dtype,
  );
  assertLogicalExpand(inputs.input.shape, outputShape, request.environment);
  const quantization = expandQuantization(
    inputs.input.quantization,
    inputs.input.shape,
    outputShape,
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['every non-singleton input axis equals its normalized target over the complete domain'],
  );
}
function concatParams(params: Readonly<Record<string, unknown>>, rank: number): number {
  assertAllowedFields(params, ['axis'], 'operator params');
  return normalizeAxis(params.axis, rank, 0);
}
function concatQuantization(
  inputs: readonly { readonly quantization?: TensorQuantization | null }[],
  axis: number,
): TensorQuantization | undefined {
  const first = inputs[0].quantization;
  if (inputs.some((input) => !quantizationEqual(input.quantization, first))) {
    if (first?.scheme !== 'per_axis' || first.axis !== axis ||
        inputs.some((input) => input.quantization?.scheme !== 'per_axis' ||
          input.quantization.axis !== axis)) {
      fail(
        'INVALID_QUANTIZATION',
        'operator inputs',
        'Concat inputs must have identical affine metadata unless concatenating their common per-axis dimension.',
      );
    }
  }
  if (first == null) return undefined;
  if (first.scheme === 'per_tensor') return first;
  if (first.axis !== axis) return first;
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  for (const input of inputs) {
    const quantization = input.quantization;
    if (quantization?.scheme !== 'per_axis' || quantization.axis !== axis) {
      fail('INVALID_QUANTIZATION', 'operator inputs', 'have incompatible per-axis metadata.');
    }
    scales.push(...quantization.scales);
    zeroPoints.push(...quantization.zero_points);
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze(scales),
    zero_points: Object.freeze(zeroPoints),
  });
}
export function inferConcat(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteVariadicInputs(request.inputs, 'input', 2);
  const rank = inputs[0].shape.length;
  if (rank < 1 || rank > 8 || inputs.some((input) => input.shape.length !== rank)) {
    fail('INVALID_RANK', 'operator inputs', 'must all have one equal rank in [1, 8].');
  }
  const dtype = inputs[0].dtype;
  if (inputs.some((input) => input.dtype !== dtype)) {
    fail('INVALID_DTYPE', 'operator inputs', 'must all have the same dtype.');
  }
  const axis = concatParams(paramsRecord(request.params), rank);
  const outputShape = [...inputs[0].shape];
  outputShape[axis] = 0;
  for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
    const input = inputs[inputIndex];
    for (let dimension = 0; dimension < rank; dimension++) {
      if (dimension !== axis && input.shape[dimension] !== inputs[0].shape[dimension]) {
        fail('SHAPE_MISMATCH', `operator input 'input${inputIndex}'.shape[${dimension}]`, 'must match input0.');
      }
    }
    outputShape[axis] = checkedShapeAdd(
      outputShape[axis],
      input.shape[axis],
      'Concat axis sum',
    );
  }
  return cloneConcreteOutput(outputShape, dtype, concatQuantization(inputs, axis));
}
export function proveConcat(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalVariadicInputs(request.inputs, request.environment, 'input', 2);
  const rank = inputs[0].shape.length;
  if (rank < 1 || rank > 8 || inputs.some((input) => input.shape.length !== rank)) {
    fail('INVALID_RANK', 'operator inputs', 'must all have one equal rank in [1, 8].');
  }
  const dtype = inputs[0].dtype;
  if (inputs.some((input) => input.dtype !== dtype)) {
    fail('INVALID_DTYPE', 'operator inputs', 'must all have the same dtype.');
  }
  const axis = concatParams(paramsRecord(request.params), rank);
  const outputShape: ShapeDimensionSpec[] = [...inputs[0].shape];
  let axisSum = 0;
  let dynamicAxis: Readonly<{ readonly inputIndex: number; readonly symbol: string }> | undefined;
  for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
    const input = inputs[inputIndex];
    for (let dimension = 0; dimension < rank; dimension++) {
      if (dimension !== axis && !dimensionsProvablyEqual(
        input.shape[dimension],
        inputs[0].shape[dimension],
        request.environment,
      )) {
        fail(
          'SHAPE_MISMATCH',
          `operator input 'input${inputIndex}'.shape[${dimension}]`,
          'is not provably equal to input0 over the complete domain.',
        );
      }
    }
    const fixedAxis = fixedDimensionValue(input.shape[axis], request.environment);
    if (fixedAxis === undefined) {
      if (dynamicAxis !== undefined) {
        fail(
          'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
          `operator input 'input${inputIndex}'.shape[${axis}]`,
          `Concat has more than one dynamic axis term ('${dynamicAxis.symbol}' and ` +
          `'${String(input.shape[axis])}'); v1 admits only one dynamic term plus fixed extents.`,
        );
      }
      dynamicAxis = Object.freeze({
        inputIndex,
        symbol: input.shape[axis] as string,
      });
      continue;
    }
    axisSum = checkedShapeAdd(axisSum, fixedAxis, 'Concat axis maximum sum');
  }

  if (dynamicAxis !== undefined) {
    if (request.declaredOutputs === undefined) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `operator input 'input${dynamicAxis.inputIndex}'.shape[${axis}]`,
        'one dynamic Concat axis requires a declared output-only symbol with an exact affine domain.',
      );
    }
    const declared = normalizeLogicalDeclaredOutput(request.declaredOutputs, request.environment);
    if (declared.shape.length !== rank) {
      fail(
        'INVALID_RANK',
        "declared output 'out'.shape",
        `must have rank ${rank}, received rank ${declared.shape.length}.`,
      );
    }
    if (declared.dtype !== dtype) {
      fail(
        'INVALID_DTYPE',
        "declared output 'out'.dtype",
        `must match the common Concat input dtype '${dtype}'.`,
      );
    }
    for (let dimension = 0; dimension < rank; dimension++) {
      if (dimension === axis) continue;
      if (!dimensionsProvablyEqual(
        declared.shape[dimension],
        outputShape[dimension],
        request.environment,
      )) {
        fail(
          'SHAPE_MISMATCH',
          `declared output 'out'.shape[${dimension}]`,
          'is not provably equal to the Concat inputs over the complete domain.',
        );
      }
      outputShape[dimension] = declared.shape[dimension];
    }
    const target = declared.shape[axis];
    if (typeof target !== 'string') {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `declared output 'out'.shape[${axis}]`,
        'must be one output-only symbol for a dynamic Concat-axis sum.',
      );
    }
    if (inputs.some((input) => input.shape.some((dimension) => dimension === target))) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `declared output 'out'.shape[${axis}]`,
        `symbol '${target}' is already used by a Concat input and is not output-only.`,
      );
    }
    const sourceDomain = legalDimensionProgression(
      dynamicAxis.symbol,
      request.environment,
      `operator input 'input${dynamicAxis.inputIndex}'.shape[${axis}]`,
    );
    const targetDomain = legalDimensionProgression(
      target,
      request.environment,
      `declared output 'out'.shape[${axis}]`,
    );
    const expectedFirst = checkedShapeAdd(
      sourceDomain.first,
      axisSum,
      'Concat affine output minimum',
    );
    const expectedLast = checkedShapeAdd(
      sourceDomain.last,
      axisSum,
      'Concat affine output maximum',
    );
    if (targetDomain.first !== expectedFirst ||
        targetDomain.last !== expectedLast ||
        targetDomain.count !== sourceDomain.count) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `declared output 'out'.shape[${axis}]`,
        `symbol '${target}' has legal progression [${targetDomain.first}, ${targetDomain.last}] ` +
        `step ${targetDomain.step}; expected exactly '${dynamicAxis.symbol}+${axisSum}' ` +
        `as [${expectedFirst}, ${expectedLast}] with ${sourceDomain.count} legal values.`,
      );
    }
    outputShape[axis] = target;
    return domainInference(
      cloneLogicalOutput(
        Object.freeze(outputShape),
        dtype,
        request.environment,
        concatQuantization(inputs, axis),
      ),
      [`${target}=${dynamicAxis.symbol}+${axisSum} exactly over the complete bounded Concat domain`],
      [{ target, source: dynamicAxis.symbol, offset: axisSum }],
    );
  }

  outputShape[axis] = axisSum;
  return domainInference(
    cloneLogicalOutput(
      Object.freeze(outputShape),
      dtype,
      request.environment,
      concatQuantization(inputs, axis),
    ),
    [`${inputs.length} fixed Concat-axis extents sum to ${axisSum}`],
  );
}
interface SplitParameters {
  readonly axis: number;
  readonly sizes: readonly number[];
}
function splitParams(
  params: Readonly<Record<string, unknown>>,
  rank: number,
  axisExtent: number,
): SplitParameters {
  assertAllowedFields(params, ['axis', 'split', 'num_outputs'], 'operator params');
  const axis = normalizeAxis(params.axis, rank, 0);
  if (params.split !== undefined && params.num_outputs !== undefined) {
    fail('INVALID_PARAMS', 'operator params', 'must specify split or num_outputs, not both.');
  }
  let sizes: readonly number[];
  if (params.split !== undefined) {
    sizes = safeIntegerArray(params.split, 'operator params.split', { positive: true });
  } else {
    const count = params.num_outputs;
    if (!Number.isSafeInteger(count) || (count as number) <= 0) {
      fail('INVALID_PARAMS', 'operator params.num_outputs', 'must be a positive safe integer.');
    }
    if (axisExtent % (count as number) !== 0) {
      fail('SHAPE_MISMATCH', 'operator params.num_outputs', `must divide axis extent ${axisExtent}.`);
    }
    sizes = Object.freeze(new Array<number>(count as number).fill(axisExtent / (count as number)));
  }
  let sum = 0;
  for (const size of sizes) sum = checkedShapeAdd(sum, size, 'Split size sum');
  if (sum !== axisExtent) {
    fail('SHAPE_MISMATCH', 'operator params.split', `sums to ${sum}, expected axis extent ${axisExtent}.`);
  }
  return Object.freeze({ axis, sizes });
}
function splitQuantization(
  quantization: TensorQuantization | null | undefined,
  axis: number,
  sizes: readonly number[],
): readonly (TensorQuantization | undefined)[] {
  if (quantization == null || quantization.scheme === 'per_tensor' || quantization.axis !== axis) {
    return Object.freeze(sizes.map(() => quantization ?? undefined));
  }
  let offset = 0;
  return Object.freeze(sizes.map((size) => {
    const output = slicedPerAxis(quantization, axis, offset, offset + size);
    offset += size;
    return output;
  }));
}
export function inferSplit(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const preliminary = paramsRecord(request.params);
  const rawAxis = normalizeAxis(preliminary.axis, inputs.input.shape.length, 0);
  const params = splitParams(preliminary, inputs.input.shape.length, inputs.input.shape[rawAxis]);
  const quantizations = splitQuantization(inputs.input.quantization, params.axis, params.sizes);
  return cloneConcreteOutputs(Object.fromEntries(params.sizes.map((size, index) => {
    const shape = [...inputs.input.shape];
    shape[params.axis] = size;
    const quantization = quantizations[index];
    return [`out${index}`, quantization === undefined
      ? { shape, dtype: inputs.input.dtype }
      : { shape, dtype: inputs.input.dtype, quantization }];
  })));
}
export function proveSplit(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const preliminary = paramsRecord(request.params);
  const axis = normalizeAxis(preliminary.axis, inputs.input.shape.length, 0);
  const extent = fixedDimensionValue(inputs.input.shape[axis], request.environment);
  if (extent === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
      `operator input 'input'.shape[${axis}]`,
      'Split output extents require a fixed input axis in v1.',
    );
  }
  const params = splitParams(preliminary, inputs.input.shape.length, extent);
  const quantizations = splitQuantization(inputs.input.quantization, params.axis, params.sizes);
  const outputs = Object.fromEntries(params.sizes.map((size, index) => {
    const shape: ShapeDimensionSpec[] = [...inputs.input.shape];
    shape[params.axis] = size;
    const quantization = quantizations[index];
    return [`out${index}`, quantization === undefined
      ? { shape: Object.freeze(shape), dtype: inputs.input.dtype }
      : { shape: Object.freeze(shape), dtype: inputs.input.dtype, quantization }];
  }));
  return domainInference(
    cloneLogicalOutputs(outputs, request.environment),
    [`fixed axis ${params.axis} is partitioned into [${params.sizes.join(', ')}]`],
  );
}
interface SliceCoordinates {
  readonly starts: readonly number[];
  readonly ends: readonly number[];
  readonly steps: readonly number[];
}
interface ConcreteSlicePlan extends SliceCoordinates {
  readonly shape: readonly number[];
}
function sliceParameterArrays(
  params: Readonly<Record<string, unknown>>,
  rank: number,
): Readonly<{
  axes: readonly number[];
  starts: readonly number[];
  ends: readonly number[];
  steps: readonly number[];
}> {
  assertAllowedFields(params, ['starts', 'ends', 'axes', 'steps'], 'operator params');
  const starts = safeIntegerArray(params.starts, 'operator params.starts');
  const ends = safeIntegerArray(params.ends, 'operator params.ends');
  if (starts.length !== ends.length) {
    fail('INVALID_PARAMS', 'operator params', 'starts and ends must have equal lengths.');
  }
  const rawAxes = params.axes ?? starts.map((_value, index) => index);
  const axes = normalizeAxes(rawAxes, rank, 'operator params.axes', { sort: false });
  const steps = params.steps === undefined
    ? Object.freeze(starts.map(() => 1))
    : safeIntegerArray(params.steps, 'operator params.steps', { positive: true });
  if (axes.length !== starts.length || steps.length !== starts.length) {
    fail('INVALID_PARAMS', 'operator params', 'starts, ends, axes, and steps must have equal lengths.');
  }
  return Object.freeze({ axes, starts, ends, steps });
}
function concreteSlicePlan(
  shape: readonly number[],
  params: Readonly<Record<string, unknown>>,
): ConcreteSlicePlan {
  if (shape.length < 1 || shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank in [1, 8].');
  }
  const normalized = sliceParameterArrays(params, shape.length);
  const starts = new Array<number>(shape.length).fill(0);
  const ends = [...shape];
  const steps = new Array<number>(shape.length).fill(1);
  const output = [...shape];
  for (let index = 0; index < normalized.axes.length; index++) {
    const axis = normalized.axes[index];
    const extent = shape[axis];
    const rawStart = normalized.starts[index];
    const rawEnd = normalized.ends[index];
    const start = Math.min(extent, Math.max(0, rawStart < 0 ? rawStart + extent : rawStart));
    const end = Math.min(extent, Math.max(0, rawEnd < 0 ? rawEnd + extent : rawEnd));
    const step = normalized.steps[index];
    if (end <= start) {
      fail('SHAPE_MISMATCH', `operator params.ends[${index}]`, 'selects an empty axis, which v1 forbids.');
    }
    const length = Math.floor((checkedShapeSubtract(end, start, 'Slice extent') - 1) / step) + 1;
    starts[axis] = start;
    ends[axis] = end;
    steps[axis] = step;
    output[axis] = length;
  }
  checkedShapeElementCount(output, 'Slice output shape');
  return Object.freeze({
    shape: Object.freeze(output),
    starts: Object.freeze(starts),
    ends: Object.freeze(ends),
    steps: Object.freeze(steps),
  });
}
function logicalSlicePlan(
  shape: TensorShapeSpec,
  params: Readonly<Record<string, unknown>>,
  environment: ShapeEnvironment,
): Readonly<{ shape: TensorShapeSpec; coordinates: SliceCoordinates }> {
  if (shape.length < 1 || shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank in [1, 8].');
  }
  const normalized = sliceParameterArrays(params, shape.length);
  const output: ShapeDimensionSpec[] = [...shape];
  if (shape.every((dimension) => fixedDimensionValue(dimension, environment) !== undefined)) {
    const fixedPlan = concreteSlicePlan(
      shape.map((dimension) => fixedDimensionValue(dimension, environment)!),
      params,
    );
    return Object.freeze({
      shape: createTensorShapeSpec(fixedPlan.shape, environment),
      coordinates: fixedPlan,
    });
  }
  const starts = new Array<number>(shape.length).fill(0);
  const ends = shape.map((dimension) => logicalDimensionMaximum(dimension, environment));
  const steps = new Array<number>(shape.length).fill(1);
  for (let index = 0; index < normalized.axes.length; index++) {
    const axis = normalized.axes[index];
    const dimension = shape[axis];
    const fixed = fixedDimensionValue(dimension, environment);
    if (fixed !== undefined) {
      const axisPlan = concreteSlicePlan([fixed], {
        starts: [normalized.starts[index]],
        ends: [normalized.ends[index]],
        axes: [0],
        steps: [normalized.steps[index]],
      });
      output[axis] = axisPlan.shape[0];
      starts[axis] = axisPlan.starts[0];
      ends[axis] = axisPlan.ends[0];
      steps[axis] = axisPlan.steps[0];
      continue;
    }
    const constraint = environment.get(dimension as string)!;
    if (normalized.starts[index] !== 0 || normalized.steps[index] !== 1 ||
        normalized.ends[index] < constraint.max) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `operator input 'input'.shape[${axis}]`,
        'a dynamic Slice axis must use the full [0, extent) identity selection.',
      );
    }
    output[axis] = dimension;
    starts[axis] = 0;
    ends[axis] = constraint.max;
    steps[axis] = 1;
  }
  return Object.freeze({
    shape: createTensorShapeSpec(output, environment),
    coordinates: Object.freeze({
      starts: Object.freeze(starts),
      ends: Object.freeze(ends),
      steps: Object.freeze(steps),
    }),
  });
}
function sliceQuantization(
  quantization: TensorQuantization | null | undefined,
  plan: SliceCoordinates,
): TensorQuantization | undefined {
  if (quantization == null || quantization.scheme === 'per_tensor') return quantization ?? undefined;
  return slicedPerAxis(
    quantization,
    quantization.axis,
    plan.starts[quantization.axis],
    plan.ends[quantization.axis],
    plan.steps[quantization.axis],
  );
}
export function inferSlice(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const plan = concreteSlicePlan(inputs.input.shape, paramsRecord(request.params));
  return cloneConcreteOutput(
    plan.shape,
    inputs.input.dtype,
    sliceQuantization(inputs.input.quantization, plan),
  );
}
export function proveSlice(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  const plan = logicalSlicePlan(inputs.input.shape, paramsRecord(request.params), request.environment);
  return domainInference(
    cloneLogicalOutput(
      plan.shape,
      inputs.input.dtype,
      request.environment,
      sliceQuantization(inputs.input.quantization, plan.coordinates),
    ),
    [inputs.input.shape.some((dimension) => typeof dimension === 'string')
      ? 'dynamic axes use full identity selections; fixed axes use exact positive-step slices'
      : 'all slice extents are fixed and checked with positive-step arithmetic'],
  );
}
function padParams(
  params: Readonly<Record<string, unknown>>,
  rank: number,
): Readonly<{ before: readonly number[]; after: readonly number[] }> {
  assertAllowedFields(params, ['pads', 'value'], 'operator params');
  const pads = safeIntegerArray(params.pads, 'operator params.pads', { nonNegative: true });
  if (pads.length !== rank * 2) {
    fail('INVALID_PARAMS', 'operator params.pads', `must have exactly ${rank * 2} entries.`);
  }
  const value = params.value ?? 0;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    fail('INVALID_PARAMS', 'operator params.value', 'must be finite.');
  }
  return Object.freeze({
    before: Object.freeze(pads.slice(0, rank)),
    after: Object.freeze(pads.slice(rank)),
  });
}
function padQuantization(
  quantization: TensorQuantization | null | undefined,
  before: readonly number[],
  after: readonly number[],
): TensorQuantization | undefined {
  if (quantization == null || quantization.scheme === 'per_tensor') return quantization ?? undefined;
  if (before[quantization.axis] !== 0 || after[quantization.axis] !== 0) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      `operator params.pads[${quantization.axis}]`,
      'must not extend a per-axis quantization dimension.',
    );
  }
  return quantization;
}
export function inferPad(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = padParams(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = inputs.input.shape.map((dimension, axis) =>
    checkedShapeAdd(
      checkedShapeAdd(dimension, params.before[axis], `Pad axis ${axis}`),
      params.after[axis],
      `Pad axis ${axis}`,
    ));
  return cloneConcreteOutput(
    outputShape,
    inputs.input.dtype,
    padQuantization(inputs.input.quantization, params.before, params.after),
  );
}
export function provePad(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = padParams(paramsRecord(request.params), inputs.input.shape.length);
  const outputShape = inputs.input.shape.map((dimension, axis) => {
    const amount = checkedShapeAdd(params.before[axis], params.after[axis], `Pad axis ${axis}`);
    if (amount === 0) return dimension;
    const fixed = fixedDimensionValue(dimension, request.environment);
    if (fixed === undefined) {
      fail(
        'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
        `operator input 'input'.shape[${axis}]`,
        'dynamic extent plus nonzero padding is not representable by one v1 output dimension.',
      );
    }
    return checkedShapeAdd(fixed, amount, `Pad axis ${axis}`);
  });
  return domainInference(
    cloneLogicalOutput(
      Object.freeze(outputShape),
      inputs.input.dtype,
      request.environment,
      padQuantization(inputs.input.quantization, params.before, params.after),
    ),
    ['nonzero padding is confined to fixed axes; dynamic axes are preserved'],
  );
}
function assertIndices(
  descriptor: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
): void {
  if (descriptor.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'indices'.dtype", 'must be int32.');
  }
  assertUnquantized(descriptor, "operator input 'indices'");
}
export function inferGather(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'indices'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertIndices(inputs.indices);
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['axis'], 'operator params');
  const axis = normalizeAxis(params.axis, inputs.input.shape.length, 0);
  const outputShape = Object.freeze([
    ...inputs.input.shape.slice(0, axis),
    ...inputs.indices.shape,
    ...inputs.input.shape.slice(axis + 1),
  ]);
  const quantization = inputs.input.quantization?.scheme === 'per_axis'
    ? inputs.input.quantization.axis === axis
      ? fail(
        'UNSAFE_QUANTIZATION_TRANSFORM',
        "operator input 'input'.quantization.axis",
        'Gather indices would reorder the per-axis affine metadata.',
      )
      : remapPerAxis(
        inputs.input.quantization,
        inputs.input.quantization.axis < axis
          ? inputs.input.quantization.axis
          : inputs.input.quantization.axis + inputs.indices.shape.length - 1,
      )
    : inputs.input.quantization ?? undefined;
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}
export function proveGather(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'indices'],
    [],
  );
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertIndices(inputs.indices);
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['axis'], 'operator params');
  const axis = normalizeAxis(params.axis, inputs.input.shape.length, 0);
  const outputShape = Object.freeze([
    ...inputs.input.shape.slice(0, axis),
    ...inputs.indices.shape,
    ...inputs.input.shape.slice(axis + 1),
  ]);
  const quantization = inputs.input.quantization?.scheme === 'per_axis'
    ? inputs.input.quantization.axis === axis
      ? fail(
        'UNSAFE_QUANTIZATION_TRANSFORM',
        "operator input 'input'.quantization.axis",
        'Gather indices would reorder the per-axis affine metadata.',
      )
      : remapPerAxis(
        inputs.input.quantization,
        inputs.input.quantization.axis < axis
          ? inputs.input.quantization.axis
          : inputs.input.quantization.axis + inputs.indices.shape.length - 1,
      )
    : inputs.input.quantization ?? undefined;
  return domainInference(
    cloneLogicalOutput(outputShape, inputs.input.dtype, request.environment, quantization),
    ['Gather replaces one data axis with the fixed-rank indices shape'],
  );
}
function logicalDimensionMinimum(
  dimension: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number {
  return typeof dimension === 'number' ? dimension : environment.get(dimension)!.min;
}
function logicalDimensionMaximum(
  dimension: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number {
  return typeof dimension === 'number' ? dimension : environment.get(dimension)!.max;
}
function gatherElementsQuantization(
  quantization: TensorQuantization | null | undefined,
  outputShape: readonly (number | string)[],
  axis: number,
  environment?: ShapeEnvironment,
): TensorQuantization | undefined {
  if (quantization == null || quantization.scheme === 'per_tensor') return quantization ?? undefined;
  if (quantization.axis === axis) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      "operator input 'input'.quantization.axis",
      'GatherElements indices would reorder the per-axis affine metadata.',
    );
  }
  const extent = environment === undefined
    ? outputShape[quantization.axis] as number
    : fixedDimensionValue(outputShape[quantization.axis], environment);
  if (extent === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT',
      `operator input 'indices'.shape[${quantization.axis}]`,
      'must be fixed to preserve per-axis affine metadata.',
    );
  }
  if (extent > quantization.scales.length) {
    fail('SHAPE_MISMATCH', 'operator input indices', 'exceeds the per-axis input extent.');
  }
  return slicedPerAxis(quantization, quantization.axis, 0, extent);
}
export function inferGatherElements(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input', 'indices'], []);
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertIndices(inputs.indices);
  if (inputs.indices.shape.length !== inputs.input.shape.length) {
    fail('INVALID_RANK', "operator input 'indices'.shape", 'must have the same rank as input.');
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['axis'], 'operator params');
  const axis = normalizeAxis(params.axis, inputs.input.shape.length, 0);
  for (let dimension = 0; dimension < inputs.input.shape.length; dimension++) {
    if (dimension !== axis && inputs.indices.shape[dimension] > inputs.input.shape[dimension]) {
      fail(
        'SHAPE_MISMATCH',
        `operator input 'indices'.shape[${dimension}]`,
        'must not exceed the corresponding input dimension.',
      );
    }
  }
  return cloneConcreteOutput(
    inputs.indices.shape,
    inputs.input.dtype,
    gatherElementsQuantization(inputs.input.quantization, inputs.indices.shape, axis),
  );
}
export function proveGatherElements(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input', 'indices'],
    [],
  );
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  assertIndices(inputs.indices);
  if (inputs.indices.shape.length !== inputs.input.shape.length) {
    fail('INVALID_RANK', "operator input 'indices'.shape", 'must have the same rank as input.');
  }
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['axis'], 'operator params');
  const axis = normalizeAxis(params.axis, inputs.input.shape.length, 0);
  for (let dimension = 0; dimension < inputs.input.shape.length; dimension++) {
    if (dimension !== axis &&
        !dimensionsProvablyEqual(
          inputs.indices.shape[dimension],
          inputs.input.shape[dimension],
          request.environment,
        ) &&
        logicalDimensionMaximum(inputs.indices.shape[dimension], request.environment) >
          logicalDimensionMinimum(inputs.input.shape[dimension], request.environment)) {
      fail(
        'INVALID_DOMAIN',
        `operator input 'indices'.shape[${dimension}]`,
        'can exceed the corresponding input dimension within the declared domain.',
      );
    }
  }
  return domainInference(
    cloneLogicalOutput(
      inputs.indices.shape,
      inputs.input.dtype,
      request.environment,
      gatherElementsQuantization(
        inputs.input.quantization,
        inputs.indices.shape,
        axis,
        request.environment,
      ),
    ),
    ['every non-gather indices extent is bounded by the minimum matching data extent'],
  );
}
