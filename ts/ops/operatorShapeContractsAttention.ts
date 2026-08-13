/* Shape contracts for SDPA, CrossSDPA and RoPE.
 *
 * Mirrors shape_contract_attention.inc on the native side and
 * tests/operator_shape_contract_attention_vectors.json.
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


export function inferSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['qkv'], ['mask']);
  const qkv = inputs.qkv;
  assertFloatTensor(qkv, "operator input 'qkv'");
  assertAttentionRank(qkv.shape, "operator input 'qkv'.shape");
  const rank = qkv.shape.length;
  const featureAxis = rank - 1;
  const packedFeature = qkv.shape[featureAxis];
  if (packedFeature % 3 !== 0) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'qkv'.shape[${featureAxis}]`,
      'must be exactly 3 times the output feature extent.',
    );
  }
  const feature = packedFeature / 3;
  const { heads } = attentionParameters(paramsRecord(request.params));
  if (feature % heads !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.heads', 'must divide the output feature extent.');
  }
  const sequence = qkv.shape[rank - 2];
  const batch = rank === 2 ? 1 : qkv.shape[0];
  assertConcreteAttentionMask(inputs.mask, batch, sequence, sequence);
  return cloneConcreteOutput([...qkv.shape.slice(0, featureAxis), feature], 'float32');
}
export function proveSDPA(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['qkv'],
    ['mask'],
  );
  const qkv = inputs.qkv;
  assertFloatTensor(qkv, "operator input 'qkv'");
  assertAttentionRank(qkv.shape, "operator input 'qkv'.shape");
  const rank = qkv.shape.length;
  const featureAxis = rank - 1;
  const packedFeature = fixedDimensionValue(qkv.shape[featureAxis], request.environment);
  if (packedFeature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE',
      `operator input 'qkv'.shape[${featureAxis}]`,
      'packed QKV width must be fixed over the complete v1 domain.',
    );
  }
  if (packedFeature % 3 !== 0) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'qkv'.shape[${featureAxis}]`,
      'must be exactly 3 times the output feature extent.',
    );
  }
  const feature = packedFeature / 3;
  const { heads } = attentionParameters(paramsRecord(request.params));
  if (feature % heads !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.heads', 'must divide the output feature extent.');
  }
  const sequence = qkv.shape[rank - 2];
  const batch = rank === 2 ? 1 : qkv.shape[0];
  assertLogicalAttentionMask(inputs.mask, batch, sequence, sequence, request.environment);
  return domainInference(
    cloneLogicalOutput([...qkv.shape.slice(0, featureAxis), feature], 'float32', request.environment),
    [
      `packed QKV width ${packedFeature} yields fixed feature width ${feature} divisible by ${heads} heads`,
      'mask geometry is one closed K/BK/QK/BQK keep-mask layout for every legal binding',
    ],
  );
}
export function inferCrossSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['q', 'k', 'v'], ['mask']);
  const query = inputs.q;
  const key = inputs.k;
  const value = inputs.v;
  assertFloatTensor(query, "operator input 'q'");
  assertFloatTensor(key, "operator input 'k'");
  assertFloatTensor(value, "operator input 'v'");
  assertAttentionRank(query.shape, "operator input 'q'.shape");
  assertAttentionRank(key.shape, "operator input 'k'.shape");
  assertAttentionRank(value.shape, "operator input 'v'.shape");
  const rank = query.shape.length;
  if (key.shape.length !== rank || value.shape.length !== rank) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v must have the same rank.');
  }
  const sequenceAxis = rank - 2;
  const featureAxis = rank - 1;
  if (rank === 3 && (key.shape[0] !== query.shape[0] || value.shape[0] !== query.shape[0])) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v batch extents must match.');
  }
  if (value.shape[sequenceAxis] !== key.shape[sequenceAxis]) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'v'.shape[${sequenceAxis}]`,
      'must equal the key sequence extent.',
    );
  }
  if (key.shape[featureAxis] !== query.shape[featureAxis] ||
      value.shape[featureAxis] !== query.shape[featureAxis]) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v feature extents must match.');
  }
  const feature = query.shape[featureAxis];
  const { heads } = attentionParameters(paramsRecord(request.params));
  if (feature % heads !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.heads', 'must divide the feature extent.');
  }
  const batch = rank === 2 ? 1 : query.shape[0];
  assertConcreteAttentionMask(
    inputs.mask,
    batch,
    query.shape[sequenceAxis],
    key.shape[sequenceAxis],
  );
  return cloneConcreteOutput(query.shape, 'float32');
}
export function proveCrossSDPA(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['q', 'k', 'v'],
    ['mask'],
  );
  const query = inputs.q;
  const key = inputs.k;
  const value = inputs.v;
  assertFloatTensor(query, "operator input 'q'");
  assertFloatTensor(key, "operator input 'k'");
  assertFloatTensor(value, "operator input 'v'");
  assertAttentionRank(query.shape, "operator input 'q'.shape");
  assertAttentionRank(key.shape, "operator input 'k'.shape");
  assertAttentionRank(value.shape, "operator input 'v'.shape");
  const rank = query.shape.length;
  if (key.shape.length !== rank || value.shape.length !== rank) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v must have the same rank.');
  }
  const sequenceAxis = rank - 2;
  const featureAxis = rank - 1;
  if (rank === 3 &&
      (!dimensionsProvablyEqual(query.shape[0], key.shape[0], request.environment) ||
       !dimensionsProvablyEqual(query.shape[0], value.shape[0], request.environment))) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v batch extents are not provably equal.');
  }
  if (!dimensionsProvablyEqual(key.shape[sequenceAxis], value.shape[sequenceAxis], request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      `operator input 'v'.shape[${sequenceAxis}]`,
      'is not provably equal to the key sequence extent.',
    );
  }
  if (!dimensionsProvablyEqual(query.shape[featureAxis], key.shape[featureAxis], request.environment) ||
      !dimensionsProvablyEqual(query.shape[featureAxis], value.shape[featureAxis], request.environment)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'q, k, and v feature extents are not provably equal.');
  }
  const feature = fixedDimensionValue(query.shape[featureAxis], request.environment);
  if (feature === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_FEATURE',
      `operator input 'q'.shape[${featureAxis}]`,
      'attention feature width must be fixed over the complete v1 domain.',
    );
  }
  const { heads } = attentionParameters(paramsRecord(request.params));
  if (feature % heads !== 0) {
    fail('SHAPE_MISMATCH', 'operator params.heads', 'must divide the feature extent.');
  }
  const batch = rank === 2 ? 1 : query.shape[0];
  assertLogicalAttentionMask(
    inputs.mask,
    batch,
    query.shape[sequenceAxis],
    key.shape[sequenceAxis],
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(query.shape, 'float32', request.environment),
    [
      `fixed feature width ${feature} is divisible by ${heads} heads`,
      'batch and K/V sequence equalities plus mask geometry hold for every legal binding',
    ],
  );
}
interface RoPEParameters {
  readonly rotaryDimension?: number;
  readonly positionOffset: number;
}
function ropeParameters(params: Readonly<Record<string, unknown>>): RoPEParameters {
  assertAllowedFields(
    params,
    ['rotary_dim', 'theta', 'position_offset', 'interleaved'],
    'operator params',
  );
  let rotaryDimension: number | undefined;
  if (params.rotary_dim !== undefined) {
    if (!Number.isSafeInteger(params.rotary_dim) || (params.rotary_dim as number) <= 0 ||
        (params.rotary_dim as number) % 2 !== 0) {
      fail(
        'INVALID_PARAMS',
        'operator params.rotary_dim',
        'must be a positive even safe integer.',
      );
    }
    rotaryDimension = params.rotary_dim as number;
  }
  const theta = params.theta ?? 10000;
  if (typeof theta !== 'number' || !Number.isFinite(theta) || theta <= 0 ||
      !Number.isFinite(Math.fround(theta)) || Math.fround(theta) <= 0) {
    fail(
      'INVALID_PARAMS',
      'operator params.theta',
      'must be positive and representable as float32.',
    );
  }
  const positionOffset = params.position_offset ?? 0;
  if (!Number.isSafeInteger(positionOffset) || (positionOffset as number) < 0 ||
      (positionOffset as number) > 0x7fffffff) {
    fail(
      'INVALID_PARAMS',
      'operator params.position_offset',
      'must be a non-negative int32 integer.',
    );
  }
  const interleaved = params.interleaved ?? false;
  if (typeof interleaved !== 'boolean') {
    fail('INVALID_PARAMS', 'operator params.interleaved', 'must be boolean.');
  }
  return Object.freeze({
    ...(rotaryDimension === undefined ? {} : { rotaryDimension }),
    positionOffset: positionOffset as number,
  });
}
function assertConcretePositionIds(
  positions: OperatorTensorDescriptor | undefined,
  rank: number,
  batch: number,
  sequence: number,
): void {
  if (positions === undefined) return;
  if (positions.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'position_ids'.dtype", "must be 'int32'.");
  }
  assertUnquantized(positions, "operator input 'position_ids'");
  if ((positions.shape.length === 1 && positions.shape[0] === sequence) ||
      (rank === 3 && positions.shape.length === 2 &&
        positions.shape[0] === batch && positions.shape[1] === sequence)) {
    return;
  }
  fail(
    'SHAPE_MISMATCH',
    "operator input 'position_ids'.shape",
    'must have shape [S], or [B,S] for a rank-3 input.',
  );
}
function assertLogicalPositionIds(
  positions: LogicalOperatorTensorDescriptor | undefined,
  rank: number,
  batch: ShapeDimensionSpec,
  sequence: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): void {
  if (positions === undefined) return;
  if (positions.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'position_ids'.dtype", "must be 'int32'.");
  }
  assertUnquantized(positions, "operator input 'position_ids'");
  const shape = positions.shape;
  if ((shape.length === 1 && dimensionsProvablyEqual(shape[0], sequence, environment)) ||
      (rank === 3 && shape.length === 2 &&
        dimensionsProvablyEqual(shape[0], batch, environment) &&
        dimensionsProvablyEqual(shape[1], sequence, environment))) {
    return;
  }
  fail(
    'SHAPE_MISMATCH',
    "operator input 'position_ids'.shape",
    'is not provably [S], or [B,S] for a rank-3 input, over the complete domain.',
  );
}
export function inferRoPE(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], ['position_ids']);
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertAttentionRank(input.shape, "operator input 'input'.shape");
  const rank = input.shape.length;
  const sequence = input.shape[rank - 2];
  const width = input.shape[rank - 1];
  const { rotaryDimension: explicitRotaryDimension, positionOffset } =
    ropeParameters(paramsRecord(request.params));
  const rotaryDimension = explicitRotaryDimension ?? width;
  if (rotaryDimension % 2 !== 0) {
    fail('INVALID_PARAMS', 'operator params.rotary_dim', 'resolved rotary width must be even.');
  }
  if (rotaryDimension > width) {
    fail('SHAPE_MISMATCH', 'operator params.rotary_dim', 'must not exceed the feature extent.');
  }
  if (sequence - 1 > 0x7fffffff - positionOffset) {
    fail(
      'INVALID_PARAMS',
      'operator params.position_offset',
      'plus the maximum sequence index must fit int32.',
    );
  }
  const batch = rank === 2 ? 1 : input.shape[0];
  assertConcretePositionIds(inputs.position_ids, rank, batch, sequence);
  return cloneConcreteOutput(input.shape, 'float32');
}
export function proveRoPE(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(
    request.inputs,
    request.environment,
    ['input'],
    ['position_ids'],
  );
  const input = inputs.input;
  assertFloatTensor(input, "operator input 'input'");
  assertAttentionRank(input.shape, "operator input 'input'.shape");
  const rank = input.shape.length;
  const sequence = input.shape[rank - 2];
  const width = input.shape[rank - 1];
  const { rotaryDimension: explicitRotaryDimension, positionOffset } =
    ropeParameters(paramsRecord(request.params));
  const fixedWidth = fixedDimensionValue(width, request.environment);
  if (explicitRotaryDimension === undefined) {
    if (fixedWidth !== undefined) {
      if (fixedWidth % 2 !== 0) {
        fail('INVALID_PARAMS', 'operator params.rotary_dim', 'resolved rotary width must be even.');
      }
    } else {
      const constraint = request.environment.get(width as string)!;
      if (constraint.multiple_of === undefined || constraint.multiple_of % 2 !== 0) {
        fail(
          'UNPROVABLE_DYNAMIC_FEATURE',
          `operator input 'input'.shape[${rank - 1}]`,
          'an implicit dynamic rotary width requires an even multiple_of constraint.',
        );
      }
    }
  } else {
    const minimumWidth = fixedWidth ?? request.environment.get(width as string)!.min;
    if (explicitRotaryDimension > minimumWidth) {
      fail(
        'SHAPE_MISMATCH',
        'operator params.rotary_dim',
        'must not exceed the minimum feature extent over the complete domain.',
      );
    }
  }
  const maximumSequence = typeof sequence === 'number'
    ? sequence
    : request.environment.get(sequence)!.max;
  if (maximumSequence - 1 > 0x7fffffff - positionOffset) {
    fail(
      'INVALID_PARAMS',
      'operator params.position_offset',
      'plus the maximum sequence index must fit int32 over the complete domain.',
    );
  }
  const batch = rank === 2 ? 1 : input.shape[0];
  assertLogicalPositionIds(
    inputs.position_ids,
    rank,
    batch,
    sequence,
    request.environment,
  );
  return domainInference(
    cloneLogicalOutput(input.shape, 'float32', request.environment),
    [
      'rank, rotary width, and position geometry are valid for every legal binding',
      'output shape and dtype equal the unquantized float32 input',
    ],
  );
}
