import type { RuntimeDType, TensorQuantization } from '../types.js';
import {
  operatorShapeContracts as generatedOperatorShapeContracts,
  operatorShapeFunctionIds as generatedOperatorShapeFunctionIds,
} from '../generated/kernelRegistry.js';
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

/* Re-exported so the module keeps the public surface it had before the
 * shared vocabulary moved into operatorShapeContractsCommon.ts. */
export {
  ATTENTION_SHAPE_FUNCTION_IDS,
  DIRECT_SHAPE_FUNCTION_IDS,
  EXTENDED_SHAPE_FUNCTION_IDS,
  OperatorShapeContractError,
  QUANTIZED_SHAPE_FUNCTION_IDS,
  SPATIAL_SHAPE_FUNCTION_IDS,
  STRUCTURAL_SHAPE_FUNCTION_IDS,
} from './operatorShapeContractsCommon.js';
export type {
  AcceptedOperatorDomainProof,
  AttentionShapeFunctionId,
  ConcreteOperatorOutputs,
  ConcreteOperatorShapeRequest,
  DirectShapeFunctionId,
  LogicalOperatorOutputs,
  LogicalOperatorTensorDescriptor,
  OperatorAffineDimensionRelation,
  OperatorDomainProof,
  OperatorDomainProofRequest,
  OperatorShapeContract,
  OperatorShapeContractErrorCode,
  OperatorShapeFunctionId,
  OperatorShapePorts,
  OperatorTensorDescriptor,
  QuantizedShapeFunctionId,
  RejectedOperatorDomainProof,
  SpatialShapeFunctionId,
  StructuralShapeFunctionId,
} from './operatorShapeContractsCommon.js';
import {
  inferQActivation,
  inferQAdd,
  inferQArgMax,
  inferQBatchMatMul,
  inferQConv2D,
  inferQDense,
  inferQEmbedding,
  inferQGroupNorm,
  inferQLayerNorm,
  inferQMaskedMean,
  inferQSDPA,
  proveQActivation,
  proveQAdd,
  proveQArgMax,
  proveQBatchMatMul,
  proveQConv2D,
  proveQDense,
  proveQEmbedding,
  proveQGroupNorm,
  proveQLayerNorm,
  proveQMaskedMean,
  proveQSDPA,
} from './operatorShapeContractsQuantized.js';
import {
  inferCrossSDPA,
  inferRoPE,
  inferSDPA,
  proveCrossSDPA,
  proveRoPE,
  proveSDPA,
} from './operatorShapeContractsAttention.js';
import {
  inferBatchMatMul,
  inferConv1D,
  inferConv2D,
  inferConvTranspose2D,
  inferGlobalAveragePool,
  inferPool2D,
  inferResize,
  inferUpsampleNearest2D,
  proveBatchMatMul,
  proveConv1D,
  proveConv2D,
  proveConvTranspose2D,
  proveGlobalAveragePool,
  provePool2D,
  proveResize,
  proveUpsampleNearest2D,
} from './operatorShapeContractsSpatial.js';
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
import {
  exactBinaryParams,
  inferActivation,
  inferCast,
  inferDense,
  inferDequantizeLinear,
  inferEmbedding,
  inferFeatureNorm,
  inferGroupNorm,
  inferIdentity,
  inferPReLU,
  inferQuantizeLinear,
  proveActivation,
  proveCast,
  proveDense,
  proveDequantizeLinear,
  proveEmbedding,
  proveFeatureNorm,
  proveGroupNorm,
  proveIdentity,
  provePReLU,
  proveQuantizeLinear,
} from './operatorShapeContractsDirect.js';
import {
  inferBatchNorm2D,
  inferConcat2,
  inferCrossAttention,
  inferDropout,
  inferInterpolate1D,
  inferLogicalNot,
  inferMask,
  inferMoELinear,
  inferMoERouter,
  inferRequantizeLinear,
  inferScan,
  inferVisionProfile,
  proveBatchNorm2D,
  proveConcat2,
  proveCrossAttention,
  proveDropout,
  proveInterpolate1D,
  proveLogicalNot,
  proveMask,
  proveMoELinear,
  proveMoERouter,
  proveRequantizeLinear,
  proveScan,
  proveVisionProfile,
} from './operatorShapeContractsExtended.js';

/* Operators are grouped by how their shape contract computes an output shape,
 * not by when the contract was written:
 *
 *   DIRECT      no algebra - the output is the input, or one axis is replaced
 *   SPATIAL     window and stride arithmetic
 *   STRUCTURAL  broadcasting and index arithmetic
 *   ATTENTION   head and sequence arithmetic
 *   QUANTIZED   the W8A8 byte-contract forms of the above
 *   EXTENDED    operators outside the core families
 *
 * The same six names key the native fragments in native/src/runtime/ and the
 * shared vector corpora in tests/, so a family can be compared across all
 * three implementations file by file.
 */







































































































function inferExactBinary(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  const a = inputs.a;
  const b = inputs.b;
  assertFloatTensor(a, "operator input 'a'");
  assertFloatTensor(b, "operator input 'b'");
  exactBinaryParams(operator, paramsRecord(request.params));
  if (!sameConcreteShape(a.shape, b.shape)) {
    fail('SHAPE_MISMATCH', 'operator inputs', 'a and b must have exactly equal shapes; broadcasting is explicit.');
  }
  return cloneConcreteOutput(a.shape, 'float32');
}

function proveExactBinary(
  operator: string,
  request: OperatorDomainProofRequest,
): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['a', 'b'], []);
  const a = inputs.a;
  const b = inputs.b;
  assertFloatTensor(a, "operator input 'a'");
  assertFloatTensor(b, "operator input 'b'");
  exactBinaryParams(operator, paramsRecord(request.params));
  if (!logicalShapesProvablyEqual(a.shape, b.shape, request.environment)) {
    fail(
      'SHAPE_MISMATCH',
      'operator inputs',
      'a and b are not provably exact-shape equal over the complete bounded domain.',
    );
  }
  return domainInference(
    cloneLogicalOutput(a.shape, 'float32', request.environment),
    ['both input shape specifications are equal for every legal binding'],
  );
}








































































































































































































type ConcreteImplementation = (
  operator: string,
  request: ConcreteOperatorShapeRequest,
) => ConcreteOperatorOutputs;
type DomainImplementation = (
  operator: string,
  request: OperatorDomainProofRequest,
) => DomainInference;

interface ShapeDefinition {
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly operators: readonly string[];
  readonly ports: OperatorShapePorts;
  readonly inferConcrete: ConcreteImplementation;
  readonly proveDomain: DomainImplementation;
}

function ports(
  requiredInputs: readonly string[],
  optionalInputs: readonly string[] = [],
  outputs: readonly string[] = ['out'],
  variadicInputs?: Readonly<{ readonly prefix: string; readonly minimum: number }>,
  variadicOutputs?: Readonly<{ readonly prefix: string; readonly minimum: number }>,
): OperatorShapePorts {
  return Object.freeze({
    requiredInputs: Object.freeze([...requiredInputs]),
    optionalInputs: Object.freeze([...optionalInputs]),
    outputs: Object.freeze([...outputs]),
    ...(variadicInputs === undefined
      ? {}
      : { variadicInputs: Object.freeze({ ...variadicInputs }) }),
    ...(variadicOutputs === undefined
      ? {}
      : { variadicOutputs: Object.freeze({ ...variadicOutputs }) }),
  });
}

const DEFINITIONS: readonly ShapeDefinition[] = Object.freeze([
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.moeRouter,
    operators: Object.freeze(['MoERouter']),
    ports: ports(['input', 'weight'], ['bias'], ['indices', 'weights']),
    inferConcrete: (_operator, request) => inferMoERouter(request),
    proveDomain: (_operator, request) => proveMoERouter(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.moeLinear,
    operators: Object.freeze(['MoELinear']),
    ports: ports(
      ['input', 'expert_weight', 'route_indices', 'route_weights'], ['expert_bias'],
    ),
    inferConcrete: (_operator, request) => inferMoELinear(request),
    proveDomain: (_operator, request) => proveMoELinear(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.crossAttention,
    operators: Object.freeze(['CrossAttention']),
    ports: ports(['q', 'kv', 'weight'], ['scale', 'bias']),
    inferConcrete: (_operator, request) => inferCrossAttention(request),
    proveDomain: (_operator, request) => proveCrossAttention(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.batchNorm2D,
    operators: Object.freeze(['BatchNorm2D']),
    ports: ports(['input', 'weight', 'bias', 'running_mean', 'running_var']),
    inferConcrete: (_operator, request) => inferBatchNorm2D(request),
    proveDomain: (_operator, request) => proveBatchNorm2D(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.interpolate1D,
    operators: Object.freeze(['Interpolate1D']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferInterpolate1D(request),
    proveDomain: (_operator, request) => proveInterpolate1D(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.logicalNot,
    operators: Object.freeze(['Not']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferLogicalNot(request),
    proveDomain: (_operator, request) => proveLogicalNot(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.mask,
    operators: Object.freeze(['Mask']),
    ports: ports(['mask', 'a', 'b']),
    inferConcrete: (_operator, request) => inferMask(request),
    proveDomain: (_operator, request) => proveMask(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.broadcast,
    operators: Object.freeze(['Broadcast']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferExpand(request),
    proveDomain: (_operator, request) => proveExpand(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.concat2,
    operators: Object.freeze(['Concat2']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferConcat2(request),
    proveDomain: (_operator, request) => proveConcat2(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.requantizeLinear,
    operators: Object.freeze(['RequantizeLinear']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferRequantizeLinear(request),
    proveDomain: (_operator, request) => proveRequantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.ssmScan,
    operators: Object.freeze(['SSMScan']),
    ports: ports(['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'], ['out', 'state']),
    inferConcrete: (_operator, request) => inferScan(request),
    proveDomain: (_operator, request) => proveScan(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.selectiveScan,
    operators: Object.freeze(['SelectiveScan']),
    ports: ports(['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'], ['out', 'state']),
    inferConcrete: (_operator, request) => inferScan(request),
    proveDomain: (_operator, request) => proveScan(request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.spatialSoftargmaxY,
    operators: Object.freeze(['SpatialSoftargmaxY']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('softargmax', request),
    proveDomain: (_operator, request) => proveVisionProfile('softargmax', request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.meanHeight,
    operators: Object.freeze(['MeanHeight']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('mean', request),
    proveDomain: (_operator, request) => proveVisionProfile('mean', request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.profileX,
    operators: Object.freeze(['ProfileX']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('profile-x', request),
    proveDomain: (_operator, request) => proveVisionProfile('profile-x', request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.profileY,
    operators: Object.freeze(['ProfileY']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('profile-y', request),
    proveDomain: (_operator, request) => proveVisionProfile('profile-y', request),
  }),
  Object.freeze({
    shapeFunctionId: EXTENDED_SHAPE_FUNCTION_IDS.dropout,
    operators: Object.freeze(['Dropout']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferDropout(request),
    proveDomain: (_operator, request) => proveDropout(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.dense,
    operators: Object.freeze(['QLinear', 'QMatMul', 'QGemm']),
    ports: ports(['input', 'weight', 'bias']),
    inferConcrete: inferQDense,
    proveDomain: proveQDense,
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.batchMatMul,
    operators: Object.freeze(['QBatchMatMul']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferQBatchMatMul(request),
    proveDomain: (_operator, request) => proveQBatchMatMul(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.conv2D,
    operators: Object.freeze(['QConv2D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferQConv2D(request),
    proveDomain: (_operator, request) => proveQConv2D(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.exactBinary,
    operators: Object.freeze(['QAdd']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferQAdd(request),
    proveDomain: (_operator, request) => proveQAdd(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.embedding,
    operators: Object.freeze(['QEmbedding']),
    ports: ports(['input', 'weight']),
    inferConcrete: (_operator, request) => inferQEmbedding(request),
    proveDomain: (_operator, request) => proveQEmbedding(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.activation,
    operators: Object.freeze(['QGELU', 'QSiLU']),
    ports: ports(['input']),
    inferConcrete: inferQActivation,
    proveDomain: proveQActivation,
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.featureNorm,
    operators: Object.freeze(['QLayerNorm']),
    ports: ports(['input', 'weight', 'bias']),
    inferConcrete: (_operator, request) => inferQLayerNorm(request),
    proveDomain: (_operator, request) => proveQLayerNorm(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.groupNorm,
    operators: Object.freeze(['QGroupNorm']),
    ports: ports(['input', 'weight', 'bias']),
    inferConcrete: (_operator, request) => inferQGroupNorm(request),
    proveDomain: (_operator, request) => proveQGroupNorm(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.maskedMean,
    operators: Object.freeze(['QMaskedMean']),
    ports: ports(['input', 'mask']),
    inferConcrete: (_operator, request) => inferQMaskedMean(request),
    proveDomain: (_operator, request) => proveQMaskedMean(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.sdpa,
    operators: Object.freeze(['QSDPA']),
    ports: ports(['q', 'k', 'v'], ['mask']),
    inferConcrete: (_operator, request) => inferQSDPA(request),
    proveDomain: (_operator, request) => proveQSDPA(request),
  }),
  Object.freeze({
    shapeFunctionId: QUANTIZED_SHAPE_FUNCTION_IDS.argMax,
    operators: Object.freeze(['QArgMax']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferQArgMax(request),
    proveDomain: (_operator, request) => proveQArgMax(request),
  }),
  Object.freeze({
    shapeFunctionId: ATTENTION_SHAPE_FUNCTION_IDS.sdpa,
    operators: Object.freeze(['SDPA']),
    ports: ports(['qkv'], ['mask']),
    inferConcrete: (_operator, request) => inferSDPA(request),
    proveDomain: (_operator, request) => proveSDPA(request),
  }),
  Object.freeze({
    shapeFunctionId: ATTENTION_SHAPE_FUNCTION_IDS.crossSDPA,
    operators: Object.freeze(['CrossSDPA']),
    ports: ports(['q', 'k', 'v'], ['mask']),
    inferConcrete: (_operator, request) => inferCrossSDPA(request),
    proveDomain: (_operator, request) => proveCrossSDPA(request),
  }),
  Object.freeze({
    shapeFunctionId: ATTENTION_SHAPE_FUNCTION_IDS.rope,
    operators: Object.freeze(['RoPE']),
    ports: ports(['input'], ['position_ids']),
    inferConcrete: (_operator, request) => inferRoPE(request),
    proveDomain: (_operator, request) => proveRoPE(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.batchMatMul,
    operators: Object.freeze(['BatchMatMul']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferBatchMatMul(request),
    proveDomain: (_operator, request) => proveBatchMatMul(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.conv1D,
    operators: Object.freeze(['Conv1D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConv1D(request),
    proveDomain: (_operator, request) => proveConv1D(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.conv2D,
    operators: Object.freeze(['Conv2D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConv2D(request),
    proveDomain: (_operator, request) => proveConv2D(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.convTranspose2D,
    operators: Object.freeze(['ConvTranspose2D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConvTranspose2D(request),
    proveDomain: (_operator, request) => proveConvTranspose2D(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.maxPool2D,
    operators: Object.freeze(['MaxPool2D']),
    ports: ports(['input']),
    inferConcrete: inferPool2D,
    proveDomain: provePool2D,
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.averagePool2D,
    operators: Object.freeze(['AveragePool2D']),
    ports: ports(['input']),
    inferConcrete: inferPool2D,
    proveDomain: provePool2D,
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.globalAveragePool,
    operators: Object.freeze(['GlobalAveragePool']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferGlobalAveragePool(request),
    proveDomain: (_operator, request) => proveGlobalAveragePool(request),
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.resize,
    operators: Object.freeze(['Resize']),
    ports: ports(['input']),
    inferConcrete: inferResize,
    proveDomain: proveResize,
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.resizeNearest2D,
    operators: Object.freeze(['ResizeNearest2D']),
    ports: ports(['input']),
    inferConcrete: inferResize,
    proveDomain: proveResize,
  }),
  Object.freeze({
    shapeFunctionId: SPATIAL_SHAPE_FUNCTION_IDS.upsampleNearest2D,
    operators: Object.freeze(['UpsampleNearest2D']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferUpsampleNearest2D(request),
    proveDomain: (_operator, request) => proveUpsampleNearest2D(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.identity,
    operators: Object.freeze(['Identity']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferIdentity(request),
    proveDomain: (_operator, request) => proveIdentity(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.activation,
    operators: ACTIVATION_OPERATORS,
    ports: ports(['input']),
    inferConcrete: inferActivation,
    proveDomain: proveActivation,
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.activation,
    operators: Object.freeze(['PReLU']),
    ports: ports(['input', 'slope']),
    inferConcrete: (_operator, request) => inferPReLU(request),
    proveDomain: (_operator, request) => provePReLU(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.cast,
    operators: Object.freeze(['Cast']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferCast(request),
    proveDomain: (_operator, request) => proveCast(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.quantizeLinear,
    operators: Object.freeze(['QuantizeLinear']),
    ports: ports(['input', 'scale'], ['zero_point']),
    inferConcrete: (_operator, request) => inferQuantizeLinear(request),
    proveDomain: (_operator, request) => proveQuantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.dequantizeLinear,
    operators: Object.freeze(['DequantizeLinear']),
    ports: ports(['input', 'scale'], ['zero_point']),
    inferConcrete: (_operator, request) => inferDequantizeLinear(request),
    proveDomain: (_operator, request) => proveDequantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.featureNorm,
    operators: Object.freeze(['LayerNorm']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: inferFeatureNorm,
    proveDomain: proveFeatureNorm,
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.featureNorm,
    operators: Object.freeze(['RMSNorm']),
    ports: ports(['input', 'weight']),
    inferConcrete: inferFeatureNorm,
    proveDomain: proveFeatureNorm,
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.groupNorm,
    operators: Object.freeze(['GroupNorm']),
    ports: ports(['input', 'weight', 'bias']),
    inferConcrete: (_operator, request) => inferGroupNorm(request),
    proveDomain: (_operator, request) => proveGroupNorm(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.dense,
    operators: Object.freeze(['Linear', 'Gemm', 'MatMul']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: inferDense,
    proveDomain: proveDense,
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.embedding,
    operators: Object.freeze(['Embedding']),
    ports: ports(['input', 'weight']),
    inferConcrete: (_operator, request) => inferEmbedding(request),
    proveDomain: (_operator, request) => proveEmbedding(request),
  }),
  Object.freeze({
    shapeFunctionId: DIRECT_SHAPE_FUNCTION_IDS.exactBinary,
    operators: Object.freeze(['Add', 'Mul']),
    ports: ports(['a', 'b']),
    inferConcrete: inferExactBinary,
    proveDomain: proveExactBinary,
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.broadcastArithmetic,
    operators: Object.freeze(['Sub', 'Div']),
    ports: ports(['a', 'b']),
    inferConcrete: inferBroadcastArithmetic,
    proveDomain: proveBroadcastArithmetic,
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.broadcastComparison,
    operators: Object.freeze(['Equal', 'GreaterOrEqual']),
    ports: ports(['a', 'b']),
    inferConcrete: inferBroadcastComparison,
    proveDomain: proveBroadcastComparison,
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.where,
    operators: Object.freeze(['Where']),
    ports: ports(['condition', 'a', 'b']),
    inferConcrete: (_operator, request) => inferWhere(request),
    proveDomain: (_operator, request) => proveWhere(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.reduction,
    operators: Object.freeze(['ReduceSum', 'ReduceMean']),
    ports: ports(['input']),
    inferConcrete: inferReduction,
    proveDomain: proveReduction,
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.argMax,
    operators: Object.freeze(['ArgMax']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferArgMax(request),
    proveDomain: (_operator, request) => proveArgMax(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.transpose,
    operators: Object.freeze(['Transpose']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferTranspose(request),
    proveDomain: (_operator, request) => proveTranspose(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.flatten,
    operators: Object.freeze(['Flatten']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferFlatten(request),
    proveDomain: (_operator, request) => proveFlatten(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.squeeze,
    operators: Object.freeze(['Squeeze']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferSqueeze(request),
    proveDomain: (_operator, request) => proveSqueeze(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.unsqueeze,
    operators: Object.freeze(['Unsqueeze']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferUnsqueeze(request),
    proveDomain: (_operator, request) => proveUnsqueeze(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.reshape,
    operators: Object.freeze(['Reshape']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferReshape(request),
    proveDomain: (_operator, request) => proveReshape(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.expand,
    operators: Object.freeze(['Expand']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferExpand(request),
    proveDomain: (_operator, request) => proveExpand(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.concat,
    operators: Object.freeze(['Concat']),
    ports: ports([], [], ['out'], { prefix: 'input', minimum: 2 }),
    inferConcrete: (_operator, request) => inferConcat(request),
    proveDomain: (_operator, request) => proveConcat(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.split,
    operators: Object.freeze(['Split']),
    ports: ports(['input'], [], [], undefined, { prefix: 'out', minimum: 1 }),
    inferConcrete: (_operator, request) => inferSplit(request),
    proveDomain: (_operator, request) => proveSplit(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.slice,
    operators: Object.freeze(['Slice']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferSlice(request),
    proveDomain: (_operator, request) => proveSlice(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.pad,
    operators: Object.freeze(['Pad']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferPad(request),
    proveDomain: (_operator, request) => provePad(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.gather,
    operators: Object.freeze(['Gather']),
    ports: ports(['input', 'indices']),
    inferConcrete: (_operator, request) => inferGather(request),
    proveDomain: (_operator, request) => proveGather(request),
  }),
  Object.freeze({
    shapeFunctionId: STRUCTURAL_SHAPE_FUNCTION_IDS.gatherElements,
    operators: Object.freeze(['GatherElements']),
    ports: ports(['input', 'indices']),
    inferConcrete: (_operator, request) => inferGatherElements(request),
    proveDomain: (_operator, request) => proveGatherElements(request),
  }),
]);

const contracts = new Map<string, OperatorShapeContract>();

for (const definition of DEFINITIONS) {
  for (const operator of definition.operators) {
    if (contracts.has(operator)) {
      throw new Error(`Duplicate operator shape-contract route '${operator}'.`);
    }
    const generatedRoute = generatedOperatorShapeContracts[operator];
    if (generatedRoute === undefined) {
      throw new Error(`Generated registry has no shape-contract route for '${operator}'.`);
    }
    if (generatedRoute.classification !== 'canonical') {
      throw new Error(
        `Generated registry classifies implemented operator '${operator}' as ` +
        `'${generatedRoute.classification}', not 'canonical'.`,
      );
    }
    if (generatedRoute.shapeFunctionId !== definition.shapeFunctionId) {
      throw new Error(
        `Generated shape-function ID for '${operator}' is ` +
        `'${generatedRoute.shapeFunctionId}', expected '${definition.shapeFunctionId}'.`,
      );
    }
    const contract: OperatorShapeContract = Object.freeze({
      operator,
      shapeFunctionId: definition.shapeFunctionId,
      ports: definition.ports,
      inferConcrete(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
        return definition.inferConcrete(operator, request);
      },
      proveDomain(request: OperatorDomainProofRequest): OperatorDomainProof {
        if (!isRecord(request) || !(request.environment instanceof ShapeEnvironment)) {
          return Object.freeze({
            supported: false,
            operator,
            shapeFunctionId: definition.shapeFunctionId,
            code: 'INVALID_REQUEST',
            reason: 'domain proof request requires a ShapeEnvironment and input descriptors.',
          });
        }
        try {
          const result = definition.proveDomain(operator, request);
          return Object.freeze({
            supported: true,
            operator,
            shapeFunctionId: definition.shapeFunctionId,
            outputs: result.outputs,
            facts: result.facts,
            affineRelations: result.affineRelations,
          });
        } catch (error) {
          if (error instanceof OperatorShapeContractError) {
            return Object.freeze({
              supported: false,
              operator,
              shapeFunctionId: definition.shapeFunctionId,
              code: error.code,
              reason: error.message,
            });
          }
          return Object.freeze({
            supported: false,
            operator,
            shapeFunctionId: definition.shapeFunctionId,
            code: 'INVALID_DOMAIN',
            reason: error instanceof Error ? error.message : String(error),
          });
        }
      },
    });
    contracts.set(operator, contract);
  }
}

/* A contract's family follows from the shape function it resolves to, so the
 * six ID tables above are the only place family membership is written down.
 * An operator that resolves to an ID in no table is a bug, not a default. */
function memberOf(
  table: Readonly<Record<string, OperatorShapeFunctionId>>,
  value: OperatorShapeFunctionId,
): boolean {
  return Object.values(table).some((candidate) => candidate === value);
}

function familyProjection<Id extends OperatorShapeFunctionId>(
  family: string,
  table: Readonly<Record<string, OperatorShapeFunctionId>>,
): Readonly<Record<string, Id>> {
  const projection: Record<string, Id> = {};
  for (const [operator, contract] of [...contracts.entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)) {
    if (!memberOf(table, contract.shapeFunctionId)) continue;
    projection[operator] = contract.shapeFunctionId as Id;
  }
  if (Object.keys(projection).length === 0) {
    throw new Error(`No operator resolves to a ${family} shape function.`);
  }
  return Object.freeze(projection);
}

/** Output shape needs no algebra: it is the input, or one axis is replaced. */
export const DIRECT_OPERATOR_SHAPE_FUNCTIONS: Readonly<Record<string, DirectShapeFunctionId>> =
  familyProjection('direct', DIRECT_SHAPE_FUNCTION_IDS);

/** Output shape follows from window and stride arithmetic. */
export const SPATIAL_OPERATOR_SHAPE_FUNCTIONS: Readonly<Record<string, SpatialShapeFunctionId>> =
  familyProjection('spatial', SPATIAL_SHAPE_FUNCTION_IDS);

/** Output shape follows from broadcasting and index arithmetic. */
export const STRUCTURAL_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, StructuralShapeFunctionId>> =
  familyProjection('structural', STRUCTURAL_SHAPE_FUNCTION_IDS);

/** Output shape follows from head and sequence arithmetic. */
export const ATTENTION_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, AttentionShapeFunctionId>> =
  familyProjection('attention', ATTENTION_SHAPE_FUNCTION_IDS);

/** Immutable projection of the canonical imported byte-operator boundary. */
export const QUANTIZED_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, QuantizedShapeFunctionId>> =
  familyProjection('quantized', QUANTIZED_SHAPE_FUNCTION_IDS);

/** Operators outside the core families. */
export const EXTENDED_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, OperatorShapeFunctionId>> =
  familyProjection('extended', EXTENDED_SHAPE_FUNCTION_IDS);

/** Every contract belongs to exactly one family; nothing may fall through. */
export const OPERATOR_SHAPE_FAMILIES = Object.freeze({
  direct: DIRECT_OPERATOR_SHAPE_FUNCTIONS,
  spatial: SPATIAL_OPERATOR_SHAPE_FUNCTIONS,
  structural: STRUCTURAL_OPERATOR_SHAPE_FUNCTIONS,
  attention: ATTENTION_OPERATOR_SHAPE_FUNCTIONS,
  quantized: QUANTIZED_OPERATOR_SHAPE_FUNCTIONS,
  extended: EXTENDED_OPERATOR_SHAPE_FUNCTIONS,
});

/** Case-sensitive and fail-closed; unregistered operators never receive a generic fallback. */
export function getOperatorShapeContract(operator: string): OperatorShapeContract {
  if (typeof operator !== 'string' || operator.length === 0) {
    fail('UNKNOWN_OPERATOR', 'operator', 'must be a non-empty registered operator name.');
  }
  const contract = contracts.get(operator);
  if (contract === undefined) {
    fail('UNKNOWN_OPERATOR', 'operator', `has no registered shape contract for '${operator}'.`);
  }
  return contract;
}

export function inferConcreteOperatorShapes(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  if (!isRecord(request)) fail('INVALID_REQUEST', 'shape inference request', 'must be an object.');
  return getOperatorShapeContract(operator).inferConcrete(request);
}

export function proveOperatorShapeDomain(
  operator: string,
  request: OperatorDomainProofRequest,
): OperatorDomainProof {
  return getOperatorShapeContract(operator).proveDomain(request);
}
