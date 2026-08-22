/* Shared vocabulary for every operator shape contract: the request/response
 * types, the six shape-function-ID families, and the assert/normalize helpers
 * that each infer/prove pair is built from.
 *
 * Depends on nothing else in ts/ops/operatorShapeContracts*, so the family
 * modules can import it without a cycle.
 */
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


/** Stable IDs are graph semantics, not backend kernel or tactic identifiers. */
export const DIRECT_SHAPE_FUNCTION_IDS = Object.freeze({
  identity: generatedOperatorShapeFunctionIds.Identity,
  activation: generatedOperatorShapeFunctionIds.ReLU,
  cast: generatedOperatorShapeFunctionIds.Cast,
  quantizeLinear: generatedOperatorShapeFunctionIds.QuantizeLinear,
  dequantizeLinear: generatedOperatorShapeFunctionIds.DequantizeLinear,
  featureNorm: generatedOperatorShapeFunctionIds.LayerNorm,
  groupNorm: generatedOperatorShapeFunctionIds.GroupNorm,
  dense: generatedOperatorShapeFunctionIds.Linear,
  embedding: generatedOperatorShapeFunctionIds.Embedding,
  exactBinary: generatedOperatorShapeFunctionIds.Add,
} as const);
export type DirectShapeFunctionId =
  (typeof DIRECT_SHAPE_FUNCTION_IDS)[keyof typeof DIRECT_SHAPE_FUNCTION_IDS];

/** Every stable shape-function ID is generated from the authoritative registry. */
/** Every stable shape-function ID is generated from the authoritative registry. */
export type OperatorShapeFunctionId =
  (typeof generatedOperatorShapeFunctionIds)[keyof typeof generatedOperatorShapeFunctionIds];

/** Window and stride arithmetic over spatial axes. */
/** Window and stride arithmetic over spatial axes. */
export const SPATIAL_SHAPE_FUNCTION_IDS = Object.freeze({
  batchMatMul: generatedOperatorShapeFunctionIds.BatchMatMul,
  conv1D: generatedOperatorShapeFunctionIds.Conv1D,
  conv2D: generatedOperatorShapeFunctionIds.Conv2D,
  convTranspose2D: generatedOperatorShapeFunctionIds.ConvTranspose2D,
  maxPool2D: generatedOperatorShapeFunctionIds.MaxPool2D,
  averagePool2D: generatedOperatorShapeFunctionIds.AveragePool2D,
  globalAveragePool: generatedOperatorShapeFunctionIds.GlobalAveragePool,
  resize: generatedOperatorShapeFunctionIds.Resize,
  resizeNearest2D: generatedOperatorShapeFunctionIds.ResizeNearest2D,
  upsampleNearest2D: generatedOperatorShapeFunctionIds.UpsampleNearest2D,
} as const);
export type SpatialShapeFunctionId =
  (typeof SPATIAL_SHAPE_FUNCTION_IDS)[keyof typeof SPATIAL_SHAPE_FUNCTION_IDS];

/** Broadcasting and index arithmetic: the shape movers. */
/** Broadcasting and index arithmetic: the shape movers. */
export const STRUCTURAL_SHAPE_FUNCTION_IDS = Object.freeze({
  broadcastArithmetic: generatedOperatorShapeFunctionIds.Sub,
  broadcastComparison: generatedOperatorShapeFunctionIds.Equal,
  where: generatedOperatorShapeFunctionIds.Where,
  reduction: generatedOperatorShapeFunctionIds.ReduceSum,
  argMax: generatedOperatorShapeFunctionIds.ArgMax,
  transpose: generatedOperatorShapeFunctionIds.Transpose,
  flatten: generatedOperatorShapeFunctionIds.Flatten,
  squeeze: generatedOperatorShapeFunctionIds.Squeeze,
  unsqueeze: generatedOperatorShapeFunctionIds.Unsqueeze,
  reshape: generatedOperatorShapeFunctionIds.Reshape,
  expand: generatedOperatorShapeFunctionIds.Expand,
  concat: generatedOperatorShapeFunctionIds.Concat,
  split: generatedOperatorShapeFunctionIds.Split,
  slice: generatedOperatorShapeFunctionIds.Slice,
  pad: generatedOperatorShapeFunctionIds.Pad,
  gather: generatedOperatorShapeFunctionIds.Gather,
  gatherElements: generatedOperatorShapeFunctionIds.GatherElements,
} as const);
export type StructuralShapeFunctionId =
  (typeof STRUCTURAL_SHAPE_FUNCTION_IDS)[keyof typeof STRUCTURAL_SHAPE_FUNCTION_IDS];

/** Head and sequence arithmetic. */
/** Head and sequence arithmetic. */
export const ATTENTION_SHAPE_FUNCTION_IDS = Object.freeze({
  sdpa: generatedOperatorShapeFunctionIds.SDPA,
  crossSDPA: generatedOperatorShapeFunctionIds.CrossSDPA,
  rope: generatedOperatorShapeFunctionIds.RoPE,
} as const);
export type AttentionShapeFunctionId =
  (typeof ATTENTION_SHAPE_FUNCTION_IDS)[keyof typeof ATTENTION_SHAPE_FUNCTION_IDS];

/** Canonical byte contracts emitted by quantized ONNX importers. */
/** Canonical byte contracts emitted by quantized ONNX importers. */
export const QUANTIZED_SHAPE_FUNCTION_IDS = Object.freeze({
  dense: generatedOperatorShapeFunctionIds.QLinear,
  batchMatMul: generatedOperatorShapeFunctionIds.QBatchMatMul,
  sdpa: generatedOperatorShapeFunctionIds.QSDPA,
  featureNorm: generatedOperatorShapeFunctionIds.QLayerNorm,
  groupNorm: generatedOperatorShapeFunctionIds.QGroupNorm,
  conv2D: generatedOperatorShapeFunctionIds.QConv2D,
  activation: generatedOperatorShapeFunctionIds.QGELU,
  exactBinary: generatedOperatorShapeFunctionIds.QAdd,
  maskedMean: generatedOperatorShapeFunctionIds.QMaskedMean,
  argMax: generatedOperatorShapeFunctionIds.QArgMax,
  embedding: generatedOperatorShapeFunctionIds.QEmbedding,
} as const);
export type QuantizedShapeFunctionId =
  (typeof QUANTIZED_SHAPE_FUNCTION_IDS)[keyof typeof QUANTIZED_SHAPE_FUNCTION_IDS];

/** Final v1 contracts that replace the old fixed-shape-only runtime routes. */
/** Final v1 contracts that replace the old fixed-shape-only runtime routes. */
export const EXTENDED_SHAPE_FUNCTION_IDS = Object.freeze({
  moeRouter: generatedOperatorShapeFunctionIds.MoERouter,
  moeLinear: generatedOperatorShapeFunctionIds.MoELinear,
  crossAttention: generatedOperatorShapeFunctionIds.CrossAttention,
  batchNorm2D: generatedOperatorShapeFunctionIds.BatchNorm2D,
  interpolate1D: generatedOperatorShapeFunctionIds.Interpolate1D,
  logicalNot: generatedOperatorShapeFunctionIds.Not,
  mask: generatedOperatorShapeFunctionIds.Mask,
  broadcast: generatedOperatorShapeFunctionIds.Broadcast,
  concat2: generatedOperatorShapeFunctionIds.Concat2,
  requantizeLinear: generatedOperatorShapeFunctionIds.RequantizeLinear,
  ssmScan: generatedOperatorShapeFunctionIds.SSMScan,
  selectiveScan: generatedOperatorShapeFunctionIds.SelectiveScan,
  spatialSoftargmaxY: generatedOperatorShapeFunctionIds.SpatialSoftargmaxY,
  meanHeight: generatedOperatorShapeFunctionIds.MeanHeight,
  profileX: generatedOperatorShapeFunctionIds.ProfileX,
  profileY: generatedOperatorShapeFunctionIds.ProfileY,
  dropout: generatedOperatorShapeFunctionIds.Dropout,
} as const);
export type OperatorShapeContractErrorCode =
  | 'UNKNOWN_OPERATOR'
  | 'INVALID_REQUEST'
  | 'INVALID_INPUT_PORTS'
  | 'INVALID_OUTPUT_PORTS'
  | 'INVALID_DESCRIPTOR'
  | 'INVALID_DTYPE'
  | 'INVALID_RANK'
  | 'INVALID_PARAMS'
  | 'SHAPE_MISMATCH'
  | 'INVALID_QUANTIZATION'
  | 'UNPROVABLE_DYNAMIC_CONTRACTION'
  | 'UNPROVABLE_DYNAMIC_FEATURE'
  | 'UNPROVABLE_DYNAMIC_CHANNEL'
  | 'UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT'
  | 'UNPROVABLE_DYNAMIC_BROADCAST'
  | 'UNPROVABLE_DYNAMIC_SHAPE_FORMULA'
  | 'UNSAFE_QUANTIZATION_TRANSFORM'
  | 'INVALID_DOMAIN';

/** Inspectable failure used before any backend-specific plan or allocation exists. */
/** Inspectable failure used before any backend-specific plan or allocation exists. */
export class OperatorShapeContractError extends Error {
  readonly code: OperatorShapeContractErrorCode;
  readonly path: string;

  constructor(code: OperatorShapeContractErrorCode, path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'OperatorShapeContractError';
    this.code = code;
    this.path = path;
  }
}
export function fail(
  code: OperatorShapeContractErrorCode,
  path: string,
  message: string,
): never {
  throw new OperatorShapeContractError(code, path, message);
}
export interface OperatorTensorDescriptor {
  readonly shape: readonly number[];
  readonly dtype: RuntimeDType;
  readonly quantization?: TensorQuantization | null;
}
export interface LogicalOperatorTensorDescriptor {
  readonly shape: TensorShapeSpec;
  readonly dtype: RuntimeDType;
  readonly quantization?: TensorQuantization | null;
}
export interface ConcreteOperatorShapeRequest {
  readonly inputs: Readonly<Record<string, OperatorTensorDescriptor>>;
  readonly params?: Readonly<Record<string, unknown>>;
  /** Internal graph output assertions; required by contracts whose dtype is output-authored. */
  readonly declaredOutputs?: Readonly<Record<string, OperatorTensorDescriptor>>;
}
export interface OperatorDomainProofRequest {
  readonly environment: ShapeEnvironment;
  readonly inputs: Readonly<Record<string, LogicalOperatorTensorDescriptor>>;
  readonly params?: Readonly<Record<string, unknown>>;
  /** Internal logical output assertions; never a node-parameter side channel. */
  readonly declaredOutputs?: Readonly<Record<string, LogicalOperatorTensorDescriptor>>;
}
export type ConcreteOperatorOutputs = Readonly<Record<string, OperatorTensorDescriptor>>;
export type LogicalOperatorOutputs = Readonly<Record<string, LogicalOperatorTensorDescriptor>>;

/** One exact output-only dimension derived as target = source + offset. */
/** One exact output-only dimension derived as target = source + offset. */
export interface OperatorAffineDimensionRelation {
  readonly target: string;
  readonly source: string;
  readonly offset: number;
}
export interface AcceptedOperatorDomainProof {
  readonly supported: true;
  readonly operator: string;
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly outputs: LogicalOperatorOutputs;
  readonly facts: readonly string[];
  readonly affineRelations: readonly OperatorAffineDimensionRelation[];
}
export interface RejectedOperatorDomainProof {
  readonly supported: false;
  readonly operator: string;
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly code: OperatorShapeContractErrorCode;
  readonly reason: string;
}
export type OperatorDomainProof =
  | AcceptedOperatorDomainProof
  | RejectedOperatorDomainProof;
export interface OperatorShapePorts {
  readonly requiredInputs: readonly string[];
  readonly optionalInputs: readonly string[];
  readonly outputs: readonly string[];
  /** Canonical numbered ports input0..inputN-1, when present. */
  readonly variadicInputs?: Readonly<{ readonly prefix: string; readonly minimum: number }>;
  /** Canonical numbered ports out0..outN-1, when present. */
  readonly variadicOutputs?: Readonly<{ readonly prefix: string; readonly minimum: number }>;
}
export interface OperatorShapeContract {
  readonly operator: string;
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly ports: OperatorShapePorts;
  inferConcrete(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs;
  proveDomain(request: OperatorDomainProofRequest): OperatorDomainProof;
}
const RUNTIME_DTYPES = new Set<RuntimeDType>(['float32', 'int32', 'int8', 'uint8']);
export function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value);
}
export function ownNames(value: object, path: string): string[] {
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail('INVALID_REQUEST', path, 'must not contain symbol-keyed fields.');
  }
  return Object.getOwnPropertyNames(value);
}
export function assertRuntimeDType(value: unknown, path: string): asserts value is RuntimeDType {
  if (typeof value !== 'string' || !RUNTIME_DTYPES.has(value as RuntimeDType)) {
    fail('INVALID_DTYPE', path, `has unsupported runtime dtype '${String(value)}'.`);
  }
}
export function assertAllowedFields(
  value: Readonly<Record<string, unknown>>,
  allowed: readonly string[],
  path: string,
): void {
  const permitted = new Set(allowed);
  const unexpected = ownNames(value, path).filter((name) => !permitted.has(name)).sort();
  if (unexpected.length !== 0) {
    fail('INVALID_PARAMS', path, `has unsupported field '${unexpected[0]}'.`);
  }
}
export function paramsRecord(
  value: Readonly<Record<string, unknown>> | undefined,
): Readonly<Record<string, unknown>> {
  if (value === undefined) return Object.freeze({});
  if (!isRecord(value)) fail('INVALID_PARAMS', 'operator params', 'must be an object.');
  ownNames(value, 'operator params');
  return value;
}
function canonicalF32Scale(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    fail('INVALID_QUANTIZATION', path, 'must be finite and positive.');
  }
  const canonical = Math.fround(value);
  if (!Number.isFinite(canonical) || canonical <= 0) {
    fail('INVALID_QUANTIZATION', path, 'must be positive and representable as float32.');
  }
  return canonical;
}
function freezeQuantization(
  source: TensorQuantization,
  dtype: RuntimeDType,
  rank: number,
  path: string,
): TensorQuantization {
  if (!isRecord(source)) {
    fail('INVALID_QUANTIZATION', path, 'must be a per_tensor or per_axis object.');
  }
  if (dtype !== 'int8' && dtype !== 'uint8') {
    fail('INVALID_QUANTIZATION', path, `is not valid for dtype '${dtype}'.`);
  }
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  if (source.scheme === 'per_tensor') {
    assertAllowedFields(source, ['scheme', 'scale', 'zero_point'], path);
    const scale = canonicalF32Scale(source.scale, `${path}.scale`);
    if (!Number.isInteger(source.zero_point) || source.zero_point < minimum ||
        source.zero_point > maximum) {
      fail(
        'INVALID_QUANTIZATION',
        `${path}.zero_point`,
        `must be an integer in [${minimum}, ${maximum}] for '${dtype}'.`,
      );
    }
    return Object.freeze({
      scheme: 'per_tensor',
      scale,
      zero_point: source.zero_point,
    });
  }
  if (source.scheme !== 'per_axis') {
    fail('INVALID_QUANTIZATION', `${path}.scheme`, "must be 'per_tensor' or 'per_axis'.");
  }
  assertAllowedFields(source, ['scheme', 'axis', 'scales', 'zero_points'], path);
  if (!Number.isInteger(source.axis) || source.axis < 0 || source.axis >= rank) {
    fail('INVALID_QUANTIZATION', `${path}.axis`, `must be in [0, ${rank}).`);
  }
  if (!Array.isArray(source.scales) || source.scales.length === 0) {
    fail('INVALID_QUANTIZATION', `${path}.scales`, 'must be a non-empty array.');
  }
  const scales = Object.freeze(source.scales.map((scale, index) =>
    canonicalF32Scale(scale, `${path}.scales[${index}]`)));
  if (!Array.isArray(source.zero_points) || source.zero_points.length !== source.scales.length ||
      source.zero_points.some((zeroPoint) => !Number.isInteger(zeroPoint) ||
        zeroPoint < minimum || zeroPoint > maximum)) {
    fail(
      'INVALID_QUANTIZATION',
      `${path}.zero_points`,
      `must match scales and contain integers in [${minimum}, ${maximum}].`,
    );
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis: source.axis,
    scales,
    zero_points: Object.freeze([...source.zero_points]),
  });
}
function concreteDescriptor(
  source: OperatorTensorDescriptor,
  path: string,
): OperatorTensorDescriptor {
  if (!isRecord(source)) fail('INVALID_DESCRIPTOR', path, 'must be an object.');
  const unexpected = ownNames(source, path)
    .filter((field) => !['shape', 'dtype', 'quantization'].includes(field)).sort();
  if (unexpected.length !== 0) {
    fail('INVALID_DESCRIPTOR', path, `has unsupported field '${unexpected[0]}'.`);
  }
  assertRuntimeDType(source.dtype, `${path}.dtype`);
  if (!Array.isArray(source.shape)) fail('INVALID_DESCRIPTOR', `${path}.shape`, 'must be an array.');
  const shape = Object.freeze([...source.shape]);
  try {
    checkedShapeElementCount(shape, `${path}.shape`);
  } catch (error) {
    fail('INVALID_DESCRIPTOR', `${path}.shape`, error instanceof Error ? error.message : String(error));
  }
  const quantization = source.quantization == null
    ? undefined
    : freezeQuantization(source.quantization, source.dtype, shape.length, `${path}.quantization`);
  if (quantization?.scheme === 'per_axis' &&
      quantization.scales.length !== shape[quantization.axis]) {
    fail(
      'INVALID_QUANTIZATION',
      `${path}.quantization.scales`,
      `has length ${quantization.scales.length}, but axis ${quantization.axis} has extent ${shape[quantization.axis]}.`,
    );
  }
  return Object.freeze(quantization === undefined
    ? { shape, dtype: source.dtype }
    : { shape, dtype: source.dtype, quantization });
}
export function fixedDimensionValue(
  dimension: number | string,
  environment: ShapeEnvironment,
): number | undefined {
  if (typeof dimension === 'number') return dimension;
  const constraint = environment.get(dimension);
  return constraint !== undefined && constraint.min === constraint.max
    ? constraint.min
    : undefined;
}
interface LegalDimensionProgression {
  readonly first: number;
  readonly last: number;
  readonly step: number;
  readonly count: number;
}
export function legalDimensionProgression(
  symbol: string,
  environment: ShapeEnvironment,
  path: string,
): LegalDimensionProgression {
  const constraint = environment.get(symbol);
  if (constraint === undefined) {
    fail('INVALID_DOMAIN', path, `references undeclared symbol '${symbol}'.`);
  }
  const step = constraint.multiple_of ?? 1;
  const remainder = constraint.min % step;
  const first = checkedShapeAdd(
    constraint.min,
    remainder === 0 ? 0 : step - remainder,
    `${path} first legal value`,
  );
  const last = checkedShapeSubtract(
    constraint.max,
    constraint.max % step,
    `${path} last legal value`,
  );
  if (first > last) {
    fail('INVALID_DOMAIN', path, `symbol '${symbol}' has an empty legal domain.`);
  }
  return Object.freeze({
    first,
    last,
    step,
    count: Math.floor((last - first) / step) + 1,
  });
}
function logicalDescriptor(
  source: LogicalOperatorTensorDescriptor,
  environment: ShapeEnvironment,
  path: string,
): LogicalOperatorTensorDescriptor {
  if (!isRecord(source)) fail('INVALID_DESCRIPTOR', path, 'must be an object.');
  const unexpected = ownNames(source, path)
    .filter((field) => !['shape', 'dtype', 'quantization'].includes(field)).sort();
  if (unexpected.length !== 0) {
    fail('INVALID_DESCRIPTOR', path, `has unsupported field '${unexpected[0]}'.`);
  }
  assertRuntimeDType(source.dtype, `${path}.dtype`);
  let shape: TensorShapeSpec;
  try {
    shape = createTensorShapeSpec(source.shape, environment, `${path}.shape`);
    checkedShapeElementCount(
      shape.map((dimension) => typeof dimension === 'number'
        ? dimension
        : environment.get(dimension)!.max),
      `${path}.maximum shape`,
    );
  } catch (error) {
    fail('INVALID_DOMAIN', `${path}.shape`, error instanceof Error ? error.message : String(error));
  }
  const quantization = source.quantization == null
    ? undefined
    : freezeQuantization(source.quantization, source.dtype, shape.length, `${path}.quantization`);
  const knownAxisExtent = quantization?.scheme === 'per_axis'
    ? fixedDimensionValue(shape[quantization.axis], environment)
    : undefined;
  if (quantization?.scheme === 'per_axis' && knownAxisExtent === undefined) {
    fail(
      'UNPROVABLE_DYNAMIC_PER_AXIS_EXTENT',
      `${path}.shape[${quantization.axis}]`,
      'per-axis quantization requires one fixed extent over the complete domain.',
    );
  }
  if (quantization?.scheme === 'per_axis' && quantization.scales.length !== knownAxisExtent) {
    fail(
      'INVALID_QUANTIZATION',
      `${path}.quantization.scales`,
      `has length ${quantization.scales.length}, but axis ${quantization.axis} has fixed extent ${knownAxisExtent}.`,
    );
  }
  return Object.freeze(quantization === undefined
    ? { shape, dtype: source.dtype }
    : { shape, dtype: source.dtype, quantization });
}
export function normalizeConcreteInputs(
  source: Readonly<Record<string, OperatorTensorDescriptor>>,
  required: readonly string[],
  optional: readonly string[],
): Readonly<Record<string, OperatorTensorDescriptor>> {
  if (!isRecord(source)) fail('INVALID_INPUT_PORTS', 'operator inputs', 'must be an object.');
  const names = ownNames(source, 'operator inputs').sort();
  const expected = new Set([...required, ...optional]);
  const missing = required.filter((name) => !names.includes(name));
  const unexpected = names.filter((name) => !expected.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail('INVALID_INPUT_PORTS', 'operator inputs', details.join('; '));
  }
  return Object.freeze(Object.fromEntries(names.map((name) => [
    name,
    concreteDescriptor(source[name], `operator input '${name}'`),
  ])));
}
export function normalizeLogicalInputs(
  source: Readonly<Record<string, LogicalOperatorTensorDescriptor>>,
  environment: ShapeEnvironment,
  required: readonly string[],
  optional: readonly string[],
): Readonly<Record<string, LogicalOperatorTensorDescriptor>> {
  if (!isRecord(source)) fail('INVALID_INPUT_PORTS', 'operator inputs', 'must be an object.');
  const names = ownNames(source, 'operator inputs').sort();
  const expected = new Set([...required, ...optional]);
  const missing = required.filter((name) => !names.includes(name));
  const unexpected = names.filter((name) => !expected.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail('INVALID_INPUT_PORTS', 'operator inputs', details.join('; '));
  }
  return Object.freeze(Object.fromEntries(names.map((name) => [
    name,
    logicalDescriptor(source[name], environment, `operator input '${name}'`),
  ])));
}
function declaredOutputNames(source: unknown): readonly string[] {
  if (!isRecord(source)) {
    fail('INVALID_OUTPUT_PORTS', 'declared outputs', "must contain exactly the 'out' descriptor.");
  }
  const names = ownNames(source, 'declared outputs').sort();
  if (names.length !== 1 || names[0] !== 'out') {
    fail('INVALID_OUTPUT_PORTS', 'declared outputs', "must contain exactly the 'out' descriptor.");
  }
  return names;
}
export function normalizeConcreteDeclaredOutput(
  source: Readonly<Record<string, OperatorTensorDescriptor>> | undefined,
): OperatorTensorDescriptor {
  declaredOutputNames(source);
  return concreteDescriptor(source!.out, "declared output 'out'");
}
export function normalizeLogicalDeclaredOutput(
  source: Readonly<Record<string, LogicalOperatorTensorDescriptor>> | undefined,
  environment: ShapeEnvironment,
): LogicalOperatorTensorDescriptor {
  declaredOutputNames(source);
  return logicalDescriptor(source!.out, environment, "declared output 'out'");
}
export function cloneConcreteOutput(
  shape: readonly number[],
  dtype: RuntimeDType,
  quantization?: TensorQuantization,
): ConcreteOperatorOutputs {
  const descriptor = concreteDescriptor(
    quantization === undefined ? { shape, dtype } : { shape, dtype, quantization },
    "operator output 'out'",
  );
  return Object.freeze({ out: descriptor });
}
export function cloneLogicalOutput(
  shape: TensorShapeSpec,
  dtype: RuntimeDType,
  environment: ShapeEnvironment,
  quantization?: TensorQuantization,
): LogicalOperatorOutputs {
  const descriptor = logicalDescriptor(
    quantization === undefined ? { shape, dtype } : { shape, dtype, quantization },
    environment,
    "operator output 'out'",
  );
  return Object.freeze({ out: descriptor });
}
export function cloneConcreteOutputs(
  descriptors: Readonly<Record<string, OperatorTensorDescriptor>>,
): ConcreteOperatorOutputs {
  const entries = ownNames(descriptors, 'operator outputs').sort().map((name) => [
    name,
    concreteDescriptor(descriptors[name], `operator output '${name}'`),
  ] as const);
  return Object.freeze(Object.fromEntries(entries));
}
export function cloneLogicalOutputs(
  descriptors: Readonly<Record<string, LogicalOperatorTensorDescriptor>>,
  environment: ShapeEnvironment,
): LogicalOperatorOutputs {
  const entries = ownNames(descriptors, 'operator outputs').sort().map((name) => [
    name,
    logicalDescriptor(descriptors[name], environment, `operator output '${name}'`),
  ] as const);
  return Object.freeze(Object.fromEntries(entries));
}
export function normalizeConcreteVariadicInputs(
  source: Readonly<Record<string, OperatorTensorDescriptor>>,
  prefix: string,
  minimum: number,
): readonly OperatorTensorDescriptor[] {
  if (!isRecord(source)) fail('INVALID_INPUT_PORTS', 'operator inputs', 'must be an object.');
  const names = ownNames(source, 'operator inputs');
  if (names.length < minimum || names.some((name) => !new RegExp(`^${prefix}(0|[1-9][0-9]*)$`).test(name))) {
    fail(
      'INVALID_INPUT_PORTS',
      'operator inputs',
      `must contain consecutive '${prefix}0'..'${prefix}N' ports with at least ${minimum} inputs.`,
    );
  }
  const ordered = [...names].sort((left, right) =>
    Number(left.slice(prefix.length)) - Number(right.slice(prefix.length)));
  if (ordered.some((name, index) => name !== `${prefix}${index}`)) {
    fail('INVALID_INPUT_PORTS', 'operator inputs', `must not contain gaps in '${prefix}N' ports.`);
  }
  return Object.freeze(ordered.map((name) =>
    concreteDescriptor(source[name], `operator input '${name}'`)));
}
export function normalizeLogicalVariadicInputs(
  source: Readonly<Record<string, LogicalOperatorTensorDescriptor>>,
  environment: ShapeEnvironment,
  prefix: string,
  minimum: number,
): readonly LogicalOperatorTensorDescriptor[] {
  if (!isRecord(source)) fail('INVALID_INPUT_PORTS', 'operator inputs', 'must be an object.');
  const names = ownNames(source, 'operator inputs');
  if (names.length < minimum || names.some((name) => !new RegExp(`^${prefix}(0|[1-9][0-9]*)$`).test(name))) {
    fail(
      'INVALID_INPUT_PORTS',
      'operator inputs',
      `must contain consecutive '${prefix}0'..'${prefix}N' ports with at least ${minimum} inputs.`,
    );
  }
  const ordered = [...names].sort((left, right) =>
    Number(left.slice(prefix.length)) - Number(right.slice(prefix.length)));
  if (ordered.some((name, index) => name !== `${prefix}${index}`)) {
    fail('INVALID_INPUT_PORTS', 'operator inputs', `must not contain gaps in '${prefix}N' ports.`);
  }
  return Object.freeze(ordered.map((name) =>
    logicalDescriptor(source[name], environment, `operator input '${name}'`)));
}
export function assertFloatTensor(
  descriptor: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  path: string,
): void {
  if (descriptor.dtype !== 'float32') fail('INVALID_DTYPE', `${path}.dtype`, "must be 'float32'.");
  if (descriptor.quantization != null) {
    fail('INVALID_QUANTIZATION', `${path}.quantization`, 'is not valid for a float operator input.');
  }
}
export function assertUnquantized(
  descriptor: { readonly quantization?: TensorQuantization | null },
  path: string,
): void {
  if (descriptor.quantization != null) {
    fail('INVALID_QUANTIZATION', `${path}.quantization`, 'must be absent.');
  }
}
export function sameConcreteShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, axis) => dimension === right[axis]);
}
export function dimensionsProvablyEqual(
  left: number | string,
  right: number | string,
  environment: ShapeEnvironment,
): boolean {
  if (left === right) return true;
  const leftFixed = fixedDimensionValue(left, environment);
  const rightFixed = fixedDimensionValue(right, environment);
  return leftFixed !== undefined && rightFixed !== undefined && leftFixed === rightFixed;
}
export function logicalShapesProvablyEqual(
  left: TensorShapeSpec,
  right: TensorShapeSpec,
  environment: ShapeEnvironment,
): boolean {
  return left.length === right.length &&
    left.every((dimension, axis) => dimensionsProvablyEqual(dimension, right[axis], environment));
}
export function assertRank(shape: readonly unknown[], minimum: number, exact: number | null, path: string): void {
  if (exact !== null && shape.length !== exact) {
    fail('INVALID_RANK', path, `must have rank ${exact}, received rank ${shape.length}.`);
  }
  if (exact === null && shape.length < minimum) {
    fail('INVALID_RANK', path, `must have rank at least ${minimum}, received rank ${shape.length}.`);
  }
}
export function normalizeAxis(
  rawAxis: unknown,
  rank: number,
  defaultAxis: number,
  path = 'operator params.axis',
): number {
  if (rank < 1) fail('INVALID_RANK', path, 'requires an input rank of at least 1.');
  const source = rawAxis ?? defaultAxis;
  if (!Number.isInteger(source)) fail('INVALID_PARAMS', path, 'must be an integer.');
  const axis = (source as number) < 0 ? (source as number) + rank : source as number;
  if (axis < 0 || axis >= rank) {
    fail('INVALID_PARAMS', path, `must resolve to an axis in [0, ${rank}).`);
  }
  return axis;
}
export function normalizeAxes(
  source: unknown,
  rank: number,
  path: string,
  options: Readonly<{ outputRank?: number; allowEmpty?: boolean; sort?: boolean }> = {},
): readonly number[] {
  if (!Array.isArray(source) || (!options.allowEmpty && source.length === 0)) {
    fail('INVALID_PARAMS', path, options.allowEmpty ? 'must be an array.' : 'must be a non-empty array.');
  }
  const normalizationRank = options.outputRank ?? rank;
  const axes = source.map((rawAxis, index) => {
    if (!Number.isInteger(rawAxis)) fail('INVALID_PARAMS', `${path}[${index}]`, 'must be an integer.');
    const axis = rawAxis < 0 ? rawAxis + normalizationRank : rawAxis;
    if (axis < 0 || axis >= normalizationRank) {
      fail(
        'INVALID_PARAMS',
        `${path}[${index}]`,
        `must resolve to an axis in [0, ${normalizationRank}).`,
      );
    }
    return axis;
  });
  if (new Set(axes).size !== axes.length) fail('INVALID_PARAMS', path, 'must contain unique axes.');
  return Object.freeze(options.sort === false ? axes : axes.sort((left, right) => left - right));
}
export function booleanParam(
  params: Readonly<Record<string, unknown>>,
  name: string,
  defaultValue: boolean,
): boolean {
  const value = params[name] ?? defaultValue;
  if (typeof value !== 'boolean') {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be boolean.');
  }
  return value;
}
export function safeIntegerArray(
  source: unknown,
  path: string,
  options: Readonly<{ positive?: boolean; nonNegative?: boolean; allowEmpty?: boolean }> = {},
): readonly number[] {
  if (!Array.isArray(source) || (!options.allowEmpty && source.length === 0)) {
    fail('INVALID_PARAMS', path, options.allowEmpty ? 'must be an array.' : 'must be a non-empty array.');
  }
  return Object.freeze(source.map((value, index) => {
    if (!Number.isSafeInteger(value) || (options.positive && value <= 0) ||
        (options.nonNegative && value < 0)) {
      const qualification = options.positive
        ? 'a positive safe integer'
        : options.nonNegative
          ? 'a non-negative safe integer'
          : 'a safe integer';
      fail('INVALID_PARAMS', `${path}[${index}]`, `must be ${qualification}.`);
    }
    return value;
  }));
}
export function quantizationEqual(
  left: TensorQuantization | null | undefined,
  right: TensorQuantization | null | undefined,
): boolean {
  if (left == null || right == null) return left == null && right == null;
  if (left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor' && right.scheme === 'per_tensor') {
    return Math.fround(left.scale) === Math.fround(right.scale) &&
      left.zero_point === right.zero_point;
  }
  if (left.scheme !== 'per_axis' || right.scheme !== 'per_axis') return false;
  return left.axis === right.axis && left.scales.length === right.scales.length &&
    left.zero_points.length === right.zero_points.length &&
    left.scales.every((value, index) => Math.fround(value) === Math.fround(right.scales[index])) &&
    left.zero_points.every((value, index) => value === right.zero_points[index]);
}
export function remapPerAxis(
  quantization: TensorQuantization | null | undefined,
  axis: number,
): TensorQuantization | undefined {
  if (quantization == null) return undefined;
  if (quantization.scheme === 'per_tensor') return quantization;
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze([...quantization.scales]),
    zero_points: Object.freeze([...quantization.zero_points]),
  });
}
export function slicedPerAxis(
  quantization: TensorQuantization,
  axis: number,
  start: number,
  end: number,
  step = 1,
): TensorQuantization {
  if (quantization.scheme === 'per_tensor') return quantization;
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  for (let index = start; index < end; index += step) {
    scales.push(quantization.scales[index]);
    zeroPoints.push(quantization.zero_points[index]);
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze(scales),
    zero_points: Object.freeze(zeroPoints),
  });
}
export function assertSameStorageDType(
  left: { readonly dtype: RuntimeDType },
  right: { readonly dtype: RuntimeDType },
  path: string,
): RuntimeDType {
  if (left.dtype !== right.dtype) {
    fail('INVALID_DTYPE', path, `must have the same dtype, received '${left.dtype}' and '${right.dtype}'.`);
  }
  return left.dtype;
}
export function concreteBroadcastShape(
  left: readonly number[],
  right: readonly number[],
  path = 'operator inputs',
): readonly number[] {
  const rank = Math.max(left.length, right.length);
  if (rank > 8) fail('INVALID_RANK', path, 'broadcast rank must not exceed 8.');
  const output = new Array<number>(rank);
  for (let axis = 0; axis < rank; axis++) {
    const leftAxis = axis - (rank - left.length);
    const rightAxis = axis - (rank - right.length);
    const leftDimension = leftAxis < 0 ? 1 : left[leftAxis];
    const rightDimension = rightAxis < 0 ? 1 : right[rightAxis];
    if (leftDimension !== rightDimension && leftDimension !== 1 && rightDimension !== 1) {
      fail(
        'SHAPE_MISMATCH',
        path,
        `shapes [${left.join(',')}] and [${right.join(',')}] are not broadcast-compatible.`,
      );
    }
    output[axis] = Math.max(leftDimension, rightDimension);
  }
  checkedShapeElementCount(output, `${path} broadcast output`);
  return Object.freeze(output);
}
export function logicalBroadcastDimension(
  left: ShapeDimensionSpec,
  right: ShapeDimensionSpec,
  environment: ShapeEnvironment,
  path: string,
): ShapeDimensionSpec {
  if (left === right) return left;
  if (left === 1) return right;
  if (right === 1) return left;
  const leftFixed = fixedDimensionValue(left, environment);
  const rightFixed = fixedDimensionValue(right, environment);
  if (leftFixed === 1) return right;
  if (rightFixed === 1) return left;
  if (leftFixed !== undefined && rightFixed !== undefined && leftFixed === rightFixed) {
    return leftFixed;
  }
  fail(
    'UNPROVABLE_DYNAMIC_BROADCAST',
    path,
    `dimensions '${String(left)}' and '${String(right)}' are not provably broadcast-compatible over the complete domain.`,
  );
}
export function logicalBroadcastShape(
  left: TensorShapeSpec,
  right: TensorShapeSpec,
  environment: ShapeEnvironment,
  path = 'operator inputs',
): TensorShapeSpec {
  const rank = Math.max(left.length, right.length);
  if (rank > 8) fail('INVALID_RANK', path, 'broadcast rank must not exceed 8.');
  const output = new Array<ShapeDimensionSpec>(rank);
  for (let axis = 0; axis < rank; axis++) {
    const leftAxis = axis - (rank - left.length);
    const rightAxis = axis - (rank - right.length);
    output[axis] = logicalBroadcastDimension(
      leftAxis < 0 ? 1 : left[leftAxis],
      rightAxis < 0 ? 1 : right[rightAxis],
      environment,
      `${path}.broadcast[${axis}]`,
    );
  }
  return createTensorShapeSpec(output, environment, `${path} broadcast output`);
}
export function checkedDimensionsProduct(dimensions: readonly number[], path: string): number {
  let product = 1;
  for (let index = 0; index < dimensions.length; index++) {
    product = checkedShapeMultiply(product, dimensions[index], path);
  }
  return product;
}
interface ProductSignature {
  readonly constant: number;
  readonly symbols: Readonly<Record<string, number>>;
}
function logicalProductSignature(
  shape: TensorShapeSpec,
  environment: ShapeEnvironment,
  path: string,
): ProductSignature {
  let constant = 1;
  const symbols: Record<string, number> = {};
  for (const dimension of shape) {
    const fixed = fixedDimensionValue(dimension, environment);
    if (fixed !== undefined) {
      constant = checkedShapeMultiply(constant, fixed, path);
    } else {
      const symbol = dimension as string;
      symbols[symbol] = (symbols[symbol] ?? 0) + 1;
    }
  }
  return Object.freeze({ constant, symbols: Object.freeze(symbols) });
}
export function logicalProductsProvablyEqual(
  left: TensorShapeSpec,
  right: TensorShapeSpec,
  environment: ShapeEnvironment,
): boolean {
  const a = logicalProductSignature(left, environment, 'left logical shape product');
  const b = logicalProductSignature(right, environment, 'right logical shape product');
  const names = [...new Set([...Object.keys(a.symbols), ...Object.keys(b.symbols)])].sort();
  return a.constant === b.constant &&
    names.every((name) => (a.symbols[name] ?? 0) === (b.symbols[name] ?? 0));
}
export function collapseLogicalDimensions(
  dimensions: TensorShapeSpec,
  environment: ShapeEnvironment,
  path: string,
): ShapeDimensionSpec {
  const signature = logicalProductSignature(dimensions, environment, path);
  const symbols = Object.entries(signature.symbols).filter(([, count]) => count !== 0);
  if (symbols.length === 0) return signature.constant;
  if (symbols.length === 1 && symbols[0][1] === 1 && signature.constant === 1) {
    return symbols[0][0];
  }
  fail(
    'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
    path,
    'the product cannot be represented by one v1 constant-or-symbol dimension.',
  );
}
function targetShapeParams(
  requestParams: Readonly<Record<string, unknown>> | undefined,
): readonly ShapeDimensionSpec[] {
  const params = paramsRecord(requestParams);
  assertAllowedFields(params, ['shape'], 'operator params');
  if (!Array.isArray(params.shape)) {
    fail('INVALID_PARAMS', 'operator params.shape', 'must be a fixed-rank array.');
  }
  return Object.freeze([...params.shape] as ShapeDimensionSpec[]);
}
export function concreteTargetShape(
  requestParams: Readonly<Record<string, unknown>> | undefined,
  declaredOutputs: Readonly<Record<string, OperatorTensorDescriptor>> | undefined,
  dtype: RuntimeDType,
): readonly number[] {
  const target = targetShapeParams(requestParams);
  const containsSymbols = target.some((dimension) => typeof dimension === 'string');
  const declared = declaredOutputs === undefined
    ? undefined
    : normalizeConcreteDeclaredOutput(declaredOutputs);
  if (declared !== undefined && declared.dtype !== dtype) {
    fail('INVALID_DTYPE', "declared output 'out'.dtype", `must be '${dtype}'.`);
  }
  if (containsSymbols && declared === undefined) {
    fail(
      'INVALID_OUTPUT_PORTS',
      'declared outputs',
      'is required to concretize a symbolic target shape.',
    );
  }
  if (declared !== undefined && declared.shape.length !== target.length) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'has a different rank from params.shape.');
  }
  const symbols = new Map<string, number>();
  const output = target.map((dimension, axis) => {
    if (typeof dimension === 'number') {
      if (!Number.isSafeInteger(dimension) || dimension <= 0) {
        fail('INVALID_PARAMS', `operator params.shape[${axis}]`, 'must be a positive safe integer or symbol.');
      }
      if (declared !== undefined && declared.shape[axis] !== dimension) {
        fail(
          'SHAPE_MISMATCH',
          `declared output 'out'.shape[${axis}]`,
          `must equal target constant ${dimension}.`,
        );
      }
      return dimension;
    }
    if (typeof dimension !== 'string' || !SHAPE_SYMBOL_PATTERN.test(dimension)) {
      fail('INVALID_PARAMS', `operator params.shape[${axis}]`, 'must be a positive safe integer or symbol.');
    }
    const value = declared!.shape[axis];
    const previous = symbols.get(dimension);
    if (previous !== undefined && previous !== value) {
      fail(
        'SHAPE_MISMATCH',
        `declared output 'out'.shape[${axis}]`,
        `conflicts with the earlier '${dimension}' target value ${previous}.`,
      );
    }
    symbols.set(dimension, value);
    return value;
  });
  checkedShapeElementCount(output, 'operator target shape');
  return Object.freeze(output);
}
export function logicalTargetShape(
  requestParams: Readonly<Record<string, unknown>> | undefined,
  environment: ShapeEnvironment,
  declaredOutputs: Readonly<Record<string, LogicalOperatorTensorDescriptor>> | undefined,
  dtype: RuntimeDType,
): TensorShapeSpec {
  const target = createTensorShapeSpec(
    targetShapeParams(requestParams),
    environment,
    'operator params.shape',
  );
  if (declaredOutputs !== undefined) {
    const declared = normalizeLogicalDeclaredOutput(declaredOutputs, environment);
    if (declared.dtype !== dtype) {
      fail('INVALID_DTYPE', "declared output 'out'.dtype", `must be '${dtype}'.`);
    }
    if (!logicalShapesProvablyEqual(target, declared.shape, environment)) {
      fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must equal params.shape.');
    }
  }
  return target;
}
export function reshapeQuantization(
  quantization: TensorQuantization | null | undefined,
  inputShape: readonly (number | string)[],
  outputShape: readonly (number | string)[],
  environment: ShapeEnvironment | undefined,
  path: string,
): TensorQuantization | undefined {
  if (quantization == null || quantization.scheme === 'per_tensor') return quantization ?? undefined;
  const equality = (left: number | string, right: number | string): boolean =>
    environment === undefined
      ? left === right
      : dimensionsProvablyEqual(left, right, environment);
  const candidates: number[] = [];
  for (let axis = 0; axis < outputShape.length; axis++) {
    if (!equality(inputShape[quantization.axis], outputShape[axis])) continue;
    const prefixEqual = environment === undefined
      ? checkedDimensionsProduct(inputShape.slice(0, quantization.axis) as readonly number[], `${path} input prefix`) ===
        checkedDimensionsProduct(outputShape.slice(0, axis) as readonly number[], `${path} output prefix`)
      : logicalProductsProvablyEqual(
        inputShape.slice(0, quantization.axis),
        outputShape.slice(0, axis),
        environment,
      );
    const suffixEqual = environment === undefined
      ? checkedDimensionsProduct(inputShape.slice(quantization.axis + 1) as readonly number[], `${path} input suffix`) ===
        checkedDimensionsProduct(outputShape.slice(axis + 1) as readonly number[], `${path} output suffix`)
      : logicalProductsProvablyEqual(
        inputShape.slice(quantization.axis + 1),
        outputShape.slice(axis + 1),
        environment,
      );
    if (prefixEqual && suffixEqual) candidates.push(axis);
  }
  if (candidates.length !== 1) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      path,
      'cannot map the per-axis coordinate uniquely through this reshape.',
    );
  }
  return remapPerAxis(quantization, candidates[0]);
}
export function finitePositiveParam(params: Readonly<Record<string, unknown>>, name: string): void {
  const value = params[name];
  if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)) {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be finite and positive.');
  }
}
export function validateActivationParams(
  operator: string,
  params: Readonly<Record<string, unknown>>,
  rank: number,
  dtype: RuntimeDType,
): void {
  if (operator === 'LeakyReLU') {
    assertAllowedFields(params, ['alpha'], 'operator params');
    const alpha = params.alpha;
    if (alpha !== undefined && (typeof alpha !== 'number' || !Number.isFinite(alpha))) {
      fail('INVALID_PARAMS', 'operator params.alpha', 'must be finite.');
    }
    return;
  }
  if (operator === 'GELU') {
    assertAllowedFields(params, ['approximate'], 'operator params');
    if (params.approximate !== undefined && params.approximate !== 'none' && params.approximate !== 'tanh') {
      fail('INVALID_PARAMS', 'operator params.approximate', "must be 'none' or 'tanh'.");
    }
    return;
  }
  if (operator === 'Clip') {
    assertAllowedFields(params, ['min', 'max'], 'operator params');
    const minimum = params.min ?? (dtype === 'int32' ? -2147483648 : Number.NEGATIVE_INFINITY);
    const maximum = params.max ?? (dtype === 'int32' ? 2147483647 : Number.POSITIVE_INFINITY);
    if (typeof minimum !== 'number' || typeof maximum !== 'number' ||
        Number.isNaN(minimum) || Number.isNaN(maximum) || minimum > maximum ||
        (dtype === 'int32' && (!Number.isInteger(minimum) || !Number.isInteger(maximum) ||
          minimum < -2147483648 || maximum > 2147483647))) {
      fail('INVALID_PARAMS', 'operator params', 'Clip min and max must be ordered numbers.');
    }
    return;
  }
  if (operator === 'Softmax' || operator === 'LogSoftmax') {
    assertAllowedFields(params, ['axis'], 'operator params');
    if (rank === 0) fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank at least 1.');
    const rawAxis = params.axis ?? -1;
    if (!Number.isInteger(rawAxis)) fail('INVALID_PARAMS', 'operator params.axis', 'must be an integer.');
    const axis = (rawAxis as number) < 0 ? (rawAxis as number) + rank : rawAxis as number;
    if (axis !== rank - 1) {
      fail('INVALID_PARAMS', 'operator params.axis', 'must resolve to the last axis.');
    }
    return;
  }
  assertAllowedFields(params, [], 'operator params');
}
export const ACTIVATION_OPERATORS = Object.freeze([
  'ReLU',
  'LeakyReLU',
  'GELU',
  'SiLU',
  'Sigmoid',
  'HardSwish',
  'HardSigmoid',
  'Tanh',
  'Sin',
  'Cos',
  'Clip',
  'Softmax',
  'LogSoftmax',
] as const);

export function assertVectorShape(shape: readonly number[], extent: number, path: string): void {
  if (shape.length !== 1 || shape[0] !== extent) {
    fail('SHAPE_MISMATCH', path, `must have shape [${extent}].`);
  }
}
export function assertLogicalVectorShape(
  shape: TensorShapeSpec,
  extent: number,
  environment: ShapeEnvironment,
  path: string,
): void {
  if (shape.length !== 1 || !dimensionsProvablyEqual(shape[0], extent, environment)) {
    fail('SHAPE_MISMATCH', path, `must have the fixed shape [${extent}].`);
  }
}
export function constantLogicalShape(shape: TensorShapeSpec, path: string): readonly number[] {
  if (shape.some((dimension) => typeof dimension !== 'number')) {
    fail('INVALID_DOMAIN', path, 'weights and affine parameters must have constant shapes.');
  }
  return shape as readonly number[];
}
export function spatialPairParam(
  source: unknown,
  defaultValue: number,
  path: string,
  allowZero: boolean,
  required = false,
): readonly [number, number] {
  if (source === undefined && required) fail('INVALID_PARAMS', path, 'is required.');
  const raw = source === undefined
    ? [defaultValue, defaultValue]
    : Array.isArray(source)
      ? source
      : [source, source];
  if (raw.length < 1 || raw.length > 2) {
    fail('INVALID_PARAMS', path, 'must be a scalar or a one/two-element array.');
  }
  const pair = [raw[0], raw[1] ?? raw[0]];
  for (let axis = 0; axis < pair.length; axis++) {
    if (!Number.isSafeInteger(pair[axis]) || pair[axis] < (allowZero ? 0 : 1)) {
      fail(
        'INVALID_PARAMS',
        `${path}[${axis}]`,
        `must be a ${allowZero ? 'non-negative' : 'positive'} safe integer.`,
      );
    }
  }
  return Object.freeze(pair as [number, number]);
}
export function canonicalLayoutParam(
  params: Readonly<Record<string, unknown>>,
  name: string,
  expected: string,
): void {
  const value = params[name];
  if (value !== undefined && value !== expected) {
    fail('INVALID_PARAMS', `operator params.${name}`, `must be '${expected}'.`);
  }
}
export function activationParam(params: Readonly<Record<string, unknown>>): number {
  const relu = params.relu ?? 0;
  if (!Number.isInteger(relu) || (relu as number) < 0 || (relu as number) > 2) {
    fail('INVALID_PARAMS', 'operator params.relu', 'must be 0, 1, or 2.');
  }
  return relu as number;
}
export function integerParam(
  params: Readonly<Record<string, unknown>>,
  name: string,
  defaultValue: number,
): number {
  const value = params[name] ?? defaultValue;
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be a positive safe integer.');
  }
  return value as number;
}
export function fullSpatialPads(
  params: Readonly<Record<string, unknown>>,
  symmetricOnly = false,
): readonly [number, number, number, number] {
  const padding = spatialPairParam(params.padding, 0, 'operator params.padding', true);
  if (params.pads === undefined) {
    return Object.freeze([padding[0], padding[1], padding[0], padding[1]]);
  }
  if (!Array.isArray(params.pads) || params.pads.length !== 4) {
    fail('INVALID_PARAMS', 'operator params.pads', 'must contain top, left, bottom, and right.');
  }
  const pads = params.pads.map((value, index) => {
    if (!Number.isSafeInteger(value) || value < 0) {
      fail('INVALID_PARAMS', `operator params.pads[${index}]`, 'must be a non-negative safe integer.');
    }
    return value;
  }) as [number, number, number, number];
  if (params.padding !== undefined &&
      (pads[0] !== padding[0] || pads[1] !== padding[1] ||
       pads[2] !== padding[0] || pads[3] !== padding[1])) {
    fail('INVALID_PARAMS', 'operator params', 'padding and pads must describe the same symmetric padding.');
  }
  if (symmetricOnly && (pads[0] !== pads[2] || pads[1] !== pads[3])) {
    fail('INVALID_PARAMS', 'operator params.pads', 'must be symmetric for this operator.');
  }
  return Object.freeze(pads);
}
export function checkedWindowOutput(
  input: number,
  kernel: number,
  stride: number,
  padBefore: number,
  padAfter: number,
  dilation: number,
  path: string,
): number {
  let output: number;
  try {
    const effectiveKernel = checkedShapeAdd(
      checkedShapeMultiply(dilation, checkedShapeSubtract(kernel, 1, path), path),
      1,
      path,
    );
    const padded = checkedShapeAdd(checkedShapeAdd(input, padBefore, path), padAfter, path);
    output = checkedShapeAdd(
      checkedShapeFloorDivide(checkedShapeSubtract(padded, effectiveKernel, path), stride, path),
      1,
      path,
    );
  } catch (error) {
    fail('INVALID_DOMAIN', path, error instanceof Error ? error.message : String(error));
  }
  if (output <= 0) fail('SHAPE_MISMATCH', path, 'produces a non-positive output extent.');
  return output;
}
export function logicalWindowOutput(
  input: ShapeDimensionSpec,
  environment: ShapeEnvironment,
  kernel: number,
  stride: number,
  padBefore: number,
  padAfter: number,
  dilation: number,
  path: string,
): ShapeDimensionSpec {
  const effectiveKernel = checkedShapeAdd(
    checkedShapeMultiply(dilation, checkedShapeSubtract(kernel, 1, path), path),
    1,
    path,
  );
  if (stride === 1 && padBefore + padAfter === effectiveKernel - 1) return input;
  const fixed = fixedDimensionValue(input, environment);
  if (fixed !== undefined) {
    return checkedWindowOutput(fixed, kernel, stride, padBefore, padAfter, dilation, path);
  }
  fail(
    'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
    path,
    `floor-window output from dynamic '${input}' is not one v1 constant-or-symbol dimension.`,
  );
}
export function assertAttentionRank(shape: readonly unknown[], path: string): void {
  if (shape.length !== 2 && shape.length !== 3) {
    fail('INVALID_RANK', path, `must have rank 2 or 3, received rank ${shape.length}.`);
  }
}
export function assertAttentionMaskDType(
  mask: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
): void {
  if (mask.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'mask'.dtype", "must be 'int32'.");
  }
  assertUnquantized(mask, "operator input 'mask'");
}
export function assertConcreteAttentionMask(
  mask: OperatorTensorDescriptor | undefined,
  batch: number,
  queries: number,
  keys: number,
): void {
  if (mask === undefined) return;
  assertAttentionMaskDType(mask);
  const shape = mask.shape;
  const valid =
    (shape.length === 1 && shape[0] === keys) ||
    (shape.length === 2 && shape[1] === keys &&
      (shape[0] === batch || shape[0] === queries)) ||
    (shape.length === 3 && shape[0] === batch &&
      shape[1] === queries && shape[2] === keys);
  if (!valid) {
    fail(
      'SHAPE_MISMATCH',
      "operator input 'mask'.shape",
      'must have shape [K], [B,K], [Q,K], or [B,Q,K].',
    );
  }
}
export function assertLogicalAttentionMask(
  mask: LogicalOperatorTensorDescriptor | undefined,
  batch: ShapeDimensionSpec,
  queries: ShapeDimensionSpec,
  keys: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): void {
  if (mask === undefined) return;
  assertAttentionMaskDType(mask);
  const shape = mask.shape;
  const equal = (left: ShapeDimensionSpec, right: ShapeDimensionSpec): boolean =>
    dimensionsProvablyEqual(left, right, environment);
  const valid =
    (shape.length === 1 && equal(shape[0], keys)) ||
    (shape.length === 2 && equal(shape[1], keys) &&
      (equal(shape[0], batch) || equal(shape[0], queries))) ||
    (shape.length === 3 && equal(shape[0], batch) &&
      equal(shape[1], queries) && equal(shape[2], keys));
  if (!valid) {
    fail(
      'SHAPE_MISMATCH',
      "operator input 'mask'.shape",
      'is not provably [K], [B,K], [Q,K], or [B,Q,K] over the complete domain.',
    );
  }
}
export type PerTensorQuantization = Extract<TensorQuantization, { readonly scheme: 'per_tensor' }>;
export type PerAxisQuantization = Extract<TensorQuantization, { readonly scheme: 'per_axis' }>;
export const I32_ACCUMULATOR_MAX = 0x7fffffff;
export function assertPerTensorByte(
  descriptor: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  path: string,
): PerTensorQuantization {
  if (descriptor.dtype !== 'int8' && descriptor.dtype !== 'uint8') {
    fail('INVALID_DTYPE', `${path}.dtype`, "must be 'int8' or 'uint8'.");
  }
  if (descriptor.quantization?.scheme !== 'per_tensor') {
    fail('INVALID_QUANTIZATION', `${path}.quantization`, 'must be per_tensor.');
  }
  return descriptor.quantization;
}
export function assertAxisZeroByteWeight(
  descriptor: { readonly shape: readonly (number | string)[]; readonly dtype: RuntimeDType;
    readonly quantization?: TensorQuantization | null },
  path: string,
): PerAxisQuantization {
  if (descriptor.dtype !== 'int8' && descriptor.dtype !== 'uint8') {
    fail('INVALID_DTYPE', `${path}.dtype`, "must be 'int8' or 'uint8'.");
  }
  if (descriptor.quantization?.scheme !== 'per_axis' || descriptor.quantization.axis !== 0) {
    fail(
      'INVALID_QUANTIZATION',
      `${path}.quantization`,
      'must be per_axis along output/row axis 0.',
    );
  }
  return descriptor.quantization;
}
export function centeredMagnitude(dtype: RuntimeDType, zeroPoint: number): number {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}
export function maximumWeightMagnitude(
  dtype: RuntimeDType,
  quantization: PerAxisQuantization,
): number {
  return Math.max(...quantization.zero_points.map((zeroPoint) =>
    centeredMagnitude(dtype, zeroPoint)));
}
export function assertI32AccumulatorBound(
  terms: number,
  leftMagnitude: number,
  rightMagnitude: number,
  path: string,
): void {
  let maximum: number;
  try {
    maximum = checkedShapeMultiply(
      checkedShapeMultiply(terms, leftMagnitude, path),
      rightMagnitude,
      path,
    );
  } catch (error) {
    fail('INVALID_DOMAIN', path, error instanceof Error ? error.message : String(error));
  }
  if (maximum > I32_ACCUMULATOR_MAX) {
    fail('INVALID_DOMAIN', path, `may require an I32 accumulator magnitude of ${maximum}.`);
  }
}
export function assertI32CenteredSumBound(
  terms: number,
  magnitude: number,
  path: string,
): void {
  let maximum: number;
  try {
    maximum = checkedShapeMultiply(terms, magnitude, path);
  } catch (error) {
    fail('INVALID_DOMAIN', path, error instanceof Error ? error.message : String(error));
  }
  if (maximum > I32_ACCUMULATOR_MAX) {
    fail('INVALID_DOMAIN', path, `may require an I32 accumulator magnitude of ${maximum}.`);
  }
}
export function assertPositiveF32Ratio(
  left: number,
  right: number,
  divisor: number,
  path: string,
): number {
  const multiplier = Math.fround(Math.fround(left * right) / divisor);
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    fail(
      'UNSAFE_QUANTIZATION_TRANSFORM',
      path,
      'requantization multiplier must be positive and representable as float32.',
    );
  }
  return multiplier;
}
export function maximumDimension(
  dimension: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number {
  return typeof dimension === 'number' ? dimension : environment.get(dimension)!.max;
}
export function concreteQuantizedDeclaredOutput(
  request: ConcreteOperatorShapeRequest,
  expectedShape: readonly number[],
): OperatorTensorDescriptor {
  const output = normalizeConcreteDeclaredOutput(request.declaredOutputs);
  const quantization = assertPerTensorByte(output, "declared output 'out'");
  if (!sameConcreteShape(output.shape, expectedShape)) {
    fail('SHAPE_MISMATCH', "declared output 'out'.shape", 'must equal the inferred output shape.');
  }
  return cloneConcreteOutput(expectedShape, output.dtype, quantization).out;
}
export interface DomainInference {
  readonly outputs: LogicalOperatorOutputs;
  readonly facts: readonly string[];
  readonly affineRelations: readonly OperatorAffineDimensionRelation[];
}
export function domainInference(
  outputs: LogicalOperatorOutputs,
  facts: readonly string[],
  affineRelations: readonly OperatorAffineDimensionRelation[] = [],
): DomainInference {
  return Object.freeze({
    outputs,
    facts: Object.freeze([...facts]),
    affineRelations: Object.freeze(affineRelations.map((relation) => Object.freeze({
      target: relation.target,
      source: relation.source,
      offset: relation.offset,
    }))),
  });
}

export function spatialScalarParam(
  source: unknown,
  defaultValue: number,
  path: string,
  allowZero: boolean,
): number {
  const raw = source === undefined
    ? defaultValue
    : Array.isArray(source) && source.length === 1
      ? source[0]
      : source;
  if (!Number.isSafeInteger(raw) || (raw as number) < (allowZero ? 0 : 1)) {
    fail(
      'INVALID_PARAMS',
      path,
      `must be a ${allowZero ? 'non-negative' : 'positive'} safe integer or one-element array.`,
    );
  }
  return raw as number;
}
export function falseOrAbsentParam(value: unknown, path: string): void {
  if (value !== undefined && value !== false && value !== 0) {
    fail('INVALID_PARAMS', path, 'must be false, 0, or absent.');
  }
}
export function checkedTransposeOutput(
  input: number,
  kernel: number,
  stride: number,
  padding: number,
  path: string,
): number {
  let output: number;
  try {
    output = checkedShapeSubtract(
      checkedShapeAdd(
        checkedShapeMultiply(checkedShapeSubtract(input, 1, path), stride, path),
        kernel,
        path,
      ),
      checkedShapeMultiply(2, padding, path),
      path,
    );
  } catch (error) {
    fail('INVALID_DOMAIN', path, error instanceof Error ? error.message : String(error));
  }
  if (output <= 0) fail('SHAPE_MISMATCH', path, 'produces a non-positive output extent.');
  return output;
}
export function logicalTransposeOutput(
  input: ShapeDimensionSpec,
  environment: ShapeEnvironment,
  kernel: number,
  stride: number,
  padding: number,
  path: string,
): ShapeDimensionSpec {
  if (stride === 1 && kernel - 2 * padding === 1) return input;
  const fixed = fixedDimensionValue(input, environment);
  if (fixed !== undefined) return checkedTransposeOutput(fixed, kernel, stride, padding, path);
  fail(
    'UNPROVABLE_DYNAMIC_SHAPE_FORMULA',
    path,
    `transposed-window output from dynamic '${input}' is not one v1 constant-or-symbol dimension.`,
  );
}
export interface AttentionParameters {
  readonly heads: number;
}
export function attentionParameters(
  params: Readonly<Record<string, unknown>>,
): AttentionParameters {
  assertAllowedFields(params, [
    'heads', 'causal', 'scale',
    'dropout', 'attention_dropout',
    'dropout_seed', 'training_seed', 'seed',
  ], 'operator params');
  if (!Number.isSafeInteger(params.heads) || (params.heads as number) <= 0) {
    fail('INVALID_PARAMS', 'operator params.heads', 'must be a positive safe integer.');
  }
  if (typeof params.causal !== 'boolean') {
    fail('INVALID_PARAMS', 'operator params.causal', 'must be boolean.');
  }
  if (params.scale !== undefined) {
    if (typeof params.scale !== 'number' || !Number.isFinite(params.scale) ||
        params.scale <= 0 || !Number.isFinite(Math.fround(params.scale)) ||
        Math.fround(params.scale) <= 0) {
      fail(
        'INVALID_PARAMS',
        'operator params.scale',
        'must be positive and representable as float32.',
      );
    }
  }
  const probabilityNames = ['dropout', 'attention_dropout']
    .filter((name) => params[name] !== undefined);
  if (probabilityNames.length > 1) {
    fail('INVALID_PARAMS', 'operator params', 'must specify at most one attention-dropout field.');
  }
  const dropout = probabilityNames.length === 0 ? 0 : params[probabilityNames[0]];
  if (typeof dropout !== 'number' || !Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    fail('INVALID_PARAMS', 'operator params.dropout', 'must be finite and in [0, 1).');
  }
  const seedNames = ['dropout_seed', 'training_seed', 'seed']
    .filter((name) => params[name] !== undefined);
  if (seedNames.length > 1) {
    fail('INVALID_PARAMS', 'operator params', 'must specify at most one attention-dropout seed field.');
  }
  const seed = seedNames.length === 0 ? 0 : params[seedNames[0]];
  if (!Number.isSafeInteger(seed) || (seed as number) < 0 || (seed as number) > 0xffffffff) {
    fail('INVALID_PARAMS', 'operator params.dropout_seed', 'must be an unsigned 32-bit integer.');
  }
  return Object.freeze({ heads: params.heads as number });
}

export function requireRankRange(
  shape: readonly unknown[],
  minimum: number,
  maximum: number,
  path: string,
): void {
  if (shape.length < minimum || shape.length > maximum) {
    fail('INVALID_RANK', path, `must have rank ${minimum} through ${maximum}, received rank ${shape.length}.`);
  }
}
export function positiveSafeIntegerParam(
  params: Readonly<Record<string, unknown>>,
  name: string,
  defaultValue?: number,
): number {
  const value = params[name] ?? defaultValue;
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be a positive safe integer.');
  }
  return value as number;
}
