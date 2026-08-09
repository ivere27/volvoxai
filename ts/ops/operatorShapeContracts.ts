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
export const WAVE_A_SHAPE_FUNCTION_IDS = Object.freeze({
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

export type WaveAShapeFunctionId =
  (typeof WAVE_A_SHAPE_FUNCTION_IDS)[keyof typeof WAVE_A_SHAPE_FUNCTION_IDS];

/** Every stable shape-function ID is generated from the authoritative registry. */
export type OperatorShapeFunctionId =
  (typeof generatedOperatorShapeFunctionIds)[keyof typeof generatedOperatorShapeFunctionIds];

export const WAVE_B_SHAPE_FUNCTION_IDS = Object.freeze({
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
  sdpa: generatedOperatorShapeFunctionIds.SDPA,
  crossSDPA: generatedOperatorShapeFunctionIds.CrossSDPA,
  rope: generatedOperatorShapeFunctionIds.RoPE,
} as const);

export type WaveBShapeFunctionId =
  (typeof WAVE_B_SHAPE_FUNCTION_IDS)[keyof typeof WAVE_B_SHAPE_FUNCTION_IDS];

/** Canonical physical-byte contracts emitted by quantized ONNX importers. */
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
export const FINAL_SHAPE_FUNCTION_IDS = Object.freeze({
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

function fail(
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

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value);
}

function ownNames(value: object, path: string): string[] {
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail('INVALID_REQUEST', path, 'must not contain symbol-keyed fields.');
  }
  return Object.getOwnPropertyNames(value);
}

function assertRuntimeDType(value: unknown, path: string): asserts value is RuntimeDType {
  if (typeof value !== 'string' || !RUNTIME_DTYPES.has(value as RuntimeDType)) {
    fail('INVALID_DTYPE', path, `has unsupported runtime dtype '${String(value)}'.`);
  }
}

function assertAllowedFields(
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

function paramsRecord(
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

function fixedDimensionValue(
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

function legalDimensionProgression(
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

function normalizeConcreteInputs(
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

function normalizeLogicalInputs(
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

function normalizeConcreteDeclaredOutput(
  source: Readonly<Record<string, OperatorTensorDescriptor>> | undefined,
): OperatorTensorDescriptor {
  declaredOutputNames(source);
  return concreteDescriptor(source!.out, "declared output 'out'");
}

function normalizeLogicalDeclaredOutput(
  source: Readonly<Record<string, LogicalOperatorTensorDescriptor>> | undefined,
  environment: ShapeEnvironment,
): LogicalOperatorTensorDescriptor {
  declaredOutputNames(source);
  return logicalDescriptor(source!.out, environment, "declared output 'out'");
}

function cloneConcreteOutput(
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

function cloneLogicalOutput(
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

function cloneConcreteOutputs(
  descriptors: Readonly<Record<string, OperatorTensorDescriptor>>,
): ConcreteOperatorOutputs {
  const entries = ownNames(descriptors, 'operator outputs').sort().map((name) => [
    name,
    concreteDescriptor(descriptors[name], `operator output '${name}'`),
  ] as const);
  return Object.freeze(Object.fromEntries(entries));
}

function cloneLogicalOutputs(
  descriptors: Readonly<Record<string, LogicalOperatorTensorDescriptor>>,
  environment: ShapeEnvironment,
): LogicalOperatorOutputs {
  const entries = ownNames(descriptors, 'operator outputs').sort().map((name) => [
    name,
    logicalDescriptor(descriptors[name], environment, `operator output '${name}'`),
  ] as const);
  return Object.freeze(Object.fromEntries(entries));
}

function normalizeConcreteVariadicInputs(
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

function normalizeLogicalVariadicInputs(
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

function assertFloatTensor(
  descriptor: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
  path: string,
): void {
  if (descriptor.dtype !== 'float32') fail('INVALID_DTYPE', `${path}.dtype`, "must be 'float32'.");
  if (descriptor.quantization != null) {
    fail('INVALID_QUANTIZATION', `${path}.quantization`, 'is not valid for a float operator input.');
  }
}

function assertUnquantized(
  descriptor: { readonly quantization?: TensorQuantization | null },
  path: string,
): void {
  if (descriptor.quantization != null) {
    fail('INVALID_QUANTIZATION', `${path}.quantization`, 'must be absent.');
  }
}

function sameConcreteShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, axis) => dimension === right[axis]);
}

function dimensionsProvablyEqual(
  left: number | string,
  right: number | string,
  environment: ShapeEnvironment,
): boolean {
  if (left === right) return true;
  const leftFixed = fixedDimensionValue(left, environment);
  const rightFixed = fixedDimensionValue(right, environment);
  return leftFixed !== undefined && rightFixed !== undefined && leftFixed === rightFixed;
}

function logicalShapesProvablyEqual(
  left: TensorShapeSpec,
  right: TensorShapeSpec,
  environment: ShapeEnvironment,
): boolean {
  return left.length === right.length &&
    left.every((dimension, axis) => dimensionsProvablyEqual(dimension, right[axis], environment));
}

function assertRank(shape: readonly unknown[], minimum: number, exact: number | null, path: string): void {
  if (exact !== null && shape.length !== exact) {
    fail('INVALID_RANK', path, `must have rank ${exact}, received rank ${shape.length}.`);
  }
  if (exact === null && shape.length < minimum) {
    fail('INVALID_RANK', path, `must have rank at least ${minimum}, received rank ${shape.length}.`);
  }
}

function normalizeAxis(
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

function normalizeAxes(
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

function booleanParam(
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

function safeIntegerArray(
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

function quantizationEqual(
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

function remapPerAxis(
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

function slicedPerAxis(
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

function assertSameStorageDType(
  left: { readonly dtype: RuntimeDType },
  right: { readonly dtype: RuntimeDType },
  path: string,
): RuntimeDType {
  if (left.dtype !== right.dtype) {
    fail('INVALID_DTYPE', path, `must have the same dtype, received '${left.dtype}' and '${right.dtype}'.`);
  }
  return left.dtype;
}

function concreteBroadcastShape(
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

function logicalBroadcastDimension(
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

function logicalBroadcastShape(
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

function checkedDimensionsProduct(dimensions: readonly number[], path: string): number {
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

function logicalProductsProvablyEqual(
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

function collapseLogicalDimensions(
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

function concreteTargetShape(
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

function logicalTargetShape(
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

function reshapeQuantization(
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

function finitePositiveParam(params: Readonly<Record<string, unknown>>, name: string): void {
  const value = params[name];
  if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value <= 0)) {
    fail('INVALID_PARAMS', `operator params.${name}`, 'must be finite and positive.');
  }
}

function validateActivationParams(
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

const ACTIVATION_OPERATORS = Object.freeze([
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

function inferIdentity(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const input = inputs.input;
  return cloneConcreteOutput(input.shape, input.dtype, input.quantization ?? undefined);
}

function proveIdentity(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const input = inputs.input;
  return domainInference(
    cloneLogicalOutput(input.shape, input.dtype, request.environment, input.quantization ?? undefined),
    ['output shape, dtype, and quantization equal the input for every legal binding'],
  );
}

function inferActivation(
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

function proveActivation(
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

function inferPReLU(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function provePReLU(request: OperatorDomainProofRequest): DomainInference {
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

function inferCast(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const input = inputs.input;
  assertUnquantized(input, "operator input 'input'");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['to'], 'operator params');
  assertRuntimeDType(params.to, 'operator params.to');
  return cloneConcreteOutput(input.shape, params.to);
}

function proveCast(request: OperatorDomainProofRequest): DomainInference {
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

function assertVectorShape(shape: readonly number[], extent: number, path: string): void {
  if (shape.length !== 1 || shape[0] !== extent) {
    fail('SHAPE_MISMATCH', path, `must have shape [${extent}].`);
  }
}

function assertLogicalVectorShape(
  shape: TensorShapeSpec,
  extent: number,
  environment: ShapeEnvironment,
  path: string,
): void {
  if (shape.length !== 1 || !dimensionsProvablyEqual(shape[0], extent, environment)) {
    fail('SHAPE_MISMATCH', path, `must have the fixed shape [${extent}].`);
  }
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

function inferQuantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQuantizeLinear(request: OperatorDomainProofRequest): DomainInference {
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

function inferDequantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveDequantizeLinear(request: OperatorDomainProofRequest): DomainInference {
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

function inferFeatureNorm(
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

function proveFeatureNorm(
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

function inferGroupNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveGroupNorm(request: OperatorDomainProofRequest): DomainInference {
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

function inferDense(
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

function constantLogicalShape(shape: TensorShapeSpec, path: string): readonly number[] {
  if (shape.some((dimension) => typeof dimension !== 'number')) {
    fail('INVALID_DOMAIN', path, 'weights and affine parameters must have constant shapes.');
  }
  return shape as readonly number[];
}

function proveDense(
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

function inferEmbedding(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveEmbedding(request: OperatorDomainProofRequest): DomainInference {
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

function exactBinaryParams(
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

function inferBroadcastArithmetic(
  _operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['a', 'b'], []);
  assertFloatTensor(inputs.a, "operator input 'a'");
  assertFloatTensor(inputs.b, "operator input 'b'");
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  return cloneConcreteOutput(concreteBroadcastShape(inputs.a.shape, inputs.b.shape), 'float32');
}

function proveBroadcastArithmetic(
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

function inferBroadcastComparison(
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

function proveBroadcastComparison(
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

function inferWhere(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['condition', 'a', 'b'], []);
  const dtype = assertWhereTypes(inputs.condition, inputs.a, inputs.b);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  const dataShape = concreteBroadcastShape(inputs.a.shape, inputs.b.shape);
  return cloneConcreteOutput(concreteBroadcastShape(inputs.condition.shape, dataShape), dtype);
}

function proveWhere(request: OperatorDomainProofRequest): DomainInference {
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

function inferReduction(
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

function proveReduction(
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

function inferArgMax(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 1, null, "operator input 'input'.shape");
  const params = argMaxParams(paramsRecord(request.params), inputs.input.shape.length);
  return cloneConcreteOutput(
    argMaxOutputShape(inputs.input.shape, params.axis, params.keepdims) as readonly number[],
    'int32',
  );
}

function proveArgMax(request: OperatorDomainProofRequest): DomainInference {
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

function inferTranspose(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveTranspose(request: OperatorDomainProofRequest): DomainInference {
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

function inferFlatten(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveFlatten(request: OperatorDomainProofRequest): DomainInference {
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

function inferSqueeze(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveSqueeze(request: OperatorDomainProofRequest): DomainInference {
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

function inferUnsqueeze(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveUnsqueeze(request: OperatorDomainProofRequest): DomainInference {
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

function inferReshape(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveReshape(request: OperatorDomainProofRequest): DomainInference {
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

function inferExpand(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const outputShape = concreteTargetShape(request.params, request.declaredOutputs, inputs.input.dtype);
  assertConcreteExpand(inputs.input.shape, outputShape);
  const quantization = expandQuantization(inputs.input.quantization, inputs.input.shape, outputShape);
  return cloneConcreteOutput(outputShape, inputs.input.dtype, quantization);
}

function proveExpand(request: OperatorDomainProofRequest): DomainInference {
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

function inferConcat(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveConcat(request: OperatorDomainProofRequest): DomainInference {
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

function inferSplit(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveSplit(request: OperatorDomainProofRequest): DomainInference {
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

function inferSlice(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  const plan = concreteSlicePlan(inputs.input.shape, paramsRecord(request.params));
  return cloneConcreteOutput(
    plan.shape,
    inputs.input.dtype,
    sliceQuantization(inputs.input.quantization, plan),
  );
}

function proveSlice(request: OperatorDomainProofRequest): DomainInference {
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

function inferPad(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function provePad(request: OperatorDomainProofRequest): DomainInference {
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

function inferGather(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveGather(request: OperatorDomainProofRequest): DomainInference {
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

function inferGatherElements(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveGatherElements(request: OperatorDomainProofRequest): DomainInference {
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

function spatialPairParam(
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

function spatialScalarParam(
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

function falseOrAbsentParam(value: unknown, path: string): void {
  if (value !== undefined && value !== false && value !== 0) {
    fail('INVALID_PARAMS', path, 'must be false, 0, or absent.');
  }
}

function canonicalLayoutParam(
  params: Readonly<Record<string, unknown>>,
  name: string,
  expected: string,
): void {
  const value = params[name];
  if (value !== undefined && value !== expected) {
    fail('INVALID_PARAMS', `operator params.${name}`, `must be '${expected}'.`);
  }
}

function activationParam(params: Readonly<Record<string, unknown>>): number {
  const relu = params.relu ?? 0;
  if (!Number.isInteger(relu) || (relu as number) < 0 || (relu as number) > 2) {
    fail('INVALID_PARAMS', 'operator params.relu', 'must be 0, 1, or 2.');
  }
  return relu as number;
}

function integerParam(
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

function fullSpatialPads(
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

function checkedWindowOutput(
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

function logicalWindowOutput(
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

function checkedTransposeOutput(
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

function logicalTransposeOutput(
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

function inferBatchMatMul(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveBatchMatMul(request: OperatorDomainProofRequest): DomainInference {
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

function inferConv1D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveConv1D(request: OperatorDomainProofRequest): DomainInference {
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

function inferConv2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveConv2D(request: OperatorDomainProofRequest): DomainInference {
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

function inferConvTranspose2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveConvTranspose2D(request: OperatorDomainProofRequest): DomainInference {
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

function inferPool2D(operator: string, request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function provePool2D(operator: string, request: OperatorDomainProofRequest): DomainInference {
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

function inferGlobalAveragePool(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['data_layout'], 'operator params');
  canonicalLayoutParam(params, 'data_layout', 'NHWC');
  return cloneConcreteOutput([inputs.input.shape[0], 1, 1, inputs.input.shape[3]], 'float32');
}

function proveGlobalAveragePool(request: OperatorDomainProofRequest): DomainInference {
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

function inferResize(operator: string, request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveResize(operator: string, request: OperatorDomainProofRequest): DomainInference {
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

function inferUpsampleNearest2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveUpsampleNearest2D(request: OperatorDomainProofRequest): DomainInference {
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

interface AttentionParameters {
  readonly heads: number;
}

function attentionParameters(
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

function assertAttentionRank(shape: readonly unknown[], path: string): void {
  if (shape.length !== 2 && shape.length !== 3) {
    fail('INVALID_RANK', path, `must have rank 2 or 3, received rank ${shape.length}.`);
  }
}

function assertAttentionMaskDType(
  mask: { readonly dtype: RuntimeDType; readonly quantization?: TensorQuantization | null },
): void {
  if (mask.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'mask'.dtype", "must be 'int32'.");
  }
  assertUnquantized(mask, "operator input 'mask'");
}

function assertConcreteAttentionMask(
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

function assertLogicalAttentionMask(
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

function inferSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveSDPA(request: OperatorDomainProofRequest): DomainInference {
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

function inferCrossSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveCrossSDPA(request: OperatorDomainProofRequest): DomainInference {
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

function inferRoPE(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveRoPE(request: OperatorDomainProofRequest): DomainInference {
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

type PerTensorQuantization = Extract<TensorQuantization, { readonly scheme: 'per_tensor' }>;
type PerAxisQuantization = Extract<TensorQuantization, { readonly scheme: 'per_axis' }>;

const I32_ACCUMULATOR_MAX = 0x7fffffff;

function assertPerTensorByte(
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

function assertAxisZeroByteWeight(
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

function centeredMagnitude(dtype: RuntimeDType, zeroPoint: number): number {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}

function maximumWeightMagnitude(
  dtype: RuntimeDType,
  quantization: PerAxisQuantization,
): number {
  return Math.max(...quantization.zero_points.map((zeroPoint) =>
    centeredMagnitude(dtype, zeroPoint)));
}

function assertI32AccumulatorBound(
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

function assertI32CenteredSumBound(
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

function assertPositiveF32Ratio(
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

function maximumDimension(
  dimension: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number {
  return typeof dimension === 'number' ? dimension : environment.get(dimension)!.max;
}

function concreteQuantizedDeclaredOutput(
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

function inferQDense(
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

function proveQDense(
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

function inferQBatchMatMul(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQBatchMatMul(request: OperatorDomainProofRequest): DomainInference {
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

function inferQConv2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQConv2D(request: OperatorDomainProofRequest): DomainInference {
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

function inferQAdd(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQAdd(request: OperatorDomainProofRequest): DomainInference {
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

function inferQActivation(
  operator: string,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  validateQActivationParams(operator, paramsRecord(request.params));
  const output = concreteQuantizedDeclaredOutput(request, inputs.input.shape);
  return Object.freeze({ out: output });
}

function proveQActivation(
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

function inferQEmbedding(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQEmbedding(request: OperatorDomainProofRequest): DomainInference {
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

function inferQLayerNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQLayerNorm(request: OperatorDomainProofRequest): DomainInference {
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

function inferQGroupNorm(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQGroupNorm(request: OperatorDomainProofRequest): DomainInference {
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

function inferQMaskedMean(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQMaskedMean(request: OperatorDomainProofRequest): DomainInference {
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

function inferQSDPA(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveQSDPA(request: OperatorDomainProofRequest): DomainInference {
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

function inferQArgMax(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertPerTensorByte(inputs.input, "operator input 'input'");
  if (inputs.input.shape.length < 2 || inputs.input.shape.length > 8) {
    fail('INVALID_RANK', "operator input 'input'.shape", 'must have rank 2 through 8.');
  }
  const axis = qArgMaxAxis(paramsRecord(request.params), inputs.input.shape.length);
  return cloneConcreteOutput(inputs.input.shape.slice(0, axis), 'int32');
}

function proveQArgMax(request: OperatorDomainProofRequest): DomainInference {
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

function requireRankRange(
  shape: readonly unknown[],
  minimum: number,
  maximum: number,
  path: string,
): void {
  if (shape.length < minimum || shape.length > maximum) {
    fail('INVALID_RANK', path, `must have rank ${minimum} through ${maximum}, received rank ${shape.length}.`);
  }
}

function positiveSafeIntegerParam(
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

function inferMoERouter(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveMoERouter(request: OperatorDomainProofRequest): DomainInference {
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

function inferMoELinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveMoELinear(request: OperatorDomainProofRequest): DomainInference {
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

function inferCrossAttention(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveCrossAttention(request: OperatorDomainProofRequest): DomainInference {
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

function inferBatchNorm2D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveBatchNorm2D(request: OperatorDomainProofRequest): DomainInference {
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

function inferInterpolate1D(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 3, "operator input 'input'.shape");
  const params = paramsRecord(request.params);
  assertAllowedFields(params, ['size'], 'operator params');
  const size = positiveSafeIntegerParam(params, 'size');
  return cloneConcreteOutput([inputs.input.shape[0], inputs.input.shape[1], size], 'float32');
}

function proveInterpolate1D(request: OperatorDomainProofRequest): DomainInference {
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

function inferLogicalNot(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertUnquantized(inputs.input, "operator input 'input'");
  if (inputs.input.dtype !== 'int32') {
    fail('INVALID_DTYPE', "operator input 'input'.dtype", "must be 'int32'.");
  }
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  return cloneConcreteOutput(inputs.input.shape, 'int32');
}

function proveLogicalNot(request: OperatorDomainProofRequest): DomainInference {
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

function inferMask(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveMask(request: OperatorDomainProofRequest): DomainInference {
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

function inferConcat2(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveConcat2(request: OperatorDomainProofRequest): DomainInference {
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

function inferRequantizeLinear(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveRequantizeLinear(request: OperatorDomainProofRequest): DomainInference {
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

function inferScan(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
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

function proveScan(request: OperatorDomainProofRequest): DomainInference {
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

function inferVisionProfile(
  kind: VisionProfileKind,
  request: ConcreteOperatorShapeRequest,
): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertAllowedFields(paramsRecord(request.params), [], 'operator params');
  assertFloatTensor(inputs.input, "operator input 'input'");
  assertRank(inputs.input.shape, 0, 4, "operator input 'input'.shape");
  return cloneConcreteOutput(concreteVisionProfileShape(kind, inputs.input.shape), 'float32');
}

function proveVisionProfile(
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

function inferDropout(request: ConcreteOperatorShapeRequest): ConcreteOperatorOutputs {
  const inputs = normalizeConcreteInputs(request.inputs, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  dropoutParams(paramsRecord(request.params));
  return cloneConcreteOutput(inputs.input.shape, 'float32');
}

function proveDropout(request: OperatorDomainProofRequest): DomainInference {
  const inputs = normalizeLogicalInputs(request.inputs, request.environment, ['input'], []);
  assertFloatTensor(inputs.input, "operator input 'input'");
  requireRankRange(inputs.input.shape, 0, 8, "operator input 'input'.shape");
  dropoutParams(paramsRecord(request.params));
  return domainInference(
    cloneLogicalOutput(inputs.input.shape, 'float32', request.environment),
    ['shape is value-independent; training randomness is keyed by seed, counter, and concrete linear index'],
  );
}

interface DomainInference {
  readonly outputs: LogicalOperatorOutputs;
  readonly facts: readonly string[];
  readonly affineRelations: readonly OperatorAffineDimensionRelation[];
}

function domainInference(
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
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.moeRouter,
    operators: Object.freeze(['MoERouter']),
    ports: ports(['input', 'weight'], ['bias'], ['indices', 'weights']),
    inferConcrete: (_operator, request) => inferMoERouter(request),
    proveDomain: (_operator, request) => proveMoERouter(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.moeLinear,
    operators: Object.freeze(['MoELinear']),
    ports: ports(
      ['input', 'expert_weight', 'route_indices', 'route_weights'], ['expert_bias'],
    ),
    inferConcrete: (_operator, request) => inferMoELinear(request),
    proveDomain: (_operator, request) => proveMoELinear(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.crossAttention,
    operators: Object.freeze(['CrossAttention']),
    ports: ports(['q', 'kv', 'weight'], ['scale', 'bias']),
    inferConcrete: (_operator, request) => inferCrossAttention(request),
    proveDomain: (_operator, request) => proveCrossAttention(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.batchNorm2D,
    operators: Object.freeze(['BatchNorm2D']),
    ports: ports(['input', 'weight', 'bias', 'running_mean', 'running_var']),
    inferConcrete: (_operator, request) => inferBatchNorm2D(request),
    proveDomain: (_operator, request) => proveBatchNorm2D(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.interpolate1D,
    operators: Object.freeze(['Interpolate1D']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferInterpolate1D(request),
    proveDomain: (_operator, request) => proveInterpolate1D(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.logicalNot,
    operators: Object.freeze(['Not']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferLogicalNot(request),
    proveDomain: (_operator, request) => proveLogicalNot(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.mask,
    operators: Object.freeze(['Mask']),
    ports: ports(['mask', 'a', 'b']),
    inferConcrete: (_operator, request) => inferMask(request),
    proveDomain: (_operator, request) => proveMask(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.broadcast,
    operators: Object.freeze(['Broadcast']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferExpand(request),
    proveDomain: (_operator, request) => proveExpand(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.concat2,
    operators: Object.freeze(['Concat2']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferConcat2(request),
    proveDomain: (_operator, request) => proveConcat2(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.requantizeLinear,
    operators: Object.freeze(['RequantizeLinear']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferRequantizeLinear(request),
    proveDomain: (_operator, request) => proveRequantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.ssmScan,
    operators: Object.freeze(['SSMScan']),
    ports: ports(['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'], ['out', 'state']),
    inferConcrete: (_operator, request) => inferScan(request),
    proveDomain: (_operator, request) => proveScan(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.selectiveScan,
    operators: Object.freeze(['SelectiveScan']),
    ports: ports(['input', 'delta', 'A', 'B', 'C'], ['D', 'z', 'initial_state'], ['out', 'state']),
    inferConcrete: (_operator, request) => inferScan(request),
    proveDomain: (_operator, request) => proveScan(request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.spatialSoftargmaxY,
    operators: Object.freeze(['SpatialSoftargmaxY']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('softargmax', request),
    proveDomain: (_operator, request) => proveVisionProfile('softargmax', request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.meanHeight,
    operators: Object.freeze(['MeanHeight']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('mean', request),
    proveDomain: (_operator, request) => proveVisionProfile('mean', request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.profileX,
    operators: Object.freeze(['ProfileX']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('profile-x', request),
    proveDomain: (_operator, request) => proveVisionProfile('profile-x', request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.profileY,
    operators: Object.freeze(['ProfileY']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferVisionProfile('profile-y', request),
    proveDomain: (_operator, request) => proveVisionProfile('profile-y', request),
  }),
  Object.freeze({
    shapeFunctionId: FINAL_SHAPE_FUNCTION_IDS.dropout,
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
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.sdpa,
    operators: Object.freeze(['SDPA']),
    ports: ports(['qkv'], ['mask']),
    inferConcrete: (_operator, request) => inferSDPA(request),
    proveDomain: (_operator, request) => proveSDPA(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.crossSDPA,
    operators: Object.freeze(['CrossSDPA']),
    ports: ports(['q', 'k', 'v'], ['mask']),
    inferConcrete: (_operator, request) => inferCrossSDPA(request),
    proveDomain: (_operator, request) => proveCrossSDPA(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.rope,
    operators: Object.freeze(['RoPE']),
    ports: ports(['input'], ['position_ids']),
    inferConcrete: (_operator, request) => inferRoPE(request),
    proveDomain: (_operator, request) => proveRoPE(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.batchMatMul,
    operators: Object.freeze(['BatchMatMul']),
    ports: ports(['a', 'b']),
    inferConcrete: (_operator, request) => inferBatchMatMul(request),
    proveDomain: (_operator, request) => proveBatchMatMul(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.conv1D,
    operators: Object.freeze(['Conv1D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConv1D(request),
    proveDomain: (_operator, request) => proveConv1D(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.conv2D,
    operators: Object.freeze(['Conv2D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConv2D(request),
    proveDomain: (_operator, request) => proveConv2D(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.convTranspose2D,
    operators: Object.freeze(['ConvTranspose2D']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: (_operator, request) => inferConvTranspose2D(request),
    proveDomain: (_operator, request) => proveConvTranspose2D(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.maxPool2D,
    operators: Object.freeze(['MaxPool2D']),
    ports: ports(['input']),
    inferConcrete: inferPool2D,
    proveDomain: provePool2D,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.averagePool2D,
    operators: Object.freeze(['AveragePool2D']),
    ports: ports(['input']),
    inferConcrete: inferPool2D,
    proveDomain: provePool2D,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.globalAveragePool,
    operators: Object.freeze(['GlobalAveragePool']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferGlobalAveragePool(request),
    proveDomain: (_operator, request) => proveGlobalAveragePool(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.resize,
    operators: Object.freeze(['Resize']),
    ports: ports(['input']),
    inferConcrete: inferResize,
    proveDomain: proveResize,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.resizeNearest2D,
    operators: Object.freeze(['ResizeNearest2D']),
    ports: ports(['input']),
    inferConcrete: inferResize,
    proveDomain: proveResize,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.upsampleNearest2D,
    operators: Object.freeze(['UpsampleNearest2D']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferUpsampleNearest2D(request),
    proveDomain: (_operator, request) => proveUpsampleNearest2D(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.identity,
    operators: Object.freeze(['Identity']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferIdentity(request),
    proveDomain: (_operator, request) => proveIdentity(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.activation,
    operators: ACTIVATION_OPERATORS,
    ports: ports(['input']),
    inferConcrete: inferActivation,
    proveDomain: proveActivation,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.activation,
    operators: Object.freeze(['PReLU']),
    ports: ports(['input', 'slope']),
    inferConcrete: (_operator, request) => inferPReLU(request),
    proveDomain: (_operator, request) => provePReLU(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.cast,
    operators: Object.freeze(['Cast']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferCast(request),
    proveDomain: (_operator, request) => proveCast(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.quantizeLinear,
    operators: Object.freeze(['QuantizeLinear']),
    ports: ports(['input', 'scale'], ['zero_point']),
    inferConcrete: (_operator, request) => inferQuantizeLinear(request),
    proveDomain: (_operator, request) => proveQuantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.dequantizeLinear,
    operators: Object.freeze(['DequantizeLinear']),
    ports: ports(['input', 'scale'], ['zero_point']),
    inferConcrete: (_operator, request) => inferDequantizeLinear(request),
    proveDomain: (_operator, request) => proveDequantizeLinear(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.featureNorm,
    operators: Object.freeze(['LayerNorm']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: inferFeatureNorm,
    proveDomain: proveFeatureNorm,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.featureNorm,
    operators: Object.freeze(['RMSNorm']),
    ports: ports(['input', 'weight']),
    inferConcrete: inferFeatureNorm,
    proveDomain: proveFeatureNorm,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.groupNorm,
    operators: Object.freeze(['GroupNorm']),
    ports: ports(['input', 'weight', 'bias']),
    inferConcrete: (_operator, request) => inferGroupNorm(request),
    proveDomain: (_operator, request) => proveGroupNorm(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.dense,
    operators: Object.freeze(['Linear', 'Gemm', 'MatMul']),
    ports: ports(['input', 'weight'], ['bias']),
    inferConcrete: inferDense,
    proveDomain: proveDense,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.embedding,
    operators: Object.freeze(['Embedding']),
    ports: ports(['input', 'weight']),
    inferConcrete: (_operator, request) => inferEmbedding(request),
    proveDomain: (_operator, request) => proveEmbedding(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_A_SHAPE_FUNCTION_IDS.exactBinary,
    operators: Object.freeze(['Add', 'Mul']),
    ports: ports(['a', 'b']),
    inferConcrete: inferExactBinary,
    proveDomain: proveExactBinary,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.broadcastArithmetic,
    operators: Object.freeze(['Sub', 'Div']),
    ports: ports(['a', 'b']),
    inferConcrete: inferBroadcastArithmetic,
    proveDomain: proveBroadcastArithmetic,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.broadcastComparison,
    operators: Object.freeze(['Equal', 'GreaterOrEqual']),
    ports: ports(['a', 'b']),
    inferConcrete: inferBroadcastComparison,
    proveDomain: proveBroadcastComparison,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.where,
    operators: Object.freeze(['Where']),
    ports: ports(['condition', 'a', 'b']),
    inferConcrete: (_operator, request) => inferWhere(request),
    proveDomain: (_operator, request) => proveWhere(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.reduction,
    operators: Object.freeze(['ReduceSum', 'ReduceMean']),
    ports: ports(['input']),
    inferConcrete: inferReduction,
    proveDomain: proveReduction,
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.argMax,
    operators: Object.freeze(['ArgMax']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferArgMax(request),
    proveDomain: (_operator, request) => proveArgMax(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.transpose,
    operators: Object.freeze(['Transpose']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferTranspose(request),
    proveDomain: (_operator, request) => proveTranspose(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.flatten,
    operators: Object.freeze(['Flatten']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferFlatten(request),
    proveDomain: (_operator, request) => proveFlatten(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.squeeze,
    operators: Object.freeze(['Squeeze']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferSqueeze(request),
    proveDomain: (_operator, request) => proveSqueeze(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.unsqueeze,
    operators: Object.freeze(['Unsqueeze']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferUnsqueeze(request),
    proveDomain: (_operator, request) => proveUnsqueeze(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.reshape,
    operators: Object.freeze(['Reshape']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferReshape(request),
    proveDomain: (_operator, request) => proveReshape(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.expand,
    operators: Object.freeze(['Expand']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferExpand(request),
    proveDomain: (_operator, request) => proveExpand(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.concat,
    operators: Object.freeze(['Concat']),
    ports: ports([], [], ['out'], { prefix: 'input', minimum: 2 }),
    inferConcrete: (_operator, request) => inferConcat(request),
    proveDomain: (_operator, request) => proveConcat(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.split,
    operators: Object.freeze(['Split']),
    ports: ports(['input'], [], [], undefined, { prefix: 'out', minimum: 1 }),
    inferConcrete: (_operator, request) => inferSplit(request),
    proveDomain: (_operator, request) => proveSplit(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.slice,
    operators: Object.freeze(['Slice']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferSlice(request),
    proveDomain: (_operator, request) => proveSlice(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.pad,
    operators: Object.freeze(['Pad']),
    ports: ports(['input']),
    inferConcrete: (_operator, request) => inferPad(request),
    proveDomain: (_operator, request) => provePad(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.gather,
    operators: Object.freeze(['Gather']),
    ports: ports(['input', 'indices']),
    inferConcrete: (_operator, request) => inferGather(request),
    proveDomain: (_operator, request) => proveGather(request),
  }),
  Object.freeze({
    shapeFunctionId: WAVE_B_SHAPE_FUNCTION_IDS.gatherElements,
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

const WAVE_A_OPERATOR_NAMES: ReadonlySet<string> = new Set([
  'Identity', ...ACTIVATION_OPERATORS, 'PReLU', 'Cast', 'QuantizeLinear',
  'DequantizeLinear', 'LayerNorm', 'RMSNorm', 'GroupNorm', 'Linear', 'Gemm',
  'MatMul', 'Embedding', 'Add', 'Mul',
]);

const QUANTIZED_OPERATOR_NAMES: ReadonlySet<string> = new Set([
  'QLinear', 'QMatMul', 'QGemm', 'QBatchMatMul', 'QConv2D', 'QAdd',
  'QEmbedding', 'QGELU', 'QSiLU', 'QLayerNorm', 'QGroupNorm',
  'QMaskedMean', 'QSDPA', 'QArgMax',
]);

const FINAL_OPERATOR_NAMES: ReadonlySet<string> = new Set([
  'MoERouter', 'MoELinear', 'CrossAttention', 'BatchNorm2D', 'Interpolate1D',
  'Not', 'Mask', 'Broadcast', 'Concat2', 'RequantizeLinear', 'SSMScan',
  'SelectiveScan', 'SpatialSoftargmaxY', 'MeanHeight', 'ProfileX', 'ProfileY',
  'Dropout',
]);

function isWaveAShapeFunctionId(value: OperatorShapeFunctionId): value is WaveAShapeFunctionId {
  return Object.values(WAVE_A_SHAPE_FUNCTION_IDS).some((candidate) => candidate === value);
}

function isWaveBShapeFunctionId(value: OperatorShapeFunctionId): value is WaveBShapeFunctionId {
  return Object.values(WAVE_B_SHAPE_FUNCTION_IDS).some((candidate) => candidate === value);
}

function isQuantizedShapeFunctionId(
  value: OperatorShapeFunctionId,
): value is QuantizedShapeFunctionId {
  return Object.values(QUANTIZED_SHAPE_FUNCTION_IDS).some((candidate) => candidate === value);
}

function waveAOperatorProjection(): Readonly<Record<string, WaveAShapeFunctionId>> {
  const projection: Record<string, WaveAShapeFunctionId> = {};
  for (const [operator, contract] of [...contracts.entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)) {
    if (!WAVE_A_OPERATOR_NAMES.has(operator)) continue;
    if (!isWaveAShapeFunctionId(contract.shapeFunctionId)) {
      throw new Error(
        `Wave-A operator '${operator}' resolved to non-Wave-A shape ID '${contract.shapeFunctionId}'.`,
      );
    }
    projection[operator] = contract.shapeFunctionId;
  }
  return Object.freeze(projection);
}

function waveBOperatorProjection(): Readonly<Record<string, WaveBShapeFunctionId>> {
  const projection: Record<string, WaveBShapeFunctionId> = {};
  for (const [operator, contract] of [...contracts.entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)) {
    if (WAVE_A_OPERATOR_NAMES.has(operator) || QUANTIZED_OPERATOR_NAMES.has(operator) ||
        FINAL_OPERATOR_NAMES.has(operator)) continue;
    if (!isWaveBShapeFunctionId(contract.shapeFunctionId)) {
      throw new Error(
        `Wave-B operator '${operator}' resolved to non-Wave-B shape ID '${contract.shapeFunctionId}'.`,
      );
    }
    projection[operator] = contract.shapeFunctionId;
  }
  return Object.freeze(projection);
}


function quantizedOperatorProjection(): Readonly<Record<string, QuantizedShapeFunctionId>> {
  const projection: Record<string, QuantizedShapeFunctionId> = {};
  for (const [operator, contract] of [...contracts.entries()]
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)) {
    if (!QUANTIZED_OPERATOR_NAMES.has(operator)) continue;
    if (!isQuantizedShapeFunctionId(contract.shapeFunctionId)) {
      throw new Error(
        `Quantized operator '${operator}' resolved to non-quantized shape ID '${contract.shapeFunctionId}'.`,
      );
    }
    projection[operator] = contract.shapeFunctionId;
  }
  return Object.freeze(projection);
}

function finalOperatorProjection(): Readonly<Record<string, OperatorShapeFunctionId>> {
  const projection: Record<string, OperatorShapeFunctionId> = {};
  for (const operator of [...FINAL_OPERATOR_NAMES].sort()) {
    const contract = contracts.get(operator);
    if (contract === undefined) {
      throw new Error(`Final-v1 operator '${operator}' has no implemented shape contract.`);
    }
    projection[operator] = contract.shapeFunctionId;
  }
  return Object.freeze(projection);
}

/** Immutable operator-to-shape-function projection for the initial Wave-A tranche. */
export const WAVE_A_OPERATOR_SHAPE_FUNCTIONS: Readonly<Record<string, WaveAShapeFunctionId>> =
  waveAOperatorProjection();

/** Immutable projection of the implemented broadly-useful Wave-B tranche. */
export const WAVE_B_OPERATOR_SHAPE_FUNCTIONS: Readonly<Record<string, WaveBShapeFunctionId>> =
  waveBOperatorProjection();

/** Immutable projection of the canonical imported byte-operator boundary. */
export const QUANTIZED_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, QuantizedShapeFunctionId>> = quantizedOperatorProjection();

/** Immutable projection for the final v1 dynamic-shape tranche. */
export const FINAL_OPERATOR_SHAPE_FUNCTIONS:
Readonly<Record<string, OperatorShapeFunctionId>> = finalOperatorProjection();

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
