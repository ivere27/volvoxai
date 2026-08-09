import type { RuntimeDType, TensorQuantization } from '../types.js';
import {
  OperatorShapeContractError,
  getOperatorShapeContract,
  proveOperatorShapeDomain,
  type LogicalOperatorTensorDescriptor,
  type OperatorAffineDimensionRelation,
  type OperatorShapeFunctionId,
  type OperatorTensorDescriptor,
} from '../ops/operatorShapeContracts.js';
import {
  PublicInputShapeContract,
  ShapeContractError,
  ShapeEnvironment,
  bindPublicInputShapes,
  canonicalShapeSignature,
  checkedShapeAdd,
  checkedShapeElementCount,
  checkedTensorByteLength,
  type ShapeDimensionSpec,
  type ShapedExecutionTensorView,
  type TensorShapeSpec,
} from '../ops/shapeSystem.js';
import {
  VOLVOX_LOGICAL_GRAPH_FORMAT,
  type AffineQuantizationReference,
  type Graph,
  type JsonValue,
  type NodeDescriptor,
  type TensorDescriptor,
} from './Graph.js';

export type ResolvedShapePlanErrorCode =
  | 'INVALID_GRAPH'
  | 'INPUT_BINDING_FAILED'
  | 'INVALID_BANK_RESIDENCY'
  | 'UNBOUND_SYMBOL'
  | 'UNKNOWN_OPERATOR'
  | 'OPERATOR_INFERENCE_FAILED'
  | 'OPERATOR_DOMAIN_UNSUPPORTED'
  | 'INPUT_PORT_MISMATCH'
  | 'OUTPUT_PORT_MISMATCH'
  | 'OUTPUT_DTYPE_MISMATCH'
  | 'OUTPUT_SHAPE_MISMATCH'
  | 'SYMBOL_CONFLICT'
  | 'BOUND_VIOLATION'
  | 'MULTIPLE_OF_VIOLATION'
  | 'QUANTIZATION_METADATA_MISSING'
  | 'QUANTIZATION_METADATA_UNEXPECTED'
  | 'QUANTIZATION_MISMATCH'
  | 'ARITHMETIC_OVERFLOW';

/** Stable graph-wide binding or proof failure raised before backend state exists. */
export class ResolvedShapePlanError extends Error {
  readonly code: ResolvedShapePlanErrorCode;
  readonly path: string;

  constructor(
    code: ResolvedShapePlanErrorCode,
    path: string,
    message: string,
    cause?: unknown,
  ) {
    super(`${path}: ${message}`, cause === undefined ? undefined : { cause });
    this.name = 'ResolvedShapePlanError';
    this.code = code;
    this.path = path;
  }
}

export type HydratedTensorQuantization = Readonly<Record<string, TensorQuantization>>;

const HYDRATED_QUANTIZATION_CACHE = new WeakMap<
  Graph,
  WeakMap<object, HydratedTensorQuantization>
>();
const IMMUTABLE_HYDRATION_GRAPHS = new WeakSet<Graph>();
const IMMUTABLE_HYDRATION_SOURCES = new WeakSet<object>();

export interface ResolvedTensorDescriptor {
  readonly name: string;
  readonly kind: 'input' | 'weight' | 'value';
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sizeBytes: number;
  readonly quantization?: TensorQuantization;
  readonly producerNodeId?: string;
  readonly producerPort?: string;
}

export interface ResolvedNodeDescriptor {
  readonly id: string;
  readonly opType: string;
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly inputs: Readonly<Record<string, ResolvedTensorDescriptor>>;
  readonly outputs: Readonly<Record<string, ResolvedTensorDescriptor>>;
  readonly params: Readonly<Record<string, JsonValue>>;
}

/**
 * Backend-independent concrete metadata for one exact public-input signature.
 * It deliberately has no caller data, Tensor storage, allocation, or buffer.
 */
export interface ResolvedShapePlan {
  readonly graphFingerprint: string;
  readonly signature: string;
  readonly symbols: Readonly<Record<string, number>>;
  readonly tensors: Readonly<Record<string, ResolvedTensorDescriptor>>;
  readonly nodes: readonly ResolvedNodeDescriptor[];
  /** Public outputs retain graph-declared order. */
  readonly outputs: readonly ResolvedTensorDescriptor[];
  readonly logicalActivationBytes: number;
  readonly weightBytes: number;
  /**
   * Resident global slot ids per declared bank, ascending. A bank absent from
   * this record is fully resident. Kernels see only the resident slice, so a
   * route to a non-resident slot is a bind-time or execution-time error.
   */
  readonly bankResidency: Readonly<Record<string, readonly number[]>>;
}

interface ResolvedShapePlanProvenance {
  readonly graph: Graph;
  readonly hydratedSource: HydratedTensorQuantization | undefined;
}

/* A plan's public shape is deliberately plain data, so backend SPI callers can
 * inspect it without depending on a class or a symbol property.  Keep the
 * canonical resolver provenance out-of-band instead: built-in materializers
 * can avoid re-proving a plan that this module just created, while copied or
 * forged lookalikes still take their complete defensive-validation path. */
const RESOLVED_SHAPE_PLAN_PROVENANCE = new WeakMap<
  ResolvedShapePlan,
  ResolvedShapePlanProvenance
>();

/** @internal Exact-identity provenance for built-in bound-graph materializers. */
export function hasCanonicalResolvedShapePlanProvenance(
  plan: ResolvedShapePlan,
  graph: Graph,
  hydratedSource: HydratedTensorQuantization | undefined,
): boolean {
  const provenance = RESOLVED_SHAPE_PLAN_PROVENANCE.get(plan);
  return provenance?.graph === graph && provenance.hydratedSource === hydratedSource;
}

/** @internal Unambiguous suffix shared by plan and materialization caches. */
export function canonicalBankResidencySuffix(
  residency: Readonly<Record<string, readonly number[]>>,
): string {
  const entries = sortedNames(residency).map((name) =>
    [name, residency[name]] as const);
  return entries.length === 0 ? '' : `|banks:${JSON.stringify(entries)}`;
}

export interface LogicalDomainTensorDescriptor {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: TensorShapeSpec;
  readonly quantization?: TensorQuantization;
}

export interface LogicalShapeDomainNodeProof {
  readonly id: string;
  readonly opType: string;
  readonly shapeFunctionId: OperatorShapeFunctionId;
  readonly inputs: Readonly<Record<string, LogicalDomainTensorDescriptor>>;
  readonly outputs: Readonly<Record<string, LogicalDomainTensorDescriptor>>;
  readonly facts: readonly string[];
}

export interface AcceptedGraphShapeDomainProof {
  readonly supported: true;
  readonly graphFingerprint: string;
  /** Output-only symbols map to the constant or source symbol that defines them. */
  readonly symbolRelations: Readonly<Record<string, ShapeDimensionSpec>>;
  /** Exact output-only affine definitions admitted by canonical operator proofs. */
  readonly affineSymbolRelations: Readonly<Record<
    string,
    Readonly<{ readonly source: string; readonly offset: number }>
  >>;
  readonly nodes: readonly LogicalShapeDomainNodeProof[];
  readonly outputs: readonly LogicalDomainTensorDescriptor[];
}

export interface RejectedGraphShapeDomainProof {
  readonly supported: false;
  readonly graphFingerprint: string;
  readonly code: ResolvedShapePlanErrorCode;
  readonly path: string;
  readonly reason: string;
}

export type GraphShapeDomainProof =
  | AcceptedGraphShapeDomainProof
  | RejectedGraphShapeDomainProof;

function fail(
  code: ResolvedShapePlanErrorCode,
  path: string,
  message: string,
  cause?: unknown,
): never {
  throw new ResolvedShapePlanError(code, path, message, cause);
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value != null && typeof value === 'object' && !Array.isArray(value) &&
    !ArrayBuffer.isView(value);
}

function hasOnlyFrozenDataFields(value: object): boolean {
  if (!Object.isFrozen(value) || Object.getOwnPropertySymbols(value).length !== 0) return false;
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== Array.prototype && prototype !== null) {
    return false;
  }
  for (const name of Object.getOwnPropertyNames(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, name);
    if (descriptor === undefined || !hasOwn(descriptor, 'value')) return false;
  }
  return true;
}

function isDeepFrozenData(value: unknown, seen = new WeakSet<object>()): boolean {
  if (value === null || typeof value !== 'object') return true;
  if (seen.has(value)) return true;
  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer ||
      !hasOnlyFrozenDataFields(value)) return false;
  seen.add(value);
  for (const name of Object.getOwnPropertyNames(value)) {
    if (Array.isArray(value) && name === 'length') continue;
    if (!isDeepFrozenData(Object.getOwnPropertyDescriptor(value, name)!.value, seen)) return false;
  }
  return true;
}

function hasImmutableHydrationInputs(graph: Graph, source: object): boolean {
  let immutableGraph = IMMUTABLE_HYDRATION_GRAPHS.has(graph);
  if (!immutableGraph) {
    immutableGraph = hasOnlyFrozenDataFields(graph) &&
      isDeepFrozenData(graph.tensors) && isDeepFrozenData(graph.quantization);
    if (immutableGraph) IMMUTABLE_HYDRATION_GRAPHS.add(graph);
  }
  let immutableSource = IMMUTABLE_HYDRATION_SOURCES.has(source);
  if (!immutableSource) {
    immutableSource = isDeepFrozenData(source);
    if (immutableSource) IMMUTABLE_HYDRATION_SOURCES.add(source);
  }
  return immutableGraph && immutableSource;
}

const UTF8_ENCODER = new TextEncoder();

/** Locale-independent unsigned UTF-8 ordering from the dynamic-v1 ADR. */
function compareCanonicalNames(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = UTF8_ENCODER.encode(left);
  const rightBytes = UTF8_ENCODER.encode(right);
  const shared = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < shared; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  const lengthDifference = leftBytes.length - rightBytes.length;
  if (lengthDifference !== 0) return lengthDifference;
  const codeUnits = Math.min(left.length, right.length);
  for (let index = 0; index < codeUnits; index++) {
    if (left.charCodeAt(index) !== right.charCodeAt(index)) {
      return left.charCodeAt(index) - right.charCodeAt(index);
    }
  }
  return left.length - right.length;
}

function sortedNames(value: object): string[] {
  return Object.getOwnPropertyNames(value).sort(compareCanonicalNames);
}

function freezeRecord<T>(entries: readonly (readonly [string, T])[]): Readonly<Record<string, T>> {
  return Object.freeze(Object.fromEntries(entries)) as Readonly<Record<string, T>>;
}

function stripErrorPath(error: { readonly message: string; readonly path: string }): string {
  const prefix = `${error.path}: `;
  return error.message.startsWith(prefix) ? error.message.slice(prefix.length) : error.message;
}

function assertGraph(graph: Graph): void {
  if (!isRecord(graph) || graph.format !== VOLVOX_LOGICAL_GRAPH_FORMAT ||
      !(graph.environment instanceof ShapeEnvironment) || !isRecord(graph.inputs) ||
      !isRecord(graph.weights) || !isRecord(graph.tensors) || !Array.isArray(graph.nodes) ||
      !Array.isArray(graph.outputs)) {
    fail('INVALID_GRAPH', 'logical graph', 'must be an immutable parsed Graph.');
  }
}

function cloneFrozenJson(value: JsonValue, path: string): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail('INVALID_GRAPH', path, 'contains a non-finite number.');
    return value;
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map((entry, index) => cloneFrozenJson(entry, `${path}[${index}]`)));
  }
  if (!isRecord(value)) fail('INVALID_GRAPH', path, 'contains a non-JSON value.');
  return freezeRecord(sortedNames(value).map((name) => [
    name,
    cloneFrozenJson(value[name] as JsonValue, `${path}.${name}`),
  ] as const));
}

function cloneNodeParams(
  params: Readonly<Record<string, JsonValue>>,
  path: string,
): Readonly<Record<string, JsonValue>> {
  return cloneFrozenJson(params, path) as Readonly<Record<string, JsonValue>>;
}

function quantizedRange(dtype: RuntimeDType, path: string): readonly [number, number] {
  if (dtype === 'int8') return [-128, 127];
  if (dtype === 'uint8') return [0, 255];
  fail('QUANTIZATION_MISMATCH', path, `tensor dtype '${dtype}' is not quantized storage.`);
}

function assertExactFields(
  value: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  path: string,
): void {
  const expectedSet = new Set(expected);
  const names = sortedNames(value);
  const missing = expected.filter((name) => !hasOwn(value, name));
  const unexpected = names.filter((name) => !expectedSet.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail('QUANTIZATION_MISMATCH', path, details.join('; '));
  }
}

function canonicalScale(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    fail('QUANTIZATION_MISMATCH', path, 'must be finite and positive.');
  }
  const result = Math.fround(value);
  if (!Number.isFinite(result) || result <= 0) {
    fail('QUANTIZATION_MISMATCH', path, 'must be positive and representable as float32.');
  }
  return result;
}

function canonicalZeroPoint(
  value: unknown,
  minimum: number,
  maximum: number,
  path: string,
): number {
  if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    fail(
      'QUANTIZATION_MISMATCH',
      path,
      `must be an integer in [${minimum}, ${maximum}].`,
    );
  }
  return value === 0 ? 0 : value as number;
}

function denseNumberArray(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value) || value.length === 0) {
    fail('QUANTIZATION_MISMATCH', path, 'must be a non-empty dense array.');
  }
  for (let index = 0; index < value.length; index++) {
    if (!hasOwn(value, index)) {
      fail('QUANTIZATION_MISMATCH', `${path}[${index}]`, 'must not be an array hole.');
    }
  }
  return value;
}

function cloneQuantization(
  source: TensorQuantization,
  reference: AffineQuantizationReference,
  tensor: TensorDescriptor,
  path: string,
): TensorQuantization {
  if (!isRecord(source)) {
    fail('QUANTIZATION_MISMATCH', path, 'must be a per_tensor or per_axis object.');
  }
  const [minimum, maximum] = quantizedRange(tensor.dtype, `${path}.dtype`);
  if (source.scheme !== reference.scheme) {
    fail(
      'QUANTIZATION_MISMATCH',
      `${path}.scheme`,
      `is '${String(source.scheme)}', but the logical reference requires '${reference.scheme}'.`,
    );
  }
  if (source.scheme === 'per_tensor') {
    assertExactFields(source, ['scheme', 'scale', 'zero_point'], path);
    return Object.freeze({
      scheme: 'per_tensor',
      scale: canonicalScale(source.scale, `${path}.scale`),
      zero_point: canonicalZeroPoint(source.zero_point, minimum, maximum, `${path}.zero_point`),
    });
  }
  if (reference.scheme !== 'per_axis') {
    fail('QUANTIZATION_MISMATCH', `${path}.scheme`, 'does not match the logical reference.');
  }

  assertExactFields(source, ['scheme', 'axis', 'scales', 'zero_points'], path);
  if (!Number.isInteger(source.axis) || source.axis !== reference.axis) {
    fail(
      'QUANTIZATION_MISMATCH',
      `${path}.axis`,
      `must equal the logical per-axis index ${reference.axis}.`,
    );
  }
  const extent = tensor.shape[reference.axis];
  if (typeof extent !== 'number') {
    fail(
      'QUANTIZATION_MISMATCH',
      `${path}.axis`,
      'targets a symbolic extent; dynamic per-axis quantization is not supported.',
    );
  }
  const rawScales = denseNumberArray(source.scales, `${path}.scales`);
  const rawZeroPoints = denseNumberArray(source.zero_points, `${path}.zero_points`);
  if (rawScales.length !== extent || rawZeroPoints.length !== extent) {
    fail(
      'QUANTIZATION_MISMATCH',
      path,
      `per-axis metadata must contain exactly ${extent} scales and zero points.`,
    );
  }
  const scales = Object.freeze(rawScales.map((scale, index) =>
    canonicalScale(scale, `${path}.scales[${index}]`)));
  const zeroPoints = Object.freeze(rawZeroPoints.map((zeroPoint, index) =>
    canonicalZeroPoint(zeroPoint, minimum, maximum, `${path}.zero_points[${index}]`)));
  return Object.freeze({
    scheme: 'per_axis',
    axis: reference.axis,
    scales,
    zero_points: zeroPoints,
  });
}

function normalizeHydratedQuantization(
  graph: Graph,
  source: HydratedTensorQuantization | undefined,
): HydratedTensorQuantization {
  const path = 'hydrated quantization';
  const values: Readonly<Record<string, TensorQuantization>> = source ?? Object.freeze({});
  if (!isRecord(values) || Object.getOwnPropertySymbols(values).length !== 0) {
    fail('QUANTIZATION_MISMATCH', path, 'must be a tensor-name record.');
  }
  /* Model snapshots deeply freeze both the logical quantization declaration
   * and its hydrated values. Once that exact identity has passed canonical
   * validation, reuse its immutable clone for all concrete shape variants.
   * Arbitrary mutable callers continue through the full validation path. */
  const cacheable = hasImmutableHydrationInputs(graph, values);
  const cached = cacheable
    ? HYDRATED_QUANTIZATION_CACHE.get(graph)?.get(values)
    : undefined;
  if (cached !== undefined) return cached;
  const references = graph.quantization?.tensors ?? Object.freeze({});
  const expectedNames = sortedNames(references);
  const actualNames = sortedNames(values);
  const expected = new Set(expectedNames);
  const actual = new Set(actualNames);
  const missing = expectedNames.filter((name) => !actual.has(name));
  const unexpected = actualNames.filter((name) => !expected.has(name));
  if (missing.length !== 0) {
    fail(
      'QUANTIZATION_METADATA_MISSING',
      path,
      `is missing target metadata for [${missing.join(', ')}].`,
    );
  }
  if (unexpected.length !== 0) {
    fail(
      'QUANTIZATION_METADATA_UNEXPECTED',
      path,
      `contains undeclared target metadata for [${unexpected.join(', ')}].`,
    );
  }
  const normalized = freezeRecord(expectedNames.map((name) => {
    const tensor = graph.tensors[name];
    if (tensor === undefined) {
      fail('INVALID_GRAPH', `quantization.tensors.${name}`, 'targets a missing logical tensor.');
    }
    return [
      name,
      cloneQuantization(values[name], references[name], tensor, `${path}.${name}`),
    ] as const;
  }));
  if (cacheable) {
    let graphCache = HYDRATED_QUANTIZATION_CACHE.get(graph);
    if (graphCache === undefined) {
      graphCache = new WeakMap<object, HydratedTensorQuantization>();
      HYDRATED_QUANTIZATION_CACHE.set(graph, graphCache);
    }
    graphCache.set(values, normalized);
    graphCache.set(normalized, normalized);
    IMMUTABLE_HYDRATION_SOURCES.add(normalized);
  }
  return normalized;
}

function quantizationEqual(
  left: TensorQuantization | null | undefined,
  right: TensorQuantization | null | undefined,
): boolean {
  if (left == null || right == null) return left == null && right == null;
  if (left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor' && right.scheme === 'per_tensor') {
    return left.scale === right.scale && left.zero_point === right.zero_point;
  }
  if (left.scheme !== 'per_axis' || right.scheme !== 'per_axis') return false;
  return left.axis === right.axis && left.scales.length === right.scales.length &&
    left.zero_points.length === right.zero_points.length &&
    left.scales.every((value, index) => value === right.scales[index]) &&
    left.zero_points.every((value, index) => value === right.zero_points[index]);
}

function validateOutputQuantization(
  tensorName: string,
  inferred: TensorQuantization | null | undefined,
  hydrated: HydratedTensorQuantization,
  path: string,
): TensorQuantization | undefined {
  const expected = hasOwn(hydrated, tensorName) ? hydrated[tensorName] : undefined;
  if (!quantizationEqual(inferred, expected)) {
    fail(
      'QUANTIZATION_MISMATCH',
      path,
      expected === undefined
        ? `operator inferred undeclared quantization for tensor '${tensorName}'.`
        : `operator quantization does not match hydrated metadata for tensor '${tensorName}'.`,
    );
  }
  return expected;
}

function checkedElementAndBytes(
  shape: readonly number[],
  dtype: RuntimeDType,
  path: string,
): readonly [number, number] {
  try {
    return [
      checkedShapeElementCount(shape, `${path}.shape`),
      checkedTensorByteLength(shape, dtype, path),
    ];
  } catch (error) {
    if (error instanceof ShapeContractError) {
      fail(
        error.code === 'ARITHMETIC_OVERFLOW' ? 'ARITHMETIC_OVERFLOW' : 'INVALID_GRAPH',
        error.path,
        stripErrorPath(error),
        error,
      );
    }
    throw error;
  }
}

function resolvedTensor(
  source: TensorDescriptor,
  shape: readonly number[],
  quantization: TensorQuantization | undefined,
  path: string,
): ResolvedTensorDescriptor {
  const concreteShape = Object.freeze([...shape]);
  const [elementCount, sizeBytes] = checkedElementAndBytes(concreteShape, source.dtype, path);
  const descriptor: ResolvedTensorDescriptor = source.kind === 'value'
    ? {
      name: source.name,
      kind: source.kind,
      dtype: source.dtype,
      shape: concreteShape,
      elementCount,
      sizeBytes,
      ...(quantization === undefined ? {} : { quantization }),
      producerNodeId: source.producerNodeId,
      producerPort: source.producerPort,
    }
    : {
      name: source.name,
      kind: source.kind,
      dtype: source.dtype,
      shape: concreteShape,
      elementCount,
      sizeBytes,
      ...(quantization === undefined ? {} : { quantization }),
    };
  return Object.freeze(descriptor);
}

function createPublicInputContract(graph: Graph): PublicInputShapeContract {
  try {
    return new PublicInputShapeContract(
      graph.environment,
      Object.values(graph.inputs).map((input) => ({
        name: input.name,
        dtype: input.dtype,
        shape: input.shape,
      })),
    );
  } catch (error) {
    if (error instanceof ShapeContractError) {
      fail('INVALID_GRAPH', error.path, stripErrorPath(error), error);
    }
    throw error;
  }
}

function bindInputs(
  graph: Graph,
  values: Readonly<Record<string, ShapedExecutionTensorView>>,
) {
  const contract = createPublicInputContract(graph);
  try {
    return bindPublicInputShapes(contract, values);
  } catch (error) {
    if (error instanceof ShapeContractError) {
      const mapped: ResolvedShapePlanErrorCode = error.code === 'ARITHMETIC_OVERFLOW'
        ? 'ARITHMETIC_OVERFLOW'
        : error.code === 'BOUND_VIOLATION'
          ? 'BOUND_VIOLATION'
          : error.code === 'MULTIPLE_OF_VIOLATION'
            ? 'MULTIPLE_OF_VIOLATION'
            : error.code === 'SYMBOL_CONFLICT'
              ? 'SYMBOL_CONFLICT'
              : 'INPUT_BINDING_FAILED';
      fail(mapped, error.path, stripErrorPath(error), error);
    }
    throw error;
  }
}

function assertSymbolsBoundForInput(
  descriptor: TensorDescriptor,
  symbols: ReadonlyMap<string, number>,
  path: string,
): void {
  for (let axis = 0; axis < descriptor.shape.length; axis++) {
    const dimension = descriptor.shape[axis];
    if (typeof dimension === 'string' && !symbols.has(dimension)) {
      fail(
        'UNBOUND_SYMBOL',
        `${path}.shape[${axis}]`,
        `uses symbol '${dimension}' before an input or earlier output binds it.`,
      );
    }
  }
}

function validateConcreteSymbol(
  environment: ShapeEnvironment,
  symbol: string,
  value: number,
  path: string,
): void {
  const constraint = environment.get(symbol);
  if (constraint === undefined) {
    fail('INVALID_GRAPH', path, `references undeclared symbol '${symbol}'.`);
  }
  if (value < constraint.min || value > constraint.max) {
    fail(
      'BOUND_VIOLATION',
      path,
      `binds '${symbol}' to ${value}, outside [${constraint.min}, ${constraint.max}].`,
    );
  }
  const multiple = constraint.multiple_of ?? 1;
  if (value % multiple !== 0) {
    fail(
      'MULTIPLE_OF_VIOLATION',
      path,
      `binds '${symbol}' to ${value}, which is not a multiple of ${multiple}.`,
    );
  }
}

function assertConcreteOutputShape(
  inferred: readonly number[],
  assertion: TensorShapeSpec,
  environment: ShapeEnvironment,
  symbols: Map<string, number>,
  path: string,
): void {
  if (inferred.length !== assertion.length) {
    fail(
      'OUTPUT_SHAPE_MISMATCH',
      path,
      `inferred rank ${inferred.length}, but the graph asserts rank ${assertion.length}.`,
    );
  }
  for (let axis = 0; axis < assertion.length; axis++) {
    const value = inferred[axis];
    const expected = assertion[axis];
    const axisPath = `${path}[${axis}]`;
    if (typeof expected === 'number') {
      if (value !== expected) {
        fail(
          'OUTPUT_SHAPE_MISMATCH',
          axisPath,
          `inferred ${value}, but the graph asserts constant ${expected}.`,
        );
      }
      continue;
    }
    validateConcreteSymbol(environment, expected, value, axisPath);
    const previous = symbols.get(expected);
    if (previous !== undefined && previous !== value) {
      fail(
        'SYMBOL_CONFLICT',
        axisPath,
        `inferred '${expected}' as ${value}, but it was already bound to ${previous}.`,
      );
    }
    if (previous === undefined) symbols.set(expected, value);
  }
}

function assertExactPorts(
  actual: object,
  expected: readonly string[],
  path: string,
  variadic?: Readonly<{ readonly prefix: string; readonly minimum: number }>,
): void {
  const actualNames = sortedNames(actual);
  const expectedNames = [...expected].sort(compareCanonicalNames);
  if (variadic === undefined) {
    if (actualNames.length === expectedNames.length &&
        actualNames.every((name, index) => name === expectedNames[index])) return;
    fail(
      'OUTPUT_PORT_MISMATCH',
      path,
      `declares [${actualNames.join(', ')}], but the shape contract requires exactly [${expectedNames.join(', ')}].`,
    );
  }
  assertNormalizedVariadicPorts(
    actualNames,
    expectedNames,
    [],
    variadic,
    path,
    'OUTPUT_PORT_MISMATCH',
  );
}

function assertNormalizedVariadicPorts(
  actualNames: readonly string[],
  required: readonly string[],
  optional: readonly string[],
  variadic: Readonly<{ readonly prefix: string; readonly minimum: number }>,
  path: string,
  code: 'INPUT_PORT_MISMATCH' | 'OUTPUT_PORT_MISMATCH',
): void {
  const fixed = new Set([...required, ...optional]);
  const missing = required.filter((name) => !actualNames.includes(name));
  const numbered: Array<readonly [number, string]> = [];
  const unexpected: string[] = [];
  for (const name of actualNames) {
    if (fixed.has(name)) continue;
    const suffix = name.startsWith(variadic.prefix)
      ? name.slice(variadic.prefix.length)
      : '';
    if (!/^(0|[1-9][0-9]*)$/.test(suffix)) {
      unexpected.push(name);
      continue;
    }
    const index = Number(suffix);
    if (!Number.isSafeInteger(index)) {
      unexpected.push(name);
      continue;
    }
    numbered.push([index, name]);
  }
  numbered.sort(([left], [right]) => left - right);
  const hasGap = numbered.some(([index], position) => index !== position);
  if (missing.length === 0 && unexpected.length === 0 && !hasGap &&
      numbered.length >= variadic.minimum) return;
  const details: string[] = [];
  if (missing.length !== 0) details.push(`missing fixed [${missing.join(', ')}]`);
  if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
  if (hasGap) details.push(`'${variadic.prefix}N' ports must be contiguous from ${variadic.prefix}0`);
  if (numbered.length < variadic.minimum) {
    details.push(`requires at least ${variadic.minimum} '${variadic.prefix}N' ports`);
  }
  fail(code, path, details.join('; '));
}

function assertVariadicInputPorts(
  actual: object,
  required: readonly string[],
  optional: readonly string[],
  variadic: Readonly<{ readonly prefix: string; readonly minimum: number }> | undefined,
  path: string,
): void {
  if (variadic === undefined) return;
  assertNormalizedVariadicPorts(
    sortedNames(actual),
    required,
    optional,
    variadic,
    path,
    'INPUT_PORT_MISMATCH',
  );
}

function concreteOperatorInputs(
  graph: Graph,
  node: NodeDescriptor,
  nodeIndex: number,
  tensors: ReadonlyMap<string, ResolvedTensorDescriptor>,
  symbols: ReadonlyMap<string, number>,
): Readonly<Record<string, ResolvedTensorDescriptor>> {
  const entries = sortedNames(node.inputs).map((port) => {
    const tensorName = node.inputs[port];
    const logical = graph.tensors[tensorName];
    if (logical === undefined) {
      fail('INVALID_GRAPH', `nodes[${nodeIndex}].inputs.${port}`, `references missing tensor '${tensorName}'.`);
    }
    assertSymbolsBoundForInput(logical, symbols, `nodes[${nodeIndex}].inputs.${port}`);
    const concrete = tensors.get(tensorName);
    if (concrete === undefined) {
      fail(
        'INVALID_GRAPH',
        `nodes[${nodeIndex}].inputs.${port}`,
        `tensor '${tensorName}' has no earlier concrete descriptor.`,
      );
    }
    return [port, concrete] as const;
  });
  return freezeRecord(entries);
}

function toOperatorDescriptor(
  tensor: ResolvedTensorDescriptor,
): OperatorTensorDescriptor {
  return tensor.quantization === undefined
    ? Object.freeze({ shape: tensor.shape, dtype: tensor.dtype })
    : Object.freeze({
      shape: tensor.shape,
      dtype: tensor.dtype,
      quantization: tensor.quantization,
    });
}

function operatorInputDescriptors(
  inputs: Readonly<Record<string, ResolvedTensorDescriptor>>,
): Readonly<Record<string, OperatorTensorDescriptor>> {
  return freezeRecord(sortedNames(inputs).map((port) => [
    port,
    toOperatorDescriptor(inputs[port]),
  ] as const));
}

function provisionalConcreteShape(
  assertion: TensorShapeSpec,
  symbols: ReadonlyMap<string, number>,
  inferredFrom: readonly number[] | undefined,
  path: string,
): readonly number[] {
  return Object.freeze(assertion.map((dimension, axis) => {
    if (typeof dimension === 'number') return dimension;
    const bound = symbols.get(dimension);
    if (bound !== undefined) return bound;
    if (inferredFrom !== undefined && inferredFrom.length === assertion.length) {
      return inferredFrom[axis];
    }
    fail(
      'UNBOUND_SYMBOL',
      `${path}[${axis}]`,
      `cannot construct an operator assertion for unbound symbol '${dimension}'.`,
    );
  }));
}

const QUANTIZED_DECLARED_OUTPUT_OPERATORS: ReadonlySet<string> = new Set([
  'QLinear', 'QMatMul', 'QGemm', 'QBatchMatMul', 'QConv2D', 'QAdd',
  'QEmbedding', 'QGELU', 'QSiLU', 'QLayerNorm', 'QGroupNorm',
  'QMaskedMean', 'QSDPA', 'RequantizeLinear',
]);

const DECLARED_OUTPUT_OPERATORS: ReadonlySet<string> = new Set([
  'QuantizeLinear', 'Reshape', 'Expand', 'Broadcast', 'Resize',
  'ResizeNearest2D',
  ...QUANTIZED_DECLARED_OUTPUT_OPERATORS,
]);

const LOGICAL_DECLARED_OUTPUT_OPERATORS: ReadonlySet<string> = new Set([
  ...DECLARED_OUTPUT_OPERATORS,
  // Concat concrete inference is always exact. Its bounded proof alone needs
  // the graph-authored output-only symbol for the affine M=Q+C relation.
  'Concat',
]);

function concreteDeclaredOutputs(
  node: NodeDescriptor,
  nodeIndex: number,
  inputs: Readonly<Record<string, ResolvedTensorDescriptor>>,
  symbols: ReadonlyMap<string, number>,
  hydrated: HydratedTensorQuantization,
): Readonly<Record<string, OperatorTensorDescriptor>> | undefined {
  if (!DECLARED_OUTPUT_OPERATORS.has(node.opType)) return undefined;
  const inputShape = inputs.input?.shape;
  return freezeRecord(sortedNames(node.outputs).map((port) => {
    const output = node.outputs[port];
    const shape = provisionalConcreteShape(
      output.shape,
      symbols,
      inputShape,
      `nodes[${nodeIndex}].outputs.${port}.shape`,
    );
    const quantization = hasOwn(hydrated, output.tensor)
      ? hydrated[output.tensor]
      : undefined;
    const descriptor: OperatorTensorDescriptor = quantization === undefined
      ? { shape, dtype: output.dtype }
      : { shape, dtype: output.dtype, quantization };
    return [port, Object.freeze(descriptor)] as const;
  }));
}

function inferNode(
  node: NodeDescriptor,
  nodeIndex: number,
  inputs: Readonly<Record<string, ResolvedTensorDescriptor>>,
  symbols: ReadonlyMap<string, number>,
  hydrated: HydratedTensorQuantization,
) {
  let contract;
  try {
    contract = getOperatorShapeContract(node.opType);
  } catch (error) {
    if (error instanceof OperatorShapeContractError) {
      fail('UNKNOWN_OPERATOR', `nodes[${nodeIndex}].opType`, stripErrorPath(error), error);
    }
    throw error;
  }
  assertVariadicInputPorts(
    node.inputs,
    contract.ports.requiredInputs,
    contract.ports.optionalInputs,
    contract.ports.variadicInputs,
    `nodes[${nodeIndex}].inputs`,
  );
  assertExactPorts(
    node.outputs,
    contract.ports.outputs,
    `nodes[${nodeIndex}].outputs`,
    contract.ports.variadicOutputs,
  );
  const declaredOutputs = concreteDeclaredOutputs(
    node,
    nodeIndex,
    inputs,
    symbols,
    hydrated,
  );
  try {
    return Object.freeze({
      contract,
      outputs: contract.inferConcrete({
        inputs: operatorInputDescriptors(inputs),
        params: node.params,
        ...(declaredOutputs === undefined ? {} : { declaredOutputs }),
      }),
    });
  } catch (error) {
    if (error instanceof OperatorShapeContractError) {
      fail(
        'OPERATOR_INFERENCE_FAILED',
        `nodes[${nodeIndex}]`,
        `${error.code}: ${error.message}`,
        error,
      );
    }
    throw error;
  }
}

function addBytes(total: number, value: number, path: string): number {
  try {
    return checkedShapeAdd(total, value, path);
  } catch (error) {
    if (error instanceof ShapeContractError) {
      fail('ARITHMETIC_OVERFLOW', error.path, stripErrorPath(error), error);
    }
    throw error;
  }
}

/** Validate a caller-supplied resident slot set against the declared banks. */
function normalizeBankResidency(
  graph: Graph,
  requested: Readonly<Record<string, readonly number[]>> | undefined,
): Readonly<Record<string, readonly number[]>> {
  if (requested === undefined) return freezeRecord([]);
  if (requested === null || typeof requested !== 'object' || Array.isArray(requested)) {
    fail('INVALID_BANK_RESIDENCY', 'bankResidency', 'must be an object.');
  }
  const entries: Array<readonly [string, readonly number[]]> = [];
  for (const name of sortedNames(requested)) {
    const path = `bankResidency.${name}`;
    const bank = hasOwn(graph.banks, name) ? graph.banks[name] : undefined;
    if (bank === undefined) fail('INVALID_BANK_RESIDENCY', path, `names no declared bank.`);
    const slots = requested[name];
    if (!Array.isArray(slots) || slots.length === 0) {
      fail('INVALID_BANK_RESIDENCY', path, 'must be a non-empty array of slot ids.');
    }
    const available = graph.weights[name].shape[0];
    let previous = -1;
    for (let index = 0; index < slots.length; index++) {
      const slot = slots[index];
      if (!Number.isSafeInteger(slot) || slot < 0 || slot >= available) {
        fail('INVALID_BANK_RESIDENCY', `${path}[${index}]`,
          `must be a slot id in [0, ${available - 1}].`);
      }
      if (slot <= previous) {
        fail('INVALID_BANK_RESIDENCY', `${path}[${index}]`,
          'must be strictly ascending and unique.');
      }
      previous = slot;
    }
    entries.push([name, Object.freeze([...slots])]);
  }
  return freezeRecord(entries);
}

interface ConcreteInputShapeBinding {
  readonly inputs: Readonly<Record<string, { readonly shape: readonly number[] }>>;
  readonly symbols: Readonly<Record<string, number>>;
  readonly signature: string;
}

function resolveBoundGraphShapesInternal(
  graph: Graph,
  binding: ConcreteInputShapeBinding,
  hydratedSource?: HydratedTensorQuantization,
  requestedResidency?: Readonly<Record<string, readonly number[]>>,
): ResolvedShapePlan {
  assertGraph(graph);
  const hydrated = normalizeHydratedQuantization(graph, hydratedSource);
  const residency = normalizeBankResidency(graph, requestedResidency);
  const symbols = new Map<string, number>(Object.entries(binding.symbols));
  const tensors = new Map<string, ResolvedTensorDescriptor>();

  // Copy only descriptor fields out of the concrete binding. Dynamic caller
  // data is never read or retained beyond the preceding validation step.
  for (const name of sortedNames(graph.inputs)) {
    const logical = graph.inputs[name];
    const bound = binding.inputs[name];
    if (bound === undefined) {
      fail('INVALID_GRAPH', `inputs.${name}`, 'has no bound public-input descriptor.');
    }
    tensors.set(name, resolvedTensor(
      logical,
      bound.shape,
      hasOwn(hydrated, name) ? hydrated[name] : undefined,
      `tensor '${name}'`,
    ));
  }
  for (const name of sortedNames(graph.weights)) {
    const logical = graph.weights[name];
    const resident = hasOwn(residency, name) ? residency[name] : undefined;
    let quantization = hasOwn(hydrated, name) ? hydrated[name] : undefined;
    if (resident !== undefined && quantization !== undefined &&
        quantization.scheme === 'per_axis' && quantization.axis === 0) {
      // Per-axis metadata on the slot axis is itself indexed by slot, so it
      // has to be sliced alongside the payload.
      const perAxis = quantization;
      quantization = Object.freeze({
        scheme: 'per_axis',
        axis: 0,
        scales: Object.freeze(resident.map((slot) => perAxis.scales[slot])),
        zero_points: Object.freeze(resident.map((slot) => perAxis.zero_points[slot])),
      });
    }
    tensors.set(name, resolvedTensor(
      logical,
      resident === undefined
        ? logical.shape
        : [resident.length, ...logical.shape.slice(1)],
      quantization,
      `tensor '${name}'`,
    ));
  }

  const nodes: ResolvedNodeDescriptor[] = [];
  for (let nodeIndex = 0; nodeIndex < graph.nodes.length; nodeIndex++) {
    const node = graph.nodes[nodeIndex];
    const inputs = concreteOperatorInputs(graph, node, nodeIndex, tensors, symbols);
    const inferred = inferNode(node, nodeIndex, inputs, symbols, hydrated);
    assertExactPorts(
      inferred.outputs,
      inferred.contract.ports.outputs,
      `nodes[${nodeIndex}] inferred outputs`,
      inferred.contract.ports.variadicOutputs,
    );
    assertExactPorts(
      inferred.outputs,
      sortedNames(node.outputs),
      `nodes[${nodeIndex}] inferred outputs`,
    );

    const outputEntries: Array<readonly [string, ResolvedTensorDescriptor]> = [];
    for (const port of sortedNames(node.outputs)) {
      const assertion = node.outputs[port];
      const output = inferred.outputs[port];
      if (output === undefined) {
        fail('OUTPUT_PORT_MISMATCH', `nodes[${nodeIndex}].outputs.${port}`, 'was not inferred.');
      }
      if (output.dtype !== assertion.dtype) {
        fail(
          'OUTPUT_DTYPE_MISMATCH',
          `nodes[${nodeIndex}].outputs.${port}.dtype`,
          `operator inferred '${output.dtype}', but the graph asserts '${assertion.dtype}'.`,
        );
      }
      assertConcreteOutputShape(
        output.shape,
        assertion.shape,
        graph.environment,
        symbols,
        `nodes[${nodeIndex}].outputs.${port}.shape`,
      );
      const quantization = validateOutputQuantization(
        assertion.tensor,
        output.quantization,
        hydrated,
        `nodes[${nodeIndex}].outputs.${port}.quantization`,
      );
      const logical = graph.tensors[assertion.tensor];
      if (logical === undefined || logical.kind !== 'value') {
        fail(
          'INVALID_GRAPH',
          `nodes[${nodeIndex}].outputs.${port}.tensor`,
          `has no logical value descriptor for '${assertion.tensor}'.`,
        );
      }
      const descriptor = resolvedTensor(
        logical,
        output.shape,
        quantization,
        `tensor '${assertion.tensor}'`,
      );
      tensors.set(assertion.tensor, descriptor);
      outputEntries.push([port, descriptor]);
    }

    nodes.push(Object.freeze({
      id: node.id,
      opType: node.opType,
      shapeFunctionId: inferred.contract.shapeFunctionId,
      inputs,
      outputs: freezeRecord(outputEntries),
      params: cloneNodeParams(node.params, `nodes[${nodeIndex}].params`),
    }));
  }

  for (const name of sortedNames(graph.tensors)) {
    if (!tensors.has(name)) {
      fail('INVALID_GRAPH', `tensors.${name}`, 'was not resolved by topological inference.');
    }
  }

  let logicalActivationBytes = 0;
  let weightBytes = 0;
  const tensorEntries = [...tensors.entries()].sort(([left], [right]) =>
    compareCanonicalNames(left, right));
  for (const [name, descriptor] of tensorEntries) {
    if (descriptor.kind === 'weight') {
      weightBytes = addBytes(weightBytes, descriptor.sizeBytes, `weight byte total at '${name}'`);
    } else {
      logicalActivationBytes = addBytes(
        logicalActivationBytes,
        descriptor.sizeBytes,
        `logical activation byte total at '${name}'`,
      );
    }
  }

  const outputs = Object.freeze(graph.outputs.map((name, index) => {
    const descriptor = tensors.get(name);
    if (descriptor === undefined) {
      fail('INVALID_GRAPH', `outputs[${index}]`, `references unresolved tensor '${name}'.`);
    }
    return descriptor;
  }));
  const symbolEntries = [...symbols.entries()].sort(([left], [right]) =>
    compareCanonicalNames(left, right));
  const plan: ResolvedShapePlan = Object.freeze({
    graphFingerprint: graph.fingerprint,
    // Two contexts with different resident slots must not share a plan.
    signature: `${binding.signature}${canonicalBankResidencySuffix(residency)}`,
    symbols: freezeRecord(symbolEntries),
    tensors: freezeRecord(tensorEntries),
    nodes: Object.freeze(nodes),
    outputs,
    logicalActivationBytes,
    weightBytes,
    bankResidency: residency,
  });
  RESOLVED_SHAPE_PLAN_PROVENANCE.set(plan, Object.freeze({
    graph,
    hydratedSource,
  }));
  return plan;
}

function resolveGraphShapesInternal(
  graph: Graph,
  values: Readonly<Record<string, ShapedExecutionTensorView>>,
  hydratedSource?: HydratedTensorQuantization,
  bankResidency?: Readonly<Record<string, readonly number[]>>,
): ResolvedShapePlan {
  assertGraph(graph);
  return resolveBoundGraphShapesInternal(
    graph,
    bindInputs(graph, values),
    hydratedSource,
    bankResidency,
  );
}

function publicSymbolLegalRange(
  graph: Graph,
  symbol: string,
  path: string,
): Readonly<{ readonly first: number; readonly last: number }> {
  const constraint = graph.environment.get(symbol);
  if (constraint === undefined) {
    fail('INVALID_GRAPH', path, `references undeclared symbol '${symbol}'.`);
  }
  const multiple = constraint.multiple_of ?? 1;
  const remainder = constraint.min % multiple;
  const first = constraint.min + (remainder === 0 ? 0 : multiple - remainder);
  const last = constraint.max - (constraint.max % multiple);
  if (first > last) {
    fail(
      'INPUT_BINDING_FAILED',
      path,
      `public symbol '${symbol}' has no legal multiple in ` +
      `[${constraint.min}, ${constraint.max}].`,
    );
  }
  return Object.freeze({ first, last });
}

function singletonPublicSymbolValue(
  graph: Graph,
  symbol: string,
  path: string,
): number {
  const { first, last } = publicSymbolLegalRange(graph, symbol, path);
  if (first !== last) {
    fail(
      'INPUT_BINDING_FAILED',
      path,
      `public symbol '${symbol}' has more than one legal value.`,
    );
  }
  return first;
}

function minimumPublicSymbolValue(
  graph: Graph,
  symbol: string,
  path: string,
): number {
  return publicSymbolLegalRange(graph, symbol, path).first;
}

function createStaticInputShapeBinding(graph: Graph): ConcreteInputShapeBinding {
  const symbols = new Map<string, number>();
  const inputEntries = sortedNames(graph.inputs).map((name) => {
    const input = graph.inputs[name];
    const shape = Object.freeze(input.shape.map((dimension, axis) => {
      if (typeof dimension === 'number') return dimension;
      const value = singletonPublicSymbolValue(
        graph,
        dimension,
        `inputs.${name}.shape[${axis}]`,
      );
      const previous = symbols.get(dimension);
      if (previous !== undefined && previous !== value) {
        fail(
          'SYMBOL_CONFLICT',
          `inputs.${name}.shape[${axis}]`,
          `public symbol '${dimension}' resolves inconsistently to ${previous} and ${value}.`,
        );
      }
      symbols.set(dimension, value);
      return value;
    }));
    return [name, Object.freeze({ shape })] as const;
  });
  const inputs = freezeRecord(inputEntries);
  const symbolEntries = [...symbols.entries()].sort(([left], [right]) =>
    compareCanonicalNames(left, right));
  return Object.freeze({
    inputs,
    symbols: freezeRecord(symbolEntries),
    signature: canonicalShapeSignature(
      sortedNames(inputs).map((name) => Object.freeze({ name, shape: inputs[name].shape })),
    ),
  });
}

function createMinimumInputShapeBinding(graph: Graph): ConcreteInputShapeBinding {
  const symbols = new Map<string, number>();
  const inputEntries = sortedNames(graph.inputs).map((name) => {
    const input = graph.inputs[name];
    const shape = Object.freeze(input.shape.map((dimension, axis) => {
      if (typeof dimension === 'number') return dimension;
      const value = minimumPublicSymbolValue(
        graph,
        dimension,
        `inputs.${name}.shape[${axis}]`,
      );
      const previous = symbols.get(dimension);
      if (previous !== undefined && previous !== value) {
        fail(
          'SYMBOL_CONFLICT',
          `inputs.${name}.shape[${axis}]`,
          `public symbol '${dimension}' resolves inconsistently to ${previous} and ${value}.`,
        );
      }
      symbols.set(dimension, value);
      return value;
    }));
    return [name, Object.freeze({ shape })] as const;
  });
  const inputs = freezeRecord(inputEntries);
  const symbolEntries = [...symbols.entries()].sort(([left], [right]) =>
    compareCanonicalNames(left, right));
  return Object.freeze({
    inputs,
    symbols: freezeRecord(symbolEntries),
    signature: canonicalShapeSignature(
      sortedNames(inputs).map((name) => Object.freeze({ name, shape: inputs[name].shape })),
    ),
  });
}

/**
 * Atomically bind one complete set of shaped public inputs and resolve all
 * concrete tensor/operator descriptors. No backend state or caller storage is
 * touched or retained.
 */
export function resolveGraphShapes(
  graph: Graph,
  values: Readonly<Record<string, ShapedExecutionTensorView>>,
  hydratedQuantization?: HydratedTensorQuantization,
  bankResidency?: Readonly<Record<string, readonly number[]>>,
): ResolvedShapePlan {
  try {
    return resolveGraphShapesInternal(
      graph, values, hydratedQuantization, bankResidency,
    );
  } catch (error) {
    if (error instanceof ResolvedShapePlanError) throw error;
    if (error instanceof ShapeContractError) {
      fail(
        error.code === 'ARITHMETIC_OVERFLOW' ? 'ARITHMETIC_OVERFLOW' : 'INVALID_GRAPH',
        error.path,
        stripErrorPath(error),
        error,
      );
    }
    if (error instanceof OperatorShapeContractError) {
      fail('OPERATOR_INFERENCE_FAILED', error.path, `${error.code}: ${error.message}`, error);
    }
    throw error;
  }
}

/**
 * Resolve the sole public-input shape of a fixed-domain logical graph without
 * allocating request-sized typed arrays. This is the metadata-only source for
 * constant-shape snapshot prebinding; any public symbol with multiple legal
 * values is rejected instead of being sampled.
 */
export function resolveStaticGraphShapes(
  graph: Graph,
  hydratedQuantization?: HydratedTensorQuantization,
): ResolvedShapePlan {
  try {
    assertGraph(graph);
    return resolveBoundGraphShapesInternal(
      graph,
      createStaticInputShapeBinding(graph),
      hydratedQuantization,
    );
  } catch (error) {
    if (error instanceof ResolvedShapePlanError) throw error;
    if (error instanceof ShapeContractError) {
      fail(
        error.code === 'ARITHMETIC_OVERFLOW' ? 'ARITHMETIC_OVERFLOW' : 'INVALID_GRAPH',
        error.path,
        stripErrorPath(error),
        error,
      );
    }
    if (error instanceof OperatorShapeContractError) {
      fail('OPERATOR_INFERENCE_FAILED', error.path, `${error.code}: ${error.message}`, error);
    }
    throw error;
  }
}

/**
 * Resolve one metadata-only preload plan at the smallest legal public shape.
 * This uses the same canonical concrete inference and bank-residency path as
 * an execution request; it allocates no request-sized typed arrays and does
 * not authorize execution with synthetic input values.
 */
export function resolveMinimumGraphShapes(
  graph: Graph,
  hydratedQuantization?: HydratedTensorQuantization,
  bankResidency?: Readonly<Record<string, readonly number[]>>,
): ResolvedShapePlan {
  try {
    assertGraph(graph);
    return resolveBoundGraphShapesInternal(
      graph,
      createMinimumInputShapeBinding(graph),
      hydratedQuantization,
      bankResidency,
    );
  } catch (error) {
    if (error instanceof ResolvedShapePlanError) throw error;
    if (error instanceof ShapeContractError) {
      fail(
        error.code === 'ARITHMETIC_OVERFLOW' ? 'ARITHMETIC_OVERFLOW' : 'INVALID_GRAPH',
        error.path,
        stripErrorPath(error),
        error,
      );
    }
    if (error instanceof OperatorShapeContractError) {
      fail('OPERATOR_INFERENCE_FAILED', error.path, `${error.code}: ${error.message}`, error);
    }
    throw error;
  }
}

type SymbolRelations = Map<string, ShapeDimensionSpec>;
type AffineSymbolRelations = Map<string, OperatorAffineDimensionRelation>;

function fixedDimension(
  dimension: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): number | undefined {
  if (typeof dimension === 'number') return dimension;
  const constraint = environment.get(dimension);
  return constraint !== undefined && constraint.min === constraint.max
    ? constraint.min
    : undefined;
}

function dimensionsProvablyEqual(
  left: ShapeDimensionSpec,
  right: ShapeDimensionSpec,
  environment: ShapeEnvironment,
): boolean {
  if (left === right) return true;
  const leftFixed = fixedDimension(left, environment);
  const rightFixed = fixedDimension(right, environment);
  return leftFixed !== undefined && rightFixed !== undefined && leftFixed === rightFixed;
}

function legalInterval(
  environment: ShapeEnvironment,
  symbol: string,
  path: string,
): readonly [number, number, number] {
  const constraint = environment.get(symbol);
  if (constraint === undefined) fail('INVALID_GRAPH', path, `references undeclared symbol '${symbol}'.`);
  const multiple = constraint.multiple_of ?? 1;
  const remainder = constraint.min % multiple;
  const first = constraint.min + (remainder === 0 ? 0 : multiple - remainder);
  const last = constraint.max - (constraint.max % multiple);
  if (first > last) fail('INVALID_GRAPH', path, `symbol '${symbol}' has an empty legal domain.`);
  return [first, last, multiple];
}

function proveDimensionWithinSymbol(
  inferred: ShapeDimensionSpec,
  assertedSymbol: string,
  environment: ShapeEnvironment,
  path: string,
): void {
  const target = environment.get(assertedSymbol);
  if (target === undefined) {
    fail('INVALID_GRAPH', path, `references undeclared symbol '${assertedSymbol}'.`);
  }
  const targetMultiple = target.multiple_of ?? 1;
  if (typeof inferred === 'number') {
    if (inferred < target.min || inferred > target.max) {
      fail(
        'BOUND_VIOLATION',
        path,
        `inferred constant ${inferred} is outside '${assertedSymbol}' bounds [${target.min}, ${target.max}].`,
      );
    }
    if (inferred % targetMultiple !== 0) {
      fail(
        'MULTIPLE_OF_VIOLATION',
        path,
        `inferred constant ${inferred} is not a multiple of ${targetMultiple}.`,
      );
    }
    return;
  }
  const [first, last, sourceMultiple] = legalInterval(environment, inferred, path);
  if (first < target.min || last > target.max) {
    fail(
      'BOUND_VIOLATION',
      path,
      `domain of '${inferred}' is not contained in '${assertedSymbol}' bounds [${target.min}, ${target.max}].`,
    );
  }
  if (first === last) {
    if (first % targetMultiple !== 0) {
      fail(
        'MULTIPLE_OF_VIOLATION',
        path,
        `the only legal value ${first} is not a multiple of ${targetMultiple}.`,
      );
    }
  } else if (sourceMultiple % targetMultiple !== 0) {
    fail(
      'MULTIPLE_OF_VIOLATION',
      path,
      `divisibility by ${sourceMultiple} does not prove divisibility by ${targetMultiple}.`,
    );
  }
}

function assertLogicalOutputShape(
  inferred: TensorShapeSpec,
  assertion: TensorShapeSpec,
  environment: ShapeEnvironment,
  relations: SymbolRelations,
  path: string,
): void {
  if (inferred.length !== assertion.length) {
    fail(
      'OUTPUT_SHAPE_MISMATCH',
      path,
      `inferred rank ${inferred.length}, but the graph asserts rank ${assertion.length}.`,
    );
  }
  for (let axis = 0; axis < assertion.length; axis++) {
    const actual = inferred[axis];
    const expected = assertion[axis];
    const axisPath = `${path}[${axis}]`;
    if (typeof expected === 'number') {
      if (!dimensionsProvablyEqual(actual, expected, environment)) {
        fail(
          'OUTPUT_SHAPE_MISMATCH',
          axisPath,
          `inferred '${String(actual)}', which is not provably constant ${expected}.`,
        );
      }
      continue;
    }
    const previous = relations.get(expected);
    if (previous !== undefined) {
      if (!dimensionsProvablyEqual(actual, previous, environment)) {
        fail(
          'SYMBOL_CONFLICT',
          axisPath,
          `inferred '${String(actual)}', but '${expected}' is already defined by '${String(previous)}'.`,
        );
      }
      continue;
    }
    proveDimensionWithinSymbol(actual, expected, environment, axisPath);
    relations.set(expected, actual);
  }
}

function registerAffineSymbolRelations(
  nodeIndex: number,
  candidates: readonly OperatorAffineDimensionRelation[],
  relations: ReadonlyMap<string, ShapeDimensionSpec>,
  affineRelations: AffineSymbolRelations,
): void {
  for (const candidate of candidates) {
    const path = `nodes[${nodeIndex}] affine relation '${candidate.target}'`;
    const sourceDefinition = relations.get(candidate.source);
    if (sourceDefinition !== candidate.source || affineRelations.has(candidate.source)) {
      fail(
        'SYMBOL_CONFLICT',
        path,
        `source symbol '${candidate.source}' must be one previously defined non-affine symbol.`,
      );
    }
    const previousAffine = affineRelations.get(candidate.target);
    if (previousAffine !== undefined) {
      if (previousAffine.source !== candidate.source ||
          previousAffine.offset !== candidate.offset) {
        fail(
          'SYMBOL_CONFLICT',
          path,
          `symbol '${candidate.target}' was already defined as ` +
          `${previousAffine.source}+${previousAffine.offset}, not ` +
          `${candidate.source}+${candidate.offset}.`,
        );
      }
      continue;
    }
    if (relations.has(candidate.target)) {
      fail(
        'SYMBOL_CONFLICT',
        path,
        `symbol '${candidate.target}' was already defined by a public input or non-affine output.`,
      );
    }
    affineRelations.set(candidate.target, Object.freeze({ ...candidate }));
  }
}

function assertLogicalSymbolsAvailable(
  descriptor: TensorDescriptor,
  relations: ReadonlyMap<string, ShapeDimensionSpec>,
  path: string,
): void {
  for (let axis = 0; axis < descriptor.shape.length; axis++) {
    const dimension = descriptor.shape[axis];
    if (typeof dimension === 'string' && !relations.has(dimension)) {
      fail(
        'UNBOUND_SYMBOL',
        `${path}.shape[${axis}]`,
        `uses symbol '${dimension}' before a public input or earlier output defines it.`,
      );
    }
  }
}

function logicalOperatorDescriptor(
  descriptor: TensorDescriptor | LogicalDomainTensorDescriptor,
  quantization: TensorQuantization | undefined,
): LogicalOperatorTensorDescriptor {
  return Object.freeze({
    shape: Object.freeze([...descriptor.shape]),
    dtype: descriptor.dtype,
    ...(quantization === undefined ? {} : { quantization }),
  });
}

function logicalDomainTensor(
  name: string,
  descriptor: LogicalOperatorTensorDescriptor,
): LogicalDomainTensorDescriptor {
  return Object.freeze({
    name,
    dtype: descriptor.dtype,
    shape: Object.freeze([...descriptor.shape]),
    ...(descriptor.quantization == null ? {} : { quantization: descriptor.quantization }),
  });
}

function provisionalLogicalShape(
  assertion: TensorShapeSpec,
  relations: ReadonlyMap<string, ShapeDimensionSpec>,
  inferredFrom: TensorShapeSpec | undefined,
  path: string,
): TensorShapeSpec {
  return Object.freeze(assertion.map((dimension, axis) => {
    if (typeof dimension === 'number') return dimension;
    const relation = relations.get(dimension);
    if (relation !== undefined) return relation;
    if (inferredFrom !== undefined && inferredFrom.length === assertion.length) {
      return inferredFrom[axis];
    }
    fail(
      'UNBOUND_SYMBOL',
      `${path}[${axis}]`,
      `cannot construct an operator assertion for undefined output symbol '${dimension}'.`,
    );
  }));
}

function logicalDeclaredOutputs(
  node: NodeDescriptor,
  nodeIndex: number,
  inputs: Readonly<Record<string, LogicalOperatorTensorDescriptor>>,
  relations: ReadonlyMap<string, ShapeDimensionSpec>,
  hydrated: HydratedTensorQuantization,
): Readonly<Record<string, LogicalOperatorTensorDescriptor>> | undefined {
  if (!LOGICAL_DECLARED_OUTPUT_OPERATORS.has(node.opType)) return undefined;
  const inputShape = inputs.input?.shape;
  return freezeRecord(sortedNames(node.outputs).map((port) => {
    const output = node.outputs[port];
    const shape = node.opType === 'Concat'
      ? Object.freeze([...output.shape])
      : provisionalLogicalShape(
        output.shape,
        relations,
        inputShape,
        `nodes[${nodeIndex}].outputs.${port}.shape`,
      );
    const quantization = hasOwn(hydrated, output.tensor)
      ? hydrated[output.tensor]
      : undefined;
    return [port, Object.freeze({
      shape,
      dtype: output.dtype,
      ...(quantization === undefined ? {} : { quantization }),
    })] as const;
  }));
}

function proveGraphShapeDomainInternal(
  graph: Graph,
  hydratedSource?: HydratedTensorQuantization,
): AcceptedGraphShapeDomainProof {
  assertGraph(graph);
  const hydrated = normalizeHydratedQuantization(graph, hydratedSource);
  const relations: SymbolRelations = new Map();
  const affineRelations: AffineSymbolRelations = new Map();
  const tensors = new Map<string, LogicalOperatorTensorDescriptor>();
  for (const name of sortedNames(graph.inputs)) {
    const input = graph.inputs[name];
    for (const dimension of input.shape) {
      if (typeof dimension === 'string') relations.set(dimension, dimension);
    }
    tensors.set(name, logicalOperatorDescriptor(
      input,
      hasOwn(hydrated, name) ? hydrated[name] : undefined,
    ));
  }
  for (const name of sortedNames(graph.weights)) {
    const weight = graph.weights[name];
    tensors.set(name, logicalOperatorDescriptor(
      weight,
      hasOwn(hydrated, name) ? hydrated[name] : undefined,
    ));
  }

  const nodeProofs: LogicalShapeDomainNodeProof[] = [];
  for (let nodeIndex = 0; nodeIndex < graph.nodes.length; nodeIndex++) {
    const node = graph.nodes[nodeIndex];
    let contract;
    try {
      contract = getOperatorShapeContract(node.opType);
    } catch (error) {
      if (error instanceof OperatorShapeContractError) {
        fail('UNKNOWN_OPERATOR', `nodes[${nodeIndex}].opType`, stripErrorPath(error), error);
      }
      throw error;
    }
    assertVariadicInputPorts(
      node.inputs,
      contract.ports.requiredInputs,
      contract.ports.optionalInputs,
      contract.ports.variadicInputs,
      `nodes[${nodeIndex}].inputs`,
    );
    assertExactPorts(
      node.outputs,
      contract.ports.outputs,
      `nodes[${nodeIndex}].outputs`,
      contract.ports.variadicOutputs,
    );
    const inputTensorEntries: Array<readonly [string, LogicalDomainTensorDescriptor]> = [];
    const operatorInputEntries: Array<readonly [string, LogicalOperatorTensorDescriptor]> = [];
    for (const port of sortedNames(node.inputs)) {
      const tensorName = node.inputs[port];
      const original = graph.tensors[tensorName];
      if (original === undefined) {
        fail('INVALID_GRAPH', `nodes[${nodeIndex}].inputs.${port}`, `references missing tensor '${tensorName}'.`);
      }
      assertLogicalSymbolsAvailable(original, relations, `nodes[${nodeIndex}].inputs.${port}`);
      const descriptor = tensors.get(tensorName);
      if (descriptor === undefined) {
        fail(
          'INVALID_GRAPH',
          `nodes[${nodeIndex}].inputs.${port}`,
          `tensor '${tensorName}' has no earlier logical descriptor.`,
        );
      }
      operatorInputEntries.push([port, descriptor]);
      inputTensorEntries.push([port, logicalDomainTensor(tensorName, descriptor)]);
    }
    const operatorInputs = freezeRecord(operatorInputEntries);
    const declaredOutputs = logicalDeclaredOutputs(
      node,
      nodeIndex,
      operatorInputs,
      relations,
      hydrated,
    );
    let proof;
    try {
      proof = proveOperatorShapeDomain(node.opType, {
        environment: graph.environment,
        inputs: operatorInputs,
        params: node.params,
        ...(declaredOutputs === undefined ? {} : { declaredOutputs }),
      });
    } catch (error) {
      if (error instanceof OperatorShapeContractError) {
        fail('UNKNOWN_OPERATOR', `nodes[${nodeIndex}].opType`, stripErrorPath(error), error);
      }
      throw error;
    }
    if (!proof.supported) {
      fail(
        'OPERATOR_DOMAIN_UNSUPPORTED',
        `nodes[${nodeIndex}]`,
        `${proof.code}: ${proof.reason}`,
      );
    }
    assertExactPorts(
      proof.outputs,
      contract.ports.outputs,
      `nodes[${nodeIndex}] proved outputs`,
      contract.ports.variadicOutputs,
    );
    assertExactPorts(
      proof.outputs,
      sortedNames(node.outputs),
      `nodes[${nodeIndex}] proved outputs`,
    );
    registerAffineSymbolRelations(
      nodeIndex,
      proof.affineRelations,
      relations,
      affineRelations,
    );
    const outputTensorEntries: Array<readonly [string, LogicalDomainTensorDescriptor]> = [];
    for (const port of sortedNames(node.outputs)) {
      const assertion = node.outputs[port];
      const output = proof.outputs[port];
      if (output === undefined) {
        fail('OUTPUT_PORT_MISMATCH', `nodes[${nodeIndex}].outputs.${port}`, 'was not proved.');
      }
      if (output.dtype !== assertion.dtype) {
        fail(
          'OUTPUT_DTYPE_MISMATCH',
          `nodes[${nodeIndex}].outputs.${port}.dtype`,
          `operator proves '${output.dtype}', but the graph asserts '${assertion.dtype}'.`,
        );
      }
      assertLogicalOutputShape(
        output.shape,
        assertion.shape,
        graph.environment,
        relations,
        `nodes[${nodeIndex}].outputs.${port}.shape`,
      );
      const quantization = validateOutputQuantization(
        assertion.tensor,
        output.quantization,
        hydrated,
        `nodes[${nodeIndex}].outputs.${port}.quantization`,
      );
      const canonicalOutput = Object.freeze({
        shape: Object.freeze([...output.shape]),
        dtype: output.dtype,
        ...(quantization === undefined ? {} : { quantization }),
      });
      tensors.set(assertion.tensor, canonicalOutput);
      outputTensorEntries.push([
        port,
        logicalDomainTensor(assertion.tensor, canonicalOutput),
      ]);
    }
    nodeProofs.push(Object.freeze({
      id: node.id,
      opType: node.opType,
      shapeFunctionId: contract.shapeFunctionId,
      inputs: freezeRecord(inputTensorEntries),
      outputs: freezeRecord(outputTensorEntries),
      facts: Object.freeze([...proof.facts]),
    }));
  }

  const outputs = Object.freeze(graph.outputs.map((name, index) => {
    const descriptor = tensors.get(name);
    if (descriptor === undefined) {
      fail('INVALID_GRAPH', `outputs[${index}]`, `references unresolved tensor '${name}'.`);
    }
    return logicalDomainTensor(name, descriptor);
  }));
  const relationEntries = [...relations.entries()].sort(([left], [right]) =>
    compareCanonicalNames(left, right));
  const affineRelationEntries = [...affineRelations.entries()]
    .sort(([left], [right]) => compareCanonicalNames(left, right))
    .map(([target, relation]) => [target, Object.freeze({
      source: relation.source,
      offset: relation.offset,
    })] as const);
  return Object.freeze({
    supported: true,
    graphFingerprint: graph.fingerprint,
    symbolRelations: freezeRecord(relationEntries),
    affineSymbolRelations: freezeRecord(affineRelationEntries),
    nodes: Object.freeze(nodeProofs),
    outputs,
  });
}

/**
 * Pure, conservative proof of canonical operator-shape legality over the whole
 * bounded logical domain. A registry gap or inconclusive operator proof rejects
 * the graph; warm/sample shapes are never treated as proof.
 */
export function proveGraphShapeDomain(
  graph: Graph,
  hydratedQuantization?: HydratedTensorQuantization,
): GraphShapeDomainProof {
  try {
    return proveGraphShapeDomainInternal(graph, hydratedQuantization);
  } catch (error) {
    const fingerprint = isRecord(graph) && typeof graph.fingerprint === 'string'
      ? graph.fingerprint
      : '';
    if (error instanceof ResolvedShapePlanError) {
      return Object.freeze({
        supported: false,
        graphFingerprint: fingerprint,
        code: error.code,
        path: error.path,
        reason: error.message,
      });
    }
    if (error instanceof OperatorShapeContractError) {
      return Object.freeze({
        supported: false,
        graphFingerprint: fingerprint,
        code: 'OPERATOR_DOMAIN_UNSUPPORTED',
        path: error.path,
        reason: `${error.code}: ${error.message}`,
      });
    }
    if (error instanceof ShapeContractError) {
      return Object.freeze({
        supported: false,
        graphFingerprint: fingerprint,
        code: error.code === 'ARITHMETIC_OVERFLOW' ? 'ARITHMETIC_OVERFLOW' : 'INVALID_GRAPH',
        path: error.path,
        reason: error.message,
      });
    }
    return Object.freeze({
      supported: false,
      graphFingerprint: fingerprint,
      code: 'INVALID_GRAPH',
      path: 'logical graph',
      reason: error instanceof Error ? error.message : String(error),
    });
  }
}
