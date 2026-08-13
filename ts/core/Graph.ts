import { runtimeDTypes, runtimeOperatorNames } from '../generated/volvoxaiEnums.js';
import type { RuntimeDType } from '../types.js';
import {
  PublicInputShapeContract,
  SHAPE_SYMBOL_PATTERN,
  ShapeContractError,
  ShapeEnvironment,
  checkedTensorByteLength,
  createTensorShapeSpec,
} from '../ops/shapeSystem.js';
import type {
  DimensionConstraintInput,
  ShapeDimensionSpec,
  TensorShapeSpec,
} from '../ops/shapeSystem.js';

export const VOLVOX_LOGICAL_GRAPH_FORMAT = 'volvox-graph/v1' as const;
export const VOLVOX_AFFINE_QUANTIZATION_FORMAT = 'volvox-affine-safetensors/v1' as const;

export type GraphErrorDiagnostic = 'INVALID_GRAPH' | 'REEXPORT_REQUIRED';

/** Stable model-load failure raised before tensors or backends are created. */
export class GraphError extends Error {
  readonly code = 'INVALID_GRAPH' as const;
  readonly diagnostic: GraphErrorDiagnostic;
  readonly path: string;

  constructor(
    path: string,
    message: string,
    diagnostic: GraphErrorDiagnostic = 'INVALID_GRAPH',
    cause?: unknown,
  ) {
    super(`[Graph] ${path}: ${message}`, cause === undefined ? undefined : { cause });
    this.name = 'GraphError';
    this.diagnostic = diagnostic;
    this.path = path;
  }
}

export interface DimensionDescriptor {
  readonly name: string;
  readonly min: number;
  readonly max: number;
  /** Canonical descriptors always materialize the schema default of one. */
  readonly multiple_of: number;
}

/**
 * Declares that axis 0 of a fixed weight is a runtime-selected slot axis: a
 * bank of LoRA families or MoE experts. The slot count is governed by a bounded
 * dimension, so slots may be added up to `max` without changing graph topology.
 */
export interface WeightBankSpec {
  readonly name: string;
  readonly dimension: string;
  readonly min: number;
  readonly max: number;
}

export interface PerTensorQuantizationReference {
  readonly scheme: 'per_tensor';
  readonly scale_tensor: string;
  readonly zero_point_tensor: string;
}

export interface PerAxisQuantizationReference {
  readonly scheme: 'per_axis';
  /** Always normalized into [0, target rank). */
  readonly axis: number;
  readonly scale_tensor: string;
  readonly zero_point_tensor: string;
}

export type AffineQuantizationReference =
  | PerTensorQuantizationReference
  | PerAxisQuantizationReference;

export interface AffineQuantizationTable {
  readonly format: typeof VOLVOX_AFFINE_QUANTIZATION_FORMAT;
  readonly tensors: Readonly<Record<string, AffineQuantizationReference>>;
}

interface TensorDescriptorBase {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: TensorShapeSpec;
  /** Present only when the central affine table targets this tensor. */
  readonly quantization?: AffineQuantizationReference;
}

export interface InputDescriptor extends TensorDescriptorBase {
  readonly kind: 'input';
}

export interface WeightDescriptor extends TensorDescriptorBase {
  readonly kind: 'weight';
  /** Weight shapes are fixed, so this refinement contains numbers only. */
  readonly shape: readonly number[];
  /** Non-null when axis 0 indexes runtime-selected slots. */
  readonly bank: WeightBankSpec | null;
}

export interface ValueDescriptor extends TensorDescriptorBase {
  readonly kind: 'value';
  readonly producerNodeId: string;
  readonly producerPort: string;
}

export type TensorDescriptor =
  | InputDescriptor
  | WeightDescriptor
  | ValueDescriptor;

export interface WeightDescriptorInput {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
}

export interface NodeOutputDescriptor {
  readonly tensor: string;
  readonly dtype: RuntimeDType;
  /** Assertion retained for canonical operator inference in DS2. */
  readonly shape: TensorShapeSpec;
  readonly quantization?: AffineQuantizationReference;
}

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | readonly JsonValue[]
  | Readonly<{ [name: string]: JsonValue }>;

export interface NodeDescriptor {
  readonly id: string;
  readonly opType: string;
  readonly inputs: Readonly<Record<string, string>>;
  readonly outputs: Readonly<Record<string, NodeOutputDescriptor>>;
  readonly params: Readonly<Record<string, JsonValue>>;
}

/** Immutable logical topology. It contains no Tensor allocation or backend state. */
export interface Graph {
  readonly format: typeof VOLVOX_LOGICAL_GRAPH_FORMAT;
  readonly environment: ShapeEnvironment;
  readonly dimensions: Readonly<Record<string, DimensionDescriptor>>;
  readonly inputs: Readonly<Record<string, InputDescriptor>>;
  readonly weights: Readonly<Record<string, WeightDescriptor>>;
  readonly banks: Readonly<Record<string, WeightBankSpec>>;
  readonly tensors: Readonly<Record<string, TensorDescriptor>>;
  readonly nodes: readonly NodeDescriptor[];
  readonly outputs: readonly string[];
  readonly quantization: AffineQuantizationTable | null;
  /** Collision-free canonical serialization suitable as a definition fingerprint. */
  readonly fingerprint: string;
}

type UnknownRecord = Record<string, unknown>;

const RUNTIME_DTYPES = new Set<unknown>(runtimeDTypes);
const RUNTIME_OPERATORS = new Set<unknown>(runtimeOperatorNames);
const UTF8_ENCODER = new TextEncoder();
const RETIRED_INLINE_AFFINE_FIELDS = new Set([
  'quantization',
  'zero_point',
  'input_scale', 'input_zero_point',
  'output_scale', 'output_zero_point',
  'weight_scale', 'weight_zero_point',
  'scales', 'zero_points',
  'scale_tensor', 'zero_point_tensor',
]);

function fail(
  path: string,
  message: string,
  diagnostic: GraphErrorDiagnostic = 'INVALID_GRAPH',
): never {
  throw new GraphError(path, message, diagnostic);
}

function isWellFormedUnicode(value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    const codeUnit = value.charCodeAt(index);
    if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
      if (index + 1 >= value.length) return false;
      const trailing = value.charCodeAt(index + 1);
      if (trailing < 0xdc00 || trailing > 0xdfff) return false;
      index++;
    } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
      return false;
    }
  }
  return true;
}

function assertWellFormedUnicode(value: string, path: string): void {
  if (!isWellFormedUnicode(value)) {
    fail(path, 'must not contain an unpaired UTF-16 surrogate.');
  }
}

function hasOwn(value: object, key: PropertyKey): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function compareUnsignedBytes(left: Uint8Array, right: Uint8Array): number {
  const shared = Math.min(left.length, right.length);
  for (let index = 0; index < shared; index++) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return left.length - right.length;
}

/** Locale-independent ordering required by the bounded-shape v1 contract. */
function compareCanonicalNames(left: string, right: string): number {
  const compared = compareUnsignedBytes(UTF8_ENCODER.encode(left), UTF8_ENCODER.encode(right));
  if (compared !== 0) return compared;
  // TextEncoder replaces isolated UTF-16 surrogates. Preserve determinism even
  // for such programmatically supplied strings, while ordinary JSON names take
  // the normative UTF-8 branch above.
  return left < right ? -1 : left > right ? 1 : 0;
}

function sortedNames(value: object): string[] {
  const names = Object.getOwnPropertyNames(value);
  for (const name of names) assertWellFormedUnicode(name, 'object member name');
  return names.sort(compareCanonicalNames);
}

function assertPlainRecord(value: unknown, path: string): UnknownRecord {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    fail(path, 'must be a decoded JSON object.');
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    fail(path, 'must be a plain decoded JSON object.');
  }
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail(path, 'must not contain symbol-keyed fields.');
  }
  for (const field of Object.getOwnPropertyNames(value)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, field)!;
    if (!hasOwn(descriptor, 'value') || descriptor.enumerable !== true) {
      fail(`${path}.${field}`, 'must be an enumerable decoded JSON data field.');
    }
  }
  return value as UnknownRecord;
}

function assertDenseArray(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) fail(path, 'must be a decoded JSON array.');
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail(path, 'must not contain symbol-keyed fields.');
  }
  for (let index = 0; index < value.length; index++) {
    if (!hasOwn(value, index)) fail(`${path}[${index}]`, 'must not be an array hole.');
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index))!;
    if (!hasOwn(descriptor, 'value') || descriptor.enumerable !== true) {
      fail(`${path}[${index}]`, 'must be an enumerable decoded JSON data item.');
    }
  }
  for (const field of Object.getOwnPropertyNames(value)) {
    if (field === 'length') continue;
    const index = Number(field);
    if (!Number.isSafeInteger(index) || index < 0 || index >= value.length || String(index) !== field) {
      fail(`${path}.${field}`, 'is not a decoded JSON array index.');
    }
  }
  return value;
}

function assertFields(
  value: UnknownRecord,
  path: string,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const allowed = new Set([...required, ...optional]);
  for (const field of sortedNames(value)) {
    if (!allowed.has(field)) fail(path, `has unsupported field '${field}'.`);
  }
  for (const field of required) {
    if (!hasOwn(value, field)) fail(path, `requires field '${field}'.`);
  }
}

function assertNonEmptyString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    fail(path, 'must be a non-empty string.');
  }
  assertWellFormedUnicode(value, path);
  return value;
}

function assertRuntimeDType(value: unknown, path: string): RuntimeDType {
  if (typeof value !== 'string' || !RUNTIME_DTYPES.has(value)) {
    fail(path, `must be one of ${runtimeDTypes.map((dtype) => `'${dtype}'`).join(', ')}.`);
  }
  return value as RuntimeDType;
}

function freezeRecord<T>(entries: readonly (readonly [string, T])[]): Readonly<Record<string, T>> {
  return Object.freeze(Object.fromEntries(entries)) as Readonly<Record<string, T>>;
}

function parseShapeSpec(
  value: unknown,
  environment: ShapeEnvironment,
  path: string,
): TensorShapeSpec {
  const shape = assertDenseArray(value, path);
  return createTensorShapeSpec(shape as readonly ShapeDimensionSpec[], environment, path);
}

function parseDimensions(
  raw: unknown,
): {
  readonly environment: ShapeEnvironment;
  readonly dimensions: Readonly<Record<string, DimensionDescriptor>>;
} {
  const dimensions = assertPlainRecord(raw, 'dimensions');
  const constraints: DimensionConstraintInput[] = [];
  for (const name of sortedNames(dimensions)) {
    if (!SHAPE_SYMBOL_PATTERN.test(name) || name.length > 64) {
      fail(
        `dimensions.${JSON.stringify(name)}`,
        'symbol names must match /^[A-Za-z][A-Za-z0-9_]{0,63}$/.',
      );
    }
    const descriptor = assertPlainRecord(dimensions[name], `dimensions.${name}`);
    assertFields(descriptor, `dimensions.${name}`, ['min', 'max'], ['multiple_of']);
    constraints.push({
      name,
      min: descriptor.min as number,
      max: descriptor.max as number,
      ...(hasOwn(descriptor, 'multiple_of')
        ? { multiple_of: descriptor.multiple_of as number }
        : {}),
    });
  }

  const environment = new ShapeEnvironment(constraints);
  const entries = environment.dimensions.map((constraint) => {
    const descriptor: DimensionDescriptor = Object.freeze({
      name: constraint.name,
      min: constraint.min,
      max: constraint.max,
      multiple_of: constraint.multiple_of ?? 1,
    });
    return [constraint.name, descriptor] as const;
  });
  return Object.freeze({ environment, dimensions: freezeRecord(entries) });
}

function parseInputs(
  raw: unknown,
  environment: ShapeEnvironment,
): Readonly<Record<string, InputDescriptor>> {
  const inputs = assertPlainRecord(raw, 'inputs');
  const candidates: Array<{ name: string; dtype: RuntimeDType; shape: TensorShapeSpec }> = [];
  for (const name of sortedNames(inputs)) {
    assertNonEmptyString(name, `inputs.${JSON.stringify(name)} name`);
    const descriptor = assertPlainRecord(inputs[name], `inputs.${name}`);
    if (hasOwn(descriptor, 'quantization')) {
      fail(
        `inputs.${name}.quantization`,
        'inline quantization is forbidden; use the central quantization table.',
      );
    }
    assertFields(descriptor, `inputs.${name}`, ['dtype', 'shape']);
    candidates.push({
      name,
      dtype: assertRuntimeDType(descriptor.dtype, `inputs.${name}.dtype`),
      shape: parseShapeSpec(descriptor.shape, environment, `inputs.${name}.shape`),
    });
  }

  // The public contract is the authoritative model-independent input
  // normalizer. Constructing it also guarantees a canonical name order.
  const contract = new PublicInputShapeContract(environment, candidates);
  return freezeRecord(contract.inputs.map((input) => [input.name, Object.freeze({
    kind: 'input' as const,
    name: input.name,
    dtype: input.dtype,
    shape: input.shape,
  })] as const));
}

function parseWeights(
  raw: readonly WeightDescriptorInput[],
  environment: ShapeEnvironment,
): Readonly<Record<string, WeightDescriptor>> {
  const weights = assertDenseArray(raw, 'weights');
  const entries: Array<readonly [string, WeightDescriptor]> = [];
  const seen = new Set<string>();
  for (let index = 0; index < weights.length; index++) {
    const path = `weights[${index}]`;
    const source = assertPlainRecord(weights[index], path);
    if (hasOwn(source, 'quantization')) {
      fail(
        `${path}.quantization`,
        'inline quantization is forbidden; use the central quantization table.',
      );
    }
    assertFields(source, path, ['name', 'dtype', 'shape']);
    const name = assertNonEmptyString(source.name, `${path}.name`);
    if (seen.has(name)) fail(`${path}.name`, `duplicates tensor '${name}'.`);
    seen.add(name);
    const dtype = assertRuntimeDType(source.dtype, `${path}.dtype`);
    const parsedShape = parseShapeSpec(source.shape, environment, `${path}.shape`);
    if (parsedShape.some((dimension) => typeof dimension !== 'number')) {
      fail(`${path}.shape`, 'weight shapes must contain constant dimensions only.');
    }
    const shape = parsedShape as readonly number[];
    // Fixed weights are validated at load rather than deferred to a binding.
    checkedTensorByteLength(shape, dtype, `weight '${name}'`);
    entries.push([name, Object.freeze({ kind: 'weight', name, dtype, shape, bank: null })]);
  }
  entries.sort(([left], [right]) => compareCanonicalNames(left, right));
  return freezeRecord(entries);
}

/**
 * Parse the optional `banks` table: weight name -> bounded dimension governing
 * its slot axis. Slot residency is a context concern; this only records that a
 * weight is sliceable along axis 0 and how far it may grow.
 */
function parseBanks(
  raw: unknown,
  dimensions: Readonly<Record<string, DimensionDescriptor>>,
  weights: Readonly<Record<string, WeightDescriptor>>,
): Readonly<Record<string, WeightBankSpec>> {
  if (raw === undefined) return freezeRecord([]);
  const banks = assertPlainRecord(raw, 'banks');
  const entries: Array<readonly [string, WeightBankSpec]> = [];
  for (const name of sortedNames(banks)) {
    const path = `banks.${name}`;
    const dimension = assertNonEmptyString(banks[name], path);
    const constraint = hasOwn(dimensions, dimension) ? dimensions[dimension] : undefined;
    if (constraint === undefined) {
      fail(path, `references undeclared dimension '${dimension}'.`);
    }
    const weight = hasOwn(weights, name) ? weights[name] : undefined;
    if (weight === undefined) {
      fail(path, `names tensor '${name}', which is not a supplied fixed weight.`);
    }
    if (weight.shape.length < 2) {
      fail(path, 'a bank weight needs a slot axis and at least one payload axis.');
    }
    const slots = weight.shape[0];
    if (slots < constraint.min || slots > constraint.max) {
      fail(
        path,
        `supplies ${slots} slots, outside the '${dimension}' bound ` +
        `[${constraint.min}, ${constraint.max}].`,
      );
    }
    if (constraint.multiple_of !== 1 && slots % constraint.multiple_of !== 0) {
      fail(path, `supplies ${slots} slots, which is not a multiple of ${constraint.multiple_of}.`);
    }
    entries.push([name, Object.freeze({
      name, dimension, min: constraint.min, max: constraint.max,
    })]);
  }
  return freezeRecord(entries);
}

function cloneLogicalJson(
  value: unknown,
  path: string,
  seen: WeakSet<object>,
): JsonValue {
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    assertWellFormedUnicode(value, path);
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail(path, 'must be a finite decoded JSON number.');
    return value;
  }
  if (typeof value !== 'object') fail(path, 'must contain decoded JSON values only.');
  if (seen.has(value)) fail(path, 'must not contain cycles or shared object aliases.');
  seen.add(value);

  if (Array.isArray(value)) {
    const source = assertDenseArray(value, path);
    return Object.freeze(source.map((entry, index) =>
      cloneLogicalJson(entry, `${path}[${index}]`, seen)));
  }

  const source = assertPlainRecord(value, path);
  const entries = sortedNames(source).map((name) => {
    if (RETIRED_INLINE_AFFINE_FIELDS.has(name)) {
      fail(
        `${path}.${name}`,
        'is retired inline affine metadata; use the central quantization table.',
      );
    }
    return [name, cloneLogicalJson(source[name], `${path}.${name}`, seen)] as const;
  });
  return freezeRecord(entries);
}

function parseParams(raw: unknown, path: string): Readonly<Record<string, JsonValue>> {
  const source = assertPlainRecord(raw, path);
  return cloneLogicalJson(source, path, new WeakSet()) as Readonly<Record<string, JsonValue>>;
}

function parseNodeInputs(
  raw: unknown,
  path: string,
  availableTensors: ReadonlySet<string>,
): Readonly<Record<string, string>> {
  const inputs = assertPlainRecord(raw, path);
  const entries: Array<readonly [string, string]> = [];
  for (const port of sortedNames(inputs)) {
    assertNonEmptyString(port, `${path} port`);
    const tensor = assertNonEmptyString(inputs[port], `${path}.${port}`);
    if (!availableTensors.has(tensor)) {
      fail(`${path}.${port}`, `references unresolved tensor '${tensor}'; inputs must be topological.`);
    }
    entries.push([port, tensor]);
  }
  return freezeRecord(entries);
}

function parseNodeOutputs(
  raw: unknown,
  path: string,
  environment: ShapeEnvironment,
  occupiedTensors: ReadonlySet<string>,
): Readonly<Record<string, NodeOutputDescriptor>> {
  const outputs = assertPlainRecord(raw, path);
  const ports = sortedNames(outputs);
  if (ports.length === 0) fail(path, 'must declare at least one output port.');
  const entries: Array<readonly [string, NodeOutputDescriptor]> = [];
  const localTensors = new Set<string>();
  for (const port of ports) {
    assertNonEmptyString(port, `${path} port`);
    const descriptorPath = `${path}.${port}`;
    const descriptor = assertPlainRecord(outputs[port], descriptorPath);
    if (hasOwn(descriptor, 'quantization')) {
      fail(
        `${descriptorPath}.quantization`,
        'inline quantization is forbidden; use the central quantization table.',
      );
    }
    assertFields(descriptor, descriptorPath, ['tensor', 'dtype', 'shape']);
    const tensor = assertNonEmptyString(descriptor.tensor, `${descriptorPath}.tensor`);
    if (occupiedTensors.has(tensor) || localTensors.has(tensor)) {
      fail(`${descriptorPath}.tensor`, `duplicates tensor '${tensor}'.`);
    }
    localTensors.add(tensor);
    entries.push([port, Object.freeze({
      tensor,
      dtype: assertRuntimeDType(descriptor.dtype, `${descriptorPath}.dtype`),
      shape: parseShapeSpec(descriptor.shape, environment, `${descriptorPath}.shape`),
    })]);
  }
  return freezeRecord(entries);
}

interface ParsedQuantization {
  readonly table: AffineQuantizationTable;
  readonly parameterNames: ReadonlySet<string>;
}

function hasExactShape(shape: readonly ShapeDimensionSpec[], expected: number): boolean {
  return shape.length === 1 && shape[0] === expected;
}

function parseAffineQuantization(
  raw: unknown,
  tensors: Readonly<Record<string, TensorDescriptor>>,
  weights: Readonly<Record<string, WeightDescriptor>>,
): ParsedQuantization {
  const root = assertPlainRecord(raw, 'quantization');
  assertFields(root, 'quantization', ['format', 'tensors']);
  if (root.format !== VOLVOX_AFFINE_QUANTIZATION_FORMAT) {
    fail(
      'quantization.format',
      `must be exactly '${VOLVOX_AFFINE_QUANTIZATION_FORMAT}'.`,
    );
  }
  const rawTensors = assertPlainRecord(root.tensors, 'quantization.tensors');
  const targetNames = sortedNames(rawTensors);
  if (targetNames.length === 0) {
    fail('quantization.tensors', 'must be a non-empty object.');
  }

  const entries: Array<readonly [string, AffineQuantizationReference]> = [];
  const parameterNames = new Set<string>();
  for (const targetName of targetNames) {
    assertNonEmptyString(targetName, 'quantization target name');
    if (!hasOwn(tensors, targetName)) {
      fail(`quantization.tensors.${targetName}`, `targets unknown tensor '${targetName}'.`);
    }
    const target = tensors[targetName];
    if (target.dtype !== 'int8' && target.dtype !== 'uint8') {
      fail(
        `quantization.tensors.${targetName}`,
        `target '${targetName}' must have int8 or uint8 storage.`,
      );
    }

    const descriptorPath = `quantization.tensors.${targetName}`;
    const source = assertPlainRecord(rawTensors[targetName], descriptorPath);
    const perAxis = source.scheme === 'per_axis';
    if (source.scheme !== 'per_tensor' && !perAxis) {
      fail(`${descriptorPath}.scheme`, `must be 'per_tensor' or 'per_axis'.`);
    }
    assertFields(
      source,
      descriptorPath,
      perAxis
        ? ['scheme', 'axis', 'scale_tensor', 'zero_point_tensor']
        : ['scheme', 'scale_tensor', 'zero_point_tensor'],
    );
    const scaleName = assertNonEmptyString(source.scale_tensor, `${descriptorPath}.scale_tensor`);
    const zeroName = assertNonEmptyString(
      source.zero_point_tensor,
      `${descriptorPath}.zero_point_tensor`,
    );
    if (scaleName === zeroName) {
      fail(descriptorPath, 'requires distinct scale and zero-point parameter tensors.');
    }
    const scale = hasOwn(weights, scaleName) ? weights[scaleName] : undefined;
    const zeroPoint = hasOwn(weights, zeroName) ? weights[zeroName] : undefined;
    if (!scale || !zeroPoint) {
      fail(descriptorPath, 'scale and zero-point parameters must exist in supplied fixed weights.');
    }
    if (scale.dtype !== 'float32') {
      fail(
        `${descriptorPath}.scale_tensor`,
        `parameter '${scaleName}' must have float32 storage.`,
      );
    }
    if (zeroPoint.dtype !== target.dtype) {
      fail(
        `${descriptorPath}.zero_point_tensor`,
        `parameter '${zeroName}' dtype must match target dtype '${target.dtype}'.`,
      );
    }

    let reference: AffineQuantizationReference;
    if (perAxis) {
      if (!Number.isSafeInteger(source.axis) || target.shape.length === 0) {
        fail(`${descriptorPath}.axis`, 'must be an integer axis for a non-scalar target.');
      }
      const rawAxis = source.axis as number;
      if (rawAxis < -target.shape.length || rawAxis >= target.shape.length) {
        fail(`${descriptorPath}.axis`, `is outside target rank ${target.shape.length}.`);
      }
      const axis = rawAxis < 0 ? rawAxis + target.shape.length : rawAxis;
      const extent = target.shape[axis];
      if (typeof extent !== 'number') {
        fail(
          `${descriptorPath}.axis`,
          `target axis ${axis} is symbolic; per-axis quantization requires a constant extent.`,
        );
      }
      if (!hasExactShape(scale.shape, extent) || !hasExactShape(zeroPoint.shape, extent)) {
        fail(descriptorPath, `parameter tensors must both have shape [${extent}].`);
      }
      reference = Object.freeze({
        scheme: 'per_axis',
        axis,
        scale_tensor: scaleName,
        zero_point_tensor: zeroName,
      });
    } else {
      if (!hasExactShape(scale.shape, 1) || !hasExactShape(zeroPoint.shape, 1)) {
        fail(descriptorPath, 'per-tensor parameter tensors must both have scalar shape [1].');
      }
      reference = Object.freeze({
        scheme: 'per_tensor',
        scale_tensor: scaleName,
        zero_point_tensor: zeroName,
      });
    }
    entries.push([targetName, reference]);
    parameterNames.add(scaleName);
    parameterNames.add(zeroName);
  }

  const references = freezeRecord(entries);
  for (const parameterName of parameterNames) {
    if (hasOwn(references, parameterName)) {
      fail(
        `quantization.tensors.${parameterName}`,
        `parameter tensor '${parameterName}' cannot itself be a quantization target.`,
      );
    }
  }
  return Object.freeze({
    table: Object.freeze({
      format: VOLVOX_AFFINE_QUANTIZATION_FORMAT,
      tensors: references,
    }),
    parameterNames,
  });
}

function attachTensorQuantization<TDescriptor extends TensorDescriptor>(
  descriptors: Readonly<Record<string, TDescriptor>>,
  references: Readonly<Record<string, AffineQuantizationReference>>,
): Readonly<Record<string, TDescriptor>> {
  return freezeRecord(Object.entries(descriptors).map(([name, descriptor]) => {
    const reference = hasOwn(references, name) ? references[name] : undefined;
    return [name, reference === undefined
      ? descriptor
      : Object.freeze({ ...descriptor, quantization: reference }) as unknown as TDescriptor] as const;
  }));
}

function attachNodeOutputQuantization(
  nodes: readonly NodeDescriptor[],
  references: Readonly<Record<string, AffineQuantizationReference>>,
): readonly NodeDescriptor[] {
  return Object.freeze(nodes.map((node) => {
    let changed = false;
    const outputs = freezeRecord(Object.entries(node.outputs).map(([port, descriptor]) => {
      const reference = hasOwn(references, descriptor.tensor)
        ? references[descriptor.tensor]
        : undefined;
      if (reference === undefined) return [port, descriptor] as const;
      changed = true;
      return [port, Object.freeze({ ...descriptor, quantization: reference })] as const;
    }));
    return changed ? Object.freeze({ ...node, outputs }) : node;
  }));
}

function validateQuantizationOperatorReferences(
  nodes: readonly NodeDescriptor[],
  references: Readonly<Record<string, AffineQuantizationReference>>,
): void {
  for (let index = 0; index < nodes.length; index++) {
    const node = nodes[index];
    if (node.opType === 'QuantizeLinear') {
      for (const descriptor of Object.values(node.outputs)) {
        const reference = hasOwn(references, descriptor.tensor)
          ? references[descriptor.tensor]
          : undefined;
        if (!reference || node.inputs.scale !== reference.scale_tensor ||
            node.inputs.zero_point !== reference.zero_point_tensor) {
          fail(
            `nodes[${index}]`,
            `QuantizeLinear parameter inputs must match output '${descriptor.tensor}' quantization references.`,
          );
        }
      }
    } else if (node.opType === 'DequantizeLinear') {
      const inputName = node.inputs.input;
      const reference = inputName !== undefined && hasOwn(references, inputName)
        ? references[inputName]
        : undefined;
      if (!reference || node.inputs.scale !== reference.scale_tensor ||
          node.inputs.zero_point !== reference.zero_point_tensor) {
        fail(
          `nodes[${index}]`,
          `DequantizeLinear parameter inputs must match input '${String(inputName)}' quantization references.`,
        );
      }
    }
  }
}

function canonicalJson(value: JsonValue): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const record = value as Readonly<Record<string, JsonValue>>;
  return `{${Object.keys(record).sort(compareCanonicalNames).map((name) =>
    `${JSON.stringify(name)}:${canonicalJson(record[name])}`).join(',')}}`;
}

function graphFingerprint(graph: Omit<Graph, 'fingerprint'>): string {
  const dimensions = Object.fromEntries(Object.entries(graph.dimensions).map(([name, descriptor]) => [
    name,
    { min: descriptor.min, max: descriptor.max, multiple_of: descriptor.multiple_of },
  ]));
  const inputs = Object.fromEntries(Object.entries(graph.inputs).map(([name, descriptor]) => [
    name,
    { dtype: descriptor.dtype, shape: descriptor.shape },
  ]));
  const weights = Object.fromEntries(Object.entries(graph.weights).map(([name, descriptor]) => [
    name,
    {
      dtype: descriptor.dtype,
      // A bank's slot extent is the bounded dimension, not the currently
      // supplied slot count, so filling a slot keeps the definition identity.
      shape: descriptor.bank === null
        ? descriptor.shape
        : [descriptor.bank.dimension, ...descriptor.shape.slice(1)],
    },
  ]));
  const banks = Object.fromEntries(Object.entries(graph.banks).map(([name, spec]) => [
    name,
    { dimension: spec.dimension, min: spec.min, max: spec.max },
  ]));
  const nodes = graph.nodes.map((node) => ({
    id: node.id,
    opType: node.opType,
    inputs: node.inputs,
    outputs: node.outputs,
    params: node.params,
  }));
  const value = {
    format: graph.format,
    banks,
    dimensions,
    inputs,
    weights,
    nodes,
    outputs: graph.outputs,
    quantization: graph.quantization,
  } as unknown as JsonValue;
  return `volvox-logical-graph/v1|${canonicalJson(value)}`;
}

/**
 * Parse the dynamic-first graph schema into a backend-independent immutable
 * model. Supplied weight metadata participates in topology and identity, but
 * no weight or activation storage is allocated here.
 */
export function parseGraphDocument(
  raw: unknown,
  weightDescriptors: readonly WeightDescriptorInput[] = [],
): Graph {
  try {
    const document = assertPlainRecord(raw, 'graph');
    if (document.format !== VOLVOX_LOGICAL_GRAPH_FORMAT) {
      fail('format', `must be exactly '${VOLVOX_LOGICAL_GRAPH_FORMAT}'.`);
    }
    for (const field of ['weights_quantization', 'weights_quantization_storage']) {
      if (hasOwn(document, field)) {
        fail(
          `graph.${field}`,
          'legacy inline quantization is forbidden; use the central quantization table.',
          'REEXPORT_REQUIRED',
        );
      }
    }
    assertFields(
      document,
      'graph',
      ['format', 'dimensions', 'inputs', 'nodes', 'outputs'],
      ['banks', 'quantization'],
    );

    const { environment, dimensions } = parseDimensions(document.dimensions);
    const baseInputs = parseInputs(document.inputs, environment);
    const parsedWeights = parseWeights(weightDescriptors, environment);
    const banks = parseBanks(document.banks, dimensions, parsedWeights);
    const baseWeights = freezeRecord(Object.entries(parsedWeights).map(
      ([name, descriptor]) => [
        name,
        hasOwn(banks, name)
          ? Object.freeze({ ...descriptor, bank: banks[name] })
          : descriptor,
      ] as const,
    ));

    const tensorEntries: Array<readonly [string, TensorDescriptor]> = [];
    const occupiedTensors = new Set<string>();
    for (const descriptor of Object.values(baseInputs)) {
      occupiedTensors.add(descriptor.name);
      tensorEntries.push([descriptor.name, descriptor]);
    }
    for (const descriptor of Object.values(baseWeights)) {
      if (occupiedTensors.has(descriptor.name)) {
        fail('weights', `tensor '${descriptor.name}' duplicates a public input.`);
      }
      occupiedTensors.add(descriptor.name);
      tensorEntries.push([descriptor.name, descriptor]);
    }

    const rawNodes = assertDenseArray(document.nodes, 'nodes');
    const nodeIds = new Set<string>();
    const nodes: NodeDescriptor[] = [];
    for (let index = 0; index < rawNodes.length; index++) {
      const path = `nodes[${index}]`;
      const source = assertPlainRecord(rawNodes[index], path);
      if (hasOwn(source, 'outputs_shape') || hasOwn(source, 'outputs_dtype')) {
        fail(
          path,
          "uses legacy 'outputs_shape'/'outputs_dtype' fields; re-export the model with unified output descriptors.",
          'REEXPORT_REQUIRED',
        );
      }
      if (hasOwn(source, 'outputs_quantization')) {
        fail(
          `${path}.outputs_quantization`,
          'inline output quantization is forbidden; use the central quantization table.',
          'REEXPORT_REQUIRED',
        );
      }
      assertFields(source, path, ['id', 'opType', 'inputs', 'outputs', 'params']);
      const id = assertNonEmptyString(source.id, `${path}.id`);
      if (nodeIds.has(id)) fail(`${path}.id`, `duplicates node id '${id}'.`);
      nodeIds.add(id);
      const opType = assertNonEmptyString(source.opType, `${path}.opType`);
      if (!RUNTIME_OPERATORS.has(opType)) {
        fail(`${path}.opType`, `uses unsupported runtime operator '${opType}'.`);
      }

      // Inputs resolve before this node publishes any outputs, preventing
      // self-reference and forward-reference ambiguities.
      const nodeInputs = parseNodeInputs(source.inputs, `${path}.inputs`, occupiedTensors);
      const nodeOutputs = parseNodeOutputs(
        source.outputs,
        `${path}.outputs`,
        environment,
        occupiedTensors,
      );
      const params = parseParams(source.params, `${path}.params`);
      const node = Object.freeze({ id, opType, inputs: nodeInputs, outputs: nodeOutputs, params });
      nodes.push(node);

      for (const [port, descriptor] of Object.entries(nodeOutputs)) {
        occupiedTensors.add(descriptor.tensor);
        tensorEntries.push([descriptor.tensor, Object.freeze({
          kind: 'value',
          name: descriptor.tensor,
          dtype: descriptor.dtype,
          shape: descriptor.shape,
          producerNodeId: id,
          producerPort: port,
        })]);
      }
    }

    const rawOutputs = assertDenseArray(document.outputs, 'outputs');
    if (rawOutputs.length === 0) fail('outputs', 'must be a non-empty array.');
    const outputNames: string[] = [];
    const seenOutputs = new Set<string>();
    for (let index = 0; index < rawOutputs.length; index++) {
      const name = assertNonEmptyString(rawOutputs[index], `outputs[${index}]`);
      if (seenOutputs.has(name)) fail(`outputs[${index}]`, `duplicates graph output '${name}'.`);
      if (!occupiedTensors.has(name)) fail(`outputs[${index}]`, `references unresolved tensor '${name}'.`);
      seenOutputs.add(name);
      outputNames.push(name);
    }

    tensorEntries.sort(([left], [right]) => compareCanonicalNames(left, right));
    const baseTensors = freezeRecord(tensorEntries);
    const parsedQuantization = hasOwn(document, 'quantization')
      ? parseAffineQuantization(document.quantization, baseTensors, baseWeights)
      : null;
    const quantization = parsedQuantization?.table ?? null;
    const references = quantization?.tensors ?? freezeRecord<AffineQuantizationReference>([]);
    if (parsedQuantization) {
      validateQuantizationOperatorReferences(nodes, references);
      for (let index = 0; index < outputNames.length; index++) {
        if (parsedQuantization.parameterNames.has(outputNames[index])) {
          fail(
            `outputs[${index}]`,
            `quantization parameter tensor '${outputNames[index]}' cannot be a public graph output.`,
          );
        }
      }
    }

    const tensors = attachTensorQuantization(baseTensors, references);
    const inputs = freezeRecord(Object.keys(baseInputs).map((name) =>
      [name, tensors[name] as InputDescriptor] as const));
    const weights = freezeRecord(Object.keys(baseWeights).map((name) =>
      [name, tensors[name] as WeightDescriptor] as const));
    const logicalNodes = attachNodeOutputQuantization(nodes, references);
    const graphWithoutFingerprint: Omit<Graph, 'fingerprint'> = Object.freeze({
      format: VOLVOX_LOGICAL_GRAPH_FORMAT,
      environment,
      dimensions,
      inputs,
      weights,
      banks,
      tensors,
      nodes: logicalNodes,
      outputs: Object.freeze(outputNames),
      quantization,
    });
    return Object.freeze({
      ...graphWithoutFingerprint,
      fingerprint: graphFingerprint(graphWithoutFingerprint),
    });
  } catch (error) {
    if (error instanceof GraphError) throw error;
    if (error instanceof ShapeContractError) {
      const prefix = `${error.path}: `;
      const message = error.message.startsWith(prefix) ? error.message.slice(prefix.length) : error.message;
      throw new GraphError(error.path, message, 'INVALID_GRAPH', error);
    }
    throw error;
  }
}
