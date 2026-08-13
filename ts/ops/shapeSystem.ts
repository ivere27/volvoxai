import type { RuntimeDType, RuntimeTypedArray } from '../types.js';
import {
  inspectDeviceTensorReference,
  type DeviceTensorReference,
} from './deviceTensorReference.js';

/**
 * Symbol names deliberately use an ASCII, locale-independent grammar. Graph
 * loading can therefore canonicalize and compare them identically in every
 * runtime implementation.
 */
export const SHAPE_SYMBOL_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;

export type ShapeContractErrorCode =
  | 'ARITHMETIC_INVALID'
  | 'ARITHMETIC_OVERFLOW'
  | 'INVALID_CONSTRAINT'
  | 'DUPLICATE_SYMBOL'
  | 'UNKNOWN_SYMBOL'
  | 'INVALID_INPUT_NAME'
  | 'DUPLICATE_INPUT'
  | 'INVALID_DTYPE'
  | 'INVALID_SHAPE_SPEC'
  | 'INVALID_INPUT_SET'
  | 'INVALID_TENSOR_VIEW'
  | 'DTYPE_MISMATCH'
  | 'RANK_MISMATCH'
  | 'DIMENSION_MISMATCH'
  | 'BOUND_VIOLATION'
  | 'MULTIPLE_OF_VIOLATION'
  | 'SYMBOL_CONFLICT'
  | 'BYTE_LENGTH_MISMATCH';

/** Stable, inspectable failure for logical-shape construction and binding. */
export class ShapeContractError extends Error {
  readonly code: ShapeContractErrorCode;
  readonly path: string;

  constructor(code: ShapeContractErrorCode, path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'ShapeContractError';
    this.code = code;
    this.path = path;
  }
}

function fail(
  code: ShapeContractErrorCode,
  path: string,
  message: string,
): never {
  throw new ShapeContractError(code, path, message);
}

function assertSafeInteger(value: number, path: string): void {
  if (!Number.isSafeInteger(value)) {
    fail('ARITHMETIC_INVALID', path, 'must be a safe integer.');
  }
}

function assertPositiveSafeInteger(value: number, path: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail('INVALID_SHAPE_SPEC', path, 'must be a positive safe integer.');
  }
}

/** Add two signed safe integers without first producing an unsafe value. */
export function checkedShapeAdd(left: number, right: number, path = 'shape addition'): number {
  assertSafeInteger(left, `${path} left operand`);
  assertSafeInteger(right, `${path} right operand`);
  if ((right > 0 && left > Number.MAX_SAFE_INTEGER - right) ||
      (right < 0 && left < Number.MIN_SAFE_INTEGER - right)) {
    fail('ARITHMETIC_OVERFLOW', path, 'exceeds the JavaScript safe integer range.');
  }
  return left + right;
}

/** Subtract two signed safe integers without first producing an unsafe value. */
export function checkedShapeSubtract(
  left: number,
  right: number,
  path = 'shape subtraction',
): number {
  assertSafeInteger(left, `${path} left operand`);
  assertSafeInteger(right, `${path} right operand`);
  if (right === Number.MIN_SAFE_INTEGER) {
    if (left >= 0) {
      fail('ARITHMETIC_OVERFLOW', path, 'exceeds the JavaScript safe integer range.');
    }
    return left - right;
  }
  return checkedShapeAdd(left, -right, path);
}

/** Multiply two signed safe integers without first producing an unsafe value. */
export function checkedShapeMultiply(
  left: number,
  right: number,
  path = 'shape multiplication',
): number {
  assertSafeInteger(left, `${path} left operand`);
  assertSafeInteger(right, `${path} right operand`);
  if (left !== 0 && Math.abs(right) > Math.floor(Number.MAX_SAFE_INTEGER / Math.abs(left))) {
    fail('ARITHMETIC_OVERFLOW', path, 'exceeds the JavaScript safe integer range.');
  }
  return left * right;
}

/** Integer floor division for shape formula operands. */
export function checkedShapeFloorDivide(
  dividend: number,
  divisor: number,
  path = 'shape floor division',
): number {
  assertSafeInteger(dividend, `${path} dividend`);
  if (!Number.isSafeInteger(divisor) || divisor <= 0) {
    fail('ARITHMETIC_INVALID', `${path} divisor`, 'must be a positive safe integer.');
  }
  const numerator = BigInt(dividend);
  const denominator = BigInt(divisor);
  let quotient = numerator / denominator;
  if (numerator < 0n && numerator % denominator !== 0n) quotient -= 1n;
  return Number(quotient);
}

/** Integer ceil division without an overflow-prone dividend + divisor - 1. */
export function checkedShapeCeilDivide(
  dividend: number,
  divisor: number,
  path = 'shape ceil division',
): number {
  assertSafeInteger(dividend, `${path} dividend`);
  if (!Number.isSafeInteger(divisor) || divisor <= 0) {
    fail('ARITHMETIC_INVALID', `${path} divisor`, 'must be a positive safe integer.');
  }
  const numerator = BigInt(dividend);
  const denominator = BigInt(divisor);
  let quotient = numerator / denominator;
  if (numerator > 0n && numerator % denominator !== 0n) quotient += 1n;
  return Number(quotient);
}

/** Validate a concrete positive fixed-rank shape and calculate its element count. */
export function checkedShapeElementCount(
  shape: readonly number[],
  path = 'tensor shape',
): number {
  if (!Array.isArray(shape)) {
    fail('INVALID_SHAPE_SPEC', path, 'must be an array.');
  }
  let elements = 1;
  for (let axis = 0; axis < shape.length; axis++) {
    const dimension = shape[axis];
    assertPositiveSafeInteger(dimension, `${path}[${axis}]`);
    elements = checkedShapeMultiply(elements, dimension, `${path} element count`);
  }
  return elements;
}

const DTYPE_BYTES: Readonly<Record<RuntimeDType, number>> = Object.freeze({
  float32: Float32Array.BYTES_PER_ELEMENT,
  int32: Int32Array.BYTES_PER_ELEMENT,
  int8: Int8Array.BYTES_PER_ELEMENT,
  uint8: Uint8Array.BYTES_PER_ELEMENT,
});

function isRuntimeDType(value: unknown): value is RuntimeDType {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(DTYPE_BYTES, value);
}

export function runtimeDTypeBytes(dtype: RuntimeDType, path = 'tensor dtype'): number {
  if (!isRuntimeDType(dtype)) {
    fail('INVALID_DTYPE', path, `has unsupported runtime dtype '${String(dtype)}'.`);
  }
  return DTYPE_BYTES[dtype];
}

/** Calculate exact logical tensor bytes using only checked arithmetic. */
export function checkedTensorByteLength(
  shape: readonly number[],
  dtype: RuntimeDType,
  path = 'tensor',
): number {
  return checkedShapeMultiply(
    checkedShapeElementCount(shape, `${path} shape`),
    runtimeDTypeBytes(dtype, `${path} dtype`),
    `${path} byte length`,
  );
}

export interface DimensionConstraintInput {
  readonly name: string;
  readonly min: number;
  readonly max: number;
  readonly multiple_of?: number;
}

export interface DimensionConstraint {
  readonly name: string;
  readonly min: number;
  readonly max: number;
  readonly multiple_of?: number;
}

const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder('utf-8', { fatal: true });

function canonicalUtf8Bytes(value: string, path: string): Uint8Array {
  const bytes = UTF8_ENCODER.encode(value);
  if (UTF8_DECODER.decode(bytes) !== value) {
    fail('INVALID_INPUT_NAME', path, 'must be a well-formed Unicode string.');
  }
  return bytes;
}

/** Compare strings by unsigned UTF-8 bytes, independent of locale. */
function canonicalStringCompare(left: string, right: string): number {
  if (left === right) return 0;
  const leftBytes = canonicalUtf8Bytes(left, 'canonical name');
  const rightBytes = canonicalUtf8Bytes(right, 'canonical name');
  const common = Math.min(leftBytes.length, rightBytes.length);
  for (let index = 0; index < common; index++) {
    if (leftBytes[index] !== rightBytes[index]) return leftBytes[index] - rightBytes[index];
  }
  return leftBytes.length - rightBytes.length;
}

function ownPropertyNames(value: object, path: string): string[] {
  if (Object.getOwnPropertySymbols(value).length !== 0) {
    fail('INVALID_CONSTRAINT', path, 'must not contain symbol-keyed fields.');
  }
  return Object.getOwnPropertyNames(value);
}

function normalizeConstraint(
  source: DimensionConstraintInput,
  index: number,
): DimensionConstraint {
  const path = `dimension constraints[${index}]`;
  if (source == null || typeof source !== 'object' || Array.isArray(source)) {
    fail('INVALID_CONSTRAINT', path, 'must be an object.');
  }
  const fields = ownPropertyNames(source, path);
  const unexpected = fields.filter((field) =>
    field !== 'name' && field !== 'min' && field !== 'max' && field !== 'multiple_of');
  if (unexpected.length !== 0) {
    fail('INVALID_CONSTRAINT', path, `has unsupported field '${unexpected.sort()[0]}'.`);
  }
  if (typeof source.name !== 'string' || !SHAPE_SYMBOL_PATTERN.test(source.name)) {
    fail(
      'INVALID_CONSTRAINT',
      `${path}.name`,
      'must match /^[A-Za-z][A-Za-z0-9_]{0,63}$/.',
    );
  }
  if (!Number.isSafeInteger(source.min) || source.min <= 0) {
    fail('INVALID_CONSTRAINT', `${path}.min`, 'must be a positive safe integer.');
  }
  if (!Number.isSafeInteger(source.max) || source.max <= 0) {
    fail('INVALID_CONSTRAINT', `${path}.max`, 'must be a positive safe integer.');
  }
  if (source.min > source.max) {
    fail('INVALID_CONSTRAINT', path, `has min ${source.min} greater than max ${source.max}.`);
  }

  let multipleOf: number | undefined;
  if (source.multiple_of !== undefined) {
    if (!Number.isSafeInteger(source.multiple_of) || source.multiple_of <= 0) {
      fail('INVALID_CONSTRAINT', `${path}.multiple_of`, 'must be a positive safe integer.');
    }
    multipleOf = source.multiple_of;
    const remainder = source.min % multipleOf;
    const adjustment = remainder === 0 ? 0 : multipleOf - remainder;
    if (adjustment > source.max - source.min) {
      fail(
        'INVALID_CONSTRAINT',
        path,
        `has no multiple of ${multipleOf} in [${source.min}, ${source.max}].`,
      );
    }
  }

  const normalized: DimensionConstraint = multipleOf === undefined
    ? { name: source.name, min: source.min, max: source.max }
    : { name: source.name, min: source.min, max: source.max, multiple_of: multipleOf };
  return Object.freeze(normalized);
}

/** Immutable, canonical symbol table for one logical model. */
export class ShapeEnvironment {
  readonly dimensions: readonly DimensionConstraint[];
  readonly #byName: ReadonlyMap<string, DimensionConstraint>;

  constructor(constraints: readonly DimensionConstraintInput[]) {
    if (!Array.isArray(constraints)) {
      fail('INVALID_CONSTRAINT', 'dimension constraints', 'must be an array.');
    }
    const normalized = Array.from(constraints, normalizeConstraint);
    normalized.sort((left, right) => canonicalStringCompare(left.name, right.name));
    const byName = new Map<string, DimensionConstraint>();
    for (const constraint of normalized) {
      if (byName.has(constraint.name)) {
        fail(
          'DUPLICATE_SYMBOL',
          `dimension '${constraint.name}'`,
          `is declared more than once.`,
        );
      }
      byName.set(constraint.name, constraint);
    }
    this.dimensions = Object.freeze(normalized);
    this.#byName = byName;
    Object.freeze(this);
  }

  get(name: string): DimensionConstraint | undefined {
    return this.#byName.get(name);
  }
}

export type ShapeDimensionSpec = number | string;
export type TensorShapeSpec = readonly ShapeDimensionSpec[];

/** Clone, validate, and freeze one constant/symbolic fixed-rank shape. */
export function createTensorShapeSpec(
  shape: readonly ShapeDimensionSpec[],
  environment: ShapeEnvironment,
  path = 'tensor shape spec',
): TensorShapeSpec {
  if (!(environment instanceof ShapeEnvironment)) {
    fail('INVALID_SHAPE_SPEC', path, 'requires a ShapeEnvironment.');
  }
  if (!Array.isArray(shape)) {
    fail('INVALID_SHAPE_SPEC', path, 'must be an array.');
  }
  const normalized = Array.from(shape, (dimension, axis) => {
    if (typeof dimension === 'number') {
      assertPositiveSafeInteger(dimension, `${path}[${axis}]`);
      return dimension;
    }
    if (typeof dimension !== 'string' || !SHAPE_SYMBOL_PATTERN.test(dimension)) {
      fail(
        'INVALID_SHAPE_SPEC',
        `${path}[${axis}]`,
        'must be a positive safe integer or a valid symbol name.',
      );
    }
    if (!environment.get(dimension)) {
      fail('UNKNOWN_SYMBOL', `${path}[${axis}]`, `references undeclared symbol '${dimension}'.`);
    }
    return dimension;
  });
  return Object.freeze(normalized);
}

export interface PublicInputShapeSpecInput {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly ShapeDimensionSpec[];
}

export interface PublicInputShapeSpec {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: TensorShapeSpec;
}

function normalizeInputSpec(
  source: PublicInputShapeSpecInput,
  index: number,
  environment: ShapeEnvironment,
): PublicInputShapeSpec {
  const path = `public inputs[${index}]`;
  if (source == null || typeof source !== 'object' || Array.isArray(source)) {
    fail('INVALID_INPUT_NAME', path, 'must be an object.');
  }
  if (typeof source.name !== 'string' || source.name.length === 0) {
    fail('INVALID_INPUT_NAME', `${path}.name`, 'must be a non-empty string.');
  }
  if (!isRuntimeDType(source.dtype)) {
    fail('INVALID_DTYPE', `${path}.dtype`, `has unsupported runtime dtype '${String(source.dtype)}'.`);
  }
  return Object.freeze({
    name: source.name,
    dtype: source.dtype,
    shape: createTensorShapeSpec(source.shape, environment, `${path}.shape`),
  });
}

/** Immutable public-input portion of a logical graph shape contract. */
export class PublicInputShapeContract {
  readonly environment: ShapeEnvironment;
  readonly inputs: readonly PublicInputShapeSpec[];
  readonly #byName: ReadonlyMap<string, PublicInputShapeSpec>;

  constructor(
    environment: ShapeEnvironment,
    inputs: readonly PublicInputShapeSpecInput[],
  ) {
    if (!(environment instanceof ShapeEnvironment)) {
      fail('INVALID_CONSTRAINT', 'shape environment', 'must be a ShapeEnvironment.');
    }
    if (!Array.isArray(inputs)) {
      fail('INVALID_INPUT_NAME', 'public inputs', 'must be an array.');
    }
    const normalized = Array.from(inputs, (input, index) =>
      normalizeInputSpec(input, index, environment));
    normalized.sort((left, right) => canonicalStringCompare(left.name, right.name));
    const byName = new Map<string, PublicInputShapeSpec>();
    for (const input of normalized) {
      if (byName.has(input.name)) {
        fail('DUPLICATE_INPUT', `public input '${input.name}'`, 'is declared more than once.');
      }
      byName.set(input.name, input);
    }
    this.environment = environment;
    this.inputs = Object.freeze(normalized);
    this.#byName = byName;
    Object.freeze(this);
  }

  get(name: string): PublicInputShapeSpec | undefined {
    return this.#byName.get(name);
  }
}

/** Dynamic execution values always carry explicit concrete shape metadata. */
export interface ShapedRuntimeTensorView<
  TData extends RuntimeTypedArray | DeviceTensorReference = RuntimeTypedArray,
> {
  readonly data: TData;
  readonly shape: readonly number[];
}

export type ShapedExecutionTensorView = ShapedRuntimeTensorView<
  RuntimeTypedArray | DeviceTensorReference
>;

export interface BoundPublicInput<
  TData extends RuntimeTypedArray | DeviceTensorReference =
    RuntimeTypedArray | DeviceTensorReference,
> {
  readonly name: string;
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly elementCount: number;
  readonly sizeBytes: number;
  /** Caller-owned tensor contents; binding never writes through this view. */
  readonly data: TData;
}

export interface ShapeBinding<
  TData extends RuntimeTypedArray | DeviceTensorReference =
    RuntimeTypedArray | DeviceTensorReference,
> {
  readonly inputs: Readonly<Record<string, BoundPublicInput<TData>>>;
  readonly symbols: Readonly<Record<string, number>>;
  readonly signature: string;
}

function typedArrayDType(value: unknown): RuntimeDType | null {
  if (value instanceof Float32Array) return 'float32';
  if (value instanceof Int32Array) return 'int32';
  if (value instanceof Int8Array) return 'int8';
  if (value instanceof Uint8Array || value instanceof Uint8ClampedArray) return 'uint8';
  return null;
}

function validateInputSet<TData extends RuntimeTypedArray | DeviceTensorReference>(
  contract: PublicInputShapeContract,
  values: Readonly<Record<string, ShapedRuntimeTensorView<TData>>>,
): string[] {
  if (values == null || typeof values !== 'object' || Array.isArray(values) || ArrayBuffer.isView(values)) {
    fail('INVALID_INPUT_SET', 'execution inputs', 'must be a named tensor-view record.');
  }
  if (Object.getOwnPropertySymbols(values).length !== 0) {
    fail('INVALID_INPUT_SET', 'execution inputs', 'must not contain symbol-keyed inputs.');
  }
  const actualNames = Object.getOwnPropertyNames(values).sort(canonicalStringCompare);
  const expectedNames = contract.inputs.map((input) => input.name);
  const expected = new Set(expectedNames);
  const actual = new Set(actualNames);
  const missing = expectedNames.filter((name) => !actual.has(name));
  const unexpected = actualNames.filter((name) => !expected.has(name));
  if (missing.length !== 0 || unexpected.length !== 0) {
    const details: string[] = [];
    if (missing.length !== 0) details.push(`missing [${missing.join(', ')}]`);
    if (unexpected.length !== 0) details.push(`unexpected [${unexpected.join(', ')}]`);
    fail(
      'INVALID_INPUT_SET',
      'execution inputs',
      `must contain exactly the declared public inputs; ${details.join('; ')}.`,
    );
  }
  return actualNames;
}

interface CanonicalShapeInput {
  readonly name: string;
  readonly shape: readonly number[];
}

/**
 * Canonical ordered signature from the accepted v1 contract. Name byte lengths
 * make separators inside a name unambiguous, and unsigned UTF-8 sorting avoids
 * object insertion order and locale dependencies.
 */
export function canonicalShapeSignature(inputs: readonly CanonicalShapeInput[]): string {
  if (!Array.isArray(inputs)) {
    fail('INVALID_INPUT_SET', 'shape signature inputs', 'must be an array.');
  }
  const seen = new Set<string>();
  const canonical = Array.from(inputs, (input, index) => {
    const path = `shape signature inputs[${index}]`;
    if (input == null || typeof input !== 'object' || Array.isArray(input)) {
      fail('INVALID_INPUT_SET', path, 'must be an object.');
    }
    if (typeof input.name !== 'string' || input.name.length === 0) {
      fail('INVALID_INPUT_NAME', `${path}.name`, 'must be a non-empty string.');
    }
    if (seen.has(input.name)) {
      fail('DUPLICATE_INPUT', `${path}.name`, `duplicates public input '${input.name}'.`);
    }
    seen.add(input.name);
    checkedShapeElementCount(input.shape, `${path}.shape`);
    const nameBytes = canonicalUtf8Bytes(input.name, `${path}.name`);
    return { name: input.name, nameBytes, shape: [...input.shape] };
  });
  canonical.sort((left, right) => canonicalStringCompare(left.name, right.name));
  const fields = canonical.flatMap((input) => [
    `${input.nameBytes.byteLength}:${input.name}`,
    `${input.shape.length}:${input.shape.join(',')}`,
  ]);
  return `v1|${fields.join('|')}`;
}

function freezeRecord<T>(entries: readonly (readonly [string, T])[]): Readonly<Record<string, T>> {
  return Object.freeze(Object.fromEntries(entries)) as Readonly<Record<string, T>>;
}

/**
 * Atomically validate and bind the complete public-input set. No caller shape
 * or data is mutated, and no partial binding is returned after a failure.
 */
export function bindPublicInputShapes<
  TData extends RuntimeTypedArray | DeviceTensorReference,
>(
  contract: PublicInputShapeContract,
  values: Readonly<Record<string, ShapedRuntimeTensorView<TData>>>,
): ShapeBinding<TData> {
  if (!(contract instanceof PublicInputShapeContract)) {
    fail('INVALID_INPUT_SET', 'shape contract', 'must be a PublicInputShapeContract.');
  }
  validateInputSet(contract, values);

  const symbols = new Map<string, number>();
  const boundEntries: Array<readonly [string, BoundPublicInput<TData>]> = [];
  for (const input of contract.inputs) {
    const path = `execution input '${input.name}'`;
    const view = values[input.name];
    if (view == null || typeof view !== 'object' || Array.isArray(view)) {
      fail('INVALID_TENSOR_VIEW', path, 'must be an object containing data and shape.');
    }
    const deviceReference = inspectDeviceTensorReference(view.data);
    const actualDType = typedArrayDType(view.data) ?? deviceReference?.dtype ?? null;
    if (actualDType == null) {
      fail('INVALID_TENSOR_VIEW', `${path}.data`,
        'must be a supported runtime typed array or a live device TensorResult.');
    }
    if (actualDType !== input.dtype) {
      fail(
        'DTYPE_MISMATCH',
        `${path}.data`,
        `has dtype '${actualDType}', but the logical descriptor requires '${input.dtype}'.`,
      );
    }
    if (!Array.isArray(view.shape)) {
      fail('INVALID_TENSOR_VIEW', `${path}.shape`, 'must be an explicit number array.');
    }
    if (view.shape.length !== input.shape.length) {
      fail(
        'RANK_MISMATCH',
        `${path}.shape`,
        `has rank ${view.shape.length}, but the logical descriptor requires rank ${input.shape.length}.`,
      );
    }

    const concreteShape = Array.from(view.shape, (dimension, axis) => {
      const dimensionPath = `${path}.shape[${axis}]`;
      if (!Number.isSafeInteger(dimension) || dimension <= 0) {
        fail('INVALID_TENSOR_VIEW', dimensionPath, 'must be a positive safe integer.');
      }
      const expected = input.shape[axis];
      if (typeof expected === 'number') {
        if (dimension !== expected) {
          fail(
            'DIMENSION_MISMATCH',
            dimensionPath,
            `is ${dimension}, but the logical descriptor requires constant ${expected}.`,
          );
        }
        return dimension;
      }

      const constraint = contract.environment.get(expected)!;
      if (dimension < constraint.min || dimension > constraint.max) {
        fail(
          'BOUND_VIOLATION',
          dimensionPath,
          `binds '${expected}' to ${dimension}, outside [${constraint.min}, ${constraint.max}].`,
        );
      }
      if (constraint.multiple_of !== undefined && dimension % constraint.multiple_of !== 0) {
        fail(
          'MULTIPLE_OF_VIOLATION',
          dimensionPath,
          `binds '${expected}' to ${dimension}, which is not a multiple of ${constraint.multiple_of}.`,
        );
      }
      const previous = symbols.get(expected);
      if (previous !== undefined && previous !== dimension) {
        fail(
          'SYMBOL_CONFLICT',
          dimensionPath,
          `binds '${expected}' to ${dimension}, but it was already bound to ${previous}.`,
        );
      }
      symbols.set(expected, dimension);
      return dimension;
    });

    const shape = Object.freeze(concreteShape);
    if (deviceReference && (deviceReference.shape.length !== shape.length ||
        deviceReference.shape.some((dimension, axis) => dimension !== shape[axis]))) {
      fail(
        'DIMENSION_MISMATCH',
        `${path}.shape`,
        `does not match the device result shape [${deviceReference.shape.join(', ')}].`,
      );
    }
    const elementCount = checkedShapeElementCount(shape, `${path}.shape`);
    const sizeBytes = checkedShapeMultiply(
      elementCount,
      runtimeDTypeBytes(input.dtype, `${path}.dtype`),
      `${path} byte length`,
    );
    const actualBytes = deviceReference?.logicalSizeBytes ??
      (view.data as RuntimeTypedArray).byteLength;
    if (actualBytes !== sizeBytes) {
      fail(
        'BYTE_LENGTH_MISMATCH',
        `${path}.data`,
        `has ${actualBytes} bytes, but shape [${shape.join(', ')}] and dtype '${input.dtype}' require ${sizeBytes}.`,
      );
    }
    boundEntries.push([input.name, Object.freeze({
      name: input.name,
      dtype: input.dtype,
      shape,
      elementCount,
      sizeBytes,
      data: view.data,
    })]);
  }

  const symbolEntries = [...symbols.entries()].sort(([left], [right]) =>
    canonicalStringCompare(left, right));
  const inputs = freezeRecord(boundEntries);
  const binding: ShapeBinding<TData> = {
    inputs,
    symbols: freezeRecord(symbolEntries),
    signature: canonicalShapeSignature(Object.values(inputs)),
  };
  return Object.freeze(binding);
}
