import { SafetensorsFile } from '../core/Safetensors.js';
import { Model } from '../core/Model.js';
import { checkpointGraphDocument } from './ModelCheckpoint.js';
import type { ShapedRuntimeTensorView } from '../ops/shapeSystem.js';
import type {
  RuntimeDType,
  RuntimeTypedArray,
  SerializedAffineQuantizationReference,
} from '../types.js';
import type { PtqSchemeValue } from '../generated/volvoxaiFullEnums.js';

export const VOLVOX_PTQ_FORMAT = 'volvox.ptq.v1';

const F32_MIN_NORMAL = 1.1754943508222875e-38;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

export type PTQDType = Extract<RuntimeDType, 'int8' | 'uint8'>;
export type PTQScheme = PtqSchemeValue;
type NumericArray = ArrayLike<number>;
type QuantizedArray = Int8Array | Uint8Array;
type QuantizedArrayConstructor = Int8ArrayConstructor | Uint8ArrayConstructor;

export interface PTQRange {
  minimum: number;
  maximum: number;
  sampleCount: number;
}

export interface PTQParameterOptions {
  dtype?: PTQDType;
  scheme?: PTQScheme;
}

export interface PTQParameters {
  dtype: PTQDType;
  scheme: PTQScheme;
  scale: number;
  zero_point: number;
  observed_min: number;
  observed_max: number;
  sample_count: number;
}

export interface PTQWeightRequest {
  name: string;
  axis?: number;
  outputName?: string;
  scaleName?: string;
  zeroPointName?: string;
  bias?: string;
  biasOutputName?: string;
  inputScale?: number;
}

export interface PTQMaterializeOptions {
  metadata?: Record<string, string>;
  includeUnselected?: boolean;
  coverage: PTQCoverageReport;
}

export interface PTQCalibrationBatchChunk {
  /** Stable identity of one logical shaped batch across all promotion chunks. */
  readonly batchId: string;
  readonly profile: string;
  readonly samples: number;
  /** Zero-based index within this logical batch's complete chunk set. */
  readonly chunkIndex: number;
  /** Exact number of chunks that will cover the declared activation universe. */
  readonly chunkCount: number;
  readonly inputs: Readonly<Record<string, ShapedRuntimeTensorView>>;
  readonly activations: Readonly<Record<string, ShapedRuntimeTensorView>>;
}

export interface PTQSymbolCoverage {
  readonly minimum: number;
  readonly maximum: number;
}

export interface PTQProfileCoverage {
  readonly name: string;
  readonly batches: number;
  readonly samples: number;
  readonly signatures: readonly string[];
  readonly symbols: Readonly<Record<string, PTQSymbolCoverage>>;
  readonly activationSamples: Readonly<Record<string, number>>;
}

export interface PTQCoverageReport {
  readonly format: 'volvox.ptq-coverage/v1';
  readonly logicalFingerprint: string;
  readonly complete: boolean;
  readonly totalBatches: number;
  readonly totalSamples: number;
  readonly profiles: readonly PTQProfileCoverage[];
}

export interface PTQCalibratorOptions {
  /** Every declared name must receive at least one shape-bearing batch. */
  readonly profiles: readonly string[];
  /** Every completed logical batch must observe exactly these F32 tensors once. */
  readonly activations: readonly string[];
}

function quantizedDomain(dtype: PTQDType): {
  minimum: number;
  maximum: number;
  ArrayType: QuantizedArrayConstructor;
} {
  if (dtype === 'int8') return { minimum: -128, maximum: 127, ArrayType: Int8Array };
  if (dtype === 'uint8') return { minimum: 0, maximum: 255, ArrayType: Uint8Array };
  throw new Error(`PTQ supports int8 or uint8 storage, got '${dtype}'.`);
}

function roundTiesToEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function positiveF32(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be finite and positive.`);
  }
  const rounded = Math.fround(value);
  if (Number.isFinite(rounded) && rounded > 0) return rounded;
  // Preserve a valid descriptor for finite subnormal calibration ranges.
  if (rounded === 0) return F32_MIN_NORMAL;
  throw new Error(`${label} is not representable as finite F32.`);
}

function valuesView(values: unknown, label: string): NumericArray {
  if (!Array.isArray(values) && (!ArrayBuffer.isView(values) || values instanceof DataView)) {
    throw new Error(`${label} must be an array or typed array.`);
  }
  return values as NumericArray;
}

function finiteRange(values: unknown, label: string) {
  const view = valuesView(values, label);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = 0; index < view.length; index++) {
    const value = view[index];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new Error(`${label} contains a non-finite value at index ${index}.`);
    }
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return { minimum, maximum, count: view.length };
}

function normalizeRange(range: PTQObserver | PTQRange | Record<string, any>): PTQRange {
  if (range instanceof PTQObserver) return range.snapshot();
  if (!range || typeof range !== 'object' || Array.isArray(range)) {
    throw new Error('PTQ range must be a PTQObserver or range object.');
  }
  const input = range as Record<string, any>;
  const minimum = input.minimum;
  const maximum = input.maximum;
  const sampleCount = input.sampleCount ?? input.sample_count ?? input.count;
  if (typeof minimum !== 'number' || !Number.isFinite(minimum) ||
      typeof maximum !== 'number' || !Number.isFinite(maximum) || minimum > maximum ||
      !Number.isSafeInteger(sampleCount) || sampleCount <= 0) {
    throw new Error('PTQ range requires finite ordered bounds and a positive sample count.');
  }
  return { minimum, maximum, sampleCount };
}

function canonicalParameters(parameters: PTQParameters | Record<string, any>) {
  if (!parameters || typeof parameters !== 'object' || Array.isArray(parameters)) {
    throw new Error('PTQ parameters must be an object.');
  }
  const input = parameters as Record<string, any>;
  const dtype = input.dtype as PTQDType;
  const domain = quantizedDomain(dtype);
  const scale = positiveF32(input.scale, 'PTQ scale');
  const zeroPoint = input.zero_point ?? input.zeroPoint;
  if (!Number.isInteger(zeroPoint) || zeroPoint < domain.minimum || zeroPoint > domain.maximum) {
    throw new Error(`PTQ zero point must be an integer in [${domain.minimum}, ${domain.maximum}].`);
  }
  return { ...parameters, dtype, scale, zero_point: zeroPoint, domain };
}

function shapeInfo(shape: readonly number[], axis: number, label: string) {
  if (!Array.isArray(shape) || shape.length === 0 || shape.length > 8) {
    throw new Error(`${label} shape must have rank 1 through 8.`);
  }
  let elements = 1;
  for (let index = 0; index < shape.length; index++) {
    const dimension = shape[index];
    if (!Number.isSafeInteger(dimension) || dimension <= 0) {
      throw new Error(`${label} shape has invalid dimension ${index}.`);
    }
    elements *= dimension;
    if (!Number.isSafeInteger(elements)) throw new Error(`${label} is too large.`);
  }
  let normalizedAxis = axis;
  if (!Number.isInteger(normalizedAxis)) throw new Error(`${label} axis must be an integer.`);
  if (normalizedAxis < 0) normalizedAxis += shape.length;
  if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
    throw new Error(`${label} axis is outside rank ${shape.length}.`);
  }
  let inner = 1;
  for (let index = normalizedAxis + 1; index < shape.length; index++) inner *= shape[index];
  return { elements, axis: normalizedAxis, channels: shape[normalizedAxis], inner };
}

function safetensorsDType(dtype: RuntimeDType): 'F32' | 'I32' | 'I8' | 'U8' {
  if (dtype === 'float32') return 'F32';
  if (dtype === 'int32') return 'I32';
  if (dtype === 'int8') return 'I8';
  if (dtype === 'uint8') return 'U8';
  throw new Error(`PTQ safetensors export does not support graph dtype '${dtype}'.`);
}

function bytesOf(buffer: unknown, label: string): Uint8Array {
  if (!ArrayBuffer.isView(buffer) || buffer instanceof DataView) {
    throw new Error(`${label} requires typed-array CPU storage.`);
  }
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

/** Mutable min/max observer intended to be reused across calibration forwards. */
export class PTQObserver {
  declare minimum: number;
  declare maximum: number;
  declare sampleCount: number;

  constructor() {
    this.reset();
  }

  reset() {
    this.minimum = Infinity;
    this.maximum = -Infinity;
    this.sampleCount = 0;
    return this;
  }

  observe(values: NumericArray) {
    const next = finiteRange(values, 'PTQ observation');
    if (next.count === 0) return this;
    const sampleCount = this.sampleCount + next.count;
    if (!Number.isSafeInteger(sampleCount)) throw new Error('PTQ observation count exceeds the safe integer range.');
    this.minimum = Math.min(this.minimum, next.minimum);
    this.maximum = Math.max(this.maximum, next.maximum);
    this.sampleCount = sampleCount;
    return this;
  }

  snapshot() {
    if (this.sampleCount <= 0) throw new Error('PTQ observer has no observations.');
    return Object.freeze({
      minimum: this.minimum,
      maximum: this.maximum,
      sampleCount: this.sampleCount,
    });
  }

  parameters(options: PTQParameterOptions = {}) {
    return derivePTQParameters(this, options);
  }
}

/** Derive one affine activation descriptor from an observed range. */
export function derivePTQParameters(
  range: PTQObserver | PTQRange | Record<string, any>,
  { dtype = 'int8', scheme = 'symmetric' }: PTQParameterOptions = {},
): Readonly<PTQParameters> {
  const observed = normalizeRange(range);
  const domain = quantizedDomain(dtype);
  let scale;
  let zeroPoint;
  if (scheme === 'symmetric') {
    const maximumMagnitude = Math.max(Math.abs(observed.minimum), Math.abs(observed.maximum));
    scale = maximumMagnitude === 0 ? 1 : maximumMagnitude / 127;
    zeroPoint = dtype === 'int8' ? 0 : 128;
  } else if (scheme === 'asymmetric') {
    const minimum = Math.min(observed.minimum, 0);
    const maximum = Math.max(observed.maximum, 0);
    if (minimum === maximum) {
      scale = 1;
      zeroPoint = 0;
    } else {
      scale = (maximum - minimum) / (domain.maximum - domain.minimum);
      const roundedScale = positiveF32(scale, 'PTQ scale');
      zeroPoint = roundTiesToEven(domain.minimum - minimum / roundedScale);
      zeroPoint = Math.max(domain.minimum, Math.min(domain.maximum, zeroPoint));
      scale = roundedScale;
    }
  } else {
    throw new Error(`Unsupported PTQ scheme '${scheme}'.`);
  }
  return Object.freeze({
    dtype,
    scheme,
    scale: positiveF32(scale, 'PTQ scale'),
    zero_point: zeroPoint,
    observed_min: observed.minimum,
    observed_max: observed.maximum,
    sample_count: observed.sampleCount,
  });
}

/**
 * Quantize an F32-like array with a per-tensor descriptor. I8 output uses the
 * full physical storage range [-128, 127]; symmetric parameters derived by
 * this module normally map their observed range into [-127, 127].
 */
export function quantizePTQ(
  values: NumericArray,
  parameters: PTQParameters | Record<string, any>,
): { data: QuantizedArray; saturationCount: number } {
  const sourceRange = finiteRange(values, 'PTQ quantization input');
  const canonical = canonicalParameters(parameters);
  const output = new canonical.domain.ArrayType(sourceRange.count);
  let saturationCount = 0;
  for (let index = 0; index < sourceRange.count; index++) {
    const unbounded = roundTiesToEven(values[index] / canonical.scale + canonical.zero_point);
    if (unbounded < canonical.domain.minimum || unbounded > canonical.domain.maximum) saturationCount++;
    output[index] = Math.max(canonical.domain.minimum, Math.min(canonical.domain.maximum, unbounded));
  }
  return { data: output, saturationCount };
}

/**
 * Pack row-major F32 weights as canonical symmetric per-axis narrow-range I8
 * storage [-127, 127], balanced around zero point 0.
 */
export function packPTQWeight(
  values: NumericArray,
  shape: readonly number[],
  { axis = 0, name = null }: { axis?: number; name?: string | null } = {},
) {
  const info = shapeInfo(shape, axis, 'PTQ weight');
  const sourceRange = finiteRange(values, 'PTQ weight');
  if (sourceRange.count !== info.elements) {
    throw new Error(`PTQ weight has ${sourceRange.count} values, expected ${info.elements}.`);
  }
  const maxima = new Float64Array(info.channels);
  for (let index = 0; index < info.elements; index++) {
    const channel = Math.floor(index / info.inner) % info.channels;
    maxima[channel] = Math.max(maxima[channel], Math.abs(values[index]));
  }
  const scales = new Float32Array(info.channels);
  for (let channel = 0; channel < info.channels; channel++) {
    scales[channel] = positiveF32(maxima[channel] === 0 ? 1 : maxima[channel] / 127,
      `PTQ weight scale ${channel}`);
  }
  const data = new Int8Array(info.elements);
  let saturationCount = 0;
  for (let index = 0; index < info.elements; index++) {
    const channel = Math.floor(index / info.inner) % info.channels;
    const unbounded = roundTiesToEven(values[index] / scales[channel]);
    if (unbounded < -127 || unbounded > 127) saturationCount++;
    data[index] = Math.max(-127, Math.min(127, unbounded));
  }
  return Object.freeze({
    ...(name == null ? {} : { name }),
    shape: Object.freeze([...shape]),
    dtype: 'int8',
    data,
    scales,
    saturationCount,
    quantization: Object.freeze({
      scheme: 'per_axis',
      axis: info.axis,
      scales: Object.freeze([...scales]),
      zero_points: Object.freeze(new Array(info.channels).fill(0)),
    }),
  });
}

/** Convert F32 bias to the I32 accumulator domain for QLinear/QConv. */
export function packPTQBias(
  values: NumericArray,
  inputScale: number,
  weightScales: NumericArray,
): Int32Array {
  finiteRange(values, 'PTQ bias');
  valuesView(weightScales, 'PTQ weight scales');
  if (values.length !== weightScales.length) {
    throw new Error('PTQ bias and per-output-channel weight scale counts must match.');
  }
  const activationScale = positiveF32(inputScale, 'PTQ input activation scale');
  const output = new Int32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const weightScale = positiveF32(weightScales[index], `PTQ weight scale ${index}`);
    const accumulatorScale = positiveF32(Math.fround(activationScale * weightScale),
      `PTQ accumulator scale ${index}`);
    const quantized = roundTiesToEven(Math.fround(values[index] / accumulatorScale));
    if (!Number.isSafeInteger(quantized) || quantized < I32_MIN || quantized > I32_MAX) {
      throw new Error(`PTQ bias ${index} is outside the I32 accumulator range.`);
    }
    output[index] = quantized;
  }
  return output;
}

interface MutablePTQProfileCoverage {
  batches: number;
  samples: number;
  signatures: Set<string>;
  symbols: Map<string, { minimum: number; maximum: number }>;
  activationSamples: Map<string, number>;
}

interface PTQInputWitness {
  readonly dtype: RuntimeDType;
  readonly shape: readonly number[];
  readonly bytes: Uint8Array;
}

interface PendingPTQActivation {
  readonly minimum: number;
  readonly maximum: number;
  readonly count: number;
}

interface PendingPTQBatch {
  readonly id: string;
  readonly profile: string;
  readonly samples: number;
  readonly signature: string;
  readonly symbols: Readonly<Record<string, number>>;
  readonly chunkCount: number;
  readonly inputs: ReadonlyMap<string, PTQInputWitness>;
  readonly chunks: Set<number>;
  readonly activations: Map<string, PendingPTQActivation>;
}

function profileName(value: unknown, label = 'PTQ profile'): string {
  if (typeof value !== 'string' || !/^[A-Za-z][A-Za-z0-9._-]{0,63}$/.test(value)) {
    throw new Error(`${label} must match /^[A-Za-z][A-Za-z0-9._-]{0,63}$/.`);
  }
  return value;
}

function calibrationBatchId(value: unknown): string {
  if (typeof value !== 'string' ||
      !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value)) {
    throw new Error(
      'PTQ calibration batchId must match /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.',
    );
  }
  return value;
}

function sameConcreteShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function checkedCoverageAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) throw new Error(`${label} exceeds the safe integer range.`);
  return result;
}

function captureInputWitness(
  snapshot: Model,
  plan: ReturnType<Model['bindShapes']>,
  inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
): ReadonlyMap<string, PTQInputWitness> {
  return new Map(Object.keys(snapshot.graph.inputs).sort().map((name) => {
    const descriptor = plan.tensors[name];
    const view = inputs[name];
    return [name, Object.freeze({
      dtype: descriptor.dtype,
      shape: Object.freeze([...descriptor.shape]),
      bytes: bytesOf(view.data, `PTQ input '${name}'`).slice(),
    })];
  }));
}

function sameInputWitness(
  expected: ReadonlyMap<string, PTQInputWitness>,
  snapshot: Model,
  plan: ReturnType<Model['bindShapes']>,
  inputs: Readonly<Record<string, ShapedRuntimeTensorView>>,
): boolean {
  const names = Object.keys(snapshot.graph.inputs).sort();
  if (names.length !== expected.size || names.some((name) => !expected.has(name))) return false;
  for (const name of names) {
    const witness = expected.get(name)!;
    const descriptor = plan.tensors[name];
    const bytes = bytesOf(inputs[name].data, `PTQ input '${name}'`);
    if (witness.dtype !== descriptor.dtype ||
        !sameConcreteShape(witness.shape, descriptor.shape) ||
        witness.bytes.byteLength !== bytes.byteLength) return false;
    for (let index = 0; index < bytes.byteLength; index++) {
      if (witness.bytes[index] !== bytes[index]) return false;
    }
  }
  return true;
}

function sameSymbols(
  expected: Readonly<Record<string, number>>,
  actual: Readonly<Record<string, number>>,
): boolean {
  const expectedNames = Object.keys(expected).sort();
  const actualNames = Object.keys(actual).sort();
  return expectedNames.length === actualNames.length &&
    expectedNames.every((name, index) =>
      name === actualNames[index] && expected[name] === actual[name]);
}

function clonePendingBatch(batch: PendingPTQBatch): PendingPTQBatch {
  return {
    ...batch,
    chunks: new Set(batch.chunks),
    activations: new Map(batch.activations),
  };
}

function coverageTotal(
  coverage: Iterable<MutablePTQProfileCoverage>,
  field: 'batches' | 'samples',
): number {
  let total = 0;
  for (const state of coverage) {
    total = checkedCoverageAdd(total, state[field], `PTQ total ${field} coverage`);
  }
  return total;
}

/**
 * Logical-model calibrator. Promotion chunks for one named, shape-bearing
 * logical batch remain private until they cover the exact declared activation
 * universe. Storage-only observations are intentionally not accepted.
 */
export class PTQCalibrator {
  readonly snapshot: Model;
  readonly profiles: readonly string[];
  readonly activations: readonly string[];
  readonly observers = new Map<string, PTQObserver>();
  readonly #activationSet: ReadonlySet<string>;
  readonly #coverage = new Map<string, MutablePTQProfileCoverage>();
  readonly #pendingBatches = new Map<string, PendingPTQBatch>();
  readonly #completedBatchIds = new Set<string>();

  constructor(snapshot: Model, options: PTQCalibratorOptions) {
    if (!(snapshot instanceof Model)) {
      throw new Error('PTQCalibrator requires a Model.');
    }
    const profiles = options?.profiles;
    if (!Array.isArray(profiles) || profiles.length === 0) {
      throw new Error('PTQCalibrator requires at least one named shape profile.');
    }
    const names = profiles.map((name, index) => profileName(name, `PTQ profile ${index}`));
    if (new Set(names).size !== names.length) throw new Error('PTQ profile names must be unique.');
    const activationNames = options?.activations;
    if (!Array.isArray(activationNames) || activationNames.length === 0 ||
        activationNames.some((name) => typeof name !== 'string' || !name)) {
      throw new Error('PTQCalibrator requires at least one non-empty activation name.');
    }
    if (new Set(activationNames).size !== activationNames.length) {
      throw new Error('PTQ activation names must be unique.');
    }
    for (const name of activationNames) {
      const descriptor = snapshot.graph.tensors[name];
      if (!descriptor || descriptor.dtype !== 'float32') {
        throw new Error(`PTQ activation '${name}' must name an F32 logical tensor.`);
      }
    }
    this.snapshot = snapshot;
    this.profiles = Object.freeze(names);
    this.activations = Object.freeze([...activationNames].sort());
    this.#activationSet = new Set(this.activations);
    this.reset();
  }

  reset(): this {
    this.observers.clear();
    this.#coverage.clear();
    this.#pendingBatches.clear();
    this.#completedBatchIds.clear();
    for (const name of this.profiles) {
      this.#coverage.set(name, {
        batches: 0,
        samples: 0,
        signatures: new Set(),
        symbols: new Map(),
        activationSamples: new Map(),
      });
    }
    return this;
  }

  observeBatchChunk(batch: PTQCalibrationBatchChunk): this {
    if (!batch || typeof batch !== 'object' || Array.isArray(batch)) {
      throw new Error('PTQ calibration batch chunk must be an object.');
    }
    const batchId = calibrationBatchId(batch.batchId);
    if (this.#completedBatchIds.has(batchId)) {
      throw new Error(`PTQ logical batch '${batchId}' was already completed.`);
    }
    const name = profileName(batch.profile);
    const coverage = this.#coverage.get(name);
    if (!coverage) throw new Error(`PTQ profile '${name}' was not declared by this calibrator.`);
    if (!Number.isSafeInteger(batch.samples) || batch.samples <= 0) {
      throw new Error(`PTQ profile '${name}' samples must be a positive safe integer.`);
    }
    if (!Number.isSafeInteger(batch.chunkCount) || batch.chunkCount <= 0 ||
        batch.chunkCount > this.activations.length) {
      throw new Error(
        `PTQ logical batch '${batchId}' chunkCount must be between 1 and ` +
        `${this.activations.length}.`,
      );
    }
    if (!Number.isSafeInteger(batch.chunkIndex) || batch.chunkIndex < 0 ||
        batch.chunkIndex >= batch.chunkCount) {
      throw new Error(
        `PTQ logical batch '${batchId}' chunkIndex must be in ` +
        `[0, ${batch.chunkCount}).`,
      );
    }
    const plan = this.snapshot.bindShapes(batch.inputs);
    if (!batch.activations || typeof batch.activations !== 'object' ||
        Array.isArray(batch.activations) || ArrayBuffer.isView(batch.activations)) {
      throw new Error(`PTQ profile '${name}' activations must be a shaped tensor-view record.`);
    }
    const activationNames = Object.keys(batch.activations).sort();
    if (activationNames.length === 0) {
      throw new Error(`PTQ logical batch '${batchId}' chunk requires activations.`);
    }
    const chunkActivations = new Map<string, PendingPTQActivation>();
    for (const activationName of activationNames) {
      if (!this.#activationSet.has(activationName)) {
        throw new Error(
          `PTQ activation '${activationName}' was not declared by this calibrator.`,
        );
      }
      const descriptor = plan.tensors[activationName];
      const view = batch.activations[activationName];
      if (!descriptor || descriptor.dtype !== 'float32') {
        throw new Error(
          `PTQ activation '${activationName}' must name an F32 tensor in the bound logical graph.`,
        );
      }
      if (!view || !(view.data instanceof Float32Array) || !Array.isArray(view.shape) ||
          !sameConcreteShape(view.shape, descriptor.shape) ||
          view.data.byteLength !== descriptor.sizeBytes) {
        throw new Error(
          `PTQ activation '${activationName}' must carry exact F32 data and concrete shape ` +
          `[${descriptor.shape.join(',')}].`,
        );
      }
      const range = finiteRange(view.data, `PTQ activation '${activationName}'`);
      chunkActivations.set(activationName, {
        minimum: range.minimum,
        maximum: range.maximum,
        count: range.count,
      });
    }

    const existing = this.#pendingBatches.get(batchId);
    if (existing && (existing.profile !== name || existing.samples !== batch.samples ||
        existing.chunkCount !== batch.chunkCount || existing.signature !== plan.signature ||
        !sameSymbols(existing.symbols, plan.symbols) ||
        !sameInputWitness(existing.inputs, this.snapshot, plan, batch.inputs))) {
      throw new Error(
        `PTQ logical batch '${batchId}' repeated chunks must preserve profile, samples, ` +
        'chunk count, shape binding, and exact input bytes.',
      );
    }
    const stagedBatch = existing ? clonePendingBatch(existing) : {
      id: batchId,
      profile: name,
      samples: batch.samples,
      signature: plan.signature,
      symbols: Object.freeze({ ...plan.symbols }),
      chunkCount: batch.chunkCount,
      inputs: captureInputWitness(this.snapshot, plan, batch.inputs),
      chunks: new Set<number>(),
      activations: new Map<string, PendingPTQActivation>(),
    };
    if (stagedBatch.chunks.has(batch.chunkIndex)) {
      throw new Error(
        `PTQ logical batch '${batchId}' chunk ${batch.chunkIndex} was already observed.`,
      );
    }
    for (const activationName of activationNames) {
      if (stagedBatch.activations.has(activationName)) {
        throw new Error(
          `PTQ logical batch '${batchId}' activation '${activationName}' ` +
          'was observed by more than one chunk.',
        );
      }
      stagedBatch.activations.set(activationName, chunkActivations.get(activationName)!);
    }
    stagedBatch.chunks.add(batch.chunkIndex);
    const remainingChunks = stagedBatch.chunkCount - stagedBatch.chunks.size;
    const remainingActivations = this.activations.length - stagedBatch.activations.size;
    if (remainingActivations < remainingChunks) {
      throw new Error(
        `PTQ logical batch '${batchId}' cannot cover one non-empty activation set ` +
        'with every remaining chunk.',
      );
    }
    if (remainingChunks > 0) {
      this.#pendingBatches.set(batchId, stagedBatch);
      return this;
    }
    if (remainingActivations !== 0 || this.activations.some(
      (activationName) => !stagedBatch.activations.has(activationName)
    )) {
      throw new Error(
        `PTQ logical batch '${batchId}' completed its chunks without the exact ` +
        'declared activation universe.',
      );
    }

    const nextBatches = checkedCoverageAdd(
      coverage.batches, 1, `PTQ profile '${name}' batch coverage`,
    );
    const nextSamples = checkedCoverageAdd(
      coverage.samples, batch.samples, `PTQ profile '${name}' sample coverage`,
    );
    checkedCoverageAdd(
      coverageTotal(this.#coverage.values(), 'batches'), 1, 'PTQ total batch coverage',
    );
    checkedCoverageAdd(
      coverageTotal(this.#coverage.values(), 'samples'),
      batch.samples,
      'PTQ total sample coverage',
    );
    const stagedActivations = this.activations.map((activationName) => {
      const range = stagedBatch.activations.get(activationName)!;
      const observer = this.observers.get(activationName);
      return {
        name: activationName,
        minimum: Math.min(observer?.minimum ?? Infinity, range.minimum),
        maximum: Math.max(observer?.maximum ?? -Infinity, range.maximum),
        observerSamples: checkedCoverageAdd(
          observer?.sampleCount ?? 0,
          range.count,
          `PTQ activation '${activationName}' observation count`,
        ),
        activationSamples: checkedCoverageAdd(
          coverage.activationSamples.get(activationName) ?? 0,
          range.count,
          `PTQ profile '${name}' activation '${activationName}' coverage`,
        ),
      };
    });
    const stagedSymbols = new Map(coverage.symbols);
    for (const [symbol, value] of Object.entries(plan.symbols)) {
      const prior = stagedSymbols.get(symbol);
      stagedSymbols.set(symbol, prior
        ? { minimum: Math.min(prior.minimum, value), maximum: Math.max(prior.maximum, value) }
        : { minimum: value, maximum: value });
    }

    // Commit only after the complete logical batch and every counter validate.
    for (const staged of stagedActivations) {
      const observer = this.observers.get(staged.name) ?? new PTQObserver();
      observer.minimum = staged.minimum;
      observer.maximum = staged.maximum;
      observer.sampleCount = staged.observerSamples;
      this.observers.set(staged.name, observer);
      coverage.activationSamples.set(staged.name, staged.activationSamples);
    }
    coverage.batches = nextBatches;
    coverage.samples = nextSamples;
    coverage.signatures.add(plan.signature);
    coverage.symbols = stagedSymbols;
    this.#pendingBatches.delete(batchId);
    this.#completedBatchIds.add(batchId);
    return this;
  }

  coverage(): PTQCoverageReport {
    const profiles = this.profiles.map((name) => {
      const state = this.#coverage.get(name)!;
      return Object.freeze({
        name,
        batches: state.batches,
        samples: state.samples,
        signatures: Object.freeze([...state.signatures].sort()),
        symbols: Object.freeze(Object.fromEntries([...state.symbols].sort(([left], [right]) =>
          left.localeCompare(right)).map(([symbol, value]) => [symbol, Object.freeze({ ...value })]))),
        activationSamples: Object.freeze(Object.fromEntries(
          [...state.activationSamples].sort(([left], [right]) => left.localeCompare(right)),
        )),
      });
    });
    return Object.freeze({
      format: 'volvox.ptq-coverage/v1' as const,
      logicalFingerprint: this.snapshot.definitionFingerprint,
      complete: this.#pendingBatches.size === 0 &&
        profiles.every((profile) => profile.batches > 0 && profile.samples > 0),
      totalBatches: profiles.reduce((total, profile) => total + profile.batches, 0),
      totalSamples: profiles.reduce((total, profile) => total + profile.samples, 0),
      profiles: Object.freeze(profiles),
    });
  }

  parameters(options: PTQParameterOptions = {}) {
    const coverage = this.coverage();
    if (!coverage.complete) {
      const missing = coverage.profiles.filter((profile) => profile.batches === 0)
        .map((profile) => profile.name);
      const pending = [...this.#pendingBatches.keys()].sort();
      throw new Error(
        `PTQ calibration is missing required shape profiles [${missing.join(', ')}]` +
        (pending.length > 0 ? ` and has incomplete logical batches [${pending.join(', ')}].` : '.'),
      );
    }
    return Object.freeze(Object.fromEntries(
      [...this.observers].sort(([left], [right]) => left.localeCompare(right))
        .map(([name, observer]) => [name, derivePTQParameters(observer, options)]),
    ));
  }
}

type PackedPTQWeight = ReturnType<typeof packPTQWeight>;

interface PTQMaterializedRecord {
  sourceName: string;
  outputName: string;
  scaleName: string;
  zeroPointName: string;
  packed: PackedPTQWeight;
  bias?: {
    sourceName: string;
    outputName: string;
    data: Int32Array;
  };
}

interface PTQReplacement {
  name: string;
  dtype: RuntimeDType;
  shape: number[];
  buffer: RuntimeTypedArray;
}

/**
 * Export selected F32 graph weights as a zero-dependency I8 safetensors file.
 *
 * Each request is `{ name, axis?, outputName?, scaleName?, zeroPointName?, bias?,
 * biasOutputName?, inputScale? }`. Unselected weights are copied by default.
 * The returned `quantization` table can be installed directly as the sole
 * `volvox-graph/v1` graph.quantization object. Scale and zero-point values live
 * only in the returned Safetensors file.
 */
export function materializePTQWeights(
  snapshot: Model,
  requests: Array<string | PTQWeightRequest>,
  options: PTQMaterializeOptions,
) {
  if (!(snapshot instanceof Model)) {
    throw new Error('materializePTQWeights requires a Model.');
  }
  if (!Array.isArray(requests) || requests.length === 0) {
    throw new Error('materializePTQWeights requires at least one weight request.');
  }
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      options.coverage?.format !== 'volvox.ptq-coverage/v1' ||
      options.coverage.logicalFingerprint !== snapshot.definitionFingerprint ||
      options.coverage.complete !== true || options.coverage.totalBatches <= 0 ||
      options.coverage.profiles.some((profile) => profile.batches <= 0 || profile.samples <= 0)) {
    throw new Error(
      'PTQ materialization requires complete named-profile coverage for the exact logical fingerprint.',
    );
  }
  const weightDescriptors = new Map(snapshot.weightDescriptors.map((descriptor) => [
    descriptor.name,
    descriptor,
  ]));
  const materialized: Readonly<PTQMaterializedRecord>[] = [];
  const replacements = new Map<string, PTQReplacement>();
  const consumedSources = new Set<string>();
  const quantizationTensors: Record<string, SerializedAffineQuantizationReference> = {};
  const claimedNames = new Set<string>();
  const claim = (name: string, label: string): void => {
    if (typeof name !== 'string' || !name || name === '__metadata__' ||
        Object.prototype.hasOwnProperty.call(Object.prototype, name)) {
      throw new Error(`${label} must be a non-empty safetensors name.`);
    }
    if (claimedNames.has(name)) throw new Error(`PTQ output tensor name '${name}' is duplicated.`);
    claimedNames.add(name);
  };

  for (const raw of requests) {
    const request = typeof raw === 'string' ? { name: raw } : raw;
    if (!request || typeof request !== 'object' || Array.isArray(request)) {
      throw new Error('PTQ weight request must be a name or object.');
    }
    const source = weightDescriptors.get(request.name);
    const sourceData = source ? snapshot.copyWeightData(source.name) : null;
    if (!source || source.dtype !== 'float32' || !(sourceData instanceof Float32Array)) {
      throw new Error(`PTQ source '${request.name}' must be an F32 weight with CPU storage.`);
    }
    if (consumedSources.has(source.name)) throw new Error(`PTQ source '${source.name}' is requested more than once.`);
    consumedSources.add(source.name);
    const outputName = request.outputName ?? source.name;
    const scaleName = request.scaleName ?? `${outputName}.scale`;
    const zeroPointName = request.zeroPointName ?? `${outputName}.zero_point`;
    claim(outputName, 'PTQ outputName');
    claim(scaleName, 'PTQ scaleName');
    claim(zeroPointName, 'PTQ zeroPointName');
    const existingOutput = snapshot.graph.tensors[outputName];
    if (existingOutput && outputName !== source.name) {
      throw new Error(`PTQ output '${outputName}' collides with graph tensor '${outputName}'.`);
    }
    if (snapshot.graph.tensors[scaleName]) {
      throw new Error(`PTQ scale '${scaleName}' collides with graph tensor '${scaleName}'.`);
    }
    if (snapshot.graph.tensors[zeroPointName]) {
      throw new Error(`PTQ zero point '${zeroPointName}' collides with graph tensor '${zeroPointName}'.`);
    }
    const packed = packPTQWeight(sourceData, source.shape, {
      axis: request.axis ?? 0,
      name: outputName,
    });
    const record: PTQMaterializedRecord = {
      sourceName: source.name, outputName, scaleName, zeroPointName, packed,
    };
    replacements.set(outputName, {
      name: outputName, dtype: 'int8', shape: [...packed.shape], buffer: packed.data,
    });
    replacements.set(scaleName, {
      name: scaleName, dtype: 'float32', shape: [packed.scales.length], buffer: packed.scales,
    });
    replacements.set(zeroPointName, {
      name: zeroPointName,
      dtype: 'int8',
      shape: [packed.scales.length],
      buffer: new Int8Array(packed.scales.length),
    });
    quantizationTensors[outputName] = Object.freeze({
      scheme: 'per_axis',
      axis: packed.quantization.axis,
      scale_tensor: scaleName,
      zero_point_tensor: zeroPointName,
    });

    if (request.bias != null) {
      if (request.inputScale == null) {
        throw new Error(`PTQ request '${source.name}' with a bias requires inputScale.`);
      }
      const bias = weightDescriptors.get(request.bias);
      const biasDataSource = bias ? snapshot.copyWeightData(bias.name) : null;
      if (!bias || bias.dtype !== 'float32' || !(biasDataSource instanceof Float32Array) ||
          biasDataSource.length !== packed.scales.length) {
        throw new Error(`PTQ bias '${request.bias}' must be an F32 weight matching the output channels.`);
      }
      if (consumedSources.has(bias.name)) {
        throw new Error(`PTQ bias source '${bias.name}' is requested more than once.`);
      }
      const biasOutputName = request.biasOutputName ?? bias.name;
      claim(biasOutputName, 'PTQ biasOutputName');
      const existingBiasOutput = snapshot.graph.tensors[biasOutputName];
      if (existingBiasOutput && biasOutputName !== bias.name) {
        throw new Error(`PTQ bias output '${biasOutputName}' collides with graph tensor '${biasOutputName}'.`);
      }
      consumedSources.add(bias.name);
      const biasData = packPTQBias(biasDataSource, request.inputScale, packed.scales);
      replacements.set(biasOutputName, {
        name: biasOutputName, dtype: 'int32', shape: [...bias.shape], buffer: biasData,
      });
      record.bias = { sourceName: bias.name, outputName: biasOutputName, data: biasData };
    }
    materialized.push(Object.freeze(record));
  }

  for (const name of claimedNames) {
    const existing = snapshot.graph.tensors[name];
    const replacingOwnSource = consumedSources.has(name);
    if (existing?.kind === 'weight' && !replacingOwnSource) {
      throw new Error(`PTQ output '${name}' collides with unselected weight '${name}'.`);
    }
  }

  const metadata = {
    ...(options.metadata || {}),
    format: VOLVOX_PTQ_FORMAT,
    logical_fingerprint: snapshot.definitionFingerprint,
    profile_coverage: JSON.stringify(options.coverage),
  };
  const file = SafetensorsFile.empty({ metadata });
  if (options.includeUnselected !== false) {
    for (const descriptor of snapshot.weightDescriptors) {
      if (consumedSources.has(descriptor.name)) continue;
      if (claimedNames.has(descriptor.name)) {
        throw new Error(`PTQ output collides with retained weight '${descriptor.name}'.`);
      }
      const data = snapshot.copyWeightData(descriptor.name);
      file.addTensor(descriptor.name, safetensorsDType(descriptor.dtype), descriptor.shape,
        bytesOf(data, `Weight '${descriptor.name}'`));
    }
  }
  for (const entry of replacements.values()) {
    file.addTensor(entry.name, safetensorsDType(entry.dtype), entry.shape,
      bytesOf(entry.buffer, `PTQ tensor '${entry.name}'`));
  }
  return Object.freeze({
    format: VOLVOX_PTQ_FORMAT,
    logicalGraph: checkpointGraphDocument(snapshot.graph),
    logicalFingerprint: snapshot.definitionFingerprint,
    coverage: options.coverage,
    weights: file,
    quantization: Object.freeze({
      format: 'volvox-affine-safetensors/v1' as const,
      tensors: Object.freeze(quantizationTensors),
    }),
    materialized: Object.freeze(materialized),
  });
}
