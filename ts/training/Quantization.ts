import { SafetensorsFile } from '../core/Safetensors.js';
import type { Graph } from '../core/Graph.js';
import type { RuntimeDType, RuntimeTypedArray, TensorQuantization } from '../types.js';

export const VOLVOX_PTQ_FORMAT = 'volvox.ptq.v1';

const F32_MIN_NORMAL = 1.1754943508222875e-38;
const I32_MIN = -2147483648;
const I32_MAX = 2147483647;

export type PTQDType = 'int8' | 'uint8';
export type PTQScheme = 'symmetric' | 'asymmetric';
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
  bias?: string;
  biasOutputName?: string;
  inputScale?: number;
}

export interface PTQMaterializeOptions {
  metadata?: Record<string, string>;
  includeUnselected?: boolean;
}

export interface PTQExecutor {
  execute(inputs: Record<string, unknown>, options?: Record<string, unknown>): Promise<any> | any;
  graph?: Graph;
  _graph?: Graph;
  capabilities?: { outputLocation?: 'host' | 'device' };
  gpuBuffers?: Map<unknown, unknown>;
}

export interface PTQReadbackContext {
  executor: PTQExecutor;
  graph: Graph;
  result: any;
  sample: Record<string, unknown>;
  sampleIndex: number;
}

export type PTQReadback = (
  name: string,
  context: PTQReadbackContext,
) => Float32Array | Promise<Float32Array>;

export interface PTQCalibrationOptions {
  graph?: Graph;
  tensorNames?: string[];
  readback?: PTQReadback | Record<string, PTQReadback>;
  calibrator?: PTQCalibrator;
  executeOptions?: Record<string, unknown>;
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

/** Named observer set for collecting CPU-visible graph tensors after forwards. */
export class PTQCalibrator {
  observers: Map<string, PTQObserver>;

  constructor() {
    this.observers = new Map();
  }

  reset(name: string | null = null) {
    if (name == null) this.observers.clear();
    else this.observers.delete(name);
    return this;
  }

  observe(name: string, values: NumericArray) {
    if (typeof name !== 'string' || !name) throw new Error('PTQ observation name must be non-empty.');
    let observer = this.observers.get(name);
    if (!observer) {
      observer = new PTQObserver();
      this.observers.set(name, observer);
    }
    observer.observe(values);
    return this;
  }

  observeGraph(graph: Graph, names: string[] | null = null) {
    if (!graph?.tensors || !(graph.tensors instanceof Map)) {
      throw new Error('PTQ observeGraph expects a VolvoxAI Graph.');
    }
    const selected = names == null
      ? [...graph.tensors.values()].filter((tensor) => !tensor.isWeight && tensor.dtype === 'float32' && tensor.buffer)
      : names.map((name) => {
        const tensor = graph.tensors.get(name);
        if (!tensor) throw new Error(`PTQ tensor '${name}' was not found.`);
        return tensor;
      });
    const prepared = selected.map((tensor) => {
      if (tensor.dtype !== 'float32' || !(tensor.buffer instanceof Float32Array)) {
        throw new Error(`PTQ tensor '${tensor.name}' requires CPU-visible F32 storage.`);
      }
      finiteRange(tensor.buffer, `PTQ tensor '${tensor.name}'`);
      return tensor;
    });
    for (const tensor of prepared) this.observe(tensor.name, tensor.buffer as Float32Array);
    return this;
  }

  parameters(options: PTQParameterOptions = {}) {
    return Object.freeze(Object.fromEntries(
      [...this.observers].map(([name, observer]) => [name, derivePTQParameters(observer, options)]),
    ));
  }
}

/**
 * Run named calibration samples through an allocated backend and observe F32
 * tensors after every forward. Device-output backends require an explicit
 * `readback(name, context)` callback so stale host mirrors are never sampled.
 */
export async function calibratePTQ(
  executor: PTQExecutor,
  samples: Iterable<Record<string, unknown>> | AsyncIterable<Record<string, unknown>>,
  options: PTQCalibrationOptions = {},
): Promise<PTQCalibrator> {
  if (!executor || typeof executor.execute !== 'function') {
    throw new Error('calibratePTQ expects an allocated backend executor.');
  }
  if (!samples || (typeof samples[Symbol.iterator] !== 'function' &&
      typeof samples[Symbol.asyncIterator] !== 'function')) {
    throw new Error('calibratePTQ samples must be an iterable or async iterable of named inputs.');
  }
  const graph = options.graph ?? executor.graph ?? executor._graph;
  if (!graph?.tensors || !(graph.tensors instanceof Map)) {
    throw new Error('calibratePTQ requires the allocated graph in options.graph or executor.graph.');
  }
  if (options.tensorNames != null && !Array.isArray(options.tensorNames)) {
    throw new Error('calibratePTQ options.tensorNames must be an array.');
  }
  const tensorNames = options.tensorNames == null
    ? [...graph.tensors.values()]
      .filter((tensor) => !tensor.isWeight && tensor.dtype === 'float32')
      .map((tensor) => tensor.name)
    : [...options.tensorNames];
  if (tensorNames.length === 0 || tensorNames.some((name) => typeof name !== 'string' || !name)) {
    throw new Error('calibratePTQ requires at least one valid tensor name.');
  }
  for (const name of tensorNames) {
    const tensor = graph.tensors.get(name);
    if (!tensor || tensor.dtype !== 'float32') {
      throw new Error(`PTQ tensor '${name}' must be a graph F32 tensor.`);
    }
  }
  const deviceOutput = executor.capabilities?.outputLocation === 'device' ||
    (!executor.capabilities && executor.gpuBuffers instanceof Map);
  const readback = options.readback;
  if (deviceOutput && readback == null) {
    throw new Error('calibratePTQ requires a readback mapping for a device-output backend.');
  }
  if (readback != null && typeof readback !== 'function' &&
      (typeof readback !== 'object' || Array.isArray(readback))) {
    throw new Error('calibratePTQ readback must be a function or name-to-function mapping.');
  }
  const calibrator = options.calibrator ?? new PTQCalibrator();
  if (!(calibrator instanceof PTQCalibrator)) {
    throw new Error('calibratePTQ options.calibrator must be a PTQCalibrator.');
  }
  let sampleIndex = 0;
  for await (const sample of samples) {
    if (!sample || typeof sample !== 'object' || Array.isArray(sample)) {
      throw new Error(`PTQ calibration sample ${sampleIndex} must be a named input object.`);
    }
    const result = await executor.execute(sample, options.executeOptions || {});
    const observations: Array<[string, Float32Array]> = [];
    for (const name of tensorNames) {
      let values;
      if (readback != null) {
        const reader = typeof readback === 'function' ? readback : readback[name];
        if (typeof reader !== 'function') {
          throw new Error(`PTQ readback mapping is missing tensor '${name}'.`);
        }
        values = await reader(name, { executor, graph, result, sample, sampleIndex });
      } else {
        values = result?.[name] ?? graph.tensors.get(name)?.buffer;
      }
      if (!(values instanceof Float32Array)) {
        throw new Error(`PTQ tensor '${name}' has no CPU-visible F32 values after sample ${sampleIndex}.`);
      }
      finiteRange(values, `PTQ tensor '${name}' at sample ${sampleIndex}`);
      observations.push([name, values]);
    }
    for (const [name, values] of observations) calibrator.observe(name, values);
    sampleIndex++;
  }
  if (sampleIndex === 0) throw new Error('calibratePTQ received no calibration samples.');
  return calibrator;
}

type PackedPTQWeight = ReturnType<typeof packPTQWeight>;

interface PTQMaterializedRecord {
  sourceName: string;
  outputName: string;
  scaleName: string;
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
 * Each request is `{ name, axis?, outputName?, scaleName?, bias?,
 * biasOutputName?, inputScale? }`. Unselected weights are copied by default.
 * The returned `weightsQuantization` object can be merged directly into a
 * Volvox blueprint's `weights_quantization` field.
 */
export function materializePTQWeights(
  graph: Graph,
  requests: Array<string | PTQWeightRequest>,
  options: PTQMaterializeOptions = {},
) {
  if (!graph?.tensors || !(graph.tensors instanceof Map)) {
    throw new Error('materializePTQWeights expects a VolvoxAI Graph.');
  }
  if (!Array.isArray(requests) || requests.length === 0) {
    throw new Error('materializePTQWeights requires at least one weight request.');
  }
  const materialized: Readonly<PTQMaterializedRecord>[] = [];
  const replacements = new Map<string, PTQReplacement>();
  const consumedSources = new Set<string>();
  const weightsQuantization: Record<string, TensorQuantization> = {};
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
    const source = graph.tensors.get(request.name);
    if (!source?.isWeight || source.dtype !== 'float32' || !(source.buffer instanceof Float32Array)) {
      throw new Error(`PTQ source '${request.name}' must be an F32 weight with CPU storage.`);
    }
    if (consumedSources.has(source.name)) throw new Error(`PTQ source '${source.name}' is requested more than once.`);
    consumedSources.add(source.name);
    const outputName = request.outputName ?? source.name;
    const scaleName = request.scaleName ?? `${outputName}.scale`;
    claim(outputName, 'PTQ outputName');
    claim(scaleName, 'PTQ scaleName');
    const existingOutput = graph.tensors.get(outputName);
    if (existingOutput && existingOutput !== source) {
      throw new Error(`PTQ output '${outputName}' collides with graph tensor '${outputName}'.`);
    }
    if (graph.tensors.has(scaleName)) {
      throw new Error(`PTQ scale '${scaleName}' collides with graph tensor '${scaleName}'.`);
    }
    const packed = packPTQWeight(source.buffer, source.shape, {
      axis: request.axis ?? 0,
      name: outputName,
    });
    const record: PTQMaterializedRecord = { sourceName: source.name, outputName, scaleName, packed };
    replacements.set(outputName, {
      name: outputName, dtype: 'int8', shape: [...packed.shape], buffer: packed.data,
    });
    replacements.set(scaleName, {
      name: scaleName, dtype: 'float32', shape: [packed.scales.length], buffer: packed.scales,
    });
    weightsQuantization[outputName] = packed.quantization;

    if (request.bias != null) {
      if (request.inputScale == null) {
        throw new Error(`PTQ request '${source.name}' with a bias requires inputScale.`);
      }
      const bias = graph.tensors.get(request.bias);
      if (!bias?.isWeight || bias.dtype !== 'float32' || !(bias.buffer instanceof Float32Array) ||
          bias.buffer.length !== packed.scales.length) {
        throw new Error(`PTQ bias '${request.bias}' must be an F32 weight matching the output channels.`);
      }
      if (consumedSources.has(bias.name)) {
        throw new Error(`PTQ bias source '${bias.name}' is requested more than once.`);
      }
      const biasOutputName = request.biasOutputName ?? bias.name;
      claim(biasOutputName, 'PTQ biasOutputName');
      const existingBiasOutput = graph.tensors.get(biasOutputName);
      if (existingBiasOutput && existingBiasOutput !== bias) {
        throw new Error(`PTQ bias output '${biasOutputName}' collides with graph tensor '${biasOutputName}'.`);
      }
      consumedSources.add(bias.name);
      const biasData = packPTQBias(bias.buffer, request.inputScale, packed.scales);
      replacements.set(biasOutputName, {
        name: biasOutputName, dtype: 'int32', shape: [...bias.shape], buffer: biasData,
      });
      record.bias = { sourceName: bias.name, outputName: biasOutputName, data: biasData };
    }
    materialized.push(Object.freeze(record));
  }

  for (const name of claimedNames) {
    const existing = graph.tensors.get(name);
    const replacingOwnSource = consumedSources.has(name);
    if (existing?.isWeight && !replacingOwnSource) {
      throw new Error(`PTQ output '${name}' collides with unselected weight '${name}'.`);
    }
  }

  const metadata = {
    ...(options.metadata || {}),
    format: VOLVOX_PTQ_FORMAT,
    weights_quantization: JSON.stringify(weightsQuantization),
  };
  const file = SafetensorsFile.empty({ metadata });
  if (options.includeUnselected !== false) {
    for (const tensor of graph.tensors.values()) {
      if (!tensor.isWeight || !tensor.buffer || consumedSources.has(tensor.name)) continue;
      if (claimedNames.has(tensor.name)) {
        throw new Error(`PTQ output collides with retained weight '${tensor.name}'.`);
      }
      file.addTensor(tensor.name, safetensorsDType(tensor.dtype), tensor.shape,
        bytesOf(tensor.buffer, `Weight '${tensor.name}'`));
    }
  }
  for (const entry of replacements.values()) {
    file.addTensor(entry.name, safetensorsDType(entry.dtype), entry.shape,
      bytesOf(entry.buffer, `PTQ tensor '${entry.name}'`));
  }
  return Object.freeze({
    format: VOLVOX_PTQ_FORMAT,
    weights: file,
    weightsQuantization: Object.freeze(weightsQuantization),
    materialized: Object.freeze(materialized),
  });
}
