import type { WasmEngine } from '../backends/WasmEngine.js';

export const WASM_PTQ_ABI_VERSION = 1;

export const WASM_PTQ_CAPABILITIES = Object.freeze({
  observer: 1 << 0,
  parameters: 1 << 1,
  quantize: 1 << 2,
  weightI8: 1 << 3,
  biasI32: 1 << 4,
});

const REQUIRED_CAPABILITIES = Object.values(WASM_PTQ_CAPABILITIES)
  .reduce((mask, capability) => mask | capability, 0);
const MAX_ALLOCATION_BYTES = 0x7ffffff0;
const MAX_U32 = 0xffffffff;
const OBSERVER_BYTES = 16;
const PARAMETERS_BYTES = 32;
const SATURATION_BYTES = 8;

const DTYPE_CODE = Object.freeze({
  int8: 1,
  uint8: 2,
});

const SCHEME_CODE = Object.freeze({
  symmetric: 0,
  asymmetric: 1,
});

export type WasmPTQDType = keyof typeof DTYPE_CODE;
export type WasmPTQScheme = keyof typeof SCHEME_CODE;

export interface WasmPTQRange {
  minimum: number;
  maximum: number;
  sampleCount: number;
}

export interface WasmPTQParameterOptions {
  dtype?: WasmPTQDType;
  scheme?: WasmPTQScheme;
}

export interface WasmPTQParameters {
  dtype: WasmPTQDType;
  scheme: WasmPTQScheme;
  scale: number;
  zero_point: number;
  observed_min: number;
  observed_max: number;
  sample_count: number;
}

export interface WasmPTQQuantized {
  data: Int8Array | Uint8Array;
  saturationCount: number;
}

export interface WasmPTQWeightOptions {
  axis?: number;
  name?: string | null;
}

export interface WasmPTQPackedWeight {
  name?: string;
  shape: readonly number[];
  dtype: 'int8';
  data: Int8Array;
  scales: Float32Array;
  saturationCount: number;
  quantization: Readonly<{
    scheme: 'per_axis';
    axis: number;
    scales: readonly number[];
    zero_points: readonly number[];
  }>;
}

type WasmArgument = number | bigint;
type WasmFunction = (...args: WasmArgument[]) => number | bigint | void;

interface WasmPTQApi {
  memory: WebAssembly.Memory;
  alloc_bytes: WasmFunction;
  reset_heap: WasmFunction;
  [name: string]: unknown;
}

type WasmPTQTypedArray =
  Float32Array | Int32Array | Int8Array | Uint8Array | Uint32Array;
type WasmPTQTypedArrayConstructor =
  Float32ArrayConstructor | Int32ArrayConstructor | Int8ArrayConstructor |
  Uint8ArrayConstructor | Uint32ArrayConstructor;

interface ArenaEntry {
  name: string;
  value: ArrayBufferView | null;
  bytes: number;
  ArrayType: WasmPTQTypedArrayConstructor;
  read: boolean;
}

interface ObserverState {
  minimum: number;
  maximum: number;
  sampleCount: number;
}

function requireFunction(api: WasmPTQApi, name: string): WasmFunction {
  const candidate = api[name];
  if (typeof candidate !== 'function') {
    throw new Error(`The full WASM module is missing PTQ export '${name}'.`);
  }
  return candidate as WasmFunction;
}

function requireFloat32Array(value: unknown, label: string): Float32Array {
  if (!(value instanceof Float32Array)) {
    throw new Error(`${label} must be a Float32Array.`);
  }
  return value;
}

function requireFiniteValues(values: Float32Array, label: string): void {
  for (let index = 0; index < values.length; index++) {
    if (!Number.isFinite(values[index])) {
      throw new Error(`${label} contains a non-finite value at index ${index}.`);
    }
  }
}

function requirePositiveF32(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be finite and positive.`);
  }
  const rounded = Math.fround(value);
  if (!Number.isFinite(rounded) || rounded <= 0) {
    throw new Error(`${label} must be representable as positive F32.`);
  }
  return rounded;
}

function safeU64(view: DataView, offset: number, label: string): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${label} exceeds the JavaScript safe integer range.`);
  }
  return Number(value);
}

function writeSafeU64(view: DataView, offset: number, value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative safe integer.`);
  }
  view.setBigUint64(offset, BigInt(value), true);
}

function saturationCount(words: Uint32Array): number {
  const count = words[0] + words[1] * 0x100000000;
  if (!Number.isSafeInteger(count)) {
    throw new Error('WASM PTQ saturation count exceeds the JavaScript safe integer range.');
  }
  return count;
}

function dtypeFromCode(code: number): WasmPTQDType {
  if (code === DTYPE_CODE.int8) return 'int8';
  if (code === DTYPE_CODE.uint8) return 'uint8';
  throw new Error(`The WASM PTQ module returned unsupported dtype code ${code}.`);
}

function schemeFromCode(code: number): WasmPTQScheme {
  if (code === SCHEME_CODE.symmetric) return 'symmetric';
  if (code === SCHEME_CODE.asymmetric) return 'asymmetric';
  throw new Error(`The WASM PTQ module returned unsupported scheme code ${code}.`);
}

function encodeObserver(state: ObserverState): Uint8Array {
  const bytes = new Uint8Array(OBSERVER_BYTES);
  const view = new DataView(bytes.buffer);
  view.setFloat32(0, state.minimum, true);
  view.setFloat32(4, state.maximum, true);
  writeSafeU64(view, 8, state.sampleCount, 'WASM PTQ observer sample count');
  return bytes;
}

function decodeObserver(bytes: Uint8Array): ObserverState {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    minimum: view.getFloat32(0, true),
    maximum: view.getFloat32(4, true),
    sampleCount: safeU64(view, 8, 'WASM PTQ observer sample count'),
  };
}

function encodeParameters(parameters: Readonly<WasmPTQParameters>): Uint8Array {
  if (!Object.prototype.hasOwnProperty.call(DTYPE_CODE, parameters.dtype)) {
    throw new Error(`Unsupported WASM PTQ dtype '${parameters.dtype}'.`);
  }
  if (!Object.prototype.hasOwnProperty.call(SCHEME_CODE, parameters.scheme)) {
    throw new Error(`Unsupported WASM PTQ scheme '${parameters.scheme}'.`);
  }
  const scale = requirePositiveF32(parameters.scale, 'WASM PTQ scale');
  const [qmin, qmax] = parameters.dtype === 'int8' ? [-128, 127] : [0, 255];
  if (!Number.isInteger(parameters.zero_point) ||
      parameters.zero_point < qmin || parameters.zero_point > qmax) {
    throw new Error(`WASM PTQ zero point must be an integer in [${qmin}, ${qmax}].`);
  }
  if (typeof parameters.observed_min !== 'number' || !Number.isFinite(parameters.observed_min) ||
      typeof parameters.observed_max !== 'number' || !Number.isFinite(parameters.observed_max) ||
      parameters.observed_min > parameters.observed_max) {
    throw new Error('WASM PTQ parameters require finite ordered observed bounds.');
  }
  const bytes = new Uint8Array(PARAMETERS_BYTES);
  const view = new DataView(bytes.buffer);
  view.setInt32(0, DTYPE_CODE[parameters.dtype], true);
  view.setInt32(4, SCHEME_CODE[parameters.scheme], true);
  view.setFloat32(8, scale, true);
  view.setInt32(12, parameters.zero_point, true);
  view.setFloat32(16, parameters.observed_min, true);
  view.setFloat32(20, parameters.observed_max, true);
  writeSafeU64(view, 24, parameters.sample_count, 'WASM PTQ parameter sample count');
  return bytes;
}

function decodeParameters(bytes: Uint8Array): Readonly<WasmPTQParameters> {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Object.freeze({
    dtype: dtypeFromCode(view.getInt32(0, true)),
    scheme: schemeFromCode(view.getInt32(4, true)),
    scale: view.getFloat32(8, true),
    zero_point: view.getInt32(12, true),
    observed_min: view.getFloat32(16, true),
    observed_max: view.getFloat32(20, true),
    sample_count: safeU64(view, 24, 'WASM PTQ parameter sample count'),
  });
}

function normalizeRange(range: WasmPTQObserver | WasmPTQRange): ObserverState {
  const value = range instanceof WasmPTQObserver ? range.snapshot() : range;
  if (!value || typeof value !== 'object') {
    throw new Error('WASM PTQ range must be an observer or range object.');
  }
  const minimum = Math.fround(value.minimum);
  const maximum = Math.fround(value.maximum);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum ||
      !Number.isSafeInteger(value.sampleCount) || value.sampleCount <= 0) {
    throw new Error('WASM PTQ range requires finite ordered F32 bounds and a positive sample count.');
  }
  return { minimum, maximum, sampleCount: value.sampleCount };
}

function shapeInfo(shape: readonly number[], axis: number): {
  shape: Int32Array;
  dimensions: readonly number[];
  elements: number;
  axis: number;
  channels: number;
} {
  if (!Array.isArray(shape) || shape.length === 0 || shape.length > 8) {
    throw new Error('WASM PTQ weight shape must have rank 1 through 8.');
  }
  let elements = 1;
  for (let index = 0; index < shape.length; index++) {
    const dimension = shape[index];
    if (!Number.isSafeInteger(dimension) || dimension <= 0 || dimension > 0x7fffffff) {
      throw new Error(`WASM PTQ weight shape has invalid dimension ${index}.`);
    }
    elements *= dimension;
    if (!Number.isSafeInteger(elements) || elements > MAX_ALLOCATION_BYTES) {
      throw new Error('WASM PTQ weight is too large for the 32-bit scratch module.');
    }
  }
  if (!Number.isInteger(axis)) throw new Error('WASM PTQ weight axis must be an integer.');
  const normalizedAxis = axis < 0 ? axis + shape.length : axis;
  if (normalizedAxis < 0 || normalizedAxis >= shape.length) {
    throw new Error(`WASM PTQ weight axis ${axis} is outside rank ${shape.length}.`);
  }
  const dimensions = Object.freeze([...shape]);
  return {
    shape: Int32Array.from(shape),
    dimensions,
    elements,
    axis: normalizedAxis,
    channels: shape[normalizedAxis],
  };
}

/** A reusable C-backed min/max observer tied to one isolated PTQ instance. */
export class WasmPTQObserver {
  private readonly ptq: WasmPTQ;
  private state: ObserverState;

  constructor(ptq: WasmPTQ) {
    this.ptq = ptq;
    this.state = ptq._resetObserver();
  }

  reset(): this {
    this.state = this.ptq._resetObserver();
    return this;
  }

  observe(values: Float32Array): this {
    this.state = this.ptq._observe(this.state, values);
    return this;
  }

  snapshot(): Readonly<WasmPTQRange> {
    if (this.state.sampleCount <= 0) {
      throw new Error('WASM PTQ observer has no observations.');
    }
    return Object.freeze({ ...this.state });
  }

  parameters(options: WasmPTQParameterOptions = {}): Readonly<WasmPTQParameters> {
    return this.ptq.deriveParameters(this, options);
  }
}

/**
 * Generic PTQ math executed by the existing C implementation in an isolated
 * full-WASM instance. Arena resets never touch the live inference engine heap.
 */
export class WasmPTQ {
  readonly abiVersion: number;
  readonly capabilities: number;
  private scratchEngine: WasmEngine | null;
  private api: WasmPTQApi | null;
  private memory: WebAssembly.Memory | null;

  private constructor(
    scratchEngine: WasmEngine,
    api: WasmPTQApi,
    abiVersion: number,
    capabilities: number,
  ) {
    this.scratchEngine = scratchEngine;
    this.api = api;
    this.memory = api.memory;
    this.abiVersion = abiVersion;
    this.capabilities = capabilities;
  }

  static async create(sourceEngine: WasmEngine): Promise<WasmPTQ> {
    if (!sourceEngine || typeof sourceEngine.fork !== 'function') {
      throw new Error('WasmPTQ.create requires an initialized WasmEngine.');
    }
    const scratchEngine = await sourceEngine.fork({ relaxedSimd: false });
    const api = scratchEngine.api as unknown as WasmPTQApi;
    if (!(api.memory instanceof WebAssembly.Memory) ||
        typeof api.alloc_bytes !== 'function' || typeof api.reset_heap !== 'function') {
      throw new Error('The full WASM module is missing memory or allocator exports for PTQ.');
    }
    for (const name of [
      'volvoxai_ptq_abi_version',
      'volvoxai_ptq_capabilities',
      'volvoxai_ptq_observer_reset',
      'volvoxai_ptq_observer_observe_f32',
      'volvoxai_ptq_calculate_params',
      'volvoxai_ptq_quantize_f32',
      'volvoxai_ptq_pack_weight_i8',
      'volvoxai_ptq_pack_bias_i32',
    ]) requireFunction(api, name);

    const abiVersion = Number(requireFunction(api, 'volvoxai_ptq_abi_version')());
    if (abiVersion !== WASM_PTQ_ABI_VERSION) {
      throw new Error(`Unsupported WASM PTQ ABI ${abiVersion}; expected ${WASM_PTQ_ABI_VERSION}.`);
    }
    const capabilities = Number(requireFunction(api, 'volvoxai_ptq_capabilities')()) >>> 0;
    if ((capabilities & REQUIRED_CAPABILITIES) !== REQUIRED_CAPABILITIES) {
      const missing = REQUIRED_CAPABILITIES & ~capabilities;
      throw new Error(`The full WASM module is missing PTQ capability bits 0x${missing.toString(16)}.`);
    }
    return new WasmPTQ(scratchEngine, api, abiVersion, capabilities);
  }

  createObserver(): WasmPTQObserver {
    this._active();
    return new WasmPTQObserver(this);
  }

  deriveParameters(
    range: WasmPTQObserver | WasmPTQRange,
    { dtype = 'int8', scheme = 'symmetric' }: WasmPTQParameterOptions = {},
  ): Readonly<WasmPTQParameters> {
    const { api } = this._active();
    if (!Object.prototype.hasOwnProperty.call(DTYPE_CODE, dtype)) {
      throw new Error(`Unsupported WASM PTQ dtype '${dtype}'.`);
    }
    if (!Object.prototype.hasOwnProperty.call(SCHEME_CODE, scheme)) {
      throw new Error(`Unsupported WASM PTQ scheme '${scheme}'.`);
    }
    const observer = encodeObserver(normalizeRange(range));
    const parameters = new Uint8Array(PARAMETERS_BYTES);
    const calculate = requireFunction(api, 'volvoxai_ptq_calculate_params');
    const { value, outputs } = this._runArena([
      this._entry('observer', observer, Uint8Array),
      this._entry('parameters', parameters, Uint8Array, true),
    ], (pointers) => calculate(
      pointers.observer,
      DTYPE_CODE[dtype],
      SCHEME_CODE[scheme],
      pointers.parameters,
    ));
    this._expectSuccess(value, 'parameter calculation');
    return decodeParameters(outputs.parameters as Uint8Array);
  }

  parameters(
    range: WasmPTQObserver | WasmPTQRange,
    options: WasmPTQParameterOptions = {},
  ): Readonly<WasmPTQParameters> {
    return this.deriveParameters(range, options);
  }

  quantize(
    values: Float32Array,
    parameters: Readonly<WasmPTQParameters>,
  ): WasmPTQQuantized {
    const { api } = this._active();
    const source = requireFloat32Array(values, 'WASM PTQ quantization input');
    requireFiniteValues(source, 'WASM PTQ quantization input');
    const encoded = encodeParameters(parameters);
    if (source.length === 0) {
      return {
        data: parameters.dtype === 'int8' ? new Int8Array() : new Uint8Array(),
        saturationCount: 0,
      };
    }
    const ArrayType = parameters.dtype === 'int8' ? Int8Array : Uint8Array;
    const quantize = requireFunction(api, 'volvoxai_ptq_quantize_f32');
    const { value, outputs } = this._runArena([
      this._entry('source', source, Float32Array),
      this._scratch('parameters', PARAMETERS_BYTES, Uint8Array, encoded),
      this._scratch('output', source.length, ArrayType, null, true),
      this._scratch('saturation', SATURATION_BYTES, Uint32Array, null, true),
    ], (pointers) => quantize(
      pointers.source,
      BigInt(source.length),
      pointers.parameters,
      pointers.output,
      pointers.saturation,
    ));
    this._expectSuccess(value, 'value quantization');
    return {
      data: outputs.output as Int8Array | Uint8Array,
      saturationCount: saturationCount(outputs.saturation as Uint32Array),
    };
  }

  packWeight(
    values: Float32Array,
    shape: readonly number[],
    { axis = 0, name = null }: WasmPTQWeightOptions = {},
  ): WasmPTQPackedWeight {
    const { api } = this._active();
    const source = requireFloat32Array(values, 'WASM PTQ weight');
    requireFiniteValues(source, 'WASM PTQ weight');
    const info = shapeInfo(shape, axis);
    if (source.length !== info.elements) {
      throw new Error(`WASM PTQ weight has ${source.length} values, expected ${info.elements}.`);
    }
    if (name != null && (typeof name !== 'string' || name.length === 0)) {
      throw new Error('WASM PTQ weight name must be a non-empty string.');
    }
    const pack = requireFunction(api, 'volvoxai_ptq_pack_weight_i8');
    const { value, outputs } = this._runArena([
      this._entry('source', source, Float32Array),
      this._entry('shape', info.shape, Int32Array),
      this._scratch('output', info.elements, Int8Array, null, true),
      this._scratch(
        'scales',
        info.channels * Float32Array.BYTES_PER_ELEMENT,
        Float32Array,
        null,
        true,
      ),
      this._scratch('saturation', SATURATION_BYTES, Uint32Array, null, true),
    ], (pointers) => pack(
      pointers.source,
      pointers.shape,
      info.dimensions.length,
      info.axis,
      pointers.output,
      pointers.scales,
      info.channels,
      pointers.saturation,
    ));
    this._expectSuccess(value, 'I8 weight packing');
    const data = outputs.output as Int8Array;
    const scales = outputs.scales as Float32Array;
    return Object.freeze({
      ...(name == null ? {} : { name }),
      shape: info.dimensions,
      dtype: 'int8' as const,
      data,
      scales,
      saturationCount: saturationCount(outputs.saturation as Uint32Array),
      quantization: Object.freeze({
        scheme: 'per_axis' as const,
        axis: info.axis,
        scales: Object.freeze([...scales]),
        zero_points: Object.freeze(new Array(info.channels).fill(0)),
      }),
    });
  }

  packBias(
    values: Float32Array,
    inputScale: number,
    weightScales: Float32Array,
  ): Int32Array {
    const { api } = this._active();
    const source = requireFloat32Array(values, 'WASM PTQ bias');
    const scales = requireFloat32Array(weightScales, 'WASM PTQ weight scales');
    requireFiniteValues(source, 'WASM PTQ bias');
    requireFiniteValues(scales, 'WASM PTQ weight scales');
    const activationScale = requirePositiveF32(inputScale, 'WASM PTQ input scale');
    if (source.length !== scales.length) {
      throw new Error('WASM PTQ bias and weight scale counts must match.');
    }
    if (source.length === 0) return new Int32Array();
    const pack = requireFunction(api, 'volvoxai_ptq_pack_bias_i32');
    const { value, outputs } = this._runArena([
      this._entry('source', source, Float32Array),
      this._entry('scales', scales, Float32Array),
      this._scratch(
        'output',
        source.length * Int32Array.BYTES_PER_ELEMENT,
        Int32Array,
        null,
        true,
      ),
    ], (pointers) => pack(
      pointers.source,
      source.length,
      activationScale,
      pointers.scales,
      scales.length,
      pointers.output,
    ));
    this._expectSuccess(value, 'I32 bias packing');
    return outputs.output as Int32Array;
  }

  _resetObserver(): ObserverState {
    const { api } = this._active();
    const reset = requireFunction(api, 'volvoxai_ptq_observer_reset');
    const { outputs } = this._runArena([
      this._scratch('observer', OBSERVER_BYTES, Uint8Array, null, true),
    ], (pointers) => reset(pointers.observer));
    return decodeObserver(outputs.observer as Uint8Array);
  }

  _observe(state: ObserverState, values: Float32Array): ObserverState {
    const { api } = this._active();
    const source = requireFloat32Array(values, 'WASM PTQ observation');
    requireFiniteValues(source, 'WASM PTQ observation');
    if (source.length === 0) return { ...state };
    if (state.sampleCount > Number.MAX_SAFE_INTEGER - source.length) {
      throw new Error('WASM PTQ observer sample count exceeds the JavaScript safe integer range.');
    }
    const observe = requireFunction(api, 'volvoxai_ptq_observer_observe_f32');
    const { value, outputs } = this._runArena([
      this._entry('observer', encodeObserver(state), Uint8Array, true),
      this._entry('source', source, Float32Array),
    ], (pointers) => observe(pointers.observer, pointers.source, BigInt(source.length)));
    this._expectSuccess(value, 'observation');
    return decodeObserver(outputs.observer as Uint8Array);
  }

  /**
   * Release references to the isolated WASM instance and its grow-only linear
   * memory. WebAssembly has no explicit memory destructor; clearing every
   * reference makes those resources eligible for collection even when this
   * disposed wrapper or one of its observers remains reachable.
   */
  dispose(): void {
    if (this.scratchEngine === null) return;
    this.scratchEngine = null;
    this.api = null;
    this.memory = null;
  }

  private _active(): { api: WasmPTQApi; memory: WebAssembly.Memory } {
    const api = this.api;
    const memory = this.memory;
    if (this.scratchEngine === null || api === null || memory === null) {
      throw new Error('WasmPTQ has been disposed.');
    }
    return { api, memory };
  }

  private _entry(
    name: string,
    value: ArrayBufferView,
    ArrayType: WasmPTQTypedArrayConstructor,
    read = false,
  ): ArenaEntry {
    return this._scratch(name, value.byteLength, ArrayType, value, read);
  }

  private _scratch(
    name: string,
    bytes: number,
    ArrayType: WasmPTQTypedArrayConstructor,
    value: ArrayBufferView | null = null,
    read = false,
  ): ArenaEntry {
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes > MAX_ALLOCATION_BYTES) {
      throw new Error(`WASM PTQ allocation '${name}' has invalid size ${bytes}.`);
    }
    return { name, value, bytes, ArrayType, read };
  }

  private _runArena(
    entries: ArenaEntry[],
    invoke: (pointers: Record<string, number>) => number | bigint | void,
  ): { value: number | bigint | void; outputs: Record<string, WasmPTQTypedArray> } {
    const { api, memory } = this._active();
    // This reset is safe because `api` belongs to the private fork created in
    // `WasmPTQ.create`, never the source engine executing the user's graph.
    api.reset_heap();
    const pointers: Record<string, number> = {};
    for (const entry of entries) {
      const pointer = Number(api.alloc_bytes(entry.bytes));
      if (!Number.isSafeInteger(pointer) || pointer < 0) {
        throw new Error(`WASM PTQ allocator returned an invalid pointer for '${entry.name}'.`);
      }
      const end = pointer + entry.bytes;
      if (!Number.isSafeInteger(end) || end > MAX_U32) {
        throw new Error(`WASM PTQ allocation '${entry.name}' exceeds the 32-bit address space.`);
      }
      if (end > memory.buffer.byteLength) {
        memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
      }
      pointers[entry.name] = pointer;
    }
    for (const entry of entries) {
      if (!entry.value) continue;
      const bytes = new Uint8Array(
        entry.value.buffer,
        entry.value.byteOffset,
        entry.value.byteLength,
      );
      new Uint8Array(memory.buffer, pointers[entry.name], entry.bytes).set(bytes);
    }
    const value = invoke(pointers);
    const outputs: Record<string, WasmPTQTypedArray> = {};
    for (const entry of entries) {
      if (!entry.read) continue;
      const bytes = memory.buffer.slice(
        pointers[entry.name],
        pointers[entry.name] + entry.bytes,
      );
      outputs[entry.name] = new entry.ArrayType(bytes) as WasmPTQTypedArray;
    }
    return { value, outputs };
  }

  private _expectSuccess(value: number | bigint | void, operation: string): void {
    if (Number(value) !== 0) {
      throw new Error(`The C WASM PTQ ${operation} rejected its inputs.`);
    }
  }
}

export function createWasmPTQ(sourceEngine: WasmEngine): Promise<WasmPTQ> {
  return WasmPTQ.create(sourceEngine);
}

export function createPTQ(sourceEngine: WasmEngine): Promise<WasmPTQ> {
  return createWasmPTQ(sourceEngine);
}
