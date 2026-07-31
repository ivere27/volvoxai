import type {
  PerAxisQuantization,
  PerAxisQuantizationInput,
  PerTensorQuantization,
  PerTensorQuantizationInput,
  RuntimeDType,
  RuntimeTypedArray,
  TensorOptions,
  TensorQuantization,
  TensorQuantizationInput,
  TensorShape,
  TensorStorage,
} from '../types.js';

export class Tensor {
  name: string;
  shape: TensorShape;
  dtype: RuntimeDType;
  isWeight: boolean;
  isInput: boolean;
  quantization: TensorQuantization | null;
  sizeBytes: number;
  buffer?: TensorStorage;

  constructor(
    name: string,
    shape: readonly number[],
    dtype: RuntimeDType = "float32",
    isWeight = false,
    options: TensorOptions = {},
  ) {
    this.name = name;
    this.shape = [...shape];
    this.dtype = dtype;
    this.isWeight = !!isWeight;
    // Inputs and intermediate values are both non-weight tensors, but they are
    // not interchangeable when validating a dynamically edited graph.  Keep an
    // explicit marker so removing a producer cannot silently turn a value into
    // a new public model input.
    this.isInput = !!options.isInput;
    // A quantized tensor owns its mapping to real values. The descriptor is
    // deliberately immutable so shape-only nodes can safely preserve it and a
    // later consumer never has to infer scales from a downstream operator.
    this.quantization = Tensor.normalizeQuantization(dtype, this.shape, options.quantization,
      `Tensor '${name}' quantization`);
    this.sizeBytes = this._calculateByteSize();
  }
  _calculateByteSize(): number {
    const elements = this.shape.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(elements)) throw new Error(`Tensor '${this.name}' is too large.`);
    const bytes = Tensor.dtypeBytes(this.dtype);
    const sizeBytes = elements * bytes;
    if (!Number.isSafeInteger(sizeBytes)) {
      throw new Error(`Tensor '${this.name}' byte size is too large.`);
    }
    return sizeBytes;
  }
  static dtypeBytes(dtype: RuntimeDType): number {
    if (dtype === "float32" || dtype === "int32") return 4;
    if (dtype === "int8" || dtype === "uint8") return 1;
    throw new Error(`Unsupported runtime tensor dtype '${dtype}'.`);
  }
  static assertCompatibleBuffer(
    dtype: RuntimeDType,
    buffer: ArrayBuffer | ArrayBufferView,
    expectedBytes: number,
    label = "Tensor buffer",
  ): asserts buffer is TensorStorage {
    if (!ArrayBuffer.isView(buffer) && !(buffer instanceof ArrayBuffer)) {
      throw new Error(`${label} must be an ArrayBuffer or typed-array view.`);
    }
    if (buffer.byteLength !== expectedBytes) {
      throw new Error(`${label} byte length ${buffer.byteLength} does not match expected ${expectedBytes}.`);
    }
    if (buffer instanceof ArrayBuffer) return;
    const valid = (dtype === "float32" && buffer instanceof Float32Array) ||
      (dtype === "int32" && buffer instanceof Int32Array) ||
      (dtype === "int8" && buffer instanceof Int8Array) ||
      (dtype === "uint8" && (buffer instanceof Uint8Array || buffer instanceof Uint8ClampedArray));
    if (!valid) throw new Error(`${label} typed storage does not match dtype '${dtype}'.`);
  }
  static assertCompatibleInput(
    dtype: RuntimeDType,
    value: unknown,
    expectedBytes: number,
    label = "Input tensor",
  ): RuntimeTypedArray {
    if (!ArrayBuffer.isView(value) || value instanceof DataView) {
      throw new Error(`${label} must be a dtype-matching typed array.`);
    }
    Tensor.assertCompatibleBuffer(dtype, value, expectedBytes, label);
    return value as RuntimeTypedArray;
  }
  static normalizeQuantization(
    dtype: RuntimeDType,
    shape: readonly number[],
    quantization: null | undefined,
    label?: string,
  ): null;
  static normalizeQuantization(
    dtype: RuntimeDType,
    shape: readonly number[],
    quantization: PerTensorQuantizationInput | PerTensorQuantization,
    label?: string,
  ): PerTensorQuantization;
  static normalizeQuantization(
    dtype: RuntimeDType,
    shape: readonly number[],
    quantization: PerAxisQuantizationInput | PerAxisQuantization,
    label?: string,
  ): PerAxisQuantization;
  static normalizeQuantization(
    dtype: RuntimeDType,
    shape: readonly number[],
    quantization: TensorQuantizationInput | TensorQuantization | null | undefined,
    label?: string,
  ): TensorQuantization | null;
  static normalizeQuantization(
    dtype: RuntimeDType,
    shape: readonly number[],
    quantization: TensorQuantizationInput | TensorQuantization | null | undefined,
    label = "Tensor quantization",
  ): TensorQuantization | null {
    if (quantization == null) return null;
    if (!quantization || typeof quantization !== "object" || Array.isArray(quantization)) {
      throw new Error(`${label} must be an object.`);
    }
    if (dtype !== "int8" && dtype !== "uint8") {
      throw new Error(`${label} requires int8 or uint8 tensor storage.`);
    }
    const minimum = dtype === "int8" ? -128 : 0;
    const maximum = dtype === "int8" ? 127 : 255;
    const scheme = quantization.scheme ?? "per_tensor";
    const zeroPoint = (value: number, valueLabel: string): number => {
      if (!Number.isInteger(value) || value < minimum || value > maximum) {
        throw new Error(`${label} ${valueLabel} must be an integer in [${minimum}, ${maximum}].`);
      }
      return value;
    };
    const scale = (value: number, valueLabel: string): number => {
      const f32 = Math.fround(value);
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 ||
          !Number.isFinite(f32) || f32 <= 0) {
        throw new Error(`${label} ${valueLabel} must be finite, positive, and representable as F32.`);
      }
      return f32;
    };
    if (scheme === "per_tensor") {
      const descriptor = quantization as PerTensorQuantizationInput;
      return Object.freeze({
        scheme,
        scale: scale(descriptor.scale, "scale"),
        zero_point: zeroPoint(descriptor.zero_point ?? 0, "zero_point"),
      });
    }
    if (scheme === "per_axis") {
      const descriptor = quantization as PerAxisQuantizationInput;
      let axis = descriptor.axis;
      if (!Number.isInteger(axis)) throw new Error(`${label} per_axis scheme requires an integer axis.`);
      if (axis < 0) axis += shape.length;
      if (axis < 0 || axis >= shape.length) throw new Error(`${label} axis is outside tensor rank ${shape.length}.`);
      if (!Array.isArray(descriptor.scales) || descriptor.scales.length !== shape[axis]) {
        throw new Error(`${label} scales must contain one value for each axis-${axis} element.`);
      }
      const scales = descriptor.scales.map((value, index) => scale(value, `scales[${index}]`));
      const sourceZeroPoints = descriptor.zero_points ?? new Array<number>(shape[axis]).fill(0);
      if (!Array.isArray(sourceZeroPoints) || sourceZeroPoints.length !== shape[axis]) {
        throw new Error(`${label} zero_points must contain one value for each axis-${axis} element.`);
      }
      const zeroPoints = sourceZeroPoints.map((value, index) => zeroPoint(value, `zero_points[${index}]`));
      return Object.freeze({
        scheme,
        axis,
        scales: Object.freeze(scales),
        zero_points: Object.freeze(zeroPoints),
      });
    }
    throw new Error(`${label} has unsupported scheme '${scheme}'.`);
  }
};
