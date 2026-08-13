import type { RuntimeDType, TensorQuantization } from '../types.js';
import { checkedShapeElementCount, runtimeDTypeBytes } from './shapeSystem.js';
import { sameQuantizationDescriptor } from './quantizedShape.js';

export const SHAPE_KERNEL_DTYPES: readonly RuntimeDType[] = Object.freeze([
  'float32', 'int32', 'int8', 'uint8',
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function storageMatches(dtype: RuntimeDType, value: unknown): boolean {
  return (dtype === 'float32' && value instanceof Float32Array) ||
    (dtype === 'int32' && value instanceof Int32Array) ||
    (dtype === 'int8' && value instanceof Int8Array) ||
    (dtype === 'uint8' &&
      (value instanceof Uint8Array || value instanceof Uint8ClampedArray));
}

function assertQuantizationDescriptor(tensor: any, label: string): void {
  const descriptor: any = tensor.quantization;
  if (descriptor == null) return;
  if (tensor.dtype !== 'int8' && tensor.dtype !== 'uint8') {
    throw new Error(`${label} cannot attach quantization metadata to ${tensor.dtype} storage.`);
  }
  if (!isRecord(descriptor)) {
    throw new Error(`${label} quantization descriptor must be an object.`);
  }
  const quantization: any = descriptor;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  const validScale = (value: unknown): value is number =>
    typeof value === 'number' && Number.isFinite(value) && value > 0 &&
      Number.isFinite(Math.fround(value)) && Math.fround(value) > 0;
  const validZeroPoint = (value: unknown): value is number =>
    Number.isInteger(value) && (value as number) >= minimum && (value as number) <= maximum;
  if (quantization.scheme === 'per_tensor') {
    if (!validScale(quantization.scale) || !validZeroPoint(quantization.zero_point)) {
      throw new Error(`${label} has an invalid per-tensor quantization descriptor.`);
    }
    return;
  }
  if (quantization.scheme !== 'per_axis' || !Number.isInteger(quantization.axis) ||
      quantization.axis < 0 || quantization.axis >= tensor.shape.length ||
      !Array.isArray(quantization.scales) || !Array.isArray(quantization.zero_points) ||
      quantization.scales.length !== tensor.shape[quantization.axis] ||
      quantization.zero_points.length !== quantization.scales.length ||
      quantization.scales.some((value: unknown) => !validScale(value)) ||
      quantization.zero_points.some((value: unknown) => !validZeroPoint(value))) {
    throw new Error(`${label} has an invalid per-axis quantization descriptor.`);
  }
}

export function assertShapeKernelTensor(
  tensor: any,
  label: string,
  options: Readonly<{
    dtypes?: readonly RuntimeDType[];
    minimumRank?: number;
    maximumRank?: number;
  }> = {},
): number {
  const dtypes = options.dtypes ?? SHAPE_KERNEL_DTYPES;
  if (!tensor || !dtypes.includes(tensor.dtype)) {
    throw new Error(`${label} must use one of the supported dtypes [${dtypes.join(', ')}].`);
  }
  if (!Array.isArray(tensor.shape)) throw new Error(`${label} shape must be an array.`);
  const minimumRank = options.minimumRank ?? 0;
  const maximumRank = options.maximumRank ?? Number.POSITIVE_INFINITY;
  if (tensor.shape.length < minimumRank || tensor.shape.length > maximumRank) {
    throw new Error(`${label} rank must be in [${minimumRank}, ${maximumRank}].`);
  }
  let elements: number;
  try {
    elements = checkedShapeElementCount(tensor.shape, `${label} shape`);
  } catch (error) {
    throw new Error(`${label} has an invalid concrete shape.`, { cause: error });
  }
  if (!storageMatches(tensor.dtype, tensor.buffer) || tensor.buffer.length !== elements) {
    throw new Error(`${label} physical storage does not match its dtype and shape.`);
  }
  const expectedBytes = elements * runtimeDTypeBytes(tensor.dtype);
  if (!Number.isSafeInteger(expectedBytes) ||
      (tensor.sizeBytes !== undefined && tensor.sizeBytes !== expectedBytes)) {
    throw new Error(`${label} byte length does not match its dtype and shape.`);
  }
  assertQuantizationDescriptor(tensor, label);
  return elements;
}

export function assertShapeKernelParams(
  node: any,
  allowed: readonly string[],
  operation: string,
): Readonly<Record<string, unknown>> {
  const params = node?.params ?? {};
  if (!isRecord(params)) throw new Error(`${operation} parameters must be an object.`);
  const permitted = new Set(allowed);
  const unexpected = Reflect.ownKeys(params)
    .filter((key): key is string => typeof key === 'string' && !permitted.has(key))
    .sort();
  if (Reflect.ownKeys(params).some((key) => typeof key !== 'string') || unexpected.length !== 0) {
    throw new Error(`${operation} has unsupported parameter '${String(unexpected[0] ?? '<symbol>')}'.`);
  }
  return params;
}

export function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, axis) => dimension === right[axis]);
}

export function assertShapeKernelOutput(
  output: any,
  expectedShape: readonly number[],
  expectedDType: RuntimeDType,
  expectedQuantization: TensorQuantization | null | undefined,
  operation: string,
): number {
  const elements = assertShapeKernelTensor(output, `${operation} output`, {
    dtypes: [expectedDType],
  });
  if (!sameShape(output.shape, expectedShape)) {
    throw new Error(
      `${operation} output shape [${output.shape}] does not match [${expectedShape}].`,
    );
  }
  if (!sameQuantizationDescriptor(output.quantization, expectedQuantization)) {
    throw new Error(`${operation} output quantization metadata does not match the canonical transform.`);
  }
  return elements;
}

export function normalizeShapeKernelAxis(
  value: unknown,
  rank: number,
  defaultAxis: number,
  operation: string,
): number {
  if (rank < 1) throw new Error(`${operation} requires rank at least one.`);
  const raw = value ?? defaultAxis;
  if (!Number.isInteger(raw)) throw new Error(`${operation} axis must be an integer.`);
  const axis = (raw as number) < 0 ? (raw as number) + rank : raw as number;
  if (axis < 0 || axis >= rank) throw new Error(`${operation} axis is outside rank ${rank}.`);
  return axis;
}

export function normalizeShapeKernelAxes(
  value: unknown,
  rank: number,
  operation: string,
): readonly number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${operation} axes must be a non-empty array.`);
  }
  const axes = value.map((raw, index) => {
    if (!Number.isInteger(raw)) throw new Error(`${operation} axis ${index} must be an integer.`);
    const axis = raw < 0 ? raw + rank : raw;
    if (axis < 0 || axis >= rank) throw new Error(`${operation} axis ${index} is outside rank ${rank}.`);
    return axis;
  });
  if (new Set(axes).size !== axes.length) throw new Error(`${operation} axes must be unique.`);
  return Object.freeze(axes.sort((left, right) => left - right));
}

export function remapShapeKernelQuantization(
  descriptor: TensorQuantization | null | undefined,
  axis: number,
): TensorQuantization | undefined {
  if (descriptor == null) return undefined;
  if (descriptor.scheme === 'per_tensor') return descriptor;
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze([...descriptor.scales]),
    zero_points: Object.freeze([...descriptor.zero_points]),
  });
}

export function sliceShapeKernelQuantization(
  descriptor: TensorQuantization | null | undefined,
  axis: number,
  start: number,
  end: number,
  step = 1,
): TensorQuantization | undefined {
  if (descriptor == null) return undefined;
  if (descriptor.scheme === 'per_tensor') return descriptor;
  const scales: number[] = [];
  const zeroPoints: number[] = [];
  for (let index = start; index < end; index += step) {
    scales.push(descriptor.scales[index]);
    zeroPoints.push(descriptor.zero_points[index]);
  }
  return Object.freeze({
    scheme: 'per_axis',
    axis,
    scales: Object.freeze(scales),
    zero_points: Object.freeze(zeroPoints),
  });
}

function checkedProduct(shape: readonly number[], operation: string): number {
  try {
    return checkedShapeElementCount(shape, operation);
  } catch (error) {
    throw new Error(`${operation} product is invalid.`, { cause: error });
  }
}

export function reshapeShapeKernelQuantization(
  descriptor: TensorQuantization | null | undefined,
  inputShape: readonly number[],
  outputShape: readonly number[],
  operation: string,
): TensorQuantization | undefined {
  if (descriptor == null || descriptor.scheme === 'per_tensor') return descriptor ?? undefined;
  const candidates: number[] = [];
  for (let axis = 0; axis < outputShape.length; axis++) {
    if (inputShape[descriptor.axis] !== outputShape[axis]) continue;
    if (checkedProduct(inputShape.slice(0, descriptor.axis), `${operation} input prefix`) !==
        checkedProduct(outputShape.slice(0, axis), `${operation} output prefix`)) continue;
    if (checkedProduct(inputShape.slice(descriptor.axis + 1), `${operation} input suffix`) !==
        checkedProduct(outputShape.slice(axis + 1), `${operation} output suffix`)) continue;
    candidates.push(axis);
  }
  if (candidates.length !== 1) {
    throw new Error(`${operation} cannot map the per-axis quantization coordinate uniquely.`);
  }
  return remapShapeKernelQuantization(descriptor, candidates[0]);
}
