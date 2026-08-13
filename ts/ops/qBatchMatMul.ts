import { Tensor } from '../core/Tensor.js';
import { roundTiesToEven } from './quantizeLinear.js';
import type { PerTensorQuantization } from '../types.js';

const MAX_RANK = 8;
const MAX_U32 = 0xffffffff;
const INT32_MAX = 0x7fffffff;

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function elementCount(shape) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32
    ? elements
    : null;
}

function typedByteStorage(tensor) {
  if (tensor?.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  if (tensor?.dtype === 'uint8') {
    return tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray;
  }
  return false;
}

function requireByteTensor(tensor, label, node, requireHostStorage) {
  const elements = elementCount(tensor?.shape);
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) || elements == null ||
      tensor.sizeBytes !== elements ||
      (requireHostStorage &&
        (!typedByteStorage(tensor) || tensor.buffer.byteLength !== elements)) ||
      (!requireHostStorage && tensor.buffer != null &&
        (!typedByteStorage(tensor) || tensor.buffer.byteLength !== elements))) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} ${label} requires contiguous I8/U8 storage.`,
    );
  }
  const quantization = Tensor.normalizeQuantization(
    tensor.dtype,
    tensor.shape,
    tensor.quantization,
    `QBatchMatMul node ${nodeLabel(node)} ${label} quantization`,
  ) as PerTensorQuantization | null;
  if (!quantization || quantization.scheme !== 'per_tensor') {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} ${label} requires per_tensor quantization.`,
    );
  }
  const scale = Math.fround(quantization.scale);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} ${label} scale is not representable as positive F32.`,
    );
  }
  return { elements, quantization, scale };
}

function contiguousStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    strides[axis] = stride;
    stride *= shape[axis];
  }
  return strides;
}

function storageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || !ArrayBuffer.isView(rightStorage) ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

/**
 * W8A8 ONNX MatMul for rank-2..8 operands.
 *
 * Both operands and the output use immutable per-tensor affine descriptors.
 * Batch axes follow right-aligned ONNX broadcasting; the matrix axes are
 * [M,K] @ [K,N]. Vector promotion is intentionally handled by the frontend.
 */
function qBatchMatMulDescriptorImpl(node, requireHostStorage) {
  const a = node?.inputs?.a;
  const b = node?.inputs?.b;
  const output = node?.outputs?.out || Object.values(node?.outputs || {})[0];
  const inputNames = Object.keys(node?.inputs || {}).sort();
  const outputNames = Object.keys(node?.outputs || {});
  if (inputNames.length !== 2 || inputNames[0] !== 'a' || inputNames[1] !== 'b' ||
      outputNames.length !== 1 || !a || !b || !output ||
      !Number.isInteger(a.shape?.length) || !Number.isInteger(b.shape?.length) ||
      a.shape.length < 2 || b.shape.length < 2 ||
      a.shape.length > MAX_RANK || b.shape.length > MAX_RANK ||
      node.params == null || typeof node.params !== 'object' ||
      Array.isArray(node.params) || Object.keys(node.params).length !== 0) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} requires exactly byte { a, b } -> { out }, rank-2..8 operands, and no parameters.`,
    );
  }

  const aDescriptor = requireByteTensor(a, 'a', node, requireHostStorage);
  const bDescriptor = requireByteTensor(b, 'b', node, requireHostStorage);
  const outputDescriptor = requireByteTensor(output, 'output', node, requireHostStorage);
  const aRank = a.shape.length;
  const bRank = b.shape.length;
  const m = a.shape[aRank - 2];
  const k = a.shape[aRank - 1];
  const otherK = b.shape[bRank - 2];
  const n = b.shape[bRank - 1];
  if (k !== otherK) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} has incompatible contraction dimensions ${k} and ${otherK}.`,
    );
  }

  const aBatch = a.shape.slice(0, -2);
  const bBatch = b.shape.slice(0, -2);
  const batchRank = Math.max(aBatch.length, bBatch.length);
  const paddedA = [...new Array(batchRank - aBatch.length).fill(1), ...aBatch];
  const paddedB = [...new Array(batchRank - bBatch.length).fill(1), ...bBatch];
  const outputBatch = new Array(batchRank);
  for (let axis = 0; axis < batchRank; axis++) {
    const left = paddedA[axis];
    const right = paddedB[axis];
    if (left !== right && left !== 1 && right !== 1) {
      throw new Error(
        `QBatchMatMul node ${nodeLabel(node)} has incompatible batch dimensions at axis ${axis}.`,
      );
    }
    outputBatch[axis] = Math.max(left, right);
  }
  const expectedOutputShape = [...outputBatch, m, n];
  if (output.shape.length !== expectedOutputShape.length ||
      output.shape.some((dimension, axis) => dimension !== expectedOutputShape[axis])) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} output shape must be [${expectedOutputShape}].`,
    );
  }

  const outputBatchCount = elementCount(outputBatch);
  if (outputBatchCount == null ||
      aDescriptor.elements !== elementCount(aBatch) * m * k ||
      bDescriptor.elements !== elementCount(bBatch) * k * n ||
      outputDescriptor.elements !== outputBatchCount * m * n) {
    throw new Error(`QBatchMatMul node ${nodeLabel(node)} has invalid tensor storage.`);
  }
  if ((output.buffer != null && output.buffer === a.buffer) ||
      (output.buffer != null && output.buffer === b.buffer) ||
      storageRangesOverlap(output, a) || storageRangesOverlap(output, b)) {
    throw new Error(`QBatchMatMul node ${nodeLabel(node)} output must not alias an input.`);
  }

  const range = (tensor, quantization) => {
    const minimum = tensor.dtype === 'int8' ? -128 : 0;
    const maximum = tensor.dtype === 'int8' ? 127 : 255;
    return Math.max(
      Math.abs(minimum - quantization.zero_point),
      Math.abs(maximum - quantization.zero_point),
    );
  };
  const maximumAccumulator = range(a, aDescriptor.quantization) *
    range(b, bDescriptor.quantization) * k;
  if (!Number.isSafeInteger(maximumAccumulator) || maximumAccumulator > INT32_MAX) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} may overflow its defined I32 accumulator.`,
    );
  }
  const multiplier = Math.fround(
    Math.fround(aDescriptor.scale * bDescriptor.scale) / outputDescriptor.scale,
  );
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(
      `QBatchMatMul node ${nodeLabel(node)} requantization multiplier is not representable as positive F32.`,
    );
  }

  const outputBatchStrides = contiguousStrides(outputBatch);
  const aContiguous = contiguousStrides(paddedA);
  const bContiguous = contiguousStrides(paddedB);
  const aBatchStrides = paddedA.map((dimension, axis) =>
    dimension === 1 && outputBatch[axis] !== 1 ? 0 : aContiguous[axis] * m * k);
  const bBatchStrides = paddedB.map((dimension, axis) =>
    dimension === 1 && outputBatch[axis] !== 1 ? 0 : bContiguous[axis] * k * n);
  return {
    a,
    b,
    output,
    aQuantization: aDescriptor.quantization,
    bQuantization: bDescriptor.quantization,
    outputQuantization: outputDescriptor.quantization,
    aScale: aDescriptor.scale,
    bScale: bDescriptor.scale,
    outputScale: outputDescriptor.scale,
    multiplier,
    batchRank,
    m,
    k,
    n,
    outputBatchCount,
    outputBatch,
    outputBatchStrides,
    aBatchStrides,
    bBatchStrides,
    outputElements: outputDescriptor.elements,
  };
}

/**
 * Validate the complete host-executable contract, including exact typed-array
 * storage for every byte tensor.
 */
export function qBatchMatMulDescriptor(node) {
  return qBatchMatMulDescriptorImpl(node, true);
}

/**
 * Validate the same arithmetic/shape/quantization contract before a device
 * backend has populated host Tensor.buffer fields. Device backends must still
 * prove that their own input/output storage exists and does not alias.
 */
export function qBatchMatMulMetadataDescriptor(node) {
  return qBatchMatMulDescriptorImpl(node, false);
}

function requantize(accumulator, descriptor) {
  const transformed = Math.fround(
    Math.fround(Math.fround(accumulator) * descriptor.multiplier) +
      descriptor.outputQuantization.zero_point,
  );
  const minimum = descriptor.output.dtype === 'int8' ? -128 : 0;
  const maximum = descriptor.output.dtype === 'int8' ? 127 : 255;
  if (Number.isNaN(transformed)) return descriptor.outputQuantization.zero_point;
  if (transformed <= minimum) return minimum;
  if (transformed >= maximum) return maximum;
  return roundTiesToEven(transformed);
}

export function _cpuQBatchMatMul(node) {
  const descriptor = qBatchMatMulDescriptor(node);
  for (let batch = 0; batch < descriptor.outputBatchCount; batch++) {
    let remaining = batch;
    let aBase = 0;
    let bBase = 0;
    for (let axis = 0; axis < descriptor.batchRank; axis++) {
      const coordinate = Math.floor(remaining / descriptor.outputBatchStrides[axis]);
      remaining %= descriptor.outputBatchStrides[axis];
      aBase += coordinate * descriptor.aBatchStrides[axis];
      bBase += coordinate * descriptor.bBatchStrides[axis];
    }
    const outputBase = batch * descriptor.m * descriptor.n;
    for (let row = 0; row < descriptor.m; row++) {
      for (let column = 0; column < descriptor.n; column++) {
        let accumulator = 0;
        for (let inner = 0; inner < descriptor.k; inner++) {
          accumulator += (
            descriptor.a.buffer[aBase + row * descriptor.k + inner] -
            descriptor.aQuantization.zero_point
          ) * (
            descriptor.b.buffer[bBase + inner * descriptor.n + column] -
            descriptor.bQuantization.zero_point
          );
        }
        descriptor.output.buffer[outputBase + row * descriptor.n + column] =
          requantize(accumulator, descriptor);
      }
    }
  }
}
