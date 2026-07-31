const MAX_BATCH_MATMUL_RANK = 8;
const MAX_U32 = 0xffffffff;

function elementCount(shape) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0 || dimension > MAX_U32)) return null;
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  return Number.isSafeInteger(elements) && elements > 0 && elements <= MAX_U32
    ? elements
    : null;
}

function tensorElements(tensor) {
  const elements = elementCount(tensor?.shape);
  if (elements == null) return null;
  if (Number.isSafeInteger(tensor?.sizeBytes) && tensor.sizeBytes !== elements * 4) return null;
  if (tensor?.buffer && tensor.buffer.length !== elements) return null;
  return elements;
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

/**
 * Canonical ONNX MatMul descriptor for matrix operands (rank 2..8).
 *
 * Vector promotion is intentionally excluded from BatchMatMul: the frontend
 * must lower rank-1 ONNX MatMul separately. Batch dimensions use NumPy/ONNX
 * right-aligned broadcasting and the final two dimensions are [M,K]@[K,N].
 */
export function batchMatMulDescriptor(node) {
  const a = node.inputs?.a;
  const b = node.inputs?.b;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const aRank = a?.shape?.length;
  const bRank = b?.shape?.length;
  const outputRank = output?.shape?.length;
  const aElements = tensorElements(a);
  const bElements = tensorElements(b);
  const outputElements = tensorElements(output);
  if (inputNames.length !== 2 || inputNames[0] !== 'a' || inputNames[1] !== 'b' ||
      outputNames.length !== 1 || !a || !b || !output ||
      a.dtype !== 'float32' || b.dtype !== 'float32' || output.dtype !== 'float32' ||
      aElements == null || bElements == null || outputElements == null ||
      !Number.isInteger(aRank) || !Number.isInteger(bRank) ||
      aRank < 2 || bRank < 2 || aRank > MAX_BATCH_MATMUL_RANK ||
      bRank > MAX_BATCH_MATMUL_RANK || node.params == null ||
      typeof node.params !== 'object' || Array.isArray(node.params) ||
      Object.keys(node.params).length !== 0) {
    throw new Error(
      `BatchMatMul node ${node.id ?? '<unnamed>'} requires exactly F32 { a, b } -> { out }, rank-2..8 operands, and no parameters.`,
    );
  }

  const m = a.shape[aRank - 2];
  const k = a.shape[aRank - 1];
  const otherK = b.shape[bRank - 2];
  const n = b.shape[bRank - 1];
  if (k !== otherK) {
    throw new Error(
      `BatchMatMul node ${node.id ?? '<unnamed>'} has incompatible contraction dimensions ${k} and ${otherK}.`,
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
        `BatchMatMul node ${node.id ?? '<unnamed>'} has incompatible batch dimensions at axis ${axis}.`,
      );
    }
    outputBatch[axis] = Math.max(left, right);
  }
  const expectedOutputShape = [...outputBatch, m, n];
  if (outputRank !== expectedOutputShape.length ||
      output.shape.some((dimension, axis) => dimension !== expectedOutputShape[axis])) {
    throw new Error(
      `BatchMatMul node ${node.id ?? '<unnamed>'} output shape must be [${expectedOutputShape}].`,
    );
  }

  const outputBatchCount = elementCount(outputBatch);
  if (outputBatchCount == null || outputElements !== outputBatchCount * m * n ||
      aElements !== elementCount(aBatch) * m * k ||
      bElements !== elementCount(bBatch) * k * n) {
    throw new Error(`BatchMatMul node ${node.id ?? '<unnamed>'} has invalid tensor storage.`);
  }

  const outputBatchStrides = contiguousStrides(outputBatch);
  const aContiguous = contiguousStrides(paddedA);
  const bContiguous = contiguousStrides(paddedB);
  const aBatchStrides = paddedA.map((dimension, axis) =>
    dimension === 1 && outputBatch[axis] !== 1 ? 0 : aContiguous[axis] * m * k);
  const bBatchStrides = paddedB.map((dimension, axis) =>
    dimension === 1 && outputBatch[axis] !== 1 ? 0 : bContiguous[axis] * k * n);

  return {
    a, b, output, aRank, bRank, batchRank, m, k, n, outputBatchCount,
    outputBatch, outputBatchStrides, aBatchStrides, bBatchStrides,
    outputElements,
  };
}

export function _cpuBatchMatMul(node) {
  const descriptor = batchMatMulDescriptor(node);
  if (!(descriptor.a.buffer instanceof Float32Array) ||
      !(descriptor.b.buffer instanceof Float32Array) ||
      !(descriptor.output.buffer instanceof Float32Array)) {
    throw new Error(`BatchMatMul node ${node.id ?? '<unnamed>'} requires physical F32 storage.`);
  }

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
        let sum = 0;
        for (let inner = 0; inner < descriptor.k; inner++) {
          sum += descriptor.a.buffer[aBase + row * descriptor.k + inner] *
            descriptor.b.buffer[bBase + inner * descriptor.n + column];
        }
        descriptor.output.buffer[outputBase + row * descriptor.n + column] = sum;
      }
    }
  }
}
