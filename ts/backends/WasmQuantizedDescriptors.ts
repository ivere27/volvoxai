/* Portable WASM descriptor builders for the W8A8 quantized operators,
 * with the storage-aliasing preflight checks each one runs first.
 *
 * Split out of WasmEngine.ts; depends only on WasmPortablePrimitives. */
import {
  WASM_PORTABLE_MAX_RANK,
  WASM_PORTABLE_MAX_U32,
  portableByteQuantizedTensor,
  portableElementCount,
  portableF32Tensor,
  portableOutput,
  portablePairParameter,
  portableTensorElements,
  sameShape,
  wasmDtypeCode,
} from './WasmPortablePrimitives.js';

export function immutableQuantizationDescriptor(tensor) {
  if (!portableByteQuantizedTensor(tensor)) return null;
  const descriptor = tensor.quantization;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  const validScale = (value) => typeof value === 'number' && Number.isFinite(value) && value > 0;
  const validZeroPoint = (value) => Number.isInteger(value) && value >= minimum && value <= maximum;
  if (!descriptor || typeof descriptor !== 'object' || !Object.isFrozen(descriptor)) return null;
  if (descriptor.scheme === 'per_tensor') {
    return validScale(descriptor.scale) && validZeroPoint(descriptor.zero_point) ? descriptor : null;
  }
  if (descriptor.scheme === 'per_axis') {
    const axis = descriptor.axis;
    if (!Number.isInteger(axis) || axis < 0 || axis >= tensor.shape.length ||
        !Array.isArray(descriptor.scales) || !Array.isArray(descriptor.zero_points) ||
        !Object.isFrozen(descriptor.scales) || !Object.isFrozen(descriptor.zero_points) ||
        descriptor.scales.length !== tensor.shape[axis] ||
        descriptor.zero_points.length !== tensor.shape[axis] ||
        !descriptor.scales.every(validScale) || !descriptor.zero_points.every(validZeroPoint)) return null;
    return descriptor;
  }
  return null;
}
export function portableQLinearDescriptor(node) {
  if (!['QLinear', 'QMatMul', 'QGemm'].includes(node.opType)) return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.a;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const biasElements = portableTensorElements(bias);
  const outputElements = portableTensorElements(output);
  if (!portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(weight) ||
      !portableByteQuantizedTensor(output) || !bias || bias.dtype !== 'int32' ||
      inputElements == null || weightElements == null || biasElements == null ||
      outputElements == null || input.shape.length < 1 ||
      output.shape.length !== input.shape.length ||
      input.shape.slice(0, -1).some((dimension, index) => dimension !== output.shape[index]) ||
      weight.shape.length !== 2 || bias.shape.length !== 1) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires typed [...,d_in] input/output, [d_out,d_in] I8/U8 weights, and I32 bias.`);
  }
  const dIn = input.shape.at(-1);
  const dOut = output.shape.at(-1);
  const rows = inputElements / dIn;
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(dIn) || dIn <= 0 ||
      !Number.isInteger(dOut) || dOut <= 0 || inputElements !== rows * dIn ||
      outputElements !== rows * dOut || !sameShape(weight.shape, [dOut, dIn]) ||
      weightElements !== dOut * dIn || !sameShape(bias.shape, [dOut]) || biasElements !== dOut) {
    throw new Error(`WASM ${node.opType} node ${node.id} has incompatible [d_out,d_in] dimensions or I32 bias.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!inputQuantization || !outputQuantization || !weightQuantization) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires immutable I8/U8 quantization metadata.`);
  }
  if (inputQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
      weightQuantization.scales.length !== dOut || weightQuantization.zero_points.length !== dOut) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires per-tensor input/output and axis-0 per-channel weight quantization.`);
  }
  const f32Scale = (value) => Math.fround(value);
  const inputScale = f32Scale(inputQuantization.scale);
  const outputScale = f32Scale(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, f32Scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM ${node.opType} node ${node.id} requires scales representable as positive F32.`);
  }
  if (!weightScales.every((scale) => {
    const multiplier = Math.fround(Math.fround(inputScale * scale) / outputScale);
    return Number.isFinite(multiplier) && multiplier > 0;
  })) {
    throw new Error(
      `WASM ${node.opType} node ${node.id} requantization multiplier is not representable as positive F32.`,
    );
  }
  return {
    kind: 'qlinear', input, weight, bias, output, rows, dIn, dOut,
    inputDtype: wasmDtypeCode(input.dtype), weightDtype: wasmDtypeCode(weight.dtype),
    outputDtype: wasmDtypeCode(output.dtype), inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}
export function portableQEmbeddingDescriptor(node) {
  if (node.opType !== 'QEmbedding') return null;
  const input = node.inputs.input;
  const weight = node.inputs.weight;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const outputElements = portableTensorElements(output);
  const outputCount = Object.values(node.outputs || {}).filter(Boolean).length;
  if (Object.keys(node.inputs || {}).length !== 2 || !input || !weight || outputCount !== 1 ||
      input.dtype !== 'int32' || inputElements == null || input.shape.length < 1 ||
      !portableByteQuantizedTensor(weight) || !portableByteQuantizedTensor(output) ||
      weightElements == null || outputElements == null || weight.shape.length !== 2) {
    throw new Error(`WASM QEmbedding node ${node.id} requires preflight-complete I32 [...token] IDs, rank-2 I8/U8 [vocab,hidden] weights, and an I8/U8 output.`);
  }
  const [vocab, hidden] = weight.shape;
  const expectedOutputShape = [...input.shape, hidden];
  if (!Number.isInteger(vocab) || vocab <= 0 || !Number.isInteger(hidden) || hidden <= 0 ||
      !sameShape(output.shape, expectedOutputShape) || weightElements !== vocab * hidden ||
      outputElements !== inputElements * hidden) {
    throw new Error(`WASM QEmbedding node ${node.id} has incompatible ID, [vocab,hidden] weight, or output dimensions.`);
  }
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!outputQuantization || !weightQuantization || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis' || weightQuantization.axis !== 0 ||
      weightQuantization.scales.length !== vocab || weightQuantization.zero_points.length !== vocab) {
    throw new Error(`WASM QEmbedding node ${node.id} requires immutable per-tensor output and axis-0 per-row weight quantization metadata.`);
  }
  const outputScale = Math.fround(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, Math.fround);
  if (!Number.isFinite(outputScale) || outputScale <= 0 ||
      !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QEmbedding node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qembedding', input, weight, output, tokens: inputElements, vocab, hidden,
    weightDtype: wasmDtypeCode(weight.dtype), outputDtype: wasmDtypeCode(output.dtype),
    outputScale, outputZeroPoint: outputQuantization.zero_point,
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}
export function portableQAddDescriptor(node) {
  if (node.opType !== 'QAdd') return null;
  const a = node.inputs.a || node.inputs.input || node.inputs.x;
  const b = node.inputs.b || node.inputs.y;
  const output = portableOutput(node);
  const elements = portableTensorElements(output);
  const relu = node.params?.relu ?? 0;
  if (!portableByteQuantizedTensor(a) || !portableByteQuantizedTensor(b) ||
      !portableByteQuantizedTensor(output) || elements == null ||
      portableTensorElements(a) !== elements || portableTensorElements(b) !== elements ||
      !sameShape(a.shape, b.shape) || !sameShape(a.shape, output.shape) ||
      !Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error(`WASM QAdd node ${node.id} requires exact-shape I8/U8 input/output tensors.`);
  }
  const aQuantization = immutableQuantizationDescriptor(a);
  const bQuantization = immutableQuantizationDescriptor(b);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!aQuantization || !bQuantization || !outputQuantization ||
      aQuantization.scheme !== 'per_tensor' || bQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QAdd node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const aScale = Math.fround(aQuantization.scale);
  const bScale = Math.fround(bQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (![aScale, bScale, outputScale].every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QAdd node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qadd', a, b, output, elements,
    aDtype: wasmDtypeCode(a.dtype), bDtype: wasmDtypeCode(b.dtype),
    outputDtype: wasmDtypeCode(output.dtype),
    aScale, bScale, outputScale,
    aZeroPoint: aQuantization.zero_point, bZeroPoint: bQuantization.zero_point,
    outputZeroPoint: outputQuantization.zero_point, relu,
  };
}
export function portableQSiLUDescriptor(node) {
  if (node.opType !== 'QSiLU') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  if (!input || !output || inputEntries.length !== 1 || inputEntries[0][1] !== input ||
      outputEntries.length !== 1 || input === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null ||
      outputElements !== inputElements || !sameShape(input.shape, output.shape) ||
      Object.keys(node.params || {}).length !== 0) {
    throw new Error(`WASM QSiLU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and no parameters.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QSiLU node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QSiLU node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qsilu', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}
export function portableQGELUDescriptor(node) {
  if (node.opType !== 'QGELU') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {}).filter(([, tensor]) => !!tensor);
  const outputEntries = Object.values(node.outputs || {}).filter(Boolean);
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const parameterNames = Object.keys(node.params || {});
  const canonicalParameters = parameterNames.length === 0 ||
    (parameterNames.length === 1 && parameterNames[0] === 'approximate' &&
      node.params.approximate === 'none');
  if (!input || !output || inputEntries.length !== 1 || inputEntries[0][1] !== input ||
      outputEntries.length !== 1 || input === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null ||
      outputElements !== inputElements || !sameShape(input.shape, output.shape) ||
      !canonicalParameters) {
    throw new Error(`WASM QGELU node ${node.id} requires distinct same-shape I8/U8 input/output tensors and only omitted parameters or approximate='none'.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QGELU node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QGELU node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qgelu', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}
export function qGroupNormByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}
export function qGroupNormAffineStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  return storage instanceof Float32Array && storage.byteLength === tensor.sizeBytes &&
    storage.length === tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT;
}
export function qGroupNormStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Do this before `_alloc` converts every tensor to a fresh WASM typed view.
// It preserves the portable operator's typed-storage and no-overlap contract
// for graph inputs supplied as ArrayBuffer views, while still allowing normal
// unallocated graph values to receive their expected typed storage.
// Do this before `_alloc` converts every tensor to a fresh WASM typed view.
// It preserves the portable operator's typed-storage and no-overlap contract
// for graph inputs supplied as ArrayBuffer views, while still allowing normal
// unallocated graph values to receive their expected typed storage.
export function preflightPortableQGroupNormStorage(node) {
  if (node.opType !== 'QGroupNorm') return;
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qGroupNormByteStorageIsCanonical(input) || !qGroupNormByteStorageIsCanonical(output) ||
      !qGroupNormAffineStorageIsCanonical(weight) || !qGroupNormAffineStorageIsCanonical(bias) ||
      qGroupNormStorageRangesOverlap(input, output) ||
      qGroupNormStorageRangesOverlap(weight, output) ||
      qGroupNormStorageRangesOverlap(bias, output)) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires canonical typed input/output storage, finite F32 [C] affine storage, and output storage distinct from every input.`);
  }
}
export function portableQGroupNormDescriptor(node, affineValues = (tensor) => tensor?.buffer) {
  if (node.opType !== 'QGroupNorm') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
  const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['num_groups', 'eps', 'data_layout'].includes(name));
  const groups = params?.num_groups;
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const [batch, height, width, channels] = input?.shape || [];
  const validAffine = (tensor) => {
    const values = affineValues(tensor);
    return portableF32Tensor(tensor) && sameShape(tensor.shape, [channels]) &&
      tensor.sizeBytes === channels * Float32Array.BYTES_PER_ELEMENT &&
      values instanceof Float32Array && values.length === channels &&
      (tensor.isWeight !== true || values.every(Number.isFinite));
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputNames.length !== 1 || !input || !weight || !bias || !output ||
      input === output || weight === output || bias === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null || outputElements !== inputElements ||
      input.shape.length !== 4 || !sameShape(input.shape, output.shape) || !allowedParameters ||
      !Number.isInteger(groups) || groups <= 0 || !Number.isInteger(channels) || channels <= 0 ||
      channels % groups !== 0 || (params?.data_layout != null && params.data_layout !== 'NHWC') ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0 || !validAffine(weight) || !validAffine(bias)) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires exact input/weight/bias inputs, matching rank-4 NHWC I8/U8 activation tensors, finite F32 [C] affine tensors, and positive num_groups/eps parameters.`);
  }

  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QGroupNorm node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QGroupNorm node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qgroupnorm', input, weight, bias, output,
    batch, height, width, channels, groups, inputScale, outputScale, epsilon,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}
export function qLayerNormByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}
export function qLayerNormAffineStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  return storage instanceof Float32Array && storage.byteLength === tensor.sizeBytes &&
    storage.length === tensor.sizeBytes / Float32Array.BYTES_PER_ELEMENT;
}
export function qLayerNormStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the source-storage contract before `_alloc` gives every tensor a
// separate WASM heap view. This keeps direct graph aliases and raw ArrayBuffer
// affine storage from being hidden by the copy into linear WASM memory.
// Preserve the source-storage contract before `_alloc` gives every tensor a
// separate WASM heap view. This keeps direct graph aliases and raw ArrayBuffer
// affine storage from being hidden by the copy into linear WASM memory.
export function preflightPortableQLayerNormStorage(node) {
  if (node.opType !== 'QLayerNorm') return;
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qLayerNormByteStorageIsCanonical(input) || !qLayerNormByteStorageIsCanonical(output) ||
      !qLayerNormAffineStorageIsCanonical(weight) || !qLayerNormAffineStorageIsCanonical(bias) ||
      qLayerNormStorageRangesOverlap(input, output) ||
      qLayerNormStorageRangesOverlap(weight, output) ||
      qLayerNormStorageRangesOverlap(bias, output)) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires canonical typed input/output storage, finite F32 [D] affine storage, and output storage distinct from every input.`);
  }
}
export function portableQLayerNormDescriptor(node, affineValues = (tensor) => tensor?.buffer) {
  if (node.opType !== 'QLayerNorm') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const weight = node.inputs?.weight;
  const bias = node.inputs?.bias;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreObject = params == null || (typeof params === 'object' && !Array.isArray(params));
  const parameterNames = paramsAreObject ? Object.keys(params || {}) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['eps', 'd_model'].includes(name));
  const epsilonSource = params?.eps ?? 1e-5;
  const epsilon = Math.fround(epsilonSource);
  const dModel = input?.shape?.at(-1);
  const validAffine = (tensor) => {
    const values = affineValues(tensor);
    return portableF32Tensor(tensor) && sameShape(tensor.shape, [dModel]) &&
      tensor.sizeBytes === dModel * Float32Array.BYTES_PER_ELEMENT &&
      values instanceof Float32Array && values.length === dModel &&
      (tensor.isWeight !== true || values.every(Number.isFinite));
  };

  if (inputNames.length !== 3 || inputNames[0] !== 'bias' || inputNames[1] !== 'input' ||
      inputNames[2] !== 'weight' || outputNames.length !== 1 || !input || !weight || !bias || !output ||
      input === output || weight === output || bias === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || inputElements == null || outputElements !== inputElements ||
      input.shape.length < 1 || !sameShape(input.shape, output.shape) || !allowedParameters ||
      !Number.isInteger(dModel) || dModel <= 0 ||
      (params?.d_model != null && (!Number.isInteger(params.d_model) || params.d_model !== dModel)) ||
      typeof epsilonSource !== 'number' || !Number.isFinite(epsilonSource) ||
      !Number.isFinite(epsilon) || epsilon <= 0 || !validAffine(weight) || !validAffine(bias)) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires exact input/weight/bias inputs, matching rank-at-least-1 I8/U8 activation tensors, finite F32 [D] affine tensors, and positive eps with optional d_model matching D.`);
  }

  const rows = inputElements / dModel;
  if (!Number.isInteger(rows) || rows <= 0 || rows > WASM_PORTABLE_MAX_U32 ||
      inputElements !== rows * dModel) {
    throw new Error(`WASM QLayerNorm node ${node.id} has an invalid contiguous [...,D] row layout.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QLayerNorm node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0) {
    throw new Error(`WASM QLayerNorm node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qlayernorm', input, weight, bias, output, rows, dModel, inputScale, outputScale, epsilon,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}
export function qSDPAByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}
export function qSDPAInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}
export function qSDPAStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Keep source alias checks before `_alloc` gives every tensor a fresh linear
// WASM view. Without this, overlapping graph views would be hidden by copies
// before qsdpa_i8u8 can preserve its no-partial-write boundary.
// Keep source alias checks before `_alloc` gives every tensor a fresh linear
// WASM view. Without this, overlapping graph views would be hidden by copies
// before qsdpa_i8u8 can preserve its no-partial-write boundary.
export function preflightPortableQSDPAStorage(node) {
  if (node.opType !== 'QSDPA') return;
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const mask = node.inputs?.mask || null;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  if (!qSDPAByteStorageIsCanonical(q) || !qSDPAByteStorageIsCanonical(k) ||
      !qSDPAByteStorageIsCanonical(v) || !qSDPAByteStorageIsCanonical(output) ||
      (mask && !qSDPAInt32StorageIsCanonical(mask)) || qSDPAStorageRangesOverlap(q, output) ||
      qSDPAStorageRangesOverlap(k, output) || qSDPAStorageRangesOverlap(v, output) ||
      (mask && qSDPAStorageRangesOverlap(mask, output))) {
    throw new Error(`WASM QSDPA node ${node.id} requires canonical typed q/k/v/output storage, optional I32 mask storage, and output storage distinct from every input.`);
  }
}
export function portableQSDPADescriptor(node) {
  if (node.opType !== 'QSDPA') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const q = node.inputs?.q;
  const k = node.inputs?.k;
  const v = node.inputs?.v;
  const hasMask = Object.prototype.hasOwnProperty.call(node.inputs || {}, 'mask');
  const mask = node.inputs?.mask;
  const output = node.outputs?.out || Object.values(node.outputs || {})[0];
  const qElements = portableTensorElements(q);
  const kElements = portableTensorElements(k);
  const vElements = portableTensorElements(v);
  const outputElements = portableTensorElements(output);
  const maskElements = hasMask ? portableTensorElements(mask) : 0;
  const params = node.params;
  const paramsAreObject = params != null && typeof params === 'object' && !Array.isArray(params);
  const parameterNames = paramsAreObject ? Object.keys(params) : [];
  const allowedParameters = paramsAreObject && parameterNames.every((name) =>
    ['heads', 'causal', 'scale'].includes(name));
  const rank = q?.shape?.length;
  const batch = rank === 2 ? 1 : q?.shape?.[0];
  const seqQ = q?.shape?.[rank - 2];
  const seqKV = k?.shape?.[rank - 2];
  const dModel = q?.shape?.[rank - 1];
  const heads = params?.heads;
  const headDim = dModel / heads;
  const scaleSource = params?.scale ?? 1 / Math.sqrt(headDim);
  const attentionScale = Math.fround(scaleSource);
  const exactInputs = (inputNames.length === 3 && inputNames[0] === 'k' && inputNames[1] === 'q' &&
    inputNames[2] === 'v') || (inputNames.length === 4 && inputNames[0] === 'k' &&
    inputNames[1] === 'mask' && inputNames[2] === 'q' && inputNames[3] === 'v');
  if (!exactInputs || outputNames.length !== 1 || !q || !k || !v || !output ||
      output === q || output === k || output === v || output === mask ||
      !portableByteQuantizedTensor(q) || !portableByteQuantizedTensor(k) ||
      !portableByteQuantizedTensor(v) || !portableByteQuantizedTensor(output) ||
      qElements == null || kElements == null || vElements == null || outputElements == null ||
      (hasMask && (mask?.dtype !== 'int32' || maskElements == null)) ||
      (rank !== 2 && rank !== 3) || k.shape.length !== rank || v.shape.length !== rank ||
      output.shape.length !== rank || !sameShape(q.shape, output.shape) || !allowedParameters ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'heads') ||
      !Number.isInteger(heads) || heads <= 0 ||
      !Object.prototype.hasOwnProperty.call(params || {}, 'causal') || typeof params?.causal !== 'boolean' ||
      !Number.isInteger(headDim) || headDim <= 0 || dModel % 4 !== 0 || headDim % 4 !== 0 ||
      headDim > 64 || typeof scaleSource !== 'number' || !Number.isFinite(scaleSource) ||
      scaleSource <= 0 || !Number.isFinite(attentionScale) || attentionScale <= 0) {
    throw new Error(`WASM QSDPA node ${node.id} requires exact q/k/v and optional I32 mask inputs, rank-2/3 I8/U8 tensors, D/head dimensions divisible by 4 with head_dim <= 64, and explicit heads/causal.`);
  }

  const kBatch = rank === 2 ? 1 : k.shape[0];
  const vBatch = rank === 2 ? 1 : v.shape[0];
  if (kBatch !== batch || vBatch !== batch || k.shape[rank - 1] !== dModel ||
      v.shape[rank - 1] !== dModel || v.shape[rank - 2] !== seqKV ||
      qElements !== batch * seqQ * dModel || kElements !== batch * seqKV * dModel ||
      vElements !== batch * seqKV * dModel || outputElements !== qElements) {
    throw new Error(`WASM QSDPA node ${node.id} has incompatible Q/K/V/output dimensions.`);
  }

  let maskMode = 0;
  if (hasMask) {
    if (mask.shape.length === 1 && mask.shape[0] === seqKV) maskMode = 1;
    // Preserve established B,K precedence if B and Q have the same value.
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === batch) maskMode = 2;
    else if (mask.shape.length === 2 && mask.shape[1] === seqKV && mask.shape[0] === seqQ) maskMode = 3;
    else if (mask.shape.length === 3 && mask.shape[0] === batch && mask.shape[1] === seqQ &&
        mask.shape[2] === seqKV) maskMode = 4;
    else throw new Error(`WASM QSDPA node ${node.id} mask requires I32 [K], [B,K], [Q,K], or [B,Q,K] storage.`);
  }

  const qQuantization = immutableQuantizationDescriptor(q);
  const kQuantization = immutableQuantizationDescriptor(k);
  const vQuantization = immutableQuantizationDescriptor(v);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!qQuantization || !kQuantization || !vQuantization || !outputQuantization ||
      qQuantization.scheme !== 'per_tensor' || kQuantization.scheme !== 'per_tensor' ||
      vQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM QSDPA node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const qScale = Math.fround(qQuantization.scale);
  const kScale = Math.fround(kQuantization.scale);
  const vScale = Math.fround(vQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const scoreMultiplier = Math.fround(Math.fround(qScale * kScale) * attentionScale);
  const maximumCenteredMagnitude = (tensor, descriptor) => {
    const minimum = tensor.dtype === 'int8' ? -128 : 0;
    const maximum = tensor.dtype === 'int8' ? 127 : 255;
    return Math.max(Math.abs(minimum - descriptor.zero_point), Math.abs(maximum - descriptor.zero_point));
  };
  const maximumRawDot = headDim * maximumCenteredMagnitude(q, qQuantization) *
    maximumCenteredMagnitude(k, kQuantization);
  const maximumScore = Math.fround(Math.fround(maximumRawDot) * scoreMultiplier);
  if (![qScale, kScale, vScale, outputScale, scoreMultiplier].every((value) =>
    Number.isFinite(value) && value > 0) || !Number.isFinite(maximumScore)) {
    throw new Error(`WASM QSDPA node ${node.id} requires scales and score range representable as finite positive F32.`);
  }
  return {
    kind: 'qsdpa', q, k, v, mask: hasMask ? mask : null, output, batch, seqQ, seqKV, dModel, heads,
    headDim, maskMode, causal: params.causal ? 1 : 0, qScale, kScale, vScale, outputScale,
    attentionScale, qZeroPoint: qQuantization.zero_point, kZeroPoint: kQuantization.zero_point,
    vZeroPoint: vQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    qDtype: wasmDtypeCode(q.dtype), kDtype: wasmDtypeCode(k.dtype),
    vDtype: wasmDtypeCode(v.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}
export function qArgMaxByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}
export function qArgMaxInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}
export function qArgMaxStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the native kernel's no-partial-write contract before per-tensor
// WASM allocation turns overlapping source views into independent copies.
// Preserve the native kernel's no-partial-write contract before per-tensor
// WASM allocation turns overlapping source views into independent copies.
export function preflightPortableQArgMaxStorage(node) {
  if (node.opType !== 'QArgMax') return;
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  if (!qArgMaxByteStorageIsCanonical(input) || !qArgMaxInt32StorageIsCanonical(output) ||
      qArgMaxStorageRangesOverlap(input, output)) {
    throw new Error(`WASM QArgMax node ${node.id} requires canonical I8/U8 input and I32 output storage distinct from input storage.`);
  }
}
export function portableQArgMaxDescriptor(node) {
  if (node.opType !== 'QArgMax') return null;
  const inputNames = Object.keys(node.inputs || {});
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const output = node.outputs?.out;
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 1 && Object.prototype.hasOwnProperty.call(params, 'axis') &&
    Number.isInteger(params.axis);
  const rank = input?.shape?.length;
  let axis = params?.axis;
  if (axis < 0) axis += rank;
  const outer = Number.isInteger(axis) ? portableElementCount(input?.shape?.slice(0, axis)) : null;
  const inner = Number.isInteger(axis) ? portableElementCount(input?.shape?.slice(axis + 1)) : null;
  const axisSize = input?.shape?.[axis];
  const expectedOutputShape = Number.isInteger(axis) && Array.isArray(input?.shape)
    ? [...input.shape.slice(0, axis), ...input.shape.slice(axis + 1)] : null;
  const inputQuantization = immutableQuantizationDescriptor(input);
  const inputScale = Math.fround(inputQuantization?.scale);
  if (inputNames.length !== 1 || inputNames[0] !== 'input' || outputNames.length !== 1 ||
      outputNames[0] !== 'out' || !input || !output || input === output ||
      !portableByteQuantizedTensor(input) || inputQuantization?.scheme !== 'per_tensor' ||
      !Number.isFinite(inputScale) || inputScale <= 0 || output.dtype !== 'int32' ||
      output.quantization != null || inputElements == null || outputElements == null ||
      !Number.isInteger(rank) || rank < 2 || rank > WASM_PORTABLE_MAX_RANK || !paramsAreExact ||
      !Number.isInteger(axis) || axis < 0 || axis >= rank || !Number.isInteger(axisSize) ||
      axisSize <= 0 || axisSize > 0x7fffffff || outer == null || inner == null || outer <= 0 || inner <= 0 ||
      inputElements !== outer * axisSize * inner || outputElements !== outer * inner ||
      !sameShape(output.shape, expectedOutputShape) || !qArgMaxByteStorageIsCanonical(input) ||
      !qArgMaxInt32StorageIsCanonical(output) || qArgMaxStorageRangesOverlap(input, output)) {
    throw new Error(`WASM QArgMax node ${node.id} requires exactly { input } -> { out }, immutable per-tensor rank-2..8 I8/U8 input, exact { axis: integer } parameters, and an unquantized I32 output with the axis removed.`);
  }
  return {
    kind: 'qargmax', input, output, axis, outer, axisSize, inner, outputElements,
    inputDtype: wasmDtypeCode(input.dtype),
  };
}
export function qMaskedMeanByteStorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  const storage = tensor.buffer;
  const matches = tensor.dtype === 'int8'
    ? storage instanceof Int8Array
    : tensor.dtype === 'uint8' && (storage instanceof Uint8Array || storage instanceof Uint8ClampedArray);
  return matches && storage.byteLength === tensor.sizeBytes;
}
export function qMaskedMeanInt32StorageIsCanonical(tensor) {
  if (!tensor || tensor.buffer == null) return true;
  return tensor.buffer instanceof Int32Array && tensor.buffer.byteLength === tensor.sizeBytes;
}
export function qMaskedMeanStorageRangesOverlap(left, right) {
  const leftStorage = left?.buffer;
  const rightStorage = right?.buffer;
  if (!ArrayBuffer.isView(leftStorage) || leftStorage instanceof DataView ||
      !ArrayBuffer.isView(rightStorage) || rightStorage instanceof DataView ||
      leftStorage.buffer !== rightStorage.buffer) return false;
  const leftStart = leftStorage.byteOffset;
  const leftEnd = leftStart + leftStorage.byteLength;
  const rightStart = rightStorage.byteOffset;
  const rightEnd = rightStart + rightStorage.byteLength;
  return leftStart < rightEnd && rightStart < leftEnd;
}

// Preserve the native kernel's typed-storage and no-overlap contract before
// `_alloc` turns graph tensors into disjoint WASM heap views.
// Preserve the native kernel's typed-storage and no-overlap contract before
// `_alloc` turns graph tensors into disjoint WASM heap views.
export function preflightPortableQMaskedMeanStorage(node) {
  if (node.opType !== 'QMaskedMean') return;
  const input = node.inputs?.input;
  const mask = node.inputs?.mask;
  const output = node.outputs?.out;
  if (!qMaskedMeanByteStorageIsCanonical(input) || !qMaskedMeanInt32StorageIsCanonical(mask) ||
      !qMaskedMeanByteStorageIsCanonical(output) || qMaskedMeanStorageRangesOverlap(input, output) ||
      qMaskedMeanStorageRangesOverlap(mask, output)) {
    throw new Error(`WASM QMaskedMean node ${node.id} requires canonical I8/U8 input/output and I32 mask storage, with output storage distinct from input and mask.`);
  }
}
export function qMaskedMeanMaximumCenteredMagnitude(dtype, zeroPoint) {
  const minimum = dtype === 'int8' ? -128 : 0;
  const maximum = dtype === 'int8' ? 127 : 255;
  return Math.max(Math.abs(minimum - zeroPoint), Math.abs(maximum - zeroPoint));
}
export function portableQMaskedMeanDescriptor(node) {
  if (node.opType !== 'QMaskedMean') return null;
  const inputNames = Object.keys(node.inputs || {}).sort();
  const outputNames = Object.keys(node.outputs || {});
  const input = node.inputs?.input;
  const mask = node.inputs?.mask;
  const output = node.outputs?.out;
  const inputElements = portableTensorElements(input);
  const maskElements = portableTensorElements(mask);
  const outputElements = portableTensorElements(output);
  const params = node.params;
  const paramsAreExact = params != null && typeof params === 'object' && !Array.isArray(params) &&
    Object.keys(params).length === 0;
  const [batch, sequence, width] = input?.shape || [];
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const inputScale = Math.fround(inputQuantization?.scale);
  const outputScale = Math.fround(outputQuantization?.scale);
  const multiplier = Math.fround(inputScale / outputScale);

  if (inputNames.length !== 2 || inputNames[0] !== 'input' || inputNames[1] !== 'mask' ||
      outputNames.length !== 1 || outputNames[0] !== 'out' || !input || !mask || !output ||
      input === output || mask === output || !portableByteQuantizedTensor(input) ||
      !portableByteQuantizedTensor(output) || mask.dtype !== 'int32' || mask.quantization != null ||
      inputElements == null || maskElements == null || outputElements == null ||
      input.shape.length !== 3 || mask.shape.length !== 2 || output.shape.length !== 2 ||
      !paramsAreExact || !Number.isInteger(batch) || !Number.isInteger(sequence) ||
      !Number.isInteger(width) || batch <= 0 || sequence <= 0 || width <= 0 ||
      batch > WASM_PORTABLE_MAX_U32 || sequence > WASM_PORTABLE_MAX_U32 || width > WASM_PORTABLE_MAX_U32 ||
      mask.shape[0] !== batch || mask.shape[1] !== sequence ||
      output.shape[0] !== batch || output.shape[1] !== width ||
      inputElements !== batch * sequence * width || maskElements !== batch * sequence ||
      outputElements !== batch * width || inputQuantization?.scheme !== 'per_tensor' ||
      outputQuantization?.scheme !== 'per_tensor' || !Number.isFinite(inputScale) || inputScale <= 0 ||
      !Number.isFinite(outputScale) || outputScale <= 0 || !Number.isFinite(multiplier) ||
      multiplier <= 0 || !qMaskedMeanByteStorageIsCanonical(input) ||
      !qMaskedMeanInt32StorageIsCanonical(mask) || !qMaskedMeanByteStorageIsCanonical(output) ||
      qMaskedMeanStorageRangesOverlap(input, output) || qMaskedMeanStorageRangesOverlap(mask, output)) {
    throw new Error(`WASM QMaskedMean node ${node.id} requires exactly { input, mask } -> { out }, immutable per-tensor I8/U8 [B,S,D] input/output [B,D], unquantized I32 [B,S] mask, and no parameters.`);
  }

  const maximumCenteredSum = sequence * qMaskedMeanMaximumCenteredMagnitude(
    input.dtype, inputQuantization.zero_point,
  );
  if (!Number.isSafeInteger(maximumCenteredSum) || maximumCenteredSum > 0x7fffffff) {
    throw new Error(`WASM QMaskedMean node ${node.id} sequence is too large for an exact I32 centered sum.`);
  }
  return {
    kind: 'qmaskedmean', input, mask, output, batch, sequence, width,
    inputScale, inputZeroPoint: inputQuantization.zero_point,
    outputScale, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
  };
}
export function portableRequantizeLinearDescriptor(node) {
  if (node.opType !== 'RequantizeLinear') return null;
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = portableOutput(node);
  const inputEntries = Object.entries(node.inputs || {});
  const inputElements = portableTensorElements(input);
  const outputElements = portableTensorElements(output);
  if (inputEntries.length !== 1 || !portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(output) ||
      inputElements == null || outputElements !== inputElements) {
    throw new Error(`WASM RequantizeLinear node ${node.id} requires one metadata-only I8/U8 input and an equal-size output.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  if (!inputQuantization || !outputQuantization || inputQuantization.scheme !== 'per_tensor' ||
      outputQuantization.scheme !== 'per_tensor') {
    throw new Error(`WASM RequantizeLinear node ${node.id} requires immutable per-tensor I8/U8 quantization metadata.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const multiplier = Math.fround(inputScale / outputScale);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(`WASM RequantizeLinear node ${node.id} has a non-representable positive scale ratio.`);
  }
  return {
    kind: 'requantizeLinear', input, output, elements: inputElements,
    inputDtype: wasmDtypeCode(input.dtype), outputDtype: wasmDtypeCode(output.dtype),
    inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
  };
}
export function portableQConv2DDescriptor(node) {
  if (node.opType !== 'QConv2D') return null;
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias || null;
  const output = portableOutput(node);
  const inputElements = portableTensorElements(input);
  const weightElements = portableTensorElements(weight);
  const outputElements = portableTensorElements(output);
  const biasElements = bias ? portableTensorElements(bias) : 0;
  if (!portableByteQuantizedTensor(input) || !portableByteQuantizedTensor(weight) ||
      !portableByteQuantizedTensor(output) || inputElements == null || weightElements == null ||
      outputElements == null || input.shape.length !== 4 || weight.shape.length !== 4 ||
      output.shape.length !== 4 || (bias && (bias.dtype !== 'int32' || biasElements == null)) ||
      (node.params?.data_layout && node.params.data_layout !== 'NHWC') ||
      (node.params?.weight_layout && node.params.weight_layout !== 'OHWI')) {
    throw new Error(`WASM QConv2D node ${node.id} requires canonical NHWC I8/U8 activations and OHWI I8/U8 weights.`);
  }
  const inputQuantization = immutableQuantizationDescriptor(input);
  const outputQuantization = immutableQuantizationDescriptor(output);
  const weightQuantization = immutableQuantizationDescriptor(weight);
  if (!inputQuantization || !outputQuantization || !weightQuantization ||
      inputQuantization.scheme !== 'per_tensor' || outputQuantization.scheme !== 'per_tensor' ||
      weightQuantization.scheme !== 'per_axis') {
    throw new Error(`WASM QConv2D node ${node.id} requires immutable per-tensor activation and per-axis weight metadata.`);
  }
  const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
  const [outputChannels, kernelHeight, kernelWidth, inputPerGroup] = weight.shape;
  const [outputBatch, outputHeight, outputWidth, outputChannelsFromOutput] = output.shape;
  const groups = node.params?.groups ?? 1;
  const [strideY, strideX] = portablePairParameter(node, 'stride', 1);
  const [dilationY, dilationX] = portablePairParameter(node, 'dilation', 1);
  const [paddingY, paddingX] = portablePairParameter(node, 'padding', 0, true);
  const pads = node.params?.pads || [paddingY, paddingX, paddingY, paddingX];
  const relu = node.params?.relu ?? 0;
  if (!Number.isInteger(groups) || groups <= 0 || groups > WASM_PORTABLE_MAX_U32 ||
      inputChannels !== inputPerGroup * groups || outputChannels !== outputChannelsFromOutput ||
      outputBatch !== batch || outputChannels % groups !== 0 || !Array.isArray(pads) ||
      pads.length !== 4 || pads.some((value) => !Number.isInteger(value) || value < 0 ||
        value > WASM_PORTABLE_MAX_U32) || !Number.isInteger(relu) || relu < 0 || relu > 2) {
    throw new Error(`WASM QConv2D node ${node.id} has incompatible grouped-convolution parameters.`);
  }
  const effectiveHeight = dilationY * (kernelHeight - 1) + 1;
  const effectiveWidth = dilationX * (kernelWidth - 1) + 1;
  const paddedHeight = inputHeight + pads[0] + pads[2];
  const paddedWidth = inputWidth + pads[1] + pads[3];
  if (![effectiveHeight, effectiveWidth, paddedHeight, paddedWidth].every(Number.isSafeInteger) ||
      paddedHeight < effectiveHeight || paddedWidth < effectiveWidth) {
    throw new Error(`WASM QConv2D node ${node.id} has non-representable stride, padding, or dilation geometry.`);
  }
  const expectedHeight = Math.floor((paddedHeight - effectiveHeight) / strideY) + 1;
  const expectedWidth = Math.floor((paddedWidth - effectiveWidth) / strideX) + 1;
  if (expectedHeight !== outputHeight || expectedWidth !== outputWidth ||
      weightElements !== outputChannels * kernelHeight * kernelWidth * inputPerGroup ||
      inputElements !== batch * inputHeight * inputWidth * inputChannels ||
      outputElements !== batch * outputHeight * outputWidth * outputChannels) {
    throw new Error(`WASM QConv2D node ${node.id} has an output shape incompatible with its canonical descriptor.`);
  }
  if (bias && (!sameShape(bias.shape, [outputChannels]) || biasElements !== outputChannels)) {
    throw new Error(`WASM QConv2D node ${node.id} bias must be an I32 vector with one value per output channel.`);
  }
  if (weightQuantization.axis !== 0 || weightQuantization.scales.length !== outputChannels ||
      weightQuantization.zero_points.length !== outputChannels) {
    throw new Error(`WASM QConv2D node ${node.id} requires per_axis weight quantization along output-channel axis 0.`);
  }
  const inputScale = Math.fround(inputQuantization.scale);
  const outputScale = Math.fround(outputQuantization.scale);
  const weightScales = Float32Array.from(weightQuantization.scales, Math.fround);
  if (!Number.isFinite(inputScale) || inputScale <= 0 || !Number.isFinite(outputScale) ||
      outputScale <= 0 || !weightScales.every((scale) => Number.isFinite(scale) && scale > 0)) {
    throw new Error(`WASM QConv2D node ${node.id} requires scales representable as positive F32.`);
  }
  return {
    kind: 'qconv2d', input, weight, bias, output,
    batch, inputHeight, inputWidth, inputChannels,
    outputHeight, outputWidth, outputChannels, kernelHeight, kernelWidth, inputPerGroup,
    strideY, strideX, dilationY, dilationX,
    paddingTop: pads[0], paddingLeft: pads[1], paddingBottom: pads[2], paddingRight: pads[3],
    groups, relu, inputScale, outputScale,
    inputZeroPoint: inputQuantization.zero_point, outputZeroPoint: outputQuantization.zero_point,
    inputDtype: wasmDtypeCode(input.dtype), weightDtype: wasmDtypeCode(weight.dtype),
    outputDtype: wasmDtypeCode(output.dtype),
    weightScales, weightZeroPoints: Int32Array.from(weightQuantization.zero_points),
  };
}
