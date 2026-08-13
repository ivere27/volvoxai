// Validation shared by byte-preserving shape operators.  These operators are
// mathematically valid on quantized storage only when the representation does
// not change: the dtype and immutable descriptor must travel unchanged.

function equalArrays(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

export function sameQuantizationDescriptor(left, right) {
  if (left == null || right == null) return left == null && right == null;
  if (left.scheme !== right.scheme) return false;
  if (left.scheme === 'per_tensor') {
    return left.scale === right.scale && left.zero_point === right.zero_point;
  }
  return left.scheme === 'per_axis' && left.axis === right.axis &&
    equalArrays(left.scales, right.scales) && equalArrays(left.zero_points, right.zero_points);
}

export function assertRawQuantizedShapeTensors(node, tensors, operation = node.opType) {
  const usesByteStorage = tensors.some((tensor) => tensor &&
    (tensor.dtype === 'int8' || tensor.dtype === 'uint8'));
  if (!usesByteStorage) return false;
  const reference = tensors[0];
  if (!reference || !['int8', 'uint8'].includes(reference.dtype) || !reference.quantization ||
      tensors.some((tensor) => !tensor || tensor.dtype !== reference.dtype ||
        !sameQuantizationDescriptor(reference.quantization, tensor.quantization))) {
    throw new Error(`${operation} node ${node.id || '<unnamed>'} requires one I8/U8 dtype and identical immutable quantization descriptors.`);
  }
  return true;
}
