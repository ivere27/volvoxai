/* Shape and dtype primitives shared by every portable WASM descriptor
 * builder.  Split out of WasmEngine.ts so the quantized descriptors can
 * depend on them without a cycle back through the engine. */
import { DataType } from '../generated/volvoxaiEnums.js';
import { Tensor } from '../core/Tensor.js';
import { normalizeSpatialPair } from '../ops/spatialParameters.js';

export function wasmDtypeCode(dtype) {
  if (dtype === 'float32') return DataType.F32;
  if (dtype === 'int32') return DataType.I32;
  if (dtype === 'int8') return DataType.I8;
  if (dtype === 'uint8') return DataType.U8;
  throw new Error(`Unsupported WASM kernel dtype '${dtype}'.`);
}
export const WASM_PORTABLE_MAX_RANK = 8;
export const WASM_PORTABLE_MAX_U32 = 0xffffffff;
/* wasm32 addresses 4 GiB in 64 KiB pages, so the heap can never exceed this. */
export function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}
export function portableElementCount(shape) {
  if (!Array.isArray(shape)) return null;
  let elements = 1;
  for (const dimension of shape) {
    if (!Number.isInteger(dimension) || dimension <= 0 || dimension > WASM_PORTABLE_MAX_U32 ||
        elements > Math.floor(WASM_PORTABLE_MAX_U32 / dimension)) return null;
    elements *= dimension;
  }
  return elements;
}
export function portableTensorElements(tensor) {
  if (!tensor) return null;
  const elements = portableElementCount(tensor.shape);
  if (elements == null) return null;
  let bytes;
  try {
    bytes = Tensor.dtypeBytes(tensor.dtype);
  } catch {
    return null;
  }
  return tensor.sizeBytes === elements * bytes ? elements : null;
}
export function portableOutput(node) {
  return node.outputs?.out || Object.values(node.outputs || {})[0];
}
export function portableF32Tensor(tensor) {
  return !!tensor && tensor.dtype === 'float32' && portableTensorElements(tensor) != null;
}
export function portableByteQuantizedTensor(tensor) {
  return !!tensor && (tensor.dtype === 'int8' || tensor.dtype === 'uint8') &&
    portableTensorElements(tensor) != null;
}
export function portablePairParameter(node, name, fallback, allowZero = false) {
  const value = node.params?.[name];
  let pair;
  if (value == null) pair = [fallback, fallback];
  else if (Array.isArray(value)) {
    if (value.length < 1 || value.length > 2) {
      throw new Error(`WASM ${node.opType} node ${node.id} ${name} must be a scalar or one/two-element array.`);
    }
    pair = [value[0], value[1] ?? value[0]];
  } else pair = [value, value];
  const minimum = allowZero ? 0 : 1;
  if (pair.some((dimension) => !Number.isInteger(dimension) || dimension < minimum ||
      dimension > WASM_PORTABLE_MAX_U32)) {
    throw new Error(`WASM ${node.opType} node ${node.id} ${name} must contain ${allowZero ? 'non-negative' : 'positive'} U32 values.`);
  }
  return pair;
}
