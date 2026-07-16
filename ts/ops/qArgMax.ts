import { Tensor } from '../core/Tensor.js';
import type { PerTensorQuantization, TensorQuantization } from '../types.js';

const MAX_U32 = 0xffffffff;
const MAX_I32 = 0x7fffffff;

function nodeLabel(node) {
  return String(node?.id ?? '<unnamed>');
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function elementCount(shape, label, node) {
  if (!Array.isArray(shape) || shape.some((dimension) =>
    !Number.isInteger(dimension) || dimension <= 0)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} ${label} requires a valid shape.`);
  }
  const elements = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0 || elements > MAX_U32) {
    throw new Error(`QArgMax node ${nodeLabel(node)} ${label} has an unsupported element count.`);
  }
  return elements;
}

function matchesByteStorage(tensor) {
  if (tensor.dtype === 'int8') return tensor.buffer instanceof Int8Array;
  return tensor.dtype === 'uint8' &&
    (tensor.buffer instanceof Uint8Array || tensor.buffer instanceof Uint8ClampedArray);
}

function requireInput(tensor, node) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} input requires I8/U8 storage.`);
  }
  if (!Array.isArray(tensor.shape) || tensor.shape.length < 2 || tensor.shape.length > 8) {
    throw new Error(`QArgMax node ${nodeLabel(node)} input requires rank 2 through 8.`);
  }
  const elements = elementCount(tensor.shape, 'input', node);
  if (tensor.sizeBytes !== elements || !matchesByteStorage(tensor) ||
      tensor.buffer.byteLength !== elements || tensor.buffer.length !== elements) {
    throw new Error(`QArgMax node ${nodeLabel(node)} input has incompatible typed storage.`);
  }
  return elements;
}

function requireOutput(tensor, node) {
  if (!tensor || tensor.dtype !== 'int32') {
    throw new Error(`QArgMax node ${nodeLabel(node)} output requires unquantized I32 storage.`);
  }
  const elements = elementCount(tensor.shape, 'output', node);
  if (tensor.quantization != null || !(tensor.buffer instanceof Int32Array) ||
      tensor.sizeBytes !== elements * Int32Array.BYTES_PER_ELEMENT ||
      tensor.buffer.byteLength !== tensor.sizeBytes || tensor.buffer.length !== elements) {
    throw new Error(`QArgMax node ${nodeLabel(node)} output requires unquantized I32 typed storage.`);
  }
  return elements;
}

function requireImmutablePerTensorQuantization(tensor, node): PerTensorQuantization {
  if (!tensor?.quantization || !Object.isFrozen(tensor.quantization)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} input requires immutable per_tensor quantization metadata.`);
  }
  const descriptor = Tensor.normalizeQuantization(tensor.dtype, tensor.shape, tensor.quantization,
    `QArgMax node ${nodeLabel(node)} input quantization`) as TensorQuantization | null;
  if (!descriptor || descriptor.scheme !== 'per_tensor') {
    throw new Error(`QArgMax node ${nodeLabel(node)} input requires immutable per_tensor quantization metadata.`);
  }
  return descriptor;
}

function storageRangesOverlap(left, right) {
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

function canonicalInput(node) {
  if (!isRecord(node?.inputs) || Object.keys(node.inputs).length !== 1 || !node.inputs.input) {
    throw new Error(`QArgMax node ${nodeLabel(node)} requires exactly { input }.`);
  }
  return node.inputs.input;
}

function canonicalOutput(node) {
  if (!isRecord(node?.outputs) || Object.keys(node.outputs).length !== 1 || !node.outputs.out) {
    throw new Error(`QArgMax node ${nodeLabel(node)} requires exactly { out }.`);
  }
  return node.outputs.out;
}

function canonicalAxis(node, rank) {
  if (!isRecord(node?.params) || Object.keys(node.params).length !== 1 ||
      !Object.prototype.hasOwnProperty.call(node.params, 'axis') ||
      !Number.isInteger(node.params.axis)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} requires exactly { axis: integer } parameters.`);
  }
  let axis = node.params.axis;
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) {
    throw new Error(`QArgMax node ${nodeLabel(node)} axis ${node.params.axis} is outside rank ${rank}.`);
  }
  return axis;
}

function canonicalDescriptor(node) {
  const input = canonicalInput(node);
  const output = canonicalOutput(node);
  const inputElements = requireInput(input, node);
  const outputElements = requireOutput(output, node);
  const quantization = requireImmutablePerTensorQuantization(input, node);
  const rank = input.shape.length;
  const axis = canonicalAxis(node, rank);
  const outer = input.shape.slice(0, axis).reduce((product, dimension) => product * dimension, 1);
  const axisSize = input.shape[axis];
  const inner = input.shape.slice(axis + 1).reduce((product, dimension) => product * dimension, 1);
  const expectedOutputShape = [...input.shape.slice(0, axis), ...input.shape.slice(axis + 1)];
  const expectedOutputElements = outer * inner;
  if (!Number.isSafeInteger(outer) || !Number.isSafeInteger(inner) ||
      !Number.isSafeInteger(expectedOutputElements) || outer <= 0 || inner <= 0 ||
      outer > MAX_U32 || inner > MAX_U32 || axisSize > MAX_I32 ||
      expectedOutputElements > MAX_U32 || inputElements !== outer * axisSize * inner ||
      outputElements !== expectedOutputElements || !sameShape(output.shape, expectedOutputShape)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} has incompatible axis or output dimensions.`);
  }
  if (input === output || storageRangesOverlap(input, output)) {
    throw new Error(`QArgMax node ${nodeLabel(node)} requires output storage distinct from input storage.`);
  }
  return { input, output, quantization, axis, outer, axisSize, inner, outputElements };
}

// Canonical byte-domain ArgMax.  Positive per-tensor affine quantization
// preserves raw I8/U8 ordering, so no decoded F32 logits or intermediate
// activation tensor is needed.  Strict `>` deliberately retains the first
// index when values tie.  Every validation completes before output writes.
export function _cpuQArgMax(node) {
  const { input, output, outer, axisSize, inner } = canonicalDescriptor(node);
  let outputIndex = 0;
  for (let outerIndex = 0; outerIndex < outer; outerIndex++) {
    for (let innerIndex = 0; innerIndex < inner; innerIndex++) {
      const base = outerIndex * axisSize * inner + innerIndex;
      let best = input.buffer[base];
      let bestIndex = 0;
      for (let axisIndex = 1; axisIndex < axisSize; axisIndex++) {
        const value = input.buffer[base + axisIndex * inner];
        if (value > best) {
          best = value;
          bestIndex = axisIndex;
        }
      }
      output.buffer[outputIndex++] = bestIndex;
    }
  }
}
