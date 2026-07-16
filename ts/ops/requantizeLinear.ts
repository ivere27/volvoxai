// Canonical typed requantization for an integer model boundary.  This is
// deliberately distinct from QuantizeLinear: both input and output already
// own immutable mappings to real values, so no mutable scale tensors are
// permitted to silently change graph semantics at execution time.

import { roundTiesToEven } from './quantizeLinear.js';

function elements(tensor) {
  return tensor?.shape?.reduce((product, dimension) => product * dimension, 1);
}

function perTensorQuantization(tensor, label) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) ||
      tensor.quantization?.scheme !== 'per_tensor') {
    throw new Error(`${label} requires I8/U8 storage with per_tensor quantization metadata.`);
  }
  return tensor.quantization;
}

function range(dtype) {
  return dtype === 'int8' ? [-128, 127] : [0, 255];
}

// RequantizeLinear changes only the byte representation of a real-valued
// activation. It is the explicit U8<->I8 (or differing-scale) bridge used by
// quantized model inputs and by exporters that must reconcile TFLite edges.
export function _cpuRequantizeLinear(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const inputElements = elements(input);
  const outputElements = elements(output);
  if (!input || !output || !Number.isSafeInteger(inputElements) || inputElements <= 0 ||
      inputElements !== outputElements || input.buffer?.length !== inputElements ||
      output.buffer?.length !== outputElements) {
    throw new Error(`RequantizeLinear node ${node.id} requires equal-size typed input and output tensors.`);
  }
  const inputQuantization = perTensorQuantization(input, `RequantizeLinear node ${node.id} input`);
  const outputQuantization = perTensorQuantization(output, `RequantizeLinear node ${node.id} output`);
  const [minimum, maximum] = range(output.dtype);
  const multiplier = Math.fround(inputQuantization.scale / outputQuantization.scale);
  if (!Number.isFinite(multiplier) || multiplier <= 0) {
    throw new Error(`RequantizeLinear node ${node.id} has a non-representable positive scale ratio.`);
  }
  for (let index = 0; index < inputElements; index++) {
    const centered = input.buffer[index] - inputQuantization.zero_point;
    const transformed = Math.fround(Math.fround(centered * multiplier) + outputQuantization.zero_point);
    let quantized;
    if (Number.isNaN(transformed)) quantized = outputQuantization.zero_point;
    else if (transformed <= minimum) quantized = minimum;
    else if (transformed >= maximum) quantized = maximum;
    else quantized = roundTiesToEven(transformed);
    output.buffer[index] = quantized;
  }
}
