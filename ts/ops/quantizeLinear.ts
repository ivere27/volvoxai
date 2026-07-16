// Canonical forward-only QuantizeLinear reference implementation.
//
// The portable W8A8 graph contract stores the quantized activation in its
// declared I8/U8 tensor, rather than in an F32 tensor with an implicit side
// channel.  Rounding is IEEE round-to-nearest, ties-to-even so the JS, WASM,
// and WebGPU implementations can agree exactly at quantization boundaries.

export function roundTiesToEven(value) {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function scalarElementCount(tensor) {
  return tensor?.shape?.reduce((product, dimension) => product * dimension, 1);
}

export function _cpuQuantizeLinear(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const scale = node.inputs.scale;
  const zeroPoint = node.inputs.zero_point || null;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const elements = input?.shape?.reduce((product, dimension) => product * dimension, 1);

  if (!input || !scale || !output || input.dtype !== 'float32' ||
      !['int8', 'uint8'].includes(output.dtype) || !Number.isSafeInteger(elements) ||
      elements <= 0 || input.buffer?.length !== elements || output.buffer?.length !== elements ||
      scale.dtype !== 'float32' || scalarElementCount(scale) !== 1 || scale.buffer?.length !== 1 ||
      output.quantization?.scheme !== 'per_tensor' ||
      (zeroPoint && (zeroPoint.dtype !== output.dtype || scalarElementCount(zeroPoint) !== 1 ||
        zeroPoint.buffer?.length !== 1))) {
    throw new Error(`QuantizeLinear node ${node.id} requires F32 input, scalar F32 scale, optional matching I8/U8 zero_point, and matching I8/U8 output with per_tensor quantization metadata.`);
  }

  const scaleValue = scale.buffer[0];
  if (!Number.isFinite(scaleValue) || scaleValue <= 0) {
    throw new Error(`QuantizeLinear node ${node.id} requires a finite positive scalar scale.`);
  }
  const minimum = output.dtype === 'int8' ? -128 : 0;
  const maximum = output.dtype === 'int8' ? 127 : 255;
  const zero = zeroPoint ? zeroPoint.buffer[0] : 0;
  if (!Number.isInteger(zero) || zero < minimum || zero > maximum) {
    throw new Error(`QuantizeLinear node ${node.id} zero_point is outside ${output.dtype}'s representable range.`);
  }
  if (Math.fround(output.quantization.scale) !== Math.fround(scaleValue) ||
      output.quantization.zero_point !== zero) {
    throw new Error(`QuantizeLinear node ${node.id} scale/zero_point inputs do not match its output quantization metadata.`);
  }

  for (let index = 0; index < elements; index++) {
    const value = input.buffer[index];
    let quantized;
    if (Number.isNaN(value)) {
      quantized = zero;
    } else {
      // fround makes the reference use the same F32 intermediate arithmetic as
      // the portable C and WGSL kernels, including infinities at extreme input
      // values. Clamp before converting to an integer so no implementation
      // depends on language-specific out-of-range casts.
      const transformed = Math.fround(Math.fround(value / scaleValue) + zero);
      if (transformed <= minimum) quantized = minimum;
      else if (transformed >= maximum) quantized = maximum;
      else quantized = roundTiesToEven(transformed);
    }
    output.buffer[index] = quantized;
  }
}
