function imageShape(tensor) {
  const shape = tensor?.shape;
  if (!Array.isArray(shape) || shape.length !== 4 || shape[0] !== 1 ||
      !shape.slice(1).every((dimension) => Number.isInteger(dimension) && dimension > 0) ||
      shape[3] !== 3) {
    throw new Error('EfficientDet image input must have NHWC shape [1, height, width, 3].');
  }
  return { height: shape[1], width: shape[2] };
}

function requireByteQuantization(tensor) {
  const descriptor = tensor?.quantization;
  const minimum = tensor.dtype === 'int8' ? -128 : 0;
  const maximum = tensor.dtype === 'int8' ? 127 : 255;
  if (descriptor?.scheme !== 'per_tensor' ||
      typeof descriptor.scale !== 'number' || !Number.isFinite(descriptor.scale) ||
      descriptor.scale <= 0 || !Number.isInteger(descriptor.zero_point) ||
      descriptor.zero_point < minimum || descriptor.zero_point > maximum) {
    throw new Error(`EfficientDet ${tensor.dtype} image input requires per-tensor quantization metadata.`);
  }
}

/**
 * Convert canvas RGBA bytes to the model's RGB input convention: zero-one F32,
 * physical U8 bytes, or signed I8 source levels centered on zero.
 */
export function efficientDetInputFromRgba(rgba, tensor) {
  const { height, width } = imageShape(tensor);
  if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
    throw new Error('EfficientDet image pixels must use U8 RGBA storage.');
  }
  const pixels = height * width;
  if (rgba.length !== pixels * 4) {
    throw new Error(`EfficientDet image has ${rgba.length} RGBA bytes; expected ${pixels * 4}.`);
  }

  let input;
  if (tensor.dtype === 'float32') {
    input = new Float32Array(pixels * 3);
  } else if (tensor.dtype === 'uint8') {
    requireByteQuantization(tensor);
    input = new Uint8Array(pixels * 3);
  } else if (tensor.dtype === 'int8') {
    requireByteQuantization(tensor);
    input = new Int8Array(pixels * 3);
  } else {
    throw new Error(`EfficientDet image input dtype '${String(tensor.dtype)}' is unsupported.`);
  }

  for (let source = 0, destination = 0; source < rgba.length; source += 4, destination += 3) {
    if (tensor.dtype === 'int8') {
      // Signed image packages use the same 256 source levels centered on zero.
      input[destination] = rgba[source] - 128;
      input[destination + 1] = rgba[source + 1] - 128;
      input[destination + 2] = rgba[source + 2] - 128;
    } else if (tensor.dtype === 'float32') {
      input[destination] = rgba[source] / 255;
      input[destination + 1] = rgba[source + 1] / 255;
      input[destination + 2] = rgba[source + 2] / 255;
    } else {
      // U8 input is already physical quantized storage. Applying the tensor
      // scale here would quantize the pixels a second time and saturate them.
      input[destination] = rgba[source];
      input[destination + 1] = rgba[source + 1];
      input[destination + 2] = rgba[source + 2];
    }
  }
  return input;
}
