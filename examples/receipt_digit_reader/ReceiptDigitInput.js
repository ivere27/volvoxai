/*
 * Receipt digit reader host-input utilities.
 *
 * This module owns exactly one contract: turning a receipt image into the
 * grayscale, bilinearly resized, symmetric-normalized F32 plane the graph
 * declares. It knows nothing about packages, compilation, sessions, or
 * decoding, so calibration and inference can share one preprocessing
 * definition instead of drifting apart.
 *
 * The producer specifies the transform as:
 *
 *   gray -> bilinear resize to (width, height) -> x = (gray / 255 - 0.5) / 0.5
 */

function fail(message) {
  throw new Error(`[ReceiptDigitInput] ${message}`);
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/** ITU-R BT.601 luma, matching PIL's `convert('L')`. */
export function luminance(red, green, blue) {
  return (red * 299 + green * 587 + blue * 114) / 1000;
}

/**
 * Reduce packed 1/3/4-channel pixels to one grayscale plane.
 *
 * An alpha channel is ignored rather than composited: the producer's PIL
 * pipeline ignores it too, and compositing here would change the values the
 * model was calibrated and trained on.
 */
export function toGrayscalePlane(image) {
  if (!isRecord(image) || !ArrayBuffer.isView(image.data) || image.data instanceof DataView
      || !Number.isInteger(image.width) || image.width <= 0
      || !Number.isInteger(image.height) || image.height <= 0) {
    fail('image must be { data, width, height } with a typed-array payload.');
  }
  const pixels = image.width * image.height;
  const channels = image.channels ?? image.data.length / pixels;
  if (!Number.isInteger(channels) || ![1, 3, 4].includes(channels)
      || image.data.length !== pixels * channels) {
    fail('image data must use exactly 1, 3, or 4 tightly packed channels.');
  }
  if (channels === 1) return Float32Array.from(image.data);
  const plane = new Float32Array(pixels);
  for (let index = 0; index < pixels; index++) {
    const base = index * channels;
    plane[index] = luminance(
      image.data[base], image.data[base + 1], image.data[base + 2],
    );
  }
  return plane;
}

/**
 * Bilinear resize of one grayscale plane, using PIL's half-pixel centers.
 *
 * PIL maps destination centers to source centers rather than corners; using
 * the corner convention instead shifts the sampled grid by up to half a pixel
 * and visibly moves thin digit strokes.
 */
export function resizeBilinear(plane, sourceWidth, sourceHeight, width, height) {
  if (!(plane instanceof Float32Array) || plane.length !== sourceWidth * sourceHeight) {
    fail('resize needs one F32 plane matching its declared source extents.');
  }
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    fail('resize needs positive integer target extents.');
  }
  if (sourceWidth === width && sourceHeight === height) return Float32Array.from(plane);
  const output = new Float32Array(width * height);
  const scaleX = sourceWidth / width;
  const scaleY = sourceHeight / height;
  for (let row = 0; row < height; row++) {
    const sourceY = (row + 0.5) * scaleY - 0.5;
    const topIndex = Math.floor(sourceY);
    const weightY = sourceY - topIndex;
    const top = Math.min(Math.max(topIndex, 0), sourceHeight - 1);
    const bottom = Math.min(Math.max(topIndex + 1, 0), sourceHeight - 1);
    for (let column = 0; column < width; column++) {
      const sourceX = (column + 0.5) * scaleX - 0.5;
      const leftIndex = Math.floor(sourceX);
      const weightX = sourceX - leftIndex;
      const left = Math.min(Math.max(leftIndex, 0), sourceWidth - 1);
      const right = Math.min(Math.max(leftIndex + 1, 0), sourceWidth - 1);
      const topLeft = plane[top * sourceWidth + left];
      const topRight = plane[top * sourceWidth + right];
      const bottomLeft = plane[bottom * sourceWidth + left];
      const bottomRight = plane[bottom * sourceWidth + right];
      const upper = topLeft + (topRight - topLeft) * weightX;
      const lower = bottomLeft + (bottomRight - bottomLeft) * weightX;
      output[row * width + column] = upper + (lower - upper) * weightY;
    }
  }
  return output;
}

/** Apply the producer's symmetric normalization in place-free form. */
export function normalizeReceiptPixels(plane) {
  if (!(plane instanceof Float32Array) || plane.length === 0) {
    fail('normalization needs one non-empty F32 plane.');
  }
  const output = new Float32Array(plane.length);
  for (let index = 0; index < plane.length; index++) {
    output[index] = (plane[index] / 255 - 0.5) / 0.5;
  }
  return output;
}

/**
 * Full host preprocessing: any supported image to the declared network plane.
 * Returns the normalized values only; the caller owns the shaped input view.
 */
export function prepareReceiptImage(image, { width, height }) {
  const plane = toGrayscalePlane(image);
  const resized = resizeBilinear(plane, image.width, image.height, width, height);
  return normalizeReceiptPixels(resized);
}

/**
 * Decode a PNG/JPEG file to raw pixels without adding a runtime dependency.
 *
 * Node has no built-in image decoder, so this defers to `sharp` when the host
 * application already has it and otherwise reports what to install. Callers
 * that already hold decoded pixels should use `prepareReceiptImage` directly.
 */
export async function decodeGrayscaleImageFile(path, { width, height }) {
  let sharp;
  try {
    ({ default: sharp } = await import('sharp'));
  } catch {
    fail(
      `decoding ${path} needs an image decoder. Install 'sharp', or preprocess `
      + 'to raw pixels and call prepareReceiptImage directly.',
    );
  }
  const { data, info } = await sharp(path)
    .greyscale()
    .resize(width, height, { fit: 'fill', kernel: 'bilinear' })
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height || info.channels !== 1) {
    fail(`decoder produced ${info.width}x${info.height}x${info.channels} for ${path}.`);
  }
  return Float32Array.from(data);
}
