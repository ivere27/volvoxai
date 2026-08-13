/*
 * TinyReceipt host-input utilities shared by the current split session and
 * its qualification tools. This module intentionally contains no model,
 * package-manifest, compilation, or execution-session compatibility surface.
 */

function fail(message) {
  throw new Error(`[TinyReceiptInput] ${message}`);
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function rawPixelSource(image) {
  if (!isRecord(image) || !ArrayBuffer.isView(image.data) || image.data instanceof DataView
      || !Number.isInteger(image.width) || image.width <= 0
      || !Number.isInteger(image.height) || image.height <= 0) {
    return null;
  }
  const pixels = image.width * image.height;
  const channels = image.channels ?? image.data.length / pixels;
  if (!Number.isInteger(channels) || ![1, 3, 4].includes(channels)
      || image.data.length !== pixels * channels) {
    fail('raw image data must use exactly 1, 3, or 4 tightly packed channels.');
  }
  return { data: image.data, width: image.width, height: image.height, channels };
}

async function drawablePixelSource(image) {
  const width = image?.naturalWidth || image?.videoWidth || image?.width;
  const height = image?.naturalHeight || image?.videoHeight || image?.height;
  if (!Number.isInteger(width) || width <= 0
      || !Number.isInteger(height) || height <= 0) {
    fail('image must be raw { data, width, height }, an ImageData-like value, or a drawable browser image.');
  }
  let canvas = null;
  if (typeof OffscreenCanvas !== 'undefined') {
    canvas = new OffscreenCanvas(width, height);
  } else if (typeof document !== 'undefined' && typeof document.createElement === 'function') {
    canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
  }
  if (!canvas) {
    fail('drawable image decoding is unavailable here; Node callers must supply raw pixels or preprocessed F32 NCHW data.');
  }
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) fail('could not create a 2D canvas context for image preprocessing.');
  context.drawImage(image, 0, 0, width, height);
  const imageData = context.getImageData(0, 0, width, height);
  return { data: imageData.data, width, height, channels: 4 };
}

function grayscalePlane(source) {
  const gray = new Float32Array(source.width * source.height);
  for (let index = 0, pixel = 0; pixel < gray.length; pixel++, index += source.channels) {
    let value;
    if (source.channels === 1) {
      value = Number(source.data[index]);
    } else {
      value = Math.round(0.299 * Number(source.data[index])
        + 0.587 * Number(source.data[index + 1])
        + 0.114 * Number(source.data[index + 2]));
    }
    if (!Number.isFinite(value)) fail('raw image pixels must be finite numbers.');
    gray[pixel] = clamp(value, 0, 255);
  }
  return gray;
}

function resizeAndNormalize(gray, inputWidth, inputHeight, outputWidth, outputHeight) {
  const axisKernel = (inputSize, outputSize, destination) => {
    const scale = inputSize / outputSize;
    const filterScale = Math.max(scale, 1);
    const center = (destination + 0.5) * scale;
    let first = clamp(Math.ceil(center - filterScale - 0.5), 0, inputSize - 1);
    let last = clamp(Math.floor(center + filterScale - 0.5), 0, inputSize - 1);
    if (first > last) first = last = clamp(Math.floor(center), 0, inputSize - 1);
    const entries = [];
    let total = 0;
    for (let source = first; source <= last; source++) {
      const weight = Math.max(0, 1 - Math.abs((source + 0.5 - center) / filterScale));
      if (weight !== 0) {
        entries.push([source, weight]);
        total += weight;
      }
    }
    if (!(total > 0)) fail('could not construct a bilinear image-resize kernel.');
    for (const entry of entries) entry[1] /= total;
    return entries;
  };

  const xKernels = Array.from(
    { length: outputWidth }, (_, x) => axisKernel(inputWidth, outputWidth, x),
  );
  const yKernels = Array.from(
    { length: outputHeight }, (_, y) => axisKernel(inputHeight, outputHeight, y),
  );
  const horizontal = new Float32Array(inputHeight * outputWidth);
  for (let y = 0; y < inputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      let pixel = 0;
      for (const [sourceX, weight] of xKernels[x]) {
        pixel += gray[y * inputWidth + sourceX] * weight;
      }
      horizontal[y * outputWidth + x] = pixel;
    }
  }

  const output = new Float32Array(outputWidth * outputHeight);
  for (let y = 0; y < outputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      let pixel = 0;
      for (const [sourceY, weight] of yKernels[y]) {
        pixel += horizontal[sourceY * outputWidth + x] * weight;
      }
      const rounded = Math.round(clamp(pixel, 0, 255));
      const unit = Math.fround(Math.fround(rounded) / Math.fround(255.0));
      const doubled = Math.fround(unit * Math.fround(2.0));
      output[y * outputWidth + x] = Math.fround(doubled - Math.fround(1.0));
    }
  }
  return output;
}

/**
 * Convert raw pixels to the split package's flat F32 grayscale image tensor.
 * The caller supplies the package-owned width and height and applies the
 * manifest's NCHW shape at the shaped execution boundary.
 */
export async function preprocessTinyReceiptImage(image, preprocessing) {
  if (!preprocessing || !Number.isInteger(preprocessing.width)
      || preprocessing.width <= 0 || !Number.isInteger(preprocessing.height)
      || preprocessing.height <= 0) {
    fail('preprocessing requires positive width and height.');
  }
  const source = rawPixelSource(image) || await drawablePixelSource(image);
  return resizeAndNormalize(
    grayscalePlane(source),
    source.width,
    source.height,
    preprocessing.width,
    preprocessing.height,
  );
}
