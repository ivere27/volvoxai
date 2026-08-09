import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  assertFalseOrAbsent,
  spatialKernelPorts,
} from './spatialKernelValidation.js';

export function _cpuResize(node) {
  if (node?.opType !== 'Resize' && node?.opType !== 'ResizeNearest2D') {
    throw new Error('Resize CPU kernel requires opType Resize or ResizeNearest2D.');
  }
  const operation = node.opType;
  const ports = spatialKernelPorts(node, [['input', 'x', 'data']], [], operation);
  const input = ports.inputs[0];
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32', 'int8', 'uint8'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(
    node,
    [
      'mode', 'coordinate_transformation_mode', 'coordinate_transform_mode',
      'nearest_mode', 'align_corners', 'antialias', 'data_layout',
    ],
    operation,
  );
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  if (params.coordinate_transform_mode !== undefined) {
    throw new Error(
      `${operation} does not define coordinate_transform_mode; use coordinate_transformation_mode.`,
    );
  }
  if (operation === 'ResizeNearest2D' &&
      params.mode !== undefined && params.mode !== 'nearest') {
    throw new Error(`${operation} mode must be 'nearest'.`);
  }
  if (operation === 'Resize' && params.mode !== undefined &&
      params.mode !== 'nearest' && params.mode !== 'linear') {
    throw new Error(`${operation} mode must be 'nearest' or 'linear'.`);
  }
  const nearest = operation === 'ResizeNearest2D' || params.mode === 'nearest';
  const byteStorage = input.dtype === 'int8' || input.dtype === 'uint8';
  if (byteStorage && input.quantization?.scheme !== 'per_tensor') {
    throw new Error(`${operation} raw byte storage requires per-tensor quantization.`);
  }
  if (byteStorage && !nearest) {
    throw new Error(`${operation} raw byte storage requires explicit nearest mode.`);
  }
  if (nearest) {
    if (params.coordinate_transformation_mode !== undefined &&
        params.coordinate_transformation_mode !== 'asymmetric') {
      throw new Error(
        `${operation} nearest resize requires coordinate_transformation_mode 'asymmetric'.`,
      );
    }
    if (params.nearest_mode !== undefined && params.nearest_mode !== 'floor') {
      throw new Error(`${operation} nearest_mode must be 'floor'.`);
    }
  } else {
    if (params.coordinate_transformation_mode !== undefined &&
        params.coordinate_transformation_mode !== 'half_pixel') {
      throw new Error(
        `${operation} linear resize requires coordinate_transformation_mode 'half_pixel'.`,
      );
    }
    if (params.nearest_mode !== undefined) {
      throw new Error(`${operation} nearest_mode is valid only for nearest resize.`);
    }
  }
  assertFalseOrAbsent(params.align_corners, operation, 'align_corners');
  assertFalseOrAbsent(params.antialias, operation, 'antialias');

  const [batch, inputHeight, inputWidth, channels] = input.shape;
  assertShapeKernelOutput(
    output,
    output.shape,
    input.dtype,
    input.quantization,
    operation,
  );
  if (output.shape.length !== 4 || output.shape[0] !== batch ||
      output.shape[3] !== channels) {
    throw new Error(`${operation} output must preserve NHWC batch and channel extents.`);
  }
  assertDistinctOutputStorage(output, [input], operation);
  const outputHeight = output.shape[1];
  const outputWidth = output.shape[2];
  const source = input.buffer;
  const destination = output.buffer;

  if (nearest) {
    for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
      for (let outputY = 0; outputY < outputHeight; outputY++) {
        const inputY = Math.min(
          inputHeight - 1,
          Math.floor(outputY * inputHeight / outputHeight),
        );
        for (let outputX = 0; outputX < outputWidth; outputX++) {
          const inputX = Math.min(
            inputWidth - 1,
            Math.floor(outputX * inputWidth / outputWidth),
          );
          for (let channel = 0; channel < channels; channel++) {
            destination[
              ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
                channels + channel
            ] = source[
              ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                channels + channel
            ];
          }
        }
      }
    }
    return;
  }

  // Bilinear half-pixel interpolation (align_corners=false).
  const scaleY = inputHeight / outputHeight;
  const scaleX = inputWidth / outputWidth;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let outputY = 0; outputY < outputHeight; outputY++) {
      const sourceY = Math.max(0, (outputY + 0.5) * scaleY - 0.5);
      const y0 = Math.min(inputHeight - 1, Math.floor(sourceY));
      const y1 = Math.min(inputHeight - 1, y0 + 1);
      const fractionY = sourceY - y0;
      for (let outputX = 0; outputX < outputWidth; outputX++) {
        const sourceX = Math.max(0, (outputX + 0.5) * scaleX - 0.5);
        const x0 = Math.min(inputWidth - 1, Math.floor(sourceX));
        const x1 = Math.min(inputWidth - 1, x0 + 1);
        const fractionX = sourceX - x0;
        for (let channel = 0; channel < channels; channel++) {
          const v00 = source[
            ((batchIndex * inputHeight + y0) * inputWidth + x0) * channels + channel
          ];
          const v01 = source[
            ((batchIndex * inputHeight + y0) * inputWidth + x1) * channels + channel
          ];
          const v10 = source[
            ((batchIndex * inputHeight + y1) * inputWidth + x0) * channels + channel
          ];
          const v11 = source[
            ((batchIndex * inputHeight + y1) * inputWidth + x1) * channels + channel
          ];
          destination[
            ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
              channels + channel
          ] = v00 * (1 - fractionY) * (1 - fractionX) +
            v01 * (1 - fractionY) * fractionX +
            v10 * fractionY * (1 - fractionX) +
            v11 * fractionY * fractionX;
        }
      }
    }
  }
}
