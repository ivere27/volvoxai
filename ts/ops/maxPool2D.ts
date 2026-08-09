import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  assertFalseOrAbsent,
  checkedWindowOutput,
  fullSpatialPads,
  spatialKernelPorts,
  spatialPair,
} from './spatialKernelValidation.js';

// MaxPool preserves integer ordering when its per-tensor I8/U8 descriptor is
// unchanged. The canonical path deliberately excludes dilated and ceil-mode
// windows so every backend implements one floor-window NHWC contract.
export function _cpuMaxPool2D(node) {
  const operation = 'MaxPool2D';
  const ports = spatialKernelPorts(
    node,
    [['input', 'x', 'data']],
    [],
    operation,
  );
  const input = ports.inputs[0];
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32', 'int8', 'uint8'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(
    node,
    ['kernel', 'stride', 'padding', 'pads', 'dilation', 'ceil_mode', 'data_layout'],
    operation,
  );
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  assertFalseOrAbsent(params.ceil_mode, operation, 'ceil_mode');
  const [kernelY, kernelX] = spatialPair(
    params.kernel,
    1,
    operation,
    'kernel',
    false,
    true,
  );
  const [strideY, strideX] = spatialPair(params.stride, 1, operation, 'stride', false);
  const [dilationY, dilationX] = spatialPair(
    params.dilation,
    1,
    operation,
    'dilation',
    false,
  );
  if (dilationY !== 1 || dilationX !== 1) {
    throw new Error(`${operation} supports only unit dilation.`);
  }
  const pads = fullSpatialPads(params, operation);
  if ((input.dtype === 'int8' || input.dtype === 'uint8') &&
      input.quantization?.scheme !== 'per_tensor') {
    throw new Error(`${operation} raw byte storage requires per-tensor quantization.`);
  }

  const [batch, inputHeight, inputWidth, channels] = input.shape;
  const outputHeight = checkedWindowOutput(
    inputHeight,
    kernelY,
    strideY,
    pads[0],
    pads[2],
    1,
    `${operation} output height`,
  );
  const outputWidth = checkedWindowOutput(
    inputWidth,
    kernelX,
    strideX,
    pads[1],
    pads[3],
    1,
    `${operation} output width`,
  );
  assertShapeKernelOutput(
    output,
    [batch, outputHeight, outputWidth, channels],
    input.dtype,
    input.quantization,
    operation,
  );
  assertDistinctOutputStorage(output, [input], operation);

  const inputBuffer = input.buffer;
  const outputBuffer = output.buffer;
  const quantized = input.dtype === 'int8' || input.dtype === 'uint8';
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let outputY = 0; outputY < outputHeight; outputY++) {
      for (let outputX = 0; outputX < outputWidth; outputX++) {
        for (let channel = 0; channel < channels; channel++) {
          let best = quantized ? (input.dtype === 'int8' ? -128 : 0) : -Infinity;
          for (let kernelOffsetY = 0; kernelOffsetY < kernelY; kernelOffsetY++) {
            for (let kernelOffsetX = 0; kernelOffsetX < kernelX; kernelOffsetX++) {
              const inputY = outputY * strideY + kernelOffsetY - pads[0];
              const inputX = outputX * strideX + kernelOffsetX - pads[1];
              if (inputY >= 0 && inputY < inputHeight &&
                  inputX >= 0 && inputX < inputWidth) {
                const value = inputBuffer[
                  ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                    channels + channel
                ];
                if (value > best) best = value;
              }
            }
          }
          outputBuffer[
            ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
              channels + channel
          ] = best;
        }
      }
    }
  }
}
