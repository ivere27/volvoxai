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

export function _cpuAveragePool2D(node) {
  const operation = 'AveragePool2D';
  const ports = spatialKernelPorts(node, [['input', 'x']], [], operation);
  const input = ports.inputs[0];
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(
    node,
    [
      'kernel', 'stride', 'padding', 'pads', 'dilation', 'ceil_mode',
      'count_include_pad', 'auto_pad', 'data_layout',
    ],
    operation,
  );
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  assertFalseOrAbsent(params.ceil_mode, operation, 'ceil_mode');
  assertFalseOrAbsent(params.count_include_pad, operation, 'count_include_pad');
  if (params.auto_pad !== undefined && params.auto_pad !== '' && params.auto_pad !== 'NOTSET') {
    throw new Error(`${operation} auto_pad must be 'NOTSET', empty, or absent.`);
  }
  const [kernelHeight, kernelWidth] = spatialPair(
    params.kernel,
    1,
    operation,
    'kernel',
    false,
    true,
  );
  const [strideHeight, strideWidth] = spatialPair(
    params.stride,
    1,
    operation,
    'stride',
    false,
  );
  const [dilationHeight, dilationWidth] = spatialPair(
    params.dilation,
    1,
    operation,
    'dilation',
    false,
  );
  if (dilationHeight !== 1 || dilationWidth !== 1) {
    throw new Error(`${operation} supports only unit dilation.`);
  }
  const pads = fullSpatialPads(params, operation, true);
  const [batch, inputHeight, inputWidth, channels] = input.shape;
  const outputHeight = checkedWindowOutput(
    inputHeight,
    kernelHeight,
    strideHeight,
    pads[0],
    pads[2],
    1,
    `${operation} output height`,
  );
  const outputWidth = checkedWindowOutput(
    inputWidth,
    kernelWidth,
    strideWidth,
    pads[1],
    pads[3],
    1,
    `${operation} output width`,
  );
  assertShapeKernelOutput(
    output,
    [batch, outputHeight, outputWidth, channels],
    'float32',
    undefined,
    operation,
  );
  assertDistinctOutputStorage(output, [input], operation);

  const inputBuffer = input.buffer;
  const outputBuffer = output.buffer;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let outputY = 0; outputY < outputHeight; outputY++) {
      for (let outputX = 0; outputX < outputWidth; outputX++) {
        for (let channel = 0; channel < channels; channel++) {
          let sum = 0;
          let count = 0;
          for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
            for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
              const inputY = outputY * strideHeight - pads[0] + kernelY;
              const inputX = outputX * strideWidth - pads[1] + kernelX;
              if (inputY >= 0 && inputY < inputHeight &&
                  inputX >= 0 && inputX < inputWidth) {
                const inputIndex =
                  ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                    channels + channel;
                sum += inputBuffer[inputIndex];
                count++;
              }
            }
          }
          const outputIndex =
            ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
              channels + channel;
          outputBuffer[outputIndex] = count > 0 ? sum / count : 0;
        }
      }
    }
  }
}
