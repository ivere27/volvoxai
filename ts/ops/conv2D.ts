import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  checkedWindowOutput,
  fullSpatialPads,
  fusedActivation,
  positiveInteger,
  spatialKernelPorts,
  spatialPair,
} from './spatialKernelValidation.js';

export function _cpuConv2D(node) {
  const operation = 'Conv2D';
  const ports = spatialKernelPorts(
    node,
    [['input'], ['weight']],
    ['bias'],
    operation,
  );
  const [input, weight] = ports.inputs;
  const biasTensor = ports.optional.bias;
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  assertShapeKernelTensor(weight, `${operation} weight`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(
    node,
    ['stride', 'padding', 'pads', 'dilation', 'groups', 'relu', 'data_layout', 'weight_layout'],
    operation,
  );
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  const weightLayout = params.weight_layout ?? 'HWIO';
  if (weightLayout !== 'HWIO' && weightLayout !== 'HWCM') {
    throw new Error(`${operation} weight_layout must be 'HWIO' or 'HWCM'.`);
  }
  const [strideY, strideX] = spatialPair(params.stride, 1, operation, 'stride', false);
  const pads = fullSpatialPads(params, operation);
  const [dilationY, dilationX] = spatialPair(
    params.dilation,
    1,
    operation,
    'dilation',
    false,
  );
  const groups = positiveInteger(params.groups, 1, operation, 'groups');
  const relu = fusedActivation(params.relu, operation);
  const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
  const [kernelHeight, kernelWidth, weightChannels, weightOutput] = weight.shape;
  const depthwise = weightLayout === 'HWCM';
  let outputChannels: number;
  if (depthwise) {
    if (groups !== inputChannels || weightChannels !== inputChannels) {
      throw new Error(
        `${operation} depthwise weights must be HWCM [kh,kw,input_channels,multiplier].`,
      );
    }
    outputChannels = inputChannels * weightOutput;
    if (!Number.isSafeInteger(outputChannels)) {
      throw new Error(`${operation} output channel count exceeds the safe integer range.`);
    }
  } else {
    if (weightLayout !== 'HWIO' || inputChannels % groups !== 0 ||
        weightOutput % groups !== 0 || weightChannels !== inputChannels / groups) {
      throw new Error(
        `${operation} grouped weights must be compatible HWIO geometry.`,
      );
    }
    outputChannels = weightOutput;
  }
  if (biasTensor !== undefined) {
    assertShapeKernelTensor(biasTensor, `${operation} bias`, {
      dtypes: ['float32'], minimumRank: 1, maximumRank: 1,
    });
    if (biasTensor.shape[0] !== outputChannels) {
      throw new Error(`${operation} bias shape must be [${outputChannels}].`);
    }
  }
  const outputHeight = checkedWindowOutput(
    inputHeight,
    kernelHeight,
    strideY,
    pads[0],
    pads[2],
    dilationY,
    `${operation} output height`,
  );
  const outputWidth = checkedWindowOutput(
    inputWidth,
    kernelWidth,
    strideX,
    pads[1],
    pads[3],
    dilationX,
    `${operation} output width`,
  );
  assertShapeKernelOutput(
    output,
    [batch, outputHeight, outputWidth, outputChannels],
    'float32',
    undefined,
    operation,
  );
  assertDistinctOutputStorage(output, [input, weight, biasTensor], operation);

  const bias = biasTensor?.buffer;
  const inBuf = input.buffer;
  const weightBuffer = weight.buffer;
  const outputBuffer = output.buffer;
  const groupOutput = outputChannels / groups;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let outputY = 0; outputY < outputHeight; outputY++) {
      for (let outputX = 0; outputX < outputWidth; outputX++) {
        for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
          let sum = 0;
          if (depthwise) {
            const multiplier = weightOutput;
            const inputChannel = Math.floor(outputChannel / multiplier);
            const multiplierIndex = outputChannel - inputChannel * multiplier;
            for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
              for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
                const inputY = outputY * strideY + kernelY * dilationY - pads[0];
                const inputX = outputX * strideX + kernelX * dilationX - pads[1];
                if (inputY >= 0 && inputY < inputHeight &&
                    inputX >= 0 && inputX < inputWidth) {
                  const inputIndex =
                    ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                      inputChannels + inputChannel;
                  const weightIndex =
                    (((kernelY * kernelWidth + kernelX) * inputChannels + inputChannel) *
                      multiplier) + multiplierIndex;
                  sum += inBuf[inputIndex] * weightBuffer[weightIndex];
                }
              }
            }
          } else {
            const group = Math.floor(outputChannel / groupOutput);
            const inputStart = group * weightChannels;
            for (let localInputChannel = 0; localInputChannel < weightChannels;
              localInputChannel++) {
              const inputChannel = inputStart + localInputChannel;
              for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
                for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
                  const inputY = outputY * strideY + kernelY * dilationY - pads[0];
                  const inputX = outputX * strideX + kernelX * dilationX - pads[1];
                  if (inputY >= 0 && inputY < inputHeight &&
                      inputX >= 0 && inputX < inputWidth) {
                    const inputIndex =
                      ((batchIndex * inputHeight + inputY) * inputWidth + inputX) *
                        inputChannels + inputChannel;
                    const weightIndex =
                      (((kernelY * kernelWidth + kernelX) * weightChannels +
                        localInputChannel) * outputChannels) + outputChannel;
                    sum += inBuf[inputIndex] * weightBuffer[weightIndex];
                  }
                }
              }
            }
          }
          if (bias) sum += bias[outputChannel];
          if (relu === 1) sum = Math.max(sum, 0);
          else if (relu === 2) sum = Math.min(Math.max(sum, 0), 6);
          outputBuffer[
            ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
              outputChannels + outputChannel
          ] = sum;
        }
      }
    }
  }
}
