import { normalizeSpatialPair } from './spatialParameters.js';
import { roundTiesToEven } from './quantizeLinear.js';

function perTensorQuantization(tensor, label) {
  if (!tensor || !['int8', 'uint8'].includes(tensor.dtype) ||
      tensor.quantization?.scheme !== 'per_tensor') {
    throw new Error(`${label} requires I8/U8 storage with per_tensor quantization metadata.`);
  }
  return tensor.quantization;
}

function weightQuantization(weight, outputChannels, label) {
  if (!weight || !['int8', 'uint8'].includes(weight.dtype) || !weight.quantization) {
    throw new Error(`${label} requires I8/U8 weight storage with quantization metadata.`);
  }
  const descriptor = weight.quantization;
  if (descriptor.scheme === 'per_axis' && descriptor.axis === 0 &&
      descriptor.scales.length === outputChannels && descriptor.zero_points.length === outputChannels) {
    return {
      scale: (channel) => descriptor.scales[channel],
      zeroPoint: (channel) => descriptor.zero_points[channel],
    };
  }
  throw new Error(`${label} requires per_axis weight quantization along output-channel axis 0.`);
}

function quantizedRange(dtype) {
  return dtype === 'int8' ? [-128, 127] : [0, 255];
}

function requantizeAccumulator(accumulator, multiplier, output, quantization, relu) {
  const [minimum, maximum] = quantizedRange(output.dtype);
  let transformed = Math.fround(Math.fround(accumulator * multiplier) + quantization.zero_point);
  // A mathematically zero accumulator multiplied by an overflowing scale is
  // NaN in IEEE arithmetic.  QuantizeLinear maps NaN to its zero point, and
  // the portable C/WGSL W8A8 kernels use the same defined boundary behavior.
  // Do this before range checks so assigning NaN to a typed byte array cannot
  // silently turn it into numeric zero instead of the output-domain zero.
  if (Number.isNaN(transformed)) transformed = quantization.zero_point;
  else if (transformed <= minimum) transformed = minimum;
  else if (transformed >= maximum) transformed = maximum;
  else transformed = roundTiesToEven(transformed);
  if (relu) {
    transformed = Math.max(transformed, quantization.zero_point);
    if (relu >= 2) {
      const upper = Math.min(maximum, Math.max(minimum,
        roundTiesToEven(Math.fround(Math.fround(6 / quantization.scale) + quantization.zero_point))));
      transformed = Math.min(transformed, upper);
    }
  }
  return transformed;
}

// Canonical W8A8 Conv2D reference: NHWC activations and OHWI weights. It uses
// a defined I32 accumulator bound and I32 bias whose scale is
// input_scale * weight_scale[channel]. Optimized WASM/GPU paths are compared
// against this implementation rather than changing its semantics.
export function _cpuQConv2D(node) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias || null;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (!input || !weight || !output || input.shape?.length !== 4 || weight.shape?.length !== 4 ||
      output.shape?.length !== 4 || input.buffer?.length !== input.shape.reduce((a, b) => a * b, 1) ||
      weight.buffer?.length !== weight.shape.reduce((a, b) => a * b, 1) ||
      output.buffer?.length !== output.shape.reduce((a, b) => a * b, 1) ||
      (node.params?.data_layout && node.params.data_layout !== 'NHWC') ||
      (node.params?.weight_layout && node.params.weight_layout !== 'OHWI')) {
    throw new Error(`QConv2D node ${node.id} requires canonical NHWC activations and OHWI weight storage.`);
  }
  const inputQuantization = perTensorQuantization(input, `QConv2D node ${node.id} input`);
  const outputQuantization = perTensorQuantization(output, `QConv2D node ${node.id} output`);
  const [batch, inputHeight, inputWidth, inputChannels] = input.shape;
  const [outputChannels, kernelHeight, kernelWidth, inputPerGroup] = weight.shape;
  const [, outputHeight, outputWidth, outputChannelsFromOutput] = output.shape;
  const groups = node.params?.groups ?? 1;
  const [strideY, strideX] = normalizeSpatialPair(node.params?.stride, 1);
  const [dilationY, dilationX] = normalizeSpatialPair(node.params?.dilation, 1);
  const [paddingY, paddingX] = normalizeSpatialPair(node.params?.padding, 0);
  const pads = node.params?.pads || [paddingY, paddingX, paddingY, paddingX];
  if (!Number.isInteger(groups) || groups <= 0 || inputChannels !== inputPerGroup * groups ||
      outputChannelsFromOutput !== outputChannels || outputChannels % groups !== 0 ||
      !Array.isArray(pads) || pads.length !== 4 || ![strideY, strideX, dilationY, dilationX, ...pads]
        .every((value) => Number.isInteger(value) && value >= 0) || strideY === 0 || strideX === 0 ||
      dilationY === 0 || dilationX === 0) {
    throw new Error(`QConv2D node ${node.id} has incompatible grouped convolution dimensions.`);
  }
  const expectedHeight = Math.floor((inputHeight + pads[0] + pads[2] - dilationY * (kernelHeight - 1) - 1) / strideY) + 1;
  const expectedWidth = Math.floor((inputWidth + pads[1] + pads[3] - dilationX * (kernelWidth - 1) - 1) / strideX) + 1;
  if (expectedHeight !== outputHeight || expectedWidth !== outputWidth) {
    throw new Error(`QConv2D node ${node.id} output shape does not match its stride, padding, and dilation.`);
  }
  if (bias && (bias.dtype !== 'int32' || bias.buffer?.length !== outputChannels || bias.shape?.length !== 1 ||
      bias.shape[0] !== outputChannels)) {
    throw new Error(`QConv2D node ${node.id} bias must be an I32 vector with one value per output channel.`);
  }
  const weightQuant = weightQuantization(weight, outputChannels, `QConv2D node ${node.id} weight`);
  const [inputMinimum, inputMaximum] = quantizedRange(input.dtype);
  const [weightMinimum, weightMaximum] = quantizedRange(weight.dtype);
  const inputMagnitude = Math.max(Math.abs(inputMinimum - inputQuantization.zero_point),
    Math.abs(inputMaximum - inputQuantization.zero_point));
  const groupOutputChannels = outputChannels / groups;
  const terms = kernelHeight * kernelWidth * inputPerGroup;
  for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
    const weightMagnitude = Math.max(Math.abs(weightMinimum - weightQuant.zeroPoint(outputChannel)),
      Math.abs(weightMaximum - weightQuant.zeroPoint(outputChannel)));
    const maximumAccumulator = inputMagnitude * weightMagnitude * terms +
      (bias ? Math.abs(bias.buffer[outputChannel]) : 0);
    if (!Number.isSafeInteger(maximumAccumulator) || maximumAccumulator > 0x7fffffff) {
      throw new Error(`QConv2D node ${node.id} may overflow its defined I32 accumulator at output channel ${outputChannel}.`);
    }
  }
  const relu = node.params?.relu ?? 0;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let outputY = 0; outputY < outputHeight; outputY++) {
      for (let outputX = 0; outputX < outputWidth; outputX++) {
        for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
          const group = Math.floor(outputChannel / groupOutputChannels);
          let accumulator = bias ? bias.buffer[outputChannel] : 0;
          const weightZeroPoint = weightQuant.zeroPoint(outputChannel);
          for (let kernelY = 0; kernelY < kernelHeight; kernelY++) {
            const inputY = outputY * strideY + kernelY * dilationY - pads[0];
            if (inputY < 0 || inputY >= inputHeight) continue;
            for (let kernelX = 0; kernelX < kernelWidth; kernelX++) {
              const inputX = outputX * strideX + kernelX * dilationX - pads[1];
              if (inputX < 0 || inputX >= inputWidth) continue;
              for (let localChannel = 0; localChannel < inputPerGroup; localChannel++) {
                const inputChannel = group * inputPerGroup + localChannel;
                const inputIndex = ((batchIndex * inputHeight + inputY) * inputWidth + inputX) * inputChannels + inputChannel;
                const weightIndex = (((outputChannel * kernelHeight + kernelY) * kernelWidth + kernelX) * inputPerGroup) + localChannel;
                accumulator += (input.buffer[inputIndex] - inputQuantization.zero_point) *
                  (weight.buffer[weightIndex] - weightZeroPoint);
              }
            }
          }
          const multiplier = Math.fround(Math.fround(inputQuantization.scale * weightQuant.scale(outputChannel)) /
            outputQuantization.scale);
          const outputIndex = ((batchIndex * outputHeight + outputY) * outputWidth + outputX) * outputChannels + outputChannel;
          output.buffer[outputIndex] = requantizeAccumulator(accumulator, multiplier, output, outputQuantization, relu);
        }
      }
    }
  }
}
