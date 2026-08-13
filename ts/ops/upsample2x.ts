import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import { checkedShapeMultiply } from './shapeSystem.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  spatialKernelPorts,
} from './spatialKernelValidation.js';

export function _cpuUpsample2x(node) {
  const operation = 'UpsampleNearest2D';
  const ports = spatialKernelPorts(node, [['input']], [], operation);
  const input = ports.inputs[0];
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(node, ['data_layout'], operation);
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  const [batch, height, width, channels] = input.shape;
  const outputHeight = checkedShapeMultiply(height, 2, `${operation} output height`);
  const outputWidth = checkedShapeMultiply(width, 2, `${operation} output width`);
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
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        for (let channel = 0; channel < channels; channel++) {
          const value = inputBuffer[
            ((batchIndex * height + y) * width + x) * channels + channel
          ];
          const outputY = y * 2;
          const outputX = x * 2;
          const topLeft =
            ((batchIndex * outputHeight + outputY) * outputWidth + outputX) *
              channels + channel;
          outputBuffer[topLeft] = value;
          outputBuffer[topLeft + channels] = value;
          outputBuffer[topLeft + outputWidth * channels] = value;
          outputBuffer[topLeft + (outputWidth + 1) * channels] = value;
        }
      }
    }
  }
}
