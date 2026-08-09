import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  spatialKernelPorts,
} from './spatialKernelValidation.js';

export function _cpuGlobalAveragePool(node) {
  const operation = 'GlobalAveragePool';
  const ports = spatialKernelPorts(node, [['input']], [], operation);
  const input = ports.inputs[0];
  const output = ports.output;
  assertShapeKernelTensor(input, `${operation} input`, {
    dtypes: ['float32'], minimumRank: 4, maximumRank: 4,
  });
  const params = assertShapeKernelParams(node, ['data_layout'], operation);
  assertCanonicalLayout(params.data_layout, 'NHWC', operation, 'data_layout');
  const [batch, height, width, channels] = input.shape;
  assertShapeKernelOutput(
    output,
    [batch, 1, 1, channels],
    'float32',
    undefined,
    operation,
  );
  assertDistinctOutputStorage(output, [input], operation);

  const spatial = height * width;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let channel = 0; channel < channels; channel++) {
      let sum = 0;
      for (let offset = 0; offset < spatial; offset++) {
        const y = Math.floor(offset / width);
        const x = offset - y * width;
        sum += input.buffer[
          ((batchIndex * height + y) * width + x) * channels + channel
        ];
      }
      output.buffer[batchIndex * channels + channel] = sum / spatial;
    }
  }
}
