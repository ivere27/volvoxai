import {
  assertShapeKernelOutput,
  assertShapeKernelParams,
  assertShapeKernelTensor,
} from './shapeKernelValidation.js';
import {
  assertCanonicalLayout,
  assertDistinctOutputStorage,
  checkedWindowOutput,
  fusedActivation,
  positiveInteger,
  spatialKernelPorts,
  spatialScalar,
} from './spatialKernelValidation.js';

/**
 * Conv1D over NLC activations [batch, length, channels] and WIO weights
 * [kernel, in_channels_per_group, out_channels].
 *
 * This is the 1-D projection of the Conv2D NHWC/HWIO contract: channels last on
 * the activation, out_channels innermost on the weight. Keeping it channels-last
 * means a Conv1D sitting between LayerNorm/Linear/attention in a sequence model
 * needs no layout transpose at either boundary, and it leaves the innermost loop
 * running contiguously over out_channels in both the weight row and the output
 * row.
 */
export function _cpuConv1D(node) {
  const operation = 'Conv1D';
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
    dtypes: ['float32'], minimumRank: 3, maximumRank: 3,
  });
  assertShapeKernelTensor(weight, `${operation} weight`, {
    dtypes: ['float32'], minimumRank: 3, maximumRank: 3,
  });
  const params = assertShapeKernelParams(
    node,
    ['stride', 'padding', 'groups', 'relu', 'data_layout', 'weight_layout'],
    operation,
  );
  assertCanonicalLayout(params.data_layout, 'NLC', operation, 'data_layout');
  assertCanonicalLayout(params.weight_layout, 'WIO', operation, 'weight_layout');
  const stride = spatialScalar(params.stride, 1, operation, 'stride', false);
  const padding = spatialScalar(params.padding, 0, operation, 'padding', true);
  const groups = positiveInteger(params.groups, 1, operation, 'groups');
  const relu = fusedActivation(params.relu, operation);
  const [batch, inputLength, inputChannels] = input.shape;
  const [kernel, weightChannels, outputChannels] = weight.shape;
  if (inputChannels % groups !== 0 || outputChannels % groups !== 0 ||
      weightChannels !== inputChannels / groups) {
    throw new Error(
      `${operation} requires WIO weights compatible with input channels and groups.`,
    );
  }
  if (biasTensor !== undefined) {
    assertShapeKernelTensor(biasTensor, `${operation} bias`, {
      dtypes: ['float32'], minimumRank: 1, maximumRank: 1,
    });
    if (biasTensor.shape[0] !== outputChannels) {
      throw new Error(`${operation} bias shape must be [${outputChannels}].`);
    }
  }
  const outputLength = checkedWindowOutput(
    inputLength,
    kernel,
    stride,
    padding,
    padding,
    1,
    `${operation} output length`,
  );
  assertShapeKernelOutput(
    output,
    [batch, outputLength, outputChannels],
    'float32',
    undefined,
    operation,
  );
  assertDistinctOutputStorage(output, [input, weight, biasTensor], operation);

  const bias = biasTensor?.buffer;
  const inBuf = input.buffer;
  const wBuf = weight.buffer;
  const outBuf = output.buffer;
  const groupOut = outputChannels / groups;
  for (let b = 0; b < batch; b++) {
    const inBase = b * inputLength * inputChannels;
    const outBase = b * outputLength * outputChannels;
    for (let x = 0; x < outputLength; x++) {
      const outRow = outBase + x * outputChannels;
      for (let oc = 0; oc < outputChannels; oc++) outBuf[outRow + oc] = bias ? bias[oc] : 0;
      for (let kernelIndex = 0; kernelIndex < kernel; kernelIndex++) {
        const inputX = x * stride + kernelIndex - padding;
        if (inputX < 0 || inputX >= inputLength) continue;
        const inRow = inBase + inputX * inputChannels;
        const weightTap = kernelIndex * weightChannels * outputChannels;
        for (let group = 0; group < groups; group++) {
          const inputChannelBase = inRow + group * weightChannels;
          const outputChannelBase = group * groupOut;
          for (let localInputChannel = 0; localInputChannel < weightChannels; localInputChannel++) {
            const value = inBuf[inputChannelBase + localInputChannel];
            const weightRow = weightTap +
              localInputChannel * outputChannels + outputChannelBase;
            for (let outputChannel = 0; outputChannel < groupOut; outputChannel++) {
              outBuf[outRow + outputChannelBase + outputChannel] +=
                value * wBuf[weightRow + outputChannel];
            }
          }
        }
      }
      if (relu !== 0) {
        for (let outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
          const index = outRow + outputChannel;
          outBuf[index] = relu === 1
            ? Math.max(outBuf[index], 0)
            : Math.min(Math.max(outBuf[index], 0), 6);
        }
      }
    }
  }
}
