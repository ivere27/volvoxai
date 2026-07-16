function product(shape) {
  return shape.reduce((value, dimension) => value * dimension, 1);
}

export function _cpuGroupNorm(node) {
  const input = node.inputs.input;
  const weight = node.inputs.weight;
  const bias = node.inputs.bias;
  const output = node.outputs.out;

  if (!input || !weight || !bias || !output ||
      input.dtype !== "float32" || weight.dtype !== "float32" ||
      bias.dtype !== "float32" || output.dtype !== "float32") {
    throw new Error(`GroupNorm node ${node.id} requires F32 input, weight, bias, and output tensors.`);
  }
  if (input.shape.length !== 4 || output.shape.length !== 4 ||
      input.shape.some((dimension, index) => output.shape[index] !== dimension)) {
    throw new Error(`GroupNorm node ${node.id} requires matching rank-4 NHWC input and output tensors.`);
  }

  const [batch, height, width, channels] = input.shape;
  const numGroups = node.params?.num_groups;
  const eps = node.params?.eps ?? 1e-5;
  if (!Number.isInteger(numGroups) || numGroups <= 0 || channels % numGroups !== 0) {
    throw new Error(`GroupNorm node ${node.id} num_groups must be a positive divisor of ${channels} channels.`);
  }
  if (weight.shape.length !== 1 || weight.shape[0] !== channels ||
      bias.shape.length !== 1 || bias.shape[0] !== channels) {
    throw new Error(`GroupNorm node ${node.id} weight and bias must have shape [${channels}].`);
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error(`GroupNorm node ${node.id} eps must be a positive finite number.`);
  }
  if (!(input.buffer instanceof Float32Array) || !(weight.buffer instanceof Float32Array) ||
      !(bias.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array) ||
      input.buffer.length !== product(input.shape) || output.buffer.length !== product(output.shape)) {
    throw new Error(`GroupNorm node ${node.id} requires shape-matching Float32Array storage.`);
  }

  const channelsPerGroup = channels / numGroups;
  const valuesPerGroup = height * width * channelsPerGroup;
  const sampleStride = height * width * channels;
  for (let n = 0; n < batch; n++) {
    const sampleOffset = n * sampleStride;
    for (let group = 0; group < numGroups; group++) {
      const firstChannel = group * channelsPerGroup;
      let mean = 0;
      for (let spatial = 0; spatial < height * width; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          mean += input.buffer[offset + localChannel];
        }
      }
      mean /= valuesPerGroup;

      let variance = 0;
      for (let spatial = 0; spatial < height * width; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const centered = input.buffer[offset + localChannel] - mean;
          variance += centered * centered;
        }
      }
      const invStd = 1 / Math.sqrt(variance / valuesPerGroup + eps);

      for (let spatial = 0; spatial < height * width; spatial++) {
        const offset = sampleOffset + spatial * channels + firstChannel;
        for (let localChannel = 0; localChannel < channelsPerGroup; localChannel++) {
          const channel = firstChannel + localChannel;
          const index = offset + localChannel;
          output.buffer[index] = (input.buffer[index] - mean) * invStd * weight.buffer[channel] + bias.buffer[channel];
        }
      }
    }
  }
}
