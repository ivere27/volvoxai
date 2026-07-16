export function _cpuExpand(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (!input || !output || !Array.isArray(input.shape) || !Array.isArray(output.shape) ||
      input.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
      output.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
      input.shape.length > output.shape.length) {
    throw new Error('Expand requires positive input/output shapes with output rank >= input rank.');
  }
  const inputElements = input.shape.reduce((count, dimension) => count * dimension, 1);
  const outputElements = output.shape.reduce((count, dimension) => count * dimension, 1);
  if (!input.buffer || !output.buffer || input.buffer.length !== inputElements || output.buffer.length !== outputElements) {
    throw new Error('Expand tensor storage does not match its shape.');
  }
  const offset = output.shape.length - input.shape.length;
  const inputStrides = new Array(input.shape.length);
  let stride = 1;
  for (let dimension = input.shape.length - 1; dimension >= 0; dimension--) {
    inputStrides[dimension] = stride;
    stride *= input.shape[dimension];
  }
  for (let dimension = 0; dimension < output.shape.length; dimension++) {
    const inputDimension = dimension < offset ? 1 : input.shape[dimension - offset];
    if (inputDimension !== 1 && inputDimension !== output.shape[dimension]) {
      throw new Error(`Expand input shape [${input.shape}] cannot broadcast to [${output.shape}].`);
    }
  }
  for (let index = 0; index < outputElements; index++) {
    let remaining = index;
    let inputIndex = 0;
    for (let dimension = output.shape.length - 1; dimension >= 0; dimension--) {
      const coordinate = remaining % output.shape[dimension];
      remaining = Math.floor(remaining / output.shape[dimension]);
      if (dimension >= offset && input.shape[dimension - offset] !== 1) {
        inputIndex += coordinate * inputStrides[dimension - offset];
      }
    }
    output.buffer[index] = input.buffer[inputIndex];
  }
}
