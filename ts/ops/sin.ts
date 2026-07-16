function unaryF32Buffers(node, name) {
  const input = node.inputs.input || node.inputs.x;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  if (!input || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
      !(input.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array) ||
      input.buffer.length !== output.buffer.length || !Array.isArray(input.shape) ||
      !Array.isArray(output.shape) || input.shape.length !== output.shape.length ||
      input.shape.some((dimension, index) => dimension !== output.shape[index])) {
    throw new Error(`${name} node ${node.id} requires same-shape F32 input and output.`);
  }
  return [input.buffer, output.buffer];
}

export function _cpuSin(node) {
  const [input, output] = unaryF32Buffers(node, 'Sin');
  for (let index = 0; index < output.length; index++) output[index] = Math.sin(input[index]);
}
