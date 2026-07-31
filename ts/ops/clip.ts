export function _cpuClip(node) {
  const input = node.inputs.input || node.inputs.x || node.inputs.data;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const sameShape = (left, right) => Array.isArray(left) && Array.isArray(right) &&
    left.length === right.length && left.every((dimension, index) => dimension === right[index]);
  if (!input || !output || !['float32', 'int32'].includes(input.dtype) ||
      output.dtype !== input.dtype || !sameShape(input.shape, output.shape) ||
      !input.buffer || !output.buffer || input.buffer.length !== output.buffer.length ||
      (input.dtype === 'float32' &&
        (!(input.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array))) ||
      (input.dtype === 'int32' &&
        (!(input.buffer instanceof Int32Array) || !(output.buffer instanceof Int32Array)))) {
    throw new Error(`Clip node ${node.id ?? '<unnamed>'} requires same-shape F32 or I32 input/output storage.`);
  }

  const scalarBound = (name, fallback) => {
    const tensor = node.inputs[name];
    if (!tensor) return node.params?.[name] ?? fallback;
    if (tensor.dtype !== input.dtype || !tensor.buffer || tensor.buffer.length !== 1) {
      throw new Error(`Clip node ${node.id ?? '<unnamed>'} ${name} must be a scalar with the input dtype.`);
    }
    return tensor.buffer[0];
  };
  const minimum = scalarBound('min', input.dtype === 'int32' ? -2147483648 : -Infinity);
  const maximum = scalarBound('max', input.dtype === 'int32' ? 2147483647 : Infinity);
  if (typeof minimum !== 'number' || typeof maximum !== 'number' ||
      Number.isNaN(minimum) || Number.isNaN(maximum) || minimum > maximum ||
      (input.dtype === 'int32' &&
        (!Number.isInteger(minimum) || !Number.isInteger(maximum) ||
          minimum < -2147483648 || maximum > 2147483647))) {
    throw new Error(`Clip node ${node.id ?? '<unnamed>'} has invalid ${input.dtype === 'int32' ? 'I32 ' : ''}bounds.`);
  }

  for (let index = 0; index < input.buffer.length; index++) {
    let value = input.buffer[index];
    if (value < minimum) value = minimum;
    if (value > maximum) value = maximum;
    output.buffer[index] = value;
  }
}
