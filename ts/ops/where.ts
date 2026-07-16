export function _cpuWhere(node) {
  const condition = node.inputs.cond || node.inputs.condition || node.inputs.mask;
  const a = node.inputs.x || node.inputs.a;
  const b = node.inputs.y || node.inputs.b;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const sameShape = (left, right) => Array.isArray(left) && Array.isArray(right) &&
    left.length === right.length && left.every((dimension, index) => dimension === right[index]);
  if (!condition || !a || !b || !output ||
      (condition.dtype !== 'float32' && condition.dtype !== 'int32') ||
      a.dtype !== 'float32' || b.dtype !== 'float32' || output.dtype !== 'float32' ||
      !sameShape(condition.shape, output.shape) || !sameShape(a.shape, output.shape) ||
      !sameShape(b.shape, output.shape) || !condition.buffer || !a.buffer || !b.buffer || !output.buffer ||
      condition.buffer.length !== output.buffer.length || a.buffer.length !== output.buffer.length ||
      b.buffer.length !== output.buffer.length) {
    throw new Error(`${node.opType || 'Where'} requires exact-shape F32 operands/output and an F32 or I32 condition.`);
  }

  for (let index = 0; index < output.buffer.length; index++) {
    // Integer conditions must remain integers: e.g. Int32.MIN_VALUE is true,
    // rather than being accidentally interpreted as the F32 bit pattern -0.
    output.buffer[index] = condition.buffer[index] !== 0 ? a.buffer[index] : b.buffer[index];
  }
}
