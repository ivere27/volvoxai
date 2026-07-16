function elementCount(shape) {
  return shape.reduce((count, dimension) => count * dimension, 1);
}

function strides(shape) {
  const out = new Array(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index--) {
    out[index] = stride;
    stride *= shape[index];
  }
  return out;
}

function assertShape(shape, label) {
  if (!Array.isArray(shape) || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    throw new Error(`${label} must have a positive integer shape.`);
  }
}

export function cpuBroadcastBinary(node, operation, operationName) {
  const a = node.inputs.a;
  const b = node.inputs.b;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  assertShape(a?.shape, `${operationName} input a`);
  assertShape(b?.shape, `${operationName} input b`);
  assertShape(output?.shape, `${operationName} output`);
  if (!a.buffer || !b.buffer || !output.buffer ||
      a.buffer.length !== elementCount(a.shape) || b.buffer.length !== elementCount(b.shape) ||
      output.buffer.length !== elementCount(output.shape)) {
    throw new Error(`${operationName} tensor storage does not match its shape.`);
  }
  const rank = Math.max(a.shape.length, b.shape.length);
  if (output.shape.length !== rank) {
    throw new Error(`${operationName} output rank does not match the broadcast rank.`);
  }
  const aOffset = rank - a.shape.length;
  const bOffset = rank - b.shape.length;
  for (let dimension = 0; dimension < rank; dimension++) {
    const ad = dimension < aOffset ? 1 : a.shape[dimension - aOffset];
    const bd = dimension < bOffset ? 1 : b.shape[dimension - bOffset];
    const expected = Math.max(ad, bd);
    if ((ad !== 1 && bd !== 1 && ad !== bd) || output.shape[dimension] !== expected) {
      throw new Error(`${operationName} shapes [${a.shape}] and [${b.shape}] do not broadcast to [${output.shape}].`);
    }
  }

  if (a.buffer.length === output.buffer.length && b.buffer.length === output.buffer.length) {
    for (let index = 0; index < output.buffer.length; index++) {
      output.buffer[index] = operation(a.buffer[index], b.buffer[index]);
    }
    return;
  }
  if (a.buffer.length === 1 && b.buffer.length === output.buffer.length) {
    const scalar = a.buffer[0];
    for (let index = 0; index < output.buffer.length; index++) output.buffer[index] = operation(scalar, b.buffer[index]);
    return;
  }
  if (b.buffer.length === 1 && a.buffer.length === output.buffer.length) {
    const scalar = b.buffer[0];
    for (let index = 0; index < output.buffer.length; index++) output.buffer[index] = operation(a.buffer[index], scalar);
    return;
  }

  const aStrides = strides(a.shape);
  const bStrides = strides(b.shape);
  for (let outputIndex = 0; outputIndex < output.buffer.length; outputIndex++) {
    let remaining = outputIndex;
    let aIndex = 0;
    let bIndex = 0;
    for (let dimension = rank - 1; dimension >= 0; dimension--) {
      const coordinate = remaining % output.shape[dimension];
      remaining = Math.floor(remaining / output.shape[dimension]);
      if (dimension >= aOffset && a.shape[dimension - aOffset] !== 1) {
        aIndex += coordinate * aStrides[dimension - aOffset];
      }
      if (dimension >= bOffset && b.shape[dimension - bOffset] !== 1) {
        bIndex += coordinate * bStrides[dimension - bOffset];
      }
    }
    output.buffer[outputIndex] = operation(a.buffer[aIndex], b.buffer[bIndex]);
  }
}
