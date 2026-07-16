function sameShape(left, right) {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
    left.every((dimension, index) => dimension === right[index]);
}

function tensorElements(tensor) {
  if (!Array.isArray(tensor?.shape)) return null;
  let elements = 1;
  for (const dimension of tensor.shape) {
    if (!Number.isInteger(dimension) || dimension <= 0 ||
        elements > Math.floor(Number.MAX_SAFE_INTEGER / dimension)) return null;
    elements *= dimension;
  }
  return tensor.buffer?.length === elements ? elements : null;
}

export function ropeDescriptor(node) {
  const input = node.inputs.input || node.inputs.x;
  const positions = node.inputs.position_ids || null;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const rank = input?.shape?.length;
  const batch = rank === 2 ? 1 : input?.shape?.[0];
  const sequence = input?.shape?.[rank - 2];
  const width = input?.shape?.[rank - 1];
  const rotaryWidth = node.params?.rotary_dim ?? width;
  const theta = node.params?.theta ?? 10000;
  const positionOffset = node.params?.position_offset ?? 0;
  const interleaved = node.params?.interleaved ?? false;
  if (!input || !output || input.dtype !== 'float32' || output.dtype !== 'float32' ||
      !(input.buffer instanceof Float32Array) || !(output.buffer instanceof Float32Array) ||
      tensorElements(input) == null || tensorElements(output) == null ||
      (rank !== 2 && rank !== 3) || !sameShape(input.shape, output.shape) ||
      !Number.isInteger(batch) || batch <= 0 || !Number.isInteger(sequence) || sequence <= 0 ||
      !Number.isInteger(width) || width <= 0 || !Number.isInteger(rotaryWidth) ||
      rotaryWidth <= 0 || rotaryWidth > width || rotaryWidth % 2 !== 0 ||
      !Number.isFinite(theta) || theta <= 0 || !Number.isInteger(positionOffset) ||
      positionOffset < 0 || positionOffset > 0x7fffffff - (sequence - 1) ||
      typeof interleaved !== 'boolean') {
    throw new Error(`RoPE node ${node.id} requires canonical rank-2/3 F32 tensors and valid rotary parameters.`);
  }
  let positionMode = 0;
  if (positions) {
    if (positions.dtype !== 'int32' || !(positions.buffer instanceof Int32Array) ||
        tensorElements(positions) == null) {
      throw new Error(`RoPE node ${node.id} position_ids must use I32 storage.`);
    }
    if (sameShape(positions.shape, [sequence])) positionMode = 1;
    else if (rank === 3 && sameShape(positions.shape, [batch, sequence])) positionMode = 2;
    else throw new Error(`RoPE node ${node.id} position_ids must have shape [S] or [B,S].`);
    for (const position of positions.buffer) {
      if (position < 0) throw new Error(`RoPE node ${node.id} position_ids must be non-negative.`);
    }
  }
  return { input, positions, output, batch, sequence, width, rotaryWidth, theta,
    positionOffset, interleaved, positionMode };
}

export function _cpuRoPE(node) {
  const descriptor = ropeDescriptor(node);
  const { input, positions, output, batch, sequence, width, rotaryWidth, theta,
    positionOffset, interleaved, positionMode } = descriptor;
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let sequenceIndex = 0; sequenceIndex < sequence; sequenceIndex++) {
      const row = batchIndex * sequence + sequenceIndex;
      const base = row * width;
      const position = positionMode === 1 ? positions.buffer[sequenceIndex] :
        positionMode === 2 ? positions.buffer[row] : positionOffset + sequenceIndex;
      const half = rotaryWidth / 2;
      for (let pair = 0; pair < half; pair++) {
        const left = interleaved ? pair * 2 : pair;
        const right = interleaved ? left + 1 : pair + half;
        const angle = position / Math.pow(theta, (2 * pair) / rotaryWidth);
        const cosine = Math.cos(angle);
        const sine = Math.sin(angle);
        const leftValue = input.buffer[base + left];
        const rightValue = input.buffer[base + right];
        output.buffer[base + left] = leftValue * cosine - rightValue * sine;
        output.buffer[base + right] = leftValue * sine + rightValue * cosine;
      }
      output.buffer.set(input.buffer.subarray(base + rotaryWidth, base + width), base + rotaryWidth);
    }
  }
}
