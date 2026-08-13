import { assertShapeKernelParams } from './shapeKernelValidation.js';

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

function f32Tensor(tensor) {
  return !!tensor && tensor.dtype === 'float32' && tensor.buffer instanceof Float32Array &&
    tensor.quantization == null && tensorElements(tensor) != null;
}

function bcMode(tensor, batch, sequence, stateWidth, rank) {
  if (!f32Tensor(tensor)) return -1;
  if (sameShape(tensor.shape, [stateWidth])) return 0;
  if (sameShape(tensor.shape, [sequence, stateWidth])) return 1;
  if (rank === 3 && sameShape(tensor.shape, [batch, sequence, stateWidth])) return 2;
  return -1;
}

export function ssmScanDescriptor(node) {
  const input = node.inputs.input || node.inputs.u;
  const delta = node.inputs.delta;
  const a = node.inputs.A || node.inputs.a;
  const b = node.inputs.B || node.inputs.b;
  const c = node.inputs.C || node.inputs.c;
  const d = node.inputs.D || node.inputs.d || null;
  const z = node.inputs.z || null;
  const initialState = node.inputs.initial_state || null;
  const output = node.outputs.out || Object.values(node.outputs || {})[0];
  const finalState = node.outputs.state || node.outputs.final_state || null;
  const params = assertShapeKernelParams(
    node, ['delta_softplus'], node.opType || 'SSMScan',
  );
  const rank = input?.shape?.length;
  const batch = rank === 2 ? 1 : input?.shape?.[0];
  const sequence = input?.shape?.[rank - 2];
  const channels = input?.shape?.[rank - 1];
  const stateWidth = a?.shape?.[1];
  const deltaSoftplus = params.delta_softplus ?? true;
  const bMode = bcMode(b, batch, sequence, stateWidth, rank);
  const cMode = bcMode(c, batch, sequence, stateWidth, rank);
  const stateShape = [batch, channels, stateWidth];
  if (!f32Tensor(input) || !f32Tensor(delta) || !f32Tensor(a) || !f32Tensor(output) ||
      (rank !== 2 && rank !== 3) || !sameShape(delta.shape, input.shape) ||
      !sameShape(output.shape, input.shape) || !Number.isInteger(batch) || batch <= 0 ||
      !Number.isInteger(sequence) || sequence <= 0 || !Number.isInteger(channels) ||
      channels <= 0 || !Number.isInteger(stateWidth) || stateWidth <= 0 ||
      !sameShape(a.shape, [channels, stateWidth]) || bMode < 0 || cMode < 0 ||
      (d && (!f32Tensor(d) || !sameShape(d.shape, [channels]))) ||
      (z && (!f32Tensor(z) || !sameShape(z.shape, input.shape))) ||
      (initialState && (!f32Tensor(initialState) || !sameShape(initialState.shape, stateShape))) ||
      !finalState || !f32Tensor(finalState) || !sameShape(finalState.shape, stateShape) ||
      typeof deltaSoftplus !== 'boolean') {
    throw new Error(`SSMScan node ${node.id} requires canonical F32 selective-scan tensors.`);
  }
  return { input, delta, a, b, c, d, z, initialState, output, finalState,
    batch, sequence, channels, stateWidth, bMode, cMode, deltaSoftplus };
}

function softplus(value) {
  if (value > 20) return value;
  if (value < -20) return Math.exp(value);
  return Math.log(1 + Math.exp(value));
}

function bcIndex(mode, batchIndex, sequenceIndex, stateIndex, sequence, stateWidth) {
  if (mode === 0) return stateIndex;
  if (mode === 1) return sequenceIndex * stateWidth + stateIndex;
  return (batchIndex * sequence + sequenceIndex) * stateWidth + stateIndex;
}

export function _cpuSSMScan(node) {
  const descriptor = ssmScanDescriptor(node);
  const { input, delta, a, b, c, d, z, initialState, output, finalState,
    batch, sequence, channels, stateWidth, bMode, cMode, deltaSoftplus } = descriptor;
  const state = initialState ? new Float32Array(initialState.buffer) :
    new Float32Array(batch * channels * stateWidth);
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    for (let sequenceIndex = 0; sequenceIndex < sequence; sequenceIndex++) {
      for (let channel = 0; channel < channels; channel++) {
        const inputIndex = (batchIndex * sequence + sequenceIndex) * channels + channel;
        const stateBase = (batchIndex * channels + channel) * stateWidth;
        const aBase = channel * stateWidth;
        const inputValue = input.buffer[inputIndex];
        const dt = deltaSoftplus ? softplus(delta.buffer[inputIndex]) : delta.buffer[inputIndex];
        let result = d ? d.buffer[channel] * inputValue : 0;
        for (let stateIndex = 0; stateIndex < stateWidth; stateIndex++) {
          const bIndex = bcIndex(bMode, batchIndex, sequenceIndex, stateIndex, sequence, stateWidth);
          const cIndex = bcIndex(cMode, batchIndex, sequenceIndex, stateIndex, sequence, stateWidth);
          const value = Math.exp(dt * a.buffer[aBase + stateIndex]) * state[stateBase + stateIndex] +
            dt * b.buffer[bIndex] * inputValue;
          state[stateBase + stateIndex] = value;
          result += state[stateBase + stateIndex] * c.buffer[cIndex];
        }
        if (z) {
          const gate = z.buffer[inputIndex];
          result *= gate / (1 + Math.exp(-gate));
        }
        output.buffer[inputIndex] = result;
      }
    }
  }
  if (finalState) finalState.buffer.set(state);
}
