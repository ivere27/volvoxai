function product(values) {
  return values.reduce((a, b) => a * b, 1);
}

/**
 * Routed expert linear layer.
 *
 * Contract:
 *   input          [..., d_in]
 *   expert_weight  [num_experts, d_in, d_out]
 *   expert_bias    [num_experts, d_out] (optional)
 *   route_indices  [..., top_k] (F32 integer values)
 *   route_weights  [..., top_k]
 *   out            [..., d_out]
 */
export function _cpuMoELinear(node) {
  const input = node.inputs.input || node.inputs.x;
  const expertWeight = node.inputs.expert_weight || node.inputs.weight;
  const expertBias = (node.inputs.expert_bias || node.inputs.bias)?.buffer || null;
  const routeIndices = node.inputs.route_indices || node.inputs.indices;
  const routeWeights = node.inputs.route_weights || node.inputs.weights;
  const output = node.outputs.out || Object.values(node.outputs)[0];
  if (!input?.buffer || !expertWeight?.buffer || !routeIndices?.buffer ||
      !routeWeights?.buffer || !output?.buffer) {
    throw new Error("MoELinear requires input, expert weights, routes, and output tensors.");
  }

  const dIn = input.shape[input.shape.length - 1];
  const dOut = output.shape[output.shape.length - 1];
  const rows = product(input.shape.slice(0, -1));
  const numExperts = expertWeight.shape[0];
  const topK = routeIndices.shape[routeIndices.shape.length - 1];
  if (expertWeight.shape.length !== 3 || expertWeight.shape[1] !== dIn || expertWeight.shape[2] !== dOut) {
    throw new Error(`MoELinear expert_weight must be [experts, ${dIn}, ${dOut}].`);
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > numExperts ||
      routeWeights.buffer.length !== rows * topK || routeIndices.buffer.length !== rows * topK ||
      output.buffer.length !== rows * dOut || (expertBias && expertBias.length !== numExperts * dOut)) {
    throw new Error("MoELinear route tensor shape does not match the input rows.");
  }

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < dOut; col++) {
      let sum = 0;
      for (let slot = 0; slot < topK; slot++) {
        const routeOffset = row * topK + slot;
        const rawExpert = routeIndices.buffer[routeOffset];
        const expert = Math.trunc(rawExpert);
        const gate = routeWeights.buffer[routeOffset];
        if (!Number.isInteger(rawExpert) || expert < 0 || expert >= numExperts || !Number.isFinite(gate)) {
          throw new Error(`MoELinear received invalid route at row ${row}, slot ${slot}.`);
        }
        let expertValue = expertBias ? expertBias[expert * dOut + col] : 0;
        const expertBase = expert * dIn * dOut;
        for (let d = 0; d < dIn; d++) {
          expertValue += input.buffer[row * dIn + d] * expertWeight.buffer[expertBase + d * dOut + col];
        }
        sum += gate * expertValue;
      }
      output.buffer[row * dOut + col] = sum;
    }
  }
}
