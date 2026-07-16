function product(values) {
  return values.reduce((a, b) => a * b, 1);
}

/**
 * Top-k token router for Mixture-of-Experts graphs.
 *
 * Contract:
 *   input   [..., d_model]
 *   weight  [d_model, num_experts]
 *   bias    [num_experts] (optional)
 *   indices [..., top_k] (F32 integer values)
 *   weights [..., top_k]
 */
export function _cpuMoERouter(node) {
  const input = node.inputs.input || node.inputs.x;
  const weight = node.inputs.weight || node.inputs.router_weight;
  const bias = node.inputs.bias?.buffer || null;
  const indices = node.outputs.indices || node.outputs.expert_indices;
  const gates = node.outputs.weights || node.outputs.expert_weights;
  if (!input?.buffer || !weight?.buffer || !indices?.buffer || !gates?.buffer) {
    throw new Error("MoERouter requires input, weight, indices, and weights tensors.");
  }

  const dModel = input.shape[input.shape.length - 1];
  const rows = product(input.shape.slice(0, -1));
  const numExperts = node.params.num_experts ?? weight.shape[weight.shape.length - 1];
  const topK = node.params.top_k ?? indices.shape[indices.shape.length - 1] ?? 2;
  const temperature = node.params.temperature ?? 1;
  const normalize = node.params.normalize !== false;
  if (weight.shape.length !== 2 || weight.shape[0] !== dModel ||
      indices.buffer.length !== rows * topK || gates.buffer.length !== rows * topK ||
      (bias && bias.length !== numExperts)) {
    throw new Error("MoERouter has incompatible router or output dimensions.");
  }
  if (!Number.isInteger(numExperts) || numExperts <= 0 || numExperts !== weight.shape[weight.shape.length - 1]) {
    throw new Error("MoERouter has an invalid num_experts.");
  }
  if (!Number.isInteger(topK) || topK <= 0 || topK > numExperts) {
    throw new Error("MoERouter top_k must be in [1, num_experts].");
  }
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new Error("MoERouter temperature must be positive and finite.");
  }

  const logits = new Float32Array(numExperts);
  const selected = new Int32Array(topK);
  for (let row = 0; row < rows; row++) {
    for (let expert = 0; expert < numExperts; expert++) {
      let value = bias ? bias[expert] : 0;
      for (let d = 0; d < dModel; d++) {
        value += input.buffer[row * dModel + d] * weight.buffer[d * numExperts + expert];
      }
      logits[expert] = value / temperature;
    }

    selected.fill(-1);
    for (let slot = 0; slot < topK; slot++) {
      let best = -1;
      let bestValue = -Infinity;
      for (let expert = 0; expert < numExperts; expert++) {
        let alreadySelected = false;
        for (let prior = 0; prior < slot; prior++) {
          if (selected[prior] === expert) { alreadySelected = true; break; }
        }
        if (!alreadySelected && (logits[expert] > bestValue ||
            (logits[expert] === bestValue && (best < 0 || expert < best)))) {
          best = expert;
          bestValue = logits[expert];
        }
      }
      selected[slot] = best;
    }

    let maxValue = -Infinity;
    for (let expert = 0; expert < numExperts; expert++) maxValue = Math.max(maxValue, logits[expert]);
    let denominator = 0;
    if (normalize) {
      for (let slot = 0; slot < topK; slot++) denominator += Math.exp(logits[selected[slot]] - maxValue);
    } else {
      for (let expert = 0; expert < numExperts; expert++) denominator += Math.exp(logits[expert] - maxValue);
    }
    for (let slot = 0; slot < topK; slot++) {
      const offset = row * topK + slot;
      indices.buffer[offset] = selected[slot];
      gates.buffer[offset] = Math.exp(logits[selected[slot]] - maxValue) / denominator;
    }
  }
}
