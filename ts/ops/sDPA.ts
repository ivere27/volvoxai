function maskElementCount(shape) {
  return shape.reduce((count, dimension) => count * dimension, 1);
}

/**
 * Build the binary attention-mask lookup shared by the CPU attention forward
 * and backward implementations. A nonzero entry keeps the key; zero masks it.
 *
 * Rank-two masks are necessarily ambiguous when batch === queries. In that
 * case they use [B, K] padding-mask semantics; [B, Q, K] remains available for
 * an explicitly query-dependent mask.
 */
export function _cpuAttentionMask(node, batch, queries, keys) {
  const mask = node.inputs?.mask;
  if (!mask) return () => true;
  if (mask.dtype !== "int32" || !(mask.buffer instanceof Int32Array)) {
    throw new Error(`Attention node ${node.id} mask must use int32 binary storage.`);
  }
  if (!Array.isArray(mask.shape) ||
      mask.buffer instanceof DataView || maskElementCount(mask.shape) !== mask.buffer.length) {
    throw new Error(`Attention node ${node.id} mask storage does not match its shape.`);
  }

  const shape = mask.shape;
  let layout;
  if (shape.length === 1 && shape[0] === keys) {
    layout = "K";
  } else if (shape.length === 2 && shape[1] === keys &&
             (shape[0] === batch || shape[0] === queries)) {
    layout = shape[0] === batch ? "BK" : "QK";
  } else if (shape.length === 3 && shape[0] === batch &&
             shape[1] === queries && shape[2] === keys) {
    layout = "BQK";
  } else {
    throw new Error(
      `Attention node ${node.id} mask must have shape [K], [B,K], [Q,K], or [B,Q,K].`,
    );
  }

  for (const value of mask.buffer) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`Attention node ${node.id} mask contains a non-finite value.`);
    }
  }

  return (batchIndex, query, key) => {
    let index;
    if (layout === "K") index = key;
    else if (layout === "BK") index = batchIndex * keys + key;
    else if (layout === "QK") index = query * keys + key;
    else index = (batchIndex * queries + query) * keys + key;
    return mask.buffer[index] !== 0;
  };
}

export function _cpuSDPA(node, execution: {
  probabilityMultiplier?: (index: number) => number;
} = {}) {
  const qkvShape = node.inputs.qkv.shape;
  const outputShape = node.outputs.out.shape;
  const rank = qkvShape.length;
  const batch = rank === 2 ? 1 : qkvShape[0];
  const outputBatch = outputShape.length === 2 ? 1 : outputShape[0];
  const seq_len = qkvShape[qkvShape.length - 2];
  const d_model = outputShape[outputShape.length - 1];
  if ((rank !== 2 && rank !== 3) || outputShape.length !== rank ||
      batch !== outputBatch || qkvShape[qkvShape.length - 1] !== 3 * d_model ||
      outputShape[outputShape.length - 2] !== seq_len) {
    throw new Error(`SDPA node ${node.id} requires compatible [batch?, seq, 3*d_model] input and [batch?, seq, d_model] output.`);
  }
  const qkv = node.inputs.qkv.buffer;
  const outBuf = node.outputs.out.buffer;
  const num_heads = node.params.heads || 8;
  const head_dim = d_model / num_heads;
  if (!Number.isInteger(num_heads) || num_heads <= 0 || !Number.isInteger(head_dim) || head_dim <= 0) {
    throw new Error(`SDPA node ${node.id} has incompatible heads and d_model.`);
  }
  const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(head_dim);
  const causal = node.params.causal !== false;
  const keeps = _cpuAttentionMask(node, batch, seq_len, seq_len);
  const probabilityMultiplier = execution.probabilityMultiplier || (() => 1);
  for (let batchIndex = 0; batchIndex < batch; batchIndex++) {
    const qkvBase = batchIndex * seq_len * d_model * 3;
    const outputBase = batchIndex * seq_len * d_model;
    for (let h = 0; h < num_heads; h++) {
      for (let q = 0; q < seq_len; q++) {
        const probabilities = new Float64Array(seq_len);
        let max_logit = -Infinity;
        let visibleKeys = 0;
        for (let k = 0; k < seq_len; k++) {
          if ((causal && k > q) || !keeps(batchIndex, q, k)) continue;
          let score = 0;
          for (let d = 0; d < head_dim; d++) {
            const q_val = qkv[qkvBase + q * (d_model * 3) + h * head_dim + d];
            const k_val = qkv[qkvBase + k * (d_model * 3) + d_model + h * head_dim + d];
            score += q_val * k_val;
          }
          score *= scale;
          probabilities[k] = score;
          if (score > max_logit) max_logit = score;
          visibleKeys++;
        }

        if (visibleKeys === 0) {
          for (let d = 0; d < head_dim; d++) {
            outBuf[outputBase + q * d_model + h * head_dim + d] = 0;
          }
          continue;
        }

        let sum_exp = 0;
        for (let k = 0; k < seq_len; k++) {
          if ((causal && k > q) || !keeps(batchIndex, q, k)) continue;
          probabilities[k] = Math.exp(probabilities[k] - max_logit);
          sum_exp += probabilities[k];
        }
        for (let d = 0; d < head_dim; d++) {
          let out_val = 0;
          for (let k = 0; k < seq_len; k++) {
            if ((causal && k > q) || !keeps(batchIndex, q, k)) continue;
            const v_val = qkv[qkvBase + k * (d_model * 3) + d_model * 2 + h * head_dim + d];
            const probabilityIndex = (((batchIndex * num_heads + h) * seq_len + q) * seq_len + k);
            out_val += (probabilities[k] / sum_exp) * probabilityMultiplier(probabilityIndex) * v_val;
          }
          outBuf[outputBase + q * d_model + h * head_dim + d] = out_val;
        }
      }
    }
  }
}
