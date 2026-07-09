export function _cpuSDPA(node) {

    const qkv = node.inputs.qkv.buffer;
    const outBuf = node.outputs.out.buffer;
    const seq_len = node.inputs.qkv.shape[1];
    const d_model = node.outputs.out.shape[2];
    const num_heads = node.params.heads || 8;
    const head_dim = d_model / num_heads;
    const scale = node.params.scale !== undefined ? node.params.scale : 1 / Math.sqrt(head_dim);
    // Causal self-attention: query q attends only to keys k <= q (matches the native
    // sdpa_f32 kernel; required for autoregressive decoders).
    for (let h = 0; h < num_heads; h++) {
      for (let q = 0; q < seq_len; q++) {
        const logits = new Float32Array(seq_len);
        let max_logit = -Infinity;
        for (let k = 0; k <= q; k++) {
          let score = 0;
          for (let d = 0; d < head_dim; d++) {
            const q_val = qkv[q * (d_model * 3) + h * head_dim + d];
            const k_val = qkv[k * (d_model * 3) + d_model + h * head_dim + d];
            score += q_val * k_val;
          }
          score *= scale;
          logits[k] = score;
          if (score > max_logit) max_logit = score;
        }
        let sum_exp = 0;
        for (let k = 0; k <= q; k++) {
          const exp_val = Math.exp(logits[k] - max_logit);
          logits[k] = exp_val;
          sum_exp += exp_val;
        }
        for (let d = 0; d < head_dim; d++) {
          let out_val = 0;
          for (let k = 0; k <= q; k++) {
            const w = logits[k] / sum_exp;
            const v_val = qkv[k * (d_model * 3) + d_model * 2 + h * head_dim + d];
            out_val += w * v_val;
          }
          outBuf[q * d_model + h * head_dim + d] = out_val;
        }
      }
    }
  }