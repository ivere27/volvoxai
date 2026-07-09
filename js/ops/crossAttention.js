export function _cpuCrossAttention(node) {

    const q_in = node.inputs.q.buffer;
    const kv_in = node.inputs.kv.buffer;
    const wBuf = node.inputs.weight.buffer;
    const scale = node.inputs.scale ? node.inputs.scale.buffer : null;
    const bias = node.inputs.bias ? node.inputs.bias.buffer : null;
    const outBuf = node.outputs.out.buffer;
    const seq_len_q = node.inputs.q.shape[1];
    const seq_len_kv = node.inputs.kv.shape[1];
    const d_model = node.outputs.out.shape[2];
    const num_heads = node.params.heads || 8;
    const head_dim = d_model / num_heads;
    const scale_factor = 1 / Math.sqrt(head_dim);
    for (let h = 0; h < num_heads; h++) {
      for (let q = 0; q < seq_len_q; q++) {
        const q_proj = new Float32Array(head_dim);
        for (let d = 0; d < head_dim; d++) {
          let sum = 0;
          const out_col = h * head_dim + d;
          for (let i = 0; i < d_model; i++) {
            sum += q_in[q * d_model + i] * wBuf[out_col * d_model + i];
          }
          if (scale) sum *= scale[out_col];
          if (bias) sum += bias[out_col];
          q_proj[d] = sum;
        }
        const logits = new Float32Array(seq_len_kv);
        let max_logit = -Infinity;
        for (let k = 0; k < seq_len_kv; k++) {
          let score = 0;
          for (let d = 0; d < head_dim; d++) {
            let k_val = 0;
            const out_col = d_model + h * head_dim + d;
            for (let i = 0; i < d_model; i++) {
              k_val += kv_in[k * d_model + i] * wBuf[out_col * d_model + i];
            }
            if (scale) k_val *= scale[out_col];
            if (bias) k_val += bias[out_col];
            score += q_proj[d] * k_val;
          }
          score *= scale_factor;
          logits[k] = score;
          if (score > max_logit) max_logit = score;
        }
        let sum_exp = 0;
        for (let k = 0; k < seq_len_kv; k++) {
          const exp_val = Math.exp(logits[k] - max_logit);
          logits[k] = exp_val;
          sum_exp += exp_val;
        }
        for (let d = 0; d < head_dim; d++) {
          let out_val = 0;
          for (let k = 0; k < seq_len_kv; k++) {
            const w = logits[k] / sum_exp;
            let v_val = 0;
            const out_col = d_model * 2 + h * head_dim + d;
            for (let i = 0; i < d_model; i++) {
              v_val += kv_in[k * d_model + i] * wBuf[out_col * d_model + i];
            }
            if (scale) v_val *= scale[out_col];
            if (bias) v_val += bias[out_col];
            out_val += w * v_val;
          }
          outBuf[q * d_model + h * head_dim + d] = out_val;
        }
      }
    }
  }