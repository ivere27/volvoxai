// --- 7. Cross SDPA ---
static int cross_key_allowed(const int32_t* mask, int mask_mode,
                             int query, int key, int seq_len_kv, int causal) {
    if (causal && key > query) return 0;
    if (!mask || mask_mode == 0) return 1;
    if (mask_mode == 1) return mask[key] != 0;                    // [K]
    if (mask_mode == 2) return mask[query * seq_len_kv + key] != 0; // [Q,K]
    return 0;
}

void cross_sdpa_f32(const float* q_in, const float* k_in, const float* v_in, float* output,
                    int seq_len_q, int seq_len_kv, int d_model, int num_heads,
                    int head_dim, float scale, const int32_t* mask,
                    int mask_mode, int causal) {
    for (int q_idx = 0; q_idx < seq_len_q; q_idx++) {
        for (int h_idx = 0; h_idx < num_heads; h_idx++) {
            float max_logit = -1e38f;
            int valid_keys = 0;
            for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
                if (!cross_key_allowed(mask, mask_mode, q_idx, k_idx, seq_len_kv, causal)) continue;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = q_in[q_idx*d_model + h_idx*head_dim + d];
                    float k = k_in[k_idx*d_model + h_idx*head_dim + d];
                    score += q * k;
                }
                score *= scale;
                if (score > max_logit) max_logit = score;
                valid_keys++;
            }
            if (!valid_keys) {
                for (int d = 0; d < head_dim; d++)
                    output[q_idx*d_model + h_idx*head_dim + d] = 0.0f;
                continue;
            }
            float sum_exp = 0.0f;
            for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
                if (!cross_key_allowed(mask, mask_mode, q_idx, k_idx, seq_len_kv, causal)) continue;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = q_in[q_idx*d_model + h_idx*head_dim + d];
                    float k = k_in[k_idx*d_model + h_idx*head_dim + d];
                    score += q * k;
                }
                score *= scale;
                sum_exp += fast_expf(score - max_logit);
            }
            for (int d = 0; d < head_dim; d++) {
                float out_val = 0.0f;
                for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
                    if (!cross_key_allowed(mask, mask_mode, q_idx, k_idx, seq_len_kv, causal)) continue;
                    float score = 0.0f;
                    for (int kd = 0; kd < head_dim; kd++) {
                        float q = q_in[q_idx*d_model + h_idx*head_dim + kd];
                        float k = k_in[k_idx*d_model + h_idx*head_dim + kd];
                        score += q * k;
                    }
                    score *= scale;
                    float w = fast_expf(score - max_logit) / sum_exp;
                    float v = v_in[k_idx*d_model + h_idx*head_dim + d];
                    out_val += w * v;
                }
                output[q_idx*d_model + h_idx*head_dim + d] = out_val;
            }
        }
    }
}
