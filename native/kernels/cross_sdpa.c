// --- 7. Cross SDPA ---
void cross_sdpa_f32(const float* q_in, const float* k_in, const float* v_in, float* output, 
                    int seq_len_q, int seq_len_kv, int d_model, int num_heads, int head_dim, float scale) {
    for (int q_idx = 0; q_idx < seq_len_q; q_idx++) {
        for (int h_idx = 0; h_idx < num_heads; h_idx++) {
            float max_logit = -1e38f;
            for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q = q_in[q_idx*d_model + h_idx*head_dim + d];
                    float k = k_in[k_idx*d_model + h_idx*head_dim + d];
                    score += q * k;
                }
                score *= scale;
                if (score > max_logit) max_logit = score;
            }
            float sum_exp = 0.0f;
            for (int k_idx = 0; k_idx < seq_len_kv; k_idx++) {
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
