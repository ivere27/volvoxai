static VX_CUDA_DEVICE vx_i32 vx_attention_key_allowed(
        const vx_i32* mask, vx_u32 mask_mode, vx_u32 batch_index,
        vx_u32 query, vx_u32 key, vx_u32 sequence_q,
        vx_u32 sequence_kv, vx_u32 causal) {
    if (causal != 0u && key > query) return 0;
    if (mask_mode == 0u) return 1;
    vx_u32 index = key;
    if (mask_mode == 2u) index = batch_index * sequence_kv + key;
    else if (mask_mode == 3u) index = query * sequence_kv + key;
    else if (mask_mode == 4u)
        index = (batch_index * sequence_q + query) * sequence_kv + key;
    return mask[index] != 0;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_sdpa_f32(
        const float* qkv, const vx_i32* mask, float* output,
        vx_u32 sequence, vx_u32 d_model, vx_u32 heads,
        vx_u32 head_dim, vx_u32 batch, float scale, vx_u32 causal,
        vx_u32 mask_mode, vx_u32 query_start, vx_u32 query_count) {
    vx_u32 task = vx_global_x();
    vx_u32 tasks = batch * query_count * heads;
    if (task >= tasks) return;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = query_start + row % query_count;
    vx_u32 batch_index = row / query_count;
    vx_u32 head_offset = head * head_dim;
    vx_u32 qkv_base = batch_index * sequence * d_model * 3u;
    vx_u32 output_base = batch_index * sequence * d_model;
    float maximum = -3.402823466e38f;
    vx_u32 valid_keys = 0u;
    for (vx_u32 key = 0; key < sequence; key++) {
        if (!vx_attention_key_allowed(mask, mask_mode, batch_index, query,
                                      key, sequence, sequence, causal)) continue;
        float score = 0.0f;
        vx_u32 query_base = qkv_base + query * d_model * 3u + head_offset;
        vx_u32 key_base = qkv_base + key * d_model * 3u + d_model +
            head_offset;
        for (vx_u32 channel = 0; channel < head_dim; channel++)
            score += qkv[query_base + channel] * qkv[key_base + channel];
        score *= scale;
        maximum = vx_max(maximum, score);
        valid_keys++;
    }
    vx_u32 destination = output_base + query * d_model + head_offset;
    if (valid_keys == 0u) {
        for (vx_u32 channel = 0; channel < head_dim; channel++)
            output[destination + channel] = 0.0f;
        return;
    }
    float denominator = 0.0f;
    for (vx_u32 key = 0; key < sequence; key++) {
        if (!vx_attention_key_allowed(mask, mask_mode, batch_index, query,
                                      key, sequence, sequence, causal)) continue;
        float score = 0.0f;
        vx_u32 query_base = qkv_base + query * d_model * 3u + head_offset;
        vx_u32 key_base = qkv_base + key * d_model * 3u + d_model +
            head_offset;
        for (vx_u32 channel = 0; channel < head_dim; channel++)
            score += qkv[query_base + channel] * qkv[key_base + channel];
        denominator += vx_exp(score * scale - maximum);
    }
    for (vx_u32 channel = 0; channel < head_dim; channel++) {
        float value = 0.0f;
        for (vx_u32 key = 0; key < sequence; key++) {
            if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                                          query, key, sequence, sequence,
                                          causal)) continue;
            float score = 0.0f;
            vx_u32 query_base = qkv_base + query * d_model * 3u + head_offset;
            vx_u32 key_base = qkv_base + key * d_model * 3u + d_model +
                head_offset;
            for (vx_u32 dimension = 0; dimension < head_dim; dimension++)
                score += qkv[query_base + dimension] *
                    qkv[key_base + dimension];
            vx_u32 value_index = qkv_base + key * d_model * 3u +
                2u * d_model + head_offset + channel;
            value += (vx_exp(score * scale - maximum) / denominator) *
                qkv[value_index];
        }
        output[destination + channel] = value;
    }
}
