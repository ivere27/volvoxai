extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sdpa_forward_f32(
        const float* qkv, const int* mask, float* output,
        const float* probabilities,
        const VxCudaTrainingSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_len || head >= params->num_heads ||
        batch >= params->batch) return;
    (void)mask;
    if (dimension >= params->head_dim) return;
    vx_u32 output_base = batch * params->seq_len * params->d_model +
        query * params->d_model + head * params->head_dim;
    float value = 0.0f;
    for (vx_u32 key = 0u; key < params->seq_len; ++key) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads,
            params->seq_len, params->seq_len);
        value = value + probabilities[index] *
            vx_cuda_training_sdpa_dropout_multiplier(
                params, batch, head, query, key) *
            vx_cuda_training_sdpa_v_value(
                qkv, params, batch, key, head, dimension);
    }
    output[output_base + dimension] = value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_sdpa_forward_f32(
        const float* q_input, const float* k_input, const float* v_input,
        const int* mask, float* output, const float* probabilities,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_q || head >= params->num_heads ||
        batch >= params->batch) return;
    (void)q_input;
    (void)k_input;
    (void)mask;
    if (dimension >= params->head_dim) return;
    vx_u32 output_base = batch * params->seq_q * params->d_model +
        query * params->d_model + head * params->head_dim;
    float value = 0.0f;
    for (vx_u32 key = 0u; key < params->seq_kv; ++key) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        value = value + probabilities[index] *
            vx_cuda_training_cross_sdpa_dropout_multiplier(
                params, batch, head, query, key) *
            vx_cuda_training_cross_sdpa_v_value(
                v_input, params, batch, key, head, dimension);
    }
    output[output_base + dimension] = value;
}
