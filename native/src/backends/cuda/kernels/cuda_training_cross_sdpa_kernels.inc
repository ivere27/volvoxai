static VX_CUDA_DEVICE int vx_cuda_training_cross_sdpa_key_allowed(
        const int* mask,
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 query, vx_u32 key) {
    return vx_cuda_training_attention_key_allowed(mask, batch, query, key,
        params->seq_q, params->seq_kv, params->causal, params->mask_mode);
}

static VX_CUDA_DEVICE float vx_cuda_training_cross_sdpa_dropout_multiplier(
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 head, vx_u32 query, vx_u32 key) {
    return vx_cuda_training_attention_dropout_multiplier(batch, head, query,
        key, params->num_heads, params->seq_q, params->seq_kv,
        params->dropout_threshold, params->dropout_seed,
        params->dropout_counter, params->dropout_scale);
}

static VX_CUDA_DEVICE float vx_cuda_training_cross_sdpa_q_value(
        const float* q_input,
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 query, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_q * params->d_model;
    return q_input[base + query * params->d_model +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE float vx_cuda_training_cross_sdpa_k_value(
        const float* k_input,
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 key, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_kv * params->d_model;
    return k_input[base + key * params->d_model +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE float vx_cuda_training_cross_sdpa_v_value(
        const float* v_input,
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 key, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_kv * params->d_model;
    return v_input[base + key * params->d_model +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE float vx_cuda_training_cross_sdpa_score(
        const float* q_input, const float* k_input,
        const VxCudaTrainingCrossSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 query, vx_u32 key, vx_u32 head) {
    float value = 0.0f;
    for (vx_u32 dimension = 0u; dimension < params->head_dim; ++dimension) {
        value = value + vx_cuda_training_cross_sdpa_q_value(q_input, params,
            batch, query, head, dimension) *
            vx_cuda_training_cross_sdpa_k_value(k_input, params, batch, key,
                                                head, dimension);
    }
    return value * params->scale;
}

extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_cross_sdpa_probabilities_f32(
        const float* q_input, const float* k_input, const int* mask,
        float* probabilities,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    VX_CUDA_TRAINING_ATTENTION_SHARED float reduction[64];
    VX_CUDA_TRAINING_ATTENTION_SHARED vx_u32 visible_reduction[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_q || head >= params->num_heads ||
        batch >= params->batch) return;

    vx_u32 visible = 0u;
    float local_maximum = -3.402823e38f;
    for (vx_u32 key = lane; key < params->seq_kv;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        probabilities[index] = 0.0f;
        if (!vx_cuda_training_cross_sdpa_key_allowed(mask, params, batch,
                                                     query, key)) continue;
        float value = vx_cuda_training_cross_sdpa_score(q_input, k_input,
            params, batch, query, key, head);
        probabilities[index] = value;
        local_maximum = local_maximum > value ? local_maximum : value;
        ++visible;
    }
    reduction[lane] = local_maximum;
    visible_reduction[lane] = visible;
    vx_cuda_training_attention_sync_threads();
    for (vx_u32 offset = VX_CUDA_TRAINING_ATTENTION_LANES / 2u;
         offset != 0u; offset >>= 1u) {
        if (lane < offset) {
            float other = reduction[lane + offset];
            reduction[lane] = reduction[lane] > other
                ? reduction[lane] : other;
            visible_reduction[lane] += visible_reduction[lane + offset];
        }
        vx_cuda_training_attention_sync_threads();
    }
    if (visible_reduction[0] == 0u) return;
    float maximum = reduction[0];

    float local_denominator = 0.0f;
    for (vx_u32 key = lane; key < params->seq_kv;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        if (!vx_cuda_training_cross_sdpa_key_allowed(mask, params, batch,
                                                     query, key)) continue;
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        float exponential = vx_cuda_training_attention_exp(
            probabilities[index] - maximum);
        probabilities[index] = exponential;
        local_denominator = local_denominator + exponential;
    }
    reduction[lane] = local_denominator;
    vx_cuda_training_attention_sync_threads();
    for (vx_u32 offset = VX_CUDA_TRAINING_ATTENTION_LANES / 2u;
         offset != 0u; offset >>= 1u) {
        if (lane < offset)
            reduction[lane] = reduction[lane] + reduction[lane + offset];
        vx_cuda_training_attention_sync_threads();
    }
    float denominator = reduction[0];
    if (denominator == 0.0f) {
        for (vx_u32 key = lane; key < params->seq_kv;
             key += VX_CUDA_TRAINING_ATTENTION_LANES) {
            probabilities[vx_cuda_training_attention_probability_index(
                batch, head, query, key, params->num_heads, params->seq_q,
                params->seq_kv)] = 0.0f;
        }
        return;
    }

    for (vx_u32 key = lane; key < params->seq_kv;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        if (!vx_cuda_training_cross_sdpa_key_allowed(mask, params, batch,
                                                     query, key)) {
            probabilities[index] = 0.0f;
            continue;
        }
        probabilities[index] = probabilities[index] / denominator;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_sdpa_dscore_f32(
        const float* v_input, const float* grad_output,
        const float* probabilities, float* dscore,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    VX_CUDA_TRAINING_ATTENTION_SHARED float reduction[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_q || head >= params->num_heads ||
        batch >= params->batch) return;

    vx_u32 output_offset = batch * params->seq_q * params->d_model +
        query * params->d_model + head * params->head_dim;
    float local_expectation = 0.0f;
    for (vx_u32 key = lane; key < params->seq_kv;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        float probability = probabilities[index];
        float direct = 0.0f;
        if (probability != 0.0f) {
            for (vx_u32 dimension = 0u; dimension < params->head_dim;
                 ++dimension) {
                direct = direct + grad_output[output_offset + dimension] *
                    vx_cuda_training_cross_sdpa_v_value(v_input, params,
                        batch, key, head, dimension);
            }
            direct = direct *
                vx_cuda_training_cross_sdpa_dropout_multiplier(
                    params, batch, head, query, key);
        }
        dscore[index] = direct;
        local_expectation = local_expectation + probability * direct;
    }
    reduction[lane] = local_expectation;
    vx_cuda_training_attention_sync_threads();
    for (vx_u32 offset = VX_CUDA_TRAINING_ATTENTION_LANES / 2u;
         offset != 0u; offset >>= 1u) {
        if (lane < offset)
            reduction[lane] = reduction[lane] + reduction[lane + offset];
        vx_cuda_training_attention_sync_threads();
    }
    float expectation = reduction[0];

    for (vx_u32 key = lane; key < params->seq_kv;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        float probability = probabilities[index];
        dscore[index] = probability == 0.0f
            ? 0.0f : probability * (dscore[index] - expectation);
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_sdpa_backward_q_f32(
        const float* q_input, const float* k_input, const float* v_input,
        const int* mask, const float* grad_output, float* grad_q,
        float* grad_k, float* grad_v, const float* probabilities,
        const float* dscore,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();

    (void)q_input;
    (void)v_input;
    (void)mask;
    (void)grad_output;
    (void)grad_k;
    (void)grad_v;
    (void)probabilities;
    if (query >= params->seq_q || head >= params->num_heads ||
        batch >= params->batch) return;

    if (dimension >= params->head_dim) return;
    float gradient = 0.0f;
    for (vx_u32 key = 0u; key < params->seq_kv; ++key) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        gradient = gradient + dscore[index] *
            vx_cuda_training_cross_sdpa_k_value(k_input, params, batch,
                                                key, head, dimension) *
            params->scale;
    }
    vx_u32 offset = batch * params->seq_q * params->d_model +
        query * params->d_model + head * params->head_dim + dimension;
    grad_q[offset] = grad_q[offset] + gradient;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_sdpa_backward_k_f32(
        const float* q_input, const float* k_input, const float* v_input,
        const int* mask, const float* grad_output, float* grad_q,
        float* grad_k, float* grad_v, const float* probabilities,
        const float* dscore,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 key = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();

    (void)k_input;
    (void)v_input;
    (void)mask;
    (void)grad_output;
    (void)grad_q;
    (void)grad_v;
    (void)probabilities;
    if (key >= params->seq_kv || head >= params->num_heads ||
        batch >= params->batch) return;

    if (dimension >= params->head_dim) return;
    float gradient = 0.0f;
    for (vx_u32 query = 0u; query < params->seq_q; ++query) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        gradient = gradient + dscore[index] *
            vx_cuda_training_cross_sdpa_q_value(q_input, params, batch,
                                                query, head, dimension) *
            params->scale;
    }
    vx_u32 offset = batch * params->seq_kv * params->d_model +
        key * params->d_model + head * params->head_dim + dimension;
    grad_k[offset] = grad_k[offset] + gradient;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_cross_sdpa_backward_v_f32(
        const float* q_input, const float* k_input, const float* v_input,
        const int* mask, const float* grad_output, float* grad_q,
        float* grad_k, float* grad_v, const float* probabilities,
        const float* dscore,
        const VxCudaTrainingCrossSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 key = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();

    (void)q_input;
    (void)k_input;
    (void)v_input;
    (void)mask;
    (void)grad_q;
    (void)grad_k;
    (void)dscore;
    if (key >= params->seq_kv || head >= params->num_heads ||
        batch >= params->batch) return;

    if (dimension >= params->head_dim) return;
    float gradient = 0.0f;
    for (vx_u32 query = 0u; query < params->seq_q; ++query) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_q,
            params->seq_kv);
        gradient = gradient + probabilities[index] *
            vx_cuda_training_cross_sdpa_dropout_multiplier(params, batch,
                head, query, key) *
            grad_output[batch * params->seq_q * params->d_model +
                query * params->d_model + head * params->head_dim +
                dimension];
    }
    vx_u32 offset = batch * params->seq_kv * params->d_model +
        key * params->d_model + head * params->head_dim + dimension;
    grad_v[offset] = grad_v[offset] + gradient;
}
