struct VxCudaTrainingSdpaBackwardParams {
    vx_u32 seq_len;
    vx_u32 d_model;
    vx_u32 num_heads;
    vx_u32 head_dim;
    vx_u32 batch;
    float scale;
    vx_u32 causal;
    vx_u32 mask_mode;
    vx_u32 dropout_threshold;
    vx_u32 dropout_seed;
    vx_u32 dropout_counter;
    float dropout_scale;
};

struct VxCudaTrainingCrossSdpaBackwardParams {
    vx_u32 seq_q;
    vx_u32 seq_kv;
    vx_u32 d_model;
    vx_u32 num_heads;
    vx_u32 head_dim;
    vx_u32 batch;
    float scale;
    vx_u32 causal;
    vx_u32 mask_mode;
    vx_u32 dropout_threshold;
    vx_u32 dropout_seed;
    vx_u32 dropout_counter;
    float dropout_scale;
    vx_u32 pad0;
    vx_u32 pad1;
    vx_u32 pad2;
};

static_assert(sizeof(int) == 4, "attention masks require 32-bit int storage");
static_assert(sizeof(VxCudaTrainingSdpaBackwardParams) == 48,
              "sdpaBackward Params must remain 48 bytes");
static_assert(sizeof(VxCudaTrainingCrossSdpaBackwardParams) == 64,
              "crossSdpaBackward Params must remain 64 bytes");

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_attention_global_y() {
    return blockIdx.y * blockDim.y + threadIdx.y;
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_attention_global_z() {
    return blockIdx.z * blockDim.z + threadIdx.z;
}

static VX_CUDA_DEVICE void vx_cuda_training_attention_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

static VX_CUDA_DEVICE float vx_cuda_training_attention_exp(float value) {
    float result;

    /* ex2 is a PTX instruction, preserving the freestanding PTX contract.
     * The finite tails match the CUDA forward softmax range policy. */
    if (value < -87.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    value *= 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

static VX_CUDA_DEVICE int vx_cuda_training_attention_key_allowed(
        const int* mask, vx_u32 batch, vx_u32 query, vx_u32 key,
        vx_u32 seq_q, vx_u32 seq_kv, vx_u32 causal, vx_u32 mask_mode) {
    if (causal != 0u && key > query) return 0;
    if (mask_mode == 0u) return 1;

    vx_u32 index = key;
    if (mask_mode == 2u) {
        index = batch * seq_kv + key;
    } else if (mask_mode == 3u) {
        index = query * seq_kv + key;
    } else if (mask_mode == 4u) {
        index = (batch * seq_q + query) * seq_kv + key;
    }
    return mask[index] != 0;
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_attention_probability_index(
        vx_u32 batch, vx_u32 head, vx_u32 query, vx_u32 key,
        vx_u32 num_heads, vx_u32 seq_q, vx_u32 seq_kv) {
    return (((batch * num_heads + head) * seq_q + query) * seq_kv + key);
}

static VX_CUDA_DEVICE float vx_cuda_training_attention_dropout_multiplier(
        vx_u32 batch, vx_u32 head, vx_u32 query, vx_u32 key,
        vx_u32 num_heads, vx_u32 seq_q, vx_u32 seq_kv,
        vx_u32 threshold, vx_u32 seed, vx_u32 counter, float scale) {
    vx_u32 index = vx_cuda_training_attention_probability_index(
        batch, head, query, key, num_heads, seq_q, seq_kv);
    return vx_cuda_training_dropout_bits(seed, counter, index) >= threshold
        ? scale : 0.0f;
}

static VX_CUDA_DEVICE float vx_cuda_training_sdpa_q_value(
        const float* qkv, const VxCudaTrainingSdpaBackwardParams* params,
        vx_u32 batch, vx_u32 query, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_len * params->d_model * 3u;
    return qkv[base + query * (params->d_model * 3u) +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE float vx_cuda_training_sdpa_k_value(
        const float* qkv, const VxCudaTrainingSdpaBackwardParams* params,
        vx_u32 batch, vx_u32 key, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_len * params->d_model * 3u;
    return qkv[base + key * (params->d_model * 3u) + params->d_model +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE float vx_cuda_training_sdpa_v_value(
        const float* qkv, const VxCudaTrainingSdpaBackwardParams* params,
        vx_u32 batch, vx_u32 key, vx_u32 head, vx_u32 dimension) {
    vx_u32 base = batch * params->seq_len * params->d_model * 3u;
    return qkv[base + key * (params->d_model * 3u) + params->d_model * 2u +
        head * params->head_dim + dimension];
}

static VX_CUDA_DEVICE int vx_cuda_training_sdpa_key_allowed(
        const int* mask, const VxCudaTrainingSdpaBackwardParams* params,
        vx_u32 batch, vx_u32 query, vx_u32 key) {
    return vx_cuda_training_attention_key_allowed(mask, batch, query, key,
        params->seq_len, params->seq_len, params->causal,
        params->mask_mode);
}

static VX_CUDA_DEVICE float vx_cuda_training_sdpa_dropout_multiplier(
        const VxCudaTrainingSdpaBackwardParams* params, vx_u32 batch,
        vx_u32 head, vx_u32 query, vx_u32 key) {
    return vx_cuda_training_attention_dropout_multiplier(batch, head, query,
        key, params->num_heads, params->seq_len, params->seq_len,
        params->dropout_threshold, params->dropout_seed,
        params->dropout_counter, params->dropout_scale);
}

static VX_CUDA_DEVICE float vx_cuda_training_sdpa_score(
        const float* qkv, const VxCudaTrainingSdpaBackwardParams* params,
        vx_u32 batch, vx_u32 query, vx_u32 key, vx_u32 head) {
    float value = 0.0f;
    for (vx_u32 dimension = 0u; dimension < params->head_dim; ++dimension) {
        value = value + vx_cuda_training_sdpa_q_value(qkv, params, batch,
            query, head, dimension) * vx_cuda_training_sdpa_k_value(qkv,
                params, batch, key, head, dimension);
    }
    return value * params->scale;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sdpa_probabilities_f32(
        const float* qkv, const int* mask, float* probabilities,
        const VxCudaTrainingSdpaBackwardParams* params) {
    VX_CUDA_TRAINING_ATTENTION_SHARED float reduction[64];
    VX_CUDA_TRAINING_ATTENTION_SHARED vx_u32 visible_reduction[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_len || head >= params->num_heads ||
        batch >= params->batch) return;

    vx_u32 visible = 0u;
    float local_maximum = -3.402823e38f;
    for (vx_u32 key = lane; key < params->seq_len;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_len,
            params->seq_len);
        probabilities[index] = 0.0f;
        if (!vx_cuda_training_sdpa_key_allowed(mask, params, batch, query,
                                               key)) continue;
        float value = vx_cuda_training_sdpa_score(qkv, params, batch, query,
                                                   key, head);
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
    for (vx_u32 key = lane; key < params->seq_len;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        if (!vx_cuda_training_sdpa_key_allowed(mask, params, batch, query,
                                               key)) continue;
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_len,
            params->seq_len);
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
        for (vx_u32 key = lane; key < params->seq_len;
             key += VX_CUDA_TRAINING_ATTENTION_LANES) {
            probabilities[vx_cuda_training_attention_probability_index(
                batch, head, query, key, params->num_heads,
                params->seq_len, params->seq_len)] = 0.0f;
        }
        return;
    }

    for (vx_u32 key = lane; key < params->seq_len;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_len,
            params->seq_len);
        if (!vx_cuda_training_sdpa_key_allowed(mask, params, batch, query,
                                               key)) {
            probabilities[index] = 0.0f;
            continue;
        }
        probabilities[index] = probabilities[index] / denominator;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sdpa_dscore_f32(
        const float* qkv, const float* grad_output,
        const float* probabilities, float* dscore,
        const VxCudaTrainingSdpaBackwardParams* params) {
    VX_CUDA_TRAINING_ATTENTION_SHARED float reduction[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 query = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 batch = vx_cuda_training_attention_global_z();
    if (query >= params->seq_len || head >= params->num_heads ||
        batch >= params->batch) return;

    vx_u32 output_base = batch * params->seq_len * params->d_model;
    float local_expectation = 0.0f;
    for (vx_u32 key = lane; key < params->seq_len;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_len,
            params->seq_len);
        float probability = probabilities[index];
        float direct = 0.0f;
        if (probability != 0.0f) {
            for (vx_u32 dimension = 0u; dimension < params->head_dim;
                 ++dimension) {
                direct = direct +
                    grad_output[output_base + query * params->d_model +
                        head * params->head_dim + dimension] *
                    vx_cuda_training_sdpa_v_value(qkv, params, batch, key,
                                                  head, dimension);
            }
            direct = direct * vx_cuda_training_sdpa_dropout_multiplier(
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

    for (vx_u32 key = lane; key < params->seq_len;
         key += VX_CUDA_TRAINING_ATTENTION_LANES) {
        vx_u32 index = vx_cuda_training_attention_probability_index(
            batch, head, query, key, params->num_heads, params->seq_len,
            params->seq_len);
        float probability = probabilities[index];
        dscore[index] = probability == 0.0f
            ? 0.0f : probability * (dscore[index] - expectation);
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_sdpa_backward_f32(
        const float* qkv, const int* mask, const float* grad_output,
        float* grad_qkv, const float* probabilities, const float* dscore,
        const VxCudaTrainingSdpaBackwardParams* params) {
    vx_u32 dimension = threadIdx.x;
    vx_u32 position = blockIdx.x;
    vx_u32 head = vx_cuda_training_attention_global_y();
    vx_u32 z = vx_cuda_training_attention_global_z();
    vx_u32 batch = z / 3u;
    vx_u32 component = z % 3u;

    if (position >= params->seq_len || head >= params->num_heads ||
        batch >= params->batch) return;

    (void)mask;
    if (dimension >= params->head_dim) return;

    vx_u32 qkv_base = batch * params->seq_len * params->d_model * 3u;
    vx_u32 output_base = batch * params->seq_len * params->d_model;
    float gradient = 0.0f;
    if (component == 0u) {
        for (vx_u32 key = 0u; key < params->seq_len; ++key) {
            vx_u32 index = vx_cuda_training_attention_probability_index(
                batch, head, position, key, params->num_heads,
                params->seq_len, params->seq_len);
            gradient = gradient + dscore[index] *
                vx_cuda_training_sdpa_k_value(qkv, params, batch, key,
                                              head, dimension) *
                params->scale;
        }
    } else if (component == 1u) {
        for (vx_u32 query = 0u; query < params->seq_len; ++query) {
            vx_u32 index = vx_cuda_training_attention_probability_index(
                batch, head, query, position, params->num_heads,
                params->seq_len, params->seq_len);
            gradient = gradient + dscore[index] *
                vx_cuda_training_sdpa_q_value(qkv, params, batch, query,
                                              head, dimension) *
                params->scale;
        }
    } else {
        for (vx_u32 query = 0u; query < params->seq_len; ++query) {
            vx_u32 index = vx_cuda_training_attention_probability_index(
                batch, head, query, position, params->num_heads,
                params->seq_len, params->seq_len);
            gradient = gradient + probabilities[index] *
                vx_cuda_training_sdpa_dropout_multiplier(params, batch,
                    head, query, position) *
                grad_output[output_base + query * params->d_model +
                    head * params->head_dim + dimension];
        }
    }
    vx_u32 offset = qkv_base + position * (params->d_model * 3u) +
        component * params->d_model + head * params->head_dim + dimension;
    grad_qkv[offset] = grad_qkv[offset] + gradient;
}
