/*
 * Manual CUDA Driver API/PTX kernels for the layerNormBackward and
 * rmsNormBackward training contracts. This file is included by
 * cuda_training_kernels.cu after the freestanding CUDA macros, vx_u32, and
 * vx_training_global_x have been defined. It requires no CUDA headers,
 * runtime, or device libraries.
 *
 * Each entry point preserves its WGSL binding order and consumes the exact
 * 16-byte Params block emitted by the corresponding planner. Required launch
 * block size is (64,1,1) for every entry point.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_NORM_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_NORM_KERNELS_INC

#if defined(__clang__)
#define VX_CUDA_TRAINING_NORM_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_NORM_SHARED __shared__
#endif

static VX_CUDA_DEVICE void vx_cuda_training_norm_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

struct VxCudaTrainingLayerNormParams {
    vx_u32 rows;
    vx_u32 d_model;
    float epsilon;
    vx_u32 has_bias;
};

struct VxCudaTrainingRmsNormParams {
    vx_u32 rows;
    vx_u32 d_model;
    float epsilon;
    vx_u32 pad;
};

static_assert(sizeof(VxCudaTrainingLayerNormParams) == 16,
              "layerNormBackward Params must remain 16 bytes");
static_assert(sizeof(VxCudaTrainingRmsNormParams) == 16,
              "rmsNormBackward Params must remain 16 bytes");

static VX_CUDA_DEVICE float vx_cuda_training_norm_sqrt(float value) {
    float result;
    asm("sqrt.rn.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}

struct VxCudaTrainingLayerNormStats {
    float mean;
    float inverse_standard_deviation;
};

static VX_CUDA_DEVICE VxCudaTrainingLayerNormStats
vx_cuda_training_layernorm_row_stats(
        const float* input, vx_u32 row,
        const VxCudaTrainingLayerNormParams* params) {
    vx_u32 offset = row * params->d_model;
    float sum = 0.0f;
    float square_sum = 0.0f;

    for (vx_u32 feature = 0u; feature < params->d_model; ++feature) {
        float value = input[offset + feature];
        sum = sum + value;
        square_sum = square_sum + value * value;
    }

    float width = (float)params->d_model;
    float mean = sum / width;
    float variance = square_sum / width - mean * mean;
    if (variance < 0.0f) variance = 0.0f;

    VxCudaTrainingLayerNormStats stats;
    stats.mean = mean;
    stats.inverse_standard_deviation =
        1.0f / vx_cuda_training_norm_sqrt(variance + params->epsilon);
    return stats;
}

/* layerNormBackward.input_main
 *
 * grid=(ceil((rows*d_model)/64),1,1). One thread owns one grad_input
 * element and reproduces the WGSL row reductions in feature order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_layernorm_backward_input_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingLayerNormParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 total = params->rows * params->d_model;

    (void)grad_weight;
    (void)grad_bias;
    if (flat >= total) return;

    vx_u32 row = flat / params->d_model;
    vx_u32 feature = flat - row * params->d_model;
    vx_u32 offset = row * params->d_model;
    VxCudaTrainingLayerNormStats stats =
        vx_cuda_training_layernorm_row_stats(input, row, params);
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;

    for (vx_u32 d = 0u; d < params->d_model; ++d) {
        float xhat = (input[offset + d] - stats.mean) *
            stats.inverse_standard_deviation;
        float scaled_gradient = grad_output[offset + d] * weight[d];
        sum_dy = sum_dy + scaled_gradient;
        sum_dy_xhat = sum_dy_xhat + scaled_gradient * xhat;
    }

    float width = (float)params->d_model;
    float xhat = (input[flat] - stats.mean) *
        stats.inverse_standard_deviation;
    float dx = stats.inverse_standard_deviation / width *
        (width * grad_output[flat] * weight[feature] - sum_dy -
         xhat * sum_dy_xhat);
    grad_input[flat] += dx;
}

/* layerNormBackward.param_main
 *
 * grid=(ceil(d_model/64),1,1). One thread owns one weight feature and, when
 * has_bias is nonzero, the matching bias feature. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_layernorm_backward_param_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingLayerNormParams* params) {
    vx_u32 feature = vx_training_global_x();
    float weight_gradient = 0.0f;
    float bias_gradient = 0.0f;

    (void)weight;
    (void)grad_input;
    if (feature >= params->d_model) return;

    for (vx_u32 row = 0u; row < params->rows; ++row) {
        VxCudaTrainingLayerNormStats stats =
            vx_cuda_training_layernorm_row_stats(input, row, params);
        vx_u32 offset = row * params->d_model + feature;
        weight_gradient = weight_gradient + grad_output[offset] *
            (input[offset] - stats.mean) *
            stats.inverse_standard_deviation;
        bias_gradient = bias_gradient + grad_output[offset];
    }

    grad_weight[feature] += weight_gradient;
    if (params->has_bias != 0u)
        grad_bias[feature] += bias_gradient;
}

/* layerNormBackward.param_cooperative
 *
 * grid=(d_model,1,1). One 64-lane block owns one affine feature. Each lane
 * scans an ascending stride-64 row subsequence and computes the row statistics
 * in feature order. Fixed shared-memory trees reduce the weight and optional
 * bias gradients without atomics or external scratch. The result is additive
 * and bit-stable for a fixed launch. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_layernorm_backward_param_cooperative_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        const VxCudaTrainingLayerNormParams* params) {
    VX_CUDA_TRAINING_NORM_SHARED float weight_partials[64];
    VX_CUDA_TRAINING_NORM_SHARED float bias_partials[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 feature = blockIdx.x;
    float weight_gradient = 0.0f;
    float bias_gradient = 0.0f;

    (void)weight;
    (void)grad_input;
    if (feature >= params->d_model) return;

    for (vx_u32 row = lane; row < params->rows;) {
        VxCudaTrainingLayerNormStats stats =
            vx_cuda_training_layernorm_row_stats(input, row, params);
        vx_u32 offset = row * params->d_model + feature;
        weight_gradient += grad_output[offset] *
            (input[offset] - stats.mean) *
            stats.inverse_standard_deviation;
        bias_gradient += grad_output[offset];
        if (64u >= params->rows - row) break;
        row += 64u;
    }

    weight_partials[lane] = weight_gradient;
    bias_partials[lane] = bias_gradient;
    vx_cuda_training_norm_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            weight_partials[lane] += weight_partials[lane + stride];
            bias_partials[lane] += bias_partials[lane + stride];
        }
        vx_cuda_training_norm_sync_threads();
    }
    if (lane == 0u) {
        grad_weight[feature] += weight_partials[0];
        if (params->has_bias != 0u)
            grad_bias[feature] += bias_partials[0];
    }
}

static VX_CUDA_DEVICE float vx_cuda_training_rmsnorm_inverse_rms(
        const float* input, vx_u32 row,
        const VxCudaTrainingRmsNormParams* params) {
    vx_u32 offset = row * params->d_model;
    float square_sum = 0.0f;

    for (vx_u32 feature = 0u; feature < params->d_model; ++feature) {
        float value = input[offset + feature];
        square_sum = square_sum + value * value;
    }

    return 1.0f / vx_cuda_training_norm_sqrt(
        square_sum / (float)params->d_model + params->epsilon);
}

/* rmsNormBackward.input_main
 *
 * grid=(ceil((rows*d_model)/64),1,1). One thread owns one grad_input
 * element and reproduces the WGSL row dot product in feature order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_rmsnorm_backward_input_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight,
        const VxCudaTrainingRmsNormParams* params) {
    vx_u32 flat = vx_training_global_x();
    vx_u32 total = params->rows * params->d_model;

    (void)grad_weight;
    if (flat >= total) return;

    vx_u32 row = flat / params->d_model;
    vx_u32 feature = flat - row * params->d_model;
    vx_u32 offset = row * params->d_model;
    float inverse_rms =
        vx_cuda_training_rmsnorm_inverse_rms(input, row, params);
    float dot = 0.0f;

    for (vx_u32 d = 0u; d < params->d_model; ++d) {
        dot = dot + grad_output[offset + d] * weight[d] *
            input[offset + d];
    }

    float dx = grad_output[flat] * weight[feature] * inverse_rms -
        input[flat] * inverse_rms * inverse_rms * inverse_rms * dot /
            (float)params->d_model;
    grad_input[flat] += dx;
}

/* rmsNormBackward.weight_main
 *
 * grid=(ceil(d_model/64),1,1). One thread owns one grad_weight feature. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_rmsnorm_backward_weight_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight,
        const VxCudaTrainingRmsNormParams* params) {
    vx_u32 feature = vx_training_global_x();
    float sum = 0.0f;

    (void)weight;
    (void)grad_input;
    if (feature >= params->d_model) return;

    for (vx_u32 row = 0u; row < params->rows; ++row) {
        vx_u32 offset = row * params->d_model + feature;
        sum = sum + grad_output[offset] * input[offset] *
            vx_cuda_training_rmsnorm_inverse_rms(input, row, params);
    }

    grad_weight[feature] += sum;
}

#undef VX_CUDA_TRAINING_NORM_SHARED

#endif
