
static VX_CUDA_DEVICE void vx_cuda_training_basic_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_basic_operand_index(
        vx_u32 output_index, vx_u32 stride_base, const vx_u32* params) {
    vx_u32 remainder = output_index;
    vx_u32 index = 0u;
    for (vx_u32 dimension = 0u; dimension < params[0]; ++dimension) {
        vx_u32 output_stride = params[8u + dimension];
        vx_u32 coordinate = remainder / output_stride;
        remainder %= output_stride;
        index += coordinate * params[stride_base + dimension];
    }
    return index;
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_basic_identity_map(
        vx_u32 stride_base, vx_u32 operand_length, const vx_u32* params) {
    if (operand_length != params[1]) return 0u;
    for (vx_u32 dimension = 0u; dimension < params[0]; ++dimension) {
        vx_u32 output_stride = params[8u + dimension];
        vx_u32 preceding_extent = dimension == 0u
            ? params[1] : params[8u + dimension - 1u];
        if (output_stride == 0u ||
            preceding_extent % output_stride != 0u) return 0u;
        /* Broadcast-stride construction uses zero for size-one dimensions,
         * even when the operand and output lengths are identical. Their
         * coordinate is always zero, so the differing stride is immaterial. */
        if (preceding_extent / output_stride > 1u &&
            params[stride_base + dimension] != output_stride) return 0u;
    }
    return 1u;
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_training_basic_scalar_map(
        vx_u32 stride_base, vx_u32 operand_length, const vx_u32* params) {
    if (operand_length != 1u) return 0u;
    for (vx_u32 dimension = 0u; dimension < params[0]; ++dimension) {
        vx_u32 output_stride = params[8u + dimension];
        vx_u32 preceding_extent = dimension == 0u
            ? params[1] : params[8u + dimension - 1u];
        if (output_stride == 0u ||
            preceding_extent % output_stride != 0u) return 0u;
        if (preceding_extent / output_stride > 1u &&
            params[stride_base + dimension] != 0u) return 0u;
    }
    return 1u;
}

static VX_CUDA_DEVICE float vx_cuda_training_basic_a_derivative(
        vx_u32 output_index, const float* b, const vx_u32* params) {
    vx_u32 kind = params[4];
    vx_u32 b_index;
    if (kind == 1u) {
        b_index = vx_cuda_training_basic_operand_index(
            output_index, 24u, params);
        return b[b_index];
    }
    if (kind == 3u) {
        b_index = vx_cuda_training_basic_operand_index(
            output_index, 24u, params);
        return 1.0f / b[b_index];
    }
    return 1.0f;
}

static VX_CUDA_DEVICE float vx_cuda_training_basic_b_derivative(
        vx_u32 output_index, const float* a, const float* b,
        const vx_u32* params) {
    vx_u32 kind = params[4];
    vx_u32 a_index;
    vx_u32 b_index;
    if (kind == 1u) {
        a_index = vx_cuda_training_basic_operand_index(
            output_index, 16u, params);
        return a[a_index];
    }
    if (kind == 2u) return -1.0f;
    if (kind == 3u) {
        a_index = vx_cuda_training_basic_operand_index(
            output_index, 16u, params);
        b_index = vx_cuda_training_basic_operand_index(
            output_index, 24u, params);
        return -a[a_index] / (b[b_index] * b[b_index]);
    }
    return 1.0f;
}

static VX_CUDA_DEVICE float vx_cuda_training_basic_activation_derivative(
        vx_u32 output_index, const float* a, const float* b,
        const vx_u32* params) {
    vx_u32 relu = params[5];
    if (params[4] != 0u || relu == 0u) return 1.0f;
    vx_u32 a_index = vx_cuda_training_basic_operand_index(
        output_index, 16u, params);
    vx_u32 b_index = vx_cuda_training_basic_operand_index(
        output_index, 24u, params);
    float value = a[a_index] + b[b_index];
    if (value <= 0.0f || (relu == 2u && value >= 6.0f)) return 0.0f;
    return 1.0f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_basic_backward_a_f32(
        const float* a, const float* b, const float* grad_output,
        float* grad_a, float* grad_b, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float partials[64];
    vx_u32 index = vx_training_global_x();
    float sum = 0.0f;

    (void)grad_b;
    if (vx_cuda_training_basic_scalar_map(16u, params[2], params)) {
        for (vx_u32 output_index = threadIdx.x;
             output_index < params[1];) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_a_derivative(
                    output_index, b, params);
            if (64u >= params[1] - output_index) break;
            output_index += 64u;
        }
        partials[threadIdx.x] = sum;
        vx_cuda_training_basic_sync_threads();
        for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
            if (threadIdx.x < stride)
                partials[threadIdx.x] += partials[threadIdx.x + stride];
            vx_cuda_training_basic_sync_threads();
        }
        if (threadIdx.x == 0u) grad_a[0] += partials[0];
        return;
    }
    if (index >= params[2]) return;
    if (vx_cuda_training_basic_identity_map(16u, params[2], params)) {
        sum += grad_output[index] *
            vx_cuda_training_basic_activation_derivative(
                index, a, b, params) *
            vx_cuda_training_basic_a_derivative(index, b, params);
        grad_a[index] += sum;
        return;
    }
    for (vx_u32 output_index = 0u; output_index < params[1]; ++output_index) {
        if (vx_cuda_training_basic_operand_index(
                output_index, 16u, params) == index) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_a_derivative(
                    output_index, b, params);
        }
    }
    grad_a[index] += sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_basic_backward_b_f32(
        const float* a, const float* b, const float* grad_output,
        float* grad_a, float* grad_b, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float partials[64];
    vx_u32 index = vx_training_global_x();
    float sum = 0.0f;

    (void)grad_a;
    if (vx_cuda_training_basic_scalar_map(24u, params[3], params)) {
        for (vx_u32 output_index = threadIdx.x;
             output_index < params[1];) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_b_derivative(
                    output_index, a, b, params);
            if (64u >= params[1] - output_index) break;
            output_index += 64u;
        }
        partials[threadIdx.x] = sum;
        vx_cuda_training_basic_sync_threads();
        for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
            if (threadIdx.x < stride)
                partials[threadIdx.x] += partials[threadIdx.x + stride];
            vx_cuda_training_basic_sync_threads();
        }
        if (threadIdx.x == 0u) grad_b[0] += partials[0];
        return;
    }
    if (index >= params[3]) return;
    if (vx_cuda_training_basic_identity_map(24u, params[3], params)) {
        sum += grad_output[index] *
            vx_cuda_training_basic_activation_derivative(
                index, a, b, params) *
            vx_cuda_training_basic_b_derivative(index, a, b, params);
        grad_b[index] += sum;
        return;
    }
    for (vx_u32 output_index = 0u; output_index < params[1]; ++output_index) {
        if (vx_cuda_training_basic_operand_index(
                output_index, 24u, params) == index) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_b_derivative(
                    output_index, a, b, params);
        }
    }
    grad_b[index] += sum;
}
