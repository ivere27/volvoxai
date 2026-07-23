/* CUDA-only small-broadcast tactics. The planner selects these explicit
 * entry points only for validated non-scalar destinations with at most 1024
 * elements. grid=(destination_length,1,1), block=(64,1,1): one block owns
 * one destination, lanes scan ascending stride-64 output subsequences, and a
 * fixed tree performs the final deterministic reduction without atomics. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_basic_backward_a_cooperative_f32(
        const float* a, const float* b, const float* grad_output,
        float* grad_a, float* grad_b, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float partials[64];
    vx_u32 destination = blockIdx.x;
    vx_u32 lane = threadIdx.x;
    float sum = 0.0f;

    (void)grad_b;
    if (destination >= params[2]) return;
    for (vx_u32 output_index = lane;
         output_index < params[1];) {
        if (vx_cuda_training_basic_operand_index(
                output_index, 16u, params) == destination) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_a_derivative(
                    output_index, b, params);
        }
        if (64u >= params[1] - output_index) break;
        output_index += 64u;
    }
    partials[lane] = sum;
    vx_cuda_training_basic_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        vx_cuda_training_basic_sync_threads();
    }
    if (lane == 0u) grad_a[destination] += partials[0];
}

extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_basic_backward_b_cooperative_f32(
        const float* a, const float* b, const float* grad_output,
        float* grad_a, float* grad_b, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float partials[64];
    vx_u32 destination = blockIdx.x;
    vx_u32 lane = threadIdx.x;
    float sum = 0.0f;

    (void)grad_a;
    if (destination >= params[3]) return;
    for (vx_u32 output_index = lane;
         output_index < params[1];) {
        if (vx_cuda_training_basic_operand_index(
                output_index, 24u, params) == destination) {
            sum += grad_output[output_index] *
                vx_cuda_training_basic_activation_derivative(
                    output_index, a, b, params) *
                vx_cuda_training_basic_b_derivative(
                    output_index, a, b, params);
        }
        if (64u >= params[1] - output_index) break;
        output_index += 64u;
    }
    partials[lane] = sum;
    vx_cuda_training_basic_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride) partials[lane] += partials[lane + stride];
        vx_cuda_training_basic_sync_threads();
    }
    if (lane == 0u) grad_b[destination] += partials[0];
}

/* CUDA-only staged scalar tactics. The host supplies
 * grid=(ceil(output_length/4096),1,1), block=(64,1,1) and a private workspace
 * containing one float per block. Each block visits its deterministic
 * interleaved output subsequences and writes one fixed-tree partial. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_basic_backward_a_scalar_partials_f32(
        const float* a, const float* b, const float* grad_output,
        float* partials, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float block_partials[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 partial = blockIdx.x;
    vx_u32 output_index = partial * 64u + lane;
    vx_u32 output_stride = gridDim.x * 64u;
    float sum = 0.0f;

    while (output_index < params[1]) {
        sum += grad_output[output_index] *
            vx_cuda_training_basic_activation_derivative(
                output_index, a, b, params) *
            vx_cuda_training_basic_a_derivative(output_index, b, params);
        if (output_stride >= params[1] - output_index) break;
        output_index += output_stride;
    }
    block_partials[lane] = sum;
    vx_cuda_training_basic_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride)
            block_partials[lane] += block_partials[lane + stride];
        vx_cuda_training_basic_sync_threads();
    }
    if (lane == 0u) partials[partial] = block_partials[0];
}

extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_basic_backward_b_scalar_partials_f32(
        const float* a, const float* b, const float* grad_output,
        float* partials, const vx_u32* params) {
    VX_CUDA_TRAINING_BASIC_SHARED float block_partials[64];
    vx_u32 lane = threadIdx.x;
    vx_u32 partial = blockIdx.x;
    vx_u32 output_index = partial * 64u + lane;
    vx_u32 output_stride = gridDim.x * 64u;
    float sum = 0.0f;

    while (output_index < params[1]) {
        sum += grad_output[output_index] *
            vx_cuda_training_basic_activation_derivative(
                output_index, a, b, params) *
            vx_cuda_training_basic_b_derivative(output_index, a, b, params);
        if (output_stride >= params[1] - output_index) break;
        output_index += output_stride;
    }
    block_partials[lane] = sum;
    vx_cuda_training_basic_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride)
            block_partials[lane] += block_partials[lane + stride];
        vx_cuda_training_basic_sync_threads();
    }
    if (lane == 0u) partials[partial] = block_partials[0];
}

/* The second stage is shared by scalar A and B. One block scans the ordered
 * workspace, applies a final fixed tree, and performs the required additive
 * update only after every first-stage block has completed in stream order. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_basic_backward_scalar_finalize_f32(
        const float* partials, float* gradient, vx_u32 partial_count) {
    VX_CUDA_TRAINING_BASIC_SHARED float final_partials[64];
    vx_u32 lane = threadIdx.x;
    float sum = 0.0f;

    for (vx_u32 partial = lane; partial < partial_count;) {
        sum += partials[partial];
        if (64u >= partial_count - partial) break;
        partial += 64u;
    }
    final_partials[lane] = sum;
    vx_cuda_training_basic_sync_threads();
    for (vx_u32 stride = 32u; stride != 0u; stride >>= 1u) {
        if (lane < stride)
            final_partials[lane] += final_partials[lane + stride];
        vx_cuda_training_basic_sync_threads();
    }
    if (lane == 0u) gradient[0] += final_partials[0];
}
