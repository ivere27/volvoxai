/* This is the native training RNG contract from
 * training_forward_dropout.inc. Unsigned 32-bit arithmetic deliberately wraps
 * at every multiply so forward and backward reconstruct the same mask without
 * storing one. */
static VX_CUDA_DEVICE vx_u32 vx_cuda_training_dropout_bits(
        vx_u32 seed, vx_u32 counter, vx_u32 index) {
    vx_u32 value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

/* Accumulate one microbatch gradient into an existing F32 accumulation
 * buffer. Host scheduling and training-graph integration are intentionally
 * outside this initial profile-safe code-generation slice. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_accumulate_f32(
        const float* gradient, float* accumulator, vx_u32 elements) {
    vx_u32 index = vx_training_global_x();
    if (index < elements) accumulator[index] += gradient[index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_dropout_f32(
        const float* input, float* output, vx_u32 elements, vx_u32 threshold,
        vx_u32 seed, vx_u32 counter, float scale) {
    vx_u32 index = vx_training_global_x();
    if (index < elements) {
        output[index] = vx_cuda_training_dropout_bits(seed, counter, index) >=
            threshold ? input[index] * scale : 0.0f;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_dropout_backward_f32(
        const float* grad_output, float* grad_input, vx_u32 elements,
        vx_u32 threshold, vx_u32 seed, vx_u32 counter, float scale) {
    vx_u32 index = vx_training_global_x();
    if (index < elements &&
        vx_cuda_training_dropout_bits(seed, counter, index) >= threshold) {
        grad_input[index] += grad_output[index] * scale;
    }
