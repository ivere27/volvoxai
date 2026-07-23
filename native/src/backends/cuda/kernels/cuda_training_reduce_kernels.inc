}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_reduce_backward_f32(
        const float* grad_output, float* grad_input, vx_u32 elements,
        vx_u32 row_width, float scale) {
    vx_u32 index = vx_training_global_x();
    if (index < elements)
        grad_input[index] += grad_output[index / row_width] * scale;
}
