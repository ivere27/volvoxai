/* Adjacent OUT_IN Linear rows can share one input load without merging
 * model weights or outputs. Each sum keeps the ordinary GEMV lane partition,
 * shuffle tree and final +0; both complete dense owners remain observable. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_gemv_pair_out_in_f32(
        const float* input, const float* weight0, const float* weight1,
        float* output0, float* output1, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight0, weight1, output0, output1, rows, d_in, d_out);
    vx_u32 lane = threadIdx.x & 31u;
    vx_u32 column = blockIdx.x * 4u + (threadIdx.x >> 5u);
    float sum0 = 0.0f, sum1 = 0.0f;
    if (rows == 1u && column < d_out)
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#endif
        for (vx_u32 k = lane; k < d_in; k += 32u) {
            float value = input[k];
            sum0 += value * weight0[column * d_in + k];
            sum1 += value * weight1[column * d_in + k];
        }
    for (vx_u32 delta = 16u; delta != 0u; delta >>= 1u) {
        sum0 += vx_cuda_gemv_shuffle_down(sum0, delta);
        sum1 += vx_cuda_gemv_shuffle_down(sum1, delta);
    }
    if (rows == 1u && lane == 0u && column < d_out) {
        output0[column] = sum0 + 0.0f;
        output1[column] = sum1 + 0.0f;
    }
}
