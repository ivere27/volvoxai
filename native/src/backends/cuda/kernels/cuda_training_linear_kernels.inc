/*
 * Manual CUDA Driver API/PTX kernels for the matMulBackward training
 * contract. This file is included by cuda_training_kernels.cu after the
 * freestanding CUDA macros and vx_u32 have been defined.
 *
 * Every entry point intentionally has the same argument layout. The training
 * scheduler can therefore reuse the seven matMulBackward bindings and unpack
 * the eight-u32 Params block once for all three commands. Each kernel ignores
 * the buffers and flags that its command does not consume.
 */
#ifndef VOLVOXAI_CUDA_TRAINING_LINEAR_KERNELS_INC
#define VOLVOXAI_CUDA_TRAINING_LINEAR_KERNELS_INC

#if defined(__clang__)
#define VX_CUDA_TRAINING_LINEAR_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_LINEAR_SHARED __shared__
#endif

static VX_CUDA_DEVICE void vx_cuda_training_linear_sync_threads() {
    asm volatile("bar.sync 0;" : : : "memory");
}

/* dX = dY * W.
 *
 * Required launch: block=(16,16,1), grid=(ceil(d_in/16),
 * ceil(rows/16),1). This is the planner's input_main 16x16 [row,d_in]
 * output-tile contract. Shared 16-wide dY/W tiles retain strictly ascending
 * reduction order. One CUDA thread owns one dX element, so no atomics are
 * needed when adding this command's contribution to grad_input. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_matmul_backward_input_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 has_bias,
        vx_u32 canonical_din_layout, vx_u32 weight_din_layout) {
    VX_CUDA_TRAINING_LINEAR_SHARED float grad_tile[16][17];
    VX_CUDA_TRAINING_LINEAR_SHARED float weight_tile[16][17];
    vx_u32 local_d = threadIdx.x;
    vx_u32 local_row = threadIdx.y;
    vx_u32 d_base = blockIdx.x * 16u;
    vx_u32 d = d_base + local_d;
    vx_u32 row = blockIdx.y * 16u + local_row;
    float sum = 0.0f;

    (void)input;
    (void)grad_weight;
    (void)grad_bias;
    (void)has_bias;
    (void)canonical_din_layout;
    for (vx_u32 column_base = 0u; column_base < d_out;
         column_base += 16u) {
        vx_u32 grad_column = column_base + local_d;
        grad_tile[local_row][local_d] =
            row < rows && grad_column < d_out
            ? grad_output[row * d_out + grad_column] : 0.0f;
        if (weight_din_layout != 0u) {
            vx_u32 load_d = d_base + local_row;
            vx_u32 load_column = column_base + local_d;
            weight_tile[local_d][local_row] =
                load_d < d_in && load_column < d_out
                ? weight[load_d * d_out + load_column] : 0.0f;
        } else {
            vx_u32 load_column = column_base + local_row;
            vx_u32 load_d = d_base + local_d;
            weight_tile[local_row][local_d] =
                load_column < d_out && load_d < d_in
                ? weight[load_column * d_in + load_d] : 0.0f;
        }
        vx_cuda_training_linear_sync_threads();

        vx_u32 tile_width = d_out - column_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 column = 0u; column < tile_width; column++)
            sum += grad_tile[local_row][column] *
                   weight_tile[column][local_d];
        vx_cuda_training_linear_sync_threads();
    }
    if (row < rows && d < d_in)
        grad_input[row * d_in + d] += sum;
}

/* dW = X^T * dY.
 *
 * Required launch: block=(16,16,1), grid=(ceil(d_out/16),
 * ceil(d_in/16),1). This is the planner's weight_main 16x16 [d_in,d_out]
 * output-tile contract. Shared transposed-X and dY tiles retain strictly
 * ascending row order. canonical_din_layout selects the destination layout
 * independently of the source-weight layout used by input_main. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_matmul_backward_weight_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 has_bias,
        vx_u32 canonical_din_layout, vx_u32 weight_din_layout) {
    VX_CUDA_TRAINING_LINEAR_SHARED float input_tile[16][17];
    VX_CUDA_TRAINING_LINEAR_SHARED float grad_tile[16][17];
    vx_u32 local_column = threadIdx.x;
    vx_u32 local_d = threadIdx.y;
    vx_u32 column = blockIdx.x * 16u + local_column;
    vx_u32 d_base = blockIdx.y * 16u;
    vx_u32 d = d_base + local_d;
    float sum = 0.0f;

    (void)weight;
    (void)grad_input;
    (void)grad_bias;
    (void)has_bias;
    (void)weight_din_layout;
    for (vx_u32 row_base = 0u; row_base < rows; row_base += 16u) {
        vx_u32 load_row = row_base + local_d;
        vx_u32 load_d = d_base + local_column;
        input_tile[local_column][local_d] =
            load_row < rows && load_d < d_in
            ? input[load_row * d_in + load_d] : 0.0f;
        grad_tile[local_d][local_column] =
            load_row < rows && column < d_out
            ? grad_output[load_row * d_out + column] : 0.0f;
        vx_cuda_training_linear_sync_threads();

        vx_u32 tile_width = rows - row_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 row_offset = 0u; row_offset < tile_width;
             row_offset++)
            sum += input_tile[local_d][row_offset] *
                   grad_tile[row_offset][local_column];
        vx_cuda_training_linear_sync_threads();
    }
    if (d < d_in && column < d_out) {
        vx_u32 canonical = column * d_in + d;
        vx_u32 destination = canonical_din_layout != 0u
            ? d * d_out + column
            : canonical;
        grad_weight[destination] += sum;
    }
}

/* db = row_sum(dY).
 *
 * Required launch: block=(64,1,1), grid=(ceil(d_out/64),1,1). This matches
 * bias_main's 64 logical lanes per planner group. One CUDA thread owns one
 * output channel and the command remains a no-op when has_bias is zero. */
extern "C" VX_CUDA_GLOBAL void
vx_cuda_training_matmul_backward_bias_f32(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, float* grad_weight, float* grad_bias,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 has_bias,
        vx_u32 canonical_din_layout, vx_u32 weight_din_layout) {
    vx_u32 column = blockIdx.x * 64u + threadIdx.x;
    float sum = 0.0f;

    (void)input;
    (void)weight;
    (void)grad_input;
    (void)grad_weight;
    (void)d_in;
    (void)canonical_din_layout;
    (void)weight_din_layout;
    if (column >= d_out || has_bias == 0u) return;
    for (vx_u32 row = 0u; row < rows; ++row)
        sum += grad_output[row * d_out + column];
    grad_bias[column] += sum;
}

#undef VX_CUDA_TRAINING_LINEAR_SHARED

#endif
