/* Canonical row-major BatchMatMul backward with right-aligned singleton
 * batch broadcasting. Transpose remains its own differentiable graph node.
 * Each block owns one destination matrix tile and reduces broadcast batches
 * in their original ascending output order. No gradient atomics are used. */
#if defined(__clang__)
#define VX_CUDA_TRAINING_BMM_SHARED __attribute__((shared))
#else
#define VX_CUDA_TRAINING_BMM_SHARED __shared__
#endif
struct VxCudaTrainingBatchMatMulParams {
    vx_u32 values[32];
};

/* Recover one matching output batch from a source matrix number and the
 * ordinal of its broadcast reduction. Strides are in source elements and
 * output batches, respectively. Every partial product was checked on host. */
static VX_CUDA_DEVICE void vx_cuda_training_bmm_batch(
        const VxCudaTrainingBatchMatMulParams& params, vx_u32 source_batch,
        vx_u32 reduced_batch, vx_u32 source_stride_base,
        vx_u32& output_batch, vx_u32& a_offset, vx_u32& b_offset) {
    const vx_u32* p = params.values;
    output_batch = a_offset = b_offset = 0u;
    for (int dim = (int)p[0] - 1; dim >= 0; dim--) {
        vx_u32 size = p[8 + dim], coordinate;
        if (p[source_stride_base + (vx_u32)dim]) {
            coordinate = source_batch % size;
            source_batch /= size;
        } else {
            coordinate = reduced_batch % size;
            reduced_batch /= size;
        }
        output_batch += coordinate * p[26 + dim];
        a_offset += coordinate * p[14 + dim];
        b_offset += coordinate * p[20 + dim];
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_batch_matmul_backward_a_f32(
        const float* a, const float* b, const float* dy, float* da, float* db,
        VxCudaTrainingBatchMatMulParams params) {
    VX_CUDA_TRAINING_BMM_SHARED float gy[16][17];
    VX_CUDA_TRAINING_BMM_SHARED float weight[16][17];
    const vx_u32* p = params.values;
    const vx_u32 m = p[1], k = p[2], n = p[3];
    const vx_u32 tx = threadIdx.x, ty = threadIdx.y;
    const vx_u32 column_tiles = (k - 1u) / 16u + 1u;
    const vx_u32 matrix_tiles = column_tiles * ((m - 1u) / 16u + 1u);
    const vx_u32 source_batch = blockIdx.x / matrix_tiles;
    const vx_u32 tile = blockIdx.x % matrix_tiles;
    const vx_u32 inner_base = (tile % column_tiles) * 16u;
    const vx_u32 row = (tile / column_tiles) * 16u + ty;
    const vx_u32 inner = inner_base + tx;
    const vx_u32 index = source_batch * (m * k) + row * k + inner;
    const bool valid = row < m && inner < k;
    const vx_u32 reductions = p[7] / (p[4] / (m * k));
    float accumulated = valid ? da[index] : 0.0f;
    (void)a; (void)db;
    for (vx_u32 reduced = 0; reduced < reductions; reduced++) {
        vx_u32 batch, a_offset, b_offset;
        vx_cuda_training_bmm_batch(params, source_batch, reduced, 14u, batch, a_offset, b_offset);
        float sum = 0.0f;
        for (vx_u32 column_base = 0; column_base < n;) {
            vx_u32 column = column_base + tx;
            gy[ty][tx] = row < m && column < n ? dy[batch * (m * n) + row * n + column] : 0.0f;
            weight[tx][ty] = inner_base + ty < k && column < n
                ? b[b_offset + (inner_base + ty) * n + column] : 0.0f;
            vx_cuda_training_linear_sync_threads();
            vx_u32 width = n - column_base;
            if (width > 16u) width = 16u;
            for (vx_u32 i = 0; i < width; i++) sum += gy[ty][i] * weight[i][tx];
            vx_cuda_training_linear_sync_threads();
            if (n - column_base <= 16u) break;
            column_base += 16u;
        }
        accumulated += sum;
    }
    if (valid) da[index] = accumulated;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_training_batch_matmul_backward_b_f32(
        const float* a, const float* b, const float* dy, float* da, float* db,
        VxCudaTrainingBatchMatMulParams params) {
    VX_CUDA_TRAINING_BMM_SHARED float input[16][17];
    VX_CUDA_TRAINING_BMM_SHARED float gy[16][17];
    const vx_u32* p = params.values;
    const vx_u32 m = p[1], k = p[2], n = p[3];
    const vx_u32 tx = threadIdx.x, ty = threadIdx.y;
    const vx_u32 column_tiles = (n - 1u) / 16u + 1u;
    const vx_u32 matrix_tiles = column_tiles * ((k - 1u) / 16u + 1u);
    const vx_u32 source_batch = blockIdx.x / matrix_tiles;
    const vx_u32 tile = blockIdx.x % matrix_tiles;
    const vx_u32 inner_base = (tile / column_tiles) * 16u;
    const vx_u32 inner = inner_base + ty;
    const vx_u32 column = (tile % column_tiles) * 16u + tx;
    const vx_u32 index = source_batch * (k * n) + inner * n + column;
    const bool valid = inner < k && column < n;
    const vx_u32 reductions = p[7] / (p[5] / (k * n));
    float accumulated = valid ? db[index] : 0.0f;
    (void)b; (void)da;
    for (vx_u32 reduced = 0; reduced < reductions; reduced++) {
        vx_u32 batch, a_offset, b_offset;
        vx_cuda_training_bmm_batch(params, source_batch, reduced, 20u, batch, a_offset, b_offset);
        for (vx_u32 row_base = 0; row_base < m;) {
            vx_u32 row = row_base + ty;
            input[tx][ty] = row < m && inner_base + tx < k
                ? a[a_offset + row * k + inner_base + tx] : 0.0f;
            gy[ty][tx] = row < m && column < n ? dy[batch * (m * n) + row * n + column] : 0.0f;
            vx_cuda_training_linear_sync_threads();
            vx_u32 width = m - row_base;
            if (width > 16u) width = 16u;
            for (vx_u32 i = 0; i < width; i++) accumulated += input[ty][i] * gy[i][tx];
            vx_cuda_training_linear_sync_threads();
            if (m - row_base <= 16u) break;
            row_base += 16u;
        }
    }
    if (valid) db[index] = accumulated;
}
#undef VX_CUDA_TRAINING_BMM_SHARED
