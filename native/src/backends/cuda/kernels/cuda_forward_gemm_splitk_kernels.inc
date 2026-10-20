/* A slice belongs to one complete warp; only final partial reduction crosses slices. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_splitk_sync_warp() {
#if defined(__CUDA_ARCH__)
    asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
#else
    vx_sync_threads();
#endif
}

/* Bounded split-K with 2, 4, 8 or 16 outer slices. Four warps reduce inner K
 * slices for each 32x32 output tile. Their padded A/W storage is reused for
 * partial outputs only after every warp finishes reading the final input tile.
 * The second launch reduces grid.z planes in fixed order and adds bias once.
 * Only NEW zero/nonfinite results repeat original bias-initialized ascending K. */
template <bool OutIn>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_splitk_body(
        const float* input, const float* weight, float* partials,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 split_k_slices) {
    if (blockDim.x != 8u || blockDim.y != 16u || blockDim.z != 1u ||
        blockIdx.z >= split_k_slices ||
        (split_k_slices != 2u && split_k_slices != 4u &&
         split_k_slices != 8u && split_k_slices != 16u)) return;
    VX_CUDA_SHARED float tile[4][1072];
    vx_u32 local_column = threadIdx.x, local_row = threadIdx.y % 4u;
    vx_u32 slice = threadIdx.y / 4u;
    vx_u32 lane = local_row * 8u + local_column;
    vx_u64 row_base = (vx_u64)blockIdx.y * 32u;
    vx_u64 column_base = (vx_u64)blockIdx.x * 32u;
    vx_u32 split_k_shift = split_k_slices == 2u ? 1u :
        split_k_slices == 4u ? 2u : split_k_slices == 8u ? 3u : 4u;
    vx_u64 outer_partition = (d_in >> split_k_shift) +
        ((d_in & (split_k_slices - 1u)) != 0u);
    vx_u64 outer_begin = blockIdx.z * outer_partition;
    if (outer_begin > d_in) outer_begin = d_in;
    vx_u64 outer_end = outer_begin + outer_partition;
    if (outer_end > d_in) outer_end = d_in;
    vx_u64 outer_length = outer_end - outer_begin;
    vx_u64 inner_partition = outer_length / 4u + (outer_length % 4u != 0u);
    vx_u64 begin = outer_begin + slice * inner_partition;
    if (begin > outer_end) begin = outer_end;
    vx_u64 end = begin + inner_partition;
    if (end > outer_end) end = outer_end;
    float sums[8][4] = {};
    for (vx_u64 k_base = 0u; k_base < inner_partition; k_base += 16u) {
        vx_u64 remaining = end - begin;
        vx_u32 width = remaining > k_base
            ? (vx_u32)(remaining - k_base < 16u ? remaining - k_base : 16u) : 0u;
        float fragment[8];
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for (vx_u32 batch = 0u; batch < 2u; batch++) {
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
            for (vx_u32 item = 0u; item < 8u; item++) {
                vx_u32 entry = lane + (batch * 8u + item) * 32u;
                vx_u32 tile_row = entry / 16u, tile_k = entry % 16u;
                vx_u64 row = row_base + tile_row;
                bool present = tile_k < width;
                vx_u64 input_k = present ? begin + k_base + tile_k : 0u;
                fragment[item] = row < rows && present
                    ? input[row * d_in + input_k] : 0.0f;
            }
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
            for (vx_u32 item = 0u; item < 8u; item++) {
                vx_u32 entry = lane + (batch * 8u + item) * 32u;
                tile[slice][(entry / 16u) * 17u + entry % 16u] = fragment[item];
            }
        }
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for (vx_u32 batch = 0u; batch < 2u; batch++) {
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
            for (vx_u32 item = 0u; item < 8u; item++) {
                vx_u32 entry = lane + (batch * 8u + item) * 32u;
                vx_u32 tile_k = OutIn ? entry % 16u : entry / 32u;
                vx_u32 tile_column = OutIn ? entry / 16u : entry % 32u;
                bool present = tile_k < width;
                vx_u64 weight_k = present ? begin + k_base + tile_k : 0u;
                vx_u64 column = column_base + tile_column;
                fragment[item] = present && column < d_out
                    ? weight[OutIn ? column * d_in + weight_k : weight_k * d_out + column]
                    : 0.0f;
            }
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
            for (vx_u32 item = 0u; item < 8u; item++) {
                vx_u32 entry = lane + (batch * 8u + item) * 32u;
                vx_u32 tile_k = OutIn ? entry % 16u : entry / 32u;
                vx_u32 tile_column = OutIn ? entry / 16u : entry % 32u;
                tile[slice][544u + tile_k * 33u + tile_column] = fragment[item];
            }
        }
        vx_cuda_matmul_splitk_sync_warp();
        if (width == 16u) {
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
            for (vx_u32 k = 0u; k < 16u; k++)
                for (vx_u32 r = 0u; r < 8u; r++) {
                    float value = tile[slice][(local_row + r * 4u) * 17u + k];
                    for (vx_u32 c = 0u; c < 4u; c++)
                        sums[r][c] += value * tile[slice][544u + k * 33u + local_column + c * 8u];
                }
        } else {
            for (vx_u32 k = 0u; k < width; k++)
                for (vx_u32 r = 0u; r < 8u; r++) {
                    float value = tile[slice][(local_row + r * 4u) * 17u + k];
                    for (vx_u32 c = 0u; c < 4u; c++)
                        sums[r][c] += value * tile[slice][544u + k * 33u + local_column + c * 8u];
                }
        }
        vx_cuda_matmul_splitk_sync_warp();
    }
    vx_sync_threads();
    for (vx_u32 r = 0u; r < 8u; r++)
        for (vx_u32 c = 0u; c < 4u; c++)
            tile[slice][(local_row + r * 4u) * 33u + local_column + c * 8u] = sums[r][c];
    vx_sync_threads();
    if (slice == 0u)
        for (vx_u32 r = 0u; r < 8u; r++)
            for (vx_u32 c = 0u; c < 4u; c++) {
                vx_u32 tile_index = (local_row + r * 4u) * 33u + local_column + c * 8u;
                vx_u64 row = row_base + local_row + r * 4u;
                vx_u64 column = column_base + local_column + c * 8u;
                if (row >= rows || column >= d_out) continue;
                float sum = tile[0][tile_index];
                sum += tile[1][tile_index];
                sum += tile[2][tile_index];
                sum += tile[3][tile_index];
                partials[(vx_u64)blockIdx.z * rows * d_out + row * d_out + column] = sum;
            }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_splitk_f32(
        const float* input, const float* weight, float* partials,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 split_k_slices,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, partials, rows, d_in, d_out, split_k_slices);
    vx_cuda_matmul_splitk_body<false>(input, weight, partials, rows, d_in, d_out, split_k_slices);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_splitk_out_in_f32(
        const float* input, const float* weight, float* partials,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 split_k_slices,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, partials, rows, d_in, d_out, split_k_slices);
    vx_cuda_matmul_splitk_body<true>(input, weight, partials, rows, d_in, d_out, split_k_slices);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_splitk_reduce_f32(
        const float* input, const float* weight, const float* bias,
        const float* partials, float* output,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 layout, vx_u32 split_k_slices,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, partials, output, rows, d_in, d_out, layout, split_k_slices);
    if (blockDim.x != 128u || blockDim.y != 1u || blockDim.z != 1u ||
        blockIdx.y != 0u || blockIdx.z != 0u || layout > 1u ||
        (split_k_slices != 2u && split_k_slices != 4u &&
         split_k_slices != 8u && split_k_slices != 16u)) return;
    vx_u64 stride = (vx_u64)rows * d_out;
    vx_u64 index = (vx_u64)blockIdx.x * 128u + threadIdx.x;
    if (index >= stride) return;
    vx_u32 column = index % d_out, row = index / d_out;
    if (d_in == 0u) {
        output[index] = bias ? bias[column] : 0.0f;
        return;
    }
    float sum = partials[index];
    for (vx_u32 part = 1u; part < split_k_slices; part++) sum += partials[part * stride + index];
    sum += bias ? bias[column] : 0.0f;
    if (sum == 0.0f || !(sum <= 3.402823466e38f && sum >= -3.402823466e38f)) {
        sum = bias ? bias[column] : 0.0f;
        for (vx_u32 k = 0u; k < d_in; k++) {
            float value = input[(vx_u64)row * d_in + k];
            float w = weight[layout ? (vx_u64)column * d_in + k : (vx_u64)k * d_out + column];
            sum = vx_add_rn(sum, vx_mul_rn(value, w));
        }
    }
    output[index] = sum;
}
