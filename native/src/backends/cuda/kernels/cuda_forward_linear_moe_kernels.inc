/* A 128-thread (8,16) block owns a 32x64 output tile. Each lane keeps four
 * rows and four columns in registers, using eight shared operand loads for
 * sixteen outputs per K. The 6336-byte shared tile is
 * the only workspace. Each output retains its ascending K multiply/add order;
 * every lane participates in both barriers, including output tails. */
template <bool OutIn>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_register_body(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out) {
    VX_CUDA_SHARED float input_tile[32][17];
    VX_CUDA_SHARED float weight_tile[16][65];
    vx_u32 local_column = threadIdx.x + (threadIdx.y / 8u) * 8u;
    vx_u32 local_row = threadIdx.y % 8u;
    vx_u32 lane = threadIdx.y * 8u + threadIdx.x;
    vx_u32 column_base = blockIdx.x * 64u;
    vx_u32 row_base = blockIdx.y * 32u;
    float sums[4][4];
    for (vx_u32 r = 0u; r < 4u; r++)
        for (vx_u32 c = 0u; c < 4u; c++) {
            vx_u32 row = row_base + local_row + r * 8u;
            vx_u32 column = column_base + local_column + c * 16u;
            sums[r][c] = row < rows && column < d_out && bias
                ? bias[column] : 0.0f;
        }

    for (vx_u32 k_base = 0u; k_base < d_in; k_base += 16u) {
        for (vx_u32 entry = lane; entry < 32u * 16u; entry += 128u) {
            vx_u32 tile_row = entry / 16u;
            vx_u32 tile_k = entry % 16u;
            vx_u32 row = row_base + tile_row, input_k = k_base + tile_k;
            input_tile[tile_row][tile_k] = row < rows && input_k < d_in
                ? input[row * d_in + input_k] : 0.0f;
        }
        for (vx_u32 entry = lane; entry < 16u * 64u; entry += 128u) {
            vx_u32 tile_k = OutIn ? entry % 16u : entry / 64u;
            vx_u32 tile_column = OutIn ? entry / 16u : entry % 64u;
            vx_u32 weight_k = k_base + tile_k;
            vx_u32 column = column_base + tile_column;
            weight_tile[tile_k][tile_column] = weight_k < d_in && column < d_out
                ? weight[OutIn ? column * d_in + weight_k : weight_k * d_out + column]
                : 0.0f;
        }
        vx_sync_threads();

        vx_u32 tile_width = d_in - k_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 k = 0u; k < tile_width; k++) {
            float values[4];
            for (vx_u32 c = 0u; c < 4u; c++)
                values[c] = weight_tile[k][local_column + c * 16u];
            for (vx_u32 r = 0u; r < 4u; r++) {
                float value = input_tile[local_row + r * 8u][k];
                for (vx_u32 c = 0u; c < 4u; c++)
                    sums[r][c] += value * values[c];
            }
        }
        vx_sync_threads();
    }
    for (vx_u32 r = 0u; r < 4u; r++)
        for (vx_u32 c = 0u; c < 4u; c++) {
            vx_u32 row = row_base + local_row + r * 8u;
            vx_u32 column = column_base + local_column + c * 16u;
            if (row < rows && column < d_out)
                output[row * d_out + column] = sums[r][c];
        }
}

/* Each compact slice is owned by one complete warp. Stage stores and reads
 * stay inside that slice; the final partial reduction still needs a CTA barrier. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_sync_warp() {
#if defined(__CUDA_ARCH__)
    asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
#else
    vx_sync_threads();
#endif
}

/* Keep the next K16 operands in registers while the current shared stage
 * is consumed. Empty rows, columns and short K partitions load exact zeros. */
template <bool OutIn>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_prefetch_stage(
        const float* input, const float* weight, float* next_input, float* next_weight,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 row_base,
        vx_u32 column_base, vx_u32 start, vx_u32 length, vx_u32 k_base, vx_u32 lane) {
    vx_u32 width = length > k_base ? length - k_base : 0u;
    if (width > 16u) width = 16u;
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
    for (vx_u32 item = 0u; item < 8u; item++) {
        vx_u32 entry = lane + item * 32u;
        vx_u32 tile_row = entry / 16u, tile_k = entry % 16u;
        vx_u32 row = row_base + tile_row;
        bool present = tile_k < width;
        vx_u32 input_k = present ? start + k_base + tile_k : 0u;
        next_input[item] = row < rows && present
            ? input[row * d_in + input_k] : 0.0f;
    }
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
    for (vx_u32 item = 0u; item < 8u; item++) {
        vx_u32 entry = lane + item * 32u;
        vx_u32 tile_k = OutIn ? entry % 16u : entry / 16u;
        vx_u32 tile_column = OutIn ? entry / 16u : entry % 16u;
        bool present = tile_k < width;
        vx_u32 weight_k = present ? start + k_base + tile_k : 0u;
        vx_u32 column = column_base + tile_column;
        next_weight[item] = present && column < d_out
            ? weight[OutIn ? column * d_in + weight_k : weight_k * d_out + column]
            : 0.0f;
    }
}

/* Four warps own fixed contiguous K partitions of one 16x16 output tile.
 * Each warp stages K16 in its own padded A/W pair (8704 bytes total), keeping
 * four rows and two columns per lane. The input tile is reused for partial
 * outputs after the final stage, then warp zero reduces slices 0,1,2,3 and
 * adds bias once. This deliberately reassociates finite sums. Zero and
 * nonfinite results are recomputed in the original bias-initialized K order. */
template <bool OutIn>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_compact_body(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out) {
    VX_CUDA_SHARED float input_tile[4][16][17];
    VX_CUDA_SHARED float weight_tile[4][16][17];
    vx_u32 local_column = threadIdx.x, local_row = threadIdx.y % 4u;
    vx_u32 slice = threadIdx.y / 4u;
    vx_u32 lane = local_row * 8u + local_column;
    vx_u32 row_base = blockIdx.y * 16u, column_base = blockIdx.x * 16u;
    if (d_in == 0u) {
        if (slice == 0u)
            for (vx_u32 r = 0u; r < 4u; r++)
                for (vx_u32 c = 0u; c < 2u; c++) {
                    vx_u32 row = row_base + local_row + r * 4u;
                    vx_u32 column = column_base + local_column + c * 8u;
                    if (row < rows && column < d_out)
                        output[row * d_out + column] = bias ? bias[column] : 0.0f;
                }
        return;
    }
    vx_u32 partition = d_in / 4u + (d_in % 4u != 0u);
    vx_u32 start = slice * partition;
    if (start > d_in) start = d_in;
    vx_u32 length = d_in - start;
    if (length > partition) length = partition;
    float sums[4][2] = {};
    float next_input[8], next_weight[8];
    vx_cuda_matmul_prefetch_stage<OutIn>(input, weight, next_input, next_weight,
        rows, d_in, d_out, row_base, column_base, start, length, 0u, lane);
    for (vx_u32 k_base = 0u; k_base < partition; k_base += 16u) {
        vx_u32 width = length > k_base ? length - k_base : 0u;
        if (width > 16u) width = 16u;
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
        for (vx_u32 item = 0u; item < 8u; item++) {
            vx_u32 entry = lane + item * 32u;
            input_tile[slice][entry / 16u][entry % 16u] = next_input[item];
            vx_u32 tile_k = OutIn ? entry % 16u : entry / 16u;
            vx_u32 tile_column = OutIn ? entry / 16u : entry % 16u;
            weight_tile[slice][tile_k][tile_column] = next_weight[item];
        }
        vx_cuda_matmul_sync_warp();
        if (partition - k_base > 16u)
            vx_cuda_matmul_prefetch_stage<OutIn>(input, weight, next_input, next_weight,
                rows, d_in, d_out, row_base, column_base, start, length, k_base + 16u, lane);
        if (width == 16u) {
#if defined(__CUDA_ARCH__)
#pragma unroll 8
#endif
            for (vx_u32 k = 0u; k < 16u; k++)
                for (vx_u32 r = 0u; r < 4u; r++) {
                    float value = input_tile[slice][local_row + r * 4u][k];
                    sums[r][0] += value * weight_tile[slice][k][local_column];
                    sums[r][1] += value * weight_tile[slice][k][local_column + 8u];
                }
        } else {
            for (vx_u32 k = 0u; k < width; k++)
                for (vx_u32 r = 0u; r < 4u; r++) {
                    float value = input_tile[slice][local_row + r * 4u][k];
                    sums[r][0] += value * weight_tile[slice][k][local_column];
                    sums[r][1] += value * weight_tile[slice][k][local_column + 8u];
                }
        }
        vx_cuda_matmul_sync_warp();
    }
    for (vx_u32 r = 0u; r < 4u; r++)
        for (vx_u32 c = 0u; c < 2u; c++)
            input_tile[slice][local_row + r * 4u][local_column + c * 8u] = sums[r][c];
    vx_sync_threads();
    if (slice == 0u)
        for (vx_u32 r = 0u; r < 4u; r++)
            for (vx_u32 c = 0u; c < 2u; c++) {
                vx_u32 tile_row = local_row + r * 4u;
                vx_u32 tile_column = local_column + c * 8u;
                vx_u32 row = row_base + tile_row, column = column_base + tile_column;
                if (row >= rows || column >= d_out) continue;
                float sum = input_tile[0][tile_row][tile_column];
                sum += input_tile[1][tile_row][tile_column];
                sum += input_tile[2][tile_row][tile_column];
                sum += input_tile[3][tile_row][tile_column];
                sum += bias ? bias[column] : 0.0f;
                if (sum == 0.0f ||
                    !(sum <= 3.402823466e38f && sum >= -3.402823466e38f)) {
                    sum = bias ? bias[column] : 0.0f;
                    for (vx_u32 k = 0u; k < d_in; k++)
                        sum += input[row * d_in + k] *
                            weight[OutIn ? column * d_in + k : k * d_out + column];
                }
                output[row * d_out + column] = sum;
            }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out);
    vx_cuda_matmul_compact_body<false>(
        input, weight, bias, output, rows, d_in, d_out);
}

/* OUT_IN is the storage used by tied embeddings/LM heads: each output row of
 * the weight contains one complete input-width vector. Keep a separate entry
 * point so the canonical IN_OUT kernel retains its existing ABI and indexing
 * contract for inference packages. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_out_in_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out);
    vx_cuda_matmul_compact_body<true>(
        input, weight, bias, output, rows, d_in, d_out);
}

/* Keep register accumulation in its own physical kernels so compact launches
 * cannot inherit its register allocation or larger static shared tile. Both
 * entries retain the same argument record and ascending K arithmetic. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_register_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out);
    vx_cuda_matmul_register_body<false>(
        input, weight, bias, output, rows, d_in, d_out);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_register_out_in_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out);
    vx_cuda_matmul_register_body<true>(
        input, weight, bias, output, rows, d_in, d_out);
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_cuda_gemv_shuffle_down(
        float value, vx_u32 delta) {
    float shuffled;
    asm("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(shuffled) : "f"(value), "r"(delta));
    return shuffled;
}

/* Decode has one input row, so a 16-row GEMM tile spends most of its work
 * on padding. IN_OUT instead coalesces eight adjacent output columns and
 * partitions K across sixteen lanes per column. One barrier publishes all
 * partial sums; the fixed partition order makes repeated launches stable.
 * This needs only 576 bytes of block-local storage, with no device workspace.
 * The eight-argument ABI, including the optional device record, matches matmul
 * for the shared host route. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_gemv_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out);
    VX_CUDA_SHARED float partials[16][9];
    vx_u32 column_lane = threadIdx.x & 7u;
    vx_u32 partition = threadIdx.x >> 3u;
    vx_u32 column = blockIdx.x * 8u + column_lane;
    float sum = 0.0f;
    if (rows == 1u && column < d_out)
        for (vx_u32 k = partition; k < d_in; k += 16u)
            sum += input[k] * weight[k * d_out + column];
    partials[partition][column_lane] = sum;
    vx_sync_threads();
    if (rows == 1u && partition == 0u && column < d_out) {
        sum = bias ? bias[column] : 0.0f;
        for (vx_u32 part = 0u; part < 16u; ++part)
            sum += partials[part][column_lane];
        output[column] = sum;
    }
}

/* Tied embeddings store OUT_IN, so each warp owns one output and reads its
 * complete weight row with contiguous lanes. Tail output warps still execute
 * every shuffle with zero partials, satisfying the full-warp participation
 * mask. No atomics or grid-wide reduction changes output ownership. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_gemv_out_in_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const float* residual, float* final_output, vx_u32 residual_first,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, bias, output, rows, d_in, d_out, residual, final_output, residual_first);
    vx_u32 lane = threadIdx.x & 31u;
    vx_u32 column = blockIdx.x * 4u + (threadIdx.x >> 5u);
    float sum = 0.0f;
    if (rows == 1u && column < d_out)
        for (vx_u32 k = lane; k < d_in; k += 32u)
            sum += input[k] * weight[column * d_in + k];
    for (vx_u32 delta = 16u; delta != 0u; delta >>= 1u)
        sum += vx_cuda_gemv_shuffle_down(sum, delta);
    if (rows == 1u && lane == 0u && column < d_out) {
        float value = sum + (bias ? bias[column] : 0.0f);
        output[column] = value;
        if (residual && final_output)
            final_output[column] = residual_first ? vx_add_rn(residual[column], value)
                                                  : vx_add_rn(value, residual[column]);
    }
}

/* LoRA inference is split into two device-resident stages. The first projects
 * the selected row window through canonical A[d_in,rank]; the second applies
 * canonical B[rank,d_out] and accumulates the scaled delta into a base Linear
 * result that already contains its bias. Keeping the rank workspace in the
 * CUDA host backend avoids both a host round trip and the O(d_out*d_in*rank)
 * recomputation of a single fully fused scalar kernel. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_lora_rank_f32(
        const float* input, const float* a, float* workspace,
        vx_u32 rows, vx_u32 d_in, vx_u32 rank) {
    vx_u32 component = vx_global_x();
    vx_u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || component >= rank) return;
    float sum = 0.0f;
    vx_u32 input_base = row * d_in;
    for (vx_u32 dimension = 0u; dimension < d_in; dimension++)
        sum += input[input_base + dimension] *
               a[dimension * rank + component];
    workspace[row * rank + component] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_lora_apply_f32(
        const float* workspace, const float* b, float* output,
        vx_u32 rows, vx_u32 rank, vx_u32 d_out, float scale) {
    vx_u32 column = vx_global_x();
    vx_u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || column >= d_out) return;
    float delta = 0.0f;
    for (vx_u32 component = 0u; component < rank; component++)
        delta += workspace[row * rank + component] *
                 b[component * d_out + column];
    vx_u32 output_index = row * d_out + column;
    output[output_index] += scale * delta;
}

/* One lane owns one row so top-k selection and its tie order stay
 * deterministic. Route indices deliberately use exact integer-valued F32,
 * matching the native differentiable MoE contract. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_router_f32(
        const float* input, const float* weight, const float* bias,
        float* route_indices, float* route_weights, vx_u32 rows,
        vx_u32 d_model, vx_u32 experts, vx_u32 top_k, float temperature,
        vx_u32 normalize, vx_u32 has_bias) {
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    float maximum = 0.0f;
    for (vx_u32 expert = 0u; expert < experts; ++expert) {
        float logit = has_bias != 0u ? bias[expert] : 0.0f;
        for (vx_u32 dimension = 0u; dimension < d_model; ++dimension)
            logit += input[row * d_model + dimension] *
                weight[dimension * experts + expert];
        logit /= temperature;
        if (expert == 0u || logit > maximum) maximum = logit;
    }
    for (vx_u32 slot = 0u; slot < top_k; ++slot) {
        vx_u32 best = 0u;
        float best_value = 0.0f;
        vx_u32 have_best = 0u;
        for (vx_u32 expert = 0u; expert < experts; ++expert) {
            vx_u32 selected = 0u;
            for (vx_u32 prior = 0u; prior < slot; ++prior)
                if ((vx_u32)route_indices[row * top_k + prior] == expert)
                    selected = 1u;
            if (selected != 0u) continue;
            float logit = has_bias != 0u ? bias[expert] : 0.0f;
            for (vx_u32 dimension = 0u; dimension < d_model; ++dimension)
                logit += input[row * d_model + dimension] *
                    weight[dimension * experts + expert];
            logit /= temperature;
            if (have_best == 0u || logit > best_value ||
                (logit == best_value && expert < best)) {
                best = expert;
                best_value = logit;
                have_best = 1u;
            }
        }
        route_indices[row * top_k + slot] = (float)best;
    }
    float denominator = 0.0f;
    vx_u32 count = normalize != 0u ? top_k : experts;
    for (vx_u32 item = 0u; item < count; ++item) {
        vx_u32 expert = normalize != 0u
            ? (vx_u32)route_indices[row * top_k + item] : item;
        float logit = has_bias != 0u ? bias[expert] : 0.0f;
        for (vx_u32 dimension = 0u; dimension < d_model; ++dimension)
            logit += input[row * d_model + dimension] *
                weight[dimension * experts + expert];
        denominator += vx_exp(logit / temperature - maximum);
    }
    for (vx_u32 slot = 0u; slot < top_k; ++slot) {
        vx_u32 expert = (vx_u32)route_indices[row * top_k + slot];
        float logit = has_bias != 0u ? bias[expert] : 0.0f;
        for (vx_u32 dimension = 0u; dimension < d_model; ++dimension)
            logit += input[row * d_model + dimension] *
                weight[dimension * experts + expert];
        route_weights[row * top_k + slot] =
            vx_exp(logit / temperature - maximum) / denominator;
    }
}

/* `experts` counts the staged rows of a possibly partially resident bank.
 * Route indices stay in the model's global slot space; slot_rows maps them to
 * staged rows and holds VX_MOE_SLOT_ABSENT (0xffffffff) for absent slots. A
 * zero slot_domain means the bank is fully resident and ids are already rows. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_linear_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, float* output, vx_u32 rows,
        vx_u32 d_in, vx_u32 d_out, vx_u32 experts, vx_u32 top_k,
        vx_u32 has_bias, const vx_u32* slot_rows, vx_u32 slot_domain) {
    vx_u32 flat = vx_global_x();
    vx_u32 count = rows * d_out;
    if (flat >= count) return;
    vx_u32 row = flat / d_out;
    vx_u32 column = flat - row * d_out;
    float sum = 0.0f;
    for (vx_u32 slot = 0u; slot < top_k; ++slot) {
        vx_u32 route = row * top_k + slot;
        vx_u32 expert = (vx_u32)route_indices[route];
        if (slot_domain != 0u) {
            if (expert >= slot_domain) continue;
            expert = slot_rows[expert];
        }
        if (expert >= experts) continue;
        float value = has_bias != 0u
            ? expert_bias[expert * d_out + column] : 0.0f;
        vx_u32 weight_base = expert * d_in * d_out;
        for (vx_u32 dimension = 0u; dimension < d_in; ++dimension)
            value += input[row * d_in + dimension] *
                expert_weight[weight_base + dimension * d_out + column];
        sum += route_weights[route] * value;
    }
    output[flat] = sum;
}

static VX_CUDA_DEVICE float vx_moe_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return vx_exp(x);
    float value = 1.0f + vx_exp(x), logarithm;
    asm("lg2.approx.f32 %0, %1;" : "=f"(logarithm) : "f"(value));
    return logarithm * 0.6931471805599453f;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_router_extended_f32(
        const float* input, const float* weight, const float* bias,
        const float* correction, const vx_i32* tokens, const vx_i32* hash,
        float* indices, float* routes, vx_u32 rows, vx_u32 width,
        vx_u32 experts, vx_u32 top_k, float temperature, vx_u32 normalize,
        vx_u32 scoring, vx_u32 groups, vx_u32 selected_groups,
        vx_u32 group_top_k, float scale, float epsilon, vx_u32 vocabulary) {
    VX_CUDA_SHARED float scratch[8192];
    vx_u32 row = blockIdx.x;
    if (row >= rows) return;
    float* scores = scratch;
    float* selection = scratch + experts;
    float* group_scores = selection + experts;
    for (vx_u32 e = threadIdx.x; e < experts; e += blockDim.x) {
        float x = bias ? bias[e] : 0.0f;
        for (vx_u32 d = 0; d < width; ++d)
            x += input[row * width + d] * weight[d * experts + e];
        scores[e] = x / temperature;
    }
    vx_sync_threads();
    if (threadIdx.x != 0) return;
    float maximum = scores[0], total = 0;
    for (vx_u32 e = 1; e < experts; ++e) maximum = vx_max(maximum, scores[e]);
    for (vx_u32 e = 0; e < experts; ++e) {
        float x = scores[e];
        scores[e] = scoring == 0 ? vx_exp(x - maximum) : scoring == 1
            ? (x >= 0 ? vx_sigmoid(x) : vx_exp(x) / (1.0f + vx_exp(x)))
            : vx_sqrt(vx_moe_softplus(x));
        selection[e] = x;
        total += scores[e];
    }
    for (vx_u32 e = 0; e < experts; ++e) {
        if (scoring == 0) scores[e] /= total;
        if (correction) selection[e] = scores[e] + correction[e];
    }
    vx_u32 per_group = experts / groups;
    for (vx_u32 g = 0; g < groups; ++g) {
            if (selected_groups == groups) { group_scores[g] = 0; continue; }
        float sum = 0;
        for (vx_u32 j = 0; j < per_group; ++j) {
            vx_u32 e = g * per_group + j, rank = 0;
            float candidate = scores[e] + (correction ? correction[e] : 0);
            for (vx_u32 l = 0; l < per_group; ++l) {
                vx_u32 other = g * per_group + l;
                float value = scores[other] + (correction ? correction[other] : 0);
                if (value > candidate || (value == candidate && other < e)) ++rank;
            }
            if (rank < group_top_k) sum += candidate;
        }
        group_scores[g] = sum;
    }
    for (vx_u32 k = 0; k < top_k; ++k) {
        vx_u32 best = experts;
        if (hash) {
            vx_i32 token = tokens[row];
            if (token < 0 || (vx_u32)token >= vocabulary) return;
            vx_i32 e = hash[(vx_u32)token * top_k + k];
            if (e < 0 || (vx_u32)e >= experts) return;
            best = (vx_u32)e;
        } else for (vx_u32 e = 0; e < experts; ++e) {
            vx_u32 g = e / per_group, rank = 0, used = 0;
            for (vx_u32 other = 0; other < groups; ++other)
                if (group_scores[other] > group_scores[g] ||
                    (group_scores[other] == group_scores[g] && other < g)) ++rank;
            if (rank >= selected_groups) continue;
            for (vx_u32 prior = 0; prior < k; ++prior)
                if ((vx_u32)indices[row * top_k + prior] == e) used = 1;
            if (!used && (best == experts || selection[e] > selection[best])) best = e;
        }
        indices[row * top_k + k] = (float)best;
    }
    float denominator = epsilon;
    for (vx_u32 k = 0; k < top_k; ++k) denominator += scores[(vx_u32)indices[row * top_k + k]];
    for (vx_u32 k = 0; k < top_k; ++k)
        routes[row * top_k + k] = scale * scores[(vx_u32)indices[row * top_k + k]] /
            (normalize ? denominator : 1.0f);
}

/* Preserve each expert's hidden activation until its down projection has
 * completed. Reducing the expert axis before SiLU/Mul would change the model. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_gated_mlp_f32(
        const float* input, const float* gate_weight, const float* up_weight,
        const float* down_weight, const float* indices, const float* routes,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 hidden,
        vx_u32 d_out, vx_u32 experts, vx_u32 top_k,
        vx_u32 input_scale, float limit) {
    VX_CUDA_SHARED float intermediate[8192];
    vx_u32 row = blockIdx.x;
    if (row >= rows) return;
    for (vx_u32 o = threadIdx.x; o < d_out; o += blockDim.x) output[row * d_out + o] = 0;
    for (vx_u32 k = 0; k < top_k; ++k) {
        vx_u32 e = (vx_u32)indices[row * top_k + k];
        float route = routes[row * top_k + k];
        if (e >= experts) return; /* Admission proves all route sources. */
        for (vx_u32 h = threadIdx.x; h < hidden; h += blockDim.x) {
            float gate = 0, up = 0;
            for (vx_u32 d = 0; d < d_in; ++d) {
                float x = input[row * d_in + d] * (input_scale ? route : 1.0f);
                vx_u32 offset = (e * d_in + d) * hidden + h;
                gate += x * gate_weight[offset];
                up += x * up_weight[offset];
            }
            if (limit > 0) { gate = vx_min(gate, limit); up = vx_max(-limit, vx_min(up, limit)); }
            intermediate[h] = gate * vx_sigmoid(gate) * up;
        }
        vx_sync_threads();
        for (vx_u32 o = threadIdx.x; o < d_out; o += blockDim.x) {
            float value = 0;
            for (vx_u32 h = 0; h < hidden; ++h)
                value += intermediate[h] * down_weight[(e * hidden + h) * d_out + o];
            output[row * d_out + o] += (input_scale ? 1.0f : route) * value;
        }
        vx_sync_threads();
    }
}
