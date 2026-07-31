/* A 16x16 block owns one output tile and advances through K in ascending
 * 16-wide slices. Padded shared rows avoid transpose-store bank conflicts for
 * OUT_IN while preserving the existing launch ABI and one-thread-per-output
 * determinism. Every lane reaches both barriers, including all tail lanes. */
template <bool OutIn>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_matmul_tiled_body(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out) {
    VX_CUDA_SHARED float input_tile[16][17];
    VX_CUDA_SHARED float weight_tile[16][17];
    vx_u32 local_column = threadIdx.x;
    vx_u32 local_row = threadIdx.y;
    vx_u32 column_base = blockIdx.x * 16u;
    vx_u32 column = column_base + local_column;
    vx_u32 row = blockIdx.y * 16u + local_row;
    float sum = row < rows && column < d_out && bias
        ? bias[column] : 0.0f;

    for (vx_u32 k_base = 0u; k_base < d_in; k_base += 16u) {
        vx_u32 input_k = k_base + local_column;
        input_tile[local_row][local_column] =
            row < rows && input_k < d_in
            ? input[row * d_in + input_k] : 0.0f;
        if (OutIn) {
            vx_u32 load_column = column_base + local_row;
            vx_u32 weight_k = k_base + local_column;
            weight_tile[local_column][local_row] =
                load_column < d_out && weight_k < d_in
                ? weight[load_column * d_in + weight_k] : 0.0f;
        } else {
            vx_u32 weight_k = k_base + local_row;
            weight_tile[local_row][local_column] =
                weight_k < d_in && column < d_out
                ? weight[weight_k * d_out + column] : 0.0f;
        }
        vx_sync_threads();

        vx_u32 tile_width = d_in - k_base;
        if (tile_width > 16u) tile_width = 16u;
        for (vx_u32 k = 0u; k < tile_width; k++)
            sum += input_tile[local_row][k] *
                   weight_tile[k][local_column];
        vx_sync_threads();
    }
    if (row < rows && column < d_out)
        output[row * d_out + column] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out) {
    vx_cuda_matmul_tiled_body<false>(
        input, weight, bias, output, rows, d_in, d_out);
}

/* OUT_IN is the storage used by tied embeddings/LM heads: each output row of
 * the weight contains one complete input-width vector. Keep a separate entry
 * point so the canonical IN_OUT kernel retains its existing ABI and indexing
 * contract for inference packages. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_out_in_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 d_out) {
    vx_cuda_matmul_tiled_body<true>(
        input, weight, bias, output, rows, d_in, d_out);
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

extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_linear_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, float* output, vx_u32 rows,
        vx_u32 d_in, vx_u32 d_out, vx_u32 experts, vx_u32 top_k,
        vx_u32 has_bias) {
    vx_u32 flat = vx_global_x();
    vx_u32 count = rows * d_out;
    if (flat >= count) return;
    vx_u32 row = flat / d_out;
    vx_u32 column = flat - row * d_out;
    float sum = 0.0f;
    for (vx_u32 slot = 0u; slot < top_k; ++slot) {
        vx_u32 route = row * top_k + slot;
        vx_u32 expert = (vx_u32)route_indices[route];
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
