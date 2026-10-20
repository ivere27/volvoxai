static VX_CUDA_DEVICE vx_i32 vx_attention_key_allowed(
        const vx_i32* mask, vx_u32 mask_mode, vx_u32 batch_index,
        vx_u32 query, vx_u32 key, vx_u32 sequence_q,
        vx_u32 sequence_kv, vx_u32 causal) {
    mask = vx_cuda_global_pointer(mask);
    if (causal != 0u && key > query) return 0;
    if (mask_mode == 0u) return 1;
    vx_u32 index = key;
    if (mask_mode == 2u) index = batch_index * sequence_kv + key;
    else if (mask_mode == 3u) index = query * sequence_kv + key;
    else if (mask_mode == 4u)
        index = (batch_index * sequence_q + query) * sequence_kv + key;
    return mask[index] != 0;
}

/* A captured grid is usable only when it covers every output tile. If row
 * dimensions change, z=0 retains the original full-width body; other z blocks
 * return at the entry. No tile address is encoded in argument storage. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_attention_output_tiled(
        vx_u32 query_sequence, vx_u32 key_sequence, vx_u32 head_dim) {
    return query_sequence == 1u && key_sequence <= 48u && head_dim <= 128u &&
        gridDim.z == (head_dim + 31u) / 32u;
}

/* One block owns a query/head. The fixed tile reuses each Q.K score across
 * value channels; it never allocates a sequence-sized score matrix. General
 * rows accumulate dots in channel order. Short cross rows use eight lanes
 * per key and a first-warp softmax. P.V is scalar ascending or four partials
 * with a fixed merge. Every route uses separate F32 multiply/add. */
static VX_CUDA_DEVICE void vx_attention_score_tile(
        const float* query_values, const float* key_values,
        const vx_i32* mask, float* scores, vx_u32 query_base,
        vx_u32 key_batch_base, vx_u32 key_stride, vx_u32 head_dim,
        vx_u32 batch_index, vx_u32 query, vx_u32 query_sequence,
        vx_u32 key_sequence, vx_u32 causal, vx_u32 mask_mode,
        vx_u32 start, vx_u32 count, float scale) {
    query_values = vx_cuda_global_pointer(query_values);
    key_values = vx_cuda_global_pointer(key_values);
    vx_u32 lane = threadIdx.x;
    if (lane < count) {
        vx_u32 key = start + lane;
        float score = 0.0f;
        if (vx_attention_key_allowed(mask, mask_mode, batch_index, query,
                                    key, query_sequence, key_sequence, causal)) {
            vx_u32 key_base = key_batch_base + key * key_stride;
            for (vx_u32 channel = 0u; channel < head_dim; channel++)
                score += query_values[query_base + channel] *
                    key_values[key_base + channel];
        }
        scores[lane] = score * scale;
    }
    vx_sync_threads();
}

/* The admitted 128-thread block covers sixteen keys per round. Eight-lane
 * groups reduce independent channel dots without crossing group boundaries
 * at the publishing lane. All lanes participate, including masked/tail keys.
 * The first warp reduces at most 48 scores; shared storage remains 48 + 3 floats.
 * P.V uses scalar ascending or four-part fixed merge; masked values are skipped. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_attention_short_row(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        float* scores, float* statistics, vx_u32 query_base,
        vx_u32 key_batch_base, vx_u32 value_batch_base, vx_u32 destination,
        vx_u32 key_stride, vx_u32 head_dim, vx_u32 batch_index,
        vx_u32 query, vx_u32 query_sequence, vx_u32 key_sequence,
        float scale, vx_u32 causal, vx_u32 mask_mode, vx_u32 active_keys) {
    query_values = vx_cuda_global_pointer(query_values);
    key_values = vx_cuda_global_pointer(key_values);
    vx_u32 lane = threadIdx.x;
    vx_u32 channel_lane = lane & 7u;
    vx_u32 group = lane / 8u;
    for (vx_u32 first = 0u; first < active_keys; first += 16u) {
        vx_u32 key = first + group;
        float score = 0.0f;
        if (key < active_keys && vx_attention_key_allowed(mask, mask_mode,
                batch_index, query, key, query_sequence, key_sequence, causal)) {
            vx_u32 key_base = key_batch_base + key * key_stride;
            for (vx_u32 channel = channel_lane; channel < head_dim; channel += 8u)
                score += query_values[query_base + channel] *
                    key_values[key_base + channel];
        }
        for (vx_u32 delta = 4u; delta != 0u; delta /= 2u)
            score += vx_cuda_attention_shuffle_down(score, delta);
        if (channel_lane == 0u && key < active_keys)
            scores[key] = score * scale;
    }
    vx_sync_threads();
    /* One warp owns both scores per lane until normalized weights are
     * published. Keep the original lane-local and shuffle reduction order. */
    if (lane < 32u) {
        vx_u32 second_key = lane + 32u;
        int first_allowed = lane < active_keys && vx_attention_key_allowed(mask,
            mask_mode, batch_index, query, lane, query_sequence, key_sequence, causal);
        int second_allowed = second_key < active_keys && vx_attention_key_allowed(mask,
            mask_mode, batch_index, query, second_key, query_sequence, key_sequence, causal);
        float first_score = first_allowed ? scores[lane] : 0.0f;
        float second_score = second_allowed ? scores[second_key] : 0.0f;
        float maximum = -3.402823466e38f;
        if (first_allowed) maximum = vx_max(maximum, first_score);
        if (second_allowed) maximum = vx_max(maximum, second_score);
        for (vx_u32 delta = 16u; delta != 0u; delta /= 2u)
            maximum = vx_max(maximum, vx_cuda_attention_shuffle_down(maximum, delta));
        maximum = vx_cuda_attention_shuffle_lane(maximum, 0u);
        float first_weight = first_allowed ? vx_exp(first_score - maximum) : 0.0f;
        float second_weight = second_allowed ? vx_exp(second_score - maximum) : 0.0f;
        float denominator = 0.0f;
        if (lane < active_keys) denominator += first_weight;
        if (second_key < active_keys) denominator += second_weight;
        for (vx_u32 delta = 16u; delta != 0u; delta /= 2u)
            denominator += vx_cuda_attention_shuffle_down(denominator, delta);
        denominator = vx_cuda_attention_shuffle_lane(denominator, 0u);
        if (denominator != 0.0f) {
            first_weight /= denominator;
            second_weight /= denominator;
        }
        if (lane < active_keys) scores[lane] = first_weight;
        if (second_key < active_keys) scores[second_key] = second_weight;
        if (lane == 0u) statistics[1] = denominator;
    }
    vx_sync_threads();
    if (vx_attention_output_tiled(query_sequence, key_sequence, head_dim)) {
        /* Each warp owns contiguous channels and every fourth key. Publish
         * all partials, including tails/empty rows, before the fixed merge. */
        VX_CUDA_SHARED float partials[128];
        vx_u32 channel = blockIdx.z * 32u + (lane & 31u);
        vx_u32 key_part = lane / 32u;
        float partial = 0.0f;
        if (channel < head_dim && statistics[1] != 0.0f)
            for (vx_u32 key = key_part; key < active_keys; key += 4u) {
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, key, query_sequence, key_sequence, causal)) continue;
                partial += scores[key] *
                    value_values[value_batch_base + key * key_stride + channel];
            }
        partials[lane] = partial;
        vx_sync_threads();
        if (lane < 32u && channel < head_dim) {
            float result = partials[lane];
            result += partials[32u + lane];
            result += partials[64u + lane];
            result += partials[96u + lane];
            output[destination + channel] = result;
        }
        return;
    }
    if (lane < head_dim) {
        float result = 0.0f;
        if (statistics[1] != 0.0f)
            for (vx_u32 key = 0u; key < active_keys; ++key) {
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, key, query_sequence, key_sequence, causal)) continue;
                result += scores[key] *
                    value_values[value_batch_base + key * key_stride + lane];
            }
        output[destination + lane] = result;
    }
}

/* Bounded rows publish every score before softmax. Independent denominator
 * groups run in parallel, then merge in the original ascending order. Keep
 * scalar QK and ascending 48-key P.V sums: reassociation can cross activation
 * quantization boundaries and amplify whole-model logit differences. */
static VX_CUDA_DEVICE void vx_attention_cached_row(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        float* scores, float* statistics, vx_u32 query_base,
        vx_u32 key_batch_base, vx_u32 value_batch_base, vx_u32 destination,
        vx_u32 key_stride, vx_u32 head_dim, vx_u32 batch_index,
        vx_u32 query, vx_u32 query_sequence, vx_u32 key_sequence,
        float scale, vx_u32 causal, vx_u32 mask_mode, vx_u32 active_keys) {
    query_values = vx_cuda_global_pointer(query_values);
    key_values = vx_cuda_global_pointer(key_values);
    vx_u32 lane = threadIdx.x;
    for (vx_u32 key = lane; key < active_keys; key += blockDim.x) {
        float score = 0.0f;
        if (vx_attention_key_allowed(mask, mask_mode, batch_index, query,
                key, query_sequence, key_sequence, causal)) {
            vx_u32 key_base = key_batch_base + key * key_stride;
            for (vx_u32 channel = 0u; channel < head_dim; channel++)
                score += query_values[query_base + channel] * key_values[key_base + channel];
        }
        scores[key] = score * scale;
    }
    vx_sync_threads();
    if (lane < 32u) {
        float maximum = -3.402823466e38f;
        float any = 0.0f;
        for (vx_u32 key = lane; key < active_keys; key += 32u) {
            if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                    query, key, query_sequence, key_sequence, causal)) continue;
            maximum = vx_max(maximum, scores[key]);
            any = 1.0f;
        }
        for (vx_u32 delta = 16u; delta; delta /= 2u) {
            maximum = vx_max(maximum, vx_cuda_attention_shuffle_down(maximum, delta));
            any = vx_max(any, vx_cuda_attention_shuffle_down(any, delta));
        }
        if (!lane) {
            statistics[0] = maximum;
            statistics[2] = any;
        }
    }
    vx_sync_threads();
    if (statistics[2] == 0.0f) {
        for (vx_u32 channel = lane; channel < head_dim; channel += blockDim.x)
            output[destination + channel] = 0.0f;
        return;
    }
    for (vx_u32 key = lane; key < active_keys; key += blockDim.x)
        scores[key] = vx_attention_key_allowed(mask, mask_mode, batch_index,
            query, key, query_sequence, key_sequence, causal)
            ? vx_exp(scores[key] - statistics[0]) : 0.0f;
    vx_sync_threads();
    if (lane < 32u) {
        float denominator = 0.0f;
        for (vx_u32 first = 0u; first < active_keys; first += 48u * 32u) {
            vx_u32 start = first + lane * 48u;
            vx_u32 count = start < active_keys ?
                (active_keys - start < 48u ? active_keys - start : 48u) : 0u;
            float partial = 0.0f;
            for (vx_u32 offset = 0u; offset < count; offset++) {
                vx_u32 key = start + offset;
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, key, query_sequence, key_sequence, causal)) continue;
                partial += scores[key];
            }
            vx_u32 parts = (active_keys - first + 47u) / 48u;
            if (parts > 32u) parts = 32u;
            for (vx_u32 part = 0u; part < parts; part++) {
                float value = vx_cuda_attention_shuffle_lane(partial, part);
                if (!lane) denominator += value;
            }
        }
        if (!lane) statistics[1] = denominator;
    }
    vx_sync_threads();
    for (vx_u32 key = lane; key < active_keys; key += blockDim.x)
        scores[key] /= statistics[1];
    vx_sync_threads();
    for (vx_u32 channel = lane; channel < head_dim; channel += blockDim.x) {
        float result = 0.0f;
        for (vx_u32 start = 0u; start < active_keys; start += 48u) {
            vx_u32 count = (active_keys - start < 48u ? active_keys - start : 48u);
            float partial = 0.0f;
            for (vx_u32 offset = 0u; offset < count; offset++) {
                vx_u32 key = start + offset;
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, key, query_sequence, key_sequence, causal)) continue;
                partial += scores[key] *
                    value_values[value_batch_base + key * key_stride + channel];
            }
            result += partial;
        }
        output[destination + channel] = result;
    }
}

template <bool BoundedReduction>
static VX_CUDA_DEVICE void vx_attention_tiled_body(
        const float* query_values, const float* key_values,
        const float* value_values, const vx_i32* mask, float* output,
        vx_u32 query_base, vx_u32 key_batch_base, vx_u32 value_batch_base,
        vx_u32 destination, vx_u32 key_stride, vx_u32 head_dim,
        vx_u32 batch_index, vx_u32 query, vx_u32 query_sequence,
        vx_u32 key_sequence, float scale, vx_u32 causal, vx_u32 mask_mode) {
    value_values = vx_cuda_global_pointer(value_values);
    output = vx_cuda_global_pointer(output);
    /* Cache the bounded cross-attention row without changing its scalar dot
     * or 48-key reduction order. Larger rows retain the streaming tile. */
    VX_CUDA_SHARED float scores[BoundedReduction ? (vx_u32)VX_CUDA_CROSS_ATTENTION_CACHE_KEYS
                                           : (vx_u32)VX_CUDA_ATTENTION_TILE_KEYS];
    /* Maximum, denominator, and whether any unmasked key exists. */
    VX_CUDA_SHARED float statistics[3];
    vx_u32 lane = threadIdx.x;
    vx_u32 active_keys = key_sequence;
    if (causal != 0u && query < key_sequence - 1u) active_keys = query + 1u;
    if (BoundedReduction && active_keys <= 48u && head_dim <= 128u) {
        vx_attention_short_row(query_values, key_values, value_values, mask,
            output, scores, statistics, query_base, key_batch_base,
            value_batch_base, destination, key_stride, head_dim, batch_index,
            query, query_sequence, key_sequence, scale, causal, mask_mode, active_keys);
        return;
    }
    if (BoundedReduction && active_keys > 48u &&
            active_keys <= VX_CUDA_CROSS_ATTENTION_CACHE_KEYS) {
        vx_attention_cached_row(query_values, key_values, value_values, mask,
            output, scores, statistics, query_base, key_batch_base,
            value_batch_base, destination, key_stride, head_dim, batch_index,
            query, query_sequence, key_sequence, scale, causal, mask_mode, active_keys);
        return;
    }
    const bool cached_scores = BoundedReduction && active_keys <= VX_CUDA_CROSS_ATTENTION_CACHE_KEYS;
    if (lane == 0u) {
        statistics[0] = -3.402823466e38f;
        statistics[1] = 0.0f;
        statistics[2] = 0.0f;
    }
    for (vx_u32 channel = lane; channel < head_dim; channel += blockDim.x)
        output[destination + channel] = 0.0f;
    vx_sync_threads();

    for (vx_u32 start = 0u; start < active_keys;) {
        vx_u32 remaining = active_keys - start;
        vx_u32 count = remaining < 48u ? remaining : 48u;
        float* tile_scores = cached_scores ? scores + start : scores;
        vx_attention_score_tile(query_values, key_values, mask, tile_scores,
            query_base, key_batch_base, key_stride, head_dim, batch_index,
            query, query_sequence, key_sequence, causal, mask_mode,
            start, count, scale);
        if (lane == 0u) {
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
            for (vx_u32 offset = 0u; offset < count; offset++) {
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, start + offset, query_sequence, key_sequence,
                        causal)) continue;
                statistics[0] = vx_max(statistics[0], tile_scores[offset]);
                statistics[2] = 1.0f;
            }
        }
        vx_sync_threads();
        start += count;
    }
    if (statistics[2] == 0.0f) return;

    /* The bounded cross row keeps the maximum pass's scores, then replaces
     * each tile with the same exponential used by the streaming denominator. */
    for (vx_u32 start = 0u; start < active_keys;) {
        vx_u32 remaining = active_keys - start;
        vx_u32 count = remaining < 48u ? remaining : 48u;
        float* tile_scores = cached_scores ? scores + start : scores;
        if (active_keys > 48u && !cached_scores)
            vx_attention_score_tile(query_values, key_values, mask, tile_scores,
                query_base, key_batch_base, key_stride, head_dim, batch_index,
                query, query_sequence, key_sequence, causal, mask_mode,
                start, count, scale);
        if (lane < count)
            tile_scores[lane] = vx_attention_key_allowed(mask, mask_mode,
                batch_index, query, start + lane, query_sequence,
                key_sequence, causal)
                ? vx_exp(tile_scores[lane] - statistics[0]) : 0.0f;
        vx_sync_threads();
        if (lane == 0u) {
            float partial = BoundedReduction ? 0.0f : statistics[1];
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
            for (vx_u32 offset = 0u; offset < count; offset++) {
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, start + offset, query_sequence, key_sequence,
                        causal)) continue;
                partial += tile_scores[offset];
            }
            statistics[1] = BoundedReduction
                ? statistics[1] + partial : partial;
        }
        vx_sync_threads();
        start += count;
    }
    for (vx_u32 start = 0u; start < active_keys;) {
        vx_u32 remaining = active_keys - start;
        vx_u32 count = remaining < 48u ? remaining : 48u;
        float* tile_scores = cached_scores ? scores + start : scores;
        if (active_keys > 48u && !cached_scores) {
            vx_attention_score_tile(query_values, key_values, mask, tile_scores,
                query_base, key_batch_base, key_stride, head_dim, batch_index,
                query, query_sequence, key_sequence, causal, mask_mode,
                start, count, scale);
            if (lane < count)
                tile_scores[lane] = vx_attention_key_allowed(mask, mask_mode,
                    batch_index, query, start + lane, query_sequence,
                    key_sequence, causal)
                    ? vx_exp(tile_scores[lane] - statistics[0]) / statistics[1]
                    : 0.0f;
        } else if (lane < count) {
            tile_scores[lane] /= statistics[1];
        }
        vx_sync_threads();
        for (vx_u32 channel = lane; channel < head_dim;
                channel += blockDim.x) {
            float result = output[destination + channel];
            float partial = BoundedReduction ? 0.0f : result;
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
            for (vx_u32 offset = 0u; offset < count; offset++) {
                vx_u32 key = start + offset;
                if (!vx_attention_key_allowed(mask, mask_mode, batch_index,
                        query, key, query_sequence, key_sequence, causal))
                    continue;
                partial += tile_scores[offset] *
                    value_values[value_batch_base + key * key_stride + channel];
            }
            output[destination + channel] = BoundedReduction
                ? result + partial : partial;
        }
        vx_sync_threads();
        start += count;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_sdpa_f32(
        const float* qkv, const vx_i32* mask, float* output,
        vx_u32 sequence, vx_u32 d_model, vx_u32 heads,
        vx_u32 head_dim, vx_u32 batch, float scale, vx_u32 causal,
        vx_u32 mask_mode, vx_u32 query_start, vx_u32 query_count) {
    vx_u32 tasks = batch * query_count * heads;
    vx_u64 task_index = (vx_u64)blockIdx.y * gridDim.x + blockIdx.x;
    if (task_index >= tasks) return;
    vx_u32 task = (vx_u32)task_index;
    vx_u32 head = task % heads;
    vx_u32 row = task / heads;
    vx_u32 query = query_start + row % query_count;
    vx_u32 batch_index = row / query_count;
    vx_u32 head_offset = head * head_dim;
    vx_u32 qkv_base = batch_index * sequence * d_model * 3u;
    vx_u32 query_base = qkv_base + query * d_model * 3u + head_offset;
    vx_u32 destination = batch_index * sequence * d_model +
        query * d_model + head_offset;
    vx_attention_tiled_body<false>(qkv, qkv, qkv, mask, output, query_base,
        qkv_base + d_model + head_offset,
        qkv_base + 2u * d_model + head_offset, destination, 3u * d_model,
        head_dim, batch_index, query, sequence, sequence, scale, causal,
        mask_mode);
}
