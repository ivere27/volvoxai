/* Exact grouped-query prefill for B1, Hq32, Hkv4, D64 and 49<=K<=2048.
 * Transposing K makes consecutive keys contiguous without changing any
 * arithmetic. Row tiles retain the cached path's scalar channel dot and
 * ascending K48 denominator/PV partial sums. The host keeps the first 48
 * causal queries on the original short-row algorithm. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_attention_transpose_k_f32(
        const float* key, float* transposed, vx_u32 keys,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        key, transposed, keys);
    key = vx_cuda_global_pointer(key);
    transposed = vx_cuda_global_pointer(transposed);
    VX_CUDA_SHARED float tile[32][33];
    vx_u32 x = threadIdx.x, y = threadIdx.y;
    vx_u32 key0 = blockIdx.x * 32u, channel0 = blockIdx.y * 32u;
    for (vx_u32 row = y; row < 32u; row += 8u)
        tile[row][x] = key0 + row < keys ?
            key[(key0 + row) * 256u + channel0 + x] : 0.0f;
    vx_sync_threads();
    for (vx_u32 row = y; row < 32u; row += 8u)
        if (key0 + x < keys)
            transposed[(channel0 + row) * keys + key0 + x] = tile[x][row];
}

template <vx_u32 Rows, vx_u32 Capacity>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_attention_prefill_exact(
        const float* q, const float* kt, const float* v, const vx_i32* mask,
        float* out, vx_u32 queries, vx_u32 keys, float scale,
        vx_u32 causal, vx_u32 masking, vx_u32 first_query) {
    VX_CUDA_SHARED float scores[Rows * Capacity];
    VX_CUDA_SHARED float stats[Rows * 3u];
    VX_CUDA_SHARED float pv[Rows * 256u];
    VX_CUDA_SHARED float q_cache[Rows * 64u];
    const vx_u32 tid = threadIdx.x, head = blockIdx.x, kvhead = head / 8u;
    const vx_u32 row0 = first_query + blockIdx.y * Rows;
    const vx_u32 qslot = tid / (256u / Rows), qlane = tid % (256u / Rows);
    const vx_u32 query = row0 + qslot;
    const vx_u32 query_active = query >= queries ? 0u :
        (causal && query + 1u < keys ? query + 1u : keys);
    vx_u32 active[Rows];
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (vx_u32 r = 0; r < Rows; ++r)
        active[r] = row0 + r >= queries ? 0u :
            (causal && row0 + r + 1u < keys ? row0 + r + 1u : keys);
    // A tiny CTA-owned Q cache; exactF32 loads/stores only. Invalid tail
    // rows initialize zero and all threads reach the added barrier.
    for (vx_u32 i=tid;i<Rows*64u;i+=256u) {
        const vx_u32 r=i/64u,ch=i%64u;
        q_cache[i]=row0+r<queries ? q[(row0+r)*2048u+head*64u+ch] : 0.0f;
    }
    vx_sync_threads();
    for (vx_u32 key = qlane; key < query_active; key += 256u / Rows) {
        float dot = 0.0f;
        if (!masking || mask[key]) {
#if defined(__CUDA_ARCH__)
            #pragma unroll 4
#endif
            for (vx_u32 ch = 0; ch < 64u; ++ch)
                dot += q_cache[qslot * 64u + ch] *
                    kt[(kvhead * 64u + ch) * keys + key];
        }
        scores[qslot * Capacity + key] = dot * scale;
    }
    vx_sync_threads();
    if (qlane < 32u) {
        float maximum = -3.402823466e38f, any = 0.0f;
        for (vx_u32 key = qlane; key < query_active; key += 32u) {
            if (masking && !mask[key]) continue;
            maximum = vx_max(maximum, scores[qslot * Capacity + key]);
            any = 1.0f;
        }
        for (vx_u32 d = 16u; d; d /= 2u) {
            maximum = vx_max(maximum, vx_cuda_attention_shuffle_down(maximum, d));
            any = vx_max(any, vx_cuda_attention_shuffle_down(any, d));
        }
        if (!qlane) { stats[qslot * 3u] = maximum; stats[qslot * 3u + 2u] = any; }
    }
    vx_sync_threads();
    vx_u32 any_row = 0;
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (vx_u32 r = 0; r < Rows; ++r) any_row |= stats[r * 3u + 2u] != 0.0f;
    if (!any_row) {
        if (tid < 64u) {
#if defined(__CUDA_ARCH__)
            #pragma unroll
#endif
            for (vx_u32 r = 0; r < Rows; ++r)
                if (row0 + r < queries) out[(row0 + r) * 2048u + head * 64u + tid] = 0.0f;
        }
        return;
    }
    for (vx_u32 key = qlane; key < query_active; key += 256u / Rows)
        scores[qslot * Capacity + key] = (!masking || mask[key]) ?
            vx_exp(scores[qslot * Capacity + key] - stats[qslot * 3u]) : 0.0f;
    vx_sync_threads();
    if (qlane < 32u) {
        float denominator = 0.0f;
        for (vx_u32 first = 0; first < query_active; first += 48u * 32u) {
            vx_u32 start = first + qlane * 48u;
            vx_u32 count = start < query_active ?
                (query_active - start < 48u ? query_active - start : 48u) : 0u;
            float partial = 0.0f;
            for (vx_u32 off = 0; off < count; ++off) {
                vx_u32 key = start + off;
                if (masking && !mask[key]) continue;
                partial += scores[qslot * Capacity + key];
            }
            vx_u32 parts = (query_active - first + 47u) / 48u;
            if (parts > 32u) parts = 32u;
            for (vx_u32 part = 0; part < parts; ++part) {
                float value = vx_cuda_attention_shuffle_lane(partial, part);
                if (!qlane) denominator += value;
            }
        }
        if (!qlane) stats[qslot * 3u + 1u] = denominator;
    }
    vx_sync_threads();
    for (vx_u32 key = qlane; key < query_active; key += 256u / Rows)
        scores[qslot * Capacity + key] = stats[qslot * 3u + 2u] != 0.0f ?
            scores[qslot * Capacity + key] / stats[qslot * 3u + 1u] : 0.0f;
    vx_sync_threads();
    const vx_u32 channel = tid % 64u, group = tid / 64u;
    float result[Rows];
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (vx_u32 r = 0; r < Rows; ++r) result[r] = 0.0f;
    vx_u32 maximum_active = 0;
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (vx_u32 r = 0; r < Rows; ++r)
        if (active[r] > maximum_active) maximum_active = active[r];
    for (vx_u32 first = 0; first < maximum_active; first += 192u) {
        float partial[Rows];
#if defined(__CUDA_ARCH__)
        #pragma unroll
#endif
        for (vx_u32 r = 0; r < Rows; ++r) partial[r] = 0.0f;
        for (vx_u32 off = 0; off < 48u; ++off) {
            vx_u32 key = first + group * 48u + off;
            if (key >= maximum_active || (masking && !mask[key])) continue;
            float value = v[key * 256u + kvhead * 64u + channel];
#if defined(__CUDA_ARCH__)
            #pragma unroll
#endif
            for (vx_u32 r = 0; r < Rows; ++r)
                if (key < active[r] && stats[r * 3u + 2u] != 0.0f)
                    partial[r] += scores[r * Capacity + key] * value;
        }
#if defined(__CUDA_ARCH__)
        #pragma unroll
#endif
        for (vx_u32 r = 0; r < Rows; ++r) pv[r * 256u + tid] = partial[r];
        vx_sync_threads();
        if (tid < 64u) {
#if defined(__CUDA_ARCH__)
            #pragma unroll
#endif
            for (vx_u32 r = 0; r < Rows; ++r)
#if defined(__CUDA_ARCH__)
                #pragma unroll
#endif
                for (vx_u32 part = 0; part < 4u; ++part)
                    if (first + part * 48u < active[r])
                        result[r] += pv[r * 256u + part * 64u + channel];
        }
        vx_sync_threads();
    }
    if (tid < 64u) {
#if defined(__CUDA_ARCH__)
        #pragma unroll
#endif
        for (vx_u32 r = 0; r < Rows; ++r)
            if (row0 + r < queries)
                out[(row0 + r) * 2048u + head * 64u + channel] = result[r];
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_attention_prefill_r4_k512_f32(
        const float* q, const float* kt, const float* v, const vx_i32* mask,
        float* out, vx_u32 queries, vx_u32 keys, float scale,
        vx_u32 causal, vx_u32 masking, vx_u32 first_query,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        q, kt, v, mask, out, queries, keys, scale, causal, masking, first_query);
    vx_attention_prefill_exact<4u, 512u>(vx_cuda_global_pointer(q),
        vx_cuda_global_pointer(kt), vx_cuda_global_pointer(v),
        vx_cuda_global_pointer(mask), vx_cuda_global_pointer(out),
        queries, keys, scale, causal, masking, first_query);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_attention_prefill_r2_k2048_f32(
        const float* q, const float* kt, const float* v, const vx_i32* mask,
        float* out, vx_u32 queries, vx_u32 keys, float scale,
        vx_u32 causal, vx_u32 masking, vx_u32 first_query,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        q, kt, v, mask, out, queries, keys, scale, causal, masking, first_query);
    vx_attention_prefill_exact<2u, 2048u>(vx_cuda_global_pointer(q),
        vx_cuda_global_pointer(kt), vx_cuda_global_pointer(v),
        vx_cuda_global_pointer(mask), vx_cuda_global_pointer(out),
        queries, keys, scale, causal, masking, first_query);
}
