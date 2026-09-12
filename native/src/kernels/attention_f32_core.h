#ifndef VOLVOXAI_ATTENTION_F32_CORE_H
#define VOLVOXAI_ATTENTION_F32_CORE_H

/*
 * The portable F32 attention body, shared by sdpa.c and cross_sdpa.c.
 *
 * This is the reference the vectorized kernel in attention_f32_tiled.h is
 * checked against, and it is what runs on a CPU without AVX2 and in the
 * freestanding wasm32 build.
 *
 * Storage is described by base pointers plus a shared q/k/v row stride and a
 * separate output stride, which covers both callers: cross attention passes
 * three tensors each strided by d_model, while self attention passes one fused
 * QKV tensor strided by 3*d_model with K and V offset into it.  That is the only
 * thing that used to differ between them, so there is one function here rather
 * than a per-caller instantiation.
 *
 * The recurrence is online softmax, so each key is visited once and the Q.K dot
 * is computed exactly once.  What it replaced walked the key axis three times;
 * in sdpa.c the third walk sat inside a loop over output channels, making that
 * pass O(seq_kv * head_dim^2) and the whole kernel cost (head_dim + 2) dot
 * products per (query, head) where one is needed.
 *
 * accurate_expf, not the cheaper fast_expf.  A two-pass softmax divides
 * exp(s - m) by a sum of terms carrying the same approximation, so fast_expf's
 * few-percent error largely cancels in the ratio.  This recurrence has no such
 * cancellation: each new maximum multiplies the running denominator and
 * accumulator by a correction factor, so the error compounds across keys.  That
 * can push the finite-difference attention gradient outside a useful tolerance. It also
 * stays a polynomial rather than expf so the wasm32 build does not take a host
 * import once per key.
 */

/*
 * The four head_dim loops of the online softmax, vectorized where WASM has
 * vectors.
 *
 * WASM runs this kernel and not the tiled one — attention_f32_tiled.h is
 * guarded by VX_SDPA_X86_AVX2 — so on that target these loops were scalar and
 * `CrossSDPA` cost 237 ms of a 1607 ms encoder.
 *
 * Three of the four are elementwise in `d` and vectorize with no change to any
 * addition order at all. The dot product is a reduction and does change it,
 * which is admissible here and only here: cross_sdpa.c already documents the
 * AVX2 and portable paths agreeing to a tolerance rather than exactly, and
 * row-versus-full consistency is preserved because a row and a full forward
 * run this same code.
 */
#if defined(__wasm_simd128__)

static float vx_attention_dot_f32(const float* a, const float* b, int n) {
    v128_t acc = wasm_f32x4_splat(0.0f);
    float lane[4];
    float sum;
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(a + d),
                                                 wasm_v128_load(b + d)));
    }
    wasm_v128_store(lane, acc);
    sum = lane[0] + lane[1] + lane[2] + lane[3];
    for (; d < n; d++) sum += a[d] * b[d];
    return sum;
}

static void vx_attention_axpy_f32(float* acc, const float* v, float w, int n) {
    const v128_t wv = wasm_f32x4_splat(w);
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        wasm_v128_store(acc + d, wasm_f32x4_add(wasm_v128_load(acc + d),
            wasm_f32x4_mul(wv, wasm_v128_load(v + d))));
    }
    for (; d < n; d++) acc[d] += w * v[d];
}

static void vx_attention_rescale_f32(float* acc, const float* v, float c, int n) {
    const v128_t cv = wasm_f32x4_splat(c);
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        wasm_v128_store(acc + d, wasm_f32x4_add(
            wasm_f32x4_mul(wasm_v128_load(acc + d), cv), wasm_v128_load(v + d)));
    }
    for (; d < n; d++) acc[d] = acc[d] * c + v[d];
}

static void vx_attention_scale_f32(float* acc, float s, int n) {
    const v128_t sv = wasm_f32x4_splat(s);
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        wasm_v128_store(acc + d, wasm_f32x4_mul(wasm_v128_load(acc + d), sv));
    }
    for (; d < n; d++) acc[d] *= s;
}

#else

/* Every other target keeps the scalar loops verbatim: AVX2 reaches the tiled
 * kernel and never arrives here, and a portable build must stay portable. */
static float vx_attention_dot_f32(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int d = 0; d < n; d++) sum += a[d] * b[d];
    return sum;
}

static void vx_attention_axpy_f32(float* acc, const float* v, float w, int n) {
    for (int d = 0; d < n; d++) acc[d] += w * v[d];
}

static void vx_attention_rescale_f32(float* acc, const float* v, float c, int n) {
    for (int d = 0; d < n; d++) acc[d] = acc[d] * c + v[d];
}

static void vx_attention_scale_f32(float* acc, float s, int n) {
    for (int d = 0; d < n; d++) acc[d] *= s;
}

#endif

static void vx_attention_f32_portable(
        const float* q_in, const float* k_in, const float* v_in,
        long qkv_stride, float* output, long out_stride,
        int seq_q, int seq_kv, int num_heads, int head_dim, float scale,
        const int32_t* mask, int mask_mode, int causal) {
    for (int q_idx = 0; q_idx < seq_q; q_idx++) {
        const int key_limit = vx_attention_key_limit(seq_kv, q_idx, causal);
        for (int h_idx = 0; h_idx < num_heads; h_idx++) {
            const long head_offset = (long)h_idx * head_dim;
            const float* query = q_in + (long)q_idx * qkv_stride + head_offset;
            /* Accumulating in place keeps the working set to the output row the
             * caller already owns; the final scaling pass normalizes it. */
            float* accumulator = output + (long)q_idx * out_stride + head_offset;
            float maximum = 0.0f;
            float denominator = 0.0f;
            int have_key = 0;
            int d;
            for (int k_idx = 0; k_idx < key_limit; k_idx++) {
                const float* key;
                const float* value;
                float score = 0.0f;
                if (!vx_attention_key_allowed(mask, mask_mode, q_idx, k_idx,
                                              seq_kv, causal)) continue;
                key = k_in + (long)k_idx * qkv_stride + head_offset;
                value = v_in + (long)k_idx * qkv_stride + head_offset;
                score = vx_attention_dot_f32(query, key, head_dim);
                score *= scale;
                if (!have_key) {
                    /* Seeding from the first surviving key keeps an infinite
                     * sentinel out of the recurrence, so no exp(-inf) is ever
                     * evaluated and the all-masked case stays a separate exit. */
                    maximum = score;
                    denominator = 1.0f;
                    for (d = 0; d < head_dim; d++) accumulator[d] = value[d];
                    have_key = 1;
                } else if (score <= maximum) {
                    const float weight = accurate_expf(score - maximum);
                    denominator += weight;
                    vx_attention_axpy_f32(accumulator, value, weight, head_dim);
                } else {
                    /* A new maximum rescales everything summed so far. */
                    const float correction = accurate_expf(maximum - score);
                    denominator = denominator * correction + 1.0f;
                    vx_attention_rescale_f32(accumulator, value, correction,
                                             head_dim);
                    maximum = score;
                }
            }
            if (!have_key) {
                for (d = 0; d < head_dim; d++) accumulator[d] = 0.0f;
            } else {
                vx_attention_scale_f32(accumulator, 1.0f / denominator,
                                       head_dim);
            }
        }
    }
}

#endif
