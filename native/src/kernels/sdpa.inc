#include "mathcompat.h"
#include "attention_f32_isa.h"
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
// --- 6. SDPA (Self Attention) ---
/*
 * Self attention over a fused QKV tensor.
 *
 * The previous implementation walked the key axis three times per (query, head)
 * and recomputed the Q.K dot on each walk — and the third walk was nested inside
 * a loop over output channels, so it recomputed the whole dot once per channel.
 * That made the final pass O(seq_kv * head_dim^2) and the kernel as a whole cost
 * (head_dim + 2) dot products per (query, head) where one is needed: 42x the
 * necessary arithmetic at this model's head_dim of 40.  cross_sdpa.c had already
 * been given the shared-probability form for exactly this reason; this file had
 * not.
 *
 * Both now share the online-softmax core in attention_f32_core.h, which visits
 * each key once, and select an AVX2 body at runtime through
 * vx_kernel_platform().  Storage is the only difference: Q, K and V are slices
 * of one tensor strided by 3*d_model, while the output is strided by d_model.
 */
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
static __attribute__((target("avx2"))) int vx_add_f32_avx2(
        const float* a, const float* b, float* out, int n) {
    int i = 0;
    if (a == b) {
        for (; i + 32 <= n; i += 32) {
            __m256 a0 = _mm256_loadu_ps(a + i);
            __m256 a1 = _mm256_loadu_ps(a + i + 8);
            __m256 a2 = _mm256_loadu_ps(a + i + 16);
            __m256 a3 = _mm256_loadu_ps(a + i + 24);
            _mm256_storeu_ps(out + i, _mm256_add_ps(a0, a0));
            _mm256_storeu_ps(out + i + 8, _mm256_add_ps(a1, a1));
            _mm256_storeu_ps(out + i + 16, _mm256_add_ps(a2, a2));
            _mm256_storeu_ps(out + i + 24, _mm256_add_ps(a3, a3));
        }
        for (; i + 8 <= n; i += 8) {
            __m256 a0 = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(a0, a0));
        }
        return i;
    }
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_add_ps(a0, b0));
        _mm256_storeu_ps(out + i + 8, _mm256_add_ps(a1, b1));
        _mm256_storeu_ps(out + i + 16, _mm256_add_ps(a2, b2));
        _mm256_storeu_ps(out + i + 24, _mm256_add_ps(a3, b3));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    return i;
}
#endif

void add_f32(const float* a, const float* b, float* out, int n) {
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (__builtin_cpu_supports("avx2")) {
        i = vx_add_f32_avx2(a, b, out, n);
    }
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i + 0);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i + 0);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        vst1q_f32(out + i + 0, vaddq_f32(a0, b0));
        vst1q_f32(out + i + 4, vaddq_f32(a1, b1));
        vst1q_f32(out + i + 8, vaddq_f32(a2, b2));
        vst1q_f32(out + i + 12, vaddq_f32(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(out + i, wasm_f32x4_add(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}


void sdpa_f32(const float* qkv, float* output, int seq_len, int d_model,
              int num_heads, int head_dim, float scale, const int32_t* mask,
              int mask_mode, int causal) {
    /* Q, K and V are consecutive d_model blocks of each row. */
    const long stride = (long)d_model * 3;
    const float* q_in = qkv;
    const float* k_in = qkv + d_model;
    const float* v_in = qkv + (long)d_model * 2;
#if VX_SDPA_X86_AVX2
    if (head_dim <= VX_SDPA_MAX_HEAD_DIM && vx_kernel_platform()->has_avx2 &&
            __builtin_cpu_supports("fma")) {
        vx_attention_tiled_avx2(q_in, k_in, v_in, stride, output, d_model,
                                seq_len, seq_len, num_heads, head_dim, scale,
                                mask, mask_mode, causal);
        return;
    }
#endif
    vx_attention_f32_portable(q_in, k_in, v_in, stride, output, d_model,
                              seq_len, seq_len, num_heads, head_dim, scale,
                              mask, mask_mode, causal);
}
