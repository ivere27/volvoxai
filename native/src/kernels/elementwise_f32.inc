#include "mathcompat.h"
#include "thread_pool.h"
#include "kernel_platform.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_ELEM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_ELEM_TARGET_AVX2
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

/* Elementwise F32 arithmetic and the smooth activations. */
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
static VX_ELEM_TARGET_AVX2 int vx_sub_f32_avx2(const float* a, const float* b, float* output, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(output + i, _mm256_sub_ps(a0, b0));
        _mm256_storeu_ps(output + i + 8, _mm256_sub_ps(a1, b1));
        _mm256_storeu_ps(output + i + 16, _mm256_sub_ps(a2, b2));
        _mm256_storeu_ps(output + i + 24, _mm256_sub_ps(a3, b3));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    return i;
}

static VX_ELEM_TARGET_AVX2 int vx_div_f32_avx2(const float* a, const float* b, float* output, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(output + i, _mm256_div_ps(a0, b0));
        _mm256_storeu_ps(output + i + 8, _mm256_div_ps(a1, b1));
        _mm256_storeu_ps(output + i + 16, _mm256_div_ps(a2, b2));
        _mm256_storeu_ps(output + i + 24, _mm256_div_ps(a3, b3));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(output + i, _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    return i;
}

static VX_ELEM_TARGET_AVX2 int vx_leakyrelu_f32_avx2(const float* input, float* output, int n, float alpha) {
    int i = 0;
    const __m256 valpha = _mm256_set1_ps(alpha);
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        __m256 x0 = _mm256_loadu_ps(input + i);
        __m256 x1 = _mm256_loadu_ps(input + i + 8);
        __m256 x2 = _mm256_loadu_ps(input + i + 16);
        __m256 x3 = _mm256_loadu_ps(input + i + 24);
        __m256 m0 = _mm256_cmp_ps(x0, zero, _CMP_GT_OQ);
        __m256 m1 = _mm256_cmp_ps(x1, zero, _CMP_GT_OQ);
        __m256 m2 = _mm256_cmp_ps(x2, zero, _CMP_GT_OQ);
        __m256 m3 = _mm256_cmp_ps(x3, zero, _CMP_GT_OQ);
        __m256 s0 = _mm256_mul_ps(x0, valpha);
        __m256 s1 = _mm256_mul_ps(x1, valpha);
        __m256 s2 = _mm256_mul_ps(x2, valpha);
        __m256 s3 = _mm256_mul_ps(x3, valpha);
        _mm256_storeu_ps(output + i, _mm256_blendv_ps(s0, x0, m0));
        _mm256_storeu_ps(output + i + 8, _mm256_blendv_ps(s1, x1, m1));
        _mm256_storeu_ps(output + i + 16, _mm256_blendv_ps(s2, x2, m2));
        _mm256_storeu_ps(output + i + 24, _mm256_blendv_ps(s3, x3, m3));
    }
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
        __m256 scaled = _mm256_mul_ps(x, valpha);
        __m256 res = _mm256_blendv_ps(scaled, x, mask);
        _mm256_storeu_ps(output + i, res);
    }
    return i;
}

static VX_ELEM_TARGET_AVX2 int vx_clip_f32_avx2(const float* input, float* output, int n, float min_val, float max_val) {
    int i = 0;
    const __m256 vmin = _mm256_set1_ps(min_val);
    const __m256 vmax = _mm256_set1_ps(max_val);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        /* Ordered comparisons preserve NaNs and the sign of an unchanged
         * zero, matching both scalar conditional assignments exactly. */
        v = _mm256_blendv_ps(v, vmin, _mm256_cmp_ps(v, vmin, _CMP_LT_OQ));
        v = _mm256_blendv_ps(v, vmax, _mm256_cmp_ps(v, vmax, _CMP_GT_OQ));
        _mm256_storeu_ps(output + i, v);
    }
    return i;
}
#endif

void sub_f32(const float* a, const float* b, float* output, int n) {
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2) i = vx_sub_f32_avx2(a, b, output, n);
#endif
#if defined(__wasm_simd128__)
    for (; i + 16 <= n; i += 16) {
        v128_t a0 = wasm_v128_load(a + i + 0);
        v128_t a1 = wasm_v128_load(a + i + 4);
        v128_t a2 = wasm_v128_load(a + i + 8);
        v128_t a3 = wasm_v128_load(a + i + 12);
        v128_t b0 = wasm_v128_load(b + i + 0);
        v128_t b1 = wasm_v128_load(b + i + 4);
        v128_t b2 = wasm_v128_load(b + i + 8);
        v128_t b3 = wasm_v128_load(b + i + 12);
        wasm_v128_store(output + i + 0, wasm_f32x4_sub(a0, b0));
        wasm_v128_store(output + i + 4, wasm_f32x4_sub(a1, b1));
        wasm_v128_store(output + i + 8, wasm_f32x4_sub(a2, b2));
        wasm_v128_store(output + i + 12, wasm_f32x4_sub(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        wasm_v128_store(output + i, wasm_f32x4_sub(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
#endif
    for (; i < n; i++) output[i] = a[i] - b[i];
}

void div_f32(const float* a, const float* b, float* output, int n) {
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2 && n >= 8) i = vx_div_f32_avx2(a, b, output, n);
#endif
#if defined(__wasm_simd128__)
    for (; i + 4 <= n; i += 4) {
        wasm_v128_store(output + i, wasm_f32x4_div(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
#endif
    for (; i < n; i++) output[i] = a[i] / b[i];
}

void silu_f32(const float* input, float* output, int n) {
    int i = 0;
#if VX_FASTMATH_WASM_SIMD
    i = vx_silu_f32_wasm_simd(input, output, n);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_silu_f32_neon(input, output, n);
#endif
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_silu_f32_avx2(input, output, n);
#endif
    for (; i < n; i++) output[i] = input[i] * (1.0f / (1.0f + accurate_expf(-input[i])));
}

#if VX_FASTMATH_ARM_NEON
static int vx_leakyrelu_f32_neon(const float* input, float* output, int n, float alpha) {
    int i = 0;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t valpha = vdupq_n_f32(alpha);
    for (; i + 16 <= n; i += 16) {
        float32x4_t x0 = vld1q_f32(input + i + 0);
        float32x4_t x1 = vld1q_f32(input + i + 4);
        float32x4_t x2 = vld1q_f32(input + i + 8);
        float32x4_t x3 = vld1q_f32(input + i + 12);
        uint32x4_t m0 = vcgtq_f32(x0, zero);
        uint32x4_t m1 = vcgtq_f32(x1, zero);
        uint32x4_t m2 = vcgtq_f32(x2, zero);
        uint32x4_t m3 = vcgtq_f32(x3, zero);
        float32x4_t s0 = vmulq_f32(x0, valpha);
        float32x4_t s1 = vmulq_f32(x1, valpha);
        float32x4_t s2 = vmulq_f32(x2, valpha);
        float32x4_t s3 = vmulq_f32(x3, valpha);
        vst1q_f32(output + i + 0, vbslq_f32(m0, x0, s0));
        vst1q_f32(output + i + 4, vbslq_f32(m1, x1, s1));
        vst1q_f32(output + i + 8, vbslq_f32(m2, x2, s2));
        vst1q_f32(output + i + 12, vbslq_f32(m3, x3, s3));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        uint32x4_t m = vcgtq_f32(x, zero);
        float32x4_t s = vmulq_f32(x, valpha);
        vst1q_f32(output + i, vbslq_f32(m, x, s));
    }
    return i;
}
#endif

#if defined(__wasm_simd128__)
static int vx_leakyrelu_f32_wasm(const float* input, float* output, int n, float alpha) {
    int i = 0;
    const v128_t zero = wasm_f32x4_splat(0.0f);
    const v128_t valpha = wasm_f32x4_splat(alpha);
    for (; i + 16 <= n; i += 16) {
        v128_t x0 = wasm_v128_load(input + i + 0);
        v128_t x1 = wasm_v128_load(input + i + 4);
        v128_t x2 = wasm_v128_load(input + i + 8);
        v128_t x3 = wasm_v128_load(input + i + 12);
        v128_t m0 = wasm_f32x4_gt(x0, zero);
        v128_t m1 = wasm_f32x4_gt(x1, zero);
        v128_t m2 = wasm_f32x4_gt(x2, zero);
        v128_t m3 = wasm_f32x4_gt(x3, zero);
        v128_t s0 = wasm_f32x4_mul(x0, valpha);
        v128_t s1 = wasm_f32x4_mul(x1, valpha);
        v128_t s2 = wasm_f32x4_mul(x2, valpha);
        v128_t s3 = wasm_f32x4_mul(x3, valpha);
        wasm_v128_store(output + i + 0, wasm_v128_bitselect(x0, s0, m0));
        wasm_v128_store(output + i + 4, wasm_v128_bitselect(x1, s1, m1));
        wasm_v128_store(output + i + 8, wasm_v128_bitselect(x2, s2, m2));
        wasm_v128_store(output + i + 12, wasm_v128_bitselect(x3, s3, m3));
    }
    for (; i + 4 <= n; i += 4) {
        v128_t x = wasm_v128_load(input + i);
        v128_t m = wasm_f32x4_gt(x, zero);
        v128_t s = wasm_f32x4_mul(x, valpha);
        wasm_v128_store(output + i, wasm_v128_bitselect(x, s, m));
    }
    return i;
}
#endif

typedef struct {
    const float* input;
    float* output;
    float alpha;
} VxLeakyReluF32Context;

static void vx_leakyrelu_f32_worker(void* context, int begin, int end) {
    VxLeakyReluF32Context* ctx = (VxLeakyReluF32Context*)context;
    const float* in = ctx->input + begin;
    float* out = ctx->output + begin;
    int count = end - begin;
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2) i = vx_leakyrelu_f32_avx2(in, out, count, ctx->alpha);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_leakyrelu_f32_neon(in, out, count, ctx->alpha);
#endif
#if defined(__wasm_simd128__)
    i = vx_leakyrelu_f32_wasm(in, out, count, ctx->alpha);
#endif
    for (; i < count; i++) out[i] = in[i] > 0.0f ? in[i] : in[i] * ctx->alpha;
}

void leakyrelu_f32(const float* input, float* output, int n, float alpha) {
    if (n <= 0) return;
    int threads = vx_kernels_thread_count();
    if (threads > 1 && n >= 131072) {
        VxLeakyReluF32Context ctx = {input, output, alpha};
        int grain = 1 + (n - 1) / (threads * 4);
        if (grain < 4096) grain = 4096;
        vx_kernels_parallel_for(n, grain, vx_leakyrelu_f32_worker, &ctx);
        return;
    }
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2) i = vx_leakyrelu_f32_avx2(input, output, n, alpha);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_leakyrelu_f32_neon(input, output, n, alpha);
#endif
#if defined(__wasm_simd128__)
    i = vx_leakyrelu_f32_wasm(input, output, n, alpha);
#endif
    for (; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : input[i] * alpha;
}

typedef struct {
    const float* input;
    float* output;
} VxTanhF32Context;

static void vx_tanh_f32_worker(void* context, int begin, int end) {
    VxTanhF32Context* ctx = (VxTanhF32Context*)context;
    const float* in = ctx->input + begin;
    float* out = ctx->output + begin;
    int count = end - begin;
    int i = 0;
#if VX_FASTMATH_WASM_SIMD
    i = vx_tanh_f32_wasm_simd(in, out, count);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_tanh_f32_neon(in, out, count);
#endif
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_tanh_f32_avx2(in, out, count);
#endif
    for (; i < count; i++) {
        out[i] = tanhf(in[i]);
    }
}

void tanh_f32(const float* input, float* output, int n) {
    if (n <= 0) return;
    int threads = vx_kernels_thread_count();
    /* Below 16K elements, waking and coordinating workers costs more than
     * the vector loop in the measured x86 and ARM paths. Keep small calls
     * on the caller; larger ranges retain their existing chunk arithmetic. */
    if (threads > 1 && n >= 16384) {
        VxTanhF32Context ctx = {input, output};
        int grain = 1 + (n - 1) / (threads * 4);
        if (grain < 256) grain = 256;
        vx_kernels_parallel_for(n, grain, vx_tanh_f32_worker, &ctx);
        return;
    }
    int i = 0;
#if VX_FASTMATH_WASM_SIMD
    i = vx_tanh_f32_wasm_simd(input, output, n);
#endif
#if VX_FASTMATH_ARM_NEON
    if (vx_kernel_platform()->has_neon) i = vx_tanh_f32_neon(input, output, n);
#endif
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_tanh_f32_avx2(input, output, n);
#endif
    for (; i < n; i++) output[i] = tanhf(input[i]);
}

void clip_f32(const float* input, float* output, int n, float min_val, float max_val) {
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2) i = vx_clip_f32_avx2(input, output, n, min_val, max_val);
#endif
    for (; i < n; i++) {
        float v = input[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        output[i] = v;
    }
}
