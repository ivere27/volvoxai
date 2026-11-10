#include "conv_f32_isa.h"
#include "kernel_platform.h"
#include "runtime_state.h"
#include "thread_pool.h"
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
extern void conv2d_depthwise_s2_avx2(const float*, float*, const float*, const float*,
                                     int, int, int, int, int, int, int, int, int, int, int, int);
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VX_CONV_USE_NEON 1
#endif
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define VX_CONV_USE_WASM_SIMD128 1
#endif

#define g_pwf32_pack (vx_engine_state_current()->conv_pwf32_pack)
#define g_dw_pw_tmp (vx_engine_state_current()->conv_dw_pw_tmp)
#define g_dw_pw_tmp_cap (vx_engine_state_current()->conv_dw_pw_tmp_cap)
#define g_f32_igemm_indir (vx_engine_state_current()->conv_f32_igemm_indir)
#define g_f32_igemm_indir_cap \
    (vx_engine_state_current()->conv_f32_igemm_indir_cap)
#define g_f32_igemm_indir_key \
    (vx_engine_state_current()->conv_f32_igemm_indir_key)
#define g_f32_igemm_zero (vx_engine_state_current()->conv_f32_igemm_zero)
#define g_f32_igemm_zero_cap \
    (vx_engine_state_current()->conv_f32_igemm_zero_cap)

static inline float vx_relu6_apply(float x, int relu) {
    if (relu) {
        if (x < 0.0f) x = 0.0f;
        if (relu >= 2 && x > 6.0f) x = 6.0f;
    }
    return x;
}

// Pack pointwise weights (stored HWIO = [ic][oc]) into an s4 tile:
// [oc/16][ic/4][4 rotated-k steps][16 oc lanes].
const float* vx_pwf32_pack_cache(int node_idx, const float* wgt, int c, int out_c) {
    if (node_idx < 0 || (size_t)node_idx >= vx_engine_state_current()->node_capacity ||
        out_c < 16 || c < 4) return NULL;
    if (g_pwf32_pack[node_idx]) return g_pwf32_pack[node_idx];
    int nb = out_c / 16;
    int k4 = c / 4;
    float* pk = (float*)malloc((size_t)nb * k4 * 64 * sizeof(float));
    if (!pk) return NULL;
    for (int ob = 0; ob < nb; ob++) {
        int oc = ob * 16;
        for (int kg = 0; kg < k4; kg++) {
            float* dst = pk + ((long)ob * k4 + kg) * 64;
            int kbase = kg * 4;
            for (int step = 0; step < 4; step++) {
                for (int j = 0; j < 16; j++) {
                    int kk = kbase + ((step + (j & 3)) & 3);
                    dst[step * 16 + j] = wgt[(long)kk * out_c + oc + j];
                }
            }
        }
    }
    g_pwf32_pack[node_idx] = pk;
    return pk;
}

// Plain GEMM weight pack: [oc/16][ic][16] with lane l holding wgt[ic*out_c + oc+l].
// Feeds a broadcast-based MR6xNR16 microkernel (12 accumulators) — no permute/shuffle
// ports, unlike the s4 path. Better on the thin-K expand convs (K=in_c small) where
// per-tile startup dominates and 6 pixels amortize it better than 4.
#define g_pwf32_pack_plain \
    (vx_engine_state_current()->conv_pwf32_pack_plain)

static const float* vx_pwf32_pack_plain_cache(int node_idx, const float* wgt, int c, int out_c) {
    if (node_idx < 0 || (size_t)node_idx >= vx_engine_state_current()->node_capacity ||
        out_c < 16 || c < 1) return NULL;
    if (g_pwf32_pack_plain[node_idx]) return g_pwf32_pack_plain[node_idx];
    int nb = out_c / 16;
    float* pk = (float*)malloc((size_t)nb * c * 16 * sizeof(float));
    if (!pk) return NULL;
    for (int ob = 0; ob < nb; ob++) {
        int oc = ob * 16;
        for (int ic = 0; ic < c; ic++) {
            float* dst = pk + ((long)ob * c + ic) * 16;
            for (int lane = 0; lane < 16; lane++)
                dst[lane] = wgt[(long)ic * out_c + oc + lane];
        }
    }
    g_pwf32_pack_plain[node_idx] = pk;
    return pk;
}

// Default-on: the plain MR6xNR16 GEMM measured ~5% faster (warm) than the s4
// permute path on this AVX2 CPU. Set VOLVOX_PW_GEMM=0 before creating an
// engine to fall back to the s4 path for that engine.
static int vx_pw_gemm_enabled(void) {
    return vx_engine_state_current()->conv_pw_gemm_enabled;
}

typedef struct {
    const float* input;
    float* output;
    const float* wgt;
    const float* bias;
    const float* addp;
    const float* pk;
    long pixels;
    int c;
    int out_c;
    int oc16;
    int relu;
} VxPwGemmRangeContext;

#if defined(__AVX2__)

static void vx_conv2d_pointwise_gemm_range_avx2(
        const VxPwGemmRangeContext* ctx, int oc_begin, int oc_end) {
    const float* input = ctx->input;
    float* output = ctx->output;
    const float* wgt = ctx->wgt;
    const float* bias = ctx->bias;
    const float* addp = ctx->addp;
    const float* pk = ctx->pk;
    const long pixels = ctx->pixels;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int relu = ctx->relu;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#define VX_PWG_RELU(v) do { if (relu) { (v) = _mm256_max_ps((v), zero); if (relu >= 2) (v) = _mm256_min_ps((v), six); } } while (0)

    int oc = oc_begin;
    for (; oc + 16 <= oc_end && oc + 16 <= ctx->oc16; oc += 16) {
        const float* wp = pk + (long)(oc >> 4) * c * 16;
        __m256 bl = bias ? _mm256_loadu_ps(bias + oc) : zero;
        __m256 bh = bias ? _mm256_loadu_ps(bias + oc + 8) : zero;

        long p = 0;
        for (; p + 6 <= pixels; p += 6) {
            const float* s0 = input + (p + 0) * c, * s1 = input + (p + 1) * c, * s2 = input + (p + 2) * c;
            const float* s3 = input + (p + 3) * c, * s4 = input + (p + 4) * c, * s5 = input + (p + 5) * c;
            float* o0 = output + (p + 0) * out_c, * o1 = output + (p + 1) * out_c, * o2 = output + (p + 2) * out_c;
            float* o3 = output + (p + 3) * out_c, * o4 = output + (p + 4) * out_c, * o5 = output + (p + 5) * out_c;

            __m256 a0l = bl, a0h = bh, a1l = bl, a1h = bh, a2l = bl, a2h = bh;
            __m256 a3l = bl, a3h = bh, a4l = bl, a4h = bh, a5l = bl, a5h = bh;
            for (int ic = 0; ic < c; ic++) {
                __m256 wl = _mm256_loadu_ps(wp + ic * 16);
                __m256 wh = _mm256_loadu_ps(wp + ic * 16 + 8);
                __m256 x;
                x = _mm256_set1_ps(s0[ic]); a0l = _mm256_fmadd_ps(x, wl, a0l); a0h = _mm256_fmadd_ps(x, wh, a0h);
                x = _mm256_set1_ps(s1[ic]); a1l = _mm256_fmadd_ps(x, wl, a1l); a1h = _mm256_fmadd_ps(x, wh, a1h);
                x = _mm256_set1_ps(s2[ic]); a2l = _mm256_fmadd_ps(x, wl, a2l); a2h = _mm256_fmadd_ps(x, wh, a2h);
                x = _mm256_set1_ps(s3[ic]); a3l = _mm256_fmadd_ps(x, wl, a3l); a3h = _mm256_fmadd_ps(x, wh, a3h);
                x = _mm256_set1_ps(s4[ic]); a4l = _mm256_fmadd_ps(x, wl, a4l); a4h = _mm256_fmadd_ps(x, wh, a4h);
                x = _mm256_set1_ps(s5[ic]); a5l = _mm256_fmadd_ps(x, wl, a5l); a5h = _mm256_fmadd_ps(x, wh, a5h);
            }
            VX_PWG_RELU(a0l); VX_PWG_RELU(a0h); VX_PWG_RELU(a1l); VX_PWG_RELU(a1h);
            VX_PWG_RELU(a2l); VX_PWG_RELU(a2h); VX_PWG_RELU(a3l); VX_PWG_RELU(a3h);
            VX_PWG_RELU(a4l); VX_PWG_RELU(a4h); VX_PWG_RELU(a5l); VX_PWG_RELU(a5h);
            if (addp) {
                const float* r0 = addp + (p + 0) * out_c + oc, * r1 = addp + (p + 1) * out_c + oc, * r2 = addp + (p + 2) * out_c + oc;
                const float* r3 = addp + (p + 3) * out_c + oc, * r4 = addp + (p + 4) * out_c + oc, * r5 = addp + (p + 5) * out_c + oc;
                a0l = _mm256_add_ps(a0l, _mm256_loadu_ps(r0)); a0h = _mm256_add_ps(a0h, _mm256_loadu_ps(r0 + 8));
                a1l = _mm256_add_ps(a1l, _mm256_loadu_ps(r1)); a1h = _mm256_add_ps(a1h, _mm256_loadu_ps(r1 + 8));
                a2l = _mm256_add_ps(a2l, _mm256_loadu_ps(r2)); a2h = _mm256_add_ps(a2h, _mm256_loadu_ps(r2 + 8));
                a3l = _mm256_add_ps(a3l, _mm256_loadu_ps(r3)); a3h = _mm256_add_ps(a3h, _mm256_loadu_ps(r3 + 8));
                a4l = _mm256_add_ps(a4l, _mm256_loadu_ps(r4)); a4h = _mm256_add_ps(a4h, _mm256_loadu_ps(r4 + 8));
                a5l = _mm256_add_ps(a5l, _mm256_loadu_ps(r5)); a5h = _mm256_add_ps(a5h, _mm256_loadu_ps(r5 + 8));
            }
            _mm256_storeu_ps(o0 + oc, a0l); _mm256_storeu_ps(o0 + oc + 8, a0h);
            _mm256_storeu_ps(o1 + oc, a1l); _mm256_storeu_ps(o1 + oc + 8, a1h);
            _mm256_storeu_ps(o2 + oc, a2l); _mm256_storeu_ps(o2 + oc + 8, a2h);
            _mm256_storeu_ps(o3 + oc, a3l); _mm256_storeu_ps(o3 + oc + 8, a3h);
            _mm256_storeu_ps(o4 + oc, a4l); _mm256_storeu_ps(o4 + oc + 8, a4h);
            _mm256_storeu_ps(o5 + oc, a5l); _mm256_storeu_ps(o5 + oc + 8, a5h);
        }
        for (; p < pixels; p++) {
            const float* src = input + p * c;
            float* dst = output + p * out_c;
            __m256 al = bl, ah = bh;
            for (int ic = 0; ic < c; ic++) {
                __m256 x = _mm256_set1_ps(src[ic]);
                al = _mm256_fmadd_ps(x, _mm256_loadu_ps(wp + ic * 16), al);
                ah = _mm256_fmadd_ps(x, _mm256_loadu_ps(wp + ic * 16 + 8), ah);
            }
            VX_PWG_RELU(al); VX_PWG_RELU(ah);
            if (addp) {
                al = _mm256_add_ps(al, _mm256_loadu_ps(addp + p * out_c + oc));
                ah = _mm256_add_ps(ah, _mm256_loadu_ps(addp + p * out_c + oc + 8));
            }
            _mm256_storeu_ps(dst + oc, al);
            _mm256_storeu_ps(dst + oc + 8, ah);
        }
    }
    for (; oc < oc_end; oc++) {
        for (long p = 0; p < pixels; p++) {
            const float* src = input + p * c;
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++) sum += src[ic] * wgt[(long)ic * out_c + oc];
            output[p * out_c + oc] = vx_relu6_apply(sum, relu) + (addp ? addp[p * out_c + oc] : 0.0f);
        }
    }
#undef VX_PWG_RELU
}

static void vx_conv2d_pointwise_gemm_worker(void* vctx, int begin, int end) {
    const VxPwGemmRangeContext* ctx = (const VxPwGemmRangeContext*)vctx;
    int oc_begin = begin * 16;
    int oc_end = end * 16;
    if (oc_end > ctx->out_c) oc_end = ctx->out_c;
    vx_conv2d_pointwise_gemm_range_avx2(ctx, oc_begin, oc_end);
}

static void vx_conv2d_pointwise_gemm_f32(int node_idx,
                                              const float* input, float* output,
                                              const float* wgt, const float* bias,
                                              const float* addp,
                                              long pixels, int c, int out_c, int relu) {
    const float* pk = vx_pwf32_pack_plain_cache(node_idx, wgt, c, out_c);
    int oc16 = (pk) ? (out_c & ~15) : 0;
    VxPwGemmRangeContext ctx = {
        .input = input, .output = output, .wgt = wgt, .bias = bias,
        .addp = addp, .pk = pk, .pixels = pixels, .c = c,
        .out_c = out_c, .oc16 = oc16, .relu = relu,
    };
    const int threads = vx_kernels_thread_count();
    const int oc_blocks = (out_c + 15) / 16;
    const uint64_t total_flops = (uint64_t)pixels * c * out_c * 2u;
    if (threads > 1 && (total_flops >= 262144u || (total_flops >= 65536u && oc_blocks >= threads * 2)) && oc_blocks > 1) {
        int grain = (oc_blocks + threads - 1) / threads;
        if (grain < 1) grain = 1;
        vx_kernels_parallel_for(oc_blocks, grain, vx_conv2d_pointwise_gemm_worker, &ctx);
    } else {
        vx_conv2d_pointwise_gemm_range_avx2(&ctx, 0, out_c);
    }
}
#endif

#if defined(VX_CONV_USE_NEON)
static void vx_conv2d_pointwise_gemm_range_neon(
        const VxPwGemmRangeContext* ctx, int oc_begin, int oc_end) {
    const float* input = ctx->input;
    float* output = ctx->output;
    const float* wgt = ctx->wgt;
    const float* bias = ctx->bias;
    const float* addp = ctx->addp;
    const float* pk = ctx->pk;
    const long pixels = ctx->pixels;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int relu = ctx->relu;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
#define VX_PWG_NEON_RELU(v) do { \
    if (relu) { \
        (v) = vmaxq_f32((v), zero); \
        if (relu >= 2) (v) = vminq_f32((v), six); \
    } \
} while (0)

    int oc = oc_begin;
    for (; oc + 16 <= oc_end && oc + 16 <= ctx->oc16; oc += 16) {
        const float* wp = pk + (long)(oc >> 4) * c * 16;
        const float32x4_t b0 = bias ? vld1q_f32(bias + oc + 0) : zero;
        const float32x4_t b1 = bias ? vld1q_f32(bias + oc + 4) : zero;
        const float32x4_t b2 = bias ? vld1q_f32(bias + oc + 8) : zero;
        const float32x4_t b3 = bias ? vld1q_f32(bias + oc + 12) : zero;

        long p = 0;
        for (; p + 4 <= pixels; p += 4) {
            const float* s0 = input + (p + 0) * c;
            const float* s1 = input + (p + 1) * c;
            const float* s2 = input + (p + 2) * c;
            const float* s3 = input + (p + 3) * c;
            float* o0 = output + (p + 0) * out_c;
            float* o1 = output + (p + 1) * out_c;
            float* o2 = output + (p + 2) * out_c;
            float* o3 = output + (p + 3) * out_c;

            float32x4_t a0_0 = b0, a0_1 = b1, a0_2 = b2, a0_3 = b3;
            float32x4_t a1_0 = b0, a1_1 = b1, a1_2 = b2, a1_3 = b3;
            float32x4_t a2_0 = b0, a2_1 = b1, a2_2 = b2, a2_3 = b3;
            float32x4_t a3_0 = b0, a3_1 = b1, a3_2 = b2, a3_3 = b3;

            for (int ic = 0; ic < c; ic++) {
                const float32x4_t w0 = vld1q_f32(wp + ic * 16 + 0);
                const float32x4_t w1 = vld1q_f32(wp + ic * 16 + 4);
                const float32x4_t w2 = vld1q_f32(wp + ic * 16 + 8);
                const float32x4_t w3 = vld1q_f32(wp + ic * 16 + 12);

                const float x0 = s0[ic];
                a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                a0_3 = vmlaq_n_f32(a0_3, w3, x0);

                const float x1 = s1[ic];
                a1_0 = vmlaq_n_f32(a1_0, w0, x1);
                a1_1 = vmlaq_n_f32(a1_1, w1, x1);
                a1_2 = vmlaq_n_f32(a1_2, w2, x1);
                a1_3 = vmlaq_n_f32(a1_3, w3, x1);

                const float x2 = s2[ic];
                a2_0 = vmlaq_n_f32(a2_0, w0, x2);
                a2_1 = vmlaq_n_f32(a2_1, w1, x2);
                a2_2 = vmlaq_n_f32(a2_2, w2, x2);
                a2_3 = vmlaq_n_f32(a2_3, w3, x2);

                const float x3 = s3[ic];
                a3_0 = vmlaq_n_f32(a3_0, w0, x3);
                a3_1 = vmlaq_n_f32(a3_1, w1, x3);
                a3_2 = vmlaq_n_f32(a3_2, w2, x3);
                a3_3 = vmlaq_n_f32(a3_3, w3, x3);
            }

            VX_PWG_NEON_RELU(a0_0); VX_PWG_NEON_RELU(a0_1); VX_PWG_NEON_RELU(a0_2); VX_PWG_NEON_RELU(a0_3);
            VX_PWG_NEON_RELU(a1_0); VX_PWG_NEON_RELU(a1_1); VX_PWG_NEON_RELU(a1_2); VX_PWG_NEON_RELU(a1_3);
            VX_PWG_NEON_RELU(a2_0); VX_PWG_NEON_RELU(a2_1); VX_PWG_NEON_RELU(a2_2); VX_PWG_NEON_RELU(a2_3);
            VX_PWG_NEON_RELU(a3_0); VX_PWG_NEON_RELU(a3_1); VX_PWG_NEON_RELU(a3_2); VX_PWG_NEON_RELU(a3_3);

            if (addp) {
                a0_0 = vaddq_f32(a0_0, vld1q_f32(addp + (p + 0) * out_c + oc + 0));
                a0_1 = vaddq_f32(a0_1, vld1q_f32(addp + (p + 0) * out_c + oc + 4));
                a0_2 = vaddq_f32(a0_2, vld1q_f32(addp + (p + 0) * out_c + oc + 8));
                a0_3 = vaddq_f32(a0_3, vld1q_f32(addp + (p + 0) * out_c + oc + 12));

                a1_0 = vaddq_f32(a1_0, vld1q_f32(addp + (p + 1) * out_c + oc + 0));
                a1_1 = vaddq_f32(a1_1, vld1q_f32(addp + (p + 1) * out_c + oc + 4));
                a1_2 = vaddq_f32(a1_2, vld1q_f32(addp + (p + 1) * out_c + oc + 8));
                a1_3 = vaddq_f32(a1_3, vld1q_f32(addp + (p + 1) * out_c + oc + 12));

                a2_0 = vaddq_f32(a2_0, vld1q_f32(addp + (p + 2) * out_c + oc + 0));
                a2_1 = vaddq_f32(a2_1, vld1q_f32(addp + (p + 2) * out_c + oc + 4));
                a2_2 = vaddq_f32(a2_2, vld1q_f32(addp + (p + 2) * out_c + oc + 8));
                a2_3 = vaddq_f32(a2_3, vld1q_f32(addp + (p + 2) * out_c + oc + 12));

                a3_0 = vaddq_f32(a3_0, vld1q_f32(addp + (p + 3) * out_c + oc + 0));
                a3_1 = vaddq_f32(a3_1, vld1q_f32(addp + (p + 3) * out_c + oc + 4));
                a3_2 = vaddq_f32(a3_2, vld1q_f32(addp + (p + 3) * out_c + oc + 8));
                a3_3 = vaddq_f32(a3_3, vld1q_f32(addp + (p + 3) * out_c + oc + 12));
            }

            vst1q_f32(o0 + oc + 0, a0_0);
            vst1q_f32(o0 + oc + 4, a0_1);
            vst1q_f32(o0 + oc + 8, a0_2);
            vst1q_f32(o0 + oc + 12, a0_3);

            vst1q_f32(o1 + oc + 0, a1_0);
            vst1q_f32(o1 + oc + 4, a1_1);
            vst1q_f32(o1 + oc + 8, a1_2);
            vst1q_f32(o1 + oc + 12, a1_3);

            vst1q_f32(o2 + oc + 0, a2_0);
            vst1q_f32(o2 + oc + 4, a2_1);
            vst1q_f32(o2 + oc + 8, a2_2);
            vst1q_f32(o2 + oc + 12, a2_3);

            vst1q_f32(o3 + oc + 0, a3_0);
            vst1q_f32(o3 + oc + 4, a3_1);
            vst1q_f32(o3 + oc + 8, a3_2);
            vst1q_f32(o3 + oc + 12, a3_3);
        }

        for (; p < pixels; p++) {
            const float* src = input + p * c;
            float* dst = output + p * out_c;
            float32x4_t a0 = b0, a1 = b1, a2 = b2, a3 = b3;
            for (int ic = 0; ic < c; ic++) {
                const float x = src[ic];
                a0 = vmlaq_n_f32(a0, vld1q_f32(wp + ic * 16 + 0), x);
                a1 = vmlaq_n_f32(a1, vld1q_f32(wp + ic * 16 + 4), x);
                a2 = vmlaq_n_f32(a2, vld1q_f32(wp + ic * 16 + 8), x);
                a3 = vmlaq_n_f32(a3, vld1q_f32(wp + ic * 16 + 12), x);
            }
            VX_PWG_NEON_RELU(a0); VX_PWG_NEON_RELU(a1); VX_PWG_NEON_RELU(a2); VX_PWG_NEON_RELU(a3);
            if (addp) {
                a0 = vaddq_f32(a0, vld1q_f32(addp + p * out_c + oc + 0));
                a1 = vaddq_f32(a1, vld1q_f32(addp + p * out_c + oc + 4));
                a2 = vaddq_f32(a2, vld1q_f32(addp + p * out_c + oc + 8));
                a3 = vaddq_f32(a3, vld1q_f32(addp + p * out_c + oc + 12));
            }
            vst1q_f32(dst + oc + 0, a0);
            vst1q_f32(dst + oc + 4, a1);
            vst1q_f32(dst + oc + 8, a2);
            vst1q_f32(dst + oc + 12, a3);
        }
    }
    for (; oc < oc_end; oc++) {
        for (long p = 0; p < pixels; p++) {
            const float* src = input + p * c;
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++) sum += src[ic] * wgt[(long)ic * out_c + oc];
            output[p * out_c + oc] = vx_relu6_apply(sum, relu) + (addp ? addp[p * out_c + oc] : 0.0f);
        }
    }
#undef VX_PWG_NEON_RELU
}

static void vx_conv2d_pointwise_gemm_worker_neon(void* vctx, int begin, int end) {
    const VxPwGemmRangeContext* ctx = (const VxPwGemmRangeContext*)vctx;
    int oc_begin = begin * 16;
    int oc_end = end * 16;
    if (oc_end > ctx->out_c) oc_end = ctx->out_c;
    vx_conv2d_pointwise_gemm_range_neon(ctx, oc_begin, oc_end);
}

static void vx_conv2d_pointwise_gemm_neon(int node_idx,
                                          const float* input, float* output,
                                          const float* wgt, const float* bias,
                                          const float* addp,
                                          long pixels, int c, int out_c, int relu) {
    const float* pk = vx_pwf32_pack_plain_cache(node_idx, wgt, c, out_c);
    int oc16 = (pk) ? (out_c & ~15) : 0;
    VxPwGemmRangeContext ctx = {
        .input = input, .output = output, .wgt = wgt, .bias = bias,
        .addp = addp, .pk = pk, .pixels = pixels, .c = c,
        .out_c = out_c, .oc16 = oc16, .relu = relu,
    };
    const int threads = vx_kernels_thread_count();
    const int oc_blocks = (out_c + 15) / 16;
    const uint64_t total_flops = (uint64_t)pixels * c * out_c * 2u;
    if (threads > 1 && (total_flops >= 262144u || (total_flops >= 65536u && oc_blocks >= threads * 2)) && oc_blocks > 1) {
        int grain = (oc_blocks + threads - 1) / threads;
        if (grain < 1) grain = 1;
        vx_kernels_parallel_for(oc_blocks, grain, vx_conv2d_pointwise_gemm_worker_neon, &ctx);
    } else {
        vx_conv2d_pointwise_gemm_range_neon(&ctx, 0, out_c);
    }
}
#endif

void vx_conv2d_pointwise_f32(int node_idx,
                                  const float* input, float* output,
                                  const float* wgt, const float* bias,
                                  const float* addp,
                                  long pixels, int c, int out_c,
                                  int relu) {
#if defined(__AVX2__)
    if (out_c >= 16 && vx_pw_gemm_enabled()) {
        vx_conv2d_pointwise_gemm_f32(node_idx, input, output, wgt, bias, addp,
                                          pixels, c, out_c, relu);
        return;
    }
#elif defined(VX_CONV_USE_NEON)
    if (out_c >= 16 && vx_pw_gemm_enabled()) {
        vx_conv2d_pointwise_gemm_neon(node_idx, input, output, wgt, bias, addp,
                                      pixels, c, out_c, relu);
        return;
    }
#endif
    const float* pk = vx_pwf32_pack_cache(node_idx, wgt, c, out_c);
#if defined(__AVX2__)
    if (out_c >= 8) {
        const __m256 zero = _mm256_setzero_ps();
        const __m256 six = _mm256_set1_ps(6.0f);
#define VX_PWF32_RELU(v) do { if (relu) { (v) = _mm256_max_ps((v), zero); if (relu >= 2) (v) = _mm256_min_ps((v), six); } } while (0)
        long p = 0;
        int k4 = c / 4;
        int oc16_limit = (pk && k4 > 0) ? (out_c & ~15) : 0;
        for (; p + 4 <= pixels; p += 4) {
            const float* s0 = input + (p + 0) * c, * s1 = input + (p + 1) * c;
            const float* s2 = input + (p + 2) * c, * s3 = input + (p + 3) * c;
            float* d0 = output + (p + 0) * out_c, * d1 = output + (p + 1) * out_c;
            float* d2 = output + (p + 2) * out_c, * d3 = output + (p + 3) * out_c;
            int oc = 0;
            for (; oc < oc16_limit; oc += 16) {
                __m256 bl = bias ? _mm256_loadu_ps(bias + oc) : zero;
                __m256 bh = bias ? _mm256_loadu_ps(bias + oc + 8) : zero;
                __m256 a0l = bl, a0h = bh, a1l = bl, a1h = bh;
                __m256 a2l = bl, a2h = bh, a3l = bl, a3h = bh;
                const float* wp = pk + (long)(oc >> 4) * k4 * 64;
                for (int kg = 0; kg < k4; kg++) {
                    __m256 x0 = _mm256_broadcast_ps((const __m128*)(s0 + kg * 4));
                    __m256 x1 = _mm256_broadcast_ps((const __m128*)(s1 + kg * 4));
                    __m256 x2 = _mm256_broadcast_ps((const __m128*)(s2 + kg * 4));
                    __m256 x3 = _mm256_broadcast_ps((const __m128*)(s3 + kg * 4));
#define VX_PWF32_S4_STEP(OFF) do { \
    __m256 wl = _mm256_loadu_ps(wp + (OFF)); \
    __m256 wh = _mm256_loadu_ps(wp + (OFF) + 8); \
    a0l = _mm256_fmadd_ps(x0, wl, a0l); a0h = _mm256_fmadd_ps(x0, wh, a0h); \
    a1l = _mm256_fmadd_ps(x1, wl, a1l); a1h = _mm256_fmadd_ps(x1, wh, a1h); \
    a2l = _mm256_fmadd_ps(x2, wl, a2l); a2h = _mm256_fmadd_ps(x2, wh, a2h); \
    a3l = _mm256_fmadd_ps(x3, wl, a3l); a3h = _mm256_fmadd_ps(x3, wh, a3h); \
} while (0)
                    VX_PWF32_S4_STEP(0);
                    x0 = _mm256_permute_ps(x0, _MM_SHUFFLE(0, 3, 2, 1));
                    x1 = _mm256_permute_ps(x1, _MM_SHUFFLE(0, 3, 2, 1));
                    x2 = _mm256_permute_ps(x2, _MM_SHUFFLE(0, 3, 2, 1));
                    x3 = _mm256_permute_ps(x3, _MM_SHUFFLE(0, 3, 2, 1));
                    VX_PWF32_S4_STEP(16);
                    x0 = _mm256_permute_ps(x0, _MM_SHUFFLE(0, 3, 2, 1));
                    x1 = _mm256_permute_ps(x1, _MM_SHUFFLE(0, 3, 2, 1));
                    x2 = _mm256_permute_ps(x2, _MM_SHUFFLE(0, 3, 2, 1));
                    x3 = _mm256_permute_ps(x3, _MM_SHUFFLE(0, 3, 2, 1));
                    VX_PWF32_S4_STEP(32);
                    x0 = _mm256_permute_ps(x0, _MM_SHUFFLE(0, 3, 2, 1));
                    x1 = _mm256_permute_ps(x1, _MM_SHUFFLE(0, 3, 2, 1));
                    x2 = _mm256_permute_ps(x2, _MM_SHUFFLE(0, 3, 2, 1));
                    x3 = _mm256_permute_ps(x3, _MM_SHUFFLE(0, 3, 2, 1));
                    VX_PWF32_S4_STEP(48);
#undef VX_PWF32_S4_STEP
                    wp += 64;
                }
                for (int ic = k4 * 4; ic < c; ic++) {
                    __m256 wl = _mm256_loadu_ps(wgt + (long)ic * out_c + oc);
                    __m256 wh = _mm256_loadu_ps(wgt + (long)ic * out_c + oc + 8);
                    __m256 x;
                    x = _mm256_set1_ps(s0[ic]); a0l = _mm256_fmadd_ps(x, wl, a0l); a0h = _mm256_fmadd_ps(x, wh, a0h);
                    x = _mm256_set1_ps(s1[ic]); a1l = _mm256_fmadd_ps(x, wl, a1l); a1h = _mm256_fmadd_ps(x, wh, a1h);
                    x = _mm256_set1_ps(s2[ic]); a2l = _mm256_fmadd_ps(x, wl, a2l); a2h = _mm256_fmadd_ps(x, wh, a2h);
                    x = _mm256_set1_ps(s3[ic]); a3l = _mm256_fmadd_ps(x, wl, a3l); a3h = _mm256_fmadd_ps(x, wh, a3h);
                }
                VX_PWF32_RELU(a0l); VX_PWF32_RELU(a0h); VX_PWF32_RELU(a1l); VX_PWF32_RELU(a1h);
                VX_PWF32_RELU(a2l); VX_PWF32_RELU(a2h); VX_PWF32_RELU(a3l); VX_PWF32_RELU(a3h);
                if (addp) {
                    const float* r0 = addp + (p + 0) * out_c + oc, * r1 = addp + (p + 1) * out_c + oc;
                    const float* r2 = addp + (p + 2) * out_c + oc, * r3 = addp + (p + 3) * out_c + oc;
                    a0l = _mm256_add_ps(a0l, _mm256_loadu_ps(r0));      a0h = _mm256_add_ps(a0h, _mm256_loadu_ps(r0 + 8));
                    a1l = _mm256_add_ps(a1l, _mm256_loadu_ps(r1));      a1h = _mm256_add_ps(a1h, _mm256_loadu_ps(r1 + 8));
                    a2l = _mm256_add_ps(a2l, _mm256_loadu_ps(r2));      a2h = _mm256_add_ps(a2h, _mm256_loadu_ps(r2 + 8));
                    a3l = _mm256_add_ps(a3l, _mm256_loadu_ps(r3));      a3h = _mm256_add_ps(a3h, _mm256_loadu_ps(r3 + 8));
                }
                _mm256_storeu_ps(d0 + oc, a0l); _mm256_storeu_ps(d0 + oc + 8, a0h);
                _mm256_storeu_ps(d1 + oc, a1l); _mm256_storeu_ps(d1 + oc + 8, a1h);
                _mm256_storeu_ps(d2 + oc, a2l); _mm256_storeu_ps(d2 + oc + 8, a2h);
                _mm256_storeu_ps(d3 + oc, a3l); _mm256_storeu_ps(d3 + oc + 8, a3h);
            }
            for (; oc + 8 <= out_c; oc += 8) {
                __m256 b = bias ? _mm256_loadu_ps(bias + oc) : zero;
                __m256 a0 = b, a1 = b, a2 = b, a3 = b;
                for (int ic = 0; ic < c; ic++) {
                    __m256 wv = _mm256_loadu_ps(wgt + (long)ic * out_c + oc);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(s0[ic]), wv, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(s1[ic]), wv, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(s2[ic]), wv, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(s3[ic]), wv, a3);
                }
                VX_PWF32_RELU(a0); VX_PWF32_RELU(a1); VX_PWF32_RELU(a2); VX_PWF32_RELU(a3);
                if (addp) {
                    a0 = _mm256_add_ps(a0, _mm256_loadu_ps(addp + (p + 0) * out_c + oc));
                    a1 = _mm256_add_ps(a1, _mm256_loadu_ps(addp + (p + 1) * out_c + oc));
                    a2 = _mm256_add_ps(a2, _mm256_loadu_ps(addp + (p + 2) * out_c + oc));
                    a3 = _mm256_add_ps(a3, _mm256_loadu_ps(addp + (p + 3) * out_c + oc));
                }
                _mm256_storeu_ps(d0 + oc, a0); _mm256_storeu_ps(d1 + oc, a1);
                _mm256_storeu_ps(d2 + oc, a2); _mm256_storeu_ps(d3 + oc, a3);
            }
            for (; oc < out_c; oc++) {
                float c0 = bias ? bias[oc] : 0.0f, c1 = c0, c2 = c0, c3 = c0;
                for (int ic = 0; ic < c; ic++) {
                    float ww = wgt[(long)ic * out_c + oc];
                    c0 += s0[ic] * ww; c1 += s1[ic] * ww; c2 += s2[ic] * ww;
                    c3 += s3[ic] * ww;
                }
                d0[oc] = vx_relu6_apply(c0, relu) + (addp ? addp[(p + 0) * out_c + oc] : 0.0f);
                d1[oc] = vx_relu6_apply(c1, relu) + (addp ? addp[(p + 1) * out_c + oc] : 0.0f);
                d2[oc] = vx_relu6_apply(c2, relu) + (addp ? addp[(p + 2) * out_c + oc] : 0.0f);
                d3[oc] = vx_relu6_apply(c3, relu) + (addp ? addp[(p + 3) * out_c + oc] : 0.0f);
            }
        }
        for (; p < pixels; p++) {
            const float* src = input + p * c;
            float* dst = output + p * out_c;
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256 acc = bias ? _mm256_loadu_ps(bias + oc) : _mm256_setzero_ps();
                for (int ic = 0; ic < c; ic++) {
                    __m256 xv = _mm256_set1_ps(src[ic]);
                    __m256 wv = _mm256_loadu_ps(wgt + (long)ic * out_c + oc);
                    acc = _mm256_fmadd_ps(xv, wv, acc);
                }
                VX_PWF32_RELU(acc);
                if (addp) acc = _mm256_add_ps(acc, _mm256_loadu_ps(addp + p * out_c + oc));
                _mm256_storeu_ps(dst + oc, acc);
            }
            for (; oc < out_c; oc++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < c; ic++) sum += src[ic] * wgt[(long)ic * out_c + oc];
                dst[oc] = vx_relu6_apply(sum, relu) + (addp ? addp[p * out_c + oc] : 0.0f);
            }
        }
#undef VX_PWF32_RELU
        return;
    }
#endif
#if defined(VX_CONV_USE_WASM_SIMD128)
    const v128_t vzero = wasm_f32x4_splat(0.0f);
    const v128_t vsix = wasm_f32x4_splat(6.0f);
#define VX_WASM_PWF32_RELU(v) do { \
        if (relu) { \
            (v) = wasm_f32x4_max((v), vzero); \
            if (relu >= 2) (v) = wasm_f32x4_min((v), vsix); \
        } \
    } while (0)

    const float* pk_plain = vx_pwf32_pack_plain_cache(node_idx, wgt, c, out_c);
    int oc16_limit = (pk_plain && out_c >= 16) ? (out_c & ~15) : 0;
    int oc = 0;
    for (; oc < oc16_limit; oc += 16) {
        const float* wp_base = pk_plain + (long)(oc >> 4) * c * 16;
        const v128_t b0 = bias ? wasm_v128_load(bias + oc + 0) : vzero;
        const v128_t b1 = bias ? wasm_v128_load(bias + oc + 4) : vzero;
        const v128_t b2 = bias ? wasm_v128_load(bias + oc + 8) : vzero;
        const v128_t b3 = bias ? wasm_v128_load(bias + oc + 12) : vzero;

        long p = 0;
        /* 3-pixel unrolled loop: 12 accumulators, 3 FMAs per weight load */
        for (; p + 3 <= pixels; p += 3) {
            const float* s0 = input + (p + 0) * c;
            const float* s1 = input + (p + 1) * c;
            const float* s2 = input + (p + 2) * c;
            float* d0 = output + (p + 0) * out_c;
            float* d1 = output + (p + 1) * out_c;
            float* d2 = output + (p + 2) * out_c;

            v128_t a0_0 = b0, a0_1 = b1, a0_2 = b2, a0_3 = b3;
            v128_t a1_0 = b0, a1_1 = b1, a1_2 = b2, a1_3 = b3;
            v128_t a2_0 = b0, a2_1 = b1, a2_2 = b2, a2_3 = b3;

            const float* wp = wp_base;
            int ic = 0;
            for (; ic + 2 <= c; ic += 2, wp += 32) {
                const v128_t x0_0 = wasm_v128_load32_splat(s0 + ic + 0);
                const v128_t x1_0 = wasm_v128_load32_splat(s1 + ic + 0);
                const v128_t x2_0 = wasm_v128_load32_splat(s2 + ic + 0);

                const v128_t w0 = wasm_v128_load(wp + 0);
                a0_0 = wasm_f32x4_add(a0_0, wasm_f32x4_mul(x0_0, w0));
                a1_0 = wasm_f32x4_add(a1_0, wasm_f32x4_mul(x1_0, w0));
                a2_0 = wasm_f32x4_add(a2_0, wasm_f32x4_mul(x2_0, w0));

                const v128_t w1 = wasm_v128_load(wp + 4);
                a0_1 = wasm_f32x4_add(a0_1, wasm_f32x4_mul(x0_0, w1));
                a1_1 = wasm_f32x4_add(a1_1, wasm_f32x4_mul(x1_0, w1));
                a2_1 = wasm_f32x4_add(a2_1, wasm_f32x4_mul(x2_0, w1));

                const v128_t w2 = wasm_v128_load(wp + 8);
                a0_2 = wasm_f32x4_add(a0_2, wasm_f32x4_mul(x0_0, w2));
                a1_2 = wasm_f32x4_add(a1_2, wasm_f32x4_mul(x1_0, w2));
                a2_2 = wasm_f32x4_add(a2_2, wasm_f32x4_mul(x2_0, w2));

                const v128_t w3 = wasm_v128_load(wp + 12);
                a0_3 = wasm_f32x4_add(a0_3, wasm_f32x4_mul(x0_0, w3));
                a1_3 = wasm_f32x4_add(a1_3, wasm_f32x4_mul(x1_0, w3));
                a2_3 = wasm_f32x4_add(a2_3, wasm_f32x4_mul(x2_0, w3));

                const v128_t x0_1 = wasm_v128_load32_splat(s0 + ic + 1);
                const v128_t x1_1 = wasm_v128_load32_splat(s1 + ic + 1);
                const v128_t x2_1 = wasm_v128_load32_splat(s2 + ic + 1);

                const v128_t w4 = wasm_v128_load(wp + 16);
                a0_0 = wasm_f32x4_add(a0_0, wasm_f32x4_mul(x0_1, w4));
                a1_0 = wasm_f32x4_add(a1_0, wasm_f32x4_mul(x1_1, w4));
                a2_0 = wasm_f32x4_add(a2_0, wasm_f32x4_mul(x2_1, w4));

                const v128_t w5 = wasm_v128_load(wp + 20);
                a0_1 = wasm_f32x4_add(a0_1, wasm_f32x4_mul(x0_1, w5));
                a1_1 = wasm_f32x4_add(a1_1, wasm_f32x4_mul(x1_1, w5));
                a2_1 = wasm_f32x4_add(a2_1, wasm_f32x4_mul(x2_1, w5));

                const v128_t w6 = wasm_v128_load(wp + 24);
                a0_2 = wasm_f32x4_add(a0_2, wasm_f32x4_mul(x0_1, w6));
                a1_2 = wasm_f32x4_add(a1_2, wasm_f32x4_mul(x1_1, w6));
                a2_2 = wasm_f32x4_add(a2_2, wasm_f32x4_mul(x2_1, w6));

                const v128_t w7 = wasm_v128_load(wp + 28);
                a0_3 = wasm_f32x4_add(a0_3, wasm_f32x4_mul(x0_1, w7));
                a1_3 = wasm_f32x4_add(a1_3, wasm_f32x4_mul(x1_1, w7));
                a2_3 = wasm_f32x4_add(a2_3, wasm_f32x4_mul(x2_1, w7));
            }
            for (; ic < c; ic++, wp += 16) {
                const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                const v128_t x1 = wasm_v128_load32_splat(s1 + ic);
                const v128_t x2 = wasm_v128_load32_splat(s2 + ic);

                const v128_t w0 = wasm_v128_load(wp + 0);
                a0_0 = wasm_f32x4_add(a0_0, wasm_f32x4_mul(x0, w0));
                a1_0 = wasm_f32x4_add(a1_0, wasm_f32x4_mul(x1, w0));
                a2_0 = wasm_f32x4_add(a2_0, wasm_f32x4_mul(x2, w0));

                const v128_t w1 = wasm_v128_load(wp + 4);
                a0_1 = wasm_f32x4_add(a0_1, wasm_f32x4_mul(x0, w1));
                a1_1 = wasm_f32x4_add(a1_1, wasm_f32x4_mul(x1, w1));
                a2_1 = wasm_f32x4_add(a2_1, wasm_f32x4_mul(x2, w1));

                const v128_t w2 = wasm_v128_load(wp + 8);
                a0_2 = wasm_f32x4_add(a0_2, wasm_f32x4_mul(x0, w2));
                a1_2 = wasm_f32x4_add(a1_2, wasm_f32x4_mul(x1, w2));
                a2_2 = wasm_f32x4_add(a2_2, wasm_f32x4_mul(x2, w2));

                const v128_t w3 = wasm_v128_load(wp + 12);
                a0_3 = wasm_f32x4_add(a0_3, wasm_f32x4_mul(x0, w3));
                a1_3 = wasm_f32x4_add(a1_3, wasm_f32x4_mul(x1, w3));
                a2_3 = wasm_f32x4_add(a2_3, wasm_f32x4_mul(x2, w3));
            }

            VX_WASM_PWF32_RELU(a0_0); VX_WASM_PWF32_RELU(a0_1); VX_WASM_PWF32_RELU(a0_2); VX_WASM_PWF32_RELU(a0_3);
            VX_WASM_PWF32_RELU(a1_0); VX_WASM_PWF32_RELU(a1_1); VX_WASM_PWF32_RELU(a1_2); VX_WASM_PWF32_RELU(a1_3);
            VX_WASM_PWF32_RELU(a2_0); VX_WASM_PWF32_RELU(a2_1); VX_WASM_PWF32_RELU(a2_2); VX_WASM_PWF32_RELU(a2_3);

            if (addp) {
                a0_0 = wasm_f32x4_add(a0_0, wasm_v128_load(addp + (p + 0) * out_c + oc + 0));
                a0_1 = wasm_f32x4_add(a0_1, wasm_v128_load(addp + (p + 0) * out_c + oc + 4));
                a0_2 = wasm_f32x4_add(a0_2, wasm_v128_load(addp + (p + 0) * out_c + oc + 8));
                a0_3 = wasm_f32x4_add(a0_3, wasm_v128_load(addp + (p + 0) * out_c + oc + 12));

                a1_0 = wasm_f32x4_add(a1_0, wasm_v128_load(addp + (p + 1) * out_c + oc + 0));
                a1_1 = wasm_f32x4_add(a1_1, wasm_v128_load(addp + (p + 1) * out_c + oc + 4));
                a1_2 = wasm_f32x4_add(a1_2, wasm_v128_load(addp + (p + 1) * out_c + oc + 8));
                a1_3 = wasm_f32x4_add(a1_3, wasm_v128_load(addp + (p + 1) * out_c + oc + 12));

                a2_0 = wasm_f32x4_add(a2_0, wasm_v128_load(addp + (p + 2) * out_c + oc + 0));
                a2_1 = wasm_f32x4_add(a2_1, wasm_v128_load(addp + (p + 2) * out_c + oc + 4));
                a2_2 = wasm_f32x4_add(a2_2, wasm_v128_load(addp + (p + 2) * out_c + oc + 8));
                a2_3 = wasm_f32x4_add(a2_3, wasm_v128_load(addp + (p + 2) * out_c + oc + 12));
            }

            wasm_v128_store(d0 + oc + 0, a0_0);
            wasm_v128_store(d0 + oc + 4, a0_1);
            wasm_v128_store(d0 + oc + 8, a0_2);
            wasm_v128_store(d0 + oc + 12, a0_3);

            wasm_v128_store(d1 + oc + 0, a1_0);
            wasm_v128_store(d1 + oc + 4, a1_1);
            wasm_v128_store(d1 + oc + 8, a1_2);
            wasm_v128_store(d1 + oc + 12, a1_3);

            wasm_v128_store(d2 + oc + 0, a2_0);
            wasm_v128_store(d2 + oc + 4, a2_1);
            wasm_v128_store(d2 + oc + 8, a2_2);
            wasm_v128_store(d2 + oc + 12, a2_3);
        }

        /* 2-pixel remainder */
        for (; p + 2 <= pixels; p += 2) {
            const float* s0 = input + (p + 0) * c;
            const float* s1 = input + (p + 1) * c;
            float* d0 = output + (p + 0) * out_c;
            float* d1 = output + (p + 1) * out_c;

            v128_t a0_0 = b0, a0_1 = b1, a0_2 = b2, a0_3 = b3;
            v128_t a1_0 = b0, a1_1 = b1, a1_2 = b2, a1_3 = b3;

            const float* wp = wp_base;
            for (int ic = 0; ic < c; ic++, wp += 16) {
                const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                const v128_t x1 = wasm_v128_load32_splat(s1 + ic);

                const v128_t w0 = wasm_v128_load(wp + 0);
                a0_0 = wasm_f32x4_add(a0_0, wasm_f32x4_mul(x0, w0));
                a1_0 = wasm_f32x4_add(a1_0, wasm_f32x4_mul(x1, w0));

                const v128_t w1 = wasm_v128_load(wp + 4);
                a0_1 = wasm_f32x4_add(a0_1, wasm_f32x4_mul(x0, w1));
                a1_1 = wasm_f32x4_add(a1_1, wasm_f32x4_mul(x1, w1));

                const v128_t w2 = wasm_v128_load(wp + 8);
                a0_2 = wasm_f32x4_add(a0_2, wasm_f32x4_mul(x0, w2));
                a1_2 = wasm_f32x4_add(a1_2, wasm_f32x4_mul(x1, w2));

                const v128_t w3 = wasm_v128_load(wp + 12);
                a0_3 = wasm_f32x4_add(a0_3, wasm_f32x4_mul(x0, w3));
                a1_3 = wasm_f32x4_add(a1_3, wasm_f32x4_mul(x1, w3));
            }

            VX_WASM_PWF32_RELU(a0_0); VX_WASM_PWF32_RELU(a0_1); VX_WASM_PWF32_RELU(a0_2); VX_WASM_PWF32_RELU(a0_3);
            VX_WASM_PWF32_RELU(a1_0); VX_WASM_PWF32_RELU(a1_1); VX_WASM_PWF32_RELU(a1_2); VX_WASM_PWF32_RELU(a1_3);

            if (addp) {
                a0_0 = wasm_f32x4_add(a0_0, wasm_v128_load(addp + (p + 0) * out_c + oc + 0));
                a0_1 = wasm_f32x4_add(a0_1, wasm_v128_load(addp + (p + 0) * out_c + oc + 4));
                a0_2 = wasm_f32x4_add(a0_2, wasm_v128_load(addp + (p + 0) * out_c + oc + 8));
                a0_3 = wasm_f32x4_add(a0_3, wasm_v128_load(addp + (p + 0) * out_c + oc + 12));

                a1_0 = wasm_f32x4_add(a1_0, wasm_v128_load(addp + (p + 1) * out_c + oc + 0));
                a1_1 = wasm_f32x4_add(a1_1, wasm_v128_load(addp + (p + 1) * out_c + oc + 4));
                a1_2 = wasm_f32x4_add(a1_2, wasm_v128_load(addp + (p + 1) * out_c + oc + 8));
                a1_3 = wasm_f32x4_add(a1_3, wasm_v128_load(addp + (p + 1) * out_c + oc + 12));
            }

            wasm_v128_store(d0 + oc + 0, a0_0);
            wasm_v128_store(d0 + oc + 4, a0_1);
            wasm_v128_store(d0 + oc + 8, a0_2);
            wasm_v128_store(d0 + oc + 12, a0_3);

            wasm_v128_store(d1 + oc + 0, a1_0);
            wasm_v128_store(d1 + oc + 4, a1_1);
            wasm_v128_store(d1 + oc + 8, a1_2);
            wasm_v128_store(d1 + oc + 12, a1_3);
        }

        /* 1-pixel remainder */
        for (; p < pixels; p++) {
            const float* s0 = input + p * c;
            float* d0 = output + p * out_c;

            v128_t a0_0 = b0, a0_1 = b1, a0_2 = b2, a0_3 = b3;

            const float* wp = wp_base;
            for (int ic = 0; ic < c; ic++, wp += 16) {
                const v128_t x0 = wasm_v128_load32_splat(s0 + ic);

                a0_0 = wasm_f32x4_add(a0_0, wasm_f32x4_mul(x0, wasm_v128_load(wp + 0)));
                a0_1 = wasm_f32x4_add(a0_1, wasm_f32x4_mul(x0, wasm_v128_load(wp + 4)));
                a0_2 = wasm_f32x4_add(a0_2, wasm_f32x4_mul(x0, wasm_v128_load(wp + 8)));
                a0_3 = wasm_f32x4_add(a0_3, wasm_f32x4_mul(x0, wasm_v128_load(wp + 12)));
            }

            VX_WASM_PWF32_RELU(a0_0); VX_WASM_PWF32_RELU(a0_1); VX_WASM_PWF32_RELU(a0_2); VX_WASM_PWF32_RELU(a0_3);

            if (addp) {
                a0_0 = wasm_f32x4_add(a0_0, wasm_v128_load(addp + p * out_c + oc + 0));
                a0_1 = wasm_f32x4_add(a0_1, wasm_v128_load(addp + p * out_c + oc + 4));
                a0_2 = wasm_f32x4_add(a0_2, wasm_v128_load(addp + p * out_c + oc + 8));
                a0_3 = wasm_f32x4_add(a0_3, wasm_v128_load(addp + p * out_c + oc + 12));
            }

            wasm_v128_store(d0 + oc + 0, a0_0);
            wasm_v128_store(d0 + oc + 4, a0_1);
            wasm_v128_store(d0 + oc + 8, a0_2);
            wasm_v128_store(d0 + oc + 12, a0_3);
        }
    }

    /* Fallback / remainders for oc */
    for (; oc + 4 <= out_c; oc += 4) {
        const v128_t b = bias ? wasm_v128_load(bias + oc) : vzero;
        long p = 0;
        for (; p + 2 <= pixels; p += 2) {
            const float* s0 = input + (p + 0) * c;
            const float* s1 = input + (p + 1) * c;
            float* d0 = output + (p + 0) * out_c;
            float* d1 = output + (p + 1) * out_c;
            v128_t a0 = b, a1 = b;
            for (int ic = 0; ic < c; ic++) {
                const v128_t wv = wasm_v128_load(wgt + (long)ic * out_c + oc);
                a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load32_splat(s0 + ic), wv));
                a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wasm_v128_load32_splat(s1 + ic), wv));
            }
            VX_WASM_PWF32_RELU(a0); VX_WASM_PWF32_RELU(a1);
            if (addp) {
                a0 = wasm_f32x4_add(a0, wasm_v128_load(addp + (p + 0) * out_c + oc));
                a1 = wasm_f32x4_add(a1, wasm_v128_load(addp + (p + 1) * out_c + oc));
            }
            wasm_v128_store(d0 + oc, a0);
            wasm_v128_store(d1 + oc, a1);
        }
        for (; p < pixels; p++) {
            const float* s0 = input + p * c;
            float* d0 = output + p * out_c;
            v128_t a0 = b;
            for (int ic = 0; ic < c; ic++) {
                const v128_t wv = wasm_v128_load(wgt + (long)ic * out_c + oc);
                a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load32_splat(s0 + ic), wv));
            }
            VX_WASM_PWF32_RELU(a0);
            if (addp) a0 = wasm_f32x4_add(a0, wasm_v128_load(addp + p * out_c + oc));
            wasm_v128_store(d0 + oc, a0);
        }
    }
    for (; oc < out_c; oc++) {
        for (long p = 0; p < pixels; p++) {
            const float* s0 = input + p * c;
            float c0 = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++) c0 += s0[ic] * wgt[(long)ic * out_c + oc];
            output[p * out_c + oc] = vx_relu6_apply(c0, relu) + (addp ? addp[p * out_c + oc] : 0.0f);
        }
    }
#undef VX_WASM_PWF32_RELU
    return;
#endif
    for (long p = 0; p < pixels; p++) {
        const float* src = input + p * c;
        float* dst = output + p * out_c;
        for (int oc = 0; oc < out_c; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < c; ic++) sum += src[ic] * wgt[(long)ic * out_c + oc];
            dst[oc] = vx_relu6_apply(sum, relu) + (addp ? addp[p * out_c + oc] : 0.0f);
        }
    }
}

static float* vx_dw_pw_tmp_cache(int node_idx, long elems) {
    if (node_idx < 0 || (size_t)node_idx >= vx_engine_state_current()->node_capacity ||
        elems <= 0) return NULL;
    if (g_dw_pw_tmp[node_idx] && g_dw_pw_tmp_cap[node_idx] >= elems) return g_dw_pw_tmp[node_idx];
    float* p = (float*)realloc(g_dw_pw_tmp[node_idx], (size_t)elems * sizeof(float));
    if (!p) return NULL;
    g_dw_pw_tmp[node_idx] = p;
    g_dw_pw_tmp_cap[node_idx] = elems;
    return p;
}

int vx_conv2d_depthwise_pointwise_f32(int dw_node_idx, int pw_node_idx,
                                           const float* input, float* output,
                                           const float* dw_wgt, const float* dw_bias,
                                           const float* pw_wgt, const float* pw_bias,
                                           const float* add,
                                           int n, int h, int w, int c,
                                           int oh, int ow, int out_c,
                                           int kh, int kw, int wc, int mult,
                                           int pw_kh, int pw_kw, int pw_in_c,
                                           int sy, int sx, const int* pads,
                                           int dw_relu, int pw_relu) {
    if (!input || !output || !dw_wgt || !pw_wgt || !pads) return 0;
    if (wc != c || mult != 1 || pw_kh != 1 || pw_kw != 1 || pw_in_c != c || add == output) return 0;

    int block = c > 0 ? 49152 / c : 1;
    if (block < 1) block = 1;
    if (block > 60) block = 60;
    if (block >= 6) block = (block / 6) * 6;
    long tmp_elems = (long)block * c;
    float* tmp = vx_dw_pw_tmp_cache(dw_node_idx, tmp_elems);
    if (!tmp) return 0;

#if defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#endif
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* out_b = output + (long)b * oh * ow * out_c;
        const float* add_b = add ? add + (long)b * oh * ow * out_c : NULL;
        long pixels = (long)oh * ow;
        for (long p0 = 0; p0 < pixels; p0 += block) {
            int bc = (int)((p0 + block <= pixels) ? block : (pixels - p0));
            for (int bp = 0; bp < bc; bp++) {
                long p = p0 + bp;
                int oy = (int)(p / ow);
                int ox = (int)(p - (long)oy * ow);
                int iy0 = oy * sy - pads[0];
                int ix0 = ox * sx - pads[1];
                float* dst = tmp + (long)bp * c;
                int ch = 0;
#if defined(__AVX2__)
                for (; ch + 8 <= c; ch += 8) {
                    __m256 acc = dw_bias ? _mm256_loadu_ps(dw_bias + ch) : zero;
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(src_b + ((long)iy * w + ix) * c + ch),
                                                  _mm256_loadu_ps(dw_wgt + ((long)ky * kw + kx) * c + ch), acc);
                        }
                    }
                    if (dw_relu) {
                        acc = _mm256_max_ps(acc, zero);
                        if (dw_relu >= 2) acc = _mm256_min_ps(acc, six);
                    }
                    _mm256_storeu_ps(dst + ch, acc);
                }
#endif
                for (; ch < c; ch++) {
                    float sum = dw_bias ? dw_bias[ch] : 0.0f;
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            sum += src_b[((long)iy * w + ix) * c + ch] *
                                   dw_wgt[((long)ky * kw + kx) * c + ch];
                        }
                    }
                    dst[ch] = vx_relu6_apply(sum, dw_relu);
                }
            }
            vx_conv2d_pointwise_f32(pw_node_idx, tmp, out_b + p0 * out_c, pw_wgt,
                                         pw_bias, add_b ? add_b + p0 * out_c : NULL,
                                         bc, c, out_c, pw_relu);
        }
    }
    return 1;
}

void vx_conv2d_dw3x3s1_f32(const float* input, float* output,
                                const float* wgt, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow,
                                const int* pads, int relu) {
    int pt = pads[0], pl = pads[1];
#if defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#endif
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy - pt;
            int row_in = (iy0 >= 0 && iy0 + 2 < h);
            int ox = 0;
            while (ox < ow) {
                int ix0 = ox - pl;
#if defined(__AVX2__)
                if (row_in && ix0 >= 0 && ix0 + 5 < w && ox + 4 <= ow) {
                    float* dst = dst_b + ((long)oy * ow + ox) * c;
                    int ch = 0;
                    for (; ch + 8 <= c; ch += 8) {
                        __m256 a0 = bias ? _mm256_loadu_ps(bias + ch) : zero;
                        __m256 a1 = a0, a2 = a0, a3 = a0;
                        for (int ky = 0; ky < 3; ky++) {
                            const float* row = src_b + ((long)(iy0 + ky) * w + ix0) * c + ch;
                            __m256 i0 = _mm256_loadu_ps(row), i1 = _mm256_loadu_ps(row + c);
                            __m256 i2 = _mm256_loadu_ps(row + 2 * c), i3 = _mm256_loadu_ps(row + 3 * c);
                            __m256 i4 = _mm256_loadu_ps(row + 4 * c), i5 = _mm256_loadu_ps(row + 5 * c);
                            const float* wr = wgt + ((long)ky * 3) * c + ch;
                            __m256 w0 = _mm256_loadu_ps(wr), w1 = _mm256_loadu_ps(wr + c), w2 = _mm256_loadu_ps(wr + 2 * c);
                            a0 = _mm256_fmadd_ps(i0, w0, a0); a0 = _mm256_fmadd_ps(i1, w1, a0); a0 = _mm256_fmadd_ps(i2, w2, a0);
                            a1 = _mm256_fmadd_ps(i1, w0, a1); a1 = _mm256_fmadd_ps(i2, w1, a1); a1 = _mm256_fmadd_ps(i3, w2, a1);
                            a2 = _mm256_fmadd_ps(i2, w0, a2); a2 = _mm256_fmadd_ps(i3, w1, a2); a2 = _mm256_fmadd_ps(i4, w2, a2);
                            a3 = _mm256_fmadd_ps(i3, w0, a3); a3 = _mm256_fmadd_ps(i4, w1, a3); a3 = _mm256_fmadd_ps(i5, w2, a3);
                        }
                        if (relu) {
                            a0 = _mm256_max_ps(a0, zero); a1 = _mm256_max_ps(a1, zero);
                            a2 = _mm256_max_ps(a2, zero); a3 = _mm256_max_ps(a3, zero);
                            if (relu >= 2) {
                                a0 = _mm256_min_ps(a0, six); a1 = _mm256_min_ps(a1, six);
                                a2 = _mm256_min_ps(a2, six); a3 = _mm256_min_ps(a3, six);
                            }
                        }
                        _mm256_storeu_ps(dst + 0 * c + ch, a0);
                        _mm256_storeu_ps(dst + 1 * c + ch, a1);
                        _mm256_storeu_ps(dst + 2 * c + ch, a2);
                        _mm256_storeu_ps(dst + 3 * c + ch, a3);
                    }
                    for (; ch < c; ch++) {
                        for (int j = 0; j < 4; j++) {
                            float sum = bias ? bias[ch] : 0.0f;
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                    sum += src_b[((long)(iy0 + ky) * w + (ix0 + j + kx)) * c + ch] *
                                           wgt[((long)ky * 3 + kx) * c + ch];
                            dst[(long)j * c + ch] = vx_relu6_apply(sum, relu);
                        }
                    }
                    ox += 4;
                    continue;
                }
#endif
                float* dst = dst_b + ((long)oy * ow + ox) * c;
                int ch = 0;
#if defined(__AVX2__)
                for (; ch + 8 <= c; ch += 8) {
                    __m256 acc = bias ? _mm256_loadu_ps(bias + ch) : zero;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(src_b + ((long)iy * w + ix) * c + ch),
                                                  _mm256_loadu_ps(wgt + ((long)ky * 3 + kx) * c + ch), acc);
                        }
                    }
                    if (relu) { acc = _mm256_max_ps(acc, zero); if (relu >= 2) acc = _mm256_min_ps(acc, six); }
                    _mm256_storeu_ps(dst + ch, acc);
                }
#endif
                for (; ch < c; ch++) {
                    float sum = bias ? bias[ch] : 0.0f;
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            sum += src_b[((long)iy * w + ix) * c + ch] * wgt[((long)ky * 3 + kx) * c + ch];
                        }
                    }
                    dst[ch] = vx_relu6_apply(sum, relu);
                }
                ox += 1;
            }
        }
    }
}

void vx_conv2d_dw5x5s1_f32(const float* input, float* output,
                                const float* wgt, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow,
                                const int* pads, int relu) {
    int pt = pads[0], pl = pads[1];
#if defined(__AVX2__)
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#endif
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy - pt;
            int row_in = (iy0 >= 0 && iy0 + 4 < h);
            int ox = 0;
            while (ox < ow) {
                int ix0 = ox - pl;
#if defined(__AVX2__)
                if (row_in && ix0 >= 0 && ix0 + 7 < w && ox + 4 <= ow) {
                    float* dst = dst_b + ((long)oy * ow + ox) * c;
                    int ch = 0;
                    for (; ch + 8 <= c; ch += 8) {
                        __m256 a0 = bias ? _mm256_loadu_ps(bias + ch) : zero;
                        __m256 a1 = a0, a2 = a0, a3 = a0;
                        for (int ky = 0; ky < 5; ky++) {
                            const float* row = src_b + ((long)(iy0 + ky) * w + ix0) * c + ch;
                            __m256 i0 = _mm256_loadu_ps(row);
                            __m256 i1 = _mm256_loadu_ps(row + c);
                            __m256 i2 = _mm256_loadu_ps(row + 2 * c);
                            __m256 i3 = _mm256_loadu_ps(row + 3 * c);
                            __m256 i4 = _mm256_loadu_ps(row + 4 * c);
                            __m256 i5 = _mm256_loadu_ps(row + 5 * c);
                            __m256 i6 = _mm256_loadu_ps(row + 6 * c);
                            __m256 i7 = _mm256_loadu_ps(row + 7 * c);
                            const float* wr = wgt + ((long)ky * 5) * c + ch;
                            __m256 w0 = _mm256_loadu_ps(wr);
                            __m256 w1 = _mm256_loadu_ps(wr + c);
                            __m256 w2 = _mm256_loadu_ps(wr + 2 * c);
                            __m256 w3 = _mm256_loadu_ps(wr + 3 * c);
                            __m256 w4 = _mm256_loadu_ps(wr + 4 * c);
                            a0 = _mm256_fmadd_ps(i0, w0, a0); a0 = _mm256_fmadd_ps(i1, w1, a0);
                            a0 = _mm256_fmadd_ps(i2, w2, a0); a0 = _mm256_fmadd_ps(i3, w3, a0);
                            a0 = _mm256_fmadd_ps(i4, w4, a0);
                            a1 = _mm256_fmadd_ps(i1, w0, a1); a1 = _mm256_fmadd_ps(i2, w1, a1);
                            a1 = _mm256_fmadd_ps(i3, w2, a1); a1 = _mm256_fmadd_ps(i4, w3, a1);
                            a1 = _mm256_fmadd_ps(i5, w4, a1);
                            a2 = _mm256_fmadd_ps(i2, w0, a2); a2 = _mm256_fmadd_ps(i3, w1, a2);
                            a2 = _mm256_fmadd_ps(i4, w2, a2); a2 = _mm256_fmadd_ps(i5, w3, a2);
                            a2 = _mm256_fmadd_ps(i6, w4, a2);
                            a3 = _mm256_fmadd_ps(i3, w0, a3); a3 = _mm256_fmadd_ps(i4, w1, a3);
                            a3 = _mm256_fmadd_ps(i5, w2, a3); a3 = _mm256_fmadd_ps(i6, w3, a3);
                            a3 = _mm256_fmadd_ps(i7, w4, a3);
                        }
                        if (relu) {
                            a0 = _mm256_max_ps(a0, zero); a1 = _mm256_max_ps(a1, zero);
                            a2 = _mm256_max_ps(a2, zero); a3 = _mm256_max_ps(a3, zero);
                            if (relu >= 2) {
                                a0 = _mm256_min_ps(a0, six); a1 = _mm256_min_ps(a1, six);
                                a2 = _mm256_min_ps(a2, six); a3 = _mm256_min_ps(a3, six);
                            }
                        }
                        _mm256_storeu_ps(dst + 0 * c + ch, a0);
                        _mm256_storeu_ps(dst + 1 * c + ch, a1);
                        _mm256_storeu_ps(dst + 2 * c + ch, a2);
                        _mm256_storeu_ps(dst + 3 * c + ch, a3);
                    }
                    for (; ch < c; ch++) {
                        for (int j = 0; j < 4; j++) {
                            float sum = bias ? bias[ch] : 0.0f;
                            for (int ky = 0; ky < 5; ky++)
                                for (int kx = 0; kx < 5; kx++)
                                    sum += src_b[((long)(iy0 + ky) * w + (ix0 + j + kx)) * c + ch] *
                                           wgt[((long)ky * 5 + kx) * c + ch];
                            dst[(long)j * c + ch] = vx_relu6_apply(sum, relu);
                        }
                    }
                    ox += 4;
                    continue;
                }
#endif
                float* dst = dst_b + ((long)oy * ow + ox) * c;
                int ch = 0;
#if defined(__AVX2__)
                for (; ch + 8 <= c; ch += 8) {
                    __m256 acc = bias ? _mm256_loadu_ps(bias + ch) : zero;
                    for (int ky = 0; ky < 5; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < 5; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(src_b + ((long)iy * w + ix) * c + ch),
                                                  _mm256_loadu_ps(wgt + ((long)ky * 5 + kx) * c + ch), acc);
                        }
                    }
                    if (relu) {
                        acc = _mm256_max_ps(acc, zero);
                        if (relu >= 2) acc = _mm256_min_ps(acc, six);
                    }
                    _mm256_storeu_ps(dst + ch, acc);
                }
#endif
                for (; ch < c; ch++) {
                    float sum = bias ? bias[ch] : 0.0f;
                    for (int ky = 0; ky < 5; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < 5; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            sum += src_b[((long)iy * w + ix) * c + ch] *
                                   wgt[((long)ky * 5 + kx) * c + ch];
                        }
                    }
                    dst[ch] = vx_relu6_apply(sum, relu);
                }
                ox += 1;
            }
        }
    }
}

void vx_conv2d_depthwise_f32(const float* input, float* output,
                                  const float* wgt, const float* bias,
                                  int n, int h, int w, int c,
                                  int oh, int ow, int out_c,
                                  int kh, int kw, int wc, int mult,
                                  int sy, int sx, const int* pads,
                                  int dy, int dx, int relu) {
    if (kh == 3 && kw == 3 && mult == 1 && sy == 1 && sx == 1 && dy == 1 && dx == 1) {
        vx_conv2d_dw3x3s1_f32(input, output, wgt, bias, n, h, w, c, oh, ow, pads, relu);
        return;
    }
    if (kh == 5 && kw == 5 && mult == 1 && sy == 1 && sx == 1 && dy == 1 && dx == 1) {
        vx_conv2d_dw5x5s1_f32(input, output, wgt, bias, n, h, w, c, oh, ow, pads, relu);
        return;
    }
    if (wc != c || out_c != c * mult) return;

    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * out_c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * sx - pads[1];
                float* dst = dst_b + ((long)oy * ow + ox) * out_c;
                if (mult == 1) {
                    int ch = 0;
#if defined(__AVX2__)
                    const __m256 zero = _mm256_setzero_ps();
                    const __m256 six = _mm256_set1_ps(6.0f);
                    int interior1x3 = (kh == 1 && kw == 3 &&
                                       iy0 >= 0 && iy0 < h &&
                                       ix0 >= 0 && ix0 + 2 * dx < w);
                    int interior3 = (kh == 3 && kw == 3 &&
                                     iy0 >= 0 && iy0 + 2 * dy < h &&
                                     ix0 >= 0 && ix0 + 2 * dx < w);
                    if (interior1x3) {
                        long cstep = (long)dx * c;
                        const float* base = src_b + ((long)iy0 * w + ix0) * c;
                        const float* w0 = wgt;
                        const float* w1 = wgt + c;
                        const float* w2 = wgt + (long)2 * c;
                        for (; ch + 8 <= c; ch += 8) {
                            __m256 acc = bias ? _mm256_loadu_ps(bias + ch) : zero;
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(base + ch), _mm256_loadu_ps(w0 + ch), acc);
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(base + cstep + ch), _mm256_loadu_ps(w1 + ch), acc);
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(base + 2 * cstep + ch), _mm256_loadu_ps(w2 + ch), acc);
                            if (relu) {
                                acc = _mm256_max_ps(acc, zero);
                                if (relu >= 2) acc = _mm256_min_ps(acc, six);
                            }
                            _mm256_storeu_ps(dst + ch, acc);
                        }
                    } else if (interior3) {
                        long rstep = (long)dy * w * c, cstep = (long)dx * c;
                        const float* base = src_b + ((long)iy0 * w + ix0) * c;
                        for (; ch + 8 <= c; ch += 8) {
                            const float* s = base + ch;
                            const float* ww = wgt + ch;
                            __m256 a0 = bias ? _mm256_loadu_ps(bias + ch) : zero;
                            __m256 a1 = zero, a2 = zero;
#define VX_DW3_F32(KY, KX, A) (A) = _mm256_fmadd_ps( \
    _mm256_loadu_ps(s + (KY) * rstep + (KX) * cstep), \
    _mm256_loadu_ps(ww + ((long)(KY) * 3 + (KX)) * c), (A))
                            VX_DW3_F32(0, 0, a0); VX_DW3_F32(0, 1, a1); VX_DW3_F32(0, 2, a2);
                            VX_DW3_F32(1, 0, a0); VX_DW3_F32(1, 1, a1); VX_DW3_F32(1, 2, a2);
                            VX_DW3_F32(2, 0, a0); VX_DW3_F32(2, 1, a1); VX_DW3_F32(2, 2, a2);
#undef VX_DW3_F32
                            __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), a2);
                            if (relu) {
                                acc = _mm256_max_ps(acc, zero);
                                if (relu >= 2) acc = _mm256_min_ps(acc, six);
                            }
                            _mm256_storeu_ps(dst + ch, acc);
                        }
                    } else
                    for (; ch + 8 <= c; ch += 8) {
                        __m256 acc = bias ? _mm256_loadu_ps(bias + ch) : _mm256_setzero_ps();
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const float* src = src_b + ((long)iy * w + ix) * c + ch;
                                const float* ww = wgt + ((long)ky * kw + kx) * c + ch;
                                acc = _mm256_fmadd_ps(_mm256_loadu_ps(src), _mm256_loadu_ps(ww), acc);
                            }
                        }
                        if (relu) {
                            acc = _mm256_max_ps(acc, zero);
                            if (relu >= 2) acc = _mm256_min_ps(acc, six);
                        }
                        _mm256_storeu_ps(dst + ch, acc);
                    }
#elif defined(VX_CONV_USE_WASM_SIMD128)
                    const v128_t zero = wasm_f32x4_splat(0.0f);
                    const v128_t six = wasm_f32x4_splat(6.0f);
                    int interior1x3 = (kh == 1 && kw == 3 &&
                                       iy0 >= 0 && iy0 < h &&
                                       ix0 >= 0 && ix0 + 2 * dx < w);
                    int interior3 = (kh == 3 && kw == 3 &&
                                     iy0 >= 0 && iy0 + 2 * dy < h &&
                                     ix0 >= 0 && ix0 + 2 * dx < w);
                    if (interior1x3) {
                        long cstep = (long)dx * c;
                        const float* base = src_b + ((long)iy0 * w + ix0) * c;
                        const float* w0 = wgt;
                        const float* w1 = wgt + c;
                        const float* w2 = wgt + (long)2 * c;
                        for (; ch + 4 <= c; ch += 4) {
                            v128_t acc = bias ? wasm_v128_load(bias + ch) : zero;
                            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(base + ch), wasm_v128_load(w0 + ch)));
                            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(base + cstep + ch), wasm_v128_load(w1 + ch)));
                            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(base + 2 * cstep + ch), wasm_v128_load(w2 + ch)));
                            if (relu) {
                                acc = wasm_f32x4_max(acc, zero);
                                if (relu >= 2) acc = wasm_f32x4_min(acc, six);
                            }
                            wasm_v128_store(dst + ch, acc);
                        }
                    } else if (interior3) {
                        long rstep = (long)dy * w * c, cstep = (long)dx * c;
                        const float* base = src_b + ((long)iy0 * w + ix0) * c;
                        for (; ch + 4 <= c; ch += 4) {
                            const float* s = base + ch;
                            const float* ww = wgt + ch;
                            v128_t acc = bias ? wasm_v128_load(bias + ch) : zero;
                            for (int ky = 0; ky < 3; ky++) {
                                for (int kx = 0; kx < 3; kx++) {
                                    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(
                                        wasm_v128_load(s + (long)ky * rstep + (long)kx * cstep),
                                        wasm_v128_load(ww + ((long)ky * 3 + kx) * c)));
                                }
                            }
                            if (relu) {
                                acc = wasm_f32x4_max(acc, zero);
                                if (relu >= 2) acc = wasm_f32x4_min(acc, six);
                            }
                            wasm_v128_store(dst + ch, acc);
                        }
                    } else {
                        for (; ch + 4 <= c; ch += 4) {
                            v128_t acc = bias ? wasm_v128_load(bias + ch) : zero;
                            for (int ky = 0; ky < kh; ky++) {
                                int iy = iy0 + ky * dy;
                                if ((unsigned)iy >= (unsigned)h) continue;
                                for (int kx = 0; kx < kw; kx++) {
                                    int ix = ix0 + kx * dx;
                                    if ((unsigned)ix >= (unsigned)w) continue;
                                    const float* src = src_b + ((long)iy * w + ix) * c + ch;
                                    const float* ww = wgt + ((long)ky * kw + kx) * c + ch;
                                    acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_v128_load(src), wasm_v128_load(ww)));
                                }
                            }
                            if (relu) {
                                acc = wasm_f32x4_max(acc, zero);
                                if (relu >= 2) acc = wasm_f32x4_min(acc, six);
                            }
                            wasm_v128_store(dst + ch, acc);
                        }
                    }
#endif
                    for (; ch < c; ch++) {
                        float sum = bias ? bias[ch] : 0.0f;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const float* src = src_b + ((long)iy * w + ix) * c;
                                const float* ww = wgt + ((long)ky * kw + kx) * c;
                                sum += src[ch] * ww[ch];
                            }
                        }
                        dst[ch] = vx_relu6_apply(sum, relu);
                    }
                } else {
                    for (int ch = 0; ch < c; ch++) {
                        for (int m = 0; m < mult; m++) {
                            int oc = ch * mult + m;
                            float sum = bias ? bias[oc] : 0.0f;
                            for (int ky = 0; ky < kh; ky++) {
                                int iy = iy0 + ky * dy;
                                if ((unsigned)iy >= (unsigned)h) continue;
                                for (int kx = 0; kx < kw; kx++) {
                                    int ix = ix0 + kx * dx;
                                    if ((unsigned)ix >= (unsigned)w) continue;
                                    sum += src_b[((long)iy * w + ix) * c + ch] *
                                           wgt[(((long)ky * kw + kx) * c + ch) * mult + m];
                                }
                            }
                            dst[oc] = vx_relu6_apply(sum, relu);
                        }
                    }
                }
            }
        }
    }
}

#if defined(VX_CONV_USE_NEON)
static int vx_conv2d_generic_neon_f32(
        const float* input, float* output, const float* wgt, const float* bias,
        int n, int h, int w, int c, int oh, int ow, int out_c,
        int kh, int kw, int sy, int sx, const int* pads,
        int dy, int dx, int relu) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
    if (out_c < 4) return 0;
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * out_c;
        for (int oy = 0; oy < oh; oy++) {
            const int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < ow; ox++) {
                const int ix0 = ox * sx - pads[1];
                float* dst = dst_b + ((long)oy * ow + ox) * out_c;
                /* Interior horizontal tiles reuse every HWIO weight vector
                 * across four adjacent output pixels.  Each accumulator still
                 * sees the authored ky/kx/ic sequence, so this changes neither
                 * convolution semantics nor floating-point reduction order. */
                if ((out_c & 7) == 0 && ox + 3 < ow && ix0 >= 0 &&
                    ix0 + 3 * sx + (kw - 1) * dx < w) {
                    for (int oc = 0; oc < out_c; oc += 8) {
                        float32x4_t acc00 = bias ? vld1q_f32(bias + oc) : zero;
                        float32x4_t acc01 = bias ? vld1q_f32(bias + oc + 4) : zero;
                        float32x4_t acc10 = acc00;
                        float32x4_t acc11 = acc01;
                        float32x4_t acc20 = acc00;
                        float32x4_t acc21 = acc01;
                        float32x4_t acc30 = acc00;
                        float32x4_t acc31 = acc01;
                        for (int ky = 0; ky < kh; ky++) {
                            const int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                const int ix = ix0 + kx * dx;
                                const float* src0 = src_b +
                                    ((long)iy * w + ix) * c;
                                const float* src1 = src0 + (long)sx * c;
                                const float* src2 = src1 + (long)sx * c;
                                const float* src3 = src2 + (long)sx * c;
                                const float* ww = wgt +
                                    (((long)ky * kw + kx) * c) * out_c + oc;
                                for (int ic = 0; ic < c; ic++) {
                                    const float32x4_t weights0 =
                                        vld1q_f32(ww + (long)ic * out_c);
                                    const float32x4_t weights1 =
                                        vld1q_f32(ww + (long)ic * out_c + 4);
                                    acc00 = vmlaq_n_f32(acc00, weights0, src0[ic]);
                                    acc01 = vmlaq_n_f32(acc01, weights1, src0[ic]);
                                    acc10 = vmlaq_n_f32(acc10, weights0, src1[ic]);
                                    acc11 = vmlaq_n_f32(acc11, weights1, src1[ic]);
                                    acc20 = vmlaq_n_f32(acc20, weights0, src2[ic]);
                                    acc21 = vmlaq_n_f32(acc21, weights1, src2[ic]);
                                    acc30 = vmlaq_n_f32(acc30, weights0, src3[ic]);
                                    acc31 = vmlaq_n_f32(acc31, weights1, src3[ic]);
                                }
                            }
                        }
                        if (relu) {
                            acc00 = vmaxq_f32(acc00, zero);
                            acc01 = vmaxq_f32(acc01, zero);
                            acc10 = vmaxq_f32(acc10, zero);
                            acc11 = vmaxq_f32(acc11, zero);
                            acc20 = vmaxq_f32(acc20, zero);
                            acc21 = vmaxq_f32(acc21, zero);
                            acc30 = vmaxq_f32(acc30, zero);
                            acc31 = vmaxq_f32(acc31, zero);
                            if (relu >= 2) {
                                acc00 = vminq_f32(acc00, six);
                                acc01 = vminq_f32(acc01, six);
                                acc10 = vminq_f32(acc10, six);
                                acc11 = vminq_f32(acc11, six);
                                acc20 = vminq_f32(acc20, six);
                                acc21 = vminq_f32(acc21, six);
                                acc30 = vminq_f32(acc30, six);
                                acc31 = vminq_f32(acc31, six);
                            }
                        }
                        vst1q_f32(dst + oc, acc00);
                        vst1q_f32(dst + oc + 4, acc01);
                        vst1q_f32(dst + out_c + oc, acc10);
                        vst1q_f32(dst + out_c + oc + 4, acc11);
                        vst1q_f32(dst + 2 * out_c + oc, acc20);
                        vst1q_f32(dst + 2 * out_c + oc + 4, acc21);
                        vst1q_f32(dst + 3 * out_c + oc, acc30);
                        vst1q_f32(dst + 3 * out_c + oc + 4, acc31);
                    }
                    ox += 3;
                    continue;
                }
                int oc = 0;
                for (; oc + 8 <= out_c; oc += 8) {
                    float32x4_t acc0 = bias ? vld1q_f32(bias + oc) : zero;
                    float32x4_t acc1 = bias ? vld1q_f32(bias + oc + 4) : zero;
                    for (int ky = 0; ky < kh; ky++) {
                        const int iy = iy0 + ky * dy;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            const int ix = ix0 + kx * dx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            const float* src = src_b + ((long)iy * w + ix) * c;
                            const float* ww = wgt +
                                (((long)ky * kw + kx) * c) * out_c + oc;
                            for (int ic = 0; ic < c; ic++) {
                                const float value = src[ic];
                                acc0 = vmlaq_n_f32(acc0,
                                    vld1q_f32(ww + (long)ic * out_c), value);
                                acc1 = vmlaq_n_f32(acc1,
                                    vld1q_f32(ww + (long)ic * out_c + 4), value);
                            }
                        }
                    }
                    if (relu) {
                        acc0 = vmaxq_f32(acc0, zero);
                        acc1 = vmaxq_f32(acc1, zero);
                        if (relu >= 2) {
                            acc0 = vminq_f32(acc0, six);
                            acc1 = vminq_f32(acc1, six);
                        }
                    }
                    vst1q_f32(dst + oc, acc0);
                    vst1q_f32(dst + oc + 4, acc1);
                }
                for (; oc + 4 <= out_c; oc += 4) {
                    float32x4_t acc = bias ? vld1q_f32(bias + oc) : zero;
                    for (int ky = 0; ky < kh; ky++) {
                        const int iy = iy0 + ky * dy;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            const int ix = ix0 + kx * dx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            const float* src = src_b + ((long)iy * w + ix) * c;
                            const float* ww = wgt +
                                (((long)ky * kw + kx) * c) * out_c + oc;
                            for (int ic = 0; ic < c; ic++)
                                acc = vmlaq_n_f32(acc,
                                    vld1q_f32(ww + (long)ic * out_c), src[ic]);
                        }
                    }
                    if (relu) {
                        acc = vmaxq_f32(acc, zero);
                        if (relu >= 2) acc = vminq_f32(acc, six);
                    }
                    vst1q_f32(dst + oc, acc);
                }
                for (; oc < out_c; oc++) {
                    float sum = bias ? bias[oc] : 0.0f;
                    for (int ky = 0; ky < kh; ky++) {
                        const int iy = iy0 + ky * dy;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            const int ix = ix0 + kx * dx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            const float* src = src_b + ((long)iy * w + ix) * c;
                            const float* ww = wgt +
                                (((long)ky * kw + kx) * c) * out_c + oc;
                            for (int ic = 0; ic < c; ic++)
                                sum += src[ic] * ww[(long)ic * out_c];
                        }
                    }
                    dst[oc] = vx_relu6_apply(sum, relu);
                }
            }
        }
    }
    return 1;
}
#endif

typedef struct {
    const float* input;
    float* output;
    const float* wgt;
    const float* bias;
    int n, h, w, c;
    int oh, ow, out_c;
    int kh, kw;
    int sy, sx;
    const int* pads;
    int dy, dx, relu;
    int groups;
    int group_out;
    int group_in;
    int in_per_group;
} VxConv2DGenericTask;

#if defined(__AVX2__)
static void vx_conv2d_generic_avx2_worker(void* context, int begin, int end) {
    const VxConv2DGenericTask* task = (const VxConv2DGenericTask*)context;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
    const long spatial_per_batch = (long)task->oh * task->ow;
    for (int p = begin; p < end; p++) {
        int b = (int)(p / spatial_per_batch);
        long rem = p % spatial_per_batch;
        int oy = (int)(rem / task->ow);
        int ox = (int)(rem % task->ow);
        const float* src_b = task->input + (long)b * task->h * task->w * task->c;
        float* dst = task->output + (long)p * task->out_c;
        int iy0 = oy * task->sy - task->pads[0];
        int ix0 = ox * task->sx - task->pads[1];
        int oc = 0;
        for (; oc + 32 <= task->out_c; oc += 32) {
            __m256 acc0 = task->bias ? _mm256_loadu_ps(task->bias + oc + 0) : zero;
            __m256 acc1 = task->bias ? _mm256_loadu_ps(task->bias + oc + 8) : zero;
            __m256 acc2 = task->bias ? _mm256_loadu_ps(task->bias + oc + 16) : zero;
            __m256 acc3 = task->bias ? _mm256_loadu_ps(task->bias + oc + 24) : zero;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++) {
                        __m256 xv = _mm256_set1_ps(src[ic]);
                        const float* w_ic = ww + (long)ic * task->out_c;
                        acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 0), acc0);
                        acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 8), acc1);
                        acc2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 16), acc2);
                        acc3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 24), acc3);
                    }
                }
            }
            if (task->relu) {
                acc0 = _mm256_max_ps(acc0, zero);
                acc1 = _mm256_max_ps(acc1, zero);
                acc2 = _mm256_max_ps(acc2, zero);
                acc3 = _mm256_max_ps(acc3, zero);
                if (task->relu >= 2) {
                    acc0 = _mm256_min_ps(acc0, six);
                    acc1 = _mm256_min_ps(acc1, six);
                    acc2 = _mm256_min_ps(acc2, six);
                    acc3 = _mm256_min_ps(acc3, six);
                }
            }
            _mm256_storeu_ps(dst + oc + 0, acc0);
            _mm256_storeu_ps(dst + oc + 8, acc1);
            _mm256_storeu_ps(dst + oc + 16, acc2);
            _mm256_storeu_ps(dst + oc + 24, acc3);
        }
        for (; oc + 8 <= task->out_c; oc += 8) {
            __m256 acc = task->bias ? _mm256_loadu_ps(task->bias + oc) : zero;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++) {
                        __m256 xv = _mm256_set1_ps(src[ic]);
                        __m256 wv = _mm256_loadu_ps(ww + (long)ic * task->out_c);
                        acc = _mm256_fmadd_ps(xv, wv, acc);
                    }
                }
            }
            if (task->relu) {
                acc = _mm256_max_ps(acc, zero);
                if (task->relu >= 2) acc = _mm256_min_ps(acc, six);
            }
            _mm256_storeu_ps(dst + oc, acc);
        }
        for (; oc < task->out_c; oc++) {
            float sum = task->bias ? task->bias[oc] : 0.0f;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++) sum += src[ic] * ww[(long)ic * task->out_c];
                }
            }
            dst[oc] = vx_relu6_apply(sum, task->relu);
        }
    }
}
#endif

#if defined(VX_CONV_USE_NEON)
static void vx_conv2d_generic_neon_worker(void* context, int begin, int end) {
    const VxConv2DGenericTask* task = (const VxConv2DGenericTask*)context;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
    const long spatial_per_batch = (long)task->oh * task->ow;
    for (int p = begin; p < end; p++) {
        int b = (int)(p / spatial_per_batch);
        long rem = p % spatial_per_batch;
        int oy = (int)(rem / task->ow);
        int ox = (int)(rem % task->ow);
        const float* src_b = task->input + (long)b * task->h * task->w * task->c;
        float* dst = task->output + (long)p * task->out_c;
        int iy0 = oy * task->sy - task->pads[0];
        int ix0 = ox * task->sx - task->pads[1];
        int oc = 0;
        for (; oc + 8 <= task->out_c; oc += 8) {
            float32x4_t acc0 = task->bias ? vld1q_f32(task->bias + oc) : zero;
            float32x4_t acc1 = task->bias ? vld1q_f32(task->bias + oc + 4) : zero;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++) {
                        float val = src[ic];
                        acc0 = vmlaq_n_f32(acc0, vld1q_f32(ww + (long)ic * task->out_c), val);
                        acc1 = vmlaq_n_f32(acc1, vld1q_f32(ww + (long)ic * task->out_c + 4), val);
                    }
                }
            }
            if (task->relu) {
                acc0 = vmaxq_f32(acc0, zero);
                acc1 = vmaxq_f32(acc1, zero);
                if (task->relu >= 2) {
                    acc0 = vminq_f32(acc0, six);
                    acc1 = vminq_f32(acc1, six);
                }
            }
            vst1q_f32(dst + oc, acc0);
            vst1q_f32(dst + oc + 4, acc1);
        }
        for (; oc + 4 <= task->out_c; oc += 4) {
            float32x4_t acc = task->bias ? vld1q_f32(task->bias + oc) : zero;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++)
                        acc = vmlaq_n_f32(acc, vld1q_f32(ww + (long)ic * task->out_c), src[ic]);
                }
            }
            if (task->relu) {
                acc = vmaxq_f32(acc, zero);
                if (task->relu >= 2) acc = vminq_f32(acc, six);
            }
            vst1q_f32(dst + oc, acc);
        }
        for (; oc < task->out_c; oc++) {
            float sum = task->bias ? task->bias[oc] : 0.0f;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->c) * task->out_c + oc;
                    for (int ic = 0; ic < task->c; ic++)
                        sum += src[ic] * ww[(long)ic * task->out_c];
                }
            }
            dst[oc] = vx_relu6_apply(sum, task->relu);
        }
    }
}
#endif

static void vx_conv2d_generic_grouped_worker(void* context, int begin, int end) {
    const VxConv2DGenericTask* task = (const VxConv2DGenericTask*)context;
    const long spatial_per_batch = (long)task->oh * task->ow;
    for (int p = begin; p < end; p++) {
        int b = (int)(p / spatial_per_batch);
        long rem = p % spatial_per_batch;
        int oy = (int)(rem / task->ow);
        int ox = (int)(rem % task->ow);
        const float* src_b = task->input + (long)b * task->h * task->w * task->c;
        float* dst = task->output + (long)p * task->out_c;
        int iy0 = oy * task->sy - task->pads[0];
        int ix0 = ox * task->sx - task->pads[1];
        for (int oc = 0; oc < task->out_c; oc++) {
            int g = oc / task->group_out;
            int in_start = g * task->group_in;
            float sum = task->bias ? task->bias[oc] : 0.0f;
            for (int ky = 0; ky < task->kh; ky++) {
                int iy = iy0 + ky * task->dy;
                if ((unsigned)iy >= (unsigned)task->h) continue;
                for (int kx = 0; kx < task->kw; kx++) {
                    int ix = ix0 + kx * task->dx;
                    if ((unsigned)ix >= (unsigned)task->w) continue;
                    const float* src = src_b + ((long)iy * task->w + ix) * task->c + in_start;
                    const float* ww = task->wgt + (((long)ky * task->kw + kx) * task->in_per_group) * task->out_c + oc;
                    for (int ci = 0; ci < task->in_per_group; ci++) sum += src[ci] * ww[(long)ci * task->out_c];
                }
            }
            dst[oc] = vx_relu6_apply(sum, task->relu);
        }
    }
}

void vx_conv2d_generic_f32(int node_idx,
                                const float* input, float* output,
                                const float* wgt, const float* bias,
                                int n, int h, int w, int c,
                                int oh, int ow, int out_c,
                                int kh, int kw, int in_per_group,
                                int groups, int sy, int sx,
                                const int* pads, int dy, int dx, int relu) {
    int group_out = out_c / groups;
    int group_in = c / groups;
#if defined(__AVX2__)
    if (groups == 1) {
        if (vx_conv2d_spatial_igemm_f32(node_idx, input, output, wgt, bias,
                                             n, h, w, c, oh, ow, out_c, kh, kw,
                                             sy, sx, pads, dy, dx, relu)) {
            return;
        }
        const long total_pixels = (long)n * oh * ow;
        const int threads = vx_kernels_thread_count();
        if (threads > 1 && total_pixels > 1 && total_pixels <= INT_MAX) {
            VxConv2DGenericTask task = {
                input, output, wgt, bias,
                n, h, w, c, oh, ow, out_c,
                kh, kw, sy, sx, pads, dy, dx, relu,
                groups, group_out, group_in, in_per_group
            };
            int grain = (int)((total_pixels + threads * 4 - 1) / (threads * 4));
            if (grain < 1) grain = 1;
            vx_kernels_parallel_for((int)total_pixels, grain,
                                    vx_conv2d_generic_avx2_worker, &task);
            return;
        }
        const __m256 zero = _mm256_setzero_ps();
        const __m256 six = _mm256_set1_ps(6.0f);
        for (int b = 0; b < n; b++) {
            const float* src_b = input + (long)b * h * w * c;
            float* dst_b = output + (long)b * oh * ow * out_c;
            for (int oy = 0; oy < oh; oy++) {
                int iy0 = oy * sy - pads[0];
                for (int ox = 0; ox < ow; ox++) {
                    int ix0 = ox * sx - pads[1];
                    float* dst = dst_b + ((long)oy * ow + ox) * out_c;
                    int oc = 0;
                    for (; oc + 32 <= out_c; oc += 32) {
                        __m256 acc0 = bias ? _mm256_loadu_ps(bias + oc + 0) : zero;
                        __m256 acc1 = bias ? _mm256_loadu_ps(bias + oc + 8) : zero;
                        __m256 acc2 = bias ? _mm256_loadu_ps(bias + oc + 16) : zero;
                        __m256 acc3 = bias ? _mm256_loadu_ps(bias + oc + 24) : zero;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const float* src = src_b + ((long)iy * w + ix) * c;
                                const float* ww = wgt + (((long)ky * kw + kx) * c) * out_c + oc;
                                for (int ic = 0; ic < c; ic++) {
                                    __m256 xv = _mm256_set1_ps(src[ic]);
                                    const float* w_ic = ww + (long)ic * out_c;
                                    acc0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 0), acc0);
                                    acc1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 8), acc1);
                                    acc2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 16), acc2);
                                    acc3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w_ic + 24), acc3);
                                }
                            }
                        }
                        if (relu) {
                            acc0 = _mm256_max_ps(acc0, zero);
                            acc1 = _mm256_max_ps(acc1, zero);
                            acc2 = _mm256_max_ps(acc2, zero);
                            acc3 = _mm256_max_ps(acc3, zero);
                            if (relu >= 2) {
                                acc0 = _mm256_min_ps(acc0, six);
                                acc1 = _mm256_min_ps(acc1, six);
                                acc2 = _mm256_min_ps(acc2, six);
                                acc3 = _mm256_min_ps(acc3, six);
                            }
                        }
                        _mm256_storeu_ps(dst + oc + 0, acc0);
                        _mm256_storeu_ps(dst + oc + 8, acc1);
                        _mm256_storeu_ps(dst + oc + 16, acc2);
                        _mm256_storeu_ps(dst + oc + 24, acc3);
                    }
                    for (; oc + 8 <= out_c; oc += 8) {
                        __m256 acc = bias ? _mm256_loadu_ps(bias + oc) : zero;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const float* src = src_b + ((long)iy * w + ix) * c;
                                const float* ww = wgt + (((long)ky * kw + kx) * c) * out_c + oc;
                                for (int ic = 0; ic < c; ic++) {
                                    __m256 xv = _mm256_set1_ps(src[ic]);
                                    __m256 wv = _mm256_loadu_ps(ww + (long)ic * out_c);
                                    acc = _mm256_fmadd_ps(xv, wv, acc);
                                }
                            }
                        }
                        if (relu) {
                            acc = _mm256_max_ps(acc, zero);
                            if (relu >= 2) acc = _mm256_min_ps(acc, six);
                        }
                        _mm256_storeu_ps(dst + oc, acc);
                    }
                    for (; oc < out_c; oc++) {
                        float sum = bias ? bias[oc] : 0.0f;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky * dy;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx * dx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const float* src = src_b + ((long)iy * w + ix) * c;
                                const float* ww = wgt + (((long)ky * kw + kx) * c) * out_c + oc;
                                for (int ic = 0; ic < c; ic++) sum += src[ic] * ww[(long)ic * out_c];
                            }
                        }
                        dst[oc] = vx_relu6_apply(sum, relu);
                    }
                }
            }
        }
        return;
    }
#endif
#if defined(VX_CONV_USE_NEON)
    if (groups == 1 && vx_kernel_platform()->has_neon) {
        if (vx_conv2d_spatial_igemm_f32(
                node_idx, input, output, wgt, bias, n, h, w, c,
                oh, ow, out_c, kh, kw, sy, sx, pads, dy, dx, relu)) return;
        const long total_pixels = (long)n * oh * ow;
        const int threads = vx_kernels_thread_count();
        if (threads > 1 && total_pixels > 1 && total_pixels <= INT_MAX && out_c >= 4) {
            VxConv2DGenericTask task = {
                input, output, wgt, bias,
                n, h, w, c, oh, ow, out_c,
                kh, kw, sy, sx, pads, dy, dx, relu,
                groups, group_out, group_in, in_per_group
            };
            int grain = (int)((total_pixels + threads * 4 - 1) / (threads * 4));
            if (grain < 1) grain = 1;
            vx_kernels_parallel_for((int)total_pixels, grain,
                                    vx_conv2d_generic_neon_worker, &task);
            return;
        }
        if (vx_conv2d_generic_neon_f32(
                input, output, wgt, bias, n, h, w, c, oh, ow, out_c,
                kh, kw, sy, sx, pads, dy, dx, relu)) return;
    }
#endif
#if defined(VX_CONV_USE_WASM_SIMD128)
    if (groups == 1) {
        if (vx_conv2d_spatial_igemm_f32(
                node_idx, input, output, wgt, bias, n, h, w, c,
                oh, ow, out_c, kh, kw, sy, sx, pads, dy, dx, relu)) return;
    }
#endif
    const long total_pixels = (long)n * oh * ow;
    const int threads = vx_kernels_thread_count();
    if (threads > 1 && total_pixels > 1 && total_pixels <= INT_MAX) {
        VxConv2DGenericTask task = {
            input, output, wgt, bias,
            n, h, w, c, oh, ow, out_c,
            kh, kw, sy, sx, pads, dy, dx, relu,
            groups, group_out, group_in, in_per_group
        };
        int grain = (int)((total_pixels + threads * 4 - 1) / (threads * 4));
        if (grain < 1) grain = 1;
        vx_kernels_parallel_for((int)total_pixels, grain,
                                vx_conv2d_generic_grouped_worker, &task);
        return;
    }
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * out_c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * sx - pads[1];
                float* dst = dst_b + ((long)oy * ow + ox) * out_c;
                for (int oc = 0; oc < out_c; oc++) {
                    int g = oc / group_out;
                    int in_start = g * group_in;
                    float sum = bias ? bias[oc] : 0.0f;
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = iy0 + ky * dy;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx * dx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            const float* src = src_b + ((long)iy * w + ix) * c + in_start;
                            const float* ww = wgt + (((long)ky * kw + kx) * in_per_group) * out_c + oc;
                            for (int ci = 0; ci < in_per_group; ci++) sum += src[ci] * ww[(long)ci * out_c];
                        }
                    }
                    dst[oc] = vx_relu6_apply(sum, relu);
                }
            }
        }
    }
}

static uint64_t f32_igemm_indirection_key(const float* input,
                                          int n, int h, int w, int c,
                                          int oh, int ow, int out_c,
                                          int kh, int kw,
                                          int sy, int sx, const int* pads,
                                          int dy, int dx) {
    uint64_t k = 1469598103934665603ULL ^ (uint64_t)(uintptr_t)input;
#define VX_IGEMM_KEY_MIX(v) do { k ^= (uint64_t)(uint32_t)(v); k *= 1099511628211ULL; } while (0)
    VX_IGEMM_KEY_MIX(n); VX_IGEMM_KEY_MIX(h); VX_IGEMM_KEY_MIX(w); VX_IGEMM_KEY_MIX(c);
    VX_IGEMM_KEY_MIX(oh); VX_IGEMM_KEY_MIX(ow); VX_IGEMM_KEY_MIX(out_c);
    VX_IGEMM_KEY_MIX(kh); VX_IGEMM_KEY_MIX(kw); VX_IGEMM_KEY_MIX(sy); VX_IGEMM_KEY_MIX(sx);
    VX_IGEMM_KEY_MIX(dy); VX_IGEMM_KEY_MIX(dx);
    VX_IGEMM_KEY_MIX(pads[0]); VX_IGEMM_KEY_MIX(pads[1]);
    VX_IGEMM_KEY_MIX(pads[2]); VX_IGEMM_KEY_MIX(pads[3]);
#undef VX_IGEMM_KEY_MIX
    return k ? k : 1;
}

#define g_f32_igemm_pack (vx_engine_state_current()->conv_f32_igemm_pack)
#define g_f32_igemm_pack_cap (vx_engine_state_current()->conv_f32_igemm_pack_cap)
#define g_f32_igemm_pack_src (vx_engine_state_current()->conv_f32_igemm_pack_src)

/* Repack HWIO spatial weights to [oc/16][tap][ic][16].
 *
 * This is the same transform `vx_pwf32_pack_plain_cache` does for pointwise
 * convolutions, with the kernel taps added: at a fixed output-channel block the
 * igemm walks `tap` then `ic`, and in that order the packed buffer is one
 * unbroken sequential stream of full cache lines. HWIO instead strides
 * `out_c * 4` bytes per input channel, which measured 90 GFLOP/s at 320
 * channels against 115 packed, with the shortfall tracking cache lines per page
 * rather than working-set size.
 *
 * The weight is immutable, so this is paid once per node — 1.77 ms for the whole
 * encoder — and costs one extra copy of the convolution weights. Lane order
 * within a block is preserved, so the reduction is unchanged and results stay
 * bit-identical.
 */
const float* vx_f32_igemm_pack_cache(int node_idx, const float* wgt,
                                     int c, int out_c, int ks) {
    if (node_idx < 0 || (size_t)node_idx >= vx_engine_state_current()->node_capacity ||
        !wgt) return NULL;
    if (c < 1 || out_c < 16 || ks < 1) return NULL;
    int blocks = out_c / 16;
    if (blocks < 1) return NULL;
    long need = (long)blocks * ks * c * 16;
    if (need <= 0) return NULL;
    /* Keyed on the source buffer as well as the extent. Conv weights are model
     * constants, but the weight-only INT8 path feeds this a per-node dequant
     * cache, so the node alone does not identify the bytes. */
    if (g_f32_igemm_pack[node_idx] && g_f32_igemm_pack_cap[node_idx] == need &&
        g_f32_igemm_pack_src[node_idx] == wgt)
        return g_f32_igemm_pack[node_idx];
    float* pk = (float*)malloc((size_t)need * sizeof(float));
    if (!pk) return NULL;
    for (int b = 0; b < blocks; b++) {
        for (int t = 0; t < ks; t++) {
            for (int ic = 0; ic < c; ic++) {
                float* dst = pk + (((long)b * ks + t) * c + ic) * 16;
                const float* src = wgt + ((long)t * c + ic) * out_c + b * 16;
                for (int lane = 0; lane < 16; lane++) dst[lane] = src[lane];
            }
        }
    }
    free(g_f32_igemm_pack[node_idx]);
    g_f32_igemm_pack[node_idx] = pk;
    g_f32_igemm_pack_cap[node_idx] = need;
    g_f32_igemm_pack_src[node_idx] = wgt;
    return pk;
}

const float** vx_f32_igemm_indirection_cache(int node_idx, const float* input,
                                             int n, int h, int w, int c,
                                             int oh, int ow, int kh, int kw,
                                             int sy, int sx, const int* pads,
                                             int dy, int dx) {
    if (node_idx < 0 || (size_t)node_idx >= vx_engine_state_current()->node_capacity || !input ||
        sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0) return NULL;
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0 || oh <= 0 || ow <= 0 || kh <= 0 || kw <= 0) return NULL;

    if (!g_f32_igemm_zero[node_idx] || g_f32_igemm_zero_cap[node_idx] < c) {
        float* z = (float*)calloc((size_t)c, sizeof(float));
        if (!z) return NULL;
        free(g_f32_igemm_zero[node_idx]);
        g_f32_igemm_zero[node_idx] = z;
        g_f32_igemm_zero_cap[node_idx] = c;
    }

    long pixels = (long)n * oh * ow;
    int ks = kh * kw;
    long need = pixels * ks;
    if (need <= 0) return NULL;
    if (!g_f32_igemm_indir[node_idx] || g_f32_igemm_indir_cap[node_idx] < need) {
        const float** p = (const float**)realloc(g_f32_igemm_indir[node_idx], (size_t)need * sizeof(float*));
        if (!p) return NULL;
        g_f32_igemm_indir[node_idx] = p;
        g_f32_igemm_indir_cap[node_idx] = need;
        g_f32_igemm_indir_key[node_idx] = 0;
    }

    uint64_t key = f32_igemm_indirection_key(input, n, h, w, c, oh, ow, 0, kh, kw, sy, sx, pads, dy, dx);
    if (g_f32_igemm_indir_key[node_idx] == key) return g_f32_igemm_indir[node_idx];

    const float* zero = g_f32_igemm_zero[node_idx];
    const float** table = g_f32_igemm_indir[node_idx];
    long p = 0;
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < ow; ox++, p++) {
                int ix0 = ox * sx - pads[1];
                const float** row = table + p * ks;
                int t = 0;
                for (int ky = 0; ky < kh; ky++) {
                    int iy = iy0 + ky * dy;
                    for (int kx = 0; kx < kw; kx++, t++) {
                        int ix = ix0 + kx * dx;
                        row[t] = ((unsigned)iy < (unsigned)h && (unsigned)ix < (unsigned)w) ?
                                 src_b + ((long)iy * w + ix) * c : zero;
                    }
                }
            }
        }
    }
    g_f32_igemm_indir_key[node_idx] = key;
    return table;
}

#if defined(__AVX2__)
typedef struct {
    float* output;
    const float* weights;
    const float* packed_weights;
    const float* bias;
    const float* const* indirection;
    int input_channels;
    int output_channels;
    int kernel_size;
    long weight_step;
    long weight_tap;
    long block_offset;
    int relu;
} VxF32IgemmSixPixelContext;

/* Each task owns six complete output pixels.  The immutable indirection and
 * packed-weight caches are finalized by the owner thread before dispatch, so
 * workers touch no engine state and write disjoint output ranges.  Keeping the
 * existing six-pixel microkernel intact preserves its per-output accumulation
 * order while allowing large image convolutions to use the context thread
 * pool. */
/* Both pool partitions use the same ordered microkernel. Keep one O3 body:
 * inlining it into each wrapper duplicates all SIMD and scalar tail loops. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void vx_f32_igemm_six_pixel_range(
        const VxF32IgemmSixPixelContext* context, int begin, int end,
        int oc_begin, int oc_end) {
    const int out_c = context->output_channels;
    const int c = context->input_channels;
    const int ks = context->kernel_size;
    const long wstep = context->weight_step;
    const long wtap = context->weight_tap;
    const int relu = context->relu;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#define VX_IGEMM_WORKER_RELU(v) do { \
    if (relu) { \
        (v) = _mm256_max_ps((v), zero); \
        if (relu >= 2) (v) = _mm256_min_ps((v), six); \
    } \
} while (0)
    int oc = oc_begin;
    for (; oc + 16 <= oc_end; oc += 16) {
        const float* wbase = context->packed_weights
            ? context->packed_weights + ((long)(oc >> 4) * ks) * c * 16
            : context->weights + oc;
        __m256 bl = context->bias ? _mm256_loadu_ps(context->bias + oc) : zero;
        __m256 bh = context->bias ? _mm256_loadu_ps(context->bias + oc + 8) : zero;
        for (int block = begin; block < end; block++) {
            const long p = (context->block_offset + (long)block) * 6;
            float* d0 = context->output + (p + 0) * out_c;
            float* d1 = context->output + (p + 1) * out_c;
            float* d2 = context->output + (p + 2) * out_c;
            float* d3 = context->output + (p + 3) * out_c;
            float* d4 = context->output + (p + 4) * out_c;
            float* d5 = context->output + (p + 5) * out_c;
            const float* const* r0 = context->indirection + (p + 0) * ks;
            const float* const* r1 = context->indirection + (p + 1) * ks;
            const float* const* r2 = context->indirection + (p + 2) * ks;
            const float* const* r3 = context->indirection + (p + 3) * ks;
            const float* const* r4 = context->indirection + (p + 4) * ks;
            const float* const* r5 = context->indirection + (p + 5) * ks;
            __m256 a0l = bl, a0h = bh, a1l = bl, a1h = bh;
            __m256 a2l = bl, a2h = bh, a3l = bl, a3h = bh;
            __m256 a4l = bl, a4h = bh, a5l = bl, a5h = bh;
            for (int t = 0; t < ks; t++) {
                const float* s0 = r0[t], * s1 = r1[t], * s2 = r2[t];
                const float* s3 = r3[t], * s4 = r4[t], * s5 = r5[t];
                const float* ww = wbase + (long)t * wtap;
                for (int ic = 0; ic < c; ic++, ww += wstep) {
                    __m256 wl = _mm256_loadu_ps(ww);
                    __m256 wh = _mm256_loadu_ps(ww + 8);
                    __m256 x;
                    x = _mm256_set1_ps(s0[ic]); a0l = _mm256_fmadd_ps(x, wl, a0l); a0h = _mm256_fmadd_ps(x, wh, a0h);
                    x = _mm256_set1_ps(s1[ic]); a1l = _mm256_fmadd_ps(x, wl, a1l); a1h = _mm256_fmadd_ps(x, wh, a1h);
                    x = _mm256_set1_ps(s2[ic]); a2l = _mm256_fmadd_ps(x, wl, a2l); a2h = _mm256_fmadd_ps(x, wh, a2h);
                    x = _mm256_set1_ps(s3[ic]); a3l = _mm256_fmadd_ps(x, wl, a3l); a3h = _mm256_fmadd_ps(x, wh, a3h);
                    x = _mm256_set1_ps(s4[ic]); a4l = _mm256_fmadd_ps(x, wl, a4l); a4h = _mm256_fmadd_ps(x, wh, a4h);
                    x = _mm256_set1_ps(s5[ic]); a5l = _mm256_fmadd_ps(x, wl, a5l); a5h = _mm256_fmadd_ps(x, wh, a5h);
                }
            }
            VX_IGEMM_WORKER_RELU(a0l); VX_IGEMM_WORKER_RELU(a0h);
            VX_IGEMM_WORKER_RELU(a1l); VX_IGEMM_WORKER_RELU(a1h);
            VX_IGEMM_WORKER_RELU(a2l); VX_IGEMM_WORKER_RELU(a2h);
            VX_IGEMM_WORKER_RELU(a3l); VX_IGEMM_WORKER_RELU(a3h);
            VX_IGEMM_WORKER_RELU(a4l); VX_IGEMM_WORKER_RELU(a4h);
            VX_IGEMM_WORKER_RELU(a5l); VX_IGEMM_WORKER_RELU(a5h);
            _mm256_storeu_ps(d0 + oc, a0l); _mm256_storeu_ps(d0 + oc + 8, a0h);
            _mm256_storeu_ps(d1 + oc, a1l); _mm256_storeu_ps(d1 + oc + 8, a1h);
            _mm256_storeu_ps(d2 + oc, a2l); _mm256_storeu_ps(d2 + oc + 8, a2h);
            _mm256_storeu_ps(d3 + oc, a3l); _mm256_storeu_ps(d3 + oc + 8, a3h);
            _mm256_storeu_ps(d4 + oc, a4l); _mm256_storeu_ps(d4 + oc + 8, a4h);
            _mm256_storeu_ps(d5 + oc, a5l); _mm256_storeu_ps(d5 + oc + 8, a5h);
        }
    }
    for (; oc + 8 <= oc_end; oc += 8) {
        __m256 a_bias = context->bias ? _mm256_loadu_ps(context->bias + oc) : zero;
        for (int block = begin; block < end; block++) {
            const long p = (context->block_offset + (long)block) * 6;
            float* d0 = context->output + (p + 0) * out_c, * d1 = context->output + (p + 1) * out_c;
            float* d2 = context->output + (p + 2) * out_c, * d3 = context->output + (p + 3) * out_c;
            float* d4 = context->output + (p + 4) * out_c, * d5 = context->output + (p + 5) * out_c;
            const float* const* r0 = context->indirection + (p + 0) * ks;
            const float* const* r1 = context->indirection + (p + 1) * ks;
            const float* const* r2 = context->indirection + (p + 2) * ks;
            const float* const* r3 = context->indirection + (p + 3) * ks;
            const float* const* r4 = context->indirection + (p + 4) * ks;
            const float* const* r5 = context->indirection + (p + 5) * ks;
            __m256 a0 = a_bias, a1 = a_bias, a2 = a_bias, a3 = a_bias, a4 = a_bias, a5 = a_bias;
            for (int t = 0; t < ks; t++) {
                const float* s0 = r0[t], * s1 = r1[t], * s2 = r2[t];
                const float* s3 = r3[t], * s4 = r4[t], * s5 = r5[t];
                const float* ww = context->weights + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) {
                    __m256 wv = _mm256_loadu_ps(ww + (long)ic * out_c);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(s0[ic]), wv, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(s1[ic]), wv, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(s2[ic]), wv, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(s3[ic]), wv, a3);
                    a4 = _mm256_fmadd_ps(_mm256_set1_ps(s4[ic]), wv, a4);
                    a5 = _mm256_fmadd_ps(_mm256_set1_ps(s5[ic]), wv, a5);
                }
            }
            VX_IGEMM_WORKER_RELU(a0); VX_IGEMM_WORKER_RELU(a1);
            VX_IGEMM_WORKER_RELU(a2); VX_IGEMM_WORKER_RELU(a3);
            VX_IGEMM_WORKER_RELU(a4); VX_IGEMM_WORKER_RELU(a5);
            _mm256_storeu_ps(d0 + oc, a0); _mm256_storeu_ps(d1 + oc, a1);
            _mm256_storeu_ps(d2 + oc, a2); _mm256_storeu_ps(d3 + oc, a3);
            _mm256_storeu_ps(d4 + oc, a4); _mm256_storeu_ps(d5 + oc, a5);
        }
    }
    for (; oc < oc_end; oc++) {
        for (int block = begin; block < end; block++) {
            const long p = (context->block_offset + (long)block) * 6;
            float* d0 = context->output + (p + 0) * out_c, * d1 = context->output + (p + 1) * out_c;
            float* d2 = context->output + (p + 2) * out_c, * d3 = context->output + (p + 3) * out_c;
            float* d4 = context->output + (p + 4) * out_c, * d5 = context->output + (p + 5) * out_c;
            const float* const* r0 = context->indirection + (p + 0) * ks;
            const float* const* r1 = context->indirection + (p + 1) * ks;
            const float* const* r2 = context->indirection + (p + 2) * ks;
            const float* const* r3 = context->indirection + (p + 3) * ks;
            const float* const* r4 = context->indirection + (p + 4) * ks;
            const float* const* r5 = context->indirection + (p + 5) * ks;
            float acc[6];
            for (int index = 0; index < 6; index++) acc[index] = context->bias ? context->bias[oc] : 0.0f;
            for (int t = 0; t < ks; t++) {
                const float* source[6] = { r0[t], r1[t], r2[t], r3[t], r4[t], r5[t] };
                const float* ww = context->weights + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) {
                    const float weight = ww[(long)ic * out_c];
                    for (int index = 0; index < 6; index++) acc[index] += source[index][ic] * weight;
                }
            }
            d0[oc] = vx_relu6_apply(acc[0], relu); d1[oc] = vx_relu6_apply(acc[1], relu);
            d2[oc] = vx_relu6_apply(acc[2], relu); d3[oc] = vx_relu6_apply(acc[3], relu);
            d4[oc] = vx_relu6_apply(acc[4], relu); d5[oc] = vx_relu6_apply(acc[5], relu);
        }
    }
#undef VX_IGEMM_WORKER_RELU
}

static void vx_f32_igemm_six_pixel_worker(void* opaque, int begin, int end) {
    const VxF32IgemmSixPixelContext* context =
        (const VxF32IgemmSixPixelContext*)opaque;
    vx_f32_igemm_six_pixel_range(context, begin, end,
                               0, context->output_channels);
}

typedef struct {
    VxF32IgemmSixPixelContext base;
    int six_pixel_blocks;
    int oc_blocks;
} VxF32IgemmSixPixelOcParallelContext;

static void vx_f32_igemm_six_pixel_oc_worker(void* opaque, int begin, int end) {
    const VxF32IgemmSixPixelOcParallelContext* context =
        (const VxF32IgemmSixPixelOcParallelContext*)opaque;
    const int oc_begin = begin * 16;
    int oc_end = end * 16;
    if (oc_end > context->base.output_channels)
        oc_end = context->base.output_channels;
    vx_f32_igemm_six_pixel_range(&context->base, 0, context->six_pixel_blocks,
                               oc_begin, oc_end);
}

typedef struct {
    const float* input;
    float* output;
    const float* packed_weights;
    const float* bias;
    int w;
    int c;
    int ow;
    int out_c;
    int kw;
    int sx;
    int pl;
    int dx;
    int relu;
    int interior_lo;
    long safe_interior_hi;
} VxConv1DDirectAvx2Context;

static void vx_conv1d_direct_avx2_worker(void* opaque, int begin, int end) {
    const VxConv1DDirectAvx2Context* ctx = (const VxConv1DDirectAvx2Context*)opaque;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int kw = ctx->kw;
    const int sx = ctx->sx;
    const int pl = ctx->pl;
    const int dx = ctx->dx;
    const int w = ctx->w;
    const int relu = ctx->relu;
    const int is_stride1 = (sx == 1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
#define VX_AVX2_DIRECT_RELU(v) do { \
    if (relu) { \
        (v) = _mm256_max_ps((v), zero); \
        if (relu >= 2) (v) = _mm256_min_ps((v), six); \
    } \
} while (0)

    for (int ox = begin; ox < end; ox += 4) {
        int rem = (ox + 4 <= end) ? 4 : (end - ox);

        for (int oc = 0; oc < out_c; oc += 16) {
            const float* weights_base = ctx->packed_weights + ((long)(oc >> 4) * kw) * c * 16;
            const __m256 bias0 = ctx->bias ? _mm256_loadu_ps(ctx->bias + oc + 0) : zero;
            const __m256 bias1 = ctx->bias ? _mm256_loadu_ps(ctx->bias + oc + 8) : zero;

            if (rem == 4 && ox >= ctx->interior_lo && ox + 4 <= ctx->safe_interior_hi && is_stride1) {
                __m256 a0l = bias0, a0h = bias1;
                __m256 a1l = bias0, a1h = bias1;
                __m256 a2l = bias0, a2h = bias1;
                __m256 a3l = bias0, a3h = bias1;

                const float* weights = weights_base;
                int ix0 = ox * sx - pl;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = ctx->input + ((long)ix0 + kx * dx) * c;
                    const float* s1 = s0 + c;
                    const float* s2 = s0 + 2 * c;
                    const float* s3 = s0 + 3 * c;

                    for (int ic = 0; ic < c; ic++, weights += 16) {
                        const __m256 wl = _mm256_loadu_ps(weights + 0);
                        const __m256 wh = _mm256_loadu_ps(weights + 8);

                        const __m256 x0 = _mm256_set1_ps(s0[ic]);
                        a0l = _mm256_fmadd_ps(x0, wl, a0l);
                        a0h = _mm256_fmadd_ps(x0, wh, a0h);

                        const __m256 x1 = _mm256_set1_ps(s1[ic]);
                        a1l = _mm256_fmadd_ps(x1, wl, a1l);
                        a1h = _mm256_fmadd_ps(x1, wh, a1h);

                        const __m256 x2 = _mm256_set1_ps(s2[ic]);
                        a2l = _mm256_fmadd_ps(x2, wl, a2l);
                        a2h = _mm256_fmadd_ps(x2, wh, a2h);

                        const __m256 x3 = _mm256_set1_ps(s3[ic]);
                        a3l = _mm256_fmadd_ps(x3, wl, a3l);
                        a3h = _mm256_fmadd_ps(x3, wh, a3h);
                    }
                }
                VX_AVX2_DIRECT_RELU(a0l); VX_AVX2_DIRECT_RELU(a0h);
                VX_AVX2_DIRECT_RELU(a1l); VX_AVX2_DIRECT_RELU(a1h);
                VX_AVX2_DIRECT_RELU(a2l); VX_AVX2_DIRECT_RELU(a2h);
                VX_AVX2_DIRECT_RELU(a3l); VX_AVX2_DIRECT_RELU(a3h);

                _mm256_storeu_ps(ctx->output + (long)(ox + 0) * out_c + oc + 0, a0l);
                _mm256_storeu_ps(ctx->output + (long)(ox + 0) * out_c + oc + 8, a0h);
                _mm256_storeu_ps(ctx->output + (long)(ox + 1) * out_c + oc + 0, a1l);
                _mm256_storeu_ps(ctx->output + (long)(ox + 1) * out_c + oc + 8, a1h);
                _mm256_storeu_ps(ctx->output + (long)(ox + 2) * out_c + oc + 0, a2l);
                _mm256_storeu_ps(ctx->output + (long)(ox + 2) * out_c + oc + 8, a2h);
                _mm256_storeu_ps(ctx->output + (long)(ox + 3) * out_c + oc + 0, a3l);
                _mm256_storeu_ps(ctx->output + (long)(ox + 3) * out_c + oc + 8, a3h);
            } else {
                for (int i = 0; i < rem; i++) {
                    int p_ox = ox + i;
                    int p_ix = p_ox * sx - pl;
                    const float* weights = weights_base;
                    __m256 a0l = bias0, a0h = bias1;
                    for (int kx = 0; kx < kw; kx++) {
                        int tap_ix = p_ix + kx * dx;
                        if ((unsigned)tap_ix >= (unsigned)w) {
                            weights += (long)c * 16;
                            continue;
                        }
                        const float* s0 = ctx->input + (long)tap_ix * c;
                        for (int ic = 0; ic < c; ic++, weights += 16) {
                            const __m256 wl = _mm256_loadu_ps(weights + 0);
                            const __m256 wh = _mm256_loadu_ps(weights + 8);
                            const __m256 x0 = _mm256_set1_ps(s0[ic]);
                            a0l = _mm256_fmadd_ps(x0, wl, a0l);
                            a0h = _mm256_fmadd_ps(x0, wh, a0h);
                        }
                    }
                    VX_AVX2_DIRECT_RELU(a0l); VX_AVX2_DIRECT_RELU(a0h);
                    _mm256_storeu_ps(ctx->output + (long)p_ox * out_c + oc + 0, a0l);
                    _mm256_storeu_ps(ctx->output + (long)p_ox * out_c + oc + 8, a0h);
                }
            }
        }
    }
#undef VX_AVX2_DIRECT_RELU
}
#endif

#if defined(VX_CONV_USE_NEON)
typedef struct {
    const float* input;
    float* output;
    const float* packed_weights;
    const float* bias;
    int w, c, ow, out_c, kw, sx, pl, dx, relu;
    int interior_lo;
    long safe_interior_hi;
} VxConv1DDirectNeonContext;

static void vx_conv1d_direct_neon_worker_oc(void* opaque, int begin, int end) {
    const VxConv1DDirectNeonContext* ctx = (const VxConv1DDirectNeonContext*)opaque;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int kw = ctx->kw;
    const int sx = ctx->sx;
    const int pl = ctx->pl;
    const int dx = ctx->dx;
    const int w = ctx->w;
    const int ow = ctx->ow;
    const int relu = ctx->relu;
    const int is_stride1 = (sx == 1);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
#define VX_NEON_DIRECT_RELU(v) do { \
    if (relu) { \
        (v) = vmaxq_f32((v), zero); \
        if (relu >= 2) (v) = vminq_f32((v), six); \
    } \
} while (0)

    for (int b = begin; b < end; b++) {
        int oc = b * 16;
        const float* weights_base = ctx->packed_weights + ((long)b * kw) * c * 16;
        const float32x4_t bias0 = ctx->bias ? vld1q_f32(ctx->bias + oc + 0) : zero;
        const float32x4_t bias1 = ctx->bias ? vld1q_f32(ctx->bias + oc + 4) : zero;
        const float32x4_t bias2 = ctx->bias ? vld1q_f32(ctx->bias + oc + 8) : zero;
        const float32x4_t bias3 = ctx->bias ? vld1q_f32(ctx->bias + oc + 12) : zero;

        int ox = 0;
        for (; ox < ctx->interior_lo && ox < ow; ox++) {
            int p_ix = ox * sx - pl;
            const float* weights = weights_base;
            float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
            for (int kx = 0; kx < kw; kx++) {
                int tap_ix = p_ix + kx * dx;
                if ((unsigned)tap_ix >= (unsigned)w) {
                    weights += (long)c * 16;
                    continue;
                }
                const float* s0 = ctx->input + (long)tap_ix * c;
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights + 0);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    const float x0 = s0[ic];
                    a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                    a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                    a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                    a0_3 = vmlaq_n_f32(a0_3, w3, x0);
                }
            }
            VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1);
            VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 0, a0_0);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 4, a0_1);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 8, a0_2);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 12, a0_3);
        }

        long safe_hi = ctx->safe_interior_hi < ow ? ctx->safe_interior_hi : ow;
        if (is_stride1) {
            for (; ox + 4 <= safe_hi; ox += 4) {
                float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
                float32x4_t a1_0 = bias0, a1_1 = bias1, a1_2 = bias2, a1_3 = bias3;
                float32x4_t a2_0 = bias0, a2_1 = bias1, a2_2 = bias2, a2_3 = bias3;
                float32x4_t a3_0 = bias0, a3_1 = bias1, a3_2 = bias2, a3_3 = bias3;

                const float* weights = weights_base;
                int ix0 = ox * sx - pl;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = ctx->input + ((long)ix0 + kx * dx) * c;
                    const float* s1 = s0 + c;
                    const float* s2 = s0 + 2 * c;
                    const float* s3 = s0 + 3 * c;

                    int ic = 0;
                    for (; ic + 2 <= c; ic += 2, weights += 32) {
                        const float32x4_t w0 = vld1q_f32(weights + 0);
                        const float32x4_t w1 = vld1q_f32(weights + 4);
                        const float32x4_t w2 = vld1q_f32(weights + 8);
                        const float32x4_t w3 = vld1q_f32(weights + 12);
                        const float32x4_t w4 = vld1q_f32(weights + 16);
                        const float32x4_t w5 = vld1q_f32(weights + 20);
                        const float32x4_t w6 = vld1q_f32(weights + 24);
                        const float32x4_t w7 = vld1q_f32(weights + 28);

                        const float x0_0 = s0[ic + 0];
                        const float x0_1 = s0[ic + 1];
                        a0_0 = vmlaq_n_f32(a0_0, w0, x0_0);
                        a0_1 = vmlaq_n_f32(a0_1, w1, x0_0);
                        a0_2 = vmlaq_n_f32(a0_2, w2, x0_0);
                        a0_3 = vmlaq_n_f32(a0_3, w3, x0_0);
                        a0_0 = vmlaq_n_f32(a0_0, w4, x0_1);
                        a0_1 = vmlaq_n_f32(a0_1, w5, x0_1);
                        a0_2 = vmlaq_n_f32(a0_2, w6, x0_1);
                        a0_3 = vmlaq_n_f32(a0_3, w7, x0_1);

                        const float x1_0 = s1[ic + 0];
                        const float x1_1 = s1[ic + 1];
                        a1_0 = vmlaq_n_f32(a1_0, w0, x1_0);
                        a1_1 = vmlaq_n_f32(a1_1, w1, x1_0);
                        a1_2 = vmlaq_n_f32(a1_2, w2, x1_0);
                        a1_3 = vmlaq_n_f32(a1_3, w3, x1_0);
                        a1_0 = vmlaq_n_f32(a1_0, w4, x1_1);
                        a1_1 = vmlaq_n_f32(a1_1, w5, x1_1);
                        a1_2 = vmlaq_n_f32(a1_2, w6, x1_1);
                        a1_3 = vmlaq_n_f32(a1_3, w7, x1_1);

                        const float x2_0 = s2[ic + 0];
                        const float x2_1 = s2[ic + 1];
                        a2_0 = vmlaq_n_f32(a2_0, w0, x2_0);
                        a2_1 = vmlaq_n_f32(a2_1, w1, x2_0);
                        a2_2 = vmlaq_n_f32(a2_2, w2, x2_0);
                        a2_3 = vmlaq_n_f32(a2_3, w3, x2_0);
                        a2_0 = vmlaq_n_f32(a2_0, w4, x2_1);
                        a2_1 = vmlaq_n_f32(a2_1, w5, x2_1);
                        a2_2 = vmlaq_n_f32(a2_2, w6, x2_1);
                        a2_3 = vmlaq_n_f32(a2_3, w7, x2_1);

                        const float x3_0 = s3[ic + 0];
                        const float x3_1 = s3[ic + 1];
                        a3_0 = vmlaq_n_f32(a3_0, w0, x3_0);
                        a3_1 = vmlaq_n_f32(a3_1, w1, x3_0);
                        a3_2 = vmlaq_n_f32(a3_2, w2, x3_0);
                        a3_3 = vmlaq_n_f32(a3_3, w3, x3_0);
                        a3_0 = vmlaq_n_f32(a3_0, w4, x3_1);
                        a3_1 = vmlaq_n_f32(a3_1, w5, x3_1);
                        a3_2 = vmlaq_n_f32(a3_2, w6, x3_1);
                        a3_3 = vmlaq_n_f32(a3_3, w7, x3_1);
                    }
                    for (; ic < c; ic++, weights += 16) {
                        const float32x4_t w0 = vld1q_f32(weights + 0);
                        const float32x4_t w1 = vld1q_f32(weights + 4);
                        const float32x4_t w2 = vld1q_f32(weights + 8);
                        const float32x4_t w3 = vld1q_f32(weights + 12);

                        const float x0 = s0[ic];
                        a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                        a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                        a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                        a0_3 = vmlaq_n_f32(a0_3, w3, x0);

                        const float x1 = s1[ic];
                        a1_0 = vmlaq_n_f32(a1_0, w0, x1);
                        a1_1 = vmlaq_n_f32(a1_1, w1, x1);
                        a1_2 = vmlaq_n_f32(a1_2, w2, x1);
                        a1_3 = vmlaq_n_f32(a1_3, w3, x1);

                        const float x2 = s2[ic];
                        a2_0 = vmlaq_n_f32(a2_0, w0, x2);
                        a2_1 = vmlaq_n_f32(a2_1, w1, x2);
                        a2_2 = vmlaq_n_f32(a2_2, w2, x2);
                        a2_3 = vmlaq_n_f32(a2_3, w3, x2);

                        const float x3 = s3[ic];
                        a3_0 = vmlaq_n_f32(a3_0, w0, x3);
                        a3_1 = vmlaq_n_f32(a3_1, w1, x3);
                        a3_2 = vmlaq_n_f32(a3_2, w2, x3);
                        a3_3 = vmlaq_n_f32(a3_3, w3, x3);
                    }
                }
                VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1); VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
                VX_NEON_DIRECT_RELU(a1_0); VX_NEON_DIRECT_RELU(a1_1); VX_NEON_DIRECT_RELU(a1_2); VX_NEON_DIRECT_RELU(a1_3);
                VX_NEON_DIRECT_RELU(a2_0); VX_NEON_DIRECT_RELU(a2_1); VX_NEON_DIRECT_RELU(a2_2); VX_NEON_DIRECT_RELU(a2_3);
                VX_NEON_DIRECT_RELU(a3_0); VX_NEON_DIRECT_RELU(a3_1); VX_NEON_DIRECT_RELU(a3_2); VX_NEON_DIRECT_RELU(a3_3);

                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 0, a0_0);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 4, a0_1);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 8, a0_2);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 12, a0_3);

                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 0, a1_0);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 4, a1_1);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 8, a1_2);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 12, a1_3);

                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 0, a2_0);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 4, a2_1);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 8, a2_2);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 12, a2_3);

                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 0, a3_0);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 4, a3_1);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 8, a3_2);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 12, a3_3);
            }
        }
        for (; ox < ow; ox++) {
            int p_ix = ox * sx - pl;
            const float* weights = weights_base;
            float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
            for (int kx = 0; kx < kw; kx++) {
                int tap_ix = p_ix + kx * dx;
                if ((unsigned)tap_ix >= (unsigned)w) {
                    weights += (long)c * 16;
                    continue;
                }
                const float* s0 = ctx->input + (long)tap_ix * c;
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights + 0);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    const float x0 = s0[ic];
                    a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                    a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                    a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                    a0_3 = vmlaq_n_f32(a0_3, w3, x0);
                }
            }
            VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1);
            VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 0, a0_0);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 4, a0_1);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 8, a0_2);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 12, a0_3);
        }
    }
#undef VX_NEON_DIRECT_RELU
}

static void vx_conv1d_direct_neon_worker_ox(void* opaque, int begin, int end) {
    const VxConv1DDirectNeonContext* ctx = (const VxConv1DDirectNeonContext*)opaque;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int kw = ctx->kw;
    const int sx = ctx->sx;
    const int pl = ctx->pl;
    const int dx = ctx->dx;
    const int w = ctx->w;
    const int relu = ctx->relu;
    const int is_stride1 = (sx == 1);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
#define VX_NEON_DIRECT_RELU(v) do { \
    if (relu) { \
        (v) = vmaxq_f32((v), zero); \
        if (relu >= 2) (v) = vminq_f32((v), six); \
    } \
} while (0)

    for (int oc = 0; oc < out_c; oc += 16) {
        const float* weights_base = ctx->packed_weights + ((long)(oc >> 4) * kw) * c * 16;
        const float32x4_t bias0 = ctx->bias ? vld1q_f32(ctx->bias + oc + 0) : zero;
        const float32x4_t bias1 = ctx->bias ? vld1q_f32(ctx->bias + oc + 4) : zero;
        const float32x4_t bias2 = ctx->bias ? vld1q_f32(ctx->bias + oc + 8) : zero;
        const float32x4_t bias3 = ctx->bias ? vld1q_f32(ctx->bias + oc + 12) : zero;

        int ox = begin;
        for (; ox < ctx->interior_lo && ox < end; ox++) {
            int p_ix = ox * sx - pl;
            const float* weights = weights_base;
            float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
            for (int kx = 0; kx < kw; kx++) {
                int tap_ix = p_ix + kx * dx;
                if ((unsigned)tap_ix >= (unsigned)w) {
                    weights += (long)c * 16;
                    continue;
                }
                const float* s0 = ctx->input + (long)tap_ix * c;
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights + 0);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    const float x0 = s0[ic];
                    a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                    a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                    a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                    a0_3 = vmlaq_n_f32(a0_3, w3, x0);
                }
            }
            VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1);
            VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 0, a0_0);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 4, a0_1);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 8, a0_2);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 12, a0_3);
        }

        long safe_hi = ctx->safe_interior_hi < end ? ctx->safe_interior_hi : end;
        if (is_stride1) {
            for (; ox + 4 <= safe_hi; ox += 4) {
                float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
                float32x4_t a1_0 = bias0, a1_1 = bias1, a1_2 = bias2, a1_3 = bias3;
                float32x4_t a2_0 = bias0, a2_1 = bias1, a2_2 = bias2, a2_3 = bias3;
                float32x4_t a3_0 = bias0, a3_1 = bias1, a3_2 = bias2, a3_3 = bias3;

                const float* weights = weights_base;
                int ix0 = ox * sx - pl;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = ctx->input + ((long)ix0 + kx * dx) * c;
                    const float* s1 = s0 + c;
                    const float* s2 = s0 + 2 * c;
                    const float* s3 = s0 + 3 * c;

                    int ic = 0;
                    for (; ic + 2 <= c; ic += 2, weights += 32) {
                        const float32x4_t w0 = vld1q_f32(weights + 0);
                        const float32x4_t w1 = vld1q_f32(weights + 4);
                        const float32x4_t w2 = vld1q_f32(weights + 8);
                        const float32x4_t w3 = vld1q_f32(weights + 12);
                        const float32x4_t w4 = vld1q_f32(weights + 16);
                        const float32x4_t w5 = vld1q_f32(weights + 20);
                        const float32x4_t w6 = vld1q_f32(weights + 24);
                        const float32x4_t w7 = vld1q_f32(weights + 28);

                        const float x0_0 = s0[ic + 0];
                        const float x0_1 = s0[ic + 1];
                        a0_0 = vmlaq_n_f32(a0_0, w0, x0_0);
                        a0_1 = vmlaq_n_f32(a0_1, w1, x0_0);
                        a0_2 = vmlaq_n_f32(a0_2, w2, x0_0);
                        a0_3 = vmlaq_n_f32(a0_3, w3, x0_0);
                        a0_0 = vmlaq_n_f32(a0_0, w4, x0_1);
                        a0_1 = vmlaq_n_f32(a0_1, w5, x0_1);
                        a0_2 = vmlaq_n_f32(a0_2, w6, x0_1);
                        a0_3 = vmlaq_n_f32(a0_3, w7, x0_1);

                        const float x1_0 = s1[ic + 0];
                        const float x1_1 = s1[ic + 1];
                        a1_0 = vmlaq_n_f32(a1_0, w0, x1_0);
                        a1_1 = vmlaq_n_f32(a1_1, w1, x1_0);
                        a1_2 = vmlaq_n_f32(a1_2, w2, x1_0);
                        a1_3 = vmlaq_n_f32(a1_3, w3, x1_0);
                        a1_0 = vmlaq_n_f32(a1_0, w4, x1_1);
                        a1_1 = vmlaq_n_f32(a1_1, w5, x1_1);
                        a1_2 = vmlaq_n_f32(a1_2, w6, x1_1);
                        a1_3 = vmlaq_n_f32(a1_3, w7, x1_1);

                        const float x2_0 = s2[ic + 0];
                        const float x2_1 = s2[ic + 1];
                        a2_0 = vmlaq_n_f32(a2_0, w0, x2_0);
                        a2_1 = vmlaq_n_f32(a2_1, w1, x2_0);
                        a2_2 = vmlaq_n_f32(a2_2, w2, x2_0);
                        a2_3 = vmlaq_n_f32(a2_3, w3, x2_0);
                        a2_0 = vmlaq_n_f32(a2_0, w4, x2_1);
                        a2_1 = vmlaq_n_f32(a2_1, w5, x2_1);
                        a2_2 = vmlaq_n_f32(a2_2, w6, x2_1);
                        a2_3 = vmlaq_n_f32(a2_3, w7, x2_1);

                        const float x3_0 = s3[ic + 0];
                        const float x3_1 = s3[ic + 1];
                        a3_0 = vmlaq_n_f32(a3_0, w0, x3_0);
                        a3_1 = vmlaq_n_f32(a3_1, w1, x3_0);
                        a3_2 = vmlaq_n_f32(a3_2, w2, x3_0);
                        a3_3 = vmlaq_n_f32(a3_3, w3, x3_0);
                        a3_0 = vmlaq_n_f32(a3_0, w4, x3_1);
                        a3_1 = vmlaq_n_f32(a3_1, w5, x3_1);
                        a3_2 = vmlaq_n_f32(a3_2, w6, x3_1);
                        a3_3 = vmlaq_n_f32(a3_3, w7, x3_1);
                    }
                    for (; ic < c; ic++, weights += 16) {
                        const float32x4_t w0 = vld1q_f32(weights + 0);
                        const float32x4_t w1 = vld1q_f32(weights + 4);
                        const float32x4_t w2 = vld1q_f32(weights + 8);
                        const float32x4_t w3 = vld1q_f32(weights + 12);

                        const float x0 = s0[ic];
                        a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                        a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                        a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                        a0_3 = vmlaq_n_f32(a0_3, w3, x0);

                        const float x1 = s1[ic];
                        a1_0 = vmlaq_n_f32(a1_0, w0, x1);
                        a1_1 = vmlaq_n_f32(a1_1, w1, x1);
                        a1_2 = vmlaq_n_f32(a1_2, w2, x1);
                        a1_3 = vmlaq_n_f32(a1_3, w3, x1);

                        const float x2 = s2[ic];
                        a2_0 = vmlaq_n_f32(a2_0, w0, x2);
                        a2_1 = vmlaq_n_f32(a2_1, w1, x2);
                        a2_2 = vmlaq_n_f32(a2_2, w2, x2);
                        a2_3 = vmlaq_n_f32(a2_3, w3, x2);

                        const float x3 = s3[ic];
                        a3_0 = vmlaq_n_f32(a3_0, w0, x3);
                        a3_1 = vmlaq_n_f32(a3_1, w1, x3);
                        a3_2 = vmlaq_n_f32(a3_2, w2, x3);
                        a3_3 = vmlaq_n_f32(a3_3, w3, x3);
                    }
                }
                VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1); VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
                VX_NEON_DIRECT_RELU(a1_0); VX_NEON_DIRECT_RELU(a1_1); VX_NEON_DIRECT_RELU(a1_2); VX_NEON_DIRECT_RELU(a1_3);
                VX_NEON_DIRECT_RELU(a2_0); VX_NEON_DIRECT_RELU(a2_1); VX_NEON_DIRECT_RELU(a2_2); VX_NEON_DIRECT_RELU(a2_3);
                VX_NEON_DIRECT_RELU(a3_0); VX_NEON_DIRECT_RELU(a3_1); VX_NEON_DIRECT_RELU(a3_2); VX_NEON_DIRECT_RELU(a3_3);

                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 0, a0_0);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 4, a0_1);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 8, a0_2);
                vst1q_f32(ctx->output + (long)(ox + 0) * out_c + oc + 12, a0_3);

                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 0, a1_0);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 4, a1_1);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 8, a1_2);
                vst1q_f32(ctx->output + (long)(ox + 1) * out_c + oc + 12, a1_3);

                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 0, a2_0);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 4, a2_1);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 8, a2_2);
                vst1q_f32(ctx->output + (long)(ox + 2) * out_c + oc + 12, a2_3);

                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 0, a3_0);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 4, a3_1);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 8, a3_2);
                vst1q_f32(ctx->output + (long)(ox + 3) * out_c + oc + 12, a3_3);
            }
        }
        for (; ox < end; ox++) {
            int p_ix = ox * sx - pl;
            const float* weights = weights_base;
            float32x4_t a0_0 = bias0, a0_1 = bias1, a0_2 = bias2, a0_3 = bias3;
            for (int kx = 0; kx < kw; kx++) {
                int tap_ix = p_ix + kx * dx;
                if ((unsigned)tap_ix >= (unsigned)w) {
                    weights += (long)c * 16;
                    continue;
                }
                const float* s0 = ctx->input + (long)tap_ix * c;
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights + 0);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    const float x0 = s0[ic];
                    a0_0 = vmlaq_n_f32(a0_0, w0, x0);
                    a0_1 = vmlaq_n_f32(a0_1, w1, x0);
                    a0_2 = vmlaq_n_f32(a0_2, w2, x0);
                    a0_3 = vmlaq_n_f32(a0_3, w3, x0);
                }
            }
            VX_NEON_DIRECT_RELU(a0_0); VX_NEON_DIRECT_RELU(a0_1);
            VX_NEON_DIRECT_RELU(a0_2); VX_NEON_DIRECT_RELU(a0_3);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 0, a0_0);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 4, a0_1);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 8, a0_2);
            vst1q_f32(ctx->output + (long)ox * out_c + oc + 12, a0_3);
        }
    }
#undef VX_NEON_DIRECT_RELU
}

typedef struct {
    float* output;
    const float* const* indir;
    const float* packed;
    const float* bias;
    int c;
    int out_c;
    int ks;
    int relu;
    long pixels;
} VxF32IgemmSixPixelNeonContext;

static void vx_f32_igemm_six_pixel_neon_worker_oc(void* opaque, int begin, int end) {
    const VxF32IgemmSixPixelNeonContext* ctx = (const VxF32IgemmSixPixelNeonContext*)opaque;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int ks = ctx->ks;
    const int relu = ctx->relu;
    const long pixels = ctx->pixels;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
#define VX_NEON_IGEMM_RELU(v) do { \
    if (relu) { \
        (v) = vmaxq_f32((v), zero); \
        if (relu >= 2) (v) = vminq_f32((v), six); \
    } \
} while (0)

    for (int b = begin; b < end; b++) {
        int oc = b * 16;
        const float* weights_base = ctx->packed + ((long)b * ks) * c * 16;
        long pixel = 0;
        for (; pixel + 6 <= pixels; pixel += 6) {
            float* destinations[6] = {
                ctx->output + (pixel + 0) * out_c + oc,
                ctx->output + (pixel + 1) * out_c + oc,
                ctx->output + (pixel + 2) * out_c + oc,
                ctx->output + (pixel + 3) * out_c + oc,
                ctx->output + (pixel + 4) * out_c + oc,
                ctx->output + (pixel + 5) * out_c + oc,
            };
            const float* const* rows[6] = {
                ctx->indir + (pixel + 0) * ks,
                ctx->indir + (pixel + 1) * ks,
                ctx->indir + (pixel + 2) * ks,
                ctx->indir + (pixel + 3) * ks,
                ctx->indir + (pixel + 4) * ks,
                ctx->indir + (pixel + 5) * ks,
            };
            const float* weights = weights_base;
            float32x4_t accumulators[6][4];
            for (int op = 0; op < 6; op++) {
                accumulators[op][0] = ctx->bias ? vld1q_f32(ctx->bias + oc) : zero;
                accumulators[op][1] = ctx->bias ? vld1q_f32(ctx->bias + oc + 4) : zero;
                accumulators[op][2] = ctx->bias ? vld1q_f32(ctx->bias + oc + 8) : zero;
                accumulators[op][3] = ctx->bias ? vld1q_f32(ctx->bias + oc + 12) : zero;
            }
            for (int tap = 0; tap < ks; tap++) {
                const float* sources[6] = {
                    rows[0][tap], rows[1][tap], rows[2][tap],
                    rows[3][tap], rows[4][tap], rows[5][tap],
                };
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    for (int op = 0; op < 6; op++) {
                        const float val = sources[op][ic];
                        accumulators[op][0] = vmlaq_n_f32(accumulators[op][0], w0, val);
                        accumulators[op][1] = vmlaq_n_f32(accumulators[op][1], w1, val);
                        accumulators[op][2] = vmlaq_n_f32(accumulators[op][2], w2, val);
                        accumulators[op][3] = vmlaq_n_f32(accumulators[op][3], w3, val);
                    }
                }
            }
            for (int op = 0; op < 6; op++) {
                VX_NEON_IGEMM_RELU(accumulators[op][0]);
                VX_NEON_IGEMM_RELU(accumulators[op][1]);
                VX_NEON_IGEMM_RELU(accumulators[op][2]);
                VX_NEON_IGEMM_RELU(accumulators[op][3]);
                vst1q_f32(destinations[op] + 0, accumulators[op][0]);
                vst1q_f32(destinations[op] + 4, accumulators[op][1]);
                vst1q_f32(destinations[op] + 8, accumulators[op][2]);
                vst1q_f32(destinations[op] + 12, accumulators[op][3]);
            }
        }
        for (; pixel < pixels; pixel++) {
            float* destination = ctx->output + pixel * out_c + oc;
            const float* const* row = ctx->indir + pixel * ks;
            const float* weights = weights_base;
            float32x4_t acc0 = ctx->bias ? vld1q_f32(ctx->bias + oc) : zero;
            float32x4_t acc1 = ctx->bias ? vld1q_f32(ctx->bias + oc + 4) : zero;
            float32x4_t acc2 = ctx->bias ? vld1q_f32(ctx->bias + oc + 8) : zero;
            float32x4_t acc3 = ctx->bias ? vld1q_f32(ctx->bias + oc + 12) : zero;
            for (int tap = 0; tap < ks; tap++) {
                const float* source = row[tap];
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float val = source[ic];
                    acc0 = vmlaq_n_f32(acc0, vld1q_f32(weights), val);
                    acc1 = vmlaq_n_f32(acc1, vld1q_f32(weights + 4), val);
                    acc2 = vmlaq_n_f32(acc2, vld1q_f32(weights + 8), val);
                    acc3 = vmlaq_n_f32(acc3, vld1q_f32(weights + 12), val);
                }
            }
            VX_NEON_IGEMM_RELU(acc0); VX_NEON_IGEMM_RELU(acc1);
            VX_NEON_IGEMM_RELU(acc2); VX_NEON_IGEMM_RELU(acc3);
            vst1q_f32(destination + 0, acc0);
            vst1q_f32(destination + 4, acc1);
            vst1q_f32(destination + 8, acc2);
            vst1q_f32(destination + 12, acc3);
        }
    }
#undef VX_NEON_IGEMM_RELU
}

static void vx_f32_igemm_six_pixel_neon_worker_pixels(void* opaque, int begin, int end) {
    const VxF32IgemmSixPixelNeonContext* ctx = (const VxF32IgemmSixPixelNeonContext*)opaque;
    const int c = ctx->c;
    const int out_c = ctx->out_c;
    const int ks = ctx->ks;
    const int relu = ctx->relu;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
#define VX_NEON_IGEMM_RELU(v) do { \
    if (relu) { \
        (v) = vmaxq_f32((v), zero); \
        if (relu >= 2) (v) = vminq_f32((v), six); \
    } \
} while (0)

    for (int block = begin; block < end; block++) {
        long pixel = (long)block * 6;
        float* destinations[6] = {
            ctx->output + (pixel + 0) * out_c,
            ctx->output + (pixel + 1) * out_c,
            ctx->output + (pixel + 2) * out_c,
            ctx->output + (pixel + 3) * out_c,
            ctx->output + (pixel + 4) * out_c,
            ctx->output + (pixel + 5) * out_c,
        };
        const float* const* rows[6] = {
            ctx->indir + (pixel + 0) * ks,
            ctx->indir + (pixel + 1) * ks,
            ctx->indir + (pixel + 2) * ks,
            ctx->indir + (pixel + 3) * ks,
            ctx->indir + (pixel + 4) * ks,
            ctx->indir + (pixel + 5) * ks,
        };
        for (int oc = 0; oc < out_c; oc += 16) {
            const float* weights = ctx->packed + ((long)(oc >> 4) * ks) * c * 16;
            float32x4_t accumulators[6][4];
            for (int op = 0; op < 6; op++) {
                accumulators[op][0] = ctx->bias ? vld1q_f32(ctx->bias + oc) : zero;
                accumulators[op][1] = ctx->bias ? vld1q_f32(ctx->bias + oc + 4) : zero;
                accumulators[op][2] = ctx->bias ? vld1q_f32(ctx->bias + oc + 8) : zero;
                accumulators[op][3] = ctx->bias ? vld1q_f32(ctx->bias + oc + 12) : zero;
            }
            for (int tap = 0; tap < ks; tap++) {
                const float* sources[6] = {
                    rows[0][tap], rows[1][tap], rows[2][tap],
                    rows[3][tap], rows[4][tap], rows[5][tap],
                };
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const float32x4_t w0 = vld1q_f32(weights);
                    const float32x4_t w1 = vld1q_f32(weights + 4);
                    const float32x4_t w2 = vld1q_f32(weights + 8);
                    const float32x4_t w3 = vld1q_f32(weights + 12);
                    for (int op = 0; op < 6; op++) {
                        const float val = sources[op][ic];
                        accumulators[op][0] = vmlaq_n_f32(accumulators[op][0], w0, val);
                        accumulators[op][1] = vmlaq_n_f32(accumulators[op][1], w1, val);
                        accumulators[op][2] = vmlaq_n_f32(accumulators[op][2], w2, val);
                        accumulators[op][3] = vmlaq_n_f32(accumulators[op][3], w3, val);
                    }
                }
            }
            for (int op = 0; op < 6; op++) {
                VX_NEON_IGEMM_RELU(accumulators[op][0]);
                VX_NEON_IGEMM_RELU(accumulators[op][1]);
                VX_NEON_IGEMM_RELU(accumulators[op][2]);
                VX_NEON_IGEMM_RELU(accumulators[op][3]);
                vst1q_f32(destinations[op] + oc + 0, accumulators[op][0]);
                vst1q_f32(destinations[op] + oc + 4, accumulators[op][1]);
                vst1q_f32(destinations[op] + oc + 8, accumulators[op][2]);
                vst1q_f32(destinations[op] + oc + 12, accumulators[op][3]);
            }
        }
    }
#undef VX_NEON_IGEMM_RELU
}
#endif

#if defined(__AVX2__) || defined(VX_CONV_USE_NEON) || defined(VX_CONV_USE_WASM_SIMD128)
static int vx_conv_weights_are_finite(const float* weights, size_t count) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    for (; i + 8 <= count; i += 8) {
        const __m256i bits = _mm256_loadu_si256((const __m256i*)(weights + i));
        const __m256i exceptional = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent), exponent);
        if (!_mm256_testz_si256(exceptional, exceptional)) return 0;
    }
#elif defined(VX_CONV_USE_NEON)
    const uint32x4_t exponent = vdupq_n_u32(0x7f800000u);
    for (; i + 4 <= count; i += 4) {
        const uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(weights + i));
        const uint32x4_t exceptional = vceqq_u32(vandq_u32(bits, exponent), exponent);
        const uint64x2_t wide = vreinterpretq_u64_u32(exceptional);
        if (vgetq_lane_u64(wide, 0) || vgetq_lane_u64(wide, 1)) return 0;
    }
#elif defined(VX_CONV_USE_WASM_SIMD128)
    const v128_t exponent = wasm_i32x4_splat(0x7f800000);
    for (; i + 4 <= count; i += 4) {
        const v128_t bits = wasm_v128_load(weights + i);
        if (wasm_v128_any_true(wasm_i32x4_eq(wasm_v128_and(bits, exponent), exponent))) return 0;
    }
#endif
    for (; i < count; i++) {
        uint32_t bits;
        memcpy(&bits, weights + i, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) return 0;
    }
    return 1;
}

/* A +0 padding buffer can turn a legitimate -0 sum into +0 even with finite
 * weights. Only a zero result can expose that difference. Re-evaluate only
 * zero boundary lanes, preserving the indirect tile's accumulation arithmetic.
 * Nonzero lanes are never overwritten and no temporary allocation is needed. */
#if defined(__clang__)
__attribute__((noinline, minsize))
#elif defined(__GNUC__)
__attribute__((noinline, optimize("Os")))
#endif
static void vx_conv2d_restore_padded_zeros(
        const float* input, float* output, const float* weights, const float* bias,
        int n, int h, int w, int c, int oh, int ow, int out_c, int kh, int kw,
        int sy, int sx, const int* pads, int dy, int dx, int relu) {
    /* SIMD ReLU already maps either zero sign to +0. Only scalar channel
     * tails can retain -0 after that epilogue. Keep this repair out of the
     * hot tile and avoid expanding its scalar loops into another SIMD tile. */
#if defined(__AVX2__)
    const int first_channel = relu ? (out_c / 8) * 8 : 0;
#else
    const int first_channel = relu ? out_c : 0;
#endif
    if (first_channel == out_c) return;
    for (int b = 0; b < n; b++) {
        const float* batch = input + (long)b * h * w * c;
        for (int oy = 0; oy < oh; oy++) {
            const int iy0 = oy * sy - pads[0];
            for (int ox = 0; ox < ow; ox++) {
                const int ix0 = ox * sx - pads[1];
                if (iy0 >= 0 && (int64_t)iy0 + (int64_t)(kh - 1) * dy < h &&
                    ix0 >= 0 && (int64_t)ix0 + (int64_t)(kw - 1) * dx < w) continue;
                const long p = ((long)b * oh + oy) * ow + ox;
                float* pixel = output + p * out_c;
                for (int oc = first_channel; oc < out_c; oc++) {
                    if (pixel[oc] != 0.0f) continue;
                    float sum = bias ? bias[oc] : 0.0f;
#if defined(__AVX2__)
                    /* All indirect tiles use FMA through the eight-column
                     * tail, followed by the same scalar remainder. */
                    const int vector_lane = oc < (out_c / 8) * 8;
                    __m128 vector_sum = _mm_set_ss(sum);
#elif defined(VX_CONV_USE_NEON)
                    const int vector_lane = 1;
                    float32x4_t vector_sum = vdupq_n_f32(sum);
#else
                    const int vector_lane = 1;
#endif
                    for (int ky = 0; ky < kh; ky++) {
                        const int iy = iy0 + ky * dy;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            const int ix = ix0 + kx * dx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            const float* src = batch + ((long)iy * w + ix) * c;
                            const float* ww = weights + (((long)ky * kw + kx) * c) * out_c + oc;
                            for (int ic = 0; ic < c; ic++) {
                                const float value = src[ic], weight = ww[(long)ic * out_c];
#if defined(__AVX2__)
                                if (vector_lane)
                                    vector_sum = _mm_fmadd_ss(_mm_set_ss(value), _mm_set_ss(weight), vector_sum);
                                else sum += value * weight;
#elif defined(VX_CONV_USE_NEON)
                                vector_sum = vmlaq_n_f32(vector_sum, vdupq_n_f32(weight), value);
#else
                                sum += value * weight;
#endif
                            }
                        }
                    }
#if defined(__AVX2__)
                    if (vector_lane) sum = _mm_cvtss_f32(vector_sum);
#elif defined(VX_CONV_USE_NEON)
                    sum = vgetq_lane_f32(vector_sum, 0);
#endif
                    sum = vx_relu6_apply(sum, relu);
                    /* This repair only changes a zero's sign. Keep every
                     * nonzero result from the original tile untouched. */
                    if (sum == 0.0f) pixel[oc] = relu && vector_lane ? 0.0f : sum;
                }
            }
        }
    }
}
#endif

int vx_conv2d_spatial_igemm_f32(int node_idx,
                                     const float* input, float* output,
                                     const float* wgt, const float* bias,
                                     int n, int h, int w, int c,
                                     int oh, int ow, int out_c,
                                     int kh, int kw,
                                     int sy, int sx, const int* pads,
                                     int dy, int dx, int relu) {
#if defined(__AVX2__) || defined(VX_CONV_USE_NEON) || defined(VX_CONV_USE_WASM_SIMD128)
    /* Indirection represents absent taps with a zero buffer. Zero times a
     * nonfinite weight is not an absent tap: it contaminates border outputs.
     * Keep the existing bounds-checking route for those weights. Actual input
     * zeros still participate in every multiplication. Check only before the
     * indirect routes below: direct 1D kernels already skip absent taps. */
    const int padded_windows = pads[0] > 0 || pads[1] > 0 ||
        (int64_t)(oh - 1) * sy - pads[0] + (int64_t)(kh - 1) * dy >= h ||
        (int64_t)(ow - 1) * sx - pads[1] + (int64_t)(kw - 1) * dx >= w;
#endif
#if defined(VX_CONV_USE_NEON) && !defined(__AVX2__)
    if (n == 1 && h == 1 && kh == 1 && oh == 1 && pads[0] == 0 && pads[2] == 0 &&
        out_c >= 16 && (out_c & 15) == 0) {
        const float* packed = vx_f32_igemm_pack_cache(node_idx, wgt, c, out_c, kw);
        if (packed) {
            const int pl = pads[1];
            const int interior_lo = sx > 0 ? (pl + sx - 1) / sx : ow;
            const int interior_hi = sx > 0 && (w - 1 + pl - (kw - 1) * dx) >= 0
                ? ((w - 1 + pl - (kw - 1) * dx) / sx) + 1 : 0;
            const long safe_interior_hi = interior_hi < ow ? interior_hi : ow;

            VxConv1DDirectNeonContext ctx = {
                input, output, packed, bias,
                w, c, ow, out_c, kw, sx, pl, dx, relu,
                interior_lo, safe_interior_hi
            };

            int threads = vx_kernels_thread_count();
            int oc_blocks = out_c / 16;
            if (threads > 1 && (long)ow * c * (int)sizeof(float) >= 262144 && ow >= threads * 32) {
                int grain = 1 + (ow - 1) / (threads * 4);
                grain = (grain + 3) & ~3;
                if (grain < 16) grain = 16;
                vx_kernels_parallel_for(ow, grain, vx_conv1d_direct_neon_worker_ox, &ctx);
            } else if (threads > 1 && oc_blocks >= threads &&
                       (threads != 2 || ow >= 64 ||
                        (uint64_t)ow * (uint32_t)c * (uint32_t)kw * (uint32_t)out_c >= 33554432ULL)) {
                /* More independent OC chunks balance larger four-thread jobs.
                 * Keep smaller work and other thread counts on the old policy.
                 * Divide the work bound first so large extents cannot overflow. */
                int grain = (threads == 4 && (uint64_t)ow * c >=
                             (16384u + (uint32_t)kw - 1u) / (uint32_t)kw)
                    ? 1 + (oc_blocks - 1) / (threads * 4)
                    : (oc_blocks + threads - 1) / threads;
                if (grain < 1) grain = 1;
                vx_kernels_parallel_for(oc_blocks, grain, vx_conv1d_direct_neon_worker_oc, &ctx);
            } else if (threads > 1 && ow >= 64) {
                int grain = 1 + (ow - 1) / (threads * 4);
                grain = (grain + 3) & ~3;
                if (grain < 16) grain = 16;
                vx_kernels_parallel_for(ow, grain, vx_conv1d_direct_neon_worker_ox, &ctx);
            } else {
                vx_conv1d_direct_neon_worker_ox(&ctx, 0, ow);
            }
            return 1;
        }
    }
    if (kh * kw <= 1 || out_c < 16 || (out_c & 15)) return 0;
    if (padded_windows && !vx_conv_weights_are_finite(
            wgt, (size_t)kh * kw * c * out_c)) return 0;
    const float** indir = vx_f32_igemm_indirection_cache(
        node_idx, input, n, h, w, c, oh, ow, kh, kw,
        sy, sx, pads, dy, dx);
    const int ks = kh * kw;
    const float* packed = vx_f32_igemm_pack_cache(
        node_idx, wgt, c, out_c, ks);
    const long pixels = (long)n * oh * ow;
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t six = vdupq_n_f32(6.0f);
    if (!indir || !packed) return 0;

    int threads = vx_kernels_thread_count();
    int six_pixel_blocks = (int)(pixels / 6);
    int oc_blocks = out_c / 16;
    VxF32IgemmSixPixelNeonContext igemm_ctx = {
        output, indir, packed, bias, c, out_c, ks, relu, pixels
    };
    if (threads > 1 && six_pixel_blocks >= threads) {
        if (six_pixel_blocks < threads * 4 && oc_blocks >= threads * 2) {
            int grain = (oc_blocks + threads - 1) / threads;
            if (grain < 1) grain = 1;
            vx_kernels_parallel_for(oc_blocks, grain, vx_f32_igemm_six_pixel_neon_worker_oc, &igemm_ctx);
        } else {
            int grain = (six_pixel_blocks + threads * 4 - 1) / (threads * 4);
            if (grain < 1) grain = 1;
            vx_kernels_parallel_for(six_pixel_blocks, grain, vx_f32_igemm_six_pixel_neon_worker_pixels, &igemm_ctx);
            for (long pixel = (long)six_pixel_blocks * 6; pixel < pixels; pixel++) {
                float* destination = output + pixel * out_c;
                const float* const* row = indir + pixel * ks;
                for (int oc = 0; oc < out_c; oc += 16) {
                    const float* weights = packed + ((long)(oc >> 4) * ks) * c * 16;
                    float32x4_t accumulator0 = bias ? vld1q_f32(bias + oc) : zero;
                    float32x4_t accumulator1 = bias ? vld1q_f32(bias + oc + 4) : zero;
                    float32x4_t accumulator2 = bias ? vld1q_f32(bias + oc + 8) : zero;
                    float32x4_t accumulator3 = bias ? vld1q_f32(bias + oc + 12) : zero;
                    for (int tap = 0; tap < ks; tap++) {
                        const float* source = row[tap];
                        for (int ic = 0; ic < c; ic++, weights += 16) {
                            const float value = source[ic];
                            accumulator0 = vmlaq_n_f32(accumulator0, vld1q_f32(weights), value);
                            accumulator1 = vmlaq_n_f32(accumulator1, vld1q_f32(weights + 4), value);
                            accumulator2 = vmlaq_n_f32(accumulator2, vld1q_f32(weights + 8), value);
                            accumulator3 = vmlaq_n_f32(accumulator3, vld1q_f32(weights + 12), value);
                        }
                    }
                    if (relu) {
                        accumulator0 = vmaxq_f32(accumulator0, zero);
                        accumulator1 = vmaxq_f32(accumulator1, zero);
                        accumulator2 = vmaxq_f32(accumulator2, zero);
                        accumulator3 = vmaxq_f32(accumulator3, zero);
                        if (relu >= 2) {
                            accumulator0 = vminq_f32(accumulator0, six);
                            accumulator1 = vminq_f32(accumulator1, six);
                            accumulator2 = vminq_f32(accumulator2, six);
                            accumulator3 = vminq_f32(accumulator3, six);
                        }
                    }
                    vst1q_f32(destination + oc, accumulator0);
                    vst1q_f32(destination + oc + 4, accumulator1);
                    vst1q_f32(destination + oc + 8, accumulator2);
                    vst1q_f32(destination + oc + 12, accumulator3);
                }
            }
        }
    } else {
        vx_f32_igemm_six_pixel_neon_worker_oc(&igemm_ctx, 0, oc_blocks);
    }
    if (padded_windows)
        vx_conv2d_restore_padded_zeros(input, output, wgt, bias,
            n, h, w, c, oh, ow, out_c, kh, kw, sy, sx, pads, dy, dx, relu);
    return 1;
#elif defined(VX_CONV_USE_WASM_SIMD128)
    const int ks = kh * kw;
    const long pixels = (long)n * oh * ow;
    if (ks <= 1) return 0;

    const v128_t zero = wasm_f32x4_splat(0.0f);
    const v128_t six = wasm_f32x4_splat(6.0f);
#define VX_WASM_IGEMM_RELU(v) do { \
    if (relu) { \
        (v) = wasm_f32x4_max((v), zero); \
        if (relu >= 2) (v) = wasm_f32x4_min((v), six); \
    } \
} while (0)
#define VX_WASM_IGEMM_RELU_SCALAR(v) do { \
    if (relu) { \
        if ((v) < 0.0f) (v) = 0.0f; \
        if (relu >= 2 && (v) > 6.0f) (v) = 6.0f; \
    } \
} while (0)

    /* High-performance direct 1D audio sequence convolution for WASM SIMD128.
     * Bypasses the 2D indirection pointer table and operates on [oc/16][kw][c][16]
     * packed weights with contiguous 64-byte vector streaming. Uses an exact 2x16
     * register tile (8 accumulators + 4 weights + 1 broadcast = 13 live v128_t),
     * completely eliminating stack spills on 16-register x86-64 WASM runtimes. */
    if (n == 1 && h == 1 && kh == 1 && oh == 1 && pads[0] == 0 && pads[2] == 0) {
        if (out_c >= 16 && (out_c & 15) == 0) {
        const float* packed = vx_f32_igemm_pack_cache(node_idx, wgt, c, out_c, kw);
        if (!packed) return 0;

        const int pl = pads[1];
        const int interior_lo = sx > 0 ? (pl + sx - 1) / sx : ow;
        const int interior_hi = sx > 0 && (w - 1 + pl - (kw - 1) * dx) >= 0
            ? ((w - 1 + pl - (kw - 1) * dx) / sx) + 1 : 0;

        const long safe_interior_hi = interior_hi < ow ? interior_hi : ow;

        for (int oc = 0; oc < out_c; oc += 16) {
            const float* weights_base = packed + ((long)(oc >> 4) * kw) * c * 16;
            const v128_t bias0 = bias ? wasm_v128_load(bias + oc + 0) : zero;
            const v128_t bias1 = bias ? wasm_v128_load(bias + oc + 4) : zero;
            const v128_t bias2 = bias ? wasm_v128_load(bias + oc + 8) : zero;
            const v128_t bias3 = bias ? wasm_v128_load(bias + oc + 12) : zero;

            long ox = 0;

            /* Leading edge pixels outside interior bounds */
            for (; ox < interior_lo && ox < ow; ox++) {
                float* dst0 = output + ox * out_c;
                const int ix0 = (int)ox * sx - pl;
                const float* weights = weights_base;
                v128_t acc0 = bias0;
                v128_t acc1 = bias1;
                v128_t acc2 = bias2;
                v128_t acc3 = bias3;
                for (int kx = 0; kx < kw; kx++) {
                    const int tap_ix = ix0 + kx * dx;
                    if ((unsigned)tap_ix >= (unsigned)w) {
                        weights += (long)c * 16;
                        continue;
                    }
                    const float* s0 = input + (long)tap_ix * c;
                    for (int ic = 0; ic < c; ic++, weights += 16) {
                        const v128_t w0 = wasm_v128_load(weights + 0);
                        const v128_t w1 = wasm_v128_load(weights + 4);
                        const v128_t w2 = wasm_v128_load(weights + 8);
                        const v128_t w3 = wasm_v128_load(weights + 12);
                        const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x0, w0));
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x0, w1));
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x0, w2));
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x0, w3));
                    }
                }
                VX_WASM_IGEMM_RELU(acc0);
                VX_WASM_IGEMM_RELU(acc1);
                VX_WASM_IGEMM_RELU(acc2);
                VX_WASM_IGEMM_RELU(acc3);
                wasm_v128_store(dst0 + oc + 0, acc0);
                wasm_v128_store(dst0 + oc + 4, acc1);
                wasm_v128_store(dst0 + oc + 8, acc2);
                wasm_v128_store(dst0 + oc + 12, acc3);
            }

            /* 3-pixel unrolled interior loop: weights_base stays pinned in L1/L2 cache! */
            const int is_stride1 = (sx == 1);
            for (; ox + 3 <= safe_interior_hi; ox += 3) {
                float* dst0 = output + (ox + 0) * out_c;
                float* dst1 = output + (ox + 1) * out_c;
                float* dst2 = output + (ox + 2) * out_c;
                const int ix0 = (int)ox * sx - pl;
                const int ix1 = (int)(ox + 1) * sx - pl;
                const int ix2 = (int)(ox + 2) * sx - pl;

                v128_t acc0_0 = bias0, acc0_1 = bias1, acc0_2 = bias2, acc0_3 = bias3;
                v128_t acc1_0 = bias0, acc1_1 = bias1, acc1_2 = bias2, acc1_3 = bias3;
                v128_t acc2_0 = bias0, acc2_1 = bias1, acc2_2 = bias2, acc2_3 = bias3;

                const float* weights = weights_base;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = input + ((long)ix0 + kx * dx) * c;
                    const float* s1 = is_stride1 ? (s0 + c) : (input + ((long)ix1 + kx * dx) * c);
                    const float* s2 = is_stride1 ? (s0 + 2 * c) : (input + ((long)ix2 + kx * dx) * c);
                    int ic = 0;
                    for (; ic + 2 <= c; ic += 2, weights += 32) {
                        const v128_t x0_0 = wasm_v128_load32_splat(s0 + ic + 0);
                        const v128_t x1_0 = wasm_v128_load32_splat(s1 + ic + 0);
                        const v128_t x2_0 = wasm_v128_load32_splat(s2 + ic + 0);

                        const v128_t w0 = wasm_v128_load(weights + 0);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0_0, w0));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1_0, w0));
                        acc2_0 = wasm_f32x4_add(acc2_0, wasm_f32x4_mul(x2_0, w0));

                        const v128_t w1 = wasm_v128_load(weights + 4);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0_0, w1));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1_0, w1));
                        acc2_1 = wasm_f32x4_add(acc2_1, wasm_f32x4_mul(x2_0, w1));

                        const v128_t w2 = wasm_v128_load(weights + 8);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0_0, w2));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1_0, w2));
                        acc2_2 = wasm_f32x4_add(acc2_2, wasm_f32x4_mul(x2_0, w2));

                        const v128_t w3 = wasm_v128_load(weights + 12);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0_0, w3));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1_0, w3));
                        acc2_3 = wasm_f32x4_add(acc2_3, wasm_f32x4_mul(x2_0, w3));

                        const v128_t x0_1 = wasm_v128_load32_splat(s0 + ic + 1);
                        const v128_t x1_1 = wasm_v128_load32_splat(s1 + ic + 1);
                        const v128_t x2_1 = wasm_v128_load32_splat(s2 + ic + 1);

                        const v128_t w4 = wasm_v128_load(weights + 16);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0_1, w4));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1_1, w4));
                        acc2_0 = wasm_f32x4_add(acc2_0, wasm_f32x4_mul(x2_1, w4));

                        const v128_t w5 = wasm_v128_load(weights + 20);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0_1, w5));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1_1, w5));
                        acc2_1 = wasm_f32x4_add(acc2_1, wasm_f32x4_mul(x2_1, w5));

                        const v128_t w6 = wasm_v128_load(weights + 24);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0_1, w6));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1_1, w6));
                        acc2_2 = wasm_f32x4_add(acc2_2, wasm_f32x4_mul(x2_1, w6));

                        const v128_t w7 = wasm_v128_load(weights + 28);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0_1, w7));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1_1, w7));
                        acc2_3 = wasm_f32x4_add(acc2_3, wasm_f32x4_mul(x2_1, w7));
                    }
                    for (; ic < c; ic++, weights += 16) {
                        const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                        const v128_t x1 = wasm_v128_load32_splat(s1 + ic);
                        const v128_t x2 = wasm_v128_load32_splat(s2 + ic);

                        const v128_t w0 = wasm_v128_load(weights + 0);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0, w0));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1, w0));
                        acc2_0 = wasm_f32x4_add(acc2_0, wasm_f32x4_mul(x2, w0));

                        const v128_t w1 = wasm_v128_load(weights + 4);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0, w1));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1, w1));
                        acc2_1 = wasm_f32x4_add(acc2_1, wasm_f32x4_mul(x2, w1));

                        const v128_t w2 = wasm_v128_load(weights + 8);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0, w2));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1, w2));
                        acc2_2 = wasm_f32x4_add(acc2_2, wasm_f32x4_mul(x2, w2));

                        const v128_t w3 = wasm_v128_load(weights + 12);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0, w3));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1, w3));
                        acc2_3 = wasm_f32x4_add(acc2_3, wasm_f32x4_mul(x2, w3));
                    }
                }

                VX_WASM_IGEMM_RELU(acc0_0);
                VX_WASM_IGEMM_RELU(acc0_1);
                VX_WASM_IGEMM_RELU(acc0_2);
                VX_WASM_IGEMM_RELU(acc0_3);
                wasm_v128_store(dst0 + oc + 0, acc0_0);
                wasm_v128_store(dst0 + oc + 4, acc0_1);
                wasm_v128_store(dst0 + oc + 8, acc0_2);
                wasm_v128_store(dst0 + oc + 12, acc0_3);

                VX_WASM_IGEMM_RELU(acc1_0);
                VX_WASM_IGEMM_RELU(acc1_1);
                VX_WASM_IGEMM_RELU(acc1_2);
                VX_WASM_IGEMM_RELU(acc1_3);
                wasm_v128_store(dst1 + oc + 0, acc1_0);
                wasm_v128_store(dst1 + oc + 4, acc1_1);
                wasm_v128_store(dst1 + oc + 8, acc1_2);
                wasm_v128_store(dst1 + oc + 12, acc1_3);

                VX_WASM_IGEMM_RELU(acc2_0);
                VX_WASM_IGEMM_RELU(acc2_1);
                VX_WASM_IGEMM_RELU(acc2_2);
                VX_WASM_IGEMM_RELU(acc2_3);
                wasm_v128_store(dst2 + oc + 0, acc2_0);
                wasm_v128_store(dst2 + oc + 4, acc2_1);
                wasm_v128_store(dst2 + oc + 8, acc2_2);
                wasm_v128_store(dst2 + oc + 12, acc2_3);
            }

            /* 2-pixel remainder */
            for (; ox + 2 <= safe_interior_hi; ox += 2) {
                float* dst0 = output + (ox + 0) * out_c;
                float* dst1 = output + (ox + 1) * out_c;
                const int ix0 = (int)ox * sx - pl;
                const int ix1 = (int)(ox + 1) * sx - pl;

                v128_t acc0_0 = bias0;
                v128_t acc0_1 = bias1;
                v128_t acc0_2 = bias2;
                v128_t acc0_3 = bias3;

                v128_t acc1_0 = bias0;
                v128_t acc1_1 = bias1;
                v128_t acc1_2 = bias2;
                v128_t acc1_3 = bias3;

                const float* weights = weights_base;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = input + ((long)ix0 + kx * dx) * c;
                    const float* s1 = is_stride1 ? (s0 + c) : (input + ((long)ix1 + kx * dx) * c);
                    int ic = 0;
                    for (; ic + 2 <= c; ic += 2, weights += 32) {
                        const v128_t x0_0 = wasm_v128_load32_splat(s0 + ic + 0);
                        const v128_t x1_0 = wasm_v128_load32_splat(s1 + ic + 0);
                        const v128_t w0 = wasm_v128_load(weights + 0);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0_0, w0));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1_0, w0));
                        const v128_t w1 = wasm_v128_load(weights + 4);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0_0, w1));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1_0, w1));
                        const v128_t w2 = wasm_v128_load(weights + 8);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0_0, w2));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1_0, w2));
                        const v128_t w3 = wasm_v128_load(weights + 12);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0_0, w3));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1_0, w3));

                        const v128_t x0_1 = wasm_v128_load32_splat(s0 + ic + 1);
                        const v128_t x1_1 = wasm_v128_load32_splat(s1 + ic + 1);
                        const v128_t w4 = wasm_v128_load(weights + 16);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0_1, w4));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1_1, w4));
                        const v128_t w5 = wasm_v128_load(weights + 20);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0_1, w5));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1_1, w5));
                        const v128_t w6 = wasm_v128_load(weights + 24);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0_1, w6));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1_1, w6));
                        const v128_t w7 = wasm_v128_load(weights + 28);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0_1, w7));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1_1, w7));
                    }
                    for (; ic < c; ic++, weights += 16) {
                        const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                        const v128_t x1 = wasm_v128_load32_splat(s1 + ic);

                        const v128_t w0 = wasm_v128_load(weights + 0);
                        acc0_0 = wasm_f32x4_add(acc0_0, wasm_f32x4_mul(x0, w0));
                        acc1_0 = wasm_f32x4_add(acc1_0, wasm_f32x4_mul(x1, w0));

                        const v128_t w1 = wasm_v128_load(weights + 4);
                        acc0_1 = wasm_f32x4_add(acc0_1, wasm_f32x4_mul(x0, w1));
                        acc1_1 = wasm_f32x4_add(acc1_1, wasm_f32x4_mul(x1, w1));

                        const v128_t w2 = wasm_v128_load(weights + 8);
                        acc0_2 = wasm_f32x4_add(acc0_2, wasm_f32x4_mul(x0, w2));
                        acc1_2 = wasm_f32x4_add(acc1_2, wasm_f32x4_mul(x1, w2));

                        const v128_t w3 = wasm_v128_load(weights + 12);
                        acc0_3 = wasm_f32x4_add(acc0_3, wasm_f32x4_mul(x0, w3));
                        acc1_3 = wasm_f32x4_add(acc1_3, wasm_f32x4_mul(x1, w3));
                    }
                }

                VX_WASM_IGEMM_RELU(acc0_0);
                VX_WASM_IGEMM_RELU(acc0_1);
                VX_WASM_IGEMM_RELU(acc0_2);
                VX_WASM_IGEMM_RELU(acc0_3);
                wasm_v128_store(dst0 + oc + 0, acc0_0);
                wasm_v128_store(dst0 + oc + 4, acc0_1);
                wasm_v128_store(dst0 + oc + 8, acc0_2);
                wasm_v128_store(dst0 + oc + 12, acc0_3);

                VX_WASM_IGEMM_RELU(acc1_0);
                VX_WASM_IGEMM_RELU(acc1_1);
                VX_WASM_IGEMM_RELU(acc1_2);
                VX_WASM_IGEMM_RELU(acc1_3);
                wasm_v128_store(dst1 + oc + 0, acc1_0);
                wasm_v128_store(dst1 + oc + 4, acc1_1);
                wasm_v128_store(dst1 + oc + 8, acc1_2);
                wasm_v128_store(dst1 + oc + 12, acc1_3);
            }

            /* 1-pixel remainder in interior */
            for (; ox < safe_interior_hi; ox++) {
                float* dst0 = output + ox * out_c;
                const int ix0 = (int)ox * sx - pl;
                const float* weights = weights_base;
                v128_t acc0 = bias0;
                v128_t acc1 = bias1;
                v128_t acc2 = bias2;
                v128_t acc3 = bias3;
                for (int kx = 0; kx < kw; kx++) {
                    const float* s0 = input + ((long)ix0 + kx * dx) * c;
                    int ic = 0;
                    for (; ic + 2 <= c; ic += 2, weights += 32) {
                        const v128_t x0_0 = wasm_v128_load32_splat(s0 + ic + 0);
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x0_0, wasm_v128_load(weights + 0)));
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x0_0, wasm_v128_load(weights + 4)));
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x0_0, wasm_v128_load(weights + 8)));
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x0_0, wasm_v128_load(weights + 12)));

                        const v128_t x0_1 = wasm_v128_load32_splat(s0 + ic + 1);
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x0_1, wasm_v128_load(weights + 16)));
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x0_1, wasm_v128_load(weights + 20)));
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x0_1, wasm_v128_load(weights + 24)));
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x0_1, wasm_v128_load(weights + 28)));
                    }
                    for (; ic < c; ic++, weights += 16) {
                        const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x0, wasm_v128_load(weights + 0)));
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x0, wasm_v128_load(weights + 4)));
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x0, wasm_v128_load(weights + 8)));
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x0, wasm_v128_load(weights + 12)));
                    }
                }
                VX_WASM_IGEMM_RELU(acc0);
                VX_WASM_IGEMM_RELU(acc1);
                VX_WASM_IGEMM_RELU(acc2);
                VX_WASM_IGEMM_RELU(acc3);
                wasm_v128_store(dst0 + oc + 0, acc0);
                wasm_v128_store(dst0 + oc + 4, acc1);
                wasm_v128_store(dst0 + oc + 8, acc2);
                wasm_v128_store(dst0 + oc + 12, acc3);
            }

            /* Trailing edge and odd remainder pixels */
            for (; ox < ow; ox++) {
                float* dst0 = output + ox * out_c;
                const int ix0 = (int)ox * sx - pl;
                const float* weights = weights_base;
                v128_t acc0 = bias0;
                v128_t acc1 = bias1;
                v128_t acc2 = bias2;
                v128_t acc3 = bias3;
                for (int kx = 0; kx < kw; kx++) {
                    const int tap_ix = ix0 + kx * dx;
                    if ((unsigned)tap_ix >= (unsigned)w) {
                        weights += (long)c * 16;
                        continue;
                    }
                    const float* s0 = input + (long)tap_ix * c;
                    for (int ic = 0; ic < c; ic++, weights += 16) {
                        const v128_t w0 = wasm_v128_load(weights + 0);
                        const v128_t w1 = wasm_v128_load(weights + 4);
                        const v128_t w2 = wasm_v128_load(weights + 8);
                        const v128_t w3 = wasm_v128_load(weights + 12);
                        const v128_t x0 = wasm_v128_load32_splat(s0 + ic);
                        acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x0, w0));
                        acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x0, w1));
                        acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x0, w2));
                        acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x0, w3));
                    }
                }
                VX_WASM_IGEMM_RELU(acc0);
                VX_WASM_IGEMM_RELU(acc1);
                VX_WASM_IGEMM_RELU(acc2);
                VX_WASM_IGEMM_RELU(acc3);
                wasm_v128_store(dst0 + oc + 0, acc0);
                wasm_v128_store(dst0 + oc + 4, acc1);
                wasm_v128_store(dst0 + oc + 8, acc2);
                wasm_v128_store(dst0 + oc + 12, acc3);
            }
        }
#undef VX_WASM_IGEMM_RELU
#undef VX_WASM_IGEMM_RELU_SCALAR
        return 1;
        }
    }

    if (out_c < 16 || (out_c & 15)) return 0;
    if (pixels * ks > 4096) return 0;
    if (padded_windows && !vx_conv_weights_are_finite(
            wgt, (size_t)kh * kw * c * out_c)) return 0;
    const float** indir = vx_f32_igemm_indirection_cache(
        node_idx, input, n, h, w, c, oh, ow, kh, kw,
        sy, sx, pads, dy, dx);
    const float* packed = vx_f32_igemm_pack_cache(
        node_idx, wgt, c, out_c, ks);
    if (!indir || !packed) return 0;
#define VX_WASM_IGEMM_RELU(v) do { \
    if (relu) { \
        (v) = wasm_f32x4_max((v), zero); \
        if (relu >= 2) (v) = wasm_f32x4_min((v), six); \
    } \
} while (0)
    long pixel = 0;
    for (; pixel + 2 <= pixels; pixel += 2) {
        float* destinations[2] = {
            output + (pixel + 0) * out_c,
            output + (pixel + 1) * out_c,
        };
        const float* const* rows[2] = {
            indir + (pixel + 0) * ks,
            indir + (pixel + 1) * ks,
        };
        for (int oc = 0; oc < out_c; oc += 16) {
            const float* weights = packed +
                ((long)(oc >> 4) * ks) * c * 16;
            v128_t acc[2][4];
            for (int p = 0; p < 2; p++) {
                acc[p][0] = bias ? wasm_v128_load(bias + oc) : zero;
                acc[p][1] = bias ? wasm_v128_load(bias + oc + 4) : zero;
                acc[p][2] = bias ? wasm_v128_load(bias + oc + 8) : zero;
                acc[p][3] = bias ? wasm_v128_load(bias + oc + 12) : zero;
            }
            for (int tap = 0; tap < ks; tap++) {
                const float* sources[2] = {
                    rows[0][tap], rows[1][tap],
                };
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const v128_t w0 = wasm_v128_load(weights);
                    const v128_t w1 = wasm_v128_load(weights + 4);
                    const v128_t w2 = wasm_v128_load(weights + 8);
                    const v128_t w3 = wasm_v128_load(weights + 12);
                    for (int p = 0; p < 2; p++) {
                        const v128_t x = wasm_v128_load32_splat(sources[p] + ic);
                        acc[p][0] = wasm_f32x4_add(acc[p][0], wasm_f32x4_mul(x, w0));
                        acc[p][1] = wasm_f32x4_add(acc[p][1], wasm_f32x4_mul(x, w1));
                        acc[p][2] = wasm_f32x4_add(acc[p][2], wasm_f32x4_mul(x, w2));
                        acc[p][3] = wasm_f32x4_add(acc[p][3], wasm_f32x4_mul(x, w3));
                    }
                }
            }
            for (int p = 0; p < 2; p++) {
                VX_WASM_IGEMM_RELU(acc[p][0]);
                VX_WASM_IGEMM_RELU(acc[p][1]);
                VX_WASM_IGEMM_RELU(acc[p][2]);
                VX_WASM_IGEMM_RELU(acc[p][3]);
                wasm_v128_store(destinations[p] + oc, acc[p][0]);
                wasm_v128_store(destinations[p] + oc + 4, acc[p][1]);
                wasm_v128_store(destinations[p] + oc + 8, acc[p][2]);
                wasm_v128_store(destinations[p] + oc + 12, acc[p][3]);
            }
        }
    }
    for (; pixel < pixels; pixel++) {
        float* destination = output + pixel * out_c;
        const float* const* row = indir + pixel * ks;
        for (int oc = 0; oc < out_c; oc += 16) {
            const float* weights = packed +
                ((long)(oc >> 4) * ks) * c * 16;
            v128_t acc0 = bias ? wasm_v128_load(bias + oc) : zero;
            v128_t acc1 = bias ? wasm_v128_load(bias + oc + 4) : zero;
            v128_t acc2 = bias ? wasm_v128_load(bias + oc + 8) : zero;
            v128_t acc3 = bias ? wasm_v128_load(bias + oc + 12) : zero;
            for (int tap = 0; tap < ks; tap++) {
                const float* source = row[tap];
                for (int ic = 0; ic < c; ic++, weights += 16) {
                    const v128_t x = wasm_v128_load32_splat(source + ic);
                    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(x, wasm_v128_load(weights)));
                    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(x, wasm_v128_load(weights + 4)));
                    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(x, wasm_v128_load(weights + 8)));
                    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(x, wasm_v128_load(weights + 12)));
                }
            }
            VX_WASM_IGEMM_RELU(acc0);
            VX_WASM_IGEMM_RELU(acc1);
            VX_WASM_IGEMM_RELU(acc2);
            VX_WASM_IGEMM_RELU(acc3);
            wasm_v128_store(destination + oc, acc0);
            wasm_v128_store(destination + oc + 4, acc1);
            wasm_v128_store(destination + oc + 8, acc2);
            wasm_v128_store(destination + oc + 12, acc3);
        }
    }
#undef VX_WASM_IGEMM_RELU
#undef VX_WASM_IGEMM_RELU_SCALAR
    if (padded_windows)
        vx_conv2d_restore_padded_zeros(input, output, wgt, bias,
            n, h, w, c, oh, ow, out_c, kh, kw, sy, sx, pads, dy, dx, relu);
    return 1;
#elif defined(__AVX2__)
    if (n == 1 && h == 1 && kh == 1 && oh == 1 && pads[0] == 0 && pads[2] == 0 &&
        out_c >= 16 && (out_c & 15) == 0) {
        const float* packed = vx_f32_igemm_pack_cache(node_idx, wgt, c, out_c, kw);
        if (packed) {
            const int pl = pads[1];
            const int interior_lo = sx > 0 ? (pl + sx - 1) / sx : ow;
            const int interior_hi = sx > 0 && (w - 1 + pl - (kw - 1) * dx) >= 0
                ? ((w - 1 + pl - (kw - 1) * dx) / sx) + 1 : 0;
            const long safe_interior_hi = interior_hi < ow ? interior_hi : ow;

            VxConv1DDirectAvx2Context ctx = {
                input, output, packed, bias,
                w, c, ow, out_c, kw, sx, pl, dx, relu,
                interior_lo, safe_interior_hi
            };

            int threads = vx_kernels_thread_count();
            if (threads > 1 && ow >= 64) {
                int grain = 1 + (ow - 1) / (threads * 4);
                grain = (grain + 3) & ~3;
                if (grain < 16) grain = 16;
                vx_kernels_parallel_for(ow, grain, vx_conv1d_direct_avx2_worker, &ctx);
            } else {
                vx_conv1d_direct_avx2_worker(&ctx, 0, ow);
            }
            return 1;
        }
    }

    if (kh * kw <= 1 || out_c < 16) return 0;
    if (padded_windows && !vx_conv_weights_are_finite(
            wgt, (size_t)kh * kw * c * out_c)) return 0;
    const float** indir = vx_f32_igemm_indirection_cache(node_idx, input, n, h, w, c, oh, ow,
                                                        kh, kw, sy, sx, pads, dy, dx);
    if (!indir) return 0;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 six = _mm256_set1_ps(6.0f);
    long pixels = (long)n * oh * ow;
    int ks = kh * kw;
    /* Packed weights stream one cache line per input channel; HWIO strides
     * out_c floats and stalls on every load past ~192 channels. Both forms feed
     * the same loop through `wstep`, so a failed pack just costs speed. Only the
     * sixteen-channel tiers are packed; the 8- and 1-channel remainders run at
     * out_c % 16 and stay on HWIO. */
    const float* packed = vx_f32_igemm_pack_cache(node_idx, wgt, c, out_c, ks);
    const long wstep = packed ? 16 : out_c;
    const long wtap = (long)c * wstep;
#define VX_IGEMM_RELU(v) do { if (relu) { (v) = _mm256_max_ps((v), zero); if (relu >= 2) (v) = _mm256_min_ps((v), six); } } while (0)
    /* Six pixels against sixteen channels is twelve accumulator chains, which
     * is what keeps both FMA pipes busy across their ~4-cycle latency; the
     * four-pixel tier below runs exactly eight and stalls. Twelve accumulators
     * plus two weight vectors and one broadcast is fifteen of the sixteen YMM
     * registers, so this is the widest tier that still avoids spilling — four
     * pixels by twenty-four channels has the same twelve chains but needs all
     * sixteen and measures slower. `vx_conv2d_pointwise_gemm_f32` already
     * blocks six pixels for the same reason. The cache-producing owner thread
     * completes all mutable setup before these independent blocks dispatch. */
    const long six_pixel_block_count = pixels / 6;
    VxF32IgemmSixPixelContext six_pixel_context = {
        .output = output,
        .weights = wgt,
        .packed_weights = packed,
        .bias = bias,
        .indirection = indir,
        .input_channels = c,
        .output_channels = out_c,
        .kernel_size = ks,
        .weight_step = wstep,
        .weight_tap = wtap,
        .block_offset = 0,
        .relu = relu,
    };
    if (six_pixel_block_count <= INT32_MAX) {
        const int six_pixel_blocks = (int)six_pixel_block_count;
        const int threads = vx_kernels_thread_count();
        const uint64_t target_chunks = threads > 0
            ? (uint64_t)(uint32_t)threads * 4u : 1u;
        int grain = threads > 0
            ? (int)((uint64_t)(uint32_t)six_pixel_blocks / target_chunks)
            : six_pixel_blocks;
        if (grain < 1) grain = 1;
        const uint64_t six_pixel_macs = (uint64_t)six_pixel_blocks * 6u *
            (uint64_t)ks * (uint64_t)c * (uint64_t)out_c;
        const int oc_blocks = (out_c + 15) / 16;
        if (threads > 1 && six_pixel_macs >= 131072u) {
            if ((six_pixel_blocks < threads * 4 && oc_blocks >= threads * 2) ||
                (six_pixel_blocks < threads * 2 && oc_blocks >= threads)) {
                VxF32IgemmSixPixelOcParallelContext oc_ctx = {
                    .base = six_pixel_context,
                    .six_pixel_blocks = six_pixel_blocks,
                    .oc_blocks = oc_blocks,
                };
                int grain = (oc_blocks + threads - 1) / threads;
                if (grain < 1) grain = 1;
                vx_kernels_parallel_for(oc_blocks, grain,
                                        vx_f32_igemm_six_pixel_oc_worker,
                                        &oc_ctx);
            } else {
                vx_kernels_parallel_for(six_pixel_blocks, grain,
                                        vx_f32_igemm_six_pixel_worker,
                                        &six_pixel_context);
            }
        } else {
            vx_f32_igemm_six_pixel_worker(&six_pixel_context, 0,
                                          six_pixel_blocks);
        }
    } else {
        /* The pool ABI uses int task indices. Preserve correctness for a
         * representable long-sized tensor by running bounded serial chunks
         * instead of narrowing the block count and sending tail writes before
         * the output base. Such tensors are impractically large today, but the
         * generic kernel contract must still avoid implementation-defined
         * narrowing and out-of-bounds access. */
        long offset = 0;
        while (offset < six_pixel_block_count) {
            const long remaining = six_pixel_block_count - offset;
            const int chunk = remaining > INT32_MAX
                ? INT32_MAX : (int)remaining;
            six_pixel_context.block_offset = offset;
            vx_f32_igemm_six_pixel_worker(&six_pixel_context, 0, chunk);
            offset += chunk;
        }
    }
    long p = six_pixel_block_count * 6;
    for (; p + 4 <= pixels; p += 4) {
        float* d0 = output + (p + 0) * out_c;
        float* d1 = output + (p + 1) * out_c;
        float* d2 = output + (p + 2) * out_c;
        float* d3 = output + (p + 3) * out_c;
        const float** r0 = indir + (p + 0) * ks;
        const float** r1 = indir + (p + 1) * ks;
        const float** r2 = indir + (p + 2) * ks;
        const float** r3 = indir + (p + 3) * ks;
        int oc = 0;
        for (; oc + 16 <= out_c; oc += 16) {
            const float* wbase = packed ? packed + ((long)(oc >> 4) * ks) * c * 16
                                        : wgt + oc;
            __m256 bl = bias ? _mm256_loadu_ps(bias + oc) : zero;
            __m256 bh = bias ? _mm256_loadu_ps(bias + oc + 8) : zero;
            __m256 a0l = bl, a0h = bh, a1l = bl, a1h = bh;
            __m256 a2l = bl, a2h = bh, a3l = bl, a3h = bh;
            for (int t = 0; t < ks; t++) {
                const float* s0 = r0[t];
                const float* s1 = r1[t];
                const float* s2 = r2[t];
                const float* s3 = r3[t];
                const float* ww = wbase + (long)t * wtap;
                for (int ic = 0; ic < c; ic++, ww += wstep) {
                    __m256 wl = _mm256_loadu_ps(ww);
                    __m256 wh = _mm256_loadu_ps(ww + 8);
                    __m256 x;
                    x = _mm256_set1_ps(s0[ic]); a0l = _mm256_fmadd_ps(x, wl, a0l); a0h = _mm256_fmadd_ps(x, wh, a0h);
                    x = _mm256_set1_ps(s1[ic]); a1l = _mm256_fmadd_ps(x, wl, a1l); a1h = _mm256_fmadd_ps(x, wh, a1h);
                    x = _mm256_set1_ps(s2[ic]); a2l = _mm256_fmadd_ps(x, wl, a2l); a2h = _mm256_fmadd_ps(x, wh, a2h);
                    x = _mm256_set1_ps(s3[ic]); a3l = _mm256_fmadd_ps(x, wl, a3l); a3h = _mm256_fmadd_ps(x, wh, a3h);
                }
            }
            VX_IGEMM_RELU(a0l); VX_IGEMM_RELU(a0h); VX_IGEMM_RELU(a1l); VX_IGEMM_RELU(a1h);
            VX_IGEMM_RELU(a2l); VX_IGEMM_RELU(a2h); VX_IGEMM_RELU(a3l); VX_IGEMM_RELU(a3h);
            _mm256_storeu_ps(d0 + oc, a0l); _mm256_storeu_ps(d0 + oc + 8, a0h);
            _mm256_storeu_ps(d1 + oc, a1l); _mm256_storeu_ps(d1 + oc + 8, a1h);
            _mm256_storeu_ps(d2 + oc, a2l); _mm256_storeu_ps(d2 + oc + 8, a2h);
            _mm256_storeu_ps(d3 + oc, a3l); _mm256_storeu_ps(d3 + oc + 8, a3h);
        }
        for (; oc + 8 <= out_c; oc += 8) {
            __m256 a0 = bias ? _mm256_loadu_ps(bias + oc) : zero;
            __m256 a1 = a0, a2 = a0, a3 = a0;
            for (int t = 0; t < ks; t++) {
                const float* s0 = r0[t];
                const float* s1 = r1[t];
                const float* s2 = r2[t];
                const float* s3 = r3[t];
                const float* ww = wgt + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) {
                    __m256 wv = _mm256_loadu_ps(ww + (long)ic * out_c);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(s0[ic]), wv, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(s1[ic]), wv, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(s2[ic]), wv, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(s3[ic]), wv, a3);
                }
            }
            VX_IGEMM_RELU(a0); VX_IGEMM_RELU(a1); VX_IGEMM_RELU(a2); VX_IGEMM_RELU(a3);
            _mm256_storeu_ps(d0 + oc, a0); _mm256_storeu_ps(d1 + oc, a1);
            _mm256_storeu_ps(d2 + oc, a2); _mm256_storeu_ps(d3 + oc, a3);
        }
        for (; oc < out_c; oc++) {
            float s0v = bias ? bias[oc] : 0.0f, s1v = s0v, s2v = s0v, s3v = s0v;
            for (int t = 0; t < ks; t++) {
                const float* x0 = r0[t];
                const float* x1 = r1[t];
                const float* x2 = r2[t];
                const float* x3 = r3[t];
                const float* ww = wgt + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) {
                    float wv = ww[(long)ic * out_c];
                    s0v += x0[ic] * wv; s1v += x1[ic] * wv;
                    s2v += x2[ic] * wv; s3v += x3[ic] * wv;
                }
            }
            d0[oc] = vx_relu6_apply(s0v, relu);
            d1[oc] = vx_relu6_apply(s1v, relu);
            d2[oc] = vx_relu6_apply(s2v, relu);
            d3[oc] = vx_relu6_apply(s3v, relu);
        }
    }
    for (; p < pixels; p++) {
        float* dst = output + p * out_c;
        const float** row = indir + p * ks;
        int oc = 0;
        for (; oc + 8 <= out_c; oc += 8) {
            __m256 acc = bias ? _mm256_loadu_ps(bias + oc) : zero;
            for (int t = 0; t < ks; t++) {
                const float* src = row[t];
                const float* ww = wgt + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(src[ic]),
                                          _mm256_loadu_ps(ww + (long)ic * out_c), acc);
                }
            }
            VX_IGEMM_RELU(acc);
            _mm256_storeu_ps(dst + oc, acc);
        }
        for (; oc < out_c; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int t = 0; t < ks; t++) {
                const float* src = row[t];
                const float* ww = wgt + ((long)t * c) * out_c + oc;
                for (int ic = 0; ic < c; ic++) sum += src[ic] * ww[(long)ic * out_c];
            }
            dst[oc] = vx_relu6_apply(sum, relu);
        }
    }
#undef VX_IGEMM_RELU
    if (padded_windows)
        vx_conv2d_restore_padded_zeros(input, output, wgt, bias,
            n, h, w, c, oh, ow, out_c, kh, kw, sy, sx, pads, dy, dx, relu);
    return 1;
#else
    (void)node_idx; (void)input; (void)output; (void)wgt; (void)bias;
    (void)n; (void)h; (void)w; (void)c; (void)oh; (void)ow; (void)out_c;
    (void)kh; (void)kw; (void)sy; (void)sx; (void)pads; (void)dy; (void)dx; (void)relu;
    return 0;
#endif
}

void vx_conv_f32_isa_free_all(void) {
    for (size_t i = 0; i < vx_engine_state_current()->node_capacity; i++) {
        free(g_pwf32_pack[i]);
        free(g_pwf32_pack_plain[i]);
        g_pwf32_pack_plain[i] = NULL;
        free(g_dw_pw_tmp[i]);
        free((void*)g_f32_igemm_indir[i]);
        free(g_f32_igemm_zero[i]);
        free(g_f32_igemm_pack[i]);
        g_f32_igemm_pack[i] = NULL;
        g_f32_igemm_pack_cap[i] = 0;
        g_f32_igemm_pack_src[i] = NULL;
        g_pwf32_pack[i] = NULL;
        g_dw_pw_tmp[i] = NULL;
        g_dw_pw_tmp_cap[i] = 0;
        g_f32_igemm_indir[i] = NULL;
        g_f32_igemm_indir_cap[i] = 0;
        g_f32_igemm_indir_key[i] = 0;
        g_f32_igemm_zero[i] = NULL;
        g_f32_igemm_zero_cap[i] = 0;
    }
}
