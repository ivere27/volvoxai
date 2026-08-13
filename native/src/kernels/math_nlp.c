#include "mathcompat.h"
#include "thread_pool.h"

#include <stddef.h>
#include <stdint.h>

// --- Missing Math & NLP Primitives ---
void sub_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] - b[i];
}
void div_f32(const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = a[i] / (b[i] + 1e-9f);
}
void silu_f32(const float* input, float* output, int n) {
    int i = 0;
#if VX_FASTMATH_WASM_SIMD
    i = vx_silu_f32_wasm_simd(input, output, n);
#endif
#if VX_FASTMATH_X86_AVX2
    if (vx_kernel_platform()->has_avx2) i = vx_silu_f32_avx2(input, output, n);
#endif
    for (; i < n; i++) output[i] = input[i] * (1.0f / (1.0f + accurate_expf(-input[i])));
}
#if defined(__wasm__) && defined(VOLVOXAI_SILU_TESTING)
WASM_EXPORT("silu_f32_scalar_reference")
void silu_f32_scalar_reference(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = input[i] * (1.0f / (1.0f + accurate_expf(-input[i])));
}
#endif
void leakyrelu_f32(const float* input, float* output, int n, float alpha) {
    for (int i = 0; i < n; i++) output[i] = input[i] > 0.0f ? input[i] : input[i] * alpha;
}
void tanh_f32(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = tanhf(input[i]);
}
void clip_f32(const float* input, float* output, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        output[i] = v;
    }
}
void rmsnorm_f32(const float* input, const float* weight, float* output, int seq_len, int d_model, double eps) {
    for (int s = 0; s < seq_len; s++) {
        double var = 0.0;
        for (int d = 0; d < d_model; d++) {
            double value = input[s*d_model + d];
            var += value * value;
        }
        var /= (double)d_model;
        double inv_std = 1.0 / __builtin_sqrt(var + eps);
        for (int d = 0; d < d_model; d++) {
            output[s*d_model + d] = (float)((double)input[s*d_model + d] * inv_std * weight[d]);
        }
    }
}
typedef struct VxSoftmaxF32RowsContext {
    const float* input;
    float* output;
    int width;
    int use_avx2;
} VxSoftmaxF32RowsContext;

#if VX_FASTMATH_X86_AVX2
/* Keep the scalar maximum and sum reductions in their original left-to-right
 * order. Only the lane-independent exponential and normalization work is
 * vectorized, so the AVX2 tier stays byte-identical to the scalar tier for
 * finite rows. Non-finite rows never enter this helper. */
static VX_FASTMATH_TARGET_AVX2 void vx_softmax_f32_finite_row_avx2(
        const float* input, float* output, int width, float maximum) {
    const __m256 maximum_lanes = _mm256_set1_ps(maximum);
    float sum = 0.0f;
    int column = 0;
    for (; column + 8 <= width; column += 8) {
        __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(input + column),
                                     maximum_lanes);
        /* Scalar accurate_expf returns zero before range reduction below -87.
         * Keeping those lanes finite avoids a spurious invalid conversion for
         * an extreme finite subtraction that rounds to negative infinity. */
        delta = _mm256_max_ps(delta, _mm256_set1_ps(-88.0f));
        _mm256_storeu_ps(output + column, vx_accurate_exp_avx2(delta));
        sum += output[column + 0];
        sum += output[column + 1];
        sum += output[column + 2];
        sum += output[column + 3];
        sum += output[column + 4];
        sum += output[column + 5];
        sum += output[column + 6];
        sum += output[column + 7];
    }
    for (; column < width; column++) {
        output[column] = accurate_expf(input[column] - maximum);
        sum += output[column];
    }
    {
        const __m256 denominator = _mm256_set1_ps(sum);
        column = 0;
        for (; column + 8 <= width; column += 8) {
            _mm256_storeu_ps(output + column, _mm256_div_ps(
                _mm256_loadu_ps(output + column), denominator));
        }
    }
    for (; column < width; column++) output[column] /= sum;
}
#endif

#if VX_FASTMATH_WASM_SIMD
/* The maximum and denominator reductions retain scalar left-to-right order;
 * only the lane-independent exponential and normalization use SIMD128.  This
 * mirrors the AVX2 route while avoiding four scalar polynomial evaluations per
 * vector on WASM. Non-finite rows remain on the scalar contract below. */
static void vx_softmax_f32_finite_row_wasm_simd(
        const float* input, float* output, int width, float maximum) {
    const v128_t maximum_lanes = wasm_f32x4_splat(maximum);
    float sum = 0.0f;
    int column = 0;
    for (; column + 4 <= width; column += 4) {
        v128_t delta = wasm_f32x4_sub(
            wasm_v128_load(input + column), maximum_lanes);
        delta = wasm_f32x4_max(delta, wasm_f32x4_splat(-88.0f));
        wasm_v128_store(output + column,
            vx_accurate_exp_wasm_simd(delta));
        sum += output[column + 0];
        sum += output[column + 1];
        sum += output[column + 2];
        sum += output[column + 3];
    }
    for (; column < width; column++) {
        output[column] = accurate_expf(input[column] - maximum);
        sum += output[column];
    }
    {
        const v128_t denominator = wasm_f32x4_splat(sum);
        column = 0;
        for (; column + 4 <= width; column += 4) {
            wasm_v128_store(output + column, wasm_f32x4_div(
                wasm_v128_load(output + column), denominator));
        }
    }
    for (; column < width; column++) output[column] /= sum;
}
#endif

static void vx_softmax_f32_rows(void* opaque, int begin, int end) {
    VxSoftmaxF32RowsContext* context =
        (VxSoftmaxF32RowsContext*)opaque;
    const int d = context->width;
    for (int i = begin; i < end; i++) {
        const size_t offset = (size_t)i * (size_t)d;
        const float* input = context->input + offset;
        float* output = context->output + offset;
        float max_val = input[0];
#if VX_FASTMATH_WASM_SIMD
        if (d >= 4) {
            int finite = __builtin_isfinite(max_val);
            for (int j = 1; j < d; j++) {
                if (!__builtin_isfinite(input[j])) finite = 0;
                if (input[j] > max_val) max_val = input[j];
            }
            if (finite) {
                vx_softmax_f32_finite_row_wasm_simd(
                    input, output, d, max_val);
                continue;
            }
        } else {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#elif VX_FASTMATH_X86_AVX2
        if (context->use_avx2 && d >= 8) {
            int finite = __builtin_isfinite(max_val);
            for (int j = 1; j < d; j++) {
                if (!__builtin_isfinite(input[j])) finite = 0;
                if (input[j] > max_val) max_val = input[j];
            }
            if (finite) {
                vx_softmax_f32_finite_row_avx2(input, output, d, max_val);
                continue;
            }
        } else {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#else
        {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#endif
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            output[j] = accurate_expf(input[j] - max_val);
            sum += output[j];
        }
        for (int j = 0; j < d; j++) {
            output[j] /= sum;
        }
    }
}

void softmax_f32(const float* input, float* output, int b, int d) {
    VxSoftmaxF32RowsContext context;
    int threads;
    int grain;
    if (!input || !output || b <= 0 || d <= 0) return;
    context.input = input;
    context.output = output;
    context.width = d;
#if VX_FASTMATH_X86_AVX2
    context.use_avx2 = vx_kernel_platform()->has_avx2;
#else
    context.use_avx2 = 0;
#endif
    threads = vx_kernels_thread_count();
    /* Keep short row batches on the caller and split only when the exponential
     * work amortizes a pool wake-up. Every worker owns complete rows, so each
     * reduction retains the scalar operation order and 1T/NT results remain
     * byte-identical. */
    if (threads <= 1 || b <= 1 ||
        (uint64_t)(uint32_t)b * (uint64_t)(uint32_t)d < 65536u) {
        vx_softmax_f32_rows(&context, 0, b);
        return;
    }
    {
        const uint64_t target_chunks = (uint64_t)(uint32_t)threads * 4u;
        grain = (int)((uint64_t)(uint32_t)b / target_chunks +
                      ((uint64_t)(uint32_t)b % target_chunks != 0u));
    }
    if (grain < 1) grain = 1;
    vx_kernels_parallel_for(b, grain, vx_softmax_f32_rows, &context);
}
