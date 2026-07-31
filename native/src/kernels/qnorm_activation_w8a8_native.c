/*
 * Native CPU work partitioning for canonical physical-byte QSiLU and
 * QLayerNorm.
 *
 * The portable exports remain the authoritative scalar/WASM implementations.
 * Native runtime descriptors have already validated the complete call, so a
 * large activation can be divided into independent byte ranges and LayerNorm
 * can be divided into independent final-axis rows.  Each worker invokes the
 * canonical implementation for its contiguous slice, preserving its exact
 * F32 operation and ties-to-even requantization order within every output.
 */
#include "quant_cpu_opt.h"
#include "w8a8_affine.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_QNORM_X86_AVX2 1
#define VX_QNORM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QNORM_X86_AVX2 0
#define VX_QNORM_TARGET_AVX2
#endif

extern int qsilu_i8u8(const void* input, void* output, uint32_t elements,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype);

extern int qlayernorm_i8u8(const void* input, const float* weight,
        const float* bias, void* output, uint32_t rows, uint32_t d_model,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype);

enum {
    /* Keep decoder prefixes on the caller: the largest TinyReceipt decoder
     * SiLU is 192*1280 elements, where a pool dispatch per layer costs more
     * than the byte-LUT work.  Encoder feature maps still clear this limit. */
    VX_QSILU_PARALLEL_ELEMENTS = 512u * 1024u,
    /* TinyReceipt's encoder has 402 final-axis rows, while a full decoder
     * prefix has at most 192.  This keeps row-level pool dispatches confined
     * to the encoder without coupling the kernel to either graph shape. */
    VX_QLAYERNORM_PARALLEL_ELEMENTS = 96u * 1024u,
};

typedef struct {
    const uint8_t* input;
    uint8_t* output;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    uint32_t input_dtype;
    uint32_t output_dtype;
    atomic_int failed;
} VxQSiLUParallelContext;

static void vx_qsilu_parallel_worker(void* opaque, int begin, int end) {
    VxQSiLUParallelContext* context = (VxQSiLUParallelContext*)opaque;
    if (begin >= end || atomic_load_explicit(
            &context->failed, memory_order_relaxed)) return;
    if (!qsilu_i8u8(context->input + begin, context->output + begin,
            (uint32_t)(end - begin), context->input_scale,
            context->input_zero_point, context->output_scale,
            context->output_zero_point, context->input_dtype,
            context->output_dtype)) {
        atomic_store_explicit(&context->failed, 1, memory_order_relaxed);
    }
}

int vx_qsilu_i8u8_native_validated(
        const void* input, void* output, uint32_t elements,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    const int threads = vx_kernels_thread_count();
    VxQSiLUParallelContext context;
    uint32_t grain;
    if (elements < VX_QSILU_PARALLEL_ELEMENTS || elements > INT_MAX ||
        threads <= 1) {
        return qsilu_i8u8(input, output, elements, input_scale,
            input_zero_point, output_scale, output_zero_point, input_dtype,
            output_dtype);
    }
    grain = (elements + (uint32_t)threads - 1u) / (uint32_t)threads;
    context = (VxQSiLUParallelContext){
        .input = (const uint8_t*)input,
        .output = (uint8_t*)output,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .input_dtype = input_dtype,
        .output_dtype = output_dtype,
    };
    atomic_init(&context.failed, 0);
    vx_kernels_parallel_for((int)elements, (int)grain,
                            vx_qsilu_parallel_worker, &context);
    return !atomic_load_explicit(&context.failed, memory_order_relaxed);
}

typedef struct {
    const uint8_t* input;
    const float* weight;
    const float* bias;
    uint8_t* output;
    uint32_t d_model;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    float epsilon;
    uint32_t input_dtype;
    uint32_t output_dtype;
    int use_avx2;
    atomic_int failed;
} VxQLayerNormParallelContext;

#if VX_QNORM_X86_AVX2
static VX_QNORM_TARGET_AVX2 __m256i vx_qlayernorm_load8_avx2(
        const uint8_t* input, uint32_t input_dtype) {
    const __m128i bytes = _mm_loadl_epi64((const __m128i*)input);
    return input_dtype == VX_DTYPE_I8
        ? _mm256_cvtepi8_epi32(bytes) : _mm256_cvtepu8_epi32(bytes);
}

static VX_QNORM_TARGET_AVX2 __m256i vx_qlayernorm_quantize8_avx2(
        __m256 transformed, int32_t zero_point, int32_t minimum,
        int32_t maximum) {
    const __m256 nan_mask = _mm256_cmp_ps(
        transformed, transformed, _CMP_UNORD_Q);
    const __m256 clamped = _mm256_min_ps(
        _mm256_max_ps(transformed, _mm256_set1_ps((float)minimum)),
        _mm256_set1_ps((float)maximum));
    const __m256 rounded = _mm256_round_ps(
        clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm256_blendv_epi8(
        _mm256_cvttps_epi32(rounded), _mm256_set1_epi32(zero_point),
        _mm256_castps_si256(nan_mask));
}

static VX_QNORM_TARGET_AVX2 void vx_qlayernorm_store8_avx2(
        uint8_t* output, uint32_t output_dtype, __m256i quantized) {
    const __m128i low = _mm256_castsi256_si128(quantized);
    const __m128i high = _mm256_extracti128_si256(quantized, 1);
    __m128i bytes;
    uint64_t packed;
    if (output_dtype == VX_DTYPE_I8) {
        const __m128i values16 = _mm_packs_epi32(low, high);
        bytes = _mm_packs_epi16(values16, values16);
    } else {
        const __m128i values16 = _mm_packus_epi32(low, high);
        bytes = _mm_packus_epi16(values16, values16);
    }
    packed = (uint64_t)_mm_cvtsi128_si64(bytes);
    memcpy(output, &packed, sizeof(packed));
}

static float vx_qlayernorm_load_f32(const float* values, uint32_t index) {
    float value;
    memcpy(&value, (const uint8_t*)values + (size_t)index * sizeof(float),
           sizeof(value));
    return value;
}

/* Statistics retain the portable scalar order.  Only the independent affine
 * and requantization lanes are vectorized, so graph-visible byte results stay
 * identical without depending on a compiler loop-vectorization decision. */
static VX_QNORM_TARGET_AVX2 void vx_qlayernorm_rows_avx2(
        const uint8_t* input, const float* weight, const float* bias,
        uint8_t* output, uint32_t rows, uint32_t d_model, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype) {
    const int32_t output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    const __m256 input_zeros = _mm256_set1_ps((float)input_zero_point);
    const __m256 input_scales = _mm256_set1_ps(input_scale);
    const __m256 output_scales = _mm256_set1_ps(output_scale);
    const __m256 output_zeros = _mm256_set1_ps((float)output_zero_point);
    for (uint32_t row = 0; row < rows; row++) {
        const size_t offset = (size_t)row * d_model;
        float sum = 0.0f;
        float centered_square_sum = 0.0f;
        float mean_raw;
        float raw_variance;
        float real_variance;
        float inverse_stddev;
        uint32_t channel = 0;
        if (input_dtype == VX_DTYPE_I8) {
            const int8_t* values = (const int8_t*)input + offset;
            for (; channel < d_model; channel++)
                sum = sum + (float)((int32_t)values[channel] - input_zero_point);
        } else {
            const uint8_t* values = input + offset;
            for (; channel < d_model; channel++)
                sum = sum + (float)((int32_t)values[channel] - input_zero_point);
        }
        mean_raw = sum / (float)d_model;
        if (input_dtype == VX_DTYPE_I8) {
            const int8_t* values = (const int8_t*)input + offset;
            for (channel = 0; channel < d_model; channel++) {
                const float centered =
                    (float)((int32_t)values[channel] - input_zero_point) - mean_raw;
                centered_square_sum = centered_square_sum + centered * centered;
            }
        } else {
            const uint8_t* values = input + offset;
            for (channel = 0; channel < d_model; channel++) {
                const float centered =
                    (float)((int32_t)values[channel] - input_zero_point) - mean_raw;
                centered_square_sum = centered_square_sum + centered * centered;
            }
        }
        raw_variance = centered_square_sum / (float)d_model;
        if (raw_variance < 0.0f) raw_variance = 0.0f;
        real_variance = (raw_variance * input_scale) * input_scale;
        if (real_variance < 0.0f) real_variance = 0.0f;
        inverse_stddev = 1.0f / sqrtf(real_variance + epsilon);
        {
            const __m256 means = _mm256_set1_ps(mean_raw);
            const __m256 inverse_stddevs = _mm256_set1_ps(inverse_stddev);
            for (channel = 0; channel + 8u <= d_model; channel += 8u) {
                const __m256 raw = _mm256_cvtepi32_ps(vx_qlayernorm_load8_avx2(
                    input + offset + channel, input_dtype));
                const __m256 centered = _mm256_sub_ps(
                    _mm256_sub_ps(raw, input_zeros), means);
                const __m256 normalized = _mm256_mul_ps(
                    _mm256_mul_ps(centered, input_scales), inverse_stddevs);
                const __m256 affine = _mm256_add_ps(
                    _mm256_mul_ps(normalized,
                        _mm256_loadu_ps(weight + channel)),
                    _mm256_loadu_ps(bias + channel));
                const __m256 transformed = _mm256_add_ps(
                    _mm256_div_ps(affine, output_scales), output_zeros);
                vx_qlayernorm_store8_avx2(output + offset + channel,
                    output_dtype, vx_qlayernorm_quantize8_avx2(transformed,
                        output_zero_point, output_minimum, output_maximum));
            }
        }
        for (; channel < d_model; channel++) {
            const int32_t stored = input_dtype == VX_DTYPE_I8
                ? (int32_t)((const int8_t*)input)[offset + channel]
                : (int32_t)input[offset + channel];
            const float raw = (float)(stored - input_zero_point);
            const float normalized =
                (raw - mean_raw) * input_scale * inverse_stddev;
            const float affine = normalized *
                vx_qlayernorm_load_f32(weight, channel) +
                vx_qlayernorm_load_f32(bias, channel);
            const float transformed =
                affine / output_scale + (float)output_zero_point;
            const int32_t quantized = vx_w8a8_requantize(transformed,
                output_minimum, output_maximum, output_zero_point);
            if (output_dtype == VX_DTYPE_I8)
                ((int8_t*)output)[offset + channel] = (int8_t)quantized;
            else output[offset + channel] = (uint8_t)quantized;
        }
    }
}
#endif

static void vx_qlayernorm_parallel_worker(void* opaque, int begin, int end) {
    VxQLayerNormParallelContext* context =
        (VxQLayerNormParallelContext*)opaque;
    const size_t offset = (size_t)begin * context->d_model;
    if (begin >= end || atomic_load_explicit(
            &context->failed, memory_order_relaxed)) return;
#if VX_QNORM_X86_AVX2
    if (context->use_avx2) {
        vx_qlayernorm_rows_avx2(context->input + offset, context->weight,
            context->bias, context->output + offset, (uint32_t)(end - begin),
            context->d_model, context->input_scale,
            context->input_zero_point, context->output_scale,
            context->output_zero_point, context->epsilon,
            context->input_dtype, context->output_dtype);
        return;
    }
#endif
    if (!qlayernorm_i8u8(context->input + offset, context->weight,
            context->bias, context->output + offset, (uint32_t)(end - begin),
            context->d_model, context->input_scale,
            context->input_zero_point, context->output_scale,
            context->output_zero_point, context->epsilon,
            context->input_dtype, context->output_dtype)) {
        atomic_store_explicit(&context->failed, 1, memory_order_relaxed);
    }
}

int vx_qlayernorm_i8u8_native_validated(
        const void* input, const float* weight, const float* bias, void* output,
        uint32_t rows, uint32_t d_model, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype) {
    const uint64_t elements = (uint64_t)rows * d_model;
    const int threads = vx_kernels_thread_count();
#if VX_QNORM_X86_AVX2
    const int use_avx2 = vx_kernel_platform()->has_avx2;
#else
    const int use_avx2 = 0;
#endif
    VxQLayerNormParallelContext context;
    uint32_t grain;
    if (elements < VX_QLAYERNORM_PARALLEL_ELEMENTS || rows > INT_MAX ||
        threads <= 1 || rows < (uint32_t)threads) {
#if VX_QNORM_X86_AVX2
        if (use_avx2) {
            vx_qlayernorm_rows_avx2(input, weight, bias, output, rows,
                d_model, input_scale, input_zero_point, output_scale,
                output_zero_point, epsilon, input_dtype, output_dtype);
            return 1;
        }
#endif
        return qlayernorm_i8u8(input, weight, bias, output, rows, d_model,
            input_scale, input_zero_point, output_scale, output_zero_point,
            epsilon, input_dtype, output_dtype);
    }
    grain = (rows + (uint32_t)threads - 1u) / (uint32_t)threads;
    context = (VxQLayerNormParallelContext){
        .input = (const uint8_t*)input,
        .weight = weight,
        .bias = bias,
        .output = (uint8_t*)output,
        .d_model = d_model,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .epsilon = epsilon,
        .input_dtype = input_dtype,
        .output_dtype = output_dtype,
        .use_avx2 = use_avx2,
    };
    atomic_init(&context.failed, 0);
    vx_kernels_parallel_for((int)rows, (int)grain,
                            vx_qlayernorm_parallel_worker, &context);
    return !atomic_load_explicit(&context.failed, memory_order_relaxed);
}
