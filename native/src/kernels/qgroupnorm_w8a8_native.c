/*
 * Native CPU parallelism for canonical physical-byte W8A8 QGroupNorm.
 *
 * qgroupnorm_i8u8() remains the authoritative, fully validating portable
 * implementation and the sole WASM path.  The physical native runtime has
 * already validated this call, so independent [batch, group] reductions can
 * run on the bound kernel pool.  Every reduction keeps the portable loop and
 * F32 operation order, preserving byte-exact output.
 */
#include "quant_cpu_opt.h"
#include "w8a8_affine.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_QGROUPNORM_X86_AVX2 1
#define VX_QGROUPNORM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QGROUPNORM_X86_AVX2 0
#define VX_QGROUPNORM_TARGET_AVX2
#endif

extern int qgroupnorm_i8u8(const void* input, const float* weight,
        const float* bias, void* output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels, uint32_t groups, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype);

enum {
    /* Below this size, pool wake-up is normally more expensive than the
     * scalar kernel.  TinyReceipt's encoder GroupNorm calls are all larger. */
    VX_QGROUPNORM_PARALLEL_ELEMENTS = 16u * 1024u,
};

typedef struct {
    const void* input;
    const float* weight;
    const float* bias;
    void* output;
    size_t area;
    size_t values_per_group;
    size_t sample_stride;
    uint32_t channels;
    uint32_t groups;
    uint32_t channels_per_group;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    float epsilon;
    uint32_t input_dtype;
    uint32_t output_dtype;
    int32_t output_minimum;
    int32_t output_maximum;
} VxQGroupNormParallelContext;

static void vx_qgroupnorm_parallel_worker(void* opaque, int begin, int end) {
    VxQGroupNormParallelContext* context =
        (VxQGroupNormParallelContext*)opaque;
    for (int task = begin; task < end; task++) {
        const uint32_t sample = (uint32_t)task / context->groups;
        const uint32_t group = (uint32_t)task % context->groups;
        const size_t sample_offset = (size_t)sample * context->sample_stride;
        const uint32_t channel_start = group * context->channels_per_group;
        float sum = 0.0f;
        for (size_t spatial = 0; spatial < context->area; spatial++) {
            const size_t offset = sample_offset +
                spatial * context->channels + channel_start;
            for (uint32_t local = 0; local < context->channels_per_group;
                 local++) {
                const float raw = (float)(vx_w8a8_byte_value(
                    context->input, context->input_dtype, offset + local) -
                    context->input_zero_point);
                sum = sum + raw;
            }
        }
        {
            const float mean_raw = sum / (float)context->values_per_group;
            float centered_square_sum = 0.0f;
            for (size_t spatial = 0; spatial < context->area; spatial++) {
                const size_t offset = sample_offset +
                    spatial * context->channels + channel_start;
                for (uint32_t local = 0; local < context->channels_per_group;
                     local++) {
                    const float raw = (float)(vx_w8a8_byte_value(
                        context->input, context->input_dtype, offset + local) -
                        context->input_zero_point);
                    const float centered = raw - mean_raw;
                    centered_square_sum =
                        centered_square_sum + centered * centered;
                }
            }
            {
                float raw_variance = centered_square_sum /
                    (float)context->values_per_group;
                if (raw_variance < 0.0f) raw_variance = 0.0f;
                {
                    const float scaled_variance =
                        raw_variance * context->input_scale;
                    float real_variance =
                        scaled_variance * context->input_scale;
                    if (real_variance < 0.0f) real_variance = 0.0f;
                    {
                        const float inverse_stddev = 1.0f /
                            sqrtf(real_variance + context->epsilon);
                        for (size_t spatial = 0; spatial < context->area;
                             spatial++) {
                            const size_t offset = sample_offset +
                                spatial * context->channels + channel_start;
                            for (uint32_t local = 0;
                                 local < context->channels_per_group; local++) {
                                const uint32_t channel = channel_start + local;
                                const float raw = (float)(
                                    vx_w8a8_byte_value(context->input,
                                        context->input_dtype, offset + local) -
                                    context->input_zero_point);
                                const float scaled =
                                    (raw - mean_raw) * context->input_scale;
                                const float normalized = scaled * inverse_stddev;
                                const float affine = normalized *
                                    vx_w8a8_affine_f32_at(
                                        context->weight, channel) +
                                    vx_w8a8_affine_f32_at(
                                        context->bias, channel);
                                const float output_scaled =
                                    affine / context->output_scale;
                                const float transformed = output_scaled +
                                    (float)context->output_zero_point;
                                const int32_t quantized =
                                    vx_w8a8_requantize(transformed,
                                        context->output_minimum,
                                        context->output_maximum,
                                        context->output_zero_point);
                                vx_w8a8_store_byte(context->output,
                                    context->output_dtype, offset + local,
                                    quantized);
                            }
                        }
                    }
                }
            }
        }
    }
}

#if VX_QGROUPNORM_X86_AVX2
static VX_QGROUPNORM_TARGET_AVX2 __m128 vx_qgroupnorm_load_group4(
        const VxQGroupNormParallelContext* context, size_t spatial_offset,
        uint32_t channel_start, uint32_t local) {
    return _mm_setr_ps(
        (float)(vx_w8a8_byte_value(context->input,
            context->input_dtype, spatial_offset + channel_start + local) -
            context->input_zero_point),
        (float)(vx_w8a8_byte_value(context->input,
            context->input_dtype, spatial_offset + channel_start +
            context->channels_per_group + local) -
            context->input_zero_point),
        (float)(vx_w8a8_byte_value(context->input,
            context->input_dtype, spatial_offset + channel_start +
            2u * context->channels_per_group + local) -
            context->input_zero_point),
        (float)(vx_w8a8_byte_value(context->input,
            context->input_dtype, spatial_offset + channel_start +
            3u * context->channels_per_group + local) -
            context->input_zero_point));
}

static VX_QGROUPNORM_TARGET_AVX2 __m128 vx_qgroupnorm_load_affine4(
        const float* values, uint32_t channel_start,
        uint32_t channels_per_group, uint32_t local) {
    return _mm_setr_ps(
        vx_w8a8_affine_f32_at(values, channel_start + local),
        vx_w8a8_affine_f32_at(values,
            channel_start + channels_per_group + local),
        vx_w8a8_affine_f32_at(values,
            channel_start + 2u * channels_per_group + local),
        vx_w8a8_affine_f32_at(values,
            channel_start + 3u * channels_per_group + local));
}

static VX_QGROUPNORM_TARGET_AVX2 __m128i vx_qgroupnorm_quantize4(
        __m128 transformed, int32_t minimum, int32_t maximum,
        int32_t nan_value) {
    const __m128 nan_mask = _mm_cmpunord_ps(transformed, transformed);
    const __m128 clamped = _mm_min_ps(
        _mm_max_ps(transformed, _mm_set1_ps((float)minimum)),
        _mm_set1_ps((float)maximum));
    const __m128 rounded = _mm_round_ps(
        clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return _mm_blendv_epi8(_mm_cvttps_epi32(rounded),
        _mm_set1_epi32(nan_value), _mm_castps_si128(nan_mask));
}

/* Four groups occupy four independent SIMD lanes.  For every lane, the
 * spatial/local traversal and each F32 addition occur in the same order as
 * the portable group-major kernel; only mutually independent groups execute
 * together.  This is especially effective for TinyReceipt's C/G=3,6,10
 * channels-last GroupNorm shapes. */
static VX_QGROUPNORM_TARGET_AVX2 void vx_qgroupnorm_group4_worker(
        void* opaque, int begin, int end) {
    VxQGroupNormParallelContext* context =
        (VxQGroupNormParallelContext*)opaque;
    const uint32_t blocks_per_sample = context->groups / 4u;
    const __m128 input_scales = _mm_set1_ps(context->input_scale);
    const __m128 output_scales = _mm_set1_ps(context->output_scale);
    const __m128 output_zeros = _mm_set1_ps(
        (float)context->output_zero_point);
    for (int task = begin; task < end; task++) {
        const uint32_t sample = (uint32_t)task / blocks_per_sample;
        const uint32_t block = (uint32_t)task % blocks_per_sample;
        const uint32_t channel_start =
            block * 4u * context->channels_per_group;
        const size_t sample_offset =
            (size_t)sample * context->sample_stride;
        __m128 sum = _mm_setzero_ps();
        __m128 centered_square_sum = _mm_setzero_ps();
        __m128 mean_raw;
        __m128 inverse_stddev;
        for (size_t spatial = 0; spatial < context->area; spatial++) {
            const size_t offset = sample_offset +
                spatial * context->channels;
            for (uint32_t local = 0; local < context->channels_per_group;
                    local++) {
                sum = _mm_add_ps(sum, vx_qgroupnorm_load_group4(
                    context, offset, channel_start, local));
            }
        }
        mean_raw = _mm_div_ps(sum,
            _mm_set1_ps((float)context->values_per_group));
        for (size_t spatial = 0; spatial < context->area; spatial++) {
            const size_t offset = sample_offset +
                spatial * context->channels;
            for (uint32_t local = 0; local < context->channels_per_group;
                    local++) {
                const __m128 centered = _mm_sub_ps(
                    vx_qgroupnorm_load_group4(
                        context, offset, channel_start, local),
                    mean_raw);
                centered_square_sum = _mm_add_ps(centered_square_sum,
                    _mm_mul_ps(centered, centered));
            }
        }
        {
            __m128 raw_variance = _mm_div_ps(centered_square_sum,
                _mm_set1_ps((float)context->values_per_group));
            __m128 real_variance;
            raw_variance = _mm_max_ps(raw_variance, _mm_setzero_ps());
            real_variance = _mm_mul_ps(
                _mm_mul_ps(raw_variance, input_scales), input_scales);
            real_variance = _mm_max_ps(real_variance, _mm_setzero_ps());
            inverse_stddev = _mm_div_ps(_mm_set1_ps(1.0f),
                _mm_sqrt_ps(_mm_add_ps(real_variance,
                    _mm_set1_ps(context->epsilon))));
        }
        for (size_t spatial = 0; spatial < context->area; spatial++) {
            const size_t offset = sample_offset +
                spatial * context->channels;
            for (uint32_t local = 0; local < context->channels_per_group;
                    local++) {
                const __m128 raw = vx_qgroupnorm_load_group4(
                    context, offset, channel_start, local);
                const __m128 normalized = _mm_mul_ps(
                    _mm_mul_ps(_mm_sub_ps(raw, mean_raw), input_scales),
                    inverse_stddev);
                const __m128 affine = _mm_add_ps(
                    _mm_mul_ps(normalized, vx_qgroupnorm_load_affine4(
                        context->weight, channel_start,
                        context->channels_per_group, local)),
                    vx_qgroupnorm_load_affine4(context->bias, channel_start,
                        context->channels_per_group, local));
                const __m128 transformed = _mm_add_ps(
                    _mm_div_ps(affine, output_scales), output_zeros);
                int32_t quantized[4];
                _mm_storeu_si128((__m128i*)quantized,
                    vx_qgroupnorm_quantize4(transformed,
                        context->output_minimum, context->output_maximum,
                        context->output_zero_point));
                for (uint32_t lane = 0; lane < 4u; lane++) {
                    vx_w8a8_store_byte(context->output,
                        context->output_dtype,
                        offset + channel_start +
                            lane * context->channels_per_group + local,
                        quantized[lane]);
                }
            }
        }
    }
}
#endif

int vx_qgroupnorm_i8u8_native_validated(
        const void* input, const float* weight, const float* bias, void* output,
        uint32_t batch, uint32_t height, uint32_t width, uint32_t channels,
        uint32_t groups, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point, float epsilon,
        uint32_t input_dtype, uint32_t output_dtype) {
    VxQGroupNormParallelContext context;
    uint64_t area;
    uint64_t elements;
    uint64_t tasks;
    int threads;
    if (!input || !weight || !bias || !output || !batch || !height || !width ||
        !channels || !groups || channels % groups) {
        return qgroupnorm_i8u8(input, weight, bias, output, batch, height, width,
            channels, groups, input_scale, input_zero_point, output_scale,
            output_zero_point, epsilon, input_dtype, output_dtype);
    }
    area = (uint64_t)height * width;
    elements = area * channels;
    if (batch && elements > UINT64_MAX / batch) {
        return qgroupnorm_i8u8(input, weight, bias, output, batch, height, width,
            channels, groups, input_scale, input_zero_point, output_scale,
            output_zero_point, epsilon, input_dtype, output_dtype);
    }
    elements *= batch;
    tasks = (uint64_t)batch * groups;
    threads = vx_kernels_thread_count();
    if (tasks > INT_MAX || area > SIZE_MAX || elements > SIZE_MAX) {
        return qgroupnorm_i8u8(input, weight, bias, output, batch, height, width,
            channels, groups, input_scale, input_zero_point, output_scale,
            output_zero_point, epsilon, input_dtype, output_dtype);
    }
    context = (VxQGroupNormParallelContext){
        .input = input, .weight = weight, .bias = bias, .output = output,
        .area = (size_t)area,
        .values_per_group = (size_t)area * (channels / groups),
        .sample_stride = (size_t)area * channels,
        .channels = channels, .groups = groups,
        .channels_per_group = channels / groups,
        .input_scale = input_scale,
        .input_zero_point = input_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .epsilon = epsilon, .input_dtype = input_dtype,
        .output_dtype = output_dtype,
        .output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0,
        .output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255,
    };
    /* ISA selection and pool selection are independent.  Vectorization is a
     * property of the shape and the CPU, so it must not be withdrawn when the
     * pool is unavailable or when the decomposition cannot fill every worker;
     * a single-threaded call still wants the group-quad kernel. */
#if VX_QGROUPNORM_X86_AVX2
    if (groups % 4u == 0u && vx_kernel_platform()->has_avx2) {
        const uint64_t group4_tasks = (uint64_t)batch * (groups / 4u);
        if (group4_tasks <= INT_MAX) {
            /* Four groups share one register, so a quad task carries roughly
             * four scalar group tasks.  Spending that width is worth more than
             * the extra workers the scalar split could have occupied. */
            if (elements >= VX_QGROUPNORM_PARALLEL_ELEMENTS && threads > 1 &&
                group4_tasks > 1u) {
                vx_kernels_parallel_for((int)group4_tasks, 1,
                                        vx_qgroupnorm_group4_worker, &context);
            } else {
                vx_qgroupnorm_group4_worker(&context, 0, (int)group4_tasks);
            }
            return 1;
        }
    }
#endif
    if (elements < VX_QGROUPNORM_PARALLEL_ELEMENTS || threads <= 1 ||
        tasks < (uint64_t)threads) {
        return qgroupnorm_i8u8(input, weight, bias, output, batch, height, width,
            channels, groups, input_scale, input_zero_point, output_scale,
            output_zero_point, epsilon, input_dtype, output_dtype);
    }
    vx_kernels_parallel_for((int)tasks, 1,
                            vx_qgroupnorm_parallel_worker, &context);
    return 1;
}
