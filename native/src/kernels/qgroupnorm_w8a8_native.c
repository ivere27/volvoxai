/*
 * Native CPU parallelism for canonical W8A8 QGroupNorm.
 *
 * qgroupnorm_i8u8() remains the authoritative, fully validating portable
 * implementation and the sole WASM path.  The physical native runtime has
 * already validated this call, so independent [batch, group] reductions can
 * run on the bound kernel pool.  Every reduction keeps the portable loop and
 * F32 operation order, preserving byte-exact output.
 */
#include "quant_cpu_isa.h"
#include "w8a8_affine.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_QGROUPNORM_X86_AVX2 1
#define VX_QGROUPNORM_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QGROUPNORM_X86_AVX2 0
#define VX_QGROUPNORM_TARGET_AVX2
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define VX_QGROUPNORM_ARM_NEON 1
#else
#define VX_QGROUPNORM_ARM_NEON 0
#endif

extern int qgroupnorm_i8u8(const void* input, const float* weight,
        const float* bias, void* output, uint32_t batch, uint32_t height,
        uint32_t width, uint32_t channels, uint32_t groups, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype);

enum {
    /* Below this size, pool wake-up is normally more expensive than the
     * scalar kernel.  Large image-encoder GroupNorm calls exceed it. */
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
 * together.  This is especially effective for narrow C/G=3,6,10
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

#if VX_QGROUPNORM_ARM_NEON
enum {
    VX_QGROUPNORM_NEON_MAX_GROUPS = 512u,
    VX_QGROUPNORM_NEON_SPATIAL_TILE = 16u,
};

/* SafeTensors permits F32 affine storage at a byte-aligned offset.  memcpy is
 * both defined for that pointer and lowered by AArch64 Clang to one unaligned
 * vector load. */
static inline float32x4_t vx_qgroupnorm_neon_load_affine4(
        const float* values, uint32_t channel) {
    float32x4_t loaded;
    memcpy(&loaded, (const unsigned char*)values +
        (size_t)channel * sizeof(float), sizeof(loaded));
    return loaded;
}

static inline float32x4_t vx_qgroupnorm_neon_load_stat4(
        const float* values, uint32_t channel,
        uint32_t channels_per_group) {
    float32x4_t loaded = vdupq_n_f32(
        values[channel / channels_per_group]);
    loaded = vsetq_lane_f32(
        values[(channel + 1u) / channels_per_group], loaded, 1);
    loaded = vsetq_lane_f32(
        values[(channel + 2u) / channels_per_group], loaded, 2);
    return vsetq_lane_f32(
        values[(channel + 3u) / channels_per_group], loaded, 3);
}

static inline float32x4_t vx_qgroupnorm_neon_load_raw4(
        const VxQGroupNormParallelContext* context, size_t index) {
    uint32_t packed;
    memcpy(&packed, (const unsigned char*)context->input + index,
        sizeof(packed));
    if (context->input_dtype == VX_DTYPE_I8) {
        const int8x8_t bytes = vreinterpret_s8_u8(vcreate_u8((uint64_t)packed));
        const int16x8_t words = vmovl_s8(bytes);
        return vcvtq_f32_s32(vmovl_s16(vget_low_s16(words)));
    }
    {
        const uint8x8_t bytes = vcreate_u8((uint64_t)packed);
        const uint16x8_t words = vmovl_u8(bytes);
        return vcvtq_f32_u32(vmovl_u16(vget_low_u16(words)));
    }
}

typedef struct {
    float32x4_t low[3];
    float32x4_t high[3];
} VxQGroupNormNeonCpg3Rows;

/* vld3 reads eight contiguous three-channel groups and deinterleaves the
 * three local-channel streams.  That replaces twenty-four scalar byte loads,
 * conversions and INS instructions with one structured load plus widening. */
static inline VxQGroupNormNeonCpg3Rows vx_qgroupnorm_neon_load_cpg3_groups8(
        const VxQGroupNormParallelContext* context, size_t index) {
    VxQGroupNormNeonCpg3Rows result;
    const float32x4_t zero = vdupq_n_f32((float)context->input_zero_point);
    if (context->input_dtype == VX_DTYPE_I8) {
        const int8x8x3_t packed = vld3_s8(
            (const int8_t*)context->input + index);
        for (uint32_t local = 0; local < 3u; local++) {
            const int16x8_t words = vmovl_s8(packed.val[local]);
            result.low[local] = vsubq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(words))), zero);
            result.high[local] = vsubq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(words))), zero);
        }
    } else {
        const uint8x8x3_t packed = vld3_u8(
            (const uint8_t*)context->input + index);
        for (uint32_t local = 0; local < 3u; local++) {
            const uint16x8_t words = vmovl_u8(packed.val[local]);
            result.low[local] = vsubq_f32(
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(words))), zero);
            result.high[local] = vsubq_f32(
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(words))), zero);
        }
    }
    return result;
}

static inline int32x4_t vx_qgroupnorm_neon_quantize4(
        float32x4_t transformed, int32_t minimum, int32_t maximum,
        int32_t nan_value) {
    const uint32x4_t nan_mask = vmvnq_u32(
        vceqq_f32(transformed, transformed));
    const float32x4_t clamped = vminq_f32(
        vmaxq_f32(transformed, vdupq_n_f32((float)minimum)),
        vdupq_n_f32((float)maximum));
    /* FCVTNS is round-to-nearest, ties-to-even and ignores the ambient
     * rounding mode, matching vx_w8a8_round_ties_even(). */
    const int32x4_t rounded = vcvtnq_s32_f32(clamped);
    return vbslq_s32(nan_mask, vdupq_n_s32(nan_value), rounded);
}

static inline void vx_qgroupnorm_neon_store4(
        VxQGroupNormParallelContext* context, size_t index,
        int32x4_t quantized) {
    uint32_t packed;
    if (context->output_dtype == VX_DTYPE_I8) {
        const int16x8_t words = vcombine_s16(vqmovn_s32(quantized),
            vdup_n_s16(0));
        const int8x8_t bytes = vqmovn_s16(words);
        packed = vget_lane_u32(vreinterpret_u32_s8(bytes), 0);
    } else {
        const uint16x8_t words = vcombine_u16(vqmovun_s32(quantized),
            vdup_n_u16(0));
        const uint8x8_t bytes = vqmovn_u16(words);
        packed = vget_lane_u32(vreinterpret_u32_u8(bytes), 0);
    }
    memcpy((unsigned char*)context->output + index, &packed, sizeof(packed));
}

/* Eight independent C/G=3 groups occupy two F32 vectors.  The row tile is the
 * important difference from a group-major implementation: all 32 TinyReceipt
 * groups revisit sixteen resident rows before advancing, instead of streaming
 * the multi-megabyte activation once per group. */
static void vx_qgroupnorm_neon_sample_worker(void* opaque, int begin, int end) {
    VxQGroupNormParallelContext* context =
        (VxQGroupNormParallelContext*)opaque;
    float sums[VX_QGROUPNORM_NEON_MAX_GROUPS];
    float squares[VX_QGROUPNORM_NEON_MAX_GROUPS];
    float means[VX_QGROUPNORM_NEON_MAX_GROUPS];
    float inverses[VX_QGROUPNORM_NEON_MAX_GROUPS];
    const float32x4_t input_scales = vdupq_n_f32(context->input_scale);
    const float32x4_t output_scales = vdupq_n_f32(context->output_scale);
    const float32x4_t output_zeros = vdupq_n_f32(
        (float)context->output_zero_point);
    const float32x4_t value_count = vdupq_n_f32(
        (float)context->values_per_group);
    for (int task = begin; task < end; task++) {
        const uint32_t sample = (uint32_t)task;
        const size_t sample_offset =
            (size_t)sample * context->sample_stride;
        for (uint32_t group = 0; group < context->groups; group++)
            sums[group] = 0.0f;
        for (size_t tile = 0; tile < context->area;
                tile += VX_QGROUPNORM_NEON_SPATIAL_TILE) {
            size_t limit = tile + VX_QGROUPNORM_NEON_SPATIAL_TILE;
            if (limit > context->area) limit = context->area;
            for (uint32_t group = 0; group < context->groups; group += 8u) {
                float32x4_t total_low = vld1q_f32(sums + group);
                float32x4_t total_high = vld1q_f32(sums + group + 4u);
                const uint32_t channel_start =
                    group * context->channels_per_group;
                for (size_t spatial = tile; spatial < limit; spatial++) {
                    const size_t offset = sample_offset +
                        spatial * context->channels + channel_start;
                    const VxQGroupNormNeonCpg3Rows rows =
                        vx_qgroupnorm_neon_load_cpg3_groups8(context, offset);
                    total_low = vaddq_f32(total_low, rows.low[0]);
                    total_high = vaddq_f32(total_high, rows.high[0]);
                    total_low = vaddq_f32(total_low, rows.low[1]);
                    total_high = vaddq_f32(total_high, rows.high[1]);
                    total_low = vaddq_f32(total_low, rows.low[2]);
                    total_high = vaddq_f32(total_high, rows.high[2]);
                }
                vst1q_f32(sums + group, total_low);
                vst1q_f32(sums + group + 4u, total_high);
            }
        }
        for (uint32_t group = 0; group < context->groups; group += 4u) {
            vst1q_f32(means + group,
                vdivq_f32(vld1q_f32(sums + group), value_count));
            vst1q_f32(squares + group, vdupq_n_f32(0.0f));
        }
        for (size_t tile = 0; tile < context->area;
                tile += VX_QGROUPNORM_NEON_SPATIAL_TILE) {
            size_t limit = tile + VX_QGROUPNORM_NEON_SPATIAL_TILE;
            if (limit > context->area) limit = context->area;
            for (uint32_t group = 0; group < context->groups; group += 8u) {
                float32x4_t total_low = vld1q_f32(squares + group);
                float32x4_t total_high = vld1q_f32(squares + group + 4u);
                const float32x4_t mean_low = vld1q_f32(means + group);
                const float32x4_t mean_high = vld1q_f32(means + group + 4u);
                const uint32_t channel_start =
                    group * context->channels_per_group;
                for (size_t spatial = tile; spatial < limit; spatial++) {
                    const size_t offset = sample_offset +
                        spatial * context->channels + channel_start;
                    const VxQGroupNormNeonCpg3Rows rows =
                        vx_qgroupnorm_neon_load_cpg3_groups8(context, offset);
                    for (uint32_t local = 0; local < 3u; local++) {
                        const float32x4_t centered_low = vsubq_f32(
                            rows.low[local], mean_low);
                        const float32x4_t centered_high = vsubq_f32(
                            rows.high[local], mean_high);
                        total_low = vmlaq_f32(
                            total_low, centered_low, centered_low);
                        total_high = vmlaq_f32(
                            total_high, centered_high, centered_high);
                    }
                }
                vst1q_f32(squares + group, total_low);
                vst1q_f32(squares + group + 4u, total_high);
            }
        }
        for (uint32_t group = 0; group < context->groups; group += 4u) {
            float32x4_t raw_variance = vdivq_f32(
                vld1q_f32(squares + group), value_count);
            float32x4_t real_variance;
            raw_variance = vmaxq_f32(raw_variance, vdupq_n_f32(0.0f));
            real_variance = vmulq_f32(
                vmulq_f32(raw_variance, input_scales), input_scales);
            real_variance = vmaxq_f32(real_variance, vdupq_n_f32(0.0f));
            vst1q_f32(inverses + group, vdivq_f32(vdupq_n_f32(1.0f),
                vsqrtq_f32(vaddq_f32(real_variance,
                    vdupq_n_f32(context->epsilon)))));
        }
        for (size_t tile = 0; tile < context->area;
                tile += VX_QGROUPNORM_NEON_SPATIAL_TILE) {
            size_t limit = tile + VX_QGROUPNORM_NEON_SPATIAL_TILE;
            if (limit > context->area) limit = context->area;
            for (uint32_t channel = 0; channel < context->channels;
                    channel += 4u) {
                const float32x4_t mean = vx_qgroupnorm_neon_load_stat4(
                    means, channel, context->channels_per_group);
                const float32x4_t inverse = vx_qgroupnorm_neon_load_stat4(
                    inverses, channel, context->channels_per_group);
                const float32x4_t gain = vx_qgroupnorm_neon_load_affine4(
                    context->weight, channel);
                const float32x4_t bias = vx_qgroupnorm_neon_load_affine4(
                    context->bias, channel);
                for (size_t spatial = tile; spatial < limit; spatial++) {
                    const size_t index = sample_offset +
                        spatial * context->channels + channel;
                    const float32x4_t raw = vsubq_f32(
                        vx_qgroupnorm_neon_load_raw4(context, index),
                        vdupq_n_f32((float)context->input_zero_point));
                    const float32x4_t scaled = vmulq_f32(
                        vsubq_f32(raw, mean), input_scales);
                    const float32x4_t normalized = vmulq_f32(scaled, inverse);
                    const float32x4_t affine = vmlaq_f32(
                        bias, normalized, gain);
                    const float32x4_t transformed = vaddq_f32(
                        vdivq_f32(affine, output_scales), output_zeros);
                    vx_qgroupnorm_neon_store4(context, index,
                        vx_qgroupnorm_neon_quantize4(transformed,
                            context->output_minimum, context->output_maximum,
                            context->output_zero_point));
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
#if VX_QGROUPNORM_ARM_NEON
    if (channels / groups == 3u && groups % 8u == 0u &&
        groups <= VX_QGROUPNORM_NEON_MAX_GROUPS && channels % 4u == 0u &&
        vx_kernel_platform()->has_neon) {
        if (elements >= VX_QGROUPNORM_PARALLEL_ELEMENTS && threads > 1 &&
            batch > 1u) {
            vx_kernels_parallel_for((int)batch, 1,
                vx_qgroupnorm_neon_sample_worker, &context);
        } else {
            vx_qgroupnorm_neon_sample_worker(&context, 0, (int)batch);
        }
        return 1;
    }
#endif
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
