#include "quant_cpu_opt.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "volvoxai_enums.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_QUANT_X86_AVX2 1
#define VX_QUANT_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QUANT_X86_AVX2 0
#define VX_QUANT_TARGET_AVX2
#endif

#if VX_QUANT_X86_AVX2
static VX_QUANT_TARGET_AVX2 __m256i vx_quantize_transformed8_avx2(
        __m256 transformed, int zero_point, int minimum, int maximum) {
    const __m256 nan_mask = _mm256_cmp_ps(
        transformed, transformed, _CMP_UNORD_Q);
    const __m256 clamped = _mm256_min_ps(
        _mm256_max_ps(transformed, _mm256_set1_ps((float)minimum)),
        _mm256_set1_ps((float)maximum));
    const __m256 rounded = _mm256_round_ps(
        clamped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256i quantized = _mm256_cvttps_epi32(rounded);
    return _mm256_blendv_epi8(
        quantized, _mm256_set1_epi32(zero_point),
        _mm256_castps_si256(nan_mask));
}

static VX_QUANT_TARGET_AVX2 void vx_store_quantized8_avx2(
        void* output, uint32_t output_dtype, uint32_t index,
        __m256i quantized) {
    const __m128i low = _mm256_castsi256_si128(quantized);
    const __m128i high = _mm256_extracti128_si256(quantized, 1);
    __m128i bytes;
    if (output_dtype == VX_DTYPE_I8) {
        const __m128i values16 = _mm_packs_epi32(low, high);
        bytes = _mm_packs_epi16(values16, values16);
    } else {
        const __m128i values16 = _mm_packus_epi32(low, high);
        bytes = _mm_packus_epi16(values16, values16);
    }
    {
        const uint64_t packed = (uint64_t)_mm_cvtsi128_si64(bytes);
        memcpy((uint8_t*)output + index, &packed, sizeof(packed));
    }
}

static VX_QUANT_TARGET_AVX2 __m256i vx_load_quantized8_i32_avx2(
        const void* input, uint32_t input_dtype, uint32_t index) {
    const __m128i bytes = _mm_loadl_epi64(
        (const __m128i*)((const uint8_t*)input + index));
    return input_dtype == VX_DTYPE_I8
        ? _mm256_cvtepi8_epi32(bytes) : _mm256_cvtepu8_epi32(bytes);
}

static VX_QUANT_TARGET_AVX2 uint32_t vx_quantize_linear_typed_avx2(
        const float* input, float scale, int zero_point, void* output,
        uint32_t output_dtype, uint32_t elements, int minimum, int maximum) {
    const __m256 scales = _mm256_set1_ps(scale);
    const __m256 zeros = _mm256_set1_ps((float)zero_point);
    uint32_t index = 0;
    for (; index + 8u <= elements; index += 8u) {
        const __m256 values = _mm256_loadu_ps(input + index);
        const __m256 transformed = _mm256_add_ps(
            _mm256_div_ps(values, scales), zeros);
        vx_store_quantized8_avx2(output, output_dtype, index,
            vx_quantize_transformed8_avx2(
                transformed, zero_point, minimum, maximum));
    }
    return index;
}

static VX_QUANT_TARGET_AVX2 int vx_qadd_i8u8_avx2(
        const void* a, const void* b, void* output, uint32_t elements,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype,
        uint32_t relu) {
    const int minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    const __m256 a_scales = _mm256_set1_ps(a_scale);
    const __m256 b_scales = _mm256_set1_ps(b_scale);
    const __m256 output_scales = _mm256_set1_ps(output_scale);
    const __m256 output_zeros = _mm256_set1_ps((float)output_zero_point);
    const __m256i a_zeros = _mm256_set1_epi32(a_zero_point);
    const __m256i b_zeros = _mm256_set1_epi32(b_zero_point);
    const __m256i relu_minimum = _mm256_set1_epi32(output_zero_point);
    __m256i relu_maximum = _mm256_set1_epi32(maximum);
    uint32_t index = 0;
    if (relu >= 2u) {
        const __m256 relu6 = _mm256_set1_ps(
            6.0f / output_scale + (float)output_zero_point);
        relu_maximum = vx_quantize_transformed8_avx2(
            relu6, 0, minimum, maximum);
    }
    for (; index + 8u <= elements; index += 8u) {
        const __m256i a_values = _mm256_sub_epi32(
            vx_load_quantized8_i32_avx2(a, a_dtype, index), a_zeros);
        const __m256i b_values = _mm256_sub_epi32(
            vx_load_quantized8_i32_avx2(b, b_dtype, index), b_zeros);
        const __m256 a_real = _mm256_mul_ps(
            _mm256_cvtepi32_ps(a_values), a_scales);
        const __m256 b_real = _mm256_mul_ps(
            _mm256_cvtepi32_ps(b_values), b_scales);
        const __m256 sum = _mm256_add_ps(a_real, b_real);
        const __m256 transformed = _mm256_add_ps(
            _mm256_div_ps(sum, output_scales), output_zeros);
        __m256i quantized = vx_quantize_transformed8_avx2(
            transformed, output_zero_point, minimum, maximum);
        if (relu) {
            quantized = _mm256_max_epi32(quantized, relu_minimum);
            if (relu >= 2u)
                quantized = _mm256_min_epi32(quantized, relu_maximum);
        }
        vx_store_quantized8_avx2(output, output_dtype, index, quantized);
    }
    return index == elements;
}
#endif

#if VX_QUANT_X86_AVX2
/* Byte dequantization is exactly reproducible in single precision.  The
 * portable kernel computes ((double)x - (double)zp) * (double)scale and then
 * rounds once to F32.  For a byte input |x - zp| < 2^9, so the exact product
 * needs at most 9 + 24 = 33 significant bits and is therefore held exactly in
 * double; rounding it to F32 yields the correctly rounded product.  An F32
 * multiply rounds the same exact product the same way, so the vector result is
 * bit-identical rather than merely close. */
static VX_QUANT_TARGET_AVX2 uint32_t vx_dequantize_linear_typed_avx2(
        const void* input, uint32_t input_dtype, float scale, int zero_point,
        float* output, uint32_t elements) {
    const __m256 scales = _mm256_set1_ps(scale);
    const __m256i zeros = _mm256_set1_epi32(zero_point);
    uint32_t index = 0;
    for (; index + 8u <= elements; index += 8u) {
        const __m256i values = _mm256_sub_epi32(
            vx_load_quantized8_i32_avx2(input, input_dtype, index), zeros);
        _mm256_storeu_ps(output + index,
            _mm256_mul_ps(_mm256_cvtepi32_ps(values), scales));
    }
    return index;
}
#endif

uint32_t vx_dequantize_linear_typed_native_prefix(
        const void* input, uint32_t input_dtype, float scale, int zero_point,
        float* output, uint32_t elements) {
#if VX_QUANT_X86_AVX2
    if (input && output && elements >= 8u &&
        (input_dtype == VX_DTYPE_I8 || input_dtype == VX_DTYPE_U8) &&
        vx_kernel_platform()->has_avx2) {
        return vx_dequantize_linear_typed_avx2(input, input_dtype, scale,
                                               zero_point, output, elements);
    }
#else
    (void)input;
    (void)input_dtype;
    (void)scale;
    (void)zero_point;
    (void)output;
    (void)elements;
#endif
    return 0;
}

uint32_t vx_quantize_linear_typed_native_prefix(
        const float* input, float scale, int zero_point, void* output,
        uint32_t output_dtype, uint32_t elements, int minimum, int maximum) {
#if VX_QUANT_X86_AVX2
    if (input && output && elements >= 8u && vx_kernel_platform()->has_avx2)
        return vx_quantize_linear_typed_avx2(
            input, scale, zero_point, output, output_dtype, elements,
            minimum, maximum);
#else
    (void)input;
    (void)scale;
    (void)zero_point;
    (void)output;
    (void)output_dtype;
    (void)elements;
    (void)minimum;
    (void)maximum;
#endif
    return 0;
}

int vx_qadd_i8u8_native_try(
        const void* a, const void* b, void* output, uint32_t elements,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype,
        uint32_t relu) {
#if VX_QUANT_X86_AVX2
    if (a && b && output && elements >= 8u &&
        elements % 8u == 0u && vx_kernel_platform()->has_avx2) {
        return vx_qadd_i8u8_avx2(
            a, b, output, elements, a_scale, a_zero_point,
            b_scale, b_zero_point, output_scale, output_zero_point,
            a_dtype, b_dtype, output_dtype, relu);
    }
#else
    (void)a;
    (void)b;
    (void)output;
    (void)elements;
    (void)a_scale;
    (void)a_zero_point;
    (void)b_scale;
    (void)b_zero_point;
    (void)output_scale;
    (void)output_zero_point;
    (void)a_dtype;
    (void)b_dtype;
    (void)output_dtype;
    (void)relu;
#endif
    return 0;
}

typedef struct {
    const float* data;
    long numel;
} VXFloatView;

static inline float relu6_apply(float x, int relu);
static inline signed char requantize_i8(float v, float out_scale, int out_zp, int relu);

#if defined(__AVX2__)
static inline void store_8xi8(signed char* dst, __m256i v);
static inline int dot_i8_i16_avx2(const signed char* x, const int16_t* w, int c, int input_zp);
static inline __m256i dup_i16_pair_epi32(int lo, int hi);
static inline __m256i dup_u8_pairzero_epi32(int lo, int hi);
static inline __m256i maddubs_pair_i32(__m256i x, __m256i w, __m256i ones16);
static inline __m256 requantize8_f32(__m256i acc, __m256 scale, __m256 bias, int relu);
static inline __m256i requantize8_i8_acc(__m256i acc, __m256 qscale, __m256 qbias,
                                         int relu, __m256 qlo, __m256 qhi,
                                         __m256i qmin, __m256i qmax);
static int64_t vx_qlinear_dot_i8_avx2(const signed char* x, const void* weight,
                                       int weight_dtype, int d_in,
                                       int input_zp, int weight_zp, int* processed);
#endif

static void vx_qconv2d_pointwise_range(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                              const signed char* wpack8z,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int out_c, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu,
                              int p_begin, int p_end) {
    VXFloatView wscale_view = { wscale_data, wscale_numel };
    VXFloatView bias_view = { bias_data, 0 };
    const VXFloatView* wscale = &wscale_view;
    const VXFloatView* bias = bias_data ? &bias_view : NULL;
#if defined(__AVX2__)
    int n_kpair = c / 2;
#endif
#if defined(__AVX2__)
    if ((yq || yf) && wi && wpack && wpack8z && input_zp == -128 && out_c >= 8) {
        __m256i qmin = _mm256_set1_epi32(-128), qmax = _mm256_set1_epi32(127);
        __m256i ones16 = _mm256_set1_epi16(1);
        __m256 inv_out = _mm256_set1_ps(out_scale > 0.0f ? 1.0f / out_scale : 1.0f);
        __m256 out_zpf = _mm256_set1_ps((float)out_zp);
        __m256 input_scale_v = _mm256_set1_ps(input_scale);
        __m256 qlo = out_zpf;
        __m256 qhi = _mm256_set1_ps(out_scale > 0.0f ? 6.0f / out_scale + (float)out_zp : (float)out_zp);
        int p = p_begin;
        for (; p + 4 <= p_end; p += 4) {
            const signed char* x0p = xq + (long)(p + 0) * c;
            const signed char* x1p = xq + (long)(p + 1) * c;
            const signed char* x2p = xq + (long)(p + 2) * c;
            const signed char* x3p = xq + (long)(p + 3) * c;
            int oc = 0;
            for (; oc + 16 <= out_c; oc += 16) {
                __m256i a0l = _mm256_setzero_si256(), a0h = _mm256_setzero_si256();
                __m256i a1l = _mm256_setzero_si256(), a1h = _mm256_setzero_si256();
                __m256i a2l = _mm256_setzero_si256(), a2h = _mm256_setzero_si256();
                __m256i a3l = _mm256_setzero_si256(), a3h = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    long off0 = ((long)(oc >> 3) * n_kpair + (k >> 1)) * 32;
                    __m256i wl = _mm256_loadu_si256((const __m256i*)(const void*)(wpack8z + off0));
                    __m256i wh = _mm256_loadu_si256((const __m256i*)(const void*)(wpack8z + off0 + (long)n_kpair * 32));
                    __m256i x0v = dup_u8_pairzero_epi32((int)x0p[k] + 128, (int)x0p[k + 1] + 128);
                    __m256i x1v = dup_u8_pairzero_epi32((int)x1p[k] + 128, (int)x1p[k + 1] + 128);
                    __m256i x2v = dup_u8_pairzero_epi32((int)x2p[k] + 128, (int)x2p[k + 1] + 128);
                    __m256i x3v = dup_u8_pairzero_epi32((int)x3p[k] + 128, (int)x3p[k + 1] + 128);
                    a0l = _mm256_add_epi32(a0l, maddubs_pair_i32(x0v, wl, ones16));
                    a0h = _mm256_add_epi32(a0h, maddubs_pair_i32(x0v, wh, ones16));
                    a1l = _mm256_add_epi32(a1l, maddubs_pair_i32(x1v, wl, ones16));
                    a1h = _mm256_add_epi32(a1h, maddubs_pair_i32(x1v, wh, ones16));
                    a2l = _mm256_add_epi32(a2l, maddubs_pair_i32(x2v, wl, ones16));
                    a2h = _mm256_add_epi32(a2h, maddubs_pair_i32(x2v, wh, ones16));
                    a3l = _mm256_add_epi32(a3l, maddubs_pair_i32(x3v, wl, ones16));
                    a3h = _mm256_add_epi32(a3h, maddubs_pair_i32(x3v, wh, ones16));
                }
                for (; k < c; k++) {
                    __m256i wl = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    __m256i wh = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc + 8)));
                    __m256i x0v = _mm256_set1_epi32((int)x0p[k] + 128);
                    __m256i x1v = _mm256_set1_epi32((int)x1p[k] + 128);
                    __m256i x2v = _mm256_set1_epi32((int)x2p[k] + 128);
                    __m256i x3v = _mm256_set1_epi32((int)x3p[k] + 128);
                    a0l = _mm256_add_epi32(a0l, _mm256_mullo_epi32(x0v, wl));
                    a0h = _mm256_add_epi32(a0h, _mm256_mullo_epi32(x0v, wh));
                    a1l = _mm256_add_epi32(a1l, _mm256_mullo_epi32(x1v, wl));
                    a1h = _mm256_add_epi32(a1h, _mm256_mullo_epi32(x1v, wh));
                    a2l = _mm256_add_epi32(a2l, _mm256_mullo_epi32(x2v, wl));
                    a2h = _mm256_add_epi32(a2h, _mm256_mullo_epi32(x2v, wh));
                    a3l = _mm256_add_epi32(a3l, _mm256_mullo_epi32(x3v, wl));
                    a3h = _mm256_add_epi32(a3h, _mm256_mullo_epi32(x3v, wh));
                }
                __m256 scalel = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), input_scale_v);
                __m256 scaleh = wscale->numel == 1
                    ? scalel
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc + 8), input_scale_v);
                __m256 bvl = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                __m256 bvh = bias ? _mm256_loadu_ps(bias->data + oc + 8) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscalel = _mm256_mul_ps(scalel, inv_out);
                    __m256 qscaleh = _mm256_mul_ps(scaleh, inv_out);
                    __m256 qbiasl = _mm256_add_ps(_mm256_mul_ps(bvl, inv_out), out_zpf);
                    __m256 qbiash = _mm256_add_ps(_mm256_mul_ps(bvh, inv_out), out_zpf);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc,     requantize8_i8_acc(a0l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 0) * out_c + oc + 8, requantize8_i8_acc(a0h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc,     requantize8_i8_acc(a1l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc + 8, requantize8_i8_acc(a1h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc,     requantize8_i8_acc(a2l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc + 8, requantize8_i8_acc(a2h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc,     requantize8_i8_acc(a3l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc + 8, requantize8_i8_acc(a3h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc,     requantize8_f32(a0l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc + 8, requantize8_f32(a0h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc,     requantize8_f32(a1l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc + 8, requantize8_f32(a1h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc,     requantize8_f32(a2l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc + 8, requantize8_f32(a2h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc,     requantize8_f32(a3l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc + 8, requantize8_f32(a3h, scaleh, bvh, relu));
                }
            }
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack8z + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 32));
                    acc0 = _mm256_add_epi32(acc0, maddubs_pair_i32(
                        dup_u8_pairzero_epi32((int)x0p[k] + 128, (int)x0p[k + 1] + 128), wv, ones16));
                    acc1 = _mm256_add_epi32(acc1, maddubs_pair_i32(
                        dup_u8_pairzero_epi32((int)x1p[k] + 128, (int)x1p[k + 1] + 128), wv, ones16));
                    acc2 = _mm256_add_epi32(acc2, maddubs_pair_i32(
                        dup_u8_pairzero_epi32((int)x2p[k] + 128, (int)x2p[k + 1] + 128), wv, ones16));
                    acc3 = _mm256_add_epi32(acc3, maddubs_pair_i32(
                        dup_u8_pairzero_epi32((int)x3p[k] + 128, (int)x3p[k + 1] + 128), wv, ones16));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_set1_epi32((int)x0p[k] + 128), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_set1_epi32((int)x1p[k] + 128), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(_mm256_set1_epi32((int)x2p[k] + 128), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(_mm256_set1_epi32((int)x3p[k] + 128), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), input_scale_v);
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc, requantize8_i8_acc(acc0, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc, requantize8_i8_acc(acc1, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc, requantize8_i8_acc(acc2, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc, requantize8_i8_acc(acc3, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc, requantize8_f32(acc0, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc, requantize8_f32(acc1, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc, requantize8_f32(acc2, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc, requantize8_f32(acc3, scale, bv, relu));
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int a0 = dot_i8_i16_avx2(x0p, ww, c, input_zp);
                int a1 = dot_i8_i16_avx2(x1p, ww, c, input_zp);
                int a2 = dot_i8_i16_avx2(x2p, ww, c, input_zp);
                int a3 = dot_i8_i16_avx2(x3p, ww, c, input_zp);
                float scale = input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc]);
                float b = bias ? bias->data[oc] : 0.0f;
                float v0 = (float)a0 * scale + b;
                float v1 = (float)a1 * scale + b;
                float v2 = (float)a2 * scale + b;
                float v3 = (float)a3 * scale + b;
                if (yq) {
                    yq[(long)(p + 0) * out_c + oc] = requantize_i8(v0, out_scale, out_zp, relu);
                    yq[(long)(p + 1) * out_c + oc] = requantize_i8(v1, out_scale, out_zp, relu);
                    yq[(long)(p + 2) * out_c + oc] = requantize_i8(v2, out_scale, out_zp, relu);
                    yq[(long)(p + 3) * out_c + oc] = requantize_i8(v3, out_scale, out_zp, relu);
                } else {
                    yf[(long)(p + 0) * out_c + oc] = relu6_apply(v0, relu);
                    yf[(long)(p + 1) * out_c + oc] = relu6_apply(v1, relu);
                    yf[(long)(p + 2) * out_c + oc] = relu6_apply(v2, relu);
                    yf[(long)(p + 3) * out_c + oc] = relu6_apply(v3, relu);
                }
            }
        }
        for (; p < p_end; p++) {
            const signed char* xp = xq + (long)p * c;
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack8z + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 32));
                    acc = _mm256_add_epi32(acc, maddubs_pair_i32(
                        dup_u8_pairzero_epi32((int)xp[k] + 128, (int)xp[k + 1] + 128), wv, ones16));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set1_epi32((int)xp[k] + 128), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), input_scale_v);
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    store_8xi8(yq + (long)p * out_c + oc, requantize8_i8_acc(acc, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)p * out_c + oc, requantize8_f32(acc, scale, bv, relu));
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int acc = dot_i8_i16_avx2(xp, ww, c, input_zp);
                float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc])
                        + (bias ? bias->data[oc] : 0.0f);
                if (yq) yq[(long)p * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
                else yf[(long)p * out_c + oc] = relu6_apply(v, relu);
            }
        }
        return;
    }
    if (wi && wpack && wpack_il && out_c >= 8) {
        __m256i qmin = _mm256_set1_epi32(-128), qmax = _mm256_set1_epi32(127);
        __m256 inv_out = _mm256_set1_ps(out_scale > 0.0f ? 1.0f / out_scale : 1.0f);
        __m256 out_zpf = _mm256_set1_ps((float)out_zp);
        __m256 qlo = out_zpf;
        __m256 qhi = _mm256_set1_ps(out_scale > 0.0f ? 6.0f / out_scale + (float)out_zp : (float)out_zp);
        int p = p_begin;
        for (; (yq || yf) && out_c >= 16 && p + 4 <= p_end; p += 4) {
            const signed char* x0p = xq + (long)(p + 0) * c;
            const signed char* x1p = xq + (long)(p + 1) * c;
            const signed char* x2p = xq + (long)(p + 2) * c;
            const signed char* x3p = xq + (long)(p + 3) * c;
            int oc = 0;
            for (; oc + 16 <= out_c; oc += 16) {
                __m256i a0l = _mm256_setzero_si256(), a0h = _mm256_setzero_si256();
                __m256i a1l = _mm256_setzero_si256(), a1h = _mm256_setzero_si256();
                __m256i a2l = _mm256_setzero_si256(), a2h = _mm256_setzero_si256();
                __m256i a3l = _mm256_setzero_si256(), a3h = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    long off0 = ((long)(oc >> 3) * n_kpair + (k >> 1)) * 16;
                    __m256i wl = _mm256_loadu_si256((const __m256i*)(const void*)(wpack_il + off0));
                    __m256i wh = _mm256_loadu_si256((const __m256i*)(const void*)(wpack_il + off0 + (long)n_kpair * 16));
                    __m256i x0v = dup_i16_pair_epi32((int)x0p[k] - input_zp, (int)x0p[k + 1] - input_zp);
                    __m256i x1v = dup_i16_pair_epi32((int)x1p[k] - input_zp, (int)x1p[k + 1] - input_zp);
                    __m256i x2v = dup_i16_pair_epi32((int)x2p[k] - input_zp, (int)x2p[k + 1] - input_zp);
                    __m256i x3v = dup_i16_pair_epi32((int)x3p[k] - input_zp, (int)x3p[k + 1] - input_zp);
                    a0l = _mm256_add_epi32(a0l, _mm256_madd_epi16(x0v, wl));
                    a0h = _mm256_add_epi32(a0h, _mm256_madd_epi16(x0v, wh));
                    a1l = _mm256_add_epi32(a1l, _mm256_madd_epi16(x1v, wl));
                    a1h = _mm256_add_epi32(a1h, _mm256_madd_epi16(x1v, wh));
                    a2l = _mm256_add_epi32(a2l, _mm256_madd_epi16(x2v, wl));
                    a2h = _mm256_add_epi32(a2h, _mm256_madd_epi16(x2v, wh));
                    a3l = _mm256_add_epi32(a3l, _mm256_madd_epi16(x3v, wl));
                    a3h = _mm256_add_epi32(a3h, _mm256_madd_epi16(x3v, wh));
                }
                for (; k < c; k++) {
                    __m256i wl = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    __m256i wh = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc + 8)));
                    __m256i x0v = _mm256_set1_epi32((int)x0p[k] - input_zp);
                    __m256i x1v = _mm256_set1_epi32((int)x1p[k] - input_zp);
                    __m256i x2v = _mm256_set1_epi32((int)x2p[k] - input_zp);
                    __m256i x3v = _mm256_set1_epi32((int)x3p[k] - input_zp);
                    a0l = _mm256_add_epi32(a0l, _mm256_mullo_epi32(x0v, wl));
                    a0h = _mm256_add_epi32(a0h, _mm256_mullo_epi32(x0v, wh));
                    a1l = _mm256_add_epi32(a1l, _mm256_mullo_epi32(x1v, wl));
                    a1h = _mm256_add_epi32(a1h, _mm256_mullo_epi32(x1v, wh));
                    a2l = _mm256_add_epi32(a2l, _mm256_mullo_epi32(x2v, wl));
                    a2h = _mm256_add_epi32(a2h, _mm256_mullo_epi32(x2v, wh));
                    a3l = _mm256_add_epi32(a3l, _mm256_mullo_epi32(x3v, wl));
                    a3h = _mm256_add_epi32(a3h, _mm256_mullo_epi32(x3v, wh));
                }
                __m256 scalel = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 scaleh = wscale->numel == 1
                    ? scalel
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc + 8), _mm256_set1_ps(input_scale));
                __m256 bvl = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                __m256 bvh = bias ? _mm256_loadu_ps(bias->data + oc + 8) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscalel = _mm256_mul_ps(scalel, inv_out);
                    __m256 qscaleh = _mm256_mul_ps(scaleh, inv_out);
                    __m256 qbiasl = _mm256_add_ps(_mm256_mul_ps(bvl, inv_out), out_zpf);
                    __m256 qbiash = _mm256_add_ps(_mm256_mul_ps(bvh, inv_out), out_zpf);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc,     requantize8_i8_acc(a0l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 0) * out_c + oc + 8, requantize8_i8_acc(a0h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc,     requantize8_i8_acc(a1l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc + 8, requantize8_i8_acc(a1h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc,     requantize8_i8_acc(a2l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc + 8, requantize8_i8_acc(a2h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc,     requantize8_i8_acc(a3l, qscalel, qbiasl, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc + 8, requantize8_i8_acc(a3h, qscaleh, qbiash, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc,     requantize8_f32(a0l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc + 8, requantize8_f32(a0h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc,     requantize8_f32(a1l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc + 8, requantize8_f32(a1h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc,     requantize8_f32(a2l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc + 8, requantize8_f32(a2h, scaleh, bvh, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc,     requantize8_f32(a3l, scalel, bvl, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc + 8, requantize8_f32(a3h, scaleh, bvh, relu));
                }
            }
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack_il + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 16));
                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x0p[k] - input_zp, (int)x0p[k + 1] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x1p[k] - input_zp, (int)x1p[k + 1] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x2p[k] - input_zp, (int)x2p[k + 1] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x3p[k] - input_zp, (int)x3p[k + 1] - input_zp), wv));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_set1_epi32((int)x0p[k] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_set1_epi32((int)x1p[k] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(_mm256_set1_epi32((int)x2p[k] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(_mm256_set1_epi32((int)x3p[k] - input_zp), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc, requantize8_i8_acc(acc0, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc, requantize8_i8_acc(acc1, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc, requantize8_i8_acc(acc2, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc, requantize8_i8_acc(acc3, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc, requantize8_f32(acc0, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc, requantize8_f32(acc1, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc, requantize8_f32(acc2, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc, requantize8_f32(acc3, scale, bv, relu));
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int a0 = dot_i8_i16_avx2(x0p, ww, c, input_zp);
                int a1 = dot_i8_i16_avx2(x1p, ww, c, input_zp);
                int a2 = dot_i8_i16_avx2(x2p, ww, c, input_zp);
                int a3 = dot_i8_i16_avx2(x3p, ww, c, input_zp);
                float scale = input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc]);
                float b = bias ? bias->data[oc] : 0.0f;
                float v0 = (float)a0 * scale + b;
                float v1 = (float)a1 * scale + b;
                float v2 = (float)a2 * scale + b;
                float v3 = (float)a3 * scale + b;
                if (yq) {
                    yq[(long)(p + 0) * out_c + oc] = requantize_i8(v0, out_scale, out_zp, relu);
                    yq[(long)(p + 1) * out_c + oc] = requantize_i8(v1, out_scale, out_zp, relu);
                    yq[(long)(p + 2) * out_c + oc] = requantize_i8(v2, out_scale, out_zp, relu);
                    yq[(long)(p + 3) * out_c + oc] = requantize_i8(v3, out_scale, out_zp, relu);
                } else {
                    yf[(long)(p + 0) * out_c + oc] = relu6_apply(v0, relu);
                    yf[(long)(p + 1) * out_c + oc] = relu6_apply(v1, relu);
                    yf[(long)(p + 2) * out_c + oc] = relu6_apply(v2, relu);
                    yf[(long)(p + 3) * out_c + oc] = relu6_apply(v3, relu);
                }
            }
        }
        for (; (yq || yf) && p + 8 <= p_end; p += 8) {
            const signed char* x0p = xq + (long)(p + 0) * c;
            const signed char* x1p = xq + (long)(p + 1) * c;
            const signed char* x2p = xq + (long)(p + 2) * c;
            const signed char* x3p = xq + (long)(p + 3) * c;
            const signed char* x4p = xq + (long)(p + 4) * c;
            const signed char* x5p = xq + (long)(p + 5) * c;
            const signed char* x6p = xq + (long)(p + 6) * c;
            const signed char* x7p = xq + (long)(p + 7) * c;
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                __m256i acc4 = _mm256_setzero_si256();
                __m256i acc5 = _mm256_setzero_si256();
                __m256i acc6 = _mm256_setzero_si256();
                __m256i acc7 = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack_il + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 16));
                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x0p[k] - input_zp, (int)x0p[k + 1] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x1p[k] - input_zp, (int)x1p[k + 1] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x2p[k] - input_zp, (int)x2p[k + 1] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x3p[k] - input_zp, (int)x3p[k + 1] - input_zp), wv));
                    acc4 = _mm256_add_epi32(acc4, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x4p[k] - input_zp, (int)x4p[k + 1] - input_zp), wv));
                    acc5 = _mm256_add_epi32(acc5, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x5p[k] - input_zp, (int)x5p[k + 1] - input_zp), wv));
                    acc6 = _mm256_add_epi32(acc6, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x6p[k] - input_zp, (int)x6p[k + 1] - input_zp), wv));
                    acc7 = _mm256_add_epi32(acc7, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x7p[k] - input_zp, (int)x7p[k + 1] - input_zp), wv));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_set1_epi32((int)x0p[k] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_set1_epi32((int)x1p[k] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(_mm256_set1_epi32((int)x2p[k] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(_mm256_set1_epi32((int)x3p[k] - input_zp), wv));
                    acc4 = _mm256_add_epi32(acc4, _mm256_mullo_epi32(_mm256_set1_epi32((int)x4p[k] - input_zp), wv));
                    acc5 = _mm256_add_epi32(acc5, _mm256_mullo_epi32(_mm256_set1_epi32((int)x5p[k] - input_zp), wv));
                    acc6 = _mm256_add_epi32(acc6, _mm256_mullo_epi32(_mm256_set1_epi32((int)x6p[k] - input_zp), wv));
                    acc7 = _mm256_add_epi32(acc7, _mm256_mullo_epi32(_mm256_set1_epi32((int)x7p[k] - input_zp), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc, requantize8_i8_acc(acc0, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 1) * out_c + oc, requantize8_i8_acc(acc1, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 2) * out_c + oc, requantize8_i8_acc(acc2, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 3) * out_c + oc, requantize8_i8_acc(acc3, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 4) * out_c + oc, requantize8_i8_acc(acc4, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 5) * out_c + oc, requantize8_i8_acc(acc5, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 6) * out_c + oc, requantize8_i8_acc(acc6, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                    store_8xi8(yq + (long)(p + 7) * out_c + oc, requantize8_i8_acc(acc7, qscale, qbias, relu, qlo, qhi, qmin, qmax));
                } else {
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc, requantize8_f32(acc0, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc, requantize8_f32(acc1, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc, requantize8_f32(acc2, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc, requantize8_f32(acc3, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 4) * out_c + oc, requantize8_f32(acc4, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 5) * out_c + oc, requantize8_f32(acc5, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 6) * out_c + oc, requantize8_f32(acc6, scale, bv, relu));
                    _mm256_storeu_ps(yf + (long)(p + 7) * out_c + oc, requantize8_f32(acc7, scale, bv, relu));
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int a0 = dot_i8_i16_avx2(x0p, ww, c, input_zp);
                int a1 = dot_i8_i16_avx2(x1p, ww, c, input_zp);
                int a2 = dot_i8_i16_avx2(x2p, ww, c, input_zp);
                int a3 = dot_i8_i16_avx2(x3p, ww, c, input_zp);
                int a4 = dot_i8_i16_avx2(x4p, ww, c, input_zp);
                int a5 = dot_i8_i16_avx2(x5p, ww, c, input_zp);
                int a6 = dot_i8_i16_avx2(x6p, ww, c, input_zp);
                int a7 = dot_i8_i16_avx2(x7p, ww, c, input_zp);
                float scale = input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc]);
                float b = bias ? bias->data[oc] : 0.0f;
                float v0 = (float)a0 * scale + b, v1 = (float)a1 * scale + b;
                float v2 = (float)a2 * scale + b, v3 = (float)a3 * scale + b;
                float v4 = (float)a4 * scale + b, v5 = (float)a5 * scale + b;
                float v6 = (float)a6 * scale + b, v7 = (float)a7 * scale + b;
                if (yq) {
                    yq[(long)(p + 0) * out_c + oc] = requantize_i8(v0, out_scale, out_zp, relu);
                    yq[(long)(p + 1) * out_c + oc] = requantize_i8(v1, out_scale, out_zp, relu);
                    yq[(long)(p + 2) * out_c + oc] = requantize_i8(v2, out_scale, out_zp, relu);
                    yq[(long)(p + 3) * out_c + oc] = requantize_i8(v3, out_scale, out_zp, relu);
                    yq[(long)(p + 4) * out_c + oc] = requantize_i8(v4, out_scale, out_zp, relu);
                    yq[(long)(p + 5) * out_c + oc] = requantize_i8(v5, out_scale, out_zp, relu);
                    yq[(long)(p + 6) * out_c + oc] = requantize_i8(v6, out_scale, out_zp, relu);
                    yq[(long)(p + 7) * out_c + oc] = requantize_i8(v7, out_scale, out_zp, relu);
                } else {
                    yf[(long)(p + 0) * out_c + oc] = relu6_apply(v0, relu);
                    yf[(long)(p + 1) * out_c + oc] = relu6_apply(v1, relu);
                    yf[(long)(p + 2) * out_c + oc] = relu6_apply(v2, relu);
                    yf[(long)(p + 3) * out_c + oc] = relu6_apply(v3, relu);
                    yf[(long)(p + 4) * out_c + oc] = relu6_apply(v4, relu);
                    yf[(long)(p + 5) * out_c + oc] = relu6_apply(v5, relu);
                    yf[(long)(p + 6) * out_c + oc] = relu6_apply(v6, relu);
                    yf[(long)(p + 7) * out_c + oc] = relu6_apply(v7, relu);
                }
            }
        }
        for (; p + 4 <= p_end; p += 4) {
            const signed char* x0p = xq + (long)(p + 0) * c;
            const signed char* x1p = xq + (long)(p + 1) * c;
            const signed char* x2p = xq + (long)(p + 2) * c;
            const signed char* x3p = xq + (long)(p + 3) * c;
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack_il + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 16));
                    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x0p[k] - input_zp, (int)x0p[k + 1] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x1p[k] - input_zp, (int)x1p[k + 1] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x2p[k] - input_zp, (int)x2p[k + 1] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)x3p[k] - input_zp, (int)x3p[k + 1] - input_zp), wv));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(_mm256_set1_epi32((int)x0p[k] - input_zp), wv));
                    acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(_mm256_set1_epi32((int)x1p[k] - input_zp), wv));
                    acc2 = _mm256_add_epi32(acc2, _mm256_mullo_epi32(_mm256_set1_epi32((int)x2p[k] - input_zp), wv));
                    acc3 = _mm256_add_epi32(acc3, _mm256_mullo_epi32(_mm256_set1_epi32((int)x3p[k] - input_zp), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    __m256i q0 = requantize8_i8_acc(acc0, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    __m256i q1 = requantize8_i8_acc(acc1, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    __m256i q2 = requantize8_i8_acc(acc2, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    __m256i q3 = requantize8_i8_acc(acc3, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    store_8xi8(yq + (long)(p + 0) * out_c + oc, q0);
                    store_8xi8(yq + (long)(p + 1) * out_c + oc, q1);
                    store_8xi8(yq + (long)(p + 2) * out_c + oc, q2);
                    store_8xi8(yq + (long)(p + 3) * out_c + oc, q3);
                } else {
                    __m256 v0 = requantize8_f32(acc0, scale, bv, relu);
                    __m256 v1 = requantize8_f32(acc1, scale, bv, relu);
                    __m256 v2 = requantize8_f32(acc2, scale, bv, relu);
                    __m256 v3 = requantize8_f32(acc3, scale, bv, relu);
                    _mm256_storeu_ps(yf + (long)(p + 0) * out_c + oc, v0);
                    _mm256_storeu_ps(yf + (long)(p + 1) * out_c + oc, v1);
                    _mm256_storeu_ps(yf + (long)(p + 2) * out_c + oc, v2);
                    _mm256_storeu_ps(yf + (long)(p + 3) * out_c + oc, v3);
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int a0 = dot_i8_i16_avx2(x0p, ww, c, input_zp);
                int a1 = dot_i8_i16_avx2(x1p, ww, c, input_zp);
                int a2 = dot_i8_i16_avx2(x2p, ww, c, input_zp);
                int a3 = dot_i8_i16_avx2(x3p, ww, c, input_zp);
                float scale = input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc]);
                float b = bias ? bias->data[oc] : 0.0f;
                float v0 = (float)a0 * scale + b, v1 = (float)a1 * scale + b;
                float v2 = (float)a2 * scale + b, v3 = (float)a3 * scale + b;
                if (yq) {
                    yq[(long)(p + 0) * out_c + oc] = requantize_i8(v0, out_scale, out_zp, relu);
                    yq[(long)(p + 1) * out_c + oc] = requantize_i8(v1, out_scale, out_zp, relu);
                    yq[(long)(p + 2) * out_c + oc] = requantize_i8(v2, out_scale, out_zp, relu);
                    yq[(long)(p + 3) * out_c + oc] = requantize_i8(v3, out_scale, out_zp, relu);
                } else {
                    yf[(long)(p + 0) * out_c + oc] = relu6_apply(v0, relu);
                    yf[(long)(p + 1) * out_c + oc] = relu6_apply(v1, relu);
                    yf[(long)(p + 2) * out_c + oc] = relu6_apply(v2, relu);
                    yf[(long)(p + 3) * out_c + oc] = relu6_apply(v3, relu);
                }
            }
        }
        for (; p < p_end; p++) {
            const signed char* xp = xq + (long)p * c;
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc = _mm256_setzero_si256();
                int k = 0;
                for (; k + 1 < c; k += 2) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                        wpack_il + ((long)(oc >> 3) * n_kpair + (k >> 1)) * 16));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                        dup_i16_pair_epi32((int)xp[k] - input_zp, (int)xp[k + 1] - input_zp), wv));
                }
                for (; k < c; k++) {
                    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)k * out_c + oc)));
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set1_epi32((int)xp[k] - input_zp), wv));
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    __m256i qi = requantize8_i8_acc(acc, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    store_8xi8(yq + (long)p * out_c + oc, qi);
                } else {
                    __m256 vf = requantize8_f32(acc, scale, bv, relu);
                    _mm256_storeu_ps(yf + (long)p * out_c + oc, vf);
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * c;
                int acc = dot_i8_i16_avx2(xp, ww, c, input_zp);
                float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc])
                        + (bias ? bias->data[oc] : 0.0f);
                if (yq) yq[(long)p * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
                else yf[(long)p * out_c + oc] = relu6_apply(v, relu);
            }
        }
        return;
    }
#endif
    for (int p = p_begin; p < p_end; p++) {
        const signed char* xp = xq + (long)p * c;
        for (int oc = 0; oc < out_c; oc++) {
            const int16_t* ww = wi + (long)oc * c;
            int acc = 0;
            for (int ic = 0; ic < c; ic++) acc += ((int)xp[ic] - input_zp) * (int)ww[ic];
            float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc])
                    + (bias ? bias->data[oc] : 0.0f);
            if (yq) yq[(long)p * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
            else yf[(long)p * out_c + oc] = relu6_apply(v, relu);
        }
    }
}

typedef struct {
    const signed char* xq; signed char* yq; float* yf;
    const int16_t* wi; const int16_t* wpack; const int16_t* wpack_il;
    const signed char* wpack8z;
    const float* wscale_data; long wscale_numel; const float* bias_data;
    int c, h, w, out_c; float input_scale;
    int input_zp; float out_scale; int out_zp; int relu;
} vx_pw_ctx;

static void vx_pw_worker(void* v, int begin, int end) {
    const vx_pw_ctx* t = (const vx_pw_ctx*)v;
    vx_qconv2d_pointwise_range(t->xq, t->yq, t->yf, t->wi, t->wpack, t->wpack_il, t->wpack8z,
                               t->wscale_data, t->wscale_numel, t->bias_data, t->c, t->h, t->w,
                               t->out_c, t->input_scale, t->input_zp, t->out_scale, t->out_zp,
                               t->relu, begin, end);
}

int vx_qconv2d_pointwise(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                              const signed char* wpack8z,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int out_c, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu) {
    int pixels = h * w;
    vx_pw_ctx ctx = { xq, yq, yf, wi, wpack, wpack_il, wpack8z, wscale_data, wscale_numel, bias_data,
                      c, h, w, out_c, input_scale, input_zp, out_scale, out_zp, relu };
    vx_kernels_parallel_for(pixels, 32, vx_pw_worker, &ctx);
    return 1;
}

static void vx_qconv2d_depthwise_range(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wtap,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int oh, int ow, int kh, int kw,
                              int sy, int sx, int pt, int pl, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu,
                              int oy_begin, int oy_end) {
    (void)oh;
    VXFloatView wscale_view = { wscale_data, wscale_numel };
    VXFloatView bias_view = { bias_data, 0 };
    const VXFloatView* wscale = &wscale_view;
    const VXFloatView* bias = bias_data ? &bias_view : NULL;
#if defined(__AVX2__)
    if (wtap && c >= 8) {
        __m256i vzp = _mm256_set1_epi32(input_zp);
        __m256i qmin = _mm256_set1_epi32(-128), qmax = _mm256_set1_epi32(127);
        __m256 inv_out = _mm256_set1_ps(out_scale > 0.0f ? 1.0f / out_scale : 1.0f);
        __m256 out_zpf = _mm256_set1_ps((float)out_zp);
        __m256 qlo = out_zpf;
        __m256 qhi = _mm256_set1_ps(out_scale > 0.0f ? 6.0f / out_scale + (float)out_zp : (float)out_zp);
        for (int oy = oy_begin; oy < oy_end; oy++) {
            int iy0 = oy * sy - pt;
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * sx - pl;
                int interior = iy0 >= 0 && ix0 >= 0 && iy0 + kh <= h && ix0 + kw <= w;
                long out_pix = (long)oy * ow + ox;
                int ch = 0;
                for (; ch + 8 <= c; ch += 8) {
                    __m256i acc = _mm256_setzero_si256();
                    if (interior) {
#define DW_ACC_TAP(TY, TX, TAP) do { \
    const signed char* xp = xq + ((long)(iy0 + (TY)) * w + (ix0 + (TX))) * c + ch; \
    const int16_t* wp = wtap + (long)(TAP) * c + ch; \
    __m256i xv = _mm256_sub_epi32(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)xp)), vzp); \
    __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)wp)); \
    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(xv, wv)); \
} while (0)
                        if (kh == 3 && kw == 3) {
                            DW_ACC_TAP(0, 0, 0); DW_ACC_TAP(0, 1, 1); DW_ACC_TAP(0, 2, 2);
                            DW_ACC_TAP(1, 0, 3); DW_ACC_TAP(1, 1, 4); DW_ACC_TAP(1, 2, 5);
                            DW_ACC_TAP(2, 0, 6); DW_ACC_TAP(2, 1, 7); DW_ACC_TAP(2, 2, 8);
                        } else if (kh == 5 && kw == 5) {
                            DW_ACC_TAP(0, 0, 0);  DW_ACC_TAP(0, 1, 1);  DW_ACC_TAP(0, 2, 2);  DW_ACC_TAP(0, 3, 3);  DW_ACC_TAP(0, 4, 4);
                            DW_ACC_TAP(1, 0, 5);  DW_ACC_TAP(1, 1, 6);  DW_ACC_TAP(1, 2, 7);  DW_ACC_TAP(1, 3, 8);  DW_ACC_TAP(1, 4, 9);
                            DW_ACC_TAP(2, 0, 10); DW_ACC_TAP(2, 1, 11); DW_ACC_TAP(2, 2, 12); DW_ACC_TAP(2, 3, 13); DW_ACC_TAP(2, 4, 14);
                            DW_ACC_TAP(3, 0, 15); DW_ACC_TAP(3, 1, 16); DW_ACC_TAP(3, 2, 17); DW_ACC_TAP(3, 3, 18); DW_ACC_TAP(3, 4, 19);
                            DW_ACC_TAP(4, 0, 20); DW_ACC_TAP(4, 1, 21); DW_ACC_TAP(4, 2, 22); DW_ACC_TAP(4, 3, 23); DW_ACC_TAP(4, 4, 24);
                        } else {
                            for (int ky = 0; ky < kh; ky++) {
                                for (int kx = 0; kx < kw; kx++) DW_ACC_TAP(ky, kx, ky * kw + kx);
                            }
                        }
#undef DW_ACC_TAP
                    } else {
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = iy0 + ky;
                            if ((unsigned)iy >= (unsigned)h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ix0 + kx;
                                if ((unsigned)ix >= (unsigned)w) continue;
                                const signed char* xp = xq + ((long)iy * w + ix) * c + ch;
                                const int16_t* wp = wtap + (long)(ky * kw + kx) * c + ch;
                                __m256i xv = _mm256_sub_epi32(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(const void*)xp)), vzp);
                                __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)wp));
                                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(xv, wv));
                            }
                        }
                    }
                    __m256 scale = wscale->numel == 1
                        ? _mm256_set1_ps(input_scale * wscale->data[0])
                        : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + ch), _mm256_set1_ps(input_scale));
                    __m256 bv = bias ? _mm256_loadu_ps(bias->data + ch) : _mm256_setzero_ps();
                    if (yq) {
                        __m256 qscale = _mm256_mul_ps(scale, inv_out);
                        __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                        __m256i qi = requantize8_i8_acc(acc, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                        store_8xi8(yq + out_pix * c + ch, qi);
                    } else {
                        __m256 vf = requantize8_f32(acc, scale, bv, relu);
                        float tmp[8];
                        _mm256_storeu_ps(tmp, vf);
                        for (int j = 0; j < 8; j++) yf[(long)(ch + j) * oh * ow + out_pix] = tmp[j];
                    }
                }
                for (; ch < c; ch++) {
                    int acc = 0;
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            int xv = xq[((long)iy * w + ix) * c + ch] - input_zp;
                            int wv = (int)wtap[(long)(ky * kw + kx) * c + ch];
                            acc += xv * wv;
                        }
                    }
                    float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[ch])
                            + (bias ? bias->data[ch] : 0.0f);
                    if (yq) yq[out_pix * c + ch] = requantize_i8(v, out_scale, out_zp, relu);
                    else yf[(long)ch * oh * ow + out_pix] = relu6_apply(v, relu);
                }
            }
        }
        return;
    }
#endif
    for (int oy = oy_begin; oy < oy_end; oy++) {
        int iy0 = oy * sy - pt;
        for (int ox = 0; ox < ow; ox++) {
            int ix0 = ox * sx - pl;
            long out_pix = (long)oy * ow + ox;
            for (int ch = 0; ch < c; ch++) {
                int acc = 0;
                for (int ky = 0; ky < kh; ky++) {
                    int iy = iy0 + ky;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int kx = 0; kx < kw; kx++) {
                        int ix = ix0 + kx;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        int xv = xq[((long)iy * w + ix) * c + ch] - input_zp;
                        int wv = (int)wtap[(long)(ky * kw + kx) * c + ch];
                        acc += xv * wv;
                    }
                }
                float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[ch])
                        + (bias ? bias->data[ch] : 0.0f);
                if (yq) yq[out_pix * c + ch] = requantize_i8(v, out_scale, out_zp, relu);
                else yf[(long)ch * oh * ow + out_pix] = relu6_apply(v, relu);
            }
        }
    }
}

typedef struct {
    const signed char* xq; signed char* yq; float* yf;
    const int16_t* wtap;
    const float* wscale_data; long wscale_numel; const float* bias_data;
    int c, h, w, oh, ow, kh, kw, sy, sx, pt, pl; float input_scale;
    int input_zp; float out_scale; int out_zp; int relu;
} vx_dw_ctx;

static void vx_dw_worker(void* v, int begin, int end) {
    const vx_dw_ctx* t = (const vx_dw_ctx*)v;
    vx_qconv2d_depthwise_range(t->xq, t->yq, t->yf, t->wtap, t->wscale_data, t->wscale_numel,
                               t->bias_data, t->c, t->h, t->w, t->oh, t->ow, t->kh, t->kw,
                               t->sy, t->sx, t->pt, t->pl, t->input_scale, t->input_zp,
                               t->out_scale, t->out_zp, t->relu, begin, end);
}

int vx_qconv2d_depthwise(const signed char* xq, signed char* yq, float* yf,
                              const int16_t* wtap,
                              const float* wscale_data, long wscale_numel, const float* bias_data,
                              int c, int h, int w, int oh, int ow, int kh, int kw,
                              int sy, int sx, int pt, int pl, float input_scale,
                              int input_zp, float out_scale, int out_zp, int relu) {
    vx_dw_ctx ctx = { xq, yq, yf, wtap, wscale_data, wscale_numel, bias_data,
                      c, h, w, oh, ow, kh, kw, sy, sx, pt, pl, input_scale,
                      input_zp, out_scale, out_zp, relu };
    vx_kernels_parallel_for(oh, 1, vx_dw_worker, &ctx);
    return 1;
}

static void vx_qconv2d_stem3s2_range(const signed char* xq, signed char* yq, float* yf,
                            const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                            const float* wscale_data, long wscale_numel, const float* bias_data,
                            int h, int w, int out_c, int oh, int ow,
                            int pt, int pl, float input_scale, int input_zp,
                            float out_scale, int out_zp, int relu,
                            int oy_begin, int oy_end) {
    VXFloatView wscale_view = { wscale_data, wscale_numel };
    VXFloatView bias_view = { bias_data, 0 };
    const VXFloatView* wscale = &wscale_view;
    const VXFloatView* bias = bias_data ? &bias_view : NULL;
#if defined(__AVX2__)
    const int n_tpair = 27 / 2;  // 13 tap-pairs; tap 26 is the leftover
    __m256i qmin = _mm256_set1_epi32(-128), qmax = _mm256_set1_epi32(127);
    __m256 inv_out = _mm256_set1_ps(out_scale > 0.0f ? 1.0f / out_scale : 1.0f);
    __m256 out_zpf = _mm256_set1_ps((float)out_zp);
    __m256 qlo = out_zpf;
    __m256 qhi = _mm256_set1_ps(out_scale > 0.0f ? 6.0f / out_scale + (float)out_zp : (float)out_zp);
    for (int oy = oy_begin; oy < oy_end; oy++) {
        int iy0 = oy * 2 - pt;
        for (int ox = 0; ox < ow; ox++) {
            int ix0 = ox * 2 - pl;
            long out_pix = (long)oy * ow + ox;
            int interior = iy0 >= 0 && ix0 >= 0 && iy0 + 3 <= h && ix0 + 3 <= w;
            if (interior) {
                // All 9 taps valid: gather the 27 (ci,tap) inputs once, then run a
                // branch-free reduction that pairs taps into _mm256_madd_epi16.
                int xv[27];
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        const signed char* xp = xq + ((long)(iy0 + ky) * w + (ix0 + kx)) * 3;
                        int base = ky * 3 + kx;
                        xv[0 * 9 + base] = (int)xp[0] - input_zp;
                        xv[1 * 9 + base] = (int)xp[1] - input_zp;
                        xv[2 * 9 + base] = (int)xp[2] - input_zp;
                    }
                }
                int oc = 0;
                for (; oc + 8 <= out_c; oc += 8) {
                    long ob = oc >> 3;
                    __m256i acc = _mm256_setzero_si256();
                    for (int tp = 0; tp < n_tpair; tp++) {
                        __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(
                            wpack_il + (ob * n_tpair + tp) * 16));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(
                            dup_i16_pair_epi32(xv[2 * tp], xv[2 * tp + 1]), wv));
                    }
                    __m256i wv26 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)26 * out_c + oc)));
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set1_epi32(xv[26]), wv26));
                    __m256 scale = wscale->numel == 1
                        ? _mm256_set1_ps(input_scale * wscale->data[0])
                        : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                    __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                    if (yq) {
                        __m256 qscale = _mm256_mul_ps(scale, inv_out);
                        __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                        __m256i qi = requantize8_i8_acc(acc, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                        store_8xi8(yq + out_pix * out_c + oc, qi);
                    } else {
                        __m256 vf = requantize8_f32(acc, scale, bv, relu);
                        float tmp[8];
                        _mm256_storeu_ps(tmp, vf);
                        for (int j = 0; j < 8; j++) yf[(long)(oc + j) * oh * ow + out_pix] = tmp[j];
                    }
                }
                for (; oc < out_c; oc++) {
                    const int16_t* ww = wi + (long)oc * 27;
                    int acc = 0;
                    for (int t = 0; t < 27; t++) acc += xv[t] * ww[t];
                    float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc])
                            + (bias ? bias->data[oc] : 0.0f);
                    if (yq) yq[out_pix * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
                    else yf[(long)oc * oh * ow + out_pix] = relu6_apply(v, relu);
                }
                continue;
            }
            int oc = 0;
            for (; oc + 8 <= out_c; oc += 8) {
                __m256i acc = _mm256_setzero_si256();
                for (int ky = 0; ky < 3; ky++) {
                    int iy = iy0 + ky;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ix0 + kx;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        const signed char* xp = xq + ((long)iy * w + ix) * 3;
                        for (int ci = 0; ci < 3; ci++) {
                            int xv = (int)xp[ci] - input_zp;
                            int t = ci * 9 + ky * 3 + kx;
                            __m256i wv = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(const void*)(wpack + (long)t * out_c + oc)));
                            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(_mm256_set1_epi32(xv), wv));
                        }
                    }
                }
                __m256 scale = wscale->numel == 1
                    ? _mm256_set1_ps(input_scale * wscale->data[0])
                    : _mm256_mul_ps(_mm256_loadu_ps(wscale->data + oc), _mm256_set1_ps(input_scale));
                __m256 bv = bias ? _mm256_loadu_ps(bias->data + oc) : _mm256_setzero_ps();
                if (yq) {
                    __m256 qscale = _mm256_mul_ps(scale, inv_out);
                    __m256 qbias = _mm256_add_ps(_mm256_mul_ps(bv, inv_out), out_zpf);
                    __m256i qi = requantize8_i8_acc(acc, qscale, qbias, relu, qlo, qhi, qmin, qmax);
                    store_8xi8(yq + out_pix * out_c + oc, qi);
                } else {
                    __m256 vf = requantize8_f32(acc, scale, bv, relu);
                    float tmp[8];
                    _mm256_storeu_ps(tmp, vf);
                    for (int j = 0; j < 8; j++) yf[(long)(oc + j) * oh * ow + out_pix] = tmp[j];
                }
            }
            for (; oc < out_c; oc++) {
                const int16_t* ww = wi + (long)oc * 27;
                int acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = iy0 + ky;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ix0 + kx;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        const signed char* xp = xq + ((long)iy * w + ix) * 3;
                        for (int ci = 0; ci < 3; ci++)
                            acc += ((int)xp[ci] - input_zp) * ww[ci * 9 + ky * 3 + kx];
                    }
                }
                float v = (float)acc * input_scale * (wscale->numel == 1 ? wscale->data[0] : wscale->data[oc])
                        + (bias ? bias->data[oc] : 0.0f);
                if (yq) yq[out_pix * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
                else yf[(long)oc * oh * ow + out_pix] = relu6_apply(v, relu);
            }
        }
    }
#else
    (void)xq; (void)yq; (void)yf; (void)wi; (void)wpack; (void)wpack_il; (void)wscale; (void)bias; (void)h; (void)w;
    (void)out_c; (void)oh; (void)ow; (void)pt; (void)pl; (void)input_scale; (void)input_zp;
    (void)out_scale; (void)out_zp; (void)relu; (void)oy_begin; (void)oy_end;
#endif
}

typedef struct {
    const signed char* xq; signed char* yq; float* yf;
    const int16_t* wi; const int16_t* wpack; const int16_t* wpack_il;
    const float* wscale_data; long wscale_numel; const float* bias_data;
    int h, w, out_c, oh, ow, pt, pl; float input_scale; int input_zp;
    float out_scale; int out_zp; int relu;
} vx_stem_ctx;

static void vx_stem_worker(void* v, int begin, int end) {
    const vx_stem_ctx* t = (const vx_stem_ctx*)v;
    vx_qconv2d_stem3s2_range(t->xq, t->yq, t->yf, t->wi, t->wpack, t->wpack_il, t->wscale_data,
                             t->wscale_numel, t->bias_data, t->h, t->w, t->out_c, t->oh, t->ow,
                             t->pt, t->pl, t->input_scale, t->input_zp, t->out_scale,
                             t->out_zp, t->relu, begin, end);
}

int vx_qconv2d_stem3s2(const signed char* xq, signed char* yq, float* yf,
                            const int16_t* wi, const int16_t* wpack, const int16_t* wpack_il,
                            const float* wscale_data, long wscale_numel, const float* bias_data,
                            int h, int w, int out_c, int oh, int ow,
                            int pt, int pl, float input_scale, int input_zp,
                            float out_scale, int out_zp, int relu) {
#if defined(__AVX2__)
    if (!wi || !wpack || !wpack_il || out_c < 8) return 0;
    vx_stem_ctx ctx = { xq, yq, yf, wi, wpack, wpack_il, wscale_data, wscale_numel, bias_data,
                        h, w, out_c, oh, ow, pt, pl, input_scale, input_zp,
                        out_scale, out_zp, relu };
    vx_kernels_parallel_for(oh, 1, vx_stem_worker, &ctx);
    return 1;
#else
    (void)xq; (void)yq; (void)yf; (void)wi; (void)wpack; (void)wpack_il; (void)wscale_data; (void)wscale_numel;
    (void)bias_data; (void)h; (void)w; (void)out_c; (void)oh; (void)ow; (void)pt; (void)pl;
    (void)input_scale; (void)input_zp; (void)out_scale; (void)out_zp; (void)relu;
    return 0;
#endif
}

enum {
    VX_T_F32 = VX_DTYPE_F32,
    VX_T_I8 = VX_DTYPE_I8,
    VX_T_U8 = VX_DTYPE_U8,
    VX_T_I32 = VX_DTYPE_I32,
    VX_T_I16 = VX_DTYPE_I16
};

static int vx_quantized_value_i32(const void* data, int dtype, long idx) {
    if (!data) return 0;
    if (dtype == VX_T_I8) return ((const signed char*)data)[idx];
    if (dtype == VX_T_U8) return ((const unsigned char*)data)[idx];
    if (dtype == VX_T_I32) {
        int32_t value;
        memcpy(&value, (const unsigned char*)data + (size_t)idx * sizeof(value), sizeof(value));
        return value;
    }
    float value;
    memcpy(&value, (const unsigned char*)data + (size_t)idx * sizeof(value), sizeof(value));
    return (int)lrintf(value);
}

static float vx_quantized_f32(const float* data, long idx) {
    float value;
    memcpy(&value, (const unsigned char*)data + (size_t)idx * sizeof(value), sizeof(value));
    return value;
}

static int32_t vx_quantized_i32(const int32_t* data, long idx) {
    int32_t value;
    memcpy(&value, (const unsigned char*)data + (size_t)idx * sizeof(value), sizeof(value));
    return value;
}

static int vx_quantized_weight_i32(const void* data, int dtype, long idx) {
    if (dtype == VX_T_I16) return ((const int16_t*)data)[idx];
    if (dtype == VX_T_U8) return ((const unsigned char*)data)[idx];
    return ((const signed char*)data)[idx];
}

int vx_qconv2d_generic(const signed char* xq, signed char* yq, float* yf,
                            const void* weight_data, int weight_dtype,
                            const float* wscale_data, long wscale_numel,
                            const void* wzp_data, int wzp_dtype, long wzp_numel,
                            const float* bias_data,
                            int c, int h, int w, int out_c, int in_per_group, int oh, int ow,
                            int kh, int kw, int sy, int sx, int pt, int pl, int groups,
                            float input_scale, int input_zp, float out_scale, int out_zp, int relu) {
    int group_out = out_c / groups;
    int group_in = c / groups;
    for (int oy = 0; oy < oh; oy++) {
        int iy0 = oy * sy - pt;
        for (int ox = 0; ox < ow; ox++) {
            int ix0 = ox * sx - pl;
            long out_pix = (long)oy * ow + ox;
            for (int oc = 0; oc < out_c; oc++) {
                int g = oc / group_out;
                int in_start = g * group_in;
                int wz = wzp_data ? vx_quantized_value_i32(wzp_data, wzp_dtype, wzp_numel == 1 ? 0 : oc) : 0;
                long wbase = (long)oc * in_per_group * kh * kw;
                int acc = 0;
                for (int ci = 0; ci < in_per_group; ci++) {
                    int ch = in_start + ci;
                    long ci_base = wbase + (long)ci * kh * kw;
                    for (int ky = 0; ky < kh; ky++) {
                        int iy = iy0 + ky;
                        if ((unsigned)iy >= (unsigned)h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int ix = ix0 + kx;
                            if ((unsigned)ix >= (unsigned)w) continue;
                            int xv = xq[((long)iy * w + ix) * c + ch] - input_zp;
                            int wv = vx_quantized_weight_i32(weight_data, weight_dtype, ci_base + ky * kw + kx) - wz;
                            acc += xv * wv;
                        }
                    }
                }
                float v = (float)acc * input_scale * (wscale_numel == 1 ? wscale_data[0] : wscale_data[oc])
                        + (bias_data ? bias_data[oc] : 0.0f);
                if (yq) yq[out_pix * out_c + oc] = requantize_i8(v, out_scale, out_zp, relu);
                else yf[(long)oc * oh * ow + out_pix] = relu6_apply(v, relu);
            }
        }
    }
    return 1;
}

int vx_qlinear_i8(const signed char* xq, signed char* yq,
                  const void* weight_data, int weight_dtype,
                  const float* wscale_data, long wscale_numel,
                  const void* wzp_data, int wzp_dtype, long wzp_numel,
                  const int32_t* bias_data,
                  int rows, int d_in, int d_out, int out_in,
                  float input_scale, int input_zp,
                  float output_scale, int output_zp) {
    if (!xq || !yq || !weight_data || !wscale_data || rows <= 0 || d_in <= 0 || d_out <= 0 ||
        (weight_dtype != VX_T_I8 && weight_dtype != VX_T_U8) ||
        (wscale_numel != 1 && wscale_numel != d_out) ||
        (wzp_data && (wzp_dtype != VX_T_I8 && wzp_dtype != VX_T_U8 && wzp_dtype != VX_T_I32)) ||
        (wzp_data && wzp_numel != 1 && wzp_numel != d_out) ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f ||
        input_zp < -128 || input_zp > 127 || output_zp < -128 || output_zp > 127) return 0;
    for (int row = 0; row < rows; row++) {
        const signed char* xrow = xq + (long)row * d_in;
        signed char* yrow = yq + (long)row * d_out;
        for (int oc = 0; oc < d_out; oc++) {
            float weight_scale = vx_quantized_f32(wscale_data, wscale_numel == 1 ? 0 : oc);
            if (!isfinite(weight_scale) || weight_scale <= 0.0f) return 0;
            int weight_zp = wzp_data
                ? vx_quantized_value_i32(wzp_data, wzp_dtype, wzp_numel == 1 ? 0 : oc)
                : 0;
            /* Keep the reference accumulation defined for unusually wide
             * rows. Normal W8A8 kernels use I32 lanes; the wider scalar type
             * prevents C signed-overflow UB before the requantization step. */
            int64_t acc = bias_data ? (int64_t)vx_quantized_i32(bias_data, oc) : 0;
            int k = 0;
#if defined(__AVX2__)
            /* OUT_IN is the common packed Linear layout. Widening to I16
             * before vpmaddwd avoids the pairwise saturation issue of
             * vpmaddubsw and supports arbitrary signed activation/weight ZP. */
            if (out_in && d_in <= 262144) {
                const unsigned char* base = (const unsigned char*)weight_data +
                    (size_t)oc * d_in;
                acc += vx_qlinear_dot_i8_avx2(xrow, base, weight_dtype, d_in,
                                               input_zp, weight_zp, &k);
            }
#endif
            for (; k < d_in; k++) {
                long wi = out_in ? (long)oc * d_in + k : (long)k * d_out + oc;
                int xv = (int)xrow[k] - input_zp;
                int wv = vx_quantized_weight_i32(weight_data, weight_dtype, wi) - weight_zp;
                acc += (int64_t)xv * wv;
            }
            float value = (float)((double)acc * (double)input_scale * (double)weight_scale);
            yrow[oc] = requantize_i8(value, output_scale, output_zp, 0);
        }
    }
    return 1;
}

static inline signed char clamp_i8_i32(int v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (signed char)v;
}

static inline signed char quantize_scalar_i8(float x, float scale, int zp) {
    float quantized = x / scale + (float)zp;
    if (isnan(quantized)) return clamp_i8_i32(zp);
    if (quantized <= -128.0f) return -128;
    if (quantized >= 127.0f) return 127;
    return (signed char)lrintf(quantized);
}

static inline float relu6_apply(float x, int relu) {
    if (!relu) return x;
    if (x < 0.0f) x = 0.0f;
    if (relu >= 2 && x > 6.0f) x = 6.0f;
    return x;
}

static inline signed char requantize_i8(float v, float out_scale, int out_zp, int relu) {
    return quantize_scalar_i8(relu6_apply(v, relu), out_scale, out_zp);
}

#if defined(__AVX2__)
static int64_t vx_qlinear_dot_i8_avx2(const signed char* x, const void* weight,
                                       int weight_dtype, int d_in,
                                       int input_zp, int weight_zp, int* processed) {
    const signed char* ws = (const signed char*)weight;
    const unsigned char* wu = (const unsigned char*)weight;
    __m256i input_zp_v = _mm256_set1_epi16((short)input_zp);
    __m256i weight_zp_v = _mm256_set1_epi16((short)weight_zp);
    __m256i sums = _mm256_setzero_si256();
    int k = 0;
    for (; k + 16 <= d_in; k += 16) {
        __m128i xb = _mm_loadu_si128((const __m128i*)(const void*)(x + k));
        const void* weight_ptr = weight_dtype == VX_T_I8
            ? (const void*)(ws + k) : (const void*)(wu + k);
        __m128i wb = _mm_loadu_si128((const __m128i*)weight_ptr);
        __m256i x16 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(xb), input_zp_v);
        __m256i w16 = weight_dtype == VX_T_I8 ? _mm256_cvtepi8_epi16(wb) : _mm256_cvtepu8_epi16(wb);
        w16 = _mm256_sub_epi16(w16, weight_zp_v);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(x16, w16));
    }
    int32_t lanes[8];
    _mm256_storeu_si256((__m256i*)(void*)lanes, sums);
    int64_t total = 0;
    for (int lane = 0; lane < 8; lane++) total += lanes[lane];
    if (processed) *processed = k;
    return total;
}

static inline void store_8xi8(signed char* dst, __m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i q16 = _mm_packs_epi32(lo, hi);
    __m128i q8 = _mm_packs_epi16(q16, q16);
    int64_t packed = (int64_t)_mm_cvtsi128_si64(q8);
    memcpy(dst, &packed, 8);
}
#endif

void vx_quantize_f32_to_i8(const float* src, signed char* dst, long n,
                           float input_scale, int input_zp, float output_scale, int output_zp) {
    if (!src || !dst || n <= 0 || output_scale <= 0.0f) return;
    long i = 0;
#if defined(__AVX2__)
    __m256 vin_zp = _mm256_set1_ps((float)input_zp);
    __m256 vout_zp = _mm256_set1_ps((float)output_zp);
    __m256 vscale = _mm256_set1_ps(input_scale > 0.0f ? input_scale / output_scale : 1.0f / output_scale);
    __m256i qmin = _mm256_set1_epi32(-128);
    __m256i qmax = _mm256_set1_epi32(127);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        if (input_scale > 0.0f) v = _mm256_sub_ps(v, vin_zp);
        __m256 qf = _mm256_add_ps(_mm256_mul_ps(v, vscale), vout_zp);
        __m256i qi = _mm256_cvtps_epi32(qf);
        qi = _mm256_min_epi32(_mm256_max_epi32(qi, qmin), qmax);
        store_8xi8(dst + i, qi);
    }
#endif
    for (; i < n; i++) {
        float v = input_scale > 0.0f ? (src[i] - (float)input_zp) * input_scale : src[i];
        dst[i] = quantize_scalar_i8(v, output_scale, output_zp);
    }
}

int vx_qmaxpool_i8_sameq(const signed char* xq, signed char* yq,
                              int c, int h, int w, int oh, int ow,
                              int ky, int kx, int sy, int sx, int py, int px,
                              int fill_zp) {
    if (!xq || !yq || c <= 0 || h <= 0 || w <= 0 || oh <= 0 || ow <= 0) return 0;
    if (ky <= 0 || kx <= 0 || ky > 8 || kx > 8 || ky * kx > 64) return 0;
    int offsets[64];
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            int nt = 0;
            for (int yy = 0; yy < ky; yy++) {
                int iy = oy * sy + yy - py;
                if ((unsigned)iy >= (unsigned)h) continue;
                for (int xx = 0; xx < kx; xx++) {
                    int ix = ox * sx + xx - px;
                    if ((unsigned)ix >= (unsigned)w) continue;
                    if (nt < (int)(sizeof(offsets) / sizeof(offsets[0]))) {
                        offsets[nt++] = (int)(((long)iy * w + ix) * c);
                    }
                }
            }
            signed char* dst = yq + ((long)oy * ow + ox) * c;
            if (nt == 0) {
                memset(dst, (unsigned char)(signed char)fill_zp, (size_t)c);
                continue;
            }
            int ch = 0;
#if defined(__AVX2__)
            for (; ch + 32 <= c; ch += 32) {
                __m256i best = _mm256_set1_epi8((char)-128);
                for (int t = 0; t < nt; t++) {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(xq + offsets[t] + ch));
                    best = _mm256_max_epi8(best, v);
                }
                _mm256_storeu_si256((__m256i*)(void*)(dst + ch), best);
            }
#endif
            for (; ch < c; ch++) {
                int best = -128;
                for (int t = 0; t < nt; t++) {
                    int qv = xq[offsets[t] + ch];
                    if (qv > best) best = qv;
                }
                dst[ch] = (signed char)best;
            }
        }
    }
    return 1;
}

#if defined(__AVX2__)
static inline int hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static inline int dot_i8_i16_avx2(const signed char* x, const int16_t* w, int c, int input_zp) {
    __m256i acc = _mm256_setzero_si256();
    __m256i vzp = _mm256_set1_epi16((short)input_zp);
    int i = 0;
    for (; i + 16 <= c; i += 16) {
        __m128i xb = _mm_loadu_si128((const __m128i*)(const void*)(x + i));
        __m256i x16 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(xb), vzp);
        __m256i w16 = _mm256_loadu_si256((const __m256i*)(const void*)(w + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x16, w16));
    }
    int sum = hsum256_epi32(acc);
    for (; i < c; i++) sum += ((int)x[i] - input_zp) * (int)w[i];
    return sum;
}

static inline __m256i dup_i16_pair_epi32(int lo, int hi) {
    uint32_t pair = (uint16_t)(int16_t)lo | ((uint32_t)(uint16_t)(int16_t)hi << 16);
    return _mm256_set1_epi32((int)pair);
}

static inline __m256i dup_u8_pairzero_epi32(int lo, int hi) {
    uint32_t pair = (uint32_t)(unsigned char)lo | ((uint32_t)(unsigned char)hi << 16);
    return _mm256_set1_epi32((int)pair);
}

static inline __m256i maddubs_pair_i32(__m256i x, __m256i w, __m256i ones16) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones16);
}

static inline __m256 requantize8_f32(__m256i acc, __m256 scale, __m256 bias, int relu) {
    __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc), scale), bias);
    if (relu) {
        v = _mm256_max_ps(v, _mm256_setzero_ps());
        if (relu >= 2) v = _mm256_min_ps(v, _mm256_set1_ps(6.0f));
    }
    return v;
}

static inline __m256i requantize8_i8_acc(__m256i acc, __m256 qscale, __m256 qbias,
                                         int relu, __m256 qlo, __m256 qhi,
                                         __m256i qmin, __m256i qmax) {
    __m256 qf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc), qscale), qbias);
    if (relu) {
        qf = _mm256_max_ps(qf, qlo);
        if (relu >= 2) qf = _mm256_min_ps(qf, qhi);
    }
    __m256i qi = _mm256_cvtps_epi32(qf);
    return _mm256_min_epi32(_mm256_max_epi32(qi, qmin), qmax);
}
#endif
