// --- Missing Final Primitives (Batch 3) ---
#include "mathcompat.h"
#include "../../include/volvoxai_enums.h"
#include "w8a8_affine.h"
#if !defined(__wasm__)
#include "quant_cpu_opt.h"
#endif
#include <stdint.h>

static double vx_typed_number(const void *values, int type, int index) {
    if (type == VX_DTYPE_F32) return (double)((const float *)values)[index];
    if (type == VX_DTYPE_I32) return (double)((const int32_t *)values)[index];
    if (type == VX_DTYPE_I8) return (double)((const int8_t *)values)[index];
    return (double)((const uint8_t *)values)[index];
}

static int vx_typed_number_dtype(int dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

void where_f32(const float* cond, const float* a, const float* b, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = cond[i] != 0.0f ? a[i] : b[i];
}

/* Canonical portable Where/Mask: F32 data/output, exact-shape F32 or I32
   condition. Keeping the condition pointer typed avoids reinterpreting an
   Int32.MIN_VALUE bit pattern as an F32 value. */
int where_typed_f32(const void *condition, int condition_dtype,
        const float *a, const float *b, float *output, uint32_t elements) {
    if (!condition || !a || !b || !output ||
        (condition_dtype != VX_DTYPE_F32 &&
         condition_dtype != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int selected = condition_dtype == VX_DTYPE_I32
            ? ((const int32_t *)condition)[index] != 0
            : ((const float *)condition)[index] != 0.0f;
        output[index] = selected ? a[index] : b[index];
    }
    return 1;
}

/* Bit-preserving Where for either F32 or I32 branch/output storage. */
WASM_EXPORT("where_typed_32")
int where_typed_32(const void *condition, int condition_dtype,
        const uint32_t *a, const uint32_t *b, uint32_t *output,
        uint32_t elements, uint32_t data_dtype) {
    if (!condition || !a || !b || !output ||
        (condition_dtype != VX_DTYPE_F32 &&
         condition_dtype != VX_DTYPE_I32) ||
        (data_dtype != VX_DTYPE_F32 && data_dtype != VX_DTYPE_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int selected = condition_dtype == VX_DTYPE_I32
            ? ((const int32_t *)condition)[index] != 0
            : ((const float *)condition)[index] != 0.0f;
        output[index] = selected ? a[index] : b[index];
    }
    return 1;
}

WASM_EXPORT("clip_i32")
int clip_i32(const int32_t *input, int32_t *output, uint32_t elements,
        int32_t minimum, int32_t maximum) {
    if (!input || !output || !elements || minimum > maximum) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int32_t value = input[index];
        if (value < minimum) value = minimum;
        if (value > maximum) value = maximum;
        output[index] = value;
    }
    return 1;
}

void dequantize_linear_f32(const float* input, const float* scale, const float* zero_point, float* output, int n) {
    double s = scale[0], zp = zero_point ? zero_point[0] : 0.0;
    for (int i = 0; i < n; i++) output[i] = (float)(((double)input[i] - zp) * s);
}

/* Dtypes use the canonical VxDataType protobuf values. The scale and output
 * are always F32. */
int dequantize_linear_typed(const void *input, int input_dtype, const float *scale,
        const void *zero_point, int zero_point_dtype, float *output, int n) {
    if (!input || !scale || !output || n < 0 ||
        !vx_typed_number_dtype(input_dtype) ||
        (zero_point && !vx_typed_number_dtype(zero_point_dtype))) return 0;
    const double zp = zero_point ? vx_typed_number(zero_point, zero_point_dtype, 0) : 0.0;
    int index = 0;
#if !defined(__wasm__)
    /* The native prefix requires an integral zero point it can hold in an I32
     * lane, so only take it when the value round-trips exactly. */
    if (zp == (double)(int)zp) {
        index = (int)vx_dequantize_linear_typed_native_prefix(
            input, (uint32_t)input_dtype, scale[0], (int)zp, output,
            (uint32_t)n);
    }
#endif
    for (; index < n; index++) {
        output[index] = (float)((vx_typed_number(input, input_dtype, index) - zp) *
            (double)scale[0]);
    }
    return 1;
}

static int vx_quantize_round_even(float value) {
    int lower = (int)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

static int vx_quantize_linear_descriptor(const float *input, const float *scale,
        const void *zero_point, uint32_t zero_point_dtype, const void *output,
        uint32_t output_dtype, int *minimum, int *maximum, int *zero) {
    if (!input || !scale || !output || !minimum || !maximum || !zero ||
        (output_dtype != VX_DTYPE_I8 && output_dtype != VX_DTYPE_U8) ||
        !vx_w8a8_finite_f32(scale[0]) || scale[0] <= 0.0f ||
        (zero_point && zero_point_dtype != output_dtype)) return 0;
    *minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    *maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    *zero = 0;
    if (zero_point) {
        *zero = output_dtype == VX_DTYPE_I8 ? ((const int8_t *)zero_point)[0] :
            ((const uint8_t *)zero_point)[0];
    }
    return 1;
}

static int vx_quantize_linear_scalar_value(float value, float scale, int zero,
        int minimum, int maximum) {
    int quantized;
    if (value != value) return zero;
    value = value / scale + (float)zero;
    if (value <= (float)minimum) quantized = minimum;
    else if (value >= (float)maximum) quantized = maximum;
    else if (!vx_w8a8_finite_f32(value)) {
        quantized = value < 0.0f ? minimum : maximum;
    } else {
        quantized = vx_quantize_round_even(value);
    }
    return quantized;
}

static void vx_quantize_linear_scalar_range(const float *input, float scale,
        int zero, void *output, uint32_t output_dtype, uint32_t begin,
        uint32_t end, int minimum, int maximum) {
    for (uint32_t index = begin; index < end; index++) {
        int quantized = vx_quantize_linear_scalar_value(input[index], scale,
            zero, minimum, maximum);
        if (output_dtype == VX_DTYPE_I8)
            ((int8_t *)output)[index] = (int8_t)quantized;
        else ((uint8_t *)output)[index] = (uint8_t)quantized;
    }
}

#if defined(__wasm__) && defined(__wasm_simd128__)
#if defined(VOLVOXAI_QUANTIZE_TESTING)
static uint32_t vx_quantize_wasm_simd_blocks = 0;

WASM_EXPORT("quantize_linear_wasm_simd_blocks")
uint32_t quantize_linear_wasm_simd_blocks(void) {
    return vx_quantize_wasm_simd_blocks;
}

WASM_EXPORT("reset_quantize_linear_wasm_simd_blocks")
void reset_quantize_linear_wasm_simd_blocks(void) {
    vx_quantize_wasm_simd_blocks = 0;
}
#endif

static v128_t vx_quantize_linear_f32x4(v128_t value, v128_t scale,
        v128_t zero, v128_t minimum, v128_t maximum) {
    const v128_t nan_lanes = wasm_f32x4_ne(value, value);
    const v128_t transformed = wasm_f32x4_add(
        wasm_f32x4_div(value, scale), wasm_f32x4_convert_i32x4(zero));
    v128_t quantized = wasm_i32x4_trunc_sat_f32x4(
        wasm_f32x4_nearest(transformed));
    quantized = wasm_i32x4_min(wasm_i32x4_max(quantized, minimum), maximum);
    return wasm_v128_bitselect(zero, quantized, nan_lanes);
}

static uint32_t vx_quantize_linear_simd128(const float *input, float scale,
        int zero, void *output, uint32_t output_dtype, uint32_t elements,
        int minimum, int maximum) {
    const v128_t scales = wasm_f32x4_splat(scale);
    const v128_t zeros = wasm_i32x4_splat(zero);
    const v128_t minima = wasm_i32x4_splat(minimum);
    const v128_t maxima = wasm_i32x4_splat(maximum);
    uint32_t index = 0;
    if (elements < 16u) return 0;
    for (; index <= elements - 16u; index += 16u) {
        const v128_t q0 = vx_quantize_linear_f32x4(
            wasm_v128_load(input + index), scales, zeros, minima, maxima);
        const v128_t q1 = vx_quantize_linear_f32x4(
            wasm_v128_load(input + index + 4u), scales, zeros, minima, maxima);
        const v128_t q2 = vx_quantize_linear_f32x4(
            wasm_v128_load(input + index + 8u), scales, zeros, minima, maxima);
        const v128_t q3 = vx_quantize_linear_f32x4(
            wasm_v128_load(input + index + 12u), scales, zeros, minima, maxima);
        v128_t packed;
        if (output_dtype == VX_DTYPE_I8) {
            packed = wasm_i8x16_narrow_i16x8(
                wasm_i16x8_narrow_i32x4(q0, q1),
                wasm_i16x8_narrow_i32x4(q2, q3));
        } else {
            packed = wasm_u8x16_narrow_i16x8(
                wasm_u16x8_narrow_i32x4(q0, q1),
                wasm_u16x8_narrow_i32x4(q2, q3));
        }
        wasm_v128_store((uint8_t *)output + index, packed);
#if defined(VOLVOXAI_QUANTIZE_TESTING)
        vx_quantize_wasm_simd_blocks++;
#endif
    }
    return index;
}
#endif

/* Canonical W8A8 QuantizeLinear. The input and scale are F32; the output is
 * physically I8/U8. A supplied zero point must use the exact output dtype.
 * NaN maps to zero point, and infinities saturate. This is deliberately a
 * separate ABI from the F32-output helper below. */
WASM_EXPORT("quantize_linear_typed")
int quantize_linear_typed(const float *input, const float *scale,
        const void *zero_point, uint32_t zero_point_dtype,
        void *output, uint32_t output_dtype, uint32_t elements) {
    int minimum, maximum, zero = 0;
    uint32_t index = 0;
    if (!vx_quantize_linear_descriptor(input, scale, zero_point,
            zero_point_dtype, output, output_dtype, &minimum, &maximum, &zero))
        return 0;
#if !defined(__wasm__)
    index = vx_quantize_linear_typed_native_prefix(
        input, scale[0], zero, output, output_dtype, elements,
        minimum, maximum);
#endif
#if defined(__wasm__) && defined(__wasm_simd128__)
    index = vx_quantize_linear_simd128(input, scale[0], zero, output,
        output_dtype, elements, minimum, maximum);
#endif
    vx_quantize_linear_scalar_range(input, scale[0], zero, output,
        output_dtype, index, elements, minimum, maximum);
    return 1;
}

#if defined(VOLVOXAI_QUANTIZE_TESTING)
/* Test-only portable oracle used to prove the SIMD path byte-for-byte against
 * the same canonical descriptor and scalar tail contract. */
WASM_EXPORT("quantize_linear_typed_scalar_reference")
int quantize_linear_typed_scalar_reference(const float *input,
        const float *scale, const void *zero_point, uint32_t zero_point_dtype,
        void *output, uint32_t output_dtype, uint32_t elements) {
    int minimum, maximum, zero;
    if (!vx_quantize_linear_descriptor(input, scale, zero_point,
            zero_point_dtype, output, output_dtype, &minimum, &maximum, &zero))
        return 0;
    vx_quantize_linear_scalar_range(input, scale[0], zero, output,
        output_dtype, 0, elements, minimum, maximum);
    return 1;
}
#endif

void quantize_linear_f32(const float* input, const float* scale, const float* zero_point, float* output, int n) {
    float s = scale[0]; float zp = zero_point ? zero_point[0] : 0.0f;
    for (int i = 0; i < n; i++) {
        float val = input[i] / s + zp;
        int quantized = (int)(val + (val >= 0.0f ? 0.5f : -0.5f));
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        output[i] = (float)quantized;
    }
}

void conv_transpose2d_f32(const float* input, const float* weight, const float* bias, float* output, int b, int in_h, int in_w, int in_c, int out_h, int out_w, int out_c, int kh, int kw, int sh, int sw, int ph, int pw) {
    for (int i = 0; i < b * out_c * out_h * out_w; i++) output[i] = bias ? bias[i % out_c] : 0.0f;
    
    for (int i_b = 0; i_b < b; i_b++) {
        for (int i_ic = 0; i_ic < in_c; i_ic++) {
            for (int iy = 0; iy < in_h; iy++) {
                for (int ix = 0; ix < in_w; ix++) {
                    float in_val = input[((long)i_b * in_h * in_w + (long)iy * in_w + ix) * in_c + i_ic];
                    for (int oc = 0; oc < out_c; oc++) {
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int oy = iy * sh - ph + ky;
                                int ox = ix * sw - pw + kx;
                                if (oy >= 0 && oy < out_h && ox >= 0 && ox < out_w) {
                                    float w_val = weight[i_ic * (out_c * kh * kw) + oc * (kh * kw) + ky * kw + kx];
                                    output[((long)i_b * out_h * out_w + (long)oy * out_w + ox) * out_c + oc] += in_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
