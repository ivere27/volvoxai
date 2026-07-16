// --- Missing Final Primitives (Batch 3) ---
#include "mathcompat.h"
#include <stdint.h>

enum {
    VX_TYPED_F32 = 0,
    VX_TYPED_I32 = 1,
    VX_TYPED_I8 = 2,
    VX_TYPED_U8 = 3,
};

static double vx_typed_number(const void *values, int type, int index) {
    if (type == VX_TYPED_F32) return (double)((const float *)values)[index];
    if (type == VX_TYPED_I32) return (double)((const int32_t *)values)[index];
    if (type == VX_TYPED_I8) return (double)((const int8_t *)values)[index];
    return (double)((const uint8_t *)values)[index];
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
        (condition_dtype != VX_TYPED_F32 && condition_dtype != VX_TYPED_I32)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        int selected = condition_dtype == VX_TYPED_I32
            ? ((const int32_t *)condition)[index] != 0
            : ((const float *)condition)[index] != 0.0f;
        output[index] = selected ? a[index] : b[index];
    }
    return 1;
}

void dequantize_linear_f32(const float* input, const float* scale, const float* zero_point, float* output, int n) {
    double s = scale[0], zp = zero_point ? zero_point[0] : 0.0;
    for (int i = 0; i < n; i++) output[i] = (float)(((double)input[i] - zp) * s);
}

/* dtype: 0=F32, 1=I32, 2=I8, 3=U8. The scale and output are always F32. */
int dequantize_linear_typed(const void *input, int input_dtype, const float *scale,
        const void *zero_point, int zero_point_dtype, float *output, int n) {
    if (!input || !scale || !output || n < 0 || input_dtype < VX_TYPED_F32 ||
        input_dtype > VX_TYPED_U8 || (zero_point &&
        (zero_point_dtype < VX_TYPED_F32 || zero_point_dtype > VX_TYPED_U8))) return 0;
    const double zp = zero_point ? vx_typed_number(zero_point, zero_point_dtype, 0) : 0.0;
    for (int index = 0; index < n; index++) {
        output[index] = (float)((vx_typed_number(input, input_dtype, index) - zp) *
            (double)scale[0]);
    }
    return 1;
}

static int vx_quantize_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_quantize_round_even(float value) {
    int lower = (int)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

/* Canonical W8A8 QuantizeLinear. The input and scale are F32; the output is
 * physically I8/U8. A supplied zero point must use the exact output dtype.
 * NaN maps to zero point, and infinities saturate. This is deliberately a
 * separate ABI from the legacy F32-output helper below. */
WASM_EXPORT("quantize_linear_typed")
int quantize_linear_typed(const float *input, const float *scale,
        const void *zero_point, uint32_t zero_point_dtype,
        void *output, uint32_t output_dtype, uint32_t elements) {
    int minimum, maximum, zero = 0;
    if (!input || !scale || !output || (output_dtype != VX_TYPED_I8 && output_dtype != VX_TYPED_U8) ||
        !vx_quantize_finite_f32(scale[0]) || scale[0] <= 0.0f ||
        (zero_point && zero_point_dtype != output_dtype)) return 0;
    minimum = output_dtype == VX_TYPED_I8 ? -128 : 0;
    maximum = output_dtype == VX_TYPED_I8 ? 127 : 255;
    if (zero_point) {
        zero = output_dtype == VX_TYPED_I8 ? ((const int8_t *)zero_point)[0] :
            ((const uint8_t *)zero_point)[0];
    }
    for (uint32_t index = 0; index < elements; index++) {
        float value = input[index];
        int quantized;
        if (value != value) {
            quantized = zero;
        } else {
            float transformed = value / scale[0] + (float)zero;
            if (transformed <= (float)minimum) quantized = minimum;
            else if (transformed >= (float)maximum) quantized = maximum;
            else if (!vx_quantize_finite_f32(transformed)) {
                quantized = transformed < 0.0f ? minimum : maximum;
            } else {
                quantized = vx_quantize_round_even(transformed);
            }
        }
        if (output_dtype == VX_TYPED_I8) ((int8_t *)output)[index] = (int8_t)quantized;
        else ((uint8_t *)output)[index] = (uint8_t)quantized;
    }
    return 1;
}

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
