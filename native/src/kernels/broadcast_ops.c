// --- The Final N-Dimensional "No Stubs" Primitives (Batch 5) ---
#include <stdint.h>

enum {
    VX_CAST_F32 = 0,
    VX_CAST_I32 = 1,
    VX_CAST_I8 = 2,
    VX_CAST_U8 = 3,
};

static int32_t vx_cast_integer_read(const void *input, int dtype, int index) {
    if (dtype == VX_CAST_I32) return ((const int32_t *)input)[index];
    if (dtype == VX_CAST_I8) return (int32_t)((const int8_t *)input)[index];
    return (int32_t)((const uint8_t *)input)[index];
}

/* Match ECMAScript TypedArray ToInt32 exactly for an F32 source: non-finite
   values become zero; finite values truncate toward zero and wrap modulo
   2^32. Doing this from bits avoids undefined C float-to-int conversions. */
static uint32_t vx_cast_f32_to_u32_mod(float value) {
    union { float f; uint32_t u; } bits = { value };
    uint32_t exponent = (bits.u >> 23u) & 0xffu;
    if (exponent == 0xffu || exponent < 127u) return 0;
    uint32_t significand = (bits.u & 0x7fffffu) | 0x800000u;
    int32_t shift = (int32_t)exponent - 127 - 23;
    uint32_t magnitude;
    if (shift >= 32) magnitude = 0;
    else if (shift >= 0) magnitude = significand << (uint32_t)shift;
    else magnitude = significand >> (uint32_t)(-shift);
    return (bits.u & 0x80000000u) ? 0u - magnitude : magnitude;
}

static int32_t vx_cast_i32_from_bits(uint32_t bits) {
    if (bits <= 0x7fffffffu) return (int32_t)bits;
    return (int32_t)(bits - 0x80000000u) - 2147483647 - 1;
}

static int32_t vx_cast_f32_to_i32(float value) {
    return vx_cast_i32_from_bits(vx_cast_f32_to_u32_mod(value));
}

static int8_t vx_cast_i8_from_u8(uint8_t value) {
    if (value <= 127u) return (int8_t)value;
    return (int8_t)(value - 128u) - 128;
}

static void vx_cast_integer_store(void *output, int dtype, int index,
        int32_t integer) {
    if (dtype == VX_CAST_I32) ((int32_t *)output)[index] = integer;
    else if (dtype == VX_CAST_I8) {
        ((int8_t *)output)[index] = vx_cast_i8_from_u8((uint8_t)integer);
    }
    else ((uint8_t *)output)[index] = (uint8_t)integer;
}

/* dtype: 0=F32, 1=I32, 2=I8, 3=U8. Integer writes retain typed-array wrapping. */
int cast_typed(const void *input, int input_dtype, void *output, int output_dtype, int n) {
    if (!input || !output || n < 0 || input_dtype < VX_CAST_F32 || input_dtype > VX_CAST_U8 ||
        output_dtype < VX_CAST_F32 || output_dtype > VX_CAST_U8) return 0;
    for (int index = 0; index < n; index++) {
        if (input_dtype == VX_CAST_F32) {
            float value = ((const float *)input)[index];
            if (output_dtype == VX_CAST_F32) ((float *)output)[index] = value;
            else vx_cast_integer_store(output, output_dtype, index, vx_cast_f32_to_i32(value));
        } else {
            int32_t value = vx_cast_integer_read(input, input_dtype, index);
            if (output_dtype == VX_CAST_F32) ((float *)output)[index] = (float)value;
            else vx_cast_integer_store(output, output_dtype, index, value);
        }
    }
    return 1;
}

void cast_i32_to_f32(const int* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = (float)input[i];
}
void cast_f32_to_i32(const float* input, int* output, int n) {
    for (int i = 0; i < n; i++) output[i] = (int)input[i];
}

void slice_4d_f32(const float* input, float* output, 
                  int in_n, int in_c, int in_h, int in_w,
                  int out_n, int out_c, int out_h, int out_w,
                  int start_n, int start_c, int start_h, int start_w,
                  int step_n, int step_c, int step_h, int step_w) {
    for (int on = 0; on < out_n; on++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_n_idx = start_n + on * step_n;
                    int in_c_idx = start_c + oc * step_c;
                    int in_h_idx = start_h + oh * step_h;
                    int in_w_idx = start_w + ow * step_w;
                    output[on * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow] = 
                        input[in_n_idx * (in_c * in_h * in_w) + in_c_idx * (in_h * in_w) + in_h_idx * in_w + in_w_idx];
                }
            }
        }
    }
}

void gather_4d_f32(const float* input, const float* indices, float* output,
                   int in_n, int in_c, int in_h, int in_w,
                   int num_indices, int axis) {
    // A simplified gather that replaces 'axis' with a 1D list of indices.
    // E.g., if axis == 1 (channel), output will have shape [in_n, num_indices, in_h, in_w]
    int out_n = axis == 0 ? num_indices : in_n;
    int out_c = axis == 1 ? num_indices : in_c;
    int out_h = axis == 2 ? num_indices : in_h;
    int out_w = axis == 3 ? num_indices : in_w;
    
    for (int on = 0; on < out_n; on++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_n_idx = axis == 0 ? (int)indices[on] : on;
                    int in_c_idx = axis == 1 ? (int)indices[oc] : oc;
                    int in_h_idx = axis == 2 ? (int)indices[oh] : oh;
                    int in_w_idx = axis == 3 ? (int)indices[ow] : ow;
                    
                    if (in_n_idx < 0) in_n_idx += in_n;
                    if (in_c_idx < 0) in_c_idx += in_c;
                    if (in_h_idx < 0) in_h_idx += in_h;
                    if (in_w_idx < 0) in_w_idx += in_w;
                    
                    output[on * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow] = 
                        input[in_n_idx * (in_c * in_h * in_w) + in_c_idx * (in_h * in_w) + in_h_idx * in_w + in_w_idx];
                }
            }
        }
    }
}

WASM_EXPORT("matmul_f32")
void matmul_f32(const float* input, const float* weight, const float* bias, float* output, int seq_len, int d_in, int d_out) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_out; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < d_in; k++) {
                sum += input[i * d_in + k] * weight[k * d_out + j];
            }
            output[i * d_out + j] = sum;
        }
    }
}


WASM_EXPORT("add_broadcast_f32")
void add_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        for (int i = 0; i < spatial; i++) {
            out[offset + i] = a[offset + i] + b_val;
        }
    }
}

WASM_EXPORT("mul_f32")
void mul_f32(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

WASM_EXPORT("mul_broadcast_f32")
void mul_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        for (int i = 0; i < spatial; i++) {
            out[offset + i] = a[offset + i] * b_val;
        }
    }
}
