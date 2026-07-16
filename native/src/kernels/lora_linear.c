#include "lora_linear.h"

#include <stdint.h>
#include <stddef.h>

static uint16_t vx_lora_load_u16_le(const void* data, size_t index) {
    const unsigned char* bytes = (const unsigned char*)data + index * 2u;
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static uint32_t vx_lora_load_u32_le(const void* data, size_t index) {
    const unsigned char* bytes = (const unsigned char*)data + index * 4u;
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static float vx_lora_f16_to_f32(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)half & 0x03ffu;
    uint32_t bits;
    union { uint32_t bits; float value; } converted;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int adjusted_exponent = -14;
            while (!(mantissa & 0x0400u)) {
                mantissa <<= 1;
                adjusted_exponent--;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((uint32_t)(adjusted_exponent + 127) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    converted.bits = bits;
    return converted.value;
}

static float vx_lora_base_value(const void* data, int dtype, size_t index) {
    if (dtype == VX_LORA_DTYPE_F32) {
        union { uint32_t bits; float value; } converted = {
            vx_lora_load_u32_le(data, index)
        };
        return converted.value;
    }
    return vx_lora_f16_to_f32(vx_lora_load_u16_le(data, index));
}

int vx_lora_apply_row_f32(const float* input, const float* a, const float* b,
                          const float* bias, float* output,
                          float* rank_workspace, int d_in, int rank, int d_out,
                          float adapter_scale, float route_scale) {
    if (!output || d_in <= 0 || rank < 0 || d_out <= 0 ||
        (rank > 0 && (!input || !a || !b || !rank_workspace)))
        return 0;
    if (rank > 0) {
        for (int component = 0; component < rank; component++) {
            float sum = 0.0f;
            for (int dimension = 0; dimension < d_in; dimension++)
                sum += input[dimension] *
                       a[(size_t)dimension * rank + component];
            rank_workspace[component] = sum;
        }
        for (int column = 0; column < d_out; column++) {
            float delta = 0.0f;
            for (int component = 0; component < rank; component++)
                delta += rank_workspace[component] *
                         b[(size_t)component * d_out + column];
            delta *= adapter_scale;
            output[column] += route_scale * delta;
        }
    }
    if (bias) {
        for (int column = 0; column < d_out; column++)
            output[column] += bias[column];
    }
    return 1;
}

int vx_lora_materialize_weight_f32(const void* base_weight, int base_dtype,
                                   int base_out_in, const float* a,
                                   const float* b, float adapter_scale,
                                   float* output, int d_in, int rank,
                                   int d_out) {
    if (!base_weight || !a || !b || !output ||
        (base_dtype != VX_LORA_DTYPE_F32 &&
         base_dtype != VX_LORA_DTYPE_F16) ||
        (base_out_in != 0 && base_out_in != 1) || d_in <= 0 || rank <= 0 ||
        d_out <= 0 || (size_t)d_in > (size_t)-1 / (size_t)d_out ||
        (size_t)d_in > (size_t)-1 / (size_t)rank ||
        (size_t)rank > (size_t)-1 / (size_t)d_out)
        return 0;
    for (int dimension = 0; dimension < d_in; dimension++) {
        for (int column = 0; column < d_out; column++) {
            size_t base_index = base_out_in ?
                (size_t)column * d_in + dimension :
                (size_t)dimension * d_out + column;
            float delta = 0.0f;
            for (int component = 0; component < rank; component++)
                delta += a[(size_t)dimension * rank + component] *
                         b[(size_t)component * d_out + column];
            delta *= adapter_scale;
            output[(size_t)dimension * d_out + column] =
                vx_lora_base_value(base_weight, base_dtype, base_index) + delta;
        }
    }
    return 1;
}
