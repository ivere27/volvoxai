#ifndef VOLVOX_Q_LINEAR_W8A8_ARM_INTERNAL_H
#define VOLVOX_Q_LINEAR_W8A8_ARM_INTERNAL_H

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "../../include/volvoxai_enums.h"
#include "w8a8_affine.h"

enum {
    VX_W8A8_ARM_I8 = VX_DTYPE_I8,
    VX_W8A8_ARM_U8 = VX_DTYPE_U8,
};

typedef struct {
    const void* input;
    const void* weight;
    const int32_t* bias;
    const float* weight_scales;
    const int32_t* weight_zero_points;
    void* output;
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    uint32_t input_dtype;
    uint32_t weight_dtype;
    uint32_t output_dtype;
} VxW8A8ArmQLinearArgs;

static inline int vx_w8a8_arm_positive_product_multiplier(
        float input_scale, float weight_scale, float output_scale) {
    volatile float product_scale = input_scale * weight_scale;
    volatile float multiplier = product_scale / output_scale;
    return vx_w8a8_finite_f32(multiplier) && multiplier > 0.0f;
}

/* The portable ABI checks I32 range after every product. A vector reduction
 * may group products differently, so only use an ARM vector kernel when this
 * absolute bound proves every prefix is in range. Otherwise the caller keeps
 * the portable kernel and its exact failure behavior. */
static inline int vx_w8a8_arm_eligible(const VxW8A8ArmQLinearArgs* args,
                                       uint32_t minimum_width) {
    size_t input_elements;
    size_t weight_elements;
    size_t output_elements;
    uint64_t product_bound;
    if (!args || !args->input || !args->weight || !args->bias ||
        !args->weight_scales || !args->weight_zero_points || !args->output ||
        !args->rows || args->d_in < minimum_width || !args->d_out ||
        !vx_w8a8_byte_dtype(args->input_dtype) ||
        !vx_w8a8_byte_dtype(args->weight_dtype) ||
        !vx_w8a8_byte_dtype(args->output_dtype) ||
        !vx_w8a8_finite_f32(args->input_scale) || args->input_scale <= 0.0f ||
        !vx_w8a8_finite_f32(args->output_scale) || args->output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(args->input_zero_point, args->input_dtype) ||
        !vx_w8a8_zero_point_valid(args->output_zero_point, args->output_dtype)) return 0;
    input_elements = args->rows;
    weight_elements = args->d_out;
    output_elements = args->rows;
    if (!vx_w8a8_mul_size(&input_elements, args->d_in) ||
        !vx_w8a8_mul_size(&weight_elements, args->d_in) ||
        !vx_w8a8_mul_size(&output_elements, args->d_out)) return 0;
    product_bound = (uint64_t)args->d_in * 65025u;
    if (product_bound > (uint64_t)INT32_MAX) return 0;
    for (uint32_t column = 0; column < args->d_out; column++) {
        uint64_t bias_magnitude = args->bias[column] < 0
            ? (uint64_t)(-(int64_t)args->bias[column]) : (uint64_t)args->bias[column];
        if (!vx_w8a8_finite_f32(args->weight_scales[column]) ||
            args->weight_scales[column] <= 0.0f ||
            !vx_w8a8_arm_positive_product_multiplier(
                args->input_scale, args->weight_scales[column],
                args->output_scale) ||
            !vx_w8a8_zero_point_valid(args->weight_zero_points[column],
                                           args->weight_dtype) ||
            bias_magnitude > (uint64_t)INT32_MAX - product_bound) return 0;
    }
    return 1;
}

/* Implemented only by the separately compiled +dotprod object. */
int vx_qlinear_i8u8_arm_dotprod_try(const VxW8A8ArmQLinearArgs* args);

#endif
