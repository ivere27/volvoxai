#ifndef VOLVOXAI_QLINEAR_MULTIPLIER_H
#define VOLVOXAI_QLINEAR_MULTIPLIER_H

#include <math.h>
#include <stdint.h>

/*
 * Build the exact multiplier consumed by canonical QLinear.  The volatile
 * intermediates are intentional: the portable/TypeScript contract rounds
 * (input_scale * weight_scale) to F32 before dividing by output_scale.
 */
static inline int vx_qlinear_compute_multiplier(
        float input_scale, float weight_scale, float output_scale,
        float* multiplier_out) {
    volatile float scale_product;
    volatile float multiplier;
    if (!multiplier_out ||
        !isfinite(input_scale) || input_scale <= 0.0f ||
        !isfinite(weight_scale) || weight_scale <= 0.0f ||
        !isfinite(output_scale) || output_scale <= 0.0f)
        return 0;
    scale_product = input_scale * weight_scale;
    multiplier = scale_product / output_scale;
    if (!isfinite(multiplier) || multiplier <= 0.0f) return 0;
    *multiplier_out = multiplier;
    return 1;
}

static inline int vx_qlinear_build_multipliers(
        float input_scale, const float* weight_scales,
        float output_scale, uint32_t channels, float* multipliers) {
    if (!weight_scales || !multipliers || channels == 0u)
        return 0;
    for (uint32_t channel = 0u; channel < channels; channel++) {
        if (!vx_qlinear_compute_multiplier(
                input_scale, weight_scales[channel], output_scale,
                &multipliers[channel]))
            return 0;
    }
    return 1;
}

#endif
