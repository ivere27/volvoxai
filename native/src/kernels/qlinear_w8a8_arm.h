#ifndef VOLVOX_Q_LINEAR_W8A8_ARM_H
#define VOLVOX_Q_LINEAR_W8A8_ARM_H

#include <stdint.h>

/* Native-only ARM candidate for the canonical W8A8 QLinear ABI.
 * It returns 1 only after writing a complete result; 0 means that the caller
 * must use the authoritative portable qlinear_i8u8 implementation instead.
 * The declaration is intentionally separate from quant_cpu_isa.h so the ARM
 * implementation can evolve without coupling to the W8A32 kernels. */
int vx_qlinear_i8u8_arm_try(const void* input, const void* weight,
                            const int32_t* bias, const float* weight_scales,
                            const int32_t* weight_zero_points, void* output,
                            uint32_t rows, uint32_t d_in, uint32_t d_out,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t weight_dtype,
                            uint32_t output_dtype);

#endif
