#ifndef VOLVOX_Q_CONV_W8A8_ARM_H
#define VOLVOX_Q_CONV_W8A8_ARM_H

#include <stdint.h>

/* Native-only ARM candidate for canonical byte NHWC/OHWI QConv2D.
 * It returns 1 only when it wrote a complete result. A 0 return leaves output
 * untouched and asks the caller to run portable qconv2d_i8u8 instead. */
int vx_qconv2d_i8u8_arm_try(const void* input, const void* weight,
                             const int32_t* bias, const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t batch, uint32_t input_height,
                             uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width,
                             uint32_t output_channels, uint32_t kernel_height,
                             uint32_t kernel_width, uint32_t input_per_group,
                             uint32_t stride_y, uint32_t stride_x,
                             uint32_t dilation_y, uint32_t dilation_x,
                             uint32_t padding_top, uint32_t padding_left,
                             uint32_t padding_bottom, uint32_t padding_right,
                             uint32_t groups, uint32_t relu, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, uint32_t input_dtype,
                             uint32_t weight_dtype, uint32_t output_dtype);

int vx_qconv2d_i8u8_arm_prepacked_try(const void* input, const void* weight,
                             const int32_t* bias, const float* weight_scales,
                             const int32_t* weight_zero_points, void* output,
                             uint32_t batch, uint32_t input_height,
                             uint32_t input_width, uint32_t input_channels,
                             uint32_t output_height, uint32_t output_width,
                             uint32_t output_channels, uint32_t kernel_height,
                             uint32_t kernel_width, uint32_t input_per_group,
                             uint32_t stride_y, uint32_t stride_x,
                             uint32_t dilation_y, uint32_t dilation_x,
                             uint32_t padding_top, uint32_t padding_left,
                             uint32_t padding_bottom, uint32_t padding_right,
                             uint32_t groups, uint32_t relu, float input_scale,
                             int32_t input_zero_point, float output_scale,
                             int32_t output_zero_point, uint32_t input_dtype,
                             uint32_t weight_dtype, uint32_t output_dtype,
                             const void* packed_qlinear_weight,
                             const void* small_c_packed_weight);

#if defined(VOLVOXAI_ARM_I8MM_OBJECT) && defined(__aarch64__)
int vx_qconv2d_i8u8_arm_i8mm_prepacked_try(const void* input,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height,
        uint32_t output_width, uint32_t output_channels,
        uint32_t kernel_height, uint32_t kernel_width,
        uint32_t input_per_group, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t groups, uint32_t relu,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t weight_dtype, uint32_t output_dtype,
        const void* packed_qlinear_weight);
#endif

/* Internal entry implemented only by the separately compiled Armv8.2+dotprod
 * object. The baseline caller validates every argument and HWCAP_ASIMDDP before
 * reaching it; it is not part of the public VolvoxAI ABI. */
int vx_qconv2d_i8u8_arm_dotprod_try(const void* input, const void* weight,
                                     const int32_t* bias, const float* weight_scales,
                                     const int32_t* weight_zero_points, void* output,
                                     uint32_t batch, uint32_t input_height,
                                     uint32_t input_width, uint32_t input_channels,
                                     uint32_t output_height, uint32_t output_width,
                                     uint32_t output_channels, uint32_t kernel_height,
                                     uint32_t kernel_width, uint32_t input_per_group,
                                     uint32_t stride_y, uint32_t stride_x,
                                     uint32_t dilation_y, uint32_t dilation_x,
                                     uint32_t padding_top, uint32_t padding_left,
                                     uint32_t padding_bottom, uint32_t padding_right,
                                     uint32_t groups, uint32_t relu, float input_scale,
                                     int32_t input_zero_point, float output_scale,
                                     int32_t output_zero_point, uint32_t input_dtype,
                                     uint32_t weight_dtype, uint32_t output_dtype);

#endif
