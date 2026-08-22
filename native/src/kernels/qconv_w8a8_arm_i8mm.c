/*
 * Arm I8MM convolution for dense NHWC/OHWI W8A8 QConv2D.
 *
 * A complete image-sized im2col matrix is deliberately avoided. At most
 * 64 KiB of byte-domain activation rows is materialized at a time, then the
 * packed SMMLA QLinear kernel consumes it. Padding is represented by the
 * input zero point, exactly matching the canonical skipped-padding dot.
 */
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)
#error "qconv_w8a8_arm_i8mm.c requires an AArch64 +i8mm compiler target"
#endif

#include "qconv_w8a8_arm.h"
#include "packed_quant_gemm.h"
#include "w8a8_affine.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    VX_QCONV_I8MM_PANEL_BYTES = 64u * 1024u,
    VX_QCONV_I8MM_PANEL_ROWS_MAX = 60u,
    VX_QCONV_I8MM_MR = 6u,
    VX_QCONV_I8MM_PAIR_SIGNED_I8 = 1u,
};

typedef struct {
    const uint8_t* input;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t padding_top;
    uint32_t padding_left;
    uint8_t padding_byte;
} VxQConvI8MMFill;

static inline void vx_qconv_i8mm_fill_generic(
        const VxQConvI8MMFill* fill, uint8_t* destination,
        uint32_t image, uint32_t output_y, uint32_t output_x) {
    for (uint32_t kernel_y = 0; kernel_y < fill->kernel_height;
         kernel_y++) {
        const int64_t input_y = (int64_t)output_y * fill->stride_y +
            (int64_t)kernel_y * fill->dilation_y - fill->padding_top;
        for (uint32_t kernel_x = 0; kernel_x < fill->kernel_width;
             kernel_x++) {
            const int64_t input_x = (int64_t)output_x * fill->stride_x +
                (int64_t)kernel_x * fill->dilation_x - fill->padding_left;
            if (input_y < 0 || input_y >= fill->input_height ||
                input_x < 0 || input_x >= fill->input_width) {
                memset(destination, fill->padding_byte,
                       fill->input_channels);
            } else {
                const size_t source =
                    (((size_t)image * fill->input_height +
                      (uint32_t)input_y) * fill->input_width +
                     (uint32_t)input_x) * fill->input_channels;
                memcpy(destination, fill->input + source,
                       fill->input_channels);
            }
            destination += fill->input_channels;
        }
    }
}

/* With unit horizontal dilation, all valid pixels from one kernel row are
 * contiguous in NHWC storage. Copy that span once and fill only its padding
 * prefix/suffix instead of issuing one libc call and one bounds branch per
 * pixel. */
static inline void vx_qconv_i8mm_fill_contiguous_x(
        const VxQConvI8MMFill* fill, uint8_t* destination,
        uint32_t image, uint32_t output_y, uint32_t output_x) {
    const size_t kernel_row_bytes =
        (size_t)fill->kernel_width * fill->input_channels;
    const int64_t input_x_origin =
        (int64_t)output_x * fill->stride_x - fill->padding_left;
    const int interior_x = input_x_origin >= 0 &&
        input_x_origin + fill->kernel_width <= fill->input_width;
    const size_t input_row_bytes =
        (size_t)fill->input_width * fill->input_channels;
    if (interior_x) {
        const int64_t input_y_origin =
            (int64_t)output_y * fill->stride_y - fill->padding_top;
        for (uint32_t kernel_y = 0; kernel_y < fill->kernel_height;
             kernel_y++) {
            const int64_t input_y = input_y_origin +
                (int64_t)kernel_y * fill->dilation_y;
            if (input_y < 0 || input_y >= fill->input_height) {
                memset(destination, fill->padding_byte, kernel_row_bytes);
            } else {
                const size_t source =
                    (size_t)image * fill->input_height * input_row_bytes +
                    (size_t)input_y * input_row_bytes +
                    (size_t)input_x_origin * fill->input_channels;
                memcpy(destination, fill->input + source, kernel_row_bytes);
            }
            destination += kernel_row_bytes;
        }
        return;
    }
    int64_t valid_first = input_x_origin < 0 ? -input_x_origin : 0;
    int64_t valid_end = (int64_t)fill->input_width - input_x_origin;
    if (valid_first > fill->kernel_width)
        valid_first = fill->kernel_width;
    if (valid_end < valid_first) valid_end = valid_first;
    if (valid_end > fill->kernel_width) valid_end = fill->kernel_width;

    const size_t prefix_bytes =
        (size_t)valid_first * fill->input_channels;
    const size_t valid_bytes =
        (size_t)(valid_end - valid_first) * fill->input_channels;
    const size_t suffix_bytes =
        kernel_row_bytes - prefix_bytes - valid_bytes;
    for (uint32_t kernel_y = 0; kernel_y < fill->kernel_height;
         kernel_y++) {
        const int64_t input_y = (int64_t)output_y * fill->stride_y +
            (int64_t)kernel_y * fill->dilation_y - fill->padding_top;
        if (input_y < 0 || input_y >= fill->input_height) {
            memset(destination, fill->padding_byte, kernel_row_bytes);
        } else {
            if (prefix_bytes)
                memset(destination, fill->padding_byte, prefix_bytes);
            if (valid_bytes) {
                const size_t source =
                    (((size_t)image * fill->input_height +
                      (uint32_t)input_y) * fill->input_width +
                     (uint32_t)(input_x_origin + valid_first)) *
                    fill->input_channels;
                memcpy(destination + prefix_bytes,
                       fill->input + source, valid_bytes);
            }
            if (suffix_bytes)
                memset(destination + prefix_bytes + valid_bytes,
                       fill->padding_byte, suffix_bytes);
        }
        destination += kernel_row_bytes;
    }
}

/* The receipt encoder's grayscale stem has K=9. Avoid three tiny memcpy calls
 * per valid input row; direct byte copies are substantially cheaper at this
 * channel count. */
static inline void vx_qconv_i8mm_fill_3x3_c1(
        const VxQConvI8MMFill* fill, uint8_t* destination,
        uint32_t image, uint32_t output_y, uint32_t output_x) {
    const int64_t input_x_origin =
        (int64_t)output_x * fill->stride_x - fill->padding_left;
    for (uint32_t kernel_y = 0; kernel_y < 3u; kernel_y++) {
        const int64_t input_y = (int64_t)output_y * fill->stride_y +
            (int64_t)kernel_y - fill->padding_top;
        if (input_y < 0 || input_y >= fill->input_height) {
            destination[0] = fill->padding_byte;
            destination[1] = fill->padding_byte;
            destination[2] = fill->padding_byte;
        } else if (input_x_origin >= 0 &&
                   input_x_origin + 2 < fill->input_width) {
            const size_t source =
                (((size_t)image * fill->input_height +
                  (uint32_t)input_y) * fill->input_width +
                 (uint32_t)input_x_origin);
            destination[0] = fill->input[source];
            destination[1] = fill->input[source + 1u];
            destination[2] = fill->input[source + 2u];
        } else {
            for (uint32_t kernel_x = 0; kernel_x < 3u; kernel_x++) {
                const int64_t input_x = input_x_origin + kernel_x;
                destination[kernel_x] =
                    input_x < 0 || input_x >= fill->input_width
                    ? fill->padding_byte
                    : fill->input[
                        (((size_t)image * fill->input_height +
                          (uint32_t)input_y) * fill->input_width +
                         (uint32_t)input_x)];
            }
        }
        destination += 3u;
    }
}

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
        const void* packed_qlinear_weight) {
    const VxPackedQ8Header* header =
        (const VxPackedQ8Header*)packed_qlinear_weight;
    uint64_t reduction64;
    uint64_t locations64;
    size_t reduction;
    size_t locations;
    size_t input_elements;
    size_t output_elements;
    uint32_t panel_capacity;
    size_t panel_bytes;
    uint8_t* panel;
    const uint8_t padding_byte = input_dtype == VX_DTYPE_I8
        ? (uint8_t)(int8_t)input_zero_point : (uint8_t)input_zero_point;
    const int32_t signed_input_zero_point = input_zero_point -
        (input_dtype == VX_DTYPE_U8 ? 128 : 0);

    if (!input || !bias || !weight_scales || !weight_zero_points || !output ||
        !header || !batch || !input_height || !input_width ||
        !input_channels || !output_height || !output_width ||
        !output_channels || !kernel_height || !kernel_width ||
        !input_per_group || !stride_y || !stride_x || !dilation_y ||
        !dilation_x || groups != 1u || relu != 0u ||
        input_per_group != input_channels ||
        output_channels % VX_PACKED_Q8_NR != 0u ||
        weight_dtype != VX_DTYPE_I8 ||
        (input_dtype != VX_DTYPE_I8 && input_dtype != VX_DTYPE_U8) ||
        (output_dtype != VX_DTYPE_I8 && output_dtype != VX_DTYPE_U8)) return 0;

    reduction64 = (uint64_t)kernel_height * kernel_width * input_per_group;
    locations64 = (uint64_t)batch * output_height * output_width;
    if (!reduction64 || reduction64 > SIZE_MAX || reduction64 > UINT32_MAX ||
        !locations64 || locations64 > SIZE_MAX ||
        locations64 > UINT32_MAX) return 0;
    reduction = (size_t)reduction64;
    locations = (size_t)locations64;
    input_elements = (size_t)batch;
    output_elements = (size_t)batch;
    if (!vx_w8a8_mul_size(&input_elements, input_height) ||
        !vx_w8a8_mul_size(&input_elements, input_width) ||
        !vx_w8a8_mul_size(&input_elements, input_channels) ||
        !vx_w8a8_mul_size(&output_elements, output_height) ||
        !vx_w8a8_mul_size(&output_elements, output_width) ||
        !vx_w8a8_mul_size(&output_elements, output_channels) ||
        vx_w8a8_ranges_overlap(input, input_elements, output,
                               output_elements)) return 0;

    if (header->magic != VX_PACKED_Q8_MAGIC ||
        header->d_in != (uint32_t)reduction ||
        header->d_out != output_channels ||
        header->weight_dtype != VX_DTYPE_I8 ||
        !(header->pair_flags & VX_QCONV_I8MM_PAIR_SIGNED_I8) ||
        header->n_blocks != output_channels / VX_PACKED_Q8_NR ||
        header->pair_n_blocks != header->n_blocks ||
        header->pair_k_blocks != (reduction + 7u) / 8u ||
        header->sums_offset < sizeof(*header) ||
        (uint64_t)header->sums_offset +
            (uint64_t)output_channels * sizeof(int32_t) >
                header->pair_data_offset ||
        (uint64_t)header->pair_data_offset +
            (uint64_t)header->pair_n_blocks *
                header->pair_k_blocks * 64u != header->bytes) return 0;

    {
        const uint64_t centered_bound = reduction64 * 128u *
            (128u + (uint64_t)(signed_input_zero_point < 0
                ? -signed_input_zero_point : signed_input_zero_point));
        if (centered_bound > INT32_MAX) return 0;
        for (uint32_t channel = 0; channel < output_channels; channel++) {
            const uint64_t bias_magnitude = bias[channel] < 0
                ? (uint64_t)(-(int64_t)bias[channel])
                : (uint64_t)bias[channel];
            if (weight_zero_points[channel] != 0 ||
                bias_magnitude > (uint64_t)INT32_MAX - centered_bound)
                return 0;
        }
    }

    if (reduction > VX_QCONV_I8MM_PANEL_BYTES) return 0;
    panel_capacity = (uint32_t)(VX_QCONV_I8MM_PANEL_BYTES / reduction);
    if (panel_capacity > VX_QCONV_I8MM_PANEL_ROWS_MAX)
        panel_capacity = VX_QCONV_I8MM_PANEL_ROWS_MAX;
    if (panel_capacity >= VX_QCONV_I8MM_MR)
        panel_capacity -= panel_capacity % VX_QCONV_I8MM_MR;
    panel_bytes = (size_t)panel_capacity * reduction;
    panel = (uint8_t*)malloc(panel_bytes);
    if (!panel) return 0;

    VxQConvI8MMFill fill = {
        .input = (const uint8_t*)input,
        .input_height = input_height,
        .input_width = input_width,
        .input_channels = input_channels,
        .kernel_height = kernel_height,
        .kernel_width = kernel_width,
        .stride_y = stride_y,
        .stride_x = stride_x,
        .dilation_y = dilation_y,
        .dilation_x = dilation_x,
        .padding_top = padding_top,
        .padding_left = padding_left,
        .padding_byte = padding_byte,
    };
    uint32_t image = 0u;
    uint32_t output_y = 0u;
    uint32_t output_x = 0u;
    const int grayscale_3x3 = input_channels == 1u &&
        kernel_height == 3u && kernel_width == 3u &&
        dilation_y == 1u && dilation_x == 1u;
    for (size_t location_base = 0; location_base < locations;
         location_base += panel_capacity) {
        uint32_t panel_rows = (uint32_t)(locations - location_base);
        if (panel_rows > panel_capacity) panel_rows = panel_capacity;
        for (uint32_t row = 0; row < panel_rows; row++) {
            uint8_t* destination = panel + (size_t)row * reduction;
            if (grayscale_3x3) {
                vx_qconv_i8mm_fill_3x3_c1(&fill, destination,
                    image, output_y, output_x);
            } else if (dilation_x == 1u) {
                vx_qconv_i8mm_fill_contiguous_x(&fill, destination,
                    image, output_y, output_x);
            } else {
                vx_qconv_i8mm_fill_generic(&fill, destination,
                    image, output_y, output_x);
            }
            output_x++;
            if (output_x == output_width) {
                output_x = 0u;
                output_y++;
                if (output_y == output_height) {
                    output_y = 0u;
                    image++;
                }
            }
        }

        if (!vx_qlinear_i8u8_arm_i8mm_packed_try(panel, header, bias,
                weight_scales, weight_zero_points,
                (uint8_t*)output + location_base * output_channels,
                panel_rows, input_scale, input_zero_point, output_scale,
                output_zero_point, input_dtype, output_dtype)) {
            free(panel);
            return 0;
        }
    }
    free(panel);
    return 1;
}
