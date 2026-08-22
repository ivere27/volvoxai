/*
 * Native CPU layout kernels for W8A8 Transpose.
 *
 * transpose_nd_i8u8() remains the fully validating portable/WASM ABI.  The
 * native runtime calls this file only after descriptor validation.  Common
 * image-layout and batched-matrix permutations use cache-sized tiles and
 * avoid integer division/modulo for every output byte.
 */
#include "quant_cpu_isa.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#define VX_TRANSPOSE_W8A8_ARM_NEON 1
#else
#define VX_TRANSPOSE_W8A8_ARM_NEON 0
#endif

extern int transpose_nd_i8u8(const uint8_t* input, uint8_t* output,
        const uint32_t* input_shape, const uint32_t* permutation,
        uint32_t rank, uint32_t elements, uint32_t dtype);

enum {
    VX_TRANSPOSE_SPATIAL_TILE = 64u,
    VX_TRANSPOSE_MATRIX_TILE = 32u,
};

typedef struct {
    const uint8_t* input;
    uint8_t* output;
    size_t area;
    uint32_t batch;
    uint32_t channels;
    uint32_t spatial_tiles;
    int use_neon;
} VxTransposeImageContext;

typedef struct {
    const uint8_t* input;
    uint8_t* output;
    uint32_t batch;
    uint32_t rows;
    uint32_t columns;
    uint32_t row_tiles;
    uint32_t column_tiles;
    int use_neon;
} VxTransposeMatrixContext;

#if VX_TRANSPOSE_W8A8_ARM_NEON
/* Transpose one byte 8x8 block.  Signed and unsigned tensors share
 * this routine because Transpose is defined on storage bytes, not values. */
static inline void vx_transpose_u8_8x8_neon(
        const uint8_t* source, size_t source_stride,
        uint8_t* destination, size_t destination_stride) {
    const uint8x8_t r0 = vld1_u8(source + 0 * source_stride);
    const uint8x8_t r1 = vld1_u8(source + 1 * source_stride);
    const uint8x8_t r2 = vld1_u8(source + 2 * source_stride);
    const uint8x8_t r3 = vld1_u8(source + 3 * source_stride);
    const uint8x8_t r4 = vld1_u8(source + 4 * source_stride);
    const uint8x8_t r5 = vld1_u8(source + 5 * source_stride);
    const uint8x8_t r6 = vld1_u8(source + 6 * source_stride);
    const uint8x8_t r7 = vld1_u8(source + 7 * source_stride);
    const uint8x8x2_t t01 = vtrn_u8(r0, r1);
    const uint8x8x2_t t23 = vtrn_u8(r2, r3);
    const uint8x8x2_t t45 = vtrn_u8(r4, r5);
    const uint8x8x2_t t67 = vtrn_u8(r6, r7);
    const uint16x4x2_t u0 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[0]), vreinterpret_u16_u8(t23.val[0]));
    const uint16x4x2_t u1 = vtrn_u16(
        vreinterpret_u16_u8(t01.val[1]), vreinterpret_u16_u8(t23.val[1]));
    const uint16x4x2_t u2 = vtrn_u16(
        vreinterpret_u16_u8(t45.val[0]), vreinterpret_u16_u8(t67.val[0]));
    const uint16x4x2_t u3 = vtrn_u16(
        vreinterpret_u16_u8(t45.val[1]), vreinterpret_u16_u8(t67.val[1]));
    const uint32x2x2_t v0 = vtrn_u32(
        vreinterpret_u32_u16(u0.val[0]), vreinterpret_u32_u16(u2.val[0]));
    const uint32x2x2_t v1 = vtrn_u32(
        vreinterpret_u32_u16(u1.val[0]), vreinterpret_u32_u16(u3.val[0]));
    const uint32x2x2_t v2 = vtrn_u32(
        vreinterpret_u32_u16(u0.val[1]), vreinterpret_u32_u16(u2.val[1]));
    const uint32x2x2_t v3 = vtrn_u32(
        vreinterpret_u32_u16(u1.val[1]), vreinterpret_u32_u16(u3.val[1]));
    vst1_u8(destination + 0 * destination_stride,
            vreinterpret_u8_u32(v0.val[0]));
    vst1_u8(destination + 1 * destination_stride,
            vreinterpret_u8_u32(v1.val[0]));
    vst1_u8(destination + 2 * destination_stride,
            vreinterpret_u8_u32(v2.val[0]));
    vst1_u8(destination + 3 * destination_stride,
            vreinterpret_u8_u32(v3.val[0]));
    vst1_u8(destination + 4 * destination_stride,
            vreinterpret_u8_u32(v0.val[1]));
    vst1_u8(destination + 5 * destination_stride,
            vreinterpret_u8_u32(v1.val[1]));
    vst1_u8(destination + 6 * destination_stride,
            vreinterpret_u8_u32(v2.val[1]));
    vst1_u8(destination + 7 * destination_stride,
            vreinterpret_u8_u32(v3.val[1]));
}
#endif

static void vx_transpose_nhwc_to_nchw_worker(void* opaque, int begin, int end) {
    VxTransposeImageContext* context = (VxTransposeImageContext*)opaque;
    for (int task = begin; task < end; task++) {
        const uint32_t batch_index =
            (uint32_t)task / context->spatial_tiles;
        const uint32_t tile = (uint32_t)task % context->spatial_tiles;
        size_t spatial_begin =
            (size_t)tile * VX_TRANSPOSE_SPATIAL_TILE;
        size_t spatial_end = spatial_begin + VX_TRANSPOSE_SPATIAL_TILE;
        const size_t input_sample =
            (size_t)batch_index * context->area * context->channels;
        const size_t output_sample =
            (size_t)batch_index * context->channels * context->area;
        if (spatial_end > context->area) spatial_end = context->area;
#if VX_TRANSPOSE_W8A8_ARM_NEON
        if (context->use_neon && context->channels >= 8u) {
            size_t spatial = spatial_begin;
            for (; spatial + 8u <= spatial_end; spatial += 8u) {
                uint32_t channel = 0;
                for (; channel + 8u <= context->channels; channel += 8u) {
                    vx_transpose_u8_8x8_neon(
                        context->input + input_sample +
                            spatial * context->channels + channel,
                        context->channels,
                        context->output + output_sample +
                            (size_t)channel * context->area + spatial,
                        context->area);
                }
                for (; channel < context->channels; channel++) {
                    uint8_t* destination = context->output + output_sample +
                        (size_t)channel * context->area + spatial;
                    const uint8_t* source = context->input + input_sample +
                        spatial * context->channels + channel;
                    for (uint32_t local = 0; local < 8u; local++) {
                        destination[local] = *source;
                        source += context->channels;
                    }
                }
            }
            spatial_begin = spatial;
        }
#endif
        for (uint32_t channel = 0; channel < context->channels; channel++) {
            uint8_t* destination = context->output + output_sample +
                (size_t)channel * context->area + spatial_begin;
            const uint8_t* source = context->input + input_sample +
                spatial_begin * context->channels + channel;
            for (size_t spatial = spatial_begin; spatial < spatial_end;
                 spatial++) {
                *destination++ = *source;
                source += context->channels;
            }
        }
    }
}

static void vx_transpose_nchw_to_nhwc_worker(void* opaque, int begin, int end) {
    VxTransposeImageContext* context = (VxTransposeImageContext*)opaque;
    for (int task = begin; task < end; task++) {
        const uint32_t batch_index =
            (uint32_t)task / context->spatial_tiles;
        const uint32_t tile = (uint32_t)task % context->spatial_tiles;
        size_t spatial_begin =
            (size_t)tile * VX_TRANSPOSE_SPATIAL_TILE;
        size_t spatial_end = spatial_begin + VX_TRANSPOSE_SPATIAL_TILE;
        const size_t input_sample =
            (size_t)batch_index * context->channels * context->area;
        const size_t output_sample =
            (size_t)batch_index * context->area * context->channels;
        if (spatial_end > context->area) spatial_end = context->area;
#if VX_TRANSPOSE_W8A8_ARM_NEON
        if (context->use_neon && context->channels >= 8u) {
            size_t spatial = spatial_begin;
            for (; spatial + 8u <= spatial_end; spatial += 8u) {
                uint32_t channel = 0;
                for (; channel + 8u <= context->channels; channel += 8u) {
                    vx_transpose_u8_8x8_neon(
                        context->input + input_sample +
                            (size_t)channel * context->area + spatial,
                        context->area,
                        context->output + output_sample +
                            spatial * context->channels + channel,
                        context->channels);
                }
                for (uint32_t local = 0; local < 8u; local++) {
                    uint8_t* destination = context->output + output_sample +
                        (spatial + local) * context->channels + channel;
                    const uint8_t* source = context->input + input_sample +
                        (size_t)channel * context->area + spatial + local;
                    for (uint32_t tail = channel; tail < context->channels;
                         tail++) {
                        *destination++ = *source;
                        source += context->area;
                    }
                }
            }
            spatial_begin = spatial;
        }
#endif
        for (size_t spatial = spatial_begin; spatial < spatial_end; spatial++) {
            uint8_t* destination = context->output + output_sample +
                spatial * context->channels;
            const uint8_t* source = context->input + input_sample + spatial;
            for (uint32_t channel = 0; channel < context->channels; channel++) {
                destination[channel] = *source;
                source += context->area;
            }
        }
    }
}

static void vx_transpose_batched_matrix_worker(void* opaque,
                                                int begin, int end) {
    VxTransposeMatrixContext* context = (VxTransposeMatrixContext*)opaque;
    const uint32_t tiles_per_batch =
        context->row_tiles * context->column_tiles;
    for (int task = begin; task < end; task++) {
        const uint32_t batch_index = (uint32_t)task / tiles_per_batch;
        const uint32_t local_tile = (uint32_t)task % tiles_per_batch;
        const uint32_t row_tile = local_tile / context->column_tiles;
        const uint32_t column_tile = local_tile % context->column_tiles;
        const uint32_t row_begin = row_tile * VX_TRANSPOSE_MATRIX_TILE;
        const uint32_t column_begin =
            column_tile * VX_TRANSPOSE_MATRIX_TILE;
        uint32_t row_end = row_begin + VX_TRANSPOSE_MATRIX_TILE;
        uint32_t column_end = column_begin + VX_TRANSPOSE_MATRIX_TILE;
        const size_t input_sample =
            (size_t)batch_index * context->rows * context->columns;
        const size_t output_sample =
            (size_t)batch_index * context->columns * context->rows;
        if (row_end > context->rows) row_end = context->rows;
        if (column_end > context->columns) column_end = context->columns;
        uint32_t row = row_begin;
#if VX_TRANSPOSE_W8A8_ARM_NEON
        if (context->use_neon) {
            for (; row + 8u <= row_end; row += 8u) {
                uint32_t column = column_begin;
                for (; column + 8u <= column_end; column += 8u) {
                    vx_transpose_u8_8x8_neon(
                        context->input + input_sample +
                            (size_t)row * context->columns + column,
                        context->columns,
                        context->output + output_sample +
                            (size_t)column * context->rows + row,
                        context->rows);
                }
                for (; column < column_end; column++) {
                    for (uint32_t local = 0; local < 8u; local++) {
                        context->output[output_sample +
                            (size_t)column * context->rows + row + local] =
                            context->input[input_sample +
                                (size_t)(row + local) * context->columns +
                                column];
                    }
                }
            }
        }
#endif
        for (; row < row_end; row++) {
            const uint8_t* source = context->input + input_sample +
                (size_t)row * context->columns + column_begin;
            for (uint32_t column = column_begin; column < column_end;
                 column++) {
                context->output[output_sample +
                    (size_t)column * context->rows + row] = *source++;
            }
        }
    }
}

static int vx_transpose_is_identity(const uint32_t* permutation,
                                    uint32_t rank) {
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (permutation[axis] != axis) return 0;
    }
    return 1;
}

int vx_transpose_nd_i8u8_native_validated(
        const uint8_t* input, uint8_t* output, const uint32_t* input_shape,
        const uint32_t* permutation, uint32_t rank, uint32_t elements,
        uint32_t dtype) {
    uint64_t product = 1;
    if (!input || !output || input == output || !input_shape || !permutation ||
        !rank || rank > 8 ||
        (dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8)) {
        return transpose_nd_i8u8(input, output, input_shape, permutation,
                                 rank, elements, dtype);
    }
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (!input_shape[axis] ||
            product > UINT64_MAX / input_shape[axis]) {
            return transpose_nd_i8u8(input, output, input_shape, permutation,
                                     rank, elements, dtype);
        }
        product *= input_shape[axis];
    }
    if (product != elements) {
        return transpose_nd_i8u8(input, output, input_shape, permutation,
                                 rank, elements, dtype);
    }
    if (vx_transpose_is_identity(permutation, rank)) {
        memcpy(output, input, elements);
        return 1;
    }
    if (rank == 4 && permutation[0] == 0 && permutation[1] == 3 &&
        permutation[2] == 1 && permutation[3] == 2) {
        VxTransposeImageContext context;
        const uint64_t area = (uint64_t)input_shape[1] * input_shape[2];
        const uint64_t spatial_tiles =
            (area + VX_TRANSPOSE_SPATIAL_TILE - 1u) /
            VX_TRANSPOSE_SPATIAL_TILE;
        const uint64_t tasks = (uint64_t)input_shape[0] * spatial_tiles;
        if (area <= SIZE_MAX && spatial_tiles <= UINT32_MAX &&
            tasks <= INT_MAX) {
            context = (VxTransposeImageContext){
                .input = input, .output = output, .area = (size_t)area,
                .batch = input_shape[0], .channels = input_shape[3],
                .spatial_tiles = (uint32_t)spatial_tiles,
                .use_neon = VX_TRANSPOSE_W8A8_ARM_NEON &&
                    vx_kernel_platform()->has_neon,
            };
            vx_kernels_parallel_for((int)tasks, 1,
                vx_transpose_nhwc_to_nchw_worker, &context);
            return 1;
        }
    }
    if (rank == 4 && permutation[0] == 0 && permutation[1] == 2 &&
        permutation[2] == 3 && permutation[3] == 1) {
        VxTransposeImageContext context;
        const uint64_t area = (uint64_t)input_shape[2] * input_shape[3];
        const uint64_t spatial_tiles =
            (area + VX_TRANSPOSE_SPATIAL_TILE - 1u) /
            VX_TRANSPOSE_SPATIAL_TILE;
        const uint64_t tasks = (uint64_t)input_shape[0] * spatial_tiles;
        if (area <= SIZE_MAX && spatial_tiles <= UINT32_MAX &&
            tasks <= INT_MAX) {
            context = (VxTransposeImageContext){
                .input = input, .output = output, .area = (size_t)area,
                .batch = input_shape[0], .channels = input_shape[1],
                .spatial_tiles = (uint32_t)spatial_tiles,
                .use_neon = VX_TRANSPOSE_W8A8_ARM_NEON &&
                    vx_kernel_platform()->has_neon,
            };
            vx_kernels_parallel_for((int)tasks, 1,
                vx_transpose_nchw_to_nhwc_worker, &context);
            return 1;
        }
    }
    if (rank == 3 && permutation[0] == 0 && permutation[1] == 2 &&
        permutation[2] == 1) {
        VxTransposeMatrixContext context;
        const uint32_t row_tiles =
            (input_shape[1] + VX_TRANSPOSE_MATRIX_TILE - 1u) /
            VX_TRANSPOSE_MATRIX_TILE;
        const uint32_t column_tiles =
            (input_shape[2] + VX_TRANSPOSE_MATRIX_TILE - 1u) /
            VX_TRANSPOSE_MATRIX_TILE;
        const uint64_t tasks = (uint64_t)input_shape[0] * row_tiles *
            column_tiles;
        if (tasks <= INT_MAX) {
            context = (VxTransposeMatrixContext){
                .input = input, .output = output, .batch = input_shape[0],
                .rows = input_shape[1], .columns = input_shape[2],
                .row_tiles = row_tiles, .column_tiles = column_tiles,
                .use_neon = VX_TRANSPOSE_W8A8_ARM_NEON &&
                    vx_kernel_platform()->has_neon,
            };
            vx_kernels_parallel_for((int)tasks, 1,
                vx_transpose_batched_matrix_worker, &context);
            return 1;
        }
    }
    return transpose_nd_i8u8(input, output, input_shape, permutation,
                             rank, elements, dtype);
}
