#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metal_engine.h"
#include "runtime_state.h"

/* macOS: cmake -B build/mac -DVOLVOXAI_ENABLE_METAL=ON && ctest --test-dir build/mac -L gpu */

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "metal training check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int close_enough(float a, float b) {
    return fabsf(a - b) <= 1.0e-4f;
}

static void fill_f32_matrix(float* values, size_t count, int multiplier, int modulus,
                            int center, float scale) {
    for (size_t i = 0; i < count; i++) {
        values[i] = (float)(((int)(i % (size_t)modulus) * multiplier) % modulus - center) * scale;
    }
}

static int check_graph_linear_case(int rows, int d_in, int d_out, int output_major,
                                   float* input, float* input_major_weight, float* bias,
                                   float* output) {
    float expected[17 * 23];
    float output_major_weight[19 * 23];
    const float* selected_weight = input_major_weight;
    if (output_major) {
        for (int k = 0; k < d_in; k++) for (int column = 0; column < d_out; column++) {
            output_major_weight[column * d_in + k] = input_major_weight[k * d_out + column];
        }
        selected_weight = output_major_weight;
    }
    for (int row = 0; row < rows; row++) for (int column = 0; column < d_out; column++) {
        float sum = bias[column];
        for (int k = 0; k < d_in; k++) {
            sum += input[row * d_in + k] * input_major_weight[k * d_out + column];
        }
        expected[row * d_out + column] = sum;
    }
    memset(output, 0, (size_t)rows * d_out * sizeof(float));
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_linear_f32(input, selected_weight, bias, output,
                                 rows, d_in, d_out, output_major) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output, (size_t)rows * d_out * sizeof(float), 0) == 1);
    for (int i = 0; i < rows * d_out; i++) CHECK(close_enough(output[i], expected[i]));
    metal_graph_reset();
    return 0;
}

static int test_graph_linear_tails_and_fallback(void) {
    float tiled_input[17 * 19];
    float tiled_weight[19 * 23];
    float tiled_bias[23];
    float tiled_output[17 * 23];
    float scalar_input[3 * 7];
    float scalar_weight[7 * 5];
    float scalar_bias[5];
    float scalar_output[3 * 5];
    fill_f32_matrix(tiled_input, 17u * 19u, 7, 17, 8, 0.0625f);
    fill_f32_matrix(tiled_weight, 19u * 23u, 11, 13, 6, 0.125f);
    fill_f32_matrix(tiled_bias, 23u, 3, 5, 2, 0.25f);
    fill_f32_matrix(scalar_input, 3u * 7u, 5, 11, 5, 0.125f);
    fill_f32_matrix(scalar_weight, 7u * 5u, 3, 7, 3, 0.25f);
    fill_f32_matrix(scalar_bias, 5u, 2, 5, 2, 0.5f);
    CHECK(check_graph_linear_case(17, 19, 23, 1, tiled_input, tiled_weight,
                                  tiled_bias, tiled_output) == 0);
    CHECK(check_graph_linear_case(3, 7, 5, 0, scalar_input, scalar_weight,
                                  scalar_bias, scalar_output) == 0);
    return 0;
}

static int test_graph_f32_shape_and_activation_closure(void) {
    const float sigmoid_input[5] = {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f};
    float sigmoid_output[5] = {0};
    const float softmax_input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
    float softmax_output[6] = {0};
    const float expand_input[2] = {2.0f, -3.0f};
    const int expand_input_shape[5] = {1, 1, 1, 2, 1};
    const int expand_output_shape[5] = {1, 2, 1, 2, 3};
    const float expected_expand[12] = {
        2.0f, 2.0f, 2.0f, -3.0f, -3.0f, -3.0f,
        2.0f, 2.0f, 2.0f, -3.0f, -3.0f, -3.0f,
    };
    float expand_output[12] = {0};
    const float gather_input[12] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    };
    const int32_t gather_indices[2] = {2, -1};
    const float expected_gather[8] = {
        4.0f, 5.0f, 4.0f, 5.0f, 10.0f, 11.0f, 10.0f, 11.0f,
    };
    float gather_output[8] = {0};
    float slice_input[24];
    const int slice_input_shape[4] = {1, 2, 3, 4};
    const int slice_output_shape[4] = {1, 2, 2, 2};
    const int slice_starts[4] = {0, 0, 0, 1};
    const int slice_steps[4] = {1, 1, 2, 2};
    const float expected_slice[8] = {
        1.0f, 3.0f, 9.0f, 11.0f, 13.0f, 15.0f, 21.0f, 23.0f,
    };
    float slice_output[8] = {0};
    for (int index = 0; index < 24; index++)
        slice_input[index] = (float)index;

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_sigmoid_f32(
              sigmoid_input, sigmoid_output, 5) == 1);
    CHECK(metal_graph_softmax_f32(
              softmax_input, softmax_output, 2, 3) == 1);
    CHECK(metal_graph_expand_f32(
              expand_input, expand_output, expand_input_shape, 5,
              expand_output_shape, 5) == 1);
    CHECK(metal_graph_gather_i32_f32(
              gather_input, gather_indices, gather_output,
              2, 3, 2, 2, 8) == 1);
    CHECK(metal_graph_slice4d_f32(
              slice_input, slice_output, slice_input_shape, 4,
              slice_output_shape, 4, slice_starts, slice_steps) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(
              sigmoid_output, sizeof(sigmoid_output), 0) == 1);
    CHECK(metal_graph_sync_host(
              softmax_output, sizeof(softmax_output), 0) == 1);
    CHECK(metal_graph_sync_host(
              expand_output, sizeof(expand_output), 0) == 1);
    CHECK(metal_graph_sync_host(
              gather_output, sizeof(gather_output), 0) == 1);
    CHECK(metal_graph_sync_host(
              slice_output, sizeof(slice_output), 0) == 1);
    for (int index = 0; index < 5; index++) {
        float expected = 1.0f / (1.0f + expf(-sigmoid_input[index]));
        CHECK(fabsf(sigmoid_output[index] - expected) < 2.0e-4f);
    }
    for (int row = 0; row < 2; row++) {
        float denominator = 0.0f;
        for (int column = 0; column < 3; column++)
            denominator += expf(softmax_input[row * 3 + column] -
                                softmax_input[row * 3 + 2]);
        for (int column = 0; column < 3; column++) {
            float expected =
                expf(softmax_input[row * 3 + column] -
                     softmax_input[row * 3 + 2]) / denominator;
            CHECK(fabsf(softmax_output[row * 3 + column] - expected) <
                  2.0e-4f);
        }
    }
    for (int index = 0; index < 12; index++)
        CHECK(close_enough(expand_output[index], expected_expand[index]));
    for (int index = 0; index < 8; index++) {
        CHECK(close_enough(gather_output[index], expected_gather[index]));
        CHECK(close_enough(slice_output[index], expected_slice[index]));
    }

    const int incompatible_expand_shape[5] = {1, 2, 1, 3, 3};
    const int overflow_shape[2] = {INT32_MAX, 2};
    const int invalid_slice_steps[4] = {1, 1, 0, 1};
    CHECK(metal_graph_expand_f32(
              expand_input, expand_output, expand_input_shape, 5,
              incompatible_expand_shape, 5) == 0);
    CHECK(metal_graph_expand_f32(
              expand_input, expand_output, overflow_shape, 2,
              overflow_shape, 2) == 0);
    CHECK(metal_graph_gather_i32_f32(
              gather_input, gather_indices, gather_output,
              2, 3, 2, 2, 7) == 0);
    CHECK(metal_graph_slice4d_f32(
              slice_input, slice_output, slice_input_shape, 4,
              slice_output_shape, 4, slice_starts,
              invalid_slice_steps) == 0);
    CHECK(metal_graph_softmax_f32(
              softmax_input, softmax_output, INT32_MAX, 2) == 0);
    CHECK(metal_graph_sigmoid_f32(
              sigmoid_input, sigmoid_output, (long)UINT32_MAX) == 0);
    metal_graph_reset();
    return 0;
}

static int test_typed_control_graph_ops(void) {
    const int32_t compare_a[2] = {1, 4};
    const int32_t compare_b[3] = {1, 3, 4};
    const uint32_t output_strides[2] = {3u, 1u};
    const uint32_t a_strides[2] = {1u, 0u};
    const uint32_t b_strides[2] = {0u, 1u};
    const int32_t expected_equal[6] = {1, 0, 0, 0, 0, 1};
    const int32_t expected_ge[6] = {1, 0, 0, 1, 1, 1};
    const int32_t expected_not[6] = {0, 1, 1, 1, 1, 0};
    int32_t equal_output[6] = {0};
    int32_t ge_output[6] = {0};
    int32_t not_output[6] = {0};
    const int32_t clip_input[4] = {-4, 0, 3, 9};
    const int32_t expected_clip[4] = {0, 0, 3, 7};
    int32_t clip_output[4] = {0};
    const int32_t cast_i32[4] = {-2, 0, 7, 10};
    const float expected_cast_f32[4] = {-2.0f, 0.0f, 7.0f, 10.0f};
    float cast_f32[4] = {0};
    const float cast_float_input[13] = {
        -2.9f, 0.0f, 7.75f,
        2147483648.0f, 2147483904.0f,
        4294967296.0f, 4294967808.0f, 6442450944.0f,
        -2147483904.0f, -4294967808.0f,
        NAN, INFINITY, -INFINITY,
    };
    const int32_t expected_cast_i32[13] = {
        -2, 0, 7,
        INT32_MIN, INT32_MIN + 256,
        0, 512, INT32_MIN,
        INT32_MAX - 255, -512,
        0, 0, 0,
    };
    int32_t cast_i32_output[13] = {0};
    const uint32_t copy_input[4] = {
        0x7fc01234u, 0x80000000u, 0xffffffffu, 0x12345678u,
    };
    uint32_t copy_output[4] = {0};
    const int32_t where_condition[4] = {0, 1, -1, 0};
    const uint32_t where_a[4] = {
        0x7fc01234u, 0x80000000u, 0x11111111u, 0x22222222u,
    };
    const uint32_t where_b[4] = {
        0x33333333u, 0x44444444u, 0xffffffffu, 0x7fa00001u,
    };
    const uint32_t expected_where[4] = {
        0x33333333u, 0x80000000u, 0x11111111u, 0x7fa00001u,
    };
    uint32_t where_output[4] = {0};
    const float argmax_input[12] = {
        1.0f, 5.0f, 3.0f, 5.0f, 3.0f, 4.0f,
        -1.0f, -2.0f, -1.0f, 7.0f, -3.0f, 7.0f,
    };
    const int32_t expected_argmax[4] = {1, 0, 0, 1};
    int32_t argmax_output[4] = {0};
    const uint32_t concat_a[4] = {
        0x7fc00011u, 0x80000000u, 0x11111111u, 0x22222222u,
    };
    const uint32_t concat_b[2] = {0xffffffffu, 0x33333333u};
    const void* concat_inputs[2] = {concat_a, concat_b};
    const long concat_sizes[2] = {4, 2};
    const int concat_axes[2] = {2, 1};
    const uint32_t expected_concat[6] = {
        0x7fc00011u, 0x80000000u, 0x11111111u,
        0x22222222u, 0xffffffffu, 0x33333333u,
    };
    uint32_t concat_output[6] = {0};

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_compare_i32(
        compare_a, 2, compare_b, 3, equal_output, 6,
        output_strides, a_strides, b_strides, 2, 0) == 1);
    CHECK(metal_graph_compare_i32(
        compare_a, 2, compare_b, 3, ge_output, 6,
        output_strides, a_strides, b_strides, 2, 1) == 1);
    CHECK(metal_graph_not_i32(equal_output, not_output, 6) == 1);
    CHECK(metal_graph_clip_i32(
        clip_input, clip_output, 4, 0, 7) == 1);
    CHECK(metal_graph_cast_typed(
        cast_i32, VX_DTYPE_I32, cast_f32, VX_DTYPE_F32, 4) == 1);
    CHECK(metal_graph_cast_typed(
        cast_float_input, VX_DTYPE_F32,
        cast_i32_output, VX_DTYPE_I32, 13) == 1);
    CHECK(metal_graph_copy_32(copy_input, copy_output, 4) == 1);
    CHECK(metal_graph_where_32(
        where_condition, where_a, where_b, where_output, 4) == 1);
    CHECK(metal_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 3u, 2u) == 1);
    CHECK(metal_graph_concat_32(
        concat_inputs, concat_sizes, concat_axes, 2,
        concat_output, 3, 2) == 1);
    CHECK(metal_graph_end_forward() == 0);

    CHECK(metal_graph_sync_host(
        equal_output, sizeof(equal_output), 0) == 1);
    CHECK(metal_graph_sync_host(ge_output, sizeof(ge_output), 0) == 1);
    CHECK(metal_graph_sync_host(not_output, sizeof(not_output), 0) == 1);
    CHECK(metal_graph_sync_host(clip_output, sizeof(clip_output), 0) == 1);
    CHECK(metal_graph_sync_host(cast_f32, sizeof(cast_f32), 0) == 1);
    CHECK(metal_graph_sync_host(
        cast_i32_output, sizeof(cast_i32_output), 0) == 1);
    CHECK(metal_graph_sync_host(copy_output, sizeof(copy_output), 0) == 1);
    CHECK(metal_graph_sync_host(where_output, sizeof(where_output), 0) == 1);
    CHECK(metal_graph_sync_host(
        argmax_output, sizeof(argmax_output), 0) == 1);
    CHECK(metal_graph_sync_host(
        concat_output, sizeof(concat_output), 0) == 1);
    CHECK(memcmp(equal_output, expected_equal, sizeof(equal_output)) == 0);
    CHECK(memcmp(ge_output, expected_ge, sizeof(ge_output)) == 0);
    CHECK(memcmp(not_output, expected_not, sizeof(not_output)) == 0);
    CHECK(memcmp(clip_output, expected_clip, sizeof(clip_output)) == 0);
    CHECK(memcmp(cast_f32, expected_cast_f32, sizeof(cast_f32)) == 0);
    CHECK(memcmp(
        cast_i32_output, expected_cast_i32,
        sizeof(cast_i32_output)) == 0);
    CHECK(memcmp(copy_output, copy_input, sizeof(copy_output)) == 0);
    CHECK(memcmp(where_output, expected_where, sizeof(where_output)) == 0);
    CHECK(memcmp(
        argmax_output, expected_argmax, sizeof(argmax_output)) == 0);
    CHECK(memcmp(
        concat_output, expected_concat, sizeof(concat_output)) == 0);

    CHECK(metal_graph_not_i32(equal_output, equal_output, 6) == 0);
    CHECK(metal_graph_clip_i32(
        clip_input, clip_output, 4, 8, 7) == 0);
    CHECK(metal_graph_cast_typed(
        cast_i32, VX_DTYPE_I32,
        cast_i32_output, VX_DTYPE_U8, 4) == 0);
    CHECK(metal_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 0u, 2u) == 0);
    metal_graph_reset();
    return 0;
}

static int test_qbatch_and_typed_transpose(void) {
    const int a_shape[4] = {2, 1, 2, 2};
    const int b_shape[3] = {3, 2, 2};
    const int output_shape[4] = {2, 3, 2, 2};
    const int invalid_output_shape[4] = {2, 2, 2, 2};
    const uint8_t a[8] = {
        129, 130, 131, 132,
        127, 128, 130, 126,
    };
    const int8_t b[12] = {
        0, -1, -1, 0,
        1, 0, -2, 1,
        -1, -2, 2, 0,
    };
    const int8_t expected_qbatch[24] = {
        4, 4, 4, 5,
        3, 6, 4, 8,
        6, 4, 9, 4,
        2, 3, 4, 2,
        2, 2, 6, 2,
        3, 4, 0, 1,
    };
    int8_t qbatch_output[24] = {0};
    const uint32_t transpose_shape[4] = {1u, 2u, 2u, 3u};
    const uint32_t transpose_permutation[4] = {0u, 2u, 3u, 1u};
    const uint32_t transpose_duplicate[4] = {0u, 2u, 2u, 1u};
    const uint8_t transpose_u8_input[12] = {
        117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127, 128,
    };
    const uint8_t transpose_u8_expected[12] = {
        117, 123, 118, 124, 119, 125,
        120, 126, 121, 127, 122, 128,
    };
    const int8_t transpose_i8_input[12] = {
        -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5,
    };
    const int8_t transpose_i8_expected[12] = {
        -6, 0, -5, 1, -4, 2,
        -3, 3, -2, 4, -1, 5,
    };
    uint8_t transpose_u8_output[12] = {0};
    int8_t transpose_i8_output[12] = {0};

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qbatch_matmul_i8u8(
        a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
        b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
        qbatch_output, invalid_output_shape, 4,
        0.25f, 3, VX_DTYPE_I8) == 0);
    CHECK(metal_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_duplicate, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(metal_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.25f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(metal_graph_transpose_i8u8(
        transpose_u8_output, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(metal_graph_qbatch_matmul_i8u8(
        a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
        b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
        qbatch_output, output_shape, 4,
        0.25f, 3, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_transpose_i8u8(
        transpose_i8_input, transpose_i8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.25f, -3,
        0.25f, -3, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_end_forward() == 0);

    CHECK(metal_graph_sync_host(
        qbatch_output, sizeof(qbatch_output), 0) == 1);
    CHECK(metal_graph_sync_host(
        transpose_u8_output, sizeof(transpose_u8_output), 0) == 1);
    CHECK(metal_graph_sync_host(
        transpose_i8_output, sizeof(transpose_i8_output), 0) == 1);
    CHECK(memcmp(
        qbatch_output, expected_qbatch, sizeof(qbatch_output)) == 0);
    CHECK(memcmp(
        transpose_u8_output, transpose_u8_expected,
        sizeof(transpose_u8_output)) == 0);
    CHECK(memcmp(
        transpose_i8_output, transpose_i8_expected,
        sizeof(transpose_i8_output)) == 0);
    metal_graph_reset();
    return 0;
}

#ifdef VOLVOX_METAL_TESTING
static int test_graph_forward_command_batch(void) {
    float input[4] = {1.0f, -2.0f, 3.5f, 4.0f};
    float intermediate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t dispatches = 0, commits = 0, waits = 0;

    metal_graph_reset();
    metal_graph_debug_reset_counters();
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(input, intermediate, 4) == 1);
    CHECK(metal_graph_copy_f32(intermediate, output, 4) == 1);
    metal_graph_debug_counters(&dispatches, &commits, &waits);
    CHECK(dispatches == 2);
    CHECK(commits == 0);
    CHECK(waits == 0);
    CHECK(metal_graph_end_forward() == 0);
    metal_graph_debug_counters(&dispatches, &commits, &waits);
    CHECK(dispatches == 2);
    CHECK(commits == 1);
    CHECK(waits == 1);
    CHECK(metal_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, input, sizeof(output)) == 0);

    /* Preserve the direct backend-call contract outside a graph forward. */
    memset(intermediate, 0, sizeof(intermediate));
    CHECK(metal_graph_copy_f32(input, intermediate, 4) == 1);
    metal_graph_debug_counters(&dispatches, &commits, &waits);
    CHECK(dispatches == 3);
    CHECK(commits == 2);
    CHECK(waits == 2);
    CHECK(metal_graph_sync_host(intermediate, sizeof(intermediate), 0) == 1);
    CHECK(memcmp(intermediate, input, sizeof(intermediate)) == 0);

    /* A CPU overwrite of a buffer referenced by queued GPU work is a safe
       segment boundary, then batching resumes for the rest of the forward. */
    const float host_override[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    metal_graph_reset();
    metal_graph_debug_reset_counters();
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(input, intermediate, 4) == 1);
    memcpy(intermediate, host_override, sizeof(intermediate));
    metal_graph_mark_host(intermediate, sizeof(intermediate), 0);
    metal_graph_debug_counters(&dispatches, &commits, &waits);
    CHECK(dispatches == 1);
    CHECK(commits == 1);
    CHECK(waits == 1);
    CHECK(metal_graph_copy_f32(intermediate, output, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    metal_graph_debug_counters(&dispatches, &commits, &waits);
    CHECK(dispatches == 2);
    CHECK(commits == 2);
    CHECK(waits == 2);
    CHECK(metal_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, host_override, sizeof(output)) == 0);
    metal_graph_reset();
    return 0;
}
#endif

static int test_tiled_qlinear_i8u8_tails(void) {
    enum { ROWS = 9, D_IN = 19, D_OUT = 36 };
    int8_t input[ROWS * D_IN];
    int8_t weight[D_OUT * D_IN];
    float scales[D_OUT];
    int32_t zero_points[D_OUT];
    int32_t bias[D_OUT];
    int8_t output[ROWS * D_OUT];
    int8_t expected[ROWS * D_OUT];
    for (int i = 0; i < ROWS * D_IN; i++) input[i] = (int8_t)((i * 5) % 7 - 3);
    for (int i = 0; i < D_OUT * D_IN; i++) weight[i] = (int8_t)((i * 3) % 5 - 2);
    for (int column = 0; column < D_OUT; column++) {
        scales[column] = 1.0f;
        zero_points[column] = 0;
        bias[column] = column % 5 - 2;
    }
    for (int row = 0; row < ROWS; row++) for (int column = 0; column < D_OUT; column++) {
        int accumulator = bias[column];
        for (int k = 0; k < D_IN; k++) {
            accumulator += input[row * D_IN + k] * weight[column * D_IN + k];
        }
        if (accumulator < -128) accumulator = -128;
        if (accumulator > 127) accumulator = 127;
        expected[row * D_OUT + column] = (int8_t)accumulator;
    }
    memset(output, 0, sizeof(output));
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qlinear_i8u8(input, weight, scales, zero_points, bias, output,
                                   ROWS, D_IN, D_OUT, 1.0f, 0, 1.0f, 0,
                                   VX_DTYPE_I8, VX_DTYPE_I8,
                                   VX_DTYPE_I8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    metal_graph_reset();
    return 0;
}

static float gelu_reference(float x, int approximate_tanh) {
    return approximate_tanh
        ? 0.5f * x * (1.0f + tanhf(0.7978845608028654f *
            (x + 0.044715f * x * x * x)))
        : 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static float gelu_derivative_reference(float x, int approximate_tanh) {
    if (!approximate_tanh) {
        return 0.5f * (1.0f + erff(x * 0.7071067811865475f)) +
            x * 0.3989422804014327f * expf(-0.5f * x * x);
    }
    float x2 = x * x;
    float u = 0.7978845608028654f * (x + 0.044715f * x * x2);
    float t = tanhf(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) *
        0.7978845608028654f * (1.0f + 0.134145f * x2);
}

static int test_gelu_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7] = {0}, tanh_output[7] = {0};
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_gelu_f32(input, exact_output, 7, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(exact_output, sizeof(exact_output), 0) == 1);
    metal_graph_begin_forward();
    CHECK(metal_graph_gelu_f32(input, tanh_output, 7, 1) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(tanh_output, sizeof(tanh_output), 0) == 1);
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_output[i] - gelu_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_output[i] - gelu_reference(input[i], 1)) < 2.0e-4f);
    }
    CHECK(fabsf(exact_output[0] - tanh_output[0]) > 1.0e-5f);

    float grad_output[7], exact_grad[7] = {0}, tanh_grad[7] = {0};
    for (int i = 0; i < 7; i++) grad_output[i] = 0.25f * (float)(i + 1);
    uint32_t exact_params[4] = {7u, 1u, 0u, 0u};
    uint32_t tanh_params[4] = {7u, 9u, 0u, 0u};
    void* exact_hosts[5] = {input, exact_output, grad_output, exact_grad, exact_params};
    void* tanh_hosts[5] = {input, tanh_output, grad_output, tanh_grad, tanh_params};
    size_t exact_bytes[5] = {sizeof(input), sizeof(exact_output), sizeof(grad_output),
                             sizeof(exact_grad), sizeof(exact_params)};
    size_t tanh_bytes[5] = {sizeof(input), sizeof(tanh_output), sizeof(grad_output),
                            sizeof(tanh_grad), sizeof(tanh_params)};
    unsigned char access[5] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE, METAL_TRAINING_READ
    };
    CHECK(metal_training_begin() == 0);
    CHECK(metal_training_dispatch("activationBackward", "main", exact_hosts, exact_bytes,
                                  access, NULL, 5, 1, 1, 1) == 0);
    CHECK(metal_training_dispatch("activationBackward", "main", tanh_hosts, tanh_bytes,
                                  access, NULL, 5, 1, 1, 1) == 0);
    CHECK(metal_training_sync(exact_grad, sizeof(exact_grad)) == 0);
    CHECK(metal_training_sync(tanh_grad, sizeof(tanh_grad)) == 0);
    metal_training_end();
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 1)) < 2.0e-4f);
    }
    metal_graph_reset();
    return 0;
}

static int test_qembedding_i8u8_packed_gather(void) {
    const int32_t ids[3] = {1, 0, 2};
    const int32_t invalid_ids[3] = {1, 3, 0};
    const float scales[3] = {0.5f, 0.25f, 1.0f};
    const int32_t i8_zero_points[3] = {0, 1, -1};
    const int32_t u8_zero_points[3] = {128, 129, 127};
    const int8_t i8_table[9] = {0, 1, -1, 2, -2, 4, 99, -101, 4};
    const uint8_t u8_table[9] = {128, 129, 127, 130, 126, 132, 227, 27, 132};
    const uint8_t expected_u8[9] = {129, 125, 131, 128, 130, 126, 255, 0, 148};
    const int8_t expected_i8[9] = {-2, -6, 0, -3, -1, -5, 127, -128, 17};
    uint8_t output_u8[9] = {0};
    int8_t output_i8[9] = {0};

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qembedding_i8u8(ids, i8_table, scales, i8_zero_points, output_u8,
                                      3u, 3u, 3u, 0.25f, 128,
                                      VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == expected_u8[index]);

    metal_graph_begin_forward();
    CHECK(metal_graph_qembedding_i8u8(ids, u8_table, scales, u8_zero_points, output_i8,
                                      3u, 3u, 3u, 0.25f, -3,
                                      VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_i8[index] == expected_i8[index]);

    memset(output_u8, 0x5a, sizeof(output_u8));
    CHECK(metal_graph_qembedding_i8u8(invalid_ids, i8_table, scales, i8_zero_points,
                                      output_u8, 3u, 3u, 3u, 0.25f, 128,
                                      VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == 0x5a);
    metal_graph_reset();
    return 0;
}

static int test_qsilu_i8u8_packed_chain(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    const int8_t expected_i8[5] = {-3, -4, -3, 0, 13};
    const uint8_t expected_u8[5] = {128, 127, 128, 131, 144};
    int8_t from_i8_i8[5] = {0};
    int8_t from_u8_i8[5] = {0};
    uint8_t from_i8_u8[5] = {0};
    uint8_t from_u8_u8[5] = {0};
    int8_t chained[5] = {0};
    int8_t alias[5] = {-8, -2, 0, 2, 8};

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qsilu_i8u8(input_i8, from_i8_i8, 5u,
                                  0.5f, 0, 0.25f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qsilu_i8u8(input_i8, from_i8_u8, 5u,
                                  0.5f, 0, 0.25f, 128,
                                  VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qsilu_i8u8(input_u8, from_u8_i8, 5u,
                                  0.5f, 128, 0.25f, -3,
                                  VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qsilu_i8u8(input_u8, from_u8_u8, 5u,
                                  0.5f, 128, 0.25f, 128,
                                  VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qsilu_i8u8(from_i8_u8, chained, 5u,
                                  0.25f, 128, 0.125f, -4,
                                  VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -5, -4, 0, 27};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(metal_graph_qsilu_i8u8(input_i8, from_i8_i8, 0u,
                                  0.5f, 0, 0.25f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(metal_graph_qsilu_i8u8(alias, alias, 5u,
                                  0.5f, 0, 0.25f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    metal_graph_reset();
    return 0;
}

/* Mirror the direct packed-byte QGELU coverage used by the other native GPU
 * backends.  Runtime execution is exercised on Apple hosts; this source also
 * guards the shared kernel ABI during generated-MSL validation elsewhere. */
static int test_qgelu_i8u8_packed_chain(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    const int8_t expected_i8[5] = {-3, -4, -3, 4, 29};
    const uint8_t expected_u8[5] = {128, 127, 128, 135, 160};
    int8_t from_i8_i8[5] = {0};
    int8_t from_u8_i8[5] = {0};
    uint8_t from_i8_u8[5] = {0};
    uint8_t from_u8_u8[5] = {0};
    int8_t chained[5] = {0};
    int8_t alias[5] = {-8, -2, 0, 2, 8};

    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qgelu_i8u8(input_i8, from_i8_i8, 5u,
                                  0.5f, 0, 0.125f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgelu_i8u8(input_i8, from_i8_u8, 5u,
                                  0.5f, 0, 0.125f, 128,
                                  VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qgelu_i8u8(input_u8, from_u8_i8, 5u,
                                  0.5f, 128, 0.125f, -3,
                                  VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgelu_i8u8(input_u8, from_u8_u8, 5u,
                                  0.5f, 128, 0.125f, 128,
                                  VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qgelu_i8u8(from_i8_u8, chained, 5u,
                                  0.125f, 128, 0.125f, -4,
                                  VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -4, -4, 2, 28};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(metal_graph_qgelu_i8u8(input_i8, from_i8_i8, 0u,
                                  0.5f, 0, 0.125f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(metal_graph_qgelu_i8u8(alias, alias, 5u,
                                  0.5f, 0, 0.125f, -3,
                                  VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    metal_graph_reset();
    return 0;
}

/* Mirror the Vulkan/OpenGL byte-domain conformance vectors on Apple hosts.
 * These avoid reduction half-LSB boundaries while still covering packed C=3
 * tails, C/G=3 group boundaries, and high unsigned input values. */
static int test_qgroupnorm_i8u8_packed_chain(void) {
    const int8_t input_i8[3] = {-5, 0, 4};
    const uint8_t input_u8[3] = {123, 128, 132};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
    const int8_t expected_i8[3] = {-11, -7, -4};
    const uint8_t expected_u8[3] = {120, 124, 127};
    int8_t from_i8_i8[3] = {0};
    int8_t from_u8_i8[3] = {0};
    uint8_t from_i8_u8[3] = {0};
    uint8_t from_u8_u8[3] = {0};
    int8_t chained[3] = {0};
    int8_t alias[3] = {-5, 0, 4};
    const int8_t boundary_input[18] = {
        -8, -3, 4, 7, 5, -1, 2, -6, 0, 8, -4, 3, -7, 6, 1, -2, 9, -5,
    };
    const int8_t boundary_expected[18] = {
        -10, -6, 8, 5, -1, -12, 7, 1, 5, 7, 6, -9, -11, -14, 4, -9, -5, -11,
    };
    const float boundary_gamma[6] = {1.0f, -0.75f, 0.5f, 1.25f, -0.5f, 0.25f};
    const float boundary_beta[6] = {0.25f, -0.5f, 0.75f, -0.25f, 0.5f, -0.75f};
    int8_t boundary_output[18] = {0};
    enum { high_elements = 3 * 57 * 6 };
    uint8_t high_input[high_elements];
    uint8_t high_output[high_elements] = {0};
    const float high_gamma[6] = {1, 1, 1, 1, 1, 1};
    const float high_beta[6] = {0, 0, 0, 0, 0, 0};

    for (int index = 0; index < high_elements; index++)
        high_input[index] = (uint8_t)(index & 1 ? 241 : 240);
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                       1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                       0.125f, -3, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                       1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                       0.125f, 128, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                       1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                       0.125f, -3, 1.0e-5f,
                                       VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                       1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                       0.125f, 128, 1.0e-5f,
                                       VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                       1u, 1u, 1u, 3u, 1u, 0.125f, 128,
                                       0.125f, -4, 1.0e-5f,
                                       VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(boundary_input, boundary_gamma, boundary_beta,
                                       boundary_output, 3u, 1u, 1u, 6u, 2u,
                                       0.25f, -1, 0.125f, -3, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qgroupnorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                       3u, 57u, 1u, 6u, 2u, 0.5f, 17,
                                       0.25f, 128, 1.0e-5f,
                                       VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(metal_graph_sync_host(boundary_output, sizeof(boundary_output), 0) == 1);
    CHECK(metal_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[3] = {-2, -8, 2};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(memcmp(boundary_output, boundary_expected, sizeof(boundary_expected)) == 0);
    for (int index = 0; index < high_elements; index += 6) {
        const uint8_t expected[6] = {125, 134, 125, 131, 122, 131};
        CHECK(memcmp(high_output + index, expected, sizeof(expected)) == 0);
    }
    CHECK(metal_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                       1u, 1u, 1u, 3u, 0u, 0.5f, -1,
                                       0.125f, -3, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(metal_graph_qgroupnorm_i8u8(alias, gamma, beta, alias,
                                       1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                       0.125f, -3, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    metal_graph_reset();
    return 0;
}

/* Apple direct coverage mirrors the Vulkan/OpenGL final-axis byte contract:
 * all typed boundaries, D=3 packed-row crossing, a resident chain, and high
 * U8 centered variance. */
static int test_qlayernorm_i8u8_packed_chain(void) {
    const int8_t input_i8[6] = {-5, 0, 4, 6, -2, 1};
    const uint8_t input_u8[6] = {123, 128, 132, 134, 126, 129};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
    const int8_t expected_i8[6] = {-11, -7, -4, 10, -11, 4};
    const uint8_t expected_u8[6] = {120, 124, 127, 141, 120, 135};
    int8_t from_i8_i8[6] = {0};
    int8_t from_u8_i8[6] = {0};
    uint8_t from_i8_u8[6] = {0};
    uint8_t from_u8_u8[6] = {0};
    int8_t chained[6] = {0};
    int8_t alias[6] = {-5, 0, 4, 6, -2, 1};
    enum { high_d_model = 1024, high_elements = 2 * high_d_model };
    uint8_t high_input[high_elements];
    uint8_t high_output[high_elements] = {0};
    float high_gamma[high_d_model];
    float high_beta[high_d_model];

    for (int index = 0; index < high_elements; index++)
        high_input[index] = (uint8_t)(index & 1 ? 241 : 240);
    for (int channel = 0; channel < high_d_model; channel++) {
        high_gamma[channel] = 1.0f;
        high_beta[channel] = 0.0f;
    }
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                       2u, 3u, 0.5f, -1, 0.125f, -3,
                                       1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                       2u, 3u, 0.5f, -1, 0.125f, 128,
                                       1.0e-5f, VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                       2u, 3u, 0.5f, 127, 0.125f, -3,
                                       1.0e-5f, VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                       2u, 3u, 0.5f, 127, 0.125f, 128,
                                       1.0e-5f, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_qlayernorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                       2u, 3u, 0.125f, 128, 0.125f, -4,
                                       1.0e-5f, VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qlayernorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                       2u, high_d_model, 0.5f, 17, 0.25f, 128,
                                       1.0e-5f, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(metal_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(metal_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(metal_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[6] = {-2, -8, 2, -2, -8, 2};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    for (int index = 0; index < high_elements; index++)
        CHECK(high_output[index] == (uint8_t)(index & 1 ? 132 : 124));
    CHECK(metal_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                       2u, 0u, 0.5f, -1, 0.125f, -3,
                                       1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(metal_graph_qlayernorm_i8u8(alias, gamma, beta, alias,
                                       2u, 3u, 0.5f, -1, 0.125f, -3,
                                       1.0e-5f, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    metal_graph_reset();
    return 0;
}

/* Apple coverage mirrors the Vulkan/OpenGL byte-domain attention contract:
 * one workgroup per query/head/batch, I8->U8->I8 device chaining, a valid
 * ordinary/no-mask dummy binding, all-masked zero-point output, and D=64. */
static int test_qsdpa_i8u8_packed_chain(void) {
    const int8_t q_i8[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    const int8_t k_i8[8] = {0, 0, -1, -1, -1, 0, 0, -2};
    const int8_t v_i8[8] = {3, -3, 0, -1, -5, 1, 2, -2};
    const uint8_t q_u8[4] = {129, 126, 131, 128};
    const uint8_t k_u8[8] = {121, 120, 119, 122, 119, 123, 120, 118};
    const uint8_t v_u8[8] = {134, 126, 132, 130, 128, 132, 136, 128};
    const int32_t mask_none[2] = {0, 0};
    const uint8_t expected_first[8] = {128, 128, 130, 128, 128, 128, 130, 127};
    const int8_t expected_second[8] = {0, 0, 2, -1, 0, 0, 2, -1};
    const uint8_t expected_zero[4] = {127, 127, 127, 127};
    uint8_t first[8] = {0};
    int8_t second[8] = {0};
    uint8_t all_masked[4] = {0};
    int8_t alias[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    int8_t alias_before[8];
    enum { head_dim = 64 };
    int8_t q64[head_dim] = {0};
    int8_t k64[head_dim] = {0};
    int8_t v64[head_dim];
    int8_t out64[head_dim] = {0};

    for (int channel = 0; channel < head_dim; channel++)
        v64[channel] = (int8_t)((channel % 15) - 7);
    memcpy(alias_before, alias, sizeof(alias));
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, first, 1u, 2u, 2u,
                                  4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                                  0.25f, 128, 0.5f,
                                  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                  VX_DTYPE_U8, 0u, 0u) == 1);
    CHECK(metal_graph_qsdpa_i8u8(first, k_i8, v_i8, NULL, second, 1u, 2u, 2u,
                                  4u, 1u, 0.25f, 128, 0.25f, -1, 0.25f, -1,
                                  0.25f, 0, 0.5f,
                                  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8,
                                  VX_DTYPE_I8, 0u, 0u) == 1);
    CHECK(metal_graph_qsdpa_i8u8(q_u8, k_u8, v_u8, mask_none, all_masked,
                                  1u, 1u, 2u, 4u, 1u, 0.25f, 128, 0.5f, 120,
                                  0.25f, 130, 0.25f, 127, 0.5f,
                                  VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                                  VX_DTYPE_U8, 0u, 1u) == 1);
    CHECK(metal_graph_qsdpa_i8u8(q64, k64, v64, NULL, out64, 1u, 1u, 1u,
                                  head_dim, 1u, 0.25f, 0, 0.25f, 0, 0.25f, 0,
                                  0.25f, 0, 1.0f,
                                  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                  VX_DTYPE_I8, 0u, 0u) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(first, sizeof(first), 0) == 1);
    CHECK(metal_graph_sync_host(second, sizeof(second), 0) == 1);
    CHECK(metal_graph_sync_host(all_masked, sizeof(all_masked), 0) == 1);
    CHECK(metal_graph_sync_host(out64, sizeof(out64), 0) == 1);
    CHECK(memcmp(first, expected_first, sizeof(first)) == 0);
    CHECK(memcmp(second, expected_second, sizeof(second)) == 0);
    CHECK(memcmp(all_masked, expected_zero, sizeof(all_masked)) == 0);
    CHECK(memcmp(out64, v64, sizeof(out64)) == 0);
    CHECK(metal_graph_qsdpa_i8u8(alias, k_i8, v_i8, NULL, alias, 1u, 2u, 2u,
                                  4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                                  0.25f, 0, 0.5f,
                                  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                  VX_DTYPE_I8, 0u, 0u) == 0);
    CHECK(memcmp(alias, alias_before, sizeof(alias)) == 0);
    metal_graph_reset();
    return 0;
}

/* qArgMaxInt8 reads packed byte storage directly and emits conventional I32
 * indices. This includes the non-word-aligned I8 input tail, signed negative
 * ordering, U8 order with an asymmetric zero point, and first-tie behavior. */
static int test_qargmax_i8u8_raw(void) {
    const int8_t input_i8[12] = {
        -5, 4, -1, 4, -1, 3,
        -128, 0, -127, 1, -126, 1,
    };
    const int32_t expected_i8[4] = {1, 0, 2, 1};
    const uint8_t input_u8[6] = {130, 3, 129, 255, 130, 254};
    const int32_t expected_u8[2] = {0, 1};
    int32_t output_i8[4] = {0};
    int32_t output_u8[2] = {0};
    int32_t sentinel[2] = {17, 23};
    const int32_t sentinel_before[2] = {17, 23};
    union {
        int32_t i32[4];
        uint8_t bytes[16];
    } alias;
    uint8_t alias_before[sizeof(alias.bytes)];

    memset(alias.bytes, 0x5a, sizeof(alias.bytes));
    memcpy(alias.bytes, input_i8, 6u);
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u,
                                   VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u,
                                   VX_DTYPE_U8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(metal_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(metal_graph_qargmax_i8u8(alias.bytes, alias.i32, 1u, 3u, 2u,
                                   VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(metal_graph_qargmax_i8u8(input_i8, sentinel, 1u, 0u, 2u,
                                   VX_DTYPE_I8) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    metal_graph_reset();
    return 0;
}

static int test_qmaskedmean_i8u8_packed(void) {
    const int8_t input_i8[24] = {
        -3, -1,  1,  1,
         1, -5, -3, -1,
        99, 98, 97, 96,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
    };
    const int32_t mask_i8[6] = {1, 1, 0, 0, 0, 0};
    const int8_t expected_i8[8] = {6, 5, 6, 6, 5, 5, 5, 5};
    const uint8_t input_u8[6] = {130, 134, 20, 20, 131, 129};
    const int32_t mask_u8[3] = {1, 0, 1};
    const uint8_t expected_u8[2] = {132, 134};
    int8_t output_i8[8] = {0};
    uint8_t output_u8[2] = {0};
    union {
        int8_t input[24];
        uint8_t bytes[24];
    } alias;
    union {
        int32_t mask[6];
        uint8_t bytes[24];
    } mask_alias;
    uint8_t alias_before[sizeof(alias.bytes)];
    uint8_t mask_before[sizeof(mask_alias.bytes)];

    memcpy(alias.input, input_i8, sizeof(input_i8));
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    memcpy(mask_alias.mask, mask_i8, sizeof(mask_i8));
    memcpy(mask_before, mask_alias.bytes, sizeof(mask_before));
    metal_graph_reset();
    metal_graph_begin_forward();
    CHECK(metal_graph_qmaskedmean_i8u8(input_i8, mask_i8, output_i8,
                                        2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(metal_graph_qmaskedmean_i8u8(input_u8, mask_u8, output_u8,
                                        1u, 3u, 2u, 0.25f, 128, 0.25f, 130,
                                        VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(metal_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(metal_graph_qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes,
                                        2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(metal_graph_qmaskedmean_i8u8(input_i8, mask_alias.mask,
                                        mask_alias.bytes, 2u, 3u, 4u,
                                        0.25f, -3, 0.5f, 5,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_alias.bytes)) == 0);
    metal_graph_reset();
    return 0;
}

static int test_activation_backward(void) {
    float input[2] = {-1.0f, 2.0f};
    float output[2] = {0.0f, 2.0f};
    float grad_output[2] = {3.0f, 4.0f};
    float grad_input[2] = {0.0f, 0.0f};
    struct {
        uint32_t length;
        uint32_t kind;
        float alpha;
        float beta;
    } params = {2u, 0u, 0.0f, 0.0f};
    void* hosts[5] = {input, output, grad_output, grad_input, &params};
    size_t bytes[5] = {sizeof(input), sizeof(output), sizeof(grad_output),
                       sizeof(grad_input), sizeof(params)};
    unsigned char access[5] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE, METAL_TRAINING_READ
    };

    CHECK(metal_training_begin() == 0);
    CHECK(metal_training_dispatch("activationBackward", "main", hosts, bytes,
                                  access, NULL, 5, 1, 1, 1) == 0);
    CHECK(metal_training_sync(grad_input, sizeof(grad_input)) == 0);
    metal_training_end();
    CHECK(close_enough(grad_input[0], 0.0f));
    CHECK(close_enough(grad_input[1], 4.0f));
    return 0;
}

static int test_multi_entry_matmul_backward(void) {
    float input[2] = {1.0f, 2.0f};
    float weight[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    float grad_output[2] = {7.0f, 8.0f};
    float grad_input[2] = {0.0f, 0.0f};
    float grad_weight[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_bias[2] = {0.0f, 0.0f};
    uint32_t params[8] = {1u, 2u, 2u, 1u, 0u, 0u, 0u, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE,
        METAL_TRAINING_READ
    };
    unsigned char weights[7] = {0, 1, 0, 0, 0, 0, 0};

    CHECK(metal_training_begin() == 0);
    CHECK(metal_training_dispatch("matMulBackward", "input_main", hosts, bytes,
                                  access, weights, 7, 1, 1, 1) == 0);
    CHECK(metal_training_dispatch("matMulBackward", "weight_main", hosts, bytes,
                                  access, weights, 7, 1, 1, 1) == 0);
    CHECK(metal_training_dispatch("matMulBackward", "bias_main", hosts, bytes,
                                  access, weights, 7, 1, 1, 1) == 0);
    CHECK(metal_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(metal_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(metal_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    metal_training_end();
    CHECK(close_enough(grad_input[0], 61.0f));
    CHECK(close_enough(grad_input[1], 76.0f));
    CHECK(close_enough(grad_weight[0], 7.0f));
    CHECK(close_enough(grad_weight[1], 14.0f));
    CHECK(close_enough(grad_weight[2], 8.0f));
    CHECK(close_enough(grad_weight[3], 16.0f));
    CHECK(close_enough(grad_bias[0], 7.0f));
    CHECK(close_enough(grad_bias[1], 8.0f));
    return 0;
}

static int test_prelu_logsoftmax_split_backward(void) {
    float input[4] = {-1.0f, 2.0f, -0.5f, 1.5f};
    float weight[2] = {0.2f, 0.4f};
    float grad_output[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad_input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_weight[2] = {0.0f, 0.0f};
    uint32_t prelu_params[4] = {4u, 2u, 2u, 0u};
    void* prelu_hosts[6] = {input, weight, grad_output, grad_input, grad_weight,
                            prelu_params};
    size_t prelu_bytes[6] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                             sizeof(grad_input), sizeof(grad_weight),
                             sizeof(prelu_params)};
    unsigned char prelu_access[6] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE, METAL_TRAINING_READ
    };
    unsigned char prelu_weights[6] = {0, 1, 0, 0, 0, 0};

    CHECK(metal_training_begin() == 0);
    CHECK(metal_training_dispatch("preluBackward", "input_main", prelu_hosts,
                                  prelu_bytes, prelu_access, prelu_weights,
                                  6, 1, 1, 1) == 0);
    CHECK(metal_training_dispatch("preluBackward", "weight_main", prelu_hosts,
                                  prelu_bytes, prelu_access, prelu_weights,
                                  6, 1, 1, 1) == 0);
    CHECK(metal_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(metal_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(close_enough(grad_input[0], 0.2f));
    CHECK(close_enough(grad_input[1], 2.0f));
    CHECK(close_enough(grad_input[2], 0.6f));
    CHECK(close_enough(grad_input[3], 4.0f));
    CHECK(close_enough(grad_weight[0], -2.5f));
    CHECK(close_enough(grad_weight[1], 0.0f));

    float log_output[2] = {-1.38629436112f, -0.28768207245f};
    float log_grad_output[2] = {2.0f, -1.0f};
    float log_grad_input[2] = {0.0f, 0.0f};
    uint32_t softmax_params[4] = {1u, 2u, 1u, 0u};
    void* softmax_hosts[4] = {log_output, log_grad_output, log_grad_input,
                              softmax_params};
    size_t softmax_bytes[4] = {sizeof(log_output), sizeof(log_grad_output),
                               sizeof(log_grad_input), sizeof(softmax_params)};
    unsigned char softmax_access[4] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE, METAL_TRAINING_READ
    };
    CHECK(metal_training_dispatch("softmaxBackward", "main", softmax_hosts,
                                  softmax_bytes, softmax_access, NULL,
                                  4, 1, 1, 1) == 0);
    CHECK(metal_training_sync(log_grad_input, sizeof(log_grad_input)) == 0);
    CHECK(close_enough(log_grad_input[0], 1.75f));
    CHECK(close_enough(log_grad_input[1], -1.75f));

    float left_grad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float right_grad[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float split_grad[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t left_params[5] = {4u, 1u, 2u, 4u, 0u};
    uint32_t right_params[5] = {4u, 1u, 2u, 4u, 2u};
    void* split_hosts[3] = {left_grad, split_grad, left_params};
    size_t split_bytes[3] = {sizeof(left_grad), sizeof(split_grad),
                             sizeof(left_params)};
    unsigned char split_access[3] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ | METAL_TRAINING_WRITE,
        METAL_TRAINING_READ
    };
    CHECK(metal_training_dispatch("splitBackward", "main", split_hosts,
                                  split_bytes, split_access, NULL,
                                  3, 1, 1, 1) == 0);
    split_hosts[0] = right_grad;
    split_hosts[2] = right_params;
    CHECK(metal_training_dispatch("splitBackward", "main", split_hosts,
                                  split_bytes, split_access, NULL,
                                  3, 1, 1, 1) == 0);
    CHECK(metal_training_sync(split_grad, sizeof(split_grad)) == 0);
    metal_training_end();
    const float expected_split[8] = {1.0f, 2.0f, 5.0f, 6.0f,
                                     3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; i++) {
        CHECK(close_enough(split_grad[i], expected_split[i]));
    }
    return 0;
}

static int test_batched_attention_forward(void) {
    /* seq=1 makes attention output exactly V, isolating batch offsets. */
    float qkv[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f};
    float sdpa_output[2] = {0.0f, 0.0f};
    metal_graph_begin_forward();
    CHECK(metal_graph_sdpa_f32(qkv, NULL, 0, sdpa_output,
                               1, 1, 1, 1, 2, 1.0f, 1, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(sdpa_output, sizeof(sdpa_output), 0) == 1);
    CHECK(close_enough(sdpa_output[0], 3.0f));
    CHECK(close_enough(sdpa_output[1], 7.0f));

    float q[2] = {1.0f, 4.0f};
    float k[2] = {2.0f, 5.0f};
    float v[2] = {11.0f, 13.0f};
    float cross_output[2] = {0.0f, 0.0f};
    metal_graph_begin_forward();
    CHECK(metal_graph_cross_sdpa_f32(q, k, v, NULL, 0, cross_output,
                                     1, 1, 1, 1, 1, 2, 1.0f, 0, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(cross_output, sizeof(cross_output), 0) == 1);
    CHECK(close_enough(cross_output[0], 11.0f));
    CHECK(close_enough(cross_output[1], 13.0f));

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t key_mask[2] = {1, 0};
    float masked_output[2] = {0.0f, 0.0f};
    metal_graph_begin_forward();
    CHECK(metal_graph_sdpa_f32(masked_qkv, key_mask, 2, masked_output,
                               2, 1, 1, 1, 1, 1.0f, 0, 1) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(masked_output, sizeof(masked_output), 0) == 1);
    CHECK(close_enough(masked_output[0], 2.0f));
    CHECK(close_enough(masked_output[1], 2.0f));

    const float parity_qkv[12] = {
        0.4f, -0.2f, 0.1f, 0.5f, 0.7f, -0.3f,
        -0.1f, 0.6f, 0.3f, -0.4f, 0.2f, 0.8f
    };
    const float parity_q[4] = {0.4f, -0.2f, -0.1f, 0.6f};
    const float parity_k[4] = {0.1f, 0.5f, 0.3f, -0.4f};
    const float parity_v[4] = {0.7f, -0.3f, 0.2f, 0.8f};
    const float expected_inference[4] = {
        0.4270835221f, 0.3004162014f, 0.4988606870f, 0.1425064802f
    };
    const float expected_training[4] = {
        0.6358339190f, -0.2725002468f, 0.1609114408f, 0.6436457634f
    };
    float parity_inference[4] = {0};
    float parity_training[4] = {0};
    float parity_cross_training[4] = {0};
    const float attention_scale = 0.7071067811865475f;
    metal_graph_begin_forward();
    CHECK(metal_graph_sdpa_f32(parity_qkv, NULL, 0, parity_inference,
                               2, 2, 1, 2, 1, attention_scale, 0, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(parity_inference, sizeof(parity_inference), 0) == 1);
    metal_graph_begin_forward();
    CHECK(metal_graph_sdpa_training_f32(parity_qkv, NULL, 0, parity_training,
                                        2, 2, 1, 2, 1, attention_scale, 0, 0,
                                        0x80000000u, 0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(parity_training, sizeof(parity_training), 0) == 1);
    metal_graph_begin_forward();
    CHECK(metal_graph_cross_sdpa_training_f32(parity_q, parity_k, parity_v, NULL, 0,
                                              parity_cross_training, 2, 2, 2, 1, 2, 1,
                                              attention_scale, 0, 0, 0x80000000u,
                                              0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(parity_cross_training, sizeof(parity_cross_training), 0) == 1);
    for (int i = 0; i < 4; i++) {
        CHECK(close_enough(parity_inference[i], expected_inference[i]));
        CHECK(close_enough(parity_training[i], expected_training[i]));
        CHECK(close_enough(parity_cross_training[i], expected_training[i]));
    }
    CHECK(!close_enough(parity_inference[0], parity_training[0]));

    int32_t tokens[2] = {1, 0};
    float embedding_weight[2] = {4.0f, 7.0f};
    float embedding_output[2] = {0.0f, 0.0f};
    metal_graph_begin_forward();
    CHECK(metal_graph_embedding_f32(tokens, embedding_weight, embedding_output, 2, 1, 2) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(embedding_output, sizeof(embedding_output), 0) == 1);
    CHECK(close_enough(embedding_output[0], 7.0f));
    CHECK(close_enough(embedding_output[1], 4.0f));

    float conv_input[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float conv_weight[1] = {2.0f};
    float conv_bias[1] = {1.0f};
    float conv_output[6] = {0};
    metal_graph_begin_forward();
    CHECK(metal_graph_conv1d_f32(conv_input, conv_weight, conv_bias, conv_output,
                                 2, 1, 3, 1, 3, 1, 1, 0, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(conv_output, sizeof(conv_output), 0) == 1);
    const float expected_conv[6] = {3.0f, 5.0f, 7.0f, 21.0f, 41.0f, 61.0f};
    for (int i = 0; i < 6; i++) CHECK(close_enough(conv_output[i], expected_conv[i]));

    metal_graph_reset();
    return 0;
}

static int test_device_resident_activation_tape(void) {
    float qkv[3] = {1.0f, 1.0f, 2.0f};
    float gpu_only_activation[1] = {-9.0f};
    metal_graph_begin_forward();
    CHECK(metal_graph_sdpa_f32(qkv, NULL, 0, gpu_only_activation,
                               1, 1, 1, 1, 1, 1.0f, 1, 0) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(gpu_only_activation[0] == -9.0f);

    float output[1] = {2.0f};
    float grad_output[1] = {3.0f};
    float grad_input[1] = {0.0f};
    struct {
        uint32_t length;
        uint32_t kind;
        float alpha;
        float beta;
    } params = {1u, 0u, 0.0f, 0.0f};
    void* hosts[5] = {gpu_only_activation, output, grad_output, grad_input, &params};
    size_t bytes[5] = {sizeof(gpu_only_activation), sizeof(output), sizeof(grad_output),
                       sizeof(grad_input), sizeof(params)};
    unsigned char access[5] = {
        METAL_TRAINING_READ, METAL_TRAINING_READ, METAL_TRAINING_READ,
        METAL_TRAINING_READ | METAL_TRAINING_WRITE, METAL_TRAINING_READ
    };
    CHECK(metal_training_begin() == 0);
    CHECK(metal_training_dispatch("activationBackward", "main", hosts, bytes,
                                  access, NULL, 5, 1, 1, 1) == 0);
    CHECK(metal_training_sync(grad_input, sizeof(grad_input)) == 0);
    metal_training_end();
    CHECK(close_enough(grad_input[0], 3.0f));
    /* The backward dispatcher read the inference MTLBuffer directly. */
    CHECK(gpu_only_activation[0] == -9.0f);
    metal_graph_reset();
    return 0;
}

typedef struct {
    VxEngineState* state;
    float base;
    int ok;
} MetalIsolationThread;

static void* run_metal_isolation_thread(void* opaque) {
    MetalIsolationThread* probe = (MetalIsolationThread*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(probe->state);
    probe->ok = 1;
    for (int iteration = 0; iteration < 32; iteration++) {
        float input[4];
        float output[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
        for (int index = 0; index < 4; index++)
            input[index] = probe->base + (float)(iteration * 4 + index);
        metal_graph_begin_forward();
        int copied = metal_graph_copy_f32(input, output, 4);
        int ended = metal_graph_end_forward();
        int synced = metal_graph_sync_host(output, sizeof(output), 0);
        if (!copied || ended != 0 || !synced ||
            memcmp(input, output, sizeof(input)) != 0) {
            probe->ok = 0;
            break;
        }
        metal_graph_reset();
    }
    vx_engine_state_scope_leave(scope);
    return NULL;
}

static int test_engine_state_isolation(VxEngineState* first) {
    VxEngineState* second =
        (VxEngineState*)calloc(1, sizeof(*second));
    CHECK(second != NULL);
    CHECK(vx_engine_state_init(second) == 0);

    float shared_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float shared_output[4] = {0};
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(shared_input, shared_output, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);

    VxEngineStateScope second_scope = vx_engine_state_scope_enter(second);
    CHECK(metal_init() == 0);
    shared_input[0] = 9.0f;
    shared_input[1] = 8.0f;
    shared_input[2] = 7.0f;
    shared_input[3] = 6.0f;
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(shared_input, shared_output, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    vx_engine_state_scope_leave(second_scope);

    CHECK(metal_graph_sync_host(shared_output, sizeof(shared_output), 0) == 1);
    const float first_expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(memcmp(shared_output, first_expected, sizeof(shared_output)) == 0);
    metal_graph_reset();

    second_scope = vx_engine_state_scope_enter(second);
    CHECK(metal_graph_sync_host(shared_output, sizeof(shared_output), 0) == 1);
    const float second_expected[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    CHECK(memcmp(shared_output, second_expected, sizeof(shared_output)) == 0);
    metal_graph_reset();
    vx_engine_state_scope_leave(second_scope);

    MetalIsolationThread first_probe = {first, 1000.0f, 0};
    MetalIsolationThread second_probe = {second, -1000.0f, 0};
    pthread_t first_thread;
    pthread_t second_thread;
    CHECK(pthread_create(&first_thread, NULL, run_metal_isolation_thread,
                         &first_probe) == 0);
    CHECK(pthread_create(&second_thread, NULL, run_metal_isolation_thread,
                         &second_probe) == 0);
    CHECK(pthread_join(first_thread, NULL) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    CHECK(first_probe.ok && second_probe.ok);

    second_scope = vx_engine_state_scope_enter(second);
    metal_cleanup();
    CHECK(metal_training_available() == 0);
    vx_engine_state_scope_leave(second_scope);
    vx_engine_state_deinit(second);
    free(second);

    VxEngineState* implicit =
        (VxEngineState*)calloc(1, sizeof(*implicit));
    CHECK(implicit != NULL);
    CHECK(vx_engine_state_init(implicit) == 0);
    VxEngineStateScope implicit_scope = vx_engine_state_scope_enter(implicit);
    CHECK(metal_init() == 0);
    vx_engine_state_scope_leave(implicit_scope);
    /* The generic engine capsule teardown must release this device reference
     * even when backend cleanup was not called explicitly. */
    vx_engine_state_deinit(implicit);
    free(implicit);

    CHECK(metal_training_available() == 1);
    float survivor_input[2] = {4.0f, 5.0f};
    float survivor_output[2] = {0};
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(survivor_input, survivor_output, 2) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(survivor_output,
                                sizeof(survivor_output), 0) == 1);
    CHECK(memcmp(survivor_input, survivor_output,
                 sizeof(survivor_input)) == 0);
    metal_graph_reset();
    return 0;
}

static int test_dynamic_shape_capacity_lifecycle(void) {
    float small_input_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float small_output_a[4] = {0};
    float small_input_b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float small_output_b[4] = {0};
    float large_input[128];
    float large_output[128] = {0};
    MetalGraphDynamicStateProbe bound = {0};
    MetalGraphDynamicStateProbe active = {0};
    MetalGraphDynamicStateProbe pooled = {0};
    MetalGraphDynamicStateProbe reused = {0};
    MetalGraphDynamicStateProbe grown = {0};
    for (int index = 0; index < 128; index++)
        large_input[index] = (float)(index - 17);

    metal_graph_reset();
    CHECK(metal_graph_bind_shape("copy:f32:[4]->[4]") == 0);
    CHECK(metal_graph_debug_dynamic_state(&bound) == 0);
    CHECK(bound.shape_generation != 0);
    CHECK(metal_graph_bind_shape("copy:f32:[4]->[4]") == 0);
    CHECK(metal_graph_debug_dynamic_state(&active) == 0);
    CHECK(active.shape_generation == bound.shape_generation);
    CHECK(metal_graph_bind_shape("") == -1);
    CHECK(metal_graph_debug_dynamic_state(&active) == 0);
    CHECK(active.shape_generation == bound.shape_generation);

    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(small_input_a, small_output_a, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(small_output_a, sizeof(small_output_a), 0) == 1);
    CHECK(memcmp(small_input_a, small_output_a, sizeof(small_input_a)) == 0);
    CHECK(metal_graph_debug_dynamic_state(&active) == 0);
    CHECK(active.active_capacity_bytes > 0 && active.pooled_capacity_bytes == 0 &&
          active.slot_count == 2);

    CHECK(metal_graph_bind_shape("copy:f32:[1,4]->[1,4]") == 0);
    CHECK(metal_graph_debug_dynamic_state(&pooled) == 0);
    CHECK(pooled.shape_generation != active.shape_generation &&
          pooled.capacity_generation == active.capacity_generation &&
          pooled.active_capacity_bytes == 0 &&
          pooled.pooled_capacity_bytes == active.active_capacity_bytes &&
          pooled.slot_count == active.slot_count);
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(small_input_b, small_output_b, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(small_output_b, sizeof(small_output_b), 0) == 1);
    CHECK(memcmp(small_input_b, small_output_b, sizeof(small_input_b)) == 0);
    CHECK(metal_graph_debug_dynamic_state(&reused) == 0);
    CHECK(reused.capacity_generation == pooled.capacity_generation &&
          reused.slot_count == pooled.slot_count &&
          reused.active_capacity_bytes == active.active_capacity_bytes);

    CHECK(metal_graph_bind_shape("copy:f32:[128]->[128]") == 0);
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(large_input, large_output, 128) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(large_output, sizeof(large_output), 0) == 1);
    CHECK(memcmp(large_input, large_output, sizeof(large_input)) == 0);
    CHECK(metal_graph_debug_dynamic_state(&grown) == 0);
    CHECK(grown.capacity_generation > reused.capacity_generation &&
          grown.active_capacity_bytes >= sizeof(large_input) * 2u &&
          grown.slot_count == reused.slot_count);
    metal_graph_reset();
    return 0;
}

static int test_dynamic_domain_reservation(void) {
    const size_t qgroupnorm_stats_bytes = 64u;
    const size_t qlayernorm_stats_bytes = 80u;
    float arena[64] = {0};
    float bootstrap_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float invariant[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    float bootstrap_output[4] = {0};
    float missing_input[4] = {0};
    float missing_output[4] = {0};
    const VolvoxAIEnginePhysicalSpan spans[2] = {
        {arena, 32u * sizeof(float)},
        {arena + 32, 32u * sizeof(float)},
    };
    const VolvoxAIEnginePhysicalSpan mismatch[2] = {
        {arena, 31u * sizeof(float)},
        {arena + 32, 32u * sizeof(float)},
    };
    MetalDomainLimits limits = {0};
    MetalGraphDynamicStateProbe before = {0};
    MetalGraphDynamicStateProbe rolled_back = {0};
    MetalGraphDynamicStateProbe reserved = {0};
    MetalGraphDynamicStateProbe same = {0};
    MetalGraphDynamicStateProbe rebound = {0};
    MetalGraphDynamicStateProbe executed = {0};

    CHECK(metal_query_domain_limits(&limits) == 0);
    CHECK(limits.maximum_buffer_bytes >= spans[0].capacity_bytes &&
          limits.maximum_workgroups[0] > 0u &&
          limits.maximum_workgroup_size[0] > 0u &&
          limits.maximum_threads_per_workgroup > 0u &&
          limits.maximum_tensor_slots >= 2u &&
          limits.maximum_bindings > 0u);

    metal_graph_reset();
    CHECK(metal_graph_bind_shape("domain:published") == 0);
    metal_graph_begin_forward();
    CHECK(metal_graph_add_f32(
              bootstrap_input, invariant, bootstrap_output, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(
              bootstrap_output, sizeof(bootstrap_output), 0) == 1);
    metal_graph_retain_weight(bootstrap_input, sizeof(bootstrap_input));
    metal_graph_demote_weight(bootstrap_input, sizeof(bootstrap_input));
    metal_graph_retain_weight(invariant, sizeof(invariant));
    CHECK(metal_graph_debug_dynamic_state(&before) == 0);
    CHECK(metal_test_fail_domain_allocation_after(1u) == 0);
    CHECK(metal_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == -2);
    CHECK(metal_graph_debug_dynamic_state(&rolled_back) == 0);
    CHECK(rolled_back.shape_generation == before.shape_generation &&
          rolled_back.capacity_generation == before.capacity_generation &&
          rolled_back.active_capacity_bytes == before.active_capacity_bytes &&
          rolled_back.pooled_capacity_bytes == before.pooled_capacity_bytes &&
          rolled_back.domain_scratch_capacity_bytes ==
              before.domain_scratch_capacity_bytes &&
          rolled_back.domain_span_count == 0u &&
          rolled_back.slot_count == before.slot_count &&
          !rolled_back.domain_enforced);
    CHECK(metal_test_fail_domain_allocation_after(3u) == 0);
    CHECK(metal_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == -2);
    CHECK(metal_graph_debug_dynamic_state(&rolled_back) == 0);
    CHECK(rolled_back.shape_generation == before.shape_generation &&
          rolled_back.capacity_generation == before.capacity_generation &&
          rolled_back.active_capacity_bytes == before.active_capacity_bytes &&
          rolled_back.pooled_capacity_bytes == before.pooled_capacity_bytes &&
          rolled_back.domain_scratch_capacity_bytes ==
              before.domain_scratch_capacity_bytes &&
          rolled_back.domain_span_count == 0u &&
          rolled_back.slot_count == before.slot_count &&
          !rolled_back.domain_enforced);

    CHECK(metal_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(metal_graph_debug_dynamic_state(&reserved) == 0);
    CHECK(reserved.shape_generation != before.shape_generation &&
          reserved.capacity_generation != before.capacity_generation &&
          reserved.active_capacity_bytes ==
              spans[0].capacity_bytes * 2u + sizeof(invariant) &&
          reserved.pooled_capacity_bytes == 0u &&
          reserved.domain_scratch_capacity_bytes ==
              qgroupnorm_stats_bytes + qlayernorm_stats_bytes &&
          reserved.domain_span_count == 2u && reserved.slot_count == 3 &&
          reserved.domain_enforced);

    CHECK(metal_graph_bind_shape_domain(
              "domain:q4", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(metal_graph_debug_dynamic_state(&same) == 0);
    CHECK(same.shape_generation == reserved.shape_generation &&
          same.capacity_generation == reserved.capacity_generation &&
          same.active_capacity_bytes == reserved.active_capacity_bytes &&
          same.domain_scratch_capacity_bytes ==
              reserved.domain_scratch_capacity_bytes &&
          same.slot_count == reserved.slot_count);

    CHECK(metal_graph_bind_shape_domain(
              "domain:q8", spans, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == 0);
    CHECK(metal_graph_debug_dynamic_state(&rebound) == 0);
    CHECK(rebound.shape_generation != same.shape_generation &&
          rebound.capacity_generation == same.capacity_generation &&
          rebound.active_capacity_bytes == same.active_capacity_bytes &&
          rebound.domain_scratch_capacity_bytes ==
              same.domain_scratch_capacity_bytes &&
          rebound.slot_count == same.slot_count);
    CHECK(metal_graph_bind_shape_domain(
              "domain:q8", spans, 2u, qgroupnorm_stats_bytes + 1u,
              qlayernorm_stats_bytes) == -1);
    CHECK(metal_graph_bind_shape_domain(
              "domain:q8", mismatch, 2u, qgroupnorm_stats_bytes,
              qlayernorm_stats_bytes) == -1);
    CHECK(metal_graph_debug_dynamic_state(&executed) == 0);
    CHECK(executed.shape_generation == rebound.shape_generation &&
          executed.capacity_generation == rebound.capacity_generation &&
          executed.active_capacity_bytes == rebound.active_capacity_bytes &&
          executed.domain_scratch_capacity_bytes ==
              rebound.domain_scratch_capacity_bytes &&
          executed.slot_count == rebound.slot_count);
    CHECK(metal_graph_bind_shape("domain:legacy") == -1);

    for (int index = 0; index < 4; index++) arena[index] = (float)(index + 3);
    metal_graph_begin_forward();
    CHECK(metal_graph_add_f32(arena, invariant, arena + 32, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(arena + 32, 4u * sizeof(float), 0) == 1);
    for (int index = 0; index < 4; index++)
        CHECK(arena[32 + index] == arena[index] + invariant[index]);
    metal_graph_begin_forward();
    CHECK(metal_graph_alias_f32(arena, arena + 32, 4) == 1);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_sync_host(arena + 32, 4u * sizeof(float), 0) == 1);
    CHECK(memcmp(arena, arena + 32, 4u * sizeof(float)) == 0);
    metal_graph_begin_forward();
    CHECK(metal_graph_copy_f32(missing_input, missing_output, 4) == 0);
    CHECK(metal_graph_end_forward() == 0);
    CHECK(metal_graph_debug_dynamic_state(&executed) == 0);
    CHECK(executed.capacity_generation == rebound.capacity_generation &&
          executed.active_capacity_bytes == rebound.active_capacity_bytes &&
          executed.slot_count == rebound.slot_count);
    metal_graph_reset();
    return 0;
}

int main(void) {
    VxEngineState* engine_state =
        (VxEngineState*)calloc(1, sizeof(*engine_state));
    CHECK(engine_state != NULL);
    CHECK(vx_engine_state_init(engine_state) == 0);
    VxEngineStateScope engine_scope =
        vx_engine_state_scope_enter(engine_state);
    if (metal_init() != 0) {
        fprintf(stderr, "Metal unavailable; skipping native Metal training test\n");
        metal_cleanup();
        vx_engine_state_scope_leave(engine_scope);
        vx_engine_state_deinit(engine_state);
        free(engine_state);
        return 0;
    }
    CHECK(metal_training_available() == 1);
    float host_current = 1.0f;
    metal_graph_reset();
    CHECK(metal_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    metal_graph_mark_host(&host_current, sizeof(host_current), 1);
    CHECK(metal_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    metal_graph_reset();
    size_t matmul_bytes[7] = {
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(float), sizeof(uint32_t)
    };
    CHECK(metal_training_supports("matMulBackward", "weight_main",
                                  matmul_bytes, 7, 1, 1, 1) == 1);
    CHECK(metal_training_supports("matMulBackward", "weight_main",
                                  matmul_bytes, 6, 1, 1, 1) == 0);
    CHECK(metal_training_supports("matMulBackward", "unknown",
                                  matmul_bytes, 7, 1, 1, 1) == 0);
#ifdef VOLVOX_METAL_TESTING
    CHECK(test_graph_forward_command_batch() == 0);
#endif
    CHECK(test_graph_f32_shape_and_activation_closure() == 0);
    CHECK(test_typed_control_graph_ops() == 0);
    CHECK(test_qbatch_and_typed_transpose() == 0);
    CHECK(test_graph_linear_tails_and_fallback() == 0);
    CHECK(test_tiled_qlinear_i8u8_tails() == 0);
    CHECK(test_batched_attention_forward() == 0);
    CHECK(test_gelu_modes() == 0);
    CHECK(test_qembedding_i8u8_packed_gather() == 0);
    CHECK(test_qsilu_i8u8_packed_chain() == 0);
    CHECK(test_qgelu_i8u8_packed_chain() == 0);
    CHECK(test_qgroupnorm_i8u8_packed_chain() == 0);
    CHECK(test_qlayernorm_i8u8_packed_chain() == 0);
    CHECK(test_qsdpa_i8u8_packed_chain() == 0);
    CHECK(test_qargmax_i8u8_raw() == 0);
    CHECK(test_qmaskedmean_i8u8_packed() == 0);
    CHECK(test_device_resident_activation_tape() == 0);
    CHECK(test_dynamic_shape_capacity_lifecycle() == 0);
    CHECK(test_dynamic_domain_reservation() == 0);
    CHECK(test_activation_backward() == 0);
    CHECK(test_multi_entry_matmul_backward() == 0);
    CHECK(test_prelu_logsoftmax_split_backward() == 0);
    CHECK(test_engine_state_isolation(engine_state) == 0);
    metal_cleanup();
    CHECK(metal_training_available() == 0);
    vx_engine_state_scope_leave(engine_scope);
    vx_engine_state_deinit(engine_state);
    free(engine_state);
    puts("metal training tests passed");
    return 0;
}
