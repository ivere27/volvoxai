#include "opengl_engine.h"
#include "runtime_state.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double volvoxai_engine_now_ms(void) { return 0.0; }
void prof_add_entry(const char* op, double ms) {
    (void)op;
    (void)ms;
}

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "OpenGL training check failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void fill_f32_matrix(float* values, size_t count, int multiplier, int modulus,
                            int center, float scale) {
    for (size_t i = 0; i < count; i++) {
        values[i] = (float)(((int)(i % (size_t)modulus) * multiplier) % modulus - center) * scale;
    }
}

static int check_one_shot_matmul_case(int rows, int d_in, int d_out,
                                      float* input, float* weight, float* bias,
                                      float* output) {
    float expected[17 * 23];
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < d_out; column++) {
            float sum = bias[column];
            for (int k = 0; k < d_in; k++) {
                sum += input[row * d_in + k] * weight[k * d_out + column];
            }
            expected[row * d_out + column] = sum;
        }
    }
    memset(output, 0, (size_t)rows * d_out * sizeof(float));
    CHECK(opengl_matmul(input, weight, bias, output, rows, d_in, d_out) == 1);
    for (int i = 0; i < rows * d_out; i++) CHECK(fabsf(output[i] - expected[i]) <= 1.0e-4f);
    return 0;
}

static int check_one_shot_matmul_tails_and_fallback(void) {
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
    CHECK(check_one_shot_matmul_case(17, 19, 23, tiled_input, tiled_weight,
                                     tiled_bias, tiled_output) == 0);
    CHECK(check_one_shot_matmul_case(3, 7, 5, scalar_input, scalar_weight,
                                     scalar_bias, scalar_output) == 0);
    return 0;
}

static int check_tiled_qlinear_i8u8_tails(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qlinear_i8u8(input, weight, scales, zero_points, bias, output,
                                    ROWS, D_IN, D_OUT, 1.0f, 0, 1.0f, 0,
                                    VX_DTYPE_I8, VX_DTYPE_I8,
                                    VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_tiled_qlinear_staged_rounding(void) {
    enum { ROWS = 2, D_IN = 16, D_OUT = 32 };
    const uint8_t input[ROWS * D_IN] = {0};
    const int8_t weight[D_OUT * D_IN] = {0};
    float weight_scales[D_OUT];
    const int32_t weight_zero_points[D_OUT] = {0};
    int32_t bias[D_OUT];
    uint8_t output[ROWS * D_OUT] = {0};
    const float input_scale = 0.028062894940376282f;
    const float weight_scale = 0.0006811817875131965f;
    const float output_scale = 0.02981325425207615f;
    const int32_t output_zero_point = 127;

    for (int column = 0; column < D_OUT; column++) {
        weight_scales[column] = weight_scale;
        bias[column] = -3899;
    }

    /* The staged f32 schedule lands on an even tie. Contracting the final
       multiply and add instead rounds to the adjacent integer 125. */
    volatile float scale_product = input_scale * weight_scale;
    volatile float multiplier = scale_product / output_scale;
    volatile float scaled = (float)bias[0] * multiplier;
    volatile float shifted = scaled + (float)output_zero_point;
    CHECK(shifted == 124.5f);
    CHECK(((int)floorf(shifted) & 1) == 0);

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qlinear_i8u8(
        input, weight, weight_scales, weight_zero_points, bias, output,
        ROWS, D_IN, D_OUT, input_scale, 0, output_scale,
        output_zero_point, VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < ROWS * D_OUT; index++)
        CHECK(output[index] == 124u);
    opengl_graph_reset();
    return 0;
}

static uint32_t dropout_bits(uint32_t seed, uint32_t counter, uint32_t index) {
    uint32_t value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
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

static int check_gelu_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7] = {0}, tanh_output[7] = {0};
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_gelu_f32(input, exact_output, 7, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(exact_output, sizeof(exact_output), 0) == 1);
    opengl_graph_begin_forward();
    CHECK(opengl_graph_gelu_f32(input, tanh_output, 7, 1) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(tanh_output, sizeof(tanh_output), 0) == 1);
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
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char is_weight[5] = {0, 0, 0, 0, 0};
    CHECK(opengl_training_begin() == 0);
    CHECK(opengl_training_dispatch("activationBackward", "main", exact_hosts, exact_bytes,
                                   access, is_weight, 5, 1, 1, 1) == 0);
    CHECK(opengl_training_dispatch("activationBackward", "main", tanh_hosts, tanh_bytes,
                                   access, is_weight, 5, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(exact_grad, sizeof(exact_grad)) == 0);
    CHECK(opengl_training_sync(tanh_grad, sizeof(tanh_grad)) == 0);
    opengl_training_end();
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 1)) < 2.0e-4f);
    }
    opengl_graph_reset();
    return 0;
}

static float layernorm_expected(const float* input, const float* weight,
                                const float* bias, int index, int d_model,
                                float epsilon) {
    int offset = (index / d_model) * d_model;
    float sum = 0.0f, square_sum = 0.0f;
    for (int i = 0; i < d_model; i++) {
        float value = input[offset + i];
        sum += value;
        square_sum += value * value;
    }
    float mean = sum / (float)d_model;
    float variance = square_sum / (float)d_model - mean * mean;
    int channel = index % d_model;
    return (input[index] - mean) / sqrtf(variance + epsilon) * weight[channel] +
        bias[channel];
}

static int check_layernorm_epsilon(void) {
    enum { ROWS = 2, D_MODEL = 4, ELEMENTS = ROWS * D_MODEL };
    const float input[ELEMENTS] = {
        -0.03f, 0.01f, 0.02f, 0.0f,
        1.0f, 1.002f, 0.998f, 1.001f
    };
    const float weight[D_MODEL] = {1.0f, 0.5f, 1.5f, 0.75f};
    const float bias[D_MODEL] = {0.1f, -0.2f, 0.3f, -0.4f};
    const float epsilons[2] = {0.25f, 1.0e-4f};
    float output[2][ELEMENTS] = {{0}};

    for (int pass = 0; pass < 2; pass++) {
        opengl_graph_reset();
        opengl_graph_begin_forward();
        CHECK(opengl_graph_layernorm_f32(input, weight, bias, output[pass],
                                         ROWS, D_MODEL, epsilons[pass]) == 1);
        CHECK(opengl_graph_end_forward() == 0);
        CHECK(opengl_graph_sync_host(output[pass], sizeof(output[pass]), 0) == 1);
        for (int i = 0; i < ELEMENTS; i++) {
            float expected = layernorm_expected(input, weight, bias, i, D_MODEL,
                                                epsilons[pass]);
            CHECK(isfinite(output[pass][i]));
            CHECK(fabsf(output[pass][i] - expected) < 5.0e-4f);
        }
    }
    CHECK(fabsf(output[0][0] - output[1][0]) > 0.5f);
    CHECK(opengl_graph_layernorm_f32(input, weight, bias, output[0],
                                     ROWS, D_MODEL, 0.0f) == 0);
    CHECK(opengl_graph_layernorm_f32(input, weight, bias, output[0],
                                     ROWS, D_MODEL, NAN) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_qlinear_i8u8_packed_chain(void) {
    /* Logical byte tails exercise the packed u32 shader view.  `hidden` stays
       on the OpenGL device between the two explicit W8A8 dispatches. */
    const int8_t input[3] = {2, -2, 4};
    const int8_t first_weight[6] = {1, 2, -1, -2, 1, 3};
    const float first_scales[2] = {0.25f, 0.5f};
    const int32_t first_zero_points[2] = {0, 0};
    const int32_t first_bias[2] = {0, 0};
    uint8_t hidden[2] = {0, 0};
    const int8_t second_weight[2] = {2, -2};
    const float second_scales[1] = {0.5f};
    const int32_t second_zero_points[1] = {0};
    const int32_t second_bias[1] = {0};
    int8_t output[1] = {0};

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qlinear_i8u8(input, first_weight, first_scales,
                                    first_zero_points, first_bias, hidden,
                                    1u, 3u, 2u, 0.5f, 0, 0.25f, 128,
                                    VX_DTYPE_I8, VX_DTYPE_I8,
                                    VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qlinear_i8u8(hidden, second_weight, second_scales,
                                    second_zero_points, second_bias, output,
                                    1u, 2u, 1u, 0.25f, 128, 0.25f, -3,
                                    VX_DTYPE_U8, VX_DTYPE_I8,
                                    VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(hidden[0] == 125u && hidden[1] == 134u);
    CHECK(output[0] == -12);
    opengl_graph_reset();
    return 0;
}

static int check_qembedding_i8u8_packed_gather(void) {
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

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qembedding_i8u8(ids, i8_table, scales, i8_zero_points, output_u8,
                                       3u, 3u, 3u, 0.25f, 128,
                                       VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == expected_u8[index]);

    opengl_graph_begin_forward();
    CHECK(opengl_graph_qembedding_i8u8(ids, u8_table, scales, u8_zero_points, output_i8,
                                       3u, 3u, 3u, 0.25f, -3,
                                       VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_i8[index] == expected_i8[index]);

    memset(output_u8, 0x5a, sizeof(output_u8));
    CHECK(opengl_graph_qembedding_i8u8(invalid_ids, i8_table, scales, i8_zero_points,
                                       output_u8, 3u, 3u, 3u, 0.25f, 128,
                                       VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == 0x5a);
    opengl_graph_reset();
    return 0;
}

static int check_qadd_requantize_i8u8_packed_chain(void) {
    /* The five-byte output exercises a padded packed-u32 tail without ever
       reading past the host buffers.  `sum` feeds the next GPU dispatch. */
    const int8_t a[5] = {-2, 1, 13, 0, 30};
    const uint8_t b[5] = {124, 128, 140, 129, 128};
    int8_t sum[5] = {0, 0, 0, 0, 0};
    uint8_t output[5] = {0, 0, 0, 0, 0};

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                                  0.5f, 0, 0.25f, 128,
                                  0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                                  VX_DTYPE_I8, 2u) == 1);
    CHECK(opengl_graph_requantize_linear_i8u8(sum, 5u, output, 5u,
                                              0.5f, -2, 1.0f, 100,
                                              VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(sum, sizeof(sum), 0) == 1);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    const int8_t expected_sum[5] = {-2, -1, 10, -2, 10};
    const uint8_t expected_output[5] = {100, 100, 106, 100, 106};
    for (int i = 0; i < 5; i++) {
        CHECK(sum[i] == expected_sum[i]);
        CHECK(output[i] == expected_output[i]);
    }
    CHECK(opengl_graph_qadd_i8u8(a, 5u, b, 4u, sum, 5u,
                                  0.5f, 0, 0.25f, 128,
                                  0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                                  VX_DTYPE_I8, 2u) == 0);
    CHECK(opengl_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                                  0.5f, 0, 0.25f, 128,
                                  0.5f, -2, VX_DTYPE_I8, VX_DTYPE_U8,
                                  VX_DTYPE_I8, 3u) == 0);
    CHECK(opengl_graph_requantize_linear_i8u8(sum, 5u, output, 4u,
                                              0.5f, -2, 1.0f, 100,
                                              VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_qsilu_i8u8_packed_chain(void) {
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

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qsilu_i8u8(input_i8, from_i8_i8, 5u,
                                   0.5f, 0, 0.25f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qsilu_i8u8(input_i8, from_i8_u8, 5u,
                                   0.5f, 0, 0.25f, 128,
                                   VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qsilu_i8u8(input_u8, from_u8_i8, 5u,
                                   0.5f, 128, 0.25f, -3,
                                   VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qsilu_i8u8(input_u8, from_u8_u8, 5u,
                                   0.5f, 128, 0.25f, 128,
                                   VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qsilu_i8u8(from_i8_u8, chained, 5u,
                                   0.25f, 128, 0.125f, -4,
                                   VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -5, -4, 0, 27};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(opengl_graph_qsilu_i8u8(input_i8, from_i8_i8, 0u,
                                   0.5f, 0, 0.25f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qsilu_i8u8(alias, alias, 5u,
                                   0.5f, 0, 0.25f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    opengl_graph_reset();
    return 0;
}

/* QGELU's only byte-domain form is the fixed A-S erf approximation.  Verify
 * every I8/U8 boundary and a packed U8 -> I8 device-resident chain. */
static int check_qgelu_i8u8_packed_chain(void) {
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

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qgelu_i8u8(input_i8, from_i8_i8, 5u,
                                   0.5f, 0, 0.125f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgelu_i8u8(input_i8, from_i8_u8, 5u,
                                   0.5f, 0, 0.125f, 128,
                                   VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qgelu_i8u8(input_u8, from_u8_i8, 5u,
                                   0.5f, 128, 0.125f, -3,
                                   VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgelu_i8u8(input_u8, from_u8_u8, 5u,
                                   0.5f, 128, 0.125f, 128,
                                   VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qgelu_i8u8(from_i8_u8, chained, 5u,
                                   0.125f, 128, 0.125f, -4,
                                   VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -4, -4, 2, 28};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(opengl_graph_qgelu_i8u8(input_i8, from_i8_i8, 0u,
                                   0.5f, 0, 0.125f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qgelu_i8u8(alias, alias, 5u,
                                   0.5f, 0, 0.125f, -3,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    opengl_graph_reset();
    return 0;
}

/* Keep the same non-boundary vectors as the Vulkan test.  The fixed 64-lane
 * reduction is deliberately exercised with a C=3 packed tail, C/G=3 group
 * boundaries, and a high-U8 centered-variance regression. */
static int check_qgroupnorm_i8u8_packed_chain(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                        1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                        0.125f, -3, 1.0e-5f,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                        1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                        0.125f, 128, 1.0e-5f,
                                        VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                        1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                        0.125f, -3, 1.0e-5f,
                                        VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                        1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                        0.125f, 128, 1.0e-5f,
                                        VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    /* The second stats/apply pair consumes a still-device-resident U8 tensor. */
    CHECK(opengl_graph_qgroupnorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                        1u, 1u, 1u, 3u, 1u, 0.125f, 128,
                                        0.125f, -4, 1.0e-5f,
                                        VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgroupnorm_i8u8(boundary_input, boundary_gamma, boundary_beta,
                                        boundary_output, 3u, 1u, 1u, 6u, 2u,
                                        0.25f, -1, 0.125f, -3, 1.0e-5f,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qgroupnorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                        3u, 57u, 1u, 6u, 2u, 0.5f, 17,
                                        0.25f, 128, 1.0e-5f,
                                        VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(opengl_graph_sync_host(boundary_output, sizeof(boundary_output), 0) == 1);
    CHECK(opengl_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
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
    CHECK(opengl_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                        1u, 1u, 1u, 3u, 0u, 0.5f, -1,
                                        0.125f, -3, 1.0e-5f,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qgroupnorm_i8u8(alias, gamma, beta, alias,
                                        1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                        0.125f, -3, 1.0e-5f,
                                        VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    opengl_graph_reset();
    return 0;
}

/* Rows with D=3 cross a packed-word boundary; test every typed boundary plus
 * a long high-U8 row where centered variance avoids cancellation. */
static int check_qlayernorm_i8u8_packed_chain(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                        2u, 3u, 0.5f, -1, 0.125f, -3,
                                        1.0e-5f, VX_DTYPE_I8,
                                        VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                        2u, 3u, 0.5f, -1, 0.125f, 128,
                                        1.0e-5f, VX_DTYPE_I8,
                                        VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                        2u, 3u, 0.5f, 127, 0.125f, -3,
                                        1.0e-5f, VX_DTYPE_U8,
                                        VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                        2u, 3u, 0.5f, 127, 0.125f, 128,
                                        1.0e-5f, VX_DTYPE_U8,
                                        VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_qlayernorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                        2u, 3u, 0.125f, 128, 0.125f, -4,
                                        1.0e-5f, VX_DTYPE_U8,
                                        VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qlayernorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                        2u, high_d_model, 0.5f, 17, 0.25f, 128,
                                        1.0e-5f, VX_DTYPE_U8,
                                        VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(opengl_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(opengl_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
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
    CHECK(opengl_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                        2u, 0u, 0.5f, -1, 0.125f, -3,
                                        1.0e-5f, VX_DTYPE_I8,
                                        VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qlayernorm_i8u8(alias, gamma, beta, alias,
                                        2u, 3u, 0.5f, -1, 0.125f, -3,
                                        1.0e-5f, VX_DTYPE_I8,
                                        VX_DTYPE_I8) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_qsdpa_i8u8_packed_chain(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, first, 1u, 2u, 2u,
                                   4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                                   0.25f, 128, 0.5f, VX_DTYPE_I8,
                                   VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8,
                                   0u, 0u) == 1);
    CHECK(opengl_graph_qsdpa_i8u8(first, k_i8, v_i8, NULL, second, 1u, 2u, 2u,
                                   4u, 1u, 0.25f, 128, 0.25f, -1, 0.25f, -1,
                                   0.25f, 0, 0.5f, VX_DTYPE_U8,
                                   VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                   0u, 0u) == 1);
    CHECK(opengl_graph_qsdpa_i8u8(q_u8, k_u8, v_u8, mask_none, all_masked,
                                   1u, 1u, 2u, 4u, 1u, 0.25f, 128, 0.5f, 120,
                                   0.25f, 130, 0.25f, 127, 0.5f,
                                   VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                                   VX_DTYPE_U8, 0u, 1u) == 1);
    CHECK(opengl_graph_qsdpa_i8u8(q64, k64, v64, NULL, out64, 1u, 1u, 1u,
                                   head_dim, 1u, 0.25f, 0, 0.25f, 0, 0.25f, 0,
                                   0.25f, 0, 1.0f, VX_DTYPE_I8,
                                   VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                   0u, 0u) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(first, sizeof(first), 0) == 1);
    CHECK(opengl_graph_sync_host(second, sizeof(second), 0) == 1);
    CHECK(opengl_graph_sync_host(all_masked, sizeof(all_masked), 0) == 1);
    CHECK(opengl_graph_sync_host(out64, sizeof(out64), 0) == 1);
    CHECK(memcmp(first, expected_first, sizeof(first)) == 0);
    CHECK(memcmp(second, expected_second, sizeof(second)) == 0);
    CHECK(memcmp(all_masked, expected_zero, sizeof(all_masked)) == 0);
    CHECK(memcmp(out64, v64, sizeof(out64)) == 0);
    CHECK(opengl_graph_qsdpa_i8u8(alias, k_i8, v_i8, NULL, alias, 1u, 2u, 2u,
                                   4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                                   0.25f, 0, 0.5f, VX_DTYPE_I8,
                                   VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                                   0u, 0u) == 0);
    CHECK(memcmp(alias, alias_before, sizeof(alias)) == 0);
    opengl_graph_reset();
    return 0;
}

/* qArgMaxInt8 reads packed byte storage directly and emits conventional I32
 * indices. This includes the non-word-aligned I8 input tail, signed negative
 * ordering, U8 order with an asymmetric zero point, and first-tie behavior. */
static int check_qargmax_i8u8_raw(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u,
                                    VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u,
                                    VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(opengl_graph_qargmax_i8u8(alias.bytes, alias.i32, 1u, 3u, 2u,
                                    VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(opengl_graph_qargmax_i8u8(input_i8, sentinel, 1u, 0u, 2u,
                                    VX_DTYPE_I8) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_qmaskedmean_i8u8_packed(void) {
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
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qmaskedmean_i8u8(input_i8, mask_i8, output_i8,
                                         2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                         VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_qmaskedmean_i8u8(input_u8, mask_u8, output_u8,
                                         1u, 3u, 2u, 0.25f, 128, 0.25f, 130,
                                         VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(opengl_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(opengl_graph_qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes,
                                         2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                         VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(opengl_graph_qmaskedmean_i8u8(input_i8, mask_alias.mask,
                                         mask_alias.bytes, 2u, 3u, 4u,
                                         0.25f, -3, 0.5f, 5,
                                         VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_alias.bytes)) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_qconv2d_i8u8_packed_chain(void) {
    const uint8_t input[36] = {
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
        1, 2, 3, 1, 2, 3, 1, 2, 3,
    };
    const int8_t weight[12] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const float scales[3] = {1.0f, 1.0f, 1.0f};
    const int32_t zero_points[3] = {0, 0, 0};
    uint8_t hidden[27] = {0};
    int8_t output[27] = {0};
    const uint8_t expected_hidden[27] = {
        1, 2, 3, 2, 4, 6, 1, 2, 3,
        2, 4, 6, 4, 6, 6, 2, 4, 6,
        1, 2, 3, 2, 4, 6, 1, 2, 3,
    };
    const int8_t expected_output[27] = {
        0, 2, 4, 2, 6, 10, 0, 2, 4,
        2, 6, 10, 6, 10, 10, 2, 6, 10,
        0, 2, 4, 2, 6, 10, 0, 2, 4,
    };

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                     1u, 3u, 4u, 3u, 3u, 3u, 3u, 2u, 2u, 1u,
                                     1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                     1.0f, 0, 1.0f, 0, VX_DTYPE_U8,
                                     VX_DTYPE_I8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_requantize_linear_i8u8(hidden, 27u, output, 27u,
                                              1.0f, 0, 0.5f, -2,
                                              VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int i = 0; i < 27; i++) {
        CHECK(hidden[i] == expected_hidden[i]);
        CHECK(output[i] == expected_output[i]);
    }
    opengl_graph_reset();

    const int8_t bias_input[1] = {-2};
    const int8_t bias_weight[1] = {3};
    const float bias_scale[1] = {0.25f};
    const int32_t bias_zero_point[1] = {0};
    const int32_t bias[1] = {4};
    int8_t bias_output[1] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qconv2d_i8u8(bias_input, bias_weight, bias_scale,
                                     bias_zero_point, bias, bias_output,
                                     1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                                     1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 1u,
                                     0.5f, 0, 0.25f, -3, VX_DTYPE_I8,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(bias_output, sizeof(bias_output), 0) == 1);
    CHECK(bias_output[0] == -3);
    CHECK(opengl_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                     1u, 3u, 4u, 3u, 2u, 3u, 3u, 2u, 2u, 1u,
                                     1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                     1.0f, 0, 1.0f, 0, VX_DTYPE_U8,
                                     VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    opengl_graph_reset();

    /* Exercise packed three-byte reduction tails and unaligned per-channel
       weight rows; the larger case below forces the tiled heuristic. */
    const uint8_t tiled_input[18] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
    };
    const int8_t tiled_weight[12] = {
        1, 1, 1, 1, 0, -1, -1, 2, 0, 0, 0, 0,
    };
    const float tiled_scales[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const int32_t tiled_zero_points[4] = {0, 0, 0, 0};
    const int32_t tiled_bias[4] = {0, 1, -2, 5};
    const int8_t tiled_expected[24] = {
        6, -1, 1, 5, 15, -1, 4, 5, 24, -1, 7, 5,
        33, -1, 10, 5, 42, -1, 13, 5, 51, -1, 16, 5,
    };
    int8_t tiled_output[24] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qconv2d_i8u8(
              tiled_input, tiled_weight, tiled_scales, tiled_zero_points,
              tiled_bias, tiled_output, 1u, 2u, 3u, 3u, 2u, 3u, 4u,
              1u, 1u, 3u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 0u,
              1.0f, 0, 1.0f, 0, VX_DTYPE_U8, VX_DTYPE_I8,
              VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(tiled_output, sizeof(tiled_output), 0) == 1);
    CHECK(memcmp(tiled_output, tiled_expected, sizeof(tiled_output)) == 0);
    opengl_graph_reset();

    const int8_t mixed_input[6] = {-1, 0, 1, 2, -2, 3};
    const uint8_t mixed_weight[12] = {
        2, 3, 4, 1, 2, 3, 5, 4, 3, 2, 2, 2,
    };
    const float mixed_scales[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    const int32_t mixed_zero_points[4] = {2, 2, 3, 2};
    const int32_t mixed_bias[4] = {1, -1, 2, 3};
    const uint8_t mixed_expected[8] = {134, 129, 131, 131, 136, 128, 135, 131};
    uint8_t mixed_output[8] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qconv2d_i8u8(
              mixed_input, mixed_weight, mixed_scales, mixed_zero_points,
              mixed_bias, mixed_output, 1u, 1u, 2u, 3u, 1u, 2u, 4u,
              1u, 1u, 3u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 0u,
              0.5f, -1, 0.125f, 128, VX_DTYPE_I8, VX_DTYPE_U8,
              VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(mixed_output, sizeof(mixed_output), 0) == 1);
    CHECK(memcmp(mixed_output, mixed_expected, sizeof(mixed_output)) == 0);
    opengl_graph_reset();

    enum { TILED_IN_C = 19, TILED_OUT_C = 32, TILED_PIXELS = 2 };
    int8_t forced_input[TILED_PIXELS * TILED_IN_C];
    uint8_t forced_weight[TILED_OUT_C * TILED_IN_C];
    float forced_scales[TILED_OUT_C];
    int32_t forced_zero_points[TILED_OUT_C];
    int32_t forced_bias[TILED_OUT_C];
    uint8_t forced_expected[TILED_PIXELS * TILED_OUT_C];
    uint8_t forced_output[TILED_PIXELS * TILED_OUT_C];
    for (int index = 0; index < TILED_PIXELS * TILED_IN_C; index++) {
        forced_input[index] = (int8_t)((index * 3) % 7 - 3);
    }
    for (int channel = 0; channel < TILED_OUT_C; channel++) {
        forced_scales[channel] = 0.25f;
        forced_zero_points[channel] = 3;
        forced_bias[channel] = channel % 5 - 2;
        for (int reduction = 0; reduction < TILED_IN_C; reduction++) {
            forced_weight[channel * TILED_IN_C + reduction] =
                (uint8_t)(1 + ((channel * 7 + reduction * 3) % 5));
        }
    }
    for (int pixel = 0; pixel < TILED_PIXELS; pixel++) {
        for (int channel = 0; channel < TILED_OUT_C; channel++) {
            int32_t accumulator = forced_bias[channel];
            for (int reduction = 0; reduction < TILED_IN_C; reduction++) {
                accumulator +=
                    ((int32_t)forced_input[pixel * TILED_IN_C + reduction] + 1) *
                    ((int32_t)forced_weight[channel * TILED_IN_C + reduction] - 3);
            }
            accumulator += 128;
            if (accumulator < 0) accumulator = 0;
            if (accumulator > 255) accumulator = 255;
            forced_expected[pixel * TILED_OUT_C + channel] = (uint8_t)accumulator;
        }
    }
    memset(forced_output, 0, sizeof(forced_output));
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qconv2d_i8u8(
              forced_input, forced_weight, forced_scales, forced_zero_points,
              forced_bias, forced_output, 1u, 1u, TILED_PIXELS, TILED_IN_C,
              1u, TILED_PIXELS, TILED_OUT_C, 1u, 1u, TILED_IN_C,
              1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 0u,
              0.5f, -1, 0.125f, 128, VX_DTYPE_I8, VX_DTYPE_U8,
              VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(forced_output, sizeof(forced_output), 0) == 1);
    CHECK(memcmp(forced_output, forced_expected, sizeof(forced_output)) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_typed_i8u8_shape_qdq_chain(void) {
    const float source_a[4] = {2.0f, 4.0f, 1.0f, 3.0f};
    const float source_b[4] = {0.0f, 5.0f, 6.0f, 1.0f};
    int8_t quantized_a[4] = {0};
    int8_t copied_a[4] = {0};
    int8_t quantized_b[4] = {0};
    int8_t concatenated[8] = {0};
    int8_t pooled[8] = {0};
    int8_t resized[18] = {0};
    float output[18] = {0};
    const void* inputs[2] = {copied_a, quantized_b};
    const uint32_t input_elements[2] = {4u, 4u};
    const uint32_t input_axes[2] = {1u, 1u};
    const float input_scales[2] = {1.0f, 1.0f};
    const int32_t input_zero_points[2] = {0, 0};
    const uint32_t input_dtypes[2] = {VX_DTYPE_I8, VX_DTYPE_I8};
    const int8_t expected[18] = {
        4, 6, 4, 6, 4, 5,
        4, 6, 4, 6, 4, 5,
        3, 6, 3, 6, 3, 1,
    };

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_quantize_typed_f32_i8u8(source_a, 4u, quantized_a,
                                                1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                                 1.0f, 0, 1.0f, 0,
                                 VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_quantize_typed_f32_i8u8(source_b, 4u, quantized_b,
                                                1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_concat_i8u8(inputs, input_elements, input_axes,
                                   input_scales, input_zero_points, input_dtypes,
                                   2u, concatenated, 8u, 2u, 1u,
                                   1.0f, 0, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_maxpool2d_i8u8(concatenated, pooled,
                                      1u, 2u, 2u, 2u, 2u, 2u,
                                      2u, 2u, 1u, 1u, 0u, 0u, 1u, 1u,
                                      1.0f, 0, 1.0f, 0,
                                      VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_resize_nearest_i8u8(pooled, resized,
                                           1u, 2u, 2u, 2u, 3u, 3u,
                                           1.0f, 0, 1.0f, 0,
                                           VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_dequantize_typed_i8u8_f32(resized, 18u,
                                                  1.0f, 0, VX_DTYPE_I8,
                                                  output) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < 18; index++) {
        CHECK(fabsf(output[index] - (float)expected[index]) < 1.0e-4f);
    }
    const float unsigned_source[3] = {-1.0f, 0.0f, 5.0f};
    uint8_t unsigned_quantized[3] = {0};
    float unsigned_output[3] = {0};
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_quantize_typed_f32_i8u8(unsigned_source, 3u,
                                                unsigned_quantized, 1.0f, 128,
                                                VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_dequantize_typed_i8u8_f32(unsigned_quantized, 3u,
                                                  1.0f, 128, VX_DTYPE_U8,
                                                  unsigned_output) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(unsigned_quantized, sizeof(unsigned_quantized), 0) == 1);
    CHECK(opengl_graph_sync_host(unsigned_output, sizeof(unsigned_output), 0) == 1);
    CHECK(unsigned_quantized[0] == 127u && unsigned_quantized[1] == 128u &&
          unsigned_quantized[2] == 133u);
    CHECK(fabsf(unsigned_output[0] + 1.0f) < 1.0e-4f);
    CHECK(fabsf(unsigned_output[1]) < 1.0e-4f);
    CHECK(fabsf(unsigned_output[2] - 5.0f) < 1.0e-4f);
    CHECK(opengl_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                                 1.0f, 0, 0.5f, 0,
                                 VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_capability_rules(void) {
    CHECK(!opengl_compute_version_supported(OPENGL_COMPUTE_API_DESKTOP, 4, 2));
    CHECK(opengl_compute_version_supported(OPENGL_COMPUTE_API_DESKTOP, 4, 3));
    CHECK(opengl_compute_version_supported(OPENGL_COMPUTE_API_DESKTOP, 5, 0));
    CHECK(!opengl_compute_version_supported(OPENGL_COMPUTE_API_GLES, 3, 0));
    CHECK(opengl_compute_version_supported(OPENGL_COMPUTE_API_GLES, 3, 1));
    CHECK(opengl_compute_version_supported(OPENGL_COMPUTE_API_GLES, 3, 2));
    CHECK(!opengl_compute_version_supported(OPENGL_COMPUTE_API_NONE, 9, 9));
    return 0;
}

static int check_lazy_training_dispatch(void) {
    int programs = -1, buffers = -1, inference_buffers = -1;
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs == 0 && buffers == 0 && inference_buffers == 0);
    CHECK(opengl_training_begin() == 0);
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs == 0 && buffers == 0 && inference_buffers == 0);

    float grad_output[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float grad_input[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    uint32_t params[4] = {4, 0, 0, 0};
    void* hosts[3] = {grad_output, grad_input, params};
    size_t bytes[3] = {sizeof(grad_output), sizeof(grad_input), sizeof(params)};
    unsigned char access[3] = {
        OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ,
    };
    unsigned char is_weight[3] = {0, 0, 0};

    CHECK(opengl_training_dispatch("notARealShader", "main", hosts, bytes, access,
                                   is_weight, 3, 1, 1, 1) == -1);
    access[1] = OPENGL_TRAINING_ACCESS_WRITE;
    CHECK(opengl_training_dispatch("copyBackward", "main", hosts, bytes, access,
                                   is_weight, 3, 1, 1, 1) == -1);
    access[1] = OPENGL_TRAINING_ACCESS_READ_WRITE;
    CHECK(opengl_training_dispatch("copyBackward", "main", hosts, bytes, access,
                                   is_weight, 3, UINT32_MAX, 1, 1) == -1);
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs == 0 && buffers == 0 && inference_buffers == 0);

    CHECK(opengl_training_dispatch("copyBackward", "main", hosts, bytes, access,
                                   is_weight, 3, 1, 1, 1) == 0);
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs == 1 && buffers == 2 && inference_buffers == 0);
    CHECK(opengl_training_sync(grad_input, sizeof(grad_input)) == 0);
    for (int i = 0; i < 4; i++) CHECK(fabsf(grad_input[i] - (0.5f + grad_output[i])) < 1e-6f);

    opengl_training_end();
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs == 1 && buffers == 0 && inference_buffers == 0);
    return 0;
}

static int check_prelu_logsoftmax_split_backward(void) {
    CHECK(opengl_training_begin() == 0);
    float input[4] = {-1.0f, 2.0f, -0.5f, 1.5f};
    float weight[2] = {0.2f, 0.4f};
    float grad_output[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad_input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_weight[2] = {0.0f, 0.0f};
    uint32_t prelu_params[4] = {4u, 2u, 2u, 0u};
    void* prelu_hosts[6] = {input, weight, grad_output, grad_input, grad_weight, prelu_params};
    size_t prelu_bytes[6] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                             sizeof(grad_input), sizeof(grad_weight), sizeof(prelu_params)};
    unsigned char prelu_access[6] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ_WRITE, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char prelu_weights[6] = {0, 1, 0, 0, 0, 0};
    CHECK(opengl_training_dispatch("preluBackward", "input_main", prelu_hosts, prelu_bytes,
                                   prelu_access, prelu_weights, 6, 1, 1, 1) == 0);
    CHECK(opengl_training_dispatch("preluBackward", "weight_main", prelu_hosts, prelu_bytes,
                                   prelu_access, prelu_weights, 6, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(opengl_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(fabsf(grad_input[0] - 0.2f) < 1.0e-5f && fabsf(grad_input[1] - 2.0f) < 1.0e-5f);
    CHECK(fabsf(grad_input[2] - 0.6f) < 1.0e-5f && fabsf(grad_input[3] - 4.0f) < 1.0e-5f);
    CHECK(fabsf(grad_weight[0] + 2.5f) < 1.0e-5f && fabsf(grad_weight[1]) < 1.0e-5f);

    float log_output[2] = {-1.38629436112f, -0.28768207245f};
    float log_grad_output[2] = {2.0f, -1.0f};
    float log_grad_input[2] = {0.0f, 0.0f};
    uint32_t softmax_params[4] = {1u, 2u, 1u, 0u};
    void* softmax_hosts[4] = {log_output, log_grad_output, log_grad_input, softmax_params};
    size_t softmax_bytes[4] = {sizeof(log_output), sizeof(log_grad_output),
                               sizeof(log_grad_input), sizeof(softmax_params)};
    unsigned char softmax_access[4] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ_WRITE, OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char softmax_weights[4] = {0, 0, 0, 0};
    CHECK(opengl_training_dispatch("softmaxBackward", "main", softmax_hosts, softmax_bytes,
                                   softmax_access, softmax_weights, 4, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(log_grad_input, sizeof(log_grad_input)) == 0);
    CHECK(fabsf(log_grad_input[0] - 1.75f) < 1.0e-5f);
    CHECK(fabsf(log_grad_input[1] + 1.75f) < 1.0e-5f);

    float left_grad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float right_grad[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float split_grad[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t left_params[5] = {4u, 1u, 2u, 4u, 0u};
    uint32_t right_params[5] = {4u, 1u, 2u, 4u, 2u};
    void* split_hosts[3] = {left_grad, split_grad, left_params};
    size_t split_bytes[3] = {sizeof(left_grad), sizeof(split_grad), sizeof(left_params)};
    unsigned char split_access[3] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char split_weights[3] = {0, 0, 0};
    CHECK(opengl_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                                   split_access, split_weights, 3, 1, 1, 1) == 0);
    split_hosts[0] = right_grad;
    split_hosts[2] = right_params;
    CHECK(opengl_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                                   split_access, split_weights, 3, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(split_grad, sizeof(split_grad)) == 0);
    const float expected_split[8] = {1.0f, 2.0f, 5.0f, 6.0f,
                                     3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; i++) CHECK(fabsf(split_grad[i] - expected_split[i]) < 1.0e-5f);
    opengl_training_end();
    return 0;
}

static int check_groupnorm_dropout_reduce_and_broadcast(void) {
    CHECK(opengl_training_begin() == 0);
    float input[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    float weight[4] = {1.0f, 0.5f, 1.5f, 0.75f};
    float grad_output[4] = {1.0f, 2.0f, -1.0f, 3.0f};
    float grad_input[4] = {0};
    float grad_weight[4] = {0};
    float grad_bias[4] = {0};
    struct {
        uint32_t batch, height, width, channels, groups, has_bias;
        float epsilon;
        uint32_t pad;
    } params = {1u, 1u, 1u, 4u, 2u, 1u, 1.0e-4f, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, &params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ_WRITE, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char group_weights[7] = {0};
    CHECK(opengl_training_dispatch("groupNormBackward", "input_main", hosts, bytes,
                                   access, group_weights, 7, 1, 1, 1) == 0);
    CHECK(opengl_training_dispatch("groupNormBackward", "param_main", hosts, bytes,
                                   access, group_weights, 7, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(opengl_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    for (int i = 0; i < 4; i++) CHECK(fabsf(grad_bias[i] - grad_output[i]) < 1.0e-5f);

    float dropout_go[4] = {1, 2, 3, 4};
    float dropout_gi[4] = {0};
    struct { uint32_t length, threshold, seed, counter; float scale; uint32_t pad[3]; }
        dropout_params = {4u, 0x80000000u, 123u, 7u, 2.0f, {0u, 0u, 0u}};
    void* dropout_hosts[3] = {dropout_go, dropout_gi, &dropout_params};
    size_t dropout_bytes[3] = {sizeof(dropout_go), sizeof(dropout_gi), sizeof(dropout_params)};
    unsigned char simple_access[3] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char simple_weights[3] = {0};
    CHECK(opengl_training_dispatch("dropoutBackward", "main", dropout_hosts,
                                   dropout_bytes, simple_access, simple_weights, 3, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(dropout_gi, sizeof(dropout_gi)) == 0);
    for (int i = 0; i < 4; i++) {
        float expected = dropout_bits(123u, 7u, (uint32_t)i) >= 0x80000000u
            ? dropout_go[i] * 2.0f : 0.0f;
        CHECK(fabsf(dropout_gi[i] - expected) < 1.0e-6f);
    }

    float reduce_go[2] = {2.0f, 4.0f};
    float reduce_gi[6] = {0};
    struct { uint32_t length, width; float scale; uint32_t pad; }
        reduce_params = {6u, 3u, 1.0f / 3.0f, 0u};
    void* reduce_hosts[3] = {reduce_go, reduce_gi, &reduce_params};
    size_t reduce_bytes[3] = {sizeof(reduce_go), sizeof(reduce_gi), sizeof(reduce_params)};
    CHECK(opengl_training_dispatch("reduceBackward", "main", reduce_hosts,
                                   reduce_bytes, simple_access, simple_weights, 3, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(reduce_gi, sizeof(reduce_gi)) == 0);
    for (int i = 0; i < 3; i++) CHECK(fabsf(reduce_gi[i] - 2.0f / 3.0f) < 1.0e-5f);
    for (int i = 3; i < 6; i++) CHECK(fabsf(reduce_gi[i] - 4.0f / 3.0f) < 1.0e-5f);

    float binary_a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float binary_b[4] = {2, 3, 5, 7};
    float binary_go[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float grad_a[12] = {0};
    float grad_b[4] = {0};
    uint32_t binary_params[32] = {3u, 12u, 12u, 4u, 1u, 0u, 0u, 0u};
    const uint32_t binary_output_strides[3] = {6u, 2u, 1u};
    const uint32_t binary_a_strides[3] = {6u, 2u, 1u};
    const uint32_t binary_b_strides[3] = {2u, 0u, 1u};
    for (int i = 0; i < 3; i++) {
        binary_params[8 + i] = binary_output_strides[i];
        binary_params[16 + i] = binary_a_strides[i];
        binary_params[24 + i] = binary_b_strides[i];
    }
    void* binary_hosts[6] = {binary_a, binary_b, binary_go, grad_a, grad_b, binary_params};
    size_t binary_bytes[6] = {sizeof(binary_a), sizeof(binary_b), sizeof(binary_go),
                              sizeof(grad_a), sizeof(grad_b), sizeof(binary_params)};
    unsigned char binary_access[6] = {
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ, OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ_WRITE, OPENGL_TRAINING_ACCESS_READ
    };
    unsigned char binary_weights[6] = {0};
    CHECK(opengl_training_dispatch("basicBackward", "a_main", binary_hosts,
                                   binary_bytes, binary_access, binary_weights,
                                   6, 1, 1, 1) == 0);
    CHECK(opengl_training_dispatch("basicBackward", "b_main", binary_hosts,
                                   binary_bytes, binary_access, binary_weights,
                                   6, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(opengl_training_sync(grad_b, sizeof(grad_b)) == 0);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(fabsf(grad_a[index] - binary_b[batch * 2 + column]) < 1.0e-5f);
        }
    CHECK(fabsf(grad_b[0] - 9.0f) < 1.0e-5f && fabsf(grad_b[1] - 12.0f) < 1.0e-5f);
    CHECK(fabsf(grad_b[2] - 27.0f) < 1.0e-5f && fabsf(grad_b[3] - 30.0f) < 1.0e-5f);
    opengl_training_end();

    opengl_graph_reset();
    float bias[4] = {0.1f, -0.2f, 0.3f, 0.4f};
    float normalized[4] = {0};
    CHECK(opengl_graph_groupnorm_f32(input, weight, bias, normalized,
                                     1, 1, 1, 4, 2, 1.0e-4f) == 1);
    CHECK(opengl_graph_sync_host(normalized, sizeof(normalized), 0) == 1);
    for (int group = 0; group < 2; group++) {
        float mean = (input[group * 2] + input[group * 2 + 1]) * 0.5f;
        float centered = input[group * 2] - mean;
        float inverse = 1.0f / sqrtf(centered * centered + 1.0e-4f);
        for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            float expected = (input[channel] - mean) * inverse * weight[channel] + bias[channel];
            CHECK(fabsf(normalized[channel] - expected) < 1.0e-4f);
        }
    }
    float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b[4] = {2, 3, 5, 7};
    float product[12] = {0};
    uint32_t output_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t a_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t b_strides[8] = {2, 0, 1, 0, 0, 0, 0, 0};
    CHECK(opengl_graph_binary_f32(a, 12, b, 4, product, 12, output_strides,
                                  a_strides, b_strides, 3, 0) == 1);
    CHECK(opengl_graph_sync_host(product, sizeof(product), 0) == 1);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(fabsf(product[index] - a[index] * b[batch * 2 + column]) < 1.0e-5f);
        }
    opengl_graph_reset();
    return 0;
}

static int check_batched_attention(void) {
    float qkv[6] = {1.0f, 2.0f, 3.0f, -1.0f, 4.0f, 7.0f};
    float sdpa_out[2] = {0.0f, 0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_sdpa_f32(qkv, NULL, 0, sdpa_out,
                                1, 1, 1, 1, 2, 1.0f, 1, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(sdpa_out, sizeof(sdpa_out), 0) == 1);
    CHECK(fabsf(sdpa_out[0] - 3.0f) < 1e-6f);
    CHECK(fabsf(sdpa_out[1] - 7.0f) < 1e-6f);

    float query[2] = {1.0f, -1.0f};
    float key[2] = {2.0f, 3.0f};
    float value[2] = {4.0f, 9.0f};
    float cross_out[2] = {0.0f, 0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_cross_sdpa_f32(query, key, value, NULL, 0, cross_out,
                                     1, 1, 1, 1, 1, 2, 1.0f, 0, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(cross_out, sizeof(cross_out), 0) == 1);
    CHECK(fabsf(cross_out[0] - 4.0f) < 1e-6f);
    CHECK(fabsf(cross_out[1] - 9.0f) < 1e-6f);

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t key_mask[2] = {1, 0};
    float masked_out[2] = {0.0f, 0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_sdpa_f32(masked_qkv, key_mask, 2, masked_out,
                                2, 1, 1, 1, 1, 1.0f, 0, 1) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(masked_out, sizeof(masked_out), 0) == 1);
    CHECK(fabsf(masked_out[0] - 2.0f) < 1e-6f);
    CHECK(fabsf(masked_out[1] - 2.0f) < 1e-6f);

    float masked_cross_q[1] = {0.0f};
    float masked_cross_k[2] = {0.0f, 0.0f};
    float masked_cross_v[2] = {3.0f, 9.0f};
    int32_t cross_mask[2] = {0, 1};
    float masked_cross_out[1] = {0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_cross_sdpa_f32(masked_cross_q, masked_cross_k, masked_cross_v,
                                      cross_mask, 2, masked_cross_out,
                                      1, 2, 1, 1, 1, 1, 1.0f, 0, 1) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(masked_cross_out, sizeof(masked_cross_out), 0) == 1);
    CHECK(fabsf(masked_cross_out[0] - 9.0f) < 1e-6f);

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
    opengl_graph_begin_forward();
    CHECK(opengl_graph_sdpa_f32(parity_qkv, NULL, 0, parity_inference,
                                2, 2, 1, 2, 1, attention_scale, 0, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(parity_inference, sizeof(parity_inference), 0) == 1);
    opengl_graph_begin_forward();
    CHECK(opengl_graph_sdpa_training_f32(parity_qkv, NULL, 0, parity_training,
                                         2, 2, 1, 2, 1, attention_scale, 0, 0,
                                         0x80000000u, 0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(parity_training, sizeof(parity_training), 0) == 1);
    opengl_graph_begin_forward();
    CHECK(opengl_graph_cross_sdpa_training_f32(parity_q, parity_k, parity_v, NULL, 0,
                                               parity_cross_training, 2, 2, 2, 1, 2, 1,
                                               attention_scale, 0, 0, 0x80000000u,
                                               0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(parity_cross_training, sizeof(parity_cross_training), 0) == 1);
    for (int i = 0; i < 4; i++) {
        CHECK(fabsf(parity_inference[i] - expected_inference[i]) < 1.0e-4f);
        CHECK(fabsf(parity_training[i] - expected_training[i]) < 1.0e-4f);
        CHECK(fabsf(parity_cross_training[i] - expected_training[i]) < 1.0e-4f);
    }
    CHECK(fabsf(parity_inference[0] - parity_training[0]) > 1.0e-4f);

    int32_t tokens[2] = {1, 0};
    float embedding_weight[2] = {4.0f, 7.0f};
    float embedding_output[2] = {0.0f, 0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_embedding_f32(tokens, embedding_weight, embedding_output, 2, 1, 2) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(embedding_output, sizeof(embedding_output), 0) == 1);
    CHECK(fabsf(embedding_output[0] - 7.0f) < 1e-6f);
    CHECK(fabsf(embedding_output[1] - 4.0f) < 1e-6f);

    float pool_input[8] = {1.0f, 4.0f, 2.0f, 3.0f, -1.0f, -2.0f, 8.0f, 5.0f};
    float pool_output[2] = {0.0f, 0.0f};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_maxpool2d_f32(pool_input, pool_output, 2, 2, 2, 1,
                                     1, 1, 2, 2, 2, 2, 0, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(pool_output, sizeof(pool_output), 0) == 1);
    CHECK(fabsf(pool_output[0] - 4.0f) < 1e-6f);
    CHECK(fabsf(pool_output[1] - 8.0f) < 1e-6f);

    float conv_input[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float conv_weight[1] = {2.0f};
    float conv_bias[1] = {1.0f};
    float conv_output[6] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_conv1d_f32(conv_input, conv_weight, conv_bias, conv_output,
                                  2, 1, 3, 1, 3, 1, 1, 0, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(conv_output, sizeof(conv_output), 0) == 1);
    const float expected_conv[6] = {3.0f, 5.0f, 7.0f, 21.0f, 41.0f, 61.0f};
    for (int i = 0; i < 6; i++) CHECK(fabsf(conv_output[i] - expected_conv[i]) < 1e-6f);

    float projected_q[2] = {1.0f, -1.0f};
    float projected_kv[2] = {3.0f, 4.0f};
    float projected_weight[3] = {1.0f, 1.0f, 2.0f};
    float projected_output[2] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_cross_attention_f32(projected_q, projected_kv, projected_weight,
                                           NULL, NULL, projected_output,
                                           1, 1, 1, 1, 1, 2, 0, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(projected_output, sizeof(projected_output), 0) == 1);
    CHECK(fabsf(projected_output[0] - 6.0f) < 1e-6f);
    CHECK(fabsf(projected_output[1] - 8.0f) < 1e-6f);

    float profile_input[4] = {1.0f, 3.0f, 10.0f, 14.0f};
    float profile_output[4] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_profile_x_f32(profile_input, profile_output, 2, 2, 1, 1) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(profile_output, sizeof(profile_output), 0) == 1);
    const float expected_profile[4] = {3.0f, 2.0f, 14.0f, 12.0f};
    for (int i = 0; i < 4; i++) CHECK(fabsf(profile_output[i] - expected_profile[i]) < 1e-6f);

    float concat_a[2] = {1.0f, 10.0f};
    float concat_b[4] = {2.0f, 3.0f, 20.0f, 30.0f};
    const float* concat_inputs[2] = {concat_a, concat_b};
    long concat_sizes[2] = {2, 4};
    int concat_axes[2] = {1, 2};
    float concat_output[6] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                                  concat_output, 3, 1, 0) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    const float expected_concat[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    for (int i = 0; i < 6; i++) CHECK(fabsf(concat_output[i] - expected_concat[i]) < 1e-6f);
    opengl_graph_begin_forward();
    CHECK(opengl_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                                  concat_output, 3, 1, 1) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    for (int i = 0; i < 6; i++) {
        CHECK(fabsf(concat_output[i] - 1.0f / (1.0f + expf(-expected_concat[i]))) < 1e-6f);
    }
    opengl_graph_reset();
    return 0;
}

static int check_inference_training_alias(void) {
    float source[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float saved_activation[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_input[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CHECK(opengl_graph_copy_f32(source, saved_activation, 4) == 1);

    uint32_t params[4] = {4, 0, 0, 0};
    void* hosts[3] = {saved_activation, grad_input, params};
    size_t bytes[3] = {sizeof(saved_activation), sizeof(grad_input), sizeof(params)};
    unsigned char access[3] = {
        OPENGL_TRAINING_ACCESS_READ,
        OPENGL_TRAINING_ACCESS_READ_WRITE,
        OPENGL_TRAINING_ACCESS_READ,
    };
    unsigned char is_weight[3] = {0, 0, 0};
    CHECK(opengl_training_begin() == 0);
    CHECK(opengl_training_dispatch("copyBackward", "main", hosts, bytes, access,
                                   is_weight, 3, 1, 1, 1) == 0);
    CHECK(opengl_training_sync(grad_input, sizeof(grad_input)) == 0);
    for (int i = 0; i < 4; i++) CHECK(fabsf(grad_input[i] - source[i]) < 1e-6f);
    opengl_training_end();
    opengl_graph_reset();
    return 0;
}

static int check_general_gather_i32(void) {
    const float input[12] = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const int32_t indices[5] = {2, 0, -1, -4, 3};
    const float expected[20] = {
        5, 6, 1, 2, 5, 6, -1, -1, -1, -1,
        11, 12, 7, 8, 11, 12, -1, -1, -1, -1,
    };
    float output[20] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_gather_i32_f32(input, indices, output,
                                      2, 3, 2, 5, 20) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < 20; index++)
        CHECK(fabsf(output[index] - expected[index]) < 1e-6f);
    opengl_graph_reset();
    return 0;
}

typedef struct {
    VxEngineState* state;
    float base;
    int ok;
} OpenGLIsolationThread;

static void* run_opengl_isolation_thread(void* opaque) {
    OpenGLIsolationThread* probe = (OpenGLIsolationThread*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(probe->state);
    float input[4] = {0};
    float output[4] = {0};
    probe->ok = 1;
    for (int iteration = 0; iteration < 32; iteration++) {
        for (int i = 0; i < 4; i++) {
            input[i] = probe->base + (float)(iteration * 4 + i);
            output[i] = -1.0f;
        }
        opengl_graph_mark_host(input, sizeof(input), 0);
        opengl_graph_begin_forward();
        int copied = opengl_graph_copy_f32(input, output, 4);
        int ended = opengl_graph_end_forward();
        int synced = opengl_graph_sync_host(output, sizeof(output), 0);
        if (!copied || ended != 0 || !synced) {
            probe->ok = 0;
            break;
        }
        for (int i = 0; i < 4; i++) {
            if (fabsf(output[i] - input[i]) > 1.0e-6f) {
                probe->ok = 0;
                break;
            }
        }
        if (!probe->ok) break;
    }
    opengl_graph_reset();
    vx_engine_state_scope_leave(scope);
    return NULL;
}

static int check_engine_state_isolation(VxEngineState* first) {
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    CHECK(second != NULL);
    CHECK(vx_engine_state_init(second) == 0);

    float shared_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float shared_output[4] = {0};
    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_copy_f32(shared_input, shared_output, 4) == 1);
    CHECK(opengl_graph_end_forward() == 0);

    VxEngineStateScope second_scope = vx_engine_state_scope_enter(second);
    CHECK(opengl_init() == 0);
    shared_input[0] = 9.0f;
    shared_input[1] = 8.0f;
    shared_input[2] = 7.0f;
    shared_input[3] = 6.0f;
    opengl_graph_begin_forward();
    CHECK(opengl_graph_copy_f32(shared_input, shared_output, 4) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    int programs = 0, buffers = 0, inference_buffers = 0;
    opengl_training_debug_resource_counts(&programs, &buffers,
                                           &inference_buffers);
    CHECK(buffers == 0 && inference_buffers == 2);
    vx_engine_state_scope_leave(second_scope);

    CHECK(opengl_graph_sync_host(shared_output, sizeof(shared_output), 0) == 1);
    const float first_expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    CHECK(memcmp(shared_output, first_expected, sizeof(shared_output)) == 0);
    opengl_graph_reset();
    opengl_training_debug_resource_counts(&programs, &buffers,
                                           &inference_buffers);
    CHECK(buffers == 0 && inference_buffers == 0);

    second_scope = vx_engine_state_scope_enter(second);
    CHECK(opengl_graph_sync_host(shared_output, sizeof(shared_output), 0) == 1);
    const float second_expected[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    CHECK(memcmp(shared_output, second_expected, sizeof(shared_output)) == 0);
    opengl_training_debug_resource_counts(&programs, &buffers,
                                           &inference_buffers);
    CHECK(buffers == 0 && inference_buffers == 2);
    opengl_graph_reset();
    vx_engine_state_scope_leave(second_scope);

    OpenGLIsolationThread first_probe = {first, 1000.0f, 0};
    OpenGLIsolationThread second_probe = {second, -1000.0f, 0};
    pthread_t first_thread;
    pthread_t second_thread;
    CHECK(pthread_create(&first_thread, NULL, run_opengl_isolation_thread,
                         &first_probe) == 0);
    CHECK(pthread_create(&second_thread, NULL, run_opengl_isolation_thread,
                         &second_probe) == 0);
    CHECK(pthread_join(first_thread, NULL) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    CHECK(first_probe.ok && second_probe.ok);

    second_scope = vx_engine_state_scope_enter(second);
    opengl_cleanup();
    CHECK(opengl_training_available() == 0);
    vx_engine_state_scope_leave(second_scope);
    vx_engine_state_deinit(second);
    free(second);

    CHECK(opengl_training_available() == 1);
    float survivor_input[2] = {4.0f, 5.0f};
    float survivor_output[2] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_copy_f32(survivor_input, survivor_output, 2) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(survivor_output, sizeof(survivor_output), 0) == 1);
    CHECK(memcmp(survivor_input, survivor_output, sizeof(survivor_input)) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_typed_control_graph_ops(void) {
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

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_compare_i32(
        compare_a, 2, compare_b, 3, equal_output, 6,
        output_strides, a_strides, b_strides, 2, 0) == 1);
    CHECK(opengl_graph_compare_i32(
        compare_a, 2, compare_b, 3, ge_output, 6,
        output_strides, a_strides, b_strides, 2, 1) == 1);
    CHECK(opengl_graph_not_i32(equal_output, not_output, 6) == 1);
    CHECK(opengl_graph_clip_i32(
        clip_input, clip_output, 4, 0, 7) == 1);
    CHECK(opengl_graph_cast_typed(
        cast_i32, VX_DTYPE_I32, cast_f32, VX_DTYPE_F32, 4) == 1);
    CHECK(opengl_graph_cast_typed(
        cast_float_input, VX_DTYPE_F32,
        cast_i32_output, VX_DTYPE_I32, 13) == 1);
    CHECK(opengl_graph_copy_32(copy_input, copy_output, 4) == 1);
    CHECK(opengl_graph_where_32(
        where_condition, where_a, where_b, where_output, 4) == 1);
    CHECK(opengl_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 3u, 2u) == 1);
    CHECK(opengl_graph_concat_32(
        concat_inputs, concat_sizes, concat_axes, 2,
        concat_output, 3, 2) == 1);
    CHECK(opengl_graph_end_forward() == 0);

    CHECK(opengl_graph_sync_host(
        equal_output, sizeof(equal_output), 0) == 1);
    CHECK(opengl_graph_sync_host(ge_output, sizeof(ge_output), 0) == 1);
    CHECK(opengl_graph_sync_host(not_output, sizeof(not_output), 0) == 1);
    CHECK(opengl_graph_sync_host(clip_output, sizeof(clip_output), 0) == 1);
    CHECK(opengl_graph_sync_host(cast_f32, sizeof(cast_f32), 0) == 1);
    CHECK(opengl_graph_sync_host(
        cast_i32_output, sizeof(cast_i32_output), 0) == 1);
    CHECK(opengl_graph_sync_host(copy_output, sizeof(copy_output), 0) == 1);
    CHECK(opengl_graph_sync_host(
        where_output, sizeof(where_output), 0) == 1);
    CHECK(opengl_graph_sync_host(
        argmax_output, sizeof(argmax_output), 0) == 1);
    CHECK(opengl_graph_sync_host(
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
    CHECK(memcmp(
        where_output, expected_where, sizeof(where_output)) == 0);
    CHECK(memcmp(
        argmax_output, expected_argmax, sizeof(argmax_output)) == 0);
    CHECK(memcmp(
        concat_output, expected_concat, sizeof(concat_output)) == 0);

    CHECK(opengl_graph_not_i32(equal_output, equal_output, 6) == 0);
    CHECK(opengl_graph_clip_i32(
        clip_input, clip_output, 4, 8, 7) == 0);
    CHECK(opengl_graph_cast_typed(
        cast_i32, VX_DTYPE_I32,
        cast_i32_output, VX_DTYPE_U8, 4) == 0);
    CHECK(opengl_graph_argmax_f32(
        argmax_input, argmax_output, 2u, 0u, 2u) == 0);
    opengl_graph_reset();
    return 0;
}

static int check_expand_f32_uniform_abi(void) {
    union {
        uint32_t bits[4];
        float values[4];
    } input = {
        .bits = {0xff800000u, 0x7fc12345u, 0x80000000u, 0x3f800000u},
    };
    union {
        uint32_t bits[48];
        float values[48];
    } output;
    const int input_shape[5] = {1, 2, 1, 1, 2};
    const int output_shape[6] = {3, 1, 2, 1, 4, 2};
    const int incompatible_output_shape[6] = {3, 1, 3, 1, 4, 2};
    const int smaller_output_shape[4] = {1, 2, 1, 2};
    const int invalid_input_shape[5] = {1, 2, 0, 1, 2};
    const int scalar_shape[1] = {1};
    const int overflow_shape[2] = {INT32_MAX, 2};
    float overlap_storage[49] = {0};
    uint32_t expected[48];
    size_t index = 0;

    for (size_t leading = 0; leading < 3; leading++) {
        for (size_t row = 0; row < 2; row++) {
            for (size_t broadcast = 0; broadcast < 4; broadcast++) {
                for (size_t lane = 0; lane < 2; lane++) {
                    expected[index++] = input.bits[row * 2 + lane];
                }
            }
        }
    }
    for (index = 0; index < 48; index++) output.bits[index] = 0xdeadbeefu;

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_expand_f32(
        input.values, output.values, input_shape, 5, output_shape, 6) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(
        output.values, sizeof(output.values), 0) == 1);
    CHECK(memcmp(output.bits, expected, sizeof(expected)) == 0);
    opengl_graph_reset();

    CHECK(opengl_graph_expand_f32(
        input.values, output.values, input_shape, 5,
        incompatible_output_shape, 6) == 0);
    CHECK(opengl_graph_expand_f32(
        input.values, output.values, input_shape, 5,
        smaller_output_shape, 4) == 0);
    CHECK(opengl_graph_expand_f32(
        input.values, output.values, invalid_input_shape, 5,
        output_shape, 6) == 0);
    CHECK(opengl_graph_expand_f32(
        input.values, output.values, scalar_shape, 1,
        overflow_shape, 2) == 0);
    CHECK(opengl_graph_expand_f32(
        overlap_storage, overlap_storage, input_shape, 5,
        output_shape, 6) == 0);
    CHECK(opengl_graph_expand_f32(
        overlap_storage, overlap_storage + 1, input_shape, 5,
        output_shape, 6) == 0);
    return 0;
}

static int check_qbatch_and_typed_transpose(void) {
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

    opengl_graph_reset();
    opengl_graph_begin_forward();
    CHECK(opengl_graph_qbatch_matmul_i8u8(
        a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
        b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
        qbatch_output, invalid_output_shape, 4,
        0.25f, 3, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_duplicate, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(opengl_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.25f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(opengl_graph_transpose_i8u8(
        transpose_u8_output, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 0);
    CHECK(opengl_graph_qbatch_matmul_i8u8(
        a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
        b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
        qbatch_output, output_shape, 4,
        0.25f, 3, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_transpose_i8u8(
        transpose_u8_input, transpose_u8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.125f, 123,
        0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(opengl_graph_transpose_i8u8(
        transpose_i8_input, transpose_i8_output, transpose_shape,
        transpose_permutation, 4u, 12u, 0.25f, -3,
        0.25f, -3, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(opengl_graph_end_forward() == 0);

    CHECK(opengl_graph_sync_host(
        qbatch_output, sizeof(qbatch_output), 0) == 1);
    CHECK(opengl_graph_sync_host(
        transpose_u8_output, sizeof(transpose_u8_output), 0) == 1);
    CHECK(opengl_graph_sync_host(
        transpose_i8_output, sizeof(transpose_i8_output), 0) == 1);
    CHECK(memcmp(
        qbatch_output, expected_qbatch, sizeof(qbatch_output)) == 0);
    CHECK(memcmp(
        transpose_u8_output, transpose_u8_expected,
        sizeof(transpose_u8_output)) == 0);
    CHECK(memcmp(
        transpose_i8_output, transpose_i8_expected,
        sizeof(transpose_i8_output)) == 0);
    opengl_graph_reset();
    return 0;
}

int main(void) {
    VxEngineState* engine_state =
        (VxEngineState*)calloc(1, sizeof(*engine_state));
    CHECK(engine_state != NULL);
    CHECK(vx_engine_state_init(engine_state) == 0);
    VxEngineStateScope engine_scope =
        vx_engine_state_scope_enter(engine_state);
    CHECK(check_capability_rules() == 0);
    OpenGLComputeCapability capability;
    CHECK(opengl_get_compute_capability(&capability) == 0);
    CHECK(capability.api == OPENGL_COMPUTE_API_NONE && !capability.compute_supported);
    CHECK(opengl_training_available() == 0);
    CHECK(opengl_training_begin() == -1);
    CHECK(opengl_training_sync((void*)&capability, sizeof(capability)) == -1);
    int8_t unavailable_a[1] = {0};
    uint8_t unavailable_b[1] = {0};
    int8_t unavailable_sum[1] = {0};
    uint8_t unavailable_output[1] = {0};
    float unavailable_f32[1] = {0.0f};
    const float unavailable_gamma[1] = {1.0f};
    const float unavailable_beta[1] = {0.0f};
    const void* unavailable_inputs[1] = {unavailable_a};
    const uint32_t unavailable_elements[1] = {1u};
    const uint32_t unavailable_axes[1] = {1u};
    const float unavailable_scales[1] = {1.0f};
    const int32_t unavailable_zero_points_typed[1] = {0};
    const uint32_t unavailable_dtypes[1] = {VX_DTYPE_I8};
    CHECK(opengl_graph_qadd_i8u8(unavailable_a, 1u, unavailable_b, 1u,
                                  unavailable_sum, 1u,
                                  1.0f, 0, 1.0f, 0, 1.0f, 0,
                                  VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_I8,
                                  0u) == 0);
    CHECK(opengl_graph_qsilu_i8u8(unavailable_a, unavailable_sum, 1u,
                                   1.0f, 0, 1.0f, 0,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qgelu_i8u8(unavailable_a, unavailable_sum, 1u,
                                   1.0f, 0, 1.0f, 0,
                                   VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qgroupnorm_i8u8(unavailable_a, unavailable_gamma,
                                       unavailable_beta, unavailable_sum,
                                       1u, 1u, 1u, 1u, 1u, 1.0f, 0,
                                       1.0f, 0, 1.0e-5f,
                                       VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_qlayernorm_i8u8(unavailable_a, unavailable_gamma,
                                       unavailable_beta, unavailable_sum,
                                       1u, 1u, 1.0f, 0, 1.0f, 0,
                                       1.0e-5f, VX_DTYPE_I8,
                                       VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_requantize_linear_i8u8(unavailable_sum, 1u,
                                              unavailable_output, 1u,
                                              1.0f, 0, 1.0f, 0,
                                              VX_DTYPE_I8, VX_DTYPE_U8) == 0);
    const int8_t unavailable_weight[1] = {1};
    const float unavailable_scale[1] = {1.0f};
    const int32_t unavailable_zero_point[1] = {0};
    CHECK(opengl_graph_qconv2d_i8u8(unavailable_a, unavailable_weight,
                                     unavailable_scale, unavailable_zero_point, NULL,
                                     unavailable_sum,
                                     1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                                     1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 0u,
                                     1.0f, 0, 1.0f, 0, VX_DTYPE_I8,
                                     VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_quantize_typed_f32_i8u8(unavailable_f32, 1u,
                                                unavailable_a, 1.0f, 0,
                                                VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_dequantize_typed_i8u8_f32(unavailable_a, 1u,
                                                  1.0f, 0, VX_DTYPE_I8,
                                                  unavailable_f32) == 0);
    CHECK(opengl_graph_copy_i8u8(unavailable_a, 1u, unavailable_sum, 1u,
                                 1.0f, 0, 1.0f, 0,
                                 VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_concat_i8u8(unavailable_inputs, unavailable_elements,
                                   unavailable_axes, unavailable_scales,
                                   unavailable_zero_points_typed, unavailable_dtypes,
                                   1u, unavailable_sum, 1u, 1u, 1u,
                                   1.0f, 0, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_maxpool2d_i8u8(unavailable_a, unavailable_sum,
                                      1u, 1u, 1u, 1u, 1u, 1u,
                                      1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u,
                                      1.0f, 0, 1.0f, 0,
                                      VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(opengl_graph_resize_nearest_i8u8(unavailable_a, unavailable_sum,
                                           1u, 1u, 1u, 1u, 1u, 1u,
                                           1.0f, 0, 1.0f, 0,
                                           VX_DTYPE_I8, VX_DTYPE_I8) == 0);

    if (opengl_init() != 0) {
        /* A machine without a compute context must reject cleanly for CPU fallback. */
        CHECK(opengl_training_available() == 0);
        opengl_cleanup();
        vx_engine_state_scope_leave(engine_scope);
        vx_engine_state_deinit(engine_state);
        free(engine_state);
        puts("OpenGL compute context unavailable; rejection/fallback checks passed");
        return 0;
    }

    CHECK(opengl_get_compute_capability(&capability) == 0);
    CHECK(capability.compute_supported);
    CHECK(opengl_compute_version_supported(capability.api, capability.major, capability.minor));
    CHECK(opengl_training_available() == 1);
    float host_current = 1.0f;
    opengl_graph_reset();
    CHECK(opengl_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    opengl_graph_mark_host(&host_current, sizeof(host_current), 1);
    CHECK(opengl_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    opengl_graph_reset();
    size_t copy_bytes[3] = {sizeof(float), sizeof(float), sizeof(uint32_t)};
    CHECK(opengl_training_supports("copyBackward", "main", copy_bytes, 3, 1, 1, 1) == 1);
    CHECK(opengl_training_supports("copyBackward", "main", copy_bytes, 2, 1, 1, 1) == 0);
    CHECK(opengl_training_supports("copyBackward", "main", copy_bytes, 3,
                                   UINT32_MAX, 1, 1) == 0);
    size_t oversized_bytes[3] = {SIZE_MAX, sizeof(float), sizeof(uint32_t)};
    CHECK(opengl_training_supports("copyBackward", "main", oversized_bytes, 3,
                                   1, 1, 1) == 0);
    CHECK(opengl_training_supports("notARealShader", "main", copy_bytes, 3,
                                   1, 1, 1) == 0);
    CHECK(check_one_shot_matmul_tails_and_fallback() == 0);
    CHECK(check_tiled_qlinear_i8u8_tails() == 0);
    CHECK(check_tiled_qlinear_staged_rounding() == 0);
    CHECK(check_lazy_training_dispatch() == 0);
    CHECK(check_prelu_logsoftmax_split_backward() == 0);
    CHECK(check_groupnorm_dropout_reduce_and_broadcast() == 0);
    CHECK(check_gelu_modes() == 0);
    CHECK(check_layernorm_epsilon() == 0);
    CHECK(check_qlinear_i8u8_packed_chain() == 0);
    CHECK(check_qembedding_i8u8_packed_gather() == 0);
    CHECK(check_qadd_requantize_i8u8_packed_chain() == 0);
    CHECK(check_qsilu_i8u8_packed_chain() == 0);
    CHECK(check_qgelu_i8u8_packed_chain() == 0);
    CHECK(check_qgroupnorm_i8u8_packed_chain() == 0);
    CHECK(check_qlayernorm_i8u8_packed_chain() == 0);
    CHECK(check_qsdpa_i8u8_packed_chain() == 0);
    CHECK(check_qargmax_i8u8_raw() == 0);
    CHECK(check_qmaskedmean_i8u8_packed() == 0);
    CHECK(check_qconv2d_i8u8_packed_chain() == 0);
    CHECK(check_typed_i8u8_shape_qdq_chain() == 0);
    CHECK(check_general_gather_i32() == 0);
    CHECK(check_typed_control_graph_ops() == 0);
    CHECK(check_expand_f32_uniform_abi() == 0);
    CHECK(check_qbatch_and_typed_transpose() == 0);
    CHECK(opengl_training_debug_compile_all() == 0);
    int programs = 0, buffers = -1, inference_buffers = -1;
    opengl_training_debug_resource_counts(&programs, &buffers, &inference_buffers);
    CHECK(programs > 1 && buffers == 0 && inference_buffers == 0);
    CHECK(check_inference_training_alias() == 0);
    CHECK(check_batched_attention() == 0);
    CHECK(check_engine_state_isolation(engine_state) == 0);
    opengl_cleanup();
    CHECK(opengl_training_available() == 0);
    CHECK(opengl_init() == 0);
    float reinitialized_input[2] = {12.0f, -3.0f};
    float reinitialized_output[2] = {0};
    opengl_graph_begin_forward();
    CHECK(opengl_graph_copy_f32(reinitialized_input, reinitialized_output, 2) == 1);
    CHECK(opengl_graph_end_forward() == 0);
    CHECK(opengl_graph_sync_host(reinitialized_output,
                                 sizeof(reinitialized_output), 0) == 1);
    CHECK(memcmp(reinitialized_input, reinitialized_output,
                 sizeof(reinitialized_input)) == 0);
    opengl_cleanup();
    CHECK(opengl_training_available() == 0);
    vx_engine_state_scope_leave(engine_scope);
    vx_engine_state_deinit(engine_state);
    free(engine_state);
    puts("OpenGL lazy compute training checks passed");
    return 0;
}
