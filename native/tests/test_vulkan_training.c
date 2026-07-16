#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vulkan_engine.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "vulkan training check failed at %s:%d: %s\n", \
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

static int check_one_shot_matmul_case(int rows, int d_in, int d_out,
                                      float* input, float* weight, float* bias,
                                      float* output) {
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < d_out; column++) {
            float expected = bias[column];
            for (int k = 0; k < d_in; k++) {
                expected += input[row * d_in + k] * weight[k * d_out + column];
            }
            output[row * d_out + column] = expected;
        }
    }
    float expected[17 * 23];
    memcpy(expected, output, (size_t)rows * d_out * sizeof(float));
    memset(output, 0, (size_t)rows * d_out * sizeof(float));
    CHECK(vk_matmul(input, weight, bias, output, rows, d_in, d_out) == 1);
    for (int i = 0; i < rows * d_out; i++) CHECK(close_enough(output[i], expected[i]));
    return 0;
}

static int test_one_shot_matmul_tails_and_fallback(void) {
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
    vk_free_weight_cache();
    CHECK(check_one_shot_matmul_case(17, 19, 23, tiled_input, tiled_weight,
                                     tiled_bias, tiled_output) == 0);
    CHECK(check_one_shot_matmul_case(3, 7, 5, scalar_input, scalar_weight,
                                     scalar_bias, scalar_output) == 0);

    /* A host allocation may be reused with a different logical matrix shape.
       The resident transpose cache must include the shape, not just its address. */
    float reshaped_weight[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float first_input[2] = {2.0f, -1.0f};
    float first_bias[3] = {0.25f, -0.5f, 1.0f};
    float first_output[3];
    float second_input[3] = {1.0f, 2.0f, 3.0f};
    float second_bias[2] = {-1.0f, 0.5f};
    float second_output[2];
    CHECK(check_one_shot_matmul_case(1, 2, 3, first_input, reshaped_weight,
                                     first_bias, first_output) == 0);
    CHECK(check_one_shot_matmul_case(1, 3, 2, second_input, reshaped_weight,
                                     second_bias, second_output) == 0);

    /* 4097x4096 F32 is one page larger than the 64 MiB resident-weight arena.
       It must decline before reading or writing through these one-float pointers. */
    float sentinel = 0.0f;
    CHECK(vk_matmul(&sentinel, &sentinel, NULL, &sentinel, 1, 4097, 4096) == 0);
    vk_free_weight_cache();
    return 0;
}

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
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(input, weight, scales, zero_points, bias, output,
                                ROWS, D_IN, D_OUT, 1.0f, 0, 1.0f, 0,
                                2u, 2u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vk_graph_reset();
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

static uint32_t dropout_bits(uint32_t seed, uint32_t counter, uint32_t index) {
    uint32_t value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
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
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    unsigned char weights[5] = {0, 0, 0, 0, 0};
    CHECK(vk_training_dispatch("activationBackward", "main", hosts, bytes,
                               access, weights, 5, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(close_enough(grad_input[0], 0.0f));
    CHECK(close_enough(grad_input[1], 4.0f));
    return 0;
}

static int test_gelu_backward_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7], tanh_output[7], grad_output[7];
    float exact_grad[7] = {0}, tanh_grad[7] = {0};
    for (int i = 0; i < 7; i++) {
        exact_output[i] = gelu_reference(input[i], 0);
        tanh_output[i] = gelu_reference(input[i], 1);
        grad_output[i] = 0.25f * (float)(i + 1);
    }
    uint32_t exact_params[4] = {7u, 1u, 0u, 0u};
    uint32_t tanh_params[4] = {7u, 9u, 0u, 0u};
    void* exact_hosts[5] = {input, exact_output, grad_output, exact_grad, exact_params};
    void* tanh_hosts[5] = {input, tanh_output, grad_output, tanh_grad, tanh_params};
    size_t exact_bytes[5] = {sizeof(input), sizeof(exact_output), sizeof(grad_output),
                             sizeof(exact_grad), sizeof(exact_params)};
    size_t tanh_bytes[5] = {sizeof(input), sizeof(tanh_output), sizeof(grad_output),
                            sizeof(tanh_grad), sizeof(tanh_params)};
    unsigned char access[5] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("activationBackward", "main", exact_hosts, exact_bytes,
                               access, NULL, 5, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("activationBackward", "main", tanh_hosts, tanh_bytes,
                               access, NULL, 5, 1, 1, 1) == 0);
    CHECK(vk_training_sync(exact_grad, sizeof(exact_grad)) == 0);
    CHECK(vk_training_sync(tanh_grad, sizeof(tanh_grad)) == 0);
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_grad[i] - grad_output[i] * gelu_derivative_reference(input[i], 1)) < 2.0e-4f);
    }
    return 0;
}

static int test_gelu_forward_modes(void) {
    float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f, 0.5f, 2.0f, 4.0f};
    float exact_output[7] = {0}, tanh_output[7] = {0};
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_gelu_f32(input, exact_output, 7, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(exact_output, sizeof(exact_output), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_gelu_f32(input, tanh_output, 7, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(tanh_output, sizeof(tanh_output), 0) == 1);
    for (int i = 0; i < 7; i++) {
        CHECK(fabsf(exact_output[i] - gelu_reference(input[i], 0)) < 2.0e-4f);
        CHECK(fabsf(tanh_output[i] - gelu_reference(input[i], 1)) < 2.0e-4f);
    }
    CHECK(fabsf(exact_output[0] - tanh_output[0]) > 1.0e-5f);
    vk_graph_reset();
    return 0;
}

static int test_qlinear_i8u8_packed_chain(void) {
    /* Every tensor here has a non-word-aligned logical byte length.  The
       second dispatch consumes `hidden` while it is still device-resident. */
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

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlinear_i8u8(input, first_weight, first_scales,
                                first_zero_points, first_bias, hidden,
                                1u, 3u, 2u, 0.5f, 0, 0.25f, 128,
                                2u, 2u, 3u) == 1);
    CHECK(vk_graph_qlinear_i8u8(hidden, second_weight, second_scales,
                                second_zero_points, second_bias, output,
                                1u, 2u, 1u, 0.25f, 128, 0.25f, -3,
                                3u, 2u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    CHECK(hidden[0] == 125u && hidden[1] == 134u);
    CHECK(output[0] == -12);
    vk_graph_reset();
    return 0;
}

static int test_qembedding_i8u8_packed_gather(void) {
    /* Nine logical output bytes exercise the packed-word tail while both
       table dtypes and output dtypes use the canonical row descriptor ABI. */
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

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qembedding_i8u8(ids, i8_table, scales, i8_zero_points, output_u8,
                                   3u, 3u, 3u, 0.25f, 128, 2u, 3u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == expected_u8[index]);

    vk_graph_begin_forward();
    CHECK(vk_graph_qembedding_i8u8(ids, u8_table, scales, u8_zero_points, output_i8,
                                   3u, 3u, 3u, 0.25f, -3, 3u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    for (int index = 0; index < 9; index++) CHECK(output_i8[index] == expected_i8[index]);

    memset(output_u8, 0x5a, sizeof(output_u8));
    CHECK(vk_graph_qembedding_i8u8(invalid_ids, i8_table, scales, i8_zero_points,
                                   output_u8, 3u, 3u, 3u, 0.25f, 128, 2u, 3u) == 0);
    for (int index = 0; index < 9; index++) CHECK(output_u8[index] == 0x5a);
    vk_graph_reset();
    return 0;
}

static int test_qadd_requantize_i8u8_packed_chain(void) {
    /* Five logical bytes force a padded second u32 word.  `sum` remains on
       the device while RequantizeLinear changes its U8 output domain. */
    const int8_t a[5] = {-2, 1, 13, 0, 30};
    const uint8_t b[5] = {124, 128, 140, 129, 128};
    int8_t sum[5] = {0, 0, 0, 0, 0};
    uint8_t output[5] = {0, 0, 0, 0, 0};

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, 2u, 3u, 2u, 2u) == 1);
    CHECK(vk_graph_requantize_linear_i8u8(sum, 5u, output, 5u,
                                          0.5f, -2, 1.0f, 100,
                                          2u, 3u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(sum, sizeof(sum), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    const int8_t expected_sum[5] = {-2, -1, 10, -2, 10};
    const uint8_t expected_output[5] = {100, 100, 106, 100, 106};
    for (int i = 0; i < 5; i++) {
        CHECK(sum[i] == expected_sum[i]);
        CHECK(output[i] == expected_output[i]);
    }
    /* Exact-shape and relu ABI validation must fail before a dispatch. */
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 4u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, 2u, 3u, 2u, 2u) == 0);
    CHECK(vk_graph_qadd_i8u8(a, 5u, b, 5u, sum, 5u,
                              0.5f, 0, 0.25f, 128,
                              0.5f, -2, 2u, 3u, 2u, 3u) == 0);
    CHECK(vk_graph_requantize_linear_i8u8(sum, 5u, output, 4u,
                                          0.5f, -2, 1.0f, 100,
                                          2u, 3u) == 0);
    vk_graph_reset();
    return 0;
}

/* Five elements require a padded packed-u32 tail. The four direct dispatches
 * cover every I8/U8 boundary pair, and the final pair proves QSiLU can consume
 * its predecessor while both byte tensors remain device-resident. */
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

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_i8, 5u,
                               0.5f, 0, 0.25f, -3, 2u, 2u) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_u8, 5u,
                               0.5f, 0, 0.25f, 128, 2u, 3u) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_u8, from_u8_i8, 5u,
                               0.5f, 128, 0.25f, -3, 3u, 2u) == 1);
    CHECK(vk_graph_qsilu_i8u8(input_u8, from_u8_u8, 5u,
                               0.5f, 128, 0.25f, 128, 3u, 3u) == 1);
    CHECK(vk_graph_qsilu_i8u8(from_i8_u8, chained, 5u,
                               0.25f, 128, 0.125f, -4, 3u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -5, -4, 0, 27};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(vk_graph_qsilu_i8u8(input_i8, from_i8_i8, 0u,
                               0.5f, 0, 0.25f, -3, 2u, 2u) == 0);
    CHECK(vk_graph_qsilu_i8u8(alias, alias, 5u,
                               0.5f, 0, 0.25f, -3, 2u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

/* QGELU uses the fixed Abramowitz-Stegun erf route.  Keep the same
 * non-word-aligned shape and all I8/U8 boundaries as QSiLU, then consume the
 * U8 result while it is still packed in Vulkan graph storage. */
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

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_i8, 5u,
                               0.5f, 0, 0.125f, -3, 2u, 2u) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_u8, 5u,
                               0.5f, 0, 0.125f, 128, 2u, 3u) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_u8, from_u8_i8, 5u,
                               0.5f, 128, 0.125f, -3, 3u, 2u) == 1);
    CHECK(vk_graph_qgelu_i8u8(input_u8, from_u8_u8, 5u,
                               0.5f, 128, 0.125f, 128, 3u, 3u) == 1);
    CHECK(vk_graph_qgelu_i8u8(from_i8_u8, chained, 5u,
                               0.125f, 128, 0.125f, -4, 3u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(memcmp(from_i8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_i8_u8, expected_u8, sizeof(expected_u8)) == 0);
    CHECK(memcmp(from_u8_i8, expected_i8, sizeof(expected_i8)) == 0);
    CHECK(memcmp(from_u8_u8, expected_u8, sizeof(expected_u8)) == 0);
    {
        const int8_t expected_chained[5] = {-4, -4, -4, 2, 28};
        CHECK(memcmp(chained, expected_chained, sizeof(expected_chained)) == 0);
    }
    CHECK(vk_graph_qgelu_i8u8(input_i8, from_i8_i8, 0u,
                               0.5f, 0, 0.125f, -3, 2u, 2u) == 0);
    CHECK(vk_graph_qgelu_i8u8(alias, alias, 5u,
                               0.5f, 0, 0.125f, -3, 2u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

/* QGroupNorm's stats/apply split must preserve raw bytes across every typed
 * boundary, including C=3 tails and C/G=3 boundaries inside packed words. */
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
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f, 2u, 2u) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, 128, 1.0e-5f, 2u, 3u) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                    0.125f, -3, 1.0e-5f, 3u, 2u) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, 127,
                                    0.125f, 128, 1.0e-5f, 3u, 3u) == 1);
    /* The second pass consumes packed U8 output without a host sync. */
    CHECK(vk_graph_qgroupnorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                    1u, 1u, 1u, 3u, 1u, 0.125f, 128,
                                    0.125f, -4, 1.0e-5f, 3u, 2u) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(boundary_input, boundary_gamma, boundary_beta,
                                    boundary_output, 3u, 1u, 1u, 6u, 2u,
                                    0.25f, -1, 0.125f, -3, 1.0e-5f, 2u, 2u) == 1);
    CHECK(vk_graph_qgroupnorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                    3u, 57u, 1u, 6u, 2u, 0.5f, 17,
                                    0.25f, 128, 1.0e-5f, 3u, 3u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(vk_graph_sync_host(boundary_output, sizeof(boundary_output), 0) == 1);
    CHECK(vk_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
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
    CHECK(vk_graph_qgroupnorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    1u, 1u, 1u, 3u, 0u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f, 2u, 2u) == 0);
    CHECK(vk_graph_qgroupnorm_i8u8(alias, gamma, beta, alias,
                                    1u, 1u, 1u, 3u, 1u, 0.5f, -1,
                                    0.125f, -3, 1.0e-5f, 2u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

/* D=3 deliberately splits two final-axis rows across a packed word. The
 * stats/apply split must still give every row its own centered byte variance. */
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
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    2u, 3u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, 2u, 2u) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_u8,
                                    2u, 3u, 0.5f, -1, 0.125f, 128,
                                    1.0e-5f, 2u, 3u) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_i8,
                                    2u, 3u, 0.5f, 127, 0.125f, -3,
                                    1.0e-5f, 3u, 2u) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(input_u8, gamma, beta, from_u8_u8,
                                    2u, 3u, 0.5f, 127, 0.125f, 128,
                                    1.0e-5f, 3u, 3u) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(from_i8_u8, zero_gamma, beta, chained,
                                    2u, 3u, 0.125f, 128, 0.125f, -4,
                                    1.0e-5f, 3u, 2u) == 1);
    CHECK(vk_graph_qlayernorm_i8u8(high_input, high_gamma, high_beta, high_output,
                                    2u, high_d_model, 0.5f, 17, 0.25f, 128,
                                    1.0e-5f, 3u, 3u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(from_i8_i8, sizeof(from_i8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_i8_u8, sizeof(from_i8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_i8, sizeof(from_u8_i8), 0) == 1);
    CHECK(vk_graph_sync_host(from_u8_u8, sizeof(from_u8_u8), 0) == 1);
    CHECK(vk_graph_sync_host(chained, sizeof(chained), 0) == 1);
    CHECK(vk_graph_sync_host(high_output, sizeof(high_output), 0) == 1);
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
    CHECK(vk_graph_qlayernorm_i8u8(input_i8, gamma, beta, from_i8_i8,
                                    2u, 0u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, 2u, 2u) == 0);
    CHECK(vk_graph_qlayernorm_i8u8(alias, gamma, beta, alias,
                                    2u, 3u, 0.5f, -1, 0.125f, -3,
                                    1.0e-5f, 2u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

/* qSDPAInt8 has one workgroup per [query, head, batch]. This verifies an
 * I8->U8->I8 device-resident chain, ordinary/no-mask dummy binding, and the
 * all-masked U8 zero-point result without allocating a score buffer. */
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
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, first, 1u, 2u, 2u,
                               4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                               0.25f, 128, 0.5f, 2u, 2u, 2u, 3u, 0u, 0u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(first, k_i8, v_i8, NULL, second, 1u, 2u, 2u,
                               4u, 1u, 0.25f, 128, 0.25f, -1, 0.25f, -1,
                               0.25f, 0, 0.5f, 3u, 2u, 2u, 2u, 0u, 0u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(q_u8, k_u8, v_u8, mask_none, all_masked, 1u,
                               1u, 2u, 4u, 1u, 0.25f, 128, 0.5f, 120,
                               0.25f, 130, 0.25f, 127, 0.5f, 3u, 3u, 3u,
                               3u, 0u, 1u) == 1);
    CHECK(vk_graph_qsdpa_i8u8(q64, k64, v64, NULL, out64, 1u, 1u, 1u,
                               head_dim, 1u, 0.25f, 0, 0.25f, 0, 0.25f, 0,
                               0.25f, 0, 1.0f, 2u, 2u, 2u, 2u, 0u, 0u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(first, sizeof(first), 0) == 1);
    CHECK(vk_graph_sync_host(second, sizeof(second), 0) == 1);
    CHECK(vk_graph_sync_host(all_masked, sizeof(all_masked), 0) == 1);
    CHECK(vk_graph_sync_host(out64, sizeof(out64), 0) == 1);
    CHECK(memcmp(first, expected_first, sizeof(first)) == 0);
    CHECK(memcmp(second, expected_second, sizeof(second)) == 0);
    CHECK(memcmp(all_masked, expected_zero, sizeof(all_masked)) == 0);
    CHECK(memcmp(out64, v64, sizeof(out64)) == 0);
    CHECK(vk_graph_qsdpa_i8u8(alias, k_i8, v_i8, NULL, alias, 1u, 2u, 2u,
                               4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                               0.25f, 0, 0.5f, 2u, 2u, 2u, 2u, 0u, 0u) == 0);
    CHECK(memcmp(alias, alias_before, sizeof(alias)) == 0);
    vk_graph_reset();
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
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u, 2u) == 1);
    CHECK(vk_graph_qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u, 3u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(vk_graph_qargmax_i8u8(alias.bytes, alias.i32, 1u, 3u, 2u, 2u) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(vk_graph_qargmax_i8u8(input_i8, sentinel, 1u, 0u, 2u, 2u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    vk_graph_reset();
    return 0;
}

/* qMaskedMeanInt8 owns packed output words (including the U8 tail here),
 * averages centered raw bytes only for nonzero I32 mask entries, and keeps
 * the all-masked row at the declared output zero point. */
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
    int8_t staged_input[127];
    int32_t staged_mask[127];
    int8_t staged_output = 0;
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
    for (uint32_t index = 0; index < 127u; index++) {
        staged_input[index] = -128;
        staged_mask[index] = 1;
    }
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qmaskedmean_i8u8(input_i8, mask_i8, output_i8,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     2u, 2u) == 1);
    CHECK(vk_graph_qmaskedmean_i8u8(input_u8, mask_u8, output_u8,
                                     1u, 3u, 2u, 0.25f, 128, 0.25f, 130,
                                     3u, 3u) == 1);
    CHECK(vk_graph_qmaskedmean_i8u8(staged_input, staged_mask, &staged_output,
                                     1u, 127u, 1u, 0.001f, 127, 0.01f, -37,
                                     2u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output_i8, sizeof(output_i8), 0) == 1);
    CHECK(vk_graph_sync_host(output_u8, sizeof(output_u8), 0) == 1);
    CHECK(vk_graph_sync_host(&staged_output, sizeof(staged_output), 0) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(staged_output == -62);
    CHECK(vk_graph_qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     2u, 2u) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(vk_graph_qmaskedmean_i8u8(input_i8, mask_alias.mask, mask_alias.bytes,
                                     2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                                     2u, 2u) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_alias.bytes)) == 0);
    vk_graph_reset();
    return 0;
}

static int test_qconv2d_i8u8_packed_chain(void) {
    /* Grouped NHWC/OHWI convolution with full nonzero padding, stride,
       dilation, ReLU6, no bias, and a non-word-aligned output tail. */
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

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                 1u, 3u, 4u, 3u, 3u, 3u, 3u, 2u, 2u, 1u,
                                 1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                 1.0f, 0, 1.0f, 0, 3u, 2u, 3u) == 1);
    /* `hidden` is deliberately not synced before this byte-domain bridge. */
    CHECK(vk_graph_requantize_linear_i8u8(hidden, 27u, output, 27u,
                                          1.0f, 0, 0.5f, -2, 3u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(hidden, sizeof(hidden), 0) == 1);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int i = 0; i < 27; i++) {
        CHECK(hidden[i] == expected_hidden[i]);
        CHECK(output[i] == expected_output[i]);
    }
    vk_graph_reset();

    /* Optional I32 bias and ordinary ReLU use the same canonical ABI. */
    const int8_t bias_input[1] = {-2};
    const int8_t bias_weight[1] = {3};
    const float bias_scale[1] = {0.25f};
    const int32_t bias_zero_point[1] = {0};
    const int32_t bias[1] = {4};
    int8_t bias_output[1] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_qconv2d_i8u8(bias_input, bias_weight, bias_scale,
                                 bias_zero_point, bias, bias_output,
                                 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
                                 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 1u,
                                 0.5f, 0, 0.25f, -3, 2u, 2u, 2u) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(bias_output, sizeof(bias_output), 0) == 1);
    CHECK(bias_output[0] == -3);
    CHECK(vk_graph_qconv2d_i8u8(input, weight, scales, zero_points, NULL, hidden,
                                 1u, 3u, 4u, 3u, 2u, 3u, 3u, 2u, 2u, 1u,
                                 1u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 2u,
                                 1.0f, 0, 1.0f, 0, 3u, 2u, 3u) == 0);
    vk_graph_reset();
    return 0;
}

static int test_typed_i8u8_shape_qdq_chain(void) {
    /* The entire chain stays byte-packed on the device until the final
       DequantizeLinear.  Concat deliberately has unaligned channel lanes,
       and Resize has a two-byte tail, covering the packed-word boundaries. */
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
    const uint32_t input_dtypes[2] = {2u, 2u};
    const int8_t expected[18] = {
        4, 6, 4, 6, 4, 5,
        4, 6, 4, 6, 4, 5,
        3, 6, 3, 6, 3, 1,
    };

    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_quantize_typed_f32_i8u8(source_a, 4u, quantized_a,
                                            1.0f, 0, 2u) == 1);
    CHECK(vk_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                             1.0f, 0, 1.0f, 0, 2u, 2u) == 1);
    CHECK(vk_graph_quantize_typed_f32_i8u8(source_b, 4u, quantized_b,
                                            1.0f, 0, 2u) == 1);
    CHECK(vk_graph_concat_i8u8(inputs, input_elements, input_axes,
                               input_scales, input_zero_points, input_dtypes,
                               2u, concatenated, 8u, 2u, 1u,
                               1.0f, 0, 2u) == 1);
    CHECK(vk_graph_maxpool2d_i8u8(concatenated, pooled,
                                  1u, 2u, 2u, 2u, 2u, 2u,
                                  2u, 2u, 1u, 1u, 0u, 0u, 1u, 1u,
                                  1.0f, 0, 1.0f, 0, 2u, 2u) == 1);
    CHECK(vk_graph_resize_nearest_i8u8(pooled, resized,
                                       1u, 2u, 2u, 2u, 3u, 3u,
                                       1.0f, 0, 1.0f, 0, 2u, 2u) == 1);
    CHECK(vk_graph_dequantize_typed_i8u8_f32(resized, 18u,
                                              1.0f, 0, 2u, output) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(output, sizeof(output), 0) == 1);
    for (int index = 0; index < 18; index++) {
        CHECK(close_enough(output[index], (float)expected[index]));
    }

    /* U8 uses the same packed tail path but has a distinct zero-point domain. */
    const float unsigned_source[3] = {-1.0f, 0.0f, 5.0f};
    uint8_t unsigned_quantized[3] = {0};
    float unsigned_output[3] = {0};
    vk_graph_reset();
    vk_graph_begin_forward();
    CHECK(vk_graph_quantize_typed_f32_i8u8(unsigned_source, 3u,
                                            unsigned_quantized, 1.0f, 128, 3u) == 1);
    CHECK(vk_graph_dequantize_typed_i8u8_f32(unsigned_quantized, 3u,
                                              1.0f, 128, 3u, unsigned_output) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(unsigned_quantized, sizeof(unsigned_quantized), 0) == 1);
    CHECK(vk_graph_sync_host(unsigned_output, sizeof(unsigned_output), 0) == 1);
    CHECK(unsigned_quantized[0] == 127u && unsigned_quantized[1] == 128u &&
          unsigned_quantized[2] == 133u);
    CHECK(close_enough(unsigned_output[0], -1.0f));
    CHECK(close_enough(unsigned_output[1], 0.0f));
    CHECK(close_enough(unsigned_output[2], 5.0f));

    /* The byte-preserving shape path cannot silently change quantization. */
    CHECK(vk_graph_copy_i8u8(quantized_a, 4u, copied_a, 4u,
                             1.0f, 0, 0.5f, 0, 2u, 2u) == 0);
    vk_graph_reset();
    return 0;
}

static int test_multi_entry_matmul_backward(void) {
    float input[2] = {1.0f, 2.0f};
    /* Native IN_OUT layout; the shader separately controls weight reads and
       the destination layout of grad_weight. */
    float weight[4] = {3.0f, 5.0f, 4.0f, 6.0f};
    float grad_output[2] = {7.0f, 8.0f};
    float grad_input[2] = {0.0f, 0.0f};
    float grad_weight[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float grad_bias[2] = {0.0f, 0.0f};
    uint32_t params[8] = {1u, 2u, 2u, 1u, 1u, 1u, 0u, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ
    };
    unsigned char weights[7] = {0, 1, 0, 0, 0, 0, 0};
    CHECK(vk_training_dispatch("matMulBackward", "input_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("matMulBackward", "weight_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("matMulBackward", "bias_main", hosts, bytes,
                               access, weights, 7, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(vk_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    CHECK(close_enough(grad_input[0], 61.0f));
    CHECK(close_enough(grad_input[1], 76.0f));
    CHECK(close_enough(grad_weight[0], 7.0f));
    CHECK(close_enough(grad_weight[1], 8.0f));
    CHECK(close_enough(grad_weight[2], 14.0f));
    CHECK(close_enough(grad_weight[3], 16.0f));
    CHECK(close_enough(grad_bias[0], 7.0f));
    CHECK(close_enough(grad_bias[1], 8.0f));
    return 0;
}

static int test_batched_attention_forward(void) {
    /* seq=1 makes attention output exactly V, which isolates batch offsets. */
    float qkv[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f};
    float sdpa_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(qkv, NULL, 0, sdpa_output,
                            1, 1, 1, 1, 2, 1.0f, 1, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(sdpa_output, sizeof(sdpa_output), 0) == 1);
    CHECK(close_enough(sdpa_output[0], 3.0f));
    CHECK(close_enough(sdpa_output[1], 7.0f));

    float q[2] = {1.0f, 4.0f};
    float k[2] = {2.0f, 5.0f};
    float v[2] = {11.0f, 13.0f};
    float cross_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_f32(q, k, v, NULL, 0, cross_output,
                                  1, 1, 1, 1, 1, 2, 1.0f, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(cross_output, sizeof(cross_output), 0) == 1);
    CHECK(close_enough(cross_output[0], 11.0f));
    CHECK(close_enough(cross_output[1], 13.0f));

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t key_mask[2] = {1, 0};
    float masked_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(masked_qkv, key_mask, 2, masked_output,
                            2, 1, 1, 1, 1, 1.0f, 0, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(masked_output, sizeof(masked_output), 0) == 1);
    CHECK(close_enough(masked_output[0], 2.0f));
    CHECK(close_enough(masked_output[1], 2.0f));

    float cross_q[1] = {0.0f};
    float cross_k[2] = {0.0f, 0.0f};
    float cross_v[2] = {3.0f, 9.0f};
    int32_t cross_mask[2] = {0, 1};
    float masked_cross_output[1] = {0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_f32(cross_q, cross_k, cross_v, cross_mask, 2,
                                  masked_cross_output, 1, 2, 1, 1, 1, 1,
                                  1.0f, 0, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(masked_cross_output, sizeof(masked_cross_output), 0) == 1);
    CHECK(close_enough(masked_cross_output[0], 9.0f));

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
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_f32(parity_qkv, NULL, 0, parity_inference,
                            2, 2, 1, 2, 1, attention_scale, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_inference, sizeof(parity_inference), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_sdpa_training_f32(parity_qkv, NULL, 0, parity_training,
                                     2, 2, 1, 2, 1, attention_scale, 0, 0,
                                     0x80000000u, 0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_training, sizeof(parity_training), 0) == 1);
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_sdpa_training_f32(parity_q, parity_k, parity_v, NULL, 0,
                                           parity_cross_training, 2, 2, 2, 1, 2, 1,
                                           attention_scale, 0, 0, 0x80000000u,
                                           0x9e3779abu, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(parity_cross_training, sizeof(parity_cross_training), 0) == 1);
    for (int i = 0; i < 4; i++) {
        CHECK(close_enough(parity_inference[i], expected_inference[i]));
        CHECK(close_enough(parity_training[i], expected_training[i]));
        CHECK(close_enough(parity_cross_training[i], expected_training[i]));
    }
    CHECK(!close_enough(parity_inference[0], parity_training[0]));

    int32_t tokens[2] = {1, 0};
    float embedding_weight[2] = {4.0f, 7.0f};
    float embedding_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_embedding_f32(tokens, embedding_weight, embedding_output, 2, 1, 2) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(embedding_output, sizeof(embedding_output), 0) == 1);
    CHECK(close_enough(embedding_output[0], 7.0f));
    CHECK(close_enough(embedding_output[1], 4.0f));

    float pool_input[8] = {1.0f, 4.0f, 2.0f, 3.0f, -1.0f, -2.0f, 8.0f, 5.0f};
    float pool_output[2] = {0.0f, 0.0f};
    vk_graph_begin_forward();
    CHECK(vk_graph_maxpool2d_f32(pool_input, pool_output, 2, 2, 2, 1,
                                 1, 1, 2, 2, 2, 2, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(pool_output, sizeof(pool_output), 0) == 1);
    CHECK(close_enough(pool_output[0], 4.0f));
    CHECK(close_enough(pool_output[1], 8.0f));

    float conv_input[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    float conv_weight[1] = {2.0f};
    float conv_bias[1] = {1.0f};
    float conv_output[6] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_conv1d_f32(conv_input, conv_weight, conv_bias, conv_output,
                              2, 1, 3, 1, 3, 1, 1, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(conv_output, sizeof(conv_output), 0) == 1);
    const float expected_conv[6] = {3.0f, 5.0f, 7.0f, 21.0f, 41.0f, 61.0f};
    for (int i = 0; i < 6; i++) CHECK(close_enough(conv_output[i], expected_conv[i]));

    float projected_q[2] = {1.0f, -1.0f};
    float projected_kv[2] = {3.0f, 4.0f};
    float projected_weight[3] = {1.0f, 1.0f, 2.0f};
    float projected_output[2] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_cross_attention_f32(projected_q, projected_kv, projected_weight,
                                       NULL, NULL, projected_output,
                                       1, 1, 1, 1, 1, 2, 0, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(projected_output, sizeof(projected_output), 0) == 1);
    CHECK(close_enough(projected_output[0], 6.0f));
    CHECK(close_enough(projected_output[1], 8.0f));

    float profile_input[4] = {1.0f, 3.0f, 10.0f, 14.0f};
    float profile_output[4] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_profile_x_f32(profile_input, profile_output, 2, 2, 1, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(profile_output, sizeof(profile_output), 0) == 1);
    const float expected_profile[4] = {3.0f, 2.0f, 14.0f, 12.0f};
    for (int i = 0; i < 4; i++) CHECK(close_enough(profile_output[i], expected_profile[i]));

    float concat_a[2] = {1.0f, 10.0f};
    float concat_b[4] = {2.0f, 3.0f, 20.0f, 30.0f};
    const float* concat_inputs[2] = {concat_a, concat_b};
    long concat_sizes[2] = {2, 4};
    int concat_axes[2] = {1, 2};
    float concat_output[6] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                              concat_output, 3, 1, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    const float expected_concat[6] = {1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f};
    for (int i = 0; i < 6; i++) CHECK(close_enough(concat_output[i], expected_concat[i]));
    vk_graph_begin_forward();
    CHECK(vk_graph_concat_f32(concat_inputs, concat_sizes, concat_axes, 2,
                              concat_output, 3, 1, 1) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(concat_output, sizeof(concat_output), 0) == 1);
    for (int i = 0; i < 6; i++) {
        CHECK(close_enough(concat_output[i], 1.0f / (1.0f + expf(-expected_concat[i]))));
    }
    return 0;
}

static int test_batched_attention_backward(void) {
    float qkv[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f};
    float grad_output[2] = {2.0f, 3.0f};
    float grad_qkv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } params = {1u, 1u, 1u, 1u, 2u, 1.0f, 1u, 0u, 0u, 0u, 0u, 1.0f};
    void* hosts[5] = {qkv, qkv, grad_output, grad_qkv, &params};
    size_t bytes[5] = {sizeof(qkv), sizeof(qkv), sizeof(grad_output), sizeof(grad_qkv),
                       sizeof(params)};
    unsigned char access[5] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("sdpaBackward", "main", hosts, bytes, access,
                               NULL, 5, 1, 1, 6) == 0);
    CHECK(vk_training_sync(grad_qkv, sizeof(grad_qkv)) == 0);
    CHECK(close_enough(grad_qkv[0], 0.0f));
    CHECK(close_enough(grad_qkv[1], 0.0f));
    CHECK(close_enough(grad_qkv[2], 2.0f));
    CHECK(close_enough(grad_qkv[3], 0.0f));
    CHECK(close_enough(grad_qkv[4], 0.0f));
    CHECK(close_enough(grad_qkv[5], 3.0f));

    float masked_qkv[6] = {0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 6.0f};
    int32_t mask[2] = {1, 0};
    float masked_grad_output[2] = {1.0f, 1.0f};
    float masked_grad_qkv[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    struct {
        uint32_t seq_len, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
    } masked_params = {2u, 1u, 1u, 1u, 1u, 1.0f, 0u, 1u, 0u, 0u, 0u, 1.0f};
    void* masked_hosts[5] = {masked_qkv, mask, masked_grad_output, masked_grad_qkv,
                             &masked_params};
    size_t masked_bytes[5] = {sizeof(masked_qkv), sizeof(mask), sizeof(masked_grad_output),
                              sizeof(masked_grad_qkv), sizeof(masked_params)};
    CHECK(vk_training_dispatch("sdpaBackward", "main", masked_hosts, masked_bytes,
                               access, NULL, 5, 2, 1, 3) == 0);
    CHECK(vk_training_sync(masked_grad_qkv, sizeof(masked_grad_qkv)) == 0);
    CHECK(close_enough(masked_grad_qkv[0], 0.0f));
    CHECK(close_enough(masked_grad_qkv[1], 0.0f));
    CHECK(close_enough(masked_grad_qkv[2], 2.0f));
    CHECK(close_enough(masked_grad_qkv[3], 0.0f));
    CHECK(close_enough(masked_grad_qkv[4], 0.0f));
    CHECK(close_enough(masked_grad_qkv[5], 0.0f));
    return 0;
}

static int test_batched_cross_attention_backward(void) {
    float q[2] = {1.0f, 4.0f};
    float k[2] = {2.0f, 5.0f};
    float v[2] = {11.0f, 13.0f};
    float grad_output[2] = {2.0f, 3.0f};
    float grad_q[2] = {0.0f, 0.0f};
    float grad_k[2] = {0.0f, 0.0f};
    float grad_v[2] = {0.0f, 0.0f};
    struct {
        uint32_t seq_q, seq_kv, d_model, num_heads, head_dim, batch;
        float scale;
        uint32_t causal, mask_mode, threshold, seed, counter;
        float dropout_scale;
        uint32_t pad[3];
    } params = {1u, 1u, 1u, 1u, 1u, 2u, 1.0f, 0u, 0u,
                0u, 0u, 0u, 1.0f, {0u, 0u, 0u}};
    void* hosts[9] = {q, k, v, q, grad_output, grad_q, grad_k, grad_v, &params};
    size_t bytes[9] = {
        sizeof(q), sizeof(k), sizeof(v), sizeof(q), sizeof(grad_output), sizeof(grad_q),
        sizeof(grad_k), sizeof(grad_v), sizeof(params)
    };
    unsigned char access[9] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ
    };
    const char* entries[3] = {"q_main", "k_main", "v_main"};
    for (int i = 0; i < 3; i++) {
        CHECK(vk_training_dispatch("crossSdpaBackward", entries[i], hosts, bytes,
                                   access, NULL, 9, 1, 1, 2) == 0);
    }
    CHECK(vk_training_sync(grad_q, sizeof(grad_q)) == 0);
    CHECK(vk_training_sync(grad_k, sizeof(grad_k)) == 0);
    CHECK(vk_training_sync(grad_v, sizeof(grad_v)) == 0);
    CHECK(close_enough(grad_q[0], 0.0f) && close_enough(grad_q[1], 0.0f));
    CHECK(close_enough(grad_k[0], 0.0f) && close_enough(grad_k[1], 0.0f));
    CHECK(close_enough(grad_v[0], 2.0f) && close_enough(grad_v[1], 3.0f));
    return 0;
}

static int test_moe_backward_bindings(void) {
    float input = 2.0f;
    float expert_weight = 3.0f;
    float expert_bias = 4.0f;
    float route_index = 0.0f;
    float route_weight = 0.5f;
    float grad_output = 7.0f;
    float grad_input = 0.0f;
    float grad_expert_weight = 0.0f;
    float grad_expert_bias = 0.0f;
    float grad_route_weight = 0.0f;
    uint32_t params[6] = {1u, 1u, 1u, 1u, 1u, 1u};
    void* hosts[11] = {
        &input, &expert_weight, &expert_bias, &route_index, &route_weight,
        &grad_output, &grad_input, &grad_expert_weight, &grad_expert_bias,
        &grad_route_weight, params
    };
    size_t bytes[11] = {
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(params)
    };
    unsigned char access[11] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ
    };
    unsigned char weights[11] = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    const char* entries[4] = {"input_main", "weight_main", "bias_main", "route_main"};
    for (int i = 0; i < 4; i++) {
        CHECK(vk_training_dispatch("moeLinearBackward", entries[i], hosts, bytes,
                                   access, weights, 11, 1, 1, 1) == 0);
    }
    CHECK(vk_training_sync(&grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(&grad_expert_weight, sizeof(grad_expert_weight)) == 0);
    CHECK(vk_training_sync(&grad_expert_bias, sizeof(grad_expert_bias)) == 0);
    CHECK(vk_training_sync(&grad_route_weight, sizeof(grad_route_weight)) == 0);
    CHECK(close_enough(grad_input, 10.5f));
    CHECK(close_enough(grad_expert_weight, 7.0f));
    CHECK(close_enough(grad_expert_bias, 3.5f));
    CHECK(close_enough(grad_route_weight, 70.0f));
    return 0;
}

static int test_prelu_logsoftmax_split_backward(void) {
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
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("preluBackward", "input_main", prelu_hosts, prelu_bytes,
                               prelu_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("preluBackward", "weight_main", prelu_hosts, prelu_bytes,
                               prelu_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(close_enough(grad_input[0], 0.2f) && close_enough(grad_input[1], 2.0f));
    CHECK(close_enough(grad_input[2], 0.6f) && close_enough(grad_input[3], 4.0f));
    CHECK(close_enough(grad_weight[0], -2.5f) && close_enough(grad_weight[1], 0.0f));

    float log_output[2] = {-1.38629436112f, -0.28768207245f};
    float log_grad_output[2] = {2.0f, -1.0f};
    float log_grad_input[2] = {0.0f, 0.0f};
    uint32_t softmax_params[4] = {1u, 2u, 1u, 0u};
    void* softmax_hosts[4] = {log_output, log_grad_output, log_grad_input, softmax_params};
    size_t softmax_bytes[4] = {sizeof(log_output), sizeof(log_grad_output),
                               sizeof(log_grad_input), sizeof(softmax_params)};
    unsigned char softmax_access[4] = {
        VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("softmaxBackward", "main", softmax_hosts, softmax_bytes,
                               softmax_access, NULL, 4, 1, 1, 1) == 0);
    CHECK(vk_training_sync(log_grad_input, sizeof(log_grad_input)) == 0);
    CHECK(close_enough(log_grad_input[0], 1.75f));
    CHECK(close_enough(log_grad_input[1], -1.75f));

    float left_grad[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float right_grad[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float split_grad[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t left_params[5] = {4u, 1u, 2u, 4u, 0u};
    uint32_t right_params[5] = {4u, 1u, 2u, 4u, 2u};
    void* split_hosts[3] = {left_grad, split_grad, left_params};
    size_t split_bytes[3] = {sizeof(left_grad), sizeof(split_grad), sizeof(left_params)};
    unsigned char split_access[3] = {
        VK_TRAINING_READ, VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                               split_access, NULL, 3, 1, 1, 1) == 0);
    split_hosts[0] = right_grad;
    split_hosts[2] = right_params;
    CHECK(vk_training_dispatch("splitBackward", "main", split_hosts, split_bytes,
                               split_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(split_grad, sizeof(split_grad)) == 0);
    const float expected_split[8] = {1.0f, 2.0f, 5.0f, 6.0f,
                                     3.0f, 4.0f, 7.0f, 8.0f};
    for (int i = 0; i < 8; i++) CHECK(close_enough(split_grad[i], expected_split[i]));
    return 0;
}

static int test_groupnorm_dropout_reduce_backward(void) {
    float input[8] = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 0.0f, 5.0f, 1.0f};
    float weight[4] = {1.0f, 0.5f, 1.5f, 0.75f};
    float grad_output[8] = {1.0f, 2.0f, -1.0f, 3.0f, 4.0f, -2.0f, 2.0f, 1.0f};
    float grad_input[8] = {0};
    float grad_weight[4] = {0};
    float grad_bias[4] = {0};
    struct {
        uint32_t batch, height, width, channels, groups, has_bias;
        float epsilon;
        uint32_t pad;
    } params = {1u, 1u, 2u, 4u, 2u, 1u, 1.0e-4f, 0u};
    void* hosts[7] = {input, weight, grad_output, grad_input, grad_weight,
                      grad_bias, &params};
    size_t bytes[7] = {sizeof(input), sizeof(weight), sizeof(grad_output),
                       sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
                       sizeof(params)};
    unsigned char access[7] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("groupNormBackward", "input_main", hosts, bytes,
                               access, NULL, 7, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("groupNormBackward", "param_main", hosts, bytes,
                               access, NULL, 7, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(vk_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(vk_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    float expected_input[8] = {0}, expected_weight[4] = {0}, expected_bias[4] = {0};
    for (int group = 0; group < 2; group++) {
        float mean = 0.0f, square_mean = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            float value = input[spatial * 4 + group * 2 + local];
            mean += value * 0.25f;
            square_mean += value * value * 0.25f;
        }
        float inverse = 1.0f / sqrtf(square_mean - mean * mean + params.epsilon);
        float sum = 0.0f, sum_xhat = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float xhat = (input[index] - mean) * inverse;
            float scaled = grad_output[index] * weight[channel];
            sum += scaled;
            sum_xhat += scaled * xhat;
            expected_weight[channel] += grad_output[index] * xhat;
            expected_bias[channel] += grad_output[index];
        }
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float xhat = (input[index] - mean) * inverse;
            float scaled = grad_output[index] * weight[channel];
            expected_input[index] = inverse * 0.25f * (4.0f * scaled - sum - xhat * sum_xhat);
        }
    }
    for (int i = 0; i < 8; i++) CHECK(close_enough(grad_input[i], expected_input[i]));
    for (int i = 0; i < 4; i++) {
        CHECK(close_enough(grad_weight[i], expected_weight[i]));
        CHECK(close_enough(grad_bias[i], expected_bias[i]));
    }

    float dropout_go[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dropout_gi[8] = {0};
    struct { uint32_t length, threshold, seed, counter; float scale; uint32_t pad[3]; }
        dropout_params = {8u, 0x80000000u, 123u, 7u, 2.0f, {0u, 0u, 0u}};
    void* dropout_hosts[3] = {dropout_go, dropout_gi, &dropout_params};
    size_t dropout_bytes[3] = {sizeof(dropout_go), sizeof(dropout_gi), sizeof(dropout_params)};
    unsigned char dropout_access[3] = {
        VK_TRAINING_READ, VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("dropoutBackward", "main", dropout_hosts, dropout_bytes,
                               dropout_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(dropout_gi, sizeof(dropout_gi)) == 0);
    for (int i = 0; i < 8; i++) {
        float expected = dropout_bits(123u, 7u, (uint32_t)i) >= 0x80000000u
            ? dropout_go[i] * 2.0f : 0.0f;
        CHECK(close_enough(dropout_gi[i], expected));
    }

    float reduce_go[2] = {2.0f, 4.0f};
    float reduce_gi[6] = {0};
    struct { uint32_t rows, width; float scale; uint32_t pad; }
        reduce_params = {6u, 3u, 1.0f / 3.0f, 0u};
    void* reduce_hosts[3] = {reduce_go, reduce_gi, &reduce_params};
    size_t reduce_bytes[3] = {sizeof(reduce_go), sizeof(reduce_gi), sizeof(reduce_params)};
    CHECK(vk_training_dispatch("reduceBackward", "main", reduce_hosts, reduce_bytes,
                               dropout_access, NULL, 3, 1, 1, 1) == 0);
    CHECK(vk_training_sync(reduce_gi, sizeof(reduce_gi)) == 0);
    for (int i = 0; i < 3; i++) CHECK(close_enough(reduce_gi[i], 2.0f / 3.0f));
    for (int i = 3; i < 6; i++) CHECK(close_enough(reduce_gi[i], 4.0f / 3.0f));

    float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b[4] = {2, 3, 5, 7};
    float binary_go[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float grad_a[12] = {0};
    float grad_b[4] = {0};
    uint32_t binary_params[32] = {3u, 12u, 12u, 4u, 1u, 0u, 0u, 0u};
    const uint32_t output_strides[3] = {6u, 2u, 1u};
    const uint32_t a_strides[3] = {6u, 2u, 1u};
    const uint32_t b_strides[3] = {2u, 0u, 1u};
    for (int i = 0; i < 3; i++) {
        binary_params[8 + i] = output_strides[i];
        binary_params[16 + i] = a_strides[i];
        binary_params[24 + i] = b_strides[i];
    }
    void* binary_hosts[6] = {a, b, binary_go, grad_a, grad_b, binary_params};
    size_t binary_bytes[6] = {sizeof(a), sizeof(b), sizeof(binary_go), sizeof(grad_a),
                              sizeof(grad_b), sizeof(binary_params)};
    unsigned char binary_access[6] = {
        VK_TRAINING_READ, VK_TRAINING_READ, VK_TRAINING_READ,
        VK_TRAINING_READ | VK_TRAINING_WRITE,
        VK_TRAINING_READ | VK_TRAINING_WRITE, VK_TRAINING_READ
    };
    CHECK(vk_training_dispatch("basicBackward", "a_main", binary_hosts, binary_bytes,
                               binary_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_dispatch("basicBackward", "b_main", binary_hosts, binary_bytes,
                               binary_access, NULL, 6, 1, 1, 1) == 0);
    CHECK(vk_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(vk_training_sync(grad_b, sizeof(grad_b)) == 0);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(close_enough(grad_a[index], b[batch * 2 + column]));
        }
    CHECK(close_enough(grad_b[0], 9.0f) && close_enough(grad_b[1], 12.0f));
    CHECK(close_enough(grad_b[2], 27.0f) && close_enough(grad_b[3], 30.0f));
    return 0;
}

static int test_groupnorm_dropout_reduce_broadcast_forward(void) {
    vk_graph_reset();
    float input[8] = {1.0f, 3.0f, 2.0f, 6.0f, 4.0f, 0.0f, 5.0f, 1.0f};
    float weight[4] = {1.0f, 0.5f, 1.5f, 0.75f};
    float bias[4] = {0.1f, -0.2f, 0.3f, 0.4f};
    float normalized[8] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_groupnorm_f32(input, weight, bias, normalized,
                                 1, 1, 2, 4, 2, 1.0e-4f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(normalized, sizeof(normalized), 0) == 1);
    for (int group = 0; group < 2; group++) {
        float mean = 0.0f, square_mean = 0.0f;
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            float value = input[spatial * 4 + group * 2 + local];
            mean += value * 0.25f;
            square_mean += value * value * 0.25f;
        }
        float inverse = 1.0f / sqrtf(square_mean - mean * mean + 1.0e-4f);
        for (int spatial = 0; spatial < 2; spatial++) for (int local = 0; local < 2; local++) {
            int channel = group * 2 + local;
            int index = spatial * 4 + channel;
            float expected = (input[index] - mean) * inverse * weight[channel] + bias[channel];
            CHECK(close_enough(normalized[index], expected));
        }
    }

    float dropout_output[8] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_dropout_f32(input, dropout_output, 8, 0x80000000u,
                               123u, 7u, 2.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(dropout_output, sizeof(dropout_output), 0) == 1);
    for (int i = 0; i < 8; i++) {
        float expected = dropout_bits(123u, 7u, (uint32_t)i) >= 0x80000000u
            ? input[i] * 2.0f : 0.0f;
        CHECK(close_enough(dropout_output[i], expected));
    }

    float a[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float b[4] = {2, 3, 5, 7};
    float product[12] = {0};
    uint32_t output_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t a_strides[8] = {6, 2, 1, 0, 0, 0, 0, 0};
    uint32_t b_strides[8] = {2, 0, 1, 0, 0, 0, 0, 0};
    vk_graph_begin_forward();
    CHECK(vk_graph_binary_f32(a, 12, b, 4, product, 12, output_strides,
                              a_strides, b_strides, 3, 0) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(product, sizeof(product), 0) == 1);
    for (int batch = 0; batch < 2; batch++) for (int row = 0; row < 3; row++)
        for (int column = 0; column < 2; column++) {
            int index = (batch * 3 + row) * 2 + column;
            CHECK(close_enough(product[index], a[index] * b[batch * 2 + column]));
        }
    float reduced[2] = {0};
    vk_graph_begin_forward();
    CHECK(vk_graph_reduce_f32(a, reduced, 2, 6, 1.0f / 6.0f) == 1);
    CHECK(vk_graph_end_forward() == 0);
    CHECK(vk_graph_sync_host(reduced, sizeof(reduced), 0) == 1);
    CHECK(close_enough(reduced[0], 3.5f) && close_enough(reduced[1], 9.5f));
    return 0;
}

int main(void) {
    if (vk_init() != 0) {
        puts("vulkan training tests skipped: no Vulkan compute device");
        return 0;
    }
    CHECK(vk_training_available() == 1);
    float host_current = 1.0f;
    vk_graph_reset();
    CHECK(vk_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    vk_graph_mark_host(&host_current, sizeof(host_current), 1);
    CHECK(vk_graph_sync_host(&host_current, sizeof(host_current), 1) == 1);
    vk_graph_reset();
    size_t copy_bytes[3] = {sizeof(float), sizeof(float), sizeof(uint32_t)};
    size_t copy_params[1] = {sizeof(uint32_t)};
    float plan_input = 0.0f, plan_output = 0.0f;
    VkTrainingTensorRequirement copy_tensors[2] = {
        {&plan_input, sizeof(plan_input)}, {&plan_output, sizeof(plan_output)}
    };
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 3, 1, 1, 1) == 1);
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 2, 1, 1, 1) == 0);
    CHECK(vk_training_supports("copyBackward", "main", copy_bytes, 3,
                               UINT32_MAX, 1, 1) == 0);
    CHECK(vk_training_supports("notARealShader", "main", copy_bytes, 3, 1, 1, 1) == 0);
    CHECK(vk_training_plan_supported(1, copy_params, copy_tensors, 2) == 1);
    size_t oversized_params[1] = {2u * 1024u * 1024u};
    VkTrainingTensorRequirement oversized_tensor = {&plan_input, SIZE_MAX};
    CHECK(vk_training_plan_supported(1, oversized_params, copy_tensors, 2) == 0);
    CHECK(vk_training_plan_supported(1, copy_params, &oversized_tensor, 1) == 0);
    CHECK(vk_training_plan_supported(4097, NULL, NULL, 0) == 0);
    CHECK(test_one_shot_matmul_tails_and_fallback() == 0);
    CHECK(test_tiled_qlinear_i8u8_tails() == 0);
    CHECK(test_qlinear_i8u8_packed_chain() == 0);
    CHECK(test_qembedding_i8u8_packed_gather() == 0);
    CHECK(test_qadd_requantize_i8u8_packed_chain() == 0);
    CHECK(test_qsilu_i8u8_packed_chain() == 0);
    CHECK(test_qgelu_i8u8_packed_chain() == 0);
    CHECK(test_qgroupnorm_i8u8_packed_chain() == 0);
    CHECK(test_qlayernorm_i8u8_packed_chain() == 0);
    CHECK(test_qsdpa_i8u8_packed_chain() == 0);
    CHECK(test_qargmax_i8u8_raw() == 0);
    CHECK(test_qmaskedmean_i8u8_packed() == 0);
    CHECK(test_qconv2d_i8u8_packed_chain() == 0);
    CHECK(test_typed_i8u8_shape_qdq_chain() == 0);
    CHECK(vk_training_begin() == 0);

    float dummy = 0.0f;
    void* hosts[1] = {&dummy};
    size_t bytes[1] = {sizeof(dummy)};
    unsigned char access[1] = {VK_TRAINING_READ};
    CHECK(vk_training_dispatch("unknown", "main", hosts, bytes, access, NULL,
                               1, 1, 1, 1) == -1);
    CHECK(test_activation_backward() == 0);
    CHECK(test_gelu_backward_modes() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_multi_entry_matmul_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_batched_attention_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_batched_cross_attention_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_moe_backward_bindings() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_prelu_logsoftmax_split_backward() == 0);
    vk_training_end();
    CHECK(vk_training_begin() == 0);
    CHECK(test_groupnorm_dropout_reduce_backward() == 0);
    vk_training_end();
    CHECK(test_batched_attention_forward() == 0);
    CHECK(test_gelu_forward_modes() == 0);
    CHECK(test_groupnorm_dropout_reduce_broadcast_forward() == 0);
    vk_cleanup();
    CHECK(vk_training_available() == 0);
    CHECK(vk_init() == 0);
    CHECK(vk_training_available() == 1);
    vk_cleanup();
    puts("vulkan training tests passed");
    return 0;
}
