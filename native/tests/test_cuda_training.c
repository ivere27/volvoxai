#include "cuda_engine.h"
#include "cuda_test_engine_scope.h"

#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        cuda_cleanup(); \
        return 1; \
    } \
} while (0)

/* Direct scheduler helpers below use function-local host storage. A binding
 * declared as a weight is stable only until that helper returns, unlike a
 * loaded engine model whose weight storage survives until graph reset. */
#define CHECK_DIRECT_CASE(expression) do { \
    CHECK((expression) == 0); \
    cuda_graph_reset(); \
} while (0)

enum { DROPOUT_ELEMENTS = 131 };

static uint32_t dropout_bits(uint32_t seed, uint32_t counter,
                             uint32_t index) {
    uint32_t value = seed ^ index ^ counter * 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int run_dropout_forward(const float* input, float* output,
                               uint32_t threshold, uint32_t seed,
                               uint32_t counter, float scale) {
    cuda_graph_begin_forward();
    if (!cuda_graph_dropout_f32(input, output, DROPOUT_ELEMENTS, threshold,
                                seed, counter, scale) ||
        cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output,
            (size_t)DROPOUT_ELEMENTS * sizeof(float), 0)) return -1;
    return 0;
}

static int test_dropout_forward(void) {
    float input[DROPOUT_ELEMENTS];
    float first[DROPOUT_ELEMENTS] = {0};
    float second[DROPOUT_ELEMENTS] = {0};
    float no_dropout[DROPOUT_ELEMENTS] = {0};
    const uint32_t threshold = 0x80000000u;
    const uint32_t seed = 0x6d2b79f5u;
    for (uint32_t index = 0; index < DROPOUT_ELEMENTS; index++)
        input[index] = ((float)(int32_t)index - 65.0f) * 0.125f;

    CHECK(run_dropout_forward(input, first, threshold, seed, 7u, 2.0f) == 0);
    CHECK(run_dropout_forward(input, second, threshold, seed, 8u, 2.0f) == 0);
    CHECK(run_dropout_forward(input, no_dropout, 0u, seed, 19u, 1.0f) == 0);
    int changed = 0;
    for (uint32_t index = 0; index < DROPOUT_ELEMENTS; index++) {
        float expected_first = dropout_bits(seed, 7u, index) >= threshold
            ? input[index] * 2.0f : 0.0f;
        float expected_second = dropout_bits(seed, 8u, index) >= threshold
            ? input[index] * 2.0f : 0.0f;
        CHECK(first[index] == expected_first);
        CHECK(second[index] == expected_second);
        CHECK(no_dropout[index] == input[index]);
        changed |= first[index] != second[index];
    }
    CHECK(changed);
    CHECK(!cuda_graph_dropout_f32(NULL, first, DROPOUT_ELEMENTS,
                                  threshold, seed, 1u, 2.0f));
    CHECK(!cuda_graph_dropout_f32(input, first, 0, threshold, seed, 1u, 2.0f));
    CHECK(!cuda_graph_dropout_f32(input, first, DROPOUT_ELEMENTS,
                                  threshold, seed, 1u, NAN));
    CHECK(!cuda_graph_dropout_f32(input, first, DROPOUT_ELEMENTS,
                                  threshold, seed, 1u, 0.5f));
    return 0;
}

static int run_dropout_backward(const float* grad_output, float* grad_input,
                                uint32_t threshold, uint32_t seed,
                                uint32_t counter, float scale) {
    uint32_t params[8] = {
        DROPOUT_ELEMENTS, threshold, seed, counter, float_bits(scale), 0u, 0u, 0u
    };
    void* hosts[3] = {(void*)grad_output, grad_input, params};
    size_t bytes[3] = {
        (size_t)DROPOUT_ELEMENTS * sizeof(float),
        (size_t)DROPOUT_ELEMENTS * sizeof(float), sizeof(params)
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};
    const uint32_t groups =
        (DROPOUT_ELEMENTS + 63u) / 64u;
    if (cuda_training_begin() != 0) return -1;
    int result = cuda_training_dispatch("dropoutBackward", "main", hosts,
        bytes, access, is_weight, 3, groups, 1u, 1u);
    if (result == 0)
        result = cuda_training_sync(grad_input, bytes[1]);
    cuda_training_end();
    return result;
}

static int test_dropout_backward(void) {
    float grad_output[DROPOUT_ELEMENTS];
    float initial[DROPOUT_ELEMENTS];
    float first[DROPOUT_ELEMENTS];
    float second[DROPOUT_ELEMENTS];
    float no_dropout[DROPOUT_ELEMENTS];
    const uint32_t threshold = 0x80000000u;
    const uint32_t seed = 0x6d2b79f5u;
    for (uint32_t index = 0; index < DROPOUT_ELEMENTS; index++) {
        grad_output[index] = ((float)(index % 17u) - 8.0f) * 0.25f;
        initial[index] = ((float)(index % 7u) - 3.0f) * 0.125f;
    }
    memcpy(first, initial, sizeof(first));
    memcpy(second, initial, sizeof(second));
    memcpy(no_dropout, initial, sizeof(no_dropout));
    CHECK(run_dropout_backward(grad_output, first, threshold, seed, 7u,
                               2.0f) == 0);
    CHECK(run_dropout_backward(grad_output, second, threshold, seed, 8u,
                               2.0f) == 0);
    CHECK(run_dropout_backward(grad_output, no_dropout, 0u, seed, 19u,
                               1.0f) == 0);
    int changed = 0;
    for (uint32_t index = 0; index < DROPOUT_ELEMENTS; index++) {
        float expected_first = initial[index];
        float expected_second = initial[index];
        if (dropout_bits(seed, 7u, index) >= threshold)
            expected_first += grad_output[index] * 2.0f;
        if (dropout_bits(seed, 8u, index) >= threshold)
            expected_second += grad_output[index] * 2.0f;
        CHECK(first[index] == expected_first);
        CHECK(second[index] == expected_second);
        CHECK(no_dropout[index] == initial[index] + grad_output[index]);
        changed |= first[index] != second[index];
    }
    CHECK(changed);
    return 0;
}

static int test_dropout_descriptors(void) {
    float grad_output[DROPOUT_ELEMENTS] = {0};
    float grad_input[DROPOUT_ELEMENTS] = {0};
    uint32_t params[8] = {
        DROPOUT_ELEMENTS, 0x80000000u, 123u, 7u, float_bits(2.0f), 0u, 0u, 0u
    };
    void* hosts[3] = {grad_output, grad_input, params};
    size_t bytes[3] = {sizeof(grad_output), sizeof(grad_input), sizeof(params)};
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char bad_access[3] = {1u, 1u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};
    const uint32_t groups = (DROPOUT_ELEMENTS + 63u) / 64u;
    CHECK(cuda_training_supports("dropoutBackward", "main", bytes, 3,
                                 groups, 1u, 1u));
    CHECK(!cuda_training_supports("dropoutBackward", "bad", bytes, 3,
                                  groups, 1u, 1u));
    CHECK(!cuda_training_supports("dropoutBackward", "main", bytes, 3,
                                  groups + 1u, 1u, 1u));
    size_t malformed_bytes[3] = {bytes[0], bytes[1], bytes[2] - sizeof(uint32_t)};
    CHECK(!cuda_training_supports("dropoutBackward", "main", malformed_bytes,
                                  3, groups, 1u, 1u));
    malformed_bytes[0] = bytes[0];
    malformed_bytes[1] = bytes[1] - sizeof(float);
    malformed_bytes[2] = bytes[2];
    CHECK(!cuda_training_supports("dropoutBackward", "main", malformed_bytes,
                                  3, groups, 1u, 1u));

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("dropoutBackward", "main", hosts, bytes,
                                 bad_access, is_weight, 3, groups, 1u, 1u) != 0);
    cuda_training_end();

    params[0] = DROPOUT_ELEMENTS - 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("dropoutBackward", "main", hosts, bytes,
                                 access, is_weight, 3, groups, 1u, 1u) != 0);
    cuda_training_end();
    params[0] = DROPOUT_ELEMENTS;

    params[4] = float_bits(NAN);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("dropoutBackward", "main", hosts, bytes,
                                 access, is_weight, 3, groups, 1u, 1u) != 0);
    cuda_training_end();
    params[4] = float_bits(2.0f);

    params[7] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("dropoutBackward", "main", hosts, bytes,
                                 access, is_weight, 3, groups, 1u, 1u) != 0);
    cuda_training_end();
    return 0;
}

static int close_f32(float actual, float expected) {
    float difference = fabsf(actual - expected);
    float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    return difference <= 2.0e-6f * scale;
}

static int test_matmul_backward_layout(uint32_t weight_din_layout,
                                       uint32_t gradient_din_layout) {
    /* Direct scheduler cases declare function-local weight identities. Close
     * the preceding case's artificial lifetime before a stack address can be
     * reused for a different weight. */
    cuda_graph_reset();
    enum { ROWS = 3, D_IN = 2, D_OUT = 4 };
    const float input[ROWS * D_IN] = {
        1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f,
    };
    const float weight_out_in[D_OUT * D_IN] = {
        0.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.5f, 1.0f, -0.5f,
    };
    const float grad_output[ROWS * D_OUT] = {
        1.0f, -2.0f, 0.5f, 3.0f,
        -1.0f, 0.25f, 2.0f, -0.5f,
        4.0f, -1.5f, 0.75f, 2.5f,
    };
    float weight[D_IN * D_OUT];
    float grad_input[ROWS * D_IN];
    float grad_weight[D_IN * D_OUT];
    float grad_bias[D_OUT];
    float expected_input[ROWS * D_IN];
    float expected_weight[D_IN * D_OUT];
    float expected_bias[D_OUT];
    uint32_t params[8] = {
        ROWS, D_IN, D_OUT, 1u, gradient_din_layout,
        weight_din_layout, 0u, 0u,
    };
    void* hosts[7] = {
        (void*)input, weight, (void*)grad_output, grad_input, grad_weight,
        grad_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t d = 0; d < D_IN; d++) {
        for (uint32_t column = 0; column < D_OUT; column++) {
            uint32_t canonical = column * D_IN + d;
            uint32_t stored = weight_din_layout
                ? d * D_OUT + column : canonical;
            weight[stored] = weight_out_in[canonical];
        }
    }
    for (uint32_t index = 0; index < ROWS * D_IN; index++)
        expected_input[index] = grad_input[index] =
            ((float)(int)index - 2.0f) * 0.125f;
    for (uint32_t index = 0; index < D_IN * D_OUT; index++)
        expected_weight[index] = grad_weight[index] =
            ((float)(int)(index % 5u) - 2.0f) * 0.0625f;
    for (uint32_t column = 0; column < D_OUT; column++)
        expected_bias[column] = grad_bias[column] =
            ((float)(int)column - 1.0f) * 0.25f;

    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t d = 0; d < D_IN; d++) {
            float sum = 0.0f;
            for (uint32_t column = 0; column < D_OUT; column++)
                sum += grad_output[row * D_OUT + column] *
                    weight_out_in[column * D_IN + d];
            expected_input[row * D_IN + d] += sum;
        }
    }
    for (uint32_t d = 0; d < D_IN; d++) {
        for (uint32_t column = 0; column < D_OUT; column++) {
            float sum = 0.0f;
            for (uint32_t row = 0; row < ROWS; row++)
                sum += input[row * D_IN + d] *
                    grad_output[row * D_OUT + column];
            uint32_t destination = gradient_din_layout
                ? d * D_OUT + column : column * D_IN + d;
            expected_weight[destination] += sum;
        }
    }
    for (uint32_t column = 0; column < D_OUT; column++)
        for (uint32_t row = 0; row < ROWS; row++)
            expected_bias[column] += grad_output[row * D_OUT + column];

    CHECK(cuda_training_supports("matMulBackward", "input_main", bytes, 7,
                                 1u, 1u, 1u));
    CHECK(cuda_training_supports("matMulBackward", "weight_main", bytes, 7,
                                 1u, 1u, 1u));
    CHECK(cuda_training_supports("matMulBackward", "bias_main", bytes, 7,
                                 1u, 1u, 1u));
    CHECK(!cuda_training_supports("matMulBackward", "invalid", bytes, 7,
                                  1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "input_main", hosts, bytes,
                                 access, is_weight, 7, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "weight_main", hosts, bytes,
                                 access, is_weight, 7, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "bias_main", hosts, bytes,
                                 access, is_weight, 7, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    cuda_training_end();

    for (uint32_t index = 0; index < ROWS * D_IN; index++)
        CHECK(close_f32(grad_input[index], expected_input[index]));
    for (uint32_t index = 0; index < D_IN * D_OUT; index++)
        CHECK(close_f32(grad_weight[index], expected_weight[index]));
    for (uint32_t column = 0; column < D_OUT; column++)
        CHECK(close_f32(grad_bias[column], expected_bias[column]));
    return 0;
}

static int test_matmul_backward_tiled_tail(uint32_t weight_din_layout,
                                           uint32_t gradient_din_layout) {
    enum { ROWS = 17, D_IN = 19, D_OUT = 23 };
    float input[ROWS * D_IN];
    float weight[D_IN * D_OUT];
    float grad_output[ROWS * D_OUT];
    float grad_input[ROWS * D_IN];
    float grad_weight[D_IN * D_OUT];
    float grad_bias[D_OUT];
    float expected_input[ROWS * D_IN];
    float expected_weight[D_IN * D_OUT];
    float expected_bias[D_OUT];
    uint32_t params[8] = {
        ROWS, D_IN, D_OUT, 1u, gradient_din_layout,
        weight_din_layout, 0u, 0u,
    };
    void* hosts[7] = {
        input, weight, grad_output, grad_input, grad_weight, grad_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t input_groups_x = (D_IN + 15u) / 16u;
    const uint32_t input_groups_y = (ROWS + 15u) / 16u;
    const uint32_t weight_groups_x = (D_OUT + 15u) / 16u;
    const uint32_t weight_groups_y = (D_IN + 15u) / 16u;
    const uint32_t bias_groups_x = (D_OUT + 63u) / 64u;

    cuda_graph_reset();
    for (uint32_t row = 0u; row < ROWS; row++) {
        for (uint32_t d = 0u; d < D_IN; d++)
            input[row * D_IN + d] =
                ((float)((row * 13u + d * 7u) % 37u) - 18.0f) * 0.03125f;
        for (uint32_t column = 0u; column < D_OUT; column++)
            grad_output[row * D_OUT + column] =
                ((float)((row * 5u + column * 11u) % 41u) - 20.0f) *
                0.015625f;
    }
    for (uint32_t d = 0u; d < D_IN; d++) {
        for (uint32_t column = 0u; column < D_OUT; column++) {
            uint32_t destination = weight_din_layout
                ? d * D_OUT + column : column * D_IN + d;
            weight[destination] =
                ((float)((d * 17u + column * 3u) % 43u) - 21.0f) *
                0.0078125f;
        }
    }
    for (uint32_t index = 0u; index < ROWS * D_IN; index++)
        expected_input[index] = grad_input[index] =
            ((float)(index % 9u) - 4.0f) * 0.00390625f;
    for (uint32_t index = 0u; index < D_IN * D_OUT; index++)
        expected_weight[index] = grad_weight[index] =
            ((float)(index % 7u) - 3.0f) * 0.001953125f;
    for (uint32_t column = 0u; column < D_OUT; column++)
        expected_bias[column] = grad_bias[column] =
            ((float)(column % 5u) - 2.0f) * 0.0078125f;

    for (uint32_t row = 0u; row < ROWS; row++) {
        for (uint32_t d = 0u; d < D_IN; d++) {
            float sum = 0.0f;
            for (uint32_t column = 0u; column < D_OUT; column++) {
                uint32_t source = weight_din_layout
                    ? d * D_OUT + column : column * D_IN + d;
                sum += grad_output[row * D_OUT + column] * weight[source];
            }
            expected_input[row * D_IN + d] += sum;
        }
    }
    for (uint32_t d = 0u; d < D_IN; d++) {
        for (uint32_t column = 0u; column < D_OUT; column++) {
            float sum = 0.0f;
            for (uint32_t row = 0u; row < ROWS; row++)
                sum += input[row * D_IN + d] *
                       grad_output[row * D_OUT + column];
            uint32_t destination = gradient_din_layout
                ? d * D_OUT + column : column * D_IN + d;
            expected_weight[destination] += sum;
        }
    }
    for (uint32_t column = 0u; column < D_OUT; column++)
        for (uint32_t row = 0u; row < ROWS; row++)
            expected_bias[column] += grad_output[row * D_OUT + column];

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "input_main", hosts,
        bytes, access, is_weight, 7, input_groups_x, input_groups_y, 1u) == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "weight_main", hosts,
        bytes, access, is_weight, 7, weight_groups_x, weight_groups_y, 1u) == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "bias_main", hosts,
        bytes, access, is_weight, 7, bias_groups_x, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < ROWS * D_IN; index++)
        CHECK(fabsf(grad_input[index] - expected_input[index]) <= 2.0e-5f);
    for (uint32_t index = 0u; index < D_IN * D_OUT; index++)
        CHECK(fabsf(grad_weight[index] - expected_weight[index]) <= 2.0e-5f);
    for (uint32_t column = 0u; column < D_OUT; column++)
        CHECK(fabsf(grad_bias[column] - expected_bias[column]) <= 2.0e-5f);
    return 0;
}

static int test_matmul_backward_descriptors(void) {
    float input[6] = {0};
    float weight[8] = {0};
    float grad_output[12] = {0};
    float grad_input[6] = {0};
    float grad_weight[8] = {0};
    float grad_bias[4] = {0};
    uint32_t params[8] = {3u, 2u, 4u, 1u, 0u, 0u, 0u, 0u};
    void* hosts[7] = {
        input, weight, grad_output, grad_input, grad_weight, grad_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "input_main", hosts, bytes,
                                 access, is_weight, 7, 2u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0; index < 6u; index++) CHECK(grad_input[index] == 0.0f);

    params[7] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("matMulBackward", "weight_main", hosts, bytes,
                                 access, is_weight, 7, 1u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0; index < 8u; index++) CHECK(grad_weight[index] == 0.0f);
    return 0;
}

static uint32_t basic_operand_index(uint32_t output_index, uint32_t base,
                                    const uint32_t params[32]) {
    uint32_t remainder = output_index;
    uint32_t index = 0u;
    for (uint32_t dimension = 0; dimension < params[0]; dimension++) {
        uint32_t coordinate = remainder / params[8u + dimension];
        remainder %= params[8u + dimension];
        index += coordinate * params[base + dimension];
    }
    return index;
}

static int test_basic_backward_kind(uint32_t kind) {
    enum { OUTPUT_LENGTH = 12, A_LENGTH = 4, B_LENGTH = 3 };
    const float a[A_LENGTH] = {0.5f, -1.25f, 2.0f, 0.75f};
    const float b[B_LENGTH] = {1.5f, -0.5f, 2.5f};
    float grad_output[OUTPUT_LENGTH];
    float grad_a[A_LENGTH];
    float grad_b[B_LENGTH];
    float expected_a[A_LENGTH];
    float expected_b[B_LENGTH];
    uint32_t params[32] = {
        3u, OUTPUT_LENGTH, A_LENGTH, B_LENGTH, kind, 0u, 0u, 0u,
        6u, 2u, 1u, 0u, 0u, 0u, 0u, 0u,
        2u, 0u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {
        (void*)a, (void*)b, grad_output, grad_a, grad_b, params,
    };
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < OUTPUT_LENGTH; index++)
        grad_output[index] = ((float)(int)(index % 7u) - 3.0f) * 0.2f;
    for (uint32_t index = 0; index < A_LENGTH; index++)
        expected_a[index] = grad_a[index] = ((float)(int)index - 1.0f) * 0.1f;
    for (uint32_t index = 0; index < B_LENGTH; index++)
        expected_b[index] = grad_b[index] = ((float)(int)index + 1.0f) * -0.05f;
    for (uint32_t output = 0; output < OUTPUT_LENGTH; output++) {
        uint32_t ai = basic_operand_index(output, 16u, params);
        uint32_t bi = basic_operand_index(output, 24u, params);
        float da = 1.0f;
        float db = 1.0f;
        if (kind == 1u) {
            da = b[bi];
            db = a[ai];
        } else if (kind == 2u) {
            db = -1.0f;
        } else if (kind == 3u) {
            da = 1.0f / b[bi];
            db = -a[ai] / (b[bi] * b[bi]);
        }
        expected_a[ai] += grad_output[output] * da;
        expected_b[bi] += grad_output[output] * db;
    }

    CHECK(cuda_training_supports("basicBackward", "a_main", bytes, 6,
                                 1u, 1u, 1u));
    CHECK(cuda_training_supports("basicBackward", "b_main", bytes, 6,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < A_LENGTH; index++)
        CHECK(close_f32(grad_a[index], expected_a[index]));
    for (uint32_t index = 0; index < B_LENGTH; index++)
        CHECK(close_f32(grad_b[index], expected_b[index]));
    return 0;
}

static int test_basic_backward_identity_kind(uint32_t kind) {
    enum { LENGTH = 6 };
    const float a[LENGTH] = {0.5f, -1.25f, 2.0f, 0.75f, -0.4f, 1.1f};
    const float b[LENGTH] = {1.5f, -0.5f, 2.5f, 1.25f, -2.0f, 0.8f};
    const float grad_output[LENGTH] = {0.2f, -0.4f, 0.6f, 0.8f, -1.0f, 1.2f};
    float grad_a[LENGTH];
    float grad_b[LENGTH];
    float expected_a[LENGTH];
    float expected_b[LENGTH];
    uint32_t params[32] = {
        3u, LENGTH, LENGTH, LENGTH, kind, 0u, 0u, 0u,
        6u, 3u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 3u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 3u, 1u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {
        (void*)a, (void*)b, (void*)grad_output, grad_a, grad_b, params,
    };
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++) {
        float da = 1.0f;
        float db = 1.0f;
        expected_a[index] = grad_a[index] = (float)index * 0.05f;
        expected_b[index] = grad_b[index] = ((float)(int)index - 2.0f) * 0.03f;
        if (kind == 1u) {
            da = b[index];
            db = a[index];
        } else if (kind == 2u) {
            db = -1.0f;
        } else if (kind == 3u) {
            da = 1.0f / b[index];
            db = -a[index] / (b[index] * b[index]);
        }
        expected_a[index] += grad_output[index] * da;
        expected_b[index] += grad_output[index] * db;
    }

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++) {
        CHECK(close_f32(grad_a[index], expected_a[index]));
        CHECK(close_f32(grad_b[index], expected_b[index]));
    }
    return 0;
}

static int test_basic_backward_scalar_b(uint32_t kind, uint32_t relu) {
    enum { LENGTH = 7 };
    const float a[LENGTH] = {-2.5f, -1.0f, -0.25f, 0.5f, 1.25f, 3.0f, 6.0f};
    const float b[1] = {1.25f};
    const float grad_output[LENGTH] = {0.3f, -0.5f, 0.7f, 0.9f, -1.1f, 1.3f, 1.5f};
    float grad_a[LENGTH];
    float grad_b[1] = {-0.2f};
    float expected_a[LENGTH];
    float expected_b = grad_b[0];
    uint32_t params[32] = {
        2u, LENGTH, LENGTH, 1u, kind, relu, 0u, 0u,
        LENGTH, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {
        (void*)a, (void*)b, (void*)grad_output, grad_a, grad_b, params,
    };
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    CHECK(kind < 4u);
    CHECK(relu <= 2u);
    CHECK(kind == 0u || relu == 0u);
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        float activation = 1.0f;
        float da = 1.0f;
        float db = 1.0f;
        expected_a[index] = grad_a[index] = (float)index * 0.025f;
        if (kind == 0u && relu != 0u) {
            float value = a[index] + b[0];
            if (value <= 0.0f || (relu == 2u && value >= 6.0f))
                activation = 0.0f;
        } else if (kind == 1u) {
            da = b[0];
            db = a[index];
        } else if (kind == 2u) {
            db = -1.0f;
        } else if (kind == 3u) {
            da = 1.0f / b[0];
            db = -a[index] / (b[0] * b[0]);
        }
        expected_a[index] += grad_output[index] * activation * da;
        expected_b += grad_output[index] * activation * db;
    }

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_a[index], expected_a[index]));
    CHECK(close_f32(grad_b[0], expected_b));
    return 0;
}

static int test_basic_backward_scalar_a(uint32_t kind, uint32_t relu) {
    enum { LENGTH = 7 };
    const float a[1] = {1.25f};
    const float b[LENGTH] = {-2.5f, -1.0f, -0.25f, 0.5f, 1.25f, 3.0f, 6.0f};
    const float grad_output[LENGTH] = {0.3f, -0.5f, 0.7f, 0.9f, -1.1f, 1.3f, 1.5f};
    float grad_a[1] = {0.15f};
    float grad_b[LENGTH];
    float expected_a = grad_a[0];
    float expected_b[LENGTH];
    uint32_t params[32] = {
        2u, LENGTH, 1u, LENGTH, kind, relu, 0u, 0u,
        LENGTH, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {
        (void*)a, (void*)b, (void*)grad_output, grad_a, grad_b, params,
    };
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    CHECK(kind < 4u);
    CHECK(relu <= 2u);
    CHECK(kind == 0u || relu == 0u);
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        float activation = 1.0f;
        float da = 1.0f;
        float db = 1.0f;
        expected_b[index] = grad_b[index] = (float)index * -0.035f;
        if (kind == 0u && relu != 0u) {
            float value = a[0] + b[index];
            if (value <= 0.0f || (relu == 2u && value >= 6.0f))
                activation = 0.0f;
        } else if (kind == 1u) {
            da = b[index];
            db = a[0];
        } else if (kind == 2u) {
            db = -1.0f;
        } else if (kind == 3u) {
            da = 1.0f / b[index];
            db = -a[0] / (b[index] * b[index]);
        }
        expected_a += grad_output[index] * activation * da;
        expected_b[index] += grad_output[index] * activation * db;
    }

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
    cuda_training_end();
    CHECK(close_f32(grad_a[0], expected_a));
    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_b[index], expected_b[index]));
    return 0;
}

static int test_basic_backward_scalar_reduction_determinism(void) {
    enum { LENGTH = 131 };
    float a[LENGTH];
    const float b[1] = {0.75f};
    float grad_output[LENGTH];
    float grad_a[LENGTH];
    float grad_b[1];
    float results[2];
    float expected = 0.125f;
    uint32_t params[32] = {
        2u, LENGTH, LENGTH, 1u, 0u, 0u, 0u, 0u,
        LENGTH, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {a, (void*)b, grad_output, grad_a, grad_b, params};
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0u; index < LENGTH; ++index) {
        a[index] = ((float)(int)(index % 23u) - 11.0f) * 0.031f;
        grad_output[index] =
            ((float)(int)((index * 17u) % 29u) - 14.0f) * 0.017f;
        grad_a[index] = (float)index * 0.001f;
        expected += grad_output[index];
    }

    for (uint32_t run = 0u; run < 2u; ++run) {
        grad_b[0] = 0.125f;
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                     access, is_weight, 6, 1u, 1u, 1u) == 0);
        CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
        cuda_training_end();
        results[run] = grad_b[0];
        if (run == 0u) cuda_graph_reset();
    }
    CHECK(float_bits(results[0]) == float_bits(results[1]));
    CHECK(close_f32(results[0], expected));
    return 0;
}

static int test_basic_backward_scalar_staged(int scalar_a, uint32_t kind,
                                             uint32_t relu) {
    enum {
        LENGTH = 8203,
        ITEMS_PER_PARTIAL = 4096,
        PARTIAL_COUNT = (LENGTH + ITEMS_PER_PARTIAL - 1) /
            ITEMS_PER_PARTIAL,
    };
    float a[LENGTH];
    float b[LENGTH];
    float grad_output[LENGTH];
    float grad_a[LENGTH];
    float grad_b[LENGTH];
    float results[2];
    const float initial = scalar_a ? 0.1875f : -0.21875f;
    float expected = initial;
    uint32_t params[32] = {
        1u, LENGTH, scalar_a ? 1u : LENGTH,
        scalar_a ? LENGTH : 1u, kind, relu, 0u, 0u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        scalar_a ? 0u : 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        scalar_a ? 1u : 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {a, b, grad_output, grad_a, grad_b, params};
    size_t bytes[6] = {
        (scalar_a ? 1u : LENGTH) * sizeof(float),
        (scalar_a ? LENGTH : 1u) * sizeof(float),
        sizeof(grad_output),
        (scalar_a ? 1u : LENGTH) * sizeof(float),
        (scalar_a ? LENGTH : 1u) * sizeof(float),
        sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};
    const char* entry = scalar_a ? "a_scalar_staged" : "b_scalar_staged";
    size_t workspace_before;
    size_t workspace_after;
    uint64_t launches_before;

    CHECK(kind < 4u);
    CHECK(relu <= 2u);
    CHECK(kind == 0u || relu == 0u);
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        a[index] = ((float)(int)((index * 11u) % 41u) - 20.0f) * 0.03125f;
        b[index] = ((float)(int)((index * 7u) % 37u) - 18.0f) * 0.0390625f;
        if (fabsf(a[index]) < 0.125f) a[index] += 0.28125f;
        if (fabsf(b[index]) < 0.125f) b[index] -= 0.328125f;
        grad_output[index] =
            ((float)(int)((index * 17u) % 43u) - 21.0f) * 0.015625f;
        grad_a[index] = (float)(int)(index % 5u) * 0.00390625f;
        grad_b[index] = (float)(int)(index % 7u) * -0.0029296875f;
    }
    if (scalar_a) {
        a[0] = 0.75f;
        grad_a[0] = initial;
    } else {
        b[0] = 0.875f;
        grad_b[0] = initial;
    }
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        float activation = 1.0f;
        float derivative = 1.0f;
        float av = scalar_a ? a[0] : a[index];
        float bv = scalar_a ? b[index] : b[0];
        if (kind == 0u && relu != 0u) {
            float value = av + bv;
            if (value <= 0.0f || (relu == 2u && value >= 6.0f))
                activation = 0.0f;
        } else if (scalar_a && kind == 1u) {
            derivative = bv;
        } else if (scalar_a && kind == 3u) {
            derivative = 1.0f / bv;
        } else if (!scalar_a && kind == 1u) {
            derivative = av;
        } else if (!scalar_a && kind == 2u) {
            derivative = -1.0f;
        } else if (!scalar_a && kind == 3u) {
            derivative = -av / (bv * bv);
        }
        expected += grad_output[index] * activation * derivative;
    }

    CHECK(cuda_training_supports("basicBackward", entry, bytes, 6,
                                 PARTIAL_COUNT, 1u, 1u));
    workspace_before = cuda_test_training_basic_workspace_bytes();
    CHECK(!cuda_training_preflight("basicBackward", entry, bytes, 6,
                                   PARTIAL_COUNT - 1u, 1u, 1u));
    CHECK(cuda_test_training_basic_workspace_bytes() == workspace_before);
    launches_before = cuda_test_launch_count();
    CHECK(cuda_training_preflight("basicBackward", entry, bytes, 6,
                                  PARTIAL_COUNT, 1u, 1u));
    workspace_after = cuda_test_training_basic_workspace_bytes();
    CHECK(workspace_after >= PARTIAL_COUNT * sizeof(float));
    CHECK(workspace_after >= workspace_before);
    CHECK(cuda_test_launch_count() == launches_before);
    CHECK(cuda_training_preflight("basicBackward", entry, bytes, 6,
                                  PARTIAL_COUNT, 1u, 1u));
    CHECK(cuda_test_training_basic_workspace_bytes() == workspace_after);

    if (scalar_a) grad_a[0] = initial;
    else grad_b[0] = initial;
    launches_before = cuda_test_launch_count();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", entry, hosts, bytes,
                                 access, is_weight, 6,
                                 PARTIAL_COUNT - 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(cuda_test_launch_count() == launches_before);
    CHECK((scalar_a ? grad_a[0] : grad_b[0]) == initial);
    for (uint32_t run = 0u; run < 2u; ++run) {
        if (scalar_a) grad_a[0] = initial;
        else grad_b[0] = initial;
        launches_before = cuda_test_launch_count();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("basicBackward", entry, hosts, bytes,
                                     access, is_weight, 6,
                                     PARTIAL_COUNT, 1u, 1u) == 0);
        CHECK(cuda_test_launch_count() == launches_before + 2u);
        CHECK(cuda_training_sync(scalar_a ? (void*)grad_a : (void*)grad_b,
                                 sizeof(float)) == 0);
        cuda_training_end();
        results[run] = scalar_a ? grad_a[0] : grad_b[0];
        if (run == 0u) cuda_graph_reset();
    }
    CHECK(float_bits(results[0]) == float_bits(results[1]));
    {
        float scale = fmaxf(1.0f, fmaxf(fabsf(results[0]), fabsf(expected)));
        CHECK(fabsf(results[0] - expected) <= 5.0e-5f * scale);
    }
    return 0;
}

static int test_basic_backward_cooperative_kind(uint32_t kind,
                                                uint32_t relu) {
    enum { OUTPUT_LENGTH = 1430, A_LENGTH = 130, B_LENGTH = 143 };
    float a[A_LENGTH];
    float b[B_LENGTH];
    float grad_output[OUTPUT_LENGTH];
    float grad_a[A_LENGTH];
    float grad_b[B_LENGTH];
    float initial_a[A_LENGTH];
    float initial_b[B_LENGTH];
    float expected_a[A_LENGTH];
    float expected_b[B_LENGTH];
    float first_a[A_LENGTH];
    float first_b[B_LENGTH];
    uint32_t params[32] = {
        4u, OUTPUT_LENGTH, A_LENGTH, B_LENGTH, kind, relu, 0u, 0u,
        286u, 26u, 2u, 1u, 0u, 0u, 0u, 0u,
        26u, 0u, 2u, 1u, 0u, 0u, 0u, 0u,
        0u, 13u, 1u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {a, b, grad_output, grad_a, grad_b, params};
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    CHECK(kind < 4u);
    CHECK(relu <= 2u);
    CHECK(kind == 0u || relu == 0u);
    for (uint32_t index = 0u; index < A_LENGTH; ++index) {
        a[index] = ((float)(int)((index * 11u) % 29u) - 14.0f) * 0.0625f;
        initial_a[index] = ((float)(int)(index % 7u) - 3.0f) * 0.015625f;
        expected_a[index] = initial_a[index];
    }
    for (uint32_t index = 0u; index < B_LENGTH; ++index) {
        b[index] = ((float)(int)((index * 7u) % 19u) - 9.0f) * 0.09375f;
        if (fabsf(b[index]) < 0.125f) b[index] += 0.375f;
        initial_b[index] = ((float)(int)(index % 5u) - 2.0f) * -0.0234375f;
        expected_b[index] = initial_b[index];
    }
    for (uint32_t output = 0u; output < OUTPUT_LENGTH; ++output) {
        uint32_t ai = basic_operand_index(output, 16u, params);
        uint32_t bi = basic_operand_index(output, 24u, params);
        float activation = 1.0f;
        float da = 1.0f;
        float db = 1.0f;
        grad_output[output] =
            ((float)(int)((output * 13u) % 31u) - 15.0f) * 0.01953125f;
        if (kind == 0u && relu != 0u) {
            float value = a[ai] + b[bi];
            if (value <= 0.0f || (relu == 2u && value >= 6.0f))
                activation = 0.0f;
        } else if (kind == 1u) {
            da = b[bi];
            db = a[ai];
        } else if (kind == 2u) {
            db = -1.0f;
        } else if (kind == 3u) {
            da = 1.0f / b[bi];
            db = -a[ai] / (b[bi] * b[bi]);
        }
        expected_a[ai] += grad_output[output] * activation * da;
        expected_b[bi] += grad_output[output] * activation * db;
    }

    CHECK(cuda_training_supports("basicBackward", "a_cooperative", bytes, 6,
                                 A_LENGTH, 1u, 1u));
    CHECK(cuda_training_supports("basicBackward", "b_cooperative", bytes, 6,
                                 B_LENGTH, 1u, 1u));
    for (uint32_t run = 0u; run < 2u; ++run) {
        memcpy(grad_a, initial_a, sizeof(grad_a));
        memcpy(grad_b, initial_b, sizeof(grad_b));
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("basicBackward", "a_cooperative",
            hosts, bytes, access, is_weight, 6, A_LENGTH, 1u, 1u) == 0);
        CHECK(cuda_training_dispatch("basicBackward", "b_cooperative",
            hosts, bytes, access, is_weight, 6, B_LENGTH, 1u, 1u) == 0);
        CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
        CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
        cuda_training_end();
        if (run == 0u) {
            memcpy(first_a, grad_a, sizeof(first_a));
            memcpy(first_b, grad_b, sizeof(first_b));
            cuda_graph_reset();
        }
    }
    CHECK(memcmp(first_a, grad_a, sizeof(first_a)) == 0);
    CHECK(memcmp(first_b, grad_b, sizeof(first_b)) == 0);
    for (uint32_t index = 0u; index < A_LENGTH; ++index)
        CHECK(close_f32(grad_a[index], expected_a[index]));
    for (uint32_t index = 0u; index < B_LENGTH; ++index)
        CHECK(close_f32(grad_b[index], expected_b[index]));
    return 0;
}

static int test_basic_backward_cooperative_descriptors(void) {
    enum { OUTPUT_LENGTH = 210, A_LENGTH = 21, B_LENGTH = 10 };
    float a[A_LENGTH] = {0};
    float b[B_LENGTH] = {0};
    float grad_output[OUTPUT_LENGTH] = {0};
    float grad_a[A_LENGTH] = {0};
    float grad_b[B_LENGTH] = {0};
    uint32_t params[32] = {
        4u, OUTPUT_LENGTH, A_LENGTH, B_LENGTH, 0u, 0u, 0u, 0u,
        70u, 14u, 2u, 1u, 0u, 0u, 0u, 0u,
        7u, 0u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 2u, 0u, 1u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {a, b, grad_output, grad_a, grad_b, params};
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    CHECK(!cuda_training_supports("basicBackward", "a_cooperative", bytes, 6,
                                  A_LENGTH - 1u, 1u, 1u));
    CHECK(!cuda_training_supports("basicBackward", "b_cooperative", bytes, 6,
                                  B_LENGTH - 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_cooperative", hosts,
        bytes, access, is_weight, 6, A_LENGTH - 1u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < A_LENGTH; ++index)
        CHECK(grad_a[index] == 0.0f);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_cooperative", hosts,
        bytes, access, is_weight, 6, B_LENGTH - 1u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < B_LENGTH; ++index)
        CHECK(grad_b[index] == 0.0f);
    return 0;
}

static int test_basic_backward_add_activation(uint32_t relu) {
    const float a[6] = {-2.0f, 0.5f, 4.0f, -1.0f, 2.0f, 8.0f};
    const float b[6] = {1.0f, 0.5f, 2.0f, 1.0f, 1.0f, -1.0f};
    const float grad_output[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float grad_a[6] = {0};
    float grad_b[6] = {0};
    uint32_t params[32] = {
        1u, 6u, 6u, 6u, 0u, relu, 0u, 0u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {
        (void*)a, (void*)b, (void*)grad_output, grad_a, grad_b, params,
    };
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0};

    CHECK(relu == 1u || relu == 2u);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("basicBackward", "b_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_a, sizeof(grad_a)) == 0);
    CHECK(cuda_training_sync(grad_b, sizeof(grad_b)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < 6u; index++) {
        float sum = a[index] + b[index];
        float expected = (sum <= 0.0f || (relu == 2u && sum >= 6.0f))
            ? 0.0f : grad_output[index];
        CHECK(grad_a[index] == expected);
        CHECK(grad_b[index] == expected);
    }
    return 0;
}

static int test_basic_backward_descriptors(void) {
    float a[4] = {0};
    float b[3] = {0};
    float grad_output[12] = {0};
    float grad_a[4] = {0};
    float grad_b[3] = {0};
    uint32_t params[32] = {
        3u, 12u, 4u, 3u, 0u, 0u, 0u, 0u,
        6u, 2u, 1u, 0u, 0u, 0u, 0u, 0u,
        2u, 0u, 1u, 0u, 0u, 0u, 0u, 0u,
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
    };
    void* hosts[6] = {a, b, grad_output, grad_a, grad_b, params};
    size_t bytes[6] = {
        sizeof(a), sizeof(b), sizeof(grad_output), sizeof(grad_a),
        sizeof(grad_b), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 0u, 0u, 0u, 0u, 0u};

    params[9] = 3u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0; index < 4u; index++) CHECK(grad_a[index] == 0.0f);

    params[9] = 2u;
    params[5] = 3u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) != 0);
    cuda_training_end();
    params[5] = 1u;
    params[4] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("basicBackward", "a_main", hosts, bytes,
                                 access, is_weight, 6, 1u, 1u, 1u) != 0);
    cuda_training_end();
    params[4] = 0u;
    params[5] = 0u;
    for (uint32_t index = 0; index < 4u; index++) CHECK(grad_a[index] == 0.0f);
    return 0;
}

static float activation_erf_approx(float value) {
    float sign = value >= 0.0f ? 1.0f : -1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t +
        1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    return sign * (1.0f - polynomial * expf(-x * x));
}

static float activation_derivative(uint32_t kind, float x, float y,
                                   float alpha, float beta) {
    if (kind == 0u) return x > 0.0f ? 1.0f : 0.0f;
    if (kind == 1u)
        return 0.5f * (1.0f + activation_erf_approx(x * 0.7071067811865476f)) +
            x * expf(-0.5f * x * x) * 0.3989422804014327f;
    if (kind == 9u) {
        const float c = 0.7978845608f;
        float u = c * (x + 0.044715f * x * x * x);
        float t = tanhf(u);
        float du = c * (1.0f + 3.0f * 0.044715f * x * x);
        return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
    }
    if (kind == 2u) {
        float sigmoid = 1.0f / (1.0f + expf(-x));
        return sigmoid + x * sigmoid * (1.0f - sigmoid);
    }
    if (kind == 3u) return y * (1.0f - y);
    if (kind == 4u) return 1.0f - y * y;
    if (kind == 5u) return x > 0.0f ? 1.0f : alpha;
    if (kind == 6u) return x > -3.0f && x < 3.0f ? 1.0f / 6.0f : 0.0f;
    if (kind == 7u) {
        if (x <= -3.0f) return 0.0f;
        if (x >= 3.0f) return 1.0f;
        return x / 3.0f + 0.5f;
    }
    if (kind == 8u) return x > alpha && x < beta ? 1.0f : 0.0f;
    return 1.0f;
}

static int test_activation_backward_kind(uint32_t kind) {
    enum { LENGTH = 13 };
    const float input[LENGTH] = {
        -5.0f, -3.0f, -2.0f, -1.0f, -0.25f, 0.0f, 0.25f,
        0.75f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    float output[LENGTH];
    float grad_output[LENGTH];
    float grad_input[LENGTH];
    float expected[LENGTH];
    float alpha = kind == 8u ? -1.0f : (kind == 5u ? 0.2f : 0.01f);
    float beta = kind == 8u ? 1.0f : 0.0f;
    uint32_t params[4] = {LENGTH, kind, float_bits(alpha), float_bits(beta)};
    void* hosts[5] = {
        (void*)input, output, grad_output, grad_input, params,
    };
    size_t bytes[5] = {
        sizeof(input), sizeof(output), sizeof(grad_output), sizeof(grad_input),
        sizeof(params),
    };
    const unsigned char access[5] = {1u, 1u, 1u, 3u, 1u};
    const unsigned char is_weight[5] = {0u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++) {
        output[index] = kind == 3u ? 1.0f / (1.0f + expf(-input[index]))
            : (kind == 4u ? tanhf(input[index]) : 0.0f);
        grad_output[index] = ((float)(int)(index % 5u) - 2.0f) * 0.3f;
        grad_input[index] = ((float)(int)(index % 3u) - 1.0f) * 0.125f;
        expected[index] = grad_input[index] + grad_output[index] *
            activation_derivative(kind, input[index], output[index], alpha, beta);
    }
    CHECK(cuda_training_supports("activationBackward", "main", bytes, 5,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("activationBackward", "main", hosts, bytes,
                                 access, is_weight, 5, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++) {
        float difference = fabsf(grad_input[index] - expected[index]);
        float scale = fmaxf(1.0f, fabsf(expected[index]));
        CHECK(difference <= 3.0e-4f * scale);
    }
    return 0;
}

static int test_activation_backward_descriptors(void) {
    float input[4] = {0};
    float output[4] = {0};
    float grad_output[4] = {0};
    float grad_input[4] = {0};
    uint32_t params[4] = {4u, 10u, float_bits(0.01f), 0u};
    void* hosts[5] = {input, output, grad_output, grad_input, params};
    size_t bytes[5] = {
        sizeof(input), sizeof(output), sizeof(grad_output), sizeof(grad_input),
        sizeof(params),
    };
    const unsigned char access[5] = {1u, 1u, 1u, 3u, 1u};
    const unsigned char is_weight[5] = {0u, 0u, 0u, 0u, 0u};

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("activationBackward", "main", hosts, bytes,
                                 access, is_weight, 5, 1u, 1u, 1u) != 0);
    cuda_training_end();
    for (uint32_t index = 0; index < 4u; index++) CHECK(grad_input[index] == 0.0f);
    return 0;
}

static int test_reduce_backward(void) {
    enum { ROWS = 4, WIDTH = 3, LENGTH = ROWS * WIDTH };
    const float grad_output[ROWS] = {1.5f, -2.0f, 0.75f, 4.0f};
    float grad_input[LENGTH];
    float expected[LENGTH];
    const float scale = 1.0f / (float)WIDTH;
    uint32_t params[4] = {LENGTH, WIDTH, float_bits(scale), 0u};
    void* hosts[3] = {(void*)grad_output, grad_input, params};
    size_t bytes[3] = {sizeof(grad_output), sizeof(grad_input), sizeof(params)};
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++) {
        grad_input[index] = ((float)(int)(index % 5u) - 2.0f) * 0.1f;
        expected[index] = grad_input[index] + grad_output[index / WIDTH] * scale;
    }
    CHECK(cuda_training_supports("reduceBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("reduceBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++)
        CHECK(close_f32(grad_input[index], expected[index]));

    params[1] = 0u;
    memcpy(grad_input, expected, sizeof(grad_input));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("reduceBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(grad_input, expected, sizeof(grad_input)) == 0);
    return 0;
}

static int test_softmax_backward_mode(uint32_t log_softmax) {
    enum { ROWS = 2, WIDTH = 4, LENGTH = ROWS * WIDTH };
    const float probabilities[LENGTH] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.25f, 0.25f, 0.125f, 0.375f,
    };
    const float grad_output[LENGTH] = {
        1.0f, -0.5f, 2.0f, 0.25f,
        -1.0f, 0.75f, 0.5f, -0.25f,
    };
    float output[LENGTH];
    float grad_input[LENGTH];
    float expected[LENGTH];
    uint32_t params[4] = {ROWS, WIDTH, log_softmax, 0u};
    void* hosts[4] = {output, (void*)grad_output, grad_input, params};
    size_t bytes[4] = {
        sizeof(output), sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[4] = {1u, 1u, 3u, 1u};
    const unsigned char is_weight[4] = {0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++) {
        output[index] = log_softmax ? logf(probabilities[index])
            : probabilities[index];
        grad_input[index] = ((float)(int)(index % 3u) - 1.0f) * 0.125f;
        expected[index] = grad_input[index];
    }
    for (uint32_t row = 0; row < ROWS; row++) {
        float reduction = 0.0f;
        for (uint32_t column = 0; column < WIDTH; column++) {
            uint32_t index = row * WIDTH + column;
            reduction += log_softmax ? grad_output[index]
                : grad_output[index] * probabilities[index];
        }
        for (uint32_t column = 0; column < WIDTH; column++) {
            uint32_t index = row * WIDTH + column;
            expected[index] += log_softmax
                ? grad_output[index] - probabilities[index] * reduction
                : probabilities[index] * (grad_output[index] - reduction);
        }
    }
    CHECK(cuda_training_supports("softmaxBackward", "main", bytes, 4,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("softmaxBackward", "main", hosts, bytes,
                                 access, is_weight, 4, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++) {
        float tolerance = log_softmax ? 8.0e-5f : 2.0e-6f;
        CHECK(fabsf(grad_input[index] - expected[index]) <= tolerance);
    }
    return 0;
}

static int test_embedding_backward(void) {
    enum { TOKENS = 5, D_MODEL = 3, VOCAB = 4 };
    const int32_t token_ids[TOKENS] = {2, 1, 2, 0, 1};
    float grad_output[TOKENS * D_MODEL];
    float grad_weight[VOCAB * D_MODEL];
    float expected[VOCAB * D_MODEL];
    uint32_t params[4] = {TOKENS, D_MODEL, VOCAB, 0u};
    void* hosts[4] = {
        (void*)token_ids, grad_output, grad_weight, params,
    };
    size_t bytes[4] = {
        sizeof(token_ids), sizeof(grad_output), sizeof(grad_weight),
        sizeof(params),
    };
    const unsigned char access[4] = {1u, 1u, 3u, 1u};
    const unsigned char is_weight[4] = {0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < TOKENS * D_MODEL; index++)
        grad_output[index] = ((float)(int)(index % 7u) - 3.0f) * 0.25f;
    for (uint32_t index = 0; index < VOCAB * D_MODEL; index++)
        expected[index] = grad_weight[index] =
            ((float)(int)(index % 4u) - 1.0f) * 0.0625f;
    for (uint32_t row = 0; row < TOKENS; row++)
        for (uint32_t feature = 0; feature < D_MODEL; feature++)
            expected[(uint32_t)token_ids[row] * D_MODEL + feature] +=
                grad_output[row * D_MODEL + feature];
    CHECK(cuda_training_supports("embeddingBackward", "main", bytes, 4,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("embeddingBackward", "main", hosts, bytes,
                                 access, is_weight, 4, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < VOCAB * D_MODEL; index++)
        CHECK(close_f32(grad_weight[index], expected[index]));
    return 0;
}

static void layernorm_reference(const float* input, const float* weight,
                                const float* grad_output, float* grad_input,
                                float* grad_weight, float* grad_bias,
                                uint32_t rows, uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t offset = row * width;
        float sum = 0.0f;
        float square_sum = 0.0f;
        for (uint32_t feature = 0; feature < width; feature++) {
            float value = input[offset + feature];
            sum += value;
            square_sum += value * value;
        }
        float mean = sum / (float)width;
        float variance = square_sum / (float)width - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        float inverse = 1.0f / sqrtf(variance + epsilon);
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;
        for (uint32_t feature = 0; feature < width; feature++) {
            float xhat = (input[offset + feature] - mean) * inverse;
            float scaled = grad_output[offset + feature] * weight[feature];
            sum_dy += scaled;
            sum_dy_xhat += scaled * xhat;
            grad_weight[feature] += grad_output[offset + feature] * xhat;
            grad_bias[feature] += grad_output[offset + feature];
        }
        for (uint32_t feature = 0; feature < width; feature++) {
            float xhat = (input[offset + feature] - mean) * inverse;
            float dx = inverse / (float)width *
                ((float)width * grad_output[offset + feature] * weight[feature] -
                 sum_dy - xhat * sum_dy_xhat);
            grad_input[offset + feature] += dx;
        }
    }
}

static int test_layernorm_backward(void) {
    enum { ROWS = 2, WIDTH = 3, AFFINE = 5, LENGTH = ROWS * WIDTH };
    const float input[LENGTH] = {1.0f, -0.5f, 2.0f, -1.0f, 0.25f, 0.75f};
    const float weight[AFFINE] = {0.5f, 1.25f, -0.75f, 9.0f, -7.0f};
    const float grad_output[LENGTH] = {1.0f, -2.0f, 0.5f, 0.25f, 1.5f, -1.0f};
    float grad_input[LENGTH];
    float grad_weight[AFFINE];
    float grad_bias[AFFINE];
    float expected_input[LENGTH];
    float expected_weight[AFFINE];
    float expected_bias[AFFINE];
    const float epsilon = 1.0e-5f;
    uint32_t params[4] = {ROWS, WIDTH, float_bits(epsilon), 1u};
    void* hosts[7] = {
        (void*)input, (void*)weight, (void*)grad_output, grad_input,
        grad_weight, grad_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++)
        expected_input[index] = grad_input[index] =
            ((float)(int)(index % 3u) - 1.0f) * 0.05f;
    for (uint32_t feature = 0; feature < AFFINE; feature++) {
        expected_weight[feature] = grad_weight[feature] =
            ((float)(int)feature - 1.0f) * 0.075f;
        expected_bias[feature] = grad_bias[feature] =
            ((float)(int)feature + 1.0f) * -0.025f;
    }
    layernorm_reference(input, weight, grad_output, expected_input,
                        expected_weight, expected_bias, ROWS, WIDTH, epsilon);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("layerNormBackward", "input_main", hosts,
                                 bytes, access, is_weight, 7, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("layerNormBackward", "param_main", hosts,
                                 bytes, access, is_weight, 7, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++)
        CHECK(fabsf(grad_input[index] - expected_input[index]) <= 2.0e-5f);
    for (uint32_t feature = 0; feature < AFFINE; feature++) {
        CHECK(fabsf(grad_weight[feature] - expected_weight[feature]) <= 2.0e-5f);
        CHECK(fabsf(grad_bias[feature] - expected_bias[feature]) <= 2.0e-6f);
    }
    return 0;
}

static int test_layernorm_param_cooperative(uint32_t rows, int has_bias) {
    enum { WIDTH = 7, AFFINE = 11, MAX_ROWS = 131,
           MAX_LENGTH = MAX_ROWS * WIDTH };
    float input[MAX_LENGTH];
    float grad_output[MAX_LENGTH];
    float weight[AFFINE];
    float initial_weight[AFFINE];
    float first_weight[AFFINE];
    float second_weight[AFFINE];
    float expected_weight[AFFINE];
    float initial_bias[AFFINE];
    float first_bias[AFFINE];
    float second_bias[AFFINE];
    float expected_bias[AFFINE];
    uint32_t dummy_bias[4] = {
        0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u,
    };
    const uint32_t expected_dummy_bias[4] = {
        0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u,
    };
    uint32_t params[4] = {rows, WIDTH, float_bits(1.0e-5f),
                          has_bias ? 1u : 0u};
    uint32_t bad_params[4];
    size_t length;
    void* hosts[7];
    size_t bytes[7];
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    CHECK(rows > 0u && rows <= MAX_ROWS);
    length = (size_t)rows * WIDTH;
    for (size_t index = 0u; index < length; ++index) {
        input[index] =
            ((float)(int)((index * 17u) % 43u) - 21.0f) * 0.03125f;
        grad_output[index] =
            ((float)(int)((index * 29u) % 47u) - 23.0f) * 0.015625f;
    }
    for (uint32_t feature = 0u; feature < AFFINE; ++feature) {
        weight[feature] =
            ((float)(int)((feature * 7u) % 13u) - 6.0f) * 0.125f;
        initial_weight[feature] =
            ((float)(int)feature - 4.0f) * 0.0078125f;
        initial_bias[feature] =
            ((float)(int)feature - 5.0f) * -0.00390625f;
    }
    memcpy(first_weight, initial_weight, sizeof(first_weight));
    memcpy(second_weight, initial_weight, sizeof(second_weight));
    memcpy(expected_weight, initial_weight, sizeof(expected_weight));
    memcpy(first_bias, initial_bias, sizeof(first_bias));
    memcpy(second_bias, initial_bias, sizeof(second_bias));
    memcpy(expected_bias, initial_bias, sizeof(expected_bias));
    {
        float unused_input[MAX_LENGTH] = {0};
        layernorm_reference(input, weight, grad_output, unused_input,
                            expected_weight, expected_bias, rows, WIDTH,
                            1.0e-5f);
    }

    hosts[0] = input;
    hosts[1] = weight;
    hosts[2] = grad_output;
    hosts[3] = input; /* Cooperative parameter backward never dereferences it. */
    hosts[4] = first_weight;
    hosts[5] = has_bias ? (void*)first_bias : (void*)dummy_bias;
    hosts[6] = params;
    bytes[0] = length * sizeof(float);
    bytes[1] = sizeof(weight);
    bytes[2] = length * sizeof(float);
    bytes[3] = length * sizeof(float);
    bytes[4] = sizeof(first_weight);
    bytes[5] = has_bias ? sizeof(first_bias) : sizeof(dummy_bias);
    bytes[6] = sizeof(params);

    CHECK(cuda_training_supports(
        "layerNormBackward", "param_cooperative", bytes, 7,
        WIDTH, 1u, 1u));
    CHECK(!cuda_training_supports(
        "layerNormBackward", "param_cooperative", bytes, 7,
        AFFINE + 1u, 1u, 1u));
    CHECK(!cuda_training_supports(
        "layerNormBackward", "param_cooperative", bytes, 7,
        (uint32_t)INT32_MAX + 1u, 1u, 1u));

    uint64_t launches = cuda_test_launch_count();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch(
        "layerNormBackward", "param_cooperative", hosts, bytes, access,
        is_weight, 7, WIDTH - 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(cuda_test_launch_count() == launches);
    CHECK(memcmp(first_weight, second_weight, sizeof(first_weight)) == 0);
    CHECK(memcmp(first_bias, second_bias, sizeof(first_bias)) == 0);

    for (uint32_t malformed = 0u; malformed < 3u; ++malformed) {
        memcpy(bad_params, params, sizeof(bad_params));
        if (malformed == 0u) bad_params[2] = float_bits(0.0f);
        else if (malformed == 1u) bad_params[3] = 2u;
        else {
            bad_params[0] = UINT32_MAX;
            bad_params[1] = 2u;
        }
        hosts[6] = bad_params;
        launches = cuda_test_launch_count();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch(
            "layerNormBackward", "param_cooperative", hosts, bytes, access,
            is_weight, 7, WIDTH, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(cuda_test_launch_count() == launches);
        CHECK(memcmp(first_weight, second_weight,
                     sizeof(first_weight)) == 0);
        CHECK(memcmp(first_bias, second_bias, sizeof(first_bias)) == 0);
    }
    hosts[6] = params;

    for (uint32_t run = 0u; run < 2u; ++run) {
        float* output_weight = run == 0u ? first_weight : second_weight;
        float* output_bias = run == 0u ? first_bias : second_bias;
        memcpy(output_weight, initial_weight, sizeof(first_weight));
        if (has_bias)
            memcpy(output_bias, initial_bias, sizeof(first_bias));
        hosts[4] = output_weight;
        hosts[5] = has_bias ? (void*)output_bias : (void*)dummy_bias;
        launches = cuda_test_launch_count();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch(
            "layerNormBackward", "param_cooperative", hosts, bytes, access,
            is_weight, 7, WIDTH, 1u, 1u) == 0);
        CHECK(cuda_test_launch_count() == launches + 1u);
        CHECK(cuda_training_sync(output_weight,
                                 sizeof(first_weight)) == 0);
        if (has_bias)
            CHECK(cuda_training_sync(output_bias, sizeof(first_bias)) == 0);
        cuda_training_end();
        if (run == 0u) cuda_graph_reset();
    }

    CHECK(memcmp(first_weight, second_weight, sizeof(first_weight)) == 0);
    if (has_bias)
        CHECK(memcmp(first_bias, second_bias, sizeof(first_bias)) == 0);
    else
        CHECK(memcmp(dummy_bias, expected_dummy_bias,
                     sizeof(dummy_bias)) == 0);
    for (uint32_t feature = 0u; feature < WIDTH; ++feature) {
        float weight_scale = fmaxf(
            1.0f, fmaxf(fabsf(first_weight[feature]),
                         fabsf(expected_weight[feature])));
        CHECK(fabsf(first_weight[feature] - expected_weight[feature]) <=
              5.0e-5f * weight_scale);
        if (has_bias) {
            float bias_scale = fmaxf(
                1.0f, fmaxf(fabsf(first_bias[feature]),
                             fabsf(expected_bias[feature])));
            CHECK(fabsf(first_bias[feature] - expected_bias[feature]) <=
                  5.0e-5f * bias_scale);
        }
    }
    for (uint32_t feature = WIDTH; feature < AFFINE; ++feature) {
        CHECK(first_weight[feature] == initial_weight[feature]);
        if (has_bias)
            CHECK(first_bias[feature] == initial_bias[feature]);
    }
    return 0;
}

static void rmsnorm_reference(const float* input, const float* weight,
                              const float* grad_output, float* grad_input,
                              float* grad_weight, uint32_t rows,
                              uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t offset = row * width;
        float square_sum = 0.0f;
        for (uint32_t feature = 0; feature < width; feature++)
            square_sum += input[offset + feature] * input[offset + feature];
        float inverse = 1.0f / sqrtf(square_sum / (float)width + epsilon);
        float dot = 0.0f;
        for (uint32_t feature = 0; feature < width; feature++) {
            dot += grad_output[offset + feature] * weight[feature] *
                input[offset + feature];
            grad_weight[feature] += grad_output[offset + feature] *
                input[offset + feature] * inverse;
        }
        for (uint32_t feature = 0; feature < width; feature++)
            grad_input[offset + feature] +=
                grad_output[offset + feature] * weight[feature] * inverse -
                input[offset + feature] * inverse * inverse * inverse * dot /
                    (float)width;
    }
}

static int test_rmsnorm_backward(void) {
    enum { ROWS = 2, WIDTH = 3, AFFINE = 5, LENGTH = ROWS * WIDTH };
    const float input[LENGTH] = {1.0f, -0.5f, 2.0f, -1.0f, 0.25f, 0.75f};
    const float weight[AFFINE] = {0.5f, 1.25f, -0.75f, 9.0f, -7.0f};
    const float grad_output[LENGTH] = {1.0f, -2.0f, 0.5f, 0.25f, 1.5f, -1.0f};
    float grad_input[LENGTH];
    float grad_weight[AFFINE];
    float expected_input[LENGTH];
    float expected_weight[AFFINE];
    const float epsilon = 1.0e-6f;
    uint32_t params[4] = {ROWS, WIDTH, float_bits(epsilon), 0u};
    void* hosts[6] = {
        (void*)input, (void*)weight, (void*)grad_output, grad_input,
        grad_weight, params,
    };
    size_t bytes[6] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 1u, 0u, 0u, 0u, 0u};

    for (uint32_t index = 0; index < LENGTH; index++)
        expected_input[index] = grad_input[index] =
            ((float)(int)(index % 3u) - 1.0f) * 0.05f;
    for (uint32_t feature = 0; feature < AFFINE; feature++)
        expected_weight[feature] = grad_weight[feature] =
            ((float)(int)feature - 1.0f) * 0.075f;
    rmsnorm_reference(input, weight, grad_output, expected_input,
                      expected_weight, ROWS, WIDTH, epsilon);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("rmsNormBackward", "input_main", hosts,
                                 bytes, access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("rmsNormBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 6, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    cuda_training_end();
    for (uint32_t index = 0; index < LENGTH; index++)
        CHECK(fabsf(grad_input[index] - expected_input[index]) <= 2.0e-5f);
    for (uint32_t feature = 0; feature < AFFINE; feature++)
        CHECK(fabsf(grad_weight[feature] - expected_weight[feature]) <= 2.0e-5f);
    return 0;
}

static float conv2d_gated_gradient_reference(
        const float* output, const float* grad_output, uint32_t index,
        uint32_t relu) {
    float value = output[index];
    if (relu == 1u && value <= 0.0f) return 0.0f;
    if (relu >= 2u && (value <= 0.0f || value >= 6.0f)) return 0.0f;
    return grad_output[index];
}

static uint32_t conv2d_weight_index_reference(
        uint32_t kernel_y, uint32_t kernel_x, uint32_t input_channel,
        uint32_t output_channel, const uint32_t params[20]) {
    uint32_t in_per_group = params[3] / params[13];
    uint32_t out_per_group = params[6] / params[13];
    uint32_t group = output_channel / out_per_group;
    uint32_t local_input = input_channel - group * in_per_group;
    if (params[19] == 0u)
        return (((kernel_y * params[8] + kernel_x) * in_per_group +
            local_input) * params[6]) + output_channel;
    if (params[19] == 1u) {
        uint32_t multiplier = params[6] / params[3];
        uint32_t multiplier_index =
            output_channel - input_channel * multiplier;
        return (((kernel_y * params[8] + kernel_x) * params[3] +
            input_channel) * multiplier) + multiplier_index;
    }
    if (params[19] == 2u)
        return (((output_channel * params[7] + kernel_y) * params[8] +
            kernel_x) * in_per_group) + local_input;
    return (kernel_y * params[8] + kernel_x) * params[6] + output_channel;
}

/* Deliberately scatter from each output element. The CUDA kernels use three
 * separate destination-owned gathers, so this provides an independent layout
 * and traversal reference for input, weight, and bias gradients. */
static void conv2d_backward_reference(
        const float* input, const float* weight, const float* output,
        const float* grad_output, float* grad_input, float* grad_weight,
        float* grad_bias, const uint32_t params[20]) {
    uint32_t in_per_group = params[3] / params[13];
    uint32_t out_per_group = params[6] / params[13];
    uint32_t multiplier = params[6] / params[3];
    int depthwise = params[19] == 1u || params[19] == 3u;
    for (uint32_t batch = 0u; batch < params[0]; batch++) {
        for (uint32_t out_y = 0u; out_y < params[4]; out_y++) {
            for (uint32_t out_x = 0u; out_x < params[5]; out_x++) {
                for (uint32_t out_channel = 0u;
                     out_channel < params[6]; out_channel++) {
                    uint32_t output_index =
                        ((batch * params[4] + out_y) * params[5] + out_x) *
                            params[6] + out_channel;
                    float gradient = conv2d_gated_gradient_reference(
                        output, grad_output, output_index, params[16]);
                    grad_bias[out_channel] += gradient;
                    for (uint32_t kernel_y = 0u;
                         kernel_y < params[7]; kernel_y++) {
                        int input_y = (int)(out_y * params[9] +
                            kernel_y * params[14]) - (int)params[11];
                        if (input_y < 0 || input_y >= (int)params[1]) continue;
                        for (uint32_t kernel_x = 0u;
                             kernel_x < params[8]; kernel_x++) {
                            int input_x = (int)(out_x * params[10] +
                                kernel_x * params[15]) - (int)params[12];
                            if (input_x < 0 || input_x >= (int)params[2])
                                continue;
                            if (depthwise) {
                                uint32_t input_channel =
                                    out_channel / multiplier;
                                uint32_t input_index = ((batch * params[1] +
                                    (uint32_t)input_y) * params[2] +
                                    (uint32_t)input_x) * params[3] +
                                    input_channel;
                                uint32_t weight_index =
                                    conv2d_weight_index_reference(
                                        kernel_y, kernel_x, input_channel,
                                        out_channel, params);
                                grad_input[input_index] +=
                                    gradient * weight[weight_index];
                                grad_weight[weight_index] +=
                                    input[input_index] * gradient;
                            } else {
                                uint32_t group = out_channel / out_per_group;
                                for (uint32_t local_input = 0u;
                                     local_input < in_per_group;
                                     local_input++) {
                                    uint32_t input_channel =
                                        group * in_per_group + local_input;
                                    uint32_t input_index =
                                        ((batch * params[1] +
                                          (uint32_t)input_y) * params[2] +
                                         (uint32_t)input_x) * params[3] +
                                        input_channel;
                                    uint32_t weight_index =
                                        conv2d_weight_index_reference(
                                            kernel_y, kernel_x,
                                            input_channel, out_channel,
                                            params);
                                    grad_input[input_index] +=
                                        gradient * weight[weight_index];
                                    grad_weight[weight_index] +=
                                        input[input_index] * gradient;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static int conv2d_close(float actual, float expected) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    return fabsf(actual - expected) <= 4.0e-5f * scale;
}

static int test_conv2d_backward_layout(uint32_t layout,
                                       uint32_t hwio_stride) {
    enum {
        MAX_INPUT = 5 * 5 * 3,
        MAX_OUTPUT = 5 * 5 * 4,
        MAX_WEIGHT = 3 * 3 * 3 * 4,
        MAX_BIAS = 6,
    };
    float input[MAX_INPUT];
    float weight[MAX_WEIGHT];
    float output[MAX_OUTPUT];
    float grad_output[MAX_OUTPUT];
    float grad_input[MAX_INPUT];
    float grad_weight[MAX_WEIGHT];
    float grad_bias[MAX_BIAS];
    float expected_input[MAX_INPUT];
    float expected_weight[MAX_WEIGHT];
    float expected_bias[MAX_BIAS];
    uint32_t params[20] = {0};
    uint32_t input_elements;
    uint32_t output_elements;
    uint32_t weight_elements;
    uint32_t input_groups;
    uint32_t weight_groups;
    uint32_t bias_groups;
    uint64_t tiled_input_launches;
    uint64_t tiled_c32_input_launches;
    uint64_t tiled_weight_launches;
    int blocked = 0;
    int passed = 0;

    CHECK(layout <= 3u && (hwio_stride == 1u || hwio_stride == 2u));
    if (layout == 1u || layout == 3u) {
        /* NHWC input, HWCM weight, multiplier=3. Stride and top padding make
         * this exercise the signed source-coordinate path too. */
        const uint32_t configured[20] = {
            1u, 4u, 3u, 2u, 3u, 2u, 6u, 2u, 2u, 2u,
            1u, 1u, 0u, 2u, 1u, 1u, 2u, 1u, 24u, 0u,
        };
        memcpy(params, configured, sizeof(params));
        params[19] = layout;
    } else {
        /* NHWC input and HWIO weights. Both input and weight destinations
         * cross the 64-thread block boundary. The 25 spatial rows and the
         * 27-element 3x3xC reduction both tail a 16x16 tiled launch. */
        uint32_t configured[20] = {
            1u, 5u, 5u, 3u, 5u, 5u, 4u, 3u, 3u, 1u,
            1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 108u, 0u,
        };
        if (layout == 0u && hwio_stride == 2u) {
            configured[4] = configured[5] = 3u;
            configured[9] = configured[10] = 2u;
        }
        memcpy(params, configured, sizeof(params));
        params[19] = layout;
    }
    input_elements = params[0] * params[1] * params[2] * params[3];
    output_elements = params[0] * params[4] * params[5] * params[6];
    weight_elements = params[18];
    CHECK(input_elements <= MAX_INPUT);
    CHECK(output_elements <= MAX_OUTPUT);
    CHECK(weight_elements <= MAX_WEIGHT);
    CHECK(params[6] <= MAX_BIAS);

    for (uint32_t index = 0u; index < input_elements; index++)
        input[index] = ((float)(int)(index % 17u) - 8.0f) * 0.125f;
    for (uint32_t index = 0u; index < weight_elements; index++)
        weight[index] = ((float)(int)(index % 11u) - 5.0f) * 0.0625f;
    for (uint32_t index = 0u; index < output_elements; index++) {
        switch (index % 6u) {
            case 0u: output[index] = -1.0f; break;
            case 1u: output[index] = 0.0f; break;
            case 2u: output[index] = 0.25f; break;
            case 3u: output[index] = 1.5f; break;
            case 4u: output[index] = 6.0f; break;
            default: output[index] = 7.0f; break;
        }
        grad_output[index] = (index & 1u ? -1.0f : 1.0f) *
            (float)(index % 7u + 1u) * 0.1f;
        if ((params[16] == 1u && output[index] <= 0.0f) ||
            (params[16] >= 2u &&
             (output[index] <= 0.0f || output[index] >= 6.0f)))
            blocked++;
        else
            passed++;
    }
    CHECK(blocked > 0 && passed > 0);
    for (uint32_t index = 0u; index < input_elements; index++)
        expected_input[index] = grad_input[index] =
            ((float)(int)(index % 5u) - 2.0f) * 0.03125f;
    for (uint32_t index = 0u; index < weight_elements; index++)
        expected_weight[index] = grad_weight[index] =
            ((float)(int)(index % 7u) - 3.0f) * 0.015625f;
    for (uint32_t index = 0u; index < params[6]; index++)
        expected_bias[index] = grad_bias[index] =
            ((float)(int)index - 2.0f) * 0.025f;
    conv2d_backward_reference(input, weight, output, grad_output,
                              expected_input, expected_weight, expected_bias,
                              params);

    void* hosts[8] = {
        input, weight, output, grad_output, grad_input, grad_weight, grad_bias,
        params,
    };
    size_t bytes[8] = {
        (size_t)input_elements * sizeof(float),
        (size_t)weight_elements * sizeof(float),
        (size_t)output_elements * sizeof(float),
        (size_t)output_elements * sizeof(float),
        (size_t)input_elements * sizeof(float),
        (size_t)weight_elements * sizeof(float),
        (size_t)params[6] * sizeof(float), sizeof(params),
    };
    const unsigned char access[8] = {1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[8] = {0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u};
    input_groups = (input_elements + 63u) / 64u;
    weight_groups = (weight_elements + 63u) / 64u;
    bias_groups = (params[6] + 63u) / 64u;
    tiled_input_launches =
        cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count();
    tiled_c32_input_launches =
        cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count();
    tiled_weight_launches =
        cuda_test_training_conv2d_weight_hwio_3x3_tiled_launch_count();

    CHECK(cuda_training_supports("conv2DBackward", "input_main", bytes, 8,
                                 input_groups, 1u, 1u));
    CHECK(cuda_training_supports("conv2DBackward", "weight_main", bytes, 8,
                                 weight_groups, 1u, 1u));
    CHECK(cuda_training_supports("conv2DBackward", "bias_main", bytes, 8,
                                 bias_groups, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                 bytes, access, is_weight, 8, input_groups,
                                 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 8, weight_groups,
                                 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "bias_main", hosts,
                                 bytes, access, is_weight, 8, bias_groups,
                                 1u, 1u) == 0);
    CHECK(cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count() ==
          tiled_input_launches + (layout == 0u ? 1u : 0u));
    CHECK(cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count() ==
          tiled_c32_input_launches);
    CHECK(cuda_test_training_conv2d_weight_hwio_3x3_tiled_launch_count() ==
          tiled_weight_launches + (layout == 0u ? 1u : 0u));
    CHECK(cuda_training_sync(grad_input, bytes[4]) == 0);
    CHECK(cuda_training_sync(grad_weight, bytes[5]) == 0);
    CHECK(cuda_training_sync(grad_bias, bytes[6]) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < input_elements; index++)
        CHECK(conv2d_close(grad_input[index], expected_input[index]));
    for (uint32_t index = 0u; index < weight_elements; index++)
        CHECK(conv2d_close(grad_weight[index], expected_weight[index]));
    for (uint32_t index = 0u; index < params[6]; index++)
        CHECK(conv2d_close(grad_bias[index], expected_bias[index]));
    return 0;
}

static int test_conv2d_backward_input_hwio_c32(uint32_t input_channels) {
    enum {
        HEIGHT = 5,
        WIDTH = 7,
        OUTPUT_CHANNELS = 17,
        MAX_INPUT_CHANNELS = 48,
        MAX_INPUT = HEIGHT * WIDTH * MAX_INPUT_CHANNELS,
        MAX_OUTPUT = HEIGHT * WIDTH * OUTPUT_CHANNELS,
        MAX_WEIGHT = 3 * 3 * MAX_INPUT_CHANNELS * OUTPUT_CHANNELS,
    };
    float input[MAX_INPUT];
    float weight[MAX_WEIGHT];
    float output[MAX_OUTPUT];
    float grad_output[MAX_OUTPUT];
    float initial_input[MAX_INPUT];
    float first_input[MAX_INPUT];
    float second_input[MAX_INPUT];
    float expected_input[MAX_INPUT];
    float scratch_weight[MAX_WEIGHT] = {0};
    float scratch_bias[OUTPUT_CHANNELS] = {0};
    uint32_t params[20] = {
        1u, HEIGHT, WIDTH, 0u, HEIGHT, WIDTH, OUTPUT_CHANNELS,
        3u, 3u, 1u, 1u, 1u, 1u, 1u, 1u, 1u,
        2u, 1u, 0u, 0u,
    };
    uint32_t input_elements;
    uint32_t output_elements = MAX_OUTPUT;
    uint32_t weight_elements;
    uint32_t public_groups;
    uint64_t tiled_launches;
    uint64_t c32_launches;

    CHECK(input_channels >= 32u && input_channels <= MAX_INPUT_CHANNELS);
    params[3] = input_channels;
    params[18] = 3u * 3u * input_channels * OUTPUT_CHANNELS;
    input_elements = HEIGHT * WIDTH * input_channels;
    weight_elements = params[18];
    public_groups = (input_elements + 63u) / 64u;
    for (uint32_t index = 0u; index < input_elements; ++index) {
        input[index] =
            ((float)(int)((index * 13u) % 37u) - 18.0f) * 0.03125f;
        initial_input[index] =
            ((float)(int)(index % 11u) - 5.0f) * 0.00390625f;
    }
    for (uint32_t index = 0u; index < weight_elements; ++index)
        weight[index] =
            ((float)(int)((index * 7u) % 29u) - 14.0f) * 0.015625f;
    for (uint32_t index = 0u; index < output_elements; ++index) {
        switch (index % 6u) {
            case 0u: output[index] = -1.0f; break;
            case 1u: output[index] = 0.0f; break;
            case 2u: output[index] = 0.25f; break;
            case 3u: output[index] = 1.5f; break;
            case 4u: output[index] = 6.0f; break;
            default: output[index] = 7.0f; break;
        }
        grad_output[index] =
            ((float)(int)((index * 19u) % 31u) - 15.0f) * 0.025f;
    }
    memcpy(first_input, initial_input,
           (size_t)input_elements * sizeof(float));
    memcpy(second_input, initial_input,
           (size_t)input_elements * sizeof(float));
    memcpy(expected_input, initial_input,
           (size_t)input_elements * sizeof(float));
    conv2d_backward_reference(
        input, weight, output, grad_output, expected_input, scratch_weight,
        scratch_bias, params);

    void* hosts[8] = {
        input, weight, output, grad_output, first_input, scratch_weight,
        scratch_bias, params,
    };
    size_t bytes[8] = {
        (size_t)input_elements * sizeof(float),
        (size_t)weight_elements * sizeof(float), sizeof(output),
        sizeof(grad_output), (size_t)input_elements * sizeof(float),
        (size_t)weight_elements * sizeof(float), sizeof(scratch_bias),
        sizeof(params),
    };
    const unsigned char access[8] = {
        1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[8] = {
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u,
    };

    CHECK(cuda_training_supports("conv2DBackward", "input_main", bytes, 8,
                                 public_groups, 1u, 1u));
    tiled_launches =
        cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count();
    c32_launches =
        cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count();
    uint64_t launches = cuda_test_launch_count();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                 bytes, access, is_weight, 8,
                                 public_groups - 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(cuda_test_launch_count() == launches);
    CHECK(memcmp(first_input, initial_input,
                 (size_t)input_elements * sizeof(float)) == 0);

    for (uint32_t run = 0u; run < 2u; ++run) {
        float* destination = run == 0u ? first_input : second_input;
        memcpy(destination, initial_input,
               (size_t)input_elements * sizeof(float));
        hosts[4] = destination;
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                     bytes, access, is_weight, 8,
                                     public_groups, 1u, 1u) == 0);
        CHECK(cuda_training_sync(destination, bytes[4]) == 0);
        cuda_training_end();
        CHECK(cuda_test_training_conv2d_input_hwio_3x3_tiled_launch_count() ==
              tiled_launches + run + 1u);
        CHECK(cuda_test_training_conv2d_input_hwio_3x3_tiled_c32_launch_count() ==
              c32_launches + run + 1u);
        if (run == 0u) cuda_graph_reset();
    }
    CHECK(memcmp(first_input, second_input,
                 (size_t)input_elements * sizeof(float)) == 0);
    for (uint32_t index = 0u; index < input_elements; ++index)
        CHECK(conv2d_close(first_input[index], expected_input[index]));
    return 0;
}

static int test_conv2d_backward_descriptors(void) {
    float input[8] = {0};
    float weight[4] = {0};
    float output[8] = {0};
    float grad_output[8] = {0};
    float grad_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float grad_weight[4] = {-1, -2, -3, -4};
    float grad_bias[2] = {0.25f, -0.5f};
    const float original_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float original_weight[4] = {-1, -2, -3, -4};
    const float original_bias[2] = {0.25f, -0.5f};
    uint32_t params[20] = {
        1u, 2u, 2u, 2u, 2u, 2u, 2u, 1u, 1u, 1u,
        1u, 0u, 0u, 1u, 1u, 1u, 0u, 1u, 4u, 0u,
    };
    void* hosts[8] = {
        input, weight, output, grad_output, grad_input, grad_weight, grad_bias,
        params,
    };
    size_t bytes[8] = {
        sizeof(input), sizeof(weight), sizeof(output), sizeof(grad_output),
        sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
        sizeof(params),
    };
    const unsigned char access[8] = {1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char bad_access[8] = {1u, 1u, 1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[8] = {0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u};
    size_t malformed_bytes[8];

    memcpy(malformed_bytes, bytes, sizeof(bytes));
    malformed_bytes[7] -= sizeof(uint32_t);
    CHECK(!cuda_training_supports("conv2DBackward", "input_main",
                                  malformed_bytes, 8, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("conv2DBackward", "invalid", bytes, 8,
                                  1u, 1u, 1u));
    CHECK(!cuda_training_supports("conv2DBackward", "input_main", bytes, 7,
                                  1u, 1u, 1u));
    CHECK(!cuda_training_supports("conv2DBackward", "input_main", bytes, 8,
                                  1u, 2u, 1u));

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                 bytes, bad_access, is_weight, 8,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
    CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
    CHECK(memcmp(grad_bias, original_bias, sizeof(grad_bias)) == 0);

    params[18] = 3u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 8,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    params[18] = 4u;
    CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
    CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
    CHECK(memcmp(grad_bias, original_bias, sizeof(grad_bias)) == 0);

    params[16] = 3u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "bias_main", hosts,
                                 bytes, access, is_weight, 8,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    params[16] = 0u;
    CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
    CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
    CHECK(memcmp(grad_bias, original_bias, sizeof(grad_bias)) == 0);

    params[19] = 4u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                 bytes, access, is_weight, 8,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    params[19] = 0u;
    CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
    CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
    CHECK(memcmp(grad_bias, original_bias, sizeof(grad_bias)) == 0);

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("conv2DBackward", "input_main", hosts,
                                 bytes, access, is_weight, 8,
                                 2u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
    CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
    CHECK(memcmp(grad_bias, original_bias, sizeof(grad_bias)) == 0);
    return 0;
}

static int test_transpose_backward(void) {
    const float grad_output[6] = {1, 2, 3, 4, 5, 6};
    float grad_input[6] = {
        0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
    };
    const float expected[6] = {
        1.25f, 3.25f, 5.25f, 2.25f, 4.25f, 6.25f,
    };
    uint32_t params[34] = {0};
    void* hosts[3] = {(void*)grad_output, grad_input, params};
    size_t bytes[3] = {
        sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    params[0] = 2u;
    params[1] = 6u;
    for (uint32_t dimension = 0u; dimension < 8u; dimension++)
        params[2u + dimension] = 1u;
    params[2] = 2u;
    params[3] = 3u;
    params[10] = 1u;
    params[11] = 0u;
    params[18] = 3u;
    params[19] = 1u;
    params[26] = 2u;
    params[27] = 1u;
    CHECK(cuda_training_supports("transposeBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("transposeBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < 6u; index++)
        CHECK(grad_input[index] == expected[index]);
    return 0;
}

static int test_concat_backward(void) {
    float grad_output[20];
    float grad_input[8];
    const float added[8] = {3, 4, 5, 6, 13, 14, 15, 16};
    uint32_t params[8] = {8u, 1u, 2u, 5u, 2u, 0u, 0u, 0u};
    void* hosts[3] = {grad_output, grad_input, params};
    size_t bytes[3] = {
        sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    for (uint32_t index = 0u; index < 20u; index++)
        grad_output[index] = (float)(index + 1u);
    for (uint32_t index = 0u; index < 8u; index++) grad_input[index] = 0.5f;
    CHECK(cuda_training_supports("concatBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("concatBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < 8u; index++)
        CHECK(grad_input[index] == 0.5f + added[index]);
    return 0;
}

static int test_split_backward(void) {
    float grad_output[8];
    float grad_input[20];
    float expected[20];
    uint32_t params[5] = {8u, 2u, 2u, 5u, 1u};
    void* hosts[3] = {grad_output, grad_input, params};
    size_t bytes[3] = {
        sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    for (uint32_t index = 0u; index < 8u; index++)
        grad_output[index] = (float)(index + 1u);
    for (uint32_t index = 0u; index < 20u; index++)
        expected[index] = grad_input[index] = -0.25f;
    for (uint32_t index = 0u; index < 8u; index++) {
        uint32_t outer = index / 4u;
        uint32_t within = index - outer * 4u;
        expected[(outer * 5u + 1u) * 2u + within] += grad_output[index];
    }
    CHECK(cuda_training_supports("splitBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("splitBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    CHECK(memcmp(grad_input, expected, sizeof(grad_input)) == 0);
    return 0;
}

static int test_resize_backward_nearest(void) {
    const float grad_output[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float grad_input[4] = {0.125f, 0.125f, 0.125f, 0.125f};
    const float expected[4] = {12.125f, 9.125f, 15.125f, 9.125f};
    uint32_t params[8] = {1u, 2u, 2u, 1u, 3u, 3u, 0u, 0u};
    void* hosts[3] = {(void*)grad_output, grad_input, params};
    size_t bytes[3] = {
        sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    CHECK(cuda_training_supports("resizeBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("resizeBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    for (uint32_t index = 0u; index < 4u; index++)
        CHECK(grad_input[index] == expected[index]);
    return 0;
}

static int test_resize_backward_bilinear_boundary(void) {
    const float grad_output[4] = {1, 2, 3, 4};
    float grad_input[1] = {-0.5f};
    uint32_t params[8] = {1u, 1u, 1u, 1u, 2u, 2u, 1u, 0u};
    void* hosts[3] = {(void*)grad_output, grad_input, params};
    size_t bytes[3] = {
        sizeof(grad_output), sizeof(grad_input), sizeof(params),
    };
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    CHECK(cuda_training_supports("resizeBackward", "main", bytes, 3,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("resizeBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    cuda_training_end();
    CHECK(grad_input[0] == 9.5f);
    return 0;
}

static int test_pooling_backward(void) {
    {
        float input[12] = {0};
        const float grad_output[2] = {12.0f, -6.0f};
        float grad_input[12];
        uint32_t params[12] = {1u, 2u, 3u, 2u};
        void* hosts[4] = {
            input, (void*)grad_output, grad_input, params,
        };
        size_t bytes[4] = {
            sizeof(input), sizeof(grad_output), sizeof(grad_input),
            sizeof(params),
        };
        const unsigned char access[4] = {1u, 1u, 3u, 1u};
        const unsigned char is_weight[4] = {0u, 0u, 0u, 0u};

        for (uint32_t index = 0u; index < 12u; index++)
            grad_input[index] = 0.75f;
        CHECK(cuda_training_supports("poolingBackward",
                                     "global_average_main", bytes, 4,
                                     1u, 1u, 1u));
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("poolingBackward",
                                     "global_average_main", hosts, bytes,
                                     access, is_weight, 4,
                                     1u, 1u, 1u) == 0);
        CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
        cuda_training_end();
        for (uint32_t index = 0u; index < 12u; index++)
            CHECK(grad_input[index] ==
                  0.75f + (index % 2u == 0u ? 2.0f : -1.0f));
    }
    {
        const float input[9] = {5, 5, 1, 5, 5, 2, 3, 4, 5};
        const float grad_output[4] = {1, 2, 3, 4};
        float grad_input[9];
        const float added[9] = {1, 2, 0, 3, 4, 0, 0, 0, 0};
        uint32_t params[12] = {
            1u, 3u, 3u, 1u, 2u, 2u, 2u, 2u, 1u, 1u, 0u, 0u,
        };
        void* hosts[4] = {
            (void*)input, (void*)grad_output, grad_input, params,
        };
        size_t bytes[4] = {
            sizeof(input), sizeof(grad_output), sizeof(grad_input),
            sizeof(params),
        };
        const unsigned char access[4] = {1u, 1u, 3u, 1u};
        const unsigned char is_weight[4] = {0u, 0u, 0u, 0u};

        for (uint32_t index = 0u; index < 9u; index++)
            grad_input[index] = -0.5f;
        CHECK(cuda_training_supports("poolingBackward", "max_pool_main",
                                     bytes, 4, 1u, 1u, 1u));
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("poolingBackward", "max_pool_main",
                                     hosts, bytes, access, is_weight, 4,
                                     1u, 1u, 1u) == 0);
        CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
        cuda_training_end();
        for (uint32_t index = 0u; index < 9u; index++)
            CHECK(grad_input[index] == -0.5f + added[index]);
    }
    return 0;
}

static int test_shape_backward_descriptors(void) {
    const unsigned char access3[3] = {1u, 3u, 1u};
    const unsigned char is_weight3[3] = {0u, 0u, 0u};
    const unsigned char access4[4] = {1u, 1u, 3u, 1u};
    const unsigned char is_weight4[4] = {0u, 0u, 0u, 0u};

    {
        float grad_output[6] = {0};
        float grad_input[6] = {1, 2, 3, 4, 5, 6};
        const float original[6] = {1, 2, 3, 4, 5, 6};
        uint32_t params[34] = {0};
        void* hosts[3] = {grad_output, grad_input, params};
        size_t bytes[3] = {
            sizeof(grad_output), sizeof(grad_input), sizeof(params),
        };
        size_t malformed[3] = {bytes[0], bytes[1], bytes[2] - 4u};
        params[0] = 2u;
        params[1] = 6u;
        for (uint32_t dimension = 0u; dimension < 8u; dimension++)
            params[2u + dimension] = 1u;
        params[2] = 2u;
        params[3] = 3u;
        params[10] = 1u;
        params[11] = 1u; /* Duplicate permutation entry. */
        params[18] = 3u;
        params[19] = 1u;
        params[26] = 2u;
        params[27] = 1u;
        CHECK(!cuda_training_supports("transposeBackward", "main",
                                      malformed, 3, 1u, 1u, 1u));
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("transposeBackward", "main", hosts,
                                     bytes, access3, is_weight3, 3,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    {
        float grad_output[20] = {0};
        float grad_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        const float original[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        uint32_t params[8] = {8u, 4u, 2u, 5u, 2u, 0u, 0u, 0u};
        void* hosts[3] = {grad_output, grad_input, params};
        size_t bytes[3] = {
            sizeof(grad_output), sizeof(grad_input), sizeof(params),
        };
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("concatBackward", "main", hosts,
                                     bytes, access3, is_weight3, 3,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    {
        float grad_output[8] = {0};
        float grad_input[20] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        };
        const float original[20] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        };
        uint32_t params[5] = {8u, 2u, 2u, 5u, 4u};
        void* hosts[3] = {grad_output, grad_input, params};
        size_t bytes[3] = {
            sizeof(grad_output), sizeof(grad_input), sizeof(params),
        };
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("splitBackward", "main", hosts, bytes,
                                     access3, is_weight3, 3,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    {
        float grad_output[9] = {0};
        float grad_input[4] = {1, 2, 3, 4};
        const float original[4] = {1, 2, 3, 4};
        uint32_t params[8] = {1u, 2u, 2u, 1u, 3u, 3u, 2u, 0u};
        void* hosts[3] = {grad_output, grad_input, params};
        size_t bytes[3] = {
            sizeof(grad_output), sizeof(grad_input), sizeof(params),
        };
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("resizeBackward", "main", hosts, bytes,
                                     access3, is_weight3, 3,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    {
        float input[12] = {0};
        float grad_output[2] = {0};
        float grad_input[12] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        };
        const float original[12] = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        };
        uint32_t params[12] = {1u, 2u, 3u, 2u, 1u};
        void* hosts[4] = {input, grad_output, grad_input, params};
        size_t bytes[4] = {
            sizeof(input), sizeof(grad_output), sizeof(grad_input),
            sizeof(params),
        };
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("poolingBackward",
                                     "global_average_main", hosts, bytes,
                                     access4, is_weight4, 4,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    {
        float input[9] = {0};
        float grad_output[4] = {0};
        float grad_input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        const float original[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        uint32_t params[12] = {
            1u, 3u, 3u, 1u, 2u, 2u, 2u, 2u, 0u, 1u, 0u, 0u,
        };
        void* hosts[4] = {input, grad_output, grad_input, params};
        size_t bytes[4] = {
            sizeof(input), sizeof(grad_output), sizeof(grad_input),
            sizeof(params),
        };
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("poolingBackward", "max_pool_main",
                                     hosts, bytes, access4, is_weight4, 4,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original, sizeof(grad_input)) == 0);
    }
    return 0;
}

typedef struct AttentionReference {
    const float* qkv;
    const float* query;
    const float* key;
    const float* value;
    const int32_t* mask;
    uint32_t batch;
    uint32_t query_sequence;
    uint32_t key_sequence;
    uint32_t d_model;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
    uint32_t causal;
    uint32_t mask_mode;
    uint32_t dropout_threshold;
    uint32_t dropout_seed;
    uint32_t dropout_counter;
    float dropout_scale;
    int packed_qkv;
} AttentionReference;

static size_t attention_q_index(const AttentionReference* reference,
                                uint32_t batch, uint32_t query,
                                uint32_t head, uint32_t dimension) {
    size_t row = (size_t)batch * reference->query_sequence + query;
    size_t offset = (size_t)head * reference->head_dim + dimension;
    return reference->packed_qkv
        ? row * reference->d_model * 3u + offset
        : row * reference->d_model + offset;
}

static size_t attention_k_index(const AttentionReference* reference,
                                uint32_t batch, uint32_t key,
                                uint32_t head, uint32_t dimension) {
    size_t row = (size_t)batch * reference->key_sequence + key;
    size_t offset = (size_t)head * reference->head_dim + dimension;
    return reference->packed_qkv
        ? row * reference->d_model * 3u + reference->d_model + offset
        : row * reference->d_model + offset;
}

static size_t attention_v_index(const AttentionReference* reference,
                                uint32_t batch, uint32_t key,
                                uint32_t head, uint32_t dimension) {
    size_t row = (size_t)batch * reference->key_sequence + key;
    size_t offset = (size_t)head * reference->head_dim + dimension;
    return reference->packed_qkv
        ? row * reference->d_model * 3u + reference->d_model * 2u + offset
        : row * reference->d_model + offset;
}

static float attention_q_value(const AttentionReference* reference,
                               uint32_t batch, uint32_t query,
                               uint32_t head, uint32_t dimension) {
    size_t index = attention_q_index(reference, batch, query, head, dimension);
    return reference->packed_qkv
        ? reference->qkv[index] : reference->query[index];
}

static float attention_k_value(const AttentionReference* reference,
                               uint32_t batch, uint32_t key,
                               uint32_t head, uint32_t dimension) {
    size_t index = attention_k_index(reference, batch, key, head, dimension);
    return reference->packed_qkv
        ? reference->qkv[index] : reference->key[index];
}

static float attention_v_value(const AttentionReference* reference,
                               uint32_t batch, uint32_t key,
                               uint32_t head, uint32_t dimension) {
    size_t index = attention_v_index(reference, batch, key, head, dimension);
    return reference->packed_qkv
        ? reference->qkv[index] : reference->value[index];
}

static int attention_key_allowed(const AttentionReference* reference,
                                 uint32_t batch, uint32_t query,
                                 uint32_t key) {
    size_t index;
    if (reference->causal && key > query) return 0;
    if (!reference->mask_mode) return 1;
    index = key;
    if (reference->mask_mode == 2u)
        index = (size_t)batch * reference->key_sequence + key;
    else if (reference->mask_mode == 3u)
        index = (size_t)query * reference->key_sequence + key;
    else if (reference->mask_mode == 4u)
        index = ((size_t)batch * reference->query_sequence + query) *
            reference->key_sequence + key;
    return reference->mask[index] != 0;
}

static float attention_dropout_multiplier(
        const AttentionReference* reference, uint32_t batch, uint32_t head,
        uint32_t query, uint32_t key) {
    uint32_t index = (uint32_t)((((uint64_t)batch * reference->heads + head) *
        reference->query_sequence + query) * reference->key_sequence + key);
    return dropout_bits(reference->dropout_seed, reference->dropout_counter,
                        index) >= reference->dropout_threshold
        ? reference->dropout_scale : 0.0f;
}

static double attention_score(const AttentionReference* reference,
                              uint32_t batch, uint32_t query,
                              uint32_t key, uint32_t head) {
    double score = 0.0;
    for (uint32_t dimension = 0; dimension < reference->head_dim; dimension++)
        score += (double)attention_q_value(reference, batch, query, head,
                                           dimension) *
            attention_k_value(reference, batch, key, head, dimension);
    return score * reference->scale;
}

static int attention_statistics(const AttentionReference* reference,
                                uint32_t batch, uint32_t query,
                                uint32_t head, double* maximum,
                                double* denominator) {
    int visible = 0;
    double local_maximum = 0.0;
    double local_denominator = 0.0;
    for (uint32_t key = 0; key < reference->key_sequence; key++) {
        double score;
        if (!attention_key_allowed(reference, batch, query, key)) continue;
        score = attention_score(reference, batch, query, key, head);
        if (!visible || score > local_maximum) local_maximum = score;
        visible = 1;
    }
    if (!visible) return 0;
    for (uint32_t key = 0; key < reference->key_sequence; key++) {
        if (!attention_key_allowed(reference, batch, query, key)) continue;
        local_denominator += expf((float)(attention_score(
            reference, batch, query, key, head) - local_maximum));
    }
    *maximum = local_maximum;
    *denominator = local_denominator;
    return 1;
}

static double attention_probability(const AttentionReference* reference,
                                    uint32_t batch, uint32_t query,
                                    uint32_t key, uint32_t head,
                                    double maximum, double denominator) {
    return expf((float)(attention_score(reference, batch, query, key, head) -
                        maximum)) / denominator;
}

static void attention_forward_reference(const AttentionReference* reference,
                                        float* output) {
    for (uint32_t batch = 0; batch < reference->batch; batch++) {
        for (uint32_t query = 0; query < reference->query_sequence; query++) {
            for (uint32_t head = 0; head < reference->heads; head++) {
                double maximum;
                double denominator;
                size_t output_base = ((size_t)batch *
                    reference->query_sequence + query) * reference->d_model +
                    (size_t)head * reference->head_dim;
                if (!attention_statistics(reference, batch, query, head,
                                          &maximum, &denominator)) {
                    for (uint32_t dimension = 0;
                         dimension < reference->head_dim; dimension++)
                        output[output_base + dimension] = 0.0f;
                    continue;
                }
                for (uint32_t dimension = 0;
                     dimension < reference->head_dim; dimension++) {
                    double value = 0.0;
                    for (uint32_t key = 0; key < reference->key_sequence;
                         key++) {
                        if (!attention_key_allowed(reference, batch, query,
                                                   key)) continue;
                        value += attention_probability(reference, batch, query,
                            key, head, maximum, denominator) *
                            attention_dropout_multiplier(reference, batch,
                                head, query, key) *
                            attention_v_value(reference, batch, key, head,
                                              dimension);
                    }
                    output[output_base + dimension] = (float)value;
                }
            }
        }
    }
}

static void attention_backward_reference(const AttentionReference* reference,
                                         const float* grad_output,
                                         float* grad_query, float* grad_key,
                                         float* grad_value) {
    for (uint32_t batch = 0; batch < reference->batch; batch++) {
        for (uint32_t query = 0; query < reference->query_sequence; query++) {
            for (uint32_t head = 0; head < reference->heads; head++) {
                double maximum;
                double denominator;
                double weighted_probability_gradient = 0.0;
                size_t output_base = ((size_t)batch *
                    reference->query_sequence + query) * reference->d_model +
                    (size_t)head * reference->head_dim;
                if (!attention_statistics(reference, batch, query, head,
                                          &maximum, &denominator)) continue;
                for (uint32_t key = 0; key < reference->key_sequence; key++) {
                    double probability;
                    double probability_gradient = 0.0;
                    float multiplier;
                    if (!attention_key_allowed(reference, batch, query, key))
                        continue;
                    probability = attention_probability(reference, batch,
                        query, key, head, maximum, denominator);
                    multiplier = attention_dropout_multiplier(reference,
                        batch, head, query, key);
                    for (uint32_t dimension = 0;
                         dimension < reference->head_dim; dimension++) {
                        float output_gradient =
                            grad_output[output_base + dimension];
                        probability_gradient += (double)output_gradient *
                            attention_v_value(reference, batch, key, head,
                                              dimension);
                        grad_value[attention_v_index(reference, batch, key,
                            head, dimension)] +=
                            (float)(probability * multiplier * output_gradient);
                    }
                    weighted_probability_gradient += probability *
                        probability_gradient * multiplier;
                }
                for (uint32_t key = 0; key < reference->key_sequence; key++) {
                    double probability;
                    double probability_gradient = 0.0;
                    double score_gradient;
                    float multiplier;
                    if (!attention_key_allowed(reference, batch, query, key))
                        continue;
                    probability = attention_probability(reference, batch,
                        query, key, head, maximum, denominator);
                    multiplier = attention_dropout_multiplier(reference,
                        batch, head, query, key);
                    for (uint32_t dimension = 0;
                         dimension < reference->head_dim; dimension++)
                        probability_gradient +=
                            (double)grad_output[output_base + dimension] *
                            attention_v_value(reference, batch, key, head,
                                              dimension);
                    score_gradient = probability *
                        (probability_gradient * multiplier -
                         weighted_probability_gradient);
                    for (uint32_t dimension = 0;
                         dimension < reference->head_dim; dimension++) {
                        size_t q_index = attention_q_index(
                            reference, batch, query, head, dimension);
                        size_t k_index = attention_k_index(
                            reference, batch, key, head, dimension);
                        grad_query[q_index] += (float)(score_gradient *
                            reference->scale * attention_k_value(reference,
                                batch, key, head, dimension));
                        grad_key[k_index] += (float)(score_gradient *
                            reference->scale * attention_q_value(reference,
                                batch, query, head, dimension));
                    }
                }
            }
        }
    }
}

static size_t attention_mask_elements(const AttentionReference* reference) {
    if (reference->mask_mode == 1u) return reference->key_sequence;
    if (reference->mask_mode == 2u)
        return (size_t)reference->batch * reference->key_sequence;
    if (reference->mask_mode == 3u)
        return (size_t)reference->query_sequence * reference->key_sequence;
    if (reference->mask_mode == 4u)
        return (size_t)reference->batch * reference->query_sequence *
            reference->key_sequence;
    return 0u;
}

static int attention_close(float actual, float expected) {
    float difference = fabsf(actual - expected);
    float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    return difference <= 3.0e-4f * scale;
}

static int run_cuda_sdpa_reference_case(
        const AttentionReference* reference, const float* grad_output,
        float* output, float* grad_qkv) {
    size_t qkv_bytes = (size_t)reference->batch * reference->query_sequence *
        reference->d_model * 3u * sizeof(float);
    size_t output_bytes = (size_t)reference->batch *
        reference->query_sequence * reference->d_model * sizeof(float);
    size_t mask_elements = attention_mask_elements(reference);
    const void* mask_host = reference->mask_mode
        ? (const void*)reference->mask : (const void*)reference->qkv;
    size_t mask_bytes = reference->mask_mode
        ? mask_elements * sizeof(int32_t) : qkv_bytes;
    uint32_t params[12] = {
        reference->query_sequence, reference->d_model, reference->heads,
        reference->head_dim, reference->batch, float_bits(reference->scale),
        reference->causal, reference->mask_mode,
        reference->dropout_threshold, reference->dropout_seed,
        reference->dropout_counter, float_bits(reference->dropout_scale),
    };
    void* hosts[5] = {
        (void*)reference->qkv, (void*)mask_host, (void*)grad_output,
        grad_qkv, params,
    };
    size_t bytes[5] = {
        qkv_bytes, mask_bytes, output_bytes, qkv_bytes, sizeof(params),
    };
    const unsigned char access[5] = {1u, 1u, 1u, 3u, 1u};
    const unsigned char is_weight[5] = {0u, 0u, 0u, 0u, 0u};
    uint64_t launches = cuda_test_launch_count();

    cuda_graph_begin_forward();
    if (!cuda_graph_sdpa_training_f32(reference->qkv, reference->mask,
            (long)mask_elements, output, (int)reference->query_sequence,
            (int)reference->d_model, (int)reference->heads,
            (int)reference->head_dim, (int)reference->batch, reference->scale,
            (int)reference->causal, (int)reference->mask_mode,
            reference->dropout_threshold, reference->dropout_seed,
            reference->dropout_counter, reference->dropout_scale) ||
        cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, output_bytes, 0) ||
        cuda_test_launch_count() - launches != 2u) return -1;

    if (!cuda_training_supports("sdpaBackward", "main", bytes, 5,
            reference->query_sequence, reference->heads,
            reference->batch * 3u) || cuda_training_begin() != 0) return -1;
    launches = cuda_test_launch_count();
    int result = cuda_training_dispatch("sdpaBackward", "main", hosts, bytes,
        access, is_weight, 5, reference->query_sequence, reference->heads,
        reference->batch * 3u);
    if (result == 0 && cuda_test_launch_count() - launches != 3u) result = -1;
    if (result == 0) result = cuda_training_sync(grad_qkv, qkv_bytes);
    cuda_training_end();
    return result;
}

static int run_cuda_cross_sdpa_forward(
        const AttentionReference* reference, float* output) {
    size_t output_bytes = (size_t)reference->batch *
        reference->query_sequence * reference->d_model * sizeof(float);
    size_t mask_elements = attention_mask_elements(reference);
    uint64_t launches = cuda_test_launch_count();
    cuda_graph_begin_forward();
    if (!cuda_graph_cross_sdpa_training_f32(reference->query, reference->key,
            reference->value, reference->mask, (long)mask_elements, output,
            (int)reference->query_sequence, (int)reference->key_sequence,
            (int)reference->d_model, (int)reference->heads,
            (int)reference->head_dim, (int)reference->batch, reference->scale,
            (int)reference->causal, (int)reference->mask_mode,
            reference->dropout_threshold, reference->dropout_seed,
            reference->dropout_counter, reference->dropout_scale) ||
        cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, output_bytes, 0) ||
        cuda_test_launch_count() - launches != 2u) return -1;
    return 0;
}

static int run_cuda_cross_sdpa_backward_entry(
        const AttentionReference* reference, const float* grad_output,
        float* grad_query, float* grad_key, float* grad_value,
        const char* entry_point) {
    size_t query_bytes = (size_t)reference->batch *
        reference->query_sequence * reference->d_model * sizeof(float);
    size_t key_bytes = (size_t)reference->batch * reference->key_sequence *
        reference->d_model * sizeof(float);
    size_t mask_elements = attention_mask_elements(reference);
    const void* mask_host = reference->mask_mode
        ? (const void*)reference->mask : (const void*)reference->query;
    size_t mask_bytes = reference->mask_mode
        ? mask_elements * sizeof(int32_t) : query_bytes;
    uint32_t params[16] = {
        reference->query_sequence, reference->key_sequence,
        reference->d_model, reference->heads, reference->head_dim,
        reference->batch, float_bits(reference->scale), reference->causal,
        reference->mask_mode, reference->dropout_threshold,
        reference->dropout_seed, reference->dropout_counter,
        float_bits(reference->dropout_scale), 0u, 0u, 0u,
    };
    void* hosts[9] = {
        (void*)reference->query, (void*)reference->key,
        (void*)reference->value, (void*)mask_host, (void*)grad_output,
        grad_query, grad_key, grad_value, params,
    };
    size_t bytes[9] = {
        query_bytes, key_bytes, key_bytes, mask_bytes, query_bytes,
        query_bytes, key_bytes, key_bytes, sizeof(params),
    };
    const unsigned char access[9] = {
        1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[9] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    uint32_t groups_x = strcmp(entry_point, "q_main") == 0
        ? reference->query_sequence : reference->key_sequence;
    float* destination = strcmp(entry_point, "q_main") == 0
        ? grad_query : (strcmp(entry_point, "k_main") == 0
            ? grad_key : grad_value);
    size_t destination_bytes = strcmp(entry_point, "q_main") == 0
        ? query_bytes : key_bytes;
    if (!cuda_training_supports("crossSdpaBackward", entry_point, bytes, 9,
            groups_x, reference->heads, reference->batch) ||
        cuda_training_begin() != 0) return -1;
    uint64_t launches = cuda_test_launch_count();
    int result = cuda_training_dispatch("crossSdpaBackward", entry_point,
        hosts, bytes, access, is_weight, 9, groups_x, reference->heads,
        reference->batch);
    if (result == 0 && cuda_test_launch_count() - launches != 3u) result = -1;
    if (result == 0)
        result = cuda_training_sync(destination, destination_bytes);
    cuda_training_end();
    return result;
}

static int test_attention_forward_and_backward(void) {
    enum {
        SELF_SEQUENCE = 2,
        SELF_D_MODEL = 2,
        SELF_QKV_ELEMENTS = SELF_SEQUENCE * SELF_D_MODEL * 3,
        CROSS_QUERY_SEQUENCE = 3,
        CROSS_KEY_SEQUENCE = 4,
        CROSS_D_MODEL = 4,
        CROSS_QUERY_ELEMENTS = CROSS_QUERY_SEQUENCE * CROSS_D_MODEL,
        CROSS_KEY_ELEMENTS = CROSS_KEY_SEQUENCE * CROSS_D_MODEL,
        CROSS_MASK_ELEMENTS = CROSS_QUERY_SEQUENCE * CROSS_KEY_SEQUENCE,
    };
    const float self_qkv[SELF_QKV_ELEMENTS] = {
        0.4f, -0.2f, 0.1f, 0.5f, 0.7f, -0.3f,
        -0.1f, 0.6f, 0.3f, -0.4f, 0.2f, 0.8f,
    };
    const float self_grad_output[SELF_SEQUENCE * SELF_D_MODEL] = {
        0.5f, -0.5f, -0.25f, 0.25f,
    };
    const float cross_query[CROSS_QUERY_ELEMENTS] = {
        0.2f, -0.1f, 0.4f, 0.3f,
        -0.5f, 0.7f, 0.1f, -0.2f,
        0.6f, 0.2f, -0.4f, 0.8f,
    };
    const float cross_key[CROSS_KEY_ELEMENTS] = {
        0.3f, 0.5f, -0.2f, 0.4f,
        -0.6f, 0.1f, 0.7f, -0.3f,
        0.8f, -0.4f, 0.2f, 0.6f,
        -0.1f, 0.9f, -0.5f, 0.2f,
    };
    const float cross_value[CROSS_KEY_ELEMENTS] = {
        0.5f, -0.7f, 0.3f, 0.1f,
        -0.2f, 0.4f, 0.9f, -0.6f,
        0.8f, 0.2f, -0.1f, 0.7f,
        -0.4f, 0.6f, 0.5f, -0.3f,
    };
    const int32_t cross_mask[CROSS_MASK_ELEMENTS] = {
        1, 1, 1, 1,
        0, 0, 0, 0,
        1, 0, 1, 1,
    };
    const float cross_grad_output[CROSS_QUERY_ELEMENTS] = {
        0.5f, -0.25f, 0.75f, -0.5f,
        -0.3f, 0.6f, -0.2f, 0.4f,
        0.9f, -0.7f, 0.1f, 0.8f,
    };
    AttentionReference self_reference = {
        self_qkv, NULL, NULL, NULL, NULL, 1u, SELF_SEQUENCE,
        SELF_SEQUENCE, SELF_D_MODEL, 1u, SELF_D_MODEL,
        0.7071067811865475f, 0u, 0u, 0u, 18u, 7u, 1.0f, 1,
    };
    AttentionReference cross_reference = {
        NULL, cross_query, cross_key, cross_value, cross_mask, 1u,
        CROSS_QUERY_SEQUENCE, CROSS_KEY_SEQUENCE, CROSS_D_MODEL, 2u, 2u,
        0.7071067811865475f, 1u, 4u, 0x80000000u, 18u, 9u, 2.0f, 0,
    };
    float self_expected_output[SELF_SEQUENCE * SELF_D_MODEL] = {0};
    float self_first_output[SELF_SEQUENCE * SELF_D_MODEL] = {0};
    float self_second_output[SELF_SEQUENCE * SELF_D_MODEL] = {0};
    float self_prefill[SELF_QKV_ELEMENTS];
    float self_expected_grad[SELF_QKV_ELEMENTS];
    float self_first_grad[SELF_QKV_ELEMENTS];
    float self_second_grad[SELF_QKV_ELEMENTS];
    float cross_expected_output[CROSS_QUERY_ELEMENTS] = {0};
    float cross_first_output[CROSS_QUERY_ELEMENTS] = {0};
    float cross_second_output[CROSS_QUERY_ELEMENTS] = {0};
    float cross_prefill_query[CROSS_QUERY_ELEMENTS];
    float cross_prefill_key[CROSS_KEY_ELEMENTS];
    float cross_prefill_value[CROSS_KEY_ELEMENTS];
    float cross_expected_query[CROSS_QUERY_ELEMENTS];
    float cross_expected_key[CROSS_KEY_ELEMENTS];
    float cross_expected_value[CROSS_KEY_ELEMENTS];
    float cross_first_query[CROSS_QUERY_ELEMENTS];
    float cross_first_key[CROSS_KEY_ELEMENTS];
    float cross_first_value[CROSS_KEY_ELEMENTS];
    float cross_second_query[CROSS_QUERY_ELEMENTS];
    float cross_second_key[CROSS_KEY_ELEMENTS];
    float cross_second_value[CROSS_KEY_ELEMENTS];
    int self_changed = 0;
    int cross_changed = 0;

    for (uint32_t index = 0; index < SELF_QKV_ELEMENTS; index++)
        self_prefill[index] = ((float)(int)(index % 7u) - 3.0f) * 0.03125f;
    memcpy(self_expected_grad, self_prefill, sizeof(self_expected_grad));
    memcpy(self_first_grad, self_prefill, sizeof(self_first_grad));
    memcpy(self_second_grad, self_prefill, sizeof(self_second_grad));
    attention_forward_reference(&self_reference, self_expected_output);
    attention_backward_reference(&self_reference, self_grad_output,
        self_expected_grad, self_expected_grad, self_expected_grad);

    for (uint32_t index = 0; index < CROSS_QUERY_ELEMENTS; index++)
        cross_prefill_query[index] =
            ((float)(int)(index % 5u) - 2.0f) * 0.025f;
    for (uint32_t index = 0; index < CROSS_KEY_ELEMENTS; index++) {
        cross_prefill_key[index] =
            ((float)(int)(index % 7u) - 3.0f) * 0.02f;
        cross_prefill_value[index] =
            ((float)(int)(index % 9u) - 4.0f) * 0.015f;
    }
    memcpy(cross_expected_query, cross_prefill_query,
           sizeof(cross_expected_query));
    memcpy(cross_expected_key, cross_prefill_key, sizeof(cross_expected_key));
    memcpy(cross_expected_value, cross_prefill_value,
           sizeof(cross_expected_value));
    memcpy(cross_first_query, cross_prefill_query, sizeof(cross_first_query));
    memcpy(cross_first_key, cross_prefill_key, sizeof(cross_first_key));
    memcpy(cross_first_value, cross_prefill_value, sizeof(cross_first_value));
    memcpy(cross_second_query, cross_prefill_query, sizeof(cross_second_query));
    memcpy(cross_second_key, cross_prefill_key, sizeof(cross_second_key));
    memcpy(cross_second_value, cross_prefill_value,
           sizeof(cross_second_value));
    attention_forward_reference(&cross_reference, cross_expected_output);
    attention_backward_reference(&cross_reference, cross_grad_output,
        cross_expected_query, cross_expected_key, cross_expected_value);

    /* One shared staged workspace must safely grow and then serve a smaller
     * descriptor without shrinking: self(S=2) -> cross(Q=3,K=4,H=2) -> self. */
    CHECK(run_cuda_sdpa_reference_case(&self_reference, self_grad_output,
                                       self_first_output,
                                       self_first_grad) == 0);
    CHECK(run_cuda_cross_sdpa_forward(&cross_reference,
                                      cross_first_output) == 0);
    CHECK(run_cuda_sdpa_reference_case(&self_reference, self_grad_output,
                                       self_second_output,
                                       self_second_grad) == 0);
    CHECK(memcmp(self_first_output, self_second_output,
                 sizeof(self_first_output)) == 0);
    CHECK(memcmp(self_first_grad, self_second_grad,
                 sizeof(self_first_grad)) == 0);
    for (uint32_t index = 0; index < SELF_SEQUENCE * SELF_D_MODEL; index++)
        CHECK(attention_close(self_first_output[index],
                              self_expected_output[index]));
    for (uint32_t index = 0; index < SELF_QKV_ELEMENTS; index++) {
        CHECK(attention_close(self_first_grad[index],
                              self_expected_grad[index]));
        self_changed |= self_expected_grad[index] != self_prefill[index];
    }
    CHECK(self_changed);

    CHECK(run_cuda_cross_sdpa_forward(&cross_reference,
                                      cross_second_output) == 0);
    CHECK(memcmp(cross_first_output, cross_second_output,
                 sizeof(cross_first_output)) == 0);
    for (uint32_t index = 0; index < CROSS_QUERY_ELEMENTS; index++)
        CHECK(attention_close(cross_first_output[index],
                              cross_expected_output[index]));
    for (uint32_t dimension = 0; dimension < CROSS_D_MODEL; dimension++)
        CHECK(cross_first_output[CROSS_D_MODEL + dimension] == 0.0f);

    /* K and V each run before Q in one repetition, proving that every staged
     * destination dispatch regenerates its own probability/dScore workspace. */
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_first_query, cross_first_key,
        cross_first_value, "k_main") == 0);
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_first_query, cross_first_key,
        cross_first_value, "v_main") == 0);
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_first_query, cross_first_key,
        cross_first_value, "q_main") == 0);
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_second_query, cross_second_key,
        cross_second_value, "v_main") == 0);
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_second_query, cross_second_key,
        cross_second_value, "k_main") == 0);
    CHECK(run_cuda_cross_sdpa_backward_entry(&cross_reference,
        cross_grad_output, cross_second_query, cross_second_key,
        cross_second_value, "q_main") == 0);
    CHECK(memcmp(cross_first_query, cross_second_query,
                 sizeof(cross_first_query)) == 0);
    CHECK(memcmp(cross_first_key, cross_second_key,
                 sizeof(cross_first_key)) == 0);
    CHECK(memcmp(cross_first_value, cross_second_value,
                 sizeof(cross_first_value)) == 0);
    for (uint32_t index = 0; index < CROSS_QUERY_ELEMENTS; index++) {
        CHECK(attention_close(cross_first_query[index],
                              cross_expected_query[index]));
        cross_changed |= cross_expected_query[index] !=
            cross_prefill_query[index];
    }
    for (uint32_t index = 0; index < CROSS_KEY_ELEMENTS; index++) {
        CHECK(attention_close(cross_first_key[index],
                              cross_expected_key[index]));
        CHECK(attention_close(cross_first_value[index],
                              cross_expected_value[index]));
        cross_changed |= cross_expected_key[index] != cross_prefill_key[index];
        cross_changed |= cross_expected_value[index] !=
            cross_prefill_value[index];
    }
    CHECK(cross_changed);
    /* Query row one is fully masked, so += must preserve its prefill exactly. */
    for (uint32_t dimension = 0; dimension < CROSS_D_MODEL; dimension++)
        CHECK(cross_first_query[CROSS_D_MODEL + dimension] ==
              cross_prefill_query[CROSS_D_MODEL + dimension]);
    return 0;
}

#include "test_cuda_training_norm_extra.inc"
#include "test_cuda_training_moe.inc"
#include "test_cuda_training_step.inc"

static int test_cuda_profiler_complete_scope(void) {
    const char* configured_path = getenv("VOLVOXAI_CUDA_PROFILE_PATH");
    if (!configured_path || !configured_path[0]) return 0;
    char path[4096];
    char alternate_path[4096];
    const float input[3] = {-1.0f, 2.0f, 3.0f};
    float output[3] = {0};
    const float grad_output[3] = {0.25f, -0.5f, 0.75f};
    float grad_input[3] = {1.0f, 2.0f, 3.0f};
    uint32_t params[4] = {3u, 0u, 0u, 0u};
    void* hosts[5] = {
        (void*)input, output, (void*)grad_output, grad_input, params,
    };
    size_t bytes[5] = {
        sizeof(input), sizeof(output), sizeof(grad_output), sizeof(grad_input),
        sizeof(params),
    };
    const unsigned char access[5] = {1u, 1u, 1u, 3u, 1u};
    const unsigned char is_weight[5] = {0u, 0u, 0u, 0u, 0u};
    FILE* profile;
    char line[1024];
    char saved_locale[128] = {0};
    const char* current_locale = setlocale(LC_NUMERIC, NULL);
    const char* comma_locales[] = {
        "de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8",
    };
    unsigned long long measured = 0u;
    int complete = 0;
    int scope_rows = 0;
    int incomplete_scope_rows = 0;
    int header_rows = 0;
    int unmapped = 0;
    int saw_forward = 0;
    int saw_backward = 0;
    int entry_rows[8] = {0};
    int signature_rows[8] = {0};
    int alternate_header_rows = 0;
    int alternate_scope_rows = 0;
    int alternate_entry_rows = 0;
    int alternate_signature_rows = 0;
    int profile_rc = 0;
    uint64_t capture_count = cuda_test_graph_capture_count();
    uint64_t graph_launch_count = cuda_test_graph_launch_count();
    uint64_t replay_count = cuda_test_graph_replay_count();

    CHECK(strlen(configured_path) < sizeof(path));
    memcpy(path, configured_path, strlen(configured_path) + 1u);
    CHECK(strlen(path) + sizeof(".alternate") <= sizeof(alternate_path));
    CHECK(snprintf(alternate_path, sizeof(alternate_path), "%s.alternate",
                   path) > 0);
    (void)remove(alternate_path);

    if (current_locale)
        (void)snprintf(saved_locale, sizeof(saved_locale), "%s", current_locale);
    /* Exercise comma-decimal output when the host has such a locale installed.
       The exact field-count assertion below remains active in the C locale. */
    for (size_t index = 0u;
         index < sizeof(comma_locales) / sizeof(comma_locales[0]); ++index) {
        const char* selected = setlocale(LC_NUMERIC, comma_locales[index]);
        if (selected && localeconv() && localeconv()->decimal_point &&
            !strcmp(localeconv()->decimal_point, ",")) break;
    }

    for (uint32_t scope = 0u; scope < 7u && !profile_rc; ++scope) {
        if (scope == 2u || scope == 4u || scope == 5u) {
            if (cuda_test_graph_capture_count() != capture_count ||
                cuda_test_graph_launch_count() != graph_launch_count ||
                cuda_test_graph_replay_count() != replay_count)
                profile_rc = -1;
            cuda_cleanup();
            if (!profile_rc && setenv("VOLVOXAI_CUDA_PROFILE_PATH",
                    scope == 4u ? alternate_path : path, 1) != 0)
                profile_rc = -1;
            if (!profile_rc && cuda_init() != 0) profile_rc = -1;
            capture_count = cuda_test_graph_capture_count();
            graph_launch_count = cuda_test_graph_launch_count();
            replay_count = cuda_test_graph_replay_count();
            if (profile_rc) break;
        }
        output[0] = output[1] = output[2] = 0.0f;
        grad_input[0] = 1.0f;
        grad_input[1] = 2.0f;
        grad_input[2] = 3.0f;
        /* Scope one uses the eager/non-loaded zero-generation sentinel; every
           later scope uses the loaded-model contract. All request replay so
           the regression also proves capture/replay suppression. Scope three
           follows cleanup/reinit; scope four deliberately fails dispatch.
           Scope five writes path B and scope six returns to path A, proving
           non-consecutive path history appends without truncation. Scope seven
           records a mixed/non-CUDA route and must remain incomplete. */
        cuda_graph_begin_runtime_forward(1, scope == 0u ? 0u : 1u);
        int forward_ok = cuda_graph_relu_f32(input, output, 3u);
        if (scope == 6u) cuda_graph_note_node_route(0);
        if (cuda_graph_end_runtime_forward(forward_ok) != 0 || !forward_ok)
            profile_rc = -1;
        if (!profile_rc && cuda_training_begin() != 0) profile_rc = -1;
        params[1] = scope == 3u ? 10u : 0u;
        if (!profile_rc) {
            int dispatch_rc = cuda_training_dispatch(
                "activationBackward", "main", hosts, bytes, access,
                is_weight, 5, 1u, 1u, 1u);
            if ((scope == 3u && dispatch_rc == 0) ||
                (scope != 3u && dispatch_rc != 0))
                profile_rc = -1;
        }
        if (!profile_rc && scope != 3u && cuda_training_sync(
                grad_input, sizeof(grad_input)) != 0)
            profile_rc = -1;
        cuda_training_end();
        params[1] = 0u;
        if (scope != 3u) cuda_graph_reset();
    }
    if (setenv("VOLVOXAI_CUDA_PROFILE_PATH", path, 1) != 0) profile_rc = -1;
    if (saved_locale[0]) (void)setlocale(LC_NUMERIC, saved_locale);
    CHECK(profile_rc == 0);
    CHECK(cuda_test_graph_capture_count() == capture_count);
    CHECK(cuda_test_graph_launch_count() == graph_launch_count);
    CHECK(cuda_test_graph_replay_count() == replay_count);
    profile = fopen(path, "r");
    CHECK(profile != NULL);
    while (fgets(line, sizeof(line), profile)) {
        if (strncmp(line, "record,scope,", 13u) == 0) header_rows++;
        if (strstr(line, ",unmapped,")) unmapped = 1;
        if (strstr(line, ",vx_cuda_unary_f32,")) saw_forward = 1;
        if (strstr(line, ",vx_cuda_training_activation_backward_f32,"))
            saw_backward = 1;
        if (strncmp(line, "entry,", 6u) == 0 ||
            strncmp(line, "signature,", 10u) == 0) {
            int is_entry = line[0] == 'e';
            char* field = strtok(line, ",");
            int field_index = 0;
            unsigned long long scope_number = 0u;
            unsigned long long aggregate_count = 0u;
            while (field) {
                if (field_index == 1)
                    scope_number = strtoull(field, NULL, 10);
                if (field_index == 11)
                    aggregate_count = strtoull(field, NULL, 10);
                field = strtok(NULL, ",");
                field_index++;
            }
            CHECK(field_index == 15);
            CHECK(scope_number >= 1u && scope_number <= 7u);
            CHECK(scope_number != 5u);
            CHECK(aggregate_count == 1u);
            if (is_entry) entry_rows[scope_number]++;
            else signature_rows[scope_number]++;
        }
        if (strncmp(line, "scope,", 6u) == 0) {
            char* field = strtok(line, ",");
            int field_index = 0;
            unsigned long long scope_number = 0u;
            scope_rows++;
            while (field) {
                if (field_index == 1)
                    scope_number = strtoull(field, NULL, 10);
                if (field_index == 2) complete = atoi(field);
                if (field_index == 11) measured = strtoull(field, NULL, 10);
                field = strtok(NULL, ",");
                field_index++;
            }
            CHECK(field_index == 15);
            CHECK(scope_number >= 1u && scope_number <= 7u);
            CHECK(scope_number != 5u);
            if (scope_number == 1u || scope_number == 4u ||
                scope_number == 7u) {
                CHECK(complete == 0);
                incomplete_scope_rows++;
            } else {
                CHECK(complete == 1);
            }
            CHECK(measured == (scope_number == 4u ? 1u : 2u));
        }
    }
    fclose(profile);
    CHECK(header_rows == 1);
    CHECK(scope_rows == 6);
    CHECK(incomplete_scope_rows == 3);
    CHECK(complete == 0);
    CHECK(measured == 2u);
    CHECK(!unmapped);
    CHECK(saw_forward);
    CHECK(saw_backward);
    CHECK(entry_rows[2] == 2 && signature_rows[2] == 2);
    CHECK(entry_rows[3] == 2 && signature_rows[3] == 2);
    CHECK(entry_rows[6] == 2 && signature_rows[6] == 2);

    profile = fopen(alternate_path, "r");
    CHECK(profile != NULL);
    while (fgets(line, sizeof(line), profile)) {
        if (strncmp(line, "record,scope,", 13u) == 0)
            alternate_header_rows++;
        CHECK(!strstr(line, ",unmapped,"));
        if (strncmp(line, "entry,", 6u) == 0 ||
            strncmp(line, "signature,", 10u) == 0) {
            int is_entry = line[0] == 'e';
            char* field = strtok(line, ",");
            int field_index = 0;
            unsigned long long scope_number = 0u;
            unsigned long long aggregate_count = 0u;
            while (field) {
                if (field_index == 1)
                    scope_number = strtoull(field, NULL, 10);
                if (field_index == 11)
                    aggregate_count = strtoull(field, NULL, 10);
                field = strtok(NULL, ",");
                field_index++;
            }
            CHECK(field_index == 15);
            CHECK(scope_number == 5u);
            CHECK(aggregate_count == 1u);
            if (is_entry) alternate_entry_rows++;
            else alternate_signature_rows++;
        }
        if (strncmp(line, "scope,", 6u) == 0) {
            char* field = strtok(line, ",");
            int field_index = 0;
            unsigned long long scope_number = 0u;
            alternate_scope_rows++;
            while (field) {
                if (field_index == 1)
                    scope_number = strtoull(field, NULL, 10);
                if (field_index == 2) complete = atoi(field);
                if (field_index == 11) measured = strtoull(field, NULL, 10);
                field = strtok(NULL, ",");
                field_index++;
            }
            CHECK(field_index == 15);
            CHECK(scope_number == 5u);
            CHECK(complete == 1);
            CHECK(measured == 2u);
        }
    }
    fclose(profile);
    CHECK(alternate_header_rows == 1);
    CHECK(alternate_scope_rows == 1);
    CHECK(alternate_entry_rows == 2);
    CHECK(alternate_signature_rows == 2);
    CHECK(remove(alternate_path) == 0);
    return 0;
}

int main(void) {
    CudaTestEngineScope engine_scope = {0};
    float grad_output[5] = {1.0f, -2.0f, 3.5f, 4.0f, -5.0f};
    float grad_input[5] = {10.0f, 20.0f, -3.0f, 0.5f, 2.0f};
    const float expected[5] = {11.0f, 18.0f, 0.5f, 4.5f, -3.0f};
    uint32_t params[4] = {5u, 0u, 0u, 0u};
    void* hosts[3] = {grad_output, grad_input, params};
    size_t bytes[3] = {sizeof(grad_output), sizeof(grad_input), sizeof(params)};
    const unsigned char access[3] = {1u, 3u, 1u};
    const unsigned char is_weight[3] = {0u, 0u, 0u};

    if (cuda_test_engine_scope_begin(&engine_scope) != 0) {
        fputs("CUDA test engine-state initialization failed\n", stderr);
        return 1;
    }

    if (cuda_init() != 0) {
        if (cuda_init_failure_is_unavailable()) {
            puts("CUDA device unavailable; skipping CUDA training scheduler test");
            cuda_test_engine_scope_end(&engine_scope);
            return 77;
        }
        fputs("CUDA device was found, but full-profile CUDA initialization failed\n",
              stderr);
        cuda_test_engine_scope_end(&engine_scope);
        return 1;
    }
    CHECK(cuda_test_caller_context_is_clear());
    CHECK(cuda_training_available());
    CHECK_DIRECT_CASE(test_cuda_profiler_complete_scope());
    CHECK_DIRECT_CASE(test_cuda_training_step_window());
    CHECK(cuda_training_supports("copyBackward", "main", bytes, 3, 1, 1, 1));
    CHECK(!cuda_training_supports("copyBackward", "bad", bytes, 3, 1, 1, 1));
    CHECK(!cuda_training_supports("copyBackward", "main", bytes, 3, 2, 1, 1));
    CHECK(!cuda_training_supports("unknownBackward", "main", bytes, 3, 1, 1, 1));

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_begin() != 0);
    CHECK(cuda_training_dispatch("copyBackward", "main", hosts, bytes,
                                 access, is_weight, 3, 1, 1, 1) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    for (int index = 0; index < 5; index++) CHECK(grad_input[index] == expected[index]);
    cuda_training_end();
    CHECK(cuda_test_caller_context_is_clear());
    cuda_graph_reset();

    CHECK_DIRECT_CASE(test_dropout_forward());
    CHECK_DIRECT_CASE(test_dropout_backward());
    CHECK_DIRECT_CASE(test_dropout_descriptors());
    CHECK_DIRECT_CASE(test_matmul_backward_layout(0u, 1u));
    CHECK_DIRECT_CASE(test_matmul_backward_layout(1u, 0u));
    CHECK_DIRECT_CASE(test_matmul_backward_tiled_tail(0u, 1u));
    CHECK_DIRECT_CASE(test_matmul_backward_tiled_tail(1u, 0u));
    CHECK_DIRECT_CASE(test_matmul_backward_descriptors());
    for (uint32_t kind = 0u; kind < 4u; kind++) {
        CHECK_DIRECT_CASE(test_basic_backward_kind(kind));
        CHECK_DIRECT_CASE(test_basic_backward_identity_kind(kind));
        CHECK_DIRECT_CASE(test_basic_backward_scalar_a(kind, 0u));
        CHECK_DIRECT_CASE(test_basic_backward_scalar_b(kind, 0u));
    }
    CHECK_DIRECT_CASE(test_basic_backward_scalar_a(0u, 1u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_a(0u, 2u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_b(0u, 1u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_b(0u, 2u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_reduction_determinism());
    for (uint32_t kind = 0u; kind < 4u; ++kind) {
        CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(1, kind, 0u));
        CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(0, kind, 0u));
    }
    CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(1, 0u, 1u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(1, 0u, 2u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(0, 0u, 1u));
    CHECK_DIRECT_CASE(test_basic_backward_scalar_staged(0, 0u, 2u));
    for (uint32_t kind = 0u; kind < 4u; ++kind)
        CHECK_DIRECT_CASE(test_basic_backward_cooperative_kind(kind, 0u));
    CHECK_DIRECT_CASE(test_basic_backward_cooperative_kind(0u, 1u));
    CHECK_DIRECT_CASE(test_basic_backward_cooperative_kind(0u, 2u));
    CHECK_DIRECT_CASE(test_basic_backward_cooperative_descriptors());
    CHECK_DIRECT_CASE(test_basic_backward_add_activation(1u));
    CHECK_DIRECT_CASE(test_basic_backward_add_activation(2u));
    CHECK_DIRECT_CASE(test_basic_backward_descriptors());
    for (uint32_t kind = 0u; kind < 10u; kind++)
        CHECK_DIRECT_CASE(test_activation_backward_kind(kind));
    CHECK_DIRECT_CASE(test_activation_backward_descriptors());
    CHECK_DIRECT_CASE(test_reduce_backward());
    CHECK_DIRECT_CASE(test_softmax_backward_mode(0u));
    CHECK_DIRECT_CASE(test_softmax_backward_mode(1u));
    CHECK_DIRECT_CASE(test_embedding_backward());
    CHECK_DIRECT_CASE(test_layernorm_backward());
    CHECK_DIRECT_CASE(test_layernorm_param_cooperative(131u, 1));
    CHECK_DIRECT_CASE(test_layernorm_param_cooperative(131u, 0));
    CHECK_DIRECT_CASE(test_layernorm_param_cooperative(17u, 1));
    CHECK_DIRECT_CASE(test_rmsnorm_backward());
    CHECK_DIRECT_CASE(test_conv2d_backward_layout(0u, 1u));
    CHECK_DIRECT_CASE(test_conv2d_backward_layout(0u, 2u));
    CHECK_DIRECT_CASE(test_conv2d_backward_layout(1u, 1u));
    CHECK_DIRECT_CASE(test_conv2d_backward_layout(2u, 1u));
    CHECK_DIRECT_CASE(test_conv2d_backward_layout(3u, 1u));
    CHECK_DIRECT_CASE(test_conv2d_backward_input_hwio_c32(32u));
    CHECK_DIRECT_CASE(test_conv2d_backward_input_hwio_c32(33u));
    CHECK_DIRECT_CASE(test_conv2d_backward_input_hwio_c32(48u));
    CHECK_DIRECT_CASE(test_conv2d_backward_descriptors());
    CHECK_DIRECT_CASE(test_transpose_backward());
    CHECK_DIRECT_CASE(test_concat_backward());
    CHECK_DIRECT_CASE(test_split_backward());
    CHECK_DIRECT_CASE(test_resize_backward_nearest());
    CHECK_DIRECT_CASE(test_resize_backward_bilinear_boundary());
    CHECK_DIRECT_CASE(test_pooling_backward());
    CHECK_DIRECT_CASE(test_shape_backward_descriptors());
    CHECK_DIRECT_CASE(test_batchnorm2d_backward());
    CHECK_DIRECT_CASE(test_groupnorm_backward());
    CHECK_DIRECT_CASE(test_groupnorm_backward_input_cooperative_tail());
    CHECK_DIRECT_CASE(test_prelu_backward());
    CHECK_DIRECT_CASE(test_norm_extra_backward_descriptors());
    CHECK_DIRECT_CASE(test_moe_linear_backward());
    CHECK_DIRECT_CASE(test_moe_router_backward_normalized());
    CHECK_DIRECT_CASE(test_moe_router_backward_full_softmax());
    CHECK_DIRECT_CASE(test_moe_device_coherence_chain());
    CHECK_DIRECT_CASE(test_moe_backward_descriptors());
    CHECK_DIRECT_CASE(test_attention_forward_and_backward());
    CHECK(cuda_test_caller_context_is_clear());

    cuda_cleanup();
    cuda_test_engine_scope_end(&engine_scope);
    puts("CUDA full-profile training scheduler substrate test passed");
    return 0;
}
