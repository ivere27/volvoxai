#include "volvoxai_training.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static const float g_source_weight[6] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f};

static int closef32(float left, float right) {
    return fabsf(left - right) <= 1.0e-6f * (1.0f + fabsf(left) + fabsf(right));
}

static int test_observer_and_parameters(void) {
    volvoxai_ptq_observer_t observer;
    volvoxai_ptq_observer_reset(&observer);
    const float first[2] = {-2.0f, 1.0f};
    const float second[2] = {3.0f, -1.0f};
    CHECK(volvoxai_ptq_observer_observe_f32(&observer, first, 2) == 0);
    CHECK(volvoxai_ptq_observer_observe_f32(&observer, second, 2) == 0);
    CHECK(observer.minimum == -2.0f && observer.maximum == 3.0f && observer.sample_count == 4);
    volvoxai_ptq_observer_t before = observer;
    const float invalid[2] = {1.0f, NAN};
    CHECK(volvoxai_ptq_observer_observe_f32(&observer, invalid, 2) != 0);
    CHECK(!memcmp(&observer, &before, sizeof(observer)));

    volvoxai_ptq_params_t params;
    CHECK(volvoxai_ptq_calculate_params(&observer, VOLVOXAI_DTYPE_I8,
                                        VOLVOXAI_PTQ_SYMMETRIC, &params) == 0);
    CHECK(params.zero_point == 0);
    CHECK(closef32(params.scale, 3.0f / 127.0f));
    const float values[3] = {-3.0f, 0.0f, 3.0f};
    int8_t output[3] = {0};
    uint64_t saturation = 99;
    CHECK(volvoxai_ptq_quantize_f32(values, 3, &params, output, &saturation) == 0);
    CHECK(output[0] == -127 && output[1] == 0 && output[2] == 127 && saturation == 0);

    volvoxai_ptq_observer_t unit_range = {-127.0f, 127.0f, 2};
    CHECK(volvoxai_ptq_calculate_params(&unit_range, VOLVOXAI_DTYPE_I8,
                                        VOLVOXAI_PTQ_SYMMETRIC, &params) == 0);
    CHECK(params.scale == 1.0f);
    const float full_range_input[4] = {-129.0f, -128.0f, 127.0f, 128.0f};
    int8_t full_range_output[4] = {0};
    saturation = 99;
    CHECK(volvoxai_ptq_quantize_f32(full_range_input, 4, &params,
                                    full_range_output, &saturation) == 0);
    CHECK(full_range_output[0] == -128 && full_range_output[1] == -128 &&
          full_range_output[2] == 127 && full_range_output[3] == 127);
    CHECK(saturation == 2);

    volvoxai_ptq_observer_reset(&observer);
    const float asymmetric_values[2] = {-1.0f, 3.0f};
    CHECK(volvoxai_ptq_observer_observe_f32(&observer, asymmetric_values, 2) == 0);
    CHECK(volvoxai_ptq_calculate_params(&observer, VOLVOXAI_DTYPE_U8,
                                        VOLVOXAI_PTQ_ASYMMETRIC, &params) == 0);
    CHECK(params.zero_point == 64);
    CHECK(closef32(params.scale, 4.0f / 255.0f));
    uint8_t asymmetric_output[3] = {0};
    const float asymmetric_input[3] = {-1.0f, 0.0f, 3.0f};
    CHECK(volvoxai_ptq_quantize_f32(asymmetric_input, 3, &params,
                                    asymmetric_output, NULL) == 0);
    CHECK(asymmetric_output[0] == 0 && asymmetric_output[1] == 64 && asymmetric_output[2] == 255);
    return 0;
}

static int test_weight_and_bias_packing(void) {
    const int shape[2] = {2, 3};
    int8_t output[6] = {0};
    float scales[2] = {0};
    uint64_t saturation = 99;
    CHECK(volvoxai_ptq_pack_weight_i8(g_source_weight, shape, 2, 0, output,
                                      scales, 2, &saturation) == 0);
    const int8_t expected[6] = {-127, 0, 127, -127, 0, 127};
    CHECK(!memcmp(output, expected, sizeof(expected)));
    CHECK(closef32(scales[0], 1.0f / 127.0f));
    CHECK(closef32(scales[1], 2.0f / 127.0f));
    CHECK(saturation == 0);

    const int narrow_shape[2] = {1, 2};
    const float narrow_source[2] = {-128.0f, 128.0f};
    int8_t narrow_output[2] = {0};
    float narrow_scale[1] = {0};
    CHECK(volvoxai_ptq_pack_weight_i8(narrow_source, narrow_shape, 2, 0,
                                      narrow_output, narrow_scale, 1, NULL) == 0);
    CHECK(narrow_output[0] == -127 && narrow_output[1] == 127);

    union {
        float aligned[6];
        int8_t bytes[sizeof(float) * 6];
    } alias = {0};
    CHECK(volvoxai_ptq_pack_weight_i8(g_source_weight, shape, 2, 0,
                                      alias.bytes, alias.aligned, 2, NULL) != 0);
    CHECK(volvoxai_ptq_pack_weight_i8(g_source_weight, shape, 2, 0, output,
                                      (float*)(uintptr_t)g_source_weight,
                                      2, NULL) != 0);
    memcpy(alias.aligned, g_source_weight, sizeof(g_source_weight));
    CHECK(volvoxai_ptq_pack_weight_i8(alias.aligned, shape, 2, 0,
                                      alias.bytes, scales, 2, NULL) != 0);

    const float bias[2] = {0.125f, -0.5f};
    const float bias_scales[2] = {0.25f, 0.5f};
    int32_t packed_bias[2] = {0};
    CHECK(volvoxai_ptq_pack_bias_i32(bias, 2, 0.5f, bias_scales, 2, packed_bias) == 0);
    CHECK(packed_bias[0] == 1 && packed_bias[1] == -2);
    return 0;
}

int main(void) {
    CHECK(volvoxai_ptq_abi_version() == VOLVOXAI_PTQ_ABI_VERSION);
    CHECK(volvoxai_ptq_capabilities() == VOLVOXAI_PTQ_CAP_ALL);
    CHECK(test_observer_and_parameters() == 0);
    CHECK(test_weight_and_bias_packing() == 0);
    puts("native PTQ tests passed");
    return 0;
}
