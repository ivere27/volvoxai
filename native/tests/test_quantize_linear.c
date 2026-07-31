#include "../src/kernels/inference_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                #condition); \
        return 1; \
    } \
} while (0)

static int round_ties_even(float value) {
    int lower = (int)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

static int expected_transformed(float transformed, int minimum, int maximum,
                                int zero) {
    if (isnan(transformed)) return zero;
    if (transformed <= (float)minimum) return minimum;
    if (transformed >= (float)maximum) return maximum;
    if (!isfinite(transformed)) return transformed < 0.0f ? minimum : maximum;
    return round_ties_even(transformed);
}

static int test_domain(uint32_t dtype, int minimum, int maximum, int zero) {
    float input[1100];
    int expected[1100];
    int8_t output_i8[1100];
    uint8_t output_u8[1100];
    float scale = 1.0f;
    int8_t zero_i8 = (int8_t)zero;
    uint8_t zero_u8 = (uint8_t)zero;
    const void *zero_data = dtype == VX_DTYPE_I8
        ? (const void *)&zero_i8 : (const void *)&zero_u8;
    void *output = dtype == VX_DTYPE_I8
        ? (void *)output_i8 : (void *)output_u8;
    uint32_t count = 0;

    for (int base = minimum - 4; base <= maximum + 4; base++) {
        const float transformed[4] = {
            (float)base, (float)base + 0.25f,
            (float)base + 0.5f, (float)base + 0.75f,
        };
        for (int lane = 0; lane < 4; lane++) {
            CHECK(count < sizeof(input) / sizeof(input[0]));
            input[count] = transformed[lane] - (float)zero;
            expected[count] = expected_transformed(transformed[lane],
                                                       minimum, maximum, zero);
            count++;
        }
    }
    input[count] = NAN;
    expected[count++] = zero;
    input[count] = INFINITY;
    expected[count++] = maximum;
    input[count] = -INFINITY;
    expected[count++] = minimum;

    CHECK(quantize_linear_typed(input, &scale, zero_data, dtype, output, dtype,
                                count) == 1);
    for (uint32_t index = 0; index < count; index++) {
        int actual = dtype == VX_DTYPE_I8
            ? (int)output_i8[index] : (int)output_u8[index];
        CHECK(actual == expected[index]);
    }
    return 0;
}

static int test_invalid_descriptors(void) {
    const float input[17] = {0};
    const float invalid_scales[] = {0.0f, -0.0f, -1.0f, NAN, INFINITY};
    int8_t zero = -3;
    uint8_t output[17];

    for (uint32_t index = 0;
         index < sizeof(invalid_scales) / sizeof(invalid_scales[0]); index++) {
        memset(output, 0x6d, sizeof(output));
        CHECK(quantize_linear_typed(input, &invalid_scales[index], &zero,
                                    VX_DTYPE_I8, output, VX_DTYPE_I8, 17) == 0);
        for (uint32_t lane = 0; lane < sizeof(output); lane++)
            CHECK(output[lane] == 0x6d);
    }
    {
        const float scale = 1.0f;
        CHECK(quantize_linear_typed(input, &scale, &zero, VX_DTYPE_U8,
                                    output, VX_DTYPE_I8, 17) == 0);
        CHECK(quantize_linear_typed(input, &scale, &zero, VX_DTYPE_I8,
                                    output, VX_DTYPE_F32, 17) == 0);
        CHECK(quantize_linear_typed(NULL, &scale, &zero, VX_DTYPE_I8,
                                    output, VX_DTYPE_I8, 17) == 0);
    }
    return 0;
}

int main(void) {
    CHECK(test_domain(VX_DTYPE_I8, -128, 127, -101) == 0);
    CHECK(test_domain(VX_DTYPE_U8, 0, 255, 231) == 0);
    CHECK(test_invalid_descriptors() == 0);
    puts("quantize linear tests passed");
    return 0;
}
