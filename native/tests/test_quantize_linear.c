#include "../src/kernels/inference_kernels.h"
#include "../src/kernels/quant_cpu_isa.h"

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

static int test_dequantize_byte_domains(void) {
    enum { COUNT = 1027 };
    int8_t input_i8[COUNT];
    uint8_t input_u8[COUNT];
    float actual[COUNT];
    float expected[COUNT];
    const float scales[] = {
        1.0f, 0.05f, -0.125f, 0.0f, -0.0f,
        0x1p-149f, 0x1.fffffep+127f,
    };
    /* These include both ends of the common I8/U8 SIMD descriptor domain:
     * every centered byte remains exactly representable as F32. */
    const int32_t zero_points[] = {0, -16776961, 16777088};

    for (int index = 0; index < COUNT; index++) {
        input_i8[index] = (int8_t)((index & 255) - 128);
        input_u8[index] = (uint8_t)(index & 255);
    }
    for (uint32_t dtype_index = 0; dtype_index < 2u; dtype_index++) {
        const void* input = dtype_index == 0u
            ? (const void*)input_i8 : (const void*)input_u8;
        const uint32_t dtype = dtype_index == 0u ? VX_DTYPE_I8 : VX_DTYPE_U8;
        for (uint32_t scale_index = 0;
             scale_index < sizeof(scales) / sizeof(scales[0]); scale_index++) {
            for (uint32_t zero_index = 0;
                 zero_index < sizeof(zero_points) / sizeof(zero_points[0]);
                 zero_index++) {
                const float scale = scales[scale_index];
                const int32_t zero = zero_points[zero_index];
                CHECK(dequantize_linear_typed(
                    input, (int)dtype, &scale, &zero, VX_DTYPE_I32,
                    actual, COUNT) == 1);
                for (int index = 0; index < COUNT; index++) {
                    const int value = dtype == VX_DTYPE_I8
                        ? (int)input_i8[index] : (int)input_u8[index];
                    expected[index] = (float)(((double)value - (double)zero) *
                                              (double)scale);
                }
                CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
            }
        }
    }
    return 0;
}

#if defined(__aarch64__)
static int test_neon_prefix_route_and_overlap(void) {
    float input[32] = {0};
    float dequantized[32];
    uint8_t quantized[32];
    union {
        float alignment;
        uint8_t bytes[160];
    } overlap = {0};

    CHECK(vx_quantize_linear_typed_native_prefix(
        input, 0.25f, 128, quantized, VX_DTYPE_U8, 32u, 0, 255) == 32u);
    CHECK(vx_dequantize_linear_typed_native_prefix(
        quantized, VX_DTYPE_U8, 0.25f, 128, dequantized, 32u) == 32u);

    /* A vector block reads farther ahead than the canonical scalar loop.
     * Overlapping calls therefore decline the prefix and retain that loop's
     * exact forward traversal rather than inventing a new alias contract. */
    CHECK(vx_quantize_linear_typed_native_prefix(
        (const float*)overlap.bytes, 0.25f, 128, overlap.bytes + 4u,
        VX_DTYPE_U8, 32u, 0, 255) == 0u);
    CHECK(vx_dequantize_linear_typed_native_prefix(
        overlap.bytes, VX_DTYPE_U8, 0.25f, 128,
        (float*)(void*)(overlap.bytes + 4u), 32u) == 0u);
    return 0;
}
#endif

int main(void) {
    CHECK(test_domain(VX_DTYPE_I8, -128, 127, -101) == 0);
    CHECK(test_domain(VX_DTYPE_U8, 0, 255, 231) == 0);
    CHECK(test_invalid_descriptors() == 0);
    CHECK(test_dequantize_byte_domains() == 0);
#if defined(__aarch64__)
    CHECK(test_neon_prefix_route_and_overlap() == 0);
#endif
    puts("quantize linear tests passed");
    return 0;
}
