#include "../src/backends/qlinear_multiplier.h"
#include "../src/kernels/w8a8_affine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                #condition);                                                 \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void test_byte_domain(void) {
    for (int raw = 0; raw < 256; raw++) {
        uint8_t byte = (uint8_t)raw;
        uint8_t stored = 0;
        const int32_t unsigned_value =
            vx_w8a8_byte_value(&byte, VX_DTYPE_U8, 0u);
        const int32_t signed_value =
            vx_w8a8_byte_value(&byte, VX_DTYPE_I8, 0u);
        CHECK(unsigned_value == raw);
        CHECK(signed_value == (int32_t)(int8_t)byte);
        vx_w8a8_store_byte(&stored, VX_DTYPE_U8, 0u, unsigned_value);
        CHECK(stored == byte);
        vx_w8a8_store_byte(&stored, VX_DTYPE_I8, 0u, signed_value);
        CHECK(stored == byte);
    }

    CHECK(vx_w8a8_byte_dtype(VX_DTYPE_I8));
    CHECK(vx_w8a8_byte_dtype(VX_DTYPE_U8));
    CHECK(!vx_w8a8_byte_dtype(VX_DTYPE_I32));
    for (int value = -129; value <= 256; value++) {
        CHECK(vx_w8a8_zero_point_valid(value, VX_DTYPE_I8) ==
              (value >= -128 && value <= 127));
        CHECK(vx_w8a8_zero_point_valid(value, VX_DTYPE_U8) ==
              (value >= 0 && value <= 255));
    }
}

static void test_round_and_saturate(void) {
    static const struct {
        float value;
        int32_t expected;
    } ties[] = {
        {-3.5f, -4}, {-2.5f, -2}, {-1.5f, -2}, {-0.5f, 0},
        { 0.5f,  0}, { 1.5f,  2}, { 2.5f,  2}, { 3.5f, 4},
    };
    for (size_t index = 0u; index < sizeof(ties) / sizeof(ties[0]); index++)
        CHECK(vx_w8a8_round_ties_even(ties[index].value) ==
              ties[index].expected);

    CHECK(vx_w8a8_requantize(-129.0f, -128, 127, 0) == -128);
    CHECK(vx_w8a8_requantize(128.0f, -128, 127, 0) == 127);
    CHECK(vx_w8a8_requantize(2.5f, -128, 127, 0) == 2);
    CHECK(vx_w8a8_requantize(NAN, 0, 255, 137) == 137);
}

static void test_multiplier_contract(void) {
    static const float scales[] = {
        0x1p-12f, 0.002f, 0.1f, 0.333333f, 1.0f, 3.14159f, 64.0f,
    };
    for (size_t input = 0u;
         input < sizeof(scales) / sizeof(scales[0]); input++) {
        for (size_t weight = 0u;
             weight < sizeof(scales) / sizeof(scales[0]); weight++) {
            for (size_t output = 0u;
                 output < sizeof(scales) / sizeof(scales[0]); output++) {
                float kernel = 0.0f;
                float backend = 0.0f;
                const int kernel_ok = vx_w8a8_multiplier(
                    scales[input], scales[weight], scales[output], &kernel);
                const int backend_ok = vx_qlinear_compute_multiplier(
                    scales[input], scales[weight], scales[output], &backend);
                CHECK(kernel_ok == backend_ok);
                if (kernel_ok)
                    CHECK(memcmp(&kernel, &backend, sizeof(kernel)) == 0);
            }
        }
    }

    {
        float multiplier = 0.0f;
        CHECK(!vx_w8a8_multiplier(0.0f, 1.0f, 1.0f, &multiplier));
        CHECK(!vx_w8a8_multiplier(1.0f, -1.0f, 1.0f, &multiplier));
        CHECK(!vx_w8a8_multiplier(1.0f, 1.0f, INFINITY, &multiplier));
        CHECK(!vx_w8a8_multiplier(0x1p-100f, 0x1p-100f, 1.0f,
                                  &multiplier));
        CHECK(!vx_w8a8_multiplier(1.0f, 1.0f, 1.0f, NULL));
    }
}

static void test_memory_contract(void) {
    unsigned char bytes[32];
    size_t elements = 12u;
    CHECK(vx_w8a8_mul_size(&elements, 7u) && elements == 84u);
    elements = (size_t)-1;
    CHECK(!vx_w8a8_mul_size(&elements, 2u));

    CHECK(!vx_w8a8_ranges_overlap(bytes, 8u, bytes + 8u, 8u));
    CHECK(vx_w8a8_ranges_overlap(bytes, 9u, bytes + 8u, 8u));
    CHECK(!vx_w8a8_ranges_overlap(bytes, 0u, bytes, sizeof(bytes)));
    CHECK(vx_w8a8_ranges_overlap(
        (const void*)(uintptr_t)-8, 32u, bytes, sizeof(bytes)));

    {
        const float expected = -1234.5f;
        unsigned char unaligned[sizeof(float) + 1u];
        float actual;
        memcpy(unaligned + 1u, &expected, sizeof(expected));
        actual = vx_w8a8_affine_f32_at(
            (const float*)(const void*)(unaligned + 1u), 0u);
        CHECK(memcmp(&actual, &expected, sizeof(actual)) == 0);
    }
}

int main(void) {
    test_byte_domain();
    test_round_and_saturate();
    test_multiplier_contract();
    test_memory_contract();
    if (failures) {
        fprintf(stderr, "w8a8 affine contract failed (%d checks)\n", failures);
        return 1;
    }
    puts("w8a8 affine contract passed");
    return 0;
}
