/*
 * Proves the canonical W8A8 affine contract before any kernel is migrated onto
 * it.  Where the domain is small the checks are exhaustive; where it is not they
 * cover the boundaries that the scattered copies disagreed about.
 *
 * The multiplier check is the important one: it asserts the header agrees
 * bit-for-bit with native/src/backends/qlinear_multiplier.h, the helper the
 * device backends already use, so CPU and GPU cannot drift apart again.
 */
#include "../src/kernels/w8a8_affine.h"
#include "../src/backends/qlinear_multiplier.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int g_failures;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

static void test_byte_access_exhaustive(void) {
    /* Every representable byte, both dtypes, load and store round trip. */
    for (int raw = 0; raw < 256; raw++) {
        uint8_t storage = (uint8_t)raw;
        int32_t as_u8 = vx_w8a8_byte_value(&storage, VX_DTYPE_U8, 0);
        int32_t as_i8 = vx_w8a8_byte_value(&storage, VX_DTYPE_I8, 0);
        CHECK(as_u8 == raw);
        CHECK(as_i8 == (int32_t)(int8_t)(uint8_t)raw);

        uint8_t out = 0;
        vx_w8a8_store_byte(&out, VX_DTYPE_U8, 0, as_u8);
        CHECK(out == (uint8_t)raw);
        vx_w8a8_store_byte(&out, VX_DTYPE_I8, 0, as_i8);
        CHECK(out == (uint8_t)raw);
    }
    /* Indexing must be element-wise, not byte-wise scaled. */
    {
        int8_t buffer[4] = { -128, -1, 0, 127 };
        CHECK(vx_w8a8_byte_value(buffer, VX_DTYPE_I8, 0) == -128);
        CHECK(vx_w8a8_byte_value(buffer, VX_DTYPE_I8, 3) == 127);
    }
}

static void test_dtype_and_zero_point(void) {
    CHECK(vx_w8a8_byte_dtype(VX_DTYPE_I8));
    CHECK(vx_w8a8_byte_dtype(VX_DTYPE_U8));
    CHECK(!vx_w8a8_byte_dtype(VX_DTYPE_I32));
    CHECK(!vx_w8a8_byte_dtype(VX_DTYPE_F32));
    CHECK(!vx_w8a8_byte_dtype(VX_DTYPE_UNSPECIFIED));

    /* Exhaustive over the whole plausible zero-point range including outside. */
    for (int v = -300; v <= 300; v++) {
        CHECK(vx_w8a8_zero_point_valid(v, VX_DTYPE_I8) ==
              (v >= -128 && v <= 127));
        CHECK(vx_w8a8_zero_point_valid(v, VX_DTYPE_U8) ==
              (v >= 0 && v <= 255));
        CHECK(!vx_w8a8_zero_point_valid(v, VX_DTYPE_I32));
    }
    {
        int32_t minimum = 0, maximum = 0;
        vx_w8a8_output_range(VX_DTYPE_I8, &minimum, &maximum);
        CHECK(minimum == -128 && maximum == 127);
        vx_w8a8_output_range(VX_DTYPE_U8, &minimum, &maximum);
        CHECK(minimum == 0 && maximum == 255);
    }
}

static void test_finite_and_scale(void) {
    const float zero = 0.0f;
    const float infinity = 1.0f / zero;
    const float nan_value = infinity - infinity;
    CHECK(vx_w8a8_finite_f32(0.0f));
    CHECK(vx_w8a8_finite_f32(-1.5e30f));
    CHECK(vx_w8a8_finite_f32(1.0e-38f));
    CHECK(!vx_w8a8_finite_f32(infinity));
    CHECK(!vx_w8a8_finite_f32(-infinity));
    CHECK(!vx_w8a8_finite_f32(nan_value));

    CHECK(vx_w8a8_scale_valid(1.0f));
    CHECK(vx_w8a8_scale_valid(1.0e-30f));
    CHECK(!vx_w8a8_scale_valid(0.0f));
    CHECK(!vx_w8a8_scale_valid(-1.0f));
    CHECK(!vx_w8a8_scale_valid(infinity));
    CHECK(!vx_w8a8_scale_valid(nan_value));
}

static void test_round_ties_even(void) {
    /* Exact ties must go to even in both directions of the number line. */
    CHECK(vx_w8a8_round_ties_even(0.5f) == 0);
    CHECK(vx_w8a8_round_ties_even(1.5f) == 2);
    CHECK(vx_w8a8_round_ties_even(2.5f) == 2);
    CHECK(vx_w8a8_round_ties_even(3.5f) == 4);
    CHECK(vx_w8a8_round_ties_even(-0.5f) == 0);
    CHECK(vx_w8a8_round_ties_even(-1.5f) == -2);
    CHECK(vx_w8a8_round_ties_even(-2.5f) == -2);
    CHECK(vx_w8a8_round_ties_even(-3.5f) == -4);
    /* Non-ties round to nearest. */
    CHECK(vx_w8a8_round_ties_even(0.4999f) == 0);
    CHECK(vx_w8a8_round_ties_even(0.5001f) == 1);
    CHECK(vx_w8a8_round_ties_even(-0.5001f) == -1);
    CHECK(vx_w8a8_round_ties_even(7.0f) == 7);
    CHECK(vx_w8a8_round_ties_even(-7.0f) == -7);
    /* Every half-integer tie across the byte range lands on an even integer. */
    for (int i = -300; i <= 300; i++) {
        float tie = (float)i + 0.5f;
        int32_t rounded = vx_w8a8_round_ties_even(tie);
        CHECK(rounded % 2 == 0);
        CHECK(rounded == i || rounded == i + 1);
    }
}

static void test_requantize(void) {
    const float zero = 0.0f;
    const float nan_value = (1.0f / zero) - (1.0f / zero);
    CHECK(vx_w8a8_requantize(-5.0f, 0, 255, 128) == 0);
    CHECK(vx_w8a8_requantize(999.0f, 0, 255, 128) == 255);
    CHECK(vx_w8a8_requantize(nan_value, 0, 255, 128) == 128);
    CHECK(vx_w8a8_requantize(2.5f, 0, 255, 128) == 2);
    CHECK(vx_w8a8_requantize(-0.5f, -128, 127, 0) == 0);
    /* Exactly on the boundary saturates rather than rounding past it. */
    CHECK(vx_w8a8_requantize(255.0f, 0, 255, 128) == 255);
    CHECK(vx_w8a8_requantize(0.0f, 0, 255, 128) == 0);
    CHECK(vx_w8a8_requantize(-128.0f, -128, 127, 0) == -128);
}

static void test_multiplier_matches_backend_helper(void) {
    /* The device backends build the multiplier through
     * vx_qlinear_compute_multiplier.  Any disagreement here is CPU/GPU drift,
     * which is exactly the defect this consolidation is meant to remove. */
    static const float scales[] = {
        1.0e-6f, 1.0e-4f, 0.002f, 0.01f, 0.05f, 0.1f, 0.333333f, 1.0f,
        1.7f, 3.14159f, 100.0f, 4096.0f, 0.015625f, 0.0625f,
    };
    const size_t count = sizeof scales / sizeof scales[0];
    size_t compared = 0;
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < count; j++) {
            for (size_t k = 0; k < count; k++) {
                float mine = 0.0f, theirs = 0.0f;
                int a = vx_w8a8_multiplier(scales[i], scales[j], scales[k], &mine);
                int b = vx_qlinear_compute_multiplier(scales[i], scales[j],
                                                      scales[k], &theirs);
                CHECK(a == b);
                if (a && b) {
                    /* Bit equality, not tolerance: a one-LSB difference here is
                     * what previously made the packed path disagree. */
                    CHECK(memcmp(&mine, &theirs, sizeof mine) == 0);
                    compared++;
                }
            }
        }
    }
    CHECK(compared > 0);

    /* Rejection cases. */
    {
        const float zero = 0.0f;
        const float infinity = 1.0f / zero;
        float out = 0.0f;
        CHECK(!vx_w8a8_multiplier(0.0f, 1.0f, 1.0f, &out));
        CHECK(!vx_w8a8_multiplier(1.0f, -1.0f, 1.0f, &out));
        CHECK(!vx_w8a8_multiplier(1.0f, 1.0f, 0.0f, &out));
        CHECK(!vx_w8a8_multiplier(infinity, 1.0f, 1.0f, &out));
        CHECK(!vx_w8a8_multiplier(1.0f, 1.0f, 1.0f, NULL));
        /* Underflow of the product to zero must be rejected, not silently used. */
        CHECK(!vx_w8a8_multiplier(1.0e-30f, 1.0e-30f, 1.0f, &out));
    }
}

static void test_mul_size_and_overlap(void) {
    {
        size_t value = 10;
        CHECK(vx_w8a8_mul_size(&value, 20) && value == 200);
        value = (size_t)-1;
        CHECK(!vx_w8a8_mul_size(&value, 2));
        value = 5;
        CHECK(vx_w8a8_mul_size(&value, 0) && value == 0);
    }
    {
        unsigned char buffer[64];
        /* Disjoint. */
        CHECK(!vx_w8a8_ranges_overlap(buffer, 16, buffer + 16, 16));
        CHECK(!vx_w8a8_ranges_overlap(buffer + 16, 16, buffer, 16));
        /* Touching but not overlapping. */
        CHECK(!vx_w8a8_ranges_overlap(buffer, 8, buffer + 8, 8));
        /* Overlapping by one byte, both orders. */
        CHECK(vx_w8a8_ranges_overlap(buffer, 9, buffer + 8, 8));
        CHECK(vx_w8a8_ranges_overlap(buffer + 8, 8, buffer, 9));
        /* Identical and contained. */
        CHECK(vx_w8a8_ranges_overlap(buffer, 32, buffer, 32));
        CHECK(vx_w8a8_ranges_overlap(buffer, 32, buffer + 4, 4));
        /* An empty range overlaps nothing.  The scattered copies disagreed
         * here; without the guard the comparison degenerates into an ordering
         * test and reports a false overlap. */
        CHECK(!vx_w8a8_ranges_overlap(buffer + 8, 0, buffer, 32));
        CHECK(!vx_w8a8_ranges_overlap(buffer, 32, buffer + 8, 0));
        CHECK(!vx_w8a8_ranges_overlap(buffer, 0, buffer, 0));
        /* A range that would wrap the address space is rejected. */
        CHECK(vx_w8a8_ranges_overlap((const void*)(uintptr_t)-8, 64,
                                     buffer, 32));
    }
}

static void test_affine_unaligned_read(void) {
    /* SafeTensors payloads need not be F32 aligned, so the reader must work
     * from an odd offset and agree with an aligned read of the same bytes. */
    union { float f; unsigned char b[4]; } source;
    unsigned char raw[sizeof(float) * 4 + 3];
    source.f = -1234.5678f;
    for (size_t offset = 0; offset < 3; offset++) {
        memcpy(raw + offset, source.b, sizeof source.b);
        float got = vx_w8a8_affine_f32_at(
            (const float*)(const void*)(raw + offset), 0);
        CHECK(memcmp(&got, &source.f, sizeof got) == 0);
    }
    {
        float values[3] = { 1.0f, -2.0f, 0.5f };
        CHECK(vx_w8a8_affine_f32_at(values, 0) == 1.0f);
        CHECK(vx_w8a8_affine_f32_at(values, 1) == -2.0f);
        CHECK(vx_w8a8_affine_f32_at(values, 2) == 0.5f);
    }
}

int main(void) {
    test_byte_access_exhaustive();
    test_dtype_and_zero_point();
    test_finite_and_scale();
    test_round_ties_even();
    test_requantize();
    test_multiplier_matches_backend_helper();
    test_mul_size_and_overlap();
    test_affine_unaligned_read();
    if (g_failures) {
        printf("w8a8 affine contract tests FAILED (%d)\n", g_failures);
        return 1;
    }
    printf("w8a8 affine contract tests passed\n");
    return 0;
}
