#include "inference_kernels.h"
#include "packed_quant_gemm.h"
#include "cpu_features.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    return 0; \
} } while (0)

static int test_w8a32(uint32_t rows, uint32_t d_in, uint32_t d_out,
                      uint32_t dtype, uint32_t out_in) {
    size_t weights_count = (size_t)d_in * d_out;
    size_t output_count = (size_t)rows * d_out;
    float* input = (float*)malloc((size_t)rows * d_in * sizeof(float));
    uint8_t* weight = (uint8_t*)malloc(weights_count);
    float* scales = (float*)malloc((size_t)d_out * sizeof(float));
    int32_t* zero_points = (int32_t*)malloc((size_t)d_out * sizeof(int32_t));
    float* bias = (float*)malloc((size_t)d_out * sizeof(float));
    float* expected = (float*)malloc(output_count * sizeof(float));
    float* actual = (float*)malloc(output_count * sizeof(float));
    uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    CHECK(input && weight && scales && zero_points && bias && expected && actual && packed);
    for (size_t i = 0; i < (size_t)rows * d_in; i++) input[i] = (float)((int)(i * 17u % 29u) - 14) / 7.0f;
    for (size_t i = 0; i < weights_count; i++) {
        if (dtype == 2u) ((int8_t*)weight)[i] = (int8_t)((int)(i * 13u % 127u) - 63);
        else weight[i] = (uint8_t)(i * 19u % 251u);
    }
    for (uint32_t n = 0; n < d_out; n++) {
        scales[n] = 0.0078125f * (float)(1u + n % 5u);
        zero_points[n] = dtype == 2u ? (int32_t)(n % 9u) - 4 : 113 + (int32_t)(n % 11u);
        bias[n] = (float)((int)n - 5) * 0.125f;
    }
    CHECK(vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out, dtype, out_in));
    if (out_in) {
        CHECK(matmul_quantized_f32(input, weight, scales, zero_points, bias,
            expected, rows, d_in, d_out, dtype, d_out, 1u, d_out));
    } else {
        for (uint32_t m = 0; m < rows; m++) for (uint32_t n = 0; n < d_out; n++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < d_in; k++) {
                size_t wi = (size_t)k * d_out + n;
                int32_t w = dtype == 2u ? ((int8_t*)weight)[wi] : weight[wi];
                sum += input[(size_t)m * d_in + k] * (w - zero_points[n]);
            }
            expected[(size_t)m * d_out + n] = (float)(sum * scales[n] + bias[n]);
        }
    }
    CHECK(vx_matmul_quantized_f32_packed(input, packed, scales, zero_points,
        bias, actual, rows, d_in, d_out, dtype, d_out, 1u, d_out));
    for (size_t i = 0; i < output_count; i++) CHECK(fabsf(expected[i] - actual[i]) <= 1.0e-6f);
    free(input); free(weight); free(scales); free(zero_points); free(bias);
    free(expected); free(actual); free(packed);
    return 1;
}

static int test_w8a8(uint32_t rows, uint32_t d_in, uint32_t d_out,
                     uint32_t input_dtype, uint32_t weight_dtype,
                     uint32_t output_dtype) {
    size_t input_count = (size_t)rows * d_in;
    size_t weight_count = (size_t)d_out * d_in;
    size_t output_count = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)malloc(input_count);
    uint8_t* weight = (uint8_t*)malloc(weight_count);
    int32_t* bias = (int32_t*)malloc((size_t)d_out * sizeof(int32_t));
    float* scales = (float*)malloc((size_t)d_out * sizeof(float));
    int32_t* zero_points = (int32_t*)malloc((size_t)d_out * sizeof(int32_t));
    uint8_t* expected = (uint8_t*)malloc(output_count);
    uint8_t* actual = (uint8_t*)malloc(output_count);
    uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    int32_t input_zp = input_dtype == 2u ? -7 : 131;
    int32_t output_zp = output_dtype == 2u ? 3 : 127;
    CHECK(input && weight && bias && scales && zero_points && expected && actual && packed);
    for (size_t i = 0; i < input_count; i++) {
        if (input_dtype == 2u) ((int8_t*)input)[i] = (int8_t)((int)(i * 23u % 127u) - 63);
        else input[i] = (uint8_t)(i * 29u % 251u);
    }
    for (size_t i = 0; i < weight_count; i++) {
        if (weight_dtype == 2u) ((int8_t*)weight)[i] = (int8_t)((int)(i * 31u % 127u) - 63);
        else weight[i] = (uint8_t)(i * 37u % 251u);
    }
    for (uint32_t n = 0; n < d_out; n++) {
        bias[n] = (int32_t)n * 17 - 91;
        scales[n] = 0.00390625f * (float)(1u + n % 7u);
        zero_points[n] = weight_dtype == 2u ? (int32_t)(n % 13u) - 6 : 119 + (int32_t)(n % 17u);
    }
    CHECK(vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out, weight_dtype, 1u));
    CHECK(qlinear_i8u8(input, weight, bias, scales, zero_points, expected,
        rows, d_in, d_out, 0.03125f, input_zp, 0.0625f, output_zp,
        input_dtype, weight_dtype, output_dtype));
    CHECK(vx_qlinear_i8u8_packed(input, packed, bias, scales, zero_points,
        actual, rows, d_in, d_out, 0.03125f, input_zp, 0.0625f, output_zp,
        input_dtype, weight_dtype, output_dtype));
    CHECK(memcmp(expected, actual, output_count) == 0);
    free(input); free(weight); free(bias); free(scales); free(zero_points);
    free(expected); free(actual); free(packed);
    return 1;
}

static int test_metadata_and_malformed_packs(void) {
    enum { d_in = 5, d_out = 3 };
    const float input[d_in] = {1.0f, -2.0f, 0.5f, 4.0f, -3.0f};
    const int8_t weight[d_in * d_out] = {
        1, 2, 3, 4, 5, -6, -7, 8, 9, 10, 11, -12, 13, -14, 15,
    };
    const float scalar_scale[1] = {0.125f};
    const int32_t scalar_zero[1] = {-3};
    const float bias[d_out] = {1.0f, -1.0f, 0.5f};
    float expected[d_out], actual[d_out];
    uint32_t bytes = vx_packed_q8_weight_size(d_in, d_out);
    uint8_t* packed = (uint8_t*)malloc(bytes);
    uint8_t* corrupted = (uint8_t*)malloc(bytes);
    CHECK(packed && corrupted && bytes > 32u);
    CHECK(!vx_pack_q8_weight(packed, bytes - 1u, weight, d_in, d_out, 2u, 1u));
    CHECK(!vx_pack_q8_weight(packed, bytes, weight, d_in, d_out, 4u, 1u));
    CHECK(!vx_pack_q8_weight(packed, bytes, weight, d_in, d_out, 2u, 2u));
    CHECK(vx_pack_q8_weight(packed, bytes, weight, d_in, d_out, 2u, 1u));
    CHECK(matmul_quantized_f32(input, weight, scalar_scale, scalar_zero, bias,
        expected, 1u, d_in, d_out, 2u, 1u, 1u, 1u));
    CHECK(vx_matmul_quantized_f32_packed(input, packed, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, 2u, 1u, 1u, 1u));
    for (uint32_t n = 0; n < d_out; n++) CHECK(fabsf(expected[n] - actual[n]) <= 1.0e-6f);
    memcpy(corrupted, packed, bytes);
    ((uint32_t*)corrupted)[0] ^= 1u;
    CHECK(!vx_matmul_quantized_f32_packed(input, corrupted, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, 2u, 1u, 1u, 1u));
    memcpy(corrupted, packed, bytes);
    ((uint32_t*)corrupted)[7] += 16u;
    CHECK(!vx_matmul_quantized_f32_packed(input, corrupted, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, 2u, 1u, 1u, 1u));
    CHECK(vx_packed_q8_weight_size((uint32_t)(INT32_MAX / 255) + 1u, 1u) == 0u);
    CHECK(vx_packed_q8_weight_size(1u, UINT32_MAX) == 0u);
    CHECK(vx_packed_q8_weight_size(UINT32_MAX, UINT32_MAX) == 0u);
    free(packed);
    free(corrupted);
    return 1;
}

static int test_w8a8_overflow_matches_portable(void) {
    enum { d_in = 40000 };
    int8_t* input = (int8_t*)malloc(d_in);
    int8_t* weight = (int8_t*)malloc(d_in);
    int8_t portable_output = 0;
    int8_t packed_output = 0;
    int32_t bias[1] = {0};
    float scale[1] = {1.0f};
    int32_t zero_point[1] = {-128};
    uint32_t bytes = vx_packed_q8_weight_size(d_in, 1u);
    void* packed = malloc(bytes);
    CHECK(input && weight && packed);
    memset(input, 127, d_in);
    memset(weight, 127, d_in);
    CHECK(vx_pack_q8_weight(packed, bytes, weight, d_in, 1u, 2u, 1u));
    CHECK(!qlinear_i8u8(input, weight, bias, scale, zero_point,
        &portable_output, 1u, d_in, 1u, 1.0f, -128, 1.0f, 0,
        2u, 2u, 2u));
    CHECK(!vx_qlinear_i8u8_packed(input, packed, bias, scale, zero_point,
        &packed_output, 1u, d_in, 1u, 1.0f, -128, 1.0f, 0,
        2u, 2u, 2u));
    free(input);
    free(weight);
    free(packed);
    return 1;
}

int main(void) {
    if (vx_packed_q8_preferred_for_native_w8a8(0u) ||
        vx_packed_q8_preferred_for_native_w8a8(1u)) return 1;
#if defined(__i386__) || defined(__x86_64__)
    if (vx_packed_q8_preferred_for_native_w8a8(2u) != !!vx_cpu_has_avx2()) return 1;
#else
    if (vx_packed_q8_preferred_for_native_w8a8(2u)) return 1;
#endif
    if (!test_w8a32(1, 13, 11, 2u, 1u) ||
        !test_w8a32(5, 17, 9, 3u, 1u) ||
        !test_w8a32(4, 15, 13, 2u, 0u) ||
        !test_w8a8(1, 13, 11, 2u, 2u, 2u) ||
        !test_w8a8(5, 17, 9, 3u, 2u, 3u) ||
        !test_w8a8(3, 19, 13, 2u, 3u, 3u) ||
        !test_metadata_and_malformed_packs() ||
        !test_w8a8_overflow_matches_portable()) return 1;
    puts("packed quantized GEMM correctness tests passed");
    return 0;
}
