#include "inference_kernels.h"
#include "packed_quant_gemm.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    return 0; \
} } while (0)

_Static_assert(VX_DTYPE_U8 == 5 && VX_DTYPE_I8 == 6 &&
               VX_DTYPE_I32 == 16,
               "native kernels use canonical protobuf DataType values");

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
        if (dtype == VX_DTYPE_I8)
            ((int8_t*)weight)[i] =
                (int8_t)((int)(i * 13u % 127u) - 63);
        else weight[i] = (uint8_t)(i * 19u % 251u);
    }
    for (uint32_t n = 0; n < d_out; n++) {
        scales[n] = 0.0078125f * (float)(1u + n % 5u);
        zero_points[n] = dtype == VX_DTYPE_I8 ?
            (int32_t)(n % 9u) - 4 : 113 + (int32_t)(n % 11u);
        bias[n] = (float)((int)n - 5) * 0.125f;
    }
    CHECK(vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out, dtype, out_in));
    if (out_in) {
        CHECK(matmul_quantized_f32(input, weight, scales, zero_points, bias,
            expected, rows, d_in, d_out, dtype, d_out, VX_DTYPE_I32,
            d_out));
    } else {
        for (uint32_t m = 0; m < rows; m++) for (uint32_t n = 0; n < d_out; n++) {
            double sum = 0.0;
            for (uint32_t k = 0; k < d_in; k++) {
                size_t wi = (size_t)k * d_out + n;
                int32_t w = dtype == VX_DTYPE_I8 ?
                    ((int8_t*)weight)[wi] : weight[wi];
                sum += input[(size_t)m * d_in + k] * (w - zero_points[n]);
            }
            expected[(size_t)m * d_out + n] = (float)(sum * scales[n] + bias[n]);
        }
    }
    CHECK(vx_matmul_quantized_f32_packed(input, packed, scales, zero_points,
        bias, actual, rows, d_in, d_out, dtype, d_out, VX_DTYPE_I32,
        d_out));
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
    int32_t input_zp = input_dtype == VX_DTYPE_I8 ? -7 : 131;
    int32_t output_zp = output_dtype == VX_DTYPE_I8 ? 3 : 127;
    CHECK(input && weight && bias && scales && zero_points && expected && actual && packed);
    for (size_t i = 0; i < input_count; i++) {
        if (input_dtype == VX_DTYPE_I8)
            ((int8_t*)input)[i] =
                (int8_t)((int)(i * 23u % 127u) - 63);
        else input[i] = (uint8_t)(i * 29u % 251u);
    }
    for (size_t i = 0; i < weight_count; i++) {
        if (weight_dtype == VX_DTYPE_I8)
            ((int8_t*)weight)[i] =
                (int8_t)((int)(i * 31u % 127u) - 63);
        else weight[i] = (uint8_t)(i * 37u % 251u);
    }
    for (uint32_t n = 0; n < d_out; n++) {
        bias[n] = (int32_t)n * 17 - 91;
        scales[n] = 0.00390625f * (float)(1u + n % 7u);
        zero_points[n] = weight_dtype == VX_DTYPE_I8 ?
            (int32_t)(n % 13u) - 6 : 119 + (int32_t)(n % 17u);
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

static int test_w8a8_threaded_all_byte_types(void) {
    static const uint32_t types[] = {VX_DTYPE_I8, VX_DTYPE_U8};
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    VxKernelThreadPoolScope scope;
    int ok = pool != NULL;
    if (!ok) return 0;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    for (uint32_t input = 0; input < 2u && ok; input++) {
        for (uint32_t weight = 0; weight < 2u && ok; weight++) {
            for (uint32_t output = 0; output < 2u && ok; output++) {
                ok = test_w8a8(113u, 37u, 257u, types[input],
                               types[weight], types[output]);
            }
        }
    }
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    return ok;
}

/* Arm I8MM's complete-panel body pairs adjacent N8 blocks into one MR6xNR16
 * tile.  M=13 executes two full tiles and one row tail; K=40 has no K8 tail;
 * N=48 executes three N16 block pairs.  Symmetric I8 weights admit the route,
 * while all activation/output byte domains and asymmetric zero points remain
 * graph-visible and must match the portable reference exactly. */
static int test_w8a8_full_k_n16_all_activation_types(void) {
    static const uint32_t types[] = {VX_DTYPE_I8, VX_DTYPE_U8};
    int ok = 1;
    for (uint32_t input = 0; input < 2u && ok; input++) {
        for (uint32_t output = 0; output < 2u && ok; output++) {
            ok = test_w8a8(13u, 40u, 48u, types[input], VX_DTYPE_I8,
                           types[output]);
        }
    }
    return ok;
}

/* Exercise the exact AVX2 K4/N16 panel directly: full-domain activations force
 * both the general split and no--128 signed-absolute PMADDUBSW dots. K=37
 * covers the padded K4 tail, and N=257 covers the scalar output tail. */
static int test_w8a8_symmetric_i8_exact_panel(void) {
    enum { rows = 113, d_in = 37, d_out = 257 };
    const size_t input_count = (size_t)rows * d_in;
    const size_t weight_count = (size_t)d_out * d_in;
    const size_t output_count = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)malloc(input_count);
    int8_t* weight = (int8_t*)malloc(weight_count);
    int32_t* bias = (int32_t*)malloc((size_t)d_out * sizeof(*bias));
    float* scales = (float*)malloc((size_t)d_out * sizeof(*scales));
    int32_t* zero_points = (int32_t*)calloc(d_out, sizeof(*zero_points));
    uint8_t* expected = (uint8_t*)malloc(output_count);
    uint8_t* actual = (uint8_t*)malloc(output_count);
    const uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    VxKernelThreadPoolScope scope;
    int ok = input && weight && bias && scales && zero_points && expected &&
        actual && packed && packed_bytes && pool;
    if (!ok) goto cleanup;
    for (size_t index = 0; index < input_count; index++)
        input[index] = (uint8_t)(index * 73u + index / 11u + 29u);
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)(uint8_t)(index * 61u + 128u);
    weight[0] = -128;
    weight[1] = 127;
    for (uint32_t column = 0; column < d_out; column++) {
        bias[column] = (int32_t)column * 31 - 4001;
        scales[column] = (float)(1u + column % 3u) / 128.0f;
    }
    if (!vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out,
                           VX_DTYPE_I8, 1u)) {
        ok = 0;
        goto cleanup;
    }
    scope = vx_kernel_thread_pool_scope_enter(pool);
    for (uint32_t input_kind = 0; input_kind < 2u && ok; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 131 : -7;
        for (uint32_t output_kind = 0; output_kind < 2u && ok; output_kind++) {
            const uint32_t output_dtype =
                output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 127 : -3;
            ok = qlinear_i8u8(input, weight, bias, scales, zero_points,
                    expected, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                vx_qlinear_i8u8_packed(input, packed, bias, scales,
                    zero_points, actual, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                memcmp(expected, actual, output_count) == 0;
        }
    }
    for (size_t index = 0; index < weight_count; index++) {
        if (weight[index] == -128) weight[index] = -127;
    }
    if (!vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out,
                           VX_DTYPE_I8, 1u)) {
        ok = 0;
        vx_kernel_thread_pool_scope_leave(scope);
        goto cleanup;
    }
    for (uint32_t input_kind = 0; input_kind < 2u && ok; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 131 : -7;
        for (uint32_t output_kind = 0; output_kind < 2u && ok; output_kind++) {
            const uint32_t output_dtype =
                output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 127 : -3;
            ok = qlinear_i8u8(input, weight, bias, scales, zero_points,
                    expected, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                vx_qlinear_i8u8_packed(input, packed, bias, scales,
                    zero_points, actual, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                memcmp(expected, actual, output_count) == 0;
        }
    }
    vx_kernel_thread_pool_scope_leave(scope);
cleanup:
    if (pool) vx_kernel_thread_pool_destroy(pool);
    free(packed);
    free(actual);
    free(expected);
    free(zero_points);
    free(scales);
    free(bias);
    free(weight);
    free(input);
    return ok;
}

/* The single-thread packed kernel handles four N8 panels at a time.  Use a K4
 * tail, an odd row tail, and an N16 remainder after the N32 body. Full-domain
 * activations plus weights outside the non-saturating |w|<=64 subset exercise
 * the exact signed-absolute spelling on AVX2 and VPDPBUSD on a VNNI tier. */
static int test_w8a8_signed_n32_single_thread(void) {
    enum { rows = 5, d_in = 37, d_out = 48 };
    const size_t input_count = (size_t)rows * d_in;
    const size_t weight_count = (size_t)d_out * d_in;
    const size_t output_count = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)malloc(input_count);
    int8_t* weight = (int8_t*)malloc(weight_count);
    int32_t* bias = (int32_t*)malloc((size_t)d_out * sizeof(*bias));
    float* scales = (float*)malloc((size_t)d_out * sizeof(*scales));
    int32_t* zero_points = (int32_t*)calloc(d_out, sizeof(*zero_points));
    uint8_t* expected = (uint8_t*)malloc(output_count);
    uint8_t* actual = (uint8_t*)malloc(output_count);
    const uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    int ok = input && weight && bias && scales && zero_points && expected &&
        actual && packed && packed_bytes;
    if (!ok) goto cleanup;
    for (size_t index = 0; index < input_count; index++)
        input[index] = (uint8_t)(index * 197u + index / 3u + 251u);
    for (size_t index = 0; index < weight_count; index++) {
        int value = (int)(index * 89u % 255u) - 127;
        weight[index] = (int8_t)value;
    }
    weight[0] = -127;
    weight[1] = 127;
    for (uint32_t column = 0; column < d_out; column++) {
        bias[column] = (int32_t)column * 97 - 1703;
        scales[column] = (float)(1u + column % 7u) / 512.0f;
    }
    ok = vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out,
                           VX_DTYPE_I8, 1u);
    for (uint32_t input_kind = 0; input_kind < 2u && ok; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 137 : -11;
        for (uint32_t output_kind = 0; output_kind < 2u && ok; output_kind++) {
            const uint32_t output_dtype =
                output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 121 : -5;
            ok = qlinear_i8u8(input, weight, bias, scales, zero_points,
                    expected, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                vx_qlinear_i8u8_packed(input, packed, bias, scales,
                    zero_points, actual, rows, d_in, d_out, 1.0f / 256.0f,
                    input_zero_point, 2.0f, output_zero_point, input_dtype,
                    VX_DTYPE_I8, output_dtype) &&
                memcmp(expected, actual, output_count) == 0;
        }
    }
cleanup:
    free(packed);
    free(actual);
    free(expected);
    free(zero_points);
    free(scales);
    free(bias);
    free(weight);
    free(input);
    return ok;
}

/* Bounded symmetric weights select PAIR_NO_SATURATE.  On a forced AVX2 tier
 * this is the N32 plain U8xI8 spelling (use_signed_abs=false), not the
 * signed-absolute body.  M=5 and K=37 cover the odd-row and K4 tails, while
 * N=48 runs one N32 group followed by its N16 remainder. */
static int test_w8a8_n32_nosat_single_thread(void) {
    enum { rows = 5, d_in = 37, d_out = 48 };
    const size_t input_count = (size_t)rows * d_in;
    const size_t weight_count = (size_t)d_out * d_in;
    const size_t output_count = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)malloc(input_count);
    int8_t* weight = (int8_t*)malloc(weight_count);
    int32_t* bias = (int32_t*)malloc((size_t)d_out * sizeof(*bias));
    float* scales = (float*)malloc((size_t)d_out * sizeof(*scales));
    int32_t* zero_points = (int32_t*)calloc(d_out, sizeof(*zero_points));
    uint8_t* expected = (uint8_t*)malloc(output_count);
    uint8_t* actual = (uint8_t*)malloc(output_count);
    const uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    int ok = input && weight && bias && scales && zero_points && expected &&
        actual && packed && packed_bytes && vx_kernels_thread_count() == 1;
    if (!ok) goto cleanup;

    for (size_t index = 0; index < input_count; index++)
        input[index] = (uint8_t)(index * 197u + index / 3u + 251u);
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int)(index * 67u % 129u) - 64);
    weight[0] = -64;
    weight[1] = 64;
    for (uint32_t column = 0; column < d_out; column++) {
        bias[column] = (int32_t)column * 53 - 1201;
        scales[column] = (float)(1u + column % 3u) / 128.0f;
    }
    ok = vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out,
                           VX_DTYPE_I8, 1u);
    for (uint32_t input_kind = 0; input_kind < 2u && ok; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 131 : -7;
        for (uint32_t output_kind = 0; output_kind < 2u && ok; output_kind++) {
            const uint32_t output_dtype =
                output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 127 : -3;
            ok = qlinear_i8u8(input, weight, bias, scales, zero_points,
                    expected, rows, d_in, d_out, 1.0f / 128.0f,
                    input_zero_point, 1.0f / 8.0f, output_zero_point,
                    input_dtype, VX_DTYPE_I8, output_dtype) &&
                vx_qlinear_i8u8_packed(input, packed, bias, scales,
                    zero_points, actual, rows, d_in, d_out, 1.0f / 128.0f,
                    input_zero_point, 1.0f / 8.0f, output_zero_point,
                    input_dtype, VX_DTYPE_I8, output_dtype) &&
                memcmp(expected, actual, output_count) == 0;
        }
    }

cleanup:
    free(packed);
    free(actual);
    free(expected);
    free(zero_points);
    free(scales);
    free(bias);
    free(weight);
    free(input);
    return ok;
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
    CHECK(!vx_pack_q8_weight(packed, bytes - 1u, weight, d_in, d_out,
                             VX_DTYPE_I8, 1u));
    CHECK(!vx_pack_q8_weight(packed, bytes, weight, d_in, d_out,
                             VX_DTYPE_F4, 1u));
    CHECK(!vx_pack_q8_weight(packed, bytes, weight, d_in, d_out,
                             VX_DTYPE_I8, 2u));
    CHECK(vx_pack_q8_weight(packed, bytes, weight, d_in, d_out,
                            VX_DTYPE_I8, 1u));
    CHECK(matmul_quantized_f32(input, weight, scalar_scale, scalar_zero, bias,
        expected, 1u, d_in, d_out, VX_DTYPE_I8, 1u, VX_DTYPE_I32, 1u));
    CHECK(vx_matmul_quantized_f32_packed(input, packed, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, VX_DTYPE_I8, 1u,
        VX_DTYPE_I32, 1u));
    for (uint32_t n = 0; n < d_out; n++) CHECK(fabsf(expected[n] - actual[n]) <= 1.0e-6f);
    memcpy(corrupted, packed, bytes);
    ((uint32_t*)corrupted)[0] ^= 1u;
    CHECK(!vx_matmul_quantized_f32_packed(input, corrupted, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, VX_DTYPE_I8, 1u,
        VX_DTYPE_I32, 1u));
    memcpy(corrupted, packed, bytes);
    ((uint32_t*)corrupted)[7] += 16u;
    CHECK(!vx_matmul_quantized_f32_packed(input, corrupted, scalar_scale,
        scalar_zero, bias, actual, 1u, d_in, d_out, VX_DTYPE_I8, 1u,
        VX_DTYPE_I32, 1u));
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
    CHECK(vx_pack_q8_weight(packed, bytes, weight, d_in, 1u,
                            VX_DTYPE_I8, 1u));
    CHECK(!qlinear_i8u8(input, weight, bias, scale, zero_point,
        &portable_output, 1u, d_in, 1u, 1.0f, -128, 1.0f, 0,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(!vx_qlinear_i8u8_packed(input, packed, bias, scale, zero_point,
        &packed_output, 1u, d_in, 1u, 1.0f, -128, 1.0f, 0,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    free(input);
    free(weight);
    free(packed);
    return 1;
}

static int test_w8a8_multiplier_representability_matches_portable(void) {
    const int8_t input[1] = {0};
    const int8_t weight[1] = {0};
    const int32_t bias[1] = {0};
    const float scale[1] = {0x1p-149f};
    const int32_t zero_point[1] = {0};
    int8_t portable_output = 41;
    int8_t packed_output = 41;
    uint32_t bytes = vx_packed_q8_weight_size(1u, 1u);
    void* packed = malloc(bytes);
    CHECK(packed);
    CHECK(vx_pack_q8_weight(
        packed, bytes, weight, 1u, 1u, VX_DTYPE_I8, 1u));
    CHECK(!qlinear_i8u8(input, weight, bias, scale, zero_point,
        &portable_output, 1u, 1u, 1u, 0x1p-149f, 0, 1.0f, 0,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(!vx_qlinear_i8u8_packed(input, packed, bias, scale, zero_point,
        &packed_output, 1u, 1u, 1u, 0x1p-149f, 0, 1.0f, 0,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(portable_output == 41 && packed_output == 41);
    free(packed);
    return 1;
}

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__)) && !defined(_WIN32)
/* One fresh process per forced tier keeps the header-only platform cache honest.
 * The bounded-weight helper above covers the AVX2 N32 plain dot.  The first
 * pack here takes the signed-absolute AVX2 route but must stay in the ordinary
 * U8 domain on a single-thread VNNI tier.  The second contains -128, so it
 * reaches the full-range VNNI N32 route while AVX2 retains its exact split N16
 * fallback. */
static int test_w8a8_forced_isa_case(const char* requested) {
    enum { rows = 5, d_in = 37, d_out = 64 };
    const size_t input_count = (size_t)rows * d_in;
    const size_t weight_count = (size_t)d_out * d_in;
    const size_t output_count = (size_t)rows * d_out;
    uint8_t input[input_count];
    int8_t signed_weight[weight_count];
    int8_t full_weight[weight_count];
    int32_t bias[d_out];
    float scales[d_out];
    int32_t zero_points[d_out];
    int8_t expected[output_count];
    int8_t actual[output_count];
    const uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    void* packed = malloc(packed_bytes);
    VxKernelPlatform platform;

    CHECK(requested && requested[0] && packed && packed_bytes);
    vx_kernel_platform_resolve(&platform);
    CHECK(strcmp(vx_kernel_isa_name(platform.isa), requested) == 0);
    CHECK(vx_kernels_thread_count() == 1);
    if (platform.isa == VX_KERNEL_ISA_AVX2)
        CHECK(test_w8a8_n32_nosat_single_thread());
    for (size_t index = 0; index < input_count; index++)
        input[index] = (uint8_t)(index * 197u + index / 3u + 251u);
    for (size_t index = 0; index < weight_count; index++) {
        signed_weight[index] = (int8_t)((int)(index * 89u % 255u) - 127);
        full_weight[index] = signed_weight[index];
    }
    signed_weight[0] = -127;
    signed_weight[1] = 127;
    full_weight[0] = -128;
    full_weight[1] = 127;
    for (uint32_t column = 0; column < d_out; column++) {
        bias[column] = (int32_t)column * 31 - 997;
        scales[column] = 1.0f / 64.0f;
        zero_points[column] = 0;
    }

    CHECK(vx_pack_q8_weight(packed, packed_bytes, signed_weight, d_in, d_out,
                            VX_DTYPE_I8, 1u));
    CHECK(vx_packed_q8_prefers_signed_activations(
              packed, zero_points, d_out) ==
          (platform.isa == VX_KERNEL_ISA_AVX2));
    zero_points[0] = 1;
    CHECK(vx_packed_q8_prefers_signed_activations(
              packed, zero_points, d_out) == 0);
    zero_points[0] = 0;
    CHECK(qlinear_i8u8(input, signed_weight, bias, scales, zero_points,
        expected, rows, d_in, d_out, 1.0f / 64.0f, 137, 1.0f, -3,
        VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(vx_qlinear_i8u8_packed(input, packed, bias, scales, zero_points,
        actual, rows, d_in, d_out, 1.0f / 64.0f, 137, 1.0f, -3,
        VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(memcmp(expected, actual, output_count) == 0);

    CHECK(vx_pack_q8_weight(packed, packed_bytes, full_weight, d_in, d_out,
                            VX_DTYPE_I8, 1u));
    CHECK(vx_packed_q8_prefers_signed_activations(
              packed, zero_points, d_out) == 0);
    CHECK(qlinear_i8u8(input, full_weight, bias, scales, zero_points,
        expected, rows, d_in, d_out, 1.0f / 64.0f, 137, 1.0f, -3,
        VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(vx_qlinear_i8u8_packed(input, packed, bias, scales, zero_points,
        actual, rows, d_in, d_out, 1.0f / 64.0f, 137, 1.0f, -3,
        VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(memcmp(expected, actual, output_count) == 0);

    free(packed);
    printf("packed N32 ISA parity passed on tier %s\n", requested);
    return 1;
}

static int test_w8a8_forced_isa_children(const char* executable) {
    static const struct {
        const char* name;
        VxKernelIsa isa;
    } tiers[] = {
        {"baseline", VX_KERNEL_ISA_BASELINE},
        {"avx2", VX_KERNEL_ISA_AVX2},
        {"avxvnni", VX_KERNEL_ISA_AVX_VNNI},
        {"avx512vnni", VX_KERNEL_ISA_AVX512_VNNI},
    };
    VxKernelPlatform host;
    size_t tested = 0;
    vx_kernel_platform_resolve(&host);
    for (size_t index = 0; index < sizeof(tiers) / sizeof(tiers[0]); index++) {
        char command[1024];
        /* The packed dispatcher intentionally prefers VEX AVX-VNNI when a
         * machine exposes both forms.  Do not report that duplicate execution
         * as EVEX parity; an AVX-512-VNNI-only host exercises that body, while
         * verify_native_packed_qgemm_vnni always checks its encoding. */
        int supported = tiers[index].isa == VX_KERNEL_ISA_BASELINE ||
            (tiers[index].isa == VX_KERNEL_ISA_AVX2 && host.has_avx2) ||
            (tiers[index].isa == VX_KERNEL_ISA_AVX_VNNI && host.has_avx_vnni) ||
            (tiers[index].isa == VX_KERNEL_ISA_AVX512_VNNI &&
             host.has_avx512_vnni && !host.has_avx_vnni);
        if (!supported) continue;
        const int command_length = snprintf(command, sizeof(command),
            "VOLVOXAI_CPU_ISA=%s \"%s\" --forced-isa",
            tiers[index].name, executable);
        CHECK(command_length > 0 && (size_t)command_length < sizeof(command));
        CHECK(system(command) == 0);
        tested++;
    }
    CHECK(tested >= 1u);
    return 1;
}
#endif

int main(int argc, char** argv) {
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__)) && !defined(_WIN32)
    if (argc == 2 && !strcmp(argv[1], "--forced-isa")) {
        const char* requested = getenv("VOLVOXAI_CPU_ISA");
        return test_w8a8_forced_isa_case(requested) ? 0 : 1;
    }
#else
    (void)argc;
    (void)argv;
#endif
    /* The K block is derived from the detected L1 rather than fixed at 960.
     * The derivation is only trustworthy if it still lands on the value it
     * replaced for the cache size that value was chosen against, so check that
     * first and check that the live block is usable at all. */
    if (!vx_qgemm_kc_is_baseline()) {
        fprintf(stderr, "quantized K block no longer reproduces its baseline "
                        "on a 32 KiB L1\n");
        return 1;
    }
    if (vx_qgemm_kc() < 64u || vx_qgemm_kc() % 64u) {
        fprintf(stderr, "quantized K block %u is not a usable block\n",
                vx_qgemm_kc());
        return 1;
    }
    if (vx_packed_q8_preferred_for_native_w8a8(
            0u, 320u, 320u, VX_DTYPE_I8, 1) ||
        vx_packed_q8_preferred_for_native_w8a8(
            1u, 320u, 320u, VX_DTYPE_I8, 1)) return 1;
#if defined(__i386__) || defined(__x86_64__)
    if (vx_packed_q8_preferred_for_native_w8a8(
            2u, 320u, 320u, VX_DTYPE_I8, 1) != !!vx_cpu_has_avx2()) return 1;
    if (vx_packed_q8_preferred_for_native_w8a8(
            1u, 8u, 320u, VX_DTYPE_I8, 1) != !!vx_cpu_has_avx2()) return 1;
#else
    if (vx_packed_q8_preferred_for_native_w8a8(
            2u, 320u, 320u, VX_DTYPE_I8, 1) !=
        !!vx_kernel_platform()->has_arm_i8mm) return 1;
    if (vx_packed_q8_preferred_for_native_w8a8(
            2u, 37u, 48u, VX_DTYPE_I8, 1) !=
        !!vx_kernel_platform()->has_arm_i8mm) return 1;
#endif
    if (vx_packed_q8_preferred_for_native_w8a8(
            2u, 320u, 320u, VX_DTYPE_U8, 1) ||
        vx_packed_q8_preferred_for_native_w8a8(
            2u, 320u, 320u, VX_DTYPE_I8, 0)) return 1;
    if (!test_w8a32(1, 13, 11, VX_DTYPE_I8, 1u) ||
        !test_w8a32(5, 17, 9, VX_DTYPE_U8, 1u) ||
        !test_w8a32(4, 15, 13, VX_DTYPE_I8, 0u) ||
        !test_w8a8(1, 13, 11, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8) ||
        !test_w8a8(5, 17, 9, VX_DTYPE_U8, VX_DTYPE_I8,
                    VX_DTYPE_U8) ||
        !test_w8a8(3, 19, 13, VX_DTYPE_I8, VX_DTYPE_U8,
                    VX_DTYPE_U8) ||
        !test_w8a8_full_k_n16_all_activation_types() ||
        !test_w8a8_threaded_all_byte_types() ||
        !test_w8a8_symmetric_i8_exact_panel() ||
        !test_w8a8_signed_n32_single_thread() ||
        !test_w8a8_n32_nosat_single_thread() ||
        !test_metadata_and_malformed_packs() ||
        !test_w8a8_overflow_matches_portable() ||
        !test_w8a8_multiplier_representability_matches_portable()) return 1;
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__)) && !defined(_WIN32)
    if (!test_w8a8_forced_isa_children(argv[0])) return 1;
#endif
    puts("packed quantized GEMM correctness tests passed");
    return 0;
}
