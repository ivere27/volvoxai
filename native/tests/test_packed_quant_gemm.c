#include "inference_kernels.h"
#include "kernel_platform.h"
#include "packed_quant_gemm.h"
#include "thread_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                #condition);                                                 \
        return 0;                                                            \
    }                                                                        \
} while (0)

static int test_pack_validation(void) {
    enum { d_in = 5, d_out = 3 };
    static const int8_t weight[d_out * d_in] = {
        -64, -7, 0, 9, 64,
        13, -22, 31, -40, 49,
        5, 4, 3, 2, 1,
    };
    const uint8_t input[d_in] = {0, 37, 128, 201, 255};
    const int32_t bias[d_out] = {17, -23, 41};
    const float scales[d_out] = {0.125f, 0.25f, 0.5f};
    const int32_t zero_points[d_out] = {0, 0, 0};
    int8_t output[d_out];
    const uint32_t bytes = vx_packed_q8_weight_size(d_in, d_out);
    uint8_t* packed = (uint8_t*)malloc(bytes);
    uint8_t* corrupted = (uint8_t*)malloc(bytes);

    CHECK(bytes >= sizeof(VxPackedQ8Header));
    CHECK(packed && corrupted);
    CHECK(vx_packed_q8_weight_size(0u, d_out) == 0u);
    CHECK(vx_packed_q8_weight_size(d_in, 0u) == 0u);
    CHECK(vx_packed_q8_weight_size(
              (uint32_t)(INT32_MAX / 255) + 1u, 1u) == 0u);
    CHECK(!vx_pack_q8_weight(
        packed, bytes - 1u, weight, d_in, d_out, VX_DTYPE_I8, 1u));
    CHECK(!vx_pack_q8_weight(
        packed, bytes, weight, d_in, d_out, VX_DTYPE_F32, 1u));
    CHECK(!vx_pack_q8_weight(
        packed, bytes, weight, d_in, d_out, VX_DTYPE_I8, 2u));
    CHECK(vx_pack_q8_weight(
        packed, bytes, weight, d_in, d_out, VX_DTYPE_I8, 1u));

    memcpy(corrupted, packed, bytes);
    ((VxPackedQ8Header*)(void*)corrupted)->magic = 0u;
    CHECK(!vx_qlinear_i8u8_packed(
        input, corrupted, bias, scales, zero_points, output,
        1u, d_in, d_out, 1.0f / 64.0f, 137, 1.0f, -3,
        VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));

    free(corrupted);
    free(packed);
    return 1;
}

typedef enum {
    VX_TEST_BOUNDED_WEIGHT,
    VX_TEST_SIGNED_WEIGHT,
    VX_TEST_FULL_WEIGHT,
} VxTestWeightCase;

static int test_byte_exact_case(VxTestWeightCase weight_case,
                                uint32_t input_dtype,
                                uint32_t weight_dtype,
                                uint32_t output_dtype,
                                uint32_t rows, uint32_t d_in,
                                uint32_t d_out) {
    const size_t input_bytes = (size_t)rows * d_in;
    const size_t weight_bytes = (size_t)d_out * d_in;
    const size_t output_bytes = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)malloc(input_bytes);
    uint8_t* weight = (uint8_t*)malloc(weight_bytes);
    int32_t bias[d_out];
    float scales[d_out];
    int32_t zero_points[d_out];
    uint8_t* portable = (uint8_t*)malloc(output_bytes);
    uint8_t* packed_output = (uint8_t*)malloc(output_bytes);
    const uint32_t packed_bytes = vx_packed_q8_weight_size(d_in, d_out);
    uint8_t* packed = (uint8_t*)malloc(packed_bytes);
    const int32_t input_zero_point = input_dtype == VX_DTYPE_I8 ? -13 : 137;
    const int32_t weight_zero_point = weight_dtype == VX_DTYPE_I8 ? 0 : 131;
    const int32_t output_zero_point = output_dtype == VX_DTYPE_I8 ? -3 : 129;

    CHECK(input && weight && portable && packed_output && packed && packed_bytes);
    for (size_t index = 0u; index < input_bytes; index++)
        input[index] = (uint8_t)(index * 197u + index / 3u + 251u);
    for (size_t index = 0u; index < weight_bytes; index++) {
        if (weight_dtype == VX_DTYPE_U8) {
            weight[index] = (uint8_t)(index * 89u + 17u);
        } else if (weight_case == VX_TEST_BOUNDED_WEIGHT) {
            weight[index] = (uint8_t)(int8_t)((int)(index * 67u % 129u) - 64);
        } else {
            weight[index] = (uint8_t)(int8_t)((int)(index * 89u % 255u) - 127);
        }
    }
    if (weight_dtype == VX_DTYPE_I8) {
        if (weight_case == VX_TEST_FULL_WEIGHT) weight[0] = (uint8_t)-128;
        else weight[0] = (uint8_t)(weight_case == VX_TEST_BOUNDED_WEIGHT ? -64 : -127);
        weight[1] = (uint8_t)(weight_case == VX_TEST_BOUNDED_WEIGHT ? 64 : 127);
    }

    for (uint32_t column = 0u; column < d_out; column++) {
        bias[column] = (int32_t)column * 31 - 997;
        scales[column] = (float)(1u + column % 7u) / 512.0f;
        zero_points[column] = weight_zero_point;
    }

    CHECK(vx_pack_q8_weight(packed, packed_bytes, weight, d_in, d_out,
                            weight_dtype, 1u));

    CHECK(qlinear_i8u8(
        input, weight, bias, scales, zero_points, portable,
        rows, d_in, d_out, 1.0f / 64.0f, input_zero_point, 1.0f,
        output_zero_point, input_dtype, weight_dtype, output_dtype));
    CHECK(vx_qlinear_i8u8_packed(
        input, packed, bias, scales, zero_points, packed_output,
        rows, d_in, d_out, 1.0f / 64.0f, input_zero_point, 1.0f,
        output_zero_point, input_dtype, weight_dtype, output_dtype));
    CHECK(memcmp(portable, packed_output, output_bytes) == 0);

    free(packed);
    free(packed_output);
    free(portable);
    free(weight);
    free(input);
    return 1;
}

static int test_byte_exact_parity(uint32_t rows, uint32_t d_in,
                                  uint32_t d_out) {
    static const uint32_t dtypes[] = {VX_DTYPE_I8, VX_DTYPE_U8};
    for (size_t input = 0; input < 2u; input++) {
        for (size_t weight = 0; weight < 2u; weight++) {
            for (size_t output = 0; output < 2u; output++) {
                CHECK(test_byte_exact_case(
                    VX_TEST_FULL_WEIGHT, dtypes[input], dtypes[weight],
                    dtypes[output], rows, d_in, d_out));
            }
        }
    }
    /* These ranges choose different AVX2 accumulation routes. */
    CHECK(test_byte_exact_case(VX_TEST_BOUNDED_WEIGHT, VX_DTYPE_U8,
                               VX_DTYPE_I8, VX_DTYPE_I8,
                               rows, d_in, d_out));
    CHECK(test_byte_exact_case(VX_TEST_SIGNED_WEIGHT, VX_DTYPE_U8,
                               VX_DTYPE_I8, VX_DTYPE_I8,
                               rows, d_in, d_out));
    return 1;
}

static int test_selection_boundary(void) {
    const VxKernelPlatform* platform = vx_kernel_platform();
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        0u, 31u, 32u, VX_DTYPE_I8, 1));
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 15u, VX_DTYPE_I8, 1));
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 32u, VX_DTYPE_U8, 1));
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 32u, VX_DTYPE_I8, 0));
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
    CHECK(vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 32u, VX_DTYPE_I8, 1) == platform->has_avx2);
    CHECK(vx_packed_q8_preferred_for_native_w8a8(
        1u, 31u, 32u, VX_DTYPE_I8, 1) == platform->has_avx2);
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        1u, 32u, 32u, VX_DTYPE_I8, 1));
#elif defined(VOLVOXAI_ARM_I8MM_OBJECT) && defined(__aarch64__)
    CHECK(vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 32u, VX_DTYPE_I8, 1) == platform->has_arm_i8mm);
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        1u, 31u, 32u, VX_DTYPE_I8, 1));
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 33u, VX_DTYPE_I8, 1));
#else
    CHECK(!vx_packed_q8_preferred_for_native_w8a8(
        2u, 31u, 32u, VX_DTYPE_I8, 1));
#endif
    return 1;
}

static int run_contract(void) {
    const VxKernelPlatform* platform = vx_kernel_platform();
    CHECK(platform->configuration_valid);
    CHECK(vx_qgemm_kc_is_baseline());
    CHECK(vx_qgemm_kc() >= 64u && vx_qgemm_kc() % 64u == 0u);
    CHECK(test_pack_validation());
    CHECK(test_selection_boundary());
    {
        VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
        VxKernelThreadPoolScope scope;
        CHECK(pool != NULL);
        scope = vx_kernel_thread_pool_scope_enter(pool);
        CHECK(vx_kernels_thread_count() == 4);
        /* 113*37*257 crosses the production parallel-product threshold while
         * leaving M/K/N tails in every tiled body. */
        CHECK(test_byte_exact_parity(113u, 37u, 257u));
        vx_kernel_thread_pool_scope_leave(scope);
        vx_kernel_thread_pool_destroy(pool);
    }
    puts("packed quantized GEMM contract passed");
    return 1;
}

static int run_forced_tier(const char* requested) {
    const char* environment = getenv("VOLVOXAI_CPU_ISA");
    const VxKernelPlatform* platform;
    CHECK(requested && environment && strcmp(requested, environment) == 0);
    platform = vx_kernel_platform();
    if (!platform->configuration_valid) {
        fprintf(stderr, "SKIP: host cannot run requested ISA tier %s\n",
                requested);
        return 77;
    }
    CHECK(strcmp(vx_kernel_isa_name(platform->requested_isa), requested) == 0);
    CHECK(strcmp(vx_kernel_isa_name(platform->isa), requested) == 0);
    CHECK(test_selection_boundary());
    CHECK(test_byte_exact_parity(5u, 37u, 65u));
    printf("packed quantized GEMM byte parity passed under clamp %s\n",
           requested);
    return 1;
}

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && !strcmp(argv[1], "--contract")))
        return run_contract() ? 0 : 1;
    if (argc == 3 && !strcmp(argv[1], "--isa")) {
        const int result = run_forced_tier(argv[2]);
        return result == 77 ? 77 : (result ? 0 : 1);
    }
    fprintf(stderr, "usage: %s [--contract | --isa TIER]\n", argv[0]);
    return 2;
}
