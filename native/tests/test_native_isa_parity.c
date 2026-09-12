#include "fast_exp.h"
#include "attention_f32_isa.h"
#include "gemm_f32.h"
#include "inference_kernels.h"
#include "kernel_platform.h"
#include "quant_cpu_isa.h"
#include "thread_pool.h"
#include "volvoxai_enums.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                #condition);                                                 \
        return 0;                                                            \
    }                                                                        \
} while (0)

static uint32_t next_random(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void* checked_alloc(size_t bytes) {
    void* allocation = malloc(bytes ? bytes : 1u);
    if (!allocation) {
        fputs("out of memory\n", stderr);
        exit(2);
    }
    return allocation;
}

static void fill_bytes(void* values, size_t count, uint32_t* state) {
    uint8_t* bytes = (uint8_t*)values;
    for (size_t index = 0; index < count; index++)
        bytes[index] = (uint8_t)(next_random(state) >> 24u);
}

static int qlinear_case(uint32_t rows, uint32_t d_in, uint32_t d_out,
                        uint32_t input_dtype, uint32_t weight_dtype,
                        uint32_t output_dtype, uint32_t seed) {
    const size_t input_count = (size_t)rows * d_in;
    const size_t weight_count = (size_t)d_out * d_in;
    const size_t output_count = (size_t)rows * d_out;
    uint8_t* input = (uint8_t*)checked_alloc(input_count);
    uint8_t* weight = (uint8_t*)checked_alloc(weight_count);
    uint8_t* portable = (uint8_t*)checked_alloc(output_count);
    uint8_t* native = (uint8_t*)checked_alloc(output_count);
    int32_t* bias = (int32_t*)checked_alloc((size_t)d_out * sizeof(*bias));
    float* scales = (float*)checked_alloc((size_t)d_out * sizeof(*scales));
    int32_t* zero_points =
        (int32_t*)checked_alloc((size_t)d_out * sizeof(*zero_points));
    const int32_t input_zero = input_dtype == VX_DTYPE_I8 ? -17 : 137;
    const int32_t weight_zero = weight_dtype == VX_DTYPE_I8 ? 0 : 131;
    const int32_t output_zero = output_dtype == VX_DTYPE_I8 ? -9 : 129;
    int result;

    fill_bytes(input, input_count, &seed);
    fill_bytes(weight, weight_count, &seed);
    for (uint32_t column = 0; column < d_out; column++) {
        bias[column] = (int32_t)(next_random(&seed) % 2001u) - 1000;
        scales[column] = (float)(1u + column % 11u) / 1024.0f;
        zero_points[column] = weight_zero;
    }
    memset(portable, 0x55, output_count);
    memset(native, 0xaa, output_count);
    result = qlinear_i8u8(input, weight, bias, scales, zero_points, portable,
                          rows, d_in, d_out, 1.0f / 64.0f, input_zero,
                          1.0f / 8.0f, output_zero, input_dtype, weight_dtype,
                          output_dtype);
    CHECK(result == vx_qlinear_i8u8_native(
        input, weight, bias, scales, zero_points, native,
        rows, d_in, d_out, 1.0f / 64.0f, input_zero,
        1.0f / 8.0f, output_zero, input_dtype, weight_dtype, output_dtype));
    CHECK(result == 1 && memcmp(portable, native, output_count) == 0);

    free(zero_points);
    free(scales);
    free(bias);
    free(native);
    free(portable);
    free(weight);
    free(input);
    return 1;
}

static int test_qlinear(void) {
    static const uint32_t dtypes[] = {VX_DTYPE_I8, VX_DTYPE_U8};
    for (size_t input = 0; input < 2u; input++)
        for (size_t weight = 0; weight < 2u; weight++)
            for (size_t output = 0; output < 2u; output++)
                CHECK(qlinear_case(5u, 192u, 33u, dtypes[input],
                                   dtypes[weight], dtypes[output],
                                   (uint32_t)(100u + input * 17u +
                                              weight * 7u + output)));
    CHECK(qlinear_case(3u, 65u, 17u, VX_DTYPE_U8, VX_DTYPE_I8,
                       VX_DTYPE_I8, 911u));
    return 1;
}

static int qconv_case(uint32_t input_dtype, uint32_t output_dtype,
                      uint32_t kernel_height, uint32_t kernel_width,
                      uint32_t padding_top, uint32_t padding_left,
                      uint32_t padding_bottom, uint32_t padding_right,
                      uint32_t seed) {
    enum {
        batch = 1, height = 5, width = 7, input_channels = 64,
        output_channels = 33
    };
    const size_t input_count =
        (size_t)batch * height * width * input_channels;
    const size_t weight_count =
        (size_t)output_channels * kernel_height * kernel_width *
        input_channels;
    const size_t output_count =
        (size_t)batch * height * width * output_channels;
    uint8_t* input = (uint8_t*)checked_alloc(input_count);
    int8_t* weight = (int8_t*)checked_alloc(weight_count);
    uint8_t* portable = (uint8_t*)checked_alloc(output_count);
    uint8_t* native = (uint8_t*)checked_alloc(output_count);
    int32_t bias[output_channels];
    float scales[output_channels];
    int32_t zero_points[output_channels] = {0};
    const int32_t input_zero = input_dtype == VX_DTYPE_I8 ? -11 : 133;
    const int32_t output_zero = output_dtype == VX_DTYPE_I8 ? 7 : 127;
    int portable_status;

    fill_bytes(input, input_count, &seed);
    fill_bytes(weight, weight_count, &seed);
    for (uint32_t channel = 0; channel < output_channels; channel++) {
        bias[channel] = (int32_t)(next_random(&seed) % 2001u) - 1000;
        scales[channel] = (float)(1u + channel % 9u) / 2048.0f;
    }
#define QCONV_ARGS(output)                                                   \
    input, weight, bias, scales, zero_points, (output),                      \
    batch, height, width, input_channels, height, width, output_channels,    \
    kernel_height, kernel_width, input_channels, 1u, 1u, 1u, 1u,             \
    padding_top, padding_left, padding_bottom, padding_right, 1u, 0u,         \
    1.0f / 64.0f, input_zero, 1.0f / 8.0f, output_zero,                      \
    input_dtype, VX_DTYPE_I8, output_dtype
    portable_status = qconv2d_i8u8(QCONV_ARGS(portable));
    CHECK(portable_status == vx_qconv2d_i8u8_native(QCONV_ARGS(native)));
    CHECK(portable_status == 1 &&
          memcmp(portable, native, output_count) == 0);
#undef QCONV_ARGS
    free(native);
    free(portable);
    free(weight);
    free(input);
    return 1;
}

static int test_qconv(void) {
    /* Standard encoder im2col route, then an asymmetric raw direct route. */
    CHECK(qconv_case(VX_DTYPE_U8, VX_DTYPE_U8,
                     3u, 3u, 1u, 1u, 1u, 1u, 1201u));
    CHECK(qconv_case(VX_DTYPE_I8, VX_DTYPE_I8,
                     2u, 3u, 0u, 1u, 1u, 1u, 1202u));
    return 1;
}

typedef struct {
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t output_dtype;
    int32_t a_zero;
    int32_t b_zero;
    int32_t output_zero;
} QBatchDtype;

static int qbatch_case(const QBatchDtype* dtypes, uint32_t seed) {
    enum { m = 17, k = 65, n = 33 };
    const size_t a_count = (size_t)m * k;
    const size_t b_count = (size_t)k * n;
    const size_t output_count = (size_t)m * n;
    uint8_t* a = (uint8_t*)checked_alloc(a_count);
    uint8_t* b = (uint8_t*)checked_alloc(b_count);
    uint8_t* portable = (uint8_t*)checked_alloc(output_count);
    uint8_t* native = (uint8_t*)checked_alloc(output_count);
    const size_t workspace_bytes =
        vx_qbatch_matmul_i8u8_native_workspace_bytes(m, k, n);
    void* workspace = workspace_bytes ? checked_alloc(workspace_bytes) : NULL;
    int portable_status;

    fill_bytes(a, a_count, &seed);
    fill_bytes(b, b_count, &seed);
    portable_status = qbatch_matmul_i8u8(
        a, b, portable, m, k, n, 1.0f / 64.0f, dtypes->a_zero,
        1.0f / 32.0f, dtypes->b_zero, 1.0f / 4.0f,
        dtypes->output_zero, dtypes->a_dtype, dtypes->b_dtype,
        dtypes->output_dtype);
    CHECK(portable_status == vx_qbatch_matmul_i8u8_native_with_workspace(
        a, b, native, m, k, n, 1.0f / 64.0f, dtypes->a_zero,
        1.0f / 32.0f, dtypes->b_zero, 1.0f / 4.0f,
        dtypes->output_zero, dtypes->a_dtype, dtypes->b_dtype,
        dtypes->output_dtype, workspace, workspace_bytes));
    CHECK(portable_status == 1 &&
          memcmp(portable, native, output_count) == 0);
    free(workspace);
    free(native);
    free(portable);
    free(b);
    free(a);
    return 1;
}

static int test_qbatch(void) {
    static const QBatchDtype dtypes[] = {
        {VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8, 137, 0, 129},
        {VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, 17, -5, -9},
        {VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, -7, 3, 0},
        {VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_U8, -11, 131, 127},
    };
    for (size_t index = 0; index < sizeof(dtypes) / sizeof(dtypes[0]); index++)
        CHECK(qbatch_case(&dtypes[index], 1701u + (uint32_t)index));
    return 1;
}

static float next_float(uint32_t* state) {
    return (float)((int32_t)(next_random(state) % 2001u) - 1000) / 1000.0f;
}

static int gemm_case(uint32_t m, uint32_t k, uint32_t n, uint32_t seed) {
    const uint32_t packed_elements = vx_gemm_f32_packed_elements(k, n);
    float* a = (float*)checked_alloc((size_t)m * k * sizeof(*a));
    float* b = (float*)checked_alloc((size_t)k * n * sizeof(*b));
    float* bias = (float*)checked_alloc((size_t)n * sizeof(*bias));
    float* packed =
        (float*)checked_alloc((size_t)packed_elements * sizeof(*packed));
    float* output = (float*)checked_alloc((size_t)m * n * sizeof(*output));
    double worst = 0.0;

    CHECK(packed_elements != 0u);
    for (size_t index = 0; index < (size_t)m * k; index++)
        a[index] = next_float(&seed);
    for (size_t index = 0; index < (size_t)k * n; index++)
        b[index] = next_float(&seed);
    for (uint32_t column = 0; column < n; column++)
        bias[column] = next_float(&seed);
    CHECK(vx_gemm_f32_pack_b(b, packed, k, n, 0));
    CHECK(vx_gemm_f32_run_packed(a, packed, bias, output, m, k, n));
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t column = 0; column < n; column++) {
            double reference = bias[column];
            double scale;
            double error;
            for (uint32_t reduction = 0; reduction < k; reduction++)
                reference += (double)a[(size_t)row * k + reduction] *
                    (double)b[(size_t)reduction * n + column];
            scale = fabs(reference) > 1.0 ? fabs(reference) : 1.0;
            error = fabs((double)output[(size_t)row * n + column] - reference) /
                scale;
            if (error > worst) worst = error;
        }
    }
    CHECK(isfinite(worst) && worst <= 2.5e-5);
    free(output);
    free(packed);
    free(bias);
    free(b);
    free(a);
    return 1;
}

static int test_gemm(const VxKernelPlatform* platform) {
    const VxGemmF32Plan edge = vx_gemm_f32_plan(7u, 113u, 53u);
    const VxGemmF32Plan decode = vx_gemm_f32_plan(1u, 320u, 320u);
    CHECK(edge.regime == VX_GEMM_F32_REGIME_BLOCK);
    CHECK(decode.regime == VX_GEMM_F32_REGIME_STREAM);
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    if (platform->has_avx2 && __builtin_cpu_supports("fma")) {
        CHECK(edge.micro != 0 && decode.micro != 0);
        if (platform->has_avx512f) {
            CHECK(edge.mr == 8u && edge.nr == 32u);
        } else {
            CHECK(edge.mr == 6u && edge.nr == 16u);
        }
    } else {
        CHECK(edge.micro == 0 && decode.micro == 0);
    }
#elif defined(__aarch64__) || defined(__arm__)
    CHECK((edge.micro != 0) == (platform->has_neon != 0));
#else
    CHECK(edge.micro == 0 && decode.micro == 0);
#endif
    CHECK(gemm_case(7u, 113u, 53u, 2301u));
    CHECK(gemm_case(1u, 320u, 320u, 2302u));
    return 1;
}

#define ATTENTION_Q 19
#define ATTENTION_KV 23
#define ATTENTION_HEADS 3
#define ATTENTION_HEAD_DIM 8
#define ATTENTION_WIDTH (ATTENTION_HEADS * ATTENTION_HEAD_DIM)

static int attention_case(const float* q, const float* k, const float* v,
                          const int32_t* mask, int mask_mode, int causal) {
#if VX_SDPA_X86_AVX2
    float portable[ATTENTION_Q * ATTENTION_WIDTH];
    float tiled[ATTENTION_Q * ATTENTION_WIDTH];
    float worst = 0.0f;
    vx_attention_f32_portable(q, k, v, ATTENTION_WIDTH, portable,
                              ATTENTION_WIDTH, ATTENTION_Q, ATTENTION_KV,
                              ATTENTION_HEADS, ATTENTION_HEAD_DIM, 0.35f,
                              mask, mask_mode, causal);
    vx_attention_tiled_avx2(q, k, v, ATTENTION_WIDTH, tiled,
                            ATTENTION_WIDTH, ATTENTION_Q, ATTENTION_KV,
                            ATTENTION_HEADS, ATTENTION_HEAD_DIM, 0.35f,
                            mask, mask_mode, causal);
    for (size_t index = 0;
         index < (size_t)ATTENTION_Q * ATTENTION_WIDTH; index++) {
        const float scale = fabsf(portable[index]) > 1.0f
            ? fabsf(portable[index]) : 1.0f;
        const float error = fabsf(tiled[index] - portable[index]) / scale;
        if (error > worst) worst = error;
    }
    CHECK(isfinite(worst) && worst <= 2.0e-5f);
#else
    (void)q;
    (void)k;
    (void)v;
    (void)mask;
    (void)mask_mode;
    (void)causal;
#endif
    return 1;
}

static int test_attention(const VxKernelPlatform* platform) {
    float q[ATTENTION_Q * ATTENTION_WIDTH];
    float k[ATTENTION_KV * ATTENTION_WIDTH];
    float v[ATTENTION_KV * ATTENTION_WIDTH];
    int32_t key_mask[ATTENTION_KV];
    int32_t query_key_mask[ATTENTION_Q * ATTENTION_KV];
    uint32_t seed = 2901u;

#if VX_SDPA_X86_AVX2
    if (!platform->has_avx2) return 1;
#else
    (void)platform;
    return 1;
#endif
    for (size_t index = 0; index < sizeof(q) / sizeof(q[0]); index++)
        q[index] = next_float(&seed);
    for (size_t index = 0; index < sizeof(k) / sizeof(k[0]); index++) {
        k[index] = next_float(&seed);
        v[index] = next_float(&seed);
    }
    for (int key = 0; key < ATTENTION_KV; key++)
        key_mask[key] = key % 3 != 0;
    for (int query = 0; query < ATTENTION_Q; query++)
        for (int key = 0; key < ATTENTION_KV; key++)
            query_key_mask[query * ATTENTION_KV + key] =
                (query * 5 + key * 3) % 4 != 0;
    CHECK(attention_case(q, k, v, NULL, 0, 0));
    CHECK(attention_case(q, k, v, NULL, 0, 1));
    CHECK(attention_case(q, k, v, key_mask, 1, 0));
    CHECK(attention_case(q, k, v, query_key_mask, 2, 1));
    return 1;
}

int main(int argc, char** argv) {
    const char* environment = getenv("VOLVOXAI_CPU_ISA");
    const VxKernelPlatform* platform;
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int ok;
    if (argc != 3 || strcmp(argv[1], "--isa") || !environment ||
        strcmp(environment, argv[2])) {
        fprintf(stderr, "usage: VOLVOXAI_CPU_ISA=TIER %s --isa TIER\n",
                argv[0]);
        return 2;
    }
    platform = vx_kernel_platform();
    if (!platform->configuration_valid) {
        fprintf(stderr, "SKIP: host cannot run requested ISA tier %s\n",
                argv[2]);
        return 77;
    }
    if (strcmp(vx_kernel_isa_name(platform->requested_isa), argv[2]) ||
        strcmp(vx_kernel_isa_name(platform->isa), argv[2]))
        return 1;
    pool = vx_kernel_thread_pool_create(4);
    if (!pool) return 1;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    ok = vx_kernels_thread_count() == 4 && test_qlinear() && test_qconv() &&
         test_qbatch() && test_gemm(platform) && test_attention(platform);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    if (!ok) return 1;
    printf("native raw ISA parity passed under clamp %s\n", argv[2]);
    return 0;
}
