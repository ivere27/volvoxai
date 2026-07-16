/*
 * Direct native W8A8 and W8A32 kernel benchmark.
 *
 * This is deliberately separate from correctness tests: timing is inherently
 * host-dependent. Before timing each workload it checks both accelerated
 * layouts against their portable references. The decoder proxy uses the
 * actual TinyReceipt incremental-row dense shapes and call counts.
 */
#define _POSIX_C_SOURCE 200809L

#include "quant_cpu_opt.h"
#include "packed_quant_gemm.h"
#include "cpu_features.h"
#include "thread_pool.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int qlinear_i8u8(const void* input, const void* weight, const int32_t* bias,
                        const float* weight_scales, const int32_t* weight_zero_points,
                        void* output, uint32_t rows, uint32_t d_in, uint32_t d_out,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t weight_dtype,
                        uint32_t output_dtype);

extern int qconv2d_i8u8(const void* input, const void* weight, const int32_t* bias,
                        const float* weight_scales, const int32_t* weight_zero_points,
                        void* output, uint32_t batch, uint32_t input_height,
                        uint32_t input_width, uint32_t input_channels,
                        uint32_t output_height, uint32_t output_width,
                        uint32_t output_channels, uint32_t kernel_height,
                        uint32_t kernel_width, uint32_t input_per_group,
                        uint32_t stride_y, uint32_t stride_x, uint32_t dilation_y,
                        uint32_t dilation_x, uint32_t padding_top, uint32_t padding_left,
                        uint32_t padding_bottom, uint32_t padding_right, uint32_t groups,
                        uint32_t relu, float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t weight_dtype,
                        uint32_t output_dtype);

extern int qsilu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype);

extern int qgelu_i8u8(const void* input, void* output, uint32_t elements,
                        float input_scale, int32_t input_zero_point,
                        float output_scale, int32_t output_zero_point,
                        uint32_t input_dtype, uint32_t output_dtype);

extern int matmul_quantized_f32(const float* input, const void* weight,
                        const float* scale, const void* zero_point,
                        const float* bias, float* output, uint32_t rows,
                        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
                        uint32_t scale_elements, uint32_t zero_point_dtype,
                        uint32_t zero_point_elements);

static double elapsed_ms(const struct timespec* start, const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
        (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static uint32_t checksum_i8(const int8_t* values, size_t count) {
    uint32_t checksum = 0x564f4c58u;
    for (size_t index = 0; index < count; index++) {
        checksum = checksum * 16777619u ^ (uint8_t)values[index];
    }
    return checksum;
}

typedef int (*QLinearW8A8Fn)(const void*, const void*, const int32_t*,
        const float*, const int32_t*, void*, uint32_t, uint32_t, uint32_t,
        float, int32_t, float, int32_t, uint32_t, uint32_t, uint32_t);

typedef int (*LinearW8A32Fn)(const float*, const void*, const float*,
        const void*, const float*, float*, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t);

typedef struct {
    const char* label;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t calls_per_token;
    int iterations;
} DecoderLinearCase;

typedef struct {
    double w8a8_raw_ms;
    double w8a8_packed_ms;
    double w8a32_packed_ms;
} DecoderLinearTiming;

static const char* raw_w8a8_path(uint32_t d_in) {
#if defined(__i386__) || defined(__x86_64__)
    if (d_in >= 64u && vx_cpu_has_avx512_vnni()) return "avx512-vnni-zmm";
    if (d_in >= 32u && vx_cpu_has_avx_vnni()) return "avx-vnni";
    if (d_in >= 16u && vx_cpu_has_avx2()) return "avx2";
#elif defined(__aarch64__) || defined(__arm__)
#if defined(VOLVOXAI_ARM_DOTPROD_OBJECT)
    if (d_in >= 16u && vx_cpu_has_arm_dotprod()) return "arm-sdot";
#endif
    if (d_in >= 8u && vx_cpu_has_neon()) return "neon";
#else
    (void)d_in;
#endif
    return "portable";
}

static const char* qconv_w8a8_path(uint32_t input_per_group) {
#if defined(__i386__) || defined(__x86_64__)
    if (input_per_group >= 64u && vx_cpu_has_avx512_vnni())
        return "avx512-vnni-zmm";
    if (input_per_group >= 32u && vx_cpu_has_avx_vnni()) return "avx-vnni";
    if (vx_cpu_has_avx2())
        return input_per_group < 16u ? "avx2-oc8-small-c" : "avx2-oc4";
#elif defined(__aarch64__) || defined(__arm__)
#if defined(VOLVOXAI_ARM_DOTPROD_OBJECT)
    if (input_per_group >= 16u && vx_cpu_has_arm_dotprod()) return "arm-sdot";
#endif
    if (input_per_group >= 8u && vx_cpu_has_neon()) return "neon";
#else
    (void)input_per_group;
#endif
    return "portable";
}

static const char* packed_w8a8_path(void) {
#if defined(__i386__) || defined(__x86_64__)
    if (vx_cpu_has_avx2()) return "avx2-n8";
#endif
    return "portable-packed";
}

static double time_w8a8(QLinearW8A8Fn function, const void* input,
        const void* weight, const int32_t* bias, const float* scales,
        const int32_t* zero_points, void* output, uint32_t d_in,
        uint32_t d_out, float input_scale, int32_t input_zero_point,
        float output_scale, int iterations) {
    enum { warmup = 3 };
    struct timespec start, end;
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                1u, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, 2u, 2u, 2u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                1u, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, 2u, 2u, 2u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_ms(&start, &end) / iterations;
}

static double time_w8a8_rows(QLinearW8A8Fn function, const void* input,
        const void* weight, const int32_t* bias, const float* scales,
        const int32_t* zero_points, void* output, uint32_t rows,
        uint32_t d_in, uint32_t d_out, float input_scale,
        int32_t input_zero_point, float output_scale, int iterations,
        int threads) {
    enum { warmup = 2 };
    struct timespec start, end;
    vx_set_num_threads(threads);
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                rows, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, 2u, 2u, 2u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                rows, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, 2u, 2u, 2u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_ms(&start, &end) / iterations;
}

static double time_w8a32(LinearW8A32Fn function, const float* input,
        const void* weight, const float* scales, const int32_t* zero_points,
        const float* bias, float* output, uint32_t d_in, uint32_t d_out,
        int iterations) {
    enum { warmup = 3 };
    struct timespec start, end;
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, scales, zero_points, bias, output,
                1u, d_in, d_out, 2u, d_out, 1u, d_out)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!function(input, weight, scales, zero_points, bias, output,
                1u, d_in, d_out, 2u, d_out, 1u, d_out)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_ms(&start, &end) / iterations;
}

/* This is a kernel-only comparison.  It deliberately feeds W8A8 and W8A32
 * the same quantized weights/scales and the same activation values (the W8A32
 * input is the exact dequantization of the I8 input).  It is not an accuracy
 * or end-to-end claim for a W8A32 TinyReceipt package, which does not exist. */
static int run_decoder_linear_case(const DecoderLinearCase* test,
                                   DecoderLinearTiming* timing) {
    const float input_scale = 1.0f / 32.0f;
    const int32_t input_zero_point = -3;
    const size_t input_count = test->d_in;
    const size_t weight_count = (size_t)test->d_out * test->d_in;
    const size_t output_count = test->d_out;
    const uint32_t packed_bytes = vx_packed_q8_weight_size(test->d_in, test->d_out);
    int8_t* input_i8 = (int8_t*)malloc(input_count);
    float* input_f32 = (float*)malloc(input_count * sizeof(*input_f32));
    int8_t* weight = (int8_t*)malloc(weight_count);
    int32_t* bias_i32 = (int32_t*)malloc(output_count * sizeof(*bias_i32));
    float* bias_f32 = (float*)malloc(output_count * sizeof(*bias_f32));
    float* scales = (float*)malloc(output_count * sizeof(*scales));
    int32_t* zero_points = (int32_t*)malloc(output_count * sizeof(*zero_points));
    int8_t* w8a8_reference = (int8_t*)malloc(output_count);
    int8_t* w8a8_raw = (int8_t*)malloc(output_count);
    int8_t* w8a8_packed = (int8_t*)malloc(output_count);
    float* w8a32_reference = (float*)malloc(output_count * sizeof(*w8a32_reference));
    float* w8a32_packed = (float*)malloc(output_count * sizeof(*w8a32_packed));
    void* packed_weight = malloc(packed_bytes);
    float output_scale = 0.0f;
    float max_abs = 0.0f;
    float max_w8a32_delta = 0.0f;
    float max_cross_mode_delta = 0.0f;
    int ok = 0;
    if (!input_i8 || !input_f32 || !weight || !bias_i32 || !bias_f32 ||
        !scales || !zero_points || !w8a8_reference || !w8a8_raw ||
        !w8a8_packed || !w8a32_reference || !w8a32_packed || !packed_weight)
        goto cleanup;
    for (size_t index = 0; index < input_count; index++) {
        input_i8[index] = (int8_t)((int)(index * 29u % 127u) - 63);
        input_f32[index] = ((float)input_i8[index] - (float)input_zero_point) * input_scale;
    }
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int)(index * 17u % 111u) - 55);
    for (uint32_t index = 0; index < test->d_out; index++) {
        bias_i32[index] = (int32_t)(index * 13u) - 901;
        scales[index] = 1.0f / (float)(64u + index % 5u * 8u);
        zero_points[index] = (int32_t)(index % 9u) - 4;
        bias_f32[index] = (float)bias_i32[index] * input_scale * scales[index];
    }
    if (!vx_pack_q8_weight(packed_weight, packed_bytes, weight, test->d_in,
            test->d_out, 2u, 1u) ||
        !matmul_quantized_f32(input_f32, weight, scales, zero_points, bias_f32,
            w8a32_reference, 1u, test->d_in, test->d_out, 2u,
            test->d_out, 1u, test->d_out) ||
        !vx_matmul_quantized_f32_packed(input_f32, packed_weight, scales,
            zero_points, bias_f32, w8a32_packed, 1u, test->d_in,
            test->d_out, 2u, test->d_out, 1u, test->d_out)) goto cleanup;
    for (size_t index = 0; index < output_count; index++) {
        float value_abs = fabsf(w8a32_reference[index]);
        float delta = fabsf(w8a32_reference[index] - w8a32_packed[index]);
        float tolerance = 1.0e-6f * fmaxf(1.0f, value_abs);
        if (value_abs > max_abs) max_abs = value_abs;
        if (delta > max_w8a32_delta) max_w8a32_delta = delta;
        if (delta > tolerance) goto cleanup;
    }
    output_scale = fmaxf(max_abs / 120.0f, 1.0e-6f);
    if (!qlinear_i8u8(input_i8, weight, bias_i32, scales, zero_points,
            w8a8_reference, 1u, test->d_in, test->d_out, input_scale,
            input_zero_point, output_scale, 0, 2u, 2u, 2u) ||
        !vx_qlinear_i8u8_native(input_i8, weight, bias_i32, scales, zero_points,
            w8a8_raw, 1u, test->d_in, test->d_out, input_scale,
            input_zero_point, output_scale, 0, 2u, 2u, 2u) ||
        !vx_qlinear_i8u8_packed(input_i8, packed_weight, bias_i32, scales,
            zero_points, w8a8_packed, 1u, test->d_in, test->d_out,
            input_scale, input_zero_point, output_scale, 0, 2u, 2u, 2u) ||
        memcmp(w8a8_reference, w8a8_raw, output_count) != 0 ||
        memcmp(w8a8_reference, w8a8_packed, output_count) != 0) goto cleanup;
    for (size_t index = 0; index < output_count; index++) {
        float dequantized = (float)w8a8_reference[index] * output_scale;
        float delta = fabsf(dequantized - w8a32_reference[index]);
        float tolerance = output_scale * 0.51f +
            1.0e-5f * fmaxf(1.0f, fabsf(w8a32_reference[index]));
        if (delta > max_cross_mode_delta) max_cross_mode_delta = delta;
        if (delta > tolerance) goto cleanup;
    }
    timing->w8a8_raw_ms = time_w8a8(vx_qlinear_i8u8_native, input_i8,
        weight, bias_i32, scales, zero_points, w8a8_raw, test->d_in,
        test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations);
    timing->w8a8_packed_ms = time_w8a8(vx_qlinear_i8u8_packed, input_i8,
        packed_weight, bias_i32, scales, zero_points, w8a8_packed,
        test->d_in, test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations);
    timing->w8a32_packed_ms = time_w8a32(vx_matmul_quantized_f32_packed,
        input_f32, packed_weight, scales, zero_points, bias_f32, w8a32_packed,
        test->d_in, test->d_out, test->iterations);
    if (timing->w8a8_raw_ms < 0.0 || timing->w8a8_packed_ms < 0.0 ||
        timing->w8a32_packed_ms < 0.0) goto cleanup;
    printf("  %-24s calls=%2u M=1 K=%4u N=%4u raw[%s]=%.4f ms "
           "packed[%s]=%.4f ms W8A32[packed-NR8]=%.4f ms "
           "W8A32/raw=%.2fx W8A32/packed=%.2fx check=pass "
           "(f32=%.3g cross=%.3g i8=0x%08x)\n",
           test->label, test->calls_per_token, test->d_in, test->d_out,
           raw_w8a8_path(test->d_in), timing->w8a8_raw_ms,
           packed_w8a8_path(), timing->w8a8_packed_ms,
           timing->w8a32_packed_ms,
           timing->w8a32_packed_ms / timing->w8a8_raw_ms,
           timing->w8a32_packed_ms / timing->w8a8_packed_ms,
           (double)max_w8a32_delta, (double)max_cross_mode_delta,
           checksum_i8(w8a8_packed, output_count));
    ok = 1;
cleanup:
    free(input_i8); free(input_f32); free(weight); free(bias_i32);
    free(bias_f32); free(scales); free(zero_points); free(w8a8_reference);
    free(w8a8_raw); free(w8a8_packed); free(w8a32_reference);
    free(w8a32_packed); free(packed_weight);
    return ok;
}

static int run_tiny_vqa_decoder_linear_mix(void) {
    /* Exact unique QLinear mix for one incremental row in the materialized
     * TinyReceipt graph: four decoder layers, one adapter and one vocab head.
     * Encoder work and cached cross-attention K/V projections are excluded. */
    static const DecoderLinearCase cases[] = {
        {"attention 320->320", 320u, 320u, 24u, 800},
        {"FFN base 320->1280", 320u, 1280u, 4u, 250},
        {"FFN LoRA-A 320->8", 320u, 8u, 4u, 10000},
        {"FFN LoRA-B 8->1280", 8u, 1280u, 4u, 1000},
        {"FFN base 1280->320", 1280u, 320u, 4u, 400},
        {"FFN LoRA-A 1280->8", 1280u, 8u, 4u, 5000},
        {"FFN LoRA-B 8->320", 8u, 320u, 4u, 3000},
        {"adapter 320->64", 320u, 64u, 1u, 4000},
        {"adapter 64->320", 64u, 320u, 1u, 2000},
        {"vocab head 320->760", 320u, 760u, 1u, 400},
    };
    double raw_total = 0.0;
    double packed_total = 0.0;
    double w8a32_total = 0.0;
    uint32_t total_calls = 0u;
    printf("CPU features: avx2=%s avx-vnni=%s avx512-vnni=%s neon=%s arm-sdot=%s\n",
        vx_cpu_has_avx2() ? "yes" : "no",
        vx_cpu_has_avx_vnni() ? "yes" : "no",
        vx_cpu_has_avx512_vnni() ? "yes" : "no",
        vx_cpu_has_neon() ? "yes" : "no",
        vx_cpu_has_arm_dotprod() ? "yes" : "no");
    puts("TinyReceipt incremental decoder dense-kernel proxy (shared I8 weights/scales; no end-to-end W8A32 package):");
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        DecoderLinearTiming timing;
        if (!run_decoder_linear_case(&cases[index], &timing)) return 0;
        raw_total += timing.w8a8_raw_ms * cases[index].calls_per_token;
        packed_total += timing.w8a8_packed_ms * cases[index].calls_per_token;
        w8a32_total += timing.w8a32_packed_ms * cases[index].calls_per_token;
        total_calls += cases[index].calls_per_token;
    }
    printf("TinyReceipt weighted dense total: calls=%u W8A8-current-M1-policy(raw)=%.3f ms "
           "W8A8-all-packed=%.3f ms W8A32-all-packed-NR8=%.3f ms "
           "W8A32/current=%.2fx W8A32/all-packed=%.2fx\n",
           total_calls, raw_total, packed_total, w8a32_total,
           w8a32_total / raw_total, w8a32_total / packed_total);
    puts("Scope: kernel-only steady-row proxy; excludes non-linear ops, cached cross K/V, encoder, JS/WASM boundaries, and model accuracy.");
    return 1;
}

typedef struct {
    const char* label;
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t calls;
    int iterations;
} SeedLinearCase;

static int run_seed_linear_case(const SeedLinearCase* test,
                                double* packed_ms, double* single_ms,
                                double* threaded_ms) {
    const float input_scale = 1.0f / 32.0f;
    const float output_scale = 1.0f / 16.0f;
    const int32_t input_zero_point = -7;
    const size_t input_count = (size_t)test->rows * test->d_in;
    const size_t weight_count = (size_t)test->d_out * test->d_in;
    const size_t output_count = (size_t)test->rows * test->d_out;
    const uint32_t packed_bytes = vx_packed_q8_weight_size(
        test->d_in, test->d_out);
    int8_t* input = (int8_t*)malloc(input_count);
    int8_t* weight = (int8_t*)malloc(weight_count);
    int32_t* bias = (int32_t*)malloc((size_t)test->d_out * sizeof(*bias));
    float* scales = (float*)malloc((size_t)test->d_out * sizeof(*scales));
    int32_t* zero_points = (int32_t*)malloc(
        (size_t)test->d_out * sizeof(*zero_points));
    int8_t* reference = (int8_t*)malloc(output_count);
    int8_t* packed_output = (int8_t*)malloc(output_count);
    int8_t* single = (int8_t*)malloc(output_count);
    int8_t* threaded = (int8_t*)malloc(output_count);
    void* packed_weight = malloc(packed_bytes);
    int ok = 0;
    if (!input || !weight || !bias || !scales || !zero_points ||
        !reference || !packed_output || !single || !threaded ||
        !packed_weight || !packed_bytes) goto cleanup;
    for (size_t index = 0; index < input_count; index++)
        input[index] = (int8_t)((int)(index * 29u % 221u) - 110);
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (int8_t)((int)(index * 37u % 211u) - 105);
    for (uint32_t column = 0; column < test->d_out; column++) {
        bias[column] = (int32_t)(column * 41u) - 503;
        scales[column] = 1.0f / (float)(48u + column % 7u * 8u);
        zero_points[column] = (int32_t)(column % 13u) - 6;
    }
    if (!vx_pack_q8_weight(packed_weight, packed_bytes, weight, test->d_in,
            test->d_out, 2u, 1u) ||
        !qlinear_i8u8(input, weight, bias, scales, zero_points, reference,
            test->rows, test->d_in, test->d_out, input_scale, input_zero_point,
            output_scale, 0, 2u, 2u, 2u) ||
        !vx_qlinear_i8u8_packed(input, packed_weight, bias, scales,
            zero_points, packed_output, test->rows, test->d_in, test->d_out,
            input_scale, input_zero_point, output_scale, 0, 2u, 2u, 2u) ||
        !vx_qlinear_i8u8_native(input, weight, bias, scales, zero_points,
            single, test->rows, test->d_in, test->d_out, input_scale,
            input_zero_point, output_scale, 0, 2u, 2u, 2u) ||
        memcmp(reference, packed_output, output_count) != 0 ||
        memcmp(reference, single, output_count) != 0) goto cleanup;
    *packed_ms = time_w8a8_rows(vx_qlinear_i8u8_packed, input, packed_weight,
        bias, scales, zero_points, packed_output, test->rows, test->d_in,
        test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations, 1);
    *single_ms = time_w8a8_rows(vx_qlinear_i8u8_native, input, weight, bias,
        scales, zero_points, single, test->rows, test->d_in, test->d_out,
        input_scale, input_zero_point, output_scale, test->iterations, 1);
    *threaded_ms = time_w8a8_rows(vx_qlinear_i8u8_native, input, weight, bias,
        scales, zero_points, threaded, test->rows, test->d_in, test->d_out,
        input_scale, input_zero_point, output_scale, test->iterations, 4);
    if (*packed_ms < 0.0 || *single_ms < 0.0 || *threaded_ms < 0.0 ||
        memcmp(reference, packed_output, output_count) != 0 ||
        memcmp(reference, single, output_count) != 0 ||
        memcmp(reference, threaded, output_count) != 0) goto cleanup;
    printf("  %-21s calls=%2u M=%3u K=%4u N=%4u packed[%s]=%.3f ms "
           "raw[%s,1t]=%.3f ms raw[%s,4t]=%.3f ms "
           "packed/raw4=%.2fx checksum=0x%08x\n",
           test->label, test->calls, test->rows, test->d_in, test->d_out,
           packed_w8a8_path(), *packed_ms,
           raw_w8a8_path(test->d_in), *single_ms,
           raw_w8a8_path(test->d_in), *threaded_ms,
           *packed_ms / *threaded_ms, checksum_i8(threaded, output_count));
    ok = 1;
cleanup:
    free(input);
    free(weight);
    free(bias);
    free(scales);
    free(zero_points);
    free(reference);
    free(packed_output);
    free(single);
    free(threaded);
    free(packed_weight);
    return ok;
}

static int run_tiny_vqa_decoder_seed_linears(void) {
    /* Exact row counts for the dominant encoder/cross-attention and decoder
     * base projections. LoRA, adapters, and the vocabulary head are outside
     * this deliberately compact kernel benchmark. */
    static const SeedLinearCase cases[] = {
        {"enc/cross attention", 402u, 320u, 320u, 32u, 16},
        {"encoder FFN up", 402u, 320u, 1280u, 6u, 6},
        {"encoder FFN down", 402u, 1280u, 320u, 6u, 6},
        {"decoder attention", 192u, 320u, 320u, 24u, 30},
        {"decoder FFN up", 192u, 320u, 1280u, 4u, 12},
        {"decoder FFN down", 192u, 1280u, 320u, 4u, 12},
    };
    double packed_total = 0.0;
    double single_total = 0.0;
    double threaded_total = 0.0;
    puts("TinyReceipt full-seed base-dense subset (exact M=402/M=192):");
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        double packed_ms, single_ms, threaded_ms;
        if (!run_seed_linear_case(&cases[index], &packed_ms, &single_ms,
                                  &threaded_ms))
            return 0;
        packed_total += packed_ms * cases[index].calls;
        single_total += single_ms * cases[index].calls;
        threaded_total += threaded_ms * cases[index].calls;
    }
    vx_set_num_threads(0);
    printf("TinyReceipt full-seed weighted base-dense subset: packed=%.3f ms "
           "raw[1t]=%.3f ms raw[4t]=%.3f ms packed/raw4=%.2fx "
           "raw1/raw4=%.2fx\n", packed_total, single_total, threaded_total,
           packed_total / threaded_total, single_total / threaded_total);
    return 1;
}

typedef int (*QActivationFn)(const void*, void*, uint32_t, float, int32_t,
                            float, int32_t, uint32_t, uint32_t);

static int32_t benchmark_round_ties_even(float value) {
    const int32_t lower = (int32_t)floorf(value);
    const float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static int8_t benchmark_quantize_i8(float transformed, int32_t zero_point) {
    int32_t quantized;
    if (transformed != transformed) quantized = zero_point;
    else if (transformed <= -128.0f) quantized = -128;
    else if (transformed >= 127.0f) quantized = 127;
    else quantized = benchmark_round_ties_even(transformed);
    return (int8_t)quantized;
}

static float benchmark_qgelu_erf(float value) {
    const float sign = value >= 0.0f ? 1.0f : -1.0f;
    const float magnitude = fabsf(value);
    const float t = 1.0f / (1.0f + 0.3275911f * magnitude);
    float polynomial = 1.061405429f * t;
    polynomial = polynomial - 1.453152027f;
    polynomial = polynomial * t;
    polynomial = polynomial + 1.421413741f;
    polynomial = polynomial * t;
    polynomial = polynomial - 0.284496736f;
    polynomial = polynomial * t;
    polynomial = polynomial + 0.254829592f;
    polynomial = polynomial * t;
    return sign * (1.0f - polynomial * expf(-(magnitude * magnitude)));
}

static int benchmark_qsilu_scalar(const void* input, void* output,
        uint32_t elements, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    const int8_t* input_i8 = (const int8_t*)input;
    int8_t* output_i8 = (int8_t*)output;
    if (input_dtype != 2u || output_dtype != 2u) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        const float value = (float)((int32_t)input_i8[index] - input_zero_point) *
            input_scale;
        const float activated = value / (1.0f + expf(-value));
        output_i8[index] = benchmark_quantize_i8(
            activated / output_scale + (float)output_zero_point,
            output_zero_point);
    }
    return 1;
}

static int benchmark_qgelu_scalar(const void* input, void* output,
        uint32_t elements, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    const int8_t* input_i8 = (const int8_t*)input;
    int8_t* output_i8 = (int8_t*)output;
    if (input_dtype != 2u || output_dtype != 2u) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        const float value = (float)((int32_t)input_i8[index] - input_zero_point) *
            input_scale;
        const float erf_input = value * 0.7071067811865476f;
        const float cdf = 0.5f * (1.0f + benchmark_qgelu_erf(erf_input));
        output_i8[index] = benchmark_quantize_i8(
            (value * cdf) / output_scale + (float)output_zero_point,
            output_zero_point);
    }
    return 1;
}

typedef struct {
    const char* label;
    QActivationFn scalar;
    QActivationFn cached;
    uint32_t elements;
    float input_scale;
    float output_scale;
    int iterations;
} ActivationCase;

static int run_activation_case(const ActivationCase* test) {
    enum { warmup = 2 };
    const int32_t input_zero_point = -3;
    const int32_t output_zero_point = -5;
    int8_t* input = (int8_t*)malloc(test->elements);
    int8_t* scalar_output = (int8_t*)malloc(test->elements);
    int8_t* cached_output = (int8_t*)malloc(test->elements);
    struct timespec start, end;
    double scalar_ms, cached_ms;
    int ok = 0;
    if (!input || !scalar_output || !cached_output) goto cleanup;
    for (uint32_t index = 0; index < test->elements; index++)
        input[index] = (int8_t)((int)(index * 37u % 255u) - 127);
    if (!test->scalar(input, scalar_output, test->elements,
            test->input_scale, input_zero_point, test->output_scale,
            output_zero_point, 2u, 2u) ||
        !test->cached(input, cached_output, test->elements,
            test->input_scale, input_zero_point, test->output_scale,
            output_zero_point, 2u, 2u) ||
        memcmp(scalar_output, cached_output, test->elements) != 0) goto cleanup;
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (!test->scalar(input, scalar_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, 2u, 2u)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < test->iterations; iteration++) {
        if (!test->scalar(input, scalar_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, 2u, 2u)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    scalar_ms = elapsed_ms(&start, &end) / test->iterations;
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (!test->cached(input, cached_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, 2u, 2u)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < test->iterations; iteration++) {
        if (!test->cached(input, cached_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, 2u, 2u)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    cached_ms = elapsed_ms(&start, &end) / test->iterations;
    printf("  %-28s elements=%7u scalar=%.4f ms kernel=%.4f ms "
           "speedup=%.2fx checksum=0x%08x\n",
           test->label, test->elements, scalar_ms, cached_ms,
           scalar_ms / cached_ms, checksum_i8(cached_output, test->elements));
    ok = 1;
cleanup:
    free(input);
    free(scalar_output);
    free(cached_output);
    return ok;
}

static int run_tiny_vqa_quantized_activations(void) {
    /* Exact materialized TinyReceipt shapes.  Keep the decoder seed immediately
     * before its M=1 row so the latter also measures descriptor-cache reuse. */
    static const ActivationCase cases[] = {
        {"QGELU router [1,160]", benchmark_qgelu_scalar, qgelu_i8u8,
            160u, 0.03125f, 0.0268933307f, 10000},
        {"QSiLU stem [1,160,336,48]", benchmark_qsilu_scalar, qsilu_i8u8,
            160u * 336u * 48u, 0.0696206465f, 0.0695981681f, 4},
        {"QGELU encoder [1,402,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            402u * 1280u, 0.0153605873f, 0.0147757018f, 8},
        {"QGELU decoder seed [1,192,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            192u * 1280u, 0.0234375f, 0.01953125f, 12},
        {"QGELU decoder row [1,1,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            1280u, 0.0234375f, 0.01953125f, 2000},
    };
    puts("TinyReceipt physical quantized activation proxy (exact scalar vs LUT-aware kernel):");
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        if (!run_activation_case(&cases[index])) return 0;
    }
    return 1;
}

static int run_tiny_vqa_grayscale_stem_qconv(void) {
    enum {
        batch = 1, input_height = 320, input_width = 672, channels = 1,
        output_height = 160, output_width = 336, output_channels = 48,
        kernel = 3,
        elements = batch * input_height * input_width * channels,
        weight_elements = output_channels * kernel * kernel * channels,
        output_elements = batch * output_height * output_width * output_channels,
        portable_iterations = 2, native_iterations = 6,
    };
    int8_t *input = (int8_t *)malloc(elements);
    int8_t *weight = (int8_t *)malloc(weight_elements);
    int8_t *portable = (int8_t *)malloc(output_elements);
    int8_t *native_single = (int8_t *)malloc(output_elements);
    int8_t *native = (int8_t *)malloc(output_elements);
    int32_t bias[output_channels];
    float scales[output_channels];
    int32_t zero_points[output_channels];
    struct timespec start, end;
    double portable_ms, native_single_ms, native_ms;
    int ok = input && weight && portable && native_single && native;
    if (!ok) goto cleanup;
    for (int index = 0; index < elements; index++)
        input[index] = (int8_t)((index * 23) % 255 - 127);
    for (int index = 0; index < weight_elements; index++)
        weight[index] = (int8_t)((index * 31) % 251 - 125);
    for (int index = 0; index < output_channels; index++) {
        bias[index] = index * 17 - 307;
        scales[index] = 1.0f / (float)(64 + index % 5 * 8);
        zero_points[index] = 0;
    }
    ok = qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
            batch, input_height, input_width, channels, output_height,
            output_width, output_channels, kernel, kernel, channels,
            2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0, 2u, 2u, 2u) == 1;
    vx_set_num_threads(1);
    ok = ok && vx_qconv2d_i8u8_native(input, weight, bias, scales,
            zero_points, native_single, batch, input_height, input_width,
            channels, output_height, output_width, output_channels, kernel,
            kernel, channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0, 2u, 2u, 2u) == 1;
    vx_set_num_threads(4);
    ok = ok && vx_qconv2d_i8u8_native(input, weight, bias, scales,
            zero_points, native, batch, input_height, input_width, channels,
            output_height, output_width, output_channels, kernel, kernel,
            channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0, 2u, 2u, 2u) == 1;
    ok = ok && memcmp(portable, native_single, output_elements) == 0 &&
        memcmp(portable, native, output_elements) == 0;
    if (!ok) goto cleanup;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < portable_iterations; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, input_height, input_width, channels, output_height,
                output_width, output_channels, kernel, kernel, channels,
                2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                2u, 2u, 2u) != 1) {
            ok = 0;
            goto cleanup;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    portable_ms = elapsed_ms(&start, &end) / portable_iterations;
    vx_set_num_threads(1);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < native_iterations; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points,
                native_single, batch, input_height, input_width, channels,
                output_height, output_width, output_channels, kernel, kernel,
                channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                2u, 2u, 2u) != 1) {
            ok = 0;
            goto cleanup;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    native_single_ms = elapsed_ms(&start, &end) / native_iterations;
    vx_set_num_threads(4);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < native_iterations; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points,
                native, batch, input_height, input_width, channels,
                output_height, output_width, output_channels, kernel, kernel,
                channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                2u, 2u, 2u) != 1) {
            ok = 0;
            goto cleanup;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    native_ms = elapsed_ms(&start, &end) / native_iterations;
    printf("TinyReceipt grayscale stem QConv2D [1,320,672,1] -> [1,160,336,48]: "
           "portable=%.3f ms single[%s]=%.3f ms tiled[%s,4t]=%.3f ms "
           "thread-speedup=%.2fx portable-speedup=%.2fx checksum=0x%08x\n",
           portable_ms, qconv_w8a8_path(channels), native_single_ms,
           qconv_w8a8_path(channels), native_ms, native_single_ms / native_ms,
           portable_ms / native_ms, checksum_i8(native, output_elements));
cleanup:
    vx_set_num_threads(0);
    free(input);
    free(weight);
    free(portable);
    free(native_single);
    free(native);
    return ok;
}

static int run_qconv(void) {
    enum {
        batch = 1, height = 16, width = 16, channels = 64, output_channels = 64,
        kernel = 3, elements = batch * height * width * channels,
        weight_elements = output_channels * kernel * kernel * channels,
        output_elements = batch * height * width * output_channels,
        warmup = 2, iterations = 10,
    };
    static int8_t input[elements];
    static int8_t weight[weight_elements];
    static int32_t bias[output_channels];
    static float scales[output_channels];
    static int32_t zero_points[output_channels];
    static int8_t portable[output_elements];
    static int8_t native_single[output_elements];
    static int8_t native[output_elements];
    struct timespec start, end;
    double portable_ms, native_single_ms, native_ms;
    for (int index = 0; index < elements; index++) input[index] = (int8_t)((index * 23) % 127 - 63);
    for (int index = 0; index < weight_elements; index++) weight[index] = (int8_t)((index * 31) % 111 - 55);
    for (int index = 0; index < output_channels; index++) {
        bias[index] = index * 7 - 107;
        scales[index] = 1.0f / (float)(64 + index % 5 * 8);
        zero_points[index] = index % 9 - 4;
    }
    if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
            batch, height, width, channels, height, width, output_channels,
            kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1) return 0;
    vx_set_num_threads(1);
    if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points,
            native_single, batch, height, width, channels, height, width,
            output_channels, kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
            2u, 2u, 2u) != 1) return 0;
    vx_set_num_threads(4);
    if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points, native,
            batch, height, width, channels, height, width, output_channels,
            kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1 ||
        memcmp(portable, native_single, sizeof(portable)) != 0 ||
        memcmp(portable, native, sizeof(portable)) != 0) return 0;
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    portable_ms = elapsed_ms(&start, &end) / iterations;
    vx_set_num_threads(1);
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points,
                native_single, batch, height, width, channels, height, width,
                output_channels, kernel, kernel, channels, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
                2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points,
                native_single, batch, height, width, channels, height, width,
                output_channels, kernel, kernel, channels, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
                2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    native_single_ms = elapsed_ms(&start, &end) / iterations;
    vx_set_num_threads(4);
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points, native,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (vx_qconv2d_i8u8_native(input, weight, bias, scales, zero_points, native,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2, 2u, 2u, 2u) != 1) return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    native_ms = elapsed_ms(&start, &end) / iterations;
    vx_set_num_threads(0);
    printf("QConv2D W8A8: portable=%.3f ms single[%s]=%.3f ms "
           "tiled[%s,4t]=%.3f ms thread-speedup=%.2fx portable-speedup=%.2fx "
           "checksum=0x%08x\n",
           portable_ms, qconv_w8a8_path(channels), native_single_ms,
           qconv_w8a8_path(channels), native_ms, native_single_ms / native_ms,
           portable_ms / native_ms, checksum_i8(native, sizeof(native)));
    return 1;
}

int main(void) {
    if (!run_tiny_vqa_decoder_linear_mix() ||
        !run_tiny_vqa_decoder_seed_linears() ||
        !run_tiny_vqa_quantized_activations() ||
        !run_tiny_vqa_grayscale_stem_qconv() ||
        !run_qconv()) {
        fputs("native W8A8/W8A32 benchmark correctness preflight failed\n", stderr);
        return 1;
    }
    return 0;
}
