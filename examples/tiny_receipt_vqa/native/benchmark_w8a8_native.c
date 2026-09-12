/*
 * Direct native W8A8 and W8A32 kernel benchmark.
 *
 * This is deliberately separate from correctness tests: timing is inherently
 * host-dependent. Before timing each workload it checks both accelerated
 * layouts against their portable references. The decoder proxy uses the
 * actual TinyReceipt incremental-row dense shapes and call counts.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "quant_cpu_isa.h"
#include "packed_quant_gemm.h"
#include "cpu_features.h"
#include "thread_pool.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#endif

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

typedef struct {
    double average_ms;
    double span_ms;
    int iterations;
} HotTiming;

static double g_minimum_hot_span_ms = 0.0;
static int g_gate_mode = 0;
static int g_single_cpu = -1;
static int g_grayscale_single_iterations = 0;
static int g_affinity_ready = 0;
static int g_affinity_ok = 1;
static int g_affinity_error = 0;
static int g_affinity_failure_threads = -1;
static int g_pool_cpu_count = 0;
static int g_thread_count_ok = 1;
static int g_thread_count_expected = -1;
static int g_thread_count_actual = -1;
#if defined(__linux__)
static cpu_set_t g_pool_cpu_affinity;
#endif

static int benchmark_set_current_affinity(
#if defined(__linux__)
        const cpu_set_t* affinity
#else
        const void* affinity
#endif
) {
#if defined(__linux__) && defined(__x86_64__)
    long result;
    __asm__ volatile(
        "syscall"
        : "=a"(result)
        : "0"((long)SYS_sched_setaffinity), "D"(0L),
          "S"((long)sizeof(*affinity)), "d"((long)affinity)
        : "rcx", "r11", "memory");
    if (result == 0) return 1;
    g_affinity_error = result < 0 ? (int)-result : (int)result;
    return 0;
#elif defined(__linux__) && defined(__aarch64__)
    register long x0 __asm__("x0") = 0;
    register long x1 __asm__("x1") = (long)sizeof(*affinity);
    register long x2 __asm__("x2") = (long)affinity;
    register long x8 __asm__("x8") = SYS_sched_setaffinity;
    __asm__ volatile(
        "svc 0"
        : "+r"(x0)
        : "r"(x1), "r"(x2), "r"(x8)
        : "memory");
    if (x0 == 0) return 1;
    g_affinity_error = x0 < 0 ? (int)-x0 : (int)x0;
    return 0;
#else
    (void)affinity;
    return 0;
#endif
}

static int benchmark_affinity_initialize(void) {
    if (!g_gate_mode) return 1;
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
    if (sched_getaffinity(0, sizeof(g_pool_cpu_affinity),
                          &g_pool_cpu_affinity) != 0) return 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &g_pool_cpu_affinity)) continue;
        g_pool_cpu_count++;
        if (g_single_cpu < 0) g_single_cpu = cpu;
    }
    if (g_pool_cpu_count < 4 || g_single_cpu < 0 ||
        !CPU_ISSET(g_single_cpu, &g_pool_cpu_affinity)) return 0;
    g_affinity_ready = 1;
    return 1;
#else
    return 0;
#endif
}

static void benchmark_affinity_worker_warmup(
        void* context, int begin, int end) {
    (void)context;
    (void)begin;
    (void)end;
}

static void benchmark_verify_thread_count(int count) {
    int actual;
    if (!g_gate_mode || count <= 0) return;
    actual = vx_kernels_thread_count();
    if (actual == count) return;
    g_thread_count_ok = 0;
    g_thread_count_expected = count;
    g_thread_count_actual = actual;
}

static void benchmark_set_num_threads(int count) {
#if defined(__linux__)
    if (g_gate_mode && g_affinity_ready) {
        cpu_set_t affinity = g_pool_cpu_affinity;
        if (count == 1) {
            CPU_ZERO(&affinity);
            CPU_SET(g_single_cpu, &affinity);
        }
        if (!benchmark_set_current_affinity(&affinity)) {
            g_affinity_ok = 0;
            g_affinity_failure_threads = count;
        }
        vx_set_num_threads(count);
        benchmark_verify_thread_count(count);
        if (count == 4 && g_affinity_ok) {
            cpu_set_t caller_affinity;
            vx_kernels_parallel_for(
                4, 1, benchmark_affinity_worker_warmup, NULL);
            CPU_ZERO(&caller_affinity);
            CPU_SET(g_single_cpu, &caller_affinity);
            if (!benchmark_set_current_affinity(&caller_affinity)) {
                g_affinity_ok = 0;
                g_affinity_failure_threads = count;
            }
        }
        return;
    }
#endif
    vx_set_num_threads(count);
    benchmark_verify_thread_count(count);
}

static int next_hot_iterations(int current, double span_ms) {
    double scaled;
    int rounded;
    if (span_ms >= g_minimum_hot_span_ms || g_minimum_hot_span_ms <= 0.0)
        return current;
    if (current <= 0 || span_ms < 0.0) return -1;
    if (span_ms == 0.0) {
        if (current > INT_MAX / 10) return -1;
        return current * 10;
    }
    scaled = (double)current * g_minimum_hot_span_ms / span_ms * 1.10;
    if (scaled <= (double)current) scaled = (double)current + 1.0;
    if (scaled > 100000000.0 || scaled > (double)INT_MAX) return -1;
    /* Keep the harness from adding a ceil() PLT slot before production text;
     * the release-layout gate requires hot symbol addresses to stay exact. */
    rounded = (int)scaled;
    if ((double)rounded < scaled) rounded++;
    return rounded;
}

static int finish_hot_timing(HotTiming* timing, int iterations,
                             double span_ms) {
    if (!timing || iterations <= 0 || span_ms < 0.0) return 0;
    if (g_minimum_hot_span_ms > 0.0 && span_ms < g_minimum_hot_span_ms)
        return 0;
    timing->average_ms = span_ms / (double)iterations;
    timing->span_ms = span_ms;
    timing->iterations = iterations;
    return 1;
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
    HotTiming w8a8_raw;
    HotTiming w8a8_packed;
    double w8a32_packed_ms;
} DecoderLinearTiming;

static const char* raw_w8a8_path(uint32_t d_in) {
#if defined(__i386__) || defined(__x86_64__)
    if (d_in >= 64u && vx_cpu_has_avx512_vnni()) return "avx512-vnni-zmm";
    if (d_in >= 32u && vx_cpu_has_avx_vnni()) return "avx-vnni";
    if (d_in >= 32u && vx_cpu_has_avx2()) return "avx2-maddubs-exact";
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

static const char* qconv_w8a8_path(uint32_t input_per_group,
                                    int dense_3x3_qlinear) {
#if defined(__i386__) || defined(__x86_64__)
    if (input_per_group >= 64u && vx_cpu_has_avx512_vnni())
        return "avx512-vnni-zmm";
    if (input_per_group >= 32u && vx_cpu_has_avx_vnni()) return "avx-vnni";
    if (vx_cpu_has_avx2()) {
        if (dense_3x3_qlinear && input_per_group >= 64u)
            return "im2col+avx2-maddubs-exact";
        return input_per_group < 16u ? "avx2-oc8-small-c" : "avx2-oc4";
    }
#elif defined(__aarch64__) || defined(__arm__)
#if defined(VOLVOXAI_ARM_DOTPROD_OBJECT)
    if (input_per_group >= 16u && vx_cpu_has_arm_dotprod()) return "arm-sdot";
#endif
    if (input_per_group >= 8u && vx_cpu_has_neon()) return "neon";
#else
    (void)input_per_group;
#endif
    (void)dense_3x3_qlinear;
    return "portable";
}

static const char* packed_w8a8_path(void) {
#if defined(__i386__) || defined(__x86_64__)
    if (vx_cpu_has_avx2()) return "avx2-k4-n16-exact";
#endif
    return "portable-packed";
}

static int time_w8a8(QLinearW8A8Fn function, const void* input,
        const void* weight, const int32_t* bias, const float* scales,
        const int32_t* zero_points, void* output, uint32_t d_in,
        uint32_t d_out, float input_scale, int32_t input_zero_point,
        float output_scale, int initial_iterations, HotTiming* timing) {
    enum { warmup = 3 };
    struct timespec start, end;
    double span_ms;
    int iterations = initial_iterations;
    int attempt;
    benchmark_set_num_threads(1);
    if (!g_affinity_ok) return 0;
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                1u, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
                VX_DTYPE_I8)) return 0;
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < iterations; iteration++)
            if (!function(input, weight, bias, scales, zero_points, output,
                    1u, d_in, d_out, input_scale, input_zero_point,
                    output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8)) return 0;
        clock_gettime(CLOCK_MONOTONIC, &end);
        span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            span_ms >= g_minimum_hot_span_ms) {
            return finish_hot_timing(timing, iterations, span_ms);
        }
        iterations = next_hot_iterations(iterations, span_ms);
        if (iterations <= 0) return 0;
    }
    return 0;
}

static int time_w8a8_rows(QLinearW8A8Fn function, const void* input,
        const void* weight, const int32_t* bias, const float* scales,
        const int32_t* zero_points, void* output, uint32_t rows,
        uint32_t d_in, uint32_t d_out, float input_scale,
        int32_t input_zero_point, float output_scale, int initial_iterations,
        int threads, HotTiming* timing) {
    enum { warmup = 2 };
    struct timespec start, end;
    double span_ms;
    int iterations = initial_iterations;
    int attempt;
    benchmark_set_num_threads(threads);
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, bias, scales, zero_points, output,
                rows, d_in, d_out, input_scale, input_zero_point,
                output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
                VX_DTYPE_I8)) return 0;
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < iterations; iteration++)
            if (!function(input, weight, bias, scales, zero_points, output,
                    rows, d_in, d_out, input_scale, input_zero_point,
                    output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8)) return 0;
        clock_gettime(CLOCK_MONOTONIC, &end);
        span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            span_ms >= g_minimum_hot_span_ms) {
            return finish_hot_timing(timing, iterations, span_ms);
        }
        iterations = next_hot_iterations(iterations, span_ms);
        if (iterations <= 0) return 0;
    }
    return 0;
}

static double time_w8a32(LinearW8A32Fn function, const float* input,
        const void* weight, const float* scales, const int32_t* zero_points,
        const float* bias, float* output, uint32_t d_in, uint32_t d_out,
        int iterations) {
    const int warmup = g_gate_mode ? 0 : 3;
    struct timespec start, end;
    for (int iteration = 0; iteration < warmup; iteration++)
        if (!function(input, weight, scales, zero_points, bias, output,
                1u, d_in, d_out, VX_DTYPE_I8, d_out, VX_DTYPE_I32,
                d_out)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!function(input, weight, scales, zero_points, bias, output,
                1u, d_in, d_out, VX_DTYPE_I8, d_out, VX_DTYPE_I32,
                d_out)) return -1.0;
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
            test->d_out, VX_DTYPE_I8, 1u) ||
        !matmul_quantized_f32(input_f32, weight, scales, zero_points, bias_f32,
            w8a32_reference, 1u, test->d_in, test->d_out, VX_DTYPE_I8,
            test->d_out, VX_DTYPE_I32, test->d_out) ||
        !vx_matmul_quantized_f32_packed(input_f32, packed_weight, scales,
            zero_points, bias_f32, w8a32_packed, 1u, test->d_in,
            test->d_out, VX_DTYPE_I8, test->d_out, VX_DTYPE_I32,
            test->d_out)) goto cleanup;
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
            input_zero_point, output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
            VX_DTYPE_I8) ||
        !vx_qlinear_i8u8_native(input_i8, weight, bias_i32, scales, zero_points,
            w8a8_raw, 1u, test->d_in, test->d_out, input_scale,
            input_zero_point, output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
            VX_DTYPE_I8) ||
        !vx_qlinear_i8u8_packed(input_i8, packed_weight, bias_i32, scales,
            zero_points, w8a8_packed, 1u, test->d_in, test->d_out,
            input_scale, input_zero_point, output_scale, 0, VX_DTYPE_I8,
            VX_DTYPE_I8, VX_DTYPE_I8) ||
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
    if (!time_w8a8(vx_qlinear_i8u8_native, input_i8,
        weight, bias_i32, scales, zero_points, w8a8_raw, test->d_in,
        test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations, &timing->w8a8_raw) ||
        !time_w8a8(vx_qlinear_i8u8_packed, input_i8,
        packed_weight, bias_i32, scales, zero_points, w8a8_packed,
        test->d_in, test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations, &timing->w8a8_packed)) goto cleanup;
    timing->w8a32_packed_ms = time_w8a32(vx_matmul_quantized_f32_packed,
        input_f32, packed_weight, scales, zero_points, bias_f32, w8a32_packed,
        test->d_in, test->d_out, g_gate_mode ? 1 : test->iterations);
    if (timing->w8a32_packed_ms < 0.0) goto cleanup;
    printf("  %-24s calls=%2u M=1 K=%4u N=%4u raw[%s]=%.6f ms "
           "packed[%s]=%.6f ms W8A32[packed-NR8]=%.6f ms "
           "W8A32/raw=%.2fx W8A32/packed=%.2fx check=pass "
           "(f32=%.3g cross=%.3g i8=0x%08x) "
           "hot-span[decoder-raw]=%.3f/%d "
           "hot-span[decoder-packed]=%.3f/%d\n",
           test->label, test->calls_per_token, test->d_in, test->d_out,
           raw_w8a8_path(test->d_in), timing->w8a8_raw.average_ms,
           packed_w8a8_path(), timing->w8a8_packed.average_ms,
           timing->w8a32_packed_ms,
           timing->w8a32_packed_ms / timing->w8a8_raw.average_ms,
           timing->w8a32_packed_ms / timing->w8a8_packed.average_ms,
           (double)max_w8a32_delta, (double)max_cross_mode_delta,
           checksum_i8(w8a8_packed, output_count),
           timing->w8a8_raw.span_ms, timing->w8a8_raw.iterations,
           timing->w8a8_packed.span_ms, timing->w8a8_packed.iterations);
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
        raw_total += timing.w8a8_raw.average_ms * cases[index].calls_per_token;
        packed_total += timing.w8a8_packed.average_ms * cases[index].calls_per_token;
        w8a32_total += timing.w8a32_packed_ms * cases[index].calls_per_token;
        total_calls += cases[index].calls_per_token;
    }
    printf("TinyReceipt weighted dense total: calls=%u W8A8-current-M1-policy(raw)=%.6f ms "
           "W8A8-all-packed=%.6f ms W8A32-all-packed-NR8=%.6f ms "
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
} PrefillLinearCase;

static int run_prefill_linear_case(const PrefillLinearCase* test,
                                HotTiming* packed_timing,
                                HotTiming* single_timing,
                                HotTiming* threaded_timing) {
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
        zero_points[column] = 0;
    }
    if (!vx_pack_q8_weight(packed_weight, packed_bytes, weight, test->d_in,
            test->d_out, VX_DTYPE_I8, 1u) ||
        !qlinear_i8u8(input, weight, bias, scales, zero_points, reference,
            test->rows, test->d_in, test->d_out, input_scale, input_zero_point,
            output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) ||
        !vx_qlinear_i8u8_packed(input, packed_weight, bias, scales,
            zero_points, packed_output, test->rows, test->d_in, test->d_out,
            input_scale, input_zero_point, output_scale, 0, VX_DTYPE_I8,
            VX_DTYPE_I8, VX_DTYPE_I8) ||
        !vx_qlinear_i8u8_native(input, weight, bias, scales, zero_points,
            single, test->rows, test->d_in, test->d_out, input_scale,
            input_zero_point, output_scale, 0, VX_DTYPE_I8, VX_DTYPE_I8,
            VX_DTYPE_I8) ||
        memcmp(reference, packed_output, output_count) != 0 ||
        memcmp(reference, single, output_count) != 0) goto cleanup;
    if (!time_w8a8_rows(vx_qlinear_i8u8_packed, input, packed_weight,
        bias, scales, zero_points, packed_output, test->rows, test->d_in,
        test->d_out, input_scale, input_zero_point, output_scale,
        test->iterations, 4, packed_timing) ||
        !time_w8a8_rows(vx_qlinear_i8u8_native, input, weight, bias,
        scales, zero_points, single, test->rows, test->d_in, test->d_out,
        input_scale, input_zero_point, output_scale, test->iterations, 1,
        single_timing) ||
        !time_w8a8_rows(vx_qlinear_i8u8_native, input, weight, bias,
        scales, zero_points, threaded, test->rows, test->d_in, test->d_out,
        input_scale, input_zero_point, output_scale, test->iterations, 4,
        threaded_timing)) goto cleanup;
    if (memcmp(reference, packed_output, output_count) != 0 ||
        memcmp(reference, single, output_count) != 0 ||
        memcmp(reference, threaded, output_count) != 0) goto cleanup;
    printf("  %-21s calls=%2u M=%3u K=%4u N=%4u packed[%s,4t]=%.6f ms "
           "raw[%s,1t]=%.6f ms raw[%s,4t]=%.6f ms "
           "packed/raw4=%.2fx checksum=0x%08x "
           "hot-span[prefill-packed]=%.3f/%d "
           "hot-span[prefill-raw1]=%.3f/%d "
           "hot-span[prefill-raw4]=%.3f/%d\n",
           test->label, test->calls, test->rows, test->d_in, test->d_out,
           packed_w8a8_path(), packed_timing->average_ms,
           raw_w8a8_path(test->d_in), single_timing->average_ms,
           raw_w8a8_path(test->d_in), threaded_timing->average_ms,
           packed_timing->average_ms / threaded_timing->average_ms,
           checksum_i8(threaded, output_count),
           packed_timing->span_ms, packed_timing->iterations,
           single_timing->span_ms, single_timing->iterations,
           threaded_timing->span_ms, threaded_timing->iterations);
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

static int run_tiny_vqa_decoder_prefill_linears(void) {
    /* Exact row counts for the dominant encoder/cross-attention and decoder
     * base projections. LoRA, adapters, and the vocabulary head are outside
     * this deliberately compact kernel benchmark. */
    static const PrefillLinearCase cases[] = {
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
    puts("TinyReceipt full-prefill symmetric-I8 base-dense subset "
         "(exact M=402/M=192):");
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        HotTiming packed, single, threaded;
        if (!run_prefill_linear_case(&cases[index], &packed, &single,
                                  &threaded))
            return 0;
        packed_total += packed.average_ms * cases[index].calls;
        single_total += single.average_ms * cases[index].calls;
        threaded_total += threaded.average_ms * cases[index].calls;
    }
    benchmark_set_num_threads(0);
    printf("TinyReceipt full-prefill weighted base-dense subset: packed=%.6f ms "
           "raw[1t]=%.6f ms raw[4t]=%.6f ms packed/raw4=%.2fx "
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
    if (input_dtype != VX_DTYPE_I8 || output_dtype != VX_DTYPE_I8) return 0;
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
    if (input_dtype != VX_DTYPE_I8 || output_dtype != VX_DTYPE_I8) return 0;
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
    HotTiming cached_timing;
    double scalar_ms;
    double cached_span_ms = 0.0;
    const int scalar_iterations = g_gate_mode ? 1 : test->iterations;
    int cached_iterations = test->iterations;
    int attempt;
    int ok = 0;
    benchmark_set_num_threads(1);
    if (!g_affinity_ok) goto cleanup;
    if (!input || !scalar_output || !cached_output) goto cleanup;
    for (uint32_t index = 0; index < test->elements; index++)
        input[index] = (int8_t)((int)(index * 37u % 255u) - 127);
    if (!test->scalar(input, scalar_output, test->elements,
            test->input_scale, input_zero_point, test->output_scale,
            output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8) ||
        !test->cached(input, cached_output, test->elements,
            test->input_scale, input_zero_point, test->output_scale,
            output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8) ||
        memcmp(scalar_output, cached_output, test->elements) != 0) goto cleanup;
    for (int iteration = 0; iteration < (g_gate_mode ? 0 : warmup); iteration++) {
        if (!test->scalar(input, scalar_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < scalar_iterations; iteration++) {
        if (!test->scalar(input, scalar_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8)) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    scalar_ms = elapsed_ms(&start, &end) / scalar_iterations;
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (!test->cached(input, cached_output, test->elements,
                test->input_scale, input_zero_point, test->output_scale,
                output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8)) goto cleanup;
    }
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < cached_iterations; iteration++) {
            if (!test->cached(input, cached_output, test->elements,
                    test->input_scale, input_zero_point, test->output_scale,
                    output_zero_point, VX_DTYPE_I8, VX_DTYPE_I8)) goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        cached_span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            cached_span_ms >= g_minimum_hot_span_ms) break;
        cached_iterations = next_hot_iterations(
            cached_iterations, cached_span_ms);
        if (cached_iterations <= 0) goto cleanup;
    }
    if (!finish_hot_timing(&cached_timing, cached_iterations,
                           cached_span_ms)) goto cleanup;
    printf("  %-28s elements=%7u scalar=%.6f ms kernel=%.6f ms "
           "speedup=%.2fx checksum=0x%08x "
           "hot-span[activation]=%.3f/%d\n",
           test->label, test->elements, scalar_ms, cached_timing.average_ms,
           scalar_ms / cached_timing.average_ms,
           checksum_i8(cached_output, test->elements),
           cached_timing.span_ms, cached_timing.iterations);
    ok = 1;
cleanup:
    free(input);
    free(scalar_output);
    free(cached_output);
    return ok;
}

static int run_tiny_vqa_quantized_activations(void) {
    /* Exact materialized TinyReceipt shapes.  Keep the decoder prefill immediately
     * before its M=1 row so the latter also measures descriptor-cache reuse. */
    static const ActivationCase cases[] = {
        {"QGELU router [1,160]", benchmark_qgelu_scalar, qgelu_i8u8,
            160u, 0.03125f, 0.0268933307f, 10000},
        {"QSiLU stem [1,160,336,48]", benchmark_qsilu_scalar, qsilu_i8u8,
            160u * 336u * 48u, 0.0696206465f, 0.0695981681f, 4},
        {"QGELU encoder [1,402,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            402u * 1280u, 0.0153605873f, 0.0147757018f, 8},
        {"QGELU decoder prefill [1,192,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            192u * 1280u, 0.0234375f, 0.01953125f, 12},
        {"QGELU decoder row [1,1,1280]", benchmark_qgelu_scalar, qgelu_i8u8,
            1280u, 0.0234375f, 0.01953125f, 2000},
    };
    puts("TinyReceipt quantized activation proxy (exact scalar vs LUT-aware kernel):");
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
        native_warmup = 2, native_iterations = 6,
    };
    int8_t *input = (int8_t *)malloc(elements);
    int8_t *weight = (int8_t *)malloc(weight_elements);
    int8_t *portable = (int8_t *)malloc(output_elements);
    int8_t *native_single = (int8_t *)malloc(output_elements);
    int8_t *native = (int8_t *)malloc(output_elements);
    const uint32_t packed_bytes = vx_packed_q8_weight_size(
        kernel * kernel * channels, output_channels);
    void* packed_weight = malloc(packed_bytes);
    int32_t bias[output_channels];
    float scales[output_channels];
    int32_t zero_points[output_channels];
    struct timespec start, end;
    HotTiming native_single_timing, native_timing;
    double portable_ms;
    double native_span_ms;
    int measured_iterations;
    const int portable_iterations = g_gate_mode ? 1 : 2;
    int attempt;
    int ok = input && weight && portable && native_single && native &&
        packed_weight && packed_bytes;
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
    ok = vx_pack_q8_weight(packed_weight, packed_bytes, weight,
        kernel * kernel * channels, output_channels, VX_DTYPE_I8, 1u) == 1;
    ok = ok && qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
            batch, input_height, input_width, channels, output_height,
            output_width, output_channels, kernel, kernel, channels,
            2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1;
    benchmark_set_num_threads(1);
    ok = ok && vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
            zero_points, native_single, batch, input_height, input_width,
            channels, output_height, output_width, output_channels, kernel,
            kernel, channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight, NULL) == 1;
    benchmark_set_num_threads(4);
    ok = ok && vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
            zero_points, native, batch, input_height, input_width, channels,
            output_height, output_width, output_channels, kernel, kernel,
            channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight, NULL) == 1;
    ok = ok && memcmp(portable, native_single, output_elements) == 0 &&
        memcmp(portable, native, output_elements) == 0;
    if (!ok) goto cleanup;
    ok = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < portable_iterations; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, input_height, input_width, channels, output_height,
                output_width, output_channels, kernel, kernel, channels,
                2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) != 1) {
            ok = 0;
            goto cleanup;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    portable_ms = elapsed_ms(&start, &end) / portable_iterations;
    benchmark_set_num_threads(1);
    for (int iteration = 0; iteration < native_warmup; iteration++) {
        if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
                zero_points, native_single, batch, input_height, input_width,
                channels, output_height, output_width, output_channels, kernel,
                kernel, channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight,
                NULL) != 1) goto cleanup;
    }
    measured_iterations = g_grayscale_single_iterations > 0
        ? g_grayscale_single_iterations : native_iterations;
    for (attempt = 0;
            attempt < (g_grayscale_single_iterations > 0 ? 1 : 6);
            attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < measured_iterations; iteration++) {
            if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
                    zero_points, native_single, batch, input_height,
                    input_width, channels, output_height, output_width,
                    output_channels, kernel, kernel, channels, 2, 2, 1, 1, 1,
                    1, 1, 1, 1, 0, 1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                    VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight,
                    NULL) != 1) goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        native_span_ms = elapsed_ms(&start, &end);
        if (g_grayscale_single_iterations > 0 ||
            g_minimum_hot_span_ms <= 0.0 ||
            native_span_ms >= g_minimum_hot_span_ms) break;
        measured_iterations = next_hot_iterations(
            measured_iterations, native_span_ms);
        if (measured_iterations <= 0) goto cleanup;
    }
    if (!finish_hot_timing(&native_single_timing, measured_iterations,
                           native_span_ms)) goto cleanup;
    benchmark_set_num_threads(4);
    for (int iteration = 0; iteration < native_warmup; iteration++) {
        if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
                zero_points, native, batch, input_height, input_width,
                channels, output_height, output_width, output_channels, kernel,
                kernel, channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight,
                NULL) != 1) goto cleanup;
    }
    measured_iterations = native_iterations;
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < measured_iterations; iteration++) {
            if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
                    zero_points, native, batch, input_height, input_width,
                    channels, output_height, output_width, output_channels,
                    kernel, kernel, channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0,
                    1.0f / 32.0f, 0, 1.0f / 16.0f, 0,
                    VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, packed_weight,
                    NULL) != 1) goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        native_span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            native_span_ms >= g_minimum_hot_span_ms) break;
        measured_iterations = next_hot_iterations(
            measured_iterations, native_span_ms);
        if (measured_iterations <= 0) goto cleanup;
    }
    if (!finish_hot_timing(&native_timing, measured_iterations,
                           native_span_ms)) goto cleanup;
    printf("TinyReceipt grayscale stem QConv2D [1,320,672,1] -> [1,160,336,48]: "
           "portable=%.6f ms single[%s]=%.6f ms tiled[%s,4t]=%.6f ms "
           "thread-speedup=%.2fx portable-speedup=%.2fx checksum=0x%08x "
           "hot-span[grayscale-qconv-single]=%.3f/%d "
           "hot-span[grayscale-qconv-4t]=%.3f/%d\n",
           portable_ms, "native-prepacked",
           native_single_timing.average_ms,
           "native-prepacked", native_timing.average_ms,
           native_single_timing.average_ms / native_timing.average_ms,
           portable_ms / native_timing.average_ms,
           checksum_i8(native, output_elements),
           native_single_timing.span_ms, native_single_timing.iterations,
           native_timing.span_ms, native_timing.iterations);
    ok = 1;
cleanup:
    benchmark_set_num_threads(0);
    free(input);
    free(weight);
    free(portable);
    free(native_single);
    free(native);
    free(packed_weight);
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
    const uint32_t packed_bytes = vx_packed_q8_weight_size(
        kernel * kernel * channels, output_channels);
    void* packed_weight = malloc(packed_bytes);
    struct timespec start, end;
    HotTiming native_single_timing, native_timing;
    double portable_ms;
    double native_span_ms;
    int measured_iterations;
    const int portable_warmup = g_gate_mode ? 0 : warmup;
    const int portable_iterations = g_gate_mode ? 1 : iterations;
    int attempt;
    int ok = 0;
    if (!packed_weight || !packed_bytes) goto cleanup;
    for (int index = 0; index < elements; index++) input[index] = (int8_t)((index * 23) % 127 - 63);
    for (int index = 0; index < weight_elements; index++) weight[index] = (int8_t)((index * 31) % 111 - 55);
    for (int index = 0; index < output_channels; index++) {
        bias[index] = index * 7 - 107;
        scales[index] = 1.0f / (float)(64 + index % 5 * 8);
        zero_points[index] = index % 9 - 4;
    }
    if (vx_pack_q8_weight(packed_weight, packed_bytes, weight,
            kernel * kernel * channels, output_channels,
            VX_DTYPE_I8, 1u) != 1) goto cleanup;
    if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
            batch, height, width, channels, height, width, output_channels,
            kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) != 1) goto cleanup;
    benchmark_set_num_threads(1);
    if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales, zero_points,
            native_single, batch, height, width, channels, height, width,
            output_channels, kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
            packed_weight, NULL) != 1) goto cleanup;
    benchmark_set_num_threads(4);
    if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales, zero_points, native,
            batch, height, width, channels, height, width, output_channels,
            kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
            packed_weight, NULL) != 1 ||
        memcmp(portable, native_single, sizeof(portable)) != 0 ||
        memcmp(portable, native, sizeof(portable)) != 0) goto cleanup;
    for (int iteration = 0; iteration < portable_warmup; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) != 1) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < portable_iterations; iteration++) {
        if (qconv2d_i8u8(input, weight, bias, scales, zero_points, portable,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) != 1) goto cleanup;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    portable_ms = elapsed_ms(&start, &end) / portable_iterations;
    benchmark_set_num_threads(1);
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales, zero_points,
                native_single, batch, height, width, channels, height, width,
                output_channels, kernel, kernel, channels, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 0, 1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                packed_weight, NULL) != 1) goto cleanup;
    }
    measured_iterations = iterations;
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < measured_iterations; iteration++) {
            if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales, zero_points,
                    native_single, batch, height, width, channels, height,
                    width, output_channels, kernel, kernel, channels, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 0, 1.0f / 32.0f, -3,
                    1.0f / 16.0f, -2, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8, packed_weight, NULL) != 1) goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        native_span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            native_span_ms >= g_minimum_hot_span_ms) break;
        measured_iterations = next_hot_iterations(
            measured_iterations, native_span_ms);
        if (measured_iterations <= 0) goto cleanup;
    }
    if (!finish_hot_timing(&native_single_timing, measured_iterations,
                           native_span_ms)) goto cleanup;
    benchmark_set_num_threads(4);
    for (int iteration = 0; iteration < warmup; iteration++) {
        if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales, zero_points, native,
                batch, height, width, channels, height, width, output_channels,
                kernel, kernel, channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                1.0f / 32.0f, -3, 1.0f / 16.0f, -2,
                VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                packed_weight, NULL) != 1) goto cleanup;
    }
    measured_iterations = iterations;
    for (attempt = 0; attempt < 6; attempt++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iteration = 0; iteration < measured_iterations; iteration++) {
            if (vx_qconv2d_i8u8_native_prepacked(input, weight, bias, scales,
                    zero_points, native, batch, height, width, channels,
                    height, width, output_channels, kernel, kernel, channels,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1.0f / 32.0f, -3,
                    1.0f / 16.0f, -2, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8, packed_weight, NULL) != 1) goto cleanup;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        native_span_ms = elapsed_ms(&start, &end);
        if (g_minimum_hot_span_ms <= 0.0 ||
            native_span_ms >= g_minimum_hot_span_ms) break;
        measured_iterations = next_hot_iterations(
            measured_iterations, native_span_ms);
        if (measured_iterations <= 0) goto cleanup;
    }
    if (!finish_hot_timing(&native_timing, measured_iterations,
                           native_span_ms)) goto cleanup;
    printf("QConv2D W8A8: portable=%.6f ms single[%s]=%.6f ms "
           "tiled[%s,4t]=%.6f ms thread-speedup=%.2fx portable-speedup=%.2fx "
           "checksum=0x%08x hot-span[qconv-single]=%.3f/%d "
           "hot-span[qconv-4t]=%.3f/%d\n",
           portable_ms, qconv_w8a8_path(channels, 1),
           native_single_timing.average_ms, qconv_w8a8_path(channels, 1),
           native_timing.average_ms,
           native_single_timing.average_ms / native_timing.average_ms,
           portable_ms / native_timing.average_ms,
           checksum_i8(native, sizeof(native)),
           native_single_timing.span_ms, native_single_timing.iterations,
           native_timing.span_ms, native_timing.iterations);
    ok = 1;
cleanup:
    benchmark_set_num_threads(0);
    free(packed_weight);
    return ok;
}

static int parse_benchmark_arguments(int argc, char** argv) {
    for (int index = 1; index < argc; index++) {
        const char* argument = argv[index];
        const char* timing_prefix = "--min-timed-ms=";
        const char* cpu_prefix = "--single-cpu=";
        const char* grayscale_iterations_prefix =
            "--grayscale-single-iterations=";
        const size_t timing_prefix_length = strlen(timing_prefix);
        const size_t cpu_prefix_length = strlen(cpu_prefix);
        const size_t grayscale_iterations_prefix_length =
            strlen(grayscale_iterations_prefix);
        if (strcmp(argument, "--gate") == 0) {
            g_gate_mode = 1;
        } else if (strncmp(argument, timing_prefix,
                           timing_prefix_length) == 0) {
            char* end = NULL;
            double value;
            errno = 0;
            value = strtod(argument + timing_prefix_length, &end);
            if (errno != 0 || !end || *end != '\0' || !isfinite(value) ||
                value < 1.0 || value > 1000.0) return 0;
            g_minimum_hot_span_ms = value;
        } else if (strncmp(argument, cpu_prefix, cpu_prefix_length) == 0) {
            char* end = NULL;
            long value;
            errno = 0;
            value = strtol(argument + cpu_prefix_length, &end, 10);
            if (errno != 0 || !end || *end != '\0' || value < 0 ||
                value > INT_MAX) return 0;
            g_single_cpu = (int)value;
        } else if (strncmp(argument, grayscale_iterations_prefix,
                           grayscale_iterations_prefix_length) == 0) {
            char* end = NULL;
            long value;
            errno = 0;
            value = strtol(argument + grayscale_iterations_prefix_length,
                           &end, 10);
            if (errno != 0 || !end || *end != '\0' || value <= 0 ||
                value > 100000000L) return 0;
            g_grayscale_single_iterations = (int)value;
        } else {
            return 0;
        }
    }
    if (g_gate_mode && g_minimum_hot_span_ms == 0.0)
        g_minimum_hot_span_ms = 50.0;
    if (!g_gate_mode && g_grayscale_single_iterations > 0) return 0;
    return !g_gate_mode || g_minimum_hot_span_ms > 0.0;
}

int main(int argc, char** argv) {
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    const char* failed_stage = NULL;
    int result;
    if (!parse_benchmark_arguments(argc, argv)) {
        fprintf(stderr, "usage: %s [--gate] [--min-timed-ms=1..1000] "
                "[--single-cpu=N] [--grayscale-single-iterations=N]\n",
                argc > 0 ? argv[0] : "benchmark_w8a8_native");
        return 2;
    }
    if (!benchmark_affinity_initialize()) {
        fputs("native W8A8 benchmark affinity initialization failed\n", stderr);
        return 2;
    }
    pool = vx_kernel_thread_pool_create(0);
    if (!pool) return 1;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    printf("Timing policy: gate=%s minimum-hot-span=%.3f ms\n",
           g_gate_mode ? "on" : "off", g_minimum_hot_span_ms);
    printf("Timing affinity: single-cpu=%d pool-cpus=%d\n",
           g_single_cpu, g_pool_cpu_count);
    if (!g_affinity_ok)
        failed_stage = "initial-affinity";
    else if (!run_tiny_vqa_decoder_linear_mix())
        failed_stage = "decoder-linear";
    else if (!run_tiny_vqa_decoder_prefill_linears())
        failed_stage = "decoder-prefill";
    else if (!run_tiny_vqa_quantized_activations())
        failed_stage = "quantized-activation";
    else if (!run_tiny_vqa_grayscale_stem_qconv())
        failed_stage = "grayscale-stem-qconv";
    else if (!run_qconv())
        failed_stage = "generic-qconv";
    if (!g_thread_count_ok && failed_stage == NULL)
        failed_stage = "thread-count";
    result = failed_stage != NULL;
    if (result) {
        fprintf(stderr,
                "native W8A8/W8A32 benchmark correctness preflight failed: "
                "stage=%s affinity-ok=%d affinity-errno=%d "
                "affinity-threads=%d thread-count-ok=%d "
                "thread-count-expected=%d thread-count-actual=%d\n",
                failed_stage, g_affinity_ok, g_affinity_error,
                g_affinity_failure_threads, g_thread_count_ok,
                g_thread_count_expected, g_thread_count_actual);
    }
    benchmark_set_num_threads(0);
    if (!g_affinity_ok && !result) {
        fprintf(stderr,
                "native W8A8 benchmark final affinity restore failed: "
                "errno=%d threads=%d\n",
                g_affinity_error, g_affinity_failure_threads);
        result = 1;
    }
    vx_kernels_shutdown();
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    return result ? 1 : 0;
}
