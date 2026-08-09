#define _POSIX_C_SOURCE 200809L

#include "inference_kernels.h"
#include "quant_cpu_opt.h"
#include "thread_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed_ms(const struct timespec* start,
                         const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
        (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static uint32_t checksum(const int8_t* values, size_t count) {
    uint32_t value = 2166136261u;
    for (size_t index = 0; index < count; index++)
        value = (value ^ (uint8_t)values[index]) * 16777619u;
    return value;
}

static double time_portable(const int8_t* q, const int8_t* k, const int8_t* v,
                            int8_t* output, uint32_t seq_q, uint32_t seq_kv,
                            uint32_t d_model, uint32_t causal, int iterations) {
    struct timespec start;
    struct timespec end;
    for (int iteration = 0; iteration < 2; iteration++)
        if (!qsdpa_i8u8(q, k, v, NULL, output, 1u, seq_q, seq_kv, d_model, 8u,
                        0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                        0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                        VX_DTYPE_I8, causal, 0u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!qsdpa_i8u8(q, k, v, NULL, output, 1u, seq_q, seq_kv, d_model, 8u,
                        0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                        0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                        VX_DTYPE_I8, causal, 0u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_ms(&start, &end) / iterations;
}

static double time_native(const int8_t* q, const int8_t* k, const int8_t* v,
                          int8_t* output, uint32_t seq_q, uint32_t seq_kv,
                          uint32_t d_model, uint32_t causal, int threads,
                          int iterations) {
    struct timespec start;
    struct timespec end;
    vx_set_num_threads(threads);
    for (int iteration = 0; iteration < 2; iteration++)
        if (!vx_qsdpa_i8u8_native_validated(
                        q, k, v, NULL, output, 1u, seq_q, seq_kv, d_model, 8u,
                        0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                        0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                        VX_DTYPE_I8, causal, 0u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < iterations; iteration++)
        if (!vx_qsdpa_i8u8_native_validated(
                        q, k, v, NULL, output, 1u, seq_q, seq_kv, d_model, 8u,
                        0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                        0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                        VX_DTYPE_I8, causal, 0u)) return -1.0;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed_ms(&start, &end) / iterations;
}

static int run_case(const char* label, uint32_t seq_q, uint32_t seq_kv,
                    uint32_t d_model, uint32_t causal, int iterations) {
    const size_t q_count = (size_t)seq_q * d_model;
    const size_t kv_count = (size_t)seq_kv * d_model;
    int8_t* q = (int8_t*)malloc(q_count);
    int8_t* k = (int8_t*)malloc(kv_count);
    int8_t* v = (int8_t*)malloc(kv_count);
    int8_t* reference = (int8_t*)malloc(q_count);
    int8_t* actual = (int8_t*)malloc(q_count);
    double portable_ms;
    double native_ms;
    if (!q || !k || !v || !reference || !actual) return -1;
    for (size_t index = 0; index < q_count; index++)
        q[index] = (int8_t)((index * 37u + 11u) % 251u - 125);
    for (size_t index = 0; index < kv_count; index++) {
        k[index] = (int8_t)((index * 29u + 7u) % 251u - 125);
        v[index] = (int8_t)((index * 43u + 3u) % 251u - 125);
    }
    if (!qsdpa_i8u8(q, k, v, NULL, reference, 1u, seq_q, seq_kv, d_model, 8u,
                    0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                    0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8, causal, 0u)) return -1;
    vx_set_num_threads(4);
    if (!vx_qsdpa_i8u8_native_validated(
                    q, k, v, NULL, actual, 1u, seq_q, seq_kv, d_model, 8u,
                    0.02f, 0, 0.018f, 0, 0.025f, 0, 0.03f, 0,
                    0.25f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                    VX_DTYPE_I8, causal, 0u) ||
        memcmp(reference, actual, q_count) != 0) {
        fprintf(stderr, "%s parity failed\n", label);
        return -1;
    }
    portable_ms = time_portable(q, k, v, reference, seq_q, seq_kv, d_model,
                                causal, iterations);
    native_ms = time_native(q, k, v, actual, seq_q, seq_kv, d_model, causal,
                            4, iterations);
    if (portable_ms <= 0.0 || native_ms <= 0.0) return -1;
    printf("qsdpa[%s,Q=%u,K=%u,D=%u,H=8] portable=%.3f ms "
           "native[4t]=%.3f ms speedup=%.2fx checksum=%08x\n",
           label, seq_q, seq_kv, d_model, portable_ms, native_ms,
           portable_ms / native_ms, checksum(actual, q_count));
    free(actual);
    free(reference);
    free(v);
    free(k);
    free(q);
    return 0;
}

int main(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(0);
    VxKernelThreadPoolScope scope;
    int result;
    if (!pool) return 1;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    result = run_case("bounded-encoder-self", 402u, 402u, 320u, 0u, 4) != 0 ||
        run_case("bounded-decoder-self", 192u, 192u, 320u, 1u, 8) != 0 ||
        run_case("bounded-decoder-cross", 192u, 402u, 320u, 0u, 5) != 0 ||
        run_case("incremental-cross-row", 1u, 402u, 320u, 0u, 300) != 0;
    vx_set_num_threads(0);
    vx_kernels_shutdown();
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    return result ? 1 : 0;
}
