#define _POSIX_C_SOURCE 200809L
#include "gemm_f32.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void scalar_linear_f32(const float* input, const float* weight,
                              const float* bias, float* output,
                              uint32_t rows, uint32_t k, uint32_t n) {
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < n; column++) {
            float sum = bias ? bias[column] : 0.0f;
            for (uint32_t inner = 0; inner < k; inner++)
                sum += input[(size_t)row * k + inner] *
                       weight[(size_t)column * k + inner];
            output[(size_t)row * n + column] = sum;
        }
    }
}

static double now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1.0e6;
}

static float value_at(size_t index) {
    uint32_t value = (uint32_t)index * 1103515245u + 12345u;
    return ((float)(int32_t)(value >> 9u) / 4194304.0f) * 0.125f;
}

static int benchmark_case(uint32_t m, uint32_t k, uint32_t n, int iterations) {
    size_t a_count = (size_t)m * k;
    size_t b_count = (size_t)k * n;
    size_t c_count = (size_t)m * n;
    uint32_t packed_count = vx_gemm_f32_packed_elements(k, n);
    float* a = (float*)malloc(a_count * sizeof(float));
    float* b = (float*)malloc(b_count * sizeof(float));
    float* bias = (float*)malloc((size_t)n * sizeof(float));
    float* reference = (float*)malloc(c_count * sizeof(float));
    float* output = (float*)malloc(c_count * sizeof(float));
    float* packed = (float*)malloc((size_t)packed_count * sizeof(float));
    if (!a || !b || !bias || !reference || !output || !packed) return 0;
    for (size_t index = 0; index < a_count; index++) a[index] = value_at(index + 1u);
    for (size_t index = 0; index < b_count; index++) b[index] = value_at(index + 991u);
    for (uint32_t index = 0; index < n; index++) bias[index] = value_at(index + 1991u);
    double pack_start = now_ms();
    if (!vx_gemm_f32_pack_b(b, packed, k, n, 1)) return 0;
    double pack_ms = now_ms() - pack_start;
    scalar_linear_f32(a, b, bias, reference, m, k, n);
    if (!vx_gemm_f32_run_packed(a, packed, bias, output, m, k, n)) return 0;
    float max_error = 0.0f;
    for (size_t index = 0; index < c_count; index++) {
        float error = fabsf(output[index] - reference[index]);
        if (error > max_error) max_error = error;
    }
    double scalar_start = now_ms();
    for (int iteration = 0; iteration < iterations; iteration++)
        scalar_linear_f32(a, b, bias, reference, m, k, n);
    double scalar_ms = now_ms() - scalar_start;
    double packed_start = now_ms();
    for (int iteration = 0; iteration < iterations; iteration++)
        vx_gemm_f32_run_packed(a, packed, bias, output, m, k, n);
    double packed_ms = now_ms() - packed_start;
    double operations = 2.0 * (double)m * k * n * iterations;
    printf("M=%u K=%u N=%u pack=%.3fms scalar=%.3fms (%.2f GF/s) "
           "packed=%.3fms (%.2f GF/s) speedup=%.2fx max_error=%.6g\n",
           m, k, n, pack_ms, scalar_ms, operations / scalar_ms / 1.0e6,
           packed_ms, operations / packed_ms / 1.0e6,
           scalar_ms / packed_ms, max_error);
    free(a);
    free(b);
    free(bias);
    free(reference);
    free(output);
    free(packed);
    return max_error <= 5.0e-3f;
}

int main(void) {
    VxGemmF32TileConfig tile = vx_gemm_f32_tile_config();
    printf("gemm_f32 policy: MR=%u NR=%u KC=%u L1D=%u budget=%u working=%u\n",
           tile.mr, tile.nr, tile.kc, tile.l1_bytes,
           tile.cache_budget_bytes, tile.working_set_bytes);
    if (!benchmark_case(1, 1024, 1024, 100)) return 1;
    if (!benchmark_case(1, 319, 333, 200)) return 1;
    if (!benchmark_case(4, 319, 333, 100)) return 1;
    if (!benchmark_case(8, 319, 333, 50)) return 1;
    if (!benchmark_case(32, 319, 333, 20)) return 1;
    return 0;
}
