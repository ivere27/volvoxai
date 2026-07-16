#include "gemm_f32.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __wasm__
#include <pthread.h>
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 0; \
    } \
} while (0)

static float test_value(uint32_t index) {
    uint32_t bits = index * 1664525u + 1013904223u;
    return ((float)(int32_t)(bits >> 8u) / 8388608.0f) * 0.25f;
}

static void reference_gemm(const float* a, const float* weight,
                           const float* bias, float* output,
                           uint32_t m, uint32_t k, uint32_t n,
                           int out_in, int accumulate) {
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t column = 0; column < n; column++) {
            size_t output_index = (size_t)row * n + column;
            float sum = (bias ? bias[column] : 0.0f) +
                        (accumulate ? output[output_index] : 0.0f);
            for (uint32_t inner = 0; inner < k; inner++) {
                size_t weight_index = out_in
                    ? (size_t)column * k + inner
                    : (size_t)inner * n + column;
                sum += a[(size_t)row * k + inner] * weight[weight_index];
            }
            output[output_index] = sum;
        }
    }
}

static int close_array(const float* actual, const float* expected, size_t count) {
    for (size_t index = 0; index < count; index++) {
        float tolerance = 2.5e-4f * fmaxf(1.0f, fabsf(expected[index]));
        if (fabsf(actual[index] - expected[index]) > tolerance) {
            fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g (tol %.9g)\n",
                    index, actual[index], expected[index], tolerance);
            return 0;
        }
    }
    return 1;
}

#ifndef __wasm__
enum { TILE_CONFIG_THREADS = 16 };

static void* read_tile_config(void* opaque) {
    VxGemmF32TileConfig* output = (VxGemmF32TileConfig*)opaque;
    *output = vx_gemm_f32_tile_config();
    return NULL;
}

static int same_tile_config(const VxGemmF32TileConfig* a,
                            const VxGemmF32TileConfig* b) {
    return a->mr == b->mr && a->nr == b->nr && a->kc == b->kc &&
        a->l1_bytes == b->l1_bytes &&
        a->cache_budget_bytes == b->cache_budget_bytes &&
        a->working_set_bytes == b->working_set_bytes;
}

static int test_concurrent_tile_initialization(void) {
    pthread_t threads[TILE_CONFIG_THREADS];
    VxGemmF32TileConfig configs[TILE_CONFIG_THREADS];
    for (int index = 0; index < TILE_CONFIG_THREADS; index++)
        CHECK(pthread_create(&threads[index], NULL, read_tile_config,
                             &configs[index]) == 0);
    for (int index = 0; index < TILE_CONFIG_THREADS; index++)
        CHECK(pthread_join(threads[index], NULL) == 0);
    for (int index = 1; index < TILE_CONFIG_THREADS; index++)
        CHECK(same_tile_config(&configs[0], &configs[index]));
    return 1;
}
#endif

static int run_case(uint32_t m, uint32_t k, uint32_t n,
                    int out_in, int with_bias, int accumulate) {
    size_t a_count = (size_t)m * k;
    size_t weight_count = (size_t)k * n;
    size_t output_count = (size_t)m * n;
    uint32_t packed_count = vx_gemm_f32_packed_elements(k, n);
    float* a = (float*)malloc(a_count * sizeof(float));
    float* weight = (float*)malloc(weight_count * sizeof(float));
    float* bias = with_bias ? (float*)malloc((size_t)n * sizeof(float)) : NULL;
    float* packed = (float*)malloc((size_t)packed_count * sizeof(float));
    float* actual = (float*)malloc(output_count * sizeof(float));
    float* expected = (float*)malloc(output_count * sizeof(float));
    CHECK(a && weight && packed && actual && expected && (!with_bias || bias));
    for (size_t index = 0; index < a_count; index++) a[index] = test_value((uint32_t)index + 11u);
    for (size_t index = 0; index < weight_count; index++) weight[index] = test_value((uint32_t)index + 101u);
    for (uint32_t index = 0; bias && index < n; index++) bias[index] = test_value(index + 701u);
    for (size_t index = 0; index < output_count; index++)
        actual[index] = expected[index] = accumulate ? test_value((uint32_t)index + 1701u) : -99.0f;
    CHECK(vx_gemm_f32_pack_b(weight, packed, k, n, out_in) == 1);
    reference_gemm(a, weight, bias, expected, m, k, n, out_in, accumulate);
    CHECK((accumulate
        ? vx_gemm_f32_run_packed_add(a, packed, bias, actual, m, k, n)
        : vx_gemm_f32_run_packed(a, packed, bias, actual, m, k, n)) == 1);
    CHECK(close_array(actual, expected, output_count));
    free(a);
    free(weight);
    free(bias);
    free(packed);
    free(actual);
    free(expected);
    return 1;
}

static int test_cache_lifetime(void) {
#ifndef __wasm__
    enum { K = 17, N = 9 };
    float weight[K * N];
    float input[K];
    float expected[N];
    float actual[N];
    for (uint32_t index = 0; index < K * N; index++) weight[index] = test_value(index + 31u);
    for (uint32_t index = 0; index < K; index++) input[index] = test_value(index + 331u);
    const float* first = vx_gemm_f32_pack_cache(7, weight, K, N, 0);
    const float* second = vx_gemm_f32_pack_cache(7, weight, K, N, 0);
    CHECK(first && second == first);
    weight[0] += 2.0f;
    vx_gemm_f32_cache_free_all();
    const float* refreshed = vx_gemm_f32_pack_cache(7, weight, K, N, 0);
    CHECK(refreshed);
    reference_gemm(input, weight, NULL, expected, 1, K, N, 0, 0);
    CHECK(vx_gemm_f32_run_packed(input, refreshed, NULL, actual, 1, K, N) == 1);
    CHECK(close_array(actual, expected, N));
    vx_gemm_f32_cache_free_all();
#endif
    return 1;
}

int main(void) {
#ifndef __wasm__
    if (!test_concurrent_tile_initialization()) return 1;
#endif
    VxGemmF32TileConfig tile = vx_gemm_f32_tile_config();
    if (tile.mr != 4u || tile.nr != 8u || tile.kc < 64u ||
        tile.working_set_bytes > tile.cache_budget_bytes ||
        vx_gemm_f32_packed_elements(0, 8) != 0u ||
        vx_gemm_f32_packed_elements(8, 0) != 0u) {
        fprintf(stderr, "invalid gemm_f32 tile or dimension policy\n");
        return 1;
    }
    if (!run_case(1, 513, 13, 1, 1, 0)) return 1; /* decode specialization + N tail */
    if (!run_case(7, 511, 16, 0, 1, 0)) return 1; /* MR tail + IN_OUT packing */
    if (!run_case(4, 37, 7, 1, 0, 0)) return 1;   /* scalar N tail */
    if (!run_case(3, 65, 9, 0, 1, 1)) return 1;   /* additive gradient semantics */
    if (!run_case(9, 257, 33, 1, 1, 0)) return 1; /* native parallel tile path */
    if (!test_cache_lifetime()) return 1;
    printf("gemm_f32 tests passed (MR=%u NR=%u KC=%u budget=%u working=%u)\n",
           tile.mr, tile.nr, tile.kc, tile.cache_budget_bytes,
           tile.working_set_bytes);
    return 0;
}
