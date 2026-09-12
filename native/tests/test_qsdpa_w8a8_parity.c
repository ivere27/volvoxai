#include "inference_kernels.h"
#include "quant_cpu_isa.h"
#include "thread_pool.h"

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

typedef struct {
    uint32_t batch;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t output_dtype;
    uint32_t causal;
    uint32_t mask_mode;
} QsdpaCase;

static uint32_t next_random(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_bytes(void* values, size_t count, uint32_t* state) {
    uint8_t* bytes = (uint8_t*)values;
    for (size_t index = 0; index < count; index++)
        bytes[index] = (uint8_t)(next_random(state) >> 24u);
}

static size_t mask_elements(const QsdpaCase* test) {
    if (!test->mask_mode) return 0u;
    if (test->mask_mode == 1u) return test->seq_kv;
    if (test->mask_mode == 2u) return (size_t)test->batch * test->seq_kv;
    if (test->mask_mode == 3u) return (size_t)test->seq_q * test->seq_kv;
    return (size_t)test->batch * test->seq_q * test->seq_kv;
}

/* Tensor payload offsets are byte-aligned, not necessarily I32-aligned. */
static void store_mask(int32_t* mask, size_t index, int32_t value) {
    uint8_t* bytes = (uint8_t*)mask + index * sizeof(value);
    const uint32_t bits = (uint32_t)value;
    bytes[0] = (uint8_t)bits;
    bytes[1] = (uint8_t)(bits >> 8u);
    bytes[2] = (uint8_t)(bits >> 16u);
    bytes[3] = (uint8_t)(bits >> 24u);
}

static int run_case(const QsdpaCase* test, uint32_t seed) {
    const size_t q_count = (size_t)test->batch * test->seq_q * test->d_model;
    const size_t kv_count =
        (size_t)test->batch * test->seq_kv * test->d_model;
    const size_t keep_count = mask_elements(test);
    uint8_t* q = (uint8_t*)malloc(q_count);
    uint8_t* k = (uint8_t*)malloc(kv_count);
    uint8_t* v = (uint8_t*)malloc(kv_count);
    uint8_t* portable = (uint8_t*)malloc(q_count);
    uint8_t* native = (uint8_t*)malloc(q_count);
    uint8_t* mask_storage = keep_count
        ? (uint8_t*)malloc(keep_count * sizeof(int32_t) + 1u) : NULL;
    int32_t* mask = mask_storage ? (int32_t*)(mask_storage + 1u) : NULL;
    const int32_t q_zero = test->q_dtype == VX_DTYPE_I8 ? -3 : 123;
    const int32_t k_zero = test->k_dtype == VX_DTYPE_I8 ? 5 : 131;
    const int32_t v_zero = test->v_dtype == VX_DTYPE_I8 ? -7 : 119;
    const int32_t output_zero =
        test->output_dtype == VX_DTYPE_I8 ? 4 : 127;

    CHECK(q && k && v && portable && native && (!keep_count || mask));
    fill_bytes(q, q_count, &seed);
    fill_bytes(k, kv_count, &seed);
    fill_bytes(v, kv_count, &seed);
    for (size_t index = 0; index < keep_count; index++)
        store_mask(mask, index, index % 5u != 0u && index % 13u != 0u);
    if (test->mask_mode >= 3u && keep_count >= 2u * test->seq_kv)
        memset((uint8_t*)mask + test->seq_kv * sizeof(*mask), 0,
               test->seq_kv * sizeof(*mask));
    CHECK(qsdpa_i8u8(
        q, k, v, mask, portable, test->batch, test->seq_q, test->seq_kv,
        test->d_model, test->heads, 0.019f, q_zero, 0.017f, k_zero,
        0.023f, v_zero, 0.031f, output_zero, 0.25f, test->q_dtype,
        test->k_dtype, test->v_dtype, test->output_dtype, test->causal,
        test->mask_mode) == 1);
    CHECK(vx_qsdpa_i8u8_native_validated(
        q, k, v, mask, native, test->batch, test->seq_q, test->seq_kv,
        test->d_model, test->heads, 0.019f, q_zero, 0.017f, k_zero,
        0.023f, v_zero, 0.031f, output_zero, 0.25f, test->q_dtype,
        test->k_dtype, test->v_dtype, test->output_dtype, test->causal,
        test->mask_mode) == 1);
    CHECK(memcmp(portable, native, q_count) == 0);

    if (test->seq_q > 2u) {
        const uint32_t query_start = 1u;
        const uint32_t query_count = test->seq_q / 2u;
        memset(native, 0x5a, q_count);
        CHECK(vx_qsdpa_i8u8_native_range_validated(
            q, k, v, mask, native, test->batch, test->seq_q, test->seq_kv,
            test->d_model, test->heads, 0.019f, q_zero, 0.017f, k_zero,
            0.023f, v_zero, 0.031f, output_zero, 0.25f, test->q_dtype,
            test->k_dtype, test->v_dtype, test->output_dtype, test->causal,
            test->mask_mode, query_start, query_count) == 1);
        for (uint32_t batch = 0; batch < test->batch; batch++) {
            for (uint32_t query = 0; query < test->seq_q; query++) {
                const size_t offset =
                    ((size_t)batch * test->seq_q + query) * test->d_model;
                if (query >= query_start &&
                    query < query_start + query_count) {
                    CHECK(memcmp(native + offset, portable + offset,
                                 test->d_model) == 0);
                } else {
                    for (uint32_t dimension = 0;
                         dimension < test->d_model; dimension++)
                        CHECK(native[offset + dimension] == 0x5a);
                }
            }
        }
    }
    free(mask_storage);
    free(native);
    free(portable);
    free(v);
    free(k);
    free(q);
    return 1;
}

int main(void) {
    static const QsdpaCase cases[] = {
        /* Odd query/key/head tails. */
        /* The product count is just above the production 4-thread threshold. */
        {1u, 17u, 323u, 96u, 8u, VX_DTYPE_U8, VX_DTYPE_U8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u},
        /* Batch-key mask and mixed activation domains. */
        {2u, 8u, 11u, 64u, 8u, VX_DTYPE_I8, VX_DTYPE_U8,
         VX_DTYPE_I8, VX_DTYPE_U8, 0u, 2u},
        /* Full mask includes an all-masked row; causal bounds keys as well. */
        {2u, 8u, 11u, 64u, 8u, VX_DTYPE_U8, VX_DTYPE_I8,
         VX_DTYPE_U8, VX_DTYPE_I8, 1u, 4u},
        /* Incremental query with key mask. */
        {1u, 1u, 17u, 64u, 8u, VX_DTYPE_I8, VX_DTYPE_I8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 1u},
    };
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    VxKernelThreadPoolScope scope;
    int ok = 1;
    if (!pool) return 1;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    if (vx_kernels_thread_count() != 4) ok = 0;
    for (size_t index = 0;
         ok && index < sizeof(cases) / sizeof(cases[0]); index++)
        ok = run_case(&cases[index], 0x51d9a77bu + (uint32_t)index);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    if (!ok) return 1;
    puts("QSDPA W8A8 portable/native parity passed");
    return 0;
}
