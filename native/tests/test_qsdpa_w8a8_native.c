#include "inference_kernels.h"
#include "quant_cpu_isa.h"
#include "thread_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return -1; \
    } \
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
} QSDPATestCase;

static uint32_t next_random(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_bytes(void* values, size_t count, uint32_t dtype,
                       uint32_t* state) {
    for (size_t index = 0; index < count; index++) {
        uint32_t bits = next_random(state);
        if (dtype == VX_DTYPE_I8) {
            ((int8_t*)values)[index] = (int8_t)((int32_t)(bits % 255u) - 127);
        } else {
            ((uint8_t*)values)[index] = (uint8_t)(bits & 255u);
        }
    }
}

static size_t mask_count(const QSDPATestCase* test) {
    if (test->mask_mode == 0u) return 0u;
    if (test->mask_mode == 1u) return test->seq_kv;
    if (test->mask_mode == 2u) return (size_t)test->batch * test->seq_kv;
    if (test->mask_mode == 3u) return (size_t)test->seq_q * test->seq_kv;
    return (size_t)test->batch * test->seq_q * test->seq_kv;
}

static void store_mask_i32(int32_t* mask, size_t index, int32_t value) {
    unsigned char* bytes = (unsigned char*)mask + index * sizeof(value);
    const uint32_t bits = (uint32_t)value;
    bytes[0] = (unsigned char)(bits & 255u);
    bytes[1] = (unsigned char)((bits >> 8u) & 255u);
    bytes[2] = (unsigned char)((bits >> 16u) & 255u);
    bytes[3] = (unsigned char)((bits >> 24u) & 255u);
}

static int run_case(const QSDPATestCase* test, uint32_t seed) {
    const size_t q_count = (size_t)test->batch * test->seq_q * test->d_model;
    const size_t kv_count = (size_t)test->batch * test->seq_kv * test->d_model;
    const size_t keep_count = mask_count(test);
    unsigned char* q = (unsigned char*)malloc(q_count);
    unsigned char* k = (unsigned char*)malloc(kv_count);
    unsigned char* v = (unsigned char*)malloc(kv_count);
    unsigned char* reference = (unsigned char*)malloc(q_count);
    unsigned char* actual = (unsigned char*)malloc(q_count);
    unsigned char* keep_storage = keep_count ?
        (unsigned char*)malloc(keep_count * sizeof(int32_t) + 1u) : NULL;
    /* SafeTensors offsets are byte-aligned, not necessarily I32-aligned. */
    int32_t* keep = keep_storage ? (int32_t*)(keep_storage + 1u) : NULL;
    int result = -1;
    CHECK(q && k && v && reference && actual && (!keep_count || keep));
    fill_bytes(q, q_count, test->q_dtype, &seed);
    fill_bytes(k, kv_count, test->k_dtype, &seed);
    fill_bytes(v, kv_count, test->v_dtype, &seed);
    for (size_t index = 0; index < keep_count; index++) {
        /* Keep most keys, but exercise all-masked query rows as well. */
        store_mask_i32(keep, index,
                       index % 17u != 0u && index % 29u != 0u);
    }
    if (test->mask_mode >= 3u && keep_count >= test->seq_kv)
        memset((unsigned char*)keep + test->seq_kv * sizeof(*keep), 0,
               test->seq_kv * sizeof(*keep));
    memset(reference, 0xa5, q_count);
    memset(actual, 0x5a, q_count);
    CHECK(qsdpa_i8u8(q, k, v, keep, reference, test->batch, test->seq_q,
                     test->seq_kv, test->d_model, test->heads,
                     0.019f, test->q_dtype == VX_DTYPE_I8 ? -3 : 123,
                     0.017f, test->k_dtype == VX_DTYPE_I8 ? 5 : 131,
                     0.023f, test->v_dtype == VX_DTYPE_I8 ? -7 : 119,
                     0.031f, test->output_dtype == VX_DTYPE_I8 ? 4 : 127,
                     0.25f, test->q_dtype, test->k_dtype, test->v_dtype,
                     test->output_dtype, test->causal, test->mask_mode) == 1);
    vx_set_num_threads(4);
    CHECK(vx_qsdpa_i8u8_native_validated(
                     q, k, v, keep, actual, test->batch, test->seq_q,
                     test->seq_kv, test->d_model, test->heads,
                     0.019f, test->q_dtype == VX_DTYPE_I8 ? -3 : 123,
                     0.017f, test->k_dtype == VX_DTYPE_I8 ? 5 : 131,
                     0.023f, test->v_dtype == VX_DTYPE_I8 ? -7 : 119,
                     0.031f, test->output_dtype == VX_DTYPE_I8 ? 4 : 127,
                     0.25f, test->q_dtype, test->k_dtype, test->v_dtype,
                     test->output_dtype, test->causal, test->mask_mode) == 1);
    CHECK(memcmp(reference, actual, q_count) == 0);
    if (test->seq_q > 2u) {
        const uint32_t query_start = 1u;
        const uint32_t query_count = test->seq_q / 2u;
        memset(actual, 0x5a, q_count);
        CHECK(vx_qsdpa_i8u8_native_range_validated(
                         q, k, v, keep, actual, test->batch, test->seq_q,
                         test->seq_kv, test->d_model, test->heads,
                         0.019f, test->q_dtype == VX_DTYPE_I8 ? -3 : 123,
                         0.017f, test->k_dtype == VX_DTYPE_I8 ? 5 : 131,
                         0.023f, test->v_dtype == VX_DTYPE_I8 ? -7 : 119,
                         0.031f, test->output_dtype == VX_DTYPE_I8 ? 4 : 127,
                         0.25f, test->q_dtype, test->k_dtype, test->v_dtype,
                         test->output_dtype, test->causal, test->mask_mode,
                         query_start, query_count) == 1);
        for (uint32_t batch = 0; batch < test->batch; batch++) {
            for (uint32_t query = 0; query < test->seq_q; query++) {
                const size_t offset = ((size_t)batch * test->seq_q + query) *
                    test->d_model;
                if (query >= query_start &&
                    query < query_start + query_count) {
                    CHECK(memcmp(actual + offset, reference + offset,
                                 test->d_model) == 0);
                } else {
                    for (uint32_t dimension = 0; dimension < test->d_model;
                         dimension++)
                        CHECK(actual[offset + dimension] == 0x5a);
                }
            }
        }
    }
    result = 0;
    free(keep_storage);
    free(actual);
    free(reference);
    free(v);
    free(k);
    free(q);
    return result;
}

int main(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(0);
    VxKernelThreadPoolScope scope;
    if (!pool) return 1;
    scope = vx_kernel_thread_pool_scope_enter(pool);
    static const QSDPATestCase cases[] = {
        /* Representative bounded decoder cross-attention dimensions. */
        {1u, 192u, 402u, 320u, 8u, VX_DTYPE_I8, VX_DTYPE_I8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u},
        {1u, 96u, 96u, 128u, 8u, VX_DTYPE_U8, VX_DTYPE_I8,
         VX_DTYPE_U8, VX_DTYPE_U8, 1u, 1u},
        /* Head dimension 12 exercises the AVX2 dot's scalar four-byte tail. */
        {1u, 17u, 23u, 96u, 8u, VX_DTYPE_U8, VX_DTYPE_U8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u},
        {2u, 48u, 64u, 128u, 8u, VX_DTYPE_I8, VX_DTYPE_U8,
         VX_DTYPE_I8, VX_DTYPE_U8, 0u, 2u},
        {2u, 48u, 64u, 128u, 8u, VX_DTYPE_U8, VX_DTYPE_U8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 3u},
        {2u, 48u, 64u, 128u, 8u, VX_DTYPE_I8, VX_DTYPE_I8,
         VX_DTYPE_U8, VX_DTYPE_U8, 1u, 4u},
        /* Incremental rows use the same runtime-dispatched exact dot route. */
        {1u, 1u, 96u, 128u, 8u, VX_DTYPE_I8, VX_DTYPE_I8,
         VX_DTYPE_I8, VX_DTYPE_I8, 0u, 1u},
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++)
        CHECK(run_case(&cases[index], 0x51d9a77bu + (uint32_t)index) == 0);
    vx_set_num_threads(0);
    vx_kernels_shutdown();
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    puts("native QSDPA parallel parity tests passed");
    return 0;
}
