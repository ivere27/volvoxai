/*
 * Native CPU parallelism for canonical physical-byte W8A8 QSDPA.
 *
 * The portable qsdpa_i8u8() kernel remains the authoritative implementation
 * and the sole WASM path.  Whole-tensor native execution can partition its
 * independent [batch, query] rows, then invoke that same kernel for each row.
 * Causal rows expose exactly their original K/V prefix and query-dependent
 * masks are reduced to the corresponding key row, so key iteration and every
 * head's arithmetic order remain unchanged.
 */
#include "quant_cpu_opt.h"
#include "thread_pool.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

extern int qsdpa_i8u8(const void* q, const void* k, const void* v,
        const int32_t* mask, void* output, uint32_t batch, uint32_t seq_q,
        uint32_t seq_kv, uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode);

enum {
    VX_QSDPA_MASK_NONE = 0u,
    VX_QSDPA_MASK_KEY = 1u,
    VX_QSDPA_MASK_BATCH_KEY = 2u,
    VX_QSDPA_MASK_QUERY_KEY = 3u,
    VX_QSDPA_MASK_BATCH_QUERY_KEY = 4u,
    /* Below this many QK scalar products, pool wake-up tends to cost more
     * than it saves.  seq_q == 1 is separately kept on the caller so the
     * incremental decoder never pays parallel scheduling overhead. */
    VX_QSDPA_PARALLEL_PRODUCTS = 512u * 1024u,
};

typedef struct {
    const void* q;
    const void* k;
    const void* v;
    const int32_t* mask;
    void* output;
    uint32_t batch;
    uint32_t seq_q;
    uint32_t seq_kv;
    uint32_t d_model;
    uint32_t heads;
    float q_scale;
    int32_t q_zero_point;
    float k_scale;
    int32_t k_zero_point;
    float v_scale;
    int32_t v_zero_point;
    float output_scale;
    int32_t output_zero_point;
    float attention_scale;
    uint32_t q_dtype;
    uint32_t k_dtype;
    uint32_t v_dtype;
    uint32_t output_dtype;
    uint32_t causal;
    uint32_t mask_mode;
    unsigned char* failed;
} VxQSDPAParallelContext;

static const int32_t* vx_qsdpa_row_mask(const VxQSDPAParallelContext* context,
                                        uint32_t batch_index,
                                        uint32_t query) {
    size_t index;
    if (!context->mask || context->mask_mode == VX_QSDPA_MASK_NONE) return NULL;
    if (context->mask_mode == VX_QSDPA_MASK_KEY) return context->mask;
    if (context->mask_mode == VX_QSDPA_MASK_BATCH_KEY) {
        index = (size_t)batch_index * context->seq_kv;
        return (const int32_t*)((const unsigned char*)context->mask +
                                index * sizeof(*context->mask));
    }
    if (context->mask_mode == VX_QSDPA_MASK_QUERY_KEY) {
        index = (size_t)query * context->seq_kv;
        return (const int32_t*)((const unsigned char*)context->mask +
                                index * sizeof(*context->mask));
    }
    index = ((size_t)batch_index * context->seq_q + query) * context->seq_kv;
    return (const int32_t*)((const unsigned char*)context->mask +
                            index * sizeof(*context->mask));
}

static void vx_qsdpa_parallel_worker(void* opaque, int begin, int end) {
    VxQSDPAParallelContext* context = (VxQSDPAParallelContext*)opaque;
    for (int task = begin; task < end; task++) {
        const uint32_t batch_index = (uint32_t)task / context->seq_q;
        const uint32_t query = (uint32_t)task % context->seq_q;
        const size_t q_offset = ((size_t)batch_index * context->seq_q + query) *
            context->d_model;
        const size_t kv_offset = (size_t)batch_index * context->seq_kv *
            context->d_model;
        uint32_t row_seq_kv = context->seq_kv;
        const int32_t* row_mask = vx_qsdpa_row_mask(context, batch_index, query);
        if (context->causal && query + 1u < row_seq_kv) row_seq_kv = query + 1u;
        context->failed[task] = (unsigned char)(qsdpa_i8u8(
                       (const unsigned char*)context->q + q_offset,
                       (const unsigned char*)context->k + kv_offset,
                       (const unsigned char*)context->v + kv_offset,
                       row_mask,
                       (unsigned char*)context->output + q_offset,
                       1u, 1u, row_seq_kv, context->d_model, context->heads,
                       context->q_scale, context->q_zero_point,
                       context->k_scale, context->k_zero_point,
                       context->v_scale, context->v_zero_point,
                       context->output_scale, context->output_zero_point,
                       context->attention_scale, context->q_dtype,
                       context->k_dtype, context->v_dtype,
                       context->output_dtype, 0u,
                       row_mask ? VX_QSDPA_MASK_KEY : VX_QSDPA_MASK_NONE) != 1);
    }
}

static int vx_qsdpa_parallel_worthwhile(uint32_t batch, uint32_t seq_q,
                                        uint32_t seq_kv, uint32_t d_model) {
    uint64_t rows;
    uint64_t products;
    int threads;
    if (seq_q <= 1u) return 0;
    threads = vx_kernels_thread_count();
    if (threads <= 1) return 0;
    rows = (uint64_t)batch * seq_q;
    if (rows > INT_MAX || rows < (uint64_t)threads * 2u) return 0;
    products = rows * seq_kv;
    if (d_model && products > UINT64_MAX / d_model) return 1;
    products *= d_model;
    return products >= VX_QSDPA_PARALLEL_PRODUCTS;
}

int vx_qsdpa_i8u8_native_validated(const void* q, const void* k, const void* v,
        const int32_t* mask, void* output, uint32_t batch, uint32_t seq_q,
        uint32_t seq_kv, uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode) {
    VxQSDPAParallelContext context;
    uint64_t rows;
    unsigned char* failed;
    int result = 1;
    if (!vx_qsdpa_parallel_worthwhile(batch, seq_q, seq_kv, d_model)) {
        return qsdpa_i8u8(q, k, v, mask, output, batch, seq_q, seq_kv,
                          d_model, heads, q_scale, q_zero_point, k_scale,
                          k_zero_point, v_scale, v_zero_point, output_scale,
                          output_zero_point, attention_scale, q_dtype, k_dtype,
                          v_dtype, output_dtype, causal, mask_mode);
    }
    rows = (uint64_t)batch * seq_q;
    failed = (unsigned char*)calloc((size_t)rows, sizeof(*failed));
    if (!failed) {
        return qsdpa_i8u8(q, k, v, mask, output, batch, seq_q, seq_kv,
                          d_model, heads, q_scale, q_zero_point, k_scale,
                          k_zero_point, v_scale, v_zero_point, output_scale,
                          output_zero_point, attention_scale, q_dtype, k_dtype,
                          v_dtype, output_dtype, causal, mask_mode);
    }
    context = (VxQSDPAParallelContext){
        .q = q, .k = k, .v = v, .mask = mask, .output = output,
        .batch = batch, .seq_q = seq_q, .seq_kv = seq_kv,
        .d_model = d_model, .heads = heads,
        .q_scale = q_scale, .q_zero_point = q_zero_point,
        .k_scale = k_scale, .k_zero_point = k_zero_point,
        .v_scale = v_scale, .v_zero_point = v_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .attention_scale = attention_scale,
        .q_dtype = q_dtype, .k_dtype = k_dtype, .v_dtype = v_dtype,
        .output_dtype = output_dtype, .causal = causal,
        .mask_mode = mask_mode, .failed = failed,
    };
    vx_kernels_parallel_for((int)rows, 1, vx_qsdpa_parallel_worker, &context);
    for (uint64_t row = 0; row < rows; row++) {
        if (failed[row]) {
            result = 0;
            break;
        }
    }
    free(failed);
    return result;
}
