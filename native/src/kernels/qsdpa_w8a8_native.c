/*
 * Native CPU parallelism for canonical W8A8 QSDPA.
 *
 * The portable qsdpa_i8u8() kernel remains the authoritative implementation
 * and the sole WASM path.  Whole-tensor native execution can partition its
 * independent [batch, query] rows, then invoke that same kernel for each row.
 * Causal rows expose exactly their original K/V prefix and query-dependent
 * masks are reduced to the corresponding key row, so key iteration and every
 * head's arithmetic order remain unchanged.
 */
#include "quant_cpu_isa.h"
#include "fast_exp.h"
#include "w8a8_affine.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include "../../include/volvoxai_enums.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#define VX_QSDPA_X86_AVX2 1
#include <immintrin.h>
#define VX_QSDPA_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_QSDPA_X86_AVX2 0
#define VX_QSDPA_TARGET_AVX2
#endif

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
    uint32_t query_start;
    uint32_t query_count;
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
    int use_avx2;
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

static int32_t vx_qsdpa_mask_value(const int32_t* mask, uint32_t key) {
    const unsigned char* bytes = (const unsigned char*)mask +
        (size_t)key * sizeof(*mask);
    const uint32_t bits = (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8u) | ((uint32_t)bytes[2] << 16u) |
        ((uint32_t)bytes[3] << 24u);
    return (int32_t)bits;
}

#if VX_QSDPA_X86_AVX2
/* The runtime validator admits only head_dim <= 64 with head_dim % 4 == 0, so a
 * centered query row fits in four 16-lane I16 registers plus one 8-lane tail. */
enum { VX_QSDPA_MAX_HEAD_DIM = 64u };

typedef struct {
    __m256i blocks[VX_QSDPA_MAX_HEAD_DIM / 16u];
    __m128i tail8;
    int32_t scalar[4];
    uint32_t block_count;
    uint32_t has_tail8;
    uint32_t scalar_begin;
    uint32_t scalar_count;
    int32_t query_sum;
} VxQSDPACenteredQuery;

static inline __attribute__((always_inline)) VX_QSDPA_TARGET_AVX2 int32_t vx_qsdpa_reduce_i32_avx2(__m256i sums) {
    __m128i folded = _mm_add_epi32(_mm256_castsi256_si128(sums),
                                   _mm256_extracti128_si256(sums, 1));
    folded = _mm_add_epi32(folded,
        _mm_shuffle_epi32(folded, _MM_SHUFFLE(1, 0, 3, 2)));
    folded = _mm_add_epi32(folded,
        _mm_shuffle_epi32(folded, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(folded);
}

/* The query row is invariant across every key in the row, so centering it once
 * removes one load, one widening convert, and one subtract per key per block
 * from the hot loop.  Values are identical to the per-key spelling. */
static VX_QSDPA_TARGET_AVX2 void vx_qsdpa_center_query_avx2(
        VxQSDPACenteredQuery* centered, const unsigned char* q,
        uint32_t dimensions, uint32_t q_dtype, int32_t q_zero_point) {
    const __m256i q_zero = _mm256_set1_epi16((int16_t)q_zero_point);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i sums = _mm256_setzero_si256();
    uint32_t dimension = 0;
    centered->block_count = 0;
    centered->has_tail8 = 0;
    centered->scalar_count = 0;
    for (; dimension + 16u <= dimensions; dimension += 16u) {
        const __m128i q8 = _mm_loadu_si128(
            (const __m128i*)(const void*)(q + dimension));
        const __m256i q16 = q_dtype == VX_DTYPE_I8
            ? _mm256_cvtepi8_epi16(q8) : _mm256_cvtepu8_epi16(q8);
        const __m256i q_centered = _mm256_sub_epi16(q16, q_zero);
        centered->blocks[centered->block_count++] = q_centered;
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(q_centered, ones));
    }
    if (dimension + 8u <= dimensions) {
        const __m128i q8 = _mm_loadl_epi64(
            (const __m128i*)(const void*)(q + dimension));
        const __m128i q16 = q_dtype == VX_DTYPE_I8
            ? _mm_cvtepi8_epi16(q8) : _mm_cvtepu8_epi16(q8);
        const __m128i q_centered = _mm_sub_epi16(q16,
            _mm_set1_epi16((int16_t)q_zero_point));
        centered->tail8 = q_centered;
        centered->has_tail8 = 1;
        sums = _mm256_add_epi32(sums, _mm256_inserti128_si256(
            _mm256_setzero_si256(), _mm_madd_epi16(q_centered, _mm_set1_epi16(1)), 0));
        dimension += 8u;
    }
    centered->scalar_begin = dimension;
    centered->query_sum = vx_qsdpa_reduce_i32_avx2(sums);
    for (; dimension < dimensions; dimension++) {
        const int32_t q_value = vx_w8a8_byte_value(q, q_dtype, dimension) - q_zero_point;
        centered->scalar[centered->scalar_count++] = q_value;
        centered->query_sum += q_value;
    }
}

/* Integer accumulation is associative and the validator has already proven the
 * worst-case centered magnitude fits I32, so every partial sum is exact and any
 * reduction order yields the same value.  A register reduction avoids the
 * store-to-load forwarding stall the previous spill-and-add sequence paid on
 * every single key. */
static inline __attribute__((always_inline)) VX_QSDPA_TARGET_AVX2 int32_t vx_qsdpa_centered_dot_avx2(
        const VxQSDPACenteredQuery* centered, const unsigned char* k,
        uint32_t k_dtype) {
    __m256i sums = _mm256_setzero_si256();
    int32_t result;
    for (uint32_t block = 0; block < centered->block_count; block++) {
        const __m128i k8 = _mm_loadu_si128(
            (const __m128i*)(const void*)(k + block * 16u));
        __m256i k16 = k_dtype == VX_DTYPE_I8
            ? _mm256_cvtepi8_epi16(k8) : _mm256_cvtepu8_epi16(k8);
        sums = _mm256_add_epi32(sums,
            _mm256_madd_epi16(centered->blocks[block], k16));
    }
    if (centered->has_tail8) {
        const __m128i k8 = _mm_loadl_epi64(
            (const __m128i*)(const void*)(k + centered->block_count * 16u));
        __m128i k16 = k_dtype == VX_DTYPE_I8
            ? _mm_cvtepi8_epi16(k8) : _mm_cvtepu8_epi16(k8);
        sums = _mm256_add_epi32(sums, _mm256_inserti128_si256(
            _mm256_setzero_si256(), _mm_madd_epi16(centered->tail8, k16), 0));
    }
    result = vx_qsdpa_reduce_i32_avx2(sums);
    for (uint32_t index = 0; index < centered->scalar_count; index++) {
        const int32_t k_value = vx_w8a8_byte_value(
            k, k_dtype, centered->scalar_begin + index);
        result += centered->scalar[index] * k_value;
    }
    return result;
}

static VX_QSDPA_TARGET_AVX2 __m256 vx_qsdpa_centered_v8_f32_avx2(
        const unsigned char* values, uint32_t dtype,
        int32_t zero_point) {
    const __m128i bytes = _mm_loadl_epi64(
        (const __m128i*)(const void*)values);
    __m256i integers = dtype == VX_DTYPE_I8
        ? _mm256_cvtepi8_epi32(bytes) : _mm256_cvtepu8_epi32(bytes);
    integers = _mm256_sub_epi32(
        integers, _mm256_set1_epi32(zero_point));
    return _mm256_cvtepi32_ps(integers);
}

/* A validated row keeps the portable online-softmax recurrence and F32
 * operation order.  Only the exact centered-byte QK dot is narrowed to I16
 * and reduced with AVX2, cutting the hot attention inner loop in half. */
static VX_QSDPA_TARGET_AVX2 void vx_qsdpa_row_avx2(
        const VxQSDPAParallelContext* context, const unsigned char* q,
        const unsigned char* k, const unsigned char* v,
        const int32_t* mask, unsigned char* output, uint32_t seq_kv) {
    const uint32_t head_dim = context->d_model / context->heads;
    const float qk_scale = context->q_scale * context->k_scale;
    const float score_scale = qk_scale * context->attention_scale;
    const __m256 v_scales = _mm256_set1_ps(context->v_scale);
    const int32_t output_minimum = context->output_dtype == VX_DTYPE_I8
        ? -128 : 0;
    const int32_t output_maximum = context->output_dtype == VX_DTYPE_I8
        ? 127 : 255;
    for (uint32_t head = 0; head < context->heads; head++) {
        const uint32_t head_base = head * head_dim;
        float accumulator[VX_QSDPA_MAX_HEAD_DIM] = {0.0f};
        float maximum_score = 0.0f;
        float sum = 0.0f;
        int have_key = 0;
        VxQSDPACenteredQuery centered_query;
        int32_t query_sum_k_zp;
        vx_qsdpa_center_query_avx2(&centered_query, q + head_base, head_dim,
                                   context->q_dtype, context->q_zero_point);
        query_sum_k_zp = centered_query.query_sum * context->k_zero_point;
        for (uint32_t key = 0; key < seq_kv; key++) {
            const size_t kv_base = (size_t)key * context->d_model + head_base;
            float score;
            float weight;
            int32_t dot;
            if (mask && vx_qsdpa_mask_value(mask, key) == 0) continue;
            dot = vx_qsdpa_centered_dot_avx2(&centered_query, k + kv_base,
                context->k_dtype);
            dot -= query_sum_k_zp;
            score = (float)dot * score_scale;
            if (!have_key) {
                uint32_t dimension = 0;
                maximum_score = score;
                sum = 1.0f;
                for (; dimension + 8u <= head_dim; dimension += 8u) {
                    const __m256 values = vx_qsdpa_centered_v8_f32_avx2(
                        v + kv_base + dimension, context->v_dtype,
                        context->v_zero_point);
                    _mm256_storeu_ps(accumulator + dimension,
                        _mm256_mul_ps(values, v_scales));
                }
                for (; dimension < head_dim; dimension++) {
                    const int32_t value = vx_w8a8_byte_value(v,
                        context->v_dtype, kv_base + dimension) -
                        context->v_zero_point;
                    accumulator[dimension] =
                        (float)value * context->v_scale;
                }
                have_key = 1;
                continue;
            }
            if (score > maximum_score) {
                uint32_t dimension = 0;
                const __m256 weights = _mm256_set1_ps(weight =
                    accurate_expf(maximum_score - score));
                sum = sum * weight + 1.0f;
                for (; dimension + 8u <= head_dim; dimension += 8u) {
                    const __m256 values = vx_qsdpa_centered_v8_f32_avx2(
                        v + kv_base + dimension, context->v_dtype,
                        context->v_zero_point);
                    const __m256 previous = _mm256_loadu_ps(
                        accumulator + dimension);
                    const __m256 retained = _mm256_mul_ps(previous, weights);
                    const __m256 incoming = _mm256_mul_ps(values, v_scales);
                    _mm256_storeu_ps(accumulator + dimension,
                        _mm256_add_ps(retained, incoming));
                }
                for (; dimension < head_dim; dimension++) {
                    const int32_t value = vx_w8a8_byte_value(v,
                        context->v_dtype, kv_base + dimension) -
                        context->v_zero_point;
                    accumulator[dimension] = accumulator[dimension] * weight +
                        (float)value * context->v_scale;
                }
                maximum_score = score;
            } else {
                uint32_t dimension = 0;
                const __m256 weights = _mm256_set1_ps(weight =
                    accurate_expf(score - maximum_score));
                sum += weight;
                for (; dimension + 8u <= head_dim; dimension += 8u) {
                    const __m256 values = vx_qsdpa_centered_v8_f32_avx2(
                        v + kv_base + dimension, context->v_dtype,
                        context->v_zero_point);
                    const __m256 weighted = _mm256_mul_ps(
                        _mm256_mul_ps(weights, values), v_scales);
                    _mm256_storeu_ps(accumulator + dimension,
                        _mm256_add_ps(
                            _mm256_loadu_ps(accumulator + dimension),
                            weighted));
                }
                for (; dimension < head_dim; dimension++) {
                    const int32_t value = vx_w8a8_byte_value(v,
                        context->v_dtype, kv_base + dimension) -
                        context->v_zero_point;
                    accumulator[dimension] +=
                        weight * (float)value * context->v_scale;
                }
            }
        }
        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
            int32_t quantized = context->output_zero_point;
            if (have_key && sum > 0.0f) {
                const float value = accumulator[dimension] / sum;
                const float transformed = value / context->output_scale +
                    (float)context->output_zero_point;
                quantized = vx_w8a8_requantize(transformed, output_minimum,
                    output_maximum, context->output_zero_point);
            }
            vx_w8a8_store_byte(output, context->output_dtype,
                head_base + dimension, quantized);
        }
    }
}
#endif

static void vx_qsdpa_parallel_worker(void* opaque, int begin, int end) {
    VxQSDPAParallelContext* context = (VxQSDPAParallelContext*)opaque;
    for (int task = begin; task < end; task++) {
        const uint32_t batch_index = (uint32_t)task / context->query_count;
        const uint32_t query = context->query_start +
            (uint32_t)task % context->query_count;
        const size_t q_offset = ((size_t)batch_index * context->seq_q + query) *
            context->d_model;
        const size_t kv_offset = (size_t)batch_index * context->seq_kv *
            context->d_model;
        uint32_t row_seq_kv = context->seq_kv;
        const int32_t* row_mask = vx_qsdpa_row_mask(context, batch_index, query);
        if (context->causal && query + 1u < row_seq_kv) row_seq_kv = query + 1u;
#if VX_QSDPA_X86_AVX2
        if (context->use_avx2) {
            vx_qsdpa_row_avx2(context,
                       (const unsigned char*)context->q + q_offset,
                       (const unsigned char*)context->k + kv_offset,
                       (const unsigned char*)context->v + kv_offset,
                       row_mask,
                       (unsigned char*)context->output + q_offset,
                       row_seq_kv);
            if (context->failed) context->failed[task] = 0u;
            continue;
        }
#endif
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

static int vx_qsdpa_i8u8_native_rows_validated(
        const void* q, const void* k, const void* v, const int32_t* mask,
        void* output, uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode, uint32_t query_start,
        uint32_t query_count) {
    VxQSDPAParallelContext context;
    uint64_t rows;
    unsigned char* failed;
    const int use_avx2 = VX_QSDPA_X86_AVX2 && vx_kernel_platform()->has_avx2;

    int result = 1;
    if (!query_count || query_start >= seq_q || query_count > seq_q - query_start)
        return 0;
    rows = (uint64_t)batch * query_count;
    if (!rows || rows > INT_MAX || rows > SIZE_MAX) return 0;
    failed = use_avx2 ? NULL :
        (unsigned char*)calloc((size_t)rows, sizeof(*failed));
    if (!use_avx2 && !failed) return 0;
    context = (VxQSDPAParallelContext){
        .q = q, .k = k, .v = v, .mask = mask, .output = output,
        .batch = batch, .seq_q = seq_q, .seq_kv = seq_kv,
        .query_start = query_start, .query_count = query_count,
        .d_model = d_model, .heads = heads,
        .q_scale = q_scale, .q_zero_point = q_zero_point,
        .k_scale = k_scale, .k_zero_point = k_zero_point,
        .v_scale = v_scale, .v_zero_point = v_zero_point,
        .output_scale = output_scale,
        .output_zero_point = output_zero_point,
        .attention_scale = attention_scale,
        .q_dtype = q_dtype, .k_dtype = k_dtype, .v_dtype = v_dtype,
        .output_dtype = output_dtype, .causal = causal,
        .mask_mode = mask_mode, .use_avx2 = use_avx2, .failed = failed,
    };
    if (vx_qsdpa_parallel_worthwhile(batch, query_count, seq_kv, d_model)) {
        vx_kernels_parallel_for((int)rows, 1,
                                vx_qsdpa_parallel_worker, &context);
    } else {
        vx_qsdpa_parallel_worker(&context, 0, (int)rows);
    }
    if (failed) {
        for (uint64_t row = 0; row < rows; row++) {
            if (failed[row]) {
                result = 0;
                break;
            }
        }
    }
    free(failed);
    return result;
}

int vx_qsdpa_i8u8_native_validated(const void* q, const void* k, const void* v,
        const int32_t* mask, void* output, uint32_t batch, uint32_t seq_q,
        uint32_t seq_kv, uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode) {
    const int use_avx2 = VX_QSDPA_X86_AVX2 && vx_kernel_platform()->has_avx2;

    if (!use_avx2 &&
        !vx_qsdpa_parallel_worthwhile(batch, seq_q, seq_kv, d_model)) {
        return qsdpa_i8u8(q, k, v, mask, output, batch, seq_q, seq_kv,
                          d_model, heads, q_scale, q_zero_point, k_scale,
                          k_zero_point, v_scale, v_zero_point, output_scale,
                          output_zero_point, attention_scale, q_dtype, k_dtype,
                          v_dtype, output_dtype, causal, mask_mode);
    }
    if (vx_qsdpa_i8u8_native_rows_validated(
            q, k, v, mask, output, batch, seq_q, seq_kv, d_model, heads,
            q_scale, q_zero_point, k_scale, k_zero_point, v_scale,
            v_zero_point, output_scale, output_zero_point, attention_scale,
            q_dtype, k_dtype, v_dtype, output_dtype, causal, mask_mode,
            0u, seq_q)) return 1;
    return qsdpa_i8u8(q, k, v, mask, output, batch, seq_q, seq_kv,
                      d_model, heads, q_scale, q_zero_point, k_scale,
                      k_zero_point, v_scale, v_zero_point, output_scale,
                      output_zero_point, attention_scale, q_dtype, k_dtype,
                      v_dtype, output_dtype, causal, mask_mode);
}

int vx_qsdpa_i8u8_native_range_validated(
        const void* q, const void* k, const void* v, const int32_t* mask,
        void* output, uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode, uint32_t query_start,
        uint32_t query_count) {
    if (query_start == 0u && query_count == seq_q) {
        return vx_qsdpa_i8u8_native_validated(
            q, k, v, mask, output, batch, seq_q, seq_kv, d_model, heads,
            q_scale, q_zero_point, k_scale, k_zero_point, v_scale,
            v_zero_point, output_scale, output_zero_point, attention_scale,
            q_dtype, k_dtype, v_dtype, output_dtype, causal, mask_mode);
    }
    return vx_qsdpa_i8u8_native_rows_validated(
        q, k, v, mask, output, batch, seq_q, seq_kv, d_model, heads,
        q_scale, q_zero_point, k_scale, k_zero_point, v_scale,
        v_zero_point, output_scale, output_zero_point, attention_scale,
        q_dtype, k_dtype, v_dtype, output_dtype, causal, mask_mode,
        query_start, query_count);
}
