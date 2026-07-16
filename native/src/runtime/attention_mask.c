#include "attention_mask.h"

/* Binary attention masks use conventional I32 storage and nonzero=allowed.
 * Rank-2 [B,K] wins the otherwise ambiguous B==Q case; callers that need an
 * explicit query mask can always use [B,Q,K]. */
int attention_mask_mode(const T* mask, int batch, int seq_q, int seq_kv) {
    if (!mask) return ATTN_MASK_NONE;
    if (mask->dtype != T_I32 || !mask->data || batch <= 0 || seq_q <= 0 || seq_kv <= 0)
        return -1;
    if (mask->ndim == 1 && mask->shape[0] == seq_kv)
        return ATTN_MASK_KEY;
    if (mask->ndim == 2 && mask->shape[1] == seq_kv) {
        if (mask->shape[0] == batch) return ATTN_MASK_BATCH_KEY;
        if (mask->shape[0] == seq_q) return ATTN_MASK_QUERY_KEY;
    }
    if (mask->ndim == 3 && mask->shape[0] == batch &&
        mask->shape[1] == seq_q && mask->shape[2] == seq_kv)
        return ATTN_MASK_BATCH_QUERY_KEY;
    return -1;
}

int attention_mask_allowed(const T* mask, int mode, int batch_index,
                           int query, int key, int seq_q, int seq_kv,
                           int causal) {
    if (causal && key > query) return 0;
    if (!mask || mode == ATTN_MASK_NONE) return 1;
    const int32_t* values = (const int32_t*)mask->data;
    long index;
    if (mode == ATTN_MASK_KEY) index = key;
    else if (mode == ATTN_MASK_BATCH_KEY) index = (long)batch_index * seq_kv + key;
    else if (mode == ATTN_MASK_QUERY_KEY) index = (long)query * seq_kv + key;
    else if (mode == ATTN_MASK_BATCH_QUERY_KEY)
        index = ((long)batch_index * seq_q + query) * seq_kv + key;
    else return 0;
    return values[index] != 0;
}

const int32_t* attention_mask_batch_view(const T* mask, int mode,
                                         int batch_index, int seq_q,
                                         int seq_kv, int* local_mode) {
    if (local_mode) *local_mode = 0;
    if (!mask || mode == ATTN_MASK_NONE) return NULL;
    const int32_t* values = (const int32_t*)mask->data;
    if (mode == ATTN_MASK_KEY) {
        if (local_mode) *local_mode = 1;
        return values;
    }
    if (mode == ATTN_MASK_BATCH_KEY) {
        if (local_mode) *local_mode = 1;
        return values + (long)batch_index * seq_kv;
    }
    if (mode == ATTN_MASK_QUERY_KEY) {
        if (local_mode) *local_mode = 2;
        return values;
    }
    if (mode == ATTN_MASK_BATCH_QUERY_KEY) {
        if (local_mode) *local_mode = 2;
        return values + (long)batch_index * seq_q * seq_kv;
    }
    return NULL;
}
