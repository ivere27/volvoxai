#ifndef VOLVOX_RUNTIME_ATTENTION_MASK_H
#define VOLVOX_RUNTIME_ATTENTION_MASK_H

/*
 * Private attention-mask shape and indexing helpers shared by native forward
 * execution and the guarded training implementation.  T remains runtime
 * owned, so this header is intentionally not part of the installed C ABI.
 */
#include "engine_internal.h"

enum {
    ATTN_MASK_NONE = 0,
    ATTN_MASK_KEY = 1,
    ATTN_MASK_BATCH_KEY = 2,
    ATTN_MASK_QUERY_KEY = 3,
    ATTN_MASK_BATCH_QUERY_KEY = 4
};

int attention_mask_mode(const T* mask, int batch, int seq_q, int seq_kv);
int attention_mask_allowed(const T* mask, int mode, int batch_index,
                           int query, int key, int seq_q, int seq_kv,
                           int causal);
const int32_t* attention_mask_batch_view(const T* mask, int mode,
                                         int batch_index, int seq_q,
                                         int seq_kv, int* local_mode);

#endif
