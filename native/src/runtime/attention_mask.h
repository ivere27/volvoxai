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

/*
 * The `mask_mode` a *kernel* takes, which is not `ATTN_MASK_*`.
 *
 * A kernel is handed a view already narrowed to one batch, so it knows only
 * whether the pointer it received is indexed by key alone or by (query, key).
 * The two encodings collide where it matters most: `2` is BATCH_KEY in one and
 * QUERY_KEY in the other, so a value passed between them under the wrong name
 * masks the wrong keys rather than failing.
 */
enum {
    VX_ATTN_MASK_LOCAL_NONE = 0,
    VX_ATTN_MASK_LOCAL_KEY = 1,
    VX_ATTN_MASK_LOCAL_QUERY_KEY = 2
};

/*
 * The layout of `mask`, classified against the operand's declared key extent.
 *
 * `seq_kv` is that extent and nothing else -- notably not a paged cache's lane
 * token capacity, which is tempting because a paged mask's *indices* are
 * lane-local logical tokens rather than pool slots. The axis size and the index
 * meaning are separate questions: the graph declares one shape and the same
 * graph is executed both with a cache bound and without one (a seed forward
 * binds nothing), so an extent only a bound cache can supply would make the
 * unbound execution of that graph invalid. The lane's own length bounds how far
 * into the axis a row may read, and the row paths check it.
 */
int attention_mask_mode(const T* mask, int batch, int seq_q, int seq_kv);
int attention_mask_allowed(const T* mask, int mode, int batch_index,
                           int query, int key, int seq_q, int seq_kv,
                           int causal);
/* One batch's view.  `local_mode` is `VX_ATTN_MASK_LOCAL_*`. */
const int32_t* attention_mask_batch_view(const T* mask, int mode,
                                         int batch_index, int seq_q,
                                         int seq_kv, int* local_mode);
/*
 * One (batch, query) pair's key row, so every layout reaches a kernel as a
 * key-only mask.
 *
 * This is what lets a row decode carry a query-dependent mask at all. Row
 * execution hands the kernel a single query and the kernel then indexes the
 * mask with the *sliced* query index, which is zero -- so a `[Sq,K]` mask would
 * be read at row zero for every row. Resolving the row here, against the query's
 * absolute position, keeps that indexing where the absolute position is still
 * known.  `local_mode` comes back as `VX_ATTN_MASK_LOCAL_KEY`.
 */
const int32_t* attention_mask_query_view(const T* mask, int mode,
                                         int batch_index, int query,
                                         int seq_q, int seq_kv,
                                         int* local_mode);

#endif
