#ifndef VOLVOX_RUNTIME_DECODE_ROW_SET_H
#define VOLVOX_RUNTIME_DECODE_ROW_SET_H

/*
 * The rows one decode step writes — native twin of `ts/backends/decodeRowSet.ts`.
 *
 * Row execution on both sides was built around a single scalar position: one
 * logical sequence, one row, and the row is one contiguous span at
 * `position * width`.  B>1 decode breaks exactly that invariant.  With `lanes`
 * sequences retained in one `[B,S,W]` activation the rows a step writes are
 * `b * S + position[b]`, and lanes at different positions make that a *set* of
 * spans; a paged K/V tensor is worse, because its rows are page-table slots that
 * bear no relation to each other at all.
 *
 * Both implementations follow the same explicit per-lane row-addressing
 * contract so per-lane lengths and page tables are not specified twice.
 *
 * Two properties, matching the TypeScript reference clause for clause:
 *
 *   1. A lane's active length is a value (`kv_length[lane]`), never a shape.
 *      The padded key extent every lane's operand shares is `key_capacity =
 *      max(kv_length)`, and the per-lane length reaches the kernel through the
 *      keep mask (design D5 stage one).  Nothing here can produce a ragged
 *      tensor.
 *   2. One lane is not a preserved second path.  It is the tier where the row
 *      indices happen to be consecutive, so the span is one subarray and the
 *      copy disappears.
 *
 * Deliberately free of engine dependencies, like paged_kv.c: it compiles into
 * the runtime, the WASM profile, and standalone consumers with nothing but the
 * C library.
 */

#include "paged_kv.h"

#include <stddef.h>

typedef enum {
    VX_DECODE_ROW_SET_OK = 0,
    VX_DECODE_ROW_SET_INVALID_ARGUMENT = -1,
    /* A logical position with no resident page behind it. */
    VX_DECODE_ROW_SET_UNMAPPED = -2
} VxDecodeRowSetStatus;

/* Span tiers, in the same order and with the same meaning as the TS twin. */
typedef enum {
    VX_DECODE_ROW_SPAN_IDENTITY = 0,
    VX_DECODE_ROW_SPAN_GATHER = 1
} VxDecodeRowSpanTier;

/* The graph keep-mask layouts the row path admits. */
typedef enum {
    VX_DECODE_KEEP_MASK_NONE = 0,
    VX_DECODE_KEEP_MASK_K = 1,
    VX_DECODE_KEEP_MASK_BK = 2,
    VX_DECODE_KEEP_MASK_QK = 3,
    VX_DECODE_KEEP_MASK_BQK = 4
} VxDecodeKeepMaskLayout;

typedef struct {
    VxDecodeKeepMaskLayout layout;
    const int* values;
    int keys;
    int queries;
} VxDecodeKeepMask;

/* One lane's page table, or a null table for an unpaged lane. */
typedef struct {
    /* Lane-local logical -> physical page; VX_PAGED_KV_UNMAPPED is unmapped.
     * NULL means this lane is unpaged. */
    const int* page_table;
    int page_tokens;
    int pages_per_lane;
} VxDecodeLanePages;

typedef struct VxDecodeRowRun VxDecodeRowRun;

typedef struct {
    int lanes;
    /* Logical write position per lane; zero for a parked lane. */
    int* positions;
    /*
     * Which lanes hold no request.
     *
     * A dense `[B,S,D]` batch cannot drop a lane without changing every
     * operand's shape, so a slot whose request retired still occupies a row.
     * It needs no page and no length: the kernel computes its row because the
     * shape says `lanes` rows exist, and the scatter does not write it back.
     * Nothing in the pool is touched, which is what makes a *released* lane —
     * whose pages went back to the free list — expressible at all.
     *
     * Distinct from an *idle* lane, which still holds its request and repeats
     * its own last row so that row stays bit-identical.  A parked lane has no
     * row to repeat.
     */
    int* parked;
    /* Lanes that actually advance.  At least one, or the step is not a step. */
    int live;
    /* I32[B]: the lane's visible key/value length.  Derived as position + 1,
     * because causal decode appends one K/V row per token and a length the row
     * path did not produce would make attention read a slot nobody wrote. */
    int* kv_lengths;
    VxDecodeLanePages* pages;
    /* max(kv_lengths): the shape every lane's attention operand shares. */
    int key_capacity;
    /* Nonzero when no lane carries a page table, which is the contiguous
     * allocation policy rather than a separate addressing path. */
    int unpaged;
    /* Owned by init/dispose. Value copies borrow these arrays for one step. */
    void* storage;
    int* scratch_indices[4];
    VxDecodeRowRun* scratch_runs;
} VxDecodeRowSet;

/*
 * A position of VX_DECODE_ROW_PARKED marks a lane that holds no request.  A
 * negative row cannot be a real position, which is what makes it usable as the
 * sentinel.
 */
#define VX_DECODE_ROW_PARKED (-1)

/*
 * Build the row set for a step.
 *
 * `pages` may be NULL for a wholly unpaged step.  A step that mixes paged and
 * unpaged lanes is refused: a tensor is paged for the whole step or for none of
 * it, and one lane addressing storage through a page table while another
 * addresses it linearly writes two lanes under two different meanings.  At
 * least one lane must not be parked, since a step in which nothing advances is
 * not a step.
 */
VxDecodeRowSetStatus vx_decode_row_set_init(VxDecodeRowSet* set, int lanes,
                                            const int* positions,
                                            const VxDecodeLanePages* pages);

void vx_decode_row_set_dispose(VxDecodeRowSet* set);

/*
 * The row each lane writes this step, in lane order.
 *
 * `paged` is per tensor, not per lane: a lane owns a page table, but only the
 * tensors in the paged set are addressed through it, and applying it to an
 * ordinary activation would send that activation's row into a K/V slot.
 */
VxDecodeRowSetStatus vx_decode_row_set_write_rows(const VxDecodeRowSet* set,
                                                  int lane_stride, int paged,
                                                  int* out_rows);

/*
 * One source row per staged row of a `[lanes, key_capacity, W]` prefix operand,
 * in lane-major logical order, with -1 for a lane's padding beyond its own
 * length.  `out_rows` holds `lanes * key_capacity` entries.
 */
VxDecodeRowSetStatus vx_decode_row_set_prefix_rows(const VxDecodeRowSet* set,
                                                   int lane_stride, int paged,
                                                   int* out_rows);

/*
 * The `[lanes, keys]` keep mask.
 *
 * `clamp_to_length` folds each lane's active length in, which is how a padded
 * batch stays bit-identical to a one-lane decode: every kernel skips a masked
 * key outright rather than scoring it and multiplying by zero, so a lane's
 * visible key set and its accumulation order are exactly what the one-lane
 * decode produces.  Zero for a cross-attention memory, which every lane reads
 * whole and which therefore has no per-lane length to publish.
 */
VxDecodeRowSetStatus vx_decode_row_set_keep_mask(const VxDecodeRowSet* set,
                                                 int keys, int clamp_to_length,
                                                 const VxDecodeKeepMask* mask,
                                                 int* out_mask);

/* The tier a set of row indices falls into, without touching any storage. */
VxDecodeRowSpanTier vx_decode_row_span_tier(const int* rows, int count);

/*
 * One contiguous run of rows shared by a source and a staged destination.
 *
 * `source` is -1 in a padding run, which names destination rows no source
 * fills.
 */
struct VxDecodeRowRun {
    int source;
    int destination;
    int rows;
};

/*
 * `rows` collapsed into the fewest contiguous runs that reproduce it.
 *
 * A staged span is `count` copies in the worst case and one in the best, and
 * which it is depends on the step rather than on the plan: two lanes at
 * adjacent positions of a shared page are one run, and the same two lanes a
 * step later may not be. Collapsing here instead of at each call site is what
 * keeps "how scattered is this step, really?" a number a test can assert.
 *
 * Returns the run count, or VX_DECODE_ROW_SET_INVALID_ARGUMENT when `capacity`
 * cannot hold it. `count` runs always suffice.
 */
int vx_decode_row_copy_runs(const int* rows, int count,
                            VxDecodeRowRun* out, int capacity);

/*
 * The staged destination rows no source fills, as runs — the complement of
 * vx_decode_row_copy_runs over the same array.
 *
 * A backend that stages into a reused buffer has to clear these. The keep mask
 * already stops a kernel from reading a lane's padding, so this is the second
 * of two defences and not the load-bearing one; but padding still holding
 * *another request's* bytes turns a masking mistake from wrong arithmetic into
 * a cross-request leak.
 */
int vx_decode_row_padding_runs(const int* rows, int count,
                               VxDecodeRowRun* out, int capacity);

/*
 * Read `rows` of `storage` into `staged`, in staged order.
 *
 * Padding rows are zeroed rather than left holding whatever the scratch pool
 * last carried, for the reason above. `storage_bytes` bounds every source row:
 * a row index that would read past the end fails the call rather than the
 * process.
 */
VxDecodeRowSetStatus vx_decode_row_gather(void* staged, const void* storage,
                                          size_t storage_bytes, size_t row_bytes,
                                          const int* rows, int count);

/*
 * Write `staged` back to the `rows` of `storage` it came from.
 *
 * `parked` is the row set's parked flags, or NULL when no lane is parked. A
 * parked lane occupies a dense row so the operand keeps its shape and the
 * kernel computed its result because the shape said the row existed; skipping
 * it here is the whole of what parking costs. Padding rows are skipped for the
 * same reason they were zeroed on the way in: they belong to nothing.
 */
VxDecodeRowSetStatus vx_decode_row_scatter(void* storage, size_t storage_bytes,
                                           const void* staged, size_t row_bytes,
                                           const int* rows, int count,
                                           const int* parked);

#endif /* VOLVOX_RUNTIME_DECODE_ROW_SET_H */
