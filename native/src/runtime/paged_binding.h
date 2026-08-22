#ifndef VOLVOX_RUNTIME_PAGED_BINDING_H
#define VOLVOX_RUNTIME_PAGED_BINDING_H

/*
 * The seam between a `VxPagedKVCache` and the native row executor.
 *
 * `paged_kv.c` is deliberately dependency-free — it knows pages, not tensors.
 * This file is where a page table becomes an address: it binds one lane of one
 * cache to one context and answers, for a given tensor and logical token row,
 * which physical row the executor should touch.
 *
 * The answer is `logical` for every tensor outside the page pool, which is
 * every tensor except the attention K/V operands.  That is what lets the
 * conversion be a single substitution at each row-offset site rather than a
 * second code path: `row` becomes `vx_paged_row(tensor, row)` and a context
 * with no binding is unchanged.
 *
 * The paged set is narrow on purpose, exactly as it is in TypeScript
 * (`assertPagedTensorDomain`).  Every operator in the native row path computes
 * its offsets as `row * width`, which is correct for a tensor whose logical
 * order is its physical order — so an operator that touches a paged tensor
 * without going through this file reads the wrong slot and produces a decoder
 * that is wrong and looks plausible.  `vx_paged_domain_supported` refuses those
 * rather than guessing.
 */

#include "engine_internal.h"
#include "decode_row_set.h"
#include "paged_kv.h"

/* `VxPagedKVCache` is opaque; runtime_state.h stores it as a forward
 * declaration so the engine state header needs no allocator internals. */

/*
 * Bind `lane` of `cache` to the current context.
 *
 * `paged_names` lists the activation tensors that live in the page pool; they
 * are the attention `k` and `v` operands and nothing else.  Passing a NULL
 * cache clears the binding, which restores the plain contiguous addressing.
 * The cache is borrowed, not owned: the caller outlives the binding.
 */
int vx_paged_bind_locked(VxPagedKVCache* cache, int lane,
                         const char* const* paged_names, int paged_count);
/* Also runs at engine shutdown: a binding names tensors, and shutdown frees
 * them, so one that survived would resolve against the next model. */
void vx_paged_unbind_locked(void);

/* Whether a page table is bound to this context. */
int vx_paged_bound_locked(void);

/* Whether this tensor's rows are addressed through the page table. */
int vx_paged_tensor_locked(const T* tensor);

/*
 * Physical row for a tensor's logical token row.
 *
 * Identity for every tensor outside the pool.  Returns -1 when a paged
 * tensor's logical row has no resident page, which every caller must treat as
 * a failure rather than as row zero.
 */
int vx_paged_row_locked(const T* tensor, int logical_row);

/*
 * The bound lane's active prefix of `pool`, in logical token order.
 *
 * Returns a pointer into context-owned scratch that stays valid until the next
 * gather of the same `slot` (0 for K, 1 for V), or NULL when a logical position
 * has no resident page or maps outside `pool_rows`.  The cache's page budget
 * and the tensor backing the pool are configured separately, so a mapping that
 * runs past the tensor is a caller error this must catch rather than read.
 *
 * A gather rather than a paged kernel, deliberately.  The native attention
 * kernel runs an online-softmax recurrence whose accumulation order a
 * page-table-aware rewrite would not reproduce, and "paged KV is numerically
 * identical to contiguous KV" would then hold only to a tolerance.  Handing the
 * *same* kernel the same values in the same order makes it bit-identical by
 * construction.  TypeScript and WebGPU pay the same copy for the same reason.
 */
const float* vx_paged_gather_f32_locked(const float* pool, int pool_rows,
                                        int width, int kv_length, int slot);

/*
 * The lane forms of the queries above.
 *
 * A binding names one lane because a scalar decode has one. A declared batch
 * has `g_decode_lanes` of them, and they are the slots the scheduler assigned:
 * batch lane `l` is cache lane `l`. That identity is stated here rather than
 * derived, because neither side can check it -- the batch knows a lane count
 * and the cache knows a lane count, and agreeing on the number says nothing
 * about agreeing on the order.
 *
 * The gather scratch is per slot, not per lane, so a lane's prefix is valid
 * only until the next gather of the same slot. Every caller consumes one lane
 * before starting the next, which is what makes two buffers enough for any
 * lane count.
 */
int vx_paged_lanes_locked(void);
int vx_paged_kv_length_lane_locked(int lane);
/*
 * This lane's mapping in the row set's own spelling.
 *
 * The batch entry point hands these to `vx_decode_row_set_init` so that the row
 * a paged operand writes comes out of `vx_decode_row_set_write_rows`, the
 * implementation TypeScript and WebGPU share a corpus with -- rather than out
 * of a second copy of `page * page_tokens + token % page_tokens` living here.
 * Returns 1 for a paged lane, 0 when nothing is bound, -1 on a bad lane.
 */
int vx_paged_lane_pages_locked(int lane, VxDecodeLanePages* out);
int vx_paged_row_lane_locked(const T* tensor, int logical_row, int lane);
const float* vx_paged_gather_lane_f32_locked(const float* pool, int pool_rows,
                                             int width, int kv_length,
                                             int slot, int lane);
/*
 * The same gather in bytes, which is what a W8A8 K/V pool needs.
 *
 * A gathered row is a run of bytes; the element type only ever decided how wide
 * one is. Keeping the F32 form as a caller of this one rather than a second
 * loop is what stops the two dtypes' page arithmetic from drifting -- the same
 * argument the F32 gather makes for not writing a page-table-aware kernel.
 */
const void* vx_paged_gather_lane_bytes_locked(const void* pool, int pool_rows,
                                              size_t row_bytes, int kv_length,
                                              int slot, int lane);

/* Active key/value length of the bound lane; -1 when nothing is bound. */
int vx_paged_kv_length_locked(void);
/* Lane-local logical -> physical page mapping, or NULL. */
const int* vx_paged_page_table_locked(void);
int vx_paged_page_tokens_locked(void);

/*
 * Whether every node in the current dirty closure keeps the paged tensors
 * inside their proven domain: read only as an attention `k`/`v` operand,
 * written by at most one producing node.
 *
 * Narrowing the paged set is the fix when this refuses, not widening the rule:
 * every additional paged tensor is another operator that must learn page-table
 * addressing.
 */
int vx_paged_domain_supported_locked(const unsigned char* selected_nodes);

#endif
