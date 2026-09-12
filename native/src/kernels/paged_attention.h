#ifndef VOLVOX_KERNELS_PAGED_ATTENTION_H
#define VOLVOX_KERNELS_PAGED_ATTENTION_H

/*
 * One query row against a paged K/V prefix.
 *
 * The native decode path used to spell the visible prefix as a pointer into a
 * contiguous cache plus a length, which assumes the address of token j is a
 * linear function of j.  That is exactly the assumption a page table removes.
 *
 * `page_table == NULL` *is* the linear layout: token j at `j * d_model`.  It is
 * not a legacy branch kept beside a new one — it is what the paged form
 * computes for the identity mapping, so contiguous and paged KV are numerically
 * identical by construction.
 *
 * Dependency-free on purpose: it links into the runtime, the WASM profile and a
 * standalone consumers with nothing but the C library.
 */

/*
 * q          : `[d_model]`, the query row.
 * key_pool   : token slot `s` at `s * d_model`; the whole physical page pool.
 * value_pool : same layout as key_pool.
 * page_table : `[ceil(kv_length / page_tokens)]` logical -> physical page, or
 *              NULL for the identity mapping.  A negative entry is refused.
 * keep_mask  : `[kv_length]` over *logical* positions, or NULL for none.  A
 *              zero entry removes that key.  Logical, not physical: the mask a
 *              decoder produces is indexed by token position, and pairing it
 *              with a physical slot is the mistake paging invites.
 * out        : `[d_model]`.
 *
 * Returns 0, or -1 on an invalid argument, an unmapped page, an entirely
 * masked-out head, or allocation failure.  No output is written on failure.
 */
int vx_sdpa_paged_row_f32(const float* q,
                          const float* key_pool, const float* value_pool,
                          const int* page_table, int page_tokens,
                          int kv_length, const int* keep_mask, float* out,
                          int d_model, int heads, int head_dim, float scale);

/* Physical token slot backing logical `position`, or -1 when unmapped. */
long vx_paged_attention_slot(const int* page_table, int page_tokens, int position);

#endif
