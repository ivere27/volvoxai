#ifndef VOLVOX_RUNTIME_PAGED_KV_H
#define VOLVOX_RUNTIME_PAGED_KV_H

/*
 * Paged (block) KV cache — native twin of `ts/core/PagedKVCache.ts`.
 *
 * The two implementations are driven by the same golden corpus
 * (`tests/paged_kv_vectors.json`), so an allocator that behaves differently
 * here fails a test instead of producing a decoder that works in the browser
 * and not on the robot.  That is the safety net the decode/row contract never
 * had: shape inference has one, the retained-row proofs have none, and this
 * surface is about to grow page tables and per-lane lengths.
 *
 * Deliberately free of engine dependencies.  It compiles into the runtime, the
 * WASM profile, and a standalone vector test with nothing but the C library —
 * KV ownership must not require the tensor table to exist.
 *
 * Three properties, matching the TypeScript reference clause for clause:
 *
 *   1. Active length is a value (`kv_length[lane]`), never a tensor shape.
 *   2. Contiguous KV is the allocation policy whose page table is the
 *      identity map, not a second addressing path.
 *   3. Every allocation is transactional: reserve, then commit or roll back to
 *      the exact prior page table.
 */

#include <stddef.h>
#include <stdint.h>

typedef enum {
    VX_PAGED_KV_OK = 0,
    VX_PAGED_KV_INVALID_ARGUMENT = -1,
    VX_PAGED_KV_CAPACITY_EXHAUSTED = -2,
    VX_PAGED_KV_PREFIX_NOT_FOUND = -3,
    VX_PAGED_KV_PREFIX_CONFLICT = -4,
    VX_PAGED_KV_STALE_PAGE = -5
} VxPagedKVStatus;

typedef enum {
    VX_PAGED_KV_POLICY_PAGED = 0,
    VX_PAGED_KV_POLICY_CONTIGUOUS = 1
} VxPagedKVPolicy;

/* An unmapped logical page. */
#define VX_PAGED_KV_UNMAPPED (-1)

/* Longest shared-prefix identity the cache stores inline. */
#define VX_PAGED_KV_PREFIX_KEY_MAX 128

typedef struct {
    int lanes;
    int page_tokens;
    int lane_token_capacity;
    /* Zero means "the fully private lanes * pages_per_lane", which is exactly
     * the contiguous high-water mark. */
    int max_pages;
    /* Zero means one; telemetry only, never addressing. */
    int bytes_per_token;
    VxPagedKVPolicy policy;
    /* Nonzero clears a page's bytes when it is recycled.  Generation tags
     * already stop a stale handle from addressing a reused page; the clear
     * bounds the damage if a runtime ever bypasses the tag. */
    int clear_on_recycle;
} VxPagedKVOptions;

typedef struct {
    long logical_bytes;
    long resident_bytes;
    long reserved_bytes;
    int resident_pages;
    int reserved_pages;
    int free_pages;
    int shared_pages;
    long fragmentation_bytes;
    int high_water_pages;
    int prefix_hits;
    int prefix_misses;
    int copy_on_writes;
    int evictions;
    int allocation_failures;
} VxPagedKVTelemetry;

typedef struct VxPagedKVCache VxPagedKVCache;

/* Open, uncommitted reservation.  Owned by the caller's stack frame.
 *
 * A reservation settles exactly once.  Committing and then rolling back the
 * same handle would double-subtract the reserved-page count and republish the
 * pre-commit length — silently, and long after the caller's mistake — so the
 * second transition is refused. */
typedef struct {
    VxPagedKVCache* cache;
    int lane;
    int tokens;
    int prior_length;
    /*
     * Identity of this exact open transition.
     *
     * Only one reservation may be open for a lane.  The id prevents a copied
     * or otherwise stale handle from settling a later reservation that happens
     * to have the same lane and prior length; the cache pointer prevents a
     * handle from settling a lookalike reservation owned by another cache, and
     * lane_generation prevents it from crossing retirement of its lane.
     */
    uint64_t id;
    int lane_generation;
    int page_count;
    int* pages;
    int* logical_pages;
    int capacity;
    int open;
} VxPagedKVReservation;

/* The cache has no internal lock.  Its runtime owner must serialize mutating
 * calls; reservation identity protects ordering and stale handles, not data
 * races between independent scheduler threads. */
VxPagedKVCache* vx_paged_kv_create(const VxPagedKVOptions* options);
void vx_paged_kv_destroy(VxPagedKVCache* cache);

/* Install the backend's page eraser used when a page is recycled. */
void vx_paged_kv_set_page_eraser(VxPagedKVCache* cache,
                                 void (*erase)(int page, void* user),
                                 void* user);

int vx_paged_kv_lanes(const VxPagedKVCache* cache);
int vx_paged_kv_page_tokens(const VxPagedKVCache* cache);
int vx_paged_kv_pages_per_lane(const VxPagedKVCache* cache);
int vx_paged_kv_lane_token_capacity(const VxPagedKVCache* cache);
int vx_paged_kv_max_pages(const VxPagedKVCache* cache);

/* `I32[B]` active lengths.  Read by kernels; never an input to shape
 * inference, which is what keeps a mixed-length batch from becoming a public
 * ragged tensor. */
const int* vx_paged_kv_lengths(const VxPagedKVCache* cache);
const int* vx_paged_kv_query_lengths(const VxPagedKVCache* cache);
const int* vx_paged_kv_lane_generations(const VxPagedKVCache* cache);
/* `I32[B, pages_per_lane]` logical -> physical page, VX_PAGED_KV_UNMAPPED. */
const int* vx_paged_kv_page_table(const VxPagedKVCache* cache);

/* Physical page of logical page `logical` of `lane` under the identity map.
 * The single expression that makes contiguous a policy, not a second path. */
int vx_paged_kv_identity_physical_page(const VxPagedKVCache* cache,
                                       int lane, int logical);
int vx_paged_kv_lane_is_contiguous(const VxPagedKVCache* cache, int lane);
int vx_paged_kv_physical_page(const VxPagedKVCache* cache, int lane, int logical);
/* Flat physical token index; under the identity map this reduces to
 * `lane * lane_token_capacity + position`. */
VxPagedKVStatus vx_paged_kv_physical_token_index(const VxPagedKVCache* cache,
                                                 int lane, int position,
                                                 long* index_out);
/* Physical token index of every active key position of `lane`, in order.
 * `out` must hold at least `kv_length[lane]` entries. */
VxPagedKVStatus vx_paged_kv_gather_active_tokens(const VxPagedKVCache* cache,
                                                 int lane, long* out);

VxPagedKVStatus vx_paged_kv_reserve(VxPagedKVCache* cache, int lane, int tokens,
                                    VxPagedKVReservation* reservation);
VxPagedKVStatus vx_paged_kv_commit(VxPagedKVCache* cache,
                                   VxPagedKVReservation* reservation);
VxPagedKVStatus vx_paged_kv_rollback(VxPagedKVCache* cache,
                                     VxPagedKVReservation* reservation);
/* Settle several independently reserved lanes as one cache transaction.  The
 * whole handle set is validated before any length, reservation identity, or
 * page ownership is mutated.  Lanes must be distinct. */
VxPagedKVStatus vx_paged_kv_commit_batch(
    VxPagedKVCache* cache,
    VxPagedKVReservation* reservations,
    size_t count);
VxPagedKVStatus vx_paged_kv_rollback_batch(
    VxPagedKVCache* cache,
    VxPagedKVReservation* reservations,
    size_t count);
/* Release a reservation's scratch arrays without touching the cache.  Commit
 * and rollback already do this; call dispose for an unused or failed output.
 * An open reservation must be settled first, because disposing it alone does
 * not settle the cache-side transition. */
void vx_paged_kv_reservation_dispose(VxPagedKVReservation* reservation);
VxPagedKVStatus vx_paged_kv_append(VxPagedKVCache* cache, int lane, int tokens);

VxPagedKVStatus vx_paged_kv_release_lane(VxPagedKVCache* cache, int lane);
/* Reset is an atomic no-op while any lane has an open reservation. */
void vx_paged_kv_reset(VxPagedKVCache* cache);

VxPagedKVStatus vx_paged_kv_publish_prefix(VxPagedKVCache* cache, const char* key,
                                           int lane, int tokens);
int vx_paged_kv_has_prefix(const VxPagedKVCache* cache, const char* key);
VxPagedKVStatus vx_paged_kv_acquire_prefix(VxPagedKVCache* cache, const char* key,
                                           int lane, int* tokens_out);
int vx_paged_kv_page_is_shared(const VxPagedKVCache* cache, int lane, int logical);
/* Move `lane`'s mapping of `logical` to a private page.  The caller copies the
 * bytes; `*from_out` and `*to_out` are equal when the page was already
 * private and no copy is needed. */
VxPagedKVStatus vx_paged_kv_copy_on_write(VxPagedKVCache* cache, int lane,
                                          int logical, int* from_out, int* to_out);
/* Evict unreferenced published prefixes, least recently used first, until at
 * least `pages` are free.  Returns the number of pages reclaimed. */
int vx_paged_kv_evict(VxPagedKVCache* cache, int pages);

void vx_paged_kv_telemetry(const VxPagedKVCache* cache, VxPagedKVTelemetry* out);
int vx_paged_kv_page_generation(const VxPagedKVCache* cache, int page);

#endif
