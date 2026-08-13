#include "paged_kv.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*
 * See paged_kv.h for the contract.  This file is the native half of a pair;
 * `ts/core/PagedKVCache.ts` is the other, and `tests/paged_kv_vectors.json`
 * drives both.  Keep the clause order below aligned with the TypeScript file
 * so a reviewer can diff them by eye — the two shape-inference implementations
 * this repository already pays for drifted precisely because nobody could.
 */

typedef struct {
    char key[VX_PAGED_KV_PREFIX_KEY_MAX];
    int* pages;
    int page_count;
    int tokens;
    /* Lanes currently holding this prefix.  Zero means evictable. */
    int lane_references;
    /* Publication order; the eviction tie-break. */
    long sequence;
    long last_use;
    int live;
} VxPagedKVPrefix;

struct VxPagedKVCache {
    int lanes;
    int page_tokens;
    int lane_token_capacity;
    int pages_per_lane;
    int max_pages;
    int bytes_per_token;
    VxPagedKVPolicy policy;
    int clear_on_recycle;

    int* page_table;
    int* kv_length;
    int* query_length;
    int* lane_generation;
    /* Zero, or the unique reservation currently allowed to mutate this lane. */
    uint64_t* lane_reservation_id;
    uint64_t next_reservation_id;
    /* Which shared prefix each lane holds, as an index into `prefixes`, or -1.
     * Tracked explicitly rather than inferred by comparing the lane's page
     * table against the prefix's pages: a lane that copies-on-write no longer
     * matches its own prefix, and the comparison would then leak the
     * reference, pinning the prefix uneviatable for the cache's whole life. */
    int* lane_prefix;

    int* ref_count;
    int* page_generation;
    /* Binary min-heap of free physical pages: lowest index first. */
    int* free_heap;
    int free_count;

    int reserved_pages;
    int high_water_pages;
    long prefix_sequence;
    long clock;

    VxPagedKVPrefix* prefixes;
    int prefix_count;
    int prefix_capacity;

    int prefix_hits;
    int prefix_misses;
    int copy_on_writes;
    int evictions;
    int allocation_failures;

    void (*erase)(int page, void* user);
    void* erase_user;
};

static int lane_valid(const VxPagedKVCache* cache, int lane) {
    return cache && lane >= 0 && lane < cache->lanes;
}

static void heap_swap(int* heap, int a, int b) {
    int value = heap[a];
    heap[a] = heap[b];
    heap[b] = value;
}

static void heap_sift_up(int* heap, int index) {
    while (index > 0) {
        int parent = (index - 1) / 2;
        if (heap[parent] <= heap[index]) break;
        heap_swap(heap, parent, index);
        index = parent;
    }
}

static void heap_sift_down(int* heap, int count, int index) {
    for (;;) {
        int left = index * 2 + 1;
        int right = left + 1;
        int smallest = index;
        if (left < count && heap[left] < heap[smallest]) smallest = left;
        if (right < count && heap[right] < heap[smallest]) smallest = right;
        if (smallest == index) break;
        heap_swap(heap, smallest, index);
        index = smallest;
    }
}

static void free_push(VxPagedKVCache* cache, int page) {
    cache->free_heap[cache->free_count] = page;
    heap_sift_up(cache->free_heap, cache->free_count);
    cache->free_count++;
}

static int free_pop(VxPagedKVCache* cache) {
    int page = cache->free_heap[0];
    cache->free_count--;
    cache->free_heap[0] = cache->free_heap[cache->free_count];
    heap_sift_down(cache->free_heap, cache->free_count, 0);
    return page;
}

/* Remove one specific page from the free heap; contiguous pinning needs it. */
static int free_take(VxPagedKVCache* cache, int page) {
    int index = -1;
    for (int scan = 0; scan < cache->free_count; scan++) {
        if (cache->free_heap[scan] == page) { index = scan; break; }
    }
    if (index < 0) return 0;
    cache->free_count--;
    cache->free_heap[index] = cache->free_heap[cache->free_count];
    /* The replacement may violate the heap in either direction. */
    heap_sift_up(cache->free_heap, index);
    heap_sift_down(cache->free_heap, cache->free_count, index);
    return 1;
}

static void note_high_water(VxPagedKVCache* cache) {
    int resident = 0;
    for (int page = 0; page < cache->max_pages; page++) {
        if (cache->ref_count[page] > 0) resident++;
    }
    if (resident > cache->high_water_pages) cache->high_water_pages = resident;
}

static void recycle_page(VxPagedKVCache* cache, int page) {
    if (cache->clear_on_recycle && cache->erase) cache->erase(page, cache->erase_user);
}

static int allocate_page(VxPagedKVCache* cache, int lane, int logical) {
    int page;
    if (cache->policy == VX_PAGED_KV_POLICY_CONTIGUOUS) {
        page = lane * cache->pages_per_lane + logical;
        if (page < 0 || page >= cache->max_pages || cache->ref_count[page] != 0 ||
            !free_take(cache, page)) {
            cache->allocation_failures++;
            return -1;
        }
        cache->ref_count[page] = 1;
        recycle_page(cache, page);
        return page;
    }
    if (cache->free_count == 0) {
        cache->allocation_failures++;
        return -1;
    }
    page = free_pop(cache);
    cache->ref_count[page] = 1;
    recycle_page(cache, page);
    return page;
}

static VxPagedKVStatus release_page(VxPagedKVCache* cache, int page) {
    if (page < 0 || page >= cache->max_pages || cache->ref_count[page] <= 0) {
        return VX_PAGED_KV_STALE_PAGE;
    }
    cache->ref_count[page]--;
    if (cache->ref_count[page] > 0) return VX_PAGED_KV_OK;
    /* The tag advances only on the transition to unreferenced, so a handle
     * taken while the page was shared stays valid while any holder keeps it. */
    cache->page_generation[page] = cache->page_generation[page] == INT_MAX
        ? 0 : cache->page_generation[page] + 1;
    free_push(cache, page);
    return VX_PAGED_KV_OK;
}

VxPagedKVCache* vx_paged_kv_create(const VxPagedKVOptions* options) {
    VxPagedKVCache* cache;
    size_t pages_per_lane;
    size_t private_pages;
    int max_pages;
    int bytes_per_token;
    if (!options || options->lanes <= 0 || options->page_tokens <= 0 ||
        options->lane_token_capacity <= 0 || options->max_pages < 0 ||
        options->bytes_per_token < 0) return NULL;
    if (options->policy != VX_PAGED_KV_POLICY_PAGED &&
        options->policy != VX_PAGED_KV_POLICY_CONTIGUOUS) return NULL;

    /* Ceil-divide without evaluating capacity + page_tokens - 1 in `int`. */
    pages_per_lane = ((size_t)options->lane_token_capacity - 1u) /
                         (size_t)options->page_tokens +
                     1u;
    if (pages_per_lane > (size_t)INT_MAX ||
        (size_t)options->lanes > (size_t)INT_MAX / pages_per_lane) {
        return NULL;
    }
    private_pages = (size_t)options->lanes * pages_per_lane;
    if (private_pages > SIZE_MAX / sizeof(int)) return NULL;
    max_pages = options->max_pages > 0 ? options->max_pages : (int)private_pages;
    bytes_per_token = options->bytes_per_token > 0 ? options->bytes_per_token : 1;
    if ((size_t)max_pages > SIZE_MAX / sizeof(int)) return NULL;

    /* Physical row indices and every byte counter exposed by telemetry use
     * `long`.  Refuse a configuration whose valid high-water mark cannot be
     * represented instead of overflowing much later in an attention binding. */
    if ((long)max_pages > LONG_MAX / options->page_tokens ||
        (long)max_pages * options->page_tokens > LONG_MAX / bytes_per_token ||
        (long)options->lanes > LONG_MAX / options->lane_token_capacity ||
        (long)options->lanes * options->lane_token_capacity >
            LONG_MAX / bytes_per_token) {
        return NULL;
    }

    cache = (VxPagedKVCache*)calloc(1, sizeof(*cache));
    if (!cache) return NULL;
    cache->lanes = options->lanes;
    cache->page_tokens = options->page_tokens;
    cache->lane_token_capacity = options->lane_token_capacity;
    cache->pages_per_lane = (int)pages_per_lane;
    cache->max_pages = max_pages;
    cache->bytes_per_token = bytes_per_token;
    cache->policy = options->policy;
    cache->clear_on_recycle = options->clear_on_recycle ? 1 : 0;
    /* Contiguous is the identity map, and the identity map needs every private
     * page to exist.  A smaller bound would silently stop reproducing the
     * contiguous baseline it exists to reproduce. */
    if (cache->policy == VX_PAGED_KV_POLICY_CONTIGUOUS &&
        (size_t)cache->max_pages < private_pages) {
        free(cache);
        return NULL;
    }

    cache->page_table = (int*)malloc(private_pages * sizeof(int));
    cache->kv_length = (int*)calloc((size_t)cache->lanes, sizeof(int));
    cache->query_length = (int*)calloc((size_t)cache->lanes, sizeof(int));
    cache->lane_generation = (int*)calloc((size_t)cache->lanes, sizeof(int));
    cache->lane_reservation_id =
        (uint64_t*)calloc((size_t)cache->lanes, sizeof(uint64_t));
    cache->lane_prefix = (int*)malloc((size_t)cache->lanes * sizeof(int));
    cache->ref_count = (int*)calloc((size_t)cache->max_pages, sizeof(int));
    cache->page_generation = (int*)calloc((size_t)cache->max_pages, sizeof(int));
    cache->free_heap = (int*)malloc((size_t)cache->max_pages * sizeof(int));
    if (!cache->page_table || !cache->kv_length || !cache->query_length ||
        !cache->lane_generation || !cache->lane_reservation_id ||
        !cache->lane_prefix || !cache->ref_count || !cache->page_generation ||
        !cache->free_heap) {
        vx_paged_kv_destroy(cache);
        return NULL;
    }
    for (size_t index = 0; index < private_pages; index++) {
        cache->page_table[index] = VX_PAGED_KV_UNMAPPED;
    }
    for (int lane = 0; lane < cache->lanes; lane++) cache->lane_prefix[lane] = -1;
    /* Seed ascending so the first allocations are 0,1,2,... — the same order
     * the contiguous identity map produces for lane zero. */
    for (int page = 0; page < cache->max_pages; page++) cache->free_heap[page] = page;
    cache->free_count = cache->max_pages;
    cache->next_reservation_id = 1u;
    return cache;
}

void vx_paged_kv_destroy(VxPagedKVCache* cache) {
    if (!cache) return;
    for (int index = 0; index < cache->prefix_count; index++) {
        free(cache->prefixes[index].pages);
    }
    free(cache->prefixes);
    free(cache->page_table);
    free(cache->kv_length);
    free(cache->query_length);
    free(cache->lane_generation);
    free(cache->lane_reservation_id);
    free(cache->lane_prefix);
    free(cache->ref_count);
    free(cache->page_generation);
    free(cache->free_heap);
    free(cache);
}

void vx_paged_kv_set_page_eraser(VxPagedKVCache* cache,
                                 void (*erase)(int page, void* user),
                                 void* user) {
    if (!cache) return;
    cache->erase = erase;
    cache->erase_user = user;
}

int vx_paged_kv_lanes(const VxPagedKVCache* cache) { return cache ? cache->lanes : 0; }
int vx_paged_kv_page_tokens(const VxPagedKVCache* cache) {
    return cache ? cache->page_tokens : 0;
}
int vx_paged_kv_pages_per_lane(const VxPagedKVCache* cache) {
    return cache ? cache->pages_per_lane : 0;
}
int vx_paged_kv_lane_token_capacity(const VxPagedKVCache* cache) {
    return cache ? cache->lane_token_capacity : 0;
}
int vx_paged_kv_max_pages(const VxPagedKVCache* cache) {
    return cache ? cache->max_pages : 0;
}
const int* vx_paged_kv_lengths(const VxPagedKVCache* cache) {
    return cache ? cache->kv_length : NULL;
}
const int* vx_paged_kv_query_lengths(const VxPagedKVCache* cache) {
    return cache ? cache->query_length : NULL;
}
const int* vx_paged_kv_lane_generations(const VxPagedKVCache* cache) {
    return cache ? cache->lane_generation : NULL;
}
const int* vx_paged_kv_page_table(const VxPagedKVCache* cache) {
    return cache ? cache->page_table : NULL;
}

int vx_paged_kv_identity_physical_page(const VxPagedKVCache* cache,
                                       int lane, int logical) {
    if (!cache) return VX_PAGED_KV_UNMAPPED;
    return lane * cache->pages_per_lane + logical;
}

int vx_paged_kv_lane_is_contiguous(const VxPagedKVCache* cache, int lane) {
    int pages;
    int base;
    if (!lane_valid(cache, lane)) return 0;
    pages = (cache->kv_length[lane] + cache->page_tokens - 1) / cache->page_tokens;
    base = lane * cache->pages_per_lane;
    for (int logical = 0; logical < pages; logical++) {
        if (cache->page_table[base + logical] !=
            vx_paged_kv_identity_physical_page(cache, lane, logical)) return 0;
    }
    return 1;
}

int vx_paged_kv_physical_page(const VxPagedKVCache* cache, int lane, int logical) {
    if (!lane_valid(cache, lane) || logical < 0 || logical >= cache->pages_per_lane) {
        return VX_PAGED_KV_UNMAPPED;
    }
    return cache->page_table[lane * cache->pages_per_lane + logical];
}

VxPagedKVStatus vx_paged_kv_physical_token_index(const VxPagedKVCache* cache,
                                                 int lane, int position,
                                                 long* index_out) {
    int logical;
    int page;
    if (!lane_valid(cache, lane) || position < 0 || !index_out) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    logical = position / cache->page_tokens;
    page = vx_paged_kv_physical_page(cache, lane, logical);
    if (page == VX_PAGED_KV_UNMAPPED) return VX_PAGED_KV_INVALID_ARGUMENT;
    *index_out = (long)page * cache->page_tokens + (position % cache->page_tokens);
    return VX_PAGED_KV_OK;
}

VxPagedKVStatus vx_paged_kv_gather_active_tokens(const VxPagedKVCache* cache,
                                                 int lane, long* out) {
    if (!lane_valid(cache, lane) || !out) return VX_PAGED_KV_INVALID_ARGUMENT;
    for (int position = 0; position < cache->kv_length[lane]; position++) {
        VxPagedKVStatus status =
            vx_paged_kv_physical_token_index(cache, lane, position, &out[position]);
        if (status != VX_PAGED_KV_OK) return status;
    }
    return VX_PAGED_KV_OK;
}

VxPagedKVStatus vx_paged_kv_reserve(VxPagedKVCache* cache, int lane, int tokens,
                                    VxPagedKVReservation* reservation) {
    int prior_length;
    int next_length;
    int base;
    int first_logical;
    int last_logical;
    int needed;
    /* The output is disposable after every return, including argument errors. */
    if (!reservation) return VX_PAGED_KV_INVALID_ARGUMENT;
    memset(reservation, 0, sizeof(*reservation));
    if (!lane_valid(cache, lane) || tokens < 0) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    if (cache->lane_reservation_id[lane] != 0 ||
        cache->next_reservation_id == 0) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    reservation->lane = lane;
    reservation->cache = cache;
    reservation->tokens = tokens;
    prior_length = cache->kv_length[lane];
    reservation->prior_length = prior_length;
    if (prior_length < 0 || prior_length > cache->lane_token_capacity ||
        tokens > cache->lane_token_capacity - prior_length) {
        cache->allocation_failures++;
        return VX_PAGED_KV_CAPACITY_EXHAUSTED;
    }
    next_length = prior_length + tokens;
    base = lane * cache->pages_per_lane;
    first_logical = prior_length / cache->page_tokens;
    last_logical = next_length == 0 ? -1 : (next_length - 1) / cache->page_tokens;
    needed = last_logical - first_logical + 1;
    if (needed > 0) {
        reservation->pages = (int*)malloc((size_t)needed * sizeof(int));
        reservation->logical_pages = (int*)malloc((size_t)needed * sizeof(int));
        if (!reservation->pages || !reservation->logical_pages) {
            vx_paged_kv_reservation_dispose(reservation);
            return VX_PAGED_KV_CAPACITY_EXHAUSTED;
        }
        reservation->capacity = needed;
    }
    for (int logical = first_logical; logical <= last_logical; logical++) {
        int page;
        if (cache->page_table[base + logical] != VX_PAGED_KV_UNMAPPED) continue;
        page = allocate_page(cache, lane, logical);
        if (page < 0) {
            /* A half-mapped lane reads another request's pages.  Undo every
             * page this call mapped before returning the failure. */
            for (int index = reservation->page_count - 1; index >= 0; index--) {
                cache->page_table[base + reservation->logical_pages[index]] =
                    VX_PAGED_KV_UNMAPPED;
                release_page(cache, reservation->pages[index]);
            }
            vx_paged_kv_reservation_dispose(reservation);
            reservation->lane = lane;
            reservation->tokens = tokens;
            reservation->prior_length = prior_length;
            return VX_PAGED_KV_CAPACITY_EXHAUSTED;
        }
        cache->page_table[base + logical] = page;
        reservation->pages[reservation->page_count] = page;
        reservation->logical_pages[reservation->page_count] = logical;
        reservation->page_count++;
    }
    cache->reserved_pages += reservation->page_count;
    reservation->id = cache->next_reservation_id++;
    reservation->lane_generation = cache->lane_generation[lane];
    cache->lane_reservation_id[lane] = reservation->id;
    reservation->open = 1;
    note_high_water(cache);
    return VX_PAGED_KV_OK;
}

static VxPagedKVStatus reservation_validate(
    VxPagedKVCache* cache,
    const VxPagedKVReservation* reservation,
    int validate_tokens) {
    int base;
    if (!cache || !reservation || !reservation->open ||
        reservation->cache != cache ||
        !lane_valid(cache, reservation->lane) || reservation->id == 0 ||
        cache->lane_reservation_id[reservation->lane] != reservation->id ||
        cache->lane_generation[reservation->lane] != reservation->lane_generation ||
        cache->kv_length[reservation->lane] != reservation->prior_length ||
        (validate_tokens && reservation->tokens < 0) ||
        reservation->prior_length < 0 ||
        reservation->prior_length > cache->lane_token_capacity ||
        (validate_tokens && reservation->tokens >
            cache->lane_token_capacity - reservation->prior_length) ||
        reservation->page_count < 0 ||
        reservation->page_count > reservation->capacity) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    if (reservation->page_count > 0 &&
        (!reservation->pages || !reservation->logical_pages)) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    base = reservation->lane * cache->pages_per_lane;
    for (int index = 0; index < reservation->page_count; index++) {
        int logical = reservation->logical_pages[index];
        int page = reservation->pages[index];
        if (logical < 0 || logical >= cache->pages_per_lane || page < 0 ||
            page >= cache->max_pages || cache->ref_count[page] <= 0 ||
            cache->page_table[base + logical] != page) {
            return VX_PAGED_KV_STALE_PAGE;
        }
    }
    return VX_PAGED_KV_OK;
}

static VxPagedKVStatus reservation_batch_validate(
    VxPagedKVCache* cache,
    VxPagedKVReservation* reservations,
    size_t count,
    int validate_tokens,
    int* total_pages_out) {
    size_t total_pages = 0;
    if (!cache || (count && !reservations) || !total_pages_out)
        return VX_PAGED_KV_INVALID_ARGUMENT;
    for (size_t index = 0; index < count; index++) {
        VxPagedKVStatus status = reservation_validate(
            cache, &reservations[index], validate_tokens);
        if (status != VX_PAGED_KV_OK) return status;
        for (size_t prior = 0; prior < index; prior++)
            if (reservations[prior].lane == reservations[index].lane)
                return VX_PAGED_KV_INVALID_ARGUMENT;
        if ((size_t)reservations[index].page_count > SIZE_MAX - total_pages)
            return VX_PAGED_KV_INVALID_ARGUMENT;
        total_pages += (size_t)reservations[index].page_count;
    }
    if (total_pages > (size_t)INT_MAX ||
        cache->reserved_pages < (int)total_pages)
        return VX_PAGED_KV_INVALID_ARGUMENT;
    *total_pages_out = (int)total_pages;
    return VX_PAGED_KV_OK;
}

VxPagedKVStatus vx_paged_kv_commit_batch(
    VxPagedKVCache* cache,
    VxPagedKVReservation* reservations,
    size_t count) {
    int total_pages;
    VxPagedKVStatus status = reservation_batch_validate(
        cache, reservations, count, 1, &total_pages);
    if (status != VX_PAGED_KV_OK) return status;
    for (size_t index = 0; index < count; index++) {
        VxPagedKVReservation* reservation = &reservations[index];
        cache->lane_reservation_id[reservation->lane] = 0;
        reservation->open = 0;
        cache->kv_length[reservation->lane] =
            reservation->prior_length + reservation->tokens;
        cache->query_length[reservation->lane] = reservation->tokens;
    }
    cache->reserved_pages -= total_pages;
    for (size_t index = 0; index < count; index++)
        vx_paged_kv_reservation_dispose(&reservations[index]);
    return VX_PAGED_KV_OK;
}

VxPagedKVStatus vx_paged_kv_rollback_batch(
    VxPagedKVCache* cache,
    VxPagedKVReservation* reservations,
    size_t count) {
    int total_pages;
    VxPagedKVStatus status = reservation_batch_validate(
        cache, reservations, count, 0, &total_pages);
    if (status != VX_PAGED_KV_OK) return status;
    for (size_t index = 0; index < count; index++) {
        VxPagedKVReservation* reservation = &reservations[index];
        int base = reservation->lane * cache->pages_per_lane;
        cache->lane_reservation_id[reservation->lane] = 0;
        reservation->open = 0;
        for (int page_index = reservation->page_count - 1;
             page_index >= 0; page_index--) {
            cache->page_table[
                base + reservation->logical_pages[page_index]] =
                    VX_PAGED_KV_UNMAPPED;
            (void)release_page(cache, reservation->pages[page_index]);
        }
        cache->kv_length[reservation->lane] = reservation->prior_length;
    }
    cache->reserved_pages -= total_pages;
    for (size_t index = 0; index < count; index++)
        vx_paged_kv_reservation_dispose(&reservations[index]);
    return VX_PAGED_KV_OK;
}

VxPagedKVStatus vx_paged_kv_commit(VxPagedKVCache* cache,
                                   VxPagedKVReservation* reservation) {
    return vx_paged_kv_commit_batch(cache, reservation, 1u);
}

VxPagedKVStatus vx_paged_kv_rollback(VxPagedKVCache* cache,
                                     VxPagedKVReservation* reservation) {
    return vx_paged_kv_rollback_batch(cache, reservation, 1u);
}

void vx_paged_kv_reservation_dispose(VxPagedKVReservation* reservation) {
    if (!reservation) return;
    free(reservation->pages);
    free(reservation->logical_pages);
    reservation->cache = NULL;
    reservation->pages = NULL;
    reservation->logical_pages = NULL;
    reservation->page_count = 0;
    reservation->capacity = 0;
    reservation->id = 0;
    reservation->lane_generation = 0;
    reservation->open = 0;
}

VxPagedKVStatus vx_paged_kv_append(VxPagedKVCache* cache, int lane, int tokens) {
    VxPagedKVReservation reservation = {0};
    VxPagedKVStatus status = vx_paged_kv_reserve(cache, lane, tokens, &reservation);
    if (status != VX_PAGED_KV_OK) {
        vx_paged_kv_reservation_dispose(&reservation);
        return status;
    }
    status = vx_paged_kv_commit(cache, &reservation);
    if (status != VX_PAGED_KV_OK) {
        vx_paged_kv_rollback(cache, &reservation);
    }
    vx_paged_kv_reservation_dispose(&reservation);
    return status;
}

VxPagedKVStatus vx_paged_kv_release_lane(VxPagedKVCache* cache, int lane) {
    int base;
    int held;
    if (!lane_valid(cache, lane)) return VX_PAGED_KV_INVALID_ARGUMENT;
    if (cache->lane_reservation_id[lane] != 0) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    base = lane * cache->pages_per_lane;
    held = cache->lane_prefix[lane];
    if (held >= 0 && held < cache->prefix_count && cache->prefixes[held].live) {
        cache->prefixes[held].lane_references--;
    }
    cache->lane_prefix[lane] = -1;
    for (int logical = 0; logical < cache->pages_per_lane; logical++) {
        int page = cache->page_table[base + logical];
        if (page == VX_PAGED_KV_UNMAPPED) continue;
        cache->page_table[base + logical] = VX_PAGED_KV_UNMAPPED;
        release_page(cache, page);
    }
    cache->kv_length[lane] = 0;
    cache->query_length[lane] = 0;
    /* The bump is what stops work submitted against the old occupant from
     * landing in the new one once a scheduler reuses the slot. */
    cache->lane_generation[lane] = cache->lane_generation[lane] == INT_MAX
        ? 0 : cache->lane_generation[lane] + 1;
    return VX_PAGED_KV_OK;
}

void vx_paged_kv_reset(VxPagedKVCache* cache) {
    if (!cache) return;
    /* Reset has no status channel.  Keep it atomic by refusing to partially
     * reset a cache while any caller owns an unsettled lane transition. */
    for (int lane = 0; lane < cache->lanes; lane++) {
        if (cache->lane_reservation_id[lane] != 0) return;
    }
    for (int lane = 0; lane < cache->lanes; lane++) vx_paged_kv_release_lane(cache, lane);
    for (int index = 0; index < cache->prefix_count; index++) {
        VxPagedKVPrefix* prefix = &cache->prefixes[index];
        if (!prefix->live) continue;
        for (int page = 0; page < prefix->page_count; page++) {
            release_page(cache, prefix->pages[page]);
        }
        free(prefix->pages);
        prefix->pages = NULL;
        prefix->live = 0;
    }
    cache->prefix_count = 0;
}

static VxPagedKVPrefix* prefix_find(const VxPagedKVCache* cache, const char* key) {
    if (!cache || !key) return NULL;
    for (int index = 0; index < cache->prefix_count; index++) {
        VxPagedKVPrefix* prefix = &cache->prefixes[index];
        if (prefix->live && !strcmp(prefix->key, key)) return prefix;
    }
    return NULL;
}

static VxPagedKVPrefix* prefix_reserve_slot(VxPagedKVCache* cache) {
    for (int index = 0; index < cache->prefix_count; index++) {
        if (!cache->prefixes[index].live) return &cache->prefixes[index];
    }
    if (cache->prefix_count == cache->prefix_capacity) {
        int capacity = cache->prefix_capacity ? cache->prefix_capacity * 2 : 8;
        VxPagedKVPrefix* grown = (VxPagedKVPrefix*)realloc(
            cache->prefixes, (size_t)capacity * sizeof(*grown));
        if (!grown) return NULL;
        memset(grown + cache->prefix_capacity, 0,
               (size_t)(capacity - cache->prefix_capacity) * sizeof(*grown));
        cache->prefixes = grown;
        cache->prefix_capacity = capacity;
    }
    return &cache->prefixes[cache->prefix_count++];
}

VxPagedKVStatus vx_paged_kv_publish_prefix(VxPagedKVCache* cache, const char* key,
                                           int lane, int tokens) {
    VxPagedKVPrefix* existing;
    VxPagedKVPrefix* slot;
    int page_count;
    int base;
    int* pages;
    if (!lane_valid(cache, lane) || !key || !key[0] ||
        strlen(key) >= VX_PAGED_KV_PREFIX_KEY_MAX ||
        cache->lane_reservation_id[lane] != 0) return VX_PAGED_KV_INVALID_ARGUMENT;
    if (tokens <= 0 || tokens > cache->kv_length[lane]) return VX_PAGED_KV_INVALID_ARGUMENT;
    /* A partial tail page is still being appended to.  Publishing one hands
     * another request a page whose bytes are about to change — the mutable
     * aliasing shared prefixes exist to forbid. */
    if (tokens % cache->page_tokens != 0) return VX_PAGED_KV_INVALID_ARGUMENT;

    page_count = tokens / cache->page_tokens;
    base = lane * cache->pages_per_lane;
    pages = (int*)malloc((size_t)page_count * sizeof(int));
    if (!pages) return VX_PAGED_KV_CAPACITY_EXHAUSTED;
    for (int logical = 0; logical < page_count; logical++) {
        int page = cache->page_table[base + logical];
        if (page == VX_PAGED_KV_UNMAPPED) {
            free(pages);
            return VX_PAGED_KV_INVALID_ARGUMENT;
        }
        pages[logical] = page;
    }

    existing = prefix_find(cache, key);
    if (existing) {
        /* One identity, one page range.  Republishing a key over different
         * pages would let two requests holding "the same" prefix read
         * different bytes, which no amount of refcounting repairs. */
        int same = existing->tokens == tokens && existing->page_count == page_count;
        for (int index = 0; same && index < page_count; index++) {
            if (existing->pages[index] != pages[index]) same = 0;
        }
        free(pages);
        if (!same) return VX_PAGED_KV_PREFIX_CONFLICT;
        existing->last_use = ++cache->clock;
        return VX_PAGED_KV_OK;
    }

    slot = prefix_reserve_slot(cache);
    if (!slot) {
        free(pages);
        return VX_PAGED_KV_CAPACITY_EXHAUSTED;
    }
    /* The cache's own reference keeps published pages resident after the
     * publishing lane is released; eviction is what removes them. */
    for (int index = 0; index < page_count; index++) cache->ref_count[pages[index]]++;
    memset(slot, 0, sizeof(*slot));
    strncpy(slot->key, key, VX_PAGED_KV_PREFIX_KEY_MAX - 1);
    slot->pages = pages;
    slot->page_count = page_count;
    slot->tokens = tokens;
    slot->lane_references = 0;
    slot->sequence = ++cache->prefix_sequence;
    slot->last_use = ++cache->clock;
    slot->live = 1;
    return VX_PAGED_KV_OK;
}

int vx_paged_kv_has_prefix(const VxPagedKVCache* cache, const char* key) {
    return prefix_find(cache, key) != NULL;
}

VxPagedKVStatus vx_paged_kv_acquire_prefix(VxPagedKVCache* cache, const char* key,
                                           int lane, int* tokens_out) {
    VxPagedKVPrefix* prefix;
    int base;
    if (!lane_valid(cache, lane) || !key ||
        cache->lane_reservation_id[lane] != 0) return VX_PAGED_KV_INVALID_ARGUMENT;
    prefix = prefix_find(cache, key);
    if (!prefix) {
        cache->prefix_misses++;
        return VX_PAGED_KV_PREFIX_NOT_FOUND;
    }
    /* A prefix is a *prefix*.  Grafting one under existing tokens would
     * renumber them. */
    if (cache->kv_length[lane] != 0) return VX_PAGED_KV_INVALID_ARGUMENT;
    base = lane * cache->pages_per_lane;
    for (int logical = 0; logical < prefix->page_count; logical++) {
        int page = prefix->pages[logical];
        cache->page_table[base + logical] = page;
        cache->ref_count[page]++;
    }
    prefix->lane_references++;
    prefix->last_use = ++cache->clock;
    cache->lane_prefix[lane] = (int)(prefix - cache->prefixes);
    cache->kv_length[lane] = prefix->tokens;
    cache->query_length[lane] = 0;
    cache->prefix_hits++;
    note_high_water(cache);
    if (tokens_out) *tokens_out = prefix->tokens;
    return VX_PAGED_KV_OK;
}

int vx_paged_kv_page_is_shared(const VxPagedKVCache* cache, int lane, int logical) {
    int page = vx_paged_kv_physical_page(cache, lane, logical);
    return page != VX_PAGED_KV_UNMAPPED && cache->ref_count[page] > 1;
}

VxPagedKVStatus vx_paged_kv_copy_on_write(VxPagedKVCache* cache, int lane,
                                          int logical, int* from_out, int* to_out) {
    int source;
    int target;
    if (!lane_valid(cache, lane) || !from_out || !to_out ||
        cache->lane_reservation_id[lane] != 0) {
        return VX_PAGED_KV_INVALID_ARGUMENT;
    }
    source = vx_paged_kv_physical_page(cache, lane, logical);
    if (source == VX_PAGED_KV_UNMAPPED) return VX_PAGED_KV_INVALID_ARGUMENT;
    if (cache->ref_count[source] <= 1) {
        *from_out = source;
        *to_out = source;
        return VX_PAGED_KV_OK;
    }
    target = allocate_page(cache, lane, logical);
    if (target < 0) return VX_PAGED_KV_CAPACITY_EXHAUSTED;
    cache->page_table[lane * cache->pages_per_lane + logical] = target;
    cache->ref_count[source]--;
    cache->copy_on_writes++;
    note_high_water(cache);
    *from_out = source;
    *to_out = target;
    return VX_PAGED_KV_OK;
}

int vx_paged_kv_evict(VxPagedKVCache* cache, int pages) {
    int reclaimed = 0;
    if (!cache || pages < 0) return 0;
    while (cache->free_count < pages) {
        VxPagedKVPrefix* victim = NULL;
        for (int index = 0; index < cache->prefix_count; index++) {
            VxPagedKVPrefix* prefix = &cache->prefixes[index];
            if (!prefix->live || prefix->lane_references > 0) continue;
            /* Deterministic: least recently used, ties broken by publication
             * order, so an identical schedule evicts an identical set. */
            if (!victim || prefix->last_use < victim->last_use ||
                (prefix->last_use == victim->last_use &&
                 prefix->sequence < victim->sequence)) {
                victim = prefix;
            }
        }
        if (!victim) break;
        for (int index = 0; index < victim->page_count; index++) {
            release_page(cache, victim->pages[index]);
        }
        free(victim->pages);
        victim->pages = NULL;
        victim->live = 0;
        reclaimed += victim->page_count;
        victim->page_count = 0;
        cache->evictions++;
    }
    return reclaimed;
}

void vx_paged_kv_telemetry(const VxPagedKVCache* cache, VxPagedKVTelemetry* out) {
    long logical_tokens = 0;
    int resident_pages = 0;
    int shared_pages = 0;
    long resident_bytes;
    long logical_bytes;
    if (!cache || !out) return;
    memset(out, 0, sizeof(*out));
    for (int lane = 0; lane < cache->lanes; lane++) logical_tokens += cache->kv_length[lane];
    for (int page = 0; page < cache->max_pages; page++) {
        if (cache->ref_count[page] <= 0) continue;
        resident_pages++;
        if (cache->ref_count[page] > 1) shared_pages++;
    }
    resident_bytes = (long)resident_pages * cache->page_tokens * cache->bytes_per_token;
    /* Shared tokens are counted once per holding lane in logical_tokens, which
     * is the honest reading of "bytes the active lengths mean" but would make
     * fragmentation negative.  Clamp rather than redefine either number. */
    logical_bytes = logical_tokens * cache->bytes_per_token;
    out->logical_bytes = logical_bytes;
    out->resident_bytes = resident_bytes;
    out->reserved_bytes =
        (long)cache->reserved_pages * cache->page_tokens * cache->bytes_per_token;
    out->resident_pages = resident_pages;
    out->reserved_pages = cache->reserved_pages;
    out->free_pages = cache->free_count;
    out->shared_pages = shared_pages;
    out->fragmentation_bytes =
        resident_bytes > logical_bytes ? resident_bytes - logical_bytes : 0;
    out->high_water_pages = cache->high_water_pages;
    out->prefix_hits = cache->prefix_hits;
    out->prefix_misses = cache->prefix_misses;
    out->copy_on_writes = cache->copy_on_writes;
    out->evictions = cache->evictions;
    out->allocation_failures = cache->allocation_failures;
}

int vx_paged_kv_page_generation(const VxPagedKVCache* cache, int page) {
    if (!cache || page < 0 || page >= cache->max_pages) return -1;
    return cache->page_generation[page];
}
