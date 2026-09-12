#include "decode_row_set.h"

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

/*
 * The addressing arithmetic, once.  Every function below is a projection of the
 * two lines in `laneRowIndex`'s TypeScript twin, and the shared corpus asserts
 * that the projections agree across the two runtimes.
 */

static int lane_row_index(const VxDecodeRowSet* set, int lane, int token,
                          int lane_stride, int paged, int* out_row) {
    const VxDecodeLanePages* pages = &set->pages[lane];
    if (!paged || pages->page_table == NULL) {
        int64_t row = (int64_t)lane * lane_stride + token;
        if (row < 0 || row > INT_MAX) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        *out_row = (int)row;
        return VX_DECODE_ROW_SET_OK;
    }
    const int logical = token / pages->page_tokens;
    if (logical >= pages->pages_per_lane) return VX_DECODE_ROW_SET_UNMAPPED;
    const int page = pages->page_table[logical];
    if (page == VX_PAGED_KV_UNMAPPED || page < 0) return VX_DECODE_ROW_SET_UNMAPPED;
    int64_t row = (int64_t)page * pages->page_tokens + (token % pages->page_tokens);
    if (row > INT_MAX) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    *out_row = (int)row;
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_set_init(VxDecodeRowSet* set, int lanes,
                                           const int* positions,
                                           const VxDecodeLanePages* pages) {
    if (set == NULL || positions == NULL || lanes < 1) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    memset(set, 0, sizeof(*set));
    const size_t lane_bytes = 7 * sizeof(int) + sizeof(VxDecodeLanePages) + sizeof(VxDecodeRowRun);
    if ((size_t)lanes > (SIZE_MAX - 16u) / lane_bytes) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    set->storage = calloc(1, (size_t)lanes * lane_bytes + 16u);
    if (!set->storage) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    set->lanes = lanes;
    set->positions = set->storage;
    set->parked = set->positions + lanes;
    set->kv_lengths = set->parked + lanes;
    for (int i = 0; i < 4; i++) set->scratch_indices[i] = set->kv_lengths + (size_t)(i + 1) * lanes;
    uintptr_t address = (uintptr_t)(set->kv_lengths + (size_t)5 * lanes);
    address = (address + _Alignof(VxDecodeLanePages) - 1) & ~(uintptr_t)(_Alignof(VxDecodeLanePages) - 1);
    set->pages = (VxDecodeLanePages*)address;
    set->scratch_runs = (VxDecodeRowRun*)(set->pages + lanes);
    int paged_lanes = 0;
    for (int lane = 0; lane < lanes; lane++) {
        if (positions[lane] == VX_DECODE_ROW_PARKED) {
            /* Row zero is a placeholder, not a destination: the scatter skips a
             * parked lane, so nothing is ever written there.  It is chosen
             * because it is always in bounds and always initialised, which
             * keeps the lane's staged *input* row readable without a page. */
            set->positions[lane] = 0;
            set->kv_lengths[lane] = 0;
            set->parked[lane] = 1;
            continue;
        }
        if (positions[lane] < 0 || positions[lane] == INT_MAX) goto invalid;
        set->live++;
        set->positions[lane] = positions[lane];
        set->kv_lengths[lane] = positions[lane] + 1;
        if (set->kv_lengths[lane] > set->key_capacity) {
            set->key_capacity = set->kv_lengths[lane];
        }
        if (pages != NULL && pages[lane].page_table != NULL) {
            if (pages[lane].page_tokens < 1 || pages[lane].pages_per_lane < 1) {
                goto invalid;
            }
            set->pages[lane] = pages[lane];
            paged_lanes++;
        }
    }
    if (set->live == 0 || (int64_t)set->lanes * set->key_capacity > INT_MAX) goto invalid;
    /* All paged or none.  Half-applied paging pairs a paged read with a linear
     * write, which is worse than no paging at all.  Parked lanes are excluded:
     * they address nothing, so they cannot address it under a second meaning. */
    if (paged_lanes != 0 && paged_lanes != set->live) {
        goto invalid;
    }
    set->unpaged = paged_lanes == 0 ? 1 : 0;
    return VX_DECODE_ROW_SET_OK;
invalid:
    vx_decode_row_set_dispose(set);
    return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
}

void vx_decode_row_set_dispose(VxDecodeRowSet* set) {
    if (!set) return;
    free(set->storage);
    memset(set, 0, sizeof(*set));
}

VxDecodeRowSetStatus vx_decode_row_set_write_rows(const VxDecodeRowSet* set,
                                                 int lane_stride, int paged,
                                                 int* out_rows) {
    if (set == NULL || out_rows == NULL || lane_stride < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int lane = 0; lane < set->lanes; lane++) {
        int row = 0;
        const int status = lane_row_index(
            set, lane, set->positions[lane], lane_stride, paged, &row);
        if (status != VX_DECODE_ROW_SET_OK) return (VxDecodeRowSetStatus)status;
        out_rows[lane] = row;
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_set_prefix_rows(const VxDecodeRowSet* set,
                                                   int lane_stride, int paged,
                                                   int* out_rows) {
    if (set == NULL || out_rows == NULL || lane_stride < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int lane = 0; lane < set->lanes; lane++) {
        const int base = lane * set->key_capacity;
        for (int token = 0; token < set->key_capacity; token++) {
            if (token >= set->kv_lengths[lane]) {
                /* The lane's padding.  The keep mask already stops a kernel from
                 * reading it; naming it -1 rather than an arbitrary slot is what
                 * makes a staging copy zero it instead of carrying another
                 * request's bytes. */
                out_rows[base + token] = -1;
                continue;
            }
            int row = 0;
            const int status = lane_row_index(
                set, lane, token, lane_stride, paged, &row);
            if (status != VX_DECODE_ROW_SET_OK) return (VxDecodeRowSetStatus)status;
            out_rows[base + token] = row;
        }
    }
    return VX_DECODE_ROW_SET_OK;
}

static int keep_mask_reads(const VxDecodeRowSet* set,
                           const VxDecodeKeepMask* mask, int lane, int key) {
    if (mask == NULL || mask->layout == VX_DECODE_KEEP_MASK_NONE ||
        mask->values == NULL) {
        return 1;
    }
    const int keys = mask->keys;
    const int queries = mask->queries;
    const int query = set->positions[lane];
    switch (mask->layout) {
        case VX_DECODE_KEEP_MASK_K:
            return mask->values[key] != 0;
        case VX_DECODE_KEEP_MASK_BK:
            return mask->values[lane * keys + key] != 0;
        case VX_DECODE_KEEP_MASK_QK:
            return mask->values[query * keys + key] != 0;
        case VX_DECODE_KEEP_MASK_BQK:
            return mask->values[(lane * queries + query) * keys + key] != 0;
        default:
            return 1;
    }
}

VxDecodeRowSetStatus vx_decode_row_set_keep_mask(const VxDecodeRowSet* set,
                                                int keys, int clamp_to_length,
                                                const VxDecodeKeepMask* mask,
                                                int* out_mask) {
    if (set == NULL || out_mask == NULL || keys < 1) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int lane = 0; lane < set->lanes; lane++) {
        const int length = clamp_to_length ? set->kv_lengths[lane] : keys;
        const int base = lane * keys;
        for (int key = 0; key < keys; key++) {
            out_mask[base + key] =
                (key < length && keep_mask_reads(set, mask, lane, key)) ? 1 : 0;
        }
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSpanTier vx_decode_row_span_tier(const int* rows, int count) {
    if (rows == NULL || count < 1 || rows[0] < 0) return VX_DECODE_ROW_SPAN_GATHER;
    for (int index = 1; index < count; index++) {
        if (rows[index] != rows[index - 1] + 1) return VX_DECODE_ROW_SPAN_GATHER;
    }
    return VX_DECODE_ROW_SPAN_IDENTITY;
}

int vx_decode_row_copy_runs(const int* rows, int count,
                            VxDecodeRowRun* out, int capacity) {
    int runs = 0;
    if (rows == NULL || out == NULL || count < 0 || capacity < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int index = 0; index < count; index++) {
        const int source = rows[index];
        /* Padding ends the current run and produces no copy, which is what
         * leaves the destination available for the zeroing pass. */
        if (source < 0) continue;
        if (runs > 0) {
            VxDecodeRowRun* last = &out[runs - 1];
            if (last->source + last->rows == source &&
                last->destination + last->rows == index) {
                last->rows++;
                continue;
            }
        }
        if (runs >= capacity) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        out[runs].source = source;
        out[runs].destination = index;
        out[runs].rows = 1;
        runs++;
    }
    return runs;
}

int vx_decode_row_padding_runs(const int* rows, int count,
                               VxDecodeRowRun* out, int capacity) {
    int runs = 0;
    if (rows == NULL || out == NULL || count < 0 || capacity < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int index = 0; index < count; index++) {
        if (rows[index] >= 0) continue;
        if (runs > 0) {
            VxDecodeRowRun* last = &out[runs - 1];
            if (last->destination + last->rows == index) {
                last->rows++;
                continue;
            }
        }
        if (runs >= capacity) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        out[runs].source = -1;
        out[runs].destination = index;
        out[runs].rows = 1;
        runs++;
    }
    return runs;
}

/* Every source row lies wholly inside `storage_bytes`.  Checked before any
 * byte moves, so a rejected span leaves the destination untouched rather than
 * half-filled. */
static VxDecodeRowSetStatus rows_in_bounds(const int* rows, int count,
                                           size_t storage_bytes, size_t row_bytes) {
    if (rows == NULL || count < 0 || row_bytes == 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int index = 0; index < count; index++) {
        size_t start;
        if (rows[index] < 0) continue;
        if ((size_t)rows[index] > (SIZE_MAX - row_bytes) / row_bytes) {
            return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        }
        start = (size_t)rows[index] * row_bytes;
        if (start + row_bytes > storage_bytes) {
            return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        }
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_gather(void* staged, const void* storage,
                                          size_t storage_bytes, size_t row_bytes,
                                          const int* rows, int count) {
    unsigned char* destination = (unsigned char*)staged;
    const unsigned char* source = (const unsigned char*)storage;
    VxDecodeRowSetStatus status;
    if (staged == NULL || storage == NULL) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    status = rows_in_bounds(rows, count, storage_bytes, row_bytes);
    if (status != VX_DECODE_ROW_SET_OK) return status;
    for (int index = 0; index < count; index++) {
        unsigned char* slot = destination + (size_t)index * row_bytes;
        if (rows[index] < 0) {
            memset(slot, 0, row_bytes);
            continue;
        }
        memcpy(slot, source + (size_t)rows[index] * row_bytes, row_bytes);
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_scatter(void* storage, size_t storage_bytes,
                                           const void* staged, size_t row_bytes,
                                           const int* rows, int count,
                                           const int* parked) {
    unsigned char* destination = (unsigned char*)storage;
    const unsigned char* source = (const unsigned char*)staged;
    VxDecodeRowSetStatus status;
    if (storage == NULL || staged == NULL) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    status = rows_in_bounds(rows, count, storage_bytes, row_bytes);
    if (status != VX_DECODE_ROW_SET_OK) return status;
    for (int index = 0; index < count; index++) {
        if (rows[index] < 0) continue;
        if (parked != NULL && parked[index]) continue;
        memcpy(destination + (size_t)rows[index] * row_bytes,
               source + (size_t)index * row_bytes, row_bytes);
    }
    return VX_DECODE_ROW_SET_OK;
}
