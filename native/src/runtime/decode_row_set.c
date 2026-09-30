#include "decode_row_set.h"

#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

/*
 * The addressing arithmetic, once.  Every function below is a projection of the
 * two lines in `slotRowIndex`'s TypeScript twin, and the shared corpus asserts
 * that the projections agree across the two runtimes.
 */

static int slot_row_index(const VxDecodeRowSet* set, int slot, int token,
                          int slot_stride, int paged, int* out_row) {
    const VxDecodeSlotPages* pages = &set->pages[slot];
    if (!paged || pages->page_table == NULL) {
        int64_t row = (int64_t)slot * slot_stride + token;
        if (row < 0 || row > INT_MAX) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
        *out_row = (int)row;
        return VX_DECODE_ROW_SET_OK;
    }
    const int logical = token / pages->page_tokens;
    if (logical >= pages->pages_per_slot) return VX_DECODE_ROW_SET_UNMAPPED;
    const int page = pages->page_table[logical];
    if (page == VX_PAGED_KV_UNMAPPED || page < 0) return VX_DECODE_ROW_SET_UNMAPPED;
    int64_t row = (int64_t)page * pages->page_tokens + (token % pages->page_tokens);
    if (row > INT_MAX) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    *out_row = (int)row;
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_set_init(VxDecodeRowSet* set, int slots,
                                           const int* positions,
                                           const VxDecodeSlotPages* pages) {
    if (set == NULL || positions == NULL || slots < 1) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    memset(set, 0, sizeof(*set));
    const size_t slot_bytes = 7 * sizeof(int) + sizeof(VxDecodeSlotPages) + sizeof(VxDecodeRowRun);
    if ((size_t)slots > (SIZE_MAX - 16u) / slot_bytes) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    set->storage = calloc(1, (size_t)slots * slot_bytes + 16u);
    if (!set->storage) return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    set->slots = slots;
    set->positions = set->storage;
    set->empty = set->positions + slots;
    set->kv_lengths = set->empty + slots;
    for (int i = 0; i < 4; i++) set->scratch_indices[i] = set->kv_lengths + (size_t)(i + 1) * slots;
    uintptr_t address = (uintptr_t)(set->kv_lengths + (size_t)5 * slots);
    address = (address + _Alignof(VxDecodeSlotPages) - 1) & ~(uintptr_t)(_Alignof(VxDecodeSlotPages) - 1);
    set->pages = (VxDecodeSlotPages*)address;
    set->scratch_runs = (VxDecodeRowRun*)(set->pages + slots);
    int paged_slots = 0;
    for (int slot = 0; slot < slots; slot++) {
        if (positions[slot] == VX_DECODE_ROW_EMPTY) {
            /* Row zero is a placeholder, not a destination: the scatter skips a
             * empty slot, so nothing is ever written there.  It is chosen
             * because it is always in bounds and always initialised, which
             * keeps the slot's staged *input* row readable without a page. */
            set->positions[slot] = 0;
            set->kv_lengths[slot] = 0;
            set->empty[slot] = 1;
            continue;
        }
        if (positions[slot] < 0 || positions[slot] == INT_MAX) goto invalid;
        set->live++;
        set->positions[slot] = positions[slot];
        set->kv_lengths[slot] = positions[slot] + 1;
        if (set->kv_lengths[slot] > set->key_capacity) {
            set->key_capacity = set->kv_lengths[slot];
        }
        if (pages != NULL && pages[slot].page_table != NULL) {
            if (pages[slot].page_tokens < 1 || pages[slot].pages_per_slot < 1) {
                goto invalid;
            }
            set->pages[slot] = pages[slot];
            paged_slots++;
        }
    }
    if (set->live == 0 || (int64_t)set->slots * set->key_capacity > INT_MAX) goto invalid;
    /* All paged or none.  Half-applied paging pairs a paged read with a linear
     * write, which is worse than no paging at all.  Empty slots are excluded:
     * they address nothing, so they cannot address it under a second meaning. */
    if (paged_slots != 0 && paged_slots != set->live) {
        goto invalid;
    }
    set->unpaged = paged_slots == 0 ? 1 : 0;
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
                                                 int slot_stride, int paged,
                                                 int* out_rows) {
    if (set == NULL || out_rows == NULL || slot_stride < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int slot = 0; slot < set->slots; slot++) {
        int row = 0;
        const int status = slot_row_index(
            set, slot, set->positions[slot], slot_stride, paged, &row);
        if (status != VX_DECODE_ROW_SET_OK) return (VxDecodeRowSetStatus)status;
        out_rows[slot] = row;
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_set_prefix_rows(const VxDecodeRowSet* set,
                                                   int slot_stride, int paged,
                                                   int* out_rows) {
    if (set == NULL || out_rows == NULL || slot_stride < 0) {
        return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
    }
    for (int slot = 0; slot < set->slots; slot++) {
        const int base = slot * set->key_capacity;
        for (int token = 0; token < set->key_capacity; token++) {
            if (token >= set->kv_lengths[slot]) {
                /* The slot's padding.  The keep mask already stops a kernel from
                 * reading it; naming it -1 rather than an arbitrary target is what
                 * makes a staging copy zero it instead of carrying another
                 * request's bytes. */
                out_rows[base + token] = -1;
                continue;
            }
            int row = 0;
            const int status = slot_row_index(
                set, slot, token, slot_stride, paged, &row);
            if (status != VX_DECODE_ROW_SET_OK) return (VxDecodeRowSetStatus)status;
            out_rows[base + token] = row;
        }
    }
    return VX_DECODE_ROW_SET_OK;
}

static int keep_mask_reads(const VxDecodeRowSet* set,
                           const VxDecodeKeepMask* mask, int slot, int key) {
    if (mask == NULL || mask->layout == VX_DECODE_KEEP_MASK_NONE ||
        mask->values == NULL) {
        return 1;
    }
    const int keys = mask->keys;
    const int queries = mask->queries;
    const int query = set->positions[slot];
    switch (mask->layout) {
        case VX_DECODE_KEEP_MASK_K:
            return mask->values[key] != 0;
        case VX_DECODE_KEEP_MASK_BK:
            return mask->values[slot * keys + key] != 0;
        case VX_DECODE_KEEP_MASK_QK:
            return mask->values[query * keys + key] != 0;
        case VX_DECODE_KEEP_MASK_BQK:
            return mask->values[(slot * queries + query) * keys + key] != 0;
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
    for (int slot = 0; slot < set->slots; slot++) {
        const int length = clamp_to_length ? set->kv_lengths[slot] : keys;
        const int base = slot * keys;
        for (int key = 0; key < keys; key++) {
            out_mask[base + key] =
                (key < length && keep_mask_reads(set, mask, slot, key)) ? 1 : 0;
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
        unsigned char* target = destination + (size_t)index * row_bytes;
        if (rows[index] < 0) {
            memset(target, 0, row_bytes);
            continue;
        }
        memcpy(target, source + (size_t)rows[index] * row_bytes, row_bytes);
    }
    return VX_DECODE_ROW_SET_OK;
}

VxDecodeRowSetStatus vx_decode_row_scatter(void* storage, size_t storage_bytes,
                                           const void* staged, size_t row_bytes,
                                           const int* rows, int count,
                                           const int* empty) {
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
        if (empty != NULL && empty[index]) continue;
        memcpy(destination + (size_t)rows[index] * row_bytes,
               source + (size_t)index * row_bytes, row_bytes);
    }
    return VX_DECODE_ROW_SET_OK;
}
