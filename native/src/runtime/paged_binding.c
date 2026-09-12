#include "paged_binding.h"
#include "vx_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * See paged_binding.h for the contract.
 *
 * The binding is context state, reached through the same thread-local engine
 * state every other decode global uses, so two contexts paging independently
 * cannot see each other's page tables.
 */

int vx_paged_bind_locked(VxPagedKVCache* cache, int lane,
                         const char* const* paged_names, int paged_count) {
    VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!cache) {
        vx_paged_unbind_locked();
        return 0;
    }
    if (lane < 0 || lane >= vx_paged_kv_lanes(cache) ||
        paged_count < 0 || (size_t)paged_count > SIZE_MAX / sizeof(binding->names[0])) return -1;
    for (int index = 0; index < paged_count; index++) {
        if (!paged_names || !paged_names[index] || !paged_names[index][0] ||
            strlen(paged_names[index]) >= sizeof(binding->names[0])) return -1;
        for (int prior = 0; prior < index; prior++)
            if (!strcmp(paged_names[index], paged_names[prior])) return -1;
    }
    if (binding->bound && binding->cache == cache && binding->lane == lane &&
        binding->name_count == paged_count) {
        int same = 1;
        for (int i = 0; i < paged_count; i++) if (strcmp(binding->names[i], paged_names[i])) same = 0;
        if (same) return 0;
    }
    char (*names)[128] = paged_count ? malloc((size_t)paged_count * sizeof(*names)) : NULL;
    if (paged_count && !names) return -1;
    for (int index = 0; index < paged_count; index++)
        snprintf(names[index], sizeof(names[index]), "%s", paged_names[index]);
    free(binding->names);
    /* Rebinding keeps the gather buffers: a decode loop rebinds every step and
     * reallocating each time is the growth this cache exists to avoid. */
    {
        unsigned char* gather[2] = { binding->gather[0], binding->gather[1] };
        size_t bytes[2] = { binding->gather_bytes[0], binding->gather_bytes[1] };
        memset(binding, 0, sizeof(*binding));
        binding->gather[0] = gather[0];
        binding->gather[1] = gather[1];
        binding->gather_bytes[0] = bytes[0];
        binding->gather_bytes[1] = bytes[1];
    }
    binding->names = names;
    binding->cache = cache;
    binding->lane = lane;
    binding->name_count = paged_count;
    binding->bound = 1;
    return 0;
}

void vx_paged_unbind_locked(void) {
    VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    for (int slot = 0; slot < 2; slot++) free(binding->gather[slot]);
    free(binding->names);
    memset(binding, 0, sizeof(*binding));
}

int vx_paged_bound_locked(void) {
    return vx_engine_state_current()->paged.bound != 0;
}

int vx_paged_tensor_locked(const T* tensor) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound || !tensor || !tensor->name[0]) return 0;
    for (int index = 0; index < binding->name_count; index++) {
        if (!strcmp(binding->names[index], tensor->name)) return 1;
    }
    return 0;
}

int vx_paged_lanes_locked(void) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    return binding->bound ? vx_paged_kv_lanes(binding->cache) : 0;
}

int vx_paged_lane_pages_locked(int lane, VxDecodeLanePages* out) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!binding->bound) return 0;
    if (lane < 0 || lane >= vx_paged_kv_lanes(binding->cache)) return -1;
    out->page_tokens = vx_paged_kv_page_tokens(binding->cache);
    out->pages_per_lane = vx_paged_kv_pages_per_lane(binding->cache);
    if (out->page_tokens <= 0 || out->pages_per_lane <= 0) return -1;
    out->page_table = vx_paged_kv_page_table(binding->cache) +
        (size_t)lane * (size_t)out->pages_per_lane;
    return 1;
}

VxDecodeRowSetStatus vx_paged_row_set_init_locked(VxDecodeRowSet* rows,
    int lanes, const int* positions) {
    VxDecodeRowSetStatus status = vx_decode_row_set_init(rows, lanes, positions, NULL);
    if (status != VX_DECODE_ROW_SET_OK || !vx_paged_bound_locked()) return status;
    if (lanes > vx_paged_lanes_locked()) goto invalid;
    for (int lane = 0; lane < lanes; lane++)
        if (!rows->parked[lane] && vx_paged_lane_pages_locked(lane, &rows->pages[lane]) != 1) goto invalid;
    rows->unpaged = 0;
    return VX_DECODE_ROW_SET_OK;
invalid:
    vx_decode_row_set_dispose(rows);
    return VX_DECODE_ROW_SET_INVALID_ARGUMENT;
}

int vx_paged_row_lane_locked(const T* tensor, int logical_row, int lane) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    int page_tokens;
    int page;
    /* The identity answer, and the reason this is one substitution rather than
     * a branch at every call site. */
    if (!vx_paged_tensor_locked(tensor) || logical_row < 0) return logical_row;
    if (lane < 0 || lane >= vx_paged_kv_lanes(binding->cache)) return -1;
    page_tokens = vx_paged_kv_page_tokens(binding->cache);
    if (page_tokens <= 0) return -1;
    page = vx_paged_kv_physical_page(binding->cache, lane,
                                     logical_row / page_tokens);
    if (page == VX_PAGED_KV_UNMAPPED) return -1;
    return page * page_tokens + (logical_row % page_tokens);
}

int vx_paged_row_locked(const T* tensor, int logical_row) {
    return vx_paged_row_lane_locked(
        tensor, logical_row, vx_engine_state_current()->paged.lane);
}

const void* vx_paged_gather_lane_bytes_locked(const void* pool, int pool_rows,
                                              size_t row_bytes, int kv_length,
                                              int slot, int lane) {
    VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    size_t needed;
    int page_tokens;
    if (!binding->bound || !pool || row_bytes == 0 || kv_length <= 0 ||
        pool_rows <= 0 || kv_length > pool_rows || slot < 0 || slot > 1 ||
        lane < 0 || lane >= vx_paged_kv_lanes(binding->cache)) return NULL;
    if ((size_t)kv_length > SIZE_MAX / row_bytes) return NULL;
    needed = (size_t)kv_length * row_bytes;
    if (binding->gather_bytes[slot] < needed) {
        size_t capacity = binding->gather_bytes[slot] ? binding->gather_bytes[slot] : needed;
        unsigned char* grown;
        while (capacity < needed) capacity *= 2;
        grown = (unsigned char*)realloc(binding->gather[slot], capacity);
        if (!grown) return NULL;
        binding->gather[slot] = grown;
        binding->gather_bytes[slot] = capacity;
    }
    page_tokens = vx_paged_kv_page_tokens(binding->cache);
    if (page_tokens <= 0) return NULL;
    for (int position = 0; position < kv_length; position++) {
        int page = vx_paged_kv_physical_page(binding->cache, lane,
                                             position / page_tokens);
        long row;
        if (page == VX_PAGED_KV_UNMAPPED) return NULL;
        row = (long)page * page_tokens + (position % page_tokens);
        if (row < 0 || row >= pool_rows) return NULL;
        memcpy((unsigned char*)binding->gather[slot] + (size_t)position * row_bytes,
               (const unsigned char*)pool + (size_t)row * row_bytes, row_bytes);
    }
    return binding->gather[slot];
}

const float* vx_paged_gather_lane_f32_locked(const float* pool, int pool_rows,
                                             int width, int kv_length,
                                             int slot, int lane) {
    if (width <= 0 || (size_t)width > SIZE_MAX / sizeof(float)) return NULL;
    return (const float*)vx_paged_gather_lane_bytes_locked(
        pool, pool_rows, (size_t)width * sizeof(float), kv_length, slot, lane);
}

const float* vx_paged_gather_f32_locked(const float* pool, int pool_rows,
                                        int width, int kv_length, int slot) {
    return vx_paged_gather_lane_f32_locked(
        pool, pool_rows, width, kv_length, slot,
        vx_engine_state_current()->paged.lane);
}

int vx_paged_kv_length_lane_locked(int lane) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound || lane < 0 ||
        lane >= vx_paged_kv_lanes(binding->cache)) return -1;
    return vx_paged_kv_lengths(binding->cache)[lane];
}

int vx_paged_kv_length_locked(void) {
    return vx_paged_kv_length_lane_locked(
        vx_engine_state_current()->paged.lane);
}

const int* vx_paged_page_table_locked(void) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound) return NULL;
    return vx_paged_kv_page_table(binding->cache) +
        (size_t)binding->lane * vx_paged_kv_pages_per_lane(binding->cache);
}

int vx_paged_page_tokens_locked(void) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    return binding->bound ? vx_paged_kv_page_tokens(binding->cache) : 0;
}

/* Attention reads a paged operand on `k` or `v`; nothing else may. */
static int paged_read_port_allowed(const Node* node, VxPortKind key) {
    if ((node->operator_kind != VX_OP_CROSS_SDPA) && (node->operator_kind != VX_OP_Q_SDPA)) return 0;
    return !(key != VX_PORT_K) || !(key != VX_PORT_V);
}

int vx_paged_domain_supported_locked(const unsigned char* selected_nodes) {
    const VxPagedBindingState* binding = &vx_engine_state_current()->paged;
    if (!binding->bound) return 1;

    for (int node_index = 0; node_index < g_nn; node_index++) {
        const Node* node = &g_n[node_index];
        if (selected_nodes && !selected_nodes[node_index]) continue;
        for (int input = 0; input < node->nin; input++) {
            const T* tensor = t_find(node->ins[input].name);
            if (!vx_paged_tensor_locked(tensor)) continue;
            if (!paged_read_port_allowed(node, node->ins[input].port)) {
                if (vx_engine_env("VOLVOXAI_ROW_DEBUG")) {
                    vx_engine_log(
                            "[row] node %d op=%s reads paged tensor %s on port %s\n",
                            node_index, vx_operator_kind_name(node->operator_kind), tensor->name, node->ins[input].key);
                }
                return 0;
            }
        }
        for (int output = 0; output < node->nout; output++) {
            const T* tensor = t_find(node->outs[output].name);
            if (!vx_paged_tensor_locked(tensor)) continue;
            for (int index = 0; index < binding->name_count; index++) {
                if (strcmp(binding->names[index], tensor->name)) continue;
                for (int prior = 0; prior < node_index; prior++) {
                    if (selected_nodes && !selected_nodes[prior]) continue;
                    for (int port = 0; port < g_n[prior].nout; port++)
                        if (!strcmp(g_n[prior].outs[port].name, tensor->name)) return 0;
                }
            }
        }
    }
    return 1;
}
