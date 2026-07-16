#include "engine_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- Activation arena: reuse physical buffers across non-overlapping lifetimes ----
// Similar to planned tensor arenas in optimized inference runtimes. Transient F32 and
// physical I8/U8 activations are pooled: owned (calloc'd by us -> NOT weights, which
// point into the blob), with a non-skipped producer node (excludes graph
// inputs/constants) and at least one consumer (excludes graph outputs and dead tensors),
// and not aliased by another tensor (excludes Reshape/Flatten passthroughs).
// last-use scans ALL nodes INCLUDING skipped ones, so conv+add residuals and other
// fused-away consumers keep their buffer live conservatively. Cuts peak activation memory
// (~88 -> ~10 MB here) which removes most cold-start first-touch page faults.
/* Arena blocks are raw byte allocations.  T.data is historically typed as a
 * float pointer, but canonical W8A8 intermediates really own one byte per
 * element.  Keeping capacities in elements would reserve four times as much
 * memory and, more importantly, would make a mixed F32/I8 reuse plan reason
 * about the wrong unit. */
static void** g_arena_bufs = NULL;
static int g_arena_nbufs = 0;
static int* g_arena_tensor_indices = NULL;
static int g_arena_tensor_count = 0;

void volvoxai_engine_free_arena(void) {
    for (int i = 0; i < g_arena_nbufs; i++) free(g_arena_bufs[i]);
    free(g_arena_bufs);
    for (int i = 0; i < g_arena_tensor_count; i++) {
        int index = g_arena_tensor_indices[i];
        if (index < 0 || index >= g_nt) continue;
        g_t[index].data = NULL;
        g_t[index].owns = 0;
    }
    free(g_arena_tensor_indices);
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
    g_arena_tensor_indices = NULL;
    g_arena_tensor_count = 0;
}

static int restore_arena_tensors(void) {
    if (g_arena_tensor_count <= 0) { volvoxai_engine_free_arena(); return 0; }
    void** replacements = (void**)calloc((size_t)g_arena_tensor_count, sizeof(void*));
    if (!replacements) return -1;
    for (int i = 0; i < g_arena_tensor_count; i++) {
        int index = g_arena_tensor_indices[i];
        if (index < 0 || index >= g_nt) { free(replacements); return -1; }
        T* tensor = &g_t[index];
        replacements[i] = calloc(tensor->numel > 0 ? (size_t)tensor->numel : 1u, tensor->elem_size);
        if (!replacements[i]) {
            for (int j = 0; j < i; j++) free(replacements[j]);
            free(replacements);
            return -1;
        }
    }
    for (int i = 0; i < g_arena_nbufs; i++) free(g_arena_bufs[i]);
    free(g_arena_bufs);
    for (int i = 0; i < g_arena_tensor_count; i++) {
        T* tensor = &g_t[g_arena_tensor_indices[i]];
        tensor->data = (float*)replacements[i];
        tensor->owns = 1;
    }
    free(replacements);
    free(g_arena_tensor_indices);
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
    g_arena_tensor_indices = NULL;
    g_arena_tensor_count = 0;
    return 0;
}

static int t_index_by_name(const char* name) {
    T* t = t_find(name);
    return t ? (int)(t - g_t) : -1;
}

static void plan_memory_arena(void) {
    const char* en = getenv("VOLVOX_ARENA");
    if (en && en[0] && !strcmp(en, "0")) return;   // default-on; VOLVOX_ARENA=0 disables
    if (g_use_vulkan || g_use_opengl || g_use_metal || g_use_nnapi) return;
    int nt = g_nt, nn = g_nn;
    if (nt <= 0 || nn <= 0) return;

    int* prod = (int*)malloc(sizeof(int) * nt);
    int* last = (int*)malloc(sizeof(int) * nt);
    int* buf_of = (int*)malloc(sizeof(int) * nt);
    char* poolable = (char*)calloc(nt, 1);
    size_t* cap = (size_t*)malloc(sizeof(size_t) * nt);
    char* busy = (char*)calloc(nt, 1);
    void** bufs = (void**)calloc((size_t)nt, sizeof(void*));
    if (!prod || !last || !buf_of || !poolable || !cap || !busy || !bufs) {
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }
    for (int i = 0; i < nt; i++) { prod[i] = -1; last[i] = -1; buf_of[i] = -1; }

    for (int j = 0; j < nn; j++) {
        if (g_n[j].skip) continue;
        int ti = t_index_by_name(g_n[j].out);
        if (ti >= 0) prod[ti] = j;
    }
    for (int j = 0; j < nn; j++) {
        for (int k = 0; k < g_n[j].nin; k++) {
            int ti = t_index_by_name(g_n[j].ins[k].name);
            if (ti >= 0 && j > last[ti]) last[ti] = j;
        }
    }
    for (int i = 0; i < nt; i++) {
        T* t = &g_t[i];
        if (!(t->owns && (t->dtype == T_F32 || t->dtype == T_I8 || t->dtype == T_U8) &&
              t->numel > 0 && t->elem_size > 0 &&
              (size_t)t->numel <= SIZE_MAX / t->elem_size)) continue;
        if (prod[i] < 0 || last[i] < 0) continue;          // graph input/const or output/dead
        int aliased = 0;
        for (int u = 0; u < nt; u++) if (u != i && g_t[u].data == t->data) { aliased = 1; break; }
        if (aliased) continue;
        poolable[i] = 1;
    }

    // Phase 1: assign each poolable tensor a buffer INDEX, growing byte capacities.
    // (Buffers are allocated once after sizing so pointers never move — a tensor is written
    // every forward within its lifetime, so its data pointer must stay valid for good.)
    int nbufs = 0;
    size_t orig_bytes = 0;
    for (int i = 0; i < nn; i++) {
        for (int t = 0; t < nt; t++) {                     // tensors produced at node i
            if (!poolable[t] || prod[t] != i) continue;
            size_t need = (size_t)g_t[t].numel * g_t[t].elem_size;
            if (need > SIZE_MAX - orig_bytes) {
                free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
                return;
            }
            orig_bytes += need;
            int best = -1;                                  // best-fit among free buffers
            for (int b = 0; b < nbufs; b++)
                if (!busy[b] && cap[b] >= need && (best < 0 || cap[b] < cap[best])) best = b;
            if (best < 0) {                                 // none fit: grow largest free, else new
                for (int b = 0; b < nbufs; b++)
                    if (!busy[b] && (best < 0 || cap[b] > cap[best])) best = b;
                if (best < 0) { best = nbufs++; cap[best] = 0; }
                if (cap[best] < need) cap[best] = need;
            }
            busy[best] = 1;
            buf_of[t] = best;
        }
        for (int t = 0; t < nt; t++)                        // free tensors last-used at node i
            if (poolable[t] && last[t] == i && buf_of[t] >= 0) busy[buf_of[t]] = 0;
    }

    // Phase 2: allocate the arena buffers once at their final sizes.
    int ok = 1;
    for (int b = 0; b < nbufs; b++) {
        bufs[b] = malloc(cap[b] > 0 ? cap[b] : 1u);
        if (!bufs[b]) { ok = 0; break; }
    }
    if (!ok) {                                              // OOM: undo, keep original calloc buffers
        for (int b = 0; b < nbufs; b++) free(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }

    // Phase 3: point each pooled tensor at its buffer, dropping its own calloc buffer.
    int pooled_count = 0;
    for (int t = 0; t < nt; t++) if (poolable[t] && buf_of[t] >= 0) pooled_count++;
    int* arena_tensor_indices = pooled_count > 0 ? (int*)malloc((size_t)pooled_count * sizeof(int)) : NULL;
    if (pooled_count > 0 && !arena_tensor_indices) {
        for (int b = 0; b < nbufs; b++) free(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }
    int pooled_index = 0;
    for (int t = 0; t < nt; t++) {
        if (!poolable[t] || buf_of[t] < 0) continue;
        free(g_t[t].data);
        g_t[t].data = (float*)bufs[buf_of[t]];
        g_t[t].owns = 0;
        arena_tensor_indices[pooled_index++] = t;
    }

    size_t arena_bytes = 0;
    for (int b = 0; b < nbufs; b++) {
        if (cap[b] > SIZE_MAX - arena_bytes) {
            /* The allocations are already valid; keep the plan and avoid an
             * overflowing debug-only total. */
            arena_bytes = SIZE_MAX;
            break;
        }
        arena_bytes += cap[b];
    }
    g_arena_bufs = bufs;
    g_arena_nbufs = nbufs;
    g_arena_tensor_indices = arena_tensor_indices;
    g_arena_tensor_count = pooled_count;
    if (g_debug)
        fprintf(stderr, "[debug] arena_plan pooled=%.1f MB -> arena=%.1f MB (%d buffers)\n",
                (double)orig_bytes / 1048576.0, (double)arena_bytes / 1048576.0, nbufs);

    free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy);
}

int volvoxai_engine_prepare_tensor_table_mutation(void) {
    return restore_arena_tensors();
}

void volvoxai_engine_finish_tensor_table_mutation(void) {
    /* Structural callers hold both locks until this boundary and may publish
     * a rebuilt immutable index. Non-structural arena restoration reaches the
     * same hook with an already-current index, making this a read-only no-op. */
    volvoxai_engine_tensor_name_index_rebuild();
    plan_memory_arena();
}
