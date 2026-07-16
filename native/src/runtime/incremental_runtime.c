#include "incremental_runtime.h"

#include "adapter_runtime_internal.h"
#include "backend_manager.h"
#include "backend_sdk.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The ordinary forward path remains stateless. These tensors and flags belong
 * only to the opt-in dependency scheduler. */
static unsigned char g_dirty[MAXT];
static int g_cache_valid;
static int g_arena_detached;
static int g_hybrid_row_active;
static unsigned char g_hybrid_row_nodes[MAXN];
static int g_hybrid_prepared_row = -1;

/* Resolve graph references once per model generation.  The persisted graph
 * spelling remains name based, but dependency scheduling is a hot execution
 * path and must not linearly scan the tensor table for every edge/token. */
typedef struct {
    uint64_t generation;
    int tensor_count;
    int node_count;
    int input_indices[MAXN][MAXIN];
    int output_indices[MAXN][MAXIN + 1];
    unsigned char input_counts[MAXN];
    unsigned char output_counts[MAXN];
    int valid;
} VxIncrementalPlan;

static VxIncrementalPlan g_incremental_plan;

static int incremental_plan_build_locked(void) {
    const uint64_t generation = volvoxai_engine_model_generation_locked();
    if (g_incremental_plan.valid &&
        g_incremental_plan.generation == generation &&
        g_incremental_plan.tensor_count == g_nt &&
        g_incremental_plan.node_count == g_nn) return 0;
    memset(&g_incremental_plan, 0, sizeof(g_incremental_plan));
    if (g_nt < 0 || g_nt > MAXT || g_nn < 0 || g_nn > MAXN) return -1;
    for (int node = 0; node < g_nn; node++) {
        Node* value = &g_n[node];
        int output_count = 0;
        if (value->nin < 0 || value->nin > MAXIN ||
            value->nout < 0 || value->nout > MAXIN) return -1;
        g_incremental_plan.input_counts[node] = (unsigned char)value->nin;
        for (int input = 0; input < value->nin; input++) {
            T* tensor = t_find(value->ins[input].name);
            ptrdiff_t index = tensor ? tensor - g_t : -1;
            if (index < 0 || index >= g_nt) return -1;
            g_incremental_plan.input_indices[node][input] = (int)index;
        }
        for (int output = 0; output < value->nout; output++) {
            T* tensor = t_find(value->outs[output].name);
            ptrdiff_t index = tensor ? tensor - g_t : -1;
            int duplicate = 0;
            if (index < 0 || index >= g_nt) return -1;
            for (int prior = 0; prior < output_count; prior++)
                if (g_incremental_plan.output_indices[node][prior] == (int)index)
                    duplicate = 1;
            if (!duplicate)
                g_incremental_plan.output_indices[node][output_count++] = (int)index;
        }
        {
            T* tensor = t_find(value->out);
            ptrdiff_t index = tensor ? tensor - g_t : -1;
            int duplicate = 0;
            if (index < 0 || index >= g_nt) return -1;
            for (int prior = 0; prior < output_count; prior++)
                if (g_incremental_plan.output_indices[node][prior] == (int)index)
                    duplicate = 1;
            if (!duplicate)
                g_incremental_plan.output_indices[node][output_count++] = (int)index;
        }
        g_incremental_plan.output_counts[node] = (unsigned char)output_count;
    }
    g_incremental_plan.generation = generation;
    g_incremental_plan.tensor_count = g_nt;
    g_incremental_plan.node_count = g_nn;
    g_incremental_plan.valid = 1;
    return 0;
}

void vx_incremental_invalidate_locked(void) {
    memset(g_dirty, 0, sizeof(g_dirty));
    memset(g_hybrid_row_nodes, 0, sizeof(g_hybrid_row_nodes));
    g_cache_valid = 0;
    g_hybrid_row_active = 0;
    g_hybrid_prepared_row = -1;
    /* DecodeSession is a stateful view of this single global cache. Any
     * legacy execution, model mutation, or failed incremental run that drops
     * the cache must also make the view require a fresh seed. */
    vx_decode_session_invalidate_cache_locked();
}

static void restore_arena_locked(void) {
    if (!g_arena_detached) return;
    volvoxai_engine_finish_tensor_table_mutation();
    g_arena_detached = 0;
    /* Arena planning changes activation host addresses. Device slot tables
     * are keyed by those addresses and must be rebuilt with the new plan. */
    vx_runtime_backend_reset();
}

void vx_incremental_prepare_ordinary_locked(void) {
    vx_incremental_invalidate_locked();
    restore_arena_locked();
}

void vx_incremental_mark_tensor_locked(T* tensor) {
    ptrdiff_t index;
    if (!tensor) return;
    index = tensor - g_t;
    if (index >= 0 && index < g_nt) {
        g_dirty[index] = 1;
        g_qt[index].valid = 0;
        g_hybrid_prepared_row = -1;
    }
}

static int node_has_dirty_input(int node) {
    if (node < 0 || node >= g_incremental_plan.node_count) return 0;
    for (int input = 0; input < g_incremental_plan.input_counts[node]; input++)
        if (g_dirty[g_incremental_plan.input_indices[node][input]]) return 1;
    return 0;
}

static void prepare_dirty_outputs(int node) {
    if (node < 0 || node >= g_incremental_plan.node_count) return;
    for (int output = 0; output < g_incremental_plan.output_counts[node]; output++) {
        int index = g_incremental_plan.output_indices[node][output];
        g_dirty[index] = 1;
        g_qt[index].valid = 0;
    }
}

static int run_has_adapter_targets(void) {
    for (int node = 0; node < g_nn; node++) {
        for (int input = 0; input < g_n[node].nin; input++) {
            if (vx_adapter_run_has_target(g_n[node].ins[input].name)) return 1;
        }
    }
    return 0;
}

static int refresh_weight_caches_if_dirty(void) {
    if (!g_weight_caches_dirty) return 0;
    return volvoxai_engine_refresh_weight_caches();
}

typedef struct {
    unsigned char tensors[MAXT];
    unsigned char nodes[MAXN];
    unsigned char boundary_inputs[MAXT];
    size_t prefix_bytes[MAXT];
    int node_count;
} VxHybridRowPlan;

static int hybrid_row_disabled(void) {
    const char* value = getenv("VOLVOXAI_DISABLE_GPU_CPU_ROW");
    return value && value[0] && strcmp(value, "0");
}

static int hybrid_device_backend_selected(void) {
    VolvoxAIEngineBackend backend;
    if (vx_sdk_selected_backend() || !vx_runtime_backend_has_graph() ||
        hybrid_row_disabled()) return 0;
    backend = vx_backend_manager_current();
    return backend == VOLVOXAI_BACKEND_VULKAN ||
        backend == VOLVOXAI_BACKEND_OPENGL ||
        backend == VOLVOXAI_BACKEND_METAL;
}

static int hybrid_tensor_prefix_bytes(const T* tensor, int row, size_t* bytes_out) {
    long rows;
    long row_elements;
    size_t elements;
    if (!tensor || !bytes_out || row < 0 || tensor->ndim < 2 ||
        tensor->shape[0] != 1 || tensor->shape[1] <= row ||
        tensor->shape[1] <= 0 || tensor->numel <= 0 || !tensor->elem_size)
        return -1;
    rows = tensor->shape[1];
    if (tensor->numel % rows) return -1;
    row_elements = tensor->numel / rows;
    if (row_elements <= 0 || (size_t)row > SIZE_MAX / (size_t)row_elements)
        return -1;
    elements = (size_t)row * (size_t)row_elements;
    if (elements > SIZE_MAX / tensor->elem_size) return -1;
    *bytes_out = elements * tensor->elem_size;
    return 0;
}

static int hybrid_node_input_index(const Node* node, const char* key) {
    if (!node || !key) return -1;
    for (int input = 0; input < node->nin; input++) {
        if (!strcmp(node->ins[input].key, key))
            return g_incremental_plan.input_indices[node - g_n][input];
    }
    return -1;
}

static int hybrid_qsdpa_dirty_kv_supported(const Node* node,
                                           const VxHybridRowPlan* plan) {
    const cJSON* causal;
    int k_index;
    int v_index;
    if (!node || !plan || strcmp(node->op, "QSDPA")) return 1;
    causal = node->params ? cJSON_GetObjectItem(node->params, "causal") : NULL;
    if (causal && cJSON_IsTrue(causal)) return 1;
    k_index = hybrid_node_input_index(node, "k");
    v_index = hybrid_node_input_index(node, "v");
    return k_index >= 0 && k_index < g_nt && v_index >= 0 && v_index < g_nt &&
        !plan->tensors[k_index] && !plan->tensors[v_index];
}

static int hybrid_row_plan_expand_locked(int row, VxHybridRowPlan* plan) {
    if (!plan) return -1;
    for (int node = 0; node < g_nn; node++) {
        int should_run = 0;
        for (int input = 0; input < g_incremental_plan.input_counts[node]; input++) {
            if (plan->tensors[g_incremental_plan.input_indices[node][input]]) {
                should_run = 1;
                break;
            }
        }
        if (!should_run) continue;
        if (!vx_runtime_node_incremental_row_compatible(&g_n[node], node, row)) return 0;
        plan->nodes[node] = 1;
        plan->node_count++;
        for (int output = 0; output < g_incremental_plan.output_counts[node]; output++)
            plan->tensors[g_incremental_plan.output_indices[node][output]] = 1;
    }
    if (!plan->node_count) return 0;

    /* Noncausal attention is row-safe only when its memory K/V is outside the
     * dirty decoder closure. Causal self-attention deliberately consumes the
     * synchronized prefix produced by dirty row-linear nodes. */
    for (int node = 0; node < g_nn; node++) {
        if (plan->nodes[node] &&
            !hybrid_qsdpa_dirty_kv_supported(&g_n[node], plan)) return 0;
    }

    for (int node = 0; node < g_nn; node++) {
        if (!plan->nodes[node]) continue;
        for (int input = 0; input < g_incremental_plan.input_counts[node]; input++) {
            int index = g_incremental_plan.input_indices[node][input];
            T* tensor = &g_t[index];
            if (!plan->tensors[index] && !tensor->is_graph_input &&
                !volvoxai_engine_tensor_is_model_weight_locked(tensor->name))
                plan->boundary_inputs[index] = 1;
        }
        for (int output = 0; output < g_incremental_plan.output_counts[node]; output++) {
            int index = g_incremental_plan.output_indices[node][output];
            if (hybrid_tensor_prefix_bytes(&g_t[index], row,
                                           &plan->prefix_bytes[index]) != 0) return 0;
        }
    }
    return 1;
}

static int hybrid_row_plan_build_locked(int row, VxHybridRowPlan* plan) {
    if (!plan || incremental_plan_build_locked() != 0) return -1;
    memset(plan, 0, sizeof(*plan));
    memcpy(plan->tensors, g_dirty, (size_t)g_nt);
    return hybrid_row_plan_expand_locked(row, plan);
}

static int hybrid_model_has_row_closure_locked(void) {
    VxHybridRowPlan plan;
    if (incremental_plan_build_locked() != 0) return 0;
    /* Decode inputs are fixed-batch sequences. Probe each independently so
     * image/question inputs used only for the seed do not make a compatible
     * token-input closure appear unsupported. The actual changed closure is
     * checked again, atomically, before ownership is transferred. */
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        if (!tensor->is_graph_input || tensor->ndim != 2 ||
            tensor->shape[0] != 1 || tensor->shape[1] <= 1) continue;
        memset(&plan, 0, sizeof(plan));
        plan.tensors[index] = 1;
        if (hybrid_row_plan_expand_locked(1, &plan) == 1) return 1;
    }
    return 0;
}

int vx_incremental_hybrid_row_active_locked(void) {
    return g_hybrid_row_active;
}

int vx_incremental_prepare_hybrid_row_locked(int row) {
    VxHybridRowPlan plan;
    size_t boundary_bytes = 0;
    size_t prefix_bytes = 0;
    int was_active = g_hybrid_row_active;
    int status;
    if (!g_loaded || row < 0) return -1;
    if (!vx_runtime_backend_has_graph()) return 1;
    /* A device seed does not provide an older prefix at row zero. Keep the
     * public row API fail-closed there instead of transferring ownership with
     * device-dirty outputs whose current row has not been synchronized. */
    if (row < 1) return 0;
    if (!was_active && !hybrid_device_backend_selected()) return 0;
    if (!g_cache_valid || g_weight_caches_dirty) return -1;
    status = hybrid_row_plan_build_locked(row, &plan);
    /* Before the one-way handoff, incompatibility is side-effect free and an
     * AUTO session may retain its GPU seed in dependency mode. Afterwards the
     * device graph is stale, so a newly unsupported dirty closure must fail
     * closed rather than attempting to re-enter it. */
    if (status <= 0) return was_active && status == 0 ? -1 : status;
    if (was_active) {
        /* Partial prefix readback discards the future seed rows for this
         * closure. A different (even individually compatible) closure could
         * therefore consume device-only prefix state. Pin the exact node set
         * transferred by the first row and fail closed on a later change. */
        if (memcmp(plan.nodes, g_hybrid_row_nodes,
                   (size_t)g_nn * sizeof(plan.nodes[0]))) return -1;
        g_hybrid_prepared_row = row;
        return 1;
    }

    /* Validate the whole closure before changing any backend ownership. A
     * compatibility miss can therefore fall back to ordinary device
     * dependency execution without rebuilding the seed. */
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        size_t bytes;
        if (!plan.boundary_inputs[index]) continue;
        bytes = (size_t)tensor->numel * tensor->elem_size;
        if (vk_sync_host_tensor(tensor) != 0) return -1;
        boundary_bytes += bytes;
    }
    for (int index = 0; index < g_nt; index++) {
        size_t bytes = plan.prefix_bytes[index];
        if (!bytes) continue;
        if (vx_runtime_backend_sync_host(g_t[index].data, bytes, 0) != 0) return -1;
        prefix_bytes += bytes;
    }
    memcpy(g_hybrid_row_nodes, plan.nodes,
           (size_t)g_nn * sizeof(plan.nodes[0]));
    g_hybrid_row_active = 1;
    g_hybrid_prepared_row = row;
    if (g_debug) {
        fprintf(stderr,
                "[debug] incremental hybrid GPU-seed/CPU-row nodes=%d "
                "prefix=%.1f KB boundary=%.1f KB\n",
                plan.node_count, (double)prefix_bytes / 1024.0,
                (double)boundary_bytes / 1024.0);
    }
    return 1;
}

int vx_incremental_forward_locked(int row) {
    int rc = -1;
    int executed = 0;
    int skipped = 0;
    int force_full;
    int weights_changed;
    int backend_forward_started = 0;
    int previous_execution_row = g_execution_row;
    const int row_execution = row >= 0;
    int adapter_targets;
    int direct_cpu;
    if (!g_loaded || g_nn == 0 || row < -1) return -1;
    if (incremental_plan_build_locked() != 0) return -1;
    /* ABI v1 can execute a complete SDPA node, but it cannot publish or
     * restore the runtime-private K/V cache rows used by this scheduler.
     * Reject the row before any backend callback so even a host-output SDK
     * cannot accidentally seed one implementation and resume another. */
    if (row_execution && !vx_incremental_row_supported_locked()) {
        vx_incremental_invalidate_locked();
        return -1;
    }
    weights_changed = g_weight_caches_dirty;
    if (refresh_weight_caches_if_dirty() != 0) {
        if (row_execution) vx_incremental_invalidate_locked();
        return -1;
    }
    /* A row update is valid only after a complete dependency-cache seed. A
     * changed weight invalidates every retained operator-cache row. Built-in
     * device graphs may relinquish a validated prefix to the CPU exactly
     * once; public backends remain outside that private ownership contract. */
    if (row_execution && (!g_cache_valid || weights_changed)) {
        vx_incremental_invalidate_locked();
        return -1;
    }
    if (row_execution && vx_runtime_backend_has_graph() &&
        g_hybrid_prepared_row != row) {
        if (vx_incremental_prepare_hybrid_row_locked(row) != 1) {
            vx_incremental_invalidate_locked();
            return -1;
        }
    }
    if (row_execution) g_hybrid_prepared_row = -1;
    if (vx_adapter_run_begin() != 0) {
        if (row_execution) vx_incremental_invalidate_locked();
        return -1;
    }
    adapter_targets = run_has_adapter_targets();
    if (row_execution && adapter_targets) goto done;
    force_full = !g_cache_valid || weights_changed || adapter_targets;
    if (row_execution && force_full) goto done;
    if (force_full) g_hybrid_row_active = 0;
    /* Once a device prefix has been synchronized, direct CPU dispatch is the
     * ownership boundary: no row wrapper may accidentally re-enter the GPU
     * with whole-tensor semantics. Ordinary CPU sessions retain the existing
     * registry-elision shortcut; host-resident NNAPI still uses the registry. */
    direct_cpu = g_hybrid_row_active ||
        (vx_decode_session_active_locked() &&
         vx_incremental_row_supported_locked() &&
         vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);
    g_active_row = row;
    if (row_execution) g_execution_row = row;
    g_prefix_rows = 0;
    if (force_full) {
        /* The ordinary arena aliases buffers after their last use in one full
         * pass. Cached branches outlive that schedule, so give every planned
         * activation independent storage before producing the cache. */
        if (volvoxai_engine_prepare_tensor_table_mutation() != 0) goto done;
        g_arena_detached = 1;
        vx_runtime_backend_reset();
        qt_invalidate_all();
        if (vx_runtime_backend_has_graph()) {
            vx_runtime_backend_begin_forward();
            backend_forward_started = 1;
        }
        vk_mark_owned_tensors_host_dirty();
    } else if (vx_runtime_backend_has_graph() && !g_hybrid_row_active) {
        /* Preserve skipped device-resident tensors. Changed graph inputs were
         * individually marked host-dirty by set_input_raw(). */
        vx_runtime_backend_begin_forward();
        backend_forward_started = 1;
    }
    if (g_debug) prof_reset();
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    for (int node = 0; node < g_nn; node++) {
        int should_run = force_full || node_has_dirty_input(node);
        if (!should_run) {
            skipped++;
            continue;
        }
        /* Invalidate an old sidecar before the operator has a chance to
         * publish a replacement, then propagate dirtiness topologically. */
        prepare_dirty_outputs(node);
        if ((direct_cpu
                ? run_node_cpu_direct(&g_n[node], node, node == g_nn - 1)
                : run_node(&g_n[node], node, node == g_nn - 1)) != 0) goto done;
        executed++;
    }
    memset(g_dirty, 0, sizeof(g_dirty));
    g_cache_valid = 1;
    rc = 0;
done:
    if (backend_forward_started) {
        double wait_t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
        if (vx_runtime_backend_end_forward() != 0) rc = -1;
        if (g_debug) prof_add_entry("GPUWait", volvoxai_engine_now_ms() - wait_t0);
    }
    if (rc == 0 && g_debug) {
        fprintf(stderr,
                "[debug] volvoxai_engine_forward_incremental%s nodes=%d executed=%d cached=%d %.3f ms\n",
                row_execution ? "_row" : "", g_nn, executed, skipped,
                volvoxai_engine_now_ms() - t0);
        prof_report();
    }
    g_active_row = -1;
    g_execution_row = previous_execution_row;
    if (rc != 0) vx_incremental_invalidate_locked();
    vx_adapter_run_end();
    return rc;
}

int vx_incremental_row_supported_locked(void) {
    if (!g_loaded || vx_sdk_selected_backend()) return 0;
    if (!vx_runtime_backend_has_graph()) return 1;
    return hybrid_device_backend_selected() && hybrid_model_has_row_closure_locked();
}

void vx_incremental_shutdown_locked(void) {
    vx_incremental_invalidate_locked();
    /* The engine has already freed every tensor and arena allocation. */
    g_arena_detached = 0;
}
