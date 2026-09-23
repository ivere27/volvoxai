#include "profiling.h"
#include "incremental_runtime.h"
#include "vx_platform.h"

#include "adapter_runtime_internal.h"
#include "paged_binding.h"
#include "backend_manager.h"
#include "call_sequence_policy.h"

#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t incremental_selection_u32(const uint8_t* bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) |
           ((uint32_t)bytes[3] << 24u);
}

static int incremental_selection_fusion_owner(int candidate,
                                               int logical_peer) {
    int peer = -1;
    if (candidate < 0 || candidate >= g_nn ||
        logical_peer < 0 || logical_peer >= g_nn)
        return 0;
    if (g_node_fusion[candidate].id == GRAPH_FUSION_CONV_ADD ||
        g_node_fusion[candidate].id == GRAPH_FUSION_DEPTHWISE_POINTWISE ||
        g_node_fusion[candidate].id == GRAPH_FUSION_CHAINED_ELEMENTWISE)
        peer = g_node_fusion[candidate].peer_idx;
    if (peer == logical_peer ||
        g_concat_sigmoid_fuse[candidate] == logical_peer)
        return 1;
    /* ReLU6 predates GraphNodeFusion and records the rewrite on the producer
     * itself. Its final output name is the skipped Clip output. */
    return g_n[candidate].fuse_relu6 &&
        !strcmp(g_n[candidate].out, g_n[logical_peer].out) &&
        (g_n[logical_peer].operator_kind == VX_OP_CLIP);
}

static int incremental_selection_alias_skip(const Node* node) {
    return node &&
        ((node->operator_kind == VX_OP_RESHAPE) ||
         (node->operator_kind == VX_OP_FLATTEN) ||
         (node->operator_kind == VX_OP_SQUEEZE) ||
         (node->operator_kind == VX_OP_UNSQUEEZE) ||
         (node->operator_kind == VX_OP_DROPOUT) ||
         (node->operator_kind == VX_OP_IDENTITY) ||
         (node->operator_kind == VX_OP_DEQUANTIZE_LINEAR));
}

/* Map one selected logical node through every nested lowering owner.  The
 * portable policy remains the dependency authority; this is only the native
 * physical projection needed when an earlier fused kernel writes a skipped
 * peer's result. */
static int incremental_selection_physical_owner_locked(int logical_node,
                                                        int* out_owner) {
    int owner = logical_node;
    if (!out_owner || logical_node < 0 || logical_node >= g_nn) return -1;
    for (int depth = 0; depth <= g_nn; ++depth) {
        int found = -1;
        for (int candidate = 0; candidate < g_nn; ++candidate) {
            if (!incremental_selection_fusion_owner(candidate, owner))
                continue;
            if (found >= 0 && found != candidate) return -1;
            found = candidate;
        }
        if (found < 0) {
            *out_owner = owner;
            return 0;
        }
        /* Every supported fusion moves a later logical peer into an earlier
         * physical producer. Preserve every hop (Add -> pointwise ->
         * depthwise), not only the final owner: row boundary/tensor
         * bookkeeping still consumes the skipped intermediate declarations. */
        if (found >= owner) return -1;
        g_portable_selected_nodes[found] = 1u;
        owner = found;
    }
    return -1;
}

void vx_incremental_selection_clear_locked(void) {
    if (g_portable_selected_nodes && g_node_capacity)
        memset(g_portable_selected_nodes, 0, g_node_capacity);
    g_portable_selection_active = 0;
}

int vx_incremental_selection_install_locked(
        const uint8_t* response_v1, uint32_t response_v1_bytes) {
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t selected_node_count;
    uint32_t counted = 0u;
    uint64_t node_end;
    if (!response_v1 ||
        response_v1_bytes < sizeof(VxCallSequencePolicyResponseV1) ||
        (uintptr_t)response_v1 % 8u || !g_loaded || g_nn <= 0 ||
        (size_t)g_nn > g_node_capacity || !g_portable_selected_nodes ||
        !vx_engine_state_current()->portable_cpu_activation_plan_enabled ||
        !vx_engine_state_current()->portable_graph_plan_request_v1 ||
        !vx_engine_state_current()->portable_graph_plan_v1 ||
        g_portable_selection_active || !g_cache_valid ||
        g_weight_caches_dirty ||
        incremental_selection_u32(response_v1 + 0u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_RESPONSE_MAGIC ||
        incremental_selection_u32(response_v1 + 4u) !=
            VOLVOXAI_CALL_SEQUENCE_POLICY_ABI_VERSION ||
        (int32_t)incremental_selection_u32(response_v1 + 8u) !=
            VX_CALL_SEQUENCE_POLICY_STATUS_OK ||
        incremental_selection_u32(
            response_v1 + offsetof(
                VxCallSequencePolicyResponseV1, written_bytes)) !=
            response_v1_bytes ||
        incremental_selection_u32(
            response_v1 + offsetof(
                VxCallSequencePolicyResponseV1, policy_kind)) !=
            VX_CALL_SEQUENCE_POLICY_FIXED_INPUT_DEPENDENCY ||
        incremental_selection_u32(
            response_v1 + offsetof(
                VxCallSequencePolicyResponseV1, selection_kind)) !=
            VX_CALL_SEQUENCE_SELECTION_EXPLICIT)
        return -1;
    node_offset = incremental_selection_u32(
        response_v1 + offsetof(VxCallSequencePolicyResponseV1, node_offset));
    node_count = incremental_selection_u32(
        response_v1 + offsetof(VxCallSequencePolicyResponseV1, node_count));
    selected_node_count = incremental_selection_u32(
        response_v1 + offsetof(
            VxCallSequencePolicyResponseV1, selected_node_count));
    node_end = (uint64_t)node_offset +
        (uint64_t)node_count * sizeof(VxCallSequenceNodeV1);
    if (node_offset != sizeof(VxCallSequencePolicyResponseV1) ||
        node_count != (uint32_t)g_nn || node_end > response_v1_bytes) {
        vx_incremental_selection_clear_locked();
        return -1;
    }
    memset(g_portable_selected_nodes, 0, g_node_capacity);
    for (uint32_t logical = 0u; logical < node_count; ++logical) {
        const uint8_t* record = response_v1 + node_offset +
            (size_t)logical * sizeof(VxCallSequenceNodeV1);
        const uint32_t flags = incremental_selection_u32(record + 4u);
        if (incremental_selection_u32(record) != logical ||
            (flags & ~(uint32_t)VX_CALL_SEQUENCE_NODE_SELECTED)) {
            vx_incremental_selection_clear_locked();
            return -1;
        }
        if (flags & VX_CALL_SEQUENCE_NODE_SELECTED) {
            g_portable_selected_nodes[logical] = 1u;
            ++counted;
        }
    }
    if (counted != selected_node_count) {
        vx_incremental_selection_clear_locked();
        return -1;
    }
    for (int logical = 0; logical < g_nn; ++logical) {
        int owner;
        if (!g_portable_selected_nodes[logical]) continue;
        if (g_n[logical].disabled ||
            incremental_selection_physical_owner_locked(logical, &owner) != 0 ||
            owner < 0 || owner >= g_nn || g_n[owner].disabled) {
            vx_incremental_selection_clear_locked();
            return -1;
        }
        if (owner == logical && g_n[logical].skip &&
            !incremental_selection_alias_skip(&g_n[logical])) {
            vx_incremental_selection_clear_locked();
            return -1;
        }
        if (!g_n[owner].skip) g_portable_selected_nodes[owner] = 1u;
        else if (!incremental_selection_alias_skip(&g_n[owner])) {
            vx_incremental_selection_clear_locked();
            return -1;
        }
    }
    g_portable_selection_active = 1;
    return 0;
}


static int incremental_output_index(int node, int output) {
    const Node* value = &g_n[node];
    return output < value->nout ? value->outs[output].tensor_index : value->primary_output_index;
}

/* Resolve graph references once per model generation.  The persisted graph
 * spelling remains name based, but dependency scheduling is a hot execution
 * path and must not linearly scan the tensor table for every edge/token. */
static int incremental_plan_build_locked(void) {
    const uint64_t generation = volvoxai_engine_model_generation_locked();
    if (g_incremental_plan.valid &&
        g_incremental_plan.generation == generation &&
        g_incremental_plan.tensor_count == g_nt &&
        g_incremental_plan.node_count == g_nn) return 0;
    g_incremental_plan.valid = 0;
    g_incremental_plan.generation = 0;
    g_incremental_plan.tensor_count = 0;
    g_incremental_plan.node_count = 0;
    if (g_nt < 0 || g_nn < 0 || (size_t)g_nt > g_tensor_capacity ||
        (size_t)g_nn > g_incremental_plan.node_capacity) return -1;
    for (int node = 0; node < g_nn; node++) {
        Node* value = &g_n[node];
        if (value->nin < 0 || value->nin > value->input_capacity ||
            value->nout < 0 || value->nout > value->output_capacity) return -1;
        for (int input = 0; input < value->nin; input++) {
            T* tensor = t_find(value->ins[input].name);
            ptrdiff_t index = tensor ? tensor - g_t : -1;
            if (index < 0 || index >= g_nt) return -1;
            value->ins[input].tensor_index = (int)index;
        }
        T* primary = t_find(value->out);
        ptrdiff_t primary_index = primary ? primary - g_t : -1;
        if (primary_index < 0 || primary_index >= g_nt) return -1;
        value->primary_output_index = (int)primary_index;
        value->dependency_output_count = value->nout;
        int primary_declared = 0;
        for (int output = 0; output < value->nout; output++) {
            T* tensor = t_find(value->outs[output].name);
            ptrdiff_t index = tensor ? tensor - g_t : -1;
            if (index < 0 || index >= g_nt) return -1;
            value->outs[output].tensor_index = (int)index;
            if (index == primary_index) primary_declared = 1;
        }
        // Private fusions can redirect the primary write beyond declared ports.
        if (!primary_declared) {
            if (value->nout == INT_MAX) return -1;
            value->dependency_output_count++;
        }
    }
    g_incremental_plan.generation = generation;
    g_incremental_plan.tensor_count = g_nt;
    g_incremental_plan.node_count = g_nn;
    g_incremental_plan.valid = 1;
    return 0;
}

/* Compatibility is queried before the hybrid GPU-to-CPU ownership handoff,
 * while g_dirty contains only the changed roots. Mirror dependency propagation
 * through preceding nodes without allocating after resident admission. */
int vx_incremental_tensor_dirty_before_node_locked(
        const T* target, int node_index) {
    ptrdiff_t target_index;
    unsigned char* dirty;
    if (!target || !g_t) return 1;
    target_index = target - g_t;
    if (target_index < 0 || target_index >= g_nt ||
        node_index < 0 || node_index > g_nn || !g_incremental_plan.valid ||
        g_incremental_plan.tensor_count != g_nt ||
        g_incremental_plan.node_count != g_nn ||
        g_incremental_scratch.tensor_capacity < (size_t)g_nt ||
        !g_incremental_scratch.compatibility_dirty)
        return 1;
    if (g_incremental_plan.row_probe_tensors)
        return g_incremental_plan.row_probe_tensors[target_index] != 0;
    dirty = g_incremental_scratch.compatibility_dirty;
    memcpy(dirty, g_dirty, (size_t)g_nt);
    for (int node = 0; node < node_index; node++) {
        int should_run = g_portable_selection_active
            ? g_portable_selected_nodes[node] != 0 : 0;
        if (!g_portable_selection_active) {
            for (int input = 0;
                 input < g_n[node].nin; input++) {
                int index = g_n[node].ins[input].tensor_index;
                if (index < 0 || index >= g_nt) return 1;
                if (dirty[index]) should_run = 1;
            }
        }
        if (!should_run) continue;
        for (int output = 0;
             output < g_n[node].dependency_output_count; output++) {
            int index = incremental_output_index(node, output);
            if (index < 0 || index >= g_nt) return 1;
            dirty[index] = 1;
        }
    }
    return dirty[target_index] != 0;
}

void vx_incremental_invalidate_locked(void) {
    vx_incremental_selection_clear_locked();
    if (g_tensor_capacity)
        memset(g_dirty, 0, g_tensor_capacity * sizeof(g_dirty[0]));
    if (g_node_capacity)
        memset(g_hybrid_row_nodes, 0,
               g_node_capacity * sizeof(g_hybrid_row_nodes[0]));
    g_cache_valid = 0;
    g_hybrid_row_active = 0;
    g_hybrid_prepared_row = -1;
    /* DecodeSession is a stateful view of this context's cache. Any
     * execution, model mutation, or failed incremental run that drops
     * the cache must also make the view require a fresh prefill. */
    vx_decode_session_invalidate_cache_locked();
}

static void restore_arena_locked(void) {
    if (!g_arena_detached) return;
    volvoxai_engine_finish_tensor_table_mutation();
    g_arena_detached = 0;
    /* Arena planning changes activation host addresses. Device slot tables
     * are keyed by those addresses and must be rebuilt with the new plan. */
    vx_runtime_backend_reset_transients();
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
        g_hybrid_prepared_row = -1;
    }
}

static int node_has_dirty_input(int node) {
    if (node < 0 || node >= g_incremental_plan.node_count) return 0;
    for (int input = 0; input < g_n[node].nin; input++)
        if (g_dirty[g_n[node].ins[input].tensor_index]) return 1;
    return 0;
}

static void prepare_dirty_outputs(int node) {
    if (node < 0 || node >= g_incremental_plan.node_count) return;
    for (int output = 0; output < g_n[node].dependency_output_count; output++) {
        int index = incremental_output_index(node, output);
        g_dirty[index] = 1;
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
    unsigned char* tensors;
    unsigned char* nodes;
    unsigned char* boundary_inputs;
    size_t* prefix_bytes;
    int node_count;
} VxHybridRowPlan;

static void hybrid_row_plan_deinit(VxHybridRowPlan* plan) {
    if (!plan) return;
    memset(plan, 0, sizeof(*plan));
}

static void hybrid_row_plan_reset(VxHybridRowPlan* plan) {
    if (!plan) return;
    if (g_nt > 0) {
        memset(plan->tensors, 0, (size_t)g_nt);
        memset(plan->boundary_inputs, 0, (size_t)g_nt);
        memset(plan->prefix_bytes, 0,
               (size_t)g_nt * sizeof(plan->prefix_bytes[0]));
    }
    if (g_nn > 0) memset(plan->nodes, 0, (size_t)g_nn);
    plan->node_count = 0;
}

static int hybrid_row_plan_init(VxHybridRowPlan* plan) {
    if (!plan || g_nt < 0 || g_nn < 0 ||
        g_incremental_scratch.tensor_capacity < (size_t)g_nt ||
        g_incremental_scratch.node_capacity < (size_t)g_nn ||
        (g_nt > 0 && (!g_incremental_scratch.hybrid_tensors ||
                      !g_incremental_scratch.hybrid_boundary_inputs ||
                      !g_incremental_scratch.hybrid_prefix_bytes)) ||
        (g_nn > 0 && !g_incremental_scratch.hybrid_nodes)) return -1;
    memset(plan, 0, sizeof(*plan));
    plan->tensors = g_incremental_scratch.hybrid_tensors;
    plan->nodes = g_incremental_scratch.hybrid_nodes;
    plan->boundary_inputs = g_incremental_scratch.hybrid_boundary_inputs;
    plan->prefix_bytes = g_incremental_scratch.hybrid_prefix_bytes;
    hybrid_row_plan_reset(plan);
    return 0;
}

static int hybrid_row_disabled(void) {
    const char* value = vx_engine_env("VOLVOXAI_DISABLE_GPU_CPU_ROW");
    return value && value[0] && strcmp(value, "0");
}

static int hybrid_device_backend_selected(void) {
    VxBackendKind backend;
    if (!vx_runtime_backend_has_graph() || hybrid_row_disabled()) return 0;
    backend = vx_backend_manager_current();
    return backend == VX_BACKEND_KIND_VULKAN ||
        backend == VX_BACKEND_KIND_OPENGL ||
        backend == VX_BACKEND_KIND_METAL;
}

/* A backend that executes rows on the device never relinquishes its prefix to
 * the host, so the one-way GPU-prefill/CPU-row handoff does not apply to it. */
static int device_row_backend_selected(void) {
    return vx_runtime_backend_has_graph() && vx_runtime_backend_has_device_rows();
}

static int cuda_resident_backend_selected(void) {
#if VOLVOXAI_ENABLE_CUDA
    return vx_runtime_backend_has_graph() &&
        vx_backend_manager_current() == VX_BACKEND_KIND_CUDA;
#else
    return 0;
#endif
}

/* Bytes of a sequence tensor that precede `row`.
 *
 * This used to require shape[0] == 1 and read the token axis from shape[1],
 * which is only the [1,S,D] spelling.  A decoder whose tensors are the equally
 * canonical sequence-major [S,D] therefore failed here — silently, since this
 * refusal carries no diagnostic — after every node in the closure had already
 * proved row-compatible.  vx_incremental_row_extent already answers the same
 * question for both spellings, so it answers it here too. */
static int hybrid_tensor_prefix_bytes(const T* tensor, int row, size_t* bytes_out) {
    long row_elements = 0;
    int rows;
    size_t elements;
    if (!tensor || !bytes_out || row < 0 || tensor->numel <= 0 ||
        !tensor->elem_size || tensor->ndim < 1) return -1;
    rows = vx_incremental_row_extent(tensor, &row_elements);
    if (rows <= row || row_elements <= 0) return -1;
    if ((size_t)row > SIZE_MAX / (size_t)row_elements) return -1;
    elements = (size_t)row * (size_t)row_elements;
    if (elements > SIZE_MAX / tensor->elem_size) return -1;
    *bytes_out = elements * tensor->elem_size;
    return 0;
}

static int hybrid_node_input_index(const Node* node, VxPortKind key) {
    if (!node || !key) return -1;
    for (int input = 0; input < node->nin; input++) {
        if (node->ins[input].port == key)
            return node->ins[input].tensor_index;
    }
    return -1;
}

static int hybrid_qsdpa_dirty_kv_supported(const Node* node,
                                           const VxHybridRowPlan* plan) {
    int k_index;
    int v_index;
    if (!node || !plan || (node->operator_kind != VX_OP_Q_SDPA)) return 1;
    if (vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0)) return 1;
    k_index = hybrid_node_input_index(node, VX_PORT_K);
    v_index = hybrid_node_input_index(node, VX_PORT_V);
    return k_index >= 0 && k_index < g_nt && v_index >= 0 && v_index < g_nt &&
        !plan->tensors[k_index] && !plan->tensors[v_index];
}

static int hybrid_row_plan_expand_locked(int row, VxHybridRowPlan* plan) {
    if (!plan) return -1;
    for (int node = 0; node < g_nn; node++) {
        int should_run = g_portable_selection_active
            ? g_portable_selected_nodes[node] != 0 : 0;
        if (!g_portable_selection_active) {
            for (int input = 0;
                 input < g_n[node].nin; input++) {
                if (plan->tensors[
                        g_n[node].ins[input].tensor_index]) {
                    should_run = 1;
                    break;
                }
            }
        }
        if (!should_run) continue;
        const unsigned char* previous_probe = g_incremental_plan.row_probe_tensors;
        g_incremental_plan.row_probe_tensors = plan->tensors;
        int compatible = vx_runtime_node_incremental_row_compatible(&g_n[node], node, row);
        g_incremental_plan.row_probe_tensors = previous_probe;
        if (!compatible) {
            /* Naming the first refusal is what makes an O(T) decode diagnosable;
             * the plan is discarded either way. */
            if (vx_engine_env("VOLVOXAI_ROW_DEBUG"))
                vx_engine_log("[row] node %d op=%s blocks row execution\n",
                        node, vx_operator_kind_name(g_n[node].operator_kind));
            return 0;
        }
        plan->nodes[node] = 1;
        plan->node_count++;
        for (int output = 0; output < g_n[node].dependency_output_count; output++)
            plan->tensors[incremental_output_index(node, output)] = 1;
    }
    if (!plan->node_count) {
        if (g_portable_selection_active) return 1;
        if (vx_engine_env("VOLVOXAI_ROW_DEBUG"))
            vx_engine_log("[row] no dirty node reaches row execution\n");
        return 0;
    }

    /* Noncausal attention is row-safe only when its memory K/V is outside the
     * dirty decoder closure. Causal self-attention deliberately consumes the
     * synchronized prefix produced by dirty row-linear nodes. */
    for (int node = 0; node < g_nn; node++) {
        if (plan->nodes[node] &&
            !hybrid_qsdpa_dirty_kv_supported(&g_n[node], plan)) {
            if (vx_engine_env("VOLVOXAI_ROW_DEBUG"))
                vx_engine_log(
                        "[row] node %d op=%s has dirty memory K/V\n",
                        node, vx_operator_kind_name(g_n[node].operator_kind));
            return 0;
        }
    }

    for (int node = 0; node < g_nn; node++) {
        if (!plan->nodes[node]) continue;
        for (int input = 0; input < g_n[node].nin; input++) {
            int index = g_n[node].ins[input].tensor_index;
            T* tensor = &g_t[index];
            if (!plan->tensors[index] && !tensor->is_graph_input &&
                !volvoxai_engine_tensor_is_model_weight_locked(tensor->name))
                plan->boundary_inputs[index] = 1;
        }
        for (int output = 0; output < g_n[node].dependency_output_count; output++) {
            int index = incremental_output_index(node, output);
            if (hybrid_tensor_prefix_bytes(&g_t[index], row,
                                           &plan->prefix_bytes[index]) != 0) {
                if (vx_engine_env("VOLVOXAI_ROW_DEBUG"))
                    vx_engine_log(
                            "[row] node %d op=%s output %s has no row prefix\n",
                            node, vx_operator_kind_name(g_n[node].operator_kind), g_t[index].name);
                return 0;
            }
        }
    }
    return 1;
}

static int hybrid_row_plan_build_locked(int row, VxHybridRowPlan* plan) {
    if (!plan || incremental_plan_build_locked() != 0) return -1;
    hybrid_row_plan_reset(plan);
    memcpy(plan->tensors, g_dirty, (size_t)g_nt);
    return hybrid_row_plan_expand_locked(row, plan);
}

static int hybrid_model_has_row_closure_locked(void) {
    VxHybridRowPlan plan = {0};
    int supported = 0;
    if (incremental_plan_build_locked() != 0) return 0;
    if (hybrid_row_plan_init(&plan) != 0) return 0;
    /* Decode inputs are fixed-batch sequences. Probe each independently so
     * image/question inputs used only for the prefill do not make a compatible
     * token-input closure appear unsupported. The actual changed closure is
     * checked again, atomically, before ownership is transferred. */
    /* The batch axis is the caller's declaration, so a `[lanes,S]` token input
     * is a decode input exactly when the step says it has that many lanes. One
     * is the ordinary case and stays first, which keeps a context that never
     * declares a batch reading precisely as it did. */
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        long width = 0;
        if (!tensor->is_graph_input || tensor->ndim < 2 ||
            vx_incremental_row_extent(tensor, &width) <= 1 || width <= 0) continue;
        hybrid_row_plan_reset(&plan);
        plan.tensors[index] = 1;
        if (hybrid_row_plan_expand_locked(1, &plan) == 1) {
            supported = 1;
            break;
        }
    }
    hybrid_row_plan_deinit(&plan);
    return supported;
}

int vx_incremental_hybrid_row_active_locked(void) {
    return g_hybrid_row_active;
}

int vx_incremental_prepare_hybrid_row_locked(int row) {
    VxHybridRowPlan plan = {0};
    size_t boundary_bytes = 0;
    size_t prefix_bytes = 0;
    int was_active = g_hybrid_row_active;
    int status;
    if (!g_loaded || row < 0) return -1;
    if (hybrid_row_plan_init(&plan) != 0) return -1;
#define VX_HYBRID_RETURN(value) \
    do { int vx_hybrid_result = (value); hybrid_row_plan_deinit(&plan); \
         return vx_hybrid_result; } while (0)
    if (!vx_runtime_backend_has_graph()) {
        if (!g_cache_valid || g_weight_caches_dirty) VX_HYBRID_RETURN(-1);
        VX_HYBRID_RETURN(hybrid_row_plan_build_locked(row, &plan));
    }
    /* A device-row backend still has to prove the closure is row-compatible --
     * the operators narrow themselves the same way either way -- but there is
     * no ownership to transfer, so the readback below is skipped entirely.
     * This is the whole difference between "decode runs on the GPU" and
     * "decode runs on the CPU that the GPU prefilled". */
    if (device_row_backend_selected()) {
        if (!g_cache_valid || g_weight_caches_dirty) VX_HYBRID_RETURN(-1);
        status = hybrid_row_plan_build_locked(row, &plan);
        if (status == 1) g_hybrid_prepared_row = row;
        VX_HYBRID_RETURN(status);
    }
    /* CPU handoff needs an older synchronized prefix. A device-owned row
     * above can start at zero when a cache lane is recycled. */
    if (row < 1) VX_HYBRID_RETURN(0);
    if (!was_active && !hybrid_device_backend_selected()) VX_HYBRID_RETURN(0);
    if (!g_cache_valid || g_weight_caches_dirty) VX_HYBRID_RETURN(-1);
    status = hybrid_row_plan_build_locked(row, &plan);
    /* Before the one-way handoff, incompatibility is side-effect free and an
     * AUTO session may retain its GPU prefill in dependency mode. Afterwards the
     * device graph is stale, so a newly unsupported dirty closure must fail
     * closed rather than attempting to re-enter it. */
    if (status <= 0) VX_HYBRID_RETURN(was_active && status == 0 ? -1 : status);
    if (was_active) {
        /* Partial prefix readback discards the future prefill rows for this
         * closure. A different (even individually compatible) closure could
         * therefore consume device-only prefix state. Pin the exact node set
         * transferred by the first row and fail closed on a later change. */
        if (memcmp(plan.nodes, g_hybrid_row_nodes,
                   (size_t)g_nn * sizeof(plan.nodes[0]))) VX_HYBRID_RETURN(-1);
        g_hybrid_prepared_row = row;
        VX_HYBRID_RETURN(1);
    }

    /* Validate the whole closure before changing any backend ownership. A
     * compatibility miss can therefore fall back to ordinary device
     * dependency execution without rebuilding the prefill. */
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        size_t bytes;
        if (!plan.boundary_inputs[index]) continue;
        bytes = (size_t)tensor->numel * tensor->elem_size;
        if (vk_sync_host_tensor(tensor) != 0) VX_HYBRID_RETURN(-1);
        boundary_bytes += bytes;
    }
    for (int index = 0; index < g_nt; index++) {
        size_t bytes = plan.prefix_bytes[index];
        if (!bytes) continue;
        if (vx_runtime_backend_sync_host(g_t[index].data, bytes, 0) != 0)
            VX_HYBRID_RETURN(-1);
        prefix_bytes += bytes;
    }
    memcpy(g_hybrid_row_nodes, plan.nodes,
           (size_t)g_nn * sizeof(plan.nodes[0]));
    g_hybrid_row_active = 1;
    g_hybrid_prepared_row = row;
    if (g_debug) {
        vx_engine_log(
                "[debug] incremental hybrid GPU-prefill/CPU-row nodes=%d "
                "prefix=%.1f KB boundary=%.1f KB\n",
                plan.node_count, (double)prefix_bytes / 1024.0,
                (double)boundary_bytes / 1024.0);
    }
    VX_HYBRID_RETURN(1);
#undef VX_HYBRID_RETURN
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
     * cannot accidentally prefill one implementation and resume another. */
    if (row_execution && !vx_incremental_row_supported_locked()) {
        vx_incremental_invalidate_locked();
        return -1;
    }
    weights_changed = g_weight_caches_dirty;
    if (g_portable_selection_active && weights_changed) {
        vx_incremental_invalidate_locked();
        return -1;
    }
    if (refresh_weight_caches_if_dirty() != 0) {
        if (row_execution) vx_incremental_invalidate_locked();
        return -1;
    }
    /* A row update is valid only after a complete dependency-cache prefill. A
     * changed weight invalidates every retained operator-cache row. Vulkan,
     * OpenGL, and Metal may relinquish a validated prefix to the CPU exactly
     * once; CUDA keeps the changed closure resident and launches row kernels. */
    if (row_execution && (!g_cache_valid || weights_changed)) {
        vx_incremental_invalidate_locked();
        return -1;
    }
    if (row_execution && vx_runtime_backend_has_graph() &&
        !device_row_backend_selected() &&
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
    /* Public FIXED evidence is authoritative for this step. Never silently
     * widen it to a legacy full refresh when mutable state outside that policy
     * (cache, weights, or adapter routing) would require more nodes. */
    if (g_portable_selection_active &&
        (!g_cache_valid || weights_changed || adapter_targets))
        goto done;
    force_full = !g_cache_valid || weights_changed || adapter_targets;
    if (row_execution && force_full) goto done;
    if (force_full) g_hybrid_row_active = 0;
    /* Once a device prefix has been synchronized, direct CPU dispatch is the
     * ownership boundary: no row wrapper may accidentally re-enter the GPU
     * with whole-tensor semantics. Ordinary CPU sessions retain the existing
     * registry-elision shortcut. */
    direct_cpu = g_hybrid_row_active ||
        (vx_decode_session_active_locked() &&
         vx_incremental_row_supported_locked() &&
         vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);
    g_active_row = row;
    if (row_execution) g_execution_row = row;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    if (force_full) {
        /* The ordinary arena aliases buffers after their last use in one full
         * pass. Cached branches outlive that schedule, so give every planned
         * activation independent storage before producing the cache. */
        if (volvoxai_engine_prepare_tensor_table_mutation() != 0) goto done;
        g_arena_detached = 1;
        vx_runtime_backend_reset_transients();
        if (vx_runtime_backend_has_graph()) {
            vx_runtime_backend_begin_forward(
                0, volvoxai_engine_model_generation_locked());
            backend_forward_started = 1;
        }
        vk_mark_owned_tensors_host_dirty();
    } else if (vx_runtime_backend_has_graph() && !g_hybrid_row_active) {
        /* Preserve skipped device-resident tensors. Changed graph inputs were
         * individually marked host-dirty by set_input_raw(). */
        vx_runtime_backend_begin_forward(
            0, volvoxai_engine_model_generation_locked());
        backend_forward_started = 1;
    }
    /*
     * The paged domain, checked once for the step rather than per node.
     *
     * Every operator in the row path computes its offsets as `row * width`,
     * which is correct only for a tensor whose logical order is its physical
     * order. An operator that touches a paged tensor without going through
     * `paged_binding.h` therefore reads the wrong slot and produces a decoder
     * that is wrong and looks plausible -- the failure this whole surface is
     * arranged to make impossible.
     *
     * `vx_paged_domain_supported_locked` was written for exactly that and had
     * no caller: the paged set was narrow enough that reasoning about it by
     * hand still worked. It stopped being narrow when the W8A8 operators
     * joined, so the check is enforced here instead of remembered.
     */
    if (vx_paged_bound_locked() && !vx_paged_domain_supported_locked(
            !force_full && g_portable_selection_active
                ? g_portable_selected_nodes : NULL)) {
        rc = -1;
        goto done;
    }
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    if (vx_trace_nodes(vx_engine_state_current()->profiling)) {
#define VX_DISPATCH_NODE(node) vx_run_node_profiled(&g_n[node], node, node == g_nn - 1, direct_cpu)
#include "incremental_execute_nodes.inc"
#undef VX_DISPATCH_NODE
    } else {
#define VX_DISPATCH_NODE(node) (direct_cpu ? run_node_cpu_direct(&g_n[node], node, node == g_nn - 1) : run_node(&g_n[node], node, node == g_nn - 1))
#include "incremental_execute_nodes.inc"
#undef VX_DISPATCH_NODE
    }
    if (g_tensor_capacity)
        memset(g_dirty, 0, g_tensor_capacity * sizeof(g_dirty[0]));
    g_cache_valid = 1;
    rc = 0;
done:
    if (backend_forward_started) {
        if (vx_runtime_backend_end_forward(rc == 0) != 0) rc = -1;
    }
    if (rc == 0 && g_debug) {
        vx_engine_log(
                "[debug] volvoxai_engine_forward_incremental%s nodes=%d executed=%d cached=%d %.3f ms\n",
                row_execution ? "_row" : "", g_nn, executed, skipped,
                volvoxai_engine_now_ms() - t0);
    }
    g_active_row = -1;
    g_execution_row = previous_execution_row;
    if (rc != 0) vx_incremental_invalidate_locked();
    vx_adapter_run_end();
    return rc;
}

int vx_incremental_row_supported_locked(void) {
    if (!g_loaded) return 0;
    if (!vx_runtime_backend_has_graph()) return 1;
    if (device_row_backend_selected())
        return hybrid_model_has_row_closure_locked();
    return hybrid_device_backend_selected() && hybrid_model_has_row_closure_locked();
}

void vx_incremental_shutdown_locked(void) {
    vx_incremental_selection_clear_locked();
    vx_incremental_invalidate_locked();
    /* A page table names tensors by name. Shutdown frees those tensors, so a
     * surviving binding would resolve names against the next model -- and its
     * gather buffers would leak. Drop it here rather than trusting every
     * caller to unbind. */
    vx_paged_unbind_locked();
    /* The engine has already freed every tensor and arena allocation. */
    g_arena_detached = 0;
}
