#ifndef VOLVOXAI_INCREMENTAL_RUNTIME_H
#define VOLVOXAI_INCREMENTAL_RUNTIME_H

#include "engine_internal.h"

#include <stdint.h>

/* Private inference scheduler. Callers hold the engine model lock. */
void vx_incremental_invalidate_locked(void);
void vx_incremental_prepare_ordinary_locked(void);
void vx_incremental_mark_tensor_locked(T* tensor);
/* Install one replay-validated FIXED terminal before the binding commit. No
 * wire pointer is retained; the private one-shot mask is consumed or cleared
 * by the same serialized operation. */
int vx_incremental_selection_install_locked(
    const uint8_t* response_v1, uint32_t response_v1_bytes);
void vx_incremental_selection_clear_locked(void);
int vx_incremental_forward_locked(int row);
/* A resumable incremental pass: begin, then next/node until next returns the
 * node count, then end. `next` names the node that runs next without running it
 * and is idempotent; a negative answer is a refusal. A stepwise pass holds no
 * device pass or adapter route between nodes, so the caller may drop the model
 * lock at any node boundary. Engine state must stay reserved to the pass. */
typedef struct VxIncrementalRun {
    int row, stepwise, force_full, weights_changed, direct_cpu;
    int previous_execution_row;
    int device_pass, forward_open, adapter_open, mark_host_dirty, begun;
    int row_prepared, cuda_pair_eligible, cuda_replayed;
    int cursor, executed, skipped;
    double started_ms;
} VxIncrementalRun;
int vx_incremental_run_begin_locked(VxIncrementalRun* run, int row, int stepwise);
int vx_incremental_run_next_locked(VxIncrementalRun* run);
int vx_incremental_run_node_locked(VxIncrementalRun* run, int node);
int vx_incremental_run_end_locked(VxIncrementalRun* run, int ok);
int vx_incremental_row_supported_locked(void);
/* A no-command row probe may borrow current projection descriptors. Dynamic
 * row proofs never escape its temporary geometry; callers restore geometry
 * before ending this scope. No model references are rebuilt by begin. */
typedef struct VxIncrementalRowProbeScope {
    uint64_t generation;
    int saved_commands_active;
} VxIncrementalRowProbeScope;
void vx_incremental_row_probe_begin_locked(VxIncrementalRowProbeScope* scope,
                                          int stepwise, int allow_descriptors);
void vx_incremental_row_probe_end_locked(const VxIncrementalRowProbeScope* scope);
int vx_incremental_tensor_dirty_before_node_locked(
    const T* target, int node_index);

/* Resolve private executable references once per model revision. A temporary
 * admission descriptor or a graph under mutation keeps the name lookup. Only
 * indices are reused: arena rebinding and metadata growth retain no pointers. */
static inline int vx_incremental_node_bindings_current(const Node* node,
                                                        const VxEngineState* state) {
    if (!node || !state || !state->nodes || !state->tensors || state->node_count <= 0 ||
        !state->incremental_plan.valid ||
        state->incremental_plan.generation != state->model_generation ||
        state->incremental_plan.node_count != state->node_count ||
        state->incremental_plan.tensor_count != state->tensor_count) return 0;
    uintptr_t begin = (uintptr_t)state->nodes, pointer = (uintptr_t)node;
    return pointer >= begin && (pointer - begin) % sizeof(Node) == 0 &&
        (pointer - begin) / sizeof(Node) < (size_t)state->node_count;
}

static inline T* vx_incremental_node_input_locked(const Node* node, int input) {
    if (!node || !node->ins || input < 0 || input >= node->nin) return NULL;
    VxEngineState* state = vx_engine_state_current();
    int index = node->ins[input].tensor_index;
    if (vx_incremental_node_bindings_current(node, state) &&
        index >= 0 && index < state->tensor_count) return &state->tensors[index];
    return t_find(node->ins[input].name);
}

static inline T* vx_incremental_node_output_locked(const Node* node, int output) {
    if (!node || output < -1 || output >= node->nout || (output >= 0 && !node->outs)) return NULL;
    VxEngineState* state = vx_engine_state_current();
    int index = output < 0 ? node->primary_output_index : node->outs[output].tensor_index;
    if (vx_incremental_node_bindings_current(node, state) &&
        index >= 0 && index < state->tensor_count) return &state->tensors[index];
    return t_find(output < 0 ? node->out : node->outs[output].name);
}

static inline const VxIncrementalTensorBinding* vx_incremental_tensor_binding_locked(
        const T* tensor, const VxEngineState* state) {
    if (!tensor || !state || !state->tensors || state->tensor_count <= 0 ||
        !state->incremental_plan.valid ||
        state->incremental_plan.generation != state->model_generation ||
        state->incremental_plan.tensor_count != state->tensor_count ||
        state->incremental_plan.node_count != state->node_count ||
        state->incremental_scratch.tensor_capacity < (size_t)state->tensor_count ||
        !state->incremental_scratch.tensor_bindings) return NULL;
    uintptr_t begin = (uintptr_t)state->tensors, pointer = (uintptr_t)tensor;
    if (pointer < begin || (pointer - begin) % sizeof(T) ||
        (pointer - begin) / sizeof(T) >= (size_t)state->tensor_count) return NULL;
    return &state->incremental_scratch.tensor_bindings[(pointer - begin) / sizeof(T)];
}

static inline int vx_incremental_tensor_is_model_weight_locked(const T* tensor) {
    if (!tensor) return 0;
    const VxIncrementalTensorBinding* prepared =
        vx_incremental_tensor_binding_locked(tensor, vx_engine_state_current());
    if (prepared && prepared->data == tensor->data &&
        prepared->numel == tensor->numel && prepared->elem_size == tensor->elem_size &&
        prepared->dtype == tensor->dtype && prepared->ndim == tensor->ndim &&
        prepared->is_graph_input == tensor->is_graph_input && tensor->ndim >= 0 &&
        tensor->ndim <= 8) {
        int same = 1;
        for (int axis = 0; axis < tensor->ndim; axis++)
            if (prepared->shape[axis] != tensor->shape[axis]) { same = 0; break; }
        if (same) return prepared->is_model_weight;
    }
    return volvoxai_engine_tensor_is_model_weight_locked(tensor->name);
}
/*
 * The token extent and the trailing width of a sequence tensor, after skipping
 * leading unit axes, so [S,D], [1,S,D] and [1,1,S,D] all answer the same.
 *
 * It lives in the header because both the dispatcher and the row planner ask
 * it, and because restating "the token axis is shape[1]" per caller is what
 * made row execution silently unreachable for the sequence-major spelling in
 * three separate places.  static inline rather than a linked symbol: not every
 * target that compiles the row planner also links the dispatcher.
 */
static inline int vx_incremental_row_extent(const T* tensor, long* width) {
    long trailing = 1;
    int axis;
    if (!tensor || tensor->ndim <= 0) return 0;
    for (axis = 0; axis < tensor->ndim && tensor->shape[axis] == 1; axis++) {}
    /*
     * A declared batch's leading axis is slots, not tokens.
     *
     * The loop above skips leading *unit* axes, which is exactly right until a
     * step declares more than one slot: the leading extent of `[B,S,D]` is then
     * B, and every caller would read B tokens of width S*D. Nothing in the
     * shape distinguishes that from a genuine `[S,1,D]` whose token axis really
     * is leading -- which is why the slot count is a declaration and is
     * consulted here rather than inferred. Answering in one place keeps the
     * dispatcher, the row planner and the operators from disagreeing.
     */
    if (g_decode_slots > 1 && axis + 1 < tensor->ndim &&
        tensor->shape[axis] == g_decode_slots) axis++;
    if (axis >= tensor->ndim) return 0;
    for (int rest = axis + 1; rest < tensor->ndim; rest++)
        trailing *= tensor->shape[rest];
    if (tensor->shape[axis] <= 0 || trailing <= 0) return 0;
    if (width) *width = trailing;
    return (int)tensor->shape[axis];
}

static inline int vx_incremental_row_tensor_3d(const T* tensor, int row) {
    long width = 0;
    return vx_incremental_row_extent(tensor, &width) > row && width > 0;
}

/* Returns 1 when a row can execute now, 0 for a compatibility fallback, and
 * -1 when synchronization failed and the retained cache is no longer safe. */
int vx_incremental_prepare_hybrid_row_locked(int row);
int vx_incremental_hybrid_row_active_locked(void);
void vx_incremental_shutdown_locked(void);

#endif
