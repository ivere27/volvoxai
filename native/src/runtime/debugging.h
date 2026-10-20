/* Private implementation of VxDebugService. No debug object is allocated by
 * ordinary inference. Readers run under the session lock and copy their view. */
#ifndef VOLVOXAI_DEBUGGING_H
#define VOLVOXAI_DEBUGGING_H
#include "execution_plan.h"
#include "volvoxai_full_enums.h"

typedef struct VxDebugSession VxDebugSession;
typedef struct VxEngineState VxEngineState;
typedef struct {
    /* Source graph node IDs; a step is selected when it executes any of them. */
    const char* const* node_ids;
    size_t node_count;
    const char* const* tensor_names;
    size_t tensor_count;
    int inputs, outputs, values;
    /* Zero leaves weights and initializers (tensors no step produces that are
     * not graph inputs) out of every snapshot. */
    int constants;
    /* Train-step targets: zero leaves gradients out of every snapshot. */
    int gradients;
    uint64_t max_bytes;
    uint32_t max_events, max_snapshots;
} VxDebugOptions;
typedef struct {
    uint64_t finite, nan, positive_infinity, negative_infinity;
    double minimum, maximum, mean, variance;
} VxDebugStatistics;
/* Statistics of storage values and, for quantized tensors, of real values. */
typedef struct {
    size_t count;
    uint32_t axis;
    int per_axis;
    float* scales;
    int32_t* zero_points;
} VxDebugQuantization;
typedef struct {
    uint64_t id, event_id;
    uint32_t tensor_id;
    VxDebugTensorStatus status;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[8];
    size_t bytes;
    unsigned char* data;
    VxDebugStatistics statistics, real_statistics;
    int has_real_statistics;
} VxDebugSnapshot;
typedef struct {
    uint64_t id;
    uint32_t step;
    VxDebugPoint point;
    size_t first_snapshot, snapshot_count;
} VxDebugEvent;
/* A decode target's slot: logical token rows before and after, and the rows
 * this operation writes. */
typedef struct {
    uint32_t slot, length_before, length_after, write_begin, write_end;
    int empty;
} VxDebugSlot;
/* The rows one causal attention step reads as its key or value operand. A
 * token's bytes are `token_bytes` at `column_offset` within a storage row. */
typedef struct {
    uint32_t id;
    VxDebugKVRole role;
    uint32_t tensor_id, attention_step;
    int writer_step; /* -1: no producing step */
    VxDataType dtype;
    uint32_t token_capacity, page_tokens, token_rank;
    int64_t token_shape[8];
    int paged;
    size_t row_bytes, column_offset, token_bytes;
} VxDebugKVCache;
typedef struct {
    VxDebugState state;
    VxDebugStopReason stop_reason;
    uint64_t revision; /* Changes on every state change; not a position. */
    uint32_t next_step;
    size_t event_count, snapshot_count;
    uint64_t retained_bytes, peak_bytes, max_bytes;
    uint64_t dropped_events, dropped_snapshots;
    int capture_complete, device_synchronization, preserves_node_boundaries;
    int modified; /* A value was replaced with vx_debug_set_tensor. */
    const VxExecutionPlan* plan;
    const VxDebugEvent* events;
    const VxDebugSnapshot* snapshots;
    const VxDebugQuantization* quantization;
    VxReport report;
    VxDebugTarget target;
    const VxDebugSlot* slots;
    uint32_t slot_count;
    const VxDebugKVCache* caches;
    size_t cache_count;
    const unsigned char* executed; /* per step, decode targets */
    int result_available;
    /* Train-step target: its owner, which publishes the TrainStep result. */
    void* train_owner;
    int train_completed;
} VxDebugView;

/* A train-step target. The Trainer builds the plan and runs its own stepwise
 * TrainStep; the session captures observations between steps. */
typedef struct {
    void* owner;
    VxEngineState* state; /* The Trainer's engine: activations and parameters. */
    const char* backend;
    /* Plan tensors at or above `gradient_base` are gradients of tensor
     * (id - gradient_base). The session owns `plan` after creation. */
    VxExecutionPlan* plan;
    uint32_t gradient_base;
    /* 0: executed; nonzero: failed, with `report` describing why. */
    int (*step)(void* owner, uint32_t step, VxReport* report);
    /* A gradient's values, or NULL when no gradient reaches that tensor. For
     * optimizer steps the accumulated gradient being applied. */
    const float* (*gradient)(void* owner, uint32_t tensor, int accumulated);
    /* ok: publish what TrainStep would return; otherwise restore the Trainer
     * like a failed TrainStep. Detaches. Nonzero when completion failed. */
    int (*finish)(void* owner, int ok, VxReport* report);
    void (*release)(void* owner);
} VxDebugTrainTarget;
VxStatus vx_debug_create_train(const VxDebugTrainTarget*, const VxDebugOptions*,
    VxDebugSession**, VxReport*);

typedef struct VxGraphExecutionOptions VxGraphExecutionOptions;
/* Runs an admitted Graph Execute one node at a time on the caller's context.
 * The context refuses other operations until the debugger detaches. */
VxStatus vx_debug_create_execute(VxExecutionContext*, const VxGraphExecutionOptions*,
    const VxTensorBinding*, size_t, const VxDebugOptions*, VxDebugSession**, VxReport*);
/* Transfers the completed packet once. No packet is published by inspection.
 * Context and output storage are borrowed while the caller retains session. */
VxStatus vx_debug_take_result(VxDebugSession*, VxResult**, VxExecutionContext**,
    int* owned, VxReport*);
/* Cancels, then releases the attached context. Idempotent. */
void vx_debug_detach(VxDebugSession*);
void vx_debug_retain(VxDebugSession*);
void vx_debug_release(VxDebugSession*);
/* Retains the attached session if its final release has not begun. */
VxDebugSession* vx_debug_context_acquire(VxExecutionContext*);
/* Called after cancellation with a preallocated completion observer. */
int vx_debug_claim_cancel_drain(VxDebugSession*);
/* Stale revision: REVISION_CONFLICT. RUNNING: BUSY. Finished: INVALID_ARGUMENT
 * with DEBUG_SESSION_FINISHED. Unknown breakpoint node: INVALID_ARGUMENT. */
VxStatus vx_debug_start(VxDebugSession*, uint64_t expected_revision, int single_step,
    const char* const* break_nodes, size_t break_count, int break_on_nonfinite, VxReport*);
/* One node at most. 1: runnable; 0: waiting for a device; -1: no active command. */
int vx_debug_progress(VxDebugSession*);
/* Replaces one non-constant input of the paused step (DAP setVariable). */
VxStatus vx_debug_set_tensor(VxDebugSession*, uint64_t expected_revision, uint32_t tensor_id,
    const VxTensorBinding* value, VxReport*);
void vx_debug_cancel(VxDebugSession*);
int vx_debug_inspect(VxDebugSession*, int (*write)(const VxDebugView*, void*), void*);

/* Live KV reads at a stop, one in flight per session. begin validates and may
 * start a device readback; poll answers 1 while it is pending; finish gathers
 * the rows in logical token order, calls write once and ends the read. */
typedef struct {
    uint32_t cache_id, slot, token_offset, token_limit;
} VxDebugKVReadRequest;
typedef struct {
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[9];
    const unsigned char* data;
    size_t bytes;
    uint32_t token_count;
    VxDebugStatistics statistics;
    int has_statistics;
    const VxDebugQuantization* quantization;
} VxDebugKVRows;
VxStatus vx_debug_kv_read_begin(VxDebugSession*, const VxDebugKVReadRequest*, VxReport*);
int vx_debug_kv_read_poll(VxDebugSession*);
VxStatus vx_debug_kv_read_finish(VxDebugSession*, int (*write)(const VxDebugKVRows*, void*),
    void*, VxReport*);
void vx_debug_kv_read_cancel(VxDebugSession*);
#endif
