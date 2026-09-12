#ifndef VOLVOX_RUNTIME_BATCH_SCHEDULER_H
#define VOLVOX_RUNTIME_BATCH_SCHEDULER_H

/*
 * Engine-independent batch scheduling policy core.
 *
 * This is the native twin of `ts/core/BatchScheduler.ts`: both exercise the
 * same callback-driven contribution grouping, token-budget, paged-KV lifecycle,
 * and telemetry policy. The native VxBatch* surface names this common policy
 * core directly. It is not the Runtime-owned production coordinator;
 * native Runtime submission and request ownership live in public_api.c and its
 * runtime-request implementation fragments.
 *
 * The core provides:
 *   - Contribution and group keys (model, adapter, shape signature, rows per lane).
 *   - Common admission and work-conserving callback dispatch.
 *   - Row decode for stateful LLM work and bulk [B,...] stateless contributions.
 *   - Telemetry: utilization = device_busy / wall, padding_waste = 1 - useful / dispatched.
 *
 * Scheduling policy and telemetry both use the real monotonic clock. The
 * latter remains a claim about the machine rather than caller-supplied time.
 */

/* Like paged_kv.c this is free of engine dependencies: what a token is and how
 * a model runs stays with the caller's step callback. */
#include "paged_kv.h"

#include <stdint.h>
#include <stddef.h>

#define VX_BATCH_NO_SLOT (-1)
/* Requests that carry no group identity land here, matching the TS defaults. */
#define VX_BATCH_DEFAULT_MODEL "default"

typedef enum {
    VX_BATCH_OK = 0,
    VX_BATCH_INVALID_ARGUMENT = -1,
    VX_BATCH_QUEUE_FULL = -2,
    VX_BATCH_SCHEDULER_CLOSED = -3,
    VX_BATCH_STEP_FAILED = -4
} VxBatchStatus;

typedef enum {
    /* The id is invalid or its terminal record has left the bounded retention
     * window. */
    VX_BATCH_REQUEST_UNKNOWN = -1,
    VX_BATCH_REQUEST_QUEUED = 0,
    VX_BATCH_REQUEST_PREFILL = 1,
    VX_BATCH_REQUEST_DECODING = 2,
    VX_BATCH_REQUEST_COMPLETED = 3,
    VX_BATCH_REQUEST_CANCELLED = 4,
    VX_BATCH_REQUEST_FAILED = 5
} VxBatchRequestState;

typedef enum {
    VX_BATCH_KIND_STATELESS = 0,
    VX_BATCH_KIND_PREFILL = 1,
    VX_BATCH_KIND_DECODE = 2
} VxBatchRequestKind;

typedef enum {
    VX_DISPATCH_WORK_CONSERVING = 0,
    VX_DISPATCH_FILL_FIRST = 1
} VxDispatchPolicyMode;

typedef struct {
    VxDispatchPolicyMode mode;
    int64_t max_wait_micros;
} VxBatchDispatchPolicy;

typedef struct {
    const char* model_id;
    const char* adapter_revision;
    const char* shape_signature_minus_batch;
    int rows_per_lane;
} VxGroupKey;

typedef struct {
    int request_id;
    VxBatchRequestKind kind;
    /*
     * Rows this contribution occupies on the batch axis: one for a decode
     * token, the chunk length for a prefill, `rows_per_lane` for a stateless
     * item. Equal to the group key's `rows_per_lane` for every member of a
     * dispatch -- that is what putting `rows_per_lane` in the key buys.
     */
    int rows;
    int slot;
    int slot_generation;
    int phase; /* 0: prefill, 1: decode */
    /* Where this step writes, in logical token order.  Zero only for a prefill
     * into an empty lane: a prefill that bound a shared prefix starts after
     * it. */
    int position;
    int tokens;
    int kv_length;
    const int* page_table;
    int page_tokens;
    int generated;
    void* payload;
    /*
     * A row that exists only to satisfy `multiple_of`.
     *
     * It duplicates the first contribution rather than carrying zeros, because
     * a zero row through a normalization produces NaN and poisons the
     * diagnosis of the rows that mattered. Its outcome is discarded and
     * reaches no request.
     */
    int padding;
} VxBatchStepWork;

typedef struct {
    int finished;
    VxBatchStatus status;
    void* value;
} VxBatchStepOutcome;

typedef struct {
    VxBatchRequestKind kind;
    VxGroupKey group_key;
    int useful_count;
    int padded_count;
    int useful_rows;
    int total_rows;
} VxBatchDispatchMetadata;

/*
 * Run one dispatch.
 *
 * `count` is the whole row count including padding rows; write one outcome per
 * row, in order. Returning anything but VX_BATCH_OK retires every request in
 * the batch -- the honest reading of a dispatch that did not happen.
 */
typedef VxBatchStatus (*VxBatchRunStep)(
    const VxBatchStepWork* works,
    int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes,
    void* user);

typedef struct {
    int request_id;
    VxBatchRequestKind kind;
    VxBatchRequestState state;
    int generated;
    VxBatchStatus status;
    void* value;
} VxBatchResult;

typedef struct {
    int dispatches;
    int rows_dispatched;
    int rows_useful;
    int64_t device_busy_micros;
    int64_t wall_micros;
    double utilization;
    double padding_waste;
    int queue_depth;
    int max_queue_depth_seen;
    /* Percentiles over a bounded recent window, not over every round ever. */
    int queue_depth_p50;
    int queue_depth_p99;
    /* Distinct group keys with work ready right now. Fragmentation indicator:
     * when it is large the batches are not forming, and any throughput claim
     * made in that state should be doubted. */
    int group_count;
    int admitted;
    int completed;
    int cancelled;
    int failed;
    int steps;
    int rounds;
    int admission_stalls;
    int queue_delay_rounds;
    int prefix_reuses;
    int prefix_publications;
    int prefill_tokens;
    int shared_prompt_tokens;
    int active_slots;
    int free_slots;
    int active_request_records;
    int max_request_records_seen;
    int retained_results;
    int max_retained_results;
} VxBatchTelemetry;

typedef struct {
    VxPagedKVCache* cache;
    int max_queue_depth;
    int token_budget_per_dispatch;
    int max_lanes;
    VxBatchDispatchPolicy policy;
    int multiple_of;
    /* Zero means the default window. */
    int queue_depth_window;
    /* Maximum terminal results retained for indexed/state lookup. Zero uses
     * 2 * max_queue_depth + lane count, matching the bounded active-record
     * window. Oldest results are evicted; opaque values remain owned by the
     * caller and are never freed by the scheduler. */
    int max_retained_results;
} VxBatchSchedulerOptions;

typedef struct VxBatchScheduler VxBatchScheduler;

/* Lifecycle */
VxBatchScheduler* vx_batch_scheduler_create(const VxBatchSchedulerOptions* options);
void vx_batch_scheduler_destroy(VxBatchScheduler* scheduler);

/* Submission */
VxBatchStatus vx_batch_scheduler_submit_stateless(
    VxBatchScheduler* scheduler,
    const char* model_id,
    const char* adapter_revision,
    const char* shape_signature_minus_batch,
    int rows,
    void* payload,
    int* id_out);

VxBatchStatus vx_batch_scheduler_submit_llm(
    VxBatchScheduler* scheduler,
    const char* model_id,
    const char* adapter_revision,
    const char* shape_signature_minus_batch,
    int prompt_tokens,
    int max_tokens,
    const char* prompt_key,
    void* payload,
    int* id_out);

/* Cancellation */
int vx_batch_scheduler_cancel(VxBatchScheduler* scheduler, int request_id);

/* Asynchronous worker seam. A dispatch owns its reservations until complete.
 * Views stay valid until complete/close; next is idempotent while pending.
 * These are internal C operations; applications use generated proto dispatch. */
typedef struct {
    uint64_t id;
    const VxBatchStepWork* works;
    const VxBatchDispatchMetadata* metadata;
    int count;
} VxBatchDispatchView;
VxBatchStatus vx_batch_scheduler_next(VxBatchScheduler*, VxBatchDispatchView*);
int vx_batch_scheduler_peek(const VxBatchScheduler*, VxBatchDispatchView*);
VxBatchStatus vx_batch_scheduler_complete(VxBatchScheduler*, uint64_t id,
    const VxBatchStepOutcome*, int count, VxBatchStatus worker_status);
VxBatchStatus vx_batch_scheduler_close_async(VxBatchScheduler*, int drain);
int vx_batch_scheduler_closed(const VxBatchScheduler*);
int vx_batch_scheduler_draining(const VxBatchScheduler*);
uint64_t vx_batch_scheduler_pending_id(const VxBatchScheduler*);
int vx_batch_scheduler_generated(const VxBatchScheduler*, int request_id);

/*
 * Advance one scheduling round.
 *
 * Policy decisions and telemetry sample the monotonic clock independently.
 */
VxBatchStatus vx_batch_scheduler_step(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int* worked_out);

VxBatchStatus vx_batch_scheduler_run_until_idle(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int max_rounds);

/* Close.  With `drain` the admitted work finishes; without it everything
 * unfinished is cancelled.  Queued work is published either way -- a request
 * that never reached a slot has nothing to drain, and abandoning it silently
 * would lose an accepted request. */
VxBatchStatus vx_batch_scheduler_close(
    VxBatchScheduler* scheduler,
    VxBatchRunStep run_step,
    void* user,
    int drain);

/* Inspection */
VxBatchRequestState vx_batch_scheduler_state(const VxBatchScheduler* scheduler, int request_id);
int vx_batch_scheduler_slot_of(const VxBatchScheduler* scheduler, int request_id);
int vx_batch_scheduler_queue_depth(const VxBatchScheduler* scheduler);
int vx_batch_scheduler_round(const VxBatchScheduler* scheduler);

int vx_batch_scheduler_result_count(const VxBatchScheduler* scheduler);
/* Results are ordered oldest-to-newest within the bounded retention window.
 * A returned pointer is invalidated by the next terminal publication. */
const VxBatchResult* vx_batch_scheduler_result(const VxBatchScheduler* scheduler, int index);
/* NULL means invalid, still active, or evicted from the retention window. */
const VxBatchResult* vx_batch_scheduler_result_for_request(
    const VxBatchScheduler* scheduler, int request_id);
/* Remove a terminal record without changing live lane ownership. */
int vx_batch_scheduler_forget_result(VxBatchScheduler*, int request_id);

void vx_batch_scheduler_telemetry(const VxBatchScheduler* scheduler, VxBatchTelemetry* out);

#endif /* VOLVOX_RUNTIME_BATCH_SCHEDULER_H */
