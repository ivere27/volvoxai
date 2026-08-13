#ifndef VOLVOX_RUNTIME_CONTINUOUS_BATCH_SCHEDULER_H
#define VOLVOX_RUNTIME_CONTINUOUS_BATCH_SCHEDULER_H

/*
 * Continuous batching — native twin of `ts/core/ContinuousBatchScheduler.ts`.
 *
 * Both are driven by `tests/continuous_batching_vectors.json`, so admission
 * order, slot assignment, page allocation and retirement cannot diverge
 * between the browser and the robot.
 *
 * This adapter is deliberately engine-independent. It owns request, slot and
 * Paged-KV policy, while the caller-supplied step callback owns actual model
 * execution. It neither owns a `VxExecutionContext` nor acts as the production
 * Runtime coordinator; see `vx_continuous_batch_scheduler_result` for its
 * separate result-ordering contract.
 *
 * A slot is a lane of a `VxPagedKVCache`.  Admission reserves the lane's
 * prompt pages before the request becomes visible; retirement bumps the lane
 * generation, so work submitted against a previous occupant can never be
 * applied to its replacement.
 *
 * Like paged_kv.c this is free of engine dependencies: what a token is and how
 * a model runs stays with the caller's step callback.
 */

#include "paged_kv.h"

#define VX_CONTINUOUS_NO_SLOT (-1)

typedef enum {
    VX_CONTINUOUS_OK = 0,
    VX_CONTINUOUS_INVALID_ARGUMENT = -1,
    VX_CONTINUOUS_QUEUE_FULL = -2,
    VX_CONTINUOUS_SCHEDULER_CLOSED = -3,
    VX_CONTINUOUS_STEP_FAILED = -4
} VxContinuousStatus;

typedef enum {
    VX_CONTINUOUS_REQUEST_QUEUED = 0,
    VX_CONTINUOUS_REQUEST_PREFILL = 1,
    VX_CONTINUOUS_REQUEST_DECODING = 2,
    VX_CONTINUOUS_REQUEST_COMPLETED = 3,
    VX_CONTINUOUS_REQUEST_CANCELLED = 4,
    VX_CONTINUOUS_REQUEST_FAILED = 5
} VxContinuousRequestState;

typedef enum {
    VX_CONTINUOUS_PHASE_PREFILL = 0,
    VX_CONTINUOUS_PHASE_DECODE = 1
} VxContinuousPhase;

/* What a slot needs to run one step.  Every field is a value, never a shape. */
typedef struct {
    int request_id;
    int slot;
    int slot_generation;
    VxContinuousPhase phase;
    /* Where this step writes, in logical token order. */
    int position;
    int tokens;
    int kv_length;
    /* Lane-local logical -> physical page mapping, `pages_per_lane` wide. */
    const int* page_table;
    int page_tokens;
    int generated;
    void* payload;
} VxContinuousStepWork;

/* One lane's result. */
typedef struct {
    /* Stop generating for this request before max_tokens. */
    int finished;
    /*
     * VX_CONTINUOUS_OK, or a failure that retires this request alone.
     *
     * For the caller that can attribute a failure to one lane.  A caller that
     * cannot returns non-OK from the callback itself, which retires every lane
     * in the batch -- the honest reading of a dispatch that did not happen.
     */
    VxContinuousStatus status;
} VxContinuousStepOutcome;

/*
 * Run one round's work.
 *
 * Takes the round's lanes together and writes one outcome per lane, in the same
 * order.  That is the shape because a batched decode step executes as one
 * dispatch per node regardless of lane count: a per-lane callback would put the
 * per-request dispatch back on top of a backend that had just removed it.
 *
 * A prefill batch holds one lane -- prompts of different lengths are not one
 * dense step -- so `count` is one there and the whole active set on decode.
 * One lane is not a second contract; it is this contract with a list of one.
 *
 * Returning anything but VX_CONTINUOUS_OK retires every request in the batch.
 */
typedef VxContinuousStatus (*VxContinuousRunStep)(const VxContinuousStepWork* works, int count,
                                        VxContinuousStepOutcome* outcomes, void* user);

typedef struct {
    int request_id;
    VxContinuousRequestState state;
    int generated;
} VxContinuousResult;

typedef struct {
    int admitted;
    int completed;
    int cancelled;
    int failed;
    int queue_depth;
    int active_slots;
    int free_slots;
    int rounds;
    /* Lane-steps advanced: how much work the rounds asked for. */
    int steps;
    /*
     * Callback invocations: how many times the backend was entered.
     *
     * Reported beside `steps` rather than instead of it.  `steps / dispatches`
     * is the batching win, and a scheduler reporting only one of the two could
     * claim that win without having produced it.
     */
    int dispatches;
    int admission_stalls;
    int max_queue_depth_seen;
    int queue_delay_rounds;
    /* Admissions that bound a resident prefix instead of prefilling it. */
    int prefix_reuses;
    int prefix_publications;
    /* Prompt tokens this scheduler actually prefilled. */
    int prefill_tokens;
    /* Prompt tokens served by a shared prefix — the work sharing avoided. */
    int shared_prompt_tokens;
} VxContinuousTelemetry;

typedef struct VxContinuousBatchScheduler VxContinuousBatchScheduler;

/* The scheduler borrows the cache; the caller keeps ownership of it. */
VxContinuousBatchScheduler* vx_continuous_batch_scheduler_create(VxPagedKVCache* cache, int max_queue_depth);
void vx_continuous_batch_scheduler_destroy(VxContinuousBatchScheduler* scheduler);

/* Enqueue a request.  Returns its id through `id_out`, or a refusal: a request
 * that could never fit one lane is rejected here rather than left to starve
 * the queue behind it.
 *
 * `prompt_key` is this prompt's identity for prefix sharing, or NULL to opt
 * out.  When a resident prefix carries the same key its pages are bound into
 * the lane instead of being recomputed, so the request skips the prefill of
 * those tokens and the pages exist once for every holder.  The key must cover
 * everything that changes the K/V it stands for — model and weight revision,
 * adapter revisions, tokenizer semantics, quantization, prompt tokens —
 * because getting it wrong hands a request another request's attention
 * state. */
VxContinuousStatus vx_continuous_batch_scheduler_submit(VxContinuousBatchScheduler* scheduler,
                                        int prompt_tokens, int max_tokens,
                                        const char* prompt_key,
                                        void* payload, int* id_out);
/* Cancel at any stage.  Returns 1 when the request moved to cancelled. */
int vx_continuous_batch_scheduler_cancel(VxContinuousBatchScheduler* scheduler, int request_id);

/* One scheduling round: admit what fits, then advance every active slot.
 *
 * The decode phase calls `run_step` once with every active lane; prefill calls
 * it once per request.  `worked_out` reports whether the round did anything. */
VxContinuousStatus vx_continuous_batch_scheduler_step(VxContinuousBatchScheduler* scheduler,
                                      VxContinuousRunStep run_step, void* user,
                                      int* worked_out);
VxContinuousStatus vx_continuous_batch_scheduler_run_until_idle(VxContinuousBatchScheduler* scheduler,
                                                VxContinuousRunStep run_step, void* user,
                                                int max_rounds);
/* Close.  With `drain` the admitted work finishes and the queue is cancelled;
 * without it everything unfinished is cancelled.  Neither abandons accepted
 * work silently — every request is published with an outcome. */
VxContinuousStatus vx_continuous_batch_scheduler_close(VxContinuousBatchScheduler* scheduler,
                                       VxContinuousRunStep run_step, void* user,
                                       int drain);

VxContinuousRequestState vx_continuous_batch_scheduler_state(const VxContinuousBatchScheduler* scheduler, int request_id);
int vx_continuous_batch_scheduler_slot_of(const VxContinuousBatchScheduler* scheduler, int request_id);
int vx_continuous_batch_scheduler_queue_depth(const VxContinuousBatchScheduler* scheduler);
/*
 * Retired requests in publication order.
 *
 * Deterministic, but deliberately not admission order: a short request
 * admitted second finishes first, and holding its result until the first
 * finishes would restore the head-of-line blocking continuous batching exists
 * to remove.  Within a round, completions publish in slot order; rounds
 * publish in sequence.
 */
int vx_continuous_batch_scheduler_result_count(const VxContinuousBatchScheduler* scheduler);
const VxContinuousResult* vx_continuous_batch_scheduler_result(const VxContinuousBatchScheduler* scheduler, int index);
void vx_continuous_batch_scheduler_telemetry(const VxContinuousBatchScheduler* scheduler,
                                  VxContinuousTelemetry* out);

#endif /* VOLVOX_RUNTIME_CONTINUOUS_BATCH_SCHEDULER_H */
