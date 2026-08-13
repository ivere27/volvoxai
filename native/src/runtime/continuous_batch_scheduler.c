/*
 * Continuous batching — native twin of `ts/core/ContinuousBatchScheduler.ts`.
 * Implemented on top of the engine-independent batch policy core.
 */

#include "continuous_batch_scheduler.h"
#include "batch_scheduler.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct VxContinuousBatchScheduler {
    VxBatchScheduler* batch;
    VxPagedKVCache* cache;
    /*
     * Projection of the batch results into this API's narrower record.
     *
     * A stable array rather than one shared slot: `vx_continuous_batch_scheduler_result`
     * documents a list, and callers walk it holding more than one pointer at a
     * time.  Handing them a single buffer would make the second call rewrite
     * the first answer under them.
     */
    VxContinuousResult* results;
    int result_capacity;
};

typedef struct {
    VxContinuousRunStep run_step;
    void* user;
} ContinuousAdapterContext;

static VxBatchStatus continuous_run_step_adapter(
    const VxBatchStepWork* works,
    int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes,
    void* user) {
    (void)metadata;
    ContinuousAdapterContext* ctx = (ContinuousAdapterContext*)user;
    VxContinuousStepWork* adapted_works = (VxContinuousStepWork*)calloc((size_t)count, sizeof(VxContinuousStepWork));
    VxContinuousStepOutcome* adapted_outcomes = (VxContinuousStepOutcome*)calloc((size_t)count, sizeof(VxContinuousStepOutcome));
    if (!adapted_works || !adapted_outcomes) {
        free(adapted_works);
        free(adapted_outcomes);
        return VX_BATCH_INVALID_ARGUMENT;
    }

    for (int i = 0; i < count; i++) {
        adapted_works[i].request_id = works[i].request_id;
        adapted_works[i].slot = works[i].slot;
        adapted_works[i].slot_generation = works[i].slot_generation;
        adapted_works[i].phase = (works[i].phase == 0) ? VX_CONTINUOUS_PHASE_PREFILL : VX_CONTINUOUS_PHASE_DECODE;
        adapted_works[i].position = works[i].position;
        adapted_works[i].tokens = works[i].tokens;
        adapted_works[i].kv_length = works[i].kv_length;
        adapted_works[i].page_table = works[i].page_table;
        adapted_works[i].page_tokens = works[i].page_tokens;
        adapted_works[i].generated = works[i].generated;
        adapted_works[i].payload = works[i].payload;
    }

    VxContinuousStatus status = ctx->run_step(adapted_works, count, adapted_outcomes, ctx->user);

    for (int i = 0; i < count; i++) {
        outcomes[i].finished = adapted_outcomes[i].finished;
        outcomes[i].status = (VxBatchStatus)adapted_outcomes[i].status;
        outcomes[i].value = NULL;
    }

    free(adapted_works);
    free(adapted_outcomes);
    return (VxBatchStatus)status;
}

VxContinuousBatchScheduler* vx_continuous_batch_scheduler_create(VxPagedKVCache* cache, int max_queue_depth) {
    VxContinuousBatchScheduler* scheduler;
    VxBatchSchedulerOptions options;
    if (!cache) return NULL;

    scheduler = (VxContinuousBatchScheduler*)calloc(1, sizeof(*scheduler));
    if (!scheduler) return NULL;

    memset(&options, 0, sizeof(options));
    options.cache = cache;
    options.max_queue_depth = max_queue_depth;
    options.max_lanes = vx_paged_kv_lanes(cache);
    if (vx_paged_kv_lane_token_capacity(cache) > INT_MAX / options.max_lanes) {
        options.token_budget_per_dispatch = INT_MAX;
    } else {
        options.token_budget_per_dispatch =
            vx_paged_kv_lane_token_capacity(cache) * options.max_lanes;
    }
    options.multiple_of = 1;

    scheduler->batch = vx_batch_scheduler_create(&options);
    if (!scheduler->batch) {
        free(scheduler);
        return NULL;
    }
    scheduler->cache = cache;
    return scheduler;
}

void vx_continuous_batch_scheduler_destroy(VxContinuousBatchScheduler* scheduler) {
    if (!scheduler) return;
    vx_batch_scheduler_destroy(scheduler->batch);
    free(scheduler->results);
    free(scheduler);
}

VxContinuousStatus vx_continuous_batch_scheduler_submit(VxContinuousBatchScheduler* scheduler,
                                        int prompt_tokens, int max_tokens,
                                        const char* prompt_key,
                                        void* payload, int* id_out) {
    if (!scheduler) return VX_CONTINUOUS_INVALID_ARGUMENT;
    return (VxContinuousStatus)vx_batch_scheduler_submit_llm(
        scheduler->batch, "default", NULL, "llm",
        prompt_tokens, max_tokens, prompt_key, payload, id_out);
}

int vx_continuous_batch_scheduler_cancel(VxContinuousBatchScheduler* scheduler, int request_id) {
    if (!scheduler) return 0;
    return vx_batch_scheduler_cancel(scheduler->batch, request_id);
}

VxContinuousStatus vx_continuous_batch_scheduler_step(VxContinuousBatchScheduler* scheduler,
                                      VxContinuousRunStep run_step, void* user,
                                      int* worked_out) {
    if (!scheduler || !run_step) return VX_CONTINUOUS_INVALID_ARGUMENT;
    ContinuousAdapterContext ctx;
    ctx.run_step = run_step;
    ctx.user = user;
    return (VxContinuousStatus)vx_batch_scheduler_step(
        scheduler->batch, 0, continuous_run_step_adapter, &ctx, worked_out);
}

VxContinuousStatus vx_continuous_batch_scheduler_run_until_idle(VxContinuousBatchScheduler* scheduler,
                                                VxContinuousRunStep run_step, void* user,
                                                int max_rounds) {
    if (!scheduler || !run_step) return VX_CONTINUOUS_INVALID_ARGUMENT;
    ContinuousAdapterContext ctx;
    ctx.run_step = run_step;
    ctx.user = user;
    return (VxContinuousStatus)vx_batch_scheduler_run_until_idle(
        scheduler->batch, 0, 1000, continuous_run_step_adapter, &ctx, max_rounds);
}

VxContinuousStatus vx_continuous_batch_scheduler_close(VxContinuousBatchScheduler* scheduler,
                                       VxContinuousRunStep run_step, void* user,
                                       int drain) {
    if (!scheduler) return VX_CONTINUOUS_INVALID_ARGUMENT;
    ContinuousAdapterContext ctx;
    ctx.run_step = run_step;
    ctx.user = user;
    return (VxContinuousStatus)vx_batch_scheduler_close(
        scheduler->batch, 0, run_step ? continuous_run_step_adapter : NULL, &ctx, drain);
}

VxContinuousRequestState vx_continuous_batch_scheduler_state(const VxContinuousBatchScheduler* scheduler, int request_id) {
    if (!scheduler) return VX_CONTINUOUS_REQUEST_QUEUED;
    return (VxContinuousRequestState)vx_batch_scheduler_state(scheduler->batch, request_id);
}

int vx_continuous_batch_scheduler_slot_of(const VxContinuousBatchScheduler* scheduler, int request_id) {
    if (!scheduler) return VX_CONTINUOUS_NO_SLOT;
    return vx_batch_scheduler_slot_of(scheduler->batch, request_id);
}

int vx_continuous_batch_scheduler_queue_depth(const VxContinuousBatchScheduler* scheduler) {
    if (!scheduler) return 0;
    return vx_batch_scheduler_queue_depth(scheduler->batch);
}

int vx_continuous_batch_scheduler_result_count(const VxContinuousBatchScheduler* scheduler) {
    if (!scheduler) return 0;
    return vx_batch_scheduler_result_count(scheduler->batch);
}

const VxContinuousResult* vx_continuous_batch_scheduler_result(const VxContinuousBatchScheduler* scheduler, int index) {
    /* Logically const: the projection is a cache of results the scheduler has
     * already published, so filling it changes nothing a caller can observe. */
    VxContinuousBatchScheduler* self = (VxContinuousBatchScheduler*)scheduler;
    int count;
    if (!scheduler) return NULL;
    count = vx_batch_scheduler_result_count(scheduler->batch);
    if (index < 0 || index >= count) return NULL;

    if (count > self->result_capacity) {
        int capacity = self->result_capacity ? self->result_capacity : 8;
        VxContinuousResult* grown;
        while (capacity < count) capacity *= 2;
        grown = (VxContinuousResult*)realloc(self->results,
                                        (size_t)capacity * sizeof(*grown));
        if (!grown) return NULL;
        self->results = grown;
        self->result_capacity = capacity;
    }
    for (int i = 0; i < count; i++) {
        const VxBatchResult* res = vx_batch_scheduler_result(scheduler->batch, i);
        if (!res) return NULL;
        self->results[i].request_id = res->request_id;
        self->results[i].state = (VxContinuousRequestState)res->state;
        self->results[i].generated = res->generated;
    }
    return &self->results[index];
}

void vx_continuous_batch_scheduler_telemetry(const VxContinuousBatchScheduler* scheduler,
                                  VxContinuousTelemetry* out) {
    if (!scheduler || !out) return;
    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(scheduler->batch, &telem);

    memset(out, 0, sizeof(*out));
    out->admitted = telem.admitted;
    out->completed = telem.completed;
    out->cancelled = telem.cancelled;
    out->failed = telem.failed;
    out->queue_depth = telem.queue_depth;
    out->active_slots = telem.active_slots;
    out->free_slots = telem.free_slots;
    out->rounds = telem.rounds;
    out->steps = telem.steps;
    out->dispatches = telem.dispatches;
    out->admission_stalls = telem.admission_stalls;
    out->max_queue_depth_seen = telem.max_queue_depth_seen;
    out->queue_delay_rounds = telem.queue_delay_rounds;
    out->prefix_reuses = telem.prefix_reuses;
    out->prefix_publications = telem.prefix_publications;
    out->prefill_tokens = telem.prefill_tokens;
    out->shared_prompt_tokens = telem.shared_prompt_tokens;
}
