#include "batch_scheduler.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void finish(VxBatchScheduler* queue, const VxBatchDispatchView* view) {
    VxBatchStepOutcome outcomes[8] = {{0}};
    assert(view->count <= 8);
    assert(vx_batch_scheduler_complete(queue, view->id, outcomes, view->count,
        VX_BATCH_OK) == VX_BATCH_OK);
}

static void stateless(void) {
    VxBatchSchedulerOptions options = {0};
    options.max_lanes = 4; options.multiple_of = 4;
    options.token_budget_per_dispatch = 12; options.max_retained_results = 2;
    VxBatchScheduler* q = vx_batch_scheduler_create(&options);
    assert(q);
    int a, b, c; char key[600]; memset(key, 'x', sizeof(key)-1); key[599] = 0;
    assert(vx_batch_scheduler_submit_stateless(q, key, NULL, "s", 3, NULL, &a) == VX_BATCH_OK);
    assert(vx_batch_scheduler_submit_stateless(q, key, "", "s", 3, NULL, &b) == VX_BATCH_OK);
    assert(vx_batch_scheduler_submit_stateless(q, key, NULL, "s", 3, NULL, &c) == VX_BATCH_OK);
    VxBatchDispatchView first, again, second;
    assert(vx_batch_scheduler_next(q, &first) == VX_BATCH_OK);
    assert(first.count == 4 && first.metadata->useful_count == 2);
    assert(first.works[0].request_id == a && first.works[1].request_id == c);
    assert(first.works[2].padding && first.works[2].request_id == -1);
    assert(strcmp(first.metadata->group_key.model_id, key) == 0);
    assert(vx_batch_scheduler_next(q, &again) == VX_BATCH_OK && first.id == again.id);
    assert(vx_batch_scheduler_complete(q, first.id + 1, NULL, 0, VX_BATCH_OK) == VX_BATCH_INVALID_ARGUMENT);
    assert(vx_batch_scheduler_cancel(q, c));
    finish(q, &first);
    assert(vx_batch_scheduler_state(q, a) == VX_BATCH_REQUEST_COMPLETED);
    assert(vx_batch_scheduler_state(q, c) == VX_BATCH_REQUEST_CANCELLED);
    assert(vx_batch_scheduler_next(q, &second) == VX_BATCH_OK);
    assert(second.id != first.id && second.count == 4 && second.metadata->useful_count == 1);
    assert(second.works[0].request_id == b);
    finish(q, &second);
    assert(vx_batch_scheduler_round(q) == 1);
    assert(vx_batch_scheduler_state(q, c) == VX_BATCH_REQUEST_UNKNOWN); /* bounded results */
    VxBatchTelemetry info; vx_batch_scheduler_telemetry(q, &info);
    assert(info.dispatches == 2 && info.rows_dispatched == 24 && info.rows_useful == 9);
    assert(info.completed == 2 && info.cancelled == 1 && info.retained_results == 2);
    assert(info.device_busy_micros <= info.wall_micros);
    vx_batch_scheduler_destroy(q);
}

static void decode(void) {
    VxPagedKVOptions pages = {0};
    pages.lanes = 2; pages.page_tokens = 2; pages.lane_token_capacity = 8; pages.max_pages = 6;
    VxPagedKVCache* cache = vx_paged_kv_create(&pages); assert(cache);
    VxBatchSchedulerOptions options = {0}; options.cache = cache; options.max_lanes = 2;
    VxBatchScheduler* q = vx_batch_scheduler_create(&options); assert(q);
    int a, b, c; char prefix[500]; memset(prefix, 'k', sizeof(prefix)-1); prefix[499] = 0;
    assert(vx_batch_scheduler_submit_llm(q, NULL, NULL, NULL, 2, 3, prefix, NULL, &a) == VX_BATCH_OK);
    VxBatchDispatchView view;
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK);
    assert(view.metadata->kind == VX_BATCH_KIND_PREFILL && view.works[0].kv_length == 2);
    VxPagedKVTelemetry telemetry; vx_paged_kv_telemetry(cache, &telemetry);
    assert(telemetry.reserved_pages == 1);
    finish(q, &view);
    assert(vx_paged_kv_has_prefix(cache, prefix));
    assert(vx_batch_scheduler_submit_llm(q, NULL, NULL, NULL, 2, 2, prefix, NULL, &b) == VX_BATCH_OK);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK);
    assert(view.metadata->kind == VX_BATCH_KIND_DECODE && view.count == 1);
    assert(view.works[0].request_id == a); /* no decode in admission round */
    finish(q, &view);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK && view.count == 2);
    assert(vx_batch_scheduler_cancel(q, a));
    int old_generation = view.works[0].slot_generation;
    finish(q, &view);
    assert(vx_batch_scheduler_state(q, a) == VX_BATCH_REQUEST_CANCELLED);
    assert(vx_batch_scheduler_state(q, b) == VX_BATCH_REQUEST_DECODING);
    assert(vx_batch_scheduler_submit_llm(q, NULL, NULL, NULL, 2, 3, prefix, NULL, &c) == VX_BATCH_OK);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK);
    assert(view.count == 1 && view.works[0].request_id == b);
    finish(q, &view);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK);
    assert(view.works[0].request_id == c && view.works[0].slot_generation > old_generation);
    uint64_t revoked = view.id;
    assert(vx_batch_scheduler_close_async(q, 0) == VX_BATCH_OK);
    assert(vx_batch_scheduler_state(q, c) == VX_BATCH_REQUEST_CANCELLED);
    assert(vx_batch_scheduler_complete(q, revoked, NULL, 0, VX_BATCH_STEP_FAILED) == VX_BATCH_INVALID_ARGUMENT);
    vx_paged_kv_telemetry(cache, &telemetry); assert(telemetry.reserved_pages == 0);
    vx_batch_scheduler_destroy(q); vx_paged_kv_destroy(cache);
}

static void failure_and_drain(void) {
    VxPagedKVOptions pages = {0}; pages.lanes = 2; pages.page_tokens = 2; pages.lane_token_capacity = 8;
    VxPagedKVCache* cache = vx_paged_kv_create(&pages); assert(cache);
    VxBatchSchedulerOptions options = {0}; options.cache = cache;
    VxBatchScheduler* q = vx_batch_scheduler_create(&options); assert(q);
    int a, b, c;
    assert(vx_batch_scheduler_submit_llm(q, "a", NULL, NULL, 2, 1, NULL, NULL, &a) == VX_BATCH_OK);
    assert(vx_batch_scheduler_submit_llm(q, "b", NULL, NULL, 2, 1, NULL, NULL, &b) == VX_BATCH_OK);
    assert(vx_batch_scheduler_submit_llm(q, "c", NULL, NULL, 2, 1, NULL, NULL, &c) == VX_BATCH_OK);
    VxBatchDispatchView view;
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK);
    assert(vx_batch_scheduler_complete(q, view.id, NULL, 0, VX_BATCH_STEP_FAILED) == VX_BATCH_OK);
    assert(vx_batch_scheduler_state(q, a) == VX_BATCH_REQUEST_FAILED);
    assert(vx_batch_scheduler_close_async(q, 1) == VX_BATCH_OK);
    assert(vx_batch_scheduler_draining(q));
    assert(vx_batch_scheduler_state(q, c) == VX_BATCH_REQUEST_CANCELLED);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK && view.works[0].request_id == b);
    finish(q, &view);
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK && view.metadata->kind == VX_BATCH_KIND_DECODE);
    finish(q, &view);
    assert(vx_batch_scheduler_state(q, b) == VX_BATCH_REQUEST_COMPLETED);
    assert(!vx_batch_scheduler_draining(q));
    assert(vx_batch_scheduler_next(q, &view) == VX_BATCH_OK && view.id == 0);
    VxPagedKVTelemetry info; vx_paged_kv_telemetry(cache, &info);
    assert(!info.resident_pages && !info.reserved_pages);
    vx_batch_scheduler_destroy(q); vx_paged_kv_destroy(cache);
}

int main(void) {
    stateless(); decode(); failure_and_drain();
    puts("batch queue: async grouping, padding, reservations, cancellation, prefix reuse, retirement and drain PASS");
    return 0;
}
