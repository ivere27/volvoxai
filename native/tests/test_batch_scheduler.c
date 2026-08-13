/*
 * Native batch-policy twin of tests/batch_scheduler.test.mjs.
 *
 * The checks below are behavioural: each one drives the scheduler and reads what
 * it did, rather than asserting that a submission was accepted. A test that
 * only counts the queue cannot tell a working group key from an absent one.
 */

#include "batch_scheduler.h"
#include "paged_kv.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_failures = 0;

#define REQUIRE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg); \
        g_failures++; \
        return; \
    } \
} while (0)

/* The locals are prefixed: a bare `a`/`e` would shadow a caller's variable of
 * that name and silently rewrite what the assertion reads. */
#define REQUIRE_EQ(act, exp, msg) do { \
    long vx_require_actual = (long)(act); \
    long vx_require_expected = (long)(exp); \
    if (vx_require_actual != vx_require_expected) { \
        fprintf(stderr, "FAIL: %s:%d: %s (expected %ld, got %ld)\n", __FILE__, __LINE__, msg, \
                vx_require_expected, vx_require_actual); \
        g_failures++; \
        return; \
    } \
} while (0)

#define REQUIRE_STR_EQ(act, exp, msg) do { \
    if (strcmp((act), (exp)) != 0) { \
        fprintf(stderr, "FAIL: %s:%d: %s (expected '%s', got '%s')\n", \
                __FILE__, __LINE__, msg, (exp), (act)); \
        g_failures++; \
        return; \
    } \
} while (0)

/* ---- recording harness ---------------------------------------------------- */

#define MAX_RECORDED 256

typedef struct {
    char group_key[VX_GROUP_KEY_STRING_MAX];
    char model_id[VX_GROUP_KEY_MODEL_MAX];
    VxBatchRequestKind kind;
    int rows_per_lane;
    int useful_count;
    int padded_count;
    int row_count;
    int useful_rows;
    int total_rows;
    int member_ids[32];
    int member_positions[32];
    int member_tokens[32];
    int member_rows[32];
    int member_padding[32];
    void* member_payloads[32];
} RecordedDispatch;

typedef struct {
    RecordedDispatch dispatches[MAX_RECORDED];
    int dispatch_count;
    /* Values the recorded payloads carried, so transparency can be compared. */
    float outputs[MAX_RECORDED];
    int output_count;
    int fail_every_lane;
} Recorder;

static void record_key(char* dest, size_t size, const VxGroupKey* key) {
    snprintf(dest, size, "%s|%s|%s|%d", key->model_id, key->adapter_revision,
             key->shape_signature_minus_batch, key->rows_per_lane);
}

static VxBatchStatus recording_runner(
    const VxBatchStepWork* works, int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes, void* user) {
    Recorder* rec = (Recorder*)user;
    RecordedDispatch* entry;

    if (rec->dispatch_count >= MAX_RECORDED) return VX_BATCH_STEP_FAILED;
    entry = &rec->dispatches[rec->dispatch_count++];
    memset(entry, 0, sizeof(*entry));
    record_key(entry->group_key, sizeof(entry->group_key), &metadata->group_key);
    snprintf(entry->model_id, sizeof(entry->model_id), "%s", metadata->group_key.model_id);
    entry->kind = metadata->kind;
    entry->rows_per_lane = metadata->group_key.rows_per_lane;
    entry->useful_count = metadata->useful_count;
    entry->padded_count = metadata->padded_count;
    entry->row_count = count;
    entry->useful_rows = metadata->useful_rows;
    entry->total_rows = metadata->total_rows;

    for (int i = 0; i < count && i < 32; i++) {
        entry->member_ids[i] = works[i].request_id;
        entry->member_positions[i] = works[i].position;
        entry->member_tokens[i] = works[i].tokens;
        entry->member_rows[i] = works[i].rows;
        entry->member_padding[i] = works[i].padding;
        entry->member_payloads[i] = works[i].payload;
    }
    for (int i = 0; i < count; i++) {
        outcomes[i].finished = 0;
        outcomes[i].status = rec->fail_every_lane ? VX_BATCH_STEP_FAILED : VX_BATCH_OK;
        outcomes[i].value = works[i].payload;
        /* A stateless payload is a float this harness "computes" on, so a
         * batched pass can be compared against a single one. */
        if (metadata->kind == VX_BATCH_KIND_STATELESS && !works[i].padding &&
            works[i].payload && rec->output_count < MAX_RECORDED) {
            float input = *(const float*)works[i].payload;
            rec->outputs[rec->output_count++] = input * 2.0f + 0.5f;
        }
    }
    return VX_BATCH_OK;
}

static VxPagedKVCache* make_cache(int lanes, int page_tokens, int capacity, int max_pages) {
    VxPagedKVOptions options;
    memset(&options, 0, sizeof(options));
    options.lanes = lanes;
    options.page_tokens = page_tokens;
    options.lane_token_capacity = capacity;
    options.max_pages = max_pages;
    return vx_paged_kv_create(&options);
}

/* ---- Stateless transparency ------------------------------------------------ */

static void test_transparency_stateless(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    float inputs[4] = { 1.5f, 2.5f, 3.5f, 4.5f };
    float oracle[4];
    int ids[4];
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 1024;
    opts.multiple_of = 1;
    memset(&rec, 0, sizeof(rec));

    /* Ground truth: the same arithmetic, one item at a time. */
    for (int i = 0; i < 4; i++) oracle[i] = inputs[i] * 2.0f + 0.5f;

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    REQUIRE(sched != NULL, "create scheduler");

    for (int i = 0; i < 4; i++) {
        REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                       sched, "model-transparency", NULL, "x:float32:[4]", 1,
                       &inputs[i], &ids[i]),
                   VX_BATCH_OK, "submit stateless");
    }

    REQUIRE_EQ(vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked),
               VX_BATCH_OK, "step");
    REQUIRE_EQ(worked, 1, "worked");

    /* One dispatch carrying all four, not four dispatches. */
    REQUIRE_EQ(rec.dispatch_count, 1, "batched into a single dispatch");
    REQUIRE_EQ(rec.dispatches[0].useful_count, 4, "four contributions");
    REQUIRE_EQ(rec.output_count, 4, "four rows computed");
    for (int i = 0; i < 4; i++) {
        REQUIRE(fabsf(rec.outputs[i] - oracle[i]) == 0.0f,
                "batched row is bit-identical to the single-item result");
    }

    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), 4, "result count");
    for (int i = 0; i < 4; i++) {
        const VxBatchResult* res = vx_batch_scheduler_result(sched, i);
        REQUIRE(res != NULL, "res exists");
        REQUIRE_EQ(res->state, VX_BATCH_REQUEST_COMPLETED, "completed");
        REQUIRE(res->value == &inputs[i], "callback outcome value reaches the result");
    }

    vx_batch_scheduler_destroy(sched);
}

/* ---- Deterministic dispatch composition ----------------------------------- */

static void run_mixed_simulation(Recorder* rec) {
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache = make_cache(2, 2, 8, 8);
    static float payload = 3.0f;

    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_lanes = 2;
    opts.token_budget_per_dispatch = 16;
    opts.multiple_of = 1;

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 2, 2, NULL, NULL, NULL);
    vx_batch_scheduler_submit_stateless(sched, "stateless-m", NULL, "d:float32:[2]", 1,
                                          &payload, NULL);

    for (int round = 0; round < 3; round++) {
        int worked = 0;
        vx_batch_scheduler_step(sched, 1000 * (round + 1), recording_runner, rec, &worked);
    }

    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

static void test_determinism(void) {
    Recorder a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));

    run_mixed_simulation(&a);
    run_mixed_simulation(&b);

    /* The claim is that the dispatch *composition* repeats -- which group, in
     * which order, holding which members at which positions. Wall time is a
     * measurement of a machine and is deliberately not part of it. */
    REQUIRE(a.dispatch_count > 0, "the run must actually have dispatched something");
    REQUIRE_EQ(a.dispatch_count, b.dispatch_count, "same number of dispatches");
    for (int i = 0; i < a.dispatch_count; i++) {
        REQUIRE_STR_EQ(a.dispatches[i].group_key, b.dispatches[i].group_key, "same group order");
        REQUIRE_EQ(a.dispatches[i].useful_count, b.dispatches[i].useful_count, "same batch size");
        REQUIRE_EQ(a.dispatches[i].total_rows, b.dispatches[i].total_rows, "same rows");
        for (int m = 0; m < a.dispatches[i].useful_count; m++) {
            REQUIRE_EQ(a.dispatches[i].member_ids[m], b.dispatches[i].member_ids[m],
                       "same members in the same lane order");
            REQUIRE_EQ(a.dispatches[i].member_positions[m], b.dispatches[i].member_positions[m],
                       "same write positions");
        }
    }
}

/* ---- Structural group separation ------------------------------------------ */

static void test_group_separation(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 1024;
    opts.multiple_of = 1;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);

    /* Four groups: model, adapter, signature and rows_per_lane each split one. */
    vx_batch_scheduler_submit_stateless(sched, "model-A", NULL, "sig1", 1, NULL, NULL);
    vx_batch_scheduler_submit_stateless(sched, "model-B", NULL, "sig1", 1, NULL, NULL);
    vx_batch_scheduler_submit_stateless(sched, "model-A", "v1", "sig1", 1, NULL, NULL);
    vx_batch_scheduler_submit_stateless(sched, "model-A", NULL, "sig2", 2, NULL, NULL);
    /* A fifth that belongs with the first: same key in every field. */
    vx_batch_scheduler_submit_stateless(sched, "model-A", NULL, "sig1", 1, NULL, NULL);

    REQUIRE_EQ(vx_batch_scheduler_queue_depth(sched), 5, "queue depth 5");

    VxBatchTelemetry before;
    vx_batch_scheduler_telemetry(sched, &before);
    REQUIRE_EQ(before.group_count, 4, "four distinct group keys are ready");

    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);

    REQUIRE_EQ(rec.dispatch_count, 4, "one dispatch per group, in the same round");
    for (int i = 0; i < rec.dispatch_count; i++) {
        for (int j = i + 1; j < rec.dispatch_count; j++) {
            REQUIRE(strcmp(rec.dispatches[i].group_key, rec.dispatches[j].group_key) != 0,
                    "no two dispatches share a group key");
        }
    }

    /* The group that holds two contributions is the one they agreed on, and
     * every member's rows matches the key the batch was labelled with. */
    for (int i = 0; i < rec.dispatch_count; i++) {
        const RecordedDispatch* entry = &rec.dispatches[i];
        for (int m = 0; m < entry->useful_count; m++) {
            REQUIRE_EQ(entry->member_rows[m], entry->rows_per_lane,
                       "every member occupies the rows the key declares");
        }
    }
    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), 5, "all five retired");

    vx_batch_scheduler_destroy(sched);
}

/* ---- Padding-result isolation --------------------------------------------- */

static void test_padding_isolation(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    float payloads[3] = { 11.0f, 22.0f, 33.0f };
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 1024;
    opts.multiple_of = 4;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    for (int i = 0; i < 3; i++) {
        vx_batch_scheduler_submit_stateless(sched, "pad", NULL, "sig", 1, &payloads[i], NULL);
    }

    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(worked, 1, "step worked");
    REQUIRE_EQ(rec.dispatch_count, 1, "one dispatch");

    const RecordedDispatch* entry = &rec.dispatches[0];
    REQUIRE_EQ(entry->useful_count, 3, "three real contributions");
    REQUIRE_EQ(entry->padded_count, 4, "padded up to multiple_of");
    REQUIRE_EQ(entry->row_count, 4, "the padding row is a real row the callback receives");
    REQUIRE_EQ(entry->member_padding[0], 0, "row 0 is real");
    REQUIRE_EQ(entry->member_padding[2], 0, "row 2 is real");
    REQUIRE_EQ(entry->member_padding[3], 1, "row 3 is padding");
    REQUIRE_EQ(entry->member_ids[3], -1, "a padding row belongs to no request");
    /* A duplicate of the first contribution, not a zero row: zeros through a
     * normalization produce NaN and poison the rows that mattered. */
    REQUIRE_EQ(entry->member_tokens[3], entry->member_tokens[0], "padding copies the first row");
    REQUIRE(entry->member_payloads[3] == entry->member_payloads[0],
            "padding preserves the duplicated input payload");

    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), 3, "only 3 results published");

    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(sched, &telem);
    REQUIRE_EQ(telem.rows_useful, 3, "rows_useful = 3");
    REQUIRE_EQ(telem.rows_dispatched, 4, "rows_dispatched = 4 (padded to multiple_of 4)");
    REQUIRE(fabs(telem.padding_waste - 0.25) < 1e-6, "padding_waste = 0.25");

    vx_batch_scheduler_destroy(sched);
}

static void test_padding_keeps_remainder_queued(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 8;
    opts.token_budget_per_dispatch = 1024;
    opts.multiple_of = 4;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    for (int i = 0; i < 6; i++) {
        vx_batch_scheduler_submit_stateless(sched, "pad", NULL, "sig", 1, NULL, NULL);
    }

    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatches[0].useful_count, 4, "n >= k sends floor(n/k)*k");
    REQUIRE_EQ(rec.dispatches[0].padded_count, 4, "nothing to pad");
    REQUIRE_EQ(vx_batch_scheduler_queue_depth(sched), 2, "the remainder waits");

    vx_batch_scheduler_step(sched, 2000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatches[1].useful_count, 2, "the remainder goes next round");
    REQUIRE_EQ(rec.dispatches[1].padded_count, 4, "padded once it can no longer fill");

    vx_batch_scheduler_destroy(sched);
}

/* ---- Error, cancellation, and drain lifecycle ----------------------------- */

static VxBatchStatus failing_runner(
    const VxBatchStepWork* works, int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes, void* user) {
    (void)works; (void)count; (void)metadata; (void)outcomes; (void)user;
    return VX_BATCH_STEP_FAILED;
}

static void test_error_and_cancellation(void) {
    VxBatchSchedulerOptions opts;
    int id1, id2;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, &id1);
    vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, &id2);

    REQUIRE_EQ(vx_batch_scheduler_cancel(sched, id2), 1, "cancel queued");
    REQUIRE_EQ(vx_batch_scheduler_state(sched, id2), VX_BATCH_REQUEST_CANCELLED, "state cancelled");

    vx_batch_scheduler_step(sched, 1000, failing_runner, NULL, &worked);
    REQUIRE_EQ(vx_batch_scheduler_state(sched, id1), VX_BATCH_REQUEST_FAILED, "state failed on error");

    vx_batch_scheduler_destroy(sched);
}

static void test_close_drain_publishes_everything(void) {
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache = make_cache(1, 4, 16, 4);
    Recorder rec;
    int admitted_id, queued_id;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_queue_depth = 8;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 4, 1, NULL, NULL, &admitted_id);
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 4, 1, NULL, NULL, &queued_id);

    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(vx_batch_scheduler_queue_depth(sched), 1, "the second request is still queued");

    vx_batch_scheduler_close(sched, 2000, recording_runner, &rec, /*drain=*/1);

    /* Draining finishes the admitted work; the queued request never reached a
     * slot, so there is nothing to drain -- but it was accepted, and an
     * accepted request must not vanish. */
    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), 2, "both requests published");
    REQUIRE_EQ(vx_batch_scheduler_state(sched, queued_id), VX_BATCH_REQUEST_CANCELLED,
               "the queued request is cancelled, not abandoned");
    REQUIRE(vx_batch_scheduler_state(sched, admitted_id) == VX_BATCH_REQUEST_COMPLETED ||
            vx_batch_scheduler_state(sched, admitted_id) == VX_BATCH_REQUEST_CANCELLED,
            "the admitted request reached a terminal state");

    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

/* ---- Bounded admission backpressure --------------------------------------- */

static void test_backpressure(void) {
    VxBatchSchedulerOptions opts;
    int id1, id2, id3;

    memset(&opts, 0, sizeof(opts));
    opts.max_queue_depth = 2;
    opts.max_lanes = 2;

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, &id1), VX_BATCH_OK, "sub 1");
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, &id2), VX_BATCH_OK, "sub 2");
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, &id3), VX_BATCH_QUEUE_FULL, "sub 3 rejected");

    REQUIRE_EQ(vx_batch_scheduler_queue_depth(sched), 2, "queue depth bounded");

    vx_batch_scheduler_destroy(sched);
}

/* ---- Dispatch and useful-row telemetry ------------------------------------ */

static void test_token_budget_and_utilization(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 8;
    opts.token_budget_per_dispatch = 4;
    opts.multiple_of = 1;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    for (int i = 0; i < 6; i++) {
        vx_batch_scheduler_submit_stateless(sched, "budget", NULL, "sig", 1, NULL, NULL);
    }

    /* The budget, not max_lanes, is what cuts the first batch at four. */
    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatches[0].useful_count, 4, "token budget caps the batch");
    REQUIRE_EQ(vx_batch_scheduler_queue_depth(sched), 2, "the rest wait a round");

    vx_batch_scheduler_step(sched, 2000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatch_count, 2, "two dispatches");
    REQUIRE_EQ(rec.dispatches[1].useful_count, 2, "the remainder");

    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(sched, &telem);
    REQUIRE_EQ(telem.rows_useful, 6, "six useful rows");
    REQUIRE_EQ(telem.rows_dispatched, 6, "no padding at multiple_of 1");
    /* Wall spans the driven session and every dispatch happens inside it, so
     * this holds as a fact about one clock rather than as a clamp. */
    REQUIRE(telem.device_busy_micros <= telem.wall_micros, "busy fits inside wall");
    REQUIRE(telem.utilization >= 0.0 && telem.utilization <= 1.0, "utilization in range");

    vx_batch_scheduler_destroy(sched);
}

static void test_bounded_queue_depth_window(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 1;
    opts.max_queue_depth = 64;
    opts.token_budget_per_dispatch = 1;
    opts.queue_depth_window = 4;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    for (int i = 0; i < 20; i++) {
        vx_batch_scheduler_submit_stateless(sched, "m", NULL, "s", 1, NULL, NULL);
    }
    for (int round = 0; round < 28; round++) {
        rec.dispatch_count = 0; /* keep the recorder from overflowing */
        vx_batch_scheduler_step(sched, 1000 * (round + 1), recording_runner, &rec, &worked);
    }

    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(sched, &telem);
    REQUIRE_EQ(telem.queue_depth_p50, 0, "the window forgot the deep part");
    REQUIRE_EQ(telem.queue_depth_p99, 0, "the window forgot the deep part");
    REQUIRE_EQ(telem.max_queue_depth_seen, 20, "the high-water mark is still reported in full");

    vx_batch_scheduler_destroy(sched);
}

/* ---- Regressions ----------------------------------------------------------- */

static void test_prefill_position_after_shared_prefix(void) {
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache = make_cache(2, 4, 32, 32);
    Recorder rec;
    int worked = 0;
    int prefill_seen = 0;
    int continuation_position = -1;
    int continuation_tokens = -1;

    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_queue_depth = 8;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);

    /* Publishes an 8-token prefix under 'sys'. */
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 8, 1, "sys", NULL, NULL);
    for (int round = 0; round < 6; round++) {
        vx_batch_scheduler_step(sched, 1000 * (round + 1), recording_runner, &rec, &worked);
    }
    /* Binds those 8 tokens and prefills a 4-token continuation on top. */
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 12, 1, "sys", NULL, NULL);
    for (int round = 0; round < 6; round++) {
        vx_batch_scheduler_step(sched, 10000 + 1000 * round, recording_runner, &rec, &worked);
    }

    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(sched, &telem);
    REQUIRE_EQ(telem.prefix_reuses, 1, "the second request bound the prefix");

    for (int i = 0; i < rec.dispatch_count; i++) {
        if (rec.dispatches[i].kind != VX_BATCH_KIND_PREFILL) continue;
        prefill_seen++;
        if (prefill_seen == 2) {
            continuation_position = rec.dispatches[i].member_positions[0];
            continuation_tokens = rec.dispatches[i].member_tokens[0];
        }
    }
    REQUIRE_EQ(prefill_seen, 2, "two prefills: the publisher and the continuation");
    REQUIRE_EQ(continuation_tokens, 4, "only the continuation is prefilled");
    /* Not zero.  The first eight tokens are pages this lane shares with the
     * publisher, so a prefill starting at zero would rewrite another request's
     * attention state. */
    REQUIRE_EQ(continuation_position, 8, "the continuation writes after the shared prefix");

    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

static void test_more_lanes_than_any_fixed_bound(void) {
    /* 70 lanes: more than the 64-entry scratch arrays this scheduler used to
     * admit into, which left the surplus wedged in PREFILL for good. */
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache = make_cache(70, 4, 16, 200);
    Recorder rec;

    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_queue_depth = 200;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    for (int i = 0; i < 70; i++) {
        vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 4, 1, NULL, NULL, NULL);
    }

    for (int round = 0; round < 400; round++) {
        int worked = 0;
        rec.dispatch_count = 0;
        rec.output_count = 0;
        if (vx_batch_scheduler_queue_depth(sched) == 0) {
            VxBatchTelemetry probe;
            vx_batch_scheduler_telemetry(sched, &probe);
            if (probe.active_slots == 0) break;
        }
        vx_batch_scheduler_step(sched, 1000 * (round + 1), recording_runner, &rec, &worked);
    }

    VxBatchTelemetry telem;
    vx_batch_scheduler_telemetry(sched, &telem);
    REQUIRE_EQ(telem.admitted, 70, "all 70 admitted");
    REQUIRE_EQ(telem.completed, 70, "all 70 completed");
    REQUIRE_EQ(telem.active_slots, 0, "no slot left holding a request that never ran");

    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

static int64_t g_virtual_now = 0;

static int64_t virtual_clock(void* user) {
    (void)user;
    return g_virtual_now;
}

static void test_fill_first_never_holds_decode(void) {
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache = make_cache(4, 4, 32, 32);
    Recorder rec;
    int decode_dispatches = 0;
    int stateless_dispatches = 0;

    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_lanes = 4;
    opts.max_queue_depth = 16;
    opts.policy.mode = VX_DISPATCH_FILL_FIRST;
    opts.policy.max_wait_micros = 1000000;
    /* Arrival stamps and the per-round override have to share a time base:
     * a latency budget compares the two. */
    opts.clock = virtual_clock;
    g_virtual_now = 1000;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    vx_batch_scheduler_submit_llm(sched, "m", NULL, "llm", 4, 4, NULL, NULL, NULL);
    vx_batch_scheduler_submit_stateless(sched, "m", NULL, "partial", 1, NULL, NULL);

    for (int round = 0; round < 4; round++) {
        int worked = 0;
        g_virtual_now += 1000;
        vx_batch_scheduler_step(sched, g_virtual_now, recording_runner, &rec, &worked);
    }
    for (int i = 0; i < rec.dispatch_count; i++) {
        if (rec.dispatches[i].kind == VX_BATCH_KIND_DECODE) decode_dispatches++;
        if (rec.dispatches[i].kind == VX_BATCH_KIND_STATELESS) stateless_dispatches++;
    }
    /* An admitted lane already holds its KV; waiting buys no batching and
     * costs a token of latency per round. */
    REQUIRE(decode_dispatches >= 2, "fill_first must not hold a resident decode");
    REQUIRE_EQ(stateless_dispatches, 0, "the partial stateless group is inside its budget");

    /* Age it past the budget: it goes without needing to fill. */
    int worked = 0;
    g_virtual_now += 2000000;
    vx_batch_scheduler_step(sched, g_virtual_now, recording_runner, &rec, &worked);
    stateless_dispatches = 0;
    for (int i = 0; i < rec.dispatch_count; i++) {
        if (rec.dispatches[i].kind == VX_BATCH_KIND_STATELESS) stateless_dispatches++;
    }
    REQUIRE_EQ(stateless_dispatches, 1, "an aged-out group goes without filling");

    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

static void test_group_order_is_oldest_first(void) {
    VxBatchSchedulerOptions opts;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 1024;
    memset(&rec, 0, sizeof(rec));

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    vx_batch_scheduler_submit_stateless(sched, "m", NULL, "rare", 1, NULL, NULL);
    for (int i = 0; i < 4; i++) {
        vx_batch_scheduler_submit_stateless(sched, "m", NULL, "common", 1, NULL, NULL);
    }

    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatch_count, 2, "both groups go in the round");
    REQUIRE_STR_EQ(rec.dispatches[0].group_key, "m||rare|1",
                   "the group holding the oldest request is dispatched first");

    vx_batch_scheduler_destroy(sched);
}

/* A callback that submits while a round holds pointers into that round's
 * batch used to pull the request records out from under it. */
typedef struct {
    VxBatchScheduler* scheduler;
    int submitted;
} ReentrantContext;

static VxBatchStatus reentrant_runner(
    const VxBatchStepWork* works, int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes, void* user) {
    ReentrantContext* ctx = (ReentrantContext*)user;
    (void)metadata;
    for (int i = 0; i < count; i++) {
        outcomes[i].finished = 0;
        outcomes[i].status = VX_BATCH_OK;
        outcomes[i].value = works[i].payload;
    }
    for (int i = 0; i < 32; i++) {
        if (vx_batch_scheduler_submit_stateless(
                ctx->scheduler, "reentrant", NULL, "sig", 1, NULL, NULL) == VX_BATCH_OK) {
            ctx->submitted++;
        }
    }
    return VX_BATCH_OK;
}

static void test_submitting_from_the_callback_is_safe(void) {
    VxBatchSchedulerOptions opts;
    ReentrantContext ctx;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 2;
    opts.max_queue_depth = 256;

    VxBatchScheduler* sched = vx_batch_scheduler_create(&opts);
    ctx.scheduler = sched;
    ctx.submitted = 0;

    for (int i = 0; i < 4; i++) {
        vx_batch_scheduler_submit_stateless(sched, "a", NULL, "one", 1, NULL, NULL);
        vx_batch_scheduler_submit_stateless(sched, "b", NULL, "two", 1, NULL, NULL);
    }
    vx_batch_scheduler_step(sched, 1000, reentrant_runner, &ctx, &worked);

    REQUIRE(ctx.submitted > 0, "the callback did submit, so the index did grow");
    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), 4,
               "the round's own batches still retired correctly");

    vx_batch_scheduler_destroy(sched);
}

static void test_group_identity_is_structured(void) {
    VxBatchSchedulerOptions opts;
    VxPagedKVCache* cache;
    VxBatchScheduler* sched;
    Recorder rec;
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 32;
    memset(&rec, 0, sizeof(rec));
    sched = vx_batch_scheduler_create(&opts);
    REQUIRE(sched != NULL, "create delimiter scheduler");
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                   sched, "a|b", "c", "d", 1, NULL, NULL),
               VX_BATCH_OK, "submit first delimiter key");
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                   sched, "a", "b|c", "d", 1, NULL, NULL),
               VX_BATCH_OK, "submit second delimiter key");
    {
        VxBatchTelemetry telemetry;
        vx_batch_scheduler_telemetry(sched, &telemetry);
        REQUIRE_EQ(telemetry.group_count, 2,
                   "telemetry uses the same structured group identity");
    }
    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatch_count, 2, "serialized key collision stays in separate batches");
    vx_batch_scheduler_destroy(sched);

    cache = make_cache(1, 2, 8, 4);
    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_lanes = 1;
    opts.token_budget_per_dispatch = 8;
    memset(&rec, 0, sizeof(rec));
    sched = vx_batch_scheduler_create(&opts);
    REQUIRE(sched != NULL, "create kind scheduler");
    vx_batch_scheduler_submit_llm(sched, "same", NULL, "same", 2, 2,
                                    NULL, NULL, NULL);
    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    rec.dispatch_count = 0;
    vx_batch_scheduler_submit_stateless(sched, "same", NULL, "same", 1,
                                          NULL, NULL);
    vx_batch_scheduler_step(sched, 2000, recording_runner, &rec, &worked);
    REQUIRE_EQ(rec.dispatch_count, 2, "decode and stateless kinds never share a batch");
    REQUIRE(rec.dispatches[0].kind != rec.dispatches[1].kind,
            "the two colliding groups retain their execution kind");
    vx_batch_scheduler_destroy(sched);
    vx_paged_kv_destroy(cache);
}

static void test_submission_bounds_and_destroy_cleanup(void) {
    VxBatchSchedulerOptions opts;
    VxBatchScheduler* sched;
    VxPagedKVCache* cache;
    Recorder rec;
    char long_model[VX_GROUP_KEY_MODEL_MAX + 1];
    char long_prompt[VX_PAGED_KV_PREFIX_KEY_MAX + 1];
    int worked = 0;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = INT_MAX;
    REQUIRE(vx_batch_scheduler_create(&opts) == NULL,
            "default queue capacity multiplication cannot overflow");

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 3;
    opts.multiple_of = 4;
    REQUIRE(vx_batch_scheduler_create(&opts) == NULL,
            "multiple_of larger than max_lanes is rejected");

    opts.max_lanes = 4;
    opts.token_budget_per_dispatch = 3;
    sched = vx_batch_scheduler_create(&opts);
    REQUIRE(sched != NULL, "create bounded scheduler");
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                   sched, "m", NULL, "s", 1, NULL, NULL),
               VX_BATCH_INVALID_ARGUMENT,
               "minimum padded dispatch cannot exceed token budget");
    vx_batch_scheduler_destroy(sched);

    memset(long_model, 'm', sizeof(long_model));
    long_model[sizeof(long_model) - 1] = '\0';
    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 1;
    sched = vx_batch_scheduler_create(&opts);
    REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                   sched, long_model, NULL, "s", 1, NULL, NULL),
               VX_BATCH_INVALID_ARGUMENT, "long group identity is rejected, not truncated");
    vx_batch_scheduler_destroy(sched);

    cache = make_cache(1, 2, 8, 4);
    memset(&opts, 0, sizeof(opts));
    opts.cache = cache;
    opts.max_lanes = 1;
    opts.token_budget_per_dispatch = 8;
    sched = vx_batch_scheduler_create(&opts);
    memset(long_prompt, 'p', sizeof(long_prompt));
    long_prompt[sizeof(long_prompt) - 1] = '\0';
    REQUIRE_EQ(vx_batch_scheduler_submit_llm(
                   sched, "m", NULL, "llm", 2, 2, long_prompt, NULL, NULL),
               VX_BATCH_INVALID_ARGUMENT, "long prompt identity is rejected, not truncated");
    memset(&rec, 0, sizeof(rec));
    REQUIRE_EQ(vx_batch_scheduler_submit_llm(
                   sched, "m", NULL, "llm", 2, 2, NULL, NULL, NULL),
               VX_BATCH_OK, "submit active request");
    vx_batch_scheduler_step(sched, 1000, recording_runner, &rec, &worked);
    REQUIRE(vx_paged_kv_lengths(cache)[0] > 0, "request owns a live cache lane");
    vx_batch_scheduler_destroy(sched);
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 0,
               "destroy releases every scheduler-owned cache lane");
    vx_paged_kv_destroy(cache);
}

static VxBatchStatus retention_runner(
    const VxBatchStepWork* works, int count,
    const VxBatchDispatchMetadata* metadata,
    VxBatchStepOutcome* outcomes, void* user) {
    (void)metadata;
    (void)user;
    for (int index = 0; index < count; index++) {
        outcomes[index].status = VX_BATCH_OK;
        outcomes[index].finished = 1;
        outcomes[index].value = works[index].payload;
    }
    return VX_BATCH_OK;
}

static void test_bounded_terminal_retention(void) {
    enum { REQUESTS = 2048, RETAINED = 3 };
    VxBatchSchedulerOptions opts;
    VxBatchScheduler* sched;
    VxBatchTelemetry telemetry;
    int first_id = 0;
    int last_id = 0;
    int worked = 0;
    int payload = 7;

    memset(&opts, 0, sizeof(opts));
    opts.max_lanes = 1;
    opts.max_queue_depth = 1;
    opts.token_budget_per_dispatch = 1;
    opts.max_retained_results = RETAINED;
    sched = vx_batch_scheduler_create(&opts);
    REQUIRE(sched != NULL, "create bounded-retention scheduler");

    for (int index = 0; index < REQUESTS; index++) {
        int id = 0;
        REQUIRE_EQ(vx_batch_scheduler_submit_stateless(
                       sched, "retention", NULL, "scalar", 1,
                       &payload, &id),
                   VX_BATCH_OK, "submit sequential retained request");
        if (index == 0) first_id = id;
        last_id = id;
        worked = 0;
        REQUIRE_EQ(vx_batch_scheduler_step(
                       sched, (int64_t)index + 1, retention_runner, NULL,
                       &worked),
                   VX_BATCH_OK, "complete sequential retained request");
        REQUIRE_EQ(worked, 1, "sequential request dispatched");
        REQUIRE(vx_batch_scheduler_result_count(sched) <= RETAINED,
                "terminal result ring never exceeds its cap");
    }

    REQUIRE_EQ(vx_batch_scheduler_result_count(sched), RETAINED,
               "only the configured terminal window remains");
    for (int index = 0; index < RETAINED; index++) {
        const VxBatchResult* result =
            vx_batch_scheduler_result(sched, index);
        REQUIRE(result != NULL, "retained result exists");
        REQUIRE_EQ(result->request_id,
                   last_id - (RETAINED - 1) + index,
                   "result ring remains oldest-to-newest");
    }
    REQUIRE_EQ(vx_batch_scheduler_state(sched, first_id),
               VX_BATCH_REQUEST_UNKNOWN,
               "evicted request id has explicit unknown state");
    REQUIRE_EQ(vx_batch_scheduler_state(sched, last_id),
               VX_BATCH_REQUEST_COMPLETED,
               "newest retained request remains queryable");
    REQUIRE(vx_batch_scheduler_result_for_request(sched, first_id) == NULL,
            "evicted request result lookup is bounded");
    REQUIRE(vx_batch_scheduler_result_for_request(sched, last_id) != NULL,
            "retained request result lookup succeeds");

    vx_batch_scheduler_telemetry(sched, &telemetry);
    REQUIRE_EQ(telemetry.completed, REQUESTS,
               "lifetime completion telemetry stays cumulative");
    REQUIRE_EQ(telemetry.active_request_records, 0,
               "terminal request records retire at round boundaries");
    REQUIRE_EQ(telemetry.max_request_records_seen, 1,
               "sequential request-record high-water stays constant");
    REQUIRE_EQ(telemetry.retained_results, RETAINED,
               "telemetry reports bounded retained results");
    REQUIRE_EQ(telemetry.max_retained_results, RETAINED,
               "telemetry reports configured result cap");
    vx_batch_scheduler_destroy(sched);
}

int main(void) {
    printf("Running test_batch_scheduler...\n");
    test_transparency_stateless();
    test_determinism();
    test_group_separation();
    test_padding_isolation();
    test_padding_keeps_remainder_queued();
    test_backpressure();
    test_error_and_cancellation();
    test_close_drain_publishes_everything();
    test_token_budget_and_utilization();
    test_bounded_queue_depth_window();
    test_prefill_position_after_shared_prefix();
    test_more_lanes_than_any_fixed_bound();
    test_fill_first_never_holds_decode();
    test_group_order_is_oldest_first();
    test_submitting_from_the_callback_is_safe();
    test_group_identity_is_structured();
    test_submission_bounds_and_destroy_cleanup();
    test_bounded_terminal_retention();

    if (g_failures == 0) {
        printf("All unified scheduler native tests passed!\n");
        return 0;
    } else {
        printf("%d tests FAILED!\n", g_failures);
        return 1;
    }
}
