/*
 * Native half of the continuous-batching contract.  Reads the same
 * `tests/continuous_batching_vectors.json` the TypeScript test reads, so
 * admission order, slot assignment, page allocation and retirement cannot
 * drift between the browser and the robot.
 *
 * The model is synthetic on both sides: one token per step, finish at
 * max_tokens.  What is compared is the scheduling.
 */

#include "continuous_batch_scheduler.h"
#include "paged_kv.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LABELS 32
#define MAX_LANES 32
#define MAX_PAGES_PER_LANE 64

static int g_failures;

static void fail_at(const char* file, int line, const char* case_name,
                    int index, const char* message) {
    fprintf(stderr, "FAIL %s:%d: [%s][%d] %s\n", file, line, case_name, index, message);
    g_failures++;
}

#define REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            fail_at(__FILE__, __LINE__, case_name, index, (message)); \
            return -1; \
        } \
    } while (0)

#define REQUIRE_EQ(actual, expected, message) \
    do { \
        long _actual = (long)(actual); \
        long _expected = (long)(expected); \
        if (_actual != _expected) { \
            char _buffer[256]; \
            snprintf(_buffer, sizeof(_buffer), "%s: expected %ld, got %ld", \
                     (message), _expected, _actual); \
            fail_at(__FILE__, __LINE__, case_name, index, _buffer); \
            return -1; \
        } \
    } while (0)

typedef struct {
    VxPagedKVCache* cache;
    VxContinuousBatchScheduler* scheduler;
    /* Corpus ids are stable labels; the scheduler assigns its own. */
    int label[MAX_LABELS];
    int assigned[MAX_LABELS];
    int label_count;
    int failing[MAX_LABELS];
    int failing_count;
} Harness;

static int harness_id(const Harness* harness, int label) {
    for (int index = 0; index < harness->label_count; index++) {
        if (harness->label[index] == label) return harness->assigned[index];
    }
    return -1;
}

static int harness_label(const Harness* harness, int request_id) {
    for (int index = 0; index < harness->label_count; index++) {
        if (harness->assigned[index] == request_id) return harness->label[index];
    }
    return -1;
}

static void harness_bind(Harness* harness, int label, int request_id) {
    for (int index = 0; index < harness->label_count; index++) {
        if (harness->label[index] == label) {
            harness->assigned[index] = request_id;
            return;
        }
    }
    if (harness->label_count >= MAX_LABELS) return;
    harness->label[harness->label_count] = label;
    harness->assigned[harness->label_count] = request_id;
    harness->label_count++;
}

/* One call carries the round's lanes, so the outcome list is per lane and in
 * the same order.  A synthetic lane outcome failure exercises the v1 atomic
 * contract: the scheduler must roll back and retire every selected member. */
static VxContinuousStatus harness_run_step(const VxContinuousStepWork* works, int count,
                                      VxContinuousStepOutcome* outcomes, void* user) {
    Harness* harness = (Harness*)user;
    for (int lane = 0; lane < count; lane++) {
        const VxContinuousStepWork* work = &works[lane];
        outcomes[lane].finished = 0;
        outcomes[lane].status = VX_CONTINUOUS_OK;
        for (int index = 0; index < harness->failing_count; index++) {
            if (harness->failing[index] != work->request_id) continue;
            harness->failing[index] = harness->failing[--harness->failing_count];
            outcomes[lane].status = VX_CONTINUOUS_STEP_FAILED;
            break;
        }
        if (outcomes[lane].status != VX_CONTINUOUS_OK) continue;
        /* The step must see a page for everything the published length covers.
         * Resolving each one is the cheapest possible stand-in for a decoder
         * and catches a scheduler that publishes a length it did not reserve. */
        for (int position = 0; position < work->kv_length; position++) {
            long slot = -1;
            if (vx_paged_kv_physical_token_index(harness->cache, work->slot, position,
                                                 &slot) != VX_PAGED_KV_OK) {
                outcomes[lane].status = VX_CONTINUOUS_STEP_FAILED;
                break;
            }
        }
    }
    return VX_CONTINUOUS_OK;
}

static const char* status_name(VxContinuousStatus status) {
    switch (status) {
        case VX_CONTINUOUS_OK: return "ok";
        case VX_CONTINUOUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case VX_CONTINUOUS_QUEUE_FULL: return "QUEUE_FULL";
        case VX_CONTINUOUS_SCHEDULER_CLOSED: return "SCHEDULER_CLOSED";
        case VX_CONTINUOUS_STEP_FAILED: return "STEP_FAILED";
    }
    return "?";
}

static const char* state_name(VxContinuousRequestState state) {
    switch (state) {
        case VX_CONTINUOUS_REQUEST_QUEUED: return "queued";
        case VX_CONTINUOUS_REQUEST_PREFILL: return "prefill";
        case VX_CONTINUOUS_REQUEST_DECODING: return "decoding";
        case VX_CONTINUOUS_REQUEST_COMPLETED: return "completed";
        case VX_CONTINUOUS_REQUEST_CANCELLED: return "cancelled";
        case VX_CONTINUOUS_REQUEST_FAILED: return "failed";
    }
    return "?";
}

static int json_int(const cJSON* object, const char* field, int fallback) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, field);
    return cJSON_IsNumber(value) ? (int)value->valuedouble : fallback;
}

static const char* json_string(const cJSON* object, const char* field,
                               const char* fallback) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, field);
    return cJSON_IsString(value) && value->valuestring ? value->valuestring : fallback;
}

static long telemetry_field(const VxContinuousTelemetry* telemetry, const char* field) {
    if (!strcmp(field, "admitted")) return telemetry->admitted;
    if (!strcmp(field, "completed")) return telemetry->completed;
    if (!strcmp(field, "cancelled")) return telemetry->cancelled;
    if (!strcmp(field, "failed")) return telemetry->failed;
    if (!strcmp(field, "queueDepth")) return telemetry->queue_depth;
    if (!strcmp(field, "activeSlots")) return telemetry->active_slots;
    if (!strcmp(field, "freeSlots")) return telemetry->free_slots;
    if (!strcmp(field, "rounds")) return telemetry->rounds;
    if (!strcmp(field, "steps")) return telemetry->steps;
    if (!strcmp(field, "admissionStalls")) return telemetry->admission_stalls;
    if (!strcmp(field, "maxQueueDepthSeen")) return telemetry->max_queue_depth_seen;
    if (!strcmp(field, "queueDelayRounds")) return telemetry->queue_delay_rounds;
    if (!strcmp(field, "prefixReuses")) return telemetry->prefix_reuses;
    if (!strcmp(field, "prefixPublications")) return telemetry->prefix_publications;
    if (!strcmp(field, "prefillTokens")) return telemetry->prefill_tokens;
    if (!strcmp(field, "sharedPromptTokens")) return telemetry->shared_prompt_tokens;
    if (!strcmp(field, "dispatches")) return telemetry->dispatches;
    return -999999;
}

static int apply_operation(Harness* harness, const cJSON* operation,
                           const char* case_name, int index) {
    VxContinuousBatchScheduler* scheduler = harness->scheduler;
    const char* op = json_string(operation, "op", "");
    const char* expected = json_string(operation, "status", "ok");

    if (!strcmp(op, "submit")) {
        int request_id = -1;
        VxContinuousStatus status = vx_continuous_batch_scheduler_submit(
            scheduler, json_int(operation, "promptTokens", -1),
            json_int(operation, "maxTokens", -1),
            json_string(operation, "promptKey", NULL), NULL, &request_id);
        REQUIRE(!strcmp(status_name(status), expected), "submit status");
        if (status == VX_CONTINUOUS_OK) harness_bind(harness, json_int(operation, "id", -1), request_id);
        return 0;
    }
    if (!strcmp(op, "cancel")) {
        int moved = vx_continuous_batch_scheduler_cancel(
            scheduler, harness_id(harness, json_int(operation, "id", -1)));
        REQUIRE_EQ(moved, cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(operation, "value")),
                   "cancel");
        return 0;
    }
    if (!strcmp(op, "fail_next_step")) {
        REQUIRE(harness->failing_count < MAX_LABELS, "failure table is full");
        harness->failing[harness->failing_count++] =
            harness_id(harness, json_int(operation, "id", -1));
        return 0;
    }
    if (!strcmp(op, "step")) {
        int worked = 0;
        REQUIRE_EQ(vx_continuous_batch_scheduler_step(scheduler, harness_run_step, harness, &worked),
                   VX_CONTINUOUS_OK, "step");
        return 0;
    }
    if (!strcmp(op, "run_until_idle")) {
        REQUIRE_EQ(vx_continuous_batch_scheduler_run_until_idle(scheduler, harness_run_step, harness, 0),
                   VX_CONTINUOUS_OK, "run_until_idle");
        return 0;
    }
    if (!strcmp(op, "close")) {
        REQUIRE_EQ(vx_continuous_batch_scheduler_close(
                       scheduler, harness_run_step, harness,
                       cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(operation, "drain"))),
                   VX_CONTINUOUS_OK, "close");
        return 0;
    }
    if (!strcmp(op, "expect_state")) {
        VxContinuousRequestState state = vx_continuous_batch_scheduler_state(
            scheduler, harness_id(harness, json_int(operation, "id", -1)));
        REQUIRE(!strcmp(state_name(state), json_string(operation, "state", "")),
                "request state");
        return 0;
    }
    if (!strcmp(op, "expect_slot")) {
        REQUIRE_EQ(vx_continuous_batch_scheduler_slot_of(
                       scheduler, harness_id(harness, json_int(operation, "id", -1))),
                   json_int(operation, "slot", -2), "slot");
        return 0;
    }
    if (!strcmp(op, "expect_generated")) {
        int request_id = harness_id(harness, json_int(operation, "id", -1));
        int found = 0;
        for (int result = 0; result < vx_continuous_batch_scheduler_result_count(scheduler); result++) {
            const VxContinuousResult* entry = vx_continuous_batch_scheduler_result(scheduler, result);
            if (entry->request_id != request_id) continue;
            REQUIRE_EQ(entry->generated, json_int(operation, "value", -1), "generated");
            found = 1;
        }
        REQUIRE(found, "request has no published result");
        return 0;
    }
    if (!strcmp(op, "expect_queue_depth")) {
        REQUIRE_EQ(vx_continuous_batch_scheduler_queue_depth(scheduler),
                   json_int(operation, "value", -1), "queue depth");
        return 0;
    }
    if (!strcmp(op, "expect_lengths")) {
        const cJSON* expected_kv = cJSON_GetObjectItemCaseSensitive(operation, "kv");
        const int* lengths = vx_paged_kv_lengths(harness->cache);
        const cJSON* entry;
        int lane = 0;
        REQUIRE(cJSON_IsArray(expected_kv), "expect_lengths needs kv");
        REQUIRE_EQ(cJSON_GetArraySize(expected_kv), vx_paged_kv_lanes(harness->cache),
                   "expect_lengths width");
        cJSON_ArrayForEach(entry, expected_kv) {
            REQUIRE_EQ(lengths[lane], (int)entry->valuedouble, "kv length");
            lane++;
        }
        return 0;
    }
    if (!strcmp(op, "expect_page_table")) {
        const cJSON* pages = cJSON_GetObjectItemCaseSensitive(operation, "pages");
        int slot = json_int(operation, "slot", -1);
        const cJSON* entry;
        int logical = 0;
        REQUIRE(cJSON_IsArray(pages), "expect_page_table needs an array");
        cJSON_ArrayForEach(entry, pages) {
            REQUIRE_EQ(vx_paged_kv_physical_page(harness->cache, slot, logical),
                       (int)entry->valuedouble, "page table entry");
            logical++;
        }
        return 0;
    }
    if (!strcmp(op, "expect_results")) {
        const cJSON* order = cJSON_GetObjectItemCaseSensitive(operation, "order");
        const cJSON* states = cJSON_GetObjectItemCaseSensitive(operation, "states");
        const cJSON* entry;
        int position = 0;
        REQUIRE(cJSON_IsArray(order), "expect_results needs order");
        REQUIRE_EQ(vx_continuous_batch_scheduler_result_count(scheduler),
                   cJSON_GetArraySize(order), "result count");
        cJSON_ArrayForEach(entry, order) {
            const VxContinuousResult* result = vx_continuous_batch_scheduler_result(scheduler, position);
            REQUIRE(result != NULL, "missing result");
            REQUIRE_EQ(harness_label(harness, result->request_id),
                       (int)entry->valuedouble, "result order");
            position++;
        }
        if (cJSON_IsArray(states)) {
            position = 0;
            cJSON_ArrayForEach(entry, states) {
                const VxContinuousResult* result = vx_continuous_batch_scheduler_result(scheduler, position);
                REQUIRE(result != NULL, "missing result");
                REQUIRE(!strcmp(state_name(result->state), entry->valuestring),
                        "result state");
                position++;
            }
        }
        return 0;
    }
    if (!strcmp(op, "expect_telemetry")) {
        VxContinuousTelemetry telemetry;
        const cJSON* field;
        vx_continuous_batch_scheduler_telemetry(scheduler, &telemetry);
        cJSON_ArrayForEach(field, operation) {
            long actual;
            /* `note` carries the derivation of the numbers beside them, so the
             * corpus can say *why* a count is what it is without a reader having
             * to re-derive it.  Documentation, not an expectation. */
            if (!strcmp(field->string, "op") || !strcmp(field->string, "note")) continue;
            actual = telemetry_field(&telemetry, field->string);
            REQUIRE(actual != -999999, "unknown telemetry field");
            REQUIRE_EQ(actual, (long)field->valuedouble, field->string);
        }
        return 0;
    }
    fail_at(__FILE__, __LINE__, case_name, index, "unknown operation");
    return -1;
}

static int run_case(const cJSON* test_case) {
    const char* case_name = json_string(test_case, "name", "?");
    const cJSON* options = cJSON_GetObjectItemCaseSensitive(test_case, "options");
    const cJSON* operations = cJSON_GetObjectItemCaseSensitive(test_case, "operations");
    const cJSON* operation;
    VxPagedKVOptions cache_options;
    Harness harness;
    int index = -1;
    int failed = 0;

    memset(&harness, 0, sizeof(harness));
    memset(&cache_options, 0, sizeof(cache_options));
    cache_options.lanes = json_int(options, "slots", 0);
    cache_options.page_tokens = json_int(options, "pageTokens", 0);
    cache_options.lane_token_capacity = json_int(options, "laneTokenCapacity", 0);
    cache_options.max_pages = json_int(options, "maxPages", 0);
    cache_options.bytes_per_token = 1;
    cache_options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache_options.clear_on_recycle = 1;

    harness.cache = vx_paged_kv_create(&cache_options);
    if (!harness.cache) {
        fail_at(__FILE__, __LINE__, case_name, -1, "cache create failed");
        return -1;
    }
    if (vx_paged_kv_lanes(harness.cache) > MAX_LANES ||
        vx_paged_kv_pages_per_lane(harness.cache) > MAX_PAGES_PER_LANE) {
        fail_at(__FILE__, __LINE__, case_name, -1, "case exceeds the driver's bounds");
        vx_paged_kv_destroy(harness.cache);
        return -1;
    }
    harness.scheduler = vx_continuous_batch_scheduler_create(
        harness.cache, json_int(options, "maxQueueDepth", 0));
    if (!harness.scheduler) {
        fail_at(__FILE__, __LINE__, case_name, -1, "scheduler create failed");
        vx_paged_kv_destroy(harness.cache);
        return -1;
    }

    cJSON_ArrayForEach(operation, operations) {
        index++;
        if (apply_operation(&harness, operation, case_name, index) != 0) {
            failed = 1;
            break;
        }
    }
    vx_continuous_batch_scheduler_destroy(harness.scheduler);
    vx_paged_kv_destroy(harness.cache);
    if (failed) return -1;
    printf("ok %s\n", case_name);
    return 0;
}

/* The property a corpus cannot express: the same schedule twice. */
static int test_repeated_schedules_agree(void) {
    const char* case_name = "repeatability";
    int index = -1;
    char first[256] = {0};
    char second[256] = {0};
    for (int attempt = 0; attempt < 2; attempt++) {
        VxPagedKVOptions options;
        Harness harness;
        char* text = attempt == 0 ? first : second;
        size_t used = 0;
        memset(&harness, 0, sizeof(harness));
        memset(&options, 0, sizeof(options));
        options.lanes = 3;
        options.page_tokens = 2;
        options.lane_token_capacity = 16;
        options.max_pages = 12;
        options.bytes_per_token = 1;
        options.policy = VX_PAGED_KV_POLICY_PAGED;
        options.clear_on_recycle = 1;
        harness.cache = vx_paged_kv_create(&options);
        REQUIRE(harness.cache != NULL, "cache create");
        harness.scheduler = vx_continuous_batch_scheduler_create(harness.cache, 16);
        REQUIRE(harness.scheduler != NULL, "scheduler create");
        {
            const int lengths[8] = {1, 5, 2, 4, 3, 1, 6, 2};
            for (int request = 0; request < 8; request++) {
                vx_continuous_batch_scheduler_submit(harness.scheduler, 2 + (request % 3) * 2,
                                          lengths[request], NULL, NULL, NULL);
            }
        }
        vx_continuous_batch_scheduler_run_until_idle(harness.scheduler, harness_run_step, &harness, 0);
        for (int result = 0;
             result < vx_continuous_batch_scheduler_result_count(harness.scheduler); result++) {
            const VxContinuousResult* entry = vx_continuous_batch_scheduler_result(harness.scheduler, result);
            used += (size_t)snprintf(text + used, 256 - used, "%d:%s:%d ",
                                     entry->request_id, state_name(entry->state),
                                     entry->generated);
        }
        vx_continuous_batch_scheduler_destroy(harness.scheduler);
        vx_paged_kv_destroy(harness.cache);
    }
    REQUIRE(!strcmp(first, second), "an identical schedule must produce an identical result list");
    printf("ok an identical schedule produces an identical result list\n");
    return 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/continuous_batching_vectors.json";
    FILE* file = fopen(path, "rb");
    long size;
    char* text;
    cJSON* root;
    const cJSON* cases;
    const cJSON* test_case;

    if (!file) {
        fprintf(stderr, "FAIL cannot open %s\n", path);
        return 1;
    }
    fseek(file, 0, SEEK_END);
    size = ftell(file);
    fseek(file, 0, SEEK_SET);
    text = (char*)malloc((size_t)size + 1);
    if (!text || fread(text, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "FAIL cannot read %s\n", path);
        fclose(file);
        free(text);
        return 1;
    }
    text[size] = '\0';
    fclose(file);

    root = cJSON_Parse(text);
    free(text);
    if (!root) {
        fprintf(stderr, "FAIL cannot parse %s\n", path);
        return 1;
    }
    if (strcmp(json_string(root, "version", ""), "volvox-continuous-batching/v1")) {
        fprintf(stderr, "FAIL unexpected corpus version\n");
        cJSON_Delete(root);
        return 1;
    }
    cases = cJSON_GetObjectItemCaseSensitive(root, "cases");
    if (!cJSON_IsArray(cases) || cJSON_GetArraySize(cases) == 0) {
        fprintf(stderr, "FAIL corpus carries no cases\n");
        cJSON_Delete(root);
        return 1;
    }
    cJSON_ArrayForEach(test_case, cases) run_case(test_case);
    cJSON_Delete(root);

    test_repeated_schedules_agree();

    if (g_failures) {
        fprintf(stderr, "FAIL %d continuous batching check(s)\n", g_failures);
        return 1;
    }
    printf("PASS continuous batching vectors\n");
    return 0;
}
