/*
 * Native half of the paged KV contract.  Reads the same
 * `tests/paged_kv_vectors.json` the TypeScript test reads, so the two
 * allocators cannot drift.  The shape system already has a corpus like this;
 * the decode/row contract had none, and page tables plus per-lane lengths are
 * exactly the surface that was about to be written twice.
 */

#include "paged_kv.h"

#include "cJSON.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RESERVATIONS 8
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
    char id[32];
    VxPagedKVReservation reservation;
    int live;
} NamedReservation;

static const char* status_name(VxPagedKVStatus status) {
    switch (status) {
        case VX_PAGED_KV_OK: return "ok";
        case VX_PAGED_KV_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case VX_PAGED_KV_CAPACITY_EXHAUSTED: return "CAPACITY_EXHAUSTED";
        case VX_PAGED_KV_PREFIX_NOT_FOUND: return "PREFIX_NOT_FOUND";
        case VX_PAGED_KV_PREFIX_CONFLICT: return "PREFIX_CONFLICT";
        case VX_PAGED_KV_STALE_PAGE: return "STALE_PAGE";
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

static NamedReservation* reservation_slot(NamedReservation* table, const char* id) {
    int free_slot = -1;
    for (int index = 0; index < MAX_RESERVATIONS; index++) {
        if (table[index].live && !strcmp(table[index].id, id)) return &table[index];
        if (!table[index].live && free_slot < 0) free_slot = index;
    }
    if (free_slot < 0) return NULL;
    snprintf(table[free_slot].id, sizeof(table[free_slot].id), "%s", id);
    table[free_slot].live = 1;
    return &table[free_slot];
}

static long telemetry_field(const VxPagedKVTelemetry* telemetry, const char* field) {
    if (!strcmp(field, "logicalBytes")) return telemetry->logical_bytes;
    if (!strcmp(field, "residentBytes")) return telemetry->resident_bytes;
    if (!strcmp(field, "reservedBytes")) return telemetry->reserved_bytes;
    if (!strcmp(field, "residentPages")) return telemetry->resident_pages;
    if (!strcmp(field, "reservedPages")) return telemetry->reserved_pages;
    if (!strcmp(field, "freePages")) return telemetry->free_pages;
    if (!strcmp(field, "sharedPages")) return telemetry->shared_pages;
    if (!strcmp(field, "fragmentationBytes")) return telemetry->fragmentation_bytes;
    if (!strcmp(field, "highWaterPages")) return telemetry->high_water_pages;
    if (!strcmp(field, "prefixHits")) return telemetry->prefix_hits;
    if (!strcmp(field, "prefixMisses")) return telemetry->prefix_misses;
    if (!strcmp(field, "copyOnWrites")) return telemetry->copy_on_writes;
    if (!strcmp(field, "evictions")) return telemetry->evictions;
    if (!strcmp(field, "allocationFailures")) return telemetry->allocation_failures;
    return -999999;
}

static int apply_operation(VxPagedKVCache* cache, const cJSON* operation,
                           NamedReservation* reservations,
                           const char* case_name, int index) {
    const char* op = json_string(operation, "op", "");
    const char* expected = json_string(operation, "status", "ok");

    if (!strcmp(op, "append")) {
        VxPagedKVStatus status = vx_paged_kv_append(
            cache, json_int(operation, "lane", -1), json_int(operation, "tokens", -1));
        REQUIRE(!strcmp(status_name(status), expected), "append status");
        return 0;
    }
    if (!strcmp(op, "reserve")) {
        NamedReservation* slot = reservation_slot(
            reservations, json_string(operation, "id", ""));
        VxPagedKVStatus status;
        REQUIRE(slot != NULL, "reservation table is full");
        status = vx_paged_kv_reserve(cache, json_int(operation, "lane", -1),
                                     json_int(operation, "tokens", -1),
                                     &slot->reservation);
        REQUIRE(!strcmp(status_name(status), expected), "reserve status");
        if (status != VX_PAGED_KV_OK) {
            vx_paged_kv_reservation_dispose(&slot->reservation);
            slot->live = 0;
        }
        return 0;
    }
    if (!strcmp(op, "commit") || !strcmp(op, "rollback")) {
        NamedReservation* slot = reservation_slot(
            reservations, json_string(operation, "id", ""));
        VxPagedKVStatus status;
        REQUIRE(slot != NULL && slot->live, "unknown reservation id");
        status = !strcmp(op, "commit")
            ? vx_paged_kv_commit(cache, &slot->reservation)
            : vx_paged_kv_rollback(cache, &slot->reservation);
        REQUIRE_EQ(status, VX_PAGED_KV_OK, "reservation transition");
        vx_paged_kv_reservation_dispose(&slot->reservation);
        slot->live = 0;
        return 0;
    }
    if (!strcmp(op, "release_lane")) {
        REQUIRE_EQ(vx_paged_kv_release_lane(cache, json_int(operation, "lane", -1)),
                   VX_PAGED_KV_OK, "release_lane");
        return 0;
    }
    if (!strcmp(op, "reset")) {
        vx_paged_kv_reset(cache);
        return 0;
    }
    if (!strcmp(op, "publish_prefix")) {
        VxPagedKVStatus status = vx_paged_kv_publish_prefix(
            cache, json_string(operation, "key", ""), json_int(operation, "lane", -1),
            json_int(operation, "tokens", -1));
        REQUIRE(!strcmp(status_name(status), expected), "publish_prefix status");
        return 0;
    }
    if (!strcmp(op, "acquire_prefix")) {
        int tokens = -1;
        VxPagedKVStatus status = vx_paged_kv_acquire_prefix(
            cache, json_string(operation, "key", ""), json_int(operation, "lane", -1),
            &tokens);
        REQUIRE(!strcmp(status_name(status), expected), "acquire_prefix status");
        if (status == VX_PAGED_KV_OK &&
            cJSON_GetObjectItemCaseSensitive(operation, "tokens")) {
            REQUIRE_EQ(tokens, json_int(operation, "tokens", -1), "acquire_prefix tokens");
        }
        return 0;
    }
    if (!strcmp(op, "copy_on_write")) {
        int from = -1;
        int to = -1;
        VxPagedKVStatus status = vx_paged_kv_copy_on_write(
            cache, json_int(operation, "lane", -1), json_int(operation, "logical", -1),
            &from, &to);
        REQUIRE(!strcmp(status_name(status), expected), "copy_on_write status");
        if (status != VX_PAGED_KV_OK) return 0;
        REQUIRE_EQ(from, json_int(operation, "from", -1), "copy_on_write from");
        REQUIRE_EQ(to, json_int(operation, "to", -1), "copy_on_write to");
        return 0;
    }
    if (!strcmp(op, "evict")) {
        REQUIRE_EQ(vx_paged_kv_evict(cache, json_int(operation, "pages", 0)),
                   json_int(operation, "reclaimed", -1), "evict reclaimed");
        return 0;
    }
    if (!strcmp(op, "expect_page_table")) {
        const cJSON* pages = cJSON_GetObjectItemCaseSensitive(operation, "pages");
        int lane = json_int(operation, "lane", -1);
        int logical = 0;
        const cJSON* entry;
        REQUIRE(cJSON_IsArray(pages), "expect_page_table needs an array");
        REQUIRE_EQ(cJSON_GetArraySize(pages), vx_paged_kv_pages_per_lane(cache),
                   "expect_page_table width");
        cJSON_ArrayForEach(entry, pages) {
            REQUIRE_EQ(vx_paged_kv_physical_page(cache, lane, logical),
                       (int)entry->valuedouble, "page table entry");
            logical++;
        }
        return 0;
    }
    if (!strcmp(op, "expect_lengths")) {
        const int* kv = vx_paged_kv_lengths(cache);
        const int* query = vx_paged_kv_query_lengths(cache);
        const cJSON* expected_kv = cJSON_GetObjectItemCaseSensitive(operation, "kv");
        const cJSON* expected_query = cJSON_GetObjectItemCaseSensitive(operation, "query");
        const cJSON* entry;
        int lane = 0;
        REQUIRE(cJSON_IsArray(expected_kv), "expect_lengths needs kv");
        REQUIRE_EQ(cJSON_GetArraySize(expected_kv), vx_paged_kv_lanes(cache),
                   "expect_lengths kv width");
        cJSON_ArrayForEach(entry, expected_kv) {
            REQUIRE_EQ(kv[lane], (int)entry->valuedouble, "kv length");
            lane++;
        }
        if (cJSON_IsArray(expected_query)) {
            lane = 0;
            cJSON_ArrayForEach(entry, expected_query) {
                REQUIRE_EQ(query[lane], (int)entry->valuedouble, "query length");
                lane++;
            }
        }
        return 0;
    }
    if (!strcmp(op, "expect_tokens")) {
        const cJSON* tokens = cJSON_GetObjectItemCaseSensitive(operation, "tokens");
        int lane = json_int(operation, "lane", -1);
        long gathered[MAX_LANES * MAX_PAGES_PER_LANE];
        const cJSON* entry;
        int position = 0;
        REQUIRE(cJSON_IsArray(tokens), "expect_tokens needs an array");
        REQUIRE(lane >= 0 && lane < vx_paged_kv_lanes(cache), "expect_tokens lane");
        REQUIRE_EQ(cJSON_GetArraySize(tokens), vx_paged_kv_lengths(cache)[lane],
                   "expect_tokens length");
        REQUIRE_EQ(vx_paged_kv_gather_active_tokens(cache, lane, gathered),
                   VX_PAGED_KV_OK, "gather");
        cJSON_ArrayForEach(entry, tokens) {
            REQUIRE_EQ(gathered[position], (long)entry->valuedouble,
                       "physical token index");
            position++;
        }
        return 0;
    }
    if (!strcmp(op, "expect_contiguous")) {
        REQUIRE_EQ(vx_paged_kv_lane_is_contiguous(cache, json_int(operation, "lane", -1)),
                   cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(operation, "value")),
                   "lane contiguity");
        return 0;
    }
    if (!strcmp(op, "expect_lane_generation")) {
        REQUIRE_EQ(vx_paged_kv_lane_generations(cache)[json_int(operation, "lane", 0)],
                   json_int(operation, "value", -1), "lane generation");
        return 0;
    }
    if (!strcmp(op, "expect_page_generation")) {
        REQUIRE_EQ(vx_paged_kv_page_generation(cache, json_int(operation, "page", -1)),
                   json_int(operation, "value", -1), "page generation");
        return 0;
    }
    if (!strcmp(op, "expect_page_shared")) {
        REQUIRE_EQ(vx_paged_kv_page_is_shared(cache, json_int(operation, "lane", -1),
                                              json_int(operation, "logical", -1)),
                   cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(operation, "value")),
                   "page shared");
        return 0;
    }
    if (!strcmp(op, "expect_has_prefix")) {
        REQUIRE_EQ(vx_paged_kv_has_prefix(cache, json_string(operation, "key", "")),
                   cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(operation, "value")),
                   "has prefix");
        return 0;
    }
    if (!strcmp(op, "expect_telemetry")) {
        VxPagedKVTelemetry telemetry;
        const cJSON* field;
        vx_paged_kv_telemetry(cache, &telemetry);
        cJSON_ArrayForEach(field, operation) {
            long actual;
            if (!strcmp(field->string, "op")) continue;
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
    const cJSON* expect_create_failure =
        cJSON_GetObjectItemCaseSensitive(test_case, "expectCreateFailure");
    const char* policy = json_string(options, "policy", "paged");
    NamedReservation reservations[MAX_RESERVATIONS];
    VxPagedKVOptions resolved;
    VxPagedKVCache* cache;
    const cJSON* operation;
    int index = -1;

    memset(reservations, 0, sizeof(reservations));
    memset(&resolved, 0, sizeof(resolved));
    resolved.lanes = json_int(options, "lanes", 0);
    resolved.page_tokens = json_int(options, "pageTokens", 0);
    resolved.lane_token_capacity = json_int(options, "laneTokenCapacity", 0);
    resolved.max_pages = json_int(options, "maxPages", 0);
    resolved.bytes_per_token = json_int(options, "bytesPerToken", 1);
    resolved.policy = !strcmp(policy, "contiguous")
        ? VX_PAGED_KV_POLICY_CONTIGUOUS : VX_PAGED_KV_POLICY_PAGED;
    resolved.clear_on_recycle = 1;

    cache = vx_paged_kv_create(&resolved);
    if (cJSON_IsTrue(expect_create_failure)) {
        if (cache) {
            fail_at(__FILE__, __LINE__, case_name, -1, "create should have failed");
            vx_paged_kv_destroy(cache);
            return -1;
        }
        printf("ok %s\n", case_name);
        return 0;
    }
    if (!cache) {
        fail_at(__FILE__, __LINE__, case_name, -1, "create failed");
        return -1;
    }
    if (vx_paged_kv_lanes(cache) > MAX_LANES ||
        vx_paged_kv_pages_per_lane(cache) > MAX_PAGES_PER_LANE) {
        fail_at(__FILE__, __LINE__, case_name, -1, "case exceeds the driver's bounds");
        vx_paged_kv_destroy(cache);
        return -1;
    }

    cJSON_ArrayForEach(operation, operations) {
        index++;
        if (apply_operation(cache, operation, reservations, case_name, index) != 0) {
            for (int slot = 0; slot < MAX_RESERVATIONS; slot++) {
                vx_paged_kv_reservation_dispose(&reservations[slot].reservation);
            }
            vx_paged_kv_destroy(cache);
            return -1;
        }
    }
    for (int slot = 0; slot < MAX_RESERVATIONS; slot++) {
        vx_paged_kv_reservation_dispose(&reservations[slot].reservation);
    }
    vx_paged_kv_destroy(cache);
    printf("ok %s\n", case_name);
    return 0;
}

/* The identity map is the whole reason contiguous is a policy and not a second
 * addressing path, so it is asserted in code as well as in the corpus. */
static int test_identity_map_reduces_to_contiguous_arithmetic(void) {
    const char* case_name = "identity map";
    VxPagedKVOptions options;
    VxPagedKVCache* cache;
    memset(&options, 0, sizeof(options));
    options.lanes = 3;
    options.page_tokens = 4;
    options.lane_token_capacity = 32;
    options.policy = VX_PAGED_KV_POLICY_CONTIGUOUS;
    cache = vx_paged_kv_create(&options);
    if (!cache) { fail_at(__FILE__, __LINE__, case_name, -1, "create"); return -1; }
    for (int lane = 0; lane < 3; lane++) {
        long expected_base = (long)lane * options.lane_token_capacity;
        if (vx_paged_kv_append(cache, lane, 21) != VX_PAGED_KV_OK) {
            fail_at(__FILE__, __LINE__, case_name, lane, "append");
            vx_paged_kv_destroy(cache);
            return -1;
        }
        for (int position = 0; position < 21; position++) {
            long slot = -1;
            if (vx_paged_kv_physical_token_index(cache, lane, position, &slot) !=
                VX_PAGED_KV_OK || slot != expected_base + position) {
                fail_at(__FILE__, __LINE__, case_name, position,
                        "identity map is not lane * capacity + position");
                vx_paged_kv_destroy(cache);
                return -1;
            }
        }
        if (!vx_paged_kv_lane_is_contiguous(cache, lane)) {
            fail_at(__FILE__, __LINE__, case_name, lane, "lane is not contiguous");
            vx_paged_kv_destroy(cache);
            return -1;
        }
    }
    vx_paged_kv_destroy(cache);
    printf("ok identity map reduces to contiguous arithmetic\n");
    return 0;
}

/* Committing and then rolling back the same handle would double-subtract the
 * reserved-page count and republish the pre-commit length.  Silent allocator
 * corruption found much later is the worst failure mode available here, so the
 * second transition is refused — in both runtimes, hence in both tests. */
static int test_reservation_settles_once(void) {
    const char* case_name = "reservation settles once";
    VxPagedKVOptions options;
    VxPagedKVReservation reservation;
    VxPagedKVCache* cache;
    VxPagedKVTelemetry telemetry;
    int index = -1;
    memset(&options, 0, sizeof(options));
    options.lanes = 1;
    options.page_tokens = 2;
    options.lane_token_capacity = 8;
    options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache = vx_paged_kv_create(&options);
    if (!cache) { fail_at(__FILE__, __LINE__, case_name, -1, "create"); return -1; }

    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 4, &reservation), VX_PAGED_KV_OK, "reserve");
    REQUIRE_EQ(vx_paged_kv_commit(cache, &reservation), VX_PAGED_KV_OK, "commit");
    REQUIRE_EQ(vx_paged_kv_rollback(cache, &reservation),
               VX_PAGED_KV_INVALID_ARGUMENT, "rollback after commit");
    REQUIRE_EQ(vx_paged_kv_commit(cache, &reservation),
               VX_PAGED_KV_INVALID_ARGUMENT, "second commit");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 4, "length after refused rollback");
    vx_paged_kv_telemetry(cache, &telemetry);
    REQUIRE_EQ(telemetry.reserved_pages, 0, "reserved pages");
    vx_paged_kv_reservation_dispose(&reservation);

    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 2, &reservation), VX_PAGED_KV_OK, "reserve again");
    REQUIRE_EQ(vx_paged_kv_rollback(cache, &reservation), VX_PAGED_KV_OK, "rollback");
    REQUIRE_EQ(vx_paged_kv_commit(cache, &reservation),
               VX_PAGED_KV_INVALID_ARGUMENT, "commit after rollback");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 4, "length after refused commit");
    vx_paged_kv_reservation_dispose(&reservation);
    vx_paged_kv_destroy(cache);
    printf("ok a reservation settles exactly once\n");
    return 0;
}

static int test_checked_sizes_and_invalid_append(void) {
    const char* case_name = "checked sizes and invalid append";
    VxPagedKVOptions options;
    VxPagedKVCache* cache;
    VxPagedKVReservation reservation;
    int index = -1;

    memset(&options, 0, sizeof(options));
    options.lanes = 65537;
    options.page_tokens = 1;
    options.lane_token_capacity = 65537;
    options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache = vx_paged_kv_create(&options);
    REQUIRE(cache == NULL, "overflowing private page table is rejected");

    options.lanes = 1;
    options.page_tokens = INT_MAX;
    options.lane_token_capacity = INT_MAX;
    options.max_pages = 1;
    cache = vx_paged_kv_create(&options);
    REQUIRE(cache != NULL, "largest representable lane creates");
    REQUIRE_EQ(vx_paged_kv_append(cache, -1, 1), VX_PAGED_KV_INVALID_ARGUMENT,
               "invalid append lane");
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, INT_MAX, &reservation),
               VX_PAGED_KV_OK, "reserve full int capacity");
    REQUIRE_EQ(vx_paged_kv_commit(cache, &reservation), VX_PAGED_KV_OK,
               "commit full int capacity");
    vx_paged_kv_reservation_dispose(&reservation);
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 1, &reservation),
               VX_PAGED_KV_CAPACITY_EXHAUSTED, "reserve addition cannot overflow");
    vx_paged_kv_reservation_dispose(&reservation);
    vx_paged_kv_destroy(cache);

    memset(&options, 0, sizeof(options));
    options.lanes = 1;
    options.page_tokens = 1;
    options.lane_token_capacity = 1;
    options.max_pages = -1;
    REQUIRE(vx_paged_kv_create(&options) == NULL, "negative max_pages is rejected");
    printf("ok checked sizes and invalid append\n");
    return 0;
}

static int test_reservation_exclusivity_and_identity(void) {
    const char* case_name = "reservation exclusivity and identity";
    VxPagedKVOptions options;
    VxPagedKVReservation first, blocked, stale, foreign, second, other_reservation;
    VxPagedKVCache* cache;
    VxPagedKVCache* other;
    int index = -1;

    memset(&options, 0, sizeof(options));
    options.lanes = 1;
    options.page_tokens = 2;
    options.lane_token_capacity = 8;
    options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache = vx_paged_kv_create(&options);
    REQUIRE(cache != NULL, "create");

    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 2, &first), VX_PAGED_KV_OK,
               "first reservation");
    stale = first;
    foreign = first;
    other = vx_paged_kv_create(&options);
    REQUIRE(other != NULL, "other cache create");
    REQUIRE_EQ(vx_paged_kv_reserve(other, 0, 2, &other_reservation),
               VX_PAGED_KV_OK, "lookalike reservation in other cache");
    REQUIRE_EQ(vx_paged_kv_commit(other, &foreign), VX_PAGED_KV_INVALID_ARGUMENT,
               "reservation cannot settle another cache");
    REQUIRE_EQ(vx_paged_kv_commit(other, &other_reservation), VX_PAGED_KV_OK,
               "other cache reservation remains current");
    vx_paged_kv_reservation_dispose(&other_reservation);
    vx_paged_kv_destroy(other);
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 1, &blocked),
               VX_PAGED_KV_INVALID_ARGUMENT, "second open reservation is rejected");
    vx_paged_kv_reservation_dispose(&blocked);
    REQUIRE_EQ(vx_paged_kv_release_lane(cache, 0), VX_PAGED_KV_INVALID_ARGUMENT,
               "active lane cannot be retired");
    vx_paged_kv_reset(cache);
    REQUIRE_EQ(vx_paged_kv_commit(cache, &first), VX_PAGED_KV_OK,
               "reset leaves active reservation intact");

    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 2, &second), VX_PAGED_KV_OK,
               "next reservation");
    REQUIRE_EQ(vx_paged_kv_commit(cache, &stale), VX_PAGED_KV_INVALID_ARGUMENT,
               "copied old handle cannot settle a new reservation");
    vx_paged_kv_reservation_dispose(&first);
    REQUIRE_EQ(vx_paged_kv_rollback(cache, &second), VX_PAGED_KV_OK,
               "current reservation still rolls back");
    vx_paged_kv_reservation_dispose(&second);
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 2, "stale handle did not change length");
    REQUIRE_EQ(vx_paged_kv_release_lane(cache, 0), VX_PAGED_KV_OK, "release settled lane");
    vx_paged_kv_destroy(cache);
    printf("ok reservation exclusivity and identity\n");
    return 0;
}

static int test_batch_settlement_is_atomic(void) {
    const char* case_name = "batch reservation settlement";
    VxPagedKVOptions options;
    VxPagedKVReservation reservations[2] = {{0}};
    VxPagedKVCache* cache;
    uint64_t second_id;
    int index = -1;

    memset(&options, 0, sizeof(options));
    options.lanes = 2;
    options.page_tokens = 2;
    options.lane_token_capacity = 8;
    options.max_pages = 8;
    cache = vx_paged_kv_create(&options);
    REQUIRE(cache != NULL, "create");
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 2, &reservations[0]),
               VX_PAGED_KV_OK, "reserve first lane");
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 1, 2, &reservations[1]),
               VX_PAGED_KV_OK, "reserve second lane");
    second_id = reservations[1].id;
    reservations[1].id++;
    REQUIRE_EQ(vx_paged_kv_commit_batch(cache, reservations, 2),
               VX_PAGED_KV_INVALID_ARGUMENT,
               "one stale member rejects the whole commit");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 0,
               "first length stayed provisional");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[1], 0,
               "second length stayed provisional");
    REQUIRE(reservations[0].open && reservations[1].open,
            "failed batch validation settled no handle");
    reservations[1].id = second_id;
    REQUIRE_EQ(vx_paged_kv_commit_batch(cache, reservations, 2),
               VX_PAGED_KV_OK, "valid batch commits together");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 2, "first committed length");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[1], 2, "second committed length");

    REQUIRE_EQ(vx_paged_kv_reserve(cache, 0, 1, &reservations[0]),
               VX_PAGED_KV_OK, "reserve first decode");
    REQUIRE_EQ(vx_paged_kv_reserve(cache, 1, 1, &reservations[1]),
               VX_PAGED_KV_OK, "reserve second decode");
    REQUIRE_EQ(vx_paged_kv_rollback_batch(cache, reservations, 2),
               VX_PAGED_KV_OK, "valid batch rolls back together");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[0], 2, "first prior length restored");
    REQUIRE_EQ(vx_paged_kv_lengths(cache)[1], 2, "second prior length restored");
    vx_paged_kv_destroy(cache);
    printf("ok batch reservations settle atomically\n");
    return 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "tests/paged_kv_vectors.json";
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
    if (strcmp(json_string(root, "version", ""), "volvox-paged-kv/v1")) {
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

    test_identity_map_reduces_to_contiguous_arithmetic();
    test_reservation_settles_once();
    test_checked_sizes_and_invalid_append();
    test_reservation_exclusivity_and_identity();
    test_batch_settlement_is_atomic();

    if (g_failures) {
        fprintf(stderr, "FAIL %d paged KV check(s)\n", g_failures);
        return 1;
    }
    printf("PASS paged KV vectors\n");
    return 0;
}
