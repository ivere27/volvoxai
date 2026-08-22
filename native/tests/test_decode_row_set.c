/*
 * Native half of the row-set addressing contract.  Reads the same
 * `tests/decode_row_set_vectors.json` the TypeScript test reads, so the two
 * implementations of "which rows does this step write, which slots does its
 * prefix come from, and which keys does each lane keep" cannot drift.
 *
 * This is the corpus the design document said was missing.  Shape inference has
 * one and the decode/row proofs had none, so a page table or a per-lane length
 * that meant something different in the two runtimes produced a decoder that
 * worked in the browser and not on the robot.  B>1 decode grows exactly that
 * surface, so the corpus arrives with it.
 */

#include "decode_row_set.h"

#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 512
#define MAX_MASK 4096
#define MAX_PAGES 256

static int g_failures;

static void fail_at(const char* file, int line, const char* case_name,
                    const char* message) {
    fprintf(stderr, "FAIL %s:%d: [%s] %s\n", file, line, case_name, message);
    g_failures++;
}

#define REQUIRE(condition, message) \
    do { \
        if (!(condition)) { \
            fail_at(__FILE__, __LINE__, case_name, (message)); \
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
            fail_at(__FILE__, __LINE__, case_name, _buffer); \
            return -1; \
        } \
    } while (0)

static int json_int(const cJSON* object, const char* field, int fallback) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, field);
    return cJSON_IsNumber(value) ? (int)value->valuedouble : fallback;
}

static int json_bool(const cJSON* object, const char* field, int fallback) {
    const cJSON* value = cJSON_GetObjectItemCaseSensitive(object, field);
    if (cJSON_IsBool(value)) return cJSON_IsTrue(value) ? 1 : 0;
    return fallback;
}

/* Copy an integer array field.  Returns the count, or -1 when absent. */
static int json_int_array(const cJSON* object, const char* field, int* out,
                          int capacity) {
    const cJSON* array = cJSON_GetObjectItemCaseSensitive(object, field);
    if (!cJSON_IsArray(array)) return -1;
    int count = 0;
    const cJSON* entry = NULL;
    cJSON_ArrayForEach(entry, array) {
        if (count >= capacity || !cJSON_IsNumber(entry)) return -1;
        out[count++] = (int)entry->valuedouble;
    }
    return count;
}

static VxDecodeKeepMaskLayout mask_layout(const char* name) {
    if (strcmp(name, "K") == 0) return VX_DECODE_KEEP_MASK_K;
    if (strcmp(name, "BK") == 0) return VX_DECODE_KEEP_MASK_BK;
    if (strcmp(name, "QK") == 0) return VX_DECODE_KEEP_MASK_QK;
    if (strcmp(name, "BQK") == 0) return VX_DECODE_KEEP_MASK_BQK;
    return VX_DECODE_KEEP_MASK_NONE;
}

static const char* tier_name(VxDecodeRowSpanTier tier) {
    return tier == VX_DECODE_ROW_SPAN_IDENTITY ? "identity" : "gather";
}

static int run_case(const cJSON* vector) {
    const cJSON* name = cJSON_GetObjectItemCaseSensitive(vector, "name");
    const char* case_name = cJSON_IsString(name) ? name->valuestring : "<unnamed>";
    const cJSON* expect = cJSON_GetObjectItemCaseSensitive(vector, "expect");
    REQUIRE(cJSON_IsObject(expect), "case has no expect object");

    int positions[VX_DECODE_ROW_SET_MAX_LANES];
    const int lanes = json_int_array(vector, "positions", positions,
                                     VX_DECODE_ROW_SET_MAX_LANES);
    REQUIRE(lanes > 0, "case has no positions array");
    REQUIRE_EQ(lanes, json_int(vector, "lanes", -1), "declared lane count");

    const int paged = json_bool(vector, "paged", 0);
    static int tables[VX_DECODE_ROW_SET_MAX_LANES][MAX_PAGES];
    VxDecodeLanePages pages[VX_DECODE_ROW_SET_MAX_LANES];
    memset(pages, 0, sizeof(pages));
    if (paged) {
        const cJSON* table_array =
            cJSON_GetObjectItemCaseSensitive(vector, "page_tables");
        REQUIRE(cJSON_IsArray(table_array), "paged case has no page_tables");
        const int page_tokens = json_int(vector, "page_tokens", 0);
        const int pages_per_lane = json_int(vector, "pages_per_lane", 0);
        REQUIRE(page_tokens > 0 && pages_per_lane > 0, "paged case has no page geometry");
        int lane = 0;
        const cJSON* entry = NULL;
        cJSON_ArrayForEach(entry, table_array) {
            REQUIRE(lane < lanes && cJSON_IsArray(entry), "page_tables shape");
            if (positions[lane] == VX_DECODE_ROW_PARKED) {
                /* A parked lane addresses nothing, so it carries no table. */
                lane++;
                continue;
            }
            int count = 0;
            const cJSON* page = NULL;
            cJSON_ArrayForEach(page, entry) {
                REQUIRE(count < MAX_PAGES && cJSON_IsNumber(page), "page_table entry");
                tables[lane][count++] = (int)page->valuedouble;
            }
            REQUIRE_EQ(count, pages_per_lane, "page_table width");
            pages[lane].page_table = tables[lane];
            pages[lane].page_tokens = page_tokens;
            pages[lane].pages_per_lane = pages_per_lane;
            lane++;
        }
        REQUIRE_EQ(lane, lanes, "page_tables lane count");
    }

    VxDecodeRowSet set;
    int expected[MAX_ROWS];
    int count;
    REQUIRE_EQ(vx_decode_row_set_init(&set, lanes, positions, paged ? pages : NULL),
               VX_DECODE_ROW_SET_OK, "row set init");
    REQUIRE_EQ(set.lanes, lanes, "lanes");
    REQUIRE_EQ(set.unpaged, paged ? 0 : 1, "unpaged");
    if (cJSON_HasObjectItem(expect, "live")) {
        REQUIRE_EQ(set.live, json_int(expect, "live", -1), "live");
        count = json_int_array(expect, "parked", expected, MAX_ROWS);
        REQUIRE_EQ(count, lanes, "parked count");
        for (int lane = 0; lane < lanes; lane++) {
            REQUIRE_EQ(set.parked[lane], expected[lane], "parked");
        }
    }
    REQUIRE_EQ(set.key_capacity, json_int(expect, "key_capacity", -1), "key_capacity");
    count = json_int_array(expect, "kv_lengths", expected, MAX_ROWS);
    REQUIRE_EQ(count, lanes, "kv_lengths count");
    for (int lane = 0; lane < lanes; lane++) {
        REQUIRE_EQ(set.kv_lengths[lane], expected[lane], "kv_length");
    }

    const int lane_stride = json_int(vector, "lane_stride", -1);
    REQUIRE(lane_stride >= 0, "case has no lane_stride");

    int rows[VX_DECODE_ROW_SET_MAX_LANES];
    REQUIRE_EQ(vx_decode_row_set_write_rows(&set, lane_stride, paged, rows),
               VX_DECODE_ROW_SET_OK, "write rows");
    count = json_int_array(expect, "write_rows", expected, MAX_ROWS);
    REQUIRE_EQ(count, lanes, "write_rows count");
    for (int lane = 0; lane < lanes; lane++) {
        REQUIRE_EQ(rows[lane], expected[lane], "write_row");
    }

    const cJSON* tier = cJSON_GetObjectItemCaseSensitive(expect, "tier");
    REQUIRE(cJSON_IsString(tier), "expect has no tier");
    REQUIRE(strcmp(tier_name(vx_decode_row_span_tier(rows, lanes)),
                   tier->valuestring) == 0, "row span tier");

    static int prefix[MAX_ROWS];
    REQUIRE(lanes * set.key_capacity <= MAX_ROWS, "prefix rows fit");
    REQUIRE_EQ(vx_decode_row_set_prefix_rows(&set, lane_stride, paged, prefix),
               VX_DECODE_ROW_SET_OK, "prefix rows");
    count = json_int_array(expect, "prefix_rows", expected, MAX_ROWS);
    REQUIRE_EQ(count, lanes * set.key_capacity, "prefix_rows count");
    for (int index = 0; index < count; index++) {
        REQUIRE_EQ(prefix[index], expected[index], "prefix_row");
    }

    /* A cross-attention memory publishes no per-lane length, so its keep mask
     * spans the whole memory and clamping is off. */
    const int clamp = json_bool(vector, "clamp_to_length", 1);
    const int keys = clamp ? set.key_capacity : json_int(vector, "memory_keys", -1);
    REQUIRE(keys > 0, "keep mask key extent");

    static int mask_values[MAX_MASK];
    VxDecodeKeepMask mask;
    memset(&mask, 0, sizeof(mask));
    const cJSON* mask_object = cJSON_GetObjectItemCaseSensitive(vector, "mask");
    if (cJSON_IsObject(mask_object)) {
        const cJSON* layout = cJSON_GetObjectItemCaseSensitive(mask_object, "layout");
        REQUIRE(cJSON_IsString(layout), "mask has no layout");
        mask.layout = mask_layout(layout->valuestring);
        REQUIRE(mask.layout != VX_DECODE_KEEP_MASK_NONE, "mask layout is known");
        mask.keys = json_int(mask_object, "keys", -1);
        mask.queries = json_int(mask_object, "queries", -1);
        REQUIRE(mask.keys > 0 && mask.queries > 0, "mask geometry");
        const int values = json_int_array(mask_object, "values", mask_values, MAX_MASK);
        REQUIRE(values > 0, "mask values");
        mask.values = mask_values;
    }

    static int keep[MAX_MASK];
    REQUIRE(lanes * keys <= MAX_MASK, "keep mask fits");
    REQUIRE_EQ(vx_decode_row_set_keep_mask(&set, keys, clamp, &mask, keep),
               VX_DECODE_ROW_SET_OK, "keep mask");
    count = json_int_array(expect, "keep_mask", expected, MAX_ROWS);
    REQUIRE_EQ(count, lanes * keys, "keep_mask count");
    for (int index = 0; index < count; index++) {
        REQUIRE_EQ(keep[index], expected[index], "keep_mask entry");
    }
    return 0;
}

/* A step that mixes paged and unpaged lanes is refused rather than resolved:
 * half-applied paging pairs a paged read with a linear write. */
static int run_mixed_paging_refusal(void) {
    const char* case_name = "mixed paged and unpaged lanes";
    int table[2] = {0, 1};
    int positions[2] = {1, 1};
    VxDecodeLanePages pages[2];
    memset(pages, 0, sizeof(pages));
    pages[0].page_table = table;
    pages[0].page_tokens = 1;
    pages[0].pages_per_lane = 2;
    VxDecodeRowSet set;
    REQUIRE_EQ(vx_decode_row_set_init(&set, 2, positions, pages),
               VX_DECODE_ROW_SET_INVALID_ARGUMENT, "mixed paging refusal");
    return 0;
}

/* An unmapped logical page is an error, never a slot number: a decode that
 * silently read page -1's arithmetic would produce a plausible wrong answer. */
static int run_unmapped_refusal(void) {
    const char* case_name = "unmapped logical page";
    int table[2] = {0, VX_PAGED_KV_UNMAPPED};
    int positions[1] = {1};
    VxDecodeLanePages pages[1];
    memset(pages, 0, sizeof(pages));
    pages[0].page_table = table;
    pages[0].page_tokens = 1;
    pages[0].pages_per_lane = 2;
    VxDecodeRowSet set;
    REQUIRE_EQ(vx_decode_row_set_init(&set, 1, positions, pages),
               VX_DECODE_ROW_SET_OK, "init");
    int rows[1];
    REQUIRE_EQ(vx_decode_row_set_write_rows(&set, 8, 1, rows),
               VX_DECODE_ROW_SET_UNMAPPED, "unmapped write row");
    return 0;
}

/* A step in which nothing advances is not a step. */
/*
 * Staging: the runs a scattered step collapses into, and the bytes they move.
 *
 * Not in the shared corpus, because the corpus fixes *addressing* and this is
 * how a backend spends it. Both halves are asserted together on purpose: runs
 * that merge correctly but copy the wrong bytes, and copies that are right one
 * row at a time while the run count claims they were one memcpy, fail the same
 * way from a caller's point of view and differently from a reviewer's.
 */
static int run_copy_runs(void) {
    const char* case_name = "copy runs collapse only what is adjacent on both sides";
    VxDecodeRowRun runs[8];
    /* Two lanes of a shared page land adjacent; the third is elsewhere. */
    const int adjacent[3] = {4, 5, 9};
    const int padded[5] = {2, -1, -1, 7, 8};
    const int scattered[3] = {9, 4, 5};
    int count;

    count = vx_decode_row_copy_runs(adjacent, 3, runs, 8);
    REQUIRE_EQ(count, 2, "adjacent rows merge");
    REQUIRE_EQ(runs[0].source, 4, "first run source");
    REQUIRE_EQ(runs[0].destination, 0, "first run destination");
    REQUIRE_EQ(runs[0].rows, 2, "first run length");
    REQUIRE_EQ(runs[1].source, 9, "second run source");
    REQUIRE_EQ(runs[1].destination, 2, "second run destination");
    REQUIRE_EQ(runs[1].rows, 1, "second run length");

    /* Padding breaks a run even where the sources either side are adjacent:
     * the destinations are not. */
    count = vx_decode_row_copy_runs(padded, 5, runs, 8);
    REQUIRE_EQ(count, 2, "padding splits the copy");
    REQUIRE_EQ(runs[1].source, 7, "run after padding");
    REQUIRE_EQ(runs[1].destination, 3, "destination after padding");
    REQUIRE_EQ(runs[1].rows, 2, "length after padding");

    count = vx_decode_row_padding_runs(padded, 5, runs, 8);
    REQUIRE_EQ(count, 1, "the gap is one run");
    REQUIRE_EQ(runs[0].destination, 1, "gap destination");
    REQUIRE_EQ(runs[0].rows, 2, "gap length");
    REQUIRE_EQ(runs[0].source, -1, "a gap has no source");

    /* Ascending sources out of destination order stay separate runs. */
    count = vx_decode_row_copy_runs(scattered, 3, runs, 8);
    REQUIRE_EQ(count, 2, "out-of-order sources do not merge across");

    REQUIRE_EQ(vx_decode_row_copy_runs(adjacent, 3, runs, 1),
               VX_DECODE_ROW_SET_INVALID_ARGUMENT, "capacity is enforced");
    return 0;
}

static int run_gather_scatter(void) {
    const char* case_name = "gather and scatter move exactly their own rows";
    /* Four rows of two bytes; row r is {10r, 10r+1}. */
    unsigned char storage[8] = {0, 1, 10, 11, 20, 21, 30, 31};
    const unsigned char original[8] = {0, 1, 10, 11, 20, 21, 30, 31};
    unsigned char staged[8];
    const int rows[3] = {3, -1, 1};
    const int parked[3] = {0, 0, 1};

    memset(staged, 0xAA, sizeof(staged));
    REQUIRE_EQ(vx_decode_row_gather(staged, storage, sizeof(storage), 2, rows, 3),
               VX_DECODE_ROW_SET_OK, "gather succeeds");
    REQUIRE_EQ(staged[0], 30, "staged row 0 low");
    REQUIRE_EQ(staged[1], 31, "staged row 0 high");
    /* Padding is zeroed, not left holding the pool's previous tenant. */
    REQUIRE_EQ(staged[2], 0, "padding zeroed low");
    REQUIRE_EQ(staged[3], 0, "padding zeroed high");
    REQUIRE_EQ(staged[4], 10, "staged row 2 low");
    REQUIRE_EQ(staged[5], 11, "staged row 2 high");

    /* Write the staged rows back after a kernel would have changed them. */
    staged[0] = 99; staged[1] = 98;
    staged[2] = 97; staged[3] = 96;
    staged[4] = 95; staged[5] = 94;
    REQUIRE_EQ(vx_decode_row_scatter(storage, sizeof(storage), staged, 2,
                                     rows, 3, parked),
               VX_DECODE_ROW_SET_OK, "scatter succeeds");
    REQUIRE_EQ(storage[6], 99, "live lane written");
    REQUIRE_EQ(storage[7], 98, "live lane written high");
    /* Lane 2 is parked and lane 1 is padding: neither reaches storage, so row 1
     * still holds what it held before the step. */
    REQUIRE_EQ(storage[2], original[2], "parked lane not written");
    REQUIRE_EQ(storage[3], original[3], "parked lane not written high");
    REQUIRE_EQ(storage[0], original[0], "untouched row 0");
    REQUIRE_EQ(storage[4], original[4], "untouched row 2");

    {
        /* A row that would read past the end is refused before any byte moves,
         * so a rejected span cannot leave a half-filled destination. */
        const int out_of_range[2] = {0, 4};
        memset(staged, 0xAA, sizeof(staged));
        REQUIRE_EQ(vx_decode_row_gather(staged, storage, sizeof(storage), 2,
                                        out_of_range, 2),
                   VX_DECODE_ROW_SET_INVALID_ARGUMENT, "out of range refused");
        REQUIRE_EQ(staged[0], 0xAA, "refusal wrote nothing");
    }
    return 0;
}

/*
 * The staged round trip reproduces the rows a one-lane decode would have
 * written, which is the property the whole tier exists to preserve.
 */
static int run_row_set_staging_round_trip(void) {
    const char* case_name = "a staggered batch stages to its own write rows";
    const int positions[3] = {2, VX_DECODE_ROW_PARKED, 5};
    VxDecodeRowSet set;
    int rows[VX_DECODE_ROW_SET_MAX_LANES];
    VxDecodeRowRun runs[VX_DECODE_ROW_SET_MAX_LANES];
    unsigned char storage[3 * 8 * 2];
    unsigned char staged[3 * 2];
    int count;

    REQUIRE_EQ(vx_decode_row_set_init(&set, 3, positions, NULL),
               VX_DECODE_ROW_SET_OK, "row set builds");
    REQUIRE_EQ(vx_decode_row_set_write_rows(&set, 8, 0, rows),
               VX_DECODE_ROW_SET_OK, "write rows resolve");
    /* lane 0 -> 2, parked lane 1 -> 8 (row zero of its lane), lane 2 -> 21. */
    REQUIRE_EQ(rows[0], 2, "lane 0 row");
    REQUIRE_EQ(rows[1], 8, "parked lane still occupies a row");
    REQUIRE_EQ(rows[2], 21, "lane 2 row");
    REQUIRE_EQ(vx_decode_row_span_tier(rows, 3), VX_DECODE_ROW_SPAN_GATHER,
               "a staggered batch is the gather tier");
    count = vx_decode_row_copy_runs(rows, 3, runs, VX_DECODE_ROW_SET_MAX_LANES);
    REQUIRE_EQ(count, 3, "three lanes, three runs");

    for (size_t index = 0; index < sizeof(storage); index++) {
        storage[index] = (unsigned char)(index & 0xFF);
    }
    REQUIRE_EQ(vx_decode_row_gather(staged, storage, sizeof(storage), 2, rows, 3),
               VX_DECODE_ROW_SET_OK, "gather succeeds");
    REQUIRE_EQ(staged[0], storage[4], "lane 0 read its own row");
    REQUIRE_EQ(staged[4], storage[42], "lane 2 read its own row");

    for (size_t index = 0; index < sizeof(staged); index++) staged[index] = 0xEE;
    REQUIRE_EQ(vx_decode_row_scatter(storage, sizeof(storage), staged, 2,
                                     rows, 3, set.parked),
               VX_DECODE_ROW_SET_OK, "scatter succeeds");
    REQUIRE_EQ(storage[4], 0xEE, "lane 0 written back");
    REQUIRE_EQ(storage[42], 0xEE, "lane 2 written back");
    REQUIRE_EQ(storage[16], 16, "the parked lane's row is untouched");
    return 0;
}

static int run_all_parked_refusal(void) {
    const char* case_name = "every lane parked";
    int positions[2] = {VX_DECODE_ROW_PARKED, VX_DECODE_ROW_PARKED};
    VxDecodeRowSet set;
    REQUIRE_EQ(vx_decode_row_set_init(&set, 2, positions, NULL),
               VX_DECODE_ROW_SET_INVALID_ARGUMENT, "all-parked refusal");
    return 0;
}

/* A parked lane addresses nothing, so it cannot break the paged uniformity
 * rule -- but a genuinely mixed step still must. */
static int run_parked_paging_uniformity(void) {
    const char* case_name = "parked lane beside mixed paging";
    int table[2] = {0, 1};
    int positions[3] = {1, 1, VX_DECODE_ROW_PARKED};
    VxDecodeLanePages pages[3];
    VxDecodeRowSet set;
    memset(pages, 0, sizeof(pages));
    pages[0].page_table = table;
    pages[0].page_tokens = 1;
    pages[0].pages_per_lane = 2;
    REQUIRE_EQ(vx_decode_row_set_init(&set, 3, positions, pages),
               VX_DECODE_ROW_SET_INVALID_ARGUMENT, "mixed paging beside a parked lane");
    pages[1] = pages[0];
    REQUIRE_EQ(vx_decode_row_set_init(&set, 3, positions, pages),
               VX_DECODE_ROW_SET_OK, "uniformly paged live lanes with one parked");
    REQUIRE_EQ(set.live, 2, "live");
    REQUIRE_EQ(set.unpaged, 0, "unpaged");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <decode_row_set_vectors.json>\n", argv[0]);
        return 2;
    }
    FILE* file = fopen(argv[1], "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* text = (char*)malloc((size_t)size + 1);
    if (text == NULL || fread(text, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        fclose(file);
        free(text);
        return 2;
    }
    text[size] = '\0';
    fclose(file);

    cJSON* document = cJSON_Parse(text);
    free(text);
    if (document == NULL) {
        fprintf(stderr, "cannot parse %s\n", argv[1]);
        return 2;
    }
    const cJSON* version = cJSON_GetObjectItemCaseSensitive(document, "version");
    if (!cJSON_IsString(version) ||
        strcmp(version->valuestring, "volvox-decode-row-set/v1") != 0) {
        fprintf(stderr, "unexpected corpus version\n");
        cJSON_Delete(document);
        return 2;
    }
    const cJSON* cases = cJSON_GetObjectItemCaseSensitive(document, "cases");
    if (!cJSON_IsArray(cases)) {
        fprintf(stderr, "corpus has no cases array\n");
        cJSON_Delete(document);
        return 2;
    }
    int executed = 0;
    const cJSON* vector = NULL;
    cJSON_ArrayForEach(vector, cases) {
        run_case(vector);
        executed++;
    }
    run_mixed_paging_refusal();
    run_unmapped_refusal();
    run_all_parked_refusal();
    run_parked_paging_uniformity();
    run_copy_runs();
    run_gather_scatter();
    run_row_set_staging_round_trip();
    cJSON_Delete(document);
    if (executed == 0) {
        fprintf(stderr, "corpus executed no cases\n");
        return 2;
    }
    if (g_failures != 0) {
        fprintf(stderr, "%d decode row set check(s) failed\n", g_failures);
        return 1;
    }
    printf("decode row set: %d case(s) passed\n", executed);
    return 0;
}
