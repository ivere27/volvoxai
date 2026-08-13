/*
 * Paged K/V and a declared batch, in the same step.
 *
 * These two were built separately and never met. Paged addressing was F32-only
 * -- the Linear output row and CrossSDPA were the only callers of
 * `vx_paged_row_locked` / `vx_paged_gather_f32_locked` -- and B>1 batching was
 * W8A8-only, so no step ever carried both. The pieces the row set kept for the
 * join (`vx_decode_row_set_write_rows`'s `paged` argument, and the lane page
 * tables it consults) had no caller at all.
 *
 * What has to be true for them to meet, and where each part fails on its own:
 *
 *   shape  — a paged K/V operand is a pool, not a batch. Every other operand is
 *            `[lanes, ...]`; the pool is shared, because lane 1's token sitting
 *            between two of lane 0's is the entire point of paging. A rank
 *            check that demanded a batch axis rejected such a step before any
 *            addressing ran.
 *   write  — each lane's projection writes into *its own* mapped slot. Linear's
 *            scalar path computes one contiguous run from `seq_range`, which
 *            for `[lanes,S,W]` is lane zero's row for every lane, and then maps
 *            that one row through lane zero's page table.
 *   read   — each lane's attention gathers *its own* prefix in logical order.
 *            The gather was written against the single bound lane.
 *
 * The oracle is the same decode run one lane at a time, with no batch and no
 * paging: LANES separate `[1,S,D]` engines. That makes the comparison a claim
 * about both features at once -- a batch that addressed lanes correctly but
 * gathered pages wrongly, or the reverse, disagrees with it.
 *
 * The mapping is scattered on purpose. Two lanes appending alternately give
 * lane 0 physical pages 0,2,4,... and lane 1 pages 1,3,5,... which is what a
 * scheduler running two requests concurrently produces, and what makes "logical
 * row" and "physical row" different numbers.
 */

#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "paged_binding.h"
#include "paged_kv.h"
#include "runtime_state.h"
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

#define LANES 2
#define SEQUENCE 4
#define WIDTH 4
#define HEADS 2
/* One page per token, so the scatter is visible at every position rather than
 * only at page boundaries. Two lanes over SEQUENCE tokens each. */
#define POOL_ROWS (LANES * SEQUENCE)

static const char* GRAPH_PATH = "/tmp/volvox-paged-batch-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-paged-batch-weights.safetensors";

static const char* PAGED_TENSORS[2] = { "self.k", "self.v" };

/*
 * Flat on purpose: every node between a projection and attention would be
 * another operator that must learn page-table addressing, and the row path
 * refuses those rather than addressing them with linear arithmetic.
 *
 * The K/V pool is spelled `[lanes, S, W]` so the projection writing into it
 * stays shape-preserving, but nothing reads that shape as a batch: the physical
 * row a page table names indexes the pool flat, and `lanes * S` rows laid out
 * `[lanes, S, W]` are the same bytes as `[1, lanes * S, W]`. Attention takes the
 * pool's extent from its row count for exactly that reason.
 */
static int write_graph_document(int lanes) {
    char graph[2048];
    FILE* handle;
    int written = snprintf(
        graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wq\"},"
        "\"outputs\":{\"out\":\"q\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}},"
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wk\"},"
        "\"outputs\":{\"out\":\"self.k\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}},"
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wv\"},"
        "\"outputs\":{\"out\":\"self.v\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}},"
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"self.k\","
        "\"v\":\"self.v\"},"
        "\"outputs\":{\"out\":\"attended\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":0.5}}],"
        "\"outputs\":[\"attended\",\"self.k\",\"self.v\"]}",
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH, HEADS);
    if (written <= 0 || (size_t)written >= sizeof(graph)) return -1;
    handle = fopen(GRAPH_PATH, "wb");
    if (!handle) return -1;
    if (fwrite(graph, 1, (size_t)written, handle) != (size_t)written) {
        fclose(handle);
        return -1;
    }
    return fclose(handle) == 0 ? 0 : -1;
}

static float weight_value(int row, int column, int seed) {
    return (float)(((row * 5 + column * 3 + seed) % 11) - 5) * 0.0625f;
}

static int write_weights(void) {
    SafetensorsFile file;
    const int shape[2] = { WIDTH, WIDTH };
    float wq[WIDTH * WIDTH];
    float wk[WIDTH * WIDTH];
    float wv[WIDTH * WIDTH];

    for (int row = 0; row < WIDTH; row++)
        for (int column = 0; column < WIDTH; column++) {
            wq[row * WIDTH + column] = weight_value(row, column, 1);
            wk[row * WIDTH + column] = weight_value(row, column, 4);
            wv[row * WIDTH + column] = weight_value(row, column, 7);
        }
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "wq", SAFETENSORS_DTYPE_F32, shape, 2,
                                 wq, sizeof(wq)) == 0);
    CHECK(safetensors_add_tensor(&file, "wk", SAFETENSORS_DTYPE_F32, shape, 2,
                                 wk, sizeof(wk)) == 0);
    CHECK(safetensors_add_tensor(&file, "wv", SAFETENSORS_DTYPE_F32, shape, 2,
                                 wv, sizeof(wv)) == 0);
    CHECK(safetensors_save(WEIGHTS_PATH, &file) == 0);
    safetensors_free(&file);
    return 0;
}

/* Lane-distinct and position-distinct, so "lane 1 read lane 0's row" is a
 * mismatch rather than a coincidence. */
static float element(int lane, int token, int channel) {
    return (float)(((lane * 11 + token * 7 + channel * 3) % 13) - 6) * 0.125f;
}

static void fill_x(float* x, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            for (int channel = 0; channel < WIDTH; channel++)
                x[(lane * SEQUENCE + token) * WIDTH + channel] =
                    element(lane, token, channel);
}

/*
 * No keep mask, deliberately.
 *
 * A mask over a shared pool would have to name per-lane extents, which is the
 * page table written a second time in another form. The node is causal instead:
 * the unpaged oracle bounds its row at `position + 1`, and the paged batch
 * bounds each lane at its published length -- which is the same number, arrived
 * at from the cache rather than from a shape.
 */
/* Whether every node that should have taken a paged batch route did. */
static int paged_batch_routes_ok(void) {
    const VxEngineState* state = vx_engine_state_current();
    int attention = 0;
    int projection = 0;
    if (!state) return 0;
    for (int index = 0; index < state->node_count; index++) {
        const char* route = state->runtime_route_backend[index];
        if (!route) continue;
        if (!strcmp(route, "cpu-cross-sdpa-paged-batch-row")) attention++;
        if (!strcmp(route, "cpu-linear-paged-batch-row")) projection++;
    }
    /* Two of the three projections write the pool; the query projection writes
     * an ordinary batched activation and takes the unpaged batch route. */
    if (attention == 1 && projection == 2) return 1;
    fprintf(stderr,
            "FAIL the step routed %d attention and %d projection nodes through "
            "the paged batch path, wanted 1 and 2\n", attention, projection);
    return 0;
}

/*
 * One batched paged decode: seed, then a row step per position.
 *
 * Every K/V row is written by a row step rather than inherited from the seed,
 * and that is forced rather than stylistic. The seed is a full forward, so it
 * writes K/V *linearly* -- lane l's token t at row `l * S + t`. A scattered
 * mapping says that row belongs to some other logical position, so a decode
 * that seeded its cache and then trusted it would be reading rows the table
 * never claimed. Stepping every position through the paged path puts each
 * lane's value where its own table says, which is what a prefill would also
 * have to do.
 *
 * Appends interleave across lanes, so lane 0 holds physical pages 0,2,4,... and
 * lane 1 holds 1,3,5,... -- what a scheduler running two requests concurrently
 * produces, and what makes "logical row" and "physical row" different numbers.
 * A lane that has reached its target position parks: it holds its request but
 * has no row to advance, so it appends nothing and writes nothing.
 */
static int run_batch(float rows[LANES][WIDTH], const int* targets) {
    VxPagedKVOptions options;
    VxPagedKVCache* cache = NULL;
    static float x[LANES * SEQUENCE * WIDTH];
    static float attended[LANES * SEQUENCE * WIDTH];
    int furthest = 0;
    int status = -1;

    memset(&options, 0, sizeof(options));
    options.lanes = LANES;
    options.page_tokens = 1;
    options.lane_token_capacity = SEQUENCE;
    options.max_pages = POOL_ROWS;
    cache = vx_paged_kv_create(&options);
    CHECK(cache != NULL);
    for (int lane = 0; lane < LANES; lane++)
        if (targets[lane] > furthest) furthest = targets[lane];

    if (write_graph_document(LANES) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto destroy;
    fill_x(x, LANES);
    if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto shutdown;

    for (int position = 0; position <= furthest; position++) {
        int step[LANES];
        for (int lane = 0; lane < LANES; lane++) {
            if (position > targets[lane]) {
                step[lane] = VOLVOXAI_DECODE_LANE_PARKED;
                continue;
            }
            step[lane] = position;
            /* One token, now, so this lane's published length covers the row
             * it is about to write and no more. */
            CHECK(vx_paged_kv_append(cache, lane, 1) == VX_PAGED_KV_OK);
        }
        /* Lane zero is the bound lane, and for a batch that choice is inert:
         * every per-lane query takes its lane as an argument. Binding is what
         * publishes the cache to the context at all. */
        if (vx_paged_bind_locked(cache, 0, PAGED_TENSORS, 2) != 0) goto shutdown;
        if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) != 0)
            goto shutdown;
        if (volvoxai_engine_forward_incremental_rows(step, LANES) != 0) {
            fprintf(stderr, "FAIL the %d-lane paged row step at position %d was "
                            "declined\n", LANES, position);
            goto shutdown;
        }
        /* Values alone would not say the paged batch path ran: a step that
         * quietly fell back would still be compared against the oracle. */
        if (!paged_batch_routes_ok()) goto shutdown;
    }
    if (volvoxai_engine_copy_tensor_f32("attended", attended,
                                        LANES * SEQUENCE * WIDTH) != 0) goto shutdown;
    for (int lane = 0; lane < LANES; lane++)
        memcpy(rows[lane],
               attended + (size_t)(lane * SEQUENCE + targets[lane]) * WIDTH,
               WIDTH * sizeof(float));
    status = 0;
shutdown:
    volvoxai_engine_shutdown();
destroy:
    vx_paged_kv_destroy(cache);
    return status;
}

/*
 * The oracle: lane `lane` alone, contiguous, unpaged.
 *
 * No cache is bound, so K/V rows land at their logical positions -- which for
 * one lane is what any page table would have produced anyway. It walks the same
 * positions the batch walks, so the two differ in exactly the two things under
 * test: how many lanes advance, and whether the rows go through a page table.
 * A lane's result must not depend on either.
 */
static int run_single_lane(int lane, int target, float row[WIDTH]) {
    static float x[SEQUENCE * WIDTH];
    static float attended[SEQUENCE * WIDTH];
    int status = -1;

    if (write_graph_document(1) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) return -1;
    for (int token = 0; token < SEQUENCE; token++)
        for (int channel = 0; channel < WIDTH; channel++)
            x[token * WIDTH + channel] = element(lane, token, channel);
    if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto shutdown;
    for (int position = 0; position <= target; position++) {
        if (volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) != 0)
            goto shutdown;
        if (volvoxai_engine_forward_incremental_row(position) != 0) {
            fprintf(stderr, "FAIL lane %d's one-lane row step at position %d was "
                            "declined\n", lane, position);
            goto shutdown;
        }
    }
    if (volvoxai_engine_copy_tensor_f32("attended", attended,
                                        SEQUENCE * WIDTH) != 0) goto shutdown;
    memcpy(row, attended + (size_t)target * WIDTH, WIDTH * sizeof(float));
    status = 0;
shutdown:
    volvoxai_engine_shutdown();
    return status;
}

static int run_all(void) {
    /* Ragged: lanes at a common position leave `lane * S + position`
     * indistinguishable from several wrong formulas, and give the two lanes the
     * same key extent. */
    const int positions[LANES] = { 3, 1 };
    static float batch_rows[LANES][WIDTH];
    static float reference_rows[LANES][WIDTH];

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(write_weights() == 0);

    if (run_batch(batch_rows, positions) != 0) return 1;
    printf("ok a %d-lane step ran with a scattered page mapping on both the "
           "projection and the attention\n", LANES);
    CHECK(vx_paged_bound_locked() == 0);

    for (int lane = 0; lane < LANES; lane++)
        if (run_single_lane(lane, positions[lane], reference_rows[lane]) != 0)
            return 1;

    /* Two lanes agreeing would make the comparison hold for the wrong reason. */
    for (int lane = 1; lane < LANES; lane++) {
        if (memcmp(reference_rows[0], reference_rows[lane],
                   WIDTH * sizeof(float)) != 0) continue;
        fprintf(stderr, "FAIL lanes 0 and %d have identical reference rows; "
                        "the fixture cannot tell them apart\n", lane);
        return 1;
    }

    for (int lane = 0; lane < LANES; lane++) {
        if (memcmp(batch_rows[lane], reference_rows[lane],
                   WIDTH * sizeof(float)) == 0) continue;
        fprintf(stderr, "FAIL lane %d differs from its unpaged one-lane decode "
                        "at:", lane);
        for (int channel = 0; channel < WIDTH; channel++) {
            if (batch_rows[lane][channel] == reference_rows[lane][channel]) continue;
            fprintf(stderr, " [%d] batch=%.9g one-lane=%.9g", channel,
                    (double)batch_rows[lane][channel],
                    (double)reference_rows[lane][channel]);
        }
        fprintf(stderr, "\n");
        return 1;
    }
    printf("ok every lane's paged batched row is bit-identical to its own "
           "unpaged one-lane decode\n");
    printf("PASS paged batch decode\n");
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    int result;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    result = run_all();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    if (result == 0) {
        remove(GRAPH_PATH);
        remove(WEIGHTS_PATH);
    }
    return result;
}
