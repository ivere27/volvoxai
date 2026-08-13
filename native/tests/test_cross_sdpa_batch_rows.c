/*
 * CrossSDPA over several lanes in one decode step.
 *
 * CrossSDPA was the last attention operator without a batch path, and the
 * reason it was last is that it is the F32 one: every other operator converted
 * for B>1 decode is W8A8, so the fixtures that established those conversions
 * could not reach it. This is the F32 batch fixture that gap called for.
 *
 * The bar is the one the rest of the batch work uses: **each lane is
 * bit-identical to its own one-lane decode.** Both sides are F32 and both run
 * the same kernel, so any difference at all is a difference in which bytes were
 * read -- which is exactly the failure an unconverted row path has, since it
 * computes one contiguous span from `seq_range` and for `[lanes,S,D]` that span
 * is lane zero's row for every lane. Plausible numbers, wrong request.
 *
 * Two engines over two graphs: one `[LANES,S,D]` batch, and LANES separate
 * `[1,S,D]` decodes fed that lane's own operands. Comparing a batch against
 * itself would only prove it is deterministic.
 *
 * The route is checked as well as the values, and it is not a formality. The
 * whole-sequence fallback computes every row of every lane and would agree with
 * the reference on all of them -- so a batch that silently declined the row
 * path would pass a values-only version of this test while proving nothing
 * about the branch it exists to cover.
 *
 * Two node shapes, because they take different arms of that branch:
 *
 *   - a cross node, whose memory every lane reads whole (no per-lane length,
 *     which is why the row set publishes `clamp_to_length = 0` for it), and
 *   - a causal self node spelled as CrossSDPA, where `seq_q == seq_kv` and the
 *     lane's own position bounds the keys it may see. That one is where a batch
 *     that used lane zero's position for every lane would show up as a wrong
 *     key extent rather than as a wrong base address.
 */

#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return -1; \
    } \
} while (0)

#define LANES 3
#define SEQUENCE 6
#define MEMORY 4
#define WIDTH 8
#define HEADS 2

/* Ragged on purpose: lanes at a common position leave `lane * S + position`
 * indistinguishable from several wrong formulas, and the causal node's key
 * extent would be the same for every lane. */
static const int POSITIONS[LANES] = { 4, 2, 5 };

static const char* GRAPH_PATH = "/tmp/volvox-cross-sdpa-batch-graph.json";

typedef enum {
    MASK_LAYOUT_BK = 0,
    MASK_LAYOUT_BQK = 1,
    MASK_LAYOUT_QK = 2
} MaskLayout;

static const char* mask_layout_name(MaskLayout layout) {
    switch (layout) {
        case MASK_LAYOUT_BK: return "BK (batch-key)";
        case MASK_LAYOUT_BQK: return "BQK (batch-query-key)";
        case MASK_LAYOUT_QK: return "QK (query-key)";
        default: return "unknown";
    }
}

/*
 * One graph, three mask spellings.
 *
 * The memory node always carries a mask; the causal-self node carries one only
 * where the layout has a query axis, because a `[B,K]` mask over a self node is
 * the same statement causality already makes and would test nothing new. The
 * query-dependent layouts are the point of running this three times: they used
 * to be refused by the row branches, and what admits them is that the mask row
 * for the query's absolute position is resolved before the kernel sees it.
 *
 * A `[Sq,K]` mask has no batch axis, which is why the QK case declares its two
 * masks lane-independent -- and why it is worth having beside BQK rather than
 * folded into it: the two reach `attention_mask_query_view` through different
 * arms.
 */
static int write_graph_document(int lanes, MaskLayout layout) {
    char graph[2048];
    /* The mask operands, spelled for this layout: `[B,K]` and `[B,Sq,K]` carry a
     * lane axis, `[Sq,K]` does not. */
    const int mem_batched = layout != MASK_LAYOUT_QK;
    const int self_masked = layout != MASK_LAYOUT_BK;
    char mem_shape[48];
    char self_input[96] = "";
    char self_port[24] = "";
    FILE* handle;
    int written;

    if (layout == MASK_LAYOUT_BQK)
        snprintf(mem_shape, sizeof(mem_shape), "[%d,%d,%d]", lanes, SEQUENCE, MEMORY);
    else if (mem_batched)
        snprintf(mem_shape, sizeof(mem_shape), "[%d,%d]", lanes, MEMORY);
    else
        snprintf(mem_shape, sizeof(mem_shape), "[%d,%d]", SEQUENCE, MEMORY);
    if (self_masked) {
        char self_shape[48];
        if (layout == MASK_LAYOUT_BQK)
            snprintf(self_shape, sizeof(self_shape), "[%d,%d,%d]",
                     lanes, SEQUENCE, SEQUENCE);
        else
            snprintf(self_shape, sizeof(self_shape), "[%d,%d]", SEQUENCE, SEQUENCE);
        if (snprintf(self_input, sizeof(self_input),
                     ",\"self_keep\":{\"shape\":%s,\"dtype\":\"int32\"}",
                     self_shape) >= (int)sizeof(self_input)) return -1;
        snprintf(self_port, sizeof(self_port), ",\"mask\":\"self_keep\"");
    }

    written = snprintf(
        graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"q\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"},"
        "\"mem_k\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"},"
        "\"mem_v\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"},"
        "\"mem_keep\":{\"shape\":%s,\"dtype\":\"int32\"},"
        "\"self_k\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"},"
        "\"self_v\":{\"shape\":[%d,%d,%d],\"dtype\":\"float32\"}%s},"
        "\"nodes\":["
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"mem_k\","
        "\"v\":\"mem_v\",\"mask\":\"mem_keep\"},"
        "\"outputs\":{\"out\":\"cross\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"heads\":%d,\"causal\":false,\"scale\":0.5}},"
        "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"self_k\","
        "\"v\":\"self_v\"%s},"
        "\"outputs\":{\"out\":\"self\"},\"outputs_shape\":{\"out\":[%d,%d,%d]},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":0.5}}],"
        "\"outputs\":[\"cross\",\"self\"]}",
        lanes, SEQUENCE, WIDTH,
        lanes, MEMORY, WIDTH,
        lanes, MEMORY, WIDTH,
        mem_shape,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH, self_input,
        lanes, SEQUENCE, WIDTH, HEADS,
        self_port,
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

/*
 * Lane-distinct, position-distinct, and bounded.
 *
 * Every element is a function of (lane, row, column), so two lanes never share
 * a value by accident -- which is what makes "lane 1 read lane 0's row" visible
 * as a mismatch rather than as a coincidence. Kept inside about +-1 so the
 * softmax has a real distribution instead of collapsing onto one key.
 */
static float element(int lane, int row, int column, int seed) {
    int mixed = (lane * 31 + row * 17 + column * 7 + seed * 11) % 23;
    return (float)(mixed - 11) * 0.08f;
}

static void fill_q(float* values, int lanes, int revision) {
    for (int lane = 0; lane < lanes; lane++)
        for (int row = 0; row < SEQUENCE; row++)
            for (int column = 0; column < WIDTH; column++)
                values[(lane * SEQUENCE + row) * WIDTH + column] =
                    element(lane, row + revision, column, 1);
}

static void fill_memory(float* keys, float* values, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int row = 0; row < MEMORY; row++)
            for (int column = 0; column < WIDTH; column++) {
                size_t index = (size_t)(lane * MEMORY + row) * WIDTH + column;
                keys[index] = element(lane, row, column, 2);
                values[index] = element(lane, row, column, 3);
            }
}

static void fill_self(float* keys, float* values, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int row = 0; row < SEQUENCE; row++)
            for (int column = 0; column < WIDTH; column++) {
                size_t index = (size_t)(lane * SEQUENCE + row) * WIDTH + column;
                keys[index] = element(lane, row, column, 4);
                values[index] = element(lane, row, column, 5);
            }
}

/*
 * The keep sets, as functions of (lane, query, key).
 *
 * Written once and read by both runners, because the whole comparison is that a
 * lane in the batch and the same lane alone are handed the same mask. Two
 * generators that agreed by inspection rather than by construction would make
 * a disagreement look like a masking bug.
 *
 * Every one drops keys the lane would otherwise see, and drops a different set
 * per lane: a batch that read lane zero's mask row for every lane has to
 * disagree with the reference rather than match it.
 */
static int mem_keeps(int lane, int query, int key) {
    return ((lane * 7 + query * 5 + key * 3) % 4) != 0;
}

static int self_keeps(int lane, int query, int key) {
    return ((lane * 5 + query * 3 + key * 7) % 5) != 0;
}

/* `[B,K]`: no query axis, so the row branch may slice the query away without
 * touching it. Lane 1 drops a memory key so the mask is not uniform across
 * lanes. */
static void fill_keep_bk(int32_t* keep, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int key = 0; key < MEMORY; key++)
            keep[lane * MEMORY + key] = (lane == 1 && key == 2) ? 0 : 1;
}

/* `[B,Sq,K]`: `lane_base` is the lane a one-lane run is standing in for, which
 * is what lets the same generator fill a batch operand and a single-lane one. */
static void fill_keep_bqk(int32_t* mem_keep, int32_t* self_keep, int lanes,
                          int lane_base) {
    for (int lane = 0; lane < lanes; lane++)
        for (int query = 0; query < SEQUENCE; query++) {
            for (int key = 0; key < MEMORY; key++)
                mem_keep[(lane * SEQUENCE + query) * MEMORY + key] =
                    mem_keeps(lane_base + lane, query, key);
            for (int key = 0; key < SEQUENCE; key++)
                self_keep[(lane * SEQUENCE + query) * SEQUENCE + key] =
                    self_keeps(lane_base + lane, query, key);
        }
}

/* `[Sq,K]`: shared by every lane, so it takes no lane argument at all. */
static void fill_keep_qk(int32_t* mem_keep, int32_t* self_keep) {
    for (int query = 0; query < SEQUENCE; query++) {
        for (int key = 0; key < MEMORY; key++)
            mem_keep[query * MEMORY + key] = mem_keeps(0, query, key);
        for (int key = 0; key < SEQUENCE; key++)
            self_keep[query * SEQUENCE + key] = self_keeps(0, query, key);
    }
}

static int set_inputs(int lanes, int revision, MaskLayout layout) {
    static float q[LANES * SEQUENCE * WIDTH];
    static float mem_k[LANES * MEMORY * WIDTH];
    static float mem_v[LANES * MEMORY * WIDTH];
    static float self_k[LANES * SEQUENCE * WIDTH];
    static float self_v[LANES * SEQUENCE * WIDTH];
    static int32_t mem_keep_bk[LANES * MEMORY];
    static int32_t mem_keep_bqk[LANES * SEQUENCE * MEMORY];
    static int32_t self_keep_bqk[LANES * SEQUENCE * SEQUENCE];
    static int32_t mem_keep_qk[SEQUENCE * MEMORY];
    static int32_t self_keep_qk[SEQUENCE * SEQUENCE];

    const size_t q_bytes = (size_t)lanes * SEQUENCE * WIDTH * sizeof(float);
    const size_t memory_bytes = (size_t)lanes * MEMORY * WIDTH * sizeof(float);

    fill_q(q, lanes, revision);
    fill_memory(mem_k, mem_v, lanes);
    fill_self(self_k, self_v, lanes);

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_F32, q, q_bytes) == 0);
    CHECK(volvoxai_engine_set_input_raw("mem_k", VOLVOXAI_DTYPE_F32,
                                        mem_k, memory_bytes) == 0);
    CHECK(volvoxai_engine_set_input_raw("mem_v", VOLVOXAI_DTYPE_F32,
                                        mem_v, memory_bytes) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_k", VOLVOXAI_DTYPE_F32,
                                        self_k, q_bytes) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_v", VOLVOXAI_DTYPE_F32,
                                        self_v, q_bytes) == 0);

    if (layout == MASK_LAYOUT_BK) {
        fill_keep_bk(mem_keep_bk, lanes);
        CHECK(volvoxai_engine_set_input_raw(
            "mem_keep", VOLVOXAI_DTYPE_I32, mem_keep_bk,
            (size_t)lanes * MEMORY * sizeof(int32_t)) == 0);
    } else if (layout == MASK_LAYOUT_BQK) {
        fill_keep_bqk(mem_keep_bqk, self_keep_bqk, lanes, 0);
        CHECK(volvoxai_engine_set_input_raw(
            "mem_keep", VOLVOXAI_DTYPE_I32, mem_keep_bqk,
            (size_t)lanes * SEQUENCE * MEMORY * sizeof(int32_t)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
            "self_keep", VOLVOXAI_DTYPE_I32, self_keep_bqk,
            (size_t)lanes * SEQUENCE * SEQUENCE * sizeof(int32_t)) == 0);
    } else {
        fill_keep_qk(mem_keep_qk, self_keep_qk);
        CHECK(volvoxai_engine_set_input_raw(
            "mem_keep", VOLVOXAI_DTYPE_I32, mem_keep_qk,
            (size_t)SEQUENCE * MEMORY * sizeof(int32_t)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
            "self_keep", VOLVOXAI_DTYPE_I32, self_keep_qk,
            (size_t)SEQUENCE * SEQUENCE * sizeof(int32_t)) == 0);
    }
    return 0;
}

/*
 * A one-lane engine is fed lane `lane`'s operands, which the shared `element`
 * and keep-mask generators produce from the lane index. Passing `lane` where the
 * batch passes its own lane index is the whole of what makes the two runs
 * comparable.
 */
static int set_single_lane_inputs(int lane, int revision, MaskLayout layout) {
    static float q[SEQUENCE * WIDTH];
    static float mem_k[MEMORY * WIDTH];
    static float mem_v[MEMORY * WIDTH];
    static float self_k[SEQUENCE * WIDTH];
    static float self_v[SEQUENCE * WIDTH];
    static int32_t mem_keep_bk[MEMORY];
    static int32_t mem_keep_bqk[SEQUENCE * MEMORY];
    static int32_t self_keep_bqk[SEQUENCE * SEQUENCE];
    static int32_t mem_keep_qk[SEQUENCE * MEMORY];
    static int32_t self_keep_qk[SEQUENCE * SEQUENCE];

    for (int row = 0; row < SEQUENCE; row++)
        for (int column = 0; column < WIDTH; column++) {
            q[row * WIDTH + column] = element(lane, row + revision, column, 1);
            self_k[row * WIDTH + column] = element(lane, row, column, 4);
            self_v[row * WIDTH + column] = element(lane, row, column, 5);
        }
    for (int row = 0; row < MEMORY; row++)
        for (int column = 0; column < WIDTH; column++) {
            mem_k[row * WIDTH + column] = element(lane, row, column, 2);
            mem_v[row * WIDTH + column] = element(lane, row, column, 3);
        }

    CHECK(volvoxai_engine_set_input_raw("q", VOLVOXAI_DTYPE_F32, q, sizeof(q)) == 0);
    CHECK(volvoxai_engine_set_input_raw("mem_k", VOLVOXAI_DTYPE_F32,
                                        mem_k, sizeof(mem_k)) == 0);
    CHECK(volvoxai_engine_set_input_raw("mem_v", VOLVOXAI_DTYPE_F32,
                                        mem_v, sizeof(mem_v)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_k", VOLVOXAI_DTYPE_F32,
                                        self_k, sizeof(self_k)) == 0);
    CHECK(volvoxai_engine_set_input_raw("self_v", VOLVOXAI_DTYPE_F32,
                                        self_v, sizeof(self_v)) == 0);

    if (layout == MASK_LAYOUT_BK) {
        for (int key = 0; key < MEMORY; key++)
            mem_keep_bk[key] = (lane == 1 && key == 2) ? 0 : 1;
        CHECK(volvoxai_engine_set_input_raw("mem_keep", VOLVOXAI_DTYPE_I32,
                                            mem_keep_bk, sizeof(mem_keep_bk)) == 0);
    } else if (layout == MASK_LAYOUT_BQK) {
        fill_keep_bqk(mem_keep_bqk, self_keep_bqk, 1, lane);
        CHECK(volvoxai_engine_set_input_raw("mem_keep", VOLVOXAI_DTYPE_I32,
                                            mem_keep_bqk, sizeof(mem_keep_bqk)) == 0);
        CHECK(volvoxai_engine_set_input_raw("self_keep", VOLVOXAI_DTYPE_I32,
                                            self_keep_bqk, sizeof(self_keep_bqk)) == 0);
    } else {
        fill_keep_qk(mem_keep_qk, self_keep_qk);
        CHECK(volvoxai_engine_set_input_raw("mem_keep", VOLVOXAI_DTYPE_I32,
                                            mem_keep_qk, sizeof(mem_keep_qk)) == 0);
        CHECK(volvoxai_engine_set_input_raw("self_keep", VOLVOXAI_DTYPE_I32,
                                            self_keep_qk, sizeof(self_keep_qk)) == 0);
    }
    return 0;
}

/* Which route every CrossSDPA node took on the step just run. Returns the
 * number of nodes that took `want`, and -1 if any took something else. */
static int cross_sdpa_route_count(const char* want) {
    const VxEngineState* state = vx_engine_state_current();
    size_t length = strlen(want);
    int matched = 0;
    if (!state) return -1;
    for (int index = 0; index < state->node_count; index++) {
        const char* route = state->runtime_route_backend[index];
        if (strcmp(state->nodes[index].op, "CrossSDPA")) continue;
        if (state->nodes[index].disabled || state->nodes[index].skip) continue;
        if (route && !strncmp(route, want, length) && route[length] == '\0') {
            matched++;
            continue;
        }
        fprintf(stderr, "FAIL node %d (CrossSDPA) was routed to %s, not %s\n",
                index, route ? route : "(nothing)", want);
        return -1;
    }
    return matched;
}

static int run_batch(MaskLayout layout, float cross[LANES][WIDTH], float self[LANES][WIDTH]) {
    static float cross_all[LANES * SEQUENCE * WIDTH];
    static float self_all[LANES * SEQUENCE * WIDTH];
    int routed;

    CHECK(write_graph_document(LANES, layout) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, NULL) == 0);
    if (set_inputs(LANES, 0, layout) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto fail;
    if (set_inputs(LANES, 1, layout) != 0) goto fail;
    if (volvoxai_engine_forward_incremental_rows(POSITIONS, LANES) != 0) {
        fprintf(stderr, "FAIL the %d-lane CrossSDPA row step was declined\n", LANES);
        goto fail;
    }
    /* Values alone cannot say the row branch ran: the whole-sequence fallback
     * computes every row of every lane and agrees with the reference on all of
     * them. This is the part that says which code produced them -- and with a
     * query-dependent mask it is the whole claim, because that layout is
     * exactly what used to fall through to it. */
    routed = cross_sdpa_route_count("cpu-cross-sdpa-batch-row");
    if (routed != 2) {
        if (routed >= 0)
            fprintf(stderr, "FAIL %d of 2 CrossSDPA nodes took the batch row "
                            "path\n", routed);
        goto fail;
    }
    if (volvoxai_engine_copy_tensor_raw("cross", cross_all, sizeof(cross_all)) != 0 ||
        volvoxai_engine_copy_tensor_raw("self", self_all, sizeof(self_all)) != 0)
        goto fail;
    for (int lane = 0; lane < LANES; lane++) {
        size_t row = (size_t)(lane * SEQUENCE + POSITIONS[lane]) * WIDTH;
        memcpy(cross[lane], cross_all + row, WIDTH * sizeof(float));
        memcpy(self[lane], self_all + row, WIDTH * sizeof(float));
    }
    volvoxai_engine_shutdown();
    return 0;
fail:
    volvoxai_engine_shutdown();
    return -1;
}

static int run_single_lane(MaskLayout layout, int lane, float cross[WIDTH], float self[WIDTH]) {
    static float cross_all[SEQUENCE * WIDTH];
    static float self_all[SEQUENCE * WIDTH];
    int routed;

    CHECK(write_graph_document(1, layout) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, NULL) == 0);
    if (set_single_lane_inputs(lane, 0, layout) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto fail;
    if (set_single_lane_inputs(lane, 1, layout) != 0) goto fail;
    if (volvoxai_engine_forward_incremental_row(POSITIONS[lane]) != 0) {
        fprintf(stderr, "FAIL lane %d's one-lane row step was declined\n", lane);
        goto fail;
    }
    /* The reference has to be the *row* path too. A one-lane run that fell back
     * to the whole sequence would still be a correct oracle for the values, but
     * it would stop this from being a comparison of two row implementations. */
    routed = cross_sdpa_route_count("cpu-cross-sdpa-row");
    if (routed != 2) {
        if (routed >= 0)
            fprintf(stderr, "FAIL lane %d: %d of 2 CrossSDPA nodes took the "
                            "one-lane row path\n", lane, routed);
        goto fail;
    }
    if (volvoxai_engine_copy_tensor_raw("cross", cross_all, sizeof(cross_all)) != 0 ||
        volvoxai_engine_copy_tensor_raw("self", self_all, sizeof(self_all)) != 0)
        goto fail;
    memcpy(cross, cross_all + (size_t)POSITIONS[lane] * WIDTH,
           WIDTH * sizeof(float));
    memcpy(self, self_all + (size_t)POSITIONS[lane] * WIDTH,
           WIDTH * sizeof(float));
    volvoxai_engine_shutdown();
    return 0;
fail:
    volvoxai_engine_shutdown();
    return -1;
}

static int compare_rows(const char* label, int lane, const float* batch,
                        const float* reference) {
    /* Bit-identical, so memcmp rather than a tolerance: both sides ran the same
     * kernel over the same values in the same order, and anything else means
     * different bytes went in. */
    if (memcmp(batch, reference, WIDTH * sizeof(float)) == 0) return 0;
    fprintf(stderr, "FAIL %s lane %d differs from its one-lane decode at:",
            label, lane);
    for (int column = 0; column < WIDTH; column++) {
        if (batch[column] == reference[column]) continue;
        fprintf(stderr, " [%d] batch=%.9g one-lane=%.9g", column,
                (double)batch[column], (double)reference[column]);
    }
    fprintf(stderr, "\n");
    return -1;
}

static int run_for_layout(MaskLayout layout) {
    static float batch_cross[LANES][WIDTH];
    static float batch_self[LANES][WIDTH];
    static float reference_cross[LANES][WIDTH];
    static float reference_self[LANES][WIDTH];

    if (run_batch(layout, batch_cross, batch_self) != 0) return 1;
    printf("ok [%s] a %d-lane CrossSDPA row step ran on the batch row path\n",
           mask_layout_name(layout), LANES);

    for (int lane = 0; lane < LANES; lane++)
        if (run_single_lane(layout, lane, reference_cross[lane],
                            reference_self[lane]) != 0) return 1;
    printf("ok [%s] each lane's one-lane decode ran on the one-lane row path\n",
           mask_layout_name(layout));

    /* Two lanes agreeing would mean the batch read one lane for both, so the
     * comparison below would hold for the wrong reason. */
    for (int lane = 1; lane < LANES; lane++) {
        if (memcmp(reference_cross[0], reference_cross[lane],
                   WIDTH * sizeof(float)) != 0) continue;
        fprintf(stderr, "FAIL [%s] lanes 0 and %d have identical reference rows; "
                        "the fixture cannot tell them apart\n",
                mask_layout_name(layout), lane);
        return 1;
    }

    for (int lane = 0; lane < LANES; lane++) {
        if (compare_rows("cross", lane, batch_cross[lane],
                         reference_cross[lane]) != 0) return 1;
        if (compare_rows("causal self", lane, batch_self[lane],
                         reference_self[lane]) != 0) return 1;
    }
    printf("ok [%s] every lane's cross and causal-self rows are bit-identical to "
           "its own one-lane decode\n", mask_layout_name(layout));
    return 0;
}

static int run_all(void) {
    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    if (run_for_layout(MASK_LAYOUT_BK) != 0) return 1;
    if (run_for_layout(MASK_LAYOUT_BQK) != 0) return 1;
    if (run_for_layout(MASK_LAYOUT_QK) != 0) return 1;
    printf("PASS CrossSDPA batch rows (BK, BQK, QK masks)\n");
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
    if (result == 0) remove(GRAPH_PATH);
    return result;
}
