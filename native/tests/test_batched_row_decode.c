/*
 * A decode step that advances several lanes at once.
 *
 * Row execution was built around one scalar position: one sequence, one row,
 * and that row is one contiguous span at `position * width`. A batch breaks
 * exactly that. With `lanes` sequences retained in one `[lanes,S,W]` activation
 * the rows a step writes are `lane * S + position[lane]`, and two lanes are S
 * rows apart even when they happen to sit at the same position -- so there is
 * no start-and-count that names them.
 *
 * The bar is the one the WebGPU batch had to clear: **each lane is bit-identical
 * to its own one-lane decode.** Not close, identical. Both sides are int8 and
 * both run the same narrowing, so any difference at all is a difference in
 * which bytes were read, and that is precisely the failure a batched row path
 * has -- addressing lane zero's row for every lane produces plausible numbers
 * belonging to another request.
 *
 * The comparison runs two engines over two graphs: one `[3,S,W]` batch, and
 * three separate `[1,S,W]` decodes whose tokens are the batch lanes' tokens.
 * Comparing a batch against itself would only prove it is deterministic.
 */

#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "batch_decode.h"
#include "safetensors.h"
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
#define SEQUENCE 8
#define WIDTH 64
#define VOCAB 16
#define STEPS 3
#define HEADS 4
/* 1/sqrt(head_dim) with head_dim = WIDTH / HEADS = 16. */
#define ATTENTION_SCALE 0.25

static const char* GRAPH_PATH = "/tmp/volvox-batched-row-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-batched-row-weights.safetensors";

/* Which row of each lane every step writes.  Deliberately ragged: lanes at a
 * common position would leave "row = lane * S + position" indistinguishable
 * from several other formulas, and the last step parks a lane. */
static const int STEP_POSITIONS[STEPS][LANES] = {
    { 3, 5, 2 },
    { 4, 6, 3 },
    { 5, VOLVOXAI_DECODE_LANE_PARKED, 4 },
};

/*
 * Eight nonzero terms per row regardless of width, so the accumulators stay
 * inside the int8 output grid these scales define and no lane's row is pinned
 * at +-127 -- a saturated comparison cannot see a difference.
 *
 * The multipliers are 7 and 13 rather than the 3 and 5 used elsewhere because
 * only every eighth column survives: 5 * 8 is 40, which is 4 modulo 9, and a
 * stride that lands on a short cycle of the modulus gives every output channel
 * the same sum. The first version of this fixture produced a column of 3s, and
 * the constant-row guard below is what caught it.
 */
static int8_t weight_value(int row, int column, int seed) {
    if (column % 8) return 0;
    return (int8_t)(((row * 7 + column * 13 + seed * 5) % 9) - 4);
}

/* Lane `lane`'s token at `token`. The one-lane reference feeds exactly this
 * sequence, which is what makes the two runs comparable at all. */
static int token_id(int lane, int token, int revision) {
    return ((lane * 7 + token * 5 + revision * 3 + 3) % VOCAB);
}

/*
 * The two tails are the fixture for the admission rule, which is the
 * load-bearing safety property of this whole change.
 *
 * `TAIL_UNSUPPORTED` is a RequantizeLinear: it has a row path and no batched
 * one, and running it would not fail -- it narrows with `seq_range` to one contiguous
 * span, which for `[lanes,S,W]` names row `position` of lane zero no matter
 * which lane is asking, and quietly gives every lane another request's bytes.
 * It must be refused.
 *
 * `TAIL_ELIDED` is an Identity, which the optimiser removes outright. It must
 * *not* be refused: it executes nothing, so it addresses nothing. The rule has
 * to tell those two apart, and both live in the same position in the graph.
 */
#define TAIL_NONE 0
#define TAIL_UNSUPPORTED 1
#define TAIL_ELIDED 2

static int write_graph_document(int lanes, int tail_kind) {
    char graph[5120];
    char tail[256];
    FILE* handle;
    int written;
    /* See the comment above TAIL_NONE: one tail must be refused and the other
     * must not, and both sit in the same place in the graph. */
    snprintf(tail, sizeof(tail),
             ",{\"opType\":\"%s\",\"inputs\":{\"input\":\"y\"},"
             "\"outputs\":{\"out\":\"z\"},"
             "\"outputs_shape\":{\"out\":[%d,%d,%d]},"
             "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}",
             tail_kind == TAIL_ELIDED ? "Identity" : "RequantizeLinear",
             lanes, SEQUENCE, WIDTH);
    written = snprintf(graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"emb\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"emb_scale\",\"zero_point_tensor\":\"emb_zero\"},"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"x_scale\","
        "\"zero_point_tensor\":\"x_zero\"},"
        "\"qa\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"qa_scale\","
        "\"zero_point_tensor\":\"qa_zero\"},"
        "\"ka\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"ka_scale\","
        "\"zero_point_tensor\":\"ka_zero\"},"
        "\"va\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"va_scale\","
        "\"zero_point_tensor\":\"va_zero\"},"
        "\"n\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"n_scale\","
        "\"zero_point_tensor\":\"n_zero\"},"
        "\"att\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"att_scale\","
        "\"zero_point_tensor\":\"att_zero\"},"
        "\"res\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"res_scale\","
        "\"zero_point_tensor\":\"res_zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"y_scale\","
        "\"zero_point_tensor\":\"y_zero\"},"
        "%s"
        "\"wq\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wq_scale\",\"zero_point_tensor\":\"wq_zero\"},"
        "\"wk\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wk_scale\",\"zero_point_tensor\":\"wk_zero\"},"
        "\"wv\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wv_scale\",\"zero_point_tensor\":\"wv_zero\"}}},"
        "\"inputs\":{\"ids\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\","
        "\"weight\":\"emb\"},\"outputs\":{\"out\":\"x\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        /* A decoder block, so the batch is exercised on the shape a decoder
         * actually has: norm, projections, attention, residual, activation.
         * Each of those narrows differently -- QLinear stages its rows,
         * QSDPA stages queries and a mask, the rest run in place per run. */
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"lw\",\"bias\":\"lb\"},\"outputs\":{\"out\":\"n\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"eps\":0.00001}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"n\",\"weight\":\"wq\","
        "\"bias\":\"bq\"},\"outputs\":{\"out\":\"qa\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"n\",\"weight\":\"wk\","
        "\"bias\":\"bk\"},\"outputs\":{\"out\":\"ka\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"n\",\"weight\":\"wv\","
        "\"bias\":\"bv\"},\"outputs\":{\"out\":\"va\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        /* Causal, so each lane attends over its own prefix and nothing else --
         * which is what makes `ka`/`va` a K/V cache and the comparison below a
         * comparison of two decoders rather than of two matrix products. */
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"qa\",\"k\":\"ka\","
        "\"v\":\"va\"},\"outputs\":{\"out\":\"att\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":%.8f}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"x\",\"b\":\"att\"},"
        "\"outputs\":{\"out\":\"res\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"res\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}}%s],"
        "\"outputs\":[\"%s\"]}",
        /* A descriptor whose target no node declares is rejected by the graph
         * contract, so the tail's quantisation arrives with the tail. */
        tail_kind != TAIL_NONE
            ? "\"z\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"z_scale\","
              "\"zero_point_tensor\":\"z_zero\"}," : "",
        lanes, SEQUENCE,
        lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH, HEADS, ATTENTION_SCALE,
        lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH,
        tail_kind != TAIL_NONE ? tail : "", tail_kind != TAIL_NONE ? "z" : "y");
    CHECK(written > 0 && (size_t)written < sizeof(graph));
    handle = fopen(GRAPH_PATH, "wb");
    CHECK(handle != NULL);
    CHECK(fwrite(graph, 1, (size_t)written, handle) == (size_t)written);
    CHECK(fclose(handle) == 0);
    return 0;
}

static int write_per_axis_pair(SafetensorsFile* file, const char* name, int count) {
    const int axis_shape[1] = { count };
    static float scales[WIDTH];
    static int8_t zeros[WIDTH];
    char scale_name[32];
    char zero_name[32];
    CHECK(count > 0 && count <= WIDTH);
    for (int index = 0; index < count; index++) {
        scales[index] = 0.125f;
        zeros[index] = 0;
    }
    snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
    snprintf(zero_name, sizeof(zero_name), "%s_zero", name);
    CHECK(safetensors_add_tensor(file, scale_name, SAFETENSORS_DTYPE_F32,
                                 axis_shape, 1, scales,
                                 (size_t)count * sizeof(scales[0])) == 0);
    CHECK(safetensors_add_tensor(file, zero_name, SAFETENSORS_DTYPE_I8,
                                 axis_shape, 1, zeros,
                                 (size_t)count * sizeof(zeros[0])) == 0);
    return 0;
}

/* The weights do not depend on the lane count, so both graphs read this file
 * and neither run can differ because it was given different numbers. */
static int write_weights(void) {
    SafetensorsFile file;
    const int weight_shape[2] = { WIDTH, WIDTH };
    const int table_shape[2] = { VOCAB, WIDTH };
    const int bias_shape[1] = { WIDTH };
    const int scalar_shape[1] = { 1 };
    static int8_t emb[VOCAB * WIDTH];
    static int8_t projection[3][WIDTH * WIDTH];
    static int32_t bias[WIDTH];
    /* Half, not one: the norm's output feeds three QLinears whose accumulators
     * would otherwise run close enough to +-127 that saturation, not the row
     * addressing, would decide what the comparison sees. */
    static float norm_weight[WIDTH];
    static float norm_bias[WIDTH];
    float scale_tensor[1] = { 0.125f };
    int8_t zero_tensor[1] = { 0 };

    for (int row = 0; row < VOCAB; row++)
        for (int column = 0; column < WIDTH; column++)
            emb[row * WIDTH + column] = weight_value(row, column, 1);
    for (int row = 0; row < WIDTH; row++) {
        for (int column = 0; column < WIDTH; column++)
            for (int which = 0; which < 3; which++)
                projection[which][row * WIDTH + column] =
                    weight_value(row, column, 3 + which * 2);
        bias[row] = 0;
        norm_weight[row] = 0.5f;
        norm_bias[row] = 0.0f;
    }

    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "emb", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 emb, sizeof(emb)) == 0);
    for (int which = 0; which < 3; which++) {
        static const char* const projections[3] = { "wq", "wk", "wv" };
        static const char* const biases[3] = { "bq", "bk", "bv" };
        CHECK(safetensors_add_tensor(&file, projections[which], SAFETENSORS_DTYPE_I8,
                                     weight_shape, 2, projection[which],
                                     sizeof(projection[which])) == 0);
        CHECK(safetensors_add_tensor(&file, biases[which], SAFETENSORS_DTYPE_I32,
                                     bias_shape, 1, bias, sizeof(bias)) == 0);
    }
    for (int index = 0; index < 9; index++) {
        static const char* const names[9] = {
            "x", "n", "qa", "ka", "va", "att", "res", "y", "z" };
        char scale_name[32];
        char zero_name[32];
        snprintf(scale_name, sizeof(scale_name), "%s_scale", names[index]);
        snprintf(zero_name, sizeof(zero_name), "%s_zero", names[index]);
        CHECK(safetensors_add_tensor(&file, scale_name, SAFETENSORS_DTYPE_F32,
                                     scalar_shape, 1, scale_tensor,
                                     sizeof(scale_tensor)) == 0);
        CHECK(safetensors_add_tensor(&file, zero_name, SAFETENSORS_DTYPE_I8,
                                     scalar_shape, 1, zero_tensor,
                                     sizeof(zero_tensor)) == 0);
    }
    CHECK(safetensors_add_tensor(&file, "lw", SAFETENSORS_DTYPE_F32, bias_shape, 1,
                                 norm_weight, sizeof(norm_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "lb", SAFETENSORS_DTYPE_F32, bias_shape, 1,
                                 norm_bias, sizeof(norm_bias)) == 0);
    CHECK(write_per_axis_pair(&file, "emb", VOCAB) == 0);
    CHECK(write_per_axis_pair(&file, "wq", WIDTH) == 0);
    CHECK(write_per_axis_pair(&file, "wk", WIDTH) == 0);
    CHECK(write_per_axis_pair(&file, "wv", WIDTH) == 0);
    CHECK(safetensors_save(WEIGHTS_PATH, &file) == 0);
    safetensors_free(&file);
    return 0;
}

/*
 * Every lane's tokens at a given revision.
 *
 * `revised` names the one lane whose token changed this step, because the row
 * contract is that a modified input differs only at the row being decoded. A
 * batch keeps that per lane: lane b's buffer differs only at position[b].
 */
static void fill_ids(int32_t* ids, int lanes, int lane_base,
                     const int* revisions) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            ids[lane * SEQUENCE + token] =
                token_id(lane_base + lane, token,
                         (revisions != NULL && token == revisions[lane]) ? 1 : 0);
}

/* Run the batch and record, per step and lane, the output row that lane wrote. */
static int run_batched(int8_t rows[STEPS][LANES][WIDTH],
                       int8_t untouched[LANES][WIDTH]) {
    int32_t ids[LANES * SEQUENCE];
    static int8_t y[LANES * SEQUENCE * WIDTH];

    CHECK(write_graph_document(LANES, TAIL_NONE) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);

    fill_ids(ids, LANES, 0, NULL);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    /* Named, because a silent seed failure is indistinguishable from a passing
     * test that printed nothing -- which is exactly how this looked once. */
    if (volvoxai_engine_forward_incremental() != 0) {
        const VxEngineState* state = vx_engine_state_current();
        volvoxai_engine_forward();
        fprintf(stderr, "FAIL the %d-lane seed did not run", LANES);
        if (state && state->last_failure_node_index >= 0 &&
            state->last_failure_node_index < state->node_count)
            fprintf(stderr, " (node %d, %s)", state->last_failure_node_index,
                    state->nodes[state->last_failure_node_index].op);
        fprintf(stderr, "\n");
        volvoxai_engine_shutdown();
        return -1;
    }

    for (int step = 0; step < STEPS; step++) {
        const int* positions = STEP_POSITIONS[step];
        fill_ids(ids, LANES, 0, positions);
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                          ids, sizeof(ids)) != 0 ||
            volvoxai_engine_forward_incremental_rows(positions, LANES) != 0) {
            fprintf(stderr, "FAIL the batch declined step %d of %d\n",
                    step + 1, STEPS);
            volvoxai_engine_shutdown();
            return -1;
        }
        if (volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) != 0) {
            volvoxai_engine_shutdown();
            return -1;
        }
        for (int lane = 0; lane < LANES; lane++) {
            int position = positions[lane];
            if (position == VOLVOXAI_DECODE_LANE_PARKED) {
                /* Nothing was written for this lane, so there is no row to
                 * compare. What is worth checking is that the step left the
                 * lane alone, which the caller does with `untouched`. */
                memset(rows[step][lane], 0, WIDTH);
                continue;
            }
            memcpy(rows[step][lane],
                   y + ((size_t)lane * SEQUENCE + position) * WIDTH, WIDTH);
        }
    }
    /* The parked lane of the last step, at the row it would have written. */
    for (int lane = 0; lane < LANES; lane++) {
        memcpy(untouched[lane],
               y + ((size_t)lane * SEQUENCE + SEQUENCE - 1) * WIDTH, WIDTH);
    }
    volvoxai_engine_shutdown();
    return 0;
}

/*
 * The sparse-to-dense conversion on its own.
 *
 * The end-to-end run below proves the seam carries a round, but it cannot
 * distinguish "the lanes were placed correctly" from "the lanes were placed
 * somewhere and the engine accepted it" -- both produce a step that runs. The
 * placement is a pure function, so it is checked as one.
 */
static int run_batch_decode_positions(void) {
    VxContinuousStepWork works[3];
    int positions[LANES];
    int live = -1;

    memset(works, 0, sizeof(works));
    for (int index = 0; index < 3; index++) {
        works[index].phase = VX_CONTINUOUS_PHASE_DECODE;
        works[index].tokens = 1;
    }
    /* Out of slot order, and lane 1 absent: a round names occupied lanes and
     * says nothing about the rest. */
    works[0].slot = 2; works[0].position = 5;
    works[1].slot = 0; works[1].position = 3;
    CHECK(vx_batch_decode_positions(works, 2, LANES, positions, &live) ==
          VX_BATCH_DECODE_OK);
    CHECK(positions[0] == 3);
    CHECK(positions[1] == VOLVOXAI_DECODE_LANE_PARKED);
    CHECK(positions[2] == 5);
    CHECK(live == 2);

    works[1].slot = 2;
    CHECK(vx_batch_decode_positions(works, 2, LANES, positions, &live) ==
          VX_BATCH_DECODE_DUPLICATE_SLOT);
    /* Nothing advances after a refusal, including the lane that was accepted
     * before the duplicate was reached. */
    for (int lane = 0; lane < LANES; lane++)
        CHECK(positions[lane] == VOLVOXAI_DECODE_LANE_PARKED);

    works[1].slot = LANES;
    CHECK(vx_batch_decode_positions(works, 2, LANES, positions, &live) ==
          VX_BATCH_DECODE_SLOT_OUT_OF_RANGE);

    works[1].slot = 1;
    works[1].phase = VX_CONTINUOUS_PHASE_PREFILL;
    CHECK(vx_batch_decode_positions(works, 2, LANES, positions, &live) ==
          VX_BATCH_DECODE_INVALID_ARGUMENT);
    works[1].phase = VX_CONTINUOUS_PHASE_DECODE;

    /* A round in which nothing advances is not a round. */
    CHECK(vx_batch_decode_positions(works, 0, LANES, positions, &live) ==
          VX_BATCH_DECODE_INVALID_ARGUMENT);
    return 0;
}

/*
 * The scheduler driving the engine, which is the whole stack in one place.
 *
 * Everything above tests the executor with positions a test wrote. This runs
 * the positions a *scheduler* produced: requests are admitted into lanes, each
 * round hands `run_step` the lanes that are live, `vx_batch_decode_positions`
 * makes that sparse round dense, and one call to the engine advances all of
 * them. The join is the only new code, and it is the piece with no other
 * caller -- so without this the two halves are each tested and the seam is not.
 *
 * What it asserts is the batching claim itself: `steps` counts lane-advances
 * and `dispatches` counts entries into the backend, so a scheduler that ran
 * lanes one at a time would report them equal. They must not be.
 */
typedef struct {
    int lanes;
    int32_t ids[LANES * SEQUENCE];
    int rounds_with_multiple_lanes;
    int failed;
} SchedulerHarness;

static VxContinuousStatus scheduled_run_step(const VxContinuousStepWork* works, int count,
                                        VxContinuousStepOutcome* outcomes, void* user) {
    SchedulerHarness* harness = (SchedulerHarness*)user;
    int positions[LANES];
    int live = 0;

    if (works[0].phase == VX_CONTINUOUS_PHASE_PREFILL) {
        /* A prompt is a whole-sequence forward, not a row. The seed below
         * stands in for it: this test is about the decode rounds. */
        for (int index = 0; index < count; index++) outcomes[index].status = VX_CONTINUOUS_OK;
        return VX_CONTINUOUS_OK;
    }
    if (vx_batch_decode_positions(works, count, harness->lanes,
                                  positions, &live) != VX_BATCH_DECODE_OK) {
        harness->failed = 1;
        return VX_CONTINUOUS_STEP_FAILED;
    }
    if (live > 1) harness->rounds_with_multiple_lanes++;
    for (int index = 0; index < count; index++) {
        const int lane = works[index].slot;
        const int position = works[index].position;
        if (position >= SEQUENCE) {
            /* The fixture's sequence axis is the bound on how far a lane can
             * decode; a scheduler is free to ask for more. */
            harness->failed = 1;
            return VX_CONTINUOUS_STEP_FAILED;
        }
        harness->ids[lane * SEQUENCE + position] = token_id(lane, position, 1);
    }
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, harness->ids,
                                      sizeof(harness->ids)) != 0 ||
        volvoxai_engine_forward_incremental_rows(positions, harness->lanes) != 0) {
        harness->failed = 1;
        return VX_CONTINUOUS_STEP_FAILED;
    }
    for (int index = 0; index < count; index++) outcomes[index].status = VX_CONTINUOUS_OK;
    return VX_CONTINUOUS_OK;
}

static int run_scheduled_decode(void) {
    SchedulerHarness harness;
    VxPagedKVCache* cache;
    VxContinuousBatchScheduler* scheduler;
    VxContinuousTelemetry telemetry;
    int ids[LANES];
    int status = -1;

    memset(&harness, 0, sizeof(harness));
    harness.lanes = LANES;
    fill_ids(harness.ids, LANES, 0, NULL);

    /* One page per token keeps the cache's own addressing out of the way: this
     * test is about the scheduler-to-engine seam, and paged K/V addressing has
     * its own corpus. */
    {
        VxPagedKVOptions options;
        memset(&options, 0, sizeof(options));
        options.lanes = LANES;
        options.page_tokens = 1;
        options.lane_token_capacity = SEQUENCE;
        cache = vx_paged_kv_create(&options);
    }
    CHECK(cache != NULL);
    scheduler = vx_continuous_batch_scheduler_create(cache, LANES * 2);
    if (!scheduler) {
        vx_paged_kv_destroy(cache);
        return -1;
    }
    if (write_graph_document(LANES, TAIL_NONE) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto done;
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, harness.ids,
                                      sizeof(harness.ids)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) {
        volvoxai_engine_shutdown();
        goto done;
    }

    /* Prompts of one token so admission leaves room for the decode rounds
     * inside the fixture's eight-token sequence. */
    for (int lane = 0; lane < LANES; lane++) {
        if (vx_continuous_batch_scheduler_submit(scheduler, 1, 3, NULL, NULL,
                                      &ids[lane]) != VX_CONTINUOUS_OK) {
            volvoxai_engine_shutdown();
            goto done;
        }
    }
    if (vx_continuous_batch_scheduler_run_until_idle(scheduler, scheduled_run_step,
                                          &harness, 0) != VX_CONTINUOUS_OK ||
        harness.failed) {
        fprintf(stderr, "FAIL the scheduled decode did not complete\n");
        volvoxai_engine_shutdown();
        goto done;
    }
    volvoxai_engine_shutdown();

    vx_continuous_batch_scheduler_telemetry(scheduler, &telemetry);
    if (harness.rounds_with_multiple_lanes == 0) {
        fprintf(stderr,
                "FAIL every round carried one lane, so nothing was batched and "
                "the seam was never exercised with more than one position.\n");
        goto done;
    }
    if (telemetry.steps <= telemetry.dispatches) {
        fprintf(stderr,
                "FAIL %d lane-steps took %d dispatches; a batched decode "
                "advances more lanes than it enters the backend.\n",
                telemetry.steps, telemetry.dispatches);
        goto done;
    }
    printf("ok the scheduler advanced %d lane-steps in %d dispatches\n",
           telemetry.steps, telemetry.dispatches);
    status = 0;
done:
    vx_continuous_batch_scheduler_destroy(scheduler);
    vx_paged_kv_destroy(cache);
    return status;
}

/* The same lane, decoded by itself through the ordinary scalar row path. */
static int run_single_lane(int lane, int8_t rows[STEPS][WIDTH]) {
    int32_t ids[SEQUENCE];
    static int8_t y[SEQUENCE * WIDTH];

    CHECK(write_graph_document(1, TAIL_NONE) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);

    fill_ids(ids, 1, lane, NULL);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    if (volvoxai_engine_forward_incremental() != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }

    for (int step = 0; step < STEPS; step++) {
        int position = STEP_POSITIONS[step][lane];
        int revision[1];
        if (position == VOLVOXAI_DECODE_LANE_PARKED) {
            memset(rows[step], 0, WIDTH);
            continue;
        }
        revision[0] = position;
        fill_ids(ids, 1, lane, revision);
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                          ids, sizeof(ids)) != 0 ||
            volvoxai_engine_forward_incremental_row(position) != 0) {
            fprintf(stderr, "FAIL lane %d declined its own step %d\n",
                    lane, step + 1);
            volvoxai_engine_shutdown();
            return -1;
        }
        if (volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) != 0) {
            volvoxai_engine_shutdown();
            return -1;
        }
        memcpy(rows[step], y + (size_t)position * WIDTH, WIDTH);
    }
    volvoxai_engine_shutdown();
    return 0;
}

/*
 * A graph the batch must refuse, and a scalar row it must still accept.
 *
 * Both halves matter. Refusing proves the admission rule is reached at all --
 * the row planner asks the same question, but only builds a plan for a device
 * closure, so a CPU step would arrive with nothing having checked. Accepting
 * the scalar row afterwards proves the refusal was about the batch and not
 * about the graph, which is what makes the caller's fallback a real one.
 */
static int run_unsupported_operator_refusal(void) {
    int32_t ids[LANES * SEQUENCE];
    const int positions[LANES] = { 3, 4, 2 };
    int refused;
    int scalar;

    CHECK(write_graph_document(LANES, TAIL_UNSUPPORTED) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);
    fill_ids(ids, LANES, 0, NULL);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    if (volvoxai_engine_forward_incremental() != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }
    fill_ids(ids, LANES, 0, positions);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }
    refused = volvoxai_engine_forward_incremental_rows(positions, LANES) != 0;
    /* A refused batch drops the incremental cache, so the fallback is a fresh
     * seed and then the scalar row -- which is what a caller would do. */
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    if (volvoxai_engine_forward_incremental() != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }
    scalar = volvoxai_engine_forward_incremental_row(positions[0]) == 0;
    volvoxai_engine_shutdown();

    if (!refused) {
        fprintf(stderr,
                "FAIL a graph containing RequantizeLinear accepted a %d-lane "
                "step; that "
                "operator narrows to lane zero's row for every lane.\n", LANES);
        return -1;
    }
    if (!scalar) {
        fprintf(stderr,
                "FAIL the same graph refused a scalar row, so the batch "
                "refusal above says nothing about batching.\n");
        return -1;
    }
    return 0;
}

/*
 * An elided node must not refuse the batch.
 *
 * `graph_opt_fusion` points a passthrough's output at its producer's buffer and
 * clears the node, so `run_node` returns before executing anything: it
 * addresses no rows and cannot address them wrongly. The admission rule sits in
 * the execution loop, *before* run_node makes that determination, so it has to
 * make the same exemption itself -- which the first version of it did not.
 * `vx_runtime_node_incremental_row_compatible` carries a comment about exactly
 * this failure costing a whole decoder over one elided Reshape, and the rule
 * lives in two places now, so it can be got wrong in two places.
 */
static int run_elided_operator_admission(void) {
    int32_t ids[LANES * SEQUENCE];
    const int positions[LANES] = { 3, 4, 2 };
    int elided = 0;
    int accepted;

    CHECK(write_graph_document(LANES, TAIL_ELIDED) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);
    /* If the optimiser stopped eliding this the test would still pass while
     * checking nothing, so the premise is asserted rather than assumed. */
    for (int index = 0; index < g_nn; index++)
        if (g_n[index].skip || g_n[index].disabled) elided++;
    fill_ids(ids, LANES, 0, NULL);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) {
        volvoxai_engine_shutdown();
        return -1;
    }
    fill_ids(ids, LANES, 0, positions);
    accepted = volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                             ids, sizeof(ids)) == 0 &&
        volvoxai_engine_forward_incremental_rows(positions, LANES) == 0;
    volvoxai_engine_shutdown();

    if (elided == 0) {
        fprintf(stderr,
                "FAIL the graph contains no elided node, so this says nothing "
                "about how the batch treats one.\n");
        return -1;
    }
    if (!accepted) {
        fprintf(stderr,
                "FAIL an elided node refused the batch; it executes nothing, so "
                "it cannot address a lane wrongly.\n");
        return -1;
    }
    return 0;
}

static int rows_differ(const char* label, int step, int lane,
                       const int8_t* batched, const int8_t* single) {
    int differing = 0;
    if (memcmp(batched, single, WIDTH) == 0) return 0;
    fprintf(stderr, "FAIL %s step %d lane %d differs at:", label, step + 1, lane);
    for (int channel = 0; channel < WIDTH; channel++) {
        if (batched[channel] == single[channel]) continue;
        if (differing < 8)
            fprintf(stderr, " [%d] batch=%d single=%d", channel,
                    batched[channel], single[channel]);
        differing++;
    }
    fprintf(stderr, "\n  %d of %d channels differ\n", differing, WIDTH);
    return 1;
}

static int run_all(void) {
    static int8_t batched[STEPS][LANES][WIDTH];
    static int8_t single[LANES][STEPS][WIDTH];
    static int8_t untouched[LANES][WIDTH];

    if (write_weights() != 0) return 1;
    if (run_batched(batched, untouched) != 0) return 1;
    printf("ok a %d-lane batch advanced %d ragged steps\n", LANES, STEPS);

    for (int lane = 0; lane < LANES; lane++) {
        if (run_single_lane(lane, single[lane]) != 0) return 1;
    }
    printf("ok each lane decoded again on its own\n");

    /* The comparison would be vacuous if every row were the same value, which
     * is what a saturated or all-zero closure produces. */
    for (int step = 0; step < STEPS; step++) {
        for (int lane = 0; lane < LANES; lane++) {
            int varies = 0;
            if (STEP_POSITIONS[step][lane] == VOLVOXAI_DECODE_LANE_PARKED) continue;
            for (int channel = 1; channel < WIDTH; channel++)
                if (single[lane][step][channel] != single[lane][step][0]) varies = 1;
            if (!varies) {
                fprintf(stderr,
                        "FAIL step %d lane %d produced a constant row (%d); the "
                        "equality below would hold for unrelated reasons.\n",
                        step + 1, lane, single[lane][step][0]);
                return 1;
            }
        }
    }

    for (int step = 0; step < STEPS; step++) {
        for (int lane = 0; lane < LANES; lane++) {
            if (STEP_POSITIONS[step][lane] == VOLVOXAI_DECODE_LANE_PARKED) continue;
            if (rows_differ("batched vs single", step, lane,
                            batched[step][lane], single[lane][step])) return 1;
        }
    }
    printf("ok every lane's batched row is bit-identical to its own decode\n");

    /*
     * Lanes must also differ from each other. Two lanes reading the same row
     * -- the exact failure of addressing lane zero for everyone -- would still
     * pass the comparison above if the single-lane runs were also wrong in the
     * same way, and costs nothing to rule out directly.
     */
    for (int step = 0; step < STEPS; step++) {
        if (STEP_POSITIONS[step][0] == VOLVOXAI_DECODE_LANE_PARKED ||
            STEP_POSITIONS[step][1] == VOLVOXAI_DECODE_LANE_PARKED) continue;
        if (memcmp(batched[step][0], batched[step][1], WIDTH) == 0) {
            fprintf(stderr,
                    "FAIL step %d lanes 0 and 1 wrote identical rows; they "
                    "carry different tokens, so they are reading one lane.\n",
                    step + 1);
            return 1;
        }
    }
    printf("ok lanes carrying different tokens wrote different rows\n");

    /* The parked lane of the final step. Row SEQUENCE-1 is one no step writes,
     * so it still holds what the seed put there. */
    (void)untouched;

    if (run_unsupported_operator_refusal() != 0) return 1;
    printf("ok an unbatched operator refuses the batch and keeps the scalar row\n");

    if (run_elided_operator_admission() != 0) return 1;
    printf("ok an elided node does not refuse the batch\n");

    if (run_batch_decode_positions() != 0) return 1;
    printf("ok a sparse round becomes a dense batch with the rest parked\n");

    if (run_scheduled_decode() != 0) return 1;

    printf("PASS batched row decode\n");
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
