/*
 * Paged K/V on the W8A8 path — the half the F32 join left out.
 *
 * `test_paged_batch_decode` joined paged addressing to B>1 through the F32
 * operators, which were the only ones that had ever seen a page table:
 * `engine_runtime_w8a8.inc` did not contain the string `paged` at all. That
 * left the two halves of a quantized decoder split down the middle — the
 * projections and the attention that a real W8A8 decode actually runs could not
 * use a paged cache.
 *
 * Three things had to change and they fail differently:
 *
 *   write   — QLinear writes its K/V row into the slot its page table names,
 *             not at `position * width`. Getting this wrong writes a real row
 *             to the wrong lane's storage.
 *   read/1  — a one-lane QSDPA gathers its prefix in logical order. Getting it
 *             wrong attends over whatever physically sits at `[0, length)`.
 *   read/N  — a batched QSDPA stages every lane's prefix into
 *             `[lanes, key_capacity, D]`. The unpaged batch deliberately does
 *             *not* stage K/V, because lane b's keys already sit where the
 *             kernel's `b * seq_kv * d_model` addressing expects them; a paged
 *             pool destroys that and nothing cheaper than staging is available.
 *
 * Both lane counts are checked against the same oracle: the identical decode
 * run one lane at a time with no cache bound at all. That makes each claim a
 * claim about paging *and* about batching — a decode that addressed lanes
 * correctly but gathered pages wrongly, or the reverse, disagrees with it.
 *
 * Bit-identical, not close: every side is int8 through the same kernels, so a
 * difference is a difference in which bytes were read.
 *
 * The whole thing runs twice: once causal-only, and once with a graph mask.
 *
 * A masked paged decode is where a mask index stops meaning "row of this
 * operand". The mask's key axis is the operand's declared extent, paged or not
 * -- it has to be, because the seed forward runs the same graph with no cache
 * bound -- but index `t` is the lane's *logical* token `t`, which under a
 * scattered mapping is not row `t` of anything. The paged run and its unpaged
 * oracle therefore carry the same keep set and must produce the same bytes,
 * while one of them reaches its keys through a page table and the other does
 * not.
 *
 * The cache's lane capacity is deliberately neither the sequence length nor the
 * pool's row count, so nothing here can pass by confusing the three.
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
#define SEQUENCE 6
#define WIDTH 8
#define VOCAB 16
#define HEADS 2
/* 1/sqrt(head_dim) with head_dim = WIDTH / HEADS = 4. */
#define ATTENTION_SCALE 0.5
/* One page per token, so the scatter shows at every position rather than only
 * at page boundaries. */
#define POOL_ROWS (LANES * SEQUENCE)
/*
 * Deliberately neither `SEQUENCE` nor `POOL_ROWS`.
 *
 * A scheduler sizes a lane's capacity from how long a request may run, not from
 * how the pool tensor happens to be spelled, so all three are independent
 * numbers in practice. Making them independent here is what stops a path that
 * confused the cache's capacity for the operand's key extent -- or for the
 * pool's row count -- from agreeing with the oracle anyway.
 */
#define LANE_CAPACITY (SEQUENCE + 2)
/*
 * The furthest position a single decoding lane reaches.
 *
 * An idle lane claims a page between this lane's at every step, so the lane's
 * own page index grows twice as fast as its position and the pool backing a
 * one-lane graph is only `SEQUENCE` rows. That, not the cache's capacity, is
 * what bounds this: a mapping past the tensor is refused rather than written.
 */
#define SINGLE_TARGET 2
/*
 * The one key this lane must not see.
 *
 * Two properties, and the fixture proves nothing without either. It has to be
 * *inside* every window the lane decodes -- a blocked key past the causal bound
 * is not a test of masking at all, because causality already excluded it and
 * the masked and unmasked runs would agree whatever the mask code did. And it
 * has to differ between lanes, or a run that read lane zero's mask row for
 * every lane would agree too.
 *
 * Both hold for every target either lane reaches here, and neither key is its
 * own query's, so no row collapses to a shorter prefix. `run_all` asserts the
 * first property on the results rather than trusting this comment.
 */
static int blocked_key(int lane) { return lane == 0 ? 1 : 0; }

static const char* GRAPH_PATH = "/tmp/volvox-w8a8-paged-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-w8a8-paged-weights.safetensors";

static const char* PAGED_TENSORS[2] = { "ka", "va" };

/*
 * A decoder's attention block, flat.
 *
 * Every node between a projection and attention would be another operator that
 * must learn page-table addressing, and the row path refuses those rather than
 * addressing them linearly -- so the fixture holds only what the paged set
 * covers.
 *
 * `ka`/`va` are the pool, spelled `[lanes, S, W]` so the projections writing
 * into them stay shape-preserving. Nothing reads that shape as a batch: the
 * physical row a page table names indexes the pool flat, and `lanes * S` rows
 * laid out `[lanes,S,W]` are the same bytes as `[1, lanes*S, W]`.
 */
static int write_graph_document(int lanes, int mask_keys) {
    char graph[4096];
    char mask_input[96] = "";
    char mask_port[24] = "";
    FILE* handle;
    int written;
    /* `mask_keys` is the mask's key axis, or zero for no mask at all. Both runs
     * declare `SEQUENCE`: the mask names the operand's key extent whether or
     * not a cache is bound, and the seed forward binds none. */
    if (mask_keys > 0) {
        if (snprintf(mask_input, sizeof(mask_input),
                     ",\"mask\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}",
                     lanes, mask_keys) >= (int)sizeof(mask_input)) return -1;
        snprintf(mask_port, sizeof(mask_port), ",\"mask\":\"mask\"");
    }
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
        "\"att\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"att_scale\","
        "\"zero_point_tensor\":\"att_zero\"},"
        "\"wq\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wq_scale\",\"zero_point_tensor\":\"wq_zero\"},"
        "\"wk\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wk_scale\",\"zero_point_tensor\":\"wk_zero\"},"
        "\"wv\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"wv_scale\",\"zero_point_tensor\":\"wv_zero\"}}},"
        "\"inputs\":{\"ids\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}%s},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\","
        "\"weight\":\"emb\"},\"outputs\":{\"out\":\"x\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wq\","
        "\"bias\":\"bq\"},\"outputs\":{\"out\":\"qa\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wk\","
        "\"bias\":\"bk\"},\"outputs\":{\"out\":\"ka\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wv\","
        "\"bias\":\"bv\"},\"outputs\":{\"out\":\"va\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        /* Causal, so each lane attends over its own prefix and nothing else --
         * which is what makes `ka`/`va` a K/V cache at all. The graph mask, when
         * present, narrows that prefix further; causality alone must not be able
         * to account for what it removes. */
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"qa\",\"k\":\"ka\","
        "\"v\":\"va\"%s},\"outputs\":{\"out\":\"att\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":%.8f}}],"
        "\"outputs\":[\"att\"]}",
        lanes, SEQUENCE, mask_input,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH,
        mask_port,
        lanes, SEQUENCE, WIDTH, HEADS, ATTENTION_SCALE);
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
 * Every term nonzero and in [-4, 4], which keeps the accumulators inside the
 * int8 grid these scales define while making each output channel depend on all
 * of them.
 *
 * The bound is worth stating because a saturated comparison cannot see a
 * difference: a projection row sums WIDTH terms of at most 4 * 4, so 128 at
 * worst, and 128 * 0.125 * 0.125 / 0.125 is 16 -- an eighth of the way to the
 * clamp. The sparser generator this replaces left one nonzero column at
 * WIDTH == 8, and a decoder whose every channel came from one input channel
 * made most keys interchangeable: masking a key changed nothing, so a mask
 * assertion could not distinguish an applied mask from a dropped one.
 */
static int8_t weight_value(int row, int column, int seed) {
    return (int8_t)(((row * 7 + column * 13 + seed * 5) % 9) - 4);
}

static int write_per_axis_pair(SafetensorsFile* file, const char* name, int count) {
    const int axis_shape[1] = { count };
    static float scales[VOCAB > WIDTH ? VOCAB : WIDTH];
    static int8_t zeros[VOCAB > WIDTH ? VOCAB : WIDTH];
    char scale_name[32];
    char zero_name[32];
    CHECK(count > 0 && (size_t)count <= sizeof(zeros));
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

/* Lane count independent, so no run can differ because it was given different
 * numbers. */
static int write_weights(void) {
    SafetensorsFile file;
    const int weight_shape[2] = { WIDTH, WIDTH };
    const int table_shape[2] = { VOCAB, WIDTH };
    const int bias_shape[1] = { WIDTH };
    const int scalar_shape[1] = { 1 };
    static int8_t emb[VOCAB * WIDTH];
    static int8_t projection[3][WIDTH * WIDTH];
    static int32_t bias[WIDTH];
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
    for (int index = 0; index < 5; index++) {
        static const char* const names[5] = { "x", "qa", "ka", "va", "att" };
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
    CHECK(write_per_axis_pair(&file, "emb", VOCAB) == 0);
    CHECK(write_per_axis_pair(&file, "wq", WIDTH) == 0);
    CHECK(write_per_axis_pair(&file, "wk", WIDTH) == 0);
    CHECK(write_per_axis_pair(&file, "wv", WIDTH) == 0);
    CHECK(safetensors_save(WEIGHTS_PATH, &file) == 0);
    safetensors_free(&file);
    return 0;
}

/* Lane-distinct and position-distinct, so "lane 1 read lane 0's row" is a
 * mismatch rather than a coincidence. */
static int token_id(int lane, int token) {
    return (lane * 7 + token * 5 + 3) % VOCAB;
}

static void fill_ids(int32_t* ids, int lanes, int lane_base) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            ids[lane * SEQUENCE + token] = token_id(lane_base + lane, token);
}

/*
 * The keep set: one blocked key per lane, everything else visible.
 *
 * Written per lane rather than shared, so a run that read lane zero's mask row
 * for every lane disagrees with the oracle instead of matching it.
 */
static void fill_mask(int32_t* mask, int lanes, int lane_base, int keys) {
    for (int lane = 0; lane < lanes; lane++)
        for (int key = 0; key < keys; key++)
            mask[lane * keys + key] = key != blocked_key(lane_base + lane);
}

/*
 * The pool is always POOL_ROWS pages and the tensor backing it always has
 * POOL_ROWS rows. A cache whose budget exceeded the tensor would map a row past
 * the end of it, which the runtime refuses -- sizing them together here is what
 * makes the fixture describe a configuration a scheduler would produce.
 *
 * The lane capacity is not the pool's shape and is not the sequence length: it
 * is how many tokens one request may hold, which is the scheduler's number. Its
 * only constraint here is that it exceed every target.
 */
static VxPagedKVCache* make_cache(void) {
    VxPagedKVOptions options;
    memset(&options, 0, sizeof(options));
    options.lanes = LANES;
    options.page_tokens = 1;
    options.lane_token_capacity = LANE_CAPACITY;
    options.max_pages = POOL_ROWS;
    return vx_paged_kv_create(&options);
}

/*
 * A paged decode of `lanes` lanes, stepping every position through the row path.
 *
 * Every K/V row is written by a row step rather than inherited from the seed,
 * and that is forced rather than stylistic: the seed is a full forward, so it
 * writes K/V *linearly*, and a scattered mapping says those rows belong to
 * other logical positions. Stepping every position puts each lane's value where
 * its own table says, which is what a prefill would also have to do.
 *
 * Appends interleave, so lane 0 holds physical pages 0,2,4,... and lane 1 holds
 * 1,3,... -- what a scheduler running two requests concurrently produces, and
 * what makes "logical row" and "physical row" different numbers. With one lane
 * the interleaving is absent but the indirection is not: a single lane still
 * takes pages in order and still reads them through the table.
 *
 * The mask, when present, is declared against `SEQUENCE`: the seed forward runs
 * before any bind, so the graph has to validate unbound too.
 */
static int run_paged(int lanes, int lane_base, const int* targets, int with_mask,
                     int8_t rows[LANES][WIDTH]) {
    VxPagedKVCache* cache = make_cache();
    const int decoy = lanes < LANES ? lanes : -1;
    static int32_t ids[LANES * SEQUENCE];
    static int32_t mask[LANES * SEQUENCE];
    static int8_t att[LANES * SEQUENCE * WIDTH];
    const size_t ids_bytes = (size_t)lanes * SEQUENCE * sizeof(int32_t);
    const size_t mask_bytes = (size_t)lanes * SEQUENCE * sizeof(int32_t);
    const size_t att_elements = (size_t)lanes * SEQUENCE * WIDTH;
    int furthest = 0;
    int scattered = 0;
    int status = -1;

    CHECK(cache != NULL);
    for (int lane = 0; lane < lanes; lane++)
        if (targets[lane] > furthest) furthest = targets[lane];

    if (write_graph_document(lanes, with_mask ? SEQUENCE : 0) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto destroy;
    fill_ids(ids, lanes, lane_base);
    if (with_mask) fill_mask(mask, lanes, lane_base, SEQUENCE);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, ids_bytes) != 0)
        goto shutdown;
    if (with_mask &&
        volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32, mask, mask_bytes) != 0)
        goto shutdown;
    if (volvoxai_engine_forward_incremental() != 0) goto shutdown;

    for (int position = 0; position <= furthest; position++) {
        int step[LANES];
        int advanced = 0;
        for (int lane = 0; lane < lanes; lane++) {
            if (position > targets[lane]) {
                step[lane] = VOLVOXAI_DECODE_LANE_PARKED;
                continue;
            }
            step[lane] = position;
            advanced++;
            if (decoy >= 0)
                CHECK(vx_paged_kv_append(cache, decoy, 1) == VX_PAGED_KV_OK);
            /* One token, now, so this lane's published length covers the row it
             * is about to write and no more. */
            CHECK(vx_paged_kv_append(cache, lane, 1) == VX_PAGED_KV_OK);
            {
                /* The premise, asserted rather than assumed: under an identity
                 * mapping every comparison below would hold whatever the paged
                 * code did. */
                long slot = -1;
                CHECK(vx_paged_kv_physical_token_index(cache, lane, position,
                                                       &slot) == VX_PAGED_KV_OK);
                if (slot != (long)position) scattered = 1;
            }
        }
        if (!advanced) continue;
        /* Lane zero is the bound lane. For a batch that choice is inert -- every
         * per-lane query takes its lane as an argument -- and for one lane it is
         * the lane. Binding is what publishes the cache to the context at all. */
        if (vx_paged_bind_locked(cache, 0, PAGED_TENSORS, 2) != 0) goto shutdown;
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                          ids, ids_bytes) != 0) goto shutdown;
        if (with_mask &&
            volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32,
                                          mask, mask_bytes) != 0) goto shutdown;
        if (lanes > 1) {
            if (volvoxai_engine_forward_incremental_rows(step, lanes) != 0) {
                fprintf(stderr, "FAIL the %d-lane paged W8A8 row step at "
                                "position %d was declined\n", lanes, position);
                goto shutdown;
            }
        } else if (volvoxai_engine_forward_incremental_row(step[0]) != 0) {
            fprintf(stderr, "FAIL the one-lane paged W8A8 row step at position "
                            "%d was declined\n", position);
            goto shutdown;
        }
    }
    if (!scattered) {
        fprintf(stderr, "FAIL the %d-lane mapping was the identity; a paged "
                        "decode matching an unpaged one says nothing\n", lanes);
        goto shutdown;
    }
    if (volvoxai_engine_copy_tensor_raw("att", att,
                                        att_elements * sizeof(int8_t)) != 0)
        goto shutdown;
    for (int lane = 0; lane < lanes; lane++)
        memcpy(rows[lane],
               att + (size_t)(lane * SEQUENCE + targets[lane]) * WIDTH, WIDTH);
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
 * No cache is bound, so K/V rows land at their logical positions. It walks the
 * same positions the paged runs walk, so the runs differ in exactly the two
 * things under test: how many lanes advance, and whether the rows go through a
 * page table. A lane's result must not depend on either.
 *
 * Its mask is the same keep set at the same stride: `blocked_key` is a property
 * of the lane, not of how the lane's keys are stored, so the two runs differ in
 * addressing alone.
 */
static int run_reference(int lane, int target, int with_mask, int8_t row[WIDTH]) {
    static int32_t ids[SEQUENCE];
    static int32_t mask[SEQUENCE];
    static int8_t att[SEQUENCE * WIDTH];
    int status = -1;

    if (write_graph_document(1, with_mask ? SEQUENCE : 0) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) return -1;
    fill_ids(ids, 1, lane);
    if (with_mask) fill_mask(mask, 1, lane, SEQUENCE);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0)
        goto shutdown;
    if (with_mask &&
        volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32, mask, sizeof(mask)) != 0)
        goto shutdown;
    if (volvoxai_engine_forward_incremental() != 0) goto shutdown;
    for (int position = 0; position <= target; position++) {
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                          ids, sizeof(ids)) != 0) goto shutdown;
        if (with_mask &&
            volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32,
                                          mask, sizeof(mask)) != 0) goto shutdown;
        if (volvoxai_engine_forward_incremental_row(position) != 0) {
            fprintf(stderr, "FAIL lane %d's unpaged row step at position %d was "
                            "declined\n", lane, position);
            goto shutdown;
        }
    }
    if (volvoxai_engine_copy_tensor_raw("att", att, sizeof(att)) != 0) goto shutdown;
    memcpy(row, att + (size_t)target * WIDTH, WIDTH);
    status = 0;
shutdown:
    volvoxai_engine_shutdown();
    return status;
}

static int compare(const char* label, int lane, const int8_t* actual,
                   const int8_t* reference) {
    if (memcmp(actual, reference, WIDTH) == 0) return 0;
    fprintf(stderr, "FAIL %s lane %d differs from its unpaged decode at:",
            label, lane);
    for (int channel = 0; channel < WIDTH; channel++) {
        if (actual[channel] == reference[channel]) continue;
        fprintf(stderr, " [%d] paged=%d unpaged=%d", channel,
                actual[channel], reference[channel]);
    }
    fprintf(stderr, "\n");
    return -1;
}

/*
 * One pass of the whole comparison, masked or not.
 *
 * `rows` receives every row this pass produced, in the order the assertions
 * below name them, so the caller can hold the unmasked pass beside the masked
 * one. A mask that changed nothing would make every equality here hold for
 * reasons unrelated to masking, and that is precisely the failure a fixture
 * whose blocked keys sat outside the causal window would not notice.
 */
static int run_all_for_mask_mode(int with_mask, int8_t rows[2 * LANES][WIDTH]) {
    /* Ragged: lanes at a common position leave `lane * S + position`
     * indistinguishable from several wrong formulas, and give the two lanes the
     * same key extent. `blocked_key` lands inside both
     * windows without being either query's own key, which is what makes the
     * masked pass differ from the unmasked one. */
    const int targets[LANES] = { 3, 1 };
    static int8_t batched[LANES][WIDTH];
    static int8_t single[LANES][WIDTH];
    static int8_t single_reference[LANES][WIDTH];
    static int8_t reference[LANES][WIDTH];
    const char* mode_name = with_mask ? "with graph mask" : "unmasked (causal)";

    for (int lane = 0; lane < LANES; lane++)
        if (run_reference(lane, targets[lane], with_mask, reference[lane]) != 0) return 1;
    /* Two lanes agreeing would make every comparison below hold for the wrong
     * reason. */
    for (int lane = 1; lane < LANES; lane++) {
        if (memcmp(reference[0], reference[lane], WIDTH) != 0) continue;
        fprintf(stderr, "FAIL [%s] lanes 0 and %d have identical reference rows; "
                        "the fixture cannot tell them apart\n", mode_name, lane);
        return 1;
    }
    /* A constant row would make the equalities hold for reasons unrelated to
     * addressing -- saturation and an all-zero closure both look like this. */
    for (int lane = 0; lane < LANES; lane++) {
        int varies = 0;
        for (int channel = 1; channel < WIDTH; channel++)
            if (reference[lane][channel] != reference[lane][0]) varies = 1;
        if (!varies) {
            fprintf(stderr, "FAIL [%s] lane %d's reference row is constant (%d)\n",
                    mode_name, lane, reference[lane][0]);
            return 1;
        }
    }
    printf("ok [%s] the unpaged one-lane oracle produced %d distinct varying rows\n",
           mode_name, LANES);

    /*
     * One lane at a time, paged: the projection writes through the table and
     * the attention gathers back through it, with no batching involved.
     *
     * A shorter prefix than the batch decodes, because the idle lane that
     * scatters the mapping also claims pages from the same pool. It is compared
     * against its own reference at the same position, so the claim is
     * unaffected -- and `blocked_key` is inside this window too, so the scalar
     * paged path carries a real mask rather than an inert one.
     */
    for (int lane = 0; lane < LANES; lane++) {
        const int one[1] = { SINGLE_TARGET };
        if (run_paged(1, lane, one, with_mask, single + lane) != 0) return 1;
        if (run_reference(lane, SINGLE_TARGET, with_mask, single_reference[lane]) != 0) return 1;
    }
    for (int lane = 0; lane < LANES; lane++)
        if (compare(with_mask ? "one-lane paged+mask" : "one-lane paged",
                    lane, single[lane], single_reference[lane]) != 0) return 1;
    printf("ok [%s] a one-lane paged W8A8 decode is bit-identical to the unpaged "
           "one\n", mode_name);

    if (run_paged(LANES, 0, targets, with_mask, batched) != 0) return 1;
    for (int lane = 0; lane < LANES; lane++)
        if (compare(with_mask ? "paged batch+mask" : "paged batch",
                    lane, batched[lane], reference[lane]) != 0)
            return 1;
    printf("ok [%s] a %d-lane paged W8A8 decode is bit-identical to each lane's "
           "own unpaged one-lane decode\n", mode_name, LANES);

    for (int lane = 0; lane < LANES; lane++) {
        memcpy(rows[lane], single[lane], WIDTH);
        memcpy(rows[LANES + lane], batched[lane], WIDTH);
    }
    return 0;
}

static int run_all(void) {
    static int8_t unmasked[2 * LANES][WIDTH];
    static int8_t masked[2 * LANES][WIDTH];
    static const char* const row_name[2 * LANES] = {
        "one-lane paged lane 0", "one-lane paged lane 1",
        "paged batch lane 0", "paged batch lane 1"
    };

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(write_weights() == 0);

    if (run_all_for_mask_mode(0, unmasked) != 0) return 1;
    if (run_all_for_mask_mode(1, masked) != 0) return 1;

    /*
     * The mask did something, everywhere.
     *
     * Every claim above is an equality between a paged run and an unpaged one,
     * and an inert mask satisfies all of them: causality alone would produce
     * both sides. `blocked_key` is chosen to fall inside every window any of
     * these four rows decodes, so each masked row must differ from the same row
     * decoded without a mask. Without this, a graph-mask path that dropped the
     * mask on the floor would be indistinguishable from one that applied it.
     */
    for (int row = 0; row < 2 * LANES; row++) {
        if (memcmp(masked[row], unmasked[row], WIDTH) != 0) continue;
        fprintf(stderr, "FAIL %s is unchanged by the graph mask; the blocked key "
                        "fell outside the window it decodes, so the masked run "
                        "proves nothing\n", row_name[row]);
        return 1;
    }
    printf("ok the graph mask changed every one of the %d compared rows\n",
           2 * LANES);

    printf("PASS W8A8 paged decode (unmasked and graph-masked)\n");
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
