/*
 * A batched decode step whose attention runs on the device.
 *
 * Every other operator in a batched decoder reached the device some time ago:
 * QEmbedding, QLinear and the elementwise ones all narrow to a contiguous run
 * of lane rows, and a run is something a binding can express. QSDPA never did.
 * Its batch form stages the per-lane query rows into a `[lanes,1,D]` operand,
 * and a *set* of rows is exactly what no binding names -- so the GPU path
 * declined it and every backend ran B>1 attention on the host, whatever the
 * rest of the step did.
 *
 * What makes it expressible is `rowIndexTransfer`: the device half of the host
 * gather and scatter. With it the step is gather the queries, attend, scatter
 * the answers -- plus, when the K/V are paged, a gather of each lane's prefix,
 * because a page table's slots are no more contiguous than the query rows are.
 *
 * Two claims, and they fail differently:
 *
 *   1. Every node of the step is routed to the device. A host fallback would
 *      still produce the right numbers -- the host batch path is correct and is
 *      what ran before -- so values alone cannot see the difference this test
 *      exists for.
 *   2. Each lane's row is bit-identical to the CPU batch's. Both sides are int8
 *      through the same kernels, so any difference is a difference in which
 *      bytes were gathered.
 *
 * Run for the contiguous case and again with a scattered page mapping, because
 * the two take different arms: the contiguous one hands K/V to the kernel where
 * they already lie, and the paged one has to assemble them first.
 *
 * And each of those twice, once with a graph mask. The keep mask the device is
 * handed is built on the host and uploaded, so a graph mask changes what is in
 * it rather than how it gets there -- but the paged arm folds a graph mask into
 * a mask that is *already* carrying each lane's staged-prefix bound, and that
 * combination has no other coverage. `run_all` asserts the mask changed the
 * answer, so a path that dropped it would fail here rather than agree.
 *
 * WIDTH is 256 so that a row clears Vulkan's 256-byte storage alignment. A
 * narrower activation is not wrong there, it just falls back to the host per
 * node -- which would make claim (1) fail for a reason that has nothing to do
 * with attention.
 */

#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "backend_manager.h"
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
#define WIDTH 256
#define VOCAB 16
#define HEADS 4
/* 1/sqrt(head_dim) with head_dim = WIDTH / HEADS = 64. */
#define ATTENTION_SCALE 0.125
#define POOL_ROWS (LANES * SEQUENCE)

static const char* GRAPH_PATH = "/tmp/volvox-device-batch-attn-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-device-batch-attn-weights.safetensors";
static const char* PAGED_TENSORS[2] = { "ka", "va" };

static int write_graph_document(int lanes, int masked) {
    char graph[4096];
    char mask_input[96] = "";
    char mask_port[24] = "";
    FILE* handle;
    int written;
    if (masked) {
        if (snprintf(mask_input, sizeof(mask_input),
                     ",\"mask\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}",
                     lanes, SEQUENCE) >= (int)sizeof(mask_input)) return -1;
        snprintf(mask_port, sizeof(mask_port), ",\"mask\":\"mask\"");
    }
    written = snprintf(graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"emb\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"emb_scale\",\"zero_point_tensor\":\"emb_zero\"},"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"u_scale\","
        "\"zero_point_tensor\":\"u_zero\"},"
        "\"qa\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"u_scale\","
        "\"zero_point_tensor\":\"u_zero\"},"
        "\"ka\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"u_scale\","
        "\"zero_point_tensor\":\"u_zero\"},"
        "\"va\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"u_scale\","
        "\"zero_point_tensor\":\"u_zero\"},"
        "\"att\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"u_scale\","
        "\"zero_point_tensor\":\"u_zero\"},"
        "\"wq\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w_scale\",\"zero_point_tensor\":\"w_zero\"},"
        "\"wk\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w_scale\",\"zero_point_tensor\":\"w_zero\"},"
        "\"wv\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w_scale\",\"zero_point_tensor\":\"w_zero\"}}},"
        "\"inputs\":{\"ids\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}%s},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\","
        "\"weight\":\"emb\"},\"outputs\":{\"out\":\"x\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wq\","
        "\"bias\":\"bias\"},\"outputs\":{\"out\":\"qa\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wk\","
        "\"bias\":\"bias\"},\"outputs\":{\"out\":\"ka\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wv\","
        "\"bias\":\"bias\"},\"outputs\":{\"out\":\"va\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"qa\",\"k\":\"ka\","
        "\"v\":\"va\"%s},\"outputs\":{\"out\":\"att\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":%.8f}}],"
        "\"outputs\":[\"att\"]}",
        lanes, SEQUENCE, mask_input,
        lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH, lanes, SEQUENCE, WIDTH,
        lanes, SEQUENCE, WIDTH, mask_port,
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

/* Sparse on purpose: eight nonzero terms per row whatever the width, so the
 * accumulators stay inside the int8 grid these scales define. A row pinned at
 * +-127 would compare equal for reasons unrelated to addressing. */
static int8_t weight_value(int row, int column, int seed) {
    if (column % 32) return 0;
    return (int8_t)(((row * 7 + column * 13 + seed * 5) % 9) - 4);
}

static int write_weights(void) {
    SafetensorsFile file;
    const int weight_shape[2] = { WIDTH, WIDTH };
    const int table_shape[2] = { VOCAB, WIDTH };
    const int bias_shape[1] = { WIDTH };
    const int scalar_shape[1] = { 1 };
    const int vocab_shape[1] = { VOCAB };
    const int width_shape[1] = { WIDTH };
    static int8_t emb[VOCAB * WIDTH];
    static int8_t projection[3][WIDTH * WIDTH];
    static int32_t bias[WIDTH];
    static float vocab_scales[VOCAB];
    static int8_t vocab_zeros[VOCAB];
    static float width_scales[WIDTH];
    static int8_t width_zeros[WIDTH];
    float unit_scale[1] = { 0.125f };
    int8_t unit_zero[1] = { 0 };

    for (int row = 0; row < VOCAB; row++) {
        vocab_scales[row] = 0.125f;
        vocab_zeros[row] = 0;
        for (int column = 0; column < WIDTH; column++)
            emb[row * WIDTH + column] = weight_value(row, column, 1);
    }
    for (int row = 0; row < WIDTH; row++) {
        width_scales[row] = 0.125f;
        width_zeros[row] = 0;
        bias[row] = 0;
        for (int column = 0; column < WIDTH; column++)
            for (int which = 0; which < 3; which++)
                projection[which][row * WIDTH + column] =
                    weight_value(row, column, 3 + which * 2);
    }

    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "emb", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 emb, sizeof(emb)) == 0);
    for (int which = 0; which < 3; which++) {
        static const char* const projections[3] = { "wq", "wk", "wv" };
        CHECK(safetensors_add_tensor(&file, projections[which], SAFETENSORS_DTYPE_I8,
                                     weight_shape, 2, projection[which],
                                     sizeof(projection[which])) == 0);
    }
    CHECK(safetensors_add_tensor(&file, "bias", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "u_scale", SAFETENSORS_DTYPE_F32, scalar_shape,
                                 1, unit_scale, sizeof(unit_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "u_zero", SAFETENSORS_DTYPE_I8, scalar_shape,
                                 1, unit_zero, sizeof(unit_zero)) == 0);
    CHECK(safetensors_add_tensor(&file, "emb_scale", SAFETENSORS_DTYPE_F32, vocab_shape,
                                 1, vocab_scales, sizeof(vocab_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "emb_zero", SAFETENSORS_DTYPE_I8, vocab_shape,
                                 1, vocab_zeros, sizeof(vocab_zeros)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_scale", SAFETENSORS_DTYPE_F32, width_shape,
                                 1, width_scales, sizeof(width_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_zero", SAFETENSORS_DTYPE_I8, width_shape,
                                 1, width_zeros, sizeof(width_zeros)) == 0);
    CHECK(safetensors_save(WEIGHTS_PATH, &file) == 0);
    safetensors_free(&file);
    return 0;
}

static void fill_ids(int32_t* ids, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            ids[lane * SEQUENCE + token] = (lane * 7 + token * 5 + 3) % VOCAB;
}

/* Which backend every executed node was routed to, compared as a prefix so a
 * kernel-qualified label like "vulkan-qsdpa-w8a8" still counts as vulkan. */
static int routes_are(const char* want) {
    const VxEngineState* state = vx_engine_state_current();
    size_t length = strlen(want);
    int matched = 0;
    if (!state) return 0;
    for (int index = 0; index < state->node_count; index++) {
        const char* route = state->runtime_route_backend[index];
        if (state->nodes[index].disabled || state->nodes[index].skip) continue;
        if (route && !strncmp(route, want, length) &&
            (route[length] == '\0' || route[length] == '-' || route[length] == '+')) {
            matched++;
            continue;
        }
        fprintf(stderr, "FAIL node %d (%s) was routed to %s, not %s\n",
                index, state->nodes[index].op, route ? route : "(nothing)", want);
        return 0;
    }
    return matched > 0;
}

/*
 * The one key this lane must not see: inside every window it decodes, and
 * different between lanes. A key past the causal bound would be excluded by
 * causality alone, and a uniform one would let a run that read lane zero's mask
 * row for every lane agree anyway. Neither is its own query's key.
 */
static int blocked_key(int lane) { return lane == 0 ? 1 : 0; }

static void fill_mask(int32_t* mask, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int key = 0; key < SEQUENCE; key++)
            mask[lane * SEQUENCE + key] = key != blocked_key(lane);
}

/*
 * One batched decode on `backend`, optionally over a scattered page mapping.
 *
 * Returns 1 when the backend is unavailable, which the caller turns into a skip
 * rather than a pass.
 */
static int run_batch(VolvoxAIEngineBackend backend, const char* route,
                     int paged, int masked, const int* targets,
                     int8_t rows[LANES][WIDTH], int* handoff) {
    const int is_cpu = backend == VOLVOXAI_BACKEND_CPU;
    VxPagedKVCache* cache = NULL;
    static int32_t ids[LANES * SEQUENCE];
    static int32_t mask[LANES * SEQUENCE];
    static int8_t att[LANES * SEQUENCE * WIDTH];
    int furthest = 0;
    int status = -1;

    *handoff = 0;
    for (int lane = 0; lane < LANES; lane++)
        if (targets[lane] > furthest) furthest = targets[lane];
    if (paged) {
        VxPagedKVOptions options;
        memset(&options, 0, sizeof(options));
        options.lanes = LANES;
        options.page_tokens = 1;
        options.lane_token_capacity = SEQUENCE;
        options.max_pages = POOL_ROWS;
        cache = vx_paged_kv_create(&options);
        CHECK(cache != NULL);
    }
    if (!is_cpu && vx_backend_manager_activate(backend) != 0) {
        vx_paged_kv_destroy(cache);
        return 1;
    }
    if (write_graph_document(LANES, masked) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto deactivate;
    fill_ids(ids, LANES);
    fill_mask(mask, LANES);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0)
        goto shutdown;
    if (masked && volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32,
                                                mask, sizeof(mask)) != 0) goto shutdown;
    if (volvoxai_engine_forward_incremental() != 0) goto shutdown;

    for (int position = 0; position <= furthest; position++) {
        int step[LANES];
        int advanced = 0;
        for (int lane = 0; lane < LANES; lane++) {
            if (position > targets[lane]) {
                step[lane] = VOLVOXAI_DECODE_LANE_PARKED;
                continue;
            }
            step[lane] = position;
            advanced++;
            if (cache) CHECK(vx_paged_kv_append(cache, lane, 1) == VX_PAGED_KV_OK);
        }
        if (!advanced) continue;
        if (cache && vx_paged_bind_locked(cache, 0, PAGED_TENSORS, 2) != 0) goto shutdown;
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                          ids, sizeof(ids)) != 0) goto shutdown;
        if (masked && volvoxai_engine_set_input_raw("mask", VOLVOXAI_DTYPE_I32,
                                                    mask, sizeof(mask)) != 0) goto shutdown;
        if (volvoxai_engine_forward_incremental_rows(step, LANES) != 0) {
            fprintf(stderr, "FAIL %s declined the %s%s %d-lane step at position %d\n",
                    route, masked ? "masked " : "", paged ? "paged" : "contiguous",
                    LANES, position);
            goto shutdown;
        }
        if (vx_incremental_hybrid_row_active_locked()) *handoff = 1;
        if (!routes_are(route)) goto shutdown;
    }
    if (volvoxai_engine_copy_tensor_raw("att", att, sizeof(att)) != 0) goto shutdown;
    for (int lane = 0; lane < LANES; lane++)
        memcpy(rows[lane],
               att + (size_t)(lane * SEQUENCE + targets[lane]) * WIDTH, WIDTH);
    status = 0;
shutdown:
    volvoxai_engine_shutdown();
deactivate:
    if (!is_cpu) vx_backend_manager_deactivate();
    vx_paged_kv_destroy(cache);
    return status;
}

static int compare(const char* label, const char* shape, int lane,
                   const int8_t* actual, const int8_t* reference) {
    int differing = 0;
    if (memcmp(actual, reference, WIDTH) == 0) return 0;
    fprintf(stderr, "FAIL %s %s lane %d differs from the CPU batch at:",
            label, shape, lane);
    for (int channel = 0; channel < WIDTH; channel++) {
        if (actual[channel] == reference[channel]) continue;
        if (differing < 8)
            fprintf(stderr, " [%d] gpu=%d cpu=%d", channel,
                    actual[channel], reference[channel]);
        differing++;
    }
    fprintf(stderr, "\n  %d of %d channels differ\n", differing, WIDTH);
    return -1;
}

static int run_case(int paged, int masked, const char* shape, const int* targets,
                    const int8_t cpu_rows[LANES][WIDTH], int* ran) {
    static const struct {
        VolvoxAIEngineBackend backend;
        const char* label;
        const char* route;
    } devices[] = {
        { VOLVOXAI_BACKEND_VULKAN, "Vulkan", "vulkan" },
        { VOLVOXAI_BACKEND_OPENGL, "OpenGL", "opengl" },
        /* CUDA joins the list by supplying `row_index_transfer`, which is the
         * one optional entry in the W8A8 ops table that gates B>1 attention. A
         * backend without it declines the batch and the host runs it -- so this
         * row is the claim that CUDA no longer does, and it skips rather than
         * passes where no CUDA device is present. */
        { VOLVOXAI_BACKEND_CUDA, "CUDA", "cuda" },
    };
    for (size_t index = 0; index < sizeof(devices) / sizeof(devices[0]); index++) {
        static int8_t gpu_rows[LANES][WIDTH];
        int handoff = 0;
        int status = run_batch(devices[index].backend, devices[index].route,
                               paged, masked, targets, gpu_rows, &handoff);
        if (status == 1) {
            printf("SKIP no %s device\n", devices[index].label);
            continue;
        }
        if (status != 0) return -1;
        if (handoff) {
            fprintf(stderr, "FAIL %s relinquished the batch to the host\n",
                    devices[index].label);
            return -1;
        }
        for (int lane = 0; lane < LANES; lane++)
            if (compare(devices[index].label, shape, lane, gpu_rows[lane],
                        cpu_rows[lane]) != 0) return -1;
        printf("ok %s ran the %s %d-lane step entirely on the device, "
               "bit-identical to the CPU batch\n", devices[index].label,
               shape, LANES);
        (*ran)++;
    }
    return 0;
}

/* The oracle for one (paged, masked) combination, plus the guards that stop it
 * from being vacuous. */
static int cpu_oracle(int paged, int masked, const char* shape,
                      const int* targets, int8_t rows[LANES][WIDTH]) {
    int handoff = 0;
    if (run_batch(VOLVOXAI_BACKEND_CPU, "cpu", paged, masked, targets, rows,
                  &handoff) != 0) return -1;
    /* Two lanes agreeing would make every comparison hold for the wrong
     * reason, and a constant row would make it hold vacuously. */
    if (memcmp(rows[0], rows[1], WIDTH) == 0) {
        fprintf(stderr, "FAIL the two lanes produced identical %s rows\n", shape);
        return -1;
    }
    for (int lane = 0; lane < LANES; lane++) {
        int varies = 0;
        for (int channel = 1; channel < WIDTH; channel++)
            if (rows[lane][channel] != rows[lane][0]) varies = 1;
        if (!varies) {
            fprintf(stderr, "FAIL lane %d's CPU %s row is constant (%d)\n",
                    lane, shape, rows[lane][0]);
            return -1;
        }
    }
    return 0;
}

static int run_all(void) {
    /* Ragged, so `lane * S + position` is distinguishable from wrong formulas
     * and the two lanes get different key extents. */
    const int targets[LANES] = { 3, 1 };
    static int8_t cpu_rows[2][2][LANES][WIDTH];
    int ran = 0;

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(write_weights() == 0);

    /*
     * Four shapes, and the paged oracle is the CPU batch over the same
     * scattered mapping: the claim here is that the device reproduces the
     * host's batch, and the host's paged batch is already checked against
     * unpaged one-lane decodes by `test_w8a8_paged_decode`.
     */
    for (int paged = 0; paged < 2; paged++) {
        for (int masked = 0; masked < 2; masked++) {
            char shape[48];
            snprintf(shape, sizeof(shape), "%s%s",
                     masked ? "masked " : "", paged ? "paged" : "contiguous");
            if (cpu_oracle(paged, masked, shape, targets,
                           cpu_rows[paged][masked]) != 0) return 1;
            printf("ok the CPU %s batch produced %d distinct varying rows\n",
                   shape, LANES);
            if (run_case(paged, masked, shape, targets,
                         (const int8_t (*)[WIDTH])cpu_rows[paged][masked],
                         &ran) != 0) return 1;
        }
    }

    /*
     * The mask did something.
     *
     * Every claim above is an equality between a device run and a host one, and
     * a graph mask that never reached either side satisfies all of them --
     * causality alone would produce both. `blocked_key` falls inside every
     * window these lanes decode, so each masked row must differ from the same
     * row decoded without a mask.
     */
    for (int paged = 0; paged < 2; paged++)
        for (int lane = 0; lane < LANES; lane++) {
            if (memcmp(cpu_rows[paged][1][lane], cpu_rows[paged][0][lane],
                       WIDTH) != 0) continue;
            fprintf(stderr, "FAIL the %s lane %d row is unchanged by the graph "
                            "mask; the masked runs prove nothing\n",
                    paged ? "paged" : "contiguous", lane);
            return 1;
        }
    printf("ok the graph mask changed every lane's row, paged and contiguous\n");

    if (!ran) return 77;
    printf("PASS device batch attention\n");
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
