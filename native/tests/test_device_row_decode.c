/*
 * A decode row that stays on the GPU.
 *
 * Vulkan, OpenGL and Metal used to hand the whole decode to the host after the
 * first row. The reason was not that their kernels could not do a row -- those
 * kernels already take a row count -- but that their slot map is keyed by
 * complete host-span bases, so a row, which is an interior pointer into an
 * activation, had no device address at all. The runtime therefore synchronised
 * the prefix to the host exactly once and never went back, and every token
 * after the first ran on the CPU of a machine that had been asked for a GPU.
 *
 * Three assertions here, and they fail differently. All are needed:
 *
 *   1. The handoff does not happen. `vx_incremental_hybrid_row_active_locked()`
 *      stays zero for the whole decode, which rules out the path this replaces:
 *      relinquish the prefix once, then decode on the host forever.
 *
 *   2. Every node of every row step is routed to the device backend. On its own
 *      (1) cannot say this, because a device-row backend never arms the hybrid
 *      flag in the first place -- the runtime skips that whole branch for it.
 *      Were the backend's graph to decline each node at dispatch, the registry's
 *      CPU backend would pick them up, the flag would still read zero, and (1)
 *      would pass while nothing ran on the device. The per-node route is the
 *      part that actually names who executed the row.
 *
 *   3. Every step is bit-identical to the CPU backend's row decode. Both sides
 *      are int8 and both run the same narrowing, so "identical" is the right
 *      bar, not "close".
 *
 * Vulkan and OpenGL both run the whole thing, and they resolve a row the same
 * way -- find the containing slot, offset into it -- but they bind it
 * differently, because Vulkan already had `(io_buffer, offset, bytes)` for every
 * binding while OpenGL bound each slot's own buffer whole. Running the identical
 * body over both is the point: a claim that holds for one binding model and not
 * the other is a claim about that model, not about device rows.
 *
 * The graph is two chained QLinear nodes on purpose: the first node's output
 * row is the second node's input row, so a window has to survive being written
 * and then read within one step. A single node would prove only half of that.
 *
 * They sit behind a QEmbedding, and the graph input is [1,S] token ids rather
 * than an [1,S,D] activation, because that is what the runtime recognises as a
 * decode input: `hybrid_model_has_row_closure_locked` looks for a rank-2 [1,S]
 * graph input and probes the closure reachable from it. Feeding embeddings in
 * directly leaves it with no candidate to probe, and the row is refused before
 * any of the device-window code this test covers gets a chance to run.
 */

#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#include "backend_manager.h"
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

/* WIDTH is 256 because that is what the device will bind, not because the
 * arithmetic needs it. A window's offset is `row * width * element size`, and
 * `graph_window_finish` requires it to be a multiple of the storage-buffer
 * alignment -- 256 on every adapter measured so far. An int8 activation
 * narrower than 256 has no bindable row at all and falls back to the host,
 * which is what an 8-wide version of this fixture did silently. */
#define SEQUENCE 8
#define WIDTH 256
#define MAX_WIDTH 768
#define VOCAB 16
#define STEPS 3

static const char* GRAPH_PATH = "/tmp/volvox-device-row-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-device-row-weights.safetensors";

/* The width is a parameter because the alignment sweep below varies it. The
 * rest of the document is fixed: three nodes, one [1,S] token input, and
 * per-tensor affine quantisation on every activation. */
static int write_graph_document(int width, int lanes) {
    char graph[2048];
    FILE* handle;
    int written = snprintf(graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"emb\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"emb_scale\",\"zero_point_tensor\":\"emb_zero\"},"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"x_scale\","
        "\"zero_point_tensor\":\"x_zero\"},"
        "\"h\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"h_scale\","
        "\"zero_point_tensor\":\"h_zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"y_scale\","
        "\"zero_point_tensor\":\"y_zero\"},"
        "\"w0\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w0_scale\",\"zero_point_tensor\":\"w0_zero\"},"
        "\"w1\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w1_scale\",\"zero_point_tensor\":\"w1_zero\"}}},"
        "\"inputs\":{\"ids\":{\"shape\":[%d,%d],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\","
        "\"weight\":\"emb\"},\"outputs\":{\"out\":\"x\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w0\","
        "\"bias\":\"b0\"},\"outputs\":{\"out\":\"h\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}},"
        "{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"h\",\"weight\":\"w1\","
        "\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{}}],"
        "\"outputs\":[\"y\"]}",
        lanes, SEQUENCE,
        lanes, SEQUENCE, width, lanes, SEQUENCE, width,
        lanes, SEQUENCE, width);
    CHECK(written > 0 && (size_t)written < sizeof(graph));
    handle = fopen(GRAPH_PATH, "wb");
    CHECK(handle != NULL);
    CHECK(fwrite(graph, 1, (size_t)written, handle) == (size_t)written);
    CHECK(fclose(handle) == 0);
    return 0;
}

/*
 * A wide weight whose rows carry only eight nonzero terms.
 *
 * Widening the tensors to reach a bindable row would otherwise widen the
 * accumulators with them: 256 full terms of int8-by-int8 overflow the output
 * grid these scales define, and a comparison whose outputs are pinned at +-127
 * on both sides stops being able to see a difference. Every operand shares the
 * same sparsity residue, so the products land on each other rather than
 * cancelling to a column of zeros -- an equally useless comparison.
 */
static int8_t weight_value(int row, int column, int seed) {
    if (column % 32) return 0;
    return (int8_t)(((row * 3 + column * 5 + seed) % 9) - 4);
}

/* Every scale is 0.125 and every zero point is 0, so a product of two
 * quantised operands re-narrows to the same grid it came from. Both accumulator
 * chains stay well inside int8 (|h| <= 16, |y| <= 64), which keeps saturation
 * out of a comparison whose whole point is exact equality. */
static int write_per_axis_pair(SafetensorsFile* file, const char* name,
                               int count) {
    const int axis_shape[1] = { count };
    static float scales[MAX_WIDTH];
    static int8_t zeros[MAX_WIDTH];
    char scale_name[32];
    char zero_name[32];
    CHECK(count > 0 && count <= MAX_WIDTH);
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

static int write_fixture(int width, int lanes) {
    SafetensorsFile file;
    const int weight_shape[2] = { width, width };
    const int table_shape[2] = { VOCAB, width };
    const int bias_shape[1] = { width };
    const int scalar_shape[1] = { 1 };
    /* Static because a 768-wide pair of weights is 1.1 MB, which is more than
     * the default stack wants to carry. */
    static int8_t emb[VOCAB * MAX_WIDTH];
    static int8_t w0[MAX_WIDTH * MAX_WIDTH];
    static int8_t w1[MAX_WIDTH * MAX_WIDTH];
    static int32_t b0[MAX_WIDTH];
    static int32_t b1[MAX_WIDTH];
    float scale_tensor[1] = { 0.125f };
    /* Zero points carry the dtype of the tensor they quantise, not int32:
     * `per_axis_scale_source` compares them against `weight->dtype` and
     * `parse_safetensors_per_tensor_quantization` against the activation's. */
    int8_t zero_tensor[1] = { 0 };

    CHECK(width > 0 && width <= MAX_WIDTH);
    for (int row = 0; row < VOCAB; row++) {
        for (int column = 0; column < width; column++) {
            emb[row * width + column] = weight_value(row, column, 1);
        }
    }
    for (int row = 0; row < width; row++) {
        for (int column = 0; column < width; column++) {
            w0[row * width + column] = weight_value(row, column, 3);
            w1[row * width + column] = weight_value(row, column, 5);
        }
        b0[row] = 0;
        b1[row] = 0;
    }

    CHECK(write_graph_document(width, lanes) == 0);

    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "emb", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 emb, (size_t)VOCAB * width) == 0);
    CHECK(safetensors_add_tensor(&file, "w0", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                                 w0, (size_t)width * width) == 0);
    CHECK(safetensors_add_tensor(&file, "w1", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                                 w1, (size_t)width * width) == 0);
    CHECK(safetensors_add_tensor(&file, "b0", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 b0, (size_t)width * sizeof(b0[0])) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 b1, (size_t)width * sizeof(b1[0])) == 0);
    for (int index = 0; index < 3; index++) {
        static const char* const names[3] = { "x", "h", "y" };
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
    CHECK(write_per_axis_pair(&file, "w0", width) == 0);
    CHECK(write_per_axis_pair(&file, "w1", width) == 0);
    CHECK(safetensors_save(WEIGHTS_PATH, &file) == 0);
    safetensors_free(&file);
    return 0;
}

/* Tokens past `active` stay at id 0. Token r's id never changes once written,
 * so row r of every activation is reproducible across the steps that follow
 * it -- which is what makes a row decode comparable to a whole-sequence one. */
static void fill_ids(int32_t* ids, int active) {
    memset(ids, 0, (size_t)SEQUENCE * sizeof(*ids));
    for (int token = 0; token < active; token++) {
        ids[token] = (token * 5 + 3) % VOCAB;
    }
}

/* The same tokens, laid out for `lanes` sequences. Lane l's token t differs
 * from lane l's token t nowhere else, so a lane in the batch carries exactly
 * what it would carry alone. */
static void fill_batch_ids(int32_t* ids, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            ids[lane * SEQUENCE + token] = (lane * 7 + token * 5 + 3) % VOCAB;
}

/*
 * Where a node is expected to run during a row step.
 *
 * The route it is compared against comes from `runtime_route_backend`, which
 * run_node() clears on entry and writes on the way out -- so after a step it
 * describes that step, for the nodes the step actually ran. A node whose inputs
 * were all clean is skipped and keeps whatever the previous step left there,
 * which is why the caller only reads it when every node is known to be dirty.
 *
 * Every node, QEmbedding included. That last one was the exception until the
 * ids stopped being bound as a window: a token row is one i32, four bytes at
 * offset `row * 4`, and no such offset is a multiple of Vulkan's 256-byte
 * storage alignment, so the ids window resolved for row 0 and nothing else.
 * Mesa's OpenGL aligns storage bindings to 4, where the same row bound fine --
 * which is the reason this is worth stating as a rule rather than tolerating
 * per driver: it made "does the embedding run on the GPU" a fact about the
 * adapter. The ids now bind whole with the row named as a shader scalar
 * (`token_offset`), so the answer is yes on every device.
 */
static const char* expected_route(const char* op, const char* device_prefix) {
    (void)op;
    return device_prefix;
}

static int routes_match_expectation(const char* device_prefix, int step) {
    const VxEngineState* state = vx_engine_state_current();
    int matched = 0;
    int wrong = 0;
    if (!state) return 0;
    for (int index = 0; index < state->node_count; index++) {
        const char* route = state->runtime_route_backend[index];
        const char* want;
        size_t length;
        if (state->nodes[index].disabled || state->nodes[index].skip) continue;
        want = expected_route(state->nodes[index].op, device_prefix);
        length = strlen(want);
        /* Route labels are "<backend>" or "<backend>-<kernel>"; a bare prefix
         * match would let "cpuish" satisfy a request for "cpu". */
        if (route && !strncmp(route, want, length) &&
            (route[length] == '\0' || route[length] == '-' ||
             route[length] == '+')) {
            matched++;
            continue;
        }
        /* Every node, not just the first: which of them fell back is the whole
         * content of this failure, and stopping at the first one turns three
         * facts into one. */
        fprintf(stderr, "FAIL step %d node %d (%s) was routed to %s, not %s\n",
                step, index, state->nodes[index].op,
                route ? route : "(nothing)", want);
        wrong++;
    }
    return !wrong && matched > 0;
}

/*
 * Seed, then three row steps, recording the row each step writes.
 *
 * `handoff_seen` and the per-step route check are the load-bearing outputs; the
 * rows they guard are only worth comparing once both hold. See the header.
 */
static int decode_steps(int8_t rows[STEPS][WIDTH], int* handoff_seen,
                        const char* label, const char* device_prefix) {
    int32_t ids[SEQUENCE];

    fill_ids(ids, 1);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);

    for (int index = 0; index < STEPS; index++) {
        int position = index + 1;
        int8_t y[SEQUENCE * WIDTH];
        fill_ids(ids, position + 1);
        CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
        /* Named rather than CHECKed: a declined row is the failure this test
         * exists to describe, and "step 2 of 3" is the first thing worth
         * knowing about it. */
        if (volvoxai_engine_forward_incremental_row(position) != 0) {
            fprintf(stderr, "FAIL %s declined the row decode at step %d of %d "
                            "(row %d)\n", label, index + 1, STEPS, position);
            return -1;
        }
        if (vx_incremental_hybrid_row_active_locked()) *handoff_seen = 1;
        /* Every node is dirty on every step here -- the token input feeds the
         * embedding, which feeds both QLinears -- so every route is this
         * step's. */
        if (!routes_match_expectation(device_prefix, index + 1)) return -1;
        CHECK(volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) == 0);
        memcpy(rows[index], y + (size_t)position * WIDTH, WIDTH);
    }
    return 0;
}

/* Teardown is unconditional. A test that returns while the engine is still
 * loaded trips the state assertion in vx_engine_state_deinit, and the abort
 * buries whatever the test had just printed about why it was returning. */
static int run_decode(VolvoxAIEngineBackend backend, const char* label,
                      const char* route, int8_t rows[STEPS][WIDTH],
                      int* handoff_seen) {
    const int is_cpu = backend == VOLVOXAI_BACKEND_CPU;
    int status;
    *handoff_seen = 0;

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    /* CPU is the default, and asking the manager to "deactivate" before any
     * backend was ever activated tears down state that was never built. */
    if (!is_cpu && vx_backend_manager_activate(backend) != 0) return 1; /* unavailable */
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);

    status = decode_steps(rows, handoff_seen, label, route);

    volvoxai_engine_shutdown();
    if (!is_cpu) vx_backend_manager_deactivate();
    return status == 0 ? 0 : -1;
}

/*
 * Which activation widths have a bindable row, measured rather than derived.
 *
 * The derivation says a row binds when `row * width` is a multiple of the
 * device's storage-buffer alignment, and that every slot base is already a
 * multiple of it, so the condition collapses to the width alone. That is worth
 * checking against a driver: `graph_alignment` is the larger of a 256-byte
 * floor and two device limits, so a future adapter can only make it coarser,
 * never finer, and the widths that stop working would do so silently.
 */
static int probe_row_width(const char* route, int* on_device) {
    int32_t ids[SEQUENCE];
    size_t length = strlen(route);
    *on_device = 0;
    fill_ids(ids, 1);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    fill_ids(ids, 2);
    CHECK(volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    /* A refusal is a datum here, not a failure: it is the host-row fallback,
     * which is exactly what a width below the alignment is expected to get. */
    if (volvoxai_engine_forward_incremental_row(1) != 0) return 0;
    {
        const VxEngineState* state = vx_engine_state_current();
        int qlinear = 0;
        int device = 0;
        if (!state) return -1;
        for (int index = 0; index < state->node_count; index++) {
            const char* label = state->runtime_route_backend[index];
            if (strcmp(state->nodes[index].op, "QLinear")) continue;
            qlinear++;
            if (label && !strncmp(label, route, length)) device++;
        }
        *on_device = qlinear > 0 && device == qlinear;
    }
    return 0;
}

static int measure_row_width(VolvoxAIEngineBackend backend, const char* route,
                             int width, int* on_device) {
    int status;
    if (write_fixture(width, 1) != 0) return -1;
    if (vx_backend_manager_activate(backend) != 0) return 1;
    if (volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) {
        vx_backend_manager_deactivate();
        return -1;
    }
    status = probe_row_width(route, on_device);
    volvoxai_engine_shutdown();
    vx_backend_manager_deactivate();
    return status;
}

static int measure_alignment_boundary(VolvoxAIEngineBackend backend,
                                      const char* label, const char* route) {
    /* 768 is in the list because it is the one common decoder width that is not
     * a power of two; at 3 x 256 it should bind, and a rule stated as "powers
     * of two" rather than "multiples of the alignment" would get it wrong. */
    static const int widths[] = { 64, 128, 192, 256, 320, 512, 768 };
    int failures = 0;
    printf("# %s: int8 activation row stride vs the device binding alignment\n",
           label);
    for (size_t index = 0; index < sizeof(widths) / sizeof(widths[0]); index++) {
        int width = widths[index];
        int on_device = 0;
        int status = measure_row_width(backend, route, width, &on_device);
        if (status == 1) return 1;
        if (status != 0) return -1;
        printf("#   width %4d (row = %4d bytes): %s\n", width, width,
               on_device ? "device" : "host");
        /* Only one direction is asserted. A multiple of 256 must bind on any
         * conforming adapter; a non-multiple failing to bind is true of every
         * Vulkan device seen so far but is not a rule -- OpenGL's alignment is
         * the driver's SSBO offset limit with no 256-byte arena floor over it,
         * so finer widths binding there is the expected reading of this table,
         * not a surprise. */
        if (width % 256 == 0 && !on_device) {
            fprintf(stderr,
                    "FAIL %s width %d is a multiple of the 256-byte alignment "
                    "but its row did not bind\n", label, width);
            failures++;
        }
    }
    return failures ? -1 : 0;
}

/*
 * Several lanes of device rows in one step.
 *
 * A one-lane row is a single contiguous span, which is what a device window
 * binds. A batch's rows are `lane * S + position[lane]`, so they are S rows
 * apart and no single window names them -- the device path dispatches once per
 * contiguous *run* of them instead, which for ragged positions is one dispatch
 * per lane and for adjacent ones is fewer.
 *
 * Compared against the same batch on the CPU rather than against per-lane
 * decodes, because the CPU batch is already established as bit-identical to
 * those (`test_batched_row_decode`); this is the GPU-against-host comparison
 * the one-lane case above makes, at more than one lane.
 */
#define BATCH_LANES 3

static int run_batch_on(VolvoxAIEngineBackend backend, const char* label,
                        const char* expect_route, const int* positions,
                        int8_t rows[BATCH_LANES][WIDTH], int* handoff_seen,
                        int* routed) {
    const int is_cpu = backend == VOLVOXAI_BACKEND_CPU;
    static int32_t ids[BATCH_LANES * SEQUENCE];
    static int8_t y[BATCH_LANES * SEQUENCE * WIDTH];
    int status = -1;

    *handoff_seen = 0;
    *routed = 0;
    if (!is_cpu && vx_backend_manager_activate(backend) != 0) return 1;
    if (write_fixture(WIDTH, BATCH_LANES) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto deactivate;
    fill_batch_ids(ids, BATCH_LANES);
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto shutdown;
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0) goto shutdown;
    if (volvoxai_engine_forward_incremental_rows(positions, BATCH_LANES) != 0) {
        fprintf(stderr, "FAIL %s declined a %d-lane device row step\n",
                label, BATCH_LANES);
        goto shutdown;
    }
    if (vx_incremental_hybrid_row_active_locked()) *handoff_seen = 1;
    *routed = routes_match_expectation(expect_route, 1);
    if (volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) != 0) goto shutdown;
    for (int lane = 0; lane < BATCH_LANES; lane++)
        memcpy(rows[lane],
               y + ((size_t)lane * SEQUENCE + positions[lane]) * WIDTH, WIDTH);
    status = 0;
shutdown:
    volvoxai_engine_shutdown();
deactivate:
    if (!is_cpu) vx_backend_manager_deactivate();
    return status;
}

static int run_batched_device_rows(VolvoxAIEngineBackend backend,
                                   const char* label, const char* route,
                                   const int8_t cpu_rows[BATCH_LANES][WIDTH]) {
    /* Ragged on purpose: lanes at one position would leave the row formula
     * indistinguishable from several wrong ones. */
    const int positions[BATCH_LANES] = { 3, 5, 2 };
    static int8_t gpu_rows[BATCH_LANES][WIDTH];
    int gpu_handoff = 0;
    int gpu_routed = 0;
    int status;

    status = run_batch_on(backend, label, route, positions, gpu_rows,
                          &gpu_handoff, &gpu_routed);
    if (status == 1) return 1;
    if (status != 0) return -1;
    if (gpu_handoff) {
        fprintf(stderr, "FAIL %s relinquished the batch to the host\n", label);
        return -1;
    }
    if (!gpu_routed) return -1;
    for (int lane = 0; lane < BATCH_LANES; lane++) {
        if (memcmp(cpu_rows[lane], gpu_rows[lane], WIDTH) == 0) continue;
        fprintf(stderr, "FAIL batched lane %d differs between %s and CPU\n",
                lane, label);
        return -1;
    }
    /* Two lanes agreeing would mean the batch read one lane for both. */
    if (memcmp(gpu_rows[0], gpu_rows[1], WIDTH) == 0) {
        fprintf(stderr,
                "FAIL batched lanes 0 and 1 wrote identical rows on %s\n",
                label);
        return -1;
    }
    return 0;
}

static int run_cpu_batch(int8_t cpu_rows[BATCH_LANES][WIDTH]) {
    const int positions[BATCH_LANES] = { 3, 5, 2 };
    int handoff = 0;
    int routed = 0;
    return run_batch_on(VOLVOXAI_BACKEND_CPU, "CPU", "cpu", positions, cpu_rows,
                        &handoff, &routed) == 0 ? 0 : -1;
}

/* One backend's whole case. Returns 1 when the device is absent, which the
 * caller turns into a skip rather than a pass. */
static int run_device_backend(VolvoxAIEngineBackend backend, const char* label,
                              const char* route,
                              const int8_t cpu_rows[STEPS][WIDTH],
                              const int8_t cpu_batch[BATCH_LANES][WIDTH]) {
    int8_t gpu_rows[STEPS][WIDTH];
    int gpu_handoff = 0;
    int status;
    /* Every stage below rewrites the fixture for its own lane count and width,
     * so each one restates what it needs rather than inheriting the last. */
    if (write_fixture(WIDTH, 1) != 0) return -1;
    status = run_decode(backend, label, route, gpu_rows, &gpu_handoff);
    if (status == 1) return 1;
    if (status != 0) return -1;

    /* The claim that separates this from the path it replaces. */
    if (gpu_handoff) {
        fprintf(stderr,
                "FAIL %s relinquished the prefix to the host; the rows ran "
                "on the CPU, so the comparison below proves nothing.\n", label);
        return -1;
    }
    printf("ok %s executed every decode row without a host handoff\n", label);
    printf("ok every node of every row step ran on %s "
           "(QEmbedding included; see expected_route)\n", label);

    for (int index = 0; index < STEPS; index++) {
        if (memcmp(cpu_rows[index], gpu_rows[index], WIDTH) != 0) {
            int differing = 0;
            fprintf(stderr, "FAIL %s step %d differs at:", label, index);
            for (int c = 0; c < WIDTH; c++) {
                if (cpu_rows[index][c] == gpu_rows[index][c]) continue;
                /* A 256-wide dump buries the signal. The first handful of
                 * disagreeing channels says whether this is one bad lane or a
                 * whole row of noise, which is the distinction worth making. */
                if (differing < 8)
                    fprintf(stderr, " [%d] cpu=%d gpu=%d", c,
                            cpu_rows[index][c], gpu_rows[index][c]);
                differing++;
            }
            fprintf(stderr, "\n  %d of %d channels differ\n", differing, WIDTH);
            return -1;
        }
    }
    printf("ok every %s decode row is bit-identical to the CPU row\n", label);

    status = run_batched_device_rows(backend, label, route, cpu_batch);
    if (status == 1) return 1;
    if (status != 0) return -1;
    printf("ok a %d-lane %s device row step matches the same batch on the CPU\n",
           BATCH_LANES, label);

    status = measure_alignment_boundary(backend, label, route);
    if (status == 1) return 1;
    if (status != 0) return -1;
    printf("ok every %s width that is a multiple of the alignment binds its "
           "row\n", label);
    return 0;
}

static int run_all(void) {
    static const struct {
        VolvoxAIEngineBackend backend;
        const char* label;
        const char* route;
    } devices[] = {
        { VOLVOXAI_BACKEND_VULKAN, "Vulkan", "vulkan" },
        { VOLVOXAI_BACKEND_OPENGL, "OpenGL", "opengl" },
    };
    int8_t cpu_rows[STEPS][WIDTH];
    static int8_t cpu_batch[BATCH_LANES][WIDTH];
    int cpu_handoff = 0;
    int ran = 0;

    if (write_fixture(WIDTH, 1) != 0) return 1;
    if (run_decode(VOLVOXAI_BACKEND_CPU, "CPU", "cpu", cpu_rows,
                   &cpu_handoff) != 0) return 1;
    printf("ok CPU row decode produced %d rows\n", STEPS);

    /* A comparison of two constant columns would pass without describing
     * anything. Saturation and an all-zero closure both look like this, and
     * both would make the equalities below vacuous. */
    for (int index = 0; index < STEPS; index++) {
        int varies = 0;
        for (int c = 1; c < WIDTH; c++)
            if (cpu_rows[index][c] != cpu_rows[index][0]) varies = 1;
        if (!varies) {
            fprintf(stderr,
                    "FAIL step %d produced a constant row (%d); the equality "
                    "below would hold for reasons unrelated to the device.\n",
                    index, cpu_rows[index][0]);
            return 1;
        }
    }

    if (run_cpu_batch(cpu_batch) != 0) return 1;

    for (size_t index = 0; index < sizeof(devices) / sizeof(devices[0]); index++) {
        int status = run_device_backend(devices[index].backend,
                                        devices[index].label,
                                        devices[index].route,
                                        (const int8_t (*)[WIDTH])cpu_rows,
                                        (const int8_t (*)[WIDTH])cpu_batch);
        if (status == 1) {
            printf("SKIP no %s device\n", devices[index].label);
            continue;
        }
        if (status != 0) return 1;
        ran++;
    }
    /* Skipping every device would otherwise report a pass for a run that
     * checked nothing on a GPU. */
    if (!ran) return 77;

    printf("PASS device row decode\n");
    return 0;
}

int main(void) {
    /* Engine state is per context and explicitly scoped; every engine entry
     * point reads it through the thread-local current pointer, so it has to
     * exist before the first call. */
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
