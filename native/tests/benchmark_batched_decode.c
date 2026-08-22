/*
 * What batching a decode step is actually worth.
 *
 * The correctness test proves each lane is bit-identical to its own one-lane
 * decode, and the scheduler test proves the round becomes one dispatch instead
 * of several. Neither says whether that is faster, and the batched path is not
 * free: it copies each lane's row into a staging buffer and copies the result
 * back out, which a one-lane decode never does.
 *
 * The comparison holds the work constant and varies only how it is dispatched:
 *
 *   batched     one `[B,S,W]` engine, one forward_incremental_rows per step
 *   sequential  one `[1,S,W]` engine, B calls to forward_incremental_row
 *
 * Both advance B lane-steps and perform the same arithmetic -- B rows of each
 * projection, B attention queries over the same key extents. What differs is
 * one kernel call with M=B and two staging copies against B kernel calls with
 * M=1 and none.
 *
 * The sequential side runs its B steps through a single engine rather than B
 * engines. Its K/V therefore belongs to one sequence rather than to B separate
 * ones, which would matter for a correctness claim and does not for a timing
 * one: an int8 GEMM does the same multiply-accumulates whatever the bytes are,
 * and attention visits a key count fixed by `position`, which is the same on
 * both sides. Correctness is `test_batched_row_decode`'s job.
 *
 * Reported per lane-step, because that is the unit a scheduler serves.
 */

#define _POSIX_C_SOURCE 200809L

#include "engine_core.h"
#include "engine_internal.h"
#include "safetensors.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LANES 8
#define SEQUENCE 256
#define VOCAB 256
/* The portable attention kernel refuses head_dim above 64 (`accumulator[64]`),
 * so the head count follows the width rather than being fixed -- otherwise a
 * 1024-wide model asks for head_dim 128 and the whole graph is refused. */
#define HEAD_DIM 64
#define WARMUP 3
#define ITERATIONS 40
/* Best of several passes. A decode step at these sizes is a fraction of a
 * millisecond, so one pass measures the scheduler as much as the kernel; the
 * minimum is the run least interrupted. */
#define REPEATS 5

static const char* GRAPH_PATH = "/tmp/volvox-batched-decode-bench-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-batched-decode-bench.safetensors";

static double elapsed_ms(const struct timespec* start,
                         const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
        (double)(end->tv_nsec - start->tv_nsec) / 1000000.0;
}

/* Dense, unlike the correctness fixture's sparse weights: saturation does not
 * change how long a multiply-accumulate takes, and a benchmark that skipped
 * seven eighths of the columns would be measuring a model nobody runs. */
static int8_t weight_value(int row, int column, int seed) {
    return (int8_t)(((row * 7 + column * 13 + seed * 5) % 15) - 7);
}

static int write_graph_document(int lanes, int width) {
    char graph[4096];
    FILE* handle;
    int written = snprintf(graph, sizeof(graph),
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
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"y_scale\","
        "\"zero_point_tensor\":\"y_zero\"},"
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
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"qa\",\"k\":\"ka\","
        "\"v\":\"va\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[%d,%d,%d]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"heads\":%d,\"causal\":true,\"scale\":%.8f}}],"
        "\"outputs\":[\"y\"]}",
        lanes, SEQUENCE,
        lanes, SEQUENCE, width, lanes, SEQUENCE, width, lanes, SEQUENCE, width,
        lanes, SEQUENCE, width, lanes, SEQUENCE, width,
        width / HEAD_DIM, 1.0 / 8.0);
    if (written <= 0 || (size_t)written >= sizeof(graph)) return -1;
    handle = fopen(GRAPH_PATH, "wb");
    if (!handle) return -1;
    if (fwrite(graph, 1, (size_t)written, handle) != (size_t)written) {
        fclose(handle);
        return -1;
    }
    return fclose(handle) == 0 ? 0 : -1;
}

static int write_per_axis_pair(SafetensorsFile* file, const char* name,
                               int count, float* scales, int8_t* zeros) {
    const int axis_shape[1] = { count };
    char scale_name[32];
    char zero_name[32];
    for (int index = 0; index < count; index++) {
        scales[index] = 0.03125f;
        zeros[index] = 0;
    }
    snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
    snprintf(zero_name, sizeof(zero_name), "%s_zero", name);
    if (safetensors_add_tensor(file, scale_name, SAFETENSORS_DTYPE_F32,
                               axis_shape, 1, scales,
                               (size_t)count * sizeof(float)) != 0) return -1;
    return safetensors_add_tensor(file, zero_name, SAFETENSORS_DTYPE_I8,
                                  axis_shape, 1, zeros, (size_t)count) == 0
        ? 0 : -1;
}

static int write_weights(int width) {
    SafetensorsFile file;
    const int weight_shape[2] = { width, width };
    const int table_shape[2] = { VOCAB, width };
    const int bias_shape[1] = { width };
    const int scalar_shape[1] = { 1 };
    int8_t* emb = (int8_t*)malloc((size_t)VOCAB * width);
    int8_t* projection = (int8_t*)malloc((size_t)width * width);
    int32_t* bias = (int32_t*)calloc((size_t)width, sizeof(int32_t));
    const int axis_max = VOCAB > width ? VOCAB : width;
    float* scales = (float*)malloc((size_t)axis_max * sizeof(float));
    int8_t* zeros = (int8_t*)malloc((size_t)axis_max);
    float scale_tensor[1] = { 0.03125f };
    int8_t zero_tensor[1] = { 0 };
    int status = -1;
    static const char* const projections[3] = { "wq", "wk", "wv" };
    static const char* const biases[3] = { "bq", "bk", "bv" };
    static const char* const activations[5] = { "x", "qa", "ka", "va", "y" };

    if (!emb || !projection || !bias || !scales || !zeros) goto done;
    for (int row = 0; row < VOCAB; row++)
        for (int column = 0; column < width; column++)
            emb[row * width + column] = weight_value(row, column, 1);
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) goto done;
    if (safetensors_add_tensor(&file, "emb", SAFETENSORS_DTYPE_I8, table_shape, 2,
                               emb, (size_t)VOCAB * width) != 0) goto free_file;
    for (int which = 0; which < 3; which++) {
        for (int row = 0; row < width; row++)
            for (int column = 0; column < width; column++)
                projection[row * width + column] =
                    weight_value(row, column, 3 + which * 2);
        if (safetensors_add_tensor(&file, projections[which], SAFETENSORS_DTYPE_I8,
                                   weight_shape, 2, projection,
                                   (size_t)width * width) != 0) goto free_file;
        if (safetensors_add_tensor(&file, biases[which], SAFETENSORS_DTYPE_I32,
                                   bias_shape, 1, bias,
                                   (size_t)width * sizeof(int32_t)) != 0) goto free_file;
    }
    for (int index = 0; index < 5; index++) {
        char scale_name[32];
        char zero_name[32];
        snprintf(scale_name, sizeof(scale_name), "%s_scale", activations[index]);
        snprintf(zero_name, sizeof(zero_name), "%s_zero", activations[index]);
        if (safetensors_add_tensor(&file, scale_name, SAFETENSORS_DTYPE_F32,
                                   scalar_shape, 1, scale_tensor,
                                   sizeof(scale_tensor)) != 0) goto free_file;
        if (safetensors_add_tensor(&file, zero_name, SAFETENSORS_DTYPE_I8,
                                   scalar_shape, 1, zero_tensor,
                                   sizeof(zero_tensor)) != 0) goto free_file;
    }
    /* emb is [VOCAB, width], so its per-axis extent is VOCAB, not width. The
     * scratch is sized for the larger of the two and reused. */
    if (write_per_axis_pair(&file, "emb", VOCAB, scales, zeros) != 0) goto free_file;
    for (int which = 0; which < 3; which++)
        if (write_per_axis_pair(&file, projections[which], width,
                                scales, zeros) != 0) goto free_file;
    status = safetensors_save(WEIGHTS_PATH, &file) == 0 ? 0 : -1;
free_file:
    safetensors_free(&file);
done:
    free(emb);
    free(projection);
    free(bias);
    free(scales);
    free(zeros);
    return status;
}

/* Positions spread so lanes carry different key extents, which is the ragged
 * case a scheduler actually produces. */
static void fill_positions(int* positions, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        positions[lane] = SEQUENCE / 2 + lane * 8;
}

static void fill_ids(int32_t* ids, int lanes) {
    for (int lane = 0; lane < lanes; lane++)
        for (int token = 0; token < SEQUENCE; token++)
            ids[lane * SEQUENCE + token] = (lane * 7 + token * 5 + 3) % VOCAB;
}

/* Returns milliseconds per lane-step, or a negative value on failure. */
static double time_batched(int lanes, int width) {
    int32_t* ids = (int32_t*)malloc((size_t)lanes * SEQUENCE * sizeof(int32_t));
    int positions[MAX_LANES];
    struct timespec start;
    struct timespec end;
    double result = -1.0;
    if (!ids) return -1.0;
    fill_ids(ids, lanes);
    fill_positions(positions, lanes);
    if (write_graph_document(lanes, width) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) goto done;
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids,
                                      (size_t)lanes * SEQUENCE * sizeof(int32_t)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto shutdown;
    for (int iteration = 0; iteration < WARMUP; iteration++) {
        /* The input is re-submitted every step because a successful row clears
         * the dirty set; without it the next call would find nothing to run and
         * time an empty graph. */
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids,
                                          (size_t)lanes * SEQUENCE * sizeof(int32_t)) != 0 ||
            volvoxai_engine_forward_incremental_rows(positions, lanes) != 0) goto shutdown;
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids,
                                          (size_t)lanes * SEQUENCE * sizeof(int32_t)) != 0 ||
            volvoxai_engine_forward_incremental_rows(positions, lanes) != 0) goto shutdown;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    result = elapsed_ms(&start, &end) / (double)ITERATIONS / (double)lanes;
shutdown:
    volvoxai_engine_shutdown();
done:
    free(ids);
    return result;
}

static double time_sequential(int lanes, int width) {
    int32_t ids[SEQUENCE];
    int positions[MAX_LANES];
    struct timespec start;
    struct timespec end;
    double result = -1.0;
    fill_ids(ids, 1);
    fill_positions(positions, lanes);
    if (write_graph_document(1, width) != 0 ||
        volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) != 0) return -1.0;
    if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) != 0 ||
        volvoxai_engine_forward_incremental() != 0) goto shutdown;
    for (int iteration = 0; iteration < WARMUP; iteration++)
        for (int lane = 0; lane < lanes; lane++)
            if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                              ids, sizeof(ids)) != 0 ||
                volvoxai_engine_forward_incremental_row(positions[lane]) != 0) goto shutdown;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int iteration = 0; iteration < ITERATIONS; iteration++)
        for (int lane = 0; lane < lanes; lane++)
            if (volvoxai_engine_set_input_raw("ids", VOLVOXAI_DTYPE_I32,
                                              ids, sizeof(ids)) != 0 ||
                volvoxai_engine_forward_incremental_row(positions[lane]) != 0) goto shutdown;
    clock_gettime(CLOCK_MONOTONIC, &end);
    result = elapsed_ms(&start, &end) / (double)ITERATIONS / (double)lanes;
shutdown:
    volvoxai_engine_shutdown();
    return result;
}

static double best_of(double (*measure)(int, int), int lanes, int width) {
    double best = -1.0;
    for (int repeat = 0; repeat < REPEATS; repeat++) {
        const double sample = measure(lanes, width);
        if (sample < 0.0) return sample;
        if (best < 0.0 || sample < best) best = sample;
    }
    return best;
}

static int run_width(int width) {
    printf("\n  width %d, sequence %d, heads %d (head_dim %d)\n",
           width, SEQUENCE, width / HEAD_DIM, HEAD_DIM);
    printf("  %-6s %14s %14s %10s\n",
           "lanes", "batched ms/step", "seq ms/step", "speedup");
    if (write_weights(width) != 0) {
        fprintf(stderr, "cannot write weights for width %d\n", width);
        return -1;
    }
    for (int lanes = 1; lanes <= MAX_LANES; lanes *= 2) {
        double batched = best_of(time_batched, lanes, width);
        double sequential = best_of(time_sequential, lanes, width);
        if (batched < 0.0 || sequential < 0.0) {
            fprintf(stderr, "  %-6d measurement failed\n", lanes);
            return -1;
        }
        printf("  %-6d %14.4f %14.4f %9.2fx\n",
               lanes, batched, sequential, sequential / batched);
    }
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    int status = 0;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    setenv("VOLVOX_ARENA", "0", 1);
    printf("batched decode: one dispatch of B lanes vs B dispatches of one\n");
    if (run_width(512) != 0 || run_width(1024) != 0) status = 1;
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    remove(GRAPH_PATH);
    remove(WEIGHTS_PATH);
    return status;
}
