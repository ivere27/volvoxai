/*
 * Paged KV through the native engine, end to end.
 *
 * `test_paged_attention.c` proves the kernel; this proves the *plumbing*: that
 * a bound page table reaches both halves of a decode step. The two halves fail
 * differently and a test that only checked one would miss the other:
 *
 *   read  — attention gathers the lane's active prefix in logical order from
 *           whatever physical slots the table names.
 *   write — the K/V projection writes its row into the mapped slot rather than
 *           at `position * width`.
 *
 * A mapping is only interesting if it is scattered, so the lane here holds
 * logical pages 0,1,2,3 at physical pages 0,2,4,6 — what a scheduler running
 * two requests concurrently produces. The oracle is the identical graph decoded
 * with no binding at all.
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

#define SEQUENCE 8
#define WIDTH 4
#define HEADS 2

static const char* GRAPH_PATH = "/tmp/volvox-paged-decode-graph.json";
static const char* WEIGHTS_PATH = "/tmp/volvox-paged-decode-weights.safetensors";

/* Flat on purpose: every node between a projection and attention would be
 * another operator that must learn page-table addressing, and the row path
 * refuses those rather than addressing them with linear arithmetic. */
static const char* GRAPH =
    "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
    "\"x\":{\"shape\":[1,8,4],\"dtype\":\"float32\"},"
    "\"keep\":{\"shape\":[1,8],\"dtype\":\"int32\"}},"
    "\"nodes\":["
    "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wq\"},"
    "\"outputs\":{\"out\":\"q\"},\"outputs_shape\":{\"out\":[1,8,4]},"
    "\"params\":{\"weight_layout\":\"dout_din\"}},"
    "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wk\"},"
    "\"outputs\":{\"out\":\"self.k\"},\"outputs_shape\":{\"out\":[1,8,4]},"
    "\"params\":{\"weight_layout\":\"dout_din\"}},"
    "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"wv\"},"
    "\"outputs\":{\"out\":\"self.v\"},\"outputs_shape\":{\"out\":[1,8,4]},"
    "\"params\":{\"weight_layout\":\"dout_din\"}},"
    "{\"opType\":\"CrossSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"self.k\","
    "\"v\":\"self.v\",\"mask\":\"keep\"},"
    "\"outputs\":{\"out\":\"attended\"},\"outputs_shape\":{\"out\":[1,8,4]},"
    "\"params\":{\"heads\":2,\"causal\":true,\"scale\":0.5}}],"
    "\"outputs\":[\"attended\",\"self.k\",\"self.v\"]}";

static const char* PAGED_TENSORS[2] = { "self.k", "self.v" };

static float weight_value(int row, int column, int seed) {
    return (float)(((row * 5 + column * 3 + seed) % 11) - 5) * 0.0625f;
}

static int write_fixture(void) {
    SafetensorsFile file;
    const int shape[2] = { WIDTH, WIDTH };
    float wq[WIDTH * WIDTH];
    float wk[WIDTH * WIDTH];
    float wv[WIDTH * WIDTH];
    FILE* handle;

    for (int row = 0; row < WIDTH; row++) {
        for (int column = 0; column < WIDTH; column++) {
            wq[row * WIDTH + column] = weight_value(row, column, 1);
            wk[row * WIDTH + column] = weight_value(row, column, 4);
            wv[row * WIDTH + column] = weight_value(row, column, 7);
        }
    }
    handle = fopen(GRAPH_PATH, "wb");
    CHECK(handle != NULL);
    CHECK(fwrite(GRAPH, 1, strlen(GRAPH), handle) == strlen(GRAPH));
    CHECK(fclose(handle) == 0);

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

static void fill_inputs(float* x, int32_t* keep, int active) {
    for (int token = 0; token < SEQUENCE; token++) {
        keep[token] = token < active ? 1 : 0;
        for (int channel = 0; channel < WIDTH; channel++) {
            x[token * WIDTH + channel] =
                (float)(((token * 7 + channel * 3) % 13) - 6) * 0.125f;
        }
    }
}

/*
 * One decode: seed, then three row steps.
 *
 * `cache` non-NULL binds a scattered mapping and advances it in step with the
 * decode; NULL is the contiguous oracle. `attended_rows` receives the attention
 * output of each step, which is where a wrong gather shows up, and `slots`
 * receives the physical slot each logical row was written to, which is where a
 * wrong write shows up.
 */
static int run_decode(VxPagedKVCache* cache, float attended_rows[3][WIDTH],
                      int slots[3], float key_pool[SEQUENCE * WIDTH]) {
    const int steps[3] = { 1, 2, 3 };
    float x[SEQUENCE * WIDTH];
    int32_t keep[SEQUENCE];

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);

    fill_inputs(x, keep, 1);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("keep", VOLVOXAI_DTYPE_I32, keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);
    /* The seed is a full forward, so it writes the retained K/V linearly. The
     * cache has to agree: logical page zero of the decoding lane is physical
     * page zero, which is what claiming it first gives. */
    if (cache) CHECK(vx_paged_kv_append(cache, 0, 1) == VX_PAGED_KV_OK);

    for (int index = 0; index < 3; index++) {
        int position = steps[index];
        fill_inputs(x, keep, position + 1);
        if (cache) {
            /* A second lane claiming a page between this lane's tokens is what
             * scatters the mapping. */
            CHECK(vx_paged_kv_append(cache, 1, 1) == VX_PAGED_KV_OK);
            CHECK(vx_paged_kv_append(cache, 0, 1) == VX_PAGED_KV_OK);
            CHECK(vx_paged_bind_locked(cache, 0, PAGED_TENSORS, 2) == 0);
        }
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) == 0);
        CHECK(volvoxai_engine_set_input_raw("keep", VOLVOXAI_DTYPE_I32, keep,
                                            sizeof(keep)) == 0);
        CHECK(volvoxai_engine_forward_incremental_row(position) == 0);
        if (cache) {
            long slot = -1;
            CHECK(vx_paged_kv_physical_token_index(cache, 0, position, &slot) ==
                  VX_PAGED_KV_OK);
            slots[index] = (int)slot;
        } else {
            slots[index] = position;
        }
        {
            float attended[SEQUENCE * WIDTH];
            CHECK(volvoxai_engine_copy_tensor_f32("attended", attended,
                                                  SEQUENCE * WIDTH) == 0);
            memcpy(attended_rows[index], attended + (size_t)position * WIDTH,
                   sizeof(float) * WIDTH);
        }
    }
    CHECK(volvoxai_engine_copy_tensor_f32("self.k", key_pool, SEQUENCE * WIDTH) == 0);
    /* Deliberately no explicit unbind: shutdown must drop the binding and its
     * gather buffers on its own, because a binding names tensors that shutdown
     * frees. The next run_decode rebinding cleanly is the check. */
    volvoxai_engine_shutdown();
    CHECK(vx_paged_bound_locked() == 0);
    return 0;
}

static int test_paged_decode_matches_contiguous(void) {
    VxPagedKVOptions options;
    VxPagedKVCache* cache;
    float contiguous_attended[3][WIDTH];
    float paged_attended[3][WIDTH];
    float contiguous_keys[SEQUENCE * WIDTH];
    float paged_keys[SEQUENCE * WIDTH];
    int contiguous_slots[3];
    int paged_slots[3];
    const int expected_slots[3] = { 2, 4, 6 };

    CHECK(run_decode(NULL, contiguous_attended, contiguous_slots,
                     contiguous_keys) == 0);

    memset(&options, 0, sizeof(options));
    options.lanes = 2;
    options.page_tokens = 1;
    options.lane_token_capacity = SEQUENCE / 2;
    options.max_pages = SEQUENCE;
    options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache = vx_paged_kv_create(&options);
    CHECK(cache != NULL);
    CHECK(run_decode(cache, paged_attended, paged_slots, paged_keys) == 0);

    /* The mapping actually scattered. A run that quietly fell back to the
     * identity map would pass everything below while proving nothing. */
    for (int index = 0; index < 3; index++) {
        CHECK(paged_slots[index] == expected_slots[index]);
    }

    for (int index = 0; index < 3; index++) {
        /* The write half: logical row N of the lane lives in the slot its page
         * table names, holding exactly what the contiguous run put at row N. */
        const float* paged_row = paged_keys + (size_t)paged_slots[index] * WIDTH;
        const float* dense_row = contiguous_keys + (size_t)(index + 1) * WIDTH;
        for (int channel = 0; channel < WIDTH; channel++) {
            CHECK(paged_row[channel] == dense_row[channel]);
        }
        /* The read half, and the one that matters: attention output is not
         * paged, so it stays at `position * width` -- but its value can only be
         * right if the gather read the mapped slots in logical order. */
        for (int channel = 0; channel < WIDTH; channel++) {
            CHECK(paged_attended[index][channel] == contiguous_attended[index][channel]);
        }
    }
    vx_paged_kv_destroy(cache);
    printf("ok paged decode is identical to contiguous decode\n");
    return 0;
}

/*
 * A paged tensor read outside attention must be refused, not addressed with
 * linear arithmetic. Widening the paged set is how this fires in practice, and
 * the failure has to be a refusal: an operator reading `position * width` from
 * the page pool produces a decoder that is wrong and looks right.
 */
static int test_paged_domain_refuses_a_foreign_reader(void) {
    /* `x` feeds three projections and attention reads it on no port, so
     * declaring it paged must be refused. */
    const char* over_wide[3] = { "self.k", "self.v", "x" };
    VxPagedKVOptions options;
    VxPagedKVCache* cache;
    float x[SEQUENCE * WIDTH];
    int32_t keep[SEQUENCE];

    memset(&options, 0, sizeof(options));
    options.lanes = 1;
    options.page_tokens = 1;
    options.lane_token_capacity = SEQUENCE;
    options.max_pages = SEQUENCE;
    options.policy = VX_PAGED_KV_POLICY_PAGED;
    cache = vx_paged_kv_create(&options);
    CHECK(cache != NULL);

    CHECK(setenv("VOLVOX_ARENA", "0", 1) == 0);
    CHECK(volvoxai_engine_init(GRAPH_PATH, WEIGHTS_PATH) == 0);
    fill_inputs(x, keep, 1);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("keep", VOLVOXAI_DTYPE_I32, keep, sizeof(keep)) == 0);
    CHECK(volvoxai_engine_forward_incremental() == 0);

    CHECK(vx_paged_bind_locked(cache, 0, PAGED_TENSORS, 2) == 0);
    CHECK(vx_paged_domain_supported_locked(NULL) == 1);
    CHECK(vx_paged_bind_locked(cache, 0, over_wide, 3) == 0);
    CHECK(vx_paged_domain_supported_locked(NULL) == 0);
    vx_paged_unbind_locked();

    volvoxai_engine_shutdown();
    vx_paged_kv_destroy(cache);
    printf("ok a paged tensor read outside attention is refused\n");
    return 0;
}

int main(void) {
    /* Engine state is per context and explicitly scoped; every engine entry
     * point reads it through the thread-local current pointer, so it has to
     * exist before the first call. */
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    int failed = 0;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    if (write_fixture() != 0) failed = 1;
    if (!failed && test_paged_decode_matches_contiguous() != 0) failed = 1;
    if (!failed && test_paged_domain_refuses_a_foreign_reader() != 0) failed = 1;
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    if (failed) return 1;
    remove(GRAPH_PATH);
    remove(WEIGHTS_PATH);
    printf("PASS paged decode\n");
    return 0;
}
