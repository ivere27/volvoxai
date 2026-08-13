/*
 * F2's exit criterion on native CPU: private paged KV is numerically identical
 * to contiguous KV.
 *
 * Identical, not close. The paged form and the contiguous form are one
 * implementation reading the same bytes in the same order through different
 * addresses, so the accumulation order per head is unchanged and the results
 * must compare bit-for-bit. Accepting a tolerance here would hide exactly the
 * kind of reordering that later turns into a silent decode divergence.
 */

#include "paged_attention.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D_MODEL 8
#define HEADS 2
#define HEAD_DIM 4
#define PAGE_TOKENS 2
#define KV_LENGTH 7
#define POOL_SLOTS 32

static int g_failures;

static void fail(const char* message) {
    fprintf(stderr, "FAIL %s\n", message);
    g_failures++;
}

static float sample(int slot, int channel, int seed) {
    return (float)(((slot * 13 + channel * 7 + seed) % 19) - 9) * 0.125f;
}

/* Scatter logical position p onto physical slot `pages[p / PAGE_TOKENS]`, and
 * build the contiguous twin holding the same tokens in logical order. */
static void build(const int* pages, int page_count,
                  float* pool_key, float* pool_value,
                  float* dense_key, float* dense_value) {
    memset(pool_key, 0, (size_t)POOL_SLOTS * D_MODEL * sizeof(float));
    memset(pool_value, 0, (size_t)POOL_SLOTS * D_MODEL * sizeof(float));
    for (int position = 0; position < KV_LENGTH; position++) {
        int page = pages[position / PAGE_TOKENS];
        long slot = (long)page * PAGE_TOKENS + (position % PAGE_TOKENS);
        for (int channel = 0; channel < D_MODEL; channel++) {
            float key = sample(position, channel, 1);
            float value = sample(position, channel, 5);
            pool_key[slot * D_MODEL + channel] = key;
            pool_value[slot * D_MODEL + channel] = value;
            dense_key[(long)position * D_MODEL + channel] = key;
            dense_value[(long)position * D_MODEL + channel] = value;
        }
    }
    (void)page_count;
}

static int compare(const char* label, const float* left, const float* right) {
    for (int channel = 0; channel < D_MODEL; channel++) {
        if (left[channel] == right[channel]) continue;
        char message[192];
        snprintf(message, sizeof(message),
                 "%s: channel %d differs (%.9g vs %.9g)",
                 label, channel, (double)left[channel], (double)right[channel]);
        fail(message);
        return -1;
    }
    return 0;
}

static void test_mapping(const char* label, const int* pages, int page_count) {
    float* pool_key = (float*)malloc((size_t)POOL_SLOTS * D_MODEL * sizeof(float));
    float* pool_value = (float*)malloc((size_t)POOL_SLOTS * D_MODEL * sizeof(float));
    float dense_key[KV_LENGTH * D_MODEL];
    float dense_value[KV_LENGTH * D_MODEL];
    float query[D_MODEL];
    float paged_out[D_MODEL];
    float dense_out[D_MODEL];
    const float scale = 0.5f;

    if (!pool_key || !pool_value) { fail("allocation"); free(pool_key); free(pool_value); return; }
    for (int channel = 0; channel < D_MODEL; channel++) query[channel] = sample(3, channel, 11);
    build(pages, page_count, pool_key, pool_value, dense_key, dense_value);

    if (vx_sdpa_paged_row_f32(query, pool_key, pool_value, pages, PAGE_TOKENS,
                              KV_LENGTH, NULL, paged_out, D_MODEL, HEADS, HEAD_DIM,
                              scale) != 0) {
        fail("paged attention refused a valid mapping");
    } else if (vx_sdpa_paged_row_f32(query, dense_key, dense_value, NULL, 1,
                                     KV_LENGTH, NULL, dense_out, D_MODEL, HEADS, HEAD_DIM,
                                     scale) != 0) {
        fail("contiguous attention refused the identity mapping");
    } else if (compare(label, paged_out, dense_out) == 0) {
        printf("ok %s\n", label);
    }
    free(pool_key);
    free(pool_value);
}

static void test_unmapped_page_is_refused(void) {
    float pool[POOL_SLOTS * D_MODEL];
    float query[D_MODEL];
    float out[D_MODEL];
    /* A canary the kernel must not overwrite: an unmapped page has to fail the
     * whole call, not leave a partially attended row behind. */
    const int pages[4] = { 0, -1, 2, 3 };
    for (int index = 0; index < POOL_SLOTS * D_MODEL; index++) pool[index] = 1.0f;
    for (int channel = 0; channel < D_MODEL; channel++) {
        query[channel] = 1.0f;
        out[channel] = -12345.0f;
    }
    if (vx_sdpa_paged_row_f32(query, pool, pool, pages, PAGE_TOKENS, KV_LENGTH,
                              NULL, out, D_MODEL, HEADS, HEAD_DIM, 0.5f) == 0) {
        fail("an unmapped page must be refused");
        return;
    }
    for (int channel = 0; channel < D_MODEL; channel++) {
        if (out[channel] != -12345.0f) {
            fail("a refused call must not write its output");
            return;
        }
    }
    printf("ok an unmapped page is refused without writing the output\n");
}

static void test_slot_resolution(void) {
    const int pages[4] = { 5, 1, 9, 0 };
    if (vx_paged_attention_slot(NULL, 1, 6) != 6) {
        fail("the identity mapping must resolve position to itself");
        return;
    }
    if (vx_paged_attention_slot(pages, PAGE_TOKENS, 0) != 10 ||
        vx_paged_attention_slot(pages, PAGE_TOKENS, 1) != 11 ||
        vx_paged_attention_slot(pages, PAGE_TOKENS, 2) != 2 ||
        vx_paged_attention_slot(pages, PAGE_TOKENS, 5) != 19) {
        fail("slot resolution");
        return;
    }
    printf("ok slot resolution matches page * pageTokens + offset\n");
}

int main(void) {
    /* The identity map, a shifted run, a reversed mapping, and the interleaving
     * two concurrently decoding requests actually produce. */
    const int identity[4] = { 0, 1, 2, 3 };
    const int shifted[4] = { 4, 5, 6, 7 };
    const int reversed[4] = { 7, 6, 5, 4 };
    const int interleaved[4] = { 0, 2, 4, 6 };

    test_slot_resolution();
    test_mapping("identity mapping matches contiguous", identity, 4);
    test_mapping("shifted run matches contiguous", shifted, 4);
    test_mapping("reversed mapping matches contiguous", reversed, 4);
    test_mapping("interleaved mapping matches contiguous", interleaved, 4);
    test_unmapped_page_is_refused();

    if (g_failures) {
        fprintf(stderr, "FAIL %d paged attention check(s)\n", g_failures);
        return 1;
    }
    printf("PASS paged attention\n");
    return 0;
}
