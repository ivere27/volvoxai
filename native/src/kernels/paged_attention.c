#include "paged_attention.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

long vx_paged_attention_slot(const int* page_table, int page_tokens, int position) {
    int page;
    if (position < 0) return -1;
    /* The identity mapping. Written as the absence of a table rather than as a
     * table the caller has to materialise, so a contiguous cache costs nothing
     * to address. */
    if (!page_table) return position;
    if (page_tokens <= 0) return -1;
    page = page_table[position / page_tokens];
    if (page < 0) return -1;
    return (long)page * page_tokens + (position % page_tokens);
}

int vx_sdpa_paged_row_f32(const float* q,
                          const float* key_pool, const float* value_pool,
                          const int* page_table, int page_tokens,
                          int kv_length, const int* keep_mask, float* out,
                          int d_model, int heads, int head_dim, float scale) {
    long* slots;
    float* scores;
    int visible = 0;
    if (!q || !key_pool || !value_pool || !out || kv_length <= 0 || d_model <= 0 ||
        heads <= 0 || head_dim <= 0 || heads * head_dim > d_model ||
        (size_t)kv_length > SIZE_MAX / sizeof(long)) return -1;

    /* Resolve the whole mapping before touching `out`: an unmapped page must
     * fail the call, not leave a half-written row behind. */
    slots = (long*)malloc((size_t)kv_length * sizeof(long));
    if (!slots) return -1;
    for (int position = 0; position < kv_length; position++) {
        long slot;
        if (keep_mask && !keep_mask[position]) { slots[position] = -1; continue; }
        slot = vx_paged_attention_slot(page_table, page_tokens, position);
        if (slot < 0) { free(slots); return -1; }
        slots[position] = slot;
        visible++;
    }
    /* Every key masked out leaves the softmax denominator at zero.  Refusing is
     * the only honest answer: there is no distribution over an empty set, and
     * dividing anyway would publish NaNs into the retained cache. */
    if (visible == 0) { free(slots); return -1; }
    scores = (float*)malloc((size_t)kv_length * sizeof(float));
    if (!scores) { free(slots); return -1; }

    for (int head = 0; head < heads; head++) {
        const float* query = q + head * head_dim;
        float maximum = -1e38f;
        float sum = 0.0f;
        for (int position = 0; position < kv_length; position++) {
            const float* key;
            float score = 0.0f;
            if (slots[position] < 0) { scores[position] = 0.0f; continue; }
            key = key_pool + slots[position] * d_model + head * head_dim;
            for (int channel = 0; channel < head_dim; channel++) {
                score += query[channel] * key[channel];
            }
            score *= scale;
            scores[position] = score;
            if (score > maximum) maximum = score;
        }
        for (int position = 0; position < kv_length; position++) {
            if (slots[position] < 0) continue;
            scores[position] = expf(scores[position] - maximum);
            sum += scores[position];
        }
        for (int channel = 0; channel < head_dim; channel++) {
            float accumulator = 0.0f;
            for (int position = 0; position < kv_length; position++) {
                if (slots[position] < 0) continue;
                accumulator += scores[position] *
                    value_pool[slots[position] * d_model + head * head_dim + channel];
            }
            out[head * head_dim + channel] = accumulator / sum;
        }
    }
    free(scores);
    free(slots);
    return 0;
}
