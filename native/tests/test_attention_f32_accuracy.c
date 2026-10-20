/* Independent two-pass F64 oracle for the production private F32 bodies. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fast_exp.h"
#include "attention_f32_isa.h"

static uint32_t state = 710913u;
static float random_value(void) {
    state = state * 1664525u + 1013904223u;
    return ((float)(state >> 8) / 16777216.0f - 0.5f) * 0.3f;
}

static int allowed(const int32_t* mask, int mode, int query, int key, int keys, int causal) {
    if (causal && key > query) return 0;
    if (!mode) return 1;
    return mask[mode == 1 ? key : query * keys + key] != 0;
}

static void reference(const float* q, const float* k, const float* v, double* output,
        int queries, int keys, int heads, int kv_heads, int width, const int32_t* mask, int mode, int causal) {
    double* scores = malloc((size_t)keys * sizeof(*scores));
    if (!scores) abort();
    for (int row = 0; row < queries; row++) for (int head = 0; head < heads; head++) {
        double maximum = -INFINITY, sum = 0.;
        const int kv_head = head / (heads / kv_heads);
        for (int key = 0; key < keys; key++) {
            double score = 0.;
            for (int channel = 0; channel < width; channel++)
                score += (double)q[(row * heads + head) * width + channel] *
                    k[(key * kv_heads + kv_head) * width + channel];
            scores[key] = allowed(mask, mode, row, key, keys, causal) ? score : -INFINITY;
            if (scores[key] > maximum) maximum = scores[key];
        }
        if (maximum != -INFINITY) for (int key = 0; key < keys; key++) {
            scores[key] = exp(scores[key] - maximum);
            sum += scores[key];
        }
        for (int channel = 0; channel < width; channel++) {
            double value = 0.;
            if (sum > 0.) for (int key = 0; key < keys; key++)
                value += scores[key] * v[(key * kv_heads + kv_head) * width + channel];
            output[(row * heads + head) * width + channel] = sum > 0. ? value / sum : 0.;
        }
    }
    free(scores);
}

static void portable(const float* q, const float* k, const float* v, float* output,
        int queries, int keys, int heads, int kv_heads, int width, const int32_t* mask, int mode, int causal) {
    if (heads == kv_heads)
        vx_attention_f32_portable(q, k, v, heads * width, output, heads * width,
            queries, keys, heads, width, 1.f, mask, mode, causal);
    else vx_attention_f32_grouped_portable(q, k, v, heads * width, kv_heads * width,
        output, heads * width, queries, keys, heads, kv_heads, width, 1.f, mask, mode, causal);
}

static int verify(const char* label, const float* actual, const double* expected, size_t count) {
    double maximum = 0.;
    for (size_t i = 0; i < count; i++) {
        const double error = fabs(actual[i] - expected[i]);
        if (error > maximum) maximum = error;
        if (!isfinite(actual[i]) || error > 2e-6 + 1e-6 * fabs(expected[i])) {
            fprintf(stderr, "%s index=%zu actual=%.9g expected=%.12g delta=%.9g\n", label, i, actual[i], expected[i], error);
            return 0;
        }
    }
    printf("%s max_abs=%.9g PASS\n", label, maximum);
    return 1;
}

static int run(int queries, int keys, int heads, int kv_heads, int width, int mode, int causal) {
    const size_t q_size = (size_t)queries * heads * width;
    const size_t kv_size = (size_t)keys * kv_heads * width;
    float* q = malloc(q_size * sizeof(*q));
    float* k = malloc(kv_size * sizeof(*k));
    float* v = malloc(kv_size * sizeof(*v));
    float* actual = malloc(q_size * sizeof(*actual));
    float* single = malloc((size_t)heads * width * sizeof(*single));
    double* expected = malloc(q_size * sizeof(*expected));
    int32_t* mask = calloc((size_t)queries * keys, sizeof(*mask));
    if (!q || !k || !v || !actual || !single || !expected || !mask) abort();
    for (size_t i = 0; i < q_size; i++) q[i] = random_value();
    for (int key = 0; key < keys; key++) for (int h = 0; h < kv_heads; h++) for (int d = 0; d < width; d++) {
        const size_t i = ((size_t)key * kv_heads + h) * width + d;
        k[i] = random_value();
        v[i] = 100.f + .01337f * (float)(d % 11) + .001f * (float)(key % 7 - 3);
    }
    for (int row = 0; row < queries; row++) for (int key = 0; key < keys; key++)
        mask[row * keys + key] = row != 1 && key % 7 != 0;
    int success = 1;
    for (int permutation = 0; permutation < 2; permutation++) {
        reference(q, k, v, expected, queries, keys, heads, kv_heads, width, mask, mode, causal);
        portable(q, k, v, actual, queries, keys, heads, kv_heads, width, mask, mode, causal);
        success &= verify("portable", actual, expected, q_size);
        for (int row = 0; row < queries; row++) {
            const int bounded = causal && row + 1 < keys ? row + 1 : keys;
            portable(q + (size_t)row * heads * width, k, v, single, 1, bounded, heads, kv_heads, width,
                mask + (mode == 2 ? (size_t)row * keys : 0), mode, 0);
            if (memcmp(single, actual + (size_t)row * heads * width, (size_t)heads * width * sizeof(*single))) {
                fprintf(stderr, "portable row/full mismatch at row %d\n", row); success = 0;
            }
        }
#if VX_SDPA_X86_AVX2
        if (heads == kv_heads && width <= VX_SDPA_MAX_HEAD_DIM &&
                __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            vx_attention_tiled_avx2(q, k, v, heads * width, actual, heads * width,
                queries, keys, heads, width, 1.f, mask, mode, causal);
            success &= verify("avx2", actual, expected, q_size);
            for (int row = 0; row < queries; row++) {
                const int bounded = causal && row + 1 < keys ? row + 1 : keys;
                vx_attention_tiled_avx2(q + (size_t)row * heads * width, k, v, heads * width, single, heads * width,
                    1, bounded, heads, width, 1.f, mask + (mode == 2 ? (size_t)row * keys : 0), mode, 0);
                if (memcmp(single, actual + (size_t)row * heads * width, (size_t)heads * width * sizeof(*single))) {
                    fprintf(stderr, "avx2 row/full mismatch at row %d\n", row); success = 0;
                }
            }
        }
#endif
        /* Reverse the key/value/mask axis together, preserving the attention
         * definition while changing reduction order. Causal cases recompute
         * their independent reference after reversal. */
        for (int left = 0, right = keys - 1; left < right; left++, right--) {
            for (int channel = 0; channel < kv_heads * width; channel++) {
                float tmp = k[left * kv_heads * width + channel];
                k[left * kv_heads * width + channel] = k[right * kv_heads * width + channel];
                k[right * kv_heads * width + channel] = tmp;
                tmp = v[left * kv_heads * width + channel];
                v[left * kv_heads * width + channel] = v[right * kv_heads * width + channel];
                v[right * kv_heads * width + channel] = tmp;
            }
            for (int row = 0; row < queries; row++) {
                int32_t tmp = mask[row * keys + left];
                mask[row * keys + left] = mask[row * keys + right]; mask[row * keys + right] = tmp;
            }
        }
    }
    free(q); free(k); free(v); free(actual); free(single); free(expected); free(mask);
    return success;
}

int main(void) {
    int ok = run(19, 1537, 2, 2, 64, 0, 0);
    ok &= run(19, 513, 2, 2, 40, 1, 0);
    ok &= run(97, 513, 2, 2, 37, 2, 1);
    ok &= run(19, 1537, 4, 2, 80, 2, 0);
    return ok ? 0 : 1;
}
