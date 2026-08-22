/*
 * The two F32 attention bodies, on the same inputs, under every mask layout.
 *
 * `cross_sdpa_f32` and `sdpa_f32` pick between a portable online-softmax core
 * and an AVX2 tiled kernel at run time. The portable one is also what WASM
 * runs, so the pair is not "fast path and fallback" -- it is two shipped
 * implementations of one contract, and only one of them is exercised on any
 * given host.
 *
 * The masking is where they diverge structurally rather than numerically. The
 * portable body tests one key at a time; the tiled body builds a per-query lane
 * mask and folds it into the score block with `_mm256_and_ps`, which means the
 * value it builds has to be an all-ones/all-zero *bit pattern* and not a float
 * that merely looks like a flag. A `-1.0f` there is `0xBF800000`, which ANDs to
 * neither, and the result is a score neither kept nor masked -- wrong by a
 * factor, silently, and only for the query-dependent layout.
 *
 * That layout had no direct coverage because the row decode paths refused it
 * and the whole-sequence path was never compared across tiers. Both now reach
 * it, so this compares them directly.
 *
 * Not bit-exact, and deliberately so: the tiled body fuses its multiply-adds
 * and groups the key reduction by tile, so the two agree to a tolerance the way
 * `test_gemm_f32_isa_parity` describes. The bound is far tighter than any of
 * the failures above would produce -- a corrupted keep mask moves a result by
 * whole units, not by rounding.
 */
/* The portable body's softmax calls it; the amalgamation supplies it to the
 * kernels, so a test compiling these headers alone has to include it too. */
#include "../src/kernels/fast_exp.h"
#include "../src/kernels/attention_f32_isa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Rounding that accumulates over the key axis, not a factor. Measured envelope
 * with headroom. */
#define VX_ATTENTION_PARITY_BOUND 2.0e-5f

#define SEQ_Q 19
#define SEQ_KV 23
#define HEADS 3
#define HEAD_DIM 8
#define WIDTH (HEADS * HEAD_DIM)

static uint32_t g_rand = 0x9e3779b9u;
static float nextf(void) {
    g_rand = g_rand * 1103515245u + 12345u;
    return (float)((int32_t)((g_rand >> 8) % 2001u) - 1000) / 1000.0f;
}

static float q_in[SEQ_Q * WIDTH];
static float k_in[SEQ_KV * WIDTH];
static float v_in[SEQ_KV * WIDTH];
static float portable_out[SEQ_Q * WIDTH];
static float tiled_out[SEQ_Q * WIDTH];
static int32_t key_mask[SEQ_KV];
static int32_t query_key_mask[SEQ_Q * SEQ_KV];

static int compare(const char* label) {
    float worst = 0.0f;
    int worst_index = -1;
    for (int index = 0; index < SEQ_Q * WIDTH; index++) {
        const float reference = portable_out[index];
        const float scale = fabsf(reference) > 1.0f ? fabsf(reference) : 1.0f;
        const float error = fabsf(tiled_out[index] - reference) / scale;
        if (error > worst) {
            worst = error;
            worst_index = index;
        }
    }
    if (worst <= VX_ATTENTION_PARITY_BOUND) {
        printf("ok %-22s worst relative difference %.3e\n", label, (double)worst);
        return 0;
    }
    fprintf(stderr, "FAIL %s: worst relative difference %.3e at [%d] "
                    "(portable %.6f, avx2 %.6f)\n",
            label, (double)worst, worst_index,
            (double)portable_out[worst_index], (double)tiled_out[worst_index]);
    return 1;
}

/*
 * `mask_mode` is the kernel's own encoding, not `ATTN_MASK_*`: the caller has
 * already narrowed the mask to one batch, so 1 means key-indexed and 2 means
 * (query, key)-indexed.
 */
static int run_case(const char* label, const int32_t* mask, int mask_mode,
                    int causal) {
    memset(portable_out, 0, sizeof(portable_out));
    memset(tiled_out, 0, sizeof(tiled_out));
    vx_attention_f32_portable(q_in, k_in, v_in, WIDTH, portable_out, WIDTH,
                              SEQ_Q, SEQ_KV, HEADS, HEAD_DIM, 0.35f,
                              mask, mask_mode, causal);
    vx_attention_tiled_avx2(q_in, k_in, v_in, WIDTH, tiled_out, WIDTH,
                            SEQ_Q, SEQ_KV, HEADS, HEAD_DIM, 0.35f,
                            mask, mask_mode, causal);
    return compare(label);
}

int main(void) {
#if !VX_SDPA_X86_AVX2
    printf("SKIP no AVX2 body in this build\n");
    return 77;
#else
    int failures = 0;
    if (!vx_kernel_platform()->has_avx2) {
        printf("SKIP host has no AVX2\n");
        return 77;
    }

    for (int index = 0; index < SEQ_Q * WIDTH; index++) q_in[index] = nextf();
    for (int index = 0; index < SEQ_KV * WIDTH; index++) {
        k_in[index] = nextf();
        v_in[index] = nextf();
    }
    /*
     * Both masks drop keys rather than merely thinning them, and the
     * query-dependent one drops a different set per query -- a mask that
     * happened to keep every key would make the layouts indistinguishable, and
     * one uniform across queries would not tell the two indexings apart.
     */
    for (int key = 0; key < SEQ_KV; key++) key_mask[key] = (key % 3) != 0;
    for (int query = 0; query < SEQ_Q; query++)
        for (int key = 0; key < SEQ_KV; key++)
            query_key_mask[query * SEQ_KV + key] = ((query * 5 + key * 3) % 4) != 0;

    failures += run_case("no mask", NULL, 0, 0);
    failures += run_case("no mask, causal", NULL, 0, 1);
    failures += run_case("key mask", key_mask, 1, 0);
    failures += run_case("key mask, causal", key_mask, 1, 1);
    failures += run_case("query-key mask", query_key_mask, 2, 0);
    failures += run_case("query-key mask, causal", query_key_mask, 2, 1);

    if (failures) {
        fprintf(stderr, "FAIL %d attention ISA parity case(s)\n", failures);
        return 1;
    }
    printf("PASS attention F32 ISA parity\n");
    return 0;
#endif
}
