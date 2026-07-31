#ifndef VOLVOXAI_ATTENTION_F32_ISA_H
#define VOLVOXAI_ATTENTION_F32_ISA_H

/*
 * Shared ISA setup and helpers for the F32 attention kernels.
 *
 * sdpa.c and cross_sdpa.c compute the same thing over different storage — one
 * reads a fused QKV tensor, the other three separate ones — and both had the
 * same defect: the key axis was walked several times per (query, head) with the
 * Q.K dot recomputed on each pass.  They now share one online-softmax core (see
 * attention_f32_core.h) and the helpers below.
 *
 * Both files are part of the kernels.c amalgamation, which is also compiled for
 * freestanding wasm32, so every x86 path is guarded and selected at runtime
 * through vx_kernel_platform() rather than by a build flag.
 */

#include "kernel_platform.h"

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_SDPA_X86_AVX2 1
#define VX_SDPA_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define VX_SDPA_X86_AVX2 0
#define VX_SDPA_TARGET_AVX2
#endif

/* Bounds the register-resident accumulator, matching VX_QSDPA_MAX_HEAD_DIM.
 * Wider heads take the portable body rather than growing the stack frame. */
enum { VX_SDPA_MAX_HEAD_DIM = 64 };

/* A causal row cannot attend past its own index, so bound the key loop instead
 * of testing every key and discarding most of them.  An incremental row path
 * already arrives pre-bounded with causal=0, so this only helps a full
 * forward — which is exactly the uncached case. */
static int vx_attention_key_limit(int seq_len_kv, int query, int causal) {
    if (!causal) return seq_len_kv;
    return query + 1 < seq_len_kv ? query + 1 : seq_len_kv;
}

static int vx_attention_key_allowed(const int32_t* mask, int mask_mode,
                                    int query, int key, int seq_len_kv,
                                    int causal) {
    if (causal && key > query) return 0;
    if (!mask || mask_mode == 0) return 1;
    if (mask_mode == 1) return mask[key] != 0;                      /* [K]   */
    if (mask_mode == 2) return mask[query * seq_len_kv + key] != 0; /* [Q,K] */
    return 0;
}

/* Both bodies build on the macros, bounds and predicates above.  The tiled one
 * additionally uses vx_accurate_exp_avx2 from fast_math.c, which the
 * amalgamation includes earlier. */
#include "attention_f32_core.h"
#include "attention_f32_tiled.h"

#endif
